#!/usr/bin/env python3
"""On-demand GPU provisioning for the workload advisor.

Scales the configured GKE GPU node pool from 0 -> 1 by applying the vLLM
Deployment/Service manifest (the cluster autoscaler picks up the pending pod
and provisions a node), waits for the server to become reachable, then replays
an uploaded workload trace against it with replay.py's own request engine.

The deployment is left running after each job so a subsequent run can skip
straight to replay if the pod is already warm (kubectl apply is idempotent,
and the pod/server wait loops return immediately once ready). Call
scale_down_gpu() explicitly to tear it down and let the node pool drop back
to 0 — nothing does this automatically.

Requires `gcloud` and `kubectl` on PATH and an already-authenticated gcloud
session (this is designed to run inside Cloud Shell).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from config import ROOT, load_config, platform_block
from cost_metrics import compute_replay_cost_metrics
from dashboard_client import LiveDashboard
from render_manifest import render_platform
from replay import TraceRecord, guess_params_b, replay_trace, resolve_api_mode, summarize
from workload_advisor import parse_workload_text

PROVISION_TIMEOUT_S = 1200  # 20 minutes, per the cold-start budget for a 0->1 GPU node.
POD_POLL_INTERVAL_S = 10
SERVER_POLL_INTERVAL_S = 3
FAILURE_WAIT_REASONS = {"ImagePullBackOff", "ErrImagePull", "CrashLoopBackOff", "InvalidImageName"}

TERMINAL_STATUSES = {"done", "error", "cancelled"}


@dataclass
class GpuJob:
    id: str
    status: str = "queued"
    message: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    log: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    cancel_requested: bool = False
    live: Optional[LiveDashboard] = None
    _pf_proc: Optional[asyncio.subprocess.Process] = None
    _manifest_path: Optional[Path] = None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "status": self.status,
            "message": self.message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "log": self.log[-30:],
            "progress": self.live.state.get("progress") if self.live else None,
            "result": self.result,
            "error": self.error,
        }


_JOBS: dict[str, GpuJob] = {}
_ACTIVE_JOB_ID: str | None = None


class GpuJobError(RuntimeError):
    pass


def _set(job: GpuJob, status: str, message: str) -> None:
    job.status = status
    job.message = message
    job.updated_at = time.time()
    job.log.append(f"[{time.strftime('%H:%M:%S')}] {status}: {message}")


def _gcp_env() -> dict[str, str]:
    """Read PROJECT_ID / ZONE / CLUSTER_NAME from the environment, falling back to configs/gpu.env."""
    keys = ("PROJECT_ID", "ZONE", "CLUSTER_NAME")
    values = {k: os.environ.get(k, "") for k in keys}
    if all(values.values()):
        return values

    env_path = ROOT / "configs" / "gpu.env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line.startswith("export "):
                continue
            body = line[len("export "):]
            if "=" not in body:
                continue
            key, _, raw_val = body.partition("=")
            key = key.strip()
            if key in keys and not values.get(key):
                values[key] = raw_val.strip().strip('"').strip("'")

    missing = [k for k in keys if not values.get(k)]
    if missing:
        raise GpuJobError(
            f"Missing {', '.join(missing)}. Set them in configs/gpu.env (see gpu.env.example) "
            "or export them before starting results_server.py."
        )
    return values


async def _run(cmd: list[str], *, timeout: float = 60.0) -> str:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        raise GpuJobError(f"Command timed out after {timeout}s: {' '.join(cmd)}") from None
    if proc.returncode != 0:
        raise GpuJobError(
            f"Command failed ({' '.join(cmd)}): {stderr.decode('utf-8', errors='replace').strip()[:800]}"
        )
    return stdout.decode("utf-8", errors="replace")


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _check_http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


async def _wait_for_pod_ready(job: GpuJob, app_label: str, deadline: float) -> None:
    while True:
        if job.cancel_requested:
            raise GpuJobError("Cancelled while waiting for the GPU node.")
        if time.time() > deadline:
            raise GpuJobError(
                "Timed out waiting for a GPU node to become available "
                f"(after {PROVISION_TIMEOUT_S}s). The cluster may be at capacity."
            )
        raw = await _run(["kubectl", "get", "pods", "-l", f"app={app_label}", "-o", "json"], timeout=30)
        items = json.loads(raw).get("items", [])
        if not items:
            _set(job, "waiting_for_node", "Waiting for GKE to schedule a GPU node (cold start: ~3-6 min)…")
        else:
            pod = items[0]
            phase = pod.get("status", {}).get("phase", "Unknown")
            statuses = pod.get("status", {}).get("containerStatuses", [])
            for cs in statuses:
                waiting = cs.get("state", {}).get("waiting", {})
                reason = waiting.get("reason")
                if reason in FAILURE_WAIT_REASONS:
                    raise GpuJobError(f"Pod container failed to start: {reason} — {waiting.get('message', '')}")
            if phase == "Running" and all(cs.get("ready") for cs in statuses):
                _set(job, "waiting_for_server", "GPU node is up — waiting for vLLM to finish loading the model…")
                return
            _set(job, "waiting_for_pod", f"GPU node found, pod phase: {phase}…")
        await asyncio.sleep(POD_POLL_INTERVAL_S)


async def _wait_for_server(job: GpuJob, local_port: int, deadline: float) -> None:
    url = f"http://127.0.0.1:{local_port}/v1/models"
    while True:
        if job.cancel_requested:
            raise GpuJobError("Cancelled while waiting for vLLM to become ready.")
        if time.time() > deadline:
            raise GpuJobError("Timed out waiting for vLLM to report ready.")
        if await asyncio.to_thread(_check_http_ok, url):
            return
        await asyncio.sleep(SERVER_POLL_INTERVAL_S)


async def _stop_port_forward(job: GpuJob) -> None:
    """Kill this job's local kubectl port-forward. Does NOT touch the cluster deployment —
    the GPU deployment is left running (warm) so the next job can skip straight to replay."""
    if job._pf_proc is not None:
        with contextlib.suppress(ProcessLookupError):
            job._pf_proc.kill()
        with contextlib.suppress(Exception):
            await job._pf_proc.wait()
        job._pf_proc = None


def _gpu_manifest_path(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    return ROOT / platform_block(cfg, "gpu")["k8s"]["output_file"]


async def scale_down_gpu() -> str:
    """Manually tear down the GPU deployment so the node pool can scale back to 0.

    Nothing calls this automatically anymore (see _run_job) — the deployment is left
    warm after each run so subsequent runs can skip provisioning. Call this explicitly
    when done to stop paying for the idle node.
    """
    manifest_path = _gpu_manifest_path()
    if not manifest_path.exists():
        return "No GPU manifest found on disk — nothing to scale down."
    await _run(["kubectl", "delete", "-f", str(manifest_path), "--ignore-not-found"], timeout=60)
    return "GPU deployment deleted — node pool will scale back to 0 after GKE's idle grace period."


async def _run_job(job: GpuJob, content: str, filename: str) -> None:
    global _ACTIVE_JOB_ID
    deadline = time.time() + PROVISION_TIMEOUT_S
    final_status = "done"
    final_message = "Complete."
    try:
        parsed = parse_workload_text(content, filename=filename)
        if parsed.errors:
            raise GpuJobError("; ".join(parsed.errors[:5]))

        cfg = load_config()
        gpu_block = platform_block(cfg, "gpu")
        k8s = gpu_block["k8s"]
        model = cfg["model"]

        env = _gcp_env()

        manifest_text = render_platform("gpu", cfg)
        manifest_path = ROOT / k8s["output_file"]
        manifest_path.write_text(manifest_text, encoding="utf-8")
        job._manifest_path = manifest_path

        _set(job, "provisioning", f"Connecting to cluster {env['CLUSTER_NAME']}…")
        await _run(
            [
                "gcloud", "container", "clusters", "get-credentials", env["CLUSTER_NAME"],
                "--zone", env["ZONE"], "--project", env["PROJECT_ID"],
            ],
            timeout=60,
        )

        _set(job, "provisioning", "Applying GPU deployment manifest…")
        await _run(["kubectl", "apply", "-f", str(manifest_path)], timeout=60)

        await _wait_for_pod_ready(job, k8s["app_label"], deadline)

        local_port = _free_local_port()
        job._pf_proc = await asyncio.create_subprocess_exec(
            "kubectl", "port-forward", f"svc/{k8s['service_name']}", f"{local_port}:{k8s.get('port', 8000)}",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await _wait_for_server(job, local_port, deadline)

        records = [
            TraceRecord(prompt=r.prompt, max_tokens=r.max_tokens, offset=r.offset)
            for r in parsed.records
        ]
        _set(job, "running", f"Replaying {len(records)} requests against the GPU…")
        job.live = LiveDashboard()
        outcome = await replay_trace(
            f"http://127.0.0.1:{local_port}",
            model,
            records,
            speed=10.0,
            concurrency=min(32, len(records)),
            warmup_count=min(3, max(0, len(records) - 1)),
            api_mode=resolve_api_mode(model, "auto"),
            infra_type="GPU (L4)",
            scrape_metrics=True,
            metrics_interval=0.5,
            live=job.live,
            peak_tflops=float(gpu_block.get("peak_tflops") or 242.0),
            params_b=guess_params_b(model),
            platform="gpu",
        )

        summary = summarize(
            outcome.results, outcome.wall_time, filename, f"http://127.0.0.1:{local_port}",
            model, "GPU (L4)", float(gpu_block.get("peak_tflops") or 242.0), guess_params_b(model),
            outcome.metrics_log, outcome.final_metrics,
        )
        summary["platform"] = "gpu"
        summary["cost_metrics"] = compute_replay_cost_metrics(summary, "gpu", cfg)

        out_dir = ROOT / "results" / "gpu" / "_advisor"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{job.id}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        job.result = summary
    except GpuJobError as exc:
        job.error = str(exc)
        final_status = "cancelled" if job.cancel_requested else "error"
        final_message = str(exc)
    except Exception as exc:  # noqa: BLE001
        job.error = f"{type(exc).__name__}: {exc}"
        final_status = "error"
        final_message = job.error
    finally:
        _set(job, "cleaning_up", "Closing port-forward (GPU deployment left running for reuse)…")
        await _stop_port_forward(job)
        _set(job, final_status, final_message)
        if _ACTIVE_JOB_ID == job.id:
            _ACTIVE_JOB_ID = None


def start_job(content: str, filename: str) -> GpuJob:
    global _ACTIVE_JOB_ID
    if _ACTIVE_JOB_ID is not None and _JOBS[_ACTIVE_JOB_ID].status not in TERMINAL_STATUSES:
        raise GpuJobError("A GPU job is already running. Wait for it to finish before starting another.")

    job = GpuJob(id=uuid.uuid4().hex[:12])
    _JOBS[job.id] = job
    _ACTIVE_JOB_ID = job.id
    asyncio.create_task(_run_job(job, content, filename))
    return job


def get_job(job_id: str) -> GpuJob | None:
    return _JOBS.get(job_id)


def cancel_job(job_id: str) -> GpuJob | None:
    job = _JOBS.get(job_id)
    if job is None:
        return None
    job.cancel_requested = True
    return job
