#!/usr/bin/env python3
"""Replay a JSONL trace against an OpenAI-compatible vLLM endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import aiohttp
import numpy as np
import openai

from config import (
    default_workload,
    load_config,
    platform_metadata,
    replay_result_path,
    replay_live_path,
    workload_profile,
    workload_trace_path,
    resolve_workload_trace_path,
)
from cost_metrics import compute_replay_cost_metrics
from gcs_results import try_upload_replay
from dashboard_client import LiveDashboard

# ---------------------------------------------------------------------------
# Hardware / infrastructure
# ---------------------------------------------------------------------------

PEAK_TFLOPS_DEFAULTS: dict[str, float] = {
    "TPU v6e": 918.0,
    "GPU (H100)": 989.0,
    "GPU (T4)": 65.0,
    "Generic": 100.0,
}

PLATFORM_INFRA: dict[str, tuple[str, float]] = {
    "tpu": ("TPU v6e", PEAK_TFLOPS_DEFAULTS["TPU v6e"]),
    "gpu": ("GPU (H100)", PEAK_TFLOPS_DEFAULTS["GPU (H100)"]),
}


def detect_infrastructure() -> tuple[str, float]:
    """Best-effort detection on the machine running replay (often Cloud Shell → Generic)."""
    if os.path.exists("/dev/accel0") or "TPU_NAME" in os.environ:
        return "TPU v6e", PEAK_TFLOPS_DEFAULTS["TPU v6e"]
    if os.path.exists("/dev/nvidia0") or "CUDA_VISIBLE_DEVICES" in os.environ:
        return "GPU (H100)", PEAK_TFLOPS_DEFAULTS["GPU (H100)"]
    return "Generic", PEAK_TFLOPS_DEFAULTS["Generic"]


def resolve_infrastructure(platform: str, peak_tflops: Optional[float]) -> tuple[str, float]:
    if platform in PLATFORM_INFRA:
        infra, peak = PLATFORM_INFRA[platform]
    else:
        infra, peak = detect_infrastructure()
    if peak_tflops is not None:
        peak = peak_tflops
    return infra, peak


def guess_params_b(model: str) -> float:
    lower = model.lower()
    if "0.5b" in lower or "0.5-b" in lower:
        return 0.49
    if "4b" in lower or "3-4b" in lower:
        return 4.0
    if "8b" in lower:
        return 8.0
    if "32b" in lower:
        return 32.0
    return 1.0


# ---------------------------------------------------------------------------
# Prometheus /metrics scraper
# ---------------------------------------------------------------------------


def metrics_base_url(target: str) -> str:
    base = target.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base.rstrip("/")


async def scrape_cumulative_metrics(
    session: aiohttp.ClientSession,
    target: str,
    retries: int = 3,
) -> dict[str, Any]:
    metrics_url = f"{metrics_base_url(target)}/metrics"
    for attempt in range(retries):
        try:
            async with session.get(metrics_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    continue
                content = await resp.text()
                kv_usage = 0.0
                queued_reqs = 0.0
                running_reqs = 0.0
                queries = 0.0
                hits = 0.0
                for line in content.splitlines():
                    if line.startswith("#") or not line.strip():
                        continue
                    if "vllm:kv_cache_usage_perc" in line or "vllm:gpu_cache_usage_perc" in line:
                        val = float(line.split()[-1])
                        kv_usage = val * 100.0 if val <= 1.0 else val
                    elif "vllm:num_requests_waiting" in line:
                        queued_reqs = float(line.split()[-1])
                    elif "vllm:num_requests_running" in line:
                        running_reqs = float(line.split()[-1])
                    elif "vllm:prefix_cache_queries_total" in line:
                        queries = float(line.split()[-1])
                    elif "vllm:prefix_cache_hits_total" in line:
                        hits = float(line.split()[-1])
                hit_rate = (hits / queries * 100.0) if queries > 0 else 0.0
                return {
                    "status": "success",
                    "kv_usage_pct": kv_usage,
                    "queued_requests": queued_reqs,
                    "running_requests": running_reqs,
                    "prefix_hit_rate_pct": hit_rate,
                    "prefix_queries": queries,
                    "prefix_hits": hits,
                }
        except Exception as exc:  # noqa: BLE001
            if attempt < retries - 1:
                await asyncio.sleep(0.5)
            else:
                return {
                    "status": f"unreachable ({type(exc).__name__})",
                    "kv_usage_pct": 0.0,
                    "queued_requests": 0.0,
                    "running_requests": 0.0,
                    "prefix_hit_rate_pct": 0.0,
                    "prefix_queries": 0.0,
                    "prefix_hits": 0.0,
                }
    return {"status": "unreachable", "kv_usage_pct": 0.0, "queued_requests": 0.0,
            "running_requests": 0.0, "prefix_hit_rate_pct": 0.0,
            "prefix_queries": 0.0, "prefix_hits": 0.0}


async def monitor_metrics(
    session: aiohttp.ClientSession,
    target: str,
    metrics_log: list[dict[str, Any]],
    stop_event: asyncio.Event,
    interval: float = 0.2,
) -> None:
    while not stop_event.is_set():
        data = await scrape_cumulative_metrics(session, target)
        if data["status"] == "success":
            metrics_log.append(data)
        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Trace + request types
# ---------------------------------------------------------------------------


@dataclass
class TraceRecord:
    prompt: str
    max_tokens: int
    offset: float


@dataclass
class RequestResult:
    index: int
    offset: float
    max_tokens: int
    status: str
    latency_ms: float
    ttft_ms: float
    prompt_tokens: int
    completion_tokens: int
    avg_itl_ms: float = 0.0
    per_request_tps: float = 0.0
    stream_token_chunks: int = 0
    error: Optional[str] = None


def load_trace(path: Path, tier: str | None = None, cfg: dict | None = None) -> list[TraceRecord]:
    from generate_trace import _enforce_max_prompt_tokens

    max_input: int | None = None
    if tier:
        profile = workload_profile(cfg or load_config(), tier)
        raw = profile.get("max_input_tokens")
        if raw is not None:
            max_input = int(raw)

    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            prompt = data["prompt"]
            if max_input is not None:
                prompt = _enforce_max_prompt_tokens(prompt, max_input)
            records.append(
                TraceRecord(
                    prompt=prompt,
                    max_tokens=int(data["max_tokens"]),
                    offset=float(data.get("offset", 0.0)),
                )
            )
    return records


def split_prompt(prompt: str) -> tuple[Optional[str], str]:
    """Split trace prompt into system + user when generate_trace used '---'."""
    if "\n\n---\n\n" in prompt:
        system, user = prompt.split("\n\n---\n\n", 1)
        return system.strip(), user.strip()
    return None, prompt


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def check_endpoint(base_url: str) -> None:
    models_url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(models_url, timeout=10) as resp:
            resp.read(200)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Cannot reach {models_url}: {exc}", file=sys.stderr)
        print(
            "\nFix: deploy vLLM, then port-forward or use the service URL:\n"
            "  kubectl port-forward svc/vllm-service 8000:8000\n"
            "  curl http://127.0.0.1:8000/v1/models\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def resolve_api_mode(model: str, api: str) -> str:
    if api != "auto":
        return api
    model_lower = model.lower()
    if "opt-125" in model_lower or "/opt-" in model_lower:
        return "completions"
    return "chat"


def chat_messages(prompt: str) -> list[dict[str, str]]:
    system, user = split_prompt(prompt)
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages


async def send_request(
    client: openai.AsyncOpenAI,
    model: str,
    record: TraceRecord,
    index: int,
    api_mode: str,
    total_requests: int = 0,
    infra_type: str = "",
    log_progress: bool = True,
) -> RequestResult:
    start = time.perf_counter()
    ttft_ms = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    token_times: list[float] = []
    last_token_at = start
    stream_chunks = 0
    try:
        stream_options = {"include_usage": True}
        if api_mode == "completions":
            stream = await client.completions.create(
                model=model,
                prompt=record.prompt,
                max_tokens=record.max_tokens,
                temperature=0,
                stream=True,
                stream_options=stream_options,
            )
        else:
            stream = await client.chat.completions.create(
                model=model,
                messages=chat_messages(record.prompt),
                max_tokens=record.max_tokens,
                temperature=0,
                stream=True,
                stream_options=stream_options,
            )
        first_token_at: Optional[float] = None
        async for chunk in stream:
            if hasattr(chunk, "usage") and chunk.usage is not None:
                prompt_tokens = chunk.usage.prompt_tokens or prompt_tokens
                completion_tokens = chunk.usage.completion_tokens or completion_tokens
            if not chunk.choices:
                continue
            if api_mode == "completions":
                content = chunk.choices[0].text
            else:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None) or getattr(
                    delta, "reasoning_content", None
                )
            if content:
                now = time.perf_counter()
                stream_chunks += 1
                if first_token_at is None:
                    first_token_at = now
                else:
                    token_times.append(now - last_token_at)
                last_token_at = now
        end = time.perf_counter()
        if first_token_at is None:
            first_token_at = end
        ttft_ms = (first_token_at - start) * 1000
        latency_ms = (end - start) * 1000
        avg_itl_ms = (statistics.mean(token_times) * 1000) if token_times else 0.0
        per_request_tps = (
            (completion_tokens or stream_chunks) / (end - start) if end > start else 0.0
        )

        if log_progress and total_requests > 0:
            log_interval = max(1, total_requests // 5)
            if (index + 1) % log_interval == 0 or index == 0 or (index + 1) == total_requests:
                prefix = f"[{infra_type} | Trace {index + 1}/{total_requests}]" if infra_type else ""
                print(
                    f"{prefix} TTFT: {ttft_ms:.1f}ms | "
                    f"Duration: {(end - start):.3f}s | "
                    f"Out tokens: {completion_tokens or stream_chunks}"
                )

        return RequestResult(
            index=index,
            offset=record.offset,
            max_tokens=record.max_tokens,
            status="ok",
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            avg_itl_ms=avg_itl_ms,
            per_request_tps=per_request_tps,
            stream_token_chunks=stream_chunks,
        )
    except Exception as exc:  # noqa: BLE001
        end = time.perf_counter()
        return RequestResult(
            index=index,
            offset=record.offset,
            max_tokens=record.max_tokens,
            status="error",
            latency_ms=(end - start) * 1000,
            ttft_ms=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            error=str(exc),
        )


async def warmup(
    client: openai.AsyncOpenAI,
    model: str,
    count: int,
    api_mode: str,
) -> None:
    for i in range(count):
        await send_request(
            client,
            model,
            TraceRecord(prompt=f"WARMUP {i}: reply ok", max_tokens=8, offset=0.0),
            index=-1,
            api_mode=api_mode,
            log_progress=False,
        )


@dataclass
class ReplayOutcome:
    results: list[RequestResult]
    wall_time: float
    metrics_log: list[dict[str, Any]] = field(default_factory=list)
    final_metrics: dict[str, Any] = field(default_factory=dict)


def compute_rolling(
    results: list[RequestResult],
    elapsed_s: float,
    peak_tflops: float,
    params_b: float,
) -> dict[str, Any]:
    ok = [r for r in results if r.status == "ok" and r.index >= 0]
    ttfts = [r.ttft_ms for r in ok]
    lats = [r.latency_ms for r in ok]
    itls = [r.avg_itl_ms for r in ok if r.avg_itl_ms > 0]
    out_tok = sum(r.completion_tokens for r in ok)
    gen_tps = out_tok / elapsed_s if elapsed_s > 0 else 0.0
    rps = len(ok) / elapsed_s if elapsed_s > 0 else 0.0
    achieved_tflops = (gen_tps * 2.0 * params_b * 1e9) / 1e12 if gen_tps > 0 else 0.0
    mfu = (achieved_tflops / peak_tflops * 100.0) if peak_tflops > 0 else 0.0
    return {
        "output_tokens": out_tok,
        "output_tps": round(gen_tps, 2),
        "rps": round(rps, 3),
        "ttft_p95_ms": round(percentile(ttfts, 95), 1),
        "latency_p95_ms": round(percentile(lats, 95), 1),
        "avg_itl_ms": round(statistics.mean(itls), 2) if itls else 0.0,
        "achieved_tflops": round(achieved_tflops, 4),
        "mfu_percent": round(mfu, 2),
    }


async def emit_server_samples(
    live: LiveDashboard,
    session: aiohttp.ClientSession,
    metrics_log: list[dict[str, Any]],
    stop_event: asyncio.Event,
    interval: float = 1.0,
) -> None:
    while not stop_event.is_set():
        if metrics_log:
            sample = metrics_log[-1]
            await live.publish(session, "server_sample", sample)
        await asyncio.sleep(interval)


async def replay_trace(
    target: str,
    model: str,
    trace: list[TraceRecord],
    speed: float,
    concurrency: int,
    warmup_count: int,
    api_mode: str,
    infra_type: str,
    scrape_metrics: bool,
    metrics_interval: float,
    live: Optional[LiveDashboard] = None,
    peak_tflops: float = 100.0,
    params_b: float = 1.0,
    platform: str = "auto",
) -> ReplayOutcome:
    base_url = target.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    check_endpoint(base_url)
    client = openai.AsyncOpenAI(api_key="EMPTY", base_url=base_url)

    metrics_log: list[dict[str, Any]] = []
    stop_event = asyncio.Event()
    final_metrics: dict[str, Any] = {}

    async with aiohttp.ClientSession() as http_session:
        monitor_task: Optional[asyncio.Task[None]] = None
        server_emit_task: Optional[asyncio.Task[None]] = None
        if scrape_metrics:
            monitor_task = asyncio.create_task(
                monitor_metrics(http_session, target, metrics_log, stop_event, metrics_interval)
            )

        if live is not None:
            await live.publish(
                http_session,
                "run_start",
                {
                    "model": model,
                    "infrastructure": infra_type,
                    "platform": platform,
                    "target": target,
                    "progress": {"completed": 0, "total": len(trace), "ok": 0, "err": 0},
                },
            )
            if scrape_metrics:
                server_emit_task = asyncio.create_task(
                    emit_server_samples(live, http_session, metrics_log, stop_event)
                )

        if warmup_count > 0:
            await warmup(client, model, warmup_count, api_mode)

        if trace:
            probe = await send_request(
                client,
                model,
                trace[0],
                index=-2,
                api_mode=api_mode,
                log_progress=False,
            )
            if probe.status != "ok":
                hint = (
                    "\nPreflight request failed before replaying the trace. "
                    "Fix this first — all requests would fail the same way.\n"
                    "  • Context too long → set --max-model-len=4096 in tiny-model.yaml and redeploy\n"
                    "  • Wrong model name → MODEL must match the server's --model flag\n"
                    "  • Bad port-forward → kubectl port-forward svc/tiny-model-service 8000:8000\n"
                )
                raise RuntimeError(f"{probe.error}{hint}")

        semaphore = asyncio.Semaphore(concurrency)
        results: list[RequestResult] = []
        replay_start = time.perf_counter()
        total = len(trace)
        results_lock = asyncio.Lock()

        async def run_one(index: int, record: TraceRecord) -> None:
            target_time = replay_start + (record.offset / speed)
            delay = target_time - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            async with semaphore:
                result = await send_request(
                    client,
                    model,
                    record,
                    index,
                    api_mode,
                    total_requests=total,
                    infra_type=infra_type,
                )
                async with results_lock:
                    results.append(result)
                    if live is not None:
                        ok_n = sum(1 for r in results if r.status == "ok")
                        err_n = sum(1 for r in results if r.status != "ok")
                        elapsed = time.perf_counter() - replay_start
                        await live.publish(
                            http_session,
                            "request_done",
                            {
                                "completed": len(results),
                                "total": total,
                                "ok": ok_n,
                                "err": err_n,
                                "request": asdict(result),
                                "rolling": compute_rolling(
                                    results, elapsed, peak_tflops, params_b
                                ),
                            },
                        )

        await asyncio.gather(*(run_one(i, record) for i, record in enumerate(trace)))
        wall_time = time.perf_counter() - replay_start

        stop_event.set()
        if server_emit_task is not None:
            await server_emit_task
        if monitor_task is not None:
            await monitor_task
        final_metrics = await scrape_cumulative_metrics(http_session, target)

    results.sort(key=lambda r: r.index)
    return ReplayOutcome(
        results=results,
        wall_time=wall_time,
        metrics_log=metrics_log,
        final_metrics=final_metrics,
    )


def build_error_summary(errors: list[RequestResult]) -> dict[str, Any]:
    msgs = [r.error for r in errors if r.error]
    counts = Counter(msgs)
    return {
        "total_failures": len(errors),
        "unique_errors": len(counts),
        "top_errors": [
            {"error": msg, "count": count} for msg, count in counts.most_common(5)
        ],
        "samples": [asdict(r) for r in errors[:3]],
    }


def summarize_server_metrics(
    metrics_log: list[dict[str, Any]],
    final_metrics: dict[str, Any],
) -> dict[str, Any]:
    if not final_metrics:
        final_metrics = {
            "status": "disabled",
            "kv_usage_pct": 0.0,
            "queued_requests": 0.0,
            "running_requests": 0.0,
            "prefix_hit_rate_pct": 0.0,
            "prefix_queries": 0.0,
            "prefix_hits": 0.0,
        }
    peak_kv = max([m["kv_usage_pct"] for m in metrics_log], default=final_metrics["kv_usage_pct"])
    peak_queued = max(
        [m["queued_requests"] for m in metrics_log], default=final_metrics["queued_requests"]
    )
    peak_running = max(
        [m["running_requests"] for m in metrics_log], default=final_metrics["running_requests"]
    )
    return {
        "endpoint_status": final_metrics.get("status", "unknown"),
        "peak_kv_cache_usage_pct": round(peak_kv, 2),
        "prefix_cache_hit_rate_pct": round(final_metrics.get("prefix_hit_rate_pct", 0.0), 2),
        "prefix_cache_queries": int(final_metrics.get("prefix_queries", 0)),
        "prefix_cache_hits": int(final_metrics.get("prefix_hits", 0)),
        "peak_queued_requests": round(peak_queued, 1),
        "peak_running_requests": round(peak_running, 1),
        "samples_collected": len(metrics_log),
        "final_snapshot": final_metrics,
    }


def summarize(
    results: list[RequestResult],
    wall_time: float,
    trace_path: str,
    target: str,
    model: str,
    infra_type: str,
    peak_tflops: float,
    params_b: float,
    metrics_log: list[dict[str, Any]],
    final_metrics: dict[str, Any],
) -> dict:
    ok = [r for r in results if r.status == "ok"]
    errors = [r for r in results if r.status != "ok"]
    latencies = [r.latency_ms for r in ok]
    ttfts = [r.ttft_ms for r in ok]
    itls = [r.avg_itl_ms for r in ok if r.avg_itl_ms > 0]
    input_tokens = sum(r.prompt_tokens for r in ok)
    output_tokens = sum(r.completion_tokens for r in ok)
    total_tokens = input_tokens + output_tokens
    throughput = total_tokens / wall_time if wall_time > 0 else 0.0
    gen_tps = output_tokens / wall_time if wall_time > 0 else 0.0
    req_per_sec = len(ok) / wall_time if wall_time > 0 else 0.0
    avg_duration_s = (statistics.mean(latencies) / 1000.0) if latencies else 0.0
    avg_itl_ms = statistics.mean(itls) if itls else 0.0

    achieved_tflops = (gen_tps * 2.0 * params_b * 1e9) / 1e12 if gen_tps > 0 else 0.0
    mfu_percent = (achieved_tflops / peak_tflops * 100.0) if peak_tflops > 0 else 0.0

    server = summarize_server_metrics(metrics_log, final_metrics)
    payload: dict[str, Any] = {
        "source": "trace_replay",
        "target": target,
        "model": model,
        "trace": trace_path,
        "infrastructure": infra_type,
        "wall_time_s": round(wall_time, 2),
        "successful_requests": len(ok),
        "failed_requests": len(errors),
        "benchmark_duration_seconds": round(wall_time, 2),
        "total_input_tokens": input_tokens,
        "total_generated_tokens": output_tokens,
        "requests_per_second": round(req_per_sec, 4),
        "output_tokens_per_second": round(gen_tps, 2),
        "throughput_tok_s": round(throughput, 1),
        "generation_tok_s": round(gen_tps, 1),
        "latency_ms": {
            "p50": round(percentile(latencies, 50), 1),
            "p95": round(percentile(latencies, 95), 1),
            "p99": round(percentile(latencies, 99), 1),
            "avg": round(avg_duration_s * 1000, 1),
        },
        "ttft_ms": {
            "p50": round(percentile(ttfts, 50), 1),
            "p95": round(percentile(ttfts, 95), 1),
            "p99": round(percentile(ttfts, 99), 1),
        },
        "itl_ms": {
            "avg": round(avg_itl_ms, 2),
            "p50": round(percentile(itls, 50), 2),
            "p95": round(percentile(itls, 95), 2),
        },
        "median_ttft_ms": round(percentile(ttfts, 50), 1),
        "p95_ttft_ms": round(percentile(ttfts, 95), 1),
        "p99_ttft_ms": round(percentile(ttfts, 99), 1),
        "hardware_utilization": {
            "model_params_b": params_b,
            "peak_tflops": peak_tflops,
            "achieved_tflops": round(achieved_tflops, 4),
            "mfu_percent": round(mfu_percent, 2),
        },
        "server_metrics": server,
    }
    if errors:
        payload["error_summary"] = build_error_summary(errors)
    return payload


def print_summary(summary: dict) -> None:
    hw = summary.get("hardware_utilization", {})
    srv = summary.get("server_metrics", {})
    itl = summary.get("itl_ms", {})

    print()
    print("=" * 60)
    print(" WORKLOAD REPLAY SUMMARY")
    print("=" * 60)
    print(f" Infrastructure           : {summary.get('infrastructure', 'unknown')}")
    print(f" Model                    : {summary['model']}")
    print(f" Wall time                : {summary['wall_time_s']:.2f} s")
    print(
        f" Requests                 : {summary['successful_requests']} ok / "
        f"{summary['failed_requests']} err"
    )
    print(f" Throughput (RPS)         : {summary['requests_per_second']:.2f} req/s")
    print(f" Generated tokens         : {summary['total_generated_tokens']}")
    print(f" Throughput (TPS)         : {summary['output_tokens_per_second']:.2f} tok/s")
    print("-" * 60)
    print(" HARDWARE UTILIZATION:")
    print(f"   Model params (B)       : {hw.get('model_params_b', 0):.2f}")
    print(f"   Achieved TFLOPS        : {hw.get('achieved_tflops', 0):.4f}")
    print(
        f"   MFU                    : {hw.get('mfu_percent', 0):.2f}% "
        f"(vs {hw.get('peak_tflops', 0):.0f} peak TFLOPS)"
    )
    print("-" * 60)
    print(" CLIENT LATENCY:")
    print(f"   TTFT p50               : {summary['ttft_ms']['p50']:.1f} ms")
    print(f"   TTFT p95               : {summary['ttft_ms']['p95']:.1f} ms")
    print(f"   Latency p95            : {summary['latency_ms']['p95']:.1f} ms")
    print(f"   Avg ITL                : {itl.get('avg', 0):.2f} ms")
    print(f"   Avg request duration   : {summary['latency_ms'].get('avg', 0):.1f} ms")
    print("-" * 60)
    print(" SERVER METRICS (/metrics):")
    print(f"   Endpoint status        : {srv.get('endpoint_status', 'n/a')}")
    print(f"   Peak KV cache usage    : {srv.get('peak_kv_cache_usage_pct', 0):.2f}%")
    print(
        f"   Prefix cache hit rate  : {srv.get('prefix_cache_hit_rate_pct', 0):.1f}% "
        f"({srv.get('prefix_cache_hits', 0)} / {srv.get('prefix_cache_queries', 0)})"
    )
    print(f"   Peak queued requests   : {srv.get('peak_queued_requests', 0):.1f}")
    print(f"   Peak running requests  : {srv.get('peak_running_requests', 0):.1f}")
    if summary.get("failed_requests", 0) > 0:
        print("-" * 60)
        print(" ERRORS (replay failed — metrics above are zero):")
        err = summary.get("error_summary") or {}
        for item in err.get("top_errors", [])[:3]:
            print(f"   [{item.get('count', '?')}x] {item.get('error', 'unknown')}")
        print()
        print(" Common fixes:")
        print("   • context length → bump --max-model-len in tiny-model.yaml (try 4096) and redeploy")
        print("   • wrong model    → pass MODEL='Qwen/Qwen2.5-0.5B-Instruct' matching the server")
        print("   • no port-forward→ kubectl port-forward svc/tiny-model-service 8000:8000")
    print("=" * 60)


def main() -> int:
    cfg = load_config()
    bench = cfg["benchmark"]
    default_model = cfg["model"]

    parser = argparse.ArgumentParser(description="Replay workload/prompts.jsonl against vLLM.")
    parser.add_argument("--target", required=True, help="Base URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--trace", default="workload/prompts.jsonl")
    parser.add_argument("--output", default="results/tpu/run_01_replay.json")
    parser.add_argument("--speed", type=float, default=10.0, help="Time compression (higher=faster)")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=int(bench.get("warmup_requests", 10)))
    parser.add_argument("--api", choices=["auto", "chat", "completions"], default="auto")
    parser.add_argument(
        "--tier",
        default=None,
        help="Workload tier (small/medium/high); sets trace/output paths when omitted",
    )
    parser.add_argument(
        "--platform",
        choices=["auto", "tpu", "gpu"],
        default="auto",
        help="Hardware label for MFU (use tpu/gpu when replay runs from Cloud Shell)",
    )
    parser.add_argument(
        "--params-b",
        type=float,
        default=None,
        help="Model size in billions for MFU estimate (default: inferred from model name)",
    )
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=None,
        help="Override peak TFLOPS for MFU denominator",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Disable background /metrics scraping",
    )
    parser.add_argument(
        "--metrics-interval",
        type=float,
        default=0.2,
        help="Seconds between /metrics polls",
    )
    parser.add_argument(
        "--dashboard-url",
        default=None,
        help="Live dashboard base URL, e.g. http://127.0.0.1:8765",
    )
    parser.add_argument(
        "--live-json",
        default=None,
        help="Write live snapshot JSON during replay (default: results/<platform>/live.json)",
    )
    args = parser.parse_args()

    tier = args.tier or default_workload(cfg)

    trace_path = resolve_workload_trace_path(tier, args.trace)
    if not trace_path.exists():
        print(f"Trace not found: {trace_path}. Run: make trace TIER={tier}", file=sys.stderr)
        return 1

    trace = load_trace(trace_path, tier=tier, cfg=cfg)
    if not trace:
        print(f"No records in {trace_path}", file=sys.stderr)
        return 1

    api_mode = resolve_api_mode(args.model, args.api)
    platform = args.platform if args.platform != "auto" else "auto"
    if platform == "auto":
        infra_type, peak_tflops = detect_infrastructure()
    else:
        infra_type, peak_tflops = resolve_infrastructure(platform, args.peak_tflops)
    if args.peak_tflops is not None and platform == "auto":
        peak_tflops = args.peak_tflops
    params_b = args.params_b if args.params_b is not None else guess_params_b(args.model)

    live_json = Path(args.live_json) if args.live_json else None
    if live_json is None and args.platform in ("tpu", "gpu"):
        live_json = replay_live_path(args.platform, tier)

    out_path = Path(args.output)
    if args.platform in ("tpu", "gpu") and str(args.output) == f"results/{args.platform}/run_01_replay.json":
        out_path = replay_result_path(args.platform, tier)

    dashboard_url = args.dashboard_url

    live = None
    if dashboard_url or live_json:
        live = LiveDashboard(dashboard_url=dashboard_url, live_json_path=live_json)

    print("=" * 60)
    print(f" Replaying {len(trace)} requests → {args.target}")
    print(f" Speed: {args.speed}x | Concurrency: {args.concurrency} | API: {api_mode}")
    print(f" Infrastructure: {infra_type} (peak {peak_tflops} TFLOPS) | Model: {params_b}B params")
    if dashboard_url:
        print(f" Live dashboard: {dashboard_url}/")
    if live_json:
        print(f" Live JSON: {live_json}")
        print(f" Results UI: run 'make ui' then open http://127.0.0.1:8787/")
    print(f" Workload tier: {tier}")
    print("=" * 60)

    try:
        outcome = asyncio.run(
            replay_trace(
                args.target,
                args.model,
                trace,
                args.speed,
                args.concurrency,
                args.warmup,
                api_mode,
                infra_type,
                scrape_metrics=not args.no_metrics,
                metrics_interval=args.metrics_interval,
                live=live,
                peak_tflops=peak_tflops,
                params_b=params_b,
                platform=args.platform,
            )
        )
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    summary = summarize(
        outcome.results,
        outcome.wall_time,
        str(trace_path),
        args.target,
        args.model,
        infra_type,
        peak_tflops,
        params_b,
        outcome.metrics_log,
        outcome.final_metrics,
    )
    summary["platform"] = args.platform if args.platform in ("tpu", "gpu") else platform
    cfg = load_config()
    tier_profile = workload_profile(cfg, tier)
    if args.platform in ("tpu", "gpu"):
        summary["workload_tier"] = tier
        summary["platform_config"] = platform_metadata(cfg, args.platform)
        summary["benchmark_contract"] = {
            "model": cfg["model"],
            "trace": str(trace_path),
            "workload_tier": tier,
            "num_requests": len(trace),
            "input_tokens": tier_profile["input_tokens"],
            "output_tokens": tier_profile["output_tokens"],
            "seed": cfg["benchmark"].get("seed"),
        }
        summary["cost_metrics"] = compute_replay_cost_metrics(summary, args.platform)
    print_summary(summary)

    if live is not None:
        async def _finish_live() -> None:
            async with aiohttp.ClientSession() as session:
                await live.publish(session, "run_complete", summary)

        asyncio.run(_finish_live())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")

    if args.platform in ("tpu", "gpu"):
        try_upload_replay(args.platform, summary, tier=tier)

    return 0 if summary["failed_requests"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
