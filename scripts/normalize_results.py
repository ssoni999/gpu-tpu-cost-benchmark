#!/usr/bin/env python3
"""Parse vllm bench serve text output into the normalized result schema."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from config import load_config

# Keys match vllm bench serve "Serving Benchmark Result" block labels.
_FIELD_MAP = {
    "successful_requests": r"Successful requests:\s+([0-9]+)",
    "failed_requests": r"Failed requests:\s+([0-9]+)",
    "benchmark_duration_seconds": r"Benchmark duration \(s\):\s+([0-9.eE+-]+)",
    "total_input_tokens": r"Total input tokens:\s+([0-9]+)",
    "total_generated_tokens": r"Total generated tokens:\s+([0-9]+)",
    "requests_per_second": r"Request throughput \(req/s\):\s+([0-9.eE+-]+)",
    "output_tokens_per_second": r"Output token throughput \(tok/s\):\s+([0-9.eE+-]+)",
    "total_tokens_per_second": r"Total Token throughput \(tok/s\):\s+([0-9.eE+-]+)",
    "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([0-9.eE+-]+)",
    "median_ttft_ms": r"Median TTFT \(ms\):\s+([0-9.eE+-]+)",
    "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([0-9.eE+-]+)",
}

# vllm versions may omit p50/p95; derive p95 from median when missing.
_P50_TTFT = re.compile(r"P50 TTFT \(ms\):\s+([0-9.eE+-]+)", re.I)
_P95_TTFT = re.compile(r"P95 TTFT \(ms\):\s+([0-9.eE+-]+)", re.I)


def _parse_float(match: re.Match[str] | None) -> float | None:
    if not match:
        return None
    return float(match.group(1))


def _coerce_metric(key: str, raw: str) -> int | float:
    if key.endswith("_requests") or key in {"total_input_tokens", "total_generated_tokens"}:
        return int(raw)
    return float(raw)


def parse_vllm_bench_output(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, pattern in _FIELD_MAP.items():
        m = re.search(pattern, text)
        if m:
            out[key] = _coerce_metric(key, m.group(1))

    p50 = _parse_float(_P50_TTFT.search(text))
    p95 = _parse_float(_P95_TTFT.search(text))
    if p50 is not None:
        out["p50_ttft_ms"] = p50
    if p95 is not None:
        out["p95_ttft_ms"] = p95
    elif "median_ttft_ms" in out:
        out["p95_ttft_ms"] = out["median_ttft_ms"]

    if "p99_ttft_ms" in out and "p95_ttft_ms" not in out:
        out["p95_ttft_ms"] = out["p99_ttft_ms"]

    return out


def parse_replay_json(data: dict[str, Any]) -> dict[str, Any]:
    """Convert replay.py summary JSON into normalize-ready metrics."""
    return {
        "successful_requests": data.get("successful_requests", 0),
        "failed_requests": data.get("failed_requests", 0),
        "benchmark_duration_seconds": data.get("benchmark_duration_seconds")
        or data.get("wall_time_s"),
        "total_input_tokens": data.get("total_input_tokens", 0),
        "total_generated_tokens": data.get("total_generated_tokens", 0),
        "requests_per_second": data.get("requests_per_second", 0.0),
        "output_tokens_per_second": data.get("output_tokens_per_second")
        or data.get("generation_tok_s"),
        "p50_ttft_ms": data.get("p50_ttft_ms") or data.get("median_ttft_ms"),
        "p95_ttft_ms": data.get("p95_ttft_ms") or data.get("ttft_ms", {}).get("p95"),
        "p99_ttft_ms": data.get("p99_ttft_ms") or data.get("ttft_ms", {}).get("p99"),
        "source": "trace_replay",
    }


def build_normalized(
    platform: str,
    raw_metrics: dict[str, Any],
    environment: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or load_config()
    platform_cfg = cfg["gpu"] if platform == "gpu" else cfg["tpu"]

    successful = int(raw_metrics.get("successful_requests", 0))
    failed = int(raw_metrics.get("failed_requests", 0))
    total_requests = successful + failed

    hourly = environment.get("hourly_cost_usd")
    if hourly is None:
        accel = environment.get("accelerator_hourly_usd", 0) or 0
        vm = environment.get("vm_hourly_usd", 0) or 0
        hourly = float(accel) + float(vm)

    return {
        "platform": platform,
        "model": cfg["model"],
        "accelerator": platform_cfg["accelerator"],
        "accelerator_count": platform_cfg["accelerator_count"],
        "framework": platform_cfg.get("framework", "vllm"),
        "benchmark_contract": cfg["benchmark"],
        "successful_requests": successful,
        "failed_requests": failed,
        "total_requests": total_requests,
        "success_rate": round(successful / total_requests, 4) if total_requests else 0.0,
        "duration_seconds": raw_metrics.get("benchmark_duration_seconds", 0.0),
        "input_tokens": raw_metrics.get("total_input_tokens", 0),
        "output_tokens": raw_metrics.get("total_generated_tokens", 0),
        "requests_per_second": raw_metrics.get("requests_per_second", 0.0),
        "output_tokens_per_second": raw_metrics.get("output_tokens_per_second", 0.0),
        "p50_ttft_ms": raw_metrics.get("p50_ttft_ms") or raw_metrics.get("median_ttft_ms"),
        "p95_ttft_ms": raw_metrics.get("p95_ttft_ms"),
        "p99_ttft_ms": raw_metrics.get("p99_ttft_ms"),
        "hourly_cost_usd": hourly,
        "environment": environment,
        "measurement_source": raw_metrics.get("source", "vllm_bench"),
        "raw_metrics": raw_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize bench or replay output.")
    parser.add_argument("--platform", required=True, choices=["gpu", "tpu"])
    parser.add_argument("--raw", help="Path to vllm bench serve .txt output")
    parser.add_argument("--replay-json", help="Path to replay.py summary JSON")
    parser.add_argument("--environment", required=True, help="Path to environment.json")
    parser.add_argument(
        "--output",
        help="Normalized JSON path (default: results/normalized/{platform}.json)",
    )
    parser.add_argument("--config", help="Path to benchmark_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    environment = json.loads(Path(args.environment).read_text(encoding="utf-8"))

    if args.replay_json:
        replay_data = json.loads(Path(args.replay_json).read_text(encoding="utf-8"))
        raw_metrics = parse_replay_json(replay_data)
    elif args.raw:
        text = Path(args.raw).read_text(encoding="utf-8")
        raw_metrics = parse_vllm_bench_output(text)
        raw_metrics["source"] = "vllm_bench"
    else:
        raise SystemExit("Provide --raw (bench text) or --replay-json (trace replay output)")

    if not raw_metrics.get("successful_requests"):
        raise SystemExit("No successful requests found in input metrics")

    normalized = build_normalized(args.platform, raw_metrics, environment, config)
    out_path = Path(args.output or f"results/normalized/{args.platform}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
