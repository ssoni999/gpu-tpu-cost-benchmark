#!/usr/bin/env python3
"""Build comparison.json from normalized GPU and TPU benchmark results."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from calculate_cost import compute_platform_costs
from config import load_config


def _pct_delta(base: float | None, other: float | None) -> float | None:
    if base is None or other is None or base == 0:
        return None
    return round((other - base) / base * 100, 2)


def _ratio(base: float | None, other: float | None) -> float | None:
    if base is None or other is None or base == 0:
        return None
    return round(other / base, 4)


def build_comparison(
    gpu: dict[str, Any],
    tpu: dict[str, Any],
    gpu_costs: dict[str, Any],
    tpu_costs: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    gpu_cpm = gpu_costs.get("cost_per_million_output_tokens_usd")
    tpu_cpm = tpu_costs.get("cost_per_million_output_tokens_usd")
    gpu_tps = gpu_costs.get("output_tokens_per_second")
    tpu_tps = tpu_costs.get("output_tokens_per_second")
    gpu_ppd = gpu_costs.get("performance_per_dollar_tok_s")
    tpu_ppd = tpu_costs.get("performance_per_dollar_tok_s")

    cheaper = None
    if gpu_cpm is not None and tpu_cpm is not None:
        cheaper = "tpu" if tpu_cpm < gpu_cpm else "gpu" if gpu_cpm < tpu_cpm else "tie"

    faster = None
    if gpu_tps is not None and tpu_tps is not None:
        faster = "tpu" if tpu_tps > gpu_tps else "gpu" if gpu_tps > tpu_tps else "tie"

    recommended = None
    if cheaper and faster:
        if cheaper == faster:
            recommended = cheaper
        elif gpu_ppd is not None and tpu_ppd is not None:
            recommended = "tpu" if tpu_ppd > gpu_ppd else "gpu"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": config["model"],
        "benchmark_contract": config["benchmark"],
        "gpu": {
            "normalized": gpu,
            "costs": gpu_costs,
        },
        "tpu": {
            "normalized": tpu,
            "costs": tpu_costs,
        },
        "headline": {
            "cost_per_million_output_tokens_usd": {
                "gpu": gpu_cpm,
                "tpu": tpu_cpm,
                "tpu_vs_gpu_pct": _pct_delta(gpu_cpm, tpu_cpm),
            },
            "output_tokens_per_second": {
                "gpu": gpu_tps,
                "tpu": tpu_tps,
                "tpu_vs_gpu_ratio": _ratio(gpu_tps, tpu_tps),
            },
            "p95_ttft_ms": {
                "gpu": gpu_costs.get("p95_ttft_ms"),
                "tpu": tpu_costs.get("p95_ttft_ms"),
            },
            "performance_per_dollar_tok_s": {
                "gpu": gpu_ppd,
                "tpu": tpu_ppd,
                "tpu_vs_gpu_ratio": _ratio(gpu_ppd, tpu_ppd),
            },
            "projected_annual_cost_usd": {
                "gpu": gpu_costs.get("projection", {}).get("projected_annual_cost_usd"),
                "tpu": tpu_costs.get("projection", {}).get("projected_annual_cost_usd"),
            },
        },
        "recommendation": {
            "lower_cost_per_million_output_tokens": cheaper,
            "higher_output_tokens_per_second": faster,
            "recommended_platform_for_workload": recommended,
        },
        "assumptions": [
            "Benchmark parameters identical on both platforms (see benchmark_contract).",
            "Both runs used vLLM bench serve with the same num_prompts, input/output lengths, and seed.",
            "Hourly costs taken from environment.json at run time — verify before quoting.",
            "Projected annual cost uses default volume assumptions in calculate_cost.py unless overridden.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare normalized GPU and TPU results.")
    parser.add_argument("--gpu", default="results/normalized/gpu.json")
    parser.add_argument("--tpu", default="results/normalized/tpu.json")
    parser.add_argument("--output", default="comparison.json")
    parser.add_argument("--rpd", type=float, default=500_000)
    parser.add_argument("--config", help="Path to benchmark_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    gpu = json.loads(Path(args.gpu).read_text(encoding="utf-8"))
    tpu = json.loads(Path(args.tpu).read_text(encoding="utf-8"))

    gpu_costs = compute_platform_costs(gpu, requests_per_day=args.rpd)
    tpu_costs = compute_platform_costs(tpu, requests_per_day=args.rpd)
    comparison = build_comparison(gpu, tpu, gpu_costs, tpu_costs, config)

    out = Path(args.output)
    out.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")

    rec = comparison["recommendation"]["recommended_platform_for_workload"]
    if rec:
        print(f"Recommended platform for this workload: {rec.upper()}")


if __name__ == "__main__":
    main()
