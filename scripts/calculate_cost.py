#!/usr/bin/env python3
"""Platform-neutral cost metrics from normalized benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def hourly_stack_cost(
    accelerator_hourly_usd: float,
    accelerator_count: int,
    vm_hourly_usd: float = 0.0,
) -> float:
    return accelerator_hourly_usd * accelerator_count + vm_hourly_usd


def cost_per_million_output_tokens(
    output_tokens: int,
    duration_seconds: float,
    hourly_cost_usd: float,
) -> float | None:
    if output_tokens <= 0 or duration_seconds <= 0:
        return None
    hours = duration_seconds / 3600.0
    run_cost = hourly_cost_usd * hours
    return (run_cost / output_tokens) * 1_000_000


def cost_per_thousand_requests(
    successful_requests: int,
    duration_seconds: float,
    hourly_cost_usd: float,
) -> float | None:
    if successful_requests <= 0 or duration_seconds <= 0:
        return None
    hours = duration_seconds / 3600.0
    run_cost = hourly_cost_usd * hours
    return (run_cost / successful_requests) * 1000


def performance_per_dollar(
    output_tokens_per_second: float,
    hourly_cost_usd: float,
) -> float | None:
    if hourly_cost_usd <= 0:
        return None
    return output_tokens_per_second / hourly_cost_usd


def projected_annual_cost(
    output_tokens_per_second: float,
    hourly_cost_usd: float,
    requests_per_day: float,
    avg_output_tokens_per_request: float,
) -> dict[str, float]:
    tokens_per_year = requests_per_day * 365 * avg_output_tokens_per_request
    needed_tps = tokens_per_year / (365 * 24 * 3600)
    measured_tps = max(output_tokens_per_second, 1e-9)
    util_factor = needed_tps / measured_tps
    annual = hourly_cost_usd * 24 * 365 * util_factor
    cost_1m = (annual / tokens_per_year) * 1_000_000 if tokens_per_year else 0.0
    return {
        "tokens_per_year": tokens_per_year,
        "needed_output_tps": round(needed_tps, 1),
        "utilization_factor": round(util_factor, 4),
        "projected_annual_cost_usd": round(annual, 0),
        "projected_cost_per_million_output_tokens_usd": round(cost_1m, 4),
    }


def compute_platform_costs(
    normalized: dict[str, Any],
    requests_per_day: float = 500_000,
    avg_output_tokens_per_request: float | None = None,
) -> dict[str, Any]:
    env = normalized.get("environment", {})
    hourly = normalized.get("hourly_cost_usd")
    if hourly is None:
        hourly = hourly_stack_cost(
            float(env.get("accelerator_hourly_usd", 0)),
            int(normalized.get("accelerator_count", 1)),
            float(env.get("vm_hourly_usd", 0)),
        )

    output_tps = float(normalized.get("output_tokens_per_second", 0))
    duration = float(normalized.get("duration_seconds", 0))
    output_tokens = int(normalized.get("output_tokens", 0))
    successful = int(normalized.get("successful_requests", 0))

    if avg_output_tokens_per_request is None:
        bench = normalized.get("benchmark_contract", {})
        avg_output_tokens_per_request = float(bench.get("output_tokens", 128))

    return {
        "platform": normalized["platform"],
        "hourly_cost_usd": round(hourly, 4),
        "cost_per_million_output_tokens_usd": _round_optional(
            cost_per_million_output_tokens(output_tokens, duration, hourly)
        ),
        "cost_per_thousand_requests_usd": _round_optional(
            cost_per_thousand_requests(successful, duration, hourly)
        ),
        "output_tokens_per_second": output_tps,
        "p95_ttft_ms": normalized.get("p95_ttft_ms"),
        "performance_per_dollar_tok_s": _round_optional(
            performance_per_dollar(output_tps, hourly)
        ),
        "projection": projected_annual_cost(
            output_tps,
            hourly,
            requests_per_day,
            avg_output_tokens_per_request,
        ),
    }


def _round_optional(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute cost metrics from normalized results.")
    parser.add_argument("--input", required=True, help="Normalized platform JSON")
    parser.add_argument("--output", help="Cost JSON output path")
    parser.add_argument("--rpd", type=float, default=500_000, help="Requests per day for projection")
    parser.add_argument("--tpr", type=float, help="Avg output tokens per request (default: from benchmark contract)")
    args = parser.parse_args()

    normalized = json.loads(Path(args.input).read_text(encoding="utf-8"))
    costs = compute_platform_costs(
        normalized,
        requests_per_day=args.rpd,
        avg_output_tokens_per_request=args.tpr,
    )

    print(json.dumps(costs, indent=2))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(costs, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
