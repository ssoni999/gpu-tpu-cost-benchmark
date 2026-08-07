"""Cost and energy metrics from replay summaries (used by UI and replay)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from calculate_cost import (
    cost_per_million_output_tokens,
    hourly_stack_cost,
    performance_per_dollar,
)
from config import ROOT, load_config

DEFAULT_PRICING = {
    "gpu_accelerator_hourly_usd": 0.35,
    "gpu_vm_hourly_usd": 0.10,
    "tpu_accelerator_hourly_usd": 1.20,
    "tpu_vm_hourly_usd": 0.0,
}


def load_environment(platform: str, root: Path | None = None) -> dict[str, Any]:
    root = root or ROOT
    path = root / "results" / platform / "run_01_environment.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def resolve_hourly_cost(platform: str, config: dict[str, Any] | None = None, env: dict | None = None) -> float:
    cfg = config or load_config()
    env = env if env is not None else load_environment(platform)
    pricing = cfg.get("pricing", {})

    if env.get("hourly_cost_usd") is not None:
        return float(env["hourly_cost_usd"])

    accel_key = f"{platform}_accelerator_hourly_usd"
    vm_key = f"{platform}_vm_hourly_usd"
    accel = env.get("accelerator_hourly_usd")
    vm = env.get("vm_hourly_usd")

    if accel is None:
        accel = pricing.get(accel_key) or DEFAULT_PRICING.get(accel_key, 0.0)
    if vm is None:
        vm = pricing.get(vm_key) or DEFAULT_PRICING.get(vm_key, 0.0)

    if accel is not None and vm is not None:
        return hourly_stack_cost(float(accel), 1, float(vm))

    block = cfg.get(platform, {})
    count = int(block.get("accelerator_count", 1))
    return hourly_stack_cost(float(accel or 0), count, float(vm or 0))


def resolve_power_watts(platform: str, config: dict[str, Any] | None = None) -> float:
    cfg = config or load_config()
    block = cfg.get(platform, {})
    power = block.get("power_watts")
    if power is not None:
        return float(power)
    pricing = cfg.get("pricing", {})
    key = f"{platform}_tdp_watts"
    return float(pricing.get(key, 100.0))


def compute_replay_cost_metrics(
    replay: dict[str, Any],
    platform: str,
    config: dict[str, Any] | None = None,
    env: dict | None = None,
) -> dict[str, Any]:
    """Derive run cost, $/token, $/s wall time, and tok/s per watt from a replay summary."""
    cfg = config or load_config()
    env = env if env is not None else load_environment(platform)

    wall_s = float(replay.get("wall_time_s") or replay.get("benchmark_duration_seconds") or 0)
    out_tokens = int(replay.get("total_generated_tokens") or 0)
    total_tokens = int(replay.get("total_input_tokens") or 0) + out_tokens
    out_tps = float(replay.get("output_tokens_per_second") or 0)

    hourly = resolve_hourly_cost(platform, cfg, env)
    watts = resolve_power_watts(platform, cfg)
    run_cost = hourly * (wall_s / 3600.0) if wall_s > 0 and hourly > 0 else 0.0

    cost_per_out_token = (run_cost / out_tokens) if out_tokens > 0 and run_cost > 0 else None
    cost_per_total_token = (run_cost / total_tokens) if total_tokens > 0 and run_cost > 0 else None
    cost_per_wall_second = (run_cost / wall_s) if wall_s > 0 and run_cost > 0 else None
    perf_per_watt = (out_tps / watts) if watts > 0 and out_tps > 0 else None
    perf_per_dollar = performance_per_dollar(out_tps, hourly)
    cost_per_million = cost_per_million_output_tokens(out_tokens, wall_s, hourly)

    return {
        "hourly_cost_usd": round(hourly, 4),
        "run_cost_usd": round(run_cost, 6),
        "cost_per_output_token_usd": _round(cost_per_out_token, 8),
        "cost_per_total_token_usd": _round(cost_per_total_token, 8),
        "cost_per_wall_second_usd": _round(cost_per_wall_second, 6),
        "cost_per_million_output_tokens_usd": _round(cost_per_million, 4),
        "performance_per_dollar_tok_s": _round(perf_per_dollar, 2),
        "performance_per_watt_tok_s": _round(perf_per_watt, 4),
        "power_watts": watts,
        "pricing_source": "environment" if env.get("hourly_cost_usd") or env.get("accelerator_hourly_usd") else "config",
    }


def _round(value: float | None, digits: int) -> float | None:
    if value is None:
        return None
    return round(value, digits)
