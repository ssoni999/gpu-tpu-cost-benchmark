"""Project replay metrics across workload tiers when tier-specific results are absent."""

from __future__ import annotations

import copy
from typing import Any

from config import load_config, workload_profile, workload_tier_ids
from cost_metrics import compute_replay_cost_metrics


def _scale_latency_block(
    block: dict[str, Any] | None,
    factor: float,
    *,
    decimals: int = 1,
) -> dict[str, Any] | None:
    if not block:
        return block
    out: dict[str, Any] = {}
    for key, val in block.items():
        if val is None:
            out[key] = val
        else:
            out[key] = round(float(val) * factor, decimals if key != "avg" else decimals)
    return out


def project_replay(
    base: dict[str, Any],
    source_tier: str,
    target_tier: str,
    platform: str,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive tier metrics from a measured replay using workload scaling factors."""
    cfg = cfg or load_config()
    if source_tier == target_tier:
        projected = copy.deepcopy(base)
        projected["workload_tier"] = target_tier
        projected["projected"] = False
        return projected

    src = workload_profile(cfg, source_tier)
    tgt = workload_profile(cfg, target_tier)

    src_prompts = max(1, int(src.get("num_prompts", 1)))
    tgt_prompts = max(1, int(tgt.get("num_prompts", 1)))
    src_in = max(1, int(src.get("input_tokens", 1)))
    tgt_in = max(1, int(tgt.get("input_tokens", 1)))
    src_out = max(1, int(src.get("output_tokens", 1)))
    tgt_out = max(1, int(tgt.get("output_tokens", 1)))

    out_ratio = (tgt_prompts * tgt_out) / (src_prompts * src_out)
    in_ratio = (tgt_prompts * tgt_in) / (src_prompts * src_in)
    req_ratio = tgt_prompts / src_prompts

    gpu_mult = float(tgt.get("gpu_throughput_factor", 1.0)) / float(
        src.get("gpu_throughput_factor", 1.0)
    )
    tpu_mult = float(tgt.get("tpu_throughput_factor", 1.0)) / float(
        src.get("tpu_throughput_factor", 1.0)
    )
    plat_mult = tpu_mult if platform == "tpu" else gpu_mult

    base_wall = float(base.get("wall_time_s") or 1.0)
    overhead_s = 1.5
    compute_s = max(0.4, base_wall - overhead_s)

    input_wall_factor = 0.3 + 0.7 * (in_ratio ** 0.85)
    output_wall_factor = out_ratio ** 0.92
    work_factor = input_wall_factor * 0.35 + output_wall_factor * 0.65

    new_compute = compute_s * work_factor / max(plat_mult, 0.01)
    new_wall = overhead_s + new_compute

    tgt_gen_tokens = int(tgt_prompts * tgt_out)
    tgt_in_tokens = int(tgt_prompts * tgt_in)
    ok_ratio = float(base.get("successful_requests") or src_prompts) / src_prompts

    new_tps = tgt_gen_tokens * ok_ratio / new_wall if new_wall > 0 else 0.0
    new_rps = tgt_prompts * ok_ratio / new_wall if new_wall > 0 else 0.0

    result = copy.deepcopy(base)
    result["wall_time_s"] = round(new_wall, 2)
    result["output_tokens_per_second"] = round(new_tps, 2)
    result["throughput_tok_s"] = round(new_tps, 1)
    result["generation_tok_s"] = round(new_tps, 1)
    result["requests_per_second"] = round(new_rps, 4)
    result["successful_requests"] = int(round(tgt_prompts * ok_ratio))
    result["total_generated_tokens"] = int(round(tgt_gen_tokens * ok_ratio))
    result["total_input_tokens"] = int(round(tgt_in_tokens * ok_ratio))

    ttft_scale = in_ratio ** 0.72
    itl_scale = (tgt_out / src_out) ** 0.35
    lat_scale = work_factor ** 0.45

    if result.get("ttft_ms"):
        result["ttft_ms"] = _scale_latency_block(result["ttft_ms"], ttft_scale)
    if result.get("itl_ms"):
        result["itl_ms"] = _scale_latency_block(result["itl_ms"], itl_scale, decimals=2)
    if result.get("latency_ms"):
        result["latency_ms"] = _scale_latency_block(result["latency_ms"], lat_scale)

    hw = result.get("hardware_utilization") or {}
    if hw:
        mfu_base = float(hw.get("mfu_percent") or 0.0)
        mfu = min(99.0, mfu_base * (plat_mult ** 0.45) * (work_factor ** 0.08))
        peak = float(hw.get("peak_tflops") or 100.0)
        hw = dict(hw)
        hw["mfu_percent"] = round(mfu, 2)
        hw["achieved_tflops"] = round(peak * mfu / 100.0, 4)
        result["hardware_utilization"] = hw

    contract = dict(result.get("benchmark_contract") or {})
    contract.update({
        "workload_tier": target_tier,
        "num_requests": tgt_prompts,
        "input_tokens": tgt_in,
        "output_tokens": tgt_out,
    })
    result["benchmark_contract"] = contract
    result["workload_tier"] = target_tier
    result["projected"] = True
    result["projected_from_tier"] = source_tier
    result["cost_metrics"] = compute_replay_cost_metrics(result, platform, cfg)
    return result


def resolve_tier_replay(
    replays_by_tier: dict[str, dict[str, Any]],
    target_tier: str,
    platform: str,
    cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str, bool]:
    """Pick measured replay or project from nearest available tier."""
    cfg = cfg or load_config()
    if target_tier in replays_by_tier:
        replay = copy.deepcopy(replays_by_tier[target_tier])
        replay["workload_tier"] = target_tier
        replay["projected"] = False
        return replay, target_tier, False

    tier_order = workload_tier_ids(cfg)
    if not replays_by_tier:
        return None, "", False

    def tier_distance(t: str) -> int:
        if t not in tier_order or target_tier not in tier_order:
            return 999
        return abs(tier_order.index(t) - tier_order.index(target_tier))

    source_tier = min(replays_by_tier.keys(), key=tier_distance)
    projected = project_replay(replays_by_tier[source_tier], source_tier, target_tier, platform, cfg)
    return projected, source_tier, True
