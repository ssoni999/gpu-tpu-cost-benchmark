#!/usr/bin/env python3
"""Analyze uploaded inference workloads and emit optimization + migration guidance."""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import (
    load_config,
    migration_diff,
    platform_metadata,
    workload_profile,
    workload_tier_ids,
    workloads_catalog,
)

MAX_LINES = 10_000
MAX_BYTES = 10 * 1024 * 1024
PROMPT_ALIASES = ("prompt", "input", "text", "content", "query", "instruction")
MAX_TOKEN_ALIASES = ("max_tokens", "max_output_tokens", "completion_tokens", "output_tokens")
OFFSET_ALIASES = ("offset", "timestamp", "arrival_time", "arrival_offset", "time_offset")
CATEGORY_ALIASES = ("category", "type", "task", "intent")
PREFIX_SPLIT = re.compile(r"\n\n---\n\n", re.MULTILINE)


@dataclass
class ParsedRecord:
    prompt: str
    max_tokens: int
    offset: float
    category: str | None = None
    line: int = 0


@dataclass
class ParseResult:
    records: list[ParsedRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Rough OpenAI-style heuristic: ~4 chars per token for English prose.
    return max(1, int(len(text) / 4))


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def parse_workload_text(content: str, filename: str = "upload.jsonl") -> ParseResult:
    result = ParseResult()
    if not content or not content.strip():
        result.errors.append("File is empty.")
        return result

    stripped = content.lstrip()
    if stripped.startswith("["):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            result.errors.append(f"Invalid JSON array: {exc}")
            return result
        if not isinstance(payload, list):
            result.errors.append("JSON root must be an array of request objects.")
            return result
        lines = [json.dumps(item) for item in payload]
    else:
        lines = content.splitlines()

    if len(lines) > MAX_LINES:
        result.errors.append(f"Too many lines ({len(lines)}). Limit is {MAX_LINES}.")
        return result

    for idx, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            result.errors.append(f"Line {idx}: invalid JSON — {exc}")
            continue
        if not isinstance(data, dict):
            result.errors.append(f"Line {idx}: expected a JSON object.")
            continue

        prompt = _first_present(data, PROMPT_ALIASES)
        if prompt is None and "messages" in data:
            messages = data["messages"]
            if isinstance(messages, list):
                parts = []
                for msg in messages:
                    if isinstance(msg, dict) and msg.get("content"):
                        role = msg.get("role", "user")
                        parts.append(f"[{role}] {msg['content']}")
                prompt = "\n".join(parts) if parts else None
        if prompt is None:
            result.errors.append(f"Line {idx}: missing prompt (expected one of {PROMPT_ALIASES}).")
            continue
        prompt = str(prompt)
        if not prompt.strip():
            result.errors.append(f"Line {idx}: prompt is empty.")
            continue

        max_tokens_raw = _first_present(data, MAX_TOKEN_ALIASES)
        if max_tokens_raw is None:
            result.warnings.append(f"Line {idx}: max_tokens missing — defaulting to 128.")
            max_tokens = 128
        else:
            try:
                max_tokens = int(max_tokens_raw)
            except (TypeError, ValueError):
                result.errors.append(f"Line {idx}: max_tokens must be an integer.")
                continue
            if max_tokens <= 0:
                result.errors.append(f"Line {idx}: max_tokens must be positive.")
                continue

        offset_raw = _first_present(data, OFFSET_ALIASES)
        offset = 0.0
        if offset_raw is not None:
            try:
                offset = float(offset_raw)
            except (TypeError, ValueError):
                result.warnings.append(f"Line {idx}: invalid offset — using 0.")
        category_raw = _first_present(data, CATEGORY_ALIASES)
        category = str(category_raw) if category_raw is not None else None

        result.records.append(
            ParsedRecord(
                prompt=prompt,
                max_tokens=max_tokens,
                offset=offset,
                category=category,
                line=idx,
            )
        )

    if not result.records and not result.errors:
        result.errors.append(f"No valid requests found in {filename}.")
    return result


def _shared_prefix_stats(records: list[ParsedRecord]) -> dict[str, Any]:
    prefixes: list[str] = []
    for rec in records:
        parts = PREFIX_SPLIT.split(rec.prompt, maxsplit=1)
        if len(parts) == 2:
            prefixes.append(parts[0])
    if not prefixes:
        return {"detected": False, "ratio": 0.0, "avg_prefix_tokens": 0}
    first = prefixes[0]
    identical = sum(1 for p in prefixes if p == first)
    ratio = identical / len(records)
    return {
        "detected": ratio >= 0.5,
        "ratio": round(ratio, 3),
        "avg_prefix_tokens": _estimate_tokens(first),
        "sample_prefix_chars": min(len(first), 200),
    }


def _classify_tier(
    num_requests: int,
    avg_input_tokens: float,
    avg_output_tokens: float,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    tier_ids = workload_tier_ids(cfg)
    best_id = tier_ids[0]
    best_score = float("inf")
    catalog = workloads_catalog(cfg)
    scores: list[dict[str, Any]] = []

    for entry in catalog:
        tier_id = entry["id"]
        profile = workload_profile(cfg, tier_id)
        exp_in = float(profile.get("input_tokens") or 1024)
        exp_out = float(profile.get("output_tokens") or 128)
        exp_n = float(profile.get("num_prompts") or 100)
        score = (
            abs(num_requests - exp_n) / max(exp_n, 1)
            + abs(avg_input_tokens - exp_in) / max(exp_in, 1)
            + abs(avg_output_tokens - exp_out) / max(exp_out, 1)
        )
        scores.append({"tier": tier_id, "label": entry["label"], "distance": round(score, 3)})
        if score < best_score:
            best_score = score
            best_id = tier_id

    matched = next(e for e in catalog if e["id"] == best_id)
    return {
        "closest_tier": best_id,
        "label": matched["label"],
        "description": matched.get("description", ""),
        "tier_scores": sorted(scores, key=lambda s: s["distance"]),
    }


def _platform_scores(
    num_requests: int,
    total_tokens: float,
    avg_input_tokens: float,
    avg_output_tokens: float,
    rps: float,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    pricing = cfg.get("pricing", {})
    gpu_hourly = float(pricing.get("gpu_accelerator_hourly_usd", 0.35)) + float(
        pricing.get("gpu_vm_hourly_usd", 0.10)
    )
    tpu_hourly = float(pricing.get("tpu_accelerator_hourly_usd", 1.20)) + float(
        pricing.get("tpu_vm_hourly_usd", 0.0)
    )

    gpu_meta = platform_metadata(cfg, "gpu")
    tpu_meta = platform_metadata(cfg, "tpu")
    gpu_tflops = float(gpu_meta.get("peak_tflops") or 65)
    tpu_tflops = float(tpu_meta.get("peak_tflops") or 197)

    # Heuristic: short/low-volume favors GPU hourly rate; sustained/high-token favors TPU FLOPs.
    volume_factor = min(1.0, num_requests / 100.0)
    context_factor = min(1.0, avg_input_tokens / 2048.0)
    sustained = min(1.0, rps * 60 / 50.0)  # >50 req/min → sustained

    gpu_fit = 0.55
    tpu_fit = 0.45
    if avg_input_tokens >= 1536 or num_requests >= 150:
        tpu_fit += 0.25
        gpu_fit -= 0.15
    if num_requests <= 40 and avg_input_tokens <= 512:
        gpu_fit += 0.25
        tpu_fit -= 0.15
    if rps >= 1.0:
        tpu_fit += 0.1 * sustained
    gpu_fit += 0.1 * (1 - context_factor) * (1 - volume_factor)
    tpu_fit += 0.15 * context_factor * volume_factor

    gpu_cost_index = gpu_hourly / max(gpu_tflops, 1)
    tpu_cost_index = tpu_hourly / max(tpu_tflops, 1)
    cost_winner = "gpu" if gpu_cost_index <= tpu_cost_index else "tpu"
    perf_winner = "tpu" if tpu_tflops > gpu_tflops else "gpu"

    recommended = "tpu" if tpu_fit >= gpu_fit else "gpu"
    return {
        "recommended_platform": recommended,
        "confidence": round(abs(tpu_fit - gpu_fit), 2),
        "gpu_fit_score": round(gpu_fit, 2),
        "tpu_fit_score": round(tpu_fit, 2),
        "cost_efficiency_leader": cost_winner,
        "peak_throughput_leader": perf_winner,
        "estimated_hourly_usd": {"gpu": gpu_hourly, "tpu": tpu_hourly},
        "total_tokens_estimate": int(total_tokens),
    }


def _build_recommendations(
    stats: dict[str, Any],
    prefix_stats: dict[str, Any],
    tier_info: dict[str, Any],
    platform_info: dict[str, Any],
    cfg: dict[str, Any],
    user_context: dict[str, Any],
) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    gpu_serving = cfg.get("gpu", {}).get("serving", {})
    tpu_serving = cfg.get("tpu", {}).get("serving", {})
    max_model_len = int(gpu_serving.get("max_model_len") or 4096)

    recs.append({
        "priority": "high",
        "category": "platform",
        "title": f"Primary platform: {platform_info['recommended_platform'].upper()}",
        "rationale": (
            f"Based on {stats['num_requests']} requests, ~{stats['avg_input_tokens']} input tokens/request, "
            f"and ~{stats['requests_per_second']:.2f} req/s, "
            f"{platform_info['recommended_platform'].upper()} aligns better with this workload shape."
        ),
        "actions": [
            f"Run `make replay TARGET=<vllm-url> PLATFORM={platform_info['recommended_platform']} "
            f"TIER={tier_info['closest_tier']}` to benchmark with the canonical trace format.",
            "Compare both platforms with the same trace before committing to migration.",
        ],
    })

    if stats["p95_input_tokens"] > max_model_len * 0.85:
        recs.append({
            "priority": "critical",
            "category": "context",
            "title": "Context length exceeds safe serving headroom",
            "rationale": (
                f"p95 input is ~{stats['p95_input_tokens']} tokens; configured max_model_len is {max_model_len}. "
                "Requests near the limit increase OOM risk and latency variance."
            ),
            "actions": [
                "Truncate or summarize prompts above 75% of max_model_len before inference.",
                f"If truncation is unacceptable, raise --max-model-len (GPU/TPU currently {max_model_len}).",
                "Split long documents into chunked retrieval steps instead of single-shot prompts.",
            ],
        })

    if prefix_stats["detected"] and prefix_stats["avg_prefix_tokens"] >= 128:
        recs.append({
            "priority": "high",
            "category": "caching",
            "title": "Enable prefix / KV-cache reuse",
            "rationale": (
                f"{int(prefix_stats['ratio'] * 100)}% of prompts share a common prefix "
                f"(~{prefix_stats['avg_prefix_tokens']} tokens)."
            ),
            "actions": [
                "Structure requests with a stable system prefix separated by `\\n\\n---\\n\\n` (this repo's trace format).",
                "Enable vLLM prefix caching or chunked prefill for repeated prefixes.",
                "Batch requests with identical prefixes to maximize cache hits.",
            ],
        })

    if stats["offset_coverage"] < 0.5:
        recs.append({
            "priority": "medium",
            "category": "scheduling",
            "title": "Add arrival-time offsets for realistic replay",
            "rationale": "Fewer than half of requests include non-zero offsets; burst replay may overestimate queue pressure.",
            "actions": [
                "Export request timestamps and map to `offset` seconds from trace start.",
                "Use `make trace` or spread offsets across span_seconds for load testing.",
            ],
        })

    if stats["max_output_tokens"] > int(gpu_serving.get("max_num_batched_tokens", 4096) / 4):
        recs.append({
            "priority": "medium",
            "category": "batching",
            "title": "Tune max_num_batched_tokens for output length",
            "rationale": f"Peak max_tokens={stats['max_output_tokens']} may limit batching efficiency.",
            "actions": [
                f"Review --max-num-batched-tokens (currently {gpu_serving.get('max_num_batched_tokens')}).",
                "Cap max_tokens to the p95 completion length observed in production logs.",
            ],
        })

    if user_context.get("current_platform") and user_context.get("target_platform"):
        src = user_context["current_platform"]
        tgt = user_context["target_platform"]
        if src != tgt:
            recs.append({
                "priority": "high",
                "category": "migration",
                "title": f"Migrate {src.upper()} → {tgt.upper()}",
                "rationale": "Cross-platform migration requires aligned serving flags and Kubernetes manifests.",
                "actions": [
                    f"Regenerate manifests: `make manifests PLATFORM={tgt}`",
                    f"Deploy vLLM image from benchmark_config ({platform_metadata(cfg, tgt)['docker_image']}).",
                    "Re-run replay on both platforms with identical trace before cutover.",
                ],
            })

    if tier_info["closest_tier"] == "high":
        recs.append({
            "priority": "medium",
            "category": "cost",
            "title": "Complex tier — validate TPU total cost of ownership",
            "rationale": "High-volume long-context workloads often favor TPU throughput but need cost verification.",
            "actions": [
                "Compare $/1M output tokens from replay cost_metrics on both platforms.",
                "Watch p95 TTFT and batch size under sustained load.",
            ],
        })

    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    recs.sort(key=lambda r: order.get(r["priority"], 9))
    return recs


def _build_migration_steps(
    cfg: dict[str, Any],
    tier_info: dict[str, Any],
    platform_info: dict[str, Any],
    user_context: dict[str, Any],
) -> list[dict[str, Any]]:
    target = user_context.get("target_platform") or platform_info["recommended_platform"]
    source = user_context.get("current_platform") or ("gpu" if target == "tpu" else "tpu")
    diff = migration_diff(cfg)
    changed = [row for row in diff if not row.get("same")]

    steps: list[dict[str, Any]] = [
        {
            "phase": "Assess",
            "title": "Normalize workload trace",
            "details": [
                "Ensure each line has `prompt`, `max_tokens`, and optional `offset` (seconds from start).",
                f"Closest benchmark tier: **{tier_info['label']}** (`{tier_info['closest_tier']}`).",
                "Save as JSONL and validate with the advisor before replay.",
            ],
        },
        {
            "phase": "Prepare",
            "title": f"Align {source.upper()} and {target.upper()} serving config",
            "details": [
                f"Model: `{cfg['model']}` (same on both platforms).",
                f"Target docker image: `{platform_metadata(cfg, target)['docker_image']}`.",
                "Diff serving flags: "
                + ", ".join(f"{r['key']}" for r in changed if r["category"] == "serving")[:200]
                or "max_model_len, batching limits",
            ],
        },
        {
            "phase": "Deploy",
            "title": f"Provision {target.upper()} cluster",
            "details": [
                f"`make manifests` → apply `{cfg[target]['k8s']['output_file']}`.",
                f"Node pool: {cfg[target]['node_pool']['name']} ({cfg[target]['node_pool']['machine_type']}).",
                "Port-forward vLLM: `kubectl port-forward svc/<service> 8000:8000`.",
            ],
        },
        {
            "phase": "Validate",
            "title": "Replay and compare metrics",
            "details": [
                f"`make replay TARGET=http://127.0.0.1:8000 PLATFORM={target} TIER={tier_info['closest_tier']}`",
                "Check tok/s, p95 TTFT, success rate, and $/1M tokens in the results UI.",
                "Only migrate production traffic after SLO parity on the same trace.",
            ],
        },
        {
            "phase": "Cutover",
            "title": "Progressive traffic shift",
            "details": [
                "Canary 5–10% traffic; monitor error rate and latency SLOs.",
                "Keep rollback manifest for source platform for one release cycle.",
                "Upload replay JSON to GCS (`make upload-results`) for team visibility.",
            ],
        },
    ]
    return steps


def analyze_workload(
    content: str,
    *,
    filename: str = "upload.jsonl",
    current_platform: str | None = None,
    target_platform: str | None = None,
    model: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = cfg or load_config()
    parsed = parse_workload_text(content, filename=filename)
    if parsed.errors:
        return {"ok": False, "errors": parsed.errors, "warnings": parsed.warnings}

    records = parsed.records
    input_tokens = [_estimate_tokens(r.prompt) for r in records]
    output_tokens = [r.max_tokens for r in records]
    offsets = [r.offset for r in records]
    span = max(offsets) - min(offsets) if offsets else 0.0
    span = span if span > 0 else max(1.0, len(records) * 0.5)
    rps = len(records) / span

    categories: dict[str, int] = {}
    for rec in records:
        if rec.category:
            categories[rec.category] = categories.get(rec.category, 0) + 1

    prefix_stats = _shared_prefix_stats(records)
    avg_in = statistics.mean(input_tokens)
    avg_out = statistics.mean(output_tokens)
    tier_info = _classify_tier(len(records), avg_in, avg_out, cfg)

    total_tokens = sum(input_tokens) + sum(output_tokens)
    platform_info = _platform_scores(
        len(records), total_tokens, avg_in, avg_out, rps, cfg,
    )

    user_context = {
        "current_platform": (current_platform or "").lower() or None,
        "target_platform": (target_platform or "").lower() or None,
        "model": model or cfg.get("model"),
    }
    if user_context["current_platform"] not in ("gpu", "tpu"):
        user_context["current_platform"] = None
    if user_context["target_platform"] not in ("gpu", "tpu"):
        user_context["target_platform"] = None

    stats = {
        "num_requests": len(records),
        "avg_input_tokens": round(avg_in, 1),
        "p95_input_tokens": round(_percentile(input_tokens, 95), 1),
        "max_input_tokens": max(input_tokens),
        "avg_output_tokens": round(avg_out, 1),
        "p95_output_tokens": round(_percentile(output_tokens, 95), 1),
        "max_output_tokens": max(output_tokens),
        "span_seconds": round(span, 2),
        "requests_per_second": round(rps, 3),
        "offset_coverage": round(sum(1 for o in offsets if o > 0) / len(offsets), 3),
        "categories": categories,
        "shared_prefix": prefix_stats,
    }

    recommendations = _build_recommendations(
        stats, prefix_stats, tier_info, platform_info, cfg, user_context,
    )
    migration_steps = _build_migration_steps(cfg, tier_info, platform_info, user_context)

    normalized_sample = [
        {
            "prompt": r.prompt[:120] + ("…" if len(r.prompt) > 120 else ""),
            "max_tokens": r.max_tokens,
            "offset": r.offset,
            **({"category": r.category} if r.category else {}),
        }
        for r in records[:3]
    ]

    return {
        "ok": True,
        "filename": filename,
        "model": user_context["model"],
        "warnings": parsed.warnings,
        "stats": stats,
        "tier": tier_info,
        "platform": platform_info,
        "recommendations": recommendations,
        "migration_steps": migration_steps,
        "sample_records": normalized_sample,
        "trace_format": {
            "required_fields": ["prompt", "max_tokens"],
            "optional_fields": ["offset", "category", "id"],
            "example": {
                "prompt": "System context…\\n\\n---\\n\\nUser question…",
                "max_tokens": 128,
                "offset": 0.0,
                "category": "qa",
            },
        },
    }


def analyze_workload_file(path: Path, **kwargs: Any) -> dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    return analyze_workload(content, filename=path.name, **kwargs)
