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
    workload_profile,
    workload_tier_ids,
    workloads_catalog,
)
from gemini_advisor import GeminiAdvisorError, generate_ai_guidance

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


def analyze_workload(
    content: str,
    *,
    filename: str = "upload.jsonl",
    current_platform: str | None = None,
    target_platform: str | None = None,
    model: str | None = None,
    cfg: dict[str, Any] | None = None,
    gpu_result: dict[str, Any] | None = None,
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
        "total_tokens_estimate": int(sum(input_tokens) + sum(output_tokens)),
        "span_seconds": round(span, 2),
        "requests_per_second": round(rps, 3),
        "offset_coverage": round(sum(1 for o in offsets if o > 0) / len(offsets), 3),
        "categories": categories,
        "shared_prefix": prefix_stats,
    }

    try:
        guidance = generate_ai_guidance(
            stats, tier_info, cfg, user_context=user_context, gpu_result=gpu_result,
        )
    except GeminiAdvisorError as exc:
        return {"ok": False, "errors": [str(exc)], "warnings": parsed.warnings}

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
        "platform": guidance["platform"],
        "recommendations": guidance["recommendations"],
        "migration_steps": guidance["migration_steps"],
        "sample_records": normalized_sample,
        "ai_generated": True,
        "used_gpu_result": gpu_result is not None,
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
