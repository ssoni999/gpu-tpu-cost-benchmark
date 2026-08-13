#!/usr/bin/env python3
"""Gemini-backed replacement for the rule-based platform advisor.

Builds a prompt from parsed workload stats and the GPU/TPU platform catalog,
then asks Gemini for a structured GPU vs TPU recommendation in the same shape
the advisor UI already renders (platform fit scores, prioritized
recommendations, migration steps).
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal

from pydantic import BaseModel

from config import ROOT, platform_metadata

DEFAULT_MODEL = "gemini-flash-latest"
REQUEST_TIMEOUT_MS = 30_000


class GeminiAdvisorError(RuntimeError):
    pass


class RecommendationItem(BaseModel):
    priority: Literal["critical", "high", "medium", "low"]
    category: str
    title: str
    rationale: str
    actions: list[str]


class MigrationStep(BaseModel):
    phase: str
    title: str
    details: list[str]


class PlatformGuidance(BaseModel):
    recommended_platform: Literal["gpu", "tpu"]
    confidence: float
    gpu_fit_score: float
    tpu_fit_score: float
    cost_efficiency_leader: Literal["gpu", "tpu"]
    peak_throughput_leader: Literal["gpu", "tpu"]
    summary: str


class AiGuidance(BaseModel):
    platform: PlatformGuidance
    recommendations: list[RecommendationItem]
    migration_steps: list[MigrationStep]


_PROMPT_TEMPLATE = """You are a GPU vs TPU inference infrastructure advisor for this benchmark project.
Recommend whether to serve the workload described below on GPU or TPU, and produce a migration playbook.

## Model
{model}

## Workload attributes (parsed from the uploaded trace)
{stats_json}

## Closest known benchmark tier
{tier_json}

## GPU platform (catalog specs — accelerator, pricing, node pool)
{gpu_json}

## TPU platform (catalog specs — accelerator, pricing, node pool)
{tpu_json}

## User-provided context
{user_context_json}

## Instructions
- Do not let ordering, verbosity, or how much detail is given about one platform influence recommended_platform — weigh GPU and TPU strictly on the numeric evidence (throughput, cost, latency) provided for each.
- Base the recommendation on workload shape (request volume, context length, output length, request rate) against each platform's throughput/cost profile from catalog specs.
- gpu_fit_score and tpu_fit_score are 0-1 and do not need to sum to 1.
- confidence is 0-1, how sure you are in recommended_platform.
- Produce 3-6 recommendations covering platform choice, context-length risk, prefix/KV-cache opportunities, batching/throughput tuning, and cost, ordered most important first.
- Produce a 5-phase migration playbook: Assess, Prepare, Deploy, Validate, Cutover — concrete and specific to this project's tooling (make manifests, make replay, kubectl, results UI).
- Keep rationale and details grounded in the numbers given; do not invent benchmark results that were not provided.
- If "current_platform" is set in the user-provided context, the workload is already running there today — frame the migration playbook as a migration FROM current_platform, not a fresh deployment.
- If "target_platform" is set in the user-provided context, the migration playbook's destination MUST be target_platform, even if it differs from recommended_platform — the user has already decided where they're headed; your job is the safe path there. If target_platform conflicts with what the data actually supports, say so explicitly in one of the recommendations (e.g. "you're targeting TPU but this workload's numbers favor GPU — proceed only if X") rather than silently overriding either value.
- If "target_platform" is not set, use recommended_platform as the migration playbook's destination.
- Never let current_platform/target_platform bias recommended_platform, gpu_fit_score, tpu_fit_score, or confidence — those must always reflect genuine analysis of the workload and platform data, independent of what the user says they're targeting.
"""


def _gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key

    env_path = ROOT / "configs" / "gemini.env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line.startswith("export "):
                continue
            body = line[len("export "):]
            if not body.startswith("GEMINI_API_KEY="):
                continue
            return body.split("=", 1)[1].strip().strip('"').strip("'")

    raise GeminiAdvisorError(
        "GEMINI_API_KEY not set. Add it to configs/gemini.env (see gemini.env.example) "
        "and restart results_server.py."
    )


def _pricing(cfg: dict[str, Any], platform: str) -> dict[str, float]:
    pricing = cfg.get("pricing", {})
    return {
        "accelerator_hourly_usd": float(pricing.get(f"{platform}_accelerator_hourly_usd", 0.0)),
        "vm_hourly_usd": float(pricing.get(f"{platform}_vm_hourly_usd", 0.0)),
    }


def _platform_context(cfg: dict[str, Any], platform: str) -> dict[str, Any]:
    meta = platform_metadata(cfg, platform)
    return {
        "accelerator": meta.get("accelerator"),
        "peak_tflops": meta.get("peak_tflops"),
        "power_watts": meta.get("power_watts"),
        "node_pool": meta.get("node_pool"),
        "pricing": _pricing(cfg, platform),
    }


def _build_prompt(
    stats: dict[str, Any],
    tier_info: dict[str, Any],
    cfg: dict[str, Any],
    user_context: dict[str, Any],
) -> str:
    return _PROMPT_TEMPLATE.format(
        model=cfg.get("model", "unknown"),
        stats_json=json.dumps(stats, indent=2, default=str),
        tier_json=json.dumps(
            {"closest_tier": tier_info.get("closest_tier"), "label": tier_info.get("label"),
             "description": tier_info.get("description")},
            indent=2,
        ),
        gpu_json=json.dumps(_platform_context(cfg, "gpu"), indent=2, default=str),
        tpu_json=json.dumps(_platform_context(cfg, "tpu"), indent=2, default=str),
        user_context_json=json.dumps(user_context, indent=2, default=str),
    )


def generate_ai_guidance(
    stats: dict[str, Any],
    tier_info: dict[str, Any],
    cfg: dict[str, Any],
    *,
    user_context: dict[str, Any] | None = None,
    model_name: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    api_key = _gemini_api_key()
    prompt = _build_prompt(stats, tier_info, cfg, user_context or {})

    try:
        client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AiGuidance,
                temperature=0.2,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        raise GeminiAdvisorError(f"Gemini request failed: {exc}") from exc

    guidance: AiGuidance | None = response.parsed
    if guidance is None:
        raise GeminiAdvisorError(f"Gemini returned no parsable guidance: {response.text[:500] if response.text else '(empty)'}")

    return guidance.model_dump()
