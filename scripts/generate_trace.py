#!/usr/bin/env python3
"""Generate a JSONL inference trace aligned with benchmark_config.yaml."""

from __future__ import annotations

import argparse
import json
import random
import secrets
import time
from pathlib import Path

from config import default_workload, load_config, resolve_workload_trace_path, workload_profile, workload_trace_meta_path, workload_trace_path
from prompt_catalog import SHARED_HANDBOOK, build_user_prompt


# Word-split budgets underestimate BPE on Qwen-style prompts; use a conservative ratio.
BPE_TOKENS_PER_WORD = 2.15
CHAT_TEMPLATE_BUFFER = 32
TOKEN_BUDGET_SAFETY = 16


def _approx_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * BPE_TOKENS_PER_WORD))


def _word_budget(token_budget: int) -> int:
    return max(1, int(token_budget / BPE_TOKENS_PER_WORD))


def _trim_to_tokens(text: str, target_tokens: int) -> str:
    words = text.split()
    if len(words) <= target_tokens:
        return text
    return " ".join(words[:target_tokens])


def _enforce_max_prompt_tokens(prompt: str, max_prompt_tokens: int) -> str:
    """Trim from the end until the conservative token estimate fits."""
    if _approx_tokens(prompt) <= max_prompt_tokens:
        return prompt
    words = prompt.split()
    while len(words) > 64 and _approx_tokens(" ".join(words)) > max_prompt_tokens:
        words = words[:-12]
    return " ".join(words)


def _append_varied_context(rng: random.Random, text: str, target_tokens: int, request_id: int) -> str:
    """Extend with unique appendix bullets until the word budget is met."""
    words = text.split()
    if len(words) >= target_tokens:
        return _trim_to_tokens(text, target_tokens)

    appendix = [
        "Appendix — supplemental facts for this request only:",
    ]
    topics = [
        "recent deploy",
        "customer escalation",
        "capacity plan",
        "dependency outage",
        "data quality check",
        "contract clause",
        "monitoring alert",
        "experiment result",
    ]
    while len(" ".join(words + appendix).split()) < target_tokens:
        topic = rng.choice(topics)
        appendix.append(
            f"- Item {len(appendix)} (req {request_id}): {topic}; "
            f"value {rng.randint(1, 9999)}; region {rng.choice(['us-central1', 'us-east1', 'eu-west4'])}; "
            f"owner team {rng.choice(['platform', 'data', 'security', 'finance'])}; "
            f"note: {rng.choice(['investigate', 'monitor', 'document', 'escalate'])}."
        )
    combined = text + "\n\n" + "\n".join(appendix)
    return _trim_to_tokens(combined, target_tokens)


def _shared_prefix(shared_prefix_tokens: int) -> str:
    prefix = SHARED_HANDBOOK.strip()
    words = prefix.split()
    if len(words) >= shared_prefix_tokens:
        return " ".join(words[:shared_prefix_tokens])
    # Extend handbook once with fixed policy bullets (same for every request).
    extra = [
        "Revision 2026.03: all production changes require change ticket.",
        "Revision 2026.03: customer-facing incidents need status page update within 15 minutes.",
        "Revision 2026.03: export controls apply to certain model weights and regions.",
    ]
    extended = prefix + "\n" + "\n".join(extra)
    return _trim_to_tokens(extended, shared_prefix_tokens)


def _generate_offsets(rng: random.Random, num_requests: int, span_seconds: float) -> list[float]:
    offsets: list[float] = []
    t = 0.0
    burst_remaining = 0
    while len(offsets) < num_requests:
        if burst_remaining > 0:
            gap = rng.uniform(0.05, 0.25)
            burst_remaining -= 1
        elif rng.random() < 0.08:
            burst_remaining = rng.randint(8, 20)
            gap = rng.uniform(0.05, 0.2)
            burst_remaining -= 1
        else:
            avg_gap = span_seconds / max(num_requests, 1)
            gap = rng.expovariate(1.0 / max(avg_gap, 0.001))
        t += gap
        offsets.append(round(t, 3))
    return offsets


def generate_trace(
    num_requests: int,
    input_tokens: int,
    output_tokens: int,
    span_seconds: float,
    seed: int,
    shared_prefix_tokens: int,
    max_model_len: int = 4096,
    max_input_tokens: int | None = None,
) -> list[dict]:
    rng = random.Random(seed)
    system_prompt = _shared_prefix(shared_prefix_tokens)
    max_prompt_tokens = max_model_len - output_tokens - CHAT_TEMPLATE_BUFFER - TOKEN_BUDGET_SAFETY
    if max_input_tokens is not None:
        max_prompt_tokens = min(max_prompt_tokens, int(max_input_tokens))
    system_tok = _approx_tokens(system_prompt)
    user_token_cap = max(64, max_prompt_tokens - system_tok - 8)
    user_token_target = min(input_tokens, user_token_cap)
    user_budget = _word_budget(user_token_target)
    offsets = _generate_offsets(rng, num_requests, span_seconds)
    records = []
    seen_prompts: set[str] = set()

    for idx, offset in enumerate(offsets):
        req_rng = random.Random(seed + idx * 9973)
        request_id = idx + 1
        user_body = build_user_prompt(req_rng, request_id)
        user_body = _append_varied_context(req_rng, user_body, user_budget, request_id)
        prompt = f"{system_prompt}\n\n---\n\n{user_body}"

        # Guarantee uniqueness even if templates collide.
        attempt = 0
        while prompt in seen_prompts and attempt < 5:
            attempt += 1
            user_body = build_user_prompt(req_rng, request_id + attempt * 1000)
            user_body = _append_varied_context(req_rng, user_body, user_budget, request_id + attempt)
            prompt = f"{system_prompt}\n\n---\n\n{user_body}"
        prompt = _enforce_max_prompt_tokens(prompt, max_prompt_tokens)
        seen_prompts.add(prompt)

        records.append(
            {
                "id": request_id,
                "category": TEMPLATE_CATEGORY(user_body),
                "prompt": prompt,
                "max_tokens": output_tokens,
                "offset": offset,
            }
        )
    return records


def TEMPLATE_CATEGORY(user_body: str) -> str:
    first_line = user_body.strip().split("\n", 1)[0].lower()
    if "review this" in first_line and "api" in first_line:
        return "api_design"
    if "review this" in first_line:
        return "code_review"
    if "on-call" in first_line:
        return "incident_triage"
    if "draft a professional reply" in first_line:
        return "customer_email"
    if "write sql" in first_line:
        return "sql_analyst"
    if "policy question" in first_line:
        return "policy_qa"
    if "prioritize these" in first_line:
        return "roadmap"
    if "runbook" in first_line:
        return "runbook"
    if "summarize these meeting" in first_line:
        return "meeting_notes"
    if "how-to doc" in first_line:
        return "docs_howto"
    if "finance narrative" in first_line:
        return "finance_forecast"
    if "security review" in first_line:
        return "security_review"
    if "compare deployment" in first_line:
        return "comparison"
    return "general"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate workload JSONL from benchmark_config.yaml.")
    parser.add_argument("-o", "--output", default="workload/prompts.jsonl")
    parser.add_argument("--meta", default="workload/trace_meta.json", help="Write seed and params here")
    parser.add_argument("--seed", type=int, default=None, help="Omit for a new random seed each run")
    parser.add_argument("--tier", default=None, help="Workload tier: small, medium, high")
    parser.add_argument("--config", help="Path to benchmark_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tier = args.tier or default_workload(cfg)
    bench = workload_profile(cfg, tier)
    num_requests = int(bench["num_prompts"])
    input_tokens = int(bench["input_tokens"])
    output_tokens = int(bench["output_tokens"])
    max_input_tokens = bench.get("max_input_tokens")
    if max_input_tokens is not None:
        input_tokens = min(input_tokens, int(max_input_tokens))
    span_seconds = float(bench.get("trace_span_seconds", max(120.0, num_requests * 2.0)))
    shared_prefix_tokens = int(bench.get("shared_prefix_tokens", min(256, input_tokens // 4)))
    max_model_len = int(
        bench.get(
            "max_model_len",
            max(
                int(cfg.get("tpu", {}).get("serving", {}).get("max_model_len", 4096)),
                int(cfg.get("gpu", {}).get("serving", {}).get("max_model_len", 4096)),
            ),
        )
    )

    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    records = generate_trace(
        num_requests=num_requests,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        span_seconds=span_seconds,
        seed=seed,
        shared_prefix_tokens=shared_prefix_tokens,
        max_model_len=max_model_len,
        max_input_tokens=int(max_input_tokens) if max_input_tokens is not None else None,
    )

    output = resolve_workload_trace_path(tier, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    categories = {r["category"] for r in records}
    meta = {
        "workload_tier": tier,
        "seed": seed,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": cfg["model"],
        "num_requests": num_requests,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "span_seconds": span_seconds,
        "unique_prompts": len({r["prompt"] for r in records}),
        "categories": sorted(categories),
        "output_path": str(output),
    }
    meta_path = Path(args.meta)
    if tier and str(args.meta) == "workload/trace_meta.json":
        meta_path = workload_trace_meta_path(tier)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Workload tier: {tier}")
    print(f"Wrote {len(records)} requests to {output}")
    print(f"Trace seed: {seed} (save this to reproduce the same workload)")
    print(f"Unique prompts: {meta['unique_prompts']}/{num_requests}")
    print(f"Categories: {', '.join(sorted(categories))}")
    print(f"Metadata: {meta_path}")
    print(f"Approx prompt tokens (first prompt): {_approx_tokens(records[0]['prompt'])}")
    print(f"Max model len (trace cap): {max_model_len} (prompt + {output_tokens} output must fit)")
    peak = max(_approx_tokens(r["prompt"]) for r in records)
    print(f"Peak approx prompt tokens: {peak} (limit {max_model_len - output_tokens - CHAT_TEMPLATE_BUFFER})")


if __name__ == "__main__":
    main()
