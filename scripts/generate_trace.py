#!/usr/bin/env python3
"""Generate a JSONL inference trace aligned with benchmark_config.yaml."""

from __future__ import annotations

import argparse
import json
import random
import secrets
import time
from pathlib import Path

from config import load_config

SHARED_SYSTEM_PROMPT = """You are a helpful enterprise assistant for Acme Corp.
Follow company policy: be concise, cite sources when available, and never share
confidential data. Use markdown for structured answers."""

USER_TOPICS = [
    "quarterly revenue breakdown",
    "customer churn analysis",
    "product roadmap priorities",
    "support ticket escalation policy",
    "security compliance checklist",
    "marketing campaign performance",
    "inventory forecast for Q4",
    "hiring plan for engineering",
    "API rate limit configuration",
    "data retention policy summary",
]


def _approx_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _pad_to_tokens(text: str, target_tokens: int) -> str:
    words = text.split()
    if len(words) >= target_tokens:
        return " ".join(words[:target_tokens])
    filler = "Additional context for the request. " * ((target_tokens - len(words)) // 4 + 1)
    return f"{text} {filler}"


def _user_context(rng: random.Random, target_input_tokens: int, shared_prefix_tokens: int) -> str:
    topic = rng.choice(USER_TOPICS)
    base = (
        f"User question about {topic}. "
        f"Please analyze the following notes and provide recommendations. "
    )
    detail = " ".join(
        f"Detail point {i}: metric value {rng.randint(10, 999)} with trend {rng.choice(['up', 'down', 'flat'])}."
        for i in range(1, 80)
    )
    # Reserve space for shared prefix when composing full prompt.
    user_budget = max(32, target_input_tokens - shared_prefix_tokens - 8)
    return _pad_to_tokens(f"{base}{detail}", user_budget)


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
) -> list[dict]:
    rng = random.Random(seed)
    system_prompt = _pad_to_tokens(SHARED_SYSTEM_PROMPT, shared_prefix_tokens)
    offsets = _generate_offsets(rng, num_requests, span_seconds)
    records = []
    for offset in offsets:
        user_context = _user_context(rng, input_tokens, shared_prefix_tokens)
        prompt = _pad_to_tokens(f"{system_prompt}\n\n---\n\n{user_context}", input_tokens)
        records.append(
            {
                "prompt": prompt,
                "max_tokens": output_tokens,
                "offset": offset,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate workload JSONL from benchmark_config.yaml.")
    parser.add_argument("-o", "--output", default="workload/prompts.jsonl")
    parser.add_argument("--meta", default="workload/trace_meta.json", help="Write seed and params here")
    parser.add_argument("--seed", type=int, default=None, help="Omit for a new random seed each run")
    parser.add_argument("--config", help="Path to benchmark_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    bench = cfg["benchmark"]
    num_requests = int(bench["num_prompts"])
    input_tokens = int(bench["input_tokens"])
    output_tokens = int(bench["output_tokens"])
    span_seconds = float(bench.get("trace_span_seconds", max(120.0, num_requests * 2.0)))
    shared_prefix_tokens = int(bench.get("shared_prefix_tokens", min(256, input_tokens // 4)))

    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    records = generate_trace(
        num_requests=num_requests,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        span_seconds=span_seconds,
        seed=seed,
        shared_prefix_tokens=shared_prefix_tokens,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    meta = {
        "seed": seed,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": cfg["model"],
        "num_requests": num_requests,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "span_seconds": span_seconds,
        "output_path": str(output),
    }
    meta_path = Path(args.meta)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(records)} requests to {output}")
    print(f"Trace seed: {seed} (save this to reproduce the same workload)")
    print(f"Metadata: {meta_path}")
    print(f"Approx input tokens (first prompt): {_approx_tokens(records[0]['prompt'])}")


if __name__ == "__main__":
    main()
