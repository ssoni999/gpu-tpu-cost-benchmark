#!/usr/bin/env python3
"""Replay a JSONL trace against an OpenAI-compatible vLLM endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import openai

from config import load_config


@dataclass
class TraceRecord:
    prompt: str
    max_tokens: int
    offset: float


@dataclass
class RequestResult:
    index: int
    offset: float
    max_tokens: int
    status: str
    latency_ms: float
    ttft_ms: float
    prompt_tokens: int
    completion_tokens: int
    error: Optional[str] = None


def load_trace(path: Path) -> list[TraceRecord]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(
                TraceRecord(
                    prompt=data["prompt"],
                    max_tokens=int(data["max_tokens"]),
                    offset=float(data.get("offset", 0.0)),
                )
            )
    return records


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def check_endpoint(base_url: str) -> None:
    models_url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(models_url, timeout=10) as resp:
            resp.read(200)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Cannot reach {models_url}: {exc}", file=sys.stderr)
        print(
            "\nFix: deploy vLLM, then port-forward or use the service URL:\n"
            "  kubectl port-forward svc/vllm-service 8000:8000\n"
            "  curl http://127.0.0.1:8000/v1/models\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def resolve_api_mode(model: str, api: str) -> str:
    if api != "auto":
        return api
    model_lower = model.lower()
    if "opt-125" in model_lower or "/opt-" in model_lower:
        return "completions"
    return "chat"


async def send_request(
    client: openai.AsyncOpenAI,
    model: str,
    record: TraceRecord,
    index: int,
    api_mode: str,
) -> RequestResult:
    start = time.perf_counter()
    ttft_ms = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    try:
        if api_mode == "completions":
            stream = await client.completions.create(
                model=model,
                prompt=record.prompt,
                max_tokens=record.max_tokens,
                temperature=0,
                stream=True,
                stream_options={"include_usage": True},
            )
        else:
            stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": record.prompt}],
                max_tokens=record.max_tokens,
                temperature=0,
                stream=True,
                stream_options={"include_usage": True},
            )
        first_token = None
        async for chunk in stream:
            if hasattr(chunk, "usage") and chunk.usage is not None:
                prompt_tokens = chunk.usage.prompt_tokens or prompt_tokens
                completion_tokens = chunk.usage.completion_tokens or completion_tokens
            if not chunk.choices:
                continue
            if api_mode == "completions":
                content = chunk.choices[0].text
            else:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None) or getattr(
                    delta, "reasoning_content", None
                )
            if content and first_token is None:
                first_token = time.perf_counter()
        end = time.perf_counter()
        if first_token is None:
            first_token = end
        ttft_ms = (first_token - start) * 1000
        latency_ms = (end - start) * 1000
        return RequestResult(
            index=index,
            offset=record.offset,
            max_tokens=record.max_tokens,
            status="ok",
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        end = time.perf_counter()
        return RequestResult(
            index=index,
            offset=record.offset,
            max_tokens=record.max_tokens,
            status="error",
            latency_ms=(end - start) * 1000,
            ttft_ms=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            error=str(exc),
        )


async def warmup(
    client: openai.AsyncOpenAI,
    model: str,
    count: int,
    api_mode: str,
) -> None:
    for i in range(count):
        await send_request(
            client,
            model,
            TraceRecord(prompt=f"WARMUP {i}: reply ok", max_tokens=8, offset=0.0),
            index=-1,
            api_mode=api_mode,
        )


async def replay_trace(
    target: str,
    model: str,
    trace: list[TraceRecord],
    speed: float,
    concurrency: int,
    warmup_count: int,
    api_mode: str,
) -> tuple[list[RequestResult], float]:
    base_url = target.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    check_endpoint(base_url)
    client = openai.AsyncOpenAI(api_key="EMPTY", base_url=base_url)

    if warmup_count > 0:
        await warmup(client, model, warmup_count, api_mode)

    semaphore = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []
    replay_start = time.perf_counter()

    async def run_one(index: int, record: TraceRecord) -> None:
        target_time = replay_start + (record.offset / speed)
        delay = target_time - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        async with semaphore:
            results.append(await send_request(client, model, record, index, api_mode))

    await asyncio.gather(*(run_one(i, record) for i, record in enumerate(trace)))
    wall_time = time.perf_counter() - replay_start
    results.sort(key=lambda r: r.index)
    return results, wall_time


def summarize(
    results: list[RequestResult],
    wall_time: float,
    trace_path: str,
    target: str,
    model: str,
) -> dict:
    ok = [r for r in results if r.status == "ok"]
    errors = [r for r in results if r.status != "ok"]
    latencies = [r.latency_ms for r in ok]
    ttfts = [r.ttft_ms for r in ok]
    input_tokens = sum(r.prompt_tokens for r in ok)
    output_tokens = sum(r.completion_tokens for r in ok)
    total_tokens = input_tokens + output_tokens
    throughput = total_tokens / wall_time if wall_time > 0 else 0.0
    gen_tps = output_tokens / wall_time if wall_time > 0 else 0.0
    req_per_sec = len(ok) / wall_time if wall_time > 0 else 0.0

    return {
        "source": "trace_replay",
        "target": target,
        "model": model,
        "trace": trace_path,
        "wall_time_s": round(wall_time, 2),
        "successful_requests": len(ok),
        "failed_requests": len(errors),
        "benchmark_duration_seconds": round(wall_time, 2),
        "total_input_tokens": input_tokens,
        "total_generated_tokens": output_tokens,
        "requests_per_second": round(req_per_sec, 4),
        "output_tokens_per_second": round(gen_tps, 2),
        "throughput_tok_s": round(throughput, 1),
        "generation_tok_s": round(gen_tps, 1),
        "latency_ms": {
            "p50": round(percentile(latencies, 50), 1),
            "p95": round(percentile(latencies, 95), 1),
            "p99": round(percentile(latencies, 99), 1),
        },
        "ttft_ms": {
            "p50": round(percentile(ttfts, 50), 1),
            "p95": round(percentile(ttfts, 95), 1),
            "p99": round(percentile(ttfts, 99), 1),
        },
        "median_ttft_ms": round(percentile(ttfts, 50), 1),
        "p95_ttft_ms": round(percentile(ttfts, 95), 1),
        "p99_ttft_ms": round(percentile(ttfts, 99), 1),
        "results": [asdict(r) for r in results],
    }


def print_summary(summary: dict) -> None:
    print()
    print("==================== Replay summary ======================")
    print(f"  Output tok/s:   {summary['output_tokens_per_second']:,.1f}")
    print(f"  TTFT p95:       {summary['ttft_ms']['p95']:.0f} ms")
    print(f"  Latency p95:    {summary['latency_ms']['p95']:.0f} ms")
    print(
        f"  Requests:       {summary['successful_requests']} ok / "
        f"{summary['failed_requests']} err"
    )
    print(f"  Wall time:      {summary['wall_time_s']:.1f}s")
    print("==========================================================")


def main() -> int:
    cfg = load_config()
    bench = cfg["benchmark"]
    default_model = cfg["model"]

    parser = argparse.ArgumentParser(description="Replay workload/prompts.jsonl against vLLM.")
    parser.add_argument("--target", required=True, help="Base URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--trace", default="workload/prompts.jsonl")
    parser.add_argument("--output", default="results/tpu/run_01_replay.json")
    parser.add_argument("--speed", type=float, default=10.0, help="Time compression (higher=faster)")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=int(bench.get("warmup_requests", 10)))
    parser.add_argument("--api", choices=["auto", "chat", "completions"], default="auto")
    args = parser.parse_args()

    trace_path = Path(args.trace)
    if not trace_path.exists():
        print(f"Trace not found: {trace_path}. Run: make trace", file=sys.stderr)
        return 1

    trace = load_trace(trace_path)
    if not trace:
        print(f"No records in {trace_path}", file=sys.stderr)
        return 1

    api_mode = resolve_api_mode(args.model, args.api)
    print(
        f"Replaying {len(trace)} requests to {args.target} "
        f"(speed={args.speed}x, api={api_mode})"
    )
    results, wall_time = asyncio.run(
        replay_trace(
            args.target,
            args.model,
            trace,
            args.speed,
            args.concurrency,
            args.warmup,
            api_mode,
        )
    )
    summary = summarize(results, wall_time, str(trace_path), args.target, args.model)
    print_summary(summary)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")

    return 0 if summary["failed_requests"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
