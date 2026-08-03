#!/usr/bin/env python3
"""Serve the benchmark results UI — reads results/*/run_01_replay.json (port 8787)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiohttp import web

from config import load_config, migration_diff, platform_metadata

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
RESULTS = ROOT / "results"

PLATFORMS = ("tpu", "gpu")


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _result_paths(platform: str) -> dict[str, Path]:
    base = RESULTS / platform
    return {
        "replay": base / "run_01_replay.json",
        "live": base / "live.json",
        "comparison": ROOT / "comparison.json",
    }


async def handle_index(_request: web.Request) -> web.Response:
    index = FRONTEND / "index.html"
    if not index.exists():
        raise web.HTTPNotFound(text="frontend/index.html missing")
    return web.Response(text=index.read_text(encoding="utf-8"), content_type="text/html")


async def handle_results(request: web.Request) -> web.Response:
    platform = request.match_info["platform"]
    if platform not in PLATFORMS:
        raise web.HTTPBadRequest(text=f"platform must be one of {PLATFORMS}")
    paths = _result_paths(platform)
    replay = _read_json(paths["replay"])
    live = _read_json(paths["live"])
    payload = {
        "platform": platform,
        "replay_path": str(paths["replay"].relative_to(ROOT)),
        "live_path": str(paths["live"].relative_to(ROOT)),
        "replay": replay,
        "live": live,
        "has_replay": replay is not None,
        "is_running": bool(live and live.get("status") == "running"),
    }
    if replay is None and live and live.get("summary"):
        payload["replay"] = live["summary"]
        payload["has_replay"] = True
    return web.json_response(payload)


async def handle_migration(_request: web.Request) -> web.Response:
    cfg = load_config()
    return web.json_response({
        "model": cfg["model"],
        "benchmark": cfg["benchmark"],
        "gpu": platform_metadata(cfg, "gpu"),
        "tpu": platform_metadata(cfg, "tpu"),
        "diff": migration_diff(cfg),
    })


async def handle_compare(_request: web.Request) -> web.Response:
    comparison = _read_json(ROOT / "comparison.json")
    tpu = _read_json(RESULTS / "tpu" / "run_01_replay.json")
    gpu = _read_json(RESULTS / "gpu" / "run_01_replay.json")
    return web.json_response({
        "comparison": comparison,
        "tpu": tpu,
        "gpu": gpu,
    })


def build_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/results/{platform}", handle_results)
    app.router.add_get("/api/migration", handle_migration)
    app.router.add_get("/api/compare", handle_compare)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark results UI — reads results/{tpu,gpu}/run_01_replay.json"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    print(f"Results UI: http://127.0.0.1:{args.port}/")
    print("  Data: results/tpu/run_01_replay.json, results/gpu/run_01_replay.json")
    print("  (Does not use port 8765 — safe alongside gpu-tpu-sim-poc)")
    print("  Cloud Shell: Web Preview → port", args.port)
    web.run_app(build_app(), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
