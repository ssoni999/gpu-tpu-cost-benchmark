#!/usr/bin/env python3
"""Serve the benchmark results UI — reads local files or GCS latest/*.json (port 8787)."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from aiohttp import web

from config import load_config, migration_diff, node_pool_specs, platform_metadata
from cost_metrics import compute_replay_cost_metrics
from gcs_results import GcsResultsStore, GcsSettings, gcs_settings, latest_replay_object

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
RESULTS = ROOT / "results"

PLATFORMS = ("tpu", "gpu")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _local_paths(platform: str) -> dict[str, Path]:
    base = RESULTS / platform
    return {
        "replay": base / "run_01_replay.json",
        "live": base / "live.json",
    }


class ResultsBackend:
    def __init__(self, source: str, gcs: GcsResultsStore | None) -> None:
        self.source = source
        self.gcs = gcs

    @property
    def data_source(self) -> str:
        if self.source == "gcs" or (self.source == "auto" and self.gcs and self.gcs.enabled):
            return "gcs"
        return "local"

    async def _gcs_read(self, fn) -> dict[str, Any] | None:
        if not self.gcs or not self.gcs.enabled:
            return None
        return await asyncio.to_thread(fn)

    async def get_replay(self, platform: str) -> tuple[dict[str, Any] | None, str]:
        gcs_path = ""
        if self.gcs and self.gcs.enabled and self.source in ("gcs", "auto"):
            gcs_path = f"gs://{self.gcs.settings.bucket}/{latest_replay_object(platform, self.gcs.settings)}"
            replay = await self._gcs_read(lambda p=platform: self.gcs.read_replay(p))  # type: ignore[union-attr]
            if replay is not None or self.source == "gcs":
                return replay, gcs_path

        paths = _local_paths(platform)
        replay = _read_json(paths["replay"])
        return replay, str(paths["replay"].relative_to(ROOT))

    async def get_live(self, platform: str) -> tuple[dict[str, Any] | None, str]:
        if self.gcs and self.gcs.enabled and self.source in ("gcs", "auto"):
            live = await self._gcs_read(lambda p=platform: self.gcs.read_live(p))  # type: ignore[union-attr]
            if live is not None or self.source == "gcs":
                obj = f"gs://{self.gcs.settings.bucket}/{self.gcs.settings.prefix}/{platform}/live.json"
                return live, obj

        paths = _local_paths(platform)
        live = _read_json(paths["live"])
        return live, str(paths["live"].relative_to(ROOT))

    async def get_comparison(self) -> dict[str, Any] | None:
        if self.gcs and self.gcs.enabled and self.source in ("gcs", "auto"):
            comp = await self._gcs_read(self.gcs.read_comparison)  # type: ignore[arg-type]
            if comp is not None or self.source == "gcs":
                return comp
        return _read_json(ROOT / "comparison.json")

    async def get_manifest(self) -> dict[str, Any]:
        if self.gcs and self.gcs.enabled and self.source in ("gcs", "auto"):
            return await self._gcs_read(self.gcs.read_manifest) or {}  # type: ignore[arg-type]
        return {}


def build_app(backend: ResultsBackend) -> web.Application:
    app = web.Application()
    app["backend"] = backend

    async def handle_index(_request: web.Request) -> web.Response:
        index = FRONTEND / "index.html"
        if not index.exists():
            raise web.HTTPNotFound(text="frontend/index.html missing")
        return web.Response(text=index.read_text(encoding="utf-8"), content_type="text/html")

    async def handle_status(_request: web.Request) -> web.Response:
        backend: ResultsBackend = _request.app["backend"]
        manifest = await backend.get_manifest()
        settings = backend.gcs.settings if backend.gcs else None
        return web.json_response({
            "data_source": backend.data_source,
            "gcs_bucket": settings.bucket if settings else None,
            "gcs_project": settings.project if settings else None,
            "gcs_prefix": settings.prefix if settings else None,
            "manifest": manifest,
        })

    async def handle_results(request: web.Request) -> web.Response:
        platform = request.match_info["platform"]
        if platform not in PLATFORMS:
            raise web.HTTPBadRequest(text=f"platform must be one of {PLATFORMS}")
        backend: ResultsBackend = request.app["backend"]

        replay, replay_path = await backend.get_replay(platform)
        live, live_path = await backend.get_live(platform)
        cost_metrics = None
        if replay:
            cost_metrics = compute_replay_cost_metrics(replay, platform)
        payload = {
            "platform": platform,
            "data_source": backend.data_source,
            "replay_path": replay_path,
            "live_path": live_path,
            "replay": replay,
            "live": live,
            "cost_metrics": cost_metrics,
            "has_replay": replay is not None,
            "is_running": bool(live and live.get("status") == "running"),
        }
        if replay is None and live and live.get("summary"):
            payload["replay"] = live["summary"]
            payload["cost_metrics"] = compute_replay_cost_metrics(live["summary"], platform)
            payload["has_replay"] = True
        return web.json_response(payload)

    async def handle_hardware(_request: web.Request) -> web.Response:
        return web.json_response(node_pool_specs(load_config()))

    async def handle_migration(_request: web.Request) -> web.Response:
        cfg = load_config()
        return web.json_response({
            "model": cfg["model"],
            "benchmark": cfg["benchmark"],
            "gpu": platform_metadata(cfg, "gpu"),
            "tpu": platform_metadata(cfg, "tpu"),
            "hardware": node_pool_specs(cfg),
            "diff": migration_diff(cfg),
        })

    async def handle_compare(request: web.Request) -> web.Response:
        backend: ResultsBackend = request.app["backend"]
        comparison = await backend.get_comparison()
        tpu_replay, _ = await backend.get_replay("tpu")
        gpu_replay, _ = await backend.get_replay("gpu")
        return web.json_response({
            "data_source": backend.data_source,
            "comparison": comparison,
            "tpu": tpu_replay,
            "gpu": gpu_replay,
        })

    app.router.add_get("/", handle_index)
    app.router.add_get("/api/status", handle_status)
    app.router.add_get("/api/results/{platform}", handle_results)
    app.router.add_get("/api/hardware", handle_hardware)
    app.router.add_get("/api/migration", handle_migration)
    app.router.add_get("/api/compare", handle_compare)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark results UI — local files or GCS latest/{tpu,gpu}/"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--source",
        choices=["auto", "gcs", "local"],
        default="auto",
        help="auto: GCS if configured, else local files",
    )
    parser.add_argument("--gcs-bucket", help="Override GCS_RESULTS_BUCKET")
    parser.add_argument("--gcs-project", help="Override GCS_RESULTS_PROJECT")
    parser.add_argument("--gcs-prefix", default=None, help="Override GCS_RESULTS_PREFIX")
    args = parser.parse_args()

    settings: GcsSettings = gcs_settings(
        bucket=args.gcs_bucket,
        project=args.gcs_project,
        prefix=args.gcs_prefix,
    )
    gcs_store = GcsResultsStore(settings=settings) if settings.enabled else None
    backend = ResultsBackend(source=args.source, gcs=gcs_store)

    print(f"Results UI: http://127.0.0.1:{args.port}/")
    if backend.data_source == "gcs":
        print(f"  Data source: GCS gs://{settings.bucket}/{settings.prefix}/")
    else:
        print("  Data source: local results/{tpu,gpu}/run_01_replay.json")
    print("  Cloud Shell: Web Preview → port", args.port)

    web.run_app(build_app(backend), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
