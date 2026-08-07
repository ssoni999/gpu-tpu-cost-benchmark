#!/usr/bin/env python3
"""Serve the benchmark results UI — reads local files or GCS latest/*.json (port 8787)."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from aiohttp import web

from config import (
    default_workload,
    legacy_replay_path,
    load_config,
    migration_diff,
    node_pool_specs,
    platform_metadata,
    replay_live_path,
    replay_matches_tier,
    replay_result_path,
    workload_tier_ids,
    workloads_catalog,
)
from cost_metrics import compute_replay_cost_metrics
from gcs_results import (
    GcsResultsStore,
    GcsSettings,
    gcs_settings,
    latest_replay_object,
    manifest_tier_entry,
    object_implies_tier,
)
from workload_projection import resolve_tier_replay

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


def _local_replay_candidates(platform: str, tier: str) -> list[Path]:
    paths = [replay_result_path(platform, tier)]
    if tier == "medium":
        paths.append(legacy_replay_path(platform))
    return paths


def _local_live_candidates(platform: str, tier: str) -> list[Path]:
    paths = [replay_live_path(platform, tier)]
    if tier == "medium":
        paths.append(RESULTS / platform / "live.json")
    return paths


def _path_trusts_tier(path: Path, tier: str, cfg: dict[str, Any] | None = None) -> bool:
    """True when file lives in a tier-specific directory (results/.../small/...)."""
    from config import workload_tier_ids

    cfg = cfg or load_config()
    return path.parent.name == tier and tier in workload_tier_ids(cfg)


def _collect_replay_file(
    path: Path,
    path_tier: str,
    found: dict[str, dict[str, Any]],
    paths: dict[str, str],
    cfg: dict[str, Any],
) -> None:
    """Index replay by path tier (small/medium/high), not JSON workload_tier."""
    replay = _read_json(path)
    if replay is None:
        return
    trusted = _path_trusts_tier(path, path_tier, cfg)
    if not trusted and not replay_matches_tier(replay, path_tier, cfg):
        return
    # First path match wins for this tier (tier-specific before legacy).
    if path_tier not in found:
        found[path_tier] = replay
        paths[path_tier] = str(path.relative_to(ROOT))


def _local_replay_mtime(platform: str, tier: str) -> float:
    best = 0.0
    for path in _local_replay_candidates(platform, tier):
        if path.exists():
            best = max(best, path.stat().st_mtime)
    return best


class ResultsBackend:
    def __init__(self, source: str, gcs: GcsResultsStore | None) -> None:
        self.source = source
        self.gcs = gcs
        self._tier_sources: dict[str, dict[str, str]] = {}

    @property
    def data_source(self) -> str:
        if self.source == "gcs":
            return "gcs"
        if self.source == "local":
            return "local"
        if self.gcs and self.gcs.enabled:
            return "auto (gcs+local)"
        return "local"

    def tier_sources(self, platform: str) -> dict[str, str]:
        return dict(self._tier_sources.get(platform, {}))

    async def _gcs_read(self, fn):
        if not self.gcs or not self.gcs.enabled:
            return None
        try:
            return await asyncio.to_thread(fn)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: GCS read failed: {exc}", flush=True)
            return None

    async def _load_local_replays(
        self, platform: str,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        cfg = load_config()
        found: dict[str, dict[str, Any]] = {}
        paths: dict[str, str] = {}
        for tier in workload_tier_ids():
            for path in _local_replay_candidates(platform, tier):
                _collect_replay_file(path, tier, found, paths, cfg)
        return found, paths

    async def _load_gcs_replays(
        self, platform: str,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        cfg = load_config()
        found: dict[str, dict[str, Any]] = {}
        paths: dict[str, str] = {}
        if not self.gcs or not self.gcs.enabled:
            return found, paths
        manifest = await self._gcs_read(self.gcs.read_manifest) or {}  # type: ignore[union-attr]
        settings = self.gcs.settings
        for tier in workload_tier_ids():
            result = await self._gcs_read(
                lambda p=platform, t=tier: self.gcs.read_replay_object(p, t)  # type: ignore[union-attr]
            )
            if not result:
                continue
            replay, obj = result
            if replay is None or tier in found:
                continue
            trusted = bool(obj and object_implies_tier(obj, tier, settings))
            if trusted or replay_matches_tier(replay, tier, cfg):
                found[tier] = replay
                if obj:
                    paths[tier] = f"gs://{settings.bucket}/{obj}"
                else:
                    paths[tier] = f"gs://{settings.bucket}/{latest_replay_object(platform, settings, tier=tier)}"
        return found, paths

    async def _merge_replay_sources(
        self, platform: str,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, str]]:
        """Load replays; in auto mode pick newest per tier (GCS wins ties)."""
        cfg = load_config()
        replays: dict[str, dict[str, Any]] = {}
        paths: dict[str, str] = {}
        sources: dict[str, str] = {}

        local_replays: dict[str, dict[str, Any]] = {}
        local_paths: dict[str, str] = {}
        gcs_replays: dict[str, dict[str, Any]] = {}
        gcs_paths: dict[str, str] = {}

        if self.source in ("local", "auto"):
            local_replays, local_paths = await self._load_local_replays(platform)
        if self.gcs and self.gcs.enabled and self.source in ("gcs", "auto"):
            gcs_replays, gcs_paths = await self._load_gcs_replays(platform)

        if self.source == "gcs":
            for tier, data in gcs_replays.items():
                replays[tier] = data
                paths[tier] = gcs_paths[tier]
                sources[tier] = "gcs"
        elif self.source == "local":
            for tier, data in local_replays.items():
                replays[tier] = data
                paths[tier] = local_paths[tier]
                sources[tier] = "local"
        else:
            for tier in workload_tier_ids():
                local_r = local_replays.get(tier)
                gcs_r = gcs_replays.get(tier)
                if local_r and gcs_r:
                    local_ts = _local_replay_mtime(platform, tier)
                    gcs_ts = self.gcs.tier_upload_epoch(platform, tier) if self.gcs else 0.0
                    if gcs_ts >= local_ts:
                        replays[tier] = gcs_r
                        paths[tier] = gcs_paths[tier]
                        sources[tier] = "gcs"
                    else:
                        replays[tier] = local_r
                        paths[tier] = local_paths[tier]
                        sources[tier] = "local"
                elif gcs_r:
                    replays[tier] = gcs_r
                    paths[tier] = gcs_paths[tier]
                    sources[tier] = "gcs"
                elif local_r:
                    replays[tier] = local_r
                    paths[tier] = local_paths[tier]
                    sources[tier] = "local"

        self._tier_sources[platform] = sources
        return replays, paths, sources

    async def get_replay(
        self,
        platform: str,
        tier: str | None = None,
        *,
        allow_projection: bool = False,
    ) -> tuple[dict[str, Any] | None, str, bool, str]:
        """Return replay for tier. Projection is opt-in (disabled by default)."""
        cfg = load_config()
        tier = tier or default_workload(cfg)

        # GCS-only: fetch the exact tier object every time (latest/{tier}/{platform}/...).
        if self.source == "gcs" and self.gcs and self.gcs.enabled:
            result = await self._gcs_read(
                lambda p=platform, t=tier: self.gcs.read_replay_object(p, t)  # type: ignore[union-attr]
            )
            if result:
                replay, obj = result
                if replay is not None:
                    settings = self.gcs.settings
                    trusted = bool(obj and object_implies_tier(obj, tier, settings))
                    if trusted or replay_matches_tier(replay, tier, cfg):
                        out = dict(replay)
                        out["projected"] = False
                        out.setdefault("workload_tier", tier)
                        path = f"gs://{settings.bucket}/{obj}" if obj else ""
                        self._tier_sources.setdefault(platform, {})[tier] = "gcs"
                        return out, path, False, tier
            return None, str(replay_result_path(platform, tier).relative_to(ROOT)), False, tier

        replays, replay_paths, _sources = await self._merge_replay_sources(platform)

        if tier in replays:
            replay = dict(replays[tier])
            replay["projected"] = False
            replay.setdefault("workload_tier", tier)
            path = replay_paths.get(tier) or str(replay_result_path(platform, tier).relative_to(ROOT))
            return replay, path, False, tier

        # Project from nearest measured tier when this tier has not been replayed yet.
        if allow_projection and replays:
            replay, source_tier, projected = resolve_tier_replay(replays, tier, platform, cfg)
            if replay is not None:
                path = f"(projected from {source_tier})" if projected else replay_paths.get(tier, "")
                return replay, path, projected, tier

        return None, str(replay_result_path(platform, tier).relative_to(ROOT)), False, tier

    async def list_measured_tiers(self, platform: str) -> list[str]:
        if self.source == "gcs" and self.gcs and self.gcs.enabled:
            cfg = load_config()
            found: list[str] = []
            for tier in workload_tier_ids():
                result = await self._gcs_read(
                    lambda p=platform, t=tier: self.gcs.read_replay_object(p, t)  # type: ignore[union-attr]
                )
                if not result or result[0] is None:
                    continue
                replay, obj = result
                trusted = bool(obj and object_implies_tier(obj, tier, self.gcs.settings))
                if trusted or replay_matches_tier(replay, tier, cfg):
                    found.append(tier)
            return sorted(found)
        replays, _, _ = await self._merge_replay_sources(platform)
        return sorted(replays.keys())

    async def get_live(self, platform: str, tier: str | None = None) -> tuple[dict[str, Any] | None, str]:
        tier = tier or default_workload()
        if self.gcs and self.gcs.enabled and self.source in ("gcs", "auto"):
            live = await self._gcs_read(
                lambda p=platform, t=tier: self.gcs.read_live(p, t)  # type: ignore[union-attr]
            )
            if live is not None or self.source == "gcs":
                obj = f"gs://{self.gcs.settings.bucket}/{self.gcs.settings.prefix}/{tier}/{platform}/live.json"
                return live, obj

        for path in _local_live_candidates(platform, tier):
            live = _read_json(path)
            if live is not None:
                return live, str(path.relative_to(ROOT))
        fallback = _local_live_candidates(platform, tier)[0]
        return None, str(fallback.relative_to(ROOT))

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


def _render_index() -> str:
    """Serve index.html with hardware specs embedded from benchmark_config.yaml."""
    index = FRONTEND / "index.html"
    text = index.read_text(encoding="utf-8")
    specs = json.dumps({
        "hardware": node_pool_specs(load_config()),
        "workloads": workloads_catalog(load_config()),
        "default_workload": default_workload(load_config()),
    })
    tag = f'<script id="embedded-hardware-specs" type="application/json">{specs}</script>'
    marker = "<!-- hardware-specs -->"
    if marker in text:
        return text.replace(marker, tag, 1)
    return text.replace("</head>", f"  {tag}\n</head>", 1)


def build_app(backend: ResultsBackend) -> web.Application:
    app = web.Application()
    app["backend"] = backend

    async def handle_index(_request: web.Request) -> web.Response:
        if not (FRONTEND / "index.html").exists():
            raise web.HTTPNotFound(text="frontend/index.html missing")
        return web.Response(text=_render_index(), content_type="text/html")

    async def handle_status(_request: web.Request) -> web.Response:
        backend: ResultsBackend = _request.app["backend"]
        manifest = await backend.get_manifest()
        settings = backend.gcs.settings if backend.gcs else None
        gcs_tiers: dict[str, dict[str, Any]] = {}
        if backend.source == "gcs" and backend.gcs and backend.gcs.enabled:
            for platform in PLATFORMS:
                gcs_tiers[platform] = {}
                for tier in workload_tier_ids():
                    result = await backend._gcs_read(
                        lambda p=platform, t=tier: backend.gcs.read_replay_object(p, t)  # type: ignore[union-attr]
                    )
                    if result and result[0]:
                        replay, obj = result
                        gcs_tiers[platform][tier] = {
                            "object": obj,
                            "output_tokens_per_second": replay.get("output_tokens_per_second"),
                            "requests": (replay.get("successful_requests") or 0)
                            + (replay.get("failed_requests") or 0),
                        }
        return web.json_response({
            "data_source": backend.data_source,
            "results_source": backend.source,
            "gcs_bucket": settings.bucket if settings else None,
            "gcs_project": settings.project if settings else None,
            "gcs_prefix": settings.prefix if settings else None,
            "manifest": manifest,
            "gcs_loaded": gcs_tiers,
            "tier_sources": {
                "tpu": backend.tier_sources("tpu"),
                "gpu": backend.tier_sources("gpu"),
            },
        })

    async def handle_workloads(_request: web.Request) -> web.Response:
        cfg = load_config()
        return web.json_response({
            "default": default_workload(cfg),
            "tiers": workloads_catalog(cfg),
        })

    async def handle_results(request: web.Request) -> web.Response:
        platform = request.match_info["platform"]
        if platform not in PLATFORMS:
            raise web.HTTPBadRequest(text=f"platform must be one of {PLATFORMS}")
        backend: ResultsBackend = request.app["backend"]
        tier = request.rel_url.query.get("workload") or request.rel_url.query.get("tier")

        allow_projection = (
            backend.source != "gcs"
            and request.rel_url.query.get("project") in ("1", "true", "yes")
        )
        replay, replay_path, projected, resolved_tier = await backend.get_replay(
            platform, tier, allow_projection=allow_projection,
        )
        live, live_path = await backend.get_live(platform, resolved_tier)
        cost_metrics = None
        if replay:
            cost_metrics = replay.get("cost_metrics") or compute_replay_cost_metrics(replay, platform)
        is_running = bool(live and live.get("status") == "running")
        resolved = resolved_tier or tier or default_workload()
        payload = {
            "platform": platform,
            "requested_tier": tier or default_workload(),
            "workload_tier": resolved,
            "projected": projected,
            "tier_missing": replay is None and not is_running,
            "available_tiers": sorted(await backend.list_measured_tiers(platform)),
            "data_source": backend.data_source,
            "results_source": backend.source,
            "replay_source": backend.tier_sources(platform).get(resolved),
            "replay_path": replay_path,
            "live_path": live_path,
            "replay": replay,
            "live": live if is_running else None,
            "cost_metrics": cost_metrics,
            "has_replay": replay is not None,
            "is_running": is_running,
        }
        # Never substitute live.json summary for a missing tier — it caused identical metrics
        # across Simple/Medium/Complex when only one replay existed.
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
        tier = request.rel_url.query.get("workload") or request.rel_url.query.get("tier")
        comparison = await backend.get_comparison()
        tpu_replay, _, _, tpu_tier = await backend.get_replay("tpu", tier)
        gpu_replay, _, _, gpu_tier = await backend.get_replay("gpu", tier)
        return web.json_response({
            "data_source": backend.data_source,
            "workload_tier": tpu_tier,
            "comparison": comparison,
            "tpu": tpu_replay,
            "gpu": gpu_replay,
        })

    async def handle_gcs_inspect(_request: web.Request) -> web.Response:
        backend: ResultsBackend = _request.app["backend"]
        if not backend.gcs or not backend.gcs.enabled:
            return web.json_response({"error": "GCS not configured"}, status=503)
        out: dict[str, Any] = {"bucket": backend.gcs.settings.bucket, "tiers": {}}
        for platform in PLATFORMS:
            out["tiers"][platform] = {}
            for tier in workload_tier_ids():
                info = await backend._gcs_read(
                    lambda p=platform, t=tier: backend.gcs.inspect_tier(p, t)  # type: ignore[union-attr]
                )
                out["tiers"][platform][tier] = info
        return web.json_response(out)

    app.router.add_get("/", handle_index)
    app.router.add_get("/api/status", handle_status)
    app.router.add_get("/api/gcs/inspect", handle_gcs_inspect)
    app.router.add_get("/api/workloads", handle_workloads)
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
        default="gcs",
        help="gcs: GCS only (default); auto: merge GCS + local; local: results/ only",
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
        gcs_store = GcsResultsStore(settings=settings, cache_ttl_s=0.0) if settings.enabled else None
    backend = ResultsBackend(source=args.source, gcs=gcs_store)

    print(f"Results UI: http://127.0.0.1:{args.port}/")
    print(f"  Results source: {args.source}")
    if backend.gcs and backend.gcs.enabled:
        print(f"  GCS: gs://{settings.bucket}/{settings.prefix}/")
    if args.source == "local":
        print("  Reading: local results/ only (set RESULTS_SOURCE=auto or gcs to use GCS)")
    elif args.source == "gcs":
        print("  Reading: GCS only")
    else:
        print("  Reading: GCS + local (newest per tier wins)")
    print("  Cloud Shell: Web Preview → port", args.port)

    web.run_app(build_app(backend), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
