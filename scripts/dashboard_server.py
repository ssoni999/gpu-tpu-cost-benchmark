#!/usr/bin/env python3
"""Live benchmark dashboard — serves replay JSON + SSE updates."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from aiohttp import web

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HTML = ROOT / "dashboard" / "index.html"

STATE: dict[str, Any] = {
    "status": "idle",
    "updated_at": time.time(),
    "platform": "tpu",
    "progress": {"completed": 0, "total": 0, "ok": 0, "err": 0},
    "recent_requests": [],
    "rolling": {},
    "server_metrics": {},
    "summary": None,
}

SUBSCRIBERS: set[asyncio.Queue[str]] = set()
FILE_MTIMES: dict[str, float] = {}


def _replay_path(platform: str) -> Path:
    return ROOT / "results" / platform / "run_01_replay.json"


def _live_path(platform: str) -> Path:
    return ROOT / "results" / platform / "live.json"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _apply_summary(summary: dict[str, Any], platform: str | None = None) -> None:
    STATE["summary"] = summary
    STATE["status"] = "complete"
    STATE["model"] = summary.get("model")
    STATE["infrastructure"] = summary.get("infrastructure")
    STATE["target"] = summary.get("target")
    STATE["trace"] = summary.get("trace")
    if platform:
        STATE["platform"] = platform
    STATE["progress"] = {
        "completed": summary.get("successful_requests", 0) + summary.get("failed_requests", 0),
        "total": summary.get("successful_requests", 0) + summary.get("failed_requests", 0),
        "ok": summary.get("successful_requests", 0),
        "err": summary.get("failed_requests", 0),
    }
    if summary.get("server_metrics"):
        STATE["server_metrics"] = summary["server_metrics"]
    STATE["updated_at"] = time.time()


def _apply_live(live: dict[str, Any]) -> None:
    STATE["updated_at"] = time.time()
    for key in ("status", "platform", "model", "infrastructure", "target", "progress",
                "rolling", "server_metrics", "recent_requests", "summary"):
        if key in live and live[key] is not None:
            STATE[key] = live[key]


async def _broadcast() -> None:
    payload = json.dumps(STATE)
    dead: list[asyncio.Queue[str]] = []
    for q in SUBSCRIBERS:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        SUBSCRIBERS.discard(q)


async def _poll_result_files(interval: float = 2.0) -> None:
    """Reload run_01_replay.json / live.json when replay writes them."""
    while True:
        for platform in ("tpu", "gpu"):
            for label, path in (("replay", _replay_path(platform)), ("live", _live_path(platform))):
                if not path.exists():
                    continue
                mtime = path.stat().st_mtime
                key = f"{label}:{platform}"
                if FILE_MTIMES.get(key) == mtime:
                    continue
                FILE_MTIMES[key] = mtime
                data = _load_json(path)
                if not data:
                    continue
                if label == "replay":
                    _apply_summary(data, platform)
                else:
                    _apply_live(data)
                await _broadcast()
        await asyncio.sleep(interval)


async def handle_index(_request: web.Request) -> web.Response:
    html_path = Path(_request.app["html_path"])
    if not html_path.exists():
        raise web.HTTPNotFound(text=f"Missing dashboard HTML: {html_path}")
    return web.Response(text=html_path.read_text(encoding="utf-8"), content_type="text/html")


async def handle_state(request: web.Request) -> web.Response:
    platform = request.query.get("platform", STATE.get("platform", "tpu"))
    replay = _load_json(_replay_path(platform))
    if replay:
        _apply_summary(replay, platform)
    return web.json_response(STATE)


async def handle_replay_get(request: web.Request) -> web.Response:
    platform = request.match_info["platform"]
    replay = _load_json(_replay_path(platform))
    if replay is None:
        raise web.HTTPNotFound(text=f"No results at results/{platform}/run_01_replay.json")
    _apply_summary(replay, platform)
    await _broadcast()
    return web.json_response(replay)


async def handle_replay_post(request: web.Request) -> web.Response:
    summary = await request.json()
    platform = request.query.get("platform") or summary.get("platform") or "tpu"
    _apply_summary(summary, platform)
    out = _replay_path(platform)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    await _broadcast()
    return web.json_response({"ok": True})


async def handle_event(request: web.Request) -> web.Response:
    body = await request.json()
    event = body.get("event", "update")
    if "state" in body and isinstance(body["state"], dict):
        incoming = body["state"]
        STATE.update(incoming)
        STATE["updated_at"] = time.time()
    elif "payload" in body:
        payload = body["payload"]
        if event == "run_start":
            STATE.update(payload)
            STATE["status"] = "running"
            STATE["summary"] = None
        elif event == "request_done":
            STATE.setdefault("progress", {}).update({
                k: payload[k] for k in ("completed", "total", "ok", "err") if k in payload
            })
            if "request" in payload:
                recent = STATE.setdefault("recent_requests", [])
                recent.append(payload["request"])
                STATE["recent_requests"] = recent[-20:]
            if "rolling" in payload:
                STATE["rolling"] = payload["rolling"]
        elif event == "server_sample":
            STATE["server_metrics"] = payload
        elif event == "run_complete":
            _apply_summary(payload, STATE.get("platform"))
        elif event == "run_error":
            STATE["status"] = "error"
            STATE["error"] = payload
        STATE["updated_at"] = time.time()
    await _broadcast()
    return web.json_response({"ok": True})


async def handle_stream(request: web.Request) -> web.StreamResponse:
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=64)
    SUBSCRIBERS.add(queue)
    resp = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await resp.prepare(request)
    await queue.put(json.dumps(STATE))
    try:
        while True:
            msg = await queue.get()
            await resp.write(f"data: {msg}\n\n".encode())
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        pass
    finally:
        SUBSCRIBERS.discard(queue)
    return resp


async def on_startup(app: web.Application) -> None:
    app["poller"] = asyncio.create_task(_poll_result_files())


async def on_cleanup(app: web.Application) -> None:
    app["poller"].cancel()
    try:
        await app["poller"]
    except asyncio.CancelledError:
        pass


def build_app(html_path: Path) -> web.Application:
    app = web.Application()
    app["html_path"] = str(html_path)
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/state", handle_state)
    app.router.add_get("/api/replay/{platform}", handle_replay_get)
    app.router.add_post("/api/replay", handle_replay_post)
    app.router.add_get("/api/stream", handle_stream)
    app.router.add_post("/api/event", handle_event)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Live benchmark dashboard server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--html", default=str(DEFAULT_HTML))
    args = parser.parse_args()

    app = build_app(Path(args.html))
    print(f"Dashboard: http://{args.host}:{args.port}/")
    print("  Reads: results/{{tpu|gpu}}/run_01_replay.json (auto-refresh every 2s)")
    print("  Cloud Shell: Web Preview → port", args.port)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
