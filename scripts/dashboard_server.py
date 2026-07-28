#!/usr/bin/env python3
"""Live benchmark dashboard — SSE + REST for replay.py metrics."""

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
    "progress": {"completed": 0, "total": 0, "ok": 0, "err": 0},
    "recent_requests": [],
    "rolling": {},
    "server_metrics": {},
}

SUBSCRIBERS: set[asyncio.Queue[str]] = set()


def _merge_state(incoming: dict[str, Any]) -> None:
    STATE.update(incoming)
    STATE["updated_at"] = time.time()


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


async def handle_index(_request: web.Request) -> web.Response:
    html_path = Path(_request.app["html_path"])
    if not html_path.exists():
        raise web.HTTPNotFound(text=f"Missing dashboard HTML: {html_path}")
    return web.Response(text=html_path.read_text(encoding="utf-8"), content_type="text/html")


async def handle_state(_request: web.Request) -> web.Response:
    return web.json_response(STATE)


async def handle_event(request: web.Request) -> web.Response:
    body = await request.json()
    event = body.get("event", "update")
    if "state" in body and isinstance(body["state"], dict):
        _merge_state(body["state"])
    elif "payload" in body:
        payload = body["payload"]
        if event == "run_start":
            _merge_state({**payload, "status": "running"})
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
            STATE["status"] = "complete"
            STATE["summary"] = payload
        elif event == "run_error":
            STATE["status"] = "error"
            STATE["error"] = payload
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


def build_app(html_path: Path) -> web.Application:
    app = web.Application()
    app["html_path"] = str(html_path)
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/state", handle_state)
    app.router.add_get("/api/stream", handle_stream)
    app.router.add_post("/api/event", handle_event)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Live benchmark dashboard server.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (0.0.0.0 for Cloud Shell)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--html", default=str(DEFAULT_HTML))
    args = parser.parse_args()

    app = build_app(Path(args.html))
    print(f"Dashboard: http://{args.host}:{args.port}/")
    print("  Cloud Shell: Web Preview → port", args.port)
    print("  Local Mac:   gcloud cloud-shell ssh --authorize-session \\")
    print(f"                 --ssh-flag='-L {args.port}:localhost:{args.port}'")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
