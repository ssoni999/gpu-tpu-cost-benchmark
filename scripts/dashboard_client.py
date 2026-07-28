"""Push live replay metrics to the dashboard server and/or a JSON snapshot file."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import aiohttp


class LiveDashboard:
    """Best-effort live metrics sink for replay.py."""

    def __init__(
        self,
        dashboard_url: Optional[str] = None,
        live_json_path: Optional[Path] = None,
    ) -> None:
        self.dashboard_url = dashboard_url.rstrip("/") if dashboard_url else None
        self.live_json_path = live_json_path
        self.state: dict[str, Any] = {
            "status": "idle",
            "updated_at": time.time(),
            "progress": {"completed": 0, "total": 0, "ok": 0, "err": 0},
            "recent_requests": [],
            "rolling": {},
            "server_metrics": {},
        }

    async def publish(
        self,
        session: Optional[aiohttp.ClientSession],
        event: str,
        payload: dict[str, Any],
    ) -> None:
        self.state["updated_at"] = time.time()
        self.state["last_event"] = event
        if event == "run_start":
            self.state.update(payload)
            self.state["status"] = "running"
            self.state["summary"] = None
        elif event == "request_done":
            self._merge_request(payload)
        elif event == "server_sample":
            self.state["server_metrics"] = payload
        elif event == "run_complete":
            self.state["status"] = "complete"
            self.state["summary"] = payload
        elif event == "run_error":
            self.state["status"] = "error"
            self.state["error"] = payload.get("message", "unknown")

        self._write_file()
        if self.dashboard_url and session:
            await self._post(session, {"event": event, "payload": payload, "state": self.state})

    def _merge_request(self, payload: dict[str, Any]) -> None:
        prog = self.state.setdefault("progress", {})
        prog["completed"] = payload.get("completed", prog.get("completed", 0))
        prog["total"] = payload.get("total", prog.get("total", 0))
        prog["ok"] = payload.get("ok", prog.get("ok", 0))
        prog["err"] = payload.get("err", prog.get("err", 0))

        recent = self.state.setdefault("recent_requests", [])
        recent.append(payload.get("request", {}))
        self.state["recent_requests"] = recent[-20:]

        self.state["rolling"] = payload.get("rolling", self.state.get("rolling", {}))

    def _write_file(self) -> None:
        if not self.live_json_path:
            return
        self.live_json_path.parent.mkdir(parents=True, exist_ok=True)
        self.live_json_path.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")

    async def _post(
        self,
        session: aiohttp.ClientSession,
        body: dict[str, Any],
    ) -> None:
        if not self.dashboard_url:
            return
        event = body.get("event", "")
        try:
            async with session.post(
                f"{self.dashboard_url}/api/event",
                json=body,
                timeout=aiohttp.ClientTimeout(total=2),
            ) as resp:
                await resp.read()
            if event == "run_complete" and "payload" in body:
                platform = body.get("state", {}).get("platform", "tpu")
                if platform in ("auto", None):
                    platform = "tpu"
                async with session.post(
                    f"{self.dashboard_url}/api/replay?platform={platform}",
                    json=body["payload"],
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    await resp.read()
        except Exception:
            pass
