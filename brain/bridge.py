"""Bridge server: WebSocket endpoint the in-game PML mod talks to.

Thin relay between the mod (game page) and the Python brain:
  - receives demo recordings from the mod → writes to fixtures/
  - serves training status + best-recording export on request
  - (watch mode / live policy streaming lands in M5b)

Run: uv run python -m brain.bridge
Listens on ws://127.0.0.1:8766 by default.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import websockets

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "fixtures"


class Bridge:
    def __init__(self):
        self.clients: set = set()
        self.training_status: dict = {"best_lap_s": None, "sim_speed": None}
        self.demo_armed = False

    async def handler(self, ws):
        self.clients.add(ws)
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                await self._handle(ws, msg)
        finally:
            self.clients.discard(ws)

    async def _handle(self, ws, msg):
        t = msg.get("type")
        if t == "hello":
            await ws.send(
                json.dumps(
                    {
                        "type": "hello_ack",
                        "protocol": 1,
                        "envs": 0,
                        "build": "0.6.3",
                        "determinism": True,
                    }
                )
            )
        elif t == "arm_demo_capture":
            self.demo_armed = True
        elif t == "demo_recording":
            # {track, frames, recording}
            FIXTURES.mkdir(exist_ok=True)
            name = f"demo_{msg.get('track','track')}_{int(time.time())}.json"
            (FIXTURES / name).write_text(
                json.dumps(
                    {
                        "track": msg.get("track"),
                        "frames": msg.get("frames"),
                        "recording": msg.get("recording"),
                        "source": "in-game capture",
                        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                    indent=2,
                )
            )
            await ws.send(json.dumps({"type": "demo_saved", "file": name}))
        elif t == "export_best":
            # best recording the brain has produced (written by train.py on new best)
            best = REPO / "runs" / "best_recording.json"
            if best.exists():
                await ws.send(json.dumps({"type": "export_recording", **json.loads(best.read_text())}))
            else:
                await ws.send(json.dumps({"type": "error", "detail": "no best recording yet"}))
        elif t == "watch_policy":
            await ws.send(json.dumps({"type": "error", "detail": "watch mode lands in M5b"}))

    async def broadcast_status(self):
        if not self.clients:
            return
        msg = json.dumps({"type": "train_status", **self.training_status})
        await asyncio.gather(*(c.send(msg) for c in self.clients), return_exceptions=True)


async def main():
    bridge = Bridge()
    async with websockets.serve(bridge.handler, "127.0.0.1", 8766):
        print("[bridge] listening on ws://127.0.0.1:8766")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
