"""Demo extraction: recording → (state, action) transitions via headless re-sim.

The human recording encodes button toggles at 1 kHz; the RL acts at 50 Hz
(every R=20 frames). For each control tick t (frame index k = t·R):
  - state_t   = decoded car state at frame k (from the re-sim dump)
  - action_t  = majority button state over frames [k, k+R) of the recording
  - reward/etc. come from the env's reward function on the next state.

Returns transitions ready for BC pretraining or demo-buffer insertion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .codec import Recording
from .env import _NodeProc, decode_car_state

REPO = Path(__file__).resolve().parent.parent


@dataclass
class Demo:
    track: str
    frames: int
    states: list[dict]  # decoded CarState per control tick
    actions: list[tuple[int, int, int, int]]  # (up, down, left, right) per control tick


def extract_demo(track: str, recording_str: str, control_hz: float = 50.0, max_frames: int = 120_000) -> Demo:
    R = max(1, round(1000 / control_hz))
    node = _NodeProc()
    try:
        resp = node.call(
            {
                "cmd": "run_recording",
                "track": track,
                "recording": recording_str,
                "max_frames": max_frames,
                "sample_every": R,
            },
            timeout=180,
        )
        if resp.get("error"):
            raise RuntimeError(resp["error"])
    finally:
        node.close()

    rec = Recording.deserialize(recording_str)
    total = min(resp["frames"], rec.num_frames + 1000)
    n_ticks = total // R

    # button state per frame, then majority per tick
    dense = rec.to_dense(total)
    states = [decode_car_state(bytes.fromhex(h)) for h in resp["samples"][:n_ticks]]
    actions = []
    for t in range(n_ticks):
        window = dense[t * R : (t + 1) * R]
        if not window:
            break
        n = len(window)
        up = sum(1 for w in window if w["up"]) / n
        down = sum(1 for w in window if w["down"]) / n
        left = sum(1 for w in window if w["left"]) / n
        right = sum(1 for w in window if w["right"]) / n
        actions.append((int(up >= 0.5), int(down >= 0.5), int(left >= 0.5), int(right >= 0.5)))

    # align: one state per tick, one action per tick (state_k → action_k)
    n = min(len(states), len(actions))
    return Demo(track=track, frames=resp["frames"], states=states[:n], actions=actions[:n])


def load_fixtures() -> list[dict]:
    """All demo fixtures. Handles both the single-lap format (human_summer1.json)
    and the leaderboard format (leaderboard_*.json, a list with `track` implied
    by filename). Each returned dict has at least {track, recording, frames?}."""
    fx = REPO / "fixtures"
    out = []
    for f in sorted(fx.glob("*.json")):
        data = json.loads(f.read_text())
        # leaderboard files: track comes from the filename `leaderboard_<track>.json`
        if f.name.startswith("leaderboard_"):
            track = f.stem[len("leaderboard_"):]
            for entry in data:
                out.append({"track": track, **entry})
        else:
            out.extend(data)
    return out


if __name__ == "__main__":
    import sys

    track = sys.argv[1] if len(sys.argv) > 1 else "summer1"
    fx = [f for f in load_fixtures() if f.get("track", track) == track]
    if not fx:
        raise SystemExit(f"no fixture for {track}")
    demo = extract_demo(track, fx[0]["recording"])
    print(f"track={demo.track} frames={demo.frames} ticks={len(demo.states)} finished?")
    gas = sum(1 for a in demo.actions if a[0])
    print(f"throttle ticks: {gas}/{len(demo.actions)}")
    print("first 10 actions:", demo.actions[:10])
    print("state@0 speed:", demo.states[0].speed_kmh, "pos:", demo.states[0].position)
