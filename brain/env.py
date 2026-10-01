"""PolyTrack gym-style environment driving the headless Node sim host.

One `PolyTrackEnv` = one car in one sim worker, stepped at the control rate
(action repeat R of 1 ms sim frames). The heavy lifting (physics, protocol,
live-controls injection) is in node/sim_host.mjs; this class talks to a
long-running `node/scripts/env_server_node.mjs` child over newline-delimited
JSON on stdio (simple, local, zero-dep) — the WebSocket bridge is for the
in-game path, not the Python↔Node training path.

Observation = decoded 227-byte car state + track-geometry features.
Action = 4 binary buttons (up/down/left/right), applied for R frames.
"""

from __future__ import annotations

import json
import math
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty

from .tracklib import load_track_geom, TrackGeom

REPO = Path(__file__).resolve().parent.parent

# VO struct offsets within the 227-byte state (after the 4-byte carId header,
# which the node side strips before sending us the struct).
# Verified against module 3899 in the main bundle.


@dataclass
class CarState:
    frames: int
    speed_kmh: float
    has_started: bool
    finish_frames: int | None
    next_checkpoint_index: int
    has_checkpoint_respawn: bool
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    collision_impulses: list[float]
    wheel_contact: list[bool]  # 4
    steering: float
    controls: dict[str, bool]


def decode_car_state(b: bytes) -> CarState:
    import struct

    dv = memoryview(b)
    frames = b[0] | (b[1] << 8) | (b[2] << 16)
    (speed,) = struct.unpack_from("<f", b, 3)
    flags = b[7]
    has_started = bool(flags & 1)
    has_finish = bool(flags & 2)
    has_respawn = bool(flags & 4)
    wheel_contact = [bool(flags & (8 << k)) for k in range(4)]
    off = 8
    finish_frames = None
    if has_finish:
        finish_frames = b[off] | (b[off + 1] << 8) | (b[off + 2] << 16)
        off += 3
    (next_cp,) = struct.unpack_from("<H", b, off)
    off += 2
    px, py, pz = struct.unpack_from("<3f", b, off)
    off += 12
    qx, qy, qz, qw = struct.unpack_from("<4f", b, off)
    off += 16
    n_imp = b[off]
    off += 1
    impulses = list(struct.unpack_from(f"<{n_imp}f", b, off))
    off += 4 * n_imp
    # wheel contacts present flags already read; skip 24 bytes per present wheel
    for w in wheel_contact:
        if w:
            off += 24
    # suspension length/velocity, delta rotation, skid (4×f32 each) then steering
    off += 16 * 4
    (steering,) = struct.unpack_from("<f", b, off)
    off += 4
    ctrl_byte = b[off]
    controls = {
        "up": bool(ctrl_byte & 1),
        "right": bool(ctrl_byte & 2),
        "down": bool(ctrl_byte & 4),
        "left": bool(ctrl_byte & 8),
        "reset": bool(ctrl_byte & 16),
    }
    return CarState(
        frames=frames,
        speed_kmh=speed,
        has_started=has_started,
        finish_frames=finish_frames,
        next_checkpoint_index=next_cp,
        has_checkpoint_respawn=has_respawn,
        position=(px, py, pz),
        quaternion=(qx, qy, qz, qw),
        collision_impulses=impulses,
        wheel_contact=wheel_contact,
        steering=steering,
        controls=controls,
    )


class _NodeProc:
    """Persistent env_server_node child; NDJSON request/response on stdio."""

    def __init__(self):
        self.proc = subprocess.Popen(
            ["node", "scripts/env_server_node.mjs"],
            cwd=REPO,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._responses: "Queue[dict]" = Queue()
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self):
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._responses.put(json.loads(line))
            except json.JSONDecodeError:
                pass

    def call(self, msg: dict, timeout: float = 60.0) -> dict:
        with self._lock:
            assert self.proc.stdin is not None
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
            try:
                return self._responses.get(timeout=timeout)
            except Empty:
                raise TimeoutError(f"env_server_node timeout on {msg.get('cmd')}")

    def close(self):
        self.proc.terminate()


class PolyTrackEnv:
    """gym-style env over one headless sim worker."""

    def __init__(
        self,
        track: str = "summer1",
        control_hz: float = 50.0,
        max_episode_frames: int = 90_000,
        terminate_on_respawn: bool = True,
        reward: dict | None = None,
    ):
        self.track_name = track
        self.geom: TrackGeom = load_track_geom(track)
        self.control_hz = control_hz
        self.R = max(1, round(1000 / control_hz))  # action repeat in sim frames
        self.max_episode_frames = max_episode_frames
        self.terminate_on_respawn = terminate_on_respawn
        self.rw = {
            "progress": 1.0,
            "checkpoint": 12.0,
            "finish": 200.0,
            "time_penalty": 0.01,
            "collision": 0.002,
            "respawn": 5.0,
            **(reward or {}),
        }
        self._node = _NodeProc()
        self._env_id = None
        self._prev_s = 0.0
        self._last_state: CarState | None = None

    # -- gym API ---------------------------------------------------------------

    def reset(self) -> dict:
        resp = self._node.call(
            {
                "cmd": "reset",
                "env_id": self._env_id,
                "track": self.track_name,
                "max_frames": self.max_episode_frames,
            },
            timeout=120,
        )
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        self._env_id = resp["env_id"]
        self._prev_s = 0.0
        self._last_state = None
        return self._obs_from_state(decode_car_state(bytes.fromhex(resp["state"])))

    def step(self, action: tuple[int, int, int, int]) -> tuple[dict, float, bool, dict]:
        """action = (up, down, left, right) as 0/1. Returns (obs, reward, done, info)."""
        up, down, left, right = action
        resp = self._node.call(
            {
                "cmd": "step",
                "env_id": self._env_id,
                "frames": self.R,
                "controls": {"up": bool(up), "down": bool(down), "left": bool(left), "right": bool(right)},
            },
            timeout=120,
        )
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        st = decode_car_state(bytes.fromhex(resp["state"]))
        self._last_state = st
        obs = self._obs_from_state(st)
        reward, done, info = self._reward_done(st)
        return obs, reward, done, info

    def close(self):
        self._node.close()

    # -- internals -------------------------------------------------------------

    def _obs_from_state(self, st: CarState) -> dict:
        s = self.geom.progress(st.position)
        cp_ahead = self.geom.lookahead_point(s, 25.0)
        cp_far = self.geom.lookahead_point(s, 60.0)
        return {
            "speed_kmh": st.speed_kmh,
            "position": st.position,
            "quaternion": st.quaternion,
            "steering": st.steering,
            "wheel_contact": st.wheel_contact,
            "progress_s": s,
            "progress_frac": (s / self.geom.total_len) if self.geom.total_len > 0 else 0.0,
            "to_cp_25m": _vec_sub(cp_ahead, st.position),
            "to_cp_60m": _vec_sub(cp_far, st.position),
            "next_checkpoint_index": st.next_checkpoint_index,
            "frames": st.frames,
            "finish_frames": st.finish_frames,
        }

    def _reward_done(self, st: CarState) -> tuple[float, bool, dict]:
        s = self.geom.progress(st.position)
        d_progress = s - self._prev_s
        self._prev_s = s

        r = self.rw["progress"] * d_progress - self.rw["time_penalty"] * (self.R / 1000.0)

        done = False
        reason = None
        # checkpoint bonus: next_checkpoint_index increments on pass
        # (worker tracks it; we reward the delta)
        # collision penalty
        if st.collision_impulses:
            r -= self.rw["collision"] * min(sum(st.collision_impulses), 4000.0)
        if st.finish_frames is not None:
            r += self.rw["finish"] - 0.001 * st.finish_frames
            done = True
            reason = "finish"
        elif st.frames >= self.max_episode_frames:
            done = True
            reason = "timeout"

        info = {"progress_s": s, "reason": reason, "frames": st.frames}
        return r, done, info


def _vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
