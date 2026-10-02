"""Vectorized PolyTrack env: N cars across the pooled headless workers.

Same NDJSON child as PolyTrackEnv, but steps/reset are batched (step_all /
reset_all), so one round-trip advances the whole fleet. This is the training
path; PolyTrackEnv stays for single-env eval/debug.
"""

from __future__ import annotations

import numpy as np

from .env import _NodeProc, decode_car_state, CarState
from .features import featurize
from .tracklib import load_track_geom


class VecPolyTrackEnv:
    def __init__(
        self,
        n_envs: int = 8,
        track: str = "summer1",
        control_hz: float = 50.0,
        max_episode_frames: int = 60_000,
        reward: dict | None = None,
    ):
        self.n = n_envs
        self.track_name = track
        self.geom = load_track_geom(track)
        self.R = max(1, round(1000 / control_hz))
        self.max_episode_frames = max_episode_frames
        self.rw = {
            "progress": 0.1,        # ~m per step → keep Q targets O(1)
            "checkpoint": 1.0,
            "finish": 10.0,
            "time_penalty": 0.002,
            "collision": 0.0005,
            "gate_progress": 0.05,  # per meter of distance-to-gate closed
            **(reward or {}),
        }
        self._node = _NodeProc()
        self._env_ids: list[int | None] = [None] * n_envs
        self._prev_s = np.zeros(n_envs, dtype=np.float64)
        self._prev_cp = np.zeros(n_envs, dtype=np.int32)
        self._prev_gate_dist: list[float | None] = [None] * n_envs
        self._states: list[CarState | None] = [None] * n_envs

    def reset_all(self) -> np.ndarray:
        resp = self._node.call(
            {
                "cmd": "reset_all",
                "envs": [
                    {"env_id": i, "track": self.track_name, "max_frames": self.max_episode_frames}
                    for i in range(self.n)
                ],
            },
            timeout=180,
        )
        feats = []
        for r in resp["results"]:
            i = r["env_id"]
            st = decode_car_state(bytes.fromhex(r["state"]))
            self._states[i] = st
            self._prev_s[i] = 0.0
            self._prev_cp[i] = st.next_checkpoint_index
            self._prev_gate_dist[i] = None
            feats.append(featurize(st, self.geom))
        return np.asarray(feats, dtype=np.float32)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """actions: (n,4) int 0/1 → (obs, reward, done, infos)"""
        resp = self._node.call(
            {
                "cmd": "step_all",
                "frames": self.R,
                "actions": [
                    {
                        "env_id": i,
                        "controls": {
                            "up": bool(actions[i, 0]),
                            "down": bool(actions[i, 1]),
                            "left": bool(actions[i, 2]),
                            "right": bool(actions[i, 3]),
                        },
                    }
                    for i in range(self.n)
                ],
            },
            timeout=180,
        )
        by_env = {r["env_id"]: r for r in resp["results"]}
        obs, rewards, dones, infos = [], [], [], []
        for i in range(self.n):
            st = decode_car_state(bytes.fromhex(by_env[i]["state"]))
            self._states[i] = st
            r, done, info = self._reward_done(i, st)
            obs.append(featurize(st, self.geom))
            rewards.append(r)
            dones.append(done)
            infos.append(info)
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            infos,
        )

    def reset_done(self, dones: np.ndarray) -> np.ndarray:
        """Reset finished envs; returns fresh obs for those slots (others NaN-filled)."""
        idxs = [i for i in range(self.n) if dones[i]]
        if not idxs:
            return np.full((self.n, 30), np.nan, dtype=np.float32)
        resp = self._node.call(
            {
                "cmd": "reset_all",
                "envs": [
                    {"env_id": i, "track": self.track_name, "max_frames": self.max_episode_frames}
                    for i in idxs
                ],
            },
            timeout=180,
        )
        out = np.full((self.n, 30), np.nan, dtype=np.float32)
        for r in resp["results"]:
            i = r["env_id"]
            st = decode_car_state(bytes.fromhex(r["state"]))
            self._states[i] = st
            self._prev_s[i] = 0.0
            self._prev_cp[i] = st.next_checkpoint_index
            self._prev_gate_dist[i] = None
            out[i] = featurize(st, self.geom)
        return out

    def _reward_done(self, i: int, st: CarState) -> tuple[float, bool, dict]:
        s = self.geom.progress(st.position)
        d_prog = s - self._prev_s[i]
        self._prev_s[i] = s
        r = self.rw["progress"] * d_prog - self.rw["time_penalty"] * (self.R / 1000.0)

        # Gate-shaping term: reward shrinking distance to the next gate. This
        # carries the car THROUGH sharp corners where the centerline polyline
        # stalls (progress can't increase until past the apex). The gate the
        # car is driving toward = its current next_checkpoint_index.
        dist = self._dist_to_next_gate(i, st)
        if dist is not None:
            prev = self._prev_gate_dist[i]
            if prev is not None:
                r += self.rw["gate_progress"] * (prev - dist)
            self._prev_gate_dist[i] = dist

        done = False
        reason = None
        cp = st.next_checkpoint_index
        if cp > self._prev_cp[i]:
            r += self.rw["checkpoint"] * (cp - self._prev_cp[i])
            self._prev_cp[i] = cp
            self._prev_gate_dist[i] = None  # re-anchor on the new gate
        if st.collision_impulses:
            r -= self.rw["collision"] * min(sum(st.collision_impulses), 4000.0)
        if st.finish_frames is not None:
            r += self.rw["finish"] - 0.001 * st.finish_frames
            done, reason = True, "finish"
        elif st.frames >= self.max_episode_frames:
            done, reason = True, "timeout"
        return r, done, {"progress_s": s, "reason": reason, "frames": st.frames}

    def _dist_to_next_gate(self, i: int, st: CarState) -> float | None:
        """Euclidean distance from the car to the center of the gate it is
        currently driving toward (next_checkpoint_index into ordered gates)."""
        gates = self.geom.gates
        if not gates:
            return None
        idx = min(st.next_checkpoint_index, len(gates) - 1)
        g = gates[idx]
        dx = st.position[0] - g.center[0]
        dy = st.position[1] - g.center[1]
        dz = st.position[2] - g.center[2]
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    def close(self):
        self._node.close()
