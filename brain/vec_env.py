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
            "progress": 1.0,
            "checkpoint": 12.0,
            "finish": 200.0,
            "time_penalty": 0.01,
            "collision": 0.002,
            **(reward or {}),
        }
        self._node = _NodeProc()
        self._env_ids: list[int | None] = [None] * n_envs
        self._prev_s = np.zeros(n_envs, dtype=np.float64)
        self._prev_cp = np.zeros(n_envs, dtype=np.int32)
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
            out[i] = featurize(st, self.geom)
        return out

    def _reward_done(self, i: int, st: CarState) -> tuple[float, bool, dict]:
        s = self.geom.progress(st.position)
        d_prog = s - self._prev_s[i]
        self._prev_s[i] = s
        r = self.rw["progress"] * d_prog - self.rw["time_penalty"] * (self.R / 1000.0)

        done = False
        reason = None
        cp = st.next_checkpoint_index
        if cp > self._prev_cp[i]:
            r += self.rw["checkpoint"] * (cp - self._prev_cp[i])
            self._prev_cp[i] = cp
        if st.collision_impulses:
            r -= self.rw["collision"] * min(sum(st.collision_impulses), 4000.0)
        if st.finish_frames is not None:
            r += self.rw["finish"] - 0.001 * st.finish_frames
            done, reason = True, "finish"
        elif st.frames >= self.max_episode_frames:
            done, reason = True, "timeout"
        return r, done, {"progress_s": s, "reason": reason, "frames": st.frames}

    def close(self):
        self._node.close()
