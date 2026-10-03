"""Vectorized PolyTrack env: N cars across the pooled headless workers.

Reward = tmrl-style path-progress (brain/reward_path.py): the demo lap's
position trace resampled to a 0.1 m polyline; per step the reward is how far
the car advanced a cursor along it (elastic forward scan + backward rewind).
No centerline, no gate distance, no speed terms — works on decoration-heavy
custom tracks and is immune to oscillation farming by construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .env import _NodeProc, decode_car_state, CarState
from .features import featurize, OBS_DIM
from .reward_path import PathReward, PathRewardConfig, resample_path
from .tracklib import load_track_geom

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "cache"


def _build_or_load_path(track: str, geom) -> tuple[np.ndarray, np.ndarray]:
    """Reward polyline + demo pace from the best example lap (cached)."""
    cache = CACHE / f"{track}_path.npz"
    if cache.exists():
        d = np.load(cache)
        return d["path"], d["speeds"]
    from .demos import extract_demo, load_fixtures

    fx = sorted(
        [f for f in load_fixtures() if f.get("track", track) == track],
        key=lambda f: f.get("frames", 1 << 30),
    )
    if not fx:
        raise RuntimeError(f"no example lap for {track} — record one first")
    demo = extract_demo(track, fx[0]["recording"])
    pos = np.array([st.position for st in demo.states])
    spd = np.array([st.speed_kmh for st in demo.states])
    path, speeds = resample_path(pos, 0.1, speeds=spd)
    cache.parent.mkdir(exist_ok=True)
    np.savez(cache, path=path, speeds=speeds)
    return path, speeds


class VecPolyTrackEnv:
    def __init__(
        self,
        n_envs: int = 8,
        track: str = "summer1",
        control_hz: float = 50.0,
        max_episode_frames: int = 60_000,
        reward_cfg: PathRewardConfig | None = None,
    ):
        self.n = n_envs
        self.track_name = track
        self.geom = load_track_geom(track)
        self.R = max(1, round(1000 / control_hz))
        self.max_episode_frames = max_episode_frames
        self.path, self.path_speeds = _build_or_load_path(track, self.geom)
        self._reward_cfg = reward_cfg or PathRewardConfig()
        self._rewards = [PathReward(self.path, self._reward_cfg, path_speeds=self.path_speeds) for _ in range(n_envs)]

        self._node = _NodeProc()
        self._env_ids: list[int | None] = [None] * n_envs
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
            self._rewards[i].reset()
            feats.append(featurize(st, self.path, self._rewards[i].cur_idx))
        return np.asarray(feats, dtype=np.float32)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
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
            obs.append(featurize(st, self.path, self._rewards[i].cur_idx))
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
        idxs = [i for i in range(self.n) if dones[i]]
        if not idxs:
            return np.full((self.n, OBS_DIM), np.nan, dtype=np.float32)
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
        out = np.full((self.n, OBS_DIM), np.nan, dtype=np.float32)
        for r in resp["results"]:
            i = r["env_id"]
            st = decode_car_state(bytes.fromhex(r["state"]))
            self._states[i] = st
            self._rewards[i].reset()
            out[i] = featurize(st, self.path, self._rewards[i].cur_idx)
        return out

    def _reward_done(self, i: int, st: CarState) -> tuple[float, bool, dict]:
        pr = self._rewards[i]
        r, path_terminated = pr.step(st.position, speed_kmh=st.speed_kmh)

        done = False
        reason = None
        if st.finish_frames is not None:
            r += pr.finish_reward()
            done, reason = True, "finish"
        elif st.frames >= self.max_episode_frames:
            done, reason = True, "timeout"
        elif path_terminated:
            done, reason = True, "no_progress"
        return r, done, {
            "progress_s": pr.progress_frac,
            "reason": reason,
            "frames": st.frames,
            "finish_frames": st.finish_frames,
        }

    def close(self):
        self._node.close()
