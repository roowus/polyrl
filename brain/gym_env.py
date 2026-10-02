"""Gymnasium wrapper over VecPolyTrackEnv, for stable-baselines3.

Action space: Discrete(16) — 4 buttons (up, down, left, right) as a bitmask.
SB3 SAC is continuous-only, so we use it via a discretized wrapper OR just use
SB3's PPO/DQN which handle Discrete natively. Default: PPO (robust, on-policy,
hard to destabilize — the right tool after 6 hand-rolled SAC failures).

Observation: the 30-dim featurizer output (already normalized-ish).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .vec_env import VecPolyTrackEnv


class PolyTrackGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, track: str = "summer1", control_hz: float = 50.0, max_episode_frames: int = 60_000):
        super().__init__()
        self.env = VecPolyTrackEnv(n_envs=1, track=track, control_hz=control_hz, max_episode_frames=max_episode_frames)
        self.action_space = spaces.Discrete(16)  # 4-button bitmask
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32)

    @staticmethod
    def _to_buttons(a: int) -> tuple[int, int, int, int]:
        return ((a >> 0) & 1, (a >> 1) & 1, (a >> 2) & 1, (a >> 3) & 1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset_all()
        self.env._prev_s[0] = 0.0
        return obs[0], {}

    def step(self, action: int):
        a = np.array([self._to_buttons(int(action))], dtype=np.int64)
        obs, rewards, dones, infos = self.env.step(a)
        reward = float(rewards[0])
        done = bool(dones[0])
        truncated = done and infos[0].get("reason") == "timeout"
        terminated = done and infos[0].get("reason") == "finish"
        if done:
            # gymnasium expects a final obs; reset for the next episode start
            fresh = self.env.reset_done(np.array([True]))
            next_obs = fresh[0]
        else:
            next_obs = obs[0]
        return next_obs, reward, terminated, truncated, infos[0]

    def close(self):
        self.env.close()


class VecPolyTrackGym(gym.Env):
    """Vectorized variant — SB3 prefers its own VecEnv, but we expose the pool
    as a single gym env with vector rewards so PPO's n_envs=1 path still gets
    8× throughput per step. (Simpler than SB3's VecEnv plumbing for v1.)"""

    metadata = {"render_modes": []}

    def __init__(self, n_envs: int = 8, track: str = "summer1", control_hz: float = 50.0):
        super().__init__()
        self.env = VecPolyTrackEnv(n_envs=n_envs, track=track, control_hz=control_hz)
        self.n = n_envs
        self.action_space = spaces.MultiDiscrete([16] * n_envs)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_envs, 30), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.env.reset_all(), {}

    def step(self, actions):
        a = np.asarray(actions).reshape(self.n, 4) if np.asarray(actions).ndim > 1 else np.stack(
            [[(a >> b) & 1 for b in range(4)] for a in actions]
        )
        obs, rewards, dones, infos = self.env.step(a)
        if dones.any():
            fresh = self.env.reset_done(dones)
            for i in range(self.n):
                if dones[i]:
                    obs[i] = fresh[i]
        # SB3 VecEnv-style: return obs with terminal_observation in info
        return obs, rewards, dones, dones.copy(), infos

    def close(self):
        self.env.close()
