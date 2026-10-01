"""Replay buffer with demo mixing (never-evicted demo pool)."""

from __future__ import annotations

import random

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int = 200_000, demo_ratio: float = 0.25, seed: int = 0):
        self.buf = []
        self.cap = capacity
        self.pos = 0
        self.demo = []  # never evicted
        self.demo_ratio = demo_ratio
        self.rng = random.Random(seed)

    def add(self, obs, action, next_obs, reward, done):
        t = (
            np.asarray(obs, dtype=np.float32),
            np.asarray(action, dtype=np.int8),
            np.asarray(next_obs, dtype=np.float32),
            np.float32(reward),
            np.float32(done),
        )
        if len(self.buf) < self.cap:
            self.buf.append(t)
        else:
            self.buf[self.pos] = t
            self.pos = (self.pos + 1) % self.cap

    def add_demo(self, transitions):
        self.demo.extend(transitions)

    def __len__(self):
        return len(self.buf) + len(self.demo)

    def sample(self, n: int):
        n_demo = min(len(self.demo), round(n * self.demo_ratio)) if self.demo else 0
        n_live = n - n_demo
        live = self.rng.sample(self.buf, min(n_live, len(self.buf))) if self.buf else []
        demo = self.rng.sample(self.demo, n_demo) if n_demo else []
        batch = live + demo
        obs, act, next_obs, rew, done = zip(*batch)
        return (
            torch.tensor(np.array(obs), dtype=torch.float32),
            torch.tensor(np.array(act), dtype=torch.long),
            torch.tensor(np.array(next_obs), dtype=torch.float32),
            torch.tensor(np.array(rew), dtype=torch.float32),
            torch.tensor(np.array(done), dtype=torch.float32),
        )
