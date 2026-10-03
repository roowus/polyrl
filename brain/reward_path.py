"""tmrl-style path-progress reward.

The reward polyline is built from an example lap's *positions* (resampled to
~10 cm), NOT from track geometry — so it works on decoration-heavy custom
tracks and needs no centerline. Reward per step = how far the car advanced the
cursor along the path, with tmrl's two key rules:

  - elastic forward scan: look up to CHECK_FORWARD points ahead for the nearest
    point; the budget refills on each improvement, so cutting a corner (skipping
    demo points) still pays. This is what lets the policy beat the demo's line.
  - backward rewind: if no forward progress, allow the cursor to slide back to
    the nearest point within CHECK_BACKWARD — makes the reward a state function
    (Markovian), immune to oscillation farming.

Reference: tmrl custom/tm/utils/compute_reward.py (RewardFunction).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def resample_path(positions: np.ndarray, step_m: float = 0.1) -> np.ndarray:
    """Resample a position trace to ~uniformly spaced points (arc-length)."""
    if len(positions) < 2:
        return positions.copy()
    pts = [positions[0]]
    acc = 0.0
    for i in range(1, len(positions)):
        a = np.asarray(pts[-1])
        b = np.asarray(positions[i])
        seg = np.linalg.norm(b - a)
        while acc + seg >= step_m:
            t = (step_m - acc) / seg
            p = a + t * (b - a)
            pts.append(p)
            a = p
            seg = np.linalg.norm(b - a)
            acc = 0.0
        acc += seg
    return np.asarray(pts)


@dataclass
class PathRewardConfig:
    step_m: float = 0.1          # polyline spacing
    check_forward: int = 500     # 50 m of elastic forward scan (tmrl default)
    check_backward: int = 10     # 1 m rewind
    max_stray: float = 100.0     # m — beyond this, freeze cursor, zero reward
    reward_scale: float = 0.01   # per point advanced (0.1 m) → 0.1 per meter
    finish_bonus: float = 100.0
    # termination: no progress for this many consecutive steps (after grace)
    grace_steps: int = 70
    failure_countdown: int = 25  # 0.5s at 50 Hz


class PathReward:
    """Stateful per-episode reward cursor over the path polyline."""

    def __init__(self, path: np.ndarray, cfg: PathRewardConfig | None = None):
        self.cfg = cfg or PathRewardConfig()
        self.path = np.asarray(path, dtype=np.float64)
        self.n = len(self.path)
        self.reset()

    def reset(self):
        self.cur_idx = 0
        self._no_progress = 0
        self._steps = 0

    def step(self, pos) -> tuple[float, bool]:
        """→ (reward, terminate). pos = car (x,y,z)."""
        cfg = self.cfg
        self._steps += 1
        p = np.asarray(pos, dtype=np.float64)

        # elastic forward scan from cursor
        best_idx = self.cur_idx
        best_d = np.linalg.norm(self.path[self.cur_idx] - p)
        temp = cfg.check_forward
        i = self.cur_idx
        while i + 1 < self.n and temp > 0:
            i += 1
            d = np.linalg.norm(self.path[i] - p)
            temp -= 1
            if d < best_d:
                best_d = d
                best_idx = i
                temp = cfg.check_forward  # refill on improvement (elastic)

        reward = 0.0
        if best_d > cfg.max_stray:
            # strayed off the path entirely — no reward, no cursor move
            pass
        elif best_idx > self.cur_idx:
            reward = (best_idx - self.cur_idx) * cfg.reward_scale
            self.cur_idx = best_idx
            self._no_progress = 0
        else:
            # no forward progress: allow backward rewind (Markovian)
            lo = max(0, self.cur_idx - cfg.check_backward)
            back_idx = self.cur_idx
            back_d = best_d
            for j in range(self.cur_idx - 1, lo - 1, -1):
                d = np.linalg.norm(self.path[j] - p)
                if d < back_d:
                    back_d = d
                    back_idx = j
            self.cur_idx = back_idx
            self._no_progress += 1

        terminate = False
        if self._steps > cfg.grace_steps and self._no_progress >= cfg.failure_countdown:
            terminate = True
        return reward, terminate

    def finish_reward(self) -> float:
        return self.cfg.finish_bonus

    @property
    def progress_frac(self) -> float:
        return self.cur_idx / max(1, self.n - 1)
