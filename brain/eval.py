"""Roll out a policy in the headless env; report finish/lap time.

Usage: uv run python -m brain.eval bc_summer1.pt --track summer1 [--episodes 3]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .agent import Actor
from .env import PolyTrackEnv
from .features import OBS_DIM, featurize


def load_actor(path: Path, device: str) -> Actor:
    ck = torch.load(path, map_location=device, weights_only=False)
    actor = Actor(ck.get("obs_dim", OBS_DIM)).to(device)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    return actor


def rollout(actor: Actor, env: PolyTrackEnv, deterministic: bool = True, max_steps: int = 3000):
    import numpy as np

    obs = env.reset()
    geom = env.geom
    collision_decay = 0.0
    for k in range(max_steps):
        st = env._last_state
        feats = torch.tensor([featurize(st, geom, collision_decay)], dtype=torch.float32)
        with torch.no_grad():
            p = actor(feats.to(next(actor.parameters()).device))
            a = (p > 0.5).float() if deterministic else torch.bernoulli(p)
        action = tuple(int(v) for v in a[0].tolist())
        obs, r, done, info = env.step(action)
        if st.collision_impulses:
            collision_decay = 0.7 * collision_decay + 0.3 * min(sum(st.collision_impulses), 4000)
        if done:
            return True, info
    return False, info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--track", default="summer1")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--stochastic", action="store_true")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    actor = load_actor(Path(args.ckpt), device)
    env = PolyTrackEnv(track=args.track, control_hz=50, max_episode_frames=60_000)
    for ep in range(args.episodes):
        ok, info = rollout(actor, env, deterministic=not args.stochastic)
        if ok and info.get("reason") == "finish":
            print(f"[eval] ep{ep}: FINISHED — {info['frames']/1000:.2f}s lap")
        else:
            print(f"[eval] ep{ep}: did not finish (frames={info['frames']}, reason={info['reason']})")
    env.close()
