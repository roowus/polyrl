"""stable-baselines3 training path — battle-tested PPO over the headless env.

Usage:
  uv run python -m brain.train_sb3 --track summer1 --steps 2000000
  uv run python -m brain.train_sb3 --track summer1 --bc runs/bc_summer1.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from .features import OBS_DIM
from .gym_env import PolyTrackGym

REPO = Path(__file__).resolve().parent.parent


class FinishCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.best = None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        for info in infos:
            if info.get("reason") == "finish":
                lap = info["frames"] / 1000
                self.logger.record("finish/lap_s", lap)
                if self.best is None or lap < self.best:
                    self.best = lap
                    self.model.save(str(Path(self.model.custom_logdir) / "best"))
                    print(f"[sb3] NEW BEST {lap:.2f}s")
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="summer1")
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--bc", default=None, help="BC checkpoint to warm-start the policy net")
    ap.add_argument("--logdir", default=str(REPO / "runs"))
    args = ap.parse_args()

    logdir = Path(args.logdir) / f"ppo_{args.track}_v1"
    logdir.mkdir(parents=True, exist_ok=True)

    env = Monitor(PolyTrackGym(track=args.track), filename=str(logdir / "monitor"))

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        ent_coef=0.01,
        verbose=0,
        tensorboard_log=str(logdir),
        device="mps" if torch.backends.mps.is_available() else "cpu",
        policy_kwargs=dict(net_arch=[256, 256]),
    )
    model.custom_logdir = str(logdir)

    if args.bc:
        ck = torch.load(args.bc, map_location="cpu", weights_only=False)
        # BC actor is mlp→head; PPO's policy_net is a fresh MLP. Warm-start by
        # copying the trunk where shapes line up, else skip (arch mismatch is
        # non-fatal — PPO learns from scratch fine).
        try:
            src = ck["actor"]
            pp = model.policy
            with torch.no_grad():
                # Actor: net.0 (Linear 30->256), net.2 (Linear 256->256), head (256->4)
                pp.mlp_extractor.policy_net[0].weight.copy_(src["net.0.weight"])
                pp.mlp_extractor.policy_net[0].bias.copy_(src["net.0.bias"])
                pp.mlp_extractor.policy_net[2].weight.copy_(src["net.2.weight"])
                pp.mlp_extractor.policy_net[2].bias.copy_(src["net.2.bias"])
            print("[sb3] warm-started trunk from BC")
        except Exception as e:
            print(f"[sb3] BC warm-start skipped: {e}")

    model.learn(total_timesteps=args.steps, callback=FinishCallback(), progress_bar=False)
    model.save(str(logdir / "final"))
    print(f"[sb3] saved {logdir}/final; best lap: {FinishCallback.best if hasattr(FinishCallback,'best') else 'see TB'}")
    env.close()


if __name__ == "__main__":
    main()
