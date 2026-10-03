"""Vectorized SAC training on the tmrl path-progress reward.

tmrl's recipe (from a full study of the repo):
  - reward = path-cursor advancement over a demo lap resampled to 0.1 m
    (elastic 50 m forward scan + 1 m rewind; Markovian; no distance terms)
  - NO BC, NO demo buffer — the demo is the reward manifold, nothing else
  - gamma 0.997 at 50 Hz (≈ tmrl's 0.995 at 20 Hz), fixed alpha 0.01,
    tiny LRs, (256,256) MLPs, UTD ~4
  - terminate after ~0.5 s of zero progress (past a grace period)

Usage:
  uv run python -m brain.train_vec --track summer1 --envs 8 --steps 200000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .agent import SAC, SacConfig
from .buffer import ReplayBuffer
from .features import OBS_DIM
from .vec_env import VecPolyTrackEnv

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="summer1")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--bc", default=None, help="(optional) BC checkpoint to warm-start the actor")
    ap.add_argument("--start-steps", type=int, default=1000)
    ap.add_argument("--updates-per-step", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--logdir", default=str(REPO / "runs"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    env = VecPolyTrackEnv(n_envs=args.envs, track=args.track, control_hz=50, max_episode_frames=60_000)

    sac = SAC(
        SacConfig(
            obs_dim=OBS_DIM,
            gamma=args.gamma,
            alpha=args.alpha,
            auto_alpha=False,
            lr=args.lr,
            bc_coef=0.0,  # tmrl: no imitation — the demo IS the reward path
        ),
        device=device,
    )
    if args.bc:
        ck = torch.load(args.bc, map_location=device, weights_only=False)
        sac.actor.load_state_dict(ck["actor"])
        print(f"[train] actor warm-started from {args.bc}")

    buffer = ReplayBuffer(demo_ratio=0.0)

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.logdir)
    except ImportError:
        writer = None

    out_path = args.out or str(Path(args.logdir) / f"sac_{args.track}.pt")
    best_progress_path = str(Path(args.logdir) / f"sac_{args.track}_bestprog.pt")
    obs = env.reset_all()
    ep_ret = np.zeros(args.envs)
    ep_len = np.zeros(args.envs, dtype=np.int64)
    ep_count = 0
    best_finish = None
    best_progress = 0.0
    total_frames = 0
    t0 = time.time()

    EVAL_EVERY = 5000
    eval_env = None

    def eval_progress() -> tuple[float, float | None]:
        """Deterministic rollout → (max progress frac, lap_s if finished)."""
        nonlocal eval_env
        if eval_env is None:
            eval_env = VecPolyTrackEnv(n_envs=1, track=args.track, control_hz=50, max_episode_frames=90_000)
        eobs = eval_env.reset_all()
        max_prog = 0.0
        lap = None
        for _ in range(4500):
            with torch.no_grad():
                p = sac.actor(torch.tensor(eobs, dtype=torch.float32, device=device))
                a = (p > 0.5).cpu().numpy().astype(np.int64)
            eobs, _, edone, einfos = eval_env.step(a)
            max_prog = max(max_prog, einfos[0]["progress_s"])
            if edone[0]:
                if einfos[0].get("reason") == "finish":
                    lap = einfos[0]["finish_frames"] / 1000
                break
        return max_prog, lap

    for step in range(1, args.steps + 1):
        if step <= args.start_steps:
            actions = np.random.randint(0, 2, (args.envs, 4))
        else:
            with torch.no_grad():
                p = sac.actor(torch.tensor(obs, dtype=torch.float32, device=device))
                actions = torch.bernoulli(p).cpu().numpy().astype(np.int64)

        next_obs, rewards, dones, infos = env.step(actions)

        for i in range(args.envs):
            buffer.add(obs[i], actions[i], next_obs[i], rewards[i], dones[i])
            ep_ret[i] += rewards[i]
            ep_len[i] += 1
            total_frames += env.R

        if dones.any():
            fresh = env.reset_done(dones)
            for i in range(args.envs):
                if dones[i]:
                    reason = infos[i].get("reason")
                    if reason == "finish":
                        lap = infos[i]["finish_frames"] / 1000
                        if best_finish is None or lap < best_finish:
                            best_finish = lap
                            sac.save(out_path)
                            print(f"[train] NEW BEST lap {lap:.2f}s (step {step})", flush=True)
                        if writer:
                            writer.add_scalar("finish/lap_s", lap, step)
                    if writer:
                        writer.add_scalar("episode/return", ep_ret[i], step)
                        writer.add_scalar("episode/len", ep_len[i], step)
                    ep_ret[i] = 0.0
                    ep_len[i] = 0
                    ep_count += 1
                    next_obs[i] = fresh[i]
        obs = next_obs

        if step > args.start_steps and len(buffer) >= args.batch:
            for _ in range(args.updates_per_step):
                metrics = sac.update(buffer.sample(args.batch))
            if writer and step % 100 == 0:
                for k, v in metrics.items():
                    writer.add_scalar(f"sac/{k}", v, step)
                writer.add_scalar("train/sim_frames", total_frames, step)
                writer.add_scalar("train/sim_speed_vs_rt", total_frames / 1000 / (time.time() - t0), step)

        if step % 1000 == 0:
            print(
                f"[train] step {step} eps {ep_count} frames {total_frames} "
                f"({total_frames/1000/(time.time()-t0):.1f}k f/s) best_lap {best_finish} "
                f"mean_ret {ep_ret.mean():.1f}",
                flush=True,
            )

        if step % EVAL_EVERY == 0:
            prog, lap = eval_progress()
            if writer:
                writer.add_scalar("eval/max_progress", prog, step)
            marker = f" (lap {lap:.2f}s)" if lap else ""
            if prog > best_progress:
                best_progress = prog
                sac.save(best_progress_path)
                print(f"[train] step {step}: NEW BEST progress {prog:.1%}{marker} → saved", flush=True)
            else:
                print(f"[train] step {step}: eval progress {prog:.1%}{marker}", flush=True)

    sac.save(out_path)
    print(f"[train] saved {out_path}; best lap {best_finish}; best progress {best_progress:.1%} → {best_progress_path}", flush=True)
    if writer:
        writer.close()
    if eval_env is not None:
        eval_env.close()
    env.close()


if __name__ == "__main__":
    main()
