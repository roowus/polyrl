"""tmrl-faithful SAC training on the path-progress reward.

Continuous [gas, brake, steer] policy → quantized to buttons at the worker.
No BC, no demo buffer (the demo IS the reward path). tmrl hyperparameters.

Usage:
  uv run python -m brain.train_tmrl --track summer1 --envs 8 --steps 200000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .agent_sac import TmrlSAC, TmrlSacConfig, quantize_to_buttons
from .buffer import ReplayBuffer
from .features import OBS_DIM
from .vec_env import VecPolyTrackEnv

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="summer1")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--start-steps", type=int, default=2000, help="random-action steps before policy")
    ap.add_argument("--updates-per-step", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--end-explore", type=float, default=0.3,
                    help="peak action-jitter amplitude at the exploration frontier (Gabriel's horn)")
    ap.add_argument("--horn-offset", type=float, default=0.03,
                    help="how far past the frontier the exploration peak sits (fraction of track)")
    ap.add_argument("--horn-width", type=float, default=0.06,
                    help="width of the exploration bump (fraction of track)")
    ap.add_argument("--resume", default=None, help="tmrl_*.pt checkpoint to resume the policy from")
    ap.add_argument("--logdir", default=str(REPO / "runs"))
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    env = VecPolyTrackEnv(n_envs=args.envs, track=args.track, control_hz=50, max_episode_frames=60_000)

    if args.resume:
        sac = TmrlSAC.load(args.resume, device=device)
        print(f"[tmrl] resumed from {args.resume}", flush=True)
    else:
        sac = TmrlSAC(TmrlSacConfig(obs_dim=OBS_DIM, gamma=args.gamma, alpha=args.alpha), device=device)
    buffer = ReplayBuffer(capacity=200_000, demo_ratio=0.0)

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.logdir)
    except ImportError:
        writer = None

    out_path = str(Path(args.logdir) / f"tmrl_{args.track}.pt")
    best_progress_path = str(Path(args.logdir) / f"tmrl_{args.track}_bestprog.pt")
    obs = env.reset_all()
    ep_ret = np.zeros(args.envs)
    ep_count = 0
    best_finish = None
    best_progress = 0.0
    total_frames = 0
    t0 = time.time()

    EVAL_EVERY = 5000
    eval_env = None

    # Gabriel's-horn frontier: the furthest path-progress the policy reliably
    # reaches. Tracked as an EMA of the per-episode 90th-percentile progress
    # across the fleet (robust to one lucky deep run). Starts at 0 and creeps
    # forward as sections are mastered; exploration concentrates just past it.
    frontier = {"pos": 0.05}

    def eval_progress():
        nonlocal eval_env
        if eval_env is None:
            eval_env = VecPolyTrackEnv(n_envs=1, track=args.track, control_hz=50, max_episode_frames=90_000)
        eobs = eval_env.reset_all()
        max_prog, lap = 0.0, None
        for _ in range(4500):
            a_cont = sac.act(torch.tensor(eobs, dtype=torch.float32), deterministic=True)
            a = np.array([quantize_to_buttons(*a_cont)], dtype=np.int64)
            eobs, _, edone, einfos = eval_env.step(a)
            max_prog = max(max_prog, einfos[0]["progress_s"])
            if edone[0]:
                if einfos[0].get("reason") == "finish":
                    lap = einfos[0]["finish_frames"] / 1000
                break
        return max_prog, lap

    for step in range(1, args.steps + 1):
        if step <= args.start_steps:
            actions = np.random.uniform(-1, 1, (args.envs, 3))
        else:
            with torch.no_grad():
                # Gabriel's-horn exploration (user's idea): mutation is
                # concentrated just PAST the competence frontier — the furthest
                # point the policy reliably reaches. As the tail gets mastered,
                # the frontier creeps forward and exploration follows it. The
                # horn flares: near-zero jitter before the frontier, a tight
                # peak right at it, tapering after.
                p, _ = sac.model.actor(torch.tensor(obs, dtype=torch.float32, device=device), deterministic=False)
                actions = p.cpu().numpy()
                if args.end_explore > 0.0:
                    prog = np.clip(obs[:, 20], 0.0, 1.0)  # path progress fraction per env
                    # Gaussian bump centered just past the frontier, width ~
                    # horn_width, amplitude end_explore. Below the frontier
                    # (already-mastered) → ~0 jitter; at the frontier → max.
                    center = frontier["pos"] + args.horn_offset
                    bump = np.exp(-0.5 * ((prog - center) / args.horn_width) ** 2)
                    jitter = (args.end_explore * bump)[:, None] * np.random.standard_normal(actions.shape)
                    actions = np.clip(actions + jitter, -1.0, 1.0)

        # quantize continuous → buttons for the env
        btn = np.array([quantize_to_buttons(*a) for a in actions], dtype=np.int64)
        next_obs, rewards, dones, infos = env.step(btn)

        # update the competence frontier from where the fleet actually got
        progs = [inf["progress_s"] for inf in infos]
        fleet_p90 = float(np.percentile(progs, 90)) if progs else 0.0
        # EMA toward the fleet's strong runners; only move forward (never
        # retreat the frontier on a bad step — mastery is sticky)
        if fleet_p90 > frontier["pos"]:
            frontier["pos"] += 0.01 * (fleet_p90 - frontier["pos"])

        for i in range(args.envs):
            buffer.add(obs[i], actions[i], next_obs[i], rewards[i], dones[i])
            ep_ret[i] += rewards[i]
            total_frames += env.R

        if dones.any():
            fresh = env.reset_done(dones)
            for i in range(args.envs):
                if dones[i]:
                    if infos[i].get("reason") == "finish":
                        lap = infos[i]["finish_frames"] / 1000
                        if best_finish is None or lap < best_finish:
                            best_finish = lap
                            sac.save(out_path)
                            print(f"[tmrl] NEW BEST lap {lap:.2f}s (step {step})", flush=True)
                        if writer:
                            writer.add_scalar("finish/lap_s", lap, step)
                    if writer:
                        writer.add_scalar("episode/return", ep_ret[i], step)
                    ep_ret[i] = 0.0
                    ep_count += 1
                    next_obs[i] = fresh[i]
        obs = next_obs

        if step > args.start_steps and len(buffer) >= args.batch:
            for _ in range(args.updates_per_step):
                metrics = sac.update(buffer.sample(args.batch))
            if writer and step % 100 == 0:
                for k, v in metrics.items():
                    writer.add_scalar(f"sac/{k}", v, step)
                writer.add_scalar("explore/frontier", frontier["pos"], step)

        if step % 1000 == 0:
            print(
                f"[tmrl] step {step} eps {ep_count} frames {total_frames} "
                f"({total_frames/1000/(time.time()-t0):.1f}k f/s) best_lap {best_finish} "
                f"mean_ret {ep_ret.mean():.1f} frontier {frontier['pos']:.1%}",
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
                print(f"[tmrl] step {step}: NEW BEST progress {prog:.1%}{marker} → saved", flush=True)
            else:
                print(f"[tmrl] step {step}: eval progress {prog:.1%}{marker}", flush=True)

    sac.save(out_path)
    print(f"[tmrl] done. best lap {best_finish}; best progress {best_progress:.1%} → {best_progress_path}", flush=True)
    if writer:
        writer.close()
    if eval_env is not None:
        eval_env.close()
    env.close()


if __name__ == "__main__":
    main()
