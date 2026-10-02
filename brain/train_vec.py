"""Vectorized SAC training: N parallel cars, batched policy inference.

Usage:
  uv run python -m brain.train_vec --track summer1 --envs 8 --steps 50000 \
      --bc runs/bc_summer1.pt --logdir runs/tb_vec_v1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .agent import SAC, SacConfig
from .buffer import ReplayBuffer
from .demos import extract_demo, load_fixtures
from .features import OBS_DIM, featurize
from .vec_env import VecPolyTrackEnv

REPO = Path(__file__).resolve().parent.parent


def demo_transitions(track: str, geom, reward: dict) -> list:
    env_dummy = None
    out = []
    for fx in load_fixtures():
        if fx.get("track", track) != track:
            continue
        demo = extract_demo(track, fx["recording"])
        prev_s = 0.0
        for k in range(len(demo.states) - 1):
            st, nst = demo.states[k], demo.states[k + 1]
            s0 = geom.progress(st.position)
            s1 = geom.progress(nst.position)
            r = reward["progress"] * (s1 - prev_s) - reward["time_penalty"] * 0.02
            prev_s = s0
            done = nst.finish_frames is not None
            if done:
                r += reward["finish"] - 0.001 * nst.finish_frames
            out.append(
                (
                    np.asarray(featurize(st, geom), dtype=np.float32),
                    np.asarray(demo.actions[k], dtype=np.int8),
                    np.asarray(featurize(nst, geom), dtype=np.float32),
                    np.float32(r),
                    np.float32(done),
                )
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="summer1")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50_000, help="vector steps (each = envs × R frames)")
    ap.add_argument("--bc", default=None)
    ap.add_argument("--start-steps", type=int, default=500)
    ap.add_argument("--updates-per-step", type=int, default=4, help="REDQ: cheap env → high UTD")
    ap.add_argument("--bc-coef", type=float, default=0.0, help="BC anchor strength on demo actions")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--demo-ratio", type=float, default=0.25)
    ap.add_argument("--logdir", default=str(REPO / "runs"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    env = VecPolyTrackEnv(n_envs=args.envs, track=args.track, control_hz=50, max_episode_frames=60_000)
    geom = env.geom

    sac = SAC(SacConfig(obs_dim=OBS_DIM, bc_coef=args.bc_coef), device=device)
    if args.bc:
        ck = torch.load(args.bc, map_location=device, weights_only=False)
        sac.actor.load_state_dict(ck["actor"])
        print(f"[train] actor warm-started from {args.bc}")

    buffer = ReplayBuffer(demo_ratio=args.demo_ratio)
    dts = demo_transitions(args.track, geom, env.rw)
    buffer.add_demo(dts)
    print(f"[train] demo transitions: {len(dts)}")

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.logdir)
    except ImportError:
        writer = None

    out_path = args.out or str(Path(args.logdir) / f"sac_{args.track}.pt")
    best_progress_path = str(Path(args.logdir) / f"sac_{args.track}_bestprog.pt")
    obs = env.reset_all()
    states = list(env._states)
    ep_ret = np.zeros(args.envs)
    ep_len = np.zeros(args.envs, dtype=np.int64)
    ep_count = 0
    best_finish = None
    best_progress = 0.0
    total_frames = 0
    t0 = time.time()

    # periodic deterministic eval → save the checkpoint with the best
    # max-progress. Every run so far collapsed AFTER mid-run peak; saving only
    # on finish (never fires) or at end kept degenerate policies. This keeps
    # the best one.
    EVAL_EVERY = 5000
    eval_env = None  # lazily created so it doesn't hold a pool slot at boot

    def eval_progress() -> float:
        nonlocal eval_env
        if eval_env is None:
            eval_env = VecPolyTrackEnv(n_envs=1, track=args.track, control_hz=50, max_episode_frames=60_000)
        eobs = eval_env.reset_all()
        st_list = eval_env._states
        max_prog = 0.0
        for _ in range(3000):
            with torch.no_grad():
                p = sac.actor(torch.tensor(eobs, dtype=torch.float32, device=device))
                a = (p > 0.5).cpu().numpy().astype(np.int64)
            eobs, _, edone, einfos = eval_env.step(a)
            max_prog = max(max_prog, einfos[0]["progress_s"] / eval_env.geom.total_len)
            if edone[0]:
                break
        return max_prog

    for step in range(1, args.steps + 1):
        if step <= args.start_steps:
            actions = np.random.randint(0, 2, (args.envs, 4))
        else:
            with torch.no_grad():
                p = sac.actor(torch.tensor(obs, dtype=torch.float32, device=device))
                actions = torch.bernoulli(p).cpu().numpy().astype(np.int64)

        next_obs, rewards, dones, infos = env.step(actions)
        next_states = list(env._states)

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
                        lap = infos[i]["frames"] / 1000
                        if best_finish is None or lap < best_finish:
                            best_finish = lap
                            sac.save(out_path)
                            print(f"[train] NEW BEST lap {lap:.2f}s (step {step})")
                        if writer:
                            writer.add_scalar("finish/lap_s", lap, step)
                    if writer:
                        writer.add_scalar("episode/return", ep_ret[i], step)
                        writer.add_scalar("episode/len", ep_len[i], step)
                    ep_ret[i] = 0.0
                    ep_len[i] = 0
                    ep_count += 1
                    next_obs[i] = fresh[i]
                    next_states[i] = env._states[i]
        obs = next_obs
        states = next_states

        if step > args.start_steps and len(buffer) >= args.batch:
            # demo_batch for the BC anchor: raw demo (s, a) pairs
            demo_batch = None
            if args.bc_coef > 0 and buffer.demo:
                import random as _r
                dsample = _r.sample(buffer.demo, min(args.batch, len(buffer.demo)))
                d_obs = torch.tensor(np.array([d[0] for d in dsample]), dtype=torch.float32)
                d_act = torch.tensor(np.array([d[1] for d in dsample]), dtype=torch.long)
                demo_batch = (d_obs, d_act)
            for _ in range(args.updates_per_step):
                metrics = sac.update(buffer.sample(args.batch), demo_batch=demo_batch)
            if writer and step % 100 == 0:
                for k, v in metrics.items():
                    writer.add_scalar(f"sac/{k}", v, step)
                writer.add_scalar("train/sim_frames", total_frames, step)
                writer.add_scalar("train/sim_speed_vs_rt", total_frames / 1000 / (time.time() - t0), step)

        if step % 1000 == 0:
            print(
                f"[train] step {step} eps {ep_count} frames {total_frames} "
                f"({total_frames/1000/(time.time()-t0):.1f}k f/s) best_lap {best_finish} "
                f"mean_ret {ep_ret.mean():.1f}"
            )

        if step % EVAL_EVERY == 0:
            prog = eval_progress()
            if writer:
                writer.add_scalar("eval/max_progress", prog, step)
            if prog > best_progress:
                best_progress = prog
                sac.save(best_progress_path)
                print(f"[train] step {step}: NEW BEST progress {prog:.1%} → saved")

    sac.save(out_path)
    print(f"[train] saved {out_path}; best finish lap: {best_finish}; best progress: {best_progress:.1%} → {best_progress_path}")
    if writer:
        writer.close()
    if eval_env is not None:
        eval_env.close()
    env.close()


if __name__ == "__main__":
    main()
