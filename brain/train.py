"""SAC training loop: env interaction + demo-buffer mixing + TensorBoard.

Usage:
  uv run python -m brain.train --track summer1 --steps 100000
  uv run python -m brain.train --track summer1 --bc runs/bc_summer1.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .agent import SAC, SacConfig, Actor
from .buffer import ReplayBuffer
from .demos import extract_demo, load_fixtures
from .env import PolyTrackEnv
from .features import OBS_DIM, featurize

REPO = Path(__file__).resolve().parent.parent


def demo_transitions(track: str, geom) -> list:
    """(s, a, s', r, done) from the human demo via headless re-sim.
    Reward is recomputed with the env's reward shaping on the demo states."""
    env = PolyTrackEnv(track=track, control_hz=50, max_episode_frames=90_000)
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
            r = (s1 - prev_s) - 0.01 * 0.02  # progress - time penalty (matches env weights)
            prev_s = s0
            done = nst.finish_frames is not None
            if done:
                r += 200.0 - 0.001 * nst.finish_frames
            out.append(
                (
                    np.asarray(featurize(st, geom), dtype=np.float32),
                    np.asarray(demo.actions[k], dtype=np.int8),
                    np.asarray(featurize(nst, geom), dtype=np.float32),
                    np.float32(r),
                    np.float32(done),
                )
            )
    env.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="summer1")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--bc", default=None, help="BC checkpoint to warm-start the actor")
    ap.add_argument("--start-steps", type=int, default=2000, help="random steps before policy takes over")
    ap.add_argument("--updates-per-step", type=int, default=1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--demo-ratio", type=float, default=0.25)
    ap.add_argument("--logdir", default=str(REPO / "runs"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    env = PolyTrackEnv(track=args.track, control_hz=50, max_episode_frames=60_000)
    geom = env.geom

    sac = SAC(SacConfig(obs_dim=OBS_DIM), device=device)
    if args.bc:
        ck = torch.load(args.bc, map_location=device, weights_only=False)
        sac.actor.load_state_dict(ck["actor"])
        print(f"[train] actor warm-started from {args.bc}")

    buffer = ReplayBuffer(demo_ratio=args.demo_ratio)
    dts = demo_transitions(args.track, geom)
    buffer.add_demo(dts)
    print(f"[train] demo transitions: {len(dts)}")

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        SummaryWriter = None
    writer = SummaryWriter(log_dir=args.logdir) if SummaryWriter else None

    out_path = args.out or str(Path(args.logdir) / f"sac_{args.track}.pt")
    obs = env.reset()
    st = env._last_state
    collision_decay = 0.0
    ep_ret, ep_len, ep_count = 0.0, 0, 0
    best_finish = None
    t0 = time.time()
    total_frames = 0

    for step in range(1, args.steps + 1):
        if step <= args.start_steps:
            action = tuple(np.random.randint(0, 2, 4))
        else:
            feats = torch.tensor([featurize(st, geom, collision_decay)], dtype=torch.float32)
            action = sac.act(feats, deterministic=False)
        next_obs, r, done, info = env.step(action)
        nst = env._last_state
        if st.collision_impulses:
            collision_decay = 0.7 * collision_decay + 0.3 * min(sum(st.collision_impulses), 4000)
        buffer.add(
            featurize(st, geom, collision_decay),
            action,
            featurize(nst, geom, collision_decay),
            r,
            done,
        )
        ep_ret += r
        ep_len += 1
        total_frames += info.get("frames_delta", env.R)
        st = nst

        if done:
            reason = info.get("reason")
            if reason == "finish":
                lap = info["frames"] / 1000
                if best_finish is None or lap < best_finish:
                    best_finish = lap
                    sac.save(out_path)
                if writer:
                    writer.add_scalar("finish/lap_s", lap, step)
            if writer:
                writer.add_scalar("episode/return", ep_ret, step)
                writer.add_scalar("episode/len", ep_len, step)
            ep_ret, ep_len = 0.0, 0
            ep_count += 1
            obs = env.reset()
            st = env._last_state
            collision_decay = 0.0

        if step > args.start_steps and len(buffer) >= args.batch:
            for _ in range(args.updates_per_step):
                metrics = sac.update(buffer.sample(args.batch))
            if writer and step % 100 == 0:
                for k, v in metrics.items():
                    writer.add_scalar(f"sac/{k}", v, step)
                writer.add_scalar("train/sim_frames", total_frames, step)
                writer.add_scalar("train/sim_speed_vs_rt", total_frames / ((time.time() - t0) * 1000), step)
        if step % 2000 == 0:
            print(
                f"[train] step {step} eps {ep_count} frames {total_frames} "
                f"({total_frames/1000/(time.time()-t0):.1f}k f/s) best_lap {best_finish}"
            )

    sac.save(out_path)
    print(f"[train] saved {out_path}; best finish lap: {best_finish}")
    if writer:
        writer.close()
    env.close()


if __name__ == "__main__":
    main()
