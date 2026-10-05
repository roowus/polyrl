"""PolyRL TAS optimizer — horn-weighted CEM over finishing recordings.

The main feature: take laps that already finish (leaderboard / hand-driven)
and optimize the time. Not RL — direct recording-space search.

Representation: a recording is per-channel sorted toggle frames at 1 kHz.
Mutation is horn-weighted (Gabriel's horn): the chance of mutating a toggle
rises along the lap — the end of the run (where the time is won/lost) gets
jittered hardest, the clean start is left alone. As a section is mastered
(better laps found), the horn's center creeps *forward* toward the start, so
optimization always targets the current weakest section. This is the user's
design: "mutate toward the end, and as the end improves, move the mutation up."

Score = finish time (fewer frames = better). Non-finishers get a progress
score as a fallback so the search can recover a finisher from a bad mutation,
but a finished lap always outranks an unfinished one.

Usage:
  uv run python -m brain.tas --track summer1 --gens 200 --pop 32
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .codec import Recording
from .demos import load_fixtures
from .env import _NodeProc
from .tracklib import load_track_geom

REPO = Path(__file__).resolve().parent.parent
CHANNELS = ("up", "right", "down", "left", "reset")


def rec_to_dense(rec: Recording, T: int) -> np.ndarray:
    d = rec.to_dense(T)
    return np.array([[x[c] for c in CHANNELS] for d_ in d for x in [d_]], dtype=bool)


def dense_to_rec(arr: np.ndarray) -> str:
    rec = Recording()
    state = {c: False for c in CHANNELS}
    for t in range(arr.shape[0]):
        for j, c in enumerate(CHANNELS):
            if bool(arr[t, j]) != state[c]:
                getattr(rec, c).append(t)
                state[c] = bool(arr[t, j])
    return rec.serialize()


def horn_weights(T: int, center_frac: float, width: float) -> np.ndarray:
    """Per-frame mutation probability: Gaussian bump at center_frac of the lap.
    The horn — wide exploration at the frontier section, ~0 elsewhere."""
    x = np.linspace(0.0, 1.0, T)
    return np.exp(-0.5 * ((x - center_frac) / width) ** 2)


class TASOptimizer:
    def __init__(self, track: str, horn_width: float = 0.12, seed: int = 0):
        self.track = track
        self.geom = load_track_geom(track)
        self.rng = np.random.default_rng(seed)
        self.node = _NodeProc()
        self.horn_width = horn_width
        self.horn_center = 0.95  # start: explore the very end of the lap
        self._load_examples()

    def _load_examples(self):
        self.examples = []  # (claimed frames, dense bool array)
        for fx in load_fixtures():
            if fx.get("track", self.track) != self.track:
                continue
            rec = Recording.deserialize(fx["recording"])
            self.examples.append((fx.get("frames", 1 << 30), rec))
        self.examples.sort(key=lambda x: x[0])
        if not self.examples:
            raise RuntimeError(f"no example laps for {self.track}")
        print(f"[tas] {len(self.examples)} example laps; fastest {self.examples[0][0]} frames", flush=True)

    def _mutate(self, base: np.ndarray, sigma_ms: float) -> np.ndarray:
        """Horn-weighted mutation: jitter toggle *times* (shift when a button
        flips, preserving the press/hold structure), with mutation probability
        following the horn (high at horn_center, ~0 at the start). We operate on
        the toggle list and rebuild — NOT inverting spans, which corrupts state."""
        T = base.shape[0]
        out = np.empty_like(base)
        pw = horn_weights(T, self.horn_center, self.horn_width)
        for j in range(len(CHANNELS)):
            col = base[:, j]
            toggles = list(np.flatnonzero(np.diff(col.astype(np.int8)) != 0) + 1)
            if col[0]:
                toggles = [0] + toggles
            new_toggles = []
            for tf in toggles:
                # mutation probability scales with the horn at this frame
                # (low base rate: most toggles stay; the horn zone gets most)
                if self.rng.random() < 0.03 + 0.5 * pw[min(tf, T - 1)]:
                    jitter = int(round(self.rng.normal(0, sigma_ms)))
                    if jitter != 0:
                        tf = int(np.clip(tf + jitter, 0, T - 1))
                new_toggles.append(tf)
            # rebuild the column from the (sorted, deduped) toggle list
            new_toggles = sorted(set(new_toggles))
            newcol = np.zeros(T, dtype=bool)
            state = False
            prev = 0
            for tf in new_toggles:
                newcol[prev:tf] = state
                state = not state
                prev = tf
            newcol[prev:] = state
            out[:, j] = newcol
        return out

    def score_batch(self, arrs: list[np.ndarray]) -> list[dict]:
        recordings = [dense_to_rec(a) for a in arrs]
        resp = self.node.call(
            {
                "cmd": "score_batch",
                "track": self.track,
                "recordings": recordings,
                "sample_every": 400,
                "max_frames": max(a.shape[0] for a in arrs) + 3000,
            },
            timeout=300,
        )
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        return [
            {
                "finished": r["finished"],
                "finishFrames": r["finishFrames"],
                "frames": r["frames"],
                "recording": rec,
            }
            for rec, r in zip(recordings, resp["results"])
        ]

    def run(self, gens: int, pop: int, out: str | None = None):
        # champion = fastest example (shortest claimed frame count)
        best_frames, best_rec_obj = self.examples[0]
        champion = rec_to_dense(best_rec_obj, int(best_frames + 2000))
        champion_frames = best_frames
        best_ever = (champion_frames, dense_to_rec(champion))
        print(f"[tas] champion seed: {champion_frames} frames = {champion_frames/1000:.2f}s", flush=True)

        sigma = 40.0  # toggle jitter in ms, annealed down to a useful floor
        gens_since_improve = 0
        for g in range(gens):
            t0 = time.time()
            sigma = max(8.0, sigma * 0.985)  # floor at 8ms — 3ms was below the noise floor
            plans = [champion.copy()]  # always re-test the champion
            for _ in range(pop - 1):
                plans.append(self._mutate(champion, sigma))
            scored = self.score_batch(plans)

            # pick the best: finished-with-fewest-frames beats everything
            def keyfn(s):
                return (1 if s["finished"] else 0, -(s["finishFrames"] or 1 << 30))
            best_idx = max(range(len(scored)), key=lambda i: keyfn(scored[i]))
            best = scored[best_idx]

            improved = False
            if best["finished"] and best["finishFrames"] is not None:
                if best_ever is None or best["finishFrames"] < best_ever[0]:
                    champion = plans[best_idx]
                    champion_frames = best["finishFrames"]
                    best_ever = (best["finishFrames"], best["recording"])
                    improved = True
                    gens_since_improve = 0
                    if out:
                        Path(out).parent.mkdir(exist_ok=True)
                        Path(out).write_text(json.dumps({
                            "track": self.track, "gen": g, "lap_seconds": best["finishFrames"] / 1000,
                            "finish_frames": best["finishFrames"], "recording": best["recording"],
                        }))

            # horn creep: advance toward the start both on improvement (that
            # section is now strong) AND on exhaustion (the current section
            # isn't yielding — move on). This is what keeps the horn moving
            # through the lap instead of parked at the finish line.
            if improved:
                self.horn_center = max(0.25, self.horn_center - 0.03)
            else:
                gens_since_improve += 1
                if gens_since_improve >= 12:
                    self.horn_center = max(0.25, self.horn_center - 0.04)
                    gens_since_improve = 0

            n_fin = sum(1 for s in scored if s["finished"])
            lap = best["finishFrames"] / 1000 if best["finished"] and best["finishFrames"] else None
            print(
                f"[tas] gen {g}: best {'LAP ' + f'{lap:.3f}s' if lap else 'no finish'} "
                f"({n_fin}/{pop} finish) | horn@{self.horn_center:.2f} sigma {sigma:.0f}ms "
                f"| {'IMPROVED' if improved else ''} | {time.time()-t0:.1f}s",
                flush=True,
            )
        return best_ever

    def close(self):
        self.node.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="summer1")
    ap.add_argument("--gens", type=int, default=200)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--horn-width", type=float, default=0.12)
    ap.add_argument("--out", default=str(REPO / "runs" / "tas_best.json"))
    args = ap.parse_args()

    tas = TASOptimizer(args.track, horn_width=args.horn_width)
    try:
        best = tas.run(args.gens, args.pop, out=args.out)
        if best:
            print(f"[tas] BEST: {best[0]} frames = {best[0]/1000:.3f}s → {args.out}", flush=True)
    finally:
        tas.close()


if __name__ == "__main__":
    main()
