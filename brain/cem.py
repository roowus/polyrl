"""CEM over PolyTrack recordings, seeded from example laps.

A candidate IS a recording (the game's own format: per-channel sorted toggle
frames at 1 kHz). No downsampling — we mutate toggle *times* directly, so the
millisecond steering timing a lap depends on survives (the 50 Hz prob-plan
approach lost exactly this and stalled at gate 1).

Mutation = Gaussian jitter of each toggle's frame time (sigma shrinks over
generations) + occasional toggle insert/delete. Scoring = the game's own
simulation: gates passed dominates, distance-to-next-gate breaks ties, finish
time wins outright. No reward shaping, so away-then-toward sections work.

Usage:
  uv run python -m brain.cem --track summer1 --gens 60 --pop 24
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
from .env import _NodeProc, decode_car_state
from .tracklib import load_track_geom

REPO = Path(__file__).resolve().parent.parent
CHANNELS = ("up", "right", "down", "left", "reset")


def rec_to_array(rec: Recording, T: int) -> np.ndarray:
    """Recording → (T, 5) bool frame matrix (dense 1 kHz)."""
    dense = rec.to_dense(T)
    return np.array([[d[c] for c in CHANNELS] for d in dense], dtype=bool)


def array_to_rec(arr: np.ndarray) -> str:
    rec = Recording()
    state = {c: False for c in CHANNELS}
    for t in range(arr.shape[0]):
        for j, c in enumerate(CHANNELS):
            if bool(arr[t, j]) != state[c]:
                getattr(rec, c).append(t)
                state[c] = bool(arr[t, j])
    return rec.serialize()


class GateScorer:
    """Score a sampled state trajectory by gates passed + distance to next."""

    def __init__(self, geom):
        self.gates = sorted(geom.gates, key=lambda g: (g.is_finish, g.index))
        self.centers = [g.center for g in self.gates]
        self.radii = [max(g.size) * 0.5 + 3.0 for g in self.gates]

    def score(self, samples: list[bytes], finished: bool, finish_frames: int | None) -> tuple[float, int]:
        if not samples:
            return -1e6, 0
        max_gate = -1
        best_dist_next = float("inf")
        for raw in samples:
            st = decode_car_state(raw)
            for gi in range(max_gate + 1, len(self.gates)):
                d = math.dist(st.position, self.centers[gi])
                if d < self.radii[gi]:
                    max_gate = gi
                else:
                    if gi == max_gate + 1:
                        best_dist_next = min(best_dist_next, d)
                    break
        gates_passed = max_gate + 1
        if finished and finish_frames is not None:
            return 1e6 - finish_frames, len(self.gates)
        approach = 0.0 if best_dist_next == float("inf") else -min(best_dist_next, 200.0)
        return gates_passed * 1000.0 + approach, gates_passed


class CEM:
    def __init__(self, track: str, horizon_s: float = 45.0, seed: int = 0):
        self.track = track
        self.geom = load_track_geom(track)
        self.scorer = GateScorer(self.geom)
        self.T = int(horizon_s * 1000)  # 1 kHz
        self.rng = np.random.default_rng(seed)
        self.node = _NodeProc()
        self.examples = []  # list of dense (T,5) bool arrays
        self._load_examples()

    def _load_examples(self):
        for fx in load_fixtures():
            if fx.get("track", self.track) != self.track:
                continue
            rec = Recording.deserialize(fx["recording"])
            arr = rec_to_array(rec, self.T)
            self.examples.append((fx.get("frames", 1 << 30), arr))
        self.examples.sort(key=lambda x: x[0])
        print(f"[cem] {len(self.examples)} example laps for {self.track}, T={self.T} frames", flush=True)

    def mutate(self, base: np.ndarray, sigma_frames: float, p_toggle: float) -> np.ndarray:
        """Jitter toggle times: convert to toggle list, perturb, rebuild."""
        out = base.copy()
        T = out.shape[0]
        for j in range(len(CHANNELS)):
            col = out[:, j]
            # toggle frame indices
            toggles = np.flatnonzero(np.diff(col.astype(np.int8)) != 0) + 1
            if col[0]:
                toggles = np.concatenate([[0], toggles])
            if len(toggles) == 0:
                continue
            jitter = self.rng.normal(0, sigma_frames, size=len(toggles))
            new_toggles = np.clip(np.round(toggles + jitter), 0, T - 1).astype(int)
            new_toggles = np.unique(new_toggles)  # dedupe (collisions drop a toggle)
            # occasional insert/delete for structural variation
            if self.rng.random() < p_toggle and len(new_toggles) > 1:
                if self.rng.random() < 0.5:
                    new_toggles = np.delete(new_toggles, self.rng.integers(len(new_toggles)))
                else:
                    new_toggles = np.sort(np.append(new_toggles, self.rng.integers(T)))
            # rebuild column
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

    def score_batch(self, arrs: list[np.ndarray]) -> list[tuple[float, int, str]]:
        recordings = [array_to_rec(a) for a in arrs]
        resp = self.node.call(
            {
                "cmd": "score_batch",
                "track": self.track,
                "recordings": recordings,
                "sample_every": 200,  # state every 200 frames for scoring
                "max_frames": self.T + 2000,
            },
            timeout=300,
        )
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        out = []
        for arr, res in zip(arrs, resp["results"]):
            samples = [bytes.fromhex(h) for h in res["samples"]]
            score, gates = self.scorer.score(samples, res["finished"], res["finishFrames"])
            out.append((score, gates, array_to_rec(arr), res.get("finished"), res.get("finishFrames")))
        return out

    def run(self, gens: int, pop: int, elite_frac: float = 0.25, out: str | None = None):
        elite_n = max(2, int(pop * elite_frac))
        best_ever = None
        for g in range(gens):
            t0 = time.time()
            # Start with TINY jitter (a few ms) and ramp up if stuck: the seed
            # examples already finish, so early gens are polish, not search.
            # Ramp only kicks in if nothing finishes for a while.
            if best_ever is not None and best_ever[3] is not None:
                # something has finished — fine-tune around the champion
                sigma = max(2.0, 25.0 * (0.96 ** g))
            else:
                # nothing finishes yet — widen the search
                sigma = min(300.0, 8.0 * (1.06 ** g))

            # population: gen 0 = the raw examples themselves (so the champion
            # is a finisher from the start); later gens = the UNMUTATED champion
            # (score can only improve), champion mutations, and example mutations
            plans = []
            n_ex = len(self.examples)
            weights = np.array([1.0 / (i + 1) for i in range(n_ex)])
            weights /= weights.sum()
            if g == 0:
                plans = [arr.copy() for _, arr in self.examples]
                while len(plans) < pop:
                    i = self.rng.choice(n_ex, p=weights)
                    plans.append(self.mutate(self.examples[i][1], 10.0, 0.02))
            else:
                if best_ever is not None:
                    plans.append(best_ever[4].copy())  # always re-test the champion
                for _ in range(pop - len(plans)):
                    if best_ever is not None and self.rng.random() < 0.6:
                        plans.append(self.mutate(best_ever[4], sigma, 0.03))
                    else:
                        i = self.rng.choice(n_ex, p=weights)
                        plans.append(self.mutate(self.examples[i][1], sigma, 0.05))

            scored = self.score_batch(plans)
            scored.sort(key=lambda x: -x[0])

            top_idx = max(range(len(scored)), key=lambda i: scored[i][0])
            top_score, top_gates, top_rec, top_fin, top_ff = scored[top_idx]
            top_arr = plans[top_idx]

            lap = (1e6 - top_score) / 1000 if top_fin else None
            if best_ever is None or top_score > best_ever[0]:
                best_ever = (top_score, top_rec, top_gates, lap, top_arr)
                if out:
                    Path(out).parent.mkdir(exist_ok=True)
                    Path(out).write_text(json.dumps({
                        "track": self.track, "gen": g, "score": top_score, "gates": top_gates,
                        "lap_seconds": lap, "finish_frames": top_ff, "recording": top_rec,
                    }))
            gates_hist = {}
            for _, gt, *_ in scored:
                gates_hist[gt] = gates_hist.get(gt, 0) + 1
            print(
                f"[cem] gen {g}: best {top_score:.0f} gates {top_gates}/{len(self.scorer.gates)}"
                + (f" LAP {lap:.2f}s" if lap else "")
                + f" sigma {sigma:.0f} | dist {dict(sorted(gates_hist.items()))} | {time.time()-t0:.1f}s",
                flush=True,
            )
        return best_ever

    def close(self):
        self.node.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="summer1")
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--horizon", type=float, default=45.0)
    ap.add_argument("--out", default=str(REPO / "runs" / "cem_best.json"))
    args = ap.parse_args()

    cem = CEM(args.track, horizon_s=args.horizon)
    try:
        best = cem.run(args.gens, args.pop, out=args.out)
        if best:
            score, rec, gates, lap, _ = best
            print(f"[cem] BEST: score {score:.0f} gates {gates} lap {lap}s → {args.out}", flush=True)
    finally:
        cem.close()


if __name__ == "__main__":
    main()
