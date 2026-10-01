"""Track geometry: parse a PolyTrack track into checkpoints + centerline.

The track body (after the save/export string is decoded by node/track_codec.mjs)
is the binary payload parsed by the game's `wo` function:

    theme u8 | sunReps u8 | offsetX i32 | offsetY i32 | offsetZ i32
    | packed u8 (h,c,A coord byte widths) | sections…

We don't parse it in Python — instead `tracklib` consumes the JSON dump
produced by `scripts/track_geom.mjs` (Node, which already has the codec +
part configs). That dump gives, per placed part: {type, typeId, x, y, z,
rotation, rotationAxis, checkpointOrder, startOrder} with detector boxes
resolved from part_configs.json.

From that we build:
  - ordered checkpoints (by checkpointOrder) with world-space center + size
  - the finish gate
  - an approximate centerline polyline: start → cp0 → cp1 → … → finish
  - progress projection: s(position) = arclength along that polyline

v0 heuristic — the polyline is straight segments between gates. Good enough
for reward shaping on mostly-line-of-sight tracks; refine with mesh sampling
later if a track's line matters.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# game constant (main.bundle.js): partSize
PART_SIZE = 4.0  # world units per tile (deduced: checkpoint centers × partSize)


@dataclass
class Gate:
    index: int
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    is_finish: bool


@dataclass
class TrackGeom:
    name: str
    track_hash: str
    start: tuple[float, float, float]
    start_quat: tuple[float, float, float, float]
    gates: list[Gate]  # ordered: checkpoints then finish
    centerline: list[tuple[float, float, float]]
    cum_len: list[float]  # cumulative arclength at each centerline vertex
    total_len: float

    def progress(self, pos: tuple[float, float, float]) -> float:
        """Project position onto the centerline → arclength s in [0, total_len]."""
        px, py, pz = pos
        best_s, best_d2 = 0.0, float("inf")
        for i in range(len(self.centerline) - 1):
            ax, ay, az = self.centerline[i]
            bx, by, bz = self.centerline[i + 1]
            abx, aby, abz = bx - ax, by - ay, bz - az
            seg_len2 = abx * abx + aby * aby + abz * abz
            if seg_len2 < 1e-9:
                continue
            t = ((px - ax) * abx + (py - ay) * aby + (pz - az) * abz) / seg_len2
            t = 0.0 if t < 0 else (1.0 if t > 1 else t)
            cx, cy, cz = ax + t * abx, ay + t * aby, az + t * abz
            d2 = (px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_s = self.cum_len[i] + t * math.sqrt(seg_len2)
        return best_s

    def lookahead_point(self, s: float, ahead: float) -> tuple[float, float, float]:
        """Point on the centerline at arclength s+ahead (clamped)."""
        target = min(s + ahead, self.total_len)
        # binary search cum_len
        lo, hi = 0, len(self.cum_len) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.cum_len[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        i = max(1, lo)
        s0, s1 = self.cum_len[i - 1], self.cum_len[i]
        t = 0.0 if s1 - s0 < 1e-9 else (target - s0) / (s1 - s0)
        ax, ay, az = self.centerline[i - 1]
        bx, by, bz = self.centerline[i]
        return (ax + t * (bx - ax), ay + t * (by - ay), az + t * (bz - az))


@lru_cache(maxsize=32)
def load_track_geom(track_name: str) -> TrackGeom:
    """Extract geometry for an official track via the Node dumper."""
    proc = subprocess.run(
        ["node", "scripts/track_geom.mjs", track_name],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"track_geom failed for {track_name}: {proc.stderr[-2000:]}")
    data = json.loads(proc.stdout)

    gates = [
        Gate(
            index=g["order"],
            center=tuple(g["center"]),
            size=tuple(g["size"]),
            is_finish=g["isFinish"],
        )
        for g in data["gates"]
    ]
    centerline = [tuple(p) for p in data["centerline"]]
    cum = [0.0]
    for i in range(1, len(centerline)):
        d = math.dist(centerline[i - 1], centerline[i])
        cum.append(cum[-1] + d)
    return TrackGeom(
        name=track_name,
        track_hash=data["trackHash"],
        start=tuple(data["start"]["position"]),
        start_quat=tuple(data["start"]["quaternion"]),
        gates=gates,
        centerline=centerline,
        cum_len=cum,
        total_len=cum[-1] if cum else 0.0,
    )
