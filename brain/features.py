"""Observation featurizer: CarState + reward path → flat float vector.

tmrl's lesson: keep the policy's input LOCAL and reactive (speed, attitude,
wheels, a short lookahead of the upcoming path) — the global map lives in the
reward, not the observation. The one addition tmrl wishes it had: a few
lookahead points of the reward polyline relative to the car, so the policy can
learn to trail-brake into a corner instead of discovering it through the value
chain alone. We have ground truth, so we take it.

Order is part of the checkpoint contract. OBS_DIM must match.
"""

from __future__ import annotations

import math

import numpy as np

from .env import CarState

# feature layout (fixed order):
#   0-2   position (x,y,z) / normalizers
#   3-6   quaternion
#   7-9   forward vector (car → world)
#   10-12 up vector
#   13    speed kmh / 200
#   14    steering (±1)
#   15-18 wheel contact (4 bool)
#   19    collision impulse (decayed) /2000
#   20    path progress fraction (cur_idx / n)
#   21-23 lookahead point at +10 m, car frame (/50)
#   24-26 lookahead point at +25 m, car frame (/50)
#   27-29 lookahead point at +50 m, car frame (/50)
#   30    curvature heading into +25 m (signed turn angle, rad/π)
OBS_DIM = 31


def quat_rotate(q, v):
    x, y, z = v
    qx, qy, qz, qw = q
    uvx = qy * z - qz * y
    uvy = qz * x - qx * z
    uvz = qx * y - qy * x
    uuvx = qy * uvz - qz * uvy
    uuvy = qz * uvx - qx * uvz
    uuvz = qx * uvy - qy * uvx
    return (
        x + 2 * (qw * uvx + uuvx),
        y + 2 * (qw * uvy + uuvy),
        z + 2 * (qw * uvz + uuvz),
    )


def _path_point_at(path: np.ndarray, idx: int) -> np.ndarray:
    return path[min(idx, len(path) - 1)]


def featurize(
    st: CarState,
    path: np.ndarray,
    cur_idx: int,
    collision_decay: float = 0.0,
) -> list[float]:
    """featurize a car state against the reward path (cursor at cur_idx)."""
    fwd = quat_rotate(st.quaternion, (0.0, 0.0, 1.0))
    up = quat_rotate(st.quaternion, (0.0, 1.0, 0.0))
    right = quat_rotate(st.quaternion, (1.0, 0.0, 0.0))
    pos = np.asarray(st.position, dtype=np.float64)

    step = 0.1  # path spacing in meters
    n = len(path)

    def lookahead(dist_m: float) -> np.ndarray:
        target = _path_point_at(path, cur_idx + int(dist_m / step))
        rel = target - pos
        # into car frame
        return np.array([
            rel @ np.asarray(right),
            rel @ np.asarray(up),
            rel @ np.asarray(fwd),
        ])

    la10 = lookahead(10.0)
    la25 = lookahead(25.0)
    la50 = lookahead(50.0)

    # curvature: signed angle between the +10m and +50m directions (in the
    # car's forward-right plane)
    a = la10[[2, 0]]  # (fwd, right) components
    b = la50[[2, 0]]
    curv = 0.0
    if np.linalg.norm(a) > 1e-6 and np.linalg.norm(b) > 1e-6:
        cos = np.clip((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)
        cross = a[0] * b[1] - a[1] * b[0]
        curv = math.atan2(cross, cos) / math.pi  # signed, ±1

    return [
        st.position[0] / 500.0,
        st.position[1] / 100.0,
        st.position[2] / 500.0,
        *st.quaternion,
        *fwd,
        *up,
        st.speed_kmh / 200.0,
        st.steering,
        *(1.0 if w else 0.0 for w in st.wheel_contact),
        collision_decay / 2000.0,
        cur_idx / max(1, n - 1),
        la10[0] / 50.0, la10[1] / 50.0, la10[2] / 50.0,
        la25[0] / 50.0, la25[1] / 50.0, la25[2] / 50.0,
        la50[0] / 50.0, la50[1] / 50.0, la50[2] / 50.0,
        curv,
    ]
