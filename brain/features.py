"""Observation featurizer: CarState + TrackGeom → flat float vector.

Kept separate from env.py so BC (from dumped demo states) and online rollouts
see the exact same features. Order is part of the checkpoint contract.
"""

from __future__ import annotations

import math

from .env import CarState
from .tracklib import TrackGeom

# feature layout (fixed order):
#   0-2   position (x,y,z)
#   3-6   quaternion
#   7-9   forward vector (car frame → world)
#   10-12 up vector
#   13    speed kmh / 200 (normalized)
#   14    speed signed by forward progress (m/s-ish, /50)
#   15    steering (±1)
#   16-19 wheel contact (4 bool)
#   20    lateral offset from centerline (/10)
#   21    progress along track / total_len
#   22-24 vector to centerline point 25 m ahead (car frame, /100)
#   25-27 vector to centerline point 60 m ahead (car frame, /100)
#   28    heading angle to far lookahead (acos of dot with forward)
#   29    collision impulse (decayed) (/2000)
OBS_DIM = 30


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


def featurize(st: CarState, geom: TrackGeom, collision_decay: float = 0.0) -> list[float]:
    fwd = quat_rotate(st.quaternion, (0.0, 0.0, 1.0))
    up = quat_rotate(st.quaternion, (0.0, 1.0, 0.0))

    s = geom.progress(st.position)
    frac = s / geom.total_len if geom.total_len > 0 else 0.0

    def car_frame(vec):
        rx = quat_rotate(st.quaternion, (1.0, 0.0, 0.0))
        ry = up
        rz = fwd
        return (
            (vec[0] * rx[0] + vec[1] * ry[0] + vec[2] * rz[0]),
            (vec[0] * rx[1] + vec[1] * ry[1] + vec[2] * rz[1]),
            (vec[0] * rx[2] + vec[1] * ry[2] + vec[2] * rz[2]),
        )

    near = car_frame(geom.lookahead_point(s, 25.0))
    far = car_frame(geom.lookahead_point(s, 60.0))

    # curvature: angle between near and far lookahead directions
    dx = far[0] - near[0]
    dy = far[1] - near[1]
    dz = far[2] - near[2]
    seg_len = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
    dot = (dx * fwd[0] + dy * fwd[1] + dz * fwd[2]) / seg_len
    heading_cos = max(-1.0, min(1.0, dot))

    speed_norm = st.speed_kmh / 200.0
    signed_speed = speed_norm * (1.0 if dot >= 0 else -1.0)

    # lateral offset: distance from centerline (progress projection gives s;
    # offset ≈ distance from the projected point)
    # v0: use distance to near lookahead point as a proxy is wrong; instead
    # compute distance from segment — reuse geom.progress projection distance
    # by finding nearest segment point directly (cheap version):
    lat = _lateral_offset(geom, st.position)

    return [
        st.position[0] / 500.0,
        st.position[1] / 100.0,
        st.position[2] / 500.0,
        *st.quaternion,
        *(c / 1.0 for c in fwd),
        *(c / 1.0 for c in up),
        speed_norm,
        signed_speed,
        st.steering,
        *(1.0 if w else 0.0 for w in st.wheel_contact),
        lat / 10.0,
        frac,
        near[0] / 100.0,
        near[1] / 100.0,
        near[2] / 100.0,
        far[0] / 100.0,
        far[1] / 100.0,
        far[2] / 100.0,
        math.acos(heading_cos),
        collision_decay / 2000.0,
    ]


def _lateral_offset(geom: TrackGeom, pos) -> float:
    px, py, pz = pos
    best = float("inf")
    cl = geom.centerline
    for i in range(len(cl) - 1):
        ax, ay, az = cl[i]
        bx, by, bz = cl[i + 1]
        abx, aby, abz = bx - ax, by - ay, bz - az
        seg2 = abx * abx + aby * aby + abz * abz
        if seg2 < 1e-9:
            continue
        t = ((px - ax) * abx + (py - ay) * aby + (pz - az) * abz) / seg2
        t = 0.0 if t < 0 else (1.0 if t > 1 else t)
        cx, cy, cz = ax + t * abx, ay + t * aby, az + t * abz
        d2 = (px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2
        if d2 < best:
            best = d2
    return math.sqrt(best) if best != float("inf") else 0.0
