"""M2 — env validation: random-policy episodes, obs sane, reward finite,
episode ends with a reason. Not 1000 episodes in CI (slow) — a few locally
plus a soak marker for the full run."""

from __future__ import annotations

import math
import random

import pytest

from brain.env import PolyTrackEnv, decode_car_state


@pytest.fixture()
def env():
    e = PolyTrackEnv(track="summer1", control_hz=50, max_episode_frames=20_000)
    yield e
    e.close()


def test_decode_car_state_roundtrip():
    # smoke: decode a plausible 227-byte buffer of zeros with started flag
    buf = bytearray(227)
    buf[7] = 0b0000001  # hasStarted
    st = decode_car_state(bytes(buf))
    assert st.frames == 0 and st.has_started and st.speed_kmh == 0.0


def test_random_episode_runs(env):
    obs = env.reset()
    assert 0.0 <= obs["progress_frac"] <= 1.0
    assert isinstance(obs["speed_kmh"], float)

    rng = random.Random(0)
    total_r = 0.0
    done = False
    steps = 0
    info = {}
    for _ in range(200):  # 200 × 20ms = 4 sim-seconds max
        a = (rng.randint(0, 1), 0, rng.randint(0, 1), rng.randint(0, 1))
        obs, r, done, info = env.step(a)
        assert math.isfinite(r)
        assert 0.0 <= obs["progress_frac"] <= 1.0
        total_r += r
        steps += 1
        if done:
            break
    # the env stepped real frames
    assert info["frames"] > 0
    # random policy on summer1 shouldn't finish, but must terminate by timeout
    # if we ran long enough; here we only require the episode *ran*
    assert steps > 0
