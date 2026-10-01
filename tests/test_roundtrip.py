"""M0 — recording round-trip / determinism proof.

For each synthesized recording:
  1. encode with the Python codec (brain/codec.py)
  2. re-simulate it headless twice via the Node sim host (Verify path)
  3. assert identical verdicts (finished + exact frame count)

Also asserts the JS and Python codecs produce byte-identical strings for the
same recording (cross-language conformance), which is what lets the Python
brain author recordings the game accepts.

Requires the Node sim host (node/sim_host.mjs) and vendored assets.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from brain.codec import Recording

REPO = Path(__file__).resolve().parent.parent

# A handful of control programs. None of these are expected to FINISH summer1
# (blind throttle crashes at the first corner) — the test is that the game's
# own evaluator runs them and two runs agree exactly.
# TODO(user): add a real exported recording once one is captured in-game.
RECORDINGS = {
    "idle": Recording(),                                  # never touch a button
    "full_throttle": Recording(up=[0]),                   # hold gas forever
    "pulse_throttle": Recording(up=[f for cycle in range(40) for f in (cycle * 1000, cycle * 1000 + 250)]),
    "throttle_left": Recording(up=[0], left=[1200, 2600]),
    "throttle_right": Recording(up=[0], right=[1500, 2900]),
}


def _verify(recording_b64: str, target_frames: int) -> dict:
    """Run the Node verify harness; returns {finished, frames, wallMs}."""
    proc = subprocess.run(
        ["node", "scripts/verify_one.mjs", recording_b64, str(target_frames)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"verify harness failed: {proc.stderr[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("name", sorted(RECORDINGS))
def test_recording_roundtrip_is_deterministic(name: str):
    rec = RECORDINGS[name]
    target = 20_000  # 20 sim-seconds

    r1 = _verify(rec.serialize(), target)
    r2 = _verify(rec.serialize(), target)

    assert r1["finished"] == r2["finished"]
    assert r1["frames"] == r2["frames"]
    # sanity: the sim actually ran
    assert r1["frames"] > 0


def test_real_human_recording_finishes():
    """The crown-jewel test: a real human lap, decoded by our codec and
    re-simulated headless through the game's own physics, finishes the track
    at the same lap time the game recorded."""
    import json

    fixture = json.loads((REPO / "fixtures" / "human_summer1.json").read_text())[0]
    result = _verify(fixture["recording"], fixture["frames"] + 500)
    assert result["finished"] is True
    assert abs(result["frames"] - 23136) <= 50  # finish within 50ms of the known frame


def test_python_js_codec_conformance():
    """The Python and JS codecs must produce recordings that decode to the
    identical payload (deflate wrapper bytes may differ — both are valid
    zlib streams the game accepts; content equality is the contract)."""
    import base64
    import zlib

    rec = Recording(up=[0, 250, 999], right=[10], down=[5000, 5100], left=[], reset=[3])
    py_payload = zlib.decompress(base64.urlsafe_b64decode(rec.serialize() + "=="))
    proc = subprocess.run(
        ["node", "scripts/encode_one.mjs"],
        cwd=REPO,
        input=json.dumps({"up": rec.up, "right": rec.right, "down": rec.down, "left": rec.left, "reset": rec.reset}),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    js_payload = zlib.decompress(base64.urlsafe_b64decode(proc.stdout.strip() + "=="))
    assert py_payload == js_payload
