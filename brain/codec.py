"""PolyTrack recording codec — Python port of the game's `Ta` class
(simulation_worker.bundle.js).

Format: 5 boolean channels (up, right, down, left, reset). Per channel, a
list of 3-byte little-endian frame numbers where the channel TOGGLES (starts
False). Serialized layout:

    [3-byte count][3-byte frame]*   (up)
    [3-byte count][3-byte frame]*   (right)
    ... down, left, reset

then zlib-deflate (raw bytes, pako level 9 compatible — any zlib stream
works for decode; we encode with zlib level 9), then base64url without
padding, with '+'→'-' and '/'→'_' (base64.urlsafe already does this).

1 frame = 1 ms of simulation time (fixed 1 kHz tick).
"""

from __future__ import annotations

import base64
import zlib
from dataclasses import dataclass, field

CHANNELS = ("up", "right", "down", "left", "reset")
_MAX_U24 = 0xFFFFFF


def _u24(n: int) -> bytes:
    if not 0 <= n <= _MAX_U24:
        raise ValueError(f"frame/count out of u24 range: {n}")
    return bytes((n & 0xFF, (n >> 8) & 0xFF, (n >> 16) & 0xFF))


def _read_u24(buf: bytes, off: int) -> tuple[int, int]:
    if off + 3 > len(buf):
        raise ValueError("recording truncated")
    return buf[off] | (buf[off + 1] << 8) | (buf[off + 2] << 16), off + 3


@dataclass
class Recording:
    """Toggle-frame recording. Each channel list holds ascending frame numbers
    at which that channel toggles state (initial state is all-False)."""

    up: list[int] = field(default_factory=list)
    right: list[int] = field(default_factory=list)
    down: list[int] = field(default_factory=list)
    left: list[int] = field(default_factory=list)
    reset: list[int] = field(default_factory=list)

    # -- frame access --------------------------------------------------------

    def get_frame(self, frame: int) -> dict[str, bool]:
        """Button states at a frame. A channel is True at frame f iff the
        number of toggle frames <= f is odd."""
        out = {}
        for ch in CHANNELS:
            toggles = getattr(self, ch)
            # count of toggles <= frame (list is ascending)
            lo, hi = 0, len(toggles)
            while lo < hi:
                mid = (lo + hi) // 2
                if toggles[mid] <= frame:
                    lo = mid + 1
                else:
                    hi = mid
            out[ch] = (lo % 2) == 1
        return out

    def record_frame(self, frame: int, state: dict[str, bool]) -> None:
        """Append a frame observation, recording toggles (mirrors the game's
        recordFrame). Frames must be strictly increasing when state changes."""
        for ch in CHANNELS:
            toggles = getattr(self, ch)
            current = (len(toggles) % 2) == 1
            if bool(state[ch]) != current:
                toggles.append(frame)

    @property
    def num_frames(self) -> int:
        """Total recorded frames (last toggle frame + 1; 0 if empty)."""
        last = 0
        for ch in CHANNELS:
            t = getattr(self, ch)
            if t:
                last = max(last, t[-1])
        return last + 1 if last else 0

    # -- serialization -------------------------------------------------------

    def to_bytes(self) -> bytes:
        out = bytearray()
        for ch in CHANNELS:
            toggles = getattr(self, ch)
            if len(toggles) > _MAX_U24:
                raise ValueError(f"channel {ch} has too many toggles: {len(toggles)}")
            out += _u24(len(toggles))
            for f in toggles:
                out += _u24(f)
        return bytes(out)

    @classmethod
    def from_bytes(cls, raw: bytes) -> "Recording":
        rec = cls()
        off = 0
        for ch in CHANNELS:
            count, off = _read_u24(raw, off)
            toggles: list[int] = []
            for _ in range(count):
                f, off = _read_u24(raw, off)
                toggles.append(f)
            if toggles != sorted(toggles):
                raise ValueError(f"channel {ch} toggle frames not ascending")
            setattr(rec, ch, toggles)
        return rec

    def serialize(self) -> str:
        """Game-format recording string (deflate9 + base64url, no padding)."""
        comp = zlib.compress(self.to_bytes(), 9)
        return base64.urlsafe_b64encode(comp).decode("ascii").rstrip("=")

    @classmethod
    def deserialize(cls, s: str) -> "Recording":
        """Parse a game recording string. Raises ValueError on bad input
        (the game returns null; we raise — callers decide)."""
        s = s.strip()
        # restore padding
        pad = (-len(s)) % 4
        try:
            comp = base64.urlsafe_b64decode(s + "=" * pad)
        except Exception as e:
            raise ValueError(f"recording is not valid base64url: {e}") from e
        try:
            raw = zlib.decompress(comp)
        except zlib.error as e:
            raise ValueError(f"recording deflate stream invalid: {e}") from e
        return cls.from_bytes(raw)

    # -- dense conversion ----------------------------------------------------

    def to_dense(self, num_frames: int | None = None) -> list[dict[str, bool]]:
        """Expand to per-frame button states. Defaults to num_frames."""
        n = num_frames if num_frames is not None else self.num_frames
        # pointer walk, O(n + toggles)
        idx = {ch: 0 for ch in CHANNELS}
        state = {ch: False for ch in CHANNELS}
        out: list[dict[str, bool]] = []
        for f in range(n):
            for ch in CHANNELS:
                toggles = getattr(self, ch)
                while idx[ch] < len(toggles) and toggles[idx[ch]] <= f:
                    state[ch] = not state[ch]
                    idx[ch] += 1
            out.append(dict(state))
        return out

    @classmethod
    def from_dense(cls, frames: list[dict[str, bool]]) -> "Recording":
        rec = cls()
        for f, st in enumerate(frames):
            rec.record_frame(f, st)
        return rec
