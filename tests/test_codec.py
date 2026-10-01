"""Codec unit tests — round-trips and game-format conformance."""

from __future__ import annotations

import base64
import zlib

import pytest

from brain.codec import CHANNELS, Recording


def test_empty_roundtrip():
    r = Recording()
    s = r.serialize()
    r2 = Recording.deserialize(s)
    assert r2 == r


def test_single_press():
    # up pressed at frame 100, released at 200
    r = Recording(up=[100, 200])
    assert r.get_frame(99)["up"] is False
    assert r.get_frame(100)["up"] is True
    assert r.get_frame(199)["up"] is True
    assert r.get_frame(200)["up"] is False
    r2 = Recording.deserialize(r.serialize())
    assert r2 == r


def test_multichannel_roundtrip():
    r = Recording(up=[0, 500, 900], right=[10, 20], down=[1000], left=[], reset=[3])
    r2 = Recording.deserialize(r.serialize())
    assert r2 == r
    d = r.to_dense(1001)
    assert d[0]["up"] is True
    assert d[10]["right"] is True and d[20]["right"] is False
    assert d[999]["down"] is False and d[1000]["down"] is True
    assert d[500]["up"] is False  # toggled off
    assert d[900]["up"] is True   # toggled on again
    assert d[3]["reset"] is True and d[1000]["reset"] is True


def test_dense_roundtrip():
    r = Recording(up=[5, 6, 7, 8], left=[1], down=[2, 4])
    dense = r.to_dense(10)
    r2 = Recording.from_dense(dense)
    assert r2 == r
    assert r2.to_dense(10) == dense


def test_wire_layout():
    # manually check byte layout: up=[0x010203], count=1
    r = Recording(up=[0x010203])
    raw = r.to_bytes()
    # up: count=1, frame=0x010203 LE → 01 00 00 03 02 01; others count=0
    assert raw[:6] == bytes([1, 0, 0, 0x03, 0x02, 0x01])
    assert raw[6:] == bytes(3 * 4)  # 4 channels × 3-byte zero counts


def test_deserialize_base64url_alphabet():
    r = Recording(up=list(range(0, 3000, 3)))  # many toggles → deflate output hits +/ in standard b64
    s = r.serialize()
    assert "+" not in s and "/" not in s and "=" not in s
    assert Recording.deserialize(s) == r


def test_deserialize_rejects_garbage():
    with pytest.raises(ValueError):
        Recording.deserialize("!!!not-base64!!!")
    with pytest.raises(ValueError):
        # valid b64 but not deflate
        Recording.deserialize(base64.urlsafe_b64encode(b"hello world").decode())


def test_out_of_range_frame():
    r = Recording()
    with pytest.raises(ValueError):
        r.up.append(0x1000000)  # > u24
        r.serialize()


def test_frame_boundaries_binary_search():
    r = Recording(up=[1000, 2000, 3000])
    for f, want in [(0, False), (999, False), (1000, True), (1999, True), (2000, False), (3000, True)]:
        assert r.get_frame(f)["up"] is want
