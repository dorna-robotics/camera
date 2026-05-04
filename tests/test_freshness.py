"""Tests for Camera._validate_freshness.

Covers all three layered checks (frame number monotonicity, device clock
advancement, wall vs device clock drift) and the baseline reset on
pipeline disruption. Uses tiny fake frames; no librealsense / hardware
required, so the suite runs anywhere.
"""

from __future__ import annotations

import pytest

from camera import Camera


class _FakeFrames:
    """Minimal stand-in for the librealsense frames object that
    ``_validate_freshness`` reads — only ``get_frame_number`` and
    ``get_timestamp`` are touched."""

    def __init__(self, *, frame_number: int, timestamp_ms: float):
        self._fn = frame_number
        self._ts = timestamp_ms

    def get_frame_number(self):
        return self._fn

    def get_timestamp(self):
        return self._ts


def _frozen_clock(t_ref):
    """Build a context that freezes ``time.time`` so wall-clock checks
    are deterministic. Returns a setter so the test can advance the
    fake clock by hand."""

    state = {"now": t_ref}
    return state


def _patch_time(monkeypatch, state):
    import camera.camera as cam_mod

    monkeypatch.setattr(cam_mod.time, "time", lambda: state["now"])


# ── First-frame baseline ──────────────────────────────────────────────────


def test_first_frame_initializes_baseline(monkeypatch):
    state = _frozen_clock(1000.0)
    _patch_time(monkeypatch, state)
    cam = Camera()

    cam._validate_freshness(_FakeFrames(frame_number=42, timestamp_ms=500.0))

    assert cam._last_frame_number == 42
    assert cam._last_frame_ts == 500.0
    assert cam._last_wall_ts == 1000.0


# ── Frame-number monotonicity ─────────────────────────────────────────────


def test_same_frame_number_raises(monkeypatch):
    state = _frozen_clock(1000.0)
    _patch_time(monkeypatch, state)
    cam = Camera()

    cam._validate_freshness(_FakeFrames(frame_number=10, timestamp_ms=100.0))
    state["now"] = 1000.1
    with pytest.raises(RuntimeError, match="frame_number 10 not after 10"):
        cam._validate_freshness(_FakeFrames(frame_number=10, timestamp_ms=200.0))


def test_decreasing_frame_number_raises(monkeypatch):
    state = _frozen_clock(1000.0)
    _patch_time(monkeypatch, state)
    cam = Camera()

    cam._validate_freshness(_FakeFrames(frame_number=10, timestamp_ms=100.0))
    state["now"] = 1000.1
    with pytest.raises(RuntimeError, match="frame_number 9 not after 10"):
        cam._validate_freshness(_FakeFrames(frame_number=9, timestamp_ms=200.0))


def test_advancing_frame_number_passes(monkeypatch):
    state = _frozen_clock(1000.0)
    _patch_time(monkeypatch, state)
    cam = Camera()

    cam._validate_freshness(_FakeFrames(frame_number=10, timestamp_ms=100.0))
    state["now"] = 1000.1
    cam._validate_freshness(_FakeFrames(frame_number=11, timestamp_ms=200.0))
    assert cam._last_frame_number == 11


# ── Device-clock advancement ──────────────────────────────────────────────


def test_same_device_timestamp_raises(monkeypatch):
    state = _frozen_clock(1000.0)
    _patch_time(monkeypatch, state)
    cam = Camera()

    cam._validate_freshness(_FakeFrames(frame_number=10, timestamp_ms=100.0))
    state["now"] = 1000.1
    with pytest.raises(RuntimeError, match="device timestamp 100"):
        cam._validate_freshness(_FakeFrames(frame_number=11, timestamp_ms=100.0))


def test_decreasing_device_timestamp_raises(monkeypatch):
    state = _frozen_clock(1000.0)
    _patch_time(monkeypatch, state)
    cam = Camera()

    cam._validate_freshness(_FakeFrames(frame_number=10, timestamp_ms=200.0))
    state["now"] = 1000.1
    with pytest.raises(RuntimeError, match="device timestamp 100"):
        cam._validate_freshness(_FakeFrames(frame_number=11, timestamp_ms=100.0))


# ── Wall-vs-device drift ──────────────────────────────────────────────────


def test_drift_under_two_seconds_does_not_trip(monkeypatch):
    """Short windows (<2s wall) skip the drift check — protects against
    USB-jitter false-positives at high call frequency."""
    state = _frozen_clock(1000.0)
    _patch_time(monkeypatch, state)
    cam = Camera()

    cam._validate_freshness(_FakeFrames(frame_number=10, timestamp_ms=100.0))
    # 1.5s wall, only 50ms device — would trip drift check if it ran,
    # but window is too short.
    state["now"] = 1001.5
    cam._validate_freshness(_FakeFrames(frame_number=11, timestamp_ms=150.0))


def test_drift_caught_after_long_wall_window(monkeypatch):
    """≥2s wall + <1s device advance = sustained underclocking; raise."""
    state = _frozen_clock(1000.0)
    _patch_time(monkeypatch, state)
    cam = Camera()

    cam._validate_freshness(_FakeFrames(frame_number=10, timestamp_ms=100.0))
    state["now"] = 1003.0   # +3s wall
    # device only advanced 0.4s — way behind wall clock
    with pytest.raises(RuntimeError, match="advanced only 0.40s while wall advanced 3.00s"):
        cam._validate_freshness(_FakeFrames(frame_number=20, timestamp_ms=500.0))


def test_drift_within_tolerance_passes(monkeypatch):
    """Device clock that keeps up with wall time within slack — fine."""
    state = _frozen_clock(1000.0)
    _patch_time(monkeypatch, state)
    cam = Camera()

    cam._validate_freshness(_FakeFrames(frame_number=10, timestamp_ms=0.0))
    state["now"] = 1003.0   # +3s wall
    # device advanced 2.5s — close to wall, well above the 1s floor
    cam._validate_freshness(_FakeFrames(frame_number=50, timestamp_ms=2500.0))


# ── Baseline reset on pipeline disruption ─────────────────────────────────


def test_set_state_to_down_resets_baseline(monkeypatch):
    state = _frozen_clock(1000.0)
    _patch_time(monkeypatch, state)
    cam = Camera()

    cam._validate_freshness(_FakeFrames(frame_number=999, timestamp_ms=99999.0))
    cam._set_state("down", "USB unplugged")
    assert cam._last_frame_number == -1
    assert cam._last_frame_ts == 0.0
    assert cam._last_wall_ts == 0.0

    # Fresh post-reset frame must be accepted even with a much smaller
    # frame number (pipeline restart can reset the device counter).
    state["now"] = 1010.0
    cam._validate_freshness(_FakeFrames(frame_number=1, timestamp_ms=33.0))
    assert cam._last_frame_number == 1


def test_ok_to_ok_does_not_reset_baseline(monkeypatch):
    state = _frozen_clock(1000.0)
    _patch_time(monkeypatch, state)
    cam = Camera()

    cam._validate_freshness(_FakeFrames(frame_number=10, timestamp_ms=100.0))
    cam.state = "ok"
    cam._set_state("ok", "msg-only update")
    assert cam._last_frame_number == 10   # preserved across same-state msg change


def test_recovering_resets_baseline(monkeypatch):
    state = _frozen_clock(1000.0)
    _patch_time(monkeypatch, state)
    cam = Camera()

    cam._validate_freshness(_FakeFrames(frame_number=500, timestamp_ms=5000.0))
    cam._set_state("recovering", "rebuilding pipeline")
    assert cam._last_frame_number == -1   # next "ok" frame establishes a new baseline
