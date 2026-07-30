"""UEyeXS driver — structural tests that run WITHOUT hardware or the
IDS SDK. They pin the Device-protocol + capture surface the vision
server's CameraPool and MQTTDeviceAdapter rely on, and the graceful
degradation contract when pyueye is absent.

Hardware behavior (real grabs, focus sweeps, unplug recovery) is
verified on a unit with the camera attached — these tests only make
sure the CLASS keeps its shape.

Run:  sudo python3 -m pytest tests/test_ueye_xs.py -v
"""

import numpy as np
import pytest

from camera.ueye_xs import UEyeXS, _UEyeIntrinsics


def test_device_protocol_surface():
    """The pool + MQTT adapter duck-type these — presence is the contract."""
    u = UEyeXS()
    for attr in ("id", "state", "msg", "on_state_change",
                 "on_hardware_available", "recover", "release", "close",
                 "connect", "get_all", "camera_matrix", "dist_coeffs",
                 "get_K", "get_D"):
        assert hasattr(u, attr), f"missing Device-protocol attr: {attr}"
    # focus surface — what handlers.camera_focus branches on
    for attr in ("focus_apply", "focus_once", "focus_region", "focus_info"):
        assert hasattr(u, attr), f"missing focus attr: {attr}"


def test_initial_state_is_down():
    u = UEyeXS()
    assert u.state == "down"
    assert u.id is None


def test_state_listener_fires_on_real_change_only():
    u = UEyeXS()
    seen = []
    u.on_state_change(lambda s, m: seen.append((s, m)))
    u._set_state("down", "not connected")   # same state+msg -> no event
    assert seen == []
    u._set_state("down", "different msg")   # same state, new msg -> fires
    assert len(seen) == 1


def test_enumeration_without_sdk_is_empty_not_error():
    assert isinstance(UEyeXS.all_device(), list)


def test_connect_without_sdk_or_device_fails_honestly():
    u = UEyeXS()
    ok = u.connect(serial_number="0000000000", raise_on_fail=False)
    assert ok is False
    assert u.state == "down"
    assert u.msg   # actionable message, not empty
    with pytest.raises(RuntimeError):
        UEyeXS().connect(serial_number="0000000000", raise_on_fail=True)


def test_focus_apply_requires_connection():
    u = UEyeXS()
    with pytest.raises(RuntimeError):
        u.focus_apply({"mode": "continuous"})
    with pytest.raises(RuntimeError):
        u.focus_region([0, 0, 100, 100])


def test_focus_info_shape_when_down():
    info = UEyeXS().focus_info()
    assert set(info) >= {"supported", "range", "mode", "position"}
    assert info["supported"] is False


def test_intrinsics_shim_matches_camera_matrix_math():
    i = _UEyeIntrinsics(2592, 1944, 2729.0, 2729.0, 1296.0, 972.0)
    u = UEyeXS()
    K = u.camera_matrix(i)
    assert K.shape == (3, 3)
    assert K[0, 0] == pytest.approx(2729.0)
    assert K[0, 2] == pytest.approx(1296.0)
    D = u.dist_coeffs(i)
    assert np.allclose(D, np.zeros(5))
