"""HikRobot driver — structural tests that run WITHOUT hardware or the
MVS SDK. They pin the Device-protocol + capture surface the vision
server's CameraPool and MQTTDeviceAdapter rely on, and the graceful
degradation contract when the MVS runtime is absent.

Hardware behavior (real grabs over PoE, unplug recovery) is verified on
a unit with the camera attached — these tests only make sure the CLASS
keeps its shape.

Run:  python3 -m pytest tests/test_hik_robot.py -v
"""

import numpy as np
import pytest

from camera.hik_robot import HikRobot, _HikIntrinsics, _ip_to_str


def test_device_protocol_surface():
    """The pool + MQTT adapter duck-type these — presence is the contract."""
    c = HikRobot()
    for attr in ("id", "state", "msg", "on_state_change",
                 "on_hardware_available", "recover", "release", "close",
                 "connect", "get_all", "camera_matrix", "dist_coeffs",
                 "get_K", "get_D"):
        assert hasattr(c, attr), f"missing Device-protocol attr: {attr}"
    # exposure/wb surface — real knobs on this sensor
    for attr in ("get_exposure", "set_exposure", "auto_exposure",
                 "white_balance"):
        assert hasattr(c, attr), f"missing control attr: {attr}"


def test_initial_state_is_down():
    c = HikRobot()
    assert c.state == "down"
    assert c.id is None
    assert c._enabled_channels == {"color"}


def test_state_listener_fires_on_real_change_only():
    c = HikRobot()
    seen = []
    c.on_state_change(lambda s, m: seen.append((s, m)))
    c._set_state("down", "not connected")   # same state+msg -> no event
    assert seen == []
    c._set_state("down", "different msg")   # same state, new msg -> fires
    assert len(seen) == 1


def test_enumeration_without_sdk_is_empty_not_error():
    assert isinstance(HikRobot.all_device(), list)


def test_connect_without_sdk_or_device_fails_honestly():
    c = HikRobot()
    ok = c.connect(ip="10.255.255.1", raise_on_fail=False)
    assert ok is False
    assert c.state == "down"
    assert c.msg   # actionable message, not empty
    with pytest.raises(RuntimeError):
        HikRobot().connect(ip="10.255.255.1", raise_on_fail=True)


def test_exposure_requires_connection():
    c = HikRobot()
    with pytest.raises(RuntimeError):
        c.get_exposure()
    with pytest.raises(RuntimeError):
        c.set_exposure(10000)
    with pytest.raises(RuntimeError):
        c.white_balance({"auto": True})


def test_ip_decode():
    assert _ip_to_str(0x0A000132) == "10.0.1.50"
    assert _ip_to_str(0xC0A80101) == "192.168.1.1"


def test_intrinsics_shim_matches_camera_matrix_math():
    i = _HikIntrinsics(2448, 2048, 3500.0, 3500.0, 1224.0, 1024.0)
    c = HikRobot()
    K = c.camera_matrix(i)
    assert K.shape == (3, 3)
    assert K[0, 0] == pytest.approx(3500.0)
    assert K[0, 2] == pytest.approx(1224.0)
    D = c.dist_coeffs(i)
    assert np.allclose(D, np.zeros(5))
