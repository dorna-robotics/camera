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

import camera.hik_robot as hik
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


def _gige_struct(ip, nic="10.0.1.40"):
    """Shape-only stand-in for MV_CC_DEVICE_INFO — just the fields
    _ensure_routable reads."""
    from types import SimpleNamespace as NS
    return NS(SpecialInfo=NS(stGigEInfo=NS(
        nCurrentIp=hik._ip_to_int(ip), nNetExport=hik._ip_to_int(nic))))


def _dev(ip):
    return {"serial_number": "SN1", "ip": ip, "name": "MV-CU060-10GC",
            "user_name": "", "camera_type": "hikrobot"}


@pytest.fixture
def stranded(monkeypatch):
    """A camera discovered on 169.254.x from a 10.0.1.40/24 host, with
    the SDK replaced by a recorder. Force-IP calls land in ``forced``,
    DeviceReset in ``resets``; discovery follows the camera: at the
    forced address after a Force-IP, absent for a beat after a reset,
    then back at ``after_reset`` (what DHCP handed it on reboot)."""
    from types import SimpleNamespace
    log = SimpleNamespace(forced=[], resets=[], after_reset="10.0.1.50",
                          reset_ok=True, gone=0)

    class FakeCam:
        def MV_CC_CreateHandle(self, st):
            return 0

        def MV_GIGE_ForceIpEx(self, ip, mask, gw):
            log.forced.append((hik._ip_to_str(ip), hik._ip_to_str(mask)))
            log.forced_since_reset = True
            return 0

        def MV_CC_OpenDevice(self, mode, key):
            return 0 if log.reset_ok else 0x80000203

        def MV_CC_SetCommandValue(self, name):
            assert name == "DeviceReset"
            log.resets.append(name)
            log.forced_since_reset = False
            log.gone = 2            # discovery misses the next two polls
            return 0

        def MV_CC_CloseDevice(self):
            return 0

        def MV_CC_DestroyHandle(self):
            return 0

    def enum():
        if log.gone:
            log.gone -= 1
            return []
        if log.resets and not log.forced_since_reset:
            ip = log.after_reset        # what DHCP handed it on reboot
        elif log.forced:
            ip = log.forced[-1][0]      # the latest Force-IP
        else:
            return []
        return [(_gige_struct(ip), _dev(ip))]

    monkeypatch.setattr(hik, "_mv", SimpleNamespace(MvCamera=FakeCam,
                                                    MV_ACCESS_Exclusive=1))
    monkeypatch.setattr(hik.time, "sleep", lambda s: None)
    monkeypatch.setattr(hik, "_nic_mask_for", lambda nic: 0xFFFFFF00)
    monkeypatch.setattr(hik, "_ping", lambda ip, timeout_s=1: ip == "10.0.1.7")
    monkeypatch.setattr(HikRobot, "_enum_raw", staticmethod(enum))
    return log


def _rescue(want="10.0.1.50"):
    return HikRobot()._ensure_routable(
        _gige_struct("169.254.70.141"), _dev("169.254.70.141"), want)


def test_routable_camera_is_left_alone(stranded):
    st, d = HikRobot()._ensure_routable(
        _gige_struct("10.0.1.77"), _dev("10.0.1.77"), "10.0.1.50")
    assert d["ip"] == "10.0.1.77"      # DHCP gave it .77 — that works, keep it
    assert stranded.forced == [] and stranded.resets == []


def test_stranded_camera_is_forced_then_rebooted_onto_its_lease(stranded):
    stranded.after_reset = "10.0.1.50"          # router hands out the reservation
    st, d = _rescue()
    assert stranded.forced == [("10.0.1.50", "255.255.255.0")]
    assert stranded.resets == ["DeviceReset"]
    assert d["ip"] == "10.0.1.50"
    assert st.SpecialInfo.stGigEInfo.nCurrentIp == hik._ip_to_int("10.0.1.50")


def test_reboot_accepts_whatever_lease_dhcp_gives(stranded):
    stranded.after_reset = "10.0.1.77"          # no reservation at this site
    st, d = _rescue()
    assert d["ip"] == "10.0.1.77"               # on-subnet: use it, don't re-force
    assert len(stranded.forced) == 1


def test_dhcp_still_dead_after_reboot_forces_once_more_never_loops(stranded):
    stranded.after_reset = "169.254.3.3"
    st, d = _rescue()
    assert d["ip"] == "10.0.1.50"
    assert [f[0] for f in stranded.forced] == ["10.0.1.50", "10.0.1.50"]
    assert stranded.resets == ["DeviceReset"]   # exactly one reset


def test_reset_refused_keeps_the_forced_address(stranded):
    stranded.reset_ok = False                   # open for reset denied
    st, d = _rescue()
    assert d["ip"] == "10.0.1.50" and stranded.resets == []


@pytest.mark.parametrize("want, needle", [
    ("", "pass ip="),                            # nothing to move it to
    ("10.0.2.50", "not on this host's subnet"),  # wrong subnet requested
    ("10.0.1.7", "already answers"),             # address taken (pings)
])
def test_stranded_camera_errors_say_why(stranded, want, needle):
    with pytest.raises(RuntimeError) as ex:
        HikRobot()._ensure_routable(
            _gige_struct("169.254.70.141"), _dev("169.254.70.141"), want)
    assert "169.254.70.141" in str(ex.value)
    assert needle in str(ex.value)
    assert stranded.forced == [] and stranded.resets == []   # never on an error path


def test_nic_mask_falls_back_to_slash_24():
    # a NIC address the OS does not have -> /24, never an exception
    assert hik._nic_mask_for("192.0.2.9") == 0xFFFFFF00


def test_intrinsics_shim_matches_camera_matrix_math():
    i = _HikIntrinsics(2448, 2048, 3500.0, 3500.0, 1224.0, 1024.0)
    c = HikRobot()
    K = c.camera_matrix(i)
    assert K.shape == (3, 3)
    assert K[0, 0] == pytest.approx(3500.0)
    assert K[0, 2] == pytest.approx(1224.0)
    D = c.dist_coeffs(i)
    assert np.allclose(D, np.zeros(5))
