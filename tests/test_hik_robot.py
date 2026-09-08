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


def test_bandwidth_to_packet_delay_matches_bench():
    # MV-CU060-10GC: 100 MHz tick, 1500 B packets. Bench: 300 Mbps ->
    # 2770 ticks (6.5 fps, two cameras at 686 Mbps, zero loss).
    assert hik._scpd_for(300, 1500, 100_000_000) == 2770
    assert hik._scpd_for(400, 1500, 100_000_000) == 1770
    assert hik._scpd_for(0, 1500, 100_000_000) == 0        # 0 = unpaced
    assert hik._scpd_for(None, 1500, 100_000_000) == 0
    assert hik._scpd_for(5000, 1500, 100_000_000) == 0     # above line rate: no gap


def _stall_cam():
    """A driver instance for the grab-policy tests: never connected, so
    give it the nominal intrinsics the frame tuple carries."""
    c = HikRobot()
    c._nominal = None
    return c


def _grabber(script):
    """_grab_bgr stand-in: pops one entry per call — "stall" raises the
    transient error, anything else is returned as the frame."""
    calls = []

    def grab(self):
        step = script.pop(0)
        calls.append(step)
        if step == "stall":
            raise hik._TransientGrabError("no frame")
        return step
    return grab, calls


def test_one_empty_grab_is_transient_not_a_reconnect(monkeypatch):
    c = _stall_cam()
    grab, calls = _grabber(["stall"])
    monkeypatch.setattr(HikRobot, "_grab_bgr", grab)
    monkeypatch.setattr(c, "recover", lambda: pytest.fail("must not reconnect"))
    with pytest.raises(hik._TransientGrabError):
        c.get_all()
    assert c._stalls == 1


def test_two_empty_grabs_in_a_row_rebuild_the_session(monkeypatch):
    c = _stall_cam()
    grab, calls = _grabber(["stall", "stall", "frame"])
    monkeypatch.setattr(HikRobot, "_grab_bgr", grab)
    monkeypatch.setattr(c, "_present", lambda: True)
    recovered = []
    monkeypatch.setattr(c, "recover", lambda: recovered.append(1) or True)
    with pytest.raises(hik._TransientGrabError):
        c.get_all()                       # 1st: transient
    out = c.get_all()                     # 2nd: stall -> recover -> grab
    assert recovered == [1]
    assert out[5] == "frame" and c._stalls == 0


def test_dead_stream_is_rebuilt_on_the_first_empty_grab(monkeypatch):
    # SDK receive counter flat across the grab window -> nothing is
    # arriving -> rebuild now, don't make the user wait for a 2nd capture
    c = _stall_cam()
    grab, calls = _grabber(["stall", "frame"])
    monkeypatch.setattr(HikRobot, "_grab_bgr", grab)
    monkeypatch.setattr(c, "net_stats", lambda: {"recv_frames": 80})
    monkeypatch.setattr(c, "_present", lambda: True)
    recovered = []
    monkeypatch.setattr(c, "recover", lambda: recovered.append(1) or True)
    out = c.get_all()
    assert recovered == [1] and out[5] == "frame" and calls == ["stall", "frame"]


def test_lossy_stream_with_frames_arriving_stays_transient(monkeypatch):
    # counter moving -> frames arrive but none complete -> lossy link,
    # keep the session (Wi-Fi bench contract)
    c = _stall_cam()
    grab, calls = _grabber(["stall"])
    monkeypatch.setattr(HikRobot, "_grab_bgr", grab)
    counter = iter([80, 95])
    monkeypatch.setattr(c, "net_stats", lambda: {"recv_frames": next(counter)})
    monkeypatch.setattr(c, "recover", lambda: pytest.fail("must not reconnect"))
    with pytest.raises(hik._TransientGrabError):
        c.get_all()


def test_a_frame_resets_the_stall_count(monkeypatch):
    c = _stall_cam()
    grab, calls = _grabber(["stall", "frame", "stall"])
    monkeypatch.setattr(HikRobot, "_grab_bgr", grab)
    monkeypatch.setattr(c, "recover", lambda: pytest.fail("must not reconnect"))
    with pytest.raises(hik._TransientGrabError):
        c.get_all()
    c.get_all()
    with pytest.raises(hik._TransientGrabError):
        c.get_all()                       # stalls: 1, 0, 1 — never reaches 2
    assert c._stalls == 1


@pytest.fixture
def armed(monkeypatch):
    """A connected-looking driver over a stub SDK: the fake camera
    records trigger commands and answers every read with a 2x2 frame."""
    import ctypes
    from types import SimpleNamespace

    class Info(ctypes.Structure):
        _fields_ = [("nWidth", ctypes.c_uint), ("nHeight", ctypes.c_uint)]

    log = SimpleNamespace(cmds=[], cleared=0)

    class FakeCam:
        def MV_CC_ClearImageBuffer(self):
            log.cleared += 1
            return 0

        def MV_CC_SetCommandValue(self, key):
            log.cmds.append(key)
            return 0

        def MV_CC_GetImageForBGR(self, buf, n, info, timeout_ms):
            info.nWidth, info.nHeight = 2, 2
            return 0

    monkeypatch.setattr(hik, "_mv", SimpleNamespace(
        MV_FRAME_OUT_INFO_EX=Info, MV_E_NODATA=0x80000007))
    c = HikRobot()
    c._cam, c.width, c.height, c.serial_number = FakeCam(), 2, 2, "SN1"
    return c, log


def test_on_demand_grab_clears_triggers_then_reads(armed):
    c, log = armed                      # default acquisition: trigger
    img = c._grab_bgr()
    assert log.cmds == ["TriggerSoftware"] and log.cleared == 1
    assert img.shape == (2, 2, 3)


def test_continuous_grab_never_triggers(armed):
    c, log = armed
    c._acquisition = "continuous"
    assert c._grab_bgr().shape == (2, 2, 3)
    assert log.cmds == [] and log.cleared == 0


def test_refused_trigger_is_fatal_not_transient(armed):
    c, log = armed
    c._cam.MV_CC_SetCommandValue = lambda key: 0x80000203   # session gone
    with pytest.raises(RuntimeError) as ex:
        c._grab_bgr()
    assert not isinstance(ex.value, hik._TransientGrabError)
    assert c.state == "down"            # handle released for the rebuild


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
