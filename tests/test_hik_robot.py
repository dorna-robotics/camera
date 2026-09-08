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


def test_no_ip_in_config_uses_a_free_stepping_stone(stranded):
    # The GUI's Add sends only the serial. Pick a free address, move the
    # camera there, reboot it, open it on the lease DHCP hands out.
    stranded.after_reset = "10.0.1.51"          # the router's reservation
    st, d = _rescue("")
    assert stranded.forced == [("10.0.1.250", "255.255.255.0")]
    assert stranded.resets == ["DeviceReset"]
    assert d["ip"] == "10.0.1.51"


def test_free_address_skips_nic_gateway_cameras_and_ping_answers(stranded, monkeypatch):
    monkeypatch.setattr(hik, "_ping", lambda ip, timeout_s=1: ip in ("10.0.1.250", "10.0.1.249"))
    monkeypatch.setattr(HikRobot, "_enum_raw", staticmethod(
        lambda: [(_gige_struct("10.0.1.248"), _dev("10.0.1.248"))]))
    c = HikRobot()
    assert c._free_address(hik._ip_to_int("10.0.1.40"), 0xFFFFFF00) == "10.0.1.247"


@pytest.mark.parametrize("want, needle", [
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


@pytest.fixture
def warming(monkeypatch):
    """A grabbing-looking driver for the connect warm-up: the fake cam
    scripts GetImageBuffer results (0 = a frame, else = a miss) via
    ``grabs`` and ExposureTime readbacks via ``exposures`` (last value
    repeats). Triggers land in ``triggers``."""
    import ctypes
    from types import SimpleNamespace

    class Frame(ctypes.Structure):
        _fields_ = [("pad", ctypes.c_uint)]

    class FloatVal(ctypes.Structure):
        _fields_ = [("fCurValue", ctypes.c_float)]

    log = SimpleNamespace(triggers=0, frees=0, grabs=[], exposures=[])

    class FakeCam:
        def MV_CC_ClearImageBuffer(self):
            return 0

        def MV_CC_SetCommandValue(self, key):
            assert key == "TriggerSoftware"
            log.triggers += 1
            return 0

        def MV_CC_GetImageBuffer(self, fr, timeout_ms):
            return log.grabs.pop(0) if log.grabs else 0

        def MV_CC_FreeImageBuffer(self, fr):
            log.frees += 1
            return 0

        def MV_CC_GetFloatValue(self, key, v):
            if key != "ExposureTime":
                return -1               # entry model without a Gain node
            if len(log.exposures) > 1:
                v.fCurValue = log.exposures.pop(0)
            else:
                v.fCurValue = log.exposures[0] if log.exposures else 10000.0
            return 0

    monkeypatch.setattr(hik, "_mv", SimpleNamespace(
        MV_FRAME_OUT=Frame, MVCC_FLOATVALUE=FloatVal))
    c = HikRobot()
    c._cam = FakeCam()
    return c, log


def _unstable_mean():
    """A _frame_mean stand-in whose brightness never settles."""
    flip = [0]

    def mean(self, fr):
        flip[0] ^= 1
        return 200.0 if flip[0] else 10.0
    return mean


def test_warmup_converges_early_on_a_stable_scene(warming, monkeypatch):
    # Warm camera, auto already settled: 1 baseline + 3 steady frames.
    c, log = warming
    monkeypatch.setattr(HikRobot, "_frame_mean", lambda self, fr: 120.0)
    c._warmup()
    assert log.triggers == 4


def test_warmup_rides_the_ramp_and_stops_at_the_plateau(warming, monkeypatch):
    # Cold dark boot: brightness climbs frame by frame; the burst keeps
    # feeding the algorithm through the climb and stops once the last 4
    # frames are flat — no fixed count involved.
    c, log = warming
    means = [10, 30, 60, 90, 110, 120, 121, 121.5, 122]
    monkeypatch.setattr(HikRobot, "_frame_mean",
                        lambda self, fr: means.pop(0) if len(means) > 1 else means[0])
    c._warmup()
    assert log.triggers == 9            # first flat window: 120..122


def test_warmup_does_not_mistake_a_slow_ramp_for_a_plateau(warming, monkeypatch):
    # THE bug measured on the bench: a cold dark camera climbs ~2 gray
    # levels per frame. Consecutive deltas within tolerance used to read
    # as "steady" after 3 frames and the burst stopped with the image
    # still dark. The window test sees the spread of the ramp and rides
    # it to the top.
    c, log = warming
    ramp = [20 + 2 * i for i in range(20)] + [60] * 10     # +2/frame, then flat
    monkeypatch.setattr(HikRobot, "_frame_mean",
                        lambda self, fr: ramp.pop(0) if len(ramp) > 1 else ramp[0])
    c._warmup()
    assert log.triggers >= 20                   # never stopped on the ramp
    assert log.triggers <= 24                   # settled within 4 of the plateau


def test_warmup_sees_through_a_black_clipped_image(warming, monkeypatch):
    # THE bug on the darker camera: its first frames are pitch black
    # (mean ~1), i.e. flat — while exposure is still climbing behind the
    # clipped image. Exposure must veto "settled" until it flattens too.
    c, log = warming
    monkeypatch.setattr(HikRobot, "_frame_mean", lambda self, fr: 1.0)
    log.exposures = [100 * 1.3 ** i for i in range(16)] + [5000.0] * 10  # climb, then flat
    c._warmup()
    assert log.triggers >= 17                   # rode the whole exposure climb
    assert log.triggers <= 21                   # settled within 4 of exposure flattening


def test_warmup_black_image_with_no_exposure_signal_runs_to_the_cap(warming, monkeypatch):
    # Pixels flat at the floor and nothing else readable: never declare a
    # black image settled on pixels alone — run to the frame cap.
    c, log = warming
    monkeypatch.setattr(HikRobot, "_frame_mean", lambda self, fr: 1.0)
    monkeypatch.setattr(c, "_get_float", lambda key: (_ for _ in ()).throw(RuntimeError(key)))
    c._warmup()
    assert log.triggers == 40


def test_warmup_never_exceeds_the_frame_cap(warming, monkeypatch):
    c, log = warming
    monkeypatch.setattr(HikRobot, "_frame_mean", _unstable_mean())
    c._warmup()
    assert log.triggers == 40           # exactly the cap, never more


def _fake_clock(monkeypatch, step):
    clock = [0.0]

    def monotonic():
        clock[0] += step
        return clock[0] - step
    monkeypatch.setattr(hik.time, "monotonic", monotonic)


def test_warmup_deadline_is_a_backstop_not_the_limit(warming, monkeypatch):
    # A real burst costs ~0.25 s a frame on the bench (6 MB frames + the
    # exposure ramp) — 40 frames is ~10 s. The old 8 s deadline cut that
    # ramp short and handed back a dark first frame. With frames flowing
    # the FRAME CAP must be what stops a scene that never settles.
    c, log = warming
    monkeypatch.setattr(HikRobot, "_frame_mean", _unstable_mean())
    _fake_clock(monkeypatch, 0.25)
    c._warmup()
    assert log.triggers == 40           # the cap, not the clock


def test_warmup_stops_on_the_wall_clock_deadline(warming, monkeypatch):
    # Pathological slow-but-alive stream (10 "seconds" per frame): the
    # wall clock is what ends it, long before the 40-frame cap.
    c, log = warming
    monkeypatch.setattr(HikRobot, "_frame_mean", _unstable_mean())
    _fake_clock(monkeypatch, 10.0)
    c._warmup()
    assert 0 < log.triggers <= 4        # deadline, not the frame cap


def test_warmup_survives_the_first_grab_timeout(warming):
    # The first grab after StartGrabbing routinely times out while the
    # GigE stream channel comes up. That used to abort the whole burst
    # (0 warm-up frames -> black first capture); it must keep going.
    c, log = warming
    log.grabs = [0x80000007]            # miss once, then frames
    c._warmup()
    assert log.triggers == 5            # miss + baseline + 3 steady


def test_warmup_grab_error_after_frames_stops_at_once(warming, monkeypatch):
    # Once frames have flowed, a miss means the stream died mid-burst:
    # stop immediately, don't spin 2 s timeouts on it. Every grabbed
    # buffer was freed, the miss freed nothing.
    c, log = warming
    monkeypatch.setattr(HikRobot, "_frame_mean", lambda self, fr: 120.0)
    log.grabs = [0, 0, 1]
    c._warmup()
    assert log.triggers == 3 and log.frees == 2


def test_warmup_gives_up_after_three_consecutive_misses(warming):
    c, log = warming
    log.grabs = [1, 1, 1]               # nothing is coming — stop asking
    c._warmup()
    assert log.triggers == 3


def test_warmup_stops_when_exposure_holds_still(warming):
    # No readable brightness (the stub frame has no buffer): falls back
    # to the ExposureTime readback and stops when IT holds still.
    c, log = warming
    log.exposures = [1000, 2000, 4000, 8000, 9800, 9900, 9950, 9990]
    c._warmup()
    assert log.triggers == 8            # 3 steady readbacks, not the cap


def test_warmup_swallows_sdk_exceptions(warming):
    c, log = warming

    def boom(fr, timeout_ms):
        raise RuntimeError("sdk fault")
    c._cam.MV_CC_GetImageBuffer = boom
    c._warmup()                         # returns quietly — never raises
    assert log.triggers == 1


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
