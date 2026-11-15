"""Luxonis OAK (RVC2, USB) driver — the platform's fourth camera type.

Built for the OAK-1 W (IMX378, 12 MP, fixed focus, ~108° wide lens) and
written against any RVC2 OAK's color camera on socket CAM_A (OAK-1,
OAK-1 W, OAK-1 Max, the color camera of an OAK-D). Duck-types the same
surface :class:`camera.Camera` (RealSense D405), :class:`camera.UEyeXS`
and :class:`camera.HikRobot` expose, so ``CameraPool`` /
``MQTTDeviceAdapter`` / ``Detection`` treat it alike:

  Device protocol   id / state / msg / on_state_change /
                    on_hardware_available / recover / release
  Capture           connect / get_all / frame-shaped 9-tuple / close
  Intrinsics        camera_matrix / dist_coeffs / get_K / get_D

What is different, stated instead of papered over:

- **Color only.** No depth, no IR — ``get_all`` returns ``None`` in the
  depth/ir slots (the 9-tuple shape is kept). Depth-dependent calls
  (``xyz``) fail loudly downstream.
- **USB, booted by the host.** The camera has no firmware of its own:
  ``connect`` boots it over USB (it shows as ``03e7:2485`` Movidius
  MyriadX until then). Addressed by ``serial_number`` = its device id
  (the "MXID", e.g. ``194430102136083300``). One process at a time.
- **Factory intrinsics, per frame.** Luxonis calibrates each unit; the
  camera matrix and the 14-coefficient (OpenCV rational) distortion of
  EXACTLY the frame delivered — after the device's own crop/scale — come
  with every frame (``ImgFrame.getTransformation``). The full-resolution
  output is a 4000 x 3000 centre crop of the 4056 x 3040 sensor, and the
  matrix says so. Authored ``K``/``D``/``native_res`` still override.
- **Fresh frames.** The camera streams; ``get_all`` returns the first
  frame whose exposure STARTED after the call (device timestamps are
  synced to the host clock), never a buffered one.
- **Controls apply a few frames late.** Exposure and white balance are
  sent to the ISP and take effect ~1-3 s later (measured at 5 fps); the
  setters wait until the frames confirm the change, and say if they
  never do.
- **Exposure** in µs (the D405's and the Hikrobot's unit). Manual
  exposure on this sensor needs a sensitivity too: ``iso`` (100-1600),
  the current one when not given. ``exposure=None`` = continuous auto.
- **White balance.** ``{"auto": True}`` continuous, ``{"once": True}``
  converge then hold, ``{"hold": True}`` freeze where it is,
  ``{"kelvin": 4500}`` fixed (1000-12000).
- **Anti-banding.** ``anti_banding=60`` (or 50) matches the exposure to
  the mains so LED / fluorescent light does not band the image; ``None``
  leaves the device default.
- **Hotplug.** A presence monitor polls the USB devices every 3 s: an
  unplugged camera goes ``down`` at once (the device handle closes), and
  while down the camera reappearing fires ``on_hardware_available``
  (cooldown 10 s) so the pool's AutoRecover reconnects it — the same
  timings as the D405 / uEye / Hikrobot paths.

``depthai`` (pip, Luxonis) is only required to USE this class — the
module imports without it, enumeration returns [] and says why once,
and ``connect()`` fails with the install line.
"""

import threading
import time

import numpy as np

from .camera import Helper

try:
    import depthai as dai
    _DAI_ERR = None
except Exception as _ex:     # absent or broken install — say so at use
    dai = None
    _DAI_ERR = str(_ex)

_WARNED = {"sdk": False}

# The full-resolution stream: every pixel the sensor gives, at a rate a
# USB 3 link and the host carry without strain (12 MP NV12 ~ 18 MB a
# frame). The output is NV12 (half the bytes of BGR over the link); the
# host converts the frames it actually uses.
DEFAULT_FPS = 10
# How long a setter waits for the frames to show its change, and how
# long a grab waits for a fresh frame.
CONTROL_TIMEOUT_S = 6.0
GRAB_TIMEOUT_S = 3.0


def _warn_sdk_missing():
    if not _WARNED["sdk"]:
        _WARNED["sdk"] = True
        print("[oak] depthai not importable (%s) — OAK cameras are not "
              "enumerated. Install it: sudo pip3 install depthai" % _DAI_ERR)


class _OakIntrinsics(object):
    """rs.intrinsics-shaped shim (fx/fy/ppx/ppy/coeffs + width/height) so
    Helper.pixel()/xyz math and camera_matrix() work unchanged."""

    def __init__(self, width, height, fx, fy, ppx, ppy, coeffs=None):
        self.width = int(width)
        self.height = int(height)
        self.fx = float(fx)
        self.fy = float(fy)
        self.ppx = float(ppx)
        self.ppy = float(ppy)
        self.coeffs = [float(v) for v in (coeffs if coeffs is not None else [0.0] * 5)]


class Oak(Helper):
    # Helper gives the pure projection math (pixel / xyz_estimate) the
    # D405 class has; Helper.xyz needs depth and honestly fails here.

    camera_type = "oak"

    def __init__(self):
        super().__init__()
        self.serial_number = None
        self.product = None            # "OAK-1-W", read off the EEPROM at connect
        self.state = "down"
        self.msg = "not connected"
        self._listeners = []
        self._available_listeners = []
        self._listeners_lock = threading.Lock()

        self._dev = None               # dai.Device, None when closed
        self._pipeline = None
        self._q = None                 # frame queue (latest frame only)
        self._ctl = None               # control queue into the ISP
        self._lock = threading.RLock()
        self._connect_kwargs = {}
        self._recover_lock = threading.Lock()
        self._last = None              # metadata of the last frame seen

        self.width = 0
        self.height = 0
        self.intr = None               # authored K/D shim, else None
        self.intr_source = "factory"
        self._factory = None           # the delivered frame's own K/D
        self.mode = "bgr"
        self.stream = None             # requested {"width","height","fps"}
        self.stream_actual = None
        self.filter = {}
        # Color-only device — camera_list reports this so channel pickers
        # offer exactly what the camera has.
        self._enabled_channels = {"color"}

        self.exposure_auto = True
        self.wb_cfg = {"auto": True}

        self._watch_stop = threading.Event()
        self._watch_thread = None

    # ── Device protocol (mirrors camera.Camera) ──────────────────────

    @property
    def id(self):
        return self.serial_number

    def on_state_change(self, callback):
        with self._listeners_lock:
            self._listeners.append(callback)

    def on_hardware_available(self, callback):
        with self._listeners_lock:
            self._available_listeners.append(callback)

    def _set_state(self, new_state, msg=""):
        new_msg = str(msg or "")
        if self.state == new_state and self.msg == new_msg:
            return
        self.state = new_state
        self.msg = new_msg
        with self._listeners_lock:
            cbs = list(self._listeners)
        for cb in cbs:
            try:
                cb(new_state, self.msg)
            except Exception:
                pass

    def release(self):
        return self.close()

    # ── Enumeration ──────────────────────────────────────────────────

    @staticmethod
    def _connected_ids():
        """Device ids of every OAK on USB — booted (ours) or not."""
        if dai is None:
            return set()
        try:
            return {d.getDeviceId() for d in dai.Device.getAllConnectedDevices()}
        except Exception:
            return set()

    @staticmethod
    def all_device():
        """Attached OAK devices: [{serial_number, name, state,
        camera_type}]. ``name`` is the product once a connect has read
        it ("OAK-1-W"), "OAK" before — enumeration does not boot the
        camera. Empty when depthai is absent, and it SAYS SO once (that
        and "no camera on USB" need different fixes)."""
        if dai is None:
            _warn_sdk_missing()
            return []
        out = []
        try:
            for d in dai.Device.getAllConnectedDevices():
                out.append({"serial_number": d.getDeviceId(),
                            "name": _PRODUCTS.get(d.getDeviceId(), "OAK"),
                            "state": str(d.state).rsplit("_", 1)[-1].lower(),
                            "camera_type": "oak"})
        except Exception:
            pass
        return out

    # ── Connect / close ──────────────────────────────────────────────

    def connect(
        self,
        serial_number="",
        stream=None,          # {"width","height","fps"}; None = full resolution at DEFAULT_FPS
        K=None,
        D=None,
        native_res=None,
        exposure=None,        # µs (D405 / Hikrobot unit); None = continuous auto
        iso=None,             # 100-1600 with a manual exposure; None = the current one
        wb=None,              # {"auto"} | {"once"} | {"hold"} | {"kelvin": N}
        anti_banding=None,    # 50 | 60 (Hz mains) | None = device default
        mode="bgr",
        filter={},
        max_tries=3,
        raise_on_fail=False,
        **_ignored,           # tolerate other drivers' keys (channels, ...)
    ):
        if dai is None:
            msg = ("depthai not available (%s) — install it: "
                   "sudo pip3 install depthai, then restart this process" % _DAI_ERR)
            self._set_state("down", msg)
            if raise_on_fail:
                raise RuntimeError(msg)
            return False

        self._connect_kwargs = dict(
            serial_number=serial_number, stream=stream, K=K, D=D,
            native_res=native_res, exposure=exposure, iso=iso, wb=wb,
            anti_banding=anti_banding, mode=mode, filter=filter,
            max_tries=max_tries,
        )
        self.filter = filter
        self.mode = mode
        self.stream = dict(stream) if stream else None

        last_ex = None
        for attempt in range(max_tries):
            try:
                with self._lock:
                    self._open(serial_number, stream, K, D, native_res,
                               exposure, iso, wb, anti_banding)
                self._set_state("ok", "")
                self._watch_start()
                return True
            except Exception as ex:
                last_ex = ex
                with self._lock:
                    self._close_quiet()
                # a just-closed OAK resets and re-enumerates in its
                # bootloader — give it time before the next boot
                time.sleep(1.5 + 1.5 * attempt)
        what = f"serial={serial_number}" if serial_number else "any"
        msg = f"OAK connect failed ({what}): {last_ex}"
        self._set_state("down", msg)
        if raise_on_fail:
            raise RuntimeError(msg)
        return False

    def _open(self, serial_number, stream, K, D, native_res, exposure, iso, wb, anti_banding):
        if serial_number:
            if str(serial_number) not in self._connected_ids():
                raise RuntimeError(f"no OAK with id {serial_number} on USB — "
                                   f"check the cable (a USB 3 port, the blue one)")
            dev = dai.Device(dai.DeviceInfo(str(serial_number)))
        else:
            dev = dai.Device()
        self._dev = dev
        self.serial_number = dev.getDeviceId()
        try:
            self.product = dev.readCalibration().getEepromData().productName or "OAK"
        except Exception:
            self.product = "OAK"
        _PRODUCTS[self.serial_number] = self.product

        p = dai.Pipeline(dev)
        cam = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        fps = float((stream or {}).get("fps") or DEFAULT_FPS)
        if stream and stream.get("width") and stream.get("height"):
            out = cam.requestOutput((int(stream["width"]), int(stream["height"])),
                                    dai.ImgFrame.Type.NV12, fps=fps)
        else:
            out = cam.requestFullResolutionOutput(dai.ImgFrame.Type.NV12, fps=fps)
        self._q = out.createOutputQueue(maxSize=1, blocking=False)
        self._ctl = cam.inputControl.createInputQueue()
        self._pipeline = p
        p.start()

        if anti_banding:
            c = dai.CameraControl()
            c.setAntiBandingMode({50: dai.CameraControl.AntiBandingMode.MAINS_50_HZ,
                                  60: dai.CameraControl.AntiBandingMode.MAINS_60_HZ}[int(anti_banding)])
            self._ctl.send(c)

        # end-to-end verification — one real frame; its size and its own
        # intrinsics are the stream's
        f = self._fresh(time.monotonic() + max(GRAB_TIMEOUT_S, 5.0))
        self.width, self.height = int(f.getWidth()), int(f.getHeight())
        self.stream_actual = {"width": self.width, "height": self.height, "fps": fps}
        self._factory = self._frame_intr(f)

        if K is not None and D is not None:
            K_, D_ = np.array(K, dtype=float), [float(v) for v in D]
            sx = sy = 1.0
            if native_res is not None:
                sx = self.width / float(native_res[0])
                sy = self.height / float(native_res[1])
            self.intr = _OakIntrinsics(self.width, self.height,
                                       K_[0, 0] * sx, K_[1, 1] * sy,
                                       K_[0, 2] * sx, K_[1, 2] * sy, D_)
            self.intr_source = "override"
        else:
            self.intr = None
            self.intr_source = "factory"

        if exposure is not None:
            self._exposure(float(exposure), iso)
        else:
            self.exposure_auto = True
        self.white_balance(wb or {"auto": True})

    def _frame_intr(self, f):
        """The factory intrinsics of the frame ``f`` as delivered."""
        tr = f.getTransformation()
        Km = np.array(tr.getIntrinsicMatrix(), dtype=float)
        return _OakIntrinsics(f.getWidth(), f.getHeight(), Km[0, 0], Km[1, 1],
                              Km[0, 2], Km[1, 2], list(tr.getDistortionCoefficients()))

    def _close_quiet(self):
        p, self._pipeline = self._pipeline, None
        dev, self._dev = self._dev, None
        self._q = self._ctl = None
        for fn in ((lambda: p.stop()) if p is not None else None,
                   (lambda: dev.close()) if dev is not None else None):
            if fn is not None:
                try:
                    fn()
                except Exception:
                    pass

    def close(self):
        self._watch_stop.set()
        with self._lock:
            self._close_quiet()
        self._set_state("down", "closed")
        return True

    # ── Presence monitor → AutoRecover trigger (D405 hotplug parity) ─

    def _watch_start(self):
        if self._watch_thread is not None and self._watch_thread.is_alive():
            return
        self._watch_stop.clear()

        def _loop():
            # while OK:   the device handle closed (XLink lost the link) or
            #             the camera gone from USB twice running = unplugged.
            #             Go down and release the handle.
            # while DOWN: the camera back on USB -> fire the AutoRecover
            #             trigger (level-triggered, cooldown).
            cooldown, last_fire, misses = 10.0, 0.0, 0
            while not self._watch_stop.wait(3.0):
                if not self.serial_number:
                    continue
                present = self.serial_number in self._connected_ids()
                if self.state == "ok":
                    dead = self._dev is None or self._dev.isClosed()
                    misses = 0 if present else misses + 1
                    if dead or misses >= 2:
                        misses = 0
                        with self._lock:
                            self._close_quiet()
                        self._set_state("down", "device lost on USB — unplugged?")
                elif self.state == "down":
                    misses = 0
                    if present and (time.monotonic() - last_fire) >= cooldown:
                        last_fire = time.monotonic()
                        with self._listeners_lock:
                            cbs = list(self._available_listeners)
                        for cb in cbs:
                            try:
                                cb()
                            except Exception:
                                pass

        self._watch_thread = threading.Thread(
            target=_loop, daemon=True, name=f"oak-watch-{self.serial_number}")
        self._watch_thread.start()

    # ── Recovery (Device protocol) ───────────────────────────────────

    def recover(self):
        with self._recover_lock:
            self._set_state("recovering", "rebooting the OAK over USB")
            with self._lock:
                self._close_quiet()
            # Fast judgment: not on USB -> fail now. A closed OAK resets
            # and comes back in its bootloader within a few seconds.
            deadline = time.time() + 10.0
            while self.serial_number and time.time() < deadline:
                if self.serial_number in self._connected_ids():
                    break
                time.sleep(0.5)
            else:
                if self.serial_number:
                    self._set_state("down", "OAK not on USB — check the cable and retry")
                    return False
            return bool(self.connect(**dict(self._connect_kwargs)))

    # ── Capture ──────────────────────────────────────────────────────

    def _fresh(self, deadline):
        """The first frame whose exposure STARTED after this call —
        never a buffered one. Raises when none arrives by ``deadline``
        (time.monotonic)."""
        if self._q is None:
            raise RuntimeError("OAK camera not connected"
                               f"{' — ' + self.msg if self.msg else ''} (call connect() again)")
        t_req = dai.Clock.now()
        while time.monotonic() < deadline:
            f = self._q.tryGet()
            if f is None:
                if self._dev is None or self._dev.isClosed():
                    raise RuntimeError("OAK device closed — lost on USB?")
                time.sleep(0.005)
                continue
            self._last = f
            if f.getTimestamp(dai.CameraExposureOffset.START) >= t_req:
                return f
        raise RuntimeError(f"no fresh frame within {GRAB_TIMEOUT_S:g} s")

    def _grab(self):
        with self._lock:
            try:
                f = self._fresh(time.monotonic() + GRAB_TIMEOUT_S)
                return f.getCvFrame(), f
            except Exception as ex:
                self._close_quiet()
                self._set_state("down", f"frame grab failed: {ex}")
                raise

    def get_all(self, align_to=None, alpha=None):
        """Same 9-tuple as Camera.get_all — depth/ir slots are None
        (color-only device); depth_int is the frame's own intrinsics, or
        the authored ones."""
        try:
            color_img, f = self._grab()
        except Exception:
            # one rebuild if the camera is still on USB (a dropped link,
            # a reset) — then the grab again, or the error stands
            if not (self.serial_number in self._connected_ids() and self.recover()):
                raise
            color_img, f = self._grab()
        if self.state != "ok":
            self._set_state("ok", "")
        self._factory = self._frame_intr(f)
        depth_int = self.intr if self.intr is not None else self._factory
        return (None, None, None, None, None, color_img, depth_int,
                None, time.time())

    # ── Intrinsics (mirrors Camera) ──────────────────────────────────

    def camera_matrix(self, depth_int, ratio=1):
        return np.array([[ratio * depth_int.fx, 0., ratio * depth_int.ppx],
                         [0., ratio * depth_int.fy, ratio * depth_int.ppy],
                         [0., 0., 1.]])

    def dist_coeffs(self, depth_int):
        return np.array(depth_int.coeffs)

    def get_K(self, stream="color"):
        i = self.intr if self.intr is not None else self._factory
        return [[i.fx, 0.0, i.ppx], [0.0, i.fy, i.ppy], [0.0, 0.0, 1.0]]

    def get_D(self, stream="color"):
        i = self.intr if self.intr is not None else self._factory
        return [float(v) for v in i.coeffs]

    # ── Exposure / white balance (sent to the ISP, confirmed by frames) ─

    def _send_and_confirm(self, build, ok, what):
        """Send the control ``build`` fills in, then read frames until
        ``ok(frame)`` holds — the ISP applies a control a few frames
        late. Raises when the frames never show it."""
        with self._lock:
            if self._ctl is None:
                raise RuntimeError("OAK camera not connected")
            c = dai.CameraControl()
            build(c)
            self._ctl.send(c)
            deadline = time.monotonic() + CONTROL_TIMEOUT_S
            while time.monotonic() < deadline:
                f = self._fresh(deadline)
                if ok(f):
                    return f
        raise RuntimeError(f"OAK did not apply {what} within {CONTROL_TIMEOUT_S:g} s")

    def _exposure(self, us, iso=None):
        us = int(round(us))
        if iso is None:
            iso = self._last.getSensitivity() if self._last is not None else 400
        iso = int(min(1600, max(100, iso)))
        self._send_and_confirm(
            lambda c: c.setManualExposure(us, iso),
            lambda f: abs(f.getExposureTime().total_seconds() * 1e6 - us) <= max(50, 0.02 * us),
            f"a {us} µs exposure")
        self.exposure_auto = False

    def get_exposure(self):
        """Current exposure time in µs — the latest frame's."""
        with self._lock:
            f = self._fresh(time.monotonic() + GRAB_TIMEOUT_S)
        return round(f.getExposureTime().total_seconds() * 1e6, 1)

    def set_exposure(self, exposure, iso=None, _locked=True):
        """Pin manual exposure, in µs (same unit as the D405), at ``iso``
        (the current sensitivity when None)."""
        self._exposure(float(exposure), iso)
        return True

    def auto_exposure(self, enable=True, _locked=True):
        if not enable:
            raise ValueError("auto_exposure(False): pin an exposure with set_exposure(µs)")
        with self._lock:
            if self._ctl is None:
                raise RuntimeError("OAK camera not connected")
            c = dai.CameraControl()
            c.setAutoExposureEnable()
            self._ctl.send(c)
        self.exposure_auto = True
        return True

    def white_balance(self, cfg, _locked=True):
        """{"auto": True} continuous, {"once": True} converge then hold,
        {"hold": True} freeze where it is, {"kelvin": N} fixed."""
        cfg = dict(cfg or {})
        M = dai.CameraControl.AutoWhiteBalanceMode
        if cfg.get("kelvin"):
            k = int(cfg["kelvin"])
            if not 1000 <= k <= 12000:
                raise ValueError("wb kelvin must be 1000-12000")
            self._send_and_confirm(lambda c: c.setManualWhiteBalance(k),
                                   lambda f: abs(f.getColorTemperature() - k) <= 100,
                                   f"a {k} K white balance")
            store = {"kelvin": k}
        elif cfg.get("auto") or cfg.get("once"):
            with self._lock:
                if self._ctl is None:
                    raise RuntimeError("OAK camera not connected")
                c = dai.CameraControl()
                c.setAutoWhiteBalanceMode(M.AUTO)
                c.setAutoWhiteBalanceLock(False)
                self._ctl.send(c)
            store = {"auto": True}
            if cfg.get("once"):
                # let auto converge on the lit scene, then hold it there
                time.sleep(2.0)
                self.white_balance({"hold": True})
                store = {"once": True}
        elif cfg.get("hold"):
            with self._lock:
                if self._ctl is None:
                    raise RuntimeError("OAK camera not connected")
                c = dai.CameraControl()
                c.setAutoWhiteBalanceLock(True)
                self._ctl.send(c)
            store = {"hold": True}
        else:
            raise ValueError('wb needs {"auto": True}, {"once": True}, {"hold": True} '
                             'or {"kelvin": N}')
        self.wb_cfg = store
        return dict(store)

    def white_balance_info(self):
        out = {"cfg": dict(self.wb_cfg)}
        if self._last is not None:
            try:
                out["kelvin"] = int(self._last.getColorTemperature())
            except Exception:
                pass
        return out


# product name per device id, once a connect has read it — so the device
# list can say "OAK-1-W" without booting the camera to ask
_PRODUCTS = {}
