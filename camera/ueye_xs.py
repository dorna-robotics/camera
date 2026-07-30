"""IDS uEye XS driver — the platform's second camera type.

Duck-types the same surface :class:`camera.Camera` (RealSense D405)
exposes, so ``CameraPool`` / ``MQTTDeviceAdapter`` / ``Detection`` treat
both alike:

  Device protocol   id / state / msg / on_state_change /
                    on_hardware_available / recover / release
  Capture           connect / get_all / frame-shaped 9-tuple / close
  Intrinsics        camera_matrix / dist_coeffs / get_K / get_D

Differences from the D405, stated instead of papered over:

- **Color only.** No depth, no IR — ``get_all`` returns ``None`` in the
  depth/ir slots (the 9-tuple shape is preserved, same as a D405 with
  those channels unsubscribed). Depth-dependent calls (``xyz``) fail
  loudly downstream.
- **Intrinsics are authored or nominal.** There is no factory
  calibration to read. Author ``K``/``D``/``native_res`` in
  ``camera_cfg`` for metric work; without them a NOMINAL pinhole
  (sensor-spec focal length, centered principal point, zero distortion)
  is reported and labeled ``"nominal"`` in ``camera_info``.
- **Autofocus.** The XS has a liquid lens: continuous AF, one-shot AF,
  and manual positions (device range probed, ~112..240 on Rev 1.1).
  ``focus_region(rect)`` prefers the SDK's native AF AOI (hardware AF
  pointed at the rect, ~1-2 s) and falls back to the region-tune sweep
  (coarse -> fine -> micro, Laplacian variance inside ``rect``,
  ~10-20 s). NOTE: the XS Rev 1.1 does NOT expose the AF-AOI capability
  (FOC_CAP_AUTOFOCUS_AOI absent — verified on hardware), so on this
  device region focus is always the sweep.
- **Exposure / white balance.** The XS ISP owns both. Exposure is
  auto-ONLY (readable via ``get_exposure``; manual sets are refused
  honestly). White balance has two states: ``{"auto": True}`` and
  ``{"hold": True}`` (freeze the current convergence) — fixed
  kelvin/rgb are ISP-rejected. Bench recipe: let auto settle on the
  lit scene, then hold.
- **Hotplug.** The uEye SDK has no librealsense-style hotplug callback,
  so a full-time presence monitor (3 s enumeration poll) substitutes:
  an unplug is detected in ~6-9 s WITHOUT waiting for a grab to fail
  (the handle is released immediately — required for the daemon to
  re-claim the device later), and a replug fires
  ``on_hardware_available`` within ~3 s of the daemon re-listing it, so
  the pool's AutoRecover reconnects with no user action — D405-parity
  timings on both edges. ``recover()`` fast-fails via sysfs when no IDS
  device is on the USB bus at all.

``pyueye`` (plus the IDS runtime it wraps) is only required to USE this
class — importing the module without it works, enumeration returns [],
and connect() raises with an actionable message.
"""

import threading
import time

import numpy as np
import cv2

try:
    from pyueye import ueye
    _PYUEYE_ERR = None
except Exception as _ex:          # ImportError or missing libueye_api.so
    ueye = None
    _PYUEYE_ERR = str(_ex)


class _UEyeIntrinsics(object):
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


class UEyeXS(object):
    # Nominal pinhole for the uEye XS (5MP 1/4" sensor, 1.4 um pixels,
    # EFL 3.82 mm -> ~2729 px at full resolution). Used ONLY when no
    # K/D are authored; camera_info labels it "nominal".
    NOMINAL_FOCAL_PX = 2729.0

    camera_type = "ueye_xs"

    def __init__(self):
        self.serial_number = None
        self.state = "down"
        self.msg = "not connected"
        self._listeners = []
        self._available_listeners = []
        self._listeners_lock = threading.Lock()

        self._h = None                 # ueye.HIDS handle, None when closed
        self._sdk_lock = threading.Lock()
        self._connect_kwargs = {}
        self._recover_lock = threading.Lock()

        self.width = 0
        self.height = 0
        self.intr = None               # authored K/D shim, else None
        self.mode = "bgr"
        self.stream = None             # {"width","height","fps"} actual
        self.stream_actual = None
        self.filter = {}

        # focus / exposure / white-balance state (last applied config)
        self.focus_supported = False
        self.focus_range = (0, 0)
        self.focus_cfg = {"mode": "continuous"}
        self.wb_cfg = {"auto": True}
        self.exposure_auto = True

        # replug watcher (runs only while state == "down")
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
    def _usb_versions():
        """USB spec versions of attached IDS devices (vendor 1409), from
        sysfs. The uEye USB descriptor carries no serial, so per-device
        matching isn't possible — all_device() stamps the version only
        when every attached uEye shares one (the common case; the XS is
        USB 2.0 by design)."""
        import glob
        out = set()
        for p in glob.glob("/sys/bus/usb/devices/*/idVendor"):
            try:
                if open(p).read().strip() == "1409":
                    v = open(p[: -len("idVendor")] + "version").read().strip()
                    try:
                        v = f"{float(v):.1f}"   # "2.00" -> "2.0", "3.20" -> "3.2"
                    except ValueError:
                        pass
                    out.add(v)
            except Exception:
                pass
        return out

    @staticmethod
    def all_device():
        """Attached uEye devices: [{serial_number, name, dev_id,
        camera_type, in_use, usb_type?}]. Empty when the SDK is absent."""
        if ueye is None:
            return []
        try:
            n = ueye.INT()
            if ueye.is_GetNumberOfCameras(n) != 0 or n.value < 1:
                return []
            lst = ueye.UEYE_CAMERA_LIST(ueye.UEYE_CAMERA_INFO * n.value)
            lst.dwCount = n.value
            if ueye.is_GetCameraList(lst) != 0:
                return []
            out = []
            for i in range(n.value):
                info = lst.uci[i]
                out.append({
                    "serial_number": bytes(info.SerNo).partition(b"\0")[0].decode(errors="replace"),
                    "name": bytes(info.Model).partition(b"\0")[0].decode(errors="replace"),
                    "dev_id": int(info.dwDeviceID),
                    "camera_type": "ueye_xs",
                    "in_use": bool(info.dwInUse),
                })
            vers = UEyeXS._usb_versions()
            if len(vers) == 1:
                usb = vers.pop()
                for d in out:
                    d["usb_type"] = usb
            return out
        except Exception:
            return []

    # ── Connect / close ──────────────────────────────────────────────

    def connect(
        self,
        serial_number="",
        stream=None,          # accepted for cfg symmetry; XS runs full sensor
        K=None,
        D=None,
        native_res=None,
        focus=None,           # {"mode": "continuous"} | {"mode":"manual","position":N}
        exposure=None,        # MUST be None — the XS ISP owns exposure (auto only)
        wb=None,              # {"auto": True} | {"hold": True}
        mode="bgr",
        filter={},
        max_tries=3,
        raise_on_fail=False,
        **_ignored,           # tolerate D405-only keys (channels, ...)
    ):
        if ueye is None:
            msg = ("pyueye / IDS uEye runtime not available (%s) — "
                   "install the IDS Software Suite + `pip3 install pyueye`"
                   % _PYUEYE_ERR)
            self._set_state("down", msg)
            if raise_on_fail:
                raise RuntimeError(msg)
            return False

        self._connect_kwargs = dict(
            serial_number=serial_number, stream=stream, K=K, D=D,
            native_res=native_res, focus=focus, exposure=exposure,
            wb=wb, mode=mode, filter=filter, max_tries=max_tries,
        )
        self.filter = filter
        self.mode = mode

        devs = self.all_device()
        if serial_number:
            match = [d for d in devs if d["serial_number"] == str(serial_number)]
            if not match:
                msg = f"uEye device not found on USB (serial={serial_number})"
                self._set_state("down", msg)
                if raise_on_fail:
                    raise RuntimeError(msg)
                return False
            dev = match[0]
            if dev.get("in_use"):
                # uEye handles are exclusive — a second process cannot open
                # the camera. Say so instead of a bare init failure.
                msg = (f"uEye {serial_number} is already in use by another "
                       f"process (another vision server / notebook holding it?)")
                self._set_state("down", msg)
                if raise_on_fail:
                    raise RuntimeError(msg)
                return False
        elif devs:
            dev = devs[0]
        else:
            self._set_state("down", "no uEye devices found")
            if raise_on_fail:
                raise RuntimeError("no uEye devices found")
            return False
        self.serial_number = dev["serial_number"]

        last_ex = None
        for _ in range(max_tries):
            try:
                self._open(dev, K, D, native_res, focus, exposure, wb)
                # end-to-end verification — one real frame
                self._grab_bgr()
                self._set_state("ok", "")
                self._watch_start()
                return True
            except Exception as ex:
                last_ex = ex
                self._close_handle_quiet()
                time.sleep(0.5)
        msg = f"uEye connect failed: {last_ex}"
        self._set_state("down", msg)
        if raise_on_fail:
            raise RuntimeError(msg)
        return False

    def _open(self, dev, K, D, native_res, focus, exposure, wb=None):
        h = ueye.HIDS(int(dev["dev_id"]) | ueye.IS_USE_DEVICE_ID)
        ret = ueye.is_InitCamera(h, None)
        if ret != 0:
            raise RuntimeError(f"is_InitCamera failed (code {ret})")
        self._h = h

        sensor = ueye.SENSORINFO()
        ueye.is_GetSensorInfo(h, sensor)
        self.width = int(sensor.nMaxWidth.value)
        self.height = int(sensor.nMaxHeight.value)

        ueye.is_SetDisplayMode(h, ueye.IS_SET_DM_DIB)
        ueye.is_SetColorMode(h, ueye.IS_CM_BGR8_PACKED)

        # full-sensor AOI
        rect = ueye.IS_RECT()
        rect.s32X, rect.s32Y = ueye.c_int(0), ueye.c_int(0)
        rect.s32Width, rect.s32Height = ueye.c_int(self.width), ueye.c_int(self.height)
        ueye.is_AOI(h, ueye.IS_AOI_IMAGE_SET_AOI, rect, ueye.sizeof(rect))

        # pixel clock max + a high framerate request — shortens
        # FreezeVideo latency (the playground's settings).
        pc_range = (ueye.c_uint * 3)()
        if ueye.is_PixelClock(h, ueye.IS_PIXELCLOCK_CMD_GET_RANGE,
                              pc_range, ueye.sizeof(pc_range)) == 0:
            ueye.is_PixelClock(h, ueye.IS_PIXELCLOCK_CMD_SET,
                               ueye.c_uint(int(pc_range[1])),
                               ueye.sizeof(ueye.c_uint()))
        actual_fps = ueye.c_double(0)
        ueye.is_SetFrameRate(h, ueye.c_double(60), actual_fps)
        self.stream = {"width": self.width, "height": self.height,
                       "fps": round(float(actual_fps.value), 2)}
        self.stream_actual = dict(self.stream)

        # white balance: authored cfg ({"auto"}/{"hold"}), else auto
        self.white_balance(wb or {"auto": True}, _locked=False)
        # exposure: the XS ISP owns it — auto only. Authored exposure is
        # unhonorable intent, so fail LOUDLY instead of silently ignoring.
        if exposure is not None:
            raise ValueError(
                "camera_cfg exposure is not supported on the uEye XS — "
                "its ISP owns exposure (auto only); remove the key or use "
                "the detection's `intensity` for software brightness")

        # focus capability probe
        fmin, fmax = ueye.c_uint(0), ueye.c_uint(0)
        r1 = ueye.is_Focus(h, ueye.FOC_CMD_GET_MANUAL_FOCUS_MIN, fmin, ueye.sizeof(fmin))
        r2 = ueye.is_Focus(h, ueye.FOC_CMD_GET_MANUAL_FOCUS_MAX, fmax, ueye.sizeof(fmax))
        self.focus_supported = (r1 == 0 and r2 == 0 and fmax.value > fmin.value)
        self.focus_range = (int(fmin.value), int(fmax.value))
        # Default: converge once, then PIN — the lens is manual from the
        # first frame (region focus / an authored position refine it).
        self.focus_apply(focus or {"mode": "once"}, _locked=False)

        # intrinsics: authored (scaled from native_res) else nominal
        if K is not None and D is not None:
            K_, D_ = np.array(K, dtype=float), np.array(D, dtype=float)
            sx = sy = 1.0
            if native_res is not None:
                sx = self.width / float(native_res[0])
                sy = self.height / float(native_res[1])
            self.intr = _UEyeIntrinsics(
                self.width, self.height,
                K_[0, 0] * sx, K_[1, 1] * sy, K_[0, 2] * sx, K_[1, 2] * sy,
                list(D_[:5]))
            self.intr_source = "override"
        else:
            self.intr = None
            self.intr_source = "nominal"
        self._nominal = _UEyeIntrinsics(
            self.width, self.height,
            self.NOMINAL_FOCAL_PX, self.NOMINAL_FOCAL_PX,
            self.width / 2.0, self.height / 2.0)

    def _close_handle_quiet(self):
        h, self._h = self._h, None
        if h is not None and ueye is not None:
            try:
                ueye.is_ExitCamera(h)
            except Exception:
                pass

    def close(self):
        self._watch_stop.set()
        with self._sdk_lock:
            self._close_handle_quiet()
        self._set_state("down", "closed")
        return True

    # ── Replug watcher → AutoRecover trigger (D405 hotplug parity) ───

    def _watch_start(self):
        if self._watch_thread is not None and self._watch_thread.is_alive():
            return
        self._watch_stop.clear()

        def _loop():
            # Full-time presence monitor — the uEye SDK has no hotplug
            # callback, so this poll is the D405-parity substitute:
            #
            # while OK:   two consecutive enumeration misses (~6 s) =
            #             unplugged. Go down NOW (no need to wait for a
            #             grab to fail and block on a dead handle) and
            #             RELEASE the handle — the daemon refuses to
            #             re-claim a re-plugged device while a stale
            #             handle exists.
            # while DOWN: device enumerable again -> fire the
            #             AutoRecover trigger (level-triggered with a
            #             cooldown; the daemon can keep a dropped device
            #             listed, so an absent->present edge is not
            #             reliable). Auto-reconnect, no user action.
            cooldown = 10.0
            last_fire = 0.0
            misses = 0
            while not self._watch_stop.wait(3.0):
                sn = self.serial_number
                if not sn:
                    continue
                if self.state == "ok":
                    # sysfs is the ground truth here: the daemon keeps a
                    # device LISTED while our own open handle holds its
                    # session, so its enumeration cannot see the unplug.
                    # (Limitation: sysfs has no per-device serial, so with
                    # several uEyes on one unit an unplug of ours is only
                    # caught when a grab fails — benches run one XS.)
                    if self._sysfs_present():
                        misses = 0
                        continue
                    misses += 1
                    if misses >= 2:
                        misses = 0
                        with self._sdk_lock:
                            self._close_handle_quiet()
                        self._set_state(
                            "down",
                            "device not detected on USB bus — unplugged?")
                elif self.state == "down":
                    # Reconnect needs the DAEMON to have re-claimed the
                    # device, so this side polls its enumeration.
                    misses = 0
                    present = any(d["serial_number"] == sn
                                  for d in self.all_device())
                    if present and (time.monotonic() - last_fire) >= cooldown:
                        last_fire = time.monotonic()
                        with self._listeners_lock:
                            cbs = list(self._available_listeners)
                        for cb in cbs:
                            try:
                                cb()
                            except Exception:
                                pass

        self._watch_thread = threading.Thread(target=_loop, daemon=True,
                                              name=f"ueye-watch-{self.serial_number}")
        self._watch_thread.start()

    # ── Recovery (Device protocol) ───────────────────────────────────

    @staticmethod
    def _sysfs_present():
        """Any USABLE IDS device (vendor 1409) on the USB bus, straight
        from sysfs — independent of the daemon's enumeration (which keeps
        a device listed while an open handle holds its session). A
        deauthorized device (authorized=0) counts as absent — it is
        unusable, and it is also how unplug is simulated in tests."""
        import glob
        for p in glob.glob("/sys/bus/usb/devices/*/idVendor"):
            try:
                if open(p).read().strip() != "1409":
                    continue
                base = p[: -len("idVendor")]
                try:
                    if open(base + "authorized").read().strip() == "0":
                        continue
                except Exception:
                    pass
                return True
            except Exception:
                pass
        return False

    def recover(self):
        with self._recover_lock:
            self._set_state("recovering", "reopening uEye handle")
            # Close FIRST: the daemon refuses to re-register a re-plugged
            # device while a stale handle exists, so checking enumeration
            # before closing can never succeed after a USB drop.
            with self._sdk_lock:
                self._close_handle_quiet()
            # Fast judgment: nothing on the USB bus at all -> fail NOW
            # (D405-parity fast-fail). Only when the device is physically
            # present but the daemon hasn't re-claimed it yet (it needed
            # our handle released first) is a short wait warranted.
            deadline = time.time() + 6.0
            while self.serial_number and time.time() < deadline:
                if any(d["serial_number"] == self.serial_number
                       for d in self.all_device()):
                    break
                if not self._sysfs_present():
                    self._set_state(
                        "down",
                        "device not detected on USB bus — reconnect the cable and retry")
                    return False
                time.sleep(0.5)
            else:
                if self.serial_number:
                    self._set_state(
                        "down",
                        "device not detected on USB bus — reconnect the cable and retry")
                    return False
            ok = self.connect(**self._connect_kwargs)
            return bool(ok)

    # ── Capture ──────────────────────────────────────────────────────

    def _grab_bgr(self, timeout_ms=10000):
        """Single FreezeVideo grab -> BGR8 ndarray (h, w, 3). Raises on
        failure and moves state to down — frame fetch is the single
        source of truth for camera health, same as the D405."""
        with self._sdk_lock:
            if self._h is None:
                self._set_state("down", "not connected")
                raise RuntimeError("uEye camera not connected")
            try:
                mem_ptr = ueye.c_mem_p()
                mem_id = ueye.c_int()
                w, h_px = self.width, self.height
                ret = ueye.is_AllocImageMem(self._h, w, h_px, 24, mem_ptr, mem_id)
                if ret != 0:
                    raise RuntimeError(f"is_AllocImageMem failed (code {ret})")
                try:
                    ueye.is_SetImageMem(self._h, mem_ptr, mem_id)
                    ret = ueye.is_FreezeVideo(self._h, ueye.c_int(int(timeout_ms)))
                    if ret != 0:
                        raise RuntimeError(f"is_FreezeVideo failed (code {ret})")
                    pitch = ueye.INT()
                    ueye.is_InquireImageMem(self._h, mem_ptr, mem_id,
                                            ueye.INT(), ueye.INT(), ueye.INT(), pitch)
                    arr = ueye.get_data(mem_ptr, w, h_px, 24, pitch, copy=True)
                finally:
                    ueye.is_FreeImageMem(self._h, mem_ptr, mem_id)
                return np.reshape(arr, (h_px, w, 3))
            except Exception as ex:
                # Release the handle IMMEDIATELY: the uEye daemon will not
                # re-register a re-plugged device while a stale handle to
                # it exists, so holding on would deadlock recovery (the
                # replug watcher polls enumeration, which stays empty
                # until the handle is freed).
                self._close_handle_quiet()
                self._set_state("down", f"frame grab failed: {ex}")
                raise

    def get_all(self, align_to=None, alpha=None):
        """Same 9-tuple as Camera.get_all — depth/ir slots are None
        (color-only device); depth_int is the effective intrinsics."""
        color_img = self._grab_bgr()
        if self.state != "ok":
            self._set_state("ok", "")
        depth_int = self.intr if self.intr is not None else self._nominal
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
        i = self.intr if self.intr is not None else self._nominal
        return [[i.fx, 0.0, i.ppx], [0.0, i.fy, i.ppy], [0.0, 0.0, 1.0]]

    def get_D(self, stream="color"):
        i = self.intr if self.intr is not None else self._nominal
        return [float(v) for v in i.coeffs]

    # ── Focus ────────────────────────────────────────────────────────

    def _focus_get_position(self):
        v = ueye.c_uint(0)
        ueye.is_Focus(self._h, ueye.FOC_CMD_GET_MANUAL_FOCUS, v, ueye.sizeof(v))
        return int(v.value)

    def _focus_set_position(self, pos):
        lo, hi = self.focus_range
        p = int(max(lo, min(hi, int(pos))))
        v = ueye.c_uint(p)
        ueye.is_Focus(self._h, ueye.FOC_CMD_SET_MANUAL_FOCUS, v, ueye.sizeof(v))
        return p

    def focus_info(self):
        out = {
            "supported": bool(self.focus_supported),
            "range": list(self.focus_range),
            "mode": self.focus_cfg.get("mode"),
            "position": None,
        }
        if self.focus_supported and self._h is not None:
            try:
                with self._sdk_lock:
                    out["position"] = self._focus_get_position()
            except Exception:
                pass
        return out

    def focus_apply(self, cfg, _locked=True):
        """Apply a focus config — idempotent, cheap when nothing changes.

        cfg: {"mode": "continuous"}                 SDK continuous AF
             {"mode": "once"}                       one-shot AF, then hold
             {"mode": "manual", "position": N}      pin the lens
        """
        if not cfg:
            return
        if self._h is None:
            raise RuntimeError("uEye camera not connected")
        cfg = dict(cfg)
        mode = cfg.get("mode", "manual" if "position" in cfg else "continuous")
        lock = self._sdk_lock if _locked else _NullCtx()
        with lock:
            if mode == "continuous":
                if self.focus_cfg.get("mode") == "continuous":
                    return
                ueye.is_Focus(self._h, ueye.FOC_CMD_SET_ENABLE_AUTOFOCUS, None, 0)
            elif mode == "once":
                # "once" is a VERB, not a state: converge the camera's AF
                # one time, then read where the lens landed and PIN it —
                # the resulting state is manual@position. The lens is
                # manual all the time; focusing is just how a position
                # gets chosen.
                ueye.is_Focus(self._h, ueye.FOC_CMD_SET_DISABLE_AUTOFOCUS, None, 0)
                ueye.is_Focus(self._h, ueye.FOC_CMD_SET_ENABLE_AUTOFOCUS_ONCE, None, 0)
                time.sleep(1.0)   # AF motor settle
                if self.focus_supported:
                    pos = self._focus_get_position()
                    self._focus_set_position(pos)
                    self.focus_cfg = {"mode": "manual", "position": int(pos)}
                    return
            elif mode == "manual":
                if "position" not in cfg:
                    raise ValueError('focus mode "manual" needs a "position"')
                if (self.focus_cfg.get("mode") == "manual"
                        and self.focus_cfg.get("position") == int(cfg["position"])):
                    return
                if not self.focus_supported:
                    raise RuntimeError("manual focus not supported by this device")
                ueye.is_Focus(self._h, ueye.FOC_CMD_SET_DISABLE_AUTOFOCUS, None, 0)
                self._focus_set_position(cfg["position"])
                time.sleep(0.15)  # lens settle
            else:
                raise ValueError(f"unknown focus mode: {mode!r}")
        self.focus_cfg = {"mode": mode, **({"position": int(cfg["position"])} if mode == "manual" else {})}

    def focus_once(self):
        self.focus_apply({"mode": "once"})

    # ── Exposure — the XS's ISP OWNS exposure (verified on hardware:
    #    every auto-shutter disable variant returns error 155 and manual
    #    sets are silently overridden). Auto-only; the live value is
    #    readable. The detection pipeline's software `intensity` is the
    #    knob for output brightness. ─────────────────────────────────

    def get_exposure(self):
        """Current exposure time in ms — whatever the ISP's auto chose."""
        if self._h is None:
            raise RuntimeError("uEye camera not connected")
        v = ueye.c_double(0)
        with self._sdk_lock:
            ueye.is_Exposure(self._h, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE,
                             v, ueye.sizeof(v))
        return round(float(v.value), 3)

    def set_exposure(self, exposure, _locked=True):
        raise RuntimeError(
            "the uEye XS ISP owns exposure — manual exposure is not "
            "supported on this device (auto only; read it with "
            "get_exposure(); use the detection's `intensity` for "
            "software brightness)")

    def auto_exposure(self, enable=True, _locked=True):
        """The XS is ALWAYS auto-exposure; enable=True is a no-op,
        enable=False is honestly refused."""
        if not enable:
            self.set_exposure(None)   # raises the same truth
        self.exposure_auto = True
        return True

    # ── White balance — two honest states (verified on hardware:
    #    kelvin / rgb multipliers are ISP-rejected on the XS):
    #        {"auto": True}   in-camera auto WB (connect default)
    #        {"hold": True}   freeze WB at its current convergence
    #    The bench recipe: let auto settle on the lit scene, then hold —
    #    deterministic color from then on (same philosophy as pinning
    #    focus). ────────────────────────────────────────────────────────

    def white_balance(self, cfg, _locked=True):
        if self._h is None:
            raise RuntimeError("uEye camera not connected")
        cfg = dict(cfg or {})
        if not (cfg.get("auto") or cfg.get("hold")):
            raise ValueError(
                'wb needs {"auto": True} or {"hold": True} — the XS ISP '
                "does not support fixed kelvin/rgb white balance; hold "
                "freezes the current auto convergence instead")
        on = ueye.c_double(0 if cfg.get("hold") else 1)
        lock = self._sdk_lock if _locked else _NullCtx()
        with lock:
            ret = ueye.is_SetAutoParameter(
                self._h, ueye.IS_SET_ENABLE_AUTO_SENSOR_WHITEBALANCE,
                on, ueye.c_double(0))
            if ret != 0:
                raise RuntimeError(f"white balance set failed (code {ret})")
        self.wb_cfg = {"hold": True} if cfg.get("hold") else {"auto": True}
        return dict(self.wb_cfg)

    def white_balance_info(self):
        out = {"cfg": dict(self.wb_cfg)}
        if self._h is not None:
            try:
                a, b = ueye.c_double(-1), ueye.c_double(0)
                with self._sdk_lock:
                    ret = ueye.is_SetAutoParameter(
                        self._h, ueye.IS_GET_ENABLE_AUTO_SENSOR_WHITEBALANCE, a, b)
                if ret == 0:
                    out["auto"] = bool(a.value)
            except Exception:
                pass
        return out

    def _region_sharpness(self, rect):
        x0, y0, x1, y1 = rect
        x0 = max(0, min(self.width - 1, int(x0)))
        y0 = max(0, min(self.height - 1, int(y0)))
        x1 = max(x0 + 1, min(self.width, int(x1)))
        y1 = max(y0 + 1, min(self.height, int(y1)))
        img = self._grab_bgr()
        gray = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        # Score at half resolution with a Gaussian pre-blur (pyrDown does
        # both): suppresses the sensor-noise variance that flattens the
        # peak in dim scenes, and matches what these optics actually
        # resolve. Verified on hardware: much stronger peak-to-floor
        # discrimination than full-res Laplacian, at 1/4 the compute.
        if gray.shape[0] >= 32 and gray.shape[1] >= 32:
            gray = cv2.pyrDown(gray)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _af_aoi(self, x0, y0, w, h):
        """Point the camera's own AF measurement window at a rect."""
        aoi = ueye.AUTOFOCUS_AOI()
        aoi.uNumberAOI = ueye.c_uint(0)
        aoi.rcAOI.s32X = ueye.c_int(int(x0))
        aoi.rcAOI.s32Y = ueye.c_int(int(y0))
        aoi.rcAOI.s32Width = ueye.c_int(int(w))
        aoi.rcAOI.s32Height = ueye.c_int(int(h))
        aoi.eWeight = ueye.AUTOFOCUS_AOI_WEIGHT_MIDDLE
        return ueye.is_Focus(self._h, ueye.FOC_CMD_SET_AUTOFOCUS_AOI,
                             aoi, ueye.sizeof(aoi))

    def _af_region_once(self, x0, y0, w, h, timeout_s=5.0):
        """Hardware region AF: set the AF AOI, one-shot the camera's own
        autofocus, wait for FOC_STATUS_FOCUSED, read where the lens
        landed and pin it there. ~1-2 s. Returns the position, or None
        when the device rejects the AOI (caller falls back to the sweep).
        The AOI is restored to full frame afterwards so later AF/Auto
        calls behave normally."""
        with self._sdk_lock:
            if self._af_aoi(x0, y0, w, h) != 0:
                return None
            ueye.is_Focus(self._h, ueye.FOC_CMD_SET_DISABLE_AUTOFOCUS, None, 0)
            ueye.is_Focus(self._h, ueye.FOC_CMD_SET_ENABLE_AUTOFOCUS_ONCE, None, 0)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            time.sleep(0.3)
            st = ueye.c_uint(0)
            with self._sdk_lock:
                r = ueye.is_Focus(self._h, ueye.FOC_CMD_GET_AUTOFOCUS_STATUS,
                                  st, ueye.sizeof(st))
            if r == 0 and st.value in (int(ueye.FOC_STATUS_FOCUSED),
                                       int(ueye.FOC_STATUS_TIMEOUT),
                                       int(ueye.FOC_STATUS_ERROR)):
                break
        with self._sdk_lock:
            pos = self._focus_get_position()
            self._focus_set_position(pos)      # pin exactly where AF landed
            self._af_aoi(0, 0, self.width, self.height)   # AOI back to full
        return int(pos)

    def focus_region(self, rect, coarse_steps=12, fine_steps=8, settle_s=0.15,
                     method="auto"):
        """Region autofocus on ``rect`` = [x0, y0, x1, y1] (full-res
        pixels). Ends with the camera in MANUAL focus at the found
        position and returns {"position", "sharpness", "method"}.

        method:
          "auto"  (default) — the camera's own hardware AF pointed at the
                  rect via the AF AOI (~1-2 s); falls back to the sweep
                  if the device rejects the AOI.
          "af"    hardware AF AOI only (raises if unsupported).
          "sweep" the 3-pass manual-lens sweep scored by Laplacian
                  variance inside the rect (~15-30 s) — deterministic,
                  uses OUR sharpness metric, needs no AF support.
        """
        if not self.focus_supported:
            raise RuntimeError("manual focus not supported by this device")
        if self._h is None:
            raise RuntimeError("uEye camera not connected")

        x0, y0, x1, y1 = [int(v) for v in rect]
        x0, x1 = max(0, min(x0, x1)), min(self.width, max(x0, x1))
        y0, y1 = max(0, min(y0, y1)), min(self.height, max(y0, y1))

        if method in ("auto", "af"):
            pos = self._af_region_once(x0, y0, max(1, x1 - x0), max(1, y1 - y0))
            if pos is not None:
                self.focus_cfg = {"mode": "manual", "position": int(pos)}
                score = self._region_sharpness([x0, y0, x1, y1])
                lo, hi = self.focus_range
                return {"position": int(pos), "sharpness": round(score, 1),
                        "method": "af", "at_range_limit": pos in (lo, hi)}
            if method == "af":
                raise RuntimeError("this device rejected the autofocus AOI")

        lo, hi = self.focus_range
        with self._sdk_lock:
            ueye.is_Focus(self._h, ueye.FOC_CMD_SET_DISABLE_AUTOFOCUS, None, 0)

        samples = {}                      # position -> score (dedup free)

        def measure(p):
            p = int(max(lo, min(hi, p)))
            if p in samples:
                return samples[p]
            with self._sdk_lock:
                self._focus_set_position(p)
            time.sleep(settle_s)
            samples[p] = self._region_sharpness(rect)
            return samples[p]

        # Pass 1: coarse across the whole lens range.
        coarse_step = max(1, (hi - lo) // (coarse_steps - 1))
        for p in np.linspace(lo, hi, coarse_steps).astype(int).tolist():
            measure(p)
        peak = max(samples, key=samples.get)

        # Pass 2: fine around the coarse peak.
        fine_lo, fine_hi = max(lo, peak - coarse_step), min(hi, peak + coarse_step)
        for p in np.linspace(fine_lo, fine_hi, fine_steps).astype(int).tolist():
            measure(p)
        peak = max(samples, key=samples.get)

        # Pass 3: parabolic refine instead of an exhaustive unit-step
        # walk — fit a parabola through the peak and its measured
        # neighbors, verify the vertex and the integer positions around
        # it. Lens positions are integers, so this lands exactly where
        # the walk would, in a fraction of the grabs.
        cand = {peak - 1, peak + 1}
        ordered = sorted(samples)
        i = ordered.index(peak)
        if 0 < i < len(ordered) - 1:
            p0, p1, p2 = ordered[i - 1], peak, ordered[i + 1]
            s0, s1, s2 = samples[p0], samples[p1], samples[p2]
            denom = (p1 - p0) * (s1 - s2) - (p1 - p2) * (s1 - s0)
            if abs(denom) > 1e-9:
                vertex = p1 - 0.5 * ((p1 - p0) ** 2 * (s1 - s2)
                                     - (p1 - p2) ** 2 * (s1 - s0)) / denom
                v = int(round(vertex))
                cand.update({v - 1, v, v + 1})
        for p in sorted(cand):
            if lo <= p <= hi:
                measure(p)

        peak = max(samples, key=samples.get)
        score = samples[peak]

        with self._sdk_lock:
            self._focus_set_position(peak)
        self.focus_cfg = {"mode": "manual", "position": int(peak)}
        # A peak ON a range endpoint is not a real optimum — the true
        # best focus lies beyond the lens's travel (target closer than
        # the minimum focus distance, typically). Say so.
        return {"position": int(peak), "sharpness": round(score, 1),
                "method": "sweep", "at_range_limit": peak in (lo, hi)}


class _NullCtx(object):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
