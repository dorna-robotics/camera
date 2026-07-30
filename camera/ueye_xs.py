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
  and manual positions (~0..255). ``focus_region(rect)`` runs the
  region-tune sweep (coarse -> fine -> micro, scored by Laplacian
  variance inside ``rect``) and settles on the sharpest manual
  position — the playground's 'r' key, as an API.
- **Hotplug.** The uEye SDK has no librealsense-style hotplug callback
  here; a lightweight watcher polls enumeration ~every 3 s WHILE DOWN
  and fires ``on_hardware_available`` when the serial reappears — so
  the pool's AutoRecover loop heals a replug exactly like the D405.

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

        # focus state
        self.focus_supported = False
        self.focus_range = (0, 0)
        self.focus_cfg = {"mode": "continuous"}

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
    def all_device():
        """Attached uEye devices: [{serial_number, name, dev_id,
        camera_type, in_use}]. Empty list when the SDK is absent."""
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
        exposure=None,        # ms, None = camera auto (hands-off)
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
            mode=mode, filter=filter, max_tries=max_tries,
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
                self._open(dev, K, D, native_res, focus, exposure)
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

    def _open(self, dev, K, D, native_res, focus, exposure):
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

        # hands-off color: auto white balance on, gains at defaults
        ueye.is_SetAutoParameter(h, ueye.IS_SET_ENABLE_AUTO_WHITEBALANCE,
                                 ueye.c_double(1), ueye.c_double(0))
        # exposure: authored value = manual; None = camera auto
        if exposure is not None:
            exp = ueye.c_double(float(exposure))
            ueye.is_Exposure(h, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE,
                             exp, ueye.sizeof(exp))
        else:
            ueye.is_SetAutoParameter(h, ueye.IS_SET_ENABLE_AUTO_SHUTTER,
                                     ueye.c_double(1), ueye.c_double(0))

        # focus capability probe
        fmin, fmax = ueye.c_uint(0), ueye.c_uint(0)
        r1 = ueye.is_Focus(h, ueye.FOC_CMD_GET_MANUAL_FOCUS_MIN, fmin, ueye.sizeof(fmin))
        r2 = ueye.is_Focus(h, ueye.FOC_CMD_GET_MANUAL_FOCUS_MAX, fmax, ueye.sizeof(fmax))
        self.focus_supported = (r1 == 0 and r2 == 0 and fmax.value > fmin.value)
        self.focus_range = (int(fmin.value), int(fmax.value))
        self.focus_apply(focus or {"mode": "continuous"}, _locked=False)

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
            was_present = True
            while not self._watch_stop.wait(3.0):
                if self.state != "down" or not self.serial_number:
                    was_present = True
                    continue
                present = any(d["serial_number"] == self.serial_number
                              for d in self.all_device())
                if present and not was_present:
                    with self._listeners_lock:
                        cbs = list(self._available_listeners)
                    for cb in cbs:
                        try:
                            cb()
                        except Exception:
                            pass
                was_present = present

        self._watch_thread = threading.Thread(target=_loop, daemon=True,
                                              name=f"ueye-watch-{self.serial_number}")
        self._watch_thread.start()

    # ── Recovery (Device protocol) ───────────────────────────────────

    def recover(self):
        with self._recover_lock:
            self._set_state("recovering", "reopening uEye handle")
            if self.serial_number and not any(
                d["serial_number"] == self.serial_number for d in self.all_device()
            ):
                self._set_state(
                    "down",
                    "device not detected on USB bus — reconnect the cable and retry")
                return False
            with self._sdk_lock:
                self._close_handle_quiet()
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
                ueye.is_Focus(self._h, ueye.FOC_CMD_SET_DISABLE_AUTOFOCUS, None, 0)
                ueye.is_Focus(self._h, ueye.FOC_CMD_SET_ENABLE_AUTOFOCUS_ONCE, None, 0)
                time.sleep(1.0)   # AF motor settle
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

    def _region_sharpness(self, rect):
        x0, y0, x1, y1 = rect
        x0 = max(0, min(self.width - 1, int(x0)))
        y0 = max(0, min(self.height - 1, int(y0)))
        x1 = max(x0 + 1, min(self.width, int(x1)))
        y1 = max(y0 + 1, min(self.height, int(y1)))
        img = self._grab_bgr()
        gray = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def focus_region(self, rect, coarse_steps=12, fine_steps=8, settle_s=0.15):
        """Region autofocus: 3-pass manual-lens sweep scored by Laplacian
        variance inside ``rect`` = [x0, y0, x1, y1] (full-res pixels).
        Settles on the sharpest position, switches the camera to manual
        focus, and returns {"position", "sharpness"}. Slow (~15-30 s) —
        run it from a tuning flow, then PIN the returned position (in
        the camera cfg or a detection's ``focus``) for production."""
        if not self.focus_supported:
            raise RuntimeError("manual focus not supported by this device")
        if self._h is None:
            raise RuntimeError("uEye camera not connected")
        lo, hi = self.focus_range
        with self._sdk_lock:
            ueye.is_Focus(self._h, ueye.FOC_CMD_SET_DISABLE_AUTOFOCUS, None, 0)

        def sweep(positions):
            best_pos, best_score = positions[0], -1.0
            for p in positions:
                with self._sdk_lock:
                    self._focus_set_position(p)
                time.sleep(settle_s)
                score = self._region_sharpness(rect)
                if score > best_score:
                    best_pos, best_score = p, score
            return best_pos, best_score

        coarse = np.linspace(lo, hi, coarse_steps).astype(int).tolist()
        coarse_step = max(1, (hi - lo) // (coarse_steps - 1))
        peak, _ = sweep(coarse)

        fine_lo, fine_hi = max(lo, peak - coarse_step), min(hi, peak + coarse_step)
        fine = np.linspace(fine_lo, fine_hi, fine_steps).astype(int).tolist()
        peak, _ = sweep(fine)

        fine_step = max(1, (fine_hi - fine_lo) // (fine_steps - 1))
        micro = list(range(max(lo, peak - fine_step), min(hi, peak + fine_step) + 1))
        peak, score = sweep(micro)

        with self._sdk_lock:
            self._focus_set_position(peak)
        self.focus_cfg = {"mode": "manual", "position": int(peak)}
        return {"position": int(peak), "sharpness": round(score, 1)}


class _NullCtx(object):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
