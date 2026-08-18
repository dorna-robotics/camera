"""Hikrobot GigE (PoE) driver — the platform's third camera type.

Duck-types the same surface :class:`camera.Camera` (RealSense D405) and
:class:`camera.UEyeXS` expose, so ``CameraPool`` / ``MQTTDeviceAdapter``
/ ``Detection`` treat all three alike:

  Device protocol   id / state / msg / on_state_change /
                    on_hardware_available / recover / release
  Capture           connect / get_all / frame-shaped 9-tuple / close
  Intrinsics        camera_matrix / dist_coeffs / get_K / get_D

Differences from the D405, stated instead of papered over:

- **Color only.** No depth, no IR — ``get_all`` returns ``None`` in the
  depth/ir slots (the 9-tuple shape is preserved, same as a D405 with
  those channels unsubscribed). Depth-dependent calls (``xyz``) fail
  loudly downstream.
- **Network camera.** Connects over GigE Vision (PoE) — address it by
  ``ip=`` (e.g. "10.0.1.50") or ``serial_number=``. Enumeration is a
  UDP broadcast discovery on the local segment; a device that is
  streaming still answers discovery, so the presence monitor works
  while we hold the camera open (unlike the uEye daemon quirk).
- **Intrinsics are authored or nominal.** A C-mount body takes an
  interchangeable lens, so there is no meaningful factory pinhole to
  ship as a default. Author ``K``/``D``/``native_res`` in ``camera_cfg``
  for metric work; without them a PLACEHOLDER pinhole (focal = image
  width ≈ 53° HFOV, centered principal point, zero distortion) is
  reported and labeled ``"nominal"`` — good enough for drawing, not for
  measuring.
- **Exposure / gain are real knobs.** Unlike the uEye XS, the sensor is
  fully controllable: ``exposure`` in µs pins ExposureAuto=Off +
  ExposureTime (same unit the D405 uses); ``exposure=None`` leaves the
  camera in continuous auto. ``gain`` (dB) works the same way.
- **White balance.** ``{"auto": True}`` continuous, ``{"once": True}``
  converge then hold, ``{"hold": True}`` freeze the current ratios.
- **Hotplug.** GigE has no cable-level hotplug callback, so a presence
  monitor polls broadcast discovery every 3 s: two consecutive misses
  (~6 s) → ``down`` and the handle is released; while down, the device
  answering discovery again fires ``on_hardware_available`` (cooldown
  10 s) so the pool's AutoRecover reconnects without user action —
  same timings as the D405/uEye paths.

The MVS runtime (``MvCameraControl`` Python bindings, installed by
Hikrobot's MVS package under ``/opt/MVS``) is only required to USE this
class — importing the module without it works, enumeration returns [],
and connect() raises with an actionable message.
"""

import os
import sys
import ctypes
import threading
import time

import numpy as np

from .camera import Helper


def _mvimport_paths():
    """Where Hikrobot's ``MvImport`` Python bindings may live.

    MVS does not pip-install its Python API — it drops a folder of
    ctypes wrappers (``MvImport/`` with ``MvCameraControl_class.py``)
    inside its install tree, and every app is expected to put that
    folder on ``sys.path`` itself. Search order, most deliberate first:

      1. ``<repo>/mvs/MvImport`` — a copy vendored into this repo
         (same standalone pattern as the IDS debs in ``ids/``; see
         ``mvs/README.md``). Copy the folder from any MVS install and
         the platform needs no path guessing at all.
      2. ``MVCAM_SDK_PATH`` — the env var the Linux MVS installer
         sets (=/opt/MVS); both the Linux and Windows tree layouts
         are probed under it in case it was set by hand.
      3. Default install trees for the current OS: ``/opt/MVS`` on
         Linux (Pi), ``Program Files[ (x86)]\\MVS`` on Windows.

    Only the *bindings* are located here. The MVS runtime library they
    wrap (``libMvCameraControl.so`` / ``MvCameraControl.dll``) must be
    installed on the machine either way — the bindings are wrappers,
    not the driver."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    cands = [os.path.join(repo, "mvs", "MvImport")]

    sdk = os.environ.get("MVCAM_SDK_PATH")
    if sdk:
        cands += [
            os.path.join(sdk, "Samples", "64", "Python", "MvImport"),
            os.path.join(sdk, "Samples", "aarch64", "Python", "MvImport"),
            os.path.join(sdk, "Development", "Samples", "Python", "MvImport"),
        ]

    if os.name == "nt":
        for pf in (os.environ.get("ProgramFiles(x86)"),
                   os.environ.get("ProgramFiles")):
            if pf:
                cands.append(os.path.join(
                    pf, "MVS", "Development", "Samples", "Python", "MvImport"))
    else:
        cands += [
            "/opt/MVS/Samples/64/Python/MvImport",        # x86_64 Linux
            "/opt/MVS/Samples/aarch64/Python/MvImport",   # Pi / arm64
            "/opt/MVS/Samples/Python/MvImport",           # older layouts
        ]
    return cands


def _win_register_runtime_dirs():
    """Windows: make sure the MVS runtime DLL is findable even when this
    process was launched BEFORE MVS was installed (its PATH predates the
    installer's edit — a very common Jupyter/VS Code situation; a kernel
    restart doesn't help because the kernel inherits the stale env from
    its parent). Probe the env var and the default install locations and
    register whichever exists, both ways: os.add_dll_directory (the
    Python 3.8+ mechanism) and a PATH prepend (dependent DLLs)."""
    if os.name != "nt":
        return
    cands = []
    env = os.environ.get("MVCAM_COMMON_RUNENV")
    if env:
        cands.append(os.path.join(env, "Win64_x64"))
    for cf in (os.environ.get("CommonProgramFiles(x86)"),
               os.environ.get("CommonProgramFiles")):
        if cf:
            cands.append(os.path.join(cf, "MVS", "Runtime", "Win64_x64"))
    for d in cands:
        if os.path.isfile(os.path.join(d, "MvCameraControl.dll")):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass
            if d not in os.environ.get("PATH", ""):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


# Import the bindings: plain import first (already on PYTHONPATH or
# vendored next to the caller), then extend sys.path with the known
# locations and retry. Failure leaves _mv=None — the module stays
# importable, enumeration returns [], connect() raises actionably.
_win_register_runtime_dirs()
try:
    import MvCameraControl_class as _mv
    _MVS_ERR = None
except Exception:
    for _p in _mvimport_paths():
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.append(_p)
    try:
        import MvCameraControl_class as _mv
        _MVS_ERR = None
    except Exception as _ex:      # ImportError or missing runtime library
        _mv = None
        _MVS_ERR = str(_ex)


_MV_INIT_LOCK = threading.Lock()
_MV_INITED = False


def _mv_ensure_init():
    """One-time SDK init (newer MVS releases require MV_CC_Initialize;
    older ones don't have it — both handled)."""
    global _MV_INITED
    if _mv is None or _MV_INITED:
        return
    with _MV_INIT_LOCK:
        if _MV_INITED:
            return
        try:
            if hasattr(_mv.MvCamera, "MV_CC_Initialize"):
                _mv.MvCamera.MV_CC_Initialize()
        except Exception:
            pass
        _MV_INITED = True


def _ip_to_str(n):
    """GigE discovery reports the IP as a host-order uint32 with the
    first octet in the top byte (the MVS samples decode it this way)."""
    return ".".join(str((int(n) >> s) & 0xFF) for s in (24, 16, 8, 0))


def _ip_to_int(ip_str):
    a, b, c, d = (int(x) for x in str(ip_str).split("."))
    return (a << 24) | (b << 16) | (c << 8) | d


def _local_ip_for(dst_ip):
    """The local NIC address the OS would route to ``dst_ip`` from —
    the connect() trick sends no traffic, it only resolves the route."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((dst_ip, 1))
        return s.getsockname()[0]
    finally:
        s.close()


def _ping(ip, timeout_s=1):
    """One ICMP ping, cross-platform, quiet. Used as the presence probe
    for cross-subnet cameras that broadcast discovery cannot see."""
    import subprocess
    if os.name == "nt":
        cmd = ["ping", "-n", "1", "-w", str(int(timeout_s * 1000)), ip]
    else:
        cmd = ["ping", "-c", "1", "-W", str(int(timeout_s)), ip]
    try:
        return subprocess.run(cmd, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL,
                              timeout=timeout_s + 2).returncode == 0
    except Exception:
        return False


def _cstr(buf):
    """NUL-terminated ctypes char/ubyte array -> str."""
    return bytes(bytearray(buf)).partition(b"\0")[0].decode(errors="replace")


_ERR_NAMES = None


def _err(ret):
    """Human-readable MVS error: 'MV_E_UDP_RECV_DATA (0x80000214)'
    instead of a bare hex code — the names come from the vendored
    MvErrorDefine constants."""
    global _ERR_NAMES
    code = ret & 0xFFFFFFFF
    if _ERR_NAMES is None and _mv is not None:
        try:
            _ERR_NAMES = {v & 0xFFFFFFFF: k for k, v in vars(_mv).items()
                          if k.startswith("MV_E_") and isinstance(v, int)}
        except Exception:
            _ERR_NAMES = {}
    name = (_ERR_NAMES or {}).get(code)
    return f"{name} (0x{code:08x})" if name else f"0x{code:08x}"


class _HikIntrinsics(object):
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


class HikRobot(Helper):
    # Inherits Helper's pure projection math (pixel/xyz_estimate) — the
    # same surface Detection calls on the D405 class. Helper.xyz needs
    # depth and honestly fails on this color-only device.

    camera_type = "hikrobot"

    def __init__(self):
        super().__init__()
        self.serial_number = None
        self.ip = None
        self.state = "down"
        self.msg = "not connected"
        self._listeners = []
        self._available_listeners = []
        self._listeners_lock = threading.Lock()

        self._cam = None               # MvCamera handle, None when closed
        self._buf = None               # reusable BGR grab buffer
        self._sdk_lock = threading.Lock()
        self._connect_kwargs = {}
        self._recover_lock = threading.Lock()
        self._cross_subnet = False     # opened by IP across a router?

        self.width = 0
        self.height = 0
        self.intr = None               # authored K/D shim, else None
        self.intr_source = "nominal"
        self.mode = "bgr"
        self.stream = None             # {"width","height","fps"} actual
        self.stream_actual = None
        self.filter = {}
        # Color-only device — camera_list reports this so channel pickers
        # (playground Source) offer exactly what the camera has.
        self._enabled_channels = {"color"}

        self.exposure_auto = True
        self.wb_cfg = {"auto": True}

        # replug watcher (presence monitor over GigE discovery)
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
    def _enum_raw():
        """[(MV_CC_DEVICE_INFO struct, info dict)] for attached GigE
        devices. The struct is what MV_CC_CreateHandle needs; the dict is
        what callers see. Empty when the SDK is absent."""
        if _mv is None:
            return []
        _mv_ensure_init()
        try:
            dl = _mv.MV_CC_DEVICE_INFO_LIST()
            if _mv.MvCamera.MV_CC_EnumDevices(_mv.MV_GIGE_DEVICE, dl) != 0:
                return []
            out = []
            for i in range(int(dl.nDeviceNum)):
                st = ctypes.cast(dl.pDeviceInfo[i],
                                 ctypes.POINTER(_mv.MV_CC_DEVICE_INFO)).contents
                gige = st.SpecialInfo.stGigEInfo
                out.append((st, {
                    "serial_number": _cstr(gige.chSerialNumber),
                    "name": _cstr(gige.chModelName),
                    "user_name": _cstr(gige.chUserDefinedName),
                    "ip": _ip_to_str(gige.nCurrentIp),
                    "camera_type": "hikrobot",
                }))
            return out
        except Exception:
            return []

    @staticmethod
    def all_device():
        """Attached Hikrobot GigE devices: [{serial_number, name,
        user_name, ip, camera_type}]. Empty when the SDK is absent."""
        return [d for _, d in HikRobot._enum_raw()]

    # ── Connect / close ──────────────────────────────────────────────

    def connect(
        self,
        serial_number="",
        ip="",                # GigE address, e.g. "10.0.1.50" — either key selects
        stream=None,          # {"width","height","fps"} — best-effort ROI/fps, actuals read back
        K=None,
        D=None,
        native_res=None,
        exposure=None,        # µs (same unit as the D405); None = continuous auto
        gain=None,            # dB; None = continuous auto
        wb=None,              # {"auto": True} | {"once": True} | {"hold": True}
        mode="bgr",
        filter={},
        max_tries=3,
        raise_on_fail=False,
        **_ignored,           # tolerate D405-only keys (channels, ...)
    ):
        if _mv is None:
            where = (r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport"
                     if os.name == "nt" else
                     "/opt/MVS/Samples/64/Python/MvImport")
            msg = ("Hikrobot MVS runtime not available (%s) — install MVS "
                   "from hikrobotics.com (it ships the Python bindings under "
                   "%s), then restart this process/kernel. Alternatively "
                   "vendor the MvImport folder into <repo>/mvs/MvImport — "
                   "see mvs/README.md" % (_MVS_ERR, where))
            self._set_state("down", msg)
            if raise_on_fail:
                raise RuntimeError(msg)
            return False

        self._connect_kwargs = dict(
            serial_number=serial_number, ip=ip, stream=stream, K=K, D=D,
            native_res=native_res, exposure=exposure, gain=gain, wb=wb,
            mode=mode, filter=filter, max_tries=max_tries,
        )
        self.filter = filter
        self.mode = mode

        devs = self._enum_raw()
        match = None
        if serial_number:
            match = [(s, d) for s, d in devs
                     if d["serial_number"] == str(serial_number)]
        elif ip:
            match = [(s, d) for s, d in devs if d["ip"] == str(ip)]
        else:
            match = devs

        self._cross_subnet = False
        if match:
            dev_struct, dev = match[0]
            self.serial_number = dev["serial_number"]
            self.ip = dev["ip"]
        elif ip and _ping(str(ip)):
            # Cross-subnet fallback: broadcast discovery is subnet-local,
            # but the camera is routable (ping answers) — e.g. host on
            # 10.0.3.x, camera on 10.0.1.x behind a router. Build the
            # GigE device descriptor by hand and open unicast. The serial
            # is read from the device after open. Presence monitoring
            # switches to ICMP ping (discovery can't see the device).
            # NOTE: routed paths (especially Wi-Fi) drop UDP stream
            # packets — fine for bench tests, use a same-subnet NIC for
            # production streaming.
            dev_struct = self._device_info_for_ip(str(ip))
            dev = None
            self.serial_number = None       # filled after open
            self.ip = str(ip)
            self._cross_subnet = True
        else:
            what = (f"serial={serial_number}" if serial_number
                    else f"ip={ip}" if ip else "any")
            msg = (f"Hikrobot device not found ({what}) — not in broadcast "
                   f"discovery and not answering ping. Check PoE link/power "
                   f"and that the camera is reachable from this host")
            self._set_state("down", msg)
            if raise_on_fail:
                raise RuntimeError(msg)
            return False

        last_ex = None
        for _ in range(max_tries):
            try:
                self._open(dev_struct, stream, K, D, native_res,
                           exposure, gain, wb)
                # cross-subnet open: discovery gave us no serial — read
                # it from the device so id/presence tracking work.
                if not self.serial_number:
                    try:
                        self.serial_number = self._read_string("DeviceSerialNumber")
                    except Exception:
                        self.serial_number = self.ip   # last-resort stable id
                # end-to-end verification — one real frame
                self._grab_bgr()
                self._set_state("ok", "")
                self._watch_start()
                return True
            except Exception as ex:
                last_ex = ex
                self._close_handle_quiet()
                time.sleep(0.5)
        msg = f"Hikrobot connect failed: {last_ex}"
        self._set_state("down", msg)
        if raise_on_fail:
            raise RuntimeError(msg)
        return False

    # feature-access helpers (handle the MVS get-struct dance once)

    def _get_int(self, key):
        v = _mv.MVCC_INTVALUE()
        ctypes.memset(ctypes.byref(v), 0, ctypes.sizeof(v))
        if self._cam.MV_CC_GetIntValue(key, v) != 0:
            raise RuntimeError(f"GetIntValue({key}) failed")
        return int(v.nCurValue)

    def _get_float(self, key):
        v = _mv.MVCC_FLOATVALUE()
        ctypes.memset(ctypes.byref(v), 0, ctypes.sizeof(v))
        if self._cam.MV_CC_GetFloatValue(key, v) != 0:
            raise RuntimeError(f"GetFloatValue({key}) failed")
        return float(v.fCurValue)

    def _read_string(self, key):
        v = _mv.MVCC_STRINGVALUE()
        ctypes.memset(ctypes.byref(v), 0, ctypes.sizeof(v))
        if self._cam.MV_CC_GetStringValue(key, v) != 0:
            raise RuntimeError(f"GetStringValue({key}) failed")
        return _cstr(v.chCurValue)

    @staticmethod
    def _device_info_for_ip(ip_str):
        """Hand-built GigE device descriptor for a camera broadcast
        discovery cannot see (different subnet, routed path). nNetExport
        tells the SDK which local NIC to bind — resolved from the OS
        routing table."""
        st = _mv.MV_CC_DEVICE_INFO()
        ctypes.memset(ctypes.byref(st), 0, ctypes.sizeof(st))
        st.nTLayerType = _mv.MV_GIGE_DEVICE
        st.SpecialInfo.stGigEInfo.nCurrentIp = _ip_to_int(ip_str)
        try:
            st.SpecialInfo.stGigEInfo.nNetExport = _ip_to_int(_local_ip_for(ip_str))
        except Exception:
            pass
        return st

    def _present(self):
        """Is the camera visible right now? Broadcast discovery on the
        local subnet; ICMP ping for cross-subnet cameras discovery
        cannot see."""
        sn = self.serial_number
        if sn and any(d["serial_number"] == sn for d in self.all_device()):
            return True
        if self._cross_subnet and self.ip:
            return _ping(self.ip)
        return False

    def _open(self, dev_struct, stream, K, D, native_res, exposure, gain, wb):
        cam = _mv.MvCamera()
        ret = cam.MV_CC_CreateHandle(dev_struct)
        if ret != 0:
            raise RuntimeError(f"MV_CC_CreateHandle failed ({_err(ret)})")
        self._cam = cam
        ret = cam.MV_CC_OpenDevice(_mv.MV_ACCESS_Exclusive, 0)
        if ret != 0:
            # exclusive GigE control — a second host/process cannot open it
            raise RuntimeError(
                f"MV_CC_OpenDevice failed ({_err(ret)}) — "
                f"is another process/host holding the camera?")

        # GigE tuning: jumbo-aware packet size (big win on PoE links)
        try:
            psize = cam.MV_CC_GetOptimalPacketSize()
            if int(psize) > 0:
                cam.MV_CC_SetIntValue("GevSCPSPacketSize", int(psize))
        except Exception:
            pass

        # free-run, no trigger
        cam.MV_CC_SetEnumValue("TriggerMode", _mv.MV_TRIGGER_MODE_OFF)

        # stream config — best-effort: sensors have step constraints, so
        # every set is attempted and the ACTUALS are read back (honest
        # stream_actual, same contract as the D405 fallback ladder).
        if stream:
            for key in ("Width", "Height"):
                want = stream.get(key.lower())
                if want:
                    try:
                        cam.MV_CC_SetIntValue(key, int(want))
                    except Exception:
                        pass
            fps = stream.get("fps")
            if fps:
                try:
                    cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
                    cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(fps))
                except Exception:
                    pass

        self.width = self._get_int("Width")
        self.height = self._get_int("Height")
        try:
            actual_fps = round(self._get_float("ResultingFrameRate"), 2)
        except Exception:
            actual_fps = None
        self.stream = {"width": self.width, "height": self.height,
                       "fps": actual_fps}
        self.stream_actual = dict(self.stream)

        # exposure: manual (µs) pins ExposureAuto=Off; None = continuous auto
        if exposure is not None:
            self.set_exposure(exposure, _locked=False)
        else:
            cam.MV_CC_SetEnumValue("ExposureAuto", 2)   # Continuous
            self.exposure_auto = True

        # gain: manual (dB) pins GainAuto=Off; None = continuous auto
        try:
            if gain is not None:
                cam.MV_CC_SetEnumValue("GainAuto", 0)
                cam.MV_CC_SetFloatValue("Gain", float(gain))
            else:
                cam.MV_CC_SetEnumValue("GainAuto", 2)
        except Exception:
            pass   # mono/entry models without a Gain node

        # white balance: authored cfg else continuous auto
        self.white_balance(wb or {"auto": True}, _locked=False)

        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"MV_CC_StartGrabbing failed ({_err(ret)})")

        # intrinsics: authored (scaled from native_res) else nominal
        # placeholder — a C-mount lens is interchangeable, so there is no
        # honest default focal; width ≈ 53° HFOV is a drawing-grade guess.
        if K is not None and D is not None:
            K_, D_ = np.array(K, dtype=float), np.array(D, dtype=float)
            sx = sy = 1.0
            if native_res is not None:
                sx = self.width / float(native_res[0])
                sy = self.height / float(native_res[1])
            self.intr = _HikIntrinsics(
                self.width, self.height,
                K_[0, 0] * sx, K_[1, 1] * sy, K_[0, 2] * sx, K_[1, 2] * sy,
                list(D_[:5]))
            self.intr_source = "override"
        else:
            self.intr = None
            self.intr_source = "nominal"
        self._nominal = _HikIntrinsics(
            self.width, self.height,
            float(self.width), float(self.width),
            self.width / 2.0, self.height / 2.0)

    def _close_handle_quiet(self):
        cam, self._cam = self._cam, None
        self._buf = None
        if cam is not None and _mv is not None:
            for call in ("MV_CC_StopGrabbing", "MV_CC_CloseDevice",
                         "MV_CC_DestroyHandle"):
                try:
                    getattr(cam, call)()
                except Exception:
                    pass

    def close(self):
        self._watch_stop.set()
        with self._sdk_lock:
            self._close_handle_quiet()
        self._set_state("down", "closed")
        return True

    # ── Presence monitor → AutoRecover trigger (D405 hotplug parity) ─

    def _watch_start(self):
        if self._watch_thread is not None and self._watch_thread.is_alive():
            return
        self._watch_stop.clear()

        def _loop():
            # GigE discovery is a UDP broadcast the camera firmware
            # answers regardless of who is streaming, so ONE poll serves
            # both directions (no sysfs split like the uEye):
            #
            # while OK:   two consecutive discovery misses (~6 s) =
            #             link/power lost. Go down NOW (no need to wait
            #             for a grab to time out) and release the handle.
            # while DOWN: device answering discovery again -> fire the
            #             AutoRecover trigger (level-triggered with a
            #             cooldown). Auto-reconnect, no user action.
            cooldown = 10.0
            last_fire = 0.0
            misses = 0
            while not self._watch_stop.wait(3.0):
                if not (self.serial_number or self.ip):
                    continue
                present = self._present()
                if self.state == "ok":
                    if present:
                        misses = 0
                        continue
                    misses += 1
                    if misses >= 2:
                        misses = 0
                        with self._sdk_lock:
                            self._close_handle_quiet()
                        self._set_state(
                            "down",
                            "device not answering GigE discovery — "
                            "PoE link lost?")
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
            target=_loop, daemon=True,
            name=f"hik-watch-{self.serial_number}")
        self._watch_thread.start()

    # ── Recovery (Device protocol) ───────────────────────────────────

    def recover(self):
        with self._recover_lock:
            self._set_state("recovering", "reopening GigE session")
            # Close FIRST — the camera's GigE control channel is
            # exclusive; a half-dead session must be torn down before a
            # reopen can succeed.
            with self._sdk_lock:
                self._close_handle_quiet()
            # Fast judgment: not answering discovery -> fail NOW
            # (D405-parity fast-fail); a camera that IS answering just
            # needs the control channel re-established.
            deadline = time.time() + 6.0
            while (self.serial_number or self.ip) and time.time() < deadline:
                if self._present():
                    break
                time.sleep(0.5)
            else:
                if self.serial_number or self.ip:
                    self._set_state(
                        "down",
                        "device not answering discovery/ping — check PoE "
                        "link/power and retry")
                    return False
            ok = self.connect(**dict(self._connect_kwargs))
            return bool(ok)

    # ── Capture ──────────────────────────────────────────────────────

    def _grab_bgr(self, timeout_ms=5000):
        """Single grab -> BGR8 ndarray (h, w, 3). MV_CC_GetImageForBGR
        converts whatever PixelFormat the sensor runs (Bayer, YUV, mono)
        to BGR in the SDK. Raises on failure and moves state to down —
        frame fetch is the single source of truth for camera health,
        same as the D405."""
        with self._sdk_lock:
            if self._cam is None:
                self._set_state("down", "not connected")
                raise RuntimeError("Hikrobot camera not connected")
            try:
                n = self.width * self.height * 3
                if self._buf is None or len(self._buf) < n:
                    self._buf = (ctypes.c_ubyte * n)()
                info = _mv.MV_FRAME_OUT_INFO_EX()
                ctypes.memset(ctypes.byref(info), 0, ctypes.sizeof(info))
                ret = self._cam.MV_CC_GetImageForBGR(
                    self._buf, n, info, int(timeout_ms))
                if ret != 0:
                    raise RuntimeError(
                        f"MV_CC_GetImageForBGR failed ({_err(ret)})")
                w, h = int(info.nWidth), int(info.nHeight)
                arr = np.frombuffer(self._buf, dtype=np.uint8,
                                    count=w * h * 3).reshape(h, w, 3).copy()
                return arr
            except Exception as ex:
                # Release the session immediately: the GigE control
                # channel is exclusive and a wedged session blocks the
                # reopen that recovery needs.
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

    # ── Exposure / gain (real knobs on this sensor, µs like the D405) ─

    def get_exposure(self):
        """Current exposure time in µs."""
        if self._cam is None:
            raise RuntimeError("Hikrobot camera not connected")
        with self._sdk_lock:
            return round(self._get_float("ExposureTime"), 1)

    def set_exposure(self, exposure, _locked=True):
        """Pin manual exposure, in µs (same unit as the D405)."""
        if self._cam is None:
            raise RuntimeError("Hikrobot camera not connected")
        lock = self._sdk_lock if _locked else _NullCtx()
        with lock:
            self._cam.MV_CC_SetEnumValue("ExposureAuto", 0)   # Off
            ret = self._cam.MV_CC_SetFloatValue("ExposureTime", float(exposure))
            if ret != 0:
                raise RuntimeError(
                    f"ExposureTime set failed ({_err(ret)})")
            self.exposure_auto = False
            return round(self._get_float("ExposureTime"), 1)

    def auto_exposure(self, enable=True, _locked=True):
        if self._cam is None:
            raise RuntimeError("Hikrobot camera not connected")
        lock = self._sdk_lock if _locked else _NullCtx()
        with lock:
            self._cam.MV_CC_SetEnumValue("ExposureAuto", 2 if enable else 0)
            self.exposure_auto = bool(enable)
            return round(self._get_float("ExposureTime"), 1)

    # ── White balance ────────────────────────────────────────────────

    def white_balance(self, cfg, _locked=True):
        """{"auto": True} continuous, {"once": True} converge-then-hold,
        {"hold": True} freeze the current ratios."""
        if self._cam is None:
            raise RuntimeError("Hikrobot camera not connected")
        cfg = dict(cfg or {})
        if cfg.get("auto"):
            val, store = 2, {"auto": True}       # Continuous
        elif cfg.get("once"):
            val, store = 1, {"once": True}       # Once (holds after converge)
        elif cfg.get("hold"):
            val, store = 0, {"hold": True}       # Off (freeze current)
        else:
            raise ValueError(
                'wb needs {"auto": True}, {"once": True} or {"hold": True}')
        lock = self._sdk_lock if _locked else _NullCtx()
        with lock:
            ret = self._cam.MV_CC_SetEnumValue("BalanceWhiteAuto", val)
            if ret != 0:
                raise RuntimeError(
                    f"white balance set failed ({_err(ret)})")
        self.wb_cfg = store
        return dict(self.wb_cfg)


class _NullCtx(object):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
