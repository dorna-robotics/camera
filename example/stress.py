"""
On-demand soak test for D405 over a 5m USB 3 cable on a Pi5.

Mimics the real app's usage pattern: camera stays connected and streaming
24/7, but get_all() is called in short bursts separated by long idle gaps.
This is what your robotics workflow actually does — not a flat-out loop.

What it measures (the things that actually matter for the real app):
  - frame freshness: wall_now - device_timestamp at the moment of grab.
    A healthy camera + cable + driver gives you <100 ms (basically the
    capture interval). High freshness = stale data path or queue issue.
  - thermal + CPU + RSS: the Pi can degrade under heat soak even with a
    cool workload. Logged so you can see throttling before fps would.
  - first failure: any get_all exception. Should be zero indefinitely.

Usage on the Pi:

    # foreground:
    python3 example/stress.py

    # background, survives SSH disconnect:
    nohup python3 -u example/stress.py > /dev/null 2>&1 &
    disown
    tail -f stress.log

    # stop:
    pkill -f stress.py
"""
import os
import sys
import time
import signal
import logging

from camera import Camera


# ── workload pattern ─────────────────────────────────────────────────
# Matches "2-3 grabs every ~30 seconds" from the real app.
STREAM        = {"width": 1280, "height": 720, "fps": 30}
CHANNELS      = ("color", "depth")
BURST_SIZE    = 3            # get_all() calls per burst
BURST_GAP_SEC = 30.0         # idle seconds between bursts
INTRA_BURST_SLEEP = 0.05     # gap between calls inside a burst
# ── logging ──────────────────────────────────────────────────────────
LOG_PATH = os.environ.get("STRESS_LOG", "stress.log")
# ─────────────────────────────────────────────────────────────────────


def setup_logging(path):
    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(path, mode="a")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)


def read_pi_temp_c():
    """Pi5 CPU temperature in °C, or None if not on a Pi."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return None


def read_proc_stats():
    """(rss_mb, cpu_pct) for this process. cpu_pct is averaged since the
    process started, which is fine for trend-watching over hours."""
    rss_mb = None
    cpu_pct = None
    try:
        # RSS — Linux /proc/self/status
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_mb = int(line.split()[1]) / 1024.0
                    break
    except Exception:
        pass
    try:
        # CPU% — sum of user+system jiffies / wall jiffies since boot,
        # close enough for "is this process burning CPU" over hours.
        with open("/proc/self/stat") as f:
            parts = f.read().split()
        utime = int(parts[13])
        stime = int(parts[14])
        starttime = int(parts[21])
        hz = os.sysconf("SC_CLK_TCK")
        with open("/proc/uptime") as f:
            uptime = float(f.read().split()[0])
        elapsed = uptime - (starttime / hz)
        if elapsed > 0:
            cpu_pct = 100.0 * (utime + stime) / hz / elapsed
    except Exception:
        pass
    return rss_mb, cpu_pct


def main():
    setup_logging(LOG_PATH)
    log = logging.getLogger("stress")

    stop = {"flag": False}
    def _stop(*_):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    log.info("=" * 70)
    log.info(
        "stress start | stream=%dx%d@%d | channels=%s | "
        "burst=%d every %.0fs | pid=%d",
        STREAM["width"], STREAM["height"], STREAM["fps"], CHANNELS,
        BURST_SIZE, BURST_GAP_SEC, os.getpid(),
    )
    log.info("=" * 70)

    cam = Camera()
    if not cam.connect(stream=STREAM, channels=CHANNELS):
        log.error("initial connect failed — exiting")
        sys.exit(1)
    log.info("connected: serial=%s usb_port=%s",
             getattr(cam, "serial_number", "?"),
             getattr(cam, "usb_port", "?"))

    def on_state(new_state, msg):
        log.warning("STATE → %s | %s", new_state, msg)
    cam.on_state_change(on_state)

    # Counters.
    t_start = time.time()
    bursts = 0
    grabs_total = 0
    errors_total = 0
    first_failure_at = None
    # Freshness aggregates over all bursts:
    freshness_min_ms = None
    freshness_max_ms = None
    # Clock-offset baseline (device-clock vs wall-clock). Set after the
    # warm-up burst — librealsense's first frames carry a startup-lag
    # that pollutes the baseline if measured at burst 1.
    BASELINE_BURST = 2          # which burst seeds the offset
    baseline_offset_ms = None   # set after BASELINE_BURST
    # Frame interval at the configured fps, in ms. Used to compute
    # gap_ms (jitter above the ideal frame spacing).
    frame_interval_ms = 1000.0 / STREAM["fps"]

    def log_burst_status(fresh_list, grab_ms_list, gap_ms_list):
        nonlocal freshness_min_ms, freshness_max_ms, baseline_offset_ms

        def _avg(xs): return sum(xs) / len(xs) if xs else float("nan")
        def _max(xs): return max(xs) if xs else float("nan")

        fresh_avg = _avg(fresh_list)
        fresh_max = _max(fresh_list)
        grab_avg  = _avg(grab_ms_list)
        grab_max  = _max(grab_ms_list)
        gap_avg   = _avg(gap_ms_list)
        gap_max   = _max(gap_ms_list)

        # All-time freshness extremes.
        if fresh_list:
            bmin, bmax = min(fresh_list), max(fresh_list)
            freshness_min_ms = bmin if freshness_min_ms is None else min(freshness_min_ms, bmin)
            freshness_max_ms = bmax if freshness_max_ms is None else max(freshness_max_ms, bmax)

        # Seed the clock-offset baseline once, on the warm-up burst.
        # Median of that burst's freshness is the constant offset between
        # device clock and wall clock; subsequent bursts subtract it so
        # fresh_delta reads ~0ms when healthy and grows when stale.
        if baseline_offset_ms is None and bursts == BASELINE_BURST and fresh_list:
            sorted_f = sorted(fresh_list)
            baseline_offset_ms = sorted_f[len(sorted_f) // 2]
            log.info("clock-offset baseline set: %.1fms (subtracted from "
                     "fresh_delta in subsequent bursts)", baseline_offset_ms)

        if baseline_offset_ms is not None and fresh_list:
            fresh_delta_avg = fresh_avg - baseline_offset_ms
            fresh_delta_max = fresh_max - baseline_offset_ms
            fresh_delta_str = (f"d_avg={fresh_delta_avg:+6.1f}ms "
                               f"d_max={fresh_delta_max:+6.1f}ms")
        else:
            fresh_delta_str = "d_avg=    n/a d_max=    n/a"

        temp = read_pi_temp_c()
        rss_mb, cpu_pct = read_proc_stats()
        log.info(
            "burst=%-5d uptime=%7.1fs  grabs=%-6d errors=%d  "
            "fresh_avg=%6.1fms %s  "
            "grab_avg=%5.1fms grab_max=%5.1fms  "
            "gap_avg=%+5.1fms gap_max=%+5.1fms  "
            "cpu=%s%%  rss=%s MB  temp=%s°C  state=%s",
            bursts, time.time() - t_start, grabs_total, errors_total,
            fresh_avg, fresh_delta_str,
            grab_avg, grab_max,
            gap_avg, gap_max,
            "n/a" if cpu_pct is None else f"{cpu_pct:.1f}",
            "n/a" if rss_mb is None else f"{rss_mb:.1f}",
            "n/a" if temp is None else f"{temp:.1f}",
            cam.state,
        )

    try:
        while not stop["flag"]:
            bursts += 1
            burst_freshness_ms = []
            burst_grab_ms = []
            burst_gap_ms = []
            prev_grab_end = None
            for _ in range(BURST_SIZE):
                if stop["flag"]:
                    break
                try:
                    # Wall-clock the get_all() call so we know how long
                    # each grab blocks the caller — independent of any
                    # clock-offset confusion. Healthy at 720p on Pi5 is
                    # roughly the frame interval (~33ms) plus a small
                    # amount of alignment/copy overhead.
                    t0 = time.time()
                    result = cam.get_all()
                    t1 = time.time()
                    grabs_total += 1
                    grab_ms = (t1 - t0) * 1000.0
                    burst_grab_ms.append(grab_ms)
                    # Inter-grab gap above one ideal frame interval. Zero
                    # is perfect spacing; positive = pipeline lagging.
                    if prev_grab_end is not None:
                        gap = (t1 - prev_grab_end) * 1000.0 - frame_interval_ms
                        burst_gap_ms.append(gap)
                    prev_grab_end = t1
                    # Freshness: device-clock timestamp vs wall-clock now.
                    # Absolute value is noisy because of clock offset;
                    # fresh_delta in the log line subtracts the baseline.
                    device_ts_sec = result[-1]
                    fresh_ms = (time.time() - device_ts_sec) * 1000.0
                    burst_freshness_ms.append(fresh_ms)
                except Exception as ex:
                    errors_total += 1
                    if first_failure_at is None:
                        first_failure_at = time.time()
                        log.error("FIRST FAILURE at uptime=%.1fs: %s",
                                  first_failure_at - t_start, ex)
                    else:
                        log.warning("grab error: %s", ex)
                if INTRA_BURST_SLEEP > 0:
                    time.sleep(INTRA_BURST_SLEEP)

            log_burst_status(burst_freshness_ms, burst_grab_ms, burst_gap_ms)

            # Idle. Sleep in small chunks so Ctrl-C is responsive.
            t_idle_end = time.time() + BURST_GAP_SEC
            while not stop["flag"] and time.time() < t_idle_end:
                time.sleep(min(0.5, t_idle_end - time.time()))

    finally:
        log.info("-" * 70)
        log.info(
            "stress end | uptime=%.1fs  bursts=%d  grabs=%d  errors=%d  "
            "first_fail=%s",
            time.time() - t_start, bursts, grabs_total, errors_total,
            "none" if first_failure_at is None
            else "%.1fs" % (first_failure_at - t_start),
        )
        try:
            cam.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
