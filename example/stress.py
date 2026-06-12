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
    # Freshness aggregates over the current burst:
    freshness_min_ms = None
    freshness_max_ms = None

    def log_burst_status(burst_freshness_ms):
        nonlocal freshness_min_ms, freshness_max_ms
        avg_ms = (sum(burst_freshness_ms) / len(burst_freshness_ms)
                  if burst_freshness_ms else float("nan"))
        max_ms = max(burst_freshness_ms) if burst_freshness_ms else float("nan")
        # Track all-time freshness extremes.
        if burst_freshness_ms:
            bmin, bmax = min(burst_freshness_ms), max(burst_freshness_ms)
            freshness_min_ms = bmin if freshness_min_ms is None else min(freshness_min_ms, bmin)
            freshness_max_ms = bmax if freshness_max_ms is None else max(freshness_max_ms, bmax)
        temp = read_pi_temp_c()
        rss_mb, cpu_pct = read_proc_stats()
        log.info(
            "burst=%-5d uptime=%7.1fs  grabs=%-6d errors=%d  "
            "fresh_avg=%6.1fms fresh_max=%6.1fms  "
            "alltime_fresh_min=%s alltime_fresh_max=%s  "
            "cpu=%s%%  rss=%s MB  temp=%s°C  state=%s",
            bursts, time.time() - t_start, grabs_total, errors_total,
            avg_ms, max_ms,
            "n/a" if freshness_min_ms is None else f"{freshness_min_ms:.1f}ms",
            "n/a" if freshness_max_ms is None else f"{freshness_max_ms:.1f}ms",
            "n/a" if cpu_pct is None else f"{cpu_pct:.1f}",
            "n/a" if rss_mb is None else f"{rss_mb:.1f}",
            "n/a" if temp is None else f"{temp:.1f}",
            cam.state,
        )

    try:
        while not stop["flag"]:
            bursts += 1
            burst_freshness_ms = []
            for _ in range(BURST_SIZE):
                if stop["flag"]:
                    break
                try:
                    # get_all returns ..., frames, ts at index [-2], [-1].
                    # ts is frames.get_timestamp()/1000 = device clock in seconds.
                    result = cam.get_all()
                    grabs_total += 1
                    device_ts_sec = result[-1]
                    # Freshness: how stale is the frame at the moment we
                    # received it? Compares the device hardware timestamp
                    # against wall clock right now. Sub-100ms = healthy.
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

            log_burst_status(burst_freshness_ms)

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
