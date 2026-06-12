import os
import sys
import time
import signal
import logging
from datetime import datetime, timezone

from camera import Camera


# ── config ───────────────────────────────────────────────────────────
STREAM   = {"width": 1280, "height": 720, "fps": 30}
CHANNELS = ("color", "depth")
LOG_PATH = os.environ.get("STRESS_LOG", "stress.log")
REPORT_EVERY_SEC = 5.0          # how often to write a status line
FRAME_TIMEOUT_SEC = 2           # per-frame wait_for_frames timeout
# ─────────────────────────────────────────────────────────────────────


def setup_logging(path):
    """Log to stdout AND a file. Line-buffered so `tail -f` shows progress
    in real time even when stdout is redirected."""
    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # File handler — line-buffered.
    fh = logging.FileHandler(path, mode="a")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    # stdout handler.
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)


def main():
    setup_logging(LOG_PATH)
    log = logging.getLogger("stress")

    # Graceful Ctrl-C / SIGTERM.
    stop = {"flag": False}
    def _stop(*_):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    log.info("=" * 60)
    log.info(
        "stress test start | stream=%dx%d@%d | channels=%s | pid=%d | log=%s",
        STREAM["width"], STREAM["height"], STREAM["fps"],
        CHANNELS, os.getpid(), os.path.abspath(LOG_PATH),
    )
    log.info("=" * 60)

    cam = Camera()

    # Connect. Let Camera's own retry+recover ladder handle the initial
    # bring-up. If it can't connect at all, we want a loud failure now,
    # not a silent infinite loop.
    ok = cam.connect(stream=STREAM, channels=CHANNELS)
    if not ok:
        log.error("initial connect failed — exiting")
        sys.exit(1)
    log.info("connected: serial=%s usb_port=%s",
             getattr(cam, "serial_number", "?"),
             getattr(cam, "usb_port", "?"))

    # State-change listener so we see USB drops / recovery in the log.
    def on_state(new_state, msg):
        log.warning("STATE → %s | %s", new_state, msg)
    cam.on_state_change(on_state)

    # Counters.
    t_start = time.time()
    t_last_report = t_start
    frames_total = 0
    frames_since_report = 0
    errors_total = 0
    errors_since_report = 0
    first_failure_at = None
    longest_run_sec = 0.0
    run_started_at = t_start

    try:
        while not stop["flag"]:
            try:
                # get_all() does color-aligned depth + images + intrinsics
                # in one call — same path your real app uses, so what we
                # measure here is what you'd actually see in production.
                cam.get_all()
                frames_total += 1
                frames_since_report += 1
            except Exception as ex:
                errors_total += 1
                errors_since_report += 1
                if first_failure_at is None:
                    first_failure_at = time.time()
                    log.error("FIRST FAILURE after %.1fs: %s",
                              first_failure_at - t_start, ex)
                # Track longest uninterrupted streaming window.
                streak = time.time() - run_started_at
                if streak > longest_run_sec:
                    longest_run_sec = streak
                run_started_at = time.time()
                # Brief pause so we don't spin the CPU when the pipeline
                # is wedged. The Camera class's hotplug callback will
                # update state on its own.
                time.sleep(0.5)

            # Periodic status line.
            now = time.time()
            if now - t_last_report >= REPORT_EVERY_SEC:
                window = now - t_last_report
                fps = frames_since_report / window if window > 0 else 0.0
                uptime = now - t_start
                log.info(
                    "uptime=%7.1fs  fps=%5.1f  frames=%d  errors=%d  "
                    "first_fail=%s  longest_streak=%.1fs  state=%s",
                    uptime, fps, frames_total, errors_total,
                    "none" if first_failure_at is None
                    else "%.1fs" % (first_failure_at - t_start),
                    max(longest_run_sec, now - run_started_at),
                    cam.state,
                )
                t_last_report = now
                frames_since_report = 0
                errors_since_report = 0

    finally:
        log.info("-" * 60)
        log.info(
            "stress test end | uptime=%.1fs  frames=%d  errors=%d  "
            "first_fail=%s  longest_streak=%.1fs",
            time.time() - t_start, frames_total, errors_total,
            "none" if first_failure_at is None
            else "%.1fs" % (first_failure_at - t_start),
            max(longest_run_sec, time.time() - run_started_at),
        )
        try:
            cam.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
