#!/usr/bin/env python3
"""
memory_profile.py - Profile any2md memory usage on Apple Silicon.

Runs an any2md command *in-process* and records, for every log line emitted
during the run, the process physical footprint (what Activity Monitor shows,
which includes Metal/GPU buffers on unified memory) plus MLX's own active and
cache memory counters. A background thread samples the footprint so peaks
between log lines are not missed.

Usage:
    python scripts/memory_profile.py -- yt long_meeting.m4a --diarize -o /tmp/out
    python scripts/memory_profile.py --repeat 3 -- yt clip.wav -o /tmp/out
    python scripts/memory_profile.py --json report.json -- pdf scan.pdf --force-ocr

Everything after ``--`` is passed to the any2md CLI unchanged. With
``--repeat N`` the command runs N times in the same process, which exposes
leaks that survive between runs (memory that should have been released but
was not).
"""

import argparse
import ctypes
import gc
import json
import logging
import os
import sys
import threading
import time
from typing import Dict, List, Optional

GB = 1024 ** 3


class _RUsageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups",
            "ri_pageins", "ri_wired_size", "ri_resident_size", "ri_phys_footprint",
            "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
            "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
            "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
            "ri_diskio_byteswritten", "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance",
            "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
            "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive",
            "ri_billed_system_time", "ri_serviced_system_time", "ri_logical_writes",
            "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles",
            "ri_billed_energy", "ri_serviced_energy", "ri_interval_max_phys_footprint",
            "ri_runnable_time",
        )
    ]


_libc = ctypes.CDLL("libc.dylib") if sys.platform == "darwin" else None


def footprint() -> Dict[str, int]:
    """Return current/lifetime-max physical footprint and RSS in bytes (macOS)."""
    if _libc is None:
        import resource
        return {"footprint": 0, "max_footprint": 0,
                "rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    info = _RUsageInfoV4()
    if _libc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) != 0:
        return {"footprint": 0, "max_footprint": 0, "rss": 0}
    return {
        "footprint": info.ri_phys_footprint,
        "max_footprint": info.ri_lifetime_max_phys_footprint,
        "rss": info.ri_resident_size,
    }


def mlx_stats() -> Dict[str, Optional[int]]:
    """Return MLX active/cache/peak memory in bytes, or Nones if MLX is absent."""
    try:
        import mlx.core as mx
    except Exception:
        return {"mlx_active": None, "mlx_cache": None, "mlx_peak": None}
    ns = mx if hasattr(mx, "get_active_memory") else mx.metal
    return {
        "mlx_active": ns.get_active_memory(),
        "mlx_cache": ns.get_cache_memory(),
        "mlx_peak": ns.get_peak_memory(),
    }


class _Recorder(logging.Handler):
    """Logging handler that snapshots memory on every log record."""

    def __init__(self, t0: float):
        super().__init__(level=logging.DEBUG)
        self.t0 = t0
        self.events: List[Dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        self.mark(msg[:120])

    def mark(self, label: str) -> None:
        snap = {"t": round(time.time() - self.t0, 2), "label": label}
        snap.update(footprint())
        snap.update(mlx_stats())
        self.events.append(snap)


class _Sampler(threading.Thread):
    """Background sampler tracking the footprint timeline between log lines."""

    def __init__(self, t0: float, interval: float = 0.25):
        super().__init__(daemon=True)
        self.t0 = t0
        self.interval = interval
        self.samples: List[List[float]] = []
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            fp = footprint()["footprint"]
            self.samples.append([round(time.time() - self.t0, 2), fp])
            self._halt.wait(self.interval)

    def halt(self) -> None:
        self._halt.set()


def _fmt(n: Optional[int]) -> str:
    return "   -  " if n is None else f"{n / GB:6.2f}"


def _run_cli(argv: List[str]) -> int:
    from any2md import cli
    sys.argv = ["any2md"] + argv
    try:
        cli.app()
    except SystemExit as exc:
        return int(exc.code or 0) if not isinstance(exc.code, str) else 1
    return 0


def main() -> int:
    if "--" not in sys.argv:
        print(__doc__, file=sys.stderr)
        return 2
    split = sys.argv.index("--")
    own_args, cli_args = sys.argv[1:split], sys.argv[split + 1:]

    parser = argparse.ArgumentParser(description="Profile any2md memory usage")
    parser.add_argument("--repeat", type=int, default=1, help="Run the command N times in-process")
    parser.add_argument("--json", dest="json_path", help="Write the full report as JSON here")
    parser.add_argument("--interval", type=float, default=0.25, help="Footprint sampling interval (s)")
    args = parser.parse_args(own_args)

    # Configure stderr logging before any2md is imported (its basicConfig calls
    # become no-ops), then attach the recorder to the same root logger.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    t0 = time.time()
    recorder = _Recorder(t0)
    logging.getLogger().addHandler(recorder)
    sampler = _Sampler(t0, args.interval)
    sampler.start()

    runs = []
    for i in range(args.repeat):
        recorder.mark(f"=== run {i + 1}/{args.repeat} start ===")
        rc = _run_cli(list(cli_args))
        gc.collect()
        recorder.mark(f"=== run {i + 1}/{args.repeat} end (rc={rc}) ===")
        snap = recorder.events[-1]
        runs.append({"run": i + 1, "rc": rc, "footprint_after": snap["footprint"],
                     "mlx_active_after": snap["mlx_active"], "mlx_cache_after": snap["mlx_cache"]})

    sampler.halt()
    sampler.join(timeout=2)
    peak = max((s[1] for s in sampler.samples), default=0)
    final = footprint()

    print("\n   t(s)  footprint  mlx_active  mlx_cache  event", file=sys.stderr)
    for ev in recorder.events:
        print(f"{ev['t']:7.1f}  {_fmt(ev['footprint'])} GB  {_fmt(ev['mlx_active'])} GB  "
              f"{_fmt(ev['mlx_cache'])} GB  {ev['label']}", file=sys.stderr)
    print(f"\nPeak footprint (sampled): {peak / GB:.2f} GB; "
          f"lifetime max: {final['max_footprint'] / GB:.2f} GB", file=sys.stderr)
    for r in runs:
        print(f"run {r['run']}: rc={r['rc']} footprint_after={_fmt(r['footprint_after'])} GB "
              f"mlx_active_after={_fmt(r['mlx_active_after'])} GB "
              f"mlx_cache_after={_fmt(r['mlx_cache_after'])} GB", file=sys.stderr)

    if args.json_path:
        with open(args.json_path, "w") as f:
            json.dump({"argv": cli_args, "repeat": args.repeat, "peak_footprint": peak,
                       "lifetime_max_footprint": final["max_footprint"], "runs": runs,
                       "events": recorder.events, "samples": sampler.samples}, f, indent=1)
    return max((r["rc"] for r in runs), default=0)


if __name__ == "__main__":
    sys.exit(main())
