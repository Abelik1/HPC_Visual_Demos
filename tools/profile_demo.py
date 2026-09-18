"""Run one demo and measure which device actually did the work.

Answers the question "are we paying for a GPU we do not use?". It runs
run_demo.py as a child process with --timings and, while it runs, samples:

* GPU utilisation and memory (nvidia-smi, every 0.5 s);
* CPU time of the child and all its workers (psutil, or /proc on Linux),
  which gives the average number of busy cores.

It then combines those with the run's own phase timings (initialisation,
simulation, render) into one JSON record, appended to --out:

    python tools/profile_demo.py fluid --profile desktop --frames 60 \\
        --backend hybrid --out benchmarks/devices.jsonl -- --setting nx=1920

Everything after "--" is passed to run_demo.py unchanged. Works the same on a
Windows desktop and inside a Slurm job on Discoverer or Leonardo.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _gpu_sampler(stop: threading.Event, samples: list, index: str | None):
    smi = shutil.which("nvidia-smi")
    if not smi:
        return
    cmd = [smi, "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"]
    if index is not None:
        cmd += ["-i", index]
    while not stop.is_set():
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
            rows = [line.split(",") for line in out.strip().splitlines() if line.strip()]
            if rows:
                samples.append((time.time(), max(float(r[1]) for r in rows), max(float(r[2]) for r in rows)))
        except (OSError, subprocess.TimeoutExpired, ValueError):
            pass
        stop.wait(0.5)


def _cpu_seconds_tree(proc) -> float:
    """CPU seconds used so far by a process and its descendants (psutil)."""
    total = 0.0
    for p in [proc] + proc.children(recursive=True):
        try:
            t = p.cpu_times()
            total += t.user + t.system
        except Exception:
            pass
    return total


def main() -> int:
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        cut = argv.index("--")
        argv, extra = argv[:cut], argv[cut + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demo")
    ap.add_argument("--profile", default="desktop")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--method", default="default")
    ap.add_argument("--label", default="")
    ap.add_argument("--run-dir")
    ap.add_argument("--keep", action="store_true", help="keep the run directory")
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "devices.jsonl"))
    a = ap.parse_args(argv)

    run_dir = Path(a.run_dir) if a.run_dir else Path(tempfile.mkdtemp(prefix=f"profile_{a.demo}_"))
    cmd = [sys.executable, str(ROOT / "run_demo.py"), a.demo, "--profile", a.profile, "--frames", str(a.frames),
           "--backend", a.backend, "--method", a.method, "--timings", "--run-dir", str(run_dir)] + extra
    gpu_samples: list = []
    stop = threading.Event()
    sampler = threading.Thread(target=_gpu_sampler, args=(stop, gpu_samples,
                                                           os.environ.get("CUDA_VISIBLE_DEVICES") or None), daemon=True)
    sampler.start()
    started = time.time()
    try:
        import psutil
    except ImportError:
        psutil = None
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    cpu_seconds, peak_rss = 0.0, 0
    ps = psutil.Process(proc.pid) if psutil else None
    while proc.poll() is None:
        if ps:
            try:
                cpu_seconds = max(cpu_seconds, _cpu_seconds_tree(ps))
                peak_rss = max(peak_rss, sum(p.memory_info().rss for p in [ps] + ps.children(recursive=True)))
            except Exception:
                pass
        time.sleep(0.5)
    output = proc.stdout.read()
    wall = time.time() - started
    stop.set()
    sampler.join(timeout=5)
    if not ps and hasattr(os, "wait4"):
        pass
    if not ps:
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            cpu_seconds = usage.ru_utime + usage.ru_stime
            peak_rss = usage.ru_maxrss * 1024
        except Exception:
            pass

    meta = {}
    try:
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    timings = meta.get("timings") or {}
    phase = {k: round((timings.get(k) or {}).get("seconds", 0.0), 2) for k in timings}
    util = [s[1] for s in gpu_samples]
    record = {
        "demo": a.demo, "label": a.label, "profile": a.profile, "method": meta.get("method", a.method),
        "backend_requested": a.backend, "backend": meta.get("backend"), "frames": a.frames,
        "extra_args": extra, "host": socket.gethostname(), "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": meta.get("status") or f"exit {proc.returncode}", "error": meta.get("error"),
        "wall_seconds": round(wall, 2), "run_elapsed": round(meta.get("elapsed") or 0, 2),
        "phases": phase,
        "settings": meta.get("settings"),
        "summary": meta.get("summary"),
        "gpu": meta.get("resources", {}).get("gpu_name") or (meta.get("kernels") and "cuda") or None,
        "gpu_util_mean": round(statistics.mean(util), 1) if util else None,
        "gpu_busy_fraction": round(sum(u >= 20 for u in util) / len(util), 3) if util else None,
        "gpu_mem_peak_mib": max((s[2] for s in gpu_samples), default=None),
        "cpu_cores_busy_avg": round(cpu_seconds / wall, 2) if wall else None,
        "cpu_seconds": round(cpu_seconds, 1), "host_cpus": os.cpu_count(),
        "peak_rss_mib": round(peak_rss / 2**20) if peak_rss else None,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    print(json.dumps({k: record[k] for k in ("demo", "label", "status", "wall_seconds", "phases",
                                             "gpu_util_mean", "gpu_busy_fraction", "cpu_cores_busy_avg")}))
    if record["status"] != "complete":
        print(output[-2000:])
    if not a.keep and not a.run_dir:
        shutil.rmtree(run_dir, ignore_errors=True)
    return 0 if record["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
