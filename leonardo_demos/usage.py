"""Measure which devices a run actually kept busy.

Used by tools/run_job.py (every cluster run records it in meta.json as
``device_usage``) and tools/profile_demo.py (local profiling). GPU utilisation
comes from nvidia-smi; CPU use is process CPU time divided by wall time, i.e.
the average number of busy cores.
"""
from __future__ import annotations

import os
import shutil
import statistics
import subprocess
import threading
import time


class GpuSampler:
    """Poll nvidia-smi in a background thread (no-op without a GPU)."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.samples: list[tuple[float, float, float]] = []
        self._stop = threading.Event()
        self._thread = None
        self.smi = shutil.which("nvidia-smi")
        # Inside a Slurm job only the allocated GPUs are visible, so the max
        # over visible devices is the job's own GPU.
        self.visible = os.environ.get("CUDA_VISIBLE_DEVICES")

    def start(self):
        if self.smi:
            self._thread = threading.Thread(target=self._run, name="gpu-sampler", daemon=True)
            self._thread.start()
        return self

    def _run(self):
        cmd = [self.smi, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"]
        while not self._stop.is_set():
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
                rows = [[float(x) for x in line.split(",")] for line in out.strip().splitlines() if line.strip()]
                if rows:
                    self.samples.append((time.time(), max(r[0] for r in rows), max(r[1] for r in rows)))
            except (OSError, subprocess.TimeoutExpired, ValueError):
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=15)
        return self

    def summary(self) -> dict:
        util = [s[1] for s in self.samples]
        if not util:
            return {"gpu_util_mean": None, "gpu_busy_fraction": None, "gpu_mem_peak_mib": None}
        return {"gpu_util_mean": round(statistics.mean(util), 1),
                "gpu_busy_fraction": round(sum(u >= 20 for u in util) / len(util), 3),
                "gpu_mem_peak_mib": max(s[2] for s in self.samples),
                "gpu_samples": len(util)}


def cpu_seconds_self_and_children() -> float:
    """CPU seconds used by this process plus its finished children."""
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


def allocation() -> dict:
    """What the scheduler gave this job (empty outside Slurm)."""
    env = os.environ
    gpus = env.get("SLURM_GPUS_ON_NODE") or env.get("SLURM_GPUS") or ""
    visible = env.get("CUDA_VISIBLE_DEVICES")
    return {"job_id": env.get("SLURM_JOB_ID"), "node": env.get("SLURMD_NODENAME"),
            "cpus_allocated": int(env["SLURM_CPUS_PER_TASK"]) if env.get("SLURM_CPUS_PER_TASK", "").isdigit() else None,
            "gpus_allocated": (int(gpus) if gpus.isdigit() else
                               len([v for v in visible.split(",") if v.strip()]) if visible else 0)}
