"""Run one dashboard job on a cluster node.

The dashboard validates a run exactly as it would for this computer, writes
the resulting keyword arguments to ``job.json`` in a fresh run directory and
submits a Slurm job that calls this script. The run is written into that same
directory, so fetching the directory home gives the ordinary saved-run layout.

While it runs, the job measures which devices it actually kept busy (GPU
utilisation from nvidia-smi, average busy CPU cores) and records that, with
what Slurm allocated, as ``device_usage`` in meta.json.

    python tools/run_job.py /path/to/runs/<run id>/job.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run_demo import run  # noqa: E402
from leonardo_demos.usage import GpuSampler, allocation, cpu_seconds_self_and_children  # noqa: E402


def record_usage(run_dir: Path, wall: float, cpu: float, gpu: GpuSampler) -> None:
    meta_path = run_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    alloc = allocation()
    usage = {**alloc, **gpu.summary(), "wall_seconds": round(wall, 1), "cpu_seconds": round(cpu, 1),
             "cpu_cores_busy_avg": round(cpu / wall, 2) if wall else None}
    if alloc.get("cpus_allocated") and usage["cpu_cores_busy_avg"] is not None:
        usage["cpu_allocation_used"] = round(usage["cpu_cores_busy_avg"] / alloc["cpus_allocated"], 3)
    meta["device_usage"] = usage
    tmp = meta_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(meta_path)


def main(job_path: str) -> None:
    job_file = Path(job_path).resolve()
    run_dir = job_file.parent
    job = json.loads(job_file.read_text(encoding="utf-8"))
    params = dict(job.get("params") or {})
    # A visitor's own picture travels beside job.json; the path the dashboard
    # recorded belongs to the other machine.
    if "_target_path" in params:
        params["_target_path"] = str(run_dir / "target.png")
    gpu = GpuSampler().start()
    started, cpu0 = time.time(), cpu_seconds_self_and_children()
    try:
        run(job["demo"], job.get("profile", "hpc"), job.get("frames", 70), params,
            job.get("backend", "auto"), run_dir, job.get("method", "default"),
            bool(job.get("timings", True)), job.get("numerical_substeps"),
            job.get("settings_override") or {}, job.get("precision", "fp32"))
    finally:
        gpu.stop()
        record_usage(run_dir, time.time() - started, cpu_seconds_self_and_children() - cpu0, gpu)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
