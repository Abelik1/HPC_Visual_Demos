"""Run one dashboard job on a cluster node.

The dashboard validates a run exactly as it would for this computer, writes
the resulting keyword arguments to ``job.json`` in a fresh run directory and
submits a Slurm job that calls this script. The run is written into that same
directory, so fetching the directory home gives the ordinary saved-run layout.

    python tools/run_job.py /path/to/runs/<run id>/job.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run_demo import run  # noqa: E402


def main(job_path: str) -> None:
    job_file = Path(job_path).resolve()
    run_dir = job_file.parent
    job = json.loads(job_file.read_text(encoding="utf-8"))
    params = dict(job.get("params") or {})
    # A visitor's own picture travels beside job.json; the path the dashboard
    # recorded belongs to the other machine.
    if "_target_path" in params:
        params["_target_path"] = str(run_dir / "target.png")
    run(job["demo"], job.get("profile", "hpc"), job.get("frames", 70), params,
        job.get("backend", "auto"), run_dir, job.get("method", "default"),
        bool(job.get("timings", True)), job.get("numerical_substeps"),
        job.get("settings_override") or {}, job.get("precision", "fp32"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
