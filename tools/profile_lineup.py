"""Profile every demo-day simulation on this machine: which device does the work?

For each demo, run tools/profile_demo.py:
* at the ``local`` preset on the CPU and on the GPU (the speed-up a GPU buys);
* at the ``desktop`` preset on the GPU path (how busy the GPU really is).

    python tools/profile_lineup.py --out benchmarks/devices_baseline.jsonl
    python tools/profile_lineup.py --only fluid,fusion_plasma --presets desktop

Results are one JSON line per run; tools/device_report.py turns them into a table.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# (demo, method, extra run_demo.py arguments). One simulation per population
# demo is not the experiment, so the AI games keep their ensemble.
CASES = [
    ("galaxy_collision_3d", "leapfrog", []),
    ("fluid", "default", []),
    ("neuro_racers", "default", []),
    ("black_hole", "schwarzschild", []),
    ("molecular_dynamics", "fold", []),
    ("molecular_dynamics", "shuttle", []),
    ("nbody_murb", "cpu+omp", []),
    ("fusion_plasma", "passive", ["--param", "_parallel_count=1"]),
    ("fusion_plasma", "guardian", []),
    ("cosmic_web", "default", ["--param", "_parallel_count=1"]),
    ("bat_vs_moth", "default", []),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "devices_baseline.jsonl"))
    ap.add_argument("--only", default="")
    ap.add_argument("--presets", default="local,desktop")
    ap.add_argument("--frames", type=int, default=40)
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    presets = a.presets.split(",")
    failures = 0
    for demo, method, extra in CASES:
        if only and demo not in only:
            continue
        runs = []
        if "local" in presets:
            runs += [("local", "cpu"), ("local", "gpu" if demo != "nbody_murb" else "cpu")]
        if "desktop" in presets:
            runs += [("desktop", "hybrid" if demo not in ("nbody_murb",) else "cpu")]
        seen = set()
        for profile, backend in runs:
            if (profile, backend) in seen:
                continue
            seen.add((profile, backend))
            cmd = [sys.executable, str(ROOT / "tools" / "profile_demo.py"), demo, "--profile", profile,
                   "--frames", str(a.frames), "--backend", backend, "--method", method,
                   "--label", f"{profile}/{backend}", "--out", a.out]
            if extra:
                cmd += ["--"] + extra
            print(" ".join(cmd[2:]), flush=True)
            failures += subprocess.run(cmd, cwd=str(ROOT)).returncode != 0
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
