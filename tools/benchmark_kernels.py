"""Measure the CUDA solver kernels on this GPU, per precision.

    python tools/benchmark_kernels.py                 # all kernels, all precisions
    python tools/benchmark_kernels.py --precisions fp32 --n3d 200000
    python tools/benchmark_kernels.py --json benchmarks/kernels_$(hostname).json

Reports:
  galaxy3d  all-pairs force: ms per evaluation and 1e9 pair interactions/s
  galaxy2d  restricted N-body: ms per saved frame (tracers x substeps)
  fluid     D2Q9 lattice Boltzmann: million lattice updates per second (MLUPS)

The first call of each kernel also autotunes its launch shape (cached in
tmp/kernel_tuning.json); that time is excluded from the measurements.
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cupy as cp  # noqa: E402

from leonardo_demos.base import RunContext  # noqa: E402
from leonardo_demos.demos.fluid import FluidDemo  # noqa: E402
from leonardo_demos.demos.galaxy_collision import GalaxyCollisionDemo  # noqa: E402
from leonardo_demos.demos.galaxy_collision_3d import GalaxyCollision3DDemo  # noqa: E402


def timed(fn, repeats, budget=20.0):
    """Mean seconds per call after one warm-up call (which also tunes)."""
    fn()
    cp.cuda.Device().synchronize()
    started = time.perf_counter()
    done = 0
    while done < repeats and (done == 0 or time.perf_counter() - started < budget):
        fn()
        done += 1
    cp.cuda.Device().synchronize()
    return (time.perf_counter() - started) / done


def context(root, demo, precision, method="default"):
    return RunContext(Path(tempfile.mkdtemp(dir=root)), demo, "hpc", 2, {}, "gpu",
                      method=method, precision=precision)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--precisions", default="fp32,mixed,fp64")
    ap.add_argument("--n3d", default="6000,50000,200000", help="comma-separated body counts")
    ap.add_argument("--tracers", type=int, default=250000)
    ap.add_argument("--lattice", default="960x540")
    ap.add_argument("--json", help="also write results to this file")
    a = ap.parse_args()
    precisions = [p.strip() for p in a.precisions.split(",") if p.strip()]
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    result = {"host": socket.gethostname(), "architecture": platform.machine(), "gpu": name,
              "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(), "results": []}
    print(f"{name} on {result['host']}")
    root = tempfile.mkdtemp()

    for precision in precisions:
        ctx = context(root, "galaxy_collision_3d", precision, "leapfrog")
        demo = GalaxyCollision3DDemo(ctx, {})
        for n in (int(x) for x in a.n3d.split(",") if x):
            pos, _, mass = demo.setup(n, .35, 1.0, 35.0, 1.5e12, 1.5e12)[:3]
            seconds = timed(lambda: demo.acceleration(pos, mass, 4.0), 20)
            row = {"kernel": "galaxy3d_all_pairs", "precision": precision, "bodies": n,
                   "ms": seconds * 1e3, "gpairs_per_s": n * n / seconds / 1e9,
                   "launch": ctx._kernels["galaxy3d_all_pairs"]["config"]}
            result["results"].append(row)
            print(f"galaxy3d  {precision:5s} N={n:>7,}  {row['ms']:9.2f} ms  "
                  f"{row['gpairs_per_s']:8.1f} Gpair/s  {row['launch']}")
        ctx.finish()

    for precision in (p for p in precisions if p in GalaxyCollisionDemo.precisions):
        ctx = context(root, "galaxy_collision", precision, "leapfrog")
        demo = GalaxyCollisionDemo(ctx, {})
        state = list(demo.setup(a.tracers, 1, .55, .75, 18)[:8])

        def frame():
            state[:6] = demo.step(*state, .0012, 10)
        seconds = timed(frame, 20)
        row = {"kernel": "galaxy2d_tracers", "precision": precision, "tracers": a.tracers,
               "substeps": 10, "ms_per_frame": seconds * 1e3}
        result["results"].append(row)
        print(f"galaxy2d  {precision:5s} {a.tracers:,} tracers x 10 substeps  {row['ms_per_frame']:8.2f} ms/frame")
        ctx.finish()

    nx, ny = (int(v) for v in a.lattice.lower().split("x"))
    for precision in (p for p in precisions if p in FluidDemo.precisions):
        ctx = context(root, "fluid", precision)
        demo = FluidDemo(ctx, {"nx": nx, "ny": ny})
        f, c, w, mask = demo.init(nx, ny, .06, 0)
        box = [f]

        def lattice():
            box[0] = demo.step(box[0], c, w, mask, .06, 100)[0]
        seconds = timed(lattice, 10)
        row = {"kernel": "fluid_lbm_d2q9", "precision": precision, "lattice": f"{nx}x{ny}",
               "mlups": nx * ny * 100 / seconds / 1e6, "launch": ctx._kernels["fluid_lbm_d2q9"]["config"]}
        result["results"].append(row)
        print(f"fluid     {precision:5s} {nx}x{ny}  {row['mlups']:8.0f} MLUPS  {row['launch']}")
        ctx.finish()

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
