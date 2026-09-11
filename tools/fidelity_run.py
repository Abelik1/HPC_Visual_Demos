"""Run galaxy_collision_3d with conservation diagnostics on every saved frame.

The run goes through the normal ``run_demo.run`` pipeline, so it produces an
ordinary dashboard run.  The solver's ``setup`` and ``step`` are wrapped (the
physics code itself is untouched) so that after every frame the full N-body
state is used to compute, on the GPU:

  * total energy (kinetic + softened potential; the force is the exact
    gradient of this Plummer potential, so the true dynamics conserve it),
  * total linear and angular momentum,
  * and, at checkpoints shared between runs of different temporal
    resolution, the complete particle positions.

Diagnostics are written to ``<run_dir>/fidelity/``.  Compare runs with
``tools/fidelity_compare.py``.  Requires a CUDA GPU (CuPy).

Example: a normal and a 10x-finer-timestep run of the same encounter

    python tools/fidelity_run.py --label normal --frames 70 --substeps 6
    python tools/fidelity_run.py --label highfi --frames 277 --substeps 15 --max-step 0.0025
    python tools/fidelity_compare.py runs/galaxy_collision_3d_fidelity_highfi \
        runs/galaxy_collision_3d_fidelity_normal

Frame counts must satisfy (frames - 1) % (align_frames - 1) == 0 so that every
checkpoint of the coarse run coincides exactly with one of the fine run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import cupy as cp  # noqa: E402

import run_demo  # noqa: E402
from leonardo_demos.demos.galaxy_collision import G, TIME_UNIT_GYR  # noqa: E402
from leonardo_demos.demos.galaxy_collision_3d import GalaxyCollision3DDemo  # noqa: E402

# Softened pair potential.  FP32 pair terms summed per 256-body tile, tiles
# accumulated in FP64: accurate to ~1e-7 relative, and fast even for 200k
# bodies on a consumer GPU.
POTENTIAL = cp.RawKernel(r'''
extern "C" __global__
void potential(const float* p, const float* m, double* out, const int n, const float soft2) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float sx[256], sy[256], sz[256], sm[256];
    float ix = 0, iy = 0, iz = 0; double acc = 0;
    if (i < n) { ix = p[3*i]; iy = p[3*i+1]; iz = p[3*i+2]; }
    for (int base = 0; base < n; base += blockDim.x) {
        const int j = base + threadIdx.x;
        if (j < n) { sx[threadIdx.x] = p[3*j]; sy[threadIdx.x] = p[3*j+1];
                     sz[threadIdx.x] = p[3*j+2]; sm[threadIdx.x] = m[j]; }
        else { sx[threadIdx.x] = sy[threadIdx.x] = sz[threadIdx.x] = sm[threadIdx.x] = 0; }
        __syncthreads();
        if (i < n) {
            const int width = (n - base) < blockDim.x ? (n - base) : blockDim.x;
            float s = 0;
            for (int k = 0; k < width; k++) {
                if (base + k == i) continue;
                const float dx = sx[k] - ix, dy = sy[k] - iy, dz = sz[k] - iz;
                s += sm[k] / sqrtf(dx*dx + dy*dy + dz*dz + soft2);
            }
            acc += (double)s;
        }
        __syncthreads();
    }
    if (i < n) out[i] = -(double)m[i] * acc;
}''', "potential")


def diagnostics(pos, vel, mass, softening):
    n = len(pos)
    out = cp.zeros(n, dtype=cp.float64)
    p32 = cp.ascontiguousarray(pos, dtype=cp.float32)
    m32 = cp.ascontiguousarray(mass, dtype=cp.float32)
    POTENTIAL(((n + 255) // 256,), (256,), (p32, m32, out, np.int32(n), np.float32(softening ** 2)))
    m = mass.astype(cp.float64)
    p = pos.astype(cp.float64)
    v = vel.astype(cp.float64)
    pe = 0.5 * G * float(out.sum())
    ke = 0.5 * float((m[:, None] * v * v).sum())
    momentum = (m[:, None] * v).sum(axis=0)
    angular = (m[:, None] * cp.cross(p, v)).sum(axis=0)
    return dict(ke=ke, pe=pe, e=ke + pe,
                p=[float(x) for x in cp.asnumpy(momentum)],
                l=[float(x) for x in cp.asnumpy(angular)],
                nonfinite=int((~cp.isfinite(pos)).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--label", required=True, help="suffix for the run directory")
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--substeps", type=int, required=True)
    ap.add_argument("--max-step", type=float, default=0.025, help="largest solver step (time units)")
    ap.add_argument("--method", default="leapfrog", choices=GalaxyCollision3DDemo.methods)
    ap.add_argument("--precision", default="fp32", choices=GalaxyCollision3DDemo.precisions)
    ap.add_argument("--particles", type=int, default=200000)
    ap.add_argument("--profile", default="hpc")
    ap.add_argument("--align-frames", type=int, default=70,
                    help="frame count of the coarsest run being compared (checkpoint grid)")
    ap.add_argument("--checkpoint-every", type=int, default=3,
                    help="save full positions every N coarse-run frames")
    ap.add_argument("--run-dir", help="default: runs/galaxy_collision_3d_fidelity_<label>")
    a = ap.parse_args()

    intervals = a.frames - 1
    base = a.align_frames - 1
    if intervals % base:
        raise SystemExit(f"(frames-1)={intervals} must be a multiple of (align_frames-1)={base}")
    ratio = intervals // base
    run_dir = Path(a.run_dir) if a.run_dir else ROOT / "runs" / f"galaxy_collision_3d_fidelity_{a.label}"
    out = run_dir / "fidelity"
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    log, state = [], {"frame": 0, "sim_s": 0.0, "diag_s": 0.0}
    original_step = GalaxyCollision3DDemo.step
    original_setup = GalaxyCollision3DDemo.setup

    def record(demo, pos, vel, mass, frame):
        started = time.perf_counter()
        d = diagnostics(pos, vel, mass, float(demo.ctx.params.get("softening", 4.0)))
        d["frame"] = frame
        d["t_gyr"] = frame / intervals * float(demo.settings.get("span_gyr", 7.5))
        log.append(d)
        if frame % (ratio * a.checkpoint_every) == 0:
            np.save(out / "ckpt" / f"pos_{frame // ratio:03d}.npy", cp.asnumpy(pos).astype(np.float32))
        state["diag_s"] += time.perf_counter() - started

    def setup(self, *args, **kw):
        result = original_setup(self, *args, **kw)
        pos, vel, mass, origin, component = result[:5]
        np.save(out / "origin.npy", np.asarray(origin))
        np.save(out / "component.npy", np.asarray(component))
        np.save(out / "mass.npy", cp.asnumpy(mass).astype(np.float64))
        record(self, pos, vel, mass, 0)
        return result

    def step(self, positions, velocities, masses, dt, steps, softening):
        cp.cuda.Device().synchronize()
        started = time.perf_counter()
        positions, velocities = original_step(self, positions, velocities, masses, dt, steps, softening)
        cp.cuda.Device().synchronize()
        state["sim_s"] += time.perf_counter() - started
        state["frame"] += 1
        state["dt_myr"] = dt * TIME_UNIT_GYR * 1000
        state["substeps"] = steps
        record(self, positions, velocities, masses, state["frame"])
        return positions, velocities

    GalaxyCollision3DDemo.setup = setup
    GalaxyCollision3DDemo.step = step
    wall = time.perf_counter()
    try:
        run_demo.run(
            "galaxy_collision_3d", a.profile, a.frames,
            {"impact": 0.35, "speed": 1.0, "disc_tilt": 35.0, "softening": 4.0},
            "hybrid", run_dir, a.method, True, None,
            {"particles": float(a.particles), "substeps": float(a.substeps), "max_step": a.max_step},
            a.precision)
    finally:
        GalaxyCollision3DDemo.setup = original_setup
        GalaxyCollision3DDemo.step = original_step
    wall = time.perf_counter() - wall

    summary = dict(label=a.label, method=a.method, precision=a.precision, frames=a.frames,
                   particles=a.particles, substeps_per_frame=state["substeps"],
                   dt_myr=state["dt_myr"], total_steps=state["substeps"] * intervals,
                   checkpoint_ratio=ratio, sim_seconds=state["sim_s"],
                   diag_seconds=state["diag_s"], wall_seconds=wall, run_dir=str(run_dir), log=log)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    e0 = log[0]["e"]
    print(f"{a.label}: {a.precision} dt={state['dt_myr']:.3f} Myr x {summary['total_steps']} steps, "
          f"sim {state['sim_s']:.1f}s, wall {wall:.1f}s, final dE/E={(log[-1]['e'] - e0) / abs(e0):+.3e}")
    print(run_dir)


if __name__ == "__main__":
    main()
