# Precision, GPU kernels, and accuracy testing

This covers the three demos with hand-written CUDA kernels:
`galaxy_collision_3d`, `galaxy_collision`, and `fluid` (the wind tunnel).

## Choosing a precision

```bash
python run_demo.py galaxy_collision_3d --profile hpc --precision mixed
sbatch --export=ALL,DEMO=fluid,PRECISION=fp64 slurm/run_demo.sbatch
scripts/submit_leonardo.sh galaxy_collision_3d 90 hpc hybrid mixed
sbatch --export=ALL,PARTICLES=200000,FRAMES=500,PRECISION=mixed scripts/run_discoverer_galaxy3d.sbatch
```

The dashboard API accepts `"precision"` in the run request; `/api/specs` lists
each demo's supported precisions, and every run records its choice in
`meta.json`. Asking a demo for a precision it does not implement is an
error, never a silent FP32 fallback.

| Mode | State (positions, velocities, populations) | Hot arithmetic | Demos |
| --- | --- | --- | --- |
| `fp32` (default) | FP32 | FP32 | all three |
| `mixed` | FP64 | FP32 pair maths, FP64 force sums | `galaxy_collision_3d` |
| `fp64` | FP64 | FP64 | all three |

On the CPU (NumPy) path, `mixed` computes everything in FP64.

### Hardware trade-off

FP64 throughput is what decides the cost. Consumer GeForce cards run FP64 at
1/64 of their FP32 rate; the data-centre GPUs used for HPC run it at about half
rate (A100: 9.7 vs 19.5 TFLOPS, non-tensor).

| Kernel | RTX 3060 Ti fp32 | mixed | fp64 |
| --- | --- | --- | --- |
| 3-D all-pairs, 200k bodies | 89 ms / force evaluation | 86 ms | 5.9 s (66x) |
| 2-D tracers, 250k x 10 substeps | 0.79 ms / frame | – | 1.71 ms |
| D2Q9 LBM, 960x540 | 3,644 MLUPS | – | 922 MLUPS |

On an A100 or B200, expect `fp64` to cost roughly 2x for the 3-D N-body and
less for the memory-bound lattice solver. These HPC figures are hardware
ratios, not yet measured runs; `tools/benchmark_kernels.py` measures them on
the node itself.

### What each mode buys (3-D galaxy, 200k bodies, 70 frames, 18 Myr steps)

Measured with `tools/fidelity_run.py` on identical initial conditions:

| Force kernel | Energy drift | Angular-momentum drift | Centre-of-mass drift |
| --- | --- | --- | --- |
| previous FP32 kernel | 2.1% | 2.5% | 1.6 km/s |
| current FP32 kernel | 2.8% | 1.1e-4 | 2.1e-3 km/s |
| mixed | 2.8% | 2.5e-9 | 8.4e-8 km/s |

The previous kernel accumulated all 200,000 pair terms in one FP32 register;
that biased sum let the whole system drift ~12 kpc over 7.5 Gyr. The current
FP32 kernel sums each 64–512-body tile separately first, which removes most of
the error at no cost; `mixed` removes the rest, and on this card costs no more
than FP32 at 200k bodies. The energy error is set by the timestep (it falls to
0.3% with a 10x smaller step) and is not a precision effect; the two values
differ only because the post-merger dynamics are chaotic.

**Recommendation:** use `mixed` for the 3-D N-body everywhere; use `fp64` for
the lattice or 2-D solvers when the printed pressures or long-time tracer
orbits must be reproducible to more digits, or on an HPC GPU where it is cheap.

## Kernels

All three are CuPy `RawKernel`s, compiled once per precision and launch shape
(CuPy caches the binaries in `CUPY_CACHE_DIR`).

- **3-D all-pairs force** (`galaxy_collision_3d.py`, `NBODY_KERNEL`). Bodies
  are packed as coalesced `(x, y, z, m)` records. Each block stages `block`
  source bodies in shared memory; each thread computes the force on
  `per_thread` targets, so each shared-memory read feeds several interactions
  (register blocking, as in NBody-EuroHPC's EPT kernels). Zero-mass padding
  lets the inner loop unroll without a bounds test. The leapfrog reuses each
  frame's closing force as the next frame's opening force (one fewer force
  evaluation per frame, bit-identical results). In the hybrid backend the CPU
  draws frame *k* while the GPU integrates frame *k+1*.
- **2-D tracers** (`galaxy_collision.py`, `TRACER_KERNEL`). Tracers do not pull
  on the galaxy centres, so the centres' orbit is integrated first in FP64 and
  tabulated per force stage. One kernel then advances every tracer through all
  of a frame's substeps in registers: one launch per frame instead of about 60
  per substep. It implements all four integrators (leapfrog, symplectic Euler,
  MUrB, RK4).
- **D2Q9 lattice Boltzmann** (`fluid.py`, `LBM_KERNEL`). One fused kernel per
  lattice step: pull streaming with periodic wrap, full-way bounce-back,
  macroscopic moments, inlet condition, and BGK collision. Each population is
  read once and written once per step (ping-pong buffers). The old
  array-expression version made ~60 full-lattice passes, including 18 `roll`
  copies.

### Measured speed-up (RTX 3060 Ti, FP32)

| | Before | After |
| --- | --- | --- |
| 3-D force, 200k bodies | 97 ms | 89 ms |
| 3-D full run: 200k bodies, 70 frames, hybrid | 88.1 s | 49.3 s |
| 2-D frame, 250k tracers x 10 substeps | 23.2 ms | 0.79 ms (29x) |
| Wind tunnel, 960x540 | 41 MLUPS | 3,644 MLUPS (89x) |

The 3-D N-body's previous kernel was already shared-memory tiled, so its
kernel gain is modest; the end-to-end gain comes from the reused force and
from overlapping rendering with integration. At 200k bodies the CPU renderer
(about 0.4 s per 1280x720 frame on 8 cores) is now the longest stage, so a
faster GPU mainly helps once rendering is parallelised further.

### Autotuning

`leonardo_demos/tuning.py` times a few candidate launch shapes on the first
run of each kernel on a given GPU model, precision, and problem-size bucket,
then caches the winner. The choice and all candidate timings are written to
each run's `meta.json` under `kernels`.

| Variable | Effect |
| --- | --- |
| `LEONARDO_DEMO_AUTOTUNE=0` | use the built-in default launch shapes |
| `LEONARDO_DEMO_TUNING_CACHE=path` | cache location (default `tmp/kernel_tuning.json`) |

Tuning takes about 1–3 s per kernel. Delete the cache after a driver or
hardware change, or if it was tuned while another job shared the GPU.

## Tools

- `tools/benchmark_kernels.py` — per-precision kernel throughput on the
  current GPU; `--json` saves a record for comparing machines.
- `tools/fidelity_run.py` — a normal dashboard run of the 3-D galaxy plus
  per-frame energy, momentum, and angular momentum from the full state, and
  full-position checkpoints (`<run>/fidelity/`).
- `tools/fidelity_compare.py REFERENCE OTHER...` — conservation table and
  trajectory differences against a reference run at shared times.

```bash
python tools/fidelity_run.py --label normal --frames 70 --substeps 6 --precision mixed
python tools/fidelity_run.py --label highfi --frames 277 --substeps 15 --max-step 0.0025 --precision mixed
python tools/fidelity_compare.py runs/galaxy_collision_3d_fidelity_highfi runs/galaxy_collision_3d_fidelity_normal
```

## Verification

`tests/test_precision_kernels.py` checks each CUDA kernel against the NumPy
reference in FP64 (agreement to about 1e-14), every precision against a direct
FP64 force sum, several launch shapes including ragged final blocks, the
hybrid render overlap against the sequential run, and autotuner caching. The
GPU tests skip automatically on machines without CUDA.
