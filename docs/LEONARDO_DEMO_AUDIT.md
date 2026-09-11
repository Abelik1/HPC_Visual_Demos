# Leonardo demo architecture and benchmark audit

This audit distinguishes “runs on a node” from “uses the node honestly.” A GPU
label is only offered where numerical state really lives in CuPy or PyTorch;
procedural NumPy/Pillow demos are routed to DCGP instead of reserving an A100.

| Demo | CPU | GPU / hybrid | Numerical method | Leonardo recommendation |
|---|---:|---:|---|---|
| Black-hole lensing | yes | CuPy | image-space thin-lens map | Booster for production resolution |
| PBH collapse | yes | no | reduced radial threshold model | DCGP; external validated solver adapter remains future work |
| Fluid | yes | CuPy | D2Q9 lattice Boltzmann | Booster for production grid; CPU is faster for tiny grids |
| Cosmic web | yes | CuPy | particle-mesh gravity + FFT Poisson solve | Booster for production grid |
| Galaxy collision | threaded NumPy | CuPy | restricted N-body, leapfrog (Euler comparison) | Booster at high tracer count; DCGP is valid |
| Reaction diffusion | yes | CuPy | Gray-Scott finite differences | Booster only when the grid/step budget amortises launches |
| Crystal | process-parallel | no | recursive anisotropic branch generator | DCGP; ensemble generation is GIL-bound and process-parallel |
| Neural wall | PyTorch | PyTorch CUDA | batched coordinate MLP search | Booster; explicit GPU failure never becomes a surrogate |
| Fusion plasma | yes | CuPy | complex Ginzburg-Landau field | Booster at production grid |
| Weather ensemble | yes | CuPy | spectral barotropic vorticity + moisture | Booster at production grid |
| Molecular dynamics | yes | CuPy | all-pairs coarse-grained polymer dynamics | Booster when particle count is large |

The implementation is deliberately one process and one GPU per simulation.
The current algorithms do not perform MPI halo exchange or distributed FFTs,
so requesting four A100s for one run would waste three. Parameter ensembles are
the clean future multi-GPU unit: one independent member per rank/GPU followed by
result collation.

## Measured workstation result

`benchmarks/latest.json` contains a 36-entry post-optimization matrix measured
on an RTX 3060 Ti with the fixed four-frame `benchmark` profile. It is a
regression/scaling diagnostic, not an A100 performance claim. The static report
is `benchmarks/index.html` and is also served by the application at
`/benchmarks/`.

The measurement disproved the suspected JPEG bottleneck. Compression took only
about 0.02–0.03 seconds for each four-frame run, and filesystem writes were
below a centisecond. Crystal finalization was slow because nine independent,
recursive geometries were generated serially. Moving those independent habits
to CPU processes reduced the measured crystal wall time from about 13.2 s to
about 5.8 s. Its timings now expose `ensemble` and `deep_zoom` separately.

Most tiny CuPy cases are slower than NumPy because process import, CUDA context
creation, and thousands of small stencil launches dominate. This is why the
report shows measured workload dimensions and why the `hpc` profile must
be benchmarked on an allocated A100 before event claims are made. Black-hole
lensing already shows the expected kernel benefit in the small profile: about
0.54 s CPU versus 0.13–0.14 s CUDA for its measured lens stage.

## Reproducing locally

```bash
python tools/benchmark_demos.py --profile benchmark --frames 4 \
  --backends cpu,gpu,hybrid --all-methods --keep-runs
python -m http.server 8000
# open http://localhost:8000/benchmarks/
```

Each case runs in a fresh subprocess, GPU stages synchronize at timing
boundaries, and `meta.json` records initialization, simulation, rendering,
visualization, JPEG, write, ensemble, and deep-zoom timings where applicable.

## Reproducing correctly on Leonardo

CPU and GPU results require different partitions and therefore separate jobs:

```bash
sbatch --account="$LEONARDO_ACCOUNT" --qos="$LEONARDO_CPU_QOS" slurm/benchmark_cpu.sbatch
sbatch --account="$LEONARDO_ACCOUNT" --qos="$LEONARDO_GPU_QOS" slurm/benchmark_gpu.sbatch
python tools/merge_benchmarks.py CPU_RESULTS.json GPU_RESULTS.json
```

Use the same `BENCH_PROFILE` and `BENCH_FRAMES` for both jobs. Start with the
fixed benchmark profile, then run the HPC profile only after confirming
the project allocation and increasing the two-hour template limit if needed.

The authoritative machine, scheduler, filesystem, and container assumptions
remain documented in `docs/LEONARDO.md`.
