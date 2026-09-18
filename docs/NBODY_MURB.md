# MUrB N-body (NBody-EuroHPC) in the dashboard

`nbody_murb` gives the external [NBody-EuroHPC](https://github.com/albtad01/NBody-EuroHPC)
code (MUrB) a front end. The dashboard does not reimplement it: it runs the
real `murb` executable, reads the `.murbtraj` trajectory it records, and
renders it the way MUrB's own OpenGL viewer does.

## Setup (once per machine)

```bat
git clone https://github.com/albtad01/NBody-EuroHPC ..\NBody-EuroHPC
scripts\build_murb_windows.bat
```

The build uses MSYS2 MinGW `g++` + Ninja (MSVC cannot compile MUrB's
GCC-style sources) and links statically, so `murb.exe` runs without MSYS2 on
`PATH`. On Linux use the repository's own `cmake --preset generic` (or
`leonardo` for CUDA). The demo finds the executable in
`../NBody-EuroHPC/build-*/bin/`, or wherever `MURB_EXE` points; `MURB_REPO`
overrides the repository location. Restart the viewer after building.

GPU implementations (`gpu+tile+full`, `gpu+tile`) appear in the Solver list
only when the executable reports `cuda=1` (a CUDA build, e.g. on Linux/WSL or
Leonardo). The Windows build is CPU/OpenMP.

## Controls

| Dashboard control | MUrB option |
|---|---|
| Initial condition | `--scheme galaxy` / `random` |
| Timestep | `--dt` |
| Solver | `--im` (cpu+omp, cpu+optim, cpu+simd, cpu+naive; GPU when built) |
| Advanced: Bodies / Iterations / Warm-up | `-n` / `-i` / `--warmup` |
| Frames | picks `--record-every` so about that many frames are recorded |
| Camera zoom, output width/height | rendering only |

The run log (`murb.log`), the exact command line, the build info and MUrB's
own timing report (ms/iteration, interactions/s, GFLOP/s) are kept in the
run's `meta.json`; the readout overlay shows them. The raw trajectory is
deleted once the frames are rendered.

Views: **MUrB camera** is MUrB's fixed camera ((0,0,5), 45° FOV, units of
1e8 m) at zoom 1; **Side** and **Top** orbit the same camera 90° around the
bodies' starting centre. All three are rendered for every frame.

## Trajectories computed on Leonardo

```bat
.venv\Scripts\python.exe scripts\import_murbtraj.py path\to\run.murbtraj --frames 140 --profile desktop
```

This renders any finalised `.murbtraj` (e.g. from `sbatch scripts/run_gpu_record.sh`
in NBody-EuroHPC) into a saved run; no local executable is needed. The run's
backend shows the implementation that recorded it (e.g. `gpu+multinode`).
