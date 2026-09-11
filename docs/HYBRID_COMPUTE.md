# Hybrid CPU + GPU execution

`--backend hybrid` is for a CUDA-capable machine where one GPU and the host CPU
should both have useful work. It is deliberately a pipeline, not an inaccurate
claim that one numerical array is being updated by NumPy and CuPy at once.

- The numerical solver stays on the GPU (CuPy, or CUDA PyTorch for the neural
  wall).
- Two CPU workers encode completed presentation frames and commit them to disk
  atomically while the GPU starts the next solver update.
- At most two frames wait in memory. Completion waits for the queue, so a run
  is never marked complete before every JPEG exists.

This is the appropriate arrangement for the existing demos: the GPU has
regular array/tensor work, while colour conversion, Pillow encoding, metadata
and filesystem I/O are CPU work. It produces real CPU/GPU overlap without
copying a simulation state between devices on every time step, which would
usually make the run slower.

## Local use

Choose **GPU + CPU pipeline** in the viewer, or run:

```bash
python run_demo.py reaction_diffusion --profile desktop --backend hybrid
```

Hybrid requires a functioning CUDA backend. It fails rather than quietly
falling back to CPU; choose `auto` or `cpu` if the machine has no CUDA device.

## Leonardo use

The Leonardo Booster has four A100 GPUs and a 32-core Intel CPU per node. The
provided job asks SLURM for one GPU and eight host CPU cores, which is a
well-contained slice of that node for one hybrid demo:

```bash
sbatch --account="$LEONARDO_ACCOUNT" \
  --export=ALL,DEMO=reaction_diffusion,FRAMES=90,PROFILE=hpc,LEONARDO_DEMO_BACKEND=hybrid \
  slurm/run_demo.sbatch
```

The CPU cores are on the same Booster node as the requested GPU. They are not
a DCGP allocation, and CPU-only work on DCGP should be submitted as a separate
job rather than assumed to share a GPU job's memory. For a later MPI/multi-GPU
version, use one rank per GPU and give each rank its matching CPU-core share;
CINECA's current container example uses four tasks and eight CPUs per task for
the four-GPU node.

For the collision demo, a CPU-only run is more than a fallback. The restricted
N-body update divides disjoint particle slices among the CPU cores granted by
SLURM, while the GPU path keeps the full state in CuPy through each leapfrog
step. Submit the two real resource shapes explicitly:

```bash
bash scripts/submit_leonardo.sh galaxy_collision 90 leonardo cpu
bash scripts/submit_leonardo.sh galaxy_collision 90 leonardo hybrid
```

Do not submit `--backend cpu` through the Booster job: that reserves an A100 it
will not use. Conversely, the DCGP template never probes or requests CUDA.

## What professional deep zoom does differently

The crystal viewer does **not** independently regenerate neighbouring image
tiles. A finite branch budget can make independently traversed tiles select
different branches, producing visible seams and detail that appears from
nowhere. Instead it uses one deterministic, addressable branch grammar: every
branch's path key fixes its position, width and jitter, and generation *d* is
an exact subset of generation *d + 1*.

The browser requests a coherent cached viewport at the current generation and
the next one. It displays the current pixel colours immediately and blends the
next generation with a cubic fractional-log2-zoom weight. Thus only the new
pixel-coverage residual fades in; existing crystal branches cannot move, vanish
or be swapped for a different set. Views are debounced for 100 ms and retain
the prior image while work is in flight. Revisited views are served from RAM or
disk cache.

There is also an experimental CUDA line-rasterizer for visible tiles. Set
`LEONARDO_DEEPZOOM_BACKEND=gpu` before starting the server to profile it. The
default is deliberately `auto` (CPU): on this desktop, the GPU path measured
about 0.58 s for a warmed 256 px tile and 4.22 s for 512 px, versus 0.08 s and
0.16 s with Pillow on CPU. The recursive branch construction is CPU work, and
many small GPU atomic line writes cost more than the CPU scanline rasterizer.
The serving process alone may use CUDA; its CPU geometry workers do not create
GPU contexts. This keeps navigation on the fastest measured implementation,
while retaining the GPU implementation for future denser/vectorized renderers.

For escape-time sets such as Mandelbrot, serious *extreme* deep zoom renderers
add arbitrary-precision reference orbits and perturbation/series methods. That
is a different numerical problem from this procedural crystal. Here each run
has a finite recorded branch range (`zoom_detail_base` through
`zoom_detail_max`): 7–14 locally, 8–16 on desktop and 9–18 on Leonardo. Those
values are configurable in `config/profiles.json`; after the final generation,
the viewer stops adding new geometry and continues to navigate the final
filtered coverage image.

Sources consulted: [CINECA Leonardo hardware and partitions](https://docs.hpc.cineca.it/hpc/leonardo.html), [CINECA's GPU container resource example](https://docs.hpc.cineca.it/services/singularity.html), and [Ultra Fractal's perturbation overview](https://www.ultrafractal.com/help/formulas/perturbationcalculations.html).
