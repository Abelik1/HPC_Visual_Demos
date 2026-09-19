# Demo-day run statistics

One table per demo, built from each run's own recorded timings (`tools/demo_stats.py`).
Physics = the solver; drawing = turning the results into frames. Where drawing runs in parallel with the physics, the two can add up to more than the total time.
“Devices” is what the job was allocated and how busy it actually kept it (runs from 19 Sep onwards).

## Discoverer demo day

**Galaxy collision - full 3D gravity** (Leapfrog (recommended)), gpu-11, 13 Sep 2026

_Run on Discoverer on 13 Sep (showcase job 7653, before right-sizing: it held a whole node)._

| | |
|---|---|
| Setup | 1,000,000 bodies, direct all-pairs gravity, 9.0 Gyr, 600 frames, mixed |
| Total time | **35 min 27 s** |
| Physics | 35 min 23 s (3.54 s/frame) |
| Drawing frames | 19 min 51 s (1.99 s/frame) |
| Compute | cupy + CPU frame workers |
| Result | physics full self-gravity · O(N²); dimensions 3 spatial; solver Leapfrog (recommended); massive particles 1,000,000 |

**Virtual wind tunnel** (Default solver), Discoverer, job 8682, 19 Sep 2026, gpu-11

| | |
|---|---|
| Setup | 7,680×4,320 lattice-Boltzmann grid, 250,000 steps, 3,000 tracers, 600 frames |
| Total time | **1 min 48 s** |
| Physics | 1 min 42 s (170 ms/frame) |
| Drawing frames | 59 s (99 ms/frame) |
| Devices | 1 GPU, busy 92% on average; 24 cores, 1.52 busy on average |
| Compute | cupy + CPU frame workers |
| Result | inlet speed 0.0400; Reynolds number 1,990; grid 7680 × 4320; obstacle preset twin cylinders |

**Neuro-Racers** (Default solver), Discoverer, job 8683, 19 Sep 2026, gpu-11

| | |
|---|---|
| Setup | 8,192 cars per search × 16 independent searches, 60 generations, 450 frames |
| Total time | **1 min 09 s** |
| Physics | 26 s (59 ms/frame) |
| Drawing frames | 21 s (47 ms/frame) |
| Devices | 1 GPU, busy 16% on average; 8 cores, 0.96 busy on average |
| Compute | cupy |
| Result | track id 0; track Oval; best lap s 6.3; best laps 8.81 |

**Black-hole lensing** (Exact ray tracing · real Gaia sky), Discoverer, job 8684, 19 Sep 2026, gpu-11

| | |
|---|---|
| Setup | 2,560×1,440 px, 3× supersampling, exact Schwarzschild rays through the Gaia sky, 600 frames |
| Total time | **3 min 33 s** |
| Physics | 38 s (63 ms/frame) |
| Drawing frames | 3 min 10 s (317 ms/frame) |
| Devices | 1 GPU, busy 17% on average; 16 cores, 4.2 busy on average |
| Compute | cupy + CPU frame workers |
| Result | black hole Gaia BH3 · 32.7 M☉; camera distance 12.0 r_s · 1,159 km; shadow 23.9° across; starlight blueshift ×1.044 |

**Molecular Machine** (Fold your own protein), AlexMainDesktop, 18 Sep 2026

_Run on the exhibition PC (RTX 3060 Ti), not yet on Discoverer._

| | |
|---|---|
| Setup | 240-bead chain, 1,500,000 Langevin steps, 120 frames |
| Total time | **1 min 20 s** |
| Physics | 1 min 11 s (595 ms/frame) |
| Drawing frames | 7 s (62 ms/frame) |
| Compute | cupy |
| Result | folded to Rg 3.162, 99% of oily beads buried, 0 salt bridges |

**Molecular Machine** (Molecular shuttle (a machine)), Discoverer, job 8685, 19 Sep 2026, gpu-11

| | |
|---|---|
| Setup | rotaxane ring on an axle, 400,000 Langevin steps, 300 frames |
| Total time | **49 s** |
| Physics | 39 s (131 ms/frame) |
| Drawing frames | 10 s (35 ms/frame) |
| Devices | 2 cores, 0.98 busy on average |
| Compute | numpy |
| Result | ring made 4 trips for 4 switch flips |

## Leonardo demo day

**MUrB N-body (NBody-EuroHPC)** (CPU · OpenMP (all cores)), Leonardo, job 58225921, 19 Sep 2026, lrdn4323

| | |
|---|---|
| Setup | 100,000 bodies, 1,440 iterations, MUrB cpu+omp, 1.04 TFLOP/s, 360 frames |
| Total time | **11 min 19 s** |
| Physics | 4 min 36 s (766 ms/frame) |
| Drawing frames | 6 min 43 s (1.12 s/frame) |
| Devices | 32 cores, 12.92 busy on average |
| Compute | MUrB cpu+omp |
| Result | code MUrB (NBody-EuroHPC); backend cpu+omp; bodies 100,000; iteration 1,440 |

**Star in a Bottle** (Mode 1 · Passive confinement), gpu-12, 13 Sep 2026

_Run on **Discoverer** on 13 Sep (showcase job 7654), not yet on Leonardo; before the drawing fix, so drawing dominates._

| | |
|---|---|
| Setup | 1,536² grid, 24,000 steps, 6,000 tracers, 5.5 T field, 32 MW heating, 600 frames |
| Total time | **16 min 11 s** |
| Physics | 31 s (52 ms/frame) |
| Drawing frames | 10 min 13 s (1.02 s/frame) |
| Compute | cupy |
| Result | mode passive confinement; magnetic field 5.5 T; heating power 32 MW; density 1.00 n₀ |

**Star in a Bottle** (Mode 2 · AI plasma guardian (3D)), gpu-12, 13 Sep 2026

_Run on **Discoverer** on 13 Sep (showcase job 7654), not yet on Leonardo._

| | |
|---|---|
| Setup | 16 simulations run together, 512² grid, 12 shots, 3,000 neural-network updates, 600 frames |
| Total time | **14 min 21 s** |
| Physics | 11 min 32 s (1.15 s/frame) |
| Drawing frames | 1 min 49 s (182 ms/frame) |
| Compute | cupy + torch·cuda |
| Result | mode AI plasma guardian; phase shot 12 of 12 running, policy frozen; control model neural policy; training so far 3,000 updates after shot 11 |

**Cosmic-web formation** (Default solver), Leonardo, job 58225927, 19 Sep 2026, lrdn0264

| | |
|---|---|
| Setup | 1,024 grid, 2,400,000 particles, 4,000 steps, 600 frames |
| Total time | **45 s** |
| Physics | 12 s (20 ms/frame) |
| Drawing frames | 24 s (40 ms/frame) |
| Devices | 1 GPU, busy 19% on average; 8 cores, 0.82 busy on average |
| Compute | cupy |
| Result | universe our universe; clumping density contrast 5.51; particles 2,400,000; mesh 1024 × 1024 |

**Bat vs Moth** (Default solver), Leonardo, job 58225928, 19 Sep 2026, lrdn0286

| | |
|---|---|
| Setup | 32,768 bats and moths per cave × 16 caves, 100 generations, 450 frames |
| Total time | **6 min 37 s** |
| Physics | 4 min 20 s (578 ms/frame) |
| Drawing frames | 33 s (72 ms/frame) |
| Devices | 1 GPU, busy 10% on average; 8 cores, 0.98 busy on average |
| Compute | cupy |
| Result | cave 11; catch rate 0.321; jamming evolved at 93; bat points 11.0 |
