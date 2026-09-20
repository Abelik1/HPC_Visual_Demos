# Demo-day run statistics

One table per demo, built from each run's own recorded timings (`tools/demo_stats.py`).
Physics = the solver. Drawing = turning results into frames; where drawing runs in parallel worker processes it is CPU time summed over the workers, so it can exceed the total time.
Devices = what the job was allocated, and how busy it actually kept it (cluster runs from 19 Sep).

## Discoverer demo day

**Galaxy collision - full 3D gravity** (Leapfrog (recommended)), gpu-11, 13 Sep 2026

_Run on Discoverer on 13 Sep (showcase job 7653), before jobs were right-sized: it held a whole node. Not re-run on purpose._

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

**Molecular Machine** (Fold your own protein), Discoverer, job 8686, 19 Sep 2026, gpu-11

| | |
|---|---|
| Setup | 1,000-bead chain, 6,000,000 Langevin steps, 300 frames |
| Total time | **22 min 22 s** |
| Physics | 21 min 47 s (4.36 s/frame) |
| Drawing frames | 29 s (97 ms/frame) |
| Devices | 1 GPU, busy 98% on average; 4 cores, 1.0 busy on average |
| Compute | cupy |
| Result | folded to Rg 5.079, 99% of oily beads buried, 0 salt bridges |

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

**Molecular Machine** (Walking motor (a Brownian ratchet)), AlexMainDesktop, 19 Sep 2026

_The walking motor is two particles: it runs in seconds on any CPU, so it was rendered on the exhibition PC._

| | |
|---|---|
| Setup | two-footed walker on a flashing ratchet, fuel 0.80, load 0.00, 800,000 Langevin steps, 300 frames |
| Total time | **27 s** |
| Physics | 13 s (45 ms/frame) |
| Drawing frames | 12 s (41 ms/frame) |
| Compute | numpy |
| Result | walked +24 steps (27 forward, 3 back) on 238 fuel flashes |

**Molecular Machine** (Rotary motor (ATP synthase)), AlexMainDesktop, 20 Sep 2026

_The rotary motor is a single angle: the physics costs 0.14 s and the rest is drawing, so it too was rendered on the exhibition PC._

| | |
|---|---|
| Setup | rotary motor stepping 120° at a time, proton flow 0.70, cargo 0.20, 120,000 Langevin steps, 300 frames |
| Total time | **15 s** |
| Physics | 0.14 s (466 µs/frame) |
| Drawing frames | 13 s (42 ms/frame) |
| Compute | numpy |
| Result | turned +14.33 times (+43 steps of 120°) on 43 fuel events |

## Leonardo demo day

**MUrB N-body (NBody-EuroHPC)** (GPU · CUDA tiled, device-resident), Leonardo, job 58232031, 19 Sep 2026, lrdn0954

_MUrB's CUDA path on one A100: about 92% of the A100's single-precision peak._

| | |
|---|---|
| Setup | 1,000,000 bodies, 480 iterations, MUrB gpu+tile+full, 17.93 TFLOP/s, 120 frames |
| Total time | **12 min 46 s** |
| Physics | 8 min 55 s (4.46 s/frame) |
| Drawing frames | 25 min 08 s (12.57 s/frame) |
| Devices | 1 GPU, busy 70% on average; 8 cores, 2.69 busy on average |
| Compute | MUrB gpu+tile+full |
| Result | code MUrB (NBody-EuroHPC); backend gpu+tile+full; bodies 1,000,000; iteration 480 |

**MUrB N-body (NBody-EuroHPC)** (CPU · OpenMP (all cores)), Leonardo, job 58225921, 19 Sep 2026, lrdn4323

_The same code on 32 CPU cores of a DCGP node, for comparison (100,000 bodies)._

| | |
|---|---|
| Setup | 100,000 bodies, 1,440 iterations, MUrB cpu+omp, 1.04 TFLOP/s, 360 frames |
| Total time | **11 min 19 s** |
| Physics | 4 min 36 s (766 ms/frame) |
| Drawing frames | 6 min 43 s (1.12 s/frame) |
| Devices | 32 cores, 12.92 busy on average |
| Compute | MUrB cpu+omp |
| Result | code MUrB (NBody-EuroHPC); backend cpu+omp; bodies 100,000; iteration 1,440 |

**Star in a Bottle** (Mode 1 · Passive confinement), Leonardo, job 58231322, 19 Sep 2026, lrdn1981

| | |
|---|---|
| Setup | 2,048² grid, 40,000 steps, 6,000 tracers, 5.5 T field, 32 MW heating, 600 frames |
| Total time | **3 min 09 s** |
| Physics | 2 min 31 s (251 ms/frame) |
| Drawing frames | 20 min 47 s (2.08 s/frame) |
| Devices | 1 GPU, busy 83% on average; 8 cores, 7.38 busy on average |
| Compute | cupy + CPU frame workers |
| Result | mode passive confinement; magnetic field 5.5 T; heating power 32 MW; density 1.00 n₀ |

**Star in a Bottle** (Mode 2 · AI plasma guardian (3D)), Leonardo, job 58296859, 20 Sep 2026, lrdn0214

_The training budget is spread geometrically over the shots (2, 5, 11, 26, 58, 131, 295, 666, 1,500 updates), so the scoreboard shows a learning curve: 82,150 markers lost untrained, then 78,780, 68,097, 12,044, 44,681, 5,853, 1,369, 1,282, 1,253, 1,256 against a no-control reference of 83,457. The step back at shot 5 is real and was kept._

| | |
|---|---|
| Setup | 16 simulations run together, 512² grid, 10 shots, 1,500 neural-network updates, 600 frames |
| Total time | **10 min 12 s** |
| Physics | 4 min 21 s (434 ms/frame) |
| Drawing frames | 2 min 50 s (284 ms/frame) |
| Devices | 1 GPU, busy 16% on average; 8 cores, 0.85 busy on average |
| Compute | cupy + torch·cuda |
| Result | mode AI plasma guardian; phase shot 10 of 10 running, policy frozen; control model neural policy; training so far 1,500 updates after shot 9 |

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
