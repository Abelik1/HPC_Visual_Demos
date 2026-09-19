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

**MUrB N-body (NBody-EuroHPC)** (GPU · CUDA tiled, device-resident), Leonardo, job 58232031, 19 Sep 2026, lrdn0954

| | |
|---|---|
| Setup | 1,000,000 bodies, 480 iterations, MUrB gpu+tile+full, 17.93 TFLOP/s, 120 frames |
| Total time | **12 min 46 s** |
| Physics | 8 min 55 s (4.46 s/frame) |
| Drawing frames | 25 min 08 s (12.57 s/frame) |
| Devices | 1 GPU, busy 70% on average; 8 cores, 2.69 busy on average |
| Compute | MUrB gpu+tile+full |
| Result | code MUrB (NBody-EuroHPC); backend gpu+tile+full; bodies 1,000,000; iteration 480 |

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

**Star in a Bottle** (Mode 2 · AI plasma guardian (3D)), Leonardo, job 58231329, 19 Sep 2026, lrdn2045

| | |
|---|---|
| Setup | 16 simulations run together, 512² grid, 10 shots, 1,500 neural-network updates, 600 frames |
| Total time | **9 min 41 s** |
| Physics | 4 min 10 s (416 ms/frame) |
| Drawing frames | 2 min 45 s (276 ms/frame) |
| Devices | 1 GPU, busy 16% on average; 8 cores, 0.86 busy on average |
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
