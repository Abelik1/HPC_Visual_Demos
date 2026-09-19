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
