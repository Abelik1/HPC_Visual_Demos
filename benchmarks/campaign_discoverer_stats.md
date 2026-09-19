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
