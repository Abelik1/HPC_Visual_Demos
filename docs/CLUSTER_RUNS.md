# Demo days and cluster runs

Two things for running the project as two separate demo days, one for
Discoverer and one for Leonardo.

## Demo-day lineups

`config/lineups.json` lists the demos each day shows, in order, plus the
archive and the video items. Both front ends read it:

* **Demo day: Discoverer / Leonardo / All demos**: buttons above the dashboard
  gallery and on the `/demo` picker. The choice is stored on the server, so every
  browser and both front ends follow it.
* **⚙ Lineups & HPC → Demo-day lineups**: move demos between the Discoverer
  list, the Leonardo list, *Not shown* and *Archive*, and reorder them. A demo
  can be in both days. Save, and the gallery and the picker update at once.
* **Video items** (Raytracer, Pac-Man Agentic) are tiles that play every video
  in their folder under `videos/` (for example `videos/raytracer/`). Give one a
  web link instead, and the tile opens that link. Add more with **+ Add a video item**.

Archived demos still run: the dashboard lists them under **Archive**, and
`/?demo=<id>` links and saved runs keep working.

Shipped lineups:

| Discoverer | Leonardo |
| --- | --- |
| Galaxy collision 3D | NBody (MUrB) |
| Virtual wind tunnel | Raytracer *(video)* |
| Neuro-Racers | Star in a Bottle |
| Black-hole lensing | Cosmic web formation |
| Molecular Machine | Bat vs Moth |
| | Pac-Man Agentic *(video)* |

Archived: 2-D galaxy collision, neural image compression, and the older
work-in-progress demos.

## Running a simulation on the cluster

In the dashboard's **Run configuration**, or in demo mode's ⚙ drawer, set
**Run on** to Discoverer or Leonardo. Choosing a cluster switches the quality
preset to HPC, and you can change it back. **Run** then opens a confirmation
listing:

* the experiment, solver, preset, frame count, compute mode and precision;
* every experiment setting, with the ones you changed highlighted;
* every simulation value, compared with the preset;
* the Slurm account, QoS, partition, **time limit** (editable here) and the
  exact `sbatch`/`srun` resources;
* warnings: no account set, an expired Leonardo certificate, a non-HPC preset.

On **Submit**, the viewer (`leonardo_demos/remote.py`):

1. uploads the solver code (and, the first time, `data/`) to the cluster checkout
   as a tarball, but only if it differs from this PC's copy;
2. writes `job.json` and `job.sbatch` into `<run folder>/<run id>/` and runs `sbatch`;
3. polls `squeue`/`sacct` and the remote `meta.json` every 15 s. The page shows
   *queued → running · frame n of N → fetching*;
4. when the job ends, streams the run directory home into `runs/<run id>` and
   the page plays it.

You can leave the page: the viewer keeps watching. If the viewer is restarted
(`python app.py`), it picks the job up again. **Lineups & HPC → Cluster jobs**
lists recent cluster runs, with **Cancel** (`scancel`) and **Open**.

Visitor-made inputs travel with the job: drawn wind-tunnel obstacles, built
brains and custom neural-wall pictures. Neuro-Racers **ghost races** use
champions saved on this PC and are refused for cluster runs.

### Setting up each cluster

**Lineups & HPC → Clusters** edits the connection. Edits are saved to
`config/clusters.local.json`, which git ignores; the defaults are in
`config/clusters.json`. **Test connection** checks the SSH login, Slurm, the
checkout, the Python environment and the run folder.

* **Discoverer** works as shipped: `abelik@login.brainplusplus.bg:2226`, account
  and QoS `ehpc-school-2026`, the existing Weka venvs (the torch venv for the
  fusion guardian). A job asks only for what its demo uses (see "Right-sized
  jobs" below). Verified end to end on 18 September 2026 (jobs 8610 and 8611,
  which still used the old whole-node shape).
* **Leonardo** needs three things first:
  1. the **Slurm account** (`saldo -b` or
     `sacctmgr show associations user=$USER format=account` on Leonardo). It is
     a placeholder until you enter it;
  2. a valid **CINECA SSH certificate**, which lasts 12 hours. Enter your UserDB
     e-mail, then **Refresh certificate** opens `scripts/leonardo_login.ps1
     -CertOnly` in a PowerShell window for the browser/OTP sign-in;
  3. a **Python venv** at `$WORK/venvs/leonardo-visual-demos`
     (docs/LEONARDO.md §3). The code checkout (`$WORK/Leonardo_Visual_Demos`) is
     created by the first run.

  The login host is pinned to `login01-ext.leonardo.cineca.it`, because the
  round-robin name serves four different host keys. Its key must be in
  `known_hosts`; see docs/TROUBLESHOOTING.md. Jobs use `boost_usr_prod` (GPU)
  and `dcgp_usr_prod` (CPU) with the partition's default QoS and a 40-minute
  limit; `boost_qos_dbg` (30 minutes) can be set as the QoS in HPC settings.
* **NBody (MUrB) on a cluster** needs `murb` built there; set `MURB_EXE` in the
  cluster's `job_setup` in `config/clusters.json`
  (`export MURB_EXE=/path/to/murb`).

## Right-sized jobs: the device plan

Until 18 September every cluster job asked Discoverer for a whole node (4 GPUs,
128 cores) and used one GPU. Slurm bills allocated resources, busy or not
(`defq` weights: 1 per GPU, 1/36 per core, memory free), so each job billed
about 7 units/hour instead of about 1.2–1.7.

`config/hpc_plan.json` → `demos` now says what each demo actually uses, and
`leonardo_demos/remote.py` builds every job's request from it:

| Demo | Device | Request | Why |
| --- | --- | --- | --- |
| 3-D galaxy | GPU | 1 GPU + 8 cores | all-pairs gravity, 24× faster on the GPU |
| Wind tunnel | GPU | 1 GPU + 24 cores | LBM ~100× faster on the GPU; cores draw frames in parallel |
| Neuro-Racers, Bat vs Moth | GPU | 1 GPU + 8 cores | population evolution (the pilot confirms at HPC size) |
| Black hole | GPU | 1 GPU + 16 cores | ray tracing 11× faster on the GPU |
| Molecular fold | GPU | 1 GPU + 4 cores | one fused CUDA kernel |
| Molecular shuttle | **CPU** | 2 cores, no GPU | 12 beads cannot keep a GPU busy |
| NBody (MUrB) | **CPU** (DCGP on Leonardo) | 32 cores | OpenMP; GPU only once murb is built with CUDA |
| Star in a Bottle, cosmic web | GPU | 1 GPU + 8 cores | a quarter of a Booster node |

On Leonardo, GPU demos go to `boost_usr_prod` and CPU demos to `dcgp_usr_prod`
(which can have its own account: `cpu_account` in HPC settings).

"Auto" compute on a cluster means the plan's backend, not whatever the node has.

## Drawing no longer idles the GPU

Several demos spent most of their run drawing frames on one core while the GPU
waited (13 Sep: wind tunnel 22 s physics / 213 s drawing; Star in a Bottle
passive 31 s / 609 s, plus ~330 s of untimed host work). Two changes:

* **Reduce on the device.** The wind tunnel and Star in a Bottle now compute
  the colour field, tracer advection, drift field, turbulence score and 3-D
  view texture where the field lives, and copy only output-sized results.
* **Draw in parallel.** `leonardo_demos/pipeline.py` (`FramePipeline`) draws
  and encodes frames in worker processes on the allocated cores while the
  solver continues; a frame is published to the viewer only when it and all
  earlier frames are on disk.

The black hole's drawing is still serial (it was being reworked separately).

## Every cluster run records what it used

`tools/run_job.py` samples `nvidia-smi` and counts busy cores; the run's
`meta.json` gets `device_usage` (GPUs/cores allocated, mean GPU utilisation,
average busy cores). `tools/demo_stats.py` turns runs into the presenters'
table: setup, total time, physics, drawing, devices, result.

Locally, `tools/profile_demo.py` / `tools/profile_lineup.py` do the same
measurement (results: `benchmarks/devices_baseline.jsonl`). GPU numbers are only
meaningful with nothing else on the GPU (a game running skews them badly).

## The HPC campaign (every demo, under 30 minutes each)

`config/hpc_plan.json` → `campaign.<machine>` is each demo's production run,
sized for ≈25 minutes (1M bodies at most). `tools/hpc_campaign.py`:

```bash
python tools/hpc_campaign.py plan discoverer            # what, where, cost; submits nothing
python tools/hpc_campaign.py pilot discoverer --confirm # full scale, few frames, all in parallel
python tools/hpc_campaign.py estimate discoverer        # production minutes from the pilots
python tools/hpc_campaign.py run discoverer --confirm   # only runs estimated under 25 min
```

Estimates are conservative: pilot physics time scaled by the work (steps,
generations, frames) plus the pilot's remaining time scaled by frames, with no
credit for overlap. A run over target is skipped unless `--force`. Results land
in `runs/` and `benchmarks/campaign_<machine>_*`.
