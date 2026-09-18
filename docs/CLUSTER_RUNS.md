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
  fusion guardian). A job takes the team's one-node / four-GPU allocation shape
  and runs the solver on one GB200. Verified end to end on 18 September 2026
  (jobs 8610 and 8611).
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
  `known_hosts`; see docs/TROUBLESHOOTING.md. The default QoS `boost_qos_dbg`
  allows 30 minutes.
* **NBody (MUrB) on a cluster** needs `murb` built there; set `MURB_EXE` in the
  cluster's `job_setup` in `config/clusters.json`
  (`export MURB_EXE=/path/to/murb`).
