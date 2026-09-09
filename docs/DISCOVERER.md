# Running on Discoverer (Brain++)

This is the runbook for the team allocation on the Brain++ Discoverer NVL72 system. It is deliberately separate from the Leonardo instructions: the two systems have different CPU architectures, CUDA stacks, Slurm settings, and storage conventions.

## Current status

The source, ARM64 environment recipe, and Slurm launchers have been staged for the `abelik` account. Do not submit the production job: GPU execution is blocked by the CUDA issue below and, as of 9 September, the team Weka filesystem is absent from both reserved compute nodes. See [Known cluster issues](#known-cluster-issues-8-9-september-2026).

## Access and storage

Connect using the registered SSH key:

```bash
ssh -p 2226 USER@login.brainplusplus.bg
```

Load Slurm before using `srun`, `sbatch`, `squeue`, `sacct`, or `scancel`:

```bash
module load slurm
```

Use only the team Weka filesystem for source, environments, caches, frames, and logs. Do not create a virtual environment, Conda installation, or large pip cache under `/home`.

```bash
export TEAM=/weka/ehpc-school-2026/$USER
mkdir -p "$TEAM"/{venvs,runs,logs,cupy-cache}
```

The deployed checkout for the `abelik` account is:

```text
/weka/ehpc-school-2026/abelik/Leonardo_Visual_Demos
```

## Hardware and Slurm model

The login node (`head-01`) is x86_64, but Discoverer GPU nodes are ARM64. The team GPU allocation uses nodes with four NVIDIA GB200/B200 GPUs. They are presented as two two-GPU Grace-Blackwell superchips, and the GPU nodes expose 144 CPU cores.

The public Discoverer+ documentation describes a separate H200 deployment with
`common-gpu`/`common` partitions and a `cuda/12/12.8.0` module. This Brain++
NVL72 allocation instead exposes the `defq` partition, B200 GPUs, and no CUDA
module under any of the documented names. Follow the team-provided account,
QoS, and resource shape for this system; do not substitute the public H200
partition names.

The site-provided allocation shape is one node, eight Slurm tasks, and four GPUs:

```bash
srun --account=ehpc-school-2026 --qos=ehpc-school-2026 \
  --nodes=1 -n 8 --gres=gpu:4 --time=01:00:00 --pty bash
```

The provided Discoverer scripts use the same account, QoS, task count, and GPU request. The reservation is selected automatically by Slurm for that account.

The 3-D galaxy solver is presently one CuPy process on **one active GPU**. It has no domain decomposition or multi-GPU force calculation. It still requests the four-GPU allocation shape required for the team; use an inner `srun --ntasks=1 --gres=gpu:1` so exactly one Python solver is launched and it receives one GPU. Do not start eight Python ranks: that would create eight uncoordinated simulations rather than one 100k-particle system.

## Architecture-safe Python environment

Do **not** create the virtual environment on `head-01`. A venv made there downloads x86_64 wheels and cannot run on an ARM64 GPU node. The smoke-test job creates the environment on a B200 node, producing `aarch64` NumPy, Pillow, and CuPy wheels.

Discoverer has a modern CUDA 13-capable driver. Its dependency file is kept separate from Leonardo's CUDA 12/A100 environment:

```text
requirements-gpu-discoverer.txt   # Discoverer B200 / CUDA 13
requirements-gpu.txt              # Leonardo A100 / CUDA 12
```

The smoke script sets `PIP_NO_CACHE_DIR=1` and writes the CuPy compilation cache under Weka, preventing package downloads from filling `/home`.

Do not use `scripts/submit_leonardo.sh`, the `slurm/run_demo*.sbatch` templates, or `tools/leonardo_preflight.py` on Discoverer. They deliberately contain Leonardo partition, module, filesystem, and Booster-only checks.

## First-time GPU smoke test

From the staged checkout:

```bash
module load slurm
cd /weka/ehpc-school-2026/abelik/Leonardo_Visual_Demos
sbatch scripts/discoverer_gpu_probe.sbatch
```

To check the non-GPU path independently, submit the deliberately CUDA-free probe:

```bash
sbatch scripts/discoverer_cpu_probe.sbatch
```

It uses the supplied account, QoS, one node, and eight Slurm tasks, but no
`--gres` line. It is appropriate for CPU-only preprocessing or a basic
cluster-access check; it does not test GPU functionality.

The job does all of the following:

1. Sources the site profile before enabling Bash `nounset` (the profile has an optional Byobu variable that is unset in a batch shell).
2. Creates the ARM64 venv on the GPU node when needed.
3. Installs `requirements-gpu-discoverer.txt` without a home-directory cache.
4. Confirms the assigned B200, CUDA runtime, and a basic GPU allocation.
5. Runs a three-frame, 384-particle `galaxy_collision_3d` job with the actual CuPy RawKernel and frame-output path.

Monitor it with:

```bash
squeue -u "$USER"
tail -f logs/discoverer-gpu-probe_JOBID.out
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode
```

Only continue if the log ends with `Discoverer GPU smoke test passed.` and the run metadata names a CuPy backend.

## 3-D galaxy orbit pilot and production run

The 3-D galaxy demo applies softened direct all-pairs gravity. With 100,000 particles, 500 saved frames, and six solver substeps per frame, it performs roughly 35 trillion pair-force evaluations. Start with a 10,000-particle, 500-frame pilot after the smoke test succeeds:

```bash
module load slurm
cd /weka/ehpc-school-2026/abelik/Leonardo_Visual_Demos
sbatch --export=ALL,PARTICLES=10000,FRAMES=500 \
  scripts/run_discoverer_galaxy3d.sbatch
```

Inspect the result before scaling:

```bash
RUN=/weka/ehpc-school-2026/abelik/runs/galaxy_collision_3d_JOBID
python -m json.tool "$RUN/meta.json" | less
find "$RUN/frames" -name 'frame_*.jpg' | wc -l
```

The full request is then:

```bash
sbatch --export=ALL,PARTICLES=100000,FRAMES=500 \
  scripts/run_discoverer_galaxy3d.sbatch
```

The production launcher uses the `leonardo` numerical profile only as a high-fidelity parameter preset, then overrides it to 100,000 particles and six substeps. It records detailed timings in `meta.json` and writes each run to:

```text
/weka/ehpc-school-2026/abelik/runs/galaxy_collision_3d_JOBID
```

The numerical calculation includes all simulated particles. The current output renderer writes 1280×720 JPEG frames, and each interactive JSON state samples at most about 9,000 display points to keep browser playback responsive. This is high numerical fidelity, not a 4K or 100k-point browser renderer.

## Retrieve a completed run to the dashboard PC

Discoverer does not need a graphical interface. It writes a portable saved-run
contract: JPEG frames, metadata, and rotatable 3-D JSON states. After a job
completes, run the following on the Windows PC from the project checkout:

```powershell
.\scripts\sync_discoverer_run.ps1 -JobId JOBID
python app.py
```

The script copies the remote run into the local `runs/` directory, which the
dashboard automatically discovers. Select the saved `galaxy_collision_3d` run
in the library to play frames or use **Rotate 3D**. It copies the entire run
and can be rerun to refresh a local copy; transfer after job completion for a
consistent 70- or 500-frame replay.

## Known cluster issues — 8–9 September 2026

The following validation jobs ran on `gpu-11` under reservation `ehpc-school-2026`:

| Job | Result | Finding |
| --- | --- | --- |
| 5874 | failed before GPU work | `/etc/profile` was sourced after `set -u`; its optional `LC_BYOBU` variable was unset. The scripts now source the profile first. |
| 5875–5876 | failed CUDA allocation | The correct ARM64 venv and one-GPU Slurm step were established, but CuPy CUDA 12.9 could not allocate device memory. |
| 5878 | diagnostic failure | B200 was idle, in Default compute mode, with 189,471 MiB free and no compute processes. |
| 5879 | failed CUDA allocation | The same simple allocation failed with CuPy CUDA 13.2, ruling out the Leonardo CUDA 12 wheel as the cause. |
| 5880 | failed CUDA allocation | A direct, documented single-task/8-CPU/16-GB/one-GPU job failed identically, ruling out the four-GPU team allocation shape and nested `srun` step. |
| 5881 | passed CPU-only probe | A standard-Python ARM64 calculation completed on `gpu-11` with eight Slurm tasks and `CUDA_VISIBLE_DEVICES` unset. |
| 5929 | batch launch failure | Fresh one-B200 batch job on `gpu-11` was killed by Slurm signal 53 before the shell script began; no output/error files were created. |
| 5930 | batch launch failure | The identical job on `gpu-12` was killed identically before script execution. |
| 5931 | batch launch failure | The organisers' exact `--nodes=1 --ntasks=8 --gres=gpu:4` shape was allocated on `gpu-11`, then killed by signal 53 before its first command. |
| 5932 | batch launch failure | A CPU-only batch job was also killed by signal 53 before its first command, showing the new failure is not CUDA-specific. |

Job 5879 proves that Slurm granted the requested resources:

```text
Node: gpu-11
Allocation: cpu=8, gres/gpu:b200=4
Inner step: gres/gpu:b200=1
GPU: NVIDIA GB200, driver 580.173.02
Memory: 189471 MiB total, 0 MiB used
Compute mode: Default
```

Despite detecting exactly one Slurm-assigned B200, a minimal `cupy.arange(1_000_000, dtype=cupy.float32)` raises:

```text
cupy_backends.cuda.api.runtime.CUDARuntimeError:
cudaErrorDevicesUnavailable: CUDA-capable device(s) is/are busy or unavailable
```

This occurred with both CuPy CUDA 12.9 and CUDA 13.2, with no GPU processes present. It also occurred in a direct single-process/one-GPU job (`--ntasks=1`, `--cpus-per-task=8`, `--mem=16G`, `--gres=gpu:1`) rather than a nested Slurm step. Conversely, job 5881 proves that the same account, QoS, Slurm scheduler, ARM64 node, and Python module work without GPU access. It is therefore a Discoverer GPU/driver/cgroup availability problem, not a simulation, allocation-shape, superchip architecture, package, or memory-capacity problem.

Ask the Discoverer operators to check GPU access for the Slurm reservation on `gpu-11`, specifically the NVIDIA device cgroup assignment and driver health. Give them this evidence and the log:

```text
/weka/ehpc-school-2026/abelik/Leonardo_Visual_Demos/logs/discoverer-gpu-probe_5879.out
```

### New storage / batch-launch failure — 9 September

The direct Slurm route itself still works: the organisers' `srun` form ran an
eight-task `hostname` command on `gpu-11`. However, direct tasks on **both** of
the reserved nodes see only an empty `/weka` directory and cannot access
`/weka/ehpc-school-2026` at all:

```text
/bin/ls: cannot access '/weka/ehpc-school-2026': No such file or directory
```

Consequently, direct tasks cannot see the staged source or the ARM64 virtual
environment, and all newly submitted `sbatch` jobs are terminated by Slurm
signal 53 before they create their output files. This is a current cluster
mount/prolog or job-launch problem, separate from the 8 September CUDA error.
Ask the operators to restore and verify the Weka mount on `gpu-11` and `gpu-12`
for the `ehpc-school-2026` reservation, and to investigate why its batch jobs
are being killed before the script starts. Re-run the one-GPU smoke test only
after that is confirmed fixed.

After they confirm a fix, resubmit the smoke test before submitting either the 10k pilot or the 100k production job.
