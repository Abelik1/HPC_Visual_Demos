#!/usr/bin/env bash
# Full direct 3-D galaxy-orbit render on Discoverer B200 GPUs.
#SBATCH --job-name=galaxy3d
#SBATCH --account=ehpc-school-2026
#SBATCH --qos=ehpc-school-2026
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=logs/galaxy3d_%j.out
#SBATCH --error=logs/galaxy3d_%j.err

source /etc/profile
set -euo pipefail
module load slurm
module load python312

TEAM=/weka/ehpc-school-2026/abelik
ROOT="$TEAM/Leonardo_Visual_Demos"
VENV="$TEAM/venvs/visual-demos"
RUNROOT="$TEAM/runs"
PARTICLES=${PARTICLES:-100000}
FRAMES=${FRAMES:-500}
# fp32 (default), mixed or fp64.  B200 FP64 runs at about half the FP32 rate.
PRECISION=${PRECISION:-fp32}
RUN_DIR="$RUNROOT/galaxy_collision_3d_${SLURM_JOB_ID}"

[[ -x "$VENV/bin/python" ]] || {
  echo "Missing ARM64 environment: submit scripts/discoverer_gpu_probe.sbatch first." >&2
  exit 2
}

mkdir -p "$TEAM/logs" "$TEAM/cupy-cache" "$RUN_DIR"
export CUPY_CACHE_DIR="$TEAM/cupy-cache"
# Kernel launch shapes are autotuned once per GPU model and cached here.
export LEONARDO_DEMO_TUNING_CACHE="$TEAM/cupy-cache/kernel_tuning.json"
export LEONARDO_DEMO_CPU_WORKERS=8
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

cd "$ROOT"
echo "Starting ${PARTICLES} particles, ${FRAMES} frames, ${PRECISION} in ${RUN_DIR}"
nvidia-smi -L

# One rank / one active B200: this model is a direct all-pairs system and the
# current solver has no domain decomposition.  Four requested GPUs satisfy
# the site's required allocation shape but do not accelerate a single orbit.
srun --ntasks=1 --gres=gpu:1 "$VENV/bin/python" run_demo.py galaxy_collision_3d \
  --profile hpc --backend hybrid --frames "$FRAMES" --precision "$PRECISION" \
  --setting particles="$PARTICLES" --setting substeps=6 --setting max_step=0.025 \
  --param impact=0.35 --param speed=1.0 --param disc_tilt=35 --param softening=4.0 \
  --timings --run-dir "$RUN_DIR"

echo "Run complete: $RUN_DIR"
