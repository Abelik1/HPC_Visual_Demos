#!/bin/bash
# Leonardo Booster: one CUDA process, one A100 and its host-core share.
# Account and QoS are supplied by scripts/submit_leonardo.sh or sbatch flags.
#SBATCH --job-name=visual-demo-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

DEMO=${DEMO:-galaxy_collision}
FRAMES=${FRAMES:-90}
PROFILE=${PROFILE:-hpc}
BACKEND=${LEONARDO_DEMO_BACKEND:-hybrid}
# fp32 (default), fp64, or mixed (galaxy_collision_3d only).  The A100 runs
# FP64 at half its FP32 rate, so fp64 is affordable here, unlike on consumer GPUs.
PRECISION=${PRECISION:-fp32}
if [[ -z "${RUNROOT:-}" ]]; then
  : "${FAST:?FAST is not set; choose a shared Leonardo output filesystem}"
  RUNROOT=$FAST/leonardo_visual_demos
fi
RUN_DIR="$RUNROOT/${DEMO}_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

module purge
module load "${LEONARDO_PROFILE_MODULE:-profile/base}"
module load "${LEONARDO_CUDA_MODULE:-cuda/12.2}"
if [[ -n "${LEONARDO_EXTRA_MODULES:-}" ]]; then
  # shellcheck disable=SC2086
  module load ${LEONARDO_EXTRA_MODULES}
fi

export LEONARDO_DEMO_BACKEND="$BACKEND"
export LEONARDO_DEMO_CPU_WORKERS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export CUPY_CACHE_DIR=${CUPY_CACHE_DIR:-${WORK:-$HOME}/.cache/cupy}
mkdir -p "$CUPY_CACHE_DIR"

cd "${SLURM_SUBMIT_DIR}"
if [[ -n "${LEONARDO_CONTAINER:-}" ]]; then
  command -v singularity >/dev/null
  [[ -r "$LEONARDO_CONTAINER" ]] || { echo "Container not readable: $LEONARDO_CONTAINER" >&2; exit 2; }
  PYTHON=(singularity exec --nv "$LEONARDO_CONTAINER" python3)
else
  if [[ -n "${LEONARDO_VENV:-}" ]]; then
    [[ -x "$LEONARDO_VENV/bin/python" ]] || { echo "Invalid LEONARDO_VENV: $LEONARDO_VENV" >&2; exit 2; }
    PYTHON=("$LEONARDO_VENV/bin/python")
  else
    PYTHON=(python3)
  fi
fi

srun --cpu-bind=cores "${PYTHON[@]}" tools/leonardo_preflight.py \
  --backend "$BACKEND" --demo "$DEMO" --run-root "$RUNROOT"
srun --cpu-bind=cores "${PYTHON[@]}" run_demo.py "$DEMO" \
  --profile "$PROFILE" --backend "$BACKEND" --frames "$FRAMES" --precision "$PRECISION" \
  --run-dir "$RUN_DIR"

echo "Run written to $RUN_DIR"
