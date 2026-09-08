#!/usr/bin/env bash
# Minimal single-process/single-GPU launch, matching Discoverer's public GPU
# job shape.  It distinguishes an allocation-shape/cgroup issue from CuPy.
#SBATCH --job-name=discoverer-one-gpu-probe
#SBATCH --account=ehpc-school-2026
#SBATCH --qos=ehpc-school-2026
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/discoverer-one-gpu-probe_%j.out
#SBATCH --error=logs/discoverer-one-gpu-probe_%j.err

source /etc/profile
set -euo pipefail
module load slurm
module load python312

TEAM=/weka/ehpc-school-2026/abelik
ROOT="$TEAM/Leonardo_Visual_Demos"
VENV="$TEAM/venvs/visual-demos"

[[ -x "$VENV/bin/python" ]] || {
  echo "Missing ARM64 CUDA 13 environment: submit discoverer_gpu_probe.sbatch first." >&2
  exit 2
}

mkdir -p "$TEAM/logs" "$TEAM/cupy-cache"
export CUPY_CACHE_DIR="$TEAM/cupy-cache"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,driver_version,compute_mode,memory.total,memory.used \
  --format=csv,noheader

cd "$ROOT"
"$VENV/bin/python" - <<'PY'
import cupy as cp

print("CuPy:", cp.__version__)
print("CUDA runtime:", cp.cuda.runtime.runtimeGetVersion())
print("Visible devices:", cp.cuda.runtime.getDeviceCount())
values = cp.arange(1_000_000, dtype=cp.float32)
print("GPU reduction:", float(cp.sum(values)))
cp.cuda.get_current_stream().synchronize()
PY

echo "One-GPU allocation probe passed."
