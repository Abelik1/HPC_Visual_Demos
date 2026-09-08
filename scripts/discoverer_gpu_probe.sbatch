#!/usr/bin/env bash
# Discoverer / Brain++ B200 smoke test.  Build the venv on the ARM64 GPU
# node, never on the x86_64 login host.
#SBATCH --job-name=discoverer-gpu-probe
#SBATCH --account=ehpc-school-2026
#SBATCH --qos=ehpc-school-2026
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:4
#SBATCH --time=00:15:00
#SBATCH --output=logs/discoverer-gpu-probe_%j.out
#SBATCH --error=logs/discoverer-gpu-probe_%j.err

# Discoverer's login profile references optional Byobu variables.  Source it
# before enabling nounset so a batch shell has the same module environment as
# an interactive login without failing on an unset cosmetic variable.
source /etc/profile
set -euo pipefail
module load slurm
module load python312

TEAM=/weka/ehpc-school-2026/abelik
ROOT="$TEAM/Leonardo_Visual_Demos"
VENV="$TEAM/venvs/visual-demos"
RUNROOT="$TEAM/runs"

mkdir -p "$TEAM/logs" "$TEAM/cupy-cache" "$RUNROOT"
export PIP_NO_CACHE_DIR=1
export CUPY_CACHE_DIR="$TEAM/cupy-cache"
export LEONARDO_DEMO_CPU_WORKERS=8
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

echo "Host: $(hostname)"
uname -m
nvidia-smi -L
echo "Allocation CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
python3 --version

# This directory was deliberately not created on the x86_64 login node.
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install --no-cache-dir -r "$ROOT/requirements-gpu-discoverer.txt"
fi

cd "$ROOT"

# Use a single Python rank.  The site allocation exposes four B200 GPUs, but
# this direct N-body implementation has one CuPy device context and cannot
# use multiple GPUs for one physically coupled system.
srun --ntasks=1 --gres=gpu:1 nvidia-smi \
  --query-gpu=index,name,driver_version,pstate,compute_mode,memory.total,memory.used \
  --format=csv,noheader
srun --ntasks=1 --gres=gpu:1 nvidia-smi \
  --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
srun --ntasks=1 --gres=gpu:1 "$VENV/bin/python" - <<'PY'
import platform
import cupy as cp

print("Python architecture:", platform.machine())
print("CuPy:", cp.__version__)
print("CUDA runtime:", cp.cuda.runtime.runtimeGetVersion())
print("Visible devices:", cp.cuda.runtime.getDeviceCount())
print("Device 0:", cp.cuda.runtime.getDeviceProperties(0)["name"])
values = cp.arange(1_000_000, dtype=cp.float32)
print("GPU reduction:", float(cp.sum(values)))
cp.cuda.get_current_stream().synchronize()
PY

# Exercise the actual RawKernel-based 3-D solver and frame-output contract.
srun --ntasks=1 --gres=gpu:1 "$VENV/bin/python" run_demo.py galaxy_collision_3d \
  --profile local --backend hybrid --frames 3 \
  --setting particles=384 --setting substeps=1 --setting span_gyr=0.01 \
  --timings --run-dir "$RUNROOT/galaxy3d_smoke_${SLURM_JOB_ID}"

echo "Discoverer GPU smoke test passed."
