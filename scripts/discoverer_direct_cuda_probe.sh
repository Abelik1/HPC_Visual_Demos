#!/usr/bin/env bash
# Run from Discoverer's login node when sbatch launch is unavailable.  This
# submits a short, direct Slurm step and verifies a real CUDA allocation.

source /etc/profile
set -euo pipefail
module load slurm

TEAM=/weka/ehpc-school-2026/abelik
VENV="$TEAM/venvs/visual-demos"

exec srun \
  --account=ehpc-school-2026 \
  --qos=ehpc-school-2026 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --gres=gpu:1 \
  --time=00:05:00 \
  bash -lc '
source /etc/profile
module load python312
exec env CUPY_CACHE_DIR=/weka/ehpc-school-2026/abelik/cupy-cache /weka/ehpc-school-2026/abelik/venvs/visual-demos/bin/python -c '"'"'
import cupy as cp

print("CuPy:", cp.__version__)
print("Visible devices:", cp.cuda.runtime.getDeviceCount())
values = cp.arange(1_000_000, dtype=cp.float32)
print("GPU reduction:", float(cp.sum(values)))
cp.cuda.get_current_stream().synchronize()
'"'"'
'
