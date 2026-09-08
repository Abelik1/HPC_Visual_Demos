#!/usr/bin/env bash
# Pure CPU/ARM64 scheduling probe for Discoverer.  It deliberately has no
# --gres request and never calls a CUDA program or imports a GPU library.
#SBATCH --job-name=discoverer-cpu-probe
#SBATCH --account=ehpc-school-2026
#SBATCH --qos=ehpc-school-2026
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=00:05:00
#SBATCH --output=logs/discoverer-cpu-probe_%j.out
#SBATCH --error=logs/discoverer-cpu-probe_%j.err

source /etc/profile
set -euo pipefail
module load slurm
module load python312

TEAM=/weka/ehpc-school-2026/abelik
mkdir -p "$TEAM/logs"

echo "Host: $(hostname)"
echo "Architecture: $(uname -m)"
echo "Allocated CPUs: ${SLURM_NTASKS:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
python3 - <<'PY'
from math import fsum

values = (1.0 / (index + 1) for index in range(2_000_000))
print("CPU calculation:", fsum(values))
PY

echo "CPU-only probe passed."
