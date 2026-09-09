#!/usr/bin/env bash
# Diagnose module visibility inside an interactive Slurm task.

source /etc/profile
set -euo pipefail
module load slurm

exec srun \
  --account=ehpc-school-2026 \
  --qos=ehpc-school-2026 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --time=00:05:00 \
  bash /weka/ehpc-school-2026/abelik/Leonardo_Visual_Demos/scripts/discoverer_python_task_probe.sh
