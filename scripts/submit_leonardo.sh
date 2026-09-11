#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=${LEONARDO_CONFIG:-$ROOT/config/leonardo.env}
if [[ -f "$CONFIG" ]]; then
  # shellcheck source=/dev/null
  source "$CONFIG"
fi

DEMO=${1:-galaxy_collision}
FRAMES=${2:-90}
PROFILE=${3:-hpc}
MODE=${4:-${LEONARDO_MODE:-hybrid}}
PRECISION=${5:-${PRECISION:-fp32}}
case "$PRECISION" in
  fp32|fp64|mixed) ;;
  *) echo "Precision must be fp32, fp64, or mixed (got: $PRECISION)" >&2; exit 2 ;;
esac
: "${LEONARDO_ACCOUNT:?Set LEONARDO_ACCOUNT to the active CINECA project account}"
if [[ "$LEONARDO_ACCOUNT" == "YOUR_ACTIVE_PROJECT_ACCOUNT" ]]; then
  echo "Replace the placeholder LEONARDO_ACCOUNT in $CONFIG" >&2
  exit 2
fi
if [[ ! "$FRAMES" =~ ^[1-9][0-9]*$ ]]; then
  echo "Frames must be a positive integer (got: $FRAMES)" >&2
  exit 2
fi
if [[ -z "${LEONARDO_RUNROOT:-}" ]]; then
  : "${FAST:?FAST is not set; log in to Leonardo or set LEONARDO_RUNROOT explicitly}"
fi
RUNROOT=${LEONARDO_RUNROOT:-$FAST/leonardo_visual_demos}

case "$MODE" in
  cpu|numpy)
    JOB=$ROOT/slurm/run_demo_cpu.sbatch
    BACKEND=cpu
    QOS=${LEONARDO_CPU_QOS:-dcgp_qos_dbg}
    ;;
  gpu|cuda|cupy)
    JOB=$ROOT/slurm/run_demo.sbatch
    BACKEND=gpu
    QOS=${LEONARDO_GPU_QOS:-boost_qos_dbg}
    ;;
  hybrid)
    JOB=$ROOT/slurm/run_demo.sbatch
    BACKEND=hybrid
    QOS=${LEONARDO_GPU_QOS:-boost_qos_dbg}
    ;;
  *)
    echo "Mode must be cpu, gpu, or hybrid (got: $MODE)" >&2
    exit 2
    ;;
esac

mkdir -p "$RUNROOT"
cd "$ROOT"
sbatch --account="$LEONARDO_ACCOUNT" --qos="$QOS" \
  --export=ALL,DEMO="$DEMO",FRAMES="$FRAMES",PROFILE="$PROFILE",RUNROOT="$RUNROOT",LEONARDO_DEMO_BACKEND="$BACKEND",PRECISION="$PRECISION" \
  "$JOB"
