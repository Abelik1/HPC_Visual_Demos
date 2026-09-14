#!/usr/bin/env bash
# Demo-day showcase renders on Discoverer B200 GPUs.
#
# Two sets, one node each, several single-GPU runs in parallel per job:
#   SET=a  galaxy_collision_3d A+B, fluid A+B      (4 GPUs)
#   SET=b  fusion_plasma passive A + guardian B     (2 GPUs)
#   SET=c  fluid A+B only                          (2 GPUs)
# PILOT=1 runs every configuration at full resolution for a handful of frames
# so per-frame cost can be measured before the long production job.
#
#   sbatch --export=ALL,SET=a,PILOT=1 scripts/run_discoverer_showcase.sbatch
#   sbatch --export=ALL,SET=a        scripts/run_discoverer_showcase.sbatch
#SBATCH --job-name=showcase
#SBATCH --account=ehpc-school-2026
#SBATCH --qos=ehpc-school-2026
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=1000G
#SBATCH --gres=gpu:4
#SBATCH --time=14:00:00
#SBATCH --output=logs/showcase_%j.out
#SBATCH --error=logs/showcase_%j.err

source /etc/profile
set -euo pipefail
module load slurm
module load python312

SET=${SET:?SET=a or SET=b}
PILOT=${PILOT:-0}
TEAM=/weka/ehpc-school-2026/abelik
ROOT="$TEAM/Leonardo_Visual_Demos"
VENV="$TEAM/venvs/visual-demos"
TORCH_VENV="$TEAM/venvs/visual-demos-torch"
RUNROOT="$TEAM/runs"
LOGDIR="$ROOT/logs/showcase_${SLURM_JOB_ID}"
TAG=$([[ "$PILOT" == 1 ]] && echo "pilot${SLURM_JOB_ID}" || echo "showcase${SLURM_JOB_ID}")

mkdir -p "$LOGDIR" "$TEAM/cupy-cache" "$RUNROOT"
export CUPY_CACHE_DIR="$TEAM/cupy-cache"
export LEONARDO_DEMO_TUNING_CACHE="$TEAM/cupy-cache/kernel_tuning.json"
export LEONARDO_DEMO_CPU_WORKERS=12
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
export PIP_NO_CACHE_DIR=1
cd "$ROOT"
echo "Host $(hostname) · set $SET · pilot $PILOT · tag $TAG"
nvidia-smi -L

# Guardian mode trains a PyTorch policy. Torch lives in its own venv so its
# CUDA wheels can never change which libraries the CuPy-only runs load.
ensure_torch_venv(){
  if srun --ntasks=1 --cpus-per-task=16 --mem=64G --gres=gpu:1 "$TORCH_VENV/bin/python" -c \
       "import torch,cupy;assert torch.cuda.is_available();print('torch',torch.__version__,torch.version.cuda)" 2>/dev/null; then
    return 0
  fi
  echo "Building $TORCH_VENV on the GPU node"
  rm -rf "$TORCH_VENV"
  srun --ntasks=1 --cpus-per-task=16 --mem=64G --gres=gpu:1 bash -c "
    set -e
    python3 -m venv '$TORCH_VENV'
    '$TORCH_VENV/bin/python' -m pip install --upgrade pip
    '$TORCH_VENV/bin/python' -m pip install -r requirements-gpu-discoverer.txt
    '$TORCH_VENV/bin/python' -m pip install torch --index-url https://download.pytorch.org/whl/cu130 \
      || '$TORCH_VENV/bin/python' -m pip install torch --index-url https://download.pytorch.org/whl/cu128
    '$TORCH_VENV/bin/python' -c 'import torch,cupy;assert torch.cuda.is_available(),\"torch has no CUDA\";print(\"torch\",torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))'
  "
}

names=(); pids=()
launch(){ # launch NAME PYTHON ARGS...
  local name=$1 python=$2; shift 2
  local dir="$RUNROOT/${name}_${TAG}"
  rm -rf "$dir"
  echo "[$(date +%T)] start $name -> $dir"
  srun --exact --ntasks=1 --cpus-per-task=16 --mem=220G --gres=gpu:1 \
    "$python" run_demo.py "$@" --timings --run-dir "$dir" > "$LOGDIR/$name.log" 2>&1 &
  names+=("$name"); pids+=($!)
}
p(){ [[ "$PILOT" == 1 ]] && echo "$1" || echo "$2"; }   # p PILOT_VALUE PRODUCTION_VALUE
launch_fluid(){
  # Lattice-Boltzmann at 3840x2160 and 5120x2880. Steps give ~1.5 flow-through
  # times so the wake fully develops. Reynolds number rises with the lattice
  # (~1400 and ~1050 here), so the inlet speed is lowered to keep BGK stable.
  # Tracers are drawn at 1280x720 whatever the lattice size: a few thousand
  # show the flow; tens of thousands cover it in a white carpet of streaks.
  launch fluid_A "$VENV/bin/python" fluid --profile hpc --backend hybrid --frames "$(p 4 600)" \
    --setting nx=3840 --setting ny=2160 --setting total_steps="$(p 400 125000)" \
    --setting tracers=2600 --setting trail=16 --setting tracer_boost=9 \
    --param speed=0.05 --param obstacle=0
  launch fluid_B "$VENV/bin/python" fluid --profile hpc --backend hybrid --frames "$(p 4 600)" \
    --setting nx=5120 --setting ny=2880 --setting total_steps="$(p 400 170000)" \
    --setting tracers=3000 --setting trail=16 --setting tracer_boost=9 \
    --param speed=0.045 --param obstacle=1
}

if [[ "$SET" == a ]]; then
  # Full direct all-pairs gravity. A: two million bodies on the recommended
  # orbit. B: a million bodies, heavier Andromeda on a grazing, strongly
  # tilted pass, with FP64 force sums. (Pilot: ~0.5 s per force evaluation at
  # 1M on one GB200, so A is ~3.5 h and B ~1.3 h including rendering.)
  launch galaxy_collision_3d_A "$VENV/bin/python" galaxy_collision_3d --profile hpc --backend hybrid \
    --method leapfrog --frames "$(p 6 600)" --precision fp32 \
    --setting particles="$(p 1000000 2000000)" --setting substeps=8 --setting max_step=0.02 --setting force_tile=256 \
    --setting span_gyr="$(p 1.5 8.0)" \
    --param impact=0.35 --param speed=1.0 --param disc_tilt=35 --param softening=3.0
  launch galaxy_collision_3d_B "$VENV/bin/python" galaxy_collision_3d --profile hpc --backend hybrid \
    --method leapfrog --frames "$(p 6 600)" --precision mixed \
    --setting particles="$(p 600000 1000000)" --setting substeps=8 --setting max_step=0.02 --setting force_tile=256 \
    --setting span_gyr="$(p 1.5 9.0)" \
    --param impact=0.7 --param speed=0.8 --param disc_tilt=70 --param softening=3.0 \
    --param milky_way_mass=1.2 --param andromeda_mass=2.2
  launch_fluid
elif [[ "$SET" == b ]]; then
  ensure_torch_venv
  # A: passive confinement on a 1536-row plasma lattice.
  launch fusion_plasma_A "$VENV/bin/python" fusion_plasma --profile hpc --backend gpu --method passive \
    --frames "$(p 4 600)" --setting n="$(p 1024 1536)" --setting total_steps="$(p 120 24000)" \
    --setting tracers=6000 --setting trail=20 --setting ensemble=1 \
    --param magnetic_field=5.5 --param heating=32 --param density=1.0 --param _parallel_count=1
  # B: the AI guardian, twelve shots, a far larger training batch, and a
  # re-test against 16 unseen instability drives.
  launch fusion_plasma_B "$TORCH_VENV/bin/python" fusion_plasma --profile hpc --backend gpu --method guardian \
    --frames "$(p 6 600)" --setting n=512 --setting total_steps="$(p 120 6000)" \
    --setting particles=3000 --setting shots="$(p 2 12)" --setting train_updates="$(p 20 3000)" \
    --setting batch=8192 --setting horizon=48 --setting display_steps=96 --setting learning_rate=0.003 \
    --setting ensemble="$(p 2 16)" --setting tracers=3000 --setting trail=16 \
    --param magnetic_field=5.0 --param heating=28 --param instability=1.25 --param _parallel_count="$(p 2 16)"
elif [[ "$SET" == c ]]; then
  launch_fluid
else
  echo "Unknown SET=$SET" >&2; exit 2
fi

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "[$(date +%T)] done  ${names[$i]}"
  else echo "[$(date +%T)] FAILED ${names[$i]} (see $LOGDIR/${names[$i]}.log)"; status=1; fi
done
# One line per run: what it cost, for sizing the next job.
for n in "${names[@]}"; do
  "$VENV/bin/python" - "$RUNROOT/${n}_${TAG}/meta.json" <<'PY' || true
import json,sys
m=json.load(open(sys.argv[1])); t=m.get('timings') or {}
sec=lambda k:round((t.get(k) or {}).get('seconds',0),1)
print(m.get('demo'),m.get('status'),'frames',m.get('frames'),'elapsed',round(m.get('elapsed') or 0),
      'sim',sec('simulation'),'render',sec('render'),'backend',m.get('backend'),'error',m.get('error'))
PY
done
exit $status
