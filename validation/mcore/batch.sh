#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --account=coreai_dlalgo_nemorl
#SBATCH --partition=batch
#SBATCH --time=04:00:00
set -euo pipefail
export SC_REPO=/lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr4963
cd "$SC_REPO"
[[ "$SC_MCORE_LOOKAHEAD" == 0 || "$SC_MCORE_LOOKAHEAD" == 1 ]]
checkpoint="$SC_REPO/validation/mcore/results/yuekaiz-mcore-la${SC_MCORE_LOOKAHEAD}-300/checkpoints"
if python - "$checkpoint" <<'PY'
from pathlib import Path
import json, sys
complete = any(json.loads(p.read_text()).get('current_step', 0) >= 300 for p in Path(sys.argv[1]).glob('step_*/training_info.json'))
sys.exit(0 if complete else 1)
PY
then
  echo "MCore LA${SC_MCORE_LOOKAHEAD} completed 300; skip allocation startup"
  exit 0
fi
export CONTAINER=/lustre/fsw/portfolios/coreai/users/yuekaiz/containers/nemo_rl.1030.sqsh
export MOUNTS=/lustre:/lustre BASE_LOG_DIR="$SC_REPO/validation/mcore/slurm"
export GPUS_PER_NODE=8 CPUS_PER_WORKER=128 OMP_NUM_THREADS=4 ENROOT_ROOTFS_WRITABLE=1
export COMMAND="bash '$SC_REPO/validation/mcore/run.sh'"
bash ray.sub
