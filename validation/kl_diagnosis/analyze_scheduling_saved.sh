#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=coreai_dlalgo_nemorl
#SBATCH --partition=batch
#SBATCH --time=00:08:00
set -euo pipefail
[[ "$USER" == yuekaiz && -n "${SLURM_JOB_ID:-}" ]]
source_job=${1:?Pass the completed capture job ID}
[[ "$source_job" =~ ^[0-9]+$ ]]
repo=/lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr4963
source_dir="validation/kl_diagnosis/results/$source_job"
[[ -s "$repo/$source_dir/scheduling-repeats.pt" ]]
srun --container-image=/lustre/fsw/portfolios/coreai/users/yuekaiz/containers/nemo_rl.1030.sqsh \
  --container-mounts=/lustre:/lustre --container-workdir="$repo" bash -c '
set -euo pipefail
source validation/scripts/environment.sh
[[ "$USER" == root && -n "${SLURM_JOB_ID:-}" ]]
/opt/nemo_rl_venv/bin/python validation/kl_diagnosis/analyze_scheduling.py "$1"
' bash "$source_dir"
