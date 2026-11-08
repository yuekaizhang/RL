#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --account=coreai_dlalgo_nemorl
#SBATCH --partition=batch
#SBATCH --time=00:25:00
set -euo pipefail
[[ "$USER" == yuekaiz && -n "${SLURM_JOB_ID:-}" ]]
repo=/lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr4963
root="validation/kl_diagnosis/results/layer-$SLURM_JOB_ID"
mkdir -p "$repo/$root"
srun --container-image=/lustre/fsw/portfolios/coreai/users/yuekaiz/containers/nemo_rl.1030.sqsh \
  --container-mounts=/lustre:/lustre --container-workdir="$repo" bash -c '
set -euo pipefail
source validation/scripts/environment.sh
[[ "$USER" == root && -n "${SLURM_JOB_ID:-}" ]]
/opt/ray_venvs/nemo_rl.models.policy.workers.automodel_policy_worker.AutomodelPolicyWorker/bin/python validation/kl_diagnosis/layer_policy.py "$1"
/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python validation/kl_diagnosis/layer_generation.py "$1"
' bash "$root"
