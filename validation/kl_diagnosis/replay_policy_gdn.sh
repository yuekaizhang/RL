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
source_job=${1:?Pass the capture job ID}
[[ "$source_job" =~ ^[0-9]+$ ]]
repo=/lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr4963
srun --container-image=/lustre/fsw/portfolios/coreai/users/yuekaiz/containers/nemo_rl.1030.sqsh \
  --container-mounts=/lustre:/lustre --container-workdir="$repo" bash -c '
set -euo pipefail
source validation/scripts/environment.sh
[[ "$USER" == root && -n "${SLURM_JOB_ID:-}" ]]
/opt/ray_venvs/nemo_rl.models.policy.workers.automodel_policy_worker.AutomodelPolicyWorker/bin/python validation/kl_diagnosis/analyze_policy_gdn.py "$1"
/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python validation/kl_diagnosis/analyze_gdn_kernels.py "$1"
' bash "validation/kl_diagnosis/results/$source_job"
