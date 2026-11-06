#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --account=coreai_dlalgo_nemorl
#SBATCH --partition=batch
#SBATCH --time=00:06:00
set -euo pipefail
repo=/lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr4963
srun --container-image=/lustre/fsw/portfolios/coreai/users/yuekaiz/containers/nemo_rl.1030.sqsh \
  --container-mounts=/lustre:/lustre --container-workdir="$repo" bash -c '
set -euo pipefail
source validation/mcore/environment.sh
[[ "$USER" == root && -n "${SLURM_JOB_ID:-}" ]]
export SC_LP_DIAG_DIR="$SC_REPO/validation/kl_diagnosis/results/recovery-preflight-$SLURM_JOB_ID"
export SC_LP_CONTROL=prefix_cache_disabled VLLM_BATCH_INVARIANT=0
mkdir -p "$SC_LP_DIAG_DIR"
/opt/nemo_rl_venv/bin/python validation/kl_diagnosis/check_recipe.py validation/kl_diagnosis/ep16_noprefix.yaml
/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python validation/kl_diagnosis/check_generation_controls.py
for lag in 0 1; do
  SC_MCORE_LOOKAHEAD="$lag" /opt/nemo_rl_venv/bin/python validation/mcore/run_to600.py --preflight
done
printf "RECOVERY_PREFLIGHT_ALL_PASSED job=%s\n" "$SLURM_JOB_ID"
'
