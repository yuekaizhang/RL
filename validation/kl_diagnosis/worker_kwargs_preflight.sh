#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --account=coreai_dlalgo_nemorl
#SBATCH --partition=batch
#SBATCH --time=00:08:00
set -euo pipefail
repo=/lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr4963
srun --container-image=/lustre/fsw/portfolios/coreai/users/yuekaiz/containers/nemo_rl.1030.sqsh \
  --container-mounts=/lustre:/lustre --container-workdir="$repo" bash -c '
set -euo pipefail
source validation/scripts/environment.sh
[[ "$USER" == root && -n "${SLURM_JOB_ID:-}" ]]
export SC_LP_DIAG_DIR="$SC_REPO/validation/kl_diagnosis/results/worker-preflight-$SLURM_JOB_ID"
export SC_LP_CONTROL=prefix_cache_disabled VLLM_BATCH_INVARIANT=0
mkdir -p "$SC_LP_DIAG_DIR"
gen_python=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python
/opt/nemo_rl_venv/bin/python validation/kl_diagnosis/check_recipe.py validation/kl_diagnosis/ep16_noprefix.yaml
"$gen_python" validation/kl_diagnosis/check_worker_kwargs.py
test_deps="/tmp/yuekaiz-kl-kwargs-pytest-$SLURM_JOB_ID"
uv --no-config pip install --python "$gen_python" --target "$test_deps" "pytest==8.4.2"
export PYTHONPATH="$test_deps:$PYTHONPATH"
"$gen_python" -m pytest -o addopts="" -q -s \
  tests/unit/models/generation/test_vllm_config.py::test_prefix_cache_config_survives_typed_validation \
  tests/unit/models/generation/test_vllm_generation.py::test_resolve_enable_prefix_caching_respects_explicit_config \
  tests/unit/models/generation/test_vllm_generation.py::test_resolve_enable_prefix_caching_uses_cuda_capability_for_auto \
  --junitxml="$SC_LP_DIAG_DIR/prefix-config-tests.xml"
printf "WORKER_PREFLIGHT_ALL_PASSED job=%s\n" "$SLURM_JOB_ID"
'
