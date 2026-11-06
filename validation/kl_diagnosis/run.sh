#!/bin/bash
set -euo pipefail
source /lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr4963/validation/scripts/environment.sh
[[ "$USER" == root && -n "${SLURM_JOB_ID:-}" ]]
for attempt in $(seq 1 120); do
  [[ -f validation/kl_diagnosis/READY ]] && break
  sleep 5
done
[[ -f validation/kl_diagnosis/READY ]]
export SC_LP_DIAG_DIR="$SC_REPO/validation/kl_diagnosis/results/$SLURM_JOB_ID"
[[ "${SC_LP_BATCH_INVARIANT:-0}" != 1 ]]
config_file=validation/kl_diagnosis/ep16.yaml
if [[ "${SC_LP_PREFIX_CACHE_AB:-0}" == 1 ]]; then
  [[ "${SC_LP_SCHEDULING_AB:-0}" == 1 ]]
  export VLLM_BATCH_INVARIANT=0
  export SC_LP_CONTROL=prefix_cache_disabled
  export SC_LP_REFERENCE_DIR="$SC_REPO/validation/kl_diagnosis/results/21384522"
  [[ -f "$SC_LP_REFERENCE_DIR/scheduling-repeats.pt" ]]
  # Declare new struct keys in YAML rather than overriding an empty kwargs map.
  config_file=validation/kl_diagnosis/ep16_noprefix.yaml
fi
if [[ "${SC_LP_SCHEDULING_AB:-0}" == 1 ]]; then
  export SC_LP_FORCE_LEGACY_A_LOG=0 SC_LP_KERNEL_AB=0
else
  export SC_LP_FORCE_LEGACY_A_LOG=1 SC_LP_KERNEL_AB=1
fi
mkdir -p "$SC_LP_DIAG_DIR"
/opt/nemo_rl_venv/bin/python validation/kl_diagnosis/check_recipe.py "$config_file"
/opt/ray_venvs/nemo_rl.models.policy.workers.automodel_policy_worker.AutomodelPolicyWorker/bin/python validation/kl_diagnosis/check_sampling.py
/opt/ray_venvs/nemo_rl.models.policy.workers.automodel_policy_worker.AutomodelPolicyWorker/bin/python -c 'import validation.kl_diagnosis.policy_worker; print("POLICY_DIAGNOSTIC_IMPORT_OK", flush=True)'
/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python -c 'import validation.kl_diagnosis.generation_worker; print("GENERATION_DIAGNOSTIC_IMPORT_OK", flush=True)'
/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python validation/kl_diagnosis/check_generation_controls.py
if [[ "${SC_LP_PREFIX_CACHE_AB:-0}" == 1 ]]; then
  /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python validation/kl_diagnosis/check_worker_kwargs.py
fi
uv run --no-project --python /opt/nemo_rl_venv/bin/python python validation/kl_diagnosis/run_diagnosis.py --config "$config_file" "logger.log_dir=$SC_LP_DIAG_DIR/logs" "checkpointing.checkpoint_dir=$SC_LP_DIAG_DIR/ckpts"
if [[ "${SC_LP_SCHEDULING_AB:-0}" != 1 ]]; then
  uv run --no-project --python /opt/nemo_rl_venv/bin/python python validation/kl_diagnosis/analyze.py "$SC_LP_DIAG_DIR"
  /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python validation/kl_diagnosis/analyze_gdn_state.py "$SC_LP_DIAG_DIR"
fi
