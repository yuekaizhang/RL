#!/bin/bash
set -euo pipefail
[[ "$USER" == yuekaiz ]]
if [[ "${SC_LP_BATCH_INVARIANT:-0}" == 1 ]]; then
  printf '%s\n' '1030 GDN_ATTN rejects batch-invariant mode; see failed21385207. Choose a supported control.' >&2
  exit 1
fi
cd /lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr4963
# Only this investigation's jobs count; other user workstreams stay untouched.
if squeue -u "$USER" -h -o '%j' | rg -q '^yuekaiz-kl-diagnosis-ep16'; then
  printf '%s\n' 'An EP16 diagnosis is already queued/running; refusing duplicate.' >&2
  exit 1
fi
export CONTAINER=/lustre/fsw/portfolios/coreai/users/yuekaiz/containers/nemo_rl.1030.sqsh
[[ -f "$CONTAINER" ]]
export MOUNTS=/lustre:/lustre
export BASE_LOG_DIR="$PWD/validation/kl_diagnosis/slurm"
export GPUS_PER_NODE=8 CPUS_PER_WORKER=128 OMP_NUM_THREADS=4
export ENROOT_ROOTFS_WRITABLE=1
export COMMAND="bash '$PWD/validation/kl_diagnosis/run.sh'"
dependency=()
if [[ -n "${1:-}" ]]; then dependency+=(--dependency="afterok:$1"); fi
job_name=yuekaiz-kl-diagnosis-ep16-r8
description='EP16 r8: A_log repeats, FlashInfer/Triton prefill and state precision controls'
if [[ "${SC_LP_SCHEDULING_AB:-0}" == 1 ]]; then
  job_name=yuekaiz-kl-diagnosis-ep16-r9-scheduling
  description='EP16 r9: fixed FP32-A_log weights/tokens; serial1, microbatch4, concurrent32 repeatability; two updates'
  export COMMAND="SC_LP_SCHEDULING_AB=1 bash '$PWD/validation/kl_diagnosis/run.sh'"
fi
if [[ "${SC_LP_PREFIX_CACHE_AB:-0}" == 1 ]]; then
  job_name=yuekaiz-kl-diagnosis-ep16-r13-noprefix
  description='EP16 r13: prefix flag in typed vllm_cfg; real worker kwargs assembly preflight; EXACT r9 tokens/policy/checkpoint600; two updates'
  export COMMAND="SC_LP_SCHEDULING_AB=1 SC_LP_PREFIX_CACHE_AB=1 bash '$PWD/validation/kl_diagnosis/run.sh'"
fi
job=$(sbatch --parsable "${dependency[@]}" --nodes=3 --ntasks-per-node=1 --cpus-per-task=128 \
  --gres=gpu:8 --exclusive --account=coreai_dlalgo_nemorl --partition=batch \
  --time=00:30:00 --job-name="$job_name" \
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"30","reason":"data_loading","description":"EP16 checkpoint600 fixed-token KL diagnosis, two updates"}}' \
  --output="$BASE_LOG_DIR/slurm-%j.out" --error="$BASE_LOG_DIR/slurm-%j.err" ray.sub)
printf '%s\t%s\n' "$job" "$description" | tee -a validation/kl_diagnosis/submitted.tsv
