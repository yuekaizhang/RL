#!/bin/bash
# In-container (ray.sub COMMAND) conversion of nano-omni Megatron ckpts to HF.
# Usage: bash convert_nano_ckpts_job.sh <step> [<step> ...]
#   steps resolve under $CKPT_ROOT (override via env, default nano-omni main line);
#   output name: $OUT_PREFIX_step<step>_hf (OUT_PREFIX default nano_omni)
set -uo pipefail
REPO=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/pivot_rl/RL
EVAL=/lustre/fsw/portfolios/coreai/users/yuekaiz/pivot_rl/tau2_eval
M=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
export PYTHONPATH="${REPO}:${REPO}/3rdparty/Gym-workspace/Gym:${REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH:-}"
export HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NRL_IGNORE_VERSION_MISMATCH=1

CKPT_ROOT=${CKPT_ROOT:-/lustre/fsw/portfolios/coreai/users/yuekaiz/pivot_rl/RL/results/tau1-nano-omni-4n8g/checkpoints}
OUT_PREFIX=${OUT_PREFIX:-nano_omni}
rc=0
for step in "$@"; do
  SRC=$CKPT_ROOT/step_$step
  OUT=$EVAL/ckpts/${OUT_PREFIX}_step${step}_hf
  echo "=== convert step_$step -> $OUT $(date) ==="
  # Frozen vision/sound towers may be absent from the Megatron ckpt; retry
  # with --no-strict if the strict pass fails.
  $M $REPO/examples/converters/convert_megatron_to_hf.py \
    --config $SRC/config.yaml \
    --megatron-ckpt-path $SRC/policy/weights/iter_0000000 \
    --hf-ckpt-path $OUT 2>&1 | tail -3 \
  || $M $REPO/examples/converters/convert_megatron_to_hf.py \
    --config $SRC/config.yaml \
    --megatron-ckpt-path $SRC/policy/weights/iter_0000000 \
    --hf-ckpt-path $OUT --no-strict 2>&1 | tail -3
  [ -f $OUT/config.json ] && echo "${OUT_PREFIX}_step_$step CONVERT_OK" || { echo "${OUT_PREFIX}_step_$step CONVERT_FAILED"; rc=1; }
done
exit $rc
