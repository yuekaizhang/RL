#!/bin/bash
# Single-node in-container smoke test for the TAU1 conversational tool-use pivot recipe.
# Run from inside an interactive 8-GPU container allocation (USER=root), NOT the login node:
#   bash yuekai_scripts/run_tau1_smoke_1n8g.sh [extra hydra overrides...]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${REPO}/results/tau1-smoke-1n8g}"

# Working tree must win over the editable /opt/nemo-rl install baked into the
# container: nemo_rl itself, the pinned Gym checkout (config_paths resolution),
# and the pinned Megatron-Bridge + Megatron-LM (the container's older
# megatron.bridge is incompatible with the repo's megatron.core and crashes
# MegatronPolicyWorker with "isinstance() arg 2 must be a type").
export PYTHONPATH="${REPO}:${REPO}/3rdparty/Gym-workspace/Gym:${REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH:-}"

export HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NRL_IGNORE_VERSION_MISMATCH=1
export NCCL_DEBUG=WARN

mkdir -p "${RESULTS_DIR}"
cd "${REPO}"

uv run --no-sync examples/nemo_gym/run_grpo_nemo_gym.py \
    --config examples/nemo_gym/grpo_qwen3_30ba3b_thinking_tau1_smoke_1n8g.yaml \
    data.train.data_path="${REPO}/datasets/conversational_tool_use_pivot/train.jsonl" \
    data.validation.data_path="${REPO}/datasets/conversational_tool_use_pivot/val.jsonl" \
    logger.log_dir="${RESULTS_DIR}/logs" \
    "$@" \
    2>&1 | tee "${RESULTS_DIR}/run.log"
