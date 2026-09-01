#!/bin/bash
# Submit the TAU1 conversational tool-use pivot GRPO run (4 nodes x 8 GPUs) via sbatch ray.sub.
# Run from the LOGIN NODE (USER=yuekaiz):  bash yuekai_scripts/submit_tau1_4n8g.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMORL="${NEMORL:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# --- Per-run inputs ---
CONFIG_PATH="${CONFIG_PATH:-examples/nemo_gym/grpo_qwen3_30ba3b_thinking_tau1.yaml}"
NUM_NODES="${NUM_NODES:-4}"
JOB_NAME="${JOB_NAME:-tau1-pivot-4n8g}"
RESULTS_DIR="${RESULTS_DIR:-${NEMORL}/results/${JOB_NAME}}"
WANDB_PROJECT="${WANDB_PROJECT:-nemo-rl-tau}"
# Extra Hydra overrides, passed via env as a space-separated string, e.g.
#   EXTRA_OVERRIDES_STR="grpo.max_num_steps=400" bash yuekai_scripts/submit_tau1_4n8g.sh
EXTRA_OVERRIDES=( ${EXTRA_OVERRIDES_STR:-} )

# --- Container ---
# Must be new enough for the working tree's nemo_rl (vLLM >= 0.25.x): the 0724
# image fails with "ImportError: ServingTokenization" at vLLM server setup.
# nemo_rl.0803.sqsh (vLLM 0.25.1) verified for the qwen tau1 runs; 0821 chosen
# for the nano-omni runs.
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/coreai/users/yuekaiz/containers/nemo_rl.0821.sqsh}"

# --- Fixed Slurm defaults (override only if asked) ---
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-coreai_dlalgo_nemorl}"
SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
SBATCH_TIME="${SBATCH_TIME:-4:00:00}"
SBATCH_DEPENDENCY="${SBATCH_DEPENDENCY:-singleton}"

# --- Environment ---
export HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/.cache/huggingface}"
export TMPDIR="${TMPDIR:-/tmp/nrl-${USER:-yuekaiz}}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NRL_IGNORE_VERSION_MISMATCH="${NRL_IGNORE_VERSION_MISMATCH:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# --- Validate ---
[[ -f "${NEMORL}/ray.sub" ]]        || { echo "ray.sub missing under ${NEMORL}" >&2; exit 1; }
[[ -f "${NEMORL}/${CONFIG_PATH}" ]] || { echo "Config not found: ${CONFIG_PATH}" >&2; exit 1; }
[[ -f "${CONTAINER}" ]]             || { echo "Container missing: ${CONTAINER}" >&2; exit 1; }
[[ -f "${NEMORL}/datasets/conversational_tool_use_pivot/train.jsonl" ]] || {
    echo "Prepared data missing; run examples/nemo_gym/prepare_conversational_tool_use_pivot_data.py first" >&2
    exit 1
}

mkdir -p "${RESULTS_DIR}"

# --- Build the in-container COMMAND ---
# PYTHONPATH prefix is REQUIRED so this working tree wins over the editable
# /opt/nemo-rl install baked into the container: nemo_rl itself, the pinned Gym
# checkout (config_paths resolution), and the pinned Megatron-Bridge +
# Megatron-LM (the container's older megatron.bridge is incompatible with the
# repo's megatron.core).
export COMMAND="\
export PYTHONPATH=${NEMORL}:${NEMORL}/3rdparty/Gym-workspace/Gym:${NEMORL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${NEMORL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:\${PYTHONPATH:-} && \
export HF_HOME=${HF_HOME} && \
export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1 && \
export NCCL_DEBUG=${NCCL_DEBUG} && \
mkdir -p ${HF_HOME} ${TMPDIR} ${RESULTS_DIR} && \
uv run --no-sync examples/nemo_gym/run_grpo_nemo_gym.py --config ${CONFIG_PATH} \
    cluster.num_nodes=${NUM_NODES} \
    cluster.gpus_per_node=${GPUS_PER_NODE} \
    data.train.data_path='${NEMORL}/datasets/conversational_tool_use_pivot/train.jsonl' \
    data.validation.data_path='${NEMORL}/datasets/conversational_tool_use_pivot/val.jsonl' \
    checkpointing.checkpoint_dir='${RESULTS_DIR}/checkpoints' \
    logger.log_dir='${RESULTS_DIR}/logs' \
    logger.wandb.project='${WANDB_PROJECT}' \
    logger.wandb.name='${JOB_NAME}' \
    ${EXTRA_OVERRIDES[*]:-}"

cd "${NEMORL}"

MOUNTS="${MOUNTS:-/lustre:/lustre}" \
sbatch \
    --nodes="${NUM_NODES}" \
    --account="${SBATCH_ACCOUNT}" \
    --job-name="nemo-rl-${JOB_NAME}" \
    --partition="${SBATCH_PARTITION}" \
    --time="${SBATCH_TIME}" \
    --dependency="${SBATCH_DEPENDENCY}" \
    --gres="gpu:${GPUS_PER_NODE}" \
    --output="${RESULTS_DIR}/slurm-%j.out" \
    --error="${RESULTS_DIR}/slurm-%j.err" \
    ray.sub

echo
echo "Submitted: ${NUM_NODES} node x ${GPUS_PER_NODE} GPUs"
echo "Config:    ${CONFIG_PATH}"
echo "Container: ${CONTAINER}"
echo "Results:   ${RESULTS_DIR}"
