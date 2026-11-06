#!/bin/bash
# Source inside an allocated container. Keep credentials out of shell tracing.
set +x
export SC_REPO=/lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr4963
export PYTHONPATH="$SC_REPO:$SC_REPO/3rdparty/Automodel-workspace/Automodel:${PYTHONPATH:-}"
export UV_NO_SYNC=1 UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export PATH="/opt/nemo_rl_venv/bin:$PATH"
export NRL_IGNORE_VERSION_MISMATCH=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 NCCL_DEBUG=WARN VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/lustre/fsw/portfolios/coreai/users/yuekaiz/.cache/huggingface
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_API_KEY="$(cat /lustre/fsw/portfolios/coreai/users/yuekaiz/.wandb_nvidia)"
export WANDB_BASE_URL=https://api.wandb.ai WANDB_MODE="${SC_WANDB_MODE:-online}"
export WANDB_DIR="$SC_REPO/validation"
unset WANDB_SERVICE _WANDB_SERVICE
mkdir -p "$HF_DATASETS_CACHE"
cd "$SC_REPO"
