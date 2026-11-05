#!/bin/bash
# Source inside the requested image and a Slurm allocation.
source /lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr4963/validation/scripts/environment.sh
export SC_BRIDGE="$SC_REPO/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
export PYTHONPATH="$SC_BRIDGE/src:$SC_BRIDGE/3rdparty/Megatron-LM:$PYTHONPATH"
export NRL_MEGATRON_CHECKPOINT_DIR="$SC_REPO/validation/mcore/cache"
mkdir -p "$NRL_MEGATRON_CHECKPOINT_DIR"
