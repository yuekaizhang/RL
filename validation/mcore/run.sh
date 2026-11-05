#!/bin/bash
set -euo pipefail
source /lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr4963/validation/mcore/environment.sh
[[ "$USER" == root && -n "${SLURM_JOB_ID:-}" ]]
uv run --no-project --python /opt/nemo_rl_venv/bin/python python validation/mcore/run.py
uv run --no-project --python /opt/nemo_rl_venv/bin/python python validation/mcore/plot.py
