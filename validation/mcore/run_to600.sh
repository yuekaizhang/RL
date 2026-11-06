#!/bin/bash
set -euo pipefail
source /lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr4963/validation/mcore/environment.sh
[[ "$USER" == root && -n "${SLURM_JOB_ID:-}" ]]
/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python validation/mcore/check_scheduler_to600.py
uv run --no-project --python /opt/nemo_rl_venv/bin/python python validation/mcore/run_to600.py
uv run --no-project --python /opt/nemo_rl_venv/bin/python python validation/mcore/plot.py --target-steps 600
