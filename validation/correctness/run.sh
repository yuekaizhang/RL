#!/bin/bash
# Execute inside an allocated nemo_rl.1030 container with at least two GPUs.
set -euo pipefail
source /lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr4963/validation/scripts/environment.sh
[[ "$USER" == root && -n "${SLURM_JOB_ID:-}" ]]
export SC_AUDIT_DIR="${SC_AUDIT_DIR:-$SC_REPO/validation/correctness/real}"
[[ ! -e "$SC_AUDIT_DIR/events.jsonl" ]] || { echo 'Use a fresh SC_AUDIT_DIR'; exit 1; }
mkdir -p "$SC_AUDIT_DIR"
run_python=(uv run --no-project --python /opt/nemo_rl_venv/bin/python python)
command=("${run_python[@]}" validation/correctness/run_audited.py
  --config validation/correctness/real_la1.yaml
  "checkpointing.checkpoint_dir=$SC_AUDIT_DIR/ckpts"
  "logger.log_dir=$SC_AUDIT_DIR/logs")
"${command[@]}" > "$SC_AUDIT_DIR/first.log" 2>&1
"${command[@]}" checkpointing.checkpoint_must_save_by=null > "$SC_AUDIT_DIR/resume.log" 2>&1
"${run_python[@]}" validation/correctness/analyze.py "$SC_AUDIT_DIR"
"${run_python[@]}" validation/correctness/check_resident.py "$SC_AUDIT_DIR" 6
