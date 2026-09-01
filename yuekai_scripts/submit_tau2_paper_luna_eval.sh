#!/bin/bash
# Submit the paper-tau_rl luna-user-sim tau2 eval via ray.sub (login node).
# Two 1n8g jobs: (airline+telecom) and (retail). ray.sub's srun flags
# (--no-container-mount-home) keep the image's /root/.cache/uv visible, which
# the /opt/ray_venvs vLLM venv needs — do NOT hand-roll sbatch for this.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL=/lustre/fsw/portfolios/coreai/users/yuekaiz/pivot_rl/tau2_eval
CONTAINER=/lustre/fsw/portfolios/coreai/users/yuekaiz/containers/nemo_rl.0803.sqsh
RESULTS_DIR=$EVAL/slurm_logs
mkdir -p "$RESULTS_DIR"
cd "$REPO"

# Optional env: TAG + CKPT select the agent under test (default = paper tau_rl).
TAG=${TAG:-paper_tau_rl_luna}
CKPT=${CKPT:-/lustre/fsw/portfolios/llmservice/users/jkyi/checkpoints/pivot_rl/tau_rl}

submit_one() {
  local name=$1; shift
  local domains="$*"
  # All jobs are batch/4h (cluster policy). Time budgeting (measured, luna
  # protocol): airline ~1h, retail ~1.5h, telecom 2.5-5h. Telecom must be
  # submitted ALONE, and for slow ckpts split via TASK_IDS halves (see
  # $EVAL/retail_nl_assertion_task_ids.txt pattern / skill notes) so each
  # half fits the 4h window — same per-task 4-trial protocol after merging.
  CONTAINER=$CONTAINER \
  MOUNTS="/lustre:/lustre" \
  COMMAND="TAG=$TAG CKPT=$CKPT TOOL_PARSER=${TOOL_PARSER:-hermes} REASONING_PARSER=${REASONING_PARSER:-qwen3} EXTRA_VLLM_FLAGS='${EXTRA_VLLM_FLAGS:-}' AGENT_LLM_ARGS='${AGENT_LLM_ARGS:-}' TASK_IDS='${TASK_IDS:-}' TASK_IDS_FILE='${TASK_IDS_FILE:-}' bash $EVAL/run_paper_luna_eval.sh $domains" \
  sbatch \
    --nodes=1 \
    --account=coreai_dlalgo_nemorl \
    --job-name="tau2-$TAG-$name${JOB_SUFFIX:-}" \
    --partition=batch \
    --time=4:00:00 \
    --gres=gpu:8 \
    --exclusive \
    --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"data_loading","description":"tau2-bench eval: agent server on GPUs 0-3, GPUs 4-7 intentionally idle (user sim is API-based)"}}' \
    --output="$RESULTS_DIR/slurm-%j.out" \
    --error="$RESULTS_DIR/slurm-%j.err" \
    ray.sub
}

# DOMAINS override: e.g. DOMAINS="telecom" submits ONE job for just that
# domain (name suffix = domain initials). Default: the standard two-job split.
if [ -n "${DOMAINS:-}" ]; then
  suffix=$(echo "$DOMAINS" | tr ' ' '\n' | cut -c1 | tr -d '\n')
  # shellcheck disable=SC2086
  submit_one "$suffix" $DOMAINS
else
  submit_one at airline telecom
  submit_one rt retail
fi
