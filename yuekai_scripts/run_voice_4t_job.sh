#!/bin/bash
# In-container (ray.sub COMMAND) tau-voice 4-trial shard job, running the
# harness from the tau2-bench-yuekai fork (branch voice-halfduplex) so results
# are reproducible from the published code.
# Env: CKPT (HF dir), TAG (unique), DOMAIN (airline|retail|telecom),
#      SHARD (k/n), NUM_TRIALS (default 4).
set -uo pipefail
EVAL=/lustre/fsw/portfolios/coreai/users/yuekaiz/pivot_rl/tau2_eval
FORK=/lustre/fsw/portfolios/coreai/users/yuekaiz/pivot_rl/tau2-bench-yuekai
V=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker
export HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/.cache/huggingface
NUM_TRIALS=${NUM_TRIALS:-4}
SHARD_TAG=$(echo "${SHARD}" | tr '/' 'o')
RUN_NAME="voice4t_${TAG}_${DOMAIN}_s${SHARD_TAG}"
mkdir -p $EVAL/voice_logs

$V/bin/python -c "from openai.types.responses import NamespaceTool" 2>/dev/null || $V/bin/pip install -U openai

CUDA_VISIBLE_DEVICES=0,1,2,3 $V/bin/vllm serve "$CKPT" --served-model-name "$TAG" --port 8000 \
  --tensor-parallel-size 4 --max-model-len 65536 --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 \
  --mamba-ssm-cache-dtype float32 --trust-remote-code \
  --gpu-memory-utilization 0.9 ${EXTRA_VLLM_FLAGS:-} > $EVAL/voice_logs/serve_${RUN_NAME}.log 2>&1 &
SERVE_PID=$!

export UV_PYTHON_INSTALL_DIR=/lustre/fsw/portfolios/coreai/users/yuekaiz/tools/uv_pythons
PY311=$(ls $UV_PYTHON_INSTALL_DIR/cpython-3.11*/bin/python3.11 2>/dev/null | head -1)
[ -n "$PY311" ] || { echo "NO PY311"; exit 1; }
cd $EVAL && CUDA_VISIBLE_DEVICES=4 PORT=8002 \
  PYTHONPATH=$EVAL/chatterbox_venv/lib/python3.11/site-packages \
  "$PY311" chatterbox_server.py > $EVAL/voice_logs/chatterbox_${RUN_NAME}.log 2>&1 &
TTS_PID=$!

for i in $(seq 1 120); do
  curl -s -m 2 localhost:8000/health >/dev/null && break
  kill -0 $SERVE_PID 2>/dev/null || { echo "VLLM SERVER DIED"; tail -30 $EVAL/voice_logs/serve_${RUN_NAME}.log; exit 1; }
  sleep 15
done
until curl -s -m 2 localhost:8002/health >/dev/null; do
  kill -0 $TTS_PID 2>/dev/null || { echo "CHATTERBOX DIED"; tail -30 $EVAL/voice_logs/chatterbox_${RUN_NAME}.log; exit 1; }
  sleep 10
done
echo "servers ready $(date)"

source /lustre/fsw/portfolios/coreai/users/yuekaiz/.bashrc_api
export TAU2_NL_ASSERTIONS_MODEL="openai/switchyard/openai/gpt-5.6-luna"
export TAU2_NL_ASSERTIONS_ARGS="{\"base_url\":\"$OPENAI_BASE_URL\",\"api_key\":\"$OPENAI_API_KEY\"}"

cd $EVAL && venv/bin/python $FORK/voice_halfduplex/run_voice_halfduplex.py \
  --domain "$DOMAIN" --agent-model "$TAG" \
  --user-llm "openai/switchyard/openai/gpt-5.6-luna" \
  --user-llm-args "{\"temperature\":0.0,\"base_url\":\"$OPENAI_BASE_URL\",\"api_key\":\"$OPENAI_API_KEY\"}" \
  --agent-max-tokens 20480 --num-trials "$NUM_TRIALS" --task-shard "$SHARD" \
  --workers 8 --save-to "$RUN_NAME" 2>&1 | tail -20
rc=$?
S=$FORK/voice_halfduplex/voice_results/$RUN_NAME/summary.json
[ -f "$S" ] && { echo "VOICE4T_OK $RUN_NAME"; head -8 "$S"; } || echo "VOICE4T_MISSING_SUMMARY $RUN_NAME"
exit $rc
