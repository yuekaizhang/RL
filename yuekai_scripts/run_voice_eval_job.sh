#!/bin/bash
# In-container (ray.sub COMMAND) tau-voice half-duplex eval for a nano-omni ckpt.
# Env: CKPT (HF dir), TAG (unique run tag), DOMAIN (default airline).
# Agent on GPUs 0-3 (vLLM TP4),
# Chatterbox TTS on GPU 4, user-sim brain = luna via the NVIDIA gateway.
set -uo pipefail
EVAL=/lustre/fsw/portfolios/coreai/users/yuekaiz/pivot_rl/tau2_eval
V=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker
export HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/.cache/huggingface
mkdir -p $EVAL/voice_logs

# Container's ray venv sometimes ships an openai too old for the vllm CLI.
$V/bin/python -c "from openai.types.responses import NamespaceTool" 2>/dev/null || $V/bin/pip install -U openai

CUDA_VISIBLE_DEVICES=0,1,2,3 $V/bin/vllm serve "$CKPT" --served-model-name "$TAG" --port 8000 \
  --tensor-parallel-size 4 --max-model-len 65536 --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 \
  --mamba-ssm-cache-dtype float32 --trust-remote-code \
  --gpu-memory-utilization 0.9 > $EVAL/voice_logs/serve_${TAG}.log 2>&1 &
SERVE_PID=$!

# chatterbox_venv was built against a uv-managed CPython under /root (gone in
# batch containers). Keep a persistent 3.11 on lustre and run the server via
# PYTHONPATH into the venv's site-packages instead of the dangling venv python.
export UV_PYTHON_INSTALL_DIR=/lustre/fsw/portfolios/coreai/users/yuekaiz/tools/uv_pythons
PY311=$(ls $UV_PYTHON_INSTALL_DIR/cpython-3.11*/bin/python3.11 2>/dev/null | head -1)
if [ -z "$PY311" ]; then
  uv python install 3.11 || { echo "PY311 INSTALL FAILED"; exit 1; }
  PY311=$(ls $UV_PYTHON_INSTALL_DIR/cpython-3.11*/bin/python3.11 2>/dev/null | head -1)
fi
[ -n "$PY311" ] || { echo "NO PY311"; exit 1; }
cd $EVAL && CUDA_VISIBLE_DEVICES=4 PORT=8002 \
  PYTHONPATH=$EVAL/chatterbox_venv/lib/python3.11/site-packages \
  "$PY311" chatterbox_server.py > $EVAL/voice_logs/chatterbox_${TAG}.log 2>&1 &
TTS_PID=$!

for i in $(seq 1 120); do
  curl -s -m 2 localhost:8000/health >/dev/null && break
  kill -0 $SERVE_PID 2>/dev/null || { echo "VLLM SERVER DIED"; tail -30 $EVAL/voice_logs/serve_${TAG}.log; exit 1; }
  sleep 15
done
until curl -s -m 2 localhost:8002/health >/dev/null; do
  kill -0 $TTS_PID 2>/dev/null || { echo "CHATTERBOX DIED"; tail -30 $EVAL/voice_logs/chatterbox_${TAG}.log; exit 1; }
  sleep 10
done
echo "servers ready $(date)"

source /lustre/fsw/portfolios/coreai/users/yuekaiz/.bashrc_api
# Route the retail nl_assertions judge to the gateway (default gpt-4.1 is
# rejected by the gateway key; missing these lost exactly the 40 judge-scored
# retail tasks in every batch voice run before 2026-09-10).
export TAU2_NL_ASSERTIONS_MODEL="openai/switchyard/openai/gpt-5.6-luna"
export TAU2_NL_ASSERTIONS_ARGS="{\"base_url\":\"$OPENAI_BASE_URL\",\"api_key\":\"$OPENAI_API_KEY\"}"
DOMAIN=${DOMAIN:-airline}
cd $EVAL && venv/bin/python run_voice_halfduplex.py \
  --domain "$DOMAIN" --agent-model "$TAG" \
  --user-llm "openai/switchyard/openai/gpt-5.6-luna" \
  --user-llm-args "{\"temperature\":0.0,\"base_url\":\"$OPENAI_BASE_URL\",\"api_key\":\"$OPENAI_API_KEY\"}" \
  --agent-max-tokens 20480 \
  --workers 8 --save-to "voice_${TAG}_${DOMAIN}" 2>&1 | tail -20
rc=$?
[ -f "$EVAL/voice_results/voice_${TAG}_${DOMAIN}/summary.json" ] && { echo "VOICE_EVAL_OK $TAG $DOMAIN"; head -8 "$EVAL/voice_results/voice_${TAG}_${DOMAIN}/summary.json"; } || echo "VOICE_EVAL_MISSING_SUMMARY $TAG $DOMAIN"
exit $rc
