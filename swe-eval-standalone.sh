#!/bin/bash
# Standalone SWE Evaluation Script
#
# Lightweight evaluation using AgentExecutionEngine + OpenAI-compatible API.
# No Ray, no verl, no FSDP — just a vLLM server and ARL sandbox.
#
# Prerequisites:
#   1. vLLM server running (e.g., vllm serve <model> --port 8000)
#   2. ARL gateway accessible (ARL_GATEWAY_URL env var)
#
# Usage:
#   bash swe-eval-standalone.sh <model_name> [n_samples] [root_dir]
#
# Examples:
#   # Greedy eval (n=1)
#   bash swe-eval-standalone.sh Qwen3-8B
#
#   # pass@5 with sampling
#   bash swe-eval-standalone.sh Qwen3-8B 5

set -x

# ============ Arguments ============
MODEL_NAME=${1:?'Usage: bash swe-eval-standalone.sh <model_name> [n_samples] [root_dir]'}
N_SAMPLES=${2:-1}
ROOT_DIR=${3:-'/mnt/bn/trae-research-models/xujunjielong'}

MODEL_PATH="$ROOT_DIR/models/$MODEL_NAME"

# ============ Environment ============
export UV_INDEX_URL=https://bytedpypi.byted.org/simple/
export HF_ENDPOINT=https://hf-mirror.com

export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118

export ARL_GATEWAY_URL="${ARL_GATEWAY_URL:-http://14.103.184.145:8080}"
export TOKENIZERS_PARALLELISM=true

# ============ vLLM Server ============
# The vLLM server should be started separately, e.g.:
#   VLLM_USE_V1=1 vllm serve $MODEL_PATH --port 8000 --tensor-parallel-size 8
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"

# ============ Paths ============
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATA_FILE="${DATA_FILE:-data/swe/SWE_Bench_Verified.parquet}"
OUTPUT_DIR="$ROOT_DIR/experiments/eval/$MODEL_NAME"

# ============ Run Evaluation ============
uv run --no-sync python3 "$SCRIPT_DIR/scripts/swe_eval_standalone.py" \
    --model "$MODEL_PATH" \
    --base_url "$VLLM_BASE_URL" \
    --data "$DATA_FILE" \
    --n_samples "$N_SAMPLES" \
    --scaffold r2egym \
    --max_steps 100 \
    --max_prompt_length 131072 \
    --max_response_length 32768 \
    --trajectory_timeout 1200 \
    --n_parallel 48 \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/eval.log"
