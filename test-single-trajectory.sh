#!/bin/bash
# Quick single-case test to verify chat_completions are saved correctly.
# Usage:
#   bash test-single-trajectory.sh <model_name> [root_dir]
#
# Assumes vLLM is already running.

set -ex

MODEL_NAME=${1:?'Usage: bash test-single-trajectory.sh <model_name> [root_dir]'}
ROOT_DIR=${2:-'/mnt/bn/trae-research-models/xujunjielong'}
MODEL_PATH="$ROOT_DIR/models/$MODEL_NAME"

# ============ Environment ============
export UV_INDEX_URL=https://bytedpypi.byted.org/simple/
export HF_ENDPOINT=https://hf-mirror.com
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"
export ARL_GATEWAY_URL="${ARL_GATEWAY_URL:-http://118.145.210.10:8080}"
export TOKENIZERS_PARALLELISM=true

VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${VLLM_PORT}/v1}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUTPUT_DIR="/tmp/test_single_trajectory"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# ============ Reinstall rllm in editable mode to pick up source changes ============
uv pip install -e .

# ============ Run 1 task only ============
uv run --no-sync python3 "$SCRIPT_DIR/scripts/swe_eval_standalone.py" \
    --data data/swe/SWE_Bench_Verified.parquet \
    --max_tasks 1 \
    --n_samples 1 \
    --scaffold r2egym \
    --max_steps 5 \
    --max_prompt_length 131072 \
    --max_response_length 32768 \
    --trajectory_timeout 600 \
    --n_parallel 1 \
    --retry_limit 1 \
    --output_dir "$OUTPUT_DIR" \
    --model "$MODEL_PATH" \
    --base_url "$VLLM_BASE_URL" \
    2>&1 | tee "$OUTPUT_DIR/run.log"

# ============ Verify output ============
CHAT_FILE="$OUTPUT_DIR/chat_completions/eval.jsonl"
echo ""
echo "========== Verification =========="
if [ ! -f "$CHAT_FILE" ]; then
    echo "FAIL: $CHAT_FILE not found"
    exit 1
fi

LINE_COUNT=$(wc -l < "$CHAT_FILE")
EMPTY_COUNT=$(grep -c '^\[\]$' "$CHAT_FILE" || true)
echo "Total lines:  $LINE_COUNT"
echo "Empty lines:  $EMPTY_COUNT"

if [ "$EMPTY_COUNT" -eq "$LINE_COUNT" ]; then
    echo "FAIL: All chat_completions are empty []"
    echo ""
    echo "Debug: checking if source code has the fix..."
    uv run --no-sync python3 -c "
import inspect, rllm.engine.agent_execution_engine as m
src = inspect.getsource(m.AgentExecutionEngine.execute_tasks)
print('chat_completions in execute_tasks:', 'chat_completions' in src)
print()
# Show the relevant lines
for i, line in enumerate(src.splitlines()):
    if 'chat_completions' in line or 'res.info' in line:
        print(f'  {i}: {line}')
"
    exit 1
else
    echo "OK: chat_completions have content"
    echo ""
    echo "First trajectory preview (first 500 chars):"
    head -1 "$CHAT_FILE" | cut -c1-500
fi
