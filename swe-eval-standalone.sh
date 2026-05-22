#!/bin/bash
# Standalone SWE Evaluation Script
# Run swe-setup.sh first.
#
# Lightweight evaluation using AgentExecutionEngine + OpenAI-compatible API.
# No Ray, no verl, no FSDP — just a vLLM/SGLang server and ARL sandbox.
#
# Usage:
#   bash swe-eval-standalone.sh <model_name> [n_samples] [root_dir]
#
# ── Backend ──────────────────────────────────────────────────────────────────
#   BACKEND=vllm   (default) — local model served by vLLM.
#   BACKEND=sglang            — local model served by SGLang.
#
# Examples:
#   # Greedy eval (n=1)
#   bash swe-eval-standalone.sh Qwen3-8B
#
#   # pass@5 with sampling
#   bash swe-eval-standalone.sh Qwen3-8B 5
#
#   # SGLang backend
#   BACKEND=sglang bash swe-eval-standalone.sh Qwen3-8B
#
#   # Dry run: harness on unmodified code (no model needed)
#   DRY_RUN=true bash swe-eval-standalone.sh dummy
#
#   # Eval with standardized pytest output
#   NORMALIZE_PYTEST=true bash swe-eval-standalone.sh Qwen3-8B
#
# Server lifecycle:
#   The script auto-starts vLLM/SGLang if no server is already serving the
#   requested model, and leaves it running after eval finishes so it
#   can be reused across runs.


# ============ Arguments ============
MODEL_NAME=${1:?'Usage: bash swe-eval-standalone.sh <model_name> [n_samples] [root_dir]'}
ROOT_DIR=${2:-'/mnt/bn/trae-research-models/xujunjielong'}
EXPERIMENT_NAME="${EXPERIMENT_NAME:-agentic-swe-eval}"

export ARL_EXPERIMENT_ID="$EXPERIMENT_NAME"

bash utils/infra/clear_arl.sh

# ============ Feature Flags ============
DRY_RUN="${DRY_RUN:-false}"
NORMALIZE_PYTEST="${NORMALIZE_PYTEST:-false}"
MAX_TASKS="${MAX_TASKS:-}"
BACKEND="${BACKEND:-sglang}"
N_SAMPLES=${N_SAMPLES:-1}

if [ "$BACKEND" != "vllm" ] && [ "$BACKEND" != "sglang" ]; then
    echo "ERROR: BACKEND must be 'vllm' or 'sglang' (got: $BACKEND)" >&2
    return 1 2>/dev/null || exit 1
fi

# ============ Environment ============
export TOKENIZERS_PARALLELISM=true

# ============ Server Config ============
SERVER_PORT="${VLLM_PORT:-8000}"
SERVER_BASE_URL="${VLLM_BASE_URL:-http://localhost:${SERVER_PORT}/v1}"
SERVER_TP="${VLLM_TP:-8}"
MODEL_PATH="$ROOT_DIR/models/$MODEL_NAME"

# Match veRL's formula: max_model_len = prompt_length + response_length
# (see verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:195)
SERVER_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-40960}"  # 131072 + 32768
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Check if the correct backend+model is already serving on the port.
check_serving_model() {
    local response
    response=$(curl -sf "http://localhost:${SERVER_PORT}/v1/models" 2>/dev/null) || return 1
    # Check model matches
    echo "$response" | grep -qi "$MODEL_NAME" || return 1
    # Check backend matches (wrong backend serving the same model → need restart)
    if [ "$BACKEND" = "sglang" ]; then
        pgrep -f "sglang.*launch_server" >/dev/null 2>&1 || return 1
    else
        pgrep -f "vllm" >/dev/null 2>&1 || return 1
    fi
    return 0
}

# Kill any existing server processes on the target port (both vLLM and SGLang).
kill_existing_server() {
    # Use lsof to find the actual process holding the port
    local pids
    pids=$(lsof -ti :$SERVER_PORT 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "Killing existing server(s) on port $SERVER_PORT (pids: $pids)..."
        echo "$pids" | xargs kill 2>/dev/null
        sleep 3
        pids=$(lsof -ti :$SERVER_PORT 2>/dev/null)
        if [ -n "$pids" ]; then
            echo "$pids" | xargs kill -9 2>/dev/null
            sleep 1
        fi
    fi
}

start_server() {
    if [ "$BACKEND" = "sglang" ]; then
        SGLANG_DP="${SGLANG_DP:-1}"
        echo "Starting SGLang server: model=$MODEL_PATH tp=$SERVER_TP dp=$SGLANG_DP port=$SERVER_PORT"
        uv run --no-sync python -m sglang.launch_server \
            --model-path "$MODEL_PATH" \
            --port "$SERVER_PORT" \
            --tp "$SERVER_TP" \
            --dp "$SGLANG_DP" \
            --context-length "$SERVER_MAX_MODEL_LEN" \
            --dtype bfloat16 \
            &>"$OUTPUT_DIR/sglang.log" &
        local pid=$!
        local log_file="$OUTPUT_DIR/sglang.log"
    else
        echo "Starting vLLM server: model=$MODEL_PATH tp=$SERVER_TP port=$SERVER_PORT"
        VLLM_USE_V1=1 uv run --no-sync vllm serve "$MODEL_PATH" \
            --port "$SERVER_PORT" \
            --tensor-parallel-size "$SERVER_TP" \
            --max-model-len "$SERVER_MAX_MODEL_LEN" \
            &>"$OUTPUT_DIR/vllm.log" &
        local pid=$!
        local log_file="$OUTPUT_DIR/vllm.log"
    fi
    echo "$BACKEND server starting (pid $pid), waiting for it to be ready..."

    local max_wait=300
    local elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        if curl -sf "http://localhost:${SERVER_PORT}/v1/models" >/dev/null 2>&1; then
            echo "$BACKEND server ready (took ${elapsed}s)."
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: $BACKEND server exited unexpectedly. Check $log_file" >&2
            tail -20 "$log_file" >&2
            exit 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo "ERROR: $BACKEND server not ready after ${max_wait}s. Check $log_file" >&2
    tail -20 "$log_file" >&2
    exit 1
}

ensure_server() {
    if check_serving_model; then
        echo "$BACKEND already serving $MODEL_NAME on port $SERVER_PORT, reusing."
        return 0
    fi
    kill_existing_server
    start_server
}

# ============ Paths ============
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATA_FILE="${DATA_FILE:-data/swe/SWE_Bench_Verified.parquet}"

if [ "$DRY_RUN" = "true" ]; then
    OUTPUT_DIR="$ROOT_DIR/experiments/eval/dry_run"
else
    OUTPUT_DIR="$ROOT_DIR/experiments/eval/$MODEL_NAME-$BACKEND"
fi

mkdir -p "$OUTPUT_DIR"

# ============ Start server (skip for dry_run) ============
if [ "$DRY_RUN" != "true" ]; then
    ensure_server
fi

# ============ Build CLI args ============
EXTRA_ARGS=""

if [ "$DRY_RUN" = "true" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --dry_run"
else
    EXTRA_ARGS="$EXTRA_ARGS --model $MODEL_PATH --base_url $SERVER_BASE_URL"
fi

if [ "$NORMALIZE_PYTEST" = "true" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --normalize_pytest"
fi

if [ -n "$MAX_TASKS" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --max_tasks $MAX_TASKS"
fi

UPLOAD_URL="${UPLOAD_URL:-http://expdata.default.svc.cluster.local:8502}"
if [ "$UPLOAD" != "false" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --upload --upload_url $UPLOAD_URL"
fi

# ============ Run Evaluation ============
python3 "$SCRIPT_DIR/utils/eval/swe_eval_standalone.py" \
    --data "$DATA_FILE" \
    --n_samples "$N_SAMPLES" \
    --scaffold r2egym \
    --max_steps 100 \
    --max_prompt_length 131072 \
    --max_response_length 32768 \
    --trajectory_timeout 1200 \
    --n_parallel 64 \
    --output_dir "$OUTPUT_DIR" \
    $EXTRA_ARGS

# pkill -f "vllm\|sglang" # if this is the last run