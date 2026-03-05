#!/bin/bash
# Standalone SWE Evaluation Script
#
# Lightweight evaluation using AgentExecutionEngine + OpenAI-compatible API.
# No Ray, no verl, no FSDP — just a vLLM server and ARL sandbox.
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
#
#   # Dry run: harness on unmodified code (no model needed)
#   DRY_RUN=true bash swe-eval-standalone.sh dummy
#
#   # Eval with standardized pytest output
#   NORMALIZE_PYTEST=true bash swe-eval-standalone.sh Qwen3-8B
#
# vLLM lifecycle:
#   The script auto-starts vLLM if no server is already serving the
#   requested model, and leaves it running after eval finishes so it
#   can be reused across runs.  To manually stop it:
#     kill $(pgrep -f "vllm serve")
#   Or to stop ALL vLLM processes:
#     pkill -f "vllm serve"

set -x

# ============ Arguments ============
MODEL_NAME=${1:?'Usage: bash swe-eval-standalone.sh <model_name> [n_samples] [root_dir]'}
N_SAMPLES=${2:-1}
ROOT_DIR=${3:-'/mnt/bn/trae-research-models/xujunjielong'}

MODEL_PATH="$ROOT_DIR/models/$MODEL_NAME"

# ============ Feature Flags ============
DRY_RUN="${DRY_RUN:-false}"
NORMALIZE_PYTEST="${NORMALIZE_PYTEST:-false}"
MAX_TASKS="${MAX_TASKS:-}"

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

# ============ vLLM Server ============
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${VLLM_PORT}/v1}"
VLLM_TP="${VLLM_TP:-8}"

# Check if vLLM is already serving the requested model.
# Returns 0 if the model is being served, 1 otherwise.
check_vllm_serving_model() {
    local response
    response=$(curl -sf "http://localhost:${VLLM_PORT}/v1/models" 2>/dev/null) || return 1
    echo "$response" | grep -q "$MODEL_PATH" && return 0
    # Also check by model name (vLLM may report just the name, not the full path)
    echo "$response" | grep -q "$MODEL_NAME" && return 0
    return 1
}

# Kill any existing vLLM processes on the target port.
kill_existing_vllm() {
    local pids
    pids=$(pgrep -f "vllm serve.*--port $VLLM_PORT" 2>/dev/null || pgrep -f "vllm.entrypoints.*--port $VLLM_PORT" 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "Killing existing vLLM server(s) on port $VLLM_PORT (pids: $pids)..."
        echo "$pids" | xargs kill 2>/dev/null
        sleep 3
        # Force kill survivors
        pids=$(pgrep -f "vllm serve.*--port $VLLM_PORT" 2>/dev/null || pgrep -f "vllm.entrypoints.*--port $VLLM_PORT" 2>/dev/null)
        if [ -n "$pids" ]; then
            echo "$pids" | xargs kill -9 2>/dev/null
            sleep 1
        fi
    fi
}

start_vllm() {
    echo "Starting vLLM server: model=$MODEL_PATH tp=$VLLM_TP port=$VLLM_PORT"
    VLLM_USE_V1=1 vllm serve "$MODEL_PATH" \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$VLLM_TP" \
        &>"$OUTPUT_DIR/vllm.log" &
    local pid=$!
    echo "vLLM server starting (pid $pid), waiting for it to be ready..."

    # Poll until the /models endpoint responds
    local max_wait=300
    local elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        if curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
            echo "vLLM server ready (took ${elapsed}s)."
            return 0
        fi
        # Check if process died
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: vLLM server exited unexpectedly. Check $OUTPUT_DIR/vllm.log" >&2
            tail -20 "$OUTPUT_DIR/vllm.log" >&2
            exit 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo "ERROR: vLLM server not ready after ${max_wait}s. Check $OUTPUT_DIR/vllm.log" >&2
    tail -20 "$OUTPUT_DIR/vllm.log" >&2
    exit 1
}

ensure_vllm() {
    if check_vllm_serving_model; then
        echo "vLLM already serving $MODEL_NAME on port $VLLM_PORT, reusing."
        return 0
    fi

    # Wrong model or no server — kill existing and start fresh
    kill_existing_vllm
    start_vllm
}

# ============ Paths ============
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATA_FILE="${DATA_FILE:-data/swe/SWE_Bench_Verified.parquet}"

if [ "$DRY_RUN" = "true" ]; then
    OUTPUT_DIR="$ROOT_DIR/experiments/eval/dry_run"
else
    OUTPUT_DIR="$ROOT_DIR/experiments/eval/$MODEL_NAME"
fi

mkdir -p "$OUTPUT_DIR"

# ============ Clean up stale sandboxes ============
# Previous crashed runs may leave pods in "allocated" state, blocking
# pool readiness.  Scale all warmpools to 0 and delete orphan sandboxes.
ARL_NAMESPACE="${ARL_NAMESPACE:-default}"
echo "Cleaning up stale ARL resources in namespace '$ARL_NAMESPACE'..."

# Scale all non-zero warmpools to 0
for pool in $(kubectl get warmpools -n "$ARL_NAMESPACE" -o jsonpath='{range .items[?(@.spec.replicas!=0)]}{.metadata.name}{"\n"}{end}' 2>/dev/null); do
    kubectl patch warmpool "$pool" -n "$ARL_NAMESPACE" --type=merge -p '{"spec":{"replicas":0}}' 2>/dev/null &
done
wait

# Delete all sandbox CRDs (cleans up allocated/orphan pods)
kubectl delete sandboxes --all -n "$ARL_NAMESPACE" --wait=false 2>/dev/null

# Wait for all managed pods to terminate before proceeding
echo "Waiting for all ARL pods to terminate..."
while kubectl get pods -l app.kubernetes.io/managed-by=arl -n "$ARL_NAMESPACE" --no-headers 2>/dev/null | grep -q .; do
    sleep 5
done
echo "Cleanup done."

# ============ Start vLLM (skip for dry_run) ============
if [ "$DRY_RUN" != "true" ]; then
    ensure_vllm
fi

# ============ Build CLI args ============
EXTRA_ARGS=""

if [ "$DRY_RUN" = "true" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --dry_run"
else
    EXTRA_ARGS="$EXTRA_ARGS --model $MODEL_PATH --base_url $VLLM_BASE_URL"
fi

if [ "$NORMALIZE_PYTEST" = "true" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --normalize_pytest"
fi

if [ -n "$MAX_TASKS" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --max_tasks $MAX_TASKS"
fi

uv pip install swebench==4.1.0

# ============ Run Evaluation ============
uv run --no-sync python3 "$SCRIPT_DIR/scripts/swe_eval_standalone.py" \
    --data "$DATA_FILE" \
    --n_samples "$N_SAMPLES" \
    --scaffold r2egym \
    --max_steps 100 \
    --max_prompt_length 131072 \
    --max_response_length 32768 \
    --trajectory_timeout 1800 \
    --n_parallel 50 \
    --output_dir "$OUTPUT_DIR" \
    $EXTRA_ARGS \
    > eval.log 2>&1
