#!/bin/bash
# SWE Trajectory Collection Script (Rejection / DPO Sampling)
# Collect SWE trajectories via a closed-source API or a local vLLM server.
# Run swe-setup.sh first (ARL sandboxes must be reachable).
#
# Usage:
#   bash swe-data-collection.sh <model_name> [root_dir]
#
# ── Backend ──────────────────────────────────────────────────────────────────
#   BACKEND=openai (default) — closed-source model via OpenAI-compatible API.
#                               Requires API_KEY / OPENAI_API_KEY.
#   BACKEND=vllm              — local model served by vLLM.
#                               The script auto-starts vLLM when no server is
#                               already up, then leaves it running afterwards.
#   BACKEND=sglang            — local model served by SGLang.
#                               Same auto-start behaviour as vLLM.
#
# ── Mode ─────────────────────────────────────────────────────────────────────
#   MODE=rejection (default) — keep only passing trajectories (SFT data).
#                              Output: parquet with single "messages" column.
#   MODE=dpo                 — cache pass+fail trajectories, emit pairs.
#                              Output: parquet with chosen/rejected/instance_id.
#
# ── Disable thinking (DISABLE_THINKING) ─────────────────────────────────────
#   Skip the model's <think> reasoning phase for faster rollout.  Injects an
#   empty <think></think> prefix so the model outputs actions directly.
#   Only effective with local backends (vllm/sglang).
#
#   DISABLE_THINKING=false (default)
#   DISABLE_THINKING=true  — disable thinking mode
#
# ── Examples ─────────────────────────────────────────────────────────────────
#   # API rejection sampling (GPT-4o)
#   bash swe-data-collection.sh gpt-4o
#
#   # API DPO sampling
#   MODE=dpo bash swe-data-collection.sh gpt-4o
#
#   # Local vLLM rejection sampling
#   BACKEND=vllm bash swe-data-collection.sh Qwen3-Coder-Next
#
#   # Local SGLang, thinking disabled (faster rollout)
#   DISABLE_THINKING=true BACKEND=sglang bash swe-data-collection.sh Qwen3-Coder-Next
#
#   # Local SGLang rejection sampling
#   BACKEND=sglang bash swe-data-collection.sh Qwen3-Coder-Next
#
#   # Local vLLM DPO sampling with custom root
#   BACKEND=vllm MODE=dpo bash swe-data-collection.sh Qwen3-Coder-Next /mnt/bn/my-bucket
#
#   # Resume an interrupted run (re-use the same OUTPUT_FILE)
#   BACKEND=vllm bash swe-data-collection.sh Qwen3-Coder-Next
#
#   # Custom parallelism / target for DPO
#   MODE=dpo BACKEND=vllm TARGET=2000 N_PARALLEL=64 bash swe-data-collection.sh Qwen3-Coder-Next

# ============ Arguments ============
MODEL_NAME=${1:?'Usage: bash swe-data-collection.sh <model_name> [root_dir]'}
ROOT_DIR=${2:-'/mnt/bn/trae-research-models-lq/xujunjielong'}
EXPERIMENT_NAME="${EXPERIMENT_NAME:-swe-data-collection}"

export ARL_EXPERIMENT_ID="$EXPERIMENT_NAME"

bash utils/infra/clear_arl.sh

# ============ Mode & Backend ============
MODE="${MODE:-rejection}"
BACKEND="${BACKEND:-openai}"

if [ "$MODE" != "rejection" ] && [ "$MODE" != "dpo" ]; then
    echo "ERROR: MODE must be 'rejection' or 'dpo' (got: $MODE)" >&2
    return 1 2>/dev/null || exit 1
fi

if [ "$BACKEND" != "openai" ] && [ "$BACKEND" != "vllm" ] && [ "$BACKEND" != "sglang" ]; then
    echo "ERROR: BACKEND must be 'openai', 'vllm', or 'sglang' (got: $BACKEND)" >&2
    return 1 2>/dev/null || exit 1
fi

# ============ Sampling config ============
TARGET="${TARGET:-10000}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_STEPS="${MAX_STEPS:-25}"
SCAFFOLD="${SCAFFOLD:-r2egym}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-32768}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-131072}"
N_PARALLEL="${N_PARALLEL:-32}"
TRAJECTORY_TIMEOUT="${TRAJECTORY_TIMEOUT:-3600}"
STEP_TIMEOUT="${STEP_TIMEOUT:-90}"
REWARD_TIMEOUT="${REWARD_TIMEOUT:-300}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
DISABLE_THINKING="${DISABLE_THINKING:-false}"

# ============ Paths ============
DATA_FILE="${DATA_FILE:-data/swe/R2E_Gym_Subset.parquet}"
OUTPUT_DIR="${ROOT_DIR}/data/swe/collected-${MODE}-${MAX_STEPS}-${MAX_RESPONSE_LENGTH}"
_SAFE_MODEL="${MODEL_NAME//\//__}"
# Compute default OUTPUT_FILE from MODE — but respect an explicit user override.
# Guard against stale values from a prior `source` invocation by checking whether
# the current value still matches the *other* mode's default pattern.
if [ "$MODE" = "dpo" ]; then
    _DEFAULT_OUTPUT="${OUTPUT_DIR}/${_SAFE_MODEL}_dpo.parquet"
    _STALE_PATTERN="_rejection.parquet"
else
    _DEFAULT_OUTPUT="${OUTPUT_DIR}/${_SAFE_MODEL}_rejection.parquet"
    _STALE_PATTERN="_dpo.parquet"
fi
if [ -z "$OUTPUT_FILE" ] || [[ "$OUTPUT_FILE" == *"$_STALE_PATTERN" ]]; then
    OUTPUT_FILE="$_DEFAULT_OUTPUT"
fi

mkdir -p "$OUTPUT_DIR"

# ============ Backend: OpenAI ============
if [ "$BACKEND" = "openai" ]; then
    BASE_URL="${BASE_URL:-https://api.openai.com/v1}"
    API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
    if [ -z "$API_KEY" ]; then
        echo "ERROR: API_KEY is not set. Export OPENAI_API_KEY or set API_KEY." >&2
        return 1 2>/dev/null || exit 1
    fi
    _MODEL_ARG="$MODEL_NAME"
    # For API backends, tokenizer defaults to model name (user can override)
    TOKENIZER="${TOKENIZER:-$MODEL_NAME}"
fi

# ============ Backend: vLLM ============
if [ "$BACKEND" = "vllm" ]; then
    export TOKENIZERS_PARALLELISM=true
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

    VLLM_PORT="${VLLM_PORT:-8000}"
    BASE_URL="http://localhost:${VLLM_PORT}/v1"
    API_KEY="EMPTY"

    VLLM_TP="${VLLM_TP:-8}"
    MODEL_PATH="$ROOT_DIR/models/$MODEL_NAME"
    # max_model_len = prompt_length + response_length (matches veRL formula)
    VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-163840}"
    _MODEL_ARG="$MODEL_PATH"
    # For vLLM, tokenizer defaults to model path
    TOKENIZER="${TOKENIZER:-$MODEL_PATH}"

    # ---- vLLM lifecycle helpers ----

    check_vllm_serving_model() {
        local response
        response=$(curl -sf "http://localhost:${VLLM_PORT}/v1/models" 2>/dev/null) || return 1
        echo "$response" | grep -q "$MODEL_PATH" && return 0
        echo "$response" | grep -q "$MODEL_NAME"  && return 0
        return 1
    }

    kill_existing_vllm() {
        local pids
        pids=$(pgrep -f "vllm serve.*--port $VLLM_PORT" 2>/dev/null \
            || pgrep -f "vllm.entrypoints.*--port $VLLM_PORT" 2>/dev/null)
        if [ -n "$pids" ]; then
            echo "Killing existing vLLM server(s) on port $VLLM_PORT (pids: $pids)..."
            echo "$pids" | xargs kill 2>/dev/null
            sleep 3
            pids=$(pgrep -f "vllm serve.*--port $VLLM_PORT" 2>/dev/null \
                || pgrep -f "vllm.entrypoints.*--port $VLLM_PORT" 2>/dev/null)
            [ -n "$pids" ] && echo "$pids" | xargs kill -9 2>/dev/null && sleep 1
        fi
    }

    start_vllm() {
        echo "Starting vLLM server: model=$MODEL_PATH tp=$VLLM_TP port=$VLLM_PORT"
        VLLM_USE_V1=1 uv run --no-sync vllm serve "$MODEL_PATH" \
            --port "$VLLM_PORT" \
            --tensor-parallel-size "$VLLM_TP" \
            --max-model-len "$VLLM_MAX_MODEL_LEN" \
            &>"$OUTPUT_DIR/vllm.log" &
        local pid=$!
        echo "vLLM server starting (pid $pid), waiting for it to be ready..."

        local max_wait=1200 elapsed=0
        while [ $elapsed -lt $max_wait ]; do
            if curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
                echo "vLLM server ready (took ${elapsed}s)."
                return 0
            fi
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "ERROR: vLLM server exited unexpectedly. Check $OUTPUT_DIR/vllm.log" >&2
                tail -20 "$OUTPUT_DIR/vllm.log" >&2
                return 1 2>/dev/null || exit 1
            fi
            sleep 5
            elapsed=$((elapsed + 5))
        done
        echo "ERROR: vLLM server not ready after ${max_wait}s. Check $OUTPUT_DIR/vllm.log" >&2
        tail -20 "$OUTPUT_DIR/vllm.log" >&2
        return 1 2>/dev/null || exit 1
    }

    ensure_vllm() {
        if check_vllm_serving_model; then
            echo "vLLM already serving $MODEL_NAME on port $VLLM_PORT, reusing."
            return 0
        fi
        kill_existing_vllm
        start_vllm || return 1
    }

    ensure_vllm || { echo "ERROR: vLLM failed to start. Aborting." >&2; return 1 2>/dev/null || exit 1; }
fi

# ============ Backend: SGLang ============
if [ "$BACKEND" = "sglang" ]; then
    export TOKENIZERS_PARALLELISM=true

    SGLANG_PORT="${SGLANG_PORT:-${VLLM_PORT:-8000}}"
    BASE_URL="http://localhost:${SGLANG_PORT}/v1"
    API_KEY="EMPTY"

    SGLANG_TP="${SGLANG_TP:-${VLLM_TP:-8}}"
    SGLANG_DP="${SGLANG_DP:-1}"
    MODEL_PATH="$ROOT_DIR/models/$MODEL_NAME"
    SGLANG_MAX_MODEL_LEN="${SGLANG_MAX_MODEL_LEN:-${VLLM_MAX_MODEL_LEN:-163840}}"
    _MODEL_ARG="$MODEL_PATH"
    TOKENIZER="${TOKENIZER:-$MODEL_PATH}"

    # ---- SGLang lifecycle helpers ----

    check_sglang_serving_model() {
        local response
        response=$(curl -sf "http://localhost:${SGLANG_PORT}/v1/models" 2>/dev/null) || return 1
        echo "$response" | grep -q "$MODEL_PATH" && return 0
        echo "$response" | grep -q "$MODEL_NAME"  && return 0
        return 1
    }

    kill_existing_sglang() {
        local pids
        pids=$(pgrep -f "sglang.*launch_server.*--port $SGLANG_PORT" 2>/dev/null \
            || pgrep -f "sglang.*--port $SGLANG_PORT" 2>/dev/null)
        if [ -n "$pids" ]; then
            echo "Killing existing SGLang server(s) on port $SGLANG_PORT (pids: $pids)..."
            echo "$pids" | xargs kill 2>/dev/null
            sleep 3
            pids=$(pgrep -f "sglang.*launch_server.*--port $SGLANG_PORT" 2>/dev/null \
                || pgrep -f "sglang.*--port $SGLANG_PORT" 2>/dev/null)
            [ -n "$pids" ] && echo "$pids" | xargs kill -9 2>/dev/null && sleep 1
        fi
    }

    start_sglang() {
        echo "Starting SGLang server: model=$MODEL_PATH tp=$SGLANG_TP dp=$SGLANG_DP port=$SGLANG_PORT"
        uv run --no-sync python -m sglang.launch_server \
            --model-path "$MODEL_PATH" \
            --port "$SGLANG_PORT" \
            --tp "$SGLANG_TP" \
            --dp "$SGLANG_DP" \
            --context-length "$SGLANG_MAX_MODEL_LEN" \
            --dtype bfloat16 \
            &>"$OUTPUT_DIR/sglang.log" &
        local pid=$!
        echo "SGLang server starting (pid $pid), waiting for it to be ready..."

        local max_wait=1200 elapsed=0
        while [ $elapsed -lt $max_wait ]; do
            if curl -sf "http://localhost:${SGLANG_PORT}/v1/models" >/dev/null 2>&1; then
                echo "SGLang server ready (took ${elapsed}s)."
                return 0
            fi
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "ERROR: SGLang server exited unexpectedly. Check $OUTPUT_DIR/sglang.log" >&2
                tail -20 "$OUTPUT_DIR/sglang.log" >&2
                return 1 2>/dev/null || exit 1
            fi
            sleep 5
            elapsed=$((elapsed + 5))
        done
        echo "ERROR: SGLang server not ready after ${max_wait}s. Check $OUTPUT_DIR/sglang.log" >&2
        tail -20 "$OUTPUT_DIR/sglang.log" >&2
        return 1 2>/dev/null || exit 1
    }

    ensure_sglang() {
        if check_sglang_serving_model; then
            echo "SGLang already serving $MODEL_NAME on port $SGLANG_PORT, reusing."
            return 0
        fi
        kill_existing_sglang
        start_sglang || return 1
    }

    ensure_sglang || { echo "ERROR: SGLang failed to start. Aborting." >&2; return 1 2>/dev/null || exit 1; }
fi

# ============ Print summary ============
echo "=========================================="
echo "  SWE Trajectory Collection"
echo "=========================================="
echo "  Backend:      $BACKEND"
echo "  Mode:         $MODE"
echo "  Model:        $_MODEL_ARG"
echo "  Tokenizer:    $TOKENIZER"
echo "  Scaffold:     $SCAFFOLD"
echo "  API URL:      $BASE_URL"
echo "  Data:         $DATA_FILE"
echo "  Output:       $OUTPUT_FILE"
echo "  Target:       $TARGET passing trajectories"
echo "  Parallel:     $N_PARALLEL"
echo "  Max steps:    $MAX_STEPS"
echo "  Max tokens:   $MAX_RESPONSE_LENGTH (response budget), $MAX_PROMPT_LENGTH (prompt limit)"
echo "  Traj timeout: ${TRAJECTORY_TIMEOUT}s"
[ "$MODE" = "rejection" ] && echo "  Checkpoint:   every $CHECKPOINT_INTERVAL passes"
echo "  Disable thinking: $DISABLE_THINKING"
echo "=========================================="
echo ""

# ============ Run ============
_DISABLE_THINKING_FLAG=""
[ "$DISABLE_THINKING" = "true" ] && _DISABLE_THINKING_FLAG="--disable_thinking"

python3 utils/collect/collect_swe_trajectories.py \
    --model "$_MODEL_ARG" \
    --base_url "$BASE_URL" \
    --api_key "$API_KEY" \
    --tokenizer "$TOKENIZER" \
    --backend "$BACKEND" \
    --scaffold "$SCAFFOLD" \
    --data "$DATA_FILE" \
    --output "$OUTPUT_FILE" \
    --mode "$MODE" \
    --target "$TARGET" \
    --temperature "$TEMPERATURE" \
    --max_steps "$MAX_STEPS" \
    --max_response_length "$MAX_RESPONSE_LENGTH" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --n_parallel "$N_PARALLEL" \
    --trajectory_timeout "$TRAJECTORY_TIMEOUT" \
    --step_timeout "$STEP_TIMEOUT" \
    --reward_timeout "$REWARD_TIMEOUT" \
    --checkpoint_interval "$CHECKPOINT_INTERVAL" \
    $_DISABLE_THINKING_FLAG
