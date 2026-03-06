#!/bin/bash
# Precompute P2A bonus maps for SWE dataset instances.
#
# Static mode (no sandbox): AST diff only, extracts patched callables, d=0.
# Dynamic mode (sandbox):   Full trace pipeline → call graph → hop distances.
#
# Usage:
#   # Static mode (fast, CPU only, no sandbox needed)
#   bash swe-precompute-bonus-maps.sh static
#
#   # Dynamic mode (needs ARL sandbox cluster)
#   bash swe-precompute-bonus-maps.sh dynamic
#
#   # Custom dataset / parallelism
#   DATA_FILE=data/swe/SWE_Bench_Verified.parquet \
#   N_PARALLEL=32 \
#   bash swe-precompute-bonus-maps.sh static
#
#   # Process only first 10 instances (for testing)
#   LIMIT=10 bash swe-precompute-bonus-maps.sh static

set -x

# ============ Arguments ============
MODE=${1:-static}

# ============ Configurable via env vars ============
DATA_FILE="${DATA_FILE:-data/swe/R2E_Gym_Subset.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-data/swe/bonus_maps}"
N_PARALLEL="${N_PARALLEL:-32}"
LIMIT="${LIMIT:-}"

# ============ Environment ============
export UV_INDEX_URL=https://bytedpypi.byted.org/simple/
export HF_ENDPOINT=https://hf-mirror.com

export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"

export ARL_GATEWAY_URL="${ARL_GATEWAY_URL:-http://118.145.210.10:8080}"

# ============ Clean up stale ARL resources (dynamic mode only) ============
if [ "$MODE" = "dynamic" ]; then
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
fi

# ============ Run ============
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

EXTRA_ARGS=""
if [ -n "$LIMIT" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --limit $LIMIT"
fi

uv run --no-sync python3 "$SCRIPT_DIR/scripts/precompute_bonus_maps.py" \
    "$DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --mode "$MODE" \
    --n_parallel "$N_PARALLEL" \
    $EXTRA_ARGS \
    2>&1 | tee precompute_bonus_maps.log
