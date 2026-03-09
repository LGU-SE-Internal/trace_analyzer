#!/bin/bash
# Precompute P2A bonus maps for SWE dataset instances.
# Run swe-setup.sh first.
#
# Static mode (no sandbox): AST diff only, extracts patched callables, d=0.
# Dynamic mode (sandbox):   Full trace pipeline → call graph → hop distances.
#
# Usage:
#   # Static mode (fast, CPU only, no sandbox needed)
#   source swe-precompute-bonus-maps.sh static
#
#   # Dynamic mode (needs ARL sandbox cluster)
#   source swe-precompute-bonus-maps.sh dynamic
#
#   # Custom dataset / parallelism
#   DATA_FILE=data/swe/SWE_Bench_Verified.parquet \
#   N_PARALLEL=32 \
#   source swe-precompute-bonus-maps.sh static
#
#   # Process only first 10 instances (for testing)
#   LIMIT=10 source swe-precompute-bonus-maps.sh static


# ============ Arguments ============
MODE=${1:-static}

export ARL_EXPERIMENT_ID="${2:-bonus-maps}"

source scripts/clear_arl.sh

# ============ Configurable via env vars ============
DATA_FILE="${DATA_FILE:-data/swe/R2E_Gym_Subset.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-data/swe/bonus_maps}"
N_PARALLEL="${N_PARALLEL:-512}"
LIMIT="${LIMIT:-}"

# ============ Run ============
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

EXTRA_ARGS=""
if [ -n "$LIMIT" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --limit $LIMIT"
fi

python3 "$SCRIPT_DIR/scripts/precompute_bonus_maps.py" \
    "$DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --mode "$MODE" \
    --n_parallel "$N_PARALLEL" \
    $EXTRA_ARGS
