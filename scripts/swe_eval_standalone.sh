#!/usr/bin/env bash
# Standalone SWE evaluation — dry-run or agent mode.
# Supports both SWE-Bench Verified and R2E-Gym datasets.
#
# Usage:
#   # Dry run — SWE-Bench Verified (with fault tracing)
#   bash scripts/swe_eval_standalone.sh --dry_run \
#       --data data/swe/SWE_Bench_Verified.parquet
#
#   # Dry run — R2E-Gym Subset
#   bash scripts/swe_eval_standalone.sh --dry_run \
#       --data data/R2E-Gym/R2E-Gym-Subset-train.parquet
#
#   # Agent eval — SWE-Bench Verified
#   bash scripts/swe_eval_standalone.sh \
#       --model /path/to/model \
#       --data data/swe/SWE_Bench_Verified.parquet
#
# Environment variables (optional):
#   ARL_GATEWAY_URL   — ARL gateway (default: http://localhost:8080)
#   N_PARALLEL        — concurrency (default: 48)
#   OUTPUT_DIR        — results directory (default: eval_results)
#   NORMALIZE_PYTEST  — set to 1 to add -rA/--tb=short (default: 1)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

N_PARALLEL="${N_PARALLEL:-48}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results}"
NORMALIZE_PYTEST="${NORMALIZE_PYTEST:-1}"

EXTRA_ARGS=()
if [[ "$NORMALIZE_PYTEST" == "1" ]]; then
    EXTRA_ARGS+=(--normalize_pytest)
fi

exec uv run python3 scripts/swe_eval_standalone.py \
    --n_parallel "$N_PARALLEL" \
    --output_dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}" \
    "$@"
