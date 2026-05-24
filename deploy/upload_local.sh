#!/usr/bin/env bash
# Upload local experiment data to expdata service.
#
# Usage:
#   bash deploy/upload_local.sh <dir> [--dry_run] [--server <url>]
#
# Examples:
#   bash deploy/upload_local.sh data/expriment/Qwen3-8B
#   bash deploy/upload_local.sh data/expriment                  # scan all subdirs
#   bash deploy/upload_local.sh data/expriment --dry_run        # preview only
#   bash deploy/upload_local.sh data/expriment --server http://other:8502
#
# Prerequisites:
#   - kubectl port-forward svc/expdata 8502:8502  (unless --server points elsewhere)
#   - Python with 'requests' installed (uses project venv if available)
#
# The script auto-detects experiment types:
#   eval:       *_n1.jsonl + chat_completions/eval.jsonl
#   collection: *.pos.jsonl / *.neg.jsonl
#   rollout:    1.jsonl, 2.jsonl, ... (numbered files with message lists)

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

if [ $# -lt 1 ]; then
    echo "Usage: bash deploy/upload_local.sh <dir> [--dry_run] [--server <url>]"
    exit 1
fi

DIR="$1"
shift

# Use project venv if available, otherwise system python
if [ -f "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
else
    PYTHON=python3
fi

PYTHONPATH="$PROJECT_ROOT" exec "$PYTHON" "$PROJECT_ROOT/utils/expdata/import_local.py" \
    --dir "$DIR" \
    --server "${EXPDATA_URL:-http://localhost:8502}" \
    "$@"
