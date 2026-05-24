#!/usr/bin/env bash
# Launch the Sandbox Explorer web UI.
# Usage: ./sandbox_explorer.sh [--port 8501] [--gateway http://...]

# ./sandbox_explorer.sh --port 8501 --gateway http://118.145.210.10:8080

set -euo pipefail
cd "$(dirname "$0")"

PORT=8501
GATEWAY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)    PORT="$2"; shift 2 ;;
    --gateway) GATEWAY="$2"; shift 2 ;;
    *)         echo "Unknown arg: $1"; exit 1 ;;
  esac
done

ARGS=(--port "$PORT")
[[ -n "$GATEWAY" ]] && ARGS+=(--gateway "$GATEWAY")

echo "Starting Sandbox Explorer on port $PORT ..."
exec python3 utils/infra/sandbox_server.py "${ARGS[@]}"
