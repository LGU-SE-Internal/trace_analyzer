#!/bin/bash
# SWE Setup Script
# !It should be applied to all workers/nodes!
# Usage: source swe-setup.sh [root]
#
# ── Rollout engine ─────────────────────────────────────────────────────────
#   ROLLOUT_ENGINE=vllm   (default) — install verl-vllm extra
#   ROLLOUT_ENGINE=sglang            — install verl-sglang extra
#
# Examples:
#   # Setup with default root directory (vLLM)
#   source swe-setup.sh
#
#   # Setup with SGLang
#   ROLLOUT_ENGINE=sglang source swe-setup.sh
#
#   # Setup with custom root directory
#   source swe-setup.sh /mnt/bn/my-bucket


# Rollout engine selection
export ROLLOUT_ENGINE="${ROLLOUT_ENGINE:-sglang}"

pip install uv

# IMPORTANT: if use BYTED cluster, set this to true
# Auto-detect BYTED cluster by checking ARNOLD_JOB_ID
if [ -n "$ARNOLD_JOB_ID" ]; then
    use_byted_venv=true
else
    use_byted_venv=false
fi

# It is important to set all env_var in all workers/nodes.
export ARL_GATEWAY_URL="http://118.145.210.10:8080"
export ARL_MIRROR_NAMESPACE="code"

# BYTED: set proxy for connections to internet
if [ "$use_byted_venv" = true ]; then
    export UV_HTTP_TIMEOUT=300
    export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
    export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
    export no_proxy="localhost,127.0.0.1"
fi

# uv venv setup
[ -d ".venv" ] || uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[verl-${ROLLOUT_ENGINE}]"
uv pip install swebench==4.1.0
### IMPORTANT: set PYTHONPATH in ray's env_args to .venv/lib/python3.11/site-packages to ensure ray workers can find the packages

# Pre-cache the datasets
python3 scripts/data/swe_dataset.py --local_dir ./data/swe

# ============ BYTERD ============
# copy k8s config
if [ "$use_byted_venv" = true ]; then
    uv pip uninstall ray wandb bytedray byted-wandb
    uv pip install bytedray[default,data,serve,bytedance]==2.10.0.34 byted-wandb --index-url https://bytedpypi.byted.org/simple/
    uv pip install "fastapi>=0.107.0,<0.113.0" # strange bug: bytedray, vllm, fastapi are not compatible
fi
uv pip install "setuptools>=64,<70"