#!/bin/bash
# Table 1 Experiment Orchestration Script
#
# Manages all experiment variants for the P2A paper Table 1.
#
# Usage:
#   bash swe-train-table1.sh <experiment|all> <model> [root]
#
# Examples:
#   bash swe-train-table1.sh all Qwen3-4B                    # Print all commands
#   bash swe-train-table1.sh sft-rloo Qwen3-4B               # Run single experiment
#   DISPATCH_MODE=sequential bash swe-train-table1.sh all Qwen3-4B  # Run all sequentially
#
# Experiments:
#   zeroshot      - Eval only (no training)
#   sft           - SFT → eval
#   grpo          - RL(GRPO) → eval
#   rloo          - RL(RLOO) → eval
#   sft-grpo      - SFT → RL(GRPO) → eval
#   sft-rloo      - SFT → RL(RLOO) → eval
#   p2a           - RL(RLOO+P2A, view_only) → eval
#   sft-p2a       - SFT → RL(RLOO+P2A, view_only) → eval
#   p2a-bash      - RL(RLOO+P2A, view_and_bash) → eval
#   sft-p2a-bash  - SFT → RL(RLOO+P2A, view_and_bash) → eval

set -euo pipefail

# ============ Arguments ============
EXPERIMENT=${1:?'Usage: bash swe-train-table1.sh <experiment|all> <model> [root]'}
MODEL_NAME=${2:?'Usage: bash swe-train-table1.sh <experiment|all> <model> [root]'}
ROOT_DIR=${3:-'/mnt/bn/trae-research-models/xujunjielong'}

# ============ Config ============
DISPATCH_MODE="${DISPATCH_MODE:-print}"  # print | sequential
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PREFIX="table1"

# P2A bonus map directory (must exist for p2a experiments)
P2A_BONUS_MAP_DIR="${P2A_BONUS_MAP_DIR:-data/swe/bonus_maps}"

# SFT checkpoint path pattern (auto-detected or overridden)
SFT_CHECKPOINT_DIR="${SFT_CHECKPOINT_DIR:-}"

# ============ Helpers ============

find_sft_checkpoint() {
    # Find the latest SFT checkpoint directory
    local sft_dir="$ROOT_DIR/experiments/verl/${PREFIX}-sft"
    if [ -n "$SFT_CHECKPOINT_DIR" ]; then
        echo "$SFT_CHECKPOINT_DIR"
        return
    fi
    if [ -d "$sft_dir" ]; then
        # Find the latest global_step_* directory
        local latest=$(ls -d "$sft_dir"/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
        if [ -n "$latest" ]; then
            echo "$latest"
            return
        fi
    fi
    echo ""
}

dispatch() {
    local name="$1"
    shift
    local cmd="$*"

    echo "=========================================="
    echo "  Experiment: $name"
    echo "=========================================="

    if [ "$DISPATCH_MODE" = "print" ]; then
        echo "$cmd"
        echo ""
    elif [ "$DISPATCH_MODE" = "sequential" ]; then
        echo "Running: $name"
        eval "$cmd"
        echo "Done: $name"
        echo ""
    else
        echo "ERROR: Unknown DISPATCH_MODE=$DISPATCH_MODE"
        exit 1
    fi
}

check_p2a_deps() {
    if [ ! -d "$P2A_BONUS_MAP_DIR" ] && [ "$DISPATCH_MODE" = "sequential" ]; then
        echo "ERROR: P2A bonus map directory not found: $P2A_BONUS_MAP_DIR"
        echo "  Run: bash swe-bonus-maps.sh static"
        return 1
    fi
    return 0
}

get_sft_ckpt_or_warn() {
    local sft_ckpt
    sft_ckpt=$(find_sft_checkpoint)
    if [ -z "$sft_ckpt" ]; then
        echo "WARNING: SFT checkpoint not found. Run 'sft' experiment first." >&2
        echo "  Expected at: $ROOT_DIR/experiments/verl/${PREFIX}-sft/global_step_*" >&2
        echo "  Or set SFT_CHECKPOINT_DIR=/path/to/checkpoint" >&2
        if [ "$DISPATCH_MODE" = "sequential" ]; then
            return 1
        fi
        echo "\$SFT_CHECKPOINT_DIR  # <-- REPLACE WITH ACTUAL PATH"
    else
        echo "$sft_ckpt"
    fi
}

# ============ Experiment Definitions ============

run_zeroshot() {
    dispatch "zeroshot" \
        "EXPERIMENT_NAME=${PREFIX}-zeroshot" \
        "bash $SCRIPT_DIR/swe-train-rl.sh $MODEL_NAME $ROOT_DIR" \
        "# NOTE: Add trainer.val_only=true to the Hydra overrides for eval-only"
}

run_sft() {
    dispatch "sft" \
        "EXPERIMENT_NAME=${PREFIX}-sft" \
        "bash $SCRIPT_DIR/swe-train-sft.sh $MODEL_NAME $ROOT_DIR"
}

run_grpo() {
    dispatch "grpo" \
        "ADV_ESTIMATOR=grpo" \
        "EXPERIMENT_NAME=${PREFIX}-grpo" \
        "bash $SCRIPT_DIR/swe-train-rl.sh $MODEL_NAME $ROOT_DIR"
}

run_rloo() {
    dispatch "rloo" \
        "ADV_ESTIMATOR=rloo" \
        "EXPERIMENT_NAME=${PREFIX}-rloo" \
        "bash $SCRIPT_DIR/swe-train-rl.sh $MODEL_NAME $ROOT_DIR"
}

run_sft_grpo() {
    local sft_ckpt
    sft_ckpt=$(get_sft_ckpt_or_warn) || return 1
    dispatch "sft-grpo" \
        "ADV_ESTIMATOR=grpo" \
        "EXPERIMENT_NAME=${PREFIX}-sft-grpo" \
        "MODEL_PATH_OVERRIDE=$sft_ckpt" \
        "bash $SCRIPT_DIR/swe-train-rl.sh $MODEL_NAME $ROOT_DIR"
}

run_sft_rloo() {
    local sft_ckpt
    sft_ckpt=$(get_sft_ckpt_or_warn) || return 1
    dispatch "sft-rloo" \
        "ADV_ESTIMATOR=rloo" \
        "EXPERIMENT_NAME=${PREFIX}-sft-rloo" \
        "MODEL_PATH_OVERRIDE=$sft_ckpt" \
        "bash $SCRIPT_DIR/swe-train-rl.sh $MODEL_NAME $ROOT_DIR"
}

# --- P2A variants (view_only tracking) ---

run_p2a() {
    check_p2a_deps || return 1
    dispatch "p2a" \
        "ADV_ESTIMATOR=rloo" \
        "EXPERIMENT_NAME=${PREFIX}-p2a" \
        "P2A_ENABLE=true" \
        "P2A_TRACKING_MODE=view_only" \
        "P2A_BONUS_MAP_DIR=$P2A_BONUS_MAP_DIR" \
        "bash $SCRIPT_DIR/swe-train-rl.sh $MODEL_NAME $ROOT_DIR"
}

run_sft_p2a() {
    local sft_ckpt
    sft_ckpt=$(get_sft_ckpt_or_warn) || return 1
    check_p2a_deps || return 1
    dispatch "sft-p2a" \
        "ADV_ESTIMATOR=rloo" \
        "EXPERIMENT_NAME=${PREFIX}-sft-p2a" \
        "MODEL_PATH_OVERRIDE=$sft_ckpt" \
        "P2A_ENABLE=true" \
        "P2A_TRACKING_MODE=view_only" \
        "P2A_BONUS_MAP_DIR=$P2A_BONUS_MAP_DIR" \
        "bash $SCRIPT_DIR/swe-train-rl.sh $MODEL_NAME $ROOT_DIR"
}

# --- P2A variants (view_and_bash tracking) ---

run_p2a_bash() {
    check_p2a_deps || return 1
    dispatch "p2a-bash" \
        "ADV_ESTIMATOR=rloo" \
        "EXPERIMENT_NAME=${PREFIX}-p2a-bash" \
        "P2A_ENABLE=true" \
        "P2A_TRACKING_MODE=view_and_bash" \
        "P2A_BONUS_MAP_DIR=$P2A_BONUS_MAP_DIR" \
        "bash $SCRIPT_DIR/swe-train-rl.sh $MODEL_NAME $ROOT_DIR"
}

run_sft_p2a_bash() {
    local sft_ckpt
    sft_ckpt=$(get_sft_ckpt_or_warn) || return 1
    check_p2a_deps || return 1
    dispatch "sft-p2a-bash" \
        "ADV_ESTIMATOR=rloo" \
        "EXPERIMENT_NAME=${PREFIX}-sft-p2a-bash" \
        "MODEL_PATH_OVERRIDE=$sft_ckpt" \
        "P2A_ENABLE=true" \
        "P2A_TRACKING_MODE=view_and_bash" \
        "P2A_BONUS_MAP_DIR=$P2A_BONUS_MAP_DIR" \
        "bash $SCRIPT_DIR/swe-train-rl.sh $MODEL_NAME $ROOT_DIR"
}

# ============ Dispatch ============

run_all() {
    echo "============================================"
    echo "  Table 1: All experiments for $MODEL_NAME"
    echo "  Mode: $DISPATCH_MODE"
    echo "============================================"
    echo ""

    # Phase 1: No dependencies
    run_zeroshot
    run_sft
    run_grpo
    run_rloo
    run_p2a
    run_p2a_bash

    # Phase 2: Depends on SFT checkpoint
    run_sft_grpo
    run_sft_rloo
    run_sft_p2a
    run_sft_p2a_bash
}

case "$EXPERIMENT" in
    all)            run_all ;;
    zeroshot)       run_zeroshot ;;
    sft)            run_sft ;;
    grpo)           run_grpo ;;
    rloo)           run_rloo ;;
    sft-grpo)       run_sft_grpo ;;
    sft-rloo)       run_sft_rloo ;;
    p2a)            run_p2a ;;
    sft-p2a)        run_sft_p2a ;;
    p2a-bash)       run_p2a_bash ;;
    sft-p2a-bash)   run_sft_p2a_bash ;;
    *)
        echo "ERROR: Unknown experiment '$EXPERIMENT'"
        echo "Valid experiments: zeroshot, sft, grpo, rloo, sft-grpo, sft-rloo, p2a, sft-p2a, p2a-bash, sft-p2a-bash, all"
        exit 1
        ;;
esac
