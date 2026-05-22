#!/bin/bash
# SWE SFT Training Script
# Main entry script with Ray.
# Run swe-setup.sh first.
# Usage: bash swe-train-sft.sh <model> [root]
#
# Examples:
#   # PerStep mode (default): per-step split, lr auto-scaled
#   bash swe-train-sft.sh Qwen3-8B
#   SFT_MODE=perstep bash swe-train-sft.sh Qwen3-8B
#
#   # MultiTurn mode: train on full trajectories directly
#   SFT_MODE=multiturn bash swe-train-sft.sh Qwen3-8B

# ============ Arguments ============
MODEL_NAME=${1:?'Usage: bash swe-train-sft.sh <model_name> [root_dir]'}
ROOT_DIR=${2:-'/mnt/bn/trae-research-models/xujunjielong'}
SFT_MODE=${SFT_MODE:-perstep}  # perstep | multiturn
EXPERIMENT_NAME="${EXPERIMENT_NAME:-agentic-swe-sft}"
NNODES=${ARNOLD_WORKER_NUM:-1}
BS_PER_NODE=${BS_PER_NODE:-512}
LR=${LR:-1e-5}

EXPERIMENT_NAME="$EXPERIMENT_NAME-$MODEL_NAME"

export ARL_EXPERIMENT_ID="$EXPERIMENT_NAME"

bash utils/infra/clear_arl.sh

# ============ Environment ============
export GLOO_SOCKET_IFNAME='eth0'
export NCCL_SOCKET_IFNAME='eth0'

# ============ Config ============
WAND_PROJECT='xujunjielong'

# ============ Distributed training (compatible with 1~N nodes) ============
NPROC_PER_NODE=8

if [ "$NNODES" -gt 1 ]; then
    TORCHRUN_ARGS="--nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE --node_rank=${ARNOLD_ID:-0} --master_addr=${ARNOLD_WORKER_0_HOST} --master_port=${ARNOLD_WORKER_0_PORT}"
else
    TORCHRUN_ARGS="--standalone --nproc_per_node=$NPROC_PER_NODE"
fi

# ============ Mode-specific config ============
if [ "$SFT_MODE" = "perstep" ]; then
    SFT_DATA=data/swe/R2EGym_SFT_Trajectories_PerStep_LastThink.parquet
    if [ ! -f "$SFT_DATA" ]; then
        echo "Preprocessing SFT data for Qwen3-Style Completion (per-step split, prompt/response format)..."
        python3 utils/collect/prepare_sft_data.py \
            --input data/swe/R2EGym_SFT_Trajectories.parquet \
            --output "$SFT_DATA" \
            --model_path $ROOT_DIR/models/$MODEL_NAME
    fi

    # Scale lr: per-step splitting increases update count by avg_steps_per_trajectory
    SFT_META="${SFT_DATA%.parquet}.meta"
    if [ -f "$SFT_META" ]; then
        AVG_STEPS=$(cat "$SFT_META")
        EFFECTIVE_LR=$(python3 -c "print(f'{float($LR) / float($AVG_STEPS):.2e}')")
        echo "LR scaling: $LR / $AVG_STEPS avg_steps = $EFFECTIVE_LR"
    else
        EFFECTIVE_LR=$LR
        echo "WARNING: $SFT_META not found, using unscaled lr=$EFFECTIVE_LR"
    fi

    MODE_ARGS=(
        data.train_files=$SFT_DATA
        data.val_files=$SFT_DATA
        data.prompt_key=prompt
        data.response_key=response
        "data.apply_chat_template_kwargs.chat_template='{{ messages[0].content }}'"
        optim.lr=$EFFECTIVE_LR
    )

elif [ "$SFT_MODE" = "multiturn" ]; then
    SFT_DATA=data/swe/R2EGym_SFT_Trajectories.parquet

    MODE_ARGS=(
        data.train_files=$SFT_DATA
        data.val_files=$SFT_DATA
        data.multiturn.enable=True
        data.multiturn.messages_key=messages
        optim.lr=$LR
    )

else
    echo "ERROR: SFT_MODE must be 'perstep' or 'multiturn', got '$SFT_MODE'"
    exit 1
fi

echo "SFT_MODE=$SFT_MODE  DATA=$SFT_DATA  EXPERIMENT=$EXPERIMENT_NAME"

# ============ Run Training ============
torchrun \
    $TORCHRUN_ARGS \
    -m verl.trainer.fsdp_sft_trainer \
    "${MODE_ARGS[@]}" \
    data.max_length=32768 \
    data.truncation=left \
    data.train_batch_size=$((BS_PER_NODE * NNODES)) \
    data.micro_batch_size_per_gpu=4 \
    trainer.total_epochs=2 \
    model.partial_pretrain=$ROOT_DIR/models/$MODEL_NAME \
    model.fsdp_config.model_dtype=bf16 \
    trainer.default_local_dir=$ROOT_DIR/experiments/verl/$EXPERIMENT_NAME \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger="['console','wandb']" \
    ulysses_sequence_parallel_size=8 \
    use_remove_padding=true

# to merge rl checkpoints, use:
# python3 -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir $ROOT_DIR/experiments/verl/$EXPERIMENT_NAME/global_step_$FINAL_STEP \
#     --target_dir $ROOT_DIR/models/$EXPERIMENT_NAME
