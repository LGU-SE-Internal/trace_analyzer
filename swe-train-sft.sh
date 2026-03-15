#!/bin/bash
# SWE SFT Training Script
# Main entry script with Ray.
# Run swe-setup.sh first.
# Usage: bash swe-train-sft.sh <model> [root]
#
# Examples:
#   # Train SFT from base model
#   bash swe-train-sft.sh Qwen3-8B
#
#   # Train with custom root directory
#   bash swe-train-sft.sh Qwen3-8B /mnt/bn/my-bucket

# ============ Arguments ============
MODEL_NAME=${1:?'Usage: bash swe-train-sft.sh <model_name> [root_dir]'}
ROOT_DIR=${2:-'/mnt/bn/trae-research-models/xujunjielong'}
EXPERIMENT_NAME="${EXPERIMENT_NAME:-agentic-swe-sft}"
NNODES=${ARNOLD_WORKER_NUM:-1}
BS_PER_NODE=${BS_PER_NODE:-256}
FINAL_STEP=${FINAL_STEP:-200}

export ARL_EXPERIMENT_ID="$EXPERIMENT_NAME"

bash scripts/clear_arl.sh

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

# ============ Run Training ============
SFT_DATA=data/swe/R2EGym_SFT_Trajectories_Qwen3.parquet
if [ ! -f "$SFT_DATA" ]; then
    echo "Preprocessing SFT data for Qwen3 (wrapping reasoning in <think> tags)..."
    uv run python scripts/prepare_sft_data.py \
        --input data/swe/R2EGym_SFT_Trajectories.parquet \
        --output "$SFT_DATA"
fi

torchrun \
    $TORCHRUN_ARGS \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$SFT_DATA \
    data.val_files=$SFT_DATA \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.max_length=32768 \
    data.truncation=right \
    data.train_batch_size=$((BS_PER_NODE * NNODES)) \
    data.micro_batch_size_per_gpu=2 \
    optim.lr=1e-5 \
    trainer.total_epochs=2 \
    model.partial_pretrain=$ROOT_DIR/models/$MODEL_NAME \
    model.fsdp_config.model_dtype=bf16 \
    trainer.default_local_dir=$ROOT_DIR/experiments/verl/$EXPERIMENT_NAME \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger="['console','wandb']" \
    ulysses_sequence_parallel_size=8 \
    use_remove_padding=true

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $ROOT_DIR/experiments/verl/$EXPERIMENT_NAME/global_step_$STEP \
    --target_dir $ROOT_DIR/models/$MODEL_NAME-P2A_SFT
