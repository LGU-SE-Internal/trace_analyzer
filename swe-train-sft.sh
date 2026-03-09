#!/bin/bash
# SWE SFT Training Script
# Main entry script with Ray.
# Run swe-setup.sh first.
# Usage: source swe-train-sft.sh <model> [root]
#
# Examples:
#   # Train SFT from base model
#   source swe-train-sft.sh Qwen3-8B
#
#   # Train with custom root directory
#   source swe-train-sft.sh Qwen3-8B /mnt/bn/my-bucket

source .venv/bin/activate

# ============ Arguments ============
MODEL_NAME=${1:?'Usage: bash swe-train-sft.sh <model_name> [root_dir]'}
ROOT_DIR=${2:-'/mnt/bn/trae-research-models/xujunjielong'}
EXPERIMENT_NAME="${EXPERIMENT_NAME:-agentic-swe-sft}"
NNODES=${ARNOLD_WORKER_NUM:-1}
BS_PER_NODE=${BS_PER_NODE:-32}

export ARL_EXPERIMENT_ID="$EXPERIMENT_NAME"

source scripts/clear_arl.sh

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
torchrun \
    $TORCHRUN_ARGS \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=data/swe/R2EGym_SFT_Trajectories.parquet \
    data.val_files=data/swe/R2EGym_SFT_Trajectories.parquet \
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
    --local_dir $ROOT_DIR/experiments/verl/$EXPERIMENT_NAME/global_step_806 \
    --target_dir $ROOT_DIR/models/$MODEL_NAME-P2A_SFT
