#!/bin/bash
# SWE SFT Training Script
# Usage: source swe-train-sft.sh <model> [root]
#
# Examples:
#   # Train SFT from base model
#   source swe-train-sft.sh Qwen3-8B
#
#   # Train with custom root directory
#   source swe-train-sft.sh Qwen3-8B /mnt/bn/my-bucket

set -x

# ============ Arguments ============
MODEL_NAME=${1:?'Usage: source dummy.sh <model_name> [root_dir]'}
ROOT_DIR=${2:-'/mnt/bn/trae-research-models/xujunjielong'}

# ============ Environment ============
export GLOO_SOCKET_IFNAME='eth0'
export NCCL_SOCKET_IFNAME='eth0'

# ============ Config ============
WAND_PROJECT='xujunjielong'
EXPERIMENT_NAME='dummy'
LR=1e-5

# ============ Distributed training (compatible with 1~N nodes) ============
NNODES=${ARNOLD_WORKER_NUM:-1}
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
    data.train_batch_size=8 \
    data.micro_batch_size_per_gpu=2 \
    optim.lr=$LR \
    trainer.total_epochs=2000 \
    model.partial_pretrain=$ROOT_DIR/models/$MODEL_NAME \
    model.fsdp_config.model_dtype=bf16 \
    trainer.default_local_dir=$ROOT_DIR/experiments/verl/$EXPERIMENT_NAME \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger="['console']" \
    ulysses_sequence_parallel_size=8 \
    use_remove_padding=true \
    > $EXPERIMENT_NAME.log 2>&1
