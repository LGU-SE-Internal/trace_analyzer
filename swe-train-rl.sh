#!/bin/bash
# SWE RL Training Script
# Main entry script with Ray.
# Run swe-setup.sh first.
# Usage: source swe-train-rl.sh <model> [root]
#
# Examples:
#   # Train from base model
#   source swe-train-rl.sh Qwen3-8B
#
#   # Train from SFT checkpoint (pass full path as model)
#   source swe-train-rl.sh experiments/verl/agentic-swe-sft/global_step_1024
#
#   # Train with custom root directory
#   source swe-train-rl.sh Qwen3-8B /mnt/bn/my-bucket
#
#   # Train with GRPO instead of RLOO
#   ADV_ESTIMATOR=grpo source swe-train-rl.sh Qwen3-8B
#
#   # Train with P2A bonus
#   P2A_ENABLE=true P2A_BONUS_MAP_DIR=data/swe/bonus_maps source swe-train-rl.sh Qwen3-8B
#
#   # Train from SFT checkpoint with custom experiment name
#   MODEL_PATH_OVERRIDE=/path/to/sft/checkpoint EXPERIMENT_NAME=sft-rloo source swe-train-rl.sh Qwen3-8B

# ============ Arguments ============
MODEL_NAME=${1:?'Usage: bash swe-train-rl.sh <model_name> [root_dir]'}
ROOT_DIR=${2:-'/mnt/bn/trae-research-models/xujunjielong'}
EXPERIMENT_NAME="${EXPERIMENT_NAME:-agentic-swe-rl}"
NNODES=${ARNOLD_WORKER_NUM:-1}
BS_PER_NODE=${BS_PER_NODE:-32}

export ARL_EXPERIMENT_ID="$EXPERIMENT_NAME"

source scripts/clear_arl.sh

# ============ Configurable via env vars (backward-compatible defaults) ============
ADV_ESTIMATOR="${ADV_ESTIMATOR:-rloo}"
P2A_ENABLE="${P2A_ENABLE:-false}"
P2A_M_MAX="${P2A_M_MAX:-3.0}"
P2A_BONUS_MAP_DIR="${P2A_BONUS_MAP_DIR:-}"
P2A_TRACKING_MODE="${P2A_TRACKING_MODE:-view_only}"  # view_only | view_and_bash
MODEL_PATH_OVERRIDE="${MODEL_PATH_OVERRIDE:-}"

# Resolve model path
if [ -n "$MODEL_PATH_OVERRIDE" ]; then
    MODEL_PATH="$MODEL_PATH_OVERRIDE"
else
    MODEL_PATH="$ROOT_DIR/models/$MODEL_NAME"
fi

# ============ Environment ============
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export GLOO_SOCKET_IFNAME='eth0'
export NCCL_SOCKET_IFNAME='eth0'

# ============ Config ============
WAND_PROJECT='xujunjielong'

# If use fsdp offload, please set:
# tensor_model_parallel=8
# actor_rollout_ref.actor.fsdp_config.param_offload=true
# actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
# actor_rollout_ref.ref.fsdp_config.param_offload=true

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/scripts/patch_verl.sh"

# ============ Build P2A overrides ============
P2A_OVERRIDES=""
if [ "$P2A_ENABLE" = "true" ]; then
    P2A_OVERRIDES="rllm.p2a.enable=true"
    P2A_OVERRIDES="$P2A_OVERRIDES rllm.p2a.m_max=$P2A_M_MAX"
    P2A_OVERRIDES="$P2A_OVERRIDES rllm.p2a.tracking_mode=$P2A_TRACKING_MODE"
    if [ -n "$P2A_BONUS_MAP_DIR" ]; then
        P2A_OVERRIDES="$P2A_OVERRIDES rllm.p2a.bonus_map_dir=$P2A_BONUS_MAP_DIR"
    fi
    echo "P2A enabled with overrides: $P2A_OVERRIDES"
fi

# ============ Run Training ============
python3 -m rllm.trainer.verl.train_agent_ppo \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    data.train_files=data/swe/R2E_Gym_Subset.parquet \
    data.val_files=data/swe/SWE_Bench_Verified.parquet \
    trainer.default_local_dir=$ROOT_DIR/experiments/verl/$EXPERIMENT_NAME \
    trainer.rollout_data_dir=$ROOT_DIR/rollouts/$EXPERIMENT_NAME \
    data.train_batch_size=$((BS_PER_NODE * NNODES)) \
    data.val_batch_size=512 \
    data.max_prompt_length=4096 \
    data.max_response_length=32768 \
    data.filter_overlong_prompts=true \
    data.filter_overlong_prompts_workers=32 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.hybrid_engine=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((BS_PER_NODE * NNODES)) \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32000 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=8 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=false \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$NNODES \
    trainer.save_freq=50 \
    trainer.test_freq=100 \
    trainer.default_hdfs_dir=null \
    rllm.env.name=swe \
    rllm.agent.name=sweagent \
    rllm.agent.max_steps=50 \
    rllm.agent.overlong_filter=true \
    rllm.agent.trajectory_timeout=1200 \
    +rllm.env.env_args.verbose=false \
    +rllm.env.env_args.scaffold=r2egym \
    +rllm.agent.agent_args.scaffold=r2egym \
    +rllm.agent.engine_args.n_parallel_agents=$((64 * NNODES)) \
    trainer.total_epochs=1000 \
    $P2A_OVERRIDES

# please keep `n_parallel_agents` on each node to be small to optimize KV cache utilization.