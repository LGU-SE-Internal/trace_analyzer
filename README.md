# rLLM-SWE: Agentic RL for Software Engineering

Fork of [rLLM](https://github.com/rllm-project/rllm) for training SWE agents via agentic reinforcement learning. Agents learn to autonomously fix real-world software bugs from SWE-Bench and R2E-Gym datasets.

Key addition: **P2A** (Program-Analysis-based Process Advantage) — a sound process supervision method that uses call-graph analysis to give bonus rewards for fault localization steps during RL training.

## Setup

Prerequisites: Python >= 3.10, `uv`, kubectl, a K8s cluster with ARL sandbox pods.

```bash
# 1. Install dependencies
bash swe-setup.sh

# 2. Verify data files exist
ls data/swe/
#   R2E_Gym_Subset.parquet          (~4500 training tasks)
#   SWE_Bench_Verified.parquet      (500 eval tasks)
#   R2EGym_SFT_Trajectories.parquet (SFT warm-up data)
```

Edit `swe-setup.sh` to set `your_k8s_config_path` to your cluster's kubeconfig before running.

## Quick Start

### SFT Warm-up

Cold-start small models (Qwen3-4B) so they can produce valid trajectories before RL.

```bash
bash swe-train-sft.sh Qwen3-4B
```

Checkpoint saved to `$ROOT_DIR/experiments/verl/agentic-swe-sft/global_step_*/`.

### RL Training (RLOO)

```bash
bash swe-train-rl.sh Qwen3-4B
```

Train from an SFT checkpoint:

```bash
MODEL_PATH_OVERRIDE=/path/to/sft/checkpoint \
EXPERIMENT_NAME=sft-rloo \
bash swe-train-rl.sh Qwen3-4B
```

Use GRPO instead of RLOO:

```bash
ADV_ESTIMATOR=grpo bash swe-train-rl.sh Qwen3-4B
```

### Standalone Evaluation

Lightweight eval using a vLLM inference server (no Ray/FSDP needed). Auto-starts vLLM if not running.

```bash
# Greedy eval (n=1)
bash swe-eval-standalone.sh Qwen3-4B

# pass@5
bash swe-eval-standalone.sh Qwen3-4B 5

# Dry run: test harness on unmodified code (no model needed)
DRY_RUN=true bash swe-eval-standalone.sh dummy
```

## P2A: Process Supervision via Program Analysis

P2A gives bonus rewards to agent steps that read code on the golden call graph (from failing test to bug location). It is a **bonus-only** scheme: on-graph steps get amplified advantage, off-graph steps are unchanged.

### Step 0: Precompute Bonus Maps

One-time preprocessing. Extracts patched callables from each task's golden patch via AST diff. No sandbox needed, runs in minutes on CPU.

```bash
python scripts/precompute_bonus_maps.py \
    data/swe/R2E_Gym_Subset.parquet \
    --output_dir data/swe/bonus_maps \
    --mode static \
    --n_parallel 32
```

Output: `data/swe/bonus_maps/{instance_id}.json` — one per task, reusable across all experiments.

For full call-graph distances (requires sandbox):

```bash
python scripts/precompute_bonus_maps.py \
    data/swe/R2E_Gym_Subset.parquet \
    --output_dir data/swe/bonus_maps \
    --mode dynamic \
    --n_parallel 50
```

### Train with P2A

```bash
P2A_ENABLE=true \
P2A_BONUS_MAP_DIR=data/swe/bonus_maps \
bash swe-train-rl.sh Qwen3-4B
```

P2A environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `P2A_ENABLE` | `false` | Enable P2A advantage reshaping |
| `P2A_BONUS_MAP_DIR` | — | Path to precomputed bonus maps (required) |
| `P2A_M_MAX` | `3.0` | Max multiplier (single hyperparameter) |
| `P2A_TRACKING_MODE` | `view_only` | `view_only` or `view_and_bash` |

`view_only` tracks `file_editor view` commands. `view_and_bash` additionally tracks `cat`, `grep`, `head`, `tail`, `sed -n` in `execute_bash`.

### Eval with Localization Analysis

After agent eval, analyze how well the agent localizes bugs:

```bash
python scripts/swe_eval_standalone.py \
    --model /path/to/model \
    --data data/swe/SWE_Bench_Verified.parquet \
    --p2a_bonus_map_dir data/swe/bonus_maps \
    --p2a_tracking_mode view_and_bash
```

Or analyze existing trajectory logs:

```bash
python scripts/analyze_localization.py \
    --trajectories /path/to/chat_completions/10.jsonl \
    --bonus_map_dir data/swe/bonus_maps \
    --tracking_mode view_and_bash
```

Both report four metrics:
1. **On-graph Read Ratio** — fraction of all steps hitting the call graph
2. **On-graph View Density** — fraction of *view steps* (steps with read actions) hitting the call graph
3. **Avg Steps to Root Cause** — step index of first patched callable hit (lower is better)
4. **Root Cause Coverage** — fraction of trajectories reading at least one patched callable

## Table 1 Experiments

Orchestration script for all paper experiments. Manages 10 variants with dependency ordering.

```bash
# Print all experiment commands (review before running)
bash swe-train-table1.sh all Qwen3-4B

# Run a single experiment
bash swe-train-table1.sh rloo Qwen3-4B
bash swe-train-table1.sh sft Qwen3-4B

# After SFT finishes, run dependent experiments
SFT_CHECKPOINT_DIR=/path/to/sft/checkpoint \
bash swe-train-table1.sh sft-p2a Qwen3-4B

# Run all sequentially (will stop on failure)
DISPATCH_MODE=sequential bash swe-train-table1.sh all Qwen3-4B
```

### Experiment Variants

| Experiment | Pipeline | Key Difference |
|------------|----------|----------------|
| `zeroshot` | eval only | No training |
| `sft` | SFT | Standard supervised fine-tuning |
| `grpo` | RL (GRPO) | Group-relative advantage |
| `rloo` | RL (RLOO) | Leave-one-out advantage |
| `sft-grpo` | SFT + RL (GRPO) | Warm-start GRPO |
| `sft-rloo` | SFT + RL (RLOO) | Warm-start RLOO |
| `p2a` | RL (RLOO + P2A view) | P2A with file_editor tracking |
| `sft-p2a` | SFT + RL (RLOO + P2A view) | Warm-start P2A |
| `p2a-bash` | RL (RLOO + P2A view+bash) | P2A with broader tracking |
| `sft-p2a-bash` | SFT + RL (RLOO + P2A view+bash) | Warm-start P2A (broad) |

### Execution Order

```
Step 0: Precompute bonus maps (CPU, minutes)

Step 1 (parallel, no dependencies):
  zeroshot, sft, grpo, rloo, p2a, p2a-bash

Step 2 (after SFT checkpoint ready):
  sft-grpo, sft-rloo, sft-p2a, sft-p2a-bash
```

All experiments log to W&B project `xujunjielong` with name prefix `table1-*`.

## `swe-train-rl.sh` Reference

Positional arguments:

| Arg | Description |
|-----|-------------|
| `$1` | Model name (e.g., `Qwen3-4B`) — resolves to `$ROOT_DIR/models/$1` |
| `$2` | Root directory (default: `/mnt/bn/trae-research-models/xujunjielong`) |

Environment variable overrides:

| Variable | Default | Description |
|----------|---------|-------------|
| `ADV_ESTIMATOR` | `rloo` | Advantage estimator (`rloo` or `grpo`) |
| `EXPERIMENT_NAME` | `agentic-swe-rl` | W&B experiment name, also determines checkpoint dir |
| `MODEL_PATH_OVERRIDE` | — | Override model path (e.g., SFT checkpoint) |

Default training config: n=8 rollouts/prompt, 50 max steps, 1200s trajectory timeout, lr=1e-6, bf16, 8 GPUs, R2E-Gym training data, SWE-Bench Verified eval.

## Project Structure

```
swe-train-rl.sh                    # RL training entry point
swe-train-sft.sh                   # SFT warm-up
swe-train-table1.sh                # Table 1 experiment orchestration
swe-eval-standalone.sh             # Standalone eval (auto-starts vLLM)
swe-setup.sh                       # Environment setup

scripts/
  precompute_bonus_maps.py         # P2A bonus map precomputation
  swe_eval_standalone.py           # Eval logic (agent or dry-run)
  analyze_localization.py          # Post-hoc localization analysis
  swe_report.py                    # Results reporting
  patch_verl.sh                    # VeRL runtime patches
  data/swe_dataset.py              # Dataset download/preparation

rllm/
  trainer/
    verl/
      agent_ppo_trainer.py         # PPO trainer (stepwise advantage, P2A reshaping)
      train_agent_ppo.py           # Training entry point (Hydra + Ray)
      p2a.py                       # P2A core: bonus maps, read parsing, multiplier
    config/
      agent_ppo_trainer.yaml       # Hydra config (includes P2A settings)
  environments/swe/
    swe.py                         # SWE environment (ARL sandbox lifecycle)
    trace.py                       # Trace pipeline (AST diff, instrumentation, call graph)
    reward.py                      # Binary reward computation
    tools/                         # Agent tool scripts (file_editor, search, bash)
  agents/
    swe_agent.py                   # SWE agent (response parsing, trajectory tracking)
    agent.py                       # Base agent (Step, Trajectory dataclasses)
  engine/
    agent_execution_engine.py      # Async trajectory rollout

data/swe/
  R2E_Gym_Subset.parquet           # Training data (~4500 tasks)
  SWE_Bench_Verified.parquet       # Eval data (500 tasks)
  bonus_maps/                      # Precomputed P2A bonus maps (generated)
```
