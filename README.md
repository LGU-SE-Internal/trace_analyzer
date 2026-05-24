# rLLM-SWE: Agentic RL for Software Engineering

Fork of [rLLM](https://github.com/rllm-project/rllm) for training SWE agents via agentic reinforcement learning. Agents learn to autonomously fix real-world software bugs from SWE-Bench and R2E-Gym datasets.

Key addition: **P2A** (Program-Analysis-based Process Advantage) — a sound process supervision method that uses call-graph analysis to give bonus rewards for fault localization steps during RL training.

## ARL Architecture

All sandbox operations in this repo are built on **ARL (Agent Runtime Layer)** — no direct Docker or K8s API calls are made. ARL provides a Gateway HTTP API backed by a K8s cluster.

Source code: Gateway + K8s controllers in `~/Documents/agent-env`, Python SDK in `.venv/lib/python3.11/site-packages/arl/`.

### Resource Hierarchy

```
WarmPool (K8s CRD, persisted in etcd)
│   Declares: image, replicas count, resource limits
│   K8s controller maintains the desired number of pods
│
├── Pod-1 [Idle]        ← Fully running, containers Ready, waiting for assignment
├── Pod-2 [Idle]
├── Pod-3 [Allocated]   ← Bound to Sandbox-A
│   ├── executor container    (user image + executor-agent process)
│   └── sidecar container     (ARL platform, gRPC/HTTP interface)
├── Pod-4 [Allocated]   ← Bound to Sandbox-B
└── ...

Sandbox (K8s CRD, persisted in etcd)
│   Lifecycle: Pending → Bound → Ready → Deleted
│   "Binding" = label an Idle pod as Allocated (no container restart)
│   Deletion = pod is killed, WarmPool creates a new Idle pod to backfill
│
Session (Gateway in-memory only, NOT persisted)
    A handle mapping session_id → Sandbox/Pod info (pod_ip, pod_name)
    Created after Sandbox reaches Ready; lost if Gateway restarts
```

**Key points:**
- Idle pods are **not** empty — they have all containers running and Ready. "Warm" means pre-initialized.
- Sandbox is **not** a container — it is a logical claim on a pod. Binding only changes a pod label (`idle` → `allocated`).
- Pods do **not** return to Idle after use. When a Sandbox is deleted, the pod is also deleted; WarmPool controller creates a fresh one.
- Sessions exist **only** in Gateway process memory (`sync.Map`). If Gateway crashes/restarts, all session mappings are lost. Orphaned Sandboxes are cleaned up by the K8s idle timeout controller (default 600s).

### Scaling Behavior

| Direction | Trigger | Mechanism |
|-----------|---------|-----------|
| **Scale up** | `sessionCount + 1 > replicas` | Gateway's PoolManager patches WarmPool replicas (coalesced, 50ms batching) |
| **Scale down** | `replicas > sessionCount + 1` sustained for 5 min | PoolManager reduces replicas in sweep loop (every 30s) |
| **Pool GC** | Pool has 0 sessions for 10 min | PoolManager deletes the WarmPool CRD entirely |
| **Sandbox GC** | No task execution for 600s | K8s SandboxReconciler deletes the Sandbox (and its pod) |

### Client SDK

| Class | Role |
|-------|------|
| `ManagedSession` | High-level session: auto-creates pool from `image` + `experiment_id`, calls `create_sandbox()` / `delete_sandbox()` |
| `SandboxSession` | Lower-level: requires pre-existing `pool_ref` |
| `GatewayClient` | HTTP client to Gateway (`ARL_GATEWAY_URL`). Endpoints: create/delete session, execute steps, manage pools/experiments |

### Session Lifecycle (from this repo's perspective)

```
SWEEnv._create_session()
  └─ ManagedSession(image=..., experiment_id=...)
  └─ session.create_sandbox()
       └─ POST /v1/managed/sessions  (blocks until pod Ready, up to 300s)
       └─ On success: session._session_id = gateway-assigned ID
       └─ On failure: session._session_id remains None

SWEEnv.close()
  └─ session.delete_sandbox()
       └─ if _session_id is None: no-op (create never succeeded)
       └─ DELETE /v1/sessions/{id}
       └─ On failure: exception silently swallowed (except: pass)
```

This means: if `create_sandbox()` fails (timeout, 5xx), `_session_id` is never set, and `delete_sandbox()` becomes a safe no-op. No orphan is created client-side. However, if Gateway crashes **after** creating the Sandbox CRD but **before** returning the HTTP response, the Sandbox becomes an orphan that only K8s idle timeout can reclaim.

## Setup

Prerequisites: `uv` (Python package manager), kubectl, a K8s cluster with ARL sandbox pods.

All Python code in this repo runs through `uv run`, which manages the virtual environment and dependencies automatically via `uv.lock`. Never use bare `python` — it won't find the project dependencies.

```bash
# 0. Install uv if not present
pip install uv

# 1. Set up environment (venv, deps, kubectl, datasets)
#    Edit swe-setup.sh first: set your_k8s_config_path to your cluster's kubeconfig
bash swe-setup.sh

# 2. Verify data files exist
ls data/swe/
#   R2E_Gym_Subset.parquet          (~4500 training tasks)
#   SWE_Bench_Verified.parquet      (500 eval tasks)
#   R2EGym_SFT_Trajectories.parquet (SFT warm-up data)
```

`swe-setup.sh` does: `uv venv --python 3.11` + `uv pip install -e ".[verl-vllm]"` + kubectl setup + dataset download. All subsequent shell scripts use `uv run --no-sync python3 ...` internally.

## Quick Start

### SFT Warm-up

Cold-start small models (Qwen3-4B) so they can produce valid trajectories before RL.

```bash
bash swe-train-sft.sh Qwen3-4B
```

Checkpoint saved to `$ROOT_DIR/experiments/verl/agentic-swe-sft/global_step_*/`.

To merge FSDP sharded checkpoints (rank*.pt) into HuggingFace safetensors:

```bash
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /path/to/global_step_xxx/actor \
    --target_dir /path/to/merged_hf_model
```

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

## Data Collection (Rejection / DPO Sampling)

`swe-data-rejection.sh` collects trajectories from any OpenAI-compatible model — either a closed-source API or a local vLLM server — and saves them as training data.

Two **modes** control what gets saved:

| Mode | Keeps | Output columns | Use for |
|------|-------|----------------|---------|
| `rejection` (default) | Passing trajectories only (reward = 1.0) | `messages` | SFT warm-up |
| `dpo` | All trajectories; pairs pass ↔ fail per instance | `chosen`, `rejected`, `instance_id` | DPO/preference training |

Two **backends** control how the model is called:

| Backend | Description | Auth |
|---------|-------------|------|
| `api` (default) | Closed-source model via OpenAI-compatible API | `API_KEY` / `OPENAI_API_KEY` |
| `vllm` | Local model served by vLLM — auto-starts if not running | none |

Both modes and both backends support **checkpoint/resume**: re-running with the same `OUTPUT_FILE` continues where an interrupted run left off.

### Usage

```bash
# Rejection sampling from a closed-source API
source swe-data-rejection.sh gpt-4o

# DPO sampling from a closed-source API
MODE=dpo source swe-data-rejection.sh gpt-4o

# Custom endpoint (Azure, proxy, etc.)
BASE_URL=https://my-proxy.example.com/v1 API_KEY=sk-xxx \
    source swe-data-rejection.sh claude-opus-4-6

# Rejection sampling with a local vLLM server
BACKEND=vllm source swe-data-rejection.sh Qwen3-8B

# DPO sampling with a local vLLM server
BACKEND=vllm MODE=dpo source swe-data-rejection.sh Qwen3-8B

# Custom root dir (model path = $ROOT_DIR/models/$MODEL_NAME)
BACKEND=vllm source swe-data-rejection.sh Qwen3-8B /mnt/bn/my-bucket

# Resume an interrupted run — same OUTPUT_FILE, same flags
BACKEND=vllm MODE=dpo source swe-data-rejection.sh Qwen3-8B
```

vLLM lifecycle mirrors `swe-eval-standalone.sh`: the server stays running after the script exits and is reused on the next call. To stop it manually: `pkill -f "vllm serve"`.

### Output files

**Rejection mode** — parquet with a single `messages` column (list of `{role, content}` dicts), identical format to `R2EGym_SFT_Trajectories.parquet`:

```
data/swe/collected/<model>_rejection.parquet
```

**DPO mode** — parquet with `chosen`, `rejected`, `instance_id` columns.
Two JSONL sidecar files accumulate the raw cache and act as the live checkpoint; the final parquet is written at the end:

```
data/swe/collected/<model>_dpo.parquet          ← final paired output
data/swe/collected/<model>_dpo.pos.jsonl        ← passing trajectory cache (resume state)
data/swe/collected/<model>_dpo.neg.jsonl        ← failing trajectory cache (resume state)
```

Each positive trajectory is paired with one negative from the same `instance_id` (round-robin when counts differ). Positives with no matching negative are excluded from the final parquet.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODE` | `rejection` | `rejection` or `dpo` |
| `BACKEND` | `api` | `api` or `vllm` |
| `API_KEY` | `$OPENAI_API_KEY` | API key (api backend) |
| `BASE_URL` | `https://api.openai.com/v1` | API endpoint (api backend) |
| `TARGET` | `5000` | Stop after this many *passing* trajectories |
| `N_PARALLEL` | `128` | Max concurrent trajectories |
| `MAX_STEPS` | `50` | Max agent steps per trajectory |
| `MAX_RESPONSE_TOKENS` | `20000` | Max tokens per model response |
| `TEMPERATURE` | `1.0` | Sampling temperature |
| `TRAJECTORY_TIMEOUT` | `1200` | Wall-clock timeout per trajectory (s) |
| `OUTPUT_DIR` | `data/swe/collected` | Output directory |
| `OUTPUT_FILE` | auto | Override output parquet path |
| `CHECKPOINT_INTERVAL` | `50` | (rejection mode) checkpoint every N new passes |
| `VLLM_PORT` | `8000` | vLLM server port (vllm backend) |
| `VLLM_TP` | `8` | Tensor parallel size (vllm backend) |
| `VLLM_MAX_MODEL_LEN` | `163840` | Max model length for vLLM (vllm backend) |

## P2A: Process Supervision via Program Analysis

P2A gives bonus rewards to agent steps that read code on the golden call graph (from failing test to bug location). It is a **bonus-only** scheme: on-graph steps get amplified advantage, off-graph steps are unchanged.

### Step 0: Precompute Bonus Maps

One-time preprocessing. Extracts patched callables from each task's golden patch via AST diff. No sandbox needed, runs in minutes on CPU.

```bash
# Static mode (fast, CPU only, no sandbox)
bash swe-precompute-bonus-maps.sh static

# Dynamic mode (full call-graph distances, needs ARL sandbox)
bash swe-precompute-bonus-maps.sh dynamic

# Custom dataset / parallelism / limit
DATA_FILE=data/swe/SWE_Bench_Verified.parquet \
N_PARALLEL=50 \
bash swe-precompute-bonus-maps.sh static
```

Output: `data/swe/bonus_maps/{instance_id}.json` — one per task, reusable across all experiments.

#### Classification Decision Tree

Each instance is classified by walking the tree top-to-bottom (first match wins):

```
Instance
│
├─ Static layer (AST diff of old_file_content vs new_file_content):
│   ├─ All GT callables only in new content? ─── newly_created
│   └─ No callable-level changes in patch?   ─── no_callable
│
└─ Dynamic layer (instrument sandbox → run tests → parse traces):
    │
    ├─ 0 trace entries captured?             ─── no_trace   (error)
    │
    ├─ Traces exist but none contain
    │  a GT callable frame (is_patched)?     ─── no_gt      (error)
    │
    ├─ All tests pass on buggy code?
    │  (0 test failures detected)            ─── all_pass   (error)
    │
    ├─ Tests fail but F2P filter
    │  removed all GT traces?                ─── no_f2p     (error)
    │
    └─ F2P→GT call chain found:
        ├─ Intermediate nodes exist?         ─── standard   (traceable)
        └─ Test calls GT directly?           ─── direct     (traceable)
```

- **F2P** (fail-to-pass): test that FAILS on buggy code, PASSES after the developer's fix.
- **GT** (ground-truth): the callable(s) modified by the developer's golden patch.
- **error=True** cases indicate environmental or data issues — traced case-by-case via `utils/p2a/debug_instance.py`.

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
uv run python3 utils/eval/swe_eval_standalone.py \
    --model /path/to/model \
    --data data/swe/SWE_Bench_Verified.parquet \
    --p2a_bonus_map_dir data/swe/bonus_maps \
    --p2a_tracking_mode view_and_bash
```

Or analyze existing trajectory logs:

```bash
uv run python3 utils/p2a/analyze_localization.py \
    --trajectories /path/to/chat_completions/10.jsonl \
    --bonus_map_dir data/swe/bonus_maps \
    --tracking_mode view_and_bash
```

Both report four metrics:
1. **On-graph Read Ratio** — fraction of all steps hitting the call graph
2. **On-graph View Density** — fraction of *view steps* (steps with read actions) hitting the call graph
3. **Avg Steps to Root Cause** — step index of first patched callable hit (lower is better)
4. **Root Cause Coverage** — fraction of trajectories reading at least one patched callable

## Trajectory Analyzer

Browser-based UI for visualizing and analyzing agent trajectories — no server needed, open the HTML file directly.

```bash
open trajectory_analyzer.html
# or: python3 -m http.server 8000 && open http://localhost:8000/trajectory_analyzer.html
```

### Two tabs

**Dashboard** — macro view of a full experiment run.

**Inspector** — micro view of individual trajectories.

---

### Dashboard: analyzing an eval experiment

1. Click **Open Experiment Folder** and select the experiment output directory.
   The folder must contain a `*_n1.jsonl` metadata file and `chat_completions/eval.jsonl`.
2. Summary cards appear immediately:
   - **Total** — trajectories loaded (and how many are missing from SWE-Bench 500)
   - **Success (loaded)** — pass rate among loaded trajectories
   - **Success (SWE-B)** — pass rate over the full 500-instance SWE-Bench Verified set
3. Charts: termination reasons, tool usage, success by repo, step distribution, error rate, etc.
4. The **Trajectory Table** at the bottom lists every trajectory. Click any row to jump to the Inspector.

**Filters and sorting:** use the text filter box, reward/termination dropdowns, or click any column header.

---

### Inspector: stepping through a trajectory

1. **From Dashboard:** click any table row — the Inspector opens with that trajectory pre-selected.
2. **From training JSONL:** switch to the Inspector tab, click **Open Completion JSONL**, and select a `.jsonl` file (one JSON message array per line, no instance_id required).
3. Use the **trajectory dropdown** to switch between trajectories. Use the search box to filter by instance ID.
4. The **Action Sequence strip** shows every step as a colored dot. Click any dot to jump to that step's detail panel.
5. The **step detail panel** shows:
   - Tool name, command, parameters
   - Execution output (truncated long outputs are expandable)
   - For `str_replace` edits: a unified diff (`+`/`-` with red/green lines)
   - For errors: a red error badge

**Step color coding:**

| Color | Meaning |
|-------|---------|
| Blue | `file_editor` / `str_replace_editor` |
| Green | `execute_bash` |
| Yellow | `search` |
| Purple | `finish` / `submit` |
| Red border | step returned an error |

---

### Golden Patch analysis

Compares agent steps against the ground-truth bug location (callable-level).

**Step 1 — Export golden patch data** (one-time, run from repo root):

```bash
uv run python3 utils/p2a/export_golden_patches.py
# Output: data/swe/golden_patches.json
```

By default reads both `data/swe/SWE_Bench_Verified.parquet` and `data/swe/R2E_Gym_Subset.parquet`. Pass custom paths as positional arguments or use `--out` for a different output path.

**Step 2 — Load in the browser:**

On the **Dashboard** load bar: click **Load Golden Patches JSON** and select `golden_patches.json`. Then click **✨ Golden** to toggle highlighting on.

**Step 3 — Read the results:**

**Macro (Dashboard):** the **Fault Localization Analysis** panel shows:

| Row | Meaning |
|-----|---------|
| 🎯 Located | Agent viewed or edited the exact golden callable |
| ↳ Viewed + Edited / Only viewed / Only edited | Sub-breakdown |
| ↳ Located & Success / Failure | Cross-tab with reward |
| ❌ Not Located | Agent never reached the golden callable |
| ↳ Not Located & Success | "Lucky fix" — patched without locating |
| 📄 No-callable patch | Golden patch only changes non-callable code — excluded from localization stats |

**Micro (Inspector):** each step in the action strip gets a colored outline based on its relationship to the golden patch:

| Color | Level | Meaning |
|-------|-------|---------|
| Blue outline | `view-file` | Viewed a file that contains the golden callable |
| Cyan outline | `view-callable` | Viewed the exact line range of the golden callable |
| Yellow outline | `edit-file` | Edited a file that contains the golden callable |
| Orange outline | `edit-callable` | Edited code that includes the golden callable |

The trajectory dropdown shows compact tags for quick scanning:
`🎯` viewed callable · `✏️` edited callable · `🔍` viewed file only · `📁` edited file only · `[📄]` no-callable patch

**Training JSONL matching:** when a `golden_patches.json` is loaded and you open a training JSONL file in the Inspector, the tool automatically tries to resolve each trajectory's instance ID by matching the `<github_issue>` text against stored problem-statement fingerprints, so golden patch highlighting works even without explicit instance IDs.

---

### Scripts referenced

| Script | Purpose |
|--------|---------|
| `utils/p2a/export_golden_patches.py` | Export callable-level golden patch data from parquet(s) to JSON |
| `utils/p2a/analyze_traceability.py` | Batch classify instances by traceability (static or bonus-map mode) |
| `utils/p2a/precompute_bonus_maps.py` | Precompute dynamic call-graph bonus maps (needs ARL sandbox) |

## Sandbox Explorer

Interactive web UI for exploring any instance's ARL sandbox environment. Useful for debugging bonus map construction, inspecting trace output, and manually running commands inside containers.

```bash
# Launch (default port 7860, default gateway http://118.145.210.10:8080)
./sandbox_explorer.sh

# Custom port or gateway
./sandbox_explorer.sh 8080 http://YOUR_GATEWAY:8080
```

Then open `http://localhost:7860`.

**Features:**
- Browse all R2E-Gym and SWE-Bench instances with search
- One-click sandbox creation for any instance (calls `env.reset()`)
- Interactive terminal with command history (↑/↓), configurable timeout
- Real-time stdout/stderr with color coding
- Gateway URL and experiment ID configurable from the UI

**Files:** `sandbox_explorer.sh`, `sandbox_explorer.html`, `utils/infra/sandbox_server.py`

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
swe-data-rejection.sh              # Rejection / DPO trajectory collection
swe-precompute-bonus-maps.sh       # P2A bonus map precomputation
swe-setup.sh                       # Environment setup
sandbox_explorer.sh                # Launch sandbox explorer UI
sandbox_explorer.html              # Sandbox explorer frontend
trajectory_analyzer.html           # Trajectory analyzer UI (open directly in browser)

utils/
  expdata/                             # Experiment tracking service
    schema.py, server.py, client.py, import_local.py
  p2a/                                 # P2A bonus maps pipeline
    precompute_bonus_maps.py, analyze_traceability.py, analyze_localization.py
    debug_instance.py, export_golden_patches.py
  eval/                                # Evaluation pipeline
    swe_eval_standalone.py, swe_report.py
  collect/                             # Data collection & preparation
    collect_swe_trajectories.py, prepare_sft_data.py
  infra/                               # Infrastructure utilities
    batch_prefetch.py, mirror_images.py, sandbox_server.py
    clear_arl.sh, patch_verl.sh, launch_litellm.sh

scripts/
  data/swe_dataset.py              # Dataset download/preparation
  dump_cfg.py                      # Config dumping utility

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
