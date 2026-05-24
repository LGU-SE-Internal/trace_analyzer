#!/usr/bin/env python3
"""
Rejection / DPO Sampling Script for SWE Trajectories

Delegates trajectory rollout to ``AgentExecutionEngine.execute_tasks()`` and
wraps it with rejection / DPO sampling logic.

Two modes:

  rejection (default)
    Keep only passing trajectories (reward == 1.0).  Stop when --target passes
    are collected.  Output: parquet with ``messages``, ``termination_reason`` columns.

  dpo
    Cache ALL trajectories (pass and fail).  Stop when --target passing
    trajectories have been collected.  Then, for every positive trajectory,
    find one negative from the same instance and emit a DPO pair.
    Output: parquet with ``chosen``, ``rejected``, ``instance_id``,
    ``chosen_termination_reason``, ``rejected_termination_reason``.

Termination reasons come from the engine:
    ENV_DONE, MAX_STEPS, TRUNCATION, TIMEOUT, ENV_TIMEOUT, PROMPT_TRUNCATION

Both modes support checkpoint/resume: re-run with the same --output path to
continue where an interrupted run left off.

Usage:
    # Rejection sampling
    python -m utils.collect.collect_swe_trajectories \\
        --model gpt-4o --api_key $OPENAI_API_KEY

    # DPO sampling
    python -m utils.collect.collect_swe_trajectories \\
        --model gpt-4o --api_key $OPENAI_API_KEY --mode dpo \\
        --output data/swe/gpt4o_dpo.parquet

    # Local model with thinking disabled (faster rollout)
    python -m utils.collect.collect_swe_trajectories \\
        --model /path/to/Qwen3-Coder-Next --backend sglang \\
        --disable_thinking

    # Resume (same --output path)
    python -m utils.collect.collect_swe_trajectories \\
        --model gpt-4o --api_key $OPENAI_API_KEY --mode dpo \\
        --output data/swe/gpt4o_dpo.parquet
"""

import argparse
import asyncio
import itertools
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------


def load_tasks(data_path: str) -> list[dict]:
    """Load and normalise tasks from a parquet / json / jsonl file."""
    from rllm.data.dataset import Dataset
    from rllm.environments.swe.trace import normalize_task

    ds = Dataset.load_data(data_path)
    tasks = ds.get_data()
    tasks = [normalize_task(t) for t in tasks]

    for task in tasks:
        if not task.get("instance_id"):
            repo = task.get("repo", task.get("repo_name", "unknown"))
            commit = task.get("base_commit", task.get("commit_hash", ""))[:12]
            task["instance_id"] = f"{repo}__{commit}"

    print(f"Loaded {len(tasks)} tasks from {data_path}")
    return tasks


# ---------------------------------------------------------------------------
# Trajectory record — a (messages, termination_reason) pair
# ---------------------------------------------------------------------------


def _rec(messages: list[dict], reason: str, instance_id: str = "") -> dict:
    return {"messages": messages, "termination_reason": reason, "instance_id": instance_id}


# ---------------------------------------------------------------------------
# Rejection-mode checkpoint helpers
# ---------------------------------------------------------------------------


def load_rejection_checkpoint(output_path: str) -> list[dict]:
    """Return previously collected trajectory records from output parquet."""
    if not os.path.exists(output_path):
        return []
    try:
        df = pd.read_parquet(output_path)
        collected = []
        for _, row in df.iterrows():
            msgs = row["messages"]
            if hasattr(msgs, "tolist"):
                msgs = [dict(m) for m in msgs.tolist()]
            elif isinstance(msgs, list | tuple):
                msgs = [dict(m) for m in msgs]
            reason = row.get("termination_reason", "UNKNOWN") if "termination_reason" in df.columns else "UNKNOWN"
            iid = row.get("instance_id", "") if "instance_id" in df.columns else ""
            collected.append(_rec(msgs, reason, iid))
        print(f"Resumed (rejection): {len(collected)} trajectories from {output_path}")
        return collected
    except Exception as exc:
        logger.warning(f"Could not load checkpoint from {output_path}: {exc}")
        return []


def save_rejection_checkpoint(collected: list[dict], output_path: str) -> None:
    """Atomically write rejection-sampling parquet checkpoint."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tmp = output_path + ".tmp"
    rows = [{"instance_id": r["instance_id"], "messages": r["messages"], "termination_reason": r["termination_reason"]} for r in collected]
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    os.replace(tmp, output_path)


# ---------------------------------------------------------------------------
# DPO-mode checkpoint helpers
# ---------------------------------------------------------------------------


def _dpo_sidecar(output_path: str, kind: str) -> str:
    """Return path of the DPO sidecar JSONL file (kind = 'pos' or 'neg')."""
    base = output_path[:-8] if output_path.endswith(".parquet") else output_path
    return f"{base}.{kind}.jsonl"


def _load_jsonl_cache(path: str) -> dict[str, list[dict]]:
    """Load {instance_id -> [record, ...]} from a JSONL sidecar."""
    cache: dict[str, list[dict]] = {}
    if not os.path.exists(path):
        return cache
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                iid = obj["instance_id"]
                reason = obj.get("termination_reason", "UNKNOWN")
                cache.setdefault(iid, []).append(_rec(obj["messages"], reason, iid))
            except Exception as exc:
                logger.warning(f"Skipping malformed JSONL line in {path}: {exc}")
    return cache


def load_dpo_checkpoint(output_path: str) -> tuple[dict, dict]:
    """Return (pos_cache, neg_cache) dicts loaded from sidecar JSONL files."""
    pos_cache = _load_jsonl_cache(_dpo_sidecar(output_path, "pos"))
    neg_cache = _load_jsonl_cache(_dpo_sidecar(output_path, "neg"))
    n_pos = sum(len(v) for v in pos_cache.values())
    n_neg = sum(len(v) for v in neg_cache.values())
    if n_pos or n_neg:
        print(f"Resumed (DPO): {n_pos} positives, {n_neg} negatives from sidecars")
    return pos_cache, neg_cache


def _append_to_jsonl(path: str, instance_id: str, record: dict) -> None:
    """Append one record to a JSONL sidecar file (append-only)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "messages": record["messages"],
                    "termination_reason": record["termination_reason"],
                }
            )
            + "\n"
        )


# ---------------------------------------------------------------------------
# DPO pairing & final parquet
# ---------------------------------------------------------------------------


def pair_dpo_data(
    pos_cache: dict[str, list[dict]],
    neg_cache: dict[str, list[dict]],
) -> list[dict]:
    """Create DPO pairs: for each positive, find one negative from the same instance.

    If an instance has k positives and m negatives (m >= 1), we emit k pairs,
    cycling through the negatives (round-robin) when k > m.
    Instances that have positives but no negatives (or vice-versa) are skipped.
    """
    pairs = []
    for iid, pos_list in pos_cache.items():
        neg_list = neg_cache.get(iid)
        if not neg_list:
            continue  # no negative to pair with
        for i, chosen_rec in enumerate(pos_list):
            rejected_rec = neg_list[i % len(neg_list)]
            pairs.append(
                {
                    "instance_id": iid,
                    "chosen": chosen_rec["messages"],
                    "rejected": rejected_rec["messages"],
                    "chosen_termination_reason": chosen_rec["termination_reason"],
                    "rejected_termination_reason": rejected_rec["termination_reason"],
                }
            )
    return pairs


def save_dpo_parquet(pairs: list[dict], output_path: str) -> None:
    """Atomically write DPO pairs to a parquet file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tmp = output_path + ".tmp"
    pd.DataFrame(pairs).to_parquet(tmp, index=False)
    os.replace(tmp, output_path)
    print(f"DPO parquet saved: {len(pairs)} pairs -> {output_path}")


# ---------------------------------------------------------------------------
# Main sampling loop  (uses AgentExecutionEngine)
# ---------------------------------------------------------------------------
#
# Backend handling:
#   - vLLM (--backend vllm): tokenizer loaded from model path, OpenAIEngine uses
#     the completion API (text prompt → text response).  Requires the model to
#     support the /v1/completions endpoint (all vLLM models do).
#   - API  (--backend api):  tokenizer loaded from --tokenizer (proxy, for length
#     tracking only), OpenAIEngine uses chat completions API (messages in/out).
#     Works with GPT-4o, Claude, or any OpenAI-compatible chat API.
#
# Concurrency:
#   Mirrors the trainer's approach (agent_ppo_trainer.generate_agent_trajectories_async):
#   we use a semaphore-guarded producer that feeds new tasks as soon as slots free
#   up, avoiding the "tail effect" of batch-based execute_tasks().


async def sample_loop(tasks: list[dict], args: argparse.Namespace) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from transformers import AutoTokenizer

    from rllm.agents.swe_agent import SWEAgent
    from rllm.engine.agent_execution_engine import AgentExecutionEngine
    from rllm.engine.rollout.openai_engine import OpenAIEngine
    from rllm.engine.rollout.verl_aligned_engine import VerlAlignedEngine
    from rllm.environments.swe.swe import SWEEnv

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Build the rollout engine explicitly:
    #
    #   --backend vllm/sglang: VerlAlignedEngine — replicates VerlEngine's
    #       encode/decode logic (token IDs in, tokenizer.decode with
    #       skip_special_tokens=True out) via /v1/completions HTTP API.
    #
    #   --backend openai: OpenAIEngine with tokenizer=None → chat completions
    #       API (/v1/chat/completions) for remote APIs (GPT-4o, Claude, etc.).
    #
    is_local = args.backend in ("vllm", "sglang")
    if is_local:
        rollout_engine = VerlAlignedEngine(
            model=args.model,
            tokenizer=tokenizer,
            base_url=args.base_url,
            api_key=args.api_key,
            max_prompt_length=args.max_prompt_length,
            max_response_length=args.max_response_length,
            sampling_params={"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k},
            disable_thinking=args.disable_thinking,
        )
        thinking_label = " (thinking DISABLED)" if args.disable_thinking else ""
        print(f"Backend: {args.backend} (VerlAlignedEngine) — tokenizer '{args.tokenizer}'{thinking_label}")
    else:
        rollout_engine = OpenAIEngine(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            tokenizer=None,
            max_prompt_length=args.max_prompt_length,
            max_response_length=args.max_response_length,
        )
        print(f"Backend: OpenAI (chat completions) — tokenizer '{args.tokenizer}' used for length tracking only")

    engine = AgentExecutionEngine(
        agent_class=SWEAgent,
        env_class=SWEEnv,
        agent_args={"scaffold": args.scaffold, "format_model_response": True},
        env_args={
            "scaffold": args.scaffold,
            "step_timeout": args.step_timeout,
            "reward_timeout": args.reward_timeout,
            "verbose": False,
        },
        # engine_name doesn't matter much — we override rollout_engine below.
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params={"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k},
        rollout_engine_args={
            "model": args.model,
            "base_url": args.base_url,
            "api_key": args.api_key,
        },
        n_parallel_agents=args.n_parallel,
        max_steps=args.max_steps,
        max_response_length=args.max_response_length,
        max_prompt_length=args.max_prompt_length,
        trajectory_timeout=args.trajectory_timeout,
        retry_limit=3,
        overlong_filter=False,
    )
    # Override the auto-constructed rollout engine with our explicit one.
    engine.rollout_engine = rollout_engine

    is_dpo = args.mode == "dpo"

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    if is_dpo:
        pos_cache, neg_cache = load_dpo_checkpoint(args.output)
        n_pass = sum(len(v) for v in pos_cache.values())
        n_neg_total = sum(len(v) for v in neg_cache.values())
    else:
        collected = load_rejection_checkpoint(args.output)
        n_pass = len(collected)
        n_neg_total = 0

    if n_pass >= args.target:
        print(f"Already have {n_pass} >= {args.target} passes. Nothing to do.")
        if is_dpo:
            pairs = pair_dpo_data(pos_cache, neg_cache)
            if pairs:
                save_dpo_parquet(pairs, args.output)
        return

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------
    n_attempts = 0
    n_fail = 0
    last_checkpoint_pass = n_pass
    reason_counter: Counter = Counter()
    done_event = asyncio.Event()

    # Sidecar paths for DPO
    pos_sidecar = _dpo_sidecar(args.output, "pos") if is_dpo else ""
    neg_sidecar = _dpo_sidecar(args.output, "neg") if is_dpo else ""

    mode_label = "DPO" if is_dpo else "Rejection"
    print(f"{mode_label} sampling: target={args.target} passes, n_parallel={args.n_parallel}\n  model={args.model}  scaffold={args.scaffold}  max_steps={args.max_steps}\n  max_response_length={args.max_response_length}  max_prompt_length={args.max_prompt_length}  T={args.temperature}\n  trajectory_timeout={args.trajectory_timeout}s  disable_thinking={args.disable_thinking}\n" + (f"  DPO negatives already cached: {n_neg_total}\n" if is_dpo else ""))

    # ------------------------------------------------------------------
    # Semaphore-based streaming loop (mirrors trainer's approach)
    # ------------------------------------------------------------------
    # Like agent_ppo_trainer.generate_agent_trajectories_async, we use a
    # semaphore to cap concurrency.  As each trajectory completes, its slot
    # opens immediately for the next task — no batch tail effect.
    semaphore = asyncio.Semaphore(args.n_parallel)
    # Index queue: reusable agent/env slots, same pattern as execute_tasks
    index_queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(args.n_parallel):
        index_queue.put_nowait(i)
    # Ensure engine has enough agent/env slots
    engine.agents = [None] * args.n_parallel
    engine.envs = [None] * args.n_parallel
    # Ensure the executor is alive
    engine.executor = ThreadPoolExecutor(max_workers=args.n_parallel * 2)

    pending: set[asyncio.Task] = set()

    def _process_trajectory(traj) -> None:
        """Process a completed trajectory (called from event loop — safe to mutate counters)."""
        nonlocal n_attempts, n_pass, n_fail, last_checkpoint_pass, n_neg_total

        n_attempts += 1
        iid = traj.task.get("instance_id", "?") if traj.task else "?"
        messages = traj.info.get("chat_completions", [])
        reason = traj.info.get("termination_reason", "UNKNOWN")
        passed = (traj.reward or 0) >= 1.0
        metrics = traj.info.get("metrics", {})
        steps = metrics.get("steps", "?")
        llm_t = metrics.get("llm_time", 0)
        env_t = metrics.get("env_time", 0)
        rwd_t = metrics.get("reward_time", 0)
        tot_t = metrics.get("total_time", 0)
        timing_str = f"steps={steps} llm={llm_t:.0f}s env={env_t:.0f}s reward={rwd_t:.0f}s total={tot_t:.0f}s"

        reason_counter[reason] += 1
        rec = _rec(messages, reason, iid)

        if is_dpo:
            if passed:
                pos_cache.setdefault(iid, []).append(rec)
                _append_to_jsonl(pos_sidecar, iid, rec)
                n_pass += 1
                print(f"[PASS {reason}] {iid} | pos={n_pass}/{args.target} neg={n_neg_total} attempts={n_attempts} | {timing_str}")
                if n_pass >= args.target:
                    done_event.set()
            else:
                neg_cache.setdefault(iid, []).append(rec)
                _append_to_jsonl(neg_sidecar, iid, rec)
                n_neg_total += 1
                n_fail += 1
                print(f"[FAIL {reason}] {iid} | pos={n_pass} neg={n_neg_total} | {timing_str}")
        else:
            if passed:
                collected.append(rec)
                n_pass += 1
                print(f"[PASS {reason}] {iid} | pass={n_pass}/{args.target} fail={n_fail} attempts={n_attempts} | {timing_str}")
                if n_pass - last_checkpoint_pass >= args.checkpoint_interval:
                    save_rejection_checkpoint(collected, args.output)
                    last_checkpoint_pass = n_pass
                    print(f"  -> checkpoint saved ({n_pass} trajectories)")
                if n_pass >= args.target:
                    done_event.set()
            else:
                n_fail += 1
                print(f"[FAIL {reason}] {iid} | pass={n_pass} fail={n_fail} | {timing_str}")

    async def _run_one(task: dict) -> None:
        """Run a single trajectory: acquire slot, run, release."""
        if done_event.is_set():
            return
        idx = await index_queue.get()
        try:
            from rllm.agents.agent import BaseAgent

            engine.envs[idx] = SWEEnv.from_dict({**task, **engine.env_args})
            engine.agents[idx] = SWEAgent(**engine.agent_args)
            assert isinstance(engine.agents[idx], BaseAgent)
            engine.agents[idx].trajectory.task = task

            traj = await engine.run_agent_trajectory_with_retry(
                idx=idx,
                mode="Text",
            )
            traj.task = task
            traj.info["chat_completions"] = engine.agents[idx].chat_completions
            _process_trajectory(traj)
        except Exception as exc:
            logger.error(f"Trajectory for {task.get('instance_id', '?')} failed: {exc}")
        finally:
            # Close env to release sandbox pod
            env = engine.envs[idx] if idx < len(engine.envs) else None
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
                engine.envs[idx] = None
            await index_queue.put(idx)

    def _task_done(t: asyncio.Task) -> None:
        pending.discard(t)
        if not t.cancelled():
            exc = t.exception()
            if exc:
                logger.error(f"Task raised: {exc!r}")

    try:
        for task in itertools.cycle(tasks):
            if done_event.is_set():
                break

            # Back-pressure: wait for a slot before creating the next task
            await semaphore.acquire()
            if done_event.is_set():
                semaphore.release()
                break

            async def _wrapped(t=task):
                try:
                    await _run_one(t)
                finally:
                    semaphore.release()

            tsk = asyncio.create_task(_wrapped())
            pending.add(tsk)
            tsk.add_done_callback(_task_done)

        # Wait for in-flight trajectories to finish
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    finally:
        engine.shutdown()

        # ------------------------------------------------------------------
        # Final save
        # ------------------------------------------------------------------
        if is_dpo:
            n_pos_final = sum(len(v) for v in pos_cache.values())
            n_neg_final = sum(len(v) for v in neg_cache.values())
            pairs = pair_dpo_data(pos_cache, neg_cache)
            n_paired = len(pairs)
            n_unpaired = n_pos_final - n_paired
            print(f"\nDPO summary: {n_pos_final} positives, {n_neg_final} negatives -> {n_paired} pairs ({n_unpaired} positives had no matching negative)")
            if pairs:
                save_dpo_parquet(pairs, args.output)
            else:
                print("WARNING: no DPO pairs produced (positives without matching negatives?)")
        else:
            if collected:
                save_rejection_checkpoint(collected, args.output)
                print(f"\nFinal save: {len(collected)} trajectories -> {args.output}")

        # ------------------------------------------------------------------
        # Stats
        # ------------------------------------------------------------------
        pass_rate = n_pass / max(n_attempts, 1)
        print(f"\nDone. passes={n_pass}  fails={n_fail}  attempts={n_attempts}  pass_rate={pass_rate:.2%}")
        print("Termination reason distribution:")
        for reason, count in reason_counter.most_common():
            print(f"  {reason:20s}  {count:>6d}  ({count / max(n_attempts, 1):.1%})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect SWE trajectories via rejection or DPO sampling")

    # Model / API
    p.add_argument("--model", required=True, help="Model name (e.g. gpt-4o, claude-opus-4-6)")
    p.add_argument("--base_url", default="https://api.openai.com/v1", help="OpenAI-compatible API base URL")
    p.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""), help="API key (default: $OPENAI_API_KEY)")
    p.add_argument("--tokenizer", default=None, help="HuggingFace tokenizer name/path (default: same as --model)")
    p.add_argument("--backend", choices=["openai", "vllm", "sglang"], default="openai", help="openai: chat completions (GPT-4o, Claude, etc.); vllm/sglang: completion API (local server)")

    # Data
    p.add_argument("--data", default="data/swe/R2E_Gym_Subset.parquet", help="Input task parquet")
    p.add_argument("--output", default="data/swe/rejection_sample.parquet", help="Output parquet path (also checkpoint)")

    # Mode
    p.add_argument("--mode", choices=["rejection", "dpo"], default="rejection", help="rejection: keep only passing trajectories (SFT data); dpo: cache all trajectories and emit chosen/rejected pairs")

    # Scaffold
    p.add_argument("--scaffold", choices=["r2egym", "sweagent"], default="r2egym", help="Tool scaffold to use (must match agent and env)")

    # Sampling
    p.add_argument("--target", type=int, default=5000, help="Stop after collecting this many *passing* trajectories")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    p.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p (default: 1.0, no filtering)")
    p.add_argument("--top_k", type=int, default=-1, help="Top-k sampling (default: -1, disabled)")
    p.add_argument("--disable_thinking", action="store_true", help="Disable thinking mode — inject empty <think></think> in generation prompt")

    # Agent / env
    p.add_argument("--max_steps", type=int, default=50, help="Max agent steps per trajectory")
    p.add_argument("--max_response_length", type=int, default=32768, help="Cumulative response token budget per trajectory (engine manages per-call max_tokens)")
    p.add_argument("--max_prompt_length", type=int, default=131072, help="Max prompt length in tokens")
    p.add_argument("--step_timeout", type=int, default=90, help="Per-step sandbox timeout (s)")
    p.add_argument("--reward_timeout", type=int, default=300, help="Reward computation timeout (s)")
    p.add_argument("--trajectory_timeout", type=int, default=1200, help="Total trajectory wall-clock timeout (s)")

    # Concurrency / checkpointing
    p.add_argument("--n_parallel", type=int, default=128, help="Max concurrent trajectories")
    p.add_argument("--checkpoint_interval", type=int, default=50, help="(rejection mode) checkpoint every N new passes")

    # Upload
    p.add_argument("--upload", action="store_true", help="Upload results to expdata service after completion")
    p.add_argument("--upload_url", type=str, default="http://expdata.default.svc.cluster.local:8502", help="Expdata service URL")

    args = p.parse_args()

    # Default tokenizer to model name
    if args.tokenizer is None:
        args.tokenizer = args.model

    return args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    args = parse_args()

    if not args.api_key:
        print("ERROR: --api_key not set and $OPENAI_API_KEY is empty", file=sys.stderr)
        sys.exit(1)

    tasks = load_tasks(args.data)
    if not tasks:
        print("ERROR: no tasks loaded", file=sys.stderr)
        sys.exit(1)

    asyncio.run(sample_loop(tasks, args))

    # ---- Upload to expdata service ----
    if args.upload:
        try:
            from utils.expdata.client import ExperimentUploader

            uploader = ExperimentUploader(args.upload_url)
            exp_name = f"collection-{Path(args.model).name}-{args.mode}"
            exp_id = uploader.create_experiment(
                name=exp_name,
                type="collection",
                model=args.model,
                backend=args.backend,
                scaffold=args.scaffold,
                dataset=args.data,
                mode=args.mode,
            )
            logging.getLogger("expdata_client").info(f"Created experiment {exp_id} on {args.upload_url}")

            if Path(args.output).exists():
                uploader.upload_collection_parquet(exp_id, args.output)

            uploader.mark_completed(exp_id, {"output": args.output, "mode": args.mode})
            logging.getLogger("expdata_client").info(f"Upload complete: experiment {exp_id}")
        except Exception as e:
            logging.getLogger("expdata_client").warning(f"Upload failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
