#!/usr/bin/env python3
"""
Standalone SWE-Bench / R2E-Gym Evaluation Script

Evaluates an agent on SWE-Bench or R2E-Gym tasks using AgentExecutionEngine
directly, bypassing the PPO training pipeline entirely.  No Ray, no verl,
no FSDP needed — just a running vLLM (or OpenAI-compatible) inference server.

All samples are evaluated without any prompt-length filtering.

Usage:
    # Greedy eval (n=1) with local vLLM — SWE-Bench Verified
    python scripts/swe_eval_standalone.py \
        --model /path/to/model \
        --data data/swe/SWE_Bench_Verified.parquet

    # Greedy eval — R2E-Gym Subset
    python scripts/swe_eval_standalone.py \
        --model /path/to/model \
        --data data/R2E-Gym/R2E-Gym-Subset-train.parquet

    # pass@5 with sampling
    python scripts/swe_eval_standalone.py \
        --model /path/to/model \
        --data data/swe/SWE_Bench_Verified.parquet \
        --n_samples 5 --temperature 1.0

    # Dry run (SWE-Bench): harness on unmodified code + fault tracing
    python scripts/swe_eval_standalone.py \
        --dry_run --normalize_pytest \
        --data data/swe/SWE_Bench_Verified.parquet

    # Dry run (R2E-Gym): same interface, different dataset
    python scripts/swe_eval_standalone.py \
        --dry_run --normalize_pytest \
        --data data/R2E-Gym/R2E-Gym-Subset-train.parquet
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rllm.data.dataset import Dataset, DatasetRegistry
from rllm.environments.swe.swe import SWEEnv
from rllm.environments.swe.trace import normalize_task

logger = logging.getLogger(__name__)


def load_tasks(args) -> list[dict]:
    """Load task dicts from parquet file or DatasetRegistry.

    Applies ``normalize_task`` so that all dataset fields (``docker_image``,
    ``instance_id``, ``patch``, …) are available as top-level keys regardless
    of whether the source is SWE-Bench (verl-wrapped) or R2E-Gym (flat).
    """
    if args.data:
        ds = Dataset.load_data(args.data)
        tasks = ds.get_data()
        print(f"Loaded {len(tasks)} tasks from {args.data}")
    elif args.dataset_name:
        ds = DatasetRegistry.load_dataset(args.dataset_name, args.dataset_split)
        if ds is None:
            print(f"Error: dataset '{args.dataset_name}' split '{args.dataset_split}' not found in registry.", file=sys.stderr)
            print("Run `python examples/swe/prepare_swe_data.py` first.", file=sys.stderr)
            sys.exit(1)
        tasks = ds.get_data()
        print(f"Loaded {len(tasks)} tasks from registry: {args.dataset_name}/{args.dataset_split}")
    else:
        print("Error: must specify --data or --dataset_name", file=sys.stderr)
        sys.exit(1)

    # Normalise: flatten SWE-bench extra_info wrapper, synthesise instance_id
    # for R2E, etc.  After this point every task has a consistent schema.
    tasks = [normalize_task(t) for t in tasks]

    # Ensure every task has an instance_id (R2E doesn't ship one natively)
    for task in tasks:
        if "instance_id" not in task or not task["instance_id"]:
            repo = task.get("repo", task.get("repo_name", "unknown"))
            commit = task.get("base_commit", task.get("commit_hash", ""))[:12]
            task["instance_id"] = f"{repo}__{commit}"

    return tasks


def repeat_tasks_for_pass_k(tasks: list[dict], n_samples: int) -> list[dict]:
    """Repeat each task n_samples times with stable uid per base task.

    Tasks sharing the same uid are grouped for pass@k computation.
    Adjacent layout: [task0_s0, task0_s1, ..., task1_s0, task1_s1, ...]
    """
    if n_samples <= 1:
        for task in tasks:
            task["_uid"] = str(uuid.uuid4())
            task["_sample_idx"] = 0
            task["_n_samples"] = 1
        return tasks

    repeated = []
    for task in tasks:
        uid = str(uuid.uuid4())
        for si in range(n_samples):
            t = task.copy()
            t["_uid"] = uid
            t["_sample_idx"] = si
            t["_n_samples"] = n_samples
            repeated.append(t)
    return repeated


def save_results_jsonl(results: list[dict], output_path: str):
    """Save results in swe_report.py-compatible JSONL format."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(results)} results to {output_path}")


def save_test_outputs(test_outputs: dict[str, str], output_dir: str):
    """Save per-instance raw test outputs for analysis."""
    outputs_dir = os.path.join(output_dir, "test_outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    for instance_id, output in test_outputs.items():
        safe_name = instance_id.replace("/", "__")
        with open(os.path.join(outputs_dir, f"{safe_name}.txt"), "w") as f:
            f.write(output)
    print(f"Saved {len(test_outputs)} test output files to {outputs_dir}")


def save_fault_traces(traces_map: dict[str, list], output_dir: str):
    """Save per-instance fault trace JSON files."""
    traces_dir = os.path.join(output_dir, "fault_traces")
    os.makedirs(traces_dir, exist_ok=True)
    saved = 0
    for instance_id, traces in traces_map.items():
        if not traces:
            continue
        safe_name = instance_id.replace("/", "__")
        with open(os.path.join(traces_dir, f"{safe_name}.json"), "w") as f:
            json.dump(traces, f, indent=2)
        saved += 1
    print(f"Saved {saved} fault trace files to {traces_dir}")


def save_trajectories(trajectories: list, output_dir: str):
    """Save per-trajectory chat completions in the same JSONL format as RL training."""
    save_dir = os.path.join(output_dir, "chat_completions")
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, "eval.jsonl")
    with open(output_path, "w") as f:
        for traj in trajectories:
            chat = traj.info.get("chat_completions", [])
            f.write(json.dumps(chat) + "\n")
    print(f"Saved {len(trajectories)} chat completions to {output_path}")


def results_from_trajectories(trajectories: list) -> list[dict]:
    """Convert Trajectory objects to swe_report.py-compatible dicts."""
    records = []
    for traj in trajectories:
        task = traj.task or {}
        records.append({
            "uid": task.get("_uid", traj.uid),
            "data_source": task.get("data_source", "swe"),
            "reward": float(traj.reward) if traj.reward is not None else 0.0,
            "sample_idx": task.get("_sample_idx", 0),
            "n_samples": task.get("_n_samples", 1),
            "instance_id": task.get("instance_id", ""),
            "repo": task.get("repo", task.get("repo_name", "")),
            "termination_reason": traj.info.get("termination_reason", "UNKNOWN"),
        })
    return records


def run_report(output_path: str):
    """Run swe_report.py on the results."""
    script_dir = Path(__file__).parent
    report_script = script_dir / "swe_report.py"
    if report_script.exists():
        print()
        subprocess.run([sys.executable, str(report_script), output_path], check=False)
    else:
        print(f"Warning: report script not found at {report_script}")


# =========================================================================
# Dry-run mode: run harness on unmodified code, no model needed
# =========================================================================

async def run_dry_run(tasks: list[dict], env_args: dict, n_parallel: int, output_dir: str):
    """Run SWE-bench / R2E harness on unmodified code to capture baseline test outputs.

    No agent, no model — just reset the sandbox and run tests.
    Works identically for both SWE-Bench Verified and R2E-Gym datasets.
    """
    from rllm.environments.swe.reward import run_tests_with_output
    from rllm.environments.swe.trace import (
        aggregate_traces,
        find_modified_callables_from_task,
        instrument_sandbox,
        parse_fault_traces,
    )

    semaphore = asyncio.Semaphore(n_parallel)
    executor = ThreadPoolExecutor(max_workers=n_parallel)
    loop = asyncio.get_event_loop()

    completed = 0
    total = len(tasks)
    results = []
    test_outputs = {}

    def _run_single_task(task):
        """Synchronous: create env, reset, run tests, close."""
        env = SWEEnv.from_dict({**task, **env_args})
        try:
            env.reset()

            # Discover modified callables from patch, then instrument sandbox
            modified_callables = []
            try:
                modified_callables = find_modified_callables_from_task(task)
                if modified_callables:
                    modified_callables = instrument_sandbox(env, modified_callables)
            except Exception as e:
                logger.warning(f"Fault trace instrumentation failed: {e}")

            reward, raw_output = run_tests_with_output(
                session=env.session,
                ds=env.entry,
                repo_path=env.repo_path,
                alt_path=env.alt_path,
                timeout=env.reward_timeout,
            )

            # Parse fault traces from output
            fault_traces = []
            if modified_callables:
                traces = parse_fault_traces(raw_output, modified_callables, env.repo_path)
                fault_traces = aggregate_traces(traces)

            return reward, raw_output, fault_traces
        finally:
            env.close()

    async def sem_wrapper(idx, task):
        nonlocal completed
        async with semaphore:
            try:
                reward, raw_output, fault_traces = await loop.run_in_executor(executor, _run_single_task, task)
            except Exception as e:
                logger.error(f"Task {idx} ({task.get('instance_id', '?')}) failed: {e}")
                reward, raw_output, fault_traces = 0.0, f"ERROR: {e}", []

            completed += 1
            instance_id = task.get("instance_id", f"task_{idx}")
            status = "PASS" if reward >= 1.0 else "FAIL"
            print(f"[{completed}/{total}] {instance_id}: {status}")

            record = {
                "uid": task.get("_uid", str(uuid.uuid4())),
                "data_source": task.get("data_source", "swe"),
                "reward": float(reward),
                "sample_idx": task.get("_sample_idx", 0),
                "n_samples": task.get("_n_samples", 1),
                "instance_id": instance_id,
                "repo": task.get("repo", task.get("repo_name", "")),
            }
            return record, instance_id, raw_output, fault_traces

    all_results = await asyncio.gather(*[sem_wrapper(i, t) for i, t in enumerate(tasks)])

    fault_traces_map = {}
    for record, instance_id, raw_output, fault_traces in all_results:
        results.append(record)
        test_outputs[instance_id] = raw_output
        if fault_traces:
            fault_traces_map[instance_id] = fault_traces

    executor.shutdown(wait=False)
    return results, test_outputs, fault_traces_map


# =========================================================================
# Regular eval mode: agent + environment interaction
# =========================================================================

def run_agent_eval(args, tasks):
    """Run agentic evaluation using AgentExecutionEngine."""
    from transformers import AutoTokenizer

    from rllm.agents.swe_agent import SWEAgent
    from rllm.engine.agent_execution_engine import AgentExecutionEngine

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    engine = AgentExecutionEngine(
        agent_class=SWEAgent,
        env_class=SWEEnv,
        agent_args={"scaffold": args.scaffold},
        env_args={
            "scaffold": args.scaffold,
            "step_timeout": args.step_timeout,
            "reward_timeout": args.reward_timeout,
            "normalize_pytest": args.normalize_pytest,
            "verbose": False,
        },
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params={
            "temperature": args.temperature,
        },
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
        retry_limit=args.retry_limit,
        overlong_filter=False,
    )

    print(f"Starting evaluation with {args.n_parallel} parallel agents...")
    print(f"  Model: {args.model}")
    print(f"  API: {args.base_url}")
    print(f"  Scaffold: {args.scaffold}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Normalize pytest: {args.normalize_pytest}")
    print()

    trajectories = asyncio.run(engine.execute_tasks(tasks))
    return results_from_trajectories(trajectories), trajectories


# =========================================================================
# P2A Localization Analysis (post-eval)
# =========================================================================

def run_localization_analysis(trajectories, bonus_map_dir: str, tracking_mode: str, output_dir: str):
    """Run Section 6.3 localization analysis on eval trajectories.

    Uses the same analysis functions as analyze_localization.py but operates
    directly on Trajectory objects instead of JSONL files.
    """
    from rllm.trainer.verl.p2a import (
        BonusMapStore,
        match_reads_to_callgraph,
        parse_read_actions,
    )

    bonus_store = BonusMapStore(bonus_map_dir)
    per_instance = []

    for traj in trajectories:
        task = traj.task or {}
        instance_id = task.get("instance_id", "")
        bonus_map = bonus_store.get(instance_id) if instance_id else None

        # Extract assistant responses from Trajectory steps
        responses = [step.model_response for step in traj.steps if step.model_response]

        n_steps = len(responses)
        n_on_graph = 0
        n_view_steps = 0  # steps containing at least one read action
        first_root_cause_step = -1
        min_distance = float("inf")
        distances = []

        for step_i, response_text in enumerate(responses):
            reads = parse_read_actions(response_text, tracking_mode=tracking_mode)
            if not reads or bonus_map is None:
                if reads:
                    n_view_steps += 1
                continue
            n_view_steps += 1
            distance = match_reads_to_callgraph(reads, bonus_map)
            if distance >= 0:
                n_on_graph += 1
                distances.append(distance)
                min_distance = min(min_distance, distance)
                if distance < 1e-6 and first_root_cause_step < 0:
                    first_root_cause_step = step_i

        if min_distance == float("inf"):
            min_distance = -1.0

        per_instance.append({
            "instance_id": instance_id,
            "reward": float(traj.reward) if traj.reward is not None else 0.0,
            "n_steps": n_steps,
            "n_view_steps": n_view_steps,
            "n_on_graph": n_on_graph,
            "first_root_cause_step": first_root_cause_step,
            "min_distance": min_distance,
            "distances": distances,
            "has_bonus_map": bonus_map is not None and bonus_map.get("traceable", False),
        })

    # Aggregate metrics
    total_steps = sum(r["n_steps"] for r in per_instance)
    total_view_steps = sum(r["n_view_steps"] for r in per_instance)
    total_on_graph = sum(r["n_on_graph"] for r in per_instance)
    root_cause_hits = sum(1 for r in per_instance if r["first_root_cause_step"] >= 0)
    steps_to_root = [r["first_root_cause_step"] for r in per_instance if r["first_root_cause_step"] >= 0]
    all_distances = [d for r in per_instance for d in r["distances"]]
    n_traj = len(per_instance)

    metrics = {
        "n_trajectories": n_traj,
        "total_steps": total_steps,
        "total_view_steps": total_view_steps,
        "total_on_graph_steps": total_on_graph,
        "on_graph_read_ratio": total_on_graph / max(total_steps, 1),
        "on_graph_view_density": total_on_graph / max(total_view_steps, 1),
        "root_cause_coverage": root_cause_hits / max(n_traj, 1),
        "root_cause_hits": root_cause_hits,
    }

    if steps_to_root:
        import numpy as np
        arr = np.array(steps_to_root)
        metrics["avg_steps_to_root_cause"] = float(arr.mean())
        metrics["median_steps_to_root_cause"] = float(np.median(arr))
    else:
        metrics["avg_steps_to_root_cause"] = -1.0

    if all_distances:
        import numpy as np
        d_arr = np.array(all_distances)
        metrics["distance_mean"] = float(d_arr.mean())
        metrics["distance_std"] = float(d_arr.std())

    # Print results
    print()
    print("=" * 60)
    print("  P2A Localization Analysis")
    print("=" * 60)
    print(f"  Tracking mode: {tracking_mode}")
    print(f"  Trajectories: {n_traj}")
    print(f"  Total steps: {total_steps}")
    print(f"  View steps (with reads): {total_view_steps}")
    print(f"  On-graph steps: {total_on_graph}")
    print()
    print(f"  [Metric 1] On-graph Read Ratio: {metrics['on_graph_read_ratio']:.4f}")
    print(f"  [Metric 1b] On-graph View Density: {metrics['on_graph_view_density']:.4f}")
    print(f"  [Metric 2] Avg Steps to Root Cause: {metrics['avg_steps_to_root_cause']:.2f}")
    if "median_steps_to_root_cause" in metrics:
        print(f"             Median: {metrics['median_steps_to_root_cause']:.1f}")
    print(f"  [Metric 3] Root Cause Coverage: {metrics['root_cause_coverage']:.4f} ({root_cause_hits}/{n_traj})")
    if "distance_mean" in metrics:
        print(f"  Mean distance: {metrics['distance_mean']:.4f} +/- {metrics['distance_std']:.4f}")
    print("=" * 60)

    # Save per-instance analysis
    loc_output = os.path.join(output_dir, "localization_analysis.json")
    os.makedirs(output_dir, exist_ok=True)
    save_data = {"aggregate": metrics, "per_instance": [
        {k: v for k, v in r.items() if k != "distances"} for r in per_instance
    ]}
    with open(loc_output, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  Saved to {loc_output}")

    return metrics


# =========================================================================
# CLI
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Standalone SWE-Bench / R2E-Gym Evaluation")

    # Data source (choose one)
    data_group = parser.add_argument_group("data source")
    data_group.add_argument("--data", type=str, default=None, help="Path to parquet/json/jsonl data file")
    data_group.add_argument("--dataset_name", type=str, default=None, help="DatasetRegistry name (e.g., SWE_Bench_Verified)")
    data_group.add_argument("--dataset_split", type=str, default="test", help="DatasetRegistry split (default: test)")

    # Model / inference (not required for dry_run)
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model", type=str, default=None, help="Model name or path (for tokenizer + API model name)")
    model_group.add_argument("--base_url", type=str, default="http://localhost:8000/v1", help="vLLM / OpenAI-compatible API base URL")
    model_group.add_argument("--api_key", type=str, default="EMPTY", help="API key (default: EMPTY)")

    # Sampling
    sample_group = parser.add_argument_group("sampling")
    sample_group.add_argument("--n_samples", type=int, default=1, help="Number of samples per task for pass@k (default: 1)")
    sample_group.add_argument("--temperature", type=float, default=None, help="Sampling temperature (default: 0 for n=1, 1.0 for n>1)")

    # Agent / Env
    agent_group = parser.add_argument_group("agent/env")
    agent_group.add_argument("--scaffold", type=str, default="r2egym", choices=["r2egym", "sweagent"], help="Scaffold type (default: r2egym)")
    agent_group.add_argument("--max_steps", type=int, default=100, help="Max agent steps per task (default: 100)")
    agent_group.add_argument("--max_prompt_length", type=int, default=131072, help="Max prompt length in tokens (default: 131072)")
    agent_group.add_argument("--max_response_length", type=int, default=32768, help="Max response length in tokens (default: 32768)")
    agent_group.add_argument("--step_timeout", type=int, default=90, help="Per-action sandbox timeout in seconds (default: 90)")
    agent_group.add_argument("--reward_timeout", type=int, default=300, help="Reward computation timeout in seconds (default: 300)")
    agent_group.add_argument("--trajectory_timeout", type=int, default=1200, help="Total trajectory wall-time timeout in seconds (default: 1200)")

    # Special modes
    mode_group = parser.add_argument_group("special modes")
    mode_group.add_argument("--dry_run", action="store_true", help="Run harness on unmodified code without any model. Captures baseline test output for analysis.")
    mode_group.add_argument("--normalize_pytest", action="store_true", help="Standardize pytest args: ensure -rA and --tb=short are present in the test script.")

    # P2A localization analysis
    p2a_group = parser.add_argument_group("P2A analysis")
    p2a_group.add_argument("--p2a_bonus_map_dir", type=str, default=None, help="Bonus map dir for P2A localization analysis (enables analysis)")
    p2a_group.add_argument("--p2a_tracking_mode", type=str, default="view_only", choices=["view_only", "view_and_bash"], help="Tracking mode for localization analysis")

    # Concurrency
    parser.add_argument("--n_parallel", type=int, default=48, help="Max concurrent agent-env trajectories (default: 48)")
    parser.add_argument("--retry_limit", type=int, default=3, help="Retries per failed trajectory (default: 3)")
    parser.add_argument("--max_tasks", type=int, default=None, help="Max number of tasks to evaluate (default: all)")

    # Output
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path (default: auto-generated)")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Output directory (default: eval_results)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate args
    if not args.dry_run and not args.model:
        print("Error: --model is required unless --dry_run is set", file=sys.stderr)
        sys.exit(1)

    # Resolve temperature
    if args.temperature is None:
        # args.temperature = 1.0 if args.n_samples > 1 else 0.0
        args.temperature = 1.0 # use 1.0 for all runs. Introduce some randomness.

    # Load and prepare tasks
    base_tasks = load_tasks(args)
    if args.max_tasks:
        base_tasks = base_tasks[:args.max_tasks]
        print(f"Limiting to first {args.max_tasks} tasks")

    if args.dry_run:
        # Dry run: no model, no agent, just harness on unmodified code
        tasks = base_tasks
        for task in tasks:
            task["_uid"] = str(uuid.uuid4())
            task["_sample_idx"] = 0
            task["_n_samples"] = 1

        env_args = {
            "scaffold": args.scaffold,
            "step_timeout": args.step_timeout,
            "reward_timeout": args.reward_timeout,
            "normalize_pytest": args.normalize_pytest,
            "verbose": False,
        }

        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        print(f"DRY RUN: running harness on {len(tasks)} unmodified instances...")
        print(f"  Normalize pytest: {args.normalize_pytest}")
        print(f"  Parallel: {args.n_parallel}")
        print()

        results, test_outputs, fault_traces_map = asyncio.run(
            run_dry_run(tasks, env_args, args.n_parallel, output_dir)
        )

        # Save results
        output_path = args.output or os.path.join(output_dir, "dry_run.jsonl")
        save_results_jsonl(results, output_path)
        save_test_outputs(test_outputs, output_dir)
        if fault_traces_map:
            save_fault_traces(fault_traces_map, output_dir)
        run_report(output_path)

    else:
        # Regular agent evaluation
        tasks = repeat_tasks_for_pass_k(base_tasks, args.n_samples)
        print(f"Total trajectories to run: {len(tasks)} ({len(base_tasks)} tasks x {args.n_samples} samples)")

        results, trajectories = run_agent_eval(args, tasks)

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            model_tag = Path(args.model).name
            sample_tag = f"n{args.n_samples}"
            output_path = os.path.join(args.output_dir, f"{model_tag}_{sample_tag}.jsonl")

        save_results_jsonl(results, output_path)
        save_trajectories(trajectories, args.output_dir)
        run_report(output_path)

        # P2A localization analysis (if bonus maps provided)
        if args.p2a_bonus_map_dir:
            run_localization_analysis(
                trajectories,
                bonus_map_dir=args.p2a_bonus_map_dir,
                tracking_mode=args.p2a_tracking_mode,
                output_dir=args.output_dir,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
