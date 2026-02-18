#!/usr/bin/env python3
"""
Standalone SWE-Bench Evaluation Script

Evaluates an agent on SWE-Bench tasks using AgentExecutionEngine directly,
bypassing the PPO training pipeline entirely. No Ray, no verl, no FSDP needed —
just a running vLLM (or OpenAI-compatible) inference server.

All samples are evaluated without any prompt-length filtering.

Usage:
    # Greedy eval (n=1) with local vLLM
    python scripts/swe_eval_standalone.py \
        --model /path/to/model \
        --data data/swe/SWE_Bench_Verified.parquet

    # pass@5 with sampling
    python scripts/swe_eval_standalone.py \
        --model /path/to/model \
        --data data/swe/SWE_Bench_Verified.parquet \
        --n_samples 5 --temperature 1.0

    # Using DatasetRegistry (from prepare_swe_data.py)
    python scripts/swe_eval_standalone.py \
        --model /path/to/model \
        --dataset_name SWE_Bench_Verified --dataset_split test
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path

from transformers import AutoTokenizer

from rllm.agents.swe_agent import SWEAgent
from rllm.data.dataset import Dataset, DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.swe.swe import SWEEnv

logger = logging.getLogger(__name__)


def load_tasks(args) -> list[dict]:
    """Load task dicts from parquet file or DatasetRegistry."""
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


def save_results_jsonl(trajectories: list, output_path: str):
    """Save results in swe_report.py-compatible JSONL format."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        for traj in trajectories:
            task = traj.task or {}
            record = {
                "uid": task.get("_uid", traj.uid),
                "data_source": task.get("data_source", "swe"),
                "reward": float(traj.reward) if traj.reward is not None else 0.0,
                "sample_idx": task.get("_sample_idx", 0),
                "n_samples": task.get("_n_samples", 1),
                "instance_id": task.get("instance_id", ""),
                "repo": task.get("repo", task.get("repo_name", "")),
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(trajectories)} results to {output_path}")


def run_report(output_path: str):
    """Run swe_report.py on the results."""
    script_dir = Path(__file__).parent
    report_script = script_dir / "swe_report.py"
    if report_script.exists():
        print()
        subprocess.run([sys.executable, str(report_script), output_path], check=False)
    else:
        print(f"Warning: report script not found at {report_script}")


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone SWE-Bench Evaluation")

    # Data source (choose one)
    data_group = parser.add_argument_group("data source")
    data_group.add_argument("--data", type=str, default=None, help="Path to parquet/json/jsonl data file")
    data_group.add_argument("--dataset_name", type=str, default=None, help="DatasetRegistry name (e.g., SWE_Bench_Verified)")
    data_group.add_argument("--dataset_split", type=str, default="test", help="DatasetRegistry split (default: test)")

    # Model / inference
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model", type=str, required=True, help="Model name or path (for tokenizer + API model name)")
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

    # Concurrency
    parser.add_argument("--n_parallel", type=int, default=48, help="Max concurrent agent-env trajectories (default: 48)")
    parser.add_argument("--retry_limit", type=int, default=3, help="Retries per failed trajectory (default: 3)")

    # Output
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path (default: auto-generated)")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Output directory (default: eval_results)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve temperature
    if args.temperature is None:
        args.temperature = 1.0 if args.n_samples > 1 else 0.0

    # Load tokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load and prepare tasks
    base_tasks = load_tasks(args)
    tasks = repeat_tasks_for_pass_k(base_tasks, args.n_samples)
    print(f"Total trajectories to run: {len(tasks)} ({len(base_tasks)} tasks x {args.n_samples} samples)")

    # Build engine
    engine = AgentExecutionEngine(
        agent_class=SWEAgent,
        env_class=SWEEnv,
        agent_args={"scaffold": args.scaffold},
        env_args={
            "scaffold": args.scaffold,
            "step_timeout": args.step_timeout,
            "reward_timeout": args.reward_timeout,
            "verbose": False,
        },
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params={
            "model": args.model,
            "temperature": args.temperature,
        },
        rollout_engine_args={
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

    # Run evaluation
    print(f"Starting evaluation with {args.n_parallel} parallel agents...")
    print(f"  Model: {args.model}")
    print(f"  API: {args.base_url}")
    print(f"  Scaffold: {args.scaffold}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max steps: {args.max_steps}")
    print()

    trajectories = asyncio.run(engine.execute_tasks(tasks))

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        model_tag = Path(args.model).name
        sample_tag = f"n{args.n_samples}"
        output_path = os.path.join(args.output_dir, f"{model_tag}_{sample_tag}.jsonl")

    # Save and report
    save_results_jsonl(trajectories, output_path)
    run_report(output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
