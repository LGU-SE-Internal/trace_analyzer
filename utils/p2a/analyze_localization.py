#!/usr/bin/env python3
"""Analyze fault localization quality of SWE agent trajectories.

Computes the three Section 6.3 metrics from the P2A paper:
  1. On-graph Read Ratio: fraction of observation steps hitting golden call graph
  2. Avg Steps to Root Cause: step index of first patched callable hit
  3. Root Cause Coverage: fraction of trajectories reading >= 1 patched callable

Works on:
  - Training rollout logs (chat_completions/*.jsonl from trainer)
  - Eval results (from swe_eval_standalone.py agent mode with --save_trajectories)
  - Dry-run fault traces (from swe_eval_standalone.py --dry_run)

Usage:
    # Analyze training rollout (requires bonus maps)
    python -m utils.p2a.analyze_localization \
        --trajectories /path/to/chat_completions/10.jsonl \
        --bonus_map_dir data/swe/bonus_maps \
        --tracking_mode view_and_bash

    # Analyze eval trajectories
    python -m utils.p2a.analyze_localization \
        --trajectories /path/to/eval/trajectories.jsonl \
        --bonus_map_dir data/swe/bonus_maps

    # Analyze with W&B logging
    python -m utils.p2a.analyze_localization \
        --trajectories /path/to/trajectories.jsonl \
        --bonus_map_dir data/swe/bonus_maps \
        --wandb_project xujunjielong \
        --wandb_run_name p2a-localization-analysis
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rllm.trainer.verl.p2a import (
    BonusMapStore,
    match_reads_to_callgraph,
    parse_read_actions,
)


def load_trajectories(path: str) -> list[dict]:
    """Load trajectory JSONL. Each line is one trajectory (list of messages or a dict)."""
    trajectories = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trajectories.append(json.loads(line))
    return trajectories


def extract_assistant_responses(trajectory) -> list[str]:
    """Extract assistant response texts from a trajectory.

    Handles multiple formats:
    - List of messages (chat completion format): [{"role": "assistant", "content": "..."}]
    - Dict with "steps" key: {"steps": [{"response": "..."}]}
    - Dict with "messages" key: {"messages": [...]}
    """
    if isinstance(trajectory, list):
        # Chat completion format: list of messages
        return [
            msg.get("content", "")
            for msg in trajectory
            if msg.get("role") == "assistant" and msg.get("content")
        ]
    elif isinstance(trajectory, dict):
        if "steps" in trajectory:
            return [s.get("response", "") for s in trajectory["steps"]]
        if "messages" in trajectory:
            return [
                msg.get("content", "")
                for msg in trajectory["messages"]
                if msg.get("role") == "assistant" and msg.get("content")
            ]
        # Single trajectory with "response" key
        if "response" in trajectory:
            return [trajectory["response"]]
    return []


def extract_instance_id(trajectory) -> str:
    """Try to extract instance_id from trajectory metadata."""
    if isinstance(trajectory, dict):
        for key in ("instance_id", "task_id", "id"):
            if key in trajectory:
                return str(trajectory[key])
        # Check nested metadata
        meta = trajectory.get("metadata", {})
        if isinstance(meta, dict):
            for key in ("instance_id", "task_id"):
                if key in meta:
                    return str(meta[key])
        extra = trajectory.get("extra_info", {})
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except (json.JSONDecodeError, TypeError):
                extra = {}
        if isinstance(extra, dict) and "instance_id" in extra:
            return str(extra["instance_id"])
    return ""


def analyze_trajectory(
    responses: list[str],
    bonus_map: dict | None,
    tracking_mode: str = "view_only",
) -> dict:
    """Analyze a single trajectory for localization metrics.

    Returns dict with:
        - n_steps: total observation steps
        - n_on_graph: steps hitting call graph
        - first_root_cause_step: step index of first d=0 hit (-1 if none)
        - min_distance: minimum distance seen (-1.0 if no on-graph hits)
        - distances: list of all on-graph distances
    """
    n_steps = len(responses)
    n_on_graph = 0
    n_view_steps = 0  # steps containing at least one read action
    first_root_cause_step = -1
    min_distance = float("inf")
    distances = []

    for step_i, response_text in enumerate(responses):
        reads = parse_read_actions(response_text, tracking_mode=tracking_mode)
        if not reads:
            continue

        n_view_steps += 1

        if bonus_map is None:
            continue

        distance = match_reads_to_callgraph(reads, bonus_map)
        if distance >= 0:
            n_on_graph += 1
            distances.append(distance)
            min_distance = min(min_distance, distance)
            if distance < 1e-6 and first_root_cause_step < 0:
                first_root_cause_step = step_i

    if min_distance == float("inf"):
        min_distance = -1.0

    return {
        "n_steps": n_steps,
        "n_view_steps": n_view_steps,
        "n_on_graph": n_on_graph,
        "first_root_cause_step": first_root_cause_step,
        "min_distance": min_distance,
        "distances": distances,
    }


def compute_aggregate_metrics(results: list[dict]) -> dict:
    """Compute aggregate localization metrics across all trajectories."""
    total_steps = 0
    total_view_steps = 0
    total_on_graph = 0
    root_cause_hits = 0
    steps_to_root_cause = []
    all_distances = []

    for r in results:
        total_steps += r["n_steps"]
        total_view_steps += r["n_view_steps"]
        total_on_graph += r["n_on_graph"]
        all_distances.extend(r["distances"])
        if r["first_root_cause_step"] >= 0:
            root_cause_hits += 1
            steps_to_root_cause.append(r["first_root_cause_step"])

    n_trajectories = len(results)
    metrics = {
        "n_trajectories": n_trajectories,
        "total_steps": total_steps,
        "total_view_steps": total_view_steps,
        "total_on_graph_steps": total_on_graph,
    }

    # Section 6.3 Metric 1: On-graph Read Ratio
    metrics["on_graph_read_ratio"] = total_on_graph / max(total_steps, 1)

    # Section 6.3 Metric 1b: On-graph View Density (on-graph / view steps only)
    metrics["on_graph_view_density"] = total_on_graph / max(total_view_steps, 1)

    # Section 6.3 Metric 2: Avg Steps to Root Cause
    if steps_to_root_cause:
        arr = np.array(steps_to_root_cause)
        metrics["avg_steps_to_root_cause"] = float(arr.mean())
        metrics["median_steps_to_root_cause"] = float(np.median(arr))
        metrics["min_steps_to_root_cause"] = int(arr.min())
        metrics["max_steps_to_root_cause"] = int(arr.max())
    else:
        metrics["avg_steps_to_root_cause"] = -1.0

    # Section 6.3 Metric 3: Root Cause Coverage
    metrics["root_cause_coverage"] = root_cause_hits / max(n_trajectories, 1)
    metrics["root_cause_hits"] = root_cause_hits

    # Distance distribution
    if all_distances:
        d_arr = np.array(all_distances)
        metrics["distance_mean"] = float(d_arr.mean())
        metrics["distance_std"] = float(d_arr.std())
        metrics["dist_d0"] = int((d_arr < 1e-6).sum())
        metrics["dist_d0_25"] = int(((d_arr >= 1e-6) & (d_arr < 0.25)).sum())
        metrics["dist_d25_50"] = int(((d_arr >= 0.25) & (d_arr < 0.5)).sum())
        metrics["dist_d50_75"] = int(((d_arr >= 0.5) & (d_arr < 0.75)).sum())
        metrics["dist_d75_100"] = int((d_arr >= 0.75).sum())

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Analyze fault localization quality of SWE agent trajectories")
    parser.add_argument("--trajectories", required=True, help="Path to trajectory JSONL file")
    parser.add_argument("--bonus_map_dir", required=True, help="Path to precomputed bonus maps directory")
    parser.add_argument("--tracking_mode", default="view_only", choices=["view_only", "view_and_bash"],
                       help="Observation tracking mode (default: view_only)")
    parser.add_argument("--output", default=None, help="Output JSON path for metrics (default: stdout)")
    parser.add_argument("--per_instance", action="store_true", help="Also output per-instance results")
    parser.add_argument("--wandb_project", default=None, help="W&B project name for logging")
    parser.add_argument("--wandb_run_name", default=None, help="W&B run name")
    args = parser.parse_args()

    # Load data
    print(f"Loading trajectories from {args.trajectories}...")
    trajectories = load_trajectories(args.trajectories)
    print(f"  Loaded {len(trajectories)} trajectories")

    bonus_store = BonusMapStore(args.bonus_map_dir)
    print(f"  Bonus maps dir: {args.bonus_map_dir}")
    print(f"  Tracking mode: {args.tracking_mode}")
    print()

    # Analyze each trajectory
    per_instance_results = []
    for traj in trajectories:
        instance_id = extract_instance_id(traj)
        responses = extract_assistant_responses(traj)
        bonus_map = bonus_store.get(instance_id) if instance_id else None

        result = analyze_trajectory(responses, bonus_map, tracking_mode=args.tracking_mode)
        result["instance_id"] = instance_id
        result["has_bonus_map"] = bonus_map is not None and bonus_map.get("traceable", False)
        per_instance_results.append(result)

    # Compute aggregate metrics
    metrics = compute_aggregate_metrics(per_instance_results)

    # Print results
    print("=" * 60)
    print("  Fault Localization Analysis Results")
    print("=" * 60)
    print(f"  Trajectories analyzed: {metrics['n_trajectories']}")
    print(f"  Total observation steps: {metrics['total_steps']}")
    print(f"  Total view steps (with reads): {metrics['total_view_steps']}")
    print(f"  Total on-graph steps: {metrics['total_on_graph_steps']}")
    print()
    print(f"  [Metric 1] On-graph Read Ratio: {metrics['on_graph_read_ratio']:.4f}")
    print(f"  [Metric 1b] On-graph View Density: {metrics['on_graph_view_density']:.4f}")
    print(f"  [Metric 2] Avg Steps to Root Cause: {metrics['avg_steps_to_root_cause']:.2f}")
    if "median_steps_to_root_cause" in metrics:
        print(f"             Median Steps to Root Cause: {metrics['median_steps_to_root_cause']:.1f}")
    print(f"  [Metric 3] Root Cause Coverage: {metrics['root_cause_coverage']:.4f} ({metrics['root_cause_hits']}/{metrics['n_trajectories']})")
    print()

    if "distance_mean" in metrics:
        print(f"  Distance distribution (on-graph steps):")
        print(f"    Mean distance: {metrics['distance_mean']:.4f} +/- {metrics['distance_std']:.4f}")
        total_on = metrics["total_on_graph_steps"]
        print(f"    d=0 (root cause): {metrics['dist_d0']} ({metrics['dist_d0']/max(total_on,1)*100:.1f}%)")
        print(f"    d in (0, 0.25):   {metrics['dist_d0_25']} ({metrics['dist_d0_25']/max(total_on,1)*100:.1f}%)")
        print(f"    d in [0.25, 0.5): {metrics['dist_d25_50']} ({metrics['dist_d25_50']/max(total_on,1)*100:.1f}%)")
        print(f"    d in [0.5, 0.75): {metrics['dist_d50_75']} ({metrics['dist_d50_75']/max(total_on,1)*100:.1f}%)")
        print(f"    d in [0.75, 1.0]: {metrics['dist_d75_100']} ({metrics['dist_d75_100']/max(total_on,1)*100:.1f}%)")

    # Save output
    if args.output:
        output_data = {"aggregate": metrics}
        if args.per_instance:
            # Remove 'distances' list to keep output compact
            for r in per_instance_results:
                r.pop("distances", None)
            output_data["per_instance"] = per_instance_results
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n  Results saved to {args.output}")

    # W&B logging
    if args.wandb_project:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or "localization-analysis",
                config={
                    "trajectories_path": args.trajectories,
                    "bonus_map_dir": args.bonus_map_dir,
                    "tracking_mode": args.tracking_mode,
                },
            )
            wandb.log({f"loc/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
            wandb.finish()
            print(f"\n  Logged to W&B project: {args.wandb_project}")
        except ImportError:
            print("\n  Warning: wandb not installed, skipping W&B logging")


if __name__ == "__main__":
    main()
