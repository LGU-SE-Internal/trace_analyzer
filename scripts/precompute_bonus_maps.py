#!/usr/bin/env python3
"""Precompute P2A bonus maps for SWE dataset instances.

Static mode (no sandbox needed):
    Extracts patched callables from AST diff, assigns d=0 to all.

Dynamic mode (requires sandbox):
    Runs full trace pipeline → builds call graph → assigns hop distances.

Usage:
    # Static mode (fast, no sandbox)
    python scripts/precompute_bonus_maps.py \
        data/swe/R2E_Gym_Subset.parquet \
        --output_dir data/swe/bonus_maps \
        --mode static

    # Dynamic mode (needs sandbox cluster)
    python scripts/precompute_bonus_maps.py \
        data/swe/R2E_Gym_Subset.parquet \
        --output_dir data/swe/bonus_maps \
        --mode dynamic \
        --n_parallel 50

Output format (per instance JSON):
    {
      "instance_id": "...",
      "patched_callables": [...],
      "call_graph_nodes": {...},
      "hop_max": int,
      "traceable": bool
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rllm.environments.swe.trace import (
    _is_test_file,
    extract_callables_from_ast,
    find_modified_callables_from_sources,
    normalize_task,
)


def _parse_file_diffs(task: dict) -> list[dict]:
    """Extract file_diffs from whichever field is present."""
    for key in ("parsed_commit_content", "parsed_commit"):
        raw = task.get(key)
        if not raw:
            continue
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
        if isinstance(raw, dict):
            fds = raw.get("file_diffs", [])
            if fds:
                return fds
    return []


def _non_test_py_diffs(file_diffs: list[dict], task: dict) -> list[dict]:
    """Filter file_diffs to non-test .py files."""
    relevant = task.get("relevant_files")
    allow_set = set(relevant) if relevant is not None else None

    result = []
    for fd in file_diffs:
        path = fd.get("header", {}).get("file", {}).get("path", "")
        if not path or not path.endswith(".py"):
            continue
        if allow_set is not None:
            if path not in allow_set:
                continue
        else:
            if _is_test_file(path):
                continue
        result.append(fd)
    return result


def compute_static_bonus_map(task: dict) -> dict:
    """Compute a static bonus map (patched callables only, all d=0).

    No sandbox needed. Extracts modified callables from AST diff.
    """
    task = normalize_task(task)
    instance_id = task.get("instance_id", task.get("commit_hash", "unknown"))
    file_diffs = _parse_file_diffs(task)
    diffs = _non_test_py_diffs(file_diffs, task)

    all_modified = []
    for fd in diffs:
        path = fd["header"]["file"]["path"]
        old_src = fd.get("old_file_content") or ""
        new_src = fd.get("new_file_content") or ""
        if old_src and new_src:
            modified = find_modified_callables_from_sources(old_src, new_src, path)
            all_modified.extend(modified)

    if not all_modified:
        return {
            "instance_id": instance_id,
            "patched_callables": [],
            "call_graph_nodes": {},
            "hop_max": 0,
            "traceable": False,
        }

    # Static mode: every patched callable is at d=0
    call_graph_nodes = {}
    for mc in all_modified:
        node_key = f"{mc['file_path']}::{mc['qualified_name']}"
        call_graph_nodes[node_key] = {
            "file_path": mc["file_path"],
            "start_line": mc["start_line"],
            "end_line": mc["end_line"],
            "hop_distance": 0,
            "normalized_distance": 0.0,
        }

    return {
        "instance_id": instance_id,
        "patched_callables": all_modified,
        "call_graph_nodes": call_graph_nodes,
        "hop_max": 0,
        "traceable": True,
    }


def compute_dynamic_bonus_map(task: dict) -> dict:
    """Compute a dynamic bonus map using the full trace pipeline.

    Requires a sandbox environment to be available.
    """
    from rllm.environments.swe.trace import (
        aggregate_traces,
        build_call_graph_from_traces,
        extract_non_test_patch,
        instrument_sandbox,
        parse_fault_traces,
    )

    task = normalize_task(task)
    instance_id = task.get("instance_id", task.get("commit_hash", "unknown"))

    # Step 1: extract patched callables via static analysis
    file_diffs = _parse_file_diffs(task)
    diffs = _non_test_py_diffs(file_diffs, task)

    all_modified = []
    for fd in diffs:
        path = fd["header"]["file"]["path"]
        old_src = fd.get("old_file_content") or ""
        new_src = fd.get("new_file_content") or ""
        if old_src and new_src:
            modified = find_modified_callables_from_sources(old_src, new_src, path)
            all_modified.extend(modified)

    if not all_modified:
        return {
            "instance_id": instance_id,
            "patched_callables": [],
            "call_graph_nodes": {},
            "hop_max": 0,
            "traceable": False,
        }

    # Step 2-5: create sandbox, instrument, run tests, parse traces
    # Requires ARL gateway + k8s cluster with warm pools
    from rllm.environments.swe.swe import SWEEnv

    env = SWEEnv.from_dict(task)
    try:
        # reset() creates the sandbox session, sets up the env, provisions tools
        env.reset()

        patch_text = extract_non_test_patch(task)
        instrumented_callables = instrument_sandbox(env, patch_text)

        if not instrumented_callables:
            return compute_static_bonus_map(task)

        # Run tests — traces are emitted to stderr by _swe_fault_tracer
        test_script = (
            "/run_tests.sh" if env.swebench_verified
            else f"{env.alt_path}/run_tests.sh"
        )
        stdout, stderr, _ = env._execute_raw(
            f"bash {test_script}", timeout=300
        )
        raw_output = f"{stdout}\n{stderr}" if stderr else stdout

        traces = parse_fault_traces(raw_output, instrumented_callables, env.repo_path)
        traces = aggregate_traces(traces)

        # Build call graph
        result = build_call_graph_from_traces(traces, all_modified)
        result["instance_id"] = instance_id
        return result

    except Exception as e:
        print(f"  [WARN] Dynamic tracing failed for {instance_id}: {e}")
        # Fallback to static
        return compute_static_bonus_map(task)
    finally:
        env.close()


def _process_one(args):
    """Worker function for parallel processing."""
    idx, task_json, output_dir, mode = args
    try:
        task = json.loads(task_json) if isinstance(task_json, str) else dict(task_json)

        if mode == "static":
            result = compute_static_bonus_map(task)
        else:
            result = compute_dynamic_bonus_map(task)

        instance_id = result["instance_id"]
        output_path = os.path.join(output_dir, f"{instance_id}.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        return idx, instance_id, result["traceable"], None
    except Exception as e:
        return idx, "unknown", False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Precompute P2A bonus maps")
    parser.add_argument("parquet_path", help="Path to dataset parquet file")
    parser.add_argument("--output_dir", required=True, help="Output directory for bonus map JSONs")
    parser.add_argument("--mode", choices=["static", "dynamic"], default="static",
                       help="static: AST diff only (fast). dynamic: full trace pipeline (needs sandbox)")
    parser.add_argument("--n_parallel", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N instances")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_parquet(args.parquet_path)
    if args.limit:
        df = df.head(args.limit)

    print(f"Processing {len(df)} instances from {args.parquet_path}")
    print(f"Mode: {args.mode}, Output: {args.output_dir}, Workers: {args.n_parallel}")

    # Prepare tasks
    work_items = []
    for idx, row in df.iterrows():
        extra_raw = row.get("extra_info", "{}")
        work_items.append((idx, extra_raw, args.output_dir, args.mode))

    # Process
    traceable_count = 0
    error_count = 0
    total = len(work_items)

    if args.n_parallel <= 1:
        for item in work_items:
            idx, instance_id, traceable, error = _process_one(item)
            if error:
                error_count += 1
                print(f"  [{idx}] ERROR: {error}")
            else:
                if traceable:
                    traceable_count += 1
            done = idx + 1
            if done % 100 == 0:
                print(f"  Progress: {done}/{total} (traceable: {traceable_count}, errors: {error_count})")
    else:
        with ThreadPoolExecutor(max_workers=args.n_parallel) as executor:
            futures = {executor.submit(_process_one, item): item for item in work_items}
            done_count = 0
            for future in as_completed(futures):
                idx, instance_id, traceable, error = future.result()
                done_count += 1
                if error:
                    error_count += 1
                elif traceable:
                    traceable_count += 1
                if done_count % 100 == 0:
                    print(f"  Progress: {done_count}/{total} (traceable: {traceable_count}, errors: {error_count})")

    print(f"\nDone! Total: {total}, Traceable: {traceable_count}, Errors: {error_count}")
    print(f"Bonus maps saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
