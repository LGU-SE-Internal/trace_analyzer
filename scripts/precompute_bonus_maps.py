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
    TRACE_FILE_PATH,
    _is_test_file,
    extract_callables_from_ast,
    find_modified_callables_from_sources,
    find_modified_callables_from_task,
    make_instance_id,
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


def classify_bonus_map(result: dict) -> str:
    """Classify a bonus map result into a case_type string.

    Categories:
      no_callables    – patch modifies no callable (variables, imports, etc.)
      static_only     – has callables but no test entries in call graph
                        (static fallback or crash bug where no F2P test reached)
      direct          – F2P test calls patched callable directly, no intermediate
      standard        – F2P test reaches patched callable through intermediate
                        production-code nodes (graded by hop_max)
    """
    patched = result.get("patched_callables", [])
    nodes = result.get("call_graph_nodes", {})
    traceable = result.get("traceable", False)

    if not patched:
        return "no_callables"

    if not traceable:
        return "no_callables"

    # Count test entries using _is_test_file (matches analyze script logic)
    n_test_entries = sum(
        1 for v in nodes.values() if _is_test_file(v.get("file_path", ""))
    )
    if n_test_entries == 0:
        return "static_only"

    # Has test entries — check for intermediate production-code nodes
    n_intermediate = sum(
        1 for v in nodes.values()
        if not _is_test_file(v.get("file_path", ""))
        and v.get("normalized_distance", 0) > 0
    )
    return "standard" if n_intermediate > 0 else "direct"


def compute_static_bonus_map(task: dict) -> dict:
    """Compute a static bonus map (patched callables only, all d=0).

    No sandbox needed. Extracts modified callables from AST diff.
    """
    task = normalize_task(task)
    instance_id = make_instance_id(task)
    all_modified = find_modified_callables_from_task(task)

    if not all_modified:
        result = {
            "instance_id": instance_id,
            "patched_callables": [],
            "call_graph_nodes": {},
            "hop_max": 0,
            "traceable": False,
        }
        result["case_type"] = classify_bonus_map(result)
        return result

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

    result = {
        "instance_id": instance_id,
        "patched_callables": all_modified,
        "call_graph_nodes": call_graph_nodes,
        "hop_max": 0,
        "traceable": True,
    }
    result["case_type"] = classify_bonus_map(result)
    return result


def compute_dynamic_bonus_map(task: dict) -> dict:
    """Compute a dynamic bonus map using the full trace pipeline.

    Requires a sandbox environment to be available.
    """
    from rllm.environments.swe.trace import (
        aggregate_traces,
        build_call_graph_from_traces,
        instrument_sandbox,
        parse_fault_traces_from_file,
    )

    task = normalize_task(task)
    instance_id = make_instance_id(task)

    # Step 1: extract patched callables via static analysis
    all_modified = find_modified_callables_from_task(task)

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

    env = SWEEnv.from_dict({**task, "experiment_id": os.environ.get("ARL_EXPERIMENT_ID", "bonus-maps")})
    try:
        # reset() creates the sandbox session, sets up the env, provisions tools
        env.reset()

        instrumented_callables = instrument_sandbox(env, all_modified)

        if not instrumented_callables:
            return compute_static_bonus_map(task)

        # Clear any stale trace file before running tests
        env._run(f"rm -f {TRACE_FILE_PATH}")

        # Run tests — traces are written to /tmp/_swe_fault_traces.jsonl
        # For R2E-Gym, inject -rA into the test script so pytest emits the
        # "short test summary info" section (needed to identify F2P tests).
        test_script = (
            "/run_tests.sh" if env.swebench_verified
            else f"{env.alt_path}/run_tests.sh"
        )
        if not env.swebench_verified:
            env._run(
                f"sed -i '/pytest/{{/-rA/!s/pytest/pytest -rA/}}' {test_script}"
            )
        stdout, stderr, _ = env._execute_raw(
            f"bash {test_script}", timeout=300
        )
        raw_output = f"{stdout}\n{stderr}" if stderr else stdout

        # Read traces from file (bypasses pytest capture)
        traces = parse_fault_traces_from_file(env, instrumented_callables, env.repo_path)

        # Diagnostic: when 0 traces, check if the trace file exists at all
        if not traces:
            trace_check, _, tc_exit = env._execute_raw(f"wc -l {TRACE_FILE_PATH} 2>/dev/null || echo MISSING")
            print(f"  [{instance_id}] 0 traces captured. "
                  f"Trace file: {trace_check.strip()}. "
                  f"Instrumented {len(instrumented_callables)} callables.")

        # Identify F2P (fail-to-pass) tests and filter traces
        f2p_test_funcs = _get_f2p_test_funcs(task, raw_output, env.swebench_verified)
        if f2p_test_funcs is not None:
            pre_filter = len(traces)
            traces = _filter_traces_to_f2p(traces, f2p_test_funcs)
            print(f"  [{instance_id}] F2P filter: {pre_filter} traces → {len(traces)} "
                  f"(F2P funcs: {f2p_test_funcs})")
        else:
            print(f"  [{instance_id}] WARNING: Could not determine F2P tests, keeping all {len(traces)} traces")

        traces = aggregate_traces(traces)

        # Build call graph, enriching non-patched node line ranges via AST
        def _read_file(rel_path: str) -> str:
            from rllm.environments.swe.trace import _read_sandbox_file
            content, exit_code = _read_sandbox_file(
                env, f"{env.repo_path}/{rel_path}"
            )
            return content if exit_code == 0 else ""

        result = build_call_graph_from_traces(traces, all_modified, file_reader=_read_file)
        result["instance_id"] = instance_id
        result["case_type"] = classify_bonus_map(result)
        return result

    except Exception as e:
        print(f"  [WARN] Dynamic tracing failed for {instance_id}: {e}")
        # Fallback to static
        return compute_static_bonus_map(task)
    finally:
        env.close()


def _get_f2p_test_funcs(
    task: dict, raw_output: str, swebench_verified: bool
) -> set[str] | None:
    """Identify fail-to-pass test function names.

    For SWE-Bench Verified: uses the ``FAIL_TO_PASS`` field from the task.
    For R2E-Gym: parses pytest output for FAILED tests (tests that fail on
    unpatched code are the F2P tests we want).

    Returns a set of **bare** test function names (e.g. ``test_foo``), or
    None if we can't determine F2P tests (in which case all traces are kept).
    """
    if swebench_verified:
        f2p_raw = task.get("FAIL_TO_PASS")
        if f2p_raw:
            if isinstance(f2p_raw, str):
                try:
                    f2p_list = json.loads(f2p_raw)
                except (json.JSONDecodeError, TypeError):
                    f2p_list = [f2p_raw]
            elif isinstance(f2p_raw, list):
                f2p_list = f2p_raw
            else:
                return None
            # Extract bare function names:
            # "tests/test_x.py::TestClass::test_method" → "test_method"
            # "tests/test_x.py::test_func" → "test_func"
            funcs = set()
            for t in f2p_list:
                parts = str(t).split("::")
                funcs.add(parts[-1])  # bare function name
            return funcs if funcs else None
    else:
        # R2E-Gym: parse pytest output for FAILED tests (= F2P on buggy code).
        # parse_log_pytest returns names like "TestClass.test_method" or "test_func"
        # (it splits on "::" and joins with ".").
        from rllm.environments.swe.reward import parse_log_pytest

        test_status = parse_log_pytest(raw_output)
        if not test_status:
            return None
        failed_funcs = set()
        for name, status in test_status.items():
            if status in ("FAILED", "ERROR"):
                # name may be "TestClass.test_method" or just "test_func"
                # Extract bare function name (last dot-segment)
                bare = name.rsplit(".", 1)[-1] if "." in name else name
                if bare:  # filter out empty strings from malformed lines
                    failed_funcs.add(bare)
        return failed_funcs if failed_funcs else None

    return None


def _filter_traces_to_f2p(
    traces: list[list[dict]], f2p_test_funcs: set[str]
) -> list[list[dict]]:
    """Keep only traces whose call chain originates from an F2P test function.

    A trace originates from an F2P test if the outermost test frame (the
    first frame in a test file) has a func_name in *f2p_test_funcs*.
    *f2p_test_funcs* contains **bare** function names (e.g. ``test_foo``).
    """
    filtered = []
    for trace in traces:
        # Find the outermost test frame (first frame in a test file)
        for frame in trace:
            file_path = frame.get("file_path", "")
            if not _is_test_file(file_path):
                continue
            func_name = frame.get("func_name", "")
            if func_name in f2p_test_funcs:
                filtered.append(trace)
                break
    return filtered


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
