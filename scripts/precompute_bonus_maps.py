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
    find_modified_callables_from_task,
    make_instance_id,
    normalize_task,
)



def classify_bonus_map(result: dict) -> str:
    """Classify a bonus map result into a case_type string.

    Traceable (dynamic tracing produced a useful call graph):
      direct          – F2P test calls patched callable directly, no intermediate
      standard        – F2P test reaches patched callable through intermediate
                        production-code nodes (graded by hop_max)

    Untraceable (no useful call graph):
      no_callables    – patch modifies no callable (variables, imports, etc.)
      newly_created   – all patched callables are newly created (only in new_file_content,
                        not in old_file_content); cannot be instrumented on buggy code
      crash           – tests crashed before reaching patched callable (trace file missing)
      no_f2p_trace    – instrumentation succeeded, tests ran, but no F2P trace captured
    """
    patched = result.get("patched_callables", [])
    nodes = result.get("call_graph_nodes", {})
    traceable = result.get("traceable", False)

    if not patched:
        if result.get("newly_created_callables"):
            return "newly_created"
        return "no_callables"

    if not traceable:
        return "no_callables"

    # Crash: instrumentation succeeded but trace file never appeared
    if result.get("crash"):
        return "crash"

    # Count test entries using _is_test_file (matches analyze script logic)
    n_test_entries = sum(
        1 for v in nodes.values() if _is_test_file(v.get("file_path", ""))
    )
    if n_test_entries == 0:
        return "no_f2p_trace"

    # Has test entries — check for intermediate production-code nodes
    n_intermediate = sum(
        1 for v in nodes.values()
        if not _is_test_file(v.get("file_path", ""))
        and v.get("normalized_distance", 0) > 0
    )
    return "standard" if n_intermediate > 0 else "direct"


def find_newly_created_callables(task: dict) -> list[dict]:
    """Find callables that only exist in new_file_content (added by the fix).

    These are pure additions — the callable doesn't exist in old_file_content
    at all — so they cannot be instrumented on the pre-fix (buggy) code.
    """
    task = normalize_task(task)
    for key in ("parsed_commit_content", "parsed_commit"):
        raw = task.get(key)
        if not raw:
            continue
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
        if not isinstance(raw, dict):
            continue
        file_diffs = raw.get("file_diffs", [])
        if not file_diffs:
            continue

        newly_created: list[dict] = []
        for fd in file_diffs:
            path = fd.get("header", {}).get("file", {}).get("path", "")
            if not path or not path.endswith(".py"):
                continue
            old_src = fd.get("old_file_content") or ""
            new_src = fd.get("new_file_content") or ""
            if not new_src:
                continue

            new_callables = extract_callables_from_ast(new_src, path)
            if old_src:
                old_callables = extract_callables_from_ast(old_src, path)
            else:
                old_callables = {}

            for qname, info in new_callables.items():
                if qname not in old_callables:
                    newly_created.append(info.to_dict())
        return newly_created

    return []


def compute_static_bonus_map(task: dict) -> dict:
    """Compute a static bonus map (patched callables only, all d=0).

    No sandbox needed. Extracts modified callables from AST diff.
    """
    task = normalize_task(task)
    instance_id = make_instance_id(task)
    all_modified = find_modified_callables_from_task(task)
    newly_created = find_newly_created_callables(task)

    if not all_modified:
        result = {
            "instance_id": instance_id,
            "patched_callables": [],
            "newly_created_callables": newly_created,
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
        "newly_created_callables": newly_created,
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

    # Step 1: extract patched callables from the developer's actual diff
    # (old_file_content vs new_file_content).  Line numbers refer to
    # old_file_content; instrument_sandbox relocates by qualified_name
    # to the actual sandbox source.
    all_modified = find_modified_callables_from_task(task)
    newly_created = find_newly_created_callables(task)

    if not all_modified:
        result = {
            "instance_id": instance_id,
            "patched_callables": [],
            "newly_created_callables": newly_created,
            "call_graph_nodes": {},
            "hop_max": 0,
            "traceable": False,
        }
        result["case_type"] = classify_bonus_map(result)
        return result

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
        stdout, stderr, test_exit = env._execute_raw(
            f"bash {test_script}", timeout=300
        )
        raw_output = f"{stdout}\n{stderr}" if stderr else stdout

        # Read traces from file (bypasses pytest capture)
        traces = parse_fault_traces_from_file(env, instrumented_callables, env.repo_path, env.alt_path)

        # Check trace file existence for crash detection
        trace_file_exists = False
        if not traces:
            trace_check, _, tc_exit = env._execute_raw(f"test -f {TRACE_FILE_PATH} && echo EXISTS || echo MISSING")
            trace_file_exists = "EXISTS" in trace_check
            print(f"  [{instance_id}] 0 traces captured. "
                  f"Trace file: {'exists (empty)' if trace_file_exists else 'MISSING'}. "
                  f"test_exit={test_exit}. "
                  f"Instrumented {len(instrumented_callables)} callables.")
        else:
            trace_file_exists = True

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
        result["newly_created_callables"] = newly_created

        # Crash detection — three cases:
        # 1. Trace file never appeared → tests crashed before any patched code ran
        # 2. ALL tests failed → systemic crash (e.g. setUp loads broken data)
        # 3. F2P traces exist but ALL come from setUp/tearDown, none from
        #    actual test_xxx methods → F2P tests crashed in their body
        _FIXTURE_NAMES = frozenset({
            "setUp", "tearDown", "setUpClass", "tearDownClass",
            "asyncSetUp", "asyncTearDown",
        })
        is_crash = False
        if not trace_file_exists and len(instrumented_callables) > 0:
            is_crash = True
        elif test_exit != 0:
            from rllm.environments.swe.reward import parse_log_pytest
            test_status = parse_log_pytest(raw_output)
            if test_status:
                n_total = len(test_status)
                n_fail = sum(1 for s in test_status.values() if s in ("FAILED", "ERROR"))
                if n_fail == n_total:
                    is_crash = True
                    print(f"  [{instance_id}] All {n_total} tests failed — crash")
        # Case 3: check if all F2P traces are fixture-only (no test_xxx frames)
        if not is_crash and traces:
            has_real_test_trace = False
            for trace in traces:
                for frame in trace:
                    fp = frame.get("file_path", "")
                    if not _is_test_file(fp):
                        continue
                    fn = frame.get("func_name", "")
                    if fn not in _FIXTURE_NAMES:
                        has_real_test_trace = True
                        break
                if has_real_test_trace:
                    break
            if not has_real_test_trace:
                is_crash = True
                print(f"  [{instance_id}] All F2P traces from fixtures only — F2P tests crashed")
        result["crash"] = is_crash

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

    A trace originates from an F2P test if ANY test-file frame has a
    func_name in *f2p_test_funcs*.

    Special case: ``setUp`` / ``tearDown`` etc. are test fixtures that run
    for every test method including F2P ones. If the outermost test-file
    frame is a fixture AND at least one F2P test exists, the trace is kept.
    """
    _FIXTURE_NAMES = frozenset({
        "setUp", "tearDown", "setUpClass", "tearDownClass",
        "asyncSetUp", "asyncTearDown",
    })

    filtered = []
    for trace in traces:
        keep = False
        for frame in trace:
            file_path = frame.get("file_path", "")
            if not _is_test_file(file_path):
                continue
            func_name = frame.get("func_name", "")
            if func_name in f2p_test_funcs:
                keep = True
                break
            if func_name in _FIXTURE_NAMES:
                keep = True
                break
        if keep:
            filtered.append(trace)
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

        case_type = result.get("case_type", "unknown")
        return idx, instance_id, case_type, None
    except Exception as e:
        return idx, "unknown", "error", str(e)


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
    from collections import Counter
    case_counts = Counter()
    error_count = 0
    total = len(work_items)

    if args.n_parallel <= 1:
        for item in work_items:
            idx, instance_id, case_type, error = _process_one(item)
            if error:
                error_count += 1
                print(f"  [{idx}] ERROR: {error}")
            case_counts[case_type] += 1
            done = idx + 1
            if done % 100 == 0:
                traceable = case_counts["direct"] + case_counts["standard"]
                print(f"  Progress: {done}/{total} (traceable: {traceable}, errors: {error_count})")
    else:
        with ThreadPoolExecutor(max_workers=args.n_parallel) as executor:
            futures = {executor.submit(_process_one, item): item for item in work_items}
            done_count = 0
            for future in as_completed(futures):
                idx, instance_id, case_type, error = future.result()
                done_count += 1
                if error:
                    error_count += 1
                case_counts[case_type] += 1
                if done_count % 100 == 0:
                    traceable = case_counts["direct"] + case_counts["standard"]
                    print(f"  Progress: {done_count}/{total} (traceable: {traceable}, errors: {error_count})")

    # Summary table
    traceable = case_counts["direct"] + case_counts["standard"]
    untraceable = total - traceable - error_count

    print(f"\n{'='*50}")
    print(f"Summary: {total} instances from {args.parquet_path}")
    print(f"{'='*50}")
    print(f"\ntraceable/          {traceable:5d}  ({100*traceable/total:.1f}%)")
    print(f"  direct             {case_counts['direct']:5d}")
    print(f"  standard           {case_counts['standard']:5d}")
    print(f"\nuntraceable/        {untraceable:5d}  ({100*untraceable/total:.1f}%)")
    print(f"  crash              {case_counts['crash']:5d}")
    print(f"  newly_created      {case_counts['newly_created']:5d}")
    print(f"  no_callables       {case_counts['no_callables']:5d}")
    print(f"  no_f2p_trace       {case_counts['no_f2p_trace']:5d}")
    if error_count:
        print(f"\nerrors              {error_count:5d}  ({100*error_count/total:.1f}%)")
    print(f"\nBonus maps saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
