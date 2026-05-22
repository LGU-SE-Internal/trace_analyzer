#!/usr/bin/env python3
"""Precompute P2A bonus maps for SWE dataset instances.

Static mode (no sandbox needed):
    Extracts patched callables from AST diff, assigns d=0 to all.

Dynamic mode (requires sandbox):
    Runs full trace pipeline → builds call graph → assigns hop distances.

Classification decision tree (evaluated top-to-bottom, first match wins):

    Static layer (AST diff of old vs new content):
      newly_created  – all GT callables only exist in new_file_content
      no_callable    – patch has no callable-level changes

    Dynamic layer (instrument → run tests → parse traces):
      no_trace       – 0 traces captured after instrumentation (error=True)
      no_gt          – traces exist but none contain a GT callable (error=True)
      all_pass       – GT traces exist but all tests pass on buggy code (error=True)
      no_f2p         – GT traces exist, tests fail, but F2P filter removed all (error=True)
      standard       – F2P→GT call chain with intermediate nodes (traceable=True)
      direct         – F2P→GT call chain, test calls GT directly (traceable=True)

Usage:
    python -m utils.p2a.precompute_bonus_maps \\
        data/swe/R2E_Gym_Subset.parquet \\
        --output_dir data/swe/bonus_maps --mode dynamic --n_parallel 50
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rllm.environments.swe.trace import (
    TRACE_FILE_PATH,
    _is_test_file,
    extract_callables_from_ast,
    find_modified_callables_from_task,
    make_instance_id,
    normalize_task,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXTURE_NAMES = frozenset(
    {
        "setUp",
        "tearDown",
        "setUpClass",
        "tearDownClass",
        "asyncSetUp",
        "asyncTearDown",
    }
)


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


_PARAMETRIZE_SUFFIX_RE = re.compile(r"\[.*\]$")


def _strip_parametrize(name: str) -> str:
    """Strip pytest parametrize suffix: ``test_foo[True]`` → ``test_foo``."""
    return _PARAMETRIZE_SUFFIX_RE.sub("", name)


def _get_f2p_test_funcs(task: dict, raw_output: str, swebench_verified: bool) -> set[str] | None:
    """Identify fail-to-pass (F2P) test function names.

    F2P = tests that FAIL on buggy code and PASS after the developer's fix.

    For SWE-Bench Verified: uses the ``FAIL_TO_PASS`` field from the task.
    For R2E-Gym: parses pytest output for FAILED tests on buggy code.

    Returns:
        set[str]: bare test function names (may be empty if no tests failed).
        None: only when we genuinely cannot parse the test output.

    Note: parametrize suffixes (``[param1-param2]``) are stripped so that
    ``test_foo[True]`` matches trace frames that only contain ``test_foo``.
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
            funcs = set()
            for t in f2p_list:
                parts = str(t).split("::")
                funcs.add(_strip_parametrize(parts[-1]))
            return funcs  # may be empty
        return None
    else:
        from rllm.environments.swe.reward import parse_log_pytest

        test_status = parse_log_pytest(raw_output)
        if not test_status:
            return None  # genuinely can't parse
        failed_funcs = set()
        for name, status in test_status.items():
            if status in ("FAILED", "ERROR"):
                bare = name.rsplit(".", 1)[-1] if "." in name else name
                bare = _strip_parametrize(bare)
                if bare:
                    failed_funcs.add(bare)
        return failed_funcs  # empty set = parsed OK but no failures


def _filter_traces_to_f2p(traces: list[list[dict]], f2p_test_funcs: set[str]) -> list[list[dict]]:
    """Keep only traces whose call chain originates from an F2P test function.

    A trace originates from an F2P test if ANY test-file frame has a
    func_name in *f2p_test_funcs*, or is a fixture (setUp/tearDown) that
    runs for every test including F2P ones.
    """
    if not f2p_test_funcs:
        return []

    filtered = []
    for trace in traces:
        keep = False
        for frame in trace:
            file_path = frame.get("file_path", "")
            if not _is_test_file(file_path):
                continue
            func_name = frame.get("func_name", "")
            bare_func_name = _strip_parametrize(func_name.rsplit(".", 1)[-1])
            if bare_func_name in f2p_test_funcs:
                keep = True
                break
            if bare_func_name in _FIXTURE_NAMES:
                keep = True
                break
        if keep:
            filtered.append(trace)
    return filtered


# ---------------------------------------------------------------------------
# Static bonus map
# ---------------------------------------------------------------------------


def compute_static_bonus_map(task: dict) -> dict:
    """Compute a static bonus map (patched callables only, all d=0)."""
    task = normalize_task(task)
    instance_id = make_instance_id(task)
    all_modified = find_modified_callables_from_task(task)
    newly_created = find_newly_created_callables(task)

    # --- Decision tree: static layer ---
    if not all_modified:
        if newly_created:
            case_type = "newly_created"
        else:
            case_type = "no_callable"
        return {
            "instance_id": instance_id,
            "case_type": case_type,
            "traceable": False,
            "error": False,
            "patched_callables": [],
            "newly_created_callables": newly_created,
            "call_graph_nodes": {},
            "hop_max": 0,
        }

    # Static mode: every patched callable is at d=0, no test info
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
        "case_type": "no_f2p",  # static mode can't determine F2P
        "traceable": False,
        "error": False,
        "patched_callables": all_modified,
        "newly_created_callables": newly_created,
        "call_graph_nodes": call_graph_nodes,
        "hop_max": 0,
    }


# ---------------------------------------------------------------------------
# Dynamic bonus map — decision tree
# ---------------------------------------------------------------------------


def compute_dynamic_bonus_map(task: dict) -> dict:
    """Compute a dynamic bonus map using the full trace pipeline.

    Implements the decision tree:
      1. newly_created / no_callable  (static layer)
      2. no_trace → no_gt → all_pass / no_f2p → standard / direct  (dynamic layer)
    """
    from rllm.environments.swe.trace import (
        aggregate_traces,
        build_call_graph_from_traces,
        instrument_sandbox,
        parse_fault_traces_from_file,
    )

    task = normalize_task(task)
    instance_id = make_instance_id(task)

    all_modified = find_modified_callables_from_task(task)
    newly_created = find_newly_created_callables(task)

    # ── Static layer ──────────────────────────────────────────────────────
    if not all_modified:
        if newly_created:
            case_type = "newly_created"
        else:
            case_type = "no_callable"
        return {
            "instance_id": instance_id,
            "case_type": case_type,
            "traceable": False,
            "error": False,
            "patched_callables": [],
            "newly_created_callables": newly_created,
            "call_graph_nodes": {},
            "hop_max": 0,
        }

    # ── Dynamic layer: instrument → run → parse ───────────────────────────
    from rllm.environments.swe.swe import SWEEnv

    env = SWEEnv.from_dict(
        {
            **task,
            "experiment_id": os.environ.get("ARL_EXPERIMENT_ID", "bonus-maps"),
        }
    )
    try:
        env.reset()

        instrumented_callables = instrument_sandbox(env, all_modified)
        if not instrumented_callables:
            # Instrumentation failed — fall back to static
            return compute_static_bonus_map(task)

        # Clear stale trace file, run tests
        env._run(f"rm -f {TRACE_FILE_PATH}")

        test_script = "/run_tests.sh" if env.swebench_verified else f"{env.alt_path}/run_tests.sh"
        if not env.swebench_verified:
            env._run(f"sed -i '/pytest/{{/-rA/!s/pytest/pytest -rA/}}' {test_script}")
        stdout, stderr, test_exit = env._execute_raw(f"bash {test_script}", timeout=300)
        raw_output = f"{stdout}\n{stderr}" if stderr else stdout

        # ── Decision node: NO_TRACE ──────────────────────────────────
        # parse_fault_traces_from_file returns only traces that contain
        # at least one is_patched=True frame (i.e. GT callable was entered).
        # "raw traces" here = traces with GT.
        raw_traces = parse_fault_traces_from_file(env, instrumented_callables, env.repo_path, env.alt_path)

        # Also check the raw trace file for total entry count (including
        # traces without GT) to distinguish no_trace from no_gt.
        trace_file_out, _, tf_exit = env._execute_raw(f"wc -l < {TRACE_FILE_PATH} 2>/dev/null || echo 0")
        try:
            total_trace_entries = int(trace_file_out.strip())
        except (ValueError, AttributeError):
            total_trace_entries = 0

        if total_trace_entries == 0:
            print(f"  [{instance_id}] no_trace: 0 trace entries. test_exit={test_exit}, instrumented={len(instrumented_callables)}")
            return _make_result(
                instance_id,
                "no_trace",
                all_modified,
                newly_created,
                error=True,
            )

        # ── Decision node: NO_GT ─────────────────────────────────────
        # total_trace_entries > 0 but raw_traces (filtered to GT) == 0
        if not raw_traces:
            print(f"  [{instance_id}] no_gt: {total_trace_entries} trace entries but 0 contain GT callables. test_exit={test_exit}")
            return _make_result(
                instance_id,
                "no_gt",
                all_modified,
                newly_created,
                error=True,
            )

        # ── Decision node: NO_F2P / ALL_PASS ──────────────────────────
        f2p_test_funcs = _get_f2p_test_funcs(task, raw_output, env.swebench_verified)

        if f2p_test_funcs is None:
            # Can't parse test output at all
            print(f"  [{instance_id}] no_f2p: f2p_test_funcs=None (parse failed). Dropping all {len(raw_traces)} traces. test_exit={test_exit}")
            return _make_result(
                instance_id,
                "no_f2p",
                all_modified,
                newly_created,
                error=True,
            )

        if len(f2p_test_funcs) == 0:
            # Tests parsed OK but none failed → all tests pass on buggy code
            print(f"  [{instance_id}] all_pass: 0 test failures on buggy code. test_exit={test_exit}")
            return _make_result(
                instance_id,
                "all_pass",
                all_modified,
                newly_created,
                error=True,
            )

        f2p_traces = _filter_traces_to_f2p(raw_traces, f2p_test_funcs)
        print(f"  [{instance_id}] F2P filter: {len(raw_traces)} → {len(f2p_traces)} (F2P funcs: {f2p_test_funcs})")

        if not f2p_traces:
            print(f"  [{instance_id}] no_f2p: F2P filter removed all traces")
            return _make_result(
                instance_id,
                "no_f2p",
                all_modified,
                newly_created,
                error=True,
            )

        # ── Build call graph from F2P+GT traces ──────────────────────
        traces = aggregate_traces(f2p_traces)

        def _read_file(rel_path: str) -> str:
            from rllm.environments.swe.trace import _read_sandbox_file

            content, exit_code = _read_sandbox_file(env, f"{env.repo_path}/{rel_path}")
            return content if exit_code == 0 else ""

        result = build_call_graph_from_traces(traces, all_modified, file_reader=_read_file)

        # ── Decision node: STANDARD vs DIRECT ────────────────────────
        nodes = result.get("call_graph_nodes", {})
        n_test_entries = sum(1 for v in nodes.values() if _is_test_file(v.get("file_path", "")))
        n_intermediate = sum(1 for v in nodes.values() if not _is_test_file(v.get("file_path", "")) and v.get("normalized_distance", 0) > 0)

        if n_test_entries > 0 and n_intermediate > 0:
            case_type = "standard"
        else:
            case_type = "direct"

        result["instance_id"] = instance_id
        result["case_type"] = case_type
        result["traceable"] = True
        result["error"] = False
        result["newly_created_callables"] = newly_created
        return result

    except Exception as e:
        print(f"  [WARN] Dynamic tracing failed for {instance_id}: {e}")
        traceback.print_exc()
        return _make_result(
            instance_id,
            "no_trace",
            all_modified,
            newly_created,
            error=True,
        )
    finally:
        env.close()


def _make_result(
    instance_id: str,
    case_type: str,
    patched_callables: list[dict],
    newly_created_callables: list[dict],
    *,
    error: bool = False,
) -> dict:
    """Build a result dict for untraceable cases."""
    return {
        "instance_id": instance_id,
        "case_type": case_type,
        "traceable": False,
        "error": error,
        "patched_callables": patched_callables,
        "newly_created_callables": newly_created_callables,
        "call_graph_nodes": {},
        "hop_max": 0,
    }


# ---------------------------------------------------------------------------
# Parallel processing & CLI
# ---------------------------------------------------------------------------


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
    parser.add_argument("--mode", choices=["static", "dynamic"], default="static", help="static: AST diff only. dynamic: full trace pipeline")
    parser.add_argument("--n_parallel", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N instances")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_parquet(args.parquet_path)
    if args.limit:
        df = df.head(args.limit)

    print(f"Processing {len(df)} instances from {args.parquet_path}")
    print(f"Mode: {args.mode}, Output: {args.output_dir}, Workers: {args.n_parallel}")

    work_items = []
    for idx, row in df.iterrows():
        extra_raw = row.get("extra_info", "{}")
        work_items.append((idx, extra_raw, args.output_dir, args.mode))

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
            done = sum(case_counts.values())
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
    sum(case_counts[ct] for ct in ("no_trace", "no_gt", "no_f2p"))

    print(f"\n{'=' * 50}")
    print(f"Summary: {total} instances from {args.parquet_path}")
    print(f"{'=' * 50}")

    print(f"\ntraceable/          {traceable:5d}  ({100 * traceable / total:.1f}%)")
    print(f"  direct             {case_counts['direct']:5d}")
    print(f"  standard           {case_counts['standard']:5d}")

    print(f"\nuntraceable/        {total - traceable:5d}  ({100 * (total - traceable) / total:.1f}%)")
    print(f"  newly_created      {case_counts['newly_created']:5d}")
    print(f"  no_callable        {case_counts['no_callable']:5d}")
    print(f"  all_pass  (error)  {case_counts['all_pass']:5d}")
    print(f"  no_trace  (error)  {case_counts['no_trace']:5d}")
    print(f"  no_gt     (error)  {case_counts['no_gt']:5d}")
    print(f"  no_f2p    (error)  {case_counts['no_f2p']:5d}")

    if error_count:
        print(f"\nprocess errors       {error_count:5d}")

    print(f"\nBonus maps saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
