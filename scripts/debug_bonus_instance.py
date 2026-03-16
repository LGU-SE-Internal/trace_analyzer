#!/usr/bin/env python3
"""Debug a single instance's sandbox execution end-to-end.

Usage:
    python scripts/debug_bonus_instance.py orange3__62549d25
    python scripts/debug_bonus_instance.py orange3__945e235a
    python scripts/debug_bonus_instance.py --index 42   # by row index

Shows:
    1. Patched callables found by AST diff
    2. Tracer module deployment result
    3. Instrumented source (first instrumented file, truncated)
    4. Raw test output (stdout + stderr)
    5. Trace file contents from sandbox
    6. Parsed F2P tests
    7. Final bonus map
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rllm.environments.swe.trace import (
    TRACE_FILE_PATH,
    _is_test_file,
    aggregate_traces,
    build_call_graph_from_traces,
    find_modified_callables_from_task,
    generate_tracer_module,
    instrument_sandbox,
    instrument_source,
    make_instance_id,
    normalize_task,
    parse_fault_traces_from_file,
)


def find_task(parquet_path: str, target_id: str | None, index: int | None) -> dict:
    df = pd.read_parquet(parquet_path)
    if index is not None:
        row = df.iloc[index]
        extra = row.get("extra_info", "{}")
        return json.loads(extra) if isinstance(extra, str) else dict(extra)

    for _, row in df.iterrows():
        extra_raw = row.get("extra_info", "{}")
        task = json.loads(extra_raw) if isinstance(extra_raw, str) else dict(extra_raw)
        task = normalize_task(task)
        iid = make_instance_id(task)
        if iid == target_id:
            return task
    raise ValueError(f"Instance {target_id} not found in {parquet_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("instance_id", nargs="?", default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--parquet", default="data/swe/R2E_Gym_Subset.parquet")
    args = parser.parse_args()

    if not args.instance_id and args.index is None:
        parser.error("Provide instance_id or --index")

    os.environ.setdefault("ARL_EXPERIMENT_ID", "debug-trace")
    os.environ.setdefault("ARL_GATEWAY_URL", "http://118.145.210.10:8080")

    task = find_task(args.parquet, args.instance_id, args.index)
    task = normalize_task(task)
    instance_id = make_instance_id(task)

    print(f"{'='*72}")
    print(f"Instance: {instance_id}")
    print(f"Docker image: {task.get('docker_image', 'N/A')}")
    print(f"{'='*72}")

    # Step 1: patched callables
    all_modified = find_modified_callables_from_task(task)
    print(f"\n[1] Patched callables ({len(all_modified)}):")
    for mc in all_modified:
        print(f"    {mc['file_path']}::{mc['qualified_name']}  "
              f"(lines {mc['start_line']}-{mc['end_line']})")

    if not all_modified:
        print("    No patched callables found. Nothing to trace.")
        return

    # Step 2: create sandbox
    from rllm.environments.swe.swe import SWEEnv

    env = SWEEnv.from_dict({
        **task,
        "experiment_id": os.environ.get("ARL_EXPERIMENT_ID", "debug-trace"),
    })
    try:
        print(f"\n[2] Creating sandbox...")
        env.reset()
        print(f"    repo_path: {env.repo_path}")
        print(f"    swebench_verified: {env.swebench_verified}")

        # Step 3: show pre-instrumentation source (first file, near patched callable)
        first_file = all_modified[0]["file_path"]
        first_start = all_modified[0]["start_line"]
        full_path = f"{env.repo_path}/{first_file}"
        src_out, _, _ = env._execute_raw(f"cat {full_path}")

        print(f"\n[3] Original source around first patched callable "
              f"({first_file}:{first_start}):")
        src_lines = src_out.splitlines()
        start = max(0, first_start - 3)
        end = min(len(src_lines), first_start + 15)
        for i in range(start, end):
            marker = ">>>" if i + 1 == first_start else "   "
            print(f"    {marker} {i+1:4d} | {src_lines[i]}")

        # Step 4: instrument
        print(f"\n[4] Instrumenting sandbox...")
        # Build file grouping for diagnostics
        callables_by_file_dbg: dict[str, list[dict]] = {}
        for c in all_modified:
            callables_by_file_dbg.setdefault(c["file_path"], []).append(c)
        instrumented = instrument_sandbox(env, all_modified)
        print(f"    Instrumented {len(instrumented)} callables")
        for mc in instrumented:
            instr_s = mc.get('instr_start_line', '?')
            instr_e = mc.get('instr_end_line', '?')
            print(f"    {mc['file_path']}::{mc['qualified_name']}  "
                  f"orig={mc['start_line']}-{mc['end_line']}  "
                  f"instr={instr_s}-{instr_e}")

        if not instrumented:
            print("    Instrumentation failed! Checking why...")
            # Try to read site-packages
            sp_out, sp_err = env._run(
                "python -c \"import site; print(site.getsitepackages())\""
            )
            print(f"    site-packages: {sp_out.strip()}")

            # Check tracer importability
            tr_out, tr_err = env._run(
                "python -c \"import _swe_fault_tracer; print('TRACER_OK')\""
            )
            print(f"    tracer import: {tr_out.strip()}")
            if tr_err:
                print(f"    tracer import err: {tr_err.strip()}")

            # Check source file existence and size
            for mc in all_modified:
                fp = f"{env.repo_path}/{mc['file_path']}"
                wc_out, _, wc_exit = env._execute_raw(f"wc -l {fp}")
                print(f"    {mc['file_path']}: {wc_out.strip() if wc_exit == 0 else 'NOT FOUND'}")

            # Try instrument_source manually and show result
            from rllm.environments.swe.trace import instrument_source as _isrc
            for file_path_key, callables in callables_by_file_dbg.items():
                fp = f"{env.repo_path}/{file_path_key}"
                src_out, _, ex = env._execute_raw(f"cat {fp}")
                if ex != 0:
                    print(f"    cat {fp} failed (exit={ex})")
                    continue
                print(f"    {file_path_key}: {len(src_out.splitlines())} lines")
                result = _isrc(src_out, callables)
                changed = result != src_out
                print(f"    instrument_source changed: {changed}")
                if not changed:
                    # Show what instrument_source tried to do
                    for c in callables:
                        sl = c['start_line']
                        lines = src_out.splitlines()
                        if sl - 1 < len(lines):
                            print(f"    line {sl}: {lines[sl-1][:120]}")
                        else:
                            print(f"    line {sl}: OUT OF RANGE (file has {len(lines)} lines)")
            return

        # Step 5: show instrumented source
        instr_out, _, _ = env._execute_raw(f"cat {full_path}")
        print(f"\n[5] Instrumented source around first patched callable:")
        instr_lines = instr_out.splitlines()
        # Find the try/import _swe_fault_tracer line
        for i, line in enumerate(instr_lines):
            if "_swe_fault_tracer" in line:
                start = max(0, i - 3)
                end = min(len(instr_lines), i + 8)
                for j in range(start, end):
                    marker = ">>>" if "_swe_fault_tracer" in instr_lines[j] else "   "
                    print(f"    {marker} {j+1:4d} | {instr_lines[j]}")
                print(f"    ...")
                break

        # Step 6: verify tracer importable
        print(f"\n[6] Verifying tracer module importable...")
        verify_out, verify_err = env._run(
            "python -c \"import _swe_fault_tracer; print('TRACER_OK')\""
        )
        print(f"    Result: {verify_out.strip()}")
        if "TRACER_OK" not in verify_out:
            print(f"    ERROR: {verify_err}")

        # Step 7: clear trace file and run tests
        env._run(f"rm -f {TRACE_FILE_PATH}")

        test_script = (
            "/run_tests.sh" if env.swebench_verified
            else f"{env.alt_path}/run_tests.sh"
        )

        # Show test script content
        script_out, _, _ = env._execute_raw(f"cat {test_script}")
        print(f"\n[7] Test script ({test_script}):")
        for line in script_out.strip().splitlines():
            print(f"    {line}")

        # Inject -rA for R2E-Gym
        if not env.swebench_verified:
            env._run(
                f"sed -i '/pytest/{{/-rA/!s/pytest/pytest -rA/}}' {test_script}"
            )
            script_out2, _, _ = env._execute_raw(f"cat {test_script}")
            print(f"\n    After -rA injection:")
            for line in script_out2.strip().splitlines():
                print(f"    {line}")

        print(f"\n[8] Running tests (timeout=300s)...")
        stdout, stderr, exit_code = env._execute_raw(
            f"bash {test_script}", timeout=300
        )

        print(f"\n[8a] Test STDOUT ({len(stdout)} chars, exit={exit_code}):")
        # Show last 100 lines of stdout (summary is at the end)
        stdout_lines = stdout.splitlines()
        if len(stdout_lines) > 100:
            print(f"    ... ({len(stdout_lines) - 100} lines omitted) ...")
        for line in stdout_lines[-100:]:
            print(f"    {line}")

        if stderr:
            print(f"\n[8b] Test STDERR ({len(stderr)} chars):")
            stderr_lines = stderr.splitlines()
            if len(stderr_lines) > 50:
                print(f"    ... ({len(stderr_lines) - 50} lines omitted) ...")
            for line in stderr_lines[-50:]:
                print(f"    {line}")

        # Step 9: check trace file
        print(f"\n[9] Trace file contents:")
        trace_out, _, trace_exit = env._execute_raw(f"cat {TRACE_FILE_PATH}")
        if trace_exit != 0:
            print(f"    MISSING — trace file does not exist (exit={trace_exit})")
            # Check if the file was created at all
            ls_out, _, _ = env._execute_raw("ls -la /tmp/_swe_fault*")
            print(f"    ls /tmp/_swe_fault*: {ls_out.strip() or 'no files'}")
        elif not trace_out.strip():
            print(f"    EMPTY — trace file exists but is empty")
        else:
            trace_lines = trace_out.strip().splitlines()
            print(f"    {len(trace_lines)} trace entries:")
            for i, line in enumerate(trace_lines[:20]):
                try:
                    entry = json.loads(line)
                    frames = entry.get("frames", [])
                    print(f"    [{i}] callable={entry.get('callable')} "
                          f"frames={len(frames)}")
                    for f in frames:
                        rel = f["file"].replace(env.repo_path + "/", "")
                        print(f"        {rel}:{f['line']}:{f['name']}")
                except json.JSONDecodeError:
                    print(f"    [{i}] RAW: {line[:120]}")
            if len(trace_lines) > 20:
                print(f"    ... and {len(trace_lines) - 20} more")

        # Step 10: parse F2P
        raw_output = f"{stdout}\n{stderr}" if stderr else stdout
        from rllm.environments.swe.reward import parse_log_pytest

        test_status = parse_log_pytest(raw_output)
        print(f"\n[10] Parsed test status ({len(test_status)} tests):")
        for name, status in sorted(test_status.items()):
            print(f"    {status:8s}  {name}")

        if not test_status:
            print("    parse_log_pytest returned empty! "
                  "'short test summary info' probably missing from output.")

        # Step 11: build bonus map (same logic as precompute_bonus_maps.py)
        print(f"\n[11] Building bonus map...")
        traces = parse_fault_traces_from_file(env, instrumented, env.repo_path, env.alt_path)
        print(f"    Raw traces: {len(traces)}")

        # F2P filtering (inlined to avoid cross-script import issues)
        f2p_funcs = None
        if not env.swebench_verified:
            # R2E-Gym: FAILED tests on buggy code = F2P tests
            failed = set()
            for name, status in test_status.items():
                if status in ("FAILED", "ERROR"):
                    bare = name.rsplit(".", 1)[-1] if "." in name else name
                    if bare:
                        failed.add(bare)
            f2p_funcs = failed if failed else None
        else:
            # SWE-Bench: use FAIL_TO_PASS field
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
                    f2p_list = []
                f2p_funcs = {str(t).split("::")[-1] for t in f2p_list} or None

        print(f"    F2P funcs: {f2p_funcs}")
        _FIXTURE_NAMES = {"setUp", "tearDown", "setUpClass", "tearDownClass",
                          "asyncSetUp", "asyncTearDown"}
        if f2p_funcs is not None:
            pre = len(traces)
            filtered = []
            for trace in traces:
                keep = False
                for frame in trace:
                    fp = frame.get("file_path", "")
                    if not _is_test_file(fp):
                        continue
                    fn = frame.get("func_name", "")
                    if fn in f2p_funcs or fn in _FIXTURE_NAMES:
                        keep = True
                        break
                if keep:
                    filtered.append(trace)
            traces = filtered
            print(f"    After F2P filter: {pre} → {len(traces)}")

        # Show F2P traces with role annotations
        print(f"\n    F2P traces ({len(traces)}):")
        for i, trace in enumerate(traces):
            print(f"    [{i}] ({len(trace)} frames):")
            for frame in trace:
                fp = frame.get("file_path", "")
                fn = frame.get("func_name", "")
                qn = frame.get("qualified_name", fn)
                ln = frame.get("line_no", 0)
                patched = frame.get("is_patched", False)
                is_test = _is_test_file(fp)
                if is_test:
                    role = "TEST-ENTRY"
                elif patched:
                    role = "PATCHED"
                else:
                    role = "INTERMEDIATE"
                print(f"        {role:12s}  {fp}:{ln}:{qn}")

        traces = aggregate_traces(traces)
        print(f"\n    After aggregation: {len(traces)}")

        def _read_file(rel_path: str) -> str:
            out, _, exit_code = env._execute_raw(f"cat {env.repo_path}/{rel_path}")
            return out if exit_code == 0 else ""

        result = build_call_graph_from_traces(traces, all_modified, file_reader=_read_file)
        result["instance_id"] = instance_id

        # Save to data/<instance_id>_debug.json
        output_path = os.path.join(
            Path(__file__).resolve().parent.parent, "data", f"{instance_id}_debug.json"
        )
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"    Bonus map saved to: {output_path}")
        print(f"    Nodes: {len(result['call_graph_nodes'])}, "
              f"hop_max: {result['hop_max']}, traceable: {result['traceable']}")
        for key, node in sorted(result["call_graph_nodes"].items(),
                                 key=lambda x: x[1]["hop_distance"]):
            print(f"    d={node['hop_distance']} (norm={node['normalized_distance']:.2f})  {key}")

    except Exception as e:
        import traceback as tb
        print(f"\n[ERROR] {e}")
        tb.print_exc()
    finally:
        print(f"\n[cleanup] Closing sandbox...")
        env.close()

        # Clean up the debug experiment to avoid resource leaks
        exp_id = os.environ.get("ARL_EXPERIMENT_ID", "debug-trace")
        gw_url = os.environ.get("ARL_GATEWAY_URL", "http://118.145.210.10:8080")
        try:
            from arl import GatewayClient
            client = GatewayClient(base_url=gw_url)
            client.delete_experiment(exp_id)
            client.close()
            print(f"Cleaned up experiment '{exp_id}'.")
        except Exception as ce:
            print(f"Cleanup warning: {ce}")
        print("Done.")


if __name__ == "__main__":
    main()
