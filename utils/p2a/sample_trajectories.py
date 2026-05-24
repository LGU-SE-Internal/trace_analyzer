#!/usr/bin/env python3
"""Sample bonus-map propagation trajectories and print human-readable chains."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


TRACEABLE_CASE_TYPES = {"direct", "standard"}
ERROR_CASE_TYPES = {"all_pass", "newly_created", "no_callable", "no_f2p", "no_gt", "no_trace"}
SEPARATOR = "=" * 80


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_missing_value(value: Any) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def _is_test_file(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    for part in parts:
        if part in {"test", "tests", "testing"}:
            return True
        if part.startswith("test_") or part.endswith("_test.py"):
            return True
    return False


def _looks_like_bare_hash(value: str) -> bool:
    return len(value) >= 20 and all(char in "0123456789abcdef" for char in value)


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def _task_from_row(row: dict[str, Any]) -> dict[str, Any]:
    extra = _parse_jsonish(row.get("extra_info"))
    task: dict[str, Any] = {}
    if isinstance(extra, dict):
        task.update(extra)
    for key, value in row.items():
        if key != "extra_info" and not _is_missing_value(value):
            task[key] = value
    if isinstance(extra, dict):
        task["extra_info"] = extra
    return task


def _make_instance_id(task: dict[str, Any], fallback: str = "unknown") -> str:
    instance_id = task.get("instance_id")
    if isinstance(instance_id, str) and instance_id and not _looks_like_bare_hash(instance_id):
        return instance_id

    repo = task.get("repo_name") or task.get("repo") or ""
    commit = task.get("commit_hash") or ""
    if not repo:
        docker = task.get("docker_image") or ""
        if isinstance(docker, str) and "/" in docker:
            image_part = docker.split("/", 1)[1]
            image_name = image_part.split(":", 1)[0]
            repo = image_name[: -len("_final")] if image_name.endswith("_final") else image_name

    if isinstance(repo, str) and isinstance(commit, str) and repo and commit:
        return f"{repo}__{commit[:8]}"
    if isinstance(commit, str) and commit:
        return commit[:12]
    if isinstance(instance_id, str) and instance_id:
        return instance_id
    return fallback


def _node_identity(node_key: str, node: dict[str, Any]) -> tuple[str, str]:
    if "::" in node_key:
        key_file_path, key_qualified_name = node_key.rsplit("::", 1)
    else:
        key_file_path, key_qualified_name = "", node_key

    file_path = str(node.get("file_path") or key_file_path)
    qualified_name = str(node.get("qualified_name") or node.get("func_name") or key_qualified_name)
    return file_path, qualified_name


def classify_case_type(bm: dict[str, Any]) -> str:
    """Return the stored case_type, or derive a conservative legacy fallback."""
    case_type = bm.get("case_type")
    if case_type:
        return str(case_type)

    patched = bm.get("patched_callables")
    patched_callables = patched if isinstance(patched, list) else []
    raw_nodes = bm.get("call_graph_nodes")
    nodes = raw_nodes if isinstance(raw_nodes, dict) else {}

    n_test_entries = 0
    n_intermediate = 0
    for node_key, raw_node in nodes.items():
        if not isinstance(raw_node, dict):
            continue
        file_path, _qualified_name = _node_identity(str(node_key), raw_node)
        hop_distance = _as_int(raw_node.get("hop_distance"))
        normalized_distance = _as_float(raw_node.get("normalized_distance"))
        if _is_test_file(file_path):
            n_test_entries += 1
        elif (hop_distance is not None and hop_distance > 0) or (normalized_distance is not None and normalized_distance > 0.0):
            n_intermediate += 1

    if not patched_callables:
        return "newly_created" if bm.get("newly_created_callables") else "no_callable"
    if n_test_entries > 0 and n_intermediate > 0:
        return "standard"
    if n_test_entries > 0:
        return "direct"
    if nodes:
        return "no_f2p"
    return "no_trace"


def _validate_bonus_map(path: Path, bm: Any) -> str | None:
    if not isinstance(bm, dict):
        return f"{path.name}: top-level JSON value is not an object"
    if "patched_callables" in bm and not isinstance(bm["patched_callables"], list):
        return f"{path.name}: patched_callables is not a list"
    if "call_graph_nodes" in bm and not isinstance(bm["call_graph_nodes"], dict):
        return f"{path.name}: call_graph_nodes is not an object"
    return None


def load_bonus_maps(bonus_maps_dir: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load valid ``<instance_id>.json`` bonus maps from a flat directory."""
    root = Path(bonus_maps_dir)
    records: list[dict[str, Any]] = []
    warnings: list[str] = []

    if not root.exists():
        return records, [f"{root}: directory does not exist"]
    if not root.is_dir():
        return records, [f"{root}: not a directory"]

    for path in sorted(root.glob("*.json")):
        try:
            bm = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            warnings.append(f"{path.name}: skipped malformed JSON ({exc})")
            continue

        invalid_reason = _validate_bonus_map(path, bm)
        if invalid_reason:
            warnings.append(invalid_reason)
            continue

        instance_id = str(bm.get("instance_id") or path.stem)
        case_type = classify_case_type(bm)
        records.append({"path": path, "bonus_map": bm, "instance_id": instance_id, "case_type": case_type})

    return records, warnings


def default_seed_for_dir(bonus_maps_dir: str | Path) -> int:
    """Derive the deterministic default seed from the directory name."""
    return int(hashlib.sha256(Path(bonus_maps_dir).name.encode()).hexdigest()[:8], 16)


def sample_records(records: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Stratified sample across observed case_type values."""
    if n <= 0 or not records:
        return []
    sorted_records = sorted(records, key=lambda record: (str(record["case_type"]), str(record["instance_id"])))
    if len(sorted_records) <= n:
        return sorted_records

    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in sorted_records:
        groups[str(record["case_type"])].append(record)

    present_types = sorted(groups)
    base = n // len(present_types)
    remainder = n % len(present_types)
    allocations = {case_type: base for case_type in present_types}
    remainder_order = sorted(present_types, key=lambda case_type: (-len(groups[case_type]), case_type))
    for case_type in remainder_order[:remainder]:
        allocations[case_type] += 1

    counts: dict[str, int] = {}
    selected_total = 0
    for case_type in present_types:
        count = min(allocations[case_type], len(groups[case_type]))
        counts[case_type] = count
        selected_total += count

    unfilled = n - selected_total
    while unfilled > 0:
        candidates = [case_type for case_type in present_types if counts[case_type] < len(groups[case_type])]
        if not candidates:
            break
        candidates.sort(key=lambda case_type: (-(len(groups[case_type]) - counts[case_type]), -len(groups[case_type]), case_type))
        for case_type in candidates:
            if unfilled <= 0:
                break
            counts[case_type] += 1
            unfilled -= 1

    selected: list[dict[str, Any]] = []
    for case_type in present_types:
        shuffled = list(groups[case_type])
        rng.shuffle(shuffled)
        selected.extend(shuffled[: counts[case_type]])

    return sorted(selected, key=lambda record: (str(record["case_type"]), str(record["instance_id"])))


def sample_bonus_maps(bonus_maps_dir: str | Path, n: int = 10, seed: int | None = None) -> tuple[list[dict[str, Any]], list[str], int]:
    """Load and sample a bonus-map directory."""
    records, warnings = load_bonus_maps(bonus_maps_dir)
    effective_seed = default_seed_for_dir(bonus_maps_dir) if seed is None else seed
    return sample_records(records, n, effective_seed), warnings, effective_seed


def _format_lines(start_line: Any, end_line: Any) -> str:
    start = "?" if start_line is None else str(start_line)
    end = "?" if end_line is None else str(end_line)
    return f"lines {start}-{end}"


def _format_callable(callable_info: dict[str, Any]) -> str:
    file_path = str(callable_info.get("file_path") or "?")
    qualified_name = str(callable_info.get("qualified_name") or callable_info.get("func_name") or "?")
    return f"{file_path}::{qualified_name} ({_format_lines(callable_info.get('start_line'), callable_info.get('end_line'))})"


def _observed_marker(node: dict[str, Any]) -> str:
    if "observed_in_trace" not in node:
        return "?"
    return "Y" if bool(node["observed_in_trace"]) else "N"


def _sorted_call_graph_nodes(nodes: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    valid_nodes = [(str(node_key), node) for node_key, node in nodes.items() if isinstance(node, dict)]

    def sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, str]:
        node_key, node = item
        hop_distance = _as_int(node.get("hop_distance"))
        if hop_distance is None:
            return (1_000_000, node_key)
        return (-hop_distance, node_key)

    return sorted(valid_nodes, key=sort_key)


def _short_text(value: Any, limit: int = 200) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, sort_keys=True)
    else:
        text = str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _extract_tests_from_value(value: Any) -> list[str]:
    parsed = _parse_jsonish(value)
    if isinstance(parsed, dict):
        for key in ("FAIL_TO_PASS", "fail_to_pass", "fail_to_pass_tests", "f2p_tests", "tests"):
            tests = _extract_tests_from_value(parsed.get(key))
            if tests:
                return tests
        failed_tests = []
        for test_name, status in parsed.items():
            if str(status).upper() in {"ERROR", "FAIL", "FAILED"}:
                failed_tests.append(str(test_name))
        return failed_tests
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    if isinstance(parsed, str) and parsed.strip():
        tests = []
        for line in parsed.splitlines():
            stripped = line.strip()
            if stripped.startswith(("ERROR ", "FAILED ")):
                parts = stripped.split()
                if len(parts) >= 2:
                    tests.append(parts[1])
        if tests:
            return tests
        stripped = parsed.strip()
        if "::" in stripped or stripped.startswith("test_"):
            return [stripped]
    return []


def _dataset_hint_from_task(task: dict[str, Any]) -> dict[str, str]:
    tests: list[str] = []
    for key in ("FAIL_TO_PASS", "fail_to_pass", "prediction", "expected_output_json"):
        tests = _extract_tests_from_value(task.get(key))
        if tests:
            break

    problem = ""
    for key in ("problem_statement", "statement", "problem", "issue"):
        problem = _short_text(task.get(key))
        if problem:
            break

    patch = ""
    for key in ("patch", "gold_patch", "original_patch"):
        patch = _short_text(task.get(key))
        if patch:
            break

    return {
        "f2p_tests": ", ".join(tests) if tests else "(unavailable)",
        "problem": problem,
        "patch": patch,
    }


def load_dataset_hints(dataset_path: str | Path) -> dict[str, dict[str, str]]:
    """Load optional problem and F2P hints from a parquet dataset."""
    import pandas as pd

    frame = pd.read_parquet(dataset_path)
    hints: dict[str, dict[str, str]] = {}
    for index, row in enumerate(frame.to_dict("records")):
        task = _task_from_row(row)
        instance_id = _make_instance_id(task, fallback=str(index))
        hints[instance_id] = _dataset_hint_from_task(task)
    return hints


def render_instance(record: dict[str, Any], dataset_hint: dict[str, str] | None = None) -> str:
    bm = record["bonus_map"]
    instance_id = str(record["instance_id"])
    case_type = str(record["case_type"])
    patched_callables = bm.get("patched_callables") if isinstance(bm.get("patched_callables"), list) else []
    unobserved_callables_present = "unobserved_patched_callables" in bm
    unobserved_callables = bm.get("unobserved_patched_callables") if isinstance(bm.get("unobserved_patched_callables"), list) else []
    raw_nodes = bm.get("call_graph_nodes") if isinstance(bm.get("call_graph_nodes"), dict) else {}
    has_observed_field = any(isinstance(node, dict) and "observed_in_trace" in node for node in raw_nodes.values())
    traceable = bool(bm.get("traceable", case_type in TRACEABLE_CASE_TYPES))
    hop_max = bm.get("hop_max", 0)

    lines = [
        SEPARATOR,
        f"INSTANCE: {instance_id}",
        f"case_type: {case_type}",
        f"traceable: {traceable}",
        f"hop_max: {hop_max}",
        f"n_patched_callables: {len(patched_callables)}",
    ]

    if unobserved_callables_present:
        note = "  (good - F5 fix is working on this one)" if len(unobserved_callables) == 0 and traceable else ""
        lines.append(f"n_unobserved_patched_callables: {len(unobserved_callables)}{note}")

    if not has_observed_field and not unobserved_callables_present:
        lines.append("(legacy shape: observed_in_trace and unobserved_patched_callables fields absent)")

    lines.extend(["", "Patched callables (static AST):"])
    if patched_callables:
        for callable_info in patched_callables:
            if isinstance(callable_info, dict):
                lines.append(f"  - {_format_callable(callable_info)}")
    else:
        lines.append("  (empty)")

    if unobserved_callables:
        lines.extend(["", "Unobserved patched callables (F5-affected):"])
        for callable_info in unobserved_callables:
            if isinstance(callable_info, dict):
                lines.append(f"  - {_format_callable(callable_info)}")

    lines.extend(["", "Call graph (test entry -> patched callable):"])
    sorted_nodes = _sorted_call_graph_nodes(raw_nodes)
    if sorted_nodes:
        for node_key, node in sorted_nodes:
            file_path, qualified_name = _node_identity(node_key, node)
            hop_distance = _as_int(node.get("hop_distance"))
            normalized_distance = _as_float(node.get("normalized_distance"))
            hop_text = "?" if hop_distance is None else str(hop_distance)
            norm_text = "?" if normalized_distance is None else f"{normalized_distance:.2f}"
            lines.append(
                f"  [hop={hop_text}, norm={norm_text}, observed={_observed_marker(node)}] "
                f"{file_path}::{qualified_name} ({_format_lines(node.get('start_line'), node.get('end_line'))})"
            )
    else:
        if case_type in ERROR_CASE_TYPES or not traceable:
            lines.append("  (empty - untraceable)")
        else:
            lines.append("  (empty)")

    if dataset_hint is not None:
        lines.extend(["", "Dataset hints:"])
        lines.append(f"  F2P tests: {dataset_hint.get('f2p_tests') or '(unavailable)'}")
        if dataset_hint.get("problem"):
            lines.append(f"  Problem (first 200 chars): \"{dataset_hint['problem']}\"")
        if dataset_hint.get("patch"):
            lines.append(f"  Patch (first 200 chars): \"{dataset_hint['patch']}\"")

    lines.append(SEPARATOR)
    return "\n".join(lines)


def render_report(records: list[dict[str, Any]], dataset_hints: dict[str, dict[str, str]] | None = None) -> str:
    if not records:
        return "No valid bonus map JSONs found.\n"

    if dataset_hints is None:
        blocks = [render_instance(record) for record in records]
    else:
        missing_hint = {"f2p_tests": "(no matching row found)", "problem": "", "patch": ""}
        blocks = [render_instance(record, dataset_hints.get(str(record["instance_id"]), missing_hint)) for record in records]
    return "\n".join(blocks) + "\n"


def build_report(bonus_maps_dir: str | Path, n: int = 10, seed: int | None = None, dataset_path: str | Path | None = None) -> tuple[str, list[str], int]:
    records, warnings, effective_seed = sample_bonus_maps(bonus_maps_dir, n=n, seed=seed)
    dataset_hints = load_dataset_hints(dataset_path) if dataset_path is not None else None
    return render_report(records, dataset_hints), warnings, effective_seed


def write_output(output_path: str | Path, content: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample bonus-map propagation trajectories for manual inspection.")
    parser.add_argument("bonus_maps_dir", help="Flat directory containing <instance_id>.json bonus maps")
    parser.add_argument("--n", type=int, default=10, help="Total number of instances to sample")
    parser.add_argument("--output", help="Optional path to also write the plain-text report")
    parser.add_argument("--dataset", help="Optional parquet dataset path for problem and F2P hints")
    parser.add_argument("--seed", type=int, help="Optional integer seed for reproducible sampling")
    args = parser.parse_args(argv)

    report, warnings, _effective_seed = build_report(args.bonus_maps_dir, n=args.n, seed=args.seed, dataset_path=args.dataset)
    for warning in warnings:
        print(f"[WARN] {warning}", file=sys.stderr)

    print(report, end="")
    if args.output:
        write_output(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
