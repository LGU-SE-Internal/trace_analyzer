#!/usr/bin/env python3
"""Analyze F5 unobserved-GT prevalence in precomputed bonus maps.

Consumes the flat directory of ``<instance_id>.json`` files written by
``precompute_bonus_maps.py`` and reports how much static GT mass sits in cases
where no usable runtime trace contributed to the map.

Usage::

    python -m utils.p2a.analyze_f5_unobserved_gt data/swe/bonus_maps
    python -m utils.p2a.analyze_f5_unobserved_gt data/swe/bonus_maps \\
        --output cache/f5_prevalence/report.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Ensure project root is on the path so we can import rllm
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_is_test_file():
    try:
        from rllm.environments.swe.trace import _is_test_file as helper

        return helper
    except ModuleNotFoundError as exc:
        # In lightweight --no-sync environments, importing rllm executes
        # rllm/__init__.py and may fail on unrelated heavy dependencies such
        # as torch. Load the trace module file directly so this analyzer remains
        # importable while still using trace.py's implementation.
        if exc.name != "torch":
            raise

    trace_path = PROJECT_ROOT / "rllm" / "environments" / "swe" / "trace.py"
    spec = importlib.util.spec_from_file_location("_f5_trace_module", trace_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load trace module from {trace_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._is_test_file


_is_test_file = _load_is_test_file()


def _classify_bonus_map_fallback(bm: dict) -> dict:
    instance_id = bm.get("instance_id", "unknown")
    patched = bm.get("patched_callables", [])
    nodes = bm.get("call_graph_nodes", {})
    hop_max = bm.get("hop_max", 0)

    n_patched = len(patched)
    n_nodes = len(nodes)
    n_test_entries = 0
    n_intermediate = 0
    for node in nodes.values():
        if not isinstance(node, dict):
            continue
        normalized = _as_float(node.get("normalized_distance"))
        if _is_test_file(node.get("file_path", "")):
            n_test_entries += 1
        elif normalized is not None and normalized > 0.0:
            n_intermediate += 1

    case_type = bm.get("case_type")
    if not case_type:
        if not patched:
            case_type = "newly_created" if bm.get("newly_created_callables") else "no_callable"
        elif n_test_entries > 0 and n_intermediate > 0:
            case_type = "standard"
        elif n_test_entries > 0:
            case_type = "direct"
        elif n_nodes > 0:
            case_type = "no_f2p"
        else:
            case_type = "no_trace"

    return {
        "instance_id": instance_id,
        "n_patched": n_patched,
        "n_nodes": n_nodes,
        "n_test_entries": n_test_entries,
        "n_intermediate": n_intermediate,
        "hop_max": hop_max,
        "case_type": case_type,
        "error": bm.get("error", case_type in ("no_trace", "no_gt", "no_f2p", "all_pass")),
        "category": "traceable" if case_type in ("direct", "standard") else "untraceable",
    }


def _load_classify_bonus_map():
    try:
        from utils.p2a.analyze_traceability import classify_bonus_map as helper

        return helper
    except ModuleNotFoundError as exc:
        # analyze_traceability imports broad project dependencies at module
        # import time. Fall back to the equivalent bonus-map classifier when
        # those unrelated dependencies are unavailable.
        if exc.name not in {"torch", "pandas"}:
            raise
        return _classify_bonus_map_fallback


classify_bonus_map = _load_classify_bonus_map()

TRACEABLE_CASE_TYPES = {"direct", "standard"}
NO_USABLE_TRACE_CASE_TYPES = {"no_trace", "no_gt", "no_f2p", "all_pass", "newly_created", "no_callable"}
STANDARD_CASE_ORDER = ["direct", "standard", "newly_created", "no_callable", "no_trace", "no_gt", "no_f2p", "all_pass"]

LIMITATIONS = [
    "Bonus map JSONs do not preserve per-frame is_patched evidence, so hop_distance=0 and normalized_distance=0.0 cannot distinguish an actually observed patched frame from an F5 static seed.",
    "A definitive traceable-instance split needs precompute_bonus_maps.py to write a per-node observed_in_trace boolean, or equivalent per-node/per-frame trace evidence.",
    "Traceable-instance metrics below are therefore sanity checks and inflation proxies, while error=True static GT mass is the hard lower bound of unobserved-but-seeded GT.",
]


def _pct(part: int, total: int) -> float:
    return 100.0 * part / total if total else 0.0


def _bar(part: int, total: int, width: int = 40) -> str:
    filled = round(part / total * width) if total else 0
    return "#" * filled + "." * (width - filled)


def _sorted_counter(counter: Counter) -> dict[str, int]:
    return {str(k): counter[k] for k in sorted(counter)}


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _static_gt_from_bonus_map(bm: dict) -> set[tuple[str, str]]:
    static_gt: set[tuple[str, str]] = set()
    for callable_info in bm.get("patched_callables", []):
        if not isinstance(callable_info, dict):
            continue
        file_path = callable_info.get("file_path")
        qualified_name = callable_info.get("qualified_name")
        if file_path and qualified_name:
            static_gt.add((str(file_path), str(qualified_name)))
    return static_gt


def _node_identity(node_key: str, node: dict) -> tuple[str, str] | None:
    if "::" in node_key:
        key_file_path, key_qualified_name = node_key.rsplit("::", 1)
        file_path = node.get("file_path") or key_file_path
        qualified_name = node.get("qualified_name") or node.get("func_name") or key_qualified_name
    else:
        file_path = node.get("file_path")
        qualified_name = node.get("qualified_name") or node.get("func_name")

    if not file_path or not qualified_name:
        return None
    return str(file_path), str(qualified_name)


def _node_file_path(node_key: str, node: dict) -> str:
    if node.get("file_path"):
        return str(node["file_path"])
    if "::" in node_key:
        return node_key.rsplit("::", 1)[0]
    return ""


def _is_hop_zero_node(node: dict) -> bool:
    normalized = _as_float(node.get("normalized_distance"))
    return node.get("hop_distance") == 0 and normalized == 0.0


def _is_intermediate_node(node: dict) -> bool:
    normalized = _as_float(node.get("normalized_distance"))
    return normalized is not None and normalized > 0.0


def _validate_bonus_map(path: Path, bm: Any) -> str | None:
    if not isinstance(bm, dict):
        return f"{path.name}: top-level JSON value is not an object"
    if "patched_callables" not in bm or not isinstance(bm["patched_callables"], list):
        return f"{path.name}: missing or invalid patched_callables"
    if "call_graph_nodes" not in bm or not isinstance(bm["call_graph_nodes"], dict):
        return f"{path.name}: missing or invalid call_graph_nodes"
    return None


def load_bonus_maps(bonus_maps_dir: str | Path) -> tuple[list[tuple[Path, dict]], list[str]]:
    """Load valid bonus map JSON files from a flat directory."""
    root = Path(bonus_maps_dir)
    loaded: list[tuple[Path, dict]] = []
    warnings: list[str] = []

    if not root.exists():
        return loaded, [f"{root}: directory does not exist"]
    if not root.is_dir():
        return loaded, [f"{root}: not a directory"]

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

        loaded.append((path, bm))

    return loaded, warnings


def analyze_bonus_map(path: Path, bm: dict) -> dict:
    """Compute F5 prevalence metrics for one bonus map."""
    classification = classify_bonus_map(bm)
    instance_id = classification.get("instance_id") or bm.get("instance_id") or path.stem
    case_type = classification["case_type"]
    is_traceable = case_type in TRACEABLE_CASE_TYPES
    is_error = bool(classification.get("error"))
    static_gt = _static_gt_from_bonus_map(bm)
    nodes = bm.get("call_graph_nodes", {})

    seeded_static_gt: set[tuple[str, str]] = set()
    claimed_observed_gt: set[tuple[str, str]] = set()
    non_test_nodes: set[tuple[str, str]] = set()
    intermediate_nodes: set[tuple[str, str]] = set()

    for node_key, node in nodes.items():
        if not isinstance(node, dict):
            continue
        file_path = _node_file_path(str(node_key), node)
        identity = _node_identity(str(node_key), node)
        if identity is None:
            continue

        is_test_node = _is_test_file(file_path)
        if not is_test_node:
            non_test_nodes.add(identity)
            if _is_intermediate_node(node):
                intermediate_nodes.add(identity)

        if identity in static_gt:
            seeded_static_gt.add(identity)
            if not is_test_node and _is_hop_zero_node(node):
                claimed_observed_gt.add(identity)

    missing_seeded_static_gt = static_gt - seeded_static_gt
    static_gt_count = len(static_gt)
    non_test_node_count = len(non_test_nodes)
    intermediate_node_count = len(intermediate_nodes)

    return {
        "instance_id": str(instance_id),
        "source_file": path.name,
        "case_type": case_type,
        "error": is_error,
        "traceable": is_traceable,
        "no_usable_trace_case": case_type in NO_USABLE_TRACE_CASE_TYPES,
        "static_gt_count": static_gt_count,
        "seeded_static_gt_count": len(seeded_static_gt),
        "claimed_observed_gt_count": len(claimed_observed_gt),
        "missing_seeded_static_gt_count": len(missing_seeded_static_gt),
        "missing_seeded_static_gt": [f"{file_path}::{qualified_name}" for file_path, qualified_name in sorted(missing_seeded_static_gt)],
        "non_test_node_count": non_test_node_count,
        "intermediate_node_count": intermediate_node_count,
        "static_gt_minus_non_test_nodes": static_gt_count - non_test_node_count,
        "static_gt_minus_intermediate_nodes": static_gt_count - intermediate_node_count,
        "error_static_gt_mass": static_gt_count if is_error else 0,
        "no_usable_trace_static_gt_mass": static_gt_count if case_type in NO_USABLE_TRACE_CASE_TYPES else 0,
        "traceable_static_gt_mass": static_gt_count if is_traceable else 0,
    }


def analyze_bonus_maps_dir(bonus_maps_dir: str | Path) -> dict:
    loaded, warnings = load_bonus_maps(bonus_maps_dir)
    instances = [analyze_bonus_map(path, bm) for path, bm in loaded]

    total = len(instances)
    case_counts = Counter(instance["case_type"] for instance in instances)
    static_gt_mass = sum(instance["static_gt_count"] for instance in instances)
    error_instances = [instance for instance in instances if instance["error"]]
    traceable_instances = [instance for instance in instances if instance["traceable"]]
    no_usable_trace_instances = [instance for instance in instances if instance["no_usable_trace_case"]]

    traceable_non_test_delta = Counter(instance["static_gt_minus_non_test_nodes"] for instance in traceable_instances)
    traceable_intermediate_delta = Counter(instance["static_gt_minus_intermediate_nodes"] for instance in traceable_instances)
    missing_seeded_instances = [instance for instance in traceable_instances if instance["missing_seeded_static_gt_count"] > 0]
    positive_intermediate_delta_instances = [instance for instance in traceable_instances if instance["static_gt_minus_intermediate_nodes"] > 0]

    return {
        "bonus_maps_dir": str(Path(bonus_maps_dir)),
        "total_instances_scanned": total,
        "files_skipped": len(warnings),
        "warnings": warnings,
        "case_type_counts": {case_type: case_counts[case_type] for case_type in sorted(case_counts)},
        "static_gt_mass": static_gt_mass,
        "error_instance_count": len(error_instances),
        "error_static_gt_mass": sum(instance["error_static_gt_mass"] for instance in error_instances),
        "no_usable_trace_instance_count": len(no_usable_trace_instances),
        "no_usable_trace_static_gt_mass": sum(instance["no_usable_trace_static_gt_mass"] for instance in no_usable_trace_instances),
        "traceable_instance_count": len(traceable_instances),
        "traceable_static_gt_mass": sum(instance["traceable_static_gt_mass"] for instance in traceable_instances),
        "traceable_missing_seeded_static_gt_instance_count": len(missing_seeded_instances),
        "traceable_missing_seeded_static_gt_total": sum(instance["missing_seeded_static_gt_count"] for instance in traceable_instances),
        "traceable_static_gt_minus_non_test_nodes_distribution": _sorted_counter(traceable_non_test_delta),
        "traceable_static_gt_minus_intermediate_nodes_distribution": _sorted_counter(traceable_intermediate_delta),
        "traceable_positive_static_gt_minus_intermediate_nodes_count": len(positive_intermediate_delta_instances),
        "instances": instances,
        "limitations": LIMITATIONS,
    }


def _print_case_breakdown(case_counts: dict[str, int], total: int) -> None:
    print("\nCase type breakdown:")
    printed = set()
    for case_type in STANDARD_CASE_ORDER:
        count = case_counts.get(case_type, 0)
        printed.add(case_type)
        print(f"  {case_type:<18s} {count:5d}  ({_pct(count, total):5.1f}%)")
    for case_type, count in sorted(case_counts.items()):
        if case_type not in printed:
            print(f"  {case_type:<18s} {count:5d}  ({_pct(count, total):5.1f}%)")


def _print_examples(instances: list[dict]) -> None:
    print("\nExamples:")
    by_case_type: dict[str, list[dict]] = {}
    for instance in instances:
        by_case_type.setdefault(instance["case_type"], []).append(instance)

    printed = set()
    for case_type in STANDARD_CASE_ORDER + sorted(by_case_type):
        if case_type in printed:
            continue
        printed.add(case_type)
        items = by_case_type.get(case_type, [])
        if not items:
            continue
        print(f"  [{case_type}]")
        for instance in items[:3]:
            print(
                f"    {instance['instance_id']}  "
                f"(static_gt={instance['static_gt_count']}, claimed_hop0_gt={instance['claimed_observed_gt_count']}, "
                f"non_test_nodes={instance['non_test_node_count']})"
            )


def print_f5_report(summary: dict) -> None:
    total = summary["total_instances_scanned"]
    static_gt_mass = summary["static_gt_mass"]
    traceable_mass = summary["traceable_static_gt_mass"]
    error_mass = summary["error_static_gt_mass"]

    print(f"\n{'=' * 64}")
    print(f"F5 Unobserved-GT Prevalence: {total} instances from {summary['bonus_maps_dir']}")
    print(f"{'=' * 64}")

    if summary["files_skipped"]:
        print(f"\nSkipped files: {summary['files_skipped']} (see warnings on stderr)")

    if total == 0:
        print("\nNo valid bonus map JSONs found.")
        print("\nLimitations:")
        for limitation in summary["limitations"]:
            print(f"  - {limitation}")
        print()
        return

    traceable_instances = summary["traceable_instance_count"]
    print(f"\nTraceable  [{_bar(traceable_instances, total)}]  {traceable_instances}/{total} ({_pct(traceable_instances, total):.1f}%)")
    print(f"\nTotal instances scanned: {total}")
    _print_case_breakdown(summary["case_type_counts"], total)

    print("\nStatic GT mass:")
    print(f"  total_static_gt_mass       {static_gt_mass:5d}")
    print(f"  error=True mass            {error_mass:5d}  ({_pct(error_mass, static_gt_mass):5.1f}%)")
    print(f"  traceable mass             {traceable_mass:5d}  ({_pct(traceable_mass, static_gt_mass):5.1f}%)")
    print(f"  no-usable-trace mass       {summary['no_usable_trace_static_gt_mass']:5d}  ({_pct(summary['no_usable_trace_static_gt_mass'], static_gt_mass):5.1f}%)")

    print("\nF5 lower-bound mass:")
    print(f"  error=True instances       {summary['error_instance_count']:5d}")
    print(f"  unobserved seeded GT       {error_mass:5d}  ({_pct(error_mass, static_gt_mass):5.1f}% of static_gt)")

    print("\nTraceable sanity checks:")
    print(f"  missing seeded static GT instances  {summary['traceable_missing_seeded_static_gt_instance_count']:5d}")
    print(f"  missing seeded static GT total      {summary['traceable_missing_seeded_static_gt_total']:5d}")
    if summary["traceable_missing_seeded_static_gt_total"]:
        print("  FLAG: Some traceable static GT entries are absent from call_graph_nodes; current F5-affected JSONs are expected to report zero here.")

    print("\nTraceable inflation proxies:")
    print("  static_gt - non_test_call_graph_nodes:")
    for delta, count in summary["traceable_static_gt_minus_non_test_nodes_distribution"].items():
        print(f"    {int(delta):>4d}: {count:5d} instance(s)")
    print("  static_gt - intermediate_nodes:")
    for delta, count in summary["traceable_static_gt_minus_intermediate_nodes_distribution"].items():
        print(f"    {int(delta):>4d}: {count:5d} instance(s)")
    print(f"  positive static_gt-minus-intermediate count: {summary['traceable_positive_static_gt_minus_intermediate_nodes_count']}")

    _print_examples(summary["instances"])

    print("\nLimitations:")
    for limitation in summary["limitations"]:
        print(f"  - {limitation}")
    print()


def write_json_output(summary: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze F5 unobserved-GT prevalence over precomputed bonus map JSONs.")
    parser.add_argument("bonus_maps_dir", help="Flat directory containing <instance_id>.json bonus maps")
    parser.add_argument("--output", help="Optional path for structured JSON output")
    args = parser.parse_args(argv)

    summary = analyze_bonus_maps_dir(args.bonus_maps_dir)
    for warning in summary["warnings"]:
        print(f"[WARN] {warning}", file=sys.stderr)

    print_f5_report(summary)

    if args.output:
        write_json_output(summary, args.output)


if __name__ == "__main__":
    main()
