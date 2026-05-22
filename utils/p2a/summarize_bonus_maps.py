#!/usr/bin/env python3
"""Aggregate summary tool over a directory of precomputed bonus-map JSONs.

Emits one JSON file (schema_version=1) and prints an ASCII table to stdout.
Reuses analyze_traceability.classify_bonus_map with an importlib fallback so
the classification decision tree is not duplicated.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA_VERSION = 1
CASE_TYPES_ORDERED = [
    "newly_created",
    "no_callable",
    "no_trace",
    "no_gt",
    "all_pass",
    "no_f2p",
    "standard",
    "direct",
]
TRACEABLE_CASE_TYPES = {"direct", "standard"}
ERROR_CASE_TYPES = {"no_trace", "no_gt", "no_f2p", "all_pass"}


def _load_is_test_file() -> Callable[[str], bool]:
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
    spec = importlib.util.spec_from_file_location("_summary_trace_module", trace_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load trace module from {trace_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._is_test_file


_is_test_file = _load_is_test_file()


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
        "error": bm.get("error", case_type in ERROR_CASE_TYPES),
        "category": "traceable" if case_type in TRACEABLE_CASE_TYPES else "untraceable",
    }


def _load_classify_bonus_map() -> Callable[[dict], dict]:
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


def _json_paths(bonus_maps_dir: Path) -> list[Path]:
    if not bonus_maps_dir.exists():
        raise FileNotFoundError(f"{bonus_maps_dir}: directory does not exist")
    if not bonus_maps_dir.is_dir():
        raise NotADirectoryError(f"{bonus_maps_dir}: not a directory")
    return sorted(bonus_maps_dir.glob("*.json"))


def _load_bonus_maps(bonus_maps_dir: Path) -> tuple[list[tuple[Path, dict]], list[dict[str, str]], int]:
    root = bonus_maps_dir.resolve()
    paths = _json_paths(root)
    loaded: list[tuple[Path, dict]] = []
    failures: list[dict[str, str]] = []

    for path in paths:
        try:
            value = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            failures.append({"path": str(path), "reason": str(exc)})
            continue

        if not isinstance(value, dict):
            failures.append({"path": str(path), "reason": "top-level JSON value is not an object"})
            continue

        loaded.append((path, value))

    return loaded, failures, len(paths)


def _iter_call_graph_nodes(bm: dict) -> Iterable[dict]:
    nodes = bm.get("call_graph_nodes", {})
    if isinstance(nodes, dict):
        values = nodes.values()
    elif isinstance(nodes, list):
        values = nodes
    else:
        values = ()

    for node in values:
        if isinstance(node, dict):
            yield node


def _list_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _numeric(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return value
    return None


def _mean(values: list[int | float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summarize_f5(loaded: list[tuple[Path, dict]], total: int) -> dict[str, Any]:
    field_present = False
    instances_with_unobserved = 0

    for _path, bm in loaded:
        has_unobserved_node = False
        for node in _iter_call_graph_nodes(bm):
            if "observed_in_trace" in node:
                field_present = True
            if node.get("observed_in_trace") is False:
                has_unobserved_node = True
        if has_unobserved_node:
            instances_with_unobserved += 1

    if not field_present:
        return {
            "observed_in_trace_field_present": False,
            "instances_with_unobserved_node_in_call_graph": None,
            "share_of_total": None,
            "interpretation": "legacy bonus_map JSONs do not include observed_in_trace; F5 prevalence is unavailable",
        }

    return {
        "observed_in_trace_field_present": True,
        "instances_with_unobserved_node_in_call_graph": instances_with_unobserved,
        "share_of_total": instances_with_unobserved / total if total else 0.0,
        "interpretation": "0 means the F5 fix is fully effective",
    }


def _summarize_unobserved_gt(loaded: list[tuple[Path, dict]], total: int) -> dict[str, Any]:
    field_present = any("unobserved_patched_callables" in bm for _path, bm in loaded)
    if not field_present:
        return {
            "field_present": False,
            "total_unobserved_callables": None,
            "mean_per_instance": None,
            "n_instances_with_at_least_one_unobserved": None,
            "share_of_static_gt_mass": None,
        }

    counts = [_list_len(bm.get("unobserved_patched_callables", [])) for _path, bm in loaded]
    total_unobserved = sum(counts)
    total_static_gt_mass = sum(_list_len(bm.get("patched_callables", [])) for _path, bm in loaded)

    return {
        "field_present": True,
        "total_unobserved_callables": total_unobserved,
        "mean_per_instance": total_unobserved / total if total else 0.0,
        "n_instances_with_at_least_one_unobserved": sum(1 for count in counts if count > 0),
        "share_of_static_gt_mass": total_unobserved / total_static_gt_mass if total_static_gt_mass else None,
    }


def _summarize_hops(hop_values: list[int | float]) -> dict[str, Any]:
    if not hop_values:
        return {
            "n_instances_with_hop_data": 0,
            "min": None,
            "mean": None,
            "median": None,
            "max": None,
        }

    return {
        "n_instances_with_hop_data": len(hop_values),
        "min": min(hop_values),
        "mean": _mean(hop_values),
        "median": statistics.median(hop_values),
        "max": max(hop_values),
    }


def summarize(bonus_maps_dir: Path) -> dict[str, Any]:
    root = bonus_maps_dir.resolve()
    loaded, parse_failures, n_scanned = _load_bonus_maps(root)
    n_parsed = len(loaded)
    case_counts: Counter[str] = Counter()
    per_case_samples = {case_type: [] for case_type in CASE_TYPES_ORDERED}
    hop_values: list[int | float] = []

    for path, bm in loaded:
        classification = classify_bonus_map(bm)
        case_type = str(classification.get("case_type", "unknown"))
        instance_id = str(classification.get("instance_id") or bm.get("instance_id") or path.stem)

        case_counts[case_type] += 1
        if case_type in per_case_samples and len(per_case_samples[case_type]) < 5:
            per_case_samples[case_type].append(instance_id)

        hop_max = _numeric(classification.get("hop_max"))
        if case_type in TRACEABLE_CASE_TYPES and hop_max is not None:
            hop_values.append(hop_max)

    histogram = {case_type: case_counts[case_type] for case_type in CASE_TYPES_ORDERED}
    histogram["_total"] = n_parsed
    histogram["_error_total"] = sum(case_counts[case_type] for case_type in ERROR_CASE_TYPES)
    histogram["_traceable_total"] = sum(case_counts[case_type] for case_type in TRACEABLE_CASE_TYPES)

    return {
        "schema_version": SCHEMA_VERSION,
        "bonus_maps_dir": str(root),
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "n_instances_scanned": n_scanned,
        "n_instances_parsed": n_parsed,
        "n_parse_failures": len(parse_failures),
        "parse_failures": parse_failures,
        "case_type_histogram": histogram,
        "f5_prevalence": _summarize_f5(loaded, n_parsed),
        "unobserved_gt_metadata": _summarize_unobserved_gt(loaded, n_parsed),
        "hop_distribution": _summarize_hops(hop_values),
        "per_case_type_samples": per_case_samples,
    }


def _pct(part: int | float, total: int | float) -> float:
    return 100.0 * part / total if total else 0.0


def _format_pct(part: int | float, total: int | float) -> str:
    return f"{_pct(part, total):4.1f}%"


def _format_number(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_table(summary: dict[str, Any]) -> str:
    total = summary["n_instances_parsed"]
    histogram = summary["case_type_histogram"]
    f5 = summary["f5_prevalence"]
    unobserved = summary["unobserved_gt_metadata"]
    hops = summary["hop_distribution"]

    lines = [
        f"== Bonus map summary: {summary['bonus_maps_dir']} ==",
        f"N parsed: {summary['n_instances_parsed']} / {summary['n_instances_scanned']} ({summary['n_parse_failures']} failures)",
        "",
        "Case-type histogram:",
    ]

    for case_type in CASE_TYPES_ORDERED[:6]:
        count = histogram[case_type]
        lines.append(f"  {case_type:<15} {count:5d}   ({_format_pct(count, total)})")

    lines.append("  ----")
    for case_type in CASE_TYPES_ORDERED[6:]:
        count = histogram[case_type]
        lines.append(f"  {case_type:<15} {count:5d}   ({_format_pct(count, total)})   traceable")

    lines.extend(
        [
            "  ----",
            f"  {'TOTAL':<15} {histogram['_total']:5d}",
            f"  {'ERRORS':<15} {histogram['_error_total']:5d}   ({_format_pct(histogram['_error_total'], total)})",
            f"  {'TRACEABLE':<15} {histogram['_traceable_total']:5d}   ({_format_pct(histogram['_traceable_total'], total)})",
            "",
            "F5 prevalence (post-fix expected: 0):",
        ]
    )

    if f5["observed_in_trace_field_present"]:
        count = f5["instances_with_unobserved_node_in_call_graph"]
        lines.append(f"  instances with unobserved hop-0 nodes: {count} ({_format_pct(count, total)})")
    else:
        lines.append("  instances with unobserved hop-0 nodes: n/a (legacy bonus_map JSONs do not include observed_in_trace)")
    lines.append(f"  observed_in_trace field present: {'YES' if f5['observed_in_trace_field_present'] else 'NO'}")

    lines.extend(
        [
            "",
            "Unobserved-GT metadata:",
            f"  total unobserved callables across corpus: {_format_number(unobserved['total_unobserved_callables'], 0)}",
            f"  mean per instance: {_format_number(unobserved['mean_per_instance'], 2)}",
            f"  instances with >=1 unobserved: {_format_number(unobserved['n_instances_with_at_least_one_unobserved'], 0)}",
        ]
    )
    share = unobserved["share_of_static_gt_mass"]
    lines.append(f"  share of static GT mass: {'n/a' if share is None else f'{share * 100.0:.1f}%'}")

    lines.extend(
        [
            "",
            "Hop distribution (traceable instances):",
            "  "
            f"min={_format_number(hops['min'])}  "
            f"mean={_format_number(hops['mean'])}  "
            f"median={_format_number(hops['median'])}  "
            f"max={_format_number(hops['max'])}",
        ]
    )

    return "\n".join(lines)


def write_json(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n")


def _default_output_path(timestamp: str) -> Path:
    safe_timestamp = timestamp.replace(":", "-")
    return PROJECT_ROOT / "cache" / f"bonus_maps_summary_{safe_timestamp}.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate case-type, F5, unobserved-GT, and hop metrics over bonus_map JSONs."
    )
    parser.add_argument("bonus_maps_dir", help="Flat directory containing <instance_id>.json bonus maps")
    parser.add_argument("--output", default=None, help="Optional path for structured JSON output")
    args = parser.parse_args(argv)

    summary = summarize(Path(args.bonus_maps_dir))
    print(render_table(summary))

    output_path = Path(args.output) if args.output else _default_output_path(summary["timestamp"])
    write_json(summary, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
