#!/usr/bin/env python3
"""Measure F1 decorator-only diff prevalence in SWE parquet datasets.

The F1 traceability gap is a static AST-diff issue: the current callable source
snippet starts at the ``def`` line, so changes that only touch a function's
decorators can be missed. This script scans dataset file diffs for those hunks
and checks them against the live ``find_modified_callables_from_sources``
implementation from ``rllm/environments/swe/trace.py``.

Usage::

    uv run python -m utils.p2a.analyze_f1_decorator_prevalence \
        data/swe/R2E_Gym_Subset.parquet \
        --output cache/f1_prevalence/report.json
"""

from __future__ import annotations

import argparse
import ast
import difflib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

CATEGORY_SWAP = "decorator-only-swap"
CATEGORY_ADD_OR_REMOVE = "decorator-add-or-remove-around-unchanged-def"
CATEGORY_F1 = "f1-affected"

_DIFF_TEXT_KEYS = ("diff_content", "unified_diff", "diff", "patch")
_HUNK_HEADER_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_len>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_len>\d+))? @@"
)
_TRACE_MODULE: ModuleType | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_trace_module() -> ModuleType:
    """Load the live trace.py module by path without importing rllm.__init__."""
    global _TRACE_MODULE
    if _TRACE_MODULE is not None:
        return _TRACE_MODULE

    trace_path = _repo_root() / "rllm" / "environments" / "swe" / "trace.py"
    spec = importlib.util.spec_from_file_location("_trace_for_f1_prevalence", trace_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load trace module from {trace_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _TRACE_MODULE = module
    return module


def _find_modified_callables_from_sources(
    old_source: str,
    new_source: str,
    file_path: str,
) -> list[dict[str, Any]]:
    trace_module = _load_trace_module()
    return trace_module.find_modified_callables_from_sources(old_source, new_source, file_path)


# This mirrors the small helper in analyze_traceability.py. We keep it local
# because importing that analyzer imports rllm at module import time in this
# worktree, which pulls optional runtime dependencies that are absent locally.
def _parse_file_diffs(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract file_diffs from whichever parsed commit field is present."""
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
            file_diffs = raw.get("file_diffs", [])
            if isinstance(file_diffs, list) and file_diffs:
                return file_diffs
    return []


def _is_test_file(path: str) -> bool:
    """Heuristic: does this path look like a test file?"""
    parts = path.replace("\\", "/").split("/")
    for part in parts:
        if part in ("tests", "test", "testing"):
            return True
        if part.startswith("test_") or part.endswith("_test.py"):
            return True
    return False


def _normalize_path(path: str) -> str:
    path = path.replace("\\", "/")
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def _file_diff_path(file_diff: dict[str, Any]) -> str:
    header = file_diff.get("header") if isinstance(file_diff.get("header"), dict) else {}
    header_file = header.get("file") if isinstance(header.get("file"), dict) else {}
    candidates = [
        header_file.get("path"),
        file_diff.get("path"),
        file_diff.get("file_path"),
        file_diff.get("new_path"),
        file_diff.get("old_path"),
    ]

    plus_file = file_diff.get("plus_file") if isinstance(file_diff.get("plus_file"), dict) else {}
    minus_file = file_diff.get("minus_file") if isinstance(file_diff.get("minus_file"), dict) else {}
    candidates.extend([plus_file.get("path"), minus_file.get("path")])

    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return _normalize_path(candidate)
    return ""


def _non_test_py_diffs(
    file_diffs: list[dict[str, Any]],
    task: dict[str, Any],
) -> list[dict[str, Any]]:
    """Filter file_diffs to non-test Python files."""
    relevant = task.get("relevant_files")
    allow_set: set[str] | None = None
    if relevant is not None and not isinstance(relevant, str):
        try:
            allow_set = {_normalize_path(str(path)) for path in relevant}
        except TypeError:
            allow_set = None

    result: list[dict[str, Any]] = []
    for file_diff in file_diffs:
        path = _file_diff_path(file_diff)
        if not path or not path.endswith(".py"):
            continue
        if allow_set is not None:
            if path not in allow_set:
                continue
        elif _is_test_file(path):
            continue
        result.append(file_diff)
    return result


def _file_sources(file_diff: dict[str, Any]) -> tuple[str, str]:
    old_source = file_diff.get("old_file_content")
    new_source = file_diff.get("new_file_content")
    if old_source is None:
        old_source = file_diff.get("old_content", file_diff.get("source_before", ""))
    if new_source is None:
        new_source = file_diff.get("new_content", file_diff.get("source_after", ""))
    return str(old_source or ""), str(new_source or "")


def _diff_from_structured_hunks(file_diff: dict[str, Any], path: str) -> str:
    hunks = file_diff.get("hunks", [])
    if not isinstance(hunks, list) or not hunks:
        return ""

    minus_file = file_diff.get("minus_file") if isinstance(file_diff.get("minus_file"), dict) else {}
    plus_file = file_diff.get("plus_file") if isinstance(file_diff.get("plus_file"), dict) else {}
    minus_path = minus_file.get("path", f"a/{path}")
    plus_path = plus_file.get("path", f"b/{path}")
    lines = [f"diff --git {minus_path} {plus_path}", f"--- {minus_path}", f"+++ {plus_path}"]

    for hunk in hunks:
        descriptor = hunk.get("descriptor", {}) if isinstance(hunk, dict) else {}
        old_range = descriptor.get("old_range", {}) if isinstance(descriptor, dict) else {}
        new_range = descriptor.get("new_range", {}) if isinstance(descriptor, dict) else {}
        old_start = old_range.get("start", 0)
        old_len = old_range.get("length", 0)
        new_start = new_range.get("start", 0)
        new_len = new_range.get("length", 0)
        section = descriptor.get("section", "")
        header = f"@@ -{old_start},{old_len} +{new_start},{new_len} @@"
        if section:
            header += f" {section}"
        lines.append(header)

        line_group = hunk.get("line_group", {}) if isinstance(hunk, dict) else {}
        all_lines = line_group.get("all_lines", []) if isinstance(line_group, dict) else []
        for line_info in all_lines:
            if not isinstance(line_info, dict):
                continue
            content = line_info.get("content", "")
            line_type = line_info.get("type", "context")
            if line_type == "context":
                lines.append(f" {content}")
            elif line_type == "deleted":
                lines.append(f"-{content}")
            elif line_type == "added":
                lines.append(f"+{content}")
    return "\n".join(lines)


def _diff_text_for_file_diff(file_diff: dict[str, Any], path: str) -> str:
    for key in _DIFF_TEXT_KEYS:
        value = file_diff.get(key)
        if isinstance(value, str) and value:
            return value

    structured = _diff_from_structured_hunks(file_diff, path)
    if structured:
        return structured

    old_source, new_source = _file_sources(file_diff)
    return "\n".join(
        difflib.unified_diff(
            old_source.splitlines(),
            new_source.splitlines(),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
    )


def classify_hunk(hunk_text: str) -> str | None:
    """Return the decorator-only category for one unified-diff hunk."""
    added_decorators = 0
    removed_decorators = 0
    added_non_decorator = 0
    removed_non_decorator = 0

    for line in hunk_text.splitlines():
        if (
            not line
            or line.startswith("@@")
            or line.startswith("diff --git ")
            or line.startswith("--- ")
            or line.startswith("+++ ")
            or line.startswith("\\")
        ):
            continue
        if line.startswith("+"):
            content = line[1:].lstrip()
            if content.startswith("@"):
                added_decorators += 1
            else:
                added_non_decorator += 1
        elif line.startswith("-"):
            content = line[1:].lstrip()
            if content.startswith("@"):
                removed_decorators += 1
            else:
                removed_non_decorator += 1

    if added_non_decorator == 0 and removed_non_decorator == 0:
        if added_decorators > 0 and removed_decorators > 0:
            return CATEGORY_SWAP
        if added_decorators > 0 or removed_decorators > 0:
            return CATEGORY_ADD_OR_REMOVE
    return None


def _iter_hunks(diff_text: str) -> list[dict[str, Any]]:
    """Split a unified diff into hunk dicts with changed old/new line numbers."""
    hunks: list[dict[str, Any]] = []
    current: list[str] = []

    for line in diff_text.splitlines():
        if line.startswith("@@"):
            if current:
                hunks.append(_build_hunk(current))
            current = [line]
        elif current:
            current.append(line)

    if current:
        hunks.append(_build_hunk(current))

    if not hunks and diff_text.strip():
        hunks.append(_build_hunk(diff_text.splitlines()))
    return hunks


def _build_hunk(lines: list[str]) -> dict[str, Any]:
    text = "\n".join(lines)
    header = lines[0] if lines else ""
    match = _HUNK_HEADER_RE.match(header)
    old_changed_lines: list[int] = []
    new_changed_lines: list[int] = []

    if match:
        old_line = int(match.group("old_start"))
        new_line = int(match.group("new_start"))
        body_lines = lines[1:]
        for line in body_lines:
            if line.startswith("\\"):
                continue
            if line.startswith("+"):
                new_changed_lines.append(new_line)
                new_line += 1
            elif line.startswith("-"):
                old_changed_lines.append(old_line)
                old_line += 1
            elif line.startswith(" "):
                old_line += 1
                new_line += 1

    return {
        "text": text,
        "old_changed_lines": old_changed_lines,
        "new_changed_lines": new_changed_lines,
    }


def _decorated_function_qnames_by_line(source: str) -> dict[int, set[str]]:
    """Map function decorator source lines to qualified callable names."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    by_line: dict[int, set[str]] = {}

    def assemble(stack: list[tuple[str, str]], leaf_name: str) -> str:
        parts: list[str] = []
        for kind, name in stack:
            parts.append(name)
            if kind == "func":
                parts.append("<locals>")
        parts.append(leaf_name)
        return ".".join(parts)

    def visit(node: ast.AST, stack: list[tuple[str, str]]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, stack + [("class", child.name)])
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qname = assemble(stack, child.name)
                for decorator in child.decorator_list:
                    start = getattr(decorator, "lineno", None)
                    end = getattr(decorator, "end_lineno", start)
                    if start is None or end is None:
                        continue
                    for lineno in range(start, end + 1):
                        by_line.setdefault(lineno, set()).add(qname)
                visit(child, stack + [("func", child.name)])
            else:
                visit(child, stack)

    visit(tree, [])
    return by_line


def _decorated_callable_qnames_for_hunk(
    old_source: str,
    new_source: str,
    hunk: dict[str, Any],
) -> list[str]:
    old_by_line = _decorated_function_qnames_by_line(old_source)
    new_by_line = _decorated_function_qnames_by_line(new_source)
    qnames: set[str] = set()

    for lineno in hunk["old_changed_lines"]:
        qnames.update(old_by_line.get(lineno, set()))
    for lineno in hunk["new_changed_lines"]:
        qnames.update(new_by_line.get(lineno, set()))

    return sorted(qnames)


def _sample_hunk(hunk_text: str, max_lines: int = 12) -> str:
    lines = hunk_text.splitlines()
    if len(lines) <= max_lines:
        return hunk_text
    return "\n".join([*lines[:max_lines], "..."])


def analyze_file_diff(file_diff: dict[str, Any]) -> dict[str, Any]:
    """Analyze one file diff and return decorator/F1 findings."""
    path = _file_diff_path(file_diff)
    old_source, new_source = _file_sources(file_diff)
    diff_text = _diff_text_for_file_diff(file_diff, path)
    hunk_records: list[dict[str, Any]] = []

    for hunk in _iter_hunks(diff_text):
        category = classify_hunk(hunk["text"])
        if category is None:
            continue
        qnames = _decorated_callable_qnames_for_hunk(old_source, new_source, hunk)
        hunk_records.append(
            {
                "file_path": path,
                "category": category,
                "sample": _sample_hunk(hunk["text"]),
                "callable_qnames": qnames,
                "f1_affected": False,
                "_has_changed_line_numbers": bool(
                    hunk["old_changed_lines"] or hunk["new_changed_lines"]
                ),
            }
        )

    modified: list[dict[str, Any]] = []
    if old_source and new_source and old_source != new_source:
        modified = _find_modified_callables_from_sources(old_source, new_source, path)
    if hunk_records and old_source and new_source:
        modified_qnames = {
            item.get("qualified_name")
            for item in modified
            if isinstance(item, dict) and item.get("qualified_name")
        }
        for record in hunk_records:
            qnames = set(record["callable_qnames"])
            if qnames:
                record["f1_affected"] = bool(qnames - modified_qnames)
            elif not record["_has_changed_line_numbers"]:
                record["f1_affected"] = not modified_qnames

    for record in hunk_records:
        record.pop("_has_changed_line_numbers", None)

    categories = {record["category"] for record in hunk_records}
    if any(record["f1_affected"] for record in hunk_records):
        categories.add(CATEGORY_F1)

    return {
        "file_path": path,
        "categories": sorted(categories),
        "hunks": hunk_records,
        "modified_callables": modified,
    }


def _parse_extra_info(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _is_missing_value(value: Any) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def _task_from_row(row: Any) -> dict[str, Any]:
    row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    extra = _parse_extra_info(row_dict.get("extra_info"))
    task = {**extra}
    for key, value in row_dict.items():
        if key != "extra_info" and not _is_missing_value(value):
            task[key] = value
    if extra:
        task["extra_info"] = extra
    return task


def _looks_like_bare_hash(value: str) -> bool:
    return len(value) >= 20 and all(char in "0123456789abcdef" for char in value)


def make_instance_id(task: dict[str, Any], fallback: str = "unknown") -> str:
    """Generate a readable instance ID for SWE-Bench or R2E-Gym rows."""
    instance_id = task.get("instance_id")
    if isinstance(instance_id, str) and instance_id and not _looks_like_bare_hash(instance_id):
        return instance_id

    repo = task.get("repo_name") or task.get("repo") or ""
    commit = task.get("commit_hash") or ""
    if isinstance(repo, str) and isinstance(commit, str) and repo and commit:
        return f"{repo}__{commit[:8]}"
    if isinstance(commit, str) and commit:
        return commit[:12]
    if isinstance(instance_id, str) and instance_id:
        return instance_id
    return fallback


def analyze_instance(task: dict[str, Any], fallback_id: str = "unknown") -> dict[str, Any]:
    file_diffs = _parse_file_diffs(task)
    relevant_diffs = _non_test_py_diffs(file_diffs, task)
    instance_id = make_instance_id(task, fallback_id)

    file_results = [analyze_file_diff(file_diff) for file_diff in relevant_diffs]
    categories: set[str] = set()
    hunk_details: list[dict[str, Any]] = []
    affected_paths: set[str] = set()

    for result in file_results:
        result_categories = set(result["categories"])
        categories.update(result_categories)
        interesting_hunks = [
            hunk
            for hunk in result["hunks"]
            if hunk["category"] in (CATEGORY_SWAP, CATEGORY_ADD_OR_REMOVE) or hunk["f1_affected"]
        ]
        if result_categories:
            affected_paths.add(result["file_path"])
        hunk_details.extend(interesting_hunks)

    return {
        "instance_id": instance_id,
        "has_non_test_py_diff": bool(relevant_diffs),
        "files_analyzed": len(relevant_diffs),
        "files": sorted(affected_paths),
        "categories": sorted(categories),
        "hunks": hunk_details,
    }


def _summarize_instances(source: str, instances: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(instances)
    with_non_test = sum(1 for item in instances if item["has_non_test_py_diff"])
    category_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {
        CATEGORY_SWAP: [],
        CATEGORY_ADD_OR_REMOVE: [],
        CATEGORY_F1: [],
    }

    for item in instances:
        for category in item["categories"]:
            if category in examples:
                category_counts[category] += 1
                if len(examples[category]) < 5:
                    examples[category].append(item["instance_id"])

    def category_summary(category: str) -> dict[str, Any]:
        count = category_counts[category]
        return {
            "instances": count,
            "share_percent": (count / total * 100) if total else 0.0,
            "example_instance_ids": examples[category],
        }

    summary = {
        "source": source,
        "total_instances": total,
        "instances_with_any_non_test_py_diff": with_non_test,
        "instances_with_zero_non_test_py_diff": total - with_non_test,
        "categories": {
            CATEGORY_SWAP: category_summary(CATEGORY_SWAP),
            CATEGORY_ADD_OR_REMOVE: category_summary(CATEGORY_ADD_OR_REMOVE),
            CATEGORY_F1: category_summary(CATEGORY_F1),
        },
    }

    details = [
        {
            "instance_id": item["instance_id"],
            "file_paths": item["files"],
            "categories": item["categories"],
            "hunks": item["hunks"],
        }
        for item in instances
        if item["categories"]
    ]

    return {"summary": summary, "details": details}


def analyze_parquet(path: str | Path) -> dict[str, Any]:
    """Analyze one parquet file and return a structured report."""
    import pandas as pd

    parquet_path = Path(path)
    df = pd.read_parquet(parquet_path)
    instances: list[dict[str, Any]] = []

    for position, (_, row) in enumerate(df.iterrows(), start=1):
        task = _task_from_row(row)
        instances.append(analyze_instance(task, fallback_id=f"{parquet_path.stem}#{position}"))
        if position % 500 == 0:
            print(f"  [{parquet_path.name}] {position}/{len(df)}...", file=sys.stderr)

    return _summarize_instances(str(parquet_path), instances)


def combine_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine multiple per-parquet reports into one aggregate summary."""
    summaries = [report["summary"] for report in reports]
    total = sum(summary["total_instances"] for summary in summaries)
    with_non_test = sum(summary["instances_with_any_non_test_py_diff"] for summary in summaries)
    details = [detail for report in reports for detail in report["details"]]

    categories: dict[str, dict[str, Any]] = {}
    for category in (CATEGORY_SWAP, CATEGORY_ADD_OR_REMOVE, CATEGORY_F1):
        count = sum(summary["categories"][category]["instances"] for summary in summaries)
        example_ids: list[str] = []
        for summary in summaries:
            for instance_id in summary["categories"][category]["example_instance_ids"]:
                if len(example_ids) < 5:
                    example_ids.append(instance_id)
        categories[category] = {
            "instances": count,
            "share_percent": (count / total * 100) if total else 0.0,
            "example_instance_ids": example_ids,
        }

    return {
        "summary": {
            "source": "combined",
            "total_instances": total,
            "instances_with_any_non_test_py_diff": with_non_test,
            "instances_with_zero_non_test_py_diff": total - with_non_test,
            "categories": categories,
        },
        "details": details,
    }


def print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    total = summary["total_instances"]
    source = summary["source"]
    source_name = Path(source).name if source != "combined" else source

    print("=" * 72)
    print(f"  {source_name}  ({total} instances) - F1 Decorator Prevalence")
    print("=" * 72)
    print()
    print(
        "  Non-test .py diffs: "
        f"{summary['instances_with_any_non_test_py_diff']:>5}/{total} "
        f"({summary['instances_with_any_non_test_py_diff'] / total * 100:.1f}%)"
        if total
        else "  Non-test .py diffs:     0/0 (0.0%)"
    )
    print(f"  Zero non-test .py diffs: {summary['instances_with_zero_non_test_py_diff']:>5}")
    print()
    print("  Category prevalence:")
    for category in (CATEGORY_SWAP, CATEGORY_ADD_OR_REMOVE, CATEGORY_F1):
        data = summary["categories"][category]
        print(
            f"    {category:<52s} "
            f"{data['instances']:>5}  ({data['share_percent']:.1f}%)"
        )

    print()
    print("  Examples:")
    for category in (CATEGORY_SWAP, CATEGORY_ADD_OR_REMOVE, CATEGORY_F1):
        examples = summary["categories"][category]["example_instance_ids"]
        print(f"    [{category}]")
        if examples:
            for instance_id in examples:
                print(f"      {instance_id}")
        else:
            print("      none")
    print()


def _json_payload(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if len(reports) == 1:
        return reports[0]
    return {"datasets": reports, "combined": combine_reports(reports)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Analyze F1 decorator-only diff prevalence in SWE parquet datasets.",
    )
    parser.add_argument(
        "parquet_files",
        nargs="+",
        help="Path(s) to parquet dataset files",
    )
    parser.add_argument(
        "--output",
        help="Optional path for a JSON report",
    )
    args = parser.parse_args(argv)

    reports = [analyze_parquet(path) for path in args.parquet_files]
    for report in reports:
        print_report(report)
    if len(reports) > 1:
        print_report(combine_reports(reports))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(_json_payload(reports), indent=2) + "\n")
        print(f"Wrote JSON report to {output_path}")


if __name__ == "__main__":
    main()
