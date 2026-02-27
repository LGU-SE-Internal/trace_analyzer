#!/usr/bin/env python3
"""Analyze the traceability of SWE dataset instances.

For each instance, compares the old and new source (pre-/post-patch) via AST
to determine how many callables are *modified in-place* — the ones that can
be instrumented for fault tracing.

Supports both:
  - **R2E-Gym** (``parsed_commit_content`` → ``file_diffs``)
  - **SWE-Bench Verified** (``parsed_commit`` → ``file_diffs``)

Usage::

    python scripts/analyze_dataset_traceability.py data/swe/R2E_Gym_Subset.parquet
    python scripts/analyze_dataset_traceability.py data/swe/SWE_Bench_Verified.parquet
    python scripts/analyze_dataset_traceability.py data/swe/*.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# Ensure project root is on the path so we can import rllm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rllm.environments.swe.trace import (
    CallableInfo,
    _is_test_file,
    extract_callables_from_ast,
    normalize_task,
)

# ── helpers ────────────────────────────────────────────────────────────────


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


def _non_test_py_diffs(
    file_diffs: list[dict],
    task: dict,
) -> list[dict]:
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


# ── per-instance analysis ──────────────────────────────────────────────────


def analyze_instance(task: dict) -> dict:
    """Return analysis dict for a single task instance."""
    task = normalize_task(task)
    file_diffs = _parse_file_diffs(task)
    diffs = _non_test_py_diffs(file_diffs, task)

    traceable: list[str] = []  # qualified names
    added: list[str] = []
    deleted: list[str] = []
    has_non_callable_changes = False

    for fd in diffs:
        path = fd["header"]["file"]["path"]
        old_src = fd.get("old_file_content") or ""
        new_src = fd.get("new_file_content") or ""

        old_callables = extract_callables_from_ast(old_src, path) if old_src else {}
        new_callables = extract_callables_from_ast(new_src, path) if new_src else {}

        old_names = set(old_callables)
        new_names = set(new_callables)

        for name in old_names & new_names:
            if old_callables[name].source != new_callables[name].source:
                traceable.append(f"{path}::{name}")

        for name in new_names - old_names:
            added.append(f"{path}::{name}")

        for name in old_names - new_names:
            deleted.append(f"{path}::{name}")

        # Detect non-callable changes: file content differs but no callable-level diffs
        callable_changes = (
            sum(
                1
                for n in old_names & new_names
                if old_callables[n].source != new_callables[n].source
            )
            + len(new_names - old_names)
            + len(old_names - new_names)
        )
        if old_src != new_src and callable_changes == 0:
            has_non_callable_changes = True

    return {
        "traceable": traceable,
        "added": added,
        "deleted": deleted,
        "has_non_callable": has_non_callable_changes,
        "files_analyzed": len(diffs),
    }


def categorize(r: dict) -> str:
    has_t = bool(r["traceable"])
    has_a = bool(r["added"])
    has_d = bool(r["deleted"])
    has_nc = r["has_non_callable"]

    if has_t:
        return "traceable"
    if not has_a and not has_d and not has_nc:
        return "no_changes_detected" if r["files_analyzed"] else "no_non_test_py_files"
    parts = []
    if has_a:
        parts.append("new_callables")
    if has_d:
        parts.append("deleted_callables")
    if has_nc:
        parts.append("non_callable")
    return "only_" + "+".join(parts)


# ── main ───────────────────────────────────────────────────────────────────


def analyze_parquet(path: str) -> list[dict]:
    df = pd.read_parquet(path)
    results = []
    for idx, row in df.iterrows():
        extra_raw = row.get("extra_info", "{}")
        task = json.loads(extra_raw) if isinstance(extra_raw, str) else dict(extra_raw)
        r = analyze_instance(task)
        r["index"] = idx
        r["repo"] = task.get("repo_name", task.get("repo", "?"))
        r["instance_id"] = task.get(
            "instance_id", task.get("commit_hash", "?")
        )[:24]
        r["category"] = categorize(r)
        results.append(r)

        done = idx + 1
        if done % 500 == 0:
            print(f"  [{Path(path).name}] {done}/{len(df)}...", file=sys.stderr)

    return results


def print_report(path: str, results: list[dict]) -> None:
    total = len(results)
    cat_counts = Counter(r["category"] for r in results)
    traceable = [r for r in results if r["category"] == "traceable"]

    print("=" * 72)
    print(f"  {Path(path).name}  ({total} instances)")
    print("=" * 72)
    print()

    # ── overview ──
    n_t = len(traceable)
    n_nt = total - n_t
    bar_w = 40
    filled = round(n_t / total * bar_w) if total else 0
    bar = "#" * filled + "." * (bar_w - filled)
    print(f"  Traceable  [{bar}]  {n_t}/{total} ({n_t/total*100:.1f}%)")
    print()

    # ── non-traceable breakdown ──
    if n_nt:
        print("  Non-traceable breakdown:")
        for cat, cnt in cat_counts.most_common():
            if cat == "traceable":
                continue
            print(f"    {cat:<44s} {cnt:>5}  ({cnt/total*100:.1f}%)")
        print()

    # ── callable count distribution ──
    t_counts = [len(r["traceable"]) for r in traceable]
    if t_counts:
        dist = Counter(t_counts)
        print("  Traceable callable count per instance:")
        for n in sorted(dist):
            print(f"    {n:>2} callable(s): {dist[n]:>5} instances")
        mean = sum(t_counts) / len(t_counts)
        print(f"    mean={mean:.2f}  max={max(t_counts)}")
        print()

    # ── co-occurrence (among traceable) ──
    if traceable:
        also_a = sum(1 for r in traceable if r["added"])
        also_d = sum(1 for r in traceable if r["deleted"])
        also_nc = sum(1 for r in traceable if r["has_non_callable"])
        print("  Among traceable instances, also have:")
        print(f"    + newly added callables:    {also_a:>5} ({also_a/len(traceable)*100:.1f}%)")
        print(f"    + fully deleted callables:  {also_d:>5} ({also_d/len(traceable)*100:.1f}%)")
        print(f"    + non-callable changes:     {also_nc:>5} ({also_nc/len(traceable)*100:.1f}%)")
        print()

    # ── examples ──
    print("  Examples:")
    for cat in ["traceable"] + [
        c for c, _ in cat_counts.most_common() if c != "traceable"
    ]:
        items = [r for r in results if r["category"] == cat]
        if not items:
            continue
        print(f"    [{cat}]")
        for r in items[:2]:
            print(f"      {r['repo']}/{r['instance_id']}")
            if r["traceable"]:
                for c in r["traceable"][:3]:
                    print(f"        -> {c}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze traceability of SWE dataset instances."
    )
    parser.add_argument(
        "parquet_files",
        nargs="+",
        help="Path(s) to parquet dataset files",
    )
    args = parser.parse_args()

    for path in args.parquet_files:
        results = analyze_parquet(path)
        print_report(path, results)


if __name__ == "__main__":
    main()
