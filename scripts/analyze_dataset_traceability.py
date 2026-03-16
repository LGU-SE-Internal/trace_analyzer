#!/usr/bin/env python3
"""Analyze and classify SWE dataset instances by traceability.

Two modes:

1. **Static mode** (parquet only): Analyzes AST diffs to determine which
   instances have in-place modified callables (potential traceability).

2. **Bonus map mode** (requires precomputed bonus maps): Reads the actual
   bonus map JSONs produced by ``precompute_bonus_maps.py`` and classifies
   each instance into a fine-grained taxonomy:

   1. Logic bugs (traceable)
      1.1 Direct  — test → patched callable, no intermediate nodes
      1.2 Standard — test → intermediate(s) → patched callable
   2. Crash/untraceable bugs
      2.1 Has patched callables but 0 traces captured
      2.2 Has patched callables, marked traceable, but no test entries (static-only fallback)
   3. No callables — AST diff found no modified callables
   4. Data inconsistency — patched callables exist but couldn't be instrumented

Usage::

    # Static analysis only (fast, no sandbox needed)
    python scripts/analyze_dataset_traceability.py data/swe/R2E_Gym_Subset.parquet

    # With precomputed bonus maps for fine-grained classification
    python scripts/analyze_dataset_traceability.py data/swe/R2E_Gym_Subset.parquet \\
        --bonus_maps_dir data/swe/bonus_maps

    # Analyze a directory of bonus map JSONs directly (no parquet needed)
    python scripts/analyze_dataset_traceability.py --bonus_maps_dir data/swe/bonus_maps
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
    _is_test_file,
    extract_callables_from_ast,
    make_instance_id,
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


# ── bonus map classification ──────────────────────────────────────────────


def classify_bonus_map(bm: dict) -> dict:
    """Classify a single bonus map JSON into the taxonomy.

    Uses ``case_type`` from the JSON when available (written by
    ``precompute_bonus_maps.py``), otherwise re-derives it.

    case_type values (flat strings matching precompute_bonus_maps.py):
        direct        — F2P test calls patched callable directly
        standard      — F2P test reaches patched callable through intermediates
        crash         — tests crashed before reaching patched callable
        newly_created — all patched callables are pure additions
        no_callables  — patch modifies no callable
        no_f2p_trace  — instrumentation succeeded but no F2P trace captured

    Returns a dict with instance stats and category/case_type fields.
    """
    instance_id = bm.get("instance_id", "unknown")
    patched = bm.get("patched_callables", [])
    nodes = bm.get("call_graph_nodes", {})
    hop_max = bm.get("hop_max", 0)
    traceable = bm.get("traceable", False)

    n_patched = len(patched)
    n_nodes = len(nodes)

    # Count node types
    n_test_entries = 0
    n_intermediate = 0
    n_patched_nodes = 0
    for key, node in nodes.items():
        nd = node.get("normalized_distance", 0)
        if _is_test_file(node.get("file_path", "")):
            n_test_entries += 1
        elif nd == 0.0:
            n_patched_nodes += 1
        else:
            n_intermediate += 1

    result = {
        "instance_id": instance_id,
        "n_patched": n_patched,
        "n_nodes": n_nodes,
        "n_test_entries": n_test_entries,
        "n_intermediate": n_intermediate,
        "hop_max": hop_max,
    }

    # Prefer pre-computed case_type from the bonus map JSON
    case_type = bm.get("case_type")
    if not case_type:
        # Re-derive using the same logic as precompute_bonus_maps.py
        if not patched:
            if bm.get("newly_created_callables"):
                case_type = "newly_created"
            else:
                case_type = "no_callables"
        elif not traceable:
            case_type = "no_callables"
        elif bm.get("crash"):
            case_type = "crash"
        elif n_test_entries == 0:
            case_type = "no_f2p_trace"
        elif n_intermediate > 0:
            case_type = "standard"
        else:
            case_type = "direct"

    result["case_type"] = case_type
    result["category"] = "traceable" if case_type in ("direct", "standard") else "untraceable"

    if bm.get("newly_created_callables"):
        result["n_newly_created"] = len(bm["newly_created_callables"])

    return result


# ── static AST analysis (original functionality) ─────────────────────────


def analyze_instance_static(task: dict) -> dict:
    """Return static analysis dict for a single task instance."""
    task = normalize_task(task)
    file_diffs = _parse_file_diffs(task)
    diffs = _non_test_py_diffs(file_diffs, task)

    traceable: list[str] = []
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

        callable_changes = (
            sum(1 for n in old_names & new_names if old_callables[n].source != new_callables[n].source)
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


def categorize_static(r: dict) -> str:
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


# ── bonus map report ─────────────────────────────────────────────────────


def load_bonus_maps(bonus_dir: str) -> dict[str, dict]:
    """Load all bonus map JSONs from a directory, keyed by instance_id."""
    maps = {}
    for p in Path(bonus_dir).glob("*.json"):
        try:
            bm = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        iid = bm.get("instance_id", p.stem)
        maps[iid] = bm
    return maps


def print_bonus_map_report(bonus_dir: str, maps: dict[str, dict]) -> None:
    total = len(maps)
    if total == 0:
        print(f"No bonus maps found in {bonus_dir}")
        return

    classifications = [classify_bonus_map(bm) for bm in maps.values()]

    # Counts by case_type (flat string matching precompute_bonus_maps.py)
    case_counts = Counter(c["case_type"] for c in classifications)

    n_traceable = case_counts.get("direct", 0) + case_counts.get("standard", 0)
    n_untraceable = total - n_traceable

    print(f"\n{'='*50}")
    print(f"Bonus Map Analysis: {total} instances from {bonus_dir}")
    print(f"{'='*50}")

    # Overview bar
    bar_w = 40
    filled = round(n_traceable / total * bar_w) if total else 0
    bar = "#" * filled + "." * (bar_w - filled)
    print(f"\nTraceable  [{bar}]  {n_traceable}/{total} ({n_traceable/total*100:.1f}%)")

    # Summary table (same format as precompute_bonus_maps.py)
    print(f"\ntraceable/          {n_traceable:5d}  ({100*n_traceable/total:.1f}%)")
    print(f"  direct             {case_counts['direct']:5d}")
    print(f"  standard           {case_counts['standard']:5d}")
    print(f"\nuntraceable/        {n_untraceable:5d}  ({100*n_untraceable/total:.1f}%)")
    print(f"  crash              {case_counts['crash']:5d}")
    print(f"  newly_created      {case_counts['newly_created']:5d}")
    print(f"  no_callables       {case_counts['no_callables']:5d}")
    print(f"  no_f2p_trace       {case_counts['no_f2p_trace']:5d}")

    # Any case_types not in the standard set
    standard_types = {"direct", "standard", "crash", "newly_created", "no_callables", "no_f2p_trace"}
    for ct, cnt in case_counts.most_common():
        if ct not in standard_types:
            print(f"  {ct:<18s} {cnt:5d}")

    # Traceable stats
    traceable_cls = [c for c in classifications if c["category"] == "traceable"]
    if traceable_cls:
        hops = [c["hop_max"] for c in traceable_cls]
        intermediates = [c["n_intermediate"] for c in traceable_cls]
        test_entries = [c["n_test_entries"] for c in traceable_cls]

        print(f"\nTraceable instance statistics:")
        print(f"  hop_max:        mean={sum(hops)/len(hops):.1f}  "
              f"median={sorted(hops)[len(hops)//2]}  max={max(hops)}")
        print(f"  intermediate:   mean={sum(intermediates)/len(intermediates):.1f}  "
              f"median={sorted(intermediates)[len(intermediates)//2]}  max={max(intermediates)}")
        print(f"  test_entries:   mean={sum(test_entries)/len(test_entries):.1f}  "
              f"median={sorted(test_entries)[len(test_entries)//2]}  max={max(test_entries)}")

        # Hop distribution
        hop_dist = Counter(hops)
        print(f"\n  hop_max distribution (traceable):")
        for h in sorted(hop_dist):
            cnt = hop_dist[h]
            print(f"    hop_max={h}: {cnt:>5} instances  ({cnt/len(traceable_cls)*100:.1f}%)")

    # Examples per case_type
    print(f"\nExamples:")
    by_ct: dict[str, list[dict]] = {}
    for c in classifications:
        by_ct.setdefault(c["case_type"], []).append(c)

    ct_order = ["standard", "direct", "crash", "newly_created", "no_callables", "no_f2p_trace"]
    for ct in ct_order:
        items = by_ct.get(ct, [])
        if not items:
            continue
        print(f"  [{ct}]")
        for c in items[:3]:
            extras = []
            if c["n_intermediate"] > 0:
                extras.append(f"intermediate={c['n_intermediate']}")
            if c["n_test_entries"] > 0:
                extras.append(f"tests={c['n_test_entries']}")
            extras.append(f"hop_max={c['hop_max']}")
            print(f"    {c['instance_id']}  ({', '.join(extras)})")
    print()


# ── static report (original) ─────────────────────────────────────────────


def analyze_parquet(path: str) -> list[dict]:
    df = pd.read_parquet(path)
    results = []
    for idx, row in df.iterrows():
        extra_raw = row.get("extra_info", "{}")
        task = json.loads(extra_raw) if isinstance(extra_raw, str) else dict(extra_raw)
        r = analyze_instance_static(task)
        task = normalize_task(task)
        r["index"] = idx
        r["repo"] = task.get("repo_name", task.get("repo", "?"))
        r["instance_id"] = make_instance_id(task)
        r["category"] = categorize_static(r)
        results.append(r)

        done = idx + 1
        if done % 500 == 0:
            print(f"  [{Path(path).name}] {done}/{len(df)}...", file=sys.stderr)

    return results


def print_static_report(path: str, results: list[dict]) -> None:
    total = len(results)
    cat_counts = Counter(r["category"] for r in results)
    traceable = [r for r in results if r["category"] == "traceable"]

    print("=" * 72)
    print(f"  {Path(path).name}  ({total} instances) — Static AST Analysis")
    print("=" * 72)
    print()

    # Overview
    n_t = len(traceable)
    n_nt = total - n_t
    bar_w = 40
    filled = round(n_t / total * bar_w) if total else 0
    bar = "#" * filled + "." * (bar_w - filled)
    print(f"  Traceable  [{bar}]  {n_t}/{total} ({n_t/total*100:.1f}%)")
    print()

    # Non-traceable breakdown
    if n_nt:
        print("  Non-traceable breakdown:")
        for cat, cnt in cat_counts.most_common():
            if cat == "traceable":
                continue
            print(f"    {cat:<44s} {cnt:>5}  ({cnt/total*100:.1f}%)")
        print()

    # Callable count distribution
    t_counts = [len(r["traceable"]) for r in traceable]
    if t_counts:
        dist = Counter(t_counts)
        print("  Traceable callable count per instance:")
        for n in sorted(dist):
            print(f"    {n:>2} callable(s): {dist[n]:>5} instances")
        mean = sum(t_counts) / len(t_counts)
        print(f"    mean={mean:.2f}  max={max(t_counts)}")
        print()

    # Co-occurrence
    if traceable:
        also_a = sum(1 for r in traceable if r["added"])
        also_d = sum(1 for r in traceable if r["deleted"])
        also_nc = sum(1 for r in traceable if r["has_non_callable"])
        print("  Among traceable instances, also have:")
        print(f"    + newly added callables:    {also_a:>5} ({also_a/len(traceable)*100:.1f}%)")
        print(f"    + fully deleted callables:  {also_d:>5} ({also_d/len(traceable)*100:.1f}%)")
        print(f"    + non-callable changes:     {also_nc:>5} ({also_nc/len(traceable)*100:.1f}%)")
        print()

    # Examples
    print("  Examples:")
    for cat in ["traceable"] + [c for c, _ in cat_counts.most_common() if c != "traceable"]:
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


# ── main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze traceability of SWE dataset instances."
    )
    parser.add_argument(
        "parquet_files",
        nargs="*",
        help="Path(s) to parquet dataset files (for static analysis)",
    )
    parser.add_argument(
        "--bonus_maps_dir",
        help="Directory containing precomputed bonus map JSONs",
    )
    args = parser.parse_args()

    if not args.parquet_files and not args.bonus_maps_dir:
        parser.error("Provide parquet file(s) and/or --bonus_maps_dir")

    # Static analysis from parquet
    for path in args.parquet_files or []:
        results = analyze_parquet(path)
        print_static_report(path, results)

    # Bonus map analysis
    if args.bonus_maps_dir:
        maps = load_bonus_maps(args.bonus_maps_dir)
        print_bonus_map_report(args.bonus_maps_dir, maps)


if __name__ == "__main__":
    main()
