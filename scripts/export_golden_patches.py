#!/usr/bin/env python3
"""Export golden patch callable data from SWE dataset parquets to JSON.

The output JSON is consumed by trajectory_analyzer.html for golden-patch
step highlighting.

Usage::

    # Export both default parquets
    python scripts/export_golden_patches.py

    # Export specific parquet(s)
    python scripts/export_golden_patches.py data/swe/SWE_Bench_Verified.parquet

    # Custom output path
    python scripts/export_golden_patches.py --out /tmp/golden_patches.json

Output format::

    {
      "astropy__astropy-12907": {
        "files": ["astropy/modeling/separable.py"],
        "callables": [
          {"file": "astropy/modeling/separable.py",
           "name": "_cstack",
           "qname": "_cstack",
           "start": 219, "end": 247}
        ]
      },
      ...
    }
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rllm.environments.swe.trace import (
    find_modified_callables_from_task,
    make_instance_id,
    normalize_task,
)


def _normalize_ps(text: str) -> str:
    """Normalize a problem statement for fingerprint-based matching.

    Strips markdown bold labels (``**Label:**``) and ``[ISSUE]`` prefixes that
    some dataset builders add, then collapses whitespace and truncates to 250
    chars — enough to uniquely identify an instance while remaining stable
    across minor formatting differences.
    """
    if not text:
        return ""
    text = re.sub(r"^\s*\[ISSUE\]\s*", "", text)
    text = re.sub(r"\*\*[A-Za-z ]+:\*\*\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:250]


def process_parquet(path: str, result: dict) -> int:
    df = pd.read_parquet(path)
    added = 0
    for idx, row in df.iterrows():
        raw = row.get("extra_info", "{}")
        extra = json.loads(raw) if isinstance(raw, str) else dict(raw)
        task = normalize_task(extra)
        iid = make_instance_id(task)
        if not iid or iid in result:
            continue
        try:
            callables = find_modified_callables_from_task(task)
        except Exception as e:
            print(f"  Warning [{iid}]: {e}", file=sys.stderr)
            callables = []

        files = sorted({c["file_path"] for c in callables})
        ps_fp = _normalize_ps(task.get("problem_statement", ""))
        result[iid] = {
            "files": files,
            "callables": [
                {
                    "file": c["file_path"],
                    "name": c["name"],
                    "qname": c["qualified_name"],
                    "start": c["start_line"],
                    "end": c["end_line"],
                }
                for c in callables
            ],
            "_fp": ps_fp,
        }
        added += 1
        if (idx + 1) % 100 == 0:
            print(f"  [{Path(path).name}] {idx + 1}/{len(df)} processed…", file=sys.stderr)

    return added


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export golden patch callables from parquet to JSON for the trajectory analyzer."
    )
    parser.add_argument(
        "parquets",
        nargs="*",
        default=[
            "data/swe/SWE_Bench_Verified.parquet",
            "data/swe/R2E_Gym_Subset.parquet",
        ],
        help="Parquet file(s) to export from (default: both SWE-Bench Verified and R2E-Gym)",
    )
    parser.add_argument(
        "--out",
        default="data/swe/golden_patches.json",
        help="Output JSON path (default: data/swe/golden_patches.json)",
    )
    args = parser.parse_args()

    result: dict = {}
    for path in args.parquets:
        p = Path(path)
        if not p.exists():
            print(f"  Skipping {path} (not found)", file=sys.stderr)
            continue
        print(f"Processing {path}…")
        added = process_parquet(path, result)
        print(f"  → Added {added} instances")

    # Build a reverse fingerprint → instance_id lookup, then strip _fp from entries
    ps_lookup: dict[str, str] = {}
    for iid, entry in result.items():
        fp = entry.pop("_fp", "")
        if fp and iid not in ps_lookup.values():
            ps_lookup[fp] = iid

    # Embed the lookup as a special top-level key (prefixed with __ to avoid
    # collision with any real instance_id)
    result["__ps_lookup__"] = ps_lookup

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result))
    print(f"\nWrote {len(result) - 1} instances + fingerprint lookup to {out}")


if __name__ == "__main__":
    main()
