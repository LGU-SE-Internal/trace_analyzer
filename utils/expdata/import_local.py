#!/usr/bin/env python3
"""
Bulk import local experiment data into the expdata server.

Scans a directory of experiment folders and uploads:
  - Eval experiments: *_n1.jsonl (metadata) + chat_completions/eval.jsonl (trajectories)
  - Collection experiments: *.pos.jsonl + *.neg.jsonl (DPO trajectories)

Usage:
    # Import all experiments under data/expriment/
    python utils/expdata/import_local.py --dir data/expriment --server http://localhost:8502

    # Import a single experiment folder
    python utils/expdata/import_local.py --dir data/expriment/Qwen3-8B --server http://localhost:8502

    # Register first (one-time), then import
    python -m utils.expdata.client register myname ~/.ssh/id_ed25519.pub --server http://localhost:8502
    python utils/expdata/import_local.py --dir data/expriment --server http://localhost:8502

    # Dry run — show what would be imported without uploading
    python utils/expdata/import_local.py --dir data/expriment --server http://localhost:8502 --dry_run
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from utils.expdata.client import ExperimentUploader

logger = logging.getLogger("expdata_import")


def detect_experiment_type(folder: Path) -> str | None:
    """Detect whether a folder is an eval, collection, or rollout experiment."""
    # Eval: has *_n1.jsonl
    if list(folder.glob("*_n1.jsonl")):
        return "eval"
    # Collection: has *.pos.jsonl or *.neg.jsonl
    if list(folder.glob("*.pos.jsonl")) or list(folder.glob("*.neg.jsonl")):
        return "collection"
    # Rollout: has numbered .jsonl files (1.jsonl, 2.jsonl, ...) with message lists
    numbered = [f for f in folder.glob("*.jsonl") if f.stem.isdigit()]
    if numbered:
        return "rollout"
    return None


def parse_jsonl(path: Path) -> list[dict]:
    """Parse JSONL file, skipping malformed lines."""
    results = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line, strict=False))
            except json.JSONDecodeError:
                logger.warning(f"  Skipping malformed line {i} in {path.name}")
    return results


def import_eval(uploader: ExperimentUploader, folder: Path, dry_run: bool = False) -> int | None:
    """Import an eval experiment folder."""
    meta_files = list(folder.glob("*_n1.jsonl"))
    if not meta_files:
        logger.error(f"  No *_n1.jsonl found in {folder}")
        return None
    meta_file = meta_files[0]
    traj_file = folder / "chat_completions" / "eval.jsonl"

    logger.info(f"  Metadata: {meta_file.name}")
    metadata = parse_jsonl(meta_file)
    if not metadata:
        logger.error(f"  Empty metadata file: {meta_file}")
        return None

    trajectories = []
    has_trajectories = traj_file.exists()
    if has_trajectories:
        logger.info("  Trajectories: chat_completions/eval.jsonl")
        trajectories = parse_jsonl(traj_file)
        logger.info(f"  Parsed {len(trajectories)} trajectories")

    # Extract model name from folder or metadata
    model_name = folder.name
    n_results = len(metadata)
    solved = sum(1 for r in metadata if r.get("reward", 0) > 0)

    logger.info(f"  Results: {n_results} ({solved} solved, {n_results - solved} failed)")

    if dry_run:
        logger.info(f"  [DRY RUN] Would upload {n_results} results + {len(trajectories)} trajectories")
        return None

    # Create experiment
    exp_id = uploader.create_experiment(
        name=model_name,
        type="eval",
        model=model_name,
        dataset=metadata[0].get("data_source", ""),
        n_samples=metadata[0].get("n_samples", 1),
    )
    logger.info(f"  Created experiment #{exp_id}")

    # Upload eval results
    uploader.upload_eval_results(exp_id, metadata)
    logger.info(f"  Uploaded {n_results} eval results")

    # Upload trajectories (convert to expected format)
    if trajectories:
        traj_records = []
        for i, (meta, msgs) in enumerate(zip(metadata, trajectories, strict=False)):
            # msgs is a list of message dicts (the chat completion for this trajectory)
            messages = msgs if isinstance(msgs, list) else []
            traj_records.append(
                {
                    "instance_id": meta.get("instance_id", ""),
                    "sample_idx": meta.get("sample_idx", 0),
                    "reward": meta.get("reward"),
                    "termination_reason": meta.get("termination_reason", ""),
                    "n_steps": len([m for m in messages if m.get("role") == "assistant"]) if messages else 0,
                    "messages": messages,
                    "role": "primary",
                }
            )
        uploader.upload_trajectories(exp_id, traj_records)
        logger.info(f"  Uploaded {len(traj_records)} trajectories")

    # Mark completed
    uploader.mark_completed(
        exp_id,
        {
            "total": n_results,
            "solved": solved,
            "solve_rate": solved / max(n_results, 1),
        },
    )
    logger.info(f"  Experiment #{exp_id} marked completed")
    return exp_id


def import_collection(uploader: ExperimentUploader, folder: Path, dry_run: bool = False) -> int | None:
    """Import a collection (DPO/rejection) experiment folder."""
    pos_files = sorted(folder.glob("*.pos.jsonl"))
    neg_files = sorted(folder.glob("*.neg.jsonl"))

    pos_entries = []
    for f in pos_files:
        logger.info(f"  Positive: {f.name}")
        pos_entries.extend(parse_jsonl(f))

    neg_entries = []
    for f in neg_files:
        logger.info(f"  Negative: {f.name}")
        neg_entries.extend(parse_jsonl(f))

    total = len(pos_entries) + len(neg_entries)
    logger.info(f"  Total: {len(pos_entries)} pos + {len(neg_entries)} neg = {total} trajectories")

    if not total:
        logger.warning("  No trajectories found, skipping")
        return None

    if dry_run:
        logger.info(f"  [DRY RUN] Would upload {total} trajectories")
        return None

    model_name = folder.name
    exp_id = uploader.create_experiment(
        name=model_name,
        type="collection",
        model=model_name,
        mode="dpo" if neg_entries else "rejection",
    )
    logger.info(f"  Created experiment #{exp_id}")

    # Upload as trajectories with role tags
    def make_records(entries, role):
        records = []
        for entry in entries:
            msgs = entry.get("messages", [])
            records.append(
                {
                    "instance_id": entry.get("instance_id", ""),
                    "sample_idx": 0,
                    "reward": 1.0 if role == "chosen" else 0.0,
                    "termination_reason": entry.get("termination_reason", ""),
                    "n_steps": len([m for m in msgs if m.get("role") == "assistant"]),
                    "messages": msgs,
                    "role": role,
                }
            )
        return records

    if pos_entries:
        records = make_records(pos_entries, "chosen")
        uploader.upload_trajectories(exp_id, records)
        logger.info(f"  Uploaded {len(records)} positive trajectories")

    if neg_entries:
        records = make_records(neg_entries, "rejected")
        uploader.upload_trajectories(exp_id, records)
        logger.info(f"  Uploaded {len(records)} negative trajectories")

    uploader.mark_completed(
        exp_id,
        {
            "pos": len(pos_entries),
            "neg": len(neg_entries),
            "instances": len({e.get("instance_id") for e in pos_entries + neg_entries}),
        },
    )
    logger.info(f"  Experiment #{exp_id} marked completed")
    return exp_id


def import_rollout(uploader: "ExperimentUploader", folder: Path, dry_run: bool = False) -> int | None:
    """Import a rollout experiment folder (numbered .jsonl files, each line is a messages list)."""
    jsonl_files = sorted(
        [f for f in folder.glob("*.jsonl") if f.stem.isdigit()],
        key=lambda f: int(f.stem),
    )
    if not jsonl_files:
        logger.error(f"  No numbered .jsonl files found in {folder}")
        return None

    all_trajectories = []
    for f in jsonl_files:
        logger.info(f"  Rollout file: {f.name}")
        entries = parse_jsonl(f)
        for i, entry in enumerate(entries):
            # Each entry is either a list of messages or a dict with messages key
            if isinstance(entry, list):
                messages = entry
            elif isinstance(entry, dict) and "messages" in entry:
                messages = entry["messages"]
            else:
                continue
            n_steps = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant")
            all_trajectories.append(
                {
                    "instance_id": f"rollout_{f.stem}_{i}",
                    "sample_idx": 0,
                    "reward": None,
                    "termination_reason": None,
                    "n_steps": n_steps,
                    "messages": messages,
                    "role": "primary",
                }
            )
        logger.info(f"    {len(entries)} trajectories")

    total = len(all_trajectories)
    logger.info(f"  Total: {total} trajectories from {len(jsonl_files)} files")

    if not total:
        logger.warning("  No trajectories found, skipping")
        return None

    if dry_run:
        logger.info(f"  [DRY RUN] Would upload {total} trajectories")
        return None

    model_name = folder.name
    exp_id = uploader.create_experiment(
        name=model_name,
        type="collection",  # reuse collection type for rollout storage
        model=model_name,
        mode="rollout",
    )
    logger.info(f"  Created experiment #{exp_id}")

    # Upload in batches to avoid OOM on large rollout datasets
    batch_size = 100
    uploaded = 0
    for i in range(0, total, batch_size):
        batch = all_trajectories[i : i + batch_size]
        uploader.upload_trajectories(exp_id, batch)
        uploaded += len(batch)
        if uploaded % 500 == 0 or uploaded == total:
            logger.info(f"  Uploaded {uploaded}/{total} trajectories")
    logger.info(f"  Upload complete: {uploaded} trajectories")

    uploader.mark_completed(
        exp_id,
        {
            "total_trajectories": total,
            "files": len(jsonl_files),
        },
    )
    logger.info(f"  Experiment #{exp_id} marked completed")
    return exp_id


def main():
    parser = argparse.ArgumentParser(description="Import local experiment data into expdata server")
    parser.add_argument("--dir", required=True, help="Experiment directory (or parent directory containing multiple experiments)")
    parser.add_argument("--server", default="http://localhost:8502", help="Expdata server URL")
    parser.add_argument("--token_path", default=None, help="Path to token file (default: ~/.config/expdata/token)")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be imported without uploading")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    root = Path(args.dir)
    if not root.exists():
        logger.error(f"Directory not found: {root}")
        sys.exit(1)

    # Check if root itself is an experiment folder
    root_type = detect_experiment_type(root)
    if root_type:
        folders = [(root, root_type)]
    else:
        # Scan subdirectories
        folders = []
        for sub in sorted(root.iterdir()):
            if not sub.is_dir():
                continue
            exp_type = detect_experiment_type(sub)
            if exp_type:
                folders.append((sub, exp_type))

    if not folders:
        logger.error(f"No experiment folders found under {root}")
        sys.exit(1)

    logger.info(f"Found {len(folders)} experiment(s) to import:")
    for folder, exp_type in folders:
        logger.info(f"  {folder.name} ({exp_type})")
    print()

    if not args.dry_run:
        kwargs = {"server_url": args.server}
        if args.token_path:
            kwargs["token_path"] = args.token_path
        uploader = ExperimentUploader(**kwargs)
        if not uploader.token:
            logger.error(
                "No token found. Login first: python -m utils.expdata.client login <username> --password <pw> --server %s",
                args.server,
            )
            sys.exit(1)
    else:
        uploader = None

    imported = 0
    for folder, exp_type in folders:
        logger.info(f"{'=' * 60}")
        logger.info(f"Importing: {folder.name} (type={exp_type})")
        try:
            if exp_type == "eval":
                exp_id = import_eval(uploader, folder, dry_run=args.dry_run)
            elif exp_type == "collection":
                exp_id = import_collection(uploader, folder, dry_run=args.dry_run)
            elif exp_type == "rollout":
                exp_id = import_rollout(uploader, folder, dry_run=args.dry_run)
            if exp_id is not None:
                imported += 1
        except Exception as e:
            logger.error(f"  Failed to import {folder.name}: {e}")
            import traceback

            traceback.print_exc()

    print()
    if args.dry_run:
        logger.info(f"Dry run complete. {len(folders)} experiment(s) would be imported.")
    else:
        logger.info(f"Import complete. {imported}/{len(folders)} experiment(s) imported successfully.")


if __name__ == "__main__":
    main()
