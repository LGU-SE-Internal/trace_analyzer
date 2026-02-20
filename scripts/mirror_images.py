#!/usr/bin/env python3
"""Mirror docker images between registries (two-stage: pull + push).

Reads parquet datasets from data/swe/, extracts all unique docker_image values,
pulls from a source registry, and pushes to a destination registry.
Defaults: hub.byted.org -> pair-diag-cn-guangzhou.cr.volces.com

The process is split into two stages that can run together or independently:
  - pull:  src registry -> local docker
  - push:  local docker -> dst registry (+ cleanup)

Usage:
    # Full mirror (pull + push + cleanup, default)
    python scripts/mirror_images.py

    # Stage 1 only: pull images to local
    python scripts/mirror_images.py --stage pull

    # Stage 2 only: push local images to dst (assumes images already pulled)
    python scripts/mirror_images.py --stage push

    # Custom source and destination registries
    python scripts/mirror_images.py --src docker.io --dst my-registry.example.com

    # Dry-run — only print commands, don't execute
    python scripts/mirror_images.py --dry-run

    # Custom parallelism
    python scripts/mirror_images.py --workers 8

    # Mirror only one dataset
    python scripts/mirror_images.py --dataset swebench
    python scripts/mirror_images.py --dataset r2egym

    # Resume from a previously failed run (skip already-pushed images)
    python scripts/mirror_images.py --skip-existing

    # Test connectivity by mirroring only the first image
    python scripts/mirror_images.py --test-one
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SRC_REGISTRY_DEFAULT = "hub.byted.org"
DST_REGISTRY_DEFAULT = "pair-diag-cn-guangzhou.cr.volces.com"

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "swe"

PARQUET_FILES = {
    "swebench": DATA_DIR / "SWE_Bench_Verified.parquet",
    "r2egym": DATA_DIR / "R2E_Gym_Subset.parquet",
}


def extract_images(parquet_path: Path) -> list[str]:
    """Extract unique docker_image values from a parquet dataset."""
    try:
        import pandas as pd
    except ImportError:
        sys.exit("pandas is required: pip install pandas pyarrow")

    df = pd.read_parquet(parquet_path)
    images = set()
    for raw in df["extra_info"]:
        info = json.loads(raw) if isinstance(raw, str) else raw
        img = info.get("docker_image") or info.get("image_name", "")
        if img:
            images.add(img)
    return sorted(images)


def docker_run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    """Run a docker command with timeout."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def image_exists_remote(full_image: str) -> bool:
    """Check if image already exists in destination registry via manifest inspect."""
    result = docker_run(["docker", "manifest", "inspect", full_image], timeout=30)
    return result.returncode == 0


def image_exists_local(full_image: str) -> bool:
    """Check if image exists in local docker."""
    result = docker_run(["docker", "image", "inspect", full_image], timeout=15)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Stage functions: each returns (image, success, msg)
# ---------------------------------------------------------------------------

def do_pull(image: str, src_registry: str, dry_run: bool = False, skip_existing: bool = False) -> tuple[str, bool, str]:
    """Pull image from src_registry to local docker."""
    src = f"{src_registry}/{image}"

    if dry_run:
        return image, True, f"[dry-run] docker pull {src}"

    if skip_existing and image_exists_local(src):
        return image, True, "skipped (already exists locally)"

    r = docker_run(["docker", "pull", src], timeout=1200)
    if r.returncode != 0:
        return image, False, f"pull failed: {r.stderr.strip()}"
    return image, True, "pulled"


def do_push(
    image: str, src_registry: str, dst_registry: str,
    dry_run: bool = False, skip_existing: bool = False,
) -> tuple[str, bool, str]:
    """Tag and push a local image to dst_registry, then clean up local copies."""
    src = f"{src_registry}/{image}"
    dst = f"{dst_registry}/{image}"

    if dry_run:
        return image, True, f"[dry-run] docker tag {src} {dst} && docker push {dst} && docker rmi {src} {dst}"

    # Skip if already in destination
    if skip_existing:
        try:
            if image_exists_remote(dst):
                return image, True, "skipped (already exists in destination)"
        except Exception:
            pass

    # Verify local image exists
    if not image_exists_local(src):
        return image, False, f"local image not found: {src} (run with --stage pull first)"

    # Tag
    r = docker_run(["docker", "tag", src, dst])
    if r.returncode != 0:
        return image, False, f"tag failed: {r.stderr.strip()}"

    # Push
    r = docker_run(["docker", "push", dst], timeout=1200)
    if r.returncode != 0:
        # Clean up the dst tag on failure, keep src so user can retry
        docker_run(["docker", "rmi", dst], timeout=60)
        return image, False, f"push failed: {r.stderr.strip()}"

    # Clean up both local copies
    docker_run(["docker", "rmi", src], timeout=60)
    docker_run(["docker", "rmi", dst], timeout=60)
    return image, True, "pushed"


def do_all(
    image: str, src_registry: str, dst_registry: str,
    dry_run: bool = False, skip_existing: bool = False,
) -> tuple[str, bool, str]:
    """Pull + push + cleanup in one shot."""
    src = f"{src_registry}/{image}"
    dst = f"{dst_registry}/{image}"

    if dry_run:
        return image, True, (
            f"[dry-run] docker pull {src} && docker tag {src} {dst}"
            f" && docker push {dst} && docker rmi {src} {dst}"
        )

    # Skip if already in destination
    if skip_existing:
        try:
            if image_exists_remote(dst):
                return image, True, "skipped (already exists in destination)"
        except Exception:
            pass

    # Pull
    r = docker_run(["docker", "pull", src], timeout=1200)
    if r.returncode != 0:
        return image, False, f"pull failed: {r.stderr.strip()}"

    # Tag
    r = docker_run(["docker", "tag", src, dst])
    if r.returncode != 0:
        docker_run(["docker", "rmi", src], timeout=60)
        return image, False, f"tag failed: {r.stderr.strip()}"

    # Push
    r = docker_run(["docker", "push", dst], timeout=1200)
    if r.returncode != 0:
        docker_run(["docker", "rmi", src], timeout=60)
        docker_run(["docker", "rmi", dst], timeout=60)
        return image, False, f"push failed: {r.stderr.strip()}"

    # Clean up
    docker_run(["docker", "rmi", src], timeout=60)
    docker_run(["docker", "rmi", dst], timeout=60)
    return image, True, "ok"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mirror SWE docker images between registries.")
    parser.add_argument(
        "--stage",
        choices=["pull", "push", "all"],
        default="all",
        help="Which stage to run: 'pull' (src->local), 'push' (local->dst+cleanup), 'all' (default, full pipeline).",
    )
    parser.add_argument("--src", type=str, default=SRC_REGISTRY_DEFAULT, help=f"Source registry (default: {SRC_REGISTRY_DEFAULT}).")
    parser.add_argument("--dst", type=str, default=DST_REGISTRY_DEFAULT, help=f"Destination registry (default: {DST_REGISTRY_DEFAULT}).")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4).")
    parser.add_argument(
        "--dataset",
        choices=["swebench", "r2egym", "all"],
        default="all",
        help="Which dataset(s) to mirror (default: all).",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip images already in destination registry (push/all stage).")
    parser.add_argument("--output", type=str, default=None, help="Write image list to a file (one per line).")
    parser.add_argument("--test-one", action="store_true", help="Only mirror the first image (for testing connectivity).")
    args = parser.parse_args()

    # Collect images
    all_images: set[str] = set()
    datasets = list(PARQUET_FILES.keys()) if args.dataset == "all" else [args.dataset]
    for ds_name in datasets:
        path = PARQUET_FILES[ds_name]
        if not path.exists():
            logger.warning(f"Parquet file not found, skipping: {path}")
            continue
        imgs = extract_images(path)
        logger.info(f"[{ds_name}] Found {len(imgs)} unique images from {path.name}")
        all_images.update(imgs)

    images = sorted(all_images)
    if args.test_one:
        images = images[:1]
    logger.info(f"Total unique images to mirror: {len(images)}")

    if not images:
        logger.warning("No images found. Exiting.")
        return

    # Optionally dump image list
    if args.output:
        Path(args.output).write_text("\n".join(images) + "\n")
        logger.info(f"Image list written to {args.output}")

    stage_desc = {
        "pull": f"Pulling: {args.src} -> local",
        "push": f"Pushing: local -> {args.dst}",
        "all":  f"Mirroring: {args.src} -> {args.dst}",
    }
    logger.info(stage_desc[args.stage])

    # Build worker function per stage
    def _make_task(img: str):
        if args.stage == "pull":
            return do_pull(img, args.src, args.dry_run, args.skip_existing)
        elif args.stage == "push":
            return do_push(img, args.src, args.dst, args.dry_run, args.skip_existing)
        else:
            return do_all(img, args.src, args.dst, args.dry_run, args.skip_existing)

    # Execute
    succeeded, failed = 0, 0
    failed_images: list[str] = []

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_make_task, img): img for img in images}
        total = len(futures)
        pbar = tqdm(total=total, desc=args.stage.capitalize(), unit="img") if tqdm else None
        for future in as_completed(futures):
            img, ok, msg = future.result()
            if ok:
                succeeded += 1
            else:
                failed += 1
                failed_images.append(img)
            if pbar:
                pbar.set_postfix_str(f"ok={succeeded} fail={failed}")
                pbar.update(1)
                if not ok:
                    tqdm.write(f"[FAIL] {img} — {msg}")
            else:
                status = "OK" if ok else "FAIL"
                logger.info(f"[{succeeded + failed}/{total}] [{status}] {img} — {msg}")
        if pbar:
            pbar.close()

    # Summary
    logger.info(f"Done. {succeeded} succeeded, {failed} failed out of {len(images)} total.")
    if failed_images:
        failed_list_path = REPO_ROOT / "scripts" / ".mirror_failed.log"
        failed_list_path.write_text("\n".join(failed_images) + "\n")
        logger.error(f"Failed images written to {failed_list_path}")
        logger.error("Re-run with the same command to retry (use --skip-existing to avoid re-pushing successes).")
        sys.exit(1)


if __name__ == "__main__":
    main()
