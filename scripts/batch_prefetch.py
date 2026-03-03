#!/usr/bin/env python3
"""Batch prefetch WarmPool images for SWE RL training.

Reads both R2E-Gym and SWE-Bench Verified parquet datasets, derives pool
names (matching SWEEnv._derive_pool_ref convention), maps docker images to
the mirror registry, and creates WarmPools via the ARL Gateway API to
pre-cache container images on K8s nodes.

Uses a sliding-window concurrency model: up to `--concurrency` pools are
warmed simultaneously. As soon as one finishes (ready → scale down), the
next starts — no batch barriers, no fast pools waiting for slow ones.

Usage:
    # Dry-run (preview what would be created)
    python scripts/batch_prefetch.py --dry-run

    # Create all pools (uses ARL_GATEWAY_URL env var or --gateway)
    python scripts/batch_prefetch.py

    # Specific dataset only
    python scripts/batch_prefetch.py --dataset r2egym
    python scripts/batch_prefetch.py --dataset swebench

    # Custom gateway and concurrency
    python scripts/batch_prefetch.py --gateway http://118.145.210.10:8080 --concurrency 20

    # Keep pools running after image pull (don't scale down)
    python scripts/batch_prefetch.py --no-scale-down-after

    # Limit number of pools (for testing)
    python scripts/batch_prefetch.py --limit 5

    # Delete all pools instead of creating
    python scripts/batch_prefetch.py --delete
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress httpx's per-request INFO logs (e.g. "HTTP Request: POST ... 500")
# so they don't drown out real errors. Genuine failures are logged by our code.
logging.getLogger("httpx").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIRROR_REGISTRY = "pair-diag-cn-guangzhou.cr.volces.com"
MIRROR_NAMESPACE = "code"
MAX_K8S_NAME_LEN = 63

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "swe"

PARQUET_FILES = {
    "r2egym": DATA_DIR / "R2E_Gym_Subset.parquet",
    "swebench": DATA_DIR / "SWE_Bench_Verified.parquet",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sanitize_pool_name(repo_name: str, commit_hash: str) -> str:
    """Create a valid K8s DNS label from repo_name + commit_hash.

    Identical to rllm.environments.swe.swe._derive_pool_ref so that the
    pool names created here match what SWEEnv looks up at training time.
    """
    safe_repo = re.sub(r"[^a-z0-9]", "-", repo_name.lower()).strip("-")
    hash_prefix = commit_hash[:8].lower()
    name = f"{safe_repo}-{hash_prefix}"
    return name[:MAX_K8S_NAME_LEN].rstrip("-")


def mirror_image(docker_image: str) -> str:
    """Convert a docker image to the mirror registry format.

    Rewrites the namespace (first path segment) to MIRROR_NAMESPACE ('code'),
    consistent with scripts/mirror_images.py rewrite_namespace().

    Examples:
        namanjain12/numpy_final:abc123
          -> pair-diag-cn-guangzhou.cr.volces.com/code/numpy_final:abc123
        slimshetty/swebench-verified:sweb.eval.x86_64.tag
          -> pair-diag-cn-guangzhou.cr.volces.com/code/swebench-verified:sweb.eval.x86_64.tag
    """
    parts = docker_image.split("/", 1)
    image_path = parts[1] if len(parts) == 2 else docker_image
    return f"{MIRROR_REGISTRY}/{MIRROR_NAMESPACE}/{image_path}"


def load_pool_specs(parquet_path: Path) -> list[dict[str, str]]:
    """Load a parquet dataset and extract unique (pool_name, image) pairs.

    Handles both top-level columns and fields nested inside extra_info JSON.
    Deduplicates by pool_name (same repo+commit -> same pool).
    """
    try:
        import pandas as pd
    except ImportError:
        sys.exit("pandas is required: pip install pandas pyarrow")

    df = pd.read_parquet(parquet_path)
    pools: dict[str, dict[str, str]] = {}

    for _, row in df.iterrows():
        # Parse extra_info if present (may contain docker_image, repo_name, etc.)
        extra_info: dict = {}
        if "extra_info" in row.index and row["extra_info"] is not None:
            raw = row["extra_info"]
            extra_info = json.loads(raw) if isinstance(raw, str) else (raw if isinstance(raw, dict) else {})

        # Resolve fields: prefer top-level columns, fallback to extra_info
        repo_name = _get_field(row, extra_info, "repo_name", "repo")
        commit_hash = _get_field(row, extra_info, "commit_hash", "base_commit")
        docker_image = _get_field(row, extra_info, "docker_image", "image_name")

        if not (repo_name and commit_hash and docker_image):
            continue

        pool_name = sanitize_pool_name(repo_name, commit_hash)
        if pool_name not in pools:
            pools[pool_name] = {
                "name": pool_name,
                "image": mirror_image(docker_image),
                "repo": repo_name,
                "hash": commit_hash,
                "original_image": docker_image,
            }

    return list(pools.values())


def _get_field(row, extra_info: dict, *keys: str) -> str:
    """Try multiple keys from row (pandas Series) then extra_info dict."""
    for key in keys:
        # Top-level column
        if key in row.index:
            val = row[key]
            if val is not None and str(val).strip():
                return str(val).strip()
        # extra_info
        val = extra_info.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Batch prefetch WarmPool images for SWE RL training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gateway",
        type=str,
        default=os.environ.get("ARL_GATEWAY_URL", "http://localhost:8080"),
        help="ARL Gateway URL (default: $ARL_GATEWAY_URL or http://localhost:8080)",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="default",
        help="Kubernetes namespace (default: default)",
    )
    parser.add_argument(
        "--dataset",
        choices=["r2egym", "swebench", "all"],
        default="all",
        help="Which dataset(s) to prefetch (default: all)",
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=1,
        help="Replicas per pool — 1 is enough for image prefetch (default: 1)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max pools warming simultaneously — sliding window, no batch barriers (default: 10)",
    )
    parser.add_argument(
        "--pool-timeout",
        type=float,
        default=600.0,
        help="Max seconds to wait for a single pool to become ready (default: 600)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
        help="Seconds between readiness polls (default: 10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of pools to create, 0 = all (default: 0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without creating pools",
    )
    parser.add_argument(
        "--skip-wait",
        action="store_true",
        help="Don't wait for pools to become ready",
    )
    parser.add_argument(
        "--scale-down-after",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale pools to 0 after image pull (default: True). Use --no-scale-down-after to keep running.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete all matching pools instead of creating them",
    )
    args = parser.parse_args()

    # --- Load pool specs from datasets ---
    datasets = list(PARQUET_FILES.keys()) if args.dataset == "all" else [args.dataset]
    all_pools: dict[str, dict[str, str]] = {}

    for ds_name in datasets:
        path = PARQUET_FILES[ds_name]
        if not path.exists():
            logger.warning(f"Parquet file not found, skipping: {path}")
            continue
        specs = load_pool_specs(path)
        logger.info(f"[{ds_name}] {len(specs)} unique pools from {path.name}")
        for spec in specs:
            all_pools.setdefault(spec["name"], spec)

    pools = sorted(all_pools.values(), key=lambda p: p["name"])

    if args.limit > 0:
        pools = pools[: args.limit]
        logger.info(f"Limited to {len(pools)} pools")

    logger.info(f"Total unique pools to prefetch: {len(pools)}")

    if not pools:
        logger.warning("No pools found. Exiting.")
        return

    # --- Dry-run ---
    if args.dry_run:
        print(f"\n=== DRY RUN: {len(pools)} pools would be created ===\n")
        for p in pools[:30]:
            print(f"  pool:  {p['name']}")
            print(f"  image: {p['image']}")
            print(f"  repo:  {p['repo']}  commit: {p['hash'][:12]}")
            print()
        if len(pools) > 30:
            print(f"  ... and {len(pools) - 30} more\n")
        return

    # --- Import ARL (only needed for non-dry-run) ---
    try:
        from arl.gateway_client import GatewayError, PoolNotReadyError
        from arl.warmpool import WarmPoolManager
    except ImportError:
        sys.exit(
            "arl library is required. Install it or activate the venv:\n"
            "  source .venv/bin/activate"
        )

    # --- Delete mode ---
    if args.delete:
        manager = WarmPoolManager(
            namespace=args.namespace,
            gateway_url=args.gateway,
        )
        deleted, skipped, failed = 0, 0, 0
        for pool in pools:
            try:
                manager.delete_warmpool(pool["name"])
                deleted += 1
                logger.info(f"Deleted: {pool['name']}")
            except GatewayError as e:
                if e.status_code == 404:
                    skipped += 1
                else:
                    failed += 1
                    logger.error(f"Failed to delete {pool['name']}: {e}")
        logger.info(f"Delete done: {deleted} deleted, {skipped} not found, {failed} failed")
        manager.close()
        return

    # --- Create mode (sliding window: create → wait → scale_down per worker) ---
    #
    # Each worker handles ONE pool's full lifecycle. This ensures at most
    # `concurrency` pods are pulling images at any time — preventing
    # registry overload from thousands of simultaneous pulls.
    logger.info(
        f"Prefetching {len(pools)} WarmPools with concurrency={args.concurrency}"
    )
    logger.info(f"Gateway: {args.gateway}, Namespace: {args.namespace}")
    if args.scale_down_after:
        logger.info("Scale-down-after enabled: pools scaled to 0 replicas after image pull")

    _lock = threading.Lock()
    counters = {"created": 0, "skipped": 0, "failed": 0, "done": 0}
    total = len(pools)

    def _prefetch_one(pool: dict[str, str]) -> tuple[str, bool, str]:
        """Full lifecycle for one pool: create/scale-up → wait → scale_down."""
        name = pool["name"]
        mgr = WarmPoolManager(
            namespace=args.namespace,
            gateway_url=args.gateway,
            timeout=args.pool_timeout,
        )
        try:
            # 1. Create or scale up existing pool
            already_exists = False
            try:
                mgr.create_warmpool(
                    name=name,
                    image=pool["image"],
                    replicas=args.replicas,
                )
            except GatewayError as e:
                if e.status_code == 409 or "already exists" in str(e):
                    already_exists = True
                    # Pool CRD exists — check if it already has ready replicas
                    try:
                        info = mgr.get_warmpool(name)
                        if info.ready_replicas >= args.replicas:
                            # Already warm, just scale down if needed
                            if args.scale_down_after:
                                mgr.scale_warmpool(name, 0)
                            with _lock:
                                counters["skipped"] += 1
                                counters["done"] += 1
                                done = counters["done"]
                            logger.info(f"  [{done}/{total}] SKIP (ready): {name}")
                            return name, True, "already_ready"
                    except Exception:
                        pass
                    # Not ready — scale up to trigger image pull
                    mgr.scale_warmpool(name, args.replicas)
                else:
                    raise

            # 2. Wait for ready
            if not args.skip_wait:
                mgr.wait_for_ready(
                    name,
                    timeout=args.pool_timeout,
                    poll_interval=args.poll_interval,
                )

            # 3. Scale down (release pod, image stays cached on node)
            if args.scale_down_after and not args.skip_wait:
                mgr.scale_warmpool(name, 0)

            with _lock:
                counters["created"] += 1
                counters["done"] += 1
                done = counters["done"]
            suffix = " → scaled to 0" if args.scale_down_after else ""
            logger.info(f"  [{done}/{total}] OK: {name}{suffix}")
            return name, True, "ok"

        except Exception as e:
            with _lock:
                counters["failed"] += 1
                counters["done"] += 1
                done = counters["done"]
            logger.error(f"  [{done}/{total}] FAIL: {name}: {e}")
            return name, False, str(e)
        finally:
            mgr.close()

    failed_names: list[str] = []
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(_prefetch_one, p): p for p in pools}
            for future in as_completed(futures):
                name, ok, msg = future.result()
                if not ok:
                    failed_names.append(name)
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")

    # --- Summary ---
    logger.info("=" * 50)
    logger.info(
        f"Done: created={counters['created']}, skipped={counters['skipped']}, "
        f"failed={counters['failed']}"
    )

    if failed_names:
        failed_path = REPO_ROOT / "scripts" / ".prefetch_failed.log"
        failed_path.write_text("\n".join(failed_names) + "\n")
        logger.error(f"Failed pools written to {failed_path}")

    if counters["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
