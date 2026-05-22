"""Per-repo dependency fixups for R2E-Gym Docker images.

R2E-Gym Docker images are built at a specific base commit.  Some repos evolve
to require new dependencies that weren't in the image's original venv.  The
official R2E-Gym project handles this via per-repo install scripts under
``src/r2egym/install_utils/``.

This module provides a lightweight equivalent: targeted shell commands that
install missing deps or strip incompatible config.

These fixups are applied in ``SWEEnv._setup_env()`` so that **all** sandbox
consumers (RL training, standalone eval, bonus map precomputation) benefit.

Note on ``uv pip install``: ARL gateway has gRPC stream timeouts (~120s).
To stay within limits:
- **Pin exact versions** to skip dependency resolution.
- **Install one package per command** to avoid batched resolution.
"""

from __future__ import annotations

# Timeout for each install command (seconds).
_INSTALL_TIMEOUT = 120

# Keys: lowercase repo name as it appears in docker_image field.
# Values: list of (command, timeout_override | None) tuples.
#
# IMPORTANT: install ONE package per command with pinned version to avoid
# ARL gateway gRPC stream timeouts during batched resolution.
REPO_FIXUPS: dict[str, list[tuple[str, int | None]]] = {
    "aiohttp": [
        # Different aiohttp commits may be missing different deps.
        # Each install is idempotent (no-op if already present).
        ("uv pip install aiohappyeyeballs==2.6.1", None),
        ("uv pip install aiosignal==1.4.0", None),
        ("uv pip install frozenlist==1.5.0", None),
        ("uv pip install attrs==24.2.0", None),
        ("uv pip install async-timeout==5.0.1", None),
    ],
    "coveragepy": [
        # setup.cfg addopts has -n3 (pytest-xdist) and --no-flaky-report (flaky).
        # These are infrastructure flags only (parallelism, reporting); they do
        # not affect test pass/fail outcomes.  Stripping avoids installing
        # optional pytest plugins.
        ("sed -i 's/-n[0-9]*//g; s/--no-flaky-report//g' /testbed/setup.cfg 2>/dev/null; true", None),
    ],
}


def apply_repo_fixups(env, docker_image: str) -> None:
    """Execute dependency fixups matching the given docker_image.

    Args:
        env: ``SWEEnv`` instance (needs ``_execute_raw`` method).
        docker_image: The Docker image string from the task, e.g.
                ``"namanjain12/aiohttp_final:006fbe03..."``.
    """
    image_lower = docker_image.lower()
    for repo_key, commands in REPO_FIXUPS.items():
        if repo_key in image_lower:
            for cmd, timeout in commands:
                timeout = timeout or _INSTALL_TIMEOUT
                stdout, stderr, exit_code = env._execute_raw(cmd, timeout=timeout)
                if exit_code != 0:
                    detail = (stderr or stdout or "")[:200]
                    print(f"[install_utils] WARN: '{cmd[:80]}' failed (exit={exit_code}): {detail}")
            break
