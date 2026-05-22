# AGENTS.md — codex / sub-agent quick reference for the `trace_analyzer` repo

This file mirrors the parts of `CLAUDE.md` that any non-Claude worker (codex CLI, general-purpose sub-agent) needs to know before touching code. **`CLAUDE.md` is the source of truth**; this file is the codex-facing extract. If the two diverge, update `CLAUDE.md` first, then re-mirror here.

## ⚠️ Always run Python via `uv run`

**Every Python invocation in this repo MUST be prefixed with `uv run`.** This is non-negotiable.

| Wrong | Right |
|---|---|
| `pytest tests/envs/test_trace.py` | `uv run pytest tests/envs/test_trace.py` |
| `python -m rllm.trainer.verl.train_agent_ppo` | `uv run python -m rllm.trainer.verl.train_agent_ppo` |
| `python3 some_script.py` | `uv run python3 some_script.py` |
| `mypy rllm/` | `uv run mypy rllm/` |
| `ruff check .` | `uv run ruff check .` |

Why: this project uses `uv` with a pinned `uv.lock`. Bare `pytest` / `python` will either fail (dependencies not on PATH) OR silently run against a different Python environment than the one this repo's lockfile pins — leading to results that diverge from CI and from human-run tests.

If you see a snippet in `CLAUDE.md`, `README.md`, or a script that uses bare `pytest`/`python` without `uv run`, mentally prepend `uv run` before executing. Treat that as a doc bug and fix it in the same PR if it falls within your file scope.

## Repository facts

- **Branch convention**: development happens on `rllm`, NOT `main`. PRs target `rllm`. See `CLAUDE.md` for the full reason.
- **Python**: >= 3.10 (do not use 3.11-only features like `code.co_qualname` unless you verify the sandbox image).
- **Test directory**: `tests/` with subdirs `tests/envs/`, `tests/agents/`.
- **Linter / formatter**: `uv run ruff check --fix .` / `uv run ruff format .`. Soft gate (not pre-commit).
- **Type checker**: `uv run mypy rllm/`.

## When in doubt

1. Read `CLAUDE.md` (next to this file) — the full project guide.
2. Read your issue body — your file_scope and acceptance criteria are authoritative.
3. Do NOT modify files outside your declared file_scope.
4. If a constraint conflicts with what you need to do, STOP and surface the conflict (via `result.md` if running under codex, via your final-message contract if you're a sub-agent). Do NOT silently expand scope.
