"""Fault trace instrumentation for SWE dry-run mode.

Instruments callables identified from the golden patch so we capture structured
call chains from test (symptom) to buggy callable (root cause) when running
the test harness on unmodified code.

Supports both SWE-Bench Verified (unified diff ``patch`` field) and R2E-Gym
(structured ``parsed_commit_content`` with ``file_diffs``).

Callable detection uses **AST comparison**: parse both the pre-patch (old) and
post-patch (new) source with ``ast``, then identify callables that appear in
both versions but with different source — these are the *in-place* modified
callables worth tracing.  Pure additions (only in new) and pure deletions
(only in old) are excluded.
"""

from __future__ import annotations

import ast
import base64
import json
import logging
import re
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rllm.environments.swe.swe import SWEEnv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch extraction — unified entry point for both datasets
# ---------------------------------------------------------------------------

def extract_non_test_patch(task: dict) -> str:
    """Return a unified-diff string of non-test changes for the given task.

    Works with both dataset formats:
      - **SWE-Bench Verified**: ``task["patch"]`` already contains only non-test
        code changes (test changes live in ``task["test_patch"]``).
      - **R2E-Gym**: no ``patch`` field; we reconstruct a unified diff from
        ``parsed_commit_content.file_diffs``, keeping only files listed in
        ``relevant_files`` (which excludes tests and docs).
    """
    # SWE-Bench path
    if "patch" in task and task["patch"]:
        return task["patch"]

    # R2E-Gym path
    pcc_raw = task.get("parsed_commit_content")
    if not pcc_raw:
        return ""
    return _reconstruct_patch_from_r2e(pcc_raw, task)


def _reconstruct_patch_from_r2e(pcc_raw: str | dict, task: dict) -> str:
    """Reconstruct a unified diff from R2E-Gym structured commit data.

    Filters to non-test ``.py`` files using ``relevant_files``.  Falls back to
    a path-based heuristic when ``relevant_files`` is absent.
    """
    if isinstance(pcc_raw, str):
        try:
            pcc = json.loads(pcc_raw)
        except (json.JSONDecodeError, TypeError):
            return ""
    else:
        pcc = pcc_raw

    file_diffs = pcc.get("file_diffs", [])
    if not file_diffs:
        return ""

    # Build allow-set of non-test .py files
    relevant = task.get("relevant_files")
    if relevant is not None:
        # relevant_files is typically a list/ndarray of non-test .py paths
        allow_set: set[str] | None = set(relevant)
    else:
        allow_set = None

    parts: list[str] = []

    for fd in file_diffs:
        header = fd.get("header", {})
        file_path = header.get("file", {}).get("path", "")
        if not file_path or not file_path.endswith(".py"):
            continue

        # Filter: use allow-set if available, otherwise path heuristic
        if allow_set is not None:
            if file_path not in allow_set:
                continue
        else:
            if _is_test_file(file_path):
                continue

        minus = fd.get("minus_file", {}).get("path", f"a/{file_path}")
        plus = fd.get("plus_file", {}).get("path", f"b/{file_path}")
        hunks = fd.get("hunks", [])
        if not hunks:
            continue

        diff_lines: list[str] = []
        diff_lines.append(f"diff --git {minus} {plus}")
        diff_lines.append(f"--- {minus}")
        diff_lines.append(f"+++ {plus}")

        for hunk in hunks:
            desc = hunk.get("descriptor", {})
            old_r = desc.get("old_range", {})
            new_r = desc.get("new_range", {})
            old_start = old_r.get("start", 0)
            old_len = old_r.get("length", 0)
            new_start = new_r.get("start", 0)
            new_len = new_r.get("length", 0)
            section = desc.get("section", "")
            header_line = f"@@ -{old_start},{old_len} +{new_start},{new_len} @@"
            if section:
                header_line += f" {section}"
            diff_lines.append(header_line)

            for line_info in hunk.get("line_group", {}).get("all_lines", []):
                content = line_info.get("content", "")
                line_type = line_info.get("type", "context")
                if line_type == "context":
                    diff_lines.append(f" {content}")
                elif line_type == "deleted":
                    diff_lines.append(f"-{content}")
                elif line_type == "added":
                    diff_lines.append(f"+{content}")

        parts.append("\n".join(diff_lines))

    return "\n".join(parts)


def _is_test_file(path: str) -> bool:
    """Heuristic: does this path look like a test file?"""
    parts = path.replace("\\", "/").split("/")
    for p in parts:
        if p in ("tests", "test", "testing"):
            return True
        if p.startswith("test_") or p.endswith("_test.py"):
            return True
    return False


# ---------------------------------------------------------------------------
# Task normalisation — flatten SWE-bench verl wrapper
# ---------------------------------------------------------------------------

def normalize_task(task: dict) -> dict:
    """Ensure all dataset fields are top-level keys.

    SWE-Bench Verified data (loaded via ``Dataset.load_data``) wraps the real
    fields inside a JSON string called ``extra_info``.  R2E-Gym data is
    already flat.  This function detects and unpacks the wrapper so that
    downstream code can use ``task["docker_image"]``, ``task["patch"]``, etc.
    regardless of the source dataset.
    """
    extra = task.get("extra_info")
    if extra is None:
        return task

    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except (json.JSONDecodeError, TypeError):
            return task
    if isinstance(extra, dict):
        merged = {**extra, **task}
        merged["extra_info"] = extra
        return merged
    return task


# ---------------------------------------------------------------------------
# 1. AST-based callable extraction
# ---------------------------------------------------------------------------

@dataclass
class CallableInfo:
    """Metadata for a single callable extracted from AST."""

    name: str
    qualified_name: str
    file_path: str
    start_line: int  # ``def`` line (1-based)
    end_line: int  # last line of the callable body
    source: str  # verbatim source text for equality comparison

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }


def extract_callables_from_ast(
    source: str,
    file_path: str,
) -> dict[str, CallableInfo]:
    """Parse *source* with :mod:`ast` and return every callable definition.

    Returns ``{qualified_name: CallableInfo}``.  Nested functions are keyed
    by their enclosing class (e.g. ``MyClass.my_method``) but **not** by
    enclosing functions to keep the key space manageable.

    Possible errors:

    * **SyntaxError** — the source uses syntax unsupported by the running
      Python version, or is simply broken.  Returns ``{}`` in that case
      (same behaviour as the old ``find_modified_callables``).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        logger.warning("SyntaxError while parsing %s — skipping", file_path)
        return {}

    lines = source.splitlines()
    callables: dict[str, CallableInfo] = {}

    def _visit(node: ast.AST, class_name: str | None = None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                _visit(child, class_name=child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = (
                    f"{class_name}.{child.name}" if class_name else child.name
                )
                start = child.lineno
                end = child.end_lineno or child.lineno
                snippet = "\n".join(lines[start - 1 : end])
                callables[qualified] = CallableInfo(
                    name=child.name,
                    qualified_name=qualified,
                    file_path=file_path,
                    start_line=start,
                    end_line=end,
                    source=snippet,
                )
                _visit(child, class_name=class_name)

    _visit(tree)
    return callables


# ---------------------------------------------------------------------------
# 2. AST-diff: find in-place modified callables
# ---------------------------------------------------------------------------

def find_modified_callables_from_sources(
    old_source: str,
    new_source: str,
    file_path: str,
) -> list[dict]:
    """Compare *old* and *new* ASTs to find callables modified **in-place**.

    A callable is "modified in-place" when it appears in **both** versions
    (matched by qualified name) but its source text differs.  This naturally
    excludes:

    * **Pure additions** — callable only in new → not instrumentable on the
      old (pre-fix) code.
    * **Pure deletions** — callable only in old → removed by the fix, not a
      target for tracing.

    Returns a list of dicts (``name``, ``file_path``, ``start_line``,
    ``end_line``, ``qualified_name``) referring to the **old** version —
    because that is the version present in the sandbox when we instrument.
    """
    old_callables = extract_callables_from_ast(old_source, file_path)
    new_callables = extract_callables_from_ast(new_source, file_path)

    modified: list[dict] = []
    for qname, old_info in old_callables.items():
        new_info = new_callables.get(qname)
        if new_info is not None and old_info.source != new_info.source:
            modified.append(old_info.to_dict())
    return modified


# ---------------------------------------------------------------------------
# 3. generate_tracer_module
# ---------------------------------------------------------------------------

def generate_tracer_module(repo_path: str) -> str:
    """Generate _swe_fault_tracer.py source to be deployed in sandbox site-packages.

    The tracer captures call stacks when instrumented callables are invoked,
    filters to repo-internal frames, deduplicates, and emits structured markers
    to stderr.
    """
    return textwrap.dedent(f"""\
        import sys
        import threading
        import traceback

        _lock = threading.Lock()
        _seen = set()
        _REPO_PATH = "{repo_path}"

        def trace(callable_name, file_path, def_lineno):
            frames = traceback.extract_stack()[:-1]
            # Filter to repo-internal frames
            repo_frames = [
                f for f in frames
                if f.filename.startswith(_REPO_PATH)
            ]
            if not repo_frames:
                return
            # Build dedup key from frame signatures
            key = (callable_name, file_path, def_lineno,
                   tuple((f.filename, f.lineno, f.name) for f in repo_frames))
            with _lock:
                if key in _seen:
                    return
                _seen.add(key)
            # Emit structured trace to stderr
            lines = []
            lines.append(f"<<<FAULT_TRACE_BEGIN:{{callable_name}}:{{file_path}}:{{def_lineno}}>>>")
            for f in repo_frames:
                lines.append(f"  {{f.filename}}:{{f.lineno}}:{{f.name}}:{{f.line}}")
            lines.append("<<<FAULT_TRACE_END>>>")
            sys.stderr.write("\\n".join(lines) + "\\n")
    """)


# ---------------------------------------------------------------------------
# 4. instrument_source
# ---------------------------------------------------------------------------

def instrument_source(source: str, callables: list[dict]) -> str:
    """Insert trace calls into source after each callable's def line.

    Sorts by line descending to preserve line numbers during insertion.
    """
    if not callables:
        return source

    lines = source.splitlines(keepends=True)

    # Sort by start_line descending so insertions don't shift earlier lines
    sorted_callables = sorted(callables, key=lambda c: c["start_line"], reverse=True)

    for c in sorted_callables:
        func_line_idx = c["start_line"] - 1  # 0-based index

        if func_line_idx >= len(lines):
            continue

        # Find insertion point: after def line, or after docstring
        insert_idx = func_line_idx + 1

        # Check for multi-line def (lines ending with \ or no colon yet)
        while insert_idx < len(lines):
            stripped = lines[insert_idx - 1].rstrip()
            if stripped.endswith(":"):
                break
            if stripped.endswith("\\"):
                insert_idx += 1
                continue
            break

        # Check if next non-empty line is a docstring
        check_idx = insert_idx
        while check_idx < len(lines) and lines[check_idx].strip() == "":
            check_idx += 1

        if check_idx < len(lines):
            stripped_line = lines[check_idx].strip()
            if stripped_line.startswith(('"""', "'''", 'r"""', "r'''")):
                # Find end of docstring
                quote = '"""' if '"""' in stripped_line else "'''"
                if stripped_line.count(quote) >= 2:
                    # Single-line docstring
                    insert_idx = check_idx + 1
                else:
                    # Multi-line docstring: find closing quote
                    doc_idx = check_idx + 1
                    while doc_idx < len(lines):
                        if quote in lines[doc_idx]:
                            insert_idx = doc_idx + 1
                            break
                        doc_idx += 1
                    else:
                        insert_idx = check_idx + 1

        # Determine indentation from the function body
        body_indent = ""
        for scan_idx in range(insert_idx, min(insert_idx + 5, len(lines))):
            candidate = lines[scan_idx]
            if candidate.strip():
                body_indent = candidate[: len(candidate) - len(candidate.lstrip())]
                break
        if not body_indent:
            # Fallback: use def line indent + 4 spaces
            def_line = lines[func_line_idx]
            def_indent = def_line[: len(def_line) - len(def_line.lstrip())]
            body_indent = def_indent + "    "

        # Check if already instrumented
        if insert_idx < len(lines) and "_swe_fault_tracer" in lines[insert_idx]:
            continue

        trace_line = (
            f'{body_indent}import _swe_fault_tracer as _ft; '
            f'_ft.trace("{c["qualified_name"]}", "{c["file_path"]}", {c["start_line"]})\n'
        )

        lines.insert(insert_idx, trace_line)

    return "".join(lines)


# ---------------------------------------------------------------------------
# 5. instrument_sandbox
# ---------------------------------------------------------------------------

def _get_patched_py_files(patch_text: str) -> set[str]:
    """Return the set of ``.py`` files touched by *patch_text*.

    Includes files from both ``--- a/`` and ``+++ b/`` headers so that pure
    additions (old is ``/dev/null``) and pure deletions (new is ``/dev/null``)
    are captured.  The caller decides which to instrument.
    """
    files: set[str] = set()
    for m in re.finditer(r"^--- a/(.+\.py)\s*$", patch_text, re.MULTILINE):
        p = m.group(1)
        if p != "/dev/null":
            files.add(p)
    for m in re.finditer(r"^\+\+\+ b/(.+\.py)\s*$", patch_text, re.MULTILINE):
        p = m.group(1)
        if p != "/dev/null":
            files.add(p)
    return files


def _get_new_contents_from_task(
    entry: dict,
    files: set[str],
) -> dict[str, str]:
    """Try to obtain post-patch file contents from the task dict.

    Works with both dataset formats:

    * **R2E-Gym**: ``parsed_commit_content`` → ``file_diffs`` → ``new_file_content``
    * **SWE-Bench Verified**: ``parsed_commit`` → ``file_diffs`` → ``new_file_content``
    """
    # Try both field names
    pcc_raw = entry.get("parsed_commit_content") or entry.get("parsed_commit")
    if not pcc_raw:
        return {}

    if isinstance(pcc_raw, str):
        try:
            pcc = json.loads(pcc_raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    else:
        pcc = pcc_raw

    result: dict[str, str] = {}
    for fd in pcc.get("file_diffs", []):
        header = fd.get("header", {})
        path = header.get("file", {}).get("path", "")
        if path in files:
            content = fd.get("new_file_content")
            if content:
                result[path] = content
    return result


def _get_new_contents_via_sandbox(
    env: "SWEEnv",
    patch_text: str,
    files: set[str],
) -> dict[str, str]:
    """Apply the golden patch in the sandbox, read new sources, then revert.

    Fallback for when post-patch content is not available in the task dict
    (e.g. older SWE-Bench snapshots without ``parsed_commit``).
    """
    patch_b64 = base64.b64encode(patch_text.encode()).decode()
    env._run(f"printf '%s' '{patch_b64}' | base64 -d > /tmp/_golden_patch.diff")

    _, err = env._run(f"cd {env.repo_path} && git apply /tmp/_golden_patch.diff")
    if "Error" in str(err):
        logger.warning("Failed to apply golden patch in sandbox: %s", err)
        env._run(f"cd {env.repo_path} && git checkout -- .")
        return {}

    result: dict[str, str] = {}
    for file_path in files:
        full_path = f"{env.repo_path}/{file_path}"
        content, read_err = env._run(f"cat {full_path}")
        if "Error" not in str(read_err):
            result[file_path] = content

    env._run(f"cd {env.repo_path} && git checkout -- .")
    return result


def instrument_sandbox(env: "SWEEnv", patch_text: str) -> list[dict]:
    """Orchestrate fault tracing instrumentation in the sandbox.

    Called after ``env.reset()``.  Deploys the tracer module and instruments
    all callables that are **modified in-place** by the golden patch — i.e.
    callables whose qualified name appears in both the old and new ASTs but
    whose source text differs.

    Returns list of modified-callable dicts (``name``, ``file_path``,
    ``start_line``, ``end_line``, ``qualified_name``).
    """
    if not patch_text:
        logger.info("Empty patch — nothing to instrument")
        return []

    # 1. Determine which non-test .py files the patch touches
    patched_files = _get_patched_py_files(patch_text)
    non_test_files = {f for f in patched_files if not _is_test_file(f)}
    if not non_test_files:
        logger.info("No non-test .py files in patch")
        return []

    # 2. Obtain post-patch (new) file contents for AST comparison
    new_contents = _get_new_contents_from_task(env.entry, non_test_files)
    remaining = non_test_files - set(new_contents)
    if remaining:
        sandbox_new = _get_new_contents_via_sandbox(env, patch_text, remaining)
        new_contents.update(sandbox_new)

    # 3. Find site-packages path & deploy tracer module
    site_output, site_err = env._run(
        "python -c \"import site; print(site.getsitepackages()[0])\""
    )
    site_packages = site_output.strip().splitlines()[0].strip()
    if not site_packages or "Error" in str(site_err):
        logger.warning("Failed to find site-packages: %s %s", site_output, site_err)
        return []

    tracer_source = generate_tracer_module(env.repo_path)
    tracer_b64 = base64.b64encode(tracer_source.encode()).decode()
    tracer_dest = f"{site_packages}/_swe_fault_tracer.py"
    env._run(f"printf '%s' '{tracer_b64}' | base64 -d > {tracer_dest}")

    # 4. For each file: read old source, compare ASTs, instrument
    all_callables: list[dict] = []

    for file_path in non_test_files:
        full_path = f"{env.repo_path}/{file_path}"

        old_source, err_code = env._run(f"cat {full_path}")
        if "Error" in str(err_code):
            logger.warning("Failed to read %s: %s", full_path, err_code)
            continue

        new_source = new_contents.get(file_path, "")
        if not new_source:
            logger.info(
                "No new source for %s — likely a pure deletion, skipping",
                file_path,
            )
            continue

        callables = find_modified_callables_from_sources(
            old_source, new_source, file_path
        )
        if not callables:
            continue

        all_callables.extend(callables)

        # Instrument the old (pre-fix) source
        instrumented = instrument_source(old_source, callables)
        if instrumented == old_source:
            continue

        # Write back via base64
        instr_b64 = base64.b64encode(instrumented.encode()).decode()
        chunk_size = 65536
        if len(instr_b64) <= chunk_size:
            env._run(
                f"printf '%s' '{instr_b64}' | base64 -d > {full_path}"
            )
        else:
            env._run(f": > {full_path}")
            for i in range(0, len(instr_b64), chunk_size):
                chunk = instr_b64[i : i + chunk_size]
                env._run(f"printf '%s' '{chunk}' | base64 -d >> {full_path}")

    logger.info(
        "Instrumented %d callables across %d files",
        len(all_callables),
        len(non_test_files),
    )
    return all_callables


# ---------------------------------------------------------------------------
# 6. parse_fault_traces
# ---------------------------------------------------------------------------

def parse_fault_traces(
    raw_output: str,
    modified_callables: list[dict],
    repo_path: str,
) -> list[list[dict]]:
    """Parse <<<FAULT_TRACE_BEGIN/END>>> blocks from raw test output.

    Each trace is a list of frame dicts with keys: file_path, line_no,
    func_name, line_content, is_patched.

    Only keeps traces that contain at least one patched frame.
    """
    if not raw_output or not modified_callables:
        return []

    # Build a set of (file_path, name) for frame-level matching
    patched_names: set[tuple[str, str]] = set()
    for c in modified_callables:
        patched_names.add((c["file_path"], c["name"]))

    traces: list[list[dict]] = []

    # Find all FAULT_TRACE blocks
    pattern = re.compile(
        r"<<<FAULT_TRACE_BEGIN:([^:]+):([^:]+):(\d+)>>>\n"
        r"(.*?)"
        r"<<<FAULT_TRACE_END>>>",
        re.DOTALL,
    )

    for match in pattern.finditer(raw_output):
        callable_name = match.group(1)
        # groups 2, 3 (file, lineno) available for future use
        frames_text = match.group(4).strip()

        if not frames_text:
            continue

        frames: list[dict] = []
        for frame_line in frames_text.splitlines():
            frame_line = frame_line.strip()
            if not frame_line:
                continue

            # Parse: file_path:line_no:func_name:line_content
            parts = frame_line.split(":", 3)
            if len(parts) < 4:
                continue

            frame_file = parts[0]
            try:
                frame_lineno = int(parts[1])
            except ValueError:
                continue
            frame_func = parts[2]
            frame_content = parts[3]

            # Make file_path relative to repo for matching
            rel_path = frame_file
            if frame_file.startswith(repo_path + "/"):
                rel_path = frame_file[len(repo_path) + 1 :]

            # Check if this frame is in a patched callable
            is_patched = (rel_path, frame_func) in patched_names

            frames.append({
                "file_path": rel_path,
                "line_no": frame_lineno,
                "func_name": frame_func,
                "line_content": frame_content,
                "is_patched": is_patched,
            })

        # Also mark the triggered callable itself
        if frames:
            has_patched = any(f["is_patched"] for f in frames)
            # The callable that was triggered is always patched
            if not has_patched:
                for f in frames:
                    if f["func_name"] == callable_name:
                        f["is_patched"] = True
                        has_patched = True

            if has_patched:
                traces.append(frames)

    return traces


# ---------------------------------------------------------------------------
# 7. aggregate_traces
# ---------------------------------------------------------------------------

def aggregate_traces(traces: list[list[dict]]) -> list[list[dict]]:
    """Deduplicate and keep only maximal traces.

    - Remove exact duplicates
    - Remove subchains: if trace A is a subsequence of trace B, drop A
    """
    if not traces:
        return []

    def _trace_key(trace: list[dict]) -> tuple:
        return tuple(
            (f["file_path"], f["line_no"], f["func_name"]) for f in trace
        )

    # Deduplicate exact matches
    seen_keys: set[tuple] = set()
    unique_traces: list[list[dict]] = []
    for trace in traces:
        key = _trace_key(trace)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_traces.append(trace)

    if len(unique_traces) <= 1:
        return unique_traces

    # Remove subchains: trace A is a subsequence of trace B if all frames
    # of A appear in B in order
    def _is_subsequence(shorter: list[dict], longer: list[dict]) -> bool:
        if len(shorter) >= len(longer):
            return False
        s_key = _trace_key(shorter)
        l_key = _trace_key(longer)
        it = iter(l_key)
        return all(frame in it for frame in s_key)

    # Sort by length descending so we check shorter against longer
    unique_traces.sort(key=len, reverse=True)
    maximal: list[list[dict]] = []

    for trace in unique_traces:
        is_subchain = False
        for existing in maximal:
            if _is_subsequence(trace, existing):
                is_subchain = True
                break
        if not is_subchain:
            maximal.append(trace)

    return maximal
