"""
P2A (Process-to-Advantage) core module.

Implements the V2 multiplicative advantage reshaping based on call-graph distance:
  A_token = A_seq * m(d)^sign(A)
  where m(d) = m_max^(1-d), d = normalized hop distance in [0,1].

Components:
  - BonusMapStore: loads precomputed bonus maps (per instance_id)
  - parse_read_actions: extracts file viewing actions from agent responses
  - match_reads_to_callgraph: matches Read actions to call graph nodes
  - compute_p2a_multiplier: V2 multiply/divide formula

Tracking modes:
  - "view_only": only track file_editor view commands
  - "view_and_bash": also track cat/grep/head/tail/sed in execute_bash
"""

import json
import os
import re
from typing import Optional


class BonusMapStore:
    """Loads and caches precomputed bonus maps from disk.

    Each bonus map is a JSON file at {bonus_map_dir}/{instance_id}.json.
    """

    def __init__(self, bonus_map_dir: str):
        self.bonus_map_dir = bonus_map_dir
        self._cache: dict[str, Optional[dict]] = {}

    def get(self, instance_id: str) -> Optional[dict]:
        if instance_id in self._cache:
            return self._cache[instance_id]

        path = os.path.join(self.bonus_map_dir, f"{instance_id}.json")
        if not os.path.exists(path):
            self._cache[instance_id] = None
            return None

        with open(path) as f:
            data = json.load(f)
        self._cache[instance_id] = data
        return data


# ---------------------------------------------------------------------------
# Path normalization
# ---------------------------------------------------------------------------

_SANDBOX_PREFIXES = ["/testbed/", "/workspace/", "/repo/"]


def _normalize_path(path: str) -> str:
    """Strip common sandbox prefixes and leading ./ from file paths."""
    path = path.strip()
    for prefix in _SANDBOX_PREFIXES:
        if path.startswith(prefix):
            return path[len(prefix):]
    if path.startswith("./"):
        return path[2:]
    return path


# ---------------------------------------------------------------------------
# file_editor view parsing
# ---------------------------------------------------------------------------

# JSON-style: <function=file_editor>{"command": "view", "path": "...", ...}</function>
_FILE_EDITOR_JSON_PATTERN = re.compile(
    r'<function=file_editor>\s*(\{[^}]*"command"\s*:\s*"view"[^}]*\})',
    re.DOTALL,
)

# XML-style: <function=file_editor><parameter name="command">view</parameter>...
_FILE_EDITOR_XML_PATTERN = re.compile(
    r'<function=file_editor>.*?<parameter name="command">view</parameter>'
    r'.*?<parameter name="path">([^<]+)</parameter>'
    r'(?:.*?<parameter name="view_range">\[(\d+),\s*(\d+)\]</parameter>)?',
    re.DOTALL,
)

# Old-style with = instead of name=: <parameter=command>view</parameter>
_FILE_EDITOR_XML_EQ_PATTERN = re.compile(
    r'<function=file_editor>.*?<parameter=command>view</parameter>'
    r'.*?<parameter=path>([^<]+)</parameter>'
    r'(?:.*?<parameter=view_range>\[(\d+),\s*(\d+)\]</parameter>)?',
    re.DOTALL,
)


def _parse_file_editor_views(text: str) -> list[dict]:
    """Parse file_editor view commands from agent response text."""
    reads = []

    # JSON-style
    for match in _FILE_EDITOR_JSON_PATTERN.finditer(text):
        try:
            payload = json.loads(match.group(1))
            if payload.get("command") != "view":
                continue
            path = _normalize_path(payload.get("path", ""))
            view_range = payload.get("view_range")
            if view_range and len(view_range) == 2:
                start_line, end_line = int(view_range[0]), int(view_range[1])
            else:
                start_line, end_line = 1, 999999
            reads.append({"file_path": path, "start_line": start_line, "end_line": end_line})
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    # XML-style (name="...")
    for match in _FILE_EDITOR_XML_PATTERN.finditer(text):
        path = _normalize_path(match.group(1))
        if match.group(2) and match.group(3):
            start_line, end_line = int(match.group(2)), int(match.group(3))
        else:
            start_line, end_line = 1, 999999
        reads.append({"file_path": path, "start_line": start_line, "end_line": end_line})

    # XML-style (=...)
    for match in _FILE_EDITOR_XML_EQ_PATTERN.finditer(text):
        path = _normalize_path(match.group(1))
        if match.group(2) and match.group(3):
            start_line, end_line = int(match.group(2)), int(match.group(3))
        else:
            start_line, end_line = 1, 999999
        reads.append({"file_path": path, "start_line": start_line, "end_line": end_line})

    return reads


# ---------------------------------------------------------------------------
# execute_bash file-viewing command parsing
# ---------------------------------------------------------------------------

# Extract the command parameter from execute_bash calls
# Supports: <function=execute_bash><parameter name="command">...</parameter>
#           <function=execute_bash><parameter=command>...</parameter>
#           <function=execute_bash>{"command": "..."}
_BASH_CMD_XML_PATTERN = re.compile(
    r'<function=execute_bash>.*?<parameter\s*(?:name=)?"?command"?>([^<]+)</parameter>',
    re.DOTALL,
)
_BASH_CMD_XML_EQ_PATTERN = re.compile(
    r'<function=execute_bash>.*?<parameter=command>([^<]+)</parameter>',
    re.DOTALL,
)
_BASH_CMD_JSON_PATTERN = re.compile(
    r'<function=execute_bash>\s*(\{[^}]*"command"\s*:[^}]*\})',
    re.DOTALL,
)

# --- Patterns for specific bash commands that view files ---

# cat <path>
# cat -n <path>
# Avoid matching cat with redirection (cat > file) or pipe-only usage
_CAT_PATTERN = re.compile(
    r'\bcat\s+(?:-[nAbeEstTv]+\s+)*'  # optional flags
    r'([^\s|><;`$&]+\.py\b)',  # file path (must end in .py to reduce false positives)
)

# head/tail [-n N] <path>
_HEAD_TAIL_PATTERN = re.compile(
    r'\b(head|tail)\s+'
    r'(?:-(?:n\s*)?(\d+)\s+)?'  # optional -n N or -N
    r'([^\s|><;`$&]+\.py\b)',
)

# sed -n 'START,ENDp' <path>
_SED_N_PATTERN = re.compile(
    r"\bsed\s+-n\s+['\"]?(\d+),(\d+)p['\"]?\s+"
    r'([^\s|><;`$&]+\.py\b)',
)

# grep -n <pattern> <path>  (with -n means showing line numbers → viewing context)
# Also: grep -rn, grep -Hn, etc.
_GREP_N_PATTERN = re.compile(
    r'\bgrep\s+(?:-[A-Za-z]*n[A-Za-z]*\s+)'  # flags must include -n
    r"(?:['\"][^'\"]*['\"]\s+|[^\s]+\s+)"  # pattern arg
    r'([^\s|><;`$&]+\.py\b)',  # file path
)

# grep <pattern> <path>  (without -n, still viewing file content)
_GREP_PATTERN = re.compile(
    r'\bgrep\s+(?:-[A-Za-z]+\s+)*'  # optional flags
    r"(?:['\"][^'\"]*['\"]\s+|[^\s]+\s+)"  # pattern arg
    r'([^\s|><;`$&]+\.py\b)',  # file path
)

# python -c "..." is NOT a file view → skip
# awk/python scripts that read files are too complex → skip

# Default context window for commands that don't specify line ranges
_DEFAULT_CONTEXT_LINES = 50  # head/tail default context


def _parse_bash_read_commands(text: str) -> list[dict]:
    """Parse execute_bash commands that view file contents.

    Extracts cat, head, tail, sed -n, grep from bash commands.
    """
    # First extract bash command strings from the tool calls
    bash_commands = []
    for match in _BASH_CMD_XML_PATTERN.finditer(text):
        bash_commands.append(match.group(1).strip())
    for match in _BASH_CMD_XML_EQ_PATTERN.finditer(text):
        bash_commands.append(match.group(1).strip())
    for match in _BASH_CMD_JSON_PATTERN.finditer(text):
        try:
            payload = json.loads(match.group(1))
            cmd = payload.get("command", "")
            if cmd:
                bash_commands.append(cmd.strip())
        except (json.JSONDecodeError, ValueError):
            continue

    # If no structured bash tool calls found, the text itself might contain
    # inline bash commands (e.g., in sweagent format where the model outputs
    # bash commands directly). We treat the entire text as potential commands.
    if not bash_commands:
        bash_commands = [text]

    reads = []
    seen = set()  # dedup

    for cmd in bash_commands:
        # cat
        for match in _CAT_PATTERN.finditer(cmd):
            path = _normalize_path(match.group(1))
            key = ("cat", path)
            if key not in seen:
                seen.add(key)
                reads.append({"file_path": path, "start_line": 1, "end_line": 999999})

        # head / tail
        for match in _HEAD_TAIL_PATTERN.finditer(cmd):
            subcmd = match.group(1)
            n_lines = int(match.group(2)) if match.group(2) else _DEFAULT_CONTEXT_LINES
            path = _normalize_path(match.group(3))
            key = (subcmd, path, n_lines)
            if key not in seen:
                seen.add(key)
                if subcmd == "head":
                    reads.append({"file_path": path, "start_line": 1, "end_line": n_lines})
                else:
                    # tail: we don't know the file length, use large range
                    reads.append({"file_path": path, "start_line": 1, "end_line": 999999})

        # sed -n 'START,ENDp'
        for match in _SED_N_PATTERN.finditer(cmd):
            start = int(match.group(1))
            end = int(match.group(2))
            path = _normalize_path(match.group(3))
            key = ("sed", path, start, end)
            if key not in seen:
                seen.add(key)
                reads.append({"file_path": path, "start_line": start, "end_line": end})

        # grep -n (line-numbered grep → actively viewing specific content)
        for match in _GREP_N_PATTERN.finditer(cmd):
            path = _normalize_path(match.group(1))
            key = ("grep_n", path)
            if key not in seen:
                seen.add(key)
                reads.append({"file_path": path, "start_line": 1, "end_line": 999999})

        # grep without -n (still accessing file content)
        for match in _GREP_PATTERN.finditer(cmd):
            path = _normalize_path(match.group(1))
            key = ("grep", path)
            if key not in seen:
                seen.add(key)
                reads.append({"file_path": path, "start_line": 1, "end_line": 999999})

    return reads


# ---------------------------------------------------------------------------
# Public API: parse_read_actions (unified entry point)
# ---------------------------------------------------------------------------

TRACKING_MODES = ("view_only", "view_and_bash")


def parse_read_actions(response_text: str, tracking_mode: str = "view_only") -> list[dict]:
    """Parse file viewing actions from agent response text.

    Args:
        response_text: The agent's response text (tool calls + reasoning).
        tracking_mode: One of:
            - "view_only": only file_editor view commands
            - "view_and_bash": also cat/grep/head/tail/sed in execute_bash

    Returns:
        List of dicts with keys: file_path, start_line, end_line.
        Lines are 1-indexed. If no range, start_line=1, end_line=999999.
    """
    reads = _parse_file_editor_views(response_text)

    if tracking_mode == "view_and_bash":
        reads.extend(_parse_bash_read_commands(response_text))

    return reads


# ---------------------------------------------------------------------------
# Call graph matching
# ---------------------------------------------------------------------------

def match_reads_to_callgraph(reads: list[dict], bonus_map: dict) -> float:
    """Match Read actions against call graph nodes.

    For each read action, check if its file_path and line range overlap with any
    call graph node. Among all matches, return the minimum normalized distance
    (i.e., max bonus).

    Args:
        reads: List of read actions from parse_read_actions().
        bonus_map: A bonus map dict with "call_graph_nodes" key.

    Returns:
        Minimum normalized distance in [0, 1] if any match found, -1.0 otherwise.
    """
    if not reads or not bonus_map:
        return -1.0

    nodes = bonus_map.get("call_graph_nodes", {})
    if not nodes:
        return -1.0

    min_distance = float("inf")

    for read in reads:
        read_path = read["file_path"]
        read_start = read["start_line"]
        read_end = read["end_line"]

        for _node_key, node in nodes.items():
            node_path = node["file_path"]
            node_start = node["start_line"]
            node_end = node["end_line"]

            # Check file path match
            if read_path != node_path:
                continue

            # Check line range overlap
            if read_start <= node_end and read_end >= node_start:
                d = node["normalized_distance"]
                min_distance = min(min_distance, d)

    if min_distance == float("inf"):
        return -1.0

    return min_distance


# ---------------------------------------------------------------------------
# V2 multiplier
# ---------------------------------------------------------------------------

def compute_p2a_multiplier(distance: float, m_max: float, advantage_sign: int) -> float:
    """Compute the V2 multiplicative advantage reshape factor.

    Formula:
        m(d) = m_max^(1-d)
        - A > 0 (positive advantage): multiply by m(d)  → amplify
        - A < 0 (negative advantage): multiply by 1/m(d) → shrink
        - A = 0: return 1.0 (no change)
        - off-graph (distance < 0): return 1.0 (no change)

    This preserves the sign of advantage unconditionally.

    Args:
        distance: Normalized distance in [0, 1], or < 0 if off-graph.
        m_max: Maximum multiplier (hyperparameter, typically 2-5).
        advantage_sign: Sign of the advantage (+1, -1, or 0).

    Returns:
        Multiplier to apply to the advantage.
    """
    # Off-graph: no modification
    if distance < 0:
        return 1.0

    # Zero advantage: no modification
    if advantage_sign == 0:
        return 1.0

    # m(d) = m_max^(1-d)
    m_d = m_max ** (1.0 - distance)

    if advantage_sign > 0:
        return m_d  # amplify good actions
    else:
        return 1.0 / m_d  # shrink bad actions
