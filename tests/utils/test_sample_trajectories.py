import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import subprocess
from collections import Counter

from utils.p2a import sample_trajectories as st

SCRIPT = Path(__file__).resolve().parents[2] / "utils" / "p2a" / "sample_trajectories.py"
MISSING = object()


def _patched_callable(file_path: str = "src/mod.py", qualified_name: str = "bug_fn") -> dict:
    return {
        "file_path": file_path,
        "qualified_name": qualified_name,
        "start_line": 10,
        "end_line": 12,
    }


def _node(file_path: str, qualified_name: str, hop_distance: int, normalized_distance: float, observed=MISSING) -> dict:
    node = {
        "file_path": file_path,
        "qualified_name": qualified_name,
        "start_line": hop_distance + 1,
        "end_line": hop_distance + 2,
        "hop_distance": hop_distance,
        "normalized_distance": normalized_distance,
    }
    if observed is not MISSING:
        node["observed_in_trace"] = observed
    return node


def _bonus_map(instance_id: str, case_type: str, call_graph_nodes: dict | None = None, include_unobserved: bool = True) -> dict:
    bm = {
        "instance_id": instance_id,
        "case_type": case_type,
        "traceable": case_type in {"direct", "standard"},
        "error": case_type in {"all_pass", "no_f2p", "no_gt", "no_trace"},
        "patched_callables": [_patched_callable()],
        "call_graph_nodes": call_graph_nodes or {},
        "hop_max": max((node.get("hop_distance", 0) for node in (call_graph_nodes or {}).values()), default=0),
    }
    if include_unobserved:
        bm["unobserved_patched_callables"] = []
    return bm


def _write_bonus_map(directory: Path, bm: dict) -> Path:
    path = directory / f"{bm['instance_id']}.json"
    path.write_text(json.dumps(bm))
    return path


def _sample_ids(directory: Path, n: int, seed: int | None = None) -> list[str]:
    sampled, warnings, _effective_seed = st.sample_bonus_maps(directory, n=n, seed=seed)
    assert warnings == []
    return [record["instance_id"] for record in sampled]


def test_synthetic_corpus_stratified_sampling(tmp_path):
    for case_type in ("standard", "direct", "no_trace", "no_gt"):
        for index in range(5):
            _write_bonus_map(tmp_path, _bonus_map(f"{case_type}-{index}", case_type))

    sampled, warnings, _effective_seed = st.sample_bonus_maps(tmp_path, n=8, seed=123)

    assert warnings == []
    assert len(sampled) == 8
    counts = Counter(record["case_type"] for record in sampled)
    assert counts["standard"] >= 1
    assert counts["direct"] >= 1
    assert counts["no_trace"] >= 1
    assert counts["no_gt"] >= 1


def test_n_larger_than_corpus_returns_everything(tmp_path):
    for case_type in ("standard", "direct", "no_trace"):
        _write_bonus_map(tmp_path, _bonus_map(case_type, case_type))

    sampled, warnings, _effective_seed = st.sample_bonus_maps(tmp_path, n=100, seed=123)

    assert warnings == []
    assert len(sampled) == 3
    assert {record["instance_id"] for record in sampled} == {"standard", "direct", "no_trace"}


def test_deterministic_seed_returns_same_instance_ids(tmp_path):
    for case_type in ("standard", "direct", "no_trace", "no_gt"):
        for index in range(5):
            _write_bonus_map(tmp_path, _bonus_map(f"{case_type}-{index}", case_type))

    first = _sample_ids(tmp_path, n=8, seed=42)
    second = _sample_ids(tmp_path, n=8, seed=42)

    assert first == second


def test_default_seed_is_deterministic_from_dir_name(tmp_path):
    for index in range(10):
        case_type = "standard" if index % 2 == 0 else "direct"
        _write_bonus_map(tmp_path, _bonus_map(f"{case_type}-{index}", case_type))

    first = _sample_ids(tmp_path, n=5)
    second = _sample_ids(tmp_path, n=5)

    assert first == second


def test_call_graph_sort_by_hop_descending(tmp_path):
    nodes = {
        "src/mod.py::bug_b": _node("src/mod.py", "bug_b", 0, 0.0, True),
        "tests/test_mod.py::test_bug": _node("tests/test_mod.py", "test_bug", 4, 1.0, True),
        "src/helper.py::helper": _node("src/helper.py", "helper", 2, 0.5, True),
        "src/mod.py::bug_a": _node("src/mod.py", "bug_a", 0, 0.0, True),
    }
    bm = _bonus_map("sort-case", "standard", nodes)
    record = {"bonus_map": bm, "instance_id": "sort-case", "case_type": "standard"}

    output = st.render_instance(record)

    hop_lines = [line for line in output.splitlines() if "[hop=" in line]
    hops = [int(line.split("[hop=", 1)[1].split(",", 1)[0]) for line in hop_lines]
    assert hops == [4, 2, 0, 0]


def test_observed_yes_no_unknown_rendering(tmp_path):
    nodes = {
        "tests/test_mod.py::test_bug": _node("tests/test_mod.py", "test_bug", 2, 1.0, True),
        "src/helper.py::helper": _node("src/helper.py", "helper", 1, 0.5, False),
        "src/mod.py::bug_fn": _node("src/mod.py", "bug_fn", 0, 0.0),
    }
    bm = _bonus_map("observed-case", "standard", nodes)
    record = {"bonus_map": bm, "instance_id": "observed-case", "case_type": "standard"}

    output = st.render_instance(record)

    assert "[hop=2, norm=1.00, observed=Y]" in output
    assert "[hop=1, norm=0.50, observed=N]" in output
    assert "[hop=0, norm=0.00, observed=?]" in output


def test_legacy_bonus_map_shape_does_not_crash_and_emits_note(tmp_path):
    nodes = {
        "tests/test_mod.py::test_bug": _node("tests/test_mod.py", "test_bug", 1, 1.0),
        "src/mod.py::bug_fn": _node("src/mod.py", "bug_fn", 0, 0.0),
    }
    bm = _bonus_map("legacy-case", "direct", nodes, include_unobserved=False)
    record = {"bonus_map": bm, "instance_id": "legacy-case", "case_type": "direct"}

    output = st.render_instance(record)

    assert "(legacy shape: observed_in_trace and unobserved_patched_callables fields absent)" in output
    assert "INSTANCE: legacy-case" in output


def test_no_traceable_instances_still_emit_blocks_with_empty_call_graph(tmp_path):
    for index in range(3):
        _write_bonus_map(tmp_path, _bonus_map(f"no-trace-{index}", "no_trace"))

    sampled, warnings, _effective_seed = st.sample_bonus_maps(tmp_path, n=10, seed=123)
    output = st.render_report(sampled)

    assert warnings == []
    assert output.count("INSTANCE:") == 3
    assert "(empty - untraceable)" in output


def test_output_file_write_matches_stdout_and_is_plain_ascii(tmp_path, capsys):
    _write_bonus_map(tmp_path, _bonus_map("direct-case", "direct"))
    output_path = tmp_path / "trajectory_samples.txt"

    exit_code = st.main([str(tmp_path), "--n", "1", "--output", str(output_path), "--seed", "42"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_path.exists()
    assert output_path.read_text() == captured.out
    captured.out.encode("ascii")


def test_cli_smoke_test_via_subprocess(tmp_path):
    _write_bonus_map(tmp_path, _bonus_map("standard-case", "standard"))

    completed = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path), "--n", "2"], capture_output=True, text=True, check=False)

    assert completed.returncode == 0
    assert "INSTANCE:" in completed.stdout
