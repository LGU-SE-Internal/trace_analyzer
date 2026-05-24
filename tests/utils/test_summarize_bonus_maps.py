import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.p2a import summarize_bonus_maps as summary_tool  # noqa: E402

MISSING = object()


def _patched_callable(index: int = 0) -> dict:
    return {
        "file_path": f"pkg/mod_{index}.py",
        "qualified_name": f"target_{index}",
        "start_line": 10,
        "end_line": 12,
    }


def _node(
    file_path: str,
    hop_distance: int,
    normalized_distance: float,
    observed_in_trace: bool | object = MISSING,
) -> dict:
    payload = {
        "file_path": file_path,
        "start_line": 10,
        "end_line": 12,
        "hop_distance": hop_distance,
        "normalized_distance": normalized_distance,
    }
    if observed_in_trace is not MISSING:
        payload["observed_in_trace"] = observed_in_trace
    return payload


def _direct_bonus_map(instance_id: str, *, hop_max: int = 1, observed_in_trace: bool | object = MISSING) -> dict:
    return {
        "instance_id": instance_id,
        "case_type": "direct",
        "patched_callables": [_patched_callable()],
        "call_graph_nodes": {
            "tests/test_mod.py::test_target": _node("tests/test_mod.py", hop_max, 1.0, observed_in_trace),
            "pkg/mod_0.py::target_0": _node("pkg/mod_0.py", 0, 0.0, observed_in_trace),
        },
        "hop_max": hop_max,
    }


def _standard_bonus_map(instance_id: str, *, hop_max: int = 3) -> dict:
    return {
        "instance_id": instance_id,
        "case_type": "standard",
        "patched_callables": [_patched_callable()],
        "call_graph_nodes": {
            "tests/test_mod.py::test_target": _node("tests/test_mod.py", hop_max, 1.0),
            "pkg/helper.py::helper": _node("pkg/helper.py", 1, 0.5),
            "pkg/mod_0.py::target_0": _node("pkg/mod_0.py", 0, 0.0),
        },
        "hop_max": hop_max,
    }


def _write_bonus_map(directory: Path, name: str, payload: dict) -> Path:
    path = directory / f"{name}.json"
    path.write_text(json.dumps(payload))
    return path


def test_synthetic_5_instance_corpus_histogram(tmp_path):
    _write_bonus_map(
        tmp_path,
        "01-newly-created",
        {
            "instance_id": "newly-created",
            "case_type": "newly_created",
            "patched_callables": [],
            "newly_created_callables": [_patched_callable()],
            "call_graph_nodes": {},
            "hop_max": 0,
        },
    )
    _write_bonus_map(
        tmp_path,
        "02-no-callable",
        {
            "instance_id": "no-callable",
            "case_type": "no_callable",
            "patched_callables": [],
            "call_graph_nodes": {},
            "hop_max": 0,
        },
    )
    _write_bonus_map(
        tmp_path,
        "03-no-trace",
        {
            "instance_id": "no-trace",
            "case_type": "no_trace",
            "patched_callables": [_patched_callable()],
            "call_graph_nodes": {},
            "hop_max": 0,
        },
    )
    _write_bonus_map(tmp_path, "04-direct", _direct_bonus_map("direct"))
    _write_bonus_map(tmp_path, "05-standard", _standard_bonus_map("standard"))

    summary = summary_tool.summarize(tmp_path)
    histogram = summary["case_type_histogram"]

    assert summary["n_instances_scanned"] == 5
    assert summary["n_instances_parsed"] == 5
    assert histogram["newly_created"] == 1
    assert histogram["no_callable"] == 1
    assert histogram["no_trace"] == 1
    assert histogram["direct"] == 1
    assert histogram["standard"] == 1
    assert histogram["no_gt"] == 0
    assert histogram["all_pass"] == 0
    assert histogram["no_f2p"] == 0
    assert histogram["_total"] == 5
    assert histogram["_error_total"] == 1
    assert histogram["_traceable_total"] == 2


def test_f5_prevalence_all_zero_post_fix(tmp_path):
    for index in range(3):
        _write_bonus_map(
            tmp_path,
            f"direct-{index}",
            _direct_bonus_map(f"direct-{index}", observed_in_trace=True),
        )

    summary = summary_tool.summarize(tmp_path)

    assert summary["f5_prevalence"]["observed_in_trace_field_present"] is True
    assert summary["f5_prevalence"]["instances_with_unobserved_node_in_call_graph"] == 0
    assert summary["f5_prevalence"]["share_of_total"] == 0.0


def test_f5_prevalence_legacy_shape_without_observed_field(tmp_path):
    for index in range(3):
        _write_bonus_map(tmp_path, f"direct-{index}", _direct_bonus_map(f"direct-{index}"))

    summary = summary_tool.summarize(tmp_path)

    assert summary["f5_prevalence"]["observed_in_trace_field_present"] is False
    assert summary["f5_prevalence"]["instances_with_unobserved_node_in_call_graph"] is None
    assert summary["f5_prevalence"]["share_of_total"] is None


def test_f5_prevalence_mixed_transition_data(tmp_path):
    _write_bonus_map(tmp_path, "direct-0", _direct_bonus_map("direct-0", observed_in_trace=False))
    _write_bonus_map(tmp_path, "direct-1", _direct_bonus_map("direct-1", observed_in_trace=True))
    _write_bonus_map(tmp_path, "direct-2", _direct_bonus_map("direct-2", observed_in_trace=True))

    summary = summary_tool.summarize(tmp_path)

    assert summary["f5_prevalence"]["observed_in_trace_field_present"] is True
    assert summary["f5_prevalence"]["instances_with_unobserved_node_in_call_graph"] == 1
    assert summary["f5_prevalence"]["share_of_total"] == pytest.approx(1 / 3)


def test_unobserved_gt_metadata(tmp_path):
    for index in range(2):
        payload = _direct_bonus_map(f"direct-{index}")
        payload["patched_callables"] = [_patched_callable(0), _patched_callable(1)]
        payload["unobserved_patched_callables"] = [_patched_callable(0), _patched_callable(1)]
        _write_bonus_map(tmp_path, f"direct-{index}", payload)
    payload = _direct_bonus_map("direct-2")
    payload["patched_callables"] = [_patched_callable(0), _patched_callable(1)]
    payload["unobserved_patched_callables"] = []
    _write_bonus_map(tmp_path, "direct-2", payload)

    summary = summary_tool.summarize(tmp_path)
    metadata = summary["unobserved_gt_metadata"]

    assert metadata["field_present"] is True
    assert metadata["total_unobserved_callables"] == 4
    assert metadata["mean_per_instance"] == pytest.approx(4 / 3)
    assert metadata["n_instances_with_at_least_one_unobserved"] == 2
    assert metadata["share_of_static_gt_mass"] == pytest.approx(4 / 6)


def test_hop_distribution(tmp_path):
    for index, hop_max in enumerate([1, 3, 5, 5, 7]):
        _write_bonus_map(tmp_path, f"direct-{index}", _direct_bonus_map(f"direct-{index}", hop_max=hop_max))

    summary = summary_tool.summarize(tmp_path)
    hops = summary["hop_distribution"]

    assert hops["n_instances_with_hop_data"] == 5
    assert hops["min"] == 1
    assert hops["max"] == 7
    assert hops["median"] == 5
    assert hops["mean"] == pytest.approx(4.2)


def test_parse_failure_handling(tmp_path):
    for index in range(4):
        _write_bonus_map(tmp_path, f"direct-{index}", _direct_bonus_map(f"direct-{index}"))
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json")

    summary = summary_tool.summarize(tmp_path)

    assert summary["n_instances_scanned"] == 5
    assert summary["n_instances_parsed"] == 4
    assert summary["n_parse_failures"] == 1
    assert summary["parse_failures"][0]["path"] == str(malformed)
    assert "Expecting property name" in summary["parse_failures"][0]["reason"]


def test_json_output_is_valid(tmp_path):
    _write_bonus_map(tmp_path, "direct", _direct_bonus_map("direct"))
    output_path = tmp_path / "summary.json"

    summary = summary_tool.summarize(tmp_path)
    summary_tool.write_json(summary, output_path)

    json.dumps(json.loads(output_path.read_text()))


def test_cli_smoke_test(tmp_path):
    _write_bonus_map(tmp_path, "direct", _direct_bonus_map("direct"))
    cache_dir = PROJECT_ROOT / "cache"
    before = set(cache_dir.glob("bonus_maps_summary_*.json")) if cache_dir.exists() else set()

    try:
        result = subprocess.run(
            [sys.executable, "-m", "utils.p2a.summarize_bonus_maps", str(tmp_path)],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        after = set(cache_dir.glob("bonus_maps_summary_*.json")) if cache_dir.exists() else set()
        for path in after - before:
            path.unlink()

    assert result.returncode == 0
    for case_type in summary_tool.CASE_TYPES_ORDERED:
        assert case_type in result.stdout
