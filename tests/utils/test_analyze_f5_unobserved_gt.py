import json
from pathlib import Path

from utils.p2a import analyze_f5_unobserved_gt as f5


def _patched_callable(file_path: str = "pkg/mod.py", qualified_name: str = "target") -> dict:
    return {
        "file_path": file_path,
        "qualified_name": qualified_name,
        "start_line": 10,
        "end_line": 12,
    }


def _node(file_path: str, hop_distance: int, normalized_distance: float) -> dict:
    return {
        "file_path": file_path,
        "start_line": 10,
        "end_line": 12,
        "hop_distance": hop_distance,
        "normalized_distance": normalized_distance,
    }


def _write_bonus_map(directory: Path, name: str, payload: dict) -> Path:
    path = directory / f"{name}.json"
    path.write_text(json.dumps(payload))
    return path


def test_direct_bonus_map_counts_static_gt_without_error_mass(tmp_path, capsys):
    _write_bonus_map(
        tmp_path,
        "direct-1",
        {
            "instance_id": "direct-1",
            "case_type": "direct",
            "traceable": True,
            "error": False,
            "patched_callables": [_patched_callable()],
            "call_graph_nodes": {
                "tests/test_mod.py::test_target": _node("tests/test_mod.py", 1, 1.0),
                "pkg/mod.py::target": _node("pkg/mod.py", 0, 0.0),
            },
            "hop_max": 1,
        },
    )

    summary = f5.analyze_bonus_maps_dir(tmp_path)

    assert summary["total_instances_scanned"] == 1
    assert summary["static_gt_mass"] == 1
    assert summary["error_static_gt_mass"] == 0
    assert summary["traceable_static_gt_mass"] == 1
    assert summary["instances"][0]["claimed_observed_gt_count"] == 1

    f5.print_f5_report(summary)
    stdout = capsys.readouterr().out
    assert "Total instances scanned: 1" in stdout
    assert "total_static_gt_mass" in stdout
    assert "traceable mass" in stdout
    assert "Limitations:" in stdout


def test_no_trace_bonus_map_contributes_all_static_gt_to_error_mass(tmp_path):
    _write_bonus_map(
        tmp_path,
        "no-trace-1",
        {
            "instance_id": "no-trace-1",
            "case_type": "no_trace",
            "traceable": False,
            "error": True,
            "patched_callables": [_patched_callable()],
            "call_graph_nodes": {},
            "hop_max": 0,
        },
    )

    summary = f5.analyze_bonus_maps_dir(tmp_path)

    assert summary["total_instances_scanned"] == 1
    assert summary["static_gt_mass"] == 1
    assert summary["error_instance_count"] == 1
    assert summary["error_static_gt_mass"] == 1
    assert summary["traceable_static_gt_mass"] == 0


def test_cli_skips_malformed_and_missing_field_json_with_warning(tmp_path, capsys):
    _write_bonus_map(
        tmp_path,
        "direct-1",
        {
            "instance_id": "direct-1",
            "case_type": "direct",
            "traceable": True,
            "error": False,
            "patched_callables": [_patched_callable()],
            "call_graph_nodes": {"pkg/mod.py::target": _node("pkg/mod.py", 0, 0.0)},
            "hop_max": 1,
        },
    )
    (tmp_path / "malformed.json").write_text("{not-json")
    (tmp_path / "missing-fields.json").write_text(json.dumps({"instance_id": "missing-fields"}))

    output_path = tmp_path / "summary.json"
    f5.main([str(tmp_path), "--output", str(output_path)])
    captured = capsys.readouterr()

    assert "[WARN]" in captured.err
    assert "malformed.json" in captured.err
    assert "missing-fields.json" in captured.err
    assert "Skipped files: 2" in captured.out
    assert "Total instances scanned: 1" in captured.out

    output = json.loads(output_path.read_text())
    assert output["total_instances_scanned"] == 1
    assert output["files_skipped"] == 2
