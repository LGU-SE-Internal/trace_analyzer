from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE_PATH = REPO_ROOT / "rllm" / "environments" / "swe" / "trace.py"

spec = importlib.util.spec_from_file_location("_trace_under_test", TRACE_PATH)
trace_mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = trace_mod
assert spec.loader is not None
spec.loader.exec_module(trace_mod)

build_call_graph_from_traces = trace_mod.build_call_graph_from_traces


def _mc(name: str, start_line: int = 10, end_line: int = 15, file_path: str = "m.py") -> dict:
    return {
        "qualified_name": name,
        "file_path": file_path,
        "start_line": start_line,
        "end_line": end_line,
    }


def _frame(
    file_path: str,
    func_name: str,
    line_no: int,
    *,
    is_patched: bool = False,
    qualified_name: str | None = None,
) -> dict:
    frame = {
        "file_path": file_path,
        "func_name": func_name,
        "line_no": line_no,
        "is_patched": is_patched,
    }
    if qualified_name is not None:
        frame["qualified_name"] = qualified_name
    return frame


def _test_frame(name: str = "test_x", line_no: int = 1) -> dict:
    return _frame("test_m.py", name, line_no)


def _patched_frame(name: str = "fix_a", line_no: int = 11) -> dict:
    return _frame("m.py", name, line_no, is_patched=True, qualified_name=name)


def _unobserved_names(graph: dict) -> list[str]:
    return [mc["qualified_name"] for mc in graph["unobserved_patched_callables"]]


def _load_classify_bonus_map():
    analyze_path = REPO_ROOT / "utils" / "p2a" / "analyze_traceability.py"
    module_names = (
        "pandas",
        "rllm",
        "rllm.environments",
        "rllm.environments.swe",
        "rllm.environments.swe.trace",
    )
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in module_names}

    try:
        sys.modules["pandas"] = previous["pandas"] if previous["pandas"] is not missing else types.ModuleType("pandas")

        rllm_pkg = types.ModuleType("rllm")
        rllm_pkg.__path__ = []
        env_pkg = types.ModuleType("rllm.environments")
        env_pkg.__path__ = []
        swe_pkg = types.ModuleType("rllm.environments.swe")
        swe_pkg.__path__ = []
        swe_pkg.trace = trace_mod
        env_pkg.swe = swe_pkg
        rllm_pkg.environments = env_pkg

        sys.modules["rllm"] = rllm_pkg
        sys.modules["rllm.environments"] = env_pkg
        sys.modules["rllm.environments.swe"] = swe_pkg
        sys.modules["rllm.environments.swe.trace"] = trace_mod

        spec = importlib.util.spec_from_file_location("_analyze_traceability_under_test", analyze_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module.classify_bonus_map
    finally:
        for name, value in previous.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def test_observed_static_gt_is_seeded_at_hop_zero_with_schema_field():
    graph = build_call_graph_from_traces([[_test_frame(), _patched_frame()]], [_mc("fix_a")])
    nodes = graph["call_graph_nodes"]

    assert nodes["m.py::fix_a"]["hop_distance"] == 0
    assert nodes["m.py::fix_a"]["normalized_distance"] == 0.0
    assert nodes["m.py::fix_a"]["observed_in_trace"] is True
    assert graph["unobserved_patched_callables"] == []
    assert graph["traceable"] is True


def test_observed_patched_frame_not_in_static_gt_still_appears_from_trace_walk():
    graph = build_call_graph_from_traces([[_test_frame(), _patched_frame("runtime_fix")]], [])
    nodes = graph["call_graph_nodes"]

    assert nodes["m.py::runtime_fix"]["hop_distance"] == 0
    assert nodes["m.py::runtime_fix"]["observed_in_trace"] is True
    assert graph["patched_callables"] == []
    assert graph["unobserved_patched_callables"] == []
    assert graph["traceable"] is True


def test_unobserved_static_gt_is_metadata_only():
    traces = [[_frame("m.py", "fix_a", 11, qualified_name="fix_a")]]

    graph = build_call_graph_from_traces(traces, [_mc("fix_a")])

    assert graph["call_graph_nodes"] == {}
    assert _unobserved_names(graph) == ["fix_a"]
    assert graph["traceable"] is False


def test_empty_traces_are_not_traceable_and_all_static_gt_is_unobserved():
    graph = build_call_graph_from_traces([], [_mc("fix_a"), _mc("fix_b", 20, 25)])

    assert graph["call_graph_nodes"] == {}
    assert graph["hop_max"] == 0
    assert _unobserved_names(graph) == ["fix_a", "fix_b"]
    assert graph["traceable"] is False


def test_single_empty_trace_is_not_traceable_and_all_static_gt_is_unobserved():
    graph = build_call_graph_from_traces([[]], [_mc("fix_a"), _mc("fix_b", 20, 25)])

    assert graph["call_graph_nodes"] == {}
    assert graph["hop_max"] == 0
    assert _unobserved_names(graph) == ["fix_a", "fix_b"]
    assert graph["traceable"] is False


def test_mixed_static_gt_only_observed_member_is_seeded():
    graph = build_call_graph_from_traces(
        [[_test_frame(), _patched_frame("fix_a")]],
        [_mc("fix_a"), _mc("fix_b", 20, 25)],
    )
    nodes = graph["call_graph_nodes"]

    assert "m.py::fix_a" in nodes
    assert "m.py::fix_b" not in nodes
    assert sum(node["hop_distance"] == 0 for node in nodes.values()) == 1
    assert _unobserved_names(graph) == ["fix_b"]
    assert graph["traceable"] is True


def test_any_true_patched_frame_counts_as_observed_for_static_gt():
    traces = [
        [
            _frame("m.py", "fix_a", 11, is_patched=False, qualified_name="fix_a"),
            _patched_frame("fix_a", 12),
        ]
    ]

    graph = build_call_graph_from_traces(traces, [_mc("fix_a")])

    assert graph["call_graph_nodes"]["m.py::fix_a"]["hop_distance"] == 0
    assert graph["call_graph_nodes"]["m.py::fix_a"]["observed_in_trace"] is True
    assert graph["unobserved_patched_callables"] == []


def test_multiple_traces_same_observed_callable_are_idempotent():
    traces = [
        [_test_frame(), _patched_frame("fix_a", 11)],
        [_test_frame(), _patched_frame("fix_a", 12)],
    ]

    graph = build_call_graph_from_traces(traces, [_mc("fix_a")])
    nodes = graph["call_graph_nodes"]

    assert list(nodes).count("m.py::fix_a") == 1
    assert nodes["m.py::fix_a"]["hop_distance"] == 0
    assert nodes["m.py::fix_a"]["observed_in_trace"] is True
    assert sum(node["hop_distance"] == 0 for node in nodes.values()) == 1


def test_intermediate_trace_nodes_keep_hop_distance_and_observed_schema():
    graph = build_call_graph_from_traces(
        [[_test_frame(), _frame("m.py", "helper", 30, qualified_name="helper"), _patched_frame("fix_a")]],
        [_mc("fix_a")],
    )
    nodes = graph["call_graph_nodes"]

    assert nodes["m.py::fix_a"]["hop_distance"] == 0
    assert nodes["m.py::helper"]["hop_distance"] == 1
    assert nodes["m.py::helper"]["observed_in_trace"] is True
    assert nodes["test_m.py::test_x"]["hop_distance"] == 2
    assert nodes["test_m.py::test_x"]["observed_in_trace"] is True
    assert graph["unobserved_patched_callables"] == []


def test_classify_bonus_map_accepts_new_call_graph_schema():
    graph = build_call_graph_from_traces(
        [[_test_frame(), _frame("m.py", "helper", 30, qualified_name="helper"), _patched_frame("fix_a")]],
        [_mc("fix_a"), _mc("fix_b", 20, 25)],
    )
    classify_bonus_map = _load_classify_bonus_map()

    result = classify_bonus_map(
        {
            "instance_id": "demo",
            "patched_callables": graph["patched_callables"],
            "call_graph_nodes": graph["call_graph_nodes"],
            "hop_max": graph["hop_max"],
            "traceable": graph["traceable"],
            "unobserved_patched_callables": graph["unobserved_patched_callables"],
        }
    )

    assert result["n_test_entries"] == 1
    assert result["n_intermediate"] == 1
    assert result["case_type"] == "standard"
    assert result["category"] == "traceable"
