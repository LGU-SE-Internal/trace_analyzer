from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE_PATH = REPO_ROOT / "rllm" / "environments" / "swe" / "trace.py"

spec = importlib.util.spec_from_file_location("_trace_under_test_direction", TRACE_PATH)
trace_mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = trace_mod
assert spec.loader is not None
spec.loader.exec_module(trace_mod)

aggregate_traces = trace_mod.aggregate_traces
build_call_graph_from_traces = trace_mod.build_call_graph_from_traces
generate_tracer_module = trace_mod.generate_tracer_module
parse_fault_traces_from_file = trace_mod.parse_fault_traces_from_file
_is_test_file = trace_mod._is_test_file


def _clean(source: str) -> str:
    return textwrap.dedent(source).lstrip()


def _install_generated_tracer(tmp_path: Path, source: str | None = None):
    source = source or generate_tracer_module(str(tmp_path))
    module = types.ModuleType("_swe_fault_tracer")
    exec(compile(source, "_swe_fault_tracer.py", "exec"), module.__dict__)
    trace_file = tmp_path / "fault-traces.jsonl"
    module._TRACE_FILE = str(trace_file)
    previous = sys.modules.get("_swe_fault_tracer")
    sys.modules["_swe_fault_tracer"] = module
    return module, trace_file, previous


def _restore_generated_tracer(previous) -> None:
    if previous is None:
        sys.modules.pop("_swe_fault_tracer", None)
    else:
        sys.modules["_swe_fault_tracer"] = previous


def _exec_source(tmp_path: Path, name: str, source: str):
    path = tmp_path / f"{name}.py"
    cleaned = _clean(source)
    path.write_text(cleaned)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__dict__["__name__"] = name
    exec(compile(cleaned, str(path), "exec"), module.__dict__)
    return module, path, cleaned


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _mc(
    qualified_name: str = "patched",
    file_path: str = "pkg/patched.py",
    start_line: int = 5,
    end_line: int = 12,
) -> dict:
    return {
        "qualified_name": qualified_name,
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


def _patched_frame(file_path: str = "pkg/patched.py", line_no: int = 10) -> dict:
    return _frame(file_path, "patched", line_no, is_patched=True, qualified_name="patched")


def _outer_to_inner_trace() -> list[dict]:
    return [
        _frame("tests/test_x.py", "test_x", 30, qualified_name="test_x"),
        _frame("pkg/middle.py", "middle", 20, qualified_name="middle"),
        _patched_frame(),
    ]


def test_generated_tracer_trace_parses_outer_to_inner_frames(tmp_path, monkeypatch):
    _tracer, trace_file, previous = _install_generated_tracer(tmp_path)
    try:
        module, path, _source = _exec_source(
            tmp_path,
            "three_deep_parse",
            """
            import _swe_fault_tracer

            def outer():
                return middle()

            def middle():
                return inner_patched()

            def inner_patched():
                _swe_fault_tracer.trace("inner_patched", "three_deep_parse.py", 9)
            """,
        )

        module.outer()

        def fake_read(_env, _path):
            return trace_file.read_text(), 0

        monkeypatch.setattr(trace_mod, "_read_sandbox_file", fake_read)
        modified = [
            {
                "file_path": path.name,
                "qualified_name": "inner_patched",
                "start_line": module.inner_patched.__code__.co_firstlineno,
                "end_line": module.inner_patched.__code__.co_firstlineno + 2,
            }
        ]

        traces = parse_fault_traces_from_file(object(), modified, str(tmp_path))
        frames = traces[0]

        assert frames[0]["func_name"].endswith("outer")
        assert frames[-1]["func_name"].endswith("inner_patched")
        assert frames[-1]["is_patched"] is True
    finally:
        _restore_generated_tracer(previous)


def test_build_call_graph_requires_outer_to_inner_for_hop_chain():
    result = build_call_graph_from_traces([_outer_to_inner_trace()], [_mc()])
    nodes = result["call_graph_nodes"]

    assert "tests/test_x.py::test_x" in nodes
    assert "pkg/patched.py::patched" in nodes
    assert nodes["tests/test_x.py::test_x"]["hop_distance"] > nodes["pkg/middle.py::middle"]["hop_distance"]
    assert nodes["pkg/middle.py::middle"]["hop_distance"] > nodes["pkg/patched.py::patched"]["hop_distance"]


def test_reversed_input_silently_omits_test_file_node():
    traces = [
        [
            _patched_frame(),
            _frame("pkg/middle.py", "middle", 20, qualified_name="middle"),
            _frame("tests/test_x.py", "test_x", 30, qualified_name="test_x"),
        ]
    ]

    result = build_call_graph_from_traces(traces, [_mc()])
    nodes = result["call_graph_nodes"]

    assert len(nodes) == 1
    assert "pkg/patched.py::patched" in nodes
    assert not any(_is_test_file(node["file_path"]) for node in nodes.values())


def test_generated_tracer_raw_jsonl_is_outer_to_inner(tmp_path):
    _tracer, trace_file, previous = _install_generated_tracer(tmp_path)
    try:
        module, _path, _source = _exec_source(
            tmp_path,
            "three_deep_raw",
            """
            import _swe_fault_tracer

            def outer():
                return middle()

            def middle():
                return inner_patched()

            def inner_patched():
                _swe_fault_tracer.trace("inner_patched", "three_deep_raw.py", 9)
            """,
        )

        module.outer()
        entries = _read_jsonl(trace_file)
        names = [frame["name"] for frame in entries[0]["frames"]]

        assert names[0].endswith("outer")
        assert names[1].endswith("middle")
        assert names[-1].endswith("inner_patched")
    finally:
        _restore_generated_tracer(previous)


def test_happy_path_call_graph_contains_test_entry():
    result = build_call_graph_from_traces([_outer_to_inner_trace()], [_mc()])

    assert any(_is_test_file(node["file_path"]) for node in result["call_graph_nodes"].values())


def test_getframe_chain_is_reversed_before_frame_processing(tmp_path):
    source = generate_tracer_module(str(tmp_path))
    source = source.replace(
        "        frames.reverse()\n",
        "        frames.reverse()\n        _CAPTURED_FRAME_NAMES[:] = [f.f_code.co_name for f in frames]\n",
    )
    tracer, _trace_file, previous = _install_generated_tracer(tmp_path, source)
    tracer._CAPTURED_FRAME_NAMES = []
    try:
        module, _path, _source = _exec_source(
            tmp_path,
            "four_deep_white_box",
            """
            import _swe_fault_tracer

            def level1():
                return level2()

            def level2():
                return level3()

            def level3():
                return level4_patched()

            def level4_patched():
                _swe_fault_tracer.trace("level4_patched", "four_deep_white_box.py", 12)
            """,
        )

        module.level1()
        stack_names = [
            name
            for name in tracer._CAPTURED_FRAME_NAMES
            if name in {"level1", "level2", "level3", "level4_patched"}
        ]

        assert stack_names == ["level1", "level2", "level3", "level4_patched"]
    finally:
        _restore_generated_tracer(previous)


def test_aggregate_traces_preserves_outer_to_inner_ordering():
    trace = _outer_to_inner_trace()

    aggregated = aggregate_traces([trace])

    assert [[frame["func_name"] for frame in item] for item in aggregated] == [["test_x", "middle", "patched"]]
