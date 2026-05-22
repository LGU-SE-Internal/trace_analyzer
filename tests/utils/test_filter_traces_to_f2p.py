from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _is_test_file(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    for part in parts:
        if part in ("tests", "test", "testing"):
            return True
        if part.startswith("test_") or part.endswith("_test.py"):
            return True
    return False


def _module(name: str, package: bool = False) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=package)
    if package:
        module.__path__ = []
    return module


def _load_precompute_bonus_maps():
    try:
        return importlib.import_module("utils.p2a.precompute_bonus_maps")
    except ModuleNotFoundError as exc:
        if exc.name not in {"pandas", "torch"}:
            raise

    sys.modules.pop("utils.p2a.precompute_bonus_maps", None)
    for name in list(sys.modules):
        if name == "rllm" or name.startswith("rllm."):
            sys.modules.pop(name, None)

    inserted: list[str] = []
    if "pandas" not in sys.modules:
        sys.modules["pandas"] = _module("pandas")
        inserted.append("pandas")

    for name in ("rllm", "rllm.environments", "rllm.environments.swe"):
        sys.modules[name] = _module(name, package=True)
        inserted.append(name)

    trace = _module("rllm.environments.swe.trace")
    trace.TRACE_FILE_PATH = "/tmp/fault_traces.jsonl"
    trace._is_test_file = _is_test_file
    trace.extract_callables_from_ast = lambda *args, **kwargs: {}
    trace.find_modified_callables_from_task = lambda *args, **kwargs: []
    trace.make_instance_id = lambda task: "stub"
    trace.normalize_task = lambda task: task
    sys.modules["rllm.environments.swe.trace"] = trace
    inserted.append("rllm.environments.swe.trace")

    try:
        return importlib.import_module("utils.p2a.precompute_bonus_maps")
    finally:
        for name in inserted:
            sys.modules.pop(name, None)


precompute = _load_precompute_bonus_maps()
_filter_traces_to_f2p = precompute._filter_traces_to_f2p


TEST_FILE = "tests/test_widget.py"
SRC_FILE = "src/widget.py"


def _frame(func_name: str, file_path: str = TEST_FILE, is_patched: bool = False) -> dict:
    return {"file_path": file_path, "func_name": func_name, "is_patched": is_patched}


def test_class_qualified_f2p_frame_is_kept():
    trace = [_frame("TestFoo.test_scale")]

    assert _filter_traces_to_f2p([trace], {"test_scale"}) == [trace]


def test_bare_f2p_frame_is_kept():
    trace = [_frame("test_scale")]

    assert _filter_traces_to_f2p([trace], {"test_scale"}) == [trace]


def test_module_level_f2p_frame_is_kept():
    trace = [_frame("test_module_level")]

    assert _filter_traces_to_f2p([trace], {"test_module_level"}) == [trace]


def test_parametrized_class_qualified_frame_is_stripped_and_kept():
    trace = [_frame("TestFoo.test_x[case1]")]

    assert _filter_traces_to_f2p([trace], {"test_x"}) == [trace]


def test_class_qualified_fixture_frame_in_test_file_is_kept():
    trace = [_frame("TestFoo.setUp", is_patched=False)]

    assert _filter_traces_to_f2p([trace], {"test_unrelated"}) == [trace]


def test_bare_fixture_frame_in_test_file_is_kept():
    trace = [_frame("setUp")]

    assert _filter_traces_to_f2p([trace], {"test_unrelated"}) == [trace]


def test_non_f2p_class_qualified_frame_is_dropped():
    trace = [_frame("TestFoo.test_unrelated")]

    assert _filter_traces_to_f2p([trace], {"test_scale"}) == []


def test_nested_call_chain_is_kept_when_any_test_frame_matches_f2p():
    trace = [
        _frame("TestFoo.test_scale", file_path=TEST_FILE),
        _frame("some_helper", file_path=SRC_FILE),
        _frame("patched_func", file_path=SRC_FILE, is_patched=True),
    ]

    assert _filter_traces_to_f2p([trace], {"test_scale"}) == [trace]


def test_empty_f2p_set_drops_any_trace_including_fixtures():
    traces = [
        [_frame("TestFoo.test_scale")],
        [_frame("TestFoo.setUp")],
    ]

    assert _filter_traces_to_f2p(traces, set()) == []


def test_matching_name_in_non_test_file_is_ignored():
    trace = [_frame("test_scale", file_path=SRC_FILE)]

    assert _filter_traces_to_f2p([trace], {"test_scale"}) == []
