from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE_PATH = REPO_ROOT / "rllm" / "environments" / "swe" / "trace.py"

spec = importlib.util.spec_from_file_location("_trace_under_test", TRACE_PATH)
trace_mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = trace_mod
assert spec.loader is not None
spec.loader.exec_module(trace_mod)

extract_callables_from_ast = trace_mod.extract_callables_from_ast
find_modified_callables_from_sources = trace_mod.find_modified_callables_from_sources
generate_tracer_module = trace_mod.generate_tracer_module
parse_fault_traces_from_file = trace_mod.parse_fault_traces_from_file
build_call_graph_from_traces = trace_mod.build_call_graph_from_traces


def _clean(source: str) -> str:
    return textwrap.dedent(source).lstrip()


def _install_generated_tracer(tmp_path: Path):
    source = generate_tracer_module(str(tmp_path))
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


def _import_source(tmp_path: Path, name: str, source: str):
    path = tmp_path / f"{name}.py"
    cleaned = _clean(source)
    path.write_text(cleaned)
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    assert module_spec.loader is not None
    module_spec.loader.exec_module(module)
    return module, path, cleaned, previous


def _restore_imported(name: str, previous) -> None:
    if previous is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = previous


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class TestF4_StaticSide:
    def test_sibling_nested_functions_get_distinct_qualified_names(self):
        source = _clean(
            """
            def a():
                def inner():
                    return 1
            def b():
                def inner():
                    return 2
            """
        )

        callables = extract_callables_from_ast(source, "m.py")

        assert {"a", "b", "a.<locals>.inner", "b.<locals>.inner"} <= set(callables)
        assert "inner" not in callables

    def test_function_nested_class_method_is_recorded(self):
        source = _clean(
            """
            def a():
                class Inner:
                    def m(self):
                        return 1
            """
        )

        callables = extract_callables_from_ast(source, "m.py")
        info = callables["a.<locals>.Inner.m"]

        assert info.name == "m"
        assert info.start_line == 3
        assert info.end_line == 4

    def test_nested_child_change_is_not_attributed_to_parent(self):
        old_source = _clean(
            """
            def a():
                def inner():
                    return 1
            def b():
                def inner():
                    return 2
            """
        )
        new_source = old_source.replace("return 1", "return 999")

        modified = find_modified_callables_from_sources(old_source, new_source, "m.py")

        assert [item["qualified_name"] for item in modified] == [
            "a.<locals>.inner"
        ]

    def test_nested_class_method_uses_class_path(self):
        source = _clean(
            """
            class A:
                class B:
                    def m(self):
                        return 1
            """
        )

        callables = extract_callables_from_ast(source, "m.py")

        assert "A.B.m" in callables

    def test_method_nested_function_uses_locals_separator(self):
        source = _clean(
            """
            class A:
                def m(self):
                    def inner():
                        return 1
                    return inner()
            """
        )

        callables = extract_callables_from_ast(source, "m.py")

        assert "A.m" in callables
        assert "A.m.<locals>.inner" in callables

    def test_top_level_free_function_regression(self):
        source = _clean(
            """
            def foo():
                return 1
            """
        )

        callables = extract_callables_from_ast(source, "m.py")

        assert list(callables) == ["foo"]

    def test_top_level_class_method_regression(self):
        source = _clean(
            """
            class A:
                def m(self):
                    return 1
            """
        )

        callables = extract_callables_from_ast(source, "m.py")

        assert "A.m" in callables

    def test_downstream_node_key_contract_stays_unique(self):
        source = _clean(
            """
            class A:
                def handle(self):
                    return 1
            class B:
                def handle(self):
                    return 2
            """
        )

        callables = extract_callables_from_ast(source, "m.py")
        node_keys = [
            f"{info.file_path}::{info.qualified_name}"
            for info in callables.values()
        ]

        assert len(node_keys) == len(set(node_keys))
        assert {"m.py::A.handle", "m.py::B.handle"} <= set(node_keys)


class TestF6_RuntimeSide:
    def test_generated_tracer_source_is_valid_and_uses_live_frames(self, tmp_path):
        source = generate_tracer_module(str(tmp_path), str(tmp_path / "alt"))

        compile(source, "_swe_fault_tracer.py", "exec")

        assert "traceback.extract_stack" not in source
        assert "sys._getframe(1)" in source

    def test_generated_tracer_records_class_qualified_method_names(self, tmp_path):
        tracer, trace_file, previous = _install_generated_tracer(tmp_path)
        try:
            module, _path, _source = _exec_source(
                tmp_path,
                "m",
                """
                import _swe_fault_tracer

                class A:
                    def handle(self):
                        _swe_fault_tracer.trace("A.handle", "m.py", 1)

                class B:
                    def handle(self):
                        _swe_fault_tracer.trace("B.handle", "m.py", 5)
                """,
            )

            module.A().handle()
            module.B().handle()
            entries = _read_jsonl(trace_file)

            assert tracer._TRACE_FILE == str(trace_file)
            assert entries[0]["frames"][0]["name"] == "A.handle"
            assert entries[1]["frames"][0]["name"] == "B.handle"
        finally:
            _restore_generated_tracer(previous)

    def test_staticmethod_gap_is_bare_name_on_python_310_recipe(self, tmp_path):
        _tracer, trace_file, previous = _install_generated_tracer(tmp_path)
        try:
            module, _path, _source = _exec_source(
                tmp_path,
                "static_gap",
                """
                import _swe_fault_tracer

                class A:
                    @staticmethod
                    def helper():
                        _swe_fault_tracer.trace("A.helper", "m.py", 1)

                A.helper()
                """,
            )

            assert module.A is not None
            entries = _read_jsonl(trace_file)

            assert entries[0]["frames"][0]["name"] == "helper"
        finally:
            _restore_generated_tracer(previous)

    def test_parser_trusts_jsonl_qualified_names(self, monkeypatch):
        entry = {
            "callable": "Other.handle",
            "file": "m.py",
            "lineno": 1,
            "frames": [
                {
                    "file": "/repo/m.py",
                    "line": 3,
                    "name": "A.handle",
                    "code": "return 1",
                }
            ],
        }

        def fake_read(_env, _path):
            return json.dumps(entry) + "\n", 0

        monkeypatch.setattr(trace_mod, "_read_sandbox_file", fake_read)
        modified = [
            {
                "file_path": "m.py",
                "qualified_name": "Different.handle",
                "start_line": 1,
                "end_line": 10,
            }
        ]

        traces = parse_fault_traces_from_file(object(), modified, "/repo")

        assert traces[0][0]["is_patched"] is True
        assert traces[0][0]["qualified_name"] == "A.handle"

    def test_call_graph_keeps_same_bare_method_names_distinct(self):
        modified = [
            {
                "file_path": "m.py",
                "qualified_name": "A.handle",
                "start_line": 2,
                "end_line": 3,
            },
            {
                "file_path": "m.py",
                "qualified_name": "B.handle",
                "start_line": 6,
                "end_line": 7,
            },
        ]
        traces = [
            [
                {
                    "file_path": "m.py",
                    "line_no": 3,
                    "func_name": "A.handle",
                    "qualified_name": "A.handle",
                    "line_content": "return 1",
                    "is_patched": True,
                }
            ],
            [
                {
                    "file_path": "m.py",
                    "line_no": 7,
                    "func_name": "B.handle",
                    "qualified_name": "B.handle",
                    "line_content": "return 2",
                    "is_patched": True,
                }
            ],
        ]

        graph = build_call_graph_from_traces(traces, modified)
        nodes = graph["call_graph_nodes"]

        assert "m.py::A.handle" in nodes
        assert "m.py::B.handle" in nodes
        assert len(nodes) == 2


class TestCrossInvariant:
    def test_class_method_static_and_runtime_names_match(self, tmp_path):
        tracer, _trace_file, previous_tracer = _install_generated_tracer(tmp_path)
        module_name = "cross_methods"
        module, path, source, previous_module = _import_source(
            tmp_path,
            module_name,
            """
            import sys

            CAPTURED = {}

            class A:
                def handle(self):
                    CAPTURED["frame"] = sys._getframe(0)

            class B:
                def handle(self):
                    CAPTURED["b_frame"] = sys._getframe(0)
            """,
        )
        try:
            callables = extract_callables_from_ast(source, str(path))
            node_keys = {
                f"{info.file_path}::{info.qualified_name}"
                for info in callables.values()
            }

            module.A().handle()
            runtime_name = tracer._resolve_qualname(module.CAPTURED["frame"])

            assert {"A.handle", "B.handle"} <= set(callables)
            assert runtime_name == "A.handle"
            assert f"{path}::{runtime_name}" in node_keys
        finally:
            _restore_imported(module_name, previous_module)
            _restore_generated_tracer(previous_tracer)

    def test_nested_function_static_and_runtime_names_match(self, tmp_path):
        tracer, _trace_file, previous_tracer = _install_generated_tracer(tmp_path)
        module_name = "cross_nested"
        module, path, source, previous_module = _import_source(
            tmp_path,
            module_name,
            """
            import sys

            CAPTURED = {}

            def outer():
                def inner():
                    CAPTURED["frame"] = sys._getframe(0)
                inner()

            outer()
            """,
        )
        try:
            callables = extract_callables_from_ast(source, str(path))
            node_keys = {
                f"{info.file_path}::{info.qualified_name}"
                for info in callables.values()
            }

            assert module.outer is not None
            runtime_name = tracer._resolve_qualname(module.CAPTURED["frame"])

            assert "outer.<locals>.inner" in callables
            assert runtime_name == "outer.<locals>.inner"
            assert f"{path}::{runtime_name}" in node_keys
        finally:
            _restore_imported(module_name, previous_module)
            _restore_generated_tracer(previous_tracer)
