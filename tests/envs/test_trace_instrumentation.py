import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_trace_module():
    trace_path = Path(__file__).resolve().parents[2] / "rllm" / "environments" / "swe" / "trace.py"
    spec = importlib.util.spec_from_file_location("_trace_under_test", trace_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


instrument_source = _load_trace_module().instrument_source


def _install_tracer_stub(monkeypatch):
    calls = []
    stub = ModuleType("_swe_fault_tracer")

    def trace(callable_name, file_path, def_lineno):
        calls.append(
            {
                "callable_name": callable_name,
                "file_path": file_path,
                "def_lineno": def_lineno,
            }
        )

    stub.trace = trace
    monkeypatch.setitem(sys.modules, "_swe_fault_tracer", stub)
    return calls


def _exec_instrumented(source, callables):
    output = instrument_source(source, callables)
    namespace = {}
    exec(compile(output, "m.py", "exec"), namespace)
    return output, namespace


class TestF2_TraceInstrumentation:
    @pytest.mark.parametrize(
        ("source", "callables", "expression", "expected_name", "expected_result"),
        [
            (
                "def g(): return 1\n",
                [{"name": "g", "qualified_name": "g", "file_path": "m.py", "start_line": 1}],
                "g()",
                "g",
                1,
            ),
            (
                "def g(): pass\n",
                [{"name": "g", "qualified_name": "g", "file_path": "m.py", "start_line": 1}],
                "g()",
                "g",
                None,
            ),
            (
                "class C:\n    def m(self): return 1\n",
                [{"name": "m", "qualified_name": "C.m", "file_path": "m.py", "start_line": 2}],
                "C().m()",
                "C.m",
                1,
            ),
            (
                "def g(): a = 1; return a\n",
                [{"name": "g", "qualified_name": "g", "file_path": "m.py", "start_line": 1}],
                "g()",
                "g",
                1,
            ),
        ],
    )
    def test_f2_oneline_suites_compile_and_trace(
        self, monkeypatch, source, callables, expression, expected_name, expected_result
    ):
        calls = _install_tracer_stub(monkeypatch)

        output, namespace = _exec_instrumented(source, callables)
        assert compile(output, "m.py", "exec")

        assert eval(expression, namespace) == expected_result
        assert calls
        assert calls[0]["callable_name"] == expected_name

    def test_f2_multiline_pass_control_still_compiles_and_traces(self, monkeypatch):
        calls = _install_tracer_stub(monkeypatch)
        source = "def f():\n    pass\n"
        callables = [{"name": "f", "qualified_name": "f", "file_path": "m.py", "start_line": 1}]

        _, namespace = _exec_instrumented(source, callables)

        assert namespace["f"]() is None
        assert calls
        assert calls[0]["callable_name"] == "f"

    def test_f2_multiline_return_regression_preserves_top_of_body_trace(self, monkeypatch):
        calls = _install_tracer_stub(monkeypatch)
        source = "def f():\n    return 1\n"
        callables = [{"name": "f", "qualified_name": "f", "file_path": "m.py", "start_line": 1}]

        output, namespace = _exec_instrumented(source, callables)

        assert output.splitlines()[1] == "    try:"
        assert namespace["f"]() == 1
        assert calls
        assert calls[0]["callable_name"] == "f"
