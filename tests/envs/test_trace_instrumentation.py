import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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


def _instrument_exec_and_call(source, function_name):
    calls = []
    stub = SimpleNamespace(trace=lambda *args: calls.append(args))
    previous_tracer = sys.modules.get("_swe_fault_tracer")
    had_previous_tracer = "_swe_fault_tracer" in sys.modules
    sys.modules["_swe_fault_tracer"] = stub

    try:
        instrumented = instrument_source(
            source,
            [
                {
                    "name": function_name,
                    "qualified_name": function_name,
                    "file_path": "m.py",
                    "start_line": 1,
                }
            ],
        )
        compiled = compile(instrumented, "m.py", "exec")
        namespace = {}
        exec(compiled, namespace)
        namespace[function_name]()
        return calls, namespace, instrumented
    finally:
        if had_previous_tracer:
            sys.modules["_swe_fault_tracer"] = previous_tracer
        else:
            del sys.modules["_swe_fault_tracer"]


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


class TestF3TokenAwareSignatureScan:
    def test_f3_signature_variants_compile_and_trace(self):
        cases = [
            (
                "f",
                'def f(\n    x="):"\n):\n    return x\n',
                None,
            ),
            ("h", 'def h(x="# foo"):\n    return x\n', None),
            ("k", 'def k(x="# foo", y=1):\n    return x, y\n', None),
            (
                "i",
                "def i(\n    x: dict = {1: 2}\n):\n    return x\n",
                None,
            ),
            (
                "j",
                'def j(\n    x: "List[int]" = None\n):\n    return x\n',
                None,
            ),
            ("g", 'def g(x=")"):\n    return x\n', None),
            ("m", 'def m():\n    """doc"""\n    return 1\n', "doc"),
        ]

        for function_name, source, expected_doc in cases:
            calls, namespace, _ = _instrument_exec_and_call(source, function_name)

            assert calls == [(function_name, "m.py", 1)], function_name
            if expected_doc is not None:
                assert namespace[function_name].__doc__ == expected_doc
