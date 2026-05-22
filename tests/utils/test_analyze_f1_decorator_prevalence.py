from __future__ import annotations

import json

import pytest

from utils.p2a import analyze_f1_decorator_prevalence as analyzer


def _file_diff(old_source: str, new_source: str, path: str = "pkg/module.py") -> dict:
    return {
        "header": {"file": {"path": path}},
        "old_file_content": old_source,
        "new_file_content": new_source,
    }


def test_property_to_cached_property_swap_is_decorator_swap_and_f1_affected():
    old_source = """\
class Widget:
    @property
    def value(self):
        return 1
"""
    new_source = """\
class Widget:
    @cached_property
    def value(self):
        return 1
"""

    result = analyzer.analyze_file_diff(_file_diff(old_source, new_source))

    assert analyzer.CATEGORY_SWAP in result["categories"]
    assert analyzer.CATEGORY_F1 in result["categories"]
    assert result["modified_callables"] == []
    assert result["hunks"][0]["callable_qnames"] == ["Widget.value"]
    assert result["hunks"][0]["f1_affected"] is True


def test_staticmethod_add_is_add_or_remove_and_f1_affected():
    old_source = """\
class Parser:
    def clean(value):
        return value.strip()
"""
    new_source = """\
class Parser:
    @staticmethod
    def clean(value):
        return value.strip()
"""

    result = analyzer.analyze_file_diff(_file_diff(old_source, new_source))

    assert analyzer.CATEGORY_ADD_OR_REMOVE in result["categories"]
    assert analyzer.CATEGORY_F1 in result["categories"]
    assert result["modified_callables"] == []
    assert result["hunks"][0]["callable_qnames"] == ["Parser.clean"]
    assert result["hunks"][0]["f1_affected"] is True


def test_body_change_in_decorated_function_is_not_f1_affected():
    old_source = """\
class Widget:
    @property
    def value(self):
        return 1
"""
    new_source = """\
class Widget:
    @property
    def value(self):
        return 2
"""

    result = analyzer.analyze_file_diff(_file_diff(old_source, new_source))

    assert analyzer.CATEGORY_SWAP not in result["categories"]
    assert analyzer.CATEGORY_ADD_OR_REMOVE not in result["categories"]
    assert analyzer.CATEGORY_F1 not in result["categories"]
    assert [item["qualified_name"] for item in result["modified_callables"]] == ["Widget.value"]


def test_analyze_instance_ignores_test_file_diffs():
    old_source = "def test_example():\n    assert True\n"
    new_source = "@pytest.mark.slow\ndef test_example():\n    assert True\n"
    task = {
        "instance_id": "repo__1",
        "parsed_commit_content": {
            "file_diffs": [_file_diff(old_source, new_source, path="tests/test_module.py")]
        },
    }

    result = analyzer.analyze_instance(task)

    assert result["has_non_test_py_diff"] is False
    assert result["categories"] == []


def test_analyze_parquet_reports_prevalence(tmp_path):
    pytest.importorskip("pyarrow")
    pd = pytest.importorskip("pandas")

    old_source = """\
class Parser:
    def clean(value):
        return value.strip()
"""
    new_source = """\
class Parser:
    @staticmethod
    def clean(value):
        return value.strip()
"""
    task = {
        "repo_name": "sample",
        "commit_hash": "abcdef0123456789",
        "relevant_files": ["pkg/module.py"],
        "parsed_commit_content": {
            "file_diffs": [_file_diff(old_source, new_source)],
        },
    }
    parquet_path = tmp_path / "sample.parquet"
    pd.DataFrame([{"extra_info": json.dumps(task)}]).to_parquet(parquet_path)

    report = analyzer.analyze_parquet(parquet_path)

    assert report["summary"]["total_instances"] == 1
    assert report["summary"]["instances_with_any_non_test_py_diff"] == 1
    assert report["summary"]["instances_with_zero_non_test_py_diff"] == 0
    assert report["summary"]["categories"][analyzer.CATEGORY_ADD_OR_REMOVE]["instances"] == 1
    assert report["summary"]["categories"][analyzer.CATEGORY_F1]["instances"] == 1
    assert report["details"][0]["instance_id"] == "sample__abcdef01"
