"""
tests/test_parser.py
─────────────────────
Unit tests for nexus.ai.parser — JSON extraction and validation.

These tests require NO external dependencies (no Ollama, no network).
Run with: pytest tests/test_parser.py -v
"""
import pytest
from nexus.ai.parser import (
    extract_json,
    validate_fix,
    validate_generation,
    validate_plan,
)


# ── extract_json ─────────────────────────────────────────────────────────────

class TestExtractJson:

    def test_clean_json(self):
        raw = '{"key": "value", "num": 42}'
        result = extract_json(raw)
        assert result == {"key": "value", "num": 42}

    def test_json_with_markdown_fences(self):
        raw = '```json\n{"key": "value"}\n```'
        result = extract_json(raw)
        assert result == {"key": "value"}

    def test_json_with_plain_fences(self):
        raw = '```\n{"key": "value"}\n```'
        result = extract_json(raw)
        assert result == {"key": "value"}

    def test_json_with_preamble(self):
        raw = 'Here is the plan you requested:\n\n{"task_type": "python_project"}'
        result = extract_json(raw)
        assert result == {"task_type": "python_project"}

    def test_json_with_trailing_text(self):
        raw = '{"status": "ok"}\n\nLet me know if you need changes!'
        result = extract_json(raw)
        assert result == {"status": "ok"}

    def test_nested_json(self):
        raw = '{"files": {"main.py": "print(1)", "test.py": "assert True"}}'
        result = extract_json(raw)
        assert result["files"]["main.py"] == "print(1)"

    def test_empty_string_returns_none(self):
        assert extract_json("") is None

    def test_whitespace_only_returns_none(self):
        assert extract_json("   \n  ") is None

    def test_invalid_json_returns_none(self):
        assert extract_json("this is not json at all") is None

    def test_truncated_json_returns_none(self):
        assert extract_json('{"key": "val') is None

    def test_json_with_escaped_quotes_in_values(self):
        raw = '{"code": "print(\\"hello\\")"}'
        result = extract_json(raw)
        assert result["code"] == 'print("hello")'


# ── validate_plan ─────────────────────────────────────────────────────────────

class TestValidatePlan:

    def _valid_plan(self) -> dict:
        return {
            "task_type": "python_project",
            "description": "A calculator",
            "steps": ["step1"],
            "files_to_generate": ["main.py", "requirements.txt"],
            "entry_point": "main.py",
            "test_command": "pytest tests/",
            "install_command": "pip install -r requirements.txt",
        }

    def test_valid_plan_passes(self):
        assert validate_plan(self._valid_plan()) is True

    def test_missing_entry_point_fails(self):
        plan = self._valid_plan()
        del plan["entry_point"]
        assert validate_plan(plan) is False

    def test_missing_test_command_fails(self):
        plan = self._valid_plan()
        del plan["test_command"]
        assert validate_plan(plan) is False

    def test_empty_files_list_fails(self):
        plan = self._valid_plan()
        plan["files_to_generate"] = []
        assert validate_plan(plan) is False

    def test_files_not_list_fails(self):
        plan = self._valid_plan()
        plan["files_to_generate"] = "main.py"
        assert validate_plan(plan) is False


# ── validate_generation ───────────────────────────────────────────────────────

class TestValidateGeneration:

    def test_valid_generation_passes(self):
        data = {"files": {"main.py": "print('hello')", "requirements.txt": "pytest\n"}}
        assert validate_generation(data) is True

    def test_missing_files_key_fails(self):
        assert validate_generation({"code": "print()"}) is False

    def test_files_not_dict_fails(self):
        assert validate_generation({"files": ["main.py"]}) is False

    def test_empty_files_dict_fails(self):
        assert validate_generation({"files": {}}) is False

    def test_empty_file_content_fails(self):
        data = {"files": {"main.py": ""}}
        assert validate_generation(data) is False

    def test_whitespace_only_content_fails(self):
        data = {"files": {"main.py": "   \n  "}}
        assert validate_generation(data) is False


# ── validate_fix ──────────────────────────────────────────────────────────────

class TestValidateFix:

    def test_valid_fix_passes(self):
        data = {"fixed_files": {"main.py": "def fixed(): return 42"}, "fix_explanation": "Fixed"}
        assert validate_fix(data) is True

    def test_empty_fixed_files_passes(self):
        # Empty is valid (AI says no changes needed) — warn but don't fail
        data = {"fixed_files": {}}
        assert validate_fix(data) is True

    def test_missing_fixed_files_fails(self):
        assert validate_fix({"explanation": "changed stuff"}) is False

    def test_fixed_files_not_dict_fails(self):
        assert validate_fix({"fixed_files": ["main.py"]}) is False
