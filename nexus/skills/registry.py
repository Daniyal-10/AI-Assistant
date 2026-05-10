"""
nexus/skills/registry.py
────────────────────────
Skill Registry — maps task patterns to known-good code scaffolds.

Instead of the LLM generating everything from scratch, the registry
provides a validated starting point. The LLM then fills in only the
task-specific logic on top of the scaffold.

Benefits:
- Reduces hallucination surface area
- Guarantees pytest is always present
- Guarantees correct file structure
- Guarantees network calls are mocked in tests
- Faster generation (LLM does less work)

Phase 1: keyword-based matching (fast, deterministic)
Phase 2: embedding-based matching (semantic, added later)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from nexus.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Skill:
    """
    A reusable code scaffold for a common task pattern.
    Files are templates — {TASK_DESCRIPTION} is replaced at runtime.
    """
    name:        str
    description: str
    keywords:    List[str]
    files:       Dict[str, str]   # filename -> template content
    test_command: str
    install_command: str = ""
    notes:       str = ""


# ── Skill definitions ─────────────────────────────────────────────────────────

_SKILLS: List[Skill] = [

    Skill(
        name        = "python_script",
        description = "Simple Python script with pytest tests",
        keywords    = ["script", "print", "calculate", "compute", "convert", "parse"],
        test_command = "pytest test_main.py -v",
        files = {
            "main.py": '''\
"""
{TASK_DESCRIPTION}
"""


def main():
    # TODO: implement task logic here
    pass


if __name__ == "__main__":
    main()
''',
            "test_main.py": '''\
"""Tests for main.py"""
import pytest
from main import main


def test_main_runs():
    """Verify main() runs without errors."""
    result = main()
    # TODO: add assertions based on expected output
    assert result is not None or result is None  # replace with real assertion
''',
        },
    ),

    Skill(
        name        = "data_processor",
        description = "CSV/JSON data processing with stdlib only",
        keywords    = ["csv", "data", "process", "parse", "read", "analyse", "analyze", "count", "filter", "sort"],
        test_command = "pytest test_processor.py -v",
        files = {
            "processor.py": '''\
"""
{TASK_DESCRIPTION}
"""
import csv
import json
from pathlib import Path
from typing import List, Dict, Any


def load_csv(filepath: str) -> List[Dict[str, str]]:
    """Load a CSV file and return list of row dicts."""
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def process(data: List[Dict[str, Any]]) -> Any:
    """Main processing logic — implement task here."""
    # TODO: implement task-specific logic
    return data


def main():
    data = load_csv("input.csv")
    result = process(data)
    print(f"Processed {len(data)} records")
    return result


if __name__ == "__main__":
    main()
''',
            "input.csv": '''\
id,name,value
1,item_a,100
2,item_b,200
3,item_c,300
''',
            "test_processor.py": '''\
"""Tests for processor.py"""
import pytest
from processor import load_csv, process


def test_load_csv(tmp_path):
    """Verify CSV loading works correctly."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("id,name,value\\n1,test,100\\n")
    data = load_csv(str(csv_file))
    assert len(data) == 1
    assert data[0]["name"] == "test"


def test_process_returns_data():
    """Verify process() returns something."""
    sample = [{"id": "1", "name": "test", "value": "100"}]
    result = process(sample)
    assert result is not None
''',
        },
    ),

    Skill(
        name        = "api_client",
        description = "HTTP API client with mocked tests (no real network calls)",
        keywords    = ["api", "fetch", "http", "url", "endpoint", "request", "download", "price", "weather", "stock"],
        test_command = "pytest test_client.py -v",
        install_command = "",
        files = {
            "client.py": '''\
"""
{TASK_DESCRIPTION}
Uses urllib.request (stdlib) — no third-party dependencies.
"""
import json
import urllib.request
from typing import Any, Dict, Optional


def fetch(url: str, timeout: int = 10) -> Dict[str, Any]:
    """Fetch JSON data from a URL."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        raw = response.read()
        return json.loads(raw)


def main():
    # TODO: replace with actual API URL and logic
    url = "https://api.example.com/data"
    data = fetch(url)
    print(f"Fetched: {data}")
    return data


if __name__ == "__main__":
    main()
''',
            "test_client.py": '''\
"""Tests for client.py — all network calls are mocked."""
import json
import pytest
from unittest.mock import patch, MagicMock
from client import fetch, main


def _mock_response(data: dict):
    """Helper to create a mock urllib response."""
    mock = MagicMock()
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    mock.read.return_value = json.dumps(data).encode()
    return mock


def test_fetch_returns_data():
    """Verify fetch() parses JSON correctly."""
    expected = {"price": 50000, "currency": "USD"}
    with patch("urllib.request.urlopen", return_value=_mock_response(expected)):
        result = fetch("https://api.example.com/data")
    assert result == expected


def test_fetch_handles_response():
    """Verify fetch() returns a dict."""
    with patch("urllib.request.urlopen", return_value=_mock_response({"key": "value"})):
        result = fetch("https://api.example.com/test")
    assert isinstance(result, dict)
    assert "key" in result
''',
        },
    ),

    Skill(
        name        = "file_utility",
        description = "File system utility — read, write, organize files",
        keywords    = ["file", "folder", "directory", "rename", "move", "copy", "delete", "organize", "list", "find"],
        test_command = "pytest test_utility.py -v",
        files = {
            "utility.py": '''\
"""
{TASK_DESCRIPTION}
"""
import os
import shutil
from pathlib import Path
from typing import List


def list_files(directory: str, extension: str = "") -> List[str]:
    """List all files in a directory, optionally filtered by extension."""
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if extension:
        return [str(f) for f in path.rglob(f"*{extension}")]
    return [str(f) for f in path.rglob("*") if f.is_file()]


def main():
    # TODO: implement task-specific file operation
    files = list_files(".")
    print(f"Found {len(files)} files")
    return files


if __name__ == "__main__":
    main()
''',
            "test_utility.py": '''\
"""Tests for utility.py"""
import pytest
from pathlib import Path
from utility import list_files


def test_list_files(tmp_path):
    """Verify list_files returns correct files."""
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("world")
    (tmp_path / "c.py").write_text("pass")

    all_files = list_files(str(tmp_path))
    assert len(all_files) == 3

    txt_files = list_files(str(tmp_path), extension=".txt")
    assert len(txt_files) == 2


def test_list_files_missing_dir():
    """Verify FileNotFoundError on missing directory."""
    with pytest.raises(FileNotFoundError):
        list_files("/nonexistent/path")
''',
        },
    ),

]


# ── Registry class ────────────────────────────────────────────────────────────

class SkillRegistry:
    """
    Matches a task to the most relevant skill scaffold.
    Phase 1: keyword-based scoring.
    """

    def __init__(self) -> None:
        self._skills = _SKILLS

    def match(
        self,
        user_input: str,
        task_type: str = "",
    ) -> Optional[Skill]:
        """
        Find the best matching skill for a given task.

        Args:
            user_input: The user's task description
            task_type:  The task_type from the LLM plan (optional)

        Returns:
            The best matching Skill, or None if no good match found.
        """
        text = (user_input + " " + task_type).lower()
        best_skill = None
        best_score = 0

        for skill in self._skills:
            score = self._score(text, skill)
            if score > best_score:
                best_score = score
                best_skill = skill

        if best_skill and best_score >= 1:
            logger.info(
                "Skill matched: '%s' (score=%d) for input: %s",
                best_skill.name,
                best_score,
                user_input[:60],
            )
            return best_skill

        logger.debug(
            "No skill matched (best_score=%d) for: %s",
            best_score,
            user_input[:60],
        )
        return None

    def _score(self, text: str, skill: Skill) -> int:
        """Count how many of the skill's keywords appear in the text."""
        return sum(1 for kw in skill.keywords if kw in text)

    def get_scaffold(
        self,
        skill: Skill,
        task_description: str,
    ) -> Dict[str, str]:
        """
        Return the skill's file templates with {TASK_DESCRIPTION} filled in.

        Args:
            skill:            The matched skill
            task_description: The user's task description

        Returns:
            Dict of filename -> rendered content
        """
        rendered = {}
        for filename, template in skill.files.items():
            rendered[filename] = template.replace(
                "{TASK_DESCRIPTION}", task_description
            )
        return rendered

    def list_skills(self) -> List[str]:
        """Return names of all registered skills."""
        return [s.name for s in self._skills]
