"""
tests/test_history.py
─────────────────────
Unit tests for persistent Task History Store.
"""
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from nexus.utils.history import TaskHistoryStore, TaskRecord


def test_history_record_creation_and_truncation(tmp_path):
    """Verify that records are appended to JSONL and summaries are truncated."""
    test_workspaces = tmp_path / "workspaces"
    test_workspaces.mkdir()

    with patch("nexus.utils.config.config.workspace_base", str(test_workspaces)):
        store = TaskHistoryStore()
        
        # Mock a Task with a very long summary
        task = MagicMock()
        task.raw_input = "Create a large scale system"
        task.status.name = "DONE"
        task.result.summary = "Correct! " * 100 # > 500 chars
        task.result.semantic_verdict = "CORRECT"
        task.plan = {"description": "Complex architectural plan"}
        task.fix_iteration = 1
        task.intent.intent.value = "TASK"

        store.record(task, "session-abc-123")

        history_file = store._get_current_file()
        assert history_file.exists()

        with open(history_file, "r", encoding="utf-8") as f:
            line = f.readline()
            data = json.loads(line)
            
            assert data["session_id"] == "session-abc-123"
            assert data["raw_input"] == "Create a large scale system"
            assert data["execution_status"] == "DONE"
            assert data["schema_version"] == "1.0"
            # Ensure hard truncation at 500
            assert len(data["final_output_summary"]) == 500
            assert data["final_output_summary"].startswith("Correct!")


def test_history_rotation_logic(tmp_path):
    """Verify that history files rotate monthly."""
    test_workspaces = tmp_path / "workspaces"
    test_workspaces.mkdir()

    with patch("nexus.utils.config.config.workspace_base", str(test_workspaces)):
        store = TaskHistoryStore()

        with patch("nexus.utils.history.datetime") as mock_dt:
            # Set time to May 2026
            mock_dt.utcnow.return_value = datetime(2026, 5, 15)
            f1 = store._get_current_file()
            assert "tasks_2026_05.jsonl" in str(f1)

            # Set time to June 2026
            mock_dt.utcnow.return_value = datetime(2026, 6, 1)
            f2 = store._get_current_file()
            assert "tasks_2026_06.jsonl" in str(f2)
            assert f1 != f2


def test_history_queries(tmp_path):
    """Verify query and search capabilities."""
    test_workspaces = tmp_path / "workspaces"
    test_workspaces.mkdir()

    with patch("nexus.utils.config.config.workspace_base", str(test_workspaces)):
        store = TaskHistoryStore()
        history_file = store._get_current_file()

        # Seed data
        records = [
            TaskRecord(raw_input="hello", intent="CHAT", execution_status="DONE"),
            TaskRecord(raw_input="build app", intent="TASK", execution_status="DONE"),
            TaskRecord(raw_input="fix error", intent="TASK", execution_status="FAILED"),
        ]
        
        with open(history_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(asdict(r)) + "\n")

        # Test get_recent (should be reverse order)
        recent = store.get_recent(2)
        assert len(recent) == 2
        assert recent[0].raw_input == "fix error"
        assert recent[1].raw_input == "build app"

        # Test filter by status
        failed = store.get_by_status("FAILED")
        assert len(failed) == 1
        assert failed[0].raw_input == "fix error"

        # Test keyword search (case-insensitive)
        search = store.search_by_keyword("BUILD")
        assert len(search) == 1
        assert search[0].raw_input == "build app"


def test_history_graceful_failure(tmp_path):
    """Verify that history write failures do not crash the recording process."""
    test_workspaces = tmp_path / "workspaces"
    test_workspaces.mkdir()

    with patch("nexus.utils.config.config.workspace_base", str(test_workspaces)):
        store = TaskHistoryStore()
        
        # Mock open to trigger an IO error
        with patch("builtins.open", side_effect=IOError("Disk Full")):
            # This should log error but NOT raise exception
            store.record(MagicMock(), "session-fail")
            
        # If we reach here without exception, test passes
        assert True
