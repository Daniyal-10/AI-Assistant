"""
tests/test_history.py
─────────────────────
Unit tests for the TaskHistory store.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from nexus.utils.history import TaskHistory, TaskRecord


def test_history_record_creation_and_truncation(tmp_path):
    test_workspaces = tmp_path / "workspaces"
    test_workspaces.mkdir()

    with patch("nexus.utils.config.config.workspace_base", str(test_workspaces)):
        store = TaskHistory()
        
        # Mock a Task with a very long summary
        task = MagicMock()
        task.raw_input = "Original input"
        task.status.name = "DONE"
        task.intent.intent.value = "TASK"
        task.plan = {"description": "Plan desc"}
        task.result.summary = "A" * 1000 # Exceeds 500 char limit
        task.result.semantic_verdict = "CORRECT"
        task.fix_iteration = 2

        store.record(task, "session-123")
        
        history_file = store._get_current_file()
        assert history_file.exists()
        
        with open(history_file, "r") as f:
            data = json.loads(f.readline())
            
        assert data["session_id"] == "session-123"
        assert len(data["final_output_summary"]) == 500
        assert data["fix_attempts"] == 2
        assert data["execution_status"] == "DONE"


def test_history_rotation_logic(tmp_path):
    test_workspaces = tmp_path / "workspaces"
    test_workspaces.mkdir()

    with patch("nexus.utils.config.config.workspace_base", str(test_workspaces)):
        store = TaskHistory()

        with patch("nexus.utils.history.datetime") as mock_dt:
            # Set time to May 2026
            mock_dt.utcnow.return_value = MagicMock(year=2026, month=5)
            fname = store._get_current_file().name
            assert "tasks_2026_05.jsonl" == fname

            # Change time to June 2026
            mock_dt.utcnow.return_value = MagicMock(year=2026, month=6)
            fname2 = store._get_current_file().name
            assert "tasks_2026_06.jsonl" == fname2


def test_history_query_filters(tmp_path):
    test_workspaces = tmp_path / "workspaces"
    test_workspaces.mkdir()

    with patch("nexus.utils.config.config.workspace_base", str(test_workspaces)):
        store = TaskHistory()
        history_file = store._get_current_file()

        # Seed data
        records = [
            {"raw_input": "Find bugs", "intent": "CODE", "execution_status": "DONE"},
            {"raw_input": "Delete logs", "intent": "TASK", "execution_status": "FAILED"},
            {"raw_input": "How are you?", "intent": "CHAT", "execution_status": "DONE"},
        ]
        
        with open(history_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        # Test filters
        assert len(store.get_by_status("DONE")) == 2
        assert len(store.get_by_intent("TASK")) == 1
        assert len(store.search_by_keyword("bugs")) == 1
        assert len(store.get_recent(5)) == 3
