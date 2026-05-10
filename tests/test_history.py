"""
tests/test_history.py
─────────────────────
Unit tests for the TaskHistory store.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from nexus.utils.config import test_config as nexus_test_config
from nexus.utils.history import TaskHistory, TaskRecord


def test_history_record_creation_and_truncation(tmp_path):
    ws = tmp_path / "workspaces"
    ws.mkdir(exist_ok=True)

    with nexus_test_config(workspace_base=str(ws)):
        store = TaskHistory()
        
        task = MagicMock()
        task.raw_input = "Original input"
        task.status.name = "DONE"
        task.intent.intent.value = "TASK"
        task.plan = {"description": "Plan desc"}
        task.result.summary = "A" * 1000
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
    ws = tmp_path / "workspaces"
    ws.mkdir(exist_ok=True)

    with nexus_test_config(workspace_base=str(ws)):
        store = TaskHistory()

        with patch("nexus.utils.history.datetime") as mock_dt:
            mock_dt.utcnow.return_value = MagicMock(year=2026, month=5)
            fname = store._get_current_file().name
            assert "tasks_2026_05.jsonl" == fname

            mock_dt.utcnow.return_value = MagicMock(year=2026, month=6)
            fname2 = store._get_current_file().name
            assert "tasks_2026_06.jsonl" == fname2


def test_history_query_filters(tmp_path):
    ws = tmp_path / "workspaces"
    ws.mkdir(exist_ok=True)

    with nexus_test_config(workspace_base=str(ws)):
        store = TaskHistory()
        history_file = store._get_current_file()

        records = [
            {"raw_input": "Find bugs", "intent": "CODE", "execution_status": "DONE"},
            {"raw_input": "Delete logs", "intent": "TASK", "execution_status": "FAILED"},
            {"raw_input": "How are you?", "intent": "CHAT", "execution_status": "DONE"},
        ]
        
        with open(history_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        assert len(store.get_by_status("DONE")) == 2
        assert len(store.get_by_intent("TASK")) == 1
        assert len(store.search_by_keyword("bugs")) == 1
        assert len(store.get_recent(5)) == 3
