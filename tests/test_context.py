"""
tests/test_context.py
─────────────────────
Unit tests for the Session Context Manager.
"""
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from nexus.core.context import SessionContext
from nexus.utils.config import config


def test_context_history_bounds():
    """Verify that conversation and task histories are strictly bounded."""
    ctx = SessionContext()

    # Fill conversation history beyond limit (20)
    for i in range(25):
        ctx.add_message("user", f"message {i}")
    
    assert len(ctx.conversation_history) == 20
    assert ctx.conversation_history[0]["content"] == "message 5"
    assert ctx.conversation_history[-1]["content"] == "message 24"

    # Fill task history beyond limit (10)
    for i in range(15):
        ctx.add_task_result(f"id-{i}", "summary", "TASK", "DONE")
    
    assert len(ctx.task_history) == 10
    assert ctx.task_history[0]["task_id"] == "id-5"
    assert ctx.task_history[-1]["task_id"] == "id-14"


def test_context_summary_generation():
    """Verify the formatted summary used for prompt injection."""
    ctx = SessionContext()
    ctx.set_project("/workspace/project-a")
    ctx.add_message("user", "Hello")
    ctx.add_message("assistant", "Hi! I am NEXUS.")
    ctx.add_message("user", "Build a scraper")
    ctx.add_task_result("abc123", "Scraper built successfully", "TASK", "DONE")

    summary = ctx.get_recent_context()
    
    assert "Active Project Path: /workspace/project-a" in summary
    assert "abc123 [TASK]: DONE | Scraper built successfully" in summary
    # Messages skip the last one (current prompt) in my implementation
    assert "USER: Hello" in summary
    assert "ASSISTANT: Hi! I am NEXUS." in summary


def test_context_atomic_serialization(tmp_path):
    """Verify atomic JSON serialization to the sessions directory."""
    # Mock workspace base to use a temp directory
    test_workspace = tmp_path / "workspaces"
    test_workspace.mkdir()
    
    with patch("nexus.utils.config.config.workspace_base", str(test_workspace)):
        ctx = SessionContext(session_id="test-session")
        ctx.add_message("user", "persist this")
        ctx.save()

        sessions_dir = tmp_path / "sessions"
        session_file = sessions_dir / "session_test-session.json"
        
        assert session_file.exists()
        
        with open(session_file, "r") as f:
            data = json.load(f)
            assert data["session_id"] == "test-session"
            assert len(data["conversation_history"]) == 1
            assert data["conversation_history"][0]["content"] == "persist this"


def test_context_save_failure_handling(tmp_path):
    """Verify that save failure doesn't crash and cleans up temp files."""
    test_workspace = tmp_path / "workspaces"
    test_workspace.mkdir()

    with patch("nexus.utils.config.config.workspace_base", str(test_workspace)):
        ctx = SessionContext(session_id="fail-test")
        
        # Mock open to raise error
        with patch("builtins.open", side_effect=IOError("Permission denied")):
            ctx.save() # Should not raise
            
        sessions_dir = tmp_path / "sessions"
        temp_file = sessions_dir / "session_fail-test.tmp"
        assert not temp_file.exists()
