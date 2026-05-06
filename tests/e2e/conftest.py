"""
tests/e2e/conftest.py
─────────────────────
Shared fixtures for E2E pipeline testing.
"""
import os
import shutil
import pytest
import json
from unittest.mock import MagicMock, patch
from nexus.core.engine import TaskEngine
from nexus.core.context import SessionContext


@pytest.fixture
def safe_workspace(tmp_path):
    """Override workspace_base to a temporary directory to avoid polluting .nexus/."""
    # Resolve to absolute path to ensure boundary checks pass
    ws_root = tmp_path.resolve()
    with patch("nexus.utils.config.config.workspace_base", str(ws_root)):
        yield ws_root


@pytest.fixture
def nexus_engine(safe_workspace):
    """
    Provides a real TaskEngine instance with a fresh context.
    The workspace is isolated to a temporary directory.
    """
    ctx = SessionContext()
    engine = TaskEngine(ctx)
    
    # Ensure any background logging or history does not crash
    # due to the temporary workspace setup.
    yield engine
    
    # Cleanup: TaskEngine creates subdirectories in workspace_base
    if os.path.exists(str(safe_workspace)):
        shutil.rmtree(str(safe_workspace), ignore_errors=True)


@pytest.fixture
def mock_llm_responses():
    """
    Returns a factory function that mocks sequential LLM responses.
    Usage:
        def test_x(nexus_engine, mock_llm_responses):
            mock_llm_responses(nexus_engine, ["resp1", "resp2"])
    """
    def _setup_mock(engine, responses: list):
        engine.ai._call_ollama = MagicMock(side_effect=responses)
    return _setup_mock
