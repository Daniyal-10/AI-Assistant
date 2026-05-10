"""
tests/conftest.py
─────────────────
Root-level pytest configuration for NEXUS test suite.

Provides:
- Automatic config isolation for every test (no real filesystem side effects)
- Shared fixtures available to all test modules
- Ensures no test ever runs against real workspace or real Ollama
"""
import os
import pytest
from pathlib import Path
from nexus.utils.config import test_config, reset_config, NexusConfig


@pytest.fixture(autouse=True)
def isolated_config(tmp_path):
    """
    Automatically applied to EVERY test in the suite.
    
    Replaces the global config with a test-safe version that:
    - Uses tmp_path as workspace_base (isolated per test)
    - Has short timeouts (fail fast in tests)
    - Has no Telegram token (no accidental bot calls)
    - Has fallback disabled (no accidental Anthropic API calls)
    
    This fixture means NO test needs to manually patch config.workspace_base.
    The old pattern:
        with patch("nexus.utils.config.config.workspace_base", str(tmp_path)):
    Is now replaced by simply using this fixture (auto-applied).
    """
    ws = tmp_path / "workspaces"
    ws.mkdir(parents=True, exist_ok=True)
    
    with test_config(
        workspace_base=str(ws),
        exec_timeout=10,
        ollama_timeout=30,
        max_fix_iterations=3,
        telegram_bot_token="",
        allowed_telegram_users=[],
        fallback_enabled=False,
        anthropic_api_key=None,
    ):
        yield


@pytest.fixture
def temp_workspace(tmp_path):
    """
    Provides a clean temporary workspace path for tests that 
    need to write files directly.
    """
    ws = tmp_path / "test_workspace"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


@pytest.fixture
def nexus_config(tmp_path):
    """
    Returns the active test NexusConfig instance.
    Use this when a test needs to inspect or further override config values.
    """
    from nexus.utils.config import get_config
    return get_config()
