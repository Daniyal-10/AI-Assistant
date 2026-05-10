"""
Configuration loader. All settings come from environment variables.
Never hardcode values here.
"""
import os
import sys
import threading
import contextlib
import dataclasses
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=False)
class NexusConfig:
    # Ollama settings
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    )
    ollama_code_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_CODE_MODEL", "qwen2.5-coder:7b")
    )
    ollama_reason_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_REASON_MODEL", "qwen2.5:7b")
    )
    ollama_timeout: int = field(
        default_factory=lambda: int(os.getenv("OLLAMA_TIMEOUT", "120"))
    )

    # Execution settings
    exec_timeout: int = field(
        default_factory=lambda: int(os.getenv("EXEC_TIMEOUT", "30"))
    )
    max_fix_iterations: int = field(
        default_factory=lambda: int(os.getenv("MAX_FIX_ITERATIONS", "3"))
    )
    workspace_base: str = field(
        default_factory=lambda: os.getenv(
            "WORKSPACE_BASE",
            os.path.join(os.path.expanduser("~"), ".nexus", "workspaces")
        )
    )
    executor_type: str = field(
        default_factory=lambda: os.getenv("NEXUS_EXECUTOR", "local").lower()
    )

    # Context management
    nexus_context_token_budget: int = field(
        default_factory=lambda: int(os.getenv("NEXUS_CONTEXT_TOKEN_BUDGET", "3000"))
    )
    nexus_task_history_limit: int = field(
        default_factory=lambda: int(os.getenv("NEXUS_TASK_HISTORY_LIMIT", "10"))
    )
    nexus_conversation_history_limit: int = field(
        default_factory=lambda: int(os.getenv("NEXUS_CONVERSATION_HISTORY_LIMIT", "20"))
    )

    # Security
    allowed_telegram_users: list = field(
        default_factory=lambda: [
            int(uid)
            for uid in os.getenv("ALLOWED_TELEGRAM_USERS", "").split(",")
            if uid.strip()
        ]
    )
    telegram_bot_token: str = field(
        default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", "")
    )

    # Anthropic Fallback settings
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    fallback_enabled: bool = field(
        default_factory=lambda: os.getenv("FALLBACK_ENABLED", "false").lower() == "true"
    )
    fallback_model: str = field(
        default_factory=lambda: os.getenv("FALLBACK_MODEL", "claude-haiku-4-5-20251001")
    )


def _validate_config(cfg: NexusConfig) -> None:
    """
    Validate config at startup. Fail fast with a clear message
    rather than a confusing error deep inside execution.
    """
    errors = []

    # Timeouts must be positive integers
    if cfg.exec_timeout <= 0:
        errors.append(f"EXEC_TIMEOUT must be > 0, got {cfg.exec_timeout}")

    if cfg.ollama_timeout <= 0:
        errors.append(f"OLLAMA_TIMEOUT must be > 0, got {cfg.ollama_timeout}")

    # Fix iterations must be sensible
    if not (1 <= cfg.max_fix_iterations <= 10):
        errors.append(
            f"MAX_FIX_ITERATIONS must be between 1 and 10, got {cfg.max_fix_iterations}"
        )

    # Workspace must be absolute path
    if not os.path.isabs(cfg.workspace_base):
        errors.append(
            f"WORKSPACE_BASE must be an absolute path, got '{cfg.workspace_base}'"
        )

    # Workspace must be writable (create it if it doesn't exist yet)
    try:
        os.makedirs(cfg.workspace_base, exist_ok=True)
        if not os.access(cfg.workspace_base, os.W_OK):
            errors.append(f"WORKSPACE_BASE is not writable: {cfg.workspace_base}")
    except OSError as e:
        errors.append(f"WORKSPACE_BASE could not be created: {e}")

    # Ollama URL must look like a URL
    if not cfg.ollama_base_url.startswith(("http://", "https://")):
        errors.append(
            f"OLLAMA_BASE_URL must start with http:// or https://, "
            f"got '{cfg.ollama_base_url}'"
        )

    # Anthropic Fallback Validation
    if cfg.fallback_enabled and not cfg.anthropic_api_key:
        errors.append("ANTHROPIC_API_KEY must be set if FALLBACK_ENABLED is True")

    if errors:
        print("\n❌ NEXUS configuration errors:")
        for err in errors:
            print(f"   • {err}")
        print("\nFix your .env file or environment variables and restart.\n")
        sys.exit(1)


# ── Config registry ───────────────────────────────────────────────────────────

_config_instance: Optional[NexusConfig] = None
_config_lock = threading.Lock()


def get_config() -> NexusConfig:
    """
    Return the global NexusConfig singleton, creating and validating it 
    on first call. Thread-safe. Safe to call at import time from any module.
    
    This replaces the old module-level `config` singleton.
    Import pattern:
        from nexus.utils.config import get_config
        cfg = get_config()
    """
    global _config_instance
    if _config_instance is not None:
        return _config_instance
    with _config_lock:
        if _config_instance is None:
            instance = NexusConfig()
            _validate_config(instance)
            _config_instance = instance
    return _config_instance


def override_config(cfg: NexusConfig) -> None:
    """
    Replace the global config instance. 
    USE ONLY IN TESTS via the test_config() context manager.
    Never call this directly in production code.
    """
    global _config_instance
    with _config_lock:
        _config_instance = cfg


def reset_config() -> None:
    """
    Clear the cached config instance, forcing re-creation on next get_config() call.
    USE ONLY IN TESTS.
    """
    global _config_instance
    with _config_lock:
        _config_instance = None


@contextlib.contextmanager
def test_config(**overrides):
    """
    Context manager for test isolation. Creates a NexusConfig with 
    provided overrides, bypasses validation, and restores original 
    config on exit.
    
    Usage in tests:
        from nexus.utils.config import test_config
        
        with test_config(workspace_base="/tmp/test", exec_timeout=5):
            engine = TaskEngine()
            # engine uses the test config
        # original config restored here
    
    Args:
        **overrides: NexusConfig field names and their test values.
    """
    original = _config_instance
    try:
        test_cfg = NexusConfig()
        for field_name, value in overrides.items():
            if not hasattr(test_cfg, field_name):
                raise ValueError(
                    f"test_config: '{field_name}' is not a valid NexusConfig field. "
                    f"Valid fields: {[f.name for f in dataclasses.fields(test_cfg)]}"
                )
            object.__setattr__(test_cfg, field_name, value)
        override_config(test_cfg)
        yield test_cfg
    finally:
        override_config(original) if original is not None else reset_config()


# ── Backward compatibility ────────────────────────────────────────────────────
# The old pattern was: from nexus.utils.config import config
# New pattern is:      from nexus.utils.config import get_config; cfg = get_config()
# This alias allows existing code to keep working during the migration.
# It is evaluated lazily — accessing `config` triggers get_config() on first use.

class _ConfigProxy:
    """
    Lazy proxy for the old `config` module attribute.
    Accessing any attribute on this proxy calls get_config() first.
    This means the old pattern `from nexus.utils.config import config`
    still works and gets the real validated config, but does NOT 
    run validation at import time.
    """
    def __getattr__(self, name: str):
        return getattr(get_config(), name)
    
    def __setattr__(self, name: str, value):
        # Allow setting on the underlying instance for monkeypatching in tests
        cfg = get_config()
        object.__setattr__(cfg, name, value)
    
    def __repr__(self):
        return f"ConfigProxy({get_config()!r})"

config = _ConfigProxy()
