"""
Configuration loader. All settings come from environment variables.
Never hardcode values here.
"""
import os
import sys
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class NexusConfig:
    # Ollama settings
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    )
    ollama_code_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_CODE_MODEL", "qwen2.5-coder:7b")
    )
    ollama_reason_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_REASON_MODEL", "llama3.1:8b")
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

    if errors:
        print("\n❌ NEXUS configuration errors:")
        for err in errors:
            print(f"   • {err}")
        print("\nFix your .env file or environment variables and restart.\n")
        sys.exit(1)


# Singleton — import this everywhere
config = NexusConfig()
_validate_config(config)
