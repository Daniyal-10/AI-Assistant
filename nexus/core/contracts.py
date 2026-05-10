"""
NEXUS Core Contracts
────────────────────
This module defines all typed data contracts that flow between pipeline stages.
These contracts ensure type safety and consistent data structures across the engine.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from nexus.core.exceptions import ConfigurationError


# ── LLM Provider Contracts ────────────────────────────────────────────────────

@dataclass(frozen=True)
class LLMRequest:
    system_prompt: str
    user_prompt: str
    model_hint: str          # "code" | "reason" — provider resolves to actual model
    temperature: float = 0.1
    max_tokens: int = 4096
    request_id: str = ""     # for correlation logging


@dataclass(frozen=True)
class LLMResponse:
    content: str
    provider_name: str
    model_used: str
    tokens_used: int = 0
    was_fallback: bool = False
    request_id: str = ""


# ── Pipeline Stage Contracts ──────────────────────────────────────────────────

@dataclass(frozen=True)
class TaskPlan:
    task_type: str
    description: str
    files_to_generate: List[str]
    entry_point: str
    test_command: str
    install_command: str
    steps: List[Any] = field(default_factory=list)
    raw_dict: Dict[str, Any] = field(default_factory=dict)  # preserve original AI output

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        # Compatibility for engine mutation
        object.__setattr__(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass(frozen=True)
class GeneratedFiles:
    files: Dict[str, str]            # filename -> content
    plan_ref: str                    # task_id this generation belongs to


@dataclass
class ExecutionOutput:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    security_blocked: bool = False
    command_used: str = ""
    duration_seconds: float = 0.0

    @property
    def success(self) -> bool:
        return (
            self.returncode == 0
            and not self.timed_out
            and not self.security_blocked
        )


@dataclass(frozen=True)
class ValidationOutcome:
    stage1_passed: bool
    stage1_exit_code: int
    semantic_verdict: str            # "CORRECT" | "INCORRECT" | "UNCERTAIN" | "SKIPPED"
    semantic_reason: str = ""
    semantic_issues: List[str] = field(default_factory=list)
    is_config_error: bool = False    # pytest exit 4/5 — path problem not code problem

    @property
    def is_success(self) -> bool:
        if not self.stage1_passed:
            return False
        return self.semantic_verdict != "INCORRECT"

    @property
    def is_terminal_failure(self) -> bool:
        # Failures the AI fix loop CANNOT fix
        return self.is_config_error


@dataclass
class FixResult:
    fixed_files: Dict[str, str]
    fix_explanation: str
    provider_used: str
    iteration: int


# ── Task Lifecycle Contract ───────────────────────────────────────────────────

@dataclass
class TaskContext:
    task_id: str
    raw_input: str
    intent: str
    session_id: str
    plan: Optional[TaskPlan] = None
    generated: Optional[GeneratedFiles] = None
    last_execution: Optional[ExecutionOutput] = None
    last_validation: Optional[ValidationOutcome] = None
    fix_iteration: int = 0
    attempt_history: List[Any] = field(default_factory=list)
    workspace_path: str = ""
    venv_path: str = ""
    created_at: str = ""


# ── Provider Health Contract ──────────────────────────────────────────────────

@dataclass(frozen=True)
class ProviderHealth:
    provider_name: str
    is_available: bool
    latency_ms: float = 0.0
    error_message: str = ""
    checked_at: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def plan_from_dict(d: Dict[str, Any]) -> TaskPlan:
    """Safely constructs a TaskPlan from a raw AI response dict."""
    required = ["task_type", "files_to_generate", "entry_point", "test_command", "install_command"]
    for field_name in required:
        if field_name not in d:
            raise ConfigurationError(
                f"Missing required field in task plan: {field_name}",
                field_name=field_name,
                expected="present in dict"
            )

    return TaskPlan(
        task_type=d["task_type"],
        description=d.get("description", ""),
        files_to_generate=d["files_to_generate"],
        entry_point=d["entry_point"],
        test_command=d["test_command"],
        install_command=d["install_command"],
        steps=d.get("steps", []),
        raw_dict=d
    )


__all__ = [
    "LLMRequest",
    "LLMResponse",
    "TaskPlan",
    "GeneratedFiles",
    "ExecutionOutput",
    "ValidationOutcome",
    "FixResult",
    "TaskContext",
    "ProviderHealth",
    "plan_from_dict",
]
