"""
nexus/core/task.py
───────────────────
Task data model and state machine.
Single source of truth for a task's lifecycle.
No business logic here — pure data + state transitions.

State diagram:
  PENDING → PLANNING → GENERATING → EXECUTING → VALIDATING → DONE
                                                     ↓    ↑
                                                  FIXING ──┘
  Any state → FAILED
"""
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from nexus.core.contracts import (
    TaskPlan,
    GeneratedFiles,
    ExecutionOutput,
    ValidationOutcome,
    plan_from_dict,
)
from nexus.core.exceptions import ConfigurationError


class TaskStatus(Enum):
    PENDING    = auto()
    PLANNING   = auto()
    GENERATING = auto()
    EXECUTING  = auto()
    VALIDATING = auto()
    FIXING     = auto()
    DONE       = auto()
    FAILED     = auto()


# ── Allowed state transitions ─────────────────────────────────────────────────

_ALLOWED_TRANSITIONS: Dict[TaskStatus, List[TaskStatus]] = {
    TaskStatus.PENDING:    [TaskStatus.PLANNING,   TaskStatus.DONE, TaskStatus.FAILED],
    TaskStatus.PLANNING:   [TaskStatus.GENERATING, TaskStatus.FAILED],
    TaskStatus.GENERATING: [TaskStatus.EXECUTING,  TaskStatus.FAILED],
    TaskStatus.EXECUTING:  [TaskStatus.VALIDATING, TaskStatus.FAILED],
    TaskStatus.VALIDATING: [TaskStatus.DONE,       TaskStatus.FIXING,     TaskStatus.FAILED],
    TaskStatus.FIXING:     [TaskStatus.VALIDATING, TaskStatus.GENERATING, TaskStatus.FAILED],
}


@dataclass
class TaskResult:
    """Final output of a completed or failed task."""
    success: bool
    output_path: Optional[str] = None
    summary: str = ""
    iterations_used: int = 0
    errors_encountered: List[str] = field(default_factory=list)


@dataclass
class Task:
    """
    Immutable identity + mutable lifecycle state for one execution unit.
    Engine mutates state fields directly via the methods below.
    Do not add business logic here.
    """
    # Identity (set at creation, never mutated)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    raw_input: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Lifecycle state
    status: TaskStatus = TaskStatus.PENDING
    fix_iteration: int = 0
    intent: Optional[Any] = None

    # Planning artifacts
    _plan: Optional[TaskPlan] = None
    generated: Optional[GeneratedFiles] = None

    # Execution context
    workspace_path: Optional[str] = None
    venv_executables: Dict[str, str] = field(default_factory=dict)
    last_execution: Optional[ExecutionOutput] = None

    # Final result
    result: Optional[TaskResult] = None

    # ── Compatibility Properties ──────────────────────────────────────────────

    @property
    def plan(self) -> Optional[TaskPlan]:
        return self._plan

    @plan.setter
    def plan(self, value: Any) -> None:
        from nexus.planning.schema import EnrichedPlan
        if isinstance(value, EnrichedPlan):
            self._plan = value
            return
        if isinstance(value, dict):
            # Only convert if it looks like a full plan, otherwise keep as dict
            # to avoid breaking partial mocks in tests.
            required = ["task_type", "files_to_generate", "entry_point", "test_command", "install_command"]
            if all(k in value for k in required):
                try:
                    self._plan = plan_from_dict(value)
                except ConfigurationError:
                    self._plan = value
            else:
                # Store as dict for backward compatibility in tests
                self._plan = value
        else:
            self._plan = value

    @property
    def generated_files(self) -> Dict[str, str]:
        return self.generated.files if self.generated else {}

    @generated_files.setter
    def generated_files(self, value: Dict[str, str]) -> None:
        self.generated = GeneratedFiles(files=value, plan_ref=self.id)

    @property
    def last_stdout(self) -> str:
        return self.last_execution.stdout if self.last_execution else ""

    @property
    def last_stderr(self) -> str:
        return self.last_execution.stderr if self.last_execution else ""

    @property
    def last_error(self) -> str:
        return ""

    # ── State machine ─────────────────────────────────────────────────────────

    def transition(self, new_status: TaskStatus) -> None:
        """
        Validate and apply a state transition.

        Raises ValueError for invalid transitions so the engine always
        knows when something has gone wrong in the lifecycle — rather than
        silently ending up in a wrong state.

        FAILED is always reachable from any state (terminal error path).
        DONE is terminal — no transitions out.
        """
        # Terminal states: no transitions allowed out
        if self.status == TaskStatus.DONE:
            raise ValueError(
                f"Cannot transition from DONE → {new_status.name}: task is complete"
            )

        # FAILED is always allowed as an escape hatch
        if new_status == TaskStatus.FAILED:
            self.status = new_status
            return

        allowed = _ALLOWED_TRANSITIONS.get(self.status, [])
        if new_status not in allowed:
            raise ValueError(
                f"Invalid transition: {self.status.name} → {new_status.name}. "
                f"Allowed from {self.status.name}: "
                f"{[s.name for s in allowed]}"
            )

        self.status = new_status

    # ── Mutation helpers ──────────────────────────────────────────────────────

    def record_execution(self, output: ExecutionOutput) -> None:
        """Store the last execution output for use in fix prompts."""
        self.last_execution = output

    def record_execution_output(
        self, stdout: str, stderr: str, error: str = ""
    ) -> None:
        """
        [DEPRECATED] Store the last execution output.
        Use record_execution(ExecutionOutput(...)) instead.
        """
        warnings.warn(
            "record_execution_output is deprecated, use record_execution instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.record_execution(
            ExecutionOutput(
                returncode=1 if error else 0,
                stdout=stdout,
                stderr=stderr,
            )
        )

    def increment_fix_iteration(self) -> None:
        self.fix_iteration += 1

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Task(id={self.id}, status={self.status.name}, "
            f"fix_iter={self.fix_iteration}, input={self.raw_input[:50]!r})"
        )
