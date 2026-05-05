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
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional


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
#
# Design rules:
#   - Forward-only in the happy path (PENDING → ... → DONE)
#   - VALIDATING → FIXING: test failed, enter fix loop
#   - FIXING → VALIDATING: re-run tests after a fix attempt
#   - Any state → FAILED: always allowed (engine error handling)
#
# Bug that was here: FIXING → VALIDATING was missing, and VALIDATING → FIXING
# was also missing. The fix loop cycles VALIDATING ↔ FIXING repeatedly, so
# both directions must exist.
#
_ALLOWED_TRANSITIONS: Dict[TaskStatus, List[TaskStatus]] = {
    TaskStatus.PENDING:    [TaskStatus.PLANNING,   TaskStatus.DONE, TaskStatus.FAILED],
    TaskStatus.PLANNING:   [TaskStatus.GENERATING, TaskStatus.FAILED],
    TaskStatus.GENERATING: [TaskStatus.EXECUTING,  TaskStatus.FAILED],
    TaskStatus.EXECUTING:  [TaskStatus.VALIDATING, TaskStatus.FAILED],
    TaskStatus.VALIDATING: [TaskStatus.DONE,       TaskStatus.FIXING,     TaskStatus.FAILED],
    TaskStatus.FIXING:     [TaskStatus.VALIDATING, TaskStatus.GENERATING, TaskStatus.FAILED],
    #                               ↑ was missing — caused the crash
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
    plan: Optional[Dict[str, Any]] = None
    generated_files: Dict[str, str] = field(default_factory=dict)

    # Execution context
    workspace_path: Optional[str] = None
    venv_executables: Dict[str, str] = field(default_factory=dict)
    last_stdout: str = ""
    last_stderr: str = ""
    last_error: str = ""

    # Final result
    result: Optional[TaskResult] = None

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

    def record_execution_output(
        self, stdout: str, stderr: str, error: str = ""
    ) -> None:
        """Store the last execution output for use in fix prompts."""
        self.last_stdout = stdout
        self.last_stderr = stderr
        self.last_error = error

    def increment_fix_iteration(self) -> None:
        self.fix_iteration += 1

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Task(id={self.id}, status={self.status.name}, "
            f"fix_iter={self.fix_iteration}, input={self.raw_input[:50]!r})"
        )
