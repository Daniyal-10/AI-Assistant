"""
Custom exceptions for NEXUS. Never use bare Exception.
Each error carries context for clean logging and retry decisions.
"""
from typing import Optional, Dict, Any


class NexusBaseException(Exception):
    """Base for all NEXUS exceptions."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation of the exception."""
        return {
            "type": self.__class__.__name__,
            "message": str(self).split(" | context=")[0],
            "context": self.context,
        }

    def __str__(self):
        if self.context:
            return f"{super().__str__()} | context={self.context}"
        return super().__str__()


class TaskPlanningError(NexusBaseException):
    """AI failed to produce a valid plan."""
    pass


class TaskGenerationError(NexusBaseException):
    """AI failed to generate valid code/files."""
    pass


class ExecutionError(NexusBaseException):
    """Code execution failed (non-zero exit or timeout)."""

    def __init__(
        self,
        message: str,
        stdout: str = "",
        stderr: str = "",
        context: Optional[Dict[str, Any]] = None,
    ):
        ctx = context or {}
        ctx.update({"stdout": stdout, "stderr": stderr})
        super().__init__(message, context=ctx)
        self.stdout = stdout
        self.stderr = stderr


class ValidationError(NexusBaseException):
    """Output did not meet expected validation criteria."""
    pass


class SafetyViolation(NexusBaseException):
    """Command or code was blocked by safety checks."""
    pass


class MaxRetriesExceeded(NexusBaseException):
    """Auto-fix loop hit the max iteration cap."""
    pass


class WorkspaceSecurityError(NexusBaseException):
    """Path traversal or safety violation in workspace."""
    pass


class OllamaConnectionError(NexusBaseException):
    """Cannot reach Ollama service."""
    pass


class CloudProviderError(NexusBaseException):
    """External cloud AI provider (e.g. Anthropic) failed."""
    pass


# ── New Exceptions ────────────────────────────────────────────────────────────

class ProviderError(NexusBaseException):
    """Raised when any LLM provider fails."""

    def __init__(self, message: str, provider_name: str, is_retryable: bool, context: Optional[Dict[str, Any]] = None):
        ctx = context or {}
        ctx.update({"provider_name": provider_name, "is_retryable": is_retryable})
        super().__init__(message, context=ctx)
        self.provider_name = provider_name
        self.is_retryable = is_retryable


class PipelineStageError(NexusBaseException):
    """Raised when a pipeline stage fails with a typed reason."""

    def __init__(self, message: str, stage_name: str, is_fatal: bool, context: Optional[Dict[str, Any]] = None):
        ctx = context or {}
        ctx.update({"stage_name": stage_name, "is_fatal": is_fatal})
        super().__init__(message, context=ctx)
        self.stage_name = stage_name
        self.is_fatal = is_fatal


class WorkspaceError(NexusBaseException):
    """General workspace operation failure."""

    def __init__(self, message: str, workspace_id: str, operation: str, context: Optional[Dict[str, Any]] = None):
        ctx = context or {}
        ctx.update({"workspace_id": workspace_id, "operation": operation})
        super().__init__(message, context=ctx)
        self.workspace_id = workspace_id
        self.operation = operation


class ConfigurationError(NexusBaseException):
    """Raised at startup for bad config."""

    def __init__(self, message: str, field_name: str, expected: str, context: Optional[Dict[str, Any]] = None):
        ctx = context or {}
        ctx.update({"field_name": field_name, "expected": expected})
        super().__init__(message, context=ctx)
        self.field_name = field_name
        self.expected = expected


__all__ = [
    "NexusBaseException",
    "TaskPlanningError",
    "TaskGenerationError",
    "ExecutionError",
    "ValidationError",
    "SafetyViolation",
    "MaxRetriesExceeded",
    "WorkspaceSecurityError",
    "OllamaConnectionError",
    "CloudProviderError",
    "ProviderError",
    "PipelineStageError",
    "WorkspaceError",
    "ConfigurationError",
]
