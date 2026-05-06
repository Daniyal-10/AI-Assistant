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
        super().__init__(message, context=context)
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
