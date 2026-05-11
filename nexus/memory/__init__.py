"""
nexus/memory
────────────
Persistent memory layer for NEXUS.
Provides preference memory, project memory, and queryable execution history.

Public API:
    from nexus.memory.manager import get_memory
    mem = get_memory()
"""
from nexus.memory.manager import get_memory, reset_memory
from nexus.memory.models import ExecutionRecord, Preference, ProjectMemory

__all__ = [
    "get_memory",
    "reset_memory",
    "Preference",
    "ProjectMemory",
    "ExecutionRecord",
]
