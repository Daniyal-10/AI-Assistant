"""
nexus/memory/models.py
──────────────────────
Typed data contracts for the memory layer.
Pure dataclasses — no DB logic here.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Preference:
    """A single user preference entry."""
    key: str
    value: str
    category: str = "general"      # "technical" | "personal" | "style" | "general"
    updated_at: str = ""


@dataclass
class ProjectMemory:
    """Persistent state for a specific project path."""
    project_path: str
    summary: str = ""              # AI-generated one-paragraph summary
    tech_stack: str = ""           # comma-separated: "Python,FastAPI,PostgreSQL"
    key_files: str = ""            # comma-separated entry points and core modules
    last_task: str = ""            # description of last task run on this project
    task_count: int = 0
    updated_at: str = ""


@dataclass
class ExecutionRecord:
    """A queryable execution history entry."""
    session_id: str
    intent: str
    raw_input: str
    status: str                    # DONE | FAILED
    summary: str = ""
    fix_attempts: int = 0
    duration_ms: int = 0
    created_at: str = ""
    id: Optional[int] = None
