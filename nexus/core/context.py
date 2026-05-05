"""
nexus/core/context.py
──────────────────────
Session Context Manager — persists state across interactions within a single session.
"""
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from nexus.executor.workspace import ProjectScanner, ProjectSnapshot
from nexus.utils.config import config
from nexus.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SessionContext:
    """
    Holds the state of the current user session.
    Bounded memory-only storage with atomic JSON serialization on exit.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    session_start: datetime = field(default_factory=datetime.utcnow)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    task_history: List[Dict[str, Any]] = field(default_factory=list)
    active_project: Optional[str] = None
    project_snapshot: Optional[ProjectSnapshot] = None

    def add_message(self, role: str, content: str) -> None:
        """Add a message to history, bounded to last 20 entries."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
        if len(self.conversation_history) > 20:
            self.conversation_history.pop(0)

    def add_task_result(self, task_id: str, summary: str, intent: str, status: str) -> None:
        """Add a task outcome to history, bounded to last 10 entries."""
        self.task_history.append({
            "task_id": task_id,
            "summary": summary,
            "intent": intent,
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        })
        if len(self.task_history) > 10:
            self.task_history.pop(0)

    def set_project(self, path: str) -> None:
        """Set the current working project path and scan it."""
        self.active_project = path
        try:
            scanner = ProjectScanner(path)
            self.project_snapshot = scanner.scan()
            logger.info(
                "Active project set and scanned: %s (%d files)",
                path,
                self.project_snapshot.total_files
            )
        except Exception as e:
            logger.error("Failed to scan project %s: %s", path, e)
            self.project_snapshot = None

    def get_recent_context(self) -> str:
        """
        Format a brief summary of the session for LLM prompt injection.
        Includes last 3 tasks and last 5 messages.
        """
        lines = ["--- SESSION CONTEXT ---"]
        if self.active_project:
            lines.append(f"Active Project Path: {self.active_project}")

        if self.task_history:
            lines.append("\nRECENT TASKS:")
            for t in self.task_history[-3:]:
                lines.append(f"  - {t['task_id']} [{t['intent']}]: {t['status']} | {t['summary']}")

        if self.conversation_history:
            lines.append("\nRECENT CONVERSATION:")
            # Skip the very last message as it's usually the current prompt
            msgs = self.conversation_history[-6:-1] if len(self.conversation_history) > 1 else []
            for m in msgs:
                content = m['content']
                if len(content) > 150:
                    content = content[:147] + "..."
                lines.append(f"  {m['role'].upper()}: {content}")

        lines.append("-----------------------")
        return "\n".join(lines)

    def save(self) -> None:
        """
        Atomically serialize session context to JSON.
        Saved in .nexus/sessions/ directory.
        """
        try:
            # Determine sessions directory relative to workspace base
            sessions_dir = Path(config.workspace_base).parent / "sessions"
            sessions_dir.mkdir(parents=True, exist_ok=True)

            file_path = sessions_dir / f"session_{self.session_id}.json"
            temp_path = file_path.with_suffix(".tmp")

            # Prepare data
            data = asdict(self)
            data["session_start"] = self.session_start.isoformat()

            # Atomic write: write to temp then rename
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            
            temp_path.replace(file_path)
            logger.info("Session context persisted to %s", file_path)

        except Exception as e:
            logger.error("Failed to persist session context: %s", e)
            # Cleanup temp file if it exists and save failed
            try:
                temp_file = Path(config.workspace_base).parent / "sessions" / f"session_{self.session_id}.tmp"
                if temp_file.exists():
                    temp_file.unlink()
            except Exception:
                pass
