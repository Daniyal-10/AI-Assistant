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


def estimate_tokens(text: str) -> int:
    """
    Fast approximation of token count for English text.
    Uses len(text) // 4 (good enough for 7B local models).
    """
    return len(text) // 4


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
    warm_history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        # Enforce token budget on warm history (max 25% of total budget)
        max_warm_tokens = config.nexus_context_token_budget // 4
        current_tokens = 0
        budgeted_history = []
        
        # Keep most recent (end of list), drop oldest (start of list) if overflow
        for record in reversed(self.warm_history):
            # Estimate tokens of the formatted representation
            record_str = f"[{record.get('intent', 'UNKNOWN')}] {record.get('execution_status', 'UNKNOWN')} | {record.get('plan_summary', '')}"
            tokens = estimate_tokens(record_str)
            if current_tokens + tokens > max_warm_tokens:
                logger.debug("Warm context token budget exceeded, dropping older records")
                break
            budgeted_history.append(record)
            current_tokens += tokens
            
        self.warm_history = list(reversed(budgeted_history))

    def add_message(self, role: str, content: str) -> None:
        """Add a message to history, bounded by hard caps from config."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
        if len(self.conversation_history) > config.nexus_conversation_history_limit:
            self.conversation_history.pop(0)

    def add_task_result(self, task_id: str, summary: str, intent: str, status: str) -> None:
        """Add a task outcome to history, bounded by hard caps from config."""
        self.task_history.append({
            "task_id": task_id,
            "summary": summary,
            "intent": intent,
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        })
        if len(self.task_history) > config.nexus_task_history_limit:
            self.task_history.pop(0)

    def get_truncated_history(self, token_budget: int) -> List[Dict[str, Any]]:
        """
        Returns conversation history trimmed to fit within token_budget.
        Keeps the most recent messages. Oldest messages are dropped first.
        Always keeps at least the last 2 messages regardless of budget.
        Returns a new list — does NOT modify self.conversation_history.
        """
        if len(self.conversation_history) <= 2:
            return list(self.conversation_history)

        accumulated_history = []
        current_tokens = 0
        
        # Iterate backwards through history (most recent first)
        for msg in reversed(self.conversation_history):
            msg_tokens = estimate_tokens(msg.get("content", ""))
            
            # Check budget
            if current_tokens + msg_tokens > token_budget:
                # Only stop if we already have at least 2 messages
                if len(accumulated_history) >= 2:
                    logger.debug(
                        "History truncation occurred: budget=%d, original=%d, returned=%d",
                        token_budget, len(self.conversation_history), len(accumulated_history)
                    )
                    break
            
            accumulated_history.append(msg)
            current_tokens += msg_tokens
            
        # Return in original chronological order
        return list(reversed(accumulated_history))

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

    def get_recent_context(self, history_override: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Format a brief summary of the session for LLM prompt injection.
        Includes last 3 tasks and a subset of conversation history.
        """
        lines = ["--- SESSION CONTEXT ---"]
        if self.active_project:
            lines.append(f"Active Project Path: {self.active_project}")

        if self.warm_history:
            lines.append("\n--- PRIOR SESSION HISTORY ---")
            for w in self.warm_history:
                # Format timestamp simply
                ts = w.get('timestamp', '')[:10] if w.get('timestamp') else 'unknown'
                lines.append(f"  - [{w.get('intent', 'UNKNOWN')}] at {ts}: {w.get('execution_status', 'UNKNOWN')} | {w.get('plan_summary', '')}")

        if self.task_history:
            lines.append("\nRECENT TASKS:")
            for t in self.task_history[-3:]:
                lines.append(f"  - {t['task_id']} [{t['intent']}]: {t['status']} | {t['summary']}")

        # Use provided history or fall back to token-budgeted truncation
        history = history_override if history_override is not None else \
                 self.get_truncated_history(token_budget=config.nexus_context_token_budget)

        if history:
            lines.append("\nRECENT CONVERSATION:")
            # Skip the very last message as it's usually the current prompt
            # But if we use truncated history, we just show what we got
            # We filter out the last one if it's identical to the current user input 
            # (handled by caller or just shown as is)
            for m in history:
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
