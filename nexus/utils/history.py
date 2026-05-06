"""
nexus/utils/history.py
──────────────────────
Persistent Task History Store — records execution metadata in rotated JSONL files.
"""
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from nexus.utils.config import config
from nexus.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TaskRecord:
    """Stable schema for task execution history."""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    raw_input: str = ""
    intent: str = "UNKNOWN"
    plan_summary: str = ""
    execution_status: str = "SKIPPED"
    semantic_verdict: str = "NOT_CHECKED"
    fix_attempts: int = 0
    final_output_summary: str = ""
    tags: List[str] = field(default_factory=list)
    schema_version: str = "1.0"


class TaskHistory:
    """
    Append-only JSONL storage with monthly rotation.
    Queries scan from the end of the file for recent-first results.
    """
    def __init__(self):
        self.history_dir = Path(config.workspace_base).parent / "history"
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def _get_current_file(self) -> Path:
        """Monthly rotation: tasks_YYYY_MM.jsonl"""
        now = datetime.utcnow()
        return self.history_dir / f"tasks_{now.year}_{now.month:02d}.jsonl"

    def record(self, task: Any, session_id: str) -> None:
        """
        Record a task completion. Fire-and-forget (non-blocking).
        Safe failure: never crashes the main pipeline.
        """
        try:
            summary = task.result.summary if hasattr(task, "result") and task.result else "No result"
            # Hard truncation at 500 chars for storage security/efficiency
            summary = summary[:500]

            # Extract intent value safely
            intent_val = "UNKNOWN"
            if hasattr(task, "intent") and task.intent:
                if hasattr(task.intent, "intent"):
                    intent_val = task.intent.intent.value
                elif hasattr(task.intent, "value"):
                    intent_val = task.intent.value

            record = TaskRecord(
                session_id=session_id,
                raw_input=task.raw_input,
                intent=intent_val,
                plan_summary=task.plan.get("description", "") if hasattr(task, "plan") and task.plan else "",
                execution_status=task.status.name,
                semantic_verdict=getattr(task.result, "semantic_verdict", "NOT_CHECKED") if hasattr(task, "result") and task.result else "NOT_CHECKED",
                fix_attempts=getattr(task, "fix_iteration", 0),
                final_output_summary=summary,
                tags=[] # Reserved for future manual tagging
            )

            self._append_jsonl(self._get_current_file(), asdict(record))
        except Exception as e:
            logger.error("Failed to record task history (non-fatal): %s", e)

    def _append_jsonl(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Atomic append-only write."""
        # Note: True 'atomic' append to JSONL in a multi-process environment 
        # usually relies on O_APPEND at the OS level. For distributed safety 
        # we would use a lock, but here we prioritize non-blocking operation.
        try:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            logger.error("IO Error appending to history: %s", e)

    def get_recent(self, n: int = 10) -> List[TaskRecord]:
        """Get last N records from the current month's history."""
        return self._query(limit=n)

    def get_by_status(self, status: str) -> List[TaskRecord]:
        """Filter records by execution status."""
        return self._query(filter_fn=lambda r: r.execution_status == status)

    def get_by_intent(self, intent: str) -> List[TaskRecord]:
        """Filter records by intent type."""
        return self._query(filter_fn=lambda r: r.intent == intent)

    def search_by_keyword(self, keyword: str) -> List[TaskRecord]:
        """Naive string search across raw user inputs."""
        keyword = keyword.lower()
        return self._query(filter_fn=lambda r: keyword in r.raw_input.lower())

    def _query(self, filter_fn: Any = None, limit: int = 1000) -> List[TaskRecord]:
        """
        Internal query engine. Scans file from end (recent-first).
        Uses a bounded backward read (max 512KB) to prevent memory bloat and guarantee speed.
        """
        file_path = self._get_current_file()
        if not file_path.exists():
            return []

        results = []
        try:
            MAX_BYTES = 512 * 1024  # 512 KB
            file_size = file_path.stat().st_size
            
            with open(file_path, "rb") as f:
                if file_size > MAX_BYTES:
                    f.seek(file_size - MAX_BYTES)
                    # Skip the first partial line
                    f.readline()
                content = f.read().decode("utf-8", errors="replace")
                
            lines = [line for line in content.splitlines() if line.strip()]
                
            for line in reversed(lines):
                if len(results) >= limit:
                    break
                
                try:
                    data = json.loads(line)
                    record = TaskRecord(**data)
                    if filter_fn is None or filter_fn(record):
                        results.append(record)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning("Corrupt line in history: %s", e)
                    continue
                    
        except Exception as e:
            logger.error("History query failed: %s", e)

        return results
