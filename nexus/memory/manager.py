"""
nexus/memory/manager.py
───────────────────────
MemoryManager — single public API for all persistent memory operations.

Usage:
    from nexus.memory.manager import get_memory
    mem = get_memory()
    mem.set_preference("code_style", "stdlib_first", category="technical")
    prefs = mem.get_preferences(category="technical")

All methods are safe to call — never raise, never crash the pipeline.
"""
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional

from nexus.memory.models import ExecutionRecord, Preference, ProjectMemory
from nexus.memory.store import MemoryStore
from nexus.utils.config import get_config
from nexus.utils.logger import get_logger

logger = get_logger(__name__)

_instance: Optional["MemoryManager"] = None
_instance_lock = threading.Lock()


def get_memory() -> "MemoryManager":
    """Return the global MemoryManager singleton."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                cfg = get_config()
                db_path = Path(cfg.workspace_base).parent / "memory.db"
                _instance = MemoryManager(db_path)
    return _instance


def reset_memory(db_path: Optional[Path] = None) -> None:
    """Reset singleton — USE ONLY IN TESTS."""
    global _instance
    with _instance_lock:
        if _instance:
            _instance._store.close()
        _instance = MemoryManager(db_path) if db_path else None


class MemoryManager:
    """
    Persistent memory operations for NEXUS.
    All public methods catch exceptions internally — memory must never
    crash the main pipeline.
    """

    def __init__(self, db_path: Path) -> None:
        self._store = MemoryStore(db_path)

    # ── Preferences ───────────────────────────────────────────────────────────

    def set_preference(
        self, key: str, value: str, category: str = "general"
    ) -> bool:
        """Upsert a user preference. Returns True on success."""
        try:
            with self._store.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO preferences (key, value, category, updated_at)
                    VALUES (?, ?, ?, datetime('now'))
                    ON CONFLICT(key) DO UPDATE SET
                        value      = excluded.value,
                        category   = excluded.category,
                        updated_at = excluded.updated_at
                    """,
                    (key, value, category),
                )
            return True
        except Exception as e:
            logger.error("set_preference failed [%s]: %s", key, e)
            return False

    def get_preference(self, key: str) -> Optional[str]:
        """Get a single preference value by key."""
        try:
            rows = self._store.execute(
                "SELECT value FROM preferences WHERE key = ?", (key,)
            )
            return rows[0]["value"] if rows else None
        except Exception as e:
            logger.error("get_preference failed [%s]: %s", key, e)
            return None

    def get_preferences(self, category: Optional[str] = None) -> List[Preference]:
        """Get all preferences, optionally filtered by category."""
        try:
            if category:
                rows = self._store.execute(
                    "SELECT * FROM preferences WHERE category = ? ORDER BY key",
                    (category,),
                )
            else:
                rows = self._store.execute(
                    "SELECT * FROM preferences ORDER BY category, key"
                )
            return [
                Preference(
                    key=r["key"],
                    value=r["value"],
                    category=r["category"],
                    updated_at=r["updated_at"],
                )
                for r in rows
            ]
        except Exception as e:
            logger.error("get_preferences failed: %s", e)
            return []

    def delete_preference(self, key: str) -> bool:
        """Remove a preference by key."""
        try:
            with self._store.transaction() as conn:
                conn.execute("DELETE FROM preferences WHERE key = ?", (key,))
            return True
        except Exception as e:
            logger.error("delete_preference failed [%s]: %s", key, e)
            return False

    # ── Project Memory ────────────────────────────────────────────────────────

    def save_project(self, project: ProjectMemory) -> bool:
        """Upsert project memory for a given path."""
        try:
            with self._store.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO project_memory
                        (project_path, summary, tech_stack, key_files,
                         last_task, task_count, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                    ON CONFLICT(project_path) DO UPDATE SET
                        summary      = excluded.summary,
                        tech_stack   = excluded.tech_stack,
                        key_files    = excluded.key_files,
                        last_task    = excluded.last_task,
                        task_count   = task_count + excluded.task_count,
                        updated_at   = excluded.updated_at
                    """,
                    (
                        project.project_path,
                        project.summary,
                        project.tech_stack,
                        project.key_files,
                        project.last_task,
                        project.task_count,
                    ),
                )
            return True
        except Exception as e:
            logger.error("save_project failed [%s]: %s", project.project_path, e)
            return False

    def get_project(self, project_path: str) -> Optional[ProjectMemory]:
        """Load project memory for a given path. Returns None if not found."""
        try:
            rows = self._store.execute(
                "SELECT * FROM project_memory WHERE project_path = ?",
                (project_path,),
            )
            if not rows:
                return None
            r = rows[0]
            return ProjectMemory(
                project_path=r["project_path"],
                summary=r["summary"],
                tech_stack=r["tech_stack"],
                key_files=r["key_files"],
                last_task=r["last_task"],
                task_count=r["task_count"],
                updated_at=r["updated_at"],
            )
        except Exception as e:
            logger.error("get_project failed [%s]: %s", project_path, e)
            return None

    def get_all_projects(self) -> List[ProjectMemory]:
        """Return all tracked projects ordered by last update."""
        try:
            rows = self._store.execute(
                "SELECT * FROM project_memory ORDER BY updated_at DESC"
            )
            return [
                ProjectMemory(
                    project_path=r["project_path"],
                    summary=r["summary"],
                    tech_stack=r["tech_stack"],
                    key_files=r["key_files"],
                    last_task=r["last_task"],
                    task_count=r["task_count"],
                    updated_at=r["updated_at"],
                )
                for r in rows
            ]
        except Exception as e:
            logger.error("get_all_projects failed: %s", e)
            return []

    # ── Execution Records ─────────────────────────────────────────────────────

    def record_execution(
        self,
        session_id: str,
        intent: str,
        raw_input: str,
        status: str,
        summary: str = "",
        fix_attempts: int = 0,
        duration_ms: int = 0,
    ) -> bool:
        """Append an execution record. Non-blocking write."""
        try:
            # Truncate at 500 chars — storage efficiency
            summary = summary[:500]
            raw_input = raw_input[:500]
            with self._store.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO execution_records
                        (session_id, intent, raw_input, status,
                         summary, fix_attempts, duration_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, intent, raw_input, status,
                     summary, fix_attempts, duration_ms),
                )
            return True
        except Exception as e:
            logger.error("record_execution failed: %s", e)
            return False

    def get_recent_executions(
        self, limit: int = 10, intent: Optional[str] = None
    ) -> List[ExecutionRecord]:
        """Get recent execution records, newest first."""
        try:
            if intent:
                rows = self._store.execute(
                    """SELECT * FROM execution_records
                       WHERE intent = ?
                       ORDER BY created_at DESC LIMIT ?""",
                    (intent, limit),
                )
            else:
                rows = self._store.execute(
                    """SELECT * FROM execution_records
                       ORDER BY created_at DESC LIMIT ?""",
                    (limit,),
                )
            return [
                ExecutionRecord(
                    id=r["id"],
                    session_id=r["session_id"],
                    intent=r["intent"],
                    raw_input=r["raw_input"],
                    status=r["status"],
                    summary=r["summary"],
                    fix_attempts=r["fix_attempts"],
                    duration_ms=r["duration_ms"],
                    created_at=r["created_at"],
                )
                for r in rows
            ]
        except Exception as e:
            logger.error("get_recent_executions failed: %s", e)
            return []

    def get_success_rate(self, intent: Optional[str] = None) -> Dict[str, float]:
        """Return success/failure counts and rate."""
        try:
            base = "SELECT status, COUNT(*) as cnt FROM execution_records"
            where = " WHERE intent = ?" if intent else ""
            sql = base + where + " GROUP BY status"
            params = (intent,) if intent else ()
            rows = self._store.execute(sql, params)

            counts = {r["status"]: r["cnt"] for r in rows}
            total = sum(counts.values())
            done = counts.get("DONE", 0)
            return {
                "total": total,
                "done": done,
                "failed": counts.get("FAILED", 0),
                "rate": round(done / total, 3) if total > 0 else 0.0,
            }
        except Exception as e:
            logger.error("get_success_rate failed: %s", e)
            return {"total": 0, "done": 0, "failed": 0, "rate": 0.0}

    # ── Context injection ─────────────────────────────────────────────────────

    def build_memory_context(
        self, project_path: Optional[str] = None
    ) -> str:
        """
        Format persistent memory into a prompt-ready block.
        Called by SessionContext.get_recent_context().
        Capped at ~800 chars to stay within token budget.
        """
        lines = []

        # Technical preferences
        tech_prefs = self.get_preferences(category="technical")
        if tech_prefs:
            lines.append("--- PERSISTENT PREFERENCES ---")
            for p in tech_prefs[:5]:
                lines.append(f"  [{p.key}]: {p.value}")

        # Project memory
        if project_path:
            proj = self.get_project(project_path)
            if proj:
                lines.append("--- PROJECT MEMORY ---")
                if proj.summary:
                    lines.append(f"  Summary: {proj.summary[:200]}")
                if proj.tech_stack:
                    lines.append(f"  Stack: {proj.tech_stack}")
                if proj.last_task:
                    lines.append(f"  Last task: {proj.last_task[:100]}")
                lines.append(f"  Tasks run: {proj.task_count}")

        # Recent success rate
        stats = self.get_success_rate()
        if stats["total"] > 0:
            lines.append(
                f"--- EXECUTION STATS: {stats['done']}/{stats['total']} "
                f"tasks succeeded ({stats['rate']*100:.0f}%) ---"
            )

        result = "\n".join(lines)
        # Hard cap — never blow the token budget
        return result[:800] if result else ""
