"""
nexus/memory/store.py
─────────────────────
SQLite-backed persistent store for NEXUS memory.

Single responsibility: raw database operations only.
No business logic. No LLM calls. No formatting.

Schema:
  preferences      — key/value user preference store
  project_memory   — per-project state and summaries
  execution_records — queryable task execution log (replaces raw JSONL queries)

Thread safety: WAL mode + per-connection isolation.
"""
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from nexus.utils.logger import get_logger

logger = get_logger(__name__)

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS preferences (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    category    TEXT NOT NULL DEFAULT 'general',
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS project_memory (
    project_path    TEXT PRIMARY KEY,
    summary         TEXT NOT NULL DEFAULT '',
    tech_stack      TEXT NOT NULL DEFAULT '',
    key_files       TEXT NOT NULL DEFAULT '',
    last_task       TEXT NOT NULL DEFAULT '',
    task_count      INTEGER NOT NULL DEFAULT 0,
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS execution_records (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    intent          TEXT NOT NULL,
    raw_input       TEXT NOT NULL,
    status          TEXT NOT NULL,
    summary         TEXT NOT NULL DEFAULT '',
    fix_attempts    INTEGER NOT NULL DEFAULT 0,
    duration_ms     INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_exec_intent   ON execution_records(intent);
CREATE INDEX IF NOT EXISTS idx_exec_status   ON execution_records(status);
CREATE INDEX IF NOT EXISTS idx_exec_created  ON execution_records(created_at);
CREATE INDEX IF NOT EXISTS idx_pref_category ON preferences(category);
"""


class MemoryStore:
    """
    Thread-safe SQLite connection pool with WAL mode.
    One instance per process — use as a singleton via MemoryManager.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._local = threading.local()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Bootstrap schema on first connection
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        logger.info("MemoryStore ready: %s", db_path)

    def _connect(self) -> sqlite3.Connection:
        """Return a thread-local connection."""
        if not getattr(self._local, "conn", None):
            conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                timeout=10,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Context manager for atomic writes."""
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def execute(
        self, sql: str, params: Tuple = ()
    ) -> List[sqlite3.Row]:
        """Read-only query. Returns list of Row objects."""
        conn = self._connect()
        try:
            cursor = conn.execute(sql, params)
            return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error("DB read error: %s | SQL: %s", e, sql[:100])
            return []

    def close(self) -> None:
        """Close thread-local connection."""
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            self._local.conn = None
