"""
nexus/core/restorer.py
──────────────────────
Session Restorer — loads recent task history into the session context at boot.
"""
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

from nexus.utils.history import TaskHistory
from nexus.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WarmContextRecord:
    """Subset of TaskRecord useful for AI prior context."""
    intent: str
    plan_summary: str
    execution_status: str
    timestamp: str


class SessionRestorer:
    """
    Single-responsibility component that runs at boot, fetches recent tasks,
    and formats them into a warm payload.
    """

    @staticmethod
    def restore(history_store: TaskHistory, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch recent history and map it to the warm context schema.
        Handles empty/missing history gracefully.
        Returns a structured payload or an empty list in chronological order.
        """
        payload = []
        try:
            records = history_store.get_recent(limit)
            # get_recent returns recent-first. Reverse it so the prompt reads chronologically.
            for record in reversed(records):
                warm_record = WarmContextRecord(
                    intent=record.intent,
                    plan_summary=record.plan_summary,
                    execution_status=record.execution_status,
                    timestamp=record.timestamp
                )
                payload.append(asdict(warm_record))
                
            if payload:
                logger.info("Session warm-start successful: loaded %d prior tasks", len(payload))
            else:
                logger.debug("Clean cold start: no prior tasks found")
                
        except Exception as e:
            logger.warning("Session warm-start failed, proceeding with cold start: %s", e)
            
        return payload
