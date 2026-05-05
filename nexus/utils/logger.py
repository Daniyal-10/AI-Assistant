"""
Structured logger for NEXUS. All components use this.
Outputs clean, leveled logs with optional task context.
"""
import logging
import os
import sys
from typing import Optional


class TaskAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        task_id = self.extra.get("task_id")
        if task_id:
            msg = f"[task:{task_id}] {msg}"
        return msg, kwargs


def get_logger(name: str, task_id: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        level = os.environ.get("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level, logging.INFO))

    if task_id:
        return TaskAdapter(logger, {"task_id": task_id})

    return logger
