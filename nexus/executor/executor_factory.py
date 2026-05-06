"""
nexus/executor/executor_factory.py
──────────────────────────────────
Runtime selector for NEXUS code executors.
Manages the singleton instance of the active executor (Local vs Container).
"""
from typing import Optional

from nexus.executor.exec_interface import BaseExecutor
from nexus.executor.local_executor import LocalExecutor
from nexus.executor.container_executor import ContainerExecutor
from nexus.utils.config import config
from nexus.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level singleton
_cached_executor: Optional[BaseExecutor] = None


def get_executor() -> BaseExecutor:
    """
    Select and return the correct executor based on NEXUS_EXECUTOR configuration.
    
    Logic:
    1. Check config.executor_type (from NEXUS_EXECUTOR env var).
    2. If "docker", attempt to use ContainerExecutor.
    3. Fall back to LocalExecutor if Docker is unavailable or if "local" is set.
    
    Returns:
        The active BaseExecutor instance (singleton).
    """
    global _cached_executor
    if _cached_executor is not None:
        return _cached_executor

    executor_type = config.executor_type

    if executor_type == "docker":
        executor = ContainerExecutor()
        if executor.is_available():
            logger.info("Executor: Docker (production mode)")
            _cached_executor = executor
            return executor
        else:
            logger.warning("Docker executor requested but Docker daemon is not available. Falling back to local.")
    
    # Default to Local
    logger.info("Executor: Local (development mode)")
    _cached_executor = LocalExecutor()
    return _cached_executor
