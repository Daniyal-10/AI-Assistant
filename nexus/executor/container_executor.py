"""
nexus/executor/container_executor.py
────────────────────────────────────
Docker-based execution backend for NEXUS. Runs code in isolated containers
using python:3.10-slim.
"""
from typing import Tuple
from nexus.executor.exec_interface import BaseExecutor
from nexus.core.exceptions import ExecutionError

try:
    from nexus.executor import docker_exec
    DOCKER_SDK_AVAILABLE = True
except ImportError:
    # We catch the import error here so the factory can still be imported
    # even if the docker SDK is missing from the environment.
    DOCKER_SDK_AVAILABLE = False


class ContainerExecutor(BaseExecutor):
    """
    Executor that runs code in isolated Docker containers.
    Production/Production-ready mode.
    """

    def _ensure_sdk(self) -> None:
        """Internal helper to raise helpful error if SDK is missing at runtime."""
        if not DOCKER_SDK_AVAILABLE:
            raise ExecutionError(
                "Docker SDK is not installed in the current environment. "
                "Container execution is unavailable. "
                "Run 'pip install docker' or set NEXUS_EXECUTOR=local"
            )

    def execute_script(
        self,
        workspace_path: str,
        venv_path: str,
        script_filename: str,
        timeout: int
    ) -> Tuple[int, str, str]:
        """Call docker_exec backend to run script in container."""
        self._ensure_sdk()
        return docker_exec.run_code_in_container(
            workspace_path, venv_path, script_filename, timeout
        )

    def execute_tests(
        self,
        workspace_path: str,
        venv_path: str,
        test_dir: str,
        timeout: int
    ) -> Tuple[int, str, str]:
        """Call docker_exec backend to run pytest in container."""
        self._ensure_sdk()
        return docker_exec.run_tests_in_container(
            workspace_path, venv_path, test_dir, timeout
        )

    def is_available(self) -> bool:
        """Check if Docker SDK is present AND daemon is reachable."""
        if not DOCKER_SDK_AVAILABLE:
            return False
        return docker_exec.is_docker_available()
