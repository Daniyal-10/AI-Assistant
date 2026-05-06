"""
nexus/executor/docker_exec.py
────────────────────────────
Docker-based execution isolation for NEXUS.
Provides OS-level security boundaries for AI-generated code.
"""
import os
import traceback
from typing import List, Tuple, Optional, Any
from pathlib import Path

import docker
from docker.errors import APIError, ContainerError, ImageNotFound
from requests.exceptions import ReadTimeout

from nexus.core.exceptions import ExecutionError
from nexus.executor.safe_exec import scan_for_forbidden_patterns
from nexus.utils.config import config
from nexus.utils.logger import get_logger

logger = get_logger(__name__)

# --- CONSTANTS ---
NEXUS_DOCKER_IMAGE = "python:3.10-slim"

# --- STATE ---
_docker_client: Optional[docker.DockerClient] = None
_docker_available_cache: Optional[bool] = None
_active_containers: set[str] = set()
_CONTAINER_TRACKING_FILE = Path.home() / ".nexus" / "active_containers.json"


def _persist_active_containers() -> None:
    """Write current active container IDs to disk for persistence across restarts."""
    try:
        _CONTAINER_TRACKING_FILE.parent.mkdir(parents=True, exist_ok=True)
        import json
        _CONTAINER_TRACKING_FILE.write_text(json.dumps(list(_active_containers)))
    except Exception:
        pass  # tracking is best-effort, never block execution


def _load_active_containers() -> None:
    """Load container IDs from disk into memory on module import."""
    global _active_containers
    try:
        if _CONTAINER_TRACKING_FILE.exists():
            import json
            ids = json.loads(_CONTAINER_TRACKING_FILE.read_text())
            _active_containers.update(ids)
    except Exception:
        pass


def _get_client() -> docker.DockerClient:
    """
    Module-level lazy singleton for Docker SDK client.
    Avoids creating a new socket connection per task.
    """
    global _docker_client
    if _docker_client is None:
        _docker_client = docker.from_env()
    return _docker_client


def is_docker_available() -> bool:
    """
    Check if Docker daemon is reachable via SDK and ping().
    Returns True if reachable, False otherwise. Caches result.
    """
    global _docker_available_cache
    if _docker_available_cache is not None:
        return _docker_available_cache

    try:
        client = _get_client()
        client.ping()
        _docker_available_cache = True
        logger.info("Docker daemon reachable — production execution mode available")
    except Exception as e:
        _docker_available_cache = False
        logger.info(f"Docker daemon not reachable — {str(e)}")
    
    return _docker_available_cache


def run_in_container(
    workspace_path: str,
    venv_path: str,
    command: List[str],
    timeout: int
) -> Tuple[int, str, str]:
    """
    Core execution function: Spawns a python:3.10-slim container with strict resource 
    limits and network isolation.
    
    Args:
        workspace_path: Absolute path to task workspace (mounted RW at /workspace)
        venv_path: Absolute path to task venv (mounted RO at /venv)
        command: List of command strings to execute
        timeout: Execution timeout in seconds
        
    Returns:
        tuple (exit_code, stdout, stderr)
        
    Raises:
        ExecutionError: If Docker is unavailable, image missing, or execution fails.
    """
    # 1. Pre-execution validation
    if not is_docker_available():
        raise ExecutionError("Docker is not available")
    
    if not os.path.exists(workspace_path):
        raise ExecutionError(f"Workspace path does not exist: {workspace_path}")
    
    if not os.path.exists(venv_path):
        raise ExecutionError(f"Venv path does not exist: {venv_path}")
    
    if not isinstance(command, list) or not command:
        raise ExecutionError("Invalid command: must be a non-empty list of strings")

    client = _get_client()
    try:
        client.images.get(NEXUS_DOCKER_IMAGE)
    except ImageNotFound:
        raise ExecutionError(f"Docker image {NEXUS_DOCKER_IMAGE} not found locally. Run ensure_image_cached() first.")

    task_id = os.path.basename(workspace_path)
    logger.info("Starting container for command: %s (workspace: %s)", command[0], task_id)

    container = None
    try:
        # 2. Spawn container with exact specs
        # network_disabled=True (no network access)
        # user="1000:1000" (non-root WSL2 user)
        # nano_cpus=1_000_000_000 (1.0 CPU core)
        # mem_limit="512m"
        # remove=True (auto-delete on exit)
        container = client.containers.run(
            image=NEXUS_DOCKER_IMAGE,
            command=command,
            volumes={
                os.path.abspath(workspace_path): {"bind": "/workspace", "mode": "rw"},
                os.path.abspath(venv_path):      {"bind": "/venv",      "mode": "ro"},
            },
            working_dir="/workspace",
            user="1000:1000",
            network_disabled=True,
            mem_limit="512m",
            nano_cpus=1_000_000_000,
            remove=False,   # Manual cleanup in finally block to avoid race condition on logs
            detach=True,
            stderr=True,
        )
        
        _active_containers.add(container.id)
        _persist_active_containers()

        # 3. Wait for completion with timeout enforcement
        try:
            result = container.wait(timeout=timeout)
            exit_code = result["StatusCode"]
            
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
            
            logger.info("Container finished: exit_code=%d (workspace: %s)", exit_code, task_id)
            return (exit_code, stdout, stderr)

        except Exception as e:
            err_str = str(e).lower()
            if isinstance(e, ReadTimeout) or "timeout" in err_str or "timed out" in err_str:
                logger.warning("Container timed out after %ds (workspace: %s)", timeout, task_id)
                try:
                    container.kill()
                except Exception as kill_err:
                    logger.error("Failed to kill timed-out container %s: %s", container.id[:12], kill_err)
                
                return (124, "", f"Execution timed out after {timeout} seconds")
            raise  # Re-raise non-timeout exceptions to outer handlers

    except ContainerError as e:
        logger.error("Docker ContainerError: %s (workspace: %s)", e, task_id)
        raise ExecutionError(f"Container failed: {str(e)}")
    except APIError as e:
        logger.error("Docker APIError: %s (workspace: %s)", e, task_id)
        raise ExecutionError(f"Docker API error: {str(e)}")
    except Exception as e:
        logger.error("Unexpected error in Docker execution: %s\n%s", e, traceback.format_exc())
        raise ExecutionError(f"Docker execution failed: {str(e)}")
    finally:
        if container:
            try:
                _active_containers.discard(container.id)
                _persist_active_containers()
                container.remove(force=True)
            except Exception:
                pass


def run_code_in_container(
    workspace_path: str,
    venv_path: str,
    script_filename: str,
    timeout: int
) -> Tuple[int, str, str]:
    """
    Security-hardened code execution: AST gate -> path validation -> Docker run.
    """
    # 1. Security validation (Path Traversal)
    if any(p in script_filename for p in ["..", "/", "\\"]) or script_filename.startswith("~"):
        raise ExecutionError("Path traversal detected in script filename")
    
    if not script_filename.endswith(".py"):
        raise ExecutionError("Script must be a .py file")
    
    full_path = os.path.join(workspace_path, script_filename)
    if not os.path.realpath(full_path).startswith(os.path.realpath(workspace_path)):
        raise ExecutionError("Script path escapes workspace boundary")

    # 2. AST Security Gate (runs on host before container starts)
    scan_for_forbidden_patterns(full_path, workspace_path)
    logger.info("AST gate passed for %s", script_filename)

    # 3. Execution via internal runner
    command = ["/venv/bin/python3", f"/workspace/{script_filename}"]
    return run_in_container(workspace_path, venv_path, command, timeout)


def run_tests_in_container(
    workspace_path: str,
    venv_path: str,
    test_dir: str,
    timeout: int
) -> Tuple[int, str, str]:
    """
    Isolated test execution: Runs pytest inside Docker with short tracebacks.
    """
    # 1. Security validation
    if ".." in test_dir or test_dir.startswith("/") or test_dir.startswith("~"):
        raise ExecutionError("Path traversal detected in test_dir")
    
    full_path = os.path.join(workspace_path, test_dir)
    if not os.path.realpath(full_path).startswith(os.path.realpath(workspace_path)):
         raise ExecutionError("Test path escapes workspace boundary")

    # 2. Execution via internal runner
    command = ["/venv/bin/pytest", f"/workspace/{test_dir}", "-v", "--tb=short"]
    return run_in_container(workspace_path, venv_path, command, timeout)


def ensure_image_cached() -> None:
    """
    Pulls python:3.10-slim if not already present locally.
    Must be called once during NEXUS setup/initialization.
    """
    client = _get_client()
    try:
        client.images.get(NEXUS_DOCKER_IMAGE)
        logger.info("Image %s already cached", NEXUS_DOCKER_IMAGE)
    except ImageNotFound:
        logger.info("Pulling %s — this runs once...", NEXUS_DOCKER_IMAGE)
        try:
            client.images.pull(NEXUS_DOCKER_IMAGE)
            logger.info("Image cached successfully")
        except Exception as e:
            logger.error("Failed to pull Docker image: %s", e)
            raise ExecutionError(f"Failed to pull {NEXUS_DOCKER_IMAGE}: {str(e)}")
    except Exception as e:
        logger.error("Docker error while checking image: %s", e)
        raise ExecutionError(f"Docker error: {str(e)}")


def cleanup_orphaned_containers() -> int:
    """
    Kills and removes any containers tracked in _active_containers that 
    are still running (e.g. following a host crash).
    
    INTEGRATION NOTE: Called by engine.py on startup and shutdown.
    Wired in during Prompt 5 (Executor Abstraction Layer).
    Do not remove this note until Prompt 5 is complete.
    """
    count = 0
    client = _get_client()
    for container_id in list(_active_containers):
        try:
            container = client.containers.get(container_id)
            logger.warning("Cleaning up orphaned container: %s", container_id[:12])
            container.kill()
            container.remove(force=True)
            _active_containers.discard(container_id)
            count += 1
        except Exception:
            _active_containers.discard(container_id)
    return count


# --- INITIALIZATION ---
_load_active_containers()
