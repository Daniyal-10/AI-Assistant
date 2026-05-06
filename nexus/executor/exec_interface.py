"""
nexus/executor/exec_interface.py
───────────────────────────────
Abstract base class for NEXUS code executors.
Defines the contract for running scripts and tests in both local and isolated modes.
"""
from abc import ABC, abstractmethod
from typing import Tuple


class BaseExecutor(ABC):
    """
    Abstract base class for code execution strategies.
    """

    @abstractmethod
    def execute_script(
        self,
        workspace_path: str,
        venv_path: str,
        script_filename: str,
        timeout: int
    ) -> Tuple[int, str, str]:
        """
        Execute a single Python script within the specified environment.
        
        Args:
            workspace_path: Absolute path to the task workspace.
            venv_path: Absolute path to the task virtual environment.
            script_filename: Name of the script to execute (relative to workspace).
            timeout: Execution timeout in seconds.
            
        Returns:
            Tuple of (exit_code, stdout, stderr).
        """
        pass

    @abstractmethod
    def execute_tests(
        self,
        workspace_path: str,
        venv_path: str,
        test_dir: str,
        timeout: int
    ) -> Tuple[int, str, str]:
        """
        Run a pytest suite within the specified environment.
        
        Args:
            workspace_path: Absolute path to the task workspace.
            venv_path: Absolute path to the task virtual environment.
            test_dir: Directory containing tests (relative to workspace).
            timeout: Execution timeout in seconds.
            
        Returns:
            Tuple of (exit_code, stdout, stderr).
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the executor backend (e.g., Docker daemon) is available.
        
        Returns:
            True if available, False otherwise.
        """
        pass
