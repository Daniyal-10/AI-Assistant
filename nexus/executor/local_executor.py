"""
nexus/executor/local_executor.py
───────────────────────────────
Local execution backend for NEXUS. Runs code directly on the host OS
using subprocesses with AST-based security gates.
"""
import os
from typing import Tuple

from nexus.executor.exec_interface import BaseExecutor
from nexus.executor.safe_exec import run_command, scan_for_forbidden_patterns


class LocalExecutor(BaseExecutor):
    """
    Executor that runs code directly on the host machine.
    Development/Fallback mode.
    """

    def execute_script(
        self,
        workspace_path: str,
        venv_path: str,
        script_filename: str,
        timeout: int
    ) -> Tuple[int, str, str]:
        """
        Execute script on host after running static analysis.
        """
        full_path = os.path.join(workspace_path, script_filename)
        
        # Security Gate
        scan_for_forbidden_patterns(full_path, workspace_path)
        
        # Execution
        # We use the python binary from the provided venv
        python_bin = os.path.join(venv_path, "bin", "python3")
        if not os.path.exists(python_bin):
            # Fallback to system python if venv bin is missing (should not happen in prod)
            python_bin = "python3"

        res = run_command([python_bin, script_filename], workspace_path, timeout)
        return (res.returncode, res.stdout, res.stderr)

    def execute_tests(
        self,
        workspace_path: str,
        venv_path: str,
        test_dir: str,
        timeout: int
    ) -> Tuple[int, str, str]:
        """
        Run pytest on host using venv pytest binary.
        """
        pytest_bin = os.path.join(venv_path, "bin", "pytest")
        if not os.path.exists(pytest_bin):
            pytest_bin = "pytest"

        res = run_command(
            [pytest_bin, test_dir, "-v", "--tb=short"], 
            workspace_path, 
            timeout
        )
        return (res.returncode, res.stdout, res.stderr)

    def is_available(self) -> bool:
        """Local execution is always available on the host OS."""
        return True
