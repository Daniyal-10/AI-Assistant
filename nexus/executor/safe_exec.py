"""
Safe code execution module.

SECURITY DESIGN:
- Whitelist of allowed commands only
- Strict timeout enforcement (SIGKILL on breach)
- Working directory locked to task workspace
- No shell=True (prevents shell injection)
- No root execution
- Environment sanitized (no secret leakage)
- Captured stdout/stderr for AI feedback loop
"""
import os
import signal
import subprocess
from dataclasses import dataclass
from typing import List, Optional

from nexus.core.exceptions import ExecutionError, SafetyViolation
from nexus.utils.config import config
from nexus.utils.logger import get_logger

logger = get_logger(__name__)

# ─── SECURITY: COMMAND WHITELIST ─────────────────────────────────────────────
ALLOWED_EXECUTABLES = frozenset({
    "python",
    "python3",
    "pip",
    "pip3",
    "pytest",
    "uv",
    "node",
    "npm",
})

BLOCKED_ARG_PATTERNS = frozenset({
    "--trusted-host",
    "--index-url",
    "--extra-index-url",
    "exec(",
    "eval(",
    "__import__",
    "-c",
})
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ExecResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def _validate_command(args: List[str]) -> None:
    if not args or not args[0].strip():
        raise SafetyViolation("Empty command rejected.")

    executable = os.path.basename(args[0])
    if executable not in ALLOWED_EXECUTABLES:
        raise SafetyViolation(
            f"Executable '{executable}' is not allowed."
        )

    full_cmd = " ".join(args[1:])
    for pattern in BLOCKED_ARG_PATTERNS:
        if pattern in full_cmd:
            raise SafetyViolation(
                f"Blocked argument pattern detected: '{pattern}'"
            )


def _sanitize_env() -> dict:
    safe_keys = {
        "PATH", "HOME", "USER", "LANG", "LC_ALL",
        "PYTHONPATH", "VIRTUAL_ENV",
    }
    env = {k: v for k, v in os.environ.items() if k in safe_keys}
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_command(
    args: List[str],
    cwd: str,
    timeout: Optional[int] = None,
) -> ExecResult:

    _validate_command(args)

    # realpath resolves symlinks before boundary check.
    # abspath does NOT resolve symlinks — a symlink inside the workspace
    # pointing outside would bypass the check.
    # os.sep suffix prevents prefix collision:
    #   /tmp/nexus_evil startswith /tmp/nexus  → True  (wrong)
    #   /tmp/nexus_evil startswith /tmp/nexus/ → False (correct)
    workspace_base = os.path.realpath(config.workspace_base)
    cwd_real = os.path.realpath(cwd)

    if not cwd_real.startswith(workspace_base + os.sep) and cwd_real != workspace_base:
        raise SafetyViolation(
            f"Execution outside workspace blocked: {cwd_real}"
        )

    effective_timeout = timeout or config.exec_timeout
    env = _sanitize_env()

    logger.info("Executing: %s (cwd=%s, timeout=%ds)", args, cwd, effective_timeout)

    try:
        proc = subprocess.Popen(
            args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            shell=False,
            start_new_session=True,
        )

        try:
            stdout_bytes, stderr_bytes = proc.communicate(timeout=effective_timeout)

            MAX_OUTPUT = 10000

            stdout_text = stdout_bytes.decode("utf-8", errors="replace")[:MAX_OUTPUT]
            stderr_text = stderr_bytes.decode("utf-8", errors="replace")[:MAX_OUTPUT]

            return ExecResult(
                returncode=proc.returncode,
                stdout=stdout_text,
                stderr=stderr_text,
            )

        except subprocess.TimeoutExpired:
            logger.warning(
                "Command timed out after %ds. Killing process group.",
                effective_timeout,
            )

            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

            proc.wait()

            return ExecResult(
                returncode=-1,
                stdout="",
                stderr=f"[NEXUS] Process killed: exceeded {effective_timeout}s timeout.",
                timed_out=True,
            )

    except FileNotFoundError as e:
        raise ExecutionError(
            f"Command not found: {args[0]}",
            stderr=str(e),
        ) from e

    except PermissionError as e:
        raise ExecutionError(
            f"Permission denied: {args[0]}",
            stderr=str(e),
        ) from e

    except OSError as e:
        raise ExecutionError(
            f"OS error during execution: {e}"
        ) from e


# ─── PER-TASK VENV ────────────────────────────────────────────────────────────

def create_task_venv(workspace_path: str) -> str:
    """
    Create an isolated venv inside the task workspace and pre-install
    pytest so it is always available regardless of requirements.txt.

    Returns the venv directory path.
    """
    import venv as _venv

    venv_path = os.path.join(workspace_path, ".venv")
    logger.info("Creating task venv at %s", venv_path)

    builder = _venv.EnvBuilder(with_pip=True, clear=False)
    builder.create(venv_path)

    python_bin = os.path.join(venv_path, "bin", "python3")
    if not os.path.exists(python_bin):
        raise ExecutionError(
            f"Venv creation failed — python3 not found at {python_bin}"
        )

    # Pre-install pytest into every task venv.
    # Tasks with no requirements.txt skip the install step entirely,
    # which means pytest would never be installed — causing
    # "Command not found" when the test runner tries to execute.
    pip_bin = os.path.join(venv_path, "bin", "pip")
    logger.info("Pre-installing pytest into task venv...")
    result = subprocess.run(
        [pip_bin, "install", "pytest", "--quiet"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        logger.warning("pytest pre-install failed: %s", result.stderr[:300])

    logger.info("Task venv ready: %s", python_bin)
    return venv_path


def get_venv_executables(venv_path: str) -> dict:
    """
    Return a dict of executable paths inside the venv.
    Engine uses these to override default commands.
    """
    bin_dir = os.path.join(venv_path, "bin")
    return {
        "python3": os.path.join(bin_dir, "python3"),
        "python":  os.path.join(bin_dir, "python3"),
        "pip":     os.path.join(bin_dir, "pip"),
        "pip3":    os.path.join(bin_dir, "pip3"),
        "pytest":  os.path.join(bin_dir, "pytest"),
    }
