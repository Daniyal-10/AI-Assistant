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
import ast
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Any

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
    security_blocked: bool = False

    @property
    def success(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.security_blocked


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


def scan_for_forbidden_patterns(filepath: str, workspace_path: str) -> None:
    """
    Perform deterministic static analysis on generated Python code using AST walking.
    Blocks dangerous operations (RCE, networking, unsafe filesystem access) before execution.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()
    except FileNotFoundError:
        raise ExecutionError(f"File not found: {filepath}")

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        err_msg = f"Failed to parse Python code in {os.path.basename(filepath)}: {e.msg} at line {e.lineno}"
        logger.warning("Static analysis gate failed: %s", err_msg)
        raise ExecutionError(err_msg)
    except Exception as e:
        err_msg = f"Failed to parse Python code in {os.path.basename(filepath)}: {e}"
        logger.warning("Static analysis gate failed: %s", err_msg)
        raise ExecutionError(err_msg)

    ws_real = Path(workspace_path).resolve()

    def _get_func_name(node: Any) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{_get_func_name(node.value)}.{node.attr}"
        return ""

    def _report_violation(name: str, line: int):
        if name.startswith("open(") or name.startswith("shutil."):
            msg = f"Illegal file access: '{name}' at line {line} in {os.path.basename(filepath)}"
        else:
            msg = f"Forbidden function call: {name} at line {line} in {os.path.basename(filepath)}"
        logger.warning("Security violation: %s", msg)
        raise ExecutionError(msg)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = _get_func_name(node.func)
            
            # Block getattr(os, "system") style obfuscation
            if isinstance(node.func, ast.Name) and node.func.id == "getattr":
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    _DANGEROUS_ATTRS = {
                        "system", "popen", "execv", "execve", "execvp",
                        "run", "Popen", "call", "check_output",
                        "connect", "urlopen",
                    }
                    if node.args[1].value in _DANGEROUS_ATTRS:
                        raise ExecutionError(
                            f"Forbidden function call: getattr obfuscation of "
                            f"'{node.args[1].value}' at line {node.lineno} "
                            f"in {os.path.basename(filepath)}"
                        )


            # 1. RCE & Dynamic Execution
            if func_name in {"eval", "exec", "__import__"}:
                # Block if first arg is not a constant/literal string
                if node.args and not isinstance(node.args[0], (ast.Constant, ast.Str)):
                    _report_violation(func_name, node.lineno)

            # 2. OS Shell/Exec
            elif func_name in {"os.system", "os.popen", "os.execv", "os.execve", "os.execvp"}:
                _report_violation(func_name, node.lineno)
            
            # 3. Subprocess
            elif any(func_name == f"subprocess.{m}" for m in ["run", "Popen", "call", "check_output"]):
                _report_violation(func_name, node.lineno)

            # 4. Networking
            elif any(func_name == f"socket.{m}" for m in ["socket", "connect", "create_connection"]):
                raise ExecutionError(f"Forbidden import: socket detected at line {node.lineno}")
            elif func_name.startswith("requests.") or func_name == "requests.request":
                if any(m in func_name for m in ["get", "post", "put", "delete", "request"]):
                    raise ExecutionError(f"Forbidden import: requests detected at line {node.lineno}")
            elif func_name in {"urllib.request.urlopen", "urllib.request.urlretrieve"}:
                _report_violation(func_name, node.lineno)

            # 5. Filesystem Boundary Checks
            elif func_name in {"open", "shutil.rmtree", "shutil.move", "shutil.copy"}:
                if node.args:
                    path_arg = node.args[0]
                    # Check for literal path
                    if isinstance(path_arg, (ast.Constant, ast.Str)):
                        path_val = path_arg.value if hasattr(path_arg, "value") else path_arg.s
                        try:
                            # Resolve path relative to workspace
                            target = (ws_real / str(path_val)).resolve()
                            if not str(target).startswith(str(ws_real)):
                                _report_violation(f"{func_name}({path_val})", node.lineno)
                        except Exception:
                            _report_violation(f"{func_name}({path_val})", node.lineno)
                    else:
                        # Dynamic path argument — too risky to allow for these sensitive functions
                        _report_violation(f"{func_name}(dynamic_path)", node.lineno)


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

    # --- SECURITY: PRE-EXECUTION STATIC ANALYSIS GATE ---
    # Scan every Python file in the workspace before execution
    for entry in os.scandir(cwd):
        if entry.is_file() and entry.name.endswith(".py"):
            scan_for_forbidden_patterns(entry.path, workspace_base)
    # ---------------------------------------------------

    logger.info("Executing: %s (cwd=%s, timeout=%ds)", args, cwd, effective_timeout)

    popen_kwargs = {
        "cwd": cwd,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "env": env,
        "shell": False,
        "text": True,
    }

    if os.name != "nt":
        # Unix: create process group for full cleanup
        popen_kwargs["preexec_fn"] = os.setsid
    else:
        # Windows: create new process group
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    try:
        proc = subprocess.Popen(args, **popen_kwargs)

        try:
            stdout, stderr = proc.communicate(timeout=effective_timeout)

            # Enforce output limits even in text mode
            MAX_OUTPUT = 10000
            stdout = (stdout or "")[:MAX_OUTPUT]
            stderr = (stderr or "")[:MAX_OUTPUT]

            return ExecResult(
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr,
            )

        except subprocess.TimeoutExpired:
            logger.warning(
                "Command timed out after %ds. Killing process group.",
                effective_timeout,
            )

            if os.name != "nt":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                # Windows: taskkill is reliable for group cleanup
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], 
                               capture_output=True)

            try:
                # Cleanup pipes and collect any remaining output
                stdout, stderr = proc.communicate(timeout=1)
                stdout = (stdout or "")[:1000]
                stderr = (stderr or "")[:1000]
            except Exception:
                stdout, stderr = "", ""

            return ExecResult(
                returncode=-1,
                stdout=stdout,
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

    if os.name == "nt":
        python_bin = os.path.join(venv_path, "Scripts", "python.exe")
        pip_bin = os.path.join(venv_path, "Scripts", "pip.exe")
    else:
        python_bin = os.path.join(venv_path, "bin", "python3")
        pip_bin = os.path.join(venv_path, "bin", "pip")

    if not os.path.exists(python_bin):
        raise ExecutionError(
            f"Venv creation failed — python executable not found at {python_bin}"
        )

    # Pre-install pytest into every task venv.
    # Tasks with no requirements.txt skip the install step entirely,
    # which means pytest would never be installed — causing
    # "Command not found" when the test runner tries to execute.
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
    if os.name == "nt":
        bin_dir = os.path.join(venv_path, "Scripts")
        python_exe = os.path.join(bin_dir, "python.exe")
        pip_exe = os.path.join(bin_dir, "pip.exe")
        pytest_exe = os.path.join(bin_dir, "pytest.exe")
        return {
            "python3": python_exe,
            "python":  python_exe,
            "pip":     pip_exe,
            "pip3":    pip_exe,
            "pytest":  pytest_exe,
        }
    else:
        bin_dir = os.path.join(venv_path, "bin")
        return {
            "python3": os.path.join(bin_dir, "python3"),
            "python":  os.path.join(bin_dir, "python3"),
            "pip":     os.path.join(bin_dir, "pip"),
            "pip3":    os.path.join(bin_dir, "pip3"),
            "pytest":  os.path.join(bin_dir, "pytest"),
        }
