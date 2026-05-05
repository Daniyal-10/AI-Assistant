#!/usr/bin/env python3
"""
NEXUS Phase 1 — Assessment Suite (corrected)
=============================================
Tests written against the ACTUAL code, not assumed class names.

Run from project root:
    python assess_phase1.py
"""

import os
import sys
import time
import zipfile
import tempfile
import traceback
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

results: list[dict] = []


def run_test(name: str, fn: Callable) -> None:
    print(f"\n{'─' * 60}")
    print(f"TEST: {name}")
    print(f"{'─' * 60}")
    try:
        fn()
        results.append({"name": name, "status": "PASS", "error": None})
        print("✅  PASS")
    except AssertionError as e:
        results.append({"name": name, "status": "FAIL", "error": str(e)})
        print(f"❌  FAIL — {e}")
    except Exception as e:
        results.append({"name": name, "status": "ERROR", "error": f"{type(e).__name__}: {e}"})
        print(f"💥  ERROR — {type(e).__name__}: {e}")
        traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# T01 — Task state machine: valid forward path
# ══════════════════════════════════════════════════════════════════════════════
def test_task_state_machine_happy_path():
    from nexus.core.task import Task, TaskStatus

    task = Task(raw_input="state machine test")
    assert task.status == TaskStatus.PENDING

    for step in [
        TaskStatus.PLANNING,
        TaskStatus.GENERATING,
        TaskStatus.EXECUTING,
        TaskStatus.VALIDATING,
        TaskStatus.DONE,
    ]:
        task.transition(step)

    assert task.status == TaskStatus.DONE


# ══════════════════════════════════════════════════════════════════════════════
# T02 — Task state machine: illegal transitions raise ValueError
# ══════════════════════════════════════════════════════════════════════════════
def test_task_state_machine_illegal_transitions():
    from nexus.core.task import Task, TaskStatus

    # Terminal state — no exit from DONE
    task = Task(raw_input="terminal test")
    for step in [TaskStatus.PLANNING, TaskStatus.GENERATING,
                 TaskStatus.EXECUTING, TaskStatus.VALIDATING, TaskStatus.DONE]:
        task.transition(step)

    try:
        task.transition(TaskStatus.PLANNING)
        raise AssertionError("Expected ValueError from DONE — none raised")
    except ValueError:
        pass

    # Illegal skip: PENDING → EXECUTING
    task2 = Task(raw_input="illegal skip")
    try:
        task2.transition(TaskStatus.EXECUTING)
        raise AssertionError("Expected ValueError for illegal skip — none raised")
    except ValueError:
        pass

    # Fix loop: VALIDATING ↔ FIXING must both work
    task3 = Task(raw_input="fix loop test")
    task3.transition(TaskStatus.PLANNING)
    task3.transition(TaskStatus.GENERATING)
    task3.transition(TaskStatus.EXECUTING)
    task3.transition(TaskStatus.VALIDATING)
    task3.transition(TaskStatus.FIXING)
    task3.transition(TaskStatus.VALIDATING)  # was the missing bug
    assert task3.status == TaskStatus.VALIDATING


# ══════════════════════════════════════════════════════════════════════════════
# T03 — Task metadata: id, raw_input, created_at, uniqueness
# ══════════════════════════════════════════════════════════════════════════════
def test_task_metadata():
    from nexus.core.task import Task

    task = Task(raw_input="metadata check")
    assert task.id, "Task must have a non-empty id"
    assert isinstance(task.id, str)
    assert task.raw_input == "metadata check"
    assert task.created_at is not None

    task2 = Task(raw_input="second task")
    assert task.id != task2.id, "Task ids must be unique"


# ══════════════════════════════════════════════════════════════════════════════
# T04 — FAILED is always reachable from any state
# ══════════════════════════════════════════════════════════════════════════════
def test_task_failed_always_reachable():
    from nexus.core.task import Task, TaskStatus

    for starting_status in [
        TaskStatus.PENDING, TaskStatus.PLANNING, TaskStatus.GENERATING,
        TaskStatus.EXECUTING, TaskStatus.VALIDATING, TaskStatus.FIXING,
    ]:
        task = Task(raw_input=f"fail from {starting_status.name}")
        task.status = starting_status
        task.transition(TaskStatus.FAILED)
        assert task.status == TaskStatus.FAILED, (
            f"FAILED must be reachable from {starting_status.name}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# T05 — Executor whitelist: rejects non-whitelisted binaries
# ══════════════════════════════════════════════════════════════════════════════
def test_executor_whitelist():
    from nexus.executor.safe_exec import run_command
    from nexus.core.exceptions import SafetyViolation

    workspace_base = Path.home() / ".nexus" / "workspaces"
    workspace_base.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=workspace_base) as cwd:
        try:
            run_command(["curl", "http://example.com"], cwd=cwd)
            raise AssertionError("curl should have been blocked — it was not")
        except SafetyViolation:
            pass
        except AssertionError:
            raise


# ══════════════════════════════════════════════════════════════════════════════
# T06 — No shell=True in executable code (comments excluded)
# ══════════════════════════════════════════════════════════════════════════════
def test_no_shell_true_in_code():
    safe_exec_path = PROJECT_ROOT / "nexus" / "executor" / "safe_exec.py"
    source = safe_exec_path.read_text()

    # Strip both # comments AND docstring lines (start with - or *)
    code_lines = [
        l for l in source.splitlines()
        if not l.strip().startswith("#")
        and not l.strip().startswith("-")
        and not l.strip().startswith("*")
        and not l.strip().startswith('"""')
        and not l.strip().startswith("'''")
    ]
    code_only = "\n".join(code_lines)

    assert "shell=True" not in code_only, (
        "SECURITY VIOLATION: shell=True found in executable code in safe_exec.py"
    )
    assert "os.system(" not in code_only, (
        "SECURITY VIOLATION: os.system() found in safe_exec.py"
    )

# ══════════════════════════════════════════════════════════════════════════════
# T07 — Workspace: path traversal blocked
# ══════════════════════════════════════════════════════════════════════════════
def test_workspace_path_traversal():
    from nexus.executor.workspace import Workspace

    ws = Workspace(task_id="traversal-test")
    ws.create()

    try:
        ws.write_files({"../../etc/passwd": "injected"})
        escaped = Path("/etc/passwd")
        if escaped.exists():
            assert "injected" not in escaped.read_text(), (
                "SECURITY VIOLATION: traversal write reached /etc/passwd"
            )
    except (PermissionError, ValueError, OSError):
        pass  # correct — blocked
    finally:
        ws.cleanup()


# ══════════════════════════════════════════════════════════════════════════════
# T08 — Workspace prefix collision: startswith needs os.sep guard
# Known bug: workspace.py uses startswith(base) without + os.sep
# ══════════════════════════════════════════════════════════════════════════════
def test_workspace_prefix_collision_in_source():
    workspace_path = PROJECT_ROOT / "nexus" / "executor" / "workspace.py"
    source = workspace_path.read_text()

    code_lines = [l for l in source.splitlines() if not l.strip().startswith("#")]
    code_only = "\n".join(code_lines)

    # Check the full line, not a regex substring — the regex [^)]+ stops at
    # the first ) inside resolve(), missing the + os.sep that follows.
    startswith_lines = [l for l in code_only.splitlines() if "startswith(" in l]

    unguarded = [
        l.strip() for l in startswith_lines
        if "os.sep" not in l and '+ "/"' not in l and "+ '/'" not in l
    ]

    assert not unguarded, (
        f"PREFIX COLLISION BUG: startswith() without os.sep guard in workspace.py\n"
        f"Unguarded lines: {unguarded}\n"
        f"Fix: startswith(str(self.base.resolve()) + os.sep)"
    )

# ══════════════════════════════════════════════════════════════════════════════
# T09 — Parser: 3-strategy JSON extraction
# ══════════════════════════════════════════════════════════════════════════════
def test_parser_json_extraction():
    from nexus.ai.parser import extract_json

    # Strategy 1: clean JSON
    r = extract_json('{"files": {"main.py": "print(1)"}, "test_file": "t.py"}')
    assert r is not None, "Strategy 1 (clean JSON) failed"
    assert "files" in r

    # Strategy 2: markdown fenced
    r = extract_json('```json\n{"files": {"x.py": "x=1"}, "test_file": "t.py"}\n```')
    assert r is not None, "Strategy 2 (fenced JSON) failed"
    assert "files" in r

    # Strategy 3: embedded in prose
    r = extract_json('Here: {"files": {"x.py": "x=1"}, "test_file": "t.py"} done.')
    assert r is not None, "Strategy 3 (embedded JSON) failed"
    assert "files" in r

    # Garbage → None, not raise
    assert extract_json("not json at all") is None, "Expected None for garbage"
    assert extract_json("") is None, "Expected None for empty string"


# ══════════════════════════════════════════════════════════════════════════════
# T10 — Config: _validate_config exits on bad values
# ══════════════════════════════════════════════════════════════════════════════
def test_config_validation_rejects_bad_values():
    from nexus.utils.config import NexusConfig, _validate_config

    bad_cfg = NexusConfig.__new__(NexusConfig)
    object.__setattr__(bad_cfg, "exec_timeout", -1)  # invalid
    object.__setattr__(bad_cfg, "ollama_timeout", 120)
    object.__setattr__(bad_cfg, "max_fix_iterations", 3)
    object.__setattr__(bad_cfg, "workspace_base", str(Path.home() / ".nexus" / "workspaces"))
    object.__setattr__(bad_cfg, "ollama_base_url", "http://127.0.0.1:11434")
    object.__setattr__(bad_cfg, "ollama_code_model", "qwen2.5-coder:7b")
    object.__setattr__(bad_cfg, "ollama_reason_model", "llama3.1:8b")
    object.__setattr__(bad_cfg, "allowed_telegram_users", [])
    object.__setattr__(bad_cfg, "telegram_bot_token", "")

    try:
        _validate_config(bad_cfg)
        raise AssertionError("Config did not exit when exec_timeout=-1")
    except SystemExit:
        pass  # correct
    except AssertionError:
        raise


# ══════════════════════════════════════════════════════════════════════════════
# T11 — Workspace: archive() produces a valid zip
# ══════════════════════════════════════════════════════════════════════════════
def test_workspace_zip_delivery():
    from nexus.executor.workspace import Workspace

    ws = Workspace(task_id="zip-assess-test")
    ws.create()
    zip_path = None

    try:
        ws.write_files({"output.py": "print('hello')", "README.md": "# test"})
        zip_path = Path(ws.archive())

        assert zip_path.exists(), f"Zip not found: {zip_path}"
        assert zip_path.suffix == ".zip"
        assert zipfile.is_zipfile(zip_path), "Not a valid zip archive"

        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert any("output.py" in n for n in names), (
                f"output.py missing from zip. Contents: {names}"
            )
    finally:
        ws.cleanup()
        if zip_path and zip_path.exists():
            zip_path.unlink()


# ══════════════════════════════════════════════════════════════════════════════
# T12 — Exception hierarchy: NexusBaseException + all subclasses
# ══════════════════════════════════════════════════════════════════════════════
def test_exception_hierarchy():
    from nexus.core import exceptions as exc

    assert hasattr(exc, "NexusBaseException"), "NexusBaseException missing"
    base = exc.NexusBaseException

    required = [
        "TaskPlanningError",
        "TaskGenerationError",
        "ExecutionError",
        "ValidationError",
        "SafetyViolation",
        "MaxRetriesExceeded",
    ]

    missing = [n for n in required if not hasattr(exc, n)]
    assert not missing, f"Missing exception classes: {missing}"

    not_subclass = [n for n in required if not issubclass(getattr(exc, n), base)]
    assert not not_subclass, f"Not subclassing NexusBaseException: {not_subclass}"

    # ExecutionError must carry stdout/stderr fields
    err = exc.ExecutionError("test", stdout="out", stderr="err")
    assert err.stdout == "out"
    assert err.stderr == "err"


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "═" * 60)
    print("  NEXUS — Phase 1 Assessment Suite (corrected)")
    print("═" * 60)
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Python       : {sys.version.split()[0]}")
    print(f"  Run at       : {time.strftime('%Y-%m-%d %H:%M:%S')}")

    run_test("T01 — State machine: happy path",          test_task_state_machine_happy_path)
    run_test("T02 — State machine: illegal transitions",  test_task_state_machine_illegal_transitions)
    run_test("T03 — Task metadata integrity",             test_task_metadata)
    run_test("T04 — FAILED reachable from any state",    test_task_failed_always_reachable)
    run_test("T05 — Executor whitelist enforcement",      test_executor_whitelist)
    run_test("T06 — No shell=True in executor code",      test_no_shell_true_in_code)
    run_test("T07 — Workspace path traversal blocked",    test_workspace_path_traversal)
    run_test("T08 — Workspace prefix collision bug",      test_workspace_prefix_collision_in_source)
    run_test("T09 — Parser 3-strategy JSON extraction",   test_parser_json_extraction)
    run_test("T10 — Config fail-fast on bad values",      test_config_validation_rejects_bad_values)
    run_test("T11 — Workspace zip delivery",              test_workspace_zip_delivery)
    run_test("T12 — Exception hierarchy completeness",    test_exception_hierarchy)

    passed  = sum(1 for r in results if r["status"] == "PASS")
    failed  = sum(1 for r in results if r["status"] == "FAIL")
    errored = sum(1 for r in results if r["status"] == "ERROR")
    total   = len(results)

    print("\n" + "═" * 60)
    print("  RESULTS")
    print("═" * 60)
    for r in results:
        icon = {"PASS": "✅", "FAIL": "❌", "ERROR": "💥"}[r["status"]]
        print(f"  {icon}  {r['name']}")
        if r["error"]:
            for line in r["error"].splitlines():
                print(f"       → {line}")

    print(f"\n  Score : {passed}/{total}")
    print(f"  Pass  : {passed}  |  Fail : {failed}  |  Error : {errored}")

    if passed == total:
        print("\n  🏆  VERDICT: Phase 1 FULLY VALIDATED — ready for Phase 2")
    elif passed >= total * 0.8:
        print("\n  ⚠️   VERDICT: Phase 1 MOSTLY COMPLETE — fix failures before Phase 2")
    elif passed >= total * 0.5:
        print("\n  🔴  VERDICT: Phase 1 PARTIAL — significant gaps remain")
    else:
        print("\n  💀  VERDICT: Phase 1 NOT READY — foundational issues present")

    print("═" * 60 + "\n")
    sys.exit(0 if failed == 0 and errored == 0 else 1)


if __name__ == "__main__":
    main()
