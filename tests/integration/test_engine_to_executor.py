"""
tests/integration/test_engine_to_executor.py
────────────────────────────────────────────
Integration tests for the handoff between engine/orchestrator and the real sandbox/executor.
"""
import os
import pytest
from unittest.mock import MagicMock, patch
from nexus.core.engine import TaskEngine
from nexus.core.context import SessionContext
from nexus.ai.router import IntentType, IntentResult
from nexus.core.task import TaskStatus


@pytest.fixture
def engine():
    """
    Real engine using a temporary workspace for sandbox testing.
    """
    ctx = SessionContext()
    # Initialize engine
    e = TaskEngine(ctx)
    yield e


@pytest.mark.integration
def test_generated_file_is_written_to_workspace(engine):
    """Verify that orchestrator-generated files are physically written to the workspace."""
    user_input = "Write files"
    intent_res = IntentResult(IntentType.TASK, 1.0, "Task", user_input)
    
    files = {
        "main.py": "print('hello')",
        "test_main.py": "def test_pass(): assert True"
    }
    
    with patch.object(engine.ai, "generate_plan", return_value={"description": "Test", "steps": []}), \
         patch.object(engine.ai, "generate_code", return_value=files), \
         patch("nexus.core.engine.validate_result", return_value=MagicMock(
             is_success=True, status="PASS", semantic_verdict="CORRECT",
             semantic_issues=[],
             stage1_result=MagicMock(returncode=0, stdout="", stderr="", timed_out=False)
         )):
        
        task = engine.run(user_input, intent_res)
        
        ws_path = task.workspace_path
        assert os.path.exists(os.path.join(ws_path, "main.py"))
        assert os.path.exists(os.path.join(ws_path, "test_main.py"))
        assert f"task_{task.id}" in ws_path


@pytest.mark.integration
def test_clean_code_passes_ast_gate_and_executes(engine):
    """Verify that safe code passes the security gate and runs successfully."""
    user_input = "Run hello"
    intent_res = IntentResult(IntentType.TASK, 1.0, "Task", user_input)
    
    files = {"main.py": "print('hello_from_nexus')"}
    
    with patch.object(engine.ai, "generate_plan", return_value={"description": "Test", "steps": []}), \
         patch.object(engine.ai, "generate_code", return_value=files), \
         patch("nexus.core.engine.validate_result", return_value=MagicMock(
             is_success=True, status="PASS", semantic_verdict="CORRECT",
             semantic_issues=[],
             stage1_result=MagicMock(returncode=0, stdout="", stderr="", timed_out=False)
         )):
        
        task = engine.run(user_input, intent_res)
        assert task.status == TaskStatus.DONE


@pytest.mark.integration
def test_forbidden_code_is_blocked_before_execution(engine):
    """Verify that the AST security gate stops malicious code before a process is spawned."""
    user_input = "Run evil"
    intent_res = IntentResult(IntentType.TASK, 1.0, "Task", user_input)
    
    # Attempt RCE via os.system
    files = {"main.py": "import os; os.system('echo pwned')"}
    
    with patch.object(engine.ai, "generate_plan", return_value={"description": "Test", "steps": []}), \
         patch.object(engine.ai, "generate_code", return_value=files), \
         patch.object(engine.ai, "generate_fix", return_value=files):
        
        task = engine.run(user_input, intent_res)
        
        assert task.status == TaskStatus.FAILED
        assert task.result.success is False
        assert "Security violation" in task.result.summary or \
               "Forbidden" in task.result.summary or \
               "blocked" in task.result.summary.lower()


@pytest.mark.integration
def test_failed_execution_triggers_fix_loop(engine):
    """Verify that a runtime failure triggers the auto-fix loop."""
    user_input = "Fix syntax"
    intent_res = IntentResult(IntentType.TASK, 1.0, "Task", user_input)
    
    plan = {
        "task_type": "script",
        "description": "Test",
        "steps": [],
        "files_to_generate": ["main.py"],
        "entry_point": "main.py",
        "test_command": "python3 main.py",
        "install_command": "",
    }
    
    fail_result = MagicMock(
        returncode=1, stdout="", stderr="RuntimeError: intentional",
        timed_out=False, security_blocked=False
    )
    pass_result = MagicMock(
        returncode=0, stdout="fixed", stderr="",
        timed_out=False, security_blocked=False
    )
    
    fail_vr = MagicMock(
        is_success=False, status="FAIL", semantic_verdict=None,
        semantic_issues=[], semantic_reason=None,
        stage1_result=fail_result
    )
    pass_vr = MagicMock(
        is_success=True, status="PASS", semantic_verdict="CORRECT",
        semantic_issues=[],
        stage1_result=pass_result
    )
    
    with patch.object(engine.ai, "generate_plan", return_value=plan), \
         patch.object(engine.ai, "generate_code", 
                      return_value={"main.py": "raise RuntimeError('intentional')"}), \
         patch.object(engine.ai, "generate_fix",
                      return_value={"main.py": "print('fixed')"}), \
         patch("nexus.core.engine.validate_result",
               side_effect=[
                   MagicMock(is_success=True),  # install
                   fail_vr,                      # first test run
                   pass_vr,                      # after fix
               ]):
        
        task = engine.run(user_input, intent_res)
        
        assert task.status == TaskStatus.DONE
        assert engine.ai.generate_fix.called
        assert task.result.iterations_used >= 1


@pytest.mark.integration
def test_fix_loop_terminates_at_max_retries(engine):
    """Verify that the engine gives up after hitting the maximum retry cap."""
    user_input = "Fail forever"
    intent_res = IntentResult(IntentType.TASK, 1.0, "Task", user_input)
    
    plan = {
        "task_type": "script",
        "description": "Test",
        "steps": [],
        "files_to_generate": ["main.py"],
        "entry_point": "main.py",
        "test_command": "python3 main.py",
        "install_command": "",
    }
    
    exec_result = MagicMock(
        returncode=1, stdout="", stderr="AssertionError: always fails",
        timed_out=False, security_blocked=False
    )
    fail_vr = MagicMock(
        is_success=False, status="FAIL",
        semantic_verdict=None, semantic_reason=None, semantic_issues=[],
        stage1_result=exec_result
    )
    # Make security_blocked explicitly False on stage1_result
    fail_vr.stage1_result.security_blocked = False
    fail_vr.stage1_result.timed_out = False
    
    with patch.object(engine.ai, "generate_plan", return_value=plan), \
         patch.object(engine.ai, "generate_code",
                      return_value={"main.py": "assert False, 'always fails'"}), \
         patch.object(engine.ai, "generate_fix",
                      return_value={"main.py": "assert False, 'still fails'"}), \
         patch("nexus.core.engine.validate_result",
               return_value=fail_vr):
        
        task = engine.run(user_input, intent_res)
        
        assert task.status == TaskStatus.FAILED
        assert task.result.iterations_used >= 1
