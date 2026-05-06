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
def engine(tmp_path):
    """
    Real engine using a temporary workspace for sandbox testing.
    """
    # Override config to use tmp_path
    with patch("nexus.utils.config.config.workspace_base", str(tmp_path)):
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
        assert "FAILED" in task.result.summary


@pytest.mark.integration
def test_failed_execution_triggers_fix_loop(engine):
    """Verify that a runtime or syntax failure triggers the auto-fix loop."""
    user_input = "Fix syntax"
    intent_res = IntentResult(IntentType.TASK, 1.0, "Task", user_input)
    
    # Mocking code gen: first returns runtime error, then returns fix
    engine.ai.generate_code = MagicMock(return_value={"main.py": "raise RuntimeError('intentional failure')"})
    engine.ai.generate_fix = MagicMock(return_value={"main.py": "print('fixed')"})
    
    # Mock validator results for: 1. Install (PASS), 2. Test First Run (FAIL), 3. Test Fix Run (PASS)
    install_pass = MagicMock(is_success=True, status="PASS")
    test_fail = MagicMock(is_success=False, status="FAIL")
    test_pass = MagicMock(is_success=True, status="PASS", semantic_verdict="CORRECT", semantic_issues=[])
    
    with patch.object(engine.ai, "generate_plan", return_value={"description": "Test", "steps": [], "test_command": "python3 main.py"}), \
         patch("nexus.core.engine.validate_result", side_effect=[install_pass, test_fail, test_pass]):
        
        task = engine.run(user_input, intent_res)
        
        assert task.status == TaskStatus.DONE
        assert engine.ai.generate_fix.called
        assert task.result.iterations_used >= 1


@pytest.mark.integration
def test_fix_loop_terminates_at_max_retries(engine):
    """Verify that the engine gives up after hitting the maximum retry cap."""
    user_input = "Fail forever"
    intent_res = IntentResult(IntentType.TASK, 1.0, "Task", user_input)
    
    # Persistent failure
    files = {"main.py": "raise Exception('Fail')"}
    
    with patch.object(engine.ai, "generate_plan", return_value={"description": "Test", "steps": []}), \
         patch.object(engine.ai, "generate_code", return_value=files), \
         patch.object(engine.ai, "generate_fix", return_value=files), \
         patch("nexus.core.engine.validate_result", return_value=MagicMock(
             is_success=False, status="FAIL", 
             stage1_result=MagicMock(returncode=1, stdout="", stderr="Error", timed_out=False),
             semantic_verdict=None, semantic_issues=[]
         )):
        
        task = engine.run(user_input, intent_res)
        
        assert task.status == TaskStatus.FAILED
        assert task.result.iterations_used >= 1
        # No unhandled exception should reach here
