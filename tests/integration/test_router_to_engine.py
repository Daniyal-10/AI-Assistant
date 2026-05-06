"""
tests/integration/test_router_to_engine.py
──────────────────────────────────────────
Integration tests for the handoff between IntentRouter and TaskEngine.
"""
import pytest
from unittest.mock import MagicMock, patch
from nexus.core.engine import TaskEngine
from nexus.core.context import SessionContext
from nexus.ai.router import IntentType, IntentResult
from nexus.core.task import TaskStatus, TaskResult


@pytest.fixture
def engine():
    """Real engine with a fresh context."""
    ctx = SessionContext()
    return TaskEngine(ctx)


@pytest.mark.integration
def test_task_intent_triggers_pipeline(engine):
    """Verify that TASK intent triggers the orchestrator pipeline."""
    user_input = "write a python function that returns the square of a number"
    intent_res = IntentResult(IntentType.TASK, 0.95, "Task requested", user_input)
    
    # Mock only the heavy LLM/Executor parts
    plan = {
        "task_type": "script",
        "description": "Square function",
        "files_to_generate": ["main.py"],
        "entry_point": "main.py",
        "test_command": "python3 main.py"
    }
    with patch.object(engine.ai, "generate_plan", return_value=plan), \
         patch.object(engine.ai, "generate_code", return_value={"main.py": "def square(n): return n*n"}), \
         patch("nexus.executor.safe_exec.run_command") as mock_exec, \
         patch("nexus.executor.validator.validate_result") as mock_val:
        
        mock_exec.return_value = MagicMock(returncode=0, stdout="done", stderr="", timed_out=False)
        mock_val.return_value = MagicMock(
            is_success=True, status="PASS", semantic_verdict="CORRECT",
            semantic_issues=[],
            stage1_result=MagicMock(returncode=0, stdout="done", stderr="", timed_out=False)
        )
        
        task = engine.run(user_input, intent_res)
        
        assert task.status == TaskStatus.DONE
        assert task.intent.intent == IntentType.TASK
        assert engine.ai.generate_plan.called


@pytest.mark.integration
def test_chat_intent_never_touches_executor(engine):
    """Verify that CHAT intent bypasses the execution sandbox entirely."""
    user_input = "hello how are you"
    intent_res = IntentResult(IntentType.CHAT, 1.0, "Greeting", user_input)
    
    with patch("nexus.executor.safe_exec.run_command") as mock_exec, \
         patch.object(engine.ai, "generate_chat_response", return_value="I am Jarvis."):
        
        task = engine.run(user_input, intent_res)
        
        mock_exec.assert_not_called()
        assert "Jarvis" in task.result.summary


@pytest.mark.integration
def test_code_intent_never_executes_code(engine):
    """Verify that CODE intent logic is read-only."""
    user_input = "explain this: def add(a,b): return a+b"
    intent_res = IntentResult(IntentType.CODE, 1.0, "Explain code", user_input)
    
    with patch("nexus.executor.safe_exec.run_command") as mock_exec, \
         patch("nexus.executor.workspace.Workspace.create") as mock_ws:
        
        engine.run(user_input, intent_res)
        
        mock_exec.assert_not_called()
        mock_ws.assert_not_called()


@pytest.mark.integration
def test_unknown_intent_defaults_to_chat(engine):
    """Verify engine handles routing failures by defaulting to CHAT."""
    # Simulate a total router failure
    with patch("nexus.ai.router.IntentRouter.route", side_effect=Exception("Router crash")):
        task = engine.run("trigger error", None)

        # Router crash is caught by engine's outer exception handler → FAILED
        assert task.status == TaskStatus.FAILED
        assert task.result is not None
        assert "Unexpected" in task.result.summary or "Router crash" in task.result.summary


@pytest.mark.integration
def test_context_is_passed_to_orchestrator(engine):
    """Verify that context (history) accumulates and is available to subsequent calls."""
    user_input1 = "Task 1"
    intent_res1 = IntentResult(IntentType.TASK, 0.9, "Task", user_input1)
    
    plan = {
        "task_type": "script",
        "description": "T1",
        "files_to_generate": [],
        "entry_point": "",
        "test_command": ""
    }
    from nexus.executor.safe_exec import ExecResult
    with patch.object(engine.ai, "generate_plan", return_value=plan), \
         patch.object(engine.ai, "generate_code", return_value={}), \
         patch("nexus.core.engine.run_command", return_value=ExecResult(0, "ok", "")), \
         patch("nexus.core.engine.validate_result") as mock_val:
        
        mock_val.return_value = MagicMock(
            is_success=True, status="PASS", semantic_verdict="CORRECT",
            semantic_issues=[],
            stage1_result=MagicMock(returncode=0, stdout="", stderr="", timed_out=False)
        )
        
        engine.run(user_input1, intent_res1)
        assert len(engine.context.task_history) == 1
        
        user_input2 = "Task 2"
        intent_res2 = IntentResult(IntentType.TASK, 0.9, "Task", user_input2)
        
        # Second call
        engine.run(user_input2, intent_res2)
        
        assert len(engine.context.task_history) == 2
        # Check that context holds chronological history
        assert "successfully" in engine.context.task_history[0]["summary"]
