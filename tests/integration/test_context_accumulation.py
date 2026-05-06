"""
tests/integration/test_context_accumulation.py
─────────────────────────────────────────────
Integration tests for SessionContext state accumulation and isolation.
"""
import pytest
from unittest.mock import MagicMock, patch
from nexus.core.engine import TaskEngine
from nexus.core.context import SessionContext
from nexus.ai.router import IntentType, IntentResult


@pytest.fixture
def engine():
    """Fresh engine with unique context."""
    ctx = SessionContext()
    return TaskEngine(ctx)


@pytest.mark.integration
def test_conversation_history_grows_across_calls(engine):
    """Verify that every interaction appends to the conversation history."""
    inputs = ["Greeting", "Question", "Request"]
    
    with patch.object(engine.ai, "generate_chat_response", return_value="Acknowledged."):
        for i, user_input in enumerate(inputs):
            intent_res = IntentResult(IntentType.CHAT, 1.0, "Chat", user_input)
            engine.run(user_input, intent_res)
            
            # Each interaction adds 2 messages (user + assistant)
            expected_len = (i + 1) * 2
            assert len(engine.context.conversation_history) == expected_len
            assert engine.context.conversation_history[-2]["role"] == "user"
            assert engine.context.conversation_history[-1]["role"] == "assistant"


@pytest.mark.integration
def test_task_history_is_recorded_on_success(engine):
    """Verify that successful tasks are recorded in context task_history."""
    user_input = "Successful Task"
    intent_res = IntentResult(IntentType.TASK, 1.0, "Task", user_input)
    
    with patch.object(engine.ai, "generate_plan", return_value={"description": "Desc", "steps": [], "test_command": "pytest"}), \
         patch.object(engine.ai, "generate_code", return_value={}), \
         patch("nexus.executor.safe_exec.run_command", return_value=MagicMock(returncode=0, stdout="")), \
         patch("nexus.core.engine.validate_result", return_value=MagicMock(
             is_success=True, status="PASS", semantic_verdict="CORRECT",
             semantic_issues=[],
             stage1_result=MagicMock(returncode=0, stdout="", stderr="", timed_out=False)
         )):
        
        task = engine.run(user_input, intent_res)
        
        assert len(engine.context.task_history) == 1
        entry = engine.context.task_history[0]
        assert entry["task_id"] == task.id
        assert entry["status"] == "DONE"


@pytest.mark.integration
def test_task_history_is_recorded_on_failure(engine):
    """Verify that failed tasks are recorded in context task_history with FAILED status."""
    user_input = "Failing Task"
    intent_res = IntentResult(IntentType.TASK, 1.0, "Task", user_input)
    
    # Simulate a definitive failure (max retries hit or validator says NO)
    with patch.object(engine.ai, "generate_plan", return_value={"description": "Desc", "steps": [], "test_command": "pytest"}), \
         patch.object(engine.ai, "generate_code", return_value={}), \
         patch("nexus.executor.safe_exec.run_command", return_value=MagicMock(returncode=1, stderr="Error")), \
         patch("nexus.core.engine.validate_result", return_value=MagicMock(
             is_success=False, status="FAIL", semantic_verdict=None,
             semantic_issues=[],
             stage1_result=MagicMock(returncode=1, stdout="", stderr="Error", timed_out=False)
         )):
        
        task = engine.run(user_input, intent_res)
        
        assert len(engine.context.task_history) == 1
        entry = engine.context.task_history[0]
        assert entry["status"] == "FAILED"


@pytest.mark.integration
def test_context_does_not_leak_between_sessions():
    """Verify that multiple engine instances maintain isolated session states."""
    engine1 = TaskEngine(SessionContext())
    engine2 = TaskEngine(SessionContext())
    
    with patch.object(engine1.ai, "generate_chat_response", return_value="R1"):
        engine1.run("Session 1 Message", IntentResult(IntentType.CHAT, 1.0, "C", "M1"))
        
    assert len(engine1.context.conversation_history) == 2
    assert len(engine2.context.conversation_history) == 0
