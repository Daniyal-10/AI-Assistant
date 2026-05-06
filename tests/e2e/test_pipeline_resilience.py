"""
tests/e2e/test_pipeline_resilience.py
─────────────────────────────────────
End-to-End tests for pipeline stability, resilience, and state management.
"""
import pytest
from unittest.mock import patch, MagicMock
from nexus.ai.router import IntentType, IntentResult


@pytest.mark.e2e
def test_empty_input_is_handled_gracefully(nexus_engine, mock_llm_responses):
    """Verify that an empty string does not cause a crash and returns a fallback."""
    mock_llm_responses(nexus_engine, ["Your input was empty. How can I help?"] * 2)
    
    # engine.run handles empty input by calling the router (which likely returns CHAT or error)
    task = nexus_engine.run("", None)
    
    assert task.result.summary
    assert len(task.result.summary) > 0


@pytest.mark.e2e
def test_very_long_input_is_handled(nexus_engine, mock_llm_responses):
    """Verify that a very large input string is processed without buffer overflows or hangs."""
    mock_llm_responses(nexus_engine, ["Input received and acknowledged."] * 2)
    long_input = "a" * 10000
    
    # Should finish near-instantaneously with mocked LLM
    task = nexus_engine.run(long_input, None)
    
    assert task.result.summary
    assert len(nexus_engine.context.conversation_history) >= 2


@pytest.mark.e2e
def test_special_characters_in_input(nexus_engine, mock_llm_responses):
    """Verify that injection-style strings are treated as literal text."""
    mock_llm_responses(nexus_engine, ["Detected potential injection or special characters."] * 2)
    tricky_input = "'; DROP TABLE tasks; -- <script>alert(1)</script>"
    
    task = nexus_engine.run(tricky_input, None)
    
    assert task.result.summary
    # Ensure no database crash or script execution occurred in the backend
    assert len(nexus_engine.context.conversation_history) >= 2


@pytest.mark.e2e
def test_unicode_input_is_handled(nexus_engine, mock_llm_responses):
    """Verify that non-ASCII characters (UTF-8) are handled correctly throughout the pipeline."""
    mock_llm_responses(nexus_engine, ["Processed Hindi request."] * 2)
    unicode_input = "हैलो नेक्सस, मुझे एक Python स्क्रिप्ट बनाओ"
    
    task = nexus_engine.run(unicode_input, None)
    
    assert task.result.summary
    assert nexus_engine.context.conversation_history[-2]["content"] == unicode_input


@pytest.mark.e2e
def test_repeated_calls_do_not_leak_state(nexus_engine, mock_llm_responses):
    """Verify that session context history remains bounded to prevent memory/token bloat."""
    # Setup many responses
    mock_llm_responses(nexus_engine, ["Response"] * 50)
    
    # 1. Run 15 sequential chat calls
    for i in range(15):
        nexus_engine.run(f"chat {i}", IntentResult(IntentType.CHAT, 1.0, "C", f"chat {i}"))
    
    # 2. Run 12 sequential task calls
    # Mock orchestrator internal calls to avoid real filesystem activity in this resilience test
    with patch.object(nexus_engine.ai, "generate_plan", return_value={"steps": [], "description": ""}), \
         patch.object(nexus_engine.ai, "generate_code", return_value={}), \
         patch("nexus.executor.safe_exec.run_command", return_value=MagicMock(
             returncode=0, stdout="", stderr="", timed_out=False
         )), \
         patch("nexus.executor.validator.validate_result", return_value=MagicMock(
             is_success=True,
             status="PASS",
             semantic_verdict="CORRECT",
             semantic_issues=[],
             stage1_result=MagicMock(returncode=0, stdout="", stderr="", timed_out=False)
         )):
        
        for i in range(12):
            nexus_engine.run(f"task {i}", IntentResult(IntentType.TASK, 1.0, "T", f"task {i}"))
            
    # Verify Bounding Logic (from context.py: max 20 msgs, 10 tasks)
    # Total calls: 15 chat + 12 task = 27 interactions
    # Each interaction adds 2 messages = 54 messages total
    assert len(nexus_engine.context.conversation_history) == 20
    assert len(nexus_engine.context.task_history) == 10
    
    # Verify task history matches the last 10
    assert isinstance(nexus_engine.context.task_history[-1]["summary"], str)
    assert len(nexus_engine.context.task_history) == 10
