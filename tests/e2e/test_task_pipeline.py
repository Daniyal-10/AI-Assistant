"""
tests/e2e/test_task_pipeline.py
───────────────────────────────
End-to-End tests for the full NEXUS task execution pipeline.
"""
import os
import pytest
import json
from unittest.mock import patch, MagicMock
from nexus.core.task import TaskStatus
from nexus.ai.router import IntentType, IntentResult


@pytest.mark.e2e
def test_hello_world_task_completes_successfully(nexus_engine, mock_llm_responses):
    """Verify that a simple TASK intent goes through planning, execution, and validation."""
    plan = {
        "task_type": "script",
        "description": "Print hello world",
        "files_to_generate": ["main.py", "test_main.py"],
        "entry_point": "main.py",
        "test_command": "pytest test_main.py",
        "install_command": "echo none"
    }
    files = {"files": {
        "main.py": "print('Hello, World!')", 
        "test_main.py": "def test_hello(): assert True"
    }}
    validation = {"verdict": "CORRECT", "reason": "PASS: output is correct", "issues": []}
    
    # Mock sequential LLM calls: generate_plan -> generate_code -> validate_semantic
    mock_llm_responses(nexus_engine, [
        json.dumps(plan),       
        json.dumps(files),      
        json.dumps(validation)  
    ])
    
    user_input = "write a python script that prints Hello, World!"
    intent_res = IntentResult(IntentType.TASK, 1.0, "Task", user_input)
    
    task = nexus_engine.run(user_input, intent_res)
    
    assert task.status == TaskStatus.DONE
    assert task.result.output_path is not None
    assert os.path.exists(task.result.output_path)
    assert task.result.output_path.endswith(".zip")
    assert len(nexus_engine.context.task_history) == 1
    assert nexus_engine.context.task_history[0]["status"] == "DONE"


@pytest.mark.e2e
def test_failing_task_activates_fix_loop(nexus_engine, mock_llm_responses):
    """Verify that execution failures (SyntaxError) trigger the orchestrator fix loop."""
    plan = {
        "task_type": "script",
        "description": "Fail then fix",
        "files_to_generate": ["main.py"],
        "entry_point": "main.py",
        "test_command": "python3 main.py",
        "install_command": "echo none"
    }
    files_broken = {"files": {"main.py": "intentional_syntax_error((("}}
    files_fixed = {"fixed_files": {"main.py": "print('fixed')"}}
    validation = {"verdict": "CORRECT", "reason": "PASS", "issues": []}
    
    # Mock calls: generate_plan -> generate_code (fails gate) -> generate_fix -> validate_semantic
    mock_llm_responses(nexus_engine, [
        json.dumps(plan),         
        json.dumps(files_broken), 
        json.dumps(files_fixed),  
        json.dumps(validation)     
    ])
    
    user_input = "write a script that prints fixed"
    intent_res = IntentResult(IntentType.TASK, 1.0, "Task", user_input)
    
    task = nexus_engine.run(user_input, intent_res)
    
    assert task.status == TaskStatus.DONE
    # iterations_used increments for every fix attempt
    assert task.result.iterations_used >= 1
    assert "fixed" in task.result.summary.lower()


@pytest.mark.e2e
def test_chat_input_returns_jarvis_response(nexus_engine, mock_llm_responses):
    """Verify that CHAT intent returns conversational text with no side effects."""
    mock_llm_responses(nexus_engine, ["All systems nominal. What can I assist with?"])
    
    user_input = "hello nexus"
    intent_res = IntentResult(IntentType.CHAT, 1.0, "Chat", user_input)
    
    task = nexus_engine.run(user_input, intent_res)
    
    assert isinstance(task.result.summary, str)
    assert len(task.result.summary.split()) >= 5
    assert task.workspace_path is None or not os.path.exists(task.workspace_path)


@pytest.mark.e2e
def test_code_explanation_returns_explanation(nexus_engine, mock_llm_responses):
    """Verify that CODE intent returns an analysis of the provided snippet."""
    mock_llm_responses(nexus_engine, ["This function adds two numbers together."])
    
    user_input = "explain this: def add(a,b): return a+b"
    intent_res = IntentResult(IntentType.CODE, 1.0, "Code", user_input)
    
    task = nexus_engine.run(user_input, intent_res)
    
    assert isinstance(task.result.summary, str)
    assert "adds" in task.result.summary
    assert task.workspace_path is None


@pytest.mark.e2e
def test_persistent_failure_returns_failed_result(nexus_engine, mock_llm_responses):
    """Verify that the pipeline stops and reports failure if the fix loop exhausts retries."""
    plan = {
        "task_type": "script",
        "description": "Persistent fail",
        "files_to_generate": ["main.py"],
        "entry_point": "main.py",
        "test_command": "python3 main.py",
        "install_command": "echo none"
    }
    files_broken = {"files": {"main.py": "def broken("}}
    
    # Fill enough responses to cover max retries (configured in engine/orchestrator)
    responses = [json.dumps(plan)] + [json.dumps(files_broken)] * 10
    mock_llm_responses(nexus_engine, responses)
    
    user_input = "write a script"
    intent_res = IntentResult(IntentType.TASK, 1.0, "Task", user_input)
    
    task = nexus_engine.run(user_input, intent_res)
    
    assert task.status == TaskStatus.FAILED
    assert len(nexus_engine.context.task_history) == 1
    assert nexus_engine.context.task_history[0]["status"] == "FAILED"
    assert "traceback" not in task.result.summary.lower() # Should be human-readable
