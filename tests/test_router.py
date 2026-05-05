"""
tests/test_router.py
────────────────────
Unit tests for the Intent Router.
"""
import pytest
from unittest.mock import MagicMock
from nexus.ai.router import IntentRouter, IntentType

def test_router_rules_chat():
    """Verify that basic greetings match CHAT via rules."""
    router = IntentRouter()
    # Mock LLM to return None to force rule-based fallback
    router.ai._call_ollama = MagicMock(return_value=None)
    
    result = router.classify("Hello NEXUS!")
    assert result.intent == IntentType.CHAT
    assert result.confidence == 0.6
    assert "Rule-based" in result.reasoning

def test_router_rules_system():
    """Verify that system-like commands match SYSTEM via rules."""
    router = IntentRouter()
    router.ai._call_ollama = MagicMock(return_value=None)
    
    result = router.classify("ls -la in the current directory")
    assert result.intent == IntentType.SYSTEM
    assert result.confidence == 0.6

def test_router_rules_code():
    """Verify that code analysis requests match CODE via rules."""
    router = IntentRouter()
    router.ai._call_ollama = MagicMock(return_value=None)
    
    result = router.classify("Can you explain what this function does?")
    assert result.intent == IntentType.CODE
    assert result.confidence == 0.6

def test_router_rules_task():
    """Verify that automation requests match TASK via rules."""
    router = IntentRouter()
    router.ai._call_ollama = MagicMock(return_value=None)
    
    result = router.classify("Write a script to scrape weather data")
    assert result.intent == IntentType.TASK
    assert result.confidence == 0.6

def test_router_llm_success():
    """Verify that valid LLM JSON response is parsed correctly."""
    router = IntentRouter()
    mock_json = '{"intent": "TASK", "confidence": 0.98, "reasoning": "Clear request for a new project"}'
    router.ai._call_ollama = MagicMock(return_value=mock_json)
    
    result = router.classify("Create a FastAPI application with PostgreSQL")
    assert result.intent == IntentType.TASK
    assert result.confidence == 0.98
    assert result.reasoning == "Clear request for a new project"

def test_router_llm_malformed_json():
    """Verify that malformed LLM JSON falls back to rules."""
    router = IntentRouter()
    router.ai._call_ollama = MagicMock(return_value="NOT JSON AT ALL")
    
    # "build" keyword should trigger TASK rule
    result = router.classify("build me a tool")
    assert result.intent == IntentType.TASK
    assert "Rule-based" in result.reasoning

def test_router_default_fallback():
    """Verify that completely ambiguous input defaults to CHAT."""
    router = IntentRouter()
    router.ai._call_ollama = MagicMock(return_value=None)
    
    result = router.classify("something completely unrelated")
    assert result.intent == IntentType.CHAT
    assert result.confidence == 0.1
    assert "Default fallback" in result.reasoning

def test_router_empty_input():
    """Verify that empty or whitespace input defaults to CHAT."""
    router = IntentRouter()
    
    result = router.classify("   ")
    assert result.intent == IntentType.CHAT
    assert "Empty input" in result.reasoning

def test_router_sanitization():
    """Verify that control characters are stripped from LLM prompt."""
    router = IntentRouter()
    # We want to check what is passed to _call_ollama
    router.ai._call_ollama = MagicMock(return_value=None)
    
    input_text = "hello\x00world\n"
    router.classify(input_text)
    
    # Check the user_prompt argument of the last call
    last_call = router.ai._call_ollama.call_args
    user_prompt = last_call.kwargs['user_prompt']
    assert "\x00" not in user_prompt
    assert "helloworld" in user_prompt.lower()
