"""
tests/test_code_intel.py
────────────────────────
Unit tests for Code Intelligence operations.
"""
import pytest
from unittest.mock import MagicMock, patch
from nexus.ai.code_intel import CodeIntel


def test_code_intel_explain():
    """Verify that explain returns a raw string response."""
    ai = MagicMock()
    ai._call_ollama.return_value = "This code initializes a database connection."
    intel = CodeIntel(ai)
    
    res = intel.explain("db = connect()", "Explain this")
    assert "database connection" in res
    assert ai._call_ollama.called


def test_code_intel_refactor_success():
    """Verify that refactor returns structured JSON."""
    ai = MagicMock()
    # Simulate a JSON response from Ollama
    ai._call_ollama.return_value = '{"reasoning": "Modernize syntax", "refactored_code": "print(f\'{x}\')"}'
    intel = CodeIntel(ai)
    
    res = intel.refactor("print x", "Update to Python 3")
    assert res["reasoning"] == "Modernize syntax"
    assert "refactored_code" in res
    assert "print(f'{x}')" in res["refactored_code"]


def test_code_intel_debug_success():
    """Verify that debug diagnoses bugs and suggests fixes."""
    ai = MagicMock()
    ai._call_ollama.return_value = '{"diagnosis": "Off-by-one error", "fix": "Change range to n+1", "confidence": 0.95}'
    intel = CodeIntel(ai)
    
    res = intel.debug("for i in range(n):", "Last element missed")
    assert res["diagnosis"] == "Off-by-one error"
    assert res["confidence"] == 0.95


def test_code_intel_review_success():
    """Verify that review provides qualitative feedback."""
    ai = MagicMock()
    ai._call_ollama.return_value = '{"issues": ["Hardcoded secret"], "suggestions": ["Use env var"], "quality_score": 3}'
    intel = CodeIntel(ai)
    
    res = intel.review("API_KEY = '12345'")
    assert res["quality_score"] == 3
    assert "Hardcoded secret" in res["issues"]


def test_code_intel_fallback_on_malformed_json():
    """Verify that operations return valid fallbacks if LLM returns garbage."""
    ai = MagicMock()
    ai._call_ollama.return_value = "NOT JSON AT ALL"
    intel = CodeIntel(ai)
    
    # Refactor should return original code on failure
    res = intel.refactor("original code", "refactor it")
    assert res["refactored_code"] == "original code"
    assert "Could not parse" in res["reasoning"]


def test_code_intel_truncation():
    """Verify that extremely long files are truncated before sending to AI."""
    ai = MagicMock()
    ai._call_ollama.return_value = "Explanation"
    intel = CodeIntel(ai)
    intel.max_chars = 10 # Artificially low limit
    
    content = "This is a very long string"
    intel.explain(content)
    
    # Check the user_prompt passed to _call_ollama
    call_args = ai._call_ollama.call_args
    user_prompt = call_args.kwargs['user_prompt']
    assert "[... content truncated for safety ...]" in user_prompt
    assert len(user_prompt.split("\n\n")[0]) == 10
