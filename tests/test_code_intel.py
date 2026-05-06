"""
tests/test_code_intel.py
────────────────────────
Unit tests for the CodeIntelligence module.
"""
import pytest
from unittest.mock import MagicMock, patch
from nexus.ai.code_intel import CodeIntelligence


def test_code_intel_explain():
    """Verify that explain returns AI text."""
    ai = MagicMock()
    ai._call_ollama.return_value = "This code initializes a database connection."
    intel = CodeIntelligence(ai)
    
    res = intel.explain("db = connect()", "Explain this")
    assert "database connection" in res


def test_code_intel_refactor():
    """Verify that refactor parses JSON correctly."""
    ai = MagicMock()
    # Simulate a JSON response from Ollama
    ai._call_ollama.return_value = '{"reasoning": "Modernize syntax", "refactored_code": "print(f\'{x}\')"}'
    intel = CodeIntelligence(ai)
    
    res = intel.refactor("print x", "Update to Python 3")
    assert res["reasoning"] == "Modernize syntax"
    assert "print(f" in res["refactored_code"]


def test_code_intel_debug():
    """Verify that debug diagnoses bugs and suggests fixes."""
    ai = MagicMock()
    ai._call_ollama.return_value = '{"diagnosis": "Off-by-one error", "fix": "Change range to n+1", "confidence": 0.95}'
    intel = CodeIntelligence(ai)
    
    res = intel.debug("for i in range(n):", "Last element missed")
    assert res["diagnosis"] == "Off-by-one error"


def test_code_intel_review():
    """Verify that review provides qualitative feedback."""
    ai = MagicMock()
    ai._call_ollama.return_value = '{"issues": ["Hardcoded secret"], "suggestions": ["Use env var"], "quality_score": 3}'
    intel = CodeIntelligence(ai)
    
    res = intel.review("API_KEY = '12345'")
    assert res["quality_score"] == 3


def test_code_intel_fallbacks():
    """Verify that operations return valid fallbacks if LLM returns garbage."""
    ai = MagicMock()
    ai._call_ollama.return_value = "NOT JSON AT ALL"
    intel = CodeIntelligence(ai)
    
    # Refactor should return original code on failure
    res = intel.refactor("original code", "refactor it")
    assert res["refactored_code"] == "original code"
    assert "Could not parse" in res["reasoning"]


def test_code_intel_truncation():
    """Verify that extremely long files are truncated before sending to AI."""
    ai = MagicMock()
    ai._call_ollama.return_value = "Explanation"
    intel = CodeIntelligence(ai)
    intel.max_chars = 10 # Artificially low limit
    
    content = "This is a very long string"
    intel.explain(content)
    
    # Check what was passed to LLM
    args, kwargs = ai._call_ollama.call_args
    user_prompt = kwargs['user_prompt']
    assert "truncated" in user_prompt
