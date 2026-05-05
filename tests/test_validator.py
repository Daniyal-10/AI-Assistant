"""
tests/test_validator.py
───────────────────────
Unit tests for Stage 1 (Structural) and Stage 2 (Semantic) validation.
"""
import pytest
from unittest.mock import MagicMock
from nexus.executor.validator import validate_result, ValidationResult
from nexus.executor.safe_exec import ExecResult

def test_stage1_structural_failure():
    """Verify that Stage 1 catch exit code errors independently."""
    # Exit code 1 = failure
    res = ExecResult(returncode=1, stdout="", stderr="ImportError: module not found")
    vr = validate_result(res)
    
    assert vr.status == "FAIL"
    assert vr.semantic_verdict is None
    assert not vr.is_success

def test_stage1_timeout_failure():
    """Verify that Stage 1 catch timeouts."""
    res = ExecResult(returncode=-1, stdout="", stderr="Killed", timed_out=True)
    vr = validate_result(res)
    
    assert vr.status == "TIMEOUT"
    assert not vr.is_success

def test_stage2_semantic_success():
    """Verify that Stage 2 confirms a correct solution when Stage 1 passes."""
    res = ExecResult(returncode=0, stdout="Output: 4", stderr="")
    ai = MagicMock()
    ai.validate_correctness.return_value = {
        "verdict": "CORRECT",
        "reason": "The code correctly calculates the sum and prints it."
    }
    
    vr = validate_result(
        res, 
        ai=ai, 
        task_description="Add 2+2", 
        generated_code="print(f'Output: {2+2}')"
    )
    
    assert vr.status == "PASS"
    assert vr.semantic_verdict == "CORRECT"
    assert vr.is_success

def test_stage2_semantic_failure():
    """Verify that Stage 2 catches a semantically wrong solution despite exit code 0."""
    # Code runs (exit 0) but produces wrong output
    res = ExecResult(returncode=0, stdout="Output: 5", stderr="")
    ai = MagicMock()
    ai.validate_correctness.return_value = {
        "verdict": "INCORRECT",
        "reason": "The code prints 5 instead of 4.",
        "issues": ["Incorrect math logic"]
    }
    
    vr = validate_result(
        res, 
        ai=ai, 
        task_description="Add 2+2", 
        generated_code="print('Output: 5')"
    )
    
    assert vr.status == "PASS"
    assert vr.semantic_verdict == "INCORRECT"
    assert not vr.is_success # Semantic failure blocks success

def test_stage2_uncertain_passes():
    """Verify that Stage 2 UNCERTAIN does not block a Stage 1 PASS."""
    res = ExecResult(returncode=0, stdout="Complex data dump", stderr="")
    ai = MagicMock()
    ai.validate_correctness.return_value = {
        "verdict": "UNCERTAIN",
        "reason": "The output is too complex to verify without deep analysis."
    }
    
    vr = validate_result(res, ai=ai, task_description="Run complex sim", generated_code="sim()")
    
    assert vr.status == "PASS"
    assert vr.semantic_verdict == "UNCERTAIN"
    assert vr.is_success

def test_stage2_skip_on_ai_error():
    """Verify that if Stage 2 fails (AI error), we fall back to Stage 1 result."""
    res = ExecResult(returncode=0, stdout="Output: 4", stderr="")
    ai = MagicMock()
    ai.validate_correctness.side_effect = Exception("Ollama connection lost")
    
    # Should not raise, should log warning and return Stage 1 result
    vr = validate_result(res, ai=ai, task_description="Add 2+2", generated_code="print(4)")
    
    assert vr.status == "PASS"
    assert vr.semantic_verdict is None
    assert vr.is_success
