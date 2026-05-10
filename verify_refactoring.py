#!/usr/bin/env python3
"""
Verification script for structural refactoring.
Tests all requirements without full environment.
"""
import sys

def test_1_contracts_imports():
    """Test: Contracts module imports work."""
    try:
        from nexus.core.contracts import (
            TaskPlan, LLMRequest, LLMResponse,
            GeneratedFiles, ExecutionOutput, ValidationOutcome,
            FixResult, TaskContext, ProviderHealth, plan_from_dict
        )
        print("✓ Test 1: Contracts imports - PASS")
        return True
    except Exception as e:
        print(f"✗ Test 1: Contracts imports - FAIL: {e}")
        return False

def test_2_exceptions_with_dict():
    """Test: New exceptions work with to_dict()."""
    try:
        from nexus.core.exceptions import (
            ProviderError, PipelineStageError, 
            WorkspaceError, ConfigurationError
        )
        
        # Test ProviderError
        e1 = ProviderError('test', provider_name='ollama', is_retryable=True)
        d1 = e1.to_dict()
        assert d1['type'] == 'ProviderError'
        assert d1['context']['provider_name'] == 'ollama'
        assert d1['context']['is_retryable'] == True
        
        # Test PipelineStageError
        e2 = PipelineStageError('stage failed', stage_name='EXECUTING', is_fatal=False)
        d2 = e2.to_dict()
        assert d2['type'] == 'PipelineStageError'
        assert d2['context']['stage_name'] == 'EXECUTING'
        
        # Test WorkspaceError
        e3 = WorkspaceError('op failed', workspace_id='ws123', operation='cleanup')
        d3 = e3.to_dict()
        assert d3['context']['workspace_id'] == 'ws123'
        
        # Test ConfigurationError
        e4 = ConfigurationError('bad field', field_name='timeout', expected='int')
        d4 = e4.to_dict()
        assert d4['context']['field_name'] == 'timeout'
        
        print("✓ Test 2: New exceptions with to_dict() - PASS")
        return True
    except Exception as e:
        print(f"✗ Test 2: New exceptions with to_dict() - FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_3_task_backward_compat():
    """Test: Task class backward-compatible properties."""
    try:
        from nexus.core.task import Task
        from nexus.core.contracts import GeneratedFiles, ExecutionOutput
        
        t = Task()
        
        # Test generated_files property (empty dict)
        assert t.generated_files == {}, f"Expected {{}}, got {t.generated_files}"
        
        # Test setting generated_files
        t.generated_files = {"test.py": "print('hello')"}
        assert t.generated_files == {"test.py": "print('hello')"}
        
        # Test last_stdout/stderr properties
        assert t.last_stdout == ""
        assert t.last_stderr == ""
        
        # Test setting last_execution
        output = ExecutionOutput(returncode=0, stdout="output", stderr="")
        t.record_execution(output)
        assert t.last_stdout == "output"
        assert t.last_stderr == ""
        
        # Test deprecated method (should emit warning)
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            t.record_execution_output("stdout", "stderr")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
        
        print("✓ Test 3: Task backward-compatible properties - PASS")
        return True
    except Exception as e:
        print(f"✗ Test 3: Task backward-compatible properties - FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_4_plan_from_dict():
    """Test: plan_from_dict() helper function."""
    try:
        from nexus.core.contracts import plan_from_dict, TaskPlan
        from nexus.core.exceptions import ConfigurationError
        
        # Valid plan dict
        plan_dict = {
            "task_type": "script",
            "description": "test",
            "files_to_generate": ["main.py"],
            "entry_point": "main.py",
            "test_command": "python3 main.py",
            "install_command": "pip install",
        }
        
        plan = plan_from_dict(plan_dict)
        assert isinstance(plan, TaskPlan)
        assert plan.task_type == "script"
        
        # Missing required field
        bad_dict = {"task_type": "script"}
        try:
            plan_from_dict(bad_dict)
            print("✗ Test 4: plan_from_dict() - FAIL: Should raise ConfigurationError")
            return False
        except ConfigurationError as e:
            assert "Missing required field" in str(e)
        
        print("✓ Test 4: plan_from_dict() helper - PASS")
        return True
    except Exception as e:
        print(f"✗ Test 4: plan_from_dict() helper - FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_5_task_plan_mutation():
    """Test: TaskPlan dict-like interface for backward compat."""
    try:
        from nexus.core.contracts import TaskPlan
        
        plan = TaskPlan(
            task_type="script",
            description="test",
            files_to_generate=["main.py"],
            entry_point="main.py",
            test_command="python3 main.py",
            install_command="pip install",
        )
        
        # Test __getitem__
        assert plan["task_type"] == "script"
        
        # Test get()
        assert plan.get("task_type") == "script"
        assert plan.get("nonexistent", "default") == "default"
        
        # Test __setitem__ (mutate - even though frozen for direct attribute)
        plan["test_command"] = "pytest"
        assert plan["test_command"] == "pytest"
        
        print("✓ Test 5: TaskPlan dict-like interface - PASS")
        return True
    except Exception as e:
        print(f"✗ Test 5: TaskPlan dict-like interface - FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_6_execution_output_properties():
    """Test: ExecutionOutput success property."""
    try:
        from nexus.core.contracts import ExecutionOutput
        
        # Success case
        output = ExecutionOutput(returncode=0, stdout="", stderr="")
        assert output.success == True
        
        # Failure case - non-zero returncode
        output = ExecutionOutput(returncode=1, stdout="", stderr="error")
        assert output.success == False
        
        # Failure case - timeout
        output = ExecutionOutput(returncode=0, stdout="", stderr="", timed_out=True)
        assert output.success == False
        
        # Failure case - security blocked
        output = ExecutionOutput(returncode=0, stdout="", stderr="", security_blocked=True)
        assert output.success == False
        
        print("✓ Test 6: ExecutionOutput properties - PASS")
        return True
    except Exception as e:
        print(f"✗ Test 6: ExecutionOutput properties - FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_7_validation_outcome_properties():
    """Test: ValidationOutcome properties."""
    try:
        from nexus.core.contracts import ValidationOutcome
        
        # Success case
        vo = ValidationOutcome(
            stage1_passed=True,
            stage1_exit_code=0,
            semantic_verdict="CORRECT"
        )
        assert vo.is_success == True
        assert vo.is_terminal_failure == False
        
        # Failure case
        vo = ValidationOutcome(
            stage1_passed=False,
            stage1_exit_code=1,
            semantic_verdict="INCORRECT"
        )
        assert vo.is_success == False
        
        # Terminal failure case
        vo = ValidationOutcome(
            stage1_passed=True,
            stage1_exit_code=4,
            semantic_verdict="UNCERTAIN",
            is_config_error=True
        )
        assert vo.is_terminal_failure == True
        
        print("✓ Test 7: ValidationOutcome properties - PASS")
        return True
    except Exception as e:
        print(f"✗ Test 7: ValidationOutcome properties - FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_8_all_exports():
    """Test: __all__ exports are complete."""
    try:
        import nexus.core.exceptions as exc_module
        import nexus.core.contracts as contract_module
        
        # Check exceptions __all__
        exc_all = exc_module.__all__
        required_exc = [
            "NexusBaseException",
            "TaskPlanningError",
            "TaskGenerationError",
            "ExecutionError",
            "ValidationError",
            "SafetyViolation",
            "MaxRetriesExceeded",
            "WorkspaceSecurityError",
            "OllamaConnectionError",
            "CloudProviderError",
            "ProviderError",
            "PipelineStageError",
            "WorkspaceError",
            "ConfigurationError",
        ]
        for exc in required_exc:
            assert exc in exc_all, f"Missing {exc} in exceptions.__all__"
        
        # Check contracts __all__
        contract_all = contract_module.__all__
        required_contracts = [
            "LLMRequest", "LLMResponse",
            "TaskPlan", "GeneratedFiles",
            "ExecutionOutput", "ValidationOutcome",
            "FixResult", "TaskContext", "ProviderHealth",
            "plan_from_dict",
        ]
        for contract in required_contracts:
            assert contract in contract_all, f"Missing {contract} in contracts.__all__"
        
        print("✓ Test 8: __all__ exports complete - PASS")
        return True
    except Exception as e:
        print(f"✗ Test 8: __all__ exports complete - FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("NEXUS Structural Refactoring Verification")
    print("=" * 60)
    
    tests = [
        test_1_contracts_imports,
        test_2_exceptions_with_dict,
        test_3_task_backward_compat,
        test_4_plan_from_dict,
        test_5_task_plan_mutation,
        test_6_execution_output_properties,
        test_7_validation_outcome_properties,
        test_8_all_exports,
    ]
    
    results = [test() for test in tests]
    
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    sys.exit(0 if all(results) else 1)
