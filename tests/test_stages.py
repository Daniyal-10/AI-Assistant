import pytest
from unittest.mock import MagicMock, patch, mock_open
from nexus.core.task import Task, TaskStatus, TaskResult
from nexus.core.pipeline import TaskExecutionContext, StageResult
from nexus.core.stages.planning import PlanningStage
from nexus.core.stages.generation import GenerationStage
from nexus.core.stages.installation import InstallStage
from nexus.core.stages.validation import ValidationStage
from nexus.core.stages.repair import RepairStage
from nexus.core.stages.completion import CompletionStage
from nexus.core.events import EventBus
from nexus.executor.safe_exec import ExecResult
from nexus.core.exceptions import MaxRetriesExceeded

@pytest.fixture
def base_context():
    task = Task(raw_input="Test prompt")
    task.id = "test-task-id"
    event_bus = EventBus()
    workspace = MagicMock()
    workspace.get_path.return_value = "/tmp/mock-workspace"
    
    config_snapshot = {
        "max_fix_iterations": 3,
        "exec_timeout": 30,
        "ollama_base_url": "http://localhost:11434"
    }
    
    ctx = TaskExecutionContext(
        task=task,
        session=MagicMock(),
        ai=MagicMock(),
        workspace=workspace,
        venv_path="/tmp/mock-workspace/.venv",
        event_bus=event_bus,
        config_snapshot=config_snapshot,
        engine=None  # Set to None to prevent MagicMock auto-attributes from triggering legacy compatibility mocks
    )
    return ctx

def test_planning_stage(base_context):
    stage = PlanningStage()
    assert stage.name == "planning"
    
    mock_plan = {"description": "Test Plan", "test_command": "pytest"}
    base_context.ai.generate_plan.return_value = mock_plan
    
    res = stage.execute(base_context)
    assert res.success is True
    assert base_context.task.plan == mock_plan
    assert base_context.task.status == TaskStatus.PLANNING

def test_generation_stage(base_context):
    stage = GenerationStage()
    assert stage.name == "generation"
    
    base_context.task.status = TaskStatus.PLANNING
    base_context.task.plan = {"description": "Test Plan"}
    mock_files = {"main.py": "print('hello')"}
    base_context.ai.generate_code.return_value = mock_files
    
    with patch("nexus.core.stages.generation.inject_conftest_if_needed"), \
         patch("nexus.core.stages.generation.normalize_test_command"):
        res = stage.execute(base_context)
        
    assert res.success is True
    assert base_context.task.generated_files == mock_files
    base_context.workspace.write_files.assert_called_once_with(mock_files)
    assert base_context.task.status == TaskStatus.GENERATING

def test_install_stage_skipped_no_requirements(base_context):
    stage = InstallStage()
    assert stage.name == "installation"
    
    base_context.task.status = TaskStatus.GENERATING
    base_context.task.plan = {"install_command": "pip install -r requirements.txt"}
    
    with patch("os.path.exists", return_value=False):
        res = stage.execute(base_context)
        
    assert res.success is True
    assert base_context.task.status == TaskStatus.EXECUTING

def test_install_stage_success(base_context):
    stage = InstallStage()
    base_context.task.status = TaskStatus.GENERATING
    base_context.task.plan = {"install_command": "pip install -r requirements.txt"}
    base_context.task.venv_executables = {"pip": "/bin/pip"}
    
    exec_result = ExecResult(returncode=0, stdout="Successfully installed", stderr="")
    
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data="dependency")), \
         patch("nexus.core.stages.installation.run_command", return_value=exec_result), \
         patch("nexus.core.stages.installation.validate_result", return_value=True):
        res = stage.execute(base_context)
        
    assert res.success is True
    assert base_context.task.status == TaskStatus.EXECUTING

def test_install_stage_failed(base_context):
    stage = InstallStage()
    base_context.task.status = TaskStatus.GENERATING
    base_context.task.plan = {"install_command": "pip install -r requirements.txt"}
    base_context.task.venv_executables = {"pip": "/bin/pip"}
    
    exec_result = ExecResult(returncode=1, stdout="", stderr="Installation failed")
    
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data="dependency")), \
         patch("nexus.core.stages.installation.run_command", return_value=exec_result), \
         patch("nexus.core.stages.installation.validate_result", return_value=False), \
         patch("nexus.core.stages.installation.build_error_context", return_value="Install error"):
        res = stage.execute(base_context)
        
    assert res.success is False
    assert res.next_stage == TaskStatus.FIXING
    assert base_context.task.last_error_context == "Install error"

def test_validation_stage_success(base_context):
    stage = ValidationStage()
    assert stage.name == "validation"
    
    base_context.task.status = TaskStatus.EXECUTING
    base_context.task.plan = {"test_command": "pytest"}
    exec_result = ExecResult(returncode=0, stdout="All tests passed", stderr="")
    mock_vr = MagicMock(is_success=True)
    
    with patch.object(stage, "_run_tests", return_value=exec_result), \
         patch("nexus.core.stages.validation.validate_result", return_value=mock_vr):
        res = stage.execute(base_context)
        
    assert res.success is True
    assert base_context.task.status == TaskStatus.VALIDATING

def test_validation_stage_failed(base_context):
    stage = ValidationStage()
    base_context.task.status = TaskStatus.EXECUTING
    base_context.task.plan = {"test_command": "pytest"}
    exec_result = ExecResult(returncode=1, stdout="", stderr="Tests failed")
    mock_vr = MagicMock(is_success=False)
    
    with patch.object(stage, "_run_tests", return_value=exec_result), \
         patch("nexus.core.stages.validation.validate_result", return_value=mock_vr), \
         patch("nexus.core.stages.validation.build_error_context", return_value="Test failure context"):
        res = stage.execute(base_context)
        
    assert res.success is False
    assert res.next_stage == TaskStatus.FIXING
    assert base_context.task.last_error_context == "Test failure context"
    assert base_context.task.last_validation_result == mock_vr

def test_repair_stage_success_first_iteration(base_context):
    stage = RepairStage()
    assert stage.name == "repair"
    
    base_context.task.status = TaskStatus.VALIDATING
    base_context.task.plan = {"test_command": "pytest"}
    base_context.task.generated_files = {"main.py": "print('fail')"}
    base_context.task.last_error_context = "AssertionError"
    base_context.task.last_validation_result = MagicMock()
    
    mock_fixed_files = {"main.py": "print('fixed')"}
    base_context.ai.generate_fix.return_value = mock_fixed_files
    
    # Mocking validation success on iteration 1
    val_res = StageResult(success=True, task=base_context.task, metadata={"validation_result": MagicMock(is_success=True)})
    
    with patch("nexus.core.stages.repair.classify_error", return_value="TEST_FAILURE"), \
         patch("nexus.core.stages.repair.inject_conftest_if_needed"), \
         patch("nexus.core.stages.repair.normalize_test_command"), \
         patch("nexus.core.stages.repair.ValidationStage.execute", return_value=val_res), \
         patch("nexus.core.stages.repair.CompletionStage.execute") as mock_comp:
        
        res = stage.execute(base_context)
        
    assert res.success is True
    assert base_context.task.generated_files == mock_fixed_files
    mock_comp.assert_called_once_with(base_context)

def test_repair_stage_exhaustion(base_context):
    stage = RepairStage()
    base_context.task.status = TaskStatus.VALIDATING
    base_context.task.plan = {"test_command": "pytest"}
    base_context.task.generated_files = {"main.py": "print('fail')"}
    base_context.task.last_error_context = "AssertionError"
    base_context.task.last_validation_result = MagicMock()
    
    mock_fixed_files = {"main.py": "print('still fail')"}
    base_context.ai.generate_fix.return_value = mock_fixed_files
    
    # Validation always fails
    mock_vr = MagicMock(is_success=False)
    mock_vr.stage1_result = MagicMock(returncode=1, stdout="", stderr="AssertionError", timed_out=False, security_blocked=False)
    val_res = StageResult(success=False, task=base_context.task, metadata={"validation_result": mock_vr})
    
    with patch("nexus.core.stages.repair.classify_error", return_value="TEST_FAILURE"), \
         patch("nexus.core.stages.repair.inject_conftest_if_needed"), \
         patch("nexus.core.stages.repair.normalize_test_command"), \
         patch("nexus.core.stages.repair.ValidationStage.execute", return_value=val_res), \
         patch("nexus.core.stages.repair.build_error_context", return_value="AssertionError"), \
         pytest.raises(MaxRetriesExceeded) as exc_info:
        
        stage.execute(base_context)
        
    assert "Fix loop exhausted" in str(exc_info.value)

def test_repair_stage_timeout_abort(base_context):
    stage = RepairStage()
    base_context.task.status = TaskStatus.VALIDATING
    base_context.task.plan = {"test_command": "pytest"}
    base_context.task.generated_files = {"main.py": "print('fail')"}
    base_context.task.last_error_context = "AssertionError"
    base_context.task.last_validation_result = MagicMock()
    
    base_context.ai.generate_fix.return_value = {"main.py": "print('still fail')"}
    
    mock_vr = MagicMock(is_success=False)
    mock_vr.stage1_result = MagicMock(returncode=-1, stdout="", stderr="Timeout", timed_out=True, security_blocked=False)
    val_res = StageResult(success=False, task=base_context.task, metadata={"validation_result": mock_vr})
    
    with patch("nexus.core.stages.repair.classify_error", return_value="TEST_FAILURE"), \
         patch("nexus.core.stages.repair.inject_conftest_if_needed"), \
         patch("nexus.core.stages.repair.normalize_test_command"), \
         patch("nexus.core.stages.repair.ValidationStage.execute", return_value=val_res), \
         pytest.raises(MaxRetriesExceeded) as exc_info:
        
        stage.execute(base_context)
        
    assert "Execution timed out" in str(exc_info.value)

def test_repair_stage_security_abort(base_context):
    stage = RepairStage()
    base_context.task.status = TaskStatus.VALIDATING
    base_context.task.plan = {"test_command": "pytest"}
    base_context.task.generated_files = {"main.py": "print('fail')"}
    base_context.task.last_error_context = "AssertionError"
    base_context.task.last_validation_result = MagicMock()
    
    base_context.ai.generate_fix.return_value = {"main.py": "print('still fail')"}
    
    mock_vr = MagicMock(is_success=False)
    mock_vr.stage1_result = MagicMock(returncode=-2, stdout="", stderr="Forbidden import", timed_out=False, security_blocked=True)
    val_res = StageResult(success=False, task=base_context.task, metadata={"validation_result": mock_vr})
    
    with patch("nexus.core.stages.repair.classify_error", return_value="TEST_FAILURE"), \
         patch("nexus.core.stages.repair.inject_conftest_if_needed"), \
         patch("nexus.core.stages.repair.normalize_test_command"), \
         patch("nexus.core.stages.repair.ValidationStage.execute", return_value=val_res), \
         pytest.raises(MaxRetriesExceeded) as exc_info:
        
        stage.execute(base_context)
        
    assert "Security violation" in str(exc_info.value)

def test_completion_stage(base_context):
    stage = CompletionStage()
    assert stage.name == "completion"
    
    res = stage.execute(base_context)
    assert res.success is True
    assert base_context.task.status == TaskStatus.DONE
    assert isinstance(base_context.task.result, TaskResult)
    assert base_context.task.result.success is True
