"""
tests/test_events.py
────────────────────
Targeted unit and integration tests for the Event Model and EventBus integration.
"""
import pytest
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch
from nexus.core.events import (
    EventBus,
    NexusEvent,
    TaskStartedEvent,
    TaskFinishedEvent,
    PipelineStageStartedEvent,
    PipelineStageFinishedEvent,
    RepairLoopStartedEvent,
    RepairIterationStartedEvent,
    RepairIterationFinishedEvent,
    RepairLoopFinishedEvent,
)
from nexus.core.engine import TaskEngine
from nexus.core.task import Task, TaskStatus, TaskResult
from nexus.ai.router import IntentResult, IntentType
from nexus.core.exceptions import MaxRetriesExceeded


# ── Unit Tests ───────────────────────────────────────────────────────────────

def test_base_event_initialization():
    """Verify that NexusEvent sets event_id, timestamp, and metadata correctly by default."""
    event = NexusEvent()
    
    # event_id must be a valid 32-char hex UUID string
    assert isinstance(event.event_id, str)
    assert len(event.event_id) == 32
    # Verify it is hex
    uuid.UUID(event.event_id)
    
    # timestamp must be a datetime object
    assert isinstance(event.timestamp, datetime)
    
    # metadata must be a dictionary and default to empty
    assert isinstance(event.metadata, dict)
    assert len(event.metadata) == 0


def test_concrete_events_inheritance():
    """Verify subclass events inherit correctly and accept customized values."""
    custom_metadata = {"env": "test", "user": "nexus"}
    event = TaskStartedEvent(
        task_id="task-1",
        raw_input="test input",
        metadata=custom_metadata
    )
    
    assert event.task_id == "task-1"
    assert event.raw_input == "test input"
    assert len(event.event_id) == 32
    assert isinstance(event.timestamp, datetime)
    assert event.metadata == custom_metadata


def test_event_to_dict_serialization():
    """Verify that event.to_dict() serializes properly and handles timestamp formats correctly."""
    custom_metadata = {"env": "prod"}
    event = TaskFinishedEvent(
        task_id="task-2",
        success=True,
        summary="Success summary",
        status="PASS",
        metadata=custom_metadata
    )
    
    serialized = event.to_dict()
    
    assert serialized["event_id"] == event.event_id
    assert serialized["timestamp"] == event.timestamp.isoformat()
    assert serialized["metadata"] == custom_metadata
    assert serialized["event_type"] == "TaskFinishedEvent"
    assert serialized["task_id"] == "task-2"
    assert serialized["success"] is True
    assert serialized["summary"] == "Success summary"
    assert serialized["status"] == "PASS"


# ── EventBus Tests ───────────────────────────────────────────────────────────

def test_event_bus_subscribe_and_emit():
    """Verify subscriber registration, dynamic type matching, and fail-safe emissions."""
    bus = EventBus()
    emitted_events = []
    
    def on_task_started(event: TaskStartedEvent):
        emitted_events.append(event)
        
    bus.subscribe(TaskStartedEvent, on_task_started)
    
    # Emit matching event
    start_ev = TaskStartedEvent(task_id="task-1", raw_input="input-1")
    bus.emit(start_ev)
    
    assert len(emitted_events) == 1
    assert emitted_events[0] == start_ev
    
    # Emit non-matching event (should not be caught)
    finish_ev = TaskFinishedEvent(task_id="task-1", success=True, summary="ok", status="PASS")
    bus.emit(finish_ev)
    assert len(emitted_events) == 1
    
    # Unsubscribe
    bus.unsubscribe(TaskStartedEvent, on_task_started)
    bus.emit(start_ev)
    assert len(emitted_events) == 1


def test_event_bus_polymorphic_subscribers():
    """Verify that subscribing to base NexusEvent correctly matches all subclass emissions."""
    bus = EventBus()
    all_events = []
    
    bus.subscribe(NexusEvent, lambda ev: all_events.append(ev))
    
    ev1 = TaskStartedEvent(task_id="t1", raw_input="r")
    ev2 = PipelineStageStartedEvent(task_id="t1", stage="PLANNING")
    ev3 = RepairLoopStartedEvent(task_id="t1", max_iterations=5)
    
    bus.emit(ev1)
    bus.emit(ev2)
    bus.emit(ev3)
    
    assert len(all_events) == 3
    assert all_events[0] == ev1
    assert all_events[1] == ev2
    assert all_events[2] == ev3


def test_event_bus_failsafe_subscriber_exception():
    """Verify that a crashed subscriber does not break dispatch to other subscribers or the emitter."""
    bus = EventBus()
    results = []
    
    def bad_subscriber(ev):
        raise ValueError("Intentional subscriber crash")
        
    def good_subscriber(ev):
        results.append(ev)
        
    bus.subscribe(NexusEvent, bad_subscriber)
    bus.subscribe(NexusEvent, good_subscriber)
    
    ev = NexusEvent()
    # Should not raise exception
    bus.emit(ev)
    
    assert len(results) == 1
    assert results[0] == ev


# ── Integration Tests ────────────────────────────────────────────────────────

def test_engine_event_emission_sequence():
    """Verify that running a task in TaskEngine emits the full set of lifecycle events in order."""
    engine = TaskEngine()
    user_input = "Run hello"
    intent_res = IntentResult(IntentType.TASK, 1.0, "Task", user_input)
    
    plan = {
        "task_type": "script",
        "description": "Test integration plan",
        "steps": [],
        "files_to_generate": ["main.py"],
        "entry_point": "main.py",
        "test_command": "python3 main.py",
        "install_command": "",
    }
    files = {"main.py": "print('hello')"}
    
    with patch.object(engine.ai, "generate_plan", return_value=plan), \
         patch.object(engine.ai, "generate_code", return_value=files), \
         patch("nexus.core.engine.validate_result", return_value=MagicMock(
             is_success=True, status="PASS", semantic_verdict="CORRECT",
             semantic_issues=[],
             stage1_result=MagicMock(returncode=0, stdout="hello", stderr="", timed_out=False)
         )):
        
        task = engine.run(user_input, intent_res)
        
        # Verify event history recorded in memory subscriber
        events = engine.event_history
        assert len(events) >= 6
        
        # 1. TaskStartedEvent
        assert isinstance(events[0], TaskStartedEvent)
        assert events[0].task_id == task.id
        assert events[0].raw_input == user_input
        
        # 2. PipelineStageStartedEvent for ROUTED/PLANNING/etc
        assert any(isinstance(e, PipelineStageStartedEvent) and e.stage == "PLANNING" for e in events)
        assert any(isinstance(e, PipelineStageFinishedEvent) and e.stage == "PLANNING" and e.status == "SUCCESS" for e in events)
        
        # 3. PipelineStageStartedEvent/FinishedEvent for GENERATING
        assert any(isinstance(e, PipelineStageStartedEvent) and e.stage == "GENERATING" for e in events)
        assert any(isinstance(e, PipelineStageFinishedEvent) and e.stage == "GENERATING" and e.status == "SUCCESS" for e in events)
        
        # 4. PipelineStageStartedEvent/FinishedEvent for EXECUTING
        assert any(isinstance(e, PipelineStageStartedEvent) and e.stage == "EXECUTING" for e in events)
        
        # 5. PipelineStageStartedEvent/FinishedEvent for VALIDATING
        assert any(isinstance(e, PipelineStageStartedEvent) and e.stage == "VALIDATING" for e in events)
        
        # 6. TaskFinishedEvent should be last or near the end
        assert isinstance(events[-1], TaskFinishedEvent)
        assert events[-1].task_id == task.id
        assert events[-1].success is True
        assert events[-1].status == "DONE"


def test_engine_fix_loop_events():
    """Verify that the repair loop emits starting, iteration-level, and ending events correctly on success."""
    engine = TaskEngine()
    task = Task(raw_input="Fix loop test input")
    task.id = "test-task-123"
    task.status = TaskStatus.VALIDATING
    task.plan = {"test_command": "python3 -m pytest"}
    task.generated_files = {"main.py": "print('fail')"}
    workspace = MagicMock()
    
    # Mock error identification, generate_fix, etc.
    with patch("nexus.core.engine.classify_error", return_value="SYNTAX_ERROR"), \
         patch.object(engine.ai, "generate_fix", return_value={"main.py": "print('fixed')"}), \
         patch.object(engine, "_stage_test") as mock_stage_test:
        
        # Iteration 1: success
        mock_stage_test.return_value = MagicMock(
            is_success=True,
            semantic_reason=None,
            semantic_issues=[],
            stage1_result=MagicMock(returncode=0, stdout="fixed", stderr="", timed_out=False)
        )
        
        engine._fix_loop(task, workspace, error_ctx="RuntimeError", last_vr=None)
        
        events = engine.event_history
        
        # RepairLoopStartedEvent
        assert any(isinstance(e, RepairLoopStartedEvent) and e.task_id == "test-task-123" and e.max_iterations > 0 for e in events)
        
        # PipelineStageStartedEvent("FIXING")
        assert any(isinstance(e, PipelineStageStartedEvent) and e.stage == "FIXING" for e in events)
        
        # RepairIterationStartedEvent
        assert any(isinstance(e, RepairIterationStartedEvent) and e.iteration == 1 and e.error_category == "SYNTAX_ERROR" for e in events)
        
        # RepairIterationFinishedEvent
        assert any(isinstance(e, RepairIterationFinishedEvent) and e.iteration == 1 and e.success is True and e.error_category == "SYNTAX_ERROR" for e in events)
        
        # RepairLoopFinishedEvent
        assert any(isinstance(e, RepairLoopFinishedEvent) and e.success is True and e.iterations_used == 1 for e in events)
        
        # PipelineStageFinishedEvent("FIXING", "SUCCESS")
        assert any(isinstance(e, PipelineStageFinishedEvent) and e.stage == "FIXING" and e.status == "SUCCESS" for e in events)


def test_engine_fix_loop_exhausted_events():
    """Verify that the repair loop emits failure and iteration events correctly on exhaustion."""
    engine = TaskEngine()
    task = Task(raw_input="Fix loop fail input")
    task.id = "test-task-456"
    task.status = TaskStatus.VALIDATING
    task.plan = {"test_command": "python3 -m pytest"}
    task.generated_files = {"main.py": "print('fail')"}
    workspace = MagicMock()
    
    with patch("nexus.core.engine.classify_error", return_value="TEST_FAILURE"), \
         patch.object(engine.ai, "generate_fix", return_value={"main.py": "print('still fail')"}), \
         patch.object(engine, "_stage_test") as mock_stage_test, \
         patch("nexus.core.engine.config") as mock_config:
        
        mock_config.max_fix_iterations = 2
        
        # Always fail
        stage1 = MagicMock(returncode=1, stdout="", stderr="AssertionError")
        stage1.timed_out = False
        stage1.security_blocked = False
        
        mock_stage_test.return_value = MagicMock(
            is_success=False,
            semantic_reason=None,
            semantic_issues=[],
            stage1_result=stage1
        )
        
        with pytest.raises(MaxRetriesExceeded):
            engine._fix_loop(task, workspace, error_ctx="AssertionError", last_vr=None)
            
        events = engine.event_history
        
        # RepairLoopStartedEvent
        assert any(isinstance(e, RepairLoopStartedEvent) and e.task_id == "test-task-456" for e in events)
        
        # PipelineStageStartedEvent("FIXING")
        assert any(isinstance(e, PipelineStageStartedEvent) and e.stage == "FIXING" for e in events)
        
        # RepairIterationStartedEvent(iteration=1) and (iteration=2)
        assert any(isinstance(e, RepairIterationStartedEvent) and e.iteration == 1 for e in events)
        assert any(isinstance(e, RepairIterationStartedEvent) and e.iteration == 2 for e in events)
        
        # RepairIterationFinishedEvent(iteration=1, success=False) and (iteration=2, success=False)
        assert any(isinstance(e, RepairIterationFinishedEvent) and e.iteration == 1 and e.success is False for e in events)
        assert any(isinstance(e, RepairIterationFinishedEvent) and e.iteration == 2 and e.success is False for e in events)
        
        # RepairLoopFinishedEvent(success=False, iterations_used=2)
        assert any(isinstance(e, RepairLoopFinishedEvent) and e.success is False and e.iterations_used == 2 for e in events)
        
        # PipelineStageFinishedEvent("FIXING", "FAILED")
        assert any(isinstance(e, PipelineStageFinishedEvent) and e.stage == "FIXING" and e.status == "FAILED" for e in events)
