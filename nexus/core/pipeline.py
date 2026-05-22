"""
NEXUS Pipeline and Execution Contracts
──────────────────────────────────────
Defines the explicit stage abstraction, context boundary, and sequential runner
for Task isolated pipeline stages.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from nexus.core.task import Task, TaskStatus

@dataclass(frozen=True)
class TaskExecutionContext:
    """
    Infrastructure-focused, immutable boundary capturing execution parameters.
    No mutable metadata or shared mutable variables are allowed here.
    """
    task: Task
    session: Optional[Any]
    ai: Any                  # AIOrchestrator
    workspace: Optional[Any]  # Workspace
    venv_path: Optional[str]
    event_bus: Any           # EventBus
    config_snapshot: Dict[str, Any]
    engine: Optional[Any] = None

@dataclass
class StageResult:
    """Explicit output contract returned from every isolated pipeline stage execution."""
    success: bool
    task: Task
    next_stage: Optional[TaskStatus] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class PipelineStage(ABC):
    """Abstract Base Class defining the contract for dumb execution units (Stages)."""
    @property
    @abstractmethod
    def name(self) -> str:
        """The distinct identifying key of the stage."""
        pass

    @abstractmethod
    def execute(self, ctx: TaskExecutionContext) -> StageResult:
        """Synchronously execute stage-specific capabilities using the context block."""
        pass

    def on_failure(self, ctx: TaskExecutionContext, error: Any) -> StageResult:
        """Default failure hook."""
        return StageResult(
            success=False,
            task=ctx.task,
            error=str(error)
        )

class Pipeline:
    """
    Dumb sequential orchestrator. Coordinates stages, passes context,
    and dispatches fail/recovery flow deterministically.
    """
    def __init__(self, stages: List[PipelineStage]) -> None:
        self.stages = {stage.name: stage for stage in stages}

    def run(self, ctx: TaskExecutionContext) -> Task:
        task = ctx.task
        
        # 1. Planning Stage
        planning = self.stages.get("planning")
        if planning:
            res = planning.execute(ctx)
            if not res.success:
                return task
                
        # 2. Generation Stage
        generation = self.stages.get("generation")
        if generation:
            res = generation.execute(ctx)
            if not res.success:
                return task
                
        # 3. Installation Stage
        installation = self.stages.get("installation")
        if installation:
            res = installation.execute(ctx)
            if not res.success:
                # Installation failure delegates execution directly to RepairStage
                repair = self.stages.get("repair")
                if repair:
                    repair.execute(ctx)
                return task
                
        # 4. Validation Stage
        validation = self.stages.get("validation")
        if validation:
            res = validation.execute(ctx)
            if res.success:
                # Validation succeeded -> proceed to completion
                completion = self.stages.get("completion")
                if completion:
                    completion.execute(ctx)
            else:
                # Check for initial timeout or security violation
                vr = res.metadata.get("validation_result")
                if vr:
                    from nexus.core.exceptions import MaxRetriesExceeded
                    if getattr(vr.stage1_result, "timed_out", False) or vr.stage1_result.returncode == -1:
                        raise MaxRetriesExceeded("Execution timed out — aborting (non-fixable)")
                    if getattr(vr.stage1_result, "security_blocked", False):
                        raise MaxRetriesExceeded(
                            f"Security violation — aborting: {vr.stage1_result.stderr[:200]}"
                        )

                # Validation failed -> proceed to repair loop
                repair = self.stages.get("repair")
                if repair:
                    repair.execute(ctx)

        return task
