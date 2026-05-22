"""
Planning Stage Implementation
"""
from nexus.core.task import TaskStatus
from nexus.core.pipeline import PipelineStage, StageResult, TaskExecutionContext
from nexus.core.events import PipelineStageStartedEvent, PipelineStageFinishedEvent
from nexus.utils.logger import get_logger

logger = get_logger(__name__)

class PlanningStage(PipelineStage):
    @property
    def name(self) -> str:
        return "planning"

    def execute(self, ctx: TaskExecutionContext) -> StageResult:
        task = ctx.task
        task.transition(TaskStatus.PLANNING)
        stage_name = task.status.name
        
        ctx.event_bus.emit(PipelineStageStartedEvent(task_id=task.id, stage=stage_name))
        try:
            plan = ctx.ai.generate_plan(task.raw_input, context=ctx.session)
            task.plan = plan
            logger.info("Plan stage complete: %s", plan.get("description", ""))
            ctx.event_bus.emit(PipelineStageFinishedEvent(task_id=task.id, stage=stage_name, status="SUCCESS"))
            return StageResult(success=True, task=task)
        except Exception as e:
            ctx.event_bus.emit(PipelineStageFinishedEvent(task_id=task.id, stage=stage_name, status="FAILED"))
            logger.exception("Error in planning stage")
            raise
