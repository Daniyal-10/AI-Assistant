"""
Completion Stage Implementation
"""
from nexus.core.task import TaskStatus, TaskResult
from nexus.core.pipeline import PipelineStage, StageResult, TaskExecutionContext
from nexus.core.events import PipelineStageStartedEvent, PipelineStageFinishedEvent
from nexus.utils.logger import get_logger

logger = get_logger(__name__)

class CompletionStage(PipelineStage):
    @property
    def name(self) -> str:
        return "completion"

    def execute(self, ctx: TaskExecutionContext) -> StageResult:
        task = ctx.task
        task.transition(TaskStatus.DONE)
        stage_name = task.status.name
        
        ctx.event_bus.emit(PipelineStageStartedEvent(task_id=task.id, stage=stage_name))
        try:
            zip_path = ctx.workspace.archive()
            iterations = task.fix_iteration
            
            task.result = TaskResult(
                success=True,
                output_path=zip_path,
                summary=(
                    "Completed successfully"
                    if iterations == 0
                    else f"Fixed after {iterations} iteration(s)"
                ),
                iterations_used=iterations,
            )
            
            logger.info("✅ DONE [%s] — %s", task.id, task.result.summary)
            if ctx.session:
                ctx.session.add_message("assistant", task.result.summary)
                
            ctx.event_bus.emit(PipelineStageFinishedEvent(task_id=task.id, stage=stage_name, status="SUCCESS"))
            return StageResult(success=True, task=task)
        except Exception as e:
            ctx.event_bus.emit(PipelineStageFinishedEvent(task_id=task.id, stage=stage_name, status="FAILED"))
            logger.exception("Error in completion stage")
            raise
