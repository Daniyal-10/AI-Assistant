"""
Generation Stage Implementation
"""
from nexus.core.task import TaskStatus
from nexus.core.pipeline import PipelineStage, StageResult, TaskExecutionContext
from nexus.core.events import PipelineStageStartedEvent, PipelineStageFinishedEvent
from nexus.core.stages.helpers import inject_conftest_if_needed, normalize_test_command
from nexus.utils.logger import get_logger

logger = get_logger(__name__)

class GenerationStage(PipelineStage):
    @property
    def name(self) -> str:
        return "generation"

    def execute(self, ctx: TaskExecutionContext) -> StageResult:
        task = ctx.task
        task.transition(TaskStatus.GENERATING)
        stage_name = task.status.name
        
        ctx.event_bus.emit(PipelineStageStartedEvent(task_id=task.id, stage=stage_name))
        try:
            files = ctx.ai.generate_code(task.plan, context=ctx.session)
            task.generated_files = files
            ctx.workspace.write_files(files)
            inject_conftest_if_needed(ctx.workspace, task)
            normalize_test_command(task, ctx.workspace)
            logger.info("Generate stage complete: %d files written", len(files))
            ctx.event_bus.emit(PipelineStageFinishedEvent(task_id=task.id, stage=stage_name, status="SUCCESS"))
            return StageResult(success=True, task=task)
        except Exception as e:
            ctx.event_bus.emit(PipelineStageFinishedEvent(task_id=task.id, stage=stage_name, status="FAILED"))
            logger.exception("Error in generation stage")
            raise
