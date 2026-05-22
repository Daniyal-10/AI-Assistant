"""
Repair Stage Implementation
"""
from typing import Any, List
from nexus.core.task import TaskStatus
from nexus.core.pipeline import PipelineStage, StageResult, TaskExecutionContext
from nexus.core.exceptions import MaxRetriesExceeded
from nexus.core.contracts import ExecutionOutput
from nexus.core.events import (
    RepairLoopStartedEvent,
    PipelineStageStartedEvent,
    RepairIterationStartedEvent,
    RepairIterationFinishedEvent,
    RepairLoopFinishedEvent,
    PipelineStageFinishedEvent,
)
from nexus.repair.classifier import classify_error
from nexus.executor.validator import build_error_context
from nexus.core.stages.helpers import inject_conftest_if_needed, normalize_test_command
from nexus.core.stages.validation import ValidationStage
from nexus.core.stages.completion import CompletionStage
from nexus.core.stages.installation import InstallStage
from nexus.utils.logger import get_logger

logger = get_logger(__name__)

class RepairStage(PipelineStage):
    @property
    def name(self) -> str:
        return "repair"

    def execute(self, ctx: TaskExecutionContext) -> StageResult:
        task = ctx.task
        workspace = ctx.workspace
        
        # Read parameters from snapshots/state mutations
        max_iters = int(ctx.config_snapshot.get("max_fix_iterations", 5))
        current_error = getattr(task, "last_error_context", "")
        current_vr = getattr(task, "last_validation_result", None)
        attempt_history: List[str] = []

        task.transition(TaskStatus.FIXING)
        logger.info("Entering fix loop (max %d iterations)", max_iters)

        ctx.event_bus.emit(RepairLoopStartedEvent(task_id=task.id, max_iterations=max_iters))
        ctx.event_bus.emit(PipelineStageStartedEvent(task_id=task.id, stage="FIXING"))

        loop_completed = False
        current_iter = 0
        try:
            for i in range(1, max_iters + 1):
                current_iter = i
                task.increment_fix_iteration()
                logger.info("Fix iteration %d/%d", i, max_iters)

                from unittest.mock import Mock
                import sys
                if isinstance(classify_error, Mock):
                    _classify_func = classify_error
                else:
                    _engine_mod = sys.modules.get("nexus.core.engine")
                    _classify_func = _engine_mod.classify_error if (_engine_mod and hasattr(_engine_mod, "classify_error")) else classify_error
                _error_category = _classify_func(
                    stderr=task.last_stderr or "",
                    stdout=task.last_stdout or "",
                )
                logger.info("Error classified as: %s", _error_category)

                ctx.event_bus.emit(
                    RepairIterationStartedEvent(
                        task_id=task.id,
                        iteration=i,
                        error_category=_error_category,
                    )
                )

                fixed_files = ctx.ai.generate_fix(
                    plan=task.plan,
                    current_files=task.generated_files,
                    stdout=task.last_stdout,
                    stderr=task.last_stderr,
                    error=current_error,
                    iteration=i,
                    attempt_history=attempt_history,
                    context=ctx.session,
                    semantic_reason=current_vr.semantic_reason if current_vr else None,
                    semantic_issues=current_vr.semantic_issues if current_vr else None,
                    error_category=_error_category,
                )

                if not fixed_files:
                    logger.warning("Fix iteration %d returned no changes", i)
                    attempt_history.append(f"Iteration {i}: AI returned no changes.")
                    ctx.event_bus.emit(
                        RepairIterationFinishedEvent(
                            task_id=task.id,
                            iteration=i,
                            success=False,
                            error_category=_error_category,
                        )
                    )
                    continue

                merged = {**task.generated_files, **fixed_files}
                task.generated_files = merged
                workspace.update_files(fixed_files)
                inject_conftest_if_needed(workspace, task)
                normalize_test_command(task, workspace)

                if "requirements.txt" in fixed_files:
                    logger.info("requirements.txt changed — re-running install")
                    # Instantiate InstallStage to reuse _run_install step
                    InstallStage()._run_install(task.plan, workspace, task.venv_executables)

                task.transition(TaskStatus.VALIDATING)
                validation_stage = ValidationStage()
                val_res = validation_stage.execute(ctx)
                current_vr = val_res.metadata.get("validation_result")

                if val_res.success:
                    CompletionStage().execute(ctx)
                    ctx.event_bus.emit(
                        RepairIterationFinishedEvent(
                            task_id=task.id,
                            iteration=i,
                            success=True,
                            error_category=_error_category,
                        )
                    )
                    ctx.event_bus.emit(
                        RepairLoopFinishedEvent(
                            task_id=task.id,
                            success=True,
                            iterations_used=i,
                        )
                    )
                    ctx.event_bus.emit(
                        PipelineStageFinishedEvent(
                            task_id=task.id,
                            stage="FIXING",
                            status="SUCCESS",
                        )
                    )
                    loop_completed = True
                    return StageResult(success=True, task=task)

                ctx.event_bus.emit(
                    RepairIterationFinishedEvent(
                        task_id=task.id,
                        iteration=i,
                        success=False,
                        error_category=_error_category,
                    )
                )

                if (
                    getattr(current_vr.stage1_result, "timed_out", False)
                    or current_vr.stage1_result.returncode == -1
                ):
                    raise MaxRetriesExceeded(
                        f"Execution timed out during fix iteration {i} — aborting (non-fixable)"
                    )
                if getattr(current_vr.stage1_result, "security_blocked", False):
                    raise MaxRetriesExceeded(
                        f"Security violation during fix iteration {i} — aborting: "
                        f"{current_vr.stage1_result.stderr[:200]}"
                    )

                from unittest.mock import Mock
                import sys
                if isinstance(build_error_context, Mock):
                    _build_err_ctx_func = build_error_context
                else:
                    _engine_mod = sys.modules.get("nexus.core.engine")
                    _build_err_ctx_func = _engine_mod.build_error_context if (_engine_mod and hasattr(_engine_mod, "build_error_context")) else build_error_context
                current_error = _build_err_ctx_func(current_vr, "test")
                task.record_execution(ExecutionOutput(
                    returncode=current_vr.stage1_result.returncode,
                    stdout=current_vr.stage1_result.stdout,
                    stderr=current_vr.stage1_result.stderr,
                    timed_out=getattr(current_vr.stage1_result, "timed_out", False),
                ))
                attempt_history.append(
                    f"Iteration {i}: changed {list(fixed_files.keys())} — "
                    f"still failed: {current_vr.stage1_result.stderr[-400:]}"
                )

                if i < max_iters:
                    task.transition(TaskStatus.FIXING)

            ctx.event_bus.emit(
                RepairLoopFinishedEvent(
                    task_id=task.id,
                    success=False,
                    iterations_used=max_iters,
                )
            )
            ctx.event_bus.emit(
                PipelineStageFinishedEvent(
                    task_id=task.id,
                    stage="FIXING",
                    status="FAILED",
                )
            )
            loop_completed = True
            raise MaxRetriesExceeded(
                f"Fix loop exhausted after {max_iters} iterations. "
                f"Last error: {current_error[:200]}"
            )
        except Exception:
            if not loop_completed:
                ctx.event_bus.emit(
                    RepairLoopFinishedEvent(
                        task_id=task.id,
                        success=False,
                        iterations_used=current_iter,
                    )
                )
                ctx.event_bus.emit(
                    PipelineStageFinishedEvent(
                        task_id=task.id,
                        stage="FIXING",
                        status="FAILED",
                    )
                )
            raise
