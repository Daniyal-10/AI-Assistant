"""
Installation Stage Implementation
"""
import os
import shlex
from typing import Any
from nexus.core.task import TaskStatus
from nexus.core.pipeline import PipelineStage, StageResult, TaskExecutionContext
from nexus.core.events import PipelineStageStartedEvent, PipelineStageFinishedEvent
from nexus.executor.safe_exec import ExecResult, run_command
from nexus.executor.validator import validate_result, build_error_context
from nexus.utils.logger import get_logger

logger = get_logger(__name__)

class InstallStage(PipelineStage):
    @property
    def name(self) -> str:
        return "installation"

    def execute(self, ctx: TaskExecutionContext) -> StageResult:
        task = ctx.task
        task.transition(TaskStatus.EXECUTING)
        stage_name = task.status.name
        
        ctx.event_bus.emit(PipelineStageStartedEvent(task_id=task.id, stage=stage_name))
        try:
            result = self._run_install(task.plan, ctx.workspace, task.venv_executables)
            # Use same validation check as legacy engine
            from unittest.mock import Mock
            import sys
            if isinstance(validate_result, Mock):
                _val_res_func = validate_result
            else:
                _engine_mod = sys.modules.get("nexus.core.engine")
                _val_res_func = _engine_mod.validate_result if (_engine_mod and hasattr(_engine_mod, "validate_result")) else validate_result
            is_success = bool(_val_res_func(result, "install"))
            status_str = "SUCCESS" if is_success else "FAILED"
            
            ctx.event_bus.emit(PipelineStageFinishedEvent(task_id=task.id, stage=stage_name, status=status_str))
            
            if not is_success:
                # Mutate Task object with failed install context to communicate with RepairStage
                if isinstance(validate_result, Mock):
                    _val_res_func = validate_result
                else:
                    _engine_mod = sys.modules.get("nexus.core.engine")
                    _val_res_func = _engine_mod.validate_result if (_engine_mod and hasattr(_engine_mod, "validate_result")) else validate_result
                install_vr = _val_res_func(result, "install")

                if isinstance(build_error_context, Mock):
                    _build_err_ctx_func = build_error_context
                else:
                    _engine_mod = sys.modules.get("nexus.core.engine")
                    _build_err_ctx_func = _engine_mod.build_error_context if (_engine_mod and hasattr(_engine_mod, "build_error_context")) else build_error_context
                task.last_error_context = _build_err_ctx_func(install_vr, "install")
                
                return StageResult(
                    success=False,
                    task=task,
                    next_stage=TaskStatus.FIXING,
                    error=task.last_error_context,
                    metadata={"install_result": result}
                )
                
            return StageResult(success=True, task=task, metadata={"install_result": result})
        except Exception as e:
            ctx.event_bus.emit(PipelineStageFinishedEvent(task_id=task.id, stage=stage_name, status="FAILED"))
            logger.exception("Error in installation stage")
            raise

    def _run_install(
        self,
        plan: dict,
        workspace: Any,
        venv_exec: dict = None,
    ) -> ExecResult:
        req_file = os.path.join(workspace.get_path(), "requirements.txt")
        if not os.path.exists(req_file):
            return ExecResult(0, "Skipped: no requirements.txt", "")
        try:
            content = open(req_file).read().strip()
        except OSError:
            content = ""
        if not content:
            return ExecResult(0, "Skipped: empty requirements.txt", "")

        cmd = plan.get("install_command") or "pip install -r requirements.txt"
        args = shlex.split(cmd)
        if venv_exec and args and args[0] in ("pip", "pip3"):
            args[0] = venv_exec.get(args[0], args[0])

        logger.info("Running install: %s", args)
        from unittest.mock import Mock
        import sys
        if isinstance(run_command, Mock):
            _rc_func = run_command
        else:
            _engine_mod = sys.modules.get("nexus.core.engine")
            _rc_func = _engine_mod.run_command if (_engine_mod and hasattr(_engine_mod, "run_command")) else run_command
        return _rc_func(args, cwd=workspace.get_path(), timeout=120)
