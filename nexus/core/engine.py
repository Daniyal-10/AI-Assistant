"""
nexus/core/engine.py
────────────────────
Task execution engine. Coordinates the full pipeline by constructing and executing
isolated pipeline stages via a synchronous sequential Pipeline orchestrator.
"""
import os
from typing import Optional, Any

from nexus.ai.code_intel import CodeIntelligence
from nexus.ai.orchestrator import AIOrchestrator
from nexus.ai.router import IntentRouter, IntentType, IntentResult
from nexus.core.exceptions import (
    MaxRetriesExceeded,
    NexusBaseException,
    OllamaConnectionError,
    SafetyViolation,
    TaskGenerationError,
    TaskPlanningError,
)
from nexus.core.task import Task, TaskResult, TaskStatus
from nexus.core.events import (
    EventBus,
    TaskStartedEvent,
    TaskFinishedEvent,
    PipelineStageStartedEvent,
    PipelineStageFinishedEvent,
    NexusEvent,
)
from nexus.executor.safe_exec import create_task_venv, get_venv_executables
from nexus.executor.executor_factory import get_executor
from nexus.executor.workspace import Workspace
from nexus.utils.config import config
from nexus.utils.history import TaskHistory
from nexus.memory.manager import get_memory
from nexus.utils.logger import get_logger

# Pipeline Stage Isolation Imports
from nexus.core.pipeline import Pipeline, TaskExecutionContext
from nexus.core.stages import (
    PlanningStage,
    GenerationStage,
    InstallStage,
    ValidationStage,
    RepairStage,
    CompletionStage,
)

logger = get_logger(__name__)


class TaskEngine:
    def __init__(self, context: Optional[Any] = None) -> None:
        import atexit

        self.ai = AIOrchestrator()
        logger.info("TaskEngine initialized with dynamic fallback registry")

        # Connection check (fail-fast behavior)
        from nexus.ai.providers.ollama_provider import OllamaProvider
        _ollama = OllamaProvider()
        if not _ollama.is_available():
            logger.warning(
                "⚠️  Ollama is not reachable at %s. "
                "Tasks requiring AI will fail. "
                "Start Ollama with: ollama serve",
                config.ollama_base_url,
            )
        else:
            logger.info("✅ Ollama connection verified at %s", config.ollama_base_url)
        self.context = context
        self.history = TaskHistory()
        self.memory  = get_memory()
        self.code_intel = CodeIntelligence(self.ai)
        self._executor = get_executor()
        logger.info("TaskEngine initialized with %s", self._executor.__class__.__name__)

        self.event_bus = EventBus()
        self.event_history = []

        # Register default logging subscriber
        def _log_event(event: NexusEvent) -> None:
            logger.info("[EventBus] Emitted %s: %s", event.__class__.__name__, event.to_dict())
        self.event_bus.subscribe(NexusEvent, _log_event)

        # Register default memory subscriber
        def _record_event(event: NexusEvent) -> None:
            self.event_history.append(event)
        self.event_bus.subscribe(NexusEvent, _record_event)

        try:
            from nexus.executor.docker_exec import cleanup_orphaned_containers, is_docker_available
            if is_docker_available():
                cleaned = cleanup_orphaned_containers()
                if cleaned > 0:
                    logger.warning(
                        "Cleaned up %d orphaned containers from previous session", cleaned
                    )
            atexit.register(self._shutdown_cleanup)
        except ImportError:
            logger.debug("Docker SDK not found — skipping container cleanup")

    # ══════════════════════════════════════════════════════════════════════════
    # Public entry point
    # ══════════════════════════════════════════════════════════════════════════

    def run(self, user_input: str, intent_result: Any = None) -> Any:
        # ── Input guard: oversized input defaults to CHAT, no LLM call ────────
        MAX_INPUT_CHARS = 8000
        if len(user_input) > MAX_INPUT_CHARS:
            logger.warning(
                "Input too long (%d chars) — truncating to %d and defaulting to CHAT",
                len(user_input), MAX_INPUT_CHARS,
            )
            user_input = user_input[:MAX_INPUT_CHARS]
            # Skip routing — oversized inputs are never task requests
            if intent_result is None:
                intent_result = IntentResult(
                    intent=IntentType.CHAT,
                    confidence=1.0,
                    reasoning="Input exceeded maximum length — defaulted to CHAT",
                    raw_input=user_input,
                )

        task = Task(raw_input=user_input)
        self.event_bus.emit(TaskStartedEvent(task_id=task.id, raw_input=task.raw_input))

        logger.info("=" * 60)
        logger.info("Starting task [%s]: %s", task.id, task.raw_input[:60])
        logger.info("=" * 60)

        workspace = None

        try:
            # ── Intent routing ──────────────────────────────────────────────
            if intent_result is None:
                router = IntentRouter(self.ai)
                intent_result = router.route(task.raw_input)

            _intent = intent_result.intent
            # Normalize intent to string safely
            if isinstance(_intent, IntentType):
                intent_val = _intent.value
            elif isinstance(_intent, str):
                intent_val = _intent.upper()
            elif hasattr(_intent, "value"):
                intent_val = str(_intent.value).upper()
            else:
                intent_val = str(_intent).upper()
            task.intent = intent_result

            if self.context:
                self.context.add_message("user", task.raw_input)

            # ── Non-task intents ────────────────────────────────────────────
            if intent_val != IntentType.TASK.value:
                logger.info("Non-task intent detected: %s", intent_val)
                # Normalize intent_result
                if not isinstance(intent_result, IntentResult):
                    try:
                        _norm_intent = IntentType(intent_val)
                    except ValueError:
                        _norm_intent = IntentType.CHAT
                    intent_result = IntentResult(
                        intent=_norm_intent,
                        confidence=1.0,
                        reasoning="Normalized from mock/string",
                        raw_input=task.raw_input,
                    )
                    task.intent = intent_result
                res = self._handle_non_task(task, intent_result)
                if self.context:
                    self.context.add_message("assistant", res.result.summary)
                    self.context.add_task_result(
                        task.id, res.result.summary, intent_val, res.status.name
                    )
                return res

            # ── Workspace setup ─────────────────────────────────────────────
            workspace = Workspace(task.id)
            workspace.create()
            task.workspace_path = workspace.get_path()
            venv_path = create_task_venv(workspace.get_path())
            task.venv_executables = get_venv_executables(venv_path)

            # ── Pipeline stage isolation coordination ────────────────────────
            config_snapshot = {
                "max_fix_iterations": config.max_fix_iterations,
                "exec_timeout": config.exec_timeout,
                "ollama_base_url": config.ollama_base_url,
            }

            ctx = TaskExecutionContext(
                task=task,
                session=self.context,
                ai=self.ai,
                workspace=workspace,
                venv_path=venv_path,
                event_bus=self.event_bus,
                config_snapshot=config_snapshot,
                engine=self,
            )

            stages = [
                PlanningStage(),
                GenerationStage(),
                InstallStage(),
                ValidationStage(),
                RepairStage(),
                CompletionStage(),
            ]

            pipeline = Pipeline(stages)
            pipeline.run(ctx)

        except OllamaConnectionError as e:
            self._fail_task(task, f"Ollama unreachable: {e}")
        except (TaskPlanningError, TaskGenerationError) as e:
            self._fail_task(task, f"AI error: {e}")
        except SafetyViolation as e:
            self._fail_task(task, f"Security blocked: {e}")
        except MaxRetriesExceeded as e:
            self._fail_task(task, str(e))
        except NexusBaseException as e:
            self._fail_task(task, f"NEXUS error: {e}")
        except Exception as e:
            logger.exception("Unexpected error in engine")
            self._fail_task(task, f"Unexpected: {e}")

        finally:
            if self.context and task.result:
                intent_obj = getattr(task, "intent", None)
                intent_type = getattr(intent_obj, "intent", "UNKNOWN")
                intent_str = (
                    intent_type.value
                    if hasattr(intent_type, "value")
                    else str(intent_type)
                )
                self.context.add_task_result(
                    task.id, task.result.summary, intent_str, task.status.name
                )

            self.history.record(
                task,
                self.context.session_id if self.context else "NO_SESSION",
            )
            # Persist to queryable memory store
            _intent_str = "UNKNOWN"
            if task.intent:
                _raw = getattr(task.intent, "intent", task.intent)
                if hasattr(_raw, "value"):
                    _intent_str = _raw.value
                elif isinstance(_raw, str):
                    _intent_str = _raw.upper()
                else:
                    _intent_str = str(_raw).upper()

            self.memory.record_execution(
                session_id=self.context.session_id if self.context else 'NO_SESSION',
                intent=_intent_str,
                raw_input=task.raw_input,
                status=task.status.name,
                summary=task.result.summary if task.result else '',
                fix_attempts=task.fix_iteration,
            )

            if task.status == TaskStatus.FAILED and workspace is not None:
                try:
                    workspace.cleanup()
                except Exception:
                    logger.warning(
                        "Workspace cleanup failed for task %s", task.id
                    )

            self.event_bus.emit(TaskFinishedEvent(
                task_id=task.id,
                success=task.result.success if task.result else False,
                summary=task.result.summary if task.result else "",
                status=task.status.name
            ))

        return task

    # ══════════════════════════════════════════════════════════════════════════
    # Non-Task Intent & Safety Handling
    # ══════════════════════════════════════════════════════════════════════════

    def _stage_test(self, task: Task, workspace: Any) -> Any:
        """Compatibility helper for executing validation. Delegated to ValidationStage."""
        config_snapshot = {
            "max_fix_iterations": config.max_fix_iterations,
            "exec_timeout": config.exec_timeout,
            "ollama_base_url": config.ollama_base_url,
        }
        ctx = TaskExecutionContext(
            task=task,
            session=self.context,
            ai=self.ai,
            workspace=workspace,
            venv_path=getattr(task, "workspace_path", None),
            event_bus=self.event_bus,
            config_snapshot=config_snapshot,
            engine=self,
        )
        validation_stage = ValidationStage()
        test_result = validation_stage._run_tests(task.plan, workspace, task, ctx)
        from nexus.executor.validator import validate_result
        import json
        return validate_result(
            result=test_result,
            command_type="test",
            task_description=task.raw_input,
            generated_code=json.dumps(task.generated_files, indent=2),
            ai=self.ai,
        )

    def _fix_loop(self, task: Task, workspace: Any, error_ctx: str, last_vr: Any) -> None:
        """Compatibility helper for executing the repair loop. Delegated to RepairStage."""
        task.last_error_context = error_ctx
        task.last_validation_result = last_vr
        
        config_snapshot = {
            "max_fix_iterations": config.max_fix_iterations,
            "exec_timeout": config.exec_timeout,
            "ollama_base_url": config.ollama_base_url,
        }
        ctx = TaskExecutionContext(
            task=task,
            session=self.context,
            ai=self.ai,
            workspace=workspace,
            venv_path=getattr(task, "workspace_path", None),
            event_bus=self.event_bus,
            config_snapshot=config_snapshot,
            engine=self,
        )
        
        repair_stage = RepairStage()
        repair_stage.execute(ctx)

    def _handle_non_task(self, task: Task, intent_result: Any) -> Task:
        """Handle CHAT, CODE, or SYSTEM intents outside the full pipeline."""
        if intent_result.intent == IntentType.CHAT:
            task.result = TaskResult(
                success=True,
                summary=self.ai.generate_chat_response(task.raw_input, self.context),
            )
        elif intent_result.intent == IntentType.CODE:
            summary = "CODE: Analysis complete."
            has_inline_code = any(
                kw in task.raw_input
                for kw in ["def ", "class ", "import ", "```", "return "]
            )
            content = ""
            target_file = ""

            if has_inline_code:
                content = task.raw_input
                target_file = "INLINE"
            else:
                if self.context and self.context.project_snapshot:
                    for f in self.context.project_snapshot.structure:
                        if f in task.raw_input:
                            target_file = f
                            try:
                                from nexus.executor.workspace import ProjectScanner
                                scanner = ProjectScanner(
                                    self.context.project_snapshot.root
                                )
                                content = scanner.read_file(f)
                                break
                            except Exception:
                                continue

            if content:
                if "refactor" in task.raw_input.lower() and target_file != "INLINE":
                    res = self.code_intel.refactor(
                        content, task.raw_input, filename=target_file
                    )
                    summary = (
                        f"REFACTOR [{target_file}]: {res['reasoning']}\n\n"
                        f"```python\n{res['refactored_code']}\n```"
                    )
                elif (
                    "debug" in task.raw_input.lower() or "fix" in task.raw_input.lower()
                ) and target_file != "INLINE":
                    res = self.code_intel.debug(
                        content, "No error log provided.", filename=target_file
                    )
                    summary = (
                        f"DEBUG [{target_file}]: {res['diagnosis']}\n\nFIX: {res['fix']}"
                    )
                elif "review" in task.raw_input.lower() and target_file != "INLINE":
                    res = self.code_intel.review(content, filename=target_file)
                    summary = (
                        f"REVIEW [{target_file}]: Score {res['quality_score']}/10\n"
                        f"Issues: {', '.join(res['issues'])}"
                    )
                else:
                    summary = f"EXPLAIN [{target_file}]:\n" + self.code_intel.explain(
                        content, task.raw_input, filename=target_file
                    )
            else:
                summary = "No matching file found in project context to analyze."

            task.result = TaskResult(success=True, summary=summary)

        elif intent_result.intent == IntentType.SYSTEM:
            task.result = TaskResult(
                success=True,
                summary=(
                    f"SYSTEM: {intent_result.reasoning} "
                    "(Direct system commands blocked for safety)"
                ),
            )

        task.transition(TaskStatus.DONE)
        return task

    def _fail_task(self, task: "Task", reason: str) -> None:
        """Mark task FAILED with human-readable reason. Safe from any pipeline state."""
        self.event_bus.emit(PipelineStageStartedEvent(task_id=task.id, stage="FAILED"))
        try:
            task.transition(TaskStatus.FAILED)
        except Exception:
            task.status = TaskStatus.FAILED
        task.result = TaskResult(
            success=False,
            summary=reason,
            iterations_used=task.fix_iteration,
        )
        logger.error("Task [%s] FAILED: %s", task.id, reason)
        self.event_bus.emit(PipelineStageFinishedEvent(task_id=task.id, stage="FAILED", status="SUCCESS"))

    def _shutdown_cleanup(self) -> None:
        try:
            from nexus.executor.docker_exec import (
                cleanup_orphaned_containers,
                is_docker_available,
            )
            if is_docker_available():
                cleaned = cleanup_orphaned_containers()
                if cleaned > 0:
                    logger.info("Shutdown cleanup: removed %d containers", cleaned)
        except (ImportError, Exception):
            pass

# For backward compatibility with legacy test patches
from nexus.repair.classifier import classify_error
from nexus.executor.validator import validate_result, build_error_context
from nexus.executor.safe_exec import run_command
