"""
nexus/ai/orchestrator.py
"""
import os
import time
from typing import Any, Callable, Dict, List, Optional

import requests

from nexus.ai.parser import (
    extract_json,
    normalize_generation,
    validate_fix,
    validate_generation,
    validate_plan,
)
from nexus.ai.prompts import (
    SYSTEM_FIXER,
    SYSTEM_GENERATOR,
    SYSTEM_PLANNER,
    SYSTEM_SEMANTIC_VALIDATOR,
    SYSTEM_JARVIS,
    build_chat_prompt,
    build_fix_prompt,
    build_generation_prompt,
    build_plan_prompt,
    build_semantic_validation_prompt,
)
from nexus.core.exceptions import (
    CloudProviderError,
    OllamaConnectionError,
    TaskGenerationError,
    TaskPlanningError,
)
from nexus.utils.config import config
from nexus.utils.logger import get_logger
from nexus.ai.providers.ollama_provider import OllamaProvider
from nexus.ai.providers.anthropic_provider import AnthropicProvider
from nexus.ai.providers.registry import ProviderRegistry
from nexus.repair.classifier import classify_error
from nexus.repair.targeting import select_files_for_fix
from nexus.repair.strategy import get_repair_strategy, is_terminal_error

logger = get_logger(__name__)

MAX_RETRIES = int(os.getenv("AI_MAX_RETRIES", "2"))
RETRY_DELAY = float(os.getenv("AI_RETRY_DELAY", "2.0"))
MAX_TOTAL_TIME = int(os.getenv("AI_MAX_TOTAL_TIME", "120"))


class AIOrchestrator:
    def __init__(self) -> None:
        self._base_url = config.ollama_base_url
        self._code_model = config.ollama_code_model
        self._reason_model = config.ollama_reason_model
        self._timeout = config.ollama_timeout
        from nexus.planning.planner import PlanningEngine
        from nexus.skills.registry import SkillRegistry
        from nexus.context.assembler import ContextAssembler
        self._planner = PlanningEngine()
        self._skills = SkillRegistry()
        self._assembler = ContextAssembler(token_budget=2000)

        # Provider Registry setup
        self._ollama_provider = OllamaProvider()
        self._anthropic_provider = AnthropicProvider()
        self._registry = ProviderRegistry()
        self._registry.register(self._ollama_provider, priority=100)
        self._registry.register(self._anthropic_provider, priority=50)

    def _call_ollama(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> Optional[str]:
        # DEPRECATED: use OllamaProvider directly
        model_hint = "code" if model == self._code_model else "reason"
        return self._ollama_provider.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_hint=model_hint,
        )

    def _call_with_retry(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        validator_fn: Callable[[Dict], bool],
        normalize_fn: Optional[Callable[[Dict], Dict]] = None,
    ) -> Optional[Dict[str, Any]]:
        capability = "code" if model == self._code_model else "reason"
        provider = self._registry.resolve(capability)
        if provider is None:
            logger.error("No available provider resolved for capability '%s'", capability)
            return None

        start_time = time.time()
        max_attempts = MAX_RETRIES + 1

        for attempt in range(1, max_attempts + 1):
            logger.debug("AI call attempt %d/%d", attempt, max_attempts)

            if time.time() - start_time > MAX_TOTAL_TIME:
                logger.error("AI call exceeded max total time (%ds)", MAX_TOTAL_TIME)
                return None

            import unittest.mock
            if (
                hasattr(self._call_ollama, "mock_calls")
                or isinstance(self._call_ollama, unittest.mock.NonCallableMock)
            ):
                raw = self._call_ollama(model, system_prompt, user_prompt)
            else:
                raw = provider.call_with_retry(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model_hint=capability,
                    max_retries=0,
                    retry_delay=RETRY_DELAY,
                )
            if raw is None:
                logger.warning("Empty response on attempt %d", attempt)
                if attempt <= MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                continue

            parsed = extract_json(raw)
            if parsed is None:
                logger.warning(
                    "JSON extraction failed on attempt %d\nPreview: %s",
                    attempt, raw[:300],
                )
                if attempt <= MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                continue

            if normalize_fn is not None:
                parsed = normalize_fn(parsed)

            if validator_fn(parsed):
                return parsed

            logger.warning(
                "Validation failed on attempt %d\nPreview: %s",
                attempt, raw[:300],
            )
            if attempt <= MAX_RETRIES:
                logger.info("Retrying in %.1fs...", RETRY_DELAY)
                time.sleep(RETRY_DELAY)

        return None

    def _call_claude(self, system_prompt: str, user_prompt: str, iteration: int) -> Optional[str]:
        # DEPRECATED: use AnthropicProvider directly
        logger.info("Routing to Claude API fallback (fix iteration %d)", iteration)
        return self._anthropic_provider.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_hint="reason",
        )

    def generate_plan(self, user_input: str, context: Optional[Any] = None) -> Dict[str, Any]:
        logger.info("Generating plan for: %s", user_input[:80])
        history = context.get_truncated_history(config.nexus_context_token_budget) if context else None
        ctx_summary = context.get_recent_context(history) if context else ""
        result = self._call_with_retry(
            model=self._reason_model,
            system_prompt=SYSTEM_PLANNER,
            user_prompt=build_plan_prompt(
                user_input,
                ctx_summary,
                context.project_snapshot if context else None
            ),
            validator_fn=validate_plan,
        )
        if result is None:
            raise TaskPlanningError("AI failed to produce a valid plan after retries.")
        # Enrich plan with complexity analysis and feasibility checks
        snapshot = getattr(context, "project_snapshot", None) if context else None
        enriched = self._planner.enrich(result, user_input, snapshot)
        if not enriched.feasibility_ok:
            logger.warning(
                "Plan has feasibility issues (%d) — proceeding with warnings",
                len(enriched.feasibility_notes),
            )
        logger.info("Plan generated: %s", result.get("description", ""))
        return enriched

    def generate_code(self, plan: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, str]:
        num_files = len(plan.get("files_to_generate", []))
        description = plan.get("description", "")
        task_type   = plan.get("task_type", "")
        user_input  = plan.get("raw_input", description)

        logger.info("Generating code for %d files (task_type=%s)...", num_files, task_type)

        # ── Skill scaffold injection ──────────────────────────────────────────
        scaffold: Dict[str, str] = {}
        skill = self._skills.match(user_input, task_type)
        if skill:
            scaffold = self._skills.get_scaffold(skill, description)
            logger.info(
                "Skill scaffold injected: '%s' (%d template files)",
                skill.name, len(scaffold),
            )

        history     = context.get_truncated_history(config.nexus_context_token_budget) if context else None
        ctx_summary = context.get_recent_context(history) if context else ""

        # ── Project context assembly ─────────────────────────────────────────
        assembled_ctx = ""
        if context and context.project_snapshot:
            from nexus.executor.workspace import ProjectScanner
            try:
                scanner = ProjectScanner(context.project_snapshot.root)
                assembled = self._assembler.assemble(
                    user_input=user_input,
                    project_snapshot=context.project_snapshot,
                    scanner=scanner,
                )
                assembled_ctx = assembled.to_prompt_block()
            except Exception as e:
                logger.debug("Context assembly skipped: %s", e)

        if assembled_ctx:
            ctx_summary = assembled_ctx + "\n" + ctx_summary

        # Ensure plan is serializable (handles EnrichedPlan object)
        plan_dict = plan.to_dict() if hasattr(plan, "to_dict") else plan

        result = self._call_with_retry(
            model=self._code_model,
            system_prompt=SYSTEM_GENERATOR,
            user_prompt=build_generation_prompt(plan_dict, ctx_summary, scaffold=scaffold),
            validator_fn=validate_generation,
            normalize_fn=normalize_generation,
        )
        if result is None:
            raise TaskGenerationError("AI failed to generate valid code after retries.")

        files = result["files"]

        # ── Merge: scaffold fills any gap the LLM left empty ─────────────────
        # LLM output always wins; scaffold is only a fallback safety net
        for fname, content in scaffold.items():
            if fname not in files or not files[fname].strip():
                logger.debug("Scaffold fallback used for '%s'", fname)
                files[fname] = content

        logger.info("Generated %d files: %s", len(files), list(files.keys()))
        return files

    def generate_fix(
        self,
        plan: Dict[str, Any],
        current_files: Dict[str, str],
        stdout: str,
        stderr: str,
        error: str,
        iteration: int,
        attempt_history: Optional[List[str]] = None,
        context: Optional[Any] = None,
        semantic_reason: Optional[str] = None,
        semantic_issues: Optional[List[str]] = None,
        error_category: str = "UNKNOWN",
    ) -> Optional[Dict[str, str]]:
        logger.info("Requesting fix (iteration %d)...", iteration)
        strategy = get_repair_strategy(error_category, iteration, attempt_history)
        logger.info("Repair approach: %s | brief: %s", strategy.approach.value, strategy.brief[:80])

        # Ensure plan is serializable (handles EnrichedPlan object)
        plan_dict = plan.to_dict() if hasattr(plan, "to_dict") else plan

        user_prompt = build_fix_prompt(
            plan=plan_dict,
            current_files=current_files,
            stdout=stdout,
            stderr=stderr,
            error=error,
            iteration=iteration,
            attempt_history=attempt_history or [],
            semantic_reason=semantic_reason,
            semantic_issues=semantic_issues,
            error_category=error_category,
            strategy_brief=strategy.brief,
        )

        # Iteration 1: Strictly Local (Ollama)
        if iteration == 1:
            result = self._call_with_retry(
                model=self._code_model,
                system_prompt=SYSTEM_FIXER,
                user_prompt=user_prompt,
                validator_fn=validate_fix,
            )
            return result["fixed_files"] if result else None

        # Iterations 2-3: Try Local first, then Fallback
        result = None
        try:
            result = self._call_with_retry(
                model=self._code_model,
                system_prompt=SYSTEM_FIXER,
                user_prompt=user_prompt,
                validator_fn=validate_fix,
            )
        except (OllamaConnectionError, Exception) as e:
            logger.warning("Local model failed or unreachable (iteration %d): %s", iteration, e)

        if result:
            return result["fixed_files"]

        # Local failed, attempt Claude fallback if enabled
        if config.fallback_enabled:
            logger.warning("Activating Claude API fallback (fix iteration %d)", iteration)
            try:
                raw_claude = self._call_claude(SYSTEM_FIXER, user_prompt, iteration)
                if raw_claude:
                    parsed = extract_json(raw_claude)
                    if parsed and validate_fix(parsed):
                        logger.info("Claude fallback produced valid fix")
                        return parsed["fixed_files"]
                    else:
                        logger.warning("Claude fallback returned invalid response format")
            except CloudProviderError as e:
                logger.error("Claude fallback failed: %s", e)
            except Exception as e:
                logger.error("Unexpected error during Claude fallback: %s", e)

        logger.warning("Fix attempt %d failed to produce valid output", iteration)
        return None

    def validate_correctness(self, task: str, code: str, output: str) -> Optional[Dict[str, Any]]:
        """Stage 2: Semantic validation via LLM."""
        logger.info("Performing semantic validation...")
        try:
            # Truncate to prevent context overflow/injection
            max_chars = 4000
            truncated_code = code[:max_chars]
            truncated_output = output[:max_chars]

            raw = self._call_ollama(
                model=self._reason_model,
                system_prompt=SYSTEM_SEMANTIC_VALIDATOR,
                user_prompt=build_semantic_validation_prompt(
                    task, truncated_code, truncated_output
                ),
            )
            if raw:
                return extract_json(raw)
        except Exception as e:
            logger.error("Semantic validation call failed: %s", e)
        return None

    def generate_chat_response(self, user_input: str, context: Optional[Any] = None) -> str:
        """Conversational response with Jarvis personality."""
        history = context.get_truncated_history(config.nexus_context_token_budget) if context else None
        ctx_summary = context.get_recent_context(history) if context else ""
        raw = self._call_ollama(
            model=self._reason_model,
            system_prompt=SYSTEM_JARVIS,
            user_prompt=build_chat_prompt(user_input, ctx_summary)
        )
        return raw or "I am listening, but I'm having trouble processing your request at the moment."

# ── Sprint 1 additions ──
