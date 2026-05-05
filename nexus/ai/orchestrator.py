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
    OllamaConnectionError,
    TaskGenerationError,
    TaskPlanningError,
)
from nexus.utils.config import config
from nexus.utils.logger import get_logger

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

    def _call_ollama(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> Optional[str]:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 4096,
            },
        }
        try:
            response = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content")
            if not content:
                logger.warning("Ollama returned empty content: %s", data)
                return None
            return content
        except requests.exceptions.ConnectionError as e:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self._base_url}. "
                f"Is Ollama running? Error: {e}"
            ) from e
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out after %ds", self._timeout)
            return None
        except requests.exceptions.RequestException as e:
            logger.error("Ollama request failed: %s", e)
            return None

    def _call_with_retry(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        validator_fn: Callable[[Dict], bool],
        normalize_fn: Optional[Callable[[Dict], Dict]] = None,
    ) -> Optional[Dict[str, Any]]:
        start_time = time.time()
        max_attempts = MAX_RETRIES + 1

        for attempt in range(1, max_attempts + 1):
            logger.debug("AI call attempt %d/%d", attempt, max_attempts)

            if time.time() - start_time > MAX_TOTAL_TIME:
                logger.error("AI call exceeded max total time (%ds)", MAX_TOTAL_TIME)
                return None

            raw = self._call_ollama(model, system_prompt, user_prompt)
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

    def generate_plan(self, user_input: str, context: Optional[Any] = None) -> Dict[str, Any]:
        logger.info("Generating plan for: %s", user_input[:80])
        ctx_summary = context.get_recent_context() if context else ""
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
        logger.info("Plan generated: %s", result.get("description", ""))
        return result

    def generate_code(self, plan: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, str]:
        logger.info(
            "Generating code for %d files...",
            len(plan.get("files_to_generate", [])),
        )
        ctx_summary = context.get_recent_context() if context else ""
        result = self._call_with_retry(
            model=self._code_model,
            system_prompt=SYSTEM_GENERATOR,
            user_prompt=build_generation_prompt(plan, ctx_summary),
            validator_fn=validate_generation,
            normalize_fn=normalize_generation,
        )
        if result is None:
            raise TaskGenerationError("AI failed to generate valid code after retries.")
        files = result["files"]
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
    ) -> Optional[Dict[str, str]]:
        logger.info("Requesting fix (iteration %d)...", iteration)
        result = self._call_with_retry(
            model=self._code_model,
            system_prompt=SYSTEM_FIXER,
            user_prompt=build_fix_prompt(
                plan=plan,
                current_files=current_files,
                stdout=stdout,
                stderr=stderr,
                error=error,
                iteration=iteration,
                attempt_history=attempt_history or [],
                semantic_reason=semantic_reason,
                semantic_issues=semantic_issues,
            ),
            validator_fn=validate_fix,
        )
        if result is None:
            logger.warning("Fix attempt %d failed to produce valid output", iteration)
            return None
        fixed = result["fixed_files"]
        explanation = result.get("fix_explanation", "No explanation provided")
        logger.info("Fix explanation: %s", explanation[:200])
        return fixed

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
        ctx_summary = context.get_recent_context() if context else ""
        raw = self._call_ollama(
            model=self._reason_model,
            system_prompt=SYSTEM_JARVIS,
            user_prompt=build_chat_prompt(user_input, ctx_summary)
        )
        return raw or "I am listening, but I'm having trouble processing your request at the moment."
