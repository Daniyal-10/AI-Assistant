"""
nexus/ai/code_intel.py
──────────────────────
Code Intelligence module — handles explanation, refactoring, debugging, 
and reviewing existing code without execution.
"""
from typing import Any, Dict, Optional

from nexus.ai.orchestrator import AIOrchestrator
from nexus.ai.parser import extract_json
from nexus.ai.prompts import (
    SYSTEM_CODE_DEBUGGER,
    SYSTEM_CODE_EXPLAINER,
    SYSTEM_CODE_REFACTORER,
    SYSTEM_CODE_REVIEWER,
    build_debug_prompt,
    build_explain_prompt,
    build_refactor_prompt,
    build_review_prompt,
)
from nexus.utils.config import config
from nexus.utils.logger import get_logger

logger = get_logger(__name__)


class CodeIntelligence:
    """
    Handles read-only reasoning about existing codebases.
    Integrates with AIOrchestrator to communicate with Ollama.
    """
    def __init__(self, ai: AIOrchestrator) -> None:
        self.ai = ai
        self.max_chars = 8000

    def _prepare(self, content: str) -> str:
        """Sanitize and truncate content for LLM safety."""
        if len(content) > self.max_chars:
            logger.warning("Code content truncated from %d to %d chars", len(content), self.max_chars)
            return content[:self.max_chars] + "\n\n[... content truncated for safety ...]"
        return content

    def explain(self, content: str, question: str = "") -> str:
        """Explain code logic and flow."""
        content = self._prepare(content)
        raw = self.ai._call_ollama(
            model=config.ollama_reason_model,
            system_prompt=SYSTEM_CODE_EXPLAINER,
            user_prompt=build_explain_prompt(content, question)
        )
        return raw or "AI failed to generate an explanation. Please try again."

    def refactor(self, content: str, instruction: str) -> Dict[str, str]:
        """Suggest refactored code with reasoning."""
        content = self._prepare(content)
        raw = self.ai._call_ollama(
            model=config.ollama_reason_model,
            system_prompt=SYSTEM_CODE_REFACTORER,
            user_prompt=build_refactor_prompt(content, instruction)
        )
        parsed = extract_json(raw) if raw else None
        return parsed or {
            "reasoning": "Could not parse AI response into structured JSON.",
            "refactored_code": content
        }

    def debug(self, content: str, error: str) -> Dict[str, Any]:
        """Diagnose bugs based on code and error logs."""
        content = self._prepare(content)
        raw = self.ai._call_ollama(
            model=config.ollama_reason_model,
            system_prompt=SYSTEM_CODE_DEBUGGER,
            user_prompt=build_debug_prompt(content, error)
        )
        parsed = extract_json(raw) if raw else None
        return parsed or {
            "diagnosis": "Failed to analyze error.",
            "fix": "No fix suggested.",
            "confidence": 0.0
        }

    def review(self, content: str) -> Dict[str, Any]:
        """Perform a qualitative code review."""
        content = self._prepare(content)
        raw = self.ai._call_ollama(
            model=config.ollama_reason_model,
            system_prompt=SYSTEM_CODE_REVIEWER,
            user_prompt=build_review_prompt(content)
        )
        parsed = extract_json(raw) if raw else None
        return parsed or {
            "issues": ["Could not complete review."],
            "suggestions": [],
            "quality_score": 0
        }
