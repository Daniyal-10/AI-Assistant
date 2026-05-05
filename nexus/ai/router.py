"""
nexus/ai/router.py
──────────────────
Intent Router module for NEXUS.
Classifies user input into CHAT, TASK, CODE, or SYSTEM.
"""
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from nexus.ai.orchestrator import AIOrchestrator
from nexus.ai.parser import extract_json
from nexus.ai.prompts import SYSTEM_ROUTER, build_router_prompt
from nexus.utils.config import config
from nexus.utils.logger import get_logger

logger = get_logger(__name__)


class IntentType(Enum):
    CHAT = "CHAT"
    TASK = "TASK"
    CODE = "CODE"
    SYSTEM = "SYSTEM"


@dataclass(frozen=True)
class IntentResult:
    intent: IntentType
    confidence: float
    reasoning: str
    raw_input: str


class IntentRouter:
    """
    Classifies user intent using a hybrid LLM + Rule-based strategy.
    """

    def __init__(self, orchestrator: Optional[AIOrchestrator] = None) -> None:
        self.ai = orchestrator or AIOrchestrator()
        # Simple rule-based patterns for fast fallbacks
        self._rules = [
            (r"^(hi|hello|hey|good morning|yo|greetings)\b", IntentType.CHAT),
            (r"\b(who are you|what can you do|help|status|about)\b", IntentType.CHAT),
            (r"\b(ls|cd|pwd|mkdir|rm|clean|workspace|system|info)\b", IntentType.SYSTEM),
            (r"\b(explain|refactor|debug|review|analyze|what does this code do)\b", IntentType.CODE),
            (r"\b(build|create|write a script|generate|automate|process|make)\b", IntentType.TASK),
        ]

    def _sanitize(self, text: str) -> str:
        """Sanitize input before classification."""
        if not text:
            return ""
        # Strip and limit length to prevent prompt injection/overload
        sanitized = text.strip()[:1000]
        # Remove common control characters
        sanitized = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", sanitized)
        return sanitized

    def _rule_classify(self, text: str) -> Optional[IntentType]:
        """Fast rule-based keyword matching."""
        text_lower = text.lower()
        for pattern, intent in self._rules:
            if re.search(pattern, text_lower):
                logger.debug("Rule-based classification matched: %s", intent.value)
                return intent
        return None

    def route(self, user_input: str) -> IntentResult:
        """Alias for classify() for backward/CLI compatibility."""
        return self.classify(user_input)

    def classify(self, user_input: str) -> IntentResult:
        """
        Classify user input into an intent.
        1. LLM-based classification (Primary)
        2. Rule-based classification (Fallback)
        3. Default to CHAT (Final fallback)
        """
        sanitized_input = self._sanitize(user_input)
        if not sanitized_input:
            return IntentResult(IntentType.CHAT, 1.0, "Empty input", user_input)

        # ── 1. LLM Classification ───────────────────────────────────────────
        try:
            # Reusing existing Ollama connection via orchestrator
            raw_response = self.ai._call_ollama(
                model=config.ollama_reason_model,
                system_prompt=SYSTEM_ROUTER,
                user_prompt=build_router_prompt(sanitized_input),
            )

            if raw_response:
                parsed = extract_json(raw_response)
                if parsed and "intent" in parsed:
                    try:
                        intent_str = parsed["intent"].upper()
                        intent = IntentType(intent_str)
                        confidence = float(parsed.get("confidence", 0.8))
                        reasoning = parsed.get("reasoning", "LLM classification")

                        logger.info(
                            "Intent classified (LLM): %s (conf: %.2f) - %s",
                            intent.value, confidence, reasoning
                        )
                        return IntentResult(intent, confidence, reasoning, user_input)
                    except (ValueError, KeyError) as e:
                        logger.warning("LLM returned invalid intent format: %s", e)
        except Exception as e:
            logger.error("LLM classification failed: %s", e)

        # ── 2. Rule-based Fallback ──────────────────────────────────────────
        rule_intent = self._rule_classify(sanitized_input)
        if rule_intent:
            return IntentResult(
                rule_intent,
                0.6,
                "Rule-based fallback matching",
                user_input
            )

        # ── 3. Final Default ────────────────────────────────────────────────
        logger.info("Classification uncertain, defaulting to CHAT")
        return IntentResult(
            IntentType.CHAT,
            0.1,
            "Default fallback (uncertain input)",
            user_input
        )
