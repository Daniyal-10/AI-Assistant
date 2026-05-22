"""
nexus/ai/providers/base.py
──────────────────────────
Abstract base class for all LLM provider backends.
"""
from abc import ABC, abstractmethod
from typing import Optional


class BaseProvider(ABC):
    """Abstract provider interface for LLM backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name (e.g. 'ollama', 'anthropic')."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the provider backend is reachable."""

    @abstractmethod
    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        model_hint: str = "",
    ) -> Optional[str]:
        """
        Send a prompt to the provider and return the text response.

        Args:
            system_prompt: The system/instruction prompt.
            user_prompt:   The user message.
            model_hint:    'code' or 'reason' — provider resolves to actual model.

        Returns:
            Response text string, or None if the call failed.
        """

    def call_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        model_hint: str = "reason",
        max_retries: int = 2,
        retry_delay: float = 2.0,
    ) -> Optional[str]:
        """
        Send a prompt to the provider and return the text response with retries.

        Args:
            system_prompt: The system/instruction prompt.
            user_prompt:   The user message.
            model_hint:    The capability hint ('code' or 'reason').
            max_retries:   Number of retry attempts.
            retry_delay:   Delay in seconds between retries.

        Returns:
            Response text string, or None if all attempts failed.
        """
        import time
        from nexus.utils.logger import get_logger
        logger = get_logger(__name__)

        max_attempts = max_retries + 1
        for attempt in range(1, max_attempts + 1):
            logger.debug("Provider %s call attempt %d/%d", self.name, attempt, max_attempts)
            try:
                raw = self.call(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model_hint=model_hint,
                )
                if raw is not None:
                    return raw
                logger.warning("Empty response from %s on attempt %d", self.name, attempt)
            except Exception as e:
                logger.warning("Exception from %s on attempt %d: %s", self.name, attempt, e)
                # If it's the last attempt, propagate the exception.
                if attempt >= max_attempts:
                    raise
            if attempt <= max_retries:
                time.sleep(retry_delay)

        return None