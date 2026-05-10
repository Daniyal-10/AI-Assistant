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