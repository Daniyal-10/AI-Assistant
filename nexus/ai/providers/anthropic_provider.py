"""
nexus/ai/providers/anthropic_provider.py
─────────────────────────────────────────
Anthropic Claude provider backend.
Extracted from nexus/ai/orchestrator.py — logic is identical.
"""
from typing import Optional

import requests

from nexus.ai.providers.base import BaseProvider
from nexus.core.exceptions import CloudProviderError
from nexus.utils.config import config
from nexus.utils.logger import get_logger

logger = get_logger(__name__)


class AnthropicProvider(BaseProvider):
    """Calls the Anthropic Claude API as a cloud fallback."""

    def __init__(self) -> None:
        self._api_key = config.anthropic_api_key
        self._model = config.fallback_model
        self._timeout = 60

    @property
    def name(self) -> str:
        return "anthropic"

    def is_available(self) -> bool:
        """Return True if fallback is enabled and API key is present."""
        return bool(config.fallback_enabled and config.anthropic_api_key)

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        model_hint: str = "",
    ) -> Optional[str]:
        """
        Call Anthropic /v1/messages endpoint.
        Returns response text or None on failure.
        Raises CloudProviderError on API errors.
        """
        if not self.is_available():
            logger.error(
                "Anthropic provider called but fallback is disabled or API key missing"
            )
            return None

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": config.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self._model,
            "max_tokens": 4096,
            "temperature": 0.1,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()

            usage = data.get("usage")
            if usage:
                logger.debug("Anthropic token usage: %s", usage)

            content = data.get("content")
            if content and len(content) > 0:
                return content[0].get("text")

            return None

        except requests.exceptions.RequestException as e:
            raise CloudProviderError(f"Anthropic API request failed: {e}")