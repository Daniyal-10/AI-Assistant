"""
nexus/ai/providers/ollama_provider.py
──────────────────────────────────────
Ollama local LLM provider backend.
Extracted from nexus/ai/orchestrator.py — logic is identical.
"""
from typing import Optional

import requests

from nexus.ai.providers.base import BaseProvider
from nexus.core.exceptions import OllamaConnectionError
from nexus.utils.config import config
from nexus.utils.logger import get_logger

logger = get_logger(__name__)


class OllamaProvider(BaseProvider):
    """Calls the local Ollama API."""

    def __init__(self) -> None:
        self._base_url = config.ollama_base_url
        self._code_model = config.ollama_code_model
        self._reason_model = config.ollama_reason_model
        self._timeout = config.ollama_timeout

    @property
    def name(self) -> str:
        return "ollama"

    def is_available(self) -> bool:
        """Ping Ollama to check if it is reachable."""
        try:
            response = requests.get(
                f"{self._base_url}/api/tags",
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False

    def _resolve_model(self, model_hint: str) -> str:
        """Map model_hint ('code' | 'reason') to actual model name."""
        if model_hint == "code":
            return self._code_model
        return self._reason_model

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        model_hint: str = "reason",
    ) -> Optional[str]:
        """
        Call Ollama /api/chat endpoint.
        Returns response text or None on failure.
        Raises OllamaConnectionError if the server is unreachable.
        """
        model = self._resolve_model(model_hint)
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