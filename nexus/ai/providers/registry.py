"""
nexus/ai/providers/registry.py
──────────────────────────────
Registry to manage and resolve LLM providers based on capabilities and availability.
"""
from typing import List, Optional, Tuple
from nexus.ai.providers.base import BaseProvider


class ProviderRegistry:
    """Registry holding registered LLM providers in priority order."""

    def __init__(self) -> None:
        # List of tuples: (provider, priority)
        self._providers: List[Tuple[BaseProvider, int]] = []

    def register(self, provider: BaseProvider, priority: int) -> None:
        """Register a provider with a given priority (higher number = higher priority)."""
        # Ensure we don't register the exact same provider instance multiple times
        self._providers = [p for p in self._providers if p[0] is not provider]
        self._providers.append((provider, priority))
        # Keep them sorted by priority descending
        self._providers.sort(key=lambda x: x[1], reverse=True)

    def resolve(self, capability: str) -> Optional[BaseProvider]:
        """
        Resolve the highest-priority available provider for the given capability.
        Returns None if no available providers can satisfy the capability.
        Raises ValueError if the capability is invalid.
        """
        valid_capabilities = {"code", "reason", "chat"}
        if capability not in valid_capabilities:
            raise ValueError(
                f"Invalid capability: '{capability}'. Must be one of {valid_capabilities}"
            )

        available = self.list_available()
        if not available:
            return None

        return available[0]

    def resolve_fallback(self, primary_name: str) -> Optional[BaseProvider]:
        """
        Return the highest-priority available fallback provider when the primary fails.
        Returns None if no fallback provider is available.
        """
        available = self.list_available()
        for provider in available:
            if provider.name != primary_name:
                return provider
        return None

    def list_available(self) -> List[BaseProvider]:
        """Return a list of all registered providers that are available, in priority order."""
        return [provider for provider, _ in self._providers if provider.is_available()]
