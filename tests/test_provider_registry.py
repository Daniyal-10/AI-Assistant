import unittest
from unittest.mock import MagicMock
from nexus.ai.providers.base import BaseProvider
from nexus.ai.providers.registry import ProviderRegistry


class TestProviderRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = ProviderRegistry()

        # Mock Provider 1 (Primary, e.g. Ollama)
        self.provider1 = MagicMock(spec=BaseProvider)
        self.provider1.name = "ollama"
        self.provider1.is_available.return_value = True

        # Mock Provider 2 (Fallback, e.g. Anthropic)
        self.provider2 = MagicMock(spec=BaseProvider)
        self.provider2.name = "anthropic"
        self.provider2.is_available.return_value = True

    def test_registry_resolves_correct_provider(self):
        """Verify ProviderRegistry resolves the highest-priority available provider."""
        self.registry.register(self.provider2, priority=50)
        self.registry.register(self.provider1, priority=100)

        # Both are available, so provider1 (higher priority) should be resolved
        resolved = self.registry.resolve("reason")
        self.assertEqual(resolved, self.provider1)

        # If provider1 becomes unavailable, provider2 (lower priority but available) should be resolved
        self.provider1.is_available.return_value = False
        resolved = self.registry.resolve("reason")
        self.assertEqual(resolved, self.provider2)

    def test_registry_falls_back_correctly(self):
        """Verify ProviderRegistry resolves the correct fallback when primary fails or is unavailable."""
        self.registry.register(self.provider2, priority=50)
        self.registry.register(self.provider1, priority=100)

        # Fallback for provider1 (ollama) should be provider2 (anthropic)
        fallback = self.registry.resolve_fallback("ollama")
        self.assertEqual(fallback, self.provider2)

        # If fallback provider is unavailable, resolve_fallback should return None
        self.provider2.is_available.return_value = False
        fallback = self.registry.resolve_fallback("ollama")
        self.assertIsNone(fallback)

    def test_registry_handles_all_unavailable_case(self):
        """Verify ProviderRegistry returns None when all providers are unavailable."""
        self.registry.register(self.provider1, priority=100)
        self.registry.register(self.provider2, priority=50)

        self.provider1.is_available.return_value = False
        self.provider2.is_available.return_value = False

        resolved = self.registry.resolve("reason")
        self.assertIsNone(resolved)
