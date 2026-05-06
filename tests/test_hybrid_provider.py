import unittest
from unittest.mock import patch, MagicMock
from nexus.ai.orchestrator import AIOrchestrator
from nexus.utils.config import NexusConfig, _validate_config
from nexus.core.exceptions import CloudProviderError, OllamaConnectionError

class TestHybridProvider(unittest.TestCase):
    def setUp(self):
        # We need to patch config before initializing AIOrchestrator to avoid validation errors
        with patch("nexus.utils.config.config") as mock_cfg:
            mock_cfg.ollama_base_url = "http://localhost"
            mock_cfg.ollama_code_model = "test-coder"
            mock_cfg.ollama_reason_model = "test-reason"
            mock_cfg.ollama_timeout = 30
            self.orchestrator = AIOrchestrator()
            
        self.plan = {"task": "test"}
        self.files = {"main.py": "print('hello')"}

    @patch("nexus.ai.orchestrator.AIOrchestrator._call_with_retry")
    @patch("nexus.ai.orchestrator.AIOrchestrator._call_claude")
    def test_iteration_1_never_calls_claude(self, mock_claude, mock_retry):
        # Local fails
        mock_retry.return_value = None
        
        result = self.orchestrator.generate_fix(
            plan=self.plan,
            current_files=self.files,
            stdout="", stderr="error", error="error",
            iteration=1
        )
        
        self.assertIsNone(result)
        mock_claude.assert_not_called()

    @patch("nexus.ai.orchestrator.AIOrchestrator._call_with_retry")
    @patch("nexus.ai.orchestrator.AIOrchestrator._call_claude")
    @patch("nexus.ai.orchestrator.config")
    @patch("nexus.ai.orchestrator.extract_json")
    @patch("nexus.ai.orchestrator.validate_fix")
    def test_iteration_2_calls_claude_on_local_failure(self, mock_val, mock_extract, mock_config, mock_claude, mock_retry):
        mock_config.fallback_enabled = True
        mock_config.anthropic_api_key = "test-key"
        mock_config.fallback_model = "test-claude"
        mock_retry.return_value = None
        mock_claude.return_value = "claude response"
        mock_extract.return_value = {"fixed_files": {"a.py": "fixed"}}
        mock_val.return_value = True
        
        result = self.orchestrator.generate_fix(
            plan=self.plan,
            current_files=self.files,
            stdout="", stderr="error", error="error",
            iteration=2
        )
        
        self.assertEqual(result, {"a.py": "fixed"})
        mock_claude.assert_called_once()

    @patch("nexus.ai.orchestrator.AIOrchestrator._call_with_retry")
    @patch("nexus.ai.orchestrator.AIOrchestrator._call_claude")
    @patch("nexus.ai.orchestrator.config")
    def test_iteration_2_no_claude_if_disabled(self, mock_config, mock_claude, mock_retry):
        mock_config.fallback_enabled = False
        mock_retry.return_value = None
        
        result = self.orchestrator.generate_fix(
            plan=self.plan,
            current_files=self.files,
            stdout="", stderr="error", error="error",
            iteration=2
        )
        
        self.assertIsNone(result)
        mock_claude.assert_not_called()

    @patch("nexus.ai.orchestrator.AIOrchestrator._call_with_retry")
    @patch("nexus.ai.orchestrator.AIOrchestrator._call_claude")
    @patch("nexus.ai.orchestrator.config")
    def test_iteration_3_both_fail_returns_none(self, mock_config, mock_claude, mock_retry):
        mock_config.fallback_enabled = True
        mock_config.anthropic_api_key = "test-key"
        mock_retry.return_value = None
        mock_claude.side_effect = CloudProviderError("failed")
        
        result = self.orchestrator.generate_fix(
            plan=self.plan,
            current_files=self.files,
            stdout="", stderr="error", error="error",
            iteration=3
        )
        
        self.assertIsNone(result)

    def test_config_validation_raises_if_fallback_enabled_without_key(self):
        cfg = MagicMock(spec=NexusConfig)
        cfg.fallback_enabled = True
        cfg.anthropic_api_key = None
        cfg.exec_timeout = 30
        cfg.ollama_timeout = 120
        cfg.max_fix_iterations = 3
        cfg.workspace_base = "/tmp"
        cfg.ollama_base_url = "http://localhost"
        
        with patch("os.path.isabs", return_value=True), \
             patch("os.makedirs"), \
             patch("os.access", return_value=True), \
             patch("sys.exit") as mock_exit:
            _validate_config(cfg)
            mock_exit.assert_called_with(1)

    @patch("nexus.ai.orchestrator.AIOrchestrator._call_with_retry")
    @patch("nexus.ai.orchestrator.AIOrchestrator._call_claude")
    @patch("nexus.ai.orchestrator.config")
    @patch("nexus.ai.orchestrator.extract_json")
    @patch("nexus.ai.orchestrator.validate_fix")
    def test_claude_invalid_response_is_failure(self, mock_val, mock_extract, mock_config, mock_claude, mock_retry):
        mock_config.fallback_enabled = True
        mock_config.anthropic_api_key = "test-key"
        mock_retry.return_value = None
        mock_claude.return_value = "bad response"
        mock_extract.return_value = {"bad": "data"}
        mock_val.return_value = False # Validation fails
        
        result = self.orchestrator.generate_fix(
            plan=self.plan,
            current_files=self.files,
            stdout="", stderr="error", error="error",
            iteration=2
        )
        
        self.assertIsNone(result)

if __name__ == "__main__":
    unittest.main()
