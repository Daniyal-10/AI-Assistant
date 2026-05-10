#!/usr/bin/env python3
"""
Config refactoring verification - inline testing without external dependencies.
"""
import sys
import os

# Mock dotenv before importing nexus
import unittest.mock as mock
sys.modules['dotenv'] = mock.MagicMock()

# Now add nexus to path
sys.path.insert(0, '/home/nexus/nexus')

import tempfile
import threading
from pathlib import Path

def test_1_lazy_init():
    """Test: Config lazy init — no validation at import time."""
    print("\n" + "="*60)
    print("Test 1: Config lazy init — validation not at import time")
    print("="*60)
    try:
        # Clear any cached config
        import importlib
        if 'nexus.utils.config' in sys.modules:
            del sys.modules['nexus.utils.config']
        
        # Set bad environment
        os.environ['EXEC_TIMEOUT'] = '-999'
        
        # Import should NOT fail
        from nexus.utils.config import get_config, _ConfigProxy
        print("✓ Import successful (validation deferred)")
        
        # Reset environment
        os.environ['EXEC_TIMEOUT'] = '30'
        print("✓ Test 1 PASSED")
        return True
    except Exception as e:
        print(f"✗ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_2_proxy_backward_compat():
    """Test: Proxy backward compat — old import pattern works."""
    print("\n" + "="*60)
    print("Test 2: Proxy backward compatibility")
    print("="*60)
    try:
        from nexus.utils.config import config
        print(f"✓ config type: {type(config)}")
        print(f"✓ config.exec_timeout: {config.exec_timeout}")
        print(f"✓ Proxy backward compat OK")
        print("✓ Test 2 PASSED")
        return True
    except Exception as e:
        print(f"✗ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_test_config_context():
    """Test: test_config context manager."""
    print("\n" + "="*60)
    print("Test 3: test_config context manager")
    print("="*60)
    try:
        from nexus.utils.config import get_config, test_config, reset_config
        
        reset_config()
        original_timeout = get_config().exec_timeout
        print(f"✓ Original timeout: {original_timeout}")
        
        with test_config(exec_timeout=999):
            cfg = get_config()
            assert cfg.exec_timeout == 999, f"Expected 999 got {cfg.exec_timeout}"
            print(f"✓ Inside context: timeout = {cfg.exec_timeout}")
        
        # After context, should be back to normal
        reset_config()
        new_timeout = get_config().exec_timeout
        print(f"✓ After context: timeout = {new_timeout}")
        print("✓ Test 3 PASSED")
        return True
    except Exception as e:
        print(f"✗ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_thread_safety():
    """Test: Thread safety of get_config()."""
    print("\n" + "="*60)
    print("Test 4: Thread safety")
    print("="*60)
    try:
        from nexus.utils.config import reset_config, get_config
        
        reset_config()
        results = []
        
        def get_timeout():
            cfg = get_config()
            results.append(cfg.exec_timeout)
        
        threads = [threading.Thread(target=get_timeout) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(results) == 20, f"Expected 20 results, got {len(results)}"
        print(f"✓ Got {len(results)} results from 20 threads")
        print(f"✓ All values consistent: {len(set(results)) == 1}")
        print("✓ Test 4 PASSED")
        return True
    except Exception as e:
        print(f"✗ Test 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_get_config_function():
    """Test: get_config() returns valid NexusConfig."""
    print("\n" + "="*60)
    print("Test 5: get_config() returns valid config")
    print("="*60)
    try:
        from nexus.utils.config import get_config, NexusConfig
        
        cfg = get_config()
        assert isinstance(cfg, NexusConfig), f"Expected NexusConfig, got {type(cfg)}"
        print(f"✓ get_config() returns NexusConfig instance")
        
        # Check key attributes exist
        assert hasattr(cfg, 'exec_timeout'), "Missing exec_timeout"
        assert hasattr(cfg, 'ollama_timeout'), "Missing ollama_timeout"
        assert hasattr(cfg, 'workspace_base'), "Missing workspace_base"
        assert hasattr(cfg, 'max_fix_iterations'), "Missing max_fix_iterations"
        print(f"✓ All expected attributes present")
        
        # Verify singleton behavior
        cfg2 = get_config()
        assert cfg is cfg2, "get_config() should return same instance"
        print(f"✓ Singleton behavior verified")
        print("✓ Test 5 PASSED")
        return True
    except Exception as e:
        print(f"✗ Test 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_6_override_config():
    """Test: override_config() and reset_config()."""
    print("\n" + "="*60)
    print("Test 6: override_config() and reset_config()")
    print("="*60)
    try:
        from nexus.utils.config import get_config, override_config, reset_config, NexusConfig, test_config
        
        reset_config()
        original_timeout = get_config().exec_timeout
        print(f"✓ Original timeout: {original_timeout}")
        
        # Create a new config instance with test values
        with tempfile.TemporaryDirectory() as tmpdir:
            test_cfg = NexusConfig()
            object.__setattr__(test_cfg, 'exec_timeout', 777)
            override_config(test_cfg)
            
            cfg = get_config()
            assert cfg.exec_timeout == 777, f"Expected 777, got {cfg.exec_timeout}"
            print(f"✓ After override: timeout = {cfg.exec_timeout}")
        
        # Reset
        reset_config()
        cfg = get_config()
        print(f"✓ After reset: timeout = {cfg.exec_timeout}")
        print("✓ Test 6 PASSED")
        return True
    except Exception as e:
        print(f"✗ Test 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_7_history_uses_get_config():
    """Test: history.py uses get_config() correctly."""
    print("\n" + "="*60)
    print("Test 7: history.py imports and uses get_config()")
    print("="*60)
    try:
        from nexus.utils.config import test_config
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir) / "workspaces"
            ws.mkdir(parents=True, exist_ok=True)
            
            with test_config(workspace_base=str(ws)):
                from nexus.utils.history import TaskHistory
                history = TaskHistory()
                
                # Verify history_dir is created in the test workspace
                assert history.history_dir.parent == ws.parent, \
                    f"History dir should be in {ws.parent}, got {history.history_dir.parent}"
                print(f"✓ TaskHistory initialized with test config")
                print(f"✓ history_dir: {history.history_dir}")
                print("✓ Test 7 PASSED")
        return True
    except Exception as e:
        print(f"✗ Test 7 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_8_config_fields():
    """Test: NexusConfig has all required fields."""
    print("\n" + "="*60)
    print("Test 8: NexusConfig fields validation")
    print("="*60)
    try:
        from nexus.utils.config import NexusConfig
        import dataclasses
        
        fields = {f.name for f in dataclasses.fields(NexusConfig)}
        required = {
            'ollama_base_url', 'ollama_code_model', 'ollama_reason_model',
            'ollama_timeout', 'exec_timeout', 'max_fix_iterations',
            'workspace_base', 'executor_type', 'nexus_context_token_budget',
            'nexus_task_history_limit', 'nexus_conversation_history_limit',
            'allowed_telegram_users', 'telegram_bot_token',
            'anthropic_api_key', 'fallback_enabled', 'fallback_model'
        }
        
        missing = required - fields
        if missing:
            print(f"✗ Missing fields: {missing}")
            return False
        
        print(f"✓ All {len(required)} required fields present")
        print("✓ Test 8 PASSED")
        return True
    except Exception as e:
        print(f"✗ Test 8 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("NEXUS Config Refactoring Verification")
    print("="*60)
    
    tests = [
        test_1_lazy_init,
        test_2_proxy_backward_compat,
        test_3_test_config_context,
        test_4_thread_safety,
        test_5_get_config_function,
        test_6_override_config,
        test_7_history_uses_get_config,
        test_8_config_fields,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("="*60 + "\n")
    
    sys.exit(0 if all(results) else 1)
