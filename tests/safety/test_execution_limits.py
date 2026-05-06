import pytest
import os
from unittest.mock import patch, MagicMock
from nexus.core.engine import TaskEngine
try:
    from nexus.executor.docker_exec import is_docker_available
except ImportError:
    def is_docker_available(): return False
from nexus.utils.config import config

@pytest.fixture
def engine():
    return TaskEngine()

@pytest.mark.safety
def test_infinite_loop_code_is_killed_by_timeout(engine, monkeypatch):
    # Set a very short timeout for testing
    monkeypatch.setattr(config, "exec_timeout", 2)
    
    with patch("nexus.ai.router.IntentRouter.route") as mock_route:
        mock_route.return_value = MagicMock(intent="TASK", confidence=1.0)
        
        with patch("nexus.ai.orchestrator.AIOrchestrator.generate_plan") as mock_plan:
            mock_plan.return_value = {
                "task_type": "script",
                "description": "loop",
                "files_to_generate": ["loop.py"],
                "entry_point": "loop.py",
                "test_command": "python3 loop.py"
            }
            
            with patch("nexus.ai.orchestrator.AIOrchestrator.generate_code") as mock_gen:
                mock_gen.return_value = {"loop.py": "import time\nwhile True: time.sleep(0.1)"}
                
                # Execution should timeout
                res = engine.run("run an infinite loop")
                
                assert res.result.success is False
                # The summary should indicate a timeout or failure after fix loop
                assert "timed out" in res.result.summary.lower() or "failed" in res.result.summary.lower()

@pytest.mark.safety
def test_fix_loop_hard_limit_is_respected(engine, monkeypatch):
    # Set max retries to 2
    monkeypatch.setattr(config, "max_fix_iterations", 2)
    
    with patch("nexus.ai.router.IntentRouter.route") as mock_route:
        mock_route.return_value = MagicMock(intent="TASK", confidence=1.0)
        
        with patch("nexus.ai.orchestrator.AIOrchestrator.generate_plan") as mock_plan:
            mock_plan.return_value = {
                "task_type": "script",
                "description": "fail",
                "files_to_generate": ["fail.py"],
                "entry_point": "fail.py",
                "test_command": "python3 fail.py"
            }
            
            with patch("nexus.ai.orchestrator.AIOrchestrator.generate_code") as mock_gen:
                # Code that always fails
                mock_gen.return_value = {"fail.py": "assert False"}
                
                with patch("nexus.ai.orchestrator.AIOrchestrator.generate_fix") as mock_fix:
                    mock_fix.return_value = {"fail.py": "assert False"}
                    
                    # Track how many times generate_fix is called
                    res = engine.run("run failing code")
                    
                    assert res.result.success is False
                    # 1 initial + 2 fix attempts = 3 total attempts
                    # generate_fix should be called twice
                    assert mock_fix.call_count <= config.max_fix_iterations

@pytest.mark.safety
@pytest.mark.requires_docker
@pytest.mark.skipif(not is_docker_available(), reason="Docker not available")
def test_memory_limit_respected_in_container(engine, monkeypatch):
    # This test requires Docker and ContainerExecutor
    monkeypatch.setenv("NEXUS_EXECUTOR", "docker")
    
    # Reload engine to pick up new executor
    engine = TaskEngine()
    
    with patch("nexus.ai.router.IntentRouter.route") as mock_route:
        mock_route.return_value = MagicMock(intent="TASK", confidence=1.0)
        
        with patch("nexus.ai.orchestrator.AIOrchestrator.generate_plan") as mock_plan:
            mock_plan.return_value = {
                "task_type": "script",
                "description": "oom",
                "files_to_generate": ["oom.py"],
                "entry_point": "oom.py",
                "test_command": "python3 oom.py"
            }
            
            with patch("nexus.ai.orchestrator.AIOrchestrator.generate_code") as mock_gen:
                # Allocate ~1GB which should exceed 512MB limit
                mock_gen.return_value = {"oom.py": "x = [0] * (1024 * 1024 * 200)"}
                
                res = engine.run("allocate too much memory")
                assert res.result.success is False
                # Should fail due to OOM or non-zero exit code
