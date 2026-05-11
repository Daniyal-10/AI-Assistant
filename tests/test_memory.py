"""
tests/test_memory.py
────────────────────
Unit tests for the persistent memory layer.
All tests use isolated in-memory or tmp_path databases.
No real ~/.nexus/memory.db is touched.
"""
import pytest
from pathlib import Path
from nexus.memory.manager import MemoryManager
from nexus.memory.models import Preference, ProjectMemory, ExecutionRecord


@pytest.fixture
def mem(tmp_path) -> MemoryManager:
    """Isolated MemoryManager backed by a tmp_path database."""
    return MemoryManager(tmp_path / "test_memory.db")


# ── Preferences ───────────────────────────────────────────────────────────────

class TestPreferences:

    def test_set_and_get_preference(self, mem):
        assert mem.set_preference("code_style", "stdlib_first", "technical") is True
        val = mem.get_preference("code_style")
        assert val == "stdlib_first"

    def test_get_nonexistent_returns_none(self, mem):
        assert mem.get_preference("nonexistent_key") is None

    def test_upsert_overwrites_existing(self, mem):
        mem.set_preference("style", "verbose", "style")
        mem.set_preference("style", "concise", "style")
        assert mem.get_preference("style") == "concise"

    def test_get_preferences_by_category(self, mem):
        mem.set_preference("lang",  "Python",      "technical")
        mem.set_preference("style", "stdlib_first", "technical")
        mem.set_preference("tone",  "concise",      "style")

        tech = mem.get_preferences(category="technical")
        assert len(tech) == 2
        keys = {p.key for p in tech}
        assert "lang" in keys and "style" in keys

    def test_get_all_preferences(self, mem):
        mem.set_preference("a", "1", "general")
        mem.set_preference("b", "2", "technical")
        all_prefs = mem.get_preferences()
        assert len(all_prefs) == 2

    def test_delete_preference(self, mem):
        mem.set_preference("temp", "value")
        assert mem.delete_preference("temp") is True
        assert mem.get_preference("temp") is None

    def test_delete_nonexistent_is_safe(self, mem):
        assert mem.delete_preference("does_not_exist") is True


# ── Project Memory ────────────────────────────────────────────────────────────

class TestProjectMemory:

    def _project(self, path="/workspace/myapp") -> ProjectMemory:
        return ProjectMemory(
            project_path=path,
            summary="A FastAPI application with PostgreSQL backend",
            tech_stack="Python,FastAPI,PostgreSQL",
            key_files="main.py,app/api.py",
            last_task="Add user authentication endpoint",
            task_count=1,
        )

    def test_save_and_get_project(self, mem):
        proj = self._project()
        assert mem.save_project(proj) is True

        loaded = mem.get_project("/workspace/myapp")
        assert loaded is not None
        assert loaded.tech_stack == "Python,FastAPI,PostgreSQL"
        assert loaded.task_count == 1

    def test_get_nonexistent_project_returns_none(self, mem):
        assert mem.get_project("/nonexistent/path") is None

    def test_upsert_increments_task_count(self, mem):
        proj = self._project()
        mem.save_project(proj)
        mem.save_project(proj)  # second save
        loaded = mem.get_project("/workspace/myapp")
        # task_count increments on each upsert
        assert loaded.task_count == 2

    def test_upsert_updates_summary(self, mem):
        mem.save_project(self._project())
        updated = ProjectMemory(
            project_path="/workspace/myapp",
            summary="Updated summary with new features",
            tech_stack="Python,FastAPI,PostgreSQL,Redis",
            key_files="main.py",
            last_task="Add Redis caching",
            task_count=1,
        )
        mem.save_project(updated)
        loaded = mem.get_project("/workspace/myapp")
        assert loaded.summary == "Updated summary with new features"
        assert "Redis" in loaded.tech_stack

    def test_get_all_projects(self, mem):
        mem.save_project(self._project("/workspace/proj_a"))
        mem.save_project(self._project("/workspace/proj_b"))
        all_proj = mem.get_all_projects()
        assert len(all_proj) == 2

    def test_multiple_projects_isolated(self, mem):
        mem.save_project(self._project("/workspace/proj_a"))
        mem.save_project(ProjectMemory(
            project_path="/workspace/proj_b",
            summary="Different project",
            tech_stack="Node.js",
            key_files="index.js",
            last_task="Setup Express",
            task_count=1,
        ))
        a = mem.get_project("/workspace/proj_a")
        b = mem.get_project("/workspace/proj_b")
        assert a.tech_stack == "Python,FastAPI,PostgreSQL"
        assert b.tech_stack == "Node.js"


# ── Execution Records ─────────────────────────────────────────────────────────

class TestExecutionRecords:

    def test_record_and_retrieve(self, mem):
        mem.record_execution(
            session_id="sess-1",
            intent="TASK",
            raw_input="write a calculator",
            status="DONE",
            summary="Calculator script generated",
            fix_attempts=0,
        )
        records = mem.get_recent_executions(limit=5)
        assert len(records) == 1
        assert records[0].status == "DONE"
        assert records[0].intent == "TASK"

    def test_records_ordered_newest_first(self, mem):
        for i in range(3):
            mem.record_execution("s", "TASK", f"task {i}", "DONE", f"summary {i}")
        records = mem.get_recent_executions(limit=3)
        # Newest first — last inserted should be first
        assert records[0].raw_input == "task 2"

    def test_filter_by_intent(self, mem):
        mem.record_execution("s", "TASK", "task 1", "DONE")
        mem.record_execution("s", "CHAT", "hello",  "DONE")
        mem.record_execution("s", "TASK", "task 2", "FAILED")

        task_records = mem.get_recent_executions(intent="TASK")
        assert len(task_records) == 2
        assert all(r.intent == "TASK" for r in task_records)

    def test_summary_truncated_at_500(self, mem):
        long_summary = "x" * 1000
        mem.record_execution("s", "TASK", "input", "DONE", summary=long_summary)
        records = mem.get_recent_executions(limit=1)
        assert len(records[0].summary) == 500

    def test_success_rate_calculation(self, mem):
        mem.record_execution("s", "TASK", "t1", "DONE")
        mem.record_execution("s", "TASK", "t2", "DONE")
        mem.record_execution("s", "TASK", "t3", "FAILED")

        stats = mem.get_success_rate()
        assert stats["total"] == 3
        assert stats["done"] == 2
        assert stats["failed"] == 1
        assert abs(stats["rate"] - 0.667) < 0.01

    def test_success_rate_empty_returns_zero(self, mem):
        stats = mem.get_success_rate()
        assert stats["total"] == 0
        assert stats["rate"] == 0.0


# ── Context injection ─────────────────────────────────────────────────────────

class TestContextInjection:

    def test_build_memory_context_empty(self, mem):
        result = mem.build_memory_context()
        assert result == ""

    def test_build_memory_context_with_preferences(self, mem):
        mem.set_preference("lang", "Python", "technical")
        mem.set_preference("style", "stdlib_first", "technical")
        result = mem.build_memory_context()
        assert "lang" in result
        assert "Python" in result

    def test_build_memory_context_with_project(self, mem):
        mem.save_project(ProjectMemory(
            project_path="/workspace/myapp",
            summary="FastAPI backend",
            tech_stack="Python,FastAPI",
            key_files="main.py",
            last_task="Add auth",
            task_count=5,
        ))
        result = mem.build_memory_context(project_path="/workspace/myapp")
        assert "FastAPI" in result
        assert "Add auth" in result

    def test_build_memory_context_capped_at_800(self, mem):
        for i in range(20):
            mem.set_preference(f"key_{i}", "x" * 100, "technical")
        result = mem.build_memory_context()
        assert len(result) <= 800

    def test_build_memory_context_with_stats(self, mem):
        mem.record_execution("s", "TASK", "t1", "DONE")
        mem.record_execution("s", "TASK", "t2", "FAILED")
        result = mem.build_memory_context()
        assert "EXECUTION STATS" in result


# ── Thread safety ─────────────────────────────────────────────────────────────

class TestThreadSafety:

    def test_concurrent_writes_do_not_corrupt(self, mem):
        import threading
        errors = []

        def write(i):
            try:
                mem.set_preference(f"key_{i}", f"val_{i}", "general")
                mem.record_execution("s", "TASK", f"task {i}", "DONE")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        prefs = mem.get_preferences()
        assert len(prefs) == 20
