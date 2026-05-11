"""
tests/test_skill_registry.py
────────────────────────────
Unit tests for the SkillRegistry — matching, scoring, and scaffold rendering.
"""
import pytest
from nexus.skills.registry import SkillRegistry, Skill


@pytest.fixture
def registry():
    return SkillRegistry()


class TestSkillMatching:

    def test_api_keyword_matches_api_client(self, registry):
        skill = registry.match("fetch the bitcoin price from an api", "api_client")
        assert skill is not None
        assert skill.name == "api_client"

    def test_csv_keyword_matches_data_processor(self, registry):
        skill = registry.match("process a csv file and count rows", "script")
        assert skill is not None
        assert skill.name == "data_processor"

    def test_file_keyword_matches_file_utility(self, registry):
        skill = registry.match("list all files in a directory", "utility")
        assert skill is not None
        assert skill.name == "file_utility"

    def test_script_keyword_matches_python_script(self, registry):
        skill = registry.match("write a script to calculate fibonacci", "script")
        assert skill is not None
        assert skill.name == "python_script"

    def test_no_match_returns_none(self, registry):
        skill = registry.match("deploy kubernetes cluster with helm charts", "devops")
        assert skill is None

    def test_empty_input_returns_none(self, registry):
        skill = registry.match("", "")
        assert skill is None

    def test_higher_score_wins(self, registry):
        # Multiple keywords for data_processor
        skill = registry.match("parse csv data and filter count rows analyze", "")
        assert skill is not None
        assert skill.name == "data_processor"


class TestScaffoldRendering:

    def test_task_description_substituted(self, registry):
        skill = registry.match("calculate fibonacci", "script")
        assert skill is not None
        rendered = registry.get_scaffold(skill, "Calculate fibonacci numbers up to N")
        assert "Calculate fibonacci numbers up to N" in rendered["main.py"]
        assert "{TASK_DESCRIPTION}" not in rendered["main.py"]

    def test_all_scaffold_files_rendered(self, registry):
        skill = registry.match("fetch weather api", "api_client")
        assert skill is not None
        rendered = registry.get_scaffold(skill, "Fetch weather data")
        assert "client.py" in rendered
        assert "test_client.py" in rendered
        for fname, content in rendered.items():
            assert "{TASK_DESCRIPTION}" not in content, f"Unrendered placeholder in {fname}"

    def test_scaffold_test_files_contain_pytest(self, registry):
        for skill in registry._skills:
            rendered = registry.get_scaffold(skill, "test task")
            for fname, content in rendered.items():
                if fname.startswith("test_"):
                    assert "import pytest" in content or "def test_" in content, \
                        f"Test file {fname} in skill '{skill.name}' missing pytest"

    def test_scaffold_no_real_network_calls(self, registry):
        """API skill tests must mock network — never make real calls."""
        skill = registry.match("fetch api data", "api_client")
        assert skill is not None
        rendered = registry.get_scaffold(skill, "fetch data")
        test_content = rendered.get("test_client.py", "")
        assert "patch" in test_content or "MagicMock" in test_content, \
            "API test scaffold must mock network calls"
        assert "urlopen" not in test_content.split("patch")[0] if "patch" in test_content else True


class TestRegistryAPI:

    def test_list_skills_returns_all_names(self, registry):
        names = registry.list_skills()
        assert "python_script"  in names
        assert "data_processor" in names
        assert "api_client"     in names
        assert "file_utility"   in names

    def test_registry_is_deterministic(self, registry):
        """Same input always returns same skill."""
        r1 = registry.match("fetch data from api endpoint", "api_client")
        r2 = registry.match("fetch data from api endpoint", "api_client")
        assert (r1 is None and r2 is None) or (r1.name == r2.name)
