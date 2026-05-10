"""
nexus/planning/planner.py
─────────────────────────
Planning Engine — enriches raw LLM plans with structured analysis.

Responsibilities:
1. Classify task complexity before code generation
2. Detect required capabilities (network, file IO, deps)
3. Run feasibility checks (catches bad plans before execution)
4. Enrich plan with metadata the fix loop can use

This runs AFTER the LLM produces a plan dict and BEFORE code generation.
It does NOT replace the LLM — it validates and enriches what the LLM returned.
"""
import re
from typing import Any, Dict, List, Optional

from nexus.planning.schema import EnrichedPlan, TaskCategory, TaskComplexity
from nexus.utils.logger import get_logger

logger = get_logger(__name__)

# Keywords that indicate network access is needed
_NETWORK_KEYWORDS = {
    "api", "http", "https", "url", "fetch", "request", "endpoint",
    "webhook", "download", "upload", "rest", "graphql", "scrape", "crawl",
}

# Keywords that indicate file I/O is needed
_FILE_IO_KEYWORDS = {
    "file", "csv", "json", "xml", "txt", "read", "write", "parse",
    "load", "save", "export", "import", "directory", "folder", "path",
}

# Keywords that suggest external dependencies
_DEP_KEYWORDS = {
    "flask", "fastapi", "django", "requests", "pandas", "numpy",
    "sqlalchemy", "celery", "redis", "boto3", "pillow", "matplotlib",
    "sklearn", "tensorflow", "torch", "selenium", "playwright",
}


class PlanningEngine:
    """
    Enriches a raw LLM plan dict with complexity analysis and feasibility checks.
    Always returns an EnrichedPlan — never raises on bad input.
    """

    def enrich(
        self,
        raw_plan: Dict[str, Any],
        user_input: str = "",
        project_snapshot: Optional[Any] = None,
    ) -> EnrichedPlan:
        """
        Main entry point. Takes a validated LLM plan dict and returns
        an EnrichedPlan with complexity, category, and feasibility analysis.

        Args:
            raw_plan:         Validated plan dict from LLM
            user_input:       Original user request (for keyword analysis)
            project_snapshot: Optional project context

        Returns:
            EnrichedPlan with all fields populated
        """
        try:
            plan = EnrichedPlan(raw_plan=raw_plan)

            combined_text = self._build_analysis_text(raw_plan, user_input)

            plan.has_network  = self._detect_network(combined_text)
            plan.has_file_io  = self._detect_file_io(combined_text)
            plan.needs_deps   = self._detect_deps(combined_text, raw_plan)
            plan.category     = self._classify_category(combined_text, raw_plan)
            plan.complexity   = self._classify_complexity(raw_plan, plan)
            plan.estimated_files = len(raw_plan.get("files_to_generate", []))

            issues = self._check_feasibility(raw_plan, plan, project_snapshot)
            plan.feasibility_notes = issues
            plan.feasibility_ok = len(issues) == 0

            self._log_enrichment(plan)
            return plan

        except Exception as e:
            logger.warning(
                "Plan enrichment failed (%s) — returning bare plan", e
            )
            return EnrichedPlan(raw_plan=raw_plan)

    # ── Analysis helpers ──────────────────────────────────────────────────────

    def _build_analysis_text(self, plan: dict, user_input: str) -> str:
        """Combine all text fields for keyword analysis."""
        parts = [
            user_input,
            plan.get("description", ""),
            plan.get("task_type", ""),
            plan.get("install_command", ""),
            " ".join(plan.get("files_to_generate", [])),
            " ".join(plan.get("steps", [])),
        ]
        return " ".join(str(p) for p in parts).lower()

    def _detect_network(self, text: str) -> bool:
        return any(kw in text for kw in _NETWORK_KEYWORDS)

    def _detect_file_io(self, text: str) -> bool:
        return any(kw in text for kw in _FILE_IO_KEYWORDS)

    def _detect_deps(self, text: str, plan: dict) -> bool:
        install_cmd = plan.get("install_command", "")
        if install_cmd and install_cmd.strip():
            return True
        return any(kw in text for kw in _DEP_KEYWORDS)

    def _classify_category(self, text: str, plan: dict) -> TaskCategory:
        task_type = plan.get("task_type", "").lower()

        if "api" in task_type or "api" in text:
            if any(w in text for w in ["flask", "fastapi", "django", "server", "endpoint"]):
                return TaskCategory.WEB
            return TaskCategory.API
        if "data" in task_type or any(w in text for w in ["csv", "pandas", "dataframe", "dataset"]):
            return TaskCategory.DATA
        if "web" in task_type or any(w in text for w in ["html", "scrape", "crawl"]):
            return TaskCategory.WEB
        if "script" in task_type or "utility" in task_type:
            return TaskCategory.SCRIPT
        return TaskCategory.UNKNOWN

    def _classify_complexity(
        self, plan: dict, enriched: EnrichedPlan
    ) -> TaskComplexity:
        files = plan.get("files_to_generate", [])
        num_files = len(files)

        if num_files >= 5 or (enriched.has_network and enriched.needs_deps):
            return TaskComplexity.COMPLEX
        if num_files >= 3 or enriched.needs_deps or enriched.has_network:
            return TaskComplexity.MODERATE
        return TaskComplexity.SIMPLE

    # ── Feasibility checks ────────────────────────────────────────────────────

    def _check_feasibility(
        self,
        plan: dict,
        enriched: EnrichedPlan,
        project_snapshot: Optional[Any],
    ) -> List[str]:
        """
        Run deterministic feasibility checks. Returns list of issues.
        Empty list = plan is feasible.
        """
        issues = []

        # Check 1: test command references a file that will be generated
        test_cmd = plan.get("test_command", "")
        files_to_generate = plan.get("files_to_generate", [])

        if test_cmd and "pytest" in test_cmd:
            # Extract target from pytest command
            parts = test_cmd.split()
            for part in parts:
                if not part.startswith("-") and part != "pytest":
                    # Check if the target file/dir is in files_to_generate
                    if not any(part in f for f in files_to_generate):
                        issues.append(
                            f"test_command references '{part}' "
                            f"which is not in files_to_generate"
                        )

        # Check 2: entry_point must be in files_to_generate
        entry_point = plan.get("entry_point", "")
        if entry_point and entry_point not in files_to_generate:
            issues.append(
                f"entry_point '{entry_point}' is not in files_to_generate"
            )

        # Check 3: network tasks must have test mocking note
        if enriched.has_network:
            has_test_file = any(
                f.startswith("test_") or "test" in f
                for f in files_to_generate
            )
            if not has_test_file:
                issues.append(
                    "Network task detected but no test file in files_to_generate. "
                    "Tests must mock network calls."
                )

        # Check 4: warn if no test file at all
        has_any_test = any(
            f.startswith("test_") or f.endswith("_test.py")
            for f in files_to_generate
        )
        if not has_any_test:
            issues.append(
                "No test file in files_to_generate. "
                "Every task must include at least one pytest test file."
            )

        # Check 5: check for files that already exist in project
        if project_snapshot:
            existing = set(project_snapshot.structure)
            for f in files_to_generate:
                if f in existing:
                    issues.append(
                        f"'{f}' already exists in project — "
                        f"plan may overwrite existing code"
                    )

        return issues

    def _log_enrichment(self, plan: EnrichedPlan) -> None:
        logger.info(
            "Plan enriched | complexity=%s | category=%s | files=%d | "
            "network=%s | deps=%s | feasible=%s",
            plan.complexity.value,
            plan.category.value,
            plan.estimated_files,
            plan.has_network,
            plan.needs_deps,
            plan.feasibility_ok,
        )
        if plan.feasibility_notes:
            for note in plan.feasibility_notes:
                logger.warning("Feasibility issue: %s", note)
