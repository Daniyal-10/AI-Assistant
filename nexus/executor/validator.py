"""
nexus/executor/validator.py
────────────────────────────
Execution result validator.

Exit code classification (pytest):
  0 = all tests passed
  1 = tests ran, some failed         → real failure, enter fix loop
  2 = interrupted                    → treat as failure
  3 = internal pytest error          → treat as failure
  4 = usage error / path not found   → configuration problem, NOT a test failure
  5 = no tests collected             → path mismatch, NOT a test failure

Exit codes 4 and 5 are NOT real test failures. They mean the test runner
could not find its target. The engine must correct the path, not the code.
Treating them as test failures sends the AI into an infinite fix loop
trying to fix code that was never actually wrong.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nexus.executor.safe_exec import ExecResult
from nexus.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ValidationResult:
    """Stage 1 (Structural) + Stage 2 (Semantic) validation output."""
    status: str  # PASS, FAIL, TIMEOUT
    stage1_result: ExecResult
    semantic_verdict: Optional[str] = None  # CORRECT, INCORRECT, UNCERTAIN
    semantic_reason: Optional[str] = None
    semantic_issues: List[str] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        """
        True if Stage 1 passed AND Stage 2 did not explicitly fail.
        UNCERTAIN or skipped Stage 2 (None) still count as success if Stage 1 passed.
        """
        if self.status != "PASS":
            return False
        return self.semantic_verdict != "INCORRECT"


# Tests ran and some failed — real failures the AI fix loop should address
_TEST_FAILURE_CODES = {1, 2, 3}

# pytest could not find or collect tests — configuration/path problem
_PYTEST_CONFIG_ERROR_CODES = {4, 5}


def validate_result(
    result: ExecResult,
    command_type: str = "test",
    task_description: str = "",
    generated_code: str = "",
    ai: Any = None,
) -> ValidationResult:
    """
    Determine if execution result is successful (Stage 1 + Stage 2).
    """
    # ── Stage 1: Structural Validation ───────────────────────────────────────
    status = "FAIL"
    if result.timed_out:
        logger.warning("Stage 1 FAILED: execution timed out")
        status = "TIMEOUT"
    elif result.returncode == 0:
        logger.info("Stage 1 PASSED: exit code 0")
        status = "PASS"
    elif command_type == "test" and result.returncode in _PYTEST_CONFIG_ERROR_CODES:
        logger.warning("Stage 1 FAILED: pytest config error (exit %d)", result.returncode)
    elif command_type == "test" and result.returncode in _TEST_FAILURE_CODES:
        logger.warning("Stage 1 FAILED: test failures detected (exit %d)", result.returncode)
    else:
        logger.warning("Stage 1 FAILED: exit code %d", result.returncode)

    vr = ValidationResult(status=status, stage1_result=result)

    # ── Stage 2: Semantic Validation ─────────────────────────────────────────
    # Only if Stage 1 passed, it's a test command, and we have an AI connection
    if status == "PASS" and command_type == "test" and ai and task_description:
        try:
            # Stage 2 has a hard timeout and must not block engine indefinitely
            semantic_data = ai.validate_correctness(
                task=task_description,
                code=generated_code,
                output=result.stdout + result.stderr
            )

            if semantic_data:
                vr.semantic_verdict = semantic_data.get("verdict", "UNCERTAIN")
                vr.semantic_reason = semantic_data.get("reason", "No reason provided")
                vr.semantic_issues = semantic_data.get("issues", [])

                if vr.semantic_verdict == "CORRECT":
                    logger.info("Stage 2 PASSED: Semantic check confirmed correctness")
                elif vr.semantic_verdict == "INCORRECT":
                    logger.warning("Stage 2 FAILED: Semantic check found logic errors")
                else:
                    logger.info("Stage 2 UNCERTAIN: Proceeding with Stage 1 result")
        except Exception as e:
            logger.warning("Stage 2 skipped due to unexpected error: %s", e)

    return vr


def is_config_error(result: ExecResult) -> bool:
    """
    Return True if the result indicates a configuration/path error
    rather than an actual test failure.

    Used by the engine to decide whether to self-correct the test path
    instead of entering the AI fix loop.
    """
    return (
        not result.timed_out
        and result.returncode in _PYTEST_CONFIG_ERROR_CODES
    )


def build_error_context(vr: ValidationResult, command_type: str = "test") -> str:
    """Build structured error summary for AI fix loop."""
    result = vr.stage1_result
    parts = [f"[{command_type.upper()} FAILURE]"]

    if vr.status == "TIMEOUT":
        parts.append("Process exceeded timeout limit and was killed.")

    if result.returncode in _PYTEST_CONFIG_ERROR_CODES:
        parts.append(
            f"pytest could not find tests (exit {result.returncode}). "
            "This is a path or configuration issue, not a code bug."
        )

    if vr.semantic_verdict == "INCORRECT":
        parts.append(f"SEMANTIC ERROR: {vr.semantic_reason}")
        if vr.semantic_issues:
            parts.append("ISSUES FOUND:\n" + "\n".join(f"  - {i}" for i in vr.semantic_issues))

    if result.stderr.strip():
        parts.append(f"STDERR (last 1500 chars):\n{result.stderr[-1500:]}")
    elif result.stdout.strip():
        parts.append(f"STDOUT (last 1500 chars):\n{result.stdout[-1500:]}")

    parts.append(f"EXIT CODE: {result.returncode}")
    return "\n\n".join(parts)
