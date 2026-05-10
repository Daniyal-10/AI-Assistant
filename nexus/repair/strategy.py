"""
nexus/repair/strategy.py
────────────────────────
Repair Strategy Selector — decides HOW to fix before calling the LLM.

For each error category, this module determines:
1. Whether the fix is deterministic (we know exactly what to change)
2. Which files to focus on (implementation vs test)
3. What the LLM should prioritize in its fix attempt
4. Whether a fix attempt is even worth trying (terminal errors)

This runs BEFORE generate_fix() to give the LLM a targeted brief
instead of a generic "fix this" prompt.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from nexus.utils.logger import get_logger

logger = get_logger(__name__)


class FixApproach(Enum):
    DETERMINISTIC = "deterministic"  # We know exactly what to do
    GUIDED        = "guided"         # LLM needs direction but path is clear
    CREATIVE      = "creative"       # LLM must reason through the problem
    TERMINAL      = "terminal"       # Cannot be fixed by AI — abort


@dataclass
class RepairStrategy:
    """
    A repair strategy for a specific error situation.
    Passed to generate_fix() to guide the LLM's fix attempt.
    """
    error_category:  str
    approach:        FixApproach
    focus_on_tests:  bool              # True = fix the test, not the impl
    focus_on_impl:   bool              # True = fix the implementation
    focus_on_deps:   bool              # True = fix requirements.txt
    is_terminal:     bool              # True = do not attempt fix
    max_iterations:  int               # Override max fix iterations for this error
    brief:           str               # Short instruction for the LLM
    skip_semantic:   bool = False      # True = skip Stage 2 for this error type
    notes:           List[str] = field(default_factory=list)


# ── Strategy definitions ──────────────────────────────────────────────────────

_STRATEGIES: Dict[str, RepairStrategy] = {

    "SYNTAX_ERROR": RepairStrategy(
        error_category = "SYNTAX_ERROR",
        approach        = FixApproach.DETERMINISTIC,
        focus_on_tests  = False,
        focus_on_impl   = True,
        focus_on_deps   = False,
        is_terminal     = False,
        max_iterations  = 1,
        brief           = (
            "Fix ONLY the syntax error on the reported line. "
            "Do not refactor or restructure any other part of the file."
        ),
        skip_semantic   = True,
        notes           = ["Syntax errors are deterministic — one iteration is enough"],
    ),

    "MODULE_NOT_FOUND": RepairStrategy(
        error_category = "MODULE_NOT_FOUND",
        approach        = FixApproach.DETERMINISTIC,
        focus_on_tests  = False,
        focus_on_impl   = True,
        focus_on_deps   = True,
        is_terminal     = False,
        max_iterations  = 2,
        brief           = (
            "Rewrite the failing code to use Python stdlib instead of the missing module. "
            "Only add to requirements.txt if stdlib genuinely cannot replace the module."
        ),
        notes           = ["Try stdlib replacement first, deps second"],
    ),

    "IMPORT_ERROR": RepairStrategy(
        error_category = "IMPORT_ERROR",
        approach        = FixApproach.DETERMINISTIC,
        focus_on_tests  = False,
        focus_on_impl   = True,
        focus_on_deps   = False,
        is_terminal     = False,
        max_iterations  = 1,
        brief           = (
            "Fix the import path only. "
            "Do not change module logic — only fix the import statement."
        ),
        skip_semantic   = True,
        notes           = ["Local import path fix — deterministic"],
    ),

    "ASSERTION_ERROR": RepairStrategy(
        error_category = "ASSERTION_ERROR",
        approach        = FixApproach.GUIDED,
        focus_on_tests  = True,
        focus_on_impl   = True,
        focus_on_deps   = False,
        is_terminal     = False,
        max_iterations  = 3,
        brief           = (
            "Read STDOUT carefully to see actual vs expected values. "
            "If the implementation logic is correct, fix the test assertion. "
            "If the implementation is wrong, fix the implementation. "
            "Never mock away the logic being tested."
        ),
        notes           = ["Could be test bug or impl bug — needs both files"],
    ),

    "FILE_NOT_FOUND": RepairStrategy(
        error_category = "FILE_NOT_FOUND",
        approach        = FixApproach.DETERMINISTIC,
        focus_on_tests  = False,
        focus_on_impl   = False,
        focus_on_deps   = False,
        is_terminal     = False,
        max_iterations  = 1,
        brief           = (
            "Generate the missing file with appropriate sample content. "
            "Include it in fixed_files."
        ),
        skip_semantic   = True,
        notes           = ["Missing file — just generate it"],
    ),

    "NETWORK_IN_TEST": RepairStrategy(
        error_category = "NETWORK_IN_TEST",
        approach        = FixApproach.DETERMINISTIC,
        focus_on_tests  = True,
        focus_on_impl   = False,
        focus_on_deps   = False,
        is_terminal     = False,
        max_iterations  = 1,
        brief           = (
            "Fix the TEST file only — mock the network call using unittest.mock.patch. "
            "Do NOT change the implementation logic at all."
        ),
        skip_semantic   = True,
        notes           = ["Always a test bug — implementation is correct"],
    ),

    "TYPE_ERROR": RepairStrategy(
        error_category = "TYPE_ERROR",
        approach        = FixApproach.GUIDED,
        focus_on_tests  = False,
        focus_on_impl   = True,
        focus_on_deps   = False,
        is_terminal     = False,
        max_iterations  = 2,
        brief           = (
            "Fix the type mismatch. Check the function signature and the call site. "
            "Add explicit type conversion if needed."
        ),
        notes           = ["Type errors are usually in the impl, rarely in tests"],
    ),

    "NAME_ERROR": RepairStrategy(
        error_category = "NAME_ERROR",
        approach        = FixApproach.DETERMINISTIC,
        focus_on_tests  = False,
        focus_on_impl   = True,
        focus_on_deps   = False,
        is_terminal     = False,
        max_iterations  = 1,
        brief           = (
            "Define the variable or import the function before its first use. "
            "Check spelling carefully."
        ),
        skip_semantic   = True,
        notes           = ["Missing definition — deterministic fix"],
    ),

    "UNKNOWN": RepairStrategy(
        error_category = "UNKNOWN",
        approach        = FixApproach.CREATIVE,
        focus_on_tests  = True,
        focus_on_impl   = True,
        focus_on_deps   = False,
        is_terminal     = False,
        max_iterations  = 3,
        brief           = (
            "Read STDERR carefully and apply the most targeted fix possible. "
            "Do not change code that is unrelated to the error."
        ),
        notes           = ["Unknown error — give LLM full context"],
    ),
}

# Fallback strategy when category not recognized
_DEFAULT_STRATEGY = _STRATEGIES["UNKNOWN"]


# ── Public API ────────────────────────────────────────────────────────────────

def get_repair_strategy(
    error_category: str,
    iteration: int = 1,
    attempt_history: Optional[List[str]] = None,
) -> RepairStrategy:
    """
    Return the repair strategy for a given error category.

    Args:
        error_category:  Category string from classify_error()
        iteration:       Current fix iteration number
        attempt_history: List of previous fix attempt descriptions

    Returns:
        RepairStrategy with approach, focus, and brief for the LLM
    """
    strategy = _STRATEGIES.get(error_category, _DEFAULT_STRATEGY)

    # If we have already tried this category's max iterations,
    # escalate to CREATIVE to try something different
    if (
        attempt_history
        and len(attempt_history) >= strategy.max_iterations
        and strategy.approach != FixApproach.CREATIVE
    ):
        logger.info(
            "Strategy escalation: %s exceeded max_iterations=%d — "
            "switching to CREATIVE approach",
            error_category,
            strategy.max_iterations,
        )
        # Return a modified copy with creative approach
        return RepairStrategy(
            error_category = strategy.error_category,
            approach        = FixApproach.CREATIVE,
            focus_on_tests  = True,
            focus_on_impl   = True,
            focus_on_deps   = strategy.focus_on_deps,
            is_terminal     = False,
            max_iterations  = strategy.max_iterations,
            brief           = (
                f"Previous {len(attempt_history)} fix attempt(s) failed. "
                "Take a completely different approach. "
                "Re-read the original task description and STDERR carefully."
            ),
            notes           = ["Escalated from deterministic to creative"],
        )

    logger.info(
        "Repair strategy: category=%s approach=%s focus=impl:%s/test:%s/deps:%s",
        error_category,
        strategy.approach.value,
        strategy.focus_on_impl,
        strategy.focus_on_tests,
        strategy.focus_on_deps,
    )
    return strategy


def is_terminal_error(error_category: str, stderr: str = "") -> bool:
    """
    Return True if this error cannot be fixed by the AI fix loop.
    Terminal errors cause the engine to abort immediately.
    """
    strategy = _STRATEGIES.get(error_category, _DEFAULT_STRATEGY)
    if strategy.is_terminal:
        return True

    # Additional terminal patterns regardless of category
    terminal_patterns = [
        "killed",
        "out of memory",
        "segmentation fault",
        "permission denied",
        "disk quota exceeded",
    ]
    stderr_lower = stderr.lower()
    return any(p in stderr_lower for p in terminal_patterns)
