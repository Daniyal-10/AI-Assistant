"""
nexus/ai/parser.py
───────────────────
Robust JSON extractor + schema validator for Ollama model responses.

Validation philosophy:
  - REQUIRED fields missing   → reject (fatal, retry)
  - OPTIONAL files empty      → normalize (skip/warn, continue)
  - File content non-string   → attempt coercion, warn, continue
  - Entire files dict empty   → reject (nothing to execute)

This distinction is critical: the old behavior rejected the entire
code generation response if `requirements.txt` was empty, causing
infinite retry loops on tasks with no dependencies.
"""
import json
import re
from typing import Any, Dict, Optional, Set

from nexus.utils.logger import get_logger

logger = get_logger(__name__)

# Files that are optional — empty content is acceptable and will be skipped.
# Required files (entry_point, test files) must have real content.
OPTIONAL_FILES: Set[str] = {"requirements.txt", ".env.example", ".gitignore"}


def extract_json(raw: str) -> Optional[Dict[str, Any]]:
    """
    Extract and parse a JSON object from a (potentially messy) LLM response.

    Strategy order:
      1. Direct parse — clean responses from well-behaved models
      2. Strip markdown fences — ```json ... ``` wrapping
      3. Brace-match scan — find the first complete {...} block in the text

    Returns parsed dict or None if all strategies fail.
    """
    if not raw or not raw.strip():
        logger.warning("AI returned empty response")
        return None

    raw = raw.strip()

    # Pre-process: replace triple-quoted strings with single-quoted equivalents
    # Local 7B models (qwen, llama) often output triple quotes inside JSON
    # which breaks all JSON parsers. We sanitize before any parse attempt.
    import re

    def _replace_triple_quotes(text: str) -> str:
        """Replace triple-quoted Python strings inside JSON with escaped single-line strings."""
        result = []
        i = 0
        while i < len(text):
            if text[i:i+3] == '"""':
                # Find the closing triple quote
                end = text.find('"""', i + 3)
                if end == -1:
                    result.append(text[i:])
                    break
                # Extract content between triple quotes
                content = text[i+3:end]
                # Escape backslashes, then escape double quotes, then collapse newlines
                content = content.replace('\\', '\\\\')
                content = content.replace('"', '\\"')
                content = content.replace('\n', '\\n')
                content = content.replace('\r', '')
                result.append('"' + content + '"')
                i = end + 3
            else:
                result.append(text[i])
                i += 1
        return ''.join(result)

    raw = _replace_triple_quotes(raw)

    # ── Strategy 1: Direct parse ──────────────────────────────────────────────
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # ── Strategy 2: Strip markdown code fences ────────────────────────────────
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", raw, flags=re.MULTILINE)
    stripped = re.sub(r"\n?```\s*$", "", stripped, flags=re.MULTILINE).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # ── Strategy 3: Brace-match scan ─────────────────────────────────────────
    first_brace = raw.find("{")
    if first_brace != -1:
        depth = 0
        in_string = False
        escape_next = False

        for idx in range(first_brace, len(raw)):
            ch = raw[idx]

            if escape_next:
                escape_next = False
                continue

            if ch == "\\" and in_string:
                escape_next = True
                continue

            if ch == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw[first_brace : idx + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError as e:
                        logger.debug("Brace-match candidate failed: %s", e)
                    break

    logger.error(
        "All JSON extraction strategies failed.\nRaw response (first 400 chars):\n%s",
        raw[:400],
    )
    return None


# ── Schema Validators ─────────────────────────────────────────────────────────

def validate_plan(data: Dict[str, Any]) -> bool:
    """Validate the planning response has all required fields."""
    required = {
        "task_type",
        "files_to_generate",
        "entry_point",
        "test_command",
        "install_command",
    }
    missing = required - set(data.keys())
    if missing:
        logger.warning("Plan response missing required fields: %s", missing)
        return False

    if not isinstance(data.get("files_to_generate"), list):
        logger.warning("Plan 'files_to_generate' is not a list")
        return False

    if not data["files_to_generate"]:
        logger.warning("Plan 'files_to_generate' is empty")
        return False

    return True


def normalize_generation(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a raw code generation response before validation.

    Actions taken:
      - Coerce non-string file content to string (with warning)
      - Drop files with empty content that are in OPTIONAL_FILES
      - Strip leading/trailing whitespace from all content

    This does NOT drop required files — those will fail validation
    and trigger a proper retry.

    Returns the normalized data dict (mutates a copy, not the original).
    """
    if "files" not in data or not isinstance(data["files"], dict):
        return data

    normalized: Dict[str, str] = {}

    for fname, content in data["files"].items():
        # Coerce non-string content
        if not isinstance(content, str):
            logger.warning(
                "File '%s' has non-string content (%s) — coercing to str",
                fname,
                type(content).__name__,
            )
            content = str(content) if content is not None else ""

        content = content.strip()

        if not content:
            if fname in OPTIONAL_FILES:
                # Safe to skip — optional file with no dependencies
                logger.debug(
                    "Skipping optional file '%s' — empty content", fname
                )
                continue
            else:
                # Keep it in the dict so validate_generation can reject it
                # with a meaningful error message
                logger.warning(
                    "Required file '%s' has empty content — will fail validation",
                    fname,
                )
                normalized[fname] = content
        else:
            normalized[fname] = content

    return {**data, "files": normalized}


def validate_generation(data: Dict[str, Any]) -> bool:
    """
    Validate the code generation response has usable file content.

    Assumes normalize_generation() has already been called — optional
    empty files have been dropped before this runs.
    """
    if "files" not in data:
        logger.warning("Generation response missing 'files' key")
        return False

    if not isinstance(data["files"], dict):
        logger.warning("Generation 'files' is not a dict")
        return False

    if not data["files"]:
        logger.warning("Generation 'files' dict is empty after normalization")
        return False

    # Every remaining file must have non-empty string content.
    # Optional files were already removed by normalize_generation().
    invalid_files = [
        fname
        for fname, content in data["files"].items()
        if not isinstance(content, str) or not content.strip()
    ]

    if invalid_files:
        logger.warning(
            "Required files have empty/invalid content: %s", invalid_files
        )
        return False

    return True


def validate_fix(data: Dict[str, Any]) -> bool:
    """Validate the fix response has usable file corrections."""
    if "fixed_files" not in data:
        logger.warning("Fix response missing 'fixed_files' key")
        return False

    if not isinstance(data["fixed_files"], dict):
        logger.warning("Fix 'fixed_files' is not a dict")
        return False

    if not data["fixed_files"]:
        logger.warning("Fix response returned empty fixed_files — no changes made")
        # Still valid — AI decided no change needed

    return True
