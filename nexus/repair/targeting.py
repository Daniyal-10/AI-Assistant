"""
nexus/repair/targeting.py
─────────────────────────
Targeted file selection for the AI fix loop.
"""
import re
from typing import Dict

from nexus.utils.logger import get_logger

logger = get_logger(__name__)


def _parse_traceback_filename(stderr: str) -> str:
    match = re.search(r'File ["\']([^"\']+)["\']', stderr)
    if match:
        full_path = match.group(1)
        return full_path.replace("\\", "/").split("/")[-1]
    return ""


def select_files_for_fix(
    current_files: Dict[str, str],
    stderr: str,
    stdout: str,
    error_category: str,
) -> Dict[str, str]:
    """
    Select a relevant subset of files to send to the LLM for a fix attempt.
    Falls back to all files if targeting fails — repair must never crash.
    """
    try:
        selected: Dict[str, str] = {}

        tb_filename = _parse_traceback_filename(stderr or "")
        if tb_filename and tb_filename in current_files:
            selected[tb_filename] = current_files[tb_filename]
            logger.debug("Targeting traceback file: %s", tb_filename)

        if error_category in ("ASSERTION_ERROR", "NETWORK_IN_TEST"):
            for fname in current_files:
                normalized = fname.replace("\\", "/")
                if (
                    normalized.startswith("test_")
                    or "/test_" in normalized
                    or normalized == "conftest.py"
                ):
                    selected[fname] = current_files[fname]

        if error_category == "MODULE_NOT_FOUND":
            if "requirements.txt" in current_files:
                selected["requirements.txt"] = current_files["requirements.txt"]
            if tb_filename and tb_filename in current_files:
                selected[tb_filename] = current_files[tb_filename]

        if not selected:
            return dict(current_files)

        logger.info(
            "Targeted fix at %d/%d files: %s",
            len(selected), len(current_files), list(selected.keys()),
        )
        return selected

    except Exception as e:
        logger.warning("File targeting failed (%s) — falling back to all files", e)
        return dict(current_files)
