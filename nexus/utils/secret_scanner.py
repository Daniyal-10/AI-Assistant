"""
nexus/utils/secret_scanner.py
─────────────────────────────
Regex-based secret detection and redaction for local LLM safety.
"""
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

from nexus.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SecretMatch:
    """Represents a potential secret found in code content."""
    pattern_name: str   # e.g. "AWS Access Key", "Generic API Key"
    line_number: int    # line where the match was found
    redacted_value: str # first 4 chars + "****" — never the full secret


# Pre-compiled regex patterns for secret detection
# Patterns are compiled at module load time for performance
PATTERNS = {
    "AWS Access Key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "AWS Secret Key": re.compile(r"(?i)aws.*[0-9a-zA-Z/+]{40}"),
    "Generic API Key": re.compile(r"(?i)api_?key\s*[=:]\s*['\"]?([\w\-]{8,})"),
    "Generic Secret": re.compile(r"(?i)secret\s*[=:]\s*['\"]?([\w\-]{16,})"),
    "Generic Password": re.compile(r"(?i)password\s*[=:]\s*['\"]?(\S{8,})"),
    "GitHub Token": re.compile(r"ghp_[a-zA-Z0-9]{20,}"),
    "Generic Bearer Token": re.compile(r"(?i)bearer\s+([a-zA-Z0-9\-._~+/]{20,})"),
    "Private Key Header": re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "Database URL with creds": re.compile(r"(postgresql|mysql|mongodb)://[^:]+:([^@]+)@"),
}


def scan_for_secrets(content: str, source_hint: str = "") -> List[SecretMatch]:
    """
    Scans content line-by-line for known secret patterns.
    Returns a list of SecretMatch objects with redacted values.
    """
    matches = []
    try:
        lines = content.splitlines()
    except Exception:
        # Handle cases where splitlines might fail (e.g. very weird encodings)
        return []

    for i, line in enumerate(lines, 1):
        for name, pattern in PATTERNS.items():
            match = pattern.search(line)
            if match:
                # Capture the group if defined (usually the value), otherwise whole match
                # For patterns like Password, group 1 is the value.
                # For patterns like AWS Access Key, the whole match is the value.
                try:
                    full_val = match.group(1) if match.groups() else match.group(0)
                except IndexError:
                    full_val = match.group(0)

                # Redact: first 4 chars + ****
                redacted = full_val[:4] + "****" if len(full_val) >= 4 else "****"
                
                matches.append(SecretMatch(
                    pattern_name=name,
                    line_number=i,
                    redacted_value=redacted
                ))
    
    return matches


def scan_content_safe(content: str, source_hint: str = "") -> Tuple[bool, List[SecretMatch]]:
    """
    Safe wrapper for secret scanning.
    Returns (is_safe, list_of_matches).
    If any secrets are found, returns (False, matches).
    Treats binary content or errors as unsafe (False).
    """
    try:
        # Handle potential binary content gracefully
        if isinstance(content, bytes):
            try:
                content = content.decode("utf-8")
            except UnicodeDecodeError:
                logger.warning("Binary content detected in %s, treating as unsafe.", source_hint)
                return False, []

        # Null byte check for binary detection
        if '\x00' in content:
            logger.warning("Binary content detected in %s (null byte), treating as unsafe.", source_hint)
            return False, []

        matches = scan_for_secrets(content, source_hint)
        
        if matches:
            # Log only pattern names to avoid accidental leakage
            pattern_names = sorted(list(set(m.pattern_name for m in matches)))
            logger.warning(
                "Secret patterns detected in %s: %s", 
                source_hint or "unknown source", 
                ", ".join(pattern_names)
            )
            return False, matches
            
        return True, []

    except Exception as e:
        # Fallback to unsafe if scanning itself fails
        logger.error("Error during secret scanning for %s: %s", source_hint, str(e))
        return False, []
