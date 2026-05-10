"""
nexus/context/assembler.py
──────────────────────────
Context Assembler — selects relevant project files for a given task.

Instead of dumping the entire session history into every prompt,
this module selects ONLY the files and context that matter for
the current task. This reduces noise, saves tokens, and improves
LLM output quality.

Strategy (Phase 1 — no embeddings required):
1. Keyword matching between task input and file names/extensions
2. Extension-based relevance (task mentions CSV → include CSV files)
3. Recency bias (recently modified files rank higher)
4. Hard token budget cap — never overflow the context window
"""
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nexus.utils.logger import get_logger

logger = get_logger(__name__)

# Max chars of file content to include per file
_MAX_FILE_CHARS = 1500

# Extension groups for keyword-based relevance
_EXT_KEYWORDS = {
    ".py":   ["python", "script", "function", "class", "module", "test"],
    ".csv":  ["csv", "data", "table", "row", "column", "excel", "spreadsheet"],
    ".json": ["json", "config", "api", "response", "payload"],
    ".txt":  ["text", "file", "read", "write", "log"],
    ".md":   ["readme", "docs", "documentation", "markdown"],
    ".html": ["html", "web", "template", "page"],
    ".sql":  ["sql", "database", "query", "table", "db"],
    ".yaml": ["yaml", "config", "docker", "deploy"],
    ".toml": ["config", "toml", "pyproject", "settings"],
}


@dataclass
class RelevantFile:
    """A file selected as relevant for the current task."""
    path: str
    content: str
    relevance_score: float
    reason: str


@dataclass
class AssembledContext:
    """
    The output of context assembly.
    Contains selected files and a formatted summary for prompt injection.
    """
    relevant_files: List[RelevantFile] = field(default_factory=list)
    total_chars: int = 0
    was_truncated: bool = False

    def to_prompt_block(self) -> str:
        """Format selected files as a prompt-ready block."""
        if not self.relevant_files:
            return ""

        lines = ["--- RELEVANT PROJECT FILES ---"]
        for rf in self.relevant_files:
            lines.append(f"\n# {rf.path} (relevance: {rf.relevance_score:.2f})")
            lines.append(rf.content[:_MAX_FILE_CHARS])
            if len(rf.content) > _MAX_FILE_CHARS:
                lines.append("... [truncated]")

        if self.was_truncated:
            lines.append("\n[Some files omitted due to token budget]")

        lines.append("--- END PROJECT FILES ---")
        return "\n".join(lines)


class ContextAssembler:
    """
    Selects relevant project files for a given task.
    Phase 1: keyword + extension matching.
    Phase 2 (later): embedding-based semantic search.
    """

    def __init__(self, token_budget: int = 3000) -> None:
        # token_budget controls how much context we inject
        # rough estimate: 1 token ≈ 4 chars
        self.char_budget = token_budget * 4

    def assemble(
        self,
        user_input: str,
        project_snapshot: Optional[Any] = None,
        scanner: Optional[Any] = None,
    ) -> AssembledContext:
        """
        Select relevant files from the project snapshot for the given task.

        Args:
            user_input:       The user's task description
            project_snapshot: ProjectSnapshot from ProjectScanner
            scanner:          ProjectScanner instance (needed to read file contents)

        Returns:
            AssembledContext with selected files and formatted prompt block
        """
        if project_snapshot is None or scanner is None:
            logger.debug("No project context available — skipping assembly")
            return AssembledContext()

        try:
            keywords = self._extract_keywords(user_input)
            scored = self._score_files(
                project_snapshot.structure, keywords
            )

            # Sort by score descending, take top candidates
            scored.sort(key=lambda x: x[1], reverse=True)
            top_files = [f for f, score in scored if score > 0.0][:10]

            result = AssembledContext()
            chars_used = 0

            for filepath in top_files:
                if chars_used >= self.char_budget:
                    result.was_truncated = True
                    break

                try:
                    content = scanner.read_file(filepath)
                except Exception as e:
                    logger.debug("Could not read %s: %s", filepath, e)
                    continue

                if not content or not content.strip():
                    continue

                score = next(
                    s for f, s in scored if f == filepath
                )
                reason = self._explain_score(filepath, keywords)

                chars_to_add = min(len(content), _MAX_FILE_CHARS)
                if chars_used + chars_to_add > self.char_budget:
                    result.was_truncated = True
                    break

                result.relevant_files.append(
                    RelevantFile(
                        path=filepath,
                        content=content,
                        relevance_score=score,
                        reason=reason,
                    )
                )
                chars_used += chars_to_add

            result.total_chars = chars_used

            logger.info(
                "Context assembled: %d files, %d chars, truncated=%s",
                len(result.relevant_files),
                result.total_chars,
                result.was_truncated,
            )
            return result

        except Exception as e:
            logger.warning("Context assembly failed (%s) — returning empty", e)
            return AssembledContext()

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _extract_keywords(self, user_input: str) -> List[str]:
        """Extract meaningful keywords from user input."""
        stop_words = {
            "a", "an", "the", "is", "it", "in", "on", "at", "to",
            "for", "of", "and", "or", "but", "with", "that", "this",
            "write", "create", "build", "make", "generate", "code",
            "script", "function", "python", "file",
        }
        words = user_input.lower().split()
        return [w.strip(".,!?") for w in words if w not in stop_words and len(w) > 2]

    def _score_files(
        self,
        file_list: List[str],
        keywords: List[str],
    ) -> List[tuple]:
        """Score each file by relevance to the keywords."""
        scored = []
        for filepath in file_list:
            score = self._score_single(filepath, keywords)
            scored.append((filepath, score))
        return scored

    def _score_single(self, filepath: str, keywords: List[str]) -> float:
        """
        Score a single file path against keywords.
        Returns float 0.0-1.0.
        """
        score = 0.0
        filepath_lower = filepath.lower()
        basename = os.path.basename(filepath_lower)
        ext = os.path.splitext(filepath_lower)[1]

        # Skip test files unless task mentions testing
        is_test = basename.startswith("test_") or basename.endswith("_test.py")
        if is_test:
            test_keywords = {"test", "testing", "pytest", "assert", "fixture"}
            if not any(kw in keywords for kw in test_keywords):
                return 0.0

        # Keyword match in filename
        for kw in keywords:
            if kw in basename:
                score += 0.4
            elif kw in filepath_lower:
                score += 0.2

        # Extension relevance
        ext_kws = _EXT_KEYWORDS.get(ext, [])
        for kw in keywords:
            if kw in ext_kws:
                score += 0.2

        # Boost for common entry points
        if basename in {"main.py", "app.py", "run.py", "cli.py"}:
            score += 0.1

        # Penalize generated/cache files
        if any(p in filepath_lower for p in ["__pycache__", ".pyc", ".egg"]):
            return 0.0

        return min(score, 1.0)

    def _explain_score(self, filepath: str, keywords: List[str]) -> str:
        """Generate a human-readable reason for why this file was selected."""
        basename = os.path.basename(filepath)
        matched = [kw for kw in keywords if kw in basename.lower()]
        if matched:
            return f"filename matches keywords: {matched}"
        ext = os.path.splitext(filepath)[1]
        if ext in _EXT_KEYWORDS:
            return f"extension {ext} relevant to task"
        return "general relevance"
