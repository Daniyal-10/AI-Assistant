"""
Workspace manager — isolated temp directory per task.

Each task gets its own directory under WORKSPACE_BASE.
Files are written atomically. Cleanup is explicit (not automatic).
"""
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from nexus.core.exceptions import WorkspaceSecurityError
from nexus.utils.config import config
from nexus.utils.logger import get_logger

logger = get_logger(__name__)


class Workspace:
    """
    Manages the filesystem space for one task execution.
    """

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.base = Path(config.workspace_base) / f"task_{task_id}"

    def create(self) -> None:
        """Create isolated workspace directory."""
        self.base.mkdir(parents=True, exist_ok=True)
        logger.debug("Workspace created: %s", self.base)

    def write_files(self, files: Dict[str, str]) -> None:
        """
        Write dict of {relative_path: content} to workspace.
        Uses atomic writes for safety.
        """
        for rel_path, content in files.items():
            target = (self.base / rel_path).resolve()

            # 🔐 Security: prevent path traversal
            if not str(target).startswith(str(self.base.resolve()) + os.sep):
                logger.error("Path traversal blocked: %s", rel_path)
                raise PermissionError(f"Invalid file path: {rel_path}")

            target.parent.mkdir(parents=True, exist_ok=True)

            # ✅ Atomic write
            temp_file = target.with_suffix(".tmp")
            temp_file.write_text(content, encoding="utf-8")
            temp_file.replace(target)

            logger.debug("Written: %s (%d bytes)", rel_path, len(content))

    def update_files(self, files: Dict[str, str]) -> None:
        """Overwrite specific files (for fix iterations)."""
        self.write_files(files)
        logger.info("Updated %d file(s)", len(files))

    def read_file(self, rel_path: str) -> Optional[str]:
        """Read a file from workspace."""
        target = self.base / rel_path
        if not target.exists():
            return None

        try:
            return target.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to read file %s: %s", rel_path, e)
            return None

    def get_path(self) -> str:
        return str(self.base)

    def list_files(self) -> Dict[str, str]:
        """Return all readable text files."""
        result = {}

        for path in self.base.rglob("*"):
            if path.is_file():
                rel = str(path.relative_to(self.base))
                try:
                    result[rel] = path.read_text(encoding="utf-8")
                except Exception:
                    continue  # skip binary/unreadable

        return result

    def archive(self) -> str:
        """Create zip of workspace."""
        zip_path = str(self.base) + ".zip"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in self.base.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(self.base))

        logger.info("Archived workspace: %s", zip_path)
        return zip_path

    def cleanup(self) -> None:
        """Safely remove workspace directory."""
        try:
            if self.base.exists() and str(self.base).startswith(str(Path(config.workspace_base)) + os.sep):
                shutil.rmtree(self.base)
                logger.debug("Workspace cleaned: %s", self.base)
            else:
                logger.warning("Cleanup skipped (unsafe path): %s", self.base)
        except Exception as e:
            logger.error("Failed to cleanup workspace: %s", e)


@dataclass
class FileEntry:
    path: str
    size_bytes: int
    extension: str
    last_modified: str


@dataclass
class ProjectSnapshot:
    root: str
    structure: List[str]  # Simplified flat tree representation
    files: List[FileEntry]
    languages: List[str]
    total_files: int


class ProjectScanner:
    """
    Read-only filesystem analyzer for local projects.
    Optimized for performance and security.
    """
    EXCLUDE_DIRS = {".git", "__pycache__", "node_modules", ".venv", ".nexus"}
    EXCLUDE_EXTS = {".key", ".pem", ".exe", ".bin", ".pyc"}
    EXCLUDE_FILES = {".env"}

    LANG_MAP = {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".html": "HTML",
        ".css": "CSS",
        ".json": "JSON",
        ".md": "Markdown",
        ".go": "Go",
        ".rs": "Rust",
        ".c": "C",
        ".cpp": "C++",
    }

    def __init__(self, root_path: str) -> None:
        self.root = os.path.realpath(root_path)
        if not os.path.isdir(self.root):
            raise NotADirectoryError(f"Invalid project root: {root_path}")

    def _is_safe(self, path: str) -> bool:
        """Verify path is inside root and doesn't escape via symlinks."""
        real_path = os.path.realpath(path)
        # Suffix with os.sep to prevent prefix collision
        return real_path.startswith(self.root + os.sep) or real_path == self.root

    def scan(self) -> ProjectSnapshot:
        """
        Scan the project directory. 
        Limits output to prevent LLM context overflow.
        """
        logger.info("Scanning project: %s", self.root)
        
        all_files: List[FileEntry] = []
        structure: List[str] = []
        languages = set()
        count = 0
        limit = 200

        for root, dirs, files in os.walk(self.root):
            # 🔐 Security: skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.EXCLUDE_DIRS]
            
            rel_root = os.path.relpath(root, self.root)
            if rel_root == ".":
                rel_root = ""

            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if f in self.EXCLUDE_FILES or ext in self.EXCLUDE_EXTS:
                    continue

                full_path = os.path.join(root, f)
                if not self._is_safe(full_path):
                    continue

                count += 1
                rel_path = os.path.join(rel_root, f)
                
                try:
                    stats = os.stat(full_path)
                    entry = FileEntry(
                        path=rel_path,
                        size_bytes=stats.st_size,
                        extension=ext,
                        last_modified=datetime.fromtimestamp(stats.st_mtime).isoformat()
                    )
                    all_files.append(entry)
                    
                    if ext in self.LANG_MAP:
                        languages.add(self.LANG_MAP[ext])
                        
                    if len(structure) < limit:
                        structure.append(rel_path)
                except OSError:
                    continue

        if count > limit:
            structure.append(f"... and {count - limit} more files")

        return ProjectSnapshot(
            root=self.root,
            structure=structure,
            files=all_files,
            languages=sorted(list(languages)),
            total_files=count
        )

    def read_file(self, rel_path: str) -> str:
        """
        Read file content with strict security and size limits.
        Refuses files > 100KB.
        """
        target = os.path.join(self.root, rel_path)
        
        if not self._is_safe(target):
            raise WorkspaceSecurityError(f"Access denied (out of bounds): {rel_path}")

        if not os.path.isfile(target):
            raise FileNotFoundError(f"File not found: {rel_path}")

        # Size check
        size = os.path.getsize(target)
        if size > 100 * 1024:
            raise WorkspaceSecurityError(f"File too large (100KB limit): {rel_path} ({size} bytes)")

        try:
            with open(target, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception as e:
            raise WorkspaceSecurityError(f"Failed to read file: {e}")
