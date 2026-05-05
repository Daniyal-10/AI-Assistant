"""
tests/test_workspace_scanner.py
──────────────────────────────
Unit tests for Project Scanner security and performance.
"""
import os
import pytest
from nexus.executor.workspace import ProjectScanner, WorkspaceSecurityError

def test_scanner_basic_scan(tmp_path):
    """Verify recursive scanning and language detection."""
    root = tmp_path / "my_project"
    root.mkdir()
    (root / "app").mkdir()
    (root / "app" / "core.py").write_text("import os")
    (root / "index.js").write_text("console.log('hi')")
    (root / ".git").mkdir() # Should be excluded
    (root / "node_modules").mkdir() # Should be excluded
    (root / "config.key").write_text("secret") # Should be excluded

    scanner = ProjectScanner(str(root))
    snapshot = scanner.scan()

    assert snapshot.total_files == 2
    assert "app/core.py" in snapshot.structure
    assert "index.js" in snapshot.structure
    assert "Python" in snapshot.languages
    assert "JavaScript" in snapshot.languages
    
    # Verify exclusions
    for path in snapshot.structure:
        assert ".git" not in path
        assert "node_modules" not in path
        assert ".key" not in path


def test_scanner_path_traversal_protection(tmp_path):
    """Verify that scanner refuses to read files outside root."""
    root = tmp_path / "project"
    root.mkdir()
    
    outside = tmp_path / "secrets.env"
    outside.write_text("DB_PASSWORD=123")
    
    scanner = ProjectScanner(str(root))
    
    # Attempt traversal
    with pytest.raises(WorkspaceSecurityError, match="Access denied"):
        scanner.read_file("../secrets.env")


def test_scanner_symlink_boundary_check(tmp_path):
    """Verify that symlinks pointing outside the project are blocked."""
    root = tmp_path / "safe_zone"
    root.mkdir()
    
    danger_zone = tmp_path / "danger_zone"
    danger_zone.mkdir()
    secret_file = danger_zone / "stolen.txt"
    secret_file.write_text("classified")
    
    # Create symlink from inside to outside
    link_path = root / "shortcut.txt"
    os.symlink(str(secret_file), str(link_path))
    
    scanner = ProjectScanner(str(root))
    
    # The scan should skip it because realpath is outside
    snapshot = scanner.scan()
    assert "shortcut.txt" not in snapshot.structure
    
    # Manual read attempt should also fail
    with pytest.raises(WorkspaceSecurityError, match="Access denied"):
        scanner.read_file("shortcut.txt")


def test_scanner_file_size_limit(tmp_path):
    """Verify that files over 100KB are refused."""
    root = tmp_path / "project"
    root.mkdir()
    
    large_file = root / "huge.log"
    # Write 101KB of data
    large_file.write_text("A" * (101 * 1024))
    
    scanner = ProjectScanner(str(root))
    
    with pytest.raises(WorkspaceSecurityError, match="File too large"):
        scanner.read_file("huge.log")


def test_scanner_performance_large_project(tmp_path):
    """Verify that scanning is fast even with many files."""
    root = tmp_path / "big_project"
    root.mkdir()
    
    # Create 500 dummy files
    for i in range(500):
        (root / f"file_{i}.txt").write_text("content")
        
    scanner = ProjectScanner(str(root))
    import time
    start = time.time()
    snapshot = scanner.scan()
    duration = time.time() - start
    
    assert duration < 1.0 # Should be very fast
    assert snapshot.total_files == 500
    # Structure should be truncated at 200
    assert len(snapshot.structure) == 201 
    assert "and 300 more files" in snapshot.structure[-1]
