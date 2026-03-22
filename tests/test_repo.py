"""
test_repo.py - Unit tests for repo.py (git repository → Markdown via repomix)

All subprocess calls are mocked — repomix does not need to be installed.
"""

import json
import subprocess
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from any2md.repo import app, _build_repomix_cmd, _check_repomix, _extract_metadata


runner = CliRunner()

# ---------------------------------------------------------------------------
# Sample repomix outputs
# ---------------------------------------------------------------------------

_SAMPLE_MARKDOWN = """\
# Repository: myrepo

## File: README.md
Hello world
"""

_SAMPLE_JSON = json.dumps(
    {
        "fileSummary": {
            "totalFiles": 5,
            "totalTokens": 1234,
        },
        "files": [
            {"path": "README.md", "content": "Hello world"},
        ],
    }
)


def _make_success(stdout: str = "", stderr: str = "") -> MagicMock:
    """Return a mock CompletedProcess with returncode=0."""
    m = MagicMock(spec=subprocess.CompletedProcess)
    m.returncode = 0
    m.stdout = stdout
    m.stderr = stderr
    return m


def _make_failure(stderr: str = "repomix error") -> MagicMock:
    """Return a mock CompletedProcess with returncode=1."""
    m = MagicMock(spec=subprocess.CompletedProcess)
    m.returncode = 1
    m.stdout = ""
    m.stderr = stderr
    return m


# ---------------------------------------------------------------------------
# _build_repomix_cmd
# ---------------------------------------------------------------------------

class TestBuildRepomixCmd:
    def test_basic_markdown_cmd(self, tmp_path):
        cmd = _build_repomix_cmd(tmp_path, "markdown", False, False)
        assert cmd[0] == "repomix"
        assert "--style" in cmd
        assert "markdown" in cmd
        assert "--stdout" in cmd
        assert str(tmp_path) in cmd

    def test_json_style(self, tmp_path):
        cmd = _build_repomix_cmd(tmp_path, "json", False, False)
        assert "json" in cmd

    def test_compress_flag_included(self, tmp_path):
        cmd = _build_repomix_cmd(tmp_path, "markdown", True, False)
        assert "--compress" in cmd

    def test_remove_comments_flag_included(self, tmp_path):
        cmd = _build_repomix_cmd(tmp_path, "markdown", False, True)
        assert "--remove-comments" in cmd

    def test_both_flags(self, tmp_path):
        cmd = _build_repomix_cmd(tmp_path, "markdown", True, True)
        assert "--compress" in cmd
        assert "--remove-comments" in cmd


# ---------------------------------------------------------------------------
# _check_repomix
# ---------------------------------------------------------------------------

class TestCheckRepomix:
    def test_raises_exit_when_missing(self):
        with patch("shutil.which", return_value=None):
            with pytest.raises(typer.Exit):
                _check_repomix()

    def test_passes_when_present(self):
        with patch("shutil.which", return_value="/usr/local/bin/repomix"):
            # Should not raise
            _check_repomix()


# ---------------------------------------------------------------------------
# _extract_metadata
# ---------------------------------------------------------------------------

class TestExtractMetadata:
    def test_parses_file_summary(self, tmp_path):
        with patch("any2md.repo._run_repomix", return_value=_make_success(_SAMPLE_JSON)):
            meta = _extract_metadata(tmp_path, False, False)
        assert meta["total_files"] == 5
        assert meta["total_tokens"] == 1234
        assert meta["repo_name"] == tmp_path.name
        assert meta["converter"] == "repo"

    def test_handles_repomix_failure_gracefully(self, tmp_path):
        with patch("any2md.repo._run_repomix", return_value=_make_failure()):
            meta = _extract_metadata(tmp_path, False, False)
        assert meta["repo_name"] == tmp_path.name
        assert "total_files" not in meta

    def test_handles_invalid_json_gracefully(self, tmp_path):
        with patch("any2md.repo._run_repomix", return_value=_make_success("not-json")):
            meta = _extract_metadata(tmp_path, False, False)
        assert meta["converter"] == "repo"
        assert "total_files" not in meta


# ---------------------------------------------------------------------------
# main command — markdown mode
# ---------------------------------------------------------------------------

class TestMainMarkdownMode:
    def test_writes_markdown_file(self, tmp_path):
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()

        def fake_run(cmd):
            if "--style" in cmd and cmd[cmd.index("--style") + 1] == "markdown":
                return _make_success(_SAMPLE_MARKDOWN)
            # metadata JSON run
            return _make_success(_SAMPLE_JSON)

        with patch("shutil.which", return_value="/usr/bin/repomix"), \
             patch("any2md.repo._run_repomix", side_effect=fake_run):
            result = runner.invoke(app, [str(repo_dir), "-o", str(tmp_path)])

        assert result.exit_code == 0
        out_file = tmp_path / "myrepo.md"
        assert out_file.exists()
        content = out_file.read_text()
        assert "---" in content          # frontmatter present
        assert "repo_name" in content
        assert "Hello world" in content

    def test_frontmatter_includes_token_count(self, tmp_path):
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()

        def fake_run(cmd):
            if cmd[cmd.index("--style") + 1] == "markdown":
                return _make_success(_SAMPLE_MARKDOWN)
            return _make_success(_SAMPLE_JSON)

        with patch("shutil.which", return_value="/usr/bin/repomix"), \
             patch("any2md.repo._run_repomix", side_effect=fake_run):
            result = runner.invoke(app, [str(repo_dir), "-o", str(tmp_path)])

        assert result.exit_code == 0
        content = (tmp_path / "myrepo.md").read_text()
        assert "total_tokens" in content
        assert "total_files" in content

    def test_repomix_failure_exits_nonzero(self, tmp_path):
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()

        with patch("shutil.which", return_value="/usr/bin/repomix"), \
             patch("any2md.repo._run_repomix", return_value=_make_failure("crash")):
            result = runner.invoke(app, [str(repo_dir), "-o", str(tmp_path)])

        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# main command — JSON passthrough mode
# ---------------------------------------------------------------------------

class TestMainJsonMode:
    def test_json_output_passthrough(self, tmp_path):
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()

        with patch("shutil.which", return_value="/usr/bin/repomix"), \
             patch("any2md.repo._run_repomix", return_value=_make_success(_SAMPLE_JSON)):
            result = runner.invoke(app, [str(repo_dir), "--json"])

        assert result.exit_code == 0
        # Stdout should be raw repomix JSON
        parsed = json.loads(result.output)
        assert "fileSummary" in parsed

    def test_json_mode_repomix_failure(self, tmp_path):
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()

        with patch("shutil.which", return_value="/usr/bin/repomix"), \
             patch("any2md.repo._run_repomix", return_value=_make_failure("boom")):
            result = runner.invoke(app, [str(repo_dir), "--json"])

        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestErrorCases:
    def test_missing_repomix_exits(self, tmp_path):
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()

        with patch("shutil.which", return_value=None):
            result = runner.invoke(app, [str(repo_dir)])

        assert result.exit_code != 0

    def test_non_directory_input_exits(self, tmp_path):
        not_a_dir = tmp_path / "file.txt"
        not_a_dir.write_text("hello")

        with patch("shutil.which", return_value="/usr/bin/repomix"):
            result = runner.invoke(app, [str(not_a_dir)])

        assert result.exit_code != 0

    def test_nonexistent_path_exits(self, tmp_path):
        missing = tmp_path / "doesnotexist"

        with patch("shutil.which", return_value="/usr/bin/repomix"):
            result = runner.invoke(app, [str(missing)])

        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Flag passthrough
# ---------------------------------------------------------------------------

class TestFlagPassthrough:
    def test_compress_flag_passed_to_repomix(self, tmp_path):
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        captured_cmds = []

        def fake_run(cmd):
            captured_cmds.append(cmd)
            if "--style" in cmd and cmd[cmd.index("--style") + 1] == "markdown":
                return _make_success(_SAMPLE_MARKDOWN)
            return _make_success(_SAMPLE_JSON)

        with patch("shutil.which", return_value="/usr/bin/repomix"), \
             patch("any2md.repo._run_repomix", side_effect=fake_run):
            result = runner.invoke(app, [str(repo_dir), "--compress", "-o", str(tmp_path)])

        assert result.exit_code == 0
        # At least one of the captured commands should include --compress
        assert any("--compress" in cmd for cmd in captured_cmds)

    def test_remove_comments_flag_passed_to_repomix(self, tmp_path):
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        captured_cmds = []

        def fake_run(cmd):
            captured_cmds.append(cmd)
            if "--style" in cmd and cmd[cmd.index("--style") + 1] == "markdown":
                return _make_success(_SAMPLE_MARKDOWN)
            return _make_success(_SAMPLE_JSON)

        with patch("shutil.which", return_value="/usr/bin/repomix"), \
             patch("any2md.repo._run_repomix", side_effect=fake_run):
            result = runner.invoke(
                app, [str(repo_dir), "--remove-comments", "-o", str(tmp_path)]
            )

        assert result.exit_code == 0
        assert any("--remove-comments" in cmd for cmd in captured_cmds)
