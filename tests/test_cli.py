"""Tests for the main CLI entry point (cli.py)."""

import os
import subprocess
import sys
import tempfile
import unittest

from any2md import __version__


class TestSilentFlag(unittest.TestCase):
    """Test --silent as a synonym for --quiet (GNU compliance)."""

    def _run(self, *args, env=None):
        run_env = os.environ.copy()
        run_env.pop("ANY2MD_QUIET", None)
        if env:
            run_env.update(env)
        return subprocess.run(
            [sys.executable, "-m", "any2md.cli", *args],
            capture_output=True,
            text=True,
            env=run_env,
        )

    def test_silent_suppresses_info_logs(self):
        """--silent should suppress INFO-level log output on stderr."""
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("name,value\nfoo,1\nbar,2\n")
            csv_path = f.name
        try:
            with tempfile.TemporaryDirectory() as out_dir:
                result = self._run("csv", csv_path, "--silent", "-o", out_dir)
            # No INFO lines should appear in stderr
            for line in result.stderr.splitlines():
                self.assertNotIn("INFO", line, f"Unexpected INFO log with --silent: {line}")
        finally:
            os.unlink(csv_path)

    def test_silent_produces_output(self):
        """--silent should not suppress the actual markdown output."""
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("name,value\nfoo,1\nbar,2\n")
            csv_path = f.name
        try:
            with tempfile.TemporaryDirectory() as out_dir:
                result = self._run("csv", csv_path, "--silent", "-o", out_dir)
            self.assertEqual(result.returncode, 0, f"Non-zero exit with --silent: {result.stderr}")
        finally:
            os.unlink(csv_path)

    def test_silent_equivalent_to_quiet(self):
        """--silent and --quiet should suppress the same log levels (no INFO on stderr)."""
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("name,value\nfoo,1\nbar,2\n")
            csv_path = f.name
        try:
            with tempfile.TemporaryDirectory() as out_dir1:
                result_silent = self._run("csv", csv_path, "--silent", "-o", out_dir1)
            with tempfile.TemporaryDirectory() as out_dir2:
                result_quiet = self._run("csv", csv_path, "--quiet", "-o", out_dir2)
            # Both should exit cleanly
            self.assertEqual(result_silent.returncode, 0)
            self.assertEqual(result_quiet.returncode, 0)
            # Neither should have any ERROR or WARNING lines (not just INFO)
            for line in result_silent.stderr.splitlines():
                self.assertNotIn("INFO", line)
            for line in result_quiet.stderr.splitlines():
                self.assertNotIn("INFO", line)
        finally:
            os.unlink(csv_path)

    def test_silent_in_help_text(self):
        """--silent should be mentioned in the --help output."""
        result = self._run("--help")
        self.assertIn("--silent", result.stdout)


class TestVersionFlag(unittest.TestCase):
    """Test --version and -V flags."""

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "any2md.cli", *args],
            capture_output=True,
            text=True,
        )

    def test_version_flag_stdout(self):
        result = self._run("--version")
        self.assertIn("any2md", result.stdout)
        self.assertIn(__version__, result.stdout)
        self.assertIn("Copyright", result.stdout)
        self.assertIn("MIT", result.stdout)

    def test_version_flag_exit_code(self):
        result = self._run("--version")
        self.assertEqual(result.returncode, 0)

    def test_short_version_flag_stdout(self):
        result = self._run("-V")
        self.assertIn("any2md", result.stdout)
        self.assertIn(__version__, result.stdout)

    def test_short_version_flag_exit_code(self):
        result = self._run("-V")
        self.assertEqual(result.returncode, 0)

    def test_version_output_format(self):
        result = self._run("--version")
        lines = result.stdout.strip().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("any2md "))
        self.assertTrue(lines[1].startswith("Copyright"))
        self.assertTrue(lines[2].startswith("License MIT"))


class TestDocsSubcommand(unittest.TestCase):
    """Test the docs subcommand that prints README.md to stdout."""

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "any2md.cli", *args],
            capture_output=True,
            text=True,
        )

    def test_docs_exit_code(self):
        """any2md docs should exit 0."""
        result = self._run("docs")
        self.assertEqual(result.returncode, 0)

    def test_docs_first_line_is_heading(self):
        """First line of docs output should be a markdown heading."""
        result = self._run("docs")
        first_line = result.stdout.splitlines()[0]
        self.assertTrue(first_line.startswith("#"), f"Expected heading, got: {first_line!r}")

    def test_docs_contains_any2md(self):
        """docs output should mention any2md."""
        result = self._run("docs")
        self.assertIn("any2md", result.stdout)

    def test_docs_output_to_stdout(self):
        """docs output should go to stdout, not stderr."""
        result = self._run("docs")
        self.assertGreater(len(result.stdout), 100, "Expected substantial stdout output")

    def test_docs_help_flag_prints_docs(self):
        """any2md docs --help should also print docs (not a typer help page)."""
        result = self._run("docs", "--help")
        self.assertEqual(result.returncode, 0)
        self.assertTrue(result.stdout.startswith("#"), "Expected markdown heading")

    def test_docs_listed_in_help(self):
        """'docs' should appear in the main --help output."""
        result = self._run("--help")
        self.assertIn("docs", result.stdout)


class TestCompletionSubcommand(unittest.TestCase):
    """Test the completion subcommand that outputs shell completion scripts."""

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "any2md.cli", *args],
            capture_output=True,
            text=True,
        )

    # --- bash ---

    def test_bash_exit_code(self):
        result = self._run("completion", "bash")
        self.assertEqual(result.returncode, 0)

    def test_bash_output_to_stdout(self):
        result = self._run("completion", "bash")
        self.assertGreater(len(result.stdout), 100)

    def test_bash_contains_function(self):
        result = self._run("completion", "bash")
        self.assertIn("_any2md_completions()", result.stdout)

    def test_bash_registers_completion(self):
        result = self._run("completion", "bash")
        self.assertIn("complete -F _any2md_completions any2md", result.stdout)

    def test_bash_lists_subcommands(self):
        result = self._run("completion", "bash")
        for cmd in ("yt", "pdf", "img", "web", "csv", "completion"):
            self.assertIn(cmd, result.stdout, f"Missing subcommand in bash completion: {cmd}")

    def test_bash_lists_global_flags(self):
        result = self._run("completion", "bash")
        for flag in ("--json", "--quiet", "--version", "--output-dir", "--format"):
            self.assertIn(flag, result.stdout, f"Missing flag in bash completion: {flag}")

    def test_bash_nothing_on_stderr(self):
        result = self._run("completion", "bash")
        # stderr should be empty (no log noise) when completing
        self.assertEqual(result.stderr.strip(), "")

    # --- zsh ---

    def test_zsh_exit_code(self):
        result = self._run("completion", "zsh")
        self.assertEqual(result.returncode, 0)

    def test_zsh_starts_with_compdef(self):
        result = self._run("completion", "zsh")
        self.assertTrue(result.stdout.startswith("#compdef any2md"))

    def test_zsh_contains_function(self):
        result = self._run("completion", "zsh")
        self.assertIn("_any2md()", result.stdout)

    def test_zsh_lists_subcommands(self):
        result = self._run("completion", "zsh")
        for cmd in ("yt", "pdf", "img", "web", "csv", "completion"):
            self.assertIn(cmd, result.stdout, f"Missing subcommand in zsh completion: {cmd}")

    def test_zsh_lists_flags(self):
        result = self._run("completion", "zsh")
        self.assertIn("--json", result.stdout)
        self.assertIn("--output-dir", result.stdout)

    def test_zsh_nothing_on_stderr(self):
        result = self._run("completion", "zsh")
        self.assertEqual(result.stderr.strip(), "")

    # --- fish ---

    def test_fish_exit_code(self):
        result = self._run("completion", "fish")
        self.assertEqual(result.returncode, 0)

    def test_fish_contains_complete_commands(self):
        result = self._run("completion", "fish")
        self.assertIn("complete -c any2md", result.stdout)

    def test_fish_lists_subcommands(self):
        result = self._run("completion", "fish")
        for cmd in ("yt", "pdf", "img", "web", "csv", "completion"):
            self.assertIn(f"-a '{cmd}'", result.stdout, f"Missing subcommand in fish completion: {cmd}")

    def test_fish_lists_flags(self):
        result = self._run("completion", "fish")
        # fish uses long-name format without leading '--': -l json, -l output-dir
        self.assertIn("-l json", result.stdout)
        self.assertIn("-l output-dir", result.stdout)

    def test_fish_nothing_on_stderr(self):
        result = self._run("completion", "fish")
        self.assertEqual(result.stderr.strip(), "")

    # --- error cases ---

    def test_unknown_shell_exits_nonzero(self):
        result = self._run("completion", "powershell")
        self.assertNotEqual(result.returncode, 0)

    def test_unknown_shell_error_message(self):
        result = self._run("completion", "powershell")
        self.assertIn("powershell", result.stderr)
        self.assertIn("bash", result.stderr)

    def test_no_args_prints_usage(self):
        result = self._run("completion")
        self.assertIn("Usage", result.stderr)
        self.assertIn("bash", result.stderr)

    def test_help_flag_prints_usage(self):
        result = self._run("completion", "--help")
        self.assertIn("Usage", result.stderr)

    # --- listed in help ---

    def test_completion_listed_in_help(self):
        result = self._run("--help")
        self.assertIn("completion", result.stdout)


if __name__ == "__main__":
    unittest.main()
