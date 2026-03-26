"""Tests for the main CLI entry point (cli.py)."""

import subprocess
import sys
import unittest

from any2md import __version__


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


if __name__ == "__main__":
    unittest.main()
