#!/usr/bin/env python3
"""
test_rst2md.py - Unit tests for rst2md.py

Tests cover:
- extract_rst_metadata: title, docinfo fields, minimal/fallback cases
- rst_to_markdown_text: pypandoc path, docutils fallback, missing-library error
- _html_to_markdown: lightweight HTML→markdown helper
- rst_to_full_markdown / rst_to_plain_text: output formatters
- process_rst_file: file I/O, frontmatter in output, .txt mode
- main CLI: batch directory mode, missing file, help
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

import rst2md
from rst2md import (
    extract_rst_metadata,
    rst_to_markdown_text,
    rst_to_full_markdown,
    rst_to_plain_text,
    process_rst_file,
    _html_to_markdown,
)


# ---------------------------------------------------------------------------
# Sample RST content
# ---------------------------------------------------------------------------

SAMPLE_RST = """\
My Document Title
=================

:Author: Jane Doe
:Date: 2026-01-15
:Version: 1.0

Introduction
------------

This is a paragraph with **bold** and *italic* text.

.. code-block:: python

   print("hello")
"""

MINIMAL_RST = """\
Just some plain text with no metadata and no title.
"""

SUBSECTION_RST = """\
Top Level
=========

Section One
-----------

Content here.

Section Two
-----------

More content.
"""


# ---------------------------------------------------------------------------
# extract_rst_metadata
# ---------------------------------------------------------------------------

class TestExtractRstMetadata(unittest.TestCase):
    def _meta(self, content: str = SAMPLE_RST, name: str = "test.rst") -> dict:
        """Helper: extract metadata from an in-memory RST string."""
        with tempfile.NamedTemporaryFile(suffix=".rst", delete=False) as f:
            path = Path(f.name)
        try:
            path.write_text(content, encoding='utf-8')
            return extract_rst_metadata(content, path)
        finally:
            path.unlink(missing_ok=True)

    def test_extracts_title(self):
        meta = self._meta(SAMPLE_RST)
        self.assertEqual(meta.get('title'), 'My Document Title')

    def test_extracts_author(self):
        meta = self._meta(SAMPLE_RST)
        self.assertEqual(meta.get('author'), 'Jane Doe')

    def test_extracts_date(self):
        meta = self._meta(SAMPLE_RST)
        self.assertEqual(meta.get('date'), '2026-01-15')

    def test_extracts_version(self):
        meta = self._meta(SAMPLE_RST)
        self.assertEqual(meta.get('version'), '1.0')

    def test_no_metadata_returns_minimal(self):
        meta = self._meta(MINIMAL_RST)
        # No docinfo fields — only mandatory keys should be present
        self.assertNotIn('author', meta)
        self.assertNotIn('title', meta)
        self.assertIn('source', meta)
        self.assertIn('fetched_at', meta)

    def test_fetched_at_present(self):
        meta = self._meta(SAMPLE_RST)
        self.assertIn('fetched_at', meta)
        # Should be parseable ISO-8601
        from datetime import datetime
        datetime.strptime(meta['fetched_at'], '%Y-%m-%dT%H:%M:%SZ')

    def test_source_is_absolute(self):
        meta = self._meta(SAMPLE_RST)
        self.assertTrue(Path(meta['source']).is_absolute())

    def test_subsection_not_confused_with_title(self):
        """The top-level heading should be the title, not the first subsection."""
        meta = self._meta(SUBSECTION_RST)
        self.assertEqual(meta.get('title'), 'Top Level')

    def test_dash_underline_title(self):
        rst = "My Section\n-----------\n\nSome content.\n"
        meta = self._meta(rst)
        self.assertEqual(meta.get('title'), 'My Section')


# ---------------------------------------------------------------------------
# _html_to_markdown (docutils fallback helper)
# ---------------------------------------------------------------------------

class TestHtmlToMarkdown(unittest.TestCase):
    def test_headings_converted(self):
        html = "<h1>Title</h1><h2>Sub</h2>"
        result = _html_to_markdown(html)
        self.assertIn('# Title', result)
        self.assertIn('## Sub', result)

    def test_bold_converted(self):
        html = "<p><strong>bold text</strong></p>"
        result = _html_to_markdown(html)
        self.assertIn('**bold text**', result)

    def test_italic_converted(self):
        html = "<p><em>italic text</em></p>"
        result = _html_to_markdown(html)
        self.assertIn('*italic text*', result)

    def test_html_entities_decoded(self):
        html = "<p>a &amp; b &lt; c &gt; d</p>"
        result = _html_to_markdown(html)
        self.assertIn('a & b < c > d', result)

    def test_strips_remaining_tags(self):
        html = "<div><span>plain</span></div>"
        result = _html_to_markdown(html)
        self.assertNotIn('<', result)
        self.assertIn('plain', result)


# ---------------------------------------------------------------------------
# rst_to_markdown_text
# ---------------------------------------------------------------------------

class TestRstToMarkdownText(unittest.TestCase):
    def test_converts_basic_rst_with_pypandoc(self):
        """pypandoc should be available in this env; smoke test a real conversion."""
        result = rst_to_markdown_text("Hello **world**\n")
        # pypandoc keeps bold markers
        self.assertIn('world', result)
        # Should be a non-empty string
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_heading_becomes_markdown_heading(self):
        rst = "My Title\n========\n\nBody text.\n"
        result = rst_to_markdown_text(rst)
        self.assertIn('My Title', result)

    def test_code_block_preserved(self):
        rst = "Example\n-------\n\n.. code-block:: python\n\n   x = 1\n"
        result = rst_to_markdown_text(rst)
        self.assertIn('x = 1', result)

    def test_import_error_falls_through_to_docutils(self):
        """If pypandoc raises ImportError, docutils fallback is tried."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'pypandoc':
                raise ImportError("No module named 'pypandoc'")
            return real_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            # Force re-import so the mock is hit inside the function
            result = rst_to_markdown_text("Hello world.\n")
        # Should still produce some output (from docutils fallback)
        self.assertIsInstance(result, str)
        self.assertIn('Hello world', result)

    def test_raises_runtime_error_if_no_library_available(self):
        """If both pypandoc and docutils are unavailable, raise RuntimeError."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name in ('pypandoc', 'docutils', 'docutils.core'):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            with self.assertRaises(RuntimeError):
                rst_to_markdown_text("Hello world.\n")


# ---------------------------------------------------------------------------
# rst_to_full_markdown / rst_to_plain_text
# ---------------------------------------------------------------------------

class TestRstToFullMarkdown(unittest.TestCase):
    def test_includes_frontmatter(self):
        meta = {'title': 'Test Doc', 'author': 'Alice', 'fetched_at': '2026-03-10T00:00:00Z'}
        result = rst_to_full_markdown("# Content\n\nBody.", metadata=meta)
        self.assertIn('---', result)
        self.assertIn('title:', result)
        self.assertIn('Body.', result)

    def test_no_metadata_no_frontmatter(self):
        result = rst_to_full_markdown("# Content\n\nBody.", metadata=None)
        self.assertNotIn('---', result)
        self.assertIn('Body.', result)

    def test_title_heading_added_when_requested(self):
        result = rst_to_full_markdown("Body text.", metadata=None, title="My Title")
        self.assertIn('# My Title', result)

    def test_no_title_when_not_requested(self):
        result = rst_to_full_markdown("Body text.", metadata=None, title=None)
        self.assertNotIn('# ', result)


class TestRstToPlainText(unittest.TestCase):
    def test_removes_heading_markers(self):
        md = "# Top Heading\n\n## Sub Heading\n\nContent."
        result = rst_to_plain_text(md)
        self.assertNotIn('#', result)
        self.assertIn('Top Heading', result)
        self.assertIn('Content.', result)

    def test_removes_bold_markers(self):
        md = "Some **bold** text."
        result = rst_to_plain_text(md)
        self.assertNotIn('**', result)
        self.assertIn('bold', result)

    def test_removes_code_fences(self):
        md = "Code:\n\n```python\nx = 1\n```\n"
        result = rst_to_plain_text(md)
        self.assertNotIn('```', result)
        self.assertIn('x = 1', result)


# ---------------------------------------------------------------------------
# process_rst_file
# ---------------------------------------------------------------------------

class TestProcessRstFile(unittest.TestCase):
    def test_creates_md_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            rst_path = tmpdir_path / "test.rst"
            rst_path.write_text(SAMPLE_RST, encoding='utf-8')
            out_dir = tmpdir_path / "out"

            out_path = process_rst_file(rst_path, out_dir, 'md')

            self.assertTrue(out_path.exists())
            self.assertEqual(out_path.suffix, '.md')
            self.assertEqual(out_path.stem, 'test')

    def test_frontmatter_in_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            rst_path = tmpdir_path / "doc.rst"
            rst_path.write_text(SAMPLE_RST, encoding='utf-8')

            out_path = process_rst_file(rst_path, tmpdir_path / "out", 'md')
            content = out_path.read_text(encoding='utf-8')

            self.assertIn('---', content)
            self.assertIn('fetched_at:', content)

    def test_txt_format_no_frontmatter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            rst_path = tmpdir_path / "doc.rst"
            rst_path.write_text(MINIMAL_RST, encoding='utf-8')

            out_path = process_rst_file(rst_path, tmpdir_path / "out", 'txt')
            self.assertEqual(out_path.suffix, '.txt')
            content = out_path.read_text(encoding='utf-8')
            # Plain text mode: no YAML delimiters
            self.assertNotIn('fetched_at:', content)

    def test_output_dir_created_if_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rst_path = Path(tmpdir) / "doc.rst"
            rst_path.write_text(SAMPLE_RST, encoding='utf-8')
            new_dir = Path(tmpdir) / "deeply" / "nested" / "dir"

            out_path = process_rst_file(rst_path, new_dir, 'md')
            self.assertTrue(out_path.exists())

    def test_title_extracted_in_frontmatter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            rst_path = tmpdir_path / "titled.rst"
            rst_path.write_text(SAMPLE_RST, encoding='utf-8')

            out_path = process_rst_file(rst_path, tmpdir_path / "out", 'md')
            content = out_path.read_text(encoding='utf-8')
            self.assertIn('My Document Title', content)


# ---------------------------------------------------------------------------
# Batch directory mode via CLI
# ---------------------------------------------------------------------------

class TestBatchMode(unittest.TestCase):
    def test_processes_all_rst_files(self):
        from typer.testing import CliRunner
        from rst2md import app

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "a.rst").write_text(SAMPLE_RST, encoding='utf-8')
            (tmpdir_path / "b.rst").write_text(MINIMAL_RST, encoding='utf-8')
            out_dir = tmpdir_path / "out"

            runner = CliRunner()
            result = runner.invoke(app, [str(tmpdir_path), "-o", str(out_dir)])

            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertTrue((out_dir / "a.md").exists())
            self.assertTrue((out_dir / "b.md").exists())

    def test_empty_directory_exits_with_error(self):
        from typer.testing import CliRunner
        from rst2md import app

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            result = runner.invoke(app, [tmpdir])
            self.assertNotEqual(result.exit_code, 0)

    def test_missing_file_exits_with_error(self):
        from typer.testing import CliRunner
        from rst2md import app

        runner = CliRunner()
        result = runner.invoke(app, ["/nonexistent/path/file.rst"])
        self.assertNotEqual(result.exit_code, 0)


class TestCLI(unittest.TestCase):
    def test_help_exits_cleanly(self):
        from typer.testing import CliRunner
        from rst2md import app

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("rst", result.output.lower())

    def test_single_file_end_to_end(self):
        """Convert a real .rst file to markdown end-to-end."""
        from typer.testing import CliRunner
        from rst2md import app

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            rst_path = tmpdir_path / "readme.rst"
            rst_path.write_text(SAMPLE_RST, encoding='utf-8')
            out_dir = tmpdir_path / "out"

            runner = CliRunner()
            result = runner.invoke(app, [str(rst_path), "-o", str(out_dir)])

            self.assertEqual(result.exit_code, 0, msg=result.output)
            out_file = out_dir / "readme.md"
            self.assertTrue(out_file.exists())
            content = out_file.read_text(encoding='utf-8')
            self.assertIn('---', content)
            self.assertIn('Introduction', content)

    def test_txt_format_flag(self):
        """The -f txt flag produces plain text output without YAML frontmatter."""
        from typer.testing import CliRunner
        from rst2md import app

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            rst_path = tmpdir_path / "doc.rst"
            rst_path.write_text(SAMPLE_RST, encoding='utf-8')
            out_dir = tmpdir_path / "out"

            runner = CliRunner()
            result = runner.invoke(app, [str(rst_path), "-o", str(out_dir), "-f", "txt"])

            self.assertEqual(result.exit_code, 0, msg=result.output)
            out_file = out_dir / "doc.txt"
            self.assertTrue(out_file.exists())
            content = out_file.read_text(encoding='utf-8')
            self.assertNotIn('fetched_at:', content)
            self.assertIn('Introduction', content)


if __name__ == "__main__":
    unittest.main()
