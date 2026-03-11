#!/usr/bin/env python3
"""
test_html2md.py — Unit tests for html2md.py

Tests cover:
- extract_meta_tags: title/description/author extraction from HTML
- _extract_meta: both attribute orderings, case insensitivity
- html_path_to_stem: safe filename derivation
- process_html_file: single-file processing with mocked model
- Batch mode: directory with multiple HTML files
- Output format: md (with frontmatter) vs txt (no frontmatter)

All mlx-lm calls are mocked so tests run without model downloads.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from html2md import (
    extract_meta_tags,
    _extract_meta,
    html_path_to_stem,
    process_html_file,
)
from md_common import OutputFormat


# ---------------------------------------------------------------------------
# TestExtractMetaTags
# ---------------------------------------------------------------------------

class TestExtractMetaTags(unittest.TestCase):
    """Tests for extract_meta_tags() — metadata from <title> and <meta> tags."""

    def test_extracts_title_from_title_tag(self):
        html = "<html><head><title>My Article</title></head><body></body></html>"
        meta = extract_meta_tags(html)
        self.assertEqual(meta.get('title'), 'My Article')

    def test_extracts_description_from_meta(self):
        html = (
            '<html><head>'
            '<meta name="description" content="A great article.">'
            '</head><body></body></html>'
        )
        meta = extract_meta_tags(html)
        self.assertEqual(meta.get('description'), 'A great article.')

    def test_extracts_author_from_meta(self):
        html = (
            '<html><head>'
            '<meta name="author" content="Jane Smith">'
            '</head><body></body></html>'
        )
        meta = extract_meta_tags(html)
        self.assertEqual(meta.get('author'), 'Jane Smith')

    def test_missing_tags_returns_empty_dict(self):
        html = "<html><body>No meta here</body></html>"
        meta = extract_meta_tags(html)
        self.assertNotIn('title', meta)
        self.assertNotIn('description', meta)
        self.assertNotIn('author', meta)

    def test_malformed_html_does_not_crash(self):
        """extract_meta_tags must not raise on broken/incomplete HTML."""
        html = "<html><head><title>Broken"
        try:
            meta = extract_meta_tags(html)
            # May or may not find title depending on regex — just no exception
        except Exception as exc:
            self.fail(f"extract_meta_tags raised on malformed HTML: {exc}")

    def test_title_whitespace_collapsed(self):
        """Multi-line or padded <title> content has leading/trailing space stripped and internal runs collapsed."""
        html = "<html><head><title>  Whitespace   Title  </title></head></html>"
        meta = extract_meta_tags(html)
        # Internal whitespace runs are collapsed to a single space
        self.assertEqual(meta.get('title'), 'Whitespace Title')

    def test_all_three_tags_present(self):
        html = (
            '<html><head>'
            '<title>Full Page</title>'
            '<meta name="description" content="Desc text.">'
            '<meta name="author" content="Bob">'
            '</head></html>'
        )
        meta = extract_meta_tags(html)
        self.assertEqual(meta['title'], 'Full Page')
        self.assertEqual(meta['description'], 'Desc text.')
        self.assertEqual(meta['author'], 'Bob')

    def test_empty_string_input(self):
        """extract_meta_tags handles completely empty input."""
        meta = extract_meta_tags('')
        self.assertEqual(meta, {})


# ---------------------------------------------------------------------------
# TestExtractMeta (private helper)
# ---------------------------------------------------------------------------

class TestExtractMetaHelper(unittest.TestCase):
    """Tests for _extract_meta() — flexible <meta> attribute order handling."""

    def test_name_before_content(self):
        html = '<meta name="description" content="Hello world">'
        self.assertEqual(_extract_meta(html, 'description'), 'Hello world')

    def test_content_before_name(self):
        """Reversed attribute order (content first, then name) is handled."""
        html = '<meta content="Reversed" name="description">'
        self.assertEqual(_extract_meta(html, 'description'), 'Reversed')

    def test_returns_none_when_missing(self):
        html = '<html></html>'
        self.assertIsNone(_extract_meta(html, 'description'))

    def test_empty_content_returns_none(self):
        html = '<meta name="author" content="">'
        self.assertIsNone(_extract_meta(html, 'author'))


# ---------------------------------------------------------------------------
# TestHtmlPathToStem
# ---------------------------------------------------------------------------

class TestHtmlPathToStem(unittest.TestCase):
    """Tests for html_path_to_stem() — output filename derivation."""

    def test_simple_name(self):
        path = Path('/tmp/article.html')
        self.assertEqual(html_path_to_stem(path), 'article')

    def test_name_with_spaces_replaced(self):
        path = Path('/tmp/my article page.html')
        stem = html_path_to_stem(path)
        self.assertNotIn(' ', stem)

    def test_truncates_to_80_chars(self):
        long_name = 'a' * 100 + '.html'
        path = Path(f'/tmp/{long_name}')
        stem = html_path_to_stem(path)
        self.assertLessEqual(len(stem), 80)

    def test_htm_extension(self):
        path = Path('/tmp/page.htm')
        self.assertEqual(html_path_to_stem(path), 'page')


# ---------------------------------------------------------------------------
# TestProcessHtmlFile
# ---------------------------------------------------------------------------

class TestProcessHtmlFile(unittest.TestCase):
    """Tests for process_html_file() — single-file conversion."""

    def _make_mock_model(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template = MagicMock(return_value="<prompt>")
        return mock_model, mock_tokenizer

    def test_writes_output_file_md(self):
        """process_html_file writes a .md file when format=md."""
        html_content = (
            '<html><head><title>Test Page</title></head>'
            '<body><p>Hello world</p></body></html>'
        )
        mock_model, mock_tokenizer = self._make_mock_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / 'test_page.html'
            html_path.write_text(html_content)
            output_dir = Path(tmpdir) / 'out'

            with patch('html2md.html_to_markdown', return_value='# Test Page\n\nHello world'):
                result = process_html_file(
                    html_path, mock_model, mock_tokenizer, output_dir, OutputFormat.md
                )

            self.assertTrue(result.exists())
            self.assertEqual(result.suffix, '.md')
            text = result.read_text()
            self.assertIn('---', text)           # frontmatter present
            self.assertIn('# Test Page', text)  # body content

    def test_writes_output_file_txt(self):
        """process_html_file writes a .txt file when format=txt (no frontmatter)."""
        html_content = '<html><head><title>Plain</title></head><body>Content</body></html>'
        mock_model, mock_tokenizer = self._make_mock_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / 'plain.html'
            html_path.write_text(html_content)
            output_dir = Path(tmpdir) / 'out'

            with patch('html2md.html_to_markdown', return_value='Content'):
                result = process_html_file(
                    html_path, mock_model, mock_tokenizer, output_dir, OutputFormat.txt
                )

            self.assertEqual(result.suffix, '.txt')
            text = result.read_text()
            self.assertNotIn('---', text)

    def test_frontmatter_contains_source_and_fetched_at(self):
        """md output always includes source path and fetched_at in frontmatter."""
        html_content = '<html><head><title>Meta Check</title></head><body></body></html>'
        mock_model, mock_tokenizer = self._make_mock_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / 'meta_check.html'
            html_path.write_text(html_content)
            output_dir = Path(tmpdir) / 'out'

            with patch('html2md.html_to_markdown', return_value='# Meta Check'):
                result = process_html_file(
                    html_path, mock_model, mock_tokenizer, output_dir, OutputFormat.md
                )

            text = result.read_text()
            self.assertIn('source:', text)
            self.assertIn('fetched_at:', text)

    def test_metadata_title_in_frontmatter(self):
        """Title extracted from <title> tag appears in md frontmatter."""
        html_content = '<html><head><title>My Title</title></head><body></body></html>'
        mock_model, mock_tokenizer = self._make_mock_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / 'titled.html'
            html_path.write_text(html_content)
            output_dir = Path(tmpdir) / 'out'

            with patch('html2md.html_to_markdown', return_value='# My Title'):
                result = process_html_file(
                    html_path, mock_model, mock_tokenizer, output_dir, OutputFormat.md
                )

            text = result.read_text()
            self.assertIn('My Title', text)

    def test_creates_output_dir_if_missing(self):
        """process_html_file creates the output directory automatically."""
        html_content = '<html><body>Test</body></html>'
        mock_model, mock_tokenizer = self._make_mock_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / 'test.html'
            html_path.write_text(html_content)
            output_dir = Path(tmpdir) / 'nested' / 'deep' / 'out'

            self.assertFalse(output_dir.exists())

            with patch('html2md.html_to_markdown', return_value='content'):
                process_html_file(
                    html_path, mock_model, mock_tokenizer, output_dir, OutputFormat.md
                )

            self.assertTrue(output_dir.exists())

    def test_returns_path_to_output_file(self):
        """process_html_file returns the Path of the written output file."""
        html_content = '<html><body>Return test</body></html>'
        mock_model, mock_tokenizer = self._make_mock_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / 'return_test.html'
            html_path.write_text(html_content)
            output_dir = Path(tmpdir) / 'out'

            with patch('html2md.html_to_markdown', return_value='content'):
                result = process_html_file(
                    html_path, mock_model, mock_tokenizer, output_dir, OutputFormat.md
                )

            self.assertIsInstance(result, Path)
            self.assertEqual(result.name, 'return_test.md')


# ---------------------------------------------------------------------------
# TestBatchMode
# ---------------------------------------------------------------------------

class TestBatchMode(unittest.TestCase):
    """Tests for batch directory processing via the CLI main() path."""

    def test_batch_processes_all_html_files(self):
        """All .html files in a directory are processed in batch mode."""
        from typer.testing import CliRunner
        from html2md import app

        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / 'html_files'
            input_dir.mkdir()
            output_dir = Path(tmpdir) / 'output'

            # Write two HTML files
            (input_dir / 'page1.html').write_text('<html><title>Page1</title><body>A</body></html>')
            (input_dir / 'page2.html').write_text('<html><title>Page2</title><body>B</body></html>')

            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_tokenizer.apply_chat_template = MagicMock(return_value='<prompt>')

            with patch('html2md.load_reader_model', return_value=(mock_model, mock_tokenizer)):
                with patch('html2md.html_to_markdown', return_value='# Content'):
                    result = runner.invoke(app, [str(input_dir), '-o', str(output_dir)])

            self.assertEqual(result.exit_code, 0, msg=result.output)
            output_files = list(output_dir.glob('*.md'))
            self.assertEqual(len(output_files), 2)

    def test_batch_handles_htm_extension(self):
        """Batch mode includes .htm files as well as .html files."""
        from typer.testing import CliRunner
        from html2md import app

        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / 'mixed'
            input_dir.mkdir()
            output_dir = Path(tmpdir) / 'out'

            (input_dir / 'old_page.htm').write_text('<html><body>Old</body></html>')

            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_tokenizer.apply_chat_template = MagicMock(return_value='<prompt>')

            with patch('html2md.load_reader_model', return_value=(mock_model, mock_tokenizer)):
                with patch('html2md.html_to_markdown', return_value='content'):
                    result = runner.invoke(app, [str(input_dir), '-o', str(output_dir)])

            self.assertEqual(result.exit_code, 0, msg=result.output)
            output_files = list(output_dir.glob('*.md'))
            self.assertEqual(len(output_files), 1)

    def test_empty_directory_exits_with_error(self):
        """Batch mode exits with code 1 when no HTML files are found."""
        from typer.testing import CliRunner
        from html2md import app

        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            empty_dir = Path(tmpdir) / 'empty'
            empty_dir.mkdir()

            result = runner.invoke(app, [str(empty_dir)])
            self.assertNotEqual(result.exit_code, 0)


# ---------------------------------------------------------------------------
# TestOutputFormat
# ---------------------------------------------------------------------------

class TestOutputFormat(unittest.TestCase):
    """Tests for output format handling in html2md."""

    def test_md_output_has_frontmatter(self):
        """OutputFormat.md produces output that starts with YAML frontmatter."""
        mock_model, mock_tokenizer = MagicMock(), MagicMock()
        mock_tokenizer.apply_chat_template = MagicMock(return_value='<prompt>')

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / 'fm_test.html'
            html_path.write_text('<html><title>FM Test</title><body>X</body></html>')
            out_dir = Path(tmpdir) / 'out'

            with patch('html2md.html_to_markdown', return_value='# FM Test\n\nX'):
                result = process_html_file(
                    html_path, mock_model, mock_tokenizer, out_dir, OutputFormat.md
                )

            text = result.read_text()
            self.assertTrue(text.startswith('---'))

    def test_txt_output_has_no_frontmatter(self):
        """OutputFormat.txt produces plain text without YAML delimiters."""
        mock_model, mock_tokenizer = MagicMock(), MagicMock()
        mock_tokenizer.apply_chat_template = MagicMock(return_value='<prompt>')

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / 'txt_test.html'
            html_path.write_text('<html><body>Plain</body></html>')
            out_dir = Path(tmpdir) / 'out'

            with patch('html2md.html_to_markdown', return_value='Plain content'):
                result = process_html_file(
                    html_path, mock_model, mock_tokenizer, out_dir, OutputFormat.txt
                )

            text = result.read_text()
            self.assertFalse(text.strip().startswith('---'))
            self.assertIn('Plain content', text)

    def test_single_file_cli_md(self):
        """CLI processes a single HTML file in md mode end-to-end."""
        from typer.testing import CliRunner
        from html2md import app

        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / 'single.html'
            html_path.write_text('<html><title>Single</title><body>Body</body></html>')
            out_dir = Path(tmpdir) / 'out'

            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_tokenizer.apply_chat_template = MagicMock(return_value='<prompt>')

            with patch('html2md.load_reader_model', return_value=(mock_model, mock_tokenizer)):
                with patch('html2md.html_to_markdown', return_value='# Single\n\nBody'):
                    result = runner.invoke(
                        app, [str(html_path), '-o', str(out_dir), '-f', 'md']
                    )

            self.assertEqual(result.exit_code, 0, msg=result.output)
            out_file = out_dir / 'single.md'
            self.assertTrue(out_file.exists())


if __name__ == "__main__":
    unittest.main()
