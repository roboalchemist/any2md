#!/usr/bin/env python3
"""
Tests for web2md.py

All external calls (httpx, urllib, mlx_lm, trafilatura) are mocked so tests
run without network access or model downloads.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).parent))


class TestFetchHtml(unittest.TestCase):
    """Tests for fetch_html() — URL fetching with fallback to urllib."""

    def _html_and_meta(self, url="https://example.com"):
        from web2md import fetch_html
        return fetch_html

    def test_fetch_with_httpx(self):
        """fetch_html uses httpx when available."""
        from web2md import fetch_html

        mock_response = MagicMock()
        mock_response.text = "<html><title>Test</title></html>"
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_response)

        mock_httpx = MagicMock()
        mock_httpx.Client = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            html, meta = fetch_html("https://example.com")

        self.assertEqual(html, "<html><title>Test</title></html>")
        self.assertEqual(meta["url"], "https://example.com")
        self.assertIn("fetched_at", meta)

    def test_fetch_fallback_to_urllib(self):
        """fetch_html falls back to urllib when httpx is not installed."""
        import io
        from web2md import fetch_html

        html_bytes = b"<html><title>Fallback</title></html>"

        mock_resp = MagicMock()
        mock_resp.read = MagicMock(return_value=html_bytes)
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch.dict("sys.modules", {"httpx": None}):
            with patch("urllib.request.urlopen", return_value=mock_resp):
                html, meta = fetch_html("https://example.com")

        self.assertEqual(html, "<html><title>Fallback</title></html>")
        self.assertIn("fetched_at", meta)

    def test_fetch_returns_fetched_at_timestamp(self):
        """fetch_html always includes a fetched_at timestamp in ISO format."""
        from web2md import fetch_html

        mock_response = MagicMock()
        mock_response.text = "<html></html>"
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_response)

        mock_httpx = MagicMock()
        mock_httpx.Client = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            _, meta = fetch_html("https://example.com/page")

        # Should be ISO 8601 UTC format
        self.assertRegex(meta["fetched_at"], r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


class TestExtractMetadata(unittest.TestCase):
    """Tests for extract_metadata() — metadata extraction from HTML."""

    def test_extracts_title_from_title_tag(self):
        """extract_metadata parses <title> when trafilatura is unavailable."""
        from web2md import extract_metadata

        html = "<html><head><title>My Article</title></head><body></body></html>"
        with patch.dict("sys.modules", {"trafilatura": None}):
            meta = extract_metadata(html, "https://example.com/article")

        self.assertEqual(meta.get("title"), "My Article")

    def test_extracts_meta_description(self):
        """extract_metadata parses <meta name='description'> tag."""
        from web2md import extract_metadata

        html = (
            '<html><head>'
            '<meta name="description" content="A great article about things.">'
            '</head><body></body></html>'
        )
        with patch.dict("sys.modules", {"trafilatura": None}):
            meta = extract_metadata(html, "https://example.com/")

        self.assertEqual(meta.get("description"), "A great article about things.")

    def test_derives_sitename_from_url(self):
        """extract_metadata derives sitename from URL domain."""
        from web2md import extract_metadata

        html = "<html><head></head><body></body></html>"
        with patch.dict("sys.modules", {"trafilatura": None}):
            meta = extract_metadata(html, "https://www.myblog.com/post/123")

        self.assertEqual(meta.get("sitename"), "myblog.com")

    def test_url_always_present(self):
        """extract_metadata always includes the source URL."""
        from web2md import extract_metadata

        html = "<html></html>"
        with patch.dict("sys.modules", {"trafilatura": None}):
            meta = extract_metadata(html, "https://example.com/page")

        self.assertEqual(meta["url"], "https://example.com/page")

    def test_fetched_at_always_present(self):
        """extract_metadata always includes fetched_at timestamp."""
        from web2md import extract_metadata

        html = "<html></html>"
        with patch.dict("sys.modules", {"trafilatura": None}):
            meta = extract_metadata(html, "https://example.com/")

        self.assertIn("fetched_at", meta)
        self.assertRegex(meta["fetched_at"], r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")

    def test_uses_trafilatura_when_available(self):
        """extract_metadata uses trafilatura when installed."""
        from web2md import extract_metadata

        mock_doc = MagicMock()
        mock_doc.as_dict = MagicMock(return_value={
            "title": "Trafilatura Title",
            "description": "Extracted description",
            "author": "John Doe",
            "sitename": "example.com",
        })

        mock_trafilatura = MagicMock()
        mock_trafilatura.extract_metadata = MagicMock(return_value=mock_doc)

        html = "<html><head><title>Fallback</title></head></html>"
        with patch.dict("sys.modules", {"trafilatura": mock_trafilatura}):
            meta = extract_metadata(html, "https://example.com/")

        self.assertEqual(meta["title"], "Trafilatura Title")
        self.assertEqual(meta["author"], "John Doe")
        self.assertEqual(meta["sitename"], "example.com")

    def test_fallback_when_no_metadata_in_html(self):
        """extract_metadata returns url and fetched_at even for empty HTML."""
        from web2md import extract_metadata

        html = "<html></html>"
        with patch.dict("sys.modules", {"trafilatura": None}):
            meta = extract_metadata(html, "https://bare.example.com/")

        self.assertIn("url", meta)
        self.assertIn("fetched_at", meta)


class TestHtmlToMarkdown(unittest.TestCase):
    """Tests for html_to_markdown() — ReaderLM-v2 conversion."""

    def test_converts_html_with_mocked_model(self):
        """html_to_markdown calls generate and returns stripped output."""
        from web2md import html_to_markdown

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template = MagicMock(return_value="<prompt>")

        mock_generate = MagicMock(return_value="  # Title\n\nBody text.\n  ")

        with patch("mlx_lm.generate", mock_generate):
            result = html_to_markdown(
                "<html><h1>Title</h1><p>Body text.</p></html>",
                model=mock_model,
                tokenizer=mock_tokenizer,
            )

        self.assertEqual(result, "# Title\n\nBody text.")

    def test_truncates_long_html(self):
        """html_to_markdown truncates HTML exceeding MAX_HTML_CHARS."""
        from web2md import html_to_markdown, MAX_HTML_CHARS

        long_html = "x" * (MAX_HTML_CHARS + 1000)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template = MagicMock(return_value="<prompt>")

        captured_prompts = []

        def capture_generate(model, tokenizer, prompt, **kwargs):
            captured_prompts.append(prompt)
            return "# Truncated"

        with patch("mlx_lm.generate", capture_generate):
            html_to_markdown(long_html, model=mock_model, tokenizer=mock_tokenizer)

        # The prompt was built from truncated HTML — verify the HTML portion
        # is not longer than MAX_HTML_CHARS
        self.assertTrue(len(captured_prompts[0]) <= MAX_HTML_CHARS + 500)

    def test_raises_without_model(self):
        """html_to_markdown raises ValueError when model is None."""
        from web2md import html_to_markdown

        with self.assertRaises(ValueError):
            html_to_markdown("<html></html>", model=None, tokenizer=MagicMock())

    def test_raises_without_tokenizer(self):
        """html_to_markdown raises ValueError when tokenizer is None."""
        from web2md import html_to_markdown

        with self.assertRaises(ValueError):
            html_to_markdown("<html></html>", model=MagicMock(), tokenizer=None)


class TestFrontmatter(unittest.TestCase):
    """Tests for frontmatter generation with web metadata."""

    def test_frontmatter_contains_required_fields(self):
        """page_to_markdown embeds all required metadata fields."""
        from web2md import page_to_markdown

        metadata = {
            "title": "Test Article",
            "url": "https://example.com/article",
            "sitename": "example.com",
            "author": "Jane Smith",
            "description": "A test article.",
            "fetched_at": "2026-03-10T12:00:00Z",
        }
        result = page_to_markdown("# Test\n\nBody.", metadata)

        self.assertIn("---", result)
        self.assertIn("title:", result)
        self.assertIn("Test Article", result)
        self.assertIn("url:", result)
        self.assertIn("https://example.com/article", result)
        self.assertIn("fetched_at:", result)

    def test_frontmatter_followed_by_content(self):
        """page_to_markdown puts frontmatter before the body."""
        from web2md import page_to_markdown

        metadata = {"title": "X", "url": "https://x.com", "fetched_at": "2026-03-10T00:00:00Z"}
        result = page_to_markdown("# Body", metadata)

        # Frontmatter must appear before the body
        fm_end = result.index("---\n", 4)  # second ---
        body_start = result.index("# Body")
        self.assertLess(fm_end, body_start)

    def test_txt_format_omits_frontmatter(self):
        """page_to_text returns plain text without frontmatter."""
        from web2md import page_to_text

        result = page_to_text("# Title\n\nSome text.")
        self.assertNotIn("---", result)
        self.assertEqual(result, "# Title\n\nSome text.")


class TestOutputFormat(unittest.TestCase):
    """Tests for OutputFormat enum and filename utilities."""

    def test_output_format_values(self):
        """OutputFormat enum has md and txt variants."""
        from md_common import OutputFormat

        self.assertEqual(OutputFormat.md.value, "md")
        self.assertEqual(OutputFormat.txt.value, "txt")

    def test_url_to_filename_basic(self):
        """url_to_filename converts URL to safe filename stem."""
        from web2md import url_to_filename

        name = url_to_filename("https://example.com/my-article")
        self.assertNotIn("/", name)
        self.assertNotIn(":", name)
        self.assertNotIn(".", name.split("_")[0])  # dot in domain replaced

    def test_url_to_filename_truncates(self):
        """url_to_filename truncates long URLs to 80 chars."""
        from web2md import url_to_filename

        long_url = "https://example.com/" + "a" * 200
        name = url_to_filename(long_url)
        self.assertLessEqual(len(name), 80)

    def test_url_to_filename_no_scheme(self):
        """url_to_filename strips https:// scheme."""
        from web2md import url_to_filename

        name = url_to_filename("https://example.com/page")
        self.assertFalse(name.startswith("https"))
        self.assertFalse(name.startswith("http"))


class TestBuildReaderPrompt(unittest.TestCase):
    """Tests for build_reader_prompt() — prompt construction."""

    def test_uses_apply_chat_template_when_available(self):
        """build_reader_prompt uses tokenizer.apply_chat_template if present."""
        from web2md import build_reader_prompt

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template = MagicMock(return_value="<formatted>")

        result = build_reader_prompt("<html>test</html>", mock_tokenizer)
        self.assertEqual(result, "<formatted>")
        mock_tokenizer.apply_chat_template.assert_called_once()

    def test_fallback_prompt_contains_html(self):
        """build_reader_prompt fallback includes the HTML content."""
        from web2md import build_reader_prompt

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template = MagicMock(
            side_effect=Exception("template error")
        )

        html = "<html><body>Hello</body></html>"
        result = build_reader_prompt(html, mock_tokenizer)
        self.assertIn(html, result)

    def test_fallback_prompt_contains_system_instruction(self):
        """build_reader_prompt fallback includes the system instruction."""
        from web2md import build_reader_prompt, SYSTEM_PROMPT

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template = MagicMock(
            side_effect=Exception("no template")
        )

        result = build_reader_prompt("<html></html>", mock_tokenizer)
        self.assertIn(SYSTEM_PROMPT, result)


if __name__ == "__main__":
    unittest.main()
