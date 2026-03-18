#!/usr/bin/env python3
"""
Tests for pdf2md.py

Run with: python -m pytest test_pdf2md.py -v
"""

import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from tomd.pdf import (
    parse_page_range,
    pages_to_markdown,
    pages_to_text,
    render_page_as_image,
    extract_page_via_vlm,
    extract_pages_hybrid,
    THIN_PAGE_THRESHOLD,
    VLM_MODEL_DEFAULT,
)
import tomd.pdf as pdf2md  # for patching


class TestPdf2Md(unittest.TestCase):

    def test_parse_page_range_single(self):
        self.assertEqual(parse_page_range("3", 10), [2])

    def test_parse_page_range_range(self):
        self.assertEqual(parse_page_range("1-5", 10), [0, 1, 2, 3, 4])

    def test_parse_page_range_mixed(self):
        self.assertEqual(parse_page_range("1-3,5,8-10", 10), [0, 1, 2, 4, 7, 8, 9])

    def test_parse_page_range_clamps_to_total(self):
        result = parse_page_range("1-100", 5)
        self.assertEqual(result, [0, 1, 2, 3, 4])

    def test_parse_page_range_out_of_bounds_ignored(self):
        result = parse_page_range("0,6", 5)
        self.assertEqual(result, [])

    def test_pages_to_markdown_basic(self):
        pages = [
            {'page': 1, 'text': 'Hello world', 'is_thin': False},
            {'page': 2, 'text': 'Second page content', 'is_thin': False},
        ]
        result = pages_to_markdown(pages, title="Test Doc")
        self.assertIn("# Test Doc", result)
        self.assertIn("## Page 1", result)
        self.assertIn("Hello world", result)
        self.assertIn("## Page 2", result)
        self.assertIn("Second page content", result)

    def test_pages_to_markdown_thin_page(self):
        pages = [
            {'page': 1, 'text': '', 'is_thin': True},
        ]
        result = pages_to_markdown(pages)
        self.assertIn("image-only", result)

    def test_pages_to_markdown_with_metadata(self):
        pages = [{'page': 1, 'text': 'Content', 'is_thin': False}]
        metadata = {
            'title': 'My PDF',
            'author': 'Jane',
            'pages': 10,
            'fetched_at': '2026-01-01T00:00:00Z',
        }
        result = pages_to_markdown(pages, metadata=metadata, title="My PDF")
        self.assertTrue(result.startswith("---"))
        self.assertIn("author: Jane", result)
        self.assertIn("pages: 10", result)
        self.assertIn("# My PDF", result)

    def test_pages_to_text(self):
        pages = [
            {'page': 1, 'text': 'First page', 'is_thin': False},
            {'page': 2, 'text': 'Second page', 'is_thin': False},
        ]
        result = pages_to_text(pages)
        self.assertEqual(result, "First page\n\nSecond page")

    def test_pages_to_text_skips_empty(self):
        pages = [
            {'page': 1, 'text': '', 'is_thin': True},
            {'page': 2, 'text': 'Content', 'is_thin': False},
        ]
        result = pages_to_text(pages)
        self.assertEqual(result, "Content")


class TestVlmFallback(unittest.TestCase):
    """Tests for VLM-based OCR fallback in pdf2md."""

    def test_render_page_as_image_returns_pil_image(self):
        """render_page_as_image() should return a PIL Image from a fitz pixmap."""
        # Build a minimal fake pixmap that returns PNG bytes
        fake_png_bytes = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100  # minimal PNG-like bytes

        fake_pix = MagicMock()
        fake_pix.tobytes.return_value = fake_png_bytes

        fake_page = MagicMock()
        fake_page.get_pixmap.return_value = fake_pix

        fake_doc = MagicMock()
        fake_doc.__getitem__ = MagicMock(return_value=fake_page)

        fake_pil_image = MagicMock()

        with patch('tomd.pdf.fitz') as mock_fitz, \
             patch('tomd.pdf.Image') as mock_pil_image_module, \
             patch('tomd.pdf.io') as mock_io:
            mock_fitz.open.return_value = fake_doc
            mock_fitz.Matrix.return_value = MagicMock()
            mock_io.BytesIO.return_value = MagicMock()
            mock_pil_image_module.open.return_value = fake_pil_image

            result = render_page_as_image(Path('/fake/doc.pdf'), page_index=2, dpi=150)

            mock_fitz.open.assert_called_once_with('/fake/doc.pdf')
            fake_doc.__getitem__.assert_called_once_with(2)
            fake_page.get_pixmap.assert_called_once()
            fake_pix.tobytes.assert_called_once_with("png")
            self.assertEqual(result, fake_pil_image)

    def test_thin_page_triggers_vlm(self):
        """extract_pages_hybrid() should use VLM for pages below THIN_PAGE_THRESHOLD when ocr=True."""
        thin_pages = [
            {'page': 1, 'text': 'x' * 10, 'is_thin': True},   # 10 chars, thin
            {'page': 2, 'text': 'x' * 200, 'is_thin': False},  # 200 chars, not thin
        ]
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_config = {}

        with patch('tomd.pdf.extract_pages', return_value=thin_pages), \
             patch('tomd.pdf.extract_page_via_vlm', return_value='VLM extracted text') as mock_vlm:
            results = extract_pages_hybrid(
                '/fake/doc.pdf',
                ocr=True,
                force_ocr=False,
                vlm_model=mock_model,
                vlm_processor=mock_processor,
                vlm_config=mock_config,
            )

        # Page 1 is thin → VLM should have been used
        self.assertTrue(results[0]['ocr_used'])
        self.assertEqual(results[0]['text'], 'VLM extracted text')

        # Page 2 is not thin → text extraction used
        self.assertFalse(results[1]['ocr_used'])
        self.assertEqual(results[1]['text'], 'x' * 200)

        # VLM called only once (for page 1)
        mock_vlm.assert_called_once()

    def test_normal_page_skips_vlm(self):
        """extract_pages_hybrid() should NOT call VLM on pages with enough text when ocr=True."""
        normal_pages = [
            {'page': 1, 'text': 'x' * 500, 'is_thin': False},
            {'page': 2, 'text': 'y' * 300, 'is_thin': False},
        ]
        with patch('tomd.pdf.extract_pages', return_value=normal_pages), \
             patch('tomd.pdf.extract_page_via_vlm') as mock_vlm:
            results = extract_pages_hybrid(
                '/fake/doc.pdf',
                ocr=True,
                force_ocr=False,
                vlm_model=MagicMock(),
                vlm_processor=MagicMock(),
                vlm_config={},
            )

        # VLM should not have been called at all
        mock_vlm.assert_not_called()
        for r in results:
            self.assertFalse(r['ocr_used'])

    def test_force_ocr_uses_vlm_on_all_pages(self):
        """extract_pages_hybrid() with force_ocr=True should use VLM on every page."""
        all_pages = [
            {'page': 1, 'text': 'x' * 500, 'is_thin': False},
            {'page': 2, 'text': '', 'is_thin': True},
            {'page': 3, 'text': 'normal content here', 'is_thin': False},
        ]
        with patch('tomd.pdf.extract_pages', return_value=all_pages), \
             patch('tomd.pdf.extract_page_via_vlm', return_value='VLM result') as mock_vlm:
            results = extract_pages_hybrid(
                '/fake/doc.pdf',
                ocr=False,
                force_ocr=True,
                vlm_model=MagicMock(),
                vlm_processor=MagicMock(),
                vlm_config={},
            )

        # VLM called for all 3 pages
        self.assertEqual(mock_vlm.call_count, 3)
        for r in results:
            self.assertTrue(r['ocr_used'])
            self.assertEqual(r['text'], 'VLM result')

    def test_no_ocr_flag_skips_vlm_entirely(self):
        """extract_pages_hybrid() with ocr=False and force_ocr=False never touches VLM."""
        pages = [
            {'page': 1, 'text': '', 'is_thin': True},
        ]
        with patch('tomd.pdf.extract_pages', return_value=pages), \
             patch('tomd.pdf.extract_page_via_vlm') as mock_vlm:
            results = extract_pages_hybrid('/fake/doc.pdf', ocr=False, force_ocr=False)

        mock_vlm.assert_not_called()
        self.assertFalse(results[0]['ocr_used'])

    def test_extract_page_via_vlm_cleans_up_temp_file(self):
        """extract_page_via_vlm() must delete the temp PNG file even if generate() raises."""
        import tempfile as tempfile_mod

        fake_image = MagicMock()
        fake_tmp_path = '/tmp/fake_test_page.png'

        fake_tmp_file = MagicMock()
        fake_tmp_file.__enter__ = MagicMock(return_value=fake_tmp_file)
        fake_tmp_file.__exit__ = MagicMock(return_value=False)
        fake_tmp_file.name = fake_tmp_path

        with patch('tomd.pdf.render_page_as_image', return_value=fake_image), \
             patch('tomd.pdf.tempfile') as mock_tempfile, \
             patch('tomd.pdf.os.unlink') as mock_unlink, \
             patch('tomd.pdf.apply_chat_template', return_value='formatted_prompt'), \
             patch('tomd.pdf.generate', return_value='Extracted markdown text') as mock_gen:

            mock_tempfile.NamedTemporaryFile.return_value = fake_tmp_file

            result = extract_page_via_vlm(
                Path('/fake/doc.pdf'),
                page_index=0,
                model=MagicMock(),
                processor=MagicMock(),
                config={},
            )

        mock_unlink.assert_called_once_with(fake_tmp_path)
        self.assertEqual(result, 'Extracted markdown text')

    def test_vlm_model_default_constant(self):
        """VLM_MODEL_DEFAULT should be the expected Qwen2.5 model string."""
        self.assertEqual(VLM_MODEL_DEFAULT, "mlx-community/Qwen2.5-VL-7B-Instruct-4bit")

    def test_thin_page_threshold_constant(self):
        """THIN_PAGE_THRESHOLD should be 50."""
        self.assertEqual(THIN_PAGE_THRESHOLD, 50)


if __name__ == "__main__":
    unittest.main()
