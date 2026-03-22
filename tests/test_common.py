#!/usr/bin/env python3
"""
test_common.py - Unit tests for any2md/common.py

Covers:
- write_json_output: valid JSON to stdout, schema, default=str for Path
- _filter_fields: simple fields, dot-notation, missing fields, multiple fields
- build_frontmatter: existing smoke tests to protect regressions
"""

import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from any2md.common import (
    _filter_fields,
    build_frontmatter,
    write_json_output,
)


# ---------------------------------------------------------------------------
# _filter_fields
# ---------------------------------------------------------------------------

class TestFilterFields(unittest.TestCase):
    """Tests for _filter_fields dot-notation field selection."""

    SAMPLE = {
        "frontmatter": {"rows": 42, "title": "test", "cols": ["a", "b"]},
        "content": "# Hello\nworld",
        "source": "/tmp/foo.csv",
        "converter": "csv",
    }

    def test_single_top_level_field(self):
        result = _filter_fields(self.SAMPLE, "content")
        self.assertEqual(result, {"content": "# Hello\nworld"})

    def test_single_nested_field(self):
        result = _filter_fields(self.SAMPLE, "frontmatter.rows")
        self.assertEqual(result, {"frontmatter": {"rows": 42}})

    def test_multiple_fields(self):
        result = _filter_fields(self.SAMPLE, "frontmatter.rows,content")
        self.assertEqual(result, {
            "frontmatter": {"rows": 42},
            "content": "# Hello\nworld",
        })

    def test_missing_top_level_field_returns_none(self):
        result = _filter_fields(self.SAMPLE, "nonexistent")
        self.assertIn("nonexistent", result)
        self.assertIsNone(result["nonexistent"])

    def test_missing_nested_field_returns_none(self):
        result = _filter_fields(self.SAMPLE, "frontmatter.missing_key")
        self.assertIsNone(result["frontmatter"]["missing_key"])

    def test_multiple_nested_fields_merged_under_parent(self):
        result = _filter_fields(self.SAMPLE, "frontmatter.rows,frontmatter.title")
        self.assertIn("frontmatter", result)
        self.assertEqual(result["frontmatter"]["rows"], 42)
        self.assertEqual(result["frontmatter"]["title"], "test")

    def test_fields_with_whitespace_stripped(self):
        result = _filter_fields(self.SAMPLE, "content , converter")
        self.assertIn("content", result)
        self.assertIn("converter", result)
        self.assertEqual(result["converter"], "csv")

    def test_all_top_level_fields(self):
        result = _filter_fields(self.SAMPLE, "frontmatter,content,source,converter")
        self.assertEqual(result["source"], "/tmp/foo.csv")
        self.assertEqual(result["converter"], "csv")

    def test_list_value_preserved(self):
        result = _filter_fields(self.SAMPLE, "frontmatter.cols")
        self.assertEqual(result["frontmatter"]["cols"], ["a", "b"])


# ---------------------------------------------------------------------------
# write_json_output
# ---------------------------------------------------------------------------

class TestWriteJsonOutput(unittest.TestCase):
    """Tests for write_json_output writing structured JSON to stdout."""

    METADATA = {"title": "Test", "rows": 3, "cols": ["a", "b"]}
    CONTENT = "| a | b |\n|---|---|\n| 1 | 2 |"
    SOURCE = "/tmp/test.csv"
    CONVERTER = "csv"

    def _capture(self, **kwargs):
        """Call write_json_output and return parsed JSON from captured stdout."""
        buf = io.StringIO()
        with patch("any2md.common.sys") as mock_sys:
            mock_sys.stdout = buf
            write_json_output(
                self.METADATA,
                self.CONTENT,
                self.SOURCE,
                self.CONVERTER,
                **kwargs,
            )
        buf.seek(0)
        return json.loads(buf.read())

    def test_output_is_valid_json(self):
        """write_json_output produces parseable JSON."""
        buf = io.StringIO()
        with patch("any2md.common.sys") as mock_sys:
            mock_sys.stdout = buf
            write_json_output(self.METADATA, self.CONTENT, self.SOURCE, self.CONVERTER)
        buf.seek(0)
        raw = buf.read()
        # Must not raise
        parsed = json.loads(raw)
        self.assertIsInstance(parsed, dict)

    def test_output_ends_with_newline(self):
        buf = io.StringIO()
        with patch("any2md.common.sys") as mock_sys:
            mock_sys.stdout = buf
            write_json_output(self.METADATA, self.CONTENT, self.SOURCE, self.CONVERTER)
        self.assertTrue(buf.getvalue().endswith("\n"))

    def test_schema_top_level_keys(self):
        result = self._capture()
        self.assertIn("frontmatter", result)
        self.assertIn("content", result)
        self.assertIn("source", result)
        self.assertIn("converter", result)

    def test_frontmatter_matches_metadata(self):
        result = self._capture()
        self.assertEqual(result["frontmatter"], self.METADATA)

    def test_content_matches(self):
        result = self._capture()
        self.assertEqual(result["content"], self.CONTENT)

    def test_source_as_string(self):
        result = self._capture()
        self.assertEqual(result["source"], self.SOURCE)
        self.assertIsInstance(result["source"], str)

    def test_converter_name(self):
        result = self._capture()
        self.assertEqual(result["converter"], self.CONVERTER)

    def test_path_object_serialized_as_string(self):
        """Path objects must serialize to string via default=str."""
        buf = io.StringIO()
        path_source = Path("/tmp/test.csv")
        with patch("any2md.common.sys") as mock_sys:
            mock_sys.stdout = buf
            write_json_output(self.METADATA, self.CONTENT, path_source, self.CONVERTER)
        buf.seek(0)
        result = json.loads(buf.read())
        self.assertIsInstance(result["source"], str)
        self.assertEqual(result["source"], "/tmp/test.csv")

    def test_fields_filters_output(self):
        result = self._capture(fields="content")
        self.assertIn("content", result)
        self.assertNotIn("frontmatter", result)
        self.assertNotIn("source", result)

    def test_fields_dot_notation(self):
        result = self._capture(fields="frontmatter.rows,content")
        self.assertIn("frontmatter", result)
        self.assertEqual(result["frontmatter"]["rows"], 3)
        self.assertNotIn("title", result.get("frontmatter", {}))
        self.assertIn("content", result)

    def test_no_fields_returns_all(self):
        """fields=None returns all four top-level keys."""
        result = self._capture(fields=None)
        self.assertEqual(set(result.keys()), {"frontmatter", "content", "source", "converter"})

    def test_empty_metadata(self):
        buf = io.StringIO()
        with patch("any2md.common.sys") as mock_sys:
            mock_sys.stdout = buf
            write_json_output({}, "", "/dev/null", "test")
        buf.seek(0)
        result = json.loads(buf.read())
        self.assertEqual(result["frontmatter"], {})
        self.assertEqual(result["content"], "")

    def test_unicode_content(self):
        unicode_content = "# 日本語\nNihongo: \u3053\u3093\u306b\u3061\u306f"
        result = self._capture()
        # Override with unicode
        buf = io.StringIO()
        with patch("any2md.common.sys") as mock_sys:
            mock_sys.stdout = buf
            write_json_output(self.METADATA, unicode_content, self.SOURCE, self.CONVERTER)
        buf.seek(0)
        result = json.loads(buf.read())
        self.assertEqual(result["content"], unicode_content)


# ---------------------------------------------------------------------------
# build_frontmatter (regression smoke tests)
# ---------------------------------------------------------------------------

class TestBuildFrontmatterRegression(unittest.TestCase):
    """Smoke tests to guard against regressions in existing build_frontmatter."""

    def test_starts_and_ends_with_delimiters(self):
        result = build_frontmatter({"title": "Hello"})
        lines = result.splitlines()
        self.assertEqual(lines[0], "---")
        self.assertEqual(lines[-1], "---")

    def test_simple_scalar(self):
        result = build_frontmatter({"title": "My Title"})
        self.assertIn("title: My Title", result)

    def test_integer_scalar(self):
        result = build_frontmatter({"rows": 42})
        self.assertIn("rows: 42", result)

    def test_list_field(self):
        result = build_frontmatter({"tags": ["a", "b"]})
        self.assertIn("tags:", result)

    def test_none_value_omitted(self):
        result = build_frontmatter({"title": "X", "empty": None})
        self.assertNotIn("empty:", result)

    def test_empty_string_omitted(self):
        result = build_frontmatter({"title": "X", "blank": ""})
        self.assertNotIn("blank:", result)

    def test_empty_list_omitted(self):
        result = build_frontmatter({"title": "X", "nada": []})
        self.assertNotIn("nada:", result)

    def test_description_block_scalar(self):
        result = build_frontmatter({"description": "line1\nline2"})
        self.assertIn("description: |", result)
        self.assertIn("  line1", result)
        self.assertIn("  line2", result)


if __name__ == "__main__":
    unittest.main()
