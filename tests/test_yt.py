#!/usr/bin/env python3
"""
Test script for yt2md.py

This script tests the functionality of yt2md.py by:
1. Testing the extract_video_id function with various inputs
2. Testing the output formatters (segments_to_srt, segments_to_markdown, segments_to_text)

Run with: python test_yt2md.py
"""

import os
import sys
import unittest

from tomd.yt import (
    extract_video_id,
    segments_to_srt,
    segments_to_markdown,
    segments_to_text,
    segments_to_markdown_diarized,
    segments_to_srt_diarized,
    segments_to_text_diarized,
    align_speakers,
    build_frontmatter,
    resolve_model,
    MODEL_ALIASES,
)


class FakeAlignedSentence:
    """Minimal stand-in for mlx_audio AlignedSentence for unit testing."""
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


class TestYt2Srt(unittest.TestCase):
    """Test cases for yt2srt.py functions"""

    def test_extract_video_id(self):
        """Test the extract_video_id function with various inputs"""
        # Test with direct video ID
        self.assertEqual(extract_video_id("dQw4w9WgXcQ"), "dQw4w9WgXcQ")

        # Test with standard YouTube URL
        self.assertEqual(
            extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
            "dQw4w9WgXcQ"
        )

        # Test with shortened URL
        self.assertEqual(
            extract_video_id("https://youtu.be/dQw4w9WgXcQ"),
            "dQw4w9WgXcQ"
        )

        # Test with URL containing additional parameters
        self.assertEqual(
            extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s"),
            "dQw4w9WgXcQ"
        )

        # Test with embedded URL
        self.assertEqual(
            extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ"),
            "dQw4w9WgXcQ"
        )

        # Test with invalid input
        with self.assertRaises(ValueError):
            extract_video_id("not-a-youtube-url")

    def test_segments_to_srt_object_format(self):
        """Test segments_to_srt with AlignedSentence-style objects (attribute access)"""
        sentences = [
            FakeAlignedSentence(0.0, 2.5, "Hello, this is a test."),
            FakeAlignedSentence(2.5, 5.0, "Testing the SRT conversion."),
            FakeAlignedSentence(5.0, 10.0, "This should generate a valid SRT file."),
        ]

        expected_srt = (
            "1\n"
            "00:00:00,000 --> 00:00:02,500\n"
            "Hello, this is a test.\n"
            "\n"
            "2\n"
            "00:00:02,500 --> 00:00:05,000\n"
            "Testing the SRT conversion.\n"
            "\n"
            "3\n"
            "00:00:05,000 --> 00:00:10,000\n"
            "This should generate a valid SRT file."
        )

        self.assertEqual(segments_to_srt(sentences), expected_srt)

    def test_segments_to_srt_dict_format(self):
        """Test segments_to_srt with dictionary format data (backward compat)"""
        sentences = [
            {"start": 0.0, "end": 2.5, "text": "Hello, this is a test."},
            {"start": 2.5, "end": 5.0, "text": "Testing the SRT conversion."},
            {"start": 5.0, "end": 10.0, "text": "This should generate a valid SRT file."},
        ]

        expected_srt = (
            "1\n"
            "00:00:00,000 --> 00:00:02,500\n"
            "Hello, this is a test.\n"
            "\n"
            "2\n"
            "00:00:02,500 --> 00:00:05,000\n"
            "Testing the SRT conversion.\n"
            "\n"
            "3\n"
            "00:00:05,000 --> 00:00:10,000\n"
            "This should generate a valid SRT file."
        )

        self.assertEqual(segments_to_srt(sentences), expected_srt)

    def test_segments_to_srt_empty(self):
        """Test segments_to_srt with empty input"""
        self.assertEqual(segments_to_srt([]), "")

    def test_resolve_model_alias(self):
        """Test that model aliases resolve to the correct HuggingFace IDs"""
        self.assertEqual(
            resolve_model("parakeet-v3"),
            "mlx-community/parakeet-tdt-0.6b-v3"
        )
        self.assertEqual(
            resolve_model("parakeet-v2"),
            "mlx-community/parakeet-tdt-0.6b-v2"
        )
        self.assertEqual(
            resolve_model("parakeet-ctc"),
            "mlx-community/parakeet-ctc-0.6b"
        )

    def test_resolve_model_passthrough(self):
        """Test that full model IDs pass through resolve_model unchanged"""
        full_id = "mlx-community/parakeet-tdt-0.6b-v3"
        self.assertEqual(resolve_model(full_id), full_id)

    def test_model_aliases_all_resolve(self):
        """Test that every alias in MODEL_ALIASES resolves to a non-empty string"""
        for alias, expected in MODEL_ALIASES.items():
            self.assertEqual(resolve_model(alias), expected)
            self.assertTrue(expected.startswith("mlx-community/"), f"Expected HF path for alias {alias}")

    def test_segments_to_markdown_with_title(self):
        """Test markdown output with title and timestamps"""
        sentences = [
            FakeAlignedSentence(0.0, 2.5, "Hello, this is a test."),
            FakeAlignedSentence(65.0, 70.0, "A minute later."),
            FakeAlignedSentence(3661.0, 3665.0, "Over an hour in."),
        ]

        result = segments_to_markdown(sentences, title="My Video")
        self.assertIn("# My Video", result)
        self.assertIn("**[00:00]** Hello, this is a test.", result)
        self.assertIn("**[01:05]** A minute later.", result)
        self.assertIn("**[01:01:01]** Over an hour in.", result)

    def test_segments_to_markdown_no_title(self):
        """Test markdown output without title"""
        sentences = [FakeAlignedSentence(0.0, 2.5, "Just text.")]
        result = segments_to_markdown(sentences)
        self.assertNotIn("#", result)
        self.assertIn("**[00:00]** Just text.", result)

    def test_segments_to_text(self):
        """Test plain text output (no timestamps)"""
        sentences = [
            FakeAlignedSentence(0.0, 2.5, "First sentence."),
            FakeAlignedSentence(2.5, 5.0, "Second sentence."),
        ]

        result = segments_to_text(sentences)
        self.assertEqual(result, "First sentence.\n\nSecond sentence.")

    def test_build_frontmatter_basic(self):
        """Test YAML frontmatter generation"""
        metadata = {
            'title': 'Test Video',
            'video_id': 'abc123def45',
            'url': 'https://www.youtube.com/watch?v=abc123def45',
            'channel': 'TestChannel',
            'upload_date': '2024-01-15',
            'duration': 120,
            'tags': ['tag1', 'tag2'],
            'fetched_at': '2024-06-01T12:00:00Z',
        }
        result = build_frontmatter(metadata)
        self.assertTrue(result.startswith("---"))
        self.assertTrue(result.endswith("---"))
        self.assertIn("title: Test Video", result)
        self.assertIn("video_id: abc123def45", result)
        self.assertIn("duration: 120", result)
        self.assertIn("tags: [tag1, tag2]", result)
        self.assertIn("fetched_at:", result)

    def test_build_frontmatter_special_chars(self):
        """Test that titles with special chars are quoted"""
        metadata = {'title': 'What is "AI"? A guide: part 1'}
        result = build_frontmatter(metadata)
        self.assertIn('title: "What is \\"AI\\"? A guide: part 1"', result)

    def test_build_frontmatter_chapters(self):
        """Test chapters formatting"""
        metadata = {
            'title': 'Video',
            'chapters': [
                {'time': '00:00', 'title': 'Intro'},
                {'time': '05:30', 'title': 'Main Topic'},
            ],
        }
        result = build_frontmatter(metadata)
        self.assertIn("chapters:", result)
        self.assertIn('  - time: "00:00"', result)
        self.assertIn("    title: Intro", result)
        self.assertIn('  - time: "05:30"', result)

    def test_build_frontmatter_multiline_description(self):
        """Test multi-line description uses YAML block scalar"""
        metadata = {
            'title': 'Video',
            'description': 'Line one\nLine two\nLine three',
        }
        result = build_frontmatter(metadata)
        self.assertIn("description: |", result)
        self.assertIn("  Line one", result)
        self.assertIn("  Line two", result)

    def test_segments_to_markdown_with_metadata(self):
        """Test markdown output includes frontmatter when metadata provided"""
        sentences = [FakeAlignedSentence(0.0, 2.5, "Hello.")]
        metadata = {
            'title': 'My Video',
            'url': 'https://example.com',
            'fetched_at': '2024-01-01T00:00:00Z',
        }
        result = segments_to_markdown(sentences, title="My Video", metadata=metadata)
        self.assertTrue(result.startswith("---"))
        self.assertIn("url:", result)
        self.assertIn("fetched_at:", result)
        self.assertIn("# My Video", result)
        self.assertIn("**[00:00]** Hello.", result)


class FakeDiarizationSegment:
    """Minimal stand-in for mlx_audio DiarizationSegment."""
    def __init__(self, start, end, speaker):
        self.start = start
        self.end = end
        self.speaker = speaker


class TestDiarization(unittest.TestCase):
    """Test cases for speaker diarization alignment and formatters."""

    def test_align_single_speaker(self):
        """All transcription segments overlap one diarization speaker."""
        trans = [
            FakeAlignedSentence(0.0, 3.0, "Hello there."),
            FakeAlignedSentence(3.0, 6.0, "How are you?"),
        ]
        diar = [FakeDiarizationSegment(0.0, 10.0, 0)]
        result = align_speakers(trans, diar)
        # Should merge into one segment since same speaker
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["speaker"], 0)
        self.assertIn("Hello there.", result[0]["text"])
        self.assertIn("How are you?", result[0]["text"])

    def test_align_two_speakers_alternating(self):
        """Speaker 0 talks 0-5s, speaker 1 talks 5-10s."""
        trans = [
            FakeAlignedSentence(0.0, 4.0, "First speaker here."),
            FakeAlignedSentence(5.0, 9.0, "Second speaker here."),
        ]
        diar = [
            FakeDiarizationSegment(0.0, 5.0, 0),
            FakeDiarizationSegment(5.0, 10.0, 1),
        ]
        result = align_speakers(trans, diar)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["speaker"], 0)
        self.assertEqual(result[1]["speaker"], 1)

    def test_align_overlapping_speakers(self):
        """Diarization has overlapping ranges, majority-overlap wins."""
        trans = [FakeAlignedSentence(2.0, 8.0, "Overlapping test.")]
        diar = [
            FakeDiarizationSegment(0.0, 4.0, 0),   # overlap: 2.0-4.0 = 2s
            FakeDiarizationSegment(3.0, 10.0, 1),   # overlap: 3.0-8.0 = 5s
        ]
        result = align_speakers(trans, diar)
        self.assertEqual(result[0]["speaker"], 1)  # speaker 1 has more overlap

    def test_align_no_diarization_segments(self):
        """Empty diarization → all segments get SPEAKER_0 fallback."""
        trans = [FakeAlignedSentence(0.0, 5.0, "No diarization.")]
        result = align_speakers(trans, [])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["speaker"], 0)

    def test_align_preserves_text(self):
        """Alignment doesn't mutate the original text."""
        trans = [FakeAlignedSentence(0.0, 5.0, "  Preserve me.  ")]
        diar = [FakeDiarizationSegment(0.0, 5.0, 0)]
        result = align_speakers(trans, diar)
        self.assertEqual(result[0]["text"], "Preserve me.")

    def test_segments_to_markdown_diarized_single_speaker(self):
        """One speaker, output has SPEAKER_0 header."""
        segs = [{"start": 0.0, "end": 5.0, "text": "Hello.", "speaker": 0}]
        result = segments_to_markdown_diarized(segs)
        self.assertIn("**SPEAKER_0**", result)
        self.assertIn("Hello.", result)

    def test_segments_to_markdown_diarized_multi_speaker(self):
        """Two speakers, alternating headers."""
        segs = [
            {"start": 0.0, "end": 5.0, "text": "Hi from zero.", "speaker": 0},
            {"start": 5.0, "end": 10.0, "text": "Hi from one.", "speaker": 1},
        ]
        result = segments_to_markdown_diarized(segs, title="Test")
        self.assertIn("**SPEAKER_0**", result)
        self.assertIn("**SPEAKER_1**", result)
        self.assertIn("# Test", result)

    def test_segments_to_markdown_diarized_merges_consecutive(self):
        """Adjacent same-speaker segments should already be merged by align_speakers."""
        # This tests the formatter with pre-merged input
        segs = [
            {"start": 0.0, "end": 10.0, "text": "Long monologue.", "speaker": 0},
        ]
        result = segments_to_markdown_diarized(segs)
        self.assertEqual(result.count("**SPEAKER_0**"), 1)

    def test_segments_to_srt_diarized(self):
        """SRT entries contain [SPEAKER_N] prefix."""
        segs = [
            {"start": 0.0, "end": 5.0, "text": "Hello.", "speaker": 0},
            {"start": 5.0, "end": 10.0, "text": "World.", "speaker": 1},
        ]
        result = segments_to_srt_diarized(segs)
        self.assertIn("[SPEAKER_0] Hello.", result)
        self.assertIn("[SPEAKER_1] World.", result)

    def test_segments_to_text_diarized(self):
        """Plain text with SPEAKER_N: prefix."""
        segs = [
            {"start": 0.0, "end": 5.0, "text": "Hello.", "speaker": 0},
            {"start": 5.0, "end": 10.0, "text": "World.", "speaker": 1},
        ]
        result = segments_to_text_diarized(segs)
        self.assertIn("SPEAKER_0: Hello.", result)
        self.assertIn("SPEAKER_1: World.", result)

    def test_diarize_adds_speakers_to_metadata(self):
        """Metadata dict gets speakers field in frontmatter."""
        metadata = {"title": "Test", "speakers": 2}
        result = build_frontmatter(metadata)
        self.assertIn("speakers: 2", result)


if __name__ == "__main__":
    unittest.main()
