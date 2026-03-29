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

from any2md.yt import (
    extract_video_id,
    segments_to_srt,
    segments_to_markdown,
    segments_to_text,
    segments_to_markdown_diarized,
    segments_to_srt_diarized,
    segments_to_text_diarized,
    align_speakers,
    _merge_diarization_segments,
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


class TestMergeDiarizationSegments(unittest.TestCase):
    """Test cases for _merge_diarization_segments()."""

    def test_empty_input(self):
        """Empty list returns empty list."""
        self.assertEqual(_merge_diarization_segments([]), [])

    def test_single_segment(self):
        """Single segment passes through unchanged."""
        segs = [FakeDiarizationSegment(0.0, 5.0, 0)]
        result = _merge_diarization_segments(segs)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].start, 0.0)
        self.assertAlmostEqual(result[0].end, 5.0)
        self.assertEqual(result[0].speaker, 0)

    def test_merges_same_speaker_within_gap(self):
        """Adjacent segments from same speaker with small gap get merged."""
        segs = [
            FakeDiarizationSegment(0.0, 5.0, 0),
            FakeDiarizationSegment(5.1, 10.0, 0),  # 0.1s gap < default 0.3s
        ]
        result = _merge_diarization_segments(segs)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].start, 0.0)
        self.assertAlmostEqual(result[0].end, 10.0)
        self.assertEqual(result[0].speaker, 0)

    def test_no_merge_different_speakers(self):
        """Adjacent segments from different speakers stay separate."""
        segs = [
            FakeDiarizationSegment(0.0, 5.0, 0),
            FakeDiarizationSegment(5.0, 10.0, 1),
        ]
        result = _merge_diarization_segments(segs)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].speaker, 0)
        self.assertEqual(result[1].speaker, 1)

    def test_no_merge_same_speaker_large_gap(self):
        """Same speaker but gap > merge_gap stays separate."""
        segs = [
            FakeDiarizationSegment(0.0, 5.0, 0),
            FakeDiarizationSegment(6.0, 10.0, 0),  # 1.0s gap > default 0.3s
        ]
        result = _merge_diarization_segments(segs)
        self.assertEqual(len(result), 2)

    def test_sorts_by_start_time(self):
        """Out-of-order segments get sorted before merging."""
        segs = [
            FakeDiarizationSegment(5.0, 10.0, 1),
            FakeDiarizationSegment(0.0, 5.0, 0),
        ]
        result = _merge_diarization_segments(segs)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0].start, 0.0)
        self.assertEqual(result[0].speaker, 0)
        self.assertAlmostEqual(result[1].start, 5.0)
        self.assertEqual(result[1].speaker, 1)

    def test_custom_merge_gap(self):
        """Custom merge_gap threshold is respected."""
        segs = [
            FakeDiarizationSegment(0.0, 5.0, 0),
            FakeDiarizationSegment(5.5, 10.0, 0),  # 0.5s gap
        ]
        # Default 0.3s — should NOT merge
        result = _merge_diarization_segments(segs, merge_gap=0.3)
        self.assertEqual(len(result), 2)
        # Custom 1.0s — should merge
        result = _merge_diarization_segments(segs, merge_gap=1.0)
        self.assertEqual(len(result), 1)

    def test_multi_chunk_boundary_merge(self):
        """Simulates chunk boundary: speaker continues across chunks."""
        # Chunk 1 ends at 10s, chunk 2 starts at 10s
        segs = [
            FakeDiarizationSegment(0.0, 5.0, 0),
            FakeDiarizationSegment(5.0, 10.0, 1),
            FakeDiarizationSegment(10.0, 15.0, 1),  # same speaker continues
            FakeDiarizationSegment(15.0, 20.0, 0),
        ]
        result = _merge_diarization_segments(segs)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].speaker, 0)
        self.assertAlmostEqual(result[0].end, 5.0)
        self.assertEqual(result[1].speaker, 1)
        self.assertAlmostEqual(result[1].start, 5.0)
        self.assertAlmostEqual(result[1].end, 15.0)  # merged
        self.assertEqual(result[2].speaker, 0)

    def test_diarize_uses_streaming(self):
        """diarize() calls generate_stream, not generate."""
        from unittest.mock import MagicMock

        from any2md.yt import diarize

        mock_model = MagicMock()
        # Simulate generate_stream yielding two chunk results
        chunk1 = MagicMock()
        chunk1.segments = [FakeDiarizationSegment(0.0, 5.0, 0)]
        chunk2 = MagicMock()
        chunk2.segments = [FakeDiarizationSegment(5.0, 10.0, 1)]
        mock_model.generate_stream.return_value = iter([chunk1, chunk2])

        result = diarize("/fake/audio.wav", mock_model)

        # Verify generate_stream was called (not generate)
        mock_model.generate_stream.assert_called_once()
        mock_model.generate.assert_not_called()
        # Verify result has merged segments from both chunks
        self.assertEqual(result.num_speakers, 2)
        self.assertEqual(len(result.segments), 2)


# ---------------------------------------------------------------------------
# Tests for segments_to_markdown_diarized with speaker_map
# ---------------------------------------------------------------------------


class TestSegmentsToMarkdownDiarizedWithSpeakerMap(unittest.TestCase):
    """Test segments_to_markdown_diarized() with identification speaker_map."""

    def _make_segs(self):
        return [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_0", "text": "Hello everyone."},
            {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_1", "text": "Hi there."},
            {"start": 10.0, "end": 15.0, "speaker": "SPEAKER_2", "text": "Good morning."},
        ]

    def test_no_speaker_map_uses_legacy_format(self):
        result = segments_to_markdown_diarized(self._make_segs())
        self.assertIn("**SPEAKER_SPEAKER_0**", result)
        self.assertIn("**SPEAKER_SPEAKER_1**", result)

    def test_matched_high_conf_shows_name_without_score(self):
        speaker_map = {
            "SPEAKER_0": {"name": "Alice", "matched": True, "distance": 0.10, "high_conf": True},
        }
        result = segments_to_markdown_diarized(self._make_segs(), speaker_map=speaker_map)
        self.assertIn("**Alice**", result)
        # Should NOT show a confidence score for high-conf
        self.assertNotIn("(0.", result.split("**Alice**")[1].split("\n")[0])

    def test_matched_medium_conf_shows_score(self):
        speaker_map = {
            "SPEAKER_0": {"name": "Joe", "matched": True, "distance": 0.25, "high_conf": False},
        }
        result = segments_to_markdown_diarized(self._make_segs(), speaker_map=speaker_map)
        self.assertIn("**Joe**", result)
        # First line with Joe should contain confidence score
        joe_line = [l for l in result.split("\n") if "**Joe**" in l][0]
        self.assertIn("(0.75)", joe_line)

    def test_unmatched_speaker_keeps_original_label(self):
        speaker_map = {
            "SPEAKER_0": {"name": "SPEAKER_0", "matched": False, "distance": None, "high_conf": False},
        }
        result = segments_to_markdown_diarized(self._make_segs(), speaker_map=speaker_map)
        self.assertIn("**SPEAKER_0**", result)

    def test_partial_map_unmatched_label_preserved(self):
        """SPEAKER_2 has no entry in map — should fall back to SPEAKER_N format."""
        speaker_map = {
            "SPEAKER_0": {"name": "Alice", "matched": True, "distance": 0.05, "high_conf": True},
            "SPEAKER_1": {"name": "Bob", "matched": True, "distance": 0.08, "high_conf": True},
            # SPEAKER_2 intentionally absent
        }
        result = segments_to_markdown_diarized(self._make_segs(), speaker_map=speaker_map)
        self.assertIn("**Alice**", result)
        self.assertIn("**Bob**", result)
        self.assertIn("**SPEAKER_SPEAKER_2**", result)

    def test_frontmatter_included_when_metadata_provided(self):
        metadata = {"title": "Test Meeting", "speakers": 2}
        result = segments_to_markdown_diarized(self._make_segs(), metadata=metadata)
        self.assertIn("title:", result)
        self.assertIn("speakers:", result)


# ---------------------------------------------------------------------------
# Tests for segments_to_srt_diarized and segments_to_text_diarized with speaker_map
# ---------------------------------------------------------------------------


class TestDiarizedFormattersWithSpeakerMap(unittest.TestCase):

    def _make_segs(self):
        return [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_0", "text": "Hello."},
            {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_1", "text": "Hi."},
        ]

    def test_srt_shows_name_when_matched(self):
        speaker_map = {
            "SPEAKER_0": {"name": "Alice", "matched": True, "distance": 0.05, "high_conf": True},
        }
        result = segments_to_srt_diarized(self._make_segs(), speaker_map=speaker_map)
        self.assertIn("[Alice]", result)

    def test_srt_keeps_speaker_n_when_unmatched(self):
        speaker_map = {
            "SPEAKER_0": {"name": "SPEAKER_0", "matched": False, "distance": None, "high_conf": False},
        }
        result = segments_to_srt_diarized(self._make_segs(), speaker_map=speaker_map)
        self.assertIn("[SPEAKER_SPEAKER_0]", result)

    def test_srt_no_map_uses_legacy_format(self):
        result = segments_to_srt_diarized(self._make_segs())
        self.assertIn("[SPEAKER_SPEAKER_0]", result)
        self.assertIn("[SPEAKER_SPEAKER_1]", result)

    def test_text_shows_name_when_matched(self):
        speaker_map = {
            "SPEAKER_1": {"name": "Bob", "matched": True, "distance": 0.07, "high_conf": True},
        }
        result = segments_to_text_diarized(self._make_segs(), speaker_map=speaker_map)
        self.assertIn("Bob:", result)

    def test_text_keeps_speaker_n_when_unmatched(self):
        speaker_map = {
            "SPEAKER_1": {"name": "SPEAKER_1", "matched": False, "distance": None, "high_conf": False},
        }
        result = segments_to_text_diarized(self._make_segs(), speaker_map=speaker_map)
        self.assertIn("SPEAKER_SPEAKER_1:", result)

    def test_text_no_map_uses_legacy_format(self):
        result = segments_to_text_diarized(self._make_segs())
        self.assertIn("SPEAKER_SPEAKER_0:", result)
        self.assertIn("SPEAKER_SPEAKER_1:", result)


# ---------------------------------------------------------------------------
# Tests for enrollment prompt logic in transcribe()
# ---------------------------------------------------------------------------


import numpy as _np
import tempfile


def _make_rand_emb(seed: int = 0) -> _np.ndarray:
    """Return a random L2-normalized 256-d float32 embedding for tests."""
    rng = _np.random.default_rng(seed)
    v = rng.standard_normal(256).astype(_np.float32)
    return v / _np.linalg.norm(v)


def _make_fake_transcribe_result():
    """Return a MagicMock that mimics mlx-audio transcribe result."""
    from unittest.mock import MagicMock
    result = MagicMock()
    result.sentences = [FakeAlignedSentence(0.0, 5.0, "hello world")]
    return result


def _make_speaker_map_with_unmatched(label: str = "SPEAKER_0", emb_seed: int = 42):
    """Build a speaker_map dict where label is unmatched (like identify_speakers() returns)."""
    emb = _make_rand_emb(emb_seed)
    return {
        label: {
            "name": label,
            "matched": False,
            "distance": None,
            "high_conf": False,
            "avg_embedding": emb,
            "segments": [{"start": 0.0, "end": 3.0}],
        }
    }


class TestTranscribeEnrollmentLogic(unittest.TestCase):
    """Test enrollment prompt/auto-enroll logic in transcribe().

    All heavy deps (mlx-audio, speaker catalog) are mocked so tests run
    without any models installed.
    """

    def _make_fake_mlx_model(self):
        """Return a mock mlx-audio model."""
        from unittest.mock import MagicMock
        model = MagicMock()
        model.generate.return_value = _make_fake_transcribe_result()
        return model

    def _patch_load(self, mock_model):
        """Return a patcher for mlx_audio.stt.load."""
        from unittest.mock import patch
        return patch("mlx_audio.stt.load", return_value=mock_model)

    def _run_transcribe(self, tmpdir, speaker_map, auto_enroll=False,
                        no_enroll=False, is_tty=False, prompt_return="",
                        _unmatched_out=None, is_json_mode_val=False):
        """Helper to run transcribe() with all deps mocked.

        Returns the path of the written output file.
        """
        from unittest.mock import patch, MagicMock
        from any2md.yt import transcribe

        import io

        audio_path = tmpdir + "/test.wav"
        open(audio_path, "w").close()

        fake_model = self._make_fake_mlx_model()

        mock_add_speaker = MagicMock(return_value="fake-speaker-id")
        mock_enroll = MagicMock(return_value="fake-enrollment-id")
        mock_open_catalog = MagicMock(return_value=MagicMock())
        mock_close_catalog = MagicMock()
        mock_next_unknown = MagicMock(side_effect=["Unknown_0", "Unknown_1", "Unknown_2"])
        mock_identify = MagicMock(return_value=speaker_map)

        # Patch all the speaker module symbols imported inside transcribe()
        speaker_patches = {
            "any2md.yt.identify_speakers": mock_identify,
            "any2md.yt.open_catalog": mock_open_catalog,
            "any2md.yt.close_catalog": mock_close_catalog,
            "any2md.yt.add_speaker": mock_add_speaker,
            "any2md.yt.enroll": mock_enroll,
            "any2md.yt._next_unknown_name": mock_next_unknown,
        }

        with self._patch_load(fake_model), \
             patch("any2md.yt.is_json_mode", return_value=is_json_mode_val), \
             patch("sys.stdout.isatty", return_value=is_tty), \
             patch("typer.prompt", return_value=prompt_return), \
             patch("any2md.yt.identify_speakers", mock_identify), \
             patch("any2md.yt.open_catalog", mock_open_catalog), \
             patch("any2md.yt.close_catalog", mock_close_catalog), \
             patch("any2md.yt.add_speaker", mock_add_speaker), \
             patch("any2md.yt.enroll", mock_enroll), \
             patch("any2md.yt._next_unknown_name", mock_next_unknown):
            result = transcribe(
                audio_path,
                output_dir=tmpdir,
                diarize_model_name=None,  # no diarization in these tests
                identify=False,  # identify is handled via mocked speaker_map
                auto_enroll=auto_enroll,
                no_enroll=no_enroll,
                _unmatched_out=_unmatched_out,
            )

        return result, mock_add_speaker, mock_enroll, mock_next_unknown

    def _apply_speaker_patches(self, stack, speaker_map, mock_add_speaker=None,
                                mock_enroll=None, mock_next_unknown=None,
                                mock_open_catalog=None, mock_close_catalog=None):
        """Enter all speaker module patches into an ExitStack.

        Since these symbols are imported inside transcribe() with
        'from any2md.speaker import ...', patching must happen at the
        any2md.speaker module level.

        Returns (mock_add_speaker, mock_enroll, mock_next_unknown).
        """
        from unittest.mock import patch, MagicMock

        if mock_add_speaker is None:
            mock_add_speaker = MagicMock(return_value="fake-sp-id")
        if mock_enroll is None:
            mock_enroll = MagicMock(return_value="fake-enr-id")
        if mock_next_unknown is None:
            mock_next_unknown = MagicMock(return_value="Unknown_0")
        if mock_open_catalog is None:
            mock_open_catalog = MagicMock(return_value=MagicMock())
        if mock_close_catalog is None:
            mock_close_catalog = MagicMock()

        stack.enter_context(patch("any2md.speaker.identify_speakers", return_value=speaker_map))
        stack.enter_context(patch("any2md.speaker.open_catalog", mock_open_catalog))
        stack.enter_context(patch("any2md.speaker.close_catalog", mock_close_catalog))
        stack.enter_context(patch("any2md.speaker.add_speaker", mock_add_speaker))
        stack.enter_context(patch("any2md.speaker.enroll", mock_enroll))
        stack.enter_context(patch("any2md.speaker._next_unknown_name", mock_next_unknown))
        return mock_add_speaker, mock_enroll, mock_next_unknown

    def _apply_diarize_patches(self, stack, diarized_segments=None):
        """Mock diarization model loading and diarize() to avoid network/model calls."""
        from unittest.mock import patch, MagicMock

        if diarized_segments is None:
            diarized_segments = [
                {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_0", "text": "hello"},
            ]

        mock_diar_model = MagicMock()
        mock_diar_output = MagicMock()
        mock_diar_output.num_speakers = 1
        mock_diar_output.segments = []

        stack.enter_context(patch("any2md.yt.load_diarization_model", return_value=mock_diar_model))
        stack.enter_context(patch("any2md.yt.diarize", return_value=mock_diar_output))
        stack.enter_context(patch("any2md.yt.align_speakers", return_value=diarized_segments))

        # Also mock speaker embedding extraction (load_speaker_model, extract_embeddings_for_segments)
        mock_spk_model = MagicMock()
        stack.enter_context(patch("any2md.speaker.load_speaker_model", return_value=mock_spk_model))
        stack.enter_context(
            patch("any2md.speaker.extract_embeddings_for_segments", side_effect=lambda m, p, s: s)
        )
        return diarized_segments

    def test_no_enroll_skips_prompts(self):
        """--no-enroll: unmatched speakers are left as SPEAKER_N, no prompts shown."""
        import contextlib
        from unittest.mock import patch, MagicMock
        from any2md.yt import transcribe

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = tmpdir + "/test.wav"
            open(audio_path, "w").close()

            fake_model = self._make_fake_mlx_model()
            mock_add_speaker = MagicMock(return_value="fake-id")
            mock_enroll = MagicMock()
            mock_prompt = MagicMock()

            diarized = [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_0", "text": "hi",
                         "embedding": _make_rand_emb(1)}]
            speaker_map = _make_speaker_map_with_unmatched("SPEAKER_0")

            with contextlib.ExitStack() as stack:
                stack.enter_context(self._patch_load(fake_model))
                self._apply_diarize_patches(stack, diarized_segments=diarized)
                self._apply_speaker_patches(
                    stack, speaker_map,
                    mock_add_speaker=mock_add_speaker, mock_enroll=mock_enroll
                )
                stack.enter_context(patch("typer.prompt", mock_prompt))
                stack.enter_context(patch("sys.stdout.isatty", return_value=True))
                transcribe(
                    audio_path,
                    output_dir=tmpdir,
                    diarize_model_name="fake-model",
                    identify=True,
                    no_enroll=True,
                )

            # no_enroll=True: no prompts, no enrollment
            mock_prompt.assert_not_called()
            mock_add_speaker.assert_not_called()
            mock_enroll.assert_not_called()

    def test_auto_enroll_enrolls_unmatched_as_unknown_n(self):
        """--auto-enroll: unmatched speakers are enrolled as Unknown_N without prompting."""
        import contextlib
        from unittest.mock import patch, MagicMock
        from any2md.yt import transcribe

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = tmpdir + "/test.wav"
            open(audio_path, "w").close()

            fake_model = self._make_fake_mlx_model()
            mock_add_speaker = MagicMock(return_value="fake-id")
            mock_enroll = MagicMock(return_value="fake-enrollment")
            mock_prompt = MagicMock()
            mock_next_unknown = MagicMock(return_value="Unknown_0")

            diarized = [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_0", "text": "hi",
                         "embedding": _make_rand_emb(2)}]
            speaker_map = _make_speaker_map_with_unmatched("SPEAKER_0")

            with contextlib.ExitStack() as stack:
                stack.enter_context(self._patch_load(fake_model))
                self._apply_diarize_patches(stack, diarized_segments=diarized)
                self._apply_speaker_patches(
                    stack, speaker_map,
                    mock_add_speaker=mock_add_speaker,
                    mock_enroll=mock_enroll,
                    mock_next_unknown=mock_next_unknown,
                )
                stack.enter_context(patch("typer.prompt", mock_prompt))
                stack.enter_context(patch("sys.stdout.isatty", return_value=False))
                transcribe(
                    audio_path,
                    output_dir=tmpdir,
                    diarize_model_name="fake-model",
                    identify=True,
                    auto_enroll=True,
                )

            # auto_enroll=True: no prompts, but enrollment happens
            mock_prompt.assert_not_called()
            mock_next_unknown.assert_called_once()
            mock_add_speaker.assert_called_once()
            mock_enroll.assert_called_once()

    def test_auto_enroll_updates_speaker_map_name(self):
        """--auto-enroll: speaker_map entry name is updated to Unknown_N after enrollment."""
        import contextlib
        from unittest.mock import patch, MagicMock
        from any2md.yt import transcribe

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = tmpdir + "/test.wav"
            open(audio_path, "w").close()

            fake_model = self._make_fake_mlx_model()
            diarized = [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_0", "text": "hi",
                         "embedding": _make_rand_emb(3)}]
            speaker_map = _make_speaker_map_with_unmatched("SPEAKER_0")

            with contextlib.ExitStack() as stack:
                stack.enter_context(self._patch_load(fake_model))
                self._apply_diarize_patches(stack, diarized_segments=diarized)
                self._apply_speaker_patches(
                    stack, speaker_map,
                    mock_add_speaker=MagicMock(return_value="sp-id"),
                    mock_enroll=MagicMock(return_value="enr-id"),
                    mock_next_unknown=MagicMock(return_value="Unknown_0"),
                )
                stack.enter_context(patch("typer.prompt"))
                stack.enter_context(patch("sys.stdout.isatty", return_value=False))
                transcribe(
                    audio_path,
                    output_dir=tmpdir,
                    diarize_model_name="fake-model",
                    identify=True,
                    auto_enroll=True,
                )

            # speaker_map entry should be updated in-place
            self.assertEqual(speaker_map["SPEAKER_0"]["name"], "Unknown_0")
            self.assertTrue(speaker_map["SPEAKER_0"]["matched"])

    def test_interactive_prompt_with_name_enrolls(self):
        """TTY mode with name entered: speaker is enrolled and map updated."""
        import contextlib
        from unittest.mock import patch, MagicMock
        from any2md.yt import transcribe

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = tmpdir + "/test.wav"
            open(audio_path, "w").close()

            fake_model = self._make_fake_mlx_model()
            diarized = [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_0", "text": "hi",
                         "embedding": _make_rand_emb(4)}]
            speaker_map = _make_speaker_map_with_unmatched("SPEAKER_0")
            mock_add_speaker = MagicMock(return_value="sp-id")
            mock_enroll = MagicMock(return_value="enr-id")

            with contextlib.ExitStack() as stack:
                stack.enter_context(self._patch_load(fake_model))
                self._apply_diarize_patches(stack, diarized_segments=diarized)
                self._apply_speaker_patches(
                    stack, speaker_map,
                    mock_add_speaker=mock_add_speaker, mock_enroll=mock_enroll
                )
                stack.enter_context(patch("typer.prompt", return_value="Alice"))
                stack.enter_context(patch("any2md.yt.is_json_mode", return_value=False))
                stack.enter_context(patch("sys.stdout.isatty", return_value=True))
                transcribe(
                    audio_path,
                    output_dir=tmpdir,
                    diarize_model_name="fake-model",
                    identify=True,
                )

            mock_add_speaker.assert_called_once()
            call_args = mock_add_speaker.call_args
            # Second argument to add_speaker should be "Alice" (conn is first)
            self.assertEqual(call_args[0][1], "Alice")
            mock_enroll.assert_called_once()
            self.assertEqual(speaker_map["SPEAKER_0"]["name"], "Alice")
            self.assertTrue(speaker_map["SPEAKER_0"]["matched"])

    def test_interactive_prompt_enter_skip_does_not_enroll(self):
        """TTY mode with Enter pressed (empty): speaker is NOT enrolled."""
        import contextlib
        from unittest.mock import patch, MagicMock
        from any2md.yt import transcribe

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = tmpdir + "/test.wav"
            open(audio_path, "w").close()

            fake_model = self._make_fake_mlx_model()
            diarized = [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_0", "text": "hi",
                         "embedding": _make_rand_emb(5)}]
            speaker_map = _make_speaker_map_with_unmatched("SPEAKER_0")
            mock_add_speaker = MagicMock(return_value="sp-id")
            mock_enroll = MagicMock()

            with contextlib.ExitStack() as stack:
                stack.enter_context(self._patch_load(fake_model))
                self._apply_diarize_patches(stack, diarized_segments=diarized)
                self._apply_speaker_patches(
                    stack, speaker_map,
                    mock_add_speaker=mock_add_speaker, mock_enroll=mock_enroll
                )
                stack.enter_context(patch("typer.prompt", return_value=""))
                stack.enter_context(patch("any2md.yt.is_json_mode", return_value=False))
                stack.enter_context(patch("sys.stdout.isatty", return_value=True))
                transcribe(
                    audio_path,
                    output_dir=tmpdir,
                    diarize_model_name="fake-model",
                    identify=True,
                )

            mock_add_speaker.assert_not_called()
            mock_enroll.assert_not_called()
            # Name stays as original label
            self.assertEqual(speaker_map["SPEAKER_0"]["name"], "SPEAKER_0")
            self.assertFalse(speaker_map["SPEAKER_0"]["matched"])

    def test_non_tty_no_prompt_no_enroll(self):
        """Non-TTY, no auto_enroll, no --json: silently leave SPEAKER_N, no prompt."""
        import contextlib
        from unittest.mock import patch, MagicMock
        from any2md.yt import transcribe

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = tmpdir + "/test.wav"
            open(audio_path, "w").close()

            fake_model = self._make_fake_mlx_model()
            diarized = [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_0", "text": "hi",
                         "embedding": _make_rand_emb(6)}]
            speaker_map = _make_speaker_map_with_unmatched("SPEAKER_0")
            mock_add_speaker = MagicMock()
            mock_enroll = MagicMock()
            mock_prompt = MagicMock()

            with contextlib.ExitStack() as stack:
                stack.enter_context(self._patch_load(fake_model))
                self._apply_diarize_patches(stack, diarized_segments=diarized)
                self._apply_speaker_patches(
                    stack, speaker_map,
                    mock_add_speaker=mock_add_speaker, mock_enroll=mock_enroll
                )
                stack.enter_context(patch("typer.prompt", mock_prompt))
                stack.enter_context(patch("any2md.yt.is_json_mode", return_value=False))
                stack.enter_context(patch("sys.stdout.isatty", return_value=False))
                transcribe(
                    audio_path,
                    output_dir=tmpdir,
                    diarize_model_name="fake-model",
                    identify=True,
                )

            # No TTY, no auto_enroll, not json: silent passthrough
            mock_prompt.assert_not_called()
            mock_add_speaker.assert_not_called()
            mock_enroll.assert_not_called()
            self.assertFalse(speaker_map["SPEAKER_0"]["matched"])

    def test_json_mode_unmatched_out_populated(self):
        """JSON mode: unmatched_out list is populated with label/embedding/segments."""
        import contextlib
        from unittest.mock import patch, MagicMock
        from any2md.yt import transcribe

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = tmpdir + "/test.wav"
            open(audio_path, "w").close()

            fake_model = self._make_fake_mlx_model()
            emb = _make_rand_emb(200)
            diarized = [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_0", "text": "hi",
                         "embedding": emb}]
            speaker_map = {
                "SPEAKER_0": {
                    "name": "SPEAKER_0",
                    "matched": False,
                    "distance": None,
                    "high_conf": False,
                    "avg_embedding": emb,
                    "segments": [{"start": 0.0, "end": 3.0}],
                }
            }
            unmatched_out: list = []

            with contextlib.ExitStack() as stack:
                stack.enter_context(self._patch_load(fake_model))
                self._apply_diarize_patches(stack, diarized_segments=diarized)
                self._apply_speaker_patches(stack, speaker_map)
                stack.enter_context(patch("typer.prompt", return_value=""))
                stack.enter_context(patch("any2md.yt.is_json_mode", return_value=True))
                stack.enter_context(patch("sys.stdout.isatty", return_value=False))
                transcribe(
                    audio_path,
                    output_dir=tmpdir,
                    diarize_model_name="fake-model",
                    identify=True,
                    _unmatched_out=unmatched_out,
                )

            # unmatched_out should have one entry for SPEAKER_0
            self.assertEqual(len(unmatched_out), 1)
            entry = unmatched_out[0]
            self.assertEqual(entry["label"], "SPEAKER_0")
            self.assertIsNotNone(entry["embedding"])
            self.assertIsInstance(entry["embedding"], list)
            self.assertEqual(len(entry["embedding"]), 256)
            self.assertEqual(entry["segments"], [{"start": 0.0, "end": 3.0}])

    def test_auto_enroll_skips_speaker_with_no_embedding(self):
        """auto_enroll skips speakers with avg_embedding=None (no segments had embeddings)."""
        import contextlib
        from unittest.mock import patch, MagicMock
        from any2md.yt import transcribe

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = tmpdir + "/test.wav"
            open(audio_path, "w").close()

            fake_model = self._make_fake_mlx_model()
            diarized = [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_0", "text": "hi"}]
            speaker_map = {
                "SPEAKER_0": {
                    "name": "SPEAKER_0",
                    "matched": False,
                    "distance": None,
                    "high_conf": False,
                    "avg_embedding": None,  # No embedding available
                    "segments": [],
                }
            }
            mock_add_speaker = MagicMock(return_value="sp-id")
            mock_enroll = MagicMock()

            with contextlib.ExitStack() as stack:
                stack.enter_context(self._patch_load(fake_model))
                self._apply_diarize_patches(stack, diarized_segments=diarized)
                self._apply_speaker_patches(
                    stack, speaker_map,
                    mock_add_speaker=mock_add_speaker, mock_enroll=mock_enroll
                )
                stack.enter_context(patch("typer.prompt"))
                stack.enter_context(patch("any2md.yt.is_json_mode", return_value=False))
                stack.enter_context(patch("sys.stdout.isatty", return_value=False))
                transcribe(
                    audio_path,
                    output_dir=tmpdir,
                    diarize_model_name="fake-model",
                    identify=True,
                    auto_enroll=True,
                )

            # avg_embedding is None — should skip enrollment
            mock_add_speaker.assert_not_called()
            mock_enroll.assert_not_called()


if __name__ == "__main__":
    unittest.main()
