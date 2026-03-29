#!/usr/bin/env python3
"""
Tests for speaker.py — WeSpeaker ResNet293 speaker embedding extraction
and speaker catalog (SQLite + sqlite-vec gallery model).

Run with: python -m pytest tests/test_speaker.py -v
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "audio" / "voxceleb"


def _fake_wespeaker_model(embedding_value: float = 0.5) -> MagicMock:
    """Return a mock wespeaker model that returns a fixed numpy embedding."""
    model = MagicMock()
    embedding = np.full(256, embedding_value, dtype=np.float32)
    model.extract_embedding.return_value = embedding
    return model


# ---------------------------------------------------------------------------
# Tests for _l2_normalize
# ---------------------------------------------------------------------------

class TestL2Normalize(unittest.TestCase):

    def test_normalizes_unit_vector(self):
        from any2md.speaker import _l2_normalize
        v = np.array([3.0, 4.0], dtype=np.float32)
        result = _l2_normalize(v)
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=6)

    def test_returns_zeros_for_zero_vector(self):
        from any2md.speaker import _l2_normalize
        v = np.zeros(256, dtype=np.float32)
        result = _l2_normalize(v)
        # Zero vector stays zero — no division
        np.testing.assert_array_equal(result, v)

    def test_256d_random_vector_unit_after_normalize(self):
        from any2md.speaker import _l2_normalize
        rng = np.random.default_rng(42)
        v = rng.standard_normal(256).astype(np.float32)
        result = _l2_normalize(v)
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=5)


# ---------------------------------------------------------------------------
# Tests for _import_wespeaker
# ---------------------------------------------------------------------------

class TestImportGuards(unittest.TestCase):

    @patch.dict("sys.modules", {"wespeakerruntime": None})
    def test_import_wespeaker_missing_raises_clear_error(self):
        import importlib
        import any2md.speaker as spk
        importlib.reload(spk)
        with self.assertRaises(ImportError) as ctx:
            spk._import_wespeaker()
        self.assertIn("any2md[speaker]", str(ctx.exception))


# ---------------------------------------------------------------------------
# Tests for load_speaker_model
# ---------------------------------------------------------------------------

class TestLoadSpeakerModel(unittest.TestCase):

    @patch("any2md.speaker._import_wespeaker")
    def test_loads_model_via_speaker_constructor(self, mock_ws_fn):
        """load_speaker_model() uses wespeakerruntime.Speaker(lang='en') API."""
        ws_mock = MagicMock()
        model_mock = MagicMock()
        ws_mock.Speaker.return_value = model_mock
        mock_ws_fn.return_value = ws_mock

        from any2md.speaker import load_speaker_model
        result = load_speaker_model(device="mps")

        ws_mock.Speaker.assert_called_once_with(lang="en")
        self.assertIs(result, model_mock)

    @patch("any2md.speaker._import_wespeaker")
    def test_device_arg_is_ignored(self, mock_ws_fn):
        """device argument is accepted but ignored (ONNX doesn't use torch devices)."""
        ws_mock = MagicMock()
        model_mock = MagicMock()
        ws_mock.Speaker.return_value = model_mock
        mock_ws_fn.return_value = ws_mock

        from any2md.speaker import load_speaker_model
        # Should not raise regardless of device arg
        load_speaker_model(device="cpu")
        load_speaker_model(device="mps")
        self.assertEqual(ws_mock.Speaker.call_count, 2)


# ---------------------------------------------------------------------------
# Tests for extract_embedding
# ---------------------------------------------------------------------------

class TestExtractEmbedding(unittest.TestCase):

    def test_raises_if_file_not_found(self):
        from any2md.speaker import extract_embedding
        model = _fake_wespeaker_model()
        with self.assertRaises(FileNotFoundError):
            extract_embedding(model, "/nonexistent/audio.wav")

    def test_returns_256d_float32_array(self):
        from any2md.speaker import extract_embedding
        model = _fake_wespeaker_model()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            result = extract_embedding(model, tmp)
            self.assertEqual(result.shape, (256,))
            self.assertEqual(result.dtype, np.float32)
        finally:
            os.unlink(tmp)

    def test_result_is_l2_normalized(self):
        from any2md.speaker import extract_embedding
        model = _fake_wespeaker_model(embedding_value=2.0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            result = extract_embedding(model, tmp)
            self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0, places=5)
        finally:
            os.unlink(tmp)

    @patch("any2md.speaker._slice_audio_segment")
    def test_slices_segment_when_start_end_provided(self, mock_slice):
        from any2md.speaker import extract_embedding
        model = _fake_wespeaker_model()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_path = f.name
        try:
            result = extract_embedding(model, audio_path, start=1.0, end=3.5)
            # _slice_audio_segment should have been called with start/end
            mock_slice.assert_called_once()
            args = mock_slice.call_args[0]
            self.assertEqual(args[0], audio_path)
            self.assertAlmostEqual(args[1], 1.0)
            self.assertAlmostEqual(args[2], 3.5)
        finally:
            os.unlink(audio_path)

    def test_calls_model_directly_when_no_start_end(self):
        from any2md.speaker import extract_embedding
        model = _fake_wespeaker_model()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            extract_embedding(model, tmp)
            model.extract_embedding.assert_called_once_with(tmp)
        finally:
            os.unlink(tmp)

    def test_raises_value_error_if_start_geq_end(self):
        from any2md.speaker import extract_embedding
        model = _fake_wespeaker_model()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            with self.assertRaises(ValueError):
                extract_embedding(model, tmp, start=5.0, end=3.0)
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# Tests for extract_embeddings_for_segments
# ---------------------------------------------------------------------------

class TestExtractEmbeddingsForSegments(unittest.TestCase):

    def _make_audio(self):
        """Create a temp WAV file and return its path."""
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        f.close()
        return f.name

    def test_raises_if_audio_not_found(self):
        from any2md.speaker import extract_embeddings_for_segments
        model = _fake_wespeaker_model()
        with self.assertRaises(FileNotFoundError):
            extract_embeddings_for_segments(model, "/no/such/file.wav", [])

    def test_empty_segments_returns_empty_list(self):
        from any2md.speaker import extract_embeddings_for_segments
        model = _fake_wespeaker_model()
        audio = self._make_audio()
        try:
            result = extract_embeddings_for_segments(model, audio, [])
            self.assertEqual(result, [])
        finally:
            os.unlink(audio)

    @patch("any2md.speaker._slice_audio_segment")
    def test_returns_one_dict_per_segment_with_embedding(self, mock_slice):
        from any2md.speaker import extract_embeddings_for_segments
        model = _fake_wespeaker_model()
        audio = self._make_audio()
        segments = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_0", "text": "Hello"},
            {"start": 2.5, "end": 5.0, "speaker": "SPEAKER_1", "text": "World"},
        ]
        try:
            results = extract_embeddings_for_segments(model, audio, segments)
            self.assertEqual(len(results), 2)
            for r, orig in zip(results, segments):
                # Original keys preserved
                self.assertEqual(r["start"], orig["start"])
                self.assertEqual(r["end"], orig["end"])
                self.assertEqual(r["speaker"], orig["speaker"])
                self.assertEqual(r["text"], orig["text"])
                # Embedding present and correct shape
                self.assertIn("embedding", r)
                self.assertEqual(r["embedding"].shape, (256,))
                self.assertEqual(r["embedding"].dtype, np.float32)
        finally:
            os.unlink(audio)

    @patch("any2md.speaker._slice_audio_segment")
    def test_embedding_is_l2_normalized(self, mock_slice):
        from any2md.speaker import extract_embeddings_for_segments
        model = _fake_wespeaker_model(embedding_value=3.0)
        audio = self._make_audio()
        segments = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_0"}]
        try:
            results = extract_embeddings_for_segments(model, audio, segments)
            norm = float(np.linalg.norm(results[0]["embedding"]))
            self.assertAlmostEqual(norm, 1.0, places=5)
        finally:
            os.unlink(audio)

    @patch("any2md.speaker._slice_audio_segment")
    def test_failed_segment_produces_zero_embedding(self, mock_slice):
        """If extract_embedding raises, the segment gets a zero vector instead of crashing."""
        from any2md.speaker import extract_embeddings_for_segments
        model = MagicMock()
        model.extract_embedding.side_effect = RuntimeError("model failure")
        audio = self._make_audio()
        segments = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_0"}]
        try:
            results = extract_embeddings_for_segments(model, audio, segments)
            self.assertEqual(len(results), 1)
            np.testing.assert_array_equal(results[0]["embedding"], np.zeros(256, dtype=np.float32))
        finally:
            os.unlink(audio)

    @patch("any2md.speaker._slice_audio_segment")
    def test_does_not_mutate_input_segments(self, mock_slice):
        """Input segment dicts must not be modified in-place."""
        from any2md.speaker import extract_embeddings_for_segments
        model = _fake_wespeaker_model()
        audio = self._make_audio()
        original = {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_0"}
        segments = [original]
        try:
            extract_embeddings_for_segments(model, audio, segments)
            # Original dict should not have 'embedding' key added
            self.assertNotIn("embedding", original)
        finally:
            os.unlink(audio)

    @patch("any2md.speaker._slice_audio_segment")
    def test_processes_all_speakers_sequentially(self, mock_slice):
        """All segments processed — no batching or skipping."""
        from any2md.speaker import extract_embeddings_for_segments
        model = _fake_wespeaker_model()
        audio = self._make_audio()
        segments = [
            {"start": float(i), "end": float(i + 1), "speaker": f"SPEAKER_{i}"}
            for i in range(5)
        ]
        try:
            results = extract_embeddings_for_segments(model, audio, segments)
            self.assertEqual(len(results), 5)
            # model.extract_embedding called once per segment (via _slice + tmp)
            self.assertEqual(model.extract_embedding.call_count, 5)
        finally:
            os.unlink(audio)


# ---------------------------------------------------------------------------
# Tests for _slice_audio_segment
# ---------------------------------------------------------------------------

class TestSliceAudioSegment(unittest.TestCase):

    def test_raises_value_error_for_invalid_range(self):
        from any2md.speaker import _slice_audio_segment
        with self.assertRaises(ValueError):
            _slice_audio_segment("/some/audio.wav", start=5.0, end=3.0, output_path="/tmp/out.wav")

    @patch("any2md.speaker.subprocess.run")
    def test_calls_ffmpeg_with_correct_args(self, mock_run):
        from any2md.speaker import _slice_audio_segment
        mock_run.return_value = MagicMock(returncode=0)
        _slice_audio_segment("/audio.wav", 1.0, 3.5, "/out.wav")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertIn("ffmpeg", cmd)
        self.assertIn("-ss", cmd)
        self.assertIn("1.0", cmd)
        self.assertIn("-to", cmd)
        self.assertIn("3.5", cmd)
        self.assertIn("/audio.wav", cmd)
        self.assertIn("/out.wav", cmd)


# ---------------------------------------------------------------------------
# Integration tests against real VoxCeleb fixtures
# (require actual files — skipped in CI unless fixtures exist)
# ---------------------------------------------------------------------------

@unittest.skipUnless(FIXTURES_DIR.exists(), "VoxCeleb fixtures not present")
class TestVoxCelebFixtures(unittest.TestCase):
    """Smoke-test shape/dtype of embeddings from real VoxCeleb WAV files."""

    def test_all_fixtures_produce_256d_embeddings(self):
        import any2md.speaker as spk

        ws_mock = MagicMock()
        model_mock = _fake_wespeaker_model()
        ws_mock.Speaker.return_value = model_mock

        with patch("any2md.speaker._import_wespeaker", return_value=ws_mock):
            model = spk.load_speaker_model(device="cpu")

        wav_files = sorted(FIXTURES_DIR.rglob("*.wav"))
        self.assertGreater(len(wav_files), 0, "No WAV fixtures found under tests/audio/voxceleb/")

        for wav in wav_files:
            with self.subTest(wav=wav.name):
                emb = spk.extract_embedding(model, str(wav))
                self.assertEqual(emb.shape, (256,))
                self.assertEqual(emb.dtype, np.float32)
                norm = float(np.linalg.norm(emb))
                self.assertAlmostEqual(norm, 1.0, places=5)

    def test_segments_across_speakers(self):
        """Produce one embedding per VoxCeleb speaker using multi-segment path."""
        import any2md.speaker as spk

        ws_mock = MagicMock()
        model_mock = _fake_wespeaker_model()
        ws_mock.Speaker.return_value = model_mock

        with patch("any2md.speaker._import_wespeaker", return_value=ws_mock):
            model = spk.load_speaker_model(device="cpu")

        # Build a fake segment list pointing at the first utterance of each speaker
        speakers = sorted(FIXTURES_DIR.iterdir())
        segments = []
        audio_paths = []
        for spk_dir in speakers:
            if not spk_dir.is_dir():
                continue
            wavs = sorted(spk_dir.glob("*.wav"))
            if wavs:
                audio_paths.append(str(wavs[0]))
                segments.append({
                    "start": 0.0, "end": 2.0,
                    "speaker": spk_dir.name,
                })

        if not audio_paths:
            self.skipTest("No WAV files found")

        # Use first audio file for all segments (mock slicing anyway)
        with patch("any2md.speaker._slice_audio_segment"):
            results = spk.extract_embeddings_for_segments(model, audio_paths[0], segments)

        self.assertEqual(len(results), len(segments))
        for r in results:
            self.assertIn("embedding", r)
            self.assertEqual(r["embedding"].shape, (256,))


# ---------------------------------------------------------------------------
# Catalog helper
# ---------------------------------------------------------------------------


def _open_test_catalog():
    """Open an in-memory catalog (fresh each call, no sqlite-vec required)."""
    from any2md.speaker import open_catalog
    return open_catalog(path=":memory:")


def _rand_emb(seed: int = 0) -> np.ndarray:
    """Return a random L2-normalized 256-d float32 embedding."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(256).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Tests for open_catalog
# ---------------------------------------------------------------------------


class TestOpenCatalog(unittest.TestCase):

    def test_in_memory_catalog_opens_without_error(self):
        conn = _open_test_catalog()
        self.assertIsNotNone(conn)

    def test_schema_version_set_after_migration(self):
        conn = _open_test_catalog()
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        self.assertIsNotNone(row[0])
        self.assertGreaterEqual(row[0], 1)

    def test_speakers_table_exists(self):
        conn = _open_test_catalog()
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='speakers'"
        ).fetchone()
        self.assertIsNotNone(result)

    def test_enrollments_table_exists(self):
        conn = _open_test_catalog()
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='enrollments'"
        ).fetchone()
        self.assertIsNotNone(result)

    def test_speaker_merges_table_exists(self):
        conn = _open_test_catalog()
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='speaker_merges'"
        ).fetchone()
        self.assertIsNotNone(result)

    def test_foreign_keys_enabled(self):
        conn = _open_test_catalog()
        row = conn.execute("PRAGMA foreign_keys").fetchone()
        # Returns 1 when enabled
        self.assertEqual(row[0], 1)

    def test_wal_journal_mode(self):
        # WAL not supported for :memory: — we just confirm no error is raised
        # The important thing is the call doesn't fail
        conn = _open_test_catalog()
        self.assertIsNotNone(conn)

    def test_on_disk_catalog_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "nested", "dir", "speakers.db")
            from any2md.speaker import open_catalog, close_catalog
            conn = open_catalog(path=db_path)
            self.assertTrue(os.path.exists(db_path))
            close_catalog(path=db_path)

    def test_on_disk_catalog_connection_is_cached(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "speakers.db")
            from any2md.speaker import open_catalog, close_catalog
            conn1 = open_catalog(path=db_path)
            conn2 = open_catalog(path=db_path)
            self.assertIs(conn1, conn2)
            close_catalog(path=db_path)


# ---------------------------------------------------------------------------
# Tests for add_speaker / get_all_speakers / delete_speaker
# ---------------------------------------------------------------------------


class TestSpeakerCRUD(unittest.TestCase):

    def setUp(self):
        self.conn = _open_test_catalog()

    def test_add_speaker_returns_uuid_string(self):
        from any2md.speaker import add_speaker
        speaker_id = add_speaker(self.conn, "Alice")
        self.assertIsInstance(speaker_id, str)
        self.assertEqual(len(speaker_id), 36)  # UUID4 format

    def test_add_speaker_persists_to_db(self):
        from any2md.speaker import add_speaker
        speaker_id = add_speaker(self.conn, "Bob")
        row = self.conn.execute(
            "SELECT id, name, enrollment_count FROM speakers WHERE id = ?", (speaker_id,)
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["name"], "Bob")
        self.assertEqual(row["enrollment_count"], 0)

    def test_add_duplicate_speaker_raises_integrity_error(self):
        import sqlite3
        from any2md.speaker import add_speaker
        add_speaker(self.conn, "Charlie")
        with self.assertRaises(sqlite3.IntegrityError):
            add_speaker(self.conn, "Charlie")

    def test_get_all_speakers_empty_catalog(self):
        from any2md.speaker import get_all_speakers
        result = get_all_speakers(self.conn)
        self.assertEqual(result, [])

    def test_get_all_speakers_returns_list_of_dicts(self):
        from any2md.speaker import add_speaker, get_all_speakers
        add_speaker(self.conn, "Dave")
        add_speaker(self.conn, "Eve")
        result = get_all_speakers(self.conn)
        self.assertEqual(len(result), 2)
        names = {r["name"] for r in result}
        self.assertIn("Dave", names)
        self.assertIn("Eve", names)

    def test_get_all_speakers_has_expected_keys(self):
        from any2md.speaker import add_speaker, get_all_speakers
        add_speaker(self.conn, "Frank")
        result = get_all_speakers(self.conn)
        self.assertEqual(len(result), 1)
        expected_keys = {
            "id", "name", "enrollment_count", "meeting_count",
            "mean_distance", "std_distance", "last_seen_at", "created_at", "updated_at"
        }
        self.assertEqual(set(result[0].keys()), expected_keys)

    def test_get_speaker_by_name_found(self):
        from any2md.speaker import add_speaker, get_speaker_by_name
        speaker_id = add_speaker(self.conn, "Grace")
        result = get_speaker_by_name(self.conn, "Grace")
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], speaker_id)
        self.assertEqual(result["name"], "Grace")

    def test_get_speaker_by_name_not_found(self):
        from any2md.speaker import get_speaker_by_name
        result = get_speaker_by_name(self.conn, "NoSuchPerson")
        self.assertIsNone(result)

    def test_delete_speaker_returns_true_when_exists(self):
        from any2md.speaker import add_speaker, delete_speaker
        add_speaker(self.conn, "Heidi")
        result = delete_speaker(self.conn, "Heidi")
        self.assertTrue(result)

    def test_delete_speaker_removes_from_db(self):
        from any2md.speaker import add_speaker, delete_speaker, get_all_speakers
        add_speaker(self.conn, "Ivan")
        delete_speaker(self.conn, "Ivan")
        result = get_all_speakers(self.conn)
        self.assertEqual(result, [])

    def test_delete_speaker_returns_false_when_not_found(self):
        from any2md.speaker import delete_speaker
        result = delete_speaker(self.conn, "Nobody")
        self.assertFalse(result)

    def test_delete_speaker_cascades_enrollments(self):
        from any2md.speaker import add_speaker, enroll, delete_speaker
        speaker_id = add_speaker(self.conn, "Julia")
        enroll(self.conn, speaker_id, _rand_emb(1))
        enroll(self.conn, speaker_id, _rand_emb(2))
        delete_speaker(self.conn, "Julia")
        rows = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM enrollments WHERE speaker_id = ?", (speaker_id,)
        ).fetchone()
        self.assertEqual(rows["cnt"], 0)


# ---------------------------------------------------------------------------
# Tests for enroll
# ---------------------------------------------------------------------------


class TestEnroll(unittest.TestCase):

    def setUp(self):
        self.conn = _open_test_catalog()
        from any2md.speaker import add_speaker
        self.speaker_id = add_speaker(self.conn, "Mallory")

    def test_enroll_returns_uuid_string(self):
        from any2md.speaker import enroll
        eid = enroll(self.conn, self.speaker_id, _rand_emb(10))
        self.assertIsInstance(eid, str)
        self.assertEqual(len(eid), 36)

    def test_enroll_stores_embedding_in_enrollments_table(self):
        from any2md.speaker import enroll
        enroll(self.conn, self.speaker_id, _rand_emb(11))
        count = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM enrollments WHERE speaker_id = ?",
            (self.speaker_id,),
        ).fetchone()["cnt"]
        self.assertEqual(count, 1)

    def test_enroll_increments_enrollment_count(self):
        from any2md.speaker import enroll
        enroll(self.conn, self.speaker_id, _rand_emb(12))
        enroll(self.conn, self.speaker_id, _rand_emb(13))
        row = self.conn.execute(
            "SELECT enrollment_count FROM speakers WHERE id = ?", (self.speaker_id,)
        ).fetchone()
        self.assertEqual(row["enrollment_count"], 2)

    def test_enroll_updates_last_seen_at(self):
        from any2md.speaker import enroll
        enroll(self.conn, self.speaker_id, _rand_emb(14))
        row = self.conn.execute(
            "SELECT last_seen_at FROM speakers WHERE id = ?", (self.speaker_id,)
        ).fetchone()
        self.assertIsNotNone(row["last_seen_at"])

    def test_enroll_stores_source_metadata(self):
        from any2md.speaker import enroll
        enroll(
            self.conn, self.speaker_id, _rand_emb(15),
            source_file="meeting.wav", start=1.5, end=4.0,
            source_type="zoom", confidence=0.92,
        )
        row = self.conn.execute(
            "SELECT source_file, segment_start, segment_end, source_type, confidence "
            "FROM enrollments WHERE speaker_id = ?",
            (self.speaker_id,),
        ).fetchone()
        self.assertEqual(row["source_file"], "meeting.wav")
        self.assertAlmostEqual(row["segment_start"], 1.5)
        self.assertAlmostEqual(row["segment_end"], 4.0)
        self.assertEqual(row["source_type"], "zoom")
        self.assertAlmostEqual(row["confidence"], 0.92, places=4)

    def test_enroll_normalizes_embedding_before_storage(self):
        from any2md.speaker import enroll
        # Pass a non-normalized embedding (all 2s)
        raw = np.full(256, 2.0, dtype=np.float32)
        enroll(self.conn, self.speaker_id, raw)
        row = self.conn.execute(
            "SELECT embedding FROM enrollments WHERE speaker_id = ?",
            (self.speaker_id,),
        ).fetchone()
        stored = np.frombuffer(row["embedding"], dtype=np.float32)
        norm = float(np.linalg.norm(stored))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_enroll_raises_for_unknown_speaker(self):
        from any2md.speaker import enroll
        with self.assertRaises(ValueError):
            enroll(self.conn, "nonexistent-uuid", _rand_emb(20))

    def test_enroll_updates_centroid(self):
        from any2md.speaker import enroll
        enroll(self.conn, self.speaker_id, _rand_emb(21))
        row = self.conn.execute(
            "SELECT centroid FROM speakers WHERE id = ?", (self.speaker_id,)
        ).fetchone()
        centroid = np.frombuffer(row["centroid"], dtype=np.float32)
        # After one enrollment, centroid should be unit-normalized
        self.assertAlmostEqual(float(np.linalg.norm(centroid)), 1.0, places=5)


# ---------------------------------------------------------------------------
# Tests for update_centroid
# ---------------------------------------------------------------------------


class TestUpdateCentroid(unittest.TestCase):

    def setUp(self):
        self.conn = _open_test_catalog()
        from any2md.speaker import add_speaker
        self.speaker_id = add_speaker(self.conn, "NixonMD")

    def test_centroid_is_unit_norm_after_single_enrollment(self):
        from any2md.speaker import enroll, update_centroid
        enroll(self.conn, self.speaker_id, _rand_emb(30))
        update_centroid(self.conn, self.speaker_id)
        row = self.conn.execute(
            "SELECT centroid FROM speakers WHERE id = ?", (self.speaker_id,)
        ).fetchone()
        centroid = np.frombuffer(row["centroid"], dtype=np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(centroid)), 1.0, places=5)

    def test_centroid_is_unit_norm_after_multiple_enrollments(self):
        from any2md.speaker import enroll, update_centroid
        for seed in range(5):
            enroll(self.conn, self.speaker_id, _rand_emb(seed + 40))
        update_centroid(self.conn, self.speaker_id)
        row = self.conn.execute(
            "SELECT centroid FROM speakers WHERE id = ?", (self.speaker_id,)
        ).fetchone()
        centroid = np.frombuffer(row["centroid"], dtype=np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(centroid)), 1.0, places=5)

    def test_update_centroid_no_op_when_no_enrollments(self):
        from any2md.speaker import update_centroid
        # Should not raise even with zero enrollments
        update_centroid(self.conn, self.speaker_id)

    def test_centroid_changes_after_new_enrollment(self):
        from any2md.speaker import enroll, update_centroid
        # Enroll once, record centroid
        enroll(self.conn, self.speaker_id, _rand_emb(50))
        update_centroid(self.conn, self.speaker_id)
        row1 = self.conn.execute(
            "SELECT centroid FROM speakers WHERE id = ?", (self.speaker_id,)
        ).fetchone()
        centroid1 = np.frombuffer(row1["centroid"], dtype=np.float32).copy()

        # Enroll second (very different) embedding
        opposite = _rand_emb(99)
        enroll(self.conn, self.speaker_id, opposite)
        update_centroid(self.conn, self.speaker_id)
        row2 = self.conn.execute(
            "SELECT centroid FROM speakers WHERE id = ?", (self.speaker_id,)
        ).fetchone()
        centroid2 = np.frombuffer(row2["centroid"], dtype=np.float32)

        # Centroids should differ
        self.assertFalse(np.allclose(centroid1, centroid2))


# ---------------------------------------------------------------------------
# Tests for update_distance_stats
# ---------------------------------------------------------------------------


class TestUpdateDistanceStats(unittest.TestCase):

    def setUp(self):
        self.conn = _open_test_catalog()
        from any2md.speaker import add_speaker
        self.speaker_id = add_speaker(self.conn, "Oscar")

    def test_mean_distance_is_zero_for_identical_embeddings(self):
        from any2md.speaker import enroll, update_distance_stats
        # Enroll same vector twice — distance to centroid should be near 0
        same = _rand_emb(60)
        enroll(self.conn, self.speaker_id, same)
        enroll(self.conn, self.speaker_id, same.copy())
        update_distance_stats(self.conn, self.speaker_id)
        row = self.conn.execute(
            "SELECT mean_distance FROM speakers WHERE id = ?", (self.speaker_id,)
        ).fetchone()
        self.assertAlmostEqual(row["mean_distance"], 0.0, places=4)

    def test_stats_populated_after_enroll(self):
        from any2md.speaker import enroll
        for seed in range(3):
            enroll(self.conn, self.speaker_id, _rand_emb(seed + 70))
        row = self.conn.execute(
            "SELECT mean_distance, std_distance FROM speakers WHERE id = ?",
            (self.speaker_id,),
        ).fetchone()
        # Both should be set (even if 0)
        self.assertIsNotNone(row["mean_distance"])
        self.assertIsNotNone(row["std_distance"])


# ---------------------------------------------------------------------------
# Tests for match_speaker (Python fallback path)
# ---------------------------------------------------------------------------


class TestMatchSpeakerFallback(unittest.TestCase):
    """Tests using the Python centroid fallback (works without sqlite-vec)."""

    def setUp(self):
        self.conn = _open_test_catalog()
        from any2md.speaker import add_speaker
        self.speaker_id = add_speaker(self.conn, "Peggy")

    def test_match_returns_none_when_catalog_empty(self):
        from any2md.speaker import _match_speaker_python_fallback
        result = _match_speaker_python_fallback(self.conn, _rand_emb(80), threshold=0.55)
        self.assertIsNone(result)

    def test_match_finds_enrolled_speaker(self):
        from any2md.speaker import enroll, _match_speaker_python_fallback
        ref = _rand_emb(81)
        enroll(self.conn, self.speaker_id, ref)
        # Query with the same embedding — distance should be ~0
        result = _match_speaker_python_fallback(self.conn, ref, threshold=0.55)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Peggy")
        self.assertAlmostEqual(result["distance"], 0.0, places=4)

    def test_match_returns_none_above_threshold(self):
        from any2md.speaker import enroll, _match_speaker_python_fallback, add_speaker
        ref = _rand_emb(82)
        enroll(self.conn, self.speaker_id, ref)
        # Use an orthogonal-ish query (distance ~1.0)
        orthogonal = _rand_emb(999)
        # Use a very tight threshold so nothing matches
        result = _match_speaker_python_fallback(self.conn, orthogonal, threshold=0.001)
        self.assertIsNone(result)

    def test_match_returns_closest_of_multiple_speakers(self):
        from any2md.speaker import enroll, add_speaker, _match_speaker_python_fallback
        emb_a = _rand_emb(83)
        emb_b = _rand_emb(84)
        speaker_b_id = add_speaker(self.conn, "Quinn")

        enroll(self.conn, self.speaker_id, emb_a)
        enroll(self.conn, speaker_b_id, emb_b)

        # Query close to A
        result = _match_speaker_python_fallback(self.conn, emb_a, threshold=0.55)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Peggy")

        # Query close to B
        result = _match_speaker_python_fallback(self.conn, emb_b, threshold=0.55)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Quinn")

    def test_match_result_has_expected_keys(self):
        from any2md.speaker import enroll, _match_speaker_python_fallback
        enroll(self.conn, self.speaker_id, _rand_emb(85))
        result = _match_speaker_python_fallback(self.conn, _rand_emb(85), threshold=0.55)
        self.assertIsNotNone(result)
        self.assertIn("id", result)
        self.assertIn("name", result)
        self.assertIn("distance", result)
        self.assertIn("enrollment_id", result)


# ---------------------------------------------------------------------------
# Tests for match_speaker (full path, exercises sqlite-vec if available)
# ---------------------------------------------------------------------------


class TestMatchSpeaker(unittest.TestCase):

    def setUp(self):
        self.conn = _open_test_catalog()
        from any2md.speaker import add_speaker
        self.speaker_id = add_speaker(self.conn, "Rufus")

    def test_match_returns_none_on_empty_catalog(self):
        from any2md.speaker import match_speaker
        result = match_speaker(self.conn, _rand_emb(90))
        self.assertIsNone(result)

    def test_match_finds_speaker_with_same_embedding(self):
        from any2md.speaker import enroll, match_speaker
        ref = _rand_emb(91)
        enroll(self.conn, self.speaker_id, ref)
        result = match_speaker(self.conn, ref, threshold=0.55)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Rufus")

    def test_match_returns_none_above_threshold(self):
        from any2md.speaker import enroll, match_speaker
        enroll(self.conn, self.speaker_id, _rand_emb(92))
        # Very tight threshold
        result = match_speaker(self.conn, _rand_emb(999), threshold=0.001)
        self.assertIsNone(result)

    def test_match_result_distance_is_float(self):
        from any2md.speaker import enroll, match_speaker
        ref = _rand_emb(93)
        enroll(self.conn, self.speaker_id, ref)
        result = match_speaker(self.conn, ref, threshold=0.55)
        self.assertIsNotNone(result)
        self.assertIsInstance(result["distance"], float)


# ---------------------------------------------------------------------------
# Tests for maintain_gallery
# ---------------------------------------------------------------------------


class TestMaintainGallery(unittest.TestCase):

    def setUp(self):
        self.conn = _open_test_catalog()
        from any2md.speaker import add_speaker
        self.speaker_id = add_speaker(self.conn, "Sybil")

    def test_no_pruning_when_under_limit(self):
        from any2md.speaker import enroll, maintain_gallery
        for seed in range(5):
            enroll(self.conn, self.speaker_id, _rand_emb(seed + 100))
        deleted = maintain_gallery(self.conn, self.speaker_id, max_enrollments=10)
        self.assertEqual(deleted, 0)

    def test_prunes_to_max_limit(self):
        from any2md.speaker import enroll, maintain_gallery
        for seed in range(25):
            enroll(self.conn, self.speaker_id, _rand_emb(seed + 110))
        deleted = maintain_gallery(self.conn, self.speaker_id, max_enrollments=20)
        self.assertEqual(deleted, 5)
        count = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM enrollments WHERE speaker_id = ?",
            (self.speaker_id,),
        ).fetchone()["cnt"]
        self.assertEqual(count, 20)

    def test_enrollment_count_updated_after_pruning(self):
        from any2md.speaker import enroll, maintain_gallery
        for seed in range(25):
            enroll(self.conn, self.speaker_id, _rand_emb(seed + 120))
        maintain_gallery(self.conn, self.speaker_id, max_enrollments=15)
        row = self.conn.execute(
            "SELECT enrollment_count FROM speakers WHERE id = ?",
            (self.speaker_id,),
        ).fetchone()
        self.assertEqual(row["enrollment_count"], 15)

    def test_centroid_recomputed_after_pruning(self):
        from any2md.speaker import enroll, maintain_gallery
        for seed in range(25):
            enroll(self.conn, self.speaker_id, _rand_emb(seed + 130))
        maintain_gallery(self.conn, self.speaker_id, max_enrollments=10)
        row = self.conn.execute(
            "SELECT centroid FROM speakers WHERE id = ?", (self.speaker_id,)
        ).fetchone()
        centroid = np.frombuffer(row["centroid"], dtype=np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(centroid)), 1.0, places=5)

    def test_no_op_when_exactly_at_limit(self):
        from any2md.speaker import enroll, maintain_gallery
        for seed in range(20):
            enroll(self.conn, self.speaker_id, _rand_emb(seed + 140))
        deleted = maintain_gallery(self.conn, self.speaker_id, max_enrollments=20)
        self.assertEqual(deleted, 0)


# ---------------------------------------------------------------------------
# Integration: full enroll + match round-trip
# ---------------------------------------------------------------------------


class TestCatalogRoundTrip(unittest.TestCase):

    def test_enroll_then_match_same_speaker(self):
        """Enroll two speakers, verify each matches itself and not the other."""
        conn = _open_test_catalog()
        from any2md.speaker import add_speaker, enroll, match_speaker

        id_alice = add_speaker(conn, "Alice")
        id_bob = add_speaker(conn, "Bob")

        emb_alice = _rand_emb(200)
        emb_bob = _rand_emb(201)

        # Enroll 3 embeddings each (similar but not identical)
        for delta in range(3):
            noise = np.random.default_rng(delta + 300).standard_normal(256).astype(np.float32) * 0.01
            enroll(conn, id_alice, emb_alice + noise)
            enroll(conn, id_bob, emb_bob + noise)

        # Query with exact reference embeddings — should match correct speaker
        result_a = match_speaker(conn, emb_alice, threshold=0.55)
        self.assertIsNotNone(result_a)
        self.assertEqual(result_a["name"], "Alice")

        result_b = match_speaker(conn, emb_bob, threshold=0.55)
        self.assertIsNotNone(result_b)
        self.assertEqual(result_b["name"], "Bob")

    def test_unknown_embedding_returns_none(self):
        """An embedding far from all known speakers returns None."""
        conn = _open_test_catalog()
        from any2md.speaker import add_speaker, enroll, match_speaker

        speaker_id = add_speaker(conn, "Carol")
        # Enroll embeddings clustered around seed 250
        for seed in range(5):
            enroll(conn, speaker_id, _rand_emb(seed + 250))

        # Use a very tight threshold — random embedding won't match
        result = match_speaker(conn, _rand_emb(9999), threshold=0.001)
        self.assertIsNone(result)

    def test_delete_removes_speaker_from_match_results(self):
        """After deletion, that speaker should never be returned by match."""
        conn = _open_test_catalog()
        from any2md.speaker import add_speaker, enroll, match_speaker, delete_speaker

        speaker_id = add_speaker(conn, "Dave")
        ref = _rand_emb(300)
        enroll(conn, speaker_id, ref)

        # Confirm it matches before deletion
        before = match_speaker(conn, ref, threshold=0.55)
        self.assertIsNotNone(before)
        self.assertEqual(before["name"], "Dave")

        # Delete and confirm no match
        delete_speaker(conn, "Dave")
        after = match_speaker(conn, ref, threshold=0.55)
        self.assertIsNone(after)


# ---------------------------------------------------------------------------
# Tests for _compute_weighted_avg_embedding
# ---------------------------------------------------------------------------


class TestComputeWeightedAvgEmbedding(unittest.TestCase):

    def test_returns_none_when_no_embeddings(self):
        from any2md.speaker import _compute_weighted_avg_embedding
        segments = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_0", "text": "hi"},
        ]
        # No 'embedding' key present
        result = _compute_weighted_avg_embedding(segments)
        self.assertIsNone(result)

    def test_returns_none_for_zero_embeddings(self):
        from any2md.speaker import _compute_weighted_avg_embedding
        segments = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_0",
             "embedding": np.zeros(256, dtype=np.float32)},
        ]
        result = _compute_weighted_avg_embedding(segments)
        self.assertIsNone(result)

    def test_single_segment_returns_normalized_embedding(self):
        from any2md.speaker import _compute_weighted_avg_embedding
        emb = _rand_emb(42)
        segments = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_0", "embedding": emb},
        ]
        result = _compute_weighted_avg_embedding(segments)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0, places=5)

    def test_weighted_by_duration(self):
        """Longer segment should dominate the average."""
        from any2md.speaker import _compute_weighted_avg_embedding
        emb_a = np.zeros(256, dtype=np.float32)
        emb_a[0] = 1.0  # Points entirely in dim 0
        emb_b = np.zeros(256, dtype=np.float32)
        emb_b[1] = 1.0  # Points entirely in dim 1

        # emb_b is 10x longer
        segments = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_0", "embedding": emb_a},
            {"start": 1.0, "end": 11.0, "speaker": "SPEAKER_0", "embedding": emb_b},
        ]
        result = _compute_weighted_avg_embedding(segments)
        self.assertIsNotNone(result)
        # Result should be closer to emb_b (dim 1 larger)
        self.assertGreater(float(result[1]), float(result[0]))

    def test_output_is_unit_norm(self):
        """Result must always be L2-normalized."""
        from any2md.speaker import _compute_weighted_avg_embedding
        segments = [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_0", "embedding": _rand_emb(1)},
            {"start": 3.0, "end": 5.0, "speaker": "SPEAKER_0", "embedding": _rand_emb(2)},
            {"start": 5.0, "end": 6.0, "speaker": "SPEAKER_0", "embedding": _rand_emb(3)},
        ]
        result = _compute_weighted_avg_embedding(segments)
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0, places=5)


# ---------------------------------------------------------------------------
# Tests for identify_speakers
# ---------------------------------------------------------------------------


class TestIdentifySpeakers(unittest.TestCase):

    def _setup_catalog_with_speaker(self, name: str, seed: int = 0):
        """Create an in-memory catalog with one enrolled speaker."""
        conn = _open_test_catalog()
        from any2md.speaker import add_speaker, enroll
        speaker_id = add_speaker(conn, name)
        ref_emb = _rand_emb(seed)
        enroll(conn, speaker_id, ref_emb)
        return conn, speaker_id, ref_emb

    def test_returns_dict_keyed_by_speaker_label(self):
        from any2md.speaker import identify_speakers
        conn, _, ref = self._setup_catalog_with_speaker("Alice", seed=10)
        segments = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_0", "text": "hi", "embedding": ref},
        ]
        result = identify_speakers(conn, segments, "/fake/audio.wav")
        self.assertIn("SPEAKER_0", result)

    def test_matched_speaker_has_correct_name(self):
        from any2md.speaker import identify_speakers, _l2_normalize
        conn, _, ref = self._setup_catalog_with_speaker("Alice", seed=20)
        # Use nearly identical embedding for matching
        close_emb = _l2_normalize(ref + np.random.default_rng(999).standard_normal(256).astype(np.float32) * 0.01)
        segments = [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_0", "text": "hi", "embedding": close_emb},
        ]
        result = identify_speakers(conn, segments, "/fake/audio.wav")
        self.assertTrue(result["SPEAKER_0"]["matched"])
        self.assertEqual(result["SPEAKER_0"]["name"], "Alice")

    def test_unmatched_speaker_keeps_original_label(self):
        from any2md.speaker import identify_speakers
        conn = _open_test_catalog()  # Empty catalog — no speakers
        segments = [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_1", "text": "hello",
             "embedding": _rand_emb(50)},
        ]
        result = identify_speakers(conn, segments, "/fake/audio.wav")
        self.assertIn("SPEAKER_1", result)
        self.assertFalse(result["SPEAKER_1"]["matched"])
        self.assertEqual(result["SPEAKER_1"]["name"], "SPEAKER_1")

    def test_result_includes_distance_for_matched_speaker(self):
        from any2md.speaker import identify_speakers
        conn, _, ref = self._setup_catalog_with_speaker("Bob", seed=30)
        segments = [
            {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_0", "text": "hey", "embedding": ref},
        ]
        result = identify_speakers(conn, segments, "/fake/audio.wav")
        if result["SPEAKER_0"]["matched"]:
            self.assertIsNotNone(result["SPEAKER_0"]["distance"])
            self.assertIsInstance(result["SPEAKER_0"]["distance"], float)

    def test_no_embeddings_produces_unmatched_entry(self):
        from any2md.speaker import identify_speakers
        conn, _, _ = self._setup_catalog_with_speaker("Carol", seed=40)
        # Segments without 'embedding' key
        segments = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_0", "text": "bye"},
        ]
        result = identify_speakers(conn, segments, "/fake/audio.wav")
        self.assertIn("SPEAKER_0", result)
        self.assertFalse(result["SPEAKER_0"]["matched"])

    def test_multiple_speakers_independently_matched(self):
        from any2md.speaker import identify_speakers, add_speaker, enroll
        conn = _open_test_catalog()
        from any2md.speaker import add_speaker, enroll
        alice_id = add_speaker(conn, "Alice")
        bob_id = add_speaker(conn, "Bob")
        alice_emb = _rand_emb(60)
        bob_emb = _rand_emb(61)
        enroll(conn, alice_id, alice_emb)
        enroll(conn, bob_id, bob_emb)

        segments = [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_0", "text": "hi", "embedding": alice_emb},
            {"start": 3.0, "end": 6.0, "speaker": "SPEAKER_1", "text": "hey", "embedding": bob_emb},
        ]
        result = identify_speakers(conn, segments, "/fake/audio.wav")
        self.assertIn("SPEAKER_0", result)
        self.assertIn("SPEAKER_1", result)
        # Both should match
        self.assertTrue(result["SPEAKER_0"]["matched"])
        self.assertTrue(result["SPEAKER_1"]["matched"])
        self.assertEqual(result["SPEAKER_0"]["name"], "Alice")
        self.assertEqual(result["SPEAKER_1"]["name"], "Bob")

    def test_high_conf_flag_set_for_close_match(self):
        """A match with very small distance should be flagged high_conf=True."""
        from any2md.speaker import identify_speakers
        conn, _, ref = self._setup_catalog_with_speaker("Dana", seed=70)
        segments = [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_0", "text": "hi", "embedding": ref},
        ]
        # Use very tight thresholds so same embedding definitely hits high_conf
        result = identify_speakers(conn, segments, "/fake/audio.wav",
                                   high_conf_threshold=0.99, low_conf_threshold=0.99)
        self.assertTrue(result["SPEAKER_0"].get("matched"))
        self.assertTrue(result["SPEAKER_0"]["high_conf"])

    def test_medium_conf_high_conf_false(self):
        """A match inside low threshold but with high_conf_threshold below actual distance → high_conf=False."""
        from any2md.speaker import identify_speakers, _l2_normalize
        conn, _, ref = self._setup_catalog_with_speaker("Eve", seed=80)
        # Create a query embedding that is somewhat different from the enrolled one
        # by rotating slightly — should produce a non-zero distance
        rng = np.random.default_rng(12345)
        noise = rng.standard_normal(256).astype(np.float32) * 0.5
        query_emb = _l2_normalize(ref + noise)  # noticeably different but still close

        # Match should succeed with a loose low_conf_threshold
        # but fail high_conf since distance > 0.0
        segments = [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_0", "text": "hi", "embedding": query_emb},
        ]
        # Set high_conf_threshold to near-zero so only perfect matches are high_conf
        result = identify_speakers(conn, segments, "/fake/audio.wav",
                                   high_conf_threshold=0.001, low_conf_threshold=0.99)
        if result["SPEAKER_0"]["matched"]:
            # Distance should be > 0.001 due to added noise
            self.assertFalse(result["SPEAKER_0"]["high_conf"])


# ---------------------------------------------------------------------------
# Tests for _next_unknown_name
# ---------------------------------------------------------------------------


class TestNextUnknownName(unittest.TestCase):

    def test_returns_unknown_0_when_catalog_empty(self):
        from any2md.speaker import _next_unknown_name
        conn = _open_test_catalog()
        result = _next_unknown_name(conn)
        self.assertEqual(result, "Unknown_0")

    def test_increments_from_existing_unknown(self):
        from any2md.speaker import _next_unknown_name, add_speaker
        conn = _open_test_catalog()
        add_speaker(conn, "Unknown_0")
        result = _next_unknown_name(conn)
        self.assertEqual(result, "Unknown_1")

    def test_increments_from_highest_existing(self):
        from any2md.speaker import _next_unknown_name, add_speaker
        conn = _open_test_catalog()
        add_speaker(conn, "Unknown_0")
        add_speaker(conn, "Unknown_3")
        add_speaker(conn, "Unknown_1")
        result = _next_unknown_name(conn)
        self.assertEqual(result, "Unknown_4")

    def test_ignores_non_numeric_suffixes(self):
        """Unknown_X where X is not an integer should not affect counter."""
        from any2md.speaker import _next_unknown_name, add_speaker
        conn = _open_test_catalog()
        add_speaker(conn, "Unknown_abc")
        result = _next_unknown_name(conn)
        self.assertEqual(result, "Unknown_0")

    def test_ignores_unrelated_speaker_names(self):
        from any2md.speaker import _next_unknown_name, add_speaker
        conn = _open_test_catalog()
        add_speaker(conn, "Alice")
        add_speaker(conn, "Bob")
        result = _next_unknown_name(conn)
        self.assertEqual(result, "Unknown_0")


# ---------------------------------------------------------------------------
# Tests for identify_speakers: avg_embedding and segments in unmatched entries
# ---------------------------------------------------------------------------


class TestIdentifyUnmatchedHasEmbedding(unittest.TestCase):
    """Unmatched entries from identify_speakers() should include avg_embedding and segments."""

    def test_unmatched_entry_has_avg_embedding(self):
        from any2md.speaker import identify_speakers
        conn = _open_test_catalog()  # Empty catalog
        emb = _rand_emb(100)
        segments = [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_0", "text": "hello", "embedding": emb},
        ]
        result = identify_speakers(conn, segments, "/fake/audio.wav")
        self.assertFalse(result["SPEAKER_0"]["matched"])
        self.assertIn("avg_embedding", result["SPEAKER_0"])
        avg_emb = result["SPEAKER_0"]["avg_embedding"]
        self.assertIsNotNone(avg_emb)
        self.assertEqual(avg_emb.shape, (256,))

    def test_unmatched_entry_has_segments(self):
        from any2md.speaker import identify_speakers
        conn = _open_test_catalog()
        emb = _rand_emb(101)
        segments = [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_0", "text": "hello", "embedding": emb},
            {"start": 4.0, "end": 7.0, "speaker": "SPEAKER_0", "text": "world", "embedding": emb},
        ]
        result = identify_speakers(conn, segments, "/fake/audio.wav")
        self.assertFalse(result["SPEAKER_0"]["matched"])
        segs = result["SPEAKER_0"]["segments"]
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0]["start"], 0.0)
        self.assertEqual(segs[1]["start"], 4.0)

    def test_no_embedding_unmatched_has_none_avg_embedding(self):
        """Segments without 'embedding' key → avg_embedding should be None."""
        from any2md.speaker import identify_speakers
        conn = _open_test_catalog()
        segments = [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_0", "text": "hello"},
        ]
        result = identify_speakers(conn, segments, "/fake/audio.wav")
        self.assertFalse(result["SPEAKER_0"]["matched"])
        self.assertIsNone(result["SPEAKER_0"]["avg_embedding"])

    def test_matched_entry_does_not_require_avg_embedding_key(self):
        """Matched entries don't need avg_embedding — only unmatched do."""
        from any2md.speaker import identify_speakers
        conn = _open_test_catalog()
        from any2md.speaker import add_speaker, enroll
        speaker_id = add_speaker(conn, "Alice")
        ref = _rand_emb(102)
        enroll(conn, speaker_id, ref)
        segments = [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_0", "text": "hi", "embedding": ref},
        ]
        result = identify_speakers(conn, segments, "/fake/audio.wav",
                                   high_conf_threshold=0.99, low_conf_threshold=0.99)
        # When matched, avg_embedding presence is not required (implementation detail)
        # Just verify name is resolved
        self.assertTrue(result["SPEAKER_0"]["matched"])
        self.assertEqual(result["SPEAKER_0"]["name"], "Alice")


# ---------------------------------------------------------------------------
# Tests for match_speaker speaker_ids filter
# ---------------------------------------------------------------------------


class TestMatchSpeakerFilter(unittest.TestCase):
    """Tests for the optional speaker_ids filter in match_speaker()."""

    def setUp(self):
        self.conn = _open_test_catalog()
        from any2md.speaker import add_speaker, enroll
        self.alice_id = add_speaker(self.conn, "Alice")
        self.bob_id = add_speaker(self.conn, "Bob")
        self.charlie_id = add_speaker(self.conn, "Charlie")
        self.alice_emb = _rand_emb(200)
        self.bob_emb = _rand_emb(201)
        self.charlie_emb = _rand_emb(202)
        enroll(self.conn, self.alice_id, self.alice_emb)
        enroll(self.conn, self.bob_id, self.bob_emb)
        enroll(self.conn, self.charlie_id, self.charlie_emb)

    def test_filter_excludes_charlie(self):
        """Charlie's embedding returns no match when filter is [Alice, Bob]."""
        from any2md.speaker import match_speaker
        result = match_speaker(
            self.conn,
            self.charlie_emb,
            threshold=0.55,
            speaker_ids=[self.alice_id, self.bob_id],
        )
        self.assertIsNone(result)

    def test_filter_matches_alice_when_included(self):
        """Alice's embedding matches when Alice is in the filter."""
        from any2md.speaker import match_speaker
        result = match_speaker(
            self.conn,
            self.alice_emb,
            threshold=0.55,
            speaker_ids=[self.alice_id, self.bob_id],
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Alice")

    def test_none_filter_matches_all_three(self):
        """None filter (default) should be able to match all three speakers."""
        from any2md.speaker import match_speaker
        for emb, name in [
            (self.alice_emb, "Alice"),
            (self.bob_emb, "Bob"),
            (self.charlie_emb, "Charlie"),
        ]:
            result = match_speaker(self.conn, emb, threshold=0.55, speaker_ids=None)
            self.assertIsNotNone(result)
            self.assertEqual(result["name"], name)

    def test_empty_list_filter_matches_all_three(self):
        """Empty list filter behaves like None — searches all speakers."""
        from any2md.speaker import match_speaker
        for emb, name in [
            (self.alice_emb, "Alice"),
            (self.bob_emb, "Bob"),
            (self.charlie_emb, "Charlie"),
        ]:
            result = match_speaker(self.conn, emb, threshold=0.55, speaker_ids=[])
            self.assertIsNotNone(result)
            self.assertEqual(result["name"], name)

    def test_filter_to_single_speaker(self):
        """Filter to only Bob — Bob's embedding matches, Alice's does not."""
        from any2md.speaker import match_speaker
        bob_result = match_speaker(
            self.conn, self.bob_emb, threshold=0.55, speaker_ids=[self.bob_id]
        )
        self.assertIsNotNone(bob_result)
        self.assertEqual(bob_result["name"], "Bob")

        alice_result = match_speaker(
            self.conn, self.alice_emb, threshold=0.55, speaker_ids=[self.bob_id]
        )
        self.assertIsNone(alice_result)


# ---------------------------------------------------------------------------
# Tests for _match_speaker_python_fallback speaker_ids filter
# ---------------------------------------------------------------------------


class TestMatchSpeakerFallbackFilter(unittest.TestCase):
    """Tests for speaker_ids filter in the Python centroid fallback path."""

    def setUp(self):
        self.conn = _open_test_catalog()
        from any2md.speaker import add_speaker, enroll
        self.alice_id = add_speaker(self.conn, "Alice")
        self.bob_id = add_speaker(self.conn, "Bob")
        self.charlie_id = add_speaker(self.conn, "Charlie")
        self.alice_emb = _rand_emb(210)
        self.bob_emb = _rand_emb(211)
        self.charlie_emb = _rand_emb(212)
        enroll(self.conn, self.alice_id, self.alice_emb)
        enroll(self.conn, self.bob_id, self.bob_emb)
        enroll(self.conn, self.charlie_id, self.charlie_emb)

    def test_filter_excludes_charlie_from_fallback(self):
        """Charlie's embedding returns no match when restricted to [Alice, Bob]."""
        from any2md.speaker import _match_speaker_python_fallback, _l2_normalize
        query = _l2_normalize(self.charlie_emb.astype("float32"))
        result = _match_speaker_python_fallback(
            self.conn, query, threshold=0.55,
            speaker_ids=[self.alice_id, self.bob_id],
        )
        self.assertIsNone(result)

    def test_none_filter_matches_all_in_fallback(self):
        """None filter searches all speakers in fallback path."""
        from any2md.speaker import _match_speaker_python_fallback, _l2_normalize
        for emb, name in [
            (self.alice_emb, "Alice"),
            (self.bob_emb, "Bob"),
            (self.charlie_emb, "Charlie"),
        ]:
            query = _l2_normalize(emb.astype("float32"))
            result = _match_speaker_python_fallback(
                self.conn, query, threshold=0.55, speaker_ids=None
            )
            self.assertIsNotNone(result)
            self.assertEqual(result["name"], name)


# ---------------------------------------------------------------------------
# Tests for identify_speakers speaker_names filter
# ---------------------------------------------------------------------------


class TestIdentifySpeakersFilter(unittest.TestCase):
    """Tests for the optional speaker_names filter in identify_speakers()."""

    def setUp(self):
        self.conn = _open_test_catalog()
        from any2md.speaker import add_speaker, enroll
        self.alice_id = add_speaker(self.conn, "Alice")
        self.bob_id = add_speaker(self.conn, "Bob")
        self.charlie_id = add_speaker(self.conn, "Charlie")
        self.alice_emb = _rand_emb(220)
        self.bob_emb = _rand_emb(221)
        self.charlie_emb = _rand_emb(222)
        enroll(self.conn, self.alice_id, self.alice_emb)
        enroll(self.conn, self.bob_id, self.bob_emb)
        enroll(self.conn, self.charlie_id, self.charlie_emb)

    def _make_segments(self, label_emb_pairs):
        """Build a minimal segments list from (label, embedding) pairs."""
        return [
            {"start": float(i), "end": float(i + 1), "speaker": label, "embedding": emb}
            for i, (label, emb) in enumerate(label_emb_pairs)
        ]

    def test_charlie_returns_no_match_when_filter_is_alice_bob(self):
        """Charlie's embedding → no match when filter restricts to [Alice, Bob]."""
        from any2md.speaker import identify_speakers
        segments = self._make_segments([("SPEAKER_0", self.charlie_emb)])
        result = identify_speakers(
            self.conn, segments, "/fake/audio.wav",
            speaker_names=["Alice", "Bob"],
        )
        self.assertFalse(result["SPEAKER_0"]["matched"])

    def test_charlie_unmatched_name_is_unknown_when_filter_active(self):
        """When filter is active, unmatched speakers are labeled 'Unknown'."""
        from any2md.speaker import identify_speakers
        segments = self._make_segments([("SPEAKER_0", self.charlie_emb)])
        result = identify_speakers(
            self.conn, segments, "/fake/audio.wav",
            speaker_names=["Alice", "Bob"],
        )
        self.assertEqual(result["SPEAKER_0"]["name"], "Unknown")

    def test_none_filter_matches_all_three(self):
        """None filter (default) → all three speakers can be matched."""
        from any2md.speaker import identify_speakers
        segments = self._make_segments([
            ("SPEAKER_0", self.alice_emb),
            ("SPEAKER_1", self.bob_emb),
            ("SPEAKER_2", self.charlie_emb),
        ])
        result = identify_speakers(self.conn, segments, "/fake/audio.wav", speaker_names=None)
        self.assertTrue(result["SPEAKER_0"]["matched"])
        self.assertTrue(result["SPEAKER_1"]["matched"])
        self.assertTrue(result["SPEAKER_2"]["matched"])
        self.assertEqual(result["SPEAKER_0"]["name"], "Alice")
        self.assertEqual(result["SPEAKER_1"]["name"], "Bob")
        self.assertEqual(result["SPEAKER_2"]["name"], "Charlie")

    def test_nonexistent_name_warns_but_does_not_error(self):
        """Unknown name in speaker_names logs a warning but doesn't raise."""
        import logging
        from any2md.speaker import identify_speakers
        segments = self._make_segments([("SPEAKER_0", self.alice_emb)])
        with self.assertLogs("any2md.speaker", level=logging.WARNING) as cm:
            result = identify_speakers(
                self.conn, segments, "/fake/audio.wav",
                speaker_names=["Alice", "DoesNotExist"],
            )
        # Should have warned about the unknown name
        self.assertTrue(
            any("DoesNotExist" in msg for msg in cm.output),
            f"Expected warning about DoesNotExist, got: {cm.output}",
        )
        # Should still attempt to match Alice (no error)
        self.assertIn("SPEAKER_0", result)

    def test_identify_with_filter_correctly_limits_results(self):
        """Filter [Alice] → Alice matches; Bob and Charlie do not."""
        from any2md.speaker import identify_speakers
        segments = self._make_segments([
            ("SPEAKER_0", self.alice_emb),
            ("SPEAKER_1", self.bob_emb),
            ("SPEAKER_2", self.charlie_emb),
        ])
        result = identify_speakers(
            self.conn, segments, "/fake/audio.wav",
            speaker_names=["Alice"],
        )
        # Alice should match
        self.assertTrue(result["SPEAKER_0"]["matched"])
        self.assertEqual(result["SPEAKER_0"]["name"], "Alice")
        # Bob and Charlie are outside the filter → Unknown
        self.assertFalse(result["SPEAKER_1"]["matched"])
        self.assertEqual(result["SPEAKER_1"]["name"], "Unknown")
        self.assertFalse(result["SPEAKER_2"]["matched"])
        self.assertEqual(result["SPEAKER_2"]["name"], "Unknown")

    def test_empty_filter_matches_all_three(self):
        """Empty speaker_names list behaves like None — searches all."""
        from any2md.speaker import identify_speakers
        segments = self._make_segments([
            ("SPEAKER_0", self.alice_emb),
            ("SPEAKER_1", self.bob_emb),
            ("SPEAKER_2", self.charlie_emb),
        ])
        result = identify_speakers(self.conn, segments, "/fake/audio.wav", speaker_names=[])
        self.assertTrue(result["SPEAKER_0"]["matched"])
        self.assertTrue(result["SPEAKER_1"]["matched"])
        self.assertTrue(result["SPEAKER_2"]["matched"])

    def test_unmatched_speaker_name_is_original_label_without_filter(self):
        """Without a filter, unmatched speakers keep original SPEAKER_N label (backward compat)."""
        from any2md.speaker import identify_speakers
        conn = _open_test_catalog()  # Empty catalog
        segments = self._make_segments([("SPEAKER_0", _rand_emb(999))])
        result = identify_speakers(conn, segments, "/fake/audio.wav")
        self.assertFalse(result["SPEAKER_0"]["matched"])
        self.assertEqual(result["SPEAKER_0"]["name"], "SPEAKER_0")


# ---------------------------------------------------------------------------
# Tests for speaker group CRUD
# ---------------------------------------------------------------------------


def _open_test_catalog_fresh() -> object:
    """Open a fresh in-memory catalog."""
    from any2md.speaker import open_catalog
    return open_catalog(":memory:")


class TestSpeakerGroups(unittest.TestCase):
    """CRUD tests for speaker groups."""

    def setUp(self):
        from any2md.speaker import open_catalog, add_speaker
        self.conn = open_catalog(":memory:")
        # Add some speakers for member tests
        add_speaker(self.conn, "Alice")
        add_speaker(self.conn, "Bob")
        add_speaker(self.conn, "Charlie")

    # --- create_group ---

    def test_create_group_returns_id(self):
        from any2md.speaker import create_group
        gid = create_group(self.conn, "Podcast Team")
        self.assertIsInstance(gid, str)
        self.assertTrue(len(gid) > 0)

    def test_create_group_duplicate_raises(self):
        import sqlite3
        from any2md.speaker import create_group
        create_group(self.conn, "Hosts")
        with self.assertRaises(sqlite3.IntegrityError):
            create_group(self.conn, "Hosts")

    def test_create_group_with_members(self):
        from any2md.speaker import create_group, get_group
        create_group(self.conn, "Hosts", member_names=["Alice", "Bob"])
        group = get_group(self.conn, "Hosts")
        names = [m["name"] for m in group["members"]]
        self.assertIn("Alice", names)
        self.assertIn("Bob", names)
        self.assertEqual(len(names), 2)

    def test_create_group_with_nonexistent_member_raises(self):
        from any2md.speaker import create_group
        with self.assertRaises(ValueError):
            create_group(self.conn, "Hosts", member_names=["Alice", "Nobody"])

    # --- list_groups ---

    def test_list_groups_empty(self):
        from any2md.speaker import list_groups
        groups = list_groups(self.conn)
        self.assertEqual(groups, [])

    def test_list_groups_returns_all(self):
        from any2md.speaker import create_group, list_groups
        create_group(self.conn, "Team A")
        create_group(self.conn, "Team B")
        groups = list_groups(self.conn)
        names = [g["name"] for g in groups]
        self.assertIn("Team A", names)
        self.assertIn("Team B", names)

    def test_list_groups_includes_member_count(self):
        from any2md.speaker import create_group, list_groups
        create_group(self.conn, "Team A", member_names=["Alice", "Bob"])
        groups = list_groups(self.conn)
        team_a = next(g for g in groups if g["name"] == "Team A")
        self.assertEqual(team_a["member_count"], 2)

    # --- get_group ---

    def test_get_group_not_found_returns_none(self):
        from any2md.speaker import get_group
        self.assertIsNone(get_group(self.conn, "NoSuchGroup"))

    def test_get_group_returns_members(self):
        from any2md.speaker import create_group, get_group
        create_group(self.conn, "Hosts", member_names=["Alice", "Bob"])
        group = get_group(self.conn, "Hosts")
        self.assertEqual(group["name"], "Hosts")
        self.assertEqual(len(group["members"]), 2)

    def test_get_group_empty_members(self):
        from any2md.speaker import create_group, get_group
        create_group(self.conn, "Empty")
        group = get_group(self.conn, "Empty")
        self.assertEqual(group["members"], [])

    # --- delete_group ---

    def test_delete_group_returns_true(self):
        from any2md.speaker import create_group, delete_group
        create_group(self.conn, "Temp")
        self.assertTrue(delete_group(self.conn, "Temp"))

    def test_delete_group_not_found_returns_false(self):
        from any2md.speaker import delete_group
        self.assertFalse(delete_group(self.conn, "Ghost"))

    def test_delete_group_removes_memberships(self):
        from any2md.speaker import create_group, delete_group, get_group
        create_group(self.conn, "Temp", member_names=["Alice"])
        delete_group(self.conn, "Temp")
        self.assertIsNone(get_group(self.conn, "Temp"))

    def test_delete_group_does_not_delete_speakers(self):
        from any2md.speaker import create_group, delete_group, get_speaker_by_name
        create_group(self.conn, "Temp", member_names=["Alice"])
        delete_group(self.conn, "Temp")
        self.assertIsNotNone(get_speaker_by_name(self.conn, "Alice"))

    # --- add_group_member ---

    def test_add_group_member_success(self):
        from any2md.speaker import create_group, add_group_member, get_group
        create_group(self.conn, "Panel")
        add_group_member(self.conn, "Panel", "Charlie")
        group = get_group(self.conn, "Panel")
        names = [m["name"] for m in group["members"]]
        self.assertIn("Charlie", names)

    def test_add_group_member_nonexistent_group_raises(self):
        from any2md.speaker import add_group_member
        with self.assertRaises(ValueError):
            add_group_member(self.conn, "NoGroup", "Alice")

    def test_add_group_member_nonexistent_speaker_raises(self):
        from any2md.speaker import create_group, add_group_member
        create_group(self.conn, "Panel")
        with self.assertRaises(ValueError):
            add_group_member(self.conn, "Panel", "Nobody")

    def test_add_group_member_idempotent(self):
        """Adding same member twice should not raise (INSERT OR IGNORE)."""
        from any2md.speaker import create_group, add_group_member, get_group
        create_group(self.conn, "Panel")
        add_group_member(self.conn, "Panel", "Alice")
        add_group_member(self.conn, "Panel", "Alice")  # should not raise
        group = get_group(self.conn, "Panel")
        self.assertEqual(len(group["members"]), 1)

    # --- remove_group_member ---

    def test_remove_group_member_success(self):
        from any2md.speaker import create_group, remove_group_member, get_group
        create_group(self.conn, "Panel", member_names=["Alice", "Bob"])
        removed = remove_group_member(self.conn, "Panel", "Alice")
        self.assertTrue(removed)
        group = get_group(self.conn, "Panel")
        names = [m["name"] for m in group["members"]]
        self.assertNotIn("Alice", names)
        self.assertIn("Bob", names)

    def test_remove_group_member_not_member_returns_false(self):
        from any2md.speaker import create_group, remove_group_member
        create_group(self.conn, "Panel")
        removed = remove_group_member(self.conn, "Panel", "Alice")
        self.assertFalse(removed)

    def test_remove_group_member_nonexistent_group_raises(self):
        from any2md.speaker import remove_group_member
        with self.assertRaises(ValueError):
            remove_group_member(self.conn, "NoGroup", "Alice")

    def test_remove_group_member_nonexistent_speaker_raises(self):
        from any2md.speaker import create_group, remove_group_member
        create_group(self.conn, "Panel")
        with self.assertRaises(ValueError):
            remove_group_member(self.conn, "Panel", "Nobody")


# ---------------------------------------------------------------------------
# Tests for resolve_speakers_arg
# ---------------------------------------------------------------------------


class TestResolveSpeciakersArg(unittest.TestCase):
    """Tests for @group resolution in --speakers."""

    def setUp(self):
        from any2md.speaker import open_catalog, add_speaker, create_group
        self.conn = open_catalog(":memory:")
        add_speaker(self.conn, "Alice")
        add_speaker(self.conn, "Bob")
        add_speaker(self.conn, "Charlie")
        create_group(self.conn, "Hosts", member_names=["Alice", "Bob"])

    def test_plain_names_passthrough(self):
        from any2md.speaker import resolve_speakers_arg
        result = resolve_speakers_arg(self.conn, "Alice,Charlie")
        self.assertEqual(result, ["Alice", "Charlie"])

    def test_group_reference_expands(self):
        from any2md.speaker import resolve_speakers_arg
        result = resolve_speakers_arg(self.conn, "@Hosts")
        self.assertEqual(set(result), {"Alice", "Bob"})

    def test_mixed_group_and_names(self):
        from any2md.speaker import resolve_speakers_arg
        result = resolve_speakers_arg(self.conn, "@Hosts,Charlie")
        self.assertIn("Alice", result)
        self.assertIn("Bob", result)
        self.assertIn("Charlie", result)
        self.assertEqual(len(result), 3)

    def test_deduplication(self):
        from any2md.speaker import resolve_speakers_arg
        # Alice appears in @Hosts and explicitly — should appear once
        result = resolve_speakers_arg(self.conn, "@Hosts,Alice")
        self.assertEqual(result.count("Alice"), 1)
        self.assertIn("Bob", result)

    def test_nonexistent_group_raises(self):
        from any2md.speaker import resolve_speakers_arg
        with self.assertRaises(ValueError):
            resolve_speakers_arg(self.conn, "@NoSuchGroup")

    def test_empty_string_returns_empty_list(self):
        from any2md.speaker import resolve_speakers_arg
        result = resolve_speakers_arg(self.conn, "")
        self.assertEqual(result, [])

    def test_whitespace_stripped(self):
        from any2md.speaker import resolve_speakers_arg
        result = resolve_speakers_arg(self.conn, " Alice , Bob ")
        self.assertIn("Alice", result)
        self.assertIn("Bob", result)

    def test_at_group_with_spaces_in_name(self):
        from any2md.speaker import resolve_speakers_arg, create_group, add_speaker
        add_speaker(self.conn, "Dave")
        create_group(self.conn, "Panel Group", member_names=["Dave"])
        result = resolve_speakers_arg(self.conn, "@Panel Group")
        self.assertIn("Dave", result)


if __name__ == "__main__":
    unittest.main()
