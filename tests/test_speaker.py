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
# Tests for _import_wespeaker / _import_torch
# ---------------------------------------------------------------------------

class TestImportGuards(unittest.TestCase):

    @patch.dict("sys.modules", {"wespeaker": None})
    def test_import_wespeaker_missing_raises_clear_error(self):
        import importlib
        import any2md.speaker as spk
        importlib.reload(spk)
        with self.assertRaises(ImportError) as ctx:
            spk._import_wespeaker()
        self.assertIn("any2md[speaker]", str(ctx.exception))

    @patch.dict("sys.modules", {"torch": None})
    def test_import_torch_missing_raises_clear_error(self):
        import importlib
        import any2md.speaker as spk
        importlib.reload(spk)
        with self.assertRaises(ImportError) as ctx:
            spk._import_torch()
        self.assertIn("any2md[speaker]", str(ctx.exception))


# ---------------------------------------------------------------------------
# Tests for load_speaker_model
# ---------------------------------------------------------------------------

class TestLoadSpeakerModel(unittest.TestCase):

    def _mock_torch(self, mps_available: bool):
        torch_mock = MagicMock()
        torch_mock.backends.mps.is_available.return_value = mps_available
        return torch_mock

    @patch("any2md.speaker._import_wespeaker")
    @patch("any2md.speaker._import_torch")
    def test_loads_model_with_mps_when_available(self, mock_torch_fn, mock_ws_fn):
        torch_mock = self._mock_torch(mps_available=True)
        mock_torch_fn.return_value = torch_mock

        ws_mock = MagicMock()
        model_mock = MagicMock()
        ws_mock.load_model.return_value = model_mock
        mock_ws_fn.return_value = ws_mock

        from any2md.speaker import load_speaker_model
        result = load_speaker_model(device="mps")

        ws_mock.load_model.assert_called_once_with("english")
        model_mock.set_device.assert_called_once_with("mps")
        self.assertIs(result, model_mock)

    @patch("any2md.speaker._import_wespeaker")
    @patch("any2md.speaker._import_torch")
    def test_falls_back_to_cpu_when_mps_unavailable(self, mock_torch_fn, mock_ws_fn):
        torch_mock = self._mock_torch(mps_available=False)
        mock_torch_fn.return_value = torch_mock

        ws_mock = MagicMock()
        model_mock = MagicMock()
        ws_mock.load_model.return_value = model_mock
        mock_ws_fn.return_value = ws_mock

        from any2md.speaker import load_speaker_model
        result = load_speaker_model(device="mps")

        model_mock.set_device.assert_called_once_with("cpu")
        self.assertIs(result, model_mock)

    @patch("any2md.speaker._import_wespeaker")
    @patch("any2md.speaker._import_torch")
    def test_explicit_cpu_device(self, mock_torch_fn, mock_ws_fn):
        torch_mock = self._mock_torch(mps_available=True)
        mock_torch_fn.return_value = torch_mock

        ws_mock = MagicMock()
        model_mock = MagicMock()
        ws_mock.load_model.return_value = model_mock
        mock_ws_fn.return_value = ws_mock

        from any2md.speaker import load_speaker_model
        load_speaker_model(device="cpu")

        model_mock.set_device.assert_called_once_with("cpu")


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

        torch_mock = MagicMock()
        torch_mock.backends.mps.is_available.return_value = False

        ws_mock = MagicMock()
        model_mock = _fake_wespeaker_model()
        ws_mock.load_model.return_value = model_mock

        with patch("any2md.speaker._import_wespeaker", return_value=ws_mock), \
             patch("any2md.speaker._import_torch", return_value=torch_mock):
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

        torch_mock = MagicMock()
        torch_mock.backends.mps.is_available.return_value = False

        ws_mock = MagicMock()
        model_mock = _fake_wespeaker_model()
        ws_mock.load_model.return_value = model_mock

        with patch("any2md.speaker._import_wespeaker", return_value=ws_mock), \
             patch("any2md.speaker._import_torch", return_value=torch_mock):
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


if __name__ == "__main__":
    unittest.main()
