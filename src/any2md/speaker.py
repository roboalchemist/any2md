#!/usr/bin/env python3
"""
speaker.py - WeSpeaker ResNet293 speaker embedding extraction + speaker catalog

Extracts 256-d L2-normalized speaker embeddings from audio segments using
WeSpeaker ResNet293 (Wespeaker/wespeaker-voxceleb-resnet293-LM) via PyTorch MPS
on Apple Silicon.

Also provides a persistent SQLite speaker catalog at ~/.config/any2md/speakers.db
backed by sqlite-vec for fast KNN search (gallery model with multiple embeddings
per speaker for robustness across channels/conditions).

Usage (embedding extraction):
    from any2md.speaker import load_speaker_model, extract_embedding, extract_embeddings_for_segments

    model = load_speaker_model(device='mps')
    embedding = extract_embedding(model, 'audio.wav')  # (256,) numpy array
    results = extract_embeddings_for_segments(model, 'audio.wav', diarized_segments)

Usage (catalog):
    from any2md.speaker import open_catalog, add_speaker, enroll, match_speaker

    conn = open_catalog()
    speaker_id = add_speaker(conn, 'Alice')
    enroll(conn, speaker_id, embedding, source_file='meeting.wav', start=0.0, end=3.5)
    match = match_speaker(conn, query_embedding)  # {'id': ..., 'name': 'Alice', 'distance': 0.08}

CLI:
    any2md speaker add "Alice" --audio file.wav
    any2md speaker list
    any2md speaker remove "Alice"
    any2md speaker merge "Alice" "Bob"
    any2md speaker stats "Alice"
    any2md speaker gallery "Alice"
"""

import json
import logging
import os
import sqlite3
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import typer

logger = logging.getLogger(__name__)

# WeSpeaker model identifier used by wespeaker.load_model()
WESPEAKER_LANG = "english"  # downloads wespeaker-voxceleb-resnet293-LM

# Audio parameters — must match what WeSpeaker expects (same as Parakeet)
AUDIO_SAMPLE_RATE = 16000  # 16kHz mono WAV

# Default catalog path
_DEFAULT_CATALOG_PATH = Path.home() / ".config" / "any2md" / "speakers.db"

# Module-level connection cache keyed by resolved path string
_connections: Dict[str, sqlite3.Connection] = {}

# Embedding dimension for WeSpeaker ResNet293
EMBEDDING_DIM = 256

# Gallery maintenance defaults
_DEFAULT_MAX_ENROLLMENTS = 20

# ---------------------------------------------------------------------------
# Migration DDL
# ---------------------------------------------------------------------------

_MIGRATIONS = [
    # Migration 1: initial schema
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY
    );

    CREATE TABLE IF NOT EXISTS speakers (
        id              TEXT PRIMARY KEY,
        name            TEXT NOT NULL UNIQUE,
        centroid        BLOB NOT NULL,
        enrollment_count INTEGER DEFAULT 0,
        meeting_count   INTEGER DEFAULT 0,
        mean_distance   REAL DEFAULT 0.0,
        std_distance    REAL DEFAULT 0.0,
        last_seen_at    TEXT,
        created_at      TEXT NOT NULL,
        updated_at      TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS enrollments (
        id              TEXT PRIMARY KEY,
        speaker_id      TEXT NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
        embedding       BLOB NOT NULL,
        source_file     TEXT,
        segment_start   REAL,
        segment_end     REAL,
        source_type     TEXT DEFAULT 'unknown',
        confidence      REAL,
        is_representative INTEGER DEFAULT 0,
        created_at      TEXT NOT NULL
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS vec_enrollments USING vec0(
        enrollment_id TEXT PRIMARY KEY,
        embedding float[256] distance_metric=cosine
    );

    CREATE TABLE IF NOT EXISTS speaker_merges (
        id              TEXT PRIMARY KEY,
        from_speaker_id TEXT NOT NULL,
        to_speaker_id   TEXT NOT NULL,
        reason          TEXT,
        merged_at       TEXT NOT NULL
    );
    """,
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC time as ISO8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _emb_to_blob(embedding: np.ndarray) -> bytes:
    """Serialize a float32 numpy array to bytes for SQLite BLOB storage."""
    return embedding.astype(np.float32).tobytes()


def _blob_to_emb(blob: bytes) -> np.ndarray:
    """Deserialize a SQLite BLOB to a float32 numpy array."""
    return np.frombuffer(blob, dtype=np.float32)


def _load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Attempt to load sqlite-vec extension. Returns True if successful."""
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return True
    except (ImportError, sqlite3.OperationalError) as e:
        logger.warning("sqlite-vec not available (%s); KNN search will use Python fallback", e)
        return False


def _run_migration(conn: sqlite3.Connection, sql: str) -> None:
    """Execute a multi-statement migration block, skipping empty statements."""
    for stmt in sql.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)


def _get_schema_version(conn: sqlite3.Connection) -> int:
    """Return the current schema version (0 if schema_version table doesn't exist)."""
    try:
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        return row[0] if row[0] is not None else 0
    except sqlite3.OperationalError:
        return 0


# ---------------------------------------------------------------------------
# Catalog lifecycle
# ---------------------------------------------------------------------------


def open_catalog(path: Optional[str] = None) -> sqlite3.Connection:
    """Open (or create) the speaker catalog database.

    On first call for a given path, creates the database file and runs all
    pending schema migrations. Subsequent calls return the cached connection.
    Loads sqlite-vec extension for KNN search if available.

    Args:
        path: Filesystem path for the SQLite database. Defaults to
              ~/.config/any2md/speakers.db. Pass ':memory:' for in-memory
              (useful for testing).

    Returns:
        sqlite3.Connection with WAL mode enabled, foreign keys on,
        and sqlite-vec loaded (if installed).
    """
    if path is None:
        resolved = str(_DEFAULT_CATALOG_PATH)
    else:
        resolved = path

    # Return cached connection if available (skip for :memory: — always fresh)
    if resolved != ":memory:" and resolved in _connections:
        return _connections[resolved]

    # Create parent directories for on-disk databases
    if resolved != ":memory:":
        db_path = Path(resolved)
        db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(resolved, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # Performance and safety pragmas
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    # Load sqlite-vec extension (non-fatal if absent)
    _load_sqlite_vec(conn)

    # Run pending migrations
    current_version = _get_schema_version(conn)
    for i, migration_sql in enumerate(_MIGRATIONS):
        migration_version = i + 1
        if current_version < migration_version:
            _run_migration(conn, migration_sql)
            conn.execute(
                "INSERT OR REPLACE INTO schema_version(version) VALUES (?)",
                (migration_version,),
            )
            conn.commit()
            logger.debug("Applied schema migration %d", migration_version)

    if resolved != ":memory:":
        _connections[resolved] = conn

    return conn


def close_catalog(path: Optional[str] = None) -> None:
    """Close and remove a cached catalog connection.

    Args:
        path: Path used when opening the catalog. Defaults to the default path.
    """
    if path is None:
        resolved = str(_DEFAULT_CATALOG_PATH)
    else:
        resolved = path

    if resolved in _connections:
        _connections[resolved].close()
        del _connections[resolved]


# ---------------------------------------------------------------------------
# Speaker CRUD
# ---------------------------------------------------------------------------


def add_speaker(conn: sqlite3.Connection, name: str) -> str:
    """Create a new speaker profile.

    Args:
        conn: Open catalog connection.
        name: Human-readable speaker name (must be unique).

    Returns:
        UUID string for the new speaker.

    Raises:
        sqlite3.IntegrityError: If a speaker with this name already exists.
    """
    speaker_id = _new_id()
    now = _now_iso()
    # Initialize centroid as a zero vector (will be updated on first enroll)
    zero_centroid = _emb_to_blob(np.zeros(EMBEDDING_DIM, dtype=np.float32))
    conn.execute(
        """
        INSERT INTO speakers (id, name, centroid, enrollment_count, meeting_count,
                              mean_distance, std_distance, last_seen_at, created_at, updated_at)
        VALUES (?, ?, ?, 0, 0, 0.0, 0.0, NULL, ?, ?)
        """,
        (speaker_id, name, zero_centroid, now, now),
    )
    conn.commit()
    logger.debug("Created speaker %r with id=%s", name, speaker_id)
    return speaker_id


def get_all_speakers(conn: sqlite3.Connection) -> List[Dict]:
    """Return all speaker profiles with enrollment counts.

    Args:
        conn: Open catalog connection.

    Returns:
        List of dicts with keys: id, name, enrollment_count, meeting_count,
        mean_distance, std_distance, last_seen_at, created_at, updated_at.
        Does NOT include centroid blobs (use get_speaker_centroid for that).
    """
    rows = conn.execute(
        """
        SELECT id, name, enrollment_count, meeting_count, mean_distance,
               std_distance, last_seen_at, created_at, updated_at
        FROM speakers
        ORDER BY name
        """
    ).fetchall()
    return [dict(row) for row in rows]


def get_speaker_by_name(conn: sqlite3.Connection, name: str) -> Optional[Dict]:
    """Look up a speaker by name.

    Args:
        conn: Open catalog connection.
        name: Speaker name (case-sensitive).

    Returns:
        Dict with speaker fields, or None if not found.
    """
    row = conn.execute(
        "SELECT id, name, enrollment_count, meeting_count, mean_distance, "
        "std_distance, last_seen_at, created_at, updated_at FROM speakers WHERE name = ?",
        (name,),
    ).fetchone()
    return dict(row) if row else None


def delete_speaker(conn: sqlite3.Connection, name: str) -> bool:
    """Delete a speaker and all their enrollments (cascades).

    Args:
        conn: Open catalog connection.
        name: Speaker name to delete.

    Returns:
        True if a speaker was deleted, False if no speaker with that name existed.
    """
    cursor = conn.execute("DELETE FROM speakers WHERE name = ?", (name,))
    conn.commit()
    deleted = cursor.rowcount > 0
    if deleted:
        logger.debug("Deleted speaker %r and their enrollments", name)
    return deleted


def get_enrollments(conn: sqlite3.Connection, speaker_id: str) -> List[Dict]:
    """Return all enrollment records for a speaker.

    Args:
        conn: Open catalog connection.
        speaker_id: UUID of the speaker.

    Returns:
        List of dicts with keys: id, speaker_id, source_file, segment_start,
        segment_end, source_type, confidence, is_representative, created_at.
        Does NOT include raw embedding blobs.
    """
    rows = conn.execute(
        """
        SELECT id, speaker_id, source_file, segment_start, segment_end,
               source_type, confidence, is_representative, created_at
        FROM enrollments
        WHERE speaker_id = ?
        ORDER BY created_at ASC
        """,
        (speaker_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def merge_speakers(
    conn: sqlite3.Connection,
    from_name: str,
    to_name: str,
    reason: Optional[str] = None,
) -> Dict:
    """Merge speaker from_name into to_name.

    Moves all enrollments from from_name to to_name, recomputes the centroid
    and distance stats for to_name, records the merge in speaker_merges, and
    deletes the from_name speaker.

    Args:
        conn: Open catalog connection.
        from_name: Name of the speaker to merge (will be deleted).
        to_name: Name of the target speaker (retains identity).
        reason: Optional human-readable reason for the merge.

    Returns:
        Dict with keys: from_id, to_id, from_name, to_name, enrollment_count,
        merge_id.

    Raises:
        ValueError: If either speaker name is not found.
    """
    from_speaker = get_speaker_by_name(conn, from_name)
    if from_speaker is None:
        raise ValueError(f"Speaker not found: {from_name!r}")
    to_speaker = get_speaker_by_name(conn, to_name)
    if to_speaker is None:
        raise ValueError(f"Speaker not found: {to_name!r}")

    from_id = from_speaker["id"]
    to_id = to_speaker["id"]

    # Move vec_enrollments rows — best-effort (sqlite-vec may not be available)
    from_enrollment_ids = conn.execute(
        "SELECT id FROM enrollments WHERE speaker_id = ?", (from_id,)
    ).fetchall()
    for row in from_enrollment_ids:
        eid = row["id"]
        try:
            conn.execute(
                "UPDATE vec_enrollments SET enrollment_id = ? WHERE enrollment_id = ?",
                (eid, eid),  # no-op on content, but ensures integrity
            )
        except sqlite3.OperationalError:
            pass

    # Move enrollments to to_speaker
    conn.execute(
        "UPDATE enrollments SET speaker_id = ? WHERE speaker_id = ?",
        (to_id, from_id),
    )

    # Record the merge audit trail
    merge_id = _new_id()
    now = _now_iso()
    conn.execute(
        """
        INSERT INTO speaker_merges (id, from_speaker_id, to_speaker_id, reason, merged_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (merge_id, from_id, to_id, reason, now),
    )

    # Delete the from_speaker (enrollments already re-parented, cascade only removes remaining)
    conn.execute("DELETE FROM speakers WHERE id = ?", (from_id,))
    conn.commit()

    # Recompute centroid and stats for to_speaker
    update_centroid(conn, to_id)
    update_distance_stats(conn, to_id)

    # Update enrollment count for to_speaker
    conn.execute(
        """
        UPDATE speakers
        SET enrollment_count = (SELECT COUNT(*) FROM enrollments WHERE speaker_id = ?),
            updated_at = ?
        WHERE id = ?
        """,
        (to_id, now, to_id),
    )
    conn.commit()

    # Fetch updated count
    updated = conn.execute(
        "SELECT enrollment_count FROM speakers WHERE id = ?", (to_id,)
    ).fetchone()
    new_count = updated["enrollment_count"] if updated else 0

    logger.debug("Merged speaker %r into %r (%d total enrollments)", from_name, to_name, new_count)
    return {
        "from_id": from_id,
        "to_id": to_id,
        "from_name": from_name,
        "to_name": to_name,
        "enrollment_count": new_count,
        "merge_id": merge_id,
    }


# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------


def enroll(
    conn: sqlite3.Connection,
    speaker_id: str,
    embedding: np.ndarray,
    source_file: Optional[str] = None,
    start: Optional[float] = None,
    end: Optional[float] = None,
    source_type: str = "unknown",
    confidence: Optional[float] = None,
) -> str:
    """Store a speaker embedding in the gallery and update the centroid.

    Stores the embedding in both the `enrollments` table (with metadata) and
    the `vec_enrollments` virtual table (for KNN search). Then recomputes the
    speaker's centroid from all enrollments.

    Args:
        conn: Open catalog connection.
        speaker_id: UUID of the speaker to enroll into.
        embedding: 256-d numpy array (will be L2-normalized before storage).
        source_file: Path to the source audio file (optional).
        start: Segment start time in seconds (optional).
        end: Segment end time in seconds (optional).
        source_type: Channel/source descriptor ('zoom', 'laptop_mic', 'phone',
                     'manual', 'unknown').
        confidence: Enrollment confidence score 0.0–1.0 (optional).

    Returns:
        UUID string for the new enrollment record.

    Raises:
        ValueError: If speaker_id does not exist in the catalog.
    """
    # Verify speaker exists
    row = conn.execute("SELECT id FROM speakers WHERE id = ?", (speaker_id,)).fetchone()
    if row is None:
        raise ValueError(f"Speaker with id={speaker_id!r} not found in catalog")

    # L2-normalize before storage
    normalized = _l2_normalize(embedding.astype(np.float32))

    enrollment_id = _new_id()
    now = _now_iso()

    # Store in enrollments table
    conn.execute(
        """
        INSERT INTO enrollments
            (id, speaker_id, embedding, source_file, segment_start, segment_end,
             source_type, confidence, is_representative, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
        """,
        (
            enrollment_id,
            speaker_id,
            _emb_to_blob(normalized),
            source_file,
            start,
            end,
            source_type,
            confidence,
            now,
        ),
    )

    # Store in vec_enrollments for KNN (best-effort — fails gracefully if sqlite-vec missing)
    try:
        conn.execute(
            "INSERT INTO vec_enrollments(enrollment_id, embedding) VALUES (?, ?)",
            (enrollment_id, _emb_to_blob(normalized)),
        )
    except sqlite3.OperationalError as e:
        logger.debug("vec_enrollments insert skipped (%s); KNN will use Python fallback", e)

    conn.commit()

    # Update speaker stats
    update_centroid(conn, speaker_id)
    update_distance_stats(conn, speaker_id)

    # Update enrollment count and last_seen_at
    conn.execute(
        """
        UPDATE speakers
        SET enrollment_count = (SELECT COUNT(*) FROM enrollments WHERE speaker_id = ?),
            last_seen_at = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (speaker_id, now, now, speaker_id),
    )
    conn.commit()

    logger.debug("Enrolled embedding %s for speaker %s", enrollment_id, speaker_id)
    return enrollment_id


# ---------------------------------------------------------------------------
# Centroid management
# ---------------------------------------------------------------------------


def update_centroid(conn: sqlite3.Connection, speaker_id: str) -> None:
    """Recompute and store the L2-normalized mean centroid for a speaker.

    Fetches all enrollment embeddings, L2-normalizes each, computes their
    mean, then L2-normalizes the mean. Stores the result in speakers.centroid.

    Args:
        conn: Open catalog connection.
        speaker_id: UUID of the speaker to update.
    """
    rows = conn.execute(
        "SELECT embedding FROM enrollments WHERE speaker_id = ?",
        (speaker_id,),
    ).fetchall()

    if not rows:
        return

    embeddings = [_blob_to_emb(row["embedding"]) for row in rows]
    # Each stored embedding is already L2-normalized, but re-normalize defensively
    normalized = [_l2_normalize(e) for e in embeddings]
    centroid = np.mean(normalized, axis=0).astype(np.float32)
    centroid = _l2_normalize(centroid)

    conn.execute(
        "UPDATE speakers SET centroid = ?, updated_at = ? WHERE id = ?",
        (_emb_to_blob(centroid), _now_iso(), speaker_id),
    )
    conn.commit()


def update_distance_stats(conn: sqlite3.Connection, speaker_id: str) -> None:
    """Compute and store mean/std cosine distance of enrollments to centroid.

    The resulting statistics enable adaptive matching thresholds — speakers
    with more consistent voices will have low std_distance.

    Args:
        conn: Open catalog connection.
        speaker_id: UUID of the speaker to update.
    """
    centroid_row = conn.execute(
        "SELECT centroid FROM speakers WHERE id = ?", (speaker_id,)
    ).fetchone()
    if centroid_row is None:
        return

    centroid = _blob_to_emb(centroid_row["centroid"])
    centroid_norm = _l2_normalize(centroid)

    rows = conn.execute(
        "SELECT embedding FROM enrollments WHERE speaker_id = ?",
        (speaker_id,),
    ).fetchall()

    if not rows:
        return

    distances = []
    for row in rows:
        emb = _l2_normalize(_blob_to_emb(row["embedding"]))
        # Cosine distance = 1 - cosine_similarity; both vectors are unit norm
        cosine_sim = float(np.dot(emb, centroid_norm))
        distances.append(1.0 - cosine_sim)

    mean_dist = float(np.mean(distances))
    std_dist = float(np.std(distances))

    conn.execute(
        "UPDATE speakers SET mean_distance = ?, std_distance = ?, updated_at = ? WHERE id = ?",
        (mean_dist, std_dist, _now_iso(), speaker_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def match_speaker(
    conn: sqlite3.Connection,
    embedding: np.ndarray,
    threshold: float = 0.55,
) -> Optional[Dict]:
    """Find the best-matching speaker for a query embedding.

    Uses sqlite-vec KNN search over the vec_enrollments virtual table
    (cosine distance metric). Falls back to Python-based centroid comparison
    if sqlite-vec is unavailable.

    The match threshold is expressed as cosine *distance* (0 = identical,
    1 = orthogonal). A threshold of 0.55 means we require at least ~0.45
    cosine similarity (moderately similar). To require closer matches, use
    a smaller threshold (e.g., 0.30 for high-confidence only).

    Args:
        conn: Open catalog connection.
        embedding: Query 256-d numpy array (will be L2-normalized internally).
        threshold: Maximum cosine distance to accept as a match (default 0.55).
                   Queries returning distance > threshold are rejected (returns None).

    Returns:
        Dict with keys {'id', 'name', 'distance', 'enrollment_id'} for the
        best matching speaker, or None if no match is above the threshold.
    """
    query = _l2_normalize(embedding.astype(np.float32))

    # Try sqlite-vec KNN first
    try:
        results = conn.execute(
            """
            SELECT ve.enrollment_id, ve.distance,
                   e.speaker_id,
                   s.name
            FROM vec_enrollments ve
            JOIN enrollments e ON e.id = ve.enrollment_id
            JOIN speakers s ON s.id = e.speaker_id
            WHERE ve.embedding MATCH ?
              AND k = 5
            ORDER BY ve.distance
            """,
            [_emb_to_blob(query)],
        ).fetchall()

        if results:
            best = results[0]
            distance = float(best["distance"])
            if distance <= threshold:
                return {
                    "id": best["speaker_id"],
                    "name": best["name"],
                    "distance": distance,
                    "enrollment_id": best["enrollment_id"],
                }
            return None

    except sqlite3.OperationalError as e:
        logger.debug("sqlite-vec KNN failed (%s); falling back to Python centroid search", e)

    # Python fallback: compare against stored centroids
    return _match_speaker_python_fallback(conn, query, threshold)


def _match_speaker_python_fallback(
    conn: sqlite3.Connection,
    query: np.ndarray,
    threshold: float,
) -> Optional[Dict]:
    """Cosine distance match against all speaker centroids (Python fallback).

    Used when sqlite-vec is not available or the vec_enrollments table is empty.
    Loads all centroids into memory — suitable for catalogs with <10k speakers.

    Args:
        conn: Open catalog connection.
        query: L2-normalized query embedding (256-d float32).
        threshold: Maximum cosine distance to accept.

    Returns:
        Best match dict or None.
    """
    rows = conn.execute(
        "SELECT id, name, centroid, enrollment_count FROM speakers WHERE enrollment_count > 0"
    ).fetchall()

    if not rows:
        return None

    best_distance = float("inf")
    best_row = None

    for row in rows:
        centroid = _l2_normalize(_blob_to_emb(row["centroid"]))
        cosine_sim = float(np.dot(query, centroid))
        distance = 1.0 - cosine_sim
        if distance < best_distance:
            best_distance = distance
            best_row = row

    if best_row is not None and best_distance <= threshold:
        return {
            "id": best_row["id"],
            "name": best_row["name"],
            "distance": best_distance,
            "enrollment_id": None,  # Not available in centroid fallback
        }
    return None


# ---------------------------------------------------------------------------
# Gallery maintenance
# ---------------------------------------------------------------------------


def maintain_gallery(
    conn: sqlite3.Connection,
    speaker_id: str,
    max_enrollments: int = _DEFAULT_MAX_ENROLLMENTS,
) -> int:
    """Prune oldest enrollments beyond max_enrollments per speaker.

    Keeps the most recent `max_enrollments` embeddings, deleting older ones
    from both `enrollments` and `vec_enrollments`. After pruning, recomputes
    the centroid and distance stats.

    Args:
        conn: Open catalog connection.
        speaker_id: UUID of the speaker to prune.
        max_enrollments: Maximum number of enrollments to retain (default 20).

    Returns:
        Number of enrollments deleted.
    """
    count_row = conn.execute(
        "SELECT COUNT(*) as cnt FROM enrollments WHERE speaker_id = ?",
        (speaker_id,),
    ).fetchone()
    total = count_row["cnt"]

    if total <= max_enrollments:
        return 0

    excess = total - max_enrollments

    # Find the oldest enrollment IDs to delete
    old_ids = conn.execute(
        """
        SELECT id FROM enrollments
        WHERE speaker_id = ?
        ORDER BY created_at ASC
        LIMIT ?
        """,
        (speaker_id, excess),
    ).fetchall()

    ids_to_delete = [row["id"] for row in old_ids]

    for eid in ids_to_delete:
        # Remove from vec_enrollments (best-effort)
        try:
            conn.execute("DELETE FROM vec_enrollments WHERE enrollment_id = ?", (eid,))
        except sqlite3.OperationalError:
            pass
        conn.execute("DELETE FROM enrollments WHERE id = ?", (eid,))

    conn.commit()

    # Recompute centroid and stats after pruning
    update_centroid(conn, speaker_id)
    update_distance_stats(conn, speaker_id)

    # Update enrollment count
    conn.execute(
        """
        UPDATE speakers
        SET enrollment_count = (SELECT COUNT(*) FROM enrollments WHERE speaker_id = ?),
            updated_at = ?
        WHERE id = ?
        """,
        (speaker_id, _now_iso(), speaker_id),
    )
    conn.commit()

    logger.debug("Pruned %d old enrollments for speaker %s", excess, speaker_id)
    return excess


# ---------------------------------------------------------------------------
# WeSpeaker embedding extraction (from ANY2-11)
# ---------------------------------------------------------------------------


def _import_wespeaker():
    """Import wespeaker, raising a clear error if not installed."""
    try:
        import wespeaker
        return wespeaker
    except ImportError as e:
        raise ImportError(
            "wespeaker not installed — run: uv pip install 'any2md[speaker]'"
        ) from e


def _import_torch():
    """Import torch, raising a clear error if not installed."""
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError(
            "torch not installed — run: uv pip install 'any2md[speaker]'"
        ) from e


def load_speaker_model(device: str = "mps") -> Any:
    """Load WeSpeaker ResNet293 speaker embedding model.

    Downloads the model from HuggingFace on first call (cached afterwards).
    Uses PyTorch MPS on Apple Silicon; falls back to CPU if MPS is unavailable.

    Args:
        device: PyTorch device string. 'mps' for Apple Silicon (default),
                'cuda' for NVIDIA GPU, 'cpu' for CPU-only.

    Returns:
        Loaded WeSpeaker model object with set_device() already called.

    Raises:
        ImportError: If wespeaker or torch is not installed.
    """
    wespeaker = _import_wespeaker()
    torch = _import_torch()

    # Resolve device — fall back to CPU if MPS is requested but not available
    if device == "mps":
        if torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            logger.warning("MPS not available, falling back to CPU for speaker embeddings")
            resolved_device = "cpu"
    else:
        resolved_device = device

    logger.info("Loading WeSpeaker ResNet293 model (device=%s)...", resolved_device)
    model = wespeaker.load_model(WESPEAKER_LANG)
    model.set_device(resolved_device)
    logger.info("WeSpeaker model loaded successfully")
    return model


def _slice_audio_segment(
    audio_path: str,
    start: float,
    end: float,
    output_path: str,
) -> None:
    """Slice an audio segment using ffmpeg.

    Args:
        audio_path: Source WAV file path (16kHz mono WAV).
        start: Start time in seconds.
        end: End time in seconds.
        output_path: Destination WAV file path.

    Raises:
        subprocess.CalledProcessError: If ffmpeg fails.
        ValueError: If start >= end.
    """
    if start >= end:
        raise ValueError(f"Segment start ({start}) must be less than end ({end})")

    cmd = [
        "ffmpeg",
        "-y",           # Overwrite output
        "-ss", str(start),
        "-to", str(end),
        "-i", audio_path,
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path,
    ]

    logger.debug("Slicing audio: ffmpeg -ss %.3f -to %.3f ...", start, end)
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else str(e)
        raise subprocess.CalledProcessError(
            e.returncode, e.cmd,
            output=e.output,
            stderr=f"ffmpeg segment slice failed: {stderr}".encode(),
        )


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a 1-D numpy array in-place (returns new array)."""
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return vec
    return vec / norm


def extract_embedding(
    model: Any,
    audio_path: str,
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> np.ndarray:
    """Extract a 256-d L2-normalized speaker embedding from an audio file or segment.

    If start/end are provided, the segment is sliced via ffmpeg to a temp WAV
    before embedding extraction. If start/end are None, the full file is used.

    Args:
        model: Loaded WeSpeaker model (from load_speaker_model()).
        audio_path: Path to 16kHz mono WAV file.
        start: Segment start time in seconds (optional).
        end: Segment end time in seconds (optional).

    Returns:
        numpy.ndarray of shape (256,) and dtype float32, L2-normalized.

    Raises:
        ImportError: If wespeaker is not installed.
        FileNotFoundError: If audio_path does not exist.
        ValueError: If start >= end when both are provided.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if start is not None and end is not None:
        # Slice to a temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            _slice_audio_segment(audio_path, start, end, tmp_path)
            raw = model.extract_embedding(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        raw = model.extract_embedding(audio_path)

    # wespeaker returns a numpy array; ensure float32 and L2-normalize
    embedding = np.array(raw, dtype=np.float32).flatten()
    return _l2_normalize(embedding)


def extract_embeddings_for_segments(
    model: Any,
    audio_path: str,
    segments: List[Dict],
) -> List[Dict]:
    """Extract speaker embeddings for a list of diarized segments.

    Processes segments sequentially (not batched) to keep MPS memory usage
    predictable on Apple Silicon. Each segment is sliced to a temp WAV via
    ffmpeg and passed to WeSpeaker for embedding extraction.

    Args:
        model: Loaded WeSpeaker model (from load_speaker_model()).
        audio_path: Path to 16kHz mono WAV file (same file used for STT).
        segments: List of diarization segment dicts, each containing at minimum:
            - 'start': float, segment start time in seconds
            - 'end': float, segment end time in seconds
            - 'speaker': str, speaker label (e.g., 'SPEAKER_0')
            May also contain 'text' and other keys (passed through unchanged).

    Returns:
        List of dicts, one per input segment, with the original keys preserved
        plus 'embedding' (numpy.ndarray of shape (256,), float32, L2-normalized).
        Example item::

            {
                'start': 0.0,
                'end': 3.5,
                'speaker': 'SPEAKER_0',
                'text': 'Hello world',
                'embedding': np.array([...], dtype=float32),  # (256,)
            }

    Raises:
        ImportError: If wespeaker is not installed.
        FileNotFoundError: If audio_path does not exist.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    results = []
    for i, seg in enumerate(segments):
        start = float(seg["start"])
        end = float(seg["end"])
        speaker_label = seg.get("speaker", f"SPEAKER_{i}")

        logger.debug(
            "Extracting embedding for segment %d/%d: %s [%.2f-%.2f]",
            i + 1, len(segments), speaker_label, start, end,
        )

        try:
            embedding = extract_embedding(model, audio_path, start=start, end=end)
        except Exception as e:
            logger.warning(
                "Failed to extract embedding for segment %d (%s, %.2f-%.2f): %s",
                i, speaker_label, start, end, e,
            )
            embedding = np.zeros(256, dtype=np.float32)

        out = dict(seg)  # copy all original keys (start, end, speaker, text, ...)
        out["embedding"] = embedding
        results.append(out)

    logger.info("Extracted %d speaker embeddings from %s", len(results), audio_path)
    return results


def identify_speakers(
    conn: sqlite3.Connection,
    segments: List[Dict],
    audio_path: str,
    high_conf_threshold: float = 0.15,
    low_conf_threshold: float = 0.30,
) -> Dict[str, Dict]:
    """Match diarized speaker labels against the speaker catalog.

    Groups segments by speaker label, computes a duration-weighted average
    embedding per speaker, then calls match_speaker() against the catalog.

    Thresholds use cosine *distance* (lower = more similar):
    - distance <= high_conf_threshold: auto-match, enroll embedding, update stats
    - high_conf_threshold < distance <= low_conf_threshold: match but flag with confidence
    - distance > low_conf_threshold: no match, keep SPEAKER_N label

    After identification, matched speakers have their meeting_count incremented,
    last_seen_at updated, and the averaged embedding enrolled into their gallery.
    maintain_gallery() is called for each updated speaker.

    Args:
        conn: Open catalog connection.
        segments: List of diarized segment dicts. Each must contain 'speaker' (str),
                  'start' (float), 'end' (float). May contain 'embedding' (numpy array)
                  if extract_embeddings_for_segments() was already called; if absent,
                  the speaker's longest segment embedding will be used.
        audio_path: Path to 16kHz mono WAV (used to extract embeddings if missing).
        high_conf_threshold: Cosine distance threshold for high-confidence auto-match
                             (default 0.15 — tight match, auto-enroll + update stats).
        low_conf_threshold: Cosine distance threshold for flagged match (default 0.30 —
                            match shown with confidence score but no gallery update).

    Returns:
        Dict mapping speaker label (e.g. 'SPEAKER_0') to a result dict:
        {
            'name': str,        # resolved name or original SPEAKER_N label
            'matched': bool,    # True if a catalog match was found
            'distance': float,  # cosine distance (0=identical); None if no match
            'high_conf': bool,  # True if distance <= high_conf_threshold
        }
        Unmatched speakers have 'matched': False, 'name': original label.
    """
    # Group segments by unique speaker label
    speaker_segments: Dict[str, List[Dict]] = {}
    for seg in segments:
        label = seg.get("speaker", "")
        if label not in speaker_segments:
            speaker_segments[label] = []
        speaker_segments[label].append(seg)

    speaker_map: Dict[str, Dict] = {}

    for label, spk_segs in speaker_segments.items():
        # Compute duration-weighted average embedding for this speaker
        avg_embedding = _compute_weighted_avg_embedding(spk_segs)

        if avg_embedding is None:
            logger.warning("No valid embeddings for speaker %s — skipping identification", label)
            speaker_map[label] = {
                "name": label,
                "matched": False,
                "distance": None,
                "high_conf": False,
                "avg_embedding": None,
                "segments": [{"start": s.get("start"), "end": s.get("end")} for s in spk_segs],
            }
            continue

        # Query catalog for best match using low_conf_threshold as outer gate
        match = match_speaker(conn, avg_embedding, threshold=low_conf_threshold)

        if match is None:
            speaker_map[label] = {
                "name": label,
                "matched": False,
                "distance": None,
                "high_conf": False,
                "avg_embedding": avg_embedding,
                "segments": [{"start": s.get("start"), "end": s.get("end")} for s in spk_segs],
            }
            logger.debug("No catalog match for speaker %s", label)
            continue

        distance = match["distance"]
        speaker_id = match["id"]
        speaker_name = match["name"]
        high_conf = distance <= high_conf_threshold

        speaker_map[label] = {
            "name": speaker_name,
            "matched": True,
            "distance": distance,
            "high_conf": high_conf,
        }
        logger.info(
            "Identified %s as %r (distance=%.3f, %s confidence)",
            label, speaker_name, distance,
            "high" if high_conf else "medium",
        )

        if high_conf:
            # Auto-enroll the averaged embedding and update speaker stats
            now = _now_iso()
            enroll(
                conn,
                speaker_id,
                avg_embedding,
                source_file=audio_path,
                source_type="auto_identify",
                confidence=round(1.0 - distance, 4),
            )
            conn.execute(
                """
                UPDATE speakers
                SET meeting_count = meeting_count + 1,
                    last_seen_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, now, speaker_id),
            )
            conn.commit()
            maintain_gallery(conn, speaker_id)
            logger.debug(
                "Auto-enrolled embedding + updated meeting_count for %r (speaker_id=%s)",
                speaker_name, speaker_id,
            )

    return speaker_map


def _compute_weighted_avg_embedding(segments: List[Dict]) -> Optional[np.ndarray]:
    """Compute a duration-weighted average embedding from a list of segments.

    Segments with an 'embedding' key are included in the average, weighted by
    (end - start). Segments without embeddings (e.g., zero-duration or failed
    extraction) are skipped.

    Args:
        segments: List of segment dicts, each with 'start', 'end', and optionally
                  'embedding' (numpy.ndarray of shape (256,)).

    Returns:
        L2-normalized 256-d numpy array, or None if no valid embeddings found.
    """
    weighted_sum = np.zeros(EMBEDDING_DIM, dtype=np.float64)
    total_weight = 0.0

    for seg in segments:
        emb = seg.get("embedding")
        if emb is None:
            continue
        emb_arr = np.array(emb, dtype=np.float64)
        # Skip zero-vector embeddings (failed extractions)
        if np.linalg.norm(emb_arr) < 1e-10:
            continue
        duration = float(seg.get("end", 0)) - float(seg.get("start", 0))
        weight = max(duration, 1e-6)  # At minimum a tiny weight to include short segs
        weighted_sum += emb_arr * weight
        total_weight += weight

    if total_weight < 1e-10:
        return None

    avg = (weighted_sum / total_weight).astype(np.float32)
    return _l2_normalize(avg)


# ---------------------------------------------------------------------------
# Auto-enrollment helpers
# ---------------------------------------------------------------------------


def _next_unknown_name(conn: sqlite3.Connection) -> str:
    """Return the next available Unknown_N name for auto-enrollment.

    Queries the catalog for existing ``Unknown_<int>`` names and returns
    ``Unknown_<max+1>`` (or ``Unknown_0`` if none exist).

    Args:
        conn: Open catalog connection.

    Returns:
        A name string like ``Unknown_0``, ``Unknown_1``, etc.
    """
    rows = conn.execute(
        "SELECT name FROM speakers WHERE name LIKE 'Unknown_%'"
    ).fetchall()
    max_n = -1
    for row in rows:
        name = row[0] if isinstance(row, (list, tuple)) else row["name"]
        suffix = name[len("Unknown_"):]
        if suffix.isdigit():
            max_n = max(max_n, int(suffix))
    return f"Unknown_{max_n + 1}"


# ---------------------------------------------------------------------------
# CLI — any2md speaker subcommand group
# ---------------------------------------------------------------------------

speaker_app = typer.Typer(
    name="speaker",
    help="Manage speaker enrollment catalog.",
    no_args_is_help=True,
)

_DEFAULT_DB_PATH = str(_DEFAULT_CATALOG_PATH)


def _fmt_date(iso_str: Optional[str]) -> str:
    """Format an ISO8601 datetime string to a compact date for table display."""
    if not iso_str:
        return "-"
    try:
        return iso_str[:10]  # 'YYYY-MM-DD'
    except Exception:
        return iso_str


def _print_table(rows: List[List[str]], headers: List[str]) -> None:
    """Print a simple fixed-width text table to stdout."""
    all_rows = [headers] + rows
    widths = [max(len(r[i]) for r in all_rows) for i in range(len(headers))]
    sep = "  "
    header_line = sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))
    divider = sep.join("-" * widths[i] for i in range(len(headers)))
    typer.echo(header_line)
    typer.echo(divider)
    for row in rows:
        typer.echo(sep.join(row[i].ljust(widths[i]) for i in range(len(headers))))


@speaker_app.command("add")
def speaker_add(
    name: str = typer.Argument(..., help="Speaker name to enroll."),
    audio: Path = typer.Option(..., "--audio", help="Path to audio file (any ffmpeg-supported format)."),
    start: Optional[float] = typer.Option(None, "--start", help="Segment start in seconds (optional)."),
    end: Optional[float] = typer.Option(None, "--end", help="Segment end in seconds (optional)."),
    db: str = typer.Option(_DEFAULT_DB_PATH, "--db", help="Path to speaker catalog database."),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output JSON to stdout."),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable verbose logging."),
) -> None:
    """Enroll a speaker from an audio file (or segment).

    Extracts a WeSpeaker embedding and adds it to the speaker catalog.
    Creates the speaker profile automatically if it does not exist.
    """
    from any2md.common import setup_logging
    setup_logging(verbose)

    # Validate audio path
    audio_path = Path(audio).resolve()
    if not audio_path.exists():
        typer.echo(f"Error: audio file not found: {audio_path}", err=True)
        raise typer.Exit(1)

    if start is not None and end is not None and start >= end:
        typer.echo(f"Error: --start ({start}) must be less than --end ({end})", err=True)
        raise typer.Exit(1)

    # Convert to 16kHz mono WAV using yt.convert_audio_for_whisper
    try:
        from any2md.yt import convert_audio_for_whisper
    except ImportError as exc:
        typer.echo(f"Error: yt module unavailable — {exc}", err=True)
        raise typer.Exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            wav_path = convert_audio_for_whisper(str(audio_path), output_dir=tmpdir)
        except Exception as exc:
            typer.echo(f"Error converting audio to WAV: {exc}", err=True)
            raise typer.Exit(1)

        # Load WeSpeaker model and extract embedding
        try:
            model = load_speaker_model()
        except ImportError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1)

        try:
            embedding = extract_embedding(model, wav_path, start=start, end=end)
        except Exception as exc:
            typer.echo(f"Error extracting embedding: {exc}", err=True)
            raise typer.Exit(1)

    # Open catalog and create or update speaker
    conn = open_catalog(db if db != _DEFAULT_DB_PATH else None)
    existing = get_speaker_by_name(conn, name)
    if existing is None:
        speaker_id = add_speaker(conn, name)
    else:
        speaker_id = existing["id"]

    # Build source_type hint from segment vs full-file
    source_type = "manual_segment" if (start is not None or end is not None) else "manual_full"

    enroll(
        conn,
        speaker_id,
        embedding,
        source_file=str(audio_path),
        start=start,
        end=end,
        source_type=source_type,
    )

    speaker = get_speaker_by_name(conn, name)
    count = speaker["enrollment_count"] if speaker else 1

    if json_output:
        typer.echo(json.dumps({
            "speaker": name,
            "speaker_id": speaker_id,
            "enrollment_count": count,
            "action": "created" if existing is None else "enrolled",
        }))
    else:
        action = "Created and enrolled" if existing is None else "Enrolled"
        typer.echo(f"{action} {name!r} ({count} total enrollment{'s' if count != 1 else ''})")


@speaker_app.command("list")
def speaker_list(
    db: str = typer.Option(_DEFAULT_DB_PATH, "--db", help="Path to speaker catalog database."),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output JSON to stdout."),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable verbose logging."),
) -> None:
    """List all enrolled speakers."""
    from any2md.common import setup_logging
    setup_logging(verbose)

    conn = open_catalog(db if db != _DEFAULT_DB_PATH else None)
    speakers = get_all_speakers(conn)

    if json_output:
        typer.echo(json.dumps(speakers, default=str))
        return

    if not speakers:
        typer.echo("No speakers enrolled. Use: any2md speaker add")
        return

    rows = [
        [
            s["name"],
            str(s["enrollment_count"]),
            str(s.get("meeting_count", 0)),
            _fmt_date(s.get("last_seen_at")),
            _fmt_date(s.get("created_at")),
            _fmt_date(s.get("updated_at")),
        ]
        for s in speakers
    ]
    _print_table(rows, ["Name", "Enrollments", "Meetings", "Last Seen", "Created", "Updated"])


@speaker_app.command("remove")
def speaker_remove(
    name: str = typer.Argument(..., help="Speaker name to remove."),
    force: bool = typer.Option(False, "--force", help="Skip confirmation prompt."),
    db: str = typer.Option(_DEFAULT_DB_PATH, "--db", help="Path to speaker catalog database."),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output JSON to stdout."),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable verbose logging."),
) -> None:
    """Remove a speaker and all their enrollments from the catalog."""
    from any2md.common import setup_logging
    setup_logging(verbose)

    conn = open_catalog(db if db != _DEFAULT_DB_PATH else None)
    existing = get_speaker_by_name(conn, name)
    if existing is None:
        typer.echo(f"Speaker not found: {name!r}", err=True)
        raise typer.Exit(1)

    enrollment_count = existing["enrollment_count"]

    if not force:
        confirmed = typer.confirm(
            f"Delete speaker {name!r} and {enrollment_count} enrollment(s)?"
        )
        if not confirmed:
            typer.echo("Aborted.")
            raise typer.Exit(0)

    deleted = delete_speaker(conn, name)

    if json_output:
        typer.echo(json.dumps({"speaker": name, "deleted": deleted, "enrollments_removed": enrollment_count}))
    else:
        if deleted:
            typer.echo(f"Removed speaker {name!r} and {enrollment_count} enrollment(s).")
        else:
            typer.echo(f"Speaker not found: {name!r}", err=True)
            raise typer.Exit(1)


@speaker_app.command("merge")
def speaker_merge(
    name_a: str = typer.Argument(..., help="Speaker to merge FROM (will be deleted)."),
    name_b: str = typer.Argument(..., help="Speaker to merge INTO (retains identity)."),
    reason: Optional[str] = typer.Option(None, "--reason", help="Optional reason for the merge."),
    db: str = typer.Option(_DEFAULT_DB_PATH, "--db", help="Path to speaker catalog database."),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output JSON to stdout."),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable verbose logging."),
) -> None:
    """Merge speaker A into speaker B (A is deleted, B retains all enrollments)."""
    from any2md.common import setup_logging
    setup_logging(verbose)

    conn = open_catalog(db if db != _DEFAULT_DB_PATH else None)

    try:
        result = merge_speakers(conn, name_a, name_b, reason=reason)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    if json_output:
        typer.echo(json.dumps(result))
    else:
        typer.echo(
            f"Merged {name_a!r} into {name_b!r}. "
            f"{name_b!r} now has {result['enrollment_count']} enrollment(s)."
        )


@speaker_app.command("stats")
def speaker_stats(
    name: str = typer.Argument(..., help="Speaker name to show stats for."),
    db: str = typer.Option(_DEFAULT_DB_PATH, "--db", help="Path to speaker catalog database."),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output JSON to stdout."),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable verbose logging."),
) -> None:
    """Show detailed stats for a speaker (enrollment count, distance stats, meeting count)."""
    from any2md.common import setup_logging
    setup_logging(verbose)

    conn = open_catalog(db if db != _DEFAULT_DB_PATH else None)
    speaker = get_speaker_by_name(conn, name)
    if speaker is None:
        typer.echo(f"Speaker not found: {name!r}", err=True)
        raise typer.Exit(1)

    if json_output:
        # Exclude centroid blob; already excluded by get_speaker_by_name
        typer.echo(json.dumps(speaker, default=str))
        return

    typer.echo(f"Speaker: {speaker['name']}")
    typer.echo(f"  Enrollments:    {speaker['enrollment_count']}")
    typer.echo(f"  Meetings:       {speaker.get('meeting_count', 0)}")
    typer.echo(f"  Mean distance:  {speaker.get('mean_distance', 0.0):.4f}")
    typer.echo(f"  Std distance:   {speaker.get('std_distance', 0.0):.4f}")
    typer.echo(f"  Last seen:      {_fmt_date(speaker.get('last_seen_at'))}")
    typer.echo(f"  Created:        {_fmt_date(speaker.get('created_at'))}")
    typer.echo(f"  Updated:        {_fmt_date(speaker.get('updated_at'))}")


@speaker_app.command("gallery")
def speaker_gallery(
    name: str = typer.Argument(..., help="Speaker name to show gallery for."),
    db: str = typer.Option(_DEFAULT_DB_PATH, "--db", help="Path to speaker catalog database."),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output JSON to stdout."),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable verbose logging."),
) -> None:
    """List all enrollments for a speaker (source type, date, confidence)."""
    from any2md.common import setup_logging
    setup_logging(verbose)

    conn = open_catalog(db if db != _DEFAULT_DB_PATH else None)
    speaker = get_speaker_by_name(conn, name)
    if speaker is None:
        typer.echo(f"Speaker not found: {name!r}", err=True)
        raise typer.Exit(1)

    enrollments = get_enrollments(conn, speaker["id"])

    if json_output:
        typer.echo(json.dumps(enrollments, default=str))
        return

    if not enrollments:
        typer.echo(f"No enrollments found for speaker {name!r}.")
        return

    typer.echo(f"Gallery for {name!r} ({len(enrollments)} enrollment(s)):")
    rows = []
    for i, e in enumerate(enrollments, 1):
        conf = f"{e['confidence']:.3f}" if e.get("confidence") is not None else "-"
        seg = "-"
        if e.get("segment_start") is not None and e.get("segment_end") is not None:
            seg = f"{e['segment_start']:.1f}-{e['segment_end']:.1f}s"
        rows.append([
            str(i),
            e.get("source_type") or "-",
            seg,
            conf,
            "yes" if e.get("is_representative") else "no",
            _fmt_date(e.get("created_at")),
            (e.get("source_file") or "-"),
        ])
    _print_table(rows, ["#", "Source Type", "Segment", "Confidence", "Representative", "Date", "File"])
