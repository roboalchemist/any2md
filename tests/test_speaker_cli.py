"""
test_speaker_cli.py - Unit tests for the `any2md speaker` CLI subcommand group.

All external dependencies (wespeaker, torch, ffmpeg) are mocked.
Tests use an in-memory SQLite database to avoid touching ~/.config/any2md/speakers.db.
"""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

from any2md.speaker import (
    _DEFAULT_CATALOG_PATH,
    EMBEDDING_DIM,
    add_speaker,
    create_group,
    enroll,
    get_all_speakers,
    get_enrollments,
    get_group,
    get_speaker_by_name,
    list_groups,
    merge_speakers,
    open_catalog,
    speaker_app,
)

runner = CliRunner()

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path) -> str:
    """Return a temporary database path (not in-memory so we can test --db flag)."""
    return str(tmp_path / "test_speakers.db")


@pytest.fixture()
def seeded_db(db_path) -> tuple:
    """Return (db_path, conn) with two speakers (Alice, Bob) already enrolled."""
    conn = open_catalog(db_path)
    alice_id = add_speaker(conn, "Alice")
    bob_id = add_speaker(conn, "Bob")

    fake_emb = np.ones(EMBEDDING_DIM, dtype=np.float32) / np.sqrt(EMBEDDING_DIM)
    enroll(conn, alice_id, fake_emb, source_file="alice.wav", source_type="manual_full")
    enroll(conn, alice_id, fake_emb, source_file="alice2.wav", source_type="manual_full")
    enroll(conn, bob_id, fake_emb, source_file="bob.wav", source_type="manual_full")

    return db_path, conn


def _fake_embedding() -> np.ndarray:
    """Return a unit-norm fake embedding."""
    v = np.ones(EMBEDDING_DIM, dtype=np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# `any2md speaker list`
# ---------------------------------------------------------------------------


class TestSpeakerList:
    def test_empty_catalog_message(self, db_path):
        open_catalog(db_path)  # create empty DB
        result = runner.invoke(speaker_app, ["list", "--db", db_path])
        assert result.exit_code == 0
        assert "No speakers enrolled" in result.output

    def test_lists_speakers_in_table(self, seeded_db):
        db_path, _ = seeded_db
        result = runner.invoke(speaker_app, ["list", "--db", db_path])
        assert result.exit_code == 0
        assert "Alice" in result.output
        assert "Bob" in result.output
        # Enrollment counts
        assert "2" in result.output  # Alice has 2
        assert "1" in result.output  # Bob has 1

    def test_json_output(self, seeded_db):
        db_path, _ = seeded_db
        result = runner.invoke(speaker_app, ["list", "--db", db_path, "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        names = [s["name"] for s in data]
        assert "Alice" in names
        assert "Bob" in names

    def test_json_includes_enrollment_count(self, seeded_db):
        db_path, _ = seeded_db
        result = runner.invoke(speaker_app, ["list", "--db", db_path, "--json"])
        data = json.loads(result.output)
        alice = next(s for s in data if s["name"] == "Alice")
        assert alice["enrollment_count"] == 2


# ---------------------------------------------------------------------------
# `any2md speaker remove`
# ---------------------------------------------------------------------------


class TestSpeakerRemove:
    def test_remove_nonexistent_speaker_exits_nonzero(self, db_path):
        open_catalog(db_path)
        result = runner.invoke(speaker_app, ["remove", "Nobody", "--force", "--db", db_path])
        assert result.exit_code != 0

    def test_remove_with_force_deletes_speaker(self, seeded_db):
        db_path, conn = seeded_db
        result = runner.invoke(speaker_app, ["remove", "Alice", "--force", "--db", db_path])
        assert result.exit_code == 0
        assert "Alice" in result.output
        # Verify deleted from catalog
        assert get_speaker_by_name(conn, "Alice") is None

    def test_remove_with_force_cascades_enrollments(self, seeded_db):
        db_path, conn = seeded_db
        alice = get_speaker_by_name(conn, "Alice")
        alice_id = alice["id"]
        runner.invoke(speaker_app, ["remove", "Alice", "--force", "--db", db_path])
        rows = conn.execute(
            "SELECT COUNT(*) as cnt FROM enrollments WHERE speaker_id = ?", (alice_id,)
        ).fetchone()
        assert rows["cnt"] == 0

    def test_remove_json_output(self, seeded_db):
        db_path, _ = seeded_db
        result = runner.invoke(speaker_app, ["remove", "Bob", "--force", "--db", db_path, "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["deleted"] is True
        assert data["speaker"] == "Bob"

    def test_remove_prompts_without_force(self, seeded_db, monkeypatch):
        """Without --force, should prompt for confirmation (we simulate 'n')."""
        db_path, conn = seeded_db
        # typer.confirm is called; simulate 'n' via input
        result = runner.invoke(speaker_app, ["remove", "Alice", "--db", db_path], input="n\n")
        assert result.exit_code == 0
        assert "Aborted" in result.output
        # Speaker should still exist
        assert get_speaker_by_name(conn, "Alice") is not None

    def test_remove_prompts_yes_deletes(self, seeded_db):
        db_path, conn = seeded_db
        result = runner.invoke(speaker_app, ["remove", "Alice", "--db", db_path], input="y\n")
        assert result.exit_code == 0
        assert get_speaker_by_name(conn, "Alice") is None


# ---------------------------------------------------------------------------
# `any2md speaker stats`
# ---------------------------------------------------------------------------


class TestSpeakerStats:
    def test_stats_unknown_speaker_exits_nonzero(self, db_path):
        open_catalog(db_path)
        result = runner.invoke(speaker_app, ["stats", "Ghost", "--db", db_path])
        assert result.exit_code != 0

    def test_stats_shows_enrollment_count(self, seeded_db):
        db_path, _ = seeded_db
        result = runner.invoke(speaker_app, ["stats", "Alice", "--db", db_path])
        assert result.exit_code == 0
        assert "Alice" in result.output
        assert "2" in result.output  # enrollment_count

    def test_stats_json_output(self, seeded_db):
        db_path, _ = seeded_db
        result = runner.invoke(speaker_app, ["stats", "Alice", "--db", db_path, "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["name"] == "Alice"
        assert data["enrollment_count"] == 2
        assert "mean_distance" in data
        assert "std_distance" in data

    def test_stats_shows_distance_fields(self, seeded_db):
        db_path, _ = seeded_db
        result = runner.invoke(speaker_app, ["stats", "Alice", "--db", db_path])
        assert "Mean distance" in result.output
        assert "Std distance" in result.output


# ---------------------------------------------------------------------------
# `any2md speaker gallery`
# ---------------------------------------------------------------------------


class TestSpeakerGallery:
    def test_gallery_unknown_speaker_exits_nonzero(self, db_path):
        open_catalog(db_path)
        result = runner.invoke(speaker_app, ["gallery", "Ghost", "--db", db_path])
        assert result.exit_code != 0

    def test_gallery_lists_enrollments(self, seeded_db):
        db_path, _ = seeded_db
        result = runner.invoke(speaker_app, ["gallery", "Alice", "--db", db_path])
        assert result.exit_code == 0
        assert "alice.wav" in result.output
        assert "alice2.wav" in result.output

    def test_gallery_json_output(self, seeded_db):
        db_path, _ = seeded_db
        result = runner.invoke(speaker_app, ["gallery", "Alice", "--db", db_path, "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 2
        source_files = [e["source_file"] for e in data]
        assert "alice.wav" in source_files
        assert "alice2.wav" in source_files

    def test_gallery_includes_source_type(self, seeded_db):
        db_path, _ = seeded_db
        result = runner.invoke(speaker_app, ["gallery", "Alice", "--db", db_path])
        assert "manual_full" in result.output

    def test_gallery_empty_speaker_shows_message(self, db_path):
        conn = open_catalog(db_path)
        add_speaker(conn, "Empty")
        result = runner.invoke(speaker_app, ["gallery", "Empty", "--db", db_path])
        assert result.exit_code == 0
        assert "No enrollments" in result.output


# ---------------------------------------------------------------------------
# `any2md speaker merge`
# ---------------------------------------------------------------------------


class TestSpeakerMerge:
    def test_merge_from_unknown_speaker_exits_nonzero(self, seeded_db):
        db_path, _ = seeded_db
        result = runner.invoke(speaker_app, ["merge", "Ghost", "Alice", "--db", db_path])
        assert result.exit_code != 0

    def test_merge_to_unknown_speaker_exits_nonzero(self, seeded_db):
        db_path, _ = seeded_db
        result = runner.invoke(speaker_app, ["merge", "Alice", "Ghost", "--db", db_path])
        assert result.exit_code != 0

    def test_merge_combines_enrollments(self, seeded_db):
        db_path, conn = seeded_db
        # Bob has 1, Alice has 2 — merge Bob into Alice
        result = runner.invoke(speaker_app, ["merge", "Bob", "Alice", "--db", db_path])
        assert result.exit_code == 0
        alice = get_speaker_by_name(conn, "Alice")
        assert alice["enrollment_count"] == 3  # 2 + 1

    def test_merge_deletes_source_speaker(self, seeded_db):
        db_path, conn = seeded_db
        runner.invoke(speaker_app, ["merge", "Bob", "Alice", "--db", db_path])
        assert get_speaker_by_name(conn, "Bob") is None

    def test_merge_output_message(self, seeded_db):
        db_path, _ = seeded_db
        result = runner.invoke(speaker_app, ["merge", "Bob", "Alice", "--db", db_path])
        assert "Bob" in result.output
        assert "Alice" in result.output

    def test_merge_json_output(self, seeded_db):
        db_path, _ = seeded_db
        result = runner.invoke(speaker_app, ["merge", "Bob", "Alice", "--db", db_path, "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["from_name"] == "Bob"
        assert data["to_name"] == "Alice"
        assert data["enrollment_count"] == 3

    def test_merge_records_audit_trail(self, seeded_db):
        db_path, conn = seeded_db
        runner.invoke(speaker_app, ["merge", "Bob", "Alice", "--db", db_path, "--reason", "same person"])
        row = conn.execute("SELECT * FROM speaker_merges").fetchone()
        assert row is not None
        assert row["reason"] == "same person"


# ---------------------------------------------------------------------------
# `any2md speaker add` — mocked (no real audio/model needed)
# ---------------------------------------------------------------------------


class TestSpeakerAdd:
    def _mock_add(self, db_path, audio_path, extra_args=None):
        """Helper: invoke speaker add with all deps mocked."""
        fake_emb = _fake_embedding()
        fake_model = MagicMock()
        fake_model.extract_embedding.return_value = fake_emb

        args = ["add", "TestSpeaker", "--audio", str(audio_path), "--db", db_path]
        if extra_args:
            args += extra_args

        with patch("any2md.speaker.load_speaker_model", return_value=fake_model), \
             patch("any2md.speaker.extract_embedding", return_value=fake_emb), \
             patch("any2md.yt.convert_audio_for_whisper", return_value=str(audio_path)):
            result = runner.invoke(speaker_app, args)
        return result

    def test_add_creates_new_speaker(self, db_path, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        result = self._mock_add(db_path, audio_file)
        assert result.exit_code == 0
        assert "TestSpeaker" in result.output

        conn = open_catalog(db_path)
        speaker = get_speaker_by_name(conn, "TestSpeaker")
        assert speaker is not None
        assert speaker["enrollment_count"] == 1

    def test_add_enrolls_existing_speaker(self, db_path, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        # First enrollment
        self._mock_add(db_path, audio_file)
        # Second enrollment — same speaker
        result = self._mock_add(db_path, audio_file)
        assert result.exit_code == 0

        conn = open_catalog(db_path)
        speaker = get_speaker_by_name(conn, "TestSpeaker")
        assert speaker["enrollment_count"] == 2

    def test_add_missing_audio_exits_nonzero(self, db_path, tmp_path):
        nonexistent = tmp_path / "missing.wav"
        result = runner.invoke(speaker_app, [
            "add", "TestSpeaker", "--audio", str(nonexistent), "--db", db_path
        ])
        assert result.exit_code != 0

    def test_add_json_output(self, db_path, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        result = self._mock_add(db_path, audio_file, extra_args=["--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["speaker"] == "TestSpeaker"
        assert data["enrollment_count"] == 1
        assert data["action"] == "created"

    def test_add_start_end_validation(self, db_path, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        result = runner.invoke(speaker_app, [
            "add", "TestSpeaker", "--audio", str(audio_file),
            "--start", "20", "--end", "10", "--db", db_path
        ])
        assert result.exit_code != 0
        assert "start" in result.output.lower() or "end" in result.output.lower()


# ---------------------------------------------------------------------------
# Catalog helper functions: get_enrollments, merge_speakers
# ---------------------------------------------------------------------------


class TestGetEnrollments:
    def test_returns_empty_list_for_new_speaker(self, db_path):
        conn = open_catalog(db_path)
        sid = add_speaker(conn, "Empty")
        result = get_enrollments(conn, sid)
        assert result == []

    def test_returns_all_enrollments(self, seeded_db):
        _, conn = seeded_db
        alice = get_speaker_by_name(conn, "Alice")
        result = get_enrollments(conn, alice["id"])
        assert len(result) == 2

    def test_enrollment_has_expected_fields(self, seeded_db):
        _, conn = seeded_db
        alice = get_speaker_by_name(conn, "Alice")
        enrollments = get_enrollments(conn, alice["id"])
        e = enrollments[0]
        assert "id" in e
        assert "source_file" in e
        assert "source_type" in e
        assert "created_at" in e
        assert "embedding" not in e  # should NOT include raw blob

    def test_ordered_by_created_at(self, db_path):
        conn = open_catalog(db_path)
        sid = add_speaker(conn, "Ordered")
        emb = _fake_embedding()
        enroll(conn, sid, emb, source_file="first.wav")
        enroll(conn, sid, emb, source_file="second.wav")
        enrollments = get_enrollments(conn, sid)
        assert enrollments[0]["source_file"] == "first.wav"
        assert enrollments[1]["source_file"] == "second.wav"


class TestMergeSpeakers:
    def test_raises_on_missing_from_speaker(self, db_path):
        conn = open_catalog(db_path)
        add_speaker(conn, "Alice")
        with pytest.raises(ValueError, match="Ghost"):
            merge_speakers(conn, "Ghost", "Alice")

    def test_raises_on_missing_to_speaker(self, db_path):
        conn = open_catalog(db_path)
        add_speaker(conn, "Alice")
        with pytest.raises(ValueError, match="Ghost"):
            merge_speakers(conn, "Alice", "Ghost")

    def test_merge_moves_enrollments(self, seeded_db):
        _, conn = seeded_db
        result = merge_speakers(conn, "Bob", "Alice")
        alice = get_speaker_by_name(conn, "Alice")
        assert alice["enrollment_count"] == 3

    def test_merge_deletes_from_speaker(self, seeded_db):
        _, conn = seeded_db
        merge_speakers(conn, "Bob", "Alice")
        assert get_speaker_by_name(conn, "Bob") is None

    def test_merge_records_audit_trail(self, seeded_db):
        _, conn = seeded_db
        result = merge_speakers(conn, "Bob", "Alice", reason="test merge")
        row = conn.execute(
            "SELECT * FROM speaker_merges WHERE id = ?", (result["merge_id"],)
        ).fetchone()
        assert row is not None
        assert row["reason"] == "test merge"
        assert row["from_speaker_id"] == result["from_id"]
        assert row["to_speaker_id"] == result["to_id"]

    def test_merge_return_dict_fields(self, seeded_db):
        _, conn = seeded_db
        result = merge_speakers(conn, "Bob", "Alice")
        assert "from_id" in result
        assert "to_id" in result
        assert "from_name" in result
        assert "to_name" in result
        assert "enrollment_count" in result
        assert "merge_id" in result

    def test_merge_recomputes_centroid(self, seeded_db):
        _, conn = seeded_db
        merge_speakers(conn, "Bob", "Alice")
        alice = get_speaker_by_name(conn, "Alice")
        # After merge, mean_distance should be updated (could be 0.0 with identical fake embeddings)
        assert alice["mean_distance"] is not None


# ---------------------------------------------------------------------------
# `any2md speaker group` CLI subcommands
# ---------------------------------------------------------------------------


@pytest.fixture()
def group_db(tmp_path) -> tuple:
    """Return (db_path, conn) with Alice, Bob, Charlie enrolled and a 'Hosts' group."""
    db_path = str(tmp_path / "group_test.db")
    conn = open_catalog(db_path)
    add_speaker(conn, "Alice")
    add_speaker(conn, "Bob")
    add_speaker(conn, "Charlie")
    create_group(conn, "Hosts", member_names=["Alice", "Bob"])
    return db_path, conn


class TestGroupCreate:
    def test_create_group_success(self, group_db):
        db_path, conn = group_db
        result = runner.invoke(speaker_app, ["group", "create", "Panel", "--db", db_path])
        assert result.exit_code == 0
        assert "Panel" in result.output

    def test_create_group_with_members(self, group_db):
        db_path, conn = group_db
        result = runner.invoke(
            speaker_app,
            ["group", "create", "Panel", "--members", "Alice,Charlie", "--db", db_path],
        )
        assert result.exit_code == 0
        group = get_group(conn, "Panel")
        names = [m["name"] for m in group["members"]]
        assert "Alice" in names
        assert "Charlie" in names

    def test_create_group_nonexistent_member_exits_nonzero(self, group_db):
        db_path, _ = group_db
        result = runner.invoke(
            speaker_app,
            ["group", "create", "Panel", "--members", "Nobody", "--db", db_path],
        )
        assert result.exit_code != 0

    def test_create_group_duplicate_exits_nonzero(self, group_db):
        db_path, _ = group_db
        # "Hosts" already exists
        result = runner.invoke(speaker_app, ["group", "create", "Hosts", "--db", db_path])
        assert result.exit_code != 0

    def test_create_group_json_output(self, group_db):
        db_path, _ = group_db
        result = runner.invoke(
            speaker_app,
            ["group", "create", "Panel", "--members", "Alice", "--db", db_path, "--json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["group"] == "Panel"
        assert "group_id" in data
        assert "Alice" in data["members"]


class TestGroupList:
    def test_list_shows_groups(self, group_db):
        db_path, _ = group_db
        result = runner.invoke(speaker_app, ["group", "list", "--db", db_path])
        assert result.exit_code == 0
        assert "Hosts" in result.output

    def test_list_empty_shows_message(self, tmp_path):
        db_path = str(tmp_path / "empty.db")
        open_catalog(db_path)
        result = runner.invoke(speaker_app, ["group", "list", "--db", db_path])
        assert result.exit_code == 0
        assert "No groups" in result.output

    def test_list_json_output(self, group_db):
        db_path, _ = group_db
        result = runner.invoke(speaker_app, ["group", "list", "--db", db_path, "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        names = [g["name"] for g in data]
        assert "Hosts" in names

    def test_list_shows_member_count(self, group_db):
        db_path, _ = group_db
        result = runner.invoke(speaker_app, ["group", "list", "--db", db_path])
        assert "2" in result.output  # Hosts has 2 members


class TestGroupShow:
    def test_show_group_success(self, group_db):
        db_path, _ = group_db
        result = runner.invoke(speaker_app, ["group", "show", "Hosts", "--db", db_path])
        assert result.exit_code == 0
        assert "Alice" in result.output
        assert "Bob" in result.output

    def test_show_nonexistent_group_exits_nonzero(self, group_db):
        db_path, _ = group_db
        result = runner.invoke(speaker_app, ["group", "show", "NoSuchGroup", "--db", db_path])
        assert result.exit_code != 0

    def test_show_json_output(self, group_db):
        db_path, _ = group_db
        result = runner.invoke(speaker_app, ["group", "show", "Hosts", "--db", db_path, "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["name"] == "Hosts"
        assert isinstance(data["members"], list)
        names = [m["name"] for m in data["members"]]
        assert "Alice" in names
        assert "Bob" in names


class TestGroupDelete:
    def test_delete_group_with_force(self, group_db):
        db_path, conn = group_db
        result = runner.invoke(speaker_app, ["group", "delete", "Hosts", "--force", "--db", db_path])
        assert result.exit_code == 0
        assert get_group(conn, "Hosts") is None

    def test_delete_nonexistent_group_exits_nonzero(self, group_db):
        db_path, _ = group_db
        result = runner.invoke(speaker_app, ["group", "delete", "NoGroup", "--force", "--db", db_path])
        assert result.exit_code != 0

    def test_delete_prompts_without_force(self, group_db):
        db_path, conn = group_db
        result = runner.invoke(
            speaker_app, ["group", "delete", "Hosts", "--db", db_path], input="n\n"
        )
        assert "Aborted" in result.output
        assert get_group(conn, "Hosts") is not None

    def test_delete_json_output(self, group_db):
        db_path, _ = group_db
        result = runner.invoke(
            speaker_app, ["group", "delete", "Hosts", "--force", "--db", db_path, "--json"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["deleted"] is True
        assert data["group"] == "Hosts"

    def test_delete_does_not_remove_speakers(self, group_db):
        db_path, conn = group_db
        runner.invoke(speaker_app, ["group", "delete", "Hosts", "--force", "--db", db_path])
        assert get_speaker_by_name(conn, "Alice") is not None


class TestGroupAddMember:
    def test_add_member_success(self, group_db):
        db_path, conn = group_db
        result = runner.invoke(
            speaker_app, ["group", "add-member", "Hosts", "Charlie", "--db", db_path]
        )
        assert result.exit_code == 0
        group = get_group(conn, "Hosts")
        names = [m["name"] for m in group["members"]]
        assert "Charlie" in names

    def test_add_member_nonexistent_group_exits_nonzero(self, group_db):
        db_path, _ = group_db
        result = runner.invoke(
            speaker_app, ["group", "add-member", "NoGroup", "Alice", "--db", db_path]
        )
        assert result.exit_code != 0

    def test_add_member_nonexistent_speaker_exits_nonzero(self, group_db):
        db_path, _ = group_db
        result = runner.invoke(
            speaker_app, ["group", "add-member", "Hosts", "Nobody", "--db", db_path]
        )
        assert result.exit_code != 0

    def test_add_member_json_output(self, group_db):
        db_path, _ = group_db
        result = runner.invoke(
            speaker_app,
            ["group", "add-member", "Hosts", "Charlie", "--db", db_path, "--json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["group"] == "Hosts"
        assert data["speaker"] == "Charlie"
        assert data["action"] == "added"


class TestGroupRemoveMember:
    def test_remove_member_success(self, group_db):
        db_path, conn = group_db
        result = runner.invoke(
            speaker_app, ["group", "remove-member", "Hosts", "Alice", "--db", db_path]
        )
        assert result.exit_code == 0
        group = get_group(conn, "Hosts")
        names = [m["name"] for m in group["members"]]
        assert "Alice" not in names
        assert "Bob" in names

    def test_remove_member_not_in_group_exits_nonzero(self, group_db):
        db_path, _ = group_db
        result = runner.invoke(
            speaker_app, ["group", "remove-member", "Hosts", "Charlie", "--db", db_path]
        )
        assert result.exit_code != 0

    def test_remove_member_nonexistent_group_exits_nonzero(self, group_db):
        db_path, _ = group_db
        result = runner.invoke(
            speaker_app, ["group", "remove-member", "NoGroup", "Alice", "--db", db_path]
        )
        assert result.exit_code != 0

    def test_remove_member_nonexistent_speaker_exits_nonzero(self, group_db):
        db_path, _ = group_db
        result = runner.invoke(
            speaker_app, ["group", "remove-member", "Hosts", "Nobody", "--db", db_path]
        )
        assert result.exit_code != 0

    def test_remove_member_json_output(self, group_db):
        db_path, _ = group_db
        result = runner.invoke(
            speaker_app,
            ["group", "remove-member", "Hosts", "Alice", "--db", db_path, "--json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["group"] == "Hosts"
        assert data["speaker"] == "Alice"
        assert data["removed"] is True
