"""Tests for the P1 archive draft store (ArchiveStore + ArchiveRecorder)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from wechat_rpa.archive_recorder import ArchiveRecorder
from wechat_rpa.archive_store import ArchiveStore, normalize_stable_key
from wechat_rpa.visible_message_parser import VisibleMessage


@pytest.fixture()
def store(tmp_path: Path) -> ArchiveStore:
    return ArchiveStore(tmp_path / "weauto.sqlite", tmp_path / "blobs")


@pytest.fixture()
def recorder(store: ArchiveStore) -> ArchiveRecorder:
    return ArchiveRecorder(store)


MSG_TEXT = {
    "side": "other",
    "sender": "刮风6-602",
    "content_type": "text",
    "text": "内高2.1的车不知道行不行",
    "bbox": [120, 340, 80, 46],
    "fingerprint": "other|text|刮风6-602|内高2.1的车不知道行不行",
}


class TestArchiveStore:
    def test_migration_creates_v1_tables(self, store: ArchiveStore) -> None:
        assert store.applied_versions() == [1]
        names = {
            row[0]
            for row in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"schema_migrations", "sessions", "captures",
                "message_drafts", "blobs"} <= names
        # P1 standard: canonical tables must NOT exist yet.
        assert not ({"participants", "messages", "attachments"} & names)

    def test_migration_is_idempotent(self, tmp_path: Path) -> None:
        db_path = tmp_path / "weauto.sqlite"
        first = ArchiveStore(db_path, tmp_path / "blobs")
        first.close()
        second = ArchiveStore(db_path, tmp_path / "blobs")
        assert second.applied_versions() == [1]
        second.close()

    def test_get_or_create_session_is_stable_by_title(
        self, store: ArchiveStore
    ) -> None:
        first = store.get_or_create_session("源上九里业主群")
        again = store.get_or_create_session("源上九里业主群")
        assert first["session_id"] == again["session_id"]
        sessions = store.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["title"] == "源上九里业主群"

    def test_wal_mode_enabled(self, store: ArchiveStore) -> None:
        mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert str(mode).lower() == "wal"

    def test_put_blob_dedupes_by_content(
        self, store: ArchiveStore, tmp_path: Path
    ) -> None:
        data = b"png-bytes-here"
        sha1 = store.put_blob(data, mime_type="image/png")
        sha2 = store.put_blob(data, mime_type="image/png")
        assert sha1 == sha2
        path = store.blob_path(sha1)
        assert path.is_file()
        assert path.read_bytes() == data
        # Stored under blobs/sha256/<2-char prefix>/<sha>
        assert path.parent.name == sha1[:2]
        assert path.parent.parent.name == "sha256"

    def test_drafts_are_append_only(self, recorder: ArchiveRecorder) -> None:
        recorder.record_snapshot(
            session_title="团购群", messages=[MSG_TEXT, MSG_TEXT], window_id=1
        )
        recorder.record_snapshot(
            session_title="团购群", messages=[MSG_TEXT], window_id=1
        )
        session = recorder.store.list_sessions()[0]
        drafts = list(recorder.store.iter_drafts(session["session_id"]))
        # No dedup: every visible message of every capture is kept.
        assert len(drafts) == 3


class TestArchiveRecorder:
    def test_records_capture_session_and_draft(
        self, recorder: ArchiveRecorder
    ) -> None:
        recorder.record_snapshot(
            session_title="团购群",
            messages=[MSG_TEXT],
            window_id=7,
            source="realtime",
            captured_at=1_700_000_000,
            parser_source="ocr",
        )
        session = recorder.store.list_sessions()[0]
        assert session["kind"] == "unknown"
        drafts = list(recorder.store.iter_drafts(session["session_id"]))
        assert len(drafts) == 1
        draft = drafts[0]
        assert draft["sender_name"] == "刮风6-602"
        assert draft["content_kind"] == "text"
        assert draft["observed_at"] == 1_700_000_000
        assert draft["source"] == "realtime"
        assert draft["parser_source"] == "ocr"
        raw = json.loads(draft["raw_json"])
        for key in (
            "side", "sender", "content_type", "text", "bbox", "mentions",
            "parser_source", "image_hash", "image_path",
        ):
            assert key in raw

    def test_accepts_visible_message_dataclass(
        self, recorder: ArchiveRecorder
    ) -> None:
        message = VisibleMessage(
            side="other",
            sender="张三",
            content_type="text",
            text="我要两盒",
            bbox=[0, 0, 10, 10],
            fingerprint="fp-1",
            mentions=["店长"],
        )
        recorder.record_snapshot("私聊-张三", [message], window_id=1)
        session = recorder.store.list_sessions()[0]
        draft = next(recorder.store.iter_drafts(session["session_id"]))
        assert draft["sender_name"] == "张三"
        assert json.loads(draft["mentions_json"]) == ["店长"]

    def test_unknown_content_kind_maps_to_unknown(
        self, recorder: ArchiveRecorder
    ) -> None:
        recorder.record_snapshot(
            "群", [{**MSG_TEXT, "content_type": "mini_program"}], window_id=1
        )
        draft = next(recorder.store.iter_drafts(recorder.store.list_sessions()[0]["session_id"]))
        assert draft["content_kind"] == "unknown"

    def test_malformed_message_does_not_drop_the_rest(
        self, recorder: ArchiveRecorder
    ) -> None:
        class Junk:
            pass

        recorder.record_snapshot(
            "群", [Junk(), MSG_TEXT], window_id=1
        )
        drafts = list(
            recorder.store.iter_drafts(
                recorder.store.list_sessions()[0]["session_id"]
            )
        )
        assert len(drafts) == 1

    def test_image_crop_goes_into_cas(
        self, recorder: ArchiveRecorder, tmp_path: Path
    ) -> None:
        crop = tmp_path / "crop.png"
        crop.write_bytes(b"fake-png-data")
        recorder.record_snapshot(
            "群",
            [{**MSG_TEXT, "content_type": "image", "image_path": str(crop)}],
            window_id=1,
        )
        draft = next(recorder.store.iter_drafts(recorder.store.list_sessions()[0]["session_id"]))
        assert draft["blob_sha256"]
        assert recorder.store.blob_path(draft["blob_sha256"]).read_bytes() == (
            b"fake-png-data"
        )

    def test_missing_image_file_keeps_draft_without_blob(
        self, recorder: ArchiveRecorder
    ) -> None:
        recorder.record_snapshot(
            "群",
            [{**MSG_TEXT, "content_type": "image",
              "image_path": "/nonexistent/crop.png"}],
            window_id=1,
        )
        draft = next(recorder.store.iter_drafts(recorder.store.list_sessions()[0]["session_id"]))
        assert draft["blob_sha256"] is None

    def test_group_target_fields_null_when_absent(
        self, recorder: ArchiveRecorder
    ) -> None:
        recorder.record_snapshot("群", [MSG_TEXT], window_id=1)
        draft = next(recorder.store.iter_drafts(recorder.store.list_sessions()[0]["session_id"]))
        assert draft["mentions_json"] is None
        assert draft["quoted_text"] is None


class TestBackfillJsonl:
    def test_import_keeps_every_line_and_extra_fields(
        self, recorder: ArchiveRecorder, tmp_path: Path
    ) -> None:
        payload_base = {
            "schema": "weauto_backfill_v1",
            "session_title": "团购群",
            "sender": "李四",
            "side": "other",
            "content_type": "text",
            "text": "拼单还差一位",
            "mentions": [],
            "ui_time_text": "昨天 21:04",
            "inferred_timestamp": "2026-11-05T21:04:00",
            "timestamp_confidence": 0.7,
            "source_page": 3,
            "observed_at": 1_700_000_100,
            "fingerprint": "fp-backfill-1",
        }
        lines = [
            json.dumps(payload_base, ensure_ascii=False),
            # Same fingerprint from a later page: kept, never merged in P1.
            json.dumps(
                {**payload_base, "source_page": 5,
                 "inferred_timestamp": "2026-11-05T21:04:30",
                 "timestamp_confidence": 0.9},
                ensure_ascii=False,
            ),
            json.dumps({"schema": "some_other_v9", "text": "ignored"}),
            "not-json",
        ]
        jsonl = tmp_path / "团购群.jsonl"
        jsonl.write_text("\n".join(lines), encoding="utf-8")

        stored = recorder.record_backfill_jsonl(jsonl)
        assert stored == 2

        session = recorder.store.list_sessions()[0]
        drafts = list(recorder.store.iter_drafts(session["session_id"]))
        assert len(drafts) == 2
        assert all(d["source"] == "backfill" for d in drafts)
        raw_last = json.loads(drafts[-1]["raw_json"])
        assert raw_last["source_page"] == 5
        assert raw_last["inferred_timestamp"] == "2026-11-05T21:04:30"
        assert raw_last["timestamp_confidence"] == 0.9
        # Per-line observed_at lands on the draft, not just the capture.
        assert drafts[0]["observed_at"] == 1_700_000_100

    def test_missing_file_returns_zero(
        self, recorder: ArchiveRecorder, tmp_path: Path
    ) -> None:
        assert recorder.record_backfill_jsonl(tmp_path / "nope.jsonl") == 0


class TestStableKey:
    def test_normalization(self) -> None:
        assert normalize_stable_key("  团购　群 ") == "团购 群"
        assert normalize_stable_key("") == ""
