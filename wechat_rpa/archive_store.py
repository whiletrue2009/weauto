"""ArchiveStore: append-only SQLite draft store for captured WeChat messages.

Implements the P1 storage standard from docs/12-archive-storage-standard.md:
- SQLite + WAL, foreign keys on, busy timeout 5000ms.
- Versioned schema via schema_migrations (v1 = draft-only tables).
- Content-addressed blob storage under blobs/sha256/<xx>/<sha256>.

Writes are append-only: drafts are never merged or dropped here. Cleaning
(dedup, sender normalization) happens in a later phase via new migrations.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import tempfile
import threading
import time
import unicodedata
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

_SESSION_KINDS = {"direct", "group", "unknown"}

_MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE IF NOT EXISTS sessions (
      session_id    TEXT PRIMARY KEY,
      stable_key    TEXT NOT NULL UNIQUE,
      title         TEXT NOT NULL,
      kind          TEXT NOT NULL CHECK(kind IN ('direct','group','unknown')),
      first_seen_at INTEGER,
      last_seen_at  INTEGER,
      created_at    INTEGER NOT NULL,
      updated_at    INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS captures (
      capture_id      TEXT PRIMARY KEY,
      session_id      TEXT NOT NULL,
      window_id       INTEGER NOT NULL,
      captured_at     INTEGER NOT NULL,
      source          TEXT NOT NULL,
      parser_source   TEXT,
      screenshot_path TEXT,
      FOREIGN KEY(session_id) REFERENCES sessions(session_id)
    );

    CREATE TABLE IF NOT EXISTS message_drafts (
      draft_id       TEXT PRIMARY KEY,
      session_id     TEXT NOT NULL,
      capture_id     TEXT NOT NULL,
      window_id      INTEGER NOT NULL,
      source         TEXT NOT NULL,
      observed_at    INTEGER NOT NULL,
      ui_time_text   TEXT,
      side           TEXT NOT NULL CHECK(side IN ('self','other','unknown')),
      sender_name    TEXT,
      content_kind   TEXT NOT NULL CHECK(content_kind IN
                       ('text','image','file','video','voice','link','sticker','system','unknown')),
      text           TEXT,
      mentions_json  TEXT,
      quoted_text    TEXT,
      bbox           TEXT,
      fingerprint    TEXT,
      parser_source  TEXT,
      ocr_confidence REAL,
      confidence     REAL,
      blob_sha256    TEXT,
      raw_json       TEXT NOT NULL,
      created_at     INTEGER NOT NULL,
      FOREIGN KEY(capture_id) REFERENCES captures(capture_id)
    );
    CREATE INDEX IF NOT EXISTS idx_drafts_session_obs
      ON message_drafts(session_id, observed_at);

    CREATE TABLE IF NOT EXISTS blobs (
      sha256        TEXT PRIMARY KEY,
      size_bytes    INTEGER NOT NULL,
      mime_type     TEXT,
      first_seen_at INTEGER NOT NULL
    );
    """,
}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def normalize_stable_key(title: str) -> str:
    """Normalize a window title into a stable session key."""
    normalized = unicodedata.normalize("NFKC", str(title or ""))
    return re.sub(r"\s+", " ", normalized).strip()


class ArchiveStore:
    """SQLite draft archive with content-addressed blob storage."""

    def __init__(self, db_path: str | Path, blob_dir: str | Path) -> None:
        self.db_path = Path(db_path)
        self.blob_dir = Path(blob_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._migrate()

    # -- schema ---------------------------------------------------------

    def _migrate(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                  version INTEGER PRIMARY KEY,
                  applied_at INTEGER NOT NULL
                )
                """
            )
            applied = {
                int(row[0])
                for row in self._conn.execute(
                    "SELECT version FROM schema_migrations"
                )
            }
            for version in sorted(_MIGRATIONS):
                if version in applied:
                    continue
                self._conn.executescript(_MIGRATIONS[version])
                self._conn.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (version, int(time.time())),
                )

    def applied_versions(self) -> list[int]:
        rows = self._conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        return [int(row[0]) for row in rows]

    # -- sessions -------------------------------------------------------

    def get_or_create_session(
        self, title: str, kind: str = "unknown"
    ) -> dict[str, Any]:
        """Return the session row for a window title, creating it if needed."""
        if kind not in _SESSION_KINDS:
            kind = "unknown"
        stable_key = normalize_stable_key(title) or "unknown"
        now = int(time.time())
        with self._lock:
            row = self._conn.execute(
                "SELECT session_id, stable_key, title, kind, first_seen_at,"
                " last_seen_at FROM sessions WHERE stable_key = ?",
                (stable_key,),
            ).fetchone()
            if row is None:
                session_id = new_id("sess")
                self._conn.execute(
                    "INSERT INTO sessions(session_id, stable_key, title, kind,"
                    " first_seen_at, last_seen_at, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (session_id, stable_key, str(title or stable_key), kind,
                     now, now, now, now),
                )
                self._conn.commit()
                return {
                    "session_id": session_id,
                    "stable_key": stable_key,
                    "title": str(title or stable_key),
                    "kind": kind,
                    "first_seen_at": now,
                    "last_seen_at": now,
                }
            session_id = str(row[0])
            self._conn.execute(
                "UPDATE sessions SET last_seen_at = ?, title = ?"
                " WHERE session_id = ?",
                (now, str(title or row[2]), session_id),
            )
            self._conn.commit()
            return {
                "session_id": session_id,
                "stable_key": stable_key,
                "title": str(title or row[2]),
                "kind": str(row[3]),
                "first_seen_at": row[4],
                "last_seen_at": now,
            }

    # -- captures / drafts ----------------------------------------------

    def insert_capture(
        self,
        session_id: str,
        window_id: int,
        captured_at: int,
        source: str,
        parser_source: str | None = None,
        screenshot_path: str | None = None,
    ) -> str:
        capture_id = new_id("cap")
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO captures(capture_id, session_id, window_id,"
                " captured_at, source, parser_source, screenshot_path)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (capture_id, session_id, int(window_id), int(captured_at),
                 str(source), parser_source, screenshot_path),
            )
        return capture_id

    def insert_drafts(self, drafts: list[dict[str, Any]]) -> int:
        """Append draft rows. Returns the number of rows inserted."""
        if not drafts:
            return 0
        now = int(time.time())
        with self._lock, self._conn:
            self._conn.executemany(
                """
                INSERT INTO message_drafts(
                  draft_id, session_id, capture_id, window_id, source,
                  observed_at, ui_time_text, side, sender_name, content_kind,
                  text, mentions_json, quoted_text, bbox, fingerprint,
                  parser_source, ocr_confidence, confidence, blob_sha256,
                  raw_json, created_at
                ) VALUES (
                  :draft_id, :session_id, :capture_id, :window_id, :source,
                  :observed_at, :ui_time_text, :side, :sender_name,
                  :content_kind, :text, :mentions_json, :quoted_text, :bbox,
                  :fingerprint, :parser_source, :ocr_confidence, :confidence,
                  :blob_sha256, :raw_json, :created_at
                )
                """,
                [{**draft, "created_at": now} for draft in drafts],
            )
        return len(drafts)

    # -- blobs ----------------------------------------------------------

    def blob_path(self, sha256_hex: str) -> Path:
        return self.blob_dir / "sha256" / sha256_hex[:2] / sha256_hex

    def put_blob(self, data: bytes, mime_type: str | None = None) -> str:
        """Store bytes under blobs/sha256/, returning the hex digest.

        Identical content is written once (CAS); the blob row insert is
        idempotent.
        """
        sha256_hex = hashlib.sha256(data).hexdigest()
        target = self.blob_path(sha256_hex)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                dir=str(target.parent), suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                os.replace(tmp_name, target)
            except BaseException:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO blobs(sha256, size_bytes, mime_type,"
                " first_seen_at) VALUES (?, ?, ?, ?)",
                (sha256_hex, len(data), mime_type, int(time.time())),
            )
        return sha256_hex

    def blob_exists(self, sha256_hex: str) -> bool:
        return self.blob_path(sha256_hex).exists()

    # -- queries (for the review exporter) --------------------------------

    def list_sessions(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT s.session_id, s.stable_key, s.title, s.kind,
                   s.first_seen_at, s.last_seen_at,
                   COUNT(d.draft_id) AS draft_count
            FROM sessions s
            LEFT JOIN message_drafts d ON d.session_id = s.session_id
            GROUP BY s.session_id
            ORDER BY s.last_seen_at DESC
            """
        ).fetchall()
        return [
            {
                "session_id": row[0],
                "stable_key": row[1],
                "title": row[2],
                "kind": row[3],
                "first_seen_at": row[4],
                "last_seen_at": row[5],
                "draft_count": row[6],
            }
            for row in rows
        ]

    def iter_drafts(self, session_id: str) -> Iterator[dict[str, Any]]:
        cursor = self._conn.execute(
            """
            SELECT draft_id, capture_id, window_id, source, observed_at,
                   ui_time_text, side, sender_name, content_kind, text,
                   mentions_json, quoted_text, bbox, fingerprint,
                   parser_source, ocr_confidence, confidence, blob_sha256,
                   raw_json, created_at
            FROM message_drafts
            WHERE session_id = ?
            ORDER BY observed_at ASC, rowid ASC
            """,
            (session_id,),
        )
        columns = [
            "draft_id", "capture_id", "window_id", "source", "observed_at",
            "ui_time_text", "side", "sender_name", "content_kind", "text",
            "mentions_json", "quoted_text", "bbox", "fingerprint",
            "parser_source", "ocr_confidence", "confidence", "blob_sha256",
            "raw_json", "created_at",
        ]
        for row in cursor:
            yield dict(zip(columns, row))

    def close(self) -> None:
        with self._lock:
            self._conn.close()
