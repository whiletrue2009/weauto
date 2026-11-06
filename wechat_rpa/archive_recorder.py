"""ArchiveRecorder: the single write entry point into the archive.

Converts parser snapshots (dict or VisibleMessage dataclass) into immutable
draft rows. No dedup, no merging, no sender normalization — every visible
message in every capture becomes a draft row (P1 append-only standard,
docs/12-archive-storage-standard.md).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .archive_store import ArchiveStore

# Allowed canonical content kinds (CHECK constraint in schema v1).
CONTENT_KINDS = {
    "text", "image", "file", "video", "voice", "link", "sticker", "system",
    "unknown",
}

RAW_JSON_REQUIRED_KEYS = (
    "side",
    "sender",
    "content_type",
    "text",
    "bbox",
    "mentions",
    "parser_source",
    "image_hash",
    "image_path",
)

# Optional per-message keys copied into raw_json when present. Covers
# vision annotations (vision_text) and the backfill producer's extra time
# provenance fields (inferred_timestamp / timestamp_confidence / source_page).
RAW_JSON_EXTRA_KEYS = (
    "ui_time_text",
    "quoted_text",
    "quote_text",
    "vision_text",
    "inferred_timestamp",
    "timestamp_confidence",
    "source_page",
)


def _message_to_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return dict(message)
    if is_dataclass(message):
        return asdict(message)
    raise TypeError(f"unsupported message payload type: {type(message)!r}")


class ArchiveRecorder:
    """Append-only recorder. Safe to call from the bot capture loop."""

    def __init__(
        self,
        store: ArchiveStore,
        *,
        save_image_crops: bool = True,
    ) -> None:
        self.store = store
        self.save_image_crops = bool(save_image_crops)

    def record_snapshot(
        self,
        session_title: str,
        messages: list[Any],
        window_id: int,
        source: str = "realtime",
        captured_at: int | None = None,
        parser_source: str = "ocr",
        session_kind: str | None = None,
    ) -> None:
        """Record one capture round: a capture row plus one draft per message."""
        observed = int(captured_at if captured_at is not None else time.time())
        session = self.store.get_or_create_session(
            session_title, kind=session_kind or "unknown"
        )
        capture_id = self.store.insert_capture(
            session_id=session["session_id"],
            window_id=window_id,
            captured_at=observed,
            source=source,
            parser_source=parser_source,
        )
        drafts: list[dict[str, Any]] = []
        for raw in messages:
            try:
                drafts.append(
                    self._build_draft(
                        raw,
                        session_id=session["session_id"],
                        capture_id=capture_id,
                        window_id=window_id,
                        source=source,
                        observed_at=observed,
                        parser_source=parser_source,
                    )
                )
            except Exception:  # noqa: BLE001, S112 - isolate malformed rows
                # A single malformed message must not drop the rest of the
                # capture; the failure is visible to the caller via counters.
                continue
        self.store.insert_drafts(drafts)

    def record_backfill_jsonl(
        self, jsonl_path: str | Path, *, schema: str = "weauto_backfill_v1"
    ) -> int:
        """Import a backfill JSONL file (one message per line) as drafts.

        Every line is imported as its own draft — including duplicate
        fingerprints from repeated pages; picking the "latest" line per
        fingerprint is cleaning-phase work. Lines whose ``schema`` differs
        from the expected value are skipped. Payloads are grouped by
        session_title so each session gets one capture row; per-line
        ``observed_at`` is kept on the draft itself. Returns the number of
        drafts stored.
        """
        path = Path(jsonl_path)
        if not path.is_file():
            return 0
        grouped: dict[str, list[dict[str, Any]]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                if str(payload.get("schema", schema)) != schema:
                    continue
                title = str(payload.get("session_title", "")) or "unknown"
                grouped.setdefault(title, []).append(payload)

        stored = 0
        for title, payloads in grouped.items():
            observed_values = [
                _epoch_or_none(p.get("observed_at")) for p in payloads
            ]
            known = [v for v in observed_values if v is not None]
            self.record_snapshot(
                session_title=title,
                messages=payloads,
                window_id=int(payloads[0].get("window_id", -1) or -1),
                source="backfill",
                captured_at=max(known) if known else None,
                parser_source=str(payloads[0].get("parser_source", "ocr")),
                session_kind=payloads[0].get("session_kind"),
            )
            stored += len(payloads)
        return stored

    # -- draft construction ----------------------------------------------

    def _build_draft(
        self,
        raw: Any,
        *,
        session_id: str,
        capture_id: str,
        window_id: int,
        source: str,
        observed_at: int,
        parser_source: str,
    ) -> dict[str, Any]:
        message = _message_to_dict(raw)
        # Backfill payloads carry their own per-line observation time.
        per_message_observed = _epoch_or_none(message.get("observed_at"))
        side = str(message.get("side", "unknown")) or "unknown"
        if side not in {"self", "other", "unknown"}:
            side = "unknown"
        content_kind = str(message.get("content_type", "")).strip().lower()
        if content_kind not in CONTENT_KINDS:
            content_kind = "unknown"

        text = message.get("text")
        text = str(text) if text else None

        mentions = message.get("mentions") or []
        mentions_json = (
            json.dumps(mentions, ensure_ascii=False) if mentions else None
        )

        quoted = message.get("quoted_text") or message.get("quote_text")
        quoted_text = str(quoted) if quoted else None

        bbox = message.get("bbox") or []
        bbox_json = (
            json.dumps([int(v) for v in bbox], ensure_ascii=False)
            if bbox else None
        )

        effective_parser = str(
            message.get("parser_source", parser_source) or parser_source
        )

        blob_sha256 = self._store_image_crop(message, effective_parser)

        raw_payload: dict[str, Any] = {
            "side": side,
            "sender": str(message.get("sender", "") or ""),
            "content_type": content_kind,
            "text": text or "",
            "bbox": bbox,
            "mentions": list(mentions),
            "parser_source": effective_parser,
            "image_hash": str(message.get("image_hash", "") or ""),
            "image_path": str(message.get("image_path", "") or ""),
        }
        for key in RAW_JSON_EXTRA_KEYS:
            value = message.get(key)
            if value not in (None, ""):
                raw_payload[key] = value

        confidence = _as_float(message.get("confidence"))
        ocr_confidence = _as_float(message.get("ocr_confidence"))

        return {
            "draft_id": _new_draft_id(),
            "session_id": session_id,
            "capture_id": capture_id,
            "window_id": int(window_id),
            "source": source,
            "observed_at": (
                per_message_observed
                if per_message_observed is not None
                else observed_at
            ),
            "ui_time_text": _optional_text(message.get("ui_time_text")),
            "side": side,
            "sender_name": _optional_text(message.get("sender")),
            "content_kind": content_kind,
            "text": text,
            "mentions_json": mentions_json,
            "quoted_text": quoted_text,
            "bbox": bbox_json,
            "fingerprint": _optional_text(message.get("fingerprint")),
            "parser_source": effective_parser,
            "ocr_confidence": ocr_confidence,
            "confidence": confidence,
            "blob_sha256": blob_sha256,
            "raw_json": json.dumps(raw_payload, ensure_ascii=False),
        }

    def _store_image_crop(
        self, message: dict[str, Any], parser_source: str
    ) -> str | None:
        """Copy the image crop file into CAS storage, returning its sha256."""
        if not self.save_image_crops:
            return None
        image_path = str(message.get("image_path", "") or "")
        if not image_path:
            return None
        path = Path(image_path)
        if not path.is_file():
            return None
        try:
            data = path.read_bytes()
        except OSError:
            return None
        # The draft row trusts only the CAS digest; the parser's own crop
        # hash (different recipe) stays available inside raw_json.
        return self.store.put_blob(data, mime_type="image/png")


def _new_draft_id() -> str:
    return f"dft_{uuid.uuid4().hex[:16]}"


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _epoch_or_none(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None
