"""Test for the static HTML review exporter."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import importlib.util

import pytest

from wechat_rpa.archive_recorder import ArchiveRecorder
from wechat_rpa.archive_store import ArchiveStore

_spec = importlib.util.spec_from_file_location(
    "export_archive_review",
    REPO_ROOT / "scripts" / "export_archive_review.py",
)
export_archive_review = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(export_archive_review)


@pytest.fixture()
def populated_store(tmp_path: Path) -> ArchiveStore:
    store = ArchiveStore(tmp_path / "weauto.sqlite", tmp_path / "blobs")
    recorder = ArchiveRecorder(store)
    recorder.record_snapshot(
        session_title="团购群",
        messages=[
            {
                "side": "other",
                "sender": "李四 <script>",
                "content_type": "text",
                "text": "拼单还差一位",
                "bbox": [1, 2, 3, 4],
                "fingerprint": "fp-1",
            },
            {
                "side": "self",
                "sender": "",
                "content_type": "unknown",
                "text": "",
                "bbox": [],
                "fingerprint": "fp-2",
            },
        ],
        window_id=1,
        captured_at=1_700_000_000,
    )
    return store


def test_export_generates_index_and_session_pages(
    populated_store: ArchiveStore, tmp_path: Path
) -> None:
    out_dir = tmp_path / "review"
    code = export_archive_review.main(
        ["--db", str(tmp_path / "weauto.sqlite"),
         "--blob-dir", str(tmp_path / "blobs"),
         "--out", str(out_dir)]
    )
    assert code == 0
    index = (out_dir / "index.html").read_text(encoding="utf-8")
    session_page = (out_dir / "session_团购群.html").read_text(encoding="utf-8")
    assert "团购群" in index
    assert "李四" in session_page
    assert "拼单还差一位" in session_page
    # HTML injection must be escaped (the page itself legitimately contains
    # a <script> tag for the filter JS, so match against the sender value).
    assert "李四 <script>" not in session_page
    assert "李四 &lt;script&gt;" in session_page

    populated_store.close()


def test_export_reports_missing_db(tmp_path: Path) -> None:
    code = export_archive_review.main(
        ["--db", str(tmp_path / "missing.sqlite"),
         "--blob-dir", str(tmp_path / "blobs"),
         "--out", str(tmp_path / "review")]
    )
    assert code == 1
