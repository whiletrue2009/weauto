"""Export archive drafts as static HTML review pages.

Reads data/archive/weauto.sqlite and generates:
  data/archive/review/index.html            session list + summary
  data/archive/review/session_<key>.html    per-session chat timeline
  data/archive/review/thumbs/<sha>.jpg      image thumbnails

Usage:
  python scripts/export_archive_review.py [--db PATH] [--blob-dir PATH]
      [--out PATH] [--session TITLE] [--limit N]

Pure stdlib + Pillow, no server: open the generated HTML directly in a browser.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from wechat_rpa.archive_store import ArchiveStore

THUMB_MAX_EDGE = 320

_PAGE_CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
       margin: 0; background: #f2f2f7; color: #1c1c1e; }
@media (prefers-color-scheme: dark) {
  body { background: #1c1c1e; color: #f2f2f7; }
}
header { padding: 14px 20px; background: #fff; border-bottom: 1px solid #ddd; }
@media (prefers-color-scheme: dark) { header { background: #2c2c2e; border-color: #444; } }
h1 { font-size: 18px; margin: 0 0 4px; }
.meta { font-size: 12px; opacity: 0.6; }
.filters { display: flex; flex-wrap: wrap; gap: 8px; padding: 10px 20px;
           position: sticky; top: 0; background: inherit; z-index: 2; }
.filters input, .filters select { padding: 6px 8px; border-radius: 8px;
    border: 1px solid #aaa; font-size: 13px; }
.timeline { padding: 12px 20px 40px; max-width: 860px; margin: 0 auto; }
.capture-group { border-radius: 12px; padding: 8px; margin: 10px 0; }
.draft { display: flex; margin: 4px 0; align-items: flex-start; }
.draft.self { justify-content: flex-end; }
.bubble { background: #fff; border-radius: 12px; padding: 8px 12px;
          max-width: 70%; box-shadow: 0 1px 2px rgba(0,0,0,0.08); }
@media (prefers-color-scheme: dark) { .bubble { background: #2c2c2e; } }
.draft.self .bubble { background: #95ec69; color: #17301c; }
@media (prefers-color-scheme: dark) { .draft.self .bubble { background: #3e5d3e; color: #eafcea; } }
.sender { font-size: 12px; opacity: 0.65; margin-bottom: 2px; }
.text { white-space: pre-wrap; word-break: break-word; font-size: 14px; }
.badge { display: inline-block; font-size: 11px; padding: 1px 7px;
         border-radius: 9px; background: #e5e5ea; color: #555; margin-left: 6px; }
@media (prefers-color-scheme: dark) { .badge { background: #48484a; color: #ccc; } }
.badge.image { background: #d6e9ff; color: #1a5390; }
.badge.file { background: #ffe9c9; color: #8a5b00; }
.badge.unknown { background: #ffd6d6; color: #901a1a; }
.thumb { max-width: 200px; max-height: 200px; border-radius: 8px; display: block; margin-top: 4px; }
.checkinfo { font-size: 11px; opacity: 0.5; margin-top: 4px;
             font-family: ui-monospace, Menlo, monospace; }
.statbar { display: flex; gap: 16px; font-size: 13px; padding: 4px 20px; opacity: 0.75; }
table.sessions { border-collapse: collapse; margin: 10px 20px; width: calc(100% - 40px); }
table.sessions td, table.sessions th { padding: 8px 10px; border-bottom: 1px solid #ddd;
     text-align: left; font-size: 14px; }
@media (prefers-color-scheme: dark) { table.sessions td, table.sessions th { border-color: #444; } }
.hidden { display: none; }
"""

_PAGE_JS = """
function applyFilters() {
  const sender = document.getElementById('f-sender').value;
  const kind = document.getElementById('f-kind').value;
  const kw = document.getElementById('f-kw').value.toLowerCase();
  const t0 = document.getElementById('f-t0').value;
  const t1 = document.getElementById('f-t1').value;
  let shown = 0;
  document.querySelectorAll('.draft').forEach(el => {
    const okS = !sender || el.dataset.sender === sender;
    const okK = !kind || el.dataset.kind === kind;
    const okW = !kw || (el.dataset.text || '').toLowerCase().includes(kw);
    const okT = (!t0 || el.dataset.obs >= t0) && (!t1 || el.dataset.obs <= t1 + '99');
    const ok = okS && okK && okW && okT;
    el.classList.toggle('hidden', !ok);
    if (ok) shown++;
  });
  document.getElementById('visible-count').textContent = shown;
}
document.addEventListener('input', e => { if (e.target.closest('.filters')) applyFilters(); });
"""


def _escape(value: object) -> str:
    return html.escape(str(value if value is not None else ""))


def _format_ts(ts: int | None) -> str:
    if not ts:
        return "-"
    return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")


def _capture_hue(capture_id: str) -> int:
    return sum(capture_id.encode("utf-8")) % 360


def _make_thumbnail(blob_path: Path, thumbs_dir: Path, sha: str) -> str | None:
    """Create (or reuse) a square-constrained JPEG thumbnail; return rel path."""
    out = thumbs_dir / f"{sha}.jpg"
    if out.exists():
        return f"thumbs/{sha}.jpg"
    try:
        from PIL import Image

        with Image.open(blob_path) as img:
            img = img.convert("RGB")
            img.thumbnail((THUMB_MAX_EDGE, THUMB_MAX_EDGE))
            img.save(out, "JPEG", quality=80)
        return f"thumbs/{sha}.jpg"
    except Exception:  # noqa: BLE001 - any PIL/IO failure just skips the thumb
        return None


def _render_draft(draft: dict, thumb_rel: str | None) -> str:
    kind = str(draft.get("content_kind") or "unknown")
    side = str(draft.get("side") or "unknown")
    sender = _escape(draft.get("sender_name") or side)
    text = draft.get("text") or ""
    observed = int(draft.get("observed_at") or 0)
    ui_time = draft.get("ui_time_text") or ""
    mentions = json.loads(draft["mentions_json"]) if draft.get("mentions_json") else []
    mention_html = "".join(
        f'<span class="badge">@{_escape(m)}</span>' for m in mentions
    )
    quoted = draft.get("quoted_text")
    quoted_html = (
        f'<div class="text" style="opacity:0.65;font-size:12px;">↩ {_escape(quoted)}</div>'
        if quoted
        else ""
    )
    image_html = ""
    if thumb_rel:
        image_html = (
            f'<img class="thumb" src="{_escape(thumb_rel)}" alt="image crop">'
        )
    fp = str(draft.get("fingerprint") or "")
    check = (
        f'cap={_escape(str(draft.get("capture_id", ""))[4:12])} '
        f'src={_escape(draft.get("source"))}/{_escape(draft.get("parser_source"))} '
        f'fp={_escape(fp[:32])} conf={_escape(draft.get("confidence"))}'
    )
    return f"""<div class="draft {side}"
      data-sender="{sender}" data-kind="{kind}"
      data-obs="{_escape(datetime.fromtimestamp(observed).isoformat())}"
      data-text="{_escape(text)}">
  <div class="bubble">
    <div class="sender">{sender}
      <span class="meta">{_escape(_format_ts(observed))}{f" · ui {_escape(ui_time)}" if ui_time else ""}</span>
      <span class="badge {kind}">{kind}</span>{mention_html}
    </div>
    {quoted_html}<div class="text">{_escape(text)}</div>{image_html}
    <div class="checkinfo">{check}</div>
  </div>
</div>"""


def export_session_page(
    store: ArchiveStore,
    session: dict,
    out_dir: Path,
    thumbs_dir: Path,
    generated_at: int,
) -> Path:
    drafts = list(store.iter_drafts(session["session_id"]))
    senders = Counter(
        str(d.get("sender_name") or d.get("side") or "?") for d in drafts
    )
    kinds = Counter(str(d.get("content_kind") or "?") for d in drafts)

    # Group consecutive drafts by capture for visual overlap inspection.
    capture_order: list[str] = []
    by_capture: dict[str, list[dict]] = {}
    for draft in drafts:
        cap = str(draft.get("capture_id"))
        if cap not in by_capture:
            by_capture[cap] = []
            capture_order.append(cap)
        by_capture[cap].append(draft)

    blocks: list[str] = []
    for idx, cap in enumerate(capture_order):
        hue = _capture_hue(cap)
        items = "".join(
            _render_draft(
                d,
                _thumb_for(store, d, thumbs_dir),
            )
            for d in by_capture[cap]
        )
        first_obs = by_capture[cap][0].get("observed_at")
        blocks.append(
            f'<div class="capture-group" style="background:hsla({hue},70%,50%,0.06);'
            f'border-left:3px solid hsla({hue},70%,45%,0.5);">'
            f'<div class="meta">capture #{idx + 1} · {_escape(cap)} · '
            f"{_escape(_format_ts(first_obs))}</div>{items}</div>"
        )

    sender_options = "".join(
        f'<option value="{_escape(s)}">{_escape(s)} ({n})</option>'
        for s, n in senders.most_common()
    )
    kind_options = "".join(
        f'<option value="{_escape(k)}">{_escape(k)} ({n})</option>'
        for k, n in kinds.most_common()
    )

    page = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(session["title"])} · WeAuto 归档复查</title>
<style>{_PAGE_CSS}</style></head><body>
<header>
  <h1>💬 {_escape(session["title"])}
    <span class="badge">{_escape(session["kind"])}</span></h1>
  <div class="meta">草稿 {len(drafts)} 条 · 捕获批次 {len(capture_order)} 个 ·
    首次 {_escape(_format_ts(session.get("first_seen_at")))} ·
    最近 {_escape(_format_ts(session.get("last_seen_at")))} ·
    导出于 {_escape(_format_ts(generated_at))}</div>
</header>
<div class="filters">
  <select id="f-sender"><option value="">全部发送者</option>{sender_options}</select>
  <select id="f-kind"><option value="">全部类型</option>{kind_options}</select>
  <input id="f-kw" type="search" placeholder="关键词…">
  <input id="f-t0" type="datetime-local" title="起始时间">
  <input id="f-t1" type="datetime-local" title="结束时间">
  <span class="meta">显示 <b id="visible-count">{len(drafts)}</b> / {len(drafts)}</span>
  <span class="meta">颜色 = 同一截图批次（重叠滚动产生的重复会以同色相邻出现）</span>
</div>
<div class="timeline">{''.join(blocks) if blocks else '<p style="opacity:0.5">暂无草稿。</p>'}</div>
<script>{_PAGE_JS}</script>
</body></html>"""

    filename = f"session_{session['stable_key']}.html"
    page_path = out_dir / filename
    page_path.write_text(page, encoding="utf-8")
    return page_path


def _thumb_for(store: ArchiveStore, draft: dict, thumbs_dir: Path) -> str | None:
    sha = draft.get("blob_sha256")
    if not sha:
        return None
    blob_path = store.blob_path(str(sha))
    if not blob_path.is_file():
        return None
    return _make_thumbnail(blob_path, thumbs_dir, str(sha))


def export_index(
    sessions: list[dict], out_dir: Path, generated_at: int
) -> Path:
    rows = "".join(
        f"""<tr>
  <td><a href="session_{_escape(s['stable_key'])}.html">{_escape(s["title"])}</a></td>
  <td><span class="badge">{_escape(s["kind"])}</span></td>
  <td>{s["draft_count"]}</td>
  <td>{_escape(_format_ts(s.get("first_seen_at")))}</td>
  <td>{_escape(_format_ts(s.get("last_seen_at")))}</td>
</tr>"""
        for s in sessions
    )
    total = sum(s["draft_count"] for s in sessions)
    page = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WeAuto 归档复查</title>
<style>{_PAGE_CSS}</style></head><body>
<header><h1>📥 WeAuto 消息草稿复查</h1>
<div class="meta">{len(sessions)} 个会话 · {total} 条草稿 ·
  导出于 {_escape(_format_ts(generated_at))}</div></header>
<table class="sessions">
<tr><th>会话</th><th>类型</th><th>草稿数</th><th>首次出现</th><th>最近活动</th></tr>
{rows}
</table></body></html>"""
    page_path = out_dir / "index.html"
    page_path.write_text(page, encoding="utf-8")
    return page_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/archive/weauto.sqlite")
    parser.add_argument("--blob-dir", default="data/archive/blobs")
    parser.add_argument("--out", default="data/archive/review")
    parser.add_argument("--session", default=None, help="only export this title")
    parser.add_argument("--limit", type=int, default=0, help="max drafts per session (0=all)")
    args = parser.parse_args(argv)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[archive-export] database not found: {db_path}")
        return 1

    store = ArchiveStore(db_path, Path(args.blob_dir))
    try:
        sessions = store.list_sessions()
        if args.session:
            sessions = [s for s in sessions if s["title"] == args.session]
        out_dir = Path(args.out)
        thumbs_dir = out_dir / "thumbs"
        out_dir.mkdir(parents=True, exist_ok=True)
        thumbs_dir.mkdir(parents=True, exist_ok=True)
        generated_at = int(time.time())
        export_index(sessions, out_dir, generated_at)
        for session in sessions:
            path = export_session_page(
                store, session, out_dir, thumbs_dir, generated_at
            )
            print(f"[archive-export] wrote {path}")
        print(f"[archive-export] index: {out_dir / 'index.html'}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
