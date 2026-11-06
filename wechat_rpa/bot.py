from __future__ import annotations

from collections import defaultdict
import concurrent.futures
import gc
import queue
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
import builtins
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import ssl
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING

import numpy as np
import pyautogui
from PIL import Image

from .action_processor import ActionProcessor
from .agent_store import ChatHistoryStore, MemoryStore, PeopleStore, SkillStore
from .archive_recorder import ArchiveRecorder
from .archive_store import ArchiveStore
from .bridge import BridgeClient
from .config import AppConfig
from .detector import ChatRowState, detect_chat_rows
from .detached_window_receiver import (
    capture_window_by_id,
    DetachedWindowInfo,
    list_detached_wechat_windows,
    request_screen_capture_access,
    safe_window_name,
    screen_capture_access_granted,
    set_detached_wechat_window_size,
    visible_window_owner_summary,
)
from .image_editing import ImageEditingError, ImageEditor
from .image_generation import ImageGenerationError, ImageGenerator
from .llm import LlmReplyGenerator, prepare_terminal_for_log_line
from .long_bridge import (
    LongBridgeClient,
    LongBridgeDelivery,
    LongBridgeOutbound,
    LongBridgeResult,
)
from .message_handler import MessageHandler
from .ocr import OcrEngine
from .people_aliases import PersonAliasResolver
from .qweather import QWeatherClient
from .sender import WeChatGuiSender
from .visible_message_parser import (
    normalize_message_fingerprint_text,
    VisibleChatSnapshot,
    VisibleMessageParser,
)
from .visible_message_state import VisibleMessageStateStore
from .window import WindowNotFoundError, get_front_window_bounds, screenshot_region

if TYPE_CHECKING:
    from .config import RegionRatio
    from .window import WindowBounds

pyautogui.PAUSE = 0.1

_TIME_RE = re.compile(r"^\d{1,2}:\d{2}$")

_BUILTIN_PRINT = builtins.print
_COLOR_ENABLED = bool(
    os.getenv("FORCE_COLOR", "").strip()
    or (
        sys.stdout.isatty()
        and (not os.getenv("NO_COLOR", "").strip())
        and os.getenv("TERM", "").lower() != "dumb"
    )
)
_COLOR_RESET = "\033[0m"
_LOG_COLOR_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\[start\]"), "\033[96m"),
    (re.compile(r"^\[init\]"), "\033[96m"),
    (re.compile(r"^\[cycle\]"), "\033[94m"),
    (re.compile(r"^\[row\]"), "\033[90m"),
    (re.compile(r"^\[event\]"), "\033[95m"),
    (re.compile(r"^\[msg-"), "\033[95m"),
    (re.compile(r"^\[batch-"), "\033[36m"),
    (re.compile(r"^\[route\]"), "\033[36m"),
    (re.compile(r"^\[decision\]"), "\033[35m"),
    (re.compile(r"^\[cooldown\]"), "\033[33m"),
    (re.compile(r"^\[queue"), "\033[94m"),
    (re.compile(r"^\[long-bridge\]"), "\033[96m"),
    (re.compile(r"^\[focus-"), "\033[36m"),
    (re.compile(r"^\[ctx\]"), "\033[97m"),
    (re.compile(r"^\[ocr"), "\033[96m"),
    (re.compile(r"^\[vision\]"), "\033[36m"),
    (re.compile(r"^\[llm\]"), "\033[35m"),
    (re.compile(r"^\[agent\]"), "\033[96m"),
    (re.compile(r"^\[heartbeat\]"), "\033[96m"),
    (re.compile(r"^\[action\]"), "\033[92m"),
    (re.compile(r"^\[sent\]"), "\033[92m"),
    (re.compile(r"^\[dry-run\]"), "\033[93m"),
    (re.compile(r"^\[reply-repeat\]"), "\033[33m"),
    (re.compile(r"^\[memory"), "\033[32m"),
    (re.compile(r"^\[admin-cmd\]"), "\033[95m"),
    (re.compile(r"^\[skip-"), "\033[90m"),
    (re.compile(r"^\[idle\]"), "\033[90m"),
    (re.compile(r"^\[error\]"), "\033[91m"),
    (re.compile(r"^\[fatal\]"), "\033[91m"),
    (re.compile(r"^Traceback \(most recent call last\):"), "\033[91m"),
    (re.compile(r"^[A-Za-z_][A-Za-z0-9_]*Error:"), "\033[91m"),
    (re.compile(r"^KeyboardInterrupt$"), "\033[91m"),
    (re.compile(r"^\[warn\]"), "\033[33m"),
]


def _colorize_log_line(text: str) -> str:
    if (not _COLOR_ENABLED) or (not text):
        return text
    clean = str(text)
    for pattern, color in _LOG_COLOR_RULES:
        if pattern.match(clean):
            return f"{color}{clean}{_COLOR_RESET}"
    return clean


def print(*args, **kwargs):  # type: ignore[override]
    if not args:
        return _BUILTIN_PRINT(*args, **kwargs)
    sep = kwargs.get("sep", " ")
    merged = sep.join(str(x) for x in args)
    file_obj = kwargs.get("file", sys.stdout)
    if file_obj in (None, sys.stdout, sys.stderr):
        prepare_terminal_for_log_line()
        merged = _colorize_log_line(merged)
    out_kwargs = dict(kwargs)
    out_kwargs["sep"] = ""
    return _BUILTIN_PRINT(merged, **out_kwargs)


@dataclass
class RowMemory:
    session_key: str
    fingerprint: str
    preview_norm: str
    last_sent_norm: str
    has_unread_badge: bool
    pending_unread: bool
    pending_normal: bool
    has_mention: bool
    pending_mention: bool
    last_replied_at: float


@dataclass
class ChatContextSnapshot:
    text: str
    last_side: str
    last_line: str
    last_user_message: str = ""
    recent_messages: list[str] | None = None
    recent_structured: list[dict] | None = None
    chat_records: list[dict] | None = None
    memory_summary: str = ""
    memory_time_hints: list[str] | None = None
    memory_people: list[dict] | None = None
    memory_facts: list[str] | None = None
    memory_events: list[str] | None = None
    memory_relations: list[dict] | None = None
    environment_text: str = ""
    schema: str = ""
    source: str = "vision"


@dataclass
class FocusResult:
    bounds: "WindowBounds"
    matched: bool
    resolved_row: ChatRowState | None
    seen_header: str = ""


@dataclass
class RecoverCaptureResult:
    bounds: "WindowBounds"
    title: str
    session_key: str
    parsed: int
    appended: int
    last_line: str = ""
    next_title: str = ""


@dataclass
class SessionState:
    short: list[str]
    history: list[dict]
    summary: str
    muted: bool
    titles: set[str]
    loaded: bool = True
    seen_content: set = field(default_factory=set)


class DetachedOcrProxy:
    """Owns the recycled capture+OCR worker subprocess.

    Replaces the in-process ``self.visible_message_parser.parse(...)`` call for
    the detached-window hot path. The worker holds all the native (Mach /
    CoreGraphics / cv2 / onnxruntime) memory that gc.collect() cannot reclaim;
    when the worker's RSS exceeds the configured limit it exits and this proxy
    spawns a fresh one. The main bot process therefore never accumulates that
    memory and never needs to restart.

    Requests are retried once with a fresh worker after transport failures,
    worker errors, or RSS-driven recycling. A second failure returns ``None``
    so the caller can skip the current scan cycle without running the leaking
    capture/OCR path in the main process.
    """

    def __init__(
        self,
        cfg: AppConfig,
        *,
        config_path: str = "config.toml",
        log_fn=None,
    ) -> None:
        self._cfg = cfg
        self._config_path = config_path
        self._log_fn = log_fn if log_fn is not None else print
        self._proc: subprocess.Popen | None = None
        self._ready = False
        self._fail_streak = 0
        self._recycle_count = 0
        self._retry_count = 0
        self._skipped_request_count = 0

    def _spawn(self) -> None:
        env = dict(os.environ)
        env["WEAUTO_CONFIG_PATH"] = str(self._config_path)
        env["WEAUTO_OCR_WORKER_MAX_RSS_MB"] = str(self._cfg.ocr_worker_max_rss_mb)
        # Detach from the bot's stdout so the worker's OCR diagnostics never
        # corrupt the bot's structured logs; the bot only reads its stdout.
        self._proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "wechat_rpa.ocr_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env,
        )
        self._ready = False
        # Wait for the worker's {"ready": true} IPC handshake.
        ready_timeout = max(2.0, float(self._cfg.ocr_worker_ready_timeout_sec))
        try:
            first = self._proc.stdout.readline()  # type: ignore[union-attr]
        except Exception:
            first = ""
        if self._proc.poll() is None and first.strip():
            try:
                payload = json.loads(first)
                self._ready = bool(payload.get("ready"))
            except (json.JSONDecodeError, ValueError):
                self._ready = False
        if not self._ready:
            # Worker failed its IPC handshake or died; tear down so the next
            # call respawns.
            self._terminate()
            self._log_fn(f"[ocr-worker] startup failed after {ready_timeout:.0f}s")

    def _terminate(self) -> None:
        proc = self._proc
        self._proc = None
        self._ready = False
        if proc is None:
            return
        for stream in (proc.stdin, proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass
        try:
            proc.terminate()
            proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _ensure_worker(self) -> bool:
        if self._proc is not None and self._proc.poll() is None and self._ready:
            return True
        self._terminate()
        self._spawn()
        return self._proc is not None and self._proc.poll() is None and self._ready

    def _recycle(self, *, reason: str) -> None:
        self._recycle_count += 1
        self._log_fn(f"[ocr-worker] recycled #{self._recycle_count} reason={reason}")
        self._terminate()

    def _request_once(self, request: dict) -> dict | None:
        """Send one JSON request and read one JSON response.

        Returns the decoded payload dict, or ``None`` on any transport failure
        (timeout, EOF, bad JSON). Handles worker RSS-exit by recycling and
        returning ``None``. Does NOT interpret business fields (snapshot/error).
        """
        if not self._ensure_worker():
            self._fail_streak += 1
            return None
        proc = self._proc
        assert proc is not None and proc.stdin is not None and proc.stdout is not None
        try:
            proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            self._recycle(reason="write_failed")
            self._fail_streak += 1
            return None

        timeout = max(2.0, float(self._cfg.ocr_worker_request_timeout_sec))
        # Use a watchdog thread to enforce the timeout without blocking forever.
        import threading as _threading

        line_holder: list[str] = []

        def _read() -> None:
            try:
                line_holder.append(proc.stdout.readline())
            except Exception:
                line_holder.append("")

        reader = _threading.Thread(target=_read, daemon=True)
        reader.start()
        reader.join(timeout=timeout)
        if reader.is_alive() or not line_holder or not line_holder[0].strip():
            # Timed out or EOF. Recycle so the next call gets a fresh worker.
            self._recycle(reason="timeout_or_eof")
            self._fail_streak += 1
            return None

        raw_line = line_holder[0]
        try:
            payload = json.loads(raw_line)
        except (json.JSONDecodeError, ValueError):
            self._recycle(reason="invalid_json")
            self._fail_streak += 1
            return None

        # Worker asked to exit (RSS limit hit): acknowledge and recycle.
        if payload.get("__exit__"):
            rss_mb = payload.get("rss_mb", "?")
            self._recycle(reason=f"rss_limit({rss_mb}MB)")
            self._fail_streak = 0
            return None

        return payload

    def _request(self, request: dict, *, validator=None) -> dict | None:
        """Run a worker request, rebuilding the worker and retrying once."""
        mode = str(request.get("mode", "parse") or "parse")
        for attempt in range(2):
            payload = self._request_once(request)
            valid_payload = bool(
                payload is not None
                and "error" not in payload
                and (validator is None or validator(payload))
            )
            if valid_payload:
                return payload
            if payload is not None:
                if "error" in payload:
                    error = str(payload.get("error", "worker_error") or "worker_error")
                    reason = f"worker_error({error[:80]})"
                else:
                    reason = f"invalid_response({mode})"
                self._recycle(reason=reason)
                self._fail_streak += 1
            if attempt == 0:
                self._retry_count += 1
                self._log_fn(
                    f"[ocr-worker] retry #{self._retry_count} mode={mode} "
                    "with fresh worker"
                )
        self._skipped_request_count += 1
        self._log_fn(
            f"[ocr-worker] request skipped #{self._skipped_request_count} "
            f"mode={mode} after retry"
        )
        return None

    def parse(
        self,
        window_id: int,
        *,
        title: str,
        image_output_dir: Path | None = None,
        include_debug: bool = False,
    ) -> VisibleChatSnapshot | None:
        request = {
            "mode": "parse",
            "window_id": int(window_id),
            "title": str(title or ""),
            "image_output_dir": str(image_output_dir) if image_output_dir else None,
            "include_debug": bool(include_debug),
        }
        payload = self._request(
            request,
            validator=lambda item: isinstance(item.get("snapshot"), dict),
        )
        if payload is None:
            return None

        snapshot_payload = payload.get("snapshot")
        assert isinstance(snapshot_payload, dict)

        self._fail_streak = 0
        try:
            return self._snapshot_from_dict(snapshot_payload)
        except Exception:
            return None

    def list_windows(self, app_name: str) -> list[dict] | None:
        """Enumerate detached WeChat windows via the worker.

        ``CGWindowListCopyWindowInfo`` leaks Mach message memory in the caller,
        so window enumeration must also run in the worker. Returns a list of
        window dicts (window_id/owner/title/x/y/width/height), or ``None`` on
        failure after the fresh-worker retry.
        """
        request = {
            "mode": "list_windows",
            "app_name": str(app_name or "WeChat"),
        }
        payload = self._request(
            request,
            validator=lambda item: isinstance(item.get("windows"), list),
        )
        if payload is None:
            return None
        windows = payload.get("windows")
        assert isinstance(windows, list)
        self._fail_streak = 0
        return [dict(w) for w in windows if isinstance(w, dict)]

    def capture_only(self, window_id: int, *, include_text_anchors: bool = False) -> tuple[Path, str, str, dict, list[dict]] | None:
        """Capture a window via the worker and return image path, hashes, size,
        and optional text anchors.

        This is the vision-path entry point: the worker does the leaky
        CoreGraphics capture and persists the PNG to a temp path it owns; the
        bot reads the file back with PIL (no Mach growth) for the vision LLM.
        The caller MUST delete ``image_path`` after use. Returns ``None`` on any
        worker failure (fail-open).
        """
        request = {
            "mode": "capture_only",
            "window_id": int(window_id),
            "include_text_anchors": bool(include_text_anchors),
        }
        payload = self._request(
            request,
            validator=lambda item: bool(
                item.get("image_path") and item.get("body_hash")
            ),
        )
        if payload is None:
            return None

        image_path_raw = payload.get("image_path")
        body_hash = payload.get("body_hash")
        stable_body_hash = payload.get("stable_body_hash") or body_hash
        assert image_path_raw and body_hash
        self._fail_streak = 0
        image_size = payload.get("image_size") or {}
        if not isinstance(image_size, dict):
            image_size = {}
        text_anchors = payload.get("text_anchors") or []
        if not isinstance(text_anchors, list):
            text_anchors = []
        return (
            Path(str(image_path_raw)),
            str(body_hash),
            str(stable_body_hash),
            dict(image_size),
            [dict(item) for item in text_anchors if isinstance(item, dict)],
        )

    @staticmethod
    def _snapshot_from_dict(payload: dict) -> VisibleChatSnapshot:
        messages = payload.get("messages") or []
        if not isinstance(messages, list):
            messages = []
        latest = payload.get("latest_message")
        debug = payload.get("debug")
        return VisibleChatSnapshot(
            schema=str(payload.get("schema", "weauto_visible_messages_v1")),
            window_id=int(payload.get("window_id", 0) or 0),
            title=str(payload.get("title", "") or ""),
            captured_at=float(payload.get("captured_at", 0.0) or 0.0),
            source=str(payload.get("source", "ocr_worker") or "ocr_worker"),
            image_size=dict(payload.get("image_size") or {"width": 0, "height": 0}),
            messages=[dict(m) for m in messages if isinstance(m, dict)],
            latest_message=dict(latest) if isinstance(latest, dict) else None,
            debug=dict(debug) if isinstance(debug, dict) else None,
        )

    def close(self) -> None:
        self._terminate()


class WeChatGuiRpaBot:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.ocr_worker_enabled = bool(getattr(cfg, "ocr_worker_enabled", False))
        self._strict_detached_worker = bool(
            self.ocr_worker_enabled
            and getattr(cfg, "receiver_mode", "detached_windows") == "detached_windows"
        )
        self.ocr_engine: OcrEngine | None = None
        self.visible_message_parser: VisibleMessageParser | None = None
        if not self._strict_detached_worker:
            self.ocr_engine = OcrEngine(cfg.ocr, log_fn=print)
            self.visible_message_parser = VisibleMessageParser(self.ocr_engine)
        self.llm_reply = LlmReplyGenerator(cfg.llm_reply, cfg.vision)
        self.llm_decision = LlmReplyGenerator(cfg.llm_decision, cfg.vision)
        self.llm_planner = LlmReplyGenerator(cfg.llm_planner, cfg.vision)
        self.llm_summary = LlmReplyGenerator(cfg.llm_summary, cfg.vision)
        self.llm_heartbeat = LlmReplyGenerator(cfg.llm_heartbeat, cfg.vision)
        self.llm = self.llm_reply
        self.image_generator = ImageGenerator(cfg.image_generation)
        self.image_editor = ImageEditor(cfg.image_editing)
        self.qweather = self._make_qweather_client()
        self.bridge_client = BridgeClient(cfg)
        self.long_bridge_client = LongBridgeClient(cfg)
        self.agent_memory = MemoryStore("data/memory")
        self.agent_skills = SkillStore("data/skills")
        self.agent_skills.cleanup()
        self.agent_people = PeopleStore("data/people")
        self.agent_people.cleanup()
        self.chat_history = ChatHistoryStore("data/chat_history")
        # Append-only draft archive (docs/12-archive-storage-standard.md).
        # Failures here must never block the reply pipeline.
        self.archive_recorder: "ArchiveRecorder | None" = None
        if getattr(getattr(cfg, "archive", None), "enabled", False):
            try:
                self.archive_recorder = ArchiveRecorder(
                    ArchiveStore(cfg.archive.path, cfg.archive.blob_dir),
                    save_image_crops=cfg.archive.save_image_crops,
                )
            except Exception as exc:
                print(f"[warn] archive init failed, archive disabled: {exc}")
        self.people_alias_resolver = PersonAliasResolver(
            cfg.people_aliases_path,
            enabled=cfg.people_aliases_enabled,
        )
        self.sender = WeChatGuiSender(cfg)
        self.heartbeat = None
        self.action_processor = ActionProcessor(self)
        self.message_handler = MessageHandler(self)
        # Recycled capture+OCR subprocess for the high-frequency detached-window
        # path. In strict detached mode the main process does not initialize an
        # OCR engine and never falls back to native capture/OCR.
        self.ocr_proxy: DetachedOcrProxy | None = None
        if self.ocr_worker_enabled:
            self.ocr_proxy = DetachedOcrProxy(cfg, log_fn=print)
        self.visible_message_state = VisibleMessageStateStore()
        self._detached_window_image_hashes: dict[int, str] = {}
        self._detached_window_text_anchors: dict[int, list[dict]] = {}
        self._detached_bootstrapped = False
        self._detached_watchdog_resume_window_ids: set[int] = set()
        self._state_lock = threading.RLock()
        self._send_lock = threading.Lock()
        self._visual_scan_paused_until = 0.0
        self._msg_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(8, max(2, os.cpu_count() or 4)),
            thread_name_prefix="msg",
        )
        self._session_queues: dict[int, queue.Queue] = {}
        self._session_busy: dict[int, bool] = {}
        self._baseline: dict[int, RowMemory] = {}
        # title_key -> (normalized_sent_text, sent_ts)
        self._sent_by_title: dict[str, tuple[str, float]] = {}
        self._latest_image_by_session: dict[str, str] = {}
        self._image_vision_by_path: dict[str, str] = {}
        self._sessions: dict[str, SessionState] = {}
        self._session_aliases: dict[str, str] = {}
        self._summary_turn_counter: dict[str, int] = defaultdict(int)
        self._memory_dirty = False
        self._memory_path = Path(self.cfg.memory_store_path)
        self._runtime_state_path = Path("data/runtime_state.json")
        self._session_index: dict[str, dict] = {}
        self._long_bridge_titles_by_id: dict[str, str] = {}
        self._last_normal_reply_at = 0.0
        self._workspace = None
        self._cycle = 0
        self._idle_streak = 0
        self._skip_first_action_pending = bool(self.cfg.skip_first_action_on_start)
        self._last_heartbeat_at = 0.0
        self._last_activity_at = 0.0
        self._last_memory_gc_at = 0.0
        self._memory_restart_pending = False
        self._mach_diag_inprocess_capture = 0
        self._mach_diag_inprocess_enum = 0
        self._mach_diag_screenshot_region = 0
        self._mach_diag_pyautogui = 0
        self._load_runtime_state()
        self._load_persistent_memory()
        self.long_bridge_client.set_outbound_handler(
            self._handle_long_bridge_outbound
        )
        self.long_bridge_client.start()

    def _to_np_rgb(self, pil_image) -> np.ndarray:
        rgb_image = None
        try:
            rgb_image = pil_image.convert("RGB")
            # Detach from PIL internal buffers so images can be closed promptly.
            return np.array(rgb_image, dtype=np.uint8, copy=True)
        finally:
            if rgb_image is not None:
                try:
                    rgb_image.close()
                except Exception:
                    pass
            try:
                pil_image.close()
            except Exception:
                pass

    def _ensure_local_ocr(self) -> tuple[OcrEngine, VisibleMessageParser]:
        if self.ocr_engine is None:
            self.ocr_engine = OcrEngine(self.cfg.ocr, log_fn=print)
        if self.visible_message_parser is None:
            self.visible_message_parser = VisibleMessageParser(self.ocr_engine)
        return self.ocr_engine, self.visible_message_parser

    @staticmethod
    def _current_rss_mb() -> int:
        try:
            proc = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(os.getpid())],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if proc.returncode == 0:
                kb = int((proc.stdout or "0").strip() or "0")
                return max(0, kb // 1024)
        except Exception:
            pass
        return 0

    def _has_pending_message_work(self) -> bool:
        with self._state_lock:
            if any(bool(v) for v in self._session_busy.values()):
                return True
            for q in self._session_queues.values():
                try:
                    if q.qsize() > 0:
                        return True
                except Exception:
                    if not q.empty():
                        return True
        return False

    def _pause_visual_scan(self, seconds: float) -> None:
        delay = max(0.0, float(seconds or 0.0))
        if delay <= 0:
            return
        with self._state_lock:
            self._visual_scan_paused_until = max(
                self._visual_scan_paused_until,
                time.time() + delay,
            )

    def _visual_scan_paused(self, now: float) -> bool:
        with self._state_lock:
            paused_until = self._visual_scan_paused_until
        return self._send_lock.locked() or now < paused_until

    def _maybe_run_memory_maintenance(self, now: float) -> None:
        interval = float(getattr(self.cfg, "memory_gc_interval_sec", 0.0) or 0.0)
        max_rss_mb = int(getattr(self.cfg, "memory_watchdog_max_rss_mb", 0) or 0)
        if interval <= 0 and max_rss_mb <= 0:
            return
        if (
            not self._memory_restart_pending
            and interval > 0
            and now - self._last_memory_gc_at < interval
        ):
            return
        self._last_memory_gc_at = now
        collected = gc.collect()
        rss_mb = self._current_rss_mb()
        if self.cfg.log_verbose and rss_mb:
            worker_diag = ""
            if self.ocr_proxy is not None:
                worker_diag = (
                    f" worker_recycles={self.ocr_proxy._recycle_count}"
                    f" worker_retries={self.ocr_proxy._retry_count}"
                    f" worker_skips={self.ocr_proxy._skipped_request_count}"
                )
            diag = (
                f"inproc_capture={getattr(self, '_mach_diag_inprocess_capture', 0)} "
                f"inproc_enum={getattr(self, '_mach_diag_inprocess_enum', 0)} "
                f"screenshot_region={getattr(self, '_mach_diag_screenshot_region', 0)} "
                f"pyautogui_ops={getattr(self, '_mach_diag_pyautogui', 0)}"
            )
            print(
                f"[memory] rss={rss_mb}MB gc_collected={collected} "
                f"{diag}{worker_diag}"
            )
        should_restart = max_rss_mb > 0 and rss_mb > max_rss_mb
        if should_restart:
            self._memory_restart_pending = True
        if self._memory_restart_pending:
            if self._has_pending_message_work():
                reason = (
                    f"rss={rss_mb}MB exceeds limit={max_rss_mb}MB"
                    if should_restart
                    else f"restart pending; rss={rss_mb}MB limit={max_rss_mb}MB"
                )
                print(
                    f"[memory] {reason}; "
                    "restart deferred until message/send queues are idle"
                )
                return
            print(
                f"[memory] rss={rss_mb}MB restart pending; "
                "queues idle, saving state and restarting process"
            )
            self._save_runtime_state(detached_watchdog_resume=True)
            self._save_persistent_memory()
            os.execv(sys.executable, [sys.executable, "-u", *sys.argv])

    @staticmethod
    def _coerce_runtime_timestamp(value: object) -> float:
        try:
            ts = float(value)
        except Exception:
            return 0.0
        if ts <= 0:
            return 0.0
        now = time.time()
        # Ignore impossible future values caused by clock changes or corrupt files.
        if ts > now + 300.0:
            return 0.0
        return ts

    def _load_runtime_state(self) -> None:
        path = getattr(self, "_runtime_state_path", Path("data/runtime_state.json"))
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception as exc:
            print(f"[warn] runtime state load failed: {exc}")
            return
        if not isinstance(raw, dict):
            return
        self._last_normal_reply_at = self._coerce_runtime_timestamp(raw.get("last_normal_reply_at"))
        self._last_heartbeat_at = self._coerce_runtime_timestamp(raw.get("last_heartbeat_at"))
        self._last_activity_at = self._coerce_runtime_timestamp(raw.get("last_activity_at"))
        restored_detached = self._restore_detached_watchdog_resume(raw)
        if self.cfg.log_verbose:
            print(
                "[runtime] loaded "
                f"normal={self._last_normal_reply_at:.0f} "
                f"heartbeat={self._last_heartbeat_at:.0f} "
                f"activity={self._last_activity_at:.0f} "
                f"path={path}"
            )
        if restored_detached:
            self._save_runtime_state()

    def _restore_detached_watchdog_resume(self, payload: dict) -> bool:
        if getattr(self.cfg, "receiver_mode", "detached_windows") != "detached_windows":
            return False
        resume = payload.get("detached_watchdog_resume")
        if not isinstance(resume, dict):
            return False
        saved_at = self._coerce_runtime_timestamp(resume.get("saved_at"))
        if saved_at <= 0 or time.time() - saved_at > 600.0:
            return False
        state = resume.get("visible_message_state")
        if not isinstance(state, dict) or not hasattr(self, "visible_message_state"):
            return False
        try:
            self.visible_message_state.load_state(state)
        except Exception as exc:
            print(f"[warn] detached watchdog resume load failed: {exc}")
            return False
        windows_raw = state.get("windows")
        if not isinstance(windows_raw, dict):
            return False
        restored_ids: set[int] = set()
        for key, item in windows_raw.items():
            try:
                window_id = int(key)
            except Exception:
                continue
            if isinstance(item, dict) and item.get("order"):
                restored_ids.add(window_id)
        if not restored_ids:
            return False
        self._detached_watchdog_resume_window_ids = restored_ids
        self._detached_bootstrapped = True
        if getattr(self.cfg, "log_verbose", False):
            print(f"[runtime] detached watchdog resume windows={len(restored_ids)}")
        return True

    def _detached_watchdog_resume_payload(self) -> dict:
        state_store = getattr(self, "visible_message_state", None)
        if state_store is None or not hasattr(state_store, "export_state"):
            return {}
        state = state_store.export_state()
        windows = state.get("windows") if isinstance(state, dict) else None
        if not isinstance(windows, dict) or not windows:
            return {}
        return {
            "version": 1,
            "saved_at": time.time(),
            "visible_message_state": state,
        }

    def _save_runtime_state(self, *, detached_watchdog_resume: bool = False) -> None:
        path = getattr(self, "_runtime_state_path", Path("data/runtime_state.json"))
        payload = {
            "version": 1,
            "saved_at": time.time(),
            "last_normal_reply_at": float(getattr(self, "_last_normal_reply_at", 0.0) or 0.0),
            "last_heartbeat_at": float(getattr(self, "_last_heartbeat_at", 0.0) or 0.0),
            "last_activity_at": float(getattr(self, "_last_activity_at", 0.0) or 0.0),
        }
        if detached_watchdog_resume:
            resume = self._detached_watchdog_resume_payload()
            if resume:
                payload["detached_watchdog_resume"] = resume
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(tmp, path)
        except Exception as exc:
            print(f"[warn] runtime state save failed: {exc}")

    def _mark_normal_reply_at(self, now: float) -> None:
        self._last_normal_reply_at = max(0.0, float(now or 0.0))
        self._save_runtime_state()

    def _mark_heartbeat_at(self, now: float) -> None:
        self._last_heartbeat_at = max(0.0, float(now or 0.0))
        self._save_runtime_state()

    def _mark_activity_at(self, now: float, *, persist: bool = True) -> None:
        self._last_activity_at = max(0.0, float(now or 0.0))
        if persist:
            self._save_runtime_state()

    def _ensure_start_activity_at(self, now: float) -> None:
        if self._last_activity_at <= 0:
            self._mark_activity_at(now, persist=False)

    @staticmethod
    def _strip_markdown_formatting(text: str) -> str:
        text = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", text)
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        text = re.sub(r"__(.+?)__", r"\1", text)
        text = re.sub(r"~~(.+?)~~", r"\1", text)
        text = re.sub(r"`{1,3}[^`\n]+`{1,3}", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        return text

    @staticmethod
    def _format_long_reply(text: str) -> str:
        lines = text.splitlines()
        out: list[str] = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                out.append("")
                continue
            out.append(stripped)
            if i + 1 >= len(lines):
                break
            next_line = lines[i + 1].strip()
            if not next_line:
                continue
            cur_starts_item = bool(re.match(r"^(?:\d+\.|\*\*|[•\-])", stripped))
            next_starts_item = bool(re.match(r"^(?:\d+\.|\*\*|[•\-])", next_line))
            if cur_starts_item and next_starts_item:
                out.append("")
        return "\n".join(out)

    def _normalize_preview(self, text: str) -> str:
        s = re.sub(r"\s+", "", text or "")
        # Suppress OCR jitter from punctuation/ellipsis differences.
        s = re.sub(r"[.…·•,，:：;；\-—_]+", "", s)
        return s

    def _normalize_session_title_display(self, title: str) -> str:
        raw = unicodedata.normalize("NFKC", str(title or "").strip())
        if not raw:
            return ""
        raw = re.sub(r"[‐‑‒–—―﹣－]+", "-", raw)
        raw = re.sub(r"\s+", "", raw)
        lower = raw.lower()
        for prefix in self.cfg.group_title_prefixes:
            norm_prefix = unicodedata.normalize("NFKC", str(prefix or "")).strip().lower()
            if norm_prefix and lower.startswith(norm_prefix):
                raw = re.sub(r"\(\s*\d+\s*\)$", "", raw)
                break
        return raw[:80]

    def _normalize_session_title_token(self, title: str) -> str:
        raw = self._normalize_session_title_display(title)
        if not raw:
            return ""
        raw = re.sub(r"\d{1,2}:\d{2}", "", raw)
        raw = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff\-_]+", "", raw).lower()
        for prefix in self.cfg.group_title_prefixes:
            norm_prefix = re.sub(
                r"[^0-9a-z\u4e00-\u9fff]+",
                "",
                unicodedata.normalize("NFKC", str(prefix or "")).lower(),
            )
            if not norm_prefix or (not raw.startswith(norm_prefix)):
                continue
            tail = raw[len(norm_prefix) :].lstrip("-_")
            raw = norm_prefix + tail
            break
        return raw

    def _normalize_session_titles(self, titles: object) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        if not isinstance(titles, list):
            return out
        for item in titles:
            clean = self._normalize_session_title_display(str(item))
            if not clean or clean in seen:
                continue
            seen.add(clean)
            out.append(clean)
        return out

    def _strip_preview_decorations(self, text: str) -> str:
        raw = (text or "").strip()
        if not raw:
            return ""
        # Remove marker prefixes like [有人@我] / [Photo] / [You were mentioned]
        for _ in range(3):
            m = re.match(r"^\s*[\[【][^\]】]{1,28}[\]】]\s*", raw)
            if not m:
                break
            raw = raw[m.end() :].strip()

        sep = "：" if "：" in raw else (":" if ":" in raw else "")
        if not sep:
            return raw

        left, right = raw.split(sep, 1)
        sender = left.strip(" []【】()（）")
        if 1 <= len(sender) <= 24:
            return right.strip()
        return raw

    def _display_width(self, text: str) -> int:
        width = 0
        for ch in str(text):
            if ch == "\t":
                width += 4
                continue
            if unicodedata.east_asian_width(ch) in {"W", "F"}:
                width += 2
            else:
                width += 1
        return width

    def _fit_col(self, text: str, width: int) -> str:
        clean = re.sub(r"\s+", " ", str(text or "")).strip()
        if width <= 0:
            return clean
        cur_w = self._display_width(clean)
        if cur_w <= width:
            return clean + (" " * (width - cur_w))

        suffix = "..."
        suffix_w = self._display_width(suffix)
        out = ""
        out_w = 0
        for ch in clean:
            ch_w = 2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1
            if out_w + ch_w + suffix_w > width:
                break
            out += ch
            out_w += ch_w
        out += suffix
        out_w += suffix_w
        return out + (" " * max(0, width - out_w))

    @staticmethod
    def _yn(value: bool) -> str:
        return "Y" if value else "N"

    def _term_width(self) -> int:
        forced = (os.getenv("WEAUTO_LOG_WIDTH", "") or "").strip()
        if forced.isdigit():
            return max(60, int(forced))
        cols_env = (os.getenv("COLUMNS", "") or "").strip()
        if cols_env.isdigit():
            return max(60, int(cols_env))
        try:
            return max(60, int(shutil.get_terminal_size((140, 24)).columns))
        except Exception:
            return 140

    def _title_key(self, title: str) -> str:
        t = self._normalize_session_title_token(title)
        return t[:24]

    def _resolve_person_name(self, name: str) -> str:
        if not bool(self.cfg.people_aliases_enabled):
            return str(name or "").strip()[:40]
        return self.people_alias_resolver.resolve(str(name or ""))

    def _resolve_mention_name(self, name: str) -> str:
        if not bool(self.cfg.mention_send_enabled):
            return ""
        raw = str(name or "").strip()
        canonical = self._resolve_person_name(raw)
        return self.people_alias_resolver.mention_for(raw, canonical)

    def _mention_mapping_source_text(self) -> str:
        return f"{self.cfg.people_aliases_path}:inline -[@...]"

    def _bridge_status_text(self) -> str:
        if self.cfg.processing_mode == "long_bridge":
            target = self.cfg.long_bridge_url.strip() or "-"
            return (
                f"mode=long_bridge target={target} "
                f"account={self.cfg.long_bridge_account_id} "
                f"status={self.long_bridge_client.status_text()} "
                f"timeout={self.cfg.long_bridge_timeout_sec:.1f}s "
                f"fail_open={self.cfg.long_bridge_fail_open}"
            )
        if self.cfg.processing_mode != "bridge":
            return "mode=native"
        if self.cfg.bridge_backend == "openclaw":
            target = self.cfg.bridge_openclaw_gateway_url.strip() or "-"
            return (
                f"mode=bridge backend=openclaw target={target} "
                f"agent={self.cfg.bridge_openclaw_agent_id} "
                f"timeout={self.cfg.bridge_timeout_sec:.1f}s "
                f"fail_open={self.cfg.bridge_fail_open}"
            )
        url = self.cfg.bridge_url.strip() or "-"
        return (
            f"mode=bridge backend=http url={url} "
            f"timeout={self.cfg.bridge_timeout_sec:.1f}s "
            f"fail_open={self.cfg.bridge_fail_open}"
        )

    @staticmethod
    def _bridge_clip(value: object, limit: int = 4000) -> str:
        return str(value or "")[:limit]

    @staticmethod
    def _sender_from_prefixed_text(text: str) -> str:
        raw = str(text or "").strip()
        sep = "：" if "：" in raw else (":" if ":" in raw else "")
        if not sep:
            return ""
        left, _ = raw.split(sep, 1)
        sender = left.strip(" []【】()（）")
        return sender if 1 <= len(sender) <= 24 else ""

    def _bridge_event_payload(
        self,
        row: ChatRowState,
        *,
        reason: str,
        is_group: bool,
        is_admin: bool,
        latest_message: str,
        chat_context: str = "",
        environment_context: str = "",
        session_context: str = "",
        workspace_context: str = "",
        memory_recall: str = "",
        source_message: dict | None = None,
        mention_reply_to: str = "",
    ) -> dict:
        source = source_message if isinstance(source_message, dict) else {}
        sender = str(source.get("sender") or mention_reply_to or "").strip()
        sender_raw = str(source.get("sender_raw") or "").strip()
        if not sender:
            sender = self._sender_from_prefixed_text(row.preview or row.text)
        content_type = str(source.get("content_type", "text") or "text").strip().lower()
        text = str(source.get("text") or latest_message or row.preview or row.text or "").strip()
        return {
            "schema": "weauto_bridge_event_v1",
            "app": self.cfg.app_name,
            "receiver_mode": self.cfg.receiver_mode,
            "event": {
                "reason": reason,
                "is_group": bool(is_group),
                "is_admin": bool(is_admin),
                "is_mention": reason == "mention",
                "timestamp": int(time.time()),
            },
            "chat": {
                "title": row.title,
                "row_idx": row.row_idx,
                "session_key": self._session_key_for_row(row),
            },
            "message": {
                "role": "user",
                "content_type": content_type,
                "text": text[:2000],
                "sender": sender[:40],
                "sender_raw": sender_raw[:40],
                "is_mention_me": reason == "mention",
                "image_path": str(source.get("image_path") or "")[:1000],
                "image_hash": str(source.get("image_hash") or "")[:120],
            },
            "context": {
                "latest_message": self._bridge_clip(latest_message, 2000),
                "chat_context": self._bridge_clip(chat_context),
                "environment_context": self._bridge_clip(environment_context),
                "session_context": self._bridge_clip(session_context, 1600),
                "workspace_context": self._bridge_clip(workspace_context),
                "memory_recall": self._bridge_clip(memory_recall),
            },
            "reply": {
                "mention_reply_to": mention_reply_to,
                "max_messages": max(1, int(self.cfg.agent_reply_max_messages_per_turn)),
            },
        }

    def _bridge_reply_text(
        self,
        row: ChatRowState,
        *,
        reason: str,
        is_group: bool,
        is_admin: bool,
        latest_message: str,
        chat_context: str = "",
        environment_context: str = "",
        session_context: str = "",
        workspace_context: str = "",
        memory_recall: str = "",
        source_message: dict | None = None,
        mention_reply_to: str = "",
    ) -> tuple[bool, str]:
        if self.cfg.processing_mode != "bridge":
            return False, ""
        if not self.bridge_client.enabled():
            print("[warn] bridge mode enabled but bridge target is empty")
            if self.cfg.bridge_fail_open:
                return False, ""
            return True, ""
        payload = self._bridge_event_payload(
            row,
            reason=reason,
            is_group=is_group,
            is_admin=is_admin,
            latest_message=latest_message,
            chat_context=chat_context,
            environment_context=environment_context,
            session_context=session_context,
            workspace_context=workspace_context,
            memory_recall=memory_recall,
            source_message=source_message,
            mention_reply_to=mention_reply_to,
        )
        try:
            if self.cfg.log_verbose:
                print(
                    f"[bridge] request row={row.row_idx:>2} "
                    f"reason={reason:<14} title={self._fit_col(row.title, 14)}"
                )
            result = self.bridge_client.request_reply(payload)
        except Exception as exc:
            print(f"[warn] bridge failed: {exc}")
            if self.cfg.bridge_fail_open:
                print("[bridge] fail-open: fallback to native pipeline")
                return False, ""
            print("[bridge] fail-closed: skip native reply")
            return True, ""
        if not result.send:
            if self.cfg.log_verbose:
                print(f"[bridge] send=false row={row.row_idx:>2}")
            return True, ""
        reply = (result.reply or "").strip()
        if not reply:
            if self.cfg.log_verbose:
                print(f"[bridge] empty reply row={row.row_idx:>2}")
            return True, ""
        if self.cfg.log_verbose:
            print(f"[bridge] reply row={row.row_idx:>2} len={len(reply)}")
        return True, reply

    def _long_bridge_event_payload(
        self,
        row: ChatRowState,
        *,
        reason: str,
        is_group: bool,
        is_admin: bool,
        latest_message: str,
        source_message: dict | None = None,
        mention_reply_to: str = "",
    ) -> dict:
        source = source_message if isinstance(source_message, dict) else {}
        sender = str(source.get("sender") or mention_reply_to or "").strip()
        if not sender:
            sender = self._sender_from_prefixed_text(row.preview or row.text)
        sender = sender or row.title
        content_type = str(source.get("content_type") or "text").strip().lower()
        text = str(source.get("text") or latest_message or row.preview or row.text or "").strip()
        attachments: list[dict] = []
        image_path = str(source.get("image_path") or "").strip()
        if image_path:
            attachment = self.long_bridge_client.build_attachment(image_path)
            if attachment:
                attachments.append(attachment)

        session_key = self._session_key_for_row(row)
        fingerprint = str(source.get("fingerprint") or "").strip()
        if fingerprint:
            event_seed = f"{session_key}|{fingerprint}|{reason}|{sender}"
        else:
            event_seed = (
                f"{session_key}|{reason}|{sender}|{time.time_ns()}|{text[:400]}"
            )
        event_id = hashlib.sha256(event_seed.encode("utf-8")).hexdigest()[:32]
        return {
            "event_id": event_id,
            "message": {
                "conversation": {
                    "id": session_key,
                    "kind": "group" if is_group else "direct",
                    "title": row.title,
                },
                "sender": {
                    "id": sender,
                    "name": sender,
                },
                "text": text[:8000],
                "content_type": content_type,
                "mentioned_bot": reason == "mention",
                "timestamp_ms": int(time.time() * 1000),
                "attachments": attachments,
                "metadata": {
                    "reason": reason,
                    "is_admin": bool(is_admin),
                    "receiver_mode": self.cfg.receiver_mode,
                },
            },
        }

    def _long_bridge_reply(
        self,
        row: ChatRowState,
        *,
        reason: str,
        is_group: bool,
        is_admin: bool,
        latest_message: str,
        source_message: dict | None = None,
        mention_reply_to: str = "",
    ) -> tuple[bool, LongBridgeResult | None]:
        if self.cfg.processing_mode != "long_bridge":
            return False, None
        if not self.long_bridge_client.enabled():
            print("[warn] long bridge mode enabled but URL or token is empty")
            if self.cfg.long_bridge_fail_open:
                return False, None
            return True, None
        try:
            payload = self._long_bridge_event_payload(
                row,
                reason=reason,
                is_group=is_group,
                is_admin=is_admin,
                latest_message=latest_message,
                source_message=source_message,
                mention_reply_to=mention_reply_to,
            )
            if self.cfg.log_verbose:
                print(
                    f"[long-bridge] request row={row.row_idx:>2} "
                    f"reason={reason:<14} title={self._fit_col(row.title, 14)}"
                )
            result = self.long_bridge_client.request_reply(payload)
        except Exception as exc:
            print(f"[warn] long bridge failed: {exc}")
            if self.cfg.long_bridge_fail_open:
                print("[long-bridge] fail-open: fallback to native pipeline")
                return False, None
            print("[long-bridge] fail-closed: skip native reply")
            return True, None
        if self.cfg.log_verbose:
            print(
                f"[long-bridge] reply row={row.row_idx:>2} "
                f"text={len(result.reply)} files={len(result.attachments)} "
                f"send={result.send}"
            )
        return True, result

    def _send_long_bridge_attachments(
        self,
        row: ChatRowState,
        attachments: list[Path],
        *,
        focused_bounds=None,
    ) -> None:
        for file_path in attachments:
            with self._send_lock:
                self._pause_visual_scan(self.cfg.mention_pause_scan_sec)
                self._send_generated_file(
                    row,
                    file_path,
                    focused_bounds=focused_bounds,
                )

    def _sync_long_bridge_conversations(self, windows: list[DetachedWindowInfo]) -> None:
        conversations: list[dict[str, object]] = []
        titles_by_id: dict[str, str] = {}
        for window in windows:
            title = self._normalize_session_title_display(window.title)
            conversation_id = self._canonical_session_key(
                title,
                0,
                remember=False,
            )
            if not conversation_id or not title:
                continue
            is_group = any(
                str(prefix or "") and title.startswith(str(prefix))
                for prefix in self.cfg.group_title_prefixes
            )
            conversations.append(
                {
                    "id": conversation_id,
                    "title": title,
                    "kind": "group" if is_group else "direct",
                }
            )
            titles_by_id[conversation_id] = title
        with self._state_lock:
            self._long_bridge_titles_by_id = titles_by_id
        self.long_bridge_client.sync_conversations(conversations)

    def _handle_long_bridge_outbound(
        self,
        outbound: LongBridgeOutbound,
    ) -> LongBridgeDelivery:
        with self._state_lock:
            known_title = self._long_bridge_titles_by_id.get(
                outbound.conversation_id.strip(),
                "",
            )
        title = known_title or outbound.conversation_title.strip()
        if not title:
            print(
                "[warn] long bridge proactive message has no conversation title; "
                f"id={outbound.conversation_id!r}"
            )
            return LongBridgeDelivery(False, "conversation_title_missing")
        if self.cfg.dry_run:
            if outbound.result.reply:
                print(f"[dry-run] proactive to={title!r} msg={outbound.result.reply!r}")
            for file_path in outbound.result.attachments:
                print(f"[dry-run] proactive to={title!r} file={file_path}")
            return LongBridgeDelivery(False, "dry_run")
        with self._send_lock:
            self._pause_visual_scan(self.cfg.mention_pause_scan_sec)
            if outbound.result.reply:
                message = self._strip_markdown_formatting(outbound.result.reply)
                sent = self.sender.paste_and_send_to_window(title, message)
                if not sent:
                    print(
                        f"[warn] proactive long bridge send window not confirmed: "
                        f"{title!r}"
                    )
                    return LongBridgeDelivery(False, "target_window_not_confirmed")
            for file_path in outbound.result.attachments:
                sent = self.sender.paste_file_and_send_to_window(title, file_path)
                if not sent:
                    print(
                        f"[warn] proactive long bridge file send not confirmed: "
                        f"{title!r} {file_path}"
                    )
                    return LongBridgeDelivery(False, "file_send_not_confirmed")
        print(
            f"[delivered] proactive to={title!r} message_id={outbound.message_id!r}"
        )
        return LongBridgeDelivery(True)

    def close(self) -> None:
        self.long_bridge_client.close()
        self._msg_executor.shutdown(wait=False, cancel_futures=True)
        if self.ocr_proxy is not None:
            self.ocr_proxy.close()

    def _canonicalize_sender_pair(self, sender: str) -> tuple[str, str]:
        raw = str(sender or "").strip()
        if not raw:
            return "", ""
        canonical = self._resolve_person_name(raw)
        if not canonical:
            return "", raw[:40]
        raw_out = raw[:40] if raw != canonical else ""
        return canonical[:40], raw_out

    def _memory_session_relpath(self, key: str) -> str:
        slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "-", key or "").strip("-").lower()
        return f"{self._memory_path.stem}.sessions/{(slug[:80] or 'session')}.json"

    def _memory_session_path(self, key: str, relpath: str = "") -> Path:
        rel = (relpath or self._memory_session_relpath(key)).strip()
        return self._memory_path.parent / rel

    def _session_state_from_index(self, key: str, meta: dict) -> SessionState:
        short = meta.get("short", [])
        titles = meta.get("titles", [])
        return SessionState(
            short=[str(x) for x in short][-max(4, self.cfg.memory_short_max_items) :]
            if isinstance(short, list)
            else [],
            history=[],
            summary=str(meta.get("summary", ""))[: max(120, self.cfg.memory_summary_max_chars)],
            muted=bool(meta.get("muted", False)),
            titles=set(self._normalize_session_titles(titles)),
            loaded=False,
        )

    def _normalize_history_items(self, history: object) -> list[dict]:
        history_items: list[dict] = []
        if isinstance(history, list):
            for item in history:
                if not isinstance(item, dict):
                    continue
                text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
                if not text:
                    continue
                sender, sender_raw = self._canonicalize_sender_pair(str(item.get("sender", "")))
                if not sender_raw:
                    sender_raw = str(item.get("sender_raw", "")).strip()[:40]
                history_items.append(
                    {
                        "role": str(item.get("role", "unknown")).strip().lower(),
                        "content_type": str(item.get("content_type", "unknown")).strip().lower(),
                        "text": text[:220],
                        "sender": sender,
                        "sender_raw": sender_raw,
                        "source": str(item.get("source", "memory")).strip()[:20],
                        "observed_at": int(item.get("observed_at", 0) or 0),
                    }
                )
        return history_items

    def _load_session_payload(self, key: str, sess: SessionState) -> None:
        meta = self._session_index.get(key, {})
        dir_name = str(meta.get("dir", key))
        history_limit = max(0, int(self.cfg.memory_history_max_items))
        history_records = (
            self.chat_history.read_recent(dir_name, limit=history_limit)
            if history_limit > 0
            else self.chat_history.load_all(dir_name)
        )
        history_items = self._normalize_history_items(history_records)
        if not sess.short and isinstance(meta.get("short", []), list):
            sess.short = [str(x) for x in meta["short"]][-max(4, self.cfg.memory_short_max_items) :]
        if not sess.summary:
            sess.summary = str(meta.get("summary", ""))[: max(120, self.cfg.memory_summary_max_chars)]
        if not sess.titles:
            sess.titles = set(self._normalize_session_titles(meta.get("titles", [])))
        if not sess.muted:
            sess.muted = bool(meta.get("muted", False))
        sess.history = history_items
        sess.loaded = True
        for item in history_items:
            sender = str(item.get("sender", "")).strip()
            text = str(item.get("text", "")).strip()
            if text:
                sess.seen_content.add((sender, re.sub(r"\s+", "", text)[:220]))

    def _get_or_create_session(self, key: str, *, load_history: bool = True) -> SessionState:
        with self._state_lock:
            sess = self._sessions.get(key)
            if sess is not None:
                if load_history and (not sess.loaded):
                    self._load_session_payload(key, sess)
                return sess
            meta = self._session_index.get(key)
            if isinstance(meta, dict):
                sess = self._session_state_from_index(key, meta)
                self._sessions[key] = sess
                if load_history:
                    self._load_session_payload(key, sess)
                return sess
            sess = SessionState(short=[], history=[], summary="", muted=False, titles=set(), loaded=True)
            self._sessions[key] = sess
            return sess

    def _remember_session_alias(self, alias: str, canonical: str) -> None:
        if not alias:
            return
        old = self._session_aliases.get(alias)
        if old == canonical:
            return
        self._session_aliases[alias] = canonical
        self._memory_dirty = True

    def _remember_session_title(self, key: str, title: str) -> None:
        clean = self._normalize_session_title_display(title)
        if not clean:
            return
        sess = self._get_or_create_session(key, load_history=False)
        if clean not in sess.titles:
            sess.titles.add(clean)
            self._memory_dirty = True

    def _sort_session_history(self, history: list[dict]) -> list[dict]:
        indexed = list(enumerate(history))
        indexed.sort(key=lambda it: (int(it[1].get("observed_at", 0) or 0), it[0]))
        return [item for _, item in indexed]

    def _merge_session_keys(
        self,
        *,
        src_key: str,
        dst_key: str,
        dst_title: str = "",
        sync_workspace: bool = True,
    ) -> bool:
        if (not src_key) or (not dst_key) or src_key == dst_key:
            return False
        src = self._get_or_create_session(src_key)
        dst = self._get_or_create_session(dst_key)
        if dst_title:
            self._remember_session_title(dst_key, dst_title)
        dst.short = (dst.short + src.short)[-max(4, self.cfg.memory_short_max_items) :]
        dst.history = self._sort_session_history(dst.history + src.history)
        hist_limit = max(0, int(self.cfg.memory_history_max_items))
        if hist_limit > 0 and len(dst.history) > hist_limit:
            dst.history = dst.history[-hist_limit:]
        if src.summary and src.summary not in dst.summary:
            glue = " | " if dst.summary else ""
            dst.summary = (dst.summary + glue + src.summary)[
                : max(120, self.cfg.memory_summary_max_chars)
            ]
        dst.muted = bool(dst.muted or src.muted)
        dst.titles.update(src.titles)

        for alias, key in list(self._session_aliases.items()):
            if key == src_key:
                self._session_aliases[alias] = dst_key
        src_alias = self._title_key(src_key)
        if src_alias:
            self._session_aliases[src_alias] = dst_key

        self._sessions.pop(src_key, None)
        self._session_index.pop(src_key, None)
        self._summary_turn_counter.pop(src_key, None)
        self._memory_dirty = True
        return True

    def _coalesce_loaded_sessions(self) -> None:
        all_keys = sorted(set(self._session_index.keys()) | set(self._sessions.keys()))
        grouped: dict[str, dict[str, object]] = {}
        original_keys: dict[str, set[str]] = {}
        changed = False

        for key in all_keys:
            names: list[str] = []
            clean_key = self._normalize_session_title_display(key)
            if clean_key:
                names.append(clean_key)
            meta = self._session_index.get(key, {})
            if isinstance(meta, dict):
                names.extend(self._normalize_session_titles(meta.get("titles", [])))
            sess = self._sessions.get(key)
            if sess and sess.titles:
                names.extend([self._normalize_session_title_display(x) for x in sess.titles])
            names = [x for x in names if x]
            canonical = self._title_key(key)
            if not canonical:
                for name in names:
                    canonical = self._title_key(name)
                    if canonical:
                        break
            if not canonical:
                continue
            bucket = grouped.setdefault(canonical, {"keys": [], "titles": set()})
            bucket["keys"].append(key)
            bucket["titles"].update(names)
            original_keys.setdefault(canonical, set()).add(key)

        for canonical, bucket in grouped.items():
            keys = sorted(set(bucket["keys"]))  # type: ignore[index]
            titles = sorted(set(bucket["titles"]))  # type: ignore[index]
            if canonical not in self._sessions and canonical not in self._session_index and titles:
                self._remember_session_title(canonical, titles[0])
                changed = True
            for src_key in keys:
                if src_key == canonical:
                    continue
                if self._merge_session_keys(
                    src_key=src_key,
                    dst_key=canonical,
                    dst_title=(titles[0] if titles else canonical),
                ):
                    changed = True
            for alias_source in set(titles) | original_keys.get(canonical, set()):
                alias = self._title_key(str(alias_source))
                if alias and self._session_aliases.get(alias) != canonical:
                    self._session_aliases[alias] = canonical
                    changed = True

        if changed:
            print("[memory] coalesced legacy session keys with normalized group titles")

    def _canonical_session_key(
        self,
        title: str,
        row_idx: int,
        *,
        remember: bool = True,
    ) -> str:
        key = self._title_key(title)
        if not key:
            return f"row-{row_idx}"
        canonical = self._session_aliases.get(key, key)
        if remember:
            self._remember_session_alias(key, canonical)
            self._remember_session_title(canonical, title)
        return canonical

    def _session_key_for_row(self, row: ChatRowState, *, remember: bool = True) -> str:
        return self._canonical_session_key(
            row.title,
            row.row_idx,
            remember=remember,
        )

    def _session_record_key(self, record: dict) -> str:
        role = str(record.get("role", "unknown")).strip().lower()
        content_type = str(record.get("content_type", "unknown")).strip().lower()
        sender = self._normalize_preview(str(record.get("sender", "")))[:40]
        text = re.sub(r"\s+", "", self._normalize_preview(str(record.get("text", ""))))[:140]
        return f"{role}|{content_type}|{sender}|{text}"

    def _append_session_record(
        self,
        row: ChatRowState,
        *,
        role: str,
        text: str,
        content_type: str = "text",
        sender: str = "",
        source: str = "runtime",
        count_turn: bool = True,
    ) -> None:
        clean = re.sub(r"\s+", " ", text or "").strip()
        if not clean:
            return

        role_map = {
            "U": "user",
            "A": "assistant",
            "user": "user",
            "assistant": "assistant",
        }
        norm_role = role_map.get(role, "unknown")
        key = self._session_key_for_row(row)
        sess = self._get_or_create_session(key)
        sender_clean = (sender or "").strip()
        sender_clean = (sender or "").strip()
        sender_raw = ""
        if norm_role == "user":
            sender_clean, sender_raw = self._canonicalize_sender_pair(sender_clean)
        else:
            sender_clean = sender_clean[:40]
        record = {
            "role": norm_role,
            "content_type": (content_type or "text").strip().lower() or "text",
            "text": clean[:220],
            "sender": sender_clean,
            "source": (source or "runtime").strip()[:20],
            "observed_at": int(time.time()),
        }
        if sender_raw:
            record["sender_raw"] = sender_raw
        dedup_text = re.sub(r"\s+", "", clean)[:220]
        content_key = (sender_clean, dedup_text)
        if content_key in sess.seen_content:
            return
        record_key = self._session_record_key(record)
        tail = sess.history[-500:]
        if any(self._session_record_key(x) == record_key for x in tail):
            return

        sess.seen_content.add(content_key)
        if len(sess.seen_content) > 50000:
            sess.seen_content = set()
        sess.history.append(record)
        if str(record.get("source", "")) != "tool":
            self.chat_history.append(row.title, record)
        hist_limit = max(0, int(self.cfg.memory_history_max_items))
        if hist_limit > 0 and len(sess.history) > hist_limit:
            del sess.history[:-hist_limit]

        short_role = "A" if norm_role == "assistant" else ("U" if norm_role == "user" else "?")
        short_sender = sender_clean[:24]
        if norm_role == "user" and short_sender:
            item = f"{short_role}({short_sender}):{clean[:128]}"
        else:
            item = f"{short_role}:{clean[:140]}"
        if (not sess.short) or sess.short[-1] != item:
            sess.short.append(item)
            max_items = max(4, self.cfg.memory_short_max_items)
            if len(sess.short) > max_items:
                del sess.short[:-max_items]

        self._remember_session_title(key, row.title)
        if norm_role == "user" and count_turn:
            self._summary_turn_counter[key] += 1
        self._memory_dirty = True

    def _session_short_item_from_record(self, record: dict) -> str:
        role = str(record.get("role", "unknown")).strip().lower()
        text = re.sub(r"\s+", " ", str(record.get("text", ""))).strip()
        sender = (str(record.get("sender", "")) or "").strip()[:24]
        short_role = "A" if role == "assistant" else ("U" if role == "user" else "?")
        if role == "user" and sender:
            return f"{short_role}({sender}):{text[:128]}"
        return f"{short_role}:{text[:140]}"

    def _refresh_session_short_from_history(self, sess: SessionState) -> None:
        items: list[str] = []
        for record in sess.history:
            text = re.sub(r"\s+", " ", str(record.get("text", ""))).strip()
            if not text:
                continue
            items.append(self._session_short_item_from_record(record))
        max_items = max(4, self.cfg.memory_short_max_items)
        sess.short = items[-max_items:]

    def _normalize_records_for_merge(
        self,
        records: list[dict] | None,
        *,
        source: str,
    ) -> list[dict]:
        incoming: list[dict] = []
        for record in records or []:
            if not isinstance(record, dict):
                continue
            # Skip our own replies when merging from a screenshot. The bot
            # already records every successfully-sent reply via
            # _append_session_item at send time, so re-reading them from the
            # screen only risks persisting Vision-LLM hallucinations (messages
            # that were never actually sent). See _detached_message_record for
            # the role assignment.
            role = str(record.get("role", "unknown")).strip().lower()
            if role == "assistant":
                continue
            text = str(record.get("text", "")).strip()
            if not text:
                continue
            sender_clean, sender_raw = self._canonicalize_sender_pair(str(record.get("sender", "")))
            if not sender_raw:
                sender_raw = str(record.get("sender_raw", "")).strip()[:40]
            incoming.append(
                {
                    "role": role,
                    "content_type": str(record.get("content_type", "unknown")).strip().lower(),
                    "text": text[:220],
                    "sender": sender_clean,
                    "sender_raw": sender_raw,
                    "source": source,
                }
            )
        return incoming

    @staticmethod
    def _max_suffix_prefix_overlap(left: list[str], right: list[str]) -> int:
        max_overlap = min(len(left), len(right))
        for size in range(max_overlap, 0, -1):
            if left[-size:] == right[:size]:
                return size
        return 0

    @staticmethod
    def _set_record_observed_range(records: list[dict], *, before: int | None = None, after: int | None = None) -> None:
        if not records:
            return
        if before is not None:
            start = int(before) - len(records)
        else:
            anchor = int(after if after is not None else time.time())
            start = anchor + 1
        for idx, record in enumerate(records):
            record["observed_at"] = start + idx

    def _rewrite_recover_workspace_session(self, row: ChatRowState) -> None:
        pass

    def _merge_session_records(
        self,
        row: ChatRowState,
        records: list[dict] | None,
        *,
        source: str = "vision",
        order_mode: str = "append",
    ) -> None:
        if not records:
            return
        key = self._session_key_for_row(row)
        sess = self._get_or_create_session(key)
        incoming = self._normalize_records_for_merge(records, source=source)
        if not incoming:
            return

        if order_mode == "recover":
            existing_head_keys = [self._session_record_key(x) for x in sess.history[:80]]
            existing_tail_keys = [self._session_record_key(x) for x in sess.history[-80:]]
            incoming_keys = [self._session_record_key(x) for x in incoming]

            prepend_overlap = self._max_suffix_prefix_overlap(incoming_keys, existing_head_keys)
            append_overlap = self._max_suffix_prefix_overlap(existing_tail_keys, incoming_keys)

            if not sess.history:
                self._set_record_observed_range(incoming, after=int(time.time()) - 1)
                sess.history = list(incoming)
                for rec in incoming:
                    self.chat_history.append(row.title, rec)
            elif prepend_overlap > 0 and prepend_overlap >= append_overlap:
                new_records = incoming[:-prepend_overlap]
                if not new_records:
                    return
                first_ts = int(sess.history[0].get("observed_at", 0) or 0)
                if first_ts <= 0:
                    first_ts = int(sess.history[-1].get("observed_at", 0) or int(time.time()))
                self._set_record_observed_range(new_records, before=first_ts)
                sess.history = list(new_records) + list(sess.history)
                for rec in new_records:
                    self.chat_history.append(row.title, rec)
            elif append_overlap > 0:
                new_records = incoming[append_overlap:]
                if not new_records:
                    return
                last_ts = int(sess.history[-1].get("observed_at", 0) or 0)
                self._set_record_observed_range(new_records, after=last_ts)
                sess.history.extend(new_records)
                for rec in new_records:
                    self.chat_history.append(row.title, rec)
            else:
                first_ts = int(sess.history[0].get("observed_at", 0) or 0)
                if first_ts <= 0:
                    first_ts = int(time.time())
                self._set_record_observed_range(incoming, before=first_ts)
                sess.history = list(incoming) + list(sess.history)
                for rec in incoming:
                    self.chat_history.append(row.title, rec)
                if self.cfg.log_verbose:
                    print(
                        f"[recover-merge] no overlap | key={self._fit_col(key, 12)} | "
                        f"prepend={len(incoming):>2}"
                    )

            hist_limit = max(0, int(self.cfg.memory_history_max_items))
            if hist_limit > 0 and len(sess.history) > hist_limit:
                sess.history = sess.history[-hist_limit:]
            self._refresh_session_short_from_history(sess)
            self._remember_session_title(key, row.title)
            self._memory_dirty = True
            return

        existing_keys = [self._session_record_key(x) for x in sess.history[-500:]]
        incoming_keys = [self._session_record_key(x) for x in incoming]
        overlap = 0
        max_overlap = min(len(existing_keys), len(incoming_keys))
        for size in range(max_overlap, 0, -1):
            if existing_keys[-size:] == incoming_keys[:size]:
                overlap = size
                break

        for record in incoming[overlap:]:
            self._append_session_record(
                row,
                role=str(record.get("role", "unknown")),
                text=str(record.get("text", "")),
                content_type=str(record.get("content_type", "unknown")),
                sender=str(record.get("sender", "")),
                source=source,
                count_turn=str(record.get("role", "")) == "user",
            )

    def _append_session_item(
        self,
        row: ChatRowState,
        role: str,
        text: str,
        count_turn: bool = True,
    ) -> None:
        self._append_session_record(
            row,
            role=role,
            text=text,
            content_type="text",
            sender="",
            source="runtime",
            count_turn=count_turn,
        )

    def _update_long_summary(self, row: ChatRowState) -> None:
        if not self.cfg.memory_enabled:
            return
        key = self._session_key_for_row(row)
        turns = self._summary_turn_counter.get(key, 0)
        if turns < max(1, self.cfg.memory_summary_update_every):
            return
        self._summary_turn_counter[key] = 0

        sess = self._get_or_create_session(key)
        short_tail = sess.short[-max(4, self.cfg.memory_summary_recent_items) :]
        if not short_tail:
            return

        prev = sess.summary
        try:
            new_summary = self.llm_summary.summarize_session(
                title=row.title, previous_summary=prev, short_items=short_tail
            )
        except Exception as exc:
            print(f"[warn] summary failed, keep old summary: {exc}")
            new_summary = prev

        if not new_summary:
            new_summary = prev
        if not new_summary and short_tail:
            new_summary = " | ".join(short_tail[-4:])
        new_summary = new_summary[: max(120, self.cfg.memory_summary_max_chars)]
        if new_summary != sess.summary:
            sess.summary = new_summary
            self._memory_dirty = True
            if self.cfg.log_verbose:
                print(
                    f"[memory-summary] key={key!r} size={len(sess.summary)} "
                    f"turns={self.cfg.memory_summary_update_every}"
                )

    def _build_session_context(self, row: ChatRowState) -> str:
        key = self._session_key_for_row(row)
        sess = self._get_or_create_session(key)
        short_n = max(2, self.cfg.memory_short_context_items)
        short_text = " | ".join(sess.short[-short_n:])
        long_text = (sess.summary or "").strip()
        if long_text and short_text:
            return f"[长期摘要]{long_text}\n[短期上下文]{short_text}"[:1200]
        if long_text:
            return f"[长期摘要]{long_text}"[:1200]
        return short_text[:1200]

    def _build_session_history_text(self, row: ChatRowState, *, max_items: int | None = None) -> str:
        key = self._session_key_for_row(row)
        sess = self._get_or_create_session(key)
        limit = max(1, int(max_items)) if max_items is not None else max(4, int(self.cfg.memory_history_context_items))
        lines: list[str] = []
        for item in sess.history[-limit:]:
            role = str(item.get("role", "unknown"))
            text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
            sender = re.sub(r"\s+", " ", str(item.get("sender", ""))).strip()[:24]
            if not text:
                continue
            ts = int(item.get("observed_at", 0) or 0)
            ts_text = datetime.fromtimestamp(ts).strftime("%m-%d %H:%M") if ts > 0 else "--"
            prefix = "A" if role == "assistant" else ("U" if role == "user" else "?")
            if role == "user" and sender:
                lines.append(f"{ts_text} {prefix}({sender}):{text[:128]}")
            else:
                lines.append(f"{ts_text} {prefix}:{text[:140]}")
        return " | ".join(lines)[:1600]

    @staticmethod
    def _normalize_history_date(value: str) -> str:
        clean = re.sub(r"\s+", "", str(value or "")).strip()
        now = datetime.now()
        if not clean or clean.lower() in {"today", "tod", "now"} or clean in {"今天", "今日", "当天"}:
            return now.strftime("%Y-%m-%d")
        if clean.lower() in {"yesterday", "yst"} or clean in {"昨天", "昨日"}:
            return (now - timedelta(days=1)).strftime("%Y-%m-%d")
        if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", clean):
            try:
                return datetime.strptime(clean, "%Y-%m-%d").strftime("%Y-%m-%d")
            except ValueError:
                return now.strftime("%Y-%m-%d")
        if re.fullmatch(r"\d{1,2}-\d{1,2}", clean):
            try:
                return datetime.strptime(f"{now.year}-{clean}", "%Y-%m-%d").strftime("%Y-%m-%d")
            except ValueError:
                return now.strftime("%Y-%m-%d")
        return now.strftime("%Y-%m-%d")

    @staticmethod
    def _format_history_records(
        records: list[dict],
        *,
        include_date: bool = False,
        max_items: int = 400,
        max_chars: int = 12000,
    ) -> str:
        lines: list[str] = []
        for item in records[-max(1, int(max_items)) :]:
            role = str(item.get("role", "unknown"))
            text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
            if not text:
                continue
            sender = re.sub(r"\s+", " ", str(item.get("sender", ""))).strip()[:24]
            ts = int(item.get("observed_at", 0) or 0)
            fmt = "%m-%d %H:%M" if include_date else "%H:%M"
            ts_text = datetime.fromtimestamp(ts).strftime(fmt) if ts > 0 else "--"
            prefix = "A" if role == "assistant" else ("U" if role == "user" else "?")
            who = f"{prefix}({sender})" if role == "user" and sender else prefix
            lines.append(f"{ts_text} {who}: {text[:260]}")
        return "\n".join(lines)[: max(800, int(max_chars))]

    def _read_chat_history_by_date_text(
        self,
        row: ChatRowState,
        *,
        date: str = "",
        max_items: int = 400,
        max_chars: int = 12000,
    ) -> str:
        date_str = self._normalize_history_date(date)
        records = self.chat_history.read_date(row.title, date_str)
        if not records:
            return f"聊天记录[{row.title}][{date_str}]: 无"
        body = self._format_history_records(
            records,
            max_items=max_items,
            max_chars=max_chars,
        )
        return (
            f"聊天记录[{row.title}][{date_str}] 总记录={len(records)} "
            f"显示={min(len(records), max(1, int(max_items)))}:\n{body}"
        )

    def _summarize_chat_history_by_date_text(
        self,
        row: ChatRowState,
        *,
        date: str = "",
        max_chars: int = 7200,
    ) -> str:
        date_str = self._normalize_history_date(date)
        records = self.chat_history.read_date(row.title, date_str)
        if not records:
            return f"群聊摘要素材[{row.title}][{date_str}]: 无当天聊天记录"

        sender_counts: dict[str, int] = defaultdict(int)
        hourly: dict[str, list[str]] = defaultdict(list)
        seen_lines: set[str] = set()
        for item in records:
            role = str(item.get("role", "unknown"))
            text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
            if not text:
                continue
            sender = re.sub(r"\s+", " ", str(item.get("sender", ""))).strip()
            who = sender if role == "user" and sender else ("萨比" if role == "assistant" else role)
            sender_counts[who] += 1
            ts = int(item.get("observed_at", 0) or 0)
            hour = datetime.fromtimestamp(ts).strftime("%H:00") if ts > 0 else "--"
            compact = re.sub(r"\s+", "", text)
            key = f"{who}|{compact[:80]}"
            if key in seen_lines:
                continue
            seen_lines.add(key)
            hourly[hour].append(f"{datetime.fromtimestamp(ts).strftime('%H:%M') if ts > 0 else '--'} {who}: {text[:180]}")

        top_senders = "、".join(
            f"{name}({count})"
            for name, count in sorted(sender_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
        )
        parts = [
            f"群聊摘要素材[{row.title}][{date_str}]",
            f"总记录={len(records)}；活跃发言人={top_senders or '无'}",
            "按时间段摘录（请据此概括主题；不要说没有记录，除非这里为空）：",
        ]
        for hour in sorted(hourly):
            entries = hourly[hour]
            if len(entries) > 12:
                indexes = sorted({round(i * (len(entries) - 1) / 11) for i in range(12)})
                entries = [entries[i] for i in indexes]
            samples = " | ".join(entries)
            parts.append(f"[{hour}] {samples}")
        return "\n".join(parts)[: max(1200, int(max_chars))]

    def _search_session_history(
        self,
        row: ChatRowState,
        *,
        query: str,
        limit: int = 10,
        context: int = 2,
    ) -> str:
        return self.chat_history.search(
            row.title,
            query,
            limit=limit,
            context=context,
        )

    def _workspace_context_for_row(
        self,
        row: ChatRowState,
        *,
        is_admin: bool,
        skill_query: str = "",
    ) -> str:
        try:
            from .prompt_context import build_prompt_context

            query = re.sub(r"\s+", " ", str(skill_query or row.preview or row.title or "").strip())[:240]
            return build_prompt_context(
                include_long_term=is_admin,
                skill_query=query,
            )
        except Exception as exc:
            if self.cfg.log_verbose:
                print(f"[warn] config context load failed: {exc}")
            return ""

    def _workspace_memory_recall_for_row(
        self,
        row: ChatRowState,
        query: str,
        *,
        is_admin: bool,
    ) -> str:
        return ""

    def _apply_session_summary(self, row: ChatRowState, summary: str) -> None:
        clean = re.sub(r"\s+", " ", summary or "").strip()
        if not clean:
            return
        key = self._session_key_for_row(row)
        sess = self._get_or_create_session(key)
        clipped = clean[: max(120, self.cfg.memory_summary_max_chars)]
        if clipped != sess.summary:
            sess.summary = clipped
            self._memory_dirty = True

    def _build_environment_context(self, snapshot: ChatContextSnapshot) -> str:
        parts: list[str] = []
        summary = re.sub(r"\s+", " ", snapshot.memory_summary or "").strip()
        if summary:
            parts.append(f"[环境总结]\n- {summary[:240]}")

        time_hints = [re.sub(r"\s+", " ", str(x or "")).strip() for x in (snapshot.memory_time_hints or [])]
        time_hints = [x[:40] for x in time_hints if x]
        if time_hints:
            parts.append("[时间线索]\n" + "\n".join(f"- {x}" for x in time_hints[:8]))

        people_lines: list[str] = []
        for person in snapshot.memory_people or []:
            if not isinstance(person, dict):
                continue
            name = re.sub(r"\s+", " ", str(person.get("name", ""))).strip()[:24]
            alias = re.sub(r"\s+", " ", str(person.get("alias", ""))).strip()[:24]
            desc = re.sub(r"\s+", " ", str(person.get("description", ""))).strip()[:80]
            if not name and not alias and not desc:
                continue
            line = name or alias or "未知人物"
            tail = []
            if alias and alias != line:
                tail.append(f"别名={alias}")
            if desc:
                tail.append(f"说明={desc}")
            if tail:
                line += "；" + "；".join(tail)
            people_lines.append(f"- {line}")
        if people_lines:
            parts.append("[人物]\n" + "\n".join(people_lines[:8]))

        facts = [re.sub(r"\s+", " ", str(x or "")).strip()[:80] for x in (snapshot.memory_facts or [])]
        facts = [x for x in facts if x]
        if facts:
            parts.append("[事实]\n" + "\n".join(f"- {x}" for x in facts[:8]))

        events = [re.sub(r"\s+", " ", str(x or "")).strip()[:80] for x in (snapshot.memory_events or [])]
        events = [x for x in events if x]
        if events:
            parts.append("[事件]\n" + "\n".join(f"- {x}" for x in events[:8]))

        relation_lines: list[str] = []
        for relation in snapshot.memory_relations or []:
            if not isinstance(relation, dict):
                continue
            subject = re.sub(r"\s+", " ", str(relation.get("subject", ""))).strip()[:24]
            rel = re.sub(r"\s+", " ", str(relation.get("relation", ""))).strip()[:24]
            target = re.sub(r"\s+", " ", str(relation.get("target", ""))).strip()[:24]
            note = re.sub(r"\s+", " ", str(relation.get("note", ""))).strip()[:80]
            if not (subject and rel and target):
                continue
            line = f"- {subject} -> {rel} -> {target}"
            if note:
                line += f"；说明={note}"
            relation_lines.append(line)
        if relation_lines:
            parts.append("[关系]\n" + "\n".join(relation_lines[:8]))

        return "\n\n".join(parts)[:2200]

    def _load_persistent_memory(self) -> None:
        if not self.cfg.memory_enabled:
            return
        index = self.chat_history.load_index()
        sessions = index.get("sessions", {})
        aliases = index.get("aliases", {})

        if not sessions and not aliases:
            if self._memory_path.exists():
                print(
                    "[memory] Old format detected. Run scripts/migrate_to_date_files.py to migrate."
                )
                self._load_persistent_memory_legacy()
            return

        for key, data in sessions.items():
            if not isinstance(data, dict):
                continue
            key_str = str(key).strip()
            if not key_str:
                continue
            meta = {
                "dir": str(data.get("dir", "")),
                "short": [str(x) for x in (data.get("short", []) or [])][-max(4, self.cfg.memory_short_max_items) :]
                if isinstance(data.get("short", []), list)
                else [],
                "summary": str(data.get("summary", ""))[: max(120, self.cfg.memory_summary_max_chars)],
                "muted": bool(data.get("muted", False)),
                "titles": self._normalize_session_titles(data.get("titles", [])),
                "message_count": int(data.get("message_count", 0) or 0),
                "updated_at": int(data.get("updated_at", 0) or 0),
            }
            self._session_index[key_str] = meta
            self._sessions[key_str] = self._session_state_from_index(key_str, meta)

        if isinstance(aliases, dict):
            for k, v in aliases.items():
                k2 = str(k).strip()
                v2 = str(v).strip()
                if k2 and v2:
                    self._session_aliases[self._title_key(k2) or k2] = self._title_key(v2) or v2

        self._memory_dirty = False
        self._coalesce_loaded_sessions()
        print(
            f"[memory] loaded sessions={len(self._sessions)} aliases={len(self._session_aliases)} "
            f"path={self._memory_path}"
        )

    def _load_persistent_memory_legacy(self) -> None:
        try:
            raw = json.loads(self._memory_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[warn] memory load failed: {exc}")
            return

        version = int(raw.get("version", 0) or 0)
        sessions = raw.get("sessions", {})
        if isinstance(sessions, dict):
            for key, data in sessions.items():
                if not isinstance(data, dict):
                    continue
                key_str = str(key).strip()
                if not key_str:
                    continue
                if version >= 3:
                    relpath = str(data.get("path", "")).strip() or self._memory_session_relpath(key_str)
                    meta = {
                        "dir": key_str,
                        "path": relpath,
                        "short": [str(x) for x in (data.get("short", []) or [])][-max(4, self.cfg.memory_short_max_items) :]
                        if isinstance(data.get("short", []), list)
                        else [],
                        "summary": str(data.get("summary", ""))[: max(120, self.cfg.memory_summary_max_chars)],
                        "muted": bool(data.get("muted", False)),
                        "titles": self._normalize_session_titles(data.get("titles", [])),
                        "history_count": int(data.get("history_count", 0) or 0),
                        "updated_at": int(data.get("updated_at", 0) or 0),
                    }
                    self._session_index[key_str] = meta
                    self._sessions[key_str] = self._session_state_from_index(key_str, meta)
                    continue

                short = data.get("short", [])
                history = data.get("history", [])
                titles = data.get("titles", [])
                history_items = self._normalize_history_items(history)
                if (not history_items) and isinstance(short, list):
                    for entry in short:
                        val = str(entry).strip()
                        if not val:
                            continue
                        role = "assistant" if val.startswith("A:") else ("user" if val.startswith("U:") else "unknown")
                        text = val.split(":", 1)[1].strip() if ":" in val else val
                        history_items.append(
                            {
                                "role": role,
                                "content_type": "text",
                                "text": text[:220],
                                "sender": "",
                                "source": "legacy",
                                "observed_at": 0,
                            }
                        )
                sess = SessionState(
                    short=[str(x) for x in short][-max(4, self.cfg.memory_short_max_items) :]
                    if isinstance(short, list)
                    else [],
                    history=history_items[
                        -max(0, self.cfg.memory_history_max_items) :
                    ]
                    if self.cfg.memory_history_max_items > 0
                    else history_items,
                    summary=str(data.get("summary", ""))[: max(120, self.cfg.memory_summary_max_chars)],
                    muted=bool(data.get("muted", False)),
                    titles=set(self._normalize_session_titles(titles)),
                    loaded=True,
                )
                self._sessions[key_str] = sess
                self._session_index[key_str] = {
                    "dir": key_str,
                    "short": list(sess.short),
                    "summary": sess.summary,
                    "muted": bool(sess.muted),
                    "titles": sorted(sess.titles),
                    "history_count": len(sess.history),
                    "updated_at": int(time.time()),
                }

        aliases = raw.get("aliases", {})
        if isinstance(aliases, dict):
            for k, v in aliases.items():
                k2 = str(k).strip()
                v2 = str(v).strip()
                if k2 and v2:
                    self._session_aliases[self._title_key(k2) or k2] = self._title_key(v2) or v2
        self._memory_dirty = False
        self._coalesce_loaded_sessions()
        print(
            f"[memory] loaded (legacy) sessions={len(self._sessions)} aliases={len(self._session_aliases)} "
            f"path={self._memory_path}"
        )

    def _save_persistent_memory(self) -> None:
        if not self.cfg.memory_enabled:
            return
        if not self._memory_dirty:
            return
        try:
            now_ts = int(time.time())
            index_data = self.chat_history.load_index()
            sessions_payload: dict[str, dict] = {}
            all_keys = set(self._session_index.keys()) | set(self._sessions.keys())
            for key in sorted(all_keys):
                sess = self._sessions.get(key)
                meta = self._session_index.get(key)
                if sess is None:
                    if isinstance(meta, dict):
                        sessions_payload[key] = dict(meta)
                    continue
                display_title = next(iter(sess.titles), "")
                dir_name = self.chat_history._safe_name(
                    display_title if display_title else key
                )
                sessions_payload[key] = {
                    "dir": dir_name,
                    "short": sess.short[-max(4, self.cfg.memory_short_max_items) :],
                    "summary": sess.summary[: max(120, self.cfg.memory_summary_max_chars)],
                    "muted": bool(sess.muted),
                    "titles": sorted(sess.titles),
                    "message_count": len(sess.history),
                    "updated_at": now_ts,
                }
                self._session_index[key] = dict(sessions_payload[key])
                if sess.loaded and sess.history:
                    for rec in sess.history:
                        self.chat_history.append(display_title or key, rec)
            index_data["sessions"] = sessions_payload
            index_data["aliases"] = self._session_aliases
            index_data["version"] = index_data.get("version", 1)
            self.chat_history.save_index(index_data)
            self._memory_dirty = False
        except Exception as exc:
            print(f"[warn] memory save failed: {exc}")

    def _is_preview_meaningfully_changed(self, old: str, new: str) -> bool:
        if not new:
            return False
        if old == new:
            return False
        ratio = SequenceMatcher(a=old, b=new).ratio()
        return ratio < 0.93

    def _is_self_echo(self, preview_norm: str, sent_norm: str) -> bool:
        if not preview_norm or not sent_norm:
            return False

        # Direct contain checks handle truncation from list preview.
        if preview_norm in sent_norm or sent_norm in preview_norm:
            return True

        # Fuzzy match preview vs sent prefix for OCR drift.
        n = min(len(preview_norm), len(sent_norm), 24)
        if n >= 6:
            ratio = SequenceMatcher(a=preview_norm[:n], b=sent_norm[:n]).ratio()
            if ratio >= 0.78:
                return True

        # Full-string fuzzy match for reordered/truncated OCR fragments.
        full_ratio = SequenceMatcher(a=preview_norm, b=sent_norm).ratio()
        if full_ratio >= 0.62:
            return True
        block = SequenceMatcher(a=preview_norm, b=sent_norm).find_longest_match(
            0, len(preview_norm), 0, len(sent_norm)
        )
        common = block.size
        min_len = min(len(preview_norm), len(sent_norm))
        if min_len >= 8 and common >= 8 and (common / max(1, min_len)) >= 0.45:
            return True
        return False

    def _is_preview_refresh_from_self(
        self,
        row: ChatRowState,
        preview_norm: str,
        prev_sent_norm: str,
        recent_sent_norm: str,
    ) -> bool:
        # Only apply to preview-refresh path to avoid suppressing real unread events.
        preview_payload_norm = self._normalize_preview(self._strip_preview_decorations(row.preview))
        preview_norms = [x for x in {preview_norm, preview_payload_norm} if x]
        if not preview_norms:
            return False

        sent_norms: list[str] = []
        for x in (prev_sent_norm, recent_sent_norm):
            if x:
                sent_norms.append(x)
        for txt in self._recent_assistant_replies(row, limit=3):
            n = self._normalize_preview(txt)
            if n:
                sent_norms.append(n)
        if not sent_norms:
            return False

        for pn in preview_norms:
            for sn in sent_norms:
                if self._is_self_echo(pn, sn):
                    return True
        return False

    def _get_recent_sent_for_row(self, row: ChatRowState, now: float) -> str:
        # 3 minutes anti-loop window
        window_sec = 180.0
        key = self._title_key(row.title)
        data = self._sent_by_title.get(key)
        if not data:
            return ""
        sent_norm, ts = data
        if now - ts > window_sec:
            self._sent_by_title.pop(key, None)
            return ""
        return sent_norm

    def _remember_sent_for_row(self, row: ChatRowState, sent_norm: str, now: float) -> None:
        if not sent_norm:
            return
        with self._state_lock:
            key = self._title_key(row.title)
            if key:
                self._sent_by_title[key] = (sent_norm, now)

    def _is_group_chat(self, row: ChatRowState) -> bool:
        title = row.title or ""
        for prefix in self.cfg.group_title_prefixes:
            if prefix and title.startswith(prefix):
                return True

        # Detached windows already expose the chat title. Their previews are
        # normalized as "sender: message" for both private and group chats, so
        # sender-prefix detection would incorrectly classify private chats as groups.
        if self.cfg.receiver_mode == "detached_windows":
            return False

        if self.cfg.group_detect_sender_prefix and "：" in (row.preview or ""):
            sender = row.preview.split("：", 1)[0].strip()
            if 1 <= len(sender) <= 24:
                return True

        return False

    def _has_sender_prefix(self, text: str) -> bool:
        raw = (text or "").strip()
        if not raw:
            return False

        sep = "：" if "：" in raw else (":" if ":" in raw else "")
        if not sep:
            return False

        sender = raw.split(sep, 1)[0].strip(" []【】()（）")
        if not (1 <= len(sender) <= 24):
            return False

        # Filter out marker-style prefixes such as [有人@我]
        deny = ["有人@我", "@我", "新消息"]
        if any(x in sender for x in deny):
            return False
        return True

    def _normalize_title_text(self, text: str) -> str:
        return self._normalize_session_title_token(text)

    def _pick_chat_title_region(self, is_group: bool | None) -> "RegionRatio":
        if is_group is True and self.cfg.chat_title_region_group_enabled:
            return self.cfg.chat_title_region_group
        if is_group is False and self.cfg.chat_title_region_private_enabled:
            return self.cfg.chat_title_region_private
        return self.cfg.chat_title_region

    def _probe_chat_header_text(
        self,
        bounds: "WindowBounds",
        is_group: bool | None = None,
    ) -> tuple[str, int, int, tuple[int, int, int, int], "RegionRatio"]:
        region = self._pick_chat_title_region(is_group)
        x = bounds.x + int(bounds.width * region.x)
        y = bounds.y + int(bounds.height * region.y)
        w = int(bounds.width * region.w)
        h = int(bounds.height * region.h)
        if w <= 0 or h <= 0:
            return "", 0, 0, (x, y, w, h), region
        # Header probe runs frequently during focus verification; avoid
        # high-res capture here to reduce long-run memory growth.
        shot = screenshot_region(x, y, w, h, high_res=False)
        bgr = self._to_np_rgb(shot)[:, :, ::-1]
        ocr_engine, _ = self._ensure_local_ocr()
        lines = ocr_engine.detect_lines(bgr)
        raw_count = len(lines)
        if not lines:
            return "", raw_count, 0, (x, y, w, h), region
        valid: list[tuple[float, float, str]] = []
        for line in lines:
            txt = (line.text or "").strip()
            if not txt or _TIME_RE.match(txt):
                continue
            norm = self._normalize_title_text(txt)
            if not norm:
                continue
            valid.append((line.y_center, line.x_center, txt))
        if not valid:
            return "", raw_count, 0, (x, y, w, h), region
        valid.sort(key=lambda it: (it[0], it[1]))
        text = " ".join([x[2] for x in valid])[:120]
        return text, raw_count, len(valid), (x, y, w, h), region

    def _extract_chat_header_text(self, bounds: "WindowBounds", is_group: bool | None = None) -> str:
        text, _, _, _, _ = self._probe_chat_header_text(bounds, is_group=is_group)
        return text

    def _log_recover_title_probe(
        self,
        bounds: "WindowBounds",
        *,
        mode_tag: str,
        forced_is_group: bool,
    ) -> None:
        probes: list[tuple[str, bool | None]] = [
            ("forced", forced_is_group),
            ("default", None),
            ("group", True),
            ("private", False),
        ]
        for label, hint in probes:
            text, raw_count, valid_count, box, region = self._probe_chat_header_text(
                bounds,
                is_group=hint,
            )
            x, y, w, h = box
            txt = self._fit_col(text, 18) if text else "<empty>"
            print(
                f"[{mode_tag}] title-probe {label:<7} | "
                f"raw={raw_count:>2} valid={valid_count:>2} | "
                f"ratio=({region.x:.6f},{region.y:.6f},{region.w:.6f},{region.h:.6f}) | "
                f"px=({x},{y},{w},{h}) | text={txt}"
            )

    def _is_chat_header_matched(self, expected_title: str, actual_header: str) -> bool:
        exp = self._normalize_title_text(expected_title)
        got = self._normalize_title_text(actual_header)
        if not exp:
            return True
        if not got:
            return False
        if exp in got or (len(got) >= 2 and got in exp):
            return True
        ratio = SequenceMatcher(a=exp, b=got).ratio()
        return ratio >= 0.62

    def _is_admin_session(self, row: ChatRowState) -> bool:
        if not self.cfg.admin_commands_enabled:
            return False
        title_key = self._title_key(row.title)
        if not title_key:
            return False
        for t in self.cfg.admin_session_titles:
            if title_key == self._title_key(t):
                return True
        return False

    def _should_handle_admin_command_locally(self, is_admin: bool) -> bool:
        """Keep slash commands local except when OpenClaw owns the turn."""
        return bool(is_admin) and self.cfg.processing_mode != "long_bridge"

    def _is_immediate_reply_event(self, row: ChatRowState, reason: str) -> bool:
        if reason == "mention":
            return True
        return not self._is_group_chat(row)

    def _normal_reply_interval_active(self) -> bool:
        return float(self.cfg.normal_reply_interval_sec) > 0.0

    def _is_normal_reply_event(self, row: ChatRowState, reason: str) -> bool:
        return (not self._is_immediate_reply_event(row, reason)) and reason == "new_message"

    def _detached_message_excerpt(self, message: dict, *, max_width: int | None = None) -> str:
        text = re.sub(r"\s+", " ", str(message.get("text", "") or "")).strip()
        if not text:
            text = str(message.get("image_hash", "") or "").strip()
        if not text:
            text = "-"
        width = max_width if max_width is not None else max(24, self._term_width() - 96)
        return self._fit_col(text, width)

    def _detached_message_fields(self, message: dict, *, include_raw: bool = False) -> str:
        sender = self._fit_col(str(message.get("sender", "") or "-"), 12)
        content_type = str(message.get("content_type", "") or "-")
        parts = [f"sender={sender}", f"type={content_type}"]
        if include_raw:
            raw = self._fit_col(str(message.get("sender_raw", "") or "-"), 12)
            parts.insert(1, f"raw={raw}")
        parts.append(f"text={self._detached_message_excerpt(message)}")
        return " ".join(parts)

    def _strip_sender_prefix(self, text: str) -> str:
        raw = (text or "").strip()
        if not raw:
            return ""
        sep = "：" if "：" in raw else (":" if ":" in raw else "")
        if not sep:
            return raw
        left, right = raw.split(sep, 1)
        if 1 <= len(left.strip()) <= 24:
            return right.strip()
        return raw

    def _extract_admin_command_text(
        self,
        row: ChatRowState,
        context_snapshot: ChatContextSnapshot,
    ) -> str:
        prefix = (self.cfg.admin_command_prefix or "/").strip() or "/"
        candidates: list[str] = []
        if context_snapshot.last_user_message:
            candidates.append(context_snapshot.last_user_message)
        if context_snapshot.last_side == "other":
            candidates.append(context_snapshot.last_line)
        candidates.append(self._strip_sender_prefix(row.preview or ""))

        for text in candidates:
            clean = (text or "").strip()
            if clean.startswith(prefix):
                return clean
        return ""

    def _display_session_name(self, key: str) -> str:
        sess = self._sessions.get(key)
        if sess and sess.titles:
            return sorted(sess.titles, key=len)[-1]
        return key

    def _normalize_agent_task_state(self, task: object) -> dict:
        if not isinstance(task, dict):
            return {}
        status = re.sub(r"\s+", " ", str(task.get("status", "")).strip()).lower()[:24]
        if status in {"", "idle"}:
            return {}
        if status not in {"running", "blocked", "waiting_user", "done"}:
            status = "running"
        normalized = {
            "status": status,
            "goal": re.sub(r"\s+", " ", str(task.get("goal", "")).strip())[:160],
            "plan": re.sub(r"\s+", " ", str(task.get("plan", "")).strip())[:280],
            "next_step": re.sub(r"\s+", " ", str(task.get("next_step", "")).strip())[:160],
            "last_result": re.sub(r"\s+", " ", str(task.get("last_result", "")).strip())[:280],
            "blocked_reason": re.sub(
                r"\s+",
                " ",
                str(task.get("blocked_reason", "")).strip(),
            )[:160],
            "continue_on_heartbeat": bool(task.get("continue_on_heartbeat", False)),
        }
        if status != "running":
            normalized["continue_on_heartbeat"] = False
        return normalized

    def _merge_agent_task_state(
        self,
        current: object,
        planned: object,
        *,
        last_result: str = "",
    ) -> dict:
        merged = self._normalize_agent_task_state(current)
        incoming = self._normalize_agent_task_state(planned)
        if incoming:
            merged.update({k: v for k, v in incoming.items() if v not in ("", None)})
            merged["status"] = incoming.get("status", merged.get("status", "running"))
            merged["continue_on_heartbeat"] = bool(incoming.get("continue_on_heartbeat", False))
        if last_result:
            merged["last_result"] = re.sub(r"\s+", " ", str(last_result).strip())[:280]
        status = str(merged.get("status", "")).strip().lower()
        if status in {"", "idle"}:
            return {}
        if status == "blocked" and (not merged.get("blocked_reason")) and last_result:
            merged["blocked_reason"] = re.sub(r"\s+", " ", str(last_result).strip())[:160]
        if status != "running":
            merged["continue_on_heartbeat"] = False
        return self._normalize_agent_task_state(merged)

    def _format_agent_task_state_for_prompt(self, task: object) -> str:
        state = self._normalize_agent_task_state(task)
        if not state:
            return ""
        lines = [
            f"status={state.get('status', '')}",
            f"goal={state.get('goal', '') or '无'}",
        ]
        if state.get("plan"):
            lines.append(f"plan={state['plan']}")
        if state.get("next_step"):
            lines.append(f"next_step={state['next_step']}")
        if state.get("last_result"):
            lines.append(f"last_result={state['last_result']}")
        if state.get("blocked_reason"):
            lines.append(f"blocked_reason={state['blocked_reason']}")
        lines.append(
            "continue_on_heartbeat="
            + ("true" if bool(state.get("continue_on_heartbeat", False)) else "false")
        )
        return "\n".join(lines)

    def _load_agent_task_state(self, *, session_key: str, title: str) -> dict:
        return {}

    def _save_agent_task_state(self, *, session_key: str, title: str, task: object) -> dict:
        return self._normalize_agent_task_state(task)

    def _virtual_session_row(
        self,
        *,
        session_key: str,
        title: str,
        preview: str,
        row_idx: int = -2,
    ) -> ChatRowState:
        title_text = (title or session_key or "__agent_task__").strip()[:80]
        alias = self._title_key(title_text)
        if alias and session_key:
            self._remember_session_alias(alias, session_key)
        if session_key and title_text:
            self._remember_session_title(session_key, title_text)
        return ChatRowState(
            row_idx=row_idx,
            text=preview[:80] or title_text,
            title=title_text,
            preview=preview[:80] or title_text,
            has_mention=False,
            has_unread_badge=False,
            fingerprint=f"agent-task-{session_key or alias or int(time.time())}",
            click_x_ratio=0.0,
            click_y_ratio=0.0,
        )

    def _resolve_session_key_by_query(self, query: str) -> str | None:
        q = (query or "").strip()
        if not q:
            return None
        q_key = self._title_key(q)
        if q_key in self._session_aliases:
            return self._session_aliases[q_key]
        if q_key in self._sessions:
            return q_key

        # Fuzzy match by known titles.
        best_key = None
        best_score = 0.0
        for key, sess in self._sessions.items():
            names = set(sess.titles)
            names.add(key)
            for name in names:
                n_key = self._title_key(name)
                if q_key and (q_key in n_key or n_key in q_key):
                    return key
                score = SequenceMatcher(a=q_key or q, b=n_key or name).ratio()
                if score > best_score:
                    best_score = score
                    best_key = key
        if best_score >= 0.72:
            return best_key
        return None

    def _is_row_muted(self, row: ChatRowState) -> bool:
        key = self._session_key_for_row(row, remember=False)
        sess = self._sessions.get(key)
        return bool(sess and sess.muted)

    def _handle_admin_command(self, cmd_line: str) -> str:
        if not cmd_line:
            return ""

        cmd = cmd_line.strip()
        parts = cmd.split()
        if not parts:
            return ""
        action = parts[0].lower()

        if action in ("/help", "/?"):
            return (
                "可用命令: /sessions, /mute 会话, /unmute 会话, "
                "/reset 会话, /merge 源会话 -> 目标会话, "
                "/remember 长期记忆"
            )

        if action == "/sessions":
            if not self._sessions:
                return "暂无会话缓存。"
            rows = []
            for key, sess in list(self._sessions.items())[:12]:
                tag = "MUTE" if sess.muted else "ON"
                rows.append(f"{tag}:{self._display_session_name(key)}")
            return "会话列表: " + " ; ".join(rows)

        if action in ("/mute", "/unmute", "/reset"):
            if len(parts) < 2:
                return f"用法: {action} 会话名"
            query = " ".join(parts[1:]).strip()
            key = self._resolve_session_key_by_query(query)
            if not key:
                return f"未找到会话: {query}"
            sess = self._get_or_create_session(key)
            if action == "/mute":
                sess.muted = True
                self._memory_dirty = True
                return f"已静音: {self._display_session_name(key)}"
            if action == "/unmute":
                sess.muted = False
                self._memory_dirty = True
                return f"已恢复: {self._display_session_name(key)}"
            sess.short = []
            sess.history = []
            sess.summary = ""
            self._summary_turn_counter[key] = 0
            self._memory_dirty = True
            return f"已重置记忆: {self._display_session_name(key)}"

        if action == "/merge":
            body = cmd[len(parts[0]) :].strip()
            if "->" in body:
                src_q, dst_q = [x.strip() for x in body.split("->", 1)]
            else:
                merge_parts = body.split()
                if len(merge_parts) < 2:
                    return "用法: /merge 源会话 -> 目标会话"
                src_q, dst_q = merge_parts[0], " ".join(merge_parts[1:])
            src_key = self._resolve_session_key_by_query(src_q)
            dst_key = self._resolve_session_key_by_query(dst_q)
            if not src_key or not dst_key:
                return f"会话未找到: src={src_q}, dst={dst_q}"
            if src_key == dst_key:
                return "源会话和目标会话相同，无需合并。"
            src_name = self._display_session_name(src_key)
            dst_name = self._display_session_name(dst_key)
            self._merge_session_keys(src_key=src_key, dst_key=dst_key, dst_title=dst_name)
            return f"已合并: {src_name} -> {dst_name}"

        if action == "/remember":
            body = cmd[len(parts[0]) :].strip()
            if not body:
                return "用法: /remember 需要长期记住的内容"
            return "已写入 MEMORY.md"

        return f"未知命令: {cmd_line}"

    def _available_agent_tools(self, *, is_admin: bool) -> list[str]:
        tools = [
            "list_skills",
            "read_skill",
            "read_memory",
            "recall_memory",
            "remember_fact",
            "write_memory",
            "update_skill",
            "write_skill",
            "delete_skill",
            "read_impression",
            "update_impression",
            "write_impression",
            "read_chat_history",
            "read_chat_history_by_date",
            "summarize_chat_history",
            "search_chat_history",
            "run_python",
        ]
        if self._has_image_generation_tool():
            tools.append("generate_image")
        if self._has_image_editing_tool():
            tools.append("edit_image")
        if self._has_wow_character_url_tool():
            tools.append("build_wow_character_url")
        if self._has_weather_tool():
            tools.append("query_weather")
        tools.append("fetch_url")
        tools.append("browse_url")
        # Keep planner whitelist aligned with runtime capability checks:
        # Expose one aggregate search tool. Provider-specific names remain
        # accepted by the executor for compatibility, but the planner should
        # not choose between providers.
        if self._has_web_search_tool():
            tools.append("web_search")
        if is_admin:
            tools.extend(
                [
                    "remember_long_term",
                    "mute_session",
                    "unmute_session",
                ]
            )
        return tools

    def _active_web_search_provider(self) -> str:
        provider = str(self.cfg.web_search_provider or "").strip().lower()
        if provider in ("volc_ark", "volc-ark", "ark", "volc", "volces"):
            return "volc_ark"
        if provider == "brave":
            return "brave"
        if provider in ("agent_reach", "agent-reach", "agentreach", "reach", "exa"):
            return "agent_reach"
        return "tavily"

    def _active_search_mode(self) -> str:
        return "llm" if self._active_web_search_provider() == "volc_ark" else "machine"

    def _resolve_tavily_api_key(self) -> str:
        if self.cfg.tavily_api_key:
            return self.cfg.tavily_api_key
        env_name = (self.cfg.tavily_api_key_env or "").strip()
        if not env_name:
            return ""
        return os.getenv(env_name, "")

    def _resolve_brave_api_key(self) -> str:
        if self.cfg.brave_api_key:
            return self.cfg.brave_api_key
        env_name = (self.cfg.brave_api_key_env or "").strip()
        if not env_name:
            return ""
        return os.getenv(env_name, "")

    def _resolve_volc_ark_api_key(self) -> str:
        if self.cfg.volc_ark_api_key:
            return self.cfg.volc_ark_api_key
        env_name = (self.cfg.volc_ark_api_key_env or "").strip()
        if not env_name:
            return ""
        return os.getenv(env_name, "")

    def _resolve_image_generation_api_key(self) -> str:
        return self.image_generator.resolve_api_key()

    def _resolve_web_search_api_key(self, provider: str) -> str:
        if provider == "volc_ark":
            return ""
        if provider == "brave":
            return self._resolve_brave_api_key()
        if provider == "agent_reach":
            return ""
        return self._resolve_tavily_api_key()

    def _web_search_enabled(self, provider: str) -> bool:
        if provider == "volc_ark":
            return False
        if provider == "agent_reach":
            return bool(self.cfg.agent_reach_enabled)
        if provider == "brave":
            return bool(self.cfg.brave_enabled)
        return bool(self.cfg.tavily_enabled)

    def _web_search_max_results(self, provider: str) -> int:
        if provider == "volc_ark":
            return max(1, int(self.cfg.volc_ark_limit))
        if provider == "agent_reach":
            return max(1, int(self.cfg.agent_reach_max_results))
        if provider == "brave":
            return max(1, int(self.cfg.brave_max_results))
        return max(1, int(self.cfg.tavily_max_results))

    def _web_search_key_hint(self, provider: str) -> str:
        if provider == "volc_ark":
            return "use tool=web_search_volc in llm mode"
        if provider == "agent_reach":
            return "install/configure mcporter + exa (via agent-reach install)"
        if provider == "brave":
            env_name = (self.cfg.brave_api_key_env or "").strip()
            return (
                f"config.brave_api_key or env {env_name}"
                if env_name
                else "config.brave_api_key"
            )
        env_name = (self.cfg.tavily_api_key_env or "").strip()
        return (
            f"config.tavily_api_key or env {env_name}"
            if env_name
            else "config.tavily_api_key"
        )

    def _has_web_search_tool(self) -> bool:
        return any(
            (
                self._has_tavily_search_tool(),
                self._has_brave_search_tool(),
                self._has_volc_web_search_tool(),
            )
        )

    def _has_tavily_search_tool(self) -> bool:
        return bool(self.cfg.tavily_enabled and self._resolve_tavily_api_key())

    def _has_brave_search_tool(self) -> bool:
        return bool(self.cfg.brave_enabled and self._resolve_brave_api_key())

    def _has_wow_character_url_tool(self) -> bool:
        return (Path("data/skills/wow-character-link") / "builder.py").is_file()

    def _volc_web_search_key_hint(self) -> str:
        env_name = (self.cfg.volc_ark_api_key_env or "").strip()
        return (
            f"config.volc_ark_api_key or env {env_name}"
            if env_name
            else "config.volc_ark_api_key"
        )

    def _resolve_qweather_api_key(self) -> str:
        if self.cfg.qweather_api_key:
            return self.cfg.qweather_api_key
        env_name = (self.cfg.qweather_api_key_env or "").strip()
        return os.getenv(env_name, "").strip() if env_name else ""

    def _make_qweather_client(self) -> QWeatherClient:
        return QWeatherClient(
            api_host=self.cfg.qweather_api_host,
            auth_type=self.cfg.qweather_auth_type,
            api_key=self._resolve_qweather_api_key(),
            jwt_key_id=self.cfg.qweather_jwt_key_id,
            jwt_project_id=self.cfg.qweather_jwt_project_id,
            jwt_private_key_path=self.cfg.qweather_jwt_private_key_path,
            jwt_ttl_sec=self.cfg.qweather_jwt_ttl_sec,
            timeout_sec=self.cfg.qweather_timeout_sec,
        )

    def _has_weather_tool(self) -> bool:
        if not bool(self.cfg.weather_enabled):
            return False
        if str(self.cfg.weather_provider or "").strip().lower() != "qweather":
            return False
        return self.qweather.is_configured()

    def _weather_status_text(self) -> str:
        if not bool(self.cfg.weather_enabled):
            return "disabled (tool=query_weather weather_enabled=false)"
        if str(self.cfg.weather_provider or "").strip().lower() != "qweather":
            return f"blocked (tool=query_weather unsupported provider={self.cfg.weather_provider})"
        return self.qweather.status_text()

    def _query_weather(self, args: dict) -> str:
        location = re.sub(
            r"\s+",
            " ",
            str(args.get("location", "") or args.get("city", "") or "").strip(),
        )[:80]
        if not location:
            location = str(self.cfg.weather_default_location or "").strip()
        days_raw = args.get("days", self.cfg.weather_forecast_days)
        try:
            days = int(days_raw)
        except Exception:
            days = int(self.cfg.weather_forecast_days)
        mode = str(args.get("mode", "") or "both").strip().lower()
        date_value = re.sub(r"\s+", "", str(args.get("date", "") or args.get("target_date", "")).strip())[:20]
        day_offset = None
        if "day_offset" in args:
            try:
                day_offset = int(args.get("day_offset"))
            except Exception:
                day_offset = None
        return self.qweather.query(
            location=location,
            days=days,
            mode=mode,
            date=date_value,
            day_offset=day_offset,
        )

    def _has_volc_web_search_tool(self) -> bool:
        if not bool(self.cfg.volc_ark_enabled):
            return False
        if not str(self.cfg.volc_ark_model or "").strip():
            return False
        return bool(self._resolve_volc_ark_api_key())

    def _image_generation_key_hint(self) -> str:
        return self.image_generator.key_hint()

    def _has_image_generation_tool(self) -> bool:
        return self.image_generator.is_available()

    def _image_generation_status_text(self) -> str:
        return f"{self.image_generator.status_text()} (tool=generate_image)"

    def _has_image_editing_tool(self) -> bool:
        return self.image_editor.is_available()

    def _image_editing_status_text(self) -> str:
        return f"{self.image_editor.status_text()} (tool=edit_image)"

    def _generate_edited_image_file(
        self,
        *,
        prompt: str,
        image_path: str = "",
        image_url: str = "",
        size: str = "",
    ) -> Path:
        return self.image_editor.edit_file(
            prompt=prompt,
            image_path=image_path,
            image_url=image_url,
            size=size,
        )

    def _volc_web_search_status_text(self) -> str:
        if not bool(self.cfg.volc_ark_enabled):
            return "disabled (tool=web_search_volc provider=volc_ark volc_ark_enabled=false)"
        if not str(self.cfg.volc_ark_model or "").strip():
            return "blocked (tool=web_search_volc missing volc_ark_model)"
        if self._resolve_volc_ark_api_key():
            return (
                "available (tool=web_search_volc provider=volc_ark "
                f"model={self.cfg.volc_ark_model} limit={max(1, min(20, int(self.cfg.volc_ark_limit)))})"
            )
        return f"blocked (tool=web_search_volc missing api key: {self._volc_web_search_key_hint()})"

    def _web_search_status_text(self) -> str:
        states = {
            "tavily": self._has_tavily_search_tool(),
            "brave": self._has_brave_search_tool(),
            "volc_ark": self._has_volc_web_search_tool(),
        }
        enabled = [name for name, available in states.items() if available]
        disabled = [name for name, available in states.items() if not available]
        status = "available" if enabled else "blocked"
        return (
            f"{status} (tool=web_search mode=aggregate "
            f"enabled={','.join(enabled) or 'none'} "
            f"unavailable={','.join(disabled) or 'none'})"
        )

    @staticmethod
    def _plan_has_tool(actions: list[dict] | None, tool_name: str) -> bool:
        if not actions:
            return False
        target = str(tool_name or "").strip()
        if not target:
            return False
        for item in actions:
            if not isinstance(item, dict):
                continue
            if str(item.get("tool", "")).strip() == target:
                return True
        return False

    @staticmethod
    def _is_opinion_prompt(text: str) -> bool:
        raw = re.sub(r"\s+", " ", (text or "").strip())
        if not raw:
            return False
        markers = (
            "有何高见",
            "怎么看",
            "你怎么看",
            "看法",
            "咋看",
            "怎么评价",
            "分析一下",
            "分析下",
            "说说",
        )
        return any(m in raw for m in markers)

    @staticmethod
    def _extract_lookup_topic_from_context(chat_context: str) -> str:
        raw = re.sub(r"\s+", " ", (chat_context or "").strip())
        if not raw:
            return ""
        cues = (
            "新闻",
            "消息",
            "分享",
            "链接",
            "http",
            "伊朗",
            "以色列",
            "美国",
            "欧洲",
            "俄乌",
            "制裁",
            "外交",
            "战争",
            "公告",
            "更新",
        )
        segments = [seg.strip() for seg in raw.split("|") if seg.strip()]
        for seg in reversed(segments):
            clean = re.sub(r"^[A-Za-z]:\s*", "", seg).strip()
            clean = re.sub(r"^\[[^\]]+\]\s*", "", clean).strip()
            if not clean:
                continue
            if any(c in clean for c in cues):
                return clean[:80]
        return ""

    @staticmethod
    def _is_payment_gate_reply(text: str) -> bool:
        raw = re.sub(r"\s+", " ", (text or "").strip())
        if not raw:
            return False
        marker_groups = (
            ("红包", "发个红包", "先发红包"),
            ("稿费",),
            ("茶钱",),
            ("转账", "打钱", "给钱"),
            ("收费", "付费"),
        )
        return any(any(m in raw for m in group) for group in marker_groups)

    @staticmethod
    def _is_deferred_reply_hint(text: str) -> bool:
        raw = re.sub(r"\s+", " ", (text or "").strip())
        if not raw:
            return False
        lowered = raw.lower()
        markers = (
            "稍等",
            "等我",
            "我去查",
            "我先查",
            "这就帮你查",
            "这就搜",
            "别催",
            "我去看看",
            "待会",
            "一会",
            "先发红包",
            "先给红包",
            "发个红包",
            "稿费",
            "给钱再说",
            "先转",
        )
        return any(m in raw for m in markers) or any(m in lowered for m in markers)

    @staticmethod
    def _is_meta_reply_hint(text: str) -> bool:
        raw = re.sub(r"\s+", " ", (text or "").strip())
        if not raw:
            return False
        lower = raw.lower()
        meta_patterns = (
            r"^顺着.+话题",
            r"^围绕.+话题",
            r"^针对.+(话题|问题|场景)",
            r"^延续.+(语气|风格|口吻|氛围)",
            r"^按.+(语气|风格|口吻|人设|氛围)",
            r"^用.+(语气|风格|口吻)",
            r".+符合.+(群聊|聊天|当前).*(氛围|风格|语气)",
            r".+(调侃|吐槽).+即可$",
            r".+(回一句|回两句|回复一句|回复两句|简短回复|简单回复)",
            r".+(不要太|别太).+(正式|生硬|严肃)",
        )
        if any(re.search(pat, raw) for pat in meta_patterns):
            return True
        meta_markers = (
            "reply_hint",
            "主回复链路",
            "用户可见回复",
            "直接回",
            "可以回",
            "建议回",
            "适合回",
            "回复即可",
            "作为回复",
        )
        return any(marker in raw or marker in lower for marker in meta_markers)

    def _normalize_planner_reply_hint(
        self,
        hint: str,
        *,
        reason: str,
        latest_message: str,
    ) -> str:
        raw = re.sub(r"\s+", " ", (hint or "").strip())
        if not raw:
            return ""
        if self._is_payment_gate_reply(raw):
            return ""
        if self._is_meta_reply_hint(raw):
            return ""
        if not self._is_deferred_reply_hint(raw):
            return raw

        latest = re.sub(r"\s+", " ", (latest_message or "").strip())
        if ("画像" in raw) or ("画像" in latest) or ("bug" in raw.lower()) or ("参数" in raw):
            return "你说得对，这条是会话画像跑偏了。我按你这条消息重新给结论。"
        if reason == "mention":
            return "收到你的@，这条我直接回答，不走延后话术。"
        return "收到，这条我按正经模式直接处理。"

    @staticmethod
    def _contextual_fallback_reply(
        *,
        reason: str,
        latest_message: str,
        fallback: str,
    ) -> str:
        latest = re.sub(r"\s+", " ", (latest_message or "").strip())
        if "画像" in latest:
            return "你说得对，这里是会话画像跑偏了，我按当前消息重新给结论。"
        if any(k in latest for k in ("有何高见", "怎么看", "看法", "分析", "为什么", "怎么", "？", "?")):
            return "这条我不玩梗，按你的问题直接给结论。"
        if reason == "mention":
            return "收到你的@，这条我按正经模式直接回复。"
        return fallback

    @staticmethod
    def _has_lookup_observation(text: str) -> bool:
        raw = re.sub(r"\s+", " ", (text or "").strip())
        if not raw:
            return False
        return ("网页检索[" in raw) or ("记忆检索[" in raw)

    @staticmethod
    def _action_args_preview(args: dict, *, limit: int = 110) -> str:
        try:
            text = json.dumps(args or {}, ensure_ascii=False, sort_keys=True)
        except Exception:
            text = "{}"
        compact = re.sub(r"\s+", " ", str(text)).strip()
        if len(compact) > limit:
            return compact[:limit] + "..."
        return compact

    def _plan_actions_with_live_progress(
        self,
        *,
        planner: LlmReplyGenerator,
        round_idx: int,
        rounds: int,
        row: ChatRowState,
        started_at: float,
        request_kwargs: dict,
    ) -> dict:
        # Keep users informed when planner call is slow; the LLM request itself is blocking.
        wait_notice_sec = 8.0
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(planner.plan_actions, **request_kwargs)
            while True:
                try:
                    return future.result(timeout=wait_notice_sec)
                except concurrent.futures.TimeoutError:
                    if self.cfg.log_verbose:
                        elapsed = time.monotonic() - started_at
                        print(
                            f"[agent-plan] round={round_idx:>2}/{rounds:<2} "
                            f"wait  row={row.row_idx:>2} elapsed={elapsed:.1f}s "
                            f"timeout={float(planner.cfg.timeout_sec):.1f}s"
                        )

    def _run_agent_planner_loop(
        self,
        *,
        planner: LlmReplyGenerator,
        row: ChatRowState,
        reason: str,
        is_group: bool,
        is_admin: bool,
        latest_message: str,
        chat_context: str,
        environment_context: str,
        session_context: str,
        workspace_context: str,
        memory_recall: str,
        tools: list[str],
        per_round_max_actions: int,
        max_rounds_override: int | None = None,
        max_total_actions_override: int | None = None,
    ) -> tuple[str, str, int, bool, dict]:
        if not tools:
            return memory_recall, "", 0, True, {}

        rounds = 1
        if self.cfg.agent_plan_loop_enabled:
            rounds = max(1, int(self.cfg.agent_plan_max_rounds))
        if max_rounds_override is not None:
            rounds = max(1, min(rounds, int(max_rounds_override)))
        per_round_limit = max(1, int(per_round_max_actions))
        total_limit = max(per_round_limit, int(self.cfg.agent_plan_max_total_actions))
        if max_total_actions_override is not None:
            total_limit = max(1, min(total_limit, int(max_total_actions_override)))
        repeat_limit = max(1, int(self.cfg.agent_plan_repeat_limit))
        obs_limit = max(1200, int(self.cfg.agent_plan_observation_max_chars))

        base_memory = memory_recall or ""
        merged_memory = base_memory
        observed_blocks: list[str] = []
        action_repeat: dict[str, int] = {}
        total_actions = 0
        final_hint = ""
        final_hint_terminal = False
        session_key = self._session_key_for_row(row)
        task_state = self._load_agent_task_state(session_key=session_key, title=row.title)

        for round_idx in range(1, rounds + 1):
            remaining = total_limit - total_actions
            if remaining <= 0:
                break
            this_round_max = min(per_round_limit, remaining)
            plan_started = 0.0
            try:
                plan_started = time.monotonic()
                if self.cfg.log_verbose:
                    print(
                        f"[agent-plan] round={round_idx:>2}/{rounds:<2} "
                        f"start row={row.row_idx:>2} "
                        f"title={self._fit_col(row.title, 14)} "
                        f"max={this_round_max:>2} timeout={float(planner.cfg.timeout_sec):.1f}s"
                    )
                plan_request = {
                    "title": row.title,
                    "is_group": is_group,
                    "reason": reason,
                    "latest_message": latest_message,
                    "chat_context": chat_context,
                    "environment_context": environment_context,
                    "session_context": session_context,
                    "workspace_context": workspace_context,
                    "memory_recall": merged_memory,
                    "available_tools": tools,
                    "max_actions": this_round_max,
                    "agent_task_state": self._format_agent_task_state_for_prompt(task_state),
                    "planner_round_idx": round_idx,
                    "planner_round_total": rounds,
                    "planner_total_actions_limit": total_limit,
                    "planner_total_actions_used": total_actions,
                    "planner_total_actions_remaining": remaining,
                }
                plan = self._plan_actions_with_live_progress(
                    planner=planner,
                    round_idx=round_idx,
                    rounds=rounds,
                    row=row,
                    started_at=plan_started,
                    request_kwargs=plan_request,
                )
                elapsed = time.monotonic() - plan_started
                planned_actions = (
                    plan.get("actions")
                    if isinstance(plan, dict) and isinstance(plan.get("actions"), list)
                    else []
                )
                raw_hint = str(plan.get("reply_hint", "")).strip() if isinstance(plan, dict) else ""
                planned_task = {}
                if isinstance(plan, dict):
                    planned_task = self._normalize_agent_task_state(plan.get("task"))
                    if planned_task:
                        task_state = self._merge_agent_task_state(task_state, planned_task)
                hint = self._normalize_planner_reply_hint(
                    raw_hint,
                    reason=reason,
                    latest_message=latest_message or row.preview or row.text or "",
                )
                if hint:
                    final_hint = hint
                    final_hint_terminal = not bool(planned_actions)
                if self.cfg.log_verbose:
                    names = ",".join(
                        str(item.get("tool", "")).strip()
                        for item in planned_actions
                        if isinstance(item, dict) and str(item.get("tool", "")).strip()
                    )
                    print(
                        f"[agent-plan] round={round_idx:>2}/{rounds:<2} "
                        f"done  row={row.row_idx:>2} elapsed={elapsed:.2f}s "
                        f"actions={len(planned_actions):>2} tools={names or '-'}"
                    )
                    if hint:
                        print(
                            f"             hint={self._fit_col(hint, max(24, self._term_width() - 19))}"
                        )
                    elif raw_hint:
                        print("             hint=(dropped by normalize)")

                filtered_actions: list[dict] = []
                for action in planned_actions:
                    if not isinstance(action, dict):
                        continue
                    tool = str(action.get("tool", "")).strip()
                    args = action.get("args") if isinstance(action.get("args"), dict) else {}
                    if not tool:
                        continue
                    signature = f"{tool}|{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
                    if action_repeat.get(signature, 0) >= repeat_limit:
                        continue
                    filtered_actions.append(action)
                if not filtered_actions:
                    task_state = self._save_agent_task_state(
                        session_key=session_key,
                        title=row.title,
                        task=task_state,
                    )
                    if self.cfg.log_verbose:
                        print(
                            f"[agent-plan] round={round_idx:>2}/{rounds:<2} "
                            f"stop (no executable actions)"
                        )
                    break

                trace, observations = self._execute_agent_actions(
                    row,
                    filtered_actions,
                    is_admin=is_admin,
                    max_actions_override=this_round_max,
                )
                executed_actions = min(len(filtered_actions), this_round_max)
                total_actions += executed_actions
                for action in filtered_actions[:executed_actions]:
                    tool = str(action.get("tool", "")).strip()
                    args = action.get("args") if isinstance(action.get("args"), dict) else {}
                    signature = f"{tool}|{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
                    action_repeat[signature] = int(action_repeat.get(signature, 0)) + 1

                if trace and self.cfg.log_verbose:
                    print(
                        f"[agent] row={row.row_idx:>2} | round={round_idx:>2} "
                        f"actions={executed_actions:>2}"
                    )
                    for ln in trace.split("\n"):
                        if ln.strip():
                            print(f"        {ln}")

                if observations:
                    observed_blocks.append(f"[round {round_idx}]\n{observations}")
                    observed = "\n\n".join(observed_blocks)
                    merged_memory = (
                        f"{base_memory}\n\n[工具执行结果]\n{observed}".strip()
                    )[:obs_limit]
                    task_state = self._merge_agent_task_state(
                        task_state,
                        planned_task,
                        last_result=observations,
                    )
                    task_state = self._save_agent_task_state(
                        session_key=session_key,
                        title=row.title,
                        task=task_state,
                    )
                else:
                    if trace:
                        task_state = self._merge_agent_task_state(
                            task_state,
                            planned_task,
                            last_result=trace,
                        )
                        task_state = self._save_agent_task_state(
                            session_key=session_key,
                            title=row.title,
                            task=task_state,
                        )
                    if self.cfg.log_verbose:
                        print(
                            f"[agent-plan] round={round_idx:>2}/{rounds:<2} "
                            "stop (no observation)"
                        )
                    break
            except Exception as exc:
                if self.cfg.log_verbose:
                    if plan_started > 0.0:
                        elapsed = time.monotonic() - plan_started
                        print(
                            f"[agent-plan] fail  round={round_idx:>2}/{rounds:<2} "
                            f"row={row.row_idx:>2} elapsed={elapsed:.2f}s"
                        )
                    else:
                        print(f"[agent-plan] fail  round={round_idx:>2}/{rounds:<2} row={row.row_idx:>2}")
                if not self.cfg.agent_actions_fail_open:
                    raise
                print(f"[warn] agent action planner failed, fail-open: {exc}")
                break

        planned_reply = ""
        observed_text = "\n\n".join(observed_blocks)
        has_lookup_observation = self._has_lookup_observation(observed_text)
        has_lookup_error = ("网页检索[" in observed_text) and ("失败" in observed_text)
        if has_lookup_error:
            planned_reply = (
                "我这轮已经发起检索了，但接口超时/异常，没拿到可用结果。"
                "我马上缩短关键词再重查一轮。"
            )
        if total_actions > 0 and final_hint and not final_hint_terminal:
            final_hint = ""
        if has_lookup_observation and final_hint:
            final_hint = ""
        if final_hint and self._is_deferred_reply_hint(final_hint):
            rewritten = self._normalize_planner_reply_hint(
                final_hint,
                reason=reason,
                latest_message=latest_message or row.preview or row.text or "",
            )
            if rewritten and (not self._is_deferred_reply_hint(rewritten)):
                final_hint = rewritten
                if self.cfg.log_verbose:
                    print(
                        f"[agent] normalize deferred reply_hint row={row.row_idx:>2} "
                        f"title={self._fit_col(row.title, 14)}"
                    )
            else:
                final_hint = self._contextual_fallback_reply(
                    reason=reason,
                    latest_message=latest_message or row.preview or row.text or "",
                    fallback="这条我按正经模式处理。",
                )
            if self.cfg.log_verbose:
                print(
                    f"[agent] rewrite deferred reply_hint row={row.row_idx:>2} "
                    f"title={self._fit_col(row.title, 14)}"
                )
        if final_hint and not planned_reply:
            fallback = self.cfg.reply_on_mention if reason == "mention" else self.cfg.reply_on_new_message
            planned_reply = self._sanitize_generated_reply(final_hint, fallback=fallback)
        task_state = self._save_agent_task_state(
            session_key=session_key,
            title=row.title,
            task=task_state,
        )
        return merged_memory, planned_reply, total_actions, True, task_state

    @staticmethod
    def _clean_web_query(query: str) -> str:
        return re.sub(r"\s+", " ", query or "").strip()[:120]

    @staticmethod
    def _compact_web_text(raw: object, *, limit: int) -> str:
        return re.sub(r"\s+", " ", str(raw or "")).strip()[:limit]

    def _build_wow_character_url(self, args: dict) -> dict:
        skill_dir = Path("data/skills/wow-character-link")
        module_path = skill_dir / "builder.py"
        if not module_path.is_file():
            return {"ok": False, "error": f"wow-character-link builder not found: {module_path}"}
        spec = importlib.util.spec_from_file_location("weauto_wow_character_link_builder", module_path)
        if spec is None or spec.loader is None:
            return {"ok": False, "error": "failed to load wow-character-link builder"}
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            build = getattr(module, "build")
            return dict(
                build(
                    character=str(args.get("character", "") or ""),
                    server=str(args.get("server", "") or ""),
                    player=str(args.get("player", "") or ""),
                    class_name=str(args.get("class_name", "") or ""),
                    skill_dir=skill_dir,
                )
            )
        except Exception as exc:
            return {"ok": False, "error": f"wow-character-link failed: {exc}"}

    @staticmethod
    def _format_wow_character_result(result: dict) -> str:
        if result.get("ok"):
            return str(result.get("message") or result.get("url") or "").strip()
        candidates = result.get("candidates")
        if isinstance(candidates, list) and candidates:
            rows = []
            for item in candidates[:8]:
                if not isinstance(item, dict):
                    continue
                rows.append(
                    f"{item.get('player', '')} / {item.get('character', '')} / "
                    f"{item.get('server', '')} / {item.get('class', '')}"
                )
            if rows:
                return str(result.get("error") or "没唯一匹配到角色") + "\n" + "\n".join(rows)
        return str(result.get("error") or "没构建出角色链接")

    @staticmethod
    def _strip_html(raw: str) -> str:
        text = str(raw or "")
        text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        return re.sub(r"\s+", " ", text).strip()

    def _open_url(self, req: urllib.request.Request, *, timeout: float, use_proxy: bool):
        if use_proxy:
            proxy_url = (
                os.getenv("HTTPS_PROXY")
                or os.getenv("https_proxy")
                or os.getenv("HTTP_PROXY")
                or os.getenv("http_proxy")
            )
            if proxy_url:
                handler = urllib.request.ProxyHandler({"https": proxy_url, "http": proxy_url})
                opener = urllib.request.build_opener(handler)
                return opener.open(req, timeout=timeout)
            return urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context())
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(req, timeout=timeout)

    def _fetch_url(self, url: str, *, max_chars: int = 6000, use_proxy: bool = True) -> str:
        clean_url = str(url or "").strip()
        if not re.match(r"^https?://", clean_url, flags=re.I):
            raise RuntimeError("fetch_url only supports http/https URL")
        req = urllib.request.Request(
            clean_url,
            method="GET",
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.5",
            },
        )
        try:
            with self._open_url(req, timeout=15, use_proxy=use_proxy) as resp:
                raw_bytes = resp.read(max(1000, int(max_chars)) * 4)
                charset = resp.headers.get_content_charset() or "utf-8"
                content_type = str(resp.headers.get("content-type", "")).lower()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"fetch_url http error: {exc.code} {detail[:300]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"fetch_url network error: {exc}") from exc
        text = raw_bytes.decode(charset, errors="replace")
        if "html" in content_type or "<html" in text[:500].lower():
            text = self._strip_html(text)
        else:
            text = re.sub(r"\s+", " ", html.unescape(text)).strip()
        return text[: max(200, int(max_chars))]

    def _browse_url(self, url: str, *, max_chars: int = 10000, use_proxy: bool = True) -> str:
        clean_url = str(url or "").strip()
        if not re.match(r"^https?://", clean_url, flags=re.I):
            raise RuntimeError("browse_url only supports http/https URL")
        try:
            from playwright.sync_api import sync_playwright
        except Exception:
            fetched = self._fetch_url(clean_url, max_chars=max_chars, use_proxy=use_proxy)
            return "Playwright unavailable; used fetch_url fallback:\n" + fetched
        try:
            with sync_playwright() as p:
                launch_kwargs = {"headless": True}
                if use_proxy:
                    proxy_url = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") or os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
                    if proxy_url:
                        launch_kwargs["proxy"] = {"server": proxy_url}
                browser = p.chromium.launch(**launch_kwargs)
                page = browser.new_page()
                try:
                    page.goto(clean_url, wait_until="domcontentloaded", timeout=15000)
                    page.wait_for_timeout(1500)
                    text = page.evaluate("() => document.body ? document.body.innerText : ''") or ""
                finally:
                    browser.close()
            text = re.sub(r"\s+", " ", str(text)).strip()
            if not text:
                return "page returned empty text"
            return text[: max(200, int(max_chars))]
        except Exception as exc:
            fetched = self._fetch_url(clean_url, max_chars=max_chars, use_proxy=use_proxy)
            return f"browse_url failed ({exc}); used fetch_url fallback:\n{fetched}"

    def _web_search_with_provider(self, provider: str, query: str) -> tuple[str, str]:
        clean_provider = str(provider or "").strip().lower()
        if clean_provider == "volc_ark":
            return clean_provider, self._volc_web_search(query)
        if clean_provider == "agent_reach":
            return clean_provider, self._agent_reach_search(query)
        if clean_provider == "brave":
            return clean_provider, self._brave_search(query)
        return "tavily", self._tavily_search(query)

    def _aggregate_web_search(self, query: str) -> str:
        clean_query = self._clean_web_query(query)
        if not clean_query:
            return ""

        searches: list[tuple[str, object]] = [
            ("tavily", self._tavily_search),
            ("brave", self._brave_search),
            ("volc_ark", self._volc_web_search),
        ]
        available = {
            "tavily": self._has_tavily_search_tool(),
            "brave": self._has_brave_search_tool(),
            "volc_ark": self._has_volc_web_search_tool(),
        }
        results = {name: "" for name, _ in searches}

        runnable = [(name, search) for name, search in searches if available[name]]
        if runnable:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(runnable)) as pool:
                futures = {
                    pool.submit(search, clean_query): name
                    for name, search in runnable
                }
                for future in concurrent.futures.as_completed(futures):
                    name = futures[future]
                    try:
                        results[name] = str(future.result() or "").strip()
                    except Exception as exc:
                        print(
                            f"[warn] aggregate web search provider={name} failed: "
                            f"{self._compact_web_text(exc, limit=240)}"
                        )

        sections = []
        for name, _ in searches:
            text = results[name]
            sections.append(f"[{name}]\n{text}" if text else f"[{name}]\n")
        return "\n\n".join(sections)

    def _format_web_search_text(
        self,
        *,
        summary: str,
        rows: list[tuple[str, str, str]],
        max_results: int,
    ) -> str:
        lines: list[str] = []
        clean_summary = self._compact_web_text(summary, limit=240)
        if clean_summary:
            lines.append(f"摘要: {clean_summary}")
        for title, url_item, snippet in rows[: max(1, int(max_results))]:
            row = " | ".join(
                [
                    x
                    for x in [
                        self._compact_web_text(title, limit=90),
                        self._compact_web_text(url_item, limit=160),
                        self._compact_web_text(snippet, limit=180),
                    ]
                    if x
                ]
            )
            if row:
                lines.append(row)
        return "\n".join(lines)[:1200]

    @staticmethod
    def _clean_image_prompt(raw: object, *, limit: int = 280) -> str:
        return ImageGenerator.clean_prompt(raw, limit=limit)

    def _normalize_image_size(self, raw: object) -> str:
        return self.image_generator.normalize_size(raw)

    def _generate_image_file(self, *, prompt: str, size: str = "") -> Path:
        try:
            return self.image_generator.generate_file(prompt=prompt, size=size)
        except ImageGenerationError as exc:
            raise RuntimeError(str(exc)) from exc

    def _generate_edited_image_file(
        self,
        *,
        prompt: str,
        image_path: str = "",
        image_url: str = "",
        size: str = "",
    ) -> Path:
        try:
            return self.image_editor.edit_file(
                prompt=prompt,
                image_path=image_path,
                image_url=image_url,
                size=size,
            )
        except ImageEditingError as exc:
            raise RuntimeError(str(exc)) from exc

    def _describe_image_file(self, image_path: str, *, prompt: str = "") -> str:
        path = Path(image_path).expanduser()
        if not path.is_file():
            raise RuntimeError(f"image file not found: {path}")
        with Image.open(path) as image:
            return self.llm_reply.describe_image(image.convert("RGB"), prompt=prompt)

    def _web_search(self, query: str) -> tuple[str, str]:
        return "tavily+brave+volc_ark", self._aggregate_web_search(query)

    @staticmethod
    def _extract_json_from_text(raw: str) -> object | None:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
        decoder = json.JSONDecoder()
        for idx, ch in enumerate(text):
            if ch not in "{[":
                continue
            try:
                value, _ = decoder.raw_decode(text, idx)
            except Exception:
                continue
            return value
        return None

    def _collect_rows_from_payload(self, payload: object) -> list[tuple[str, str, str]]:
        rows: list[tuple[str, str, str]] = []

        def add_row(title: object, url_item: object, snippet: object) -> None:
            t = self._compact_web_text(title, limit=90)
            u = self._compact_web_text(url_item, limit=160)
            s = self._compact_web_text(snippet, limit=180)
            if t or u or s:
                rows.append((t, u, s))

        def walk(node: object) -> None:
            if isinstance(node, list):
                for item in node:
                    walk(item)
                return
            if not isinstance(node, dict):
                return

            title = node.get("title", "")
            url_item = node.get("url", node.get("link", node.get("source", "")))
            snippet = (
                node.get("snippet")
                or node.get("content")
                or node.get("description")
                or node.get("text")
                or ""
            )
            if title or url_item or snippet:
                add_row(title, url_item, snippet)

            for key in (
                "results",
                "items",
                "entries",
                "documents",
                "organic",
                "data",
                "web",
                "hits",
            ):
                child = node.get(key)
                if child is not None:
                    walk(child)

            # Some MCP responses return text blocks containing JSON payload.
            content = node.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    text = str(block.get("text", "")).strip()
                    if not text:
                        continue
                    parsed = self._extract_json_from_text(text)
                    if parsed is None:
                        add_row("", "", text)
                    else:
                        walk(parsed)

        walk(payload)

        deduped: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for item in rows:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def _collect_rows_from_agent_reach_text(self, raw_text: str) -> list[tuple[str, str, str]]:
        text = str(raw_text or "")
        if not text:
            return []

        # Handle both plain text output and JS-inspect style escaped strings.
        normalized = text.replace("\\r\\n", "\n").replace("\\n", "\n")
        normalized = re.sub(r"'\s*\+\s*'", "", normalized)

        rows: list[tuple[str, str, str]] = []
        chunks = normalized.split("Title:")
        for part in chunks[1:]:
            block = part.strip()
            if not block:
                continue
            title_line, _, rest = block.partition("\n")
            title = title_line.strip().strip("'\"")

            url_match = re.search(r"(?:^|\n)\s*URL:\s*(.+)", rest)
            url_item = url_match.group(1).strip().strip("'\"") if url_match else ""

            text_match = re.search(r"(?:^|\n)\s*Text:\s*", rest)
            snippet = rest[text_match.end() :].strip() if text_match else rest.strip()
            snippet = snippet.strip("'\"")

            t = self._compact_web_text(title, limit=90)
            u = self._compact_web_text(url_item, limit=160)
            s = self._compact_web_text(snippet, limit=180)
            if t or u or s:
                rows.append((t, u, s))

        deduped: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for item in rows:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def _agent_reach_search(self, query: str) -> str:
        clean_query = self._clean_web_query(query)
        if not clean_query:
            return ""

        cmd = str(self.cfg.agent_reach_mcporter_cmd or "mcporter").strip()
        if not cmd:
            raise RuntimeError("agent_reach mcporter command missing")
        mcporter = shutil.which(cmd)
        if not mcporter:
            raise RuntimeError(f"agent_reach command not found: {cmd}")

        max_results = self._web_search_max_results("agent_reach")
        timeout_sec = max(1.0, float(self.cfg.agent_reach_timeout_sec))
        escaped_query = clean_query.replace("\\", "\\\\").replace('"', '\\"')
        expr = f'exa.web_search_exa(query: "{escaped_query}", numResults: {max_results})'
        proc = subprocess.run(
            [mcporter, "call", expr],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        stdout = str(proc.stdout or "").strip()
        stderr = str(proc.stderr or "").strip()
        if proc.returncode != 0:
            detail = self._compact_web_text(stderr or stdout, limit=300)
            raise RuntimeError(f"agent_reach exa call failed: {detail or proc.returncode}")

        payload = self._extract_json_from_text(stdout)
        rows: list[tuple[str, str, str]] = []
        if payload is None:
            rows = self._collect_rows_from_agent_reach_text(stdout)
            if not rows:
                detail = self._compact_web_text(stdout, limit=300)
                raise RuntimeError(f"agent_reach exa response parse failed: {detail}")

        summary = ""
        if isinstance(payload, dict):
            summary = self._compact_web_text(
                payload.get("summary", payload.get("answer", payload.get("message", ""))),
                limit=240,
            )
            rows = self._collect_rows_from_payload(payload)
        elif isinstance(payload, list):
            rows = self._collect_rows_from_payload(payload)
        elif not rows:
            rows = self._collect_rows_from_agent_reach_text(stdout)
        if (not summary) and rows:
            summary = self._compact_web_text("；".join(x[2] for x in rows if x[2])[:800], limit=240)
        return self._format_web_search_text(summary=summary, rows=rows, max_results=max_results)

    def _tavily_search(self, query: str) -> str:
        clean_query = self._clean_web_query(query)
        if not clean_query:
            return ""
        api_key = self._resolve_tavily_api_key()
        if not api_key:
            raise RuntimeError("tavily api key missing")

        max_results = self._web_search_max_results("tavily")
        base = (self.cfg.tavily_base_url or "https://api.tavily.com").rstrip("/")
        url = base if base.endswith("/search") else (base + "/search")
        payload = {
            "api_key": api_key,
            "query": clean_query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": True,
            "include_raw_content": False,
        }
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            ssl_ctx = ssl.create_default_context()
            with urllib.request.urlopen(
                req,
                timeout=max(1.0, float(self.cfg.tavily_timeout_sec)),
                context=ssl_ctx,
            ) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"tavily http error: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"tavily network error: {exc}") from exc

        data = json.loads(raw)
        answer = self._compact_web_text(data.get("answer", ""), limit=240)
        results = data.get("results") if isinstance(data.get("results"), list) else []
        rows: list[tuple[str, str, str]] = []
        for item in results[:max_results]:
            if not isinstance(item, dict):
                continue
            rows.append(
                (
                    self._compact_web_text(item.get("title", ""), limit=90),
                    self._compact_web_text(item.get("url", ""), limit=160),
                    self._compact_web_text(item.get("content", ""), limit=180),
                )
            )
        return self._format_web_search_text(summary=answer, rows=rows, max_results=max_results)

    def _brave_search(self, query: str) -> str:
        clean_query = self._clean_web_query(query)
        if not clean_query:
            return ""
        api_key = self._resolve_brave_api_key()
        if not api_key:
            raise RuntimeError("brave api key missing")

        max_results = self._web_search_max_results("brave")
        timeout_sec = max(1.0, float(self.cfg.brave_timeout_sec))
        base = (self.cfg.brave_base_url or "https://api.search.brave.com/res/v1/web").rstrip("/")
        url = base if base.endswith("/search") else (base + "/search")
        query_url = f"{url}?{urllib.parse.urlencode({'q': clean_query, 'count': max_results})}"
        req = urllib.request.Request(
            url=query_url,
            method="GET",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": api_key,
            },
        )
        try:
            ssl_ctx = ssl.create_default_context()
            with urllib.request.urlopen(
                req,
                timeout=timeout_sec,
                context=ssl_ctx,
            ) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"brave http error: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"brave network error: {exc}") from exc

        data = json.loads(raw)
        web_obj = data.get("web") if isinstance(data.get("web"), dict) else {}
        results = web_obj.get("results")
        if not isinstance(results, list):
            results = data.get("results")
        if not isinstance(results, list):
            results = []

        rows: list[tuple[str, str, str]] = []
        for item in results[:max_results]:
            if not isinstance(item, dict):
                continue
            description = self._compact_web_text(item.get("description", ""), limit=180)
            extra = ""
            extra_raw = item.get("extra_snippets")
            if isinstance(extra_raw, list):
                extra_parts: list[str] = []
                for part in extra_raw[:2]:
                    clean_part = self._compact_web_text(part, limit=90)
                    if clean_part:
                        extra_parts.append(clean_part)
                if extra_parts:
                    extra = " ".join(extra_parts)
            snippet = self._compact_web_text(
                f"{description} {extra}".strip(),
                limit=180,
            )
            rows.append(
                (
                    self._compact_web_text(item.get("title", ""), limit=90),
                    self._compact_web_text(item.get("url", ""), limit=160),
                    snippet,
                )
            )

        summary_parts = [snippet for _, _, snippet in rows if snippet][:2]
        summary = self._compact_web_text("；".join(summary_parts), limit=240)
        return self._format_web_search_text(summary=summary, rows=rows, max_results=max_results)

    def _volc_web_search(self, query: str) -> str:
        clean_query = self._clean_web_query(query)
        if not clean_query:
            return ""

        api_key = self._resolve_volc_ark_api_key()
        if not api_key:
            raise RuntimeError("volc_ark api key missing")
        model = str(self.cfg.volc_ark_model or "").strip()
        if not model:
            raise RuntimeError("volc_ark model missing")

        limit = max(1, min(20, int(self.cfg.volc_ark_limit)))
        max_keyword = max(1, min(50, int(self.cfg.volc_ark_max_keyword)))
        timeout_sec = max(1.0, float(self.cfg.volc_ark_timeout_sec))
        base = (self.cfg.volc_ark_base_url or "https://ark.cn-beijing.volces.com/api/v3").rstrip("/")
        url = base if base.endswith("/responses") else (base + "/responses")

        payload = {
            "model": model,
            # Trusted single-search mode: do not allow multiple search rounds in one response.
            "max_tool_calls": 1,
            "tools": [
                {
                    "type": "web_search",
                    "max_keyword": max_keyword,
                    "limit": limit,
                }
            ],
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": clean_query,
                        }
                    ],
                }
            ],
        }
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        try:
            ssl_ctx = ssl.create_default_context()
            with urllib.request.urlopen(
                req,
                timeout=timeout_sec,
                context=ssl_ctx,
            ) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"volc_ark http error: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"volc_ark network error: {exc}") from exc

        data = json.loads(raw)
        output = data.get("output") if isinstance(data.get("output"), list) else []
        answer = ""
        rows: list[tuple[str, str, str]] = []
        seen_urls: set[str] = set()

        for item in output:
            if not isinstance(item, dict):
                continue
            if str(item.get("type", "")).strip() != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if not answer:
                    answer = self._compact_web_text(block.get("text", ""), limit=240)
                annotations = block.get("annotations")
                if not isinstance(annotations, list):
                    continue
                for ann in annotations:
                    if not isinstance(ann, dict):
                        continue
                    url_item = self._compact_web_text(
                        ann.get("url", ann.get("website_url", ann.get("source_url", ""))),
                        limit=160,
                    )
                    if (not url_item) or (url_item in seen_urls):
                        continue
                    seen_urls.add(url_item)
                    title = self._compact_web_text(
                        ann.get("title", ann.get("source_title", "")),
                        limit=90,
                    )
                    rows.append((title, url_item, ""))

        if (not answer) and rows:
            answer = self._compact_web_text(
                "；".join(x[0] or x[1] for x in rows if (x[0] or x[1])),
                limit=240,
            )
        if (not answer) and (not rows):
            raise RuntimeError("volc_ark response missing message/annotations")
        return self._format_web_search_text(summary=answer, rows=rows, max_results=limit)

    def _execute_agent_actions(
        self,
        row: ChatRowState,
        actions: list[dict] | None,
        *,
        is_admin: bool,
        max_actions_override: int | None = None,
    ) -> tuple[str, str]:
        return self.action_processor.execute_agent_actions(
            row,
            actions,
            is_admin=is_admin,
            max_actions_override=max_actions_override,
        )

    def _load_heartbeat_tasks(self) -> str:
        return ""

    @staticmethod
    def _parse_heartbeat_direct_actions(tasks_text: str) -> list[dict]:
        actions: list[dict] = []
        seen: set[str] = set()
        for line in (tasks_text or "").splitlines():
            clean = line.strip()
            if not clean:
                continue
            lowered = clean.lower()
            if "maintain_memory" in lowered and "maintain_memory" not in seen:
                days = 7
                m = re.search(r"days\s*[:=]\s*(\d{1,2})", clean, flags=re.IGNORECASE)
                if not m:
                    m = re.search(r"days\"\s*:\s*(\d{1,2})", clean, flags=re.IGNORECASE)
                if not m:
                    m = re.search(r"(\d{1,2})\s*天", clean)
                if m:
                    try:
                        days = int(m.group(1))
                    except Exception:
                        days = 7
                days = max(1, min(14, days))
                actions.append(
                    {
                        "tool": "maintain_memory",
                        "args": {"days": days},
                        "reason": "heartbeat direct task",
                    }
                )
                seen.add("maintain_memory")
            if (
                "maintain_person_impressions" in lowered
                and "maintain_person_impressions" not in seen
            ):
                days = 30
                max_people = 6
                m = re.search(r"days\s*[:=]\s*(\d{1,4})", clean, flags=re.IGNORECASE)
                if not m:
                    m = re.search(r"days\"\s*:\s*(\d{1,4})", clean, flags=re.IGNORECASE)
                if not m:
                    m = re.search(r"(\d{1,4})\s*天", clean)
                if m:
                    try:
                        days = int(m.group(1))
                    except Exception:
                        days = 30
                mp = re.search(r"max_people\s*[:=]\s*(\d{1,3})", clean, flags=re.IGNORECASE)
                if not mp:
                    mp = re.search(r"max_people\"\s*:\s*(\d{1,3})", clean, flags=re.IGNORECASE)
                if not mp:
                    mp = re.search(r"最多\s*(\d{1,3})\s*人", clean)
                if mp:
                    try:
                        max_people = int(mp.group(1))
                    except Exception:
                        max_people = 6
                actions.append(
                    {
                        "tool": "maintain_person_impressions",
                        "args": {
                            "days": max(1, min(3650, days)),
                            "max_people": max(1, min(200, max_people)),
                        },
                        "reason": "heartbeat direct task",
                    }
                )
                seen.add("maintain_person_impressions")
            if "refine_persona_files" in lowered and "refine_persona_files" not in seen:
                actions.append(
                    {
                        "tool": "refine_persona_files",
                        "args": {},
                        "reason": "heartbeat direct task",
                    }
                )
                seen.add("refine_persona_files")
        return actions

    @staticmethod
    def _update_managed_block(
        raw: str,
        *,
        start_marker: str,
        end_marker: str,
        body: str,
    ) -> str:
        current = raw or ""
        payload = (body or "").strip()
        if not payload:
            return current
        block = f"{start_marker}\n{payload}\n{end_marker}"
        s_idx = current.find(start_marker)
        e_idx = current.find(end_marker)
        if s_idx >= 0 and e_idx > s_idx:
            e_end = e_idx + len(end_marker)
            merged = current[:s_idx].rstrip() + "\n\n" + block + current[e_end:]
            return merged.rstrip() + "\n"
        merged = current.rstrip()
        if merged:
            merged += "\n\n"
        merged += block + "\n"
        return merged

    def _collect_recent_daily_memory(self, *, days: int, max_chars: int = 6000) -> str:
        return ""

    def _heartbeat_llm_backends(self) -> list[tuple[str, LlmReplyGenerator]]:
        backends: list[tuple[str, LlmReplyGenerator]] = []
        seen: set[int] = set()
        for name, llm in (
            ("heartbeat", self.llm_heartbeat),
            ("summary", self.llm_summary),
        ):
            if not llm.is_enabled():
                continue
            marker = id(llm)
            if marker in seen:
                continue
            seen.add(marker)
            backends.append((name, llm))
        return backends

    def _heartbeat_maintain_memory(self, *, days: int) -> tuple[bool, str]:
        return False, "skip (memory maintenance archived)"

    def _heartbeat_maintain_person_impressions(
        self,
        *,
        days: int,
        max_people: int,
    ) -> tuple[bool, str]:
        return False, "skip (person impression maintenance archived)"

    def _heartbeat_refine_persona_files(self) -> tuple[bool, str]:
        return False, "skip (persona refinement archived)"

    def _heartbeat_virtual_row(self) -> ChatRowState:
        title = (
            str(self.cfg.admin_session_titles[0]).strip()
            if self.cfg.admin_session_titles
            else "__heartbeat__"
        )
        return ChatRowState(
            row_idx=-1,
            text="heartbeat",
            title=title or "__heartbeat__",
            preview="heartbeat",
            has_mention=False,
            has_unread_badge=False,
            fingerprint=f"heartbeat-{int(time.time())}",
            click_x_ratio=0.0,
            click_y_ratio=0.0,
        )

    def _run_heartbeat_pending_agent_task(self, now: float, rows: list[ChatRowState]) -> bool:
        return False

    def _available_heartbeat_tools(self) -> list[str]:
        return [
            "read_memory",
            "recall_memory",
            "remember_fact",
            "read_impression",
            "update_impression",
            "read_chat_history",
            "read_chat_history_by_date",
            "summarize_chat_history",
            "search_chat_history",
        ]

    def _heartbeat_identity_text(self) -> str:
        paths = [Path("data/identity.md"), Path("data/config/IDENTITY.md")]
        parts: list[str] = []
        for path in paths:
            try:
                if path.is_file():
                    text = path.read_text(encoding="utf-8", errors="replace").strip()
                    if text and text not in parts:
                        parts.append(text)
            except OSError:
                continue
        return "\n\n".join(parts)[:3000]

    def _heartbeat_recent_chat_activity(self, *, max_chats: int = 10, max_lines: int = 60) -> str:
        items: list[tuple[int, str, list[dict]]] = []
        for key in list(self._sessions.keys()):
            sess = self._get_or_create_session(key)
            history = [x for x in sess.history if isinstance(x, dict)]
            if not history:
                continue
            last_ts = max(int(x.get("observed_at", 0) or 0) for x in history[-20:])
            titles = sorted(str(x).strip() for x in sess.titles if str(x).strip())
            title = titles[0] if titles else key
            items.append((last_ts, title, history[-10:]))
        items.sort(key=lambda x: x[0], reverse=True)

        lines: list[str] = []
        for _, title, history in items[:max_chats]:
            if len(lines) >= max_lines:
                break
            lines.append(f"[{title}]")
            for record in history:
                if len(lines) >= max_lines:
                    break
                ts = int(record.get("observed_at", 0) or 0)
                ts_text = datetime.fromtimestamp(ts).strftime("%m-%d %H:%M") if ts > 0 else "--"
                role = str(record.get("role", "unknown")).strip()
                sender = str(record.get("sender", "")).strip()
                text = re.sub(r"\s+", " ", str(record.get("text", ""))).strip()
                if not text:
                    continue
                who = sender if sender else ("萨比" if role == "assistant" else role)
                lines.append(f"{ts_text} {who}: {text[:180]}")
        return "\n".join(lines)[:6000]

    def _heartbeat_recent_people(self, *, max_people: int, per_chat_records: int) -> list[str]:
        if max_people <= 0:
            return []
        candidates: list[tuple[int, str]] = []
        skip_names = {
            "",
            "user",
            "unknown",
            "unknown user",
            "sender",
            "member",
            "someone",
            "other",
            "self",
            "assistant",
            "system",
            "系统",
            "用户",
            "未知",
            "未知用户",
            "群成员",
            "其他人",
            "大家",
            "有人",
            "萨比",
            "sabi",
        }
        for key in list(self._sessions.keys()):
            sess = self._get_or_create_session(key)
            history = [x for x in sess.history[-max(1, per_chat_records) :] if isinstance(x, dict)]
            for record in history:
                if str(record.get("role", "")).strip().lower() != "user":
                    continue
                text = re.sub(r"\s+", " ", str(record.get("text", "") or "")).strip()
                if not text:
                    continue
                sender = re.sub(r"\s+", " ", str(record.get("sender", "") or "")).strip()
                if not sender:
                    sender = re.sub(r"\s+", " ", str(record.get("sender_raw", "") or "")).strip()
                if not sender:
                    continue
                sender_key = sender.strip().lower()
                if sender_key in skip_names or sender.startswith("群-") or sender.endswith("群"):
                    continue
                if re.fullmatch(r"[\d_ -]+", sender_key):
                    continue
                name = self._resolve_person_name(sender) or sender
                name = re.sub(r"_[0-9a-f]{8}$", "", name).strip()[:40]
                if not name or name.lower() in skip_names or name.startswith("群-"):
                    continue
                candidates.append((int(record.get("observed_at", 0) or 0), name))

        candidates.sort(key=lambda item: item[0], reverse=True)
        people: list[str] = []
        seen: set[str] = set()
        for _, name in candidates:
            key = re.sub(r"\s+", "", name).lower()
            if key in seen:
                continue
            seen.add(key)
            people.append(name)
            if len(people) >= max(1, max_people):
                break
        return people

    def _heartbeat_current_state(self, now: float, *, target_people: list[str] | None = None) -> str:
        parts: list[str] = []
        core = self.agent_memory.read("core").strip()
        if core:
            parts.append(f"[memories]\n{core}")
        timeline = self.agent_memory.read("timeline").strip()
        if timeline:
            parts.append(f"[timeline]\n{timeline}")
        people_names = self.agent_people.list()
        if people_names:
            parts.append(
                "[people impressions]\n"
                "Person impressions are available on demand, not preloaded. "
                "These are records about people, not the agent's core/timeline memory. "
                "Use read_impression with a canonical Chinese name before updating an existing record.\n"
                f"Available names: {', '.join(people_names[:120])}"
            )
        targets = [str(x).strip() for x in (target_people or []) if str(x).strip()]
        if targets:
            parts.append(
                "[heartbeat target people]\n"
                "Short-term recent speakers selected by code. Read and update only these people unless recent chat clearly requires otherwise.\n"
                + "\n".join(f"- {name}" for name in targets)
            )
        recent = self._heartbeat_recent_chat_activity()
        if recent:
            parts.append(f"[recent chat activity]\n{recent}")
        now_text = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")
        parts.append(f"[current_time]\n{now_text}")
        return "\n\n".join(parts)

    def _heartbeat_prompt_text(self, now: float, *, target_people: list[str] | None = None) -> str:
        targets = [str(x).strip() for x in (target_people or []) if str(x).strip()]
        target_text = ", ".join(targets) if targets else "none"
        return (
            f"[system]\n{self.cfg.llm_reply.system_prompt}\n\n"
            f"[identity]\n{self._heartbeat_identity_text()}\n\n"
            f"[current state]\n{self._heartbeat_current_state(now, target_people=targets)}\n\n"
            f"[heartbeat instruction]\n{self.cfg.heartbeat_prompt}\n\n"
            "This is a scheduled self-reflection heartbeat. "
            f"Target people selected from short-term recent speakers: {target_text}. "
            "Focus on these target people only. "
            "Use remember_fact for durable memory additions. "
            "Use update_impression to append or merge observations about people; it preserves existing records. "
            "Their existing impressions have already been read before this write phase when available. "
            "Update up to 4 relevant people only when recent chat activity contains enough evidence.\n\n"
            "IMPORTANT remember_fact rules:\n"
            "- scope=core stores stable facts, user preferences, self-improvement rules, and durable knowledge\n"
            "- scope=timeline stores dated events and recent activity\n"
            "- Keep each fact short and evidence-based.\n\n"
            "- If the same event/fact is already present with different wording or source time, return no action\n"
            "- Do not write timeline entries for minor progress updates unless they change the user's durable understanding\n\n"
            "IMPORTANT update_impression rules:\n"
            "- Add only observations grounded in recent chat activity or provided context\n"
            "- Add only stable preferences, repeated behavior, relationships, or facts likely to matter later\n"
            "- If the existing impression already says the same thing with different wording, return no action\n"
            "- Do not invent relationships, private facts, or motives\n"
            "- Use canonical Chinese name (NOT a file path)\n"
            "Supported internal write-phase tools: remember_fact, update_impression. "
            "If no target person has a meaningful new observation, return no actions."
        )

    def _heartbeat_memory_prompt_text(
        self,
        now: float,
        *,
        recent_chat: str,
    ) -> str:
        now_text = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"[system]\n{self.cfg.llm_reply.system_prompt}\n\n"
            f"[identity]\n{self._heartbeat_identity_text()}\n\n"
            f"[current_time]\n{now_text}\n\n"
            f"[short-term recent chat]\n{recent_chat or '无'}\n\n"
            "[heartbeat phase]\n"
            "Phase 1: memory maintenance. The previous read phase has already produced current memory observations, "
            "and short-term recent chat is provided here. Your only job is to append durable memory if the read observations "
            "and recent chat contain a meaningful stable fact, rule, user preference, or dated event. Use remember_fact only. "
            "Before writing, compare against the read memory observations and avoid duplicate notes caused by repeated chat events. "
            "If an existing memory already covers the same event/fact with different wording, return no actions. "
            "Do not write minor repeated progress updates, temporary chatter, or source-time-only changes. "
            "Do not update person impressions in this phase. If nothing is worth remembering, return no actions."
        )[:12000]

    def _heartbeat_person_prompt_text(
        self,
        now: float,
        *,
        person: str,
        recent_chat: str,
        existing_impression: str,
    ) -> str:
        now_text = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"[system]\n{self.cfg.llm_reply.system_prompt}\n\n"
            f"[identity]\n{self._heartbeat_identity_text()}\n\n"
            f"[current_time]\n{now_text}\n\n"
            f"[target person]\n{person}\n\n"
            f"[existing impression for {person}]\n{existing_impression or '无'}\n\n"
            f"[short-term recent chat]\n{recent_chat or '无'}\n\n"
            "[heartbeat phase]\n"
            f"Phase 2: person impression maintenance for exactly one target person: {person}. "
            "The code selected this person from short-term recent speakers. The previous read phase has already produced their existing impression. "
            "Your only job is to call update_impression for this exact person when recent chat contains a new, grounded, "
            "useful observation about them. Do not update any other person. "
            "Do not repeat facts already present in the existing impression, even if the wording or source time differs. "
            "Only write stable preferences, repeated behavior, relationships, or facts likely to matter later. "
            "If there is no meaningful new observation, return no actions."
        )[:12000]

    def _run_heartbeat(self, now: float, rows: list[ChatRowState]) -> bool:
        if not self.cfg.heartbeat_enabled:
            return False
        if not self._heartbeat_llm_backends():
            return False
        if not self.cfg.agent_actions_enabled:
            return False

        row = self._heartbeat_virtual_row()
        if rows:
            for item in rows:
                if self._is_admin_session(item):
                    row = item
                    break

        is_admin = bool(self.cfg.admin_commands_enabled)
        action_limit = int(self.cfg.heartbeat_max_actions)
        if action_limit <= 0:
            return False

        recent_chat = self._heartbeat_recent_chat_activity(max_chats=10, max_lines=80)
        total_actions = 0
        memory_actions = 0
        person_write_actions = 0
        person_read_actions = 0
        processed_people: list[str] = []

        memory_read_trace, memory_read_observations = self._execute_agent_actions(
            row,
            [{"tool": "read_memory", "args": {"name": "all"}, "reason": "记忆流程化读档"}],
            is_admin=is_admin,
            max_actions_override=1,
        )
        total_actions += 1
        if memory_read_trace and self.cfg.log_verbose:
            for ln in memory_read_trace.split("\n"):
                if ln.strip():
                    print(f"        {ln}")
        memory_read_context = (
            "[工具执行结果]\n"
            "[memory read phase]\n"
            f"{memory_read_observations or '记忆读取无内容'}\n\n"
            "[short-term chat read phase]\n"
            f"{recent_chat or '无'}"
        )[:6000]

        memory_write_budget = min(2, max(0, action_limit - total_actions))
        if memory_write_budget > 0:
            _, _, memory_actions, _, _ = self._run_agent_planner_loop(
                planner=self.llm_heartbeat,
                row=row,
                reason="heartbeat",
                is_group=False,
                is_admin=is_admin,
                latest_message="Heartbeat memory phase. Read current memory and short-term chat, then append durable memory if needed.",
                chat_context=recent_chat,
                environment_context=self._heartbeat_memory_prompt_text(
                    now,
                    recent_chat=recent_chat,
                ),
                session_context="",
                workspace_context="",
                memory_recall=memory_read_context,
                tools=["remember_fact"],
                per_round_max_actions=memory_write_budget,
                max_rounds_override=1,
                max_total_actions_override=memory_write_budget,
            )
            total_actions += memory_actions

        people_limit = max(0, int(self.cfg.heartbeat_max_people))
        people_history_records = max(
            1,
            int(self.cfg.heartbeat_people_history_records),
        )
        target_people = self._heartbeat_recent_people(
            max_people=people_limit,
            per_chat_records=people_history_records,
        )
        if target_people and self.cfg.log_verbose:
            print(f"[heartbeat] selected people={', '.join(target_people)}")
        for person in target_people:
            if total_actions >= action_limit:
                break
            read_trace, read_observations = self._execute_agent_actions(
                row,
                [{"tool": "read_impression", "args": {"name": person}, "reason": "人物印象流程化读档"}],
                is_admin=is_admin,
                max_actions_override=1,
            )
            processed_people.append(person)
            person_read_actions += 1
            total_actions += 1
            if read_trace and self.cfg.log_verbose:
                for ln in read_trace.split("\n"):
                    if ln.strip():
                        print(f"        {ln}")
            if total_actions >= action_limit:
                continue

            _, _, update_count, _, _ = self._run_agent_planner_loop(
                planner=self.llm_heartbeat,
                row=row,
                reason="heartbeat",
                is_group=False,
                is_admin=is_admin,
                latest_message=f"Heartbeat person phase. Update impression for {person} only if needed.",
                chat_context=recent_chat,
                environment_context=self._heartbeat_person_prompt_text(
                    now,
                    person=person,
                    recent_chat=recent_chat,
                    existing_impression=self.agent_people.read(person).strip(),
                ),
                session_context="",
                workspace_context="",
                memory_recall=(
                    f"[工具执行结果]\n[read impression for {person}]\n{read_observations}".strip()
                    if read_observations
                    else ""
                ),
                tools=["update_impression"],
                per_round_max_actions=1,
                max_rounds_override=1,
                max_total_actions_override=1,
            )
            person_write_actions += update_count
            total_actions += update_count

        if total_actions <= 0:
            if self.cfg.log_verbose:
                print("[heartbeat] no actions planned")
            return False

        print(
            f"[heartbeat] ran actions={total_actions:>2} "
            f"title={self._fit_col(row.title, 14)} "
            f"memory_writes={memory_actions} person_reads={person_read_actions} "
            f"person_writes={person_write_actions} people={','.join(processed_people)}"
        )
        self._memory_dirty = True
        return True

    def _maybe_run_heartbeat(self, now: float, rows: list[ChatRowState]) -> bool:
        if not self.cfg.heartbeat_enabled:
            return False
        interval = float(self.cfg.heartbeat_interval_sec)
        if interval <= 0:
            return False
        if self._last_heartbeat_at > 0 and now - self._last_heartbeat_at < interval:
            return False
        min_idle = float(self.cfg.heartbeat_min_idle_sec)
        if min_idle > 0 and self._last_activity_at > 0 and now - self._last_activity_at < min_idle:
            return False
        self._mark_heartbeat_at(now)
        return self._run_heartbeat(now, rows)

    def _is_ignored_title(self, row: ChatRowState) -> bool:
        return self._is_ignored_title_text(row.title)

    def _is_ignored_title_text(self, title: str) -> bool:
        title = (title or "").strip()
        if not title:
            return False
        if any(keyword and keyword in title for keyword in self.cfg.ignore_title_keywords):
            return True
        if any(title == kw for kw in self.cfg.ignore_exact_titles if kw):
            return True
        return False

    def _should_reply_group(self, row: ChatRowState, reason: str) -> bool:
        if reason == "mention":
            return True
        preview = row.preview or ""
        if any(keyword and keyword in preview for keyword in self.cfg.group_reply_keywords):
            return True
        if self.cfg.group_only_reply_when_mentioned:
            return False
        return True

    def _should_use_llm_decision(self, is_group: bool) -> bool:
        if not self.llm_decision.is_enabled():
            return False
        if not self.cfg.llm_decision.decision_enabled:
            return False
        if is_group and self.cfg.llm_decision.decision_on_group:
            return True
        if (not is_group) and self.cfg.llm_decision.decision_on_private:
            return True
        return False

    def _llm_should_reply(self, row: ChatRowState, reason: str, is_group: bool) -> bool:
        return self._llm_should_reply_with_context(
            row=row,
            reason=reason,
            is_group=is_group,
            chat_context="",
            environment_context="",
            session_context="",
            workspace_context="",
            memory_recall="",
        )

    def _llm_should_reply_with_context(
        self,
        row: ChatRowState,
        reason: str,
        is_group: bool,
        chat_context: str,
        environment_context: str,
        session_context: str,
        workspace_context: str,
        memory_recall: str,
    ) -> bool:
        if reason == "mention":
            return True
        if not self._should_use_llm_decision(is_group):
            return True
        try:
            should_reply, why = self.llm_decision.should_reply(
                title=row.title,
                preview=row.preview,
                reason=reason,
                is_group=is_group,
                chat_context=chat_context,
                environment_context=environment_context,
                session_context=session_context,
                workspace_context=workspace_context,
                memory_recall=memory_recall,
            )
            reason_w = max(24, self._term_width() - 46)
            print(
                f"[llm] row={row.row_idx:>2} | grp={self._yn(is_group)} | "
                f"decision={('reply' if should_reply else 'skip'):<5} | "
                f"reason={self._fit_col(why, reason_w)}"
            )
            return should_reply
        except Exception as exc:
            if self.cfg.llm.decision_fail_open:
                print(f"[warn] llm decision failed, fail-open: {exc}")
                return True
            print(f"[warn] llm decision failed, skip reply: {exc}")
            return False

    def _collect_focus_candidates(
        self,
        rows: list[ChatRowState],
        event_row: ChatRowState,
        reason: str,
    ) -> list[ChatRowState]:
        candidates: list[ChatRowState] = []
        seen_keys: set[str] = set()

        def _add_candidate(item: ChatRowState) -> None:
            key = self._title_key(item.title) or f"row-{item.row_idx}"
            if key in seen_keys:
                return
            seen_keys.add(key)
            candidates.append(item)

        _add_candidate(event_row)
        for item in rows:
            if self._is_ignored_title(item):
                continue
            prev = self._baseline.get(item.row_idx)
            pending_unread = bool(
                prev is not None
                and (prev.pending_unread or prev.pending_normal)
            )
            is_active = item.has_unread_badge or pending_unread
            if reason == "mention":
                is_active = is_active or item.has_mention or bool(
                    prev is not None and prev.pending_mention
                )
            if is_active:
                _add_candidate(item)
        return candidates

    def _match_focus_candidate(
        self,
        header: str,
        expected_row: ChatRowState,
        candidates: list[ChatRowState] | None,
    ) -> ChatRowState | None:
        clean_header = (header or "").strip()
        if not clean_header or not candidates:
            return None
        expected_key = self._title_key(expected_row.title)
        for item in candidates:
            item_key = self._title_key(item.title)
            if expected_key and item_key and item_key == expected_key:
                continue
            if self._is_chat_header_matched(item.title, clean_header):
                return item
        return None

    def _scan_rows_from_bounds(self, bounds: "WindowBounds") -> list[ChatRowState]:
        shot = screenshot_region(
            bounds.x,
            bounds.y,
            bounds.width,
            bounds.height,
            # Sidebar scanning is high-frequency; keep capture in normal
            # resolution to avoid runaway native memory usage.
            high_res=False,
        )
        shot_rgb = self._to_np_rgb(shot)
        ocr_engine, _ = self._ensure_local_ocr()
        detected = detect_chat_rows(shot_rgb, bounds, self.cfg, ocr_engine)
        return detected.rows

    def _find_row_in_snapshot(
        self,
        rows: list[ChatRowState],
        expected_row: ChatRowState,
    ) -> ChatRowState | None:
        expected_key = self._title_key(expected_row.title)
        if expected_key:
            for item in rows:
                if self._title_key(item.title) == expected_key:
                    return item
        for item in rows:
            if item.row_idx == expected_row.row_idx and self._is_same_snapshot_session(
                expected_row, item
            ):
                return item
        return None

    def _is_same_snapshot_session(
        self,
        expected_row: ChatRowState,
        candidate: ChatRowState,
    ) -> bool:
        expected_key = self._title_key(expected_row.title)
        candidate_key = self._title_key(candidate.title)
        if expected_key and candidate_key and expected_key == candidate_key:
            return True

        expected_preview = self._normalize_preview(expected_row.preview)
        candidate_preview = self._normalize_preview(candidate.preview)
        if expected_preview and candidate_preview:
            if expected_preview in candidate_preview or candidate_preview in expected_preview:
                return True
            ratio = SequenceMatcher(
                a=expected_preview[:80],
                b=candidate_preview[:80],
            ).ratio()
            if ratio >= 0.82:
                return True
        return False

    def _sync_row_snapshot(self, row: ChatRowState, latest: ChatRowState) -> None:
        row.text = latest.text
        row.title = latest.title
        row.preview = latest.preview
        row.has_mention = latest.has_mention
        row.has_unread_badge = latest.has_unread_badge
        row.fingerprint = latest.fingerprint
        row.click_x_ratio = latest.click_x_ratio
        row.click_y_ratio = latest.click_y_ratio

    def _click_until_unread_cleared(
        self,
        row: ChatRowState,
        *,
        bounds: "WindowBounds",
    ) -> "WindowBounds":
        if not row.has_unread_badge:
            return bounds

        latest = bounds
        attempts = 0
        max_attempts = 40
        while attempts < max_attempts:
            time.sleep(0.5)
            try:
                latest = get_front_window_bounds(self.cfg.app_name)
                rows = self._scan_rows_from_bounds(latest)
            except Exception as exc:
                if self.cfg.log_verbose or self.cfg.debug_scan:
                    print(f"[badge-check] row={row.row_idx:>2} | scan failed: {exc}")
                return latest

            matched = self._find_row_in_snapshot(rows, row)
            if matched is None:
                if self.cfg.log_verbose or self.cfg.debug_scan:
                    print(
                        f"[badge-check] row={row.row_idx:>2} | "
                        f"title={self._fit_col(row.title, 14)} not found"
                    )
                return latest

            self._sync_row_snapshot(row, matched)

            if not matched.has_unread_badge:
                if self.cfg.log_verbose or self.cfg.debug_scan:
                    print(
                        f"[badge-cleared] row={row.row_idx:>2} | "
                        f"title={self._fit_col(matched.title, 14)} checks={attempts + 1}"
                    )
                return latest

            attempts += 1
            click_x = latest.x + int(latest.width * matched.click_x_ratio)
            click_y = latest.y + int(latest.height * matched.click_y_ratio)
            if self.cfg.log_verbose or self.cfg.debug_scan:
                print(
                    f"[badge-reclick] row={row.row_idx:>2} | "
                    f"title={self._fit_col(matched.title, 14)} attempt={attempts}/{max_attempts}"
                )
            self._safe_click(click_x, click_y)

        if self.cfg.log_verbose or self.cfg.debug_scan:
            print(
                f"[badge-still-unread] row={row.row_idx:>2} | "
                f"title={self._fit_col(row.title, 14)} attempts={max_attempts}"
            )
        return latest

    def _focus_chat(
        self,
        row: ChatRowState,
        *,
        focus_candidates: list[ChatRowState] | None = None,
        ensure_unread_clear: bool = False,
    ) -> FocusResult:
        self._activate_wechat()
        time.sleep(self.cfg.activate_wait_sec)
        bounds = get_front_window_bounds(self.cfg.app_name)
        is_group = self._is_group_chat(row)
        tries = max(1, self.cfg.focus_verify_max_clicks)
        latest = bounds
        last_header = ""

        if not self.cfg.focus_verify_enabled:
            row_x = latest.x + int(latest.width * row.click_x_ratio)
            row_y = latest.y + int(latest.height * row.click_y_ratio)
            self._safe_click(row_x, row_y)
            if ensure_unread_clear:
                latest = self._click_until_unread_cleared(row, bounds=latest)
            time.sleep(self.cfg.post_select_wait_sec)
            latest = get_front_window_bounds(self.cfg.app_name)
            return FocusResult(bounds=latest, matched=True, resolved_row=row, seen_header="")

        for i in range(1, tries + 1):
            latest = get_front_window_bounds(self.cfg.app_name)

            # First verify current focus without clicking.
            header = self._extract_chat_header_text(latest, is_group=is_group)
            last_header = header
            if self._is_chat_header_matched(row.title, header):
                if self.cfg.log_verbose:
                    seen_w = max(20, self._term_width() - 22)
                    print(
                        f"[focus-ok] row={row.row_idx:>2} | try={i:>2}/{tries:<2} | "
                        f"expect={self._fit_col(row.title, 14)}"
                    )
                    print(f"           seen={self._fit_col(header, seen_w)}")
                if ensure_unread_clear:
                    latest = self._click_until_unread_cleared(row, bounds=latest)
                    time.sleep(self.cfg.post_select_wait_sec)
                return FocusResult(bounds=latest, matched=True, resolved_row=row, seen_header=header)

            swapped = self._match_focus_candidate(header, row, focus_candidates)
            if swapped is not None:
                if self.cfg.log_verbose:
                    seen_w = max(20, self._term_width() - 22)
                    print(
                        f"[focus-swap] row={row.row_idx:>2} | try={i:>2}/{tries:<2} | "
                        f"expect={self._fit_col(row.title, 14)} -> "
                        f"use={self._fit_col(swapped.title, 14)}"
                    )
                    print(f"            seen={self._fit_col(header, seen_w)}")
                if ensure_unread_clear:
                    latest = self._click_until_unread_cleared(swapped, bounds=latest)
                    time.sleep(self.cfg.post_select_wait_sec)
                return FocusResult(
                    bounds=latest,
                    matched=True,
                    resolved_row=swapped,
                    seen_header=header,
                )

            # Not matched: click row once, then verify again.
            row_x = latest.x + int(latest.width * row.click_x_ratio)
            row_y = latest.y + int(latest.height * row.click_y_ratio)
            self._safe_click(row_x, row_y)
            if ensure_unread_clear:
                latest = self._click_until_unread_cleared(row, bounds=latest)
            time.sleep(self.cfg.post_select_wait_sec)
            latest = get_front_window_bounds(self.cfg.app_name)
            header = self._extract_chat_header_text(latest, is_group=is_group)
            last_header = header
            if self._is_chat_header_matched(row.title, header):
                if self.cfg.log_verbose:
                    seen_w = max(20, self._term_width() - 22)
                    print(
                        f"[focus-ok] row={row.row_idx:>2} | try={i:>2}/{tries:<2} | "
                        f"expect={self._fit_col(row.title, 14)}"
                    )
                    print(f"           seen={self._fit_col(header, seen_w)}")
                return FocusResult(bounds=latest, matched=True, resolved_row=row, seen_header=header)

            swapped = self._match_focus_candidate(header, row, focus_candidates)
            if swapped is not None:
                if self.cfg.log_verbose:
                    seen_w = max(20, self._term_width() - 22)
                    print(
                        f"[focus-swap] row={row.row_idx:>2} | try={i:>2}/{tries:<2} | "
                        f"expect={self._fit_col(row.title, 14)} -> "
                        f"use={self._fit_col(swapped.title, 14)}"
                    )
                    print(f"            seen={self._fit_col(header, seen_w)}")
                if ensure_unread_clear:
                    latest = self._click_until_unread_cleared(swapped, bounds=latest)
                    time.sleep(self.cfg.post_select_wait_sec)
                return FocusResult(
                    bounds=latest,
                    matched=True,
                    resolved_row=swapped,
                    seen_header=header,
                )

            if self.cfg.log_verbose:
                seen_w = max(20, self._term_width() - 22)
                print(
                    f"[focus-retry] row={row.row_idx:>2} | try={i:>2}/{tries:<2} | "
                    f"expect={self._fit_col(row.title, 14)}"
                )
                print(f"              seen={self._fit_col(header, seen_w)}")
            time.sleep(max(0.02, self.cfg.focus_verify_wait_sec))
        if self.cfg.log_verbose:
            print(f"[focus-fail] row={row.row_idx:>2} | expect={self._fit_col(row.title, 14)}")
        return FocusResult(bounds=latest, matched=False, resolved_row=None, seen_header=last_header)

    def _extract_chat_context(
        self,
        bounds: "WindowBounds",
        *,
        title: str = "",
        reason: str = "",
        is_group: bool = False,
        session_context: str = "",
        session_history: str = "",
        workspace_context: str = "",
        memory_recall: str = "",
        latest_hint: str = "",
        preview: str = "",
    ) -> ChatContextSnapshot:
        x = bounds.x + int(bounds.width * self.cfg.chat_context_region.x)
        y = bounds.y + int(bounds.height * self.cfg.chat_context_region.y)
        w = int(bounds.width * self.cfg.chat_context_region.w)
        h = int(bounds.height * self.cfg.chat_context_region.h)
        if w <= 0 or h <= 0:
            return ChatContextSnapshot(text="", last_side="unknown", last_line="", source="none")

        # Vision context capture may run frequently in busy chats; keep normal
        # resolution to avoid runaway memory growth on macOS capture pipeline.
        shot = screenshot_region(x, y, w, h, high_res=False)
        try:
            parsed = self.llm.analyze_chat_image(
                image=shot,
                title=title,
                reason=reason,
                is_group=is_group,
                session_context=session_context,
                session_history=session_history,
                workspace_context=workspace_context,
                memory_recall=memory_recall,
                latest_hint=latest_hint,
                preview=preview,
            )
            context = parsed.get("context") if isinstance(parsed.get("context"), dict) else {}
            environment = (
                parsed.get("environment")
                if isinstance(parsed.get("environment"), dict)
                else (
                    parsed.get("memory_update")
                    if isinstance(parsed.get("memory_update"), dict)
                    else {}
                )
            )

            last_speaker = context.get("last_speaker", parsed.get("last_speaker", "unknown"))
            if last_speaker not in ("self", "other", "unknown"):
                last_speaker = "unknown"
            raw_last = context.get("last_message") or parsed.get("last_message") or ""
            if isinstance(raw_last, dict):
                last_message = str(raw_last.get("text", "")).strip()[:220]
            else:
                last_message = str(raw_last).strip()[:220]
            last_user_message = str(
                context.get("last_user_message", parsed.get("last_user_message", ""))
            ).strip()[:220]
            recent_messages = context.get("recent_messages") or parsed.get("recent_messages") or []
            if not isinstance(recent_messages, list):
                recent_messages = []
            recent_messages = [str(x).strip() for x in recent_messages if str(x).strip()][:20]

            text = " | ".join(recent_messages)[:900]
            if not text:
                text = last_message[:900]
            if self.cfg.log_verbose:
                last_w = max(24, self._term_width() - 15)
                print(
                    f"[vision] speaker={last_speaker:<7} | recent={len(recent_messages):>2}"
                )
                print(f"         last={self._fit_col(last_message, last_w)}")
            snapshot = ChatContextSnapshot(
                text=text,
                last_side=last_speaker,
                last_line=last_message,
                last_user_message=last_user_message,
                recent_messages=recent_messages,
                recent_structured=(
                    context.get("recent_structured")
                    if isinstance(context.get("recent_structured"), list)
                    else (parsed.get("recent_structured") or [])
                ),
                chat_records=(
                    context.get("chat_records")
                    if isinstance(context.get("chat_records"), list)
                    else (parsed.get("chat_records") or parsed.get("recent_structured") or [])
                ),
                memory_summary=str(environment.get("summary", "")).strip(),
                memory_time_hints=(
                    environment.get("time_hints")
                    if isinstance(environment.get("time_hints"), list)
                    else []
                ),
                memory_people=(
                    environment.get("people")
                    if isinstance(environment.get("people"), list)
                    else []
                ),
                memory_facts=(
                    environment.get("facts")
                    if isinstance(environment.get("facts"), list)
                    else []
                ),
                memory_events=(
                    environment.get("events")
                    if isinstance(environment.get("events"), list)
                    else []
                ),
                memory_relations=(
                    environment.get("relations")
                    if isinstance(environment.get("relations"), list)
                    else []
                ),
                schema=str(parsed.get("schema", "")),
                source="vision",
            )
            snapshot.environment_text = self._build_environment_context(snapshot)
            return snapshot
        except Exception as exc:
            if self.cfg.vision.fail_open:
                print(f"[warn] vision parse failed: {exc}")
                return ChatContextSnapshot(text="", last_side="unknown", last_line="", source="vision")
            raise
        finally:
            try:
                shot.close()
            except Exception:
                pass

    def _log_cycle_snapshot(self, rows: list[ChatRowState], now: float) -> None:
        if not self.cfg.log_verbose:
            return
        print("")
        print(
            f"[cycle] id={self._cycle:>4} | ts={int(now):>10} | "
            f"rows={len(rows):>2} | baseline={len(self._baseline):>2}"
        )
        limit = max(1, self.cfg.log_snapshot_rows)
        for row in rows[:limit]:
            group = self._is_group_chat(row)
            key = self._session_key_for_row(row, remember=False)
            preview_w = max(16, self._term_width() - 76)
            print(
                f"[row]   idx={row.row_idx:>2} | grp={self._yn(group)} | "
                f"unrd={self._yn(row.has_unread_badge)} | @={self._yn(row.has_mention)} | "
                f"key={self._fit_col(key, 10)} | title={self._fit_col(row.title, 14)} | "
                f"preview={self._fit_col(row.preview, preview_w)}"
            )

    def _set_baseline(self, rows: list[ChatRowState], now: float) -> None:
        self._baseline = {
            row.row_idx: RowMemory(
                session_key=self._session_key_for_row(row, remember=False),
                fingerprint=row.fingerprint,
                preview_norm=self._normalize_preview(row.preview),
                last_sent_norm="",
                has_unread_badge=(
                    False if self.cfg.process_existing_unread_on_start else row.has_unread_badge
                ),
                pending_unread=(row.has_unread_badge if self.cfg.process_existing_unread_on_start else False),
                pending_normal=False,
                has_mention=(False if self.cfg.process_existing_unread_on_start else row.has_mention),
                pending_mention=(row.has_mention if self.cfg.process_existing_unread_on_start else False),
                last_replied_at=0.0,
            )
            for row in rows
        }
        print(f"[init] baseline rows={len(rows):>2} at={now:.0f}")

    def _pick_event(self, rows: list[ChatRowState], now: float) -> tuple[ChatRowState, str] | None:
        mention_candidates: list[ChatRowState] = []
        unread_candidates: list[ChatRowState] = []
        preview_candidates: list[ChatRowState] = []

        for row in rows:
            if self._is_ignored_title(row):
                if self.cfg.log_verbose or self.cfg.debug_scan:
                    print(f"[skip-title] row={row.row_idx} title={row.title!r}")
                continue

            prev = self._baseline.get(row.row_idx)
            if prev is None:
                self._baseline[row.row_idx] = RowMemory(
                    session_key=self._session_key_for_row(row, remember=False),
                    fingerprint=row.fingerprint,
                    preview_norm=self._normalize_preview(row.preview),
                    last_sent_norm="",
                    has_unread_badge=row.has_unread_badge,
                    pending_unread=row.has_unread_badge,
                    pending_normal=False,
                    has_mention=row.has_mention,
                    pending_mention=row.has_mention,
                    last_replied_at=0.0,
                )
                continue

            row_key = self._session_key_for_row(row, remember=False)
            if prev.session_key != row_key:
                # Row index got rebound to another session after list reordering.
                # Reset row memory to avoid carrying unread/pending state across chats.
                self._baseline[row.row_idx] = RowMemory(
                    session_key=row_key,
                    fingerprint=row.fingerprint,
                    preview_norm=self._normalize_preview(row.preview),
                    last_sent_norm="",
                    has_unread_badge=row.has_unread_badge,
                    pending_unread=False,
                    pending_normal=False,
                    has_mention=row.has_mention,
                    pending_mention=row.has_mention,
                    last_replied_at=prev.last_replied_at,
                )
                if self.cfg.log_verbose or self.cfg.debug_scan:
                    print(
                        f"[row-rebind] row={row.row_idx:>2} | "
                        f"key={self._fit_col(row_key, 14)}"
                    )
                continue

            preview_norm = self._normalize_preview(row.preview)
            preview_changed = self._is_preview_meaningfully_changed(
                prev.preview_norm, preview_norm
            )
            unread_rise = row.has_unread_badge and not prev.has_unread_badge
            unread_pending = prev.pending_unread
            normal_pending = prev.pending_normal
            mention_rise = row.has_mention and not prev.has_mention
            mention_pending = prev.pending_mention

            prev.fingerprint = row.fingerprint
            prev.session_key = row_key
            prev.preview_norm = preview_norm
            prev.has_unread_badge = row.has_unread_badge
            if unread_rise:
                prev.pending_unread = True
            if not row.has_unread_badge:
                prev.pending_unread = False
            prev.has_mention = row.has_mention
            if mention_rise:
                prev.pending_mention = True
            elif not row.has_mention:
                prev.pending_mention = False

            recent_sent_norm = self._get_recent_sent_for_row(row, now)
            self_echo = self._is_self_echo(preview_norm, prev.last_sent_norm) or self._is_self_echo(
                preview_norm, recent_sent_norm
            )
            recent_self_preview = self._is_preview_refresh_from_self(
                row=row,
                preview_norm=preview_norm,
                prev_sent_norm=prev.last_sent_norm,
                recent_sent_norm=recent_sent_norm,
            )
            self_preview_refresh = preview_changed and recent_self_preview

            if self.cfg.debug_scan and (preview_changed or unread_rise or mention_rise):
                print(
                    "[scan] "
                    f"row={row.row_idx} title={row.title!r} preview={row.preview!r} "
                    f"preview_changed={preview_changed} "
                    f"unread={row.has_unread_badge} mention={row.has_mention} "
                    f"self_echo={self_echo} pending_unread={unread_pending or unread_rise} "
                    f"pending_mention={mention_pending or mention_rise}"
                )

            unread_active = row.has_unread_badge

            if not (
                preview_changed
                or unread_rise
                or unread_pending
                or normal_pending
                or mention_rise
                or mention_pending
                or unread_active
            ):
                continue

            # Prevent reply loops when the preview is our own last sent text.
            if self_echo:
                continue
            if row.has_unread_badge and recent_self_preview:
                if self.cfg.log_verbose or self.cfg.debug_scan:
                    print(
                        f"[skip-self-unread] row={row.row_idx:>2} | "
                        f"title={self._fit_col(row.title, 14)}"
                    )
                    print(
                        f"                  preview={self._fit_col(row.preview, max(24, self._term_width() - 27))}"
                    )
                continue
            if self_preview_refresh:
                if self.cfg.log_verbose or self.cfg.debug_scan:
                    print(
                        f"[skip-self-preview] row={row.row_idx:>2} | "
                        f"title={self._fit_col(row.title, 14)}"
                    )
                    print(
                        f"                  preview={self._fit_col(row.preview, max(24, self._term_width() - 27))}"
                    )
                continue

            if now - prev.last_replied_at < self.cfg.action_cooldown_sec:
                if self.cfg.log_verbose or self.cfg.debug_scan:
                    remain = self.cfg.action_cooldown_sec - (now - prev.last_replied_at)
                    print(
                        f"[skip-cooldown] row={row.row_idx:>2} title={row.title!r} "
                        f"remain={max(0.0, remain):.1f}s"
                    )
                continue

            if row.has_mention and (mention_rise or mention_pending):
                mention_candidates.append(row)
            elif row.has_unread_badge or unread_pending or normal_pending:
                unread_candidates.append(row)
            elif self.cfg.trigger_on_preview_change and preview_changed:
                preview_candidates.append(row)
            else:
                continue

        def _choose(candidates: list[ChatRowState], reason: str) -> tuple[ChatRowState, str] | None:
            for row in candidates:
                is_group = self._is_group_chat(row)
                is_admin = self._is_admin_session(row)
                if (not is_admin) and self._is_row_muted(row):
                    if self.cfg.log_verbose or self.cfg.debug_scan:
                        print(
                            f"[skip-muted] row={row.row_idx} title={row.title!r} "
                            f"reason={reason}"
                        )
                    continue
                if (
                    is_group
                    and reason == "new_message"
                    and self.cfg.group_require_sender_prefix_for_new_message
                    and (not self._has_sender_prefix(row.preview))
                ):
                    if self.cfg.log_verbose or self.cfg.debug_scan:
                        print(
                            f"[skip-group-prefix] row={row.row_idx} title={row.title!r} "
                            f"preview={row.preview!r}"
                        )
                    continue
                if is_group and self._should_use_llm_decision(is_group):
                    if (
                        self._is_normal_reply_event(row, reason)
                        and self._normal_reply_interval_active()
                        and (now - self._last_normal_reply_at) < self.cfg.normal_reply_interval_sec
                    ):
                        remain = self.cfg.normal_reply_interval_sec - (
                            now - self._last_normal_reply_at
                        )
                        mem = self._baseline.get(row.row_idx)
                        if mem is not None and not row.has_unread_badge:
                            mem.pending_normal = True
                        if self.cfg.log_verbose or self.cfg.debug_scan:
                            print(
                                f"[skip-normal-interval] row={row.row_idx:>2} "
                                f"title={row.title!r} remain={max(0.0, remain):.1f}s"
                            )
                        continue
                    return row, reason
                if is_group and not self._should_reply_group(row, reason):
                    if self.cfg.log_verbose or self.cfg.debug_scan:
                        print(
                            f"[skip-rule] group row={row.row_idx} title={row.title!r} "
                            f"reason={reason} preview={row.preview!r} "
                            f"group_only_reply_when_mentioned={self.cfg.group_only_reply_when_mentioned}"
                        )
                    continue
                if (
                    self._is_normal_reply_event(row, reason)
                    and self._normal_reply_interval_active()
                    and (now - self._last_normal_reply_at) < self.cfg.normal_reply_interval_sec
                ):
                    remain = self.cfg.normal_reply_interval_sec - (
                        now - self._last_normal_reply_at
                    )
                    mem = self._baseline.get(row.row_idx)
                    if mem is not None and not row.has_unread_badge:
                        mem.pending_normal = True
                    if self.cfg.log_verbose or self.cfg.debug_scan:
                        print(
                            f"[skip-normal-interval] row={row.row_idx:>2} "
                            f"title={row.title!r} remain={max(0.0, remain):.1f}s"
                        )
                    continue
                return row, reason
            return None

        chosen = _choose(mention_candidates, "mention")
        if chosen:
            return chosen
        chosen = _choose(unread_candidates, "new_message")
        if chosen:
            return chosen
        return _choose(preview_candidates, "new_message")

    def _recent_assistant_replies(self, row: ChatRowState, limit: int) -> list[str]:
        key = self._session_key_for_row(row)
        sess = self._get_or_create_session(key)
        out: list[str] = []
        for item in reversed(sess.short):
            if not item.startswith("A:"):
                continue
            txt = item.split(":", 1)[1].strip() if ":" in item else ""
            if not txt:
                continue
            out.append(txt)
            if len(out) >= max(1, limit):
                break
        return out

    def _is_reply_too_similar(self, reply: str, recent_replies: list[str]) -> bool:
        norm = self._normalize_preview(reply)
        if not norm or len(norm) < 4:
            return False
        threshold = max(0.6, min(0.98, self.cfg.llm_reply.anti_repeat_similarity))
        for old in recent_replies:
            old_norm = self._normalize_preview(old)
            if not old_norm:
                continue
            if norm in old_norm or old_norm in norm:
                return True
            n = min(len(norm), len(old_norm), 40)
            if n < 6:
                continue
            ratio = SequenceMatcher(a=norm[:n], b=old_norm[:n]).ratio()
            if ratio >= threshold:
                return True
        return False

    def _is_no_reply_signal(self, text: str) -> bool:
        raw = re.sub(r"\s+", " ", (text or "").strip())
        if not raw:
            return False
        upper = raw.upper()
        if re.fullmatch(r"\s*<?\[?\s*NO[\s_\-]?REPLY\s*\]?>?\s*", upper):
            return True
        lowered = raw.lower().strip(" 。.!！?？")
        exact_markers = {
            "无需回复",
            "不需要回复",
            "不回复",
            "保持沉默",
            "继续观察",
            "skip reply",
            "no reply",
            "stay silent",
            "heartbeat_ok",
        }
        return lowered in exact_markers

    def _reply_text(
        self,
        row: ChatRowState,
        reason: str,
        chat_context: str,
        environment_context: str,
        session_context: str,
        workspace_context: str,
        memory_recall: str,
        latest_message: str = "",
    ) -> str:
        fallback = (
            self.cfg.reply_on_mention if reason == "mention" else self.cfg.reply_on_new_message
        )
        if not self.llm_reply.is_reply_enabled():
            return fallback
        recent_replies = self._recent_assistant_replies(
                    row, max(1, self.cfg.llm_reply.anti_repeat_window)
        )
        retries = (
            max(0, self.cfg.llm_reply.anti_repeat_retry)
            if self.cfg.llm_reply.anti_repeat_enabled
            else 0
        )
        try:
            avoid = list(recent_replies)
            for attempt in range(retries + 1):
                text = self.llm_reply.generate(
                    title=row.title,
                    preview=row.preview,
                    reason=reason,
                    latest_message=latest_message,
                    chat_context=chat_context,
                    environment_context=environment_context,
                    session_context=session_context,
                    workspace_context=workspace_context,
                    memory_recall=memory_recall,
                    avoid_replies=(avoid if attempt > 0 else []),
                    allow_no_reply_signal=(
                        self.cfg.group_allow_llm_no_reply if self._is_group_chat(row) else False
                    ),
                )
                clean = self._sanitize_generated_reply(text, fallback=fallback)
                if self._is_payment_gate_reply(text):
                    if self.cfg.log_verbose:
                        print(
                            f"[reply-filter] payment-gate detected row={row.row_idx:>2} "
                            f"title={self._fit_col(row.title, 14)} attempt={attempt + 1}/{retries + 1}"
                        )
                    avoid.append(re.sub(r"\s+", " ", text).strip()[:90])
                    if attempt < retries:
                        continue
                    clean = self._contextual_fallback_reply(
                        reason=reason,
                        latest_message=latest_message or row.preview or row.text or "",
                        fallback=fallback,
                    )
                if self._is_no_reply_signal(clean):
                    if self._is_group_chat(row) and self.cfg.group_allow_llm_no_reply:
                        if self.cfg.log_verbose:
                            print(
                                f"[skip-no-reply] row={row.row_idx:>2} | "
                                f"title={self._fit_col(row.title, 14)}"
                            )
                        return ""
                    # If disabled (or in private), avoid accidental silence from model meta output.
                    if self.cfg.log_verbose:
                        print(
                            f"[warn] no-reply signal ignored, fallback used "
                            f"title={row.title!r}"
                        )
                    clean = fallback
                if self.cfg.llm_reply.anti_repeat_enabled and self._is_reply_too_similar(clean, recent_replies):
                    print(
                        f"[reply-repeat] row={row.row_idx} title={row.title!r} "
                        f"attempt={attempt + 1}/{retries + 1} reply={clean!r}"
                    )
                    avoid.append(clean)
                    if attempt < retries:
                        continue
                print(
                    f"[{self.llm_reply.reply_backend_name()}] generated reply len={len(text)} "
                    f"sanitized_len={len(clean)} "
                    f"attempt={attempt + 1}/{retries + 1}"
                )
                return clean
            return fallback
        except Exception as exc:
            print(f"[warn] reply backend failed, fallback to template: {exc}")
            return fallback

    def _sanitize_generated_reply(self, text: str, fallback: str) -> str:
        raw = (text or "").strip()
        if not raw:
            return fallback
        # Remove markdown fences and collapse lines.
        raw = raw.replace("```", "").replace("\r", "\n")
        lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
        merged = " ".join(lines).strip()
        if self._is_no_reply_signal(raw) or self._is_no_reply_signal(merged):
            return "[NO_REPLY]"

        def _clip_reply(s: str) -> str:
            val = re.sub(r"\s+", " ", (s or "").strip())
            return val

        def _looks_like_heading(s: str) -> bool:
            x = (s or "").strip().lower()
            return bool(
                re.match(r"^(\d+[\.)]|[-*])\s*", x)
                or x.startswith("thinking")
                or x.startswith("analysis")
                or x.startswith("reasoning")
                or x.startswith("步骤")
                or x.startswith("思考")
                or x.startswith("推理")
            )

        suspicious_markers = [
            "thinking process",
            "analyze the request",
            "analysis:",
            "reasoning:",
            "let me think",
            "chain of thought",
            "步骤",
            "思考过程",
            "推理过程",
            "分析请求",
        ]
        lower = merged.lower()
        has_reasoning = any(marker in lower for marker in suspicious_markers)

        # Try to extract explicit final answer block first.
        final_markers = [
            "final answer",
            "final response",
            "final reply",
            "最终回答",
            "最终回复",
            "回复如下",
            "答复如下",
        ]
        candidate = ""
        for mk in final_markers:
            idx = lower.rfind(mk)
            if idx >= 0:
                tail = merged[idx + len(mk) :].lstrip("：: -")
                candidate = _clip_reply(tail)
                if candidate:
                    break

        # If no explicit final marker, use the last non-heading line as fallback extraction.
        if not candidate and has_reasoning:
            for line in reversed(lines):
                if _looks_like_heading(line):
                    continue
                if len(line) < 2:
                    continue
                if re.search(r"[\u4e00-\u9fff]", line):
                    candidate = _clip_reply(line)
                    if candidate:
                        break

        # Avoid overly long multi-paragraph output for chat reply.
        if candidate:
            result = candidate
        else:
            result = "\n".join(lines).strip()
            if not result:
                return fallback

        if self._is_no_reply_signal(result):
            return "[NO_REPLY]"
        if self._is_payment_gate_reply(result):
            return fallback

        # If no CJK and no punctuation, treat as low quality.
        if not re.search(r"[\u4e00-\u9fff]", result):
            return fallback
        return result

    def _activate_wechat(self) -> None:
        self.sender.activate()

    def _safe_click(self, x: int, y: int) -> None:
        self.sender.safe_click(x, y)

    def _perform_scroll(self, amount: int) -> None:
        total = int(amount)
        if total == 0:
            return
        direction = 1 if total > 0 else -1
        remaining = abs(total)
        while remaining > 0:
            step = min(120, remaining)
            pyautogui.scroll(direction * step)
            remaining -= step
            time.sleep(0.015)

    def _send_delay_sec(self) -> float:
        return self.sender.send_delay_sec()

    @staticmethod
    def _apple_quote(raw: str) -> str:
        return WeChatGuiSender.apple_quote(raw)

    def _paste_and_send(self, message: str) -> None:
        self.sender.paste_and_send(message)

    def _paste_file_and_send(self, file_path: Path) -> bool:
        return self.sender.paste_file_and_send(file_path)

    def _send_generated_file(
        self,
        row: ChatRowState,
        file_path: Path,
        *,
        focused_bounds=None,
    ) -> bool:
        if self.cfg.dry_run:
            print(f"[dry-run] file={str(file_path)}")
            return True

        if self.cfg.receiver_mode == "detached_windows":
            ok = self.sender.paste_file_and_send_to_window(row.title, file_path)
            if not ok:
                print(f"[warn] detached file send not confirmed: {row.title!r}")
                return False
            file_w = max(24, self._term_width() - 13)
            print(f"[sent] to={self._fit_col(row.title, 14)} file={self._fit_col(str(file_path), file_w)}")
            return True

        if focused_bounds is not None:
            bounds = focused_bounds
        else:
            focus_result = self._focus_chat(
                row,
                ensure_unread_clear=False,
            )
            if (not focus_result.matched) or (focus_result.resolved_row is None):
                seen_w = max(24, self._term_width() - 17)
                print(
                    f"[skip-focus] row={row.row_idx:>2} | "
                    f"expect={self._fit_col(row.title, 14)}"
                )
                if focus_result.seen_header:
                    print(f"            seen={self._fit_col(focus_result.seen_header, seen_w)}")
                return False
            bounds = focus_result.bounds

        if self.cfg.focus_verify_enabled:
            seen = self._extract_chat_header_text(
                bounds, is_group=self._is_group_chat(row)
            )
            if not self._is_chat_header_matched(row.title, seen):
                seen_w = max(24, self._term_width() - 17)
                print(
                    f"[skip-focus] row={row.row_idx:>2} | "
                    f"expect={self._fit_col(row.title, 14)}"
                )
                print(f"            seen={self._fit_col(seen, seen_w)}")
                return False

        input_x = bounds.x + int(bounds.width * self.cfg.input_point.x)
        input_y = bounds.y + int(bounds.height * self.cfg.input_point.y)
        self._safe_click(input_x, input_y)
        time.sleep(0.08)

        ok = self._paste_file_and_send(file_path)
        if not ok:
            return False
        file_w = max(24, self._term_width() - 13)
        print(f"[sent] file={self._fit_col(str(file_path), file_w)}")
        return True

    def _remember_latest_image_for_row(self, row: ChatRowState, image_path: str) -> None:
        with self._state_lock:
            key = self._session_key_for_row(row)
            self._latest_image_by_session[key] = image_path

    def _latest_image_for_row(self, row: ChatRowState) -> str:
        return self._latest_image_by_session.get(self._session_key_for_row(row), "")

    def _image_followup_context_for_text(self, row: ChatRowState, text: str) -> str:
        clean = re.sub(r"\s+", " ", str(text or "")).strip()
        if not clean:
            return ""
        image_path = self._latest_image_for_row(row)
        if not image_path:
            return ""
        triggers = (
            "图",
            "图片",
            "照片",
            "这张",
            "这个图",
            "这图",
            "看图",
            "识别",
            "是什么",
            "改图",
            "修图",
            "编辑",
            "换成",
            "去掉",
            "加上",
        )
        if not any(token in clean for token in triggers):
            return ""
        cached = self._image_vision_by_path.get(image_path, "")
        if not cached:
            try:
                cached = self._describe_image_file(
                    image_path,
                    prompt=(
                        "这是当前微信会话最近收到的图片。请用中文描述图片内容；"
                        "如果有文字，请转写；如果适合识别物体/植物/截图内容，也请直接说明。"
                    ),
                )
            except Exception as exc:
                cached = f"[图片识别失败: {exc}]"
            cached = re.sub(r"\s+", " ", cached).strip()[:800]
            if cached:
                self._image_vision_by_path[image_path] = cached
        return f"[最近收到的图片]\npath={image_path}\nvision={cached[:800]}" if cached else ""

    def _reply(
        self,
        row: ChatRowState,
        reason: str,
        focused_bounds=None,
        chat_context: str = "",
        environment_context: str = "",
        session_context: str = "",
        workspace_context: str = "",
        memory_recall: str = "",
        latest_message: str = "",
        force_message: str = "",
        *,
        mention_reply_to: str = "",
    ) -> str:
        preview_w = max(24, self._term_width() - 18)
        print(
            f"[action] row={row.row_idx:>2} | reason={reason:<14} | "
            f"title={self._fit_col(row.title, 14)}"
        )
        print(f"         preview={self._fit_col(row.preview, preview_w)}")
        message = force_message or self._reply_text(
            row,
            reason,
            chat_context,
            environment_context,
            session_context,
            workspace_context,
            memory_recall,
            latest_message=latest_message,
        )
        if message:
            message = self._strip_markdown_formatting(message)
            if reason == "mention":
                message = self._format_long_reply(message)
        if not (message or "").strip():
            if self.cfg.log_verbose:
                print(
                    f"[skip-empty-reply] row={row.row_idx:>2} | "
                    f"title={self._fit_col(row.title, 14)}"
                )
            return ""
        if self.cfg.dry_run:
            msg_w = max(24, self._term_width() - 16)
            print(f"[dry-run] msg={self._fit_col(message, msg_w)}")
            return message

        if self.cfg.receiver_mode == "detached_windows":
            mention_name = self._resolve_mention_name(mention_reply_to)
            send_text = f"@{mention_name or mention_reply_to} {message}" if mention_reply_to else message
            with self._send_lock:
                self._pause_visual_scan(self.cfg.mention_pause_scan_sec)
                if mention_name:
                    print(
                        f"[mention-send] sender={mention_reply_to!r} "
                        f"mention={mention_name!r} title={row.title!r}"
                    )
                    raised = self.sender.mention_and_send_to_window(
                        row.title,
                        mention_name,
                        message,
                    )
                else:
                    if mention_reply_to and self.cfg.mention_send_enabled:
                        print(
                            f"[mention-send] no mapping for sender={mention_reply_to!r}; "
                            "fallback=text"
                        )
                    raised = self.sender.paste_and_send_to_window(row.title, send_text)
                self._pause_visual_scan(self.cfg.mention_pause_scan_sec)
            if not raised:
                print(f"[warn] detached send window raise not confirmed: {row.title!r}")
                return ""
            msg_w = max(24, self._term_width() - 12)
            print(f"[sent] to={self._fit_col(row.title, 14)} msg={self._fit_col(send_text, msg_w)}")
            return message

        if focused_bounds is not None:
            bounds = focused_bounds
        else:
            focus_result = self._focus_chat(
                row,
                ensure_unread_clear=(reason == "new_message"),
            )
            if (not focus_result.matched) or (focus_result.resolved_row is None):
                seen_w = max(24, self._term_width() - 17)
                print(
                    f"[skip-focus] row={row.row_idx:>2} | "
                    f"expect={self._fit_col(row.title, 14)}"
                )
                if focus_result.seen_header:
                    print(f"            seen={self._fit_col(focus_result.seen_header, seen_w)}")
                return ""
            bounds = focus_result.bounds
        if self.cfg.focus_verify_enabled:
            seen = self._extract_chat_header_text(
                bounds, is_group=self._is_group_chat(row)
            )
            if not self._is_chat_header_matched(row.title, seen):
                seen_w = max(24, self._term_width() - 17)
                print(
                    f"[skip-focus] row={row.row_idx:>2} | "
                    f"expect={self._fit_col(row.title, 14)}"
                )
                print(f"            seen={self._fit_col(seen, seen_w)}")
                return ""

        input_x = bounds.x + int(bounds.width * self.cfg.input_point.x)
        input_y = bounds.y + int(bounds.height * self.cfg.input_point.y)
        self._safe_click(input_x, input_y)
        time.sleep(0.08)

        mention_name = self._resolve_mention_name(mention_reply_to)
        send_text = f"@{mention_name or mention_reply_to} {message}" if mention_reply_to else message
        with self._send_lock:
            self._pause_visual_scan(self.cfg.mention_pause_scan_sec)
            if mention_name:
                print(
                    f"[mention-send] sender={mention_reply_to!r} "
                    f"mention={mention_name!r} title={row.title!r}"
                )
                self.sender.mention_and_send(mention_name, message)
            else:
                if mention_reply_to and self.cfg.mention_send_enabled:
                    print(
                        f"[mention-send] no mapping for sender={mention_reply_to!r}; "
                        "fallback=text"
                    )
                self._paste_and_send(send_text)
            self._pause_visual_scan(self.cfg.mention_pause_scan_sec)
        msg_w = max(24, self._term_width() - 12)
        print(f"[sent] msg={self._fit_col(send_text, msg_w)}")
        return message

    def _recover_countdown(self, seconds: int) -> None:
        total = max(0, int(seconds))
        if total <= 0:
            return
        for remain in range(total, 0, -1):
            print(f"[recover] start in {remain}")
            time.sleep(1.0)

    def _detached_windows(self):
        windows = self._enumerate_detached_windows()
        filters = [str(x).strip() for x in self.cfg.detached_window_title_filter if str(x).strip()]
        if filters:
            windows = [w for w in windows if any(token in w.title for token in filters)]
        app_titles = {
            part.strip().casefold()
            for part in str(getattr(self.cfg, "app_name", "WeChat") or "WeChat").split("|")
            if part.strip()
        }
        if app_titles.intersection({"wechat", "weixin", "微信"}):
            app_titles.update({"wechat", "weixin", "微信"})
        return [
            w
            for w in windows
            if w.title
            and w.title.strip().casefold() not in app_titles
            and not self._is_ignored_title_text(w.title)
        ]

    def _enumerate_detached_windows(self) -> list[DetachedWindowInfo]:
        """Enumerate detached WeChat windows, isolating the Mach-leaking
        ``CGWindowListCopyWindowInfo`` call in the worker when enabled.

        In worker mode, a failed request skips the current scan cycle. It must
        not fall back to the main process because each Quartz enumeration can
        leave unreclaimable Mach-message mappings behind.
        """
        if self.ocr_proxy is not None:
            raw = self.ocr_proxy.list_windows(self.cfg.app_name)
            if raw is None:
                return []
            try:
                return [
                    DetachedWindowInfo(
                        window_id=int(w.get("window_id", 0)),
                        owner=str(w.get("owner", "") or ""),
                        title=str(w.get("title", "") or ""),
                        x=int(w.get("x", 0)),
                        y=int(w.get("y", 0)),
                        width=int(w.get("width", 0)),
                        height=int(w.get("height", 0)),
                    )
                    for w in raw
                ]
            except (TypeError, ValueError):
                return []
        self._mach_diag_inprocess_enum += 1
        return list_detached_wechat_windows(self.cfg.app_name)

    def _resize_detached_windows_on_start(self) -> None:
        if not bool(getattr(self.cfg, "detached_window_resize_on_start", False)):
            return
        target_w = int(getattr(self.cfg, "detached_window_standard_width", 0) or 0)
        target_h = int(getattr(self.cfg, "detached_window_standard_height", 0) or 0)
        if target_w <= 0 or target_h <= 0:
            print(
                "[warn] detached window resize skipped: invalid target "
                f"{target_w}x{target_h}"
            )
            return
        windows = self._detached_windows()
        changed = 0
        skipped = 0
        failed = 0
        for window in windows:
            if abs(int(window.width) - target_w) <= 1 and abs(int(window.height) - target_h) <= 1:
                skipped += 1
                continue
            try:
                ok = set_detached_wechat_window_size(
                    window,
                    width=target_w,
                    height=target_h,
                )
            except Exception as exc:
                failed += 1
                print(f"[warn] detached window resize failed title={window.title!r}: {exc}")
                continue
            if ok:
                changed += 1
                print(
                    f"[resize] detached title={window.title!r} "
                    f"{window.width}x{window.height} -> {target_w}x{target_h}"
                )
            else:
                failed += 1
                print(f"[warn] detached window resize target not found title={window.title!r}")
        print(
            f"[start] window-resize: size={target_w}x{target_h} "
            f"changed={changed} skipped={skipped} failed={failed}"
        )

    def _canonicalize_visible_message(self, message: dict) -> dict:
        if not isinstance(message, dict):
            return message
        if str(message.get("side", "")).strip() != "other":
            return message
        sender_raw = str(message.get("sender", "") or "").strip()
        if not sender_raw:
            # Fallback: try to extract sender from text beginning
            text = str(message.get("text", "") or "").strip()
            if text:
                self.people_alias_resolver._maybe_reload()
                for alias, canonical in self.people_alias_resolver._mapping.items():
                    if text.startswith(alias + " ") or text.startswith(alias + "：") or text.startswith(alias + ":"):
                        # Found a known person name at the beginning
                        message["sender_raw"] = alias
                        message["sender"] = canonical
                        # Remove the name from text
                        remainder = text[len(alias):].lstrip(" ：:")
                        message["text"] = remainder
                        return message
            return message
            return message
        sender, raw = self._canonicalize_sender_pair(sender_raw)
        if not sender:
            message["sender_raw"] = raw or sender_raw[:40]
            message["sender"] = ""
            return message
        if raw:
            message["sender_raw"] = raw
        message["sender"] = sender
        return message

    def _canonicalize_visible_messages(self, messages: list[dict]) -> list[dict]:
        return [self._canonicalize_visible_message(message) for message in messages]

    @staticmethod
    def _detached_chat_body_hash(image: Image.Image) -> str:
        return VisibleMessageParser.chat_body_hash(image)

    @staticmethod
    def _detached_stable_chat_body_hash(image: Image.Image) -> str:
        return VisibleMessageParser.chat_body_hash(image, mask_media=True)

    @staticmethod
    def _detached_anchor_key(anchor: dict) -> str:
        return str(anchor.get("key") or anchor.get("text") or "").strip()

    @staticmethod
    def _detached_unique_anchor_map(anchors: list[dict]) -> dict[str, dict]:
        counts: dict[str, int] = defaultdict(int)
        for anchor in anchors:
            key = WeChatGuiRpaBot._detached_anchor_key(anchor)
            if key:
                counts[key] += 1
        out: dict[str, dict] = {}
        for anchor in anchors:
            key = WeChatGuiRpaBot._detached_anchor_key(anchor)
            if key and counts.get(key) == 1:
                out[key] = anchor
        return out

    @staticmethod
    def _detached_median(values: list[float]) -> float:
        ordered = sorted(values)
        if not ordered:
            return 0.0
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[mid])
        return (float(ordered[mid - 1]) + float(ordered[mid])) / 2.0

    @classmethod
    def _detached_text_anchor_changed(
        cls,
        previous: list[dict],
        current: list[dict],
        *,
        min_common: int = 3,
        shift_threshold: float = 40.0,
        jitter_px: float = 8.0,
        cluster_px: float = 24.0,
    ) -> bool | None:
        prev_map = cls._detached_unique_anchor_map(previous)
        curr_map = cls._detached_unique_anchor_map(current)
        if len(prev_map) < min_common or len(curr_map) < min_common:
            return None

        common = sorted(set(prev_map) & set(curr_map))
        if len(common) < min_common:
            return None

        deltas: list[float] = []
        for key in common:
            try:
                dy = float(curr_map[key].get("y", 0.0)) - float(prev_map[key].get("y", 0.0))
            except (TypeError, ValueError):
                continue
            if abs(dy) > jitter_px:
                deltas.append(dy)

        if deltas:
            median = cls._detached_median(deltas)
            same_direction = [
                dy
                for dy in deltas
                if (dy > 0 and median > 0) or (dy < 0 and median < 0)
            ]
            clustered = [dy for dy in same_direction if abs(dy - median) <= cluster_px]
            if (
                abs(median) >= shift_threshold
                and len(same_direction) >= max(min_common, int(len(common) * 0.65))
                and len(clustered) >= max(min_common, int(len(same_direction) * 0.70))
            ):
                return True

        prev_max_y = max(float(anchor.get("y", 0.0) or 0.0) for anchor in prev_map.values())
        new_tail = [
            anchor
            for key, anchor in curr_map.items()
            if key not in prev_map
            and float(anchor.get("y", 0.0) or 0.0) > prev_max_y + shift_threshold
        ]
        if new_tail:
            return True

        return False

    @staticmethod
    def _vision_sender_is_self(sender: str) -> bool:
        clean = re.sub(r"\s+", "", str(sender or "").strip()).lower()
        return clean in {"我", "自己", "本人", "me", "self"}

    @staticmethod
    def _vision_content_type(content: str | None) -> str:
        text = str(content or "").strip()
        if text.startswith("[图片]") or text.startswith("[图片:"):
            return "image"
        if re.match(r"^\[(表情|语音|文件|视频|链接|转账|位置|小程序|名片)\]", text):
            return "mixed"
        return "unknown" if content is None else "text"

    @staticmethod
    def _vision_image_description(content: str) -> str:
        text = str(content or "").strip()
        text = re.sub(r"^\[图片\]\s*", "", text)
        text = re.sub(r"^\[图片:[^\]]*\]\s*", "", text)
        return text.strip()

    @classmethod
    def _vision_messages_to_visible_messages(cls, messages: list[dict]) -> list[dict]:
        visible: list[dict] = []
        occurrence_counts: dict[str, int] = {}
        for idx, item in enumerate(messages):
            if not isinstance(item, dict):
                continue
            sender_raw = str(item.get("sender", "") or "").strip()
            if not sender_raw:
                continue
            raw_content = item.get("content", "")
            content = None if raw_content is None else str(raw_content).strip()
            text = "" if content is None else content
            side = "self" if cls._vision_sender_is_self(sender_raw) else "other"
            sender = "self" if side == "self" else sender_raw
            content_type = cls._vision_content_type(content)
            fingerprint_text = normalize_message_fingerprint_text(text)
            fingerprint_base = (
                f"{side}|{content_type}|{sender_raw}|{fingerprint_text}"
            )
            occurrence = occurrence_counts.get(fingerprint_base, 0)
            occurrence_counts[fingerprint_base] = occurrence + 1
            message: dict = {
                "side": side,
                "sender": sender,
                "content_type": content_type,
                "text": text,
                "bbox": [0, 0, 0, 0],
                "fingerprint": f"vision|{fingerprint_base}|{occurrence}",
                "source": "detached_window_vision_json",
                "vision_index": idx,
            }
            if side == "other":
                message["sender_raw"] = sender_raw[:40]
            mentions = re.findall(r"@[^\s@]+", text)
            if mentions:
                message["mentions"] = mentions
            if content_type == "image":
                desc = cls._vision_image_description(text)
                if desc:
                    message["vision_text"] = desc[:800]
            visible.append(message)
        return visible

    def _parse_detached_snapshot_with_vision(
        self,
        image: Image.Image,
        *,
        window_id: int,
        title: str,
    ) -> VisibleChatSnapshot:
        parsed = self.llm.parse_visible_messages_from_image(image, title)
        messages = self._vision_messages_to_visible_messages(parsed.get("messages") or [])
        messages = self._canonicalize_visible_messages(messages)
        width, height = image.size
        return VisibleChatSnapshot(
            schema="weauto_visible_messages_v1",
            window_id=int(window_id),
            title=str(title or ""),
            captured_at=round(time.time(), 3),
            source="detached_window_vision_json",
            image_size={"width": int(width), "height": int(height)},
            messages=messages,
            latest_message=messages[-1] if messages else None,
            debug=None,
        )

    @staticmethod
    def _detached_resume_text_key(message: dict) -> str:
        content_type = str(message.get("content_type", "text") or "text").strip().lower()
        text = str(message.get("text", "") or "").strip()
        if content_type == "image":
            text = " ".join(
                part
                for part in [
                    text,
                    str(message.get("vision_text", "") or "").strip(),
                    str(message.get("image_hash", "") or "").strip(),
                ]
                if part
            )
        return re.sub(r"\s+", " ", text).strip()

    def _detached_snapshot_from_restored_state(self, *, window_id: int, title: str) -> VisibleChatSnapshot | None:
        messages = self.visible_message_state.messages_for_window(window_id)
        if not messages:
            return None
        messages = self._canonicalize_visible_messages(messages)
        return VisibleChatSnapshot(
            schema="weauto_visible_messages_v1",
            window_id=int(window_id),
            title=str(title or ""),
            captured_at=round(time.time(), 3),
            source="detached_window_watchdog_resume",
            image_size={"width": 0, "height": 0},
            messages=messages,
            latest_message=messages[-1] if messages else None,
            debug=None,
        )

    def _capture_detached_image(self, window_id: int) -> tuple[Image.Image, str, str, list[dict]] | None:
        """Capture a detached window and return image, hashes, and text anchors.

        When the recycled capture+OCR worker is enabled, the leaky
        ``CGWindowListCreateImage`` capture runs in the worker subprocess: the
        worker persists the PNG to a temp path it owns and returns the path +
        body hash, and we read the file back with PIL (a plain file read, no
        CoreGraphics, no Mach growth). The returned image carries a
        ``_weauto_tmp_path`` attribute the loop deletes after use.

        Otherwise (worker disabled) this does the in-process capture + hash.
        A worker failure returns ``None`` and skips this scan cycle.
        """
        if self.ocr_proxy is not None:
            result = self.ocr_proxy.capture_only(window_id, include_text_anchors=True)
            if result is not None:
                image_path, body_hash, stable_body_hash, _image_size, text_anchors = result
                try:
                    image = Image.open(image_path)
                    image.load()
                except Exception:
                    try:
                        image_path.unlink()
                    except Exception:
                        pass
                    return None
                # Tag the temp path so the loop can delete it after use.
                try:
                    image._weauto_tmp_path = image_path  # type: ignore[attr-defined]
                except Exception:
                    pass
                return image, body_hash, stable_body_hash, text_anchors
            return None
        self._mach_diag_inprocess_capture += 1
        image = capture_window_by_id(
            window_id,
            backend=self.cfg.detached_window_capture_backend,
        )
        try:
            body_hash = self._detached_chat_body_hash(image)
            stable_body_hash = self._detached_stable_chat_body_hash(image)
            text_anchors = (
                VisibleMessageParser.text_anchors(image, self.ocr_engine)
                if self.ocr_engine is not None
                else []
            )
        except Exception:
            body_hash = ""
            stable_body_hash = body_hash
            text_anchors = []
        return image, body_hash, stable_body_hash, text_anchors

    @staticmethod
    def _close_detached_image(image) -> None:
        """Close a detached capture image and delete its worker temp file.

        When the capture ran through the worker, the PIL image is backed by a
        temp PNG on disk; closing the image alone would leave that file behind,
        so unlink it too. No-op for in-process captures (no temp path).
        """
        if image is None:
            return
        tmp_path = getattr(image, "_weauto_tmp_path", None)
        try:
            image.close()
        except Exception:
            pass
        if tmp_path is not None:
            try:
                Path(str(tmp_path)).unlink()
            except Exception:
                pass

    def _parse_detached_snapshot_ocr(
        self,
        image,
        *,
        window_id: int,
        title: str,
        title_slug: str = "",
        image_output_dir: Path | None = None,
        include_debug: bool = False,
    ) -> VisibleChatSnapshot:
        """Run the new-message OCR parse for a detached window.

        When the recycled capture+OCR worker is enabled, this dispatches to it
        (the worker captures by window_id and returns a JSON snapshot, so the
        ``image`` argument is unused on that branch — its native memory stays
        isolated in the worker). Otherwise it uses the in-process parser.
        A worker failure raises so the detached loop closes the image and skips
        the current window without contaminating the main process.
        """
        if self.ocr_proxy is not None:
            snapshot = self.ocr_proxy.parse(
                window_id,
                title=title,
                image_output_dir=image_output_dir,
                include_debug=include_debug,
            )
            if snapshot is not None:
                return snapshot
            raise RuntimeError(
                f"ocr worker unavailable after retry: window_id={window_id}"
            )
        _, parser = self._ensure_local_ocr()
        return parser.parse(
            image,
            window_id=window_id,
            title=title,
            image_output_dir=image_output_dir,
            include_debug=include_debug,
        )

    def _detached_local_snapshot_has_unseen_content(
        self,
        *,
        local_messages: list[dict],
        restored_messages: list[dict],
    ) -> bool:
        restored_keys = {
            key
            for key in (self._detached_resume_text_key(message) for message in restored_messages)
            if key
        }
        local_keys = [
            key
            for key in (self._detached_resume_text_key(message) for message in local_messages)
            if key
        ]
        if len(local_keys) > len(restored_keys) and any(key not in restored_keys for key in local_keys):
            return True
        return any(key not in restored_keys for key in local_keys[-3:])

    def _detached_message_record(self, message: dict, *, window_title: str = "") -> dict:
        side = str(message.get("side", "")).strip()
        content_type = str(message.get("content_type", "text")).strip().lower() or "text"
        text = re.sub(r"\s+", " ", str(message.get("text", "") or "")).strip()
        if content_type == "image":
            image_hash = str(message.get("image_hash", "")).strip()
            vision_text = re.sub(r"\s+", " ", str(message.get("vision_text", "") or "")).strip()
            if image_hash:
                text = f"[图片:{image_hash}]"
                if vision_text:
                    text = f"{text} {vision_text[:260]}"
            elif text:
                text = text[:320]
            else:
                text = "[图片]"
        sender = str(message.get("sender", "") or "").strip()[:40]
        if (not sender) and side == "other" and window_title:
            sender = window_title[:40]
        return {
            "role": "assistant" if side == "self" else "user",
            "content_type": content_type,
            "text": text,
            "sender": sender,
            "sender_raw": str(message.get("sender_raw", "") or "").strip()[:40],
        }

    def _detached_preview_text(self, message: dict) -> str:
        sender = str(message.get("sender", "") or "").strip()
        content_type = str(message.get("content_type", "text")).strip().lower()
        text = re.sub(r"\s+", " ", str(message.get("text", "") or "")).strip()
        if content_type == "image":
            text = "[图片]"
        if sender:
            return f"{sender}：{text}"
        return text

    def _detached_row_for_message(self, *, window_id: int, title: str, message: dict) -> ChatRowState:
        preview = self._detached_preview_text(message)
        has_mention = self.visible_message_state.is_mention(
            message,
            list(self.cfg.group_reply_keywords) + list(self.cfg.mention_keywords),
        )
        if self.cfg.mention_any_at and "@" in preview:
            has_mention = True
        fingerprint = str(message.get("fingerprint", "") or "")
        return ChatRowState(
            row_idx=int(window_id),
            text=preview,
            title=title,
            preview=preview,
            has_mention=has_mention,
            has_unread_badge=False,
            fingerprint=fingerprint,
            click_x_ratio=-1.0,
            click_y_ratio=-1.0,
        )

    def _detached_context_text(self, messages: list[dict], *, limit: int = 18) -> str:
        lines: list[str] = []
        for msg in messages[-limit:]:
            side = str(msg.get("side", ""))
            role = "我" if side == "self" else (str(msg.get("sender", "") or "对方"))
            content_type = str(msg.get("content_type", "text"))
            text = re.sub(r"\s+", " ", str(msg.get("text", "") or "")).strip()
            if content_type == "image":
                vision_text = re.sub(r"\s+", " ", str(msg.get("vision_text", "") or "")).strip()
                if not vision_text:
                    image_path = str(msg.get("image_path", "") or "").strip()
                    vision_text = self._image_vision_by_path.get(image_path, "")
                image_hash = str(msg.get("image_hash", "") or "").strip()
                text = f"[图片:{image_hash}]"
                if vision_text:
                    text += f" {vision_text[:260]}"
            if text:
                lines.append(f"{role}: {text}")
        return "\n".join(lines)[-1800:]

    def _analyze_detached_image_message(self, message: dict) -> str:
        if str(message.get("content_type", "")).strip().lower() != "image":
            return ""
        cached = str(message.get("vision_text", "") or "").strip()
        if cached:
            return cached
        image_path = str(message.get("image_path", "") or "").strip()
        if not image_path:
            return ""
        if image_path in self._image_vision_by_path:
            cached = self._image_vision_by_path[image_path]
            if cached:
                message["vision_text"] = cached
                return cached
        try:
            desc = self._describe_image_file(
                image_path,
                prompt=(
                    "这是微信聊天里用户发送的一张图片。请用中文描述图片内容；"
                    "如果有文字，请转写；如果适合识别物体/植物/截图内容，也请直接说明。"
                ),
            )
        except Exception as exc:
            desc = f"[图片识别失败: {exc}]"
        desc = re.sub(r"\s+", " ", desc).strip()[:800]
        if desc:
            message["vision_text"] = desc
            self._image_vision_by_path[image_path] = desc
        return desc

    def _select_detached_messages_to_handle(
        self,
        *,
        window_id: int,
        title: str,
        new_messages: list[dict],
        now: float,
    ) -> list[tuple[dict, bool]]:
        immediate: list[tuple[int, dict, bool]] = []
        normal_group: list[tuple[int, dict]] = []
        incoming_count = 0
        skipped: defaultdict[str, int] = defaultdict(int)
        log_verbose = bool(getattr(self.cfg, "log_verbose", False))
        for idx, message in enumerate(new_messages):
            if not self.visible_message_state.is_incoming(message):
                continue
            incoming_count += 1
            row = self._detached_row_for_message(window_id=window_id, title=title, message=message)
            if self._is_ignored_title(row):
                skipped["ignored_title"] += 1
                if log_verbose:
                    print(
                        f"[batch-skip] detached title={title!r} reason=ignored_title "
                        f"{self._detached_message_fields(message)}"
                    )
                continue
            is_admin = self._is_admin_session(row)
            if (not is_admin) and self._is_row_muted(row):
                skipped["muted"] += 1
                if log_verbose:
                    print(
                        f"[batch-skip] detached title={title!r} reason=muted "
                        f"{self._detached_message_fields(message)}"
                    )
                continue

            content_type = str(message.get("content_type", "text")).strip().lower()
            if content_type == "image" and not self.cfg.detached_reply_on_image:
                skipped["image_disabled"] += 1
                if log_verbose:
                    print(
                        f"[batch-skip] detached title={title!r} reason=image_disabled "
                        f"{self._detached_message_fields(message)}"
                    )
                continue

            reason = "mention" if row.has_mention else "new_message"
            is_group = self._is_group_chat(row)
            if is_group and not self._should_reply_group(row, reason):
                skipped["group_rule"] += 1
                if log_verbose:
                    print(
                        f"[batch-skip] detached title={title!r} reason=group_rule "
                        f"event={reason} preview={self._fit_col(row.preview or '-', max(24, self._term_width() - 78))}"
                    )
                continue

            if self._is_normal_reply_event(row, reason):
                normal_group.append((idx, message))
            else:
                immediate.append((idx, message, False))

        selected = list(immediate)
        if normal_group:
            normal_cooldown_active = (
                self._normal_reply_interval_active()
                and (now - self._last_normal_reply_at) < self.cfg.normal_reply_interval_sec
            )
            if not normal_cooldown_active:
                # For ordinary group chatter, answer only the newest message in
                # the batch. Mentions/private/admin messages above are all kept.
                latest_normal = max(normal_group, key=lambda item: item[0])
                selected.append((latest_normal[0], latest_normal[1], True))
                self._mark_normal_reply_at(now)
                if log_verbose:
                    print(
                        f"[batch-select] detached title={title!r} policy=latest_normal "
                        f"incoming={incoming_count} immediate={len(immediate)} normal={len(normal_group)} "
                        f"selected={len(selected)} {self._detached_message_fields(latest_normal[1])}"
                    )
            elif log_verbose:
                remain = self.cfg.normal_reply_interval_sec - (now - self._last_normal_reply_at)
                latest_normal = max(normal_group, key=lambda item: item[0])
                skipped["normal_cooldown"] += len(normal_group)
                print(
                    f"[cooldown] detached title={title!r} reason=normal_reply_interval "
                    f"drop={len(normal_group)} remain={max(0.0, remain):.1f}s "
                    f"last_normal={self._last_normal_reply_at:.0f} {self._detached_message_fields(latest_normal[1])}"
                )
        elif log_verbose and immediate:
            print(
                f"[batch-select] detached title={title!r} policy=immediate_only "
                f"incoming={incoming_count} immediate={len(immediate)} normal=0 selected={len(selected)}"
            )

        if log_verbose and incoming_count and not selected:
            skipped_text = ",".join(f"{key}:{value}" for key, value in sorted(skipped.items())) or "-"
            print(
                f"[batch-result] detached title={title!r} incoming={incoming_count} selected=0 "
                f"skipped={skipped_text}"
            )

        return [
            (message, allow_normal_cooldown_pass)
            for _, message, allow_normal_cooldown_pass in sorted(selected, key=lambda item: item[0])
        ]

    def _handle_detached_new_message(
        self,
        *,
        window_id: int,
        title: str,
        messages: list[dict],
        message: dict,
        now: float,
        allow_normal_cooldown_pass: bool = False,
    ) -> None:
        row = self._detached_row_for_message(window_id=window_id, title=title, message=message)
        if not self.visible_message_state.is_incoming(message):
            return
        if self._is_ignored_title(row):
            if self.cfg.log_verbose:
                print(
                    f"[skip-rule] detached title={title!r} reason=ignored_title "
                    f"{self._detached_message_fields(message)}"
                )
            return
        is_admin = self._is_admin_session(row)
        if (not is_admin) and self._is_row_muted(row):
            if self.cfg.log_verbose:
                print(
                    f"[skip-muted] detached title={title!r} reason=muted "
                    f"{self._detached_message_fields(message)}"
                )
            return

        latest_text = re.sub(r"\s+", " ", str(message.get("text", "") or "")).strip()
        content_type = str(message.get("content_type", "text")).strip().lower()
        if content_type == "image":
            image_path = str(message.get("image_path", "") or "").strip()
            if image_path:
                self._remember_latest_image_for_row(row, image_path)
        if content_type == "image" and not self.cfg.detached_reply_on_image:
            if self.cfg.log_verbose:
                print(
                    f"[skip-image] detached title={title!r} reason=image_disabled "
                    f"{self._detached_message_fields(message)}"
                )
            return
        if content_type == "image":
            image_desc = self._analyze_detached_image_message(message)
            if image_desc:
                latest_text = f"[图片] {image_desc}"

        reason = "mention" if row.has_mention else "new_message"
        is_group = self._is_group_chat(row)
        if is_group and not self._should_reply_group(row, reason):
            if self.cfg.log_verbose:
                print(
                    f"[skip-rule] detached title={title!r} reason=group_rule event={reason} "
                    f"preview={row.preview!r}"
                )
            return
        if (
            self._is_normal_reply_event(row, reason)
            and self._normal_reply_interval_active()
            and (now - self._last_normal_reply_at) < self.cfg.normal_reply_interval_sec
            and not allow_normal_cooldown_pass
        ):
            if self.cfg.log_verbose:
                remain = self.cfg.normal_reply_interval_sec - (now - self._last_normal_reply_at)
                print(
                    f"[cooldown] detached title={title!r} reason=normal_reply_interval "
                    f"drop=1 remain={max(0.0, remain):.1f}s "
                    f"{self._detached_message_fields(message)}"
                )
            return

        image_followup_context = self._image_followup_context_for_text(row, latest_text)
        chat_context = self._detached_context_text(messages)
        if image_followup_context:
            chat_context = f"{chat_context}\n{image_followup_context}".strip()
        session_context = self._build_session_context(row)
        workspace_context = self._workspace_context_for_row(
            row,
            is_admin=is_admin,
            skill_query=latest_text or row.preview or row.title,
        )
        memory_recall = self._workspace_memory_recall_for_row(
            row,
            latest_text or row.preview or row.title,
            is_admin=is_admin,
        )

        if self._should_handle_admin_command_locally(is_admin):
            cmd_line = self._extract_admin_command_text(row, self._detached_context_snapshot(row.preview))
            if cmd_line:
                self._append_session_item(row, "U", cmd_line)
                ack = self._handle_admin_command(cmd_line)
                reply_text = self._reply(
                    row,
                    "admin_command",
                    chat_context=chat_context,
                    session_context=session_context,
                    workspace_context=workspace_context,
                    memory_recall=memory_recall,
                    force_message=ack or "命令已执行。",
                )
                sent_norm = self._normalize_preview(reply_text)
                self._remember_sent_for_row(row, sent_norm, now)
                self._append_session_item(row, "A", reply_text)
                self._save_persistent_memory()
                return

        should_reply = (
            True
            if self.cfg.processing_mode == "long_bridge"
            else (
                True
                if self._is_immediate_reply_event(row, reason)
                else self._llm_should_reply_with_context(
                    row,
                    reason,
                    is_group,
                    chat_context,
                    "",
                    session_context,
                    workspace_context,
                    memory_recall,
                )
            )
        )
        if not should_reply:
            if self.cfg.log_verbose:
                print(
                    f"[decision] detached title={title!r} decision=skip reason={reason} "
                    f"{self._detached_message_fields(message)}"
                )
            self._save_persistent_memory()
            return
        if self.cfg.log_verbose:
            route = "long_bridge" if self.cfg.processing_mode == "long_bridge" else "native"
            print(
                f"[decision] detached title={title!r} decision=reply route={route} "
                f"reason={reason} group={self._yn(is_group)} admin={self._yn(is_admin)} "
                f"{self._detached_message_fields(message)}"
            )

        mention_reply_to = (
            (message.get("sender") or "").strip()
            if reason == "mention" and is_group
            else ""
        )
        long_bridge_handled, long_bridge_result = self._long_bridge_reply(
            row,
            reason=reason,
            is_group=is_group,
            is_admin=is_admin,
            latest_message=latest_text or row.preview,
            source_message=message,
            mention_reply_to=mention_reply_to,
        )
        if long_bridge_handled:
            sent_text = ""
            if long_bridge_result and long_bridge_result.send:
                if long_bridge_result.reply:
                    sent_text = self._reply(
                        row,
                        reason,
                        latest_message=latest_text or row.preview,
                        force_message=long_bridge_result.reply,
                        mention_reply_to=mention_reply_to,
                    )
                self._send_long_bridge_attachments(
                    row,
                    long_bridge_result.attachments,
                )
            if sent_text:
                sent_norm = self._normalize_preview(sent_text)
                self._remember_sent_for_row(row, sent_norm, now)
                self._append_session_item(row, "A", sent_text)
                if self._is_normal_reply_event(row, reason):
                    self._mark_normal_reply_at(now)
            self._save_persistent_memory()
            return

        bridge_handled, bridge_reply = self._bridge_reply_text(
            row,
            reason=reason,
            is_group=is_group,
            is_admin=is_admin,
            latest_message=latest_text or row.preview,
            chat_context=chat_context,
            session_context=session_context,
            workspace_context=workspace_context,
            memory_recall=memory_recall,
            source_message=message,
            mention_reply_to=mention_reply_to,
        )
        if bridge_handled:
            if bridge_reply:
                reply_text = self._reply(
                    row,
                    reason,
                    chat_context=chat_context,
                    session_context=session_context,
                    workspace_context=workspace_context,
                    memory_recall=memory_recall,
                    latest_message=latest_text or row.preview,
                    force_message=bridge_reply,
                    mention_reply_to=mention_reply_to,
                )
                if reply_text:
                    sent_norm = self._normalize_preview(reply_text)
                    self._remember_sent_for_row(row, sent_norm, now)
                    self._append_session_item(row, "A", reply_text)
                    if self._is_normal_reply_event(row, reason):
                        self._mark_normal_reply_at(now)
            self._save_persistent_memory()
            return

        planner_hint = ""
        if self.cfg.agent_actions_enabled:
            tools = self._available_agent_tools(is_admin=is_admin)
            if tools:
                memory_recall, planner_hint, _, _, _ = self._run_agent_planner_loop(
                    planner=self.llm_planner,
                    row=row,
                    reason=reason,
                    is_group=is_group,
                    is_admin=is_admin,
                    latest_message=latest_text or row.preview,
                    chat_context=chat_context,
                    environment_context="",
                    session_context=session_context,
                    workspace_context=workspace_context,
                    memory_recall=memory_recall,
                    tools=tools,
                    per_round_max_actions=self.cfg.agent_actions_max_per_turn,
                )
                if planner_hint:
                    memory_recall = (
                        f"[planner reply hint]\n{planner_hint}\n\n"
                        + memory_recall
                    )[:3600]

        reply_text = self._reply(
            row,
            reason,
            chat_context=chat_context,
            session_context=session_context,
            workspace_context=workspace_context,
            memory_recall=memory_recall,
            latest_message=latest_text or row.preview,
            mention_reply_to=mention_reply_to,
        )
        if reply_text:
            sent_norm = self._normalize_preview(reply_text)
            self._remember_sent_for_row(row, sent_norm, now)
            self._append_session_item(row, "A", reply_text)
            if self._is_normal_reply_event(row, reason):
                self._mark_normal_reply_at(now)
        self._save_persistent_memory()

    def _process_session_queue(self, window_id: int) -> None:
        q = self._session_queues.get(window_id)
        if q is None:
            return
        while not q.empty():
            item = q.get_nowait()
            if len(item) >= 5:
                title, messages, message, now, allow_normal_cooldown_pass = item[:5]
            else:
                title, messages, message, now = item
                allow_normal_cooldown_pass = False
            try:
                if self.visible_message_state.is_incoming(message):
                    self._handle_detached_new_message(
                        window_id=window_id, title=title,
                        messages=messages, message=message, now=now,
                        allow_normal_cooldown_pass=allow_normal_cooldown_pass,
                    )
            except Exception as exc:
                print(f"[warn] detached msg queue error window={window_id}: {exc}")
        with self._state_lock:
            self._session_busy[window_id] = False
        if q and not q.empty():
            with self._state_lock:
                if not self._session_busy.get(window_id):
                    self._session_busy[window_id] = True
                    self._msg_executor.submit(
                        self._process_session_queue, window_id
                    )

    @staticmethod
    def _detached_context_snapshot(text: str):
        from types import SimpleNamespace

        return SimpleNamespace(
            text=text,
            last_side="other",
            last_line=text,
            last_user_message=text,
            recent_messages=[text] if text else [],
            recent_structured=None,
            chat_records=None,
            memory_summary="",
            memory_time_hints=None,
            memory_people=None,
            memory_facts=None,
            memory_events=None,
            memory_relations=None,
            environment_text="",
            schema="detached",
            source="detached",
        )

    def run_detached_window_forever(self) -> None:
        self._ensure_start_activity_at(time.time())
        image_root = Path(self.cfg.detached_window_output_dir).expanduser()
        screen_capture_access = (
            "delegated-to-worker"
            if self.ocr_worker_enabled
            else screen_capture_access_granted()
        )
        print("[start] WeChat detached-window receiver started")
        print(
            f"[start] receiver=detached_windows poll={self.cfg.poll_interval_sec:.1f}s "
            f"normal_reply={self.cfg.normal_reply_interval_sec:.1f}s "
            f"dry_run={self.cfg.dry_run}"
        )
        print(f"[start] capture-backend={self.cfg.detached_window_capture_backend}")
        if self.ocr_worker_enabled:
            print(
                f"[start] ocr-worker: enabled (isolated capture+OCR subprocess, "
                f"recycle at {self.cfg.ocr_worker_max_rss_mb}MB)"
            )
        else:
            print("[start] ocr-worker: disabled (in-process capture+OCR)")
        print(f"[start] screen-capture-access={screen_capture_access}")
        if (not self.ocr_worker_enabled) and screen_capture_access is False:
            requested_access = request_screen_capture_access()
            print(
                "[warn] Screen Recording permission is missing for this launcher "
                f"(request_result={requested_access}). Grant WeAuto access in System Settings, "
                "then quit and reopen WeAuto."
            )
        print(
            f"[start] mention-send: enabled={self.cfg.mention_send_enabled} "
            f"source={self._mention_mapping_source_text()} "
            f"backspace_wait={self.cfg.mention_after_backspace_delay_sec:.2f}s"
        )
        print(f"[start] processing: {self._bridge_status_text()}")
        print(f"[start] image-dir={image_root}")
        print(f"[start] image-gen: {self._image_generation_status_text()}")
        print(f"[start] image-edit: {self._image_editing_status_text()}")
        print("[start] memory-sqlite: disabled (archived)")
        self._resize_detached_windows_on_start()
        empty_window_diagnostic_logged = False
        while True:
            self._cycle += 1
            now = time.time()
            if self._visual_scan_paused(now):
                if self.cfg.log_verbose:
                    print(f"[scan-paused] detached id={self._cycle}")
                self._maybe_run_memory_maintenance(time.time())
                time.sleep(0.2)
                continue
            windows = self._detached_windows()
            if windows:
                self._sync_long_bridge_conversations(windows)
            if windows:
                empty_window_diagnostic_logged = False
            elif not empty_window_diagnostic_logged:
                empty_window_diagnostic_logged = True
                if self.ocr_worker_enabled:
                    print(
                        "[warn] detached window enumeration returned 0; "
                        "worker request failed or no matching windows"
                    )
                else:
                    print(
                        "[warn] detached window enumeration returned 0: "
                        f"screen_capture_access={screen_capture_access_granted()} "
                        f"visible_layer0_owners={visible_window_owner_summary()}"
                    )
            if self.cfg.log_verbose:
                names = ", ".join([w.title for w in windows]) or "-"
                print(f"[cycle] id={self._cycle:>4} detached_windows={len(windows)} {self._fit_col(names, max(24, self._term_width() - 34))}")
            any_new = False
            for window in windows:
                image = None
                try:
                    title_slug = safe_window_name(window.title)
                    snapshot = None
                    resume_snapshot = None
                    if int(window.window_id) in self._detached_watchdog_resume_window_ids:
                        resume_snapshot = self._detached_snapshot_from_restored_state(
                            window_id=window.window_id,
                            title=window.title,
                        )
                    capture_result = self._capture_detached_image(window.window_id)
                    if capture_result is None:
                        raise RuntimeError(f"window capture failed: window_id={window.window_id}")
                    image, _captured_body_hash, captured_stable_body_hash, captured_text_anchors = capture_result
                    image_hash = ""
                    if resume_snapshot is not None:
                        try:
                            image_hash = f"{window.title}|{captured_stable_body_hash}"
                            self._detached_window_image_hashes[window.window_id] = image_hash
                        except Exception:
                            image_hash = ""
                        try:
                            local_snapshot = self._parse_detached_snapshot_ocr(
                                image,
                                window_id=window.window_id,
                                title=window.title,
                                title_slug=title_slug,
                                image_output_dir=image_root / title_slug / "images",
                                include_debug=False,
                            )
                            local_snapshot.messages = self._canonicalize_visible_messages(
                                local_snapshot.messages
                            )
                            if self._detached_local_snapshot_has_unseen_content(
                                local_messages=local_snapshot.messages,
                                restored_messages=resume_snapshot.messages,
                            ):
                                if self.cfg.log_verbose:
                                    print(
                                        f"[resume] detached title={self._fit_col(window.title, 14)} "
                                        "local change detected; vision deferred"
                                    )
                            else:
                                if self.cfg.log_verbose:
                                    print(
                                        f"[resume] detached title={self._fit_col(window.title, 14)} "
                                        f"messages={len(resume_snapshot.messages)}"
                                    )
                        except Exception as exc:
                            snapshot = resume_snapshot
                            if self.cfg.log_verbose:
                                print(
                                    f"[resume] detached title={self._fit_col(window.title, 14)} "
                                    f"using restored state after local precheck failed: {exc}"
                                )
                        snapshot = resume_snapshot
                        self._detached_watchdog_resume_window_ids.discard(int(window.window_id))
                    if self.cfg.detached_vision_parse_enabled and self.llm.is_vision_enabled():
                        if snapshot is None:
                            previous_anchors = self._detached_window_text_anchors.get(window.window_id, [])
                            anchor_changed = (
                                self._detached_text_anchor_changed(previous_anchors, captured_text_anchors)
                                if previous_anchors and captured_text_anchors
                                else None
                            )
                            if captured_text_anchors:
                                self._detached_window_text_anchors[window.window_id] = captured_text_anchors
                            if anchor_changed is False:
                                if self.cfg.log_verbose:
                                    print(
                                        f"[skip-anchor-stable] detached title={window.title!r} "
                                        f"anchors={len(captured_text_anchors)}"
                                    )
                                if image is not None:
                                    self._close_detached_image(image)
                                    image = None
                                continue
                            image_hash = f"{window.title}|{captured_stable_body_hash}"
                            old_hash = self._detached_window_image_hashes.get(window.window_id, "")
                            if anchor_changed is None and old_hash and old_hash == image_hash:
                                if image is not None:
                                    self._close_detached_image(image)
                                    image = None
                                continue
                            try:
                                snapshot = self._parse_detached_snapshot_with_vision(
                                    image,
                                    window_id=window.window_id,
                                    title=window.title,
                                )
                                self._detached_window_image_hashes[window.window_id] = image_hash
                                self._detached_watchdog_resume_window_ids.discard(int(window.window_id))
                                if self.cfg.log_verbose:
                                    print(
                                        f"[vision] detached title={self._fit_col(window.title, 14)} "
                                        f"messages={len(snapshot.messages)}"
                                    )
                            except Exception as exc:
                                print(f"[warn] detached vision parse failed title={window.title!r}: {exc}")
                                if not self.cfg.vision.fail_open:
                                    self._close_detached_image(image)
                                    image = None
                                    continue
                                print("[vision] fail-open: fallback to local OCR parser")
                    if snapshot is None:
                        snapshot = self._parse_detached_snapshot_ocr(
                            image,
                            window_id=window.window_id,
                            title=window.title,
                            title_slug=title_slug,
                            image_output_dir=image_root / title_slug / "images",
                            include_debug=self.cfg.detached_debug_save,
                        )
                        if image_hash:
                            self._detached_window_image_hashes[window.window_id] = image_hash
                    for msg in snapshot.messages:
                        if (not msg.get("sender")) and msg.get("side") == "other" and window.title:
                            msg["sender"] = window.title[:40]
                    snapshot.messages = self._canonicalize_visible_messages(snapshot.messages)
                    snapshot.latest_message = snapshot.messages[-1] if snapshot.messages else None
                except Exception as exc:
                    print(f"[warn] detached capture/parse failed title={window.title!r}: {exc}")
                    self._close_detached_image(image)
                    image = None
                    continue

                if self.cfg.detached_debug_save and image is not None:
                    debug_dir = image_root / title_slug / "debug"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    image.save(debug_dir / "latest_window.png")
                    (debug_dir / "latest_messages.json").write_text(
                        json.dumps(snapshot.messages, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                self._close_detached_image(image)
                image = None

                if self.archive_recorder is not None:
                    try:
                        snapshot_parser = (
                            "vision"
                            if "vision" in str(snapshot.source or "")
                            else "ocr"
                        )
                        self.archive_recorder.record_snapshot(
                            session_title=window.title,
                            messages=snapshot.messages,
                            window_id=window.window_id,
                            source="realtime",
                            captured_at=int(snapshot.captured_at),
                            parser_source=snapshot_parser,
                        )
                    except Exception as exc:
                        print(
                            f"[warn] archive record failed title={window.title!r}: {exc}"
                        )

                row_for_merge = ChatRowState(
                    row_idx=int(window.window_id),
                    text=window.title,
                    title=window.title,
                    preview="",
                    has_mention=False,
                    has_unread_badge=False,
                    fingerprint=f"window-{window.window_id}",
                    click_x_ratio=-1.0,
                    click_y_ratio=-1.0,
                )
                for image_message in snapshot.messages:
                    if str(image_message.get("content_type", "")).strip().lower() == "image":
                        self._remember_latest_image_for_row(
                            row_for_merge,
                            str(image_message.get("image_path", "") or ""),
                        )
                records = [self._detached_message_record(message, window_title=window.title) for message in snapshot.messages]
                self._merge_session_records(row_for_merge, records, source="detached")
                new_messages = self.visible_message_state.update(
                    window_id=window.window_id,
                    messages=snapshot.messages,
                )
                if not self._detached_bootstrapped and not self.cfg.detached_process_existing_on_start:
                    continue
                for message in new_messages:
                    if self.visible_message_state.is_incoming(message):
                        any_new = True
                        print(
                            f"[msg-new] detached title={self._fit_col(window.title, 14)} "
                            f"{self._detached_message_fields(message, include_raw=True)}"
                        )
                messages_to_handle = self._select_detached_messages_to_handle(
                    window_id=window.window_id,
                    title=window.title,
                    new_messages=new_messages,
                    now=now,
                )
                incoming_new_count = sum(
                    1 for m in new_messages if self.visible_message_state.is_incoming(m)
                )
                if self.cfg.log_verbose and incoming_new_count:
                    print(
                        f"[route] detached title={self._fit_col(window.title, 14)} "
                        f"incoming={incoming_new_count} enqueue={len(messages_to_handle)} "
                        f"queue_before={self._session_queues.get(window.window_id).qsize() if window.window_id in self._session_queues else 0}"
                    )
                q = self._session_queues.setdefault(window.window_id, queue.Queue())
                for message, allow_normal_cooldown_pass in messages_to_handle:
                    q.put((window.title, snapshot.messages, message, now, allow_normal_cooldown_pass))
                if self._session_busy.get(window.window_id):
                    if self.cfg.log_verbose and messages_to_handle:
                        print(f"[queue] window={window.window_id} busy, {q.qsize()} queued")
                if not self._session_busy.get(window.window_id) and not q.empty():
                    self._session_busy[window.window_id] = True
                    self._msg_executor.submit(
                        self._process_session_queue, window.window_id
                    )
            if not self._detached_bootstrapped:
                self._detached_bootstrapped = True
                print(f"[init] detached baseline windows={len(windows)}")
            if any_new:
                self._mark_activity_at(now)
            else:
                self._maybe_run_heartbeat(now, [])
            self._save_persistent_memory()
            self._maybe_run_memory_maintenance(time.time())
            time.sleep(max(0.2, float(self.cfg.poll_interval_sec)))

    def _detect_active_chat_title(
        self,
        bounds: "WindowBounds",
        *,
        forced_is_group: bool | None = None,
    ) -> str:
        candidates: list[tuple[int, str]] = []

        def _append_candidate(hint: bool | None) -> bool:
            try:
                text = self._extract_chat_header_text(bounds, is_group=hint).strip()
            except Exception:
                text = ""
            if not text:
                return False
            norm = self._normalize_title_text(text)
            score = len(norm)
            if hint is None:
                score += 2
            if any("\u4e00" <= ch <= "\u9fff" for ch in text):
                score += 1
            candidates.append((score, text[:120]))
            return True

        if forced_is_group is None:
            for hint in (None, True, False):
                _append_candidate(hint)
        else:
            # Recover mode honors the user's selected chat type first.
            for _ in range(3):
                if _append_candidate(forced_is_group):
                    break
            if not candidates:
                _append_candidate(None)
            if not candidates:
                # Last-resort fallback: opposite type-specific region.
                # This helps when one calibrated title region drifts while the other still captures text.
                opposite = (not forced_is_group)
                if _append_candidate(opposite):
                    fallback_name = "group" if opposite else "private"
                    print(f"[recover-title] fallback region hit: {fallback_name}")
        if not candidates:
            return ""
        candidates.sort(key=lambda it: it[0], reverse=True)
        return candidates[0][1]

    def _recover_virtual_row(self, title: str, page: int) -> ChatRowState:
        return ChatRowState(
            row_idx=-1,
            text="recover",
            title=(title or "").strip() or "unknown-session",
            preview="",
            has_mention=False,
            has_unread_badge=False,
            fingerprint=f"recover-{int(time.time())}-{page}",
            click_x_ratio=0.5,
            click_y_ratio=0.5,
        )

    def _recover_wait_next_page(self) -> None:
        prompt = "[recover] 上滑到更早聊天后按 Enter 继续下一页，Ctrl-C 结束。"
        try:
            input(prompt)
        except EOFError:
            print("[recover] stdin unavailable, auto-continue in 2s")
            time.sleep(2.0)

    def _prompt_recover_chat_type(self, *, mode_tag: str) -> bool:
        prompt = f"[{mode_tag}] 选择会话类型: 1=群聊 2=私聊，输入数字后回车: "
        while True:
            try:
                answer = input(prompt).strip()
            except EOFError:
                print(f"[warn] {mode_tag} stdin unavailable, fallback=群聊")
                return True

            if answer == "1":
                print(f"[{mode_tag}] 已选择: 群聊")
                return True
            if answer == "2":
                print(f"[{mode_tag}] 已选择: 私聊")
                return False
            print(f"[warn] {mode_tag} 无效输入，请输入 1 或 2")

    def _recover_title_match_score(self, left: str, right: str) -> float:
        lhs = self._normalize_title_text(left)
        rhs = self._normalize_title_text(right)
        if not lhs or not rhs:
            return 0.0
        if lhs == rhs:
            return 100.0
        if lhs in rhs or rhs in lhs:
            return 88.0 - abs(len(lhs) - len(rhs))
        return SequenceMatcher(a=lhs, b=rhs).ratio() * 50.0

    def _recover_sidebar_title(
        self,
        bounds: "WindowBounds",
        *,
        header_title: str = "",
        last_title: str = "",
        mode_tag: str = "recover",
    ) -> str:
        try:
            rows = self._scan_rows_from_bounds(bounds)
        except Exception as exc:
            if self.cfg.log_verbose:
                print(f"[warn] {mode_tag} sidebar scan failed: {exc}")
            return ""

        rows = [item for item in rows if self._normalize_session_title_display(item.title)]
        if not rows:
            return ""

        if last_title:
            expected = self._recover_virtual_row(last_title, -1)
            matched = self._find_row_in_snapshot(rows, expected)
            if matched and matched.title:
                title = self._normalize_session_title_display(matched.title)
                if title:
                    if title != self._normalize_session_title_display(last_title):
                        print(
                            f"[{mode_tag}] title-from-row | "
                            f"prev={self._fit_col(last_title, 16)} -> use={self._fit_col(title, 16)}"
                        )
                    return title

        best_title = ""
        best_score = 0.0
        for item in rows:
            title = self._normalize_session_title_display(item.title)
            if not title:
                continue
            score = self._recover_title_match_score(title, header_title)
            if score > best_score:
                best_score = score
                best_title = title
        if best_title and best_score >= 42.0:
            if header_title and best_title != self._normalize_session_title_display(header_title):
                print(
                    f"[{mode_tag}] title-from-row | "
                    f"header={self._fit_col(header_title, 16)} -> use={self._fit_col(best_title, 16)}"
                )
            return best_title
        return ""

    def _recover_resolve_title(
        self,
        bounds: "WindowBounds",
        *,
        last_title: str = "",
        mode_tag: str = "recover",
        forced_is_group: bool | None = None,
    ) -> str:
        title_ocr = self._detect_active_chat_title(
            bounds,
            forced_is_group=forced_is_group,
        )
        sidebar_title = self._recover_sidebar_title(
            bounds,
            header_title=title_ocr,
            last_title=last_title,
            mode_tag=mode_tag,
        )
        if sidebar_title and title_ocr and (not self._is_chat_header_matched(sidebar_title, title_ocr)):
            print(
                f"[{mode_tag}] row/header mismatch | "
                f"row={self._fit_col(sidebar_title, 16)} | header={self._fit_col(title_ocr, 16)}"
            )
        title_seed = sidebar_title or title_ocr
        next_title = self._stabilize_recover_title(
            title_seed,
            last_title=last_title,
            mode_tag=mode_tag,
        ) or "unknown-session"
        if not title_seed:
            print(
                f"[warn] {mode_tag} title OCR empty, fallback={self._fit_col(next_title, 16)}"
            )
        return next_title

    def _prompt_recover_title(
        self,
        *,
        mode_tag: str,
        forced_is_group: bool,
    ) -> str:
        while True:
            self._activate_wechat()
            time.sleep(self.cfg.activate_wait_sec)
            try:
                bounds = get_front_window_bounds(self.cfg.app_name)
            except WindowNotFoundError as exc:
                print(f"[warn] {mode_tag} window not ready: {exc}")
                continue

            self._log_recover_title_probe(
                bounds,
                mode_tag=mode_tag,
                forced_is_group=forced_is_group,
            )
            title = self._recover_resolve_title(
                bounds,
                last_title="",
                mode_tag=mode_tag,
                forced_is_group=forced_is_group,
            )
            if not title or title == "unknown-session":
                print(f"[warn] {mode_tag} 标题识别失败，正在重新识别")
                continue

            try:
                answer = input(
                    f"[{mode_tag}] 识别标题={self._fit_col(title, 24)} | 1=对 2=不对，输入数字后回车: "
                ).strip()
            except EOFError:
                print(f"[warn] {mode_tag} stdin unavailable, accept={self._fit_col(title, 24)}")
                return title

            if answer == "1":
                print(f"[{mode_tag}] 已确认标题: {self._fit_col(title, 24)}")
                return title
            if answer == "2":
                print(f"[{mode_tag}] 重新识别标题")
                continue
            print(f"[warn] {mode_tag} 无效输入，请输入 1 或 2")

    def _stabilize_recover_title(
        self,
        title_ocr: str,
        *,
        last_title: str = "",
        mode_tag: str = "recover",
    ) -> str:
        candidate = self._normalize_session_title_display(title_ocr)
        previous = self._normalize_session_title_display(last_title)
        if candidate and previous and self._is_chat_header_matched(previous, candidate):
            if candidate != previous:
                print(
                    f"[{mode_tag}] title-stabilized | "
                    f"ocr={self._fit_col(candidate, 16)} -> use={self._fit_col(previous, 16)}"
                )
            return previous

        if candidate:
            resolved_key = self._resolve_session_key_by_query(candidate)
            if resolved_key:
                resolved_title = self._normalize_session_title_display(
                    self._display_session_name(resolved_key)
                )
                if resolved_title and self._is_chat_header_matched(resolved_title, candidate):
                    if candidate != resolved_title:
                        print(
                            f"[{mode_tag}] title-corrected | "
                            f"ocr={self._fit_col(candidate, 16)} -> use={self._fit_col(resolved_title, 16)}"
                        )
                    return resolved_title

        return candidate or previous

    def _recover_capture_page(
        self,
        *,
        page: int,
        last_title: str = "",
        mode_tag: str = "recover",
        forced_is_group: bool | None = None,
        fixed_title: str = "",
    ) -> RecoverCaptureResult:
        self._activate_wechat()
        time.sleep(self.cfg.activate_wait_sec)
        bounds = get_front_window_bounds(self.cfg.app_name)

        if fixed_title:
            title = self._normalize_session_title_display(fixed_title) or fixed_title.strip()
            next_title = title
        else:
            next_title = self._recover_resolve_title(
                bounds,
                last_title=last_title,
                mode_tag=mode_tag,
                forced_is_group=forced_is_group,
            )
            title = next_title

        row = self._recover_virtual_row(title, page)
        session_key = self._session_key_for_row(row)
        sess_before = self._get_or_create_session(session_key)
        before_count = len(sess_before.history)
        is_group = forced_is_group if forced_is_group is not None else self._is_group_chat(row)

        session_context = self._build_session_context(row)
        session_history = self._build_session_history_text(row)
        workspace_context = self._workspace_context_for_row(
            row,
            is_admin=False,
            skill_query=title or session_key,
        )
        memory_recall = self._workspace_memory_recall_for_row(
            row,
            title or session_key,
            is_admin=False,
        )

        snapshot = self._extract_chat_context(
            bounds,
            title=title,
            reason="recover",
            is_group=is_group,
            session_context=session_context,
            session_history=session_history,
            workspace_context=workspace_context,
            memory_recall=memory_recall,
            latest_hint="recover_page_scan",
            preview="",
        )

        records = list(snapshot.chat_records or [])
        if records:
            self._merge_session_records(
                row,
                records,
                source="recover",
                order_mode="recover",
            )
        if snapshot.memory_summary:
            self._apply_session_summary(row, snapshot.memory_summary)
        self._rewrite_recover_workspace_session(row)
        self._save_persistent_memory()

        sess_after = self._get_or_create_session(session_key)
        appended = max(0, len(sess_after.history) - before_count)
        print(
            f"[{mode_tag}] page={page} done | title={self._fit_col(title, 16)} "
            f"| key={self._fit_col(session_key, 12)} | parsed={len(records):>2} | appended={appended:>2}"
        )
        if snapshot.last_line:
            line_w = max(24, self._term_width() - 18)
            print(f"          last={self._fit_col(snapshot.last_line, line_w)}")
        return RecoverCaptureResult(
            bounds=bounds,
            title=title,
            session_key=session_key,
            parsed=len(records),
            appended=appended,
            last_line=snapshot.last_line,
            next_title=next_title,
        )

    def _recover_scroll_probe(self, bounds: "WindowBounds") -> np.ndarray:
        x = bounds.x + int(bounds.width * self.cfg.chat_context_region.x)
        y = bounds.y + int(bounds.height * self.cfg.chat_context_region.y)
        w = int(bounds.width * self.cfg.chat_context_region.w)
        h = int(bounds.height * self.cfg.chat_context_region.h)
        if w <= 0 or h <= 0:
            return np.zeros((1, 1), dtype=np.uint8)
        # Scroll probe is high-frequency in recover mode; normal resolution is
        # enough for diff checks and keeps memory stable.
        shot = screenshot_region(x, y, w, h, high_res=False)
        gray_img = None
        try:
            gray_img = shot.convert("L")
            gray = np.array(gray_img, dtype=np.uint8, copy=True)
        finally:
            if gray_img is not None:
                try:
                    gray_img.close()
                except Exception:
                    pass
            try:
                shot.close()
            except Exception:
                pass
        if gray.ndim != 2 or gray.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)
        pad_y = max(1, int(gray.shape[0] * 0.04))
        pad_x = max(1, int(gray.shape[1] * 0.04))
        if gray.shape[0] > pad_y * 2 and gray.shape[1] > pad_x * 2:
            gray = gray[pad_y:-pad_y, pad_x:-pad_x]
        return gray[::3, ::3]

    def _recover_scroll_changed(
        self,
        before: np.ndarray,
        after: np.ndarray,
    ) -> tuple[bool, float, float]:
        if before.shape != after.shape:
            return True, 999.0, 1.0
        if before.size == 0 or after.size == 0:
            return False, 0.0, 0.0
        diff = np.abs(before.astype(np.int16) - after.astype(np.int16))
        mean_diff = float(diff.mean())
        changed_ratio = float((diff >= 12).mean())
        moved = (mean_diff >= 3.0) or (changed_ratio >= 0.015)
        return moved, mean_diff, changed_ratio

    def _recover_auto_click_xy(self, bounds: "WindowBounds") -> tuple[int, int]:
        point = self.cfg.recover_auto_click_point
        if not (0.0 <= point.x <= 1.0 and 0.0 <= point.y <= 1.0):
            raise ValueError(
                "recover_auto_click_point invalid; run ./carlibrate_recover_auto.sh config.toml first"
            )
        return (
            bounds.x + int(bounds.width * point.x),
            bounds.y + int(bounds.height * point.y),
        )

    def _recover_auto_scroll_once(self, bounds: "WindowBounds", *, page: int) -> bool:
        amount = max(1, abs(int(self.cfg.recover_auto_scroll_amount)))
        pause_sec = max(0.15, float(self.cfg.recover_auto_scroll_pause_sec))
        before = self._recover_scroll_probe(bounds)
        click_x, click_y = self._recover_auto_click_xy(bounds)
        self._safe_click(click_x, click_y)
        time.sleep(0.08)
        self._perform_scroll(amount)
        time.sleep(pause_sec)
        latest_bounds = get_front_window_bounds(self.cfg.app_name)
        after = self._recover_scroll_probe(latest_bounds)
        moved, mean_diff, changed_ratio = self._recover_scroll_changed(before, after)
        if moved:
            print(
                f"[recover-auto] page={page} scrolled | steps={amount} "
                f"| diff={mean_diff:.2f} | change={changed_ratio:.3f}"
            )
        else:
            print(
                f"[recover-auto] stop | top reached | steps={amount} "
                f"| diff={mean_diff:.2f} | change={changed_ratio:.3f}"
            )
        return moved

    def run_recover_mode(self, *, countdown_sec: int = 3) -> None:
        self._ensure_start_activity_at(time.time())
        page = 0
        forced_is_group = self._prompt_recover_chat_type(mode_tag="recover")
        fixed_title = self._prompt_recover_title(
            mode_tag="recover",
            forced_is_group=forced_is_group,
        )
        print("[start] recover mode started")
        print(f"[start] fixed title={self._fit_col(fixed_title, 24)}")
        print("[start] 启动时会执行: 标题识别 -> 人工确认")
        print("[start] 每页会执行: 聊天截图分析 -> 记忆落盘")
        print("[start] 完成后请手动上滑聊天记录，再按 Enter 开始下一页")
        print("[start] Ctrl-C 可随时退出，已写入的记忆会保留")

        try:
            while True:
                page += 1
                print("")
                print(f"[recover] page={page} ready")
                self._recover_countdown(countdown_sec)

                try:
                    self._recover_capture_page(
                        page=page,
                        mode_tag="recover",
                        forced_is_group=forced_is_group,
                        fixed_title=fixed_title,
                    )
                except WindowNotFoundError as exc:
                    print(f"[warn] recover window not ready: {exc}")
                    self._recover_wait_next_page()
                    page -= 1
                    continue
                except Exception as exc:
                    print(f"[warn] recover capture failed: {exc}")
                    self._recover_wait_next_page()
                    continue
                self._recover_wait_next_page()
        except KeyboardInterrupt:
            self._save_persistent_memory()
            print("")
            print(f"[recover] stopped | pages={page}")

    def run_recover_auto_mode(self, *, countdown_sec: int = 3) -> None:
        self._ensure_start_activity_at(time.time())
        page = 0
        forced_is_group = self._prompt_recover_chat_type(mode_tag="recover-auto")
        fixed_title = self._prompt_recover_title(
            mode_tag="recover-auto",
            forced_is_group=forced_is_group,
        )
        print("[start] recover-auto mode started")
        print(f"[start] fixed title={self._fit_col(fixed_title, 24)}")
        print("[start] 启动时会执行: 标题识别 -> 人工确认")
        print("[start] 每页会执行: 聊天截图分析 -> 记忆落盘 -> 安全点击 -> 自动上滑")
        print("[start] 到达最顶且无法继续上滑后会自动停止")
        try:
            while True:
                page += 1
                print("")
                print(f"[recover-auto] page={page} ready")
                self._recover_countdown(countdown_sec)
                try:
                    result = self._recover_capture_page(
                        page=page,
                        mode_tag="recover-auto",
                        forced_is_group=forced_is_group,
                        fixed_title=fixed_title,
                    )
                except WindowNotFoundError as exc:
                    print(f"[warn] recover-auto window not ready: {exc}")
                    break
                except Exception as exc:
                    print(f"[warn] recover-auto capture failed: {exc}")
                    break

                try:
                    if not self._recover_auto_scroll_once(result.bounds, page=page):
                        break
                except Exception as exc:
                    print(f"[warn] recover-auto scroll failed: {exc}")
                    break
        except KeyboardInterrupt:
            print("")
            print(f"[recover-auto] interrupted | pages={page}")
        finally:
            self._save_persistent_memory()
            print(f"[recover-auto] stopped | pages={page}")

    def run_forever(self) -> None:
        if self.cfg.receiver_mode == "detached_windows":
            self.run_detached_window_forever()
            return

        self._ensure_start_activity_at(time.time())
        print("[start] WeChat GUI RPA started")
        print("[start] perms: Accessibility + Screen Recording")
        print(
            f"[start] decision: enabled={self.cfg.llm_decision.decision_enabled} "
            f"grp={self.cfg.llm_decision.decision_on_group} "
            f"priv={self.cfg.llm_decision.decision_on_private} "
            f"mention_only={self.cfg.group_only_reply_when_mentioned} "
            f"group_no_reply={self.cfg.group_allow_llm_no_reply}"
        )
        print(
            f"[start] cadence: poll={self.cfg.poll_interval_sec:.1f}s "
            f"cooldown={self.cfg.action_cooldown_sec:.1f}s "
            f"normal_reply={self.cfg.normal_reply_interval_sec:.1f}s "
            f"send_delay={self.cfg.send_after_paste_delay_sec:.2f}s "
            f"(immediate=private/@)"
        )
        print(
            f"[start] mention-send: enabled={self.cfg.mention_send_enabled} "
            f"source={self._mention_mapping_source_text()} "
            f"backspace_wait={self.cfg.mention_after_backspace_delay_sec:.2f}s"
        )
        print(f"[start] processing: {self._bridge_status_text()}")
        print(
            f"[start] reply-backend: {self.llm_reply.reply_backend_name()} "
            f"(vision={self.llm_reply.is_vision_enabled()} llm={self.llm_reply.is_enabled()})"
        )
        print(
            f"[start] llm-profiles: reply={self.cfg.llm_reply.model} "
            f"decision={self.cfg.llm_decision.model} "
            f"planner={self.cfg.llm_planner.model} "
            f"summary={self.cfg.llm_summary.model} "
            f"heartbeat={self.cfg.llm_heartbeat.model}"
        )
        print(f"[start] tone: sarcasm_level={self.cfg.llm_reply.sarcasm_level}")
        admin_titles = ",".join(self.cfg.admin_session_titles) if self.cfg.admin_session_titles else "-"
        admin_w = max(24, self._term_width() - 17)
        path_w = max(20, self._term_width() - 35)
        print(
            f"[start] memory: enabled={self.cfg.memory_enabled} "
            f"path={self._fit_col(str(self._memory_path), path_w)}"
        )
        print(
            f"[start] agent-actions: enabled={self.cfg.agent_actions_enabled} "
            f"max={self.cfg.agent_actions_max_per_turn} "
            f"reply_max={self.cfg.agent_reply_max_messages_per_turn} "
            f"fail_open={self.cfg.agent_actions_fail_open} "
            f"loop={self.cfg.agent_plan_loop_enabled} "
            f"rounds={self.cfg.agent_plan_max_rounds} "
            f"total={self.cfg.agent_plan_max_total_actions}"
        )
        print(
            f"[start] heartbeat: enabled={self.cfg.heartbeat_enabled} "
            f"interval={self.cfg.heartbeat_interval_sec:.1f}s "
            f"idle_min={self.cfg.heartbeat_min_idle_sec:.1f}s "
            f"max_actions={self.cfg.heartbeat_max_actions} "
            f"max_people={self.cfg.heartbeat_max_people} "
            f"people_history={self.cfg.heartbeat_people_history_records}"
        )
        print(f"[start] web-search: {self._web_search_status_text()}")
        print(f"[start] image-gen: {self._image_generation_status_text()}")
        print(f"[start] image-edit: {self._image_editing_status_text()}")
        print("[start] memory-sqlite: disabled (archived)")
        print("[start] rerank: disabled (archived)")
        print(f"        admin={self._fit_col(admin_titles, admin_w)}")
        while True:
            self._cycle += 1
            now = time.time()
            if self._visual_scan_paused(now):
                if self.cfg.log_verbose:
                    print(f"[scan-paused] id={self._cycle}")
                self._maybe_run_memory_maintenance(time.time())
                time.sleep(0.2)
                continue
            try:
                bounds = get_front_window_bounds(self.cfg.app_name)
            except WindowNotFoundError as exc:
                print(f"[warn] {exc}")
                self._maybe_run_memory_maintenance(time.time())
                time.sleep(self.cfg.poll_interval_sec)
                continue

            # Main loop capture is high-frequency (poll interval), so avoid
            # high-res mode to keep memory stable in long-running sessions.
            shot = screenshot_region(
                bounds.x,
                bounds.y,
                bounds.width,
                bounds.height,
                high_res=False,
            )
            shot_rgb = self._to_np_rgb(shot)

            ocr_engine, _ = self._ensure_local_ocr()
            detected = detect_chat_rows(shot_rgb, bounds, self.cfg, ocr_engine)
            self._log_cycle_snapshot(detected.rows, now)
            if not self._baseline:
                self._set_baseline(detected.rows, now)
                self._maybe_run_memory_maintenance(time.time())
                time.sleep(self.cfg.poll_interval_sec)
                continue

            event = self._pick_event(detected.rows, now)
            if event:
                self._idle_streak = 0
                self._mark_activity_at(now)
                row, reason = event
                self.message_handler.handle_event(
                    rows=detected.rows,
                    row=row,
                    reason=reason,
                    now=now,
                )
            else:
                self._idle_streak += 1
                if self.cfg.log_verbose:
                    print(f"[idle] id={self._cycle} streak={self._idle_streak}")
                self._maybe_run_heartbeat(now, detected.rows)
                self._save_persistent_memory()

            self._maybe_run_memory_maintenance(time.time())
            time.sleep(self.cfg.poll_interval_sec)
