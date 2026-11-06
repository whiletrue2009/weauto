from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass
class RegionRatio:
    x: float
    y: float
    w: float
    h: float


@dataclass
class PointRatio:
    x: float
    y: float


@dataclass
class UnreadBadgeConfig:
    min_blob_pixels: int = 70


@dataclass
class UnreadBadgeCircleConfig:
    enabled: bool = False
    # Row-local ratios with left-bottom anchor: x from left, y from bottom.
    x: float = 0.86
    y: float = 0.66
    r: float = 0.18


@dataclass
class OcrConfig:
    # Primary OCR backend: rapidocr | paddleocr | cnocr
    backend: str = "rapidocr"
    # Optional A/B backend for side-by-side comparison logs.
    # Empty string means disabled.
    ab_compare_backend: str = ""
    # Sampling rate in [0, 1]. 0 disables A/B runs.
    ab_compare_sample_rate: float = 0.0
    # Truncate A/B preview text in logs.
    ab_compare_max_text_len: int = 120
    # Image enhancement options (used by rapidocr/cnocr; paddle relies on its own pipeline).
    enhance: bool = False
    target_short_side: int = 900
    max_upscale: float = 3.2
    # Paddle/CnOCR language code.
    paddle_lang: str = "ch"
    # Old PaddleOCR argument; harmless when backend is not paddleocr.
    paddle_use_angle_cls: bool = False


@dataclass
class LlmConfig:
    enabled: bool = False
    base_url: str = "https://api.openai.com/v1"
    base_url_env: str = "OPENAI_BASE_URL"
    api_key: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "gpt-4o-mini"
    api_format: str = "openai"
    ollama_native: bool = False
    ollama_think: str = ""
    # Controls think mode when using OpenAI-compatible /chat/completions.
    # valid: default | on | off
    openai_compat_think_mode: str = "default"
    temperature: float = 0.3
    presence_penalty: float = 0.2
    frequency_penalty: float = 0.2
    # <=0 means do not send max_tokens to provider (no client-side cap).
    max_tokens: int = 0
    timeout_sec: float = 20.0
    interest_hint: str = ""
    # Reply sarcasm style: off | low | medium | high
    sarcasm_level: str = "low"
    system_prompt: str = (
        "你是微信助手。请基于给定会话标题和预览内容生成简短、礼貌、自然的中文回复，"
        "长度控制在1-2句，不要编造事实。"
    )
    decision_enabled: bool = True
    decision_on_group: bool = False
    decision_on_private: bool = False
    decision_fail_open: bool = False
    # <=0 means do not send max_tokens to provider (no client-side cap).
    decision_max_tokens: int = 0
    decision_read_chat_context: bool = True
    decision_system_prompt: str = (
        "你是消息分流器。给定会话标题、预览和触发原因，判断是否值得自动回复。"
        "仅输出JSON，格式为: {\"decision\":\"reply|skip\",\"reason\":\"<=20字\"}。"
    )
    summary_enabled: bool = True
    # <=0 means do not send max_tokens to provider (no client-side cap).
    summary_max_tokens: int = 0
    summary_system_prompt: str = (
        "你是会话摘要器。给定历史摘要和最近对话，输出新的简短中文摘要，"
        "覆盖关键信息、事实、偏好和待办。只输出摘要正文。"
    )
    anti_repeat_enabled: bool = True
    anti_repeat_window: int = 4
    anti_repeat_similarity: float = 0.82
    anti_repeat_retry: int = 1
    reasoning_exclude: bool = True
    reasoning_effort: str = "low"
    # Provider-specific maximum internal reasoning tokens. <=0 uses provider default.
    reasoning_budget: int = 0
    debug_log_payload: bool = False
    debug_log_response: bool = False


@dataclass
class VisionConfig:
    enabled: bool = False
    base_url: str = ""
    base_url_env: str = "OPENAI_BASE_URL"
    api_key: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    model: str = ""
    ollama_native: bool = False
    ollama_think: str = ""
    # Controls think mode when using OpenAI-compatible /chat/completions.
    # valid: default | on | off
    openai_compat_think_mode: str = "default"
    timeout_sec: float = 20.0
    # <=0 means do not send max_tokens to provider (no client-side cap).
    max_tokens: int = 0
    # Some OpenAI-compatible vision models do not support response_format=json_object.
    # Turn this off to skip the first failing request and send raw prompt-only JSON constraints.
    response_format_json_object: bool = True
    fail_open: bool = True
    reasoning_exclude: bool = True
    reasoning_effort: str = "low"
    reasoning_budget: int = 0
    debug_log_payload: bool = False
    debug_log_response: bool = False
    system_prompt: str = (
        "你是微信聊天截图解析器。请严格输出 JSON，schema=wechat_context_v2；"
        "不要输出 markdown，不要输出额外解释。"
    )


@dataclass
class ArchiveConfig:
    # Append-only message draft archive (docs/12-archive-storage-standard.md).
    enabled: bool = False
    path: str = "data/archive/weauto.sqlite"
    blob_dir: str = "data/archive/blobs"
    save_image_crops: bool = True


@dataclass
@dataclass
class ImageGenerationConfig:
    enabled: bool = False
    provider: str = "dashscope_z_image"
    base_url: str = "https://dashscope.aliyuncs.com/api/v1/services/aigc"
    base_url_env: str = "DASHSCOPE_BASE_URL"
    api_key: str = ""
    api_key_env: str = "DASHSCOPE_API_KEY"
    model: str = "z-image-turbo"
    timeout_sec: float = 90.0
    download_timeout_sec: float = 45.0
    default_size: str = "1024x1024"
    output_dir: str = "data/generated_images"


@dataclass
class ImageEditingConfig:
    enabled: bool = False
    provider: str = "dashscope_qwen_image_edit"
    base_url: str = "https://dashscope.aliyuncs.com/api/v1/services/aigc"
    base_url_env: str = "DASHSCOPE_BASE_URL"
    api_key: str = ""
    api_key_env: str = "DASHSCOPE_API_KEY"
    model: str = "qwen-image-2.0-pro"
    timeout_sec: float = 120.0
    download_timeout_sec: float = 45.0
    default_size: str = ""
    output_dir: str = "data/edited_images"
    watermark: bool = False
    prompt_extend: bool = True
    max_input_bytes: int = 10 * 1024 * 1024


@dataclass
class AppConfig:
    app_name: str = "WeChat"
    # native: local planner/reply/memory pipeline.
    # bridge: send normalized chat event to an external processor, then use local sender.
    processing_mode: str = "native"
    # detached_windows: capture each detached chat window by native window id.
    # legacy_list: old whole-app left-list detector.
    receiver_mode: str = "detached_windows"
    detached_window_title_filter: list[str] = field(default_factory=list)
    detached_window_output_dir: str = "data/detached_window_images"
    # auto: screencapture on macOS, Win32 PrintWindow on Windows.
    # macOS also accepts screencapture/quartz; Windows accepts win32.
    detached_window_capture_backend: str = "auto"
    detached_debug_save: bool = False
    detached_window_resize_on_start: bool = False
    detached_window_standard_width: int = 852
    detached_window_standard_height: int = 970
    # When enabled, the high-frequency detached-window capture+OCR pipeline
    # runs in a dedicated child process that the main bot auto-recycles when
    # its RSS exceeds ocr_worker_max_rss_mb. This isolates the native (Mach /
    # CoreGraphics / cv2 / onnxruntime) memory growth that cannot be reclaimed
    # by gc.collect(), so the main bot process never needs to restart.
    ocr_worker_enabled: bool = False
    ocr_worker_max_rss_mb: int = 2048
    ocr_worker_request_timeout_sec: float = 20.0
    ocr_worker_ready_timeout_sec: float = 15.0
    detached_reply_on_image: bool = False
    detached_process_existing_on_start: bool = False
    # When enabled, changed detached chat windows are parsed by the vision LLM
    # into messages JSON. Local OCR remains the fallback.
    detached_vision_parse_enabled: bool = False
    poll_interval_sec: float = 2.0
    action_cooldown_sec: float = 8.0
    normal_reply_interval_sec: float = 300.0
    dry_run: bool = True
    activate_wait_sec: float = 0.6
    click_move_duration_sec: float = 0.18
    mouse_down_hold_sec: float = 0.03
    post_select_wait_sec: float = 0.35
    send_after_paste_delay_sec: float = 0.5
    focus_verify_enabled: bool = True
    focus_verify_max_clicks: int = 3
    focus_verify_wait_sec: float = 0.20
    trigger_on_preview_change: bool = True
    debug_scan: bool = False
    log_verbose: bool = False
    log_snapshot_rows: int = 6
    process_existing_unread_on_start: bool = True
    skip_first_action_on_start: bool = True
    memory_gc_interval_sec: float = 60.0
    memory_watchdog_max_rss_mb: int = 0
    memory_enabled: bool = True
    memory_store_path: str = "data/chat_history/_index.json"
    memory_short_max_items: int = 12
    memory_short_context_items: int = 8
    memory_summary_update_every: int = 4
    memory_summary_recent_items: int = 10
    memory_summary_max_chars: int = 500
    memory_history_context_items: int = 24
    # 0 means unlimited.
    memory_history_max_items: int = 0
    memory_append_max_items: int = 200
    impression_append_max_items: int = 80
    people_aliases_enabled: bool = True
    people_aliases_path: str = "data/config/PEOPLE_ALIASES.md"
    mention_send_enabled: bool = False
    mention_trigger_suffix: str = "A"
    mention_after_paste_delay_sec: float = 0.40
    mention_after_backspace_delay_sec: float = 0.70
    mention_after_confirm_delay_sec: float = 0.45
    mention_pause_scan_sec: float = 2.0
    admin_commands_enabled: bool = True
    admin_session_titles: list[str] = field(default_factory=lambda: ["example_admin"])
    admin_command_prefix: str = "/"
    bridge_backend: str = "http"
    bridge_url: str = ""
    bridge_api_key: str = ""
    bridge_api_key_env: str = "WEAUTO_BRIDGE_API_KEY"
    bridge_timeout_sec: float = 120.0
    bridge_fail_open: bool = True
    bridge_openclaw_cli: str = "openclaw"
    bridge_openclaw_gateway_url: str = ""
    bridge_openclaw_gateway_url_env: str = "OPENCLAW_GATEWAY_URL"
    bridge_openclaw_gateway_token: str = ""
    bridge_openclaw_gateway_token_env: str = "OPENCLAW_GATEWAY_TOKEN"
    bridge_openclaw_agent_id: str = "main"
    bridge_openclaw_session_prefix: str = "agent:main:weauto"
    bridge_openclaw_thinking: str = "off"
    bridge_openclaw_deliver: bool = False
    bridge_openclaw_expect_final: bool = True
    bridge_openclaw_strip_proxy_env: bool = True
    long_bridge_url: str = ""
    long_bridge_url_env: str = "WEAUTO_LONG_BRIDGE_URL"
    long_bridge_token: str = ""
    long_bridge_token_env: str = "WEAUTO_LONG_BRIDGE_TOKEN"
    long_bridge_account_id: str = "default"
    long_bridge_timeout_sec: float = 180.0
    long_bridge_reconnect_min_sec: float = 1.0
    long_bridge_reconnect_max_sec: float = 30.0
    long_bridge_heartbeat_sec: float = 20.0
    long_bridge_attachment_max_mb: int = 20
    long_bridge_inbound_dedupe_sec: float = 120.0
    long_bridge_fail_open: bool = True
    agent_actions_enabled: bool = True
    agent_actions_max_per_turn: int = 2
    # Max reply messages planner can emit in one trigger cycle.
    agent_reply_max_messages_per_turn: int = 3
    agent_actions_fail_open: bool = True
    agent_plan_loop_enabled: bool = True
    agent_plan_max_rounds: int = 3
    agent_plan_max_total_actions: int = 6
    agent_plan_repeat_limit: int = 2
    agent_plan_observation_max_chars: int = 5200
    web_search_provider: str = "tavily"
    tavily_enabled: bool = False
    tavily_base_url: str = "https://api.tavily.com"
    tavily_api_key: str = ""
    tavily_api_key_env: str = "TAVILY_API_KEY"
    tavily_max_results: int = 3
    tavily_timeout_sec: float = 8.0
    brave_enabled: bool = False
    brave_base_url: str = "https://api.search.brave.com/res/v1/web"
    brave_api_key: str = ""
    brave_api_key_env: str = "BRAVE_SEARCH_API_KEY"
    brave_max_results: int = 3
    brave_timeout_sec: float = 8.0
    # Volcengine Ark built-in web search (independent tool: web_search_volc)
    volc_ark_enabled: bool = False
    volc_ark_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    volc_ark_api_key: str = ""
    volc_ark_api_key_env: str = "ARK_API_KEY"
    volc_ark_model: str = "doubao-seed-1-8-251228"
    volc_ark_limit: int = 8
    volc_ark_max_keyword: int = 3
    volc_ark_timeout_sec: float = 20.0
    weather_enabled: bool = False
    weather_provider: str = "qweather"
    weather_default_location: str = "临沧"
    weather_forecast_days: int = 3
    qweather_api_host: str = ""
    qweather_auth_type: str = "jwt"
    qweather_api_key: str = ""
    qweather_api_key_env: str = "QWEATHER_API_KEY"
    qweather_jwt_key_id: str = ""
    qweather_jwt_project_id: str = ""
    qweather_jwt_private_key_path: str = "data/secrets/qweather-ed25519-private.pem"
    qweather_jwt_ttl_sec: int = 900
    qweather_timeout_sec: float = 8.0
    # When True, run_python sandbox blocks imports and limits builtins for safety.
    # Set False on trusted machines to allow full Python (import, file I/O, etc).
    python_sandbox_restricted: bool = True

    # Agent Reach provider (via mcporter + Exa MCP)
    agent_reach_enabled: bool = False
    agent_reach_mcporter_cmd: str = "mcporter"
    agent_reach_max_results: int = 5
    agent_reach_timeout_sec: float = 12.0
    heartbeat_enabled: bool = False
    heartbeat_interval_sec: float = 300.0
    heartbeat_min_idle_sec: float = 20.0
    heartbeat_max_actions: int = 24
    heartbeat_max_people: int = 10
    heartbeat_people_history_records: int = 100
    heartbeat_prompt: str = (
        "Scheduled self-reflection heartbeat. Maintain data/memory/core.md, "
        "data/memory/timeline.md, and data/people/*.md only when recent chat "
        "activity justifies it."
    )
    reply_on_new_message: str = "已收到消息，我稍后回复你。"
    reply_on_mention: str = "收到@，我看到了，稍后处理。"
    mention_keywords: list[str] = field(
        default_factory=lambda: ["@我", "有人@我", "[有人@我]"]
    )
    mention_any_at: bool = False
    group_title_prefixes: list[str] = field(default_factory=lambda: ["群"])
    group_detect_sender_prefix: bool = True
    group_require_sender_prefix_for_new_message: bool = True
    group_only_reply_when_mentioned: bool = True
    # Allow reply LLM to explicitly skip sending in group chats via [NO_REPLY].
    group_allow_llm_no_reply: bool = True
    group_reply_keywords: list[str] = field(
        default_factory=lambda: ["@助手", "@机器人", "机器人", "bot", "小助手", "@萨比", "萨比"]
    )
    ignore_title_keywords: list[str] = field(default_factory=lambda: ["折叠的聊天"])
    ignore_exact_titles: list[str] = field(
        default_factory=lambda: ["微信", "WeChat", "Weixin"]
    )
    use_manual_row_boxes: bool = False
    manual_row_boxes_path: str = "data/manual_row_boxes.json"
    row_title_region_enabled: bool = False
    # Row-local ratios with left-bottom anchor: x from left, y from bottom.
    row_title_region: RegionRatio = field(
        default_factory=lambda: RegionRatio(x=0.24, y=0.52, w=0.58, h=0.42)
    )
    preview_region_enabled: bool = False
    # Row-local ratios with left-bottom anchor: x from left, y from bottom.
    preview_text_region: RegionRatio = field(
        default_factory=lambda: RegionRatio(x=0.24, y=0.10, w=0.72, h=0.52)
    )

    list_region: RegionRatio = field(
        default_factory=lambda: RegionRatio(x=0.065, y=0.12, w=0.325, h=0.82)
    )
    rows_max: int = 8
    row_height_ratio: float = 0.145
    chat_context_region: RegionRatio = field(
        default_factory=lambda: RegionRatio(x=0.40, y=0.16, w=0.57, h=0.63)
    )
    chat_title_region: RegionRatio = field(
        default_factory=lambda: RegionRatio(x=0.40, y=0.01, w=0.57, h=0.10)
    )
    chat_title_region_group_enabled: bool = False
    chat_title_region_private_enabled: bool = False
    chat_title_region_group: RegionRatio = field(
        default_factory=lambda: RegionRatio(x=0.40, y=0.01, w=0.57, h=0.10)
    )
    chat_title_region_private: RegionRatio = field(
        default_factory=lambda: RegionRatio(x=0.40, y=0.01, w=0.57, h=0.10)
    )
    skip_if_latest_chat_from_self: bool = True
    # If false, self-latest skip applies to groups only.
    skip_if_latest_chat_from_self_private: bool = False

    input_point: PointRatio = field(default_factory=lambda: PointRatio(x=0.73, y=0.92))
    recover_auto_click_point: PointRatio = field(
        default_factory=lambda: PointRatio(x=-1.0, y=-1.0)
    )
    recover_auto_scroll_amount: int = 900
    recover_auto_scroll_pause_sec: float = 0.7
    unread_badge: UnreadBadgeConfig = field(default_factory=UnreadBadgeConfig)
    unread_badge_circle: UnreadBadgeCircleConfig = field(
        default_factory=UnreadBadgeCircleConfig
    )
    ocr: OcrConfig = field(default_factory=OcrConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    llm_reply: LlmConfig = field(default_factory=LlmConfig)
    llm_decision: LlmConfig = field(default_factory=LlmConfig)
    llm_planner: LlmConfig = field(default_factory=LlmConfig)
    llm_summary: LlmConfig = field(default_factory=LlmConfig)
    llm_heartbeat: LlmConfig = field(default_factory=LlmConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    image_generation: ImageGenerationConfig = field(default_factory=ImageGenerationConfig)
    image_editing: ImageEditingConfig = field(default_factory=ImageEditingConfig)


def _load_region(data: dict, key: str, default: RegionRatio) -> RegionRatio:
    raw = data.get(key, {})
    return RegionRatio(
        x=float(raw.get("x", default.x)),
        y=float(raw.get("y", default.y)),
        w=float(raw.get("w", default.w)),
        h=float(raw.get("h", default.h)),
    )


def _load_point(data: dict, key: str, default: PointRatio) -> PointRatio:
    raw = data.get(key, {})
    return PointRatio(
        x=float(raw.get("x", default.x)),
        y=float(raw.get("y", default.y)),
    )


def _load_ollama_think(raw: object, default: str) -> str:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return "true" if raw else "false"
    return str(raw).strip()


def _load_openai_think_mode(raw: object, default: str = "default") -> str:
    value = str(raw if raw is not None else default).strip().lower()
    if value in ("default", "auto", ""):
        return "default"
    if value in ("on", "true", "1", "enable", "enabled"):
        return "on"
    if value in ("off", "false", "0", "disable", "disabled"):
        return "off"
    return "default"


def _load_sarcasm_level(raw: object, default: str = "low") -> str:
    value = str(raw if raw is not None else default).strip().lower()
    if value in ("off", "none", "0", "false", "disable", "disabled", "no", "关", "关闭"):
        return "off"
    if value in ("low", "1", "mild", "light", "轻", "轻度"):
        return "low"
    if value in ("medium", "mid", "2", "normal", "default", "auto", "中", "中等"):
        return "medium"
    if value in ("high", "3", "strong", "heavy", "高", "重"):
        return "high"
    return _load_sarcasm_level(default, "low") if value != default else "low"


def _load_web_search_provider(raw: object, default: str = "tavily") -> str:
    value = str(raw if raw is not None else default).strip().lower()
    if value in ("tavily",):
        return "tavily"
    if value in ("brave", "brave_search", "brave-search"):
        return "brave"
    if value in ("agent_reach", "agent-reach", "agentreach", "reach", "exa"):
        return "agent_reach"
    if value in ("volc_ark", "volc-ark", "ark", "volc", "volces"):
        return "volc_ark"
    return _load_web_search_provider(default, "tavily") if value != default else "tavily"


def _load_weather_provider(raw: object, default: str = "qweather") -> str:
    value = str(raw if raw is not None else default).strip().lower()
    if value in ("qweather", "heweather", "hefeng", "和风天气"):
        return "qweather"
    return _load_weather_provider(default, "qweather") if value != default else "qweather"


def _load_qweather_auth_type(raw: object, default: str = "jwt") -> str:
    value = str(raw if raw is not None else default).strip().lower()
    if value in ("jwt", "json_web_token", "json-web-token"):
        return "jwt"
    if value in ("api_key", "apikey", "key"):
        return "api_key"
    return _load_qweather_auth_type(default, "jwt") if value != default else "jwt"


def _load_api_format(raw: object, default: str = "openai") -> str:
    value = str(raw if raw is not None else default).strip().lower()
    if value in ("openai", "openai_compat", "openai-compatible", "chat_completions"):
        return "openai"
    if value in ("anthropic", "anthropic_compat", "anthropic-compatible", "messages"):
        return "anthropic"
    return _load_api_format(default, "openai") if value != default else "openai"


def _load_image_generation_provider(raw: object, default: str = "openai_compat") -> str:
    value = str(raw if raw is not None else default).strip().lower()
    if value in ("openai", "openai_compat", "openai-compatible"):
        return "openai_compat"
    if value in ("dashscope", "dashscope_z_image", "dashscope-z-image", "aliyun", "bailian"):
        return "dashscope_z_image"
    return (
        _load_image_generation_provider(default, "openai_compat")
        if value != default
        else "openai_compat"
    )


def _load_llm_config(raw_obj: object, default: LlmConfig) -> LlmConfig:
    raw = raw_obj if isinstance(raw_obj, dict) else {}
    base_url = str(raw.get("base_url", default.base_url)).strip().rstrip("/")
    base_url_env = str(raw.get("base_url_env", default.base_url_env)).strip()
    if (not base_url) and base_url_env:
        base_url = str(os.getenv(base_url_env, "")).strip().rstrip("/")
    return LlmConfig(
        enabled=bool(raw.get("enabled", default.enabled)),
        base_url=base_url,
        base_url_env=base_url_env,
        api_key=str(raw.get("api_key", default.api_key)),
        api_key_env=str(raw.get("api_key_env", default.api_key_env)),
        model=str(raw.get("model", default.model)),
        api_format=_load_api_format(raw.get("api_format", default.api_format), default.api_format),
        ollama_native=bool(raw.get("ollama_native", default.ollama_native)),
        ollama_think=_load_ollama_think(
            raw.get("ollama_think", default.ollama_think),
            default.ollama_think,
        ),
        openai_compat_think_mode=_load_openai_think_mode(
            raw.get("openai_compat_think_mode", default.openai_compat_think_mode),
            default.openai_compat_think_mode,
        ),
        temperature=float(raw.get("temperature", default.temperature)),
        presence_penalty=float(raw.get("presence_penalty", default.presence_penalty)),
        frequency_penalty=float(raw.get("frequency_penalty", default.frequency_penalty)),
        max_tokens=int(raw.get("max_tokens", default.max_tokens)),
        timeout_sec=float(raw.get("timeout_sec", default.timeout_sec)),
        interest_hint=str(raw.get("interest_hint", default.interest_hint)),
        sarcasm_level=_load_sarcasm_level(
            raw.get("sarcasm_level", default.sarcasm_level),
            default.sarcasm_level,
        ),
        system_prompt=str(raw.get("system_prompt", default.system_prompt)),
        decision_enabled=bool(raw.get("decision_enabled", default.decision_enabled)),
        decision_on_group=bool(raw.get("decision_on_group", default.decision_on_group)),
        decision_on_private=bool(raw.get("decision_on_private", default.decision_on_private)),
        decision_fail_open=bool(raw.get("decision_fail_open", default.decision_fail_open)),
        decision_max_tokens=int(raw.get("decision_max_tokens", default.decision_max_tokens)),
        decision_read_chat_context=bool(
            raw.get("decision_read_chat_context", default.decision_read_chat_context)
        ),
        decision_system_prompt=str(
            raw.get("decision_system_prompt", default.decision_system_prompt)
        ),
        summary_enabled=bool(raw.get("summary_enabled", default.summary_enabled)),
        summary_max_tokens=int(raw.get("summary_max_tokens", default.summary_max_tokens)),
        summary_system_prompt=str(
            raw.get("summary_system_prompt", default.summary_system_prompt)
        ),
        anti_repeat_enabled=bool(raw.get("anti_repeat_enabled", default.anti_repeat_enabled)),
        anti_repeat_window=int(raw.get("anti_repeat_window", default.anti_repeat_window)),
        anti_repeat_similarity=float(
            raw.get("anti_repeat_similarity", default.anti_repeat_similarity)
        ),
        anti_repeat_retry=int(raw.get("anti_repeat_retry", default.anti_repeat_retry)),
        reasoning_exclude=bool(raw.get("reasoning_exclude", default.reasoning_exclude)),
        reasoning_effort=str(raw.get("reasoning_effort", default.reasoning_effort)),
        reasoning_budget=int(raw.get("reasoning_budget", default.reasoning_budget)),
        debug_log_payload=bool(raw.get("debug_log_payload", default.debug_log_payload)),
        debug_log_response=bool(raw.get("debug_log_response", default.debug_log_response)),
    )


def load_config(path: str | Path | None) -> AppConfig:
    cfg = AppConfig()
    if path is None:
        return cfg

    config_path = Path(path)
    if not config_path.exists():
        return cfg

    data = tomllib.loads(config_path.read_text(encoding="utf-8"))

    cfg.app_name = str(data.get("app_name", cfg.app_name))
    processing_mode = str(data.get("processing_mode", cfg.processing_mode)).strip().lower()
    cfg.processing_mode = (
        processing_mode
        if processing_mode in {"native", "bridge", "long_bridge"}
        else "native"
    )
    receiver_mode = str(data.get("receiver_mode", cfg.receiver_mode)).strip().lower()
    cfg.receiver_mode = receiver_mode if receiver_mode in {"detached_windows", "legacy_list"} else "detached_windows"
    title_filter = data.get("detached_window_title_filter", cfg.detached_window_title_filter)
    if isinstance(title_filter, list):
        cfg.detached_window_title_filter = [str(x).strip() for x in title_filter if str(x).strip()]
    cfg.detached_window_output_dir = str(
        data.get("detached_window_output_dir", cfg.detached_window_output_dir)
    ).strip() or cfg.detached_window_output_dir
    capture_backend = str(
        data.get("detached_window_capture_backend", cfg.detached_window_capture_backend)
    ).strip().lower()
    cfg.detached_window_capture_backend = (
        capture_backend
        if capture_backend in {"auto", "screencapture", "quartz", "win32"}
        else "auto"
    )
    cfg.detached_debug_save = bool(data.get("detached_debug_save", cfg.detached_debug_save))
    cfg.detached_window_resize_on_start = bool(
        data.get("detached_window_resize_on_start", cfg.detached_window_resize_on_start)
    )
    try:
        cfg.detached_window_standard_width = max(
            0,
            int(data.get("detached_window_standard_width", cfg.detached_window_standard_width)),
        )
    except (TypeError, ValueError):
        pass
    try:
        cfg.detached_window_standard_height = max(
            0,
            int(data.get("detached_window_standard_height", cfg.detached_window_standard_height)),
        )
    except (TypeError, ValueError):
        pass
    cfg.ocr_worker_enabled = bool(data.get("ocr_worker_enabled", cfg.ocr_worker_enabled))
    cfg.ocr_worker_max_rss_mb = max(
        256,
        int(data.get("ocr_worker_max_rss_mb", cfg.ocr_worker_max_rss_mb)),
    )
    cfg.ocr_worker_request_timeout_sec = max(
        2.0,
        float(data.get("ocr_worker_request_timeout_sec", cfg.ocr_worker_request_timeout_sec)),
    )
    cfg.ocr_worker_ready_timeout_sec = max(
        2.0,
        float(data.get("ocr_worker_ready_timeout_sec", cfg.ocr_worker_ready_timeout_sec)),
    )
    cfg.detached_reply_on_image = bool(
        data.get("detached_reply_on_image", cfg.detached_reply_on_image)
    )
    cfg.detached_process_existing_on_start = bool(
        data.get("detached_process_existing_on_start", cfg.detached_process_existing_on_start)
    )
    cfg.detached_vision_parse_enabled = bool(
        data.get("detached_vision_parse_enabled", cfg.detached_vision_parse_enabled)
    )
    cfg.poll_interval_sec = float(data.get("poll_interval_sec", cfg.poll_interval_sec))
    cfg.action_cooldown_sec = float(data.get("action_cooldown_sec", cfg.action_cooldown_sec))
    cfg.normal_reply_interval_sec = float(
        data.get("normal_reply_interval_sec", cfg.normal_reply_interval_sec)
    )
    cfg.dry_run = bool(data.get("dry_run", cfg.dry_run))
    cfg.activate_wait_sec = float(data.get("activate_wait_sec", cfg.activate_wait_sec))
    cfg.click_move_duration_sec = float(
        data.get("click_move_duration_sec", cfg.click_move_duration_sec)
    )
    cfg.mouse_down_hold_sec = float(data.get("mouse_down_hold_sec", cfg.mouse_down_hold_sec))
    cfg.post_select_wait_sec = float(data.get("post_select_wait_sec", cfg.post_select_wait_sec))
    cfg.send_after_paste_delay_sec = float(
        data.get("send_after_paste_delay_sec", cfg.send_after_paste_delay_sec)
    )
    cfg.focus_verify_enabled = bool(data.get("focus_verify_enabled", cfg.focus_verify_enabled))
    cfg.focus_verify_max_clicks = int(
        data.get("focus_verify_max_clicks", cfg.focus_verify_max_clicks)
    )
    cfg.focus_verify_wait_sec = float(
        data.get("focus_verify_wait_sec", cfg.focus_verify_wait_sec)
    )
    cfg.trigger_on_preview_change = bool(
        data.get("trigger_on_preview_change", cfg.trigger_on_preview_change)
    )
    cfg.debug_scan = bool(data.get("debug_scan", cfg.debug_scan))
    cfg.log_verbose = bool(data.get("log_verbose", cfg.log_verbose))
    cfg.log_snapshot_rows = int(data.get("log_snapshot_rows", cfg.log_snapshot_rows))
    cfg.process_existing_unread_on_start = bool(
        data.get("process_existing_unread_on_start", cfg.process_existing_unread_on_start)
    )
    cfg.skip_first_action_on_start = bool(
        data.get("skip_first_action_on_start", cfg.skip_first_action_on_start)
    )
    cfg.memory_gc_interval_sec = float(
        data.get("memory_gc_interval_sec", cfg.memory_gc_interval_sec)
    )
    cfg.memory_watchdog_max_rss_mb = int(
        data.get("memory_watchdog_max_rss_mb", cfg.memory_watchdog_max_rss_mb)
    )
    cfg.memory_enabled = bool(data.get("memory_enabled", cfg.memory_enabled))
    cfg.memory_store_path = str(data.get("memory_store_path", cfg.memory_store_path))
    cfg.memory_short_max_items = int(
        data.get("memory_short_max_items", cfg.memory_short_max_items)
    )
    cfg.memory_short_context_items = int(
        data.get("memory_short_context_items", cfg.memory_short_context_items)
    )
    cfg.memory_summary_update_every = int(
        data.get("memory_summary_update_every", cfg.memory_summary_update_every)
    )
    cfg.memory_summary_recent_items = int(
        data.get("memory_summary_recent_items", cfg.memory_summary_recent_items)
    )
    cfg.memory_summary_max_chars = int(
        data.get("memory_summary_max_chars", cfg.memory_summary_max_chars)
    )
    cfg.memory_history_context_items = int(
        data.get("memory_history_context_items", cfg.memory_history_context_items)
    )
    cfg.memory_history_max_items = int(
        data.get("memory_history_max_items", cfg.memory_history_max_items)
    )
    cfg.memory_append_max_items = int(
        data.get("memory_append_max_items", cfg.memory_append_max_items)
    )
    cfg.impression_append_max_items = int(
        data.get("impression_append_max_items", cfg.impression_append_max_items)
    )
    cfg.people_aliases_enabled = bool(
        data.get("people_aliases_enabled", cfg.people_aliases_enabled)
    )
    cfg.people_aliases_path = str(data.get("people_aliases_path", cfg.people_aliases_path))
    cfg.mention_send_enabled = bool(
        data.get("mention_send_enabled", cfg.mention_send_enabled)
    )
    cfg.mention_trigger_suffix = str(
        data.get("mention_trigger_suffix", cfg.mention_trigger_suffix)
    )[:8] or "A"
    cfg.mention_after_paste_delay_sec = float(
        data.get("mention_after_paste_delay_sec", cfg.mention_after_paste_delay_sec)
    )
    cfg.mention_after_backspace_delay_sec = float(
        data.get("mention_after_backspace_delay_sec", cfg.mention_after_backspace_delay_sec)
    )
    cfg.mention_after_confirm_delay_sec = float(
        data.get("mention_after_confirm_delay_sec", cfg.mention_after_confirm_delay_sec)
    )
    cfg.mention_pause_scan_sec = float(
        data.get("mention_pause_scan_sec", cfg.mention_pause_scan_sec)
    )
    cfg.admin_commands_enabled = bool(
        data.get("admin_commands_enabled", cfg.admin_commands_enabled)
    )
    admin_session_titles = data.get("admin_session_titles", cfg.admin_session_titles)
    if isinstance(admin_session_titles, list):
        cfg.admin_session_titles = [str(x) for x in admin_session_titles]
    cfg.admin_command_prefix = str(data.get("admin_command_prefix", cfg.admin_command_prefix))
    bridge_backend = str(data.get("bridge_backend", cfg.bridge_backend)).strip().lower()
    cfg.bridge_backend = bridge_backend if bridge_backend in {"http", "openclaw"} else "http"
    cfg.bridge_url = str(data.get("bridge_url", cfg.bridge_url)).strip()
    cfg.bridge_api_key = str(data.get("bridge_api_key", cfg.bridge_api_key))
    cfg.bridge_api_key_env = str(data.get("bridge_api_key_env", cfg.bridge_api_key_env)).strip()
    if (not cfg.bridge_api_key) and cfg.bridge_api_key_env:
        cfg.bridge_api_key = str(os.getenv(cfg.bridge_api_key_env, "")).strip()
    cfg.bridge_timeout_sec = float(data.get("bridge_timeout_sec", cfg.bridge_timeout_sec))
    cfg.bridge_fail_open = bool(data.get("bridge_fail_open", cfg.bridge_fail_open))
    if cfg.bridge_timeout_sec < 1.0:
        cfg.bridge_timeout_sec = 1.0
    cfg.bridge_openclaw_cli = str(
        data.get("bridge_openclaw_cli", cfg.bridge_openclaw_cli)
    ).strip() or "openclaw"
    cfg.bridge_openclaw_gateway_url = str(
        data.get("bridge_openclaw_gateway_url", cfg.bridge_openclaw_gateway_url)
    ).strip()
    cfg.bridge_openclaw_gateway_url_env = str(
        data.get("bridge_openclaw_gateway_url_env", cfg.bridge_openclaw_gateway_url_env)
    ).strip()
    if (not cfg.bridge_openclaw_gateway_url) and cfg.bridge_openclaw_gateway_url_env:
        cfg.bridge_openclaw_gateway_url = str(
            os.getenv(cfg.bridge_openclaw_gateway_url_env, "")
        ).strip()
    cfg.bridge_openclaw_gateway_token = str(
        data.get("bridge_openclaw_gateway_token", cfg.bridge_openclaw_gateway_token)
    )
    cfg.bridge_openclaw_gateway_token_env = str(
        data.get("bridge_openclaw_gateway_token_env", cfg.bridge_openclaw_gateway_token_env)
    ).strip()
    if (not cfg.bridge_openclaw_gateway_token) and cfg.bridge_openclaw_gateway_token_env:
        cfg.bridge_openclaw_gateway_token = str(
            os.getenv(cfg.bridge_openclaw_gateway_token_env, "")
        ).strip()
    cfg.bridge_openclaw_agent_id = str(
        data.get("bridge_openclaw_agent_id", cfg.bridge_openclaw_agent_id)
    ).strip() or "main"
    cfg.bridge_openclaw_session_prefix = str(
        data.get("bridge_openclaw_session_prefix", cfg.bridge_openclaw_session_prefix)
    ).strip() or "agent:main:weauto"
    cfg.bridge_openclaw_thinking = str(
        data.get("bridge_openclaw_thinking", cfg.bridge_openclaw_thinking)
    ).strip()
    cfg.bridge_openclaw_deliver = bool(
        data.get("bridge_openclaw_deliver", cfg.bridge_openclaw_deliver)
    )
    cfg.bridge_openclaw_expect_final = bool(
        data.get("bridge_openclaw_expect_final", cfg.bridge_openclaw_expect_final)
    )
    cfg.bridge_openclaw_strip_proxy_env = bool(
        data.get("bridge_openclaw_strip_proxy_env", cfg.bridge_openclaw_strip_proxy_env)
    )
    cfg.long_bridge_url = str(
        data.get("long_bridge_url", cfg.long_bridge_url)
    ).strip()
    cfg.long_bridge_url_env = str(
        data.get("long_bridge_url_env", cfg.long_bridge_url_env)
    ).strip()
    if (not cfg.long_bridge_url) and cfg.long_bridge_url_env:
        cfg.long_bridge_url = str(os.getenv(cfg.long_bridge_url_env, "")).strip()
    cfg.long_bridge_token = str(
        data.get("long_bridge_token", cfg.long_bridge_token)
    )
    cfg.long_bridge_token_env = str(
        data.get("long_bridge_token_env", cfg.long_bridge_token_env)
    ).strip()
    if (not cfg.long_bridge_token) and cfg.long_bridge_token_env:
        cfg.long_bridge_token = str(
            os.getenv(cfg.long_bridge_token_env, "")
        ).strip()
    cfg.long_bridge_account_id = str(
        data.get("long_bridge_account_id", cfg.long_bridge_account_id)
    ).strip() or "default"
    cfg.long_bridge_timeout_sec = max(
        5.0,
        float(data.get("long_bridge_timeout_sec", cfg.long_bridge_timeout_sec)),
    )
    cfg.long_bridge_reconnect_min_sec = max(
        0.2,
        float(
            data.get(
                "long_bridge_reconnect_min_sec",
                cfg.long_bridge_reconnect_min_sec,
            )
        ),
    )
    cfg.long_bridge_reconnect_max_sec = max(
        cfg.long_bridge_reconnect_min_sec,
        float(
            data.get(
                "long_bridge_reconnect_max_sec",
                cfg.long_bridge_reconnect_max_sec,
            )
        ),
    )
    cfg.long_bridge_heartbeat_sec = max(
        5.0,
        float(data.get("long_bridge_heartbeat_sec", cfg.long_bridge_heartbeat_sec)),
    )
    cfg.long_bridge_attachment_max_mb = max(
        1,
        int(
            data.get(
                "long_bridge_attachment_max_mb",
                cfg.long_bridge_attachment_max_mb,
            )
        ),
    )
    cfg.long_bridge_inbound_dedupe_sec = max(
        0.0,
        float(
            data.get(
                "long_bridge_inbound_dedupe_sec",
                cfg.long_bridge_inbound_dedupe_sec,
            )
        ),
    )
    cfg.long_bridge_fail_open = bool(
        data.get("long_bridge_fail_open", cfg.long_bridge_fail_open)
    )
    cfg.agent_actions_enabled = bool(
        data.get("agent_actions_enabled", cfg.agent_actions_enabled)
    )
    cfg.agent_actions_max_per_turn = int(
        data.get("agent_actions_max_per_turn", cfg.agent_actions_max_per_turn)
    )
    cfg.agent_reply_max_messages_per_turn = int(
        data.get(
            "agent_reply_max_messages_per_turn",
            cfg.agent_reply_max_messages_per_turn,
        )
    )
    cfg.agent_actions_fail_open = bool(
        data.get("agent_actions_fail_open", cfg.agent_actions_fail_open)
    )
    cfg.agent_plan_loop_enabled = bool(
        data.get("agent_plan_loop_enabled", cfg.agent_plan_loop_enabled)
    )
    cfg.agent_plan_max_rounds = int(
        data.get("agent_plan_max_rounds", cfg.agent_plan_max_rounds)
    )
    cfg.agent_plan_max_total_actions = int(
        data.get("agent_plan_max_total_actions", cfg.agent_plan_max_total_actions)
    )
    cfg.agent_plan_repeat_limit = int(
        data.get("agent_plan_repeat_limit", cfg.agent_plan_repeat_limit)
    )
    cfg.agent_plan_observation_max_chars = int(
        data.get(
            "agent_plan_observation_max_chars",
            cfg.agent_plan_observation_max_chars,
        )
    )
    if cfg.agent_plan_max_rounds < 1:
        cfg.agent_plan_max_rounds = 1
    if cfg.agent_plan_max_total_actions < 1:
        cfg.agent_plan_max_total_actions = 1
    if cfg.agent_plan_repeat_limit < 1:
        cfg.agent_plan_repeat_limit = 1
    if cfg.agent_plan_observation_max_chars < 1200:
        cfg.agent_plan_observation_max_chars = 1200
    if cfg.agent_reply_max_messages_per_turn < 1:
        cfg.agent_reply_max_messages_per_turn = 1
    cfg.web_search_provider = _load_web_search_provider(
        data.get("web_search_provider", cfg.web_search_provider),
        cfg.web_search_provider,
    )
    cfg.tavily_enabled = bool(data.get("tavily_enabled", cfg.tavily_enabled))
    cfg.tavily_base_url = str(data.get("tavily_base_url", cfg.tavily_base_url)).rstrip("/")
    cfg.tavily_api_key = str(data.get("tavily_api_key", cfg.tavily_api_key))
    cfg.tavily_api_key_env = str(data.get("tavily_api_key_env", cfg.tavily_api_key_env))
    cfg.tavily_max_results = int(data.get("tavily_max_results", cfg.tavily_max_results))
    cfg.tavily_timeout_sec = float(data.get("tavily_timeout_sec", cfg.tavily_timeout_sec))
    cfg.brave_enabled = bool(data.get("brave_enabled", cfg.brave_enabled))
    cfg.brave_base_url = str(data.get("brave_base_url", cfg.brave_base_url)).rstrip("/")
    cfg.brave_api_key = str(data.get("brave_api_key", cfg.brave_api_key))
    cfg.brave_api_key_env = str(data.get("brave_api_key_env", cfg.brave_api_key_env))
    cfg.brave_max_results = int(data.get("brave_max_results", cfg.brave_max_results))
    cfg.brave_timeout_sec = float(data.get("brave_timeout_sec", cfg.brave_timeout_sec))
    cfg.volc_ark_enabled = bool(data.get("volc_ark_enabled", cfg.volc_ark_enabled))
    cfg.volc_ark_base_url = str(
        data.get("volc_ark_base_url", cfg.volc_ark_base_url)
    ).rstrip("/")
    cfg.volc_ark_api_key = str(data.get("volc_ark_api_key", cfg.volc_ark_api_key))
    cfg.volc_ark_api_key_env = str(data.get("volc_ark_api_key_env", cfg.volc_ark_api_key_env))
    cfg.volc_ark_model = str(data.get("volc_ark_model", cfg.volc_ark_model)).strip()
    cfg.volc_ark_limit = int(data.get("volc_ark_limit", cfg.volc_ark_limit))
    cfg.volc_ark_max_keyword = int(
        data.get("volc_ark_max_keyword", cfg.volc_ark_max_keyword)
    )
    cfg.volc_ark_timeout_sec = float(
        data.get("volc_ark_timeout_sec", cfg.volc_ark_timeout_sec)
    )
    cfg.weather_enabled = bool(data.get("weather_enabled", cfg.weather_enabled))
    cfg.weather_provider = _load_weather_provider(
        data.get("weather_provider", cfg.weather_provider),
        cfg.weather_provider,
    )
    cfg.weather_default_location = str(
        data.get("weather_default_location", cfg.weather_default_location)
    ).strip()
    cfg.weather_forecast_days = int(
        data.get("weather_forecast_days", cfg.weather_forecast_days)
    )
    cfg.qweather_api_host = str(data.get("qweather_api_host", cfg.qweather_api_host)).strip().rstrip("/")
    cfg.qweather_auth_type = _load_qweather_auth_type(
        data.get("qweather_auth_type", cfg.qweather_auth_type),
        cfg.qweather_auth_type,
    )
    cfg.qweather_api_key = str(data.get("qweather_api_key", cfg.qweather_api_key))
    cfg.qweather_api_key_env = str(
        data.get("qweather_api_key_env", cfg.qweather_api_key_env)
    ).strip()
    cfg.qweather_jwt_key_id = str(
        data.get("qweather_jwt_key_id", cfg.qweather_jwt_key_id)
    ).strip()
    cfg.qweather_jwt_project_id = str(
        data.get("qweather_jwt_project_id", cfg.qweather_jwt_project_id)
    ).strip()
    cfg.qweather_jwt_private_key_path = str(
        data.get("qweather_jwt_private_key_path", cfg.qweather_jwt_private_key_path)
    ).strip()
    cfg.qweather_jwt_ttl_sec = int(
        data.get("qweather_jwt_ttl_sec", cfg.qweather_jwt_ttl_sec)
    )
    cfg.qweather_timeout_sec = float(
        data.get("qweather_timeout_sec", cfg.qweather_timeout_sec)
    )
    cfg.python_sandbox_restricted = bool(data.get("python_sandbox_restricted", cfg.python_sandbox_restricted))
    cfg.agent_reach_enabled = bool(data.get("agent_reach_enabled", cfg.agent_reach_enabled))
    cfg.agent_reach_mcporter_cmd = str(
        data.get("agent_reach_mcporter_cmd", cfg.agent_reach_mcporter_cmd)
    ).strip()
    cfg.agent_reach_max_results = int(
        data.get("agent_reach_max_results", cfg.agent_reach_max_results)
    )
    cfg.agent_reach_timeout_sec = float(
        data.get("agent_reach_timeout_sec", cfg.agent_reach_timeout_sec)
    )
    if cfg.volc_ark_limit < 1:
        cfg.volc_ark_limit = 1
    if cfg.volc_ark_limit > 20:
        cfg.volc_ark_limit = 20
    if cfg.volc_ark_max_keyword < 1:
        cfg.volc_ark_max_keyword = 1
    if cfg.volc_ark_max_keyword > 50:
        cfg.volc_ark_max_keyword = 50
    if cfg.volc_ark_timeout_sec < 1.0:
        cfg.volc_ark_timeout_sec = 1.0
    if cfg.weather_forecast_days < 1:
        cfg.weather_forecast_days = 1
    if cfg.weather_forecast_days > 30:
        cfg.weather_forecast_days = 30
    if cfg.qweather_jwt_ttl_sec < 60:
        cfg.qweather_jwt_ttl_sec = 60
    if cfg.qweather_jwt_ttl_sec > 86400:
        cfg.qweather_jwt_ttl_sec = 86400
    if cfg.qweather_timeout_sec < 1.0:
        cfg.qweather_timeout_sec = 1.0
    cfg.heartbeat_enabled = bool(data.get("heartbeat_enabled", cfg.heartbeat_enabled))
    cfg.heartbeat_interval_sec = float(
        data.get("heartbeat_interval_sec", cfg.heartbeat_interval_sec)
    )
    cfg.heartbeat_min_idle_sec = float(
        data.get("heartbeat_min_idle_sec", cfg.heartbeat_min_idle_sec)
    )
    cfg.heartbeat_max_actions = int(
        data.get("heartbeat_max_actions", cfg.heartbeat_max_actions)
    )
    cfg.heartbeat_max_people = int(
        data.get("heartbeat_max_people", cfg.heartbeat_max_people)
    )
    cfg.heartbeat_people_history_records = int(
        data.get(
            "heartbeat_people_history_records",
            cfg.heartbeat_people_history_records,
        )
    )
    cfg.heartbeat_prompt = str(data.get("heartbeat_prompt", cfg.heartbeat_prompt))
    cfg.reply_on_new_message = str(
        data.get("reply_on_new_message", cfg.reply_on_new_message)
    )
    cfg.reply_on_mention = str(data.get("reply_on_mention", cfg.reply_on_mention))

    mention_keywords = data.get("mention_keywords", cfg.mention_keywords)
    if isinstance(mention_keywords, list):
        cfg.mention_keywords = [str(x) for x in mention_keywords]
    cfg.mention_any_at = bool(data.get("mention_any_at", cfg.mention_any_at))
    group_title_prefixes = data.get("group_title_prefixes", cfg.group_title_prefixes)
    if isinstance(group_title_prefixes, list):
        cfg.group_title_prefixes = [str(x) for x in group_title_prefixes]
    cfg.group_detect_sender_prefix = bool(
        data.get("group_detect_sender_prefix", cfg.group_detect_sender_prefix)
    )
    cfg.group_require_sender_prefix_for_new_message = bool(
        data.get(
            "group_require_sender_prefix_for_new_message",
            cfg.group_require_sender_prefix_for_new_message,
        )
    )
    cfg.group_only_reply_when_mentioned = bool(
        data.get("group_only_reply_when_mentioned", cfg.group_only_reply_when_mentioned)
    )
    cfg.group_allow_llm_no_reply = bool(
        data.get("group_allow_llm_no_reply", cfg.group_allow_llm_no_reply)
    )
    group_reply_keywords = data.get("group_reply_keywords", cfg.group_reply_keywords)
    if isinstance(group_reply_keywords, list):
        cfg.group_reply_keywords = [str(x) for x in group_reply_keywords]
    ignore_title_keywords = data.get("ignore_title_keywords", cfg.ignore_title_keywords)
    if isinstance(ignore_title_keywords, list):
        cfg.ignore_title_keywords = [str(x) for x in ignore_title_keywords]
    ignore_exact_titles = data.get("ignore_exact_titles", cfg.ignore_exact_titles)
    if isinstance(ignore_exact_titles, list):
        cfg.ignore_exact_titles = [str(x) for x in ignore_exact_titles]
    cfg.use_manual_row_boxes = bool(
        data.get("use_manual_row_boxes", cfg.use_manual_row_boxes)
    )
    cfg.manual_row_boxes_path = str(
        data.get("manual_row_boxes_path", cfg.manual_row_boxes_path)
    )
    cfg.row_title_region_enabled = bool(
        data.get("row_title_region_enabled", cfg.row_title_region_enabled)
    )
    cfg.row_title_region = _load_region(
        data, "row_title_region", cfg.row_title_region
    )
    cfg.preview_region_enabled = bool(
        data.get("preview_region_enabled", cfg.preview_region_enabled)
    )
    cfg.preview_text_region = _load_region(
        data, "preview_text_region", cfg.preview_text_region
    )

    cfg.list_region = _load_region(data, "list_region", cfg.list_region)
    cfg.rows_max = int(data.get("rows_max", cfg.rows_max))
    cfg.row_height_ratio = float(data.get("row_height_ratio", cfg.row_height_ratio))
    cfg.chat_context_region = _load_region(data, "chat_context_region", cfg.chat_context_region)
    cfg.chat_title_region = _load_region(data, "chat_title_region", cfg.chat_title_region)
    cfg.chat_title_region_group_enabled = bool(
        data.get("chat_title_region_group_enabled", cfg.chat_title_region_group_enabled)
    )
    cfg.chat_title_region_private_enabled = bool(
        data.get("chat_title_region_private_enabled", cfg.chat_title_region_private_enabled)
    )
    if "chat_title_region_group" in data:
        cfg.chat_title_region_group = _load_region(
            data, "chat_title_region_group", cfg.chat_title_region_group
        )
    else:
        cfg.chat_title_region_group = cfg.chat_title_region
    if "chat_title_region_private" in data:
        cfg.chat_title_region_private = _load_region(
            data, "chat_title_region_private", cfg.chat_title_region_private
        )
    else:
        cfg.chat_title_region_private = cfg.chat_title_region
    cfg.skip_if_latest_chat_from_self = bool(
        data.get("skip_if_latest_chat_from_self", cfg.skip_if_latest_chat_from_self)
    )
    cfg.skip_if_latest_chat_from_self_private = bool(
        data.get(
            "skip_if_latest_chat_from_self_private",
            cfg.skip_if_latest_chat_from_self_private,
        )
    )

    cfg.input_point = _load_point(data, "input_point", cfg.input_point)
    cfg.recover_auto_click_point = _load_point(
        data, "recover_auto_click_point", cfg.recover_auto_click_point
    )
    cfg.recover_auto_scroll_amount = int(
        data.get("recover_auto_scroll_amount", cfg.recover_auto_scroll_amount)
    )
    cfg.recover_auto_scroll_pause_sec = float(
        data.get("recover_auto_scroll_pause_sec", cfg.recover_auto_scroll_pause_sec)
    )

    badge = data.get("unread_badge", {})
    cfg.unread_badge = UnreadBadgeConfig(
        min_blob_pixels=int(badge.get("min_blob_pixels", cfg.unread_badge.min_blob_pixels))
    )
    badge_circle = data.get("unread_badge_circle", {})
    cfg.unread_badge_circle = UnreadBadgeCircleConfig(
        enabled=bool(badge_circle.get("enabled", cfg.unread_badge_circle.enabled)),
        x=float(badge_circle.get("x", cfg.unread_badge_circle.x)),
        y=float(badge_circle.get("y", cfg.unread_badge_circle.y)),
        r=float(badge_circle.get("r", cfg.unread_badge_circle.r)),
    )
    ocr_raw = data.get("ocr", {})
    cfg.ocr = OcrConfig(
        backend=str(ocr_raw.get("backend", cfg.ocr.backend)).strip().lower()
        or cfg.ocr.backend,
        ab_compare_backend=str(
            ocr_raw.get("ab_compare_backend", cfg.ocr.ab_compare_backend)
        )
        .strip()
        .lower(),
        ab_compare_sample_rate=float(
            ocr_raw.get("ab_compare_sample_rate", cfg.ocr.ab_compare_sample_rate)
        ),
        ab_compare_max_text_len=int(
            ocr_raw.get("ab_compare_max_text_len", cfg.ocr.ab_compare_max_text_len)
        ),
        enhance=bool(ocr_raw.get("enhance", cfg.ocr.enhance)),
        target_short_side=int(ocr_raw.get("target_short_side", cfg.ocr.target_short_side)),
        max_upscale=float(ocr_raw.get("max_upscale", cfg.ocr.max_upscale)),
        paddle_lang=str(ocr_raw.get("paddle_lang", cfg.ocr.paddle_lang)).strip()
        or cfg.ocr.paddle_lang,
        paddle_use_angle_cls=bool(
            ocr_raw.get("paddle_use_angle_cls", cfg.ocr.paddle_use_angle_cls)
        ),
    )
    cfg.ocr.ab_compare_sample_rate = max(0.0, min(1.0, cfg.ocr.ab_compare_sample_rate))
    if cfg.ocr.ab_compare_max_text_len < 24:
        cfg.ocr.ab_compare_max_text_len = 24
    if cfg.ocr.target_short_side < 320:
        cfg.ocr.target_short_side = 320
    if cfg.ocr.max_upscale < 1.0:
        cfg.ocr.max_upscale = 1.0

    llm_raw = data.get("llm", {})
    cfg.llm = _load_llm_config(llm_raw, cfg.llm)
    cfg.llm_reply = _load_llm_config(data.get("llm_reply", {}), cfg.llm)
    cfg.llm_decision = _load_llm_config(data.get("llm_decision", {}), cfg.llm)
    cfg.llm_planner = _load_llm_config(data.get("llm_planner", {}), cfg.llm)
    cfg.llm_summary = _load_llm_config(data.get("llm_summary", {}), cfg.llm)
    cfg.llm_heartbeat = _load_llm_config(data.get("llm_heartbeat", {}), cfg.llm)
    archive_raw = data.get("archive", {})
    cfg.archive = ArchiveConfig(
        enabled=bool(archive_raw.get("enabled", cfg.archive.enabled)),
        path=str(archive_raw.get("path", cfg.archive.path)),
        blob_dir=str(archive_raw.get("blob_dir", cfg.archive.blob_dir)),
        save_image_crops=bool(
            archive_raw.get("save_image_crops", cfg.archive.save_image_crops)
        ),
    )
    vision_raw = data.get("vision", {})
    vision_base_url = str(vision_raw.get("base_url", cfg.vision.base_url)).strip().rstrip("/")
    vision_base_url_env = str(vision_raw.get("base_url_env", cfg.vision.base_url_env)).strip()
    if (not vision_base_url) and vision_base_url_env:
        vision_base_url = str(os.getenv(vision_base_url_env, "")).strip().rstrip("/")
    cfg.vision = VisionConfig(
        enabled=bool(vision_raw.get("enabled", cfg.vision.enabled)),
        base_url=vision_base_url,
        base_url_env=vision_base_url_env,
        api_key=str(vision_raw.get("api_key", cfg.vision.api_key)),
        api_key_env=str(vision_raw.get("api_key_env", cfg.vision.api_key_env)),
        model=str(vision_raw.get("model", cfg.vision.model)),
        ollama_native=bool(vision_raw.get("ollama_native", cfg.vision.ollama_native)),
        ollama_think=_load_ollama_think(
            vision_raw.get("ollama_think", cfg.vision.ollama_think),
            cfg.vision.ollama_think,
        ),
        openai_compat_think_mode=_load_openai_think_mode(
            vision_raw.get("openai_compat_think_mode", cfg.vision.openai_compat_think_mode),
            cfg.vision.openai_compat_think_mode,
        ),
        timeout_sec=float(vision_raw.get("timeout_sec", cfg.vision.timeout_sec)),
        max_tokens=int(vision_raw.get("max_tokens", cfg.vision.max_tokens)),
        response_format_json_object=bool(
            vision_raw.get(
                "response_format_json_object",
                cfg.vision.response_format_json_object,
            )
        ),
        fail_open=bool(vision_raw.get("fail_open", cfg.vision.fail_open)),
        reasoning_exclude=bool(
            vision_raw.get("reasoning_exclude", cfg.vision.reasoning_exclude)
        ),
        reasoning_effort=str(vision_raw.get("reasoning_effort", cfg.vision.reasoning_effort)),
        reasoning_budget=int(
            vision_raw.get("reasoning_budget", cfg.vision.reasoning_budget)
        ),
        debug_log_payload=bool(
            vision_raw.get("debug_log_payload", cfg.vision.debug_log_payload)
        ),
        debug_log_response=bool(
            vision_raw.get("debug_log_response", cfg.vision.debug_log_response)
        ),
        system_prompt=str(vision_raw.get("system_prompt", cfg.vision.system_prompt)),
    )
    if not cfg.vision.base_url:
        cfg.vision.base_url = cfg.llm.base_url
    if not cfg.vision.api_key:
        cfg.vision.api_key = cfg.llm.api_key
    if not cfg.vision.model:
        cfg.vision.model = cfg.llm.model

    image_gen_raw = data.get("image_generation", {})
    image_gen_base_url = str(
        image_gen_raw.get("base_url", cfg.image_generation.base_url)
    ).strip().rstrip("/")
    image_gen_base_url_env = str(
        image_gen_raw.get("base_url_env", cfg.image_generation.base_url_env)
    ).strip()
    if (not image_gen_base_url) and image_gen_base_url_env:
        image_gen_base_url = str(os.getenv(image_gen_base_url_env, "")).strip().rstrip("/")
    image_gen_model = str(image_gen_raw.get("model", cfg.image_generation.model)).strip()
    image_gen_provider_default = cfg.image_generation.provider
    if "provider" not in image_gen_raw:
        legacy_hint = " ".join([image_gen_base_url, image_gen_model]).lower()
        if "siliconflow" in legacy_hint or "flux" in legacy_hint:
            image_gen_provider_default = "openai_compat"
    cfg.image_generation = ImageGenerationConfig(
        enabled=bool(image_gen_raw.get("enabled", cfg.image_generation.enabled)),
        provider=_load_image_generation_provider(
            image_gen_raw.get("provider", image_gen_provider_default),
            image_gen_provider_default,
        ),
        base_url=image_gen_base_url,
        base_url_env=image_gen_base_url_env,
        api_key=str(image_gen_raw.get("api_key", cfg.image_generation.api_key)),
        api_key_env=str(image_gen_raw.get("api_key_env", cfg.image_generation.api_key_env)),
        model=image_gen_model,
        timeout_sec=float(image_gen_raw.get("timeout_sec", cfg.image_generation.timeout_sec)),
        download_timeout_sec=float(
            image_gen_raw.get(
                "download_timeout_sec",
                cfg.image_generation.download_timeout_sec,
            )
        ),
        default_size=str(
            image_gen_raw.get("default_size", cfg.image_generation.default_size)
        ).strip(),
        output_dir=str(image_gen_raw.get("output_dir", cfg.image_generation.output_dir)).strip(),
    )
    if not cfg.image_generation.base_url:
        cfg.image_generation.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc"
    if not cfg.image_generation.model:
        cfg.image_generation.model = "z-image-turbo"
    if not cfg.image_generation.default_size:
        cfg.image_generation.default_size = "1024x1024"
    if not cfg.image_generation.output_dir:
        cfg.image_generation.output_dir = "data/generated_images"
    if cfg.image_generation.timeout_sec < 5.0:
        cfg.image_generation.timeout_sec = 5.0
    if cfg.image_generation.download_timeout_sec < 5.0:
        cfg.image_generation.download_timeout_sec = 5.0

    image_edit_raw = data.get("image_editing", {})
    if isinstance(image_edit_raw, dict):
        image_edit_base_url = str(
            image_edit_raw.get(
                "base_url",
                cfg.image_generation.base_url or cfg.image_editing.base_url,
            )
        ).strip().rstrip("/")
        image_edit_base_url_env = str(
            image_edit_raw.get(
                "base_url_env",
                cfg.image_generation.base_url_env or cfg.image_editing.base_url_env,
            )
        ).strip()
        if (not image_edit_base_url) and image_edit_base_url_env:
            image_edit_base_url = str(os.getenv(image_edit_base_url_env, "")).strip().rstrip("/")
        cfg.image_editing = ImageEditingConfig(
            enabled=bool(image_edit_raw.get("enabled", cfg.image_editing.enabled)),
            provider=str(
                image_edit_raw.get("provider", cfg.image_editing.provider)
            ).strip().lower() or "dashscope_qwen_image_edit",
            base_url=image_edit_base_url,
            base_url_env=image_edit_base_url_env,
            api_key=str(
                image_edit_raw.get(
                    "api_key",
                    cfg.image_generation.api_key or cfg.image_editing.api_key,
                )
            ),
            api_key_env=str(
                image_edit_raw.get(
                    "api_key_env",
                    cfg.image_generation.api_key_env or cfg.image_editing.api_key_env,
                )
            ),
            model=str(image_edit_raw.get("model", cfg.image_editing.model)).strip(),
            timeout_sec=float(image_edit_raw.get("timeout_sec", cfg.image_editing.timeout_sec)),
            download_timeout_sec=float(
                image_edit_raw.get(
                    "download_timeout_sec",
                    cfg.image_editing.download_timeout_sec,
                )
            ),
            default_size=str(
                image_edit_raw.get("default_size", cfg.image_editing.default_size)
            ).strip(),
            output_dir=str(
                image_edit_raw.get("output_dir", cfg.image_editing.output_dir)
            ).strip(),
            watermark=bool(image_edit_raw.get("watermark", cfg.image_editing.watermark)),
            prompt_extend=bool(
                image_edit_raw.get("prompt_extend", cfg.image_editing.prompt_extend)
            ),
            max_input_bytes=int(
                image_edit_raw.get("max_input_bytes", cfg.image_editing.max_input_bytes)
            ),
        )
    if not cfg.image_editing.base_url:
        cfg.image_editing.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc"
    if not cfg.image_editing.model:
        cfg.image_editing.model = "qwen-image-2.0-pro"
    if not cfg.image_editing.output_dir:
        cfg.image_editing.output_dir = "data/edited_images"
    if cfg.image_editing.timeout_sec < 5.0:
        cfg.image_editing.timeout_sec = 5.0
    if cfg.image_editing.download_timeout_sec < 5.0:
        cfg.image_editing.download_timeout_sec = 5.0
    if cfg.image_editing.max_input_bytes <= 0:
        cfg.image_editing.max_input_bytes = 10 * 1024 * 1024
    if cfg.send_after_paste_delay_sec < 0.0:
        cfg.send_after_paste_delay_sec = 0.0

    if cfg.heartbeat_interval_sec < 5.0:
        cfg.heartbeat_interval_sec = 5.0
    if cfg.heartbeat_min_idle_sec < 0.0:
        cfg.heartbeat_min_idle_sec = 0.0
    if cfg.heartbeat_max_actions < 0:
        cfg.heartbeat_max_actions = 0
    if cfg.heartbeat_max_people < 0:
        cfg.heartbeat_max_people = 0
    if cfg.heartbeat_people_history_records < 1:
        cfg.heartbeat_people_history_records = 1
    if cfg.memory_append_max_items < 20:
        cfg.memory_append_max_items = 20
    if cfg.impression_append_max_items < 20:
        cfg.impression_append_max_items = 20

    return cfg
