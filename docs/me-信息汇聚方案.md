# WeAuto 信息汇聚方案

## 1. 目标与边界

目标是在 WeAuto 现有 GUI RPA 能力的基础上，把私聊和微信群中可见的消息整理到本机轻量数据库中，并逐步归档图片、文件等附件。

需要先明确技术边界：WeAuto 当前方案不 Hook、不注入微信进程、不读取微信数据库，只能获取“可见界面上的信息”。因此：

- OCR 可能识别错字或漏识别。
- 界面上可能只显示 `HH:MM`，历史消息日期需要推断。
- 文件卡片可能只有文件名和大小，不一定能直接拿到文件本体。
- 图片可以先保存气泡裁剪图，但不一定自动获得原图。
- 若目标是“完整、准确、含原始文件”，需要读取或解密微信本地存储，或复用类似 WeFlow 的数据访问层；这将改变 WeAuto 当前技术边界。

建议先实现“可见信息归档”，再逐步增强媒体获取，避免第一版就依赖高风险 UI 自动点击。

## 2. 能力分层

方案拆成三层能力：

1. **实时增量归档**：被动监听独立微信窗口，OCR / Vision 解析后入库，基本不干扰聊天操作。
2. **历史回补**：自动滚动会话窗口，把可见历史逐屏解析入库，必然干扰前台焦点和滚动，需要显式授权的“独占归档模式”。
3. **媒体归档**：图片至少能保存可见气泡截图；文件和原始图片需要点击、预览、另存为或读取微信本地数据，难度和干扰完全不同。

## 3. 技术选型

推荐主存储使用 **SQLite + WAL 模式**，附件使用 **文件系统内容寻址存储**，数据库只存附件元数据。

| 方案 | 优点 | 问题 | 结论 |
|---|---|---|---|
| SQLite | 单文件、零服务、跨平台、Python 标准库支持、事务可靠、适合单机写入 | 不适合把所有 BLOB 全塞进去；中文 FTS 需要额外处理 | 推荐 |
| 文件系统附件库 | 天然适合图片和文件、易去重、易备份、可被外部工具查看 | 需要自行管理一致性和孤儿文件 | 推荐，与 SQLite 配合 |
| SQLite 直接存 BLOB | 备份最简单 | 数据库膨胀、VACUUM 慢、部分读取不方便 | 只适合小缩略图 |
| DuckDB | 分析能力强 | 更适合分析型查询，不适合常驻 App 主 OLTP 库 | 可作为后续导出分析层 |
| PostgreSQL | 功能强 | 需要服务端运维，单机轻量场景过重 | 不推荐 |
| SQLCipher | 可加密整库 | 增加依赖、迁移和备份复杂度 | 可选增强，不作为第一步 |

建议目录：

```text
data/
  archive/
    weauto.sqlite
    blobs/
      sha256/
        ab/
        cd/
        ...
    thumbnails/
    inbox/
```

附件按 SHA-256 存储，例如：

```text
data/archive/blobs/sha256/ab/abcdef123456...
```

数据库中保存：

```text
sha256、size、mime、original_filename、storage_path、status
```

这样重复图片或文件可以自然去重，后续也更容易迁移到移动硬盘、NAS 或对象存储。

## 4. 数据表设计

不要把 OCR 解析结果直接当作最终事实，应分成“观测”和“归一化消息”两层。GUI 解析存在重复、OCR 漂移、同文本多次出现、滚动重叠等问题，两层结构便于后续修正和合并。

核心表设计：

```sql
CREATE TABLE sessions (
  session_id TEXT PRIMARY KEY,
  stable_key TEXT NOT NULL UNIQUE,
  title TEXT NOT NULL,
  kind TEXT NOT NULL CHECK(kind IN ('direct', 'group', 'unknown')),
  first_seen_at INTEGER,
  last_seen_at INTEGER,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE participants (
  participant_id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  normalized_name TEXT NOT NULL,
  alias_of TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE session_participants (
  session_id TEXT NOT NULL,
  participant_id TEXT NOT NULL,
  display_name_in_session TEXT,
  first_seen_at INTEGER,
  last_seen_at INTEGER,
  PRIMARY KEY (session_id, participant_id),
  FOREIGN KEY(session_id) REFERENCES sessions(session_id),
  FOREIGN KEY(participant_id) REFERENCES participants(participant_id)
);

CREATE TABLE captures (
  capture_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  window_id INTEGER NOT NULL,
  captured_at INTEGER NOT NULL,
  source TEXT NOT NULL,
  parser_config_hash TEXT,
  window_image_hash TEXT,
  screenshot_path TEXT,
  FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

CREATE TABLE messages (
  message_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  participant_id TEXT,
  direction TEXT NOT NULL CHECK(direction IN ('self', 'other', 'unknown')),
  content_kind TEXT NOT NULL,
  text TEXT,
  observed_at INTEGER NOT NULL,
  ui_time_text TEXT,
  inferred_timestamp INTEGER,
  timestamp_confidence REAL NOT NULL DEFAULT 0,
  source_fingerprint TEXT NOT NULL,
  dedupe_key TEXT NOT NULL,
  sequence_no INTEGER,
  confidence REAL NOT NULL DEFAULT 0,
  raw_json TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  UNIQUE(session_id, dedupe_key)
);

CREATE TABLE message_observations (
  observation_id TEXT PRIMARY KEY,
  message_id TEXT,
  capture_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  observed_text TEXT,
  sender_name TEXT,
  content_kind TEXT,
  bbox TEXT,
  fingerprint TEXT,
  parser_source TEXT,
  ocr_confidence REAL,
  created_at INTEGER NOT NULL,
  FOREIGN KEY(capture_id) REFERENCES captures(capture_id),
  FOREIGN KEY(message_id) REFERENCES messages(message_id)
);

CREATE TABLE attachments (
  attachment_id TEXT PRIMARY KEY,
  message_id TEXT NOT NULL,
  kind TEXT NOT NULL CHECK(kind IN ('image', 'file', 'video', 'voice', 'unknown')),
  status TEXT NOT NULL CHECK(status IN (
    'visible_crop',
    'downloaded_original',
    'external_download',
    'unavailable',
    'pending',
    'failed'
  )),
  blob_sha256 TEXT,
  storage_path TEXT,
  thumbnail_path TEXT,
  original_filename TEXT,
  mime_type TEXT,
  size_bytes INTEGER,
  metadata_json TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  FOREIGN KEY(message_id) REFERENCES messages(message_id)
);

CREATE TABLE archive_jobs (
  job_id TEXT PRIMARY KEY,
  job_type TEXT NOT NULL,
  session_id TEXT,
  status TEXT NOT NULL,
  started_at INTEGER,
  finished_at INTEGER,
  cursor_json TEXT,
  error TEXT,
  stats_json TEXT
);
```

如果 SQLite 构建支持 FTS5，可以增加全文索引：

```sql
CREATE VIRTUAL TABLE message_fts USING fts5(
  message_id UNINDEXED,
  text,
  tokenize='trigram'
);
```

中文搜索建议保留 `LIKE` 兜底，因为 trigram 对两字词、错别字和 OCR 漂移仍有限制。搜索需求变强后再考虑 jieba 分词或专用搜索索引。

## 5. 与现有代码的接入点

现有链路已经具备大部分基础：

- 独立窗口捕获：`wechat_rpa/detached_window_receiver.py`
- OCR / Vision 消息结构：`wechat_rpa/visible_message_parser.py`
- 增量状态：`wechat_rpa/visible_message_state.py`
- Bot 主循环：`wechat_rpa/bot.py`
- 现有文本历史：`wechat_rpa/agent_store.py`

建议新增：

```text
wechat_rpa/archive_store.py
wechat_rpa/archive_recorder.py
```

`ArchiveStore` 负责：

1. 初始化 SQLite schema。
2. 开启 WAL、foreign_keys、busy_timeout。
3. 写入 sessions、captures、observations、messages、attachments。
4. 维护 FTS 索引。
5. 提供统计和查询 API。
6. 处理内容寻址附件写入。

`ArchiveRecorder` 只依赖 Bot 传入的 `VisibleChatSnapshot`，不直接理解 GUI 细节。这样实时监听、历史回补和测试 fixture 都能复用同一套入库逻辑。

实时链路：

```text
detached window
  -> capture
  -> OCR / Vision snapshot
  -> VisibleMessageStateStore 增量检测
  -> ArchiveRecorder
       -> message_observations
       -> canonical messages
       -> attachment metadata
  -> 原有 reply / bridge / memory 流程
```

第一阶段建议“双写”：继续保留现有 `ChatHistoryStore` Markdown，保证 Agent 现有能力不受影响；SQLite 作为新增归档层。稳定后再考虑让 Agent 工具优先查询 SQLite。

## 6. 图片、文件和媒体处理

### 图片

当前解析器已经能识别图片气泡，并可以保存裁剪图：

- `content_type = "image"`
- `image_hash`
- `image_path`

这类图片先入库为：

```text
status = visible_crop
```

它表示“从聊天窗口截图裁剪出来的可见图片”，不能冒充原图。

若要获取原图，需要新增媒体获取模式：

```text
定位图片气泡
  -> 点击图片
  -> 等待预览窗口
  -> 调用另存为或保存按钮
  -> 写入 data/archive/inbox
  -> 计算 sha256
  -> 移入 blobs
  -> 更新 attachment status = downloaded_original
```

这会抢占鼠标、键盘、剪贴板和前台窗口，必须放在授权独占模式中。

### 文件

GUI 中文件通常表现为文件卡片，OCR 可能只能读到文件名、大小等信息。第一阶段保存为：

```text
content_kind = file
attachment status = pending 或 unavailable
text = OCR 得到的文件名和大小描述
```

第二阶段再做自动下载：

```text
点击文件卡片
  -> 等待下载完成或另存为窗口
  -> 监听 data/archive/inbox
  -> sha256 入库
```

折中方案是监听微信下载目录：文件下载进入指定目录后，后台计算 hash，并根据时间、文件名、会话上下文尝试关联消息。这个方案比自动点击稳定，但需要人工或半自动触发下载。

### 视频、语音

难度更高。GUI 上通常只能看到卡片或时长。没有本地数据访问时，自动获取原始媒体需要专用 UI 流程且不稳定。第一阶段只保存卡片信息和截图证据。

## 7. 历史回补设计

历史回补必须单独做模式，不能混在普通监听中。

建议新增：

```text
processing_mode = "archive"
archive_only = true
```

或命令行模式：

```bash
python run.py --mode archive --backfill --session "会话名"
```

独占归档流程：

```text
用户确认授权
  -> 暂停普通自动回复
  -> 禁用 LLM 发送
  -> 激活目标独立窗口
  -> 标准化窗口尺寸
  -> 记录初始底部状态
  -> 循环：
       1. 截图
       2. OCR / Vision 解析
       3. 写 capture + observations
       4. 合并 canonical messages
       5. 向上滚动一屏
       6. 等待聊天内容稳定
       7. 检测是否无新增、达到日期或达到页数上限
  -> 可选逐个处理媒体卡片
  -> 回滚到底部
  -> 恢复普通监听
```

配置示例：

```toml
[archive]
enabled = true
path = "data/archive/weauto.sqlite"
blob_dir = "data/archive/blobs"
store_window_screenshots = false
store_message_crops = true
create_thumbnails = true
write_mode = "dual"
fts_enabled = true

[archive.realtime]
record_existing_visible_messages_on_start = false
record_self_messages = true
record_incoming_messages = true

[archive.backfill]
enabled = false
exclusive_mode = true
max_pages_per_run = 500
scroll_wait_sec = 0.8
fetch_media = false
stop_on_user_input = true
```

历史回补需要激活窗口、改变前台焦点、滚动聊天窗口、可能点击图片或文件、可能使用保存对话框、可能占用剪贴板，因此不能作为后台常驻任务，应做成显式授权的一次性 job。

安全机制：

1. 显式确认：Web UI 或 CLI 要求输入确认词，例如 `ARCHIVE`。
2. 只归档不发送：archive mode 强制跳过所有回复。
3. 用户输入检测：检测鼠标移动、键盘输入或窗口切换后暂停任务。
4. 紧急停止：全局快捷键或控制面板停止按钮。
5. 断点恢复：`archive_jobs.cursor_json` 保存窗口、滚动页、锚点 fingerprint、媒体处理队列。

## 8. 去重与顺序

GUI 归档不能只依赖现有 fingerprint。文本 fingerprint 通常由方向、类型、发送者和归一化文本组成，存在以下问题：

- 同一个人发两次“好的”会撞 fingerprint。
- OCR 把“你好”识别成“你妤”会导致重复。
- 滚动重叠时同一消息出现在多屏截图里。
- 群昵称变更会导致 sender 不一致。
- GUI 不一定提供稳定消息 ID。

建议：

1. 每次截图先写 `message_observations`，作为不可变观测事实。
2. 后台归一化任务再合并为 `messages`。
3. 合并依据包括 session、direction、sender normalized、text similarity、bbox 顺序、前后消息上下文、capture overlap、ui_time_text。
4. 完全相同文本不强行覆盖，允许同 fingerprint 多条 canonical message。
5. 每条消息保存 confidence 和 raw_json，不丢弃 OCR 原始结果。
6. 时间字段区分：
   - `observed_at`：WeAuto 看到消息的时间，可靠。
   - `ui_time_text`：界面显示的时间文本，可能缺失。
   - `inferred_timestamp`：推断出的历史时间，不一定准确。

实时消息的 `observed_at` 可靠；历史回补中的历史时间只能根据日期分隔、滚动位置和 `HH:MM` 推断，不能当作微信原始时间。

## 9. 群聊身份

GUI 能稳定拿到的是“当前显示昵称”，不一定有稳定微信 ID。建议表结构保留：

```text
participants.display_name
participants.normalized_name
participants.alias_of
```

并结合现有：

```text
data/config/PEOPLE_ALIASES.md
```

先在会话内建立稳定身份：

```text
participant_key = hash(session_id + normalized_display_name)
```

跨群是否为同一个人，需要人工别名或更底层数据源。不要自动把不同群里同名的人合并为同一 participant。

## 10. 实施阶段

### 第一阶段：实时消息入 SQLite

范围：

- 新增 `ArchiveStore`。
- 新增 `ArchiveRecorder`。
- Bot 每次拿到 snapshot 后写库。
- 文本、图片裁剪图、未知卡片先保存观测数据。
- 继续保留原 `ChatHistoryStore`。

验收：

- 新消息能入库。
- 重复截图不重复生成 canonical message。
- 图片裁剪图有 sha256 和路径。
- Bot 崩溃重启后不丢已提交事务。
- 不影响现有自动回复。

### 第二阶段：查询与 Agent 工具

新增工具：

```text
search_archive
read_archive_messages
summarize_archive_session
list_archive_attachments
```

Agent 可从 SQLite 查询更完整历史，而不必扫描 Markdown。

### 第三阶段：历史回补

实现独占模式和滚动采集。

验收重点：

- 可断点恢复。
- 可停止。
- 不自动回复。
- 能记录每页 capture。
- 能合并重叠区域。
- 明确标记低置信度和缺失时间。

### 第四阶段：媒体获取

优先顺序：

1. 监听微信下载目录，导入已下载文件。
2. 自动点击图片并保存原图。
3. 自动点击文件卡片并下载。
4. 视频、语音最后做。

不建议第一版自动点击所有媒体，误点、下载失败、窗口弹窗变化都会降低稳定性。

## 11. 数据安全

聊天记录和附件高度敏感，第一版应做到：

```text
data/archive/ 权限设置为当前用户可读写
Web UI 不提供附件下载 API
Web UI 默认只绑定 127.0.0.1
数据库和 blob 目录纳入备份
保留 archive_jobs 和 captures 审计记录
```

如果本机磁盘没有 FileVault 或 BitLocker，SQLite 本身是明文。加密可选：

- 依赖系统全盘加密，简单可靠。
- 引入 SQLCipher，复杂但单库加密。

建议先用系统盘加密和目录权限，归档链路稳定后再考虑 SQLCipher。

## 12. 推荐落地路径

第一版：

```text
SQLite WAL + 文件系统 CAS 附件库
实时归档独立窗口里的可见消息
图片先保存气泡裁剪图
文件先保存文件名、大小、卡片截图和 pending 状态
继续双写现有 Markdown 历史
```

第二版：

```text
授权独占历史回补
滚动采集并合并重复观测
支持断点和紧急停止
```

第三版：

```text
媒体自动获取或下载目录监听
原图和文件以 sha256 入 CAS
再考虑 FTS 分词和更完整分析
```

该方案符合 WeAuto 当前边界，能先稳定拿到大量有价值数据，同时不会一开始就把系统绑死在高风险 UI 自动点击上。若完整原始媒体最终是硬需求，应单独评估读取微信本地存储的方案，而不是让 GUI RPA 假装拥有数据库能力。
