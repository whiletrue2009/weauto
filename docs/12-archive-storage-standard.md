# 消息归档存储标准（P1 草稿库）

本文档是各会话共同遵守的结构化存储规范。第一阶段只做"数据搜集"：把 Bot 实时捕获的
可见消息作为**草稿**追加写入本地 SQLite。清洗（去重、发送者归一、时间推断）是后续阶段，
不在本阶段承诺范围内。

背景与完整方案讨论见
[me-信息汇聚方案.md](me-信息汇聚方案.md)。

## 1. 阶段定位

| 阶段 | 内容 | 状态 |
|---|---|---|
| P1 数据搜集 | 草稿库 + CAS 附件 + HTML 复查导出 | 本文档范围 |
| P2 清洗 | 去重、发送者归一/重名区分、时间推断，migration v2 引入 participants/messages | 数据攒够后启动 |
| P3 历史回补 | 滚动采集可见历史，只调用 `Recorder.record_snapshot(source="backfill")` | 独立会话实现 |
| P4 媒体获取 | 原图/文件下载、下载目录监听 | 后续 |

P1 的写入原则：

1. **只追加，不修改，不删除**（append-only）。每轮截图产生的每条可见消息都是一条草稿。
2. 人为滚动或历史回补导致同一消息多次入镜 → **预期行为**，重复草稿全部保留，留给
   清洗阶段处理。
3. `fingerprint` 照常计算并存储，但**不做唯一约束、不参与写入判断**，只是清洗原料。
4. OCR 原文和置信度全量保留，任何后续修正都可追溯到原始观测。

## 2. 存储介质与目录约定

```text
data/archive/
  weauto.sqlite          主库（WAL 模式）
  blobs/sha256/ab/ab12…  附件内容寻址存储（CAS），图片裁剪图等
  review/                HTML 复查导出（index.html + session_*.html + thumbs/）
  inbox/                 未来的下载暂存区（P4 之前不使用）
```

硬约定：

- SQLite + WAL；连接必须开启 `foreign_keys=ON`、`busy_timeout=5000`
- 所有 `*_at` 时间字段为 **INTEGER Unix 秒（UTC）**，禁止毫秒或本地时间混入
- 所有 ID 形如 `<前缀>_<16位hex>`（uuid4 随机生成，非内容派生）：
  `sess_`、`cap_`、`dft_`、`blo` 不适用（blob 主键即 sha256）

## 3. Schema 版本管理

```sql
CREATE TABLE schema_migrations (
  version INTEGER PRIMARY KEY,
  applied_at INTEGER NOT NULL
);
```

- 建库即写 migration v1（全量建表），之后任何表结构变更必须追加 v2、v3…
- 禁止直接改表不留记录
- **其他会话不允许绕过 ArchiveStore/ArchiveRecorder 接口直接写 SQL**（migration 除外），
  保证所有写入走同一套归一化逻辑

## 4. 私聊与群聊建模

统一以「会话」为锚点，差异收敛在 `sessions.kind` 与字段语义：

| | 私聊（kind=direct） | 群聊（kind=group） |
|---|---|---|
| 会话标题 | 对方昵称 | 群名 |
| 发送者 | `side='other'` 时 sender 即对方 | `side='other'` 时 sender = 群成员昵称 |
| 目标对象 | 隐含为我，无需字段 | 默认全群，缺失时不臆造 |
| 目标线索 | 不需要 | `mentions_json`（@提及）+ `quoted_text`（可见引用片段） |

群消息目标对象分三层，P1 只承诺前两层：

1. 显式 @ 提及 → `mentions_json`（可靠）
2. 引用回复 → `quoted_text`（OCR 可见才存，否则 NULL）
3. 语义指向 → 不在 P1 范围，P2 之后由分类写入预留字段

**群消息目标对象缺失是合法状态，用 NULL 表示，不用猜测值填充。**

## 5. 表结构（migration v1）

```sql
CREATE TABLE sessions (
  session_id    TEXT PRIMARY KEY,
  stable_key    TEXT NOT NULL UNIQUE,   -- 规范化窗口标题，跨重启锚点
  title         TEXT NOT NULL,
  kind          TEXT NOT NULL CHECK(kind IN ('direct','group','unknown')),
  first_seen_at INTEGER,
  last_seen_at  INTEGER,
  created_at    INTEGER NOT NULL,
  updated_at    INTEGER NOT NULL
);

CREATE TABLE captures (
  capture_id      TEXT PRIMARY KEY,
  session_id      TEXT NOT NULL,
  window_id       INTEGER NOT NULL,
  captured_at     INTEGER NOT NULL,
  source          TEXT NOT NULL,        -- 'realtime' | 'backfill'
  parser_source   TEXT,                 -- 'ocr' | 'vision'
  screenshot_path TEXT,
  FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

CREATE TABLE message_drafts (
  draft_id       TEXT PRIMARY KEY,
  session_id     TEXT NOT NULL,
  capture_id     TEXT NOT NULL,
  window_id      INTEGER NOT NULL,
  source         TEXT NOT NULL,
  observed_at    INTEGER NOT NULL,
  ui_time_text   TEXT,
  side           TEXT NOT NULL CHECK(side IN ('self','other','unknown')),
  sender_name    TEXT,                  -- 原样字符串，不归一化、不合并
  content_kind   TEXT NOT NULL CHECK(content_kind IN
                   ('text','image','file','video','voice','link','sticker','system','unknown')),
  text           TEXT,
  mentions_json  TEXT,
  quoted_text    TEXT,
  bbox           TEXT,
  fingerprint    TEXT,                  -- 仅存备用，清洗阶段使用
  parser_source  TEXT,
  ocr_confidence REAL,
  confidence     REAL,
  blob_sha256    TEXT,                  -- 图片裁剪图 → blobs
  raw_json       TEXT NOT NULL,
  created_at     INTEGER NOT NULL,
  FOREIGN KEY(capture_id) REFERENCES captures(capture_id)
);
CREATE INDEX idx_drafts_session_obs ON message_drafts(session_id, observed_at);

CREATE TABLE blobs (
  sha256        TEXT PRIMARY KEY,
  size_bytes    INTEGER NOT NULL,
  mime_type     TEXT,
  first_seen_at INTEGER NOT NULL
);
```

P1 明确**不含** `participants`、`session_participants`、`messages`、`attachments`；
发送者关联、重名区分、附件元数据属于 P2 清洗工作，届时以 migration 引入。

## 6. raw_json 规范

`raw_json` 固定为规范化后的解析器输出，空值保留键不省略：

```json
{
  "side": "other",
  "sender": "刮风6-602",
  "content_type": "text",
  "text": "内高2.1的车不知道行不行",
  "bbox": [120, 340, 80, 46],
  "mentions": [],
  "parser_source": "ocr",
  "image_hash": "",
  "image_path": ""
}
```

## 7. 代码接线与接口契约

```text
detached window 截图
  → OCR/Vision snapshot（现有）
  → VisibleMessageStateStore 增量检测（现有）
  → ArchiveRecorder.record_snapshot(...)      ← 新增，唯一写入入口
       ├─ captures + message_drafts（每轮必写）
       └─ 图片裁剪图 → blobs CAS（sha256）
  → 现有 reply/memory 流程（不受影响，Markdown 双写保留）
```

冻结的接口签名（P3 回补会话依赖）：

```python
ArchiveStore(db_path: str | Path, blob_dir: str | Path)
    # 自动建库 + migration + WAL pragmas

ArchiveRecorder(store).record_snapshot(
    session_title: str,
    messages: list[VisibleMessage],    # 现有 dataclass，直接复用
    window_id: int,
    source: str,                       # 'realtime' | 'backfill'
    captured_at: int | None = None,
    parser_source: str = "ocr",
) -> None
```

Bot 接线要求：`record_snapshot` 调用必须 try/except 包裹，归档失败**不得阻断**
回复主链路；Markdown 历史（`ChatHistoryStore`）继续双写保留。

会话分工：

| 模块 | 归属 |
|---|---|
| `wechat_rpa/archive_store.py` / `archive_recorder.py` / bot 接线 / HTML 导出 | P1 会话（结构化存储） |
| 历史回补滚动采集 | P3 会话，只调用 `record_snapshot(source="backfill")`，不自带存储逻辑 |
| `[archive]` 配置段 | P1 会话维护 `config.toml.example` |

## 8. HTML 复查视图

形态：**静态 HTML 导出**，不做 Web 服务（避免新的网络暴露面）。

```text
scripts/export_archive_review.py
  → 读取 data/archive/weauto.sqlite
  → data/archive/review/index.html            会话列表 + 汇总统计
  → data/archive/review/session_<key>.html    每会话时间线
  → data/archive/review/thumbs/               图片缩略图（相对路径引用）
```

每会话页内容：

- 按时间排序气泡时间线：self 靠右 / other 靠左，sender + 双时间
  （`observed_at` 精确时间 + 界面 `HH:MM`）
- 内容类型徽标（文本/图片/文件/未知），图片内联缩略图
- 检查信息：`capture_id`、`source`、`fingerprint` 短哈希、confidence
- 同一 capture 的草稿按颜色分组，肉眼可辨滚动重叠造成的重复
- 页内原生 JS 过滤器：发送者、类型、时间范围、关键词（单文件无依赖）

归档运行期间或之后可随时重跑导出，生成最新复查页面。

## 9. 配置约定

```toml
[archive]
enabled = false                       # 默认关闭，显式开启
path = "data/archive/weauto.sqlite"
blob_dir = "data/archive/blobs"
```

## 10. P1 验收清单

- [ ] 新消息草稿持续入库（text/image 至少两类）
- [ ] 重复观测全部保留，不丢弃、不合并
- [ ] 图片裁剪图写入 blobs 且有 sha256
- [ ] Bot 崩溃重启后已提交事务不丢失（WAL）
- [ ] 归档失败不影响自动回复主链路
- [ ] HTML 复查页能还原时间线，暴露重复/错位/OCR 错字
- [ ] migration 幂等：重复初始化不破坏已有数据
