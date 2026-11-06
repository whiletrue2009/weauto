# WeAuto 运行时架构图

下面两张图都是 Mermaid 源码。相比聊天窗口里的默认渲染，这里把字号调大到 22px，并增加了节点间距。用 VS Code/GitHub/Typora 的 Mermaid 预览打开本文件即可放大查看。

## 1. 当前纯采集模式

当前本机配置为：

```toml
receiver_mode = "detached_windows"
reply_enabled = false
dry_run = true
agent_actions_enabled = false
detached_vision_parse_enabled = false

[llm]
enabled = false

[vision]
enabled = false
```

含义是：只做本机窗口捕获、本地 OCR、消息结构化、去重和本地历史保存；不进入回复决策、Agent 工具、LLM、Vision、桥接和 GUI 发送。

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontSize":"22px","primaryColor":"#eef6ff","primaryBorderColor":"#386fa4","secondaryColor":"#f7f7f7","tertiaryColor":"#fffbe8","lineColor":"#555555"},"flowchart":{"nodeSpacing":55,"rankSpacing":75,"curve":"basis"}}}%%
flowchart TB
    Start["./start_app.sh"] --> AppMain["app.main<br/>控制面板入口"]
    AppMain --> Supervisor["BotSupervisor<br/>子进程 / 日志 / 自动重启"]
    Supervisor --> Bot["run.py<br/>WeChatGuiRpaBot 子进程"]

    AppMain --> WebServer["本机 HTTP Server<br/>127.0.0.1"]
    WebServer --> WebUI["Web 控制台<br/>只读状态 / 日志 / 历史"]
    WebServer --> LocalData["本地 data 与 logs"]
    Supervisor --> LocalData

    subgraph Collector["纯采集主循环：本机处理"]
        direction TB
        Enum["Quartz 枚举窗口<br/>微信 owner + layer 0 + 标题"]
        Capture["按 window_id 捕获<br/>screencapture / Quartz"]
        OCR["RapidOCR ONNX<br/>PP-OCRv4 det + rec<br/>本机 CPU 推理"]
        Parse["VisibleMessageParser<br/>气泡 / 左右侧 / 发送者 / 文本"]
        Diff["VisibleMessageState<br/>增量去重"]
        History["ChatHistoryStore<br/>data/chat_history/*.txt"]
        Images["图片裁剪<br/>data/detached_window_images"]

        Enum --> Capture --> OCR --> Parse --> Diff
        Diff --> History
        Parse --> Images
    end

    Bot --> Collector
    History --> LocalData
    Images --> LocalData

    Diff --> Cutoff["reply_enabled = false<br/>回复队列入口关闭"]
    Cutoff --> Stop["不调用 LLM / Vision / 工具<br/>不点击微信 / 不发送消息"]
    style Cutoff fill:#ffe8e8,stroke:#c0392b,stroke-width:3px
    style Stop fill:#f4f4f4,stroke:#777777,stroke-dasharray: 6 4
```

## 2. WeAuto 完整能力架构

这张图保留项目原本的完整处理链路，方便以后重新打开回复能力时对照。当前配置中，红色虚线右侧的外呼和发送链路不会执行。

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontSize":"22px","primaryColor":"#eef6ff","primaryBorderColor":"#386fa4","secondaryColor":"#f7f7f7","tertiaryColor":"#fffbe8","lineColor":"#555555"},"flowchart":{"nodeSpacing":55,"rankSpacing":75,"curve":"basis"}}}%%
flowchart TB
    subgraph Runtime["运行与控制"]
        Launcher["启动器<br/>start_app.sh / WeAuto.app"]
        Panel["app.main<br/>菜单栏 + Web 控制台"]
        Supervisor["BotSupervisor"]
        Bot["WeChatGuiRpaBot<br/>detached receiver 主循环"]
        Launcher --> Panel --> Supervisor --> Bot
    end

    subgraph Receive["接收与识别"]
        Windows["macOS/Windows 窗口系统"]
        Capture["窗口捕获"]
        LocalOCR["本地 RapidOCR ONNX"]
        OptionalVision["可选 Vision LLM<br/>当前关闭"]
        Parser["可见消息解析"]
        State["增量状态与去重"]
        Windows --> Capture --> LocalOCR --> Parser --> State
        Capture -. 可选 .-> OptionalVision --> Parser
    end

    subgraph Store["本机数据"]
        Chat["聊天文本历史<br/>JSON 索引 + 日期 txt"]
        Media["图片/文件裁剪"]
        Memory["记忆 / 技能 / 人物索引"]
        RuntimeState["runtime_state.json"]
        Logs["运行日志"]
        State --> Chat
        Parser --> Media
        Bot --> Memory
        Bot --> RuntimeState
        Supervisor --> Logs
    end

    subgraph Processing["回复处理：当前被 reply_enabled=false 关闭"]
        Policy["事件策略<br/>私聊 / 群聊 / @ / 冷却"]
        Decision["LLM 决策"]
        Planner["LLM Planner"]
        Tools["Agent 工具<br/>浏览 / 搜索 / Python / 记忆"]
        Bridge["HTTP / OpenClaw / Long Bridge"]
        ReplyGen["回复生成"]
        Policy --> Decision --> Planner --> Tools
        Planner --> Bridge
        Tools --> ReplyGen
    end

    State --> Policy
    ReplyGen --> Sender["GUI Sender<br/>激活窗口 / 粘贴 / 点击发送"]
    Sender --> WeChat["微信客户端"]

    subgraph External["外部服务：当前不会访问"]
        LLM["OpenAI-compatible LLM<br/>例如智谱 GLM"]
        Search["搜索 / 天气 / 图片服务"]
        BridgeRemote["远端 bridge 服务"]
        Decision -.-> LLM
        Planner -.-> LLM
        ReplyGen -.-> LLM
        Tools -.-> Search
        Bridge -.-> BridgeRemote
    end

    style Policy fill:#ffe8e8,stroke:#c0392b,stroke-width:3px
    style Sender fill:#ffe8e8,stroke:#c0392b,stroke-width:3px
    style LLM fill:#fff0f0,stroke:#c0392b,stroke-dasharray: 6 4
    style Search fill:#fff0f0,stroke:#c0392b,stroke-dasharray: 6 4
    style BridgeRemote fill:#fff0f0,stroke:#c0392b,stroke-dasharray: 6 4
```

## 3. 数据和隐私边界

| 类别 | 当前状态 | 数据去向 |
|---|---|---|
| 窗口枚举/截图 | 开启 | 本机内存与本地文件 |
| RapidOCR OCR | 开启 | 本机 ONNX Runtime CPU 推理 |
| 聊天文本历史 | 开启 | `data/chat_history/` |
| 图片裁剪 | 开启 | `data/detached_window_images/` |
| LLM / Vision | 关闭 | 不访问外部服务 |
| Agent 浏览/搜索/Python 工具 | 关闭 | 不执行 |
| GUI 回复发送 | 关闭 | 不操作微信输入框 |
| Web 控制台 | 开启 | 只绑定本机地址，Web API 只读 |

注意：聊天历史里的时间是 WeAuto 观察到消息的时间，不等于微信消息的原始发送时间。当前 GUI 截图方案无法稳定拿到微信内部消息 ID 和原始时间戳。
