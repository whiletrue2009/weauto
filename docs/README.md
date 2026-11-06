# WeAuto 文档

这套文档依据当前仓库的代码、配置、启动脚本、控制面板、插件和测试重新编写，不继承旧版
文档结构。`config.toml.example` 与代码始终是配置行为的最终依据。

## 阅读路径

### 第一次使用

1. [系统总览](01-system-overview.md)
2. [安装与首次运行](02-installation-and-first-run.md)
3. [配置指南](03-configuration.md)
4. [运行方式与控制面板](04-running-and-control-panel.md)

### 理解功能

1. [消息接收与回复流程](05-message-processing.md)
2. [Agent、记忆、人物与技能](06-agent-memory-and-tools.md)
3. [桥接与外部集成](07-bridges-and-integrations.md)
4. [消息归档存储标准](12-archive-storage-standard.md)

### 运维与开发

1. [数据、日志与维护](08-data-and-maintenance.md)
2. [架构与代码地图](09-architecture-and-code-map.md)
3. [开发与测试](10-development-and-testing.md)
4. [故障排查](11-troubleshooting.md)

## 文档边界

- 当前推荐接收模式是 `detached_windows`。
- 当前推荐日常入口是 macOS 的 `WeAuto.app` 或 Windows 的 `WeAuto.vbs`。
- Web 控制台是只读界面；Bot 启停操作在菜单栏 App 中。
- `legacy_list`、恢复脚本和旧数据迁移能力仍存在，但不会与推荐路径混写。
- 密钥、个人聊天、人物印象和私有配置不应提交到公共仓库。

## 快速命令

```bash
cp config.toml.example config.toml
./start_app.sh
```

Windows 对应命令是 `Copy-Item config.toml.example config.toml` 和 `.\start_app.ps1`。

确认 `dry_run = true` 时识别正常，再启用实际发送。首次环境初始化完成后，可以直接双击
`WeAuto.app`。
