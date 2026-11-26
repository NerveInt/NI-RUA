# NI-RUA Agent 插件（`@nerveint/ni-rua-plugin`）

[English](README.md) · [简体中文](README.zh-CN.md)

将你现有的 **Claude Code**、**Codex** 或 **OpenCode** 会话直接作为 NI-RUA 机器人运行的执行智能体（Actor）。插件通过零摩擦的 **Stdio MCP 懒启动桥接器**（`ni-rua-plugin mcp`）自动透明代理已运行的本地 NI-RUA 网关，或在首次 `run_begin` 时自动拉起本地 SO-101 网关——**单窗口即可直接对话控制真机 SO-101，无需另开第二个终端**。插件同时内置三项技能（Skills）：

- `ni-rua-setup`：由智能体端到端引导安装、`doctor` 体检、`ni-rua-so101` 串口/相机绑定与调试、S0–S6 `/calibration` 标定及常见故障排查，就绪后无缝交接给 `ni-rua-robot`；
- `ni-rua-robot`：指导智能体如何开始（`run_begin`）、执行并结束（`run_finish`）机器人回合、安全停机及异常恢复；
- `ni-rua-experiments`：指导在 NI-RUA 源码仓库中工作的智能体遵循队列、报告与审计规范开展实验。

完整架构与生命周期说明见 [Agent 插件指南](https://github.com/NerveInt/NI-RUA/blob/main/docs/development/agent-plugin.zh-CN.md)。真机 SO-101 装配与标定见 [SO-101 快速上手指南](https://github.com/NerveInt/NI-RUA/blob/main/docs/guides/so101-quickstart.zh-CN.md) 与 [SO-101 标定指南](https://github.com/NerveInt/NI-RUA/blob/main/docs/guides/so101-calibration.zh-CN.md)。

## 快速开始（单窗口控制 SO-101）

环境要求：**Node >= 18**、macOS 或 Linux（Windows 仅支持 WSL），以及至少一个宿主 CLI（`claude`、`codex` 或 `opencode` 2.x）。

```sh
# 1. 一键初始化：安装宿主插件 + 隔离的 Python 3.12 网关与 SO-101 硬件运行环境（默认包含 SO-101）
npx -y @nerveint/ni-rua-plugin setup codex    # 或：claude | opencode | all

# 2. 一次性检测 SO-101 USB 串口与相机
ni-rua-so101 init-config

# 3. 直接打开 Codex / Claude Code / OpenCode —— 无需另起网关终端！
codex
# 对话示例：“检查 SO-101 机械臂状态并轻轻挥手。”
# 首次调用 `run_begin` 时，MCP 桥接器会自动拉起本地 SO-101 网关并在返回顶部附带 Web 控制台地址（http://127.0.0.1:8781/console）。
```

如果仅需连接远程/仿真网关、跳过本地 SO-101 硬件依赖：

```sh
npx -y @nerveint/ni-rua-plugin setup codex --no-so101
npx -y @nerveint/ni-rua-plugin doctor
```

### 手动启动网关与日常 CLI（`npm i -g`）

如果你已在 `http://127.0.0.1:8790/mcp`（或 `NI_RUA_MCP_URL`）手动启动了 RoboDojo、LIBERO、SO-101 或 `--dry-run` 网关，stdio 桥接器会自动探测并透明代理：

```sh
npm i -g @nerveint/ni-rua-plugin

# 零硬件 30 秒验证 Agent <-> MCP 全链路
ni-rua-plugin gateway --dry-run

# 手动启动纯本地 CPU SO-101 网关（冷启动缺标定文件时自动附加 --observation-profile rgb_proprio_v1）
ni-rua-plugin gateway --config so101

# 单独安装 / 卸载 / 体检
ni-rua-plugin install <claude|codex|opencode> [--scope user|project|local] [--port N]
ni-rua-plugin uninstall <claude|codex|opencode> [--scope user|project|local]
ni-rua-plugin doctor [--so101] [--json] [--gateway-url URL]
```

## 安装器与懒启动桥接机制

- **持久副本安装**：安装器先将本包复制到 `~/.local/share/ni-rua/agent-plugin/<version>/`，再将各宿主配置指向 `node <durable>/bin/cli.mjs mcp`（无需全局安装 `npm i -g`，也不受 `npx` 临时缓存清理影响）。所有配置改动记录于 `installed.json`，`uninstall` 精确回滚。
- **启动零开销**：`initialize` 立即响应，不拉起 Python 或触碰硬件；`tools/list` 快速探测上游 HTTP 网关（约 300ms），未启动时返回内置静态工具 Schema，零报错启动。
- **单例懒启动**：首次 `tools/call` 时若本地网关尚未运行（且未设 `NI_RUA_AUTO_START=0`），桥接器获取 `~/.local/share/ni-rua/gateway.lock` 单例锁，在后台启动本地网关（配置取自 `NI_RUA_GATEWAY_CONFIG`，默认 `so101`，日志位于 `~/.local/share/ni-rua/logs/gateway.log`），发送 `notifications/tools/list_changed`，并在 `run_begin` 返回顶部注入 `/console` 安全横幅。

| 选项 / 环境变量 | 含义 |
|---|---|
| `--no-so101` | `setup`：跳过创建隔离的 Python 3.12 SO-101 硬件环境（默认安装；`--so101` 保留为兼容别名）。 |
| `--skip-gateway` | `setup`：仅安装宿主插件，跳过 Python 运行环境创建。 |
| `--so101` | `doctor`：检查 USB 串口、Linux `dialout` 权限、`so101.local.json` 与相机标定。 |
| `--dry-run` | `gateway`：在 `http://127.0.0.1:8790/mcp` 启动零硬件内存模拟 MCP 服务。 |
| `--config <preset\|path>` | `gateway`：传给 `ni-rua` 的内置预设名（`so101`、`so101_standalone`、`ni-rua`）或配置文件路径。 |
| `--port N` | `install` / `doctor` / `mcp`：MCP 端口（默认 `8790`）；`gateway`：Web 控制台端口（`--dry-run` 下为 MCP 端口）。 |
| `--gateway-url URL` | `doctor` / `mcp`：指定上游 MCP HTTP URL。 |
| `NI_RUA_MCP_URL` | 覆盖上游网关 MCP URL（默认 `http://127.0.0.1:8790/mcp` 或安装时 `--port`）。 |
| `NI_RUA_AUTO_START=0` | 关闭 stdio 桥接器的懒启动功能（仅透传代理模式）。 |
| `NI_RUA_GATEWAY_CONFIG` | 懒启动使用的配置预设/路径（默认 `so101`）。 |
| `--json` | `doctor`：输出结构化 JSON 体检结果。 |
| `--scope user\|project\|local` | 仅适用于 Claude Code（`install` / `uninstall` / `setup`），默认 `user`。 |
| `--no-host-env` / `--no-allow` | Claude Code：跳过写入四个 MCP 环境变量或 `permissions.allow`。 |
| `--no-direct-catalog` | Codex：跳过向 `$CODEX_HOME/config.toml` 写入 `model_catalog_json`。 |
| `--force` | 强制替换同版本不同内容的持久副本，或在 `setup` 中重建托管 Python 环境。 |
| `--claude-bin`、`--codex-bin`、`--opencode-bin` | 指定宿主可执行文件路径。 |

## 各宿主配置说明

- **Claude Code**：注册 `ni_rua@ni_rua`、`.mcp.json`（`command: "node"`, `args: ["${CLAUDE_PLUGIN_ROOT}/bin/cli.mjs", "mcp"]`, `alwaysLoad: true`）、防压缩 `SessionStart`（`compact`）Hook、四个 MCP 环境变量及 `permissions.allow += ["mcp__plugin_ni_rua_ni_rua"]`。
- **Codex**：注册 `ni_rua@ni_rua`，安装副本中的 `.codex-mcp.json` 指向 `node <durable>/bin/cli.mjs mcp`（`default_tools_approval_mode: "approve"`、`tool_timeout_sec: 3600`、逐工具 `output_token_limit`），并在 `$CODEX_HOME/config.toml` 写入 `model_catalog_json = "<copy>/codex-models.direct.json"`，避免 Linux `code-mode host` 崩溃。
- **OpenCode（2.x）**：写入 `opencode.json` 的 `plugins`、`tool_output`、`media.image`，注册本地 stdio 桥接器（`type: "local"`, `command: ["node", "<durable>/bin/cli.mjs", "mcp"]`）与 `compaction` Hook；拒绝 1.x 版本。

## 安全提醒

在智能体中按 `Esc` 取消工具调用**不会停止真实机械臂**。紧急情况请立即使用 NI-RUA Web 控制台（`http://127.0.0.1:8781/console`）的 **Hold / E-STOP** 或物理 12V 断电开关。

## 源码维护

`.mcp.json`、`.codex-mcp.json` 与 `manifest.generated.json` 由工具注册表自动生成，严禁手工修改：

```sh
mamba run -n ni-rua python scripts/gen_agent_plugin_codex_mcp.py --check
npm --prefix agent-plugin test
agent-plugin/test/tarball-check.sh
```
