# NI-RUA Agent 插件（`@nerveint/ni-rua-plugin`）

[English](README.md) · [简体中文](README.zh-CN.md)

将你现有的 **Claude Code**、**Codex** 或 **OpenCode** 会话直接作为 NI-RUA 机器人运行的执行智能体（Actor）。插件通过 MCP 连接本地 NI-RUA 网关，并附带两项技能（Skills）：

- `ni-rua-robot`：指导智能体如何开始（`run_begin`）、执行并结束（`run_finish`）机器人回合、如何安全停机以及网关不可达时的恢复步骤；
- `ni-rua-experiments`：指导在 NI-RUA 源码仓库中工作的智能体遵循队列、报告与审计规范开展实验。

完整架构与生命周期说明见 [Agent 插件指南](https://github.com/NerveInt/NI-RUA/blob/main/docs/development/agent-plugin.zh-CN.md)。真机 SO-101 装配与标定见 [SO-101 快速上手指南](https://github.com/NerveInt/NI-RUA/blob/main/docs/guides/so101-quickstart.zh-CN.md) 与 [SO-101 标定指南](https://github.com/NerveInt/NI-RUA/blob/main/docs/guides/so101-calibration.zh-CN.md)。

## 快速开始

环境要求：**Node >= 18**、macOS 或 Linux（Windows 仅支持 WSL），以及至少一个宿主 CLI（`claude`、`codex` 或 `opencode` 2.x）。

### 一键初始化（`npx`）

```sh
# 安装宿主插件 + 隔离的 Python 3.12 NI-RUA 网关环境（~/.local/share/ni-rua/envs/gateway）
npx -y @nerveint/ni-rua-plugin setup claude    # 或：codex | opencode | all

# 附加 --so101 同时创建隔离的 SO-101 硬件运行环境（~/.local/share/ni-rua/envs/so101）
npx -y @nerveint/ni-rua-plugin setup claude --so101

# 全栈体检（宿主插件、Python/IK 栈、网关连通性）
npx -y @nerveint/ni-rua-plugin doctor          # 附加 --so101 同步检查串口/标定配置
```

### 日常使用（`npm i -g`）

```sh
npm i -g @nerveint/ni-rua-plugin

# 1. 零硬件 30 秒验证 Agent <-> MCP 全链路
ni-rua-plugin gateway --dry-run

# 2. 启动纯本地 CPU SO-101 网关（冷启动缺标定文件时自动附加 --observation-profile rgb_proprio_v1）
ni-rua-plugin gateway --config so101

# 3. 单独安装 / 卸载 / 体检
ni-rua-plugin install <claude|codex|opencode> [--scope user|project|local] [--port N]
ni-rua-plugin uninstall <claude|codex|opencode> [--scope user|project|local]
ni-rua-plugin doctor [--so101] [--json] [--gateway-url URL]
```

## 安装器机制

安装器会先将本包复制到持久目录 `~/.local/share/ni-rua/agent-plugin/<version>/`，再向各宿主注册（避免 `npx` 临时缓存被清理后插件失效）。重复运行 `install` 幂等且可无缝升级；所有宿主命令与配置变更都会在执行前打印并记录于 `installed.json`，`uninstall` 仅精确撤销本工具写入的项并保留用户自定义设置。

| 选项 | 含义 |
|---|---|
| `--so101` | `setup`：额外创建隔离的 Python 3.12 SO-101 硬件环境（`lerobot`、`feetech-servo-sdk`、本地 CPU IK）；`doctor`：检查 USB 串口、Linux `dialout` 权限、`so101.local.json` 与相机标定。 |
| `--skip-gateway` | `setup`：仅安装宿主插件，跳过 Python 网关环境创建。 |
| `--dry-run` | `gateway`：在 `http://127.0.0.1:8790/mcp` 启动零硬件内存模拟 MCP 服务（支持 `run_begin`、`robot_observe`、`robot_action`、`run_finish`）。 |
| `--config <preset\|path>` | `gateway`：传给 `ni-rua` 的内置预设名（`so101`、`so101_standalone`、`ni-rua`）或配置文件路径。 |
| `--port N` | `install` / `doctor`：MCP 端口（默认 `8790`）；`gateway`：Web 控制台端口（`--dry-run` 下为 MCP 端口）。 |
| `--gateway-url URL` | `doctor`：指定要探测的 MCP URL。 |
| `--json` | `doctor`：输出结构化 JSON 体检结果。 |
| `--scope user\|project\|local` | 仅适用于 Claude Code（`install` / `uninstall` / `setup`），默认 `user`。 |
| `--no-host-env` / `--no-allow` | Claude Code：跳过写入四个 MCP 环境变量或 `permissions.allow`。 |
| `--no-direct-catalog` | Codex：跳过向 `$CODEX_HOME/config.toml` 写入 `model_catalog_json`。 |
| `--force` | 强制替换同版本不同内容的持久副本，或在 `setup` 中重建托管 Python 环境。 |
| `--claude-bin`、`--codex-bin`、`--opencode-bin` | 指定宿主可执行文件路径。 |

## 各宿主配置说明

- **Claude Code**：注册 `ni_rua@ni_rua`、`.mcp.json`（`type: http`, `alwaysLoad: true`）、防压缩 `SessionStart`（`compact`）Hook、四个 MCP 环境变量及 `permissions.allow += ["mcp__plugin_ni_rua_ni_rua"]`。
- **Codex**：注册 `ni_rua@ni_rua`、`.codex-mcp.json`（`default_tools_approval_mode: "approve"`、`tool_timeout_sec: 3600`、逐工具 `output_token_limit`），并在 `$CODEX_HOME/config.toml` 写入 `model_catalog_json = "<copy>/codex-models.direct.json"`，避免 Linux `code-mode host` 崩溃。
- **OpenCode（2.x）**：写入 `opencode.json` 的 `plugins`、`tool_output`、`media.image` 并注册 `compaction` Hook；拒绝 1.x 版本。

## 安全提醒

在智能体中按 `Esc` 取消工具调用**不会停止真实机械臂**。紧急情况请立即使用 NI-RUA Web 控制台（`http://127.0.0.1:8781/console`）的 **Hold / E-STOP** 或物理 12V 断电开关。

## 源码维护

`.codex-mcp.json` 与 `manifest.generated.json` 由工具注册表自动生成，严禁手工修改：

```sh
mamba run -n ni-rua python scripts/gen_agent_plugin_codex_mcp.py --check
npm --prefix agent-plugin test
agent-plugin/test/tarball-check.sh
```
