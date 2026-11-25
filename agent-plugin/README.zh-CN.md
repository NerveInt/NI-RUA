# NI-RUA Agent 插件

[English](README.md) · [简体中文](README.zh-CN.md)

将你现有的 Claude Code、Codex 或 OpenCode 会话直接作为 NI-RUA 机器人运行的执行智能体（Actor）。该插件通过 MCP（由本地 NI-RUA 网关提供服务）向智能体暴露 NI-RUA 机器人工具，并附带两项技能（Skills）：

- `ni-rua-robot`：指导智能体如何开始（`run_begin`）、执行并结束（`run_finish`）一次机器人运行，如何停止机械臂，以及网关不可达时如何恢复；
- `ni-rua-experiments`：指导在 NI-RUA 源码仓库中工作的智能体如何遵循现有队列、报告与审计工具规范来启动、恢复、回放、统计和审计实验。

本包不包含任何 Python 代码或机器人软件。NI-RUA Harness、机器人运行环境与 Provider Profile 需单独安装（克隆仓库，并按 `environment.yml` 创建 `ni-rua` 环境）。完整说明（与托管模式共存、生命周期门控、控制台行为、恢复流程与验证状态）见 [Agent 插件指南](../docs/development/agent-plugin.zh-CN.md)。如果是首次搭建真机 SO-101 机械臂，请先阅读 [SO-101 真机快速上手指南](../docs/guides/so101-quickstart.zh-CN.md) 与 [SO-101 标定指南](../docs/guides/so101-calibration.zh-CN.md)。

## 安装

环境要求：Node 18 或更新版本，macOS 或 Linux（Windows 仅支持通过 WSL 运行），以及你希望使用的宿主 CLI。

```sh
npx --package ./ni-rua-agent-plugin-<version>.tgz ni-rua-plugin install claude   # 或：node bin/cli.mjs install claude
node bin/cli.mjs install codex
node bin/cli.mjs install opencode
node bin/cli.mjs doctor
node bin/cli.mjs uninstall <claude|codex|opencode> [--scope user|project|local]
```

本包尚未发布到 npm（在维护者决定发布前，`package.json` 保持 `"private": true`）；请直接从源码仓库安装（在 `agent-plugin/` 目录下执行 `npm pack`，或直接运行 `node agent-plugin/bin/cli.mjs`，其命令入口名为 `ni-rua-plugin`）。安装器会先将本包复制到 `~/.local/share/ni-rua/agent-plugin/<version>/`，再从该持久副本执行安装。重复运行 `install` 是幂等的，同时也是升级路径。版本目录不可变：若同版本号下内容发生变化，除非传入 `--force`，否则安装器会拒绝覆盖。每条宿主命令与配置改动都会在执行前打印，并记录在副本旁的 `installed.json` 中，因此 `uninstall` 会精确移除它写入的内容，并保留你自行配置的值。

| 选项 | 含义 |
|---|---|
| `--port N` | 写入已安装副本三个宿主入口的 MCP 端口（默认 `8790`，对应网关的 `agent.external.mcp_port` / `--external-mcp-port`）。 |
| `--scope user\|project\|local` | 仅适用于 Claude Code（对 `install` 和 `uninstall` 均生效），默认为 `user`。`project` 与 `local` 作用于当前目录（该工作区必须已被 Claude Code 信任）。 |
| `--no-host-env` | Claude Code：不写入下述四个 `env` 环境变量键。 |
| `--no-allow` | Claude Code：不添加 `permissions.allow` 条目。 |
| `--no-direct-catalog` | Codex：不向 `$CODEX_HOME/config.toml` 写入 `model_catalog_json`。 |
| `--force` | 强制替换同版本号但内容不同的持久副本。 |
| `--claude-bin`、`--codex-bin`、`--opencode-bin` | 指定要使用的宿主可执行文件，而不是 `PATH` 中找到的第一个。 |

各宿主安装后获得的改动：

- **Claude Code**：执行 `claude plugin marketplace add <copy>` 与 `claude plugin install ni_rua@ni_rua -s <scope>`。插件的 `.mcp.json` 条目为 `type: http`、`alwaysLoad: true`，`hooks/hooks.json` 注册了 `SessionStart`（`compact`）Hook，用于在活跃的 NI-RUA 回合发生上下文压缩时提醒智能体重新调用 `run_begin`。安装器会在对应作用域的 `settings.json` 中写入 `env` 键 `MAX_MCP_OUTPUT_TOKENS`、`MCP_TOOL_TIMEOUT`、`CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS`、`CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH`（你已设置的键会保留）以及 `permissions.allow += ["mcp__plugin_ni_rua_ni_rua"]`（缺少该权限条目会导致首次机器人工具调用被拒）。使用 `--scope project` 或 `--scope local` 时，在交互式 `claude` 中信任当前工作区（或目录本身没有 `.git` 时的外层 `.git` 根目录）之前，Claude Code 会忽略项目/本地级别的 `permissions.allow` 与 `env`；`install` 和 `doctor` 都会检测未信任状态并给出警告。
- **Codex**：执行 `codex plugin marketplace add <copy>` 与 `codex plugin add ni_rua@ni_rua`（升级时会先移除再重新添加，因为 Codex 会将插件复制到自身缓存中）。`.codex-mcp.json` 包含 `default_tools_approval_mode: "approve"`、`tool_timeout_sec: 3600` 以及逐工具的 `output_token_limit`（缺少该上限时 Codex 会将长结果截断到约 12,000 字符）。除非 `model_catalog_json` 已设置或传入了 `--no-direct-catalog`，安装器还会向 `$CODEX_HOME/config.toml` 添加一行顶层配置 `model_catalog_json = "<copy>/codex-models.direct.json"`（并在 `uninstall` 时精确移除该行），将内置模型切换为 `tool_mode: "direct"` 与 `supports_search_tool: false`，使 MCP 工具直接送达模型，避免 Linux 上 `code-mode host` 崩溃或 `tool_search` 间接层问题。
- **OpenCode**：要求 OpenCode **2.x**（已验证 `2.2.7`）；`install opencode` 会拒绝 OpenCode 1.x 二进制，因为 1.x 不支持顶层 `plugins` 与 `tool_output`。安装器写入全局 `opencode.json`（位于 `$OPENCODE_CONFIG_DIR` 或 `$XDG_CONFIG_HOME/opencode/`）：向 `plugins` 写入副本的绝对路径（`opencode plugin add` 不接受本地路径）；若未设置则将 `tool_output` 设为 `{max_lines: 100000, max_bytes: 16000000}`（否则超过约 50 KB 的工具结果会被截断）；若未设置则将 `media.image` 设为 `{auto_resize: false, max_width: 1000000, max_height: 1000000, max_base64_bytes: 1073741824}`（防止大型 PNG 工具图像被有损重编码为 JPEG）。`opencode/plugin.js` 还会注册 `compaction` 会话 Hook，避免活跃的 NI-RUA 回合将内部压缩提示误认为用户请求，并指示摘要在恢复后优先重新调用 `run_begin`。若配置文件是 JSONC 或无法解析，安装器会拒绝写入并打印需要手动修改的精确内容。

## 使用方法

1. 先在 NI-RUA 源码仓库中以外部智能体模式启动网关：
   `mamba run -n ni-rua python -u -m ni_rua.apps.gateway --agent-backend external --config <config> --run-dir <fresh dir>`。
   若使用单机纯 CPU 无 GPU 的 SO-101 真机，`<config>` 填 `configs/so101_standalone.json`（首次开箱尚未完成 `/calibration` 标定时需附加 `--observation-profile rgb_proprio_v1`；仅在对接可选远程 VGGT/cuRobo GPU 服务时改用 `configs/so101_gateway.json`）。
   网关会在 `http://127.0.0.1:8790/mcp` 上提供 MCP 服务。
2. 启动你的智能体会话。`ni_rua` 工具列表会出现；智能体调用 `run_begin`、执行操作，并在最后调用 `run_finish`。
3. 如果智能体先于网关启动：Claude Code 执行 `/mcp` → `ni_rua` → Reconnect（或开启新会话）；OpenCode 重启或重载；Codex 开启新会话。网关切换机器人并重启后也需要同样操作。

在智能体中取消工具调用（按 Esc）不会让机械臂停下。请使用 NI-RUA 控制台中的停止（stop）、保持（hold）或紧急停止（estop）。

`ni-rua-plugin doctor` 会检查 Node 版本、各宿主版本是否符合已验证版本（Claude Code 2.1.321、2.1.325 和 2.1.328，Codex 0.160.0、0.167.0 和 0.169.0，OpenCode 2.2.7）、各宿主是否已列出该插件、Claude Code 环境变量键与权限条目是否生效，以及目标端口上的网关能否响应 `initialize` 和 `tools/list`。

## 目录结构与维护

`.claude-plugin/`（Claude Code 插件清单与 marketplace）、`.mcp.json`（Claude Code MCP 入口）、`.codex-plugin/` 与 `.codex-mcp.json`（Codex）、`opencode/plugin.js`（OpenCode 入口，同时注册两项技能）、`skills/`、`bin/cli.mjs`。

`.codex-mcp.json` 由工具注册表自动生成，严禁手工编辑：
`mamba run -n ni-rua python scripts/gen_agent_plugin_codex_mcp.py`（加 `--check` 执行校验）。测试命令：在本目录运行 `npm test`，在仓库根目录运行 `tests/packaging/npm/test_agent_plugin_package.py`。`test/tarball-check.sh` 会使用固定版本宿主二进制，将打包后的 tarball 安装到隔离的 HOME 目录中进行核验。

当前已验证范围：上述宿主版本已通过脚本化模型验证 HTTP 连接至 NI-RUA MCP 服务。该结果属于传输层证据，不构成对真实模型运行的全面支持承诺。
