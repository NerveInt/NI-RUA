# Agent 插件：用你自己的智能体驱动 NI-RUA

[English](agent-plugin.md) · [简体中文](agent-plugin.zh-CN.md)

Agent 插件允许你把正在使用的 Claude Code、Codex 或 OpenCode 会话直接作为一次 NI-RUA 运行的执行智能体（Actor），同时也支持让该智能体在 NI-RUA 源码仓库中调度和分析实验。插件包含两套能力，二者的验证保证不同。在依赖任一能力前，请先阅读[验证状态](#verification-status)：传输链路已使用脚本化模型完成验证，各宿主也各通过了一次短时真实模型会话（属于连通性冒烟测试，而非全面支持承诺）。

## 两种模式，互不排斥

NI-RUA 提供两种使用方式，同一份代码仓库和同一套安装可同时支持二者；每次运行通过 `--agent-backend` 选择模式。

- **托管模式（Hosted mode）** 是完整框架路径：NI-RUA 自行启动并监管 `codex`、`claude_code` 或 `opencode`，配套模型中继、预算控制、来源追溯、OS 沙箱、批量队列与基准评测。这是端到端控制与正式评测的标准路径。
- **插件模式（Plugin mode，本指南主题）** 是可选扩展：`--agent-backend external` 把机器人工具直接交给你自己的智能体，用于交互式与探索式操作。

安装插件不会影响托管运行。托管启动从不读取安装器写入的宿主配置目录，也不会绑定插件端口；运维技能 `ni-rua-experiments` 还允许你自己的智能体去调度托管模式的评测队列，因此两种模式可以结合使用。各宿主的隔离事实如下（基于源码核查，并由 [`tests/packaging/npm/test_agent_plugin_coexistence.py`](../../tests/packaging/npm/test_agent_plugin_coexistence.py) 锁定，该测试会将安装器条目写入伪造的用户配置目录并构造每种托管启动）：

| 宿主（托管模式） | 安装器写入的内容 | 托管运行为何会忽略它 |
| --- | --- | --- |
| Claude Code | `settings.json` 中的用户级 `enabledPlugins`、`extraKnownMarketplaces`、`env`、`permissions.allow` | [`options.py`](../../ni_rua/cortex/agents/claude_code/options.py)：每次运行独立的 `CLAUDE_CONFIG_DIR`（`private/claude-config`）、`setting_sources=[]`、`strict_mcp_config=True` 以及干净的启动环境变量 |
| Codex | `~/.codex` 的 `config.toml` 与 `plugins/cache/` 中的 `[marketplaces.ni_rua]`、`[plugins."ni_rua@ni_rua"]` 与 `model_catalog_json` | [`codex/session.py`](../../ni_rua/cortex/agents/codex/session.py) `_prepare`：每次运行独立的 `private/codex-home` 与自动生成的 `config.toml`（`features.plugins = false`、独立 `model_catalog_json`、不含 marketplace 或 plugin 表）；从 `model.codex_home` 仅读取 provider 配置块与凭据，不复制任何插件或技能文件；`CODEX_HOME` 指向私有目录 |
| OpenCode | 全局 `opencode.json` 中的 `plugins` 条目（以及 `tool_output`） | [`config.py`](../../ni_rua/cortex/agents/opencode/config.py)：每次运行独立的 `OPENCODE_CONFIG_DIR` == `$XDG_CONFIG_HOME/opencode`，独立的 XDG data/cache/state 目录，以及不含 `plugins` 的生成配置 |

只有 `external` 后端会绑定插件端口（`agent.external.mcp_port` / `--external-mcp-port`，默认 `8790`，见 [`external/session.py`](../../ni_rua/cortex/agents/external/session.py)）；托管后端使用 `NiRuaMcpServer` 默认的临时端口 `0` 与单局 Bearer 令牌，托管启动不会探测 `8790`。已知余项：运维人员仍可有意将 `model.codex_home` 或 Claude 的 `claude_settings` 上游指向同一个主目录，此时仅会读取 [Agent Backends](agent-backends.md) 中列明的 provider 与凭据项；上述共存隔离已在启动构造层验证，尚未对每个宿主跑过真实智能体交叉实测。

## 插件的两套能力

| | 工具侧（新增能力） | 调度侧（基于既有工具的文档技能） |
| --- | --- | --- |
| 定位 | `external` 智能体后端：`--agent-backend external` 启动机器人、`Harness` 以及既有的 [`NiRuaMcpServer`](../../ni_rua/cortex/agents/mcp_server.py)（固定回环端口），不拉起智能体也不启动模型中继，由你自己的智能体通过 MCP 驱动运行。 | `ni-rua-experiments` 技能指导你的智能体按照 [AGENTS.md](../../AGENTS.md) 规则操作单局网关、`ni-rua-queue`、`ni-rua-report`、`ni-rua-audit` 与实验库。 |
| 代码 | [`ni_rua/cortex/agents/external/`](../../ni_rua/cortex/agents/external/session.py)、[`run_bracket.py`](../../ni_rua/runtime/tools/plugins/run_bracket.py) | [`agent-plugin/skills/ni-rua-experiments/SKILL.md`](../../agent-plugin/skills/ni-rua-experiments/SKILL.md)；无新增 Python 代码 |
| 可驱动的 Actor | 你自己的智能体（交互式） | 队列中的每个用例仍使用 NI-RUA 托管的 Actor（`codex`、`claude_code`、`opencode`） |
| 前置条件 | 源码仓库、`ni-rua` 环境、机器人运行环境 | 源码仓库、`ni-rua` 环境、`configs/profiles/` 中的 Provider Profile、所选后端固定版本的智能体运行时，以及运行 RoboDojo 基准所需的 GPU 宿主机（普通笔记本仍可用于回放、统计与审计） |

外部智能体运行仅限交互式，不可入队：队列会拒绝 `agent_backend="external"`（错误码 `external_backend_not_queueable`，见 [`queue.py`](../../ni_rua/apps/evaluation/queue.py)），因为排队用例没有人工或外部会话接入。切勿将 `external` 运行写入评测 manifest。

## 插件包内容与独立安装项

npm 包 [`agent-plugin/`](../../agent-plugin/README.zh-CN.md)（包名 `ni-rua-agent-plugin`，安装器命令 `ni-rua-plugin`，零第三方依赖）仅包含指向 `http://127.0.0.1:8790/mcp` 的各宿主 HTTP MCP 清单、两项技能（面向执行者的 `ni-rua-robot` 与面向运维者的 `ni-rua-experiments`）以及安装器（`install`、`uninstall`、`doctor`）。该包不含任何 Python 代码或机器人软件。

与任何 NI-RUA 运行一样，以下组件需单独安装：

- Python Harness（克隆仓库、`mamba env create -f environment.yml` 创建 `ni-rua` 环境）、机器人环境（RoboDojo 或 SO-101 部署环境），以及（仅在运行托管队列用例时）Provider Profile；
- Node 18 或更新版本（安装器为纯 Node 实现，仅使用内置模块）以及你要使用的宿主 CLI；
- 操作系统平台：macOS 与 Linux。Windows 仅支持通过 WSL 运行（网关依赖 `fcntl`）。

该包尚未发布至 npm（`"private": true`），请直接从源码仓库运行（`node agent-plugin/bin/cli.mjs`）或从 `npm pack` 打出的 tarball 安装。

## 快速开始

1. 先在源码仓库中启动网关，再开启智能体会话（任务仅来源于 `--prompt`；见[生命周期门控：`run_begin` 与 `run_finish`](#the-bracket-run_begin-and-run_finish)）。外部模式绝不可带 `--benchmark`。按你的机器人选择对应命令并指定控制台 `--port`：

   ```sh
   # RoboDojo 仿真（需 1 张空闲 GPU；沿用 configs/ni-rua.json 中的 RoboDojo 路径，见 running.zh-CN.md）
   mamba run -n ni-rua python -u -m ni_rua.apps.gateway --agent-backend external \
     --task stack_bowls --gpu 0 --port 8767 --run-dir runs/my-external --prompt "<task>"

   # SO-101 机械臂 —— 单机本地纯 CPU 默认配置（真机运行须遵循正常真机安全流程，且操作员必须在机台旁值守）
   mamba run -n ni-rua python -u -m ni_rua.apps.gateway --agent-backend external \
     --config configs/so101_standalone.json --port 8781 --run-dir runs/my-external --prompt "<task>"
   ```

   首次搭建 SO-101 且尚未生成 `camera_calibration.json` 时，启动网关必须附加 `--observation-profile rgb_proprio_v1`，以便打开 `http://127.0.0.1:8781/calibration` 完成引导标定；标定应用完成后，日常单机纯 CPU 运行直接使用 `--config configs/so101_standalone.json` 即可。仅在对接可选远程 VGGT/SAM/cuRobo GPU 服务时改用 `--config configs/so101_gateway.json`。两份配置都在 `backend.deployment` 中指定了 SO-101 机台部署文件（即你的 `configs/so101.local.json`；详见 [SO-101 真机快速上手指南](../guides/so101-quickstart.zh-CN.md)、[SO-101 标定指南](../guides/so101-calibration.zh-CN.md)、[SO-101 配置说明](so101-configuration.md) 与 [SO-101 集成指南第 6.3 节](so101-integration.md)），请按本机路径修改。首次使用前请先构建控制台前端：`npm --prefix web ci && npm --prefix web run build`，然后打开 `http://127.0.0.1:<port>/console`（`<port>` 为上方 `--port`）。`--run-dir` 必须是一个尚不存在的新目录。

   网关会在 `http://127.0.0.1:8790/mcp` 上提供 MCP 服务（对应 `agent.external.mcp_port`，可用 `--external-mcp-port` 覆盖，见[配置指南](configuration.zh-CN.md)中的 Agent Backend 配置节），并在启动时打印端点地址。若端口已被占用，启动会拒绝并输出一行指明该端口的提示。对于 RoboDojo 运行，`run_begin` 还会通过 `native_task` 和 `task_notes` 透传环境原生的 `task_instruction`（含 `task_instruction_override`）、`public_task_rules` 与 `default_task_notes`。

   如需在没有实体机械臂的情况下试用插件，见[无硬件试运行（仅限源码仓库）](#dry-run-without-hardware-checkout-only)。

2. 将插件安装到宿主并执行自检：

   ```sh
   node agent-plugin/bin/cli.mjs install claude     # 或：codex | opencode
   node agent-plugin/bin/cli.mjs doctor
   ```

3. 启动你自己的智能体会话并让它执行任务。`ni_rua` 工具列表会出现；智能体调用 `run_begin`、开展操作，最后调用 `run_finish`。

请始终通过 NI-RUA 控制台（即网关的 `--port`）执行停止（stop）、保持（hold）或紧急停止（estop）。

<a id="dry-run-without-hardware-checkout-only"></a>
## 无硬件试运行（仅限源码仓库）

测试套件中的场景替身（scene double）在模拟舵机总线与渲染相机之上运行真实的 SO-101 后端与适配器（无需串口，也无需实体机械臂）。它属于测试辅助模块，不包含在发行的 wheel 中。在源码仓库根目录执行：

```sh
D=$(mktemp -d)
PYTHONPATH=tests mamba run -n ni-rua python - "$D" <<'EOF'
import json, sys
from support import so101_scene
d = sys.argv[1]
deployment, scene = so101_scene.write_scene(d + "/scene", so101_scene.scene_document("red"))
features = json.load(open("configs/so101_gateway.json"))["features"]
features.update(terminal=False)
json.dump(dict(version=1, backend=dict(environment="so101_real", robot_model_id="so101", deployment=str(deployment)),
               features=features, agent=dict(backend="external"),
               run=dict(experiment_root=d + "/exp", category="dry_run", task="desk_free_task"),
               remote=dict(enabled=False)), open(d + "/ni-rua.json", "w"))
EOF
PYTHONPATH=tests:. NI_RUA_SO101_SCENE="$D/scene/scene.json" mamba run -n ni-rua python -u -m support.so101_scene \
  --config "$D/ni-rua.json" --agent-backend external --run-dir "$D/run" --port 8782 --no-remote --prompt "<task>"
```

由于没有远程感知服务，基于 VGGT 的感知会回报 `unavailable`，场景中在 `[0.23, 0.08]` m 处放置了一块红色方块。可用于试用 `install`、`doctor` 与控制台，但不代表真实硬件运动能力。

## 安装器写入的内容

安装器会先将插件包复制到 `~/.local/share/ni-rua/agent-plugin/<version>/`，再从该持久副本执行安装（因为 `npx` 缓存路径不稳定，且 Codex 会把本地 marketplace 复制进自身缓存）。每条宿主命令与配置改动都会在执行前打印。源码见 [`agent-plugin/bin/cli.mjs`](../../agent-plugin/bin/cli.mjs)。

| 选项 | 含义 |
| --- | --- |
| `--port N` | 写入副本三个宿主入口的 MCP 端口（默认 `8790` = `agent.external.mcp_port`）。所有已安装宿主共用同一副本，因此共用同一端口；若已有其他宿主安装记录，尝试改用不同 `--port` 会被拒绝。 |
| `--scope user\|project\|local` | 仅限 Claude Code（`install` 与 `uninstall` 均适用）；默认 `user`。`project` 与 `local` 作用于当前目录（工作区须已在 Claude Code 中获信任）。 |
| `--no-host-env`、`--no-allow` | Claude Code 选项：跳过写入 `env` 键或跳过写入 `permissions.allow`。 |
| `--no-direct-catalog` | Codex 选项：跳过向 `$CODEX_HOME/config.toml` 写入 `model_catalog_json`。 |
| `--force` | 强制覆盖同版本号但内容不同的持久副本（默认拒绝覆盖）。 |
| `--claude-bin`、`--codex-bin`、`--opencode-bin` | 指定宿主可执行文件路径，取代 `PATH` 中的首个匹配项。 |

**Claude Code。**（使用默认 `--scope user` 时，该配置对该用户的**所有** Claude Code 会话与任意项目生效，而不仅限于 NI-RUA 会话：MCP 服务会在启动时自动加载且机器人工具默认放行。如果你还用 Claude Code 做其他项目开发，请在 NI-RUA 仓库目录下改用 `--scope project`；另见[已知余项（已记录，未加固）](#residuals-documented-not-hardened)。注意：在未受信任的工作区中，Claude Code 会忽略项目/本地级 `.claude/settings.json`——包括 `permissions.allow` 与 `env`——并且当当前目录不含独立 `.git` 时会沿上层 `.git` 根目录解析信任状态；请在该工作区中启动一次交互式 `claude` 确认信任，或运行 `ni-rua-plugin doctor` 核查。）安装器会执行 `claude plugin marketplace add <copy>` 与 `claude plugin install ni_rua@ni_rua -s <scope>`；插件的 `.mcp.json` 条目为 `{"type": "http", "url": ..., "alwaysLoad": true}`。在对应作用域的 `settings.json`（`user` 为 `<CLAUDE_CONFIG_DIR 或 ~/.claude>/settings.json`，`project` 为 `.claude/settings.json`，`local` 为 `.claude/settings.local.json`）中，会添加以下四个 `env` 键与一条权限：

| 键 | 值 | 原因 |
| --- | --- | --- |
| `MAX_MCP_OUTPUT_TOKENS` | `2000000` | Token 上限；防止 Claude Code 改写大段文本结果 |
| `MCP_TOOL_TIMEOUT` | `3600000` | 毫秒；单次 MCP 调用超时上限（默认 60 秒） |
| `CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS` | `0` | 禁止将长耗时 MCP 调用自动转入后台 |
| `CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH` | `1000000` | 字符上限；默认 4096 会截断自动生成的字段说明 |
| `permissions.allow` | `mcp__plugin_ni_rua_ni_rua` | 缺少该权限条目会导致默认权限模式下首次机器人调用被拒 |

这些键与 [`claude_code/options.py`](../../ni_rua/cortex/agents/claude_code/options.py) 托管模式使用的配置一致，唯一刻意未写入的是 `CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT=0`：正常调用期间的进度保活（progress keepalive）已经能维持连接，若写入该键，网关在调用中途崩溃时会一直挂起直到手动 `/mcp` 重连，而不会在 300 秒后超时失败。你原先已设置的键会被保留；若保留值低于表格要求或启用了自动转后台，导致插件失效，`install` 和 `doctor` 会输出警告。

**Codex。** 执行 `codex plugin marketplace add <copy>` 与 `codex plugin add ni_rua@ni_rua`。升级或更换端口时会先移除旧插件与 marketplace（Codex 不支持原地重指）。与 Claude Code 类似，安装后的插件对该用户的 Codex 会话可见且工具默认放行。自动生成的 [`.codex-mcp.json`](../../agent-plugin/.codex-mcp.json) 包含 `default_tools_approval_mode: "approve"`、`tool_timeout_sec: 3600`（硬上限，进度通知不会延长它）、`startup_timeout_sec: 30`，以及针对注册表中每个 Actor 工具的 `output_token_limit: 100000`：缺少该上限时 Codex 会将所有长结果截断到约 12,000 字符（SO-101 的开场 briefing 长达 60,091 字符）。该工具映射表与直连模型目录 [`codex-models.direct.json`](../../agent-plugin/codex-models.direct.json) 均由 [`scripts/gen_agent_plugin_codex_mcp.py`](../../scripts/gen_agent_plugin_codex_mcp.py) 生成，切勿手工编辑。由于 Codex 0.160.0/0.169.0 将 `gpt-6.1-sol` 等内置模型标记为 `tool_mode="code_mode_only"`（在 Linux 上会导致 `code-mode host` 以 `SIGSEGV`/gRPC `BrokenPipe` 崩溃）及 `supports_search_tool=true`（会将 MCP 工具隐藏到 Responses `tool_search` 后方，在不支持 `tool_search` 的供应商上报错），`codex-models.direct.json` 将所有模型设为 `tool_mode="direct"` 与 `supports_search_tool=false`；除非 `$CODEX_HOME/config.toml` 已存在该配置或传入 `--no-direct-catalog`，`install codex` 会在顶层添加 `model_catalog_json = "<copy>/codex-models.direct.json"`，并在 `uninstall codex` 时精确删除该行。

**OpenCode。** 要求 OpenCode **2.x**（已验证 `2.2.7`）；`install opencode` 在写入任何内容前会拒绝 OpenCode 1.x 二进制，因为 1.x 不接受顶层 `plugins` 和 `tool_output`。[`opencode/plugin.js`](../../agent-plugin/opencode/plugin.js) 注册了远程 MCP 入口（`oauth: false`、`codemode: false` 以保持注册表工具作为直接 `ni_rua_<tool>` 调用、`timeout.execution: 3600000` ms，因为在 2.2.7 中进度通知不会顺延超时）、两项技能，以及一个 `compaction` 会话 Hook（防止活跃的 NI-RUA 回合把内部压缩提示误当成用户请求，并要求压缩摘要优先重新调用 `run_begin`）。由于 `opencode plugin add` 不接受本地路径，安装器直接写入全局 `opencode.json`（位于 `$OPENCODE_CONFIG_DIR` 或 `$XDG_CONFIG_HOME/opencode/`，仅接受严格 JSON；遇到 JSONC 或无法解析的文件时会拒绝修改并打印手工编辑片段）：向 `plugins` 添加副本绝对路径；若未配置则写入 `tool_output: {"max_lines": 100000, "max_bytes": 16000000}`（与托管模式一致，否则约 50 KB 以上结果会被截断，SO-101 的 60,091 字符 briefing 会被砍成 50,135 字符）；若未配置则写入 `media.image: {"auto_resize": false, "max_width": 1000000, "max_height": 1000000, "max_base64_bytes": 1073741824}`（防止大型 PNG 工具图像被有损重编码为 JPEG）。

**记录、卸载与升级。** 安装器在 `~/.local/share/ni-rua/agent-plugin/installed.json`（schema 2）中记录所有操作：按宿主与配置目录分条记录（Claude Code 按 `CLAUDE_CONFIG_DIR` + `scope`，`project`/`local` 还包含工作目录；Codex 按 `CODEX_HOME`；OpenCode 按配置文件路径），确保不同 profile 或项目互不覆盖。`uninstall <host>` 仅撤销对应记录添加的条目，并保留你后续自行修改的值；卸载时需使用与安装时一致的 `CLAUDE_CONFIG_DIR`、`CODEX_HOME`、`XDG_CONFIG_HOME` 与 `scope`。如果宿主命令因“未安装”以外的原因失败，安装器会保留记录与副本、打印手动命令并以非零状态退出。安装器绝不删除非自身创建的内容，保留符号链接配置文件及其权限位，并通过锁目录（`<data>/.lock`）串行化并发执行。重新运行 `install` 即为升级路径；版本目录不可变，同一版本号下内容变化时需升级版本号或传入 `--force`。所有拒绝校验都在首次写入前完成，因此被拒绝的重装不会破坏现有可用配置。相关逻辑由 [`agent-plugin/test/installer.test.mjs`](../../agent-plugin/test/installer.test.mjs) 回归测试覆盖。

<a id="the-bracket-run_begin-and-run_finish"></a>
## 生命周期门控：`run_begin` 与 `run_finish`

这是两个仅在 `agent.backend="external"` 时注册的标准注册表工具（[`run_bracket.py`](../../ni_rua/runtime/tools/plugins/run_bracket.py)），角色为 `actor`，声明副作用为 `session`（见 [Tool Plugins](tool-plugins.md#side-effects-and-program-authorization)）。二者均不产生物理动作，也不评判任务成败。其工具描述本身即为契约，核心要点如下：

- **`run_begin`**（无入参）返回 `kind="briefing"`，包含：`instructions`（运行启动指令，与托管模式同源，由 `EmbodimentProfile` 渲染并经脱敏处理，采用本地源变体，不含联网搜索、工作目录、沙箱声明或 run 目录路径）、`task {text, source}`、可选的 `native_task {task, instruction, public_task_rules}` 与 `task_notes`（在 RoboDojo 等仿真环境或传入 `--task-notes` 时填充）、`budget`（原生步数统计，`null` 表示未知）以及 `turn.opened`。在已开启的回合内再次调用 `run_begin` 会重读 briefing 并触发 `Harness.model_context_reset("external_rebriefing")`，使决策提醒附带在下一次工具结果中；回合内的首次 `run_begin` 不会触发重置。
- **`run_finish`**（入参 `summary`，字符串，长度 1 至 16,000 字符）将当前回合以 `completed` 关闭，并将 `summary` 记录为本回合最终答复（transcript 来源标记为 `run_finish_summary`，而非原生自由文本）；返回 `{turn_closed: true}`。辅助程序不会随之自动停止：请先读取 `program_status`；处于等待复核暂停态的任务可通过 `program_run resume decision=reject` 终止。
- **无默认任务。** `task.text` 仅来自 `--prompt` 或 `run.prompt`；`task.source` 为 `operator_prompt` 或 `none`（后者需由用户直接向智能体陈述任务）。绝不会套用开发验收默认提示词，以免误驱动实体机械臂。`--resume` 默认沿用归档任务，传入 `--prompt` 时会覆盖它。
- **简报门控（Briefed-turn gate）。** 其他所有工具仅在由 `run_begin` 开启且已成功下发 briefing 的活跃回合内提供服务。否则调用会被直接拒绝，不执行任何操作，也不写入 transcript 行（MCP 原始日志会记录该次拒绝，并产生事件 `agent.tool_refused {code, tool}`）：未开场时返回 `external_briefing_required: call run_begin first; it returns the operating instructions and the task`；回合关闭后返回 `external_turn_closed` 并用一句话说明原因：`finished`（已调用 `run_finish`）、`operator_stop`（操作员已停止运行，在操作员恢复前持续拒绝运动）或 `operator_reset`（操作员已重置环境，此前的观测与计划不再有效）。若 `run_begin` 在下发 briefing 前失败，其开启的回合会立即关闭（`interrupted`，`external_bracket_call_failed`）。回合开启期间允许任意调用者进入，因此宿主自身的子智能体可共享该运行的当前回合。
- **并发与顺序。** 门控按分派顺序而非 HTTP 到达顺序生效。若某调用在 `run_begin` 获准前被分派，会被直接拒绝且不执行任何操作，重新发起即可；后端未加入等待阻塞。工具描述已明确告知智能体须等待 `run_begin` 返回后再调用其他 NI-RUA 工具。
- **单步内多次调用的顺序。** <a id="order-of-one-steps-calls"></a>外部模式按 `tools/call` 请求**到达**顺序依次串行执行有状态（ordered）调用；并行安全（parallel-safe）调用可以重叠。由于外部模式没有模型中继，NI-RUA 看不到模型的原始输出，因此无法得知模型生成工具调用的先后顺序，而三款宿主发来的 `_meta` 字段也均未携带序号：OpenCode 2.2.7 发送 `ai.opencode/sessionID` 与 `progressToken`，Claude Code 发送 `claudecode/toolUseId`（仅为 ID 而非位置序号），Codex 发送 thread、session 与 turn ID。JSON-RPC ID 只是客户端的请求计数器，不可当作顺序依据（无法区分序号间隙是否来自其他类型请求）。对各宿主的实际影响：Claude Code 会等上一条有状态调用返回后再发下一条，因此执行顺序与模型一致；OpenCode 会一次性并发发出单步内的全部调用，因此同一步内的两条有状态调用可能乱序执行（在托管模式修复前曾测得：高负载下 96 个含三条有状态调用的步骤中有 4 步出现交换）。托管模式不存在此余项（见 [Agent Backends](agent-backends.md#mcp-tool-transport)）。在外部模式下使用 OpenCode 或任何并发发出单步调用的客户端时，凡对先后顺序敏感的调用，请拆到不同步骤中（等待前一次返回后再发下一次）。
- **重新获取简报（Re-briefing）。** 发生上下文压缩、开启新会话或恢复运行后，须再次调用 `run_begin`。MCP 的 `initialize.instructions`（仅外部模式发送的一段简短固定文案；托管模式不发送）以及 `ni-rua-robot` 技能均明确要求这样做，并同时提醒取消工具调用不会停止机械臂。

回归测试：[`tests/cortex/external/test_external_config_and_session.py`](../../tests/cortex/external/test_external_config_and_session.py)、[`tests/cortex/external/test_external_workbench.py`](../../tests/cortex/external/test_external_workbench.py)。

## 外部模式下的操作员控制台

控制台整体工作方式与其他模式一致，但有以下差异（见 [`session.py`](../../ni_rua/cortex/agents/external/session.py)、[`workbench.py`](../../ni_rua/apps/gateway/workbench.py)）：

- **Agent 栏。** 显示只读的“External agent”卡片（[`web/agent-settings-view.js`](../../web/agent-settings-view.js)）：展示 MCP 端点与两项事实，绝不宣称“已连接”（服务端无状态，无法判断智能体是否仍在线）：最近一次 `initialize` 的客户端信息（`clientInfo` 名称与版本，标注为自报 / self-reported）以及最近一次工具调用的时间（尚未调用过时中文界面显示 `"尚无工具调用"`，英文界面显示 `"No tool calls yet"`）。仅在从未收到过 `initialize` 时展示等待文案。`ni-rua-plugin doctor` 会以 `ni-rua-plugin-doctor` 名称发送 `initialize`；卡片仅对该精确名称标注为 doctor 探测而非 Agent（唯一按名匹配规则，见 [`web/agent-settings-model.js`](../../web/agent-settings-model.js) 中的 `DOCTOR_CLIENT_NAME`；其他所有名称均按自报展示）。回放已归档的外部运行时，显示 `"已结束的运行（回放）"`（英文界面为 `"Finished run (replay)"`，由 `client_view()["replay"]` 驱动），而非等待卡片。
- **被拒绝的控制台指令。** `prompt`、`steer`、`reply`、`set_model`、`set_effort`、`set_service_tier`、`apply_agent_settings`、`handoff_run` 与 `retry_model_requests` 均返回 `<command>_unsupported_by_backend`；不允许通过交接（handoff）启动或切出外部运行（只能从命令行启动）。
- **活跃智能体回合绝不阻塞操作员命令**：`resume`、`go_home`、标定与观察者抓拍均可随时执行。机械臂运动仍受 `request_stop`、机器人锁、活跃动作与活跃程序保护。
- **`stop`、`estop` 与网关关闭** 会将活跃回合标记为 `interrupted` 并关闭（`agent.turn_closed_by_operator {reason: "operator_stop"}`）；`reset` 会在重置实际完成后以 `operator_reset` 关闭回合（被拒绝的 reset 不会改变状态）。智能体下一次调用将收到指明原因的 `external_turn_closed`。
- **`stop` 之后的 `resume` 会立即交回运动权**给仍在会话中的智能体：一旦智能体再次调用 `run_begin`，后续运动调用将立即获准执行。按下 `resume` 前请务必先核对控制台状态。
- **在智能体中取消工具调用（按 Esc、中断生成）绝不会停止机械臂。** 已获准执行的动作会继续走完。请使用控制台的停止、保持或紧急停止按钮。

<a id="recovery"></a>
## 故障恢复

基于 HTTP-first 专项验证的实测结论（报告见 [Agent Plugin Spikes 第 3 节](reports/agent-plugin-spikes.md#3-http-first-gono-go)；脚本化模型、无头模式）：

- **务必先启动网关，再启动智能体会话。** 若智能体先启动：Claude Code 执行 `/mcp` → `ni_rua` → Reconnect（或开启新会话）；OpenCode 重启或重载；Codex 开启新会话。
- **在两次工具调用之间重启网关**无需在任何宿主做额外操作；MCP 服务端是无状态的。
- **重启网关并切换了另一台机器人**（工具目录变化）时，必须按上述方式重连或开启新会话，因为没有宿主会自动重新拉取 `tools/list`；长生命周期会话会保留旧的工具目录与说明。若发现工具描述与 briefing 不一致，请重新连接。
- **单次调用期间网关崩溃。** Codex 会立即报错（"Transport closed"）。Claude Code 与 OpenCode 会持续等待，直到你按 Esc 或手动重连；Claude Code 随后会告知模型该工具可能已经执行。机械臂当时可能仍在运动：再次发起运动前请先用只读工具检查状态。（使用 Claude Code 默认空闲超时会在 300 秒后报错；安装器正是为此刻意保留了该默认值。）
- **网关晚启动导致 `ni_rua` 服务状态为 `failed`。** 按上述步骤重连即可。Claude Code 自身仅在约 1 秒内重试数次便会放弃。

## 信息边界与基准有效性

- **边界。** manifest 记录 `agent_backend.kind="external"` 与 `information_boundary="mcp_boundary_only"`：智能体仅能看到工具目录、`initialize.instructions`、工具返回值与拒绝响应；NI-RUA 看不到它的提示词、模型、推理过程或其他工具。元数据脱敏器、短引用（`short_id`）及 [Actor 信息边界](actor-information-boundary.md) 中的其余规则仍适用于 NI-RUA 发送的所有数据。
- **仅限探索式交互运行。** `ni-rua-report` 会将 `kind=external` 单独归类并标记 `exploratory=true`（[`report.py`](../../ni_rua/apps/evaluation/report.py)）。`--benchmark` 与 `--exit-after-turn` 会被拒绝（`external_backend_is_interactive_only`），`features.sandbox_enabled`（`external_backend_cannot_sandbox`）、`features.visual_review_model`、`model.price_card`（`external_backend_has_no_model_relay`）以及禁用生命周期门控工具同样会被拒绝。允许 `--resume`；`handoff_run` 双向均拒绝 `external`。
- **未知即为 `null`，绝不填零。** 由于没有模型中继，manifest 中的 `model`、`provider`、`effort`、`price_card`、`model_request_limit` 与 `no_progress_timeout_s` 均为 `null`，summary 中的 `served_models`、`token_usage` 与 `model_requests` 也均为 `null`。不存在模型请求预算、实际 served model 核验或模型侧图像裁剪。
- **NI-RUA 依然强制执行：** 单一物理入口、运动准入、分阶段护栏与容差、简报门控、元数据脱敏与类型化回执。**NI-RUA 无法约束：** 智能体自身使用的模型、提示词与上下文、其自带的其他原生工具（智能体保留全部原生能力，包括 shell 以及——除非你在宿主中禁用——联网搜索）、读取 run 目录，或宿主的上下文压缩方式。
- **长会话会更早触发上下文压缩。** 开场 briefing 长达 46,000 至 60,000 字符且工具目录庞大，因此宿主会比托管运行更早触发压缩；每次压缩后务必重新 `run_begin`。run 目录属于观察者记录；若智能体私自读取 run 目录、仿真资产或评测器状态，该次运行即丧失可比性（Actor 技能已写明禁止，但 NI-RUA 无法在 OS 层拦截）。
- 切勿用外部模式运行声称单因素实验收益：模型、宿主、提示词与权限模式都在同时变化。

## 各宿主实测差异事实表

| 事实 | 宿主 | 影响与处理 |
| --- | --- | --- |
| 每个 MCP 工具的嵌套 Schema 关键字都会被剥离（`description`、`minimum`/`maximum`、`minLength`/`maxLength`、`maxItems`、`examples`、`pattern`、`default`、嵌套 `if`/`then`/`oneOf` 等）：在 X5 目录上 101,033 字符的 Schema 仅有 23,439 字符送达模型，stdio 与 HTTP 均如此（在专项验证 0.160.0/0.167.0 上测得，在端到端 0.160.0/0.169.0 上复测同样如此：83,779 字符变为 20,994 字符，纯 enum 节点会被推断补充 `type`，含 `oneOf` 的节点会退化为 `{}`）。托管模式使用 `dynamicTools`，不受此影响。 | Codex | 每个工具描述中自动生成的字段指南保留了参数物理含义、单位与边界，供模型阅读；运行时仍由注册表严格校验。Codex 会在每个工具描述末尾追加 ` This tool is part of plugin \`NI-RUA\`.`（38 字符）。 |
| 未配置逐工具 `output_token_limit` 时，约 12,000 字符以上的结果会被截断。 | Codex | 由 `.codex-mcp.json` 统一配置。 |
| 每张图像后会附带一段本地文本块 `[Image: source: <config dir>/projects/.../*.png]`；超出宿主上限的图像会被重编码（一张 2400×1627 PNG 被转为 437,663 字节 JPEG）；常规相机尺寸的 WebP/JPEG 帧按字节原样送达。 | Claude Code | 已记录的宿主差异；未为此修改图像管线逻辑。 |
| 默认 `tool_output` 截断阈值约 50 KB；超出默认尺寸上限的图像会被重编码。 | OpenCode | 安装器自动写入 `tool_output` 与 `media.image`。 |
| 工具描述默认长度上限为 4096 字符。 | Claude Code | 安装器写入 `CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH`。 |
| 网关重启后没有任何宿主会自动重新拉取工具列表。 | 全部 | 手动重连，见[故障恢复](#recovery)。 |
| 进度通知会重置 Claude Code 的 300 秒空闲计时器，但不会延长 Codex 的 `tool_timeout_sec` 或 OpenCode 的 `timeout.execution`。 | 全部 | 二者均配置为 3600 秒。 |
| 客户端可能通过 `_meta` ID 标识自身（OpenCode `ai.opencode/sessionID`；Codex thread/session/turn ID；Claude Code `claudecode/toolUseId`）。 | 全部 | 仅保留在 MCP 原始日志中；不用于调用者策略。这些字段均不携带调用在模型单步内的位置序号。 |
| 单次模型步骤内的所有工具调用会一次性并发发出，包括有状态调用。 | OpenCode | 外部模式按到达顺序执行，可能与模型生成顺序不同（见[单步内多次调用的顺序](#order-of-one-steps-calls)）。 |

<a id="residuals-documented-not-hardened"></a>
## 已知余项（已记录，未加固）

- 端点不设 Bearer 令牌：静态宿主配置无法预知每局动态令牌，且项目规范禁止把密钥塞进环境变量。`127.0.0.1:8790` 仅通过回环绑定、`Host` 校验以及拒绝外部 `Origin`（与控制台同规则；MCP 宿主不发送 `Origin`）防护。托管模式仍保留每局独立 Bearer 令牌。
- 与回环上的控制台 HTTP API 一样，当前操作系统用户的任意本地进程都可以通过该端口驱动机器人。
- 智能体自身的 shell 可以访问当前用户可见的设备、run 目录与凭据；NI-RUA 不启动任何沙箱来隔离外部宿主。
- 外部智能体的 Git 上下文不受 NI-RUA 控制：若在 Git 仓库内启动宿主，宿主可能会把仓库信息带入自身上下文（Claude Code 会带入分支、Git 用户、状态与近期提交标题；Codex 会读入直到仓库根目录的每一份 `AGENTS.md`）并发送给供应商（Codex 会在每次回合请求元数据中附带仓库根目录、远端 URL 与最新 commit 哈希）。NI-RUA 仅对自身拉起的托管智能体关闭这些行为（见 [Native capability policy](agent-backends.md#native-capability-policy)）；因此当实验可比性重要或不希望仓库信息发往供应商时，请在仓库目录之外启动外部宿主。
- 客户端身份（`clientInfo`、`_meta` ID）完全由客户端自报，仅用于展示。
- 客户端在单次模型步骤中并发发出的有状态调用会按到达顺序执行，可能与模型输出的顺序不一致（见[单步内多次调用的顺序](#order-of-one-steps-calls)）。
- 安装器会修改用户配置文件（Claude Code `settings.json`、OpenCode `opencode.json`）及宿主插件注册表；所有修改都会先打印并记录供 `uninstall` 精确撤销。
- 每个端口仅允许一个外部网关；第二个同端口实例会在启动时被拒绝。
- **默认安装会影响当前用户的全部会话。** 使用默认 `--scope user` 时，该用户在任何项目下的每一次 Claude Code 会话都会加载 `ni_rua` 服务（`alwaysLoad`）且调用机器人工具无需权限弹窗（`permissions.allow`）。网关未运行时这会表现为一个失败的服务条目，网关运行时则会暴露完整机器人工具目录。Codex 通过 `default_tools_approval_mode: "approve"` 具有同样效果。如果你还用 Claude Code 从事其他开发工作，请在 NI-RUA 仓库目录中改用 `--scope project` 安装。安装器默认值保持不变。

<a id="verification-status"></a>
## 验证状态

传输层已使用脚本化模型在以下版本完成验证：Claude Code `2.1.321` / `2.1.325`，Codex `0.160.0`（宿主端到端）、`0.167.0`（专项验证与 tarball 检查）与 `0.169.0`（宿主端到端），以及 OpenCode `2.2.7`。这与 [`agent-plugin/manifest.generated.json`](../../agent-plugin/manifest.generated.json) 中的 `verified_hosts` 一致（由 [`plugin_contract.py`](../../ni_rua/cortex/agents/plugin_contract.py) 生成并由 `cli.mjs` 读取，也是 `doctor` 判定的已验证版本列表）。运维唤醒（Operator wake）已在 Claude Code `2.1.321` 与 OpenCode `2.2.7` 上完成无头验证。2026-11-15 在三款固定版本宿主上各通过了一次短时真实模型会话（[真实模型会话](#real-model-sessions)），并使用 Claude Code `2.1.325` 在真机 SO-101 机械臂上通过了一次短时会话（[真机 SO-101 会话](#real-so-101-session)）；每个宿主仅一次短任务远不足以宣称全面支持。

已有证据及各自证明的范围：

- Python：`mamba run -n ni-rua python scripts/run_tests.py tests/cortex/external/test_external_config_and_session.py tests/cortex/external/test_external_workbench.py tests/packaging/npm/` 覆盖生命周期门控、操作员权限、manifest 空值字段、各类拒绝条件、端点安全（`Origin`、端口、指令）以及三款宿主入口与工具注册表的一致性。
- Node：`node --test agent-plugin/test/*.test.mjs` 在临时 `HOME` 中使用伪造的 `claude` / `codex` / `opencode` 可执行文件验证安装、精确卸载、升级、并发锁与 `doctor`。
- 宿主端到端：`NI_RUA_HOST_E2E=1 NI_RUA_HOST_E2E_LONG=1 PYTHONPATH=tests mamba run -n ni-rua python -m unittest -v tests/cortex/external/test_external_host_e2e.py`（[`tests/cortex/external/test_external_host_e2e.py`](../../tests/cortex/external/test_external_host_e2e.py)）：macOS 上 5/5 全部通过，耗时 563.8 秒；详细结果见下表。其纯 CPU 支撑单测位于 [`tests/cortex/external/test_external_host_e2e_support.py`](../../tests/cortex/external/test_external_host_e2e_support.py)。在线真实模型模式见[真实模型会话](#real-model-sessions)。
- 宿主二进制：`agent-plugin/test/tarball-check.sh` 将打包后的 tarball 安装到隔离 HOME，并在固定版本及用户本机版本的 Claude Code、Codex 与 OpenCode 上校验（含 Codex 0.167.0 在内的五个二进制全部通过，83 项通过、0 项失败，使用专项验证 fixture 运行）。该脚本证明安装、插件与 MCP 入口、技能注册及卸载正确，不证明真实模型运行。
- 专项验证测量（入口语法、HTTP 下的 `alwaysLoad`、默认权限、送达长度、图像、长耗时调用、重启恢复、唤醒）汇总于 [Agent Plugin Spikes](reports/agent-plugin-spikes.md)，每行均标明了实测（run）、二进制核验（binary）或推断（inference）。

**尚未验证的范围：** 交互式 TUI（Claude `/mcp` 重连是通过其 stream-json 控制通道驱动的，尚未实测 Codex TUI 晚启动网关恢复、OpenCode TUI 重载及空闲交互会话唤醒）、Claude Code `2.1.325` 与 Codex `0.169.0` 在短时端到端之外的长调用与唤醒、通过网关运行 X5 目录、超过 332 秒的调用、Linux 宿主机、除各一次短时会话外的真实模型运行，以及除一次短时会话外的真机 SO-101 操作（[真机 SO-101 会话](#real-so-101-session)）。

### 各宿主一致性端到端结果

命令：`NI_RUA_HOST_E2E=1 NI_RUA_HOST_E2E_LONG=1 PYTHONPATH=tests mamba run -n ni-rua python -m unittest -v tests/cortex/external/test_external_host_e2e.py`。结果：macOS 上 5/5 全部通过，总耗时 563.8 秒，使用 SO-101 场景替身与脚本化伪模型，各宿主运行在默认权限模式下且仅应用安装器写入的配置。这属于脚本化模型的传输层证据，不构成对任何宿主的全面支持承诺。完整细节与测量数据见 [Agent Plugin Spikes](reports/agent-plugin-spikes.md#4-host-conformance-end-to-end-suite)。

| 宿主 | 工具列表 | 文本长度 | 图像 | 非只读调用 | 长耗时调用 | 是否发送 Origin | 归档 Store | `doctor` | 唤醒（Wake） | 总耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Claude Code 2.1.321 | 19/19，完整 Schema，无延迟加载 | 含 instructions；briefing 60,492 = 60,492；全文完整 | 5 张字节一致（2 张相机 webp 640×480） | 5 次均执行 | 332 s，回执 completed | 否 | ok | ok，19 工具 | ok，退出 10.0 s 后发起新请求 | 346.5 s |
| Claude Code 2.1.325 | ok | ok | ok | ok | 不适用（短跑） | 否 | ok | 不适用 | 不适用 | 5.9 s |
| Codex 0.160.0 | ok（已知 Schema 投影，+38 字符描述后缀） | ok | ok | ok | 91 s，ok | 否 | ok | ok | 不适用（Codex 沿用 `codex queue`） | 96.0 s |
| Codex 0.169.0 | ok（同上） | ok | ok | ok | 不适用 | 否 | ok | 不适用 | 不适用 | 7.9 s |
| OpenCode 2.2.7 | ok | ok（60 kB briefing 完整；依赖安装器写入的 `tool_output`） | ok | ok | 92 s，ok | 否 | ok | ok | ok，10.0 s | 107.1 s |

“Store ok”表示：每条 `tool_call` 记录均带有 turn id；最终答复等于 `run_finish` 的 `summary` 且来源为 `run_finish_summary`；`agent.client_attached` 记录了自报的 `clientInfo` 且版本与二进制匹配；关闭后的 `summary.json` 正确填入外部模式的 `null` 字段。“Origin sent: 否”表示三款宿主均未向端点发送 `Origin` 头，因此拒绝外部 `Origin` 的规则不会误伤任何宿主。

长耗时调用属于测试专用模式：场景替身在动作完成后人为保持 Worker 响应（[`tests/support/so101_scene.py`](../../tests/support/so101_scene.py) 中的 `execute_hold_s`），因为若把延迟放在运动本身，会触发 `robot_action` 300 秒 `timeout_s` 上限及 SO-101 120 秒 `plan.max_age_s` 过期限制；若在运动前保持 330 秒，回执会报 `batch_plan_expired_real_world_moved`。尚未验证：交互式 TUI、真实模型（见[真实模型会话](#real-model-sessions)）、经由网关的 X5 目录、超过 332 秒的调用、Linux。

<a id="real-model-sessions"></a>
### 真实模型会话

命令（可选触发，会调用计费供应商）：`NI_RUA_HOST_E2E_LIVE=1 NI_RUA_HOST_E2E_LIVE_PROFILE=<profile> NI_RUA_HOST_E2E_LIVE_CLAUDE_MODEL=<id> NI_RUA_HOST_E2E_LIVE_CODEX_MODEL=<id> NI_RUA_HOST_E2E_LIVE_OPENCODE_MODEL=<id> PYTHONPATH=tests mamba run -n ni-rua python -m unittest -v tests.cortex.external.test_external_host_e2e.ExternalHostLiveTest`（[`tests/cortex/external/test_external_host_e2e.py`](../../tests/cortex/external/test_external_host_e2e.py)，“Live mode”）。环境配置与一致性测试相同：隔离 HOME、真实安装器、连接 SO-101 场景替身（无实体机械臂）的真实网关，各固定版本宿主运行于默认权限模式且仅应用安装器设置。宿主直接与真实供应商通信，无任何脚本编排。网关的 `--prompt` 为 [`tests/support/host_e2e.py`](../../tests/support/host_e2e.py) 中的 `LIVE_TASK`：以 `CUBE_COLOUR:` 行报告方块颜色、将末端执行器向上移动 5 mm 并从回执确认，最后调用 `run_finish` 结束。发给宿主的唯一用户消息为 `"Operate the robot through the ni_rua MCP tools."`，其中既未提及 `run_begin` 也未提及具体任务。

2026-11-15 在 macOS 上通过 OpenCode Console 供应商跑出的结果（每个宿主一次会话，方块颜色按宿主随机）：

| 宿主 | 模型 | 首个调用 `run_begin` | `robot_observe` 图像 | `robot_action` 完成 | `run_finish` | 颜色 | 请求数 | 会话墙钟耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Claude Code 2.1.321 | `claude-sonnet-5-5`（served：`claude-sonnet-5-5`） | 是；此前 0 调用、0 拒绝 | 是，2 张 webp 640×480 | 是，实测 dz 0.005 m | 是；summary = 最终答复，回合 `completed` | purple，正确 | 5 | 16.0 s |
| Codex 0.160.0 | `gpt-5.6-terra`（Codex 不报告 served id） | 是；此前 0 次 NI-RUA 调用、0 拒绝 | 是 | 是，实测 dz 0.005 m | 是；同上 | red，正确 | 11 | 39.2 s |
| OpenCode 2.2.7 | `claude-haiku-5-5`（OpenCode 不报告 served id） | 是；此前 0 调用、0 拒绝 | 是 | 是（先 preview 再 execute），实测 dz 0.005 m | 是；同上 | yellow，正确 | 7 | 21.2 s |

- **工具调用序列。**
  - Claude Code：`run_begin`、`robot_observe`、`task_plan`、`robot_action`、`run_finish`。
  - Codex：先执行一条 shell 命令读取已安装的 `ni-rua-robot` `SKILL.md`，随后 `run_begin`、`robot_observe`、`task_plan`、`robot_action`、`task_plan`、`run_finish`。
  - OpenCode：`run_begin`、`robot_observe`、`task_plan`、`robot_action`（preview）、`robot_action`（execute）、`run_finish`。
- **拒绝、权限与报错。** 没有任何宿主出现 `external_briefing_required` 拒绝、权限被拒或宿主报错。Claude Code 运行在 `permissionMode` `default`，Codex 运行在 approval `never` 与 `read-only` 沙箱。没有任何模型尝试调用默认模式会拒绝的原生工具。
- **各宿主报告的用量。**
  - Claude Code：1,022 输出 token、121,833 缓存写入 token 与 419,231 缓存读取输入 token；Claude Code 自估费用 $0.36。
  - Codex：386,880 输入 token（其中 337,793 命中缓存）与 1,668 输出 token。
  - OpenCode：1,427 输出 token、663 推理 token、115,977 缓存写入 token 与 594,864 缓存读取 token；OpenCode 报告费用 $0.055。
- **归档后的最终答复。** 归档保存的是已将短引用（例如 `obs_4`）解析为完整 ID 的 `run_finish` summary。
- **Codex 首次尝试记录。** `gpt-5.3-codex` 仅发了 1 次请求且未调用任何工具：它在文本中写到将先调用 `run_begin`，却反过来向用户询问任务并结束了回合（导致 `codex exec` 退出）。Codex 0.160.0 没有该模型的元数据并警告已回退默认元数据。重试改用二进制内置识别的 `gpt-5.6-terra`。单个存在混杂因素的样本不足以否定 `initialize.instructions` 文案，因此未改动文案。
- **Claude Code 首次会话记录。** 首次会话（6 次请求、20.7 秒、相同工具序列）命中了测试自身的两处校验瑕疵：
  - 将归档最终答复与原始 summary 按字节直接比对，未处理短引用解析；
  - 颜色正则误匹配了正文中的 `"cube colour from"`。

  修复这两项测试断言后重跑得到上表结果。

这组结果仅证明：三款固定版本宿主各自搭配一个模型时，能够依照 `initialize.instructions` 与工具描述通过插件完成一次短时“观察并微移”任务。它不说明操作质量、其他模型表现或更长会话的稳定性。仅 Claude Code 报告了实际 served model ID。

<a id="real-so-101-session"></a>
### 真机 SO-101 会话

2026-11-15（UTC 11:53）在 macOS 上完成的一次真机 SO-101 机械臂会话。配置如下：

- **网关。** `--agent-backend external`，从该分支直接运行。配置基于常规 SO-101 运行配置：将 agent 切换为 `external`，移除远程 VGGT 服务，并关闭依赖 VGGT 的感知特性（`external_perception`、`object_table`、`scene3d`、`vggt_height`）。机台刚刚完成重新标定。
- **宿主。** 隔离配置目录中的 Claude Code `2.1.325`，使用默认参数通过 `ni-rua-plugin install claude` 安装插件。
- **模型。** 经由 OpenCode Console 调用 `claude-sonnet-5-5`。
- **提示词。** 唯一用户消息直接给出了任务：描述桌面、将夹爪竖直上移 2 cm、核验并移回原位。未提及 `run_begin`。

实测过程：

- **工具序列。** `run_begin`（首条调用，无拒绝）、`robot_observe`、`task_plan`、`robot_action`、`robot_action`、`robot_action`、`run_finish`。三次 `robot_action` 依次为上移动作的 preview、其 execute，以及使用 `execute_if_clean` 的下移回程；上移动作采用了 `speed_preset` `slower`。
- **上移动作。** 回执 `completed`，残差 1.5 mm。实测 TCP z 从 0.0822 m 升至 0.1034 m，横向漂移保持在 1 mm 以内。
- **下移回程。** 回执 `completed`，残差 0.2 mm。
- **回合结束。** `run_finish` 以 `completed` 关闭回合。模型对桌面的描述与实物一致：三块方块、一把螺丝刀、一个盒子和一台笔记本电脑。
- **开销。** 8 个模型回合，33.7 秒，Claude Code 自估约 $0.43。
- **归档记录。** manifest 中 `agent_backend.kind = "external"`、`information_boundary = "mcp_boundary_only"`，`model_request_limit` 与 `no_progress_timeout_s` 均为 `null`；`summary.json` 中 `served_models`、`token_usage` 与 `token_accounting` 均为 `null`。
- **关闭。** 网关通过 SIGINT 干净退出。

这只是一次小幅慢速运动的短时会话，仅证明外部插件链路在实体硬件上跑通，不构成操作能力评判。

## 运维唤醒方案

见 [Experiment Completion Hook](experiment-completion-hook.md#waking-the-agent-that-launched-the-queue-per-host)：Codex 沿用既有完成回调 Hook；Claude Code 与 OpenCode 则将队列作为智能体自身的后台 shell 任务启动（仅完成无头验证）。

## 暂缓项（不在当前版本中）

stdio 桥接、服务发现记录、`ni_rua_status` 工具、工具描述拆分、调用者身份与单一驱动者规则、基准风格的外部运行、`--wake-command`、PreCompact Hook 与对应路由、面向外部智能体的模型中继、外部模式下的证据辅助工具（evidence helpers）以及 `ni-rua-library` 命令行入口。更长的真实模型任务、更多真机 SO-101 会话与正式发布 npm 包待维护者后续推进。
