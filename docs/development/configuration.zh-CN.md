[English](configuration.md) · [简体中文](configuration.zh-CN.md)

# 统一配置与功能开关

`ni_rua.core.config.Config` 是新运行的统一配置入口。仓库中的默认配置文件为 `configs/ni-rua.json`，完整的带注释示例见 [`configs/ni-rua.example.json`](../../configs/ni-rua.example.json)。在实际使用中，你只需要指定希望覆盖的字段；所有省略的字段均继承自 `Config` 类的默认值。由于 JSON 格式不支持注释，本文档作为所有配置字段、依赖关系以及历史环境变量映射的权威参考。

构建 wheel 包时，`scripts/build_package.py` 会将仓库默认文件复制到包中作为 `ni_rua/default_config.json`。在仓库检出目录之外运行时，已安装的 wheel 包会回退使用该打包副本，从而确保无论从源码还是 wheel 包启动，新实验都共享完全一致的默认值。显式的 `--config` / `NI_RUA_CONFIG` 覆盖仍具有更高优先级，而 `Config.from_dict()` 在加载旧归档时会保留历史默认值。

<a id="usage-and-override-precedence"></a>
## 使用方法与覆盖优先级

下文所有项目命令均在 Miniforge 的 `ni-rua` 环境中运行；关于环境创建以及已解析的 Python 解释器和 Codex SDK/CLI 版本的核验，请参阅[运行指南](running.zh-CN.md#环境准备)。

```sh
# 使用仓库默认配置运行
mamba run -n ni-rua python -m ni_rua.apps.gateway --run-dir runs/my-run --gpu 4

# 使用自定义实验配置；CLI 参数仅对本次运行覆盖相应设置
mamba run -n ni-rua python -m ni_rua.apps.gateway --config configs/my-experiment.json \
  --run-dir runs/my-run --gpu 4 --no-enable-wait

# 校验并打印完整的实际生效配置，不启动模型、GPU 或后台服务
mamba run -n ni-rua python -m ni_rua.apps.gateway --config configs/my-experiment.json --print-config

# 将所有统一配置入口指向特定配置文件
export NI_RUA_CONFIG=/absolute/path/ni-rua.json
```

配置层的合并顺序如下：
1. `Config` 类的硬编码默认值
2. 兼容的历史环境变量
3. 仓库默认文件 `configs/ni-rua.json`（或已安装 wheel 包中的内置回退配置）
4. 本地仓库覆盖文件 `configs/ni-rua.local.json`（若存在）
5. 显式指定的 JSON 配置文件（`--config` 或 `NI_RUA_CONFIG`）
6. 显式的命令行 CLI 参数

嵌套对象按字段逐层合并，因此仅覆盖 `features` 的实验配置仍会继承默认模型和本地 Codex 主目录，而显式指定旧模型设置的配置仍可覆盖较新的默认值。`--config` 优先于 `NI_RUA_CONFIG`；当两者均省略时，使用合并后的公开与本地默认值。每个配置文件在加载时都会受到严格校验，文件内部的相对路径均相对于该配置文件所在目录进行解析。未传入的 CLI 参数绝不会覆盖文件中的配置值。任何未知的字段名称、无效的类型、非有限数值以及不支持的枚举值都会在任何服务、模型或机器人 Worker 启动前被直接拒绝报错。

JSON 文件中的相对路径相对于包含该文件的目录解析，而 CLI 与历史环境变量路径相对于当前工作目录解析（两者均会展开 `~`）。保存的生效配置快照中存储解析后的绝对路径。最小有效 JSON 文件为 `{"version": 1}`，因为 `Config` 类提供了所有字段的默认值。由于 `configs/ni-rua.example.json` 中使用了 `~` 路径，将其复制到其他机器时请检查其 `backend` 和 `policy` 部分。

数组字段会整体替换原有列表——例如，传入 `--tag trial` 会替换配置文件中定义的标签，而 `--clear-tags` 会清空该列表。JSON 中的 `null` 仅对 Schema 中显式允许 `null` 的可选字段有效；严禁使用 `null` 来禁用布尔标志（请使用 `false` 代替）。

例如，以下配置文件保留了默认的标定观测配置，禁用了 `robot_wait` 和 $\pi_{0.5}$，并启用了跟踪保护和紧凑 Actor 载荷：

```json
{
  "version": 1,
  "features": {
    "enable_wait": false,
    "pi05": false,
    "tracking_guard": true,
    "actor_payload_profile": "compact_v1"
  },
  "images": {
    "reuse_exact_images": false
  }
}
```

新运行默认启用 `robot_wait`。禁用它仅会移除模型调用 `robot_wait` 的能力，而不会改变观测是否会推进物理仿真时间（`robot_observe`、模型思考时间以及系统墙钟 sleep 绝不会推进暂停的仿真）。

Python 调用方共享同一个不可变的 `Config` 对象，因此各个模块绝不会重新读取环境变量或发生默认值分歧：

```python
from ni_rua.core.config import Config

config = Config.load("configs/my-experiment.json").validate_launch()
assert config.features.enable_wait is False

# 返回新的 Config 实例而不修改原始对象；显式的 False 绝不会被当作“已省略”
candidate = config.with_overrides({"features": {"enable_wait": True}})
snapshot = candidate.to_dict()

# 从归档字典加载时不会重新读取当前环境变量或默认文件
restored = Config.from_dict(snapshot)
```

`Workbench`、`Harness`、`RobotBackend`、`CodexSession` 以及实验库均接受 `config=` 参数，同时为了向后兼容保留了现有的显式构造关键字参数。加载配置并执行 `validate_launch()` 不会启动任何外部服务。

<a id="run-and-model-settings"></a>
## 运行与模型设置

| 字段 | 默认值 / 取值范围 | 用途与约束 |
| --- | --- | --- |
| `run.task` | `stack_bowls` | 原生任务名称。 |
| `run.seed` / `run.layout_id` | `0` / `0` | 非负整数，指定初始环境条件。 |
| `run.gpu` | `4` | 机器人仿真器和本地策略 Worker 的 GPU 编号；评测队列会按用例分配实际 GPU。 |
| `run.run_dir` | `null` | 新运行的显式目录；为 `null` 时由实验库自动分配。 |
| `run.experiment_root` | `~/exp/ni-rua` | 自动分配运行目录时使用的根归档目录。 |
| `run.category` / `run.title` / `run.tag` | `main` / `null` / `[]` | 归档分类、人类可读标题和标签列表。 |
| `run.port` | `0` | 仅回环绑定的 HTTP 端口；`0` 表示启动时自动选择临时空闲端口。 |
| `run.prompt` | `null` | 开发运行的自定义提示词；为 `null` 时使用默认接口检查提示词（官方基准测试始终使用原生公开任务指令）。 |
| `run.benchmark` | `false` | 启用官方基准生命周期：原生任务指令、同线程轮次延续、独立原生评测及终态清理。 |
| `run.exit_after_turn` | `false` | 首轮完成后退出；基准测试运行会忽略此项并持续运行直到满足协议终止条件。 |
| `run.resume` / `run.replay` | `false` / `false` | 互斥标志，要求提供已存在的 `run_dir`；官方基准测试仅允许全新运行。 |
| `run.max_actions` | `null` | 开发检查的高层动作上限；官方 `--benchmark` 运行中拒绝此设置。 |
| `run.wall_budget_s` | `null` | 开发运行的全回合墙钟预算（秒）；官方 `--benchmark` 运行中拒绝此设置。 |
| `run.task_notes` / `run.task_notes_source` | `null` / `null` | 附加在默认操作经验之后的自定义公开基准笔记与来源；必须成对作为非空字符串提供（自定义正文最多 8,000 字符）。 |
| `model.name` / `model.effort` | `gpt-6.1-sol` / `high`（新运行的仓库默认值） | 主 Actor 模型标识符与推理努力等级，彼此独立配置。 |
| `model.verbosity` | `high`（仓库默认；`'low'`, `'medium'`, `'high'`） | 主 Actor 散文输出冗长度等级（`--verbosity`）；可选择 `'low'` 或 `'medium'` 来精简非推理正文生成，同时保持 `model.effort` 推理深度不变。 |
| `model.codex_home` | `~/.codex`（新运行的仓库默认值） | 读取本地 Codex 供应商路由和凭据的宿主机目录；宿主机扩展不会被继承，会话在单次运行专属私有目录中运行，操作系统隔离由 `features.sandbox_enabled` 控制。 |
| `model.price_card` | `null` | 可选的显式定价卡 JSON 文件路径；成本估算绝不会修改原始 Token 证据。 |
| `model.native_memories` | `true`（新运行的仓库默认值） | 控制 Codex 与 Claude Code 的私有原生记忆（每运行专属私有目录；设为 `false` 时关闭 Codex 原生记忆并在 Claude Code 设置 `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`），不修改宿主机配置或 NI-RUA 的 `operation_memory`。 |
| `model.transport_recovery.max_rounds` | `10` | 明确可重试错误在 12 次自动重试耗尽后，或在上游 `transport_incomplete` 断开后的最大冷却重发轮数（每轮 1 次 HTTP 尝试）；`0` 禁用恢复，最大 `20`。 |
| `model.transport_recovery.initial_delay_s` / `backoff_multiplier` / `max_delay_s` | `60` / `2` / `600` | 轮次间的冷却时间表：`60`, `120`, `240`, `480`, `600`... 秒。 |
| `model.transport_recovery.max_episode_delay_s` | `3600` | 整个回合（单次代理生命周期）内累积的传输恢复冷却时间上限（秒）；一旦超出，运行正常终止。 |
| `model.transport_recovery.transport_incomplete` | `true` | 是否也从缺乏终态响应的上游传输断开中恢复（例如 `IncompleteRead`）；`false` 仅恢复明确可重试的供应商错误。 |

配置文件绝不存储 API Token 或复制供应商机密。供应商端点、身份验证方法和上游 URL 继续由 `model.codex_home` 内的原生 Codex 配置和凭据文件管理，环境变量中的供应商 API Key 继续按常规方式解析。

在本地工作站上，你可以使用 Git 忽略的 `configs/ni-rua.local.json` 指向专用的供应商主目录——例如 `{"version": 1, "model": {"codex_home": "~/.config/ni-rua/providers/my-provider"}}`。在 `configs/ni-rua.local.json` 中仅保留目录路径，并将供应商的 `config.toml` 和包含密钥的 `auth.json` 放置在该外部目录中，目录权限设为 `0700`，文件权限设为 `0600`（包括 `configs/ni-rua.local.json` 本身）。严禁在统一 JSON 配置文件内写入原始 API Key，因为解析后的配置会被写入清单、实验归档和网页报告中。使用专用的 Codex 主目录可以防止供应商变更影响交互式 Codex CLI 会话或已经在跑的队列，并且 `configs/ni-rua.local.json` 绝不会打包进 wheel 中。显式的 `--config` 仍可覆盖 `configs/ni-rua.local.json`，且 `Config.from_dict()` 和已冻结的队列均不会读取本地覆盖文件，从而确保本地默认值变更时历史快照绝不会被重新解释。

`configs/ni-rua.json` 和 `configs/ni-rua.example.json` 对新运行均显式选择了 `gpt-6.1-sol`、`high` 和 `~/.codex`，而本地默认值或显式配置可指向其他隔离的主目录。启动时，NI-RUA 从所选主目录的 `config.toml` 中读取当前活跃的供应商，并从 `auth.json` 或供应商声明的认证设置中读取凭据，使真实凭据严格保存在宿主机中转代理内部。Actor 的单次运行专属私有主目录接收占位凭据，不从源目录复制真实的 API Key、`.credentials.json` 或宿主机扩展，且绝不写回源目录。严禁将机密提交到受版本控制的配置文件、清单、命令行参数或日志中，也严禁将源 Codex 主目录提交到 Git。`model.price_card` 默认为 `null`，以免意外套用其他供应商或模型的定价规则。

主 Actor 的模型名称和推理努力等级严格由统一配置管理，不受源 Codex 主目录内部同名首选项的覆盖。`Harness` 固定了 `web_search = "disabled"`、`service_tier = "default"` 以及 `Responses` 传输协议；原生供应商标识符 `ni_rua_model_proxy` 指向本地计费中转代理，该代理将请求转发到选定本地 Codex 供应商定义的外部上游。启用时，可选的辅助视觉审查模型复用为主会话准备的供应商、中转代理和认证，在其独立的隔离主目录与线程中运行 `gpt-6.1-sol` / `high`，不再重复读取源 Codex 主目录。

新运行的默认模型选择仅存在于仓库配置文件中；`Config.from_dict()` 不会查阅 `configs/ni-rua.json`，因此静态的 `Config()` 默认值（`gpt-6.1-sol` / `high` / `~/.codex`）与 `configs/ni-rua.json` 完全一致；归档始终在 `effective_config.json` 中显式记录模型，重放与比对必须读取该文件。带有冻结 `ni_rua_config` 的评测队列继续使用其冻结快照。从旧清单模板启动新实验时，请显式传入 `--config configs/ni-rua.json` 或重新生成清单。请注意，队列冻结的是 `codex_home` 路径而非内联机密文件，因此每个用例在启动时都会读取该路径下的供应商文件；若未来运行需要切换供应商，请创建新的供应商主目录，而不是直接修改正在运行的队列所引用的目录。历史遗留的 `~/.config/ni-rua/experiments.json` 文件仅支持 `codex_home` 和 `price_card` 路径；未使用定价卡时请完全省略 `price_card`（而不是将其设置为 JSON `null`）。

`manifest.ni_rua_config` 存储解析后的生效配置快照；即便历史快照仅包含部分字段，缺失的字段也会按照历史默认值进行解释。新运行可通过 `--config` 提供部分配置覆盖，该选项会先加载当前默认值并冻结完整的解析后快照。

<a id="code-execution-sandbox"></a>
## 代码执行沙箱

`features.sandbox_enabled` 是一个全局布尔开关，在 `Config` 类和 `configs/ni-rua.json` 中均默认为 `false`；显式设为 `true` 即可启用操作系统级隔离。它同时控制各个后端主智能体会话的 Linux bubblewrap 命名空间（Codex 及其原生终端、子智能体、Shell 命令和记忆固化；Claude Code 和 OpenCode 及其原生工具和子智能体）、辅助 Codex 证据智能体的 bubblewrap 命名空间，以及 `program_run` 的 Landlock/seccomp 隔离。其在基准测试中的目的是防止模型读取 RoboDojo / 仿真器源码（防止奖励作弊，即 reward hacking）；公开的 Harness 源码在 `/harness` 路径下保持可读。因此各后端跨 Arm 的 RoboDojo 比对均在开启沙箱的情况下运行；各后端的挂载详情见 [Agent 后端](agent-backends.md#sandbox-featuressandbox_enabled-all-backends)。当设为 `false` 时，NI-RUA 既不探测也不加载这些隔离机制，因此宿主机无需安装 `bwrap`、`libseccomp`，也不要求非特权用户命名空间和 Landlock（标准运行时依赖仍需满足）。

例如，将以下内容保存为 `configs/no-sandbox.json`，以便在本地诊断运行中关闭沙箱：

```json
{
  "version": 1,
  "features": {
    "sandbox_enabled": false
  }
}
```

```sh
mamba run -n ni-rua python -m ni_rua.apps.gateway --config configs/no-sandbox.json --print-config
mamba run -n ni-rua python -m ni_rua.apps.gateway --config configs/no-sandbox.json \
  --run-dir runs/no-sandbox-trial --gpu 4
```

禁用沙箱（`false`）时，每次运行仍使用单次运行专属的隔离 `HOME` / `CODEX_HOME`、本地模型中继和占位凭据，不继承宿主机的 MCP 服务、插件或技能，且 `program_run` 仍保留其文件描述符代理、工具权限检查、动作审查、预算和资源限制（`RLIMIT`）。然而，执行的代码将拥有启动用户的宿主机文件系统和网络权限——私有目录和占位 Token 并不能构成操作系统级安全边界，形式化观测限制仅在应用层执行。因此，未开启沙箱的运行无法在操作系统层面证明宿主机文件、凭据、评测器或观察者产物是不可访问的；请始终记录运行是否启用了沙箱。

禁用沙箱时，公开 Harness 源码的只读副本存放在 `<run>/private/actor-source`，而不是绑定挂载在 `/harness`。`sandbox_enabled` 设置会持久化保存在生效配置快照中；修改磁盘上的配置文件绝不会热切换正在运行的实验或已冻结的队列。拒绝跨不同沙箱模式恢复会话（请启动新运行），缺少隔离标记的历史完全访问权限会话亦不可就地迁移。

下文的隔离、挂载与受控网络规则在显式启用 `features.sandbox_enabled=true` 时生效。在 `CodexSession._prepare()` 期间，`Harness` 在 Linux bubblewrap 沙箱内启动整个原生 Codex app-server。隔离的 mount、PID、IPC 和网络命名空间覆盖主线程、子智能体、Shell 工具和记忆固化，在当前用户的 UID/GID 下运行，并清除所有 Linux capabilities。`bwrap`、`sudo` 和 `setpriv` 从当前 `PATH` 和平台默认 `PATH` 中自动发现。NI-RUA 优先使用非特权用户命名空间；仅在宿主机内核禁用非特权用户命名空间时，才回退到使用 `sudo -n` 创建命名空间并立即通过 `setpriv` 放弃特权。若沙箱初始化失败，启动过程会直接中止，绝不静默回退到未沙箱化的宿主机执行。启动门禁（`factory.backend_requirements`）对 `codex`、`claude_code` 和 `opencode` 一视同仁地接受沙箱；若宿主机缺少 bubblewrap，则会在会话启动时、原生智能体进程创建前被拒绝（Codex 报错 `codex_sandbox_wrap_not_found_on_path_or_os_defpath`，Claude Code 和 OpenCode 报错 `agent_sandbox_bubblewrap_not_found_on_path_or_os_defpath`）。在网关用户拥有附加组的宿主机上，非特权路径无法在用户命名空间内丢弃这些附加组，因此也会采用有 root 权限的 `sudo -n` 回退。

运行目录、源码快照路径和原生 Codex 二进制文件分别从运行配置、已安装的 `ni_rua` 包以及锁定的 `openai-codex-cli-bin` 包中解析，不硬编码用户名、检出路径或 Conda 安装目录。IPC 套接字使用 `XDG_RUNTIME_DIR`（未设置时使用 Python 的平台临时目录）；运行时目录必须由当前用户拥有、权限为 `0700`，且路径足够短以符合 Unix 域套接字限制（Linux 上 107 字节，macOS 上 103 字节）。若缺少 `XDG_RUNTIME_DIR` 且平台临时目录过长（例如 macOS 上带有 8 位以上 UID 的每用户目录），Codex 会改用 `/tmp/ni-rua-ipc-<uid>`，并具有相同的归属和 `0700` 检查；显式设置但过长的 `XDG_RUNTIME_DIR` 会被直接拒绝（`codex_ipc_path_too_long`）。系统 `python3` 从平台默认 `PATH` 发现，其标准库路径从此解释器查询，所需的 ELF 共享库通过 `ldd` 解析——NI-RUA 绝不会将整个 `/usr`、Conda 环境或宿主机的 `site-packages` / `dist-packages` 整体挂载进去。诸如 `/harness`、`/tmp` 和 `/usr/bin` 之类的路径均指沙箱内部的挂载点，而非宿主机配置路径。

沙箱内部根据 [`system_runtime.py`](../../ni_rua/cortex/session/system_runtime.py) 中的显式白名单暴露系统工具，涵盖标准 Shell 工具、文件操作、文本搜索与比较（`grep`、`diff`、`cmp` 等）、文本处理、校验和、归档以及本地 Git；`python` 指向发布的系统 `python3`，并发布 C.UTF-8 语言环境文件，以便在设置 `LC_ALL=C.UTF-8`（原生智能体运行方式）时运行 Shell 命令不会打印语言环境警告，且文本工具支持多字节字符。对于 Codex，`rg` 是锁定的 Codex 发行版自带的版本；其他智能体后端则从活跃环境中获取 `rg`。宿主机上未安装的可选工具会直接忽略，且绝不会向列表中添加网络客户端、提权与会话工具、其他解释器或系统包管理器（`EXCLUDED_COMMANDS`；Python 包通过 `python3 -m pip` 和私有代理安装）。相同的用户空间环境会发布到 Codex、Claude Code 和 OpenCode 的沙箱中。在沙箱内部，Python 可以通过私有 PyPI 代理安装科学计算包，而宿主机的其他二进制文件、编译器工具链以及 `/usr` 中的系统包管理器则不对外暴露。

在外层 bubblewrap 容器内部，Codex 以 `approval_policy="never"` 和 `sandbox_mode="danger-full-access"` 运行，从而使原生工具可以在**外层操作系统沙箱内部**自由运作。操作系统隔离绝不依赖模型自觉遵守提示词指令。生成的原生配置包括：

```toml
web_search = "disabled"
approval_policy = "never"
sandbox_mode = "danger-full-access"
model_verbosity = "high"
service_tier = "default"
project_root_markers = []   # 不继承运行目录上层 git 仓库的 AGENTS.md（信息边界）

[features]
shell_tool = true
unified_exec = true
memories = true
multi_agent = true
multi_agent_v2 = true
hooks = true
goals = true
browser_use = false
browser_use_external = false
computer_use = false
apps = false
plugins = false
artifact = false
image_generation = false
network_proxy = false
code_mode_host = false
fast_mode = false
```

沙箱内部，`HOME` 和 `CODEX_HOME` 指向本次运行的 Actor 工作区与私有会话目录，系统运行时依赖只读挂载，`/tmp` 属于沙箱私有。`/harness` 暴露了公开 Harness、`ToolSpec`、动作和程序接口的只读快照，具体范围由 `ni_rua/cortex/session/actor_source.py` 中的白名单定义。它排除了 RoboDojo、后端任务实现、评测器、参考策略、`.git` 元数据、历史补丁、实验数据库、宿主机主目录以及同级仓库。`/harness` 仅用于检查接口契约；绝不会为了满足随意 import 而挂载完整代码检出。

隔离的网络命名空间无法直接访问宿主机网络或公网。所有网络流量均经由单次运行专属的 Unix 域套接字转发至本地计费模型中继、只读账户中继以及受控的 Python 包代理。该包代理严格仅允许发往 `pypi.org` 和 `files.pythonhosted.org` 的 HTTPS 请求，从而使 `python3 -m pip install <package>` 安装到 Actor 的私有工作区和缓存中；任意网站、宿主机回环服务、SSH、Git 远程仓库以及其他包镜像均不可访问。Web 搜索、浏览器工具以及外部应用/MCP 集成均被禁用，模型中继会拒绝服务端 Web 搜索或远程工具调用。

宿主机供应商路由和凭据由 `Harness` 读取并保存在宿主机中继内部；沙箱内的 Codex 进程仅接收本地中继 URL 和占位凭据。支持的宿主机身份验证方法包括 `auth.json` 中的 API Key 或 ChatGPT 访问令牌、供应商 `env_key` 变量以及 Bearer 令牌，自定义认证头严格限定为 `Authorization` 和 `ChatGPT-Account-Id`。真实的 `auth.json` 文件、`.credentials.json`、MCP 服务器、插件、技能以及宿主机对话历史/记忆绝不复制到运行中，源 Codex 主目录（或 CC Switch 配置）亦绝不被修改。请注意，ChatGPT 访问令牌在宿主机网关启动时仅读取一次——若令牌在运行途中过期，沙箱无法自动刷新宿主机登录会话。

动态机器人工具继续使用直接工具目录（`code_mode=false` 且 `code_mode_host=false`），因此图像输出在 `Responses` 中以结构化图像项交付。模型名称和推理努力等级来自统一配置，而原生后台请求（如记忆固化或会话命名）则在其自身的实际角色和模型下独立记账。`model.native_memories=true` 启用每运行原生的文件记忆（设为 `false` 可禁用），这与 NI-RUA 的 `operation_memory` 和 `grasp-memory` 完全独立。

独立的辅助证据智能体在其专属的独立 bubblewrap 命名空间中运行，拥有空工作区与专用 IPC 套接字——既不挂载主 Actor 的工作区，也不挂载 `/harness` 或 PyPI 代理——并且仅接收选定的证据载荷，所有工具均被禁用。`program_run` 保留其专用的文件描述符代理与 Landlock/seccomp 过滤器，将 Python 导入限制在标准库与获准的科学计算包内，同时禁用宿主机 `.pth` 文件以及未获准的 Conda/`site-packages` 目录。来自程序的所有物理机器人动作依然必须经过单一的 `Harness.execute` 准入点。

<a id="feature-flags"></a>
## 功能开关

| 字段 | 新运行默认值 | 受控行为 |
| --- | --- | --- |
| `features.sandbox_enabled` | `false` | 全局开关，控制主智能体（所有后端）、辅助 Codex 证据智能体的 bubblewrap 命名空间，以及 `program_run` 的 Landlock/seccomp 隔离（仅在显式设为 `true` 时启用）。为 `false` 时，代码以启动用户的宿主机文件/网络权限运行，同时保留应用层工具校验、动作审查和预算；详见[代码执行沙箱](#code-execution-sandbox)。 |
| `features.observation_profile` | `rgb_proprio_calibrated_v1` | 允许 RGB 图像、本体状态、精确相机/机器人标定以及公开机器人模型。亦支持旧版 `rgb_proprio_v1`，而启用原生深度采集和数值深度工具则需显式指定 `rgbd_proprio_calibrated_v1`。 |
| `features.motion_review` | `true` | 启用建议性动作预览、缓存计划执行以及可选的抓取规划/记忆工具；数值风险诊断不会阻断执行。在未标定的旧版观测配置下会自动禁用。 |
| `features.default_task_notes` | `true` | 向新基准测试运行注入带版本的操作经验笔记；禁用时仍可使用自定义 `run.task_notes`，非基准冒烟测试不会注入默认笔记。 |
| `features.self_briefing` | `false` | 可选的动作前模型调用，在首轮前撰写简短操作笔记（`ni_rua/cortex/context/self_briefing.py`）；默认禁用（`--[no-]self-briefing`）以节省约 35 秒的首动作前轮次延迟。 |
| `features.startup_inline_grounding` | `true` | 将初始 `obs_1` 测量的机器人状态、`cam_head` 全景 RGB 以及 `obs3d`（`object_table` + `scene3d` 句柄）直接内联嵌入开局首轮（`--[no-]startup-inline-grounding`），消除单独的启动 `robot_observe` 调用。 |
| `features.startup_grounding_views` | `overview` | 选择 `startup_inline_grounding=true` 时嵌入的相机视角（`'overview'` 仅嵌入 `cam_head`；`'all_cameras'` 嵌入 `cam_head` 与手腕视角）。 |
| `features.action_trajectory_delivery` | `automatic` | 动作后轨迹叠加图策略（`'automatic'` 或 `'diagnostic'`）：`'automatic'`（默认）在执行动作后附带轨迹叠加图；`'diagnostic'` 在正常 `completed` 批次上返回紧凑 3D 数值轨迹摘要且包含 `0` 张轨迹图，并在停止/异常时附带单面板 `640×480` 复合轨迹叠加图。 |
| `features.budgeted_action_reply` | `true` | 在所有智能体后端上按决策价值编排每条 Actor 动作回复（执行结果、场景变化、动作后观测、逐阶段细节、轨迹反馈）。若智能体宿主机对单次工具结果有已知文本限制（Codex 基于锁定模型目录为 48,000 UTF-8 字节，Claude Code 为 500,000 字节；OpenCode 和外部宿主机未知），且回复超出限制，则按价值最低优先原则将内容块替换为指向返回该内容的既有调用的引用。`false`：不编排、不替换。详见 [Actor 回复投递](tool-output-contract.md#actor-reply-delivery)（`--[no-]budgeted-action-reply`；队列用例选项）。 |
| `features.vision_compare` | `true` | 启用 RGB 特征对比（`vision_compare`）以及依赖它的 RGB 技能；禁用它不影响显式启用的 RGB-D 技能。 |
| `features.robotic_view` | `false` | [手腕视角求解器](robotic-view.md)：从调用方指定的相机视点计算 `link6` 位姿；需要已标定观测，仅返回预览参数而不移动机器人。 |
| `features.stereo_guide` | `false` | [双目极线引导](stereo-guide.md)：将源像素的极线渲染到另一个同步相机视图上；需要无畸变的已标定针孔相机，不进行自动特征匹配或三角化。 |
| `features.depth_auto_feedback` | `false` | 针对注册目标的动作前后自动数值深度校验；需要 RGB-D 配置，不增加额外拍照或物理动作。 |
| `features.depth_skills` | `false` | 有界抓取并提升以及放置释放后退深度技能（需要 RGB-D 配置）。由于 `skill_run` 执行工具已在 2026-10 重构中移除，该标志当前仅影响 `depth_query` 中 `skill_help` 目录的 `enabled` 字段。 |
| `features.depth_event_payload` | `false` | 允许具有可靠深度测量的例行中间动作以数值摘要替代内联图像，同时仍归档完整图像；需要 RGB-D 配置和 `depth_auto_feedback`。 |
| `features.enable_wait` | `true` | 暴露 `robot_wait` 以显式推进仿真 1–50 个原生步长并返回时序图像帧。 |
| `features.pi05` | `false` | 暴露与 ARX X5 适配的 $\pi_{0.5}$ 短视距策略前缀工具。 |
| `features.visual_review_model` | `null` | 默认禁用辅助视觉/操作校验；显式设为 `gpt-6.1-sol` 可启用 `vision_review`，共享回合的模型请求预算。 |
| `features.tracking_guard` | `false` | 当持续的关节跟踪误差超出阈值时中止末端执行器运动，返回部分执行证据。 |
| `features.joint_replan` | `true` | 搜索更近的有效 $2\pi$ 等价关节目标，仅在经校验且原生步长更短时替换轨迹。 |
| `features.motion_execution_mode` | `resampled` | 官方 Actor 和基准测试运行使用 `resampled` 统一动作；`feedback` 严格限定于受信任的历史固定输入诊断，要求 `motion_review=false` 和 `tracking_guard=false`，在基准测试启动前会被拒绝。 |
| `features.default_motion_speed_scale` | `1.0` | 阶段未指定速度预设或显式倍率时的默认手臂运动速度倍率（ARX X5 上为 `0.25`–`2.0`；不同本体发布自己的范围，SO-101 为 `0.25`–`1.0`）；显式阶段设置优先于此默认值，夹爪速度独立默认为 `1.0`，`feedback` 模式仅允许 `1.0`。 |
| `features.airborne_handoff` | `true` | 允许空中双臂交接模板；`--no-airborne-handoff` 恢复需要中间支撑面的 `surface_only` 硬约束。两种模式下双臂同步运动、独立推进和手臂队列均保持可用。 |
| `features.handoff_preference` | `surface` | `--handoff-preference surface/airborne` 在允许空中交接时选择交接引导偏好。默认组合（`airborne_handoff=true` + `handoff_preference="surface"`）解析为 `prefer_surface`：模型自主选择，推荐物理支撑面而非强制。`true + "airborne"` 解析为 `prefer_airborne`。旧版布尔配置省略 `handoff_preference` 时，`true` 恢复旧版 `prefer_airborne`，`false` 恢复 `surface_only`；历史归档同样保留其记录的策略。设置 `airborne_handoff=false` 且 `handoff_preference="airborne"` 会被拒绝报错。 |
| `features.actor_payload_profile` | `lean_v2` | 面向模型的工具回复展示。`lean_v1` 剥离例行存储元数据和可查询的长引用，不舍入测量值；`lean_v2`（默认）额外以 6 位小数打印浮点数，并在存储证据的完整读取（`history_read_trace`, `detail="full"`）之外，省略原生 Worker 帧 ID、空残差和 `program_run` 描述语句。亦支持 `full_v1` 和 `compact_v1`。程序接口、类型化数据和原始归档证据始终保留完整数据；详见 [Actor 信息边界](actor-information-boundary.md) 与 [有效数字表](tool-output-contract.md#reply-shaping-order-and-the-digits-each-number-ends-with)（`--actor-payload-profile`；队列用例选项）。 |
| `features.reply_same_as` / `features.reply_float32` | `true` | Actor 回复的无损紧凑压缩：同一文本项内部重复的值仅打印一次，后续副本通过 JSON Pointer（`same_as`）指向它；float32 精确数值以保持相同 float32 的最短小数打印。详见 [Actor 回复投递](tool-output-contract.md#lossless-compaction-reply_same_as-reply_float32)（`--[no-]reply-same-as`, `--[no-]reply-float32`；队列用例选项）。 |
| `features.reply_static_once` | `true` | 在例行 Actor 回复中省略固定解释性数值（预算和步长语义、`collision_scope`、固定 `trajectory_feedback` 范围说明等）；这些事实在 `robot_action` 描述中一次性声明（仅包含适用于当前机器人的内容），报告失败、护栏暂停、滑脱或未知结果的回复会保留它们（模型请求的检查点停止属于例行情况）。亦将回复中的“在此读取其余内容”条目重写为单个 `"detail": {"tool", "arguments"}` 字段，仅命名宿主机已加载的工具。详见 [Actor 回复投递](tool-output-contract.md#fixed-notes-stated-once-and-one-detail-entry-reply_static_once)（`--[no-]reply-static-once`；队列用例选项）。 |
| `features.action_reply_profile` | `summary_first` | `summary_first` 为每条执行回复提供一种固定结构：内联展示执行结果、每个阶段条目（无 `grasp_contact_feedback` 的干净完成阶段）、场景变化、动作后观测和 `compact_3d_trajectory`；通过引用提供干净完成阶段的实测/规划关节采样、通过的运动安全行和 `grasp_contact_feedback`；在失败、护栏暂停或滑脱后，受影响阶段周围的实测采样也会内联提供（模型请求的检查点停止属于例行情况）。`full` 保留所有块内联展示。独立于 `budgeted_action_reply`（后者仍在最上层强制执行已知宿主机限制）。详见 [Actor 回复投递](tool-output-contract.md#summary-first-execution-replies-action_reply_profile)（`--action-reply-profile`；队列用例选项）。 |
| `features.reply_compact_encoding` | `true` | 将 Actor 回复中的固定结构列表（轨迹采样、`compact_3d_trajectory.stages`、事件关键帧、物体增量、碰撞见证、周期性检查、关键位姿，以及逐阶段 `effective_tolerances` / `stage_costs` 映射）打印为 `{"columns", "rows"}` 表格，一次性声明未移动的手臂（`unchanged_arms`），并以 4 位有效数字打印 `derived_motion` 的有限差分量；测量值按原样展示。不适用于存储证据的完整读取。详见 [Actor 回复投递](tool-output-contract.md#compact-encoding-reply_compact_encoding)（`--[no-]reply-compact-encoding`；队列用例选项）。 |
| `features.tool_catalog_profile` | `layered_v3` | `--tool-catalog-profile` 选择工具描述如何声明各输入契约：`full_v1`（散文形式字段指南），`guide_only_v1`（`full_v1` 描述；Codex 动态工具 Schema 省略工具字段指南已逐字承载的属性描述，仅限 Codex 会话），`compact_v1`（类型化表示法的相同契约，描述每个字段），`layered_v1`（类型化表示法；罕用字段仅列出路径与签名，其完整行发布在 Actor 工作目录的 `tool_reference.md` 中，最少使用的工具延迟加载），`layered_v2` 或 `layered_v3`（默认）。`layered_v2` 与 `layered_v1` 有四处不同：罕用字段保留其描述开头（短时完整，否则在单词处截断并以 `...` 结尾）；嵌套在罕用字段内的字段仅列在 `tool_reference.md` 中；联合类型罕用形式的规则行（罕用的 `geometry_query` 请求种类）替换为命名它们的一行；且在 Actor 目录中，较早加载工具已相同声明的字段行变为 `<path>: same as in <tool>`。当智能体宿主机按需加载延迟工具时，`layered_v2` 还在启动指令中逐行标出这些工具。`layered_v3` 是当前分层形式和默认值：`layered_v2` 且不带额外延迟加载（除了基础目录已延迟的工具外，不额外发布按需发现工具，因此启动指令中不添加按需工具索引），以及一个描述 `tool_reference.md`（每字段一行）的罕用字段块，而不是硬塞 Shell 命令。`layered_v1` 和 `layered_v2` 保持不变，以便使用它们记录的运行能正常加载。输入 Schema、校验和工具名称绝不改变；改变的仅是描述和 `deferLoading`。类型化表示法无法表达的 Schema 构造使用该工具的 `full_v1` 散文指南发布。分层配置的罕用字段和工具是一套经验选择，目前仅对 ARX-X5 有使用证据；对于其他机器人，两者均等同于 `compact_v1`，未列出的任何字段都会完整描述。使用 `agent.backend=external` 时，分层配置以 `compact_v1` 发布。默认配对（`layered_v3` 搭配 `robot_urdf_delivery=file`）仅在仿真 ARX-X5 与 Codex 后端的小规模配对对比中评测过；在其他后端或 SO-101 上没有全量基准测试集的评测与任务证据。`full_v1` 可恢复早期的散文目录。详见 [工具注册](tool-plugins.md#catalogue-profiles)。 |
| `features.robot_urdf_delivery` | `file` | `--robot-urdf-delivery` 选择 Actor 从何处获取机器人 URDF 文本：`file` 表示在其工作目录中作为 `robot.urdf`（启动指令会指明；`robot_information` 仍返回文本），或 `inline` 表示在启动指令中内联展示。没有自身工作目录的会话（`agent.backend=external`）始终内联获取。`inline` 逐字节恢复早期的启动文本。 |
| `features.completion_guidance` | `hook` | 基准测试运行默认在轮次完成时进行感知状态的归位检查；`initial` 仅在初始提示词中注入提醒，`both` 同时启用两者，`none` 禁用归位检查同时保留通用延续。 |
| `features.grasp_memory_dir` | `~/.local/share/ni-rua/grasp-memory` | 跨运行抓取记忆目录，仅在启用抓取记忆功能时查阅。 |
| `features.disabled_tools` | `[]` | 单独禁用特定已知工具名称（优先级高于功能组启用）。在基准测试运行中无法禁用 `robot_capabilities`，因为网关在启动时会通过注册表查询它；禁用 `operation_memory` 也会从启动指令中省略其章节。 |
| `features.lean_projection` | `false` | 上游请求历史投影（`ni_rua/core/presentation/lean_projection.py`，默认禁用；仅重写发往上游的请求体，不修改 Codex 线程历史）：将已闭合历史块的非锚点工具输出中的内联图像替换为文本占位符，并将 6 个高频工具（`robot_action`, `program_run`, `robot_observe`, `robot_wait`, `geometry_query`, `history_view_observation`）的冗长文本输出替换为确定性摘要；启用时，观测载荷也停止重复先前轮次中交付的静态字段。**潜在性能权衡**：压缩历史反馈与图像会丢弃闭合块中非锚点步骤的像素细节、完整轨迹采样和冗长的 `stdout` 日志（除非模型显式调用 `history_view_observation` 重新读取），这可能影响长视距空间推理或多步故障恢复；请默认保持 `false`。 |
| `features.lean_projection_k` | `1` | 尾部块中保留完整内联图像的最近工具步骤数（`1`–`8`，仅在 `lean_projection_block=0` 时生效）。 |
| `features.lean_projection_block` | `8` | 块对齐模式（启用 `lean_projection` 时的默认策略）：省略独立的尾部窗口；当前打开块中的工具步骤与每个闭合块的第一个步骤（锚点）原样发送，而闭合块中的非锚点步骤替换为摘要/占位符。由于每 `B` 步仅重写一次，块内请求 `n+1` 是请求 `n` 的严格前缀扩展，且在块边界上锚点请求保持有效前缀，因此仅新闭合的块需要重新预填充。在块模式下，决策边界提醒（`_task_decision_context`）放置在固定槽位，而非附加在每个请求的最末尾（初始请求中紧贴任务消息前的一个槽位，加上该块中第一个带有保存计划的请求上创建的每块一个槽位，位于最新模型轮次之前 / 上一个工具输出之后）。每个槽位保留其创建时的快照，闭合块槽位保持在原位，且“暂无计划”提醒仅出现在初始槽位中（见 `lean_projection.py` 文档字符串）。设为 `0` 选择滑动尾部窗口模式（请注意，在测试的兼容 `Responses` 的中转代理上，一旦尾部窗口滑动，上游 KV 缓存不会复用前缀，因此缓存命中停留在滑动前的最后一个请求）。 |
| `features.lean_request_dump` | `0` | 诊断设置：将前 `N` 个发往上游的请求体（内联图像字节替换为摘要）写入 `private/request-dump/`；`0` 禁用转储。 |
| `features.external_perception` | `true` | 外部视觉模型（VGGT + SAM）总开关。`Config` 类和 `configs/ni-rua.json` 默认对 `features.vggt_endpoint`（默认 `http://127.0.0.1:8331`）启用 `vggt_height`（`vggt_height_mode="optional"`）、`object_table`、`scene3d`、`object_table_observe` 和 `scene3d_observe`，而可选子开关（`persistent_object_tracking`、`multiview_scene3d`、`held_perception`、`pose_skill`）保持 `false`，除非显式启用。若服务不可达或请求失败，对应字段标记为 `unknown` / `unavailable` 并附带诊断原因，同时观测和动作回执正常交付，不中止回合。在核心子标志保持默认 `true` 时设置 `external_perception=false` 会禁用所有外部感知调用；在 `external_perception=true` 时将所有核心子标志设为 `false` 会启用核心感知栈；任何混合组合均遵循各个单独子标志。命令行参数遵循 `--[no-]external-perception` 和匹配的逐子标志选项；详见 `docs/vggt_height.md`、`docs/object_table.md`、`docs/scene3d.md` 和 `docs/obs3d.md`。 |
| `features.persistent_object_tracking` | `false` | 可选的 SAM 3.1 流式视频跟踪会话（`session_id` 有界滑动窗口，回合停止时自动关闭），用于跨帧持久物体 ID（`track_meta.persistent_ids=true`）、工作区级 `object_state_delta` 和 `geometry_query(kind="reanchor_tracks")`。 |
| `features.multiview_scene3d` | `false` | 可选的手腕相机残差门控点云融合（`multiview_cluster.py`），当活跃手腕相机具有近场桌面视锥覆盖时在 `obs3d` 上触发，并将重建缓存于 `MULTIVIEW_CACHE`。 |
| `features.held_perception` | `false` | 可选的手持物体与机器人网格遮罩感知（默认禁用；需要 `object_table` 或 `object_table_observe` 以及支持 `object-table-v5-hp1` 的服务，可自动回退到 `object-table-v5`）：将物体表中的 6 cm 连杆胶囊体近似替换为根据测量关节角渲染的前向运动学机器人网格遮罩（膨胀 3 px）。夹爪闭合夹持物体的任何手臂都会获得专用的手持物体行（`quality: "held_fk_depth"`, `support: "held_<arm>"`），其尺寸和主轴随 `link6` 从抓取前物体表行刚性传递，并居中于 FK 抓取中心（沿主轴将细长物体平移携带偏移量，受 45 度轴冲突门控）。不存在抓取前行时，根据经 FK 抓取深度重缩放的 SAM 遮罩内 VGGT 点估算尺寸。手持行绕过空中到桌面的重投影，`value.held[id]` 报告估算方法、不确定性和源行，并在开发者指令中附加一句话；禁用时，工具目录、指令和物体表保持不变（详见 `docs/held_perception.md`）。 |
| `features.vggt_support_z_m` | `"profile"` | 缩放 VGGT 深度的支撑面高度（米，观测坐标系）。`"profile"`（CLI `profile`）取运行本体的 `support_surface_z_m`（ARX X5 桌面 `0.765`；SO-101 没有该配置，因此遵循观测标定的 `table_z_m`）；`[-3, 3]` 内的数值对所有机器人均具有权威性（包含 `0.765` 的归档和配置保持不变）；`null`（CLI `observation`）始终使用观测的标定，并在缺少标定时使感知不可用。在配置档案存在之前，默认值为数值 `0.765`；X5 解析为相同的值。详见 [SO101 配置](so101-configuration.md#vggt-support-height-source)。 |
| `features.vggt_camera_id` | `cam_head` | 当 `object_table` / `scene3d` 请求省略 `camera_id` 时以及用于自动 `obs3d` 附件的相机：静态全景相机。当观测到的机器人没有该 ID 的相机时，共享默认值 `cam_head` 替换为该本体的 `overview_camera`（SO-101 上为 `fixed`）；显式指定的 ID 优先。 |
| `features.record_video` | `true` | 自动启动原生三相机视频录制。 |
| `features.export_video` | `true` | 在运行终态处理期间导出录制的视频（当 `record_video=false` 时自动跳过）。 |
| `features.terminal` | `true` | 启动智能体后端的只读原生终端用户界面（Codex，未开启沙箱的 OpenCode；Claude Code 没有终端 UI，详见 [Agent 后端](agent-backends.md#native-terminal-featuresterminal)）；无头运行且不需要 TUI 时请禁用。 |

`feedback` 运动执行模式严格保留给受信的 Python 诊断脚本（如 `scripts/probes/feedback_motion_probe.py --mode feedback`）；它不支持公开的 `robot_action` 工具，且不能用于 Actor 或程序进行统一双臂操作。`Config.validate_launch()` 和评测队列在为基准测试启动模型或 Worker 前均会拒绝 `feedback` 模式。非基准测试的诊断工作台可传入 `--motion-execution-mode feedback --no-motion-review --no-tracking-guard`，但这不会暴露公开的手臂动作工具；在 `feedback` 模式下记录的历史归档保持可读与可重放。在 `feedback` 模式下，Worker 对每个路径点规划一次，选择限位内最接近的 $2\pi$ 等价关节目标（核验全连杆 URDF 等价性与目标位姿精度），并通过原生 `take_action` 入口反复下发指令，直到位置误差在请求容差内、姿态误差不超过 `0.1 rad`，且所有活动臂关节速度在两个连续原生步长内均不超过 `0.1 rad/s`。若最佳位置/姿态误差或关节目标误差在 15 个原生步长内未充分改善，无论瞬时速度如何，执行均以 `feedback_stalled` 停止（这也限制了达到位置但速度无法静止的情况）；每个动作仍受 10 秒仿真时间（默认 250 原生步长）、动作墙钟截止时间、取消操作以及原生终止条件的限制，而不改变回合的官方步长上限。

`feedback` 模式保留顺序用户路径点和 `after_motion` 夹爪驻留时间，同时显式拒绝缓存轨迹预览和 `during_motion` 分数夹爪时序。由于闭环反馈跟踪规划的终点而非逐字遵循初始开环轨迹，在初始规划期间执行的碰撞检查不能保证执行路径无碰撞。`reference_resampled_steps` 仅提供基于初始轨迹行计数的诊断估计，而实际步长记账始终使用回执中的 `executed_native_steps`。`robot_wait` 和关节策略前缀的步长记账保持不变。

在 RoboDojo 移瓶任务的受控对比中，`feedback` 模式减少了短视距运动的步长计数，但未提高全任务成功率；对比暴露出在高瞬时速度和长发条关节目标下的失速检测盲区，这两项此后均已修复。`feedback` 模式仅保留用于受信的固定输入诊断，而官方统一动作默认并强制要求 `resampled`。

工具在配置中是否启用与模型是否选择调用它是两个独立的问题。`features.disabled_tools` 同时过滤模型的工具目录、直接 Actor 调用和 `program_run` 代理调用。从程序内部发起的物理动作通过完全相同的 `robot_action` 执行边界与审批工作流，无法通过旧动作名称绕过。禁用 `program_run` 或单独工具绝不能替代原生观测与执行协议的校验。

原生采集、模型工具、程序以及历史查询中默认禁用深度感知。关于三个显式的 RGB-D 功能开关、数值接口和技能参数，请参阅 [RGB-D 数值工具](rgbd-numeric-tools.md)。公开的 `depth_query` 工具仅在 `rgbd_proprio_calibrated_v1` 配置下暴露（旧版 `depth_move` 已在 2026-10 重构中移除），且在 `disabled_tools` 中添加 `depth_query` 也会禁用 `geometry_query` 上的深度别名。将任何深度功能开关与非 RGB-D 观测配置结合会在启动时引发配置校验错误。

<a id="motion-and-gripper-thresholds-features"></a>
### 运动与夹爪阈值（`features.*`）

**ARX X5** 的容差、失速保护（stall-guard）与握持保护（grip-guard）预设是这些字段的函数，每次运行由 `core.config.motion_thresholds` 解析为运行的 `EmbodimentProfile`（详见 [本体配置](embodiment-profile.md#how-thresholds-are-configured)）。非默认值会同时更改预设、校验器和 `robot_action` 的 Schema 文本（决策 Q2）；随附的配置携带默认值，因此与早期运行无异。SO-101 不读取这些字段：其数值位于部署文件中（见下文）。

| 字段（单位） | 默认值 | 含义 |
| --- | --- | --- |
| `tight_position_tolerance_m` / `default_` / `relaxed_` / `compliant_` / `coarse_position_tolerance_m` (m) | 0.003 / 0.010 / 0.015 / 0.015 / 0.025 | 预设 `tight`、`normal`（默认）、`relaxed`、`compliant`、`coarse` 的 TCP 位置容差；范围 0.001..0.05 |
| `max_position_tolerance_m` (m) | 0.10 | 接受的最大显式 `position_tolerance_m` |
| `tight_orientation_tolerance_rad` / `orientation_tolerance_rad` / `relaxed_` / `coarse_orientation_tolerance_rad` (rad) | 0.05 / 0.15 / 0.20 / 0.35 | 相同预设的姿态容差；`max_orientation_tolerance_rad`（$\pi$）是最大显式值，3.14159 及以上的值表示不约束姿态 |
| `stall_guard_strict_divergence_m` / `stall_guard_contact_stop_divergence_m` / `stall_guard_compliant_divergence_m` (m) | 0.010 / 0.020 / 0.020 | 失速预设 `strict`、`contact_stop`（及其别名）、`compliant` 的 `max_tracking_divergence_m`；`stall_guard_min_divergence_m` 0.005 和 `stall_guard_max_divergence_m` 0.15 限制显式值范围；`stall_guard_step_velocity_threshold_m` (0.0015 m) 和 `stall_guard_consecutive_steps` (2) 用于微调 X5 失速检测器 |
| `max_gripper_aperture_m` (m) | 0.088 | ARX X5 指尖完全张开度，配置档案的 `gripper.full_open_aperture_m`（joint7 + joint8 行程）：显式 `min_aperture_m` / `max_aperture_m` 的上限、握持保护最大值、感知 `opening_check` 和深度 `max_opening_m` 的默认值。X5 唯一的完全张开值（V-D） |
| `grip_non_empty_min_aperture_m` / `grip_thin_` / `grip_rigid_` / `grip_wide_min_aperture_m` (m) | 0.0015 / 0.0005 / 0.002 / 0.003 | 预设 `non_empty`、`thin_object`、`rigid_object`、`wide_object` 高于空闭读数的最小指尖间隙；每项最多为 `max_gripper_aperture_m` |
| `grip_empty_threshold_mm`, `grip_thin_threshold_mm`, `grip_holding_threshold_mm`, `grip_solid_threshold_mm` (mm) | 0.5 / 1.5 / 3.0 / 3.5 | 旧版 X5 保护名称（`empty_only`、`thin`、`solid`）以及闭爪降级的毫米阈值 |

在本体重构中退役的字段（决策 V-E）：`soft_position_tolerance_m`、`grip_thin_max_aperture_m`、`grip_rigid_max_aperture_m`、`grip_wide_max_aperture_m`、`max_min_grip_m`、`closed_jaw_goal_max`、`checkpoint_empty_jaw_measured_max`。没有任何代码读取它们。它们已从 `FeaturesConfig` 及新运行的 `effective_config.json` 中移除；仍携带它们的配置或已归档的 `effective_config.json` 可原样加载，这些键会被忽略丢弃（`core.config.RETIRED_FEATURE_FIELDS`）。

**SO-101 部署文件**（`configs/so101.local.json`，每个字段均在 `deployment.FIELDS` 中声明了单位、范围和默认值；详见 [SO-101 集成规范 §4.1 与 §4.7](so101-integration.md#41-grip-guard-and-checkpoint_after_gripper)）。夹爪：`gripper.empty_close_aperture_m`（米，默认 `null` = 未测量；握持保护预设的零点，范围 0..0.05），`gripper.full_open_aperture_m`（米，默认 `null` = `jaw_open_rad` 时的 CAD 开度；显式开度限制的边界，范围 0..0.2），`gripper.preset_margins_m`（高于空闭间隙的米数，表示为 `non_empty`、`thin_object`、`rigid_object`、`wide_object` 的 `[min, max or null]`；默认值为 0.0015 / 0.0005 / 0.002 / 0.003，无最大值），`gripper.hold_squeeze_rad`（夹爪关节弧度，范围 0..夹爪行程，默认 `null` = 关闭：在闭合夹爪接触稳定后，夹爪目标变为测量角度减去该挤压量，从而使舵机以有界力矩保持；也是 `robot_action` 中 `grip_squeeze_rad` 的默认值，仅在该键设置时提供）和 `gripper.max_hold_squeeze_rad`（弧度，默认 `null` = 夹爪行程；阶段可请求的最大 `grip_squeeze_rad`）；详见 [SO-101 集成规范 §4.13](so101-integration.md#413-bounded-holding-squeeze-gripperhold_squeeze_rad-grip_squeeze_rad)。运动：未指定容差移动时应用的 `motion.default_position_tolerance_m`（0.01 m）和 `default_orientation_tolerance_rad`（0.15 rad），`motion.tolerance_presets`（每个预设名称对应的 `[m, rad]`，默认值如上表，加上 `unconstrained_orientation` 的 `[0.015, 3.14159]`），`motion.default_timeout_s`（60 秒），`max_timeout_s`（300 秒），`default_wait_timeout_s`（30 秒），以及 `motion.arrival_correction_rounds` / `arrival_correction_max_offset_rad`（3 / 0.1 弧度，在判定到位前运行的下垂修正）。`gripper.preset_margins_m`、`motion.tolerance_presets` 和 `motion.default_wait_timeout_s` 均为可选配置：省略它们的部署文件加载时使用上述共享默认值，已提供的值绝不会被覆盖。可选的 `ik_service`（默认缺失 = 打包的本地 IK）：`solver`（`"curobo"`），`url`（回环 `http://127.0.0.1:<port>`，即 SSH 端口转发的本地端），`timeout_s`（每请求秒数，默认 10，范围 (0, 60]），`num_solutions`（每位姿最佳关节解数，默认 8，范围 1 到 32）和 `table_top_z_m`（`so101_base` 坐标系下的米数，默认 `null` = 无桌面盒，范围 [-0.3, 0.3]）。配置该服务将启用 cuRobo GPU 服务的 IK 规划并提供 `approach_tolerance_rad` / `roll_tolerance_rad`；规划的关节限位边距 `motion.joint_limit_margin_rad` 必须与构建服务模型时使用的边距一致。语义、故障行为以及该服务详见 [SO-101 集成规范 §12](so101-integration.md#12-ik-planning-the-curobo-ik-service-and-modereach)。

<a id="state-aware-completion-guidance-homeposecheck"></a>
### 感知状态的任务完成引导（`HomePoseCheck`）

`completion_guidance="hook"` 或 `"both"` 使用 [`HomePoseCheck`](../../ni_rua/runtime/actions/home_check.py) 组件。在每个回合开始时——即原生重置和场景稳定后、Actor 执行首个动作前——`HomePoseCheck` 从首个公开观测中记录每条手臂的实测 `ee_pose`（ARX X5：双臂，`env_origin` 坐标系下的 `link6` 位置与 `quaternion_xyzw` 姿态；SO-101：手臂 `right`，`so101_base` 坐标系下的 `gripper_frame_link`；标识来自运行的 `EmbodimentProfile.home_check`）。此初始基准保留在观测历史和归位参考记录中，不会被后续观测覆盖，仅在显式环境重置或会话恢复开启新回合时重新记录。新基准测试运行默认采用 `hook`，它既不在初始提示词中预先注入归位提醒，也不在每轮无条件发送归位警告；非基准测试的开发运行不触发基准轮次延续。恢复和重放的运行遵循其归档设置（缺少该字段的历史归档默认为 `none`）。

当主 Actor 发出 `turn.completed`、原生回合尚未结束且标准调度护栏允许延续时，`HomePoseCheck` 根据 RoboDojo 默认的 `all_robot_back_to_origin` 几何规则比对最新完整观测与记录的初始位姿：双臂沿每个笛卡尔位置轴的绝对差值必须**最多为 0.15 m**，且最短四元数旋转角必须**最多为 20 度**（边界包含）。四元数 `q` 与 `-q` 表示相同姿态，忽略多圈关节缠绕与夹爪开度。虽然这符合整理摆放类任务的终态归位条件，但并不代表所有任务的完整成功标准；在依赖其处理其他任务族之前，请核实特定任务的归位惯例。

| 检查结果 | 发送的延续提示词 |
| --- | --- |
| `away`：存在有效的双臂测量值且至少一臂超出容差 | 显式提醒模型手臂尚未回到初始位姿，提供初始观测的历史引用，并指出其他任务条件可能亦未满足。 |
| `home`：双臂均在容差范围内 | 发送标准延续提示词（"The task is not finished yet; please continue."）。 |
| `unknown`：缺少初始/当前测量值、观测陈旧、标识/坐标系不匹配或数据无效 | 发送相同的标准延续提示词，不将证据缺失诊断为机器人未归位。 |

`HomePoseCheck` 仅读取获准的机器人状态产物——绝不发起动作、自动重置手臂或查询私有评测器与物体真值。准入任何物理动作都会使当前观测证据失效；中间进度快照无法替代终态观测，因此只有完整的动作后终态采集或随后的 `robot_observe` 才能恢复有效检查状态。在仿真暂停期间，可以对同一完整观测进行反复检查，只读的 `history_*` 查询不改变活跃观测证据。原生终止状态、累积模型请求预算、超时、未确认动作和活跃程序护栏始终优先于延续钩子。

每个回合会记录一个 `completion_guidance.home_reference` 事件以保存初始参考位姿，清单和任务提示词配置记录策略版本、容差、初始参考与三个提示词模板。每次延续尝试均记录一个 `benchmark.actor_turn_continuation` 事件，包含检查结果、每臂误差、初始与当前观测引用、`continuation_kind` 以及所选的精确提示词。请注意，该事件记录延续轮次的发起；请核实后续会话消息以确认上游交付。在状态感知 `hook` 模式下，静态的 `continuation_prompt` 字段为 `null`——请查阅 `continuation_prompts` 与逐轮事件。

例如，以默认的状态感知 `hook` 启动基准测试无需额外参数：

```bash
mamba run -n ni-rua python -m ni_rua.apps.gateway --benchmark --task arrange_largest_number_random \
  --run-dir runs/arrangement-home-check
```

传入 `--completion-guidance none` 可禁用归位检查，传入 `--completion-guidance both` 则兼顾初始提示词提醒。队列清单用例在省略该字段时继承统一配置（默认为 `hook`），并可通过 `"completion_guidance": "none"` 或 `"both"` 在用例级别覆盖。

所有现存的 CLI 参数均与其对应的否定形式一同支持，包括 `--no-default-task-notes`、`--no-vision-compare`、`--no-enable-wait`、`--no-pi05`、`--no-tracking-guard`、`--no-joint-replan`、`--no-motion-review`、`--no-terminal`、`--no-record-video` 和 `--no-export-video`。`--no-visual-review-model` 可禁用配置文件中启用的辅助审查模型。传入 `--disabled-tools robot_wait vision_review` 会替换已禁用的工具列表，而传入不带参数的 `--disabled-tools` 会清空该列表（它绝不会强制启用因功能开关或观测配置而不可用的工具）。

<a id="current-actor-tools-and-availability-controls"></a>
## 当前 Actor 工具与可用性控制

活跃的工具目录由 `ToolRegistry` 中的公开 `ToolSpec` 声明生成，并根据解析后的配置、观测配置档案与后端能力进行过滤（`ni_rua.runtime.harness.TOOLS` 仅作为历史兼容导出保留，不代表实时目录）。下表列出了 20 个内置工具声明加上 2 个可选视角工具；禁用时，可选工具会从目录中剔除。任何工具均可通过 `features.disabled_tools` 禁用，且配置标志绝不能绕过角色检查、观测权限或物理动作准入。

| 工具 | 基础可用性 / 前提条件 | 用途 |
| --- | --- | --- |
| `task_plan` | 默认启用 | 在首个物理动作前提交初始任务计划，或修订并读取活跃计划。 |
| `robot_observe` | 默认启用 | 采集当前获准的相机图像与同步的实测机器人状态。 |
| `robot_capabilities` | 默认启用 | 查询可用能力、单位、坐标系、TCP 定义与预算。 |
| `robot_information` | 默认启用 | 读取启动文档、原生 URDF、固定物理规格与获准标定。 |
| `robot_contact_overlay` | 已标定观测配置 | 将机器人参考几何渲染叠加到获准的 RGB 图像上。 |
| `robot_action` | 已标定观测配置，`resampled` 模式，支持统一动作的后端 | 预览多阶段运动批次、执行缓存计划，并跨单臂或双臂结合观测检查点检查轨迹。 |
| `robot_action_status` | 默认启用 | 查询先前提交动作的执行状态而不重新执行。 |
| `robot_wait` | `features.enable_wait=true` | 保持双臂关节与夹爪目标，同时显式推进仿真步长。 |
| `robot_policy_step` | `features.pi05=true` 且匹配的策略服务在运行 | 针对局部子目标执行有界的 $\pi_{0.5}$ 关节空间策略前缀。 |
| `history_list_actions` | 默认启用 | 分页浏览不可变的历史动作请求与执行回执。 |
| `history_view_observation` | 默认启用 | 检查历史相机图像与机器人状态，支持可选的相机选择、ROI 裁剪与缩放。 |
| `history_read_trace` | 默认启用 | 读取获准的规划、指令与实测关节/位姿轨迹。 |
| `history_annotate_action` | 默认启用 | 向历史动作追加 Actor 诊断笔记，不修改测量值或状态。 |
| `geometry_query` | 已标定观测配置（深度查询亦需 RGB-D 配置） | 计算像素射线、3D 到 2D 投影与获准的数值几何查询。 |
| `robotic_view` | `features.robotic_view=true`，已标定配置，手腕相机具有已发布的固定安装链 | 只读求解器，将期望的相机视点转换为对应的 `link6` 位姿。 |
| `stereo_guide` | `features.stereo_guide=true` 且同一形式观测中有两台无畸变针孔相机 | 在目标相机视图上渲染源像素及其对应的极线，不进行自动双目匹配。 |
| `depth_query` | 显式 RGB-D 观测配置 | 查询数值深度测量值与候选几何假设，不执行物理动作。 |
| `program_run` | 受控程序运行器服务 | 执行有界的 Python 程序，其物理动作默认需要主模型审批。 |
| `program_status` | 受控程序运行器服务 | 读取程序执行状态、历史源码与动作结果。 |
| `vision_compare` | `features.vision_compare=true` | 计算两次获准 RGB 观测之间的像素级特征运动。 |
| `vision_review` | `features.visual_review_model` 与辅助审查服务已启用 | 可选的辅助视觉或操作证据审查，保留不确定性且不授予动作权限。 |
| `operation_memory` | 默认启用 | 查询或追加归属于证据的操作经验，范围限于当前任务或跨任务共享。 |

`robot_action` 默认预览运动批次并执行完全一致的缓存计划；`features.motion_review` 控制历史单臂适配器，本身并不禁用统一批处理接口。旧版动作名称、队列、模板、技能、`scene_*`、`operation_verify` 与 `depth_move` 已在 2026-10 重构中移除，不会出现在 Actor 或程序目录中。历史 Python 适配器与历史功能开关在其兼容契约下继续解释；详见 [统一动作与程序审批](unified-actions.md) 与 [工具插件](tool-plugins.md)。

`robot_action` 中 `kind="move"` 阶段内的同步夹爪动作按请求通过 `gripper_goal_normalized` 和满足 `0 <= start_fraction < end_fraction <= 1` 的可选 `gripper_motion={"start_fraction": 0.2, "end_fraction": 0.8}` 窗口指定。省略 `gripper_motion` 时，夹爪指令在手臂运动达到阶段目标后发出；独立夹爪驱动使用 `kind="gripper"` 搭配 `goal_normalized`。这些是逐阶段请求参数而非全局配置标志。执行遵循缓存轨迹，仅指令完成绝不证明物理接触、稳定抓取或任务成功。

<a id="optional-view-tools"></a>
### 可选视角工具

`robotic_view` 与 `stereo_guide` 是独立的布尔标志，默认均为 `false`。两者均可单独启用：

```json
{
  "version": 1,
  "features": {
    "robotic_view": true,
    "stereo_guide": false
  }
}
```

网关命令行参数 `--robotic-view` / `--no-robotic-view` 与 `--stereo-guide` / `--no-stereo-guide` 可覆盖配置文件。队列清单中的单例用例可显式覆盖 `"robotic_view": true` 或 `"stereo_guide": false`（省略时继承统一配置；两个字段均不接受字符串或 `null`）。会话恢复与重放遵循归档标志，将旧归档中缺失的字段视为 `false`。

禁用时，对应的插件不会被注册，该工具及其文档均不会交付给 Actor 或程序。即便通过功能开关启用，传入 `--disabled-tools robotic_view stereo_guide` 仍可按名称抑制它们。工具文档严格通过各工具的 `ToolSpec` 交付，不修改全局任务提示词或 Task Notes，且这两个工具均不移动机器人、不捕获新观测、亦不推断任务成败。关于参数 Schema 与使用示例，请参阅 [手腕视角求解器](robotic-view.md) 与 [双目极线引导](stereo-guide.md)。

<a id="remote-gpu-services-remote"></a>
## 远程 GPU 服务（`remote`）

可选的 `remote` 节点指定一个 SSH 宿主机及其上的后台服务（例如 VGGT/SAM 3.1 感知服务）。当 `remote.enabled` 为 `true` 时，网关（以及 `python -m ni_rua.apps.remote --config <file>`）会建立 SSH 端口转发，复用健康的现有服务或在显存充足的 GPU 上后台分离启动它，在 `GET /v1/remote/status` 提供健康状态，并在退出时关闭转发，且仅在 `remote.stop_on_exit` 为 true 时停止自身启动的服务。该节点默认禁用，未使用的节点会从导出的配置快照中省略（因此 X5、仿真及冻结代码检出的队列映射均不受影响），`--remote` / `--no-remote` 可为单次运行覆盖 `remote.enabled`。`features.vggt_endpoint` 必须是 `remote.services` 中声明的转发本地端。字段含义、单位、默认值、启动/复用/停止规则及状态格式详见 [SO-101 集成规范第 9 节](so101-integration.md#9-distributed-deployment-mac--remote-linux-gpu-server)；可直接合并的示例为 `configs/so101.remote.example.json`。SO-101 cuRobo IK 服务是第二个服务条目（`name` 为 `curobo_ik`，`remote_port` 8341，`workdir` 为部署的 `services/curobo_ik` 目录，`command` 为 `bash start.sh`，`stop` 为 `bash stop.sh`，`health_path` 为 `/health`，`gpu` 为 `auto`；示例见 [SO-101 集成规范 §12.3](so101-integration.md#123-the-service-and-how-the-gateway-starts-it)），部署文件的 `ik_service.url` 指向它。这些内容均不进入模型上下文。

<a id="agent-backend-agent"></a>
## Agent 后端（`agent`）

可选的 `agent` 节点选择驱动 Actor 的智能体运行时。`agent.backend` 可为 `codex`（默认）、`claude_code`、`opencode` 或 `external`（NI-RUA 不启动智能体，由你自己的智能体通过 MCP 接入，详见 [Agent 插件](agent-plugin.zh-CN.md)）；`--agent-backend` 可为单次运行覆盖它，所有其他 `agent.*` 字段仅在配置文件中设置。后端的模型供应商由**供应商 Profile** 指定（`upstream.source: "profile"`，每个供应商与密钥对应一个 Git 忽略的文件 `<agent.profiles_dir>/<id>.local.json`；格式、导入命令、目录缓存与控制台见 [Agent 后端](agent-backends.md#provider-profiles)）；早期的配置源（`claude_settings`, `opencode_config`, `codex_home`, `explicit`）对既有配置、归档和恢复会话保持可读。当 `agent` 等于默认值时，它在导出的快照、`effective_config.json` 和清单中会被省略，因此 Codex 运行保持字节完全一致，旧归档解析为 `codex`。`model.name`、`model.effort`、`model.price_card` 和 `model.transport_recovery` 适用于每个托管后端；`model.codex_home` 仅适用于 Codex（亦供 OpenCode 在 `upstream.source="codex_home"` 时复用）；`model.native_memories` 适用于 Codex 与 Claude Code。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `agent.backend` | `codex` | 智能体运行时。`external` 仅限交互模式：若搭配 `--benchmark` / `--exit-after-turn` 则拒绝启动并报错 `external_backend_is_interactive_only`；若搭配 `features.sandbox_enabled` 报 `external_backend_cannot_sandbox`；搭配 `lean_projection` 报 `lean_projection_requires_codex_backend`；搭配 `features.visual_review_model` 或 `model.price_card` 报 `external_backend_has_no_model_relay`；若 `run_begin` / `run_finish` 在 `features.disabled_tools` 中报 `external_bracket_tool_cannot_be_disabled`；端口无法绑定时报 `external_mcp_port_in_use` / `external_mcp_port_unavailable`；队列拒绝此模式（`external_backend_not_queueable`）。任务仅来自 `--prompt` / `run.prompt`。 |
| `agent.external.mcp_port` | `8790` | 仅限 `external`。本次运行的 MCP 端点 `http://127.0.0.1:<mcp_port>/mcp` 监听的回环 TCP 端口（整数 0..65535），用于配置智能体的插件条目（与实验库的 8780 和场景服务的 8770 并列）；`0` = 启动时选择临时端口（仅限测试：静态智能体配置无法追踪它）。无需 Bearer 凭据（绑定回环、检查 `Host`、拒绝非本源 `Origin`）；每个端口对应一个外部网关。未修改时该节点从快照中省略。`ni-rua-plugin install --port N` 会将相同端口写入宿主机配置条目。 |
| `agent.external.context_delivery` | `delta` | 仅限 `external`（`delta` 或 `full`）。默认 `delta`，在 `ExternalToolBridge` 上启用单主写回租约（重叠并发调用与非主链 OpenCode 子会话自动回退 `isolated` 完整输出，新回合与新客户端自动刷新纪元）；显式设为 `full` 时每次调用均启用 `Harness.context_epoch_volatile`。因无模型中继，上下文压缩依赖插件软钩子与 `run_begin` 重简报，无中继侧裁图/复图，且 Claude Code HTTP 下串行子智能体调用无法从协议层与主链租约分离。 |
| `agent.profiles_dir` | `null` | 供应商 Profile 目录（`<id>.local.json`，权限 0600，Git 忽略）；`null` = `<checkout>/configs/profiles`。相对路径相对于设置它的配置文件解析，因此每个 worktree 的运行配置都可指向主检出目录。控制台 Profile 列表、`handoff_run`、目录缓存（`.cache/`）和 `presets.json` 均使用它。为 `null` 时从快照中省略。 |
| `agent.codex_upstream` | `null` | Codex 的上游：`null` = `model.codex_home`（`config.toml` + `auth.json`），保持不变且从快照中省略；否则在 `openai_responses` 上为 `{"source": "profile", "settings": <profile path>}`（`tool_result_images` / `tool_schema_max_depth` 必须保持默认值；拒绝 `x-api-key` Responses 协议，报 `codex_profile_auth_scheme_unsupported`）。使用扁平结构（非 `agent.codex.upstream`）是因为队列通过 `getattr(config.agent, backend).upstream` 读取。 |
| `agent.claude_code.cli_path` | `null` | Claude Code CLI 的绝对路径；`null` = `claude-agent-sdk==0.2.180` 附带的 CLI（可选扩展 `ni-rua-harness[claude]`）。 |
| `agent.claude_code.upstream` | Anthropic | 上游模型端点（见下文），通常为 `{"source": "profile", "settings": <profile path>}`；显式默认值为 `anthropic_messages`、`https://api.anthropic.com`、`ANTHROPIC_API_KEY`、`x-api-key`、`anthropic`。 |
| `agent.claude_code.context_delivery` / `agent.opencode.context_delivery` | `delta` | `delta`（四后端默认均为 `delta`）= 仅在主链交付自上次以来的增量（启用 `subagents` 不再强制覆盖为 `full`，子智能体与重叠并发调用自动走 `isolated` 完整输出）；`full` = 显式切回每轮完整上下文（设置 `Harness.context_epoch_volatile`，不假定先前交付的图像或静态元数据依然存在）。所选后端的生效模式为 `config.agent.context_delivery`（Codex 固定 `delta`，Claude Code / OpenCode / External 默认 `delta` 且均支持显式切回 `full`）；`robot_observe` 描述中指明仅当为 `delta` 时摘要才省略未变的标定。 |
| `agent.claude_code.subagents` / `agent.opencode.subagents` | `true` / `true` | 是否允许原生子智能体工具（Claude Code 与 OpenCode 默认均为 `true`）。 |
| `agent.claude_code.auto_compact_window_tokens` | `null` | 触发原生自动压缩的 Token 窗口（100000..1000000）；`null` = CLI 默认值。 |
| `agent.opencode.executable` | `null` | 锁定的 OpenCode 平台二进制文件的绝对路径（npm 压缩包 `@opencode/cli-<target>@2.2.7` 中的 `package/bin/opencode`，安装步骤见 [Agent 后端](agent-backends.md#installing-the-pinned-binary)；网关的 `PATH` 上必须包含 `rg`）。 |
| `agent.opencode.upstream` | 未设置 | 上游模型端点：`{"source": "profile", "settings": <profile path>}`（当 Profile 同时支持两种协议时需带 `wire_api`），或旧版形式：显式 `wire_api`、`base_url`、`api_key_env`；`source: "codex_home"`；带 `settings` 和 `provider` 的 `source: "opencode_config"`（从 OpenCode 配置文件读取端点、密钥与协议，见 [Agent 后端](agent-backends.md#opencode-with-one-multi-model-provider)）。`wire_api` 为 `anthropic_messages` 或 `openai_responses`。 |
| `agent.opencode.context_limit_tokens` | `null` | 上下文窗口 Token 数（正整数），声明为本次运行各模型的 `limit.context`；OpenCode 在该值减去其 15% 缓冲区处触发原生自动压缩。`null` = 供应商目录中该模型的 `context_window`，否则为 `200000`。在供应商目录缺失或对该模型有误时设置，或用于提前触发压缩。 |
| `agent.opencode.output_limit_tokens` | `null` | 单次模型调用可生成的 Token 数（正整数），声明为本次运行各模型的 `limit.output` 并作为 `max_tokens` / `max_output_tokens` 发往上游。显式值按原样发送。`null` = `min(catalogue max_output_tokens, 32000)`，否则为 `32000`：存在 32000 上限是因为目录列出的值可能会被网关作为请求值拒绝（部分多模型网关列出高达 1048576 但拒绝超过 32768）。在供应商需要更低上限（如限制在 16384 的模型）或允许更高上限时进行设置。生效值及其来源（`explicit`, `catalogue`, `catalogue_capped`, `fallback`）记录在 `agent_backend.effective_limits` 中。 |
| `agent.opencode.thinking_display` | `null` | 添加到每个推理努力变体的 Anthropic `thinking.display`：`null` = 不发送该字段（非 Anthropic 官方网关会拒绝它）；`"summarized"` = 端点返回摘要思考文本（针对 `api.anthropic.com` 设置）；`"omitted"` = 不返回思考文本。仅适用于 `anthropic_messages` 协议：其他（或未设置的）`upstream.wire_api` 会拒绝该配置并报错 `opencode_thinking_display_requires_anthropic_messages`。记录在 `agent_backend.native_effort.thinking_display`。 |
| `agent.<claude_code\|opencode>.upstream.tool_result_images` | `inline` | 中继在最终上游请求体中放置智能体工具结果内图像的位置：`inline` = 保持不变；`user_blocks` = 按顺序原样移动到工具结果紧随其后的标注用户块中，适用于拒绝工具结果内图像的供应商（HTTP 400 `PROVIDER_TOOL_RESULT_INVALID`）。详见 [Agent 后端](agent-backends.md#configuration-per-backend)。 |
| `agent.<claude_code\|opencode>.upstream.tool_schema_max_depth` | `null` | 供应商接受的最大工具输入 Schema 嵌套深度（非空 JSON 对象与数组，根节点 = 1；整数 >= 3）；发布的 MCP 目录中更深的子 Schema 会被折叠为宽容节点，并将原始子 Schema 作为文本承载，同时注册表仍针对完整 Schema 进行校验。`null` = 无限制。部分多模型网关上限为 13。 |
| `agent.evidence.client` | `native` | 证据辅助程序（`features.visual_review_model`, `features.self_briefing`）的客户端。`native` = 后端自身自带的辅助程序（Codex: `CodexEvidenceAgent`，一个新的临时 app-server 线程）；没有自带程序的后端使用中继客户端。`relay` = 各后端均使用 `RelayEvidenceClient`。 |
| `agent.evidence.upstream` | `null` | `features.visual_review_model` 请求的上游（字段同下文 `upstream`），在本次运行的中继上绑定为 `"evidence"`；`null` = Actor 的上游。仅限中继客户端；`wire_api` 必须是 `openai_responses` 或 `anthropic_messages`（`source: "codex_home"` 隐含 `openai_responses`）。 |

<a id="evidence-helpers-agentevidence"></a>
### 证据辅助程序（`agent.evidence`）

`RelayEvidenceClient`（[`ni_rua/cortex/agents/evidence.py`](../../ni_rua/cortex/agents/evidence.py)）通过本次运行自身的模型中继，以一次全新的、流式的、免工具请求响应单个证据包：无需智能体进程、主目录、历史记录或记忆。它接受与 `CodexEvidenceAgent` 相同的关键字与 `review_evidence(...)` 调用，并引发相同的错误字符串，因此 `VisualReviewer` 与 self-briefing 保持原样运行，并继续自行校验答案 Schema。由于请求经过中继，它使用共享的 180 次调用预算（仅在交付可用答案时计费）、归因追踪（`source` 为 `sol_visual_review` 或 `self_briefing`，线程为 `evidence-<request_id>`）、重试机制、传输恢复门禁、受阻传输门禁以及出口脱敏过滤。Self-briefing 始终在 Actor 的上游向 Actor 的模型发起请求。视觉审查在设置了 `agent.evidence.upstream` 时通过 `"evidence"` 绑定向 `features.visual_review_model` 请求，否则向 Actor 的上游请求；非 Codex 后端且 Actor 协议非 `openai_responses` 时，除非设置了 `agent.evidence.upstream`，否则拒绝启动 `visual_review_model`（`evidence_upstream_required_for_visual_review_model`）。在首个启用了新（上游，模型）对证据特性的基准测试运行前，用户必须授权一次实时探测，以确认供应商接受结构化输出请求；在此之前，拒绝将作为常规的记录在案失败（视觉审查向上抛给工具，跳过 self-briefing），绝不静默回退。

公平性规则：默认设置让 Codex 保持在 `CodexEvidenceAgent` 上，从而使现有 Codex 实验与历史记录保持可比。若 Harness 对比实验的分支启用了 `self_briefing` 或 `visual_review_model`，则必须在其 Codex 分支上设置 `agent.evidence.client="relay"`。生效的客户端记录在清单的 `evidence_client`（仅在启用证据特性时写入）与 `agent_backend.evidence_client` 中；若同一清单中启用了证据的用例使用了不同客户端，队列会在 `state.json`（`warnings`）与 `events.jsonl`（`queue_warning`）中写入 `comparison_evidence_path_mismatch` 警告。将 Codex 从原生切换为中继是一项独立的单变量改动：中继请求不携带 Codex Harness 上下文，并将其格式命名为 `evidence_answer`（Codex 发送 `codex_output_schema`、`text.verbosity` 和 `reasoning.context`）。默认的 `agent.evidence` 节点从快照中省略，缺少 `EvidenceConfig` 的冻结代码检出会拒绝非默认配置并报错 `unified_config_incompatible:agent.evidence`。

`upstream` 字段：`source`（`profile`：由 `settings` 命名的 NI-RUA 供应商 Profile；Profile 拥有密钥，并在配置加载时写入 `wire_api`、`base_url`、`auth_scheme`、`provider` = Profile ID 以及兼容字段，因此 `effective_config.json` 显示具体端点而绝不显示密钥，启动门禁在配置不再匹配其 Profile 时报错 `profile_upstream_stale`，缺少密钥时报错 `profile_missing_key`，以及在中继无法转发凭据头时报错 `profile_auth_scheme_unsupported`，如 `openai_responses` 上的 `x-api-key`；或旧版的 `explicit`、`codex_home`；Claude Code 的 `claude_settings`：从 `settings` 指定的官方配置文件读取端点、密钥与转发的供应商变量；OpenCode 的 `opencode_config`：从 `settings` 指定的官方 OpenCode 配置文件读取端点、密钥与协议，条目由 `provider` 选定；详见 [Agent 后端](agent-backends.md#configuration-per-backend)），`wire_api`（`anthropic_messages`, `openai_responses`, `openai_chat`），`base_url`（不含凭据、查询参数或片段的 HTTP(S) URL），`api_key_env`（存放密钥的环境变量名称；其值绝不保存在配置、清单或队列记录中），`auth_scheme`（`bearer` 或 `x-api-key`），以及 `provider`（用于定价卡与恢复身份标识的标签）。在队列中，用例可设置 `agent_backend`、`model` 和 `effort`；配置 Schema 缺少 `AgentConfig` 的冻结代码检出会拒绝 `agent_backend` 用例并报错 `unified_config_incompatible:agent`，而不是作为 Codex 运行。实验库组条件与报告条件公开 `agent_backend`（早于该节点的归档为 `codex`）。混合后端对比（同一清单中的 Codex、Claude Code 和 OpenCode 用例，例如每后端一个 `experiment_condition`）在每个层级将后端记录为显式条件；附加字段列于下表。

`claude_code` 启动门禁与运行时（`cortex/agents/factory.py`, `cortex/agents/claude_code/session.py`）：除了共享的拒绝条件外（`features.lean_projection`；缺少可用证据上游的 `features.visual_review_model`；`model.effort` 必须为 `low`、`medium`、`high`、`xhigh` 或 `max`；支持 `features.sandbox_enabled=true`，见[代码执行沙箱](#code-execution-sandbox)），除非 `agent.claude_code.upstream` 是提供 `anthropic_messages` 的 Profile、`claude_settings` 文件或带有 `base_url` 与 `api_key_env` 的显式 `anthropic_messages` 端点，否则拒绝启动并报错 `claude_code_upstream_requires_anthropic_messages`；除非已安装 `claude-agent-sdk==0.2.180` 且 CLI（内置或 `cli_path`）报告 `2.1.321`（在空临时主目录中运行一次 `--version`），否则拒绝启动并报错 `agent_backend_unavailable:claude_code: <reason>; install ni-rua-harness[claude]`。启动时网关从指定的环境变量中读取密钥（未设置时报错 `claude_code_upstream_key_missing`）；CLI 自身仅能看到占位密钥以及运行本地的中继与 MCP 端点。恢复的运行依据其归档配置进行门禁核验，且必须匹配记录的 `runtime/agent` 身份（后端、会话 ID、隔离模式、工具摘要、模型、供应商、计费基准、协议、SDK 与 CLI 版本），且在 `private/claude-config` 下仍保留 CLI 会话历史；非正常退出后的恢复会使模型上下文纪元失效并安排决策提醒。`apps/evaluation/checkpoint.py` 与 `turn_probe.py` 拒绝非 Codex 归档（`checkpoint_requires_codex_archive:<backend>`）；Claude Code 归档的重放查看器不需要运行时。操作员控制台发起的模型变更（`set_model` / `apply_agent_settings`）在下一个提示词前生效（CLI 先通过中继发送一次计费的 `max_tokens: 1` 验证请求；被拒绝的变更保持先前的模型）；推理努力等级变更从下一次会话启动生效，控制台通过交接启动后继运行（`handoff_run`）。当 NI-RUA 轮次已结束时 CLI 自身发起的原生轮次（后台 Shell 或监控结束、计划唤醒）被采纳为 NI-RUA 轮次（`agent.turn_started {origin: "native_wakeup"}`, `capabilities.native_wakeup`）；其工具调用携带该轮次 ID，在此期间的 `run_turn` 会加入其中，且控制台 `prompt` 命令像对待任何运行中的轮次一样等待其结束。运行轮次之外的注册表工具调用会被拒绝；启用 `agent.claude_code.subagents` 时不再强制将 `context_delivery` 覆盖为 `full`（子智能体调用走 `isolated` 完整输出，主链遵循 `agent.claude_code.context_delivery`，默认 `delta` 且支持显式切回 `full`，记录为 `agent_backend.context_delivery_policy`）。

`opencode` 启动门禁与运行时（`cortex/agents/factory.py`, `cortex/agents/opencode/session.py`）：除了共享的拒绝条件外，若上游协议为 `openai_chat` 会被拒绝并报错 `opencode_upstream_wire_unsupported:openai_chat`（中继仅在 `anthropic_messages` 和 `openai_responses` 上为 OpenCode 提供服务）；除非显式上游指定了 `wire_api`、`base_url` 和 `api_key_env`（或使用 `source: "codex_home"`，它隐含 `openai_responses` 并复用 `model.codex_home` 的 Codex 凭据源），否则报错 `opencode_upstream_incomplete`；除非 `agent.opencode.executable` 是绝对可执行路径且其 `--version` 打印 `opencode v2.2.7`（安装：下载 npm 压缩包 `@opencode/cli-<target>@2.2.7`，提取 `package/bin/opencode` 到 `~/.local/share/ni-rua/opencode-2.2.7/opencode`；请勿使用 curl 安装脚本，亦不要置于全局 `PATH` 上），否则报错 `agent_backend_unavailable:opencode: <reason>`。NI-RUA 仅将该二进制文件作为 `serve` 以及带 `--server` 的 TUI 运行；任何其他子命令都会在运行之外启动分离的后台服务。启动时会话还要求 `GET /api/info` 返回状态码 200 且版本为 `2.2.7`（启动中返回 503），`GET /openapi.json` 路由列表与签入的 `opencode_routes.json` 一致（170 个操作），`GET /api/model` 准确列出声明的中继模型及其限制、输入模态与推理努力变体（`opencode_served_models_mismatch`），且宿主机上必须安装 `rg`；`agent_backend.native_tools.enabled` 记录中继转发的首个带有工具的模型请求的原生工具名称（2.x 没有工具列表路由）。`opencode serve` 在空闲回环端口上运行，带有每运行唯一的 `OPENCODE_PASSWORD`、私有 XDG 目录以及存放生成的 `opencode.json` 的只读配置目录（`private/opencode/xdg/config/opencode`）：`relay` 是唯一可用的供应商，权限为 `read * allow` 加上对 `webfetch`、`websearch` 和 `question` 的拒绝（当 `agent.opencode.subagents=false` 时亦拒绝 `subagent`），`experimental.todo: true` 保留 todo 工具，`ni_rua` MCP 服务关闭 Code Mode（保持直接的 `ni_rua_<tool>` 调用），且 `media.image.auto_resize: false` 阻止 OpenCode 重新编码工具结果图像。运行期间选定的模型通过原子重写该文件进行声明（OpenCode 在约 0.3 秒内重载）；仅声明的模型可以使用，因此子智能体的 `model` 参数无法脱离本次运行记录的模型。当 `POST /api/session/{id}/prompt` 返回 200 并附带用户消息 ID 时准入轮次，该点构成轮次边界（4xx 是明确拒绝并恢复上一轮次；5xx、超时或断开连接视为未知准入并断开，绝不自动重试）。轮次在该边界后的最后一条 `idle` 消息处结束，此时私有服务器无活动会话且收件箱为空，由 `GET /api/session/active`、`GET /api/session/{id}/inbox` 以及 `GET /api/session/{id}/message` 的每页数据综合判定，绝不仅凭 `/api/event`；带有子智能体的轮次需要两次相隔 0.5 秒的相同静止快照，保持空闲而无终态答案的会话将以 `opencode_turn_incomplete` 失败。操作员干预在下一个步骤边界加入。中断操作首先为主要会话和所有已知子智能体会话阻断中继，然后调用 `POST /api/session/{id}/interrupt`。权限请求直接拒绝（`agent.permission_auto_rejected`），轮次随后以 `opencode_permission_rejected` 失败。后台子智能体在运行时保持轮次打开，无宿主轮次时启动的原生执行被采纳为 NI-RUA 轮次。`model.effort` 为两种协议上的推理努力等级：每个声明的中继模型将每个等级声明为一个显式请求体片段，通过会话的模型引用选择。`anthropic_messages` 接受 `low`, `medium`, `high`, `xhigh`, `max`；`openai_responses` 接受 `none`, `minimal`, `low`, `medium`, `high`, `xhigh`。其他值报错 `agent_effort_unsupported:opencode:<effort>`。`set_effort` 从下一轮生效。`set_model` 接受任何模型 ID：未声明的模型先被声明（原子重写配置，约 2 秒内热重载），并在**仅在下一个宿主轮次开始时**切换原生会话，绝不在轮次中途切换。在 `agent.opencode.subagents=true`（默认）下，`context_delivery` 不再强制覆盖为 `full`（子会话走 `isolated` 完整输出，主链遵循 `agent.opencode.context_delivery`，默认 `delta` 且支持显式切回 `full`，记录为 `agent_backend.context_delivery_policy`），并追踪子会话：其记录行带有子会话 ID 和 `parent_item_id`，绝非最终答案文本。恢复会话要求匹配记录的身份与运行数据库中的会话。完整的运行时描述、完整配置示例与安装步骤请参阅 [Agent 后端](agent-backends.md#opencode-runtime-opencode-serve-227)。

<a id="mixed-backend-experiment-fields-report-and-library"></a>
#### 混合后端实验字段（报告与实验库）

来自 `python -m ni_rua.apps.evaluation.report`（`conditions` 与运行数据行）的每运行字段，以及来自实验库的每组字段。缺失记录为 `null`（未知），绝不基于默认值猜测（除非注明了旧归档默认值）。

| 字段 | 出现位置 | 含义与单位 | 来源 / 旧归档默认值 |
|---|---|---|---|
| `conditions.agent_backend` | 报告、实验库 | 后端种类：`codex`, `claude_code`, `opencode`, `external`（报告为其自身探索性条件：`information_boundary="mcp_boundary_only"`, `exploratory=true`）。 | 清单 `agent_backend.kind`；实验库亦查阅 `ni_rua_config.agent.backend` 与逐用例覆盖，运行记录的种类在不同时优先（警告 `recorded_agent_backend_differs_from_configured`；不一致的尝试返回 `null`，警告 `recorded_agent_backend_differs_between_attempts`）。无此节点的归档：`codex`。 |
| `conditions.agent_cli` | 报告 | 仅 CLI 版本字符串（无路径，无哈希）。 | `agent_backend.cli.version`；Codex 回退到清单的 `codex_cli`。 |
| `conditions.evidence_client` | 报告、实验库 | 证据辅助程序的生效客户端：`codex_agent` 或 `relay`（见上述公平性规则）。对比条件的组成部分：启用证据特性的各分支必须共享该设置。 | `agent_backend.evidence_client`；实验库由最终后端与 `agent.evidence.client` 派生。后端引入前的 Codex 归档：`codex_agent`；无记录的其他后端：`null`。 |
| `conditions.native_effort` | 报告（对象）、实验库（规范 JSON 字符串） | 后端实际应用的推理努力等级（Claude Code `{"effort": ...}`，OpenCode `{"variant": ...}`，Codex `{"model_reasoning_effort": ...}`），并列于共享的 `model.effort` 旋钮。 | `agent_backend.native_effort`；未记录时为 `null`。 |
| `conditions.counting_basis` | 报告 | 模型调用计费规则的对比基准。Codex 记录 `delivered_usable_actor_helper_compact_responses_v2`，其他后端记录 `delivered_usable_generation_completions_v1`；`COUNTING_EQUIVALENCE`（`cortex/session/usage.py`）将两者映射为 `all_delivered_usable_generation_completions`，因此它们作为统一基准对比。未声明的字符串自成独立基准。 | `agent_backend.counting`，否则为 `model_requests.counting`。 |
| `counting` | 报告行 | 记录的原始计费字符串（显示在基准旁边）。 | `summary.model_requests.counting`。 |
| `served_models` | `summary.json`, 报告行 | `{供应商报告的模型名称: 已交付的生成完成数}`（无单位计数）；无模型名称的响应计入 `unknown`。显示在配置的 `conditions.model` 旁边；不宣称等同（供应商别名）。 | 由 Codex 会话与 MCP 后端写入 `runtime/model_requests.served_models`；报告回退查阅此项，对于旧 Codex 归档由 `model.*` 事件派生。 |
| `model_usage_relay` | `summary.json`, 报告行 | 中继观察到的所有交付完成项累加的 Token 计数器（`input_tokens`, `cached_input_tokens`, `cache_write_input_tokens`, `output_tokens`, `reasoning_output_tokens`, `total_tokens`；单位 Token），加上 `calls` 与 `unknown_calls`。只要有一次调用缺少该计数器，总计即为 `null`。对所有后端测量口径相同；Codex 使用 `account_responses_usage` 字段映射。 | `runtime/model_usage_relay`。在此记录之前恢复的 Codex 运行将其早期调用保留为未知。 |
| `model_records_settled` | `summary.json`, 报告行 | 在写入 summary 的时刻，中继是否已跟上其交付的每次完成项：`served_models`、`model_requests` 和 `model_usage_relay`（以及 Claude Code 和 OpenCode 上的 `token_usage`）均包含它们。中继在智能体获得答案后在其回调线程上写入这些记录，因此可能滞后于已结束的轮次；网关在事件循环外等待它们（`ModelRequestProxy.settle`），最多等待 `SETTLE_TIMEOUT_S` = 5 秒；该策略等于一次 SQLite 锁等待（`Store.LOCK_WAIT_S`），也是释放机器人和交接的最大延迟时间。在此之后交付的完成项位于归档中而非 summary 中。`false`：超时先发生，副本可能缺少最后几次完成项，报告改读归档的 `runtime/*` 记录。无中继时为 `null`。 | `episode.settle_model_records`；早于该设计的归档中缺失。 |
| `model_records_possibly_incomplete` | 报告行 | 当 `model_records_settled` 为 `false` 且归档没有 `runtime/*` 记录可替换 summary 副本时为 `true`，报告的数值可能缺少最后的完成项；否则为 `false`。 | `report.relay_records`；`scripts/summarize_mixed_usage.py` 中规则相同。 |
| `model_usage_relay_source` | 报告行 | `recorded` 或 `derived_from_model_events`（旧 Codex 归档，相同字段映射），无记录时为 `null`。 | |
| `usage_methods` | 报告行 | 每个计数器 `{native, relay, delta}`，其中 `native` 为后端自身累积使用量（`token_usage`），`delta = relay - native`（Token，任一未知时为 `null`）。两种测量方法并列呈现，绝不合并。 | |
| `agent_turn_status` | `summary.json` | 所有后端最后一次智能体轮次的终态状态；为了兼容性，`codex_turn_status` 保持相同数值。 | |
| `agent_settings` | `summary.json` | 控制台智能体设置（backend, profile, provider, wire, model, effort, service tier, thinking_display）的 `{initial, final, changes, turns, mixed}`，派生自 `run.turn_settings` 与 `operator.*_changed` 事件。`mixed: true` = 轮次以不同设置运行，或首轮后会话收到了热更改：非单一条件样本。 | `episode.agent_settings_summary`；早于该特性的归档中缺失。 |
| `handoff`, `successor_run_id`, `successor_handoff` | `summary.json`, `manifest.json` (`handoff`) | 控制台交接链接（[Agent 后端](agent-backends.md#handoff_run)）：交接产生的运行记录其 `handoff {predecessor_run_id, request_id, requested_at, fallback, carried, prompt_source}`；交接出的运行指明其 `successor_run_id` 与 `successor_handoff.status`（`handed_off`，或在清理未确认时为 `abandoned`）。每次运行均为其独立条件；调用链绝不作为单一样本。 | 未发生交接的运行缺失。 |

对于 Claude Code 和 OpenCode，`token_accounting.source` 为 `relay.<wire>.usage.sum`，报告保留该标签；缺少缓存写入定价的定价卡在此处给出部分估算（`cache_write_pricing_unknown`）。

<a id="machine-paths-and-image-pipeline"></a>
## 机器路径与图像管线

| 配置字段 | 历史环境变量 / 默认值 | 描述 |
| --- | --- | --- |
| `backend.root` | `NI_RUA_ROBODOJO_ROOT` / `~/code/robodojo` | 独立的 RoboDojo 代码库检出目录。 |
| `backend.environment` | `robodojo` | 显式环境构造器标识符；未知的环境会中止启动。 |
| `backend.robot_model_id` | `arx-x5` | 所选环境支持的机器人绑定（RoboDojo 目前仅接受 `arx-x5`）。 |
| `backend.python` | `NI_RUA_ROBODOJO_PYTHON` / `~/miniforge3/envs/RoboDojo/bin/python` | Isaac Sim 机器人仿真 Worker 的 Python 解释器。 |
| `policy.python` | `NI_RUA_PI05_PYTHON` / `~/.local/share/ni-rua-openpi-arx/bin/python` | OpenPI 策略 Worker 的专用 Python 解释器。 |
| `policy.checkpoint` | `NI_RUA_PI05_CHECKPOINT` | 适配 ARX X5 的官方 $\pi_{0.5}$ 检查点权重（完整默认路径见 `configs/ni-rua.example.json`）。 |
| `policy.server_url` | `NI_RUA_PI05_SERVER_URL` / `null` | 可选的复用预先启动策略推理服务的 URL。 |
| `features.grasp_memory_dir` | `NI_RUA_GRASP_MEMORY_DIR` | 跨运行抓取记忆存储目录。 |
| `model.codex_home` | `CODEX_HOME` | 本地 Codex 供应商路由和凭据的源目录。 |
| `model.price_card` | `NI_RUA_PRICE_CARD` | 用于 Token 成本估算的可选定价卡 JSON 路径。 |

| `images` 字段 | 默认值 | 约束与行为 |
| --- | --- | --- |
| `transport_format` | `webp` | `webp` 或 `jpeg`；若 WebP 编码不可用则自动回退到 `jpeg`。 |
| `webp_quality` | `85` | 整数，范围 `1`–`100`。 |
| `webp_method` | `4` | 整数，压缩算法努力程度，范围 `0`–`6`。 |
| `jpeg_quality` | `95` | 模型传输 JPEG 质量，整数，范围 `1`–`100`。 |
| `storage_jpeg_quality` | `95` | 归档 RGB 产物的 JPEG 质量，整数，范围 `1`–`100`。 |
| `feedback_policy` | `all` | `all` 返回所有相机视图；`adaptive` 对相对于上次交付给模型的图像未发生变化的例行动作和模板反馈视图进行去重，同时在初始观测、上下文压缩/会话切换后、显式 `robot_observe` 调用、失败以及关键交接检查时保留所有视图（可通过 `--image-feedback-policy` 覆盖）。 |
| `action_feedback_policy` | `all` | 动作后自动观测图像策略（`'all'`, `'on_anomaly'`, `'none'`, `'focus_roi'`, `'head_only'`, `'adaptive'`；CLI `--action-image-feedback-policy`）。在默认的 `all` 下，动作后观测交付更新后的相机视图；当设为 `'on_anomaly'` 时，带有有效 3D 感知的干净 `completed` 动作省略例行传感器图像，改用 `object_table`、`object_state_delta` 和 4D `process_trajectory`，仅在阶段停止/失败、`slip_detected=true` 或 3D 感知不可用时自动附带诊断图像；显式的 `robot_observe` 调用继续遵循 `feedback_policy` 或调用方的 `image_delivery`。 |
| `reuse_exact_images` | `true` | 在单个发往上游的模型请求内部复用字节完全相同的图像，不合并不同 `detail` 级别或修改压缩请求。 |
| `trim_max_inline_images` | `60` | 当启用 `trim_oversized_requests` 时，内联图像数量超过此阈值的发往上游请求会使用相同的“保留头尾、替换中间”窗口修剪至上限的大约 $2/3$（默认 40 张图像），从而保持在典型的 64 张供应商限制之内。设为 `0` 可禁用基于数量的修剪；非零值必须至少为 `24`。 |
| `trim_oversized_requests` | `true` | 当发往上游的 `/responses` 请求体超过 4.5 MB 时，`ModelRequestProxy` 保留初始和最新的图像，同时用带有 `observation_id` 的文本占位符替换中间轮次的图像（每个原生压缩周期最多触发一次），并在上游返回 `image_compression_failed` 时进一步修剪后重试。设为 `false` 则原样转发完整的 Codex 历史记录而不对 `image_compression_failed` 进行重试。独立于 `features.lean_projection` 运作；详见 [图像管线](image-pipeline.md)。 |

历史遗留的 `configs/images.json`、`NI_RUA_IMAGE_CONFIG` 和 `NI_RUA_IMAGE_SETTINGS` 继续作为兼容输入获得支持，其中统一配置的 `images` 节点优先级最高。相机分辨率、相机顺序、仿真采样率和动作时序属于固定的协议属性而非图像压缩设置，因此像素坐标契约保持不变。

<a id="evaluation-queue-experiment-library-and-standalone-workflows"></a>
## 评测队列、实验库与独立工作流

`scripts/benchmark_queue.py --config ...` 在队列调度与逐用例网关运行中均使用相同的统一配置文件。队列清单仍定义待运行的用例列表，清单中显式的逐用例字段可覆盖特定实验条件，省略的标志则继承自统一配置。启动时，队列写入 `effective_config.json` 用于审计和显存溢出（OOM）重试，因此随后修改磁盘上的配置文件绝不会改变活跃队列的运行条件。类似地，`scripts/benchmark_group.py --config ...` 会将解析后的配置直接嵌入生成的队列清单中。

每个清单用例可包含一个可选的非空字符串 `experiment_condition` 来标注其实验分组；该字段严格保存在队列清单和 `state.json` 中供实验库分组使用，绝不会传给网关命令行参数、运行时配置或模型上下文。

在预检（preflight）期间，队列会对每个目标冻结运行时检出中的 `ni_rua/core/config.py` 进行静态语法解析，而不执行目标代码。经过审计的向后兼容差异——即较旧的冻结代码库早于新增配置字段（`features.sandbox_enabled`、`robotic_view`、`stereo_guide`、`lean_compact`、`queue.eligible_gpu_indices`、运动阈值字段、外部感知字段以及 `analysis` 节点）的情况——会自动处理：当这些字段等于其当前类的默认值（例如 `sandbox_enabled=false` 或 `eligible_gpu_indices=null`）时，逐用例启动快照会省略未知的键，并在 `config.json` 内部的 `manifest.config_compatibility` 中记录该投影。若任何新字段被设置为非默认值，或检测到未经审计的 Schema 差异，队列会在 GPU 准入前拒绝该清单。队列清单保留完整的统一配置快照，OOM 恢复亦验证完全相同的兼容性投影。

| 配置字段 | 默认值 | 用途 |
| --- | --- | --- |
| `queue.max_parallel` | `60` | 新队列的最大并发用例数（显式 `null` 亦解析为 `60`，队列清单中的显式上限优先）。 |
| `queue.eligible_gpu_indices` | `null` | 合格 GPU 编号白名单（默认 `null` 启用所有可见 GPU；配置或清单中的显式非空列表会限制准入）。 |
| `queue.memory_per_case_mib` | `10240` | 每用例 GPU 显存预留（`10 GiB`，经过实际 RoboDojo Isaac Sim + cuRobo Worker 使用校准，使得 8-GPU 节点可在无 OOM 的情况下打满 `max_parallel=60`）。 |
| `queue.gpu_headroom_mib` | `0` | 每张设备额外要求的未分配 GPU 显存安全余量。 |
| `queue.wait_timeout_s` | `1800` | 等待可用 GPU 资源时的超时秒数。 |
| `queue.oom_retry.max_retries` | `2` | 确认原生 GPU OOM 后的自动全新回合重试次数（`0` 禁用重试）。 |
| `queue.oom_retry.initial_delay_s` | `60` | 首次 OOM 重试前的冷却延迟秒数。 |
| `queue.oom_retry.backoff_multiplier` | `2` | 后续 OOM 重试之间的指数退避乘数。 |
| `queue.oom_retry.max_delay_s` | `300` | OOM 重试之间的最大冷却延迟秒数。 |
| `queue.wake_enabled` | `true` | 启用或禁用队列完成通知回调钩子。 |
| `queue.wake_thread` / `wake_codex` / `wake_remote` | `null` / `codex` / `null` | 完成通知的目标会话 UUID、Codex 可执行路径与 app-server 套接字端点；当 `wake_thread` 为 `null` 时，回退到 `~/.config/ni-rua/completion-thread`（若存在），`wake_remote` 默认为用户的共享控制套接字（详见 [实验完成回调协议](experiment-completion-hook.md)）。 |
| `library.root` | `~/exp/ni-rua` | 常驻实验库服务扫描的根目录。 |
| `library.port` | `8780` | 实验库服务的回环 HTTP 端口。 |
| `library.scan_interval` | `1` | 后台目录扫描间隔（秒）。 |
| `library.public_origin` | `null` | 部署在带身份验证的反向代理后面时允许的精确 HTTPS Origin。 |
| `library.task_source_root` | `null` | 用于审查面板的任务评分源码读取的可选目录路径。 |
| `scene.root` | `runs/local_scene` | 独立场景服务的数据目录（嵌入式网关服务始终使用 `<run>/twin-service`）。 |
| `scene.port` | `8770` | 独立场景服务的回环 HTTP 端口。 |
| `scene.max_workers` | `2` | 场景服务中的最大并发受限 CPU Worker 任务数。 |
| `scene.max_queue` | `4` | 场景服务中的最大排队任务数。 |
| `scene.max_blob_store_bytes` | `536870912` | 场景服务的总 Blob 存储预算（默认 `512 MiB`）。 |

`NI_RUA_EXPERIMENT_CONFIG` 指向的旧实验设置文件以及现有队列清单/用例字段继续获得支持。新队列要求每个 `case.cwd` 均指向支持统一配置的代码检出；缺少 `--config` 的历史冻结检出在 GPU 准入或输出目录创建前会以 `unified_config_requires_supported_gateway` 导致预检失败。历史旧队列的 OOM 恢复会复用其记录的启动命令，不需要升级代码检出。官方基准测试预算不可被队列覆盖，且 OOM 重试仅在原生 OOM 证据与进程清理均获确认后触发——将每次重试写入全新目录，同时保留所有失败尝试。

你可以动态更新正在运行的队列的并发上限：

```bash
mamba run -n ni-rua python scripts/benchmark_queue.py --output /path/to/running-queue --set-max-parallel 15
```

该命令原子地更新目标队列目录内部的 `control.json`（`{"max_parallel": 15}`），不读取当前配置文件也不启动任务；它仅接受正整数，并拒绝已终止的队列或早于 `control.json` 的旧队列。调度器在每个轮询周期和每次用例准入前都会读取 `control.json`：调高 `max_parallel` 会在下一次循环迭代中立即触发准入检查，而无需等待 90 秒的 GPU 采样间隔；调低 `max_parallel` 绝不会杀死活跃用例，仅暂停补位直到活跃运行数降至新上限以下。GPU 显存预留和共享租约规则依然有效。无效或不可读的 `control.json` 会被忽略，保持先前限制并在每个不同连续错误时记录一次 `queue_control_invalid`。

一旦 `control.json` 写入，CLI 会打印 `requested`；通过 `state.json.scheduling.max_parallel` 和 `events.jsonl` 中的 `max_parallel_changed` 事件（记录更改时的旧值、新值与活跃运行计数）确认实际生效。初始的 `manifest.json`、`config.json` 和 `effective_config.json` 保持不可变启动快照，运行时调度调整由 `state.json` 和 `events.jsonl` 跟踪。从较早代码启动的队列进程需要使用当前入口重新启动才能支持动态 `control.json` 更新。

`scripts/motion_review_eval.py prepare --config ...` 从统一配置继承模型、推理努力等级、供应商主目录、机器路径和图像设置，将解析后的配置冻结进成对评测协议与生成的队列清单中；汇总统计核验保存的配置而非读取当前默认值，早先生成的 `gpt-6-astra` 协议继续使用其记录的模型和推理努力等级。

在 `scripts/reference_episode.py` 中，直接 API 客户端与机器人后端共享单个解析后的 `Config` 实例；`--config`、`--model`、`--effort` 和 `--codex-home` 可显式覆盖设置，`--print-config` 可在不读取凭据或启动模型/机器人的情况下打印实际生效配置。新参考运行默认使用 `~/.codex` 与 `gpt-6.1-sol` / `high`，在每个请求中发送模型名称、`reasoning.effort` 和 `service_tier="default"`。由于该独立客户端直接调用上游 API 而不经过 Codex app-server，它严格从所选供应商声明的 `env_key`、其 Bearer 令牌配置或所选主目录下 `auth.json` 中的 API Key 读取凭据——不支持交互式 ChatGPT 网页登录令牌，且除非 `env_key` 显式声明，否则绝不回退到环境中的 `OPENAI_API_KEY`。缺失的供应商、端点或凭据会在发送任何模型请求前快速失败报错。

下述独立操作与诊断命令属于单次工作流，由其自身的命令行参数或 HTTP 载荷控制，而非全局布尔功能开关：

| 工作流 | 入口点与控制 |
| --- | --- |
| 独立场景客户端重放 | `scene_client` 参数（`URL`, `delay`, `duplicate`, `reverse`）属于重放测试参数；场景服务自身支持统一的 `scene` 配置节点以及 `--config` 与 `--print-config`。 |
| 固定 API 参考策略 | `scripts/reference_episode.py` 在其独立参考协议下接受 `--config`、模型覆盖参数、`--max-calls`（`1`–`170`）、`--action-timeout-s`（`0.1`–`300`）以及任务/种子/GPU 参数。 |
| 原生策略诊断 | `native_policy_validate.py`、`pi05_atomic_handoff.py` 及相关探测接受实验专用输入，如 `prefix`、`prompt`、`noise seed`、`hold-limit`、`episode` 和 `ee-smoke`。 |
| 动作审查预检（2026-10 移除） | `motion_review_preflight.py --output ... --wait-timeout-s 600` 使用独立的诊断等待超时，不受 1800 秒队列默认值的约束。 |
| 几何校验 | `geometry_validate.py` 参数（`synthetic`, `raw-recording`, `ROI`, `heldout`）为单次校验运行选择数据集与评测切片。 |
| 观测图像压缩 | `compact_observation_images.py --apply` 显式从已封存的归档中移除已核验的原始图像，绝不在回合进行期间隐式运行。 |
| 常驻实验库服务安装 | `install_library_service.py --install` 安装并启用用户 `systemd` 单元（省略 `--install` 时将单元定义打印到 `stdout`）。 |
| 观察者端点 | `/v1/observer/record`、原生评测、隐藏场景快照以及视频导出由 Harness/操作员显式调用，绝不注册为 Actor 工具。 |
| 归档导入与报告 | `ni_rua.apps.library.experiments` 与 `benchmark_report.py` 接受单次输入/输出路径与过滤标志。 |

<a id="resumption-replay-and-reproducibility"></a>
## 会话恢复、重放与可复现性

每次运行的归档都会在历史清单字段旁保存完整的已解析配置，以便旧版分析脚本继续正常工作。保存的快照记录了完整的生效运行时条件，而非仅记录用户提供的局部覆盖。

当恢复或重放既有运行时，实验语义严格从归档中恢复，而不读取磁盘上的当前默认文件（而本地服务参数如 HTTP 监听端口仍可更改）。缺少 `enable_wait` 的旧清单保持其为 `false`，缺少 `vision_compare` 的保持其为 `true`，缺少 `actor_payload_profile` 的默认回退到 `full_v1`，缺少 `default_task_notes` 的保持为 `false`。在回复投递设置诞生前写入的快照或清单在恢复时将其视为缺席（`budgeted_action_reply`、`reply_same_as`、`reply_float32`、`reply_static_once`、`reply_compact_encoding` 均为 `false`，`action_reply_profile` 为 `full`；`REPLY_DELIVERY_ABSENT`），而在目录设置诞生前写入的快照将 `tool_catalog_profile` 恢复为 `full_v1`、将 `robot_urdf_delivery` 恢复为 `inline`（`CATALOGUE_DELIVERY_ABSENT`；两组均为 `FEATURE_ABSENT_VALUES`），从而确保恢复或重放的运行保持其实际运行时的形态；每个键均独立恢复（记录了更早出现的 `tool_catalog_profile` 但未记录 `robot_urdf_delivery` 的归档保留其 Profile 并获得 `inline`），声明了键的快照则保留其声明的值。当评测队列将配置映射到缺少这些字段的冻结代码检出时亦套用相同数值：仅缺席值可被忽略，当前默认值则执行闭合失败策略；对于具有 `tool_catalog_profile` 或 `robot_urdf_delivery` 但列出的选项不含所请求值的代码检出，队列停止并报错 `unified_config_incompatible:features.<field>=<value>`，而不是强行发送或丢弃。检查点切片（`turn_probe`）亦恢复相同的数值。完全缺失配置对象的运行（运行时的测试替身）发布散文形式的 `full_v1` 目录；每个 Harness 均持有一份配置。在引入 `default_task_notes` 或 `vision_compare` 之前保存的统一配置快照同样恢复其各自的历史默认值。Task Notes 始终复用归档中记录的精确正文、来源和版本，而不重新生成或追加当前的默认笔记。缺少 `reuse_exact_images` 的旧图像设置保持为 `false`，缺少 `trim_max_inline_images` 的恢复为 `0`（仅按字节大小修剪）。若已归档的模型请求记录账本中没有 `transport_recovery` 记录，自动传输恢复在恢复运行时保持禁用；否则按照记录的策略恢复。请记住，`--resume` 会创建全新的仿真器回合/代数，而非恢复回合途中的物理状态，同时保留累积的线程预算与 Token 记账。

因此，修改 `configs/ni-rua.json` 中的默认设置仅影响新发起的运行。若要进行受控的成对消融实验，请创建两个全新的运行目录，保存其各自解析后的配置，并在每个 `manifest.json` 中核验实际的工具目录与观测配置档案——绝不要假定历史运行会自动迁移到较新的默认值。

<a id="fixed-runtime-protocol-and-boundaries"></a>
## 固定运行时协议与边界

以下常量与能力边界是有意固定在代码中的，而不作为任意的 JSON 配置开关暴露，从而确保修改配置文件不会意外改变基准测试规则或侵犯 Actor 信息边界：

| 范畴 | 固定协议规则 |
| --- | --- |
| 官方模型预算 | 每回合 `180` 次已交付的有效 Actor / Helper / 压缩响应；没有硬编码的单请求墙钟超时；明确可重试的上游 API 错误由 `Harness` 自动退避重试最多 `12` 次（SDK 内部重试设为 `0`），随后根据 `model.transport_recovery` 进行有界冷却重发（详见 [模型主导操作](model-led-operation.md)）；失败或不完整的尝试单独记账。 |
| 官方运行监督 | `900` 秒环境初始化截止时间；活跃运行期间，仅在连续 `1,800` 秒没有任何模型、程序、机器人或仿真器活动后停止；官方基准测试没有整回合墙钟上限。 |
| 决策边界 | 物理动作之间只读工具查询的次数无人为上限；完成一次模型轮次不终止原生回合。 |
| 原生动作 | 逐任务原生步长限制、任务成功标准与控制频率由 RoboDojo 控制，具有单一物理动作写入者与幂等回执。 |
| 统一动作 Schema | `robot_action` 每批接受 `1`–`32` 个阶段（每臂最多 `16` 个阶段）；双臂沿共享时间线执行，臂内顺序、跨臂依赖与 `observe` 检查点由批处理契约强制保证。`robot_wait` 与策略前缀单次调用最多执行 `50` 个原生步长，动作超时时间限定在 `0.1`–`300` 秒。 |
| 历史原子动作 | 历史单臂 `EEAction` 请求最多支持 `8` 个连续路径点，仅供受信的兼容调用方使用，不限制 `robot_action` 批次。 |
| 动作审查接触诊断 | 最终连续进近窗口在阶段目标 `60 mm` / `20 deg` 范围内；放置接触允许最多 `3 mm` 几何穿透容差（默认 `2 mm`），报告数值风险而不阻断执行，同时保留原始重叠事实。 |
| 跟踪保护 | 当关节跟踪误差连续 `5` 个原生步长超过 `0.5 rad` 时触发，中止动作且不自动修改轨迹。 |
| 关节重规划 | 仅在动作超时剩余至少 `15` 秒时尝试，仅当候选轨迹完全经过验证且原生步长更短时予以替换。 |
| 程序资源限制与预算（在两种沙箱模式下均强制执行） | `1 GiB` 地址空间、`8 MiB` 文件大小限制、`128 KiB` 源码限制；每程序活跃墙钟预算默认 `180` 秒（上限 `300` 秒），不包括辅助模型调用与主模型动作审批的等待时间；工具调用次数无上限；单线程数值库，动作准入留有 `5` 秒安全余量。 |
| 独立场景服务限制 | 每个 Blob 最大 `32 MiB`；每个任务的队列与墙钟预算默认各为 `30` 秒（逐请求校验），Worker 数量、队列深度与总 Blob 存储在 `scene` 节点下配置。 |
| 原生 Codex 能力 | 原生动态工具直接返回文本与图像，可选的本地工具搜索按需加载已注册工具；启用 Shell、文件操作与原生记忆，禁用 Web 搜索与外部扩展。操作系统命名空间与 `program_run` Landlock/seccomp 隔离由 `features.sandbox_enabled` 控制，原生线程准入严格将机器人工具调用限制在主 Actor 线程。 |
| 自主执行 | 启动指令要求主模型自主决定计划、参数与程序动作审批，无需询问人类操作员；原生用户输入请求立即接收空答案，活跃会话上的命令/文件审批自动返回 `acceptForSession`，请求的权限在会话范围内无须人类介入即可授予。Harness 观测投影与动作准入始终强制执行，且在 `features.sandbox_enabled=true` 时 Landlock 保持生效。 |
| 观测权限 | 原生深度默认禁止，仅在显式 RGB-D 配置下允许；所有配置下均严禁获取隐藏物体位姿、接触真值与私有评测器状态。 |
| 视频时序 | 三相机录制在仿真时间以 `5 Hz` 采样，外加观测边界帧，保留精确的非均匀帧映射，而非以全原生物理速率录制。 |
| 停止与评分 | 未确认的停止、`unknown` 动作结果与基础设施故障在评分策略下保持未评分，绝不能通过配置虚构任务成功或掩盖失败。 |

直接工具模式通过每运行私有的 `model_catalog_json` 固定，该文件保留了来自 Codex SDK/CLI `0.160.0` 的原生模型目录，同时将 `tool_mode` 设置为 `"direct"`（因为原生模型元数据优先于功能开关，仅禁用 `code_mode` 是不够的）。模型名称、推理努力等级、上下文窗口与其他能力保持不变；关于目录来源、校验和与升级流程，请参阅 [`third_party/codex/README.md`](../../third_party/codex/README.md)。

若未来的研究需要对这些固定协议常量进行实验性变更，请引入显式的带版本配置字段，附带校验、清单记录与回归测试，并明确区分官方基准测试运行与探索性开发运行。

对于 `--benchmark` 和 `--exit-after-turn` 运行（交互式开发工作台在空闲时不自动退出），运行时监督自最近一次真实活动起计算不活跃时间：发起模型请求、接收到真实的上游传输字节、发出代码执行事件/输出、推进机器人/仿真执行均会重置 30 分钟（`1,800` 秒）的不活跃计时器。仅有挂起的请求、合成保活、运行时状态心跳或存活的子进程不会重置计时器。只要有任何子系统取得真实进展，回合就绝不会因总耗费墙钟时间而终止；若所有子系统持续静默 1,800 秒，活跃请求将被取消，运行进入停止与资源清理序列，将 `runtime_inactivity_timeout` 记录为未评分的基础设施中断。明确可重试的 API 错误由 `Harness` 自动重试；一旦即时重试耗尽或发生上游传输断开——并且在此期间没有物理动作在执行或未确认、没有程序处于活跃状态——中继会进入冷却期并根据 `model.transport_recovery` 重新发送完全相同的请求，仅在不可重试错误、恢复门禁被拒或恢复预算耗尽时终止。详见 [模型主导操作](model-led-operation.md)。

达到 900 秒环境启动截止时间将关闭动作准入并启动资源清理，但这并不保证整个网关进程会在 900 或 905 秒内退出。[`RuntimeSupervision`](../../ni_rua/runtime/supervision.py) 保持对启动任务的所有权，而 [`Workbench`](../../ni_rua/apps/gateway/workbench.py) 每 50 毫秒轮询一次发布的机器人 Worker、SDK 桥接、app-server 和终端子进程句柄（包括发出停止请求后衍生的进程），先发送 `SIGTERM`，并在 5 秒后升级为 `SIGKILL`，同时记录信号与退出证据。

在 [`RobotProcess`](../../ni_rua/robot/backends/robodojo/robot_process.py) 中，单次请求截止时间涵盖获取写锁、写入管道、发送取消操作以及等待 Worker 响应；部分写入或缺少回执将保持为 `unknown`，不进行自动动作重放。管道故障后，计划的机器人关闭等待时间总计最多 27 秒，而在最坏分支下标准关闭最多耗时 237 秒（涵盖关闭 RPC、进程退出与读取线程 join）。这些边界排除了操作系统调度延迟或不可中断的内核 I/O；无法安全终止的 Python 或原生线程仍由其对应对象拥有，而不是被虚假记录为已释放。强制执行严格的整进程退出截止时间需要外部进程监控器。关于 CPU 子进程回归测试，请参阅 [`test_robot_process_deadline.py`](../../tests/robot/robodojo/test_robodojo_motion_runtime.py) 与 [`test_runtime_shutdown.py`](../../tests/runtime/harness/test_harness_lifecycle_and_shutdown.py)。

`features.arm_queue` 和 `features.motion_cost_preview` 默认启用，可通过 `--[no-]arm-queue` 和 `--[no-]motion-cost-preview` 进行覆盖。两者均需要 `resampled` 执行模式，且 `motion_cost_preview` 额外需要已标定的观测配置。禁用 `arm_queue` 也会拒绝原始的 `ArmQueueAction` 调用，禁用 `motion_cost_preview` 也会拒绝对应的程序调用。队列清单可以将 `eligible_gpu_indices` 指定为唯一非负 GPU 编号的列表；省略时，所有可见 GPU 均保持合格以供自动准入；设置时，被排除的 GPU 绝不会被选定，而合格的 GPU 仍遵循共享 GPU 租约与显存检查。

<a id="bounded-continuation-execution-checks-and-manipulation-evidence"></a>
## 有界延续、执行检查与操作证据

`features` 下的四个独立布尔标志对新运行默认均为 `true`，可通过命令行参数 `--[no-]bounded-continuation`、`--[no-]motion-safety-guard`、`--[no-]trajectory-preview` 和 `--[no-]manipulation-protocol`（以及队列清单用例中匹配的 snake_case 字段）进行控制。在恢复或重放缺少这些字段的历史清单时，它们默认为 `false` 以保留历史行为：

- `bounded_continuation`：首个完成的 `final` 答案建立基线；此后若未发生物理动作进展，在没有新工具证据的情况下连续 `3` 次给出 `final` 答案或总计最多 `6` 次恢复 `final` 轮次后，回合终止（原生回合终止始终优先）。新的只读工具探索仅授予有限的恢复轮次，且绝不查阅私有评测器分数来做此决定。
- `motion_safety_guard`：在执行前检查编译后的关节轨迹是否存在异常关节绕道/缠绕以及剩余的原生步长预算；若被拒绝，返回诊断证据而不改变关节指令或自动重规划，同时保留已执行的步长计数。
- `trajectory_preview`：向动作成本预览附加候选关节配置、公开前向运动学位姿以及同步的双臂模型净空距离，并启用 `robot_action(mode="inspect")` 在特定原生步长查询轨迹状态或将其投影到当前 RGB 图像上。严格使用公开机器人模型、实测关节反馈、公开标定与候选指令——绝不使用任务物体的真值（详见 [运动安全与轨迹证据](motion-safety-and-trajectory-evidence.md)）。
- `manipulation_protocol`：使 Task Notes、`task_plan` 和有界模板围绕显式接触位置、放置几何与观测证据对齐。未知的几何可记录为 `unknown`，完成的夹爪释放标记为 `released_unverified` 直到通过观测核验。双臂交接权限与偏好仍由 `airborne_handoff` 和 `handoff_preference` 控制，完整保留双臂并行执行。

SO-101 标定控制及其离线 SAM/LK/VGGT 证据工具拥有独立的[配置归属与优先级指南](so101-configuration.md)，涵盖共享证据 JSON 与显式基于观测推导的 VGGT 支撑面高度。
