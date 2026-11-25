# 运行指南

[English](running.md) · [简体中文](running.zh-CN.md)

本指南涵盖环境搭建、单局运行启动、会话恢复、只读回放与批量基准评测。框架整体架构见 [README](../../README.zh-CN.md)；SO-101 真机硬件准备与标定见 [SO-101 真机快速上手指南](../guides/so101-quickstart.zh-CN.md) 与 [SO-101 标定指南](../guides/so101-calibration.zh-CN.md)；使用你自己的 Claude Code、Codex 或 OpenCode 驱动运行见 [Agent 插件指南](agent-plugin.zh-CN.md)；配置字段与特性开关见 [配置指南](configuration.zh-CN.md)；当前统一动作接口见 [Unified Actions and Program Approval](unified-actions.md)。

<a id="environment-setup"></a>
## 环境准备

Harness 的开发、测试、脚本与服务一律运行在 Miniforge 管理的 `ni-rua` 环境中，严禁创建项目 `.venv`。在环境未激活时，命令统一以 `mamba run -n ni-rua python` 为前缀，或先显式激活 `NI-RUA` 再运行 `python`。机器人 Worker 会调用独立安装的 RoboDojo / Isaac Sim 环境（见 [RoboDojo 运行环境配置](robodojo-setup.zh-CN.md)），或者在驱动 SO-101 真机时调用隔离的 `so101` 硬件环境（`mamba env create -f environment.so101.yml`；见 [SO-101 真机快速上手指南](../guides/so101-quickstart.zh-CN.md) 与 [SO-101 标定指南](../guides/so101-calibration.zh-CN.md)）。可选的 OpenPI 策略同样运行在各自独立的 Python 环境中，不与 Harness 环境合并仿真或策略依赖。

运行前请确认已具备以下条件：

- 完整的 Git 源码仓库，以及 `environment.yml` / `requirements.lock.txt` 中声明的依赖。
- 已安装好的 RoboDojo 仓库、任务资产、兼容的 Isaac / CUDA 运行时（固定版本与补丁见 [RoboDojo 运行环境配置](robodojo-setup.zh-CN.md)），以及一张空闲 GPU。
- 本地原生 Codex 配置，其中包含可访问所选模型的供应商与鉴权凭据；新实验默认使用 `~/.codex`、模型 `gpt-6.1-sol` 与推理强度 `high`。供应商配置、鉴权及旧版兼容细节见[配置指南](configuration.zh-CN.md)。
- 操作系统沙箱默认开启（`features.sandbox_enabled=true`），要求 Linux 宿主机支持 Landlock ABI >= 4 与 `libseccomp`；若不满足隔离前置条件，框架会直接拒绝启动，绝不静默降级。
- 沙箱开启（默认）时，主 Codex 会话额外依赖从当前环境发现的 Linux、`bwrap`、系统 `python3`、C++ 运行时库与 `ldd` / `ldconfig`。优先使用非特权用户命名空间；仅在宿主机禁用了用户命名空间时，才会回退使用 `sudo -n` 与 `setpriv` 执行降权。在 Debian/Ubuntu 上对应软件包为 `bubblewrap`、`util-linux`、`python3`、`libc-bin`、`libstdc++6`，另加 `ripgrep` 以向 Claude Code 与 OpenCode 后端的原生搜索工具提供 `rg`（Codex 使用其固定发行包自带的 `rg`）。沙箱在当前 UID/GID 下运行并清除全部 capabilities，隔离不可用时会中止启动。沙箱内禁用联网搜索，Python 包仅允许通过受控 PyPI 代理安装至私有工作区；路径与 IPC 细节见[配置指南](configuration.zh-CN.md)。

当显式关闭沙箱（`features.sandbox_enabled=false`）时，主/辅 Codex 会话与 `program_run` 不启用 OS 级隔离，执行的代码拥有与启动用户相同的宿主机文件与网络权限，但 Harness 的观测投影、动作准入与程序审批机制仍然生效。在缺少沙箱前置条件的宿主机上如需关闭 OS 隔离，可通过 `--config` 传入 `{"version": 1, "features": {"sandbox_enabled": false}}`；完整说明见[配置指南](configuration.zh-CN.md)中的代码执行沙箱章节。

在仓库根目录创建主环境并构建 Web 前端（若 `ni-rua` 环境已存在可跳过创建步骤）：

```sh
mamba env create -f environment.yml
mamba run -n ni-rua python -c "import sys; from importlib.metadata import version; print(sys.executable); print('SDK', version('openai-codex')); print('CLI', version('openai-codex-cli-bin'))"
npm --prefix web ci
npm --prefix web run build
```

NI-RUA 使用随 Python SDK 打包的原生 Codex 二进制，而非 `PATH` 上的包装脚本，并会在每次运行中归档解析出的 SDK/CLI 版本与二进制路径。启动或恢复运行前，NI-RUA 会校验两个包的版本是否与项目协议快照一致；任何不匹配都会在停止既有进程或拉起新 app-server 之前抛出 `codex_version_mismatch`。请始终通过 `requirements.lock.txt` 更新 `ni-rua` 环境中的依赖，而不要只升级 `PATH` 上的全局 `codex` 二进制。外部 RoboDojo 版本、补丁及可选 OpenPI 环境见 [RoboDojo 运行环境配置](robodojo-setup.zh-CN.md)。

如需将 NI-RUA 安装到隔离位置，请使用下方脚本构建完整 wheel。构建脚本会在临时目录中汇总全部 Python 子包、已编译 Web 资源、策略桥接脚本与固定参考材料，不会改动源码仓库或部署服务。请始终使用 `scripts/build_package.py`，切勿直接运行 `pip wheel .`（后者会漏掉未打包的 Web 资源）：

```sh
mamba run -n ni-rua python scripts/build_package.py --output dist
mamba run -n ni-rua python scripts/check_package.py dist/*.whl
mamba run -n ni-rua python -m pip install dist/*.whl
```

`check_package.py` 会用当前解释器执行 `pip install --no-deps --target` 安装到临时目录，并在 Python `-I` 隔离模式下于仓库外验证 wheel，全程不创建也不使用 `.venv`。安装进 `ni-rua` 环境后，即可直接使用命令行入口 `ni_rua`、`ni-rua-queue`、`ni-rua-report` 与 `ni-rua-audit`。原有源码脚本 `scripts/benchmark_queue.py`、`benchmark_report.py` 与 `audit_run.py` 继续转发至相同实现。RoboDojo、Isaac 与可选策略环境仍保持外部配置；wheel 绝不打包模型权重、凭据或外部仿真环境。固定基准组生成器与开发探针仍需在完整源码仓库中运行。

默认配置文件为 [`configs/ni-rua.json`](../../configs/ni-rua.json)。对于纯本地无 GPU 运行的 SO-101 真机，使用 `configs/so101_standalone.json`（首次开箱尚未完成 `/calibration` 标定时需附加 `--observation-profile rgb_proprio_v1`）；仅在对接可选远程 VGGT/SAM/cuRobo GPU 服务时改用 `configs/so101_gateway.json`（详见 [SO-101 真机快速上手指南](../guides/so101-quickstart.zh-CN.md)）。你可以在 Git 忽略的 `configs/ni-rua.local.json` 中覆盖机器专属默认值；供应商配置与 API 密钥请存放在仓库外的外部 Codex 主目录中，本地 JSON 里只记录其路径（见[配置指南](configuration.zh-CN.md)）。如需自定义本机路径或模型配置，可从 [`configs/ni-rua.example.json`](../../configs/ni-rua.example.json) 复制相关字段到自己的配置文件中。默认后端路径为 `~/code/robodojo` 与 `~/miniforge3/envs/RoboDojo/bin/python`，也可通过 `NI_RUA_ROBODOJO_ROOT` 与 `NI_RUA_ROBODOJO_PYTHON` 覆盖。

```sh
# 仅检查解析后的生效配置，不启动机器人、模型或服务。
mamba run -n ni-rua python -m ni_rua.apps.gateway \
  --config configs/my-experiment.json --print-config
```

配置合并优先级从低到高依次为：类默认值 → 兼容的旧版环境变量 → 仓库默认配置 → 本地默认配置（`configs/ni-rua.local.json`，若存在）→ 显式配置文件 → 显式 CLI 参数。每次新运行都会快照保存其生效配置，恢复与回放则读取归档配置。模型凭据与机器密钥严禁提交进版本库。

<a id="launching-a-single-run"></a>
## 启动单局运行

运行默认以无人值守（unattended）模式执行。模型自主决定策略、工具参数与程序动作审批，自行处理恢复与推进，不向人工操作员弹窗提问。原生用户输入请求会立即收到空答复，属于当前会话的命令/文件审批会自动以 `acceptForSession` 批准。开启 `features.sandbox_enabled`（默认）时，附加权限严格限定在外层命名空间内且程序受 Landlock 隔离；显式关闭时，原生命令可访问启动用户有权的宿主机文件与网络资源，而 Harness 观测投影、动作准入与程序审批仍保持生效。人工停止控制随时可用。

在机器上选择一张空闲 GPU，并为新运行指定一个尚不存在的目录。以下命令会同时拉起实时模型会话与机器人仿真：

```sh
mamba run -n ni-rua python -u -m ni_rua.apps.gateway \
  --benchmark --task general_pickup --seed 0 --layout-id 0 \
  --gpu 0 --port 8767 --run-dir runs/my-first-task
```

在浏览器中打开 `http://127.0.0.1:8767`，即可查看机器人相机画面、原生 Codex 终端、已执行动作与运行记录。`--benchmark` 使用环境提供的公开原生任务指令，并按原生步数上限与终态条件运行；结束后会自动执行独立评测器、导出视频并释放硬件资源。

开发调试期间，可以用自定义提示词和受限预算执行接口检查：

```sh
mamba run -n ni-rua python -u -m ni_rua.apps.gateway \
  --gpu 0 --port 8767 --run-dir runs/my-interface-check \
  --prompt 'Observe the current scene, submit a plan, execute a small end-effector motion, and inspect the feedback.' \
  --max-actions 2 --wall-budget-s 900 --exit-after-turn
```

`--max-actions` 与 `--wall-budget-s` 仅限开发运行，正式 `--benchmark` 运行会拒绝它们。正式基准保留模型请求预算与无进展超时保护；精确计量与终止语义见[配置指南](configuration.zh-CN.md)与 [Model-Led Operation](model-led-operation.md)。

<a id="interaction-stopping-and-resuming"></a>
## 交互、停止与恢复

Web UI 交互通过 `Harness` 提交到同一个原生 Codex 线程，写控制权通过租约（lease）协调。Web 终端展示原生输出，新提示词与停止请求由网关处理。无人值守请求会自动处理并留痕，因此对已处理或过期的审批/提问弹窗进行人工回复会被拒绝。当没有浏览器会话持有写租约时，也可以通过 CLI 提交提示词：

```sh
curl -fsS http://127.0.0.1:8767/v1/control \
  -H 'Content-Type: application/json' \
  --data '{"request_id":"review-1","command":"prompt","prompt":"Review the previous request and measured feedback; do not move the robot."}'
```

每次请求的 `request_id` 必须唯一；重发完全相同的载荷会返回原始响应，而用同一个 `request_id` 发送不同内容会被拒绝。仅当上一回合与动作均已结束且当前运行仍接受工作时，才会受理新提示词。

在 Web UI 点击停止或发送 `command: "stop"` 时，系统会先关闭动作准入并请求取消，再中断模型；最终停止状态以实际执行回执为准。关闭浏览器标签页不会停止机器人。发送 `Ctrl-C` 或 `SIGTERM` 可彻底关闭网关服务。`SIGHUP`（关闭运行网关的终端）会走同样的有序退出路径，且当控制台输出无法继续写入（关闭的 tty 或 `| tee` 管道）时会直接丢弃输出而不会中断收尾，从而确保 `summary.json` 与 `runtime.json` 正常落盘。忽略 `SIGHUP`（如 `nohup`）启动的网关在终端关闭后会继续运行。`SIGKILL` 或进程崩溃无法被捕获：此时运行不会生成 `summary.json`，心跳超时后实验库会将其显示为 `unknown`。

```sh
# 只读回放，既不需要 GPU 也不需要模型。
mamba run -n ni-rua python -m ni_rua.apps.gateway \
  --replay --run-dir runs/my-first-task --port 8767

# 恢复开发运行的原始线程（会创建全新的物理回合）。
mamba run -n ni-rua python -m ni_rua.apps.gateway \
  --resume --run-dir runs/my-interface-check --port 8767
```

恢复运行不会重建先前的物理仿真器状态——重新加载的环境会启动新的 episode/generation，且任何仍在运行的网关或机器人 Worker 进程都会阻止恢复。`--benchmark` 不能与 `--resume` 组合使用；单独传入 `--resume` 会复用归档的基准设置，但新回合不构成原始 rollout 的连续物理执行。`features.sandbox_enabled` 必须与归档会话模式一致，切换沙箱模式需要启动新运行。查看历史结果请使用只读 `--replay`。

<a id="experiment-library-and-batch-evaluation"></a>
## 实验库与批量评测

省略 `--run-dir` 时，新运行会自动按 UTC+8 日期归档至 `~/exp/ni-rua/runs/YYYY/MM/DD/category/task/run-id/`。可通过 `--category`、`--title` 与 `--tag` 为运行打标。实验库默认监听 `http://127.0.0.1:8780`，提供检索、实时工作台入口与只读回放：

```sh
mamba run -n ni-rua python -m ni_rua.apps.library
```

持久化 systemd 部署、原始归档导入与远程访问配置见 [Experiment Library](experiment-library.md)。克隆源码仓库不会复制历史运行归档、不会部署系统服务，也不会授予远程访问权限。

固定基准套件 [`gpt-as-policy-10`](../../benchmarks/gpt-as-policy-10/README.md) 包含 50 个用例。请先离线生成全新的 manifest，再通过评测队列启动（输出文件与队列目录均必须尚不存在）：

```sh
mamba run -n ni-rua python scripts/benchmark_group.py --group gpt-as-policy-10 \
  --run-root ~/exp/ni-rua/benchmark/gpt-as-policy-10/trial-001 \
  --output /tmp/gpt-as-policy-10-trial-001.json

mamba run -n ni-rua python -m ni_rua.apps.evaluation.queue \
  /tmp/gpt-as-policy-10-trial-001.json \
  --output ~/exp/ni-rua/queues/gpt-as-policy-10-trial-001
```

基准组协议不会覆盖你的本地模型或推理强度设置。队列会根据解析后的配置管理 GPU 准入、并发度与故障恢复，并在每次运行中精确记录版本、种子、观测配置及所有协议差异。可借助该基准组定义在完全相同的用例集合上评估改动；严格复现还需核对上游版本与全部运行时条件。

如需在不启动机器人或模型的情况下离线汇总多局已完成运行的统计指标：

```sh
mamba run -n ni-rua python scripts/benchmark_report.py \
  /path/to/run1 /path/to/run2 --output /path/to/new-report.json
```

<a id="verification-and-standalone-services"></a>
## 验证与独立服务

在仓库根目录运行标准验证套件：

```sh
mamba run -n ni-rua python scripts/run_tests.py -v
npm --prefix web run check
npm --prefix web run build
```

如需审计运行证据，请执行 `mamba run -n ni-rua python scripts/audit_run.py <run-dir>`。默认模式下会检查记录关联与产物完整性，同时保留未确认动作，且不强制要求特定模型或原生日志。传入 `--profile native` 会进一步校验原生模型调用、实际图像投递、已执行动作与已完成回合；再加 `--require-continuity` 还会要求多回合连续性。如果原图已被压缩导致源字节无法核验，`native` 审计档位会显式报错。任何审计档位均不能替代原生评测器的任务计分。

`mamba run -n ni-rua python scripts/benchmark_report.py <run-dir> ...` 会在汇报整体可读覆盖率的同时，保留损坏归档的逐目录错误；缺失的 token 用量会保持为 `unknown`，而不会被当作零成本或任务失败。评测队列会在启动时以及每个用例启动前校验源码文件的字节摘要，一旦检测到源码漂移便会立即停止后续调度并清理其名下进程。正式基准测试请务必使用专用且未改动的独立 checkout，因为启动前校验无法在操作系统层将可写工作树变成只读。

如需运行真实机器人验证矩阵，执行 `mamba run -n ni-rua python scripts/robot_validation.py --gpu <idle-gpu> --run-dir <new-dir>`，该脚本会分配一张 GPU 并执行物理机器人运动；通过 CPU 单元测试不能替代真实任务验证。项目没有托管 CI：请按 [Validation Guide](validation.md) 在本地执行 CPU 测试套件、Schema 校验、Web 前端检查与仓库外 wheel 安装校验；真实 GPU/模型验收运行作为实验记录独立归档。

独立场景服务可通过 `mamba run -n ni-rua python -m ni_rua.perception.scene.scene_service --root <data-dir> --port 8770` 启动，并在 `/v1/openapi.json` 暴露 OpenAPI 规范。其支持导入获准观测、有界几何重建、版本管理与快照；未实现完整 Isaac 物理推演。该模块对被测环境不持有动作或重置权限，也不会以 `scene_*` 工具形式暴露给新任务。更多设计文档与验证入口见[文档索引](../README.zh-CN.md)。
