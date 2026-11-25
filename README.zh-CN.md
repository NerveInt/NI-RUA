<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/ni-rua-lockup-dark.svg" />
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/ni-rua-lockup-light.svg" />
  <img src="docs/assets/ni-rua-lockup-light.svg" alt="NI-RUA" width="260" />
</picture>

<p><b>面向具身机器人操作的智能体执行框架</b></p>

[English](README.md) · [简体中文](README.zh-CN.md) · [文档索引](docs/README.zh-CN.md) · [许可证 (MIT)](LICENSE)

</div>

---

## NI-RUA 是什么？

**NI-RUA** 让编程智能体（**Claude Code**、**Codex** 或 **OpenCode**）直接操作真机与仿真机器人。智能体查看相机画面与关节状态、制定计划并调用动作工具；NI-RUA 负责守住物理执行边界，实时校验多阶段容差与安全护栏，并保存完整可回放的运行归档。

---

## 它能做什么？

| 核心能力 | 说明 |
| :--- | :--- |
| **真机与仿真统一驱动** | 支持 **SO-101** 桌面真机机械臂（macOS / Linux，纯 CPU 开箱即用，也可按需对接远程 GPU 3D 感知与 IK）、[RoboDojo](https://robodojo-benchmark.com/) 仿真双臂 **ARX X5**，以及 [LIBERO](https://libero-project.github.io/) 仅仿真（simulation-only）单臂 **Franka Emika Panda**（`libero-franka-panda`）。 |
| **自带 Agent 或完整托管** | 可通过 MCP 插件（`ni-rua-plugin`）直接接入你常用的 Claude Code / Codex / OpenCode，也可由 NI-RUA 托管运行智能体、OS 沙箱、请求预算与批量评测队列。 |
| **安全闭环与透明护栏** | 支持预览-执行分离、可配置阶段容差、防碰撞与防空抓护栏，以及事实与展示严格分离的结构化物理回执。 |
| **双语控制台与实验库** | 内置中英双语 Web 界面，提供实时相机、S0–S6 引导标定（`/calibration`）、硬件级 **停止 / 保持 / 急停 E-STOP**（`/console`）、阶段回放与跨方案对比实验库。 |

---

## 快速上手

### 1. 安装

```bash
git clone https://github.com/NerveInt/NI-RUA.git
cd NI-RUA

# 安装主框架环境（Python 3.12）
mamba env create -f environment.yml

# 构建 Web 控制台前端（Node 22）
npm --prefix web ci && npm --prefix web run build
```

> 使用 **SO-101** 真机？请额外创建隔离硬件环境 `mamba env create -f environment.so101.yml`（或 `pip install -r requirements-so101.txt`），并按 [SO-101 快速上手](docs/guides/so101-quickstart.zh-CN.md) 与 [标定指南](docs/guides/so101-calibration.zh-CN.md) 完成首次标定。
> 可选安装独立 CLI（`ni-rua`、`ni-rua-queue`、`ni-rua-report`、`ni-rua-audit`）：执行 `mamba run -n ni-rua python scripts/build_package.py --output dist && mamba run -n ni-rua python -m pip install dist/*.whl`。

### 2. 运行

#### 方式 A — 接入你自己的 Agent（MCP 插件）

安装插件（仅需一次，Node >= 18）：

```bash
node agent-plugin/bin/cli.mjs install claude   # 或：codex | opencode
node agent-plugin/bin/cli.mjs doctor
```

以外部智能体模式启动网关（SO-101 单机纯 CPU 示例）：

```bash
mamba run -n ni-rua python -u -m ni_rua.apps.gateway --agent-backend external \
  --config configs/so101_standalone.json --port 8781 --run-dir runs/so101-demo \
  --prompt "观察桌面，将夹爪垂直抬高 2 cm，确认后返回原位。"
```

随后在另一个终端打开你的智能体，让它通过 `ni_rua` MCP 工具操作机器人（`run_begin` → `robot_observe` → `robot_action` → `run_finish`）。

> [!CAUTION]
> **停止、保持与急停（E-STOP）均在 Web 控制台中**（<http://127.0.0.1:8781/console?lang=zh-CN>）。在 Claude Code、Codex 或 OpenCode 中按 `Esc` **不会**让真机停下，操作期间请始终保持 12 V 物理电源开关触手可及。

#### 方式 B — 托管模式（由 NI-RUA 启动并监管智能体）

```bash
# 运行 SO-101 真机（Web 控制台：http://127.0.0.1:8781/console?lang=zh-CN）
mamba run -n ni-rua python -u -m ni_rua.apps.gateway \
  --config configs/so101_standalone.json --port 8781 --run-dir runs/so101-hosted \
  --prompt "观察工作区，抓起方块并放到右侧。"

# 运行 RoboDojo 仿真基准（需 Linux x86_64 与 NVIDIA GPU）
mamba run -n ni-rua python -u -m ni_rua.apps.gateway \
  --benchmark --task general_pickup --seed 0 --layout-id 0 \
  --gpu 0 --port 8767 --run-dir runs/demo-task
```

打开 <http://127.0.0.1:8767> 即可查看多相机画面、内嵌智能体终端、动作预览与完整归档。

---

## 文档导航

- **[完整文档索引](docs/README.zh-CN.md)** — 架构设计、协议契约与 3D 感知参考
- **[SO-101 快速上手](docs/guides/so101-quickstart.zh-CN.md)** 与 **[标定指南](docs/guides/so101-calibration.zh-CN.md)** — 真机装配、机台验收与 S0–S6 标定
- **[Agent 插件指南](docs/development/agent-plugin.zh-CN.md)** — 通过 MCP 连接 Claude Code、Codex 与 OpenCode
- **[运行指南](docs/development/running.zh-CN.md)** 与 **[配置说明](docs/development/configuration.zh-CN.md)** — 归档回放、批量评测队列、沙箱与 Provider 配置
- **[RoboDojo 环境配置](docs/development/robodojo-setup.zh-CN.md)** — 仿真依赖与补丁说明
- **[参与贡献](CONTRIBUTING.md)** 与 **[AGENTS.md](AGENTS.md)** — 开发准则与测试验证

---

## 许可证

版权所有 © 2026 **Nerve Intelligence**。本项目基于 [MIT 许可证](LICENSE) 发布，第三方依赖声明保留于 [`third_party/`](third_party)。
