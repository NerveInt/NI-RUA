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

**NI-RUA** 让编程智能体（**Codex**、**Claude Code** 或 **OpenCode**）直接操作真机与仿真机器人。智能体查看相机画面与关节状态、制定计划并调用动作工具；NI-RUA 负责守住物理执行边界，实时校验多阶段容差与安全护栏，并保存完整可回放的运行归档。

---

## 它能做什么？

| 核心能力 | 说明 |
| :--- | :--- |
| **真机与仿真统一驱动** | 支持 **SO-101** 桌面真机机械臂（macOS / Linux，纯 CPU 开箱即用，也可按需对接远程 GPU 3D 感知与 IK）、[RoboDojo](https://robodojo-benchmark.com/) 仿真双臂 **ARX X5**，以及 [LIBERO](https://libero-project.github.io/) 仅仿真（simulation-only）单臂 **Franka Emika Panda**（`libero-franka-panda`）。 |
| **零摩擦插件或完整托管** | 通过 `@nerveint/ni-rua-plugin` 一键安装面向 Codex / Claude Code / OpenCode 的按需启动 stdio MCP 桥接（`ni-rua-plugin mcp`），或以**托管模式**统一监管智能体、OS 沙箱、预算与批量评测队列。 |
| **安全闭环与透明护栏** | 支持预览-执行分离、可配置阶段容差、防碰撞与自适应防空抓护栏，以及事实与展示严格分离的结构化物理回执。 |
| **双语控制台与实验库** | 内置中英双语 Web 界面，提供实时相机、S0–S6 引导标定（`/calibration`）、硬件级 **停止 / 保持 / 急停 E-STOP**（`/console`）、阶段回放与跨方案对比实验库。 |

---

## 快速上手（默认 SO-101 真机）

需要 **Node >= 18** 与 **Python >= 3.12**（或 `uv`）。

### 1. 一次性安装

```bash
npx -y @nerveint/ni-rua-plugin setup codex   # 也支持：claude | opencode | all
```

自动安装 `ni_rua` MCP 插件，并在 `~/.local/share/ni-rua/envs/` 下创建隔离的 Python 3.12 网关与 SO-101 真机硬件运行时（如需跳过 SO-101 硬件依赖，可加 `--no-so101`）。

### 2. 绑定 SO-101 硬件（仅首次）

```bash
~/.local/share/ni-rua/envs/gateway/bin/ni-rua-so101 init-config   # 若已在 PATH 中可直接运行 `ni-rua-so101 init-config`
```

自动探测 USB 串口与 UVC 相机并写入 `~/.config/ni-rua/so101.local.json`（也可直接打开智能体，由内置 **`ni-rua-setup`** 技能全程引导硬件绑定、故障排查与 `http://127.0.0.1:8781/calibration?lang=zh-CN` 的 S0–S6 标定）。

### 3. 直接打开 Codex（或 Claude Code / OpenCode）操控机械臂

```text
$ codex
> Use NI-RUA to inspect the table, lift the SO-101 gripper 2 cm, verify, and return.
```

内置 **`ni-rua-setup`** 与 **`ni-rua-robot`** 技能让智能体端到端包办安装配置、标定与操作执行；`ni_rua` stdio MCP 桥接在日常写代码时零后台进程、零 MCP 报错，首次调用 `run_begin` 时按需拉起本地单例 SO-101 网关，并在 `http://127.0.0.1:8781/console?lang=zh-CN` 提供实时 Web 控制台。

> [!CAUTION]
> **停止、保持与急停（E-STOP）均在 Web 控制台中**（<http://127.0.0.1:8781/console?lang=zh-CN>）。在 Codex、Claude Code 或 OpenCode 中按 `Esc` **不会**让真机停下，操作期间请始终保持 12 V 物理电源开关触手可及。

---

## 文档与进阶模式

- **零硬件试运行（Dry Run）：** 运行 `npx -y @nerveint/ni-rua-plugin gateway --dry-run`（或 `npm i -g @nerveint/ni-rua-plugin && ni-rua-plugin gateway --dry-run`）在 `http://127.0.0.1:8790/mcp` 启动模拟 MCP 服务，stdio 桥接会自动透明代理到已运行的手动网关。
- **源码安装、托管评测与仿真（RoboDojo / LIBERO）：** 克隆仓库（`mamba env create -f environment.yml && npm --prefix web ci && npm --prefix web run build`）或从 PyPI 安装 `ni-rua-harness`，即可运行托管模式、并发评测队列、RoboDojo 双臂仿真与 LIBERO 仿真（`configs/libero.example.json`），详见 [运行指南](docs/development/running.zh-CN.md) 与 [RoboDojo 环境配置](docs/development/robodojo-setup.zh-CN.md)。
- **[SO-101 快速上手](docs/guides/so101-quickstart.zh-CN.md)** 与 **[标定指南](docs/guides/so101-calibration.zh-CN.md)** — 真机装配检查、机台验收（`ni-rua-so101`）、S0–S6 标定及可选远程 GPU 感知/IK。
- **[Agent 插件指南](docs/development/agent-plugin.zh-CN.md)** — 按需 stdio MCP 桥接（`ni-rua-plugin mcp`）、`NI_RUA_AUTO_START=0` / `NI_RUA_MCP_URL` 环境变量与手动网关用法。
- **[完整文档索引](docs/README.zh-CN.md)** 与 **[配置说明](docs/development/configuration.zh-CN.md)** — 架构设计、工具契约、3D 感知、归档回放、沙箱与 Provider 配置。
- **[参与贡献](CONTRIBUTING.md)** 与 **[AGENTS.md](AGENTS.md)** — 开发准则与测试验证。

---

## 许可证

版权所有 © 2026 **Nerve Intelligence**。本项目基于 [MIT 许可证](LICENSE) 发布，第三方依赖声明保留于 [`third_party/`](third_party)。
