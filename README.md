<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/ni-rua-lockup-dark.svg" />
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/ni-rua-lockup-light.svg" />
  <img src="docs/assets/ni-rua-lockup-light.svg" alt="NI-RUA" width="260" />
</picture>

<p><b>An Agent Harness for Embodied Robot Manipulation</b></p>

[![npm](https://img.shields.io/npm/v/%40nerveint%2Fni-rua-plugin?label=npm&color=CB3837)](https://www.npmjs.com/package/@nerveint/ni-rua-plugin)
[![PyPI](https://img.shields.io/pypi/v/ni-rua-harness?label=PyPI&color=3775A9)](https://pypi.org/project/ni-rua-harness/)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.12-3776AB)](pyproject.toml)
[![Node](https://img.shields.io/badge/Node-%3E%3D18-339933)](agent-plugin/package.json)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[English](README.md) · [简体中文](README.zh-CN.md) · [Documentation](docs/README.md) · [License (MIT)](LICENSE)

</div>

---

## What is NI-RUA?

**NI-RUA** lets coding agents (**Codex**, **Claude Code**, or **OpenCode**) directly operate physical and simulated robots. The agent observes camera feeds and joint state, plans, and invokes motion tools; NI-RUA owns the physical execution boundary, enforces stage tolerances and safety guards in real time, and records complete, replayable run archives.

---

## What It Can Do

| Capability | Highlights |
| :--- | :--- |
| **Real & Simulated Robots** | Real **SO-101** desktop arm (macOS / Linux, pure CPU out of the box or optional remote GPU 3D perception & IK), dual-arm **ARX X5** in [RoboDojo](https://robodojo-benchmark.com/), and simulation-only **Franka Emika Panda** (`libero-franka-panda`) in [LIBERO](https://libero-project.github.io/). |
| **Zero-Friction Agent Plugin or Hosted** | One-command setup via `@nerveint/ni-rua-plugin` with a lazy-loading stdio MCP bridge (`ni-rua-plugin mcp`) for Codex / Claude Code / OpenCode, or **hosted mode** supervising the agent runtime, OS sandbox, budgets, and parallel evaluation queues. |
| **Safe Closed-Loop Execution** | Preview-before-execute motion batches, configurable stage tolerances, stall & adaptive empty-grasp guards, and typed physical receipts separated from display text. |
| **Operator Console & Archives** | Bilingual Web UI (`en` / `zh-CN`) with live cameras, guided S0–S6 calibration (`/calibration`), hardware **Stop / Hold / E-STOP** (`/console`), phase replay, and an experiment comparison library. |

---

## Quick Start

### Option A — Agent Plugin (SO-101, Zero-Friction)

Requires **Node >= 18** and **Python >= 3.12** (or `uv`).

**1. Install once**

```bash
npx -y @nerveint/ni-rua-plugin setup codex   # also supports: claude | opencode | all
```

Installs the `ni_rua` MCP plugin, bundled skills (`ni-rua-setup`, `ni-rua-robot`), and isolated Python 3.12 gateway & SO-101 hardware runtimes under `~/.local/share/ni-rua/envs/` (pass `--no-so101` to opt out of SO-101 hardware deps).

**2. Open Codex (or Claude Code / OpenCode) and talk to it**

**First-time setup & calibration** — let `ni-rua-setup` guide environment checks, hardware binding, and S0–S6 calibration at `http://127.0.0.1:8781/calibration?lang=en` *(or bind manually via `~/.local/share/ni-rua/envs/gateway/bin/ni-rua-so101 init-config`)*:

```text
$ codex
> Use the ni-rua-setup skill to set up my SO-101 arm from scratch: check the environment, bind the USB serial port and cameras, guide me through S0-S6 calibration, and verify everything is ready.
```

**Operate the arm** — drive SO-101 in one window via `ni-rua-robot` and the lazy stdio MCP bridge (silent during normal coding; auto-starts the singleton SO-101 gateway on first `run_begin` and opens `http://127.0.0.1:8781/console?lang=en`):

```text
$ codex
> Use the ni-rua-robot skill to inspect the table, lift the SO-101 gripper 2 cm, verify, and return.
```

> [!CAUTION]
> **Stop, Hold, and E-STOP live in the Web console** (<http://127.0.0.1:8781/console?lang=en>). Pressing `Esc` in Codex, Claude Code, or OpenCode does **not** stop the physical arm. Always keep a physical 12 V power cutoff within reach.

### Option B — Full Hosted Harness (Download, Deploy & Launch)

**Install** (PyPI for quick hosted CLI, or from source for the full Web console & simulation benchmarks):

```bash
# Quick PyPI install (`ni-rua`, `ni-rua-so101`, `ni-rua-queue`, `ni-rua-report`, `ni-rua-audit`)
pip install ni-rua-harness

# Or full from-source deployment (Python 3.12 + Web console on Node 22)
git clone https://github.com/NerveInt/NI-RUA.git && cd NI-RUA
mamba env create -f environment.yml
mamba env create -f environment.so101.yml          # optional: real SO-101 hardware worker
npm --prefix web ci && npm --prefix web run build
```

**Launch** (hosted SO-101 real arm or RoboDojo / LIBERO simulation benchmarks):

```bash
# Hosted SO-101 real arm (Web console: http://127.0.0.1:8781/console)
ni-rua --config so101 --port 8781 --run-dir runs/so101-hosted \
  --prompt "Inspect the workspace, pick up the cube, and place it on the right."
# From source: mamba run -n ni-rua python -u -m ni_rua.apps.gateway --config configs/so101_standalone.json ...

# Hosted RoboDojo simulation benchmark (Linux x86_64 + NVIDIA GPU; Web workbench: http://127.0.0.1:8767)
mamba run -n ni-rua python -u -m ni_rua.apps.gateway \
  --benchmark --task general_pickup --seed 0 --layout-id 0 \
  --gpu 0 --port 8767 --run-dir runs/demo-task
```

For LIBERO simulation (`libero-franka-panda`), pass `--config configs/libero.example.json`; for multi-GPU parallel benchmark queues, use `ni-rua-queue` (see [Running Guide](docs/development/running.md) & [RoboDojo Setup](docs/development/robodojo-setup.md)).

---

## Documentation

- **Zero-hardware dry run:** `npx -y @nerveint/ni-rua-plugin gateway --dry-run` serves mock MCP tools on `http://127.0.0.1:8790/mcp` (auto-proxied by the stdio bridge).
- **[SO-101 Quickstart](docs/guides/so101-quickstart.md)** & **[Calibration Guide](docs/guides/so101-calibration.md)** — hardware checklist, commissioning (`ni-rua-so101`), S0–S6 calibration, and optional remote GPU perception/IK.
- **[Agent Plugin Guide](docs/development/agent-plugin.md)** — lazy stdio MCP bridge (`ni-rua-plugin mcp`), `NI_RUA_AUTO_START=0` / `NI_RUA_MCP_URL` overrides, and manual gateway workflows.
- **[Running Guide](docs/development/running.md)** & **[RoboDojo Setup](docs/development/robodojo-setup.md)** — hosted workflows, replay, parallel queues (`ni-rua-queue`), and simulation setup.
- **[Documentation Index](docs/README.md)** & **[Configuration](docs/development/configuration.md)** — architecture, tool contracts, 3D perception, sandboxing, and provider profiles.
- **[Contributing](CONTRIBUTING.md)** & **[AGENTS.md](AGENTS.md)** — development guidelines and test suite.

---

## License

Copyright © 2026 **Nerve Intelligence**. Released under the [MIT License](LICENSE). Third-party notices are preserved in [`third_party/`](third_party).
