<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/ni-rua-lockup-dark.svg" />
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/ni-rua-lockup-light.svg" />
  <img src="docs/assets/ni-rua-lockup-light.svg" alt="NI-RUA" width="260" />
</picture>

<p><b>An Agent Harness for Embodied Robot Manipulation</b></p>

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

## Quick Start (SO-101)

Requires **Node >= 18** and **Python >= 3.12** (or `uv`).

### 1. Install once

```bash
npx -y @nerveint/ni-rua-plugin setup codex   # also supports: claude | opencode | all
```

Installs the `ni_rua` MCP plugin plus isolated Python 3.12 gateway and SO-101 hardware runtimes under `~/.local/share/ni-rua/envs/` automatically (pass `--no-so101` to opt out of SO-101 hardware deps).

### 2. Bind your SO-101 hardware (first time only)

```bash
~/.local/share/ni-rua/envs/gateway/bin/ni-rua-so101 init-config   # or `ni-rua-so101 init-config` if on PATH
```

Detects your USB serial bus and UVC cameras into `~/.config/ni-rua/so101.local.json` (or just open your agent and let the bundled **`ni-rua-setup`** skill walk you through binding, diagnostics, and S0–S6 calibration at `http://127.0.0.1:8781/calibration?lang=en`).

### 3. Open Codex (or Claude Code / OpenCode) and drive the arm

```text
$ codex
> Use NI-RUA to inspect the table, lift the SO-101 gripper 2 cm, verify, and return.
```

The bundled **`ni-rua-setup`** + **`ni-rua-robot`** skills let the agent handle onboarding, calibration, and execution end-to-end, while the `ni_rua` stdio MCP bridge stays silent during normal coding (zero background gateway processes or MCP errors), auto-starts the singleton local SO-101 gateway on the first `run_begin` call, and surfaces the live Web console at `http://127.0.0.1:8781/console?lang=en`.

> [!CAUTION]
> **Stop, Hold, and E-STOP live in the Web console** (<http://127.0.0.1:8781/console?lang=en>). Pressing `Esc` in Codex, Claude Code, or OpenCode does **not** stop the physical arm. Always keep a physical 12 V power cutoff within reach.

---

## Documentation & Advanced Modes

- **Zero-hardware dry run:** Start `npx -y @nerveint/ni-rua-plugin gateway --dry-run` (or `npm i -g @nerveint/ni-rua-plugin && ni-rua-plugin gateway --dry-run`) to serve mock MCP tools on `http://127.0.0.1:8790/mcp`; the stdio bridge transparently proxies to any already-running manual gateway.
- **From-source, hosted benchmarks & simulation (RoboDojo / LIBERO):** Clone the repo (`mamba env create -f environment.yml && npm --prefix web ci && npm --prefix web run build`) or install `ni-rua-harness` from PyPI for hosted runs, parallel evaluation queues, dual-arm RoboDojo, and LIBERO simulation (`configs/libero.example.json`). See the [Running Guide](docs/development/running.md) and [RoboDojo Setup](docs/development/robodojo-setup.md).
- **[SO-101 Quickstart](docs/guides/so101-quickstart.md)** & **[Calibration Guide](docs/guides/so101-calibration.md)** — hardware checklist, commissioning (`ni-rua-so101`), S0–S6 calibration, and optional remote GPU perception/IK.
- **[Agent Plugin Guide](docs/development/agent-plugin.md)** — lazy stdio MCP bridge (`ni-rua-plugin mcp`), `NI_RUA_AUTO_START=0` / `NI_RUA_MCP_URL` overrides, and manual gateway workflows.
- **[Documentation Index](docs/README.md)** & **[Configuration](docs/development/configuration.md)** — architecture, tool contracts, 3D perception, replay, sandboxing, and provider profiles.
- **[Contributing](CONTRIBUTING.md)** & **[AGENTS.md](AGENTS.md)** — development guidelines and test suite.

---

## License

Copyright © 2026 **Nerve Intelligence**. Released under the [MIT License](LICENSE). Third-party notices are preserved in [`third_party/`](third_party).
