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

**NI-RUA** lets coding agents (**Claude Code**, **Codex**, or **OpenCode**) directly operate physical and simulated robots. The agent observes camera feeds and joint state, plans, and invokes motion tools; NI-RUA owns the physical execution boundary, enforces stage tolerances and safety guards in real time, and records complete, replayable run archives.

---

## What It Can Do

| Capability | Highlights |
| :--- | :--- |
| **Real & Simulated Robots** | Real **SO-101** desktop arm (macOS / Linux, pure CPU or optional remote GPU 3D perception & IK), dual-arm **ARX X5** in [RoboDojo](https://robodojo-benchmark.com/), and simulation-only **Franka Emika Panda** (`libero-franka-panda`) in [LIBERO](https://libero-project.github.io/). |
| **Bring Your Own Agent or Hosted** | One-line MCP setup for Claude Code / Codex / OpenCode via `@nerveint/ni-rua-plugin` (`ni-rua-plugin`), or **hosted mode** where NI-RUA supervises the agent runtime, OS sandbox, budgets, and batch evaluation queues. |
| **Safe Closed-Loop Execution** | Preview-before-execute motion batches, configurable stage tolerances, stall & empty-grasp guards, and typed physical receipts separated from display text. |
| **Operator Console & Archives** | Bilingual Web UI (`zh-CN` / `en`) for live cameras, guided S0–S6 calibration (`/calibration`), hardware **Stop / Hold / E-STOP** (`/console`), phase replay, and an experiment comparison library. |

---

## Quick Start

### Option A — One-Line Setup (`npx` / `npm i -g`, No Repo Clone Needed)

Requires **Node >= 18** and **Python >= 3.12** (or `uv`, auto-bootstrapped into isolated envs under `~/.local/share/ni-rua/envs/`).

```bash
# One-shot setup: installs the MCP plugin + isolated Python 3.12 gateway runtime
npx -y @nerveint/ni-rua-plugin setup claude        # or: codex | opencode | all
# Add --so101 to also bootstrap the isolated SO-101 hardware worker environment:
npx -y @nerveint/ni-rua-plugin setup claude --so101

# Recommended for daily use (installs the `ni-rua-plugin` CLI globally)
npm i -g @nerveint/ni-rua-plugin
ni-rua-plugin doctor                               # add --so101 for hardware/serial checks
```

Verify immediately with **zero hardware** or launch a real **SO-101** desktop arm:

```bash
# 1. Zero-hardware MCP smoke test (serves dry-run tools on http://127.0.0.1:8790/mcp)
ni-rua-plugin gateway --dry-run

# 2. Real SO-101 arm (detect serial/cameras once, then launch pure-CPU gateway)
ni-rua-so101 init-config                           # writes ~/.config/ni-rua/so101.local.json
ni-rua-plugin gateway --config so101 --port 8781 --run-dir runs/so101-demo \
  --prompt "Inspect the table, lift the gripper 2 cm, verify, and return."
```

Then open your agent (`claude`, `codex`, or `opencode`) in another terminal and ask it to operate the robot via the `ni_rua` MCP tools (`run_begin` → `robot_observe` → `robot_action` → `run_finish`).

> [!CAUTION]
> **Stop, Hold, and E-STOP live in the Web console** (<http://127.0.0.1:8781/console?lang=en>). Pressing `Esc` in Claude Code, Codex, or OpenCode does **not** stop the physical arm. Always keep a physical 12 V power cutoff within reach.

### Option B — From Source & Hosted Benchmarks

```bash
git clone https://github.com/NerveInt/NI-RUA.git
cd NI-RUA

# Core harness environment (Python 3.12) & Web operator console (Node 22)
mamba env create -f environment.yml
npm --prefix web ci && npm --prefix web run build

# Hosted SO-101 real arm (Web console: http://127.0.0.1:8781/console?lang=en)
mamba run -n ni-rua python -u -m ni_rua.apps.gateway \
  --config configs/so101_standalone.json --port 8781 --run-dir runs/so101-hosted \
  --prompt "Inspect the workspace, pick up the cube, and place it on the right."

# Hosted RoboDojo simulation benchmark (requires Linux x86_64 + NVIDIA GPU)
mamba run -n ni-rua python -u -m ni_rua.apps.gateway \
  --benchmark --task general_pickup --seed 0 --layout-id 0 \
  --gpu 0 --port 8767 --run-dir runs/demo-task
```

Open <http://127.0.0.1:8767> for live cameras, the embedded agent terminal, action previews, and run archives. Standalone Python CLIs (`ni-rua`, `ni-rua-so101`, `ni-rua-queue`, `ni-rua-report`, `ni-rua-audit`) are also published as `ni-rua-harness` on PyPI.

---

## Documentation

- **[Documentation Index](docs/README.md)** — full architecture, protocol, and perception reference
- **[SO-101 Quickstart](docs/guides/so101-quickstart.md)** & **[Calibration](docs/guides/so101-calibration.md)** — real-robot setup, commissioning (`ni-rua-so101`), and S0–S6 calibration
- **[Agent Plugin Guide](docs/development/agent-plugin.md)** — connecting Claude Code, Codex, and OpenCode over MCP (`@nerveint/ni-rua-plugin`)
- **[Running Guide](docs/development/running.md)** & **[Configuration](docs/development/configuration.md)** — replay, evaluation queues, sandboxing, and provider profiles
- **[RoboDojo Setup](docs/development/robodojo-setup.md)** — simulation environment and patches
- **[Contributing](CONTRIBUTING.md)** & **[AGENTS.md](AGENTS.md)** — contributor guidelines and test suite

---

## License

Copyright © 2026 **Nerve Intelligence**. Released under the [MIT License](LICENSE). Third-party notices are preserved in [`third_party/`](third_party).
