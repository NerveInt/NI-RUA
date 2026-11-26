# Changelog

Version numbers correspond to `pyproject.toml` and matching annotated Git tags. This file tracks software releases—task performance claims always require independent benchmark runs under frozen source, configuration, and model conditions.

---

## [0.1.1]

Zero-friction SO-101 onboarding and lazy-loading stdio MCP bridge for external coding agents.

- **Lazy-Loading Stdio MCP Bridge (`ni-rua-plugin mcp`)**: Installed directly into Codex, Claude Code, and OpenCode from the durable plugin copy so normal coding sessions incur zero background gateway processes and zero MCP connection errors. On the first `run_begin` call, the bridge automatically starts the singleton local SO-101 gateway (including automatic `--observation-profile rgb_proprio_v1` cold-start calibration mode when uncalibrated) and prints the live Web `/console` E-STOP URL (`http://127.0.0.1:8781/console`).
- **Transparent Manual Gateway Proxying & Overrides**: When a manual HTTP gateway is already running on `NI_RUA_MCP_URL` (default `http://127.0.0.1:8790/mcp`, e.g. `ni-rua-plugin gateway --dry-run`, RoboDojo, or LIBERO), `ni-rua-plugin mcp` transparently proxies JSON-RPC calls (preserving `Mcp-Session-Id` and SSE responses). Supports `NI_RUA_AUTO_START=0` (proxy-only) and `NI_RUA_GATEWAY_CONFIG` overrides.
- **SO-101 Runtime Included by Default in `setup`**: `npx -y @nerveint/ni-rua-plugin setup <codex|claude|opencode|all>` now bootstraps both the isolated Python 3.12 gateway (`~/.local/share/ni-rua/envs/gateway`) and isolated SO-101 hardware runtime (`~/.local/share/ni-rua/envs/so101`) by default (`--no-so101` opts out; `--so101` remains accepted as a compatible alias).
- **Bundled `ni-rua-setup` Agent Skill**: Added `ni-rua-setup` alongside `ni-rua-robot` and `ni-rua-experiments` across Codex, Claude Code, and OpenCode so agents can guide environment checks (`ni-rua-plugin doctor`), SO-101 hardware binding (`ni-rua-so101 init-config` / `cameras` / `commission`), S0–S6 `/calibration`, dry-run verification, and handoff to `ni-rua-robot` end-to-end.
- **SO-101-First Quick Start & Documentation**: Streamlined `README.md`, `README.zh-CN.md`, the Agent Plugin guides, and the SO-101 Quickstart guides around the single-window Codex / Claude Code / OpenCode workflow.

---

## 0.1.0

Initial release of **NI-RUA** by **Nerve Intelligence** (`ni-rua-harness`, Python package `ni_rua`, CLI commands `ni-rua`, `ni-rua-queue`, `ni-rua-report`, `ni-rua-audit`).

- **Unified Embodied Agent Harness**: Six-layer Python architecture (`core/`, `robot/`, `perception/`, `cortex/`, `runtime/`, `apps/`) with typed tool contracts (`ToolSpec` / `ToolRegistry`), single physical execution boundary (`Harness.execute`), and strict actor information boundaries.
- **Multi-Robot & Multi-Environment Support**: Real desktop **SO-101** arm (standalone CPU or optional remote GPU 3D perception & cuRobo IK), dual-arm **ARX X5** in RoboDojo simulation, and simulation-only single-arm **Franka Emika Panda** in LIBERO.
- **Swappable Agent Backends & MCP Plugin**: Supports hosted **Codex**, **Claude Code**, and **OpenCode** runtimes with OS sandboxing, provider profiles, and model handoff, plus external-agent mode via the `@nerveint/ni-rua-plugin` (`ni-rua-plugin`) MCP integration.
- **3D Perception & Control Guards**: Inline `obs3d` grounding, `object_table`, `scene3d` point-cloud handles, VGGT support-plane height estimation, preview-before-execute multi-stage motion batches, configurable stage tolerances, stall guards, and adaptive grip guards.
- **Bilingual Web Workbench & Experiment Library**: `zh-CN` / `en` operator console (`/console`) with hardware stop/hold/E-STOP controls, guided S0–S6 calibration wizard (`/calibration`), phase-synchronized trajectory replay, HD offline replay, and a permanent experiment comparison library.
- **Evaluation Queue & Integrity Auditing**: Parallel GPU evaluation queue (`ni-rua-queue`) with OOM recovery and completion hooks, typed reporting (`ni-rua-report`), and archive integrity auditing (`ni-rua-audit`).
