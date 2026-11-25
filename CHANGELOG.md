# Changelog

Version numbers correspond to `pyproject.toml` and matching annotated Git tags. This file tracks software releases—task performance claims always require independent benchmark runs under frozen source, configuration, and model conditions.

---

## 0.1.0

Initial release of **NI-RUA** by **Nerve Intelligence** (`ni-rua-harness`, Python package `ni_rua`, CLI commands `ni-rua`, `ni-rua-queue`, `ni-rua-report`, `ni-rua-audit`).

- **Unified Embodied Agent Harness**: Six-layer Python architecture (`core/`, `robot/`, `perception/`, `cortex/`, `runtime/`, `apps/`) with typed tool contracts (`ToolSpec` / `ToolRegistry`), single physical execution boundary (`Harness.execute`), and strict actor information boundaries.
- **Multi-Robot & Multi-Environment Support**: Real desktop **SO-101** arm (standalone CPU or optional remote GPU 3D perception & cuRobo IK), dual-arm **ARX X5** in RoboDojo simulation, and simulation-only single-arm **Franka Emika Panda** in LIBERO.
- **Swappable Agent Backends & MCP Plugin**: Supports hosted **Codex**, **Claude Code**, and **OpenCode** runtimes with OS sandboxing, provider profiles, and model handoff, plus external-agent mode via the `ni-rua-agent-plugin` (`ni-rua-plugin`) MCP integration.
- **3D Perception & Control Guards**: Inline `obs3d` grounding, `object_table`, `scene3d` point-cloud handles, VGGT support-plane height estimation, preview-before-execute multi-stage motion batches, configurable stage tolerances, stall guards, and adaptive grip guards.
- **Bilingual Web Workbench & Experiment Library**: `zh-CN` / `en` operator console (`/console`) with hardware stop/hold/E-STOP controls, guided S0–S6 calibration wizard (`/calibration`), phase-synchronized trajectory replay, HD offline replay, and a permanent experiment comparison library.
- **Evaluation Queue & Integrity Auditing**: Parallel GPU evaluation queue (`ni-rua-queue`) with OOM recovery and completion hooks, typed reporting (`ni-rua-report`), and archive integrity auditing (`ni-rua-audit`).
