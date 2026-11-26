# NI-RUA agent plugin (`@nerveint/ni-rua-plugin`)

[English](README.md) · [简体中文](README.zh-CN.md)

Use your own **Claude Code**, **Codex**, or **OpenCode** session as the actor of an NI-RUA robot run. The plugin connects the host through a zero-friction **stdio MCP lazy bridge** (`ni-rua-plugin mcp`) that proxies any already-running local NI-RUA gateway or automatically starts your local SO-101 gateway on first `run_begin`—so you can control your SO-101 arm in a single terminal window. It also ships three skills:

- `ni-rua-setup`: end-to-end onboarding, `doctor` diagnostics, `ni-rua-so101` hardware binding & commissioning, guided S0–S6 `/calibration`, and troubleshooting before handing off to `ni-rua-robot`;
- `ni-rua-robot`: how the agent opens (`run_begin`), runs, and closes (`run_finish`) a robot turn, how to stop the arm safely, and how to recover from errors;
- `ni-rua-experiments`: how an agent working inside an NI-RUA checkout launches, resumes, replays, reports, and audits experiments with the queue, report, and audit tools.

For full architecture and lifecycle details, see the [Agent Plugin Guide](https://github.com/NerveInt/NI-RUA/blob/main/docs/development/agent-plugin.md). For real SO-101 arm bring-up and calibration, see the [SO-101 Quickstart](https://github.com/NerveInt/NI-RUA/blob/main/docs/guides/so101-quickstart.md) and [SO-101 Calibration Guide](https://github.com/NerveInt/NI-RUA/blob/main/docs/guides/so101-calibration.md).

## Quick Start (1-Window SO-101 Flow)

Requires **Node >= 18**, macOS or Linux (Windows only through WSL), and at least one supported host CLI (`claude`, `codex`, or `opencode` 2.x).

### 1. Install once

```sh
npx -y @nerveint/ni-rua-plugin setup codex    # or: claude | opencode | all
```

Installs the host plugin, bundled skills, and isolated Python 3.12 gateway & SO-101 hardware envs (skip SO-101 hardware deps with `--no-so101`, or verify anytime with `npx -y @nerveint/ni-rua-plugin doctor`).

### 2. Open Codex (or Claude Code / OpenCode) and talk to it

**First-time setup & calibration** (`ni-rua-setup` skill):

```text
$ codex
> Use the ni-rua-setup skill to set up my SO-101 arm from scratch: check the environment, bind the USB serial port and cameras, guide me through S0-S6 calibration, and verify everything is ready.
```

*(Prefer manual hardware binding? Run `~/.local/share/ni-rua/envs/gateway/bin/ni-rua-so101 init-config` first.)*

**Operate the arm** (`ni-rua-robot` skill — on first `run_begin`, the lazy stdio MCP bridge auto-starts the singleton SO-101 gateway and prints the live Web console URL `http://127.0.0.1:8781/console`):

```text
$ codex
> Use the ni-rua-robot skill to inspect the table, lift the SO-101 gripper 2 cm, verify, and return.
```

### Manual gateway & daily CLI (`npm i -g`)

If you manually start a RoboDojo, LIBERO, SO-101, or `--dry-run` HTTP gateway on `http://127.0.0.1:8790/mcp` (or `NI_RUA_MCP_URL`), the stdio bridge detects it automatically and proxies to it transparently:

```sh
npm i -g @nerveint/ni-rua-plugin

# Zero-hardware 30-second MCP connectivity check
ni-rua-plugin gateway --dry-run

# Manual local CPU SO-101 gateway (auto-adds --observation-profile rgb_proprio_v1 on cold start)
ni-rua-plugin gateway --config so101

# Plugin-only install / uninstall / diagnostics
ni-rua-plugin install <claude|codex|opencode> [--scope user|project|local] [--port N]
ni-rua-plugin uninstall <claude|codex|opencode> [--scope user|project|local]
ni-rua-plugin doctor [--so101] [--json] [--gateway-url URL]
```

## How the installer and lazy bridge work

- **Durable install**: The installer copies the package to `~/.local/share/ni-rua/agent-plugin/<version>/` and wires each host to `node <durable>/bin/cli.mjs mcp` (so `npx` cache eviction never breaks host registrations and no global `npm i -g` is required). Every host command and config edit is recorded in `installed.json` for exact `uninstall`.
- **Zero-overhead startup**: `initialize` returns immediately without spawning Python or touching hardware; `tools/list` probes the upstream HTTP gateway (~300 ms) and falls back to bundled static schemas when offline.
- **Lazy singleton auto-start**: On the first `tools/call` when no gateway is running (unless `NI_RUA_AUTO_START=0`), the bridge acquires `~/.local/share/ni-rua/gateway.lock`, spawns the detached local gateway (`NI_RUA_GATEWAY_CONFIG`, default `so101`, logs in `~/.local/share/ni-rua/logs/gateway.log`), notifies the host via `notifications/tools/list_changed`, and prepends the live `/console` safety banner to `run_begin`.

| Option / Env | Meaning |
|---|---|
| `--no-so101` | `setup`: skip creating the isolated Python 3.12 SO-101 hardware env (installed by default; `--so101` is kept as an alias). |
| `--skip-gateway` | `setup`: install only the host plugin(s), skip Python runtime bootstrap. |
| `--so101` | `doctor`: check SO-101 USB serial, `dialout` group, deployment JSON, and camera calibration. |
| `--dry-run` | `gateway`: run a zero-hardware in-memory MCP server on `http://127.0.0.1:8790/mcp`. |
| `--config <preset\|path>` | `gateway`: config preset (`so101`, `so101_standalone`, `ni-rua`) or JSON path passed to `ni-rua`. |
| `--port N` | `install` / `doctor` / `mcp`: MCP port (default `8790`). `gateway`: Web console port (or MCP port in `--dry-run`). |
| `--gateway-url URL` | `doctor` / `mcp`: target a specific upstream MCP HTTP URL. |
| `NI_RUA_MCP_URL` | Override upstream gateway MCP URL (default `http://127.0.0.1:8790/mcp` or installed `--port`). |
| `NI_RUA_AUTO_START=0` | Disable lazy auto-start in the stdio bridge (proxy-only mode). |
| `NI_RUA_GATEWAY_CONFIG` | Config preset/path used by lazy auto-start (default `so101`). |
| `--json` | `doctor`: emit machine-readable JSON diagnostics. |
| `--scope user\|project\|local` | Claude Code only (`install`, `uninstall`, `setup`). Default `user`. |
| `--no-host-env` / `--no-allow` | Claude Code: skip writing the four MCP `env` keys or `permissions.allow`. |
| `--no-direct-catalog` | Codex: skip writing `model_catalog_json` into `$CODEX_HOME/config.toml`. |
| `--force` | Replace a same-version durable copy with differing content, or recreate managed Python envs in `setup`. |
| `--claude-bin`, `--codex-bin`, `--opencode-bin` | Override host executable discovery on `PATH`. |

## What each host gets

- **Claude Code**: registers `ni_rua@ni_rua` with `.mcp.json` (`command: "node"`, `args: ["${CLAUDE_PLUGIN_ROOT}/bin/cli.mjs", "mcp"]`, `alwaysLoad: true`), a `SessionStart` (`compact`) hook reminding the model to re-call `run_begin` after compaction, the four MCP `env` keys (`MAX_MCP_OUTPUT_TOKENS`, `MCP_TOOL_TIMEOUT`, `CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS`, `CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH`), and `permissions.allow += ["mcp__plugin_ni_rua_ni_rua"]`.
- **Codex**: installs `ni_rua@ni_rua` with `.codex-mcp.json` pointing to `node <durable>/bin/cli.mjs mcp` (`default_tools_approval_mode: "approve"`, `tool_timeout_sec: 3600`, per-tool `output_token_limit`) and sets `model_catalog_json = "<copy>/codex-models.direct.json"` in `$CODEX_HOME/config.toml` so MCP tools are delivered directly without Linux `code-mode host` crashes.
- **OpenCode (2.x)**: adds the durable copy path to `opencode.json` `plugins`, registers the local stdio bridge (`type: "local"`, `command: ["node", "<durable>/bin/cli.mjs", "mcp"]`), raises `tool_output` (`max_lines: 100000`, `max_bytes: 16000000`), sets lossless `media.image`, and registers the `compaction` hook. OpenCode 1.x is refused.

## Safety

Cancelling a tool call in the agent (`Esc`) does **not** stop a physical robot arm. Always use **Hold / E-STOP** in the NI-RUA Web console (`http://127.0.0.1:8781/console`) or keep a physical 12 V power cutoff within reach.

## Maintenance (from repo checkout)

`.mcp.json`, `.codex-mcp.json`, and `manifest.generated.json` are generated from the central tool registry—do not edit by hand:

```sh
mamba run -n ni-rua python scripts/gen_agent_plugin_codex_mcp.py --check
npm --prefix agent-plugin test
agent-plugin/test/tarball-check.sh
```
