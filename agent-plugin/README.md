# NI-RUA agent plugin (`@nerveint/ni-rua-plugin`)

[English](README.md) · [简体中文](README.zh-CN.md)

Use your own **Claude Code**, **Codex**, or **OpenCode** session as the actor of an NI-RUA robot run. The plugin connects the host to a local NI-RUA gateway over MCP and ships two skills:

- `ni-rua-robot`: how the agent opens (`run_begin`), runs, and closes (`run_finish`) a robot turn, how to stop the arm, and how to recover when the gateway is not reachable;
- `ni-rua-experiments`: how an agent working inside an NI-RUA checkout launches, resumes, replays, reports, and audits experiments with the queue, report, and audit tools.

For full architecture and lifecycle details, see the [Agent Plugin Guide](https://github.com/NerveInt/NI-RUA/blob/main/docs/development/agent-plugin.md). For real SO-101 arm bring-up and calibration, see the [SO-101 Quickstart](https://github.com/NerveInt/NI-RUA/blob/main/docs/guides/so101-quickstart.md) and [SO-101 Calibration Guide](https://github.com/NerveInt/NI-RUA/blob/main/docs/guides/so101-calibration.md).

## Quick Start

Requires **Node >= 18**, macOS or Linux (Windows only through WSL), and at least one supported host CLI (`claude`, `codex`, or `opencode` 2.x).

### One-line setup (`npx`)

```sh
# Install plugin + isolated Python 3.12 NI-RUA gateway (~/.local/share/ni-rua/envs/gateway)
npx -y @nerveint/ni-rua-plugin setup claude    # or: codex | opencode | all

# Add --so101 to also bootstrap the isolated SO-101 hardware worker env (~/.local/share/ni-rua/envs/so101)
npx -y @nerveint/ni-rua-plugin setup claude --so101

# Verify hosts, Python/IK stack, and gateway connectivity
npx -y @nerveint/ni-rua-plugin doctor          # add --so101 for USB serial / calibration checks
```

### Daily driver (`npm i -g`)

```sh
npm i -g @nerveint/ni-rua-plugin

# 1. Zero-hardware 30-second MCP connectivity check
ni-rua-plugin gateway --dry-run

# 2. Local CPU SO-101 arm (auto-adds --observation-profile rgb_proprio_v1 on cold start)
ni-rua-plugin gateway --config so101

# 3. Plugin-only install / uninstall / diagnostics
ni-rua-plugin install <claude|codex|opencode> [--scope user|project|local] [--port N]
ni-rua-plugin uninstall <claude|codex|opencode> [--scope user|project|local]
ni-rua-plugin doctor [--so101] [--json] [--gateway-url URL]
```

## How the installer works

The installer copies the package to `~/.local/share/ni-rua/agent-plugin/<version>/` and installs from that durable copy (so `npx` cache eviction never breaks host registrations). Re-running `install` is idempotent and also performs upgrades. Every host command and config edit is printed before execution and recorded in `installed.json`, so `uninstall` removes only what this tool added and preserves user-edited keys.

| Option | Meaning |
|---|---|
| `--so101` | `setup`: also create the isolated Python 3.12 SO-101 worker env (`lerobot`, `feetech-servo-sdk`, CPU IK). `doctor`: check SO-101 USB serial, `dialout` group, deployment JSON, and camera calibration. |
| `--skip-gateway` | `setup`: install only the host plugin(s), skip Python gateway bootstrap. |
| `--dry-run` | `gateway`: run a zero-hardware in-memory MCP server on `http://127.0.0.1:8790/mcp` (`run_begin`, `robot_observe`, `robot_action`, `run_finish`). |
| `--config <preset\|path>` | `gateway`: config preset (`so101`, `so101_standalone`, `ni-rua`) or JSON path passed to `ni-rua`. |
| `--port N` | `install` / `doctor`: MCP port (default `8790`). `gateway`: Web console port (or MCP port in `--dry-run`). |
| `--gateway-url URL` | `doctor`: probe a specific MCP endpoint URL. |
| `--json` | `doctor`: emit machine-readable JSON diagnostics. |
| `--scope user\|project\|local` | Claude Code only (`install`, `uninstall`, `setup`). Default `user`. |
| `--no-host-env` / `--no-allow` | Claude Code: skip writing the four MCP `env` keys or `permissions.allow`. |
| `--no-direct-catalog` | Codex: skip writing `model_catalog_json` into `$CODEX_HOME/config.toml`. |
| `--force` | Replace a same-version durable copy with differing content, or recreate managed Python envs in `setup`. |
| `--claude-bin`, `--codex-bin`, `--opencode-bin` | Override host executable discovery on `PATH`. |

## What each host gets

- **Claude Code**: registers `ni_rua@ni_rua` with `.mcp.json` (`type: http`, `alwaysLoad: true`), a `SessionStart` (`compact`) hook reminding the model to re-call `run_begin` after compaction, the four MCP `env` keys (`MAX_MCP_OUTPUT_TOKENS`, `MCP_TOOL_TIMEOUT`, `CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS`, `CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH`), and `permissions.allow += ["mcp__plugin_ni_rua_ni_rua"]`.
- **Codex**: installs `ni_rua@ni_rua` with `.codex-mcp.json` (`default_tools_approval_mode: "approve"`, `tool_timeout_sec: 3600`, per-tool `output_token_limit`) and sets `model_catalog_json = "<copy>/codex-models.direct.json"` in `$CODEX_HOME/config.toml` so MCP tools are delivered directly without Linux `code-mode host` crashes.
- **OpenCode (2.x)**: adds the durable copy path to `opencode.json` `plugins`, raises `tool_output` (`max_lines: 100000`, `max_bytes: 16000000`), sets lossless `media.image`, and registers the `compaction` hook. OpenCode 1.x is refused.

## Safety

Cancelling a tool call in the agent (`Esc`) does **not** stop a physical robot arm. Always use **Hold / E-STOP** in the NI-RUA Web console (`http://127.0.0.1:8781/console`) or keep a physical 12 V power cutoff within reach.

## Maintenance (from repo checkout)

`.codex-mcp.json` and `manifest.generated.json` are generated from the central tool registry—do not edit by hand:

```sh
mamba run -n ni-rua python scripts/gen_agent_plugin_codex_mcp.py --check
npm --prefix agent-plugin test
agent-plugin/test/tarball-check.sh
```
