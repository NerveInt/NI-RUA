# NI-RUA agent plugin

[English](README.md) · [简体中文](README.zh-CN.md)

Use your own Claude Code, Codex or OpenCode session as the actor of an NI-RUA robot run. The plugin gives the agent
the NI-RUA robot tools over MCP (served by a local NI-RUA gateway) and two skills:

- `ni-rua-robot`: how the agent starts (`run_begin`), works and finishes (`run_finish`) a robot run, how to stop the
  arm, and how to recover when the gateway is not reachable;
- `ni-rua-experiments`: how an agent working in an NI-RUA checkout launches, resumes, replays, reports and audits
  experiments with the existing queue, report and audit tools.

This package contains no Python and no robot software. The NI-RUA harness, the robot environment and the provider
profiles are a separate install (clone the repository, create the `ni-rua` environment from `environment.yml`). For
full details (coexistence with hosted mode, `run_begin`/`run_finish` semantics, console behavior, recovery, and
verification status), see the [Agent Plugin Guide](../docs/development/agent-plugin.md). If you are setting up a real
SO-101 arm for the first time, start with the [SO-101 Quickstart](../docs/guides/so101-quickstart.md) and
[SO-101 Calibration Guide](../docs/guides/so101-calibration.md).

## Install

Requires Node 18 or newer, macOS or Linux (Windows only through WSL), and the host CLI you want to use.

```sh
npx --package ./ni-rua-agent-plugin-<version>.tgz ni-rua-plugin install claude   # or: node bin/cli.mjs install claude
node bin/cli.mjs install codex
node bin/cli.mjs install opencode
node bin/cli.mjs doctor
node bin/cli.mjs uninstall <claude|codex|opencode> [--scope user|project|local]
```

The package is not published to npm (`package.json` is `"private": true` until the maintainer decides to publish); install from the repository (`npm pack` in `agent-plugin/`, or run
`node agent-plugin/bin/cli.mjs` (the binary name is `ni-rua-plugin`)). The installer first copies the package to
`~/.local/share/ni-rua/agent-plugin/<version>/` and installs from there. Re-running `install` is idempotent and is also
the upgrade path. A version directory is immutable: different content under the same version is refused unless
`--force`. Every host command and settings change is printed before it runs and recorded in `installed.json` next to
the copy, so `uninstall` removes exactly what was added and leaves values you set yourself.

| Option | Meaning |
|---|---|
| `--port N` | MCP port written into the installed copy's three entries (default 8790, the gateway's `agent.external.mcp_port` / `--external-mcp-port`). |
| `--scope user\|project\|local` | Claude Code only (applies to `install` and `uninstall`). Default `user`. `project` and `local` act on the current directory (workspace must be trusted by Claude Code). |
| `--no-host-env` | Claude Code: do not write the four `env` keys below. |
| `--no-allow` | Claude Code: do not add the `permissions.allow` entry. |
| `--no-direct-catalog` | Codex: do not write `model_catalog_json` into `$CODEX_HOME/config.toml`. |
| `--force` | Replace a durable copy of the same version that has different content. |
| `--claude-bin`, `--codex-bin`, `--opencode-bin` | Host executable to use instead of the first one on `PATH`. |

What each host gets:

- **Claude Code**: `claude plugin marketplace add <copy>` and `claude plugin install ni_rua@ni_rua -s <scope>`. The
  plugin's `.mcp.json` entry is `type: http`, `alwaysLoad: true`, and `hooks/hooks.json` registers a
  `SessionStart` (`compact`) hook that reminds the agent to re-call `run_begin` when compaction happens during an
  active NI-RUA turn. In that scope's `settings.json` the installer adds
  `env` keys `MAX_MCP_OUTPUT_TOKENS`, `MCP_TOOL_TIMEOUT`, `CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS`,
  `CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH` (keys you already set are kept) and
  `permissions.allow += ["mcp__plugin_ni_rua_ni_rua"]`, without which the first robot call is refused. With
  `--scope project` or `--scope local`, Claude Code ignores project/local `permissions.allow` and `env` until the
  workspace (or its enclosing `.git` root when the directory has no `.git` of its own) has been trusted in interactive
  `claude`; both `install` and `doctor` check and warn if untrusted.
- **Codex**: `codex plugin marketplace add <copy>` and `codex plugin add ni_rua@ni_rua` (an upgrade removes and adds
  again, because Codex copies the plugin into its cache). `.codex-mcp.json` carries `default_tools_approval_mode:
  "approve"`, `tool_timeout_sec: 3600` and a per-tool `output_token_limit`; without that limit Codex cuts every long
  result to about 12,000 characters. Unless `model_catalog_json` is already set or `--no-direct-catalog` is passed,
  the installer also adds a top-level `model_catalog_json = "<copy>/codex-models.direct.json"` line to
  `$CODEX_HOME/config.toml` (and removes that exact line on `uninstall`), switching built-in models to
  `tool_mode: "direct"` and `supports_search_tool: false` so MCP tools are delivered directly without Linux
  `code-mode host` crashes or `tool_search` indirection.
- **OpenCode**: requires OpenCode **2.x** (verified `2.2.7`); `install opencode` refuses OpenCode 1.x binaries because
  1.x rejects top-level `plugins` and `tool_output`. Writes the global `opencode.json` (`$OPENCODE_CONFIG_DIR` or
  `$XDG_CONFIG_HOME/opencode/`): `plugins` gets the absolute path of the copy (`opencode plugin add` accepts no local
  path), `tool_output` gets `{max_lines: 100000, max_bytes: 16000000}` unless you set it (without it results over
  about 50 KB are cut), and `media.image` gets `{auto_resize: false, max_width: 1000000, max_height: 1000000,
  max_base64_bytes: 1073741824}` unless you set it (prevents lossy JPEG re-encoding of large PNG tool images).
  `opencode/plugin.js` also registers a `compaction` session hook that keeps active NI-RUA turns from mistaking
  internal compaction prompts for user requests and instructs the summary to re-call `run_begin` first. A config
  that is JSONC or does not parse is refused with the exact edit to make by hand.

## Use

1. Start an NI-RUA gateway first, from an NI-RUA checkout, in external-agent mode:
   `mamba run -n ni-rua python -u -m ni_rua.apps.gateway --agent-backend external --config <config> --run-dir <fresh dir>`.
   For a local CPU-only SO-101 arm, use `--config configs/so101_standalone.json` (add
   `--observation-profile rgb_proprio_v1` on first boot before `/calibration` is completed; use
   `configs/so101_gateway.json` only when connecting optional remote VGGT/cuRobo GPU services).
   It serves MCP on `http://127.0.0.1:8790/mcp`.
2. Start the agent session. The `ni_rua` tools appear; the agent calls `run_begin`, works, and calls `run_finish`.
3. If the agent was started before the gateway: Claude Code `/mcp`, ni_rua, Reconnect (or a new session); OpenCode
   restart or reload; Codex start a new session. A gateway restarted with another robot needs the same.

Cancelling a call in the agent (Esc) does not stop the arm. Use the NI-RUA console stop, hold and estop.

`ni-rua-plugin doctor` checks the Node version, the host versions against the verified ones (Claude Code 2.1.321,
2.1.325 and 2.1.328, Codex 0.160.0, 0.167.0 and 0.169.0, OpenCode 2.2.7), whether each host lists the plugin, whether the Claude Code env
keys and permission entry are in effect, and whether a gateway answers `initialize` and `tools/list` on the port.

## Layout and maintenance

`.claude-plugin/` (Claude Code manifest and marketplace), `.mcp.json` (Claude Code MCP entry), `.codex-plugin/` and
`.codex-mcp.json` (Codex), `opencode/plugin.js` (OpenCode entry; also registers both skills), `skills/`, `bin/cli.mjs`.

`.codex-mcp.json` is generated from the tool registry; do not edit it by hand:
`mamba run -n ni-rua python scripts/gen_agent_plugin_codex_mcp.py` (`--check` to verify). Tests: `npm test` here, and
`tests/packaging/npm/test_agent_plugin_package.py` from the repository root. `test/tarball-check.sh` installs the packed tarball into
isolated homes with the pinned host binaries.

Verified so far: the entries connect over HTTP to an NI-RUA MCP server on the host versions above with a scripted
model. That is transport evidence, not a claim of support for real-model runs.
