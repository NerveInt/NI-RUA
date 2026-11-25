# Agent Plugin Spikes and Host Conformance

Summary of what was measured on the real Claude Code, Codex and OpenCode binaries before and while building the
[agent plugin](../agent-plugin.md) and the `external` backend. Every probe used isolated config homes (a fresh
temporary `HOME` plus `CLAUDE_CONFIG_DIR`, `CODEX_HOME`, `XDG_*`), a loopback fake model or a scripted one, a stub or
real `NiRuaMcpServer` and no real provider, key, robot or network; the user's own `~/.claude`, `~/.codex` and
`~/.config/opencode` were not touched. Scratch fixtures are not part of this repository; the repeatable evidence is
[`tests/cortex/external/test_external_host_e2e.py`](../../../tests/cortex/external/test_external_host_e2e.py),
[`agent-plugin/test/tarball-check.sh`](../../../agent-plugin/test/tarball-check.sh) and the Node installer tests.
Labels: **run** observed by running the binary; **binary** read in the binary's help, strings or schemas;
**inference** reasoned, not observed.

Hosts: Claude Code 2.1.321 (bundled with `claude-agent-sdk==0.2.180`) and the user's 2.1.325; Codex 0.160.0 (bundled
with `codex-cli-bin`) and the user's 0.167.0 (spikes, tarball check) or 0.169.0 (end-to-end suite); OpenCode 2.2.7 (the pin,
`~/.local/share/ni-rua/opencode-2.2.7/opencode`). macOS arm64, Node 25.2.1.

## 1. Plugin formats and install paths (run unless marked)

- **Claude Code.** `.claude-plugin/plugin.json` (only `name` required) and `marketplace.json` (`{name, owner,
  plugins:[{name, source}]}`); MCP in the plugin's `.mcp.json`; skills in `skills/<name>/SKILL.md`. `claude plugin
  marketplace add <dir>` then `claude plugin install <plugin>@<market> [-s user|project|local]`; a directory
  marketplace is read in place, not copied. `enabledPlugins` and `extraKnownMarketplaces` land in the scope's
  `settings.json` (`user`: config dir, `project`: `.claude/settings.json`, `local`: `.claude/settings.local.json`).
  Tool names are `mcp__plugin_<plugin>_<server>__<tool>`. A plugin cannot carry native tools, only MCP, skills,
  commands, agents and hooks. A marketplace entry with an npm source works (binary fetches via `npm`); installing
  straight from a registry was gated for the account.
- **Codex.** `.codex-plugin/plugin.json` with `mcpServers` and `skills` pointing to a file and a directory; `codex
  plugin marketplace add <dir>` and `codex plugin add <plugin>@<market>` copy the plugin to
  `$CODEX_HOME/plugins/cache/<market>/<plugin>/<version>/` and write `[marketplaces.<m>]` and `[plugins."<id>"]` to
  `config.toml`. A marketplace of the same name from another source is refused, so an upgrade is remove then add.
- **OpenCode 2.2.7.** A plugin is a package directory (`package.json` `exports` `./opencode/plugin.js`,
  `"type": "module"`) whose default export is `{id, setup}`; `setup(ctx)` can transform MCP entries
  (`ctx.mcp.transform`) and add skills (`ctx.skill.transform`). **`opencode plugin add` refuses every local form**
  (absolute directory, `file:`, a tarball): "Plugin target must be an npm registry package or Git package
  specifier". The loader does accept an absolute directory in the config: `"plugins": ["<abs dir>"]` in
  `$XDG_CONFIG_HOME/opencode/opencode.json`, loaded per location at first use.

## 2. Host permission and MCP-client defaults

All Claude Code rows are from `claude -p` in the default permission mode against a probe plugin.

| Question | Claude Code 2.1.321 and 2.1.325 | Codex 0.160.0 and 0.167.0 | OpenCode 2.2.7 |
| --- | --- | --- | --- |
| Is `initialize.instructions` shown to the model? | yes, in a system-reminder under `## plugin:<plugin>:<server>` | yes, as the MCP namespace description (section 3) | yes, in `<mcp_instructions>` (section 3) |
| Are MCP tools deferred? | **yes by default** (listed by name, loaded through ToolSearch); `alwaysLoad: true` in the plugin's `.mcp.json` entry loads them whole on both versions | no | no |
| Non-read-only tool call in the default mode | **refused** until the user allows `mcp__plugin_<plugin>_<server>` (`permissions.allow`); the server never receives the call | **refused** (`MCP tool call requires approval`) unless the entry sets `default_tools_approval_mode: "approve"`, which is honoured from the plugin's MCP file; read-only tools run | executed, no prompt |
| Can the plugin set host settings itself? | **no**: manifest `settings.env` and `settings.permissions` validate but are not applied, installed or not; user `settings.json` `env` and `permissions.allow` are | the plugin MCP file carries approval mode, timeouts and per-tool limits | `plugin.js` sets the remote entry (`codemode`, `timeout`); global `tool_output` only through the config file |
| Description cap | 4096 characters (+ suffix) by default; `CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH` lifts it, and it is a host-process setting | none seen (36,009 characters whole) | none seen |
| Text result cut | results over the output limit become a `<persisted-output>` pointer unless the tool publishes `_meta["anthropic/maxResultSizeChars"]` (the server publishes 10,000,000, delivered intact) | **cut to 11,984 characters** (head, `...N tokens truncated...`, tail) unless a per-tool or global `output_token_limit` is set | cut at about 50 KB by the global `tool_output` default |
| `anyOf` at the top of an input schema | the tool is dropped from the session and listed as unavailable (the server projects such schemas to a portable form) | not exercised in 17a | not exercised |
| Caller identity in `tools/call` `_meta` | `claudecode/toolUseId`; `CLAUDE_CODE_SESSION_ID` only in a stdio server's environment (equal to the run's session id, new on `--fork-session`); an HTTP server gets none | `callId`, `x-codex-turn-metadata {session_id, thread_id, turn_id, ...}`, `threadId`, `sessionId`, `plugin_id` | `ai.opencode/sessionID` (binary; a subagent's own id not exercised) |
| `notifications/tools/list_changed` | Claude Code re-lists tools | no second `tools/list` | not exercised |
| Timeouts | 300 s no-response/progress abort (`CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT`, `notifications/progress` resets it), `MCP_TOOL_TIMEOUT` ms is the hard cap, interactive sessions may auto-background long calls (`CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS`) | `tool_timeout_sec` | `timeout.execution` |
| Wake from outside | a background shell task finishing; `asyncRewake` hook field (a plugin hook's exit code 2 reached the model in `-p`; an idle interactive session not verified) | `codex queue --thread <id> --message <text>` | a background `shell` finishing |

Other measured facts: Claude Code images arrive as a text block, the image and a host-added `[Image: source: <config
dir>/projects/.../*.png]` text block; images above 512,000 bytes or 2000 px per side are re-encoded.

## 3. HTTP-first go/no-go

Method: the real `NiRuaMcpServer` (plain HTTP, fixed port, no bearer, `initialize.instructions`) with the real
X5 and SO-101 catalogues (17 tools then; 19 with the bracket tools), the measured briefing texts (X5 46,112 and
SO-101 60,091 characters), camera-sized frames and a scripted model, each host installed exactly as the installer
installs it.

**Verdict: go on all five binaries, including Codex's MCP client. No stdio bridge.** Every host connected with three
server-side changes only (fixed port, no bearer, `initialize.instructions`); all tools were delivered up front;
instructions were present; non-read-only calls ran in the default modes with the planned entries; camera-sized
WebP/JPEG frames were byte-identical.

- **Claude Code.** `{"type": "http", "url": ..., "alwaysLoad": true}` works in a plugin `.mcp.json` on both versions;
  description and schema lengths equal what the server sent (a schema's serialized length differs by 24 characters
  only because JavaScript prints `1.0` as `1`). Briefings 46,112 and 60,091 and a synthetic 120,000-character result
  arrived whole with the four env keys. `big.png` 2400x1627 (1,310,180 bytes) was re-encoded to a 437,663-byte JPEG.
  A 330 s call survived. Control without the env keys: the server's 5 s progress keepalive keeps a 330 s call alive,
  and without keepalive it was aborted at 329 s, so `notifications/progress` does reset the 300 s idle timer over
  HTTP.
- **Codex.** The rmcp client (protocol 2025-06-18) works over streamable HTTP with SSE replies and progress frames.
  Tools arrive in one `mcp__ni_rua` namespace; Codex appends ` This tool is part of plugin \`ni_rua\`.` (38
  characters) to every description. **Nested schema keywords are stripped from every MCP tool** (X5 catalogue 101,033
  schema characters, 23,439 delivered; `description` of nested nodes, `minimum`/`maximum`, `minLength`/`maxLength`,
  `maxItems`, `examples`, `pattern`, `default`, nested `if`/`then`/`oneOf`, ...); a stdio entry is cut identically,
  so it is the MCP client, not the transport, and host mode (`dynamicTools`) is unaffected. Without per-tool
  `output_token_limit` the 46,112, 60,091 and 120,000 character results arrived as 11,984; with 100000 they arrived
  whole. `default_tools_approval_mode: "approve"` and the limit are honoured from the plugin file. A 90 s call
  survived with `tool_timeout_sec: 3600`; with an edited `tool_timeout_sec: 20` a 35 s call failed at 20.0 s although
  the server sent progress every 5 s, so that setting is the hard bound.
- **OpenCode.** The `ctx.mcp.transform` remote entry connects (`oauth: false`, `codemode: false` keeps registry
  tools as direct `ni_rua_<tool>` calls). **The SO-101 briefing arrived as 50,135 of 60,091 characters** and a
  120,000-character result as 51,093, each with a pointer to a private file: the global `tool_output` limit, which the
  remote MCP entry cannot raise; the host-mode value `{"max_lines": 100000, "max_bytes": 16000000}` fixes it. A 90 s
  call survived with `timeout.execution: 3600000`; with 20000 a 35 s call was cancelled at 20.0 s although progress
  ran every 5 s, so progress does not extend it in 2.2.7. Arguments are validated against the schema before sending.
  `big.png` was re-encoded to a 300,388-byte JPEG by the default `media.image`; 640x480 frames were untouched.
- **Recovery (all observed headless).**
  - Gateway restarted between calls: transparent on all three (the server is stateless, no re-`initialize`).
  - Gateway started after the host: Claude Code `failed` (ECONNREFUSED), the model's call got "No such tool"; it
    reconnects by itself only if the gateway is up within about a second, `/mcp` reconnect (the stream-json
    `mcp_reconnect`) fixed it. OpenCode `failed` and stayed so 35 s after the gateway came up until a location reload.
    Codex `exec` had no `mcp__ni_rua` namespace in that turn (it retried and connected 0.8 s after the gateway).
  - Gateway restarted with another catalogue: no host re-lists tools; OpenCode kept serving the old catalogue (22,328
    versus 35,902 characters for `robot_action`) until the reload.
  - Gateway killed during a call: Codex errors at once (`Transport closed`); Claude Code and OpenCode hang until Esc
    or reconnect (Claude Code then tells the model the tool may have run; with the default idle timeout the call
    would instead fail after 300 s, which is why the installer does not set `CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT=0`).
- **Operator wake (`claude -p` stream-json, `opencode serve`).** A background shell task (`sleep 20; echo done`)
  was allowed in the default mode, the turn ended, and 20 s later the host sent a new model request by itself whose
  last user message was the task notification (`<task-notification>` on Claude Code, `<shell ... state="completed">`
  on OpenCode). `codex queue --thread` exists on 0.160.0 and 0.167.0 and is what the completion hook runs.

Not verified in the spikes: interactive TUIs on any host, Claude Code 2.1.325 and Codex 0.167.0 beyond items 1 to 3,
calls longer than 330 s, any real model.

## 4. Host conformance (end-to-end suite)

[`tests/cortex/external/test_external_host_e2e.py`](../../../tests/cortex/external/test_external_host_e2e.py) (support code in
[`tests/support/host_e2e.py`](../../../tests/support/host_e2e.py)) runs a real gateway `--agent-backend external`
over the SO-101 scene double, installs the plugin with the real installer into a fresh home, and drives each host
headless in its default permission mode against a scripted fake model replaying `run_begin`, `task_plan`,
`robot_observe` (two 640x480 camera frames), one long `robot_action`, two parallel reads and `run_finish`.

```sh
NI_RUA_HOST_E2E=1 NI_RUA_HOST_E2E_LONG=1 PYTHONPATH=tests mamba run -n ni-rua python -m unittest -v test_agent_external_host_e2e
```

Result: 5 tests OK in 563.8 s on macOS. The table is in the [guide](../agent-plugin.md#per-host-conformance-results).
Facts that came out of it:

- Codex's schema projection is the same on 0.160.0 and 0.169.0: 83,779 characters become 20,994; an enum-only node
  gets an inferred `type`, and a node containing `oneOf` becomes `{}`.
- The test-only long-call mode holds the worker's answer after the motion completed (`execute_hold_s` in
  [`tests/support/so101_scene.py`](../../../tests/support/so101_scene.py)). A slow motion would hit the 300 s
  `robot_action` `timeout_s` cap and the SO-101 `plan.max_age_s` of 120 s, and a hold placed before the motion made
  the 330 s receipt `batch_plan_expired_real_world_moved`.
- Not verified: interactive TUIs, real models, the X5 catalogue through a gateway, calls longer than 332 s, Linux.
