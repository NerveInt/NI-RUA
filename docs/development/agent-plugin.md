# Agent Plugin: Your Own Agent Drives NI-RUA

[English](agent-plugin.md) · [简体中文](agent-plugin.zh-CN.md)

The agent plugin lets the Claude Code, Codex or OpenCode session you already use act as the actor of an NI-RUA run, and lets that same agent schedule and analyse experiments from an NI-RUA checkout. It has two halves with different guarantees. Read [Verification status](#verification-status) before relying on either: the transport is verified with a scripted model, and one short real-model session per host passed (a smoke check, not a support claim).

## Two modes, not mutually exclusive

NI-RUA has two ways to use it, and one checkout and one install serve both; the mode is chosen per run with `--agent-backend`.

- **Hosted mode** is the full framework: NI-RUA launches and supervises `codex`, `claude_code` or `opencode` with its relay, budgets, provenance, sandbox, queue and benchmarks. It is the path for end-to-end control and formal evaluation.
- **Plugin mode** (this guide) is an optional add-on: `--agent-backend external` gives your own agent the robot tools, interactively and exploratorily.

Installing the plugin does not change hosted runs. A hosted launch never reads the host config home the installer writes to, and never binds the plugin port; the operator skill `ni-rua-experiments` lets your own agent schedule hosted-mode queues, so the two modes can be combined. Facts per host (code reading, pinned by [`tests/packaging/npm/test_agent_plugin_coexistence.py`](../../tests/packaging/npm/test_agent_plugin_coexistence.py), which writes the installer's entries into a fake user config home and builds each hosted launch):

| Host (hosted) | What the installer writes | Why a hosted run ignores it |
| --- | --- | --- |
| Claude Code | user-scope `enabledPlugins`, `extraKnownMarketplaces`, `env`, `permissions.allow` in `settings.json` | [`options.py`](../../ni_rua/cortex/agents/claude_code/options.py): run-private `CLAUDE_CONFIG_DIR` (`private/claude-config`), `setting_sources=[]`, `strict_mcp_config=True`, a clean launch environment |
| Codex | `[marketplaces.ni_rua]`, `[plugins."ni_rua@ni_rua"]` and `model_catalog_json` in `config.toml` and `plugins/cache/` of `~/.codex` | [`codex/session.py`](../../ni_rua/cortex/agents/codex/session.py) `_prepare`: a run-private `private/codex-home` with a generated `config.toml` (`features.plugins = false`, its own `model_catalog_json`, no marketplace or plugin table); from `model.codex_home` only the provider block and credential are read, no plugin or skill file is copied; `CODEX_HOME` points at the private home |
| OpenCode | a `plugins` entry (and `tool_output`) in the global `opencode.json` | [`config.py`](../../ni_rua/cortex/agents/opencode/config.py): run-private `OPENCODE_CONFIG_DIR` == `$XDG_CONFIG_HOME/opencode`, private XDG data, cache and state, and a generated config without `plugins` |

Only the `external` backend binds the plugin port (`agent.external.mcp_port` / `--external-mcp-port`, 8790, [`external/session.py`](../../ni_rua/cortex/agents/external/session.py)); the hosted backends use `NiRuaMcpServer`'s default ephemeral port 0 with a per-run bearer, and a hosted launch does not probe 8790. Residuals: an operator can still point `model.codex_home` or a Claude `claude_settings` upstream at the same home on purpose, which reads only the provider and credential entries named in [Agent backends](agent-backends.md); and this is verified at launch-construction level, not with a real agent per host.

## The two halves

| | Tool half (new capability) | Scheduling half (documentation over existing tools) |
| --- | --- | --- |
| What | The `external` agent backend: `--agent-backend external` brings up the robot, the `Harness` and the existing [`NiRuaMcpServer`](../../ni_rua/cortex/agents/mcp_server.py) on a fixed loopback port, launches no agent and no model relay, and your own agent drives the run through MCP. | The `ni-rua-experiments` skill teaches your agent to run single gateways, `ni-rua-queue`, `ni-rua-report`, `ni-rua-audit` and the experiment library by the rules in [AGENTS.md](../../AGENTS.md). |
| Code | [`ni_rua/cortex/agents/external/`](../../ni_rua/cortex/agents/external/session.py), [`run_bracket.py`](../../ni_rua/runtime/tools/plugins/run_bracket.py) | [`agent-plugin/skills/ni-rua-experiments/SKILL.md`](../../agent-plugin/skills/ni-rua-experiments/SKILL.md); no new Python |
| Actors it can run | Your own agent, interactively | Queue cases keep NI-RUA-hosted actors (`codex`, `claude_code`, `opencode`) |
| Prerequisites | `@nerveint/ni-rua-plugin` + `ni-rua-harness` (auto-installed into isolated Python 3.12 envs via `setup`, or a local checkout) and the robot environment | A checkout, the `ni-rua` environment, a provider profile in `configs/profiles/`, the pinned agent runtime of the chosen backend, and a GPU host for RoboDojo benchmarks (a laptop can still replay, report and audit) |

An external run is interactive-only and not queueable: the queue refuses `agent_backend="external"` (`external_backend_not_queueable`, [`queue.py`](../../ni_rua/apps/evaluation/queue.py)), because nobody would attach an agent to a queued case. Put external runs in no manifest.

## What the package contains and what `setup` installs

The npm package [`agent-plugin/`](../../agent-plugin/README.md) (published as `@nerveint/ni-rua-plugin`, CLI binary `ni-rua-plugin`, zero npm dependencies) holds the per-host manifests wired to the **lazy-loading stdio MCP bridge (`ni-rua-plugin mcp`)**, three skills (`ni-rua-setup` for end-to-end install, hardware binding, S0–S6 calibration, and diagnostics; `ni-rua-robot` for the actor; and `ni-rua-experiments` for the operator), and the CLI (`setup`, `mcp`, `gateway`, `install`, `uninstall`, `doctor`). The npm tarball itself contains no Python binaries.

- **One-line zero-clone install (SO-101 included by default):** `npx -y @nerveint/ni-rua-plugin setup <codex|claude|opencode|all>` installs the host plugin AND bootstraps two isolated Python 3.12 runtimes by default: `~/.local/share/ni-rua/envs/gateway` (`ni-rua-harness`, preferring `uv` and falling back to `python3.12 -m venv`) and `~/.local/share/ni-rua/envs/so101` (SO-101 hardware dependencies). Pass `--no-so101` to skip SO-101 hardware dependencies (`--so101` remains accepted as a backwards-compatible alias), or `--skip-gateway` to install only the host plugin manifests.
- **Global CLI (recommended for manual gateway / doctor use):** `npm i -g @nerveint/ni-rua-plugin` exposes `ni-rua-plugin` on `PATH` for `gateway`, `doctor`, `install`, and `uninstall`. (The installed host plugin always invokes the durable copy under `~/.local/share/ni-rua/agent-plugin/<version>/bin/cli.mjs mcp`, so `npx`-only installs work reliably without global install.)
- **Requirements:** Node 18 or newer, Python 3.12 (or `uv`), the host CLI you want to use, and macOS or Linux (Windows only through WSL, because the gateway uses `fcntl`).
- **From a source checkout:** you can still run `node agent-plugin/bin/cli.mjs` directly against the `mamba` `ni-rua` environment.

## Quick start

### Default single-window SO-101 workflow (lazy stdio MCP bridge)

1. Install the plugin and isolated Python 3.12 gateway + SO-101 runtimes once:

   ```sh
   npx -y @nerveint/ni-rua-plugin setup codex         # or: claude | opencode | all (--no-so101 to opt out)
   ```

2. Bind your SO-101 serial port and cameras once (writes `~/.config/ni-rua/so101.local.json`):

   ```sh
   ~/.local/share/ni-rua/envs/gateway/bin/ni-rua-so101 init-config   # or `ni-rua-so101 init-config` if on PATH
   ```

3. Open `codex` (or `claude` / `opencode`) directly and ask it to set up, calibrate, or operate the arm — no second gateway terminal is required:

   ```text
   $ codex
   > Use NI-RUA to inspect the table, lift the SO-101 gripper 2 cm, verify, and return.
   ```

   Together, the bundled **`ni-rua-setup`** and **`ni-rua-robot`** skills let your agent guide hardware binding (`ni-rua-so101 init-config` / `cameras` / `commission`), S0–S6 calibration, troubleshooting (`ni-rua-plugin doctor`), and robot execution end-to-end. On the first `run_begin` call, `ni-rua-plugin mcp` automatically launches the singleton local SO-101 gateway (adding `--observation-profile rgb_proprio_v1` automatically if no `camera_calibration.json` exists yet) and prepends the live Web console E-STOP URL (`http://127.0.0.1:8781/console`, plus `/calibration` if uncalibrated).

### How the lazy stdio MCP bridge (`ni-rua-plugin mcp`) works

- **Zero overhead during normal coding sessions:** `initialize` responds immediately without starting a gateway or touching hardware. `tools/list` probes `NI_RUA_MCP_URL` (default `http://127.0.0.1:8790/mcp`, ~300 ms); if no gateway is running yet, it returns the bundled static tool definitions so Codex, Claude Code, and OpenCode show zero MCP connection errors.
- **Transparent proxy to manual gateways:** On `tools/list` or `tools/call`, if an upstream HTTP gateway is already reachable at `NI_RUA_MCP_URL` (such as `ni-rua-plugin gateway --dry-run`, a manual SO-101 gateway, RoboDojo, or LIBERO), the bridge transparently proxies JSON-RPC over HTTP (preserving `Mcp-Session-Id` and handling both JSON and `text/event-stream` SSE responses).
- **Singleton lazy auto-start:** If no gateway is running on `tools/call` and `NI_RUA_AUTO_START` is not `"0"`, the bridge acquires `~/.local/share/ni-rua/gateway.lock`, spawns a detached local gateway (`--agent-backend external --config <NI_RUA_GATEWAY_CONFIG || "so101"> --port 8781 --external-mcp-port 8790 --run-dir ~/.local/share/ni-rua/runs/run-<timestamp>`, updating `~/.local/share/ni-rua/runs/latest` symlink and logging to `~/.local/share/ni-rua/logs/gateway.log`), initializes the upstream MCP session, emits `notifications/tools/list_changed`, and forwards the call. If startup fails (e.g., missing runtime or `so101.local.json`), it fails fast with an actionable MCP tool error pointing to `setup` or `ni-rua-so101 init-config`.

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `NI_RUA_MCP_URL` | `http://127.0.0.1:8790/mcp` | Upstream HTTP MCP gateway endpoint to probe and proxy. |
| `NI_RUA_AUTO_START` | enabled (unset) | Set to `0` to disable automatic local SO-101 gateway startup (proxy-only mode; fails fast unless a manual gateway is running). |
| `NI_RUA_GATEWAY_CONFIG` | `so101` | Config preset or file path used when the bridge auto-starts a local gateway. |

### Manual gateway mode (dry run, RoboDojo, LIBERO, custom `--run-dir` / `--prompt`)

Start a gateway manually before (or alongside) your agent session whenever you want dry-run testing, explicit `--run-dir` / `--prompt`, remote GPU perception, or simulation (RoboDojo / LIBERO). An external run is never `--benchmark`; the stdio bridge automatically detects the running gateway on `http://127.0.0.1:8790/mcp` and proxies to it:

```sh
# Optional: keep `ni-rua-plugin` on PATH and run diagnostics
npm i -g @nerveint/ni-rua-plugin
ni-rua-plugin doctor --so101

# 1. Zero-hardware smoke test (lightweight local dry-run MCP server on 127.0.0.1:8790/mcp)
ni-rua-plugin gateway --dry-run

# 2. Manual SO-101 gateway — standalone local CPU-only default (auto-adds --observation-profile rgb_proprio_v1 on cold start before calibration)
ni-rua-plugin gateway --config so101 --port 8781 --run-dir runs/my-external --prompt "<task>"

# 3. RoboDojo simulation from a checkout (one idle GPU; see running.md)
mamba run -n ni-rua python -u -m ni_rua.apps.gateway --agent-backend external \
  --task stack_bowls --gpu 0 --port 8767 --run-dir runs/my-external --prompt "<task>"

# 4. LIBERO simulation (simulation-only Franka Emika Panda) from a checkout
mamba run -n ni-rua python -u -m ni_rua.apps.gateway --agent-backend external \
  --config configs/libero.example.json --port 8768 --run-dir runs/my-libero --prompt "<task>"
```

For first-time SO-101 setup before `camera_calibration.json` exists, both the lazy bridge auto-start and `ni-rua-plugin gateway --config so101` automatically detect the missing calibration file and append `--observation-profile rgb_proprio_v1` so you can open `http://127.0.0.1:8781/calibration?lang=en` immediately (when invoking `ni-rua` directly, pass `--observation-profile rgb_proprio_v1` explicitly). Once calibration is applied, daily local CPU-only runs use `--config so101` (bundled `so101_standalone.json`) directly. Use `--config configs/so101_gateway.json` only when connecting optional remote VGGT/SAM/cuRobo GPU services. Deployment configuration is read from `~/.config/ni-rua/so101.local.json` (generated via `ni-rua-so101 init-config`) or `./configs/so101.local.json` inside a checkout; see [SO-101 Quickstart](../guides/so101-quickstart.md), [SO-101 Calibration Guide](../guides/so101-calibration.md), [SO-101 configuration](so101-configuration.md), and [SO-101 integration, section 6.3](so101-integration.md). Open `http://127.0.0.1:<port>/console?lang=en` (the `--port` above); `--run-dir` must be a new directory.

The gateway serves HTTP MCP on `http://127.0.0.1:8790/mcp` (`agent.external.mcp_port`, overridable via `--external-mcp-port` and matched via `NI_RUA_MCP_URL`, see [Configuration](configuration.md#agent-backend-agent)) and prints the endpoint at start. A taken port refuses the launch with one sentence naming the port. For RoboDojo runs, `run_begin` also delivers the native `task_instruction` (with any `task_instruction_override`), `public_task_rules` and `default_task_notes` in `native_task` and `task_notes`.

Use the NI-RUA console (the gateway's `--port`, default `http://127.0.0.1:8781/console?lang=en`) to stop, hold or emergency-stop the arm.

## Dry run without hardware

- **Built-in CLI dry run (no checkout, no Python needed):** `ni-rua-plugin gateway --dry-run` starts a lightweight local MCP server on `http://127.0.0.1:8790/mcp` exposing `run_begin`, `robot_observe`, `robot_action`, and `run_finish` so you can verify `Agent ↔ MCP` end to end in seconds; `ni-rua-plugin mcp` proxies to it automatically.
- **Full SO-101 scene double (checkout only):** The test suite's scene double serves the real SO-101 backend and adapter over a simulated servo bus and rendered cameras (no serial port, no arm). It is test support, not part of the wheel. From the checkout:

```sh
D=$(mktemp -d)
PYTHONPATH=tests mamba run -n ni-rua python - "$D" <<'EOF'
import json, sys
from support import so101_scene
d = sys.argv[1]
deployment, scene = so101_scene.write_scene(d + "/scene", so101_scene.scene_document("red"))
features = json.load(open("configs/so101_gateway.json"))["features"]
features.update(terminal=False)
json.dump(dict(version=1, backend=dict(environment="so101_real", robot_model_id="so101", deployment=str(deployment)),
               features=features, agent=dict(backend="external"),
               run=dict(experiment_root=d + "/exp", category="dry_run", task="desk_free_task"),
               remote=dict(enabled=False)), open(d + "/ni-rua.json", "w"))
EOF
PYTHONPATH=tests:. NI_RUA_SO101_SCENE="$D/scene/scene.json" mamba run -n ni-rua python -u -m support.so101_scene \
  --config "$D/ni-rua.json" --agent-backend external --run-dir "$D/run" --port 8782 --no-remote --prompt "<task>"
```

The VGGT-backed perception reports `unavailable` (no service), and the cube is red at `[0.23, 0.08]` m. Use it to try `install`, `doctor` and the console; it proves nothing about real motion.

## What the installer writes

The installer first copies the package to `~/.local/share/ni-rua/agent-plugin/<version>/` and installs from that durable copy (`npx` caches are not stable, and Codex copies a local marketplace into its own cache). Every host command and settings change is printed before it runs. Source: [`agent-plugin/bin/cli.mjs`](../../agent-plugin/bin/cli.mjs).

| Option | Meaning |
| --- | --- |
| `--port N` | Upstream HTTP MCP port written into the durable copy's stdio bridge configuration (default 8790 = `agent.external.mcp_port`, overridable at runtime via `NI_RUA_MCP_URL`). One port is shared by all installs, because they load the same copy; a different `--port` while other installs exist is refused. |
| `--scope user\|project\|local` | Claude Code only (`install` and `uninstall`); default `user`. `project` and `local` act on the current directory (workspace must be trusted in Claude Code). |
| `--no-host-env`, `--no-allow` | Claude Code opt-outs for the `env` keys and the `permissions.allow` entry. |
| `--no-direct-catalog` | Codex opt-out for writing `model_catalog_json` into `$CODEX_HOME/config.toml`. |
| `--force` | Replace a durable copy of the same version that has different content (refused otherwise). |
| `--claude-bin`, `--codex-bin`, `--opencode-bin` | Host executable instead of the first one on `PATH`. |

**Claude Code.** (With the default `--scope user` this is for every Claude Code session of that user, in any project, not only NI-RUA sessions: the lazy stdio bridge loads cleanly with zero gateway process and its robot tools are pre-approved. Use `--scope project`, run from the NI-RUA checkout, if you prefer scoping to one repo; see [Residuals](#residuals-documented-not-hardened). Note that Claude Code ignores project/local `.claude/settings.json` — including `permissions.allow` and `env` — in an untrusted workspace, and resolves workspace trust at the enclosing `.git` root when the directory has no `.git` of its own; open interactive `claude` once in that workspace to accept trust, or check with `ni-rua-plugin doctor`.) `claude plugin marketplace add <copy>` and `claude plugin install ni_rua@ni_rua -s <scope>`; the plugin's `.mcp.json` launches the local stdio bridge (`{"command": "node", "args": ["${CLAUDE_PLUGIN_ROOT}/bin/cli.mjs", "mcp"]}`). In that scope's `settings.json` (`<CLAUDE_CONFIG_DIR or ~/.claude>/settings.json` for `user`, `.claude/settings.json` for `project`, `.claude/settings.local.json` for `local`) it adds four `env` keys and one permission:

| Key | Value | Why |
| --- | --- | --- |
| `MAX_MCP_OUTPUT_TOKENS` | `2000000` | tokens; stops Claude Code rewriting big text results |
| `MCP_TOOL_TIMEOUT` | `3600000` | ms; bounds one MCP call (default 60 s) |
| `CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS` | `0` | never move a long MCP call to the background |
| `CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH` | `1000000` | characters; the default 4096 truncates the generated field guides |
| `permissions.allow` | `mcp__plugin_ni_rua_ni_rua` | without it the first robot call is refused in the default permission mode |

These are the host-mode keys of [`claude_code/options.py`](../../ni_rua/cortex/agents/claude_code/options.py) minus `CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT=0`: progress keepalives already keep healthy calls alive, and with that key a gateway crash mid-call hangs until a `/mcp` reconnect instead of failing after 300 s. Keys you already set are kept. A kept value that defeats the plugin (a numeric key below the table's value, or auto-backgrounding on) is reported as a warning by `install` and `doctor`.

**Codex.** `codex plugin marketplace add <copy>` and `codex plugin add ni_rua@ni_rua`. An upgrade or a changed port removes the plugin and marketplace first (Codex cannot repoint one). Codex is the same: the installed plugin is available to the Codex sessions of that user via the local stdio bridge (`node <copy>/bin/cli.mjs mcp`) and its tools are pre-approved. The generated [`.codex-mcp.json`](../../agent-plugin/.codex-mcp.json) carries `default_tools_approval_mode: "approve"`, `tool_timeout_sec: 3600` (a hard bound; progress does not extend it), `startup_timeout_sec: 30` and, for every actor tool of the registry, `output_token_limit: 100000`: without that limit Codex cuts every long result to about 12,000 characters (the SO-101 briefing is 60,091). The map and the direct model catalog [`codex-models.direct.json`](../../agent-plugin/codex-models.direct.json) are generated by [`scripts/gen_agent_plugin_codex_mcp.py`](../../scripts/gen_agent_plugin_codex_mcp.py); do not edit them by hand. Because Codex 0.160.0/0.169.0 marks built-in models such as `gpt-6.1-sol` as `tool_mode="code_mode_only"` (which crashes `code-mode host` on Linux with `SIGSEGV`/gRPC `BrokenPipe`) and `supports_search_tool=true` (which hides MCP tools behind Responses `tool_search` and fails on providers without `tool_search`), `codex-models.direct.json` sets `tool_mode="direct"` and `supports_search_tool=false` on every model; `install codex` adds a top-level `model_catalog_json = "<copy>/codex-models.direct.json"` line to `$CODEX_HOME/config.toml` unless already set or `--no-direct-catalog` is passed, and `uninstall codex` removes that exact line.

**OpenCode.** Requires OpenCode **2.x** (verified `2.2.7`); `install opencode` refuses OpenCode 1.x binaries before writing anything because 1.x rejects top-level `plugins` and `tool_output`. [`opencode/plugin.js`](../../agent-plugin/opencode/plugin.js) registers the local stdio MCP bridge (`node <copy>/bin/cli.mjs mcp`, `codemode: false` so registry tools stay direct `ni_rua_<tool>` calls, `timeout.execution: 3600000` ms because progress does not extend it in 2.2.7), all three skills (`ni-rua-setup`, `ni-rua-robot`, `ni-rua-experiments`), and a `compaction` session hook that keeps active NI-RUA turns from mistaking internal compaction prompts for user requests and tells the summary to call `run_begin` first after compaction. `opencode plugin add` accepts no local path, so the installer writes the global `opencode.json` (`$OPENCODE_CONFIG_DIR` or `$XDG_CONFIG_HOME/opencode/`, strict JSON only; JSONC or an unparsable file is refused with the exact edit to make by hand): `plugins` gets the absolute path of the copy, `tool_output` gets `{"max_lines": 100000, "max_bytes": 16000000}` unless you set it (the host-mode value; without it results over about 50 KB are cut, and the SO-101 briefing arrives as 50,135 of 60,091 characters), and `media.image` gets `{"auto_resize": false, "max_width": 1000000, "max_height": 1000000, "max_base64_bytes": 1073741824}` unless you set it (prevents lossy JPEG re-encoding of large PNG tool images).

**Records, uninstall, upgrade.** The installer records what it did in `~/.local/share/ni-rua/agent-plugin/installed.json` (schema 2): one entry per host and config home (Claude Code: `CLAUDE_CONFIG_DIR` + scope, plus the directory for `project` and `local`; Codex: `CODEX_HOME`; OpenCode: the config file), so two profiles or projects never overwrite each other. `uninstall <host>` removes exactly what that entry added and leaves values you changed since; run it with the same `CLAUDE_CONFIG_DIR`, `CODEX_HOME`, `XDG_CONFIG_HOME` and scope as the install. If a host command fails for a reason other than "not installed", the record and the copy are kept, the manual commands are printed and the exit code is non-zero. It never removes what it did not create, keeps symlinked settings files and their mode, and a lock directory (`<data>/.lock`) serialises runs. Re-running `install` is the upgrade path; a version directory is immutable, so changed content under the same version needs a new version or `--force`. Every refusal runs before the first write, so a refused reinstall leaves the working configuration untouched. Covered by [`agent-plugin/test/installer.test.mjs`](../../agent-plugin/test/installer.test.mjs).

## The bracket: `run_begin` and `run_finish`

Two registry tools ([`run_bracket.py`](../../ni_rua/runtime/tools/plugins/run_bracket.py)), role `actor`, declared effect `session` (see [Tool Plugins](tool-plugins.md#side-effects-and-program-authorization)), registered only for `agent.backend="external"`. Neither has a physical effect or judges task success. The descriptions are the contract; in summary:

- **`run_begin`** (no inputs) returns `kind="briefing"`: `instructions` (the run's startup instructions, the host-mode text rendered from the `EmbodimentProfile` and scrubbed, with a local-source variant that has no web-search, working-directory or sandbox claims and no run-directory path; with the default `features.robot_urdf_delivery="file"`, raw URDF text is omitted from `run_begin` and the instructions direct the agent to call the read-only `robot_information` tool on demand instead of inlining 46k–60k URDF prose), `task {text, source}`, optional `native_task {task, instruction, public_task_rules}` and `task_notes` (populated on simulated environments such as RoboDojo or when `--task-notes` is given), `budget` (native step counts, `null` = unknown), `turn.opened`, and—on the first `run_begin` of a simulated-clock run when `features.startup_inline_grounding=true` (strict 5-gate guard: non-resume, `simulated_steps` clock only, first `opened=true`, `action_count==0`, `latest==initial_observation_id`)—inline `startup_inline_grounding` text and initial camera images (never replayed on re-briefing, real-time arms, or after actions). Under `context_delivery="delta"`, `run_begin` also seeds `CapabilityDelivery` for the current epoch. A `run_begin` inside an already-open turn re-reads the briefing and calls `Harness.model_context_reset("external_rebriefing")`, so the decision reminder rides on the next tool result; the first `run_begin` of a turn does not. Opening a subsequent turn or attaching a new external client refreshes the context epoch so a fresh client never inherits an earlier client's delta baselines.
- **`run_finish`** (`summary`, string, 1 to 16,000 characters) closes the turn as `completed` and records the summary as the turn's final answer (transcript origin `run_finish_summary`, not native prose); result `{turn_closed: true}`. Helper programs keep running: read `program_status` first; a job paused for review stops with `program_run resume decision=reject`.
- **No default task.** `task.text` comes only from `--prompt` or `run.prompt`; `task.source` is `operator_prompt` or `none`, and then the user states the task to the agent. The development-acceptance default prompt is never used, because it moves the robot. A resume keeps the archived task; a resume with `--prompt` replaces it.
- **The briefed-turn gate.** Every other tool is served only while a turn opened by `run_begin` is open and its briefing was delivered. Otherwise the call is refused with nothing executed and no transcript row (the MCP raw log has it; event `agent.tool_refused {code, tool}`): `external_briefing_required: call run_begin first; it returns the operating instructions and the task`, or, after a briefed turn closed, `external_turn_closed` with one sentence for the reason: `finished` (`run_finish`), `operator_stop` (the operator stopped the run; motion stays refused until the operator resumes) or `operator_reset` (the operator reset the environment; earlier observations and plans no longer describe it). A bracket call that fails before delivering a briefing closes the turn it opened (`interrupted`, `external_bracket_call_failed`). Any caller is admitted while a turn is open, so the host's own subagents share the run's turn.
- **Concurrency.** The gate follows dispatch order, not HTTP arrival. A call dispatched before `run_begin` was admitted is refused with nothing executed and can simply be repeated; no wait was added. The tool description tells the agent to wait for the `run_begin` result before other NI-RUA calls.
- **Order of one step's calls.** <a id="order-of-one-steps-calls"></a>External mode executes ordered (stateful) calls in the order their `tools/call` requests ARRIVE, one after the other; parallel-safe calls may overlap. NI-RUA does not see the model's response here (no relay), so it cannot know the order the model emitted the calls in, and no `_meta` field of any of the three hosts carries one: OpenCode 2.2.7 sends `ai.opencode/sessionID` and `progressToken`, Claude Code `claudecode/toolUseId` (an id, not a position), Codex thread, session and turn ids. JSON-RPC ids are the client's request counter and are not used as an order (a gap cannot be told from a request of another kind). What that means per host: Claude Code sends an ordered call only after the previous one was answered, so its order is the model's; OpenCode sends all calls of a step at once, and two ordered calls of one step can then execute swapped (measured in host mode before the fix there: 4 of 96 steps with three ordered calls, under load). Host mode does not have this residual ([Agent Backends](agent-backends.md#mcp-tool-transport)). In external mode with OpenCode, or any client that sends a step's calls concurrently, issue calls whose order matters in separate steps (wait for each result before the next call).
- **Context delivery (`agent.external.context_delivery`, default `"delta"`, opt-out `"full"`).** External runs default to `"delta"` with a single-main write-back lease on `ExternalToolBridge`: overlapping concurrent calls and non-main OpenCode child-session calls (`ai.opencode/sessionID`) automatically fall back to `isolated` full-delivery mode without mutating main baselines, and new turns or new clients refresh the context epoch. Because external mode has no model relay (`ModelRequestProxy` / `ContinuityTracker`), hard boundaries remain: compaction detection relies on soft plugin hooks plus `run_begin` re-briefing (silent host pruning cannot be detected automatically), relay-side image trimming (`trim_oversized_requests`/`trim_max_inline_images`) and exact-image reuse (`reuse_exact_images`) are unavailable, and Claude Code HTTP `_meta` carries no session id (so serial subagent calls while the main chain is blocked cannot be separated from the main lease at the protocol layer). Set `agent.external.context_delivery="full"` to force `Harness.context_epoch_volatile` on every call.
- **Re-briefing.** Call `run_begin` again after a context compaction, a new session or a resume. The MCP `initialize.instructions` (a short fixed text, external only; host mode sends none) and the `ni-rua-robot` skill say so, and also say that cancelling a call does not stop the arm.

Regressions: [`tests/cortex/external/test_external_config_and_session.py`](../../tests/cortex/external/test_external_config_and_session.py), [`tests/cortex/external/test_external_workbench.py`](../../tests/cortex/external/test_external_workbench.py).

## Operator console in external mode

The console works as in every mode, with these differences ([`session.py`](../../ni_rua/cortex/agents/external/session.py), [`workbench.py`](../../ni_rua/apps/gateway/workbench.py)):

- **Agent bar.** Read-only "External agent" card ([`web/agent-settings-view.js`](../../web/agent-settings-view.js)): the MCP endpoint and two facts, never "connected" (the server is stateless and cannot tell whether an agent is still there): the client of the newest `initialize` (`clientInfo` name and version, labelled self-reported) and the time of the newest tool call ("No tool calls yet" when there was none). The waiting text shows only while no `initialize` has ever arrived. `ni-rua-plugin doctor` sends an `initialize` as `ni-rua-plugin-doctor`; the card labels exactly that name as a doctor probe, not an agent (the one name match, `DOCTOR_CLIENT_NAME` in [`web/agent-settings-model.js`](../../web/agent-settings-model.js); any other name is shown as self-reported). A replay of an archived external run shows "Finished run (replay)" (`client_view()["replay"]`), not a waiting card.
- **Refused commands.** `prompt`, `steer`, `reply`, `set_model`, `set_effort`, `set_service_tier`, `apply_agent_settings`, `handoff_run` and `retry_model_requests` fail with `<command>_unsupported_by_backend`; no handoff starts or leaves an external run (it starts only from the command line).
- **An open agent turn never blocks operator commands**: `resume`, `go_home`, calibration and observer captures. Motion stays protected by `request_stop`, the robot lock, the active action and active programs.
- **`stop`, `estop` and gateway shutdown** close an open turn as `interrupted` (`agent.turn_closed_by_operator {reason: "operator_stop"}`); `reset` closes it with `operator_reset` once the reset has happened (a refused reset changes nothing). The agent's next call gets `external_turn_closed` naming the action.
- **After `stop`, `resume` hands motion back at once** to a still-connected agent: once the agent has called `run_begin` again, the next motion call is served immediately. Read the console state before pressing `resume`.
- **Cancelling a tool call (Esc, interrupt) in the agent does NOT stop the arm.** An admitted action keeps running. Use the console stop, hold or estop.

## Recovery

With the `v0.1.1` lazy stdio bridge (`ni-rua-plugin mcp`), agents no longer fail to connect when started before a gateway: normal sessions receive bundled static tool definitions cleanly, SO-101 auto-starts on first `run_begin`, and a manual gateway started later on `NI_RUA_MCP_URL` is picked up transparently on the next call. Historical HTTP-first spike behaviour (report [Agent Plugin Spikes, section 3](reports/agent-plugin-spikes.md#3-http-first-gono-go); scripted model, headless) still applies to upstream HTTP restarts and mid-call crashes:

- **Switching to a different robot** (different tool catalogue) after a gateway restart still benefits from `notifications/tools/list_changed` or a reconnect/new session (Claude Code `/mcp` → `ni_rua` → Reconnect, OpenCode reload, Codex new session), because older clients may cache initial tool schemas. If tool descriptions no longer match the briefing, reconnect.
- **Restarting the gateway between two calls** needs nothing on any host; both the stdio bridge proxy and the HTTP server recover cleanly.
- **A gateway crash during a call.** Codex errors at once ("Transport closed"). Claude Code and OpenCode wait until you press Esc or reconnect; Claude Code then tells the model the tool may have run. The arm may still have been moving: check state with read-only tools before repeating a motion. (With the Claude Code default idle timeout the call fails after 300 s; the installer leaves that default for this reason.)
- **Manual proxy-only mode (`NI_RUA_AUTO_START=0`).** If no gateway is listening on `NI_RUA_MCP_URL`, `tools/call` returns an actionable MCP error telling you to start `ni-rua-plugin gateway`; once started, retry the call without restarting the agent.

## Information boundary and benchmark validity

- **Boundary.** The manifest records `agent_backend.kind="external"` and `information_boundary="mcp_boundary_only"`: the agent sees only the tool catalogue, `initialize.instructions`, tool results and refusals; NI-RUA sees none of its prompts, model, reasoning or other tools. The metadata scrubber, short ids and the other [actor information boundary](actor-information-boundary.md) rules apply to everything NI-RUA sends.
- **Exploratory, interactive-only.** `ni-rua-report` lists `kind=external` as its own condition with `exploratory=true` ([`report.py`](../../ni_rua/apps/evaluation/report.py)). `--benchmark` and `--exit-after-turn` are refused (`external_backend_is_interactive_only`), as are `features.sandbox_enabled` (`external_backend_cannot_sandbox`), `features.visual_review_model`, `model.price_card` (`external_backend_has_no_model_relay`) and a disabled bracket tool. A resume is allowed; `handoff_run` refuses `external` in either direction.
- **Unknown, never zero.** There is no relay, so `model`, `provider`, `effort`, `price_card`, `model_request_limit` and `no_progress_timeout_s` are `null` in the manifest, and `served_models`, `token_usage` and `model_requests` are `null` in the summary. There is no model request budget, no served-model verification, no relay `ContinuityTracker` for silent compaction detection, and no relay-side image trim or exact-image reuse.
- **What NI-RUA still enforces:** the single physical entry point, motion admission, the stage guards and tolerances, the briefing gate, the single-main delta lease, the scrubber and the typed receipts. **What it cannot:** the agent's own model, its prompts and context, its other native tools (the agent keeps all of them, including shell and, unless you disable it in the host, web search), reading run directories, or how its context is compacted.
- **Briefing size and compaction.** With the default `features.robot_urdf_delivery="file"`, `run_begin` directs the external agent to read URDF on demand via `robot_information` rather than inlining 46,000 to 60,000 characters of URDF text (setting `robot_urdf_delivery="inline"` restores inline URDF). Compaction hooks remain advisory/soft (and Codex has none), so re-brief with `run_begin` after each compaction. A run directory is an observer record; an agent that reads it, simulator assets or evaluator state makes the run not comparable (the actor skill says so, and NI-RUA cannot prevent it).
- Do not claim a single-factor result from an external run: model, host, prompt and permission mode all vary.

## Host differences recorded as facts

| Fact | Host | Consequence |
| --- | --- | --- |
| Nested schema keywords are stripped from every MCP tool (`description`, `minimum`/`maximum`, `minLength`/`maxLength`, `maxItems`, `examples`, `pattern`, `default`, nested `if`/`then`/`oneOf`, ...): 23,439 of 101,033 schema characters reach the model on the X5 catalogue, over stdio as well as HTTP (measured on 0.160.0 and 0.167.0 in the spike, and again on 0.160.0 and 0.169.0 in the host end-to-end: 83,779 characters become 20,994, an enum-only node gets an inferred `type` and a node containing `oneOf` becomes `{}`). Host mode uses `dynamicTools` and is unaffected. | Codex | The generated field guide in each tool description keeps meanings, units and bounds readable; runtime validation is the registry's. Codex appends ` This tool is part of plugin \`NI-RUA\`.` (38 characters) to each description. |
| Without a per-tool `output_token_limit`, results over about 12,000 characters are cut. | Codex | Set by `.codex-mcp.json`. |
| A local text block `[Image: source: <config dir>/projects/.../*.png]` follows every image; images over the host's limit are re-encoded (a 2400x1627 PNG became a 437,663-byte JPEG); camera-sized WebP/JPEG frames arrive byte-identical. | Claude Code | A recorded host difference; no image logic was changed for it. |
| The default `tool_output` cut is about 50 KB; images above the default resize limit are re-encoded. | OpenCode | The installer sets `tool_output` and `media.image`. |
| The default description cap is 4096 characters. | Claude Code | `CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH` is set. |
| No host re-lists tools after a gateway restart. | all | Reconnect, see [Recovery](#recovery). |
| Progress notifications reset Claude Code's 300 s idle timer but do not extend Codex's `tool_timeout_sec` or OpenCode's `timeout.execution`. | all | Both are set to 3600 s. |
| A client may identify itself through `_meta` ids (OpenCode `ai.opencode/sessionID`; Codex thread, session and turn ids; Claude Code `claudecode/toolUseId`). | all | Kept only in the MCP raw log; no caller policy. None of them carries the position of a call in its model step. |
| All tool calls of one model step are sent at once, ordered ones included. | OpenCode | External mode executes them in arrival order, which can differ from the model's order ([Order of one step's calls](#order-of-one-steps-calls)). |

## Residuals (documented, not hardened)

- The endpoint has no bearer: static host configs cannot know a per-run token, and keys in environment variables are against the project's rules. Loopback bind, the `Host` check and a refusal of any foreign `Origin` (the console rule; MCP hosts send none) are the only gates on `127.0.0.1:8790`. Host mode keeps its per-run bearer.
- Any local process of your user can drive the robot through that port, as with the console HTTP API on loopback.
- The agent's own shell can reach devices, run directories and credentials of your user; NI-RUA launches nothing to sandbox.
- Your own agent's git context is outside NI-RUA's control: started inside a git repository, the host may put that repository into its own context (Claude Code its branch, git user, status and recent commit subjects; Codex every `AGENTS.md` up to the repository root) and send it to its provider (Codex adds the repository root, remote URLs and latest commit hash to every turn request's metadata). NI-RUA switches this off only for the agents it launches ([Native capability policy](agent-backends.md#native-capability-policy)), so start the host agent outside a repository when a comparable run matters or when the repository must not reach the provider.
- Client identity (`clientInfo`, `_meta` ids) is self-reported. Under `context_delivery="delta"`, OpenCode's `ai.opencode/sessionID` isolates child sessions and `clientInfo.name` provides conservative truncation limits, while Claude Code HTTP `_meta` carries no session id (so serial subagent calls while the main chain is blocked remain a residual boundary that can acquire the single-main lease).
- Ordered calls that a client sends concurrently within one model step execute in arrival order, which can differ from the order the model wrote them in ([Order of one step's calls](#order-of-one-steps-calls)).
- The installer edits user settings (Claude Code `settings.json`, OpenCode `opencode.json`) and the host's plugin registries; everything is printed first and recorded for `uninstall`.
- One external gateway per port; a second one is refused at launch.
- **The default install reaches every session of your user.** With the default `--scope user`, every Claude Code session of that user loads the lazy `ni_rua` stdio bridge cleanly with zero background gateway process and pre-approved robot tools (`permissions.allow`). Codex is the same through `default_tools_approval_mode: "approve"`. Because the bridge auto-starts the local SO-101 gateway only when the agent explicitly invokes an `ni_rua` tool, normal coding sessions stay silent; if you want to prevent accidental robot auto-start from non-robot sessions, set `NI_RUA_AUTO_START=0` or install Claude Code with `--scope project` from a dedicated directory.

## Verification status

Transport verified with a scripted model on Claude Code 2.1.321/2.1.325, Codex 0.160.0 (host end-to-end), 0.167.0 (spike and tarball check) and 0.169.0 (host end-to-end), and OpenCode 2.2.7. These are the same versions as `verified_hosts` in [`agent-plugin/manifest.generated.json`](../../agent-plugin/manifest.generated.json) (generated from [`plugin_contract.py`](../../ni_rua/cortex/agents/plugin_contract.py) and loaded by `cli.mjs`), the list `doctor` calls verified. Operator wake verified headless on Claude Code 2.1.321 and OpenCode 2.2.7. One short real-model session per pinned host passed on 2026-11-15 ([Real-model sessions](#real-model-sessions)), and one short session drove the real SO-101 arm from Claude Code 2.1.325 ([Real SO-101 session](#real-so-101-session)); one task per host is too little to call any host supported.

Evidence, with what each proves:

- Python: `mamba run -n ni-rua python scripts/run_tests.py tests/cortex/external/test_external_config_and_session.py tests/cortex/external/test_external_workbench.py tests/packaging/npm/` covers the bracket, the gate, operator authority, manifest nulls, refusals, the endpoint (`Origin`, port, instructions) and the agreement of the three host entries with the registry.
- Node: `node --test agent-plugin/test/*.test.mjs`: installer against fake `claude` / `codex` / `opencode` executables in a temporary `HOME`, exact uninstall, upgrade, locking, `doctor`.
- Host end-to-end: `NI_RUA_HOST_E2E=1 NI_RUA_HOST_E2E_LONG=1 PYTHONPATH=tests mamba run -n ni-rua python -m unittest -v tests/cortex/external/test_external_host_e2e.py` ([`tests/cortex/external/test_external_host_e2e.py`](../../tests/cortex/external/test_external_host_e2e.py)): 5 of 5 OK in 563.8 s on macOS; results below. Its CPU support suite is [`tests/cortex/external/test_external_host_e2e_support.py`](../../tests/cortex/external/test_external_host_e2e_support.py). Live mode: [Real-model sessions](#real-model-sessions).
- Host binaries: `agent-plugin/test/tarball-check.sh` installs the packed tarball into isolated homes on the pinned Claude Code, Codex and OpenCode and on the user's versions (83 passed, 0 failed on all five binaries with Codex 0.167.0, run with the spike fixtures). It proves the install, the plugin and MCP entry, the skills and uninstall, not a real-model session.
- The spike measurements (entry syntax, `alwaysLoad` on HTTP, permission defaults, delivered lengths, images, long calls, restarts, wake) are in the report [Agent Plugin Spikes](reports/agent-plugin-spikes.md); each row is labelled run, binary or inference.

Not verified: interactive TUIs (Claude `/mcp` reconnect was driven through its stream-json control, Codex TUI recovery after a late gateway, OpenCode TUI reload, the wake in an idle interactive session), Claude Code 2.1.325 and Codex 0.169.0 beyond the short end-to-end run (no long call, no wake), the X5 catalogue through a gateway, calls longer than 332 s, Linux, real models beyond the one short session per pinned host, the real SO-101 arm beyond one short session ([Real SO-101 session](#real-so-101-session)).

### Per-host conformance results

Command: `NI_RUA_HOST_E2E=1 NI_RUA_HOST_E2E_LONG=1 PYTHONPATH=tests mamba run -n ni-rua python -m unittest -v tests/cortex/external/test_external_host_e2e.py`. Result: 5 of 5 OK, 563.8 s, on macOS, with the SO-101 scene double, a scripted fake model, and each host in its default permission mode with only the installer's settings. This is transport evidence with a scripted model, not a claim that any host is supported. Details and measurements: [Agent Plugin Spikes](reports/agent-plugin-spikes.md#4-host-conformance-end-to-end-suite).

| Host | Tools | Lengths | Image | Non-read-only calls | Long call | Origin sent | Store | `doctor` | Wake | Wall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Claude Code 2.1.321 | 19/19, full schema, none deferred | instructions present; briefing 60,492 = 60,492; all text whole | 5 identical (2 camera webp 640x480) | 5 executed | 332 s, receipt completed | no | ok | ok, 19 tools | ok, new request 10.0 s after exit | 346.5 s |
| Claude Code 2.1.325 | ok | ok | ok | ok | n/a (short run) | no | ok | n/a | n/a | 5.9 s |
| Codex 0.160.0 | ok (known schema projection, +38-character description suffix) | ok | ok | ok | 91 s, ok | no | ok | ok | n/a (Codex keeps `codex queue`) | 96.0 s |
| Codex 0.169.0 | ok (same) | ok | ok | ok | n/a | no | ok | n/a | n/a | 7.9 s |
| OpenCode 2.2.7 | ok | ok (60 kB briefing whole; depends on the installer's `tool_output`) | ok | ok | 92 s, ok | no | ok | ok | ok, 10.0 s | 107.1 s |

"Store ok" means: every `tool_call` row has a turn id; the final answer equals the `run_finish` summary with origin `run_finish_summary`; `agent.client_attached` holds the self-reported `clientInfo`, whose version matches the binary; and `summary.json` after shutdown has the external null fields. "Origin sent: no" means no host sent an `Origin` header to the endpoint, so the foreign-`Origin` refusal affects none of them.

The long call is a test-only mode: the scene double holds the worker's answer after the motion completed (`execute_hold_s` in [`tests/support/so101_scene.py`](../../tests/support/so101_scene.py)), because a slow motion would hit the 300 s `robot_action` `timeout_s` cap and the SO-101 `plan.max_age_s` of 120 s, and a hold placed before the motion made the 330 s receipt `batch_plan_expired_real_world_moved`. Not verified: interactive TUIs, real models (see [Real-model sessions](#real-model-sessions)), the X5 catalogue through a gateway, calls longer than 332 s, Linux.

### Real-model sessions

Command (opt-in, calls a charged provider): `NI_RUA_HOST_E2E_LIVE=1 NI_RUA_HOST_E2E_LIVE_PROFILE=<profile> NI_RUA_HOST_E2E_LIVE_CLAUDE_MODEL=<id> NI_RUA_HOST_E2E_LIVE_CODEX_MODEL=<id> NI_RUA_HOST_E2E_LIVE_OPENCODE_MODEL=<id> PYTHONPATH=tests mamba run -n ni-rua python -m unittest -v tests.cortex.external.test_external_host_e2e.ExternalHostLiveTest` ([`tests/cortex/external/test_external_host_e2e.py`](../../tests/cortex/external/test_external_host_e2e.py), "Live mode"). Setup is the same as the conformance run: an isolated home, the real installer, the real gateway over the SO-101 scene double (no arm), and each pinned host in its default permission mode with only the installer's settings. The host talks to a real provider and nothing is scripted. The gateway's `--prompt` is `LIVE_TASK` in [`tests/support/host_e2e.py`](../../tests/support/host_e2e.py): name the cube colour as a `CUBE_COLOUR:` line, move the end effector 5 mm up and confirm it from the receipt, finish with `run_finish`. The host's only user message is "Operate the robot through the ni_rua MCP tools.", which names neither `run_begin` nor the task.

Results on 2026-11-15, macOS, provider OpenCode Console, one session per host. The cube colour was random per host:

| Host | Model | `run_begin` first | `robot_observe` images | `robot_action` completed | `run_finish` | Colour | Requests | Session wall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Claude Code 2.1.321 | `claude-sonnet-5-5` (served: `claude-sonnet-5-5`) | yes; 0 calls before, 0 refusals | yes, 2 webp 640x480 | yes, measured dz 0.005 m | yes; summary = final answer, turn `completed` | purple, correct | 5 | 16.0 s |
| Codex 0.160.0 | `gpt-5.6-terra` (served id not reported by Codex) | yes; 0 NI-RUA calls before, 0 refusals | yes | yes, measured dz 0.005 m | yes; same | red, correct | 11 | 39.2 s |
| OpenCode 2.2.7 | `claude-haiku-5-5` (served id not reported by OpenCode) | yes; 0 calls before, 0 refusals | yes | yes (preview, then execute), measured dz 0.005 m | yes; same | yellow, correct | 7 | 21.2 s |

- **Tool sequences.**
  - Claude Code: `run_begin`, `robot_observe`, `task_plan`, `robot_action`, `run_finish`.
  - Codex: one shell command that read the installed `ni-rua-robot` `SKILL.md`, then `run_begin`, `robot_observe`, `task_plan`, `robot_action`, `task_plan`, `run_finish`.
  - OpenCode: `run_begin`, `robot_observe`, `task_plan`, `robot_action` (preview), `robot_action` (execute), `run_finish`.
- **Refusals, permissions, errors.** No host had an `external_briefing_required` refusal, a permission denial or a host error. Claude Code ran in `permissionMode` `default` and Codex with approval `never` and the `read-only` sandbox. No model tried a native tool that the default mode refuses.
- **Usage, as reported by the host.**
  - Claude Code: 1,022 output tokens, 121,833 cache-write and 419,231 cache-read input tokens; Claude Code's own estimate is $0.36.
  - Codex: 386,880 input tokens (337,793 cached) and 1,668 output tokens.
  - OpenCode: 1,427 output, 663 reasoning, 115,977 cache-write and 594,864 cache-read tokens; OpenCode reports $0.055.
- **The archived final answer.** It is the `run_finish` summary with its short ids (for example `obs_4`) resolved to full ids, as the archive keeps them.
- **First Codex attempt.** `gpt-5.3-codex` made 1 request and no tool call. It wrote that it would start with `run_begin`, then asked the user for the task and ended the turn, which ends `codex exec`. Codex 0.160.0 has no model metadata for that model and warned that it fell back to default metadata. The retry used `gpt-5.6-terra`, which the binary knows. One confounded sample is no evidence against the `initialize.instructions` wording, so the wording was not changed.
- **Claude Code first session.** The first Claude Code session (6 requests, 20.7 s, the same sequence) failed two checks of the test itself:
  - The final answer was compared byte for byte with the summary, without resolving short ids.
  - The colour pattern also matched the prose "cube colour from".

  Both checks were fixed, and the table shows the rerun.

This shows only that three pinned hosts, each with one model, carried one short observe-and-move task through the plugin by following `initialize.instructions` and the tool descriptions. It says nothing about manipulation quality, other models or longer sessions. Only Claude Code reports the served model id.

### Real SO-101 session

One session on the real SO-101 arm, 2026-11-15 (UTC 11:53), macOS. The setup:

- **Gateway.** `--agent-backend external`, run from this branch. The config is the usual SO-101 run config with the agent switched to `external`, the remote VGGT service removed, and the VGGT-dependent perception features (`external_perception`, `object_table`, `scene3d`, `vggt_height`) turned off. The deployment was freshly calibrated.
- **Host.** Claude Code 2.1.325 in an isolated config home, with the plugin installed by `ni-rua-plugin install claude` using its defaults.
- **Model.** `claude-sonnet-5-5` through OpenCode Console.
- **Prompt.** The only user message named the task: describe the table, move the gripper straight up 2 cm, verify it, and move it back. It did not name `run_begin`.

What happened:

- **Tool sequence.** `run_begin` (first call, no refusal), `robot_observe`, `task_plan`, `robot_action`, `robot_action`, `robot_action`, `run_finish`. The three `robot_action` calls were a preview of the up move, its execute, and the down move with `execute_if_clean`; the up move ran with `speed_preset` `slower`.
- **Up move.** Receipt `completed`, with a residual of 1.5 mm. The measured TCP z went from 0.0822 m to 0.1034 m, and the lateral drift stayed under 1 mm.
- **Down move.** Receipt `completed`, with a residual of 0.2 mm.
- **Turn.** `run_finish` closed the turn as `completed`. The model's scene description matched the table: three cubes, a screwdriver, a box and a laptop.
- **Cost.** 8 model turns, 33.7 s, about $0.43 by Claude Code's estimate.
- **Records.** The manifest has `agent_backend.kind = "external"` and `information_boundary = "mcp_boundary_only"`; `model_request_limit` and `no_progress_timeout_s` are null. `summary.json` has `served_models`, `token_usage` and `token_accounting` null.
- **Shutdown.** The gateway was stopped with SIGINT and exited cleanly.

This is one short session of small, slow motions. It proves the external path on hardware, not manipulation quality.

## Operator wake recipes

See [Experiment Completion Hook](experiment-completion-hook.md#waking-the-agent-that-launched-the-queue-per-host): Codex uses the existing hook; Claude Code and OpenCode run the queue as the agent's own background shell task (verified headless only).

## Deferred, not in this release

Discovery records, an `ni_rua_status` tool, description splitting, caller identity and a one-driver rule, benchmark-style external runs, `--wake-command`, PreCompact hooks and the hook route, a relay for external agents, evidence helpers in external mode, and an `ni-rua-library` console script. Longer real-model tasks and expanded real SO-101 validation remain ongoing.
