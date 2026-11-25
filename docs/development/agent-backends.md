# Agent Backends

NI-RUA drives the actor through a swappable agent runtime. `agent.backend` (config) / `--agent-backend` (CLI, one
run) selects one of four backends: three hosted ones, `codex` (default), `claude_code` and `opencode`, where NI-RUA
launches the agent, and `external`, where it does not and the user's own agent connects over MCP. All four implement
the same backend-neutral contract, so `Workbench`, `Harness`, the reports and the experiment library treat them
uniformly; the evaluation queue runs only the three hosted ones, and only `agent.*` sub-settings and the recorded
native differences below vary. `external` is a different kind of backend (no agent process, no model relay, no model
or token record, interactive runs only); this guide's per-backend sections describe the hosted three, and
[External backend](#external-backend-agentbackend--external) and the [Agent Plugin guide](agent-plugin.md) cover it. A backend's model
provider is a [provider profile](#provider-profiles) (one git-ignored file per provider and key); the web console
switches backend, profile, model and reasoning settings through the
[Agent bar](#console-agent-bar-live-settings-and-handoff), and a backend or profile change hands off to a successor
run. See the table below and [Installing the pinned binary](#installing-the-pinned-binary) for the install recipes, and
[Configuration](configuration.md#agent-backend-agent) for every `agent.*` field.

## Overview

| | Codex | Claude Code | OpenCode | External |
|---|---|---|---|---|
| Native runtime | `openai-codex` app-server | `claude-agent-sdk==0.2.180` (bundles CLI `2.1.321`) | `opencode` 2.2.7 platform binary (npm tarball `@opencode/cli-<target>@2.2.7`) | none: your own Claude Code / Codex / OpenCode, launched by you |
| Tool transport | native `dynamicTools` (in-process) | host-owned MCP over loopback HTTP | host-owned MCP over loopback HTTP | MCP over loopback HTTP on the fixed port `agent.external.mcp_port` (8790), no bearer |
| Model wire(s) | OpenAI Responses | Anthropic Messages | Anthropic Messages or OpenAI Responses | none: no relay, model requests do not pass through NI-RUA |
| Optional extra | built in | `pip install -e '.[claude]' -c requirements-claude.lock.txt` | none on PyPI: extract the npm tarball to `~/.local/share/ni-rua/opencode-2.2.7/opencode` ([install](#installing-the-pinned-binary)), absolute `executable` path; needs `rg` | the npm package `agent-plugin/` ([Agent Plugin](agent-plugin.md)); no Python extra |
| Native terminal attach ([details](#native-terminal-featuresterminal)) | yes (`codex resume --remote`) | no (the SDK session cannot be attached) | yes (`opencode --server <url> --session <id>`; not when sandboxed) | no (it is your own terminal) |
| Native evidence agent | yes (`CodexEvidenceAgent`) | no (relay client) | no (relay client) | no (refused: no relay) |
| Native memory / cross-turn | yes (Codex home) | yes (run-private `CLAUDE_CONFIG_DIR`) | no | whatever your agent has; unrecorded |
| Queue, benchmark, sandbox | yes | yes | yes | no: interactive-only, exploratory |

## Architecture

### `AgentSession` contract and capabilities

Every backend subclasses `AgentSession` ([`ni_rua/cortex/agents/base.py`](../../ni_rua/cortex/agents/base.py)): a
blocking, thread-safe, one-native-thread/one-active-turn object with the lifecycle `start()` / `run_turn()` /
`interrupt()` / `close()`, plain-string `state` (`not_started`, `idle`, `running`, `completed`, `interrupted`,
`failed`, `disconnected`, `closed`, `replay`), and completion observed through `turn_done`, `state`, and
`turn_result`. `run_turn()` returns at admission, never at completion. `cortex/agents/factory.py` (`create_session`,
`backend_requirements`, `agent_backend`) picks the subclass named by `agent.backend` and imports only that backend's
SDK, so a replay viewer of a Claude Code or OpenCode archive opens with no native runtime installed, and
`backend_requirements` refuses a launch configuration a backend cannot honour (missing runtime, incompatible
upstream wire, unsupported `model.effort`, …) with one actionable sentence before any run directory, robot, or
native process exists.

Optional features are declared as `AgentCapabilities` flags (`steer`, `native_requests`, `terminal_attach`, `fork`,
`deferred_tools`, `native_evidence_agent`, `sandbox`, `native_memories`, `live_effort`, `native_wakeup`,
`compaction_signal`, …) instead of callers probing the backend name; see the dataclass docstring for every flag's
meaning. `CodexSession` ([`ni_rua/cortex/session/codex.py`](../../ni_rua/cortex/agents/codex/session.py)) declares the
richest set (terminal attach, fork, deferred tools, native evidence agent, live effort, sandbox, native memories).
`ClaudeCodeSession` ([`ni_rua/cortex/agents/claude_code/session.py`](../../ni_rua/cortex/agents/claude_code/session.py))
declares `steer`, `native_wakeup` (R4.2 below), `sandbox`, `native_memories`, `turn_images`, `compaction_signal="native+relay"`, `context_delivery="delta"`.
`OpenCodeSession` ([`ni_rua/cortex/agents/opencode/session.py`](../../ni_rua/cortex/agents/opencode/session.py))
declares `steer`, `live_effort`, `native_wakeup` (a background subagent or shell can start work on its own, see below), `sandbox`, `terminal_attach`, `turn_images`, `compaction_signal="native+relay"`, `context_delivery="delta"`.

**Self-initiated native turns (Claude Code and OpenCode).** Claude Code can start a turn by itself while the host turn is terminal
or idle (a background shell finishing, a scheduled wake-up; `CronCreate`/`CronDelete`/`CronList`/`ScheduleWakeup`
stay enabled). The session adopts it as an NI-RUA turn (`agent.turn_started {origin: "native_wakeup"}`, a new turn
id, `turn_done` cleared, ending at the next native idle); a `run_turn()` issued while such a turn is running joins
it instead of raising `turn_already_active`. OpenCode 2.2.7 does the same for a background subagent's or shell's
completion report ([details](#opencode-turn-contract-subagents-and-background-work)); Codex never starts a turn unprompted.

### MCP tool transport

Claude Code and OpenCode both connect to one host-owned loopback MCP endpoint,
[`NI-RUAMcpServer`](../../ni_rua/cortex/agents/mcp_server.py) (streamable HTTP, the minimal in-repo JSON-RPC subset
both clients use — no `mcp` package dependency). `tools/list` publishes the full registry catalogue with the
session's presentation scrub, projecting each `inputSchema` to a portable form (top-level
`anyOf`/`oneOf`/`allOf`/`if`/`then`/`else` removed; nested ones untouched) because Claude Code silently drops an MCP
tool whose schema has one and the Anthropic API rejects it. `tools/call` is the only path from an MCP call to
`Harness.handle_tool`, through [`ToolDispatcher`](../../ni_rua/cortex/agents/dispatch.py) (ordered concurrent
dispatch, keyed off the registry's own `ToolSpec.parallel`: a parallel-safe call may overlap other parallel-safe
calls, but execution order always equals submission order for ordered calls — no name list in the transport).
[`McpSessionMixin`/`McpToolBridge`](../../ni_rua/cortex/agents/mcp_session.py) hold the turn helpers, relay wiring,
and shutdown order both MCP backends share.

**Order of one model step's calls.** A model step may emit several tool calls, and the order the model wrote them in
is the order they must take effect in. Arrival order at the endpoint is not that order: OpenCode 2.2.7 sends every
call of a step at once, each `tools/call` on its own connection, ordered (stateful) ones included, and each request is
served by its own handler thread. Measured on macOS under 8-way load with three ordered calls in one step: 4 of 96
steps executed as `[1, 3, 2]`, three with the requests logged in that order and one with the log in emitted order
(the inversion then happened between NI-RUA's own handler threads). The JSON-RPC ids stayed in emitted order every
time, so the inversion is in delivery, not in the client's call order. Claude Code sends an ordered call only after
the previous one was answered (0 of 40).

Host mode therefore submits calls in the model's emitted order, taken from the model response itself:

- The relay buffers every answer until it is validated, so before the first byte reaches the agent it reads the
  answer's tool calls in emitted order (`WireAdapter.emitted_tool_calls`: Anthropic `tool_use` blocks by content
  index, Responses `function_call` items by `output_index`) and hands them to the session
  (`ModelRequestProxy.on_tool_calls` -> `McpSessionMixin._model_tool_calls` -> `McpToolBridge.announce_model_step`).
  The bytes are forwarded unchanged; Codex installs no such callback (its dynamic tools arrive on one ordered stream).
- Each arriving call is correlated with its announced place
  ([`EmittedOrder`](../../ni_rua/cortex/agents/dispatch.py)), exactly or not at all: by the model's call id when the
  client sends it (Claude Code `_meta["claudecode/toolUseId"]`); otherwise (OpenCode sends no call id) by the
  client's session key (`_meta["ai.opencode/sessionID"]`, equal to the `x-opencode-session-id` of that session's
  model requests, child sessions included), the tool name and the JSON arguments within that session's newest step.
  Calls with the same tool and arguments are interchangeable. There is no match by tool name alone.
- A call without an exact place runs in arrival order, as before, and is recorded:
  `agent.tool_order_unmatched {tool, reason}` (observer data), `reason` one of `no_step_known` (nothing announced),
  `call_id_unknown` (an id that is no open call of a kept step: never emitted, replayed, or older than the 64 steps
  kept), `no_session_key` (neither id nor session key), `session_key_mismatch` (no step announced under the client's
  key), `arguments_differ` (the client sent other arguments than the model emitted) and `not_in_step`. On the pinned
  clients none of these occurs in a normal step (conformance item 6 asserts it), so the event is the sign that the
  correlation stopped working, for example after a client upgrade.
- A call is handed to the `ToolDispatcher` only after every call emitted before it in the step was handed over or
  refused. The dispatcher's rule then applies unchanged, so parallel-safe calls still overlap and an ordered call
  runs after everything emitted before it. The hold comes before the transcript row, so rows start in execution
  order. After a hold the admission is checked again: if the turn ended, a new turn began or the session is closing,
  the call is refused as `native_tool_call_stale_connection`, exactly like a call arriving then; nothing runs.
- The wait is bounded by a documented constant, `EMITTED_ORDER_WAIT_S` = 5 s (not a setting). Rationale: the three
  `tools/call` requests of one OpenCode 2.2.7 step were logged 0.6 to 8.3 ms apart, at most 10 ms apart under 8-way
  load, so 5 s never decides between calls in flight. It only limits how long a call waits for an earlier sibling
  the client never sends (it refused the call itself: unknown tool, arguments its own schema check rejected, an
  interrupted step). That sibling is written off after 5 s, once for the whole step, and the calls that did arrive
  run in emitted order. The call that waited is recorded as `agent.tool_order_wait_expired {tool, position,
  step_size, matched_by, waited_s, missing_positions}` (observer data); expiry is never a refusal, and if the
  missing call arrives later it runs on arrival. Worst case: 5 s in that situation; 10 s only if, after the
  write-off, an earlier call that did arrive is stuck in its own bookkeeping (Store write, session lock) for
  another 5 s. That second window cannot be removed without an unbounded wait or letting a call overtake an arrived
  earlier one.

Delivery-uncertainty handling, typed results, call ids and transcript rows are unchanged; the hold happens on the
call's own MCP handler thread before the dispatcher, with the SSE keepalive running. Regressions:
[`tests/test_agent_dispatch.py`](../../tests/cortex/agents/test_agent_contracts_fast.py) (`EmittedOrderTests`),
[`tests/test_agent_mcp_session.py`](../../tests/cortex/agents/test_agent_mcp.py) (`EmittedOrderBridgeTests`: real endpoint,
inverted HTTP arrival), [`tests/test_model_proxy_continuity.py`](../../tests/cortex/relay/test_model_proxy_continuity_and_guard.py)
(`EmittedToolCallsRelayTest`), and conformance item 6 against the real pinned binaries
([`tests/support/agent_conformance.py`](../../tests/support/agent_conformance.py): `test_ordering_barrier`,
`test_ordered_calls_run_in_emitted_order`; for OpenCode the first call of the step is delivered last).

The `external` backend has no relay, so it has no emitted order: see
[Agent Plugin](agent-plugin.md#order-of-one-steps-calls). See [MCP tool catalogue](#mcp-tool-catalogue-claude-code-and-opencode) below for the recorded delivery differences from the Codex native path.

The `external` backend reuses the same endpoint class with a fixed port, no bearer (a foreign `Origin` is refused) and an `initialize.instructions` text, and swaps the host-mode admission for the `run_begin` / `run_finish` bracket ([External backend](#external-backend-agentbackend--external)).

### Relay wire adapters and client profiles

One relay, [`ModelRequestProxy`](../../ni_rua/cortex/relay/server.py), carries every generic policy (the
180-call budget, retries, the recovery gate, cancellation, evidence attribution, activity clocks, the image-trim/reuse
state machine, egress scrub). Format-specific behaviour lives in a `WireAdapter`
([`cortex/session/wire/`](../../ni_rua/cortex/session/wire/)): `openai_responses.py` (Codex and OpenCode),
`anthropic_messages.py` (Claude Code and OpenCode); `openai_chat` has no relay adapter yet (OpenCode refuses to start
on it). A `ClientProfile` ([`wire/base.py`](../../ni_rua/cortex/session/wire/base.py)) selects per-client delivery
behaviour: `early_stream` (Codex: stream opens immediately; OpenCode/Claude Code on `client_loop_guard`: deferred
delivery, no byte sent until the attempt is validated) and `client_loop_guard` (see
[Relay client-loop guards](#relay-client-loop-guards-claude-code-and-opencode) below). The Codex profile's bytes are
pinned by golden fixtures (`test_wire_golden.py`) and never touched by the guards.

### Continuity / compaction detection

With `context_delivery="delta"` (the default across all four backends; `claude_code`, `opencode`, and `external` also support explicitly switching back to `"full"`) the Harness may leave out of a main-chain tool result what it already delivered in the current image-context epoch (subagent and overlapping concurrent calls run in `isolated` full-delivery mode). Because Claude Code and OpenCode manage their own history (auto-compaction, pruning, client rewrites can drop old context), the relay runs a synchronous [`ContinuityTracker`](../../ni_rua/cortex/relay/continuity.py) on every generation request: Claude Code marks subagent/fork chains via `x-claude-code-agent-id` (and falls back to `(thread_key, system_key)`), while OpenCode isolates child sessions by `x-opencode-session-id` with `system_key=None` on the main session so simultaneous compaction and system updates never stall tracking; each request gets a `call_purpose` (`main`/`helper`/`subagent`), and a new history is compared against the last one delivered to the agent. A discontinuity (`history_shortened`, `history_rewritten`,
`images_removed`, `system_changed`) calls the session's `on_context_reset(reason)`, which schedules the Harness's
decision-context reminder on the next actor tool response — the same mechanism Codex drives from its own native
compaction event. Native hints (Claude Code's `PreCompact` hook / `compact_boundary`; OpenCode's
`session.compaction.started` event and its `compaction` message) are extra triggers, never a substitute.

### Transcript, usage, and evidence

[`TranscriptWriter`](../../ni_rua/cortex/agents/transcript.py) writes every backend's rows into the existing
`codex_item` state kind with Codex's own field shapes (`backend`, `native_id`, `message_id`, `parent_item_id` stored
alongside), so `Store.completed_final_texts`, library replay, and the web readers cannot tell backends apart;
final-answer stamping (last top-level assistant message of a `completed` turn with text and no tool call) is
computed the same way for every backend. [`RelayEvidenceClient`](../../ni_rua/cortex/agents/evidence.py)
(`agent.evidence.client="relay"`, the default for every backend except Codex) answers one evidence packet with one
fresh, streamed, tool-free request through the run's own relay — no agent process, home directory, history, or
memory — offering the same constructor keywords and `review_evidence()` call as `CodexEvidenceAgent` so
`VisualReviewer` and `self_briefing.generate` use either unchanged.

## Provider profiles

A provider profile is how a run names its model provider. It is one git-ignored JSON file per provider and key,
`<profiles_dir>/<id>.local.json`, in NI-RUA's own format (schema `ni_rua.provider_profile.v1`,
[`ni_rua/cortex/agents/profiles.py`](../../ni_rua/cortex/agents/profiles.py)). The profile holds the key and the wires
it serves. One key can therefore drive every backend whose wire the profile has: Codex runs on `openai_responses`,
Claude Code on `anthropic_messages`, and OpenCode on either (`factory.BACKEND_WIRES`). The older official-format
sources (`claude_settings`, `opencode_config`, `codex_home`, `explicit`) are still read for existing configurations,
archives and resume ([Configuration per backend](#configuration-per-backend)). The console lists and creates profiles
only.

### Profile file and directory

The directory is `agent.profiles_dir`. Null (the default) means `<checkout>/configs/profiles`, which is ignored
through `configs/profiles/` in `.gitignore`. A relative path is resolved next to the config file that sets it, so
every worktree's run config can point at the main checkout's directory with one line. The id is the file stem,
`[a-z0-9][a-z0-9_-]{0,39}`. Files are written atomically with mode 0600, and the directory is created with mode 0700.
The file below is
[`configs/profile.example.json`](../../configs/profile.example.json); every value is a placeholder, so never commit
a real key:

```json
{
  "schema": "ni_rua.provider_profile.v1",
  "name": "Example Gateway",
  "credential": {"api_key": "sk-REPLACE-WITH-YOUR-KEY"},
  "wires": {
    "anthropic_messages": {"base_url": "https://gateway.example/gateway", "auth_scheme": "bearer"},
    "openai_responses": {"base_url": "https://gateway.example/gateway/v1", "auth_scheme": "bearer"}
  },
  "compat": {"tool_result_images": "user_blocks", "tool_schema_max_depth": 13},
  "backends": {
    "claude_code": {"env": {"CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0"}},
    "opencode": {"thinking_display": null}
  },
  "defaults": {"model": "example-model", "effort": "high"}
}
```

| Field | Meaning, default |
|---|---|
| `schema` | Must be `ni_rua.provider_profile.v1`. |
| `name` | Display name in the console, 1–60 characters. Default: the id. |
| `credential.api_key` | The provider key (printable ASCII, no spaces). Only the gateway reads it, and sends it upstream in the header that the wire's `auth_scheme` names. It is never shown again, never archived and never given to the agent. When it is absent, the launch gate refuses the profile with `profile_missing_key`. |
| `wires.<wire>` | One entry per wire the provider serves: `anthropic_messages` and/or `openai_responses`. At least one. |
| `wires.<wire>.base_url` | Upstream base URL in the relay's convention. Anthropic Messages has no `/v1` (the relay appends `/v1/messages`); OpenAI Responses includes `/v1` (the relay appends `/responses`). |
| `wires.<wire>.auth_scheme` | `bearer` (the default, `Authorization: Bearer <key>`) or `x-api-key`. See [one auth scheme per wire](#one-auth-scheme-per-wire). |
| `wires.<wire>.models_url` | Optional explicit model listing URL. Default: derived from `base_url` (`model_catalogue.models_url`). |
| `compat.tool_result_images`, `compat.tool_schema_max_depth` | The provider compatibility opt-ins ([below](#configuration-per-backend)), with the same meaning, defaults and bounds as the `UpstreamConfig` fields. They apply to Claude Code and OpenCode only. |
| `backends.claude_code.env` | Variables forwarded to the Claude Code CLI, for example `CLAUDE_CODE_THINKING_DISPLAY_UPDATES=0` for a gateway that refuses `thinking.display`, or `CLAUDE_CODE_MAX_OUTPUT_TOKENS`. They are validated like a settings file's `env`, so reserved names are refused. Default `{}`. |
| `backends.opencode.thinking_display` | `null` (the default), `summarized` or `omitted`. Used as `agent.opencode.thinking_display` when the run does not set it. It applies on the `anthropic_messages` wire only. |
| `defaults.model`, `defaults.effort` | Pre-selected in the console when the profile is chosen. Optional. |

The same table, from `profiles.FIELD_HELP`, generates the module docstring and the console form's help text.
Validation errors name the file and the field, never a value (`ValueError("profile_<code>:<file>: ...")`).

There are three ways to create profiles:

- Copy `configs/profile.example.json` to `configs/profiles/<id>.local.json` and edit it.
- In the console, open ⚙ → "New profile" and paste the key. The key field is write-only: an empty key field on edit
  keeps the stored key, and no endpoint ever returns it. "Remove" renames the file to `<id>.local.json.removed-<UTC>`
  and is refused for a profile the running configuration uses.
- Run the one-shot import of the official-format key files that predate profiles. It reads Claude Code settings
  files `configs/claude*.local.json`, OpenCode configs `configs/opencode*.local.json` (V1 and V2 shapes), and Codex
  provider homes `~/.config/ni-rua/providers/<name>/`. Files with the same key are merged into one profile with
  several wires when nothing conflicts. The source files are never changed, and an existing profile is never
  overwritten:

```sh
mamba run -n ni-rua python -m ni_rua.cortex.agents.profiles import --dry-run   # print the plan, write nothing
mamba run -n ni-rua python -m ni_rua.cortex.agents.profiles import             # write the profiles
mamba run -n ni-rua python -m ni_rua.cortex.agents.profiles list               # public view, never a key
```

`import` also takes `--source-dir`, `--codex-homes` (`''` = none), `--profiles-dir` and `--config` (a run
configuration read for compat opt-ins; repeatable). A ChatGPT-login Codex home has no API key and stays
`model.codex_home`.

### Selecting a profile in a run configuration

`UpstreamConfig.source = "profile"`, with `settings` set to the profile path, selects a profile for each backend's
actor upstream:

```json
// configs/ni-rua.local.json (relative paths are resolved next to this file; a worktree's run config can use the
// main checkout's absolute paths instead)
{"agent": {"backend": "claude_code",
           "profiles_dir": "profiles",
           "claude_code": {"upstream": {"source": "profile", "settings": "profiles/gateway.local.json"}},
           "opencode": {"upstream": {"source": "profile", "settings": "profiles/gateway.local.json",
                                     "wire_api": "anthropic_messages"}},
           "codex_upstream": {"source": "profile", "settings": "profiles/gateway.local.json"}}}
```

Each backend reads only its own entry, so a configuration can name the upstream of every backend it may run on.

- The profile owns the upstream's non-secret fields. `Config.load` and `profiles.apply_profile` write them from the
  file: `wire_api` (the profile's wire for that backend, or the one you name when the profile serves two, as with
  OpenCode), `base_url`, `auth_scheme`, `provider` (= the profile id), `tool_result_images` and
  `tool_schema_max_depth`; `api_key_env` stays null. `effective_config.json` and the manifest therefore record the
  concrete endpoint, never the key.
- The launch gate (`factory.backend_requirements` through `profiles.resolve_profile_upstream`) refuses a profile that
  does not load (`profile_<code>`), has no key (`profile_missing_key`), serves no wire the backend runs on
  (`profile_no_common_wire`), or no longer matches the configuration (`profile_upstream_stale`, when the file changed
  after the configuration was built).
- `agent.codex_upstream` is Codex's upstream. Null (the default) keeps `model.codex_home` unchanged and is left out of
  `to_dict()`, so existing Codex manifests stay byte-identical. It accepts only `source: "profile"` on
  `openai_responses`. The generated private `config.toml` matches the Codex-home one except for the provider label and
  base URL. The field is flat because the evaluation queue reads `getattr(config.agent, backend).upstream`.
- The agent only ever sees the relay's placeholder key; the relay adds the profile's credential upstream. The
  manifest records `agent_backend.upstream_profile {id, name, wire_api, wires}` (names only). For Claude Code it also
  records the names of the forwarded variables (`native_settings.settings_env`), never their values.

#### One auth scheme per wire

Each wire carries its own `auth_scheme`, because the relay adapter of each wire forwards a fixed set of credential
headers (`WireAdapter.auth_header_names`):

- The Anthropic Messages adapter forwards `x-api-key` and `Authorization`, so it accepts `x-api-key` and `bearer`.
- The OpenAI Responses adapter forwards only `Authorization`, so it accepts `bearer` only.

An `openai_responses` wire with `auth_scheme: "x-api-key"` would pass a naive check and then fail at the first
request. The matrix therefore marks it incompatible (`profile_auth_scheme_unsupported`), and the gate refuses it at
launch (`profile_auth_scheme_unsupported:<file>`; Codex: `codex_profile_auth_scheme_unsupported`). A provider such as
the OpenCode Console uses `x-api-key` on its Anthropic path and `bearer` on its Responses path: write each wire's own
scheme.

### Model catalogue cache

There is one model listing per (profile, wire). It is read with the profile's key at `models_url`, or at the URL
derived from `base_url` (`model_catalogue.fetch_upstream_catalogue`: OpenAI-style, Anthropic-style and models.dev
shapes; at most 10 pages of 5 s each).

- **Where it lives.** The listing is cached in memory for the whole gateway process and on disk at
  `<profiles_dir>/.cache/<id>.<wire>.json`. The disk copy holds status, times, the profile version and the models,
  never a key, header or URL, and the directory has its own `.gitignore` of `*`. The console and the running session
  share one process-wide instance (`model_listing.shared_catalogue_cache`), so a successor run keeps it, and other
  processes read the disk copy.
- **Freshness.** An answer is fresh for 600 s, failures included. A changed profile file (`st_mtime_ns`) invalidates
  its entries, and concurrent reads of one key share a single request. A failed refresh returns the last good list
  marked `stale: true`.
- **States.** `ok`, `truncated` (page cap), `unavailable` (with `reason` and `failure`: `key_refused`, `not_found`,
  `rate_limited`, `request_refused`, `provider_error`, `timeout`, `network`, `invalid_listing`, `incomplete`,
  `credential_unavailable`) and `not_read`. The console shows `loading` while the first read runs.
- **Refresh.** `GET /v1/console/profiles/{id}/models?wire=<wire>&refresh=1`. On `openai_responses`, the pinned
  native Codex catalogue (`codex_models.json`) fills fields a listed model does not report. Listing claims always win,
  and an unknown field is never drawn as supported.
- **Parsed claims.** Each listed model carries `reasoning` / `reasoning_efforts` (and toggle, default and budget
  claims) parsed from `capabilities.reasoning`.
- **Manual entry.** Typing a model id never depends on the listing. An id that is not listed is applied with the note
  "Not in provider list; determined by provider".

### Reasoning options and their scopes

The server generates the console's reasoning controls (`reasoning_options.reasoning_options`). No per-model table
exists anywhere:

| Control | Choices | Applies |
|---|---|---|
| Effort | Backend choices: Claude Code `low`..`max`; OpenCode `EFFORT_CHOICES[wire]`; Codex every `model.effort`. Each choice also carries `model: true/false/null` from the listing's `reasoning_efforts` (Codex: `supported_reasoning_levels` of `codex_models.json`). | `next_turn` where the backend has `live_effort` (Codex, OpenCode); `next_session` on Claude Code, which needs a handoff. |
| Speed (`service_tier`) | `default`, `priority` and `ultrafast`, plus the relay's per-model probes (`served_tier` = what the provider reported serving). Codex only (`capabilities.service_tier`). | `next_model_request`, also mid-turn. |
| `thinking_display` | `null`, `summarized`, `omitted`. OpenCode on `anthropic_messages` only. | `next_session` (handoff). |

The flags combine as follows:

- A choice with `backend: false` is disabled.
- A choice with `backend: true, model: false` stays selectable and is marked "Model support not declared". The provider judges
  it, and nothing is substituted.
- The last applied `{effort, service_tier}` per `<profile>/<backend>/<model>` is remembered in
  `<profiles_dir>/presets.json`, which holds no secret. Choosing that model offers the preset again; applying it is
  still an explicit click.

## Console Agent bar, live settings and handoff

### The Agent bar

The console's Agent panel starts with the Agent bar, built from `web/agent-settings-model.js`,
`agent-settings-view.js` and `ui-combobox.js`. It has chips for backend, profile, model and effort, plus speed on Codex
and ⚙. They open anchored popovers on desktop and bottom sheets on phones.

- The **model** combobox lists the profile's catalogue with badges (image input, tools, context window, efforts).
  Rows a backend cannot use stay visible, disabled, with their reason shown as secondary text. Free text is always
  accepted.
- A **live** change (model; effort on Codex and OpenCode; speed) is applied with one `apply_agent_settings` and is
  shown optimistically until its receipt arrives.
- A **session-scoped** change (backend, profile, wire, `thinking_display`, effort on Claude Code) opens the
  "New session required" tray instead. The tray shows the exact diff and the server's dry run of the exact selection
  (`POST /v1/console/agent-options/check`). "Switch and create new run" is enabled only when that dry run passes and
  `handoff.allowed` is true; otherwise the reason is shown.
- After confirmation, a progress overlay follows the handoff ("Finalizing current run" → "Starting new run" → "Ready"). It survives the
  listener restart and ends on the successor's run id, or on a fallback banner that names the reason.
- Every write takes the writer lease first. When another window holds the lease, the bar is read-only.
- On a gateway without these endpoints (`agent-options` 404), the bar falls back to `set_model` / `set_effort` /
  `set_service_tier`.
- With `agent.backend="external"` the bar is a read-only "External agent" card: the MCP endpoint, the client's self-reported `clientInfo` name and version, and the last call time, or "waiting for an agent". Every write command is refused (`<command>_unsupported_by_backend`) and `handoff_run` neither starts nor leaves an external run.

### Endpoints

All read endpoints need no lease and send `Cache-Control: no-store`. No endpoint ever returns a key.

| Endpoint | Returns |
|---|---|
| `GET /v1/console/agent-options[?refresh=1]` | `{current, backends, profiles, matrix, reasoning, catalogues, handoff, served, presets, profile_schema, profiles_dir}`. `matrix` has one cell per (backend, profile) with the first failing reason: `backend_unavailable:<problem>`, `profile_invalid:<code>`, `no_common_wire`, `profile_missing_key`, `profile_auth_scheme_unsupported`, or `gate:<code>` (the launch gate's sentence verbatim). The first call runs each CLI's `--version` once, about 3 s with real profiles. `handoff` is `{allowed, blockers: [{code, message}], disable_torque_on_disconnect, last}`. |
| `GET /v1/console/profiles/{id}/models?wire=<wire>&refresh=0\|1` | The cached or freshly read listing: `{status, reason, failure, fetched_at, age_s, stale, models}`. |
| `POST /v1/console/agent-options/check` | Body `{backend, profile, wire_api, model, effort, thinking_display?}`. A dry run of the exact selection with no side effect: `apply_profile` plus the launch gate, then `handoff_run`'s selection refusal `price_card_mismatch`. Answers `{ok, reason, message, wire_api, reasoning, handoff}`. `handoff` is the blocker view for this selection: the run's blockers plus the selection's, each `{code, message, detail?}`. The provider key check, which reads the listing, runs only at `handoff_run`. |
| `POST /v1/console/profiles` | Body `{client_id, writer_token, id, document, keep_key}`, or `{..., id, api_key}` to replace only the key. Needs the lease and the same origin. It **never passes through `/v1/control`**, whose bodies are archived, and the body is never stored. Returns the `ProfileInfo` (`key_present`, never the key). Event `operator.profile_saved {id, wires, compat, key_changed, created}`. |
| `POST /v1/console/profiles/{id}/remove` | Same checks. Renames the file and is refused while the profile is in use. Event `operator.profile_removed`. |

### `apply_agent_settings`

`POST /v1/control {command: "apply_agent_settings", model?, effort?, service_tier?}` needs the lease. It applies the
live fields in the order model, effort, tier, through the same code paths and events as `set_model`, `set_effort` and
`set_service_tier`. Those three commands stay available.

- **Receipt.** `{status: applied|partial|rejected|unknown, results: {<field>: {status: applied|rejected|unsupported|
  skipped|unknown, ...}}}`. The first field that is not applied stops the rest (`skipped`,
  `previous_setting_not_applied`). A setter whose outcome is unknown answers HTTP 409 and marks the run `unknown`,
  exactly as `set_*` does.
- **Session-scoped differences.** If the request differs from the running session in a session-scoped field
  (backend, profile, wire_api, thinking_display, or effort on Claude Code), nothing is applied. The answer is
  `status: "handoff_required"` with the exact `diff` (`{field: {current, requested}}`) and `session_fields`.
- **Refusals** (HTTP 400): `apply_agent_settings_requires_a_setting`, `invalid_agent_backend`,
  `profile_invalid_id`, `invalid_wire_api` and `invalid_thinking_display`. The per-field refusals are the `set_*`
  ones, for example `invalid_model` and `set_model_conflicts_with_price_card`.
- **Events.** `operator.agent_settings_applied {request_id, status, results}`. The three `operator.*_changed` events
  now also carry `backend`, `profile` and `wire_api`.

### `handoff_run`

`POST /v1/control {command: "handoff_run", backend, profile, wire_api, model, effort, thinking_display?, prompt?}`
needs the lease ([`ni_rua/apps/gateway/extensions/handoff.py`](../../ni_rua/apps/gateway/extensions/handoff.py)). There is no backend switch
inside a run: one run directory is one agent session and one comparable condition. A handoff finalises the current
run and starts a successor run in the same gateway process, on the same port, with the browser page left open.

1. **Gate, with nothing changed yet.** A refusal is HTTP 400 `{error: "<code>: <sentence>; nothing was changed"}`.
   - Run blockers (`handoff.blockers`): `replay_is_read_only`, `benchmark_run_owned_by_queue` (benchmark or
     `exit_after_turn`), `run_starting`, `agent_turn_running`, `action_in_progress`, `program_active_or_unresolved`,
     `unknown_unacknowledged`, `torque_not_on`, `handoff_in_progress`, `handoff_not_available` (no run supervisor).
   - Selection refusals: `handoff_requires_backend` / `_profile`, `invalid_agent_backend`, `profile_invalid_id`,
     `invalid_wire_api`, `invalid_model`, `invalid_effort`, `invalid_thinking_display`, `invalid_task_prompt`,
     `profile_not_found`, `profile_invalid:<code>`, and the dry run's reasons (`no_common_wire`, `wire_required`,
     `wire_not_in_profile`, `profile_missing_key`, `config_invalid:<code>`, `backend_unavailable`, `gate:<code>`).
   - `price_card_mismatch`: the run has a price card without a rate for the successor's provider and model.
   - `handoff_profile_key_refused`: the profile's listing, read now, answers HTTP 401/403. A listing that is merely
     unavailable does not block.
2. **Accept.** A fresh run directory is allocated, `handoff/<request_id>` and the event `run.handoff_requested
   {request_id, successor_run_id, from, to, prompt_source, carried}` are written, and motion admission closes. The
   receipt is `{status: "handoff_accepted", successor_run_id, predecessor_run_id}`. Later mutating commands on the
   closing run answer `handoff_in_progress`; safety commands stay available. The same `request_id` replays the
   receipt, also from the successor.
3. **The current run is finalised exactly like a clean shutdown.** The robot worker closes through its normal close
   op, and the handoff commands no motion. Torque follows the deployment's `disable_torque_on_disconnect`.
   `summary.json` gets `status: "stopped"`, `successor_run_id` and `successor_handoff {..., status: handed_off}`.
4. **The successor starts** from the run's own configuration with `apply_profile` applied. Its first prompt is the
   operator's `prompt`, or by default an observe-only prompt that names nothing about the handoff, profile or
   provider. The operator's speed cap is carried over: it is re-applied through `set_runtime_limits` before any agent
   turn, with event origin `handoff`. Nothing else is carried (no observation, plan, history or lease), and the new
   agent learns nothing about the old run. `manifest.json`, `summary.json` and `GET /v1/run` carry `handoff:
   {predecessor_run_id, request_id, requested_at, fallback, carried, prompt_source}`, and the event
   `run.handoff_started` is written. The remote services stay up and are not restarted.
5. **Fallback.** A successor that cannot start, or cannot even be built, is finalised as `failed`. ONE fallback run
   then starts with the previous configuration and the model and effort it was running, and records the event
   `run.handoff_fallback {failed_run_id, reason}`. `GET /v1/run` `handoff.fallback.reason` and agent-options
   `handoff.last` name the reason. If the fallback fails too, the gateway exits as on any failed start. A successor
   that started and fails later never triggers a fallback.

Directory outcomes:

- After a shutdown signal, an allocated successor that never started is finalised as `cancelled`
  (`gateway_shutdown_before_start`).
- A successor stopped during its own start is `stopped`, with no fallback.
- When the predecessor's cleanup cannot be confirmed (`final_status` unknown, for example a robot worker whose exit
  is unconfirmed), no successor starts, because a second robot worker could disturb the arm. The allocated directory
  is finalised as `cancelled` (`predecessor_cleanup_unconfirmed`).

### Records (observer data, never model context)

| Record | Content |
|---|---|
| `run.turn_settings` (event) | At every host turn admission (`origin`: `startup`, `prompt`, `benchmark_continuation`): `{turn_id, backend, profile, provider, wire_api, model, effort, service_tier, thinking_display, revision, served_models_at_admission}`. |
| `runtime/agent_settings` (state) | The current tuple plus `revision`, `changed_at` (UTC), `turn_count`, `mixed`. |
| `summary.agent_settings` | `{initial, final, changes, turns, mixed}`. `mixed: true` when the turns ran with different (model, effort, tier) tuples, or when a change that reached the running session came after the first turn. A comparison must not treat a mixed run as one condition. |
| Handoff | Old run: `run.handoff_requested`, `handoff/<request_id>`, `summary.successor_run_id` / `successor_handoff`. New run: `manifest.handoff`, `summary.handoff`, `run.handoff_started` or `run.handoff_fallback`. Outcome directories: `failed` or `cancelled`, as above. |
| Served model | The source of truth is unchanged: per-request `served_model` and `runtime/model_requests.served_models`. The bar shows requested against last served. |

### Operator guidance

- Put each provider and key into one profile once (import, or "New profile"). Point every worktree's run config at
  the same directory with `agent.profiles_dir`.
- Switch models within a run with the model chip. Switch backend or provider with a handoff, because the archive
  stays one condition per run. A run whose model or effort changed mid-run is `mixed`; do not use it as a
  single-condition sample.
- Handoff needs an idle run: no turn, action or program, unknowns acknowledged, torque on. With
  `disable_torque_on_disconnect: false` the arm holds its pose across the swap (the tray says so). With `true`,
  support the arm first.
- A benchmark or queue run never hands off; the queue owns those runs.
- Each run prints `gateway_process pid=<pid> python=<exe> ni_rua=<ni_rua/__init__.py>` after its `Workbench http://...`
  line, and the SO-101 worker logs `so101_worker_start ... ni_rua=...` in `private/robot/robot-worker.log`. Use them
  to check which checkout served a run, and that a successor ran in the same process.
- Accepted residuals (security is secondary): the key travels from the browser to the gateway over loopback HTTP;
  any local process that can reach the port and take the lease can write a profile; key files are plain text with
  mode 0600.

Verified live (`test_live_provider_e2e`, handoff scenario, rendered SO-101 scene;
[Validation](validation.md#live-provider-end-to-end)). The chain covered Claude Code
`claude-haiku-5-5`, then `apply_agent_settings` `claude-sonnet-5-5`, then OpenCode `deepseek-flash`,
then Codex `gpt-6.1-terra`. Every turn observed with images, made its 5 mm move and named the cube colour.
Acceptance to successor ready took 3.7 s and 2.3 s. On the real SO-101, Claude Code handed off to Codex
in 7.9 s with torque held ([SO-101 integration](so101-integration.md#64-agent-bar-model-switching-and-handoff)).

## Configuration per backend

The sections below describe the official-format credential sources that predate profiles (`claude_settings`,
`opencode_config`, `codex_home`, `explicit`). They stay readable for existing configurations, archives and resume. The
provider facts, opt-ins and measured limits apply to profiles unchanged: a profile's `compat` section and
`backends.claude_code.env` carry the same switches.

**Claude Code key in the official Claude format.** Instead of an environment variable, point the upstream at a
project-local Claude Code settings file (git-ignored `configs/claude.local.json`):

```json
// configs/claude.local.json  (same "env" block as ~/.claude/settings.json)
{"env": {"ANTHROPIC_BASE_URL": "https://relay.example", "ANTHROPIC_AUTH_TOKEN": "sk-..."}}
// configs/ni-rua.local.json
{"agent": {"claude_code": {"upstream": {"source": "claude_settings", "settings": "claude.local.json"}}}}
```

The gateway reads the endpoint and key from that file (`ANTHROPIC_AUTH_TOKEN` → `Authorization: Bearer`,
`ANTHROPIC_API_KEY` → `x-api-key`); only the file path is archived with the run, never the key, and the agent only
sees a local placeholder key (same model as `model.codex_home` for Codex). `ANTHROPIC_BASE_URL` is the root the relay
appends `/v1/messages` to, so it has no trailing `/v1` (the Claude Code convention).

Every other entry of the file's `env` object is forwarded to the CLI environment unchanged, as Claude Code itself
applies the `env` block of `settings.json` (`read_claude_settings_env` in
[`claude_code/options.py`](../../ni_rua/cortex/agents/claude_code/options.py)). Provider quirks therefore stay in the
provider's own file:

- `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN` and `ANTHROPIC_API_KEY` are consumed by the host relay and never
  forwarded.
- A variable NI-RUA sets itself in the CLI environment (`NI_RUA_OWNED_ENV`: `CLAUDE_CONFIG_DIR`, the model variables
  and `ANTHROPIC_MODEL`, the relay/MCP/telemetry/signalling switches, `CLAUDE_STREAM_IDLE_TIMEOUT_MS`,
  `CLAUDE_CODE_TMPDIR`, `XDG_RUNTIME_DIR`, `CLAUDE_CODE_AUTO_COMPACT_WINDOW`, `CLAUDE_CODE_DISABLE_AUTO_MEMORY`,
  `IS_SANDBOX`, `PATH`/`HOME`/`TMPDIR`/`LANG`, the process marker and the SDK spawn variables) is refused at launch
  with `claude_settings_env_reserved:<file>: <NAME> ...`; it is never dropped silently or allowed to override NI-RUA's
  value. Values must be strings (`claude_settings_env_not_string`).
- The forwarded variables also reach the `env -i` launcher and, with `features.sandbox_enabled`, the sandbox namespace.
- Provenance records only their names (`manifest.agent_backend.native_settings.settings_env`), never the values (a
  value may be a header secret); none of it enters model context.

Example: a multi-model gateway (DeepSeek, Kimi, GLM, Qwen, ... behind one Anthropic Messages endpoint) that rejects
Claude Code's `thinking: {"type": "adaptive", "display": "updates"}` with HTTP 400
`PROVIDER_THINKING_DISPLAY_UNSUPPORTED` on every request (the turn fails on its first model call). The CLI's own switch
`CLAUDE_CODE_THINKING_DISPLAY_UPDATES=0` makes it send no `display` field:

```json
// configs/claude.local.json
{"env": {"ANTHROPIC_BASE_URL": "https://gateway.example/gateway",
         "ANTHROPIC_AUTH_TOKEN": "sk-...",
         "CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0"}}
```

A model whose provider output limit is below the CLI's own `max_tokens` (32000 for a non-Claude model id) can be
given `CLAUDE_CODE_MAX_OUTPUT_TOKENS` in the same block; NI-RUA never clamps it on its own. Measured: `kimi-k2.5`
(provider `max_output_tokens` 16384) refuses the CLI's `max_tokens: 32000` with HTTP 400 `OUTPUT_UPPER_BOUND_EXCEEDED`
(the message is in the relay evidence, see below); with `"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "16384"` the CLI sends 16384
and the turn completes. The variable applies to every model of the run, including models switched to later. The
self-briefing helper (`RelayEvidenceClient`, `max_tokens` 16000, structured output, initial camera image in the user
message) completed on `kimi-k2.6` and `kimi-k2.5` through the same gateway.

**Provider compatibility opt-ins (`agent.<backend>.upstream`, Claude Code and OpenCode actor upstream).** Two
transparent, per-upstream switches for providers that refuse requests every other provider accepts. Both default to
"off" (request bytes unchanged; Codex and the golden fixtures are unaffected), both are recorded as
`manifest.agent_backend.provider_compat` (operator data, never model context), and both change only how the request is
shaped, never what the agent or the registry sees:

- `tool_result_images`: `"inline"` (default) or `"user_blocks"`. Measured on the gateway above (2026-11-13): an image
  inside a tool result (`tool_result.content` on `/v1/messages`, a list `function_call_output.output` on
  `/v1/responses`) is refused with HTTP 400 `PROVIDER_TOOL_RESULT_INVALID` for every model tried (kimi-k2.6,
  qwen3.8-flash, qwen3.8-max, deepseek-flash), and on `/v1/responses` any list output is refused, even one without
  images; the same image as a user content block is accepted. With `"user_blocks"` the relay moves, as the last step
  before each attempt leaves (after the egress scrub and the existing image trim/reuse, which are unchanged),
  every tool-result image to a labelled block right after the tool results
  ([`WireAdapter.move_tool_result_images`](../../ni_rua/cortex/session/wire/base.py)):
  - Anthropic Messages: in each user message the `image` blocks leave `tool_result.content` (the tool result keeps its
    text, or gets "(the image of this tool result follows after the tool results)" when none is left) and follow the
    message's last `tool_result` block, each after the text block `[image <k> of tool result <tool_use_id>]`;
  - OpenAI Responses: the `input_image` parts leave each `function_call_output.output` list, a text-only list becomes
    its newline-joined text, and one user message with the labels and images is inserted directly after each
    contiguous run of `function_call_output` items.
  Images are moved as they are: same bytes, same order, nothing re-encoded, resized or dropped; the placement is a
  function of the body only, so a repeated history gives identical bytes (prompt cache stays stable). Every attempt
  event carries `tool_result_images_moved` (count). Models listed with `image_input: false` still refuse any image
  (`IMAGE_INPUT_UNSUPPORTED`).
- `tool_schema_max_depth`: an integer >= 3, or `null` (default, no limit). Measured on the gateway above with
  kimi-k2.6, qwen3.8-flash and deepseek-flash: a tool `input_schema` is accepted exactly when its JSON nesting depth,
  counting every non-empty object and array (the schema root is 1; a `properties` map, a `required`, `enum` or
  `examples` list and every nested schema count; `{}` and `[]` do not; `mcp_server.json_depth`), is at most 13; deeper
  gives HTTP 400 `REQUEST_SCHEMA_VIOLATION (param: tools)`. (Six nested object properties = depth 13 pass, seven = 15
  fail; the same six with an `enum` list on the leaf = 14 fail.) When set, the MCP catalogue NI-RUA publishes to the
  agent ([`mcp_server.fold_schema_depth`](../../ni_rua/cortex/agents/mcp_server.py)) folds, deepest first and as little
  as possible, each sub-schema that would exceed the limit into a node that keeps its `type` (object or array),
  accepts any value of it structurally and has the description: the original description + " JSON schema of this
  value (not expanded: the provider limits tool-schema depth): " + the compact JSON of the original sub-schema, so
  the model keeps the full contract as text. Arguments are validated by the registry against the full schema, so a
  wrong argument inside a folded node gets the same precise error as without folding. The folded tools and JSON
  pointers are recorded (`provider_compat.folded_schemas`). With `13`, the whole real catalogue (Claude Code's native
  tools plus every registry tool) fits with one folded node, `geometry_query`
  `/properties/request/oneOf/7/properties/action/properties/path/properties/waypoints/items` (the `preview_ee`
  waypoint pose objects; the full catalogue measured 17 there).

Complete example for that gateway (Claude Code; the same two keys apply to `agent.opencode.upstream`, whose complete
example is [below](#opencode-with-one-multi-model-provider)):

```json
// configs/claude.local.json (git-ignored; official Claude Code settings format)
{"env": {"ANTHROPIC_BASE_URL": "https://gateway.example/gateway",
         "ANTHROPIC_AUTH_TOKEN": "sk-...",
         "CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0"}}
// configs/ni-rua.local.json
{"agent": {"backend": "claude_code", "claude_code": {"upstream": {
    "source": "claude_settings", "settings": "claude.local.json", "provider": "gateway",
    "tool_result_images": "user_blocks", "tool_schema_max_depth": 13}}}}
```

Verified live (CLI 2.1.321 + relay + that gateway, 2026-11-13): a `look` tool returning a green test image in its tool
result, with `user_blocks`, was answered "green" by kimi-k2.6 and qwen3.8-flash; with `user_blocks` plus depth 13 and
the whole registry catalogue, qwen3.8-flash and deepseek-flash completed the same turn at the CLI's `max_tokens`
32000. kimi-k2.6 then fails with HTTP 400 `INPUT_UPPER_BOUND_EXCEEDED`: on that gateway it refuses inputs above
about 50k tokens (a 40k-token text passes, 55k fails, although its listed `context_window` is 256000), and the full
catalogue alone is about 60k (Claude Code's system prompt and native tools included); that is a provider limit
NI-RUA does not work around.

The whole framework was then run end to end against that gateway (`test_live_provider_e2e`, see
[Validation](validation.md#live-provider-end-to-end); 2026-11-13, both opt-ins on, `model.effort=high`, the SO-101 run
features, a rendered SO-101 scene behind the real `so101_real` backend). The gateway CLI, the console routes, the full
17-tool catalogue (all offered by the CLI) and the relay were real. Each turn ran `robot_observe` (images moved to user
blocks), `task_plan` and a `robot_action` 5 mm move with a completed receipt, and named the rendered cube colour correctly.
This held on qwen3.8-flash, deepseek-flash, qwen3.8-max and kimi-k3, including the console `set_model` switches
qwen3.8-flash -> deepseek-flash and qwen3.8-max -> kimi-k3. kimi-k2.6 was refused on its first request (HTTP 400
`INPUT_UPPER_BOUND_EXCEEDED`, shown in the console's transport block with that message). Recovery worked both after a
refused startup turn and in the middle of a run: `set_model` to a working model, then "Retry model request"
(`retry_model_requests`), then the next prompt, with no gateway restart.

**Switching models during a run (operator console).** The Agent bar's model chip ([above](#the-agent-bar)) is a
combobox: type any model id (or pick one) and press Enter; `apply_agent_settings` (same code path as `set_model`)
applies it from the next agent turn. The gateway checks only that the id is well formed (`console.MODEL_ID`); the
backend refuses what it cannot run and the console shows that error. Suggestions come from
`AgentSession.model_options()`, else the running profile's cached listing, else (Codex home) the pinned native Codex
catalogue; there is no hardcoded model list. A profile run reads its listing through the shared catalogue cache
([Model catalogue cache](#model-catalogue-cache)). Claude Code with `source: "claude_settings"` reads the provider's model listing once at
session start on a background thread ([`model_catalogue.py`](../../ni_rua/cortex/agents/model_catalogue.py); the start
never waits for it): `GET <ANTHROPIC_BASE_URL>/v1/models` with the host credential (plus `anthropic-version`; an
`x-api-key` credential that the listing refuses with 401/403 is retried once as `Authorization: Bearer`). It reads
the OpenAI-style gateway shape (`data[].id/name/protocols/capabilities`, `next_cursor`) and the Anthropic shape
(`data[].id/display_name`, `has_more`/`last_id` -> `after_id`), at most 10 pages of 5 s each, and keeps the models
whose `protocols` (when listed) include `MESSAGES` and whose `capabilities.tools` is not false. Labels are
`name (id)`, with ` · No image input` when `capabilities.image_input` is false (the robot observations are images). A
failed listing changes nothing but the suggestions (the default list plus the current model). The result is recorded
once as the observer event `agent.model_catalogue` (status, reason, the usable models with `context_window`,
`max_output_tokens` and `image_input`); it never enters model context. The session owns that thread: `close()` (every
backend, so also a handoff or a gateway shutdown) detaches it first (`ModelListingMixin._close_model_catalogue`), and a
listing that ends afterwards is dropped, not recorded, and says so in one gateway stderr line. For a `profile` source
the process-wide catalogue cache still keeps the answer for the console and the successor run; a direct read (every
other source) is not cached, so its dropped answer is gone. A Codex session object that is started again
(`CodexSession.start`) reattaches a read that is still running (`_reopen_model_catalogue`). A listing still being read
at close is archived while the Store is open: `model_listing_pending: true` in `agent.session_closed` (Claude Code,
OpenCode) or in `codex.session_close_incomplete` (Codex), next to `threads_alive` (reader threads that outlived their
bounded join); both keys are absent, and the Codex event is not written, on a close without them. The Store's own
lifetime rules are in [Implementation Architecture](architecture.md#read-access-and-information-boundary)
([`tests/test_store_close_race.py`](../../tests/core/storage/test_store_close_race.py)). Claude Code applies a switch before the next
prompt with the SDK's `set_model`, whose CLI first sends one charged `max_tokens: 1` validation request without
tools; a refused switch keeps the previous model (`agent.model_change_rejected`). Verified live (CLI 2.1.321, a
multi-model Anthropic Messages gateway): `kimi-k2.6` -> `deepseek-flash` between two tool-calling turns, the
validation request and the next turn served by the new model (`served_models`). OpenCode declares an undeclared
model by rewriting its config file and switches the native session at the next turn start, see
[`set_model`](#opencode-set_model-and-model_options); its suggestions come from the same listing (`source: explicit` or
`opencode_config`).

**Provider errors and recovery.** On the Claude Code and OpenCode client profiles
(`ClientProfile.record_provider_message`) the relay's private failure evidence (`response_error` in
`model_request_finished`, `model_transport_blocked`, `runtime/model_requests.transport_error`) also keeps a bounded
operator copy of the provider's error message (`message`: at most 300 characters, control characters removed,
credential-looking runs replaced by `[redacted]`; `wire.base.provider_error_message`). Client bytes stay the neutral
relay error, so the message never reaches the agent or model context; the Codex profile is unchanged (its relay error
carries the evidence to the client, pinned by `tests/test_wire_golden.py`). After an explicit provider error the relay
blocks every later model request (manual retry allowed). The console shows the block reason with that message and a
"Retry model request" button (`retry_model_requests`, lease-gated, never automatic): switch to a model that can serve the
request first, then retry; the next turn continues without a gateway restart (Claude Code: the switch is applied before
the next prompt, see above). When the provider lists the configured model with `image_input: false`, the session
records one `agent.model_image_input_unsupported` event and prints one gateway warning, and the console marks the
model control; nothing is refused or switched automatically.

See [Configuration — Agent Backend](configuration.md#agent-backend-agent) for the full `agent.*` field reference
(`agent.backend`, `agent.profiles_dir`, `agent.codex_upstream`, `agent.claude_code.*`, `agent.opencode.*`,
`agent.evidence.*`, upstream `source`/`settings`/`wire_api`/`base_url`/`api_key_env`/`auth_scheme`/`provider`). In short:

- **Claude Code**: `agent.claude_code.upstream` must be a profile serving `anthropic_messages` (`source: "profile"`), `source: "claude_settings"` or an explicit `anthropic_messages` endpoint; `cli_path` is
  optional (defaults to the SDK-bundled CLI); `subagents` (default `true`), `context_delivery` (default `"delta"`,
  opt-out `"full"`), and `auto_compact_window_tokens` are config-file only. `model.effort` must be `low`, `medium`,
  `high`, `xhigh`, or `max`.
- **OpenCode**: `agent.opencode.executable` (absolute path of the pinned 2.2.7 binary,
  [install](#installing-the-pinned-binary)) and `agent.opencode.upstream` (`source: "profile"`, `anthropic_messages` or
  `openai_responses` explicitly, `source: "codex_home"`, or `source: "opencode_config"`; see the
  [complete example](#opencode-with-one-multi-model-provider)) are required; `context_limit_tokens` and
  `output_limit_tokens` default to `null` (per-model provider listing, then `200000` / `32000`,
  [limits](#opencode-declared-models-and-limits)); `thinking_display` defaults to `null`. `model.effort` choices depend
  on the selected wire (`anthropic_messages`: `low`..`max`; `openai_responses`: `none`..`xhigh`).
- **Evidence**: `agent.evidence.client` (`native` default, `relay`) and `agent.evidence.upstream` select the client
  for `features.visual_review_model` / `features.self_briefing`; see
  [Configuration](configuration.md#evidence-helpers-agentevidence).

### Installing the pinned binary

NI-RUA pins OpenCode **2.2.7** exactly (`opencode.config.OPENCODE_VERSION`); 2.x replaces 1.x, there is no dual
support and no fallback to a 1.19.5 binary. The binary is not installed through `npm i`, the curl installer or a package
manager, and it must not be on `PATH`: the curl installer replaces an existing 1.x install and edits shell profiles,
and a binary found on `PATH` is never used (only the absolute `agent.opencode.executable` is). Both macOS and Linux:

1. Pick the target of the host: `darwin-arm64`, `darwin-x64`, `linux-x64`, `linux-x64-baseline` (a CPU without AVX2),
   `linux-arm64` (a `-musl` variant of each Linux target also exists).
2. Download the tarball of that target from the npm registry and extract the one file `package/bin/opencode`:

   ```sh
   TARGET=darwin-arm64        # or darwin-x64, linux-x64, linux-x64-baseline, linux-arm64
   DEST=~/.local/share/ni-rua/opencode-2.2.7
   mkdir -p "$DEST" && cd "$DEST"
   curl -fsSLO "https://registry.npmjs.org/@opencode/cli-$TARGET/-/cli-$TARGET-2.2.7.tgz"
   tar -xzf "cli-$TARGET-2.2.7.tgz" --strip-components=2 package/bin/opencode
   ./opencode --version       # prints: opencode v2.2.7
   ```

3. Set `agent.opencode.executable` to the absolute path `~/.local/share/ni-rua/opencode-2.2.7/opencode` (expanded to absolute path on startup).
4. Install ripgrep (`rg`) on the host and keep it on the gateway's `PATH`. OpenCode's `grep` needs it; without it
   OpenCode tries to download one from `github.com:443`, so the launch is refused with `opencode_ripgrep_missing`. The
   host exposes it to the agent through the run's private `bin/rg`.

`--version` is probed once in a temporary home before launch (`config.probe_cli_version`); a wrong version, a missing
file or a relative path refuses the launch with `agent_backend_unavailable:opencode: <reason>; <install sentence>`
([`factory.opencode_runtime_problem`](../../ni_rua/cortex/agents/factory.py)). The binary's `sha256` is recorded with
the run. NI-RUA runs the binary only as `serve` and as the TUI with `--server`: any other subcommand that talks to a
server would start a detached background service outside the run.

### OpenCode with one multi-model provider

One OpenRouter-style provider (one base URL and one key serving DeepSeek, Kimi, GLM, Qwen, ...) is configured in a
project-local, git-ignored OpenCode config file in the official format, like `codex_home` for Codex and
`claude_settings` for Claude Code. Either OpenCode shape is read
([`opencode/upstream_file.py`](../../ni_rua/cortex/agents/opencode/upstream_file.py)): the V2 shape
`providers.<id>.{package, settings.{baseURL, apiKey}}` of OpenCode 2.x (the pinned runtime), and the V1 shape
`provider.<id>.{npm, options.{baseURL, apiKey}}` of OpenCode 1.x (still accepted so an existing file keeps working; it
only supplies endpoint, key and wire, never the runtime). Complete example for a multi-model Anthropic Messages gateway (one
Anthropic Messages endpoint serving qwen, deepseek, kimi, minimax, ...); the URL and key are placeholders, never commit
a real key (`configs/opencode.local.json` is git-ignored):

```json
// configs/opencode.local.json  (OpenCode 2.x official format; plain JSON, no comments)
{"providers": {"gw": {"package": "aisdk:@ai-sdk/anthropic",
                      "settings": {"baseURL": "https://gateway.example/gateway/v1",
                                   "apiKey": "sk-REPLACE-WITH-YOUR-KEY"}}}}
```

```json
// configs/ni-rua.local.json  (the agent.opencode block; merge it into your existing file)
{"agent": {"backend": "opencode",
           "opencode": {"executable": "~/.local/share/ni-rua/opencode-2.2.7/opencode",
                        "upstream": {"source": "opencode_config",
                                     "settings": "opencode.local.json",
                                     "provider": "gw",
                                     "wire_api": "anthropic_messages",
                                     "tool_result_images": "user_blocks",
                                     "tool_schema_max_depth": 13}}}}
```

```sh
mamba run -n ni-rua python -m ni_rua.apps.gateway --config configs/ni-rua.json --agent-backend opencode --model <id>
```

`<id>` is any model id the gateway serves (for example `qwen3.8-flash`); it is declared in OpenCode and sent upstream
exactly as written. A relative `settings` is resolved next to the config file that sets it.

- `upstream.provider` selects the entry (optional when the file declares one); it is also the provider label recorded
  with the run.
- The wire comes from `upstream.wire_api` when set (it wins), else from the package: `@ai-sdk/anthropic` →
  `anthropic_messages`, `@ai-sdk/openai` → `openai_responses` (the `aisdk:` prefix is optional); any other package
  without `wire_api` is refused (`opencode_config_wire_unknown`). `openai_chat` is refused by the launch gate
  (`opencode_upstream_wire_unsupported:openai_chat`): the relay has no Chat Completions adapter, and the images a
  provider refuses inside tool results are handled by `tool_result_images` instead.
- `baseURL` is mapped to the URL the SDK itself would call: `@ai-sdk/anthropic` calls `<baseURL>/messages`, so the
  relay base is `baseURL` without its trailing `/v1` (a base without `/v1` is refused,
  `opencode_config_base_url_unsupported`); `@ai-sdk/openai` calls `<baseURL>/responses`, so it is used unchanged.
  Missing `baseURL` = the SDK default (`https://api.anthropic.com/v1`, `https://api.openai.com/v1`). The key is sent as
  the SDK sends it (`x-api-key` / `Authorization: Bearer`).
- `apiKey` must be written literally: OpenCode's `{env:...}` / `{file:...}` substitutions are refused
  (`opencode_config_key_substitution_unsupported`). Errors (`opencode_config_*:<file>[:<id>]`) name the file and the
  entry, never the key. The key is read by the gateway only: it is not in `Config.to_dict()`, events or provenance,
  and the agent's environment and generated OpenCode config keep the relay placeholder key
  (`ni-rua-local-relay`).
- Provider-compatibility keys (see [Provider compatibility opt-ins](#configuration-per-backend) above) sit next to
  these in `agent.opencode.upstream`. `tool_result_images: "user_blocks"` is needed whenever the provider answers an
  image inside a tool result with HTTP 400 `PROVIDER_TOOL_RESULT_INVALID` (OpenCode puts every `robot_observe` image
  there); `tool_schema_max_depth: 13` folds the registry catalogue to the depth such a gateway accepts. Both apply to
  OpenCode exactly as to Claude Code; both are recorded in `agent_backend.provider_compat`.
- `agent.opencode.thinking_display` stays `null` for such gateways (they refuse Anthropic's `thinking.display`); set
  `"summarized"` for `api.anthropic.com` ([effort variants](#opencode-effort-variants-and-thinking_display)).
- The launch gate (`factory._opencode_requirements`) reads the file, and `model.effort` choices follow the file's wire.
- `opencode_config` (like `claude_settings`) is an actor-upstream source of its own backend only: `agent.evidence.upstream`
  refuses both (`evidence_upstream_source_unsupported`), and `agent.claude_code.upstream` refuses `opencode_config`.

Verified live (OpenCode 2.2.7 + relay + a multi-model gateway as above, `anthropic_messages`,
`tool_result_images=user_blocks`, `tool_schema_max_depth=13`, `thinking_display` null, the full SO-101 registry
catalogue through `test_live_provider_e2e` with `NI_RUA_LIVE_E2E_BACKEND=opencode`, see
[Validation](validation.md#live-provider-end-to-end)): `qwen3.8-flash` → console `set_model` `deepseek-flash` →
`set_model` `minimax-m3.1-flash-preview`. Every turn observed (images), made a 5 mm move with a completed typed
receipt and named the cube colour; the two later models were declared by config hot reload and switched at the next
turn start; clean shutdown.

Known provider limits (not worked around by NI-RUA):

- **`kimi-k2.6` input cap.** The gateway refuses inputs above about 50k tokens with HTTP 400
  `INPUT_UPPER_BOUND_EXCEEDED` although the model listing says `context_window` 256000. The full registry catalogue plus
  OpenCode's own system prompt and tools already exceed it, so the first request is refused (the console shows the
  transport block with the provider message); recover with `set_model` to another model, then "Retry model request".
- **Models without image input.** The provider listing flags them `image_input: false`. Such a model is declared with
  `capabilities.input = ["text"]`, OpenCode then replaces every tool-result image with its own text `ERROR: Cannot read
  image (this model does not support image input). Inform the user.` (model-visible, no host logic), and the run
  records `agent.model_image_input_unsupported`. Robot observations are images, so pick a model with image input.
- **Output cap.** The gateway lists `max_output_tokens` up to 1048576 but refuses `max_tokens` above 32768, and a model
  with a 16384 limit (`kimi-k2.5`) refuses 32000: NI-RUA caps a catalogue value at 32000 and you set
  `agent.opencode.output_limit_tokens` (for example `16384`) for the lower one ([limits](#opencode-declared-models-and-limits)).

## OpenCode runtime (`opencode serve` 2.2.7)

Everything below is what [`opencode/`](../../ni_rua/cortex/agents/opencode/) does today; the code is the source of
truth. Field-level meanings are in [Configuration](configuration.md#agent-backend-agent).

### OpenCode launch, start checks and the config file

`OpenCodeSession` spawns one private `opencode serve --hostname 127.0.0.1 --port <free port>` in the actor directory
with a clean environment: `HOME` = the actor dir, a per-run `OPENCODE_PASSWORD` (HTTP basic auth, user `opencode`; the
host client and the TUI read it from the environment, it never goes on an argv), private XDG data/cache/state
directories, the process marker, and four isolation flags the binary still reads (`OPENCODE_DISABLE_PROJECT_CONFIG`,
`OPENCODE_DISABLE_CLAUDE_CODE`, `OPENCODE_DISABLE_AUTOUPDATE`, `OPENCODE_DISABLE_MODELS_FETCH`). Start order
(`session.py` module docstring) and the checks that fail it:

1. resume identity (`resume_agent_identity_conflict:<key>`; `cli_version` must equal `2.2.7`);
2. managed config (`opencode_managed_config_present`: `/etc/opencode`, `/Library/Application Support/opencode`, managed
   plists override the generated permissions), exact `--version`, `rg`;
3. the relay, then the host MCP endpoint (Code Mode off), then the model declaration and the launch;
4. `GET /api/info` must answer 200 with version `2.2.7` (it answers 503 with the same body while starting; not ready
   is polled, never an error), then the `GET /openapi.json` route list must equal the checked-in
   [`opencode_routes.json`](../../ni_rua/cortex/agents/opencode/opencode_routes.json) (170 operations), then
   `GET /api/model` must list exactly the declared relay models with their limits, input modalities and variants
   (`opencode_served_models_mismatch`), then `GET /api/mcp` must report `ni_rua` connected
   (`opencode_mcp_connect_timeout` otherwise);
5. session create with the model reference (or the existence check on resume), history adoption, event/reconcile/watch
   threads.

The generated configuration is one file, `<run>/private/opencode/xdg/config/opencode/opencode.json`, in the native V2
shape, in the one read-only config directory that is both `OPENCODE_CONFIG_DIR` and `$XDG_CONFIG_HOME/opencode`
(`OPENCODE_CONFIG_DIR` replaces OpenCode's global config dir, so nothing else is read). It holds: `update`/`share`/
`snapshots` off; the configured `model` as `relay/<id>`; `instructions` = the private `NI-RUA.md` (the instruction channel,
present in every request including after a steer and after compaction); `tool_output` limits that pass a 365 kB text result
whole; `media.image.auto_resize: false` with raised limits (OpenCode otherwise re-encodes PNG tool-result images above
393,216 base64 characters to JPEG; this is harness configuration that switches OpenCode's own re-encoding off, no
NI-RUA image logic changes); `experimental.todo: true`; the permission list ([policy](#opencode-227-ni_ruacortexagentsopencodeconfigpy)); `policies`
that deny every provider but `relay` (otherwise the signed-out `opencode/*` free models are selectable); the `relay`
provider (`package` per wire: `@opencode/ai/providers/anthropic-compatible` or
`@opencode/ai/providers/openai-compatible/responses`, `baseURL` = the run's relay, placeholder key, no header/chunk
timeouts, the declared models); and `mcp.servers.ni_rua` (`remote`, bearer header, `oauth: false`, `codemode: false` so
registry tools stay direct `ni_rua_<tool>` calls with image results, timeouts 30 s start/catalog, 600 s execution). The
host rewrites this file atomically (temp file, fsync, rename; the read-only directory is lifted and restored) to declare a
model selected during the run; OpenCode hot-reloads it within about 0.3 s, and the agent never gets a writable config.

### OpenCode declared models and limits

Only the configured model and every model the operator selects during the run are declared, never the whole provider
catalogue: OpenCode's `subagent` tool and Code Mode can route to any declared model, which would leave the run's
recorded model and price card. Per declared model:

| Field | Source, in this order |
|---|---|
| `limit.context` (tokens; compaction starts at this minus OpenCode's 15% buffer) | explicit `agent.opencode.context_limit_tokens` (used exactly, for every model of the run) → provider listing `context_window` → `200000` |
| `limit.output` (tokens; sent upstream as `max_tokens` / `max_output_tokens`) | explicit `agent.opencode.output_limit_tokens` (used exactly) → `min(listing max_output_tokens, 32000)` → `32000` |
| `capabilities.input` | `["text"]` when the listing says `image_input: false`, else `["text", "image"]` (also when unknown) |

The 32000 ceiling on a listed value exists because catalogues list numbers gateways refuse as a request value. The
listing is read in the background from `GET <base>/v1/models` with the host credential for `source: explicit` and
`opencode_config` ([`model_catalogue.py`](../../ni_rua/cortex/agents/model_catalogue.py)); the start waits for it only
for a bounded time and falls back to the constants. The result is recorded as `agent_backend.effective_limits`
(`{context, output, source}` of the current model; `source` per field is `explicit`, `catalogue`, `catalogue_capped`
or `fallback`) and `agent_backend.declared_models` (`{id, limits, input}` per model). OpenCode sends the output limit
as `max_tokens`; Claude Code sends up to 128000 on its own, a recorded harness difference.

### OpenCode effort variants and `thinking_display`

`model.effort` is declared on every model as one explicit `variant` per tier (an exact request-body fragment) and
selected through the native session's model reference `{id, providerID: relay, variant}`. Explicit variants replace the
ones the provider package would generate (the generated Anthropic ones carry `thinking.display: "summarized"`, which
gateways refuse). The mapping is the identity, no per-model table: a tier the model rejects fails at the provider.

| Wire | Tiers (`config.EFFORT_CHOICES`) | Variant body |
|---|---|---|
| `anthropic_messages` | `low`, `medium`, `high`, `xhigh`, `max` | `{"thinking": {"type": "adaptive"}, "output_config": {"effort": <tier>}}` |
| `openai_responses` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh` | `{"reasoning": {"effort": <tier>}}` (the package itself also sends `store: false` and `include: ["reasoning.encrypted_content"]`) |

Another value refuses the launch with `agent_effort_unsupported:opencode:<effort>`. `agent.opencode.thinking_display`
(`null` default = not sent, `"summarized"`, `"omitted"`) adds `thinking.display` to every Anthropic variant's `thinking`
object: whether an Anthropic endpoint returns summarized thinking text. Non-Anthropic gateways refuse the field, hence
the `null` default; use `"summarized"` for `api.anthropic.com`. It exists only on the `anthropic_messages` wire (another or
an unset `wire_api` is `opencode_thinking_display_requires_anthropic_messages`). `set_effort` applies from the next turn
(`capabilities.live_effort`). Recorded as `agent_backend.native_effort`
(`{variant, body, thinking_display, declared_variants, wire_field}`). Residual differences to Claude Code: OpenCode sends `max_tokens` = the output limit above; on the `openai_responses` wire (`@opencode/ai/providers/openai-compatible/responses`), OpenCode 2.2.7 does not trigger `VerbosityPlugin` (no forced `text.verbosity: "low"`), and NI-RUA passes `model.verbosity` via `settings.textVerbosity` as well as wire-supported `service_tier`.

### OpenCode turn contract, subagents and background work

- **Admission.** `POST /api/session/{id}/prompt` answers 200 with the id of the user message, the turn boundary. A 4xx is a
  definite rejection (the previous state is restored, `AgentRejected`); a 5xx, timeout or dropped connection is unknown
  admission, the session disconnects and the request is never retried. The host's `userMessage` transcript row is
  written before the request, bound to the returned id on 200 and withdrawn on a 4xx; no text matching is involved.
- **Turn end.** The turn ends at the last `idle` message after the boundary (outcome `succeeded`, `failed` or
  `interrupted`) while no session of the private server is active and the inbox is empty. If OpenCode follows a
  failed or interrupted execution with an empty successful execution, the verdict retains the earlier outcome and
  its deciding `idle` ID. A new user, synthetic, assistant or compaction message starts a fresh execution for this
  check. A delivered Anthropic Messages `stop_reason="refusal"` counts toward the model request budget even when
  OpenCode ends the turn with `opencode_turn_failed:provider.content-filter`. The verdict is computed from
  an authoritative snapshot (`GET /api/session/active`, `.../inbox`, every page of `.../message`) by
  [`verdict.turn_verdict`](../../ni_rua/cortex/agents/opencode/verdict.py), never from `/api/event` alone (the stream has
  no replay; every gap is reconciled, and a watchdog reconciles every 10 s). A turn with subagents is confirmed only when
  the same quiescent snapshot is seen twice 0.5 s apart; a verdict that stays non-terminal while nothing is active ends
  as `failed: opencode_turn_incomplete` (a server that died mid-turn writes no `idle`). Error codes carry only the typed
  error kind and status (`opencode_turn_failed:<type>[:<status>]`), never provider text.
- **Idle.** An idle main session with no running execution, an empty inbox and nothing active anywhere is a finished
  turn; the idle watchdog reads only `GET /api/session/active` while no turn runs.
- **Steer.** A prompt sent while a turn runs joins at the next step boundary (a steer during the final text causes one
  more model call in the same execution). It never switches the model.
- **Interrupt.** The relay is blocked first for the main session and every known subagent session (in-flight calls are
  cancelled and new ones denied until the next turn), then `POST /api/session/{id}/interrupt` answers synchronously and
  cascades to foreground subagents; every subagent session still active is interrupted the same way (a background child
  is not reached once its parent is idle). A steer after an interrupt is refused (`turn_interrupt_requested`).
- **Permission requests.** The generated config leaves nothing to ask (`read * allow`, the rest default), so none is
  expected; if one appears (a future default), it is answered `reject` at once (`agent.permission_auto_rejected`) because
  an ask waits for a human in a headless run. OpenCode 2.2.7 then ends the execution without an `idle`, and the turn
  fails `opencode_permission_rejected`. A missed `permission.asked` frame is found by the watchdog's poll.
- **Subagents.** The tool is `subagent` (foreground unless `background: true`); child sessions are discovered from
  `session.created` with a `parentID` and the tool item's `state.metadata.sessionID`. Their transcript rows carry
  `thread_id` = the child session id and `parent_item_id` = the `subagent` tool row, and are never final-answer text. A
  child's registry calls are attributed to the main host turn (NI-RUA does not use a caller identity, although OpenCode 2.2.7 sends
  `_meta["ai.opencode/sessionID"]` with every `tools/call`; the id stays in the MCP raw log only). With `agent.opencode.subagents=true`
  (the default), `context_delivery` is no longer forced to `full`: subagent calls run in `isolated` full-delivery mode while the main chain follows `agent.opencode.context_delivery` (default `"delta"`, opt-out `"full"`).
- **Background adoption (`capabilities.native_wakeup=True`).** A background subagent keeps the turn open while it
  runs, because every registry call it makes must run inside a turn (`McpToolBridge` refuses calls outside one). When it
  ends, its report is enqueued to the parent, which runs again; a native execution that starts while no host turn runs (a
  background subagent's or shell's completion report) is adopted as an NI-RUA turn (`agent.turn_started {origin:
  "native_wakeup"}`, a new turn id, ending at the next `idle`), and a `run_turn` during it joins it
  (`agent.turn_joined`). A background shell never holds a turn; its completion notice is adopted.
- **Compaction.** OpenCode compacts natively once the history passes `limit.context` minus its buffer: one main-chain
  request with tools and system unchanged, then the history is replaced by a checkpoint message. The early signal is the
  `session.compaction.started` event, the authoritative one the `compaction` message (once per id: whichever is read
  first, on its own thread, starts the epoch and is named in its `signal`); both feed `agent.compaction` and the
  decision reminder, and the relay's continuity check still fires. A summary that does not
  match OpenCode's template fails the turn (`opencode_compaction_failed`). The summary request declares tools, so it is a
  charged main-chain call.

### OpenCode `set_model` and `model_options`

`set_model` takes any model id (console free text) and is blocking (the console runs it on a worker thread). A model the
run has not declared is declared first: limits and image input from the table above, the entry added to the config file
by an atomic rewrite, then `GET /api/model` must list it with its declaration within 2 s; otherwise one
`POST /api/location/reload` is sent and the listing is checked again; otherwise the file is restored,
`agent.model_declare_failed` is recorded and `opencode_model_declare_failed:<id>` is raised with the selection unchanged.
Success records `agent.model_declared` (`method` `hot_reload` or `reload`). The model is OpenCode session state: the
native session is switched with `POST /api/session/{id}/model` (model reference with the variant) **only at the start
of the next host turn** when the selected `(model, variant)` differs (`agent.model_changed`; a 4xx keeps the previous
model, `agent.model_change_rejected`). Never mid-turn, never by a steer, never by a joined native turn. Model switches
seen in the timeline are reported as `agent.model_switched`. An id the server does not know fails the next prompt at the
provider (`provider.no-route`), which is why `set_model` declares first. `model_options()` returns the provider listing
filtered by wire and tools (labels `name (id)`, ` · No image input` for `image_input: false`) as console suggestions only; it
is `None` (the console's default list) until the listing was read or when it failed. The in-run switch was verified
live across three models (see the example above).

### OpenCode provenance

`manifest.agent_backend` (observer data, never model context) records for OpenCode: `cli` (`version` `2.2.7`,
`executable`, `sha256`), `wire_api`, `native_model_id`, `provider_alias` (`relay`), `provider_package`,
`route_list_sha256` (of `GET /openapi.json`), `native_effort`, `effective_limits`, `declared_models`,
`provider_compat`, `native_tools` (`enabled`: the tool names of the first main-thread request that declared tools, as
the relay forwarded it, also recorded once as the event `agent.native_tools`, `null` until then since 2.x has no
tool-listing route; `disabled`: `{name, reason}` with `online_access`, `waits_for_human`, `subagents_disabled`), and
`native_settings` (`flags` (4), `config_channel: "config_file"`, `config_dir_read_only`, `rg`, `media_image`,
`tool_output`, `experimental.todo`, `mcp_codemode: false`, provider timeouts). The 1.19.5 keys `compaction_prune`,
`title_agent` and `native_effort.provider_options` are gone.

### OpenCode and archives of 1.19.5 runs

Replay, the viewer, the library and the reports read neutral data only (transcript rows in the Store, `runtime/agent`,
`manifest.agent_backend`, `agent-raw.jsonl`), never native OpenCode messages, and constructing `OpenCodeSession` starts
nothing, so a 1.19.5 archive still opens: its row ids are part ids (`prt_...`, 2.x: `<message id>:<index|tool id>`) and
readers treat ids as opaque; its manifest has nine flags, `compaction_prune` and `title_agent`, which readers read with
`.get`; `runtime/agent.cli_version` tells the two apart. `tests/fixtures/opencode/legacy_1_19_5/archive.json` is one
authentic frozen 1.19.5 run (`LegacyArchive1195` in `tests/test_agent_opencode_projector.py` opens it through
`Store.completed_final_texts`, `load_native_evidence`, `read_agent_identity` and `enter_replay`). A 1.19.5 run cannot be
**resumed** with 2.2.7: the identity check refuses it (`resume_agent_identity_conflict:cli_version`), and native
databases are not migrated.

## Native capability policy

Embedded agents keep their own native tools and abilities (shell, file edit, own code, subagents, todo, skills, ...);
the harness does not restrict an agent to NI-RUA's tools and prefers each harness's default behaviour. Only two kinds of
tool are removed: tools that reach the internet (outside information would break benchmark validity) and tools that
block forever waiting for a human in a headless run. The removals are recorded in `manifest.agent_backend.native_tools`
as `{"enabled": [...], "disabled": [{"name", "reason"}]}` with reasons `online_access`, `waits_for_human` or
`subagents_disabled`; this is observer-only data and never enters model context.

One more native default is changed on two backends, for the information boundary (AGENTS.md "Context carries only
necessary information", "Restricted Information Sources"), not for tools: a run directory below a git repository (a
checkout's `runs/`, or any directory under an empty `.git` such as a shared host's `/tmp/.git`) must not put that host
repository into the actor's context, nor (Codex) send its metadata to the provider. Each is the smallest switch the
pinned binary has for it, and
[`test_agent_prompt_parity.py`](../../tests/cortex/agents/test_agent_prompt_parity.py) (`*HostRepositoryBoundary`) reproduces the
native leak and asserts that nothing of a planted repository (branch, git user, remote, commit, tracked and
untracked files, `AGENTS.md`, `CLAUDE.md`) reaches the model on any backend, below a real repository and below an
empty `.git`: in the first request, a subagent's or child session's requests (Claude Code, OpenCode), the requests
around Claude Code's native compaction and a second user turn. OpenCode's compaction and Codex's subagents are not
captured: the first needs a small declared context and many large tool results over several turns, and Codex's switch
is one `config.toml` its subagents share.

### Codex (0.160.0, `ni_rua/cortex/session/codex.py`)

- `project_root_markers = []` in the generated `config.toml` (`BOUNDARY_CONFIG_LINES`): the actor directory is the
  project root. With the default `[".git"]` Codex reads every `AGENTS.md` from the repository root down to the actor
  directory into the actor context. The actor's own `<actor>/AGENTS.md` is still read and nothing else of the request
  changes (measured).
- `GIT_CEILING_DIRECTORIES=<run>` in the environment of an unsandboxed Codex (`boundary_env`; the actor app-server,
  its native terminal and the native evidence agent, whose ceiling is the directory above its own workspace). The
  boundary towards the provider: below a git repository Codex otherwise adds `workspaces` (the repository root path,
  its remote URLs and its latest commit hash) to the turn metadata of every turn request, in the header
  `X-Codex-Turn-Metadata` and in `client_metadata`. That is not model context, but it leaves the machine, often to a
  third-party relay. (0.160.0 drops the user and password part of a remote URL, not the rest.) With the ceiling, git
  discovery stops above the actor directory, no repository above the run is listed and every other header and body
  byte is unchanged; a repository the agent creates in its actor directory still works. One entry is Codex's own and
  stays: memory consolidation runs in `<codex home>/memories`, a git repository Codex creates itself, and its request
  may list that repository (its path inside the run, which the request carries anyway as the working directory, and
  its baseline commit; no remote). Seen on Linux in 5 of 6 runs, not on macOS; removing it would mean turning native
  memory generation off or rewriting Codex's provider metadata in the relay. Measured with the real binary in
  [`test_codex_repository_metadata.py`](../../tests/cortex/codex/test_codex_evidence_and_metadata.py), below a real repository and
  below an empty `.git`. The evidence agent's ephemeral threads sent no `workspaces` even without it (0.160.0); it
  gets the ceiling because it starts Codex the same way. A sandboxed Codex needs nothing: the namespace starts from an
  empty environment and shows the directories above the actor directory as empty mount-point skeletons, with no
  `.git` (`test_agent_sandbox`, run below a checkout). Recorded as `agent_backend.native_settings` =
  `{"config": ["project_root_markers = []"], "env": {"GIT_CEILING_DIRECTORIES": "<run>"}}` (`env` empty when
  sandboxed).

### Claude Code (2.1.321, `ni_rua/cortex/agents/claude_code/options.py`)

- No `tools=` allowlist option exists: the agent gets the CLI's **default** native tool set minus a denylist
  (`DISALLOWED_TOOLS`): `WebSearch`, `WebFetch` (`online_access`; `CLAUDE_CODE_DISABLE_WEB_FETCH=1` also set), and
  `AskUserQuestion`, `EnterPlanMode`, `ExitPlanMode` (`waits_for_human`, plan exit needs a human approval). The
  subagent tool is `Agent` on the wire and `Task` in the CLI's own init message (2.1.321); both names are denied
  together only when `agent.claude_code.subagents=false` (`subagents_disabled`; default `true`).
- `permission_mode="bypassPermissions"` (parity with Codex `approval_policy="never"`): nothing prompts, so no
  hand-maintained allowlist is needed to avoid `dontAsk` denying every non-read-only call of an unlisted tool. The
  CLI refuses to run as uid 0 under bypass unless it detects a deliberate sandbox (`IS_SANDBOX=1`, bubblewrap,
  docker); the launcher sets `IS_SANDBOX=1` only when the gateway itself runs as uid 0 or inside the NI-RUA
  sandbox.
- Kept at the harness default (design rule: prefer the harness default; remove only online access and
  human-waiting tools): Bash (including background runs), Read/Edit/Write/NotebookEdit/Glob/Grep, `Task`
  (subagents), `TaskStop`, `Monitor`, `Workflow`, `SendMessage`/`ListAgents`, `EnterWorktree`/`ExitWorktree`,
  `ReportFindings`, `Skill` (bundled skills and slash commands only), and the scheduling tools
  `CronCreate`/`CronDelete`/`CronList`/`ScheduleWakeup` (kept enabled — a 2026-11-12 user decision against an
  earlier "no wall-clock scheduling" exception; see [Residuals](#residuals-claude-code)). Native memory
  (`model.native_memories`) is left at the harness default (on): it writes to and reads from the run-private
  `CLAUDE_CONFIG_DIR`, never a path outside the run, so it never crosses runs the way Codex's own home-directory
  memory does not either; `setting_sources=[]` keeps host-global and nested `CLAUDE.md`/`AGENTS.md` discovery off.
- `CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH=1000000` and `ENABLE_TOOL_SEARCH=false` (every registry tool published
  eagerly, matching Claude Code's and OpenCode's `deferred_tools=False`) are set so nothing the model needs is cut.
- `CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS=1` (information boundary, see above; recorded in `native_settings.env`). Without
  it, any `.git` entry above the actor directory makes the CLI put a gitStatus block into the first user message: the
  host repository's branch, git user name, untracked host paths and recent commit subjects. It is the CLI's only switch
  for that block (verified on 2.1.321 and 2.1.325; the `includeGitInstructions` setting reads the same flag, and the
  variable wins). It also removes the generic `# Git` commit/PR advice (667 characters, no host data) from the native
  Bash tool description; nothing else of the request changes. A provider settings file cannot re-enable it
  (`claude_settings_env_reserved`).

### Residuals (Claude Code)

- Root caveat: the exemption `IS_SANDBOX=1`/bubblewrap/docker is a CLI-side string check, not a verified sandbox;
  Docker itself is not detected by the CLI.
- A native subagent (`Task`/`Agent`) runs in the same CLI process tree and its MCP calls carry
  `x-claude-code-agent-id` / `claudecode/toolUseId`, but the registry sees them through the same
  `Harness.handle_tool` barrier as the main actor.
- CLAUDE.md/AGENTS.md auto-discovery stays off everywhere (a validity rule: discovery walks up into the
  developer's archive root and the user's home, pulling host-dependent, non-reproducible files into context); native
  memory under `CLAUDE_CONFIG_DIR` is unaffected.
- Below a git repository (or any `.git`) two git-derived facts remain, without any repository content: the
  `# Environment` line `Is a git repository: true`, and the name of the run-private memory directory, which the CLI
  derives from the repository root (a prefix of the working directory the model already sees); the boundary test pins
  both exactly, derived from the `.git` above each run's own directory (a shared host's empty `/tmp/.git` included).
  Runs outside any repository, or with the sandbox on, get the same prompt on every host.
- Scheduling tools have no wall-clock-aware episode semantics: a timer that fires after the episode ended is
  dropped (the session is closed, or the Harness is no longer accepting); a turn held open by pending scheduled work
  still ends through the ordinary no-progress / inactivity stop.

### OpenCode (2.2.7, `ni_rua/cortex/agents/opencode/config.py`)

- Permission list of the generated config (V2 `permissions`, one `{action, resource, effect}` rule each): `read * allow`
  (closes the one shipped `ask`, `read *.env`, which would wait for a human forever), and `deny` of `webfetch`,
  `websearch` and `question`, plus `subagent` only when `agent.opencode.subagents` is false (default true). There is no
  `* allow` rule (it would re-enable `explore`'s shipped denies) and no `codesearch`, `external_directory` or
  `doom_loop` entry (not 2.x actions or already allowed by default). A `*` deny removes the tool from the model's tool list.
  - `webfetch`, `websearch`: `online_access`; Code Mode's `fetch` is governed by the `webfetch` permission, so it is
    covered too.
  - `question`: `waits_for_human`.
  - `subagent`: `subagents_disabled` (only when the option is turned off).
- Kept (OpenCode 2.x names): `shell`, `read`, `edit`/`write` (`patch` for GPT-family ids), `glob`, `grep`, `subagent`
  (child sessions are the harness's concern), `todowrite` (`experimental.todo: true`; off by default in 2.x), `skill` with
  OpenCode's bundled skills (`opencode`, `report`, `browser`), Code Mode `execute`, and the registry tools as
  `ni_rua_<tool>` (Code Mode off). The 1.x names `bash`, `task` and `apply_patch` no longer exist.
- Not enabled or denied: LSP is not run in 2.x and `formatter` defaults off (the keys are left out); no scheduling tool
  is offered by default; `policies tool.use` is not used (it blocks at call time but leaves the tool listed).
- External discovery is off: `OPENCODE_DISABLE_PROJECT_CONFIG` (project config, `.opencode/` agents, commands, skills,
  plugins, and `AGENTS.md` discovery upward from the working directory) and `OPENCODE_DISABLE_CLAUDE_CODE`
  (`~/.claude` skills; `CLAUDE.md` is never read in 2.x). Only one provider is usable (`provider.use` policies).
- `native_tools.enabled` is the tool names of the first main-thread request that declared tools, exactly as the relay
  forwarded them (2.x has no tool-listing route), so it is the list the model really got.

### Residuals (OpenCode)

- The permission list is enforced by OpenCode; a system-managed OpenCode config would override it, so start-up refuses
  when one exists (`opencode_managed_config_present`).
- A native subagent runs in a child session of the same server; its calls are subject to the same permission list. A
  background subagent can wake the parent (adopted, see above).
- Shell commands run by the agent are not network-isolated by this policy; only the dedicated online tools are removed.
- The generated config (with the MCP bearer) and `OPENCODE_PASSWORD` are readable from the agent's shell, so the agent can
  call its own server's API. Security is a secondary goal of this backend: documented, not hardened.
- Below a git repository the `<env>` block says `Is directory a git repo: yes` and names the repository root as
  `Workspace root folder` (outside a repository: the actor directory); no repository content reaches the model. 2.2.7
  finds the repository by walking up for `.git` and asking git for its top level, with no switch for either, so these
  two lines stay; the boundary test pins them exactly.
- Nested `AGENTS.md` discovery below the workspace stays on: a file the agent itself wrote in a subdirectory of its
  actor dir is loaded when it reads there (the upward discovery is off). `~/.agents/skills` is still discovered; `HOME` is
  the actor dir, so only what the agent wrote itself. The built-in skills `opencode`, `report` and `browser` stay listed.
- `session_move` and `worktree_*` Code Mode tools stay (native ability): a moved session changes its working directory.
- `experimental.todo` may be renamed or removed in a later 2.x: the pin is exact and every bump must re-check it.
- Detached children (`nohup ... &`, `start_new_session`) survive OpenCode's interrupt and SIGTERM; the process-marker
  teardown scan is therefore still required ([Process teardown](#process-teardown-claude-code-and-opencode)).
- Not yet verified on Linux with 2.2.7: the sandboxed run (Bun's native-library extraction into the `/tmp` tmpfs,
  chunked SSE through the inbound Unix forwarder, config hot reload through a read-only bind, `linux-x64` versus
  `linux-x64-baseline`). The 1.19.5 Linux verification does not carry over.

## Native terminal (`features.terminal`)

With `features.terminal=true` (default) and `capabilities.terminal_attach`, the Workbench starts one read-only native
TUI after the first turn from `session.terminal_attach_spec()` and serves it as the console's TUI tab (`/v1/terminal`
WebSocket, bytes kept in `<run>/terminal.bin` for replay, launch metadata without the environment in
`terminal.json`). The spec gives `cwd`, `env`, `attach_uri`, `thread_id` and either Codex's `command_prefix`
(`<prefix> resume --remote <uri> --no-alt-screen <thread>`) or a complete `command` argv
([`AgentSession.terminal_attach_spec`](../../ni_rua/cortex/agents/base.py),
[`Terminal`](../../ni_rua/cortex/session/terminal.py)). The view is read-only by construction: `Terminal` never
writes to the PTY (the browser can only resize it), so the TUI never receives a keystroke. A backend whose spec is
`None` (capability off, session not connected, or the backend skips it) gets no terminal: the manifest records
`terminal_session_id=null`, the snapshot `terminal: null`, and the web console hides the TUI tab (the tab label uses
the backend name). The TUI is stopped by `Workbench.close` before the session closes.

- **Codex**: `codex resume --remote unix://<app-server socket>` inside the actor's sandbox prefix, pinned to the
  session's model and effort (unchanged).
- **OpenCode** (2.2.7; the only version-specific part is
  [`config.tui_attach_command`](../../ni_rua/cortex/agents/opencode/config.py)): `<pinned binary> --server
  http://127.0.0.1:<port> --session <main session> <actor dir>` (the 1.x `attach` subcommand no longer exists), in the
  actor dir, with exactly the server's environment (private XDG dirs, the read-only config dir, the disable flags, the
  process marker); the basic-auth password is read from `OPENCODE_PASSWORD` in that environment, never put on the argv.
  Only an existing session id is passed (`--session` creates a missing one) and `--server` is what keeps the TUI from
  spawning a detached background service. Verified with the pinned binary and a fake upstream
  (`tests/test_agent_opencode_terminal_smoke.py`, gated on `NI_RUA_OPENCODE_EXECUTABLE`): the TUI renders the session
  and follows later turns live; attaching makes no model call, creates no session, spawns no service and changes no
  message; `Terminal.close` ends it, and a TUI left open is stopped by `session.close()` through the process marker. The
  TUI writes only OpenCode's own UI state (`xdg/state/opencode/latest/tui/tabs.json`). With `features.sandbox_enabled=true`
  the terminal is skipped (`terminal: null`): the sandboxed server is reachable only through its inbound Unix socket,
  which the TUI cannot use, and opening a host TCP listener for it would widen the sandbox boundary.
- **Claude Code**: no terminal (`terminal_attach=False`). The actor is driven through the SDK in `--print`
  stream-json mode, and the CLI's live views attach only to sessions the CLI itself runs interactively. Checked with
  the pinned CLI 2.1.321, a fake upstream and an isolated `CLAUDE_CONFIG_DIR`:
  `claude attach <session id>` and `claude logs <id>` answer `No job matching '<id>'` (the SDK session is listed by
  `claude agents --json` as `kind: interactive` but is not a daemon job); `claude --bg` refuses `--print`
  ("--bg and --print conflict ... the job would be unattachable"), so a background actor would need a different,
  interactive driving path (PTY input instead of the SDK's admission, steer, interrupt, `session_state_changed`
  idle, adoption and `--replay-user-messages` echoes), and `--bg` also starts a per-user daemon in `/tmp/cc-daemon-<uid>`
  outside the run; the agent view (`claude agents`) lists only daemon jobs. A `claude --resume <id>` TUI on a copy of
  the transcript (private config dir, relay URL pointed at a recorder) renders the conversation and makes no request,
  but it is a snapshot taken at start: pointed at the live transcript it did not show the next turn, and it appended a
  row of its own to that file (`tools-used`), so it would have to run on a copy that is re-rendered by restarting the
  viewer whenever the transcript grows. It is also an interactive client (it would send a request on any input) and
  needs onboarding/trust state seeded in its config. None of these is a faithful live read-only view, so the backend
  keeps the "Records" tab, which renders the same transcript live from the Store.

## Relay client-loop guards (Claude Code and OpenCode)

The model relay (`ni_rua/cortex/session/model_proxy.py`, `ModelRequestProxy`) runs two transport guards on client
profiles with `client_loop_guard` (`claude_code`, `opencode`; never `codex`, whose relay bytes are pinned by
`tests/test_wire_golden.py`). Both refuse the request without forwarding or charging it, block the transport with
`manual_retry_allowed=True`, and answer with the profile's single-attempt denial (`ni_rua_client_retry_runaway`).

- **Runaway guard** (`RUNAWAY_MAX_IDENTICAL_ADMISSIONS = 8` within `RUNAWAY_WINDOW_S = 2.0` s): byte-identical
  consecutive bodies on one thread, as when a stream ends without a terminal. `error_type="ClientRetryRunaway"`, event
  `model_client_retry_runaway`.
- **Consume guard** (`CONSUME_GUARD_MAX_IDLE_COMPLETIONS = 3`): a main-chain request whose history adds only the
  model's own earlier output (`WireAdapter.history_model_output_flags`) to the last delivered main request is *idle*,
  because the client did not act on the completion it was given. After 3 delivered idle completions in a row, the next
  idle request is refused: `error_type="ClientDidNotConsumeCompletion"`, evidence
  `classification="client_did_not_consume_completion"`, event `model_client_did_not_consume_completion`. Observed (historical, on
  opencode 1.19.5; the guard is client-agnostic and stays) when provider usage carries `input_tokens: null`: every answer was charged, OpenCode could not parse
  it and re-requested (180 calls in about 4 s on the Responses wire). With the guard the loop costs 4 charged
  completions on either wire. Any tool result, user/steer message, text merged into the last client message, or
  continuity discontinuity resets the count. Helper and subagent requests are not compared. Measured false positives:
  none in the 420 delivered main completions of the captured Claude Code and OpenCode sessions
  (`tests/fixtures/continuity/`) (`tests/test_model_proxy_consume_guard.py`).
- Residuals: a client that adds its own note to every re-request counts as adding input and is not detected. A loop
  inside a subagent chain is not compared. Both are still bounded by the budget. Provider usage is delivered unchanged
  (no normalisation of invalid counters).

## Sandbox (`features.sandbox_enabled`, all backends)

The sandbox exists for RoboDojo benchmark runs: the model must not read the RoboDojo / simulator source (reward
hacking). Every backend runs its native agent in the same bubblewrap namespace (`AgentSandbox`,
[`agent_sandbox.py`](../../ni_rua/cortex/session/agent_sandbox.py); Codex through its subclass `CodexSandbox`), and
`factory.backend_requirements` accepts `features.sandbox_enabled=true` for `codex`, `claude_code` and `opencode`.

- Inside: the agent's own binary (read-only), its run-private state directories (read-write), the actor directory
  (cwd and `HOME`), the published Harness source snapshot at `/harness` (read-only; written to
  `<run>/private/actor-source`), the system runtime, `rg` and the PyPI proxy. The relay (and the MCP endpoint for
  Claude Code and OpenCode) is reached at its unchanged `127.0.0.1` port through run-owned Unix sockets; the host
  reaches OpenCode's HTTP server only through `<ipc>/inbound/<port>.sock` (no host TCP listener).
  OpenCode 2.2.7 inside the namespace: the pinned binary read-only at `/opt/ni-rua-bin/opencode`, `rg` at
  `/opt/ni-rua-bin/rg`, XDG data/cache/state read-write, the one config directory (`xdg/config/opencode`) and the
  instructions file read-only, `/harness` read-only, and a `/tmp` tmpfs that must allow execution (Bun extracts about
  16 MB of native libraries there at start). The host still rewrites `opencode.json` on its side of the read-only bind to
  declare a model selected during the run (hot reload, else `POST /api/location/reload`;
  [`config.sandbox_spec`](../../ni_rua/cortex/agents/opencode/config.py)). The terminal is not offered when sandboxed.
  This 2.2.7 layout has plan and launch-gate tests on every host; its real-namespace run on Linux is still open
  (see [Residuals (OpenCode)](#residuals-opencode)).
- Never mounted: the checkout, the user's home, RoboDojo and simulator assets, the run's observer data. The network
  namespace has only `lo`.
- Recorded: `runtime/agent.isolation` and `manifest.agent_backend.isolation` =
  `bubblewrap_namespaces_package_egress_v1` (`disabled` without the sandbox); a resume across modes is refused.
- Host requirement: Linux with `bwrap` and system `python3`; unprivileged user namespaces are preferred, otherwise the
  rootful `sudo -n bwrap` + `setpriv --clear-groups` fallback is used (also on hosts where the gateway user has
  supplementary groups). The launch gate cannot probe this, so a host without bubblewrap fails at session start with
  `codex_sandbox_bubblewrap_not_found_on_path_or_os_defpath` (Codex) or
  `agent_sandbox_bubblewrap_not_found_on_path_or_os_defpath` (Claude Code, OpenCode) before the native agent
  starts; nothing ever falls back to unsandboxed execution.
- Verification: conformance item 28 (`AgentBackendConformance.test_sandbox_namespace` for Claude Code and both
  OpenCode wires; `tests/test_codex_sandbox_conformance.py` for Codex) on a Linux host, see
  [Validation](validation.md#host-capabilities).
- Residuals: the sandbox userland is the explicit `SHELL_COMMANDS` list of
  [`system_runtime.py`](../../ni_rua/cortex/session/system_runtime.py) plus `python3` and its alias `python` (for example `perl`, `id` and
  `setsid` are absent; `EXCLUDED_COMMANDS` names what is never published). Inside the namespace, teardown of detached children relies on the pid namespace ending with
  its init; outside it, on the process marker scan.

## MCP tool catalogue (Claude Code and OpenCode)

`tools/list` (`mcp_server.mcp_tool_catalog`) publishes the registry actor catalogue with the session's presentation
scrub. Recorded harness differences from the Codex path:

- **Portable input schema.** Top-level `anyOf`/`oneOf`/`allOf`/`if`/`then`/`else` are removed from every `inputSchema`
  (`portable_input_schema`; nested ones are untouched). Claude Code 2.1.321 silently drops an MCP tool whose schema
  has one, and Anthropic's API rejects it (OpenCode on `anthropic_messages` forwards it unchanged). Only `task_plan`
  has one today (operation-dependent required fields). The rules stay in the tool description (the generated
  `input: ... Object rules: ...` clause of the field guide, asserted for every registry tool by
  `tests/test_tool_registry_prose_bounds.py`) and `Harness.handle_tool` validates against the full registry schema.
- **Description cap.** Claude Code truncates MCP descriptions to 4096 characters; the launch environment sets
  `CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH=1000000` (recorded in `native_settings.env`) so every generated field guide
  arrives complete.

## Process teardown (Claude Code and OpenCode)

Both sessions stop their native process through one shared path, `isolation.stop_agent_tree`
(`ni_rua/cortex/agents/isolation.py`). The launch environment carries `NI_RUA_AGENT_INSTANCE=<runtime/agent.instance_id>`,
which every native tool child inherits. At close the session kills the process tree it observed and every process that
carries this marker, then scans again; `agent_processes.confirmed` (and with it `runtime/agent.clean_shutdown`) is true
only when nothing targeted survived and the final scan found no marked process. This also catches children a shell
call detached (`cmd &`, `setsid`) after they were re-parented to pid 1, and it applies after an agent crash. A resumed
session first stops what the previous instance of the run left running (`agent.previous_process_stopped`; the resume
is refused when that cannot be verified).

- Linux reads `/proc/<pid>/environ`. On macOS with SIP the kernel withholds the environment of Apple platform
  binaries (`/bin/sh`, `/usr/bin/perl`, `/bin/sleep`, ...) from other processes. For those the scan uses the working
  directory instead: such a process is selected when it is orphaned, or is a child of a marked process, and its cwd
  lies inside the actor directory. An operator's shell that merely `cd`s into the run is never selected.
- Residuals (documented, not mitigated): agent code can drop the marker (`env -u`, `env -i`). On macOS, a platform
  binary started that way and running outside the actor directory is not found. The marker value is visible to the
  agent's shell; it is an opaque session id, not a credential.

## Comparison protocol for harness experiments

A benchmark comparison across backends is a harness-identity comparison, not a model comparison, unless every other
condition is held fixed (AGENTS.md "Establish the Comparison Protocol First"). Hold fixed across arms:

- **Model** (`model.name`) and **upstream provider** (`agent.<backend>.upstream.provider`): the same served model on
  the same provider, verified by `conditions.agent_cli` / the recorded SDK/CLI versions, not assumed from config.
- **Reasoning effort tier** (`model.effort`): pick a tier every backend in the comparison natively accepts
  (`factory.NATIVE_EFFORT_CHOICES`, OpenCode's `config.EFFORT_CHOICES[wire]`) — Codex accepts every tier, Claude
  Code accepts `low`/`medium`/`high`/`xhigh`/`max`, OpenCode's choices depend on the selected wire.
- **Evidence client** (`agent.evidence.client`): set `"relay"` on every arm when `features.visual_review_model` or
  `features.self_briefing` is on (the fairness rule, [Configuration](configuration.md#agent-backend-agent)) — a
  Codex arm left on `"native"` would compare `CodexEvidenceAgent` against `RelayEvidenceClient` on the other arms,
  not the agent runtime.
- **Subagents setting** (`agent.claude_code.subagents` / `agent.opencode.subagents`): record as an explicit
  condition; it no longer forces `context_delivery="full"` (subagent calls run in `isolated` full-delivery mode while
  the main chain follows `context_delivery`, which defaults to `"delta"` on all backends and supports opting back into
  `"full"`), though flipping it changes native tool availability and subagent activity.
- **Sandbox mode** (`features.sandbox_enabled`): hold fixed across arms; it is frozen for the run's lifetime and a
  resume across modes is rejected.
- Task, variant, seed, layout, and budgets, as for any comparison (AGENTS.md "Experiments & Evaluation").

What is a condition of the backend itself, not an experimental intervention to control away — each is a recorded,
by-design difference, not a bug:

- **Tool-catalogue delivery.** Codex defers 7 of 17 registry tools behind native `tool_search` (eager on Claude Code
  and OpenCode, whose own tool search stays off); Codex drops schema keywords (bounds, examples, conditionals,
  nested property descriptions of `robot_action` / `geometry_query.request`) that Claude Code and OpenCode on
  Anthropic Messages keep, while OpenCode on OpenAI Responses drops a similar (but not identical) subset. The
  dropped bounds/units/defaults stay in the generated field-guide prose on every backend (AGENTS.md's
  per-parameter documentation requirement), so the contract is never silently incomplete.
- **Generation limits.** OpenCode sends `max_tokens`/`max_output_tokens` = the declared `limit.output` of the model
  (`agent.opencode.output_limit_tokens`, else the provider listing capped at 32000, else 32000;
  [limits](#opencode-declared-models-and-limits)); Claude Code sends up to 128000 for a recognised Claude model id (32000 otherwise); Codex has no
  separate cap.
- **Verbosity / reasoning fields.** Codex sends `text.verbosity: "high"` and `reasoning.context: "all_turns"`;
  on the `openai_responses` wire (`@opencode/ai/providers/openai-compatible/responses`), OpenCode 2.2.7 does not
  trigger `VerbosityPlugin` (no forced `"low"`) and forwards `model.verbosity` via `settings.textVerbosity` along with
  wire-supported `service_tier`; Claude Code has no verbosity field.
- **Native base prompts and tool catalogues.** Each backend's own base instructions, environment block, memory
  section, skills list, and native tool set are part of the condition under test, not rewritten by NI-RUA (see
  [`agent-backend-prompt-parity.md`](reports/agent-backend-prompt-parity.md) section 4 for the measured sizes).
- **Subagent attribution.** A Claude Code or OpenCode subagent's MCP registry calls are attributed to the main host
  turn (NI-RUA does not use a caller identity: the HTTP request carries none, but MCP `_meta` does on every host, OpenCode 2.2.7 `ai.opencode/sessionID`, Codex thread, session and turn ids, and Claude Code `claudecode/toolUseId`; a Claude Code stdio MCP server also gets `CLAUDE_CODE_SESSION_ID` in its environment, an HTTP one does not; the ids stay in the MCP raw log); child-session usage is charged to the main thread with no per-subagent
  breakdown. Codex's own multi-agent mode has its own accounting.
- **No Chat Completions wire yet.** `openai_chat` has no relay adapter in this checkout; OpenCode refuses to start
  on it.
- **Steer timing.** A Claude Code steer joins the active turn through `query()` mid-stream; an OpenCode steer (`POST
  /api/session/{id}/prompt` while a turn runs) answers at once but joins only at the next step boundary.
- **Self-initiated turns.** Claude Code's native wake-ups (background shells, scheduled wake-ups) and OpenCode's
  (a background subagent's or shell's completion report, which also holds a turn open while the subagent runs) are
  adopted as NI-RUA turns (`native_wakeup=True`); Codex never starts a turn unprompted.

## External backend (`agent.backend = "external"`)

NI-RUA launches no agent and no model relay: `ExternalSession` ([`ni_rua/cortex/agents/external/session.py`](../../ni_rua/cortex/agents/external/session.py)) serves the run's registry tools on `http://127.0.0.1:<agent.external.mcp_port>/mcp` (`NI-RUAMcpServer` with `serverInfo.name` `ni_rua`, no bearer, a short fixed `initialize.instructions`) and the user's own Claude Code, Codex or OpenCode drives the run. Every capability flag is `False` (`compaction_signal="none"`), `model_proxy=None`, `owned_processes()=[]`. To align external runs with hosted defaults while respecting the no-relay boundary: `agent.external.context_delivery` defaults to `"delta"` (opt-out `"full"`), using a single-main write-back lease on `ExternalToolBridge` (overlapping concurrent calls and non-main OpenCode child sessions automatically fall back to `isolated` full-delivery mode; new turns and new clients refresh the context epoch); the first `run_begin` on a simulated-clock run delivers `startup_inline_grounding` under strict 5-gate guards (never replayed on re-briefing, real-time arms, or after actions); and `robot_urdf_delivery="file"` (default) directs the agent to fetch URDF on demand via `robot_information` rather than inlining 46k–60k URDF prose into `run_begin`. No-relay hard boundaries remain: there is no `ContinuityTracker` (compaction relies on soft plugin hooks and `run_begin` re-briefing), no relay-side image trimming (`trim_oversized_requests`/`trim_max_inline_images`) or exact-image reuse (`reuse_exact_images`), and Claude Code HTTP `_meta` carries no session id (so serial subagent calls while the main chain is blocked cannot be separated from the main lease at the protocol layer). Turns belong to the agent: `run_begin` opens one, `run_finish` closes it, and the dispatcher gates every other tool on a briefed open turn. The console cannot prompt, steer, switch model or hand off (`<command>_unsupported_by_backend`); `stop`, `hold`, `estop`, `resume` and `reset` are unchanged and an open agent turn never blocks them. The manifest records `agent_backend.kind="external"`, `information_boundary="mcp_boundary_only"` and `null` for `model`, `provider`, `effort`, `price_card`, `model_request_limit` and `no_progress_timeout_s`; the summary has `null` served models and token usage, never 0. It coexists with the hosted backends: choose per run, and an installed plugin does not change a hosted run ([Two modes, not mutually exclusive](agent-plugin.md#two-modes-not-mutually-exclusive)). Details, the installer, the recovery recipes and the residuals are in the [Agent Plugin guide](agent-plugin.md); launch refusals are listed under [`agent.backend`](configuration.md#agent-backend-agent).

## Known limitations and residuals

- The sandbox (all three backends, see [Sandbox](#sandbox-featuressandbox_enabled-all-backends)) needs Linux with
  bubblewrap; it was verified on a Linux host on the rootful `sudo -n bwrap` path, the unprivileged user-namespace path
  was verified for Claude Code and OpenCode only before the final fixes. The OpenCode verification was done on 1.19.5;
  the 2.2.7 port is covered by plan and launch-gate tests on every host, and its real-namespace run on Linux is still open.
- `features.lean_projection` (Responses-shaped history projection) is Codex-only
  (`lean_projection_requires_codex_backend`).
- Order of one model step's tool calls: host mode executes them in the model's emitted order
  ([MCP tool transport](#mcp-tool-transport)); a call waits 5 s (10 s at the very most) for an earlier call the
  client never sent, and a call without an exact place (for example arguments the client rewrote) runs in arrival
  order with an `agent.tool_order_unmatched` record. External mode has arrival order only
  ([Agent Plugin](agent-plugin.md#order-of-one-steps-calls)).
- `external` runs are interactive-only and exploratory: not queueable, no benchmark mode, no sandbox, no call budget, unknown served model and usage, no image trim; the agent keeps all its own tools ([Agent Plugin](agent-plugin.md#information-boundary-and-benchmark-validity)).
- Claude Code's own MCP image re-encoding (images over 512,000 bytes or 2000 px per side) is a recorded per-harness
  difference, not mitigated by new host logic (image transport otherwise uses the same trim/reuse logic, byte for
  byte, on every backend).
- `apps/evaluation/checkpoint.py` and `turn_probe.py` refuse non-Codex archives (native thread fork is Codex-only;
  `AgentCapabilities.fork`).
- A resumed non-Codex run is gated against its archived configuration, not the invocation's; a resume after an
  unclean exit invalidates the model context epoch on every backend.
- A handoff builds the successor's native runtime in the same process right after the predecessor's closed (no
  re-exec). Verified live across all three backends on the rendered scene and on the real SO-101 (2026-11-14); a
  successor that cannot start falls back once to the previous settings. Profiles and the console API carry the
  accepted security residuals listed under [Operator guidance](#operator-guidance).

## Reports

- [Agent Backend Prompt Parity](reports/agent-backend-prompt-parity.md) — captured-request comparison of the first
  actor generation request per backend: instruction channel placement, tool-catalogue delivery, native additions,
  and the `actor_instructions.py` wording review.
- [Agent Backend Spikes](reports/agent-backend-spikes.md) — summary of the pre-implementation spikes against the
  real Claude Code CLI and OpenCode binary, and the Linux sandbox verification.
- [Agent Plugin Spikes](reports/agent-plugin-spikes.md) — what was measured on the real Claude Code, Codex and OpenCode
  binaries for the agent plugin (plugin formats, permission and MCP-client defaults, HTTP-first go/no-go, recovery,
  wake) and the per-host end-to-end conformance results.
