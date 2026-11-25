# Validation and Release Guide

Validation must demonstrate that changes behave correctly along their actual call paths and clearly state the scope of collected evidence. For minor documentation edits, checking links and content is sufficient; changes affecting permissions, physical execution, state transitions, archiving, or scoring require regression tests covering failure and edge paths.

## Environment and Standard Checks

Run all commands from the repository root. Development uses the Miniforge `ni-rua` environment by default—never create a project `.venv` or upgrade an environment in place while live experiments are using it. NI-RUA requires Python >= 3.11; dependency versions are governed by `environment.yml`, `requirements.lock.txt`, and `pyproject.toml`, with the Codex SDK/CLI currently pinned to `0.160.0`. Before running checks, verify the resolved SDK/CLI versions from the `ni-rua` interpreter rather than relying on whatever `codex` binary is on `PATH` (see [Environment Setup](running.md#environment-setup)). `ffmpeg` is required for video-related checks.

```sh
# Skip environment creation if NI-RUA already exists; verify dependencies per the Running Guide first.
mamba env create -f environment.yml
npm ci --prefix web
mamba run -n ni-rua python scripts/run_tests.py
npm run check --prefix web
npm run build --prefix web
mamba run -n ni-rua python scripts/build_package.py --output dist
mamba run -n ni-rua python scripts/check_package.py dist/ni_rua_harness-0.2.0-py3-none-any.whl
git diff --check
```

`scripts/run_tests.py` discovers the full `unittest` suite under the 2D zoned tree `tests/<zone>/<subzone>/test_*.py` (`core`, `robot`, `perception`, `cortex`, `runtime`, `apps`, `packaging`) and executes it across isolated persistent worker processes (capping per-worker BLAS/OpenMP/Tokio thread pools via `DEFAULT_SUBPROCESS_THREAD_ENV` and assigning each worker an independent short `TMPDIR` / `XDG_RUNTIME_DIR`). `tests/packaging/` intentionally omits `__init__.py` so it never shadows PyPI `packaging` (needed by `setuptools`), and `scripts/run_tests.py` / `scripts/run_tests_per_module.py` import `tests/packaging/<subzone>/*.py` under `tests.packaging.<subzone>.<module>`.

Use these selectors with `scripts/run_tests.py` for fast feedback:
- `--zone <z>` / `--subzone <s>`: target one or more architectural zones (`core`, `robot`, `perception`, `cortex`, `runtime`, `apps`, `packaging`) or subzones (`robot.so101` or `so101`).
- `--tier fast | component | packaging`: filter by module speed/isolation tier declared via `NI_RUA_TEST_TIER`.
- `--changed [<base_ref>]` (`--changed-scope auto|direct|full`, `--explain`, `--strict-empty`): select affected tests from `git diff` via the 6-layer bounded dependency DAG.

Use `-j <N>` or `NI_RUA_TEST_JOBS=<N>` to override worker concurrency (`-j 1` runs serially; `PYTHONPATH=tests mamba run -n ni-rua python -m unittest discover -s tests` covers all zones with `__init__.py`, and pass `tests/packaging/<subzone>` explicitly when using `unittest` directly). When validating subsequent releases, pass the exact path to the newly built wheel so `dist/*.whl` does not inadvertently pick up older wheels. `check_package.py` installs the wheel into a temporary directory via `pip --target` using the current interpreter and verifies it outside the repository checkout under Python's `-I` isolated mode without creating a virtual environment. The CPU test suite requires neither a live model nor a GPU simulator; a few asset-mapping tests are skipped when an external RoboDojo checkout is absent. To exercise those asset tests, set `NI_RUA_ROBODOJO_ROOT` and `NI_RUA_ROBODOJO_PYTHON` to your local installation paths and re-run, recording pass and skip counts separately. Native app-server tests run against a local loopback model stub; if SDK/CLI versions do not match, switch to the proper environment rather than skipping failures.

### Running the Full CPU Suite

[`scripts/run_tests_per_module.py`](../../scripts/run_tests_per_module.py) runs every zoned `tests/<zone>/<subzone>/test_*.py` module in its own interpreter (the one that runs the script, so start it with `mamba run -n ni-rua python`) with an unchanged environment, `--jobs` modules at a time, and isolates hangs/crashes per module. For fast parallel execution within a single run, prefer `mamba run -n ni-rua python scripts/run_tests.py`. A module that exceeds `--timeout` seconds is killed with its process group and reported as `timeout`; a module whose interpreter dies (a signal such as SIGSEGV, `os._exit`, or a non-zero exit after its tests) is reported as `crashed`. Either way the run continues, and the report keeps the end of the module's output, which includes a `faulthandler` stack dump of every thread taken shortly before the kill or at the crash. One module's hang, crash or leaked state cannot affect another.

```sh
mamba run -n ni-rua python scripts/run_tests_per_module.py --jobs 6 --timeout 900 --report /tmp/ni-rua-cpu.json
mamba run -n ni-rua python scripts/run_tests_per_module.py core.config.test_config 'cortex.agents.*'      # a subset
mamba run -n ni-rua python scripts/run_tests_per_module.py --baseline /tmp/ni-rua-cpu.json --report /tmp/ni-rua-cpu-2.json
```

The summary lists the module and test counts, every failing test id, the timeouts and crashes with their output tails, and the skip reasons with their counts. The JSON report also holds tracebacks, skip reasons and passing ids per module. `--baseline <report.json>` prints the failures that are new and the baseline failures that now pass, are now skipped, or did not run; it also orders modules longest first using the baseline durations. `--log-dir <dir>` keeps each module's full output. The exit status is non-zero on any failure, error, unexpected success, timeout or crash. Ctrl-C or SIGTERM stops the run: no further module starts, the running modules are killed with their process groups and reported as `interrupted`, the partial report is marked `"interrupted"` and lists the modules that did not run (`not_run`), and the exit status is 130 or 143. A timeout or an interrupt reaches only processes that stayed in the module's process group; a child that started its own session is not reached. Keep `--jobs` moderate on a shared machine.

`mamba run -n ni-rua python scripts/run_tests.py -j 1` runs all 7 zones (including `tests/packaging/*`) sequentially in a single worker. Plain `mamba run -n ni-rua python -m unittest discover -s tests` runs the 6 package zones under `tests/` in one process (skipping `tests/packaging/*` because `tests/packaging/__init__.py` is intentionally omitted so `tests/` never shadows PyPI `packaging`; run `PYTHONPATH=. mamba run -n ni-rua python -m unittest discover -s tests/packaging/npm -t .` and `-s tests/packaging/sandbox -t .` separately when invoking stdlib `unittest discover` directly). In single-process runs, a hang or a crash in one module stops the whole run, and state left by one module (threads, event loops, patched globals, native libraries) is visible to every later module. Add `-X faulthandler` (`python -X faulthandler -m unittest discover -s tests`) so that a native crash prints the stack of every thread instead of ending the run silently.

## Targeted Regressions by Scope

Native Codex sandbox regressions with isolation enabled (`features.sandbox_enabled=true`) require Linux, a discoverable `bwrap` binary, and system `python3`; unprivileged user namespaces are preferred, falling back to `sudo -n` and `setpriv` when user namespaces are disabled on the host. These tests verify isolation boundaries using real files, subprocesses, and a loopback Responses stub while covering tool installation path variations and IPC directory configurations. Actual PyPI package downloads and authenticated upstream model calls are recorded separately and are never triggered implicitly by the CPU test suite. Regressions for `features.sandbox_enabled=false` must verify that the primary Codex session, terminal, evidence agent, and `program_run` start cleanly while sandbox probing/loading is forbidden, that private home directories, placeholder credentials, broker review, and resource limits remain enforced, and that cross-mode session resumption is rejected. Because host file and loopback network access are expected when the sandbox is disabled, test them against temporary files and local stubs—never report unsandboxed test passes as proof of OS isolation.

The table below lists targeted test modules under `tests/<zone>/<subzone>/` (for example, `mamba run -n ni-rua python scripts/run_tests.py runtime.tools.test_tool_registry runtime.harness.test_harness_core`, `mamba run -n ni-rua python scripts/run_tests.py --zone runtime --subzone tools`, or `PYTHONPATH=tests mamba run -n ni-rua python -m unittest runtime.tools.test_tool_registry runtime.harness.test_harness_core`). When changes span multiple public boundaries or prepare a release, run the full CPU test suite after targeted tests pass.

| Change Scope | Primary Test Modules | Behavior to Verify |
| --- | --- | --- |
| `ToolSpec` / handlers / output | `runtime/tools/test_tool_registry`, `runtime/harness/test_harness_core`, `runtime/programs/test_program_review` | Both actor and program entry points, zero side effects on invalid inputs, role permissions, typed facts, and no action replay on post-execution presentation errors. |
| Actions / worker / stopping | `runtime/actions/test_unified_actions`, `robot/motion/test_motion_guards_and_wait`, `runtime/harness/test_harness_lifecycle_and_shutdown` | Budgets, idempotency, and checkpoints; `partial`/`stopped`/`unknown` handling; terminal capture failure preserves receipts and invalidates stale observations. |
| Scene evidence | `perception/scene/test_scene_contracts`, `perception/scene/test_scene_service` | Per-entity provenance and clocks, dirty/freshness invalidation, and service/job lifecycle. |
| Calibrated projection and overlays | `perception/scene/test_projection_validity`, `runtime/harness/test_harness_geometry_and_contact`, `runtime/actions/test_trajectory_feedback_and_tools`, `perception/scene/test_scene_contracts`, `robot/motion/test_batch_motion`, `perception/external/test_object_table`, `perception/depth/test_depth_geometry_and_capture` | A point beyond the camera's distortion-model monotonic range is not visible and has no pixel (real SO-101 wrist-camera case), points inside it keep OpenCV's exact pixel, zero distortion is byte-identical, every caller reports such a point as outside the image and draws nothing for it, and results stay strict JSON within the scene contract. |
| Store / scoring / archives / reports | `core/contracts/test_contracts`, `apps/library/test_library_and_read_models`, `apps/library/test_experiments_and_reports`, `apps/library/test_experiment_comparisons` | Unchanged file bytes/hashes/mtimes on sealed WAL archives, corrupted archive isolation, fixed case denominator, complete references and scopes, and preserved scoring facts on import. |
| Evaluation queue and completion hook | `apps/evaluation/test_benchmark_queue`, `apps/evaluation/test_queue_conditions_and_hook` | Exact `argv` recognition across current and legacy entry points, PID/start-ticks/output ownership checks, and OOM retry boundaries. |
| Web UI / read-only API | `npm run check --prefix web`, corresponding Python API tests, browser fixtures | Pagination, terminal state refresh, read-only replay, input injection guards, and mobile layout. |
| Embodiment profile, per-robot catalogue and instructions, X5 parity | `robot/embodiment/test_embodiment_x5_golden`, `robot/embodiment/test_embodiment_profile`, `robot/embodiment/test_embodiment_schema`, `robot/embodiment/test_embodiment_contracts`, `robot/embodiment/test_embodiment_prompt_cleanup`, `robot/embodiment/test_embodiment_perception_evidence`, `runtime/harness/test_harness_lifecycle_and_shutdown`, `robot/so101/test_so101_motion_features` | X5 goldens byte-identical except the encoded allowances ([Embodiment Profile](embodiment-profile.md#what-the-goldens-protect)); profile and capability agreement; every published bound enforced at its limit; no other robot's words in a catalogue; SO-101 motion features on the fake arm. CPU only: native X5 motion and the real SO-101 arm need their own evidence. |
| SO-101 camera binding | `robot/so101/test_so101_camera_and_recalibration` | Mocked AVFoundation and v4l device lists only (no camera opened): macOS OpenCV index from the uniqueID-sorted list, a missing or ambiguous `unique_id` / `device_name` refuses with no index fallback (the wrist-unplugged case), index-only binding still opens and reports its device, unverified OpenCV order, Linux by-id/by-path, device-list change on open, worker refusal before the bus connects and an operator-only binding record. The real AVFoundation read is checked by running `scripts/so101_cameras.py` on the Mac. |
| Config / assets / entry points / wheel | `core/config/test_config`, `core/runner/test_parallel_runner`, full wheel build and check | Out-of-checkout imports and CLI console scripts, parallel test runner isolation and fixture boundaries, default config bundling, static web assets, and provenance metadata. |
| Native Codex tools and permissions | `packaging/sandbox/test_codex_sandbox`, `packaging/sandbox/test_agent_sandbox`, `packaging/sandbox/test_agent_claude_sandbox` and `packaging/sandbox/test_agent_opencode_sandbox` (plan and launch-gate tests run everywhere; the real-namespace tests need `bubblewrap_namespaces`), `cortex/agents/test_agent_workbench`, `runtime/actions/test_action_compilation_and_source`, `cortex/codex/test_codex_config_and_provider`, `runtime/programs/test_program_isolation`, `cortex/codex/test_codex_session`, `cortex/codex/test_codex_evidence_and_metadata` | Real native shell/memory operations, host file/network denial when sandboxed, zero sandbox dependency with preserved broker/review/budgets when unsandboxed (default), resume mode consistency, source snapshot publishing and placeholder auth, and dynamic image transport. |
| Sandbox boundary, all backends (conformance item 28) | `cortex/claude_code/test_claude_code_conformance`, `cortex/opencode/test_opencode_conformance` (`test_sandbox_namespace`), `packaging/sandbox/test_codex_sandbox` (`CodexSandboxConformance`) | With `features.sandbox_enabled=true` the agent's own native shell runs a probe inside the namespace: `/harness` holds exactly the published source snapshot (read-only); the checkout, regular files of the user's home and a planted fake RoboDojo root next to the run directory are unreadable; the only interface is `lo`, the forwarded relay/MCP ports connect, a host-only loopback port and public addresses (`1.1.1.1:443`, `8.8.8.8:53`) fail within 5 s; a registry tool call round-trips; close leaves no bwrap, supervisor, agent or detached child process (tracked by pid and start time) and removes the run IPC directory; `runtime/agent.isolation` and provenance record `bubblewrap_namespaces_package_egress_v1`. OpenCode additionally reaches its server only through the inbound unix socket. Needs Linux with bubblewrap and the pinned runtimes (`requires_claude_cli`, `requires_opencode` with `NI_RUA_OPENCODE_EXECUTABLE`, `requires_codex_binary`). |
| Relay evidence client | `cortex/relay/test_relay_evidence_client` | Per-wire request bodies through the relay against loopback fake servers, attribution, shared budget across the actor and `evidence` bindings, cancellation, recovery gate, egress scrub and alias restore, failure strings identical to `CodexEvidenceAgent`, the Codex native/relay switch, manifest `evidence_client`, a non-Codex self-briefing start, and the queue's evidence-path warning. Provider acceptance of the structured-output request is not covered (needs a user-authorised live probe). |
| Agent backend conformance bindings | `cortex/claude_code/test_claude_code_conformance`, `cortex/opencode/test_opencode_conformance` (both relay wires), [`tests/support/agent_conformance.py`](../../tests/support/agent_conformance.py) | The shared conformance mixin's numbered items (tool catalogue delivery, image/MCP contract, the emitted order of one model step's tool calls (item 6: a rendezvous proves that parallel-safe calls overlap, and for OpenCode the step's first call is delivered last), continuity, subagent attribution, native-tool provenance, turn adoption, teardown) bound against each backend's real session and a loopback fake model; backend-specific `expectedFailure`s are recorded with the defect they track, not silently skipped. |
| Agent backend prompt parity | `cortex/agents/test_agent_prompt_parity`, [`tests/support/prompt_capture.py`](../../tests/support/prompt_capture.py) | Byte-identical NI-RUA instructions and first user turn across Codex, Claude Code, and OpenCode (both relay wires); every registry tool reaches the model; no wording in `actor_instructions.py` is backend-specific. See [`agent-backend-prompt-parity.md`](reports/agent-backend-prompt-parity.md). |
| Model relay client-loop guards | `cortex/relay/test_model_proxy_continuity_and_guard`, `cortex/codex/test_wire_golden`, `cortex/relay/test_wire_anthropic_and_responses`, `cortex/relay/test_wire_opencode` | Byte-identical Codex relay bytes (golden fixtures, never touched by a `client_loop_guard` profile), the runaway and consume guards refusing without forwarding or charging a request, and the client-visible denial strings OpenCode does not retry. |
| Provider profiles, console agent settings and handoff | `cortex/agents/test_agent_profiles`, `cortex/codex/test_codex_config_and_provider`, `cortex/relay/test_model_catalogue`, `cortex/agents/test_agent_contracts_fast`, `apps/gateway/test_gateway_agent_settings`, `apps/gateway/test_gateway_handoff`, `npm run check --prefix web` (`agent-settings-model.test.mjs`, `ui-combobox.test.mjs`), browser checks `web/check-agent-settings.mjs` and `web/check-console-shell.mjs` | Profiles load, refuse with a code that names the file and never contain the key (exceptions, `repr`, `ProfileInfo`, `Config.to_dict()`, every run file); `Config()` / Codex snapshots byte-identical while `agent.codex_upstream` / `agent.profiles_dir` are null; one auth scheme per wire; sessions send the profile credential upstream and the placeholder to the agent; the catalogue cache (TTL, refresh, mtime, single flight, stale); generated reasoning options; agent-options, the dry run (including `price_card_mismatch`), profile writes outside `/v1/control`, `apply_agent_settings`, `run.turn_settings` and `summary.agent_settings.mixed`; `handoff_run` blockers, the successor on the same port, the carried speed cap, the one fallback run and signals mid-handoff (fakes on port 0). Real runtimes and providers need the live scenario below. |
| Agent host capabilities (non-sandbox) | `cortex/agents/test_agent_contracts_fast`, `cortex/agents/test_agent_isolation_and_replay`, `cortex/agents/test_agent_transcript_and_recovery`, `cortex/agents/test_agent_mcp`, `cortex/relay/test_context_signal`, `cortex/agents/test_agent_workbench`, `cortex/agents/test_agent_session_and_followups` | Backend-neutral `AgentSession`/`AgentCapabilities` contract, factory launch gating (`backend_requirements`), MCP tool bridge dispatch ordering, transcript row shapes and `agent.*` events, relay context-continuity signal, and process teardown markers; these run on every host (no kernel capability required). |
| External agent backend and agent plugin | `cortex/external/test_external_config_and_session`, `cortex/external/test_external_workbench`, `packaging/npm/test_agent_plugin_package`, `cortex/external/test_external_host_e2e_support` (and the opt-in `cortex/external/test_external_host_e2e`); host goldens `cortex/codex/test_wire_golden`, `robot/embodiment/test_embodiment_x5_golden`, `cortex/agents/test_agent_mcp` | Launch refusals and the queue/handoff refusals; the `run_begin` / `run_finish` bracket, the briefed-turn gate (including a call dispatched before `run_begin` was admitted: refused, nothing executed), operator stop/reset/resume against an open turn, the no-bearer endpoint (`Origin`, fixed port, `initialize.instructions`), manifest and summary nulls, `ToolRegistry.effect_of`; the Codex limits map equals the registry and the three host entries agree on URL and port; hosted backends unchanged byte for byte. See [External agent backend and agent plugin](#external-agent-backend-and-agent-plugin). |
| Live provider end-to-end (opt-in, charged) | `cortex/agents/test_live_provider_e2e`, [`tests/support/so101_scene.py`](../../tests/support/so101_scene.py) | The real gateway CLI and `serve` (Workbench, HTTP and console routes, shutdown, finalization) with the Claude Code or OpenCode backend (`NI_RUA_LIVE_E2E_BACKEND`), the full SO-101 actor catalogue (the SO-101 run features) and a real provider named by a provider profile (`NI_RUA_LIVE_E2E_PROFILE`) or the agent tool's own provider file (`NI_RUA_LIVE_E2E_SETTINGS`): a startup turn, then a console `set_model` + `prompt` turn per further model; or (`NI_RUA_LIVE_E2E_HANDOFF`) `apply_agent_settings` turns and a `handoff_run` chain across backends with the chain checked in the archive. Asserts that the agent offers every registry tool, `robot_observe` returns images, the answer names the rendered cube colour, `robot_action` returns a completed typed receipt, no transport block, `served_models` lists every model, the console offers the provider catalogue, and a clean shutdown; prints the relay's operator evidence on failure. The robot is the real `so101_real` backend and SO-101 adapter with only the worker process in-process, a simulated servo bus and rendered calibrated cameras. Skipped unless `NI_RUA_LIVE_E2E_PROFILE` or `NI_RUA_LIVE_E2E_SETTINGS` is set; proves provider and framework compatibility, not motion quality. See [Live provider end-to-end](#live-provider-end-to-end). |

The suite gives the same result at any checkout location and under any account. A test process never reads the checkout's git-ignored `configs/ni-rua.local.json`: importing a zone package (`tests/<zone>/__init__.py`) points `ni_rua.core.config.LOCAL_CONFIG_PATH` at the never-created `configs/ni-rua.local.json.absent` ([`support.paths.ignore_local_config`](../../tests/support/paths.py)), and a test of the local layer patches it to a file of its own. A module that starts a real `Workbench` calls `support.paths.isolate_home()` from `setUpModule`, so the grasp memory store every fresh `Workbench` opens at the `~/.local/share/ni-rua/grasp-memory` default (and every other `~` default) is in a temporary home, never the operator's. Compare a `~`-derived Config path with `Path(...).expanduser().resolve()`: Config resolves path fields, and a home below a symlink (macOS `/tmp`, a linked `/home`) differs from its merely expanded spelling. Neither switch reaches a subprocess a test starts (the live end-to-end gateway below still reads the local file).

Use `find tests -name 'test_*.py'` (or `rg --no-ignore --files tests`) to check test module paths before running, and add new regression tests to the existing test module owned by the affected subsystem. Fault-injection tests must assert persisted state facts and side-effect counts rather than merely matching error message substrings. If a historical minimal reproducer asserts buggy legacy behavior, keep it as baseline documentation rather than including it in the green regression suite.

### Live Provider End-to-End

`cortex/agents/test_live_provider_e2e` is the only CPU-side test that calls a real, charged model provider, so it never runs
unless asked for. It starts `python -m support.so101_scene` (the real gateway CLI with the Workbench's
`create_backend` replaced by the rendered SO-101 scene backend) on a loopback port and drives it like the web
console. Environment: `NI_RUA_LIVE_E2E_BACKEND` (`claude_code`, default, or `opencode`; `opencode` also needs
`NI_RUA_OPENCODE_EXECUTABLE` = the pinned 2.2.7 binary and `rg`), `NI_RUA_LIVE_E2E_PROFILE` (path of a provider
profile, `configs/profiles/<id>.local.json`, `upstream.source=profile`; its directory becomes `agent.profiles_dir`;
also asserted: `manifest.agent_backend.upstream_profile` names the profile and wire, and the key is in no file of the
run), `NI_RUA_LIVE_E2E_WIRE` (the profile wire, for OpenCode with a two-wire profile), or instead
`NI_RUA_LIVE_E2E_SETTINGS` (path of the agent
tool's provider file in its official format: a Claude Code settings file, `upstream.source=claude_settings`, or an
OpenCode config file, `upstream.source=opencode_config`; a path, never a key),
`NI_RUA_LIVE_E2E_MODELS` (comma-separated, default `kimi-k2.6`; model 1 serves the startup turn, each further
model is switched to with `set_model`; OpenCode declares each new model in its config file and switches at the next
turn), `NI_RUA_LIVE_E2E_PROVIDER` (Claude Code: provider label; OpenCode: the provider id in the file),
`NI_RUA_LIVE_E2E_CONFIG`
(JSON object deep-merged into the generated config, for example the provider compatibility opt-ins),
`NI_RUA_LIVE_E2E_EFFORT` (default `high`), `NI_RUA_LIVE_E2E_TURN_TIMEOUT_S` (default 900),
`NI_RUA_LIVE_E2E_CUBE_COLOUR` (default random), `NI_RUA_LIVE_E2E_RUN_ROOT` (keep the run there) and
`NI_RUA_LIVE_E2E_EXPECT_REFUSED` (models the provider is known to refuse: their failed turn must show a retryable
transport block with the provider's status and message, and the test recovers through `set_model`,
`retry_model_requests` and `prompt`). The features are
the SO-101 run features with `remote.enabled=false` and `features.vggt_endpoint` on a closed loopback port, so the
VGGT parts report `unavailable` while the catalogue and instructions stay those of the real run. A
`configs/ni-rua.local.json` of the checkout still applies underneath. Each turn prints one `LIVE_E2E_TURN` JSON line
(model, status, seconds, tool calls, robot actions with measured z change, request count, HTTP statuses, request
latency, served models).

Handoff scenario: `NI_RUA_LIVE_E2E_HANDOFF` (needs `NI_RUA_LIVE_E2E_PROFILE`; it replaces the model-switch scenario)
lists hops separated by `;`, each `<backend>:<profile>:<model>[:<wire>]` (`profile` = an id or file in the profile
directory; `wire` only when the profile serves two the backend can use; `NI_RUA_LIVE_E2E_EFFORT` is every hop's
effort). The gateway starts on `NI_RUA_LIVE_E2E_BACKEND` with model 1, each further model of
`NI_RUA_LIVE_E2E_MODELS` is applied with `apply_agent_settings` and gets a turn, then every hop is a console
`handoff_run` whose successor startup turn is asserted like every turn. After the stop the archive must show the
chain: predecessor `summary.status: stopped`, `successor_run_id`, `successor_handoff.status: handed_off` and one
`run.handoff_requested`; successor `manifest.handoff.predecessor_run_id` / `request_id` (no fallback), its backend
and profile, one `run.handoff_started`; `run.turn_settings` of every turn with its backend, profile, wire and model;
each run's `served_models` limited to its own models; `summary.agent_settings.mixed` true exactly for a run whose
model changed; `cleanup_status: completed` and (MCP backends) `clean_shutdown`; no key of any profile of the chain in
any file. Every backend of the chain must be installed (skip reason otherwise). One `LIVE_E2E_HANDOFF` line per hop
reports the `handoff_run` time and acceptance → successor built (`run.handoff_started`) and → successor ready
(`run.started`) from the archived UTC event times; `LIVE_E2E_CHAIN` lists the runs.

Reference live results:

| Scenario | Backends, profiles, models | Result |
| --- | --- | --- |
| Profile, model switch | Claude Code + multi-model gateway (`backends.claude_code.env` forwards `CLAUDE_CODE_THINKING_DISPLAY_UPDATES=0`, compat `user_blocks` / 13), `deepseek-flash`, 2 turns | passed: observe with images, 5 mm moves with completed receipts, cube colour named, listing `ok` (28 usable), forwarded variable names in the manifest, clean shutdown, no key in any file |
| Handoff chain | Claude Code + anthropic-gateway `claude-haiku-5-5` → `apply_agent_settings` `claude-sonnet-5-5` → OpenCode + multi-model-gateway `deepseek-flash` → Codex + responses-gateway `gpt-6.1-terra` | passed: 4 turns, chain recorded, `mixed` only on run 1; acceptance → successor ready 3.7 s (OpenCode) and 2.3 s (Codex) |

```sh
NI_RUA_LIVE_E2E_SETTINGS=configs/claude.local.json NI_RUA_LIVE_E2E_MODELS=qwen3.8-flash,deepseek-flash \
  NI_RUA_LIVE_E2E_CONFIG='{"agent": {"claude_code": {"upstream": {"tool_result_images": "user_blocks", "tool_schema_max_depth": 13}}}}' \
  PYTHONPATH=tests mamba run -n ni-rua python -m unittest -v cortex.agents.test_live_provider_e2e
NI_RUA_LIVE_E2E_BACKEND=opencode NI_RUA_LIVE_E2E_SETTINGS=configs/opencode.local.json \
  NI_RUA_OPENCODE_EXECUTABLE=$HOME/.local/share/ni-rua/opencode-2.2.7/opencode \
  NI_RUA_LIVE_E2E_MODELS=qwen3.8-flash,deepseek-flash,minimax-m3.1-flash-preview \
  NI_RUA_LIVE_E2E_CONFIG='{"agent": {"opencode": {"upstream": {"wire_api": "anthropic_messages", "tool_result_images": "user_blocks", "tool_schema_max_depth": 13}}}}' \
  PYTHONPATH=tests mamba run -n ni-rua python -m unittest -v cortex.agents.test_live_provider_e2e
# provider profile (compat opt-ins and Claude Code variables come from the profile)
NI_RUA_LIVE_E2E_PROFILE=configs/profiles/provider-a.local.json NI_RUA_LIVE_E2E_MODELS=deepseek-flash \
  PYTHONPATH=tests mamba run -n ni-rua python -m unittest -v cortex.agents.test_live_provider_e2e
# handoff chain across the three backends
NI_RUA_LIVE_E2E_PROFILE=configs/profiles/provider-b.local.json \
  NI_RUA_LIVE_E2E_MODELS=claude-haiku-5-5,claude-sonnet-5-5 \
  NI_RUA_LIVE_E2E_HANDOFF='opencode:provider-a:deepseek-flash;codex:provider-c:gpt-6.1-terra' \
  NI_RUA_OPENCODE_EXECUTABLE=$HOME/.local/share/ni-rua/opencode-2.2.7/opencode \
  PYTHONPATH=tests mamba run -n ni-rua python -m unittest -v cortex.agents.test_live_provider_e2e
```

The live test never uses port 8781 (the gateway binds port 0). The real-arm handoff check is recorded in
[SO-101 integration §6.4](so101-integration.md#64-agent-bar-model-switching-and-handoff).

### External agent backend and agent plugin

CPU-only checks, run from the repository root:

```sh
mamba run -n ni-rua python scripts/run_tests.py \
  cortex.external.test_external_config_and_session \
  cortex.external.test_external_workbench \
  packaging.npm.test_agent_plugin_package
node --test agent-plugin/test/*.test.mjs        # installer and doctor against fake host executables; Node >= 18
mamba run -n ni-rua python scripts/gen_agent_plugin_codex_mcp.py --check   # exit 1 when agent-plugin/.codex-mcp.json is stale
```

Pass the test files to `node --test` as a glob: on Node 25 a bare directory argument (`node --test agent-plugin/test`) fails. After a change to an actor tool's name or to the registry, regenerate the Codex map with the same script without `--check`. The generator imports the `ni_rua` package of the checkout it lives in.

`agent-plugin/test/tarball-check.sh` packs the package, installs the unpacked tarball into fresh isolated homes (`env -i`, never your real `~/.claude`, `~/.codex` or `~/.config/opencode`) on the pinned host binaries and on the user's, checks the plugin, the MCP entry and the skills, uninstalls, and compares the home. It reads the pinned Claude Code and Codex from the `ni-rua` environment, the pinned OpenCode from `~/.local/share/ni-rua/opencode-2.2.7/opencode` (`CLAUDE_PINNED`, `CODEX_PINNED`, `OPENCODE_PINNED`, `CLAUDE_USER`, `CODEX_USER` override the paths; a missing binary is skipped). `SPIKE_DIR` (absolute path of the spike fixtures: a real `NI-RUAMcpServer` stub and a scripted fake model) is optional: without it a small Node MCP server is used and the model-level "skills and tools reach the model" checks are skipped. It needs no provider and calls no model.

Opt-in host end-to-end ([`tests/cortex/external/test_external_host_e2e.py`](../../tests/cortex/external/test_external_host_e2e.py), support code [`tests/support/host_e2e.py`](../../tests/support/host_e2e.py)): a real gateway with `--agent-backend external` over the SO-101 scene double, the plugin installed by the real installer into a fresh home, and each host binary (Claude Code 2.1.321 and the user's, Codex 0.160.0 and the user's, OpenCode 2.2.7) in its default permission mode against a loopback scripted model. No real model, provider or arm is involved. Without `NI_RUA_HOST_E2E=1` the module is skipped with the reason.

```sh
NI_RUA_HOST_E2E=1 [NI_RUA_HOST_E2E_LONG=1] PYTHONPATH=tests mamba run -n ni-rua python -m unittest -v cortex.external.test_external_host_e2e
PYTHONPATH=tests mamba run -n ni-rua python -m unittest cortex.external.test_external_host_e2e_support   # CPU only, 22 tests, no host binary
```

Environment variables: `NI_RUA_HOST_E2E=1` runs it; `NI_RUA_HOST_E2E_LONG=1` makes the long call last 330 s on Claude Code 2.1.321 (past its 300 s MCP idle timer) and 90 s on Codex 0.160.0 and OpenCode 2.2.7 through the scene's test-only `execute_hold_s`, otherwise it is a short call; `NI_RUA_HOST_E2E_HOSTS` is a comma-separated subset of the target keys (default all five); `NI_RUA_HOST_E2E_ROOT` keeps every home, gateway run and model request under that directory (default a removed temporary directory); `NI_RUA_HOST_E2E_CLAUDE_PINNED`, `..._CLAUDE_USER`, `..._CODEX_PINNED`, `..._CODEX_USER` and `NI_RUA_HOST_E2E_OPENCODE` override the binary paths (a missing binary skips that host with the reason). Run time: 563.8 s for the five hosts with the long call on macOS (58.6 s without `NI_RUA_HOST_E2E_LONG`). One `HOST_E2E {...}` line per host reports every result. Recorded results: [Agent Plugin](agent-plugin.md#per-host-conformance-results).

Live mode of the same module (opt-in, calls a real, charged provider, never in the normal suite): `ExternalHostLiveTest` runs one real-model session per pinned host. The setup is the same: isolated home, real installer, real gateway over the scene double, the default permission mode. The provider comes from a provider profile; the key goes only to the host process and the kept evidence is scrubbed of it. The gateway's `--prompt` asks the model to name the cube colour, move the end effector 5 mm up and finish with `run_finish`. The host's only user message says to operate the robot through the ni_rua MCP tools. The test checks that the first NI-RUA call is `run_begin`, that `robot_observe` returned images, that a `robot_action` receipt is `completed`, that the `run_finish` summary is the turn's final answer with the turn `completed`, and that the colour is right. One `HOST_E2E_LIVE {...}` line per host records the tool sequence, request count, token usage, served model, refusals and wall time. Without `NI_RUA_HOST_E2E_LIVE=1` it is skipped with the reason.

```sh
NI_RUA_HOST_E2E_LIVE=1 NI_RUA_HOST_E2E_LIVE_PROFILE=<profile id or configs/profiles/<id>.local.json> \
  NI_RUA_HOST_E2E_LIVE_CLAUDE_MODEL=<id> NI_RUA_HOST_E2E_LIVE_CODEX_MODEL=<id> NI_RUA_HOST_E2E_LIVE_OPENCODE_MODEL=<id> \
  [NI_RUA_HOST_E2E_HOSTS=claude-pinned,codex-pinned,opencode-pinned] [NI_RUA_HOST_E2E_ROOT=<keep evidence here>] \
  PYTHONPATH=tests mamba run -n ni-rua python -m unittest -v cortex.external.test_external_host_e2e.ExternalHostLiveTest
```

Further variables:

- A host without a model variable is skipped.
- `NI_RUA_HOST_E2E_LIVE_OPENCODE_WIRE` (`anthropic_messages`, the default, or `openai_responses`) picks OpenCode's wire. Claude Code needs the profile's `anthropic_messages` wire and Codex its `openai_responses` wire.
- `NI_RUA_HOST_E2E_LIVE_TIMEOUT_S` (default 600 s) and `NI_RUA_HOST_E2E_LIVE_MAX_REQUESTS` (default 40) cap each session.
- `NI_RUA_HOST_E2E_LIVE_CUBE_COLOUR` fixes the colour (default random per host).
- For Codex, pick a model the pinned binary has metadata for. Otherwise Codex falls back to default metadata.

Recorded results: [Agent Plugin, Real-model sessions](agent-plugin.md#real-model-sessions).

A passing run of the scripted suites proves the software layers and the transport with a scripted model. A passing live run proves one short task per host and model, not that a real model operates a robot well through a given host.

### Host Capabilities

Sandbox and OS-isolation tests exercise real kernel features, so they run only on hosts that provide them and are skipped elsewhere instead of failing. Each gated test is decorated with `require(<probe>, "<what the test exercises>")` from [`tests/support/host_capabilities.py`](../../tests/support/host_capabilities.py). A probe checks the actual capability with the production lookup or a minimal kernel call, never the platform name alone, and is cached per test process. Probes that change process state (namespaces, seccomp filters) run in a short-lived child process.

| Probe | What it checks | Gated tests (examples) |
| --- | --- | --- |
| `bubblewrap_namespaces` | `bwrap` and system `python3` are found, and a namespace sandbox starts: first the production unprivileged user-namespace probe, then the `sudo -n` + `setpriv` fallback (5 s per attempt) | `packaging/sandbox/test_codex_sandbox`, sandbox-on cases in `cortex/codex/test_codex_config_and_provider`, `cortex/codex/test_codex_evidence_and_metadata`, `runtime/programs/test_program_isolation`, the real-namespace tests of `packaging/sandbox/test_agent_sandbox`, `packaging/sandbox/test_agent_claude_sandbox`, `packaging/sandbox/test_agent_opencode_sandbox`, and conformance item 28 (`cortex/claude_code/test_claude_code_conformance`, `cortex/opencode/test_opencode_conformance`, `packaging/sandbox/test_codex_sandbox`) |
| `program_isolation` | Landlock ABI >= 4, working seccomp filters, and `ldd`/`ldconfig` resolving `libseccomp.so.2`, as needed by bounded programs with the sandbox on (`program_services` fixtures default to it) | `runtime/programs/test_program_isolation`, sandbox-on cases in `runtime/programs/test_program_review`, `runtime/programs/test_program_execution`, `apps/evaluation/test_benchmark_and_matrix` |
| `landlock` | Linux on x86_64/aarch64 and a Landlock ABI >= 1 from the version query (nothing is restricted) | `runtime/programs/test_program_isolation` |
| `seccomp` | A child process loads the production socket-denial filter and `socket()` is denied (20 s timeout) | Component of `program_isolation` and `native_replay_sandbox` |
| `native_replay_sandbox` | Landlock ABI >= 3, working seccomp filters, and `find_library("seccomp")` | Capturing cases in `cortex/agents/test_agent_isolation_and_replay` |
| `linux_runtime_tools` | `ldd` and `ldconfig` on the production search paths, and `ldconfig` resolves `libc.so.6` | `cortex/agents/test_agent_contracts_fast` |
| `linux_proc` | `/proc/<pid>/stat` start ticks and `/proc/self/cmdline` are readable | PID-identity cases in `apps/evaluation/test_benchmark_queue` and `apps/evaluation/test_queue_conditions_and_hook` |
| `pidfd_syscall_fallback` | With `os.pidfd_open` hidden, the production raw-syscall fallback opens a pidfd for its own process and signals it (Linux x86_64/aarch64 only) | `test_pidfd` fallback case in `cortex/agents/test_agent_isolation_and_replay` |
| `address_space_limit` | A child process can cap `RLIMIT_AS` at 1 GiB, as the unsandboxed `program_run` launch does (missing on macOS, where `program_run` records the cap under `unenforced_limits`) | `expect()` selects the memory-limit expectation in `runtime/programs/test_program_isolation` (enforced cap, or recorded as unenforced); strict mode fails the fallback |

Running item 28 on a benchmark host (strict mode, pinned runtimes, no model and no network beyond loopback):

```sh
NI_RUA_REQUIRE_SANDBOX_TESTS=1 NI_RUA_OPENCODE_EXECUTABLE=$HOME/.local/share/ni-rua/opencode-2.2.7/opencode \
  mamba run -n ni-rua python scripts/run_tests.py \
  cortex.claude_code.test_claude_code_conformance.ClaudeCodeConformance.test_sandbox_namespace \
  cortex.opencode.test_opencode_conformance.OpenCodeAnthropicConformance.test_sandbox_namespace \
  cortex.opencode.test_opencode_conformance.OpenCodeResponsesConformance.test_sandbox_namespace \
  packaging.sandbox.test_codex_sandbox.CodexSandboxConformance
```

Unix socket paths must fit in 107 bytes on Linux (103 on macOS). The tests bind theirs below short `/tmp` roots or a short `XDG_RUNTIME_DIR` of their own, so the suite also passes with a deep `TMPDIR` or one inside the checkout (see [Running the Suite on a Linux Host](#running-the-suite-on-a-linux-host)). A real sandboxed Claude Code or OpenCode session needs a short `TMPDIR` or `XDG_RUNTIME_DIR` for its IPC directory (it refuses to start otherwise; Codex falls back to `/tmp`). The checkout, the user's home and the fake RoboDojo root are never mount sources; a checkout above a mounted run directory appears inside the sandbox only as the empty directories on the way to the mount point.

On macOS every probe reports the capability as missing. A Linux host with bubblewrap, unprivileged user namespaces (or passwordless `sudo`), Landlock ABI >= 4, and libseccomp is expected to pass every probe. Run `PYTHONPATH=tests mamba run -n ni-rua python -m support.host_capabilities` to print each probe's result for the current host. A skipped test appears in unittest output as `skipped 'requires host capability <probe>: <reason> [probe: <why it is missing>]'`, and skips are counted separately from passes and failures.

On hosts that are supposed to support the sandbox, such as Linux benchmark hosts, set `NI_RUA_REQUIRE_SANDBOX_TESTS=1`. With this variable set, a missing capability fails the test with `MissingHostCapability` instead of skipping it: a gated method fails, and a gated class errors in `setUpClass`. This prevents a capable host from silently skipping the sandbox suite, whether the host lost a capability or the production code a probe reuses has regressed. A test that checks the enforced behaviour where the capability exists and a documented fallback elsewhere branches on `expect(<probe>, "<what the test checks>")` in its body; strict mode makes a missing capability fail there too. Gate only tests that cannot run without the capability. Do not gate failures caused by temporary-path canonicalization (`/var` versus `/private/var` on macOS), Unix-socket path length, timing, or missing optional tools; fix those or record them separately.

### Running the Suite on a Linux Host

The full CPU suite in strict mode on a shared Linux host, with the optional agent runtimes, without a robot, a GPU or a model provider. Prerequisites: `git`, `curl`, bubblewrap (`bwrap`), `setpriv` (util-linux), passwordless `sudo -n` for `setpriv`, the `ni-rua` environment, and a kernel that passes the probes of the table above (Landlock ABI >= 4 and libseccomp for `program_isolation`; `PYTHONPATH=tests python -m support.host_capabilities` prints them). Everything goes into one fresh temporary directory; the `ni-rua` environment is only read:

```sh
T=$(mktemp -d /tmp/ni-rua-suite.XXXX); mkdir -p "$T/src" "$T/home" "$T/tmp"
git archive HEAD | tar -x -C "$T/src"            # committed files only: never configs/*.local.json or a key
PY=$(mamba run -n ni-rua python -c 'import sys; print(sys.executable)')
# The claude extra at exactly the requirements-claude.lock.txt pins (claude-agent-sdk 0.2.180 bundles the CLI).
"$PY" -m pip install --no-deps --no-cache-dir --target "$T/site" -r "$T/src/requirements-claude.lock.txt"
# OpenCode 2.2.7 (agent-backends.md, "Installing the pinned binary") and ripgrep, which OpenCode needs on PATH.
(cd "$T" && curl -fsSLO https://registry.npmjs.org/@opencode/cli-linux-x64/-/cli-linux-x64-2.2.7.tgz \
  && tar -xzf cli-linux-x64-2.2.7.tgz --strip-components=2 package/bin/opencode && rm cli-linux-x64-2.2.7.tgz)
"$T/opencode" --version                          # prints: opencode v2.2.7
"$PY" -m pip install --no-deps --no-cache-dir --target "$T/rg" ripgrep==15.1.0
cd "$T/src"
sudo -n env "PATH=$T/rg/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" "HOME=$T/home" \
  setpriv --reuid="$(id -u)" --regid="$(id -g)" --clear-groups \
  env USER="$(id -un)" LOGNAME="$(id -un)" NI_RUA_REQUIRE_SANDBOX_TESTS=1 CUDA_VISIBLE_DEVICES= TMPDIR="$T/tmp" \
      PYTHONPATH="$T/site" NI_RUA_OPENCODE_EXECUTABLE="$T/opencode" \
  "$PY" scripts/run_tests_per_module.py --jobs 6 --timeout 900 --report "$T/report.json" --log-dir "$T/logs"
cp "$T/report.json" ~/ni-rua-linux-report.json && rm -rf "$T"   # keep the report, remove everything else
```

- `git archive HEAD` tests the committed tree only: commit what you want tested.
- `sudo` is used only for `setpriv --clear-groups`: it drops the user's supplementary groups so that bubblewrap gets an unprivileged user namespace (`bubblewrap_namespaces`). `sudo` resets the environment, so every variable is named, `USER` and `LOGNAME` included (`sudo` sets them to `root`).
- `$T/site` holds only the claude extra's pins; none of them is in `requirements.lock.txt`, so it shadows nothing of the environment.
- `HOME` and `TMPDIR` in the temporary directory keep the run away from the user's own `~/.codex`, `~/.claude` and `~/.config/opencode`. `TMPDIR="$T/tmp"` is deeper than `/tmp` on purpose; the suite passes with it, and also with the checkout as the parent of `HOME` and `TMPDIR` (Unix socket paths: see [Host Capabilities](#host-capabilities)).
- `--jobs 6` or fewer on a shared machine; `CUDA_VISIBLE_DEVICES=` keeps the suite off GPUs that other jobs use (no test needs one). The suite does not depend on how it is launched: tests that send SIGHUP or SIGINT set the dispositions they model, so `nohup` or a background job is fine.
- Remaining skips, each with its reason: checks that need git history (`git archive` has no `.git`), RoboDojo and garment installs, the opt-in host end-to-end (`NI_RUA_HOST_E2E=1`) and slow tests (`NI_RUA_SLOW_TESTS=1`), and every test that calls a real, charged model. On macOS, also set `NI_RUA_OPENCODE_EXECUTABLE` (e.g. `~/.local/share/ni-rua/opencode-2.2.7/opencode`), or every OpenCode-gated test is skipped and "green" says nothing about OpenCode.
- The X5 goldens and the X5 evidence corpus were recorded on macOS arm64 (`RECORDING_HOST`), where they are compared byte for byte. On another host [`embodiment_golden.host_noise`](../../tests/support/embodiment_golden.py) accepts floats within 4 ULP and a recorded Config path whose root that host resolves differently (macOS `/home` -> `/System/Volumes/Data/home`), and [`host_equivalent`](../../tests/robot/embodiment/test_embodiment_perception_evidence.py) additionally an artifact reference whose content is proven equivalent (a PNG with the same pixels, a JSON document written by the same serializer; JPEG overlays keep their hash) and anti-aliased edge pixels off by one level. The items a known host tolerates are pinned (`HOST_TOLERANCE` in both files): another tolerated item fails until it is reviewed. Every run prints the tolerated items of each golden family and of the evidence corpus, also when green.
- A `.git` entry above the run (even an empty one, such as a shared host's `/tmp/.git`) puts no host repository data into the actor's context: Claude Code runs with `CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS=1` (no gitStatus block) and Codex with `project_root_markers = []` (no `AGENTS.md` from above the actor directory). [`test_agent_prompt_parity.py`](../../tests/cortex/agents/test_agent_prompt_parity.py) asserts it unconditionally and with planted hosts (`*HostRepositoryBoundary`: a real repository and an empty `.git`; first request, subagent, compaction and second-turn requests). What still differs from a reference run below no planted host is pinned exactly: Claude Code `Is a git repository` and its memory directory's name, OpenCode `Is directory a git repo` and the workspace root; never repository content. Those lines are derived for every capture from the `.git` above its own directory, the reference's included, so the test also holds where the temporary directory itself is below a `.git` (an empty `/tmp/.git`: Claude Code then says `Is a git repository: true` in the reference too, and only the memory directory's name differs).
- Port-watching software on a shared host may probe every new loopback listener (for example, a `Go-http-client/1.1` sends `GET /` to `127.0.0.1:<port>` and `localhost:<port>`). The fake upstreams keep such a body-less `GET /` apart from the requests a test counts ([`is_host_port_probe`](../../tests/support/fake_model_servers.py)); a test with its own listener must tolerate a connection that is not its client.

## Browser and Native Evidence

The synthetic-only mode of `web/check-task-review.mjs` injects synthetic data in a headless browser without launching a model or robot. Build the web assets first, then run with an installed Playwright setup (setting `PLAYWRIGHT_MODULE` to your local module path if needed):

```sh
NI_RUA_BROWSER=firefox \
NI_RUA_URL=http://127.0.0.1:18787 \
NI_RUA_PREVIEW_ASSETS="$PWD/web/dist" \
NI_RUA_SYNTHETIC_ONLY=1 \
NI_RUA_SCREENSHOT_DIR=/tmp/ni-rua-browser-check \
node web/check-task-review.mjs
```

This check validates synthetic Web UI states, interactive controls, and read-only boundaries, but does not prove live task replay synchronization or GPU execution. Changes to physical motion semantics additionally require running on explicitly authorized hardware with frozen source code, configuration, provider, model, task, seed, and layout, recording the complete action requests, execution receipts, measured trajectories, independent native evaluator scores, original/replay media, and audit results. Passing software tests, launching a queue, model self-reports, or producing a video file are never substitutes for verified task success, and prior experiments do not automatically serve as evidence for modified source code.

`audit_run.py --profile native` normally requires a completed Codex turn to satisfy session checks; if `Harness` interrupts an active turn after a native terminal state has already been recorded, the audit accepts it only when the archive simultaneously proves that the terminal state came from the private native evaluator, the turn had started on the same thread, and the owner's interrupt request occurred after the native terminal timestamp. Such turns are counted under `native_terminal_interrupted_turns` without incrementing `native_completed_turns` and do not imply task success; a standalone `interrupted` turn never satisfies the native audit. `--require-continuity` still requires at least two completed turns.

Evaluation reports preserve `null` for missing, invalid, or unknown runs against a fixed case denominator, detailing how many runs were scorable alongside their failure categories. Integrity audits, raw RGB byte verification, reconstructed image availability, and task evaluation scores are reported as separate dimensions. After importing legacy archives, verify both key native summary fields and byte-level immutability of the source directory.

In `benchmark_report.py`, per-run `recovery_guidance` distinguishes the manifest's `configured_version`, the count of `benchmark.actor_turn_continuation` attempts on the primary thread, and the count of matching `sent turn/start` requests in `codex-raw.jsonl`. After restoring short references via the saved alias map, the reporter performs an exact match between the full event prompt and the `text` input items of native requests: each request is matched at most once, and secondary threads, `received` log entries, or partial prompt substrings are excluded. `versions` breaks down attempt and sent counts by the recovery version actually recorded in each event rather than backfilling from the configured version, and separates standard continuations from continuations carrying recovery context. Reports never copy raw prompt bodies or hashes.

`native_sent_recovery_count` only proves that the log recorded a request submitting that continuation prompt to the native runtime—it does not prove that the upstream accepted the request, that the model followed the guidance, or that task performance improved. `unmatched_continuation_attempt_count` indicates recorded continuation attempts for which no corresponding native send entry was found. For older archives lacking raw logs or primary thread identity, sent counts remain `null`; when both the primary thread identity and complete readable logs exist with zero matching events, the count is `0`. Malformed JSON-RPC log entries remain isolated in the report denominator under existing corrupted-archive rules rather than being coerced to zero triggers (see `apps/library/test_experiments_and_reports`).

## Release Checklist

Before cutting a release, save any uncommitted work and branch baselines, confirming the independent changes on both target `main` and the branch being merged. Use a standard merge commit to preserve ancestry rather than `reset` or `force push`, resolving conflicts according to module ownership and test coverage. Existing release tags are immutable annotated tags and must never be moved.

On the final merged tree, bump the package version in `pyproject.toml`, update the release notes, and run the complete software, web, and wheel verification suite. Verify the wheel metadata version, the commits pointed to by the previous and new tags, that both the prior `main` HEAD and merged branch HEAD are ancestors of the new `main`, that features unique to `main` are preserved, and that the working tree is clean. Create the new annotated tag only on the verified commit, push normally, and inspect remote refs to confirm. Synchronizing a Git tag and package version does not publish to PyPI; external deployments and official benchmark runs are separate deliverables.
