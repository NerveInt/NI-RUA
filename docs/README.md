# Documentation Index

[English](README.md) · [简体中文](README.zh-CN.md)

If you are new to NI-RUA, start with the main [README](../README.md) ([English](../README.md) / [简体中文](../README.zh-CN.md)).

The tables below organize all documentation by topic. Development rules and architectural boundaries are governed by [AGENTS.md](../AGENTS.md), while the authoritative runtime tool contracts live in [`ni_rua/runtime/tools/definitions.py`](../ni_rua/runtime/tools/definitions.py) and [`ni_rua/runtime/tools/builtins.py`](../ni_rua/runtime/tools/builtins.py).

---

## 🌟 SO-101 Real-Robot & Core Guides (Bilingual)

| Guide | English | Simplified Chinese | Covers |
| :--- | :--- | :--- | :--- |
| SO-101 Quickstart | [so101-quickstart.md](guides/so101-quickstart.md) | [so101-quickstart.zh-CN.md](guides/so101-quickstart.zh-CN.md) | Hardware/power checklist, `ni-rua` & `so101` setup, camera `unique_id` binding, pure local CPU vs. optional remote GPU perception/IK, commissioning, and safety controls. |
| SO-101 Calibration Guide | [so101-calibration.md](guides/so101-calibration.md) | [so101-calibration.zh-CN.md](guides/so101-calibration.zh-CN.md) | ChArUco board selection, 1:1 A4 printing & mounting, guided S0–S6 calibration workflow (`/calibration`), and post-calibration troubleshooting. |
| Agent Plugin (`ni-rua-plugin`) | [agent-plugin.md](development/agent-plugin.md) · [CLI README](../agent-plugin/README.md) | [agent-plugin.zh-CN.md](development/agent-plugin.zh-CN.md) · [CLI README (zh-CN)](../agent-plugin/README.zh-CN.md) | Driving runs with your own Claude Code, Codex, or OpenCode (`--agent-backend external`), `run_begin`/`run_finish`, console behavior, and verification status. |
| Running Guide | [running.md](development/running.md) | [running.zh-CN.md](development/running.zh-CN.md) | Environment setup, wheel packaging (`scripts/build_package.py`), single runs, stopping/resuming, replay, and batch evaluation. |
| Configuration Guide | [configuration.md](development/configuration.md) | [configuration.zh-CN.md](development/configuration.zh-CN.md) | Unified configuration precedence, model settings, OS sandbox, feature flags, remote GPU services, and agent backend options. |
| RoboDojo Setup | [robodojo-setup.md](development/robodojo-setup.md) | [robodojo-setup.zh-CN.md](development/robodojo-setup.zh-CN.md) | Pinned RoboDojo / IsaacLab / cuRobo commits, required compatibility patches, and locating the isolated simulation interpreter. |

---

## 🚀 Getting Started & Operations

| Document | What It Covers |
| :--- | :--- |
| [Running Guide](development/running.md) | Preparing the environment, starting interactive or benchmark runs, pausing/stopping, resuming sessions, replaying archives, and scheduling batch evaluations. |
| [RoboDojo Environment Setup](development/robodojo-setup.md) | Pinned RoboDojo / IsaacLab commits, required compatibility patches, and how NI-RUA locates the isolated RoboDojo interpreter. |
| [SO-101 Integration](development/so101-integration.md) | Single-arm SO-101 hardware integration: real-robot studio (`/console`), deterministic [camera binding](development/so101-integration.md#34-camera-binding-unique_id-device_name-and-index_or_path) (`scripts/so101_cameras.py`), guided S0–S6 calibration (`/calibration`) with S6 height & XY checks, CAD [grasp-centre input](development/so101-integration.md#49-grasp-centre-input-grasp_centre_m), `mode=reach` & optional cuRobo IK `path_check` (§12), `transit` sag telemetry & `motion_compensation` (§4.12), bounded holding squeeze `grip_squeeze_rad` (§4.13), FK mesh occlusion / cross-view / `contact_top_z` ([perception bench](development/so101-perception-bench.md)), motion-only H.264 video export, Agent bar handoff, speed-tier probing, and remote SAM/VGGT tunnel status. |
| [SO-101 Calibration & Evidence Configuration](development/so101-configuration.md) | File ownership, explicit overrides, effective settings, SAM/LK/VGGT, observation-bound support height and [legacy helper settings](development/so101-legacy-calibration-settings.md). |
| [Configuration Guide](development/configuration.md) | Configuring models (`verbosity`), machine paths, the OS sandbox, startup inline grounding, observation/action image feedback policies, tool availability, video recording, the agent backend (`agent.profiles_dir`, `agent.codex_upstream`, `upstream.source="profile"`), and its [evidence helper client](development/configuration.md#evidence-helpers-agentevidence). |
| [Experiment Library & Web Service](development/experiment-library.md) | Archiving, searching, live monitoring, side-by-side comparison, replay, and the optional `experiment-groups.json` registry format. |
| [HD Offline Replay](development/hd-replay.md) | Re-simulating archived trajectories in RoboDojo and exporting high-definition videos without calling the model. |
| [Experiment Completion Hook](development/experiment-completion-hook.md) | How the evaluation queue notifies callers when a batch finishes, and the [per-host wake recipe](development/experiment-completion-hook.md#waking-the-agent-that-launched-the-queue-per-host) for Claude Code and OpenCode. |
| [Agent Plugin Spikes](development/reports/agent-plugin-spikes.md) | Measured host facts behind the agent plugin: plugin formats, permission and MCP-client defaults, HTTP-first go/no-go, delivered lengths, recovery, wake, and the per-host end-to-end results. |
| [Agent Plugin](development/agent-plugin.md) | Driving a run with your own Claude Code, Codex or OpenCode: the `external` backend, the npm package and installer, the `run_begin` / `run_finish` bracket, console behaviour, recovery, residuals and verification status. |
| [Benchmark OOM Retries](development/benchmark-oom-retries.md) | When and how the queue schedules a fresh episode retry after a native GPU out-of-memory failure. |
| [Token & Cost Accounting](development/token-accounting.md) | How token usage is metered and converted into cost estimates via pricing cards. |
| [Web Workbench & Validation](../web/README.md) | Building the frontend bundle and validating the terminal, phase timeline, and video replay in headless browsers. |

---

## 🛠 Architecture & Contributor Guides

| Document | Purpose |
| :--- | :--- |
| [AGENTS.md](../AGENTS.md) | Essential reading for developers and coding agents: architectural boundaries, first principles, experiment discipline, and delivery checks. |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Environment setup, CPU/web/package checks, and pull request guidelines. |
| [Implementation Architecture](development/architecture.md) | Six-layer package structure, call paths, state ownership, and module dependency rules. |
| [Embodiment Profile](development/embodiment-profile.md) | The per-run `EmbodimentProfile` (fields and units), how a run gets one, how the schema, tool text and instructions are rendered from it, threshold configuration, the `supports` tags (including `reach` and `approach_roll`), how to add a robot, the X5 golden fixtures and allowance mechanism, and the user-approved X5-visible changes. |
| [Agent Backends](development/agent-backends.md) | The swappable agent runtime (Codex / Claude Code / OpenCode, plus the [`external`](development/agent-backends.md#external-backend-agentbackend--external) backend where your own agent connects): [provider profiles](development/agent-backends.md#provider-profiles) (format, import command, one auth scheme per wire, model catalogue cache, reasoning options), the console [Agent bar, `apply_agent_settings` and `handoff_run`](development/agent-backends.md#console-agent-bar-live-settings-and-handoff) with their refusal codes and archive records, `AgentSession` contract, MCP tool transport, relay wire adapters, native capability policy per backend, OpenCode 2.2.7 install, config and runtime contract (turns, subagents, model declaration and switching), comparison protocol for harness experiments, and known limitations; [prompt parity](development/reports/agent-backend-prompt-parity.md) and [spike findings](development/reports/agent-backend-spikes.md). |
| [Extension Guide](development/extensions.md) | Choosing the right extension layer: operation memory, task notes, plugins, tools, or robot backends. |
| [Tool Contracts & Plugins](development/tool-plugins.md) | Declaring schemas, `focus_3d` 1:1 local crops, `reanchor_tracks`, roles/effects, narrow `ToolContext` access, typed receipts, and dual actor/program regressions. |
| [Tool Output Contract](development/tool-output-contract.md) & [Cost Summary & Input Diagnostics](development/tool-input-output.md) | Structured tool responses, on-demand `help` requests, preview cost summaries, and actionable validation errors. |
| [Robot & Environment Integration](development/robot-environment-integration.md) | Onboarding a new manipulator or simulator/real-world environment (including its embodiment profile) and validating supported combinations. |
| [Robot Backend Boundaries](development/robot-backend-boundaries.md) | Separating pure validation, trajectory compilation, IPC transport, and native worker execution. |
| [Validation & Release Guide](development/validation.md) | Selecting test suites by change scope, [running the full CPU suite one module per process](development/validation.md#running-the-full-cpu-suite) (`scripts/run_tests_per_module.py`), the opt-in [live provider end-to-end test](development/validation.md#live-provider-end-to-end) (profiles, handoff chain), host-capability gating of sandbox tests, the [Linux host recipe](development/validation.md#running-the-suite-on-a-linux-host), distinguishing software checks from native task evidence, and cutting releases. |
| [Changelog](../CHANGELOG.md) | Version-by-version release notes and compatibility history. |

---

## 🤖 Runtime Interfaces & Manipulation Behavior

| Document | Topic |
| :--- | :--- |
| [Unified Actions & Program Review](development/unified-actions.md) | Task plans (`plan_update`), multi-stage `robot_action` batches (`preflight`, `execute_if_clean`, `preview`), `mode=reach` IK queries, split orientation tolerances & SO-101 `grasp_centre_m` input, `image_delivery` & 4D `process_trajectory` receipts, inter-arm dependencies, program review, and operation memory. |
| [Program & Task Result Semantics](development/program-result-semantics.md) | Distinguishing program exit status, physical motion completion, and native evaluator task scores. |
| [Robot Information Interface](development/robot-information.md) | Robot kinematic structure, URDF, native meshes, camera calibration, and physical specifications. |
| [Actor Information Boundary](development/actor-information-boundary.md) | What data the main model, auxiliary models, and helper programs are permitted to read—and how provenance metadata is stripped (including the native agent's working-directory exception). |
| [Image Pipeline & Storage](development/image-pipeline.md) | Model-bound images, archive images, compression, deduplication, and video encoding. |
| [Adaptive Camera View Selection](development/adaptive-view-selection.md) | How post-action feedback selects relevant camera views. |
| [Dual-Arm Synchronized Motion](development/dual-arm-motion.md) & [In-Motion Gripper Commands](development/synchronized-gripper.md) | Coordinated dual-arm trajectories, handovers, and synchronized gripper actuation during motion. |
| [Per-Action Motion Speed](development/motion-speed.md) | Semantics, bounds, and defaults of stage speed scaling factors. |
| [Motion Planning Failure Feedback](development/motion-failure-feedback.md) | Planning diagnostics, candidate trajectory feedback, and partial execution receipts. |
| [Joint Travel Admission & Trajectory Evidence](development/motion-safety-and-trajectory-evidence.md) | Principal-branch joint constraints, safety admission checks, and executed trajectory evidence. |
| [Motion Cost & Contact Overlay](development/motion-cost-and-contact-overlay.md) | Preview cost summaries and `robot_contact_overlay` visual projections. |
| [Model-Led Operation](development/model-led-operation.md) | Division of responsibility between model transport, risk reference summaries, and optional visual verification. |
| [Turn Progress Supervision](development/actor-progress.md) | Bounded continuation policy when an actor ends a benchmark turn prematurely. |
| [Wrist Camera Viewpoint Solver](development/robotic-view.md) & [Stereo Epipolar Guide](development/stereo-guide.md) | Two optional geometric tools (disabled by default). |
| [RGB-D Numeric Tools](development/rgbd-numeric-tools.md) | Depth queries, limitations, and configuration under explicit RGB-D observation conditions. |
| [$\pi_{0.5}$ Inference Service](development/pi05-inference-service.md) | Interface and execution boundaries for the optional local policy bridge (disabled by default). |
| [Official Public Task Rules](development/public-task-rules.md) | Verbatim snapshot, prompt injection, and backward compatibility for the 42 official RoboDojo task specifications. |
| [Default Task Notes](development/task-notes.md) | General manipulation notes injected at startup, versioning, and feature toggles. |

---

## 👁 3D Perception & Context Notes

| Document | Topic |
| :--- | :--- |
| [obs3d](obs3d.md) | Startup inline `obs_1` grounding, zero-turn 3D scene perception attached to default observations, and operator SAM overlays, VGGT estimates, and saved replay. |
| [object_table](object_table.md), [scene3d](scene3d.md), [vggt_height](vggt_height.md) | 3D object table (with opt-in persistent IDs & `reanchor_tracks`, SO-101 [cross-view check](object_table.md#cross-view-check), `contact_top_z`, and `robot_occlusion`), point-cloud handles (`scene3d_helpers`, `cavity_profile`, `PointCloudSpatialField`, opt-in multi-view fusion), and support-plane height estimation via the [VGGT Service](../services/vggt_height/README.md). |
| [held_perception](held_perception.md) | Opt-in in-hand object state and geometry perception; FK robot pixel mask from the arm meshes on ARX-X5 and SO-101. |
| [pose_skill](pose_skill.md) | Opt-in pose targets and grasp geometry references (disabled by default). |
| [lean_context](lean_context.md) | Prompt-cache-preserving history projection (disabled by default). |
| [scene_truth](scene_truth.md) | Observer-only ground-truth recording for offline geometric auditing (never exposed to the model). |

---

## 📐 Historical & Foundational Design Specifications

The documents in `docs/design/` preserve the foundational architecture specifications. Where early design drafts mention standalone motion tools, templates, skills, or `scene_*` tool names, the current implementation has converged onto the unified `robot_action` and program review interface described above.

| Document | Scope |
| :--- | :--- |
| [Embodied Harness Design](design/embodied-harness-design.md) | Core philosophy, division of labor between the model and executor, observation boundaries, programs, and long-term roadmap. |
| [Shared Interface & Event Protocol](design/shared-protocol.md) | Run and action identities, `ObservationBundle`, `EEAction`, event streams, snapshots, and media references. |
| [Web Workbench & Codex Integration](design/web-workbench-and-codex.md) | Coupling native Codex sessions, PTY terminals, multi-camera robot views, and phase-synchronized history inspection. |
| [Local Scene & Short-Horizon Rollout](design/local-scene-and-rollout.md) | Constructing queryable 3D scenes and snapshots from permitted observations (full physical rollout worker remains future work). |

> [!NOTE]
> The live Cortex tool catalogue and input validation are assembled from [`ni_rua/runtime/tools/builtins.py`](../ni_rua/runtime/tools/builtins.py) and [`ni_rua/runtime/tools/definitions.py`](../ni_rua/runtime/tools/definitions.py). Availability at runtime depends on configuration, observation conditions, backend capabilities, and external service readiness. Models can pass `request={"kind": "help"}` to supported query tools for on-demand field documentation.
