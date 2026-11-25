# Tool Output Contract and On-Demand Details

Tool execution, raw receipts, and stored evidence are kept strictly separate. Routine delivery to the model provides the essential information needed to decide the next step, while complete evidence remains accessible via explicit detail parameters, pagination, or artifact references. Output projection never executes robot actions, advances simulation steps, or judges grasp retention, physical contact, or task success.

## Internal Results Do Not Depend on Display Serialization

Tool handlers return a `ToolResponse` that preserves wire keys while exposing structured plans, raw action receipts, or `ObservationBundle` instances to the broker through internal `kind`/`data` attributes. Review workflows, `action_summary`, and `final_observation_id` in `ProgramRunner` do not parse `contentItems` or trust program `stdout`. External JSON responses retain the original `{success, contentItems}` envelope; typed internal attributes neither expand the client wire protocol nor leak hidden metadata.

`ToolOutput`, model summaries, image compositions, and artifact references handle presentation, while raw physical execution results are persisted independently. Once a handler has triggered a physical action, a failure in optional display or feedback assembly must never hide the receipt or suggest replaying the action. See [Tool Registration and Extensions](tool-plugins.md) for the full extension contract.

## Shared Components

`ni_rua/core/tool_output.py` provides the lightweight `ToolOutput(summary, details=None, extra_content=())` helper. `content_items()` produces the primary JSON text item, an optional detail text item, and any trailing images; `response()` produces the standard `{success, contentItems}` envelope without adding mandatory wrappers around existing `{request, receipt}` payloads. `select_fields` copies only fields that are actually present without backfilling missing values; `DETAIL_SCHEMA` and `resolve_detail` share detail-level selection; `validate_page` validates integer pagination; and `bounded_text` retains head/tail text slices while reporting original byte ranges and omitted byte counts.

Public tools declare their inputs, typed outputs, roles, and side effects via `ToolSpec`, and `ToolRegistry` generates the catalog and enforces validation uniformly. Domain layers own their result semantics. The table below also lists retained internal compatibility modules; listing them here does not mean legacy templates or skills are re-exposed to the actor:

| Return Path | Owning Module | Default Presentation |
| --- | --- | --- |
| Manipulation templates | `manipulation_result.py` (removed) | Phase, action status/reason/executed counts, observation IDs, and unverified object state, followed by references to the full receipt and program output |
| Standard actions | `actor_payload.py` | Preserves request, execution result, and errors; moves repetitive numeric checks from known passing guards behind an explicit full-receipt read handle |
| History and status | `history_output.py` | Bounded action index, status summary, and paginatable exact traces |
| Programs and fixed skills | `program_result.py` | Process status and log head/tail excerpts; known fixed skills can project structured execution results, while arbitrary Python `stdout` remains opaque |
| Visual observations | `observation_output.py` | Measured robot state, current-frame camera world poses, current vs. historical flags, and remaining budgets; static intrinsics and joint conventions are delivered in full once and referenced by version thereafter |
| Capability discovery | `capability_output.py` | Returns full capabilities on first read; subsequent reads in the same context return the version and current budgets, re-delivering in full if anything changes |
| Action trajectories | `trajectory_feedback.py`, `feedback_delivery.py` | Default `automatic`: sampled planned/measured joints, measured trajectory overlay image, tracking deviation, and full-trace reference; failures and handoff diagnostics add planned and local candidate overlays; optional `diagnostic` preserves the no-image routine success summary |
| Candidate trajectories | `trajectory_evidence.py`, `manipulation_tools.py` | Key poses, minimum model clearance, and anomalies; the full candidate command sequence is stored in an artifact for paginated reading |

## Input Conventions and Compatibility

`robot_capabilities`, `robot_observe`, `history_view_observation`, `robot_action_status`, `history_list_actions`, `history_read_trace`, `program_run`, and `program_status` accept an optional `detail="summary" | "full"`. These are additive optional parameters; legacy argument shapes remain valid. When internal program calls omit explicit summary or pagination parameters, they receive the full structure so existing numerical scripts continue to work unchanged. The internal state interface and raw logs of `ProgramRunner` remain untouched, and the model presentation layer reads complete artifacts independently.

History lists and traces use `start` / `limit` pagination (default `25`, maximum `100`), returning `total` and `next_start`. In `history_read_trace`, `path` is a JSON Pointer such as `/segments/0/q_rad`. Pagination never reorders, rounds, or passes off subsampled points as a complete trajectory. `detail="full"` cannot be combined with pagination selectors and restores the complete original payload. Candidate trajectories are indexed by native step number and retain their own `native_steps` selection and pagination rules; see [Joint Travel Admission and Candidate Trajectory Evidence](motion-safety-and-trajectory-evidence.md).

For example, call `history_list_actions({"start": 0, "limit": 25})`, follow `next_start`, or use the per-entry `full_result` handle to inspect a single action in full without expanding every receipt in the episode. Inside `program_run`, read complete artifacts via `NiRuaClient().artifact(ref)`; read-only queries never consume action steps. Full-resolution images remain retrievable per camera via `history_view_observation`; image selection and text summarization operate independently.

## Intra-Session Reuse and Trajectory Delivery

`MetadataDelivery.prepare/commit` reuses only known static fields that have already been delivered in a committed model reply. Static camera intrinsics, distortion coefficients, joint orderings, and coordinate conventions are versioned by exact value; they are re-delivered in full whenever values change, the environment resets, the actor thread changes, native context compaction occurs, or the model transport resets. Unknown fields, provenance flags, dynamic camera extrinsics, joint/gripper measurements, history flags, and current budgets are always inlined. `metadata_delivery.static_refs` points to the earlier observation and version, and passing `detail="full"` on the current observation restores every field. Explicit full reads, internal program observations, and superseded `robot_wait` temporal samples do not consume the model's first-delivery baseline. If context invalidation occurs while assembling a reply, delivery falls back to inlining full metadata.

`static_refs.bundle` is a version reference for a shared set of observation conventions covering explicitly enumerated run/environment identifiers, static camera dimensions and pixel conventions, clock units, and budget/geometry descriptions repeated across fields (it does not correspond solely to a `bundle` sub-object). Cameras remain keyed by ID with list order and camera IDs always inlined, and live timestamps, synchronization/availability flags, measurements, and budget numbers are sent every time. Reuse is enabled only when doing so reduces total byte size after accounting for the reference itself; unknown descriptions or changed values trigger full inlining.

By default, `features.action_trajectory_delivery="automatic"` returns sampled planned/measured joints and a measured trajectory overlay image with each action. Routine completed actions render only the measured path while still returning planned joint samples and tracking deviations in the payload (the planned-path overlay remains available on demand), preserving full panel resolution. The optional `"diagnostic"` mode omits the routine overlay image only when execution completed, native timelines are synchronized, planned and measured samples are complete, geometry and result observations are available, and no remaining candidates or anomalies exist. Both modes preserve the exact `trajectory_feedback_ref` artifact containing the full joint sequences and public forward kinematics. Failures, partial execution, unknown outcomes, missing samples or geometry, and handoff checks continue to return detailed diagnostics and available trajectory overlays. Action completion still does not prove grasp retention, post-release stability, or task success; always verify against current RGB images and object states.

Calling `robot_action({"mode": "inspect", "trajectory_ref": "<trajectory_feedback_ref>", "path_kind": "measured", "camera_id": "cam_head"})` re-renders the trajectory of an executed action on demand; `path_kind="planned"` inspects archived commands, and `local_N` inspects local candidates listed in `available_paths`. The response preserves exact measured gripper positions, `sim_step`, per-path timing semantics, and native boundary alignment flags, supporting existing sample selectors, pagination, and `detail="full"`. Overlays are rendered onto the execution-start observation and camera pose and are explicitly labeled as historical—they do not reconstruct the post-action scene. Omit `path_kind` when inspecting a candidate preview; rendering a candidate overlay still requires the preview to be bound to the current observation. None of these inspection calls execute robot motion.

## Actor Reply Delivery

Six settings shape what an actor tool reply looks like after presentation. They are on by default, apply to every agent backend (Codex, Claude Code, OpenCode, external MCP hosts; all tool calls go through `Harness.handle_tool`), and each can be turned off on its own. With all six off and `features.actor_payload_profile="lean_v1"`, instructions, tool catalogue and replies are byte for byte what they were before these settings existed (`robot.embodiment.test_embodiment_contracts.ReplyDeliveryDefaultsTests`).

| Setting | Default | Off value | Effect |
|---|---|---|---|
| `features.budgeted_action_reply` | `true` | `false` | Action replies laid out by decision value; fitted to the host's text limit when one is known |
| `features.action_reply_profile` | `summary_first` | `full` | Fixed shape of an execution reply: bulky blocks by reference |
| `features.reply_static_once` | `true` | `false` | Fixed explanatory notes stated once in the catalogue; one `detail` read entry |
| `features.reply_compact_encoding` | `true` | `false` | Listed numeric structures as column tables; an unchanged arm once; derived quantities at 4 significant digits |
| `features.reply_same_as` | `true` | `false` | A value repeated inside one text item printed once |
| `features.reply_float32` | `true` | `false` | Float32-exact numbers printed with their shortest decimal |

All are CLI flags (`--[no-]budgeted-action-reply`, `--action-reply-profile`, ...) and queue case options. Program callers (`program_run`), typed `ToolResponse.data` and stored evidence are never changed by any of them. These are delivery measures: on the archived tool replies of one full benchmark arm (47 episodes, 610 execution replies) the default form brings the text of an execution reply from a median of about 49 kB (56% over the Codex host limit, where the host then cut the post-action state or the object table out of roughly half of them) to about 21 kB (none over the limit, both delivered whole in every reply), and all tool-reply text of an episode from about 0.95 MB to about 0.49 MB. Their effect on task success is measured separately and is not established by these figures, and there is no evaluation on a physical SO-101.

### Host limits for one tool result

An agent host may cut the text of one tool result before the model sees it. `AgentSession.tool_output_text_budget_bytes()` states that limit in UTF-8 bytes, or `None` when no limit is known. The Harness lays action replies out on every backend; it replaces blocks by references only against a limit stated here.

| Backend | Budget | Rule and evidence |
|---|---|---|
| Codex (CLI 0.160.0) | 48,000 bytes | 1.2 × the model's `truncation_policy` limit (10,000 tokens) at one token per started 4 UTF-8 bytes of a text item; images cost nothing. Items are taken in order; the first text item that does not fit is cut in its middle (`…N tokens truncated…`) and every later text item is dropped. Measured with forked requests: 47,984 + 16 bytes pass, 47,996 + 16 do not; a fork replays recorded outputs under the same limit. `codex_tool_output_budget_bytes(model)` reads the pinned catalogue entry, and the session writes `tool_output_token_limit` into the native configuration. |
| Claude Code (CLI bundled with the pinned SDK) | 500,000 bytes | With this harness's settings (`MAX_MCP_OUTPUT_TOKENS`, the `anthropic/maxResultSizeChars` hint on every tool) a result whose text blocks total at most 500,000 characters reaches the model whole, block by block; above that the whole result becomes one `<persisted-output>` block (a file path and the first ~2,000 characters). A result that carries an image is kept whole beyond it. Measured with a scripted model and the next `/v1/messages` request read back: 250,000 + 250,000 whole, 250,000 + 250,100 replaced, 600,000 + image whole (`cortex.agents.test_agent_prompt_parity.ClaudeCodeToolResultLimit`). Bytes never undercount characters, so the budget is on the safe side. |
| OpenCode 2.2.7 | none | MCP tool results arrived whole, text item by text item, up to 600,000 bytes (3 × 200,000 with an image; one item of 400,000; 3,000 lines), on both upstream wires. No limit was found in that range (a 1 MB result did not complete within the probe's time and was not characterised), so replies are laid out and nothing is replaced (`OpenCodeToolResultLimit`). |
| external | none | The host is whatever MCP client connects; its limit cannot be known to the Harness. Replies are laid out and nothing is replaced. |

### Layout by decision value (`budgeted_action_reply`)

Without layout a `robot_action` execution reply is `[result JSON, overlay images, observation JSON, camera images, object-table JSON]`. With the setting on, `Harness.handle_tool` passes every presented reply of `kind="action"` through [`fit_reply`](../../ni_rua/core/presentation/reply_budget.py):

1. execution result: the result JSON without per-stage detail and trajectory feedback. Stage entries that did not complete cleanly and motion-safety rows that did not pass stay here.
2. `object_state_delta` + `object_table` (+ `scene3d` handle), followed by its overlay image when one is delivered.
3. post-action observation (robot state, camera metadata), followed by its camera images; `robot_wait` samples keep their place before it. Never reduced, so `metadata_delivery.static_refs` point at text that was delivered.
4. `{"action_id", "stages"}`: the cleanly completed stages.
5. `{"action_id", "trajectory_feedback"}`, followed by the trajectory overlay images.

Only against a known host limit, and only while the text is over it, blocks are replaced, least valuable first, by `{"omitted": "by_reference", ...}`; afterwards the most valuable replaced block that still fits is given back. No value is rounded, renamed or deleted, and a block is replaced only if an existing entry returns it:

| Block (replacement order for a clean completed action) | Replaced by | Read with |
|---|---|---|
| passing rows of `receipt.motion_safety_checks` (the summary and every non-passing row stay) | `receipt.motion_safety_checks_by_reference` | `robot_action_status(action_id, detail="full")` → `receipt.motion_safety_checks` |
| `trajectory_feedback.planned.samples` | note in place | `robot_action(mode="inspect", action_id, path_kind="planned", native_steps=[<the omitted rows>])` |
| `trajectory_feedback.measured.samples` | note in place | `robot_action(mode="inspect", action_id, path_kind="measured", native_steps=[<the omitted rows>])` |
| `trajectory_feedback.compact_3d_trajectory` | note in place | `program_run`: `NiRuaClient.artifact(trajectory_feedback_ref)` → `summary.compact_3d_trajectory` |
| `stages` of cleanly completed stages | `stages_by_reference` (with `stage_ids`) | `robot_action_status(action_id, detail="full")` → `receipt.stage_results` |

The sample reads name the `native_step` of every row they stand for, because `robot_action(mode="inspect")` without a selector returns only the first, middle and last row; one call names up to `INSPECT_STEP_LIMIT` (25, `core/contracts.py`, also the `native_steps` and `max_samples` bound of the public schema). More omitted rows than that are read as pages (`start_native_step`, `max_samples=25`; follow `next_native_step`). A read always returns the rows the unshaped reply carried inline, with the same values.

After a failure, pause, unknown outcome or detected slip the order is passing safety rows, planned samples, completed stages' detail, measured samples, `compact_3d_trajectory`: the measured path outranks the detail of stages that did complete. If the replaceable blocks do not free enough, the reply is returned in this layout with `fits=false` and the host cuts it at its least valuable end. `fit_reply_report` returns what was done; the Harness records it as an internal `actor_reply.fitted` event whenever a block was replaced or the reply does not fit. The trajectory overlay image follows the trajectory JSON in this layout, so the relay's image-trimming placeholder labels it without an observation id.

### Summary-first execution replies (`action_reply_profile`)

`summary_first` gives every execution reply one fixed shape, with or without a host limit; if a limit is known and the fixed shape still exceeds it, replacement continues with the remaining blocks in the order above. `full` keeps every block inline unless the limit forces a replacement.

| Reply | Inline | By reference |
|---|---|---|
| Routine: clean `completed` receipt, `batch_status` completed, no `failure_feedback`, no `anomaly_reason`, trajectory feedback available; or a stop the model asked for that reports nothing else | execution result (receipt with status, reason, terminal residual, guard and contact summaries, budgets); every stage entry, a cleanly completed one without the fields of `STAGE_DETAIL_BY_REFERENCE` (`grasp_contact_feedback`); `object_state_delta` + `object_table`; post-action observation and images; trajectory feedback with `compact_3d_trajectory` | measured samples, planned samples, passing motion-safety rows, `grasp_contact_feedback` of cleanly completed stages |
| Exception: receipt failed or unknown (`tracking_error` ...), a stop by a guard, `failure_feedback`, `anomaly_reason` (slip), or trajectory feedback unavailable | the same, plus the measured samples of the affected step window: every sample whose `native_step` lies in it and the one sample before and after it. Stages that did not complete cleanly and safety rows that did not pass are part of the execution result in every profile. | planned samples, passing motion-safety rows, `grasp_contact_feedback` of cleanly completed stages, measured samples outside the window (`samples_by_reference` with their count and the call that names exactly those rows' `native_steps`) |
| Exception without an identifiable window | the same, plus all measured samples | planned samples, passing motion-safety rows, `grasp_contact_feedback` of cleanly completed stages |

A stage entry keeps every other field on every robot: outcome, terminal residuals, `effective_tolerances`, resolved poses, grip and guard facts, step counts, and any field the profile does not know (the SO-101 `ik`, `transit`, `arrival_correction` and `grasp_centre` reports stay inline). The window spans the `compact_3d_trajectory.stages` rows of the affected stages (`receipt.failed_stage_id`, `receipt.checkpoint_stage_id(s)`, and every stage entry that is neither a clean completed one nor `pending`), the `slip_onset_step` of a detected slip, and, when the receipt status is not `completed`, the step execution stopped at (`receipt.completed_native_steps`). A program's pending review is not an action reply and is not affected. Without a host limit the `full` profile's replacement order never runs; where it does, it still treats every stop as an exception.

While the profile is `summary_first`, the `robot_action` description states this layout (`summary_first_catalogue_note`; it names the object table and `grasp_contact_feedback` only when the run's catalogue has them). The developer instructions are unchanged: what they say an action result carries (`grip_facts` of completed stages, the receipt, the updated observation, trajectory feedback, the object table) stays inline in this profile.

#### Requested checkpoints and guard stops

Every checkpoint pause has `status="stopped"`, `reason="observation_checkpoint"` and `paused_at_checkpoint=true` (`pause_receipt` in `robot/motion/batch_scheduler.py`), so those fields do not tell who asked for the stop. `receipt.checkpoint_reason_code` does, and it is the only field used (`feedback_delivery.MODEL_REQUESTED_CHECKPOINT_CODES`, `is_requested_checkpoint_receipt`):

| `checkpoint_reason_code` | Written when | Class |
|---|---|---|
| `observation_checkpoint` | the batch reached an `observe` stage the model put there | model-requested: routine |
| `jaw_checkpoint` | `checkpoint_after_gripper` paused after a jaw stage | model-requested: routine |
| `aperture_below_min_m`, `aperture_above_max_m` | grip guard | guard: exception |
| `carry_slip_aperture_below_min_m` | carry slip guard during a later move | guard: exception |
| any other value, or no code (older receipts) | cannot be told apart | exception |

A requested stop is routine only when it reports nothing else: batch `awaiting_observation`, stop confirmed, no `error`, `slip_detected_at`, `effective_grip_guard` or invalid `resume_validation`, every stage entry clean completed or `pending`, no `failure_feedback`, no `anomaly_reason` and trajectory feedback available. A requested checkpoint during which a slip was detected (`anomaly_reason`) is an exception. The same rule decides whether `reply_static_once` keeps the fixed notes.

### Fixed notes stated once and one detail entry (`reply_static_once`)

**`drop_static_once(items, plan)`** ([`reply_compaction.py`](../../ni_rua/core/presentation/reply_compaction.py)) leaves a value out of a reply only when all of the following hold:

1. its key and its exact value are in the run's `StaticOncePlan` (explanatory text that is the same in every reply: budget semantics and `unknown_values_are_null`, `collision_scope`, `step_semantics`, the `grasp_contact_feedback.interpretation` sentence, the joint-travel policy sentence, the fixed four `information_sources`, the fixed `trajectory_feedback` scope/timing notes, and the two `program_run` result notes). A producer that changes its wording, or a diagnostic that only shares the key name (a failure `interpretation`, another `collision_scope`), is delivered;
2. the same fact is stated where the model always has it: the `program_run` description for the two program notes, and for all others the note that `ToolRegistry.catalog(role="actor")` inserts into the `robot_action` description (before its input field guide);
3. the note applies to this robot. `static_once_plan(tools, capabilities, features)` builds the note and the set of omitted values from groups, and a group is stated, and its values left out, only when the run's `robot_action` description already speaks of what the group is about: native step budgets (`native step`), grasp-contact fields (`contact_region`), joint-travel admission (`joint_travel_preset`), `link6`; the trajectory groups need `features.trajectory_preview`; `collision_scope` is stated with the text the robot backend publishes in its capabilities and only that text is left out. A robot without those features (the SO-101) gets neither the sentences nor the omissions, and no robot-specific word reaches another robot's catalogue;
4. the tool call succeeded and `reports_exception` is false for every text item of the reply. A reply keeps every note when it has a receipt or stage entry that is none of (a) a clean `completed` one with `batch_status` completed, (b) a stop the model asked for or (c) a `pending` stage that never ran; or a non-empty `failure_feedback`, `anomaly_reason`, `blocking_risks`, `pending_review` or `error`; or unavailable trajectory feedback; or a program result whose `status` is not `succeeded`.

A dictionary or list that held nothing but such values is left out with them (`step_semantics`); a row is never removed from a list of objects, and a list of strings is shortened only by its listed items. Numbers, states, residuals, `effective_tolerances`, reasons, ids and references are never touched. Strings a robot backend writes per observation (camera `synchronization`) are not fixed notes and stay.

**`unify_detail_entries(items, direct_tools)`** rewrites the ways a reply says "read the rest here" into one field at the same place:

```json
"detail": {"tool": "robot_action", "arguments": {"mode": "inspect", "trajectory_ref": "artifact:ref_49", "path_kind": "measured"}}
```

| Replaced entry | Call |
|---|---|
| `trajectory_feedback.detail_access` (a sentence: the artifact holds every exact row) | the read of that artifact, `program_run` with `NiRuaClient.artifact(trajectory_feedback_ref)` (the whole document); only when `program_run` is not loaded, `robot_action` `mode="inspect"`, `trajectory_ref=<trajectory_feedback_ref>`, `path_kind="measured"`, `start_native_step=0`, `max_samples=25` (a page; follow `next_native_step`). `trajectory_feedback_ref` stays and the catalogue note says what the artifact holds |
| `next_action` when it is nothing but `{"mode": "inspect", "plan_id": ...}` | `robot_action` with those arguments (a continuation `next_action` is a command and stays) |
| `motion_safety_check_summary.full_result`, history `full_result`, `full_result` of a `program_run` reply | the named lookup (`robot_action_status`, `program_status`) |
| `metadata_delivery.read_full` | `history_view_observation` with `observation_id`, `detail="full"` |
| `read` (+ `at`) of a `{"omitted": "by_reference"}` stand-in of the reply layout | the named lookup, or the artifact read |

Entries name only tools the host has loaded (`ToolRegistry.loaded_tool_names`). `robot_action_status` and `program_status` are loaded up front on every host (they carry no `deferLoading`; the MCP hosts publish every tool eagerly anyway), so a lookup of either becomes a `detail` entry naming it, for example `{"tool": "program_status", "arguments": {"job_id": "job_3", "detail": "full"}}` for the `full_result` of a `program_run` reply; calling it as written returns the full result. An entry of a `by_reference` stand-in keeps `at` beside it, the dotted path in the call's `data` that the entry stands for. A lookup of a tool the host loads on demand (`deferLoading`, native tool search on Codex) is left as the producer wrote it, since the model may not be able to load it; an artifact read becomes a `program_run` call printing that part of the artifact. What is returned is unchanged. A place that already has a field named `detail` is left as it is.

### Compact encoding (`reply_compact_encoding`)

[`encode_compact_items`](../../ni_rua/core/presentation/reply_encoding.py) rewrites the canonical JSON text items of every actor reply:

1. **Derived quantities at 4 significant digits.** Only the fields of `DERIVED_QUANTITIES` inside a `derived_motion` object: `joint_velocity_rad_s`, `ee_linear_velocity_m_s`, `ee_angular_velocity_rad_s`, `jaw_velocity_m_s`, `joint_acceleration_rad_s2`, `ee_linear_acceleration_m_s2`, `ee_angular_acceleration_rad_s2`, `jaw_acceleration_m_s2`. They are finite differences over one recorded interval (0.04 s in the simulator) of float32 measurements (spacing 1.2e-7 near 1), so a velocity carries an error of about 4e-6 and an acceleration about 1.5e-4; rounding to 4 significant digits changes a value by at most 5e-4 of itself. Every other number, including measured `qdot_rad_s`, joint angles, poses, residuals, tolerances and apertures, is left as presented; a field that is not in the table is treated as a measurement.
2. **An arm that did not move is stated once.** In an object with `samples`, an arm whose every number outside `derived_motion` stays within one float32 step (`2**-23 * max(1, |value|)`) of the first sample, whose strings and keys are identical, and whose derived quantities are no larger than such a step over the interval gives (so a jaw that moves keeps the arm in the samples), moves to `unchanged_arms[arm]` with `"unchanged": true` and `max_abs_joint_change_rad`, and leaves the samples.
3. **Tables, for listed structures only.** The structures of `TABLE_STRUCTURES` become `{"columns": [...], "rows": [[...], ...]}`; nested objects are flattened into dotted column names. The path inside the text item decides, not the tool, so the same structure in a program's action summary, a wait reply or a history entry is encoded the same way:

   | Structure (trailing path) | Form |
   |---|---|
   | `trajectory_feedback.measured.samples` | table |
   | `trajectory_feedback.planned.samples` | table of the command rows; the measured start state, which has another shape, is kept whole under `first` |
   | `compact_3d_trajectory.stages`, `process_trajectory.event_keyframes` | table |
   | `collision_model_witnesses`, `periodic_checks`, `trajectory_evidence.key_poses` | table |
   | `object_state_delta.moved` / `appeared` / `vacated` | table |
   | `effective_tolerances`, `stage_costs` when they map stage ids to objects | table with `keyed_by: "id"`; the `id` column holds the former keys |

   A listed structure is converted when it has at least two entries (after `first`, for planned samples), every entry is an object, no key contains a dot, no column holds a real `null` in one row while the field is absent in another, and the table prints shorter. `optional` lists column prefixes that some rows do not have (a `null` cell there means "no such field"); `common` holds string columns with the same value in every row that has them, and strings are lifted only inside a table. Everything else keeps its JSON form, in particular the task plan `ledger` / `attention`, `advisory_notes`, `policy_definitions`, `motion_safety_checks`, `bundle.cameras`, `source_views`, `stages` / `stage_results`, `failure_feedback.candidates`, `batch_results`, history `actions`, `episode_ledger` entries, `arms`, `overlays`, `static_refs` and `grasp_hints`.

The `robot_action` description states how to read tables and `unchanged_arms` and lists the derived fields (`COMPACT_ENCODING_CATALOGUE_NOTE`). The encoding is reversible except for rules 1 and 2: decoding the archived replies named above gave back every value exactly, apart from derived quantities (largest relative change 4.97e-4) and arm entries stated once (largest folded difference 6.0e-8, largest unprinted derived value 1.4e-6). A full read of stored evidence (`history_read_trace`, any `detail="full"`) is not encoded.

### Lossless compaction (`reply_same_as`, `reply_float32`)

- `reply_same_as`: inside one text item, an object, array (at least 80 characters) or string (at least 60 characters) that is byte-identical to a value printed earlier in the same item is printed once. The later copy becomes `{"same_as": "<JSON Pointer>"}` (objects and arrays) or the string `"same_as:<JSON Pointer>"`, where the RFC 6901 pointer is rooted at that text item and names the first copy; for example `"receipt": {"action_budget": {"same_as": "/action_budget"}}` states that the receipt budget equals the top-level `action_budget`. Pointers never cross text items, and a copy is kept inline when the pointer would save fewer than 40 characters. While `reply_same_as` is on, the `robot_action` description ends its reply notes with `SAME_AS_CATALOGUE_NOTE` (`core/presentation/reply_compaction.py`), which says how to read a `same_as` pointer; it is absent while the setting is off.
- `reply_float32`: a number that is exactly representable as a float32 (a float32 measurement widened to float64 before printing, such as `1.0667834281921387`) is printed with the shortest decimal that parses back to the same float32 (`1.0667834`). Numbers that are not float32-exact are printed unchanged, so nothing is rounded below its source precision.

`features.tool_catalog_profile="guide_only_v1"` (the default is `layered_v3`; `--tool-catalog-profile`; queue case option) applies the same "state it once" rule to the Codex dynamic-tool catalogue: an `inputSchema` property `description` that the tool description's generated "Input field guide" already contains verbatim is omitted from the schema; the tool description is unchanged and a description the guide lacks stays in the schema. It is consumed by the Codex session only.

### Reply shaping order and the digits each number ends with

`Harness.handle_tool` presents a result and shapes the reply in one order; every step is a pure function of the previous step's content items, and a failing step delivers the reply as the previous step left it:

0. Presentation (`ActorOutput.present`): metadata projection, then the payload profile. `lean_v2` (default) prints every float with six decimals and leaves out raw worker frame ids, empty residues and, for `program_run` / `program_status`, the two description sentences `result_semantics` and `timeout_semantics`. A full read of stored evidence (`history_read_trace`, any `detail="full"`) skips the profile's display step.
1. `reply_static_once`: `drop_static_once`, decided on the presented values.
2. `kind="action"`: `fit_reply` lays the reply out when `budgeted_action_reply` is on (blocks replaced only against a known host limit, until the text of step 3 fits) or `action_reply_profile="summary_first"` (fixed shape).
3. On the text items of the final layout, and of every layout step 2 weighs: `unify_detail_entries` (`reply_static_once`), then `reply_compact_encoding` (not on a full read), then `reply_same_as`, then `reply_float32`.

So the budget is spent on the shortened text, and `same_as` is computed inside the text item a value ends up in, on its final structure. Applying `same_as` to the presented reply first and laying it out afterwards would leave pointers into other text items and into replaced blocks (every one of the 610 archived execution replies named above); in this order a copy whose first occurrence was moved or replaced is printed in full instead, and `dangling_same_as(items) == []` for every delivered reply. Detail entries and the compact encoding are applied before `same_as` because they change the structure a pointer runs through; a pointer may name a value in a table's `common`.

| Number | `lean_v2` (default) | `lean_v1` / `full_v1` |
|---|---|---|
| Measurement (joint angles, poses, residuals, tolerances, apertures, measured `qdot_rad_s`, calibration) | six decimals (step 0); `reply_float32` then finds nothing shorter | source digits; a float32-exact value at its shortest equal decimal (`reply_float32`) |
| Derived quantity (`DERIVED_QUANTITIES` in `derived_motion`) | six decimals, then 4 significant digits (so a value below 1e-2 keeps the digits six decimals leave) | 4 significant digits |
| A full read of stored evidence | source digits; float32-exact values at their shortest equal decimal | the same |
| Typed `ToolResponse.data`, program replies, artifacts | source digits | source digits |

`lean_v2` and `reply_static_once` do not overlap: the profile removes the `program_run` description sentences from every program reply before step 1 sees them, and `reply_static_once` then leaves out only the remaining fixed note (`action_summary.semantics`) of a program that succeeded; a failed or waiting program (recognised by its `status`, also after the profile removed `result_semantics`) keeps it. The six-decimal display is the precision of `lean_v2` itself, chosen with that profile; choose `lean_v1` to keep source digits in the display.

### SO-101 and other robots

The settings are robot-independent, and what is robot-specific is read from the run: catalogue notes and omitted notes follow the run's catalogue and capabilities (above), stage entries keep every robot-specific report inline, and the listed table structures are converted only where they occur. On the fake SO-101 backend, preview (`grasp_3d_review`, `stage_plans`), execution, `mode="reach"` and `robot_observe` replies pass through the default settings without error, and compared with the `lean_v1` presentation of the same typed result every value arrives apart from the documented display rounding and blocks given by reference (`robot.so101.test_so101_grasp_and_touch`). This is a software check; behaviour on a physical arm has not been evaluated.

Tests: `core.presentation.test_actor_payload` (`ActionReplyBudgetTests`, `ReplyCompactionTests`, `ReplyShapingOrderTests`, `ReplyStaticOnceTests`, `SummaryFirstReplyTests`, `ReplyCompactEncodingTests`, `LeanV2WithReplySettingsTests`; the test-side decoders are in `tests/support/reply_defaults.py`), `runtime.harness.test_harness_fast` (all 128 setting combinations through the Harness), `runtime.actions.test_unified_actions` and `runtime.actions.test_rollout_4d_ledger` (the notes are the ones the real producers write; detail entries resolve through the real tools), `robot.embodiment.test_embodiment_contracts` (default and legacy start), `cortex.agents.test_agent_prompt_parity` (per backend with the pinned binaries: the notes in the published catalogue, a shaped reply in the model request, host limits), `cortex.external.test_external_workbench`, `cortex.codex.test_codex_config_and_provider`, `core.config.test_config` (defaults, CLI, queue, archives).

## Semantics That Must Never Be Compressed Away

- A program status of `succeeded` only means the Python process exited with code `0`; a template or action inside it may still have `stopped`. An action marked `failed` or `stopped` may have executed several steps before halting, and unknown executed step counts must never be backfilled as `0`.
- Native action steps, physics/control steps, and model request counts are distinct metrics and must never be conflated. On a shared dual-arm timeline, one arm may finish opening its gripper before the other arm fails; if no gripper execution report exists, its state must remain explicitly `unknown`.
- `arm_results` may reflect the last check from an earlier waypoint and must retain the corresponding phase, timestep, waypoint index, and `endpoint_result_semantics`. Command completion does not imply that the physical fingers reached the target opening or that an object is held.
- Manipulation templates report `released_unverified` only when every phase completes normally. Missing or corrupted output, failures, or unknown states are never upgraded to a successful release, `physical_claims_verified` remains `false`, and delivering a fresh image never automatically upgrades status to `retained`.
- `bundle.action_budget` is a historical snapshot taken at capture time, whereas `current_action_budget` reflects the live controller budget; `observation_is_current` and `current_observation_id` distinguish historical frames from the current state. Stored historical observations are immutable.
- Candidate trajectories are not future measurements, and executed trajectory feedback strictly separates planned commands from measured states. Robot-model sphere clearance is not proof of object collision avoidance. Camera outputs preserve live world poses, pixel transforms, and provenance while keeping static intrinsics resolvable via delivered version references, without inventing object, contact, or scoring ground truth.
- Every summary must provide a working handle to read the complete original payload. Errors, unknown fields, and unrecognized anomalies are retained conservatively; never strip data while leaving an unresolvable `available=true`.

Static payload-size replays only demonstrate changes in serialized byte volume; task performance, model request counts, token usage, and wall-clock savings must be validated separately through controlled experiments and trajectory/video review.
