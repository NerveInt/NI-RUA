# Embodiment Profile: one source of truth per robot

Everything the harness says or enforces about the robot of a run (arm ids, TCP name, frame, gripper numbers, speed
and tolerance ranges, clock, cameras, which `robot_action` fields exist) comes from one frozen
[`EmbodimentProfile`](../../ni_rua/robot/model/embodiments.py). The profile is built once per run. Shared code
(validators, the `robot_action` schema, tool text, actor instructions, perception hints, trajectory evidence, home
check, presentation) reads it; it does not define an arm set, a gripper number or a camera role itself. The audit that
motivated this and the work-package plan are in the design report 13 (private working notes, not in this repository);
this page describes what the merged code does, and the code is the source of truth.

The profile is audit and configuration data. It is never added to the capability document and never enters model
context as such; the model sees only text and schemas rendered from it (see [Model-visible text](#model-visible-text-is-rendered-from-the-profile)).

## What a profile is

`EmbodimentProfile` is a frozen dataclass with no methods that decide anything (`FrozenDict` for every mapping,
validated in `__post_init__`; an inconsistent profile raises `ValueError("invalid_embodiment_profile:<field>")`).
Units: lengths in m, angles in rad (`*_deg` names: degrees), times in s, rates in Hz, speed scales are dimensionless
multipliers of the backend's nominal speed.

| Group | Fields (unit) | Meaning |
| --- | --- | --- |
| identity | `robot` (`robot_model_id`, `frame_id`, `tcp_config_id`), `tcp_link`, `tcp_label`, `limits_profiles` | `frame_id` is the frame of every pose; `tcp_link` is the URDF link that `position_m` / `quaternion_xyzw` target; `tcp_label` is the TCP name used in field help (`link6/flange` on X5, `gripper_frame_link` on SO-101); `limits_profiles[0]` is the id reported as `capabilities.limits_profile`. |
| arms | `arms`, `max_simultaneous_arms`, `arm_joint_names`, `jaw_joint_names`, `stages_per_arm`, `max_stages` | Ordered arm ids (X5 `left`, `right`; SO-101 `right`); joint names in `robot_state` order; stage limits 16 per arm and 32 per batch. `dual_arm` is in `supports` exactly when `max_simultaneous_arms > 1`. |
| gripper | `gripper.empty_close_aperture_m` (m or `None`), `full_open_aperture_m` (m), `full_open_source`, `preset_margins_m` ({preset: (min m, max m)}), `aperture_reading`, `fingertip_beyond_reference_m` (m), `grasp_reference`, `grasp_reference_offset_m` (m), `approach_axis_local`, `jaw_opening_axis_local`, `mask_segment`, `contact_sections`, `jaw_grasp`, `hold_squeeze` | `hold_squeeze` (`HoldSqueeze(default_rad, max_rad)`, rad of the jaw joint, or `None`) is the bounded holding squeeze of closing jaw stages and the range of the jaw-stage field `grip_squeeze_rad`; `None` (ARX-X5, an SO-101 deployment without `gripper.hold_squeeze_rad`) offers no such field. `empty_close_aperture_m` is the measured aperture after an empty full close and the zero point of the grip-guard presets (preset bound = this + margin; `None` = unmeasured, presets cannot resolve). `full_open_aperture_m` is the one full-opening value of the robot (explicit `min_aperture_m` / `max_aperture_m` bound, grip-guard maximum, perception `opening_check`, depth `max_opening_m` default). `grasp_reference` is `tcp_offset` (TCP plus offset along the approach axis, X5 0.1501 m) or `jaw_fingertip_midpoint` (CAD fingertip midpoint at the measured jaw angle, SO-101). `fingertip_beyond_reference_m` is measured from a different point on each robot: X5 from the grasp reference (the pad centre, 0.0075 m), SO-101 from the TCP `gripper_frame_link` (0.0081 m, the moving jaw's peak CAD reach beyond the TCP over the jaw range; the fixed jaw reaches 0.0063 m), because the SO-101 grasp reference moves with the jaw. The SO-101 TCP lies on the fixed jaw's inner fingertip face, not between the fingers ([SO-101 integration §2.1](so101-integration.md#21-gripper-geometry-for-grasping)). `contact_sections` (X5 only) names the contact regions. `jaw_grasp` (`JawGraspGeometry`, SO-101 only; set exactly when `pose_inputs` has `jaw_grasp_centre`) carries the CAD bounds of the grasp-centre input in the TCP frame, m: `fixed_face_x_m` (0.0), `min_grasp_depth_m` / `max_grasp_depth_m` (-0.0053 / 0.0611: fixed fingertip end and pocket floor), `tcp_plane_gap_m` (0.0618: jaw gap above which the moving fingertip is behind the TCP plane) and `max_gap_m` (0.1101 at the default `jaw_open_rad`: widest gap between the fingertips along the jaw line). |
| poses and features | `pose_inputs`, `supports`, `stall_guard_semantics`, `vertical_window_semantics` | `pose_inputs` is a subset of `tcp_quaternion`, `grasp_angles` (the X5 top-down `angles_deg` convention) and `jaw_grasp_centre` (the SO-101 grasp centre between a fixed and a moving jaw). `supports` is a subset of the tags below. The two texts are the field help of the stall guard and of `max_upward_dz_m` on this robot. |
| speed | `speed_scale`, `gripper_speed_scale` (`min`, `max`, `default`), `speed_presets`, `gripper_speed_presets` | Accepted multiplier ranges (X5 arm 0.25..2, jaw 0.25..1; SO-101 0.25..1 for both) and only the named presets inside the range. |
| tolerances | `tolerance_presets` ({name: (m, rad)}), `default_tolerance_preset`, `default_tolerance`, `max_position_tolerance_m`, `max_orientation_tolerance_rad`, `stall_guard_presets_m`, `tracking_divergence_range_m`, `arrival_correction_limits` | `arrival_correction_limits` (`ArrivalCorrectionLimits`: `max_rounds` count in [1, 10], `max_offset_rad` rad in (0, 0.5]; default `None`) is set exactly when the deployment's sag correction is on (SO-101 `motion.arrival_correction_rounds` / `arrival_correction_max_offset_rad`, both above 0) and then comes with the `arrival_correction` tag: it is the default and the schema maximum of the two fields. `profile_record` writes it only when set (the X5 record is unchanged); an older SO-101 archive restores with the tag and no limits, and its schema then shows the shared maxima without calling them the default. `default_tolerance` is applied when a move stage sets neither a preset nor an explicit tolerance (X5: the `normal` preset; SO-101: the deployment's `motion.default_*_tolerance_*`, so `tolerance_preset` may be absent from a receipt). `stall_guard_presets_m` maps a preset name to its `max_tracking_divergence_m` in m (`None` = off). |
| time | `clock`, `control_hz`, `step_domain`, `step_label`, `per_request_step_limit`, `default_action_timeout_s`, `default_wait_timeout_s`, `max_timeout_s` | `simulated_steps` (the world advances only during execution) or `real_time`. X5: 25 Hz, 250 native steps per request, 120 s / 30 s / 300 s. SO-101: the deployment's command rate, no step limit, deployment timeouts (example file 60 s / 30 s / 300 s). |
| cameras | `cameras`, `overview_camera`, `wrist_cameras`, `camera_examples`, `support_surface_z_m` (m) | X5 `cam_head`, `cam_left_wrist`, `cam_right_wrist`; SO-101 `fixed`, `wrist`. `support_surface_z_m` is the support height perception scales VGGT depth with: X5 table top 0.765 m, SO-101 `None` (the observation's calibrated `table_z_m`). |
| context switches | `handoff`, `home_check`, `policy_step` | Cross-arm handoff policy, the identity of the completion-guidance home check (version, frame, TCP, arms), and whether the matched learned policy can drive the robot. X5 true / policy / true; SO-101 false / policy / false. |

### `supports` tags

Each tag gates a group of `robot_action` fields. A field whose tag is missing is absent from the schema, the tool
description and the field guide, and one generated sentence names the missing features in plain words (decision Q3).
The backend contract still refuses such a field by name (`unsupported_capability: <field>`), never accepts and ignores it.

| Tag | Gates | X5 | SO-101 |
| --- | --- | --- | --- |
| `gripper_motion` | `gripper_motion` start/end fractions, `during_motion` jaw events | yes (not in feedback mode) | yes |
| `joint_travel` | `joint_travel_preset`, `max_joint_travel_rad`, `max_backtracking_rad` | yes | no |
| `max_upward_dz` | `max_upward_dz_m` (help text from `vertical_window_semantics`) | yes | yes (acceptance rule only) |
| `stall_guard` | `stall_guard_preset`, `max_tracking_divergence_m` (help from `stall_guard_semantics`) | yes | yes |
| `clearance` | `clearance_preset`, `clearance_height_m` | yes | no |
| `review_proxies` | `obstacles`, `held_object(s)`, `contact_entity_id`, `initial_contact`, `placement_contact`, obstacle registry | yes | no |
| `obstacle_object_ids` | `obstacle_object_ids` resolved from the object table | yes | only with the deployment's `ik_service` (boxes of the collision world) |
| `settle` | the bounded `settle` check | yes | yes |
| `execute_if_clean` | mode `execute_if_clean`, `auto_execute_if_clean` | yes | yes |
| `dual_arm` | two arms in one request, `dual_arm_progression`, `arm_options` | yes (not in feedback mode) | no |
| `held_perception` | FK robot pixel mask from the arm meshes and the held-object row ([held_perception](../held_perception.md#embodiments-arx-x5-and-so-101); one `*Body` per robot) | yes | yes |
| `reach` | mode `reach` (per-stage IK queries, no plan, no motion; [unified actions](unified-actions.md#modereach-and-split-orientation-tolerances-so-101-profile-tags)) | no | yes (always) |
| `approach_roll` | `approach_tolerance_rad`, `roll_tolerance_rad` (split orientation bounds) | no | only with the deployment's `ik_service` (cuRobo; `deployment.embodiment_profile`) |
| `cross_view` | `geometry_query` kind `object_table` with `target_camera_ids` also returns `value.cross_view` ([cross-view check](../object_table.md#cross-view-check)) | no (outputs kept as benchmarked) | with both the `fixed` and the `wrist` camera |
| `arrival_correction` | `arrival_correction_rounds`, `arrival_correction_max_offset_rad` (bounds of the goal shift after a move settles; schema maxima = `arrival_correction_limits`) | no | when the deployment corrects (`motion.arrival_correction_rounds` and `arrival_correction_max_offset_rad` above 0) |
| `path_check` | the preview's `path_check` review (sampled collision check of the joint path between targets), named among the `execute_if_clean` reviews | no | only with the deployment's `ik_service` |
| `collision_world` | `obstacles`, `held_object` as boxes of the planner's collision world (no proxy review, no registry); with `obstacle_object_ids` also that field | no | only with the deployment's `ik_service` |

Pose inputs: `tcp_quaternion` is `position_m` + `quaternion_xyzw` of `tcp_link`; `grasp_angles` adds
`grasp_position_m` + `angles_deg` (and, with `gripper.contact_sections`, `contact_region` / `contact_depth_m`,
`grasp_3d_review`, `pose_plan`, the `across_yaw_deg` hints). Profiles without `grasp_angles` get convention-free
perception facts instead (`tcp_z_m`, `jaw_line_yaw_deg` in the base frame).

Two catalogue sentences also follow the profile rather than a tag (`robot_action_schema._build`): with `clock="real_time"` the tool description says that `execute` runs cached IK joint targets re-interpolated from the joints measured at execution (not "exact cached rows"), and the `normal` preset carries no vertical contact/load compliance note (that is the X5 batch executor's implicit vertical window); without any preview review (`motion_safety`, `grasp_3d_review`, `geometry_review`, `path_check`) `execute_if_clean` is described by what gates it, planning status and `downgrade_reasons` (`closed_jaw_before_grasp:<arm>` only with a measured `empty_close_aperture_m`). The reviews a profile runs, their clean conditions, scopes on a real-time robot and `downgrade_reasons` codes come from one list (`embodiments.preview_reviews`, `PREVIEW_REVIEW_CLEAN`, `PREVIEW_REVIEW_DOWNGRADES`), which the robot_action text and the SO-101 capability document (`unified_action.execute_if_clean`) both render. The X5 catalogue is unchanged.

`jaw_grasp_centre` (SO-101; requires `gripper.jaw_grasp` and `grasp_reference="jaw_fingertip_midpoint"`) adds the
move fields `embodiments.JAW_GRASP_CENTRE_FIELDS`: `grasp_centre_m` ([x, y, z] m in `frame_id`), `object_width_m` (m,
required), `fixed_jaw_clearance_m` (m, default 0), `grasp_depth_m` (m, default 0) and `fixed_jaw_yaw_deg` (deg,
straight-down shorthand for `quaternion_xyzw`). The backend contract converts them to the `tcp_link` target
(SO-101: `so101.contract.grasp_centre_pose`; [SO-101 integration §4.9](so101-integration.md#49-grasp-centre-input-grasp_centre_m)).
With the tag the schema publishes those fields, `mode=reach` accepts them, `robot_state` carries the measured
`jaw_grasp` block, the object table emits `grasp_centre_z_m`, `jaw_lines` and an `opening_check` against the jaw-gap
range instead of `tcp_z_m` / `jaw_line_yaw_deg`, and preview runs the fixed-jaw `grasp_3d_review`. A profile without
the tag (ARX-X5) renders none of it. `profile_record` writes `gripper.jaw_grasp` only when it is set, so the X5
record and every older archive are unchanged and still restore.

## How a run gets its profile

1. The backend builds it, purely and parent-side (no worker needed). The hook is `embodiment_profile()` on the
   backend class: [`robodojo.RobotBackend`](../../ni_rua/robot/backends/robodojo/robodojo.py) calls
   `arx_x5_profile(binding=…, thresholds=…, motion_execution_mode=…)` with the backend's RoboDojo binding and the run's
   `motion_thresholds`; [`so101.RobotBackend`](../../ni_rua/robot/backends/so101/backend.py) calls
   [`deployment.embodiment_profile(deployment)`](../../ni_rua/robot/backends/so101/deployment.py) on the loaded
   deployment file. The SO-101 worker builds the same profile from the same file, so parent and worker cannot disagree.
2. `Harness.__init__` calls `resolve_embodiment_profile(robot, config)` before the tool registry is created and stores
   the result as `harness.embodiment`. A backend without the hook (test doubles, `trusted_legacy` integrations) gets
   `arx_x5_profile(thresholds=config.features)`, which equals `ARX_X5_PROFILE` at default thresholds. That fallback is
   the one remaining definition of the "default robot". A hook that returns anything but an `EmbodimentProfile`
   fails with `invalid_embodiment_profile`.
3. `Harness.start` archives it once as a `configuration` record `embodiment_profile-<digest>` (`profile_record`, schema
   `ni_rua.embodiment_profile.v1`) and exposes the id as `harness.embodiment_record_id`; the Workbench writes it into
   `manifest.json` as `embodiment_record_id`. The record is observer data and never enters model context (a scrub test
   covers every actor-scope artifact).
4. Offline readers use the archived profile, never the one the current code would build.
   `archived_embodiment_record(store, record_id)` finds the record (the manifest's id, else the store's last such
   record) and `profile_from_record(record)` is the exact inverse of `profile_record` (an unknown schema or a changed
   field set raises `unsupported_embodiment_profile_record`). Resume and replay name the archived record
   (`Workbench.embodiment_record()`), and `apps.evaluation.checkpoint.recompute_tool_output` restores it. An archive
   written before the profile existed has no record: nothing requires one, a RoboDojo-origin run is read with the X5
   defaults (as `capability_arms` always did), and perception keeps the pre-profile default-camera rule.

Capabilities remain what the backend reports. The worker fills the overlapping keys (`arms`, `frame_id`,
`action_speed`, `grip_guard`, ...) from the profile and a contract test asserts profile and capabilities agree key by key
([`tests/robot/embodiment/test_embodiment_profile.py`](../../tests/robot/embodiment/test_embodiment_profile.py)).

## Model-visible text is rendered from the profile

X5 sees X5 numbers and SO-101 sees SO-101 numbers because the same builders render both from the run's profile:

| Output | Builder | Profile input |
| --- | --- | --- |
| `robot_action` schema, stage schemas, field help, tool description | [`robot_action_schema.build(profile)`](../../ni_rua/runtime/actions/robot_action_schema.py) (cached per profile; `ROBOT_ACTION_TOOL` and `_STAGE_SCHEMAS` remain as the default-robot rendering) | arms, TCP label, frame, `gripper` zero point / full opening / preset margins, speed and tolerance ranges and presets, stall presets, timeouts, stage limits, clock, tags |
| Other tool definitions (camera examples, arm ids, frame, clock wording, `robot_wait` rate and timeouts, contact overlay, `preview_ee`) | [`public_definitions(profile=…)`](../../ni_rua/runtime/tools/definitions.py); `create_registry(runtime=harness)` passes `harness.embodiment` | cameras, arms, `frame_id`, `tcp_label`, `clock`, `control_hz` |
| Request validation | `unified_actions` validates against the schema of the run's registry, not the module constant | same schema as the catalogue |
| Actor instructions | [`actor_instructions.instructions(…, profile=)`](../../ni_rua/cortex/context/actor_instructions.py) assembles named sections; the backend can own its motion section (`actor_motion_instructions()`, SO-101: [`contract.actor_motion_instructions`](../../ni_rua/robot/backends/so101/contract.py)); the handoff section exists only when `profile.handoff` | clock, arms, `pose_inputs`, TCP |
| Perception text and hints | [`obs3d.developer_instructions(features, profile)`](../../ni_rua/perception/external/obs3d.py), object table, `scene3d`, pose skill | cameras, `frame_id`, `gripper.*`, `pose_inputs`, `supports` |
| Default first prompt, manifest labels | [`workbench.default_prompt(profile)`](../../ni_rua/apps/gateway/workbench.py), `data_origin` from capabilities, `clock_policy` from `clock`, `runtime_protocol` from `limits_profiles[0]` | arms, clock |
| Lean projection, home check, presentation | `core/presentation/lean_projection.py` (compact TCP key `<tcp_link>_xyz`), [`home_check_policy(profile)`](../../ni_rua/runtime/actions/home_check.py), `measured_delta` and `MetadataDelivery` loops | `tcp_link`, `home_check`, `arms` |

Rules the builders follow:

- Every number in a help text enters through the profile. Metres are spelled with trailing zeros stripped
  (`0.0015m`, `0.088m`), millimetres with `%g` (`1.5mm`, `88mm`).
- A field whose tag the profile lacks is absent, not documented and rejected. One sentence lists the absent features.
- The X5 rendering of every builder is byte-identical to the text before the refactor except the approved changes
  below. The golden fixtures enforce that ([What the goldens protect](#what-the-goldens-protect)).
- Numbers named in the catalogue must equal what the validator enforces:
  [`tests/robot/embodiment/test_embodiment_schema.py`](../../tests/robot/embodiment/test_embodiment_schema.py) probes every published bound (value at the
  limit accepted, just beyond rejected) for both robots, and checks the payload delivered after native tool conversion.
- The SO-101 catalogue contains none of `left`, `env_origin`, `link6`, `cam_head`, `88`, `25 Hz`, `simulat`,
  `native budget` or the field names it does not honour.

## How thresholds are configured

| Robot | Where the numbers live | Rule |
| --- | --- | --- |
| ARX X5 (RoboDojo) | `features.*` threshold fields of the run configuration ([Configuration](configuration.md#motion-and-gripper-thresholds-features)) resolved by `core.config.motion_thresholds` | Preset tables are functions of the resolved thresholds (`tolerance_presets`, `grip_guard_preset_margins`, `stall_guard_presets`), computed per run. A non-default `features.grip_rigid_min_aperture_m` or `features.tight_position_tolerance_m` changes the preset, the validator and the schema text together (decision Q2). With the shipped configs all values are the defaults and the output is unchanged. `features.max_gripper_aperture_m` (0.088 m) is the X5 full opening. |
| SO-101 | the deployment file `configs/so101.local.json` (template [`configs/so101.example.json`](../../configs/so101.example.json)); every field is declared in `deployment.FIELDS` with unit, range, default and doc | `features.grip_*`, `features.*_tolerance_*` and `features.max_gripper_aperture_m` are not read for SO-101 (decision Q4). Fields: `gripper.empty_close_aperture_m`, `gripper.full_open_aperture_m`, `gripper.preset_margins_m`, `motion.tolerance_presets`, `motion.default_position_tolerance_m` / `default_orientation_tolerance_rad`, `motion.default_timeout_s`, `max_timeout_s`, `default_wait_timeout_s`. Details in [SO-101 integration](so101-integration.md#41-grip-guard-and-checkpoint_after_gripper). |
| Support height | `features.vggt_support_z_m` | Default `"profile"` takes `profile.support_surface_z_m` (X5 0.765 m; SO-101 none, so the calibrated `table_z_m` of the observation). A number in [-3, 3] m is authoritative for every robot; `null` always uses the observation's calibration. Archives and configs that carry `0.765` keep it. |

The stall-guard presets (`off`, `contact_stop` 0.02 m, `strict` 0.01 m; accepted explicit `max_tracking_divergence_m`
0.005..0.15 m) are shared run thresholds on both robots (`features.stall_guard_*` on X5, code defaults on SO-101). The
schema publishes only `off`, `contact_stop` and `strict` (`PUBLISHED_STALL_GUARD_PRESETS`); the X5 aliases `firm`,
`normal`, `sensitive`, `compliant` remain accepted by the X5 validator but are unpublished and refused by SO-101 at
stage and batch level.

Retired fields (decision V-E): `soft_position_tolerance_m`, `grip_thin_max_aperture_m`, `grip_rigid_max_aperture_m`,
`grip_wide_max_aperture_m`, `max_min_grip_m`, `closed_jaw_goal_max`, `checkpoint_empty_jaw_measured_max` were read by
no code. They are removed from `FeaturesConfig`; documents that still carry them load and the keys are dropped
(`core.config.RETIRED_FEATURE_FIELDS`).

## How to add a robot

A new robot needs one builder, one backend hook and honest `supports` tags. No registry of profiles, no inheritance,
no feature flags.

1. **Identity and facts.** Add a `RobotDescription` (stable `robot_model_id`, `frame_id`, `tcp_config_id`) next to
   `ARX_X5` / `SO101` in `robot/model/embodiments.py`, and the packaged static facts under
   `robot/model/robot_information_data/<robot>/physical.json` (arm and gripper joint order, TCP link, public arm slots;
   see [Robot and Environment Integration](robot-environment-integration.md#3-static-information-files-required-for-a-new-robot-arm)).
2. **Write the builder.** A pure function that returns an `EmbodimentProfile` from the packaged facts plus the
   deployment or binding of that robot: `arx_x5_profile(...)` and `so101.deployment.embodiment_profile(deployment)` are
   the two examples. Rules: no worker, no network, no import of the backend contract (the contract reads the profile;
   the builder calling it would recurse), numpy only lazily, every threshold a documented deployment or config field
   with unit, range and default (not a constant in shared code). Unmeasured values stay `None`
   (`gripper.empty_close_aperture_m`); do not invent a zero point. Construction validates the profile.
3. **Choose `supports` and `pose_inputs` honestly.** List only what the backend executes and the worker verifies. If
   you set `max_upward_dz`, supply `vertical_window_semantics`; if you set `stall_guard`, supply
   `stall_guard_semantics`. A robot without a fixed grasp-angle convention keeps `pose_inputs={"tcp_quaternion"}`
   (`grasp_angles` also requires `grasp_reference="tcp_offset"`). A gripper with one fixed and one moving jaw can
   offer `jaw_grasp_centre` once its builder supplies `gripper.jaw_grasp` from CAD and its contract converts the
   request.
4. **Add the hook.** Give the backend class `embodiment_profile()` returning the builder's result (see
   `robodojo.RobotBackend.embodiment_profile` and `so101.RobotBackend.embodiment_profile`) and register the backend in
   `robot/backends/factory.py`. Without the hook the Harness falls back to the X5 default profile, which is only
   acceptable for test doubles.
5. **Report capabilities from the same profile.** The worker's capability document (`arms`, `frame_id`,
   `tcp_config_id`, `limits_profile`, `action_speed`, `grip_guard`, `bounded_wait`, `motion_tolerance_defaults`,
   `default_timeout_s`, ...) must be filled from the profile; add the robot to the key-by-key agreement test.
6. **Refuse by name what you do not honour.** `compile_action` / `validate_action` of the backend raise
   `unsupported_capability: <field>` for every field whose tag you did not set, and for every path field the worker
   would ignore. The generated schema already omits those fields.
7. **Implement the optional features individually** ([5.3](robot-environment-integration.md#53-optional-features-must-be-adapted-individually)):
   post-action geometry needs `tcp_link`, arm joint names and a worker-captured execution-start observation
   (`before_observation`); overlays are keyed by `profile.arms`; the contact overlay needs `contact_sections` or draws the
   CAD fingertips (as on SO-101); the home check needs `home_check`.
8. **Prove it.** Extend [`tests/robot/embodiment/test_embodiment_profile.py`](../../tests/robot/embodiment/test_embodiment_profile.py) (facts, thresholds reach the
   profile, capability agreement, record never reaches model context), [`tests/robot/embodiment/test_embodiment_schema.py`](../../tests/robot/embodiment/test_embodiment_schema.py)
   (catalogue words, every bound enforced at the limit), [`tests/robot/embodiment/test_embodiment_prompt_cleanup.py`](../../tests/robot/embodiment/test_embodiment_prompt_cleanup.py),
   and [`tests/robot/embodiment/test_embodiment_perception_evidence.py`](../../tests/robot/embodiment/test_embodiment_perception_evidence.py), and run the X5 goldens: a new robot must
   not change a single X5 byte.
9. **Document** the supported and refused fields with units in a robot guide like [SO-101 integration](so101-integration.md).

## What the goldens protect

Golden fixtures freeze what the X5 actor sees and what the runtime decides at the revision before the refactor
(`BASELINE_REVISION = 5fd93c6`, in [`tests/support/embodiment_golden.py`](../../tests/support/embodiment_golden.py)).
Six families are captured through the runtime's own builders and stored under `tests/fixtures/embodiment/<embodiment>/<family>.json`:

| Family | Content |
| --- | --- |
| `catalogue` | `public_definitions(**flags)` for the perception flag combinations `create_registry` passes, plus the actor and program catalogues of a real Workbench start for every shipped X5 configuration (`robot_action` schema included) |
| `instructions` | `actor_instructions.instructions` for calibrated / legacy / depth contracts, sandbox, operation memory and handoff variants, plus the `obs3d`, pose, held, verification and depth sentences |
| `startup` | complete startup instructions, first actor turn, tool names, capabilities and task-prompt record of `Workbench.start` with `FakeAgentSession` |
| `compile` | schema admission, `compile_request` output and backend validation for a stored 362-request corpus (every stage field at its bounds and just outside) |
| `guard` | the seeded 140,483-case differential of grip-guard decisions plus exhaustive validate / bounds / evaluate tables |
| `config` | `Config().to_dict()`, `motion_thresholds()`, the shipped configs as loaded and each startup's `effective_config.json` |

[`tests/robot/embodiment/test_embodiment_x5_golden.py`](../../tests/robot/embodiment/test_embodiment_x5_golden.py) recaptures each X5 family from the current
tree and requires byte identity (order-preserving compact JSON, run-specific values redacted). Related freezes:
[`tests/robot/embodiment/test_embodiment_perception_evidence.py`](../../tests/robot/embodiment/test_embodiment_perception_evidence.py) pins 61 digests of feedback, overlays,
contact overlays, home-check sequences and metadata delivery, and compares every X5 hint, audit and obstacle
output with the `5fd93c6` code loaded from git and proves only V-C and V-D differ. The SO-101 fixtures record the
SO-101 behaviour of that revision, are labelled "expected to change, never asserted", and are only checked for provenance.
Native X5 behaviour (cuRobo worker, `batch_motion`) cannot run on CPU; the goldens prove the compiled requests and text,
not native motion.

### Printing a difference and regenerating

```bash
# Differential report of the current tree against the fixtures (non-zero exit when an item changed):
PYTHONPATH=tests mamba run -n ni-rua python -m support.embodiment_golden --diff [--embodiment x5|so101] [--family F]
# Rewrite fixtures (refuses without --force, and refuses a dirty ni_rua/ or configs/ tree without --allow-dirty):
PYTHONPATH=tests mamba run -n ni-rua python -m support.embodiment_golden --write --force --embodiment E --family F
```

The X5 fixtures are the baseline: do not regenerate them to make a failing test pass. Each fixture header records
`captured_from.revision` and `production_tree_clean`, and `FixtureProvenanceTests` requires every fixture, X5 and SO-101,
to come from a clean `5fd93c6` tree, so a `--write` at a later commit makes that test fail unless `BASELINE_REVISION` is
changed too. Moving the baseline is a deliberate decision (it discards the proof "X5 unchanged since `5fd93c6`" and all
allowances below); it needs the same explicit approval as an X5-visible change. A difference you did not intend is a bug
in the change, not in the fixture.

### Adding an approved allowance

An X5-visible change is allowed only after the user approved it, and it is encoded as an exact allowance rather than a
rewritten fixture. The mechanism is `ALLOWANCES` / `apply_allowances` in `tests/support/embodiment_golden.py`:

1. Append one record to `ALLOWANCES` inside your own block delimited by `# --- WPn allowances ---` and
   `# --- end WPn ---`. Fields: `id`, `package`, `embodiment` (`x5`), `family`, `occurrences` (total edit count) and either
   `edits` (`{item: [(JSON pointer inside the item, old value, new value)]}`, `ABSENT` for a key that is removed or
   added) or `replacements` (`{item: [(JSON pointer of a text, "" = the item itself, old substring, new substring,
   occurrences)]}`), or both. Name every affected item and every site; do not use patterns.
2. `check_allowance` verifies that `occurrences` equals the sum of the edits and replacement counts.
3. `compare()` takes the recorded item, applies exactly its allowances (every recorded old value must be exactly the
   declared one, every old substring must occur exactly the declared number of times) and calls the item equal only if
   the current tree reproduces that expectation byte for byte. Any other byte, or a missing allowed change, is
   "changed". The ids used are listed in the report.
4. The section 6 correction below keeps its own mechanism (`APPROVED_CORRECTION`, applied to every catalogue that carries
   `robot_action` at once; `test_correction_allows_only_its_own_bytes` compares with `allow_allowances=False`). A change
   that no golden item shows is pinned by a differential test instead (V-C: `WP4_DIFFERENTIAL` and
   `tests/test_embodiment_perception.py`).
5. Record the change and its reason in the next section.

## X5-visible changes approved by the user (2026-11-13)

Everything else about X5 is byte-identical to `5fd93c6` in the goldens. Each change below was approved explicitly and is
encoded as an exact allowance (or pinned by a differential test).

| Change | What X5 sees | Reason | Where encoded |
| --- | --- | --- | --- |
| Schema-number correction (section 6) | Five source strings, 21 occurrences, +7 bytes per catalogue that carries `robot_action`: preset minimums `non_empty` 2 mm to 1.5 mm, `rigid_object` 4 mm to 2 mm, `wide_object` 15 mm to 3 mm (`[min 0.002m]` to `[min 0.0015m]` etc., and `[>=2mm]` to `[>=1.5mm]` etc. in the tool description). `thin_object` 0.5 mm, "full open 88mm, closed 0m" and `[0, 0.088]` were already right | The published text disagreed with the runtime presets (`non_empty` 0.0015 m, `thin_object` 0.0005 m, `rigid_object` 0.002 m, `wide_object` 0.003 m) | `APPROVED_CORRECTION` |
| V-C: one X5 pad offset | `scene_obstacles` target-exemption centre moves 3.1 mm along `link6` +x (0.147 m to 0.1501 m); no text changes | Three values existed for one fact (`roboprobe_pose.GRASP_POINT_OFFSET_M` is 0.1501 m) | `WP4_DIFFERENTIAL`, `tests/test_embodiment_perception.py` |
| V-D: one X5 full opening, 0.088 m | "88.8 mm" becomes "88 mm" in the `pose_plan` description and the obs3d developer instruction (14 golden occurrences); `opening_check` / `tight` / `too_wide` boundaries shift by 0.8 mm; `depth_query` `max_opening_m` default and maximum 0.088807166 m become 0.088 m (17 catalogue items, 102 occurrences) | Three values (0.088 guard, 0.0888 hints, 0.088807166 depth) for one fact; 0.088 m is the joint7 + joint8 travel, the others were the 0.8 mm wider pad gap | WP4 and WP12 allowances `design-13-V-D-*` |
| V-E: seven unread threshold fields removed | `effective_config.json` and `Config.to_dict()` lose seven keys (147 removals in 21 golden items); old configs and archives still load; nothing model-visible | Dead configuration: no code read them | WP5 allowance `design-13-V-E-unread-threshold-fields` |
| V-F: `max_upward_dz_m` help matches the code | One sentence at three sites per catalogue (69 occurrences): the defaults include the `max(pos_tol, ...)` terms and the 0.009 m condition of the batch executor | The old text omitted terms the code applies | WP2 allowance `design-13-V-F-max-upward-dz-help` |
| F7: `features.vggt_support_z_m` defaults to `"profile"` | `Config()` and the shipped `configs/ni-rua.json` / `ni-rua.example.json` carry `"profile"` instead of `0.765` (14 config golden items, archive only, not model-visible); X5 resolves to the same 0.765 m | The X5 table height was a global feature default that SO-101 had to override by hand | WP12 allowance `design-13-F7-support-height-from-profile` |
| Q1: one-time catalogue change on resume | The catalogue digest of an X5 run changes once (the correction above, V-D and V-F change published strings). A Codex thread resumed from an archive whose recorded `tools_digest` predates the change is refused with `resume_tool_registry_conflict` (`cortex/session/codex.py`); the code does not pin the old text, so such a run is continued from a fresh run directory. The provider prompt cache misses once | Q1 was decided as "accept the one-time text change" | none (behaviour of the existing digest check) |

Also decided with these (not X5 byte changes): Q2, `features.*` threshold overrides take effect on X5 (the values
change presets, validators and text together; shipped configs carry the defaults); Q3, fields a robot cannot honour are
absent from its schema; Q4, SO-101 thresholds live in the deployment file; Q5, SO-101 grasp hints are convention-free
facts (`tcp_z_m`, `jaw_line_yaw_deg`); Q6, supervised real-arm validation of the SO-101 motion features (done).

Q5 leaves a gap that the 2026-11-15 audit of two real-arm runs found: the hints give no x/y for the TCP, and
`jaw_line_yaw_deg` is an undirected line while the SO-101 jaw is one-sided (TCP on the fixed jaw's inner face, the
gap on TCP -X), so yaw and yaw + 180 deg are different grasps and part of the straight-down yaws is excluded by the
wrist_roll limits. The actor text now states these facts ([SO-101 integration §2.1](so101-integration.md#21-gripper-geometry-for-grasping));
the grasp-centre input that would close the gap is separate work (see the grasp centre section there).
