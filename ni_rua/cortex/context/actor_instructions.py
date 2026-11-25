"""One startup contract for planning and interaction, separate from tool schemas.

``instructions`` assembles named sections in a fixed order, each rendered from the run's ``EmbodimentProfile``
(``harness.embodiment``, design 13 section 5.2; ``profile=None`` is ``embodiments.ARX_X5_PROFILE``). The profile
facts each section reads:

* ``## Task and evidence`` and ``## Task plan and current concerns``: ``clock``. ``simulated_steps``: the world waits
  for the model and advances only during physical actions; ``real_time``: it keeps moving while the model reasons.
* ``## Tools and execution``: ``arms``. The shared-timeline sentence (one batch moves several arms together) is
  present only for an embodiment with more than one arm.
* ``## Motion planning`` + ``## Coordinates and timing``: the backend's own section (``embodiment_motion``, e.g.
  SO-101 ``contract.actor_motion_instructions``), else the ARX-X5 sections. Those describe ``angles_deg`` grasp poses,
  contact sections and two arms, so a profile without the ``grasp_angles`` pose input must supply its own.
* ``## Feedback and next decision``: ``pose_inputs`` (how a ``gripper_relative`` result becomes a move target:
  ``grasp_position_m`` for ``grasp_angles`` profiles, the TCP ``position_m`` otherwise) and ``clock`` (a native-step
  budget exists only on a simulated step clock).
* ``## Programs and reviews`` / ``## Reusable experience``: embodiment independent.
* ``## Observation contract``: the caller's text; ``legacy_observation_instructions(profile)`` names the TCP link.
* ``## Run handoff policy``: only for an embodiment with a cross-arm handoff (``profile.handoff``).

The ARX-X5 rendering of every section is byte-identical to the text before the profile existed (WP0 goldens):
``TASK_INSTRUCTIONS``, ``INSTRUCTIONS`` (tools .. reusable experience with the sandbox and operation memory) and the
observation-contract constants are that rendering.
"""
from .task_planning import PLANNING_INSTRUCTIONS

#: The world-clock sentence of the task section, per ``EmbodimentProfile.clock``.
WORLD_CLOCK_SENTENCES = {
    "simulated_steps": ("The world advances during physical actions, including robot_wait; observations and "
                        "wall-clock sleep do not advance it."),
    "real_time": ("The world runs in real time: objects and people can move while you reason, observe or sleep. "
                  "The robot is commanded only by physical actions, including robot_wait, which holds the last "
                  "commanded joint goals for a measured wall-clock duration."),
}


def _task_section(clock):
    return ("## Task and evidence\n"
            "Use the supplied task, images, measured robot state and available tools to complete the task.\n"
            "Run autonomously: decide the plan, tool arguments and program action reviews yourself.\n"
            "Resolve problems and continue without asking for human input or confirmation.\n"
            + WORLD_CLOCK_SENTENCES[clock] + "\n"
            "Runtime capabilities are supplied below. Use their coordinate conventions and limits.\n"
            "Do not read hidden object state, evaluator internals or private task assets.\n")


TASK_INSTRUCTIONS = _task_section("simulated_steps")

# task_planning.PLANNING_INSTRUCTIONS is written for the simulated step clock. Its two clock sentences, and their
# real-time rendering; every other sentence is embodiment independent.
PLANNING_REAL_TIME_SENTENCES = (
    ("experience or preview motions; these do not advance the world.",
     "experience or preview motions; these do not move the robot."),
    ("Waiting advances simulation and needs this plan too.",
     "robot_wait is a physical action and needs this plan too."),
)


def planning_instructions(clock):
    """``task_planning.PLANNING_INSTRUCTIONS`` rendered for ``clock`` (``EmbodimentProfile.clock``).

    ``simulated_steps``: verbatim. ``real_time``: the two clock sentences of ``PLANNING_REAL_TIME_SENTENCES`` are
    replaced. Raises ``ValueError("planning_clock_sentence_changed: ...")`` when the planning text no longer contains
    one of them exactly once, so an edit there cannot silently keep the simulated wording on a real-time robot."""
    text = PLANNING_INSTRUCTIONS
    if clock == "real_time":
        for simulated, real_time in PLANNING_REAL_TIME_SENTENCES:
            if text.count(simulated) != 1:
                raise ValueError("planning_clock_sentence_changed: " + simulated)
            text = text.replace(simulated, real_time)
    elif clock != "simulated_steps":
        raise ValueError(f"invalid_clock: {clock!r}")
    return text


OPERATION_MEMORY_INSTRUCTIONS = """
## Reusable experience
operation_memory separates current-task records from reusable cross-task experience. Record useful lessons
with action/observation evidence, without duplicating every action log. Recheck shared experience against
the current scene; old absolute coordinates and prior success are not current evidence.
"""

SANDBOX_INSTRUCTIONS = """Use native shell and file tools for computation, files and memory inside your sandbox.
Read the public Harness implementation under /harness. Install Python packages with python3 -m pip;
packages and working files stay in your sandbox. Web search and general network access are unavailable.
"""

def _local_source_instructions(source_root, *, allow_shell=True, allow_external_read=True):
    """Replaces ``SANDBOX_INSTRUCTIONS`` when the actor runs without OS sandbox (Harness source under ``source_root``)."""
    if allow_shell and allow_external_read:
        return ("Use native shell and file tools for computation, files and memory in your working directory.\n"
                f"Read the public Harness implementation under {source_root}.\n"
                "Code runs without OS sandbox isolation. Web search is disabled.\n")
    tool_line = (
        "Use native shell and file tools for computation, files and memory in your working directory.\n"
        if allow_shell
        else "Use allowed file tools inside your working directory for files and memory, and program_run for Python "
             "computation; native shell tools are disabled.\n"
    )
    if allow_external_read:
        source_line = f"Read the public Harness implementation under {source_root}.\n"
        tail_line = "Code runs without OS sandbox isolation. Web search is disabled.\n"
    else:
        source_line = (
            "File access is confined to your working directory (external directory reads are disabled); "
            "read workspace references (such as tool_reference.md, robot.urdf, and any workspace actor-source copy) "
            "inside your working directory.\n"
        )
        tail_line = "Web search is disabled.\n"
    return tool_line + source_line + tail_line


#: Replaces ``SANDBOX_INSTRUCTIONS`` / the local-source block for an agent NI-RUA does not run (``agent.backend``
#: ``"external"``): NI-RUA knows nothing of that agent's tools, files, sandbox or network, so the text claims none of
#: them, and it names no run-directory path (the published snapshot lies under the run's ``private/``).
EXTERNAL_AGENT_SOURCE_INSTRUCTIONS = (
    "NI-RUA does not run or restrict this agent. The NI-RUA Harness source is public and may be read;\n"
    "run directories, simulator assets and evaluator state are not to be read.\n"
    "Helper APIs such as client.artifact(ref) are available only inside program_run Python programs, not as top-level MCP tools.\n")

#: 外部后端在 ``features.robot_urdf_delivery="file"`` 时替换简报中的内联 URDF 声明（不暴露本地路径或运行目录）。
EXTERNAL_URDF_ON_DEMAND_CLAUSE = "The original URDF is available on demand via the read-only robot_information tool,"
#: 外部后端按需读取 URDF 的几何指引（与 ``EXTERNAL_URDF_ON_DEMAND_CLAUSE`` 配套，绝不宣称本地工作目录或路径）。
EXTERNAL_URDF_ON_DEMAND_GUIDANCE = (
    "The original URDF (metres, radians, kilograms) is omitted from this startup briefing to keep context compact; "
    "call the read-only robot_information tool on demand when you need full URDF joint or inertial records. "
    "Read its STL/GLB meshes from the native ZIP with client.artifact using resource_ref when you need surface geometry. "
)


# Only for an embodiment with more than one arm.
SHARED_TIMELINE_SENTENCE = ("To move both arms together, put their stages in ONE robot_action batch; both arms execute\n"
                            "on its shared timeline. ")


def _tools_section(arm_count, source_root, external_agent=False, *, pose_form="grasp_angles", supports_6r_ik=True,
                   allow_shell=True, allow_external_read=True):
    source = (EXTERNAL_AGENT_SOURCE_INSTRUCTIONS if external_agent else SANDBOX_INSTRUCTIONS if source_root is None
              else _local_source_instructions(source_root, allow_shell=allow_shell,
                                              allow_external_read=allow_external_read))
    if pose_form == "grasp_angles" and supports_6r_ik:
        program_sentence = (
            "Use program_run (`import scene3d_helpers as s3`) on demand for fast 3D point-cloud queries, "
            "6R source-to-destination reachability screening (`s3.check_source_to_dest`, `s3.solve_reachable_pose`), "
            "multi-candidate `mode='preflight'` screening, repetitive multi-cycle manipulation loops, or coordinating "
            "multi-step 3D geometry and actions; call top-level `robot_action` directly when you want immediate rendered "
            "trajectory overlays and post-action RGB views in the main turn. Physical actions must go through the supplied Harness tools.\n"
        )
    elif pose_form == "grasp_angles":
        program_sentence = (
            "Use program_run (`import scene3d_helpers as s3`) on demand for fast 3D point-cloud queries, "
            "multi-candidate `mode='preflight'` screening, repetitive multi-cycle manipulation loops, or coordinating "
            "multi-step 3D geometry and actions; call top-level `robot_action` directly when you want immediate rendered "
            "trajectory overlays and post-action RGB views in the main turn. Physical actions must go through the supplied Harness tools.\n"
        )
    else:
        program_sentence = "Use program_run for loops that call robot tools. Physical actions must go through the supplied Harness tools.\n"
    return ("## Tools and execution\n"
            "Call the supplied tools directly. Their results contain text and independent images; use the images\n"
            "as visual feedback. No JavaScript wrapper, image extraction or base64 printing is needed.\n"
            + source
            + program_sentence
            + "Python client.call returns success, kind, data and display contentItems. Read typed data directly:\n"
            "for kind='action', receipt is reply['data']['receipt']; for kind='plan', data holds the preview.\n"
            "Missing/untyped data is unknown; never infer execution from display text or success alone.\n"
            "program_run/program_status forward tool preview/history images\n"
            "separately and return the final observation; do not print image base64 into stdout.\n"
            "Await separate physical-action tool calls\n"
            "sequentially. " + (SHARED_TIMELINE_SENTENCE if arm_count > 1 else "")
            + "Never execute or retry an action while parsing its return.\n"
            "Tool success is distinct from physical success.\n")


#: The ARX-X5 "Motion planning" and "Coordinates and timing" sections (grasp_angles poses, contact sections, two
#: arms, 88 mm jaw, env_origin / link6). A backend's ``actor_motion_instructions()`` replaces both.
ARX_X5_MOTION_INSTRUCTIONS = """## Motion planning
robot_action is the common interface for arm and gripper motion. Before choosing an arm or introducing a cross-arm
transfer, check whether ONE single arm can reach both source and destination (via preview or `s3.check_source_to_dest`,
as objects near the midline `|x| <= 0.18 m` are often directly deliverable by the destination-side arm with zero
handoff); when cross-arm transfer is physically required, follow the Run handoff policy. Match batch granularity to
physical uncertainty rather than fragmenting into 1-stage nudges: combine clear multi-stage manipulation sequences
(approach -> grasp -> lift -> transport -> place -> release -> withdraw) into one `robot_action` batch with conditional
`grip_guard_preset` and execute in a single turn via `robot_action(mode='execute_if_clean', plan_update={...},
observation_id=..., stages=[...])` (or `mode='direct'` / `program_run(allow_direct_actions=True)`). `execute_if_clean`
plans the full batch, checks kinematic/joint-limit `motion_safety` and any explicit `obstacle_object_ids`/`obstacles`,
and executes immediately when `blocking_risks` is empty while non-blocking `advisory_notes` do not block execution
(returning cached `plan_id` without moving if `blocking_risks` is non-empty). Pass `obstacle_object_ids=[...]` only for
non-target intermediate objects to route over/around (each listed ID becomes an observation-scoped solid world AABB
cuboid; never pass the target object or hollow racks, hooks, slots, or open containers being entered). Reserve two-turn `mode='preview'` -> `mode='execute'` for inspecting rendered
trajectory overlays when swept-arm clearance around complex 3D structures is genuinely ambiguous, and use zero-image
`mode='preflight'` (or `program_run`) for fast multi-candidate kinematic checks. Omit `image_delivery` on physical
`robot_action` calls so post-action `cam_head`, both close-range wrist RGB views, and trajectory feedback arrive
together automatically.
Omit redundant `observe` checkpoints on jaw opening, open-area release-and-retract, or at the end of a batch (every
physical `robot_action` already returns updated `cam_head`, both wrist RGB views, `grip_facts`, and `obs3d`). Stage
closed-loop visual verification right before high-precision contact: use `checkpoint_after_gripper=true` when
deformable sheet/edge co-motion is uncertain before lifting or folding, or place a single pre-entry `observe` stage at
hover/align (inspecting the close-range wrist RGB views or `robot_observe(focus_3d=...)` + two-view `triangulate`)
before sub-centimetre hole/slot/post/thread entry; resume with `mode='execute'` and the same `plan_id` if aligned, or
pass `source_plan_id` + `start_from_stage_id` (with `stage_overrides`) to recompile only the remaining stages.
Execution never silently replans.
Completed stages report objective `grip_facts` (`measured_jaw_before`, `measured_jaw_after`, `jaw_delta`,
`measured_aperture_m`, `aperture_delta_m`; ARX-X5 full open is 0.088 m, closed is 0.0 m). Match `grip_guard_preset` to
incompressible thickness: reserve `non_empty` (>=0.0015 m), `rigid_object` (>=0.002 m), and `wide_object` (>=0.003 m) strictly for incompressible
rigid blocks/cylinders >=8 mm thick so empty closures (~0.0 m) pause while valid grasps continue into lift; compressible
or soft items, thin items (<=4 mm), slender rods/tubes (<=8 mm), and deformable sheets pinch to 0.000–0.002 m, so use
`thin_object` (>=0.0005 m) or `none` to avoid false `aperture_below_min_m` pauses. When rotating an offset or heavy object
in mid-air, seat fingers deeper near its center of mass (`contact_region='finger_middle'` or positive `contact_depth_m`
when support clearance allows). On open-jaw grasp moves (`allow_jaw_symmetry=true` by default; set `false` when finger
polarity must stay fixed), the planner automatically resolves the exact 180° parallel-jaw twin if the primary branch hits
IK or wrist-winding limits (and resolves small contact-invariant approach tilts only when `allow_approach_tilt=true` is
opted in, reporting `resolved_angles_deg`), while closed-jaw/holding moves keep rigid orientation. Inside open-top containers,
keep `[0, 0, yaw]` vertical during descent/release when reachable so the forearm stays vertical above surrounding rims/flaps; when slanted/lateral
geometry or extended reaches constrain a top-down approach, use a 3D approach tilt (`[pitch, roll, yaw]`, e.g. via
`s3.solve_reachable_pose` / `s3.check_source_to_dest` or `allow_approach_tilt=true`). When withdrawing from a narrow slot or housing,
retract clear along the entry axis before opening the jaws wide so open fingers do not snag surrounding walls. For repetitive
multi-cycle operations, pack multiple cycles into one `robot_action` batch (up to 32 stages) or run verified loops in
`program_run` at normal speed to conserve step and turn budgets. Match joint-travel bounds to the maneuver: `standard`
catches unintended 360° wrist winding on short moves; set `joint_travel_preset` (`'extended'` or `'unconstrained'`) or
custom `max_joint_travel_rad`/`max_backtracking_rad` when wide-arc sweeps, large orientation flips, or deformable
stretches are intentional; if final return to initial `obs_1['robot_state']` flange poses hits `path_infeasible` from a
rotated posture, use `s3.plan_joint_safe_return` (`safe_transition_stages`).
Stages on each arm are ordered; `depends_on` adds cross-arm prerequisites. Independent work can overlap.
Robot-model clearance and 2D overlays do not prove task-object clearance, contact, support or retention.

## Coordinates and timing
Positions are metres in env_origin. Use `grasp_position_m` + `angles_deg` for all task poses; reserve raw EE targets
(`position_m` + `quaternion_xyzw`) strictly for replaying saved measured `link6` `ee_pose` dicts (such as final return,
since `angles_deg=[0,0,0]` maps to `TOP_DOWN_QUAT_WXYZ=[0.5,-0.5,0.5,0.5]`, not identity quaternion).
Grasp targets describe the center BETWEEN opposing fingers at the selected contact section;
the tool converts its link6 offset once, so do not add a manual offset. angles_deg=[pitch,roll,yaw]
are extrinsic world x,y,z rotations from the top-down reference, not conventional absolute Euler angles.
[0,0,0] points tool +x down and opens the jaws along world -x. contact_region selects pad_center,
finger_middle or fingertip; contact_depth_m is the alternative depth toward the palm from pad_center.
An omitted arm speed uses this run's default; speed_preset and speed_scale are alternatives.
Use independent jaw speed settings. Prefer normal clear transit and slower short contact approaches;
slower commands cost more native steps. during-motion jaw schedules are command timing, not contact detection.
Match arrival tolerances and stall guards to each stage's orthogonal contact physics:
- `normal` (0.010 m, 0.15 rad, default): standard workspace moves, surface grasps (add `stall_guard_preset='contact_stop'` on low-clearance support descents `fingertip_support_clearance_m <= 0.010` m so real contact stops and holds pose while keeping `0.010` m planar centering), and loaded lift/transport/hover moves (pair `max_downward_dz_m` in `[0, 0.15]` m, e.g. `0.020–0.060` m, when held payload weight causes static joint-PD `-Z` gravity sag while horizontal `(dx, dy)` stays aligned).
- `tight` (0.003 m, 0.05 rad): millimeter free-space pre-entry or handoff alignment before contact (pair `max_downward_dz_m` if held payload weight causes static `-Z` gravity sag at alignment; avoid `tight` on physical contact seating where contact reaction torque deflects wrist PD).
- `compliant` (0.015 m, 0.20 rad, `max_upward_dz_m=0.040` m): contact seating and placing onto support where physical support stops Z above target (increase explicit `max_upward_dz_m` up to `0.15` m when deeper upward contact stop travel is expected; keep `normal` rather than `compliant` on small/thin grasps so a `+0.040` m window does not accept stopping above the object).
- `coarse` (0.025 m, 0.35 rad): unloaded transit, approach waypoints, post-release retraction, or compliant lateral travel (and
  `unconstrained_orientation` when wrist yaw is free)."""

# How a geometry_query gripper_relative result becomes a move target, per accepted pose form.
GRIPPER_RELATIVE_TARGET_SENTENCES = {
    "grasp_angles": ("jaw_midplane_normal): for a held object or tool tip, subtract point_minus_grasp_world_m from the "
                     "destination point to\ncompute the required grasp_position_m at the same gripper orientation.\n"),
    "tcp_quaternion": ("jaw_midplane_normal): for a held object or tool tip at the same gripper orientation, the move "
                       "position_m that\nputs that point on a destination is the observation's measured ee_pose "
                       "position_m plus (destination - that point).\n"),
}
#: The budget sentence of the feedback section, per ``EmbodimentProfile.clock``.
BUDGET_SENTENCES = {
    "simulated_steps": "Keep enough native-step budget for the remaining operation, verification and any task-required final return.\n",
    "real_time": "Keep enough action and time budget for the remaining operation, verification and any task-required final return.\n",
}


def _feedback_section(pose_form, clock):
    triangulate_helper = " or s3.triangulate_rays" if pose_form == "grasp_angles" else ""
    return ("## Feedback and next decision\n"
            "Action results include actual completion/partial execution and the updated observation (RGB images, measured\n"
            "robot_state, grip_facts, trajectory feedback, and inline 3D object_table/object_state_delta/scene3d when enabled).\n"
            "Inspect that feedback before requesting another observation or correcting a target. Distinguish planned commands,\n"
            "measured state, and your own object judgments; missing evidence remains unknown. After failure use actual state and\n"
            "the completed range; do not replay completed work.\n"
            "When sub-centimetre local geometry (insert holes, narrow slots, connector ports, elevated hooks) or\n"
            "deformable cloth folds/corners require visual detail beyond coarse 3D boxes and 1 cm voxels, request 1:1 native\n"
            "sensor-resolution local crops on demand via robot_observe(focus_3d={'target_m': [x, y, z], 'radius_m': ...}) or\n"
            "robot_observe(focus_3d={'object_id': 'o2'}) (or zero-step history_view_observation(..., focus_3d={...})); after picking\n"
            "[local_u, local_v] on a crop, convert back to full-frame pixel coordinates via\n"
            "full_uv = [crop_xywh_px[0] + local_u, crop_xywh_px[1] + local_v] for direct use in geometry_query (triangulate /\n"
            f"pixel_plane){triangulate_helper}.\n"
            "For 3D localization with geometry_query, prefer triangulate across two calibrated views when a 3D object feature is\n"
            "matchable in both; use pixel_plane on demand when a target/destination lies on a known support plane or single-view\n"
            "occlusion prevents stereo matching (a single projected point on target_camera_id lies on the source ray and only tests\n"
            "the assumed plane height along the cyan normal-offset band, so use plane_origin_alt_m or triangulate when object top vs\n"
            "table height matters). Both triangulate and pixel_plane return gripper_relative (point_minus_grasp_world_m and\n"
            + GRIPPER_RELATIVE_TARGET_SENTENCES[pose_form]
            + "The latest action receipt is already inline; in multi-step recovery or long episodes, use history_list_actions\n"
            "with anchor='failures', 'gripper_commands' or 'events' on demand to compare earlier attempts and measured_delta.\n"
            + BUDGET_SENTENCES[clock]
            + "Images of historical frames retain historical state; current budgets are separately identified.\n"
            "Use normal task progress to test your assumptions where possible: a useful approach, lift or withdrawal\n"
            "can also provide evidence. Add a separate probe only if its answer changes the next decision and is not\n"
            "available in existing feedback. Before a long batch, place an observation checkpoint only where later\n"
            "stages depend on uncertain earlier outcomes; deterministic independent stages can stay together.\n")


PROGRAMS_INSTRUCTIONS = """## Programs and reviews
program_run defaults to returning new actions for YOUR review as the main actor, not human approval.
Inspect the returned evidence and autonomously resume with approve/reject; do not wait for a person.
Only an explicit allow_direct_actions at program start delegates direct execution for that program.
Program work time excludes its own helper-model and main-model review waits; cancellation remains available.
Program exit status does not prove its physical actions succeeded. Use client.action(..., require_completed=True)
(raises ActionIncompleteError on non-completed receipts) or stop loops on failure, pause or lack of progress.
Direct-action programs do not receive a new model decision between inner actions. Put evidence-dependent
branching in the program or return to actor review when judgment is needed; a stored concern is not a monitor."""


def _execution_sections(*, arm_count, source_root, motion, pose_form, clock, operation_memory, external_agent=False,
                        supports_6r_ik=True, allow_shell=True, allow_external_read=True):
    """Tools .. reusable experience (the part between the task plan and the observation contract)."""
    return (_tools_section(arm_count, source_root, external_agent, pose_form=pose_form, supports_6r_ik=supports_6r_ik,
                           allow_shell=allow_shell, allow_external_read=allow_external_read)
            + "\n" + motion.rstrip("\n") + "\n\n"
            + _feedback_section(pose_form, clock) + "\n" + PROGRAMS_INSTRUCTIONS
            + (OPERATION_MEMORY_INSTRUCTIONS if operation_memory else "\n"))


#: ARX-X5 rendering of the execution sections (sandbox on, operation memory on).
INSTRUCTIONS = _execution_sections(arm_count=2, source_root=None, motion=ARX_X5_MOTION_INSTRUCTIONS,
                                   pose_form="grasp_angles", clock="simulated_steps", operation_memory=True)

CALIBRATED_OBSERVATION_INSTRUCTIONS = """Your observation contract supplies RGB, robot proprioception, exact camera intrinsics/extrinsics and hand-eye transforms, and the disclosed robot model. Use geometry_query to derive estimates from these permitted inputs. Calibration does not supply object depth, identity, pose, contact truth or evaluator state. Triangulation requires valid corresponding visible points and sufficient parallax; state uncertainty and assumptions."""


def _legacy_observation(tcp_link, clock):
    hidden = "simulator scene state" if clock == "simulated_steps" else "hidden object state"
    return (f"Your observation contract is RGB, measured arm joints and {tcp_link} poses, and your gripper command "
            "history. Static robot structure is separately supplied by robot_information; it does not supply current "
            f"base or camera calibration. Exact camera intrinsics, extrinsics, hand-eye transforms, {hidden}, contact "
            "truth and scoring internals are unavailable. Do not assume metric depth or invent calibration; use visible "
            "evidence and report uncertainty.")


#: ARX-X5 rendering of ``legacy_observation_instructions``.
LEGACY_OBSERVATION_INSTRUCTIONS = _legacy_observation("link6", "simulated_steps")


def _profile(profile):
    if profile is None:
        from ...robot.model.embodiments import ARX_X5_PROFILE
        return ARX_X5_PROFILE
    return profile


def legacy_observation_instructions(profile=None):
    """Observation contract of the uncalibrated actor profile (RGB + proprioception) for ``profile``
    (``EmbodimentProfile``; None = ARX-X5): names the measured TCP link (``profile.tcp_link``) and, on a simulated
    step clock, the simulator scene state among the unavailable inputs. ARX-X5: ``LEGACY_OBSERVATION_INSTRUCTIONS``."""
    profile = _profile(profile)
    return _legacy_observation(profile.tcp_link, profile.clock)


def _supports_6r_ik(profile):
    from ...perception.external.scene3d import ik_prefilter_staged
    return ik_prefilter_staged(profile)


def instructions(observation_contract, handoff_text=None, *, source_root=None, operation_memory=True,
                 embodiment_motion=None, profile=None, external_agent=False, allow_shell=True,
                 allow_external_read=True):
    """Startup contract (module docstring) for the run's embodiment.

    ``observation_contract``: text of the ``## Observation contract`` section (``CALIBRATED_OBSERVATION_INSTRUCTIONS``,
    ``legacy_observation_instructions(profile)`` or the depth contract), used verbatim. ``handoff_text``: the run's
    handoff policy text (``task_notes.handoff_policy(...)["text"]``), or None for no ``## Run handoff policy`` section;
    a text for an embodiment without a cross-arm handoff (``profile.handoff`` false) raises
    ``ValueError("handoff_policy_requires_cross_arm_embodiment")``. ``source_root``: None when the actor runs in the
    OS sandbox (Harness source at /harness), else the directory holding the published Harness source.
    ``operation_memory``: False only when the operation_memory tool is disabled. ``embodiment_motion``: the backend's
    own "Motion planning" + "Coordinates and timing" text (``RobotBackend.actor_motion_instructions()``); None uses
    the ARX-X5 sections, which a profile without the ``grasp_angles`` pose input refuses
    (``ValueError("embodiment_motion_section_required")``). ``profile``: ``EmbodimentProfile``; None = ARX-X5.
    ``external_agent``: True for an agent NI-RUA does not run (``agent.backend="external"``): the source-access block
    is ``EXTERNAL_AGENT_SOURCE_INSTRUCTIONS`` and ``source_root`` is ignored; default False (every hosted agent).
    ``allow_shell`` / ``allow_external_read``: False when host confinement disables native shell or reads outside the
    actor working directory (affects ``_local_source_instructions`` only).
    """
    profile = _profile(profile)
    if handoff_text is not None and not profile.handoff:
        raise ValueError("handoff_policy_requires_cross_arm_embodiment")
    grasp_angles = "grasp_angles" in profile.pose_inputs
    if embodiment_motion is None and not grasp_angles:
        raise ValueError("embodiment_motion_section_required")
    supports_6r_ik = _supports_6r_ik(profile)
    execution = _execution_sections(
        arm_count=len(profile.arms), source_root=source_root,
        motion=ARX_X5_MOTION_INSTRUCTIONS if embodiment_motion is None else embodiment_motion,
        pose_form="grasp_angles" if grasp_angles else "tcp_quaternion", clock=profile.clock,
        operation_memory=operation_memory, external_agent=external_agent,
        supports_6r_ik=supports_6r_ik, allow_shell=allow_shell, allow_external_read=allow_external_read)
    text = (_task_section(profile.clock) + "\n## Task plan and current concerns\n" + planning_instructions(profile.clock)
            + "\n\n" + execution + "\n## Observation contract\n" + observation_contract)
    if handoff_text is not None:
        text += "\n\n## Run handoff policy\n" + handoff_text
    return text


OPERATION_NOTES = """Reusable operation experience: select only the rows that matter now and reselect them when
the scene or evidence changes. These are fallible reminders, not a fixed sequence,
extra completion rules or an action checklist. Turn a selected row into a short
task_plan attention entry only when it affects the next decision; normal task
progress can provide the check.

| Situation | Visible check | Adjustment when evidence disagrees |
| --- | --- | --- |
| Task, roles or geometry are uncertain | Match the public task to the visible object, support and destination; prefer two-view triangulate when matchable (for sub-centimetre holes, slots, ports, hooks or cloth folds/corners, request 1:1 local crops via robot_observe(focus_3d={...}) or history_view_observation(..., focus_3d={...}) and convert crop coordinates [local_u, local_v] via full_uv = [crop_xywh_px[0] + local_u, crop_xywh_px[1] + local_v]), or use pixel_plane with cross-camera projection checks on demand when a target lies on a known support plane. A robot contact reference or image overlap is not object contact. | Correct the task interpretation or geometric relation; inspect a targeted view or focus_3d crop only when it can distinguish the alternatives. |
| Grasp, transport or cooperation is uncertain | Check target co-motion and in-place progress (via RGB visual feedback, grip_facts, object_state_delta and trajectory feedback), retention through turns, fingertip/support clearance and whole-arm/held-object sweep. Execute clear open-area batches in one turn via mode='execute_if_clean' or mode='direct' (pass obstacle_object_ids only for non-target intermediate obstacles to cross over, never for the target object or hollow racks/hooks/slots/open containers being entered), and inspect rendered trajectory overlays via mode='preview' -> mode='execute' when swept-arm clearance around complex 3D structures is ambiguous. Choose the contact section from object height and mass distribution (deeper contact is not automatically better: `contact_region='finger_middle'` reduces slip during mid-air wrist rotations when support clearance permits, whereas shallow `pad_center` + `contact_stop` avoids support collision on low-clearance surface picks); on open-jaw grasp moves (`allow_jaw_symmetry=true` by default; set `false` for strict finger polarity), the planner automatically resolves the exact 180° parallel-jaw twin (and small contact-invariant approach tilts when `allow_approach_tilt=true` is opted in); keep `[0, 0, yaw]` vertical during descent/release inside open-top containers when reachable so the forearm clears rims/flaps, and tilt `[pitch, roll, yaw]` when slanted/lateral geometry or extended reaches constrain a top-down approach. Match verification to incompressible thickness: reserve rigid_object/wide_object for incompressible rigid blocks/cylinders >=8 mm thick; for compressible/soft items, thin items (<=4 mm), slender rods/tubes (<=8 mm), or deformable sheets where a pinched jaw closes to 0.000–0.002 m, use thin_object or none (or pause with checkpoint_after_gripper=true when deformable/edge co-motion is uncertain before lifting or folding), checking visual RGB evidence alongside grip_facts (measured_aperture_m). | Change the contact, support, arm or route hypothesis before repeating a failed pattern; lift clear of the support and neighbors before lateral transport or folding, and retract clear along the entry axis before opening jaws wide inside narrow slots/housings, as a closed jaw or initial lift is not proof of retention. |
| Working arm cannot reach or object needs two arms | Before picking, check whether one single arm can reach both source and destination (via full-batch preview or s3.check_source_to_dest), or whether a wide/deformable object needs synchronized two-arm support or folding. | Choose the single arm that reaches both source and destination with zero handoff when available; when cross-arm transfer is genuinely needed, follow the Run handoff policy; put synchronized two-arm grasps/folds in one shared-timeline robot_action batch, and use joint_travel_preset='extended' (or custom max_joint_travel_rad/max_backtracking_rad) when wide-arc sweeps, large orientation flips or deformable folding require wider joint travel. |
| Placement or release is uncertain | Check the required position/orientation, support or rim clearance, then separation and continued presence after withdrawal; execute open-area placements uninterrupted without trailing observe checkpoints, while placing at most one pre-entry hover observe when sub-centimetre hole/slot/post/stack clearance requires visual verification of in-hand alignment. Near the destination is not completion. | Keep the placement unresolved and choose a useful correction or verification; an open jaw and completed receipt do not prove release. |
| A motion or expected event fails | Compare the measured arrival (such as actual_contact_position_m, approach_shortfall_m, endpoint_residual_m and first_divergence) and visible event with the intended target, or use history_list_actions(anchor='failures' or 'gripper_commands') on demand in multi-step recovery to compare prior measured_delta; count observed object events, not commands. | State a cause hypothesis and choose a distinguishing observation or changed mechanism. Match orthogonal tolerances to stage physics: use tolerance_preset='coarse' (0.025 m, 0.35 rad) for unloaded transit, approach waypoints or retract, 'normal' (0.010 m, 0.15 rad) for surface grasps (adding stall_guard_preset='contact_stop' on low-clearance support descents) and loaded lift/transport/hover moves (pairing max_downward_dz_m in [0, 0.15] m when holding a payload that sags in -Z under gravity), 'tight' (0.003 m, 0.05 rad) for millimeter free-space pre-entry or handoff alignment before contact, and 'compliant' (0.015 m, 0.20 rad, max_upward_dz_m=0.040 m, or explicit max_upward_dz_m up to 0.15 m) for contact seating onto support; if final return to initial flange poses hits path_infeasible, use s3.plan_joint_safe_return (safe_transition_stages). Read endpoint_residual_m={dx_m, dy_m, dz_m} by axis: horizontal (dx_m, dy_m) error means planar misalignment, whereas vertical dz_m (>0 stopped higher on contact, <0 sagged under load) reflects vertical contact or payload compliance. If a motion stops while an object is still held between the fingers (in_hand_tracking_error), lift clear and adjust horizontal alignment before descending again rather than releasing at a blocked pose. |
| Parallel work or budget matters | Compare dependencies, shared sweeps and the remaining cost of manipulation, verification, recovery and required return. | Overlap independent work when safe; otherwise sequence it and preserve reserve for correction and verification. |

Budget for the whole remaining operation, verification, recovery and any task-required
final return. Avoid unnecessary intermediate home returns between subtasks; keep working
arms in clear workspace postures and return to initial poses only when required at final
completion. Recheck object order, reachability and whole-arm/held-object clearance as the scene changes."""


def operation_notes(profile=None):
    """Embodiment-aware rendering of ``OPERATION_NOTES`` (ARX-X5 / None returns ``OPERATION_NOTES`` verbatim)."""
    if profile is None:
        return OPERATION_NOTES
    dual_arm = len(profile.arms) > 1
    has_handoff = bool(profile.handoff)
    supports_6r_ik = _supports_6r_ik(profile)
    has_obstacles = "obstacle_object_ids" in profile.supports
    has_joint_travel = "joint_travel" in profile.supports
    has_execute_if_clean = "execute_if_clean" in profile.supports
    has_stall_guard = "stall_guard" in profile.supports
    has_max_upward_dz = "max_upward_dz" in profile.supports
    has_reach = "reach" in profile.supports
    grasp_angles = "grasp_angles" in profile.pose_inputs
    has_home_check = profile.home_check is not None
    if (
        dual_arm
        and has_handoff
        and supports_6r_ik
        and has_obstacles
        and has_joint_travel
        and has_execute_if_clean
        and has_stall_guard
        and has_max_upward_dz
        and grasp_angles
        and has_home_check
    ):
        return OPERATION_NOTES

    row_task = (
        "| Task, roles or geometry are uncertain | Match the public task to the visible object, support and "
        "destination; prefer two-view triangulate when matchable (for sub-centimetre holes, slots, ports, hooks or "
        "cloth folds/corners, request 1:1 local crops via robot_observe(focus_3d={...}) or "
        "history_view_observation(..., focus_3d={...}) and convert crop coordinates [local_u, local_v] via "
        "full_uv = [crop_xywh_px[0] + local_u, crop_xywh_px[1] + local_v]), or use pixel_plane with cross-camera "
        "projection checks on demand when a target lies on a known support plane. A robot contact reference or image "
        "overlap is not object contact. | Correct the task interpretation or geometric relation; inspect a targeted "
        "view or focus_3d crop only when it can distinguish the alternatives. |"
    )

    grasp_title = "Grasp, transport or cooperation is uncertain" if dual_arm else "Grasp or transport is uncertain"
    exec_clause = (
        "Execute clear open-area batches in one turn via mode='execute_if_clean' or mode='direct'"
        if has_execute_if_clean
        else "Execute clear open-area batches via mode='direct' or mode='preview' -> mode='execute'"
    )
    obstacle_clause = (
        " (pass obstacle_object_ids only for non-target intermediate obstacles to cross over, never for the target "
        "object or hollow racks/hooks/slots/open containers being entered)"
        if has_obstacles
        else ""
    )
    contact_clause = (
        " Choose the contact section from object height and mass distribution (deeper contact is not automatically "
        "better: `contact_region='finger_middle'` reduces slip during mid-air wrist rotations when support clearance "
        "permits, whereas shallow `pad_center`"
        + (" + `contact_stop`" if has_stall_guard else "")
        + " avoids support collision on low-clearance surface picks);"
        + (
            " on open-jaw grasp moves (`allow_jaw_symmetry=true` by default; set `false` for strict finger polarity), "
            "the planner automatically resolves the exact 180° parallel-jaw twin (and small contact-invariant approach "
            "tilts when `allow_approach_tilt=true` is opted in);"
            if supports_6r_ik
            else ""
        )
        + " keep `[0, 0, yaw]` vertical during descent/release inside open-top containers when reachable so the "
        "forearm clears rims/flaps, and tilt `[pitch, roll, yaw]` when slanted/lateral geometry or extended reaches "
        "constrain a top-down approach."
        if grasp_angles
        else " Choose grasp depth and approach orientation from object height, support clearance and jaw geometry."
    )
    row_grasp = (
        f"| {grasp_title} | Check target co-motion and in-place progress (via RGB visual feedback, grip_facts, "
        f"object_state_delta and trajectory feedback), retention through turns, fingertip/support clearance and "
        f"whole-arm/held-object sweep. {exec_clause}{obstacle_clause}, and inspect rendered trajectory overlays via "
        f"mode='preview' -> mode='execute' when swept-arm clearance around complex 3D structures is ambiguous."
        f"{contact_clause} Match verification to incompressible thickness: reserve rigid_object/wide_object for "
        "incompressible rigid blocks/cylinders >=8 mm thick; for compressible/soft items, thin items (<=4 mm), slender "
        "rods/tubes (<=8 mm), or deformable sheets where a pinched jaw closes near empty-close aperture, use "
        "thin_object or none (or pause with checkpoint_after_gripper=true when deformable/edge co-motion is uncertain "
        "before lifting or folding), checking visual RGB evidence alongside grip_facts (measured_aperture_m). | "
        + (
            "Change the contact, support, arm or route hypothesis before repeating a failed pattern; "
            if dual_arm
            else "Change the contact, support or route hypothesis before repeating a failed pattern; "
        )
        + "lift clear of the support and neighbors before lateral transport or folding, and retract clear along the "
        "entry axis before opening jaws wide inside narrow slots/housings, as a closed jaw or initial lift is not "
        "proof of retention. |"
    )

    if dual_arm:
        check_reach = (
            "via full-batch preview or s3.check_source_to_dest"
            if supports_6r_ik
            else "via full-batch preview" + (" or mode='reach'" if has_reach else "")
        )
        handoff_clause = (
            "when cross-arm transfer is genuinely needed, follow the Run handoff policy; "
            if has_handoff
            else ""
        )
        travel_clause = (
            ", and use joint_travel_preset='extended' (or custom max_joint_travel_rad/max_backtracking_rad) when "
            "wide-arc sweeps, large orientation flips or deformable folding require wider joint travel"
            if has_joint_travel
            else ""
        )
        row_reach = (
            f"| Working arm cannot reach or object needs two arms | Before picking, check whether one single arm can "
            f"reach both source and destination ({check_reach}), or whether a wide/deformable object needs "
            f"synchronized two-arm support or folding. | Choose the single arm that reaches both source and "
            f"destination when available; {handoff_clause}put synchronized two-arm grasps/folds in one "
            f"shared-timeline robot_action batch{travel_clause}. |"
        )
    else:
        reach_tool = "mode='reach' or full-batch mode='preview'" if has_reach else "full-batch mode='preview'"
        travel_clause = (
            " Use joint_travel_preset='extended' (or custom max_joint_travel_rad/max_backtracking_rad) when wide-arc "
            "sweeps or large orientation flips require wider joint travel."
            if has_joint_travel
            else ""
        )
        row_reach = (
            f"| Target pose is hard to reach or near workspace limits | Before committing to a grasp or placement, "
            f"verify reachability across source and destination poses (via {reach_tool}) and check approach "
            f"orientation and wrist/forearm clearance. | Adjust approach orientation, contact section, or an "
            f"intermediate staging/regrasp pose on a stable support when a direct grasp-to-place posture hits "
            f"kinematic or workspace limits.{travel_clause} |"
        )

    row_place = (
        "| Placement or release is uncertain | Check the required position/orientation, support or rim clearance, then "
        "separation and continued presence after withdrawal; execute open-area placements uninterrupted without "
        "trailing observe checkpoints, while placing at most one pre-entry hover observe when sub-centimetre "
        "hole/slot/post/stack clearance requires visual verification of in-hand alignment. Near the destination is not "
        "completion. | Keep the placement unresolved and choose a useful correction or verification; an open jaw and "
        "completed receipt do not prove release. |"
    )

    stall_note = (
        " (adding stall_guard_preset='contact_stop' on low-clearance support descents)"
        if has_stall_guard
        else ""
    )
    tight_scope = "millimeter free-space pre-entry or handoff alignment" if has_handoff else "millimeter free-space pre-entry alignment"
    compliant_note = (
        ", max_upward_dz_m=0.040 m, or explicit max_upward_dz_m up to 0.15 m"
        if has_max_upward_dz
        else ""
    )
    safe_return_note = (
        "; if final return to initial flange poses hits path_infeasible, use s3.plan_joint_safe_return "
        "(safe_transition_stages)"
        if supports_6r_ik and has_home_check
        else ""
    )
    row_fail = (
        "| A motion or expected event fails | Compare the measured arrival (such as actual_contact_position_m, "
        "approach_shortfall_m, endpoint_residual_m and first_divergence) and visible event with the intended target, "
        "or use history_list_actions(anchor='failures' or 'gripper_commands') on demand in multi-step recovery to "
        "compare prior measured_delta; count observed object events, not commands. | State a cause hypothesis and "
        "choose a distinguishing observation or changed mechanism. Match orthogonal tolerances to stage physics: use "
        "tolerance_preset='coarse' (0.025 m, 0.35 rad) for unloaded transit, approach waypoints or retract, 'normal' "
        f"(0.010 m, 0.15 rad) for surface grasps{stall_note} and loaded lift/transport/hover moves (pairing "
        "max_downward_dz_m in [0, 0.15] m when holding a payload that sags in -Z under gravity), 'tight' (0.003 m, "
        f"0.05 rad) for {tight_scope} before contact, and 'compliant' (0.015 m, 0.20 rad{compliant_note}) for contact "
        f"seating onto support{safe_return_note}. Read endpoint_residual_m={{dx_m, dy_m, dz_m}} by axis: horizontal "
        "(dx_m, dy_m) error means planar misalignment, whereas vertical dz_m (>0 stopped higher on contact, <0 sagged "
        "under load) reflects vertical contact or payload compliance. If a motion stops while an object is still held "
        "between the fingers (in_hand_tracking_error), lift clear and adjust horizontal alignment before descending "
        "again rather than releasing at a blocked pose. |"
    )

    if dual_arm:
        row_budget = (
            "| Parallel work or budget matters | Compare dependencies, shared sweeps and the remaining cost of "
            "manipulation, verification, recovery and required return. | Overlap independent work when safe; "
            "otherwise sequence it and preserve reserve for correction and verification. |"
        )
        footer = (
            "Budget for the whole remaining operation, verification, recovery and any task-required\n"
            "final return. Avoid unnecessary intermediate home returns between subtasks; keep working\n"
            "arms in clear workspace postures and return to initial poses only when required at final\n"
            "completion. Recheck object order, reachability and whole-arm/held-object clearance as the scene changes."
        )
    else:
        return_part = " and any task-required final return" if has_home_check else ""
        return_tail = " and return to the initial pose only when required at final completion" if has_home_check else ""
        row_budget = (
            f"| Sequencing or budget matters | Compare subtask dependencies, clearance and the remaining cost of "
            f"manipulation, verification and recovery{return_part}. | Sequence subtasks cleanly and preserve reserve "
            f"for correction and verification. |"
        )
        footer = (
            f"Budget for the whole remaining operation, verification and recovery{return_part}. Avoid unnecessary\n"
            f"intermediate home returns between subtasks; keep the arm in a clear workspace posture{return_tail}.\n"
            "Recheck object order, reachability and arm/held-object clearance as the scene changes."
        )

    return (
        "Reusable operation experience: select only the rows that matter now and reselect them when\n"
        "the scene or evidence changes. These are fallible reminders, not a fixed sequence,\n"
        "extra completion rules or an action checklist. Turn a selected row into a short\n"
        "task_plan attention entry only when it affects the next decision; normal task\n"
        "progress can provide the check.\n\n"
        "| Situation | Visible check | Adjustment when evidence disagrees |\n"
        "| --- | --- | --- |\n"
        f"{row_task}\n"
        f"{row_grasp}\n"
        f"{row_reach}\n"
        f"{row_place}\n"
        f"{row_fail}\n"
        f"{row_budget}\n\n"
        f"{footer}"
    )


def task_notes(notes, *, profile=None):
    """Retain custom notes and archive settings without legacy tool instructions."""
    from copy import deepcopy
    result = deepcopy(notes)
    settings = result.get("config", {})
    parts = [operation_notes(profile)] if settings.get("default_enabled") else []
    if settings.get("custom_text"):
        parts.append(settings["custom_text"])
    result["text"] = "\n\n".join(parts) or None
    if settings.get("default_enabled"):
        settings["default_version"] = "unified_operation_notes_v7"
        sources = ["ni_rua:unified_operation_notes_v7 (ni_rua/actor_instructions.py; docs/development/task-notes.md)"]
        if settings.get("custom_source"):
            sources.append(settings["custom_source"])
        result["source"] = "; ".join(sources)
    return result
