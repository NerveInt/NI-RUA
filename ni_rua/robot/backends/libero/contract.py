"""Pure request compilation, validation, and actor prompt sections for simulation-only ``libero-franka-panda``.

This module contains zero MuJoCo, robosuite, or LIBERO runtime imports so parent-side validation,
action compilation, and prompt construction run cleanly in any Python environment.
"""
from __future__ import annotations

import math
from typing import Any

from ....core.action_validation import (
    MAX_HOLD_NATIVE_STEPS,
    embodiment_or_default,
    finite_vector,
    validate_gripper_motion,
)
from ....core.contracts import new_id
from ....core.embodiment import EmbodimentProfile, PUBLISHED_STALL_GUARD_PRESETS
from ...embodiments.libero_franka_panda import grasp_contact, pose_convention
from ...embodiments.libero_franka_panda.profile import (
    ARM_ID,
    LIBERO_FRANKA_PANDA,
    PANDA_LIMITS_PROFILES,
    libero_franka_panda_profile,
)
from ...motion.grip_guard import (
    CARRY_SLIP_GUARD_FIELDS,
    CLOSING_JAW_GOAL_BELOW,
    DOWNWARD_MOVE_DZ_M,
    GRIP_GUARD_FIELDS,
    GRIP_SQUEEZE_FIELD,
    SLIP_MAX_GAP_ABOVE_EMPTY_CLOSE_M,
    SLIP_MIN_JAW_DROP,
    preset_aperture_bounds,
    validate_batch_carry_slip_guard,
    validate_grip_guard_fields,
    validate_grip_squeeze,
)
from ...motion.motion_speed import SPEED_INPUT_FIELDS
from ...motion.motion_tolerance import MAX_DOWNWARD_DZ_M, MAX_UPWARD_DZ_M

ENVIRONMENT = "libero"
FRAME_ID = LIBERO_FRANKA_PANDA.frame_id
TCP_CONFIG_ID = LIBERO_FRANKA_PANDA.tcp_config_id
ROBOT_MODEL_ID = LIBERO_FRANKA_PANDA.robot_model_id
LIMITS_PROFILE = PANDA_LIMITS_PROFILES[0]
ARM = ARM_ID
ARMS: tuple[str, ...] = (ARM_ID,)
MAX_WAIT_STEPS = MAX_HOLD_NATIVE_STEPS
MAX_WAYPOINTS = 8

WORKSPACE_MIN_M: tuple[float, float, float] = (-1.2, -1.2, -0.6)
WORKSPACE_MAX_M: tuple[float, float, float] = (1.6, 1.2, 1.6)

CONTACT_FIELDS: tuple[str, ...] = ("contact_region", "contact_depth_m")
UNSUPPORTED_LEAF_FIELDS: tuple[str, ...] = (
    "dual_arm_progression",
    "arm_options",
    "waypoint_speed_presets",
    "waypoint_speed_scales",
    "grasp_centre_m",
    "object_width_m",
    "fixed_jaw_clearance_m",
    "grasp_depth_m",
    "fixed_jaw_yaw_deg",
    "input_grasp_centre",
    GRIP_SQUEEZE_FIELD,
)
UNSUPPORTED_BATCH_FIELDS: tuple[str, ...] = (
    "joint_travel_preset",
    "max_joint_travel_rad",
    "max_backtracking_rad",
    "review_proxies",
    "approach_tolerance_rad",
    "roll_tolerance_rad",
    "arrival_correction_rounds",
    "arrival_correction_max_offset_rad",
    GRIP_SQUEEZE_FIELD,
)
PATH_FIELDS: tuple[str, ...] = (
    "kind",
    "orientation",
    "blend_radius_m",
    "waypoints",
    "position_tolerance_m",
    "orientation_tolerance_rad",
    "max_upward_dz_m",
    "max_downward_dz_m",
    "tolerance_preset",
    "stall_guard_preset",
    "max_tracking_divergence_m",
)
DURING_MOTION_EVENT_FIELDS = frozenset({"arm", "phase", "goal_normalized", "start_fraction", "end_fraction"})


def _profile(profile: EmbodimentProfile | None = None) -> EmbodimentProfile:
    if isinstance(profile, EmbodimentProfile) and profile.robot_model_id == ROBOT_MODEL_ID:
        return profile
    return libero_franka_panda_profile()


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _number(value: Any, low: float, high: float, name: str, *, low_open: bool = True) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value > high
        or value < low
        or (low_open and value <= low)
    ):
        bracket = "(" if low_open else "["
        raise ValueError(f"invalid_action: {name} must be in {bracket}{low}, {high}]")
    return float(value)


def _arm(args: dict[str, Any], arms: tuple[str, ...] = ARMS) -> str:
    arm = args.get("arm")
    if arm not in arms:
        raise ValueError(f"unsupported_arm:{arm}")
    return str(arm)


def check_workspace(position_m: list[float] | tuple[float, ...]) -> None:
    for idx, (val, lo, hi, axis) in enumerate(
        zip(position_m, WORKSPACE_MIN_M, WORKSPACE_MAX_M, ("x", "y", "z"))
    ):
        if not _finite(val) or not (lo <= float(val) <= hi):
            raise ValueError(
                f"target_outside_workspace: {axis}={val} m outside [{lo}, {hi}] m in {FRAME_ID}"
            )


def offered_stall_guard_presets(profile: EmbodimentProfile | None = None) -> dict[str, float | None]:
    prof = _profile(profile)
    return {
        name: prof.stall_guard_presets_m[name]
        for name in PUBLISHED_STALL_GUARD_PRESETS
        if name in prof.stall_guard_presets_m
    }


def guard_parameters(profile: EmbodimentProfile | None = None) -> dict[str, Any]:
    prof = _profile(profile)
    return {
        "profile": prof,
        "empty_close_aperture_m": prof.gripper.empty_close_aperture_m,
        "max_aperture_m": prof.gripper.full_open_aperture_m,
    }


def grip_guard_capability(profile: EmbodimentProfile | None = None) -> dict[str, Any]:
    prof = _profile(profile)
    zero = float(prof.gripper.empty_close_aperture_m or 0.0)
    presets = {
        name: dict(
            zip(
                ("min_aperture_m", "max_aperture_m"),
                (None if v is None else round(v, 4) for v in preset_aperture_bounds(name, profile=prof)),
            )
        )
        for name in prof.gripper.preset_margins_m
        if name != "none"
    }
    return {
        "simulation_only": True,
        "empty_close_aperture_m": round(zero, 4),
        "full_open_aperture_m": round(float(prof.gripper.full_open_aperture_m), 4),
        "presets": presets,
        "post_grasp_recheck": {
            "available": True,
            "closing_goal_below": CLOSING_JAW_GOAL_BELOW,
            "min_jaw_drop_normalized": SLIP_MIN_JAW_DROP,
            "max_gap_above_empty_close_m": SLIP_MAX_GAP_ABOVE_EMPTY_CLOSE_M,
            "downward_move_skip_dz_m": DOWNWARD_MOVE_DZ_M,
        },
    }


def compile_action(
    kind: str,
    args: dict[str, Any],
    identity: dict[str, Any],
    capabilities: dict[str, Any],
) -> dict[str, Any]:
    """Compile a single public ``move`` / ``grasp`` / ``gripper`` / ``wait`` request into an ``EEAction``."""
    if kind not in {"move", "grasp", "gripper", "wait"}:
        raise ValueError(f"unsupported_libero_action_kind:{kind}")
    for field in UNSUPPORTED_LEAF_FIELDS:
        if field in args:
            raise ValueError(f"unsupported_capability: {field}")
    if kind != "grasp" and any(field in args for field in CONTACT_FIELDS):
        raise ValueError("contact_selection_requires_grasp_pose")
    if "gripper_motion" in args:
        if "during_motion" not in (capabilities.get("gripper_phases") or ("after_motion", "during_motion")):
            raise ValueError("unsupported_capability: gripper_motion")
        if kind not in {"move", "grasp"} or "gripper_goal_normalized" not in args:
            raise ValueError("invalid_gripper_motion: requires a move and gripper_goal_normalized")

    arms = tuple(capabilities.get("arms") or ARMS)
    defaults = capabilities.get("motion_tolerance_defaults") or {
        "position_tolerance_m": 0.010,
        "orientation_tolerance_rad": 0.15,
    }
    default_timeout = capabilities.get("default_timeout_s", 120.0)
    if kind == "wait":
        default_timeout = (capabilities.get("bounded_wait") or {}).get("default_timeout_s", 30.0)
    timeout = args.get("timeout_s", default_timeout)

    grasp_input: dict[str, Any] | None = None
    waypoints: list[dict[str, Any]] = []
    path: dict[str, Any] = {
        "kind": "hold" if kind in ("gripper", "wait") else "backend_free",
        "orientation": "backend_free",
        "blend_radius_m": 0.0,
        "waypoints": waypoints,
    }

    if kind == "grasp":
        if capabilities.get("tcp_config_id") != TCP_CONFIG_ID:
            raise ValueError("unsupported_grasp_tcp")
        arm = _arm(args, arms)
        via = args.get("via_grasp_poses", [])
        if not isinstance(via, list) or len(via) > 7:
            raise ValueError("invalid_via_grasp_poses: at most seven intermediate poses")
        pose_fields = {"grasp_position_m", "angles_deg"}
        if any(
            not isinstance(point, dict)
            or not pose_fields.issubset(point)
            or (set(point) - pose_fields - set(CONTACT_FIELDS))
            for point in via
        ):
            raise ValueError(
                "invalid_via_grasp_pose: requires grasp_position_m, angles_deg and optional contact selection"
            )
        contact = grasp_contact.resolve_contact_spec(args)
        contact_frames = []
        for point in [*via, args]:
            finite_vector(point.get("grasp_position_m"), 3, "grasp_position_m")
            finite_vector(point.get("angles_deg"), 3, "angles_deg")
            for value, (lower, upper) in zip(point["angles_deg"], pose_convention.ANGLE_BOUNDS.values()):
                if not lower <= value <= upper:
                    raise ValueError("grasp_angle_out_of_bounds")
            values = dict(
                zip(
                    (*pose_convention.POSITION_AXES, *pose_convention.ANGLE_AXES),
                    [*point["grasp_position_m"], *point["angles_deg"]],
                )
            )
            selected_contact = grasp_contact.resolve_contact_spec(point, inherited=contact)
            contact_frames.append(selected_contact)
            pose = grasp_contact.values_to_contact_pose(values, selected_contact).tolist()
            waypoints.append({arm: {"position_m": pose[:3], "quaternion_xyzw": [*pose[4:], pose[3]]}})
        grasp_input = {
            "position_m": [float(v) for v in args["grasp_position_m"]],
            "angles_deg": [float(v) for v in args["angles_deg"]],
            "convention": "libero_panda_top_down_world_xyz_degrees",
            "offset_m": float(contact["offset_m"]),
        }
        if any(field in point for point in [*via, args] for field in CONTACT_FIELDS):
            grasp_input.update(
                contact_region=contact["contact_region"],
                contact_depth_m=contact["contact_depth_m"],
                contact_selection={field: args[field] for field in CONTACT_FIELDS if field in args},
                waypoint_contact_frames=contact_frames,
            )
        if via:
            grasp_input["via_grasp_poses"] = via
    elif kind == "move":
        arm = _arm(args, arms)
        waypoints.append(
            {
                arm: {
                    "position_m": args["position_m"],
                    "quaternion_xyzw": args["quaternion_xyzw"],
                }
            }
        )

    if kind in ("move", "grasp"):
        if "tolerance_preset" in args:
            path["tolerance_preset"] = args["tolerance_preset"]
            if "position_tolerance_m" in args:
                path["position_tolerance_m"] = args["position_tolerance_m"]
            if "orientation_tolerance_rad" in args:
                path["orientation_tolerance_rad"] = args["orientation_tolerance_rad"]
        else:
            path["position_tolerance_m"] = args.get("position_tolerance_m", defaults.get("position_tolerance_m", 0.010))
            path["orientation_tolerance_rad"] = args.get(
                "orientation_tolerance_rad", defaults.get("orientation_tolerance_rad", 0.15)
            )
        for extra_key in (
            "max_upward_dz_m",
            "max_downward_dz_m",
            "stall_guard_preset",
            "max_tracking_divergence_m",
        ):
            if extra_key in args:
                path[extra_key] = args[extra_key]

    action: dict[str, Any] = {
        "schema_version": "0.1",
        **identity,
        "world_kind": "observed",
        "action_id": args.get("action_id", new_id("action")),
        "based_on_observation_id": args["observation_id"],
        "frame_id": capabilities.get("frame_id", FRAME_ID),
        "tcp_config_id": capabilities.get("tcp_config_id", TCP_CONFIG_ID),
        "mode": "absolute",
        "path": path,
        "gripper_events": [],
        "limits_profile": capabilities.get("limits_profile", LIMITS_PROFILE),
        "timeout_s": timeout,
    }
    for key in SPEED_INPUT_FIELDS:
        if key in args:
            action[key] = args[key]
    if grasp_input is not None:
        action["input_grasp_pose"] = grasp_input
    if kind == "gripper":
        action["gripper_events"] = [
            {"arm": _arm(args, arms), "phase": "after_motion", "goal_normalized": args["goal_normalized"]}
        ]
    elif kind == "wait":
        action["hold_native_steps"] = args["native_steps"]
    if "gripper_goal_normalized" in args:
        action["gripper_events"] = [
            {"arm": _arm(args, arms), "phase": "after_motion", "goal_normalized": args["gripper_goal_normalized"]}
        ]
    if "gripper_motion" in args:
        timing = validate_gripper_motion(args["gripper_motion"])
        action["gripper_events"][0].update(
            phase="during_motion",
            start_fraction=timing["start_fraction"],
            end_fraction=timing["end_fraction"],
        )
    return action


def _validate_path_fields(path: dict[str, Any], profile: EmbodimentProfile) -> None:
    honoured = set(PATH_FIELDS)
    for field in sorted(set(path) - honoured):
        raise ValueError(f"unsupported_capability: {field}")
    if path.get("max_upward_dz_m") is not None:
        _number(path["max_upward_dz_m"], 0.0, MAX_UPWARD_DZ_M, "max_upward_dz_m", low_open=False)
    if path.get("max_downward_dz_m") is not None:
        _number(path["max_downward_dz_m"], 0.0, MAX_DOWNWARD_DZ_M, "max_downward_dz_m", low_open=False)
    preset = path.get("tolerance_preset")
    if preset is not None and preset not in profile.tolerance_presets:
        raise ValueError(f"unsupported_capability: tolerance_preset {preset}")
    stall = path.get("stall_guard_preset")
    divergence = path.get("max_tracking_divergence_m")
    offered = offered_stall_guard_presets(profile)
    if stall is not None and stall not in offered:
        raise ValueError(f"unsupported_capability: stall_guard_preset {stall}")
    if divergence is not None:
        _number(divergence, *profile.tracking_divergence_range_m, "max_tracking_divergence_m", low_open=False)


def validate_grip_guard(mapping: dict[str, Any], profile: EmbodimentProfile | None = None) -> dict[str, Any] | None:
    prof = _profile(profile)
    if not any(field in mapping for field in GRIP_GUARD_FIELDS):
        return None
    return validate_grip_guard_fields(mapping, **guard_parameters(prof))


def validate_leaf(action: dict[str, Any], profile: EmbodimentProfile | None = None) -> str | None:
    prof = _profile(profile)
    arms = prof.arms
    if action.get("type", "EEAction") != "EEAction":
        raise ValueError("unsupported_capability: " + str(action.get("type")))
    if action.get("mode") != "absolute" or action.get("frame_id") != prof.frame_id:
        raise ValueError(f"unsupported_capability: absolute pose in {prof.frame_id} required")
    if action.get("tcp_config_id") != prof.tcp_config_id:
        raise ValueError(f"unsupported_capability: TCP must be {prof.tcp_config_id}")
    if action.get("limits_profile") not in prof.limits_profiles:
        raise ValueError(f"unsupported_capability: limits_profile must be one of {prof.limits_profiles}")
    for field in UNSUPPORTED_LEAF_FIELDS:
        if field in action:
            raise ValueError(f"unsupported_capability: {field}")
    for field, scale in (("speed_scale", prof.speed_scale), ("gripper_speed_scale", prof.gripper_speed_scale)):
        if field in action:
            _number(action[field], scale.min, scale.max, field, low_open=False)
    _number(action.get("timeout_s"), 0.0, prof.max_timeout_s, "timeout_s")
    path = action.get("path")
    if not isinstance(path, dict):
        raise ValueError("invalid_action: path must be an object")
    points = path.get("waypoints", [])
    events = action.get("gripper_events", [])
    if not isinstance(points, list) or not isinstance(events, list):
        raise ValueError("invalid_action: waypoints and gripper_events must be lists")
    if path.get("blend_radius_m", 0) != 0 or path.get("orientation", "backend_free") != "backend_free":
        raise ValueError("unsupported_capability: blending or orientation path constraints")
    _validate_path_fields(path, prof)
    for event in events:
        phase = event.get("phase") if isinstance(event, dict) else None
        if phase == "during_motion":
            if set(event) != DURING_MOTION_EVENT_FIELDS:
                raise ValueError(
                    "invalid_gripper_motion: a during_motion event has arm, phase, goal_normalized, start_fraction, end_fraction"
                )
            validate_gripper_motion({k: event[k] for k in ("start_fraction", "end_fraction")})
            if path.get("kind") != "backend_free" or not points:
                raise ValueError("invalid_gripper_motion: requires a moving arm")
        elif phase != "after_motion":
            raise ValueError(f"unsupported_capability:gripper_events:{phase}")
        if event.get("arm") not in arms:
            raise ValueError(f"unsupported_arm:{event.get('arm')}")
        _number(event.get("goal_normalized"), 0.0, 1.0, "goal_normalized", low_open=False)
    if len(events) > 1:
        raise ValueError("unsupported_capability: at most one gripper event")
    if path.get("kind") == "hold":
        if points:
            raise ValueError("unsupported_capability: hold requires empty waypoints")
        if "hold_native_steps" in action:
            steps = action["hold_native_steps"]
            if type(steps) is not int or not 1 <= steps <= MAX_WAIT_STEPS or events:
                raise ValueError(
                    f"invalid_action: hold_native_steps must be an integer in [1, {MAX_WAIT_STEPS}] without gripper events"
                )
            return None
        if len(events) != 1:
            raise ValueError("invalid_action: hold requires a standalone gripper event")
        return str(events[0]["arm"])
    if path.get("kind") != "backend_free" or not 1 <= len(points) <= MAX_WAYPOINTS:
        raise ValueError(f"unsupported_capability: 1..{MAX_WAYPOINTS} backend_free waypoints required")
    if "hold_native_steps" in action:
        raise ValueError("invalid_action: hold_native_steps requires a hold path")
    for point in points:
        if not isinstance(point, dict) or len(point) != 1 or not set(point) <= set(arms):
            raise ValueError(
                "unsupported_arm:" + ",".join(sorted(set(point) - set(arms)))
                if isinstance(point, dict)
                else "invalid_pose: waypoint must be an object"
            )
        pose = next(iter(point.values()))
        if not isinstance(pose, dict):
            raise ValueError("invalid_pose: pose must be an object")
        for field, size in (("position_m", 3), ("quaternion_xyzw", 4)):
            value = pose.get(field)
            if (
                not isinstance(value, list)
                or len(value) != size
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)
            ):
                raise ValueError("invalid_pose: finite position_m[3] and quaternion_xyzw[4] required")
        if abs(math.sqrt(sum(x * x for x in pose["quaternion_xyzw"])) - 1.0) > 1e-3:
            raise ValueError("invalid_pose: quaternion must be normalized")
        check_workspace(pose["position_m"])
    pos_tol = path.get("position_tolerance_m", prof.default_tolerance.position_m)
    ori_tol = path.get("orientation_tolerance_rad", prof.default_tolerance.orientation_rad)
    _number(pos_tol, 0.0, prof.max_position_tolerance_m, "position_tolerance_m")
    _number(ori_tol, 0.0, prof.max_orientation_tolerance_rad, "orientation_tolerance_rad")
    return str(next(iter(points[0])))


def validate_action(action: dict[str, Any], profile: EmbodimentProfile | None = None) -> tuple[str | None, None]:
    """Backend numeric and capability gate for ``EEAction`` and ``BatchAction``."""
    prof = _profile(profile)
    arms = prof.arms
    kind = action.get("type", "EEAction")
    if kind == "BatchAction":
        stages = action.get("stages")
        if not isinstance(stages, list) or not 1 <= len(stages) <= prof.max_stages:
            raise ValueError("batch_requires_one_to_thirty_two_stages")
        if action.get("frame_id") != prof.frame_id or action.get("tcp_config_id") != prof.tcp_config_id:
            raise ValueError("unsupported_batch_frame_or_tcp")
        if action.get("limits_profile") not in prof.limits_profiles:
            raise ValueError("unsupported_batch_limits")
        _number(action.get("timeout_s"), 0.0, prof.max_timeout_s, "timeout_s")
        for field in UNSUPPORTED_BATCH_FIELDS:
            if field in action or any(field in stage for stage in stages if isinstance(stage, dict)):
                raise ValueError(f"unsupported_capability: {field}")
        if action.get("max_upward_dz_m") is not None:
            _number(action["max_upward_dz_m"], 0.0, MAX_UPWARD_DZ_M, "max_upward_dz_m", low_open=False)
        if action.get("max_downward_dz_m") is not None:
            _number(action["max_downward_dz_m"], 0.0, MAX_DOWNWARD_DZ_M, "max_downward_dz_m", low_open=False)
        if "checkpoint_after_gripper" in action and not isinstance(action["checkpoint_after_gripper"], bool):
            raise ValueError("invalid_checkpoint_after_gripper")
        validate_grip_guard(action, prof)
        validate_batch_carry_slip_guard(action, **guard_parameters(prof))
        for stage in stages:
            if not isinstance(stage, dict):
                raise ValueError("invalid_batch_stage")
            if stage.get("kind") == "observe":
                if stage.get("arm") not in (None, *arms):
                    raise ValueError(f"unsupported_arm:{stage.get('arm')}")
                if any(
                    field in stage
                    for field in (*GRIP_GUARD_FIELDS, *CARRY_SLIP_GUARD_FIELDS, "checkpoint_after_gripper")
                ):
                    raise ValueError("batch_observe_requires_reason")
                continue
            if stage.get("arm") not in arms:
                raise ValueError(f"unsupported_arm:{stage.get('arm')}")
            leaf = stage.get("action")
            if not isinstance(leaf, dict):
                raise ValueError("batch_command_requires_arm_action")
            validate_leaf(leaf, prof)
            if stage.get("kind") == "move" and (
                leaf["path"]["kind"] != "backend_free" or len(leaf["path"]["waypoints"]) != 1
            ):
                raise ValueError("batch_move_requires_one_owned_target")
            if stage.get("kind") == "move" and any(
                event.get("phase") != "during_motion" for event in leaf.get("gripper_events", [])
            ):
                raise ValueError("batch_after_motion_jaw_requires_separate_stage")
            if stage.get("kind") == "gripper" and (
                leaf["path"]["kind"] != "hold" or len(leaf.get("gripper_events", [])) != 1
            ):
                raise ValueError("batch_gripper_requires_one_jaw_event")
            jaw_fields = [
                field
                for field in (*GRIP_GUARD_FIELDS, "checkpoint_after_gripper")
                if field in stage
            ]
            if jaw_fields and stage.get("kind") != "gripper" and not leaf.get("gripper_events"):
                raise ValueError(
                    f"unsupported_capability: {jaw_fields[0]} applies to jaw stages; this move stage has no jaw command"
                )
            if "checkpoint_after_gripper" in stage and not isinstance(stage["checkpoint_after_gripper"], bool):
                raise ValueError("invalid_checkpoint_after_gripper")
            validate_grip_guard(stage, prof)
            validate_grip_squeeze(stage, profile=prof)
        return arms[0], None
    if kind != "EEAction":
        raise ValueError(f"unsupported_capability: {kind}")
    return validate_leaf(action, prof), None


def actor_motion_instructions(
    profile: EmbodimentProfile | None = None,
    *,
    table_z_m: float | None = None,
) -> str:
    """Simulation-only motion and coordinate instructions for ``libero-franka-panda``."""
    prof = _profile(profile)
    offered = offered_stall_guard_presets(prof)
    presets_txt = "; ".join(f"{name} {val:g} m" for name, val in offered.items() if val is not None)
    low_div, high_div = prof.tracking_divergence_range_m
    table_clause = (
        f"In this episode's scene, the measured primary support/table surface is at z={table_z_m:.3f} m in {prof.frame_id}."
        if table_z_m is not None
        else (
            f"Scene support surface height table_z_m varies across LIBERO arenas (-0.012 m on standard tabletop arenas, "
            f"-0.010 m on living-room tables, 0.000 m on floor arenas in {prof.frame_id}); read "
            "camera_calibration.joint_calibration.table_z_m or object_table bottom_z rather than assuming z=0."
        )
    )
    return f"""## Motion planning
SIMULATION-ONLY EMBODIMENT: You control ONE simulated 7-DoF Franka Emika Panda arm + parallel-jaw PandaGripper inside the LIBERO benchmark (robosuite MuJoCo OperationalSpaceController at {prof.control_hz:g} Hz). This model is strictly simulation-only and NOT calibrated for physical Franka hardware.
Its only arm id is "{prof.arms[0]}"; any other arm id is rejected. Simulation time advances ONLY during robot_action / robot_wait execution and pauses completely while you reason.
robot_action plans and executes 1..{prof.max_stages} sequential stages on "{prof.arms[0]}":
- Preview (default) checks 7-DoF kinematic reachability for every move stage via URDF IK from the current measured joints (chaining planned stage joints without advancing physics), renders the 3D candidate trajectory overlay on "{prof.overview_camera}", and returns a plan_id.
- Execute runs each move stage closed-loop at {prof.control_hz:g} Hz directly through robosuite's native OSC_POSE controller (max per-step delta 0.05 m / 0.5 rad scaled by speed_scale in [{prof.speed_scale.min:g}, {prof.speed_scale.max:g}]) and verifies measured arrival on {prof.tcp_label}.
- mode='reach' checks up to {prof.max_stages} candidate move poses independently from the current measured joints without creating a plan or moving.
- After a stage pauses (observe stage, checkpoint_after_gripper, grip_guard, carry_slip_guard) or fails, continue from current measured joints via source_plan_id + start_from_stage_id (with optional stage_overrides) or resume an awaiting_observation checkpoint with mode='execute' and the same plan_id.
- Stall guard (move stages): stall_guard_preset (default off; {presets_txt}) or max_tracking_divergence_m in [{low_div:g}, {high_div:g}] m {prof.stall_guard_semantics}. Sustained physical stall is required over multiple {prof.control_hz:g} Hz steps so initial inertia never false-triggers.
- Vertical arrival windows: max_upward_dz_m (0..{MAX_UPWARD_DZ_M:g} m) accepts arrivals resting above target when horizontal (dx, dy) and orientation are within tolerance (combine with stall_guard_preset='contact_stop' for gentle support/seating descents); max_downward_dz_m (0..{MAX_DOWNWARD_DZ_M:g} m) accepts deeper seated arrivals below target.
- Jaw during a move: gripper_goal_normalized with gripper_motion {{start_fraction, end_fraction}} ramps the gripper target continuously during the move; without gripper_motion, the jaw actuates after the move arrives.

## Coordinates and timing
All coordinates are metres in {prof.frame_id} (Panda link0 base frame: +X forward across the workspace, +Y left, +Z up).
Move stages accept EITHER:
1. Direct TCP pose: position_m + quaternion_xyzw [qx, qy, qz, qw] of {prof.tcp_label} (the point directly between the left and right finger pad centres, 0.0 m offset; local +Z points outward along the finger approach direction, local +-X is the parallel-jaw opening line, local +Y points toward the "{prof.wrist_cameras[prof.arms[0]]}" wrist camera).
2. Grasp pose: grasp_position_m + angles_deg = [pitch_deg, roll_deg, yaw_deg] with optional contact_region ('pad_center' = 0.0 m at {prof.tcp_label}, 'finger_middle' = +0.0176 m deeper toward palm, 'fingertip' = -0.0093 m distal tip). At angles_deg = [0, 0, 0] (quaternion_xyzw == [0, 1, 0, 0]), approach (+Z) points straight down [0, 0, -1] and the jaws (+X) open along world +-X; at angles_deg = [0, 0, -90], quaternion_xyzw == [0.707107, 0.707107, 0, 0] (LIBERO's nominal spawn orientation of {prof.tcp_label}, opening the jaws along world +-Y).
Default arrival tolerances are {prof.default_tolerance.position_m * 1000:.0f} mm and {prof.default_tolerance.orientation_rad:g} rad.
{table_clause}
Gripper control is continuous closed-loop aperture servo: goal_normalized in [0.0, 1.0] maps linearly to target_aperture_m = goal_normalized * {prof.gripper.full_open_aperture_m:.3f} m across [0.000, {prof.gripper.full_open_aperture_m:.3f}] m (no binary snap; partial openings such as 0.35 hold ~0.028 m). On an empty goal=0.0 close, the 4 mm finger pad collision boxes meet at measured_aperture_m ≈ {prof.gripper.empty_close_aperture_m:.4f} m (1.0 mm). Grip-guard presets ('non_empty', 'thin_object', 'rigid_object', 'wide_object') resolve relative to this {prof.gripper.empty_close_aperture_m:.4f} m empty-close baseline; use grip_guard_preset='non_empty' (and optional carry_slip_guard=true) on grasp closes so an empty grasp pauses before lifting air.
"""
