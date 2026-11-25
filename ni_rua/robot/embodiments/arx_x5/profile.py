"""ARX-X5 dual-arm 6-DoF embodiment profile, asset binding, and kinematic helpers."""
from __future__ import annotations

from dataclasses import dataclass
import functools
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

from ....core.embodiment import (
    EmbodimentProfile,
    FrozenDict,
    GRIPPER_SPEED_PRESETS,
    GripperProfile,
    HomeCheckProfile,
    LinkSegment,
    MAX_BATCH_STAGES,
    MAX_STAGES_PER_ARM,
    RobotDescription,
    SPEED_PRESETS,
    SUPPORT_TAGS,
    ScaleRange,
    grip_guard_preset_margins,
    presets_within,
    register_default_embodiment,
    stall_guard_presets,
    tolerance_presets,
)

if TYPE_CHECKING:
    from ...model.kinematics import KinematicChainSpec

_DATA_ROOT = Path(__file__).resolve().parents[2] / "model" / "robot_information_data"


@dataclass(frozen=True)
class RoboDojoRobotBinding:
    """Paths are relative to RoboDojo root, except layout_directory (Assets root)."""

    robot: RobotDescription
    environment_config_path: Path
    urdf_path: Path
    robot_config_path: Path
    curobo_config_path: Path
    layout_directory: Path


ARX_X5 = RobotDescription(
    robot_model_id="arx-x5",
    frame_id="env_origin",
    tcp_config_id="arx-x5-link6-v1",
)

_ROBODOJO_ROBOTS = {
    ARX_X5.robot_model_id: RoboDojoRobotBinding(
        robot=ARX_X5,
        environment_config_path=Path("env_cfg/arx_x5.yml"),
        urdf_path=Path("Assets/Robots/x5/X5A.urdf"),
        robot_config_path=Path("Assets/Robots/x5/robot_config.yml"),
        curobo_config_path=Path("Assets/Robots/x5/curobo.yml"),
        layout_directory=Path("Eval_Layout/RoboDojo/arx_x5"),
    ),
}


def get_robodojo_robot(robot_model_id: str = ARX_X5.robot_model_id) -> RoboDojoRobotBinding:
    """Select an explicitly supported binding; never substitute another robot."""
    try:
        return _ROBODOJO_ROBOTS[robot_model_id]
    except (KeyError, TypeError):
        raise ValueError(f"unsupported_robodojo_robot: {robot_model_id!r}") from None


@functools.lru_cache(maxsize=None)
def physical_facts(directory: str = "arx_x5") -> dict:
    """Packaged ``robot_information_data/<directory>/physical.json`` (read-only; do not mutate)."""
    return json.loads((_DATA_ROOT / directory / "physical.json").read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def arx_x5_defaults() -> dict:
    """Packaged ``arx_x5/defaults.json`` (thresholds and preset upward +Z compliance windows; read-only)."""
    return json.loads((Path(__file__).resolve().parent / "defaults.json").read_text(encoding="utf-8"))


ROBODOJO_REQUEST_DURATION_LIMIT_S = 10.0
X5_DEFAULT_ACTION_TIMEOUT_S = 120.0
X5_DEFAULT_WAIT_TIMEOUT_S = 30.0
X5_MAX_TIMEOUT_S = 300.0
X5_GRIPPER_SPEED_SCALE = (0.25, 1.0)
X5_OVERVIEW_CAMERA = "cam_head"
X5_WRIST_CAMERAS = (("left", "cam_left_wrist"), ("right", "cam_right_wrist"))
X5_CAMERA_EXAMPLES = ("cam_head", "cam_right_wrist")
X5_MASK_REACH_M = 0.19
X5_LIMITS_PROFILES = ("robodojo-native-v1", "simulation-slow-v1")
X5_STALL_GUARD_SEMANTICS = (
    "stops at first support/object contact when horizontal (dx, dy) is within tolerance and "
    "holds the contact pose so attached gripper stages execute"
)
X5_VERTICAL_WINDOW_SEMANTICS = (
    'Move only: optional upward +Z compliance window in metres [0, 0.15] when horizontal (dx, dy) and orientation '
    'are within tolerance and measured Z rests above target (+dz_m <= max_upward_dz_m). Defaults when omitted: '
    '0.040 m when tolerance_preset="compliant"/"seating", max(position_tolerance_m, 0.040 m) when '
    'stall_guard_preset="contact_stop", and 0 m on all other moves. Set an explicit positive value (e.g. 0.04-0.12 m) '
    'for deep support/seating stops, or 0 for strict 3D Z arrival. Resolved effective_tolerances are reported in '
    'preview and stage_results.'
)
X5_SUPPORTS = frozenset(SUPPORT_TAGS) - {
    "reach",
    "approach_roll",
    "cross_view",
    "path_check",
    "collision_world",
    "arrival_correction",
    "robot_occlusion",
    "contact_top",
    "cross_view_attach",
}
X5_FEEDBACK_UNSUPPORTED = frozenset({"dual_arm", "gripper_motion"})
X5_CAPSULE_RADII_M = (0.035, 0.032, 0.034, 0.029, 0.036, 0.020)


def arx_x5_profile(*, binding=None, thresholds=None, motion_execution_mode: str = "resampled") -> EmbodimentProfile:
    """Build the ARX-X5 ``EmbodimentProfile`` from packaged facts, optional RoboDojo binding, and run thresholds."""
    from ....core.config import motion_thresholds
    from . import grasp_contact, pose_convention as roboprobe_pose

    if motion_execution_mode not in ("resampled", "feedback"):
        raise ValueError("invalid_motion_execution_mode")
    if binding is not None and (not isinstance(binding, RoboDojoRobotBinding) or binding.robot != ARX_X5):
        raise ValueError("unsupported_embodiment_binding")
    t = motion_thresholds(thresholds)
    facts = physical_facts("arx_x5")
    kinematics, harness = facts["kinematics"], facts["control"]["default_harness"]
    joints = {joint["name"]: joint for joint in kinematics["joints"]}
    arms = tuple(mount["arm"] for mount in kinematics["default_mounts"])
    jaw = tuple(kinematics["gripper_joint_order"])
    feedback = motion_execution_mode == "feedback"
    control_hz = float(harness["native_control_hz"])
    low, high = harness["speed_scale_range"]
    speed = ScaleRange(float(low), float(high), 1.0)
    gripper_speed = ScaleRange(*X5_GRIPPER_SPEED_SCALE, 1.0)
    presets = tolerance_presets(t, upward_dz_m=arx_x5_defaults()["preset_upward_dz_m"])
    gripper = GripperProfile(
        empty_close_aperture_m=float(sum(joints[name]["declared_limits"]["lower"] for name in jaw)),
        full_open_aperture_m=t["max_gripper_aperture_m"],
        full_open_source="run threshold max_gripper_aperture_m (ARX-X5 finger travel joint7 + joint8)",
        preset_margins_m=grip_guard_preset_margins(t),
        aperture_reading=(
            "measured normalized jaw opening (finger joint position over its 0..0.044 m travel, "
            "clipped to [0, 1]) times full_open_aperture_m; 0.0 m at full close. The distal pad "
            "surfaces are 0.8 mm further apart than this reading"
        ),
        fingertip_beyond_reference_m=-grasp_contact.CONTACT_DEPTH_PRESETS_M["fingertip"],
        grasp_reference="tcp_offset",
        grasp_reference_offset_m=roboprobe_pose.GRASP_POINT_OFFSET_M,
        approach_axis_local=tuple(float(v) for v in kinematics["pose_interface"]["tool_axis_link6"]),
        jaw_opening_axis_local=tuple(float(v) for v in joints[jaw[0]]["axis"]),
        mask_segment=LinkSegment(
            kinematics["end_effector_link"],
            tuple(float(v) for v in kinematics["pose_interface"]["tool_axis_link6"]),
            X5_MASK_REACH_M,
        ),
        contact_sections=FrozenDict(grasp_contact.CONTACT_DEPTH_PRESETS_M),
    )
    return EmbodimentProfile(
        robot=binding.robot if binding is not None else ARX_X5,
        tcp_link=kinematics["end_effector_link"],
        tcp_label="link6/flange",
        limits_profiles=X5_LIMITS_PROFILES,
        arms=arms,
        max_simultaneous_arms=1 if feedback else len(arms),
        arm_joint_names=FrozenDict((arm, tuple(kinematics["arm_joint_order"])) for arm in arms),
        jaw_joint_names=FrozenDict((arm, jaw) for arm in arms),
        stages_per_arm=MAX_STAGES_PER_ARM,
        max_stages=MAX_BATCH_STAGES,
        gripper=gripper,
        pose_inputs=frozenset({"tcp_quaternion", "grasp_angles"}),
        supports=X5_SUPPORTS - X5_FEEDBACK_UNSUPPORTED if feedback else X5_SUPPORTS,
        stall_guard_semantics=X5_STALL_GUARD_SEMANTICS,
        vertical_window_semantics=X5_VERTICAL_WINDOW_SEMANTICS,
        speed_scale=speed,
        gripper_speed_scale=gripper_speed,
        speed_presets=presets_within(SPEED_PRESETS, speed),
        gripper_speed_presets=presets_within(GRIPPER_SPEED_PRESETS, gripper_speed),
        tolerance_presets=presets,
        default_tolerance_preset="normal",
        default_tolerance=presets["normal"],
        max_position_tolerance_m=t["max_position_tolerance_m"],
        max_orientation_tolerance_rad=t["max_orientation_tolerance_rad"],
        stall_guard_presets_m=stall_guard_presets(t),
        tracking_divergence_range_m=(t["stall_guard_min_divergence_m"], t["stall_guard_max_divergence_m"]),
        clock="simulated_steps",
        control_hz=control_hz,
        step_domain="adapter_control_step",
        step_label="native step",
        per_request_step_limit=(
            math.ceil(ROBODOJO_REQUEST_DURATION_LIMIT_S * control_hz) if binding is not None else None
        ),
        default_action_timeout_s=X5_DEFAULT_ACTION_TIMEOUT_S,
        default_wait_timeout_s=X5_DEFAULT_WAIT_TIMEOUT_S,
        max_timeout_s=X5_MAX_TIMEOUT_S,
        cameras=(X5_OVERVIEW_CAMERA, *(camera for _, camera in X5_WRIST_CAMERAS)),
        overview_camera=X5_OVERVIEW_CAMERA,
        wrist_cameras=FrozenDict(X5_WRIST_CAMERAS),
        camera_examples=X5_CAMERA_EXAMPLES,
        support_surface_z_m=roboprobe_pose.TABLE_SURFACE_Z,
        handoff=True,
        home_check=HomeCheckProfile("measured-bimanual-link6-home-v1", ARX_X5.frame_id, ARX_X5.tcp_config_id, arms),
        policy_step=True,
    )


@functools.lru_cache(maxsize=1)
def arx_x5_chain() -> KinematicChainSpec:
    """Return the canonical 6-DoF ``KinematicChainSpec`` of one ARX-X5 arm from ``arx_x5/physical.json``."""
    from ...model.kinematics import chain_from_manifest

    kin = physical_facts("arx_x5")["kinematics"]
    return chain_from_manifest(
        {"base_link": "base_link", "joints": kin["joints"]},
        joint_names=kin["arm_joint_order"],
        tip_link=kin["end_effector_link"],
        base_link="base_link",
    )


@functools.lru_cache(maxsize=1)
def arx_x5_default_mounts() -> dict[str, "object"]:
    """Return ``{arm: 4x4 T_world_base}`` nominal dual-arm mounts from ``arx_x5/physical.json`` (``m``)."""
    import numpy as np
    from ...model.kinematics import quat_wxyz_to_matrix

    mounts = {}
    for item in physical_facts("arx_x5")["kinematics"]["default_mounts"]:
        tf = np.eye(4, dtype=np.float64)
        tf[:3, :3] = quat_wxyz_to_matrix(item["quaternion_wxyz"])
        tf[:3, 3] = [float(v) for v in item["position_environment_m"]]
        mounts[str(item["arm"])] = tf
    return mounts


@functools.lru_cache(maxsize=None)
def _default_profile() -> EmbodimentProfile:
    return arx_x5_profile()


register_default_embodiment(_default_profile)
