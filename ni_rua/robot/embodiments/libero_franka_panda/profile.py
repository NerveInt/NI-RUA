"""Simulated LIBERO Franka Emika Panda 7-DoF embodiment profile and URDF kinematics (simulation-only).

Models ONLY the simulated Franka Emika Panda arm + PandaGripper inside the LIBERO benchmark
(robosuite MuJoCo ``SingleArm`` + ``OperationalSpaceController``). NOT calibrated for physical
Franka Emika hardware. All FK/IK helpers delegate directly to ``ni_rua.robot.model.kinematics``
with zero duplicate solver math.
"""
from __future__ import annotations

import copy
import functools
import json
import math
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING
import zipfile

from ....core.embodiment import (
    EmbodimentProfile,
    FrozenDict,
    GRIPPER_SPEED_PRESETS,
    GripperProfile,
    LinkSegment,
    MAX_BATCH_STAGES,
    RobotDescription,
    SPEED_PRESETS,
    ScaleRange,
    grip_guard_preset_margins,
    presets_within,
    stall_guard_presets,
    tolerance_presets,
)
from ...model.robot_model import urdf_manifest

if TYPE_CHECKING:
    from ...model.kinematics import KinematicChainSpec

LIBERO_FRANKA_PANDA = RobotDescription(
    robot_model_id="libero-franka-panda",
    frame_id="franka_base",
    tcp_config_id="libero-panda-grip-site-v1",
)

ARM_ID = "arm0"
ARM_JOINTS: tuple[str, ...] = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
)
GRIPPER_JOINTS: tuple[str, ...] = ("panda_finger_joint1", "panda_finger_joint2")
BASE_LINK = "panda_link0"
TCP_LINK = "gripper0_grip_site"
WRIST_CAMERA_PARENT_LINK = "panda_hand"

PANDA_OVERVIEW_CAMERA = "agentview"
PANDA_WRIST_CAMERAS: tuple[tuple[str, str], ...] = (("arm0", "wrist"),)
PANDA_CAMERA_EXAMPLES: tuple[str, ...] = ("agentview", "wrist")
PANDA_MASK_LINK = "panda_hand"
PANDA_MASK_REACH_M = 0.14
PANDA_LIMITS_PROFILES: tuple[str, ...] = ("libero-osc-20hz-v1", "simulation-slow-v1")
PANDA_DEFAULT_ACTION_TIMEOUT_S = 120.0
PANDA_DEFAULT_WAIT_TIMEOUT_S = 30.0
PANDA_MAX_TIMEOUT_S = 300.0
PANDA_GRIPPER_SPEED_SCALE = (0.25, 1.0)
PANDA_EMPTY_CLOSE_APERTURE_M = 0.0010
PANDA_FULL_OPEN_APERTURE_M = 0.080

PANDA_STALL_GUARD_SEMANTICS = (
    "stops at first support/object contact when horizontal (dx, dy) is within tolerance and "
    "holds the contact pose so attached gripper stages execute"
)
PANDA_VERTICAL_WINDOW_SEMANTICS = (
    'Move only: optional upward +Z compliance window in metres [0, 0.15] when horizontal (dx, dy) and orientation '
    'are within tolerance and measured Z rests above target (+dz_m <= max_upward_dz_m). Defaults when omitted: '
    '0.040 m when tolerance_preset="compliant"/"seating", max(position_tolerance_m, 0.040 m) when '
    'stall_guard_preset="contact_stop", and 0 m on all other moves. Set an explicit positive value (e.g. 0.04-0.12 m) '
    'for deep support/seating stops, or 0 for strict 3D Z arrival. Resolved effective_tolerances are reported in '
    'preview and stage_results.'
)
PANDA_SUPPORTS = frozenset({
    "gripper_motion",
    "max_upward_dz",
    "stall_guard",
    "settle",
    "execute_if_clean",
    "held_perception",
    "reach",
})

_PACKAGE = Path(__file__).resolve().parents[2] / "model" / "robot_information_data" / "libero_franka_panda"


@functools.lru_cache(maxsize=1)
def panda_facts() -> dict:
    """Packaged ``robot_information_data/libero_franka_panda/physical.json`` (simulation-only)."""
    return json.loads((_PACKAGE / "physical.json").read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def panda_defaults() -> dict:
    """Packaged ``libero_franka_panda/defaults.json``."""
    return json.loads((Path(__file__).resolve().parent / "defaults.json").read_text(encoding="utf-8"))


def _validated(model: dict) -> dict:
    names = {joint["name"]: joint for joint in model["joints"]}
    for name in ARM_JOINTS:
        if names.get(name, {}).get("type") != "revolute" or "limits" not in names[name]:
            raise ValueError(f"libero_franka_panda_missing_revolute_joint:{name}")
    for name in GRIPPER_JOINTS:
        if names.get(name, {}).get("type") != "prismatic" or "limits" not in names[name]:
            raise ValueError(f"libero_franka_panda_missing_prismatic_joint:{name}")
    if model["base_link"] != BASE_LINK or TCP_LINK not in {joint["child"] for joint in model["joints"]}:
        raise ValueError("libero_franka_panda_missing_base_or_tcp")
    return model


@functools.lru_cache(maxsize=1)
def _packaged_model() -> dict:
    native = panda_facts()["geometry"]["native_model"]
    with zipfile.ZipFile(_PACKAGE / f"{native['resource_id']}.zip") as archive:
        data = archive.read(native["assembly_member"])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / Path(native["assembly_member"]).name
        path.write_bytes(data)
        return _validated(urdf_manifest(path))


def load_model(urdf_path=None) -> dict:
    """Return ``urdf_manifest()`` of ``panda.urdf`` (independent deep copy per call)."""
    if urdf_path is not None:
        return _validated(urdf_manifest(urdf_path))
    return copy.deepcopy(_packaged_model())


def joint_limits_rad(model=None) -> dict[str, tuple[float, float]]:
    """URDF-declared ``(lower_rad, upper_rad)`` bounds in ``rad`` for the 7 Panda arm joints."""
    manifest = _packaged_model() if model is None else model
    names = {joint["name"]: joint for joint in manifest["joints"]}
    return {
        name: (float(names[name]["limits"]["lower"]), float(names[name]["limits"]["upper"]))
        for name in ARM_JOINTS
    }


def libero_franka_panda_profile(*, thresholds=None, per_request_step_limit: int | None = None) -> EmbodimentProfile:
    """Build the simulation-only ``libero-franka-panda`` ``EmbodimentProfile``."""
    from . import grasp_contact, pose_convention

    defaults = panda_defaults()
    t = dict(defaults["thresholds"])
    if thresholds is not None:
        from ....core.config import ThresholdMap

        if isinstance(thresholds, ThresholdMap):
            t.update(thresholds._overrides)
        elif isinstance(thresholds, dict):
            t.update(thresholds)
        else:
            overrides = getattr(getattr(thresholds, "_embodiment_thresholds", None), "_overrides", None)
            if isinstance(overrides, dict):
                t.update(overrides)
    facts = panda_facts()
    kinematics = facts["kinematics"]
    harness = facts["control"]["default_harness"]
    control_hz = float(harness["native_control_hz"])
    low, high = harness["speed_scale_range"]
    speed = ScaleRange(float(low), float(high), 1.0)
    gripper_speed = ScaleRange(*PANDA_GRIPPER_SPEED_SCALE, 1.0)
    presets = tolerance_presets(t, upward_dz_m=defaults["preset_upward_dz_m"])
    gripper = GripperProfile(
        empty_close_aperture_m=PANDA_EMPTY_CLOSE_APERTURE_M,
        full_open_aperture_m=float(t.get("max_gripper_aperture_m", PANDA_FULL_OPEN_APERTURE_M)),
        full_open_source="simulated LIBERO PandaGripper finger travel (0..0.04 m + 0..-0.04 m = 0.080 m)",
        preset_margins_m=grip_guard_preset_margins(t),
        aperture_reading=(
            "measured finger separation qpos[0] + abs(qpos[1]) in metres across [0.0010, 0.0800] m "
            "(empty full close settles at 0.0010 m due to 4 mm half-thickness pad collision boxes; "
            "inner pad gap is max(0.0, measured_aperture_m - 0.0010))"
        ),
        fingertip_beyond_reference_m=-grasp_contact.CONTACT_DEPTH_PRESETS_M["fingertip"],
        grasp_reference="tcp_offset",
        grasp_reference_offset_m=pose_convention.GRASP_POINT_OFFSET_M,
        approach_axis_local=(0.0, 0.0, 1.0),
        jaw_opening_axis_local=(1.0, 0.0, 0.0),
        mask_segment=LinkSegment(PANDA_MASK_LINK, (0.0, 0.0, 1.0), PANDA_MASK_REACH_M),
        contact_sections=FrozenDict(grasp_contact.CONTACT_DEPTH_PRESETS_M),
    )
    return EmbodimentProfile(
        robot=LIBERO_FRANKA_PANDA,
        tcp_link=kinematics["tcp_link"],
        tcp_label="gripper0_grip_site",
        limits_profiles=PANDA_LIMITS_PROFILES,
        arms=(ARM_ID,),
        max_simultaneous_arms=1,
        arm_joint_names=FrozenDict({ARM_ID: ARM_JOINTS}),
        jaw_joint_names=FrozenDict({ARM_ID: GRIPPER_JOINTS}),
        stages_per_arm=MAX_BATCH_STAGES,
        max_stages=MAX_BATCH_STAGES,
        gripper=gripper,
        pose_inputs=frozenset({"tcp_quaternion", "grasp_angles"}),
        supports=PANDA_SUPPORTS,
        stall_guard_semantics=PANDA_STALL_GUARD_SEMANTICS,
        vertical_window_semantics=PANDA_VERTICAL_WINDOW_SEMANTICS,
        speed_scale=speed,
        gripper_speed_scale=gripper_speed,
        speed_presets=presets_within(SPEED_PRESETS, speed),
        gripper_speed_presets=presets_within(GRIPPER_SPEED_PRESETS, gripper_speed),
        tolerance_presets=presets,
        default_tolerance_preset="normal",
        default_tolerance=presets["normal"],
        max_position_tolerance_m=float(t["max_position_tolerance_m"]),
        max_orientation_tolerance_rad=float(t["max_orientation_tolerance_rad"]),
        stall_guard_presets_m=stall_guard_presets(t),
        tracking_divergence_range_m=(
            float(t["stall_guard_min_divergence_m"]),
            float(t["stall_guard_max_divergence_m"]),
        ),
        clock="simulated_steps",
        control_hz=control_hz,
        step_domain="libero_control_step",
        step_label="native step",
        per_request_step_limit=per_request_step_limit,
        default_action_timeout_s=PANDA_DEFAULT_ACTION_TIMEOUT_S,
        default_wait_timeout_s=PANDA_DEFAULT_WAIT_TIMEOUT_S,
        max_timeout_s=PANDA_MAX_TIMEOUT_S,
        cameras=(PANDA_OVERVIEW_CAMERA, *(cam for _, cam in PANDA_WRIST_CAMERAS)),
        overview_camera=PANDA_OVERVIEW_CAMERA,
        wrist_cameras=FrozenDict(PANDA_WRIST_CAMERAS),
        camera_examples=PANDA_CAMERA_EXAMPLES,
        support_surface_z_m=None,
        handoff=False,
        home_check=None,
        policy_step=False,
    )


@functools.lru_cache(maxsize=1)
def _default_panda_profile() -> EmbodimentProfile:
    return libero_franka_panda_profile()


@functools.lru_cache(maxsize=1)
def panda_chain() -> KinematicChainSpec:
    """Return the canonical 7-DoF ``KinematicChainSpec`` from ``panda_link0`` to ``gripper0_grip_site``."""
    from ...model.kinematics import chain_from_manifest

    manifest = _packaged_model()
    return chain_from_manifest(
        manifest,
        joint_names=ARM_JOINTS,
        tip_link=TCP_LINK,
        base_link=BASE_LINK,
    )


def _arm_vector(q_arm_rad):
    import numpy as np

    q = np.asarray(q_arm_rad, dtype=np.float64).reshape(-1)
    if q.shape != (len(ARM_JOINTS),) or not np.isfinite(q).all():
        raise ValueError("libero_franka_panda_requires_seven_finite_arm_joint_values_rad")
    return q


def _validate_target(target_T_base_tcp):
    import numpy as np

    target = np.asarray(target_T_base_tcp, dtype=np.float64)
    if target.shape != (4, 4) or not np.isfinite(target).all():
        raise ValueError("libero_franka_panda_ik_target_must_be_finite_4x4")
    rot = target[:3, :3]
    if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-5) or abs(float(np.linalg.det(rot)) - 1.0) > 1e-5:
        raise ValueError("libero_franka_panda_ik_target_rotation_invalid")
    return target


def fk_all(model, q_arm_rad, *, finger_q_m: tuple[float, float] = (0.04, -0.04)):
    """4x4 ``T_base_link`` for every URDF link in ``franka_base`` via ``kinematics.fk_links``."""
    import numpy as np
    from ...model.kinematics import fk_links

    manifest = _packaged_model() if model is None else model
    q = _arm_vector(q_arm_rad)
    f1, f2 = float(finger_q_m[0]), float(finger_q_m[1])
    if not (math.isfinite(f1) and math.isfinite(f2)):
        raise ValueError("invalid_finger_joint_state_m")
    joint_map = {joint["name"]: 0.0 for joint in manifest["joints"] if joint["type"] != "fixed"}
    for name, val in zip(ARM_JOINTS, q):
        joint_map[name] = float(val)
    joint_map["panda_finger_joint1"] = f1
    joint_map["panda_finger_joint2"] = f2
    return fk_links(manifest, joint_map, T_world_base=np.eye(4, dtype=np.float64))


def fk(model, q_arm_rad, link: str = TCP_LINK, *, finger_q_m: tuple[float, float] = (0.04, -0.04)):
    """4x4 ``T_base_link`` of ``link`` (default ``gripper0_grip_site``) in ``franka_base``."""
    poses = fk_all(model, q_arm_rad, finger_q_m=finger_q_m)
    if link not in poses:
        raise ValueError(f"libero_franka_panda_unknown_link:{link}")
    return poses[link]


def ik(
    model,
    target_T_base_tcp,
    q_seed_rad,
    *,
    position_weight: float = 1.0,
    orientation_weight: float = 0.5,
    posture_weight: float = 1e-3,
    max_iterations: int = 200,
    limits=None,
    position_tolerance_m: float = 1e-4,
    orientation_tolerance_rad: float = 1e-3,
    restarts: int = 8,
) -> dict:
    """7-DoF posture-regularized DLS IK for ``gripper0_grip_site`` via ``kinematics.solve_ik``."""
    from ...model.kinematics import IKOptions, IKTarget, chain_from_manifest, solve_ik

    target = _validate_target(target_T_base_tcp)
    seed = _arm_vector(q_seed_rad)
    if model is None and limits is None:
        chain = panda_chain()
    else:
        manifest = _packaged_model() if model is None else model
        arm_bounds = joint_limits_rad(manifest)
        for name, value in (limits or {}).items():
            if name not in ARM_JOINTS:
                raise ValueError(f"libero_franka_panda_ik_unknown_limit_joint:{name}")
            arm_bounds[name] = (float(value[0]), float(value[1]))
        chain = chain_from_manifest(
            manifest,
            joint_names=ARM_JOINTS,
            tip_link=TCP_LINK,
            base_link=BASE_LINK,
            joint_limits=arm_bounds,
        )
    options = IKOptions(
        tasks=("pose", "posture") if posture_weight > 0 else ("pose",),
        position_weight=float(position_weight),
        orientation_weight=float(orientation_weight),
        posture_weight=float(posture_weight),
        max_iterations=int(max_iterations),
        position_tolerance_m=float(position_tolerance_m),
        orientation_tolerance_rad=float(orientation_tolerance_rad),
        restarts=int(restarts),
        damping="adaptive_lm",
    )
    return solve_ik(chain, IKTarget(T_base_tip=target, posture_rad=seed), seed, options).as_dict()


def ik_position_priority(
    model,
    target_T_base_tcp,
    q_seed_rad,
    *,
    max_iterations: int = 200,
    limits=None,
    position_tolerance_m: float = 1e-4,
    restarts: int = 8,
) -> dict:
    """Position-only 7-DoF IK for unconstrained-orientation targets via ``kinematics.solve_ik``."""
    from ...model.kinematics import IKOptions, IKTarget, chain_from_manifest, solve_ik

    target = _validate_target(target_T_base_tcp)
    seed = _arm_vector(q_seed_rad)
    if model is None and limits is None:
        chain = panda_chain()
    else:
        manifest = _packaged_model() if model is None else model
        arm_bounds = joint_limits_rad(manifest)
        for name, value in (limits or {}).items():
            if name not in ARM_JOINTS:
                raise ValueError(f"libero_franka_panda_ik_unknown_limit_joint:{name}")
            arm_bounds[name] = (float(value[0]), float(value[1]))
        chain = chain_from_manifest(
            manifest,
            joint_names=ARM_JOINTS,
            tip_link=TCP_LINK,
            base_link=BASE_LINK,
            joint_limits=arm_bounds,
        )
    options = IKOptions(
        tasks=("pose", "posture"),
        position_weight=1.0,
        orientation_weight=0.0,
        posture_weight=1e-3,
        max_iterations=int(max_iterations),
        position_tolerance_m=float(position_tolerance_m),
        orientation_tolerance_rad=math.pi,
        restarts=int(restarts),
        damping="adaptive_lm",
    )
    return solve_ik(chain, IKTarget(T_base_tip=target, posture_rad=seed), seed, options).as_dict()
