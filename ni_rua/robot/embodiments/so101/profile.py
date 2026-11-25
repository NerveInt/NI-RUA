"""SO-101 5-DoF embodiment identity and URDF kinematics routed onto ``ni_rua.robot.model.kinematics``.

Units: metres (``m``) and radians (``rad``). Frames: ``base_link`` (``so101_base``) and TCP
link ``gripper_frame_link``. Pure NumPy over the packaged URDF manifest; zero serial, camera,
simulator, or LeRobot imports.
"""
from __future__ import annotations

import copy
import functools
import json
import math
from pathlib import Path
import tempfile
import zipfile

from ....core.embodiment import RobotDescription
from ...model.robot_model import urdf_manifest

SO101 = RobotDescription(
    robot_model_id="so101",
    frame_id="so101_base",
    tcp_config_id="so101-gripper-frame-v1",
)

ARM_JOINTS: tuple[str, ...] = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER_JOINT = "gripper"
BASE_LINK = "base_link"
TCP_LINK = "gripper_frame_link"
WRIST_CAMERA_PARENT_LINK = "gripper_link"

_PACKAGE = Path(__file__).resolve().parents[2] / "model" / "robot_information_data" / "so101"


def _validated(model: dict) -> dict:
    names = {joint["name"]: joint for joint in model["joints"]}
    for name in (*ARM_JOINTS, GRIPPER_JOINT):
        if names.get(name, {}).get("type") != "revolute" or "limits" not in names[name]:
            raise ValueError(f"so101_model_missing_revolute_joint:{name}")
    if model["base_link"] != BASE_LINK or TCP_LINK not in {joint["child"] for joint in model["joints"]}:
        raise ValueError("so101_model_missing_base_or_tcp")
    return model


@functools.lru_cache(maxsize=1)
def _packaged_model() -> dict:
    """Validated manifest of the packaged URDF, built once per process (zip extract + XML parse)."""
    native = json.loads((_PACKAGE / "physical.json").read_text(encoding="utf-8"))["geometry"]["native_model"]
    with zipfile.ZipFile(_PACKAGE / f"{native['resource_id']}.zip") as archive:
        data = archive.read(native["assembly_member"])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / Path(native["assembly_member"]).name
        path.write_bytes(data)
        return _validated(urdf_manifest(path))


def load_model(urdf_path=None) -> dict:
    """Return ``urdf_manifest()`` of the SO-101 URDF (from ``urdf_path`` or packaged ``so101/geometry_model.zip``).

    The packaged manifest is parsed once per process; every call returns an independent deep copy, so callers may
    mutate it freely."""
    if urdf_path is not None:
        return _validated(urdf_manifest(urdf_path))
    return copy.deepcopy(_packaged_model())


def joint_limits_rad(model) -> dict[str, tuple[float, float]]:
    """URDF-declared ``(lower_rad, upper_rad)`` bounds in ``rad`` for the five arm joints and the jaw."""
    names = {joint["name"]: joint for joint in model["joints"]}
    return {
        name: (float(names[name]["limits"]["lower"]), float(names[name]["limits"]["upper"]))
        for name in (*ARM_JOINTS, GRIPPER_JOINT)
    }


def so101_chain(model=None, *, limits=None):
    """Build the 5-DoF ``KinematicChainSpec`` from ``base_link`` to ``gripper_frame_link``."""
    from ...model.kinematics import chain_from_manifest

    manifest = load_model() if model is None else model
    bounds = joint_limits_rad(manifest)
    arm_bounds = {name: bounds[name] for name in ARM_JOINTS}
    for name, value in (limits or {}).items():
        if name not in ARM_JOINTS:
            raise ValueError(f"so101_ik_unknown_limit_joint:{name}")
        lo, hi = float(value[0]), float(value[1])
        if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
            raise ValueError(f"so101_ik_invalid_limits:{name}")
        arm_bounds[name] = (lo, hi)
    return chain_from_manifest(
        manifest,
        joint_names=ARM_JOINTS,
        tip_link=TCP_LINK,
        base_link=BASE_LINK,
        joint_limits=arm_bounds,
    )


def _arm_vector(q_arm_rad):
    import numpy as np

    q = np.asarray(q_arm_rad, dtype=np.float64).reshape(-1)
    if q.shape != (len(ARM_JOINTS),) or not np.isfinite(q).all():
        raise ValueError("so101_requires_five_finite_arm_joint_values_rad")
    return q


def _validate_target(target_T_base_tcp):
    import numpy as np

    target = np.asarray(target_T_base_tcp, dtype=np.float64)
    if target.shape != (4, 4) or not np.isfinite(target).all():
        raise ValueError("so101_ik_target_must_be_finite_4x4")
    rotation = target[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or abs(float(np.linalg.det(rotation)) - 1.0) > 1e-6:
        raise ValueError("so101_ik_target_rotation_invalid")
    return target


def _chain(model, q_arm, gripper_rad):
    """Return ``(T_base_link per link, joint (origin, axis) in base for arm joints)`` backed by ``fk_all``."""
    import numpy as np

    poses = fk_all(model, q_arm, gripper_rad=gripper_rad)
    axes = {}
    for joint in model["joints"]:
        if joint["name"] in ARM_JOINTS and joint["type"] != "fixed":
            frame = poses[joint["parent"]] @ np.asarray(joint["T_parent_joint"], dtype=np.float64)
            axis = np.asarray(joint["axis"], dtype=np.float64)
            axis = axis / np.linalg.norm(axis)
            axes[joint["name"]] = (frame[:3, 3].copy(), frame[:3, :3] @ axis)
    return poses, axes


def fk_all(model, q_arm_rad, *, gripper_rad: float = 0.0):
    """4x4 ``T_base_link`` for every URDF link (delegating to ``kinematics.fk_links``)."""
    import numpy as np
    from ...model.kinematics import fk_links

    q = _arm_vector(q_arm_rad)
    g_val = float(gripper_rad)
    if not math.isfinite(g_val):
        raise ValueError("invalid_joint_state_or_axis")
    # Preserve exact legacy behavior when URDF contains non-arm joints omitted from state (defaulting to 0.0)
    joint_map = {joint["name"]: 0.0 for joint in model["joints"] if joint["type"] != "fixed" and "mimic" not in joint}
    for name, val in zip(ARM_JOINTS, q):
        joint_map[name] = float(val)
    joint_map[GRIPPER_JOINT] = g_val
    return fk_links(model, joint_map, T_world_base=np.eye(4, dtype=np.float64))


def fk(model, q_arm_rad, link: str = TCP_LINK):
    """4x4 ``T_base_link`` of ``link`` (default TCP ``gripper_frame_link``)."""
    poses = fk_all(model, q_arm_rad)
    if link not in poses:
        raise ValueError(f"so101_unknown_link:{link}")
    return poses[link]


def ik(
    model,
    target_T_base_tcp,
    q_seed_rad,
    *,
    position_weight: float = 1.0,
    orientation_weight: float = 0.3,
    max_iterations: int = 200,
    limits=None,
    position_tolerance_m: float = 1e-3,
    orientation_tolerance_rad: float = 1e-2,
    restarts: int = 8,
) -> dict:
    """Damped-least-squares IK for the 5-DoF SO-101 arm TCP ``gripper_frame_link`` via ``kinematics.solve_ik``."""
    from ...model.kinematics import IKOptions, IKTarget, solve_ik

    wp, wo = float(position_weight), float(orientation_weight)
    if not (wp > 0 and wo >= 0 and math.isfinite(wp) and math.isfinite(wo)):
        raise ValueError("so101_ik_invalid_weights")
    target = _validate_target(target_T_base_tcp)
    seed = _arm_vector(q_seed_rad)
    chain = so101_chain(model, limits=limits)
    options = IKOptions(
        tasks=("pose",),
        position_weight=wp,
        orientation_weight=wo,
        max_iterations=int(max_iterations),
        position_tolerance_m=float(position_tolerance_m),
        orientation_tolerance_rad=float(orientation_tolerance_rad),
        restarts=int(restarts),
        damping="adaptive_lm",
    )
    return solve_ik(chain, IKTarget(T_base_tip=target), seed, options).as_dict()


def ik_position_priority(
    model,
    target_T_base_tcp,
    q_seed_rad,
    *,
    max_iterations: int = 200,
    limits=None,
    position_tolerance_m: float = 1e-3,
    orientation_tolerance_rad: float = 1e-2,
    restarts: int = 8,
) -> dict:
    """Position-first null-space IK for the 5-DoF SO-101 arm TCP ``gripper_frame_link`` via ``kinematics.solve_ik``."""
    from ...model.kinematics import IKOptions, IKTarget, solve_ik

    target = _validate_target(target_T_base_tcp)
    seed = _arm_vector(q_seed_rad)
    chain = so101_chain(model, limits=limits)
    options = IKOptions(
        tasks=("position", "orientation"),
        max_iterations=int(max_iterations),
        position_tolerance_m=float(position_tolerance_m),
        orientation_tolerance_rad=float(orientation_tolerance_rad),
        restarts=int(restarts),
        damping="adaptive_lm",
    )
    return solve_ik(chain, IKTarget(T_base_tip=target), seed, options).as_dict()
