"""Pinocchio + Pink + DAQP + Coal backend for SO-101 dynamics, inverse kinematics, and collision checking.

Physical conventions & units:
- Frame ``base_link`` (``so101_base``) is right-handed with ``+Z`` upward and gravity ``[0, 0, -9.80665] m/s^2``.
- TCP frame ``gripper_frame_link`` has ``+Z`` along the outward finger approach axis and ``+X`` toward the fixed jaw.
- Positions, penetrations, halfspaces, boxes, and spheres are in metres (``m``).
- Joint positions, limits, step resolutions, and orientation residuals are in radians (``rad``).
- Generalized gravity torques follow NI-RUA's ``-dU/dq`` sign convention in Newton-metres (``N*m``).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import functools
import json
import math
from pathlib import Path
import struct
import time
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

import coal
import numpy as np
import pink
import pinocchio as pin
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ARM_JOINTS: tuple[str, ...] = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER_JOINT = "gripper"
BASE_LINK = "base_link"
TCP_LINK = "gripper_frame_link"
ATTACHED_LINK = "attached_object"
GRAVITY_M_S2 = 9.80665
MAX_REPORTED_COLLISIONS = 6

# Rigid wrist/jaw assembly links never checked against a held object attached to ``gripper_frame_link``.
GRIPPER_GROUP_LINKS: frozenset[str] = frozenset(
    {
        "gripper_link",
        "moving_jaw_so101_v1_link",
        "wrist_camera_mount_link",
        "wrist_camera_link",
        "gripper_frame_link",
    }
)

_SO101_PACKAGE = Path(__file__).resolve().parent / "robot_information_data" / "so101"


def _packaged_archive_and_member(archive_path: str | Path | None = None) -> tuple[Path, str | None]:
    if archive_path is None:
        native = json.loads((_SO101_PACKAGE / "physical.json").read_text(encoding="utf-8"))["geometry"]["native_model"]
        return _SO101_PACKAGE / f"{native['resource_id']}.zip", str(native["assembly_member"])
    return Path(archive_path).resolve(), None


@functools.lru_cache(maxsize=4)
def _packaged_urdf_xml() -> str:
    zip_path, member = _packaged_archive_and_member(None)
    assert member is not None
    with zipfile.ZipFile(zip_path) as archive:
        return archive.read(member).decode("utf-8")


def _normalize_urdf_xml_for_pinocchio(xml_bytes: bytes) -> str:
    """Ensure minimal URDFs (omitting zero ``<inertia>`` or ``effort``/``velocity`` attributes) parse cleanly in urdfdom."""
    root = ET.fromstring(xml_bytes)
    for inertial in root.findall(".//link/inertial"):
        if inertial.find("inertia") is None:
            ET.SubElement(
                inertial,
                "inertia",
                ixx="0",
                ixy="0",
                ixz="0",
                iyy="0",
                iyz="0",
                izz="0",
            )
    for limit in root.findall(".//joint/limit"):
        if "effort" not in limit.attrib:
            limit.set("effort", "0")
        if "velocity" not in limit.attrib:
            limit.set("velocity", "0")
    return ET.tostring(root, encoding="unicode")


@functools.lru_cache(maxsize=8)
def _cached_gravity_model(cache_key: tuple[str, int, int] | None) -> tuple[pin.Model, list[int], list[int], int | None]:
    if cache_key is None:
        xml_text = _packaged_urdf_xml()
    else:
        xml_text = _normalize_urdf_xml_for_pinocchio(Path(cache_key[0]).read_bytes())
    model = pin.buildModelFromXML(xml_text)
    model.gravity.linear = np.array([0.0, 0.0, -GRAVITY_M_S2], dtype=np.float64)
    arm_idx_q: list[int] = []
    arm_idx_v: list[int] = []
    for joint_name in ARM_JOINTS:
        if not model.existJointName(joint_name):
            raise ValueError(f"so101_model_missing_revolute_joint:{joint_name}")
        jid = model.getJointId(joint_name)
        arm_idx_q.append(int(model.joints[jid].idx_q))
        arm_idx_v.append(int(model.joints[jid].idx_v))
    jaw_idx_q: int | None = None
    if model.existJointName(GRIPPER_JOINT):
        jaw_jid = model.getJointId(GRIPPER_JOINT)
        jaw_idx_q = int(model.joints[jaw_jid].idx_q)
    return model, arm_idx_q, arm_idx_v, jaw_idx_q


def _arm_vector(q_arm_rad: Any) -> np.ndarray:
    q = np.asarray(q_arm_rad, dtype=np.float64).reshape(-1)
    if q.shape != (len(ARM_JOINTS),) or not np.isfinite(q).all():
        raise ValueError("so101_requires_five_finite_arm_joint_values_rad")
    return q


def _validate_rigid_target(target_T_base_tcp: Any) -> np.ndarray:
    target = np.asarray(target_T_base_tcp, dtype=np.float64)
    if target.shape != (4, 4) or not np.isfinite(target).all():
        raise ValueError("so101_ik_target_must_be_finite_4x4")
    rot = target[:3, :3]
    if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-6) or abs(float(np.linalg.det(rot)) - 1.0) > 1e-6:
        raise ValueError("so101_ik_target_rotation_invalid")
    return target


def _manifest_gravity_key(
    manifest: Mapping[str, Any],
    inertials: Mapping[str, Any],
) -> tuple[
    str,
    tuple[tuple[str, str, str, str, bytes, bytes | None, tuple[str, float, float] | None], ...],
    tuple[tuple[str, float, float, float, float], ...],
]:
    from .kinematics import _finite_array, rigid_transform

    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("joints"), (list, tuple)):
        raise ValueError("robot_model_missing_joints_list")
    base_link = str(manifest.get("base_link", BASE_LINK))
    joint_specs: list[tuple[str, str, str, str, bytes, bytes | None, tuple[str, float, float] | None]] = []
    for idx_j, joint in enumerate(manifest["joints"]):
        j_name = str(joint.get("name", f"joint_{idx_j}"))
        kind = str(joint["type"])
        parent = str(joint["parent"])
        child = str(joint["child"])
        origin = np.ascontiguousarray(rigid_transform(joint["T_parent_joint"]), dtype=np.float64)
        if kind == "fixed":
            joint_specs.append((j_name, "fixed", parent, child, origin.tobytes(), None, None))
            continue
        if kind not in ("revolute", "continuous", "prismatic"):
            raise ValueError(f"unsupported_joint_type:{kind}")
        axis = _finite_array(joint["axis"], (3,))
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            raise ValueError("invalid_joint_state_or_axis")
        unit_axis = np.ascontiguousarray(axis / norm, dtype=np.float64)
        raw_mimic = joint.get("mimic")
        mimic_spec = (
            (str(raw_mimic["joint"]), float(raw_mimic.get("multiplier", 1.0)), float(raw_mimic.get("offset", 0.0)))
            if isinstance(raw_mimic, Mapping)
            else None
        )
        joint_specs.append((j_name, kind, parent, child, origin.tobytes(), unit_axis.tobytes(), mimic_spec))

    inertial_specs: list[tuple[str, float, float, float, float]] = []
    for link_name in sorted(inertials):
        info = inertials[link_name]
        mass_val = float(info["mass_kg"])
        com = [float(v) for v in info["com_m"]]
        if len(com) != 3 or not np.isfinite([mass_val, *com]).all() or mass_val < 0.0:
            raise ValueError(f"so101_inertial_invalid:{link_name}")
        if mass_val > 0.0:
            inertial_specs.append((str(link_name), mass_val, com[0], com[1], com[2]))

    return base_link, tuple(joint_specs), tuple(inertial_specs)


@functools.lru_cache(maxsize=16)
def _cached_manifest_gravity_model(
    base_link: str,
    joint_specs: tuple[tuple[str, str, str, str, bytes, bytes | None, tuple[str, float, float] | None], ...],
    inertial_specs: tuple[tuple[str, float, float, float, float], ...],
) -> tuple[
    pin.Model,
    list[int],
    list[int],
    int | None,
    dict[str, int],
    tuple[tuple[int, str, float, float], ...],
]:
    from .kinematics import _pin_joint_model

    p_model = pin.Model()
    p_model.gravity.linear = np.array([0.0, 0.0, -GRAVITY_M_S2], dtype=np.float64)
    link_parent_joint: dict[str, int] = {base_link: 0}
    link_rel_placement: dict[str, pin.SE3] = {base_link: pin.SE3.Identity()}
    joint_idx_q: dict[str, int] = {}
    joint_idx_v: dict[str, int] = {}
    mimic_entries: list[tuple[int, str, float, float]] = []

    for idx_j, (j_name, kind, parent, child, origin_bytes, axis_bytes, mimic_spec) in enumerate(joint_specs):
        if parent not in link_parent_joint:
            raise KeyError(parent)
        p_jid = link_parent_joint[parent]
        origin = np.frombuffer(origin_bytes, dtype=np.float64).reshape(4, 4)
        placement = link_rel_placement[parent] * pin.SE3(origin[:3, :3], origin[:3, 3])
        if kind == "fixed":
            link_parent_joint[child] = p_jid
            link_rel_placement[child] = placement
        else:
            assert axis_bytes is not None
            axis = np.frombuffer(axis_bytes, dtype=np.float64).reshape(3)
            jm = _pin_joint_model(pin, kind, axis)
            pin_j_name = j_name if not p_model.existJointName(j_name) else f"{idx_j}:{j_name}"
            jid = int(p_model.addJoint(p_jid, jm, placement, pin_j_name))
            link_parent_joint[child] = jid
            link_rel_placement[child] = pin.SE3.Identity()
            idx_q = int(p_model.joints[jid].idx_q)
            idx_v = int(p_model.joints[jid].idx_v)
            if mimic_spec is not None:
                mimic_entries.append((idx_q, mimic_spec[0], mimic_spec[1], mimic_spec[2]))
            else:
                joint_idx_q[j_name] = idx_q
                joint_idx_v[j_name] = idx_v

    zero_rot_inertia = np.zeros((3, 3), dtype=np.float64)
    for link_name, mass_val, cx, cy, cz in inertial_specs:
        if link_name in link_parent_joint and mass_val > 0.0:
            inertia = pin.Inertia(
                mass_val,
                np.array([cx, cy, cz], dtype=np.float64),
                zero_rot_inertia,
            )
            p_model.appendBodyToJoint(link_parent_joint[link_name], inertia, link_rel_placement[link_name])

    arm_idx_q: list[int] = []
    arm_idx_v: list[int] = []
    for joint_name in ARM_JOINTS:
        if joint_name not in joint_idx_q:
            raise ValueError(f"so101_model_missing_revolute_joint:{joint_name}")
        arm_idx_q.append(joint_idx_q[joint_name])
        arm_idx_v.append(joint_idx_v[joint_name])
    jaw_idx_q = joint_idx_q.get(GRIPPER_JOINT)
    return p_model, arm_idx_q, arm_idx_v, jaw_idx_q, joint_idx_q, tuple(mimic_entries)


def pinocchio_gravity_torques(
    urdf_path: Mapping[str, Any] | str | Path | Sequence[float] | np.ndarray | None = None,
    q_arm_rad: Mapping[str, Any] | Sequence[float] | np.ndarray | None = None,
    q_arm_rad_pos: Sequence[float] | np.ndarray | None = None,
    *,
    model: Mapping[str, Any] | None = None,
    inertials: Mapping[str, Any] | None = None,
    gripper_rad: float = 0.0,
) -> np.ndarray:
    """Compute exact RNEA static gravity torques ``-dU/dq`` (``N*m``) for the 5 SO-101 arm joints.

    Parameters:
        urdf_path: Optional filesystem path to an SO-101 URDF file, ``None`` to load the packaged
            ``so101/geometry_model.zip`` assembly URDF, or an in-memory ``urdf_manifest`` ``dict``
            paired with ``inertials`` (positional ``(model, inertials, q_arm_rad)`` or keyword ``inertials=...``).
        q_arm_rad: Five finite arm joint angles in ``ARM_JOINTS`` order in ``rad`` (or ``inertials`` ``dict``
            when called as ``pinocchio_gravity_torques(model, inertials, q_arm_rad)``).
        q_arm_rad_pos: Five finite arm joint angles in ``rad`` when called positionally as
            ``pinocchio_gravity_torques(model, inertials, q_arm_rad)``.
        model: Optional explicit in-memory ``urdf_manifest`` ``dict`` (paired with ``inertials``).
        inertials: Optional ``{link: {"mass_kg": float, "com_m": [x, y, z]}}`` dict in ``kg`` and ``m``.
        gripper_rad: Jaw joint angle in ``rad`` (default ``0.0``).

    Returns:
        ``float64`` NumPy array of shape ``(5,)`` in ``N*m`` where positive values mean gravity pulls the
        joint toward larger URDF angles (matching ``so101_gravity.gravity_torques`` to ``< 1e-14 N*m``).

    Example:
        ``tau_nm = pinocchio_gravity_torques(None, [0.0, -1.0, 1.0, 0.6, 0.0], gripper_rad=0.0)``
    """
    manifest_arg: Mapping[str, Any] | None = model
    inertials_arg: Mapping[str, Any] | None = inertials
    q_raw: Any = q_arm_rad
    resolved_urdf: str | Path | None = None

    if manifest_arg is not None:
        if isinstance(q_arm_rad, Mapping) and q_arm_rad_pos is not None:
            inertials_arg = q_arm_rad
            q_raw = q_arm_rad_pos
    elif isinstance(urdf_path, Mapping):
        manifest_arg = urdf_path
        if isinstance(q_arm_rad, Mapping):
            inertials_arg = q_arm_rad
            q_raw = q_arm_rad_pos
    elif q_arm_rad is None and q_arm_rad_pos is None and urdf_path is not None and not isinstance(urdf_path, (str, Path)):
        q_raw = urdf_path
        resolved_urdf = None
    elif isinstance(urdf_path, (str, Path)) or urdf_path is None:
        resolved_urdf = urdf_path
    else:
        raise TypeError("invalid_pinocchio_gravity_torques_model_argument")

    if q_raw is None:
        raise ValueError("so101_requires_five_finite_arm_joint_values_rad")
    q_arm = _arm_vector(q_raw)
    g_val = float(gripper_rad)
    if not math.isfinite(g_val):
        raise ValueError("invalid_joint_state_or_axis")

    if manifest_arg is not None:
        if inertials_arg is None:
            raise ValueError("so101_gravity_manifest_requires_inertials")
        base_link, j_specs, i_specs = _manifest_gravity_key(manifest_arg, inertials_arg)
        p_model, arm_idx_q, arm_idx_v, jaw_idx_q, joint_idx_q, mimic_entries = _cached_manifest_gravity_model(
            base_link, j_specs, i_specs
        )
        p_data = p_model.createData()
        q_full = np.zeros(p_model.nq, dtype=np.float64)
        q_full[arm_idx_q] = q_arm
        if jaw_idx_q is not None:
            q_full[jaw_idx_q] = g_val
        for idx_q_m, master_name, mult, offs in mimic_entries:
            master_val = float(q_full[joint_idx_q[master_name]]) if master_name in joint_idx_q else 0.0
            q_full[idx_q_m] = master_val * mult + offs
        tau_rnea = pin.computeGeneralizedGravity(p_model, p_data, q_full)
        return np.asarray(-tau_rnea[arm_idx_v], dtype=np.float64).copy()

    cache_key: tuple[str, int, int] | None = None
    if resolved_urdf is not None:
        resolved = Path(resolved_urdf).resolve()
        st = resolved.stat()
        cache_key = (str(resolved), int(st.st_mtime_ns), int(st.st_size))

    p_model, arm_idx_q, arm_idx_v, jaw_idx_q = _cached_gravity_model(cache_key)
    p_data = p_model.createData()
    q_full = np.zeros(p_model.nq, dtype=np.float64)
    q_full[arm_idx_q] = q_arm
    if jaw_idx_q is not None:
        q_full[jaw_idx_q] = g_val
    tau_rnea = pin.computeGeneralizedGravity(p_model, p_data, q_full)
    return np.asarray(-tau_rnea[arm_idx_v], dtype=np.float64).copy()


@functools.lru_cache(maxsize=2)
def _cached_reduced_so101_model() -> tuple[pin.Model, dict[str, tuple[float, float]]]:
    full_model = pin.buildModelFromXML(_packaged_urdf_xml())
    jaw_jid = full_model.getJointId(GRIPPER_JOINT)
    q_ref = np.zeros(full_model.nq, dtype=np.float64)
    reduced = pin.buildReducedModel(full_model, [jaw_jid], q_ref)
    if tuple(reduced.names)[1:] != ARM_JOINTS:
        raise ValueError(f"so101_reduced_model_joint_order_mismatch:{tuple(reduced.names)[1:]}")
    if not reduced.existFrame(TCP_LINK):
        raise ValueError("so101_reduced_model_missing_tcp_frame")
    default_limits = {
        name: (float(reduced.lowerPositionLimit[i]), float(reduced.upperPositionLimit[i]))
        for i, name in enumerate(ARM_JOINTS)
    }
    return reduced, default_limits


def _build_so101_seeds(q_seed_clamped: np.ndarray, lower: np.ndarray, upper: np.ndarray, restarts: int) -> list[np.ndarray]:
    seeds = [q_seed_clamped.copy()]
    n_extra = max(0, int(restarts))
    if n_extra == 0:
        return seeds
    mid = 0.5 * (lower + upper)
    span = upper - lower
    structured = [
        mid,
        np.clip(np.array([q_seed_clamped[0], -0.85, 0.85, 0.75, q_seed_clamped[4]], dtype=np.float64), lower, upper),
        np.clip(np.array([q_seed_clamped[0], -0.45, 0.45, 1.10, 0.0], dtype=np.float64), lower, upper),
        np.clip(q_seed_clamped + np.array([0.0, -0.35, 0.45, -0.25, 0.0], dtype=np.float64), lower, upper),
        np.clip(q_seed_clamped + np.array([0.0, 0.35, -0.45, 0.25, 0.0], dtype=np.float64), lower, upper),
        np.clip(mid + 0.25 * span * np.array([0.0, -1.0, 1.0, 0.5, -0.5], dtype=np.float64), lower, upper),
        np.clip(mid + 0.25 * span * np.array([0.0, -0.5, 0.5, -0.5, 0.5], dtype=np.float64), lower, upper),
    ]
    rng = np.random.default_rng(0)
    for item in structured:
        if len(seeds) >= 1 + n_extra:
            break
        seeds.append(item)
    while len(seeds) < 1 + n_extra:
        seeds.append(rng.uniform(lower, upper))
    return seeds[: 1 + n_extra]


def solve_so101_pink_ik(
    target_T_base_tcp: np.ndarray | Sequence[Sequence[float]],
    q_seed_rad: Sequence[float] | np.ndarray,
    limits: Mapping[str, tuple[float, float] | Sequence[float]] | None = None,
    *,
    position_weight: float = 1.0,
    orientation_costs: tuple[float, float, float] | Sequence[float] = (0.3, 0.3, 0.3),
    max_iterations: int = 60,
    restarts: int = 8,
    position_tolerance_m: float = 1e-3,
    orientation_tolerance_rad: float = 1e-2,
    return_all_candidates: bool = False,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Solve 5-DoF SO-101 TCP inverse kinematics via ``Pink + DAQP`` over the reduced Pinocchio model.

    Parameters:
        target_T_base_tcp: ``4x4`` homogeneous target pose of ``gripper_frame_link`` in ``base_link`` (``m``).
        q_seed_rad: Five initial arm joint angles in ``ARM_JOINTS`` order (``rad``).
        limits: Mapping ``{joint_name: (lower_rad, upper_rad)}`` in ``rad`` for ``ARM_JOINTS``; ``None`` uses URDF bounds.
        position_weight: Scalar position cost weight in ``1/m`` (``> 0``, default ``1.0``).
        orientation_costs: 3-tuple ``(w_x, w_y, w_z)`` of local ``gripper_frame_link`` rotation weights in ``1/rad``
            (default ``(0.3, 0.3, 0.3)``; ``(w_app, w_app, 0.0)`` leaves roll about local ``+Z`` approach free).
        max_iterations: Maximum QP iterations per seed (default ``60``).
        restarts: Number of extra deterministic multi-start seeds beyond ``q_seed_rad`` (default ``8``).
        position_tolerance_m: Convergence threshold on Euclidean position residual in ``m`` (default ``1e-3``).
        orientation_tolerance_rad: Convergence threshold on constrained orientation residual in ``rad`` (default ``1e-2``).
        return_all_candidates: When ``True``, evaluates all ``1 + restarts`` seeds and returns ``list[dict]`` ordered
            best-first; when ``False`` (default), returns the best candidate ``dict``.

    Returns:
        ``{"q_rad": [5 floats], "converged": bool, "iterations": int, "position_error_m": float, "orientation_error_rad": float}``
        (or ``list[dict]`` of length ``1 + restarts`` when ``return_all_candidates=True``).

    Example:
        ``sol = solve_so101_pink_ik(target_T, [0.0, -0.9, 0.9, 1.0, 0.0], limits)``
    """
    target = _validate_rigid_target(target_T_base_tcp)
    seed = _arm_vector(q_seed_rad)
    wp = float(position_weight)
    ori_arr = np.asarray(orientation_costs, dtype=np.float64).reshape(-1)
    if not (math.isfinite(wp) and wp > 0.0) or ori_arr.shape != (3,) or not np.isfinite(ori_arr).all() or np.any(ori_arr < 0.0):
        raise ValueError("so101_ik_invalid_weights")

    base_model, default_limits = _cached_reduced_so101_model()
    resolved_limits = dict(default_limits)
    if limits is not None:
        for name, bounds in limits.items():
            if name not in ARM_JOINTS:
                raise ValueError(f"so101_ik_unknown_limit_joint:{name}")
            lo_j, hi_j = float(bounds[0]), float(bounds[1])
            if not (math.isfinite(lo_j) and math.isfinite(hi_j) and lo_j <= hi_j):
                raise ValueError(f"so101_ik_invalid_limits:{name}")
            resolved_limits[name] = (lo_j, hi_j)

    lower = np.array([resolved_limits[name][0] for name in ARM_JOINTS], dtype=np.float64)
    upper = np.array([resolved_limits[name][1] for name in ARM_JOINTS], dtype=np.float64)
    model = base_model.copy()
    model.lowerPositionLimit[:] = lower
    model.upperPositionLimit[:] = upper
    data = model.createData()

    pos_tol = float(position_tolerance_m)
    ori_tol = float(orientation_tolerance_rad)
    max_iters = max(1, int(max_iterations))
    ori_active = bool(np.any(ori_arr > 0.0))

    seed_clamped = np.clip(seed, lower, upper)
    # Also compute pan-pointed seed toward target XY for fast global convergence on large pan turns
    pan_toward = math.atan2(float(target[1, 3]), float(target[0, 3]))
    seeds = _build_so101_seeds(seed_clamped, lower, upper, restarts)
    if len(seeds) >= 3 and math.isfinite(pan_toward):
        pan_seed = seed_clamped.copy()
        pan_seed[0] = float(np.clip(pan_toward, lower[0], upper[0]))
        seeds[2] = pan_seed

    target_se3 = pin.SE3(target[:3, :3], target[:3, 3])
    frame_task = pink.tasks.FrameTask(
        TCP_LINK,
        position_cost=wp,
        orientation_cost=ori_arr,
        lm_damping=1e-3 if ori_active else 0.0,
    )
    frame_task.set_target(target_se3)
    posture_task = pink.tasks.PostureTask(cost=1e-3)
    posture_task.set_target(seed_clamped)
    config_limit = pink.limits.ConfigurationLimit(model, config_limit_gain=1.0)
    tasks = [frame_task, posture_task]

    def _ori_converged(T_ach: np.ndarray, full_ori_err: float) -> tuple[bool, float]:
        if not ori_active:
            return True, 0.0
        rel_local = T_ach[:3, :3].T @ target[:3, :3]
        rotvec_local = Rotation.from_matrix(rel_local).as_rotvec()
        weighted_rot = float(np.sum((ori_arr * rotvec_local) ** 2))
        if ori_arr[2] == 0.0 and ori_arr[0] > 0.0 and ori_arr[1] > 0.0:
            z_dot = max(-1.0, min(1.0, float(rel_local[2, 2])))
            return math.acos(z_dot) <= ori_tol, weighted_rot
        return full_ori_err <= ori_tol, weighted_rot

    scored_candidates: list[tuple[tuple[bool, float, float], dict[str, Any]]] = []
    for start_q in seeds:
        q_init = np.clip(start_q, lower, upper)
        cfg = pink.Configuration(model, data, q_init)
        T_cur = cfg.get_transform_frame_to_world(TCP_LINK).homogeneous
        pos_err = float(np.linalg.norm(target[:3, 3] - T_cur[:3, 3]))
        ori_err = float(np.linalg.norm(Rotation.from_matrix(target[:3, :3] @ T_cur[:3, :3].T).as_rotvec()))
        ori_ok, w_rot2 = _ori_converged(T_cur, ori_err)
        best_q = q_init.copy()
        best_pos = pos_err
        best_ori = ori_err
        best_conv = (pos_err <= pos_tol) and ori_ok
        best_obj = (wp * pos_err) ** 2 + w_rot2
        iters_run = 0

        for it in range(1, max_iters + 1):
            if best_pos <= pos_tol * 0.1 and (not ori_active or best_ori <= ori_tol * 0.1):
                break
            try:
                vel = pink.solve_ik(
                    cfg,
                    tasks,
                    dt=1.0,
                    solver="daqp",
                    damping=1e-6,
                    limits=[config_limit],
                    safety_break=False,
                )
            except Exception as exc:
                from ...core.tool_errors import INTERNAL_BUG_TYPES
                if isinstance(exc, (*tuple(t for t in INTERNAL_BUG_TYPES if t is not ArithmeticError), ZeroDivisionError, KeyError)):
                    raise
                break
            iters_run = it
            v_inf = float(np.max(np.abs(vel)))
            if v_inf > 0.85:
                vel = vel * (0.85 / v_inf)
            q_prev = cfg.q.copy()
            cfg.integrate_inplace(vel, 1.0)
            q_now = np.clip(cfg.q, lower, upper)
            if not np.array_equal(q_now, cfg.q):
                cfg.update(q_now)
            T_cur = cfg.get_transform_frame_to_world(TCP_LINK).homogeneous
            pos_err = float(np.linalg.norm(target[:3, 3] - T_cur[:3, 3]))
            ori_err = float(np.linalg.norm(Rotation.from_matrix(target[:3, :3] @ T_cur[:3, :3].T).as_rotvec()))
            ori_ok, w_rot2 = _ori_converged(T_cur, ori_err)
            obj = (wp * pos_err) ** 2 + w_rot2
            conv_now = (pos_err <= pos_tol) and ori_ok
            if (conv_now and not best_conv) or (conv_now == best_conv and obj < best_obj):
                best_q = cfg.q.copy()
                best_pos = pos_err
                best_ori = ori_err
                best_conv = conv_now
                best_obj = obj
            if float(np.max(np.abs(cfg.q - q_prev))) < 1e-9:
                break

        travel = float(np.max(np.abs(best_q - seed)))
        entry = {
            "q_rad": [float(v) for v in best_q],
            "converged": bool(best_conv),
            "iterations": int(iters_run),
            "position_error_m": float(best_pos),
            "orientation_error_rad": float(best_ori),
        }
        scored_candidates.append(((not best_conv, best_obj, travel), entry))
        if not return_all_candidates and best_conv and best_pos <= min(pos_tol, 5e-4):
            return entry

    scored_candidates.sort(key=lambda item: item[0])
    if return_all_candidates:
        return [item[1] for item in scored_candidates]
    return scored_candidates[0][1]


def _stl_vertices(data: bytes) -> np.ndarray:
    """Parse binary or ASCII STL bytes into ``(M, 3)`` ``float64`` vertices in ``m`` (before URDF scale)."""
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        if count and len(data) == 84 + 50 * count:
            record = np.dtype([("normal", "<3f4"), ("vertices", "<9f4"), ("attribute", "<u2")])
            return np.frombuffer(data, dtype=record, count=count, offset=84)["vertices"].reshape(-1, 3).astype(np.float64)
    verts = [
        line.split()[1:4]
        for line in data.decode("ascii", "strict").splitlines()
        if line.strip().startswith("vertex")
    ]
    if not verts or len(verts) % 3:
        raise ValueError("so101_mesh_unreadable_stl")
    return np.asarray(verts, dtype=np.float64).reshape(-1, 3)


def _convex_hull_from_vertices(verts: np.ndarray) -> coal.Convex:
    """Build a compact C++ ``coal.Convex`` hull preserving exact axis extrema and 1.5 mm outer envelope."""
    extreme_indices: list[int] = []
    for axis in range(3):
        extreme_indices.append(int(np.argmin(verts[:, axis])))
        extreme_indices.append(int(np.argmax(verts[:, axis])))
    q_ijk = np.rint(verts * 666.6667).astype(np.int64)
    keys_1d = (q_ijk[:, 0] << 40) ^ (q_ijk[:, 1] << 20) ^ q_ijk[:, 2]
    _, uniq_idx = np.unique(keys_1d, return_index=True)
    keep_idx = np.unique(np.concatenate([uniq_idx, np.asarray(extreme_indices, dtype=np.int64)]))
    v_sub = verts[keep_idx]
    hull = ConvexHull(v_sub)
    hull_pts = np.ascontiguousarray(v_sub[hull.vertices], dtype=np.float64)
    remap = np.empty(len(v_sub), dtype=np.int32)
    remap[hull.vertices] = np.arange(len(hull.vertices), dtype=np.int32)
    tris = remap[hull.simplices]
    tri_vec = coal.StdVec_Triangle()
    tri_vec.extend(coal.Triangle(int(a), int(b), int(c)) for a, b, c in tris)
    pts_vec = coal.StdVec_Vec3s()
    pts_vec.extend(hull_pts)
    return coal.Convex(pts_vec, tri_vec)


class SO101CoalCollisionEngine:
    """Exact C++ ``Pinocchio + Coal`` self, table, scene-obstacle, and held-object collision engine for SO-101.

    Uses ``coal.Convex`` hulls for proximal arm/camera links and explicit ``coal.Box`` palm + finger-pad primitives
    derived from ``gripper_cad`` for ``gripper_link`` and ``moving_jaw_so101_v1_link`` so objects situated inside
    the open-jaw pocket are never falsely flagged as colliding with a bridged jaw convex hull.
    """

    def __init__(self, archive_path: str | Path | None = None) -> None:
        from ..embodiments.so101 import gripper_cad as gg, profile as s101

        zip_path, member = _packaged_archive_and_member(archive_path)
        with zipfile.ZipFile(zip_path) as archive:
            if member is None:
                member = next(name for name in archive.namelist() if name.endswith(".urdf"))
            urdf_bytes = archive.read(member)
            self.model: pin.Model = pin.buildModelFromXML(urdf_bytes.decode("utf-8"))
            self.arm_idx_q = [int(self.model.joints[self.model.getJointId(name)].idx_q) for name in ARM_JOINTS]
            self.jaw_idx_q = int(self.model.joints[self.model.getJointId(GRIPPER_JOINT)].idx_q)
            self.tcp_frame_id = int(self.model.getFrameId(TCP_LINK))

            geom_model = pin.GeometryModel()
            geom_links: list[str] = []
            mesh_cache: dict[tuple[str, tuple[float, float, float], bool], coal.Convex] = {}
            root = ET.fromstring(urdf_bytes)

            for link_node in root.findall("link"):
                link_name = str(link_node.get("name") or "")
                if not link_name or link_name in ("gripper_link", "moving_jaw_so101_v1_link"):
                    continue
                if not self.model.existFrame(link_name):
                    continue
                fid = int(self.model.getFrameId(link_name))
                p_joint = int(self.model.frames[fid].parentJoint)
                f_place = self.model.frames[fid].placement
                cols = link_node.findall("collision") or link_node.findall("visual")
                for col_idx, col_el in enumerate(cols):
                    mesh_el = col_el.find("geometry/mesh")
                    if mesh_el is None:
                        continue
                    fname = str(mesh_el.attrib["filename"])
                    scale_vals = tuple(float(v) for v in mesh_el.get("scale", "1 1 1").split())
                    scale_tup = (scale_vals[0], scale_vals[1], scale_vals[2])
                    orig_el = col_el.find("origin")
                    rot_col = (
                        Rotation.from_euler(
                            "xyz",
                            [float(v) for v in orig_el.get("rpy", "0 0 0").split()],
                            degrees=False,
                        ).as_matrix()
                        if orig_el is not None
                        else np.eye(3, dtype=np.float64)
                    )
                    pos_col = (
                        np.array([float(v) for v in orig_el.get("xyz", "0 0 0").split()], dtype=np.float64)
                        if orig_el is not None
                        else np.zeros(3, dtype=np.float64)
                    )
                    is_base = link_name == BASE_LINK
                    if is_base:
                        verts = _stl_vertices(archive.read(fname)) * np.asarray(scale_tup, dtype=np.float64)
                        verts_link = verts @ rot_col.T + pos_col
                        # Clip mounting-plate protrusion below z = 0 m in base_link (matching build_model.py)
                        verts_link[:, 2] = np.maximum(verts_link[:, 2], 0.0005)
                        cvx = _convex_hull_from_vertices(verts_link)
                        placement = f_place
                    else:
                        cache_key = (fname, scale_tup, False)
                        if cache_key not in mesh_cache:
                            verts = _stl_vertices(archive.read(fname)) * np.asarray(scale_tup, dtype=np.float64)
                            mesh_cache[cache_key] = _convex_hull_from_vertices(verts)
                        cvx = mesh_cache[cache_key]
                        placement = f_place * pin.SE3(rot_col, pos_col)
                    geom_model.addGeometryObject(
                        pin.GeometryObject(f"{link_name}_{col_idx}", p_joint, fid, placement, cvx)
                    )
                    geom_links.append(link_name)

        # Open-jaw explicit coal.Box primitives derived directly from CAD sections in gripper_cad._geometry
        cad = gg._geometry(archive_path)
        poses0 = s101.fk_all(cad.model, (0.0,) * len(ARM_JOINTS), gripper_rad=0.0)
        T_grip_tcp = np.linalg.inv(poses0["gripper_link"]) @ poses0[TCP_LINK]
        T_mov_tcp0 = np.linalg.inv(cad.T_tcp_moving(0.0))

        body_tcp = cad.body.reshape(-1, 3)
        mov0_tcp = cad.moving(0.0).reshape(-1, 3)
        palm_split_z = float(cad.palm_z) + 0.004
        mid_split_z = 0.5 * (palm_split_z + float(cad.fixed_tip[2]))

        for link_name, pts_tcp, T_link_from_tcp in (
            ("gripper_link", body_tcp, T_grip_tcp),
            ("moving_jaw_so101_v1_link", mov0_tcp, T_mov_tcp0),
        ):
            fid = int(self.model.getFrameId(link_name))
            p_joint = int(self.model.frames[fid].parentJoint)
            f_place = self.model.frames[fid].placement
            bands = (
                pts_tcp[:, 2] <= palm_split_z,
                (pts_tcp[:, 2] >= palm_split_z) & (pts_tcp[:, 2] <= mid_split_z),
                pts_tcp[:, 2] >= mid_split_z,
            )
            for b_idx, mask in enumerate(bands):
                sub = pts_tcp[mask]
                if len(sub) == 0:
                    continue
                lo, hi = sub.min(axis=0), sub.max(axis=0)
                dims = np.maximum(hi - lo, 1e-4)
                centre_tcp = 0.5 * (lo + hi)
                T_box = np.eye(4, dtype=np.float64)
                T_box[:3, 3] = centre_tcp
                T_link_box = T_link_from_tcp @ T_box
                placement = f_place * pin.SE3(T_link_box[:3, :3], T_link_box[:3, 3])
                geom_model.addGeometryObject(
                    pin.GeometryObject(
                        f"{link_name}_box_{b_idx}",
                        p_joint,
                        fid,
                        placement,
                        coal.Box(float(dims[0]), float(dims[1]), float(dims[2])),
                    )
                )
                geom_links.append(link_name)

        # Prune adjacent link pairs, co-moving gripper assembly pairs, and resting neutral/ready self-contacts
        probe_gm = pin.GeometryModel()
        for go in geom_model.geometryObjects:
            probe_gm.addGeometryObject(go)
        probe_gm.addAllCollisionPairs()
        probe_data = self.model.createData()
        probe_gd = probe_gm.createData()
        resting_pairs: set[tuple[str, str]] = set()
        for q_probe in (
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.174533], dtype=np.float64),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.74533], dtype=np.float64),
            np.array([0.0, -0.95, 0.9, 1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, -0.95, 0.9, 1.0, 0.0, 1.74533], dtype=np.float64),
        ):
            pin.computeCollisions(self.model, probe_data, probe_gm, probe_gd, q_probe, False)
            for cp_idx, cp in enumerate(probe_gm.collisionPairs):
                if probe_gd.collisionResults[cp_idx].isCollision():
                    la, lb = geom_links[int(cp.first)], geom_links[int(cp.second)]
                    if la != lb:
                        resting_pairs.add((la, lb) if la < lb else (lb, la))
        resting_pairs.add(("wrist_camera_mount_link", "wrist_link"))

        for i in range(geom_model.ngeoms):
            for j in range(i + 1, geom_model.ngeoms):
                la, lb = geom_links[i], geom_links[j]
                if la == lb:
                    continue
                if la in GRIPPER_GROUP_LINKS and lb in GRIPPER_GROUP_LINKS:
                    continue
                pj_i = int(geom_model.geometryObjects[i].parentJoint)
                pj_j = int(geom_model.geometryObjects[j].parentJoint)
                if abs(pj_i - pj_j) <= 1:
                    continue
                pair_key = (la, lb) if la < lb else (lb, la)
                if pair_key in resting_pairs:
                    continue
                geom_model.addCollisionPair(pin.CollisionPair(i, j))

        self.geom_model = geom_model
        self.geom_links = tuple(geom_links)
        self.table_check_indices = tuple(i for i, link in enumerate(self.geom_links) if link != BASE_LINK)
        self.non_gripper_indices = tuple(
            i for i, link in enumerate(self.geom_links) if link not in GRIPPER_GROUP_LINKS
        )
        self._table_gm_cache: dict[float | None, tuple[pin.GeometryModel, tuple[tuple[str, Any, Any], ...]]] = {}

    def _build_query_geom_model(
        self,
        table_top_z_m: float | None,
        obstacles: list[tuple[str, coal.CollisionGeometry, np.ndarray]],
        attached: list[tuple[coal.CollisionGeometry, np.ndarray, np.ndarray]],
    ) -> tuple[pin.GeometryModel, tuple[tuple[str, Any, Any], ...]]:
        """Build a unified C++ ``pin.GeometryModel`` checking self, table, obstacles, and attached objects in one C++ pass.

        Pair metadata tuple entries have the form:
        - ``("self", link_a, link_b)`` for robot self-collision pairs
        - ``("table", link_name, "table")`` for robot-vs-table pairs
        - ``("world", link_name, obs_id)`` for robot/attached-vs-obstacle (and attached-vs-table) pairs
        - ``("att_self", link_a, link_b)`` for attached-object-vs-arm pairs
        """
        tz_key = None if table_top_z_m is None else float(table_top_z_m)
        if not obstacles and not attached and tz_key in self._table_gm_cache:
            return self._table_gm_cache[tz_key]

        gm = pin.GeometryModel()
        for go in self.geom_model.geometryObjects:
            gm.addGeometryObject(go)

        meta: list[tuple[str, Any, Any]] = []
        for cp in self.geom_model.collisionPairs:
            i, j = int(cp.first), int(cp.second)
            gm.addCollisionPair(pin.CollisionPair(i, j))
            la, lb = self.geom_links[i], self.geom_links[j]
            meta.append(("self", la if la < lb else lb, lb if la < lb else la))

        table_gid: int | None = None
        if tz_key is not None:
            hs = coal.Halfspace(np.array([0.0, 0.0, 1.0], dtype=np.float64), tz_key)
            table_gid = int(gm.addGeometryObject(pin.GeometryObject("__table__", 0, 0, pin.SE3.Identity(), hs)))
            for idx in self.table_check_indices:
                gm.addCollisionPair(pin.CollisionPair(idx, table_gid))
                meta.append(("table", self.geom_links[idx], "table"))

        obs_gids: list[tuple[int, str]] = []
        eye3 = np.eye(3, dtype=np.float64)
        for obs_idx, (obs_id, obs_geom, obs_ctr) in enumerate(obstacles):
            gid = int(
                gm.addGeometryObject(pin.GeometryObject(f"__obs_{obs_idx}", 0, 0, pin.SE3(eye3, obs_ctr), obs_geom))
            )
            obs_gids.append((gid, obs_id))
            for idx in range(len(self.geom_links)):
                gm.addCollisionPair(pin.CollisionPair(idx, gid))
                meta.append(("world", self.geom_links[idx], obs_id))

        if attached:
            tcp_frame = self.model.frames[self.tcp_frame_id]
            tcp_pj = int(tcp_frame.parentJoint)
            tcp_place = tcp_frame.placement
            for att_idx, (att_geom, R_loc, p_loc) in enumerate(attached):
                att_place = tcp_place * pin.SE3(R_loc, p_loc)
                att_gid = int(
                    gm.addGeometryObject(
                        pin.GeometryObject(f"__att_{att_idx}", tcp_pj, self.tcp_frame_id, att_place, att_geom)
                    )
                )
                if table_gid is not None:
                    gm.addCollisionPair(pin.CollisionPair(att_gid, table_gid))
                    meta.append(("world", ATTACHED_LINK, "table"))
                for gid, obs_id in obs_gids:
                    gm.addCollisionPair(pin.CollisionPair(att_gid, gid))
                    meta.append(("world", ATTACHED_LINK, obs_id))
                for idx in self.non_gripper_indices:
                    gm.addCollisionPair(pin.CollisionPair(att_gid, idx))
                    la = self.geom_links[idx]
                    meta.append(("att_self", ATTACHED_LINK if ATTACHED_LINK < la else la, la if ATTACHED_LINK < la else ATTACHED_LINK))

        compiled = (gm, tuple(meta))
        if not obstacles and not attached and len(self._table_gm_cache) < 16:
            self._table_gm_cache[tz_key] = compiled
        return compiled

    @staticmethod
    def _compile_world_obstacles(world: Mapping[str, Any] | None) -> list[tuple[str, coal.CollisionGeometry, np.ndarray]]:
        if not world:
            return []
        compiled: list[tuple[str, coal.CollisionGeometry, np.ndarray]] = []
        for item in world.get("obstacles") or ():
            obs_id = str(item["id"])
            if "min_m" in item and "max_m" in item:
                lo = np.asarray(item["min_m"], dtype=np.float64).reshape(3)
                hi = np.asarray(item["max_m"], dtype=np.float64).reshape(3)
                dims = hi - lo
                if np.any(dims <= 0.0) or not np.isfinite(dims).all():
                    raise ValueError(f"invalid_obstacle_bounds:{obs_id}")
                ctr = 0.5 * (lo + hi)
                compiled.append((obs_id, coal.Box(float(dims[0]), float(dims[1]), float(dims[2])), ctr))
            elif "centre_m" in item and "radius_m" in item:
                ctr = np.asarray(item["centre_m"], dtype=np.float64).reshape(3)
                rad = float(item["radius_m"])
                if not (math.isfinite(rad) and rad > 0.0) or not np.isfinite(ctr).all():
                    raise ValueError(f"invalid_obstacle_sphere:{obs_id}")
                compiled.append((obs_id, coal.Sphere(rad), ctr))
            else:
                raise ValueError(f"unsupported_obstacle_spec:{obs_id}")
        return compiled

    @staticmethod
    def _compile_attached_geometries(
        world: Mapping[str, Any] | None,
    ) -> list[tuple[coal.CollisionGeometry, np.ndarray, np.ndarray]]:
        """Return ``[(geom, R_tcp_obj, p_tcp_obj)]`` in ``gripper_frame_link`` coordinates."""
        if not world:
            return []
        if world.get("attached_box") is not None:
            box = world["attached_box"]
            ctr = np.asarray(box["centre_tcp_m"], dtype=np.float64).reshape(3)
            half = np.asarray(box["half_extents_m"], dtype=np.float64).reshape(3)
            rot = (
                np.asarray(box["rotation_tcp"], dtype=np.float64).reshape(3, 3)
                if box.get("rotation_tcp") is not None
                else np.eye(3, dtype=np.float64)
            )
            dims = np.maximum(2.0 * half, 1e-5)
            return [(coal.Box(float(dims[0]), float(dims[1]), float(dims[2])), rot, ctr)]
        out: list[tuple[coal.CollisionGeometry, np.ndarray, np.ndarray]] = []
        eye3 = np.eye(3, dtype=np.float64)
        for sph in world.get("attached_spheres") or ():
            vals = [float(v) for v in sph]
            if len(vals) != 4 or vals[3] <= 0.0 or not all(math.isfinite(v) for v in vals):
                continue
            out.append((coal.Sphere(vals[3]), eye3, np.array(vals[:3], dtype=np.float64)))
        return out

    def _evaluate_sample_on_gm(
        self,
        data: pin.Data,
        gm: pin.GeometryModel,
        gd: pin.GeometryData,
        meta: tuple[tuple[str, Any, Any], ...],
        q_arm: np.ndarray,
        jaw_val: float,
        dist_req: coal.DistanceRequest,
    ) -> dict[str, Any]:
        q_full = np.zeros(self.model.nq, dtype=np.float64)
        q_full[self.arm_idx_q] = q_arm
        q_full[self.jaw_idx_q] = jaw_val

        has_any = bool(pin.computeCollisions(self.model, data, gm, gd, q_full, False))
        if not has_any:
            return {
                "self": [],
                "world": [],
                "self_and_table_free": True,
                "obstacle_free": True,
            }

        geom_objs = gm.geometryObjects
        oMg = gd.oMg
        col_results = gd.collisionResults
        pairs = gm.collisionPairs

        self_deepest: dict[tuple[str, str], float] = {}
        world_by_link: dict[str, dict[str, Any]] = {}
        robot_self_free = True
        robot_table_free = True

        for cp_idx, (kind, first_id, second_id) in enumerate(meta):
            if not col_results[cp_idx].isCollision():
                continue
            cp = pairs[cp_idx]
            i, j = int(cp.first), int(cp.second)
            tf_i = coal.Transform3s(oMg[i].rotation, oMg[i].translation)
            tf_j = coal.Transform3s(oMg[j].rotation, oMg[j].translation)
            dist_res = coal.DistanceResult()
            dist = float(coal.distance(geom_objs[i].geometry, tf_i, geom_objs[j].geometry, tf_j, dist_req, dist_res))
            pen = round(max(-dist, 1e-5), 5) if dist < 0.0 else 1e-5

            if kind == "self":
                robot_self_free = False
                pair_key = (str(first_id), str(second_id))
                if pen > self_deepest.get(pair_key, 0.0):
                    self_deepest[pair_key] = pen
            elif kind == "att_self":
                pair_key = (str(first_id), str(second_id))
                if pen > self_deepest.get(pair_key, 0.0):
                    self_deepest[pair_key] = pen
            else:
                if kind == "table":
                    robot_table_free = False
                link_name = str(first_id)
                obs_name = str(second_id)
                entry = world_by_link.setdefault(link_name, {"link": link_name, "penetration_m": 0.0, "obstacles": []})
                if pen > entry["penetration_m"]:
                    entry["penetration_m"] = pen
                if obs_name not in entry["obstacles"]:
                    entry["obstacles"].append(obs_name)

        self_list = [
            {"links": [pair[0], pair[1]], "penetration_m": round(float(depth), 5)}
            for pair, depth in sorted(self_deepest.items(), key=lambda item: -item[1])[:MAX_REPORTED_COLLISIONS]
        ]
        world_list = [
            {
                "link": item["link"],
                "penetration_m": round(float(item["penetration_m"]), 5),
                "obstacles": list(item["obstacles"]),
            }
            for item in sorted(world_by_link.values(), key=lambda entry: -entry["penetration_m"])[:MAX_REPORTED_COLLISIONS]
        ]
        self_and_table_free = bool(robot_self_free and robot_table_free)
        obstacle_free = bool(self_and_table_free and not self_list and not world_list)
        return {
            "self": self_list,
            "world": world_list,
            "self_and_table_free": self_and_table_free,
            "obstacle_free": obstacle_free,
        }

    def check_state(
        self,
        q_arm_rad: Sequence[float] | np.ndarray,
        jaw_rad: float = 0.0,
        *,
        table_top_z_m: float | None = None,
        world: dict[str, Any] | None = None,
        check_world_obstacles: bool = True,
    ) -> dict[str, Any]:
        """Check one SO-101 arm + jaw configuration against self, support table, and optional scene/held obstacles.

        Parameters:
            q_arm_rad: Five finite arm joint angles in ``ARM_JOINTS`` order (``rad``).
            jaw_rad: URDF ``gripper`` joint angle in ``rad`` (default ``0.0``).
            table_top_z_m: Optional support-plane height ``z`` in ``base_link`` (``m``); ``None`` skips table checks.
            world: Optional collision world dict ``{"obstacles": [...], "attached_box": ..., "attached_spheres": [...]}``.
            check_world_obstacles: When ``True`` (default) and ``world`` is supplied, evaluates scene ``obstacles``
                and attached held-object geometry in addition to self-collision and ``table_top_z_m``.

        Returns:
            ``{"self": [{"links": [a, b], "penetration_m": float}], "world": [{"link": str, "penetration_m": float, "obstacles": [str]}], "self_and_table_free": bool, "obstacle_free": bool}``.

        Example:
            ``verdict = engine.check_state([0.0, -0.95, 0.9, 1.0, 0.0], jaw_rad=1.2, table_top_z_m=0.0)``
        """
        q_arm = _arm_vector(q_arm_rad)
        jaw_val = float(jaw_rad)
        if not math.isfinite(jaw_val):
            raise ValueError("so101_jaw_rad_must_be_finite")
        tz_val: float | None = None
        if table_top_z_m is not None:
            tz_val = float(table_top_z_m)
            if not math.isfinite(tz_val):
                raise ValueError("invalid_table_top_z_m")

        obs_list = self._compile_world_obstacles(world) if (check_world_obstacles and world) else []
        att_list = self._compile_attached_geometries(world) if (check_world_obstacles and world) else []
        gm, meta = self._build_query_geom_model(tz_val, obs_list, att_list)
        data = self.model.createData()
        gd = gm.createData()
        return self._evaluate_sample_on_gm(data, gm, gd, meta, q_arm, jaw_val, coal.DistanceRequest(True))

    def check_path(
        self,
        segments: list[dict[str, Any]],
        *,
        table_top_z_m: float | None = None,
        world: dict[str, Any] | None = None,
        max_joint_step_rad: float = 0.01,
    ) -> dict[str, Any]:
        """Sweep straight joint-space segments at ``<= max_joint_step_rad`` resolution (100% ``CuroboClient.check_path`` contract).

        Parameters:
            segments: List of segment dicts ``{"start_q_rad": [5], "end_q_rad": [5], "start_jaw_rad": float, "end_jaw_rad": float, "attached": bool}`` in ``rad``.
            table_top_z_m: Optional support-plane height ``z`` in ``base_link`` (``m``).
            world: Optional collision world dict ``{"obstacles": [...], "attached_box": ..., "attached_spheres": [...]}``.
            max_joint_step_rad: Maximum joint change between adjacent samples in ``rad`` (default ``0.01``).

        Returns:
            ``{"check_ms": float, "samples": int, "segments": [{"samples": int, "colliding_samples": int, "first_collision": None | dict}]}``.

        Example:
            ``report = engine.check_path([{"start_q_rad": [0]*5, "end_q_rad": [0.2]*5}], table_top_z_m=0.0)``
        """
        step = float(max_joint_step_rad)
        if not (math.isfinite(step) and step > 0.0):
            raise ValueError("invalid_max_joint_step_rad")
        if not segments:
            return {"check_ms": 0.0, "samples": 0, "segments": []}

        started = time.perf_counter()
        tz_val: float | None = None
        if table_top_z_m is not None:
            tz_val = float(table_top_z_m)
            if not math.isfinite(tz_val):
                raise ValueError("invalid_table_top_z_m")

        obs_list = self._compile_world_obstacles(world)
        att_list = self._compile_attached_geometries(world)
        gm_att, meta_att = self._build_query_geom_model(tz_val, obs_list, att_list)
        gd_att = gm_att.createData()
        if att_list and any(not bool(seg.get("attached", True)) for seg in segments):
            gm_unatt, meta_unatt = self._build_query_geom_model(tz_val, obs_list, [])
            gd_unatt = gm_unatt.createData()
        else:
            gm_unatt, meta_unatt, gd_unatt = gm_att, meta_att, gd_att

        data = self.model.createData()
        dist_req = coal.DistanceRequest(True)

        total_samples = 0
        out_segments: list[dict[str, Any]] = []
        for seg in segments:
            q0 = _arm_vector(seg["start_q_rad"])
            q1 = _arm_vector(seg["end_q_rad"])
            j0 = float(seg.get("start_jaw_rad", 0.0))
            j1 = float(seg.get("end_jaw_rad", 0.0))
            if not (math.isfinite(j0) and math.isfinite(j1)):
                raise ValueError("so101_jaw_rad_must_be_finite")
            use_attached = bool(seg.get("attached", True))
            gm_cur, meta_cur, gd_cur = (gm_att, meta_att, gd_att) if use_attached else (gm_unatt, meta_unatt, gd_unatt)
            max_change = max(float(np.max(np.abs(q1 - q0))), abs(j1 - j0))
            n_samples = 1 if max_change == 0.0 else 1 + max(1, int(math.ceil(max_change / step - 1e-9)))
            total_samples += n_samples
            colliding_samples = 0
            first_collision: dict[str, Any] | None = None

            for s_idx in range(n_samples):
                frac = (s_idx / (n_samples - 1)) if n_samples > 1 else 0.0
                q_s = q1.copy() if s_idx == n_samples - 1 and n_samples > 1 else (q0 + frac * (q1 - q0))
                j_s = j1 if s_idx == n_samples - 1 and n_samples > 1 else (j0 + frac * (j1 - j0))
                state = self._evaluate_sample_on_gm(data, gm_cur, gd_cur, meta_cur, q_s, j_s, dist_req)
                if not state["obstacle_free"]:
                    colliding_samples += 1
                    if first_collision is None:
                        first_collision = {
                            "sample_index": int(s_idx),
                            "fraction": round(float(frac), 4),
                            "q_rad": [float(v) for v in q_s],
                            "jaw_rad": float(j_s),
                            "world": state["world"],
                            "self": state["self"],
                        }

            out_segments.append(
                {
                    "samples": int(n_samples),
                    "colliding_samples": int(colliding_samples),
                    "first_collision": first_collision,
                }
            )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "check_ms": round(float(elapsed_ms), 3),
            "samples": int(total_samples),
            "segments": out_segments,
        }


@functools.lru_cache(maxsize=4)
def _cached_collision_engine(resolved_archive: str | None) -> SO101CoalCollisionEngine:
    return SO101CoalCollisionEngine(resolved_archive)


def so101_collision_engine(archive_path: str | Path | None = None) -> SO101CoalCollisionEngine:
    """Return the process-cached ``SO101CoalCollisionEngine`` instance for ``archive_path`` (or packaged SO-101)."""
    key = None if archive_path is None else str(Path(archive_path).resolve())
    return _cached_collision_engine(key)


__all__ = [
    "ARM_JOINTS",
    "ATTACHED_LINK",
    "BASE_LINK",
    "GRAVITY_M_S2",
    "GRIPPER_JOINT",
    "SO101CoalCollisionEngine",
    "TCP_LINK",
    "pinocchio_gravity_torques",
    "so101_collision_engine",
    "solve_so101_pink_ik",
]
