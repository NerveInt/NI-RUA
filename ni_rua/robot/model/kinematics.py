"""Generic N-DoF serial/tree robot forward kinematics, analytical Jacobians, and inverse kinematics.

Module import time is 100% pure Python stdlib + NumPy (zero native C++ imports at import time),
preserving zero-dependency compatibility for isolated workers that import
``KinematicChainSpec`` / ``chain_from_manifest``. Runtime kinematics and IK calls lazily invoke
``Pinocchio`` + ``Pink`` + ``DAQP`` (with exact analytical-Jacobian Levenberg-Marquardt / pure-NumPy
fallbacks when native bindings or QP solvers are unavailable) and ``scipy.spatial.transform.Rotation``.

Physical conventions & units:
- Translations and link lengths: metres (``m``).
- Joint angles (revolute/continuous), orientation errors, trust regions, and step caps: radians (``rad``).
- Prismatic joint positions and limits: metres (``m``).
- Rigid transforms: ``4x4`` homogeneous ``SE(3)`` matrices ``[[R, p], [0, 0, 0, 1]]`` with ``R in SO(3)``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import functools
import math
from typing import Any

import numpy as np

_INTERNAL_TIP_FRAME = "__ni_rua_tip__"


def _finite_array(value: Any, shape: tuple[int, ...] | None = None, *, error_code: str = "invalid_finite_array") -> np.ndarray:
    """Convert ``value`` to a finite ``float64`` NumPy array, optionally enforcing ``shape``."""
    arr = np.asarray(value, dtype=np.float64)
    if (shape is not None and arr.shape != shape) or not np.isfinite(arr).all():
        raise ValueError(error_code)
    return arr


def rigid_transform(value: Any) -> np.ndarray:
    """Validate and return a ``4x4`` rigid homogeneous transform in ``SE(3)`` (translation in ``m``).

    Raises ``ValueError("invalid_finite_array")`` or ``ValueError("invalid_rigid_transform")`` if
    non-finite, non-orthogonal, reflection (``det != +1``), or invalid homogeneous bottom row.
    """
    matrix = _finite_array(value, (4, 4))
    rot = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7)
        or not np.allclose(rot.T @ rot, np.eye(3), atol=1e-5)
        or abs(float(np.linalg.det(rot)) - 1.0) > 1e-5
    ):
        raise ValueError("invalid_rigid_transform")
    return matrix


def rpy_xyz_to_matrix(xyz_m: Sequence[float], rpy_rad: Sequence[float]) -> np.ndarray:
    """Convert URDF joint origin ``(xyz_m, rpy_rad)`` (``m``, ``rad``) to a ``4x4`` homogeneous transform."""
    from scipy.spatial.transform import Rotation

    rpy = _finite_array(rpy_rad, (3,))
    xyz = _finite_array(xyz_m, (3,))
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = Rotation.from_euler("xyz", rpy, degrees=False).as_matrix()
    mat[:3, 3] = xyz
    return mat


def quat_xyzw_to_matrix(quat_xyzw: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert a non-zero quaternion ``[qx, qy, qz, qw]`` (dimensionless) to a ``3x3`` rotation matrix."""
    from scipy.spatial.transform import Rotation

    q = _finite_array(quat_xyzw, (4,), error_code="invalid_target_quaternion")
    if float(np.linalg.norm(q)) <= 1e-12:
        raise ValueError("invalid_target_quaternion")
    return np.asarray(Rotation.from_quat(q, scalar_first=False).as_matrix(), dtype=np.float64)


def quat_wxyz_to_matrix(quat_wxyz: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert a non-zero quaternion ``[qw, qx, qy, qz]`` (dimensionless) to a ``3x3`` rotation matrix."""
    from scipy.spatial.transform import Rotation

    q = _finite_array(quat_wxyz, (4,), error_code="invalid_target_quaternion")
    if float(np.linalg.norm(q)) <= 1e-12:
        raise ValueError("invalid_target_quaternion")
    return np.asarray(Rotation.from_quat(q, scalar_first=True).as_matrix(), dtype=np.float64)


def rodrigues(axis: Sequence[float] | np.ndarray, angle_rad: float) -> np.ndarray:
    """Compute ``3x3`` rotation matrix ``exp([unit_axis * angle_rad]_x)`` for ``angle_rad`` in ``rad``."""
    from scipy.spatial.transform import Rotation

    ax = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(ax))
    if norm < 1e-9 or not math.isfinite(angle_rad):
        raise ValueError("invalid_joint_state_or_axis")
    return np.asarray(Rotation.from_rotvec((ax / norm) * float(angle_rad)).as_matrix(), dtype=np.float64)


def rotation_log(matrix: np.ndarray) -> np.ndarray:
    """Compute the ``SO(3)`` logarithmic map ``R -> axis * angle`` (shape ``(3,)``, unit ``rad``)."""
    from scipy.spatial.transform import Rotation

    rot = _finite_array(matrix, (3, 3))
    return np.asarray(Rotation.from_matrix(rot).as_rotvec(), dtype=np.float64)


def rotation_log_batch(rot_cur: np.ndarray, rot_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched rotation error ``log(rot_target @ rot_cur^T)`` for ``rot_cur`` ``(B, 3, 3)`` and ``rot_target`` ``(3, 3)``.

    Returns ``(err_vec_rad, angle_rad)`` with shapes ``(B, 3)`` and ``(B,)`` in ``rad``.
    """
    from scipy.spatial.transform import Rotation

    cur = np.asarray(rot_cur, dtype=np.float64)
    tgt = np.asarray(rot_target, dtype=np.float64).reshape(3, 3)
    if cur.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float64)
    rot_err = tgt[None, :, :] @ cur.swapaxes(1, 2)
    err_vec = np.asarray(Rotation.from_matrix(rot_err).as_rotvec(), dtype=np.float64).reshape(cur.shape[0], 3)
    theta = np.linalg.norm(err_vec, axis=1)
    return err_vec, theta


def pose_delta_vectors(current_T: np.ndarray, target_T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(dp_m, dr_rad)`` 3-vectors between two ``4x4`` poses in the same reference frame (``m``, ``rad``)."""
    cur = np.asarray(current_T, dtype=np.float64).reshape(4, 4)
    tgt = np.asarray(target_T, dtype=np.float64).reshape(4, 4)
    dp = tgt[:3, 3] - cur[:3, 3]
    dr = rotation_log(tgt[:3, :3] @ cur[:3, :3].T)
    return dp, dr


def pose_error(current_T: np.ndarray, target_T: np.ndarray) -> tuple[float, float]:
    """Return ``(position_error_m, orientation_error_rad)`` between two ``4x4`` poses (``m``, ``rad``).

    Example:
        ``pos_err_m, ori_err_rad = pose_error(T_current, T_target)``
    """
    dp, dr = pose_delta_vectors(current_T, target_T)
    return float(np.linalg.norm(dp)), float(np.linalg.norm(dr))


def _pin_joint_model(pin: Any, j_type: str, unit_axis: np.ndarray) -> Any:
    if j_type in ("revolute", "continuous"):
        if np.allclose(unit_axis, [1.0, 0.0, 0.0], atol=1e-12):
            return pin.JointModelRX()
        if np.allclose(unit_axis, [0.0, 1.0, 0.0], atol=1e-12):
            return pin.JointModelRY()
        if np.allclose(unit_axis, [0.0, 0.0, 1.0], atol=1e-12):
            return pin.JointModelRZ()
        return pin.JointModelRevoluteUnaligned(unit_axis)
    if j_type == "prismatic":
        if np.allclose(unit_axis, [1.0, 0.0, 0.0], atol=1e-12):
            return pin.JointModelPX()
        if np.allclose(unit_axis, [0.0, 1.0, 0.0], atol=1e-12):
            return pin.JointModelPY()
        if np.allclose(unit_axis, [0.0, 0.0, 1.0], atol=1e-12):
            return pin.JointModelPZ()
        return pin.JointModelPrismaticUnaligned(unit_axis)
    raise ValueError("unsupported_joint_type")


@functools.lru_cache(maxsize=32)
def _cached_manifest_pin_model(
    base_link: str,
    joint_specs: tuple[tuple[str, str, str, str, bytes, bytes | None], ...],
) -> tuple[Any, dict[str, int], list[tuple[int, str, tuple[str, float, float] | None]]]:
    import pinocchio as pin

    p_model = pin.Model()
    base_fid = int(p_model.addFrame(pin.Frame(base_link, 0, 0, pin.SE3.Identity(), pin.FrameType.BODY)))
    link_frame_ids: dict[str, int] = {base_link: base_fid}
    link_parent_joint: dict[str, int] = {base_link: 0}
    link_rel_placement: dict[str, Any] = {base_link: pin.SE3.Identity()}
    active_q_entries: list[tuple[int, str, tuple[str, float, float] | None]] = []

    for j_name, kind, parent, child, origin_bytes, axis_bytes in joint_specs:
        if parent not in link_frame_ids:
            raise KeyError(parent)
        p_fid = link_frame_ids[parent]
        p_jid = link_parent_joint[parent]
        rel_p = link_rel_placement[parent]
        origin = np.frombuffer(origin_bytes, dtype=np.float64).reshape(4, 4)
        placement = rel_p * pin.SE3(origin[:3, :3], origin[:3, 3])
        if kind == "fixed":
            c_fid = int(p_model.addFrame(pin.Frame(child, p_jid, p_fid, placement, pin.FrameType.BODY)))
            link_frame_ids[child] = c_fid
            link_parent_joint[child] = p_jid
            link_rel_placement[child] = placement
        else:
            assert axis_bytes is not None
            axis = np.frombuffer(axis_bytes, dtype=np.float64).reshape(3)
            jm = _pin_joint_model(pin, kind, axis)
            jid = int(p_model.addJoint(p_jid, jm, placement, j_name))
            c_fid = int(p_model.addFrame(pin.Frame(child, jid, p_fid, pin.SE3.Identity(), pin.FrameType.BODY)))
            link_frame_ids[child] = c_fid
            link_parent_joint[child] = jid
            link_rel_placement[child] = pin.SE3.Identity()
            active_q_entries.append((int(p_model.joints[jid].idx_q), j_name, None))

    return p_model, link_frame_ids, active_q_entries


def fk_manifest(
    model: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    T_world_base: np.ndarray | None = None,
    require_source_kind: bool = True,
) -> dict[str, np.ndarray]:
    """Evaluate forward kinematics for every link in a disclosed URDF manifest dictionary.

    Uses C++ ``Pinocchio`` when installed, with an exact pure-NumPy tree-walk fallback for isolated
    simulator workers without ``pinocchio``. Supports ``fixed``, ``revolute``, ``continuous``,
    ``prismatic``, and ``mimic`` joints. Returns ``{link_name: 4x4 T_world_link}`` with translations in ``m``.
    """
    if require_source_kind:
        joints = state.get("joints", {}) if isinstance(state, Mapping) else {}
        names = joints.get("names", state.get("joint_names", []))
        values = joints.get("position_rad", joints.get("q", state.get("q_rad", state.get("joint_positions_rad", []))))
        source = joints.get("source_kind", state.get("source_kind"))
        if source not in ("measurement", "estimate") or len(names) != len(values) or len(set(names)) != len(names):
            raise ValueError("fk_requires_measured_or_estimated_named_joints")
        positions: dict[str, float] = {str(k): float(v) for k, v in zip(names, values)}
    else:
        positions = {str(k): float(v) for k, v in state.items()}

    base_link = str(model.get("base_link", "base"))
    base_tf = rigid_transform(T_world_base if T_world_base is not None else model.get("T_world_base", np.eye(4)))

    try:
        import pinocchio as pin
    except ImportError:
        pin = None

    if pin is None:
        poses: dict[str, np.ndarray] = {base_link: base_tf.copy()}
        for joint in model["joints"]:
            kind = str(joint["type"])
            parent = str(joint["parent"])
            child = str(joint["child"])
            origin = np.ascontiguousarray(rigid_transform(joint["T_parent_joint"]), dtype=np.float64)
            parent_pose = poses[parent]
            if kind == "fixed":
                poses[child] = parent_pose @ origin
                continue
            if kind not in ("revolute", "continuous", "prismatic"):
                raise ValueError("unsupported_joint_type")
            mimic = joint.get("mimic")
            if mimic:
                base_val = positions[mimic["joint"]] if require_source_kind else positions.get(mimic["joint"], 0.0)
                val = float(base_val) * float(mimic["multiplier"]) + float(mimic["offset"])
            else:
                val = float(positions[joint["name"]] if require_source_kind else positions.get(joint["name"], 0.0))
            axis = _finite_array(joint["axis"], (3,))
            norm = float(np.linalg.norm(axis))
            if norm < 1e-9 or not math.isfinite(val):
                raise ValueError("invalid_joint_state_or_axis")
            unit_axis = axis / norm
            motion = np.eye(4, dtype=np.float64)
            if kind in ("revolute", "continuous"):
                motion[:3, :3] = rodrigues(unit_axis, val)
            else:
                motion[:3, 3] = unit_axis * val
            poses[child] = parent_pose @ origin @ motion
        return poses

    joint_specs: list[tuple[str, str, str, str, bytes, bytes | None]] = []
    q_values_by_joint: dict[str, float] = {}
    for idx_j, joint in enumerate(model["joints"]):
        j_name = str(joint.get("name", f"joint_{idx_j}"))
        kind = str(joint["type"])
        parent = str(joint["parent"])
        child = str(joint["child"])
        origin = np.ascontiguousarray(rigid_transform(joint["T_parent_joint"]), dtype=np.float64)
        if kind == "fixed":
            joint_specs.append((j_name, "fixed", parent, child, origin.tobytes(), None))
            continue
        if kind not in ("revolute", "continuous", "prismatic"):
            raise ValueError("unsupported_joint_type")
        mimic = joint.get("mimic")
        if mimic:
            base_val = positions[mimic["joint"]] if require_source_kind else positions.get(mimic["joint"], 0.0)
            val = float(base_val) * float(mimic["multiplier"]) + float(mimic["offset"])
        else:
            val = float(positions[joint["name"]] if require_source_kind else positions.get(joint["name"], 0.0))
        axis = _finite_array(joint["axis"], (3,))
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9 or not math.isfinite(val):
            raise ValueError("invalid_joint_state_or_axis")
        unit_axis = np.ascontiguousarray(axis / norm, dtype=np.float64)
        unique_j_name = f"{idx_j}:{j_name}"
        joint_specs.append((unique_j_name, kind, parent, child, origin.tobytes(), unit_axis.tobytes()))
        q_values_by_joint[unique_j_name] = val

    p_model, link_frame_ids, active_q_entries = _cached_manifest_pin_model(base_link, tuple(joint_specs))
    p_data = p_model.createData()
    q_vec = np.zeros(p_model.nq, dtype=np.float64)
    for idx_q, unique_j_name, _ in active_q_entries:
        q_vec[idx_q] = q_values_by_joint[unique_j_name]
    pin.framesForwardKinematics(p_model, p_data, q_vec)

    is_identity_base = bool(np.allclose(base_tf, np.eye(4), atol=1e-15))
    poses = {}
    for link_name, fid in link_frame_ids.items():
        mat = np.asarray(p_data.oMf[fid].homogeneous, dtype=np.float64)
        poses[link_name] = mat.copy() if is_identity_base else (base_tf @ mat)
    return poses


@dataclass(frozen=True)
class KinematicChainSpec:
    """Immutable N-DoF serial kinematic chain specification.

    Attributes:
        joint_names: Tuple of ``N`` active joint names in proximal-to-distal order.
        joint_types: Tuple of ``N`` joint motion types (``'revolute'``, ``'continuous'``, or ``'prismatic'``).
        parent_transforms: Array of shape ``(N, 4, 4)`` giving each active joint's fixed origin transform
            relative to the preceding active joint frame (or ``base_link`` for index 0), with translations in ``m``.
        joint_axes: Array of shape ``(N, 3)`` of unit local motion axes.
        joint_limits_rad: Array of shape ``(N, 2)`` of ``[lower, upper]`` joint bounds (``rad`` for rotary, ``m`` for prismatic).
        tip_transform: ``4x4`` fixed transform from the last active joint frame to ``tip_link`` (``m``).
        base_link: Name of the chain root link.
        tip_link: Name of the end-effector / TCP link.
        link_names: Optional tuple of ``N`` child link names corresponding to each active joint.
        T_world_base: Optional ``4x4`` world-to-base transform (``m``).
        rest_posture_rad: Optional ``(N,)`` nominal joint posture (``rad``) for null-space redundancy resolution.
    """

    joint_names: tuple[str, ...]
    joint_types: tuple[str, ...]
    parent_transforms: np.ndarray
    joint_axes: np.ndarray
    joint_limits_rad: np.ndarray
    tip_transform: np.ndarray
    base_link: str = "base_link"
    tip_link: str = "tcp"
    link_names: tuple[str, ...] = ()
    T_world_base: np.ndarray | None = None
    rest_posture_rad: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = len(self.joint_names)
        if n < 1:
            raise ValueError("kinematic_chain_requires_at_least_one_active_joint")
        if len(self.joint_types) != n or any(t not in ("revolute", "continuous", "prismatic") for t in self.joint_types):
            raise ValueError("invalid_kinematic_chain_joint_types")
        p_tf = _finite_array(self.parent_transforms, (n, 4, 4), error_code="invalid_parent_transforms")
        axes = _finite_array(self.joint_axes, (n, 3), error_code="invalid_joint_axes")
        norms = np.linalg.norm(axes, axis=1)
        if np.any(norms < 1e-9):
            raise ValueError("zero_joint_axis_in_chain")
        axes = axes / norms[:, None]
        lims = _finite_array(self.joint_limits_rad, (n, 2), error_code="invalid_joint_limits_rad")
        if np.any(lims[:, 0] > lims[:, 1]):
            raise ValueError("invalid_joint_limits_rad")
        tip_tf = _finite_array(self.tip_transform, (4, 4), error_code="invalid_tip_transform")
        links = tuple(self.link_names) if self.link_names else tuple(f"link{i + 1}" for i in range(n))
        if len(links) != n:
            raise ValueError("invalid_chain_link_names")
        object.__setattr__(self, "parent_transforms", p_tf)
        object.__setattr__(self, "joint_axes", axes)
        object.__setattr__(self, "joint_limits_rad", lims)
        object.__setattr__(self, "tip_transform", tip_tf)
        object.__setattr__(self, "link_names", links)
        rotary_mask = np.array([t in ("revolute", "continuous") for t in self.joint_types], dtype=bool)
        skews = np.zeros((n, 3, 3), dtype=np.float64)
        skews[:, 0, 1] = -axes[:, 2]
        skews[:, 0, 2] = axes[:, 1]
        skews[:, 1, 0] = axes[:, 2]
        skews[:, 1, 2] = -axes[:, 0]
        skews[:, 2, 0] = -axes[:, 1]
        skews[:, 2, 1] = axes[:, 0]
        object.__setattr__(self, "_rotary_mask", rotary_mask)
        object.__setattr__(self, "_all_rotary", bool(np.all(rotary_mask)))
        object.__setattr__(self, "_axis_skews", skews)
        object.__setattr__(self, "_axis_sq", skews @ skews)
        object.__setattr__(self, "_has_tip_offset", bool(np.any(np.abs(tip_tf - np.eye(4)) > 1e-12)))
        if self.T_world_base is not None:
            object.__setattr__(self, "T_world_base", _finite_array(self.T_world_base, (4, 4)))
        if self.rest_posture_rad is not None:
            object.__setattr__(self, "rest_posture_rad", _finite_array(self.rest_posture_rad, (n,)))

    @property
    def dof(self) -> int:
        """Number of active degrees of freedom ``N``."""
        return len(self.joint_names)

    @property
    def parent_rots(self) -> np.ndarray:
        """Shape ``(N, 3, 3)`` fixed parent-to-joint rotations."""
        return self.parent_transforms[:, :3, :3]

    @property
    def parent_pos_m(self) -> np.ndarray:
        """Shape ``(N, 3)`` fixed parent-to-joint translations in ``m``."""
        return self.parent_transforms[:, :3, 3]

    @property
    def max_reach_m(self) -> float:
        """Conservative geometric reach upper bound from joint 1 (shoulder) to ``tip_link`` in ``m``."""
        tip_len = float(np.linalg.norm(self.tip_transform[:3, 3]))
        body_len = float(np.sum(np.linalg.norm(self.parent_transforms[1:, :3, 3], axis=1))) if self.dof > 1 else 0.0
        return body_len + tip_len + 0.05

    def with_limits(self, limits: Mapping[str, Sequence[float]] | np.ndarray | Sequence[Sequence[float]]) -> KinematicChainSpec:
        """Return a copy of this ``KinematicChainSpec`` with updated joint bounds."""
        if isinstance(limits, Mapping):
            bounds = self.joint_limits_rad.copy()
            index_by_name = {name: i for i, name in enumerate(self.joint_names)}
            for name, pair in limits.items():
                if name not in index_by_name:
                    raise ValueError(f"unknown_limit_joint:{name}")
                lo, hi = float(pair[0]), float(pair[1])
                if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
                    raise ValueError(f"invalid_joint_limits:{name}")
                bounds[index_by_name[name]] = [lo, hi]
        else:
            bounds = _finite_array(limits, (self.dof, 2), error_code="invalid_joint_limits_rad").copy()
            if np.any(bounds[:, 0] > bounds[:, 1]):
                raise ValueError("invalid_joint_limits_rad")
        return KinematicChainSpec(
            joint_names=self.joint_names,
            joint_types=self.joint_types,
            parent_transforms=self.parent_transforms,
            joint_axes=self.joint_axes,
            joint_limits_rad=bounds,
            tip_transform=self.tip_transform,
            base_link=self.base_link,
            tip_link=self.tip_link,
            link_names=self.link_names,
            T_world_base=self.T_world_base,
            rest_posture_rad=self.rest_posture_rad,
        )


def _pin_bundle(chain: KinematicChainSpec) -> tuple[Any, tuple[int, ...], int]:
    """Build or retrieve cached ``(pin.Model, link_frame_ids, tip_frame_id)`` on ``chain``."""
    cached = getattr(chain, "_pin_cache", None)
    if cached is not None:
        return cached
    import pinocchio as pin

    p_model = pin.Model()
    p_jid = 0
    p_fid = int(p_model.addFrame(pin.Frame(chain.base_link, 0, 0, pin.SE3.Identity(), pin.FrameType.BODY)))
    link_fids: list[int] = []
    for idx in range(chain.dof):
        tf = chain.parent_transforms[idx]
        placement = pin.SE3(tf[:3, :3], tf[:3, 3])
        jm = _pin_joint_model(pin, chain.joint_types[idx], chain.joint_axes[idx])
        jid = int(p_model.addJoint(p_jid, jm, placement, chain.joint_names[idx]))
        l_fid = int(p_model.addFrame(pin.Frame(chain.link_names[idx], jid, p_fid, pin.SE3.Identity(), pin.FrameType.BODY)))
        link_fids.append(l_fid)
        p_jid = jid
        p_fid = l_fid

    tip_tf = chain.tip_transform
    tip_se3 = pin.SE3(tip_tf[:3, :3], tip_tf[:3, 3])
    tip_fid = int(p_model.addFrame(pin.Frame(_INTERNAL_TIP_FRAME, p_jid, p_fid, tip_se3, pin.FrameType.OP_FRAME)))
    p_model.lowerPositionLimit[:] = chain.joint_limits_rad[:, 0]
    p_model.upperPositionLimit[:] = chain.joint_limits_rad[:, 1]
    bundle = (p_model, tuple(link_fids), tip_fid)
    object.__setattr__(chain, "_pin_cache", bundle)
    return bundle


def _pure_rpy_xyz_to_matrix(xyz_m: Sequence[float], rpy_rad: Sequence[float]) -> np.ndarray:
    """Zero-dependency URDF origin ``(xyz_m, rpy_rad)`` helper so ``chain_from_manifest`` needs only stdlib + NumPy."""
    r, p, y = (float(v) for v in rpy_rad)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    mat[:3, 3] = [float(v) for v in xyz_m]
    return mat


def _extract_origin_4x4(joint: Mapping[str, Any]) -> np.ndarray:
    if "T_parent_joint" in joint:
        return _finite_array(joint["T_parent_joint"], (4, 4), error_code="invalid_T_parent_joint_in_robot_model").copy()
    origin = joint.get("origin")
    if isinstance(origin, Mapping):
        xyz = tuple(float(v) for v in origin.get("xyz_m", origin.get("xyz", (0.0, 0.0, 0.0))))
        rpy = tuple(float(v) for v in origin.get("rpy_rad", origin.get("rpy", (0.0, 0.0, 0.0))))
        return _pure_rpy_xyz_to_matrix(xyz, rpy)
    xyz = tuple(float(v) for v in joint.get("xyz", joint.get("xyz_m", (0.0, 0.0, 0.0))))
    rpy = tuple(float(v) for v in joint.get("rpy", joint.get("rpy_rad", (0.0, 0.0, 0.0))))
    return _pure_rpy_xyz_to_matrix(xyz, rpy)


def _extract_limits_pair(joint: Mapping[str, Any], j_type: str) -> tuple[float, float]:
    raw_lim = joint.get("limits", joint.get("declared_limits", joint.get("limits_rad", joint.get("limit"))))
    if isinstance(raw_lim, Mapping) and "lower" in raw_lim and "upper" in raw_lim:
        return (float(raw_lim["lower"]), float(raw_lim["upper"]))
    if isinstance(raw_lim, (list, tuple, np.ndarray)) and len(raw_lim) == 2:
        return (float(raw_lim[0]), float(raw_lim[1]))
    if j_type == "continuous":
        return (-2.0 * math.pi, 2.0 * math.pi)
    return (-math.pi, math.pi)


def chain_from_manifest(
    model: Mapping[str, Any],
    *,
    joint_names: Sequence[str] | None = None,
    tip_link: str | None = None,
    base_link: str | None = None,
    arm_prefix: str | None = None,
    expected_dof: int | None = None,
    joint_limits: Mapping[str, Sequence[float]] | np.ndarray | Sequence[Sequence[float]] | None = None,
    rest_posture_rad: Sequence[float] | None = None,
) -> KinematicChainSpec:
    """Construct an N-DoF ``KinematicChainSpec`` from a URDF manifest or ``physical.json`` kinematics dict.

    Pure Python + NumPy so simulator workers without Pinocchio can construct chain specs at startup.
    """
    if not isinstance(model, Mapping):
        raise TypeError("robot_model_must_be_mapping")
    raw_joints = model.get("joints")
    if not isinstance(raw_joints, (list, tuple)) or not raw_joints:
        raise ValueError("robot_model_missing_joints_list")

    resolved_base = str(base_link or model.get("base_link", "base_link"))
    resolved_tip = str(tip_link or model.get("end_effector_link") or model.get("tcp_link") or "")

    by_child: dict[str, Mapping[str, Any]] = {}
    has_topology = True
    for item in raw_joints:
        if not isinstance(item, Mapping) or "parent" not in item or "child" not in item:
            has_topology = False
            break
        by_child[str(item["child"])] = item

    chain_joints: list[Mapping[str, Any]] = []
    if has_topology and resolved_tip and resolved_tip in by_child:
        cursor = resolved_tip
        visited: set[str] = set()
        while cursor != resolved_base:
            if cursor in visited or cursor not in by_child:
                raise ValueError(f"kinematic_path_not_connected:{resolved_base}->{resolved_tip}")
            visited.add(cursor)
            j = by_child[cursor]
            chain_joints.append(j)
            cursor = str(j["parent"])
        chain_joints.reverse()
    else:
        wanted_names = set(joint_names) if joint_names is not None else None
        has_prefixed = bool(
            arm_prefix and any(isinstance(j, Mapping) and str(j.get("name", "")).startswith(arm_prefix) for j in raw_joints)
        )
        for j in raw_joints:
            if not isinstance(j, Mapping) or j.get("mimic") is not None:
                continue
            j_name = str(j.get("name", ""))
            if has_prefixed and arm_prefix and not j_name.startswith(arm_prefix):
                continue
            if wanted_names is None and ("finger" in j_name or "gripper" in j_name):
                continue
            chain_joints.append(j)

    traced_topology = bool(has_topology and resolved_tip and resolved_tip in by_child)
    wanted_order = tuple(str(n) for n in joint_names) if joint_names is not None else None
    active_names: list[str] = []
    active_types: list[str] = []
    active_parent_tfs: list[np.ndarray] = []
    active_axes: list[np.ndarray] = []
    active_limits: list[tuple[float, float]] = []
    active_links: list[str] = []
    accum_fixed = np.eye(4, dtype=np.float64)

    for j in chain_joints:
        if j.get("mimic") is not None:
            continue
        j_name = str(j.get("name", ""))
        j_type = str(j.get("type", "revolute"))
        t_pj = _extract_origin_4x4(j)
        if j_type == "fixed" or (wanted_order is not None and j_name not in wanted_order):
            if not traced_topology and wanted_order is not None and j_type != "fixed" and j_name not in wanted_order:
                continue
            accum_fixed = accum_fixed @ t_pj
            continue
        if j_type not in ("revolute", "continuous", "prismatic"):
            raise ValueError(f"unsupported_joint_type:{j_type}")
        if expected_dof is not None and len(active_names) >= expected_dof and wanted_order is None and not (has_topology and resolved_tip):
            break
        eff_parent = accum_fixed @ t_pj
        accum_fixed = np.eye(4, dtype=np.float64)
        ax_raw = _finite_array(j.get("axis", (0.0, 0.0, 1.0)), (3,), error_code=f"invalid_{j_name}_axis")
        ax_norm = float(np.linalg.norm(ax_raw))
        if ax_norm <= 1e-9:
            raise ValueError(f"zero_joint_axis_in_robot_model:{j_name}")
        active_names.append(j_name)
        active_types.append(j_type)
        active_parent_tfs.append(eff_parent)
        active_axes.append(ax_raw / ax_norm)
        active_limits.append(_extract_limits_pair(j, j_type))
        active_links.append(str(j.get("child", f"link{len(active_names)}")))

    if wanted_order is not None and tuple(active_names) != wanted_order:
        raise ValueError(f"chain_joint_order_mismatch:expected={wanted_order},got={tuple(active_names)}")
    if expected_dof is not None and len(active_names) != expected_dof:
        raise ValueError(f"robot_model_requires_{expected_dof}_active_joints_got_{len(active_names)}")
    if not active_names:
        raise ValueError("robot_model_has_no_active_joints")

    tip_tf = accum_fixed
    if model.get("T_joint6_flange") is not None:
        tip_tf = tip_tf @ _finite_array(model["T_joint6_flange"], (4, 4))
    elif model.get("T_tip_flange") is not None:
        tip_tf = tip_tf @ _finite_array(model["T_tip_flange"], (4, 4))

    t_wb = None
    if model.get("T_world_base") is not None:
        t_wb = _finite_array(model["T_world_base"], (4, 4), error_code="invalid_T_world_base_in_robot_model").copy()

    lims_arr = (
        _finite_array(model["joint_limits_rad"], (len(active_names), 2))
        if model.get("joint_limits_rad") is not None
        else np.asarray(active_limits, dtype=np.float64)
    )
    spec = KinematicChainSpec(
        joint_names=tuple(active_names),
        joint_types=tuple(active_types),
        parent_transforms=np.stack(active_parent_tfs, axis=0),
        joint_axes=np.stack(active_axes, axis=0),
        joint_limits_rad=lims_arr,
        tip_transform=tip_tf,
        base_link=resolved_base,
        tip_link=resolved_tip or active_links[-1],
        link_names=tuple(active_links),
        T_world_base=t_wb,
        rest_posture_rad=np.asarray(rest_posture_rad, dtype=np.float64) if rest_posture_rad is not None else None,
    )
    return spec.with_limits(joint_limits) if joint_limits is not None else spec


def is_pink_compat_type_error(exc: BaseException) -> bool:
    """Return True only for C++ Boost.Python converter or QP-solver keyword incompatibilities in ``Pink + DAQP``."""
    if not isinstance(exc, TypeError):
        return False
    msg = str(exc)
    return (
        ("No to_python (by-value) converter found for C++ type" in msg and "vector<bool" in msg)
        or "unexpected keyword argument 'primal_start'" in msg
    )


def _fk_links_numpy(chain: KinematicChainSpec, q_vec: np.ndarray, base_tf: np.ndarray) -> dict[str, np.ndarray]:
    cur = base_tf.copy()
    poses: dict[str, np.ndarray] = {chain.base_link: cur.copy()}
    for idx in range(chain.dof):
        frame = cur @ chain.parent_transforms[idx]
        axis_loc = chain.joint_axes[idx]
        val = float(q_vec[idx])
        motion = np.eye(4, dtype=np.float64)
        if chain.joint_types[idx] in ("revolute", "continuous"):
            motion[:3, :3] = rodrigues(axis_loc, val)
        else:
            motion[:3, 3] = axis_loc * val
        cur = frame @ motion
        poses[chain.link_names[idx]] = cur.copy()
    poses[chain.tip_link] = (cur @ chain.tip_transform).copy()
    return poses


def _fk_jacobian_numpy(chain: KinematicChainSpec, q_vec: np.ndarray, base_tf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cur = base_tf.copy()
    origins = np.empty((chain.dof, 3), dtype=np.float64)
    axes_world = np.empty((chain.dof, 3), dtype=np.float64)
    for idx in range(chain.dof):
        frame = cur @ chain.parent_transforms[idx]
        axis_local = chain.joint_axes[idx]
        origins[idx] = frame[:3, 3]
        axes_world[idx] = frame[:3, :3] @ axis_local
        motion = np.eye(4, dtype=np.float64)
        if chain.joint_types[idx] in ("revolute", "continuous"):
            motion[:3, :3] = rodrigues(axis_local, float(q_vec[idx]))
        else:
            motion[:3, 3] = axis_local * float(q_vec[idx])
        cur = frame @ motion
    tip = cur @ chain.tip_transform
    tip_pos = tip[:3, 3]
    jac = np.zeros((6, chain.dof), dtype=np.float64)
    for idx in range(chain.dof):
        if chain.joint_types[idx] in ("revolute", "continuous"):
            jac[:3, idx] = np.cross(axes_world[idx], tip_pos - origins[idx])
            jac[3:, idx] = axes_world[idx]
        else:
            jac[:3, idx] = axes_world[idx]
    return tip, jac


def _fk_batch_numpy(chain: KinematicChainSpec, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bsz, dof = q.shape
    has_tip_offset = bool(chain._has_tip_offset)  # type: ignore[attr-defined]
    cos_q = np.cos(q)
    sin_q = np.sin(q)
    rot_cur = np.broadcast_to(np.eye(3, dtype=np.float64), (bsz, 3, 3)).copy()
    pos_cur = np.zeros((bsz, 3), dtype=np.float64)
    joint_pos = np.empty((bsz, dof, 3), dtype=np.float64)
    joint_axes = np.empty((bsz, dof, 3), dtype=np.float64)
    link_origins = np.zeros((bsz, dof + 1, 3), dtype=np.float64)
    eye3 = np.eye(3, dtype=np.float64)
    p_rots = chain.parent_rots
    p_pos = chain.parent_pos_m
    skews = chain._axis_skews  # type: ignore[attr-defined]
    axis_sq = chain._axis_sq  # type: ignore[attr-defined]
    rotary_mask = chain._rotary_mask  # type: ignore[attr-defined]
    all_rotary = bool(chain._all_rotary)  # type: ignore[attr-defined]
    for idx in range(dof):
        pos_cur = pos_cur + (rot_cur @ p_pos[idx])
        rot_joint = rot_cur @ p_rots[idx]
        joint_pos[:, idx, :] = pos_cur
        joint_axes[:, idx, :] = rot_joint @ chain.joint_axes[idx]
        if rotary_mask[idx]:
            c = cos_q[:, idx][:, None, None]
            s = sin_q[:, idx][:, None, None]
            rot_motion = eye3[None, :, :] + s * skews[idx][None, :, :] + (1.0 - c) * axis_sq[idx][None, :, :]
            rot_cur = rot_joint @ rot_motion
        else:
            pos_cur = pos_cur + q[:, idx][:, None] * joint_axes[:, idx, :]
            rot_cur = rot_joint
        link_origins[:, idx + 1, :] = pos_cur
    if has_tip_offset:
        tip_tf = chain.tip_transform
        pos_cur = pos_cur + (rot_cur @ tip_tf[:3, 3])
        rot_cur = rot_cur @ tip_tf[:3, :3]
        link_origins[:, dof, :] = pos_cur
    r_vec = pos_cur[:, None, :] - joint_pos
    ax0, ax1, ax2 = joint_axes[:, :, 0], joint_axes[:, :, 1], joint_axes[:, :, 2]
    rv0, rv1, rv2 = r_vec[:, :, 0], r_vec[:, :, 1], r_vec[:, :, 2]
    if all_rotary:
        jac_np = np.empty((bsz, 6, dof), dtype=np.float64)
        jac_np[:, 0, :] = ax1 * rv2 - ax2 * rv1
        jac_np[:, 1, :] = ax2 * rv0 - ax0 * rv2
        jac_np[:, 2, :] = ax0 * rv1 - ax1 * rv0
        jac_np[:, 3, :] = ax0
        jac_np[:, 4, :] = ax1
        jac_np[:, 5, :] = ax2
    else:
        jac_np = np.zeros((bsz, 6, dof), dtype=np.float64)
        for idx in range(dof):
            if rotary_mask[idx]:
                jac_np[:, 0, idx] = ax1[:, idx] * rv2[:, idx] - ax2[:, idx] * rv1[:, idx]
                jac_np[:, 1, idx] = ax2[:, idx] * rv0[:, idx] - ax0[:, idx] * rv2[:, idx]
                jac_np[:, 2, idx] = ax0[:, idx] * rv1[:, idx] - ax1[:, idx] * rv0[:, idx]
                jac_np[:, 3:, idx] = joint_axes[:, idx, :]
            else:
                jac_np[:, :3, idx] = joint_axes[:, idx, :]
    return pos_cur, rot_cur, jac_np, link_origins


def fk_links(
    chain_or_model: KinematicChainSpec | Mapping[str, Any],
    q: Sequence[float] | np.ndarray | Mapping[str, float],
    *,
    joint_names: Sequence[str] | None = None,
    extra_joints: Mapping[str, float] | None = None,
    T_world_base: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute ``4x4`` link transforms (translations in ``m``) for a ``KinematicChainSpec`` or URDF manifest via ``Pinocchio`` (with pure-NumPy fallback)."""
    if isinstance(chain_or_model, KinematicChainSpec):
        chain = chain_or_model
        if isinstance(q, Mapping):
            q_vec = _finite_array([q[name] for name in chain.joint_names], (chain.dof,))
        else:
            q_vec = _finite_array(q, (chain.dof,), error_code="invalid_joint_vector")
        base_tf = (
            rigid_transform(T_world_base)
            if T_world_base is not None
            else (chain.T_world_base.copy() if chain.T_world_base is not None else np.eye(4, dtype=np.float64))
        )
        try:
            import pinocchio as pin

            p_model, link_fids, tip_fid = _pin_bundle(chain)
            p_data = p_model.createData()
            pin.framesForwardKinematics(p_model, p_data, q_vec)
            is_identity_base = bool(np.allclose(base_tf, np.eye(4), atol=1e-15))
            poses = {chain.base_link: base_tf.copy()}
            for idx, fid in enumerate(link_fids):
                mat = np.asarray(p_data.oMf[fid].homogeneous, dtype=np.float64)
                poses[chain.link_names[idx]] = mat.copy() if is_identity_base else (base_tf @ mat)
            tip_mat = np.asarray(p_data.oMf[tip_fid].homogeneous, dtype=np.float64)
            poses[chain.tip_link] = tip_mat.copy() if is_identity_base else (base_tf @ tip_mat)
            return poses
        except ImportError:
            return _fk_links_numpy(chain, q_vec, base_tf)
        except TypeError as exc:
            if not is_pink_compat_type_error(exc):
                raise
            return _fk_links_numpy(chain, q_vec, base_tf)

    if isinstance(q, Mapping):
        values = {str(k): float(v) for k, v in q.items()}
    else:
        if joint_names is None:
            raise ValueError("fk_manifest_sequence_requires_joint_names")
        q_arr = _finite_array(q, (len(joint_names),))
        values = {str(name): float(val) for name, val in zip(joint_names, q_arr)}
    if extra_joints:
        for k, v in extra_joints.items():
            values[str(k)] = float(v)
    return fk_manifest(chain_or_model, values, T_world_base=T_world_base, require_source_kind=False)


def fk_jacobian(
    chain: KinematicChainSpec,
    q: Sequence[float] | np.ndarray,
    *,
    T_world_base: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ``(T_tip, J_geometric)`` for an N-DoF ``KinematicChainSpec`` via ``Pinocchio`` (with pure-NumPy fallback).

    Returns:
        - ``T_tip``: ``(4, 4)`` homogeneous pose of ``chain.tip_link`` (translations in ``m``).
        - ``J_geometric``: ``(6, N)`` analytical geometric Jacobian (``pin.LOCAL_WORLD_ALIGNED``) where rows ``0..3``
          are linear velocity partials (``m/rad`` or ``m/m``) and rows ``3..6`` are angular velocity axes (``rad/rad``).
    """
    q_vec = _finite_array(q, (chain.dof,), error_code="invalid_joint_vector")
    base_tf = (
        rigid_transform(T_world_base)
        if T_world_base is not None
        else (chain.T_world_base.copy() if chain.T_world_base is not None else np.eye(4, dtype=np.float64))
    )
    try:
        import pinocchio as pin

        p_model, _, tip_fid = _pin_bundle(chain)
        p_data = p_model.createData()
        pin.forwardKinematics(p_model, p_data, q_vec)
        pin.updateFramePlacements(p_model, p_data)
        tip_base = np.asarray(p_data.oMf[tip_fid].homogeneous, dtype=np.float64)
        jac_base = np.asarray(
            pin.computeFrameJacobian(p_model, p_data, q_vec, tip_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED),
            dtype=np.float64,
        ).reshape(6, chain.dof)
        if np.allclose(base_tf, np.eye(4), atol=1e-15):
            return tip_base.copy(), jac_base.copy()
        rot_wb = base_tf[:3, :3]
        jac_world = np.empty_like(jac_base)
        jac_world[:3, :] = rot_wb @ jac_base[:3, :]
        jac_world[3:, :] = rot_wb @ jac_base[3:, :]
        return base_tf @ tip_base, jac_world
    except ImportError:
        return _fk_jacobian_numpy(chain, q_vec, base_tf)
    except TypeError as exc:
        if not is_pink_compat_type_error(exc):
            raise
        return _fk_jacobian_numpy(chain, q_vec, base_tf)


def fk_batch(
    chain: KinematicChainSpec,
    q_batch_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batched N-DoF forward kinematics and analytical geometric Jacobian in ``base_link`` frame via ``Pinocchio`` (with pure-NumPy fallback).

    Parameters:
        chain: ``KinematicChainSpec`` with ``N = chain.dof`` active joints.
        q_batch_rad: Joint array of shape ``(B, N)`` or ``(N,)`` in ``rad`` (or ``m`` for prismatic).

    Returns:
        Tuple ``(pos_tip_m, rot_tip, jac_base, link_origins_m)``:
        - ``pos_tip_m``: ``(B, 3)`` tip position in ``base_link`` (``m``).
        - ``rot_tip``: ``(B, 3, 3)`` tip rotation in ``base_link``.
        - ``jac_base``: ``(B, 6, N)`` analytical geometric Jacobian in ``base_link``.
        - ``link_origins_m``: ``(B, N + 1, 3)`` origins ``[base_link, joint_1..joint_N/tip]`` in ``m``.
    """
    q = np.asarray(q_batch_rad, dtype=np.float64)
    if q.ndim == 1:
        q = q[None, :]
    if q.ndim != 2 or q.shape[1] != chain.dof:
        raise ValueError(f"fk_batch_shape_mismatch:expected_(*,{chain.dof}),got_{q.shape}")
    if not np.isfinite(q).all():
        raise ValueError("invalid_joint_vector")
    bsz, dof = q.shape
    has_tip_offset = bool(chain._has_tip_offset)  # type: ignore[attr-defined]
    try:
        import pinocchio as pin

        p_model, link_fids, tip_fid = _pin_bundle(chain)
        p_data = p_model.createData()
        pos_tip = np.empty((bsz, 3), dtype=np.float64)
        rot_tip = np.empty((bsz, 3, 3), dtype=np.float64)
        jac_out = np.empty((bsz, 6, dof), dtype=np.float64)
        link_origins = np.zeros((bsz, dof + 1, 3), dtype=np.float64)
        for b in range(bsz):
            row = q[b]
            pin.forwardKinematics(p_model, p_data, row)
            pin.updateFramePlacements(p_model, p_data)
            oMtip = p_data.oMf[tip_fid]
            pos_tip[b] = oMtip.translation
            rot_tip[b] = oMtip.rotation
            jac_out[b] = np.asarray(
                pin.computeFrameJacobian(p_model, p_data, row, tip_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED),
                dtype=np.float64,
            ).reshape(6, dof)
            for j_idx, fid in enumerate(link_fids):
                link_origins[b, j_idx + 1] = p_data.oMf[fid].translation
            if has_tip_offset:
                link_origins[b, dof] = oMtip.translation
        return pos_tip, rot_tip, jac_out, link_origins
    except ImportError:
        return _fk_batch_numpy(chain, q)
    except TypeError as exc:
        if not is_pink_compat_type_error(exc):
            raise
        return _fk_batch_numpy(chain, q)


@dataclass(frozen=True)
class IKTarget:
    """Target end-effector pose in ``base_link`` frame (``m``) and optional redundant-DoF posture (``rad``)."""

    T_base_tip: np.ndarray
    posture_rad: np.ndarray | None = None

    def __post_init__(self) -> None:
        mat = _finite_array(self.T_base_tip, (4, 4), error_code="ik_target_must_be_finite_4x4")
        rot = mat[:3, :3]
        if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-6) or abs(float(np.linalg.det(rot)) - 1.0) > 1e-6:
            raise ValueError("ik_target_rotation_invalid")
        object.__setattr__(self, "T_base_tip", mat)
        if self.posture_rad is not None:
            object.__setattr__(self, "posture_rad", _finite_array(self.posture_rad))

    @classmethod
    def from_pose(
        cls,
        position_m: Sequence[float] | np.ndarray,
        quaternion_xyzw: Sequence[float] | np.ndarray | None = None,
        *,
        rotation: np.ndarray | None = None,
        T_world_base: np.ndarray | None = None,
        posture_rad: Sequence[float] | np.ndarray | None = None,
    ) -> IKTarget:
        """Build an ``IKTarget`` from ``(position_m, quaternion_xyzw | rotation)``, optionally transforming from world to base."""
        pos = _finite_array(position_m, (3,), error_code="invalid_target_position_m")
        if rotation is not None:
            rot = _finite_array(rotation, (3, 3), error_code="ik_target_rotation_invalid")
        elif quaternion_xyzw is not None:
            rot = quat_xyzw_to_matrix(quaternion_xyzw)
        else:
            raise ValueError("ik_target_requires_quaternion_xyzw_or_rotation")
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = rot
        mat[:3, 3] = pos
        if T_world_base is not None:
            wb = rigid_transform(T_world_base)
            wb_inv = np.eye(4, dtype=np.float64)
            wb_inv[:3, :3] = wb[:3, :3].T
            wb_inv[:3, 3] = -(wb[:3, :3].T @ wb[:3, 3])
            mat = wb_inv @ mat
        return cls(
            T_base_tip=mat,
            posture_rad=np.asarray(posture_rad, dtype=np.float64) if posture_rad is not None else None,
        )


@dataclass(frozen=True)
class IKOptions:
    """Configuration options for ``solve_ik`` across 5-DoF, 6-DoF, and 7-DoF+ chains (``Pink + DAQP`` or LM/DLS fallback).

    Attributes:
        tasks: Task hierarchy tuple:
            - ``("pose",)``: weighted 6D pose IK (``orientation_weight=0`` solves position-only).
            - ``("position", "orientation")``: hierarchical position-priority IK (reaches 3D position first,
              then minimizes orientation residual on the position manifold; ideal for 5-DoF arms).
            - ``("pose", "posture")``: primary 6D pose IK + null-space posture attraction toward
              ``target.posture_rad`` / ``chain.rest_posture_rad`` (ideal for 7-DoF+ redundant arms).
        position_weight: Weight on Euclidean position residual in ``1/m`` (default ``1.0``).
        orientation_weight: Weight on SO(3) rotation residual in ``1/rad`` (default ``0.3``).
        posture_weight: Null-space posture attraction weight (dimensionless, default ``0.15``).
        position_tolerance_m: Convergence threshold for position error in ``m`` (default ``1e-3``).
        orientation_tolerance_rad: Convergence threshold for rotation error in ``rad`` (default ``1e-2``).
        max_iterations: Maximum solver iterations per seed attempt (non-negative ``int``, default ``200``).
        restarts: Number of extra deterministic seeds tried when ``q_seed`` does not converge (default ``0``).
        damping: ``"adaptive_lm"`` or ``"error_scaled"`` (used by the analytical-Jacobian LM/DLS fallback).
        trust_region_rad: Optional infinity-norm box bound ``|q - q_seed|_inf <= trust_region_rad`` in ``rad``.
        max_step_rad: Optional positive per-iteration joint step infinity-norm cap in ``rad``.
        limit_policy: ``"clamp"`` (hard clamp to joint limits) or ``"wrap_then_clamp"`` (periodic wrap near seed first).
        rng_seed: Deterministic PRNG seed for multi-start restarts (default ``0``).
    """

    tasks: tuple[str, ...] = ("pose",)
    position_weight: float = 1.0
    orientation_weight: float = 0.3
    posture_weight: float = 0.15
    position_tolerance_m: float = 1e-3
    orientation_tolerance_rad: float = 1e-2
    max_iterations: int = 200
    restarts: int = 0
    damping: str = "adaptive_lm"
    trust_region_rad: float | None = None
    max_step_rad: float | None = None
    limit_policy: str = "clamp"
    rng_seed: int = 0

    def __post_init__(self) -> None:
        if self.tasks not in (("pose",), ("position", "orientation"), ("pose", "posture")):
            raise ValueError(f"unsupported_ik_tasks:{self.tasks}")
        if not (math.isfinite(self.position_weight) and self.position_weight > 0.0):
            raise ValueError("ik_invalid_position_weight")
        if not (math.isfinite(self.orientation_weight) and self.orientation_weight >= 0.0):
            raise ValueError("ik_invalid_orientation_weight")
        if not (math.isfinite(self.posture_weight) and self.posture_weight >= 0.0):
            raise ValueError("ik_invalid_posture_weight")
        if not (math.isfinite(self.position_tolerance_m) and self.position_tolerance_m >= 0.0):
            raise ValueError("ik_invalid_position_tolerance_m")
        if not (math.isfinite(self.orientation_tolerance_rad) and self.orientation_tolerance_rad >= 0.0):
            raise ValueError("ik_invalid_orientation_tolerance_rad")
        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, (int, np.integer)) or int(self.max_iterations) < 0:
            raise ValueError("ik_invalid_max_iterations")
        if isinstance(self.restarts, bool) or not isinstance(self.restarts, (int, np.integer)) or int(self.restarts) < 0:
            raise ValueError("ik_invalid_restarts")
        if self.damping not in ("adaptive_lm", "error_scaled"):
            raise ValueError(f"unsupported_ik_damping:{self.damping}")
        if self.trust_region_rad is not None and not (math.isfinite(self.trust_region_rad) and self.trust_region_rad >= 0.0):
            raise ValueError("invalid_trust_region_rad")
        if self.max_step_rad is not None and not (math.isfinite(self.max_step_rad) and self.max_step_rad > 0.0):
            raise ValueError("ik_invalid_max_step_rad")
        if self.limit_policy not in ("clamp", "wrap_then_clamp"):
            raise ValueError(f"unsupported_ik_limit_policy:{self.limit_policy}")


@dataclass(frozen=True)
class IKResult:
    """True achieved inverse-kinematics solution and physical residuals (never masks non-convergence)."""

    q_rad: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    converged: bool
    iterations: int

    def as_dict(self) -> dict[str, Any]:
        """Serialize to the standard ``{q_rad, position_error_m, orientation_error_rad, converged, iterations}`` dict."""
        return {
            "q_rad": [float(v) for v in self.q_rad],
            "position_error_m": float(self.position_error_m),
            "orientation_error_rad": float(self.orientation_error_rad),
            "converged": bool(self.converged),
            "iterations": int(self.iterations),
        }


def _apply_limits(q: np.ndarray, ref: np.ndarray, lower: np.ndarray, upper: np.ndarray, policy: str, rotary_mask: np.ndarray) -> np.ndarray:
    if policy == "wrap_then_clamp":
        wrapped = ref + (((q - ref) + math.pi) % math.tau - math.pi)
        use_wrap = rotary_mask & (wrapped >= lower) & (wrapped <= upper)
        q = np.where(use_wrap, wrapped, q)
    return np.clip(q, lower, upper)


_PINK_DAQP_CACHE: bool | None = None


def _pink_daqp_usable() -> bool:
    """Probe once per process whether ``Pinocchio + Pink + DAQP`` executes end-to-end cleanly."""
    global _PINK_DAQP_CACHE
    if _PINK_DAQP_CACHE is not None:
        return _PINK_DAQP_CACHE
    try:
        import pink
        import pinocchio as pin

        probe_model = pin.Model()
        j0 = int(probe_model.addJoint(0, pin.JointModelRZ(), pin.SE3.Identity(), "j0"))
        j1 = int(probe_model.addJoint(j0, pin.JointModelRY(), pin.SE3.Identity(), "j1"))
        probe_model.addFrame(pin.Frame(_INTERNAL_TIP_FRAME, j1, 0, pin.SE3.Identity(), pin.FrameType.OP_FRAME))
        probe_model.lowerPositionLimit[:] = np.array([-1.0, -1.0], dtype=np.float64)
        probe_model.upperPositionLimit[:] = np.array([1.0, 1.0], dtype=np.float64)
        probe_data = probe_model.createData()
        q_zero = np.zeros(2, dtype=np.float64)
        cfg = pink.Configuration(probe_model, probe_data, q_zero)
        lim = pink.limits.ConfigurationLimit(probe_model, config_limit_gain=1.0)
        ft = pink.tasks.FrameTask(_INTERNAL_TIP_FRAME, position_cost=1.0, orientation_cost=0.1)
        ft.set_target(pin.SE3.Identity())
        pt = pink.tasks.PostureTask(cost=1e-3)
        pt.set_target(q_zero)
        vel = pink.solve_ik(cfg, [ft, pt], dt=1.0, solver="daqp", damping=1e-6, limits=[lim], safety_break=False)
        _PINK_DAQP_CACHE = bool(vel is not None and np.asarray(vel).shape == (2,) and np.isfinite(vel).all())
    except ImportError:
        _PINK_DAQP_CACHE = False
    except TypeError as exc:
        if not is_pink_compat_type_error(exc):
            raise
        _PINK_DAQP_CACHE = False
    except Exception as exc:
        if not (
            exc.__class__.__module__.startswith("qpsolvers")
            and exc.__class__.__name__ in ("SolverNotFound", "NoSolverSelected")
        ):
            raise
        _PINK_DAQP_CACHE = False
    return _PINK_DAQP_CACHE


def ik_backend() -> str:
    """Return active inverse-kinematics solver backend identifier: ``'pink_daqp'`` or ``'dls'``."""
    return "pink_daqp" if _pink_daqp_usable() else "dls"


def _pink_run_loop(
    p_model: Any,
    p_data: Any,
    target_T: np.ndarray,
    q_init: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    pos_cost: float,
    ori_cost: float,
    posture_target: np.ndarray,
    posture_cost: float,
    max_iters: int,
    pos_tol: float,
    ori_tol: float,
    step_cap: float,
    lm_damping: float = 1e-3,
    qp_damping: float = 1e-6,
    stop_early: bool = True,
) -> tuple[np.ndarray, float, float, int]:
    import pink
    import pinocchio as pin

    cfg = pink.Configuration(p_model, p_data, q_init.copy())
    ft = pink.tasks.FrameTask(
        _INTERNAL_TIP_FRAME,
        position_cost=float(pos_cost),
        orientation_cost=float(ori_cost),
        lm_damping=float(lm_damping) if ori_cost > 0.0 else 0.0,
    )
    ft.set_target(pin.SE3(target_T[:3, :3], target_T[:3, 3]))
    tasks: list[Any] = [ft]
    if posture_cost > 0.0:
        pt = pink.tasks.PostureTask(cost=float(posture_cost))
        pt.set_target(np.clip(posture_target, lower, upper))
        tasks.append(pt)
    lim = pink.limits.ConfigurationLimit(p_model, config_limit_gain=1.0)

    T_cur = cfg.get_transform_frame_to_world(_INTERNAL_TIP_FRAME).homogeneous
    best_pe, best_oe = pose_error(T_cur, target_T)
    best_q = cfg.q.copy()
    best_obj = (pos_cost * best_pe) ** 2 + (ori_cost * best_oe) ** 2
    if posture_cost > 5e-3:
        best_obj += (posture_cost * float(np.linalg.norm(best_q - posture_target))) ** 2
    iters_done = 0

    for it in range(1, max_iters + 1):
        if stop_early and best_pe <= pos_tol * 0.05 and (ori_cost == 0.0 or best_oe <= ori_tol * 0.05):
            break
        try:
            vel = pink.solve_ik(
                cfg,
                tasks,
                dt=1.0,
                solver="daqp",
                damping=float(qp_damping),
                limits=[lim],
                safety_break=False,
            )
        except (
            TypeError,
            AttributeError,
            KeyError,
            IndexError,
            NameError,
            UnboundLocalError,
            ImportError,
            ZeroDivisionError,
            AssertionError,
            NotImplementedError,
            RecursionError,
            SyntaxError,
            BufferError,
        ):
            raise
        except Exception:
            break
        iters_done = it
        v_max = float(np.max(np.abs(vel)))
        if v_max > step_cap:
            vel = vel * (step_cap / v_max)
        q_prev = cfg.q.copy()
        cfg.integrate_inplace(vel, 1.0)
        q_clamped = np.clip(cfg.q, lower, upper)
        if not np.array_equal(q_clamped, cfg.q):
            cfg.update(q_clamped)
        T_cur = cfg.get_transform_frame_to_world(_INTERNAL_TIP_FRAME).homogeneous
        pe, oe = pose_error(T_cur, target_T)
        obj = (pos_cost * pe) ** 2 + (ori_cost * oe) ** 2
        if posture_cost > 5e-3:
            obj += (posture_cost * float(np.linalg.norm(cfg.q - posture_target))) ** 2
        if obj < best_obj:
            best_obj = obj
            best_pe, best_oe = pe, oe
            best_q = cfg.q.copy()
        if float(np.max(np.abs(cfg.q - q_prev))) < 1e-10:
            break

    return best_q, best_pe, best_oe, iters_done


def _box_lm_step(
    H: np.ndarray,
    g: np.ndarray,
    lo_dq: np.ndarray,
    hi_dq: np.ndarray,
    lam: float,
    step_cap: float,
) -> np.ndarray:
    """Primal active-set convex box QP ``min 0.5 dq^T (H + lam I) dq - g^T dq`` on ``[max(lo_dq, -step_cap), min(hi_dq, step_cap)]``."""
    dof = g.shape[0]
    lo_b = np.maximum(lo_dq, -step_cap)
    hi_b = np.minimum(hi_dq, step_cap)
    A = H + lam * np.eye(dof, dtype=np.float64)
    x = np.clip(np.zeros(dof, dtype=np.float64), lo_b, hi_b)
    state = np.zeros(dof, dtype=np.int8)
    locked = (hi_b - lo_b) <= 1e-15
    state[locked] = 1
    x[locked] = lo_b[locked]
    scale_ref = max(1.0, float(np.max(np.abs(g))), float(np.max(np.abs(A))))
    kkt_tol = 1e-12 * scale_ref
    max_as_iters = max(64, 3 ** min(dof, 6), 16 * dof)

    for _ in range(max_as_iters):
        f_idx = np.flatnonzero(state == 0)
        if f_idx.size > 0:
            a_idx = np.flatnonzero(state != 0)
            rhs = g[f_idx] - (A[np.ix_(f_idx, a_idx)] @ x[a_idx] if a_idx.size > 0 else 0.0)
            x_f_unc = np.linalg.solve(A[np.ix_(f_idx, f_idx)], rhs)
            p_f = x_f_unc - x[f_idx]
            under = x_f_unc < lo_b[f_idx] - 1e-14
            over = x_f_unc > hi_b[f_idx] + 1e-14
            if np.any(under) or np.any(over):
                alpha = 1.0
                block_local = -1
                block_sign = 0
                for loc_i, j_idx in enumerate(f_idx):
                    pj = float(p_f[loc_i])
                    if under[loc_i] and pj < 0.0:
                        cand_a = (float(lo_b[j_idx]) - float(x[j_idx])) / pj
                        if cand_a <= alpha:
                            alpha = max(0.0, cand_a)
                            block_local = loc_i
                            block_sign = -1
                    elif over[loc_i] and pj > 0.0:
                        cand_a = (float(hi_b[j_idx]) - float(x[j_idx])) / pj
                        if cand_a <= alpha:
                            alpha = max(0.0, cand_a)
                            block_local = loc_i
                            block_sign = 1
                if block_local >= 0:
                    x[f_idx] = np.clip(x[f_idx] + alpha * p_f, lo_b[f_idx], hi_b[f_idx])
                    b_idx = int(f_idx[block_local])
                    x[b_idx] = lo_b[b_idx] if block_sign < 0 else hi_b[b_idx]
                    state[b_idx] = block_sign
                    continue
            x[f_idx] = np.clip(x_f_unc, lo_b[f_idx], hi_b[f_idx])

        r = g - (A @ x)
        best_viol = kkt_tol
        release_idx = -1
        for idx in range(dof):
            if locked[idx] or state[idx] == 0:
                continue
            viol = float(r[idx]) if state[idx] == -1 else float(-r[idx])
            if viol > best_viol:
                best_viol = viol
                release_idx = idx
        if release_idx < 0:
            break
        state[release_idx] = 0

    x = np.clip(x, lo_b, hi_b)
    if float(0.5 * (x @ (A @ x)) - (g @ x)) > 0.0:
        d_c = np.clip(g, lo_b, hi_b)
        denom = float(d_c @ (A @ d_c))
        tau = min(1.0, max(0.0, float(g @ d_c) / denom)) if denom > 1e-24 else 0.0
        x = np.clip(tau * d_c, lo_b, hi_b)
    return x


def _dls_run_loop(
    chain: KinematicChainSpec,
    target_T: np.ndarray,
    q_init: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    pos_cost: float,
    ori_cost: float,
    posture_target: np.ndarray,
    posture_cost: float,
    max_iters: int,
    pos_tol: float,
    ori_tol: float,
    step_cap: float,
    damping_mode: str = "adaptive_lm",
    stop_early: bool = True,
) -> tuple[np.ndarray, float, float, int]:
    """Box-constrained analytical-Jacobian Levenberg-Marquardt / DLS solver matching ``_pink_run_loop``."""
    dof = chain.dof
    eye4 = np.eye(4, dtype=np.float64)
    eye_n = np.eye(dof, dtype=np.float64)
    scale = max(float(pos_cost), float(ori_cost), float(posture_cost), 1e-12)
    wp_n = float(pos_cost) / scale
    wo_n = float(ori_cost) / scale
    wpost_n = float(posture_cost) / scale
    pt_clamped = np.clip(posture_target, lower, upper)

    def _eval_obj(dp_v: np.ndarray, dr_v: np.ndarray, q_v: np.ndarray) -> float:
        val = float((wp_n * wp_n) * (dp_v @ dp_v) + (wo_n * wo_n) * (dr_v @ dr_v))
        if wpost_n > 0.0:
            dq_p = q_v - pt_clamped
            val += float((wpost_n * wpost_n) * (dq_p @ dq_p))
        return val

    def _best_metric(pe_v: float, oe_v: float, q_v: np.ndarray) -> float:
        val = (pos_cost * pe_v) ** 2 + (ori_cost * oe_v) ** 2
        if posture_cost > 5e-3:
            val += (posture_cost * float(np.linalg.norm(q_v - pt_clamped))) ** 2
        return val

    q = np.clip(q_init.copy(), lower, upper)
    T_cur, jac = fk_jacobian(chain, q, T_world_base=eye4)
    dp, dr = pose_delta_vectors(T_cur, target_T)
    cur_obj = _eval_obj(dp, dr, q)
    best_pe = float(np.linalg.norm(dp))
    best_oe = float(np.linalg.norm(dr))
    best_q = q.copy()
    best_obj = _best_metric(best_pe, best_oe, best_q)
    lam = 1e-5
    iters_done = 0

    for it in range(1, max_iters + 1):
        if stop_early and best_pe <= pos_tol * 0.05 and (ori_cost == 0.0 or best_oe <= ori_tol * 0.05):
            break
        j_w = np.vstack([jac[:3] * wp_n, jac[3:] * wo_n])
        e_w = np.concatenate([dp * wp_n, dr * wo_n])
        H = j_w.T @ j_w
        g = j_w.T @ e_w
        if wpost_n > 0.0:
            w2 = wpost_n * wpost_n
            H = H + w2 * eye_n
            g = g + w2 * (pt_clamped - q)
        lo_dq = lower - q
        hi_dq = upper - q

        if damping_mode == "error_scaled":
            lam_es = max(1e-6, min(1e-2, 0.02 * float(np.dot(e_w, e_w))))
            accepted_es = False
            while lam_es <= 1e4:
                dq = _box_lm_step(H, g, lo_dq, hi_dq, lam_es, step_cap)
                q_cand = np.clip(q + dq, lower, upper)
                step_delta = float(np.max(np.abs(q_cand - q)))
                if step_delta < 1e-12:
                    break
                T_cand, jac_cand = fk_jacobian(chain, q_cand, T_world_base=eye4)
                dp_cand, dr_cand = pose_delta_vectors(T_cand, target_T)
                cand_obj = _eval_obj(dp_cand, dr_cand, q_cand)
                if cand_obj <= cur_obj:
                    iters_done = it
                    q, T_cur, jac, dp, dr, cur_obj = q_cand, T_cand, jac_cand, dp_cand, dr_cand, cand_obj
                    pe, oe = float(np.linalg.norm(dp)), float(np.linalg.norm(dr))
                    m_val = _best_metric(pe, oe, q)
                    if m_val < best_obj:
                        best_obj = m_val
                        best_pe, best_oe = pe, oe
                        best_q = q.copy()
                    accepted_es = step_delta >= 1e-10
                    break
                lam_es *= 8.0
            if not accepted_es:
                break
            continue

        accepted = False
        trial_lam = lam
        while trial_lam <= 1e4:
            dq = _box_lm_step(H, g, lo_dq, hi_dq, trial_lam, step_cap)
            q_cand = np.clip(q + dq, lower, upper)
            step_delta = float(np.max(np.abs(q_cand - q)))
            if step_delta < 1e-12:
                break
            T_cand, jac_cand = fk_jacobian(chain, q_cand, T_world_base=eye4)
            dp_cand, dr_cand = pose_delta_vectors(T_cand, target_T)
            cand_obj = _eval_obj(dp_cand, dr_cand, q_cand)
            if cand_obj < cur_obj:
                iters_done = it
                q, T_cur, jac, dp, dr, cur_obj = q_cand, T_cand, jac_cand, dp_cand, dr_cand, cand_obj
                lam = max(trial_lam / 3.0, 1e-9)
                pe, oe = float(np.linalg.norm(dp)), float(np.linalg.norm(dr))
                m_val = _best_metric(pe, oe, q)
                if m_val < best_obj:
                    best_obj = m_val
                    best_pe, best_oe = pe, oe
                    best_q = q.copy()
                accepted = step_delta >= 1e-10 and (cur_obj > 0.0)
                if step_delta < 1e-10:
                    accepted = False
                break
            trial_lam *= 4.0
        if not accepted:
            break

    return best_q, best_pe, best_oe, iters_done


def _solve_ik_with_runner(
    chain: KinematicChainSpec,
    ik_target: IKTarget,
    seed: np.ndarray,
    seeds: list[np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    opts: IKOptions,
    *,
    use_pink: bool,
) -> IKResult:
    rotary_mask = chain._rotary_mask  # type: ignore[attr-defined]
    seed_bounded = seeds[0]
    pos_tol = float(opts.position_tolerance_m)
    ori_tol = float(opts.orientation_tolerance_rad)
    wp = float(opts.position_weight)
    wo = float(opts.orientation_weight)
    max_iters = max(1, int(opts.max_iterations))
    step_cap = float(opts.max_step_rad) if opts.max_step_rad is not None else 0.85

    p_model: Any = None
    p_data: Any = None
    if use_pink:
        base_p_model, _, _ = _pin_bundle(chain)
        p_model = base_p_model.copy()
        p_model.lowerPositionLimit[:] = lower
        p_model.upperPositionLimit[:] = upper
        p_data = p_model.createData()

    def _run_stage(
        q_init: np.ndarray,
        *,
        pos_cost: float,
        ori_cost: float,
        posture_target: np.ndarray,
        posture_cost: float,
        stage_iters: int,
        stage_step_cap: float,
        lm_damping: float = 1e-3,
        qp_damping: float = 1e-6,
        stop_early: bool = True,
    ) -> tuple[np.ndarray, float, float, int]:
        if use_pink:
            return _pink_run_loop(
                p_model,
                p_data,
                ik_target.T_base_tip,
                q_init,
                lower,
                upper,
                pos_cost=pos_cost,
                ori_cost=ori_cost,
                posture_target=posture_target,
                posture_cost=posture_cost,
                max_iters=stage_iters,
                pos_tol=pos_tol,
                ori_tol=ori_tol,
                step_cap=stage_step_cap,
                lm_damping=lm_damping,
                qp_damping=qp_damping,
                stop_early=stop_early,
            )
        return _dls_run_loop(
            chain,
            ik_target.T_base_tip,
            q_init,
            lower,
            upper,
            pos_cost=pos_cost,
            ori_cost=ori_cost,
            posture_target=posture_target,
            posture_cost=posture_cost,
            max_iters=stage_iters,
            pos_tol=pos_tol,
            ori_tol=ori_tol,
            step_cap=stage_step_cap,
            damping_mode=opts.damping,
            stop_early=stop_early,
        )

    if opts.tasks == ("position", "orientation"):
        best_pp: tuple[tuple[bool, float], np.ndarray, float, float] | None = None
        total_iters = 0
        it_pos = max(1, (max_iters + 1) // 2)
        it_ori = max(0, max_iters - it_pos)
        for start in seeds:
            q_s1, pe_s1, oe_s1, it1 = _run_stage(
                _apply_limits(start, seed, lower, upper, opts.limit_policy, rotary_mask),
                pos_cost=1.0,
                ori_cost=0.0,
                posture_target=seed_bounded,
                posture_cost=2e-3,
                stage_iters=it_pos,
                stage_step_cap=step_cap,
                qp_damping=1e-4,
            )
            total_iters += it1
            if pe_s1 <= pos_tol and it_ori > 0:
                q_s2, pe_s2, oe_s2, it2 = _run_stage(
                    q_s1,
                    pos_cost=500.0,
                    ori_cost=0.5,
                    posture_target=q_s1,
                    posture_cost=1e-6,
                    stage_iters=it_ori,
                    stage_step_cap=min(step_cap, 0.08),
                    lm_damping=1e-4,
                    qp_damping=1e-6,
                )
                total_iters += it2
                q_sol = q_s2 if (pe_s2 <= pos_tol and oe_s2 <= oe_s1) else q_s1
            else:
                q_sol = q_s1

            q_sol = _apply_limits(q_sol, seed, lower, upper, opts.limit_policy, rotary_mask)
            tip_fk = fk_links(chain, q_sol, T_world_base=np.eye(4, dtype=np.float64))[chain.tip_link]
            p_err, o_err = pose_error(tip_fk, ik_target.T_base_tip)
            key_pp = (p_err > pos_tol, o_err if p_err <= pos_tol else p_err)
            if best_pp is None or key_pp < best_pp[0]:
                best_pp = (key_pp, q_sol, p_err, o_err)
            if p_err <= pos_tol and o_err <= ori_tol:
                break

        assert best_pp is not None
        _, q_best, p_best, o_best = best_pp
        return IKResult(
            q_rad=q_best,
            position_error_m=p_best,
            orientation_error_rad=o_best,
            converged=bool(p_best <= pos_tol and o_best <= ori_tol),
            iterations=total_iters,
        )

    use_posture = opts.tasks == ("pose", "posture") and float(opts.posture_weight) > 0.0
    if use_posture:
        if ik_target.posture_rad is not None:
            posture_q = _finite_array(ik_target.posture_rad, (chain.dof,))
        elif chain.rest_posture_rad is not None:
            posture_q = chain.rest_posture_rad
        else:
            posture_q = seed_bounded
    else:
        posture_q = seed_bounded

    def _is_done(p_e: float, o_e: float) -> bool:
        return p_e <= pos_tol and (wo == 0.0 or o_e <= ori_tol)

    best_pose: tuple[tuple[bool, float], np.ndarray, float, float] | None = None
    total_iters = 0
    it_stage2 = max(1, min(max_iters // 4, max_iters - 1)) if max_iters > 1 else 0
    it_stage1 = max(1, max_iters - it_stage2)
    for start in seeds:
        q_start = _apply_limits(start, seed, lower, upper, opts.limit_policy, rotary_mask)
        if use_posture:
            q_mid, _, _, it1 = _run_stage(
                q_start,
                pos_cost=wp * 50.0,
                ori_cost=wo * 50.0,
                posture_target=posture_q,
                posture_cost=float(opts.posture_weight),
                stage_iters=it_stage1,
                stage_step_cap=min(step_cap, 0.35),
                lm_damping=1e-4,
                stop_early=False,
            )
            if it_stage2 > 0:
                q_sol, _, _, it2 = _run_stage(
                    q_mid,
                    pos_cost=wp,
                    ori_cost=wo,
                    posture_target=q_mid,
                    posture_cost=1e-4,
                    stage_iters=it_stage2,
                    stage_step_cap=step_cap,
                    lm_damping=1e-6,
                    qp_damping=1e-8,
                )
            else:
                q_sol, it2 = q_mid, 0
            iters = it1 + it2
        else:
            p_reg = 3e-3 if wo == 0.0 else 5e-4
            q_sol, _, _, iters = _run_stage(
                q_start,
                pos_cost=wp,
                ori_cost=wo,
                posture_target=seed_bounded,
                posture_cost=p_reg,
                stage_iters=max_iters,
                stage_step_cap=step_cap,
                qp_damping=1e-4 if wo == 0.0 else 1e-6,
            )

        total_iters += iters
        q_sol = _apply_limits(q_sol, seed, lower, upper, opts.limit_policy, rotary_mask)
        tip_fk = fk_links(chain, q_sol, T_world_base=np.eye(4, dtype=np.float64))[chain.tip_link]
        p_err, o_err = pose_error(tip_fk, ik_target.T_base_tip)
        conv = _is_done(p_err, o_err)
        if use_posture and conv:
            cost = float(opts.posture_weight) * float(np.linalg.norm(q_sol - posture_q)) + 1e-4 * p_err
        else:
            cost = (wp * p_err) ** 2 + (wo * o_err) ** 2
        key_pose = (not conv, cost)
        if best_pose is None or key_pose < best_pose[0]:
            best_pose = (key_pose, q_sol, p_err, o_err)
        if conv:
            break

    assert best_pose is not None
    _, q_best, p_best, o_best = best_pose
    return IKResult(
        q_rad=q_best,
        position_error_m=p_best,
        orientation_error_rad=o_best,
        converged=_is_done(p_best, o_best),
        iterations=total_iters,
    )


def solve_ik(
    chain: KinematicChainSpec,
    target: IKTarget | np.ndarray,
    q_seed_rad: Sequence[float] | np.ndarray,
    options: IKOptions | None = None,
) -> IKResult:
    """Solve generic N-DoF inverse kinematics via ``Pinocchio + Pink + DAQP`` (or analytical-Jacobian LM fallback) with honest FK verification.

    Supports 5-DoF underactuated arms (``tasks=("pose",)`` or ``("position", "orientation")``),
    6-DoF spherical/offset-wrist arms, and 7-DoF+ redundant arms (``tasks=("pose", "posture")``).
    """
    global _PINK_DAQP_CACHE
    opts = options if options is not None else IKOptions()
    ik_target = target if isinstance(target, IKTarget) else IKTarget(T_base_tip=np.asarray(target, dtype=np.float64))
    seed = _finite_array(q_seed_rad, (chain.dof,), error_code="ik_seed_dimension_mismatch")

    lower = chain.joint_limits_rad[:, 0].copy()
    upper = chain.joint_limits_rad[:, 1].copy()
    if opts.trust_region_rad is not None:
        tr = float(opts.trust_region_rad)
        lower = np.maximum(lower, seed - tr)
        upper = np.minimum(upper, seed + tr)
        if np.any(lower > upper):
            tip0 = fk_links(chain, seed, T_world_base=np.eye(4, dtype=np.float64))[chain.tip_link]
            p0, r0 = pose_error(tip0, ik_target.T_base_tip)
            return IKResult(q_rad=seed.copy(), position_error_m=p0, orientation_error_rad=r0, converged=False, iterations=0)

    rotary_mask = chain._rotary_mask  # type: ignore[attr-defined]
    seed_bounded = _apply_limits(seed, seed, lower, upper, opts.limit_policy, rotary_mask)
    if opts.max_iterations == 0:
        tip0 = fk_links(chain, seed_bounded, T_world_base=np.eye(4, dtype=np.float64))[chain.tip_link]
        p0, r0 = pose_error(tip0, ik_target.T_base_tip)
        conv0 = p0 <= opts.position_tolerance_m and (
            opts.tasks != ("position", "orientation") and opts.orientation_weight == 0.0
            or r0 <= opts.orientation_tolerance_rad
        )
        return IKResult(q_rad=seed_bounded, position_error_m=p0, orientation_error_rad=r0, converged=bool(conv0), iterations=0)

    seeds = [seed_bounded]
    n_restarts = int(opts.restarts)
    if n_restarts > 0:
        rng = np.random.default_rng(opts.rng_seed)
        seeds.append(0.5 * (lower + upper))
        seeds.extend(rng.uniform(lower, upper) for _ in range(max(0, n_restarts - 1)))
        seeds = seeds[: 1 + n_restarts]

    if _pink_daqp_usable():
        try:
            return _solve_ik_with_runner(chain, ik_target, seed, seeds, lower, upper, opts, use_pink=True)
        except ImportError:
            _PINK_DAQP_CACHE = False
        except TypeError as exc:
            if not is_pink_compat_type_error(exc):
                raise
            _PINK_DAQP_CACHE = False

    return _solve_ik_with_runner(chain, ik_target, seed, seeds, lower, upper, opts, use_pink=False)


def _solve_ik_batch_dls(
    chain: KinematicChainSpec,
    pos_base_target: np.ndarray,
    rot_base_target: np.ndarray,
    q_out: np.ndarray,
    ref: np.ndarray,
    *,
    pos_tol_m: float,
    rot_tol_rad: float,
    rot_weight: float,
    max_iters: int,
    max_step_rad: float,
    posture_rad: np.ndarray | None,
    posture_weight: float,
) -> np.ndarray:
    """Multi-branch analytical-Jacobian Levenberg-Marquardt IK over ``(B, N)`` seeds in ``base_link`` frame."""
    lim_lo = chain.joint_limits_rad[:, 0]
    lim_hi = chain.joint_limits_rad[:, 1]
    bsz, dof = q_out.shape
    if bsz == 0:
        return q_out
    rotary_mask = chain._rotary_mask  # type: ignore[attr-defined]
    target_T = np.eye(4, dtype=np.float64)
    target_T[:3, :3] = rot_base_target
    target_T[:3, 3] = pos_base_target

    pt_cost = float(posture_weight) if (posture_rad is not None and posture_weight > 0.0) else 5e-4
    converged_branches = 0
    for b in range(bsz):
        q_start = q_out[b].copy()
        ref_b = ref[0] if ref.shape[0] == 1 else ref[b]
        pt_target = np.clip(posture_rad if (posture_rad is not None and posture_weight > 0.0) else q_start, lim_lo, lim_hi)
        q_sol, pe_f, oe_f, _ = _dls_run_loop(
            chain,
            target_T,
            q_start,
            lim_lo,
            lim_hi,
            pos_cost=1.0,
            ori_cost=float(rot_weight),
            posture_target=pt_target,
            posture_cost=pt_cost,
            max_iters=max_iters,
            pos_tol=pos_tol_m,
            ori_tol=rot_tol_rad,
            step_cap=float(max_step_rad),
            damping_mode="adaptive_lm",
            stop_early=True,
        )
        wrapped = ref_b + (((q_sol - ref_b) + math.pi) % math.tau - math.pi)
        wrap_ok = rotary_mask & (wrapped >= lim_lo) & (wrapped <= lim_hi)
        q_sol = np.where(wrap_ok, wrapped, q_sol)
        q_sol = np.clip(q_sol, lim_lo, lim_hi)
        q_out[b] = q_sol
        if pe_f <= pos_tol_m and oe_f <= rot_tol_rad:
            converged_branches += 1
            if converged_branches >= 3 and b >= 3:
                break
    return q_out


def solve_ik_batch(
    chain: KinematicChainSpec,
    pos_base_target: np.ndarray,
    rot_base_target: np.ndarray,
    seeds: np.ndarray,
    *,
    wrap_ref: np.ndarray | None = None,
    pos_tol_m: float = 0.002,
    rot_tol_rad: float = 0.02,
    rot_weight: float = 0.25,
    max_iters: int = 36,
    max_step_rad: float = 0.60,
    posture_rad: np.ndarray | None = None,
    posture_weight: float = 0.0,
) -> np.ndarray:
    """Multi-branch ``Pinocchio + Pink + DAQP`` (or analytical-Jacobian LM fallback) IK over ``(B, N)`` seeds in ``base_link`` frame.

    Evaluates homotopy branches around ``q_anchor`` and returns an optimized ``(B, N)`` joint array in ``rad``.
    """
    global _PINK_DAQP_CACHE
    pos_tgt = _finite_array(pos_base_target, (3,), error_code="invalid_target_position_m")
    rot_tgt = _finite_array(rot_base_target, (3, 3), error_code="ik_target_rotation_invalid")
    if not (math.isfinite(pos_tol_m) and pos_tol_m >= 0.0) or not (math.isfinite(rot_tol_rad) and rot_tol_rad >= 0.0):
        raise ValueError("ik_invalid_tolerance")
    if not (math.isfinite(rot_weight) and rot_weight >= 0.0) or not (math.isfinite(posture_weight) and posture_weight >= 0.0):
        raise ValueError("ik_invalid_weight")
    if isinstance(max_iters, bool) or not isinstance(max_iters, (int, np.integer)) or int(max_iters) < 0:
        raise ValueError("ik_invalid_max_iterations")
    max_iters = int(max_iters)
    if not (math.isfinite(max_step_rad) and max_step_rad > 0.0):
        raise ValueError("ik_invalid_max_step_rad")

    lim_lo = chain.joint_limits_rad[:, 0]
    lim_hi = chain.joint_limits_rad[:, 1]
    seeds_arr = np.asarray(seeds, dtype=np.float64)
    if seeds_arr.ndim != 2 or seeds_arr.shape[1] != chain.dof or not np.isfinite(seeds_arr).all():
        raise ValueError("ik_seed_dimension_mismatch")
    bsz, dof = seeds_arr.shape
    if bsz == 0:
        return np.empty((0, dof), dtype=np.float64)
    posture_vec = _finite_array(posture_rad, (dof,)) if posture_rad is not None else None
    q_out = np.clip(seeds_arr, lim_lo[None, :], lim_hi[None, :]).copy()
    ref = (
        q_out[0:1].copy()
        if wrap_ref is None
        else _finite_array(np.asarray(wrap_ref, dtype=np.float64).reshape(-1, dof))
    )
    rotary_mask = chain._rotary_mask  # type: ignore[attr-defined]
    if max_iters == 0:
        for b in range(bsz):
            ref_b = ref[0] if ref.shape[0] == 1 else ref[b]
            wrapped = ref_b + (((q_out[b] - ref_b) + math.pi) % math.tau - math.pi)
            wrap_ok = rotary_mask & (wrapped >= lim_lo) & (wrapped <= lim_hi)
            q_out[b] = np.clip(np.where(wrap_ok, wrapped, q_out[b]), lim_lo, lim_hi)
        return q_out

    if _pink_daqp_usable():
        try:
            import pink
            import pinocchio as pin

            base_p_model, _, _ = _pin_bundle(chain)
            p_model = base_p_model.copy()
            p_model.lowerPositionLimit[:] = lim_lo
            p_model.upperPositionLimit[:] = lim_hi
            p_data = p_model.createData()

            target_T = np.eye(4, dtype=np.float64)
            target_T[:3, :3] = rot_tgt
            target_T[:3, 3] = pos_tgt
            target_se3 = pin.SE3(rot_tgt, pos_tgt)

            ft = pink.tasks.FrameTask(
                _INTERNAL_TIP_FRAME,
                position_cost=1.0,
                orientation_cost=float(rot_weight),
                lm_damping=1e-3,
            )
            ft.set_target(target_se3)
            pt_cost = float(posture_weight) if (posture_vec is not None and posture_weight > 0.0) else 5e-4
            pt = pink.tasks.PostureTask(cost=pt_cost)
            lim = pink.limits.ConfigurationLimit(p_model, config_limit_gain=1.0)
            tasks = [ft, pt]

            cap_iters = int(max_iters)
            stall_tol = 1e-4 if float(max_step_rad) >= 0.05 else max(1e-9, 1e-4 * (float(max_step_rad) / 0.05))
            cfg = pink.Configuration(p_model, p_data, q_out[0].copy())
            converged_branches = 0
            q_pink = q_out.copy()
            for b in range(bsz):
                q_start = q_pink[b].copy()
                ref_b = ref[0] if ref.shape[0] == 1 else ref[b]
                pt_target = np.clip(
                    posture_vec if (posture_vec is not None and posture_weight > 0.0) else q_start,
                    lim_lo,
                    lim_hi,
                )
                pt.set_target(pt_target)
                cfg.update(q_start)
                prev_pe = float("inf")
                for step_i in range(cap_iters):
                    T_cur = cfg.get_transform_frame_to_world(_INTERNAL_TIP_FRAME).homogeneous
                    pe, oe = pose_error(T_cur, target_T)
                    if pe <= 0.5 * pos_tol_m and oe <= 0.5 * rot_tol_rad:
                        break
                    if step_i >= 6 and pe > 0.15 and abs(prev_pe - pe) < stall_tol:
                        break
                    prev_pe = pe
                    try:
                        vel = pink.solve_ik(
                            cfg,
                            tasks,
                            dt=1.0,
                            solver="daqp",
                            damping=1e-6,
                            limits=[lim],
                            safety_break=False,
                        )
                    except (
                        TypeError,
                        AttributeError,
                        KeyError,
                        IndexError,
                        NameError,
                        UnboundLocalError,
                        ImportError,
                        ZeroDivisionError,
                        AssertionError,
                        NotImplementedError,
                        RecursionError,
                        SyntaxError,
                        BufferError,
                    ):
                        raise
                    except Exception:
                        break
                    v_max = float(np.max(np.abs(vel)))
                    if v_max > max_step_rad:
                        vel = vel * (max_step_rad / v_max)
                    q_prev = cfg.q.copy()
                    cfg.integrate_inplace(vel, 1.0)
                    q_clamped = np.clip(cfg.q, lim_lo, lim_hi)
                    if not np.array_equal(q_clamped, cfg.q):
                        cfg.update(q_clamped)
                    if float(np.max(np.abs(cfg.q - q_prev))) < 1e-8:
                        break

                q_sol = cfg.q.copy()
                wrapped = ref_b + (((q_sol - ref_b) + math.pi) % math.tau - math.pi)
                wrap_ok = rotary_mask & (wrapped >= lim_lo) & (wrapped <= lim_hi)
                q_sol = np.where(wrap_ok, wrapped, q_sol)
                q_sol = np.clip(q_sol, lim_lo, lim_hi)
                q_pink[b] = q_sol

                T_fin = cfg.get_transform_frame_to_world(_INTERNAL_TIP_FRAME).homogeneous
                pe_f, oe_f = pose_error(T_fin, target_T)
                if pe_f <= pos_tol_m and oe_f <= rot_tol_rad:
                    converged_branches += 1
                    if converged_branches >= 3 and b >= 3:
                        break

            return q_pink
        except ImportError:
            _PINK_DAQP_CACHE = False
        except TypeError as exc:
            if not is_pink_compat_type_error(exc):
                raise
            _PINK_DAQP_CACHE = False

    return _solve_ik_batch_dls(
        chain,
        pos_tgt,
        rot_tgt,
        q_out,
        ref,
        pos_tol_m=pos_tol_m,
        rot_tol_rad=rot_tol_rad,
        rot_weight=rot_weight,
        max_iters=max_iters,
        max_step_rad=max_step_rad,
        posture_rad=posture_vec,
        posture_weight=posture_weight,
    )


__all__ = [
    "IKOptions",
    "IKResult",
    "IKTarget",
    "KinematicChainSpec",
    "chain_from_manifest",
    "fk_batch",
    "fk_jacobian",
    "fk_links",
    "fk_manifest",
    "ik_backend",
    "is_pink_compat_type_error",
    "pose_delta_vectors",
    "pose_error",
    "quat_wxyz_to_matrix",
    "quat_xyzw_to_matrix",
    "rigid_transform",
    "rodrigues",
    "rotation_log",
    "rotation_log_batch",
    "rpy_xyz_to_matrix",
    "solve_ik",
    "solve_ik_batch",
]
