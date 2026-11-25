"""Fast batched N-DoF reachability, homotopy seed generation, and link-capsule clearance over ``kinematics.py``.

Reads default ARX-X5 kinematics, mounts, and capsule radii from ``ni_rua.robot.embodiments.arx_x5``
(backed by ``arx_x5/physical.json``; or ``_x5_kinematics.json`` when copied into an isolated
``program_run`` workspace) and delegates all FK, Jacobian, rotation-log, and batched DLS IK
computations to ``ni_rua.robot.model.kinematics``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import functools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from ..embodiments.arx_x5 import X5_CAPSULE_RADII_M, arx_x5_chain, arx_x5_default_mounts
    from ..model.kinematics import (
        KinematicChainSpec,
        chain_from_manifest,
        fk_batch,
        quat_wxyz_to_matrix,
        rotation_log_batch,
        rpy_xyz_to_matrix,
        solve_ik_batch,
    )
else:
    from kinematics import (  # type: ignore[no-redef, import-not-found]
        KinematicChainSpec,
        chain_from_manifest,
        fk_batch,
        quat_wxyz_to_matrix,
        rotation_log_batch,
        rpy_xyz_to_matrix,
        solve_ik_batch,
    )

    X5_CAPSULE_RADII_M = (0.035, 0.032, 0.034, 0.029, 0.036, 0.020)

    @functools.lru_cache(maxsize=1)
    def _sandbox_x5_kinematics() -> dict[str, Any]:
        parent = Path(__file__).resolve().parent
        path = parent / "_kinematics.json"
        if not path.is_file():
            path = parent / "_x5_kinematics.json"
        return json.loads(path.read_text(encoding="utf-8"))

    @functools.lru_cache(maxsize=1)
    def arx_x5_chain() -> KinematicChainSpec:
        kin = _sandbox_x5_kinematics()
        base_link = str(kin.get("base_link") or "base_link")
        return chain_from_manifest(
            {"base_link": base_link, "joints": kin["joints"]},
            joint_names=kin["arm_joint_order"],
            tip_link=kin["end_effector_link"],
            base_link=base_link,
        )

    @functools.lru_cache(maxsize=1)
    def arx_x5_default_mounts() -> dict[str, np.ndarray]:
        mounts: dict[str, np.ndarray] = {}
        for item in _sandbox_x5_kinematics().get("default_mounts") or ():
            tf = np.eye(4, dtype=np.float64)
            tf[:3, :3] = quat_wxyz_to_matrix(item["quaternion_wxyz"])
            tf[:3, 3] = [float(v) for v in item["position_environment_m"]]
            mounts[str(item["arm"])] = tf
        return mounts


_CAPSULE_RADII_M = np.asarray(X5_CAPSULE_RADII_M, dtype=np.float64)
_CAPSULE_NAMES = ("base_shoulder", "upper_arm", "forearm", "wrist", "gripper_hand", "wrist_camera")
_WRIST_CAM_P0_LINK6_M = np.array([0.057, 0.0, 0.008], dtype=np.float64)
_WRIST_CAM_P1_LINK6_M = np.array([0.0845, 0.0, 0.052], dtype=np.float64)
_SELF_COLLISION_PAIRS = ((0, 3), (0, 4), (0, 5), (1, 4), (1, 5), (2, 4))
_SELF_COLLISION_PAIR_I = np.array([p[0] for p in _SELF_COLLISION_PAIRS], dtype=np.int64)
_SELF_COLLISION_PAIR_J = np.array([p[1] for p in _SELF_COLLISION_PAIRS], dtype=np.int64)
_SELF_COLLISION_BUFFER_M = -0.008

_MANIFOLD_SIGN_LATTICE_6D = np.array(
    [
        [1.0, 1.0, 1.0, -1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0, 1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0, 1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0, -1.0, -1.0, 1.0],
        [-1.0, -1.0, -1.0, -1.0, 1.0, -1.0],
        [0.5, 1.0, 0.5, -0.5, 0.0, 0.0],
        [-0.5, 1.0, 0.5, -0.5, 0.0, 0.0],
        [0.0, -0.5, -0.5, 0.5, 0.5, 0.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class Serial6RChainSpec:
    """Compatibility wrapper over ``KinematicChainSpec`` adding link collision capsule radii (``m``)."""

    joint_names: tuple[str, ...]
    parent_rots: np.ndarray
    parent_pos_m: np.ndarray
    joint_axes_unit: np.ndarray
    joint_limits_rad: np.ndarray
    flange_fixed_transform: np.ndarray
    base_pose_world: np.ndarray | None = None
    capsule_radii_m: np.ndarray = field(default_factory=lambda: _CAPSULE_RADII_M.copy())
    joint_types: tuple[str, ...] = field(default_factory=lambda: ("revolute",) * 6)

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    @property
    def max_reach_m(self) -> float:
        flange_len = float(np.linalg.norm(self.flange_fixed_transform[:3, 3]))
        return float(np.sum(np.linalg.norm(self.parent_pos_m[1:], axis=1))) + flange_len + 0.05

    def to_kinematic_chain(self) -> KinematicChainSpec:
        cached = getattr(self, "_cached_k_chain", None)
        if cached is not None:
            return cached
        n = len(self.joint_names)
        j_types = tuple(self.joint_types) if len(self.joint_types) == n else ("revolute",) * n
        parent_tfs = np.broadcast_to(np.eye(4, dtype=np.float64), (n, 4, 4)).copy()
        parent_tfs[:, :3, :3] = self.parent_rots
        parent_tfs[:, :3, 3] = self.parent_pos_m
        chain = KinematicChainSpec(
            joint_names=self.joint_names,
            joint_types=j_types,
            parent_transforms=parent_tfs,
            joint_axes=self.joint_axes_unit,
            joint_limits_rad=self.joint_limits_rad,
            tip_transform=self.flange_fixed_transform,
            T_world_base=self.base_pose_world,
        )
        object.__setattr__(self, "_cached_k_chain", chain)
        return chain

    @classmethod
    def from_kinematic_chain(
        cls,
        chain: KinematicChainSpec,
        *,
        capsule_radii_m: np.ndarray | None = None,
    ) -> Serial6RChainSpec:
        inst = cls(
            joint_names=chain.joint_names,
            parent_rots=chain.parent_rots.copy(),
            parent_pos_m=chain.parent_pos_m.copy(),
            joint_axes_unit=chain.joint_axes.copy(),
            joint_limits_rad=chain.joint_limits_rad.copy(),
            flange_fixed_transform=chain.tip_transform.copy(),
            base_pose_world=chain.T_world_base.copy() if chain.T_world_base is not None else None,
            capsule_radii_m=(
                np.asarray(capsule_radii_m, dtype=np.float64).copy()
                if capsule_radii_m is not None
                else _CAPSULE_RADII_M.copy()
            ),
            joint_types=tuple(chain.joint_types),
        )
        object.__setattr__(inst, "_cached_k_chain", chain)
        return inst


@functools.lru_cache(maxsize=1)
def default_arx_x5_chain_spec() -> Serial6RChainSpec:
    """Build the default ARX-X5 ``Serial6RChainSpec`` from ``arx_x5/physical.json``."""
    return Serial6RChainSpec.from_kinematic_chain(arx_x5_chain())


def resolve_6r_chain_spec(
    arm: str = "right",
    robot_model: Mapping[str, Any] | Serial6RChainSpec | KinematicChainSpec | None = None,
    joint_limits_rad: Any = None,
) -> Serial6RChainSpec:
    """从运行时 `robot_model` 动态解析指定机械臂的 6R 串联运动学链与限位；省略时回退至 `arx_x5/physical.json`。

    参数:
        arm: 机械臂侧名称（`'left'` 或 `'right'`），默认 `'right'`。
        robot_model: 可选运行时机器人运动学模型字典、`Serial6RChainSpec` 或 `KinematicChainSpec`；省略时使用默认 ARX X5 硬件规格（平移单位 `m`）。
        joint_limits_rad: 可选显式覆写的 `(6, 2)` 关节角上下限数组，单位 `rad`。

    返回:
        `Serial6RChainSpec` 实例（包含连杆固定变换 `m`、关节单位旋转轴与关节限位 `rad`）。

    示例:
        `spec = resolve_6r_chain_spec('right', robot_model=runtime_model)`
    """
    if isinstance(robot_model, Serial6RChainSpec):
        base_spec = robot_model
    elif isinstance(robot_model, KinematicChainSpec):
        base_spec = Serial6RChainSpec.from_kinematic_chain(robot_model)
    elif robot_model is None:
        base_spec = default_arx_x5_chain_spec()
    else:
        if not isinstance(robot_model, Mapping):
            raise TypeError("robot_model_must_be_mapping_or_Serial6RChainSpec")
        arms_sub = robot_model.get("arms")
        sub_model: Mapping[str, Any] = (
            arms_sub[arm]
            if isinstance(arms_sub, Mapping) and isinstance(arms_sub.get(arm), Mapping)
            else robot_model
        )
        flat_model = dict(robot_model)
        flat_model.update(sub_model)
        raw_j = flat_model.get("joints")
        child_links = {
            str(j["child"])
            for j in (raw_j if isinstance(raw_j, (list, tuple)) else ())
            if isinstance(j, Mapping) and "child" in j
        }
        for tip_key in ("end_effector_link", "tcp_link"):
            tip_val = flat_model.get(tip_key)
            if tip_val is not None and str(tip_val) not in child_links:
                flat_model.pop(tip_key, None)
        try:
            k_chain = chain_from_manifest(
                flat_model,
                arm_prefix=f"{arm}_",
                expected_dof=6,
            )
        except ValueError as exc:
            msg = str(exc)
            if msg == "robot_model_has_no_active_joints":
                raise ValueError("robot_model_requires_6_revolute_joints_got_0") from None
            if msg.startswith("robot_model_requires_6_active_joints_got_"):
                cnt = msg.rsplit("_", 1)[-1]
                raise ValueError(f"robot_model_requires_6_revolute_joints_got_{cnt}") from None
            raise
        cap_radii = (
            np.asarray(sub_model["capsule_radii_m"], dtype=np.float64).reshape(6)
            if sub_model.get("capsule_radii_m") is not None
            else _CAPSULE_RADII_M.copy()
        )
        base_spec = Serial6RChainSpec.from_kinematic_chain(k_chain, capsule_radii_m=cap_radii)

    if joint_limits_rad is not None:
        lims = np.asarray(joint_limits_rad, dtype=np.float64).reshape(base_spec.dof, 2)
        if not np.isfinite(lims).all() or np.any(lims[:, 0] > lims[:, 1]):
            raise ValueError("invalid_joint_limits_rad")
        return Serial6RChainSpec(
            joint_names=base_spec.joint_names,
            parent_rots=base_spec.parent_rots,
            parent_pos_m=base_spec.parent_pos_m,
            joint_axes_unit=base_spec.joint_axes_unit,
            joint_limits_rad=lims,
            flange_fixed_transform=base_spec.flange_fixed_transform,
            base_pose_world=base_spec.base_pose_world,
            capsule_radii_m=base_spec.capsule_radii_m,
            joint_types=base_spec.joint_types,
        )
    if not np.isfinite(base_spec.joint_limits_rad).all() or np.any(
        base_spec.joint_limits_rad[:, 0] > base_spec.joint_limits_rad[:, 1]
    ):
        raise ValueError("invalid_joint_limits_rad")
    return base_spec


def fk_batch_base(
    q_batch_rad: np.ndarray,
    chain_spec: Serial6RChainSpec | KinematicChainSpec | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """在单臂基座系 (`base_link`) 下经 `kinematics.fk_batch` 批量计算正运动学与解析雅可比。

    参数:
        q_batch_rad: 形状 `(B, 6)` 或 `(6,)` 的关节角矩阵，单位 `rad`。
        chain_spec: 可选 `Serial6RChainSpec` 或 `KinematicChainSpec`；默认采用 ARX X5 硬件链。

    返回:
        四元组 `(pos_link6_m, rot_link6, jac_base, link_origins_m)`（位置单位 `m`，雅可比线速度偏导单位 `m/rad`）。

    示例:
        `pos_m, rot, jac, origins_m = fk_batch_base(np.zeros((1, 6)))`
    """
    if chain_spec is None:
        k_chain = arx_x5_chain()
    elif isinstance(chain_spec, KinematicChainSpec):
        k_chain = chain_spec
    else:
        k_chain = chain_spec.to_kinematic_chain()
    return fk_batch(k_chain, q_batch_rad)


def _finite_vec_checked(values: Any, size: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (size,) or not np.isfinite(arr).all():
        raise ValueError(f"invalid_{name}: expected {size} finite numbers")
    return arr


def resolve_arm_base_transform(
    arm: str,
    robot_state: dict[str, Any] | None = None,
    *,
    robot_model: Mapping[str, Any] | Serial6RChainSpec | KinematicChainSpec | None = None,
) -> np.ndarray:
    """解析指定机械臂的 `T_world_base` (`4x4` 齐次刚体变换矩阵，平移单位 `m`)。

    参数:
        arm: 机械臂侧名称（`'left'` 或 `'right'`）。
        robot_state: 可选实测机器人状态字典（含 `q_rad` `rad` 与 `ee_pose.position_m` `m`）。
        robot_model: 可选运行时运动学模型字典或 `Serial6RChainSpec`。

    返回:
        形状 `(4, 4)` 的 `T_world_base` 齐次矩阵，平移单位 `m`。

    示例:
        `T_wb = resolve_arm_base_transform('right', robot_state=rs)`
    """
    mounts = arx_x5_default_mounts()
    if arm not in mounts and (not isinstance(robot_state, dict) or arm not in (robot_state.get("arms") or {})):
        raise ValueError(f"invalid_arm: {arm!r}; expected 'left' or 'right'")
    chain = resolve_6r_chain_spec(arm, robot_model=robot_model)
    dof = chain.dof
    if isinstance(robot_state, dict):
        arm_st = (robot_state.get("arms") or {}).get(arm)
        if isinstance(arm_st, dict):
            q_raw = arm_st.get("q_rad")
            ee_raw = arm_st.get("ee_pose")
            if (
                isinstance(q_raw, (list, tuple))
                and len(q_raw) == dof
                and isinstance(ee_raw, dict)
                and isinstance(ee_raw.get("position_m"), (list, tuple))
                and isinstance(ee_raw.get("quaternion_xyzw"), (list, tuple))
            ):
                q_arr = _finite_vec_checked(q_raw, dof, "q_rad")
                ee_pos = _finite_vec_checked(ee_raw["position_m"], 3, "ee_position_m")
                qx, qy, qz, qw = _finite_vec_checked(ee_raw["quaternion_xyzw"], 4, "quaternion_xyzw")
                rot_wl6 = quat_wxyz_to_matrix(np.array([qw, qx, qy, qz], dtype=np.float64))
                t_wl6 = np.eye(4, dtype=np.float64)
                t_wl6[:3, :3] = rot_wl6
                t_wl6[:3, 3] = ee_pos

                pos_bl6, rot_bl6, _, _ = fk_batch_base(q_arr[None, :], chain_spec=chain)
                t_bl6_inv = np.eye(4, dtype=np.float64)
                r_t = rot_bl6[0].T
                t_bl6_inv[:3, :3] = r_t
                t_bl6_inv[:3, 3] = -(r_t @ pos_bl6[0])
                return t_wl6 @ t_bl6_inv

    if chain.base_pose_world is not None:
        return chain.base_pose_world.copy()
    return mounts[arm].copy()


def build_manifold_perturbation_seeds(
    q_anchor: np.ndarray,
    lim_lo: np.ndarray,
    lim_hi: np.ndarray,
) -> np.ndarray:
    """Construct ``(12, N)`` deterministic orthogonal perturbation seeds around ``q_anchor`` (``rad``)."""
    dof = int(q_anchor.shape[0])
    if dof == 6:
        lattice = _MANIFOLD_SIGN_LATTICE_6D
    elif dof < 6:
        lattice = _MANIFOLD_SIGN_LATTICE_6D[:, :dof]
    else:
        pad = np.zeros((len(_MANIFOLD_SIGN_LATTICE_6D), dof - 6), dtype=np.float64)
        for j in range(dof - 6):
            pad[:, j] = _MANIFOLD_SIGN_LATTICE_6D[:, j % 6] * 0.5
        lattice = np.hstack([_MANIFOLD_SIGN_LATTICE_6D, pad])

    half_span = np.minimum(0.5 * (lim_hi - lim_lo), math.pi)
    offsets = np.empty((len(lattice), dof), dtype=np.float64)
    offsets[:4] = 0.32 * lattice[:4] * half_span[None, :]
    offsets[4:8] = 0.68 * lattice[4:8] * half_span[None, :]
    offsets[8:] = 0.50 * lattice[8:] * half_span[None, :]

    perturbed = q_anchor[None, :] + offsets
    wrapped = q_anchor[None, :] + (((perturbed - q_anchor[None, :]) + math.pi) % math.tau - math.pi)
    in_bounds = (wrapped >= lim_lo[None, :]) & (wrapped <= lim_hi[None, :])
    perturbed = np.where(in_bounds, wrapped, np.clip(perturbed, lim_lo[None, :], lim_hi[None, :]))
    return np.vstack([q_anchor[None, :], perturbed])


def solve_dls_branches(
    seeds: np.ndarray,
    pos_base_target: np.ndarray,
    rot_base_target: np.ndarray,
    lim_lo: np.ndarray,
    lim_hi: np.ndarray,
    wrap_ref: np.ndarray,
    pos_tol_m: float,
    rot_tol_rad: float,
    max_iters: int,
    chain_spec: Serial6RChainSpec | KinematicChainSpec,
) -> np.ndarray:
    """Run vectorized multi-seed analytical-Jacobian DLS IK via ``kinematics.solve_ik_batch``."""
    k_chain = chain_spec if isinstance(chain_spec, KinematicChainSpec) else chain_spec.to_kinematic_chain()
    cur_lims = k_chain.joint_limits_rad
    if not (np.array_equal(cur_lims[:, 0], lim_lo) and np.array_equal(cur_lims[:, 1], lim_hi)):
        k_chain = k_chain.with_limits(np.column_stack([lim_lo, lim_hi]))
    return solve_ik_batch(
        k_chain,
        pos_base_target,
        rot_base_target,
        seeds,
        wrap_ref=wrap_ref,
        pos_tol_m=pos_tol_m,
        rot_tol_rad=rot_tol_rad,
        max_iters=max_iters,
    )


def capsules_from_fk_base(
    rot_l6: np.ndarray,
    origins_b: np.ndarray,
    t_world_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform ``base_link`` FK origins into world-frame 6-capsule segment endpoints ``(p0_m, p1_m)`` (``m``)."""
    r_wb = t_world_base[:3, :3]
    p_wb = t_world_base[:3, 3]
    origins_w = (origins_b @ r_wb.T) + p_wb[None, None, :]
    rot_wl6 = r_wb[None, :, :] @ rot_l6
    pos_wl6 = origins_w[:, -1, :]
    tool_x = rot_wl6[:, :, 0]

    p0 = np.empty((origins_w.shape[0], 6, 3), dtype=np.float64)
    p1 = np.empty((origins_w.shape[0], 6, 3), dtype=np.float64)
    p0[:, 0, :], p1[:, 0, :] = origins_w[:, 0, :], origins_w[:, 2, :]
    p0[:, 1, :], p1[:, 1, :] = origins_w[:, 2, :], origins_w[:, 3, :]
    p0[:, 2, :], p1[:, 2, :] = origins_w[:, 3, :], origins_w[:, 4, :]
    p0[:, 3, :], p1[:, 3, :] = origins_w[:, 4, :], pos_wl6
    p0[:, 4, :] = pos_wl6 + 0.020 * tool_x
    p1[:, 4, :] = pos_wl6 + 0.1576 * tool_x
    p0[:, 5, :] = pos_wl6 + (rot_wl6 @ _WRIST_CAM_P0_LINK6_M)
    p1[:, 5, :] = pos_wl6 + (rot_wl6 @ _WRIST_CAM_P1_LINK6_M)
    return p0, p1


def arm_capsules_world(
    q_batch_rad: np.ndarray,
    t_world_base: np.ndarray,
    *,
    chain_spec: Serial6RChainSpec | KinematicChainSpec | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute world-frame 6-capsule endpoints ``(p0_m, p1_m)`` of shape ``(B, 6, 3)`` in ``m``."""
    _, rot_l6, _, origins_b = fk_batch_base(q_batch_rad, chain_spec=chain_spec)
    return capsules_from_fk_base(rot_l6, origins_b, t_world_base)


def segment_pair_distances(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> np.ndarray:
    """Vectorized minimum Euclidean distance between line segments ``(a0->a1)`` and ``(b0->b1)`` in ``m``."""
    u = a1 - a0
    v = b1 - b0
    w = a0 - b0
    uu = np.sum(u * u, axis=-1)
    vv = np.sum(v * v, axis=-1)
    uv = np.sum(u * v, axis=-1)
    uw = np.sum(u * w, axis=-1)
    vw = np.sum(v * w, axis=-1)
    denom = uu * vv - uv * uv

    s = np.where(denom > 1e-12, np.clip((uv * vw - vv * uw) / np.maximum(denom, 1e-12), 0.0, 1.0), 0.0)
    t = np.where(vv > 1e-12, np.clip((uv * s + vw) / np.maximum(vv, 1e-12), 0.0, 1.0), 0.0)
    s = np.where(uu > 1e-12, np.clip((uv * t - uw) / np.maximum(uu, 1e-12), 0.0, 1.0), 0.0)
    t = np.where(vv > 1e-12, np.clip((uv * s + vw) / np.maximum(vv, 1e-12), 0.0, 1.0), 0.0)

    diff = w + s[..., None] * u - t[..., None] * v
    return np.linalg.norm(diff, axis=-1)


def self_and_table_clearances(
    p0_w: np.ndarray,
    p1_w: np.ndarray,
    table_z_m: float | None,
    capsule_radii_m: np.ndarray = _CAPSULE_RADII_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute non-adjacent self-collision clearance and optional support-plane clearance in ``m``."""
    seg_dist = segment_pair_distances(
        p0_w[:, _SELF_COLLISION_PAIR_I, :],
        p1_w[:, _SELF_COLLISION_PAIR_I, :],
        p0_w[:, _SELF_COLLISION_PAIR_J, :],
        p1_w[:, _SELF_COLLISION_PAIR_J, :],
    )
    radii_sum = (
        capsule_radii_m[_SELF_COLLISION_PAIR_I]
        + capsule_radii_m[_SELF_COLLISION_PAIR_J]
        + _SELF_COLLISION_BUFFER_M
    )
    min_self_m = np.min(seg_dist - radii_sum[None, :], axis=1)

    if table_z_m is None:
        min_table_m = np.full(p0_w.shape[0], np.inf, dtype=np.float64)
    else:
        struct_pts_z = np.stack(
            [p1_w[:, 1, 2], p1_w[:, 2, 2], p1_w[:, 3, 2], p0_w[:, 5, 2], p1_w[:, 5, 2]],
            axis=1,
        )
        struct_radii = np.array([0.022, 0.020, 0.018, 0.016, 0.016], dtype=np.float64)
        min_table_m = np.min(struct_pts_z - struct_radii[None, :] - float(table_z_m), axis=1)
    return min_self_m, min_table_m


__all__ = [
    "Serial6RChainSpec",
    "_CAPSULE_NAMES",
    "_CAPSULE_RADII_M",
    "_SELF_COLLISION_PAIRS",
    "arm_capsules_world",
    "build_manifold_perturbation_seeds",
    "capsules_from_fk_base",
    "default_arx_x5_chain_spec",
    "fk_batch_base",
    "resolve_6r_chain_spec",
    "resolve_arm_base_transform",
    "rotation_log_batch",
    "rpy_xyz_to_matrix",
    "segment_pair_distances",
    "self_and_table_clearances",
    "solve_dls_branches",
]
