"""通用 6R 串联机械臂解析正逆运动学、关节行程准入与连杆胶囊净空核（`ik_prefilter.py`）。

核心职责：
1. 纯 NumPy 向量化解析雅可比阻尼最小二乘逆运动学（Batched Analytical-Jacobian DLS IK）：
   以当前实测关节角 `q_seed` 为中心结合均匀确定性正交流形扰动并行求解，无任何面向桌面俯抓的硬编码姿态种子。
2. 支持从运行时 `robot_model` 动态解析任意 6R 串联机械臂的连杆固定变换、任意三维单位旋转轴与关节上下限；
   未传入 `robot_model` 时回退至公开披露的 ARX X5 URDF 硬件几何真值。
3. 零默认桌面高度假设（`table_z_m: float | None = None`），仅当调用方显式给出支撑面高度时检查结构连杆穿透。
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .pose_math import (
        CONTACT_DEPTH_PRESETS_M,
        GRASP_POINT_OFFSET_M,
        angles_deg_to_quat_wxyz,
        quat_multiply,
        quat_slerp_wxyz,
        quat_wxyz_to_angles_deg,
        quat_wxyz_to_matrix,
    )
else:
    from pose_math import (  # type: ignore[no-redef]
        CONTACT_DEPTH_PRESETS_M,
        GRASP_POINT_OFFSET_M,
        angles_deg_to_quat_wxyz,
        quat_multiply,
        quat_slerp_wxyz,
        quat_wxyz_to_angles_deg,
        quat_wxyz_to_matrix,
    )

ARMS = ("left", "right")
JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")

CANONICAL_ANGLE_BOUNDS_DEG = (
    (-180.0, 180.0),
    (-90.0, 90.0),
    (-180.0, 180.0),
)

MIN_CONTACT_DEPTH_M = -0.0075
MAX_CONTACT_DEPTH_M = 0.0625

JOINT_TRAVEL_PRESETS_RAD = {
    "standard": (math.pi, 0.5 * math.pi),
    "extended": (1.5 * math.pi, math.pi),
    "unconstrained": (None, None),
}

if __package__:
    from ...robot.model.kinematics import solve_ik_batch as _solve_pink_ik_batch
    from ...robot.motion.reachability import (
        Serial6RChainSpec,
        _CAPSULE_NAMES,
        _CAPSULE_RADII_M,
        _SELF_COLLISION_PAIRS,
        arm_capsules_world,
        build_manifold_perturbation_seeds as _build_manifold_perturbation_seeds,
        capsules_from_fk_base as _capsules_from_fk_base,
        default_arx_x5_chain_spec,
        fk_batch_base,
        resolve_6r_chain_spec,
        resolve_arm_base_transform,
        rotation_log_batch as _rotation_error_vec_base,
        rpy_xyz_to_matrix as _rpy_xyz_to_matrix,
        segment_pair_distances as _segment_pair_distances,
        self_and_table_clearances as _self_and_table_clearances,
        solve_dls_branches as _solve_dls_branches,
    )
else:
    from kinematics import solve_ik_batch as _solve_pink_ik_batch  # type: ignore[no-redef, import-not-found]
    from _reachability import (  # type: ignore[no-redef, import-not-found]
        Serial6RChainSpec,
        _CAPSULE_NAMES,
        _CAPSULE_RADII_M,
        _SELF_COLLISION_PAIRS,
        arm_capsules_world,
        build_manifold_perturbation_seeds as _build_manifold_perturbation_seeds,
        capsules_from_fk_base as _capsules_from_fk_base,
        default_arx_x5_chain_spec,
        fk_batch_base,
        resolve_6r_chain_spec,
        resolve_arm_base_transform,
        rotation_log_batch as _rotation_error_vec_base,
        rpy_xyz_to_matrix as _rpy_xyz_to_matrix,
        segment_pair_distances as _segment_pair_distances,
        self_and_table_clearances as _self_and_table_clearances,
        solve_dls_branches as _solve_dls_branches,
    )

_ARX_X5_CHAIN_SPEC_CACHE: Serial6RChainSpec | None = None


def _default_x5_chain_spec() -> Serial6RChainSpec:
    global _ARX_X5_CHAIN_SPEC_CACHE
    if _ARX_X5_CHAIN_SPEC_CACHE is None:
        _ARX_X5_CHAIN_SPEC_CACHE = default_arx_x5_chain_spec()
    return _ARX_X5_CHAIN_SPEC_CACHE


def __getattr__(name: str) -> Any:
    if name == "_ARX_X5_CHAIN_SPEC":
        return _default_x5_chain_spec()
    if name == "DEFAULT_JOINT_LIMITS_RAD":
        return _default_x5_chain_spec().joint_limits_rad.copy()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _finite_vec(values: Any, size: int, name: str) -> np.ndarray:
    """校验并返回指定长度的有限浮点一维向量，非法输入直接抛出 `ValueError`。"""
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (size,) or not np.isfinite(arr).all():
        raise ValueError(f"invalid_{name}: expected {size} finite numbers")
    return arr


def _resolve_contact_offset_m(contact_region: str = "pad_center", contact_depth_m: float | None = None) -> float:
    """解析夹爪接触截面相对 `link6` 法兰沿局部 `+X` 轴的轴向偏移距离（单位 `m`）。"""
    if contact_depth_m is not None:
        if type(contact_depth_m) not in (int, float) or not math.isfinite(contact_depth_m):
            raise ValueError("invalid_contact_depth_m: expected finite number")
        depth = float(contact_depth_m)
        if not MIN_CONTACT_DEPTH_M <= depth <= MAX_CONTACT_DEPTH_M:
            raise ValueError(f"contact_depth_m_out_of_range: {depth}")
        return float(GRASP_POINT_OFFSET_M - depth)
    if contact_region not in CONTACT_DEPTH_PRESETS_M:
        raise ValueError(f"invalid_contact_region: {contact_region!r}")
    return float(GRASP_POINT_OFFSET_M - CONTACT_DEPTH_PRESETS_M[contact_region])


_WORKSPACE_SNAP_PATH = Path(__file__).resolve().parent / "_spatial_snapshot.json"
_SNAPSHOT_CACHE: tuple[int | None, dict[str, Any] | None] | None = None


def _autoload_workspace_robot_state() -> dict[str, Any] | None:
    """在沙盒工作区内按文件修改时间缓存读取预载 `_spatial_snapshot.json` 中的实测 `robot_state`（不注入默认桌面高度）。"""
    global _SNAPSHOT_CACHE
    if not _WORKSPACE_SNAP_PATH.is_file():
        _SNAPSHOT_CACHE = (None, None)
        return None
    try:
        st_mtime = _WORKSPACE_SNAP_PATH.stat().st_mtime_ns
    except OSError:
        _SNAPSHOT_CACHE = (None, None)
        return None
    if _SNAPSHOT_CACHE is not None and _SNAPSHOT_CACHE[0] == st_mtime:
        return _SNAPSHOT_CACHE[1]
    _SNAPSHOT_CACHE = (None, None)
    try:
        snap = json.loads(_WORKSPACE_SNAP_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    rs = snap.get("robot_state") if isinstance(snap, dict) else None
    parsed_rs = rs if isinstance(rs, dict) else None
    _SNAPSHOT_CACHE = (st_mtime, parsed_rs)
    return parsed_rs


def check_pose_reachability(
    arm: str,
    position_m: Any,
    angles_deg: Any = None,
    *,
    quaternion_xyzw: Any = None,
    pose_kind: str = "grasp",
    contact_region: str = "pad_center",
    contact_depth_m: float | None = None,
    q_seed: Any = None,
    robot_model: Mapping[str, Any] | Serial6RChainSpec | None = None,
    robot_state: dict[str, Any] | None = None,
    base_pose_world: np.ndarray | None = None,
    joint_limits_rad: Any = None,
    table_z_m: float | None = None,
    pos_tol_m: float = 0.002,
    rot_tol_rad: float = 0.02,
) -> dict[str, Any]:
    """检验单臂三维目标位姿的 6R 逆运动学可达性、关节限位裕度、单臂自碰撞及可选支撑面净空。

    参数:
        arm: 目标机械臂名称，取值 `'left'` 或 `'right'`。
        position_m: 目标三维坐标 `[x, y, z]`，单位 `m`（`env_origin` 世界系）。`pose_kind='grasp'` 时为夹爪接触参考点；
            `pose_kind='flange'` 时为 `link6` 法兰中心。
        angles_deg: 夹爪姿态角 `[pitch_deg, roll_deg, yaw_deg]`，单位 `deg`（规范域 `[-180,180] x [-90,90] x [-180,180]`）；
            `pose_kind='grasp'` 时必填。
        quaternion_xyzw: 可选 `link6` 法兰单位四元数 `[qx, qy, qz, qw]`（无量纲），`pose_kind='flange'` 时必填。
        pose_kind: 位姿语义模式，`'grasp'`（默认）或 `'flange'`。
        contact_region: 接触截面名称（`'pad_center'` | `'finger_middle'` | `'fingertip'`），默认 `'pad_center'`。
        contact_depth_m: 可选自定义接触截面深度偏移，单位 `m`（范围 `[-0.0075, 0.0625]`），默认 `None`。
        q_seed: 可选逆解初值 6 轴关节角 `[q1..q6]`，单位 `rad`；省略时从 `robot_state` 读取，再无则取关节限位中点并做均匀确定性流形扰动。
        robot_model: 可选运行时机器人运动学模型字典或 `Serial6RChainSpec`；省略时使用 ARX X5 URDF 硬件规格。
        robot_state: 可选实测机器人状态字典；省略时在沙盒内尝试读取预载 `_spatial_snapshot.json`。
        base_pose_world: 可选显式 `4x4` `T_world_base` 齐次变换矩阵（平移单位 `m`）。
        joint_limits_rad: 可选 `(6, 2)` 关节角上下限数组，单位 `rad`。
        table_z_m: 可选显式支撑面高度，单位 `m`；默认 `None`（不假设任何固定桌面高度，仅显式给定时检查结构件潜入支撑面）。
        pos_tol_m: 运动学可达位置残差上限，单位 `m`（`> 0`），默认 `0.002 m`。
        rot_tol_rad: 运动学可达姿态残差上限，单位 `rad`（`> 0`），默认 `0.02 rad`。

    返回:
        字典 `{reachable, joint_q_rad, pos_err_m, rot_err_rad, min_joint_margin_rad, self_clearance_m, table_clearance_m, joint_travel_from_seed_rad, flange_position_m, reason}`。

    示例:
        `res = check_pose_reachability('right', [0.18, -0.15, 0.82], [0.0, 0.0, -30.0])`
    """
    if arm not in ARMS:
        raise ValueError(f"invalid_arm: {arm!r}; expected 'left' or 'right'")
    pos_in = _finite_vec(position_m, 3, "position_m")
    if pos_tol_m <= 0 or rot_tol_rad <= 0:
        raise ValueError("pos_tol_m and rot_tol_rad must be positive")
    eff_table_z: float | None = None
    if table_z_m is not None:
        eff_table_z = float(table_z_m)
        if not math.isfinite(eff_table_z):
            raise ValueError("invalid_table_z_m: expected finite number or None")

    eff_rs = (
        robot_state
        if robot_state is not None
        else (_autoload_workspace_robot_state() if (base_pose_world is None or q_seed is None) else None)
    )
    chain = resolve_6r_chain_spec(arm, robot_model=robot_model, joint_limits_rad=joint_limits_rad)

    if pose_kind == "grasp":
        if angles_deg is None:
            raise ValueError("angles_deg is required when pose_kind='grasp'")
        ang = _finite_vec(angles_deg, 3, "angles_deg")
        for idx, (lo, hi) in enumerate(CANONICAL_ANGLE_BOUNDS_DEG):
            if not (lo - 1e-7 <= ang[idx] <= hi + 1e-7):
                return dict(
                    reachable=False,
                    joint_q_rad=None,
                    pos_err_m=None,
                    rot_err_rad=None,
                    min_joint_margin_rad=None,
                    self_clearance_m=None,
                    table_clearance_m=None,
                    joint_travel_from_seed_rad=None,
                    flange_position_m=None,
                    reason="angles_out_of_canonical_bounds",
                )
        quat_wxyz = angles_deg_to_quat_wxyz(float(ang[0]), float(ang[1]), float(ang[2]))
        rot_world_target = quat_wxyz_to_matrix(quat_wxyz)
        offset_m = _resolve_contact_offset_m(contact_region=contact_region, contact_depth_m=contact_depth_m)
        flange_world_target = pos_in - offset_m * rot_world_target[:, 0]
    elif pose_kind == "flange":
        if quaternion_xyzw is None:
            raise ValueError("quaternion_xyzw is required when pose_kind='flange'")
        q_xyzw = _finite_vec(quaternion_xyzw, 4, "quaternion_xyzw")
        q_norm = float(np.linalg.norm(q_xyzw))
        if q_norm <= 1e-8:
            raise ValueError("quaternion_xyzw must have non-zero norm")
        q_xyzw = q_xyzw / q_norm
        rot_world_target = quat_wxyz_to_matrix(np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64))
        flange_world_target = pos_in.copy()
    else:
        raise ValueError(f"invalid_pose_kind: {pose_kind!r}; expected 'grasp' or 'flange'")

    if base_pose_world is not None:
        t_wb = np.asarray(base_pose_world, dtype=np.float64)
        if t_wb.shape != (4, 4) or not np.isfinite(t_wb).all():
            raise ValueError("invalid_base_pose_world: expected finite 4x4 matrix")
    else:
        t_wb = resolve_arm_base_transform(arm, eff_rs, robot_model=chain)

    r_bw = t_wb[:3, :3].T
    pos_base_target = r_bw @ (flange_world_target - t_wb[:3, 3])
    rot_base_target = r_bw @ rot_world_target

    limits = chain.joint_limits_rad
    lim_lo = limits[:, 0]
    lim_hi = limits[:, 1]

    seed_vec: np.ndarray | None = None
    if q_seed is not None:
        seed_vec = _finite_vec(q_seed, 6, "q_seed")
    elif isinstance(eff_rs, dict):
        arm_q = ((eff_rs.get("arms") or {}).get(arm) or {}).get("q_rad")
        if isinstance(arm_q, (list, tuple)) and len(arm_q) == 6:
            seed_vec = _finite_vec(arm_q, 6, "q_rad")

    q_mid = 0.5 * (lim_lo + lim_hi)
    q_anchor = np.clip(seed_vec, lim_lo, lim_hi) if seed_vec is not None else q_mid
    wrap_ref = q_anchor[None, :]
    seeds = _build_manifold_perturbation_seeds(q_anchor, lim_lo, lim_hi)

    k_chain = chain.to_kinematic_chain()
    if not (np.array_equal(k_chain.joint_limits_rad[:, 0], lim_lo) and np.array_equal(k_chain.joint_limits_rad[:, 1], lim_hi)):
        k_chain = k_chain.with_limits(np.column_stack([lim_lo, lim_hi]))

    def _eval_candidates(cand_seeds: np.ndarray):
        qb = _solve_pink_ik_batch(
            k_chain,
            pos_base_target,
            rot_base_target,
            cand_seeds,
            wrap_ref=wrap_ref,
            pos_tol_m=pos_tol_m,
            rot_tol_rad=rot_tol_rad,
            max_iters=20,
        )
        p_c, r_c, _, orig_b = fk_batch_base(qb, chain_spec=chain)
        p_e = np.sqrt(np.sum((pos_base_target[None, :] - p_c) ** 2, axis=1))
        _, r_e = _rotation_error_vec_base(r_c, rot_base_target)
        m_lo = qb - lim_lo[None, :]
        m_hi = lim_hi[None, :] - qb
        min_m = np.minimum(np.min(m_lo, axis=1), np.min(m_hi, axis=1))
        p0w, p1w = _capsules_from_fk_base(r_c, orig_b, t_wb)
        s_clr, t_clr = _self_and_table_clearances(p0w, p1w, eff_table_z, capsule_radii_m=chain.capsule_radii_m)
        c_m = (p_e <= pos_tol_m) & (r_e <= rot_tol_rad)
        l_m = min_m >= -1e-6
        f_m = c_m & l_m & (s_clr >= 0.0) & (t_clr >= 0.0)
        return qb, orig_b, p_e, r_e, min_m, s_clr, t_clr, c_m, l_m, f_m

    # Fast primary pass on anchor + first homotopy flip (~0.45 ms); expand to full manifold lattice only if needed.
    (
        q_batch,
        origins_b,
        pos_errs,
        rot_errs,
        min_margins,
        self_clearances,
        table_clearances,
        conv_mask,
        lim_mask,
        feasible_mask,
    ) = _eval_candidates(seeds[:2])
    if not np.any(feasible_mask) and len(seeds) > 2:
        (
            qb2,
            orig2,
            pe2,
            re2,
            mm2,
            sc2,
            tc2,
            cm2,
            lm2,
            fm2,
        ) = _eval_candidates(seeds[2:])
        q_batch = np.vstack([q_batch, qb2])
        origins_b = np.vstack([origins_b, orig2])
        pos_errs = np.concatenate([pos_errs, pe2])
        rot_errs = np.concatenate([rot_errs, re2])
        min_margins = np.concatenate([min_margins, mm2])
        self_clearances = np.concatenate([self_clearances, sc2])
        table_clearances = np.concatenate([table_clearances, tc2])
        conv_mask = np.concatenate([conv_mask, cm2])
        lim_mask = np.concatenate([lim_mask, lm2])
        feasible_mask = np.concatenate([feasible_mask, fm2])

    table_mask = table_clearances >= 0.0

    ref_q = seed_vec if seed_vec is not None else q_mid
    travel_from_ref = np.sum(np.abs(q_batch - ref_q[None, :]), axis=1)

    if np.any(feasible_mask):
        cand_idx = np.flatnonzero(feasible_mask)
        best = int(cand_idx[int(np.argmin(travel_from_ref[cand_idx]))])
        reason = "ok"
        reachable = True
    else:
        if np.any(conv_mask):
            c_idx = np.flatnonzero(conv_mask)
            best = int(c_idx[int(np.argmin(travel_from_ref[c_idx]))])
            if not lim_mask[best]:
                reason = "joint_limit_exceeded"
            elif not table_mask[best]:
                reason = "arm_body_below_table"
            else:
                reason = "self_collision"
        else:
            best = int(np.argmin(pos_errs + 0.20 * rot_errs))
            j2_pos_b = origins_b[best, 2, :]
            dist_from_shoulder = float(np.linalg.norm(pos_base_target - j2_pos_b))
            reason = "workspace_out_of_reach" if dist_from_shoulder > chain.max_reach_m else "ik_unconverged"
        reachable = False

    q_sel = np.clip(q_batch[best], lim_lo, lim_hi) if reachable else q_batch[best]
    return dict(
        reachable=bool(reachable),
        joint_q_rad=[round(float(v), 6) for v in q_sel],
        pos_err_m=round(float(pos_errs[best]), 6),
        rot_err_rad=round(float(rot_errs[best]), 6),
        min_joint_margin_rad=round(float(min_margins[best]), 6),
        self_clearance_m=round(float(self_clearances[best]), 5),
        table_clearance_m=round(float(table_clearances[best]), 5) if eff_table_z is not None else None,
        joint_travel_from_seed_rad=round(float(np.max(np.abs(q_sel - ref_q))), 5) if seed_vec is not None else None,
        flange_position_m=[round(float(v), 5) for v in flange_world_target],
        reason=reason,
    )


def _check_joint_travel_admission(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    *,
    limits: np.ndarray,
    joint_names: tuple[str, ...] = JOINT_NAMES,
    joint_travel_preset: str | None = None,
    max_joint_travel_rad: float | None = None,
    max_backtracking_rad: float | None = None,
) -> tuple[bool, float, list[dict[str, Any]]]:
    """复刻 `motion_safety.py` 的单段关节行程与周期缠绕准入判定，返回 `(ok, max_travel_rad, witnesses)`。"""
    preset = "standard" if joint_travel_preset is None else str(joint_travel_preset)
    if preset not in JOINT_TRAVEL_PRESETS_RAD:
        raise ValueError(f"invalid_joint_travel_preset: {preset!r}")
    min_travel, min_excess = JOINT_TRAVEL_PRESETS_RAD[preset]
    if max_joint_travel_rad is not None:
        if type(max_joint_travel_rad) not in (int, float) or not math.isfinite(max_joint_travel_rad) or max_joint_travel_rad <= 0:
            raise ValueError("invalid_max_joint_travel_rad")
        min_travel = float(max_joint_travel_rad)
    if max_backtracking_rad is not None:
        if type(max_backtracking_rad) not in (int, float) or not math.isfinite(max_backtracking_rad) or max_backtracking_rad <= 0:
            raise ValueError("invalid_max_backtracking_rad")
        min_excess = float(max_backtracking_rad)

    net = q_goal - q_start
    travel = np.abs(net)
    max_step_travel = float(np.max(travel))
    witnesses: list[dict[str, Any]] = []
    if min_travel is None or min_excess is None:
        return True, max_step_travel, witnesses

    for i, j_name in enumerate(joint_names):
        wrapped = ((float(net[i]) + math.pi) % math.tau) - math.pi
        winding_excess = abs(float(net[i])) - abs(wrapped)
        equiv = float(q_start[i] + wrapped)
        lo, hi = float(limits[i, 0]), float(limits[i, 1])
        eligible = (lo - 1e-9 <= float(q_start[i]) <= hi + 1e-9) and (lo - 1e-9 <= equiv <= hi + 1e-9)
        if float(travel[i]) > min_travel + 1e-9 and eligible and winding_excess >= min_excess - 1e-9:
            witnesses.append(
                dict(
                    joint_name=j_name,
                    joint_index=i,
                    start_q_rad=round(float(q_start[i]), 5),
                    goal_q_rad=round(float(q_goal[i]), 5),
                    total_travel_rad=round(float(travel[i]), 5),
                    periodic_excess_rad=round(float(winding_excess), 5),
                )
            )
    return len(witnesses) == 0, max_step_travel, witnesses


def _build_coal_capsule(coal_mod: Any, p0: np.ndarray, p1: np.ndarray, radius_m: float) -> tuple[Any, Any]:
    diff = p1 - p0
    lz = float(np.linalg.norm(diff))
    r_val = float(radius_m)
    if lz <= 1e-9:
        return coal_mod.Sphere(r_val), coal_mod.Transform3s(np.eye(3, dtype=np.float64), p0)
    u = diff / lz
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64) if abs(float(u[0])) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_ax = np.cross(ref, u)
    x_ax /= float(np.linalg.norm(x_ax))
    y_ax = np.cross(u, x_ax)
    rot = np.column_stack([x_ax, y_ax, u])
    return coal_mod.Capsule(r_val, lz), coal_mod.Transform3s(rot, 0.5 * (p0 + p1))


def bimanual_capsule_clearance_m(
    q_left_batch: np.ndarray,
    q_right_batch: np.ndarray,
    t_wb_left: np.ndarray,
    t_wb_right: np.ndarray,
    *,
    left_chain: Serial6RChainSpec | None = None,
    right_chain: Serial6RChainSpec | None = None,
) -> tuple[float, tuple[str, str] | None]:
    """批量计算双臂同步轨迹采样点 `(S, 6)` 之间的 `6x6` `coal.Capsule` 包络精确有符号最小净空距离（单位 `m`）及最近连杆对名称。"""
    import coal

    lc = _default_x5_chain_spec() if left_chain is None else left_chain
    rc = _default_x5_chain_spec() if right_chain is None else right_chain
    p0_l, p1_l = arm_capsules_world(q_left_batch, t_wb_left, chain_spec=lc)
    p0_r, p1_r = arm_capsules_world(q_right_batch, t_wb_right, chain_spec=rc)
    req = coal.DistanceRequest(True)
    n_samples = p0_l.shape[0]
    min_clr = float("inf")
    best_pair: tuple[int, int] = (0, 0)

    for s_idx in range(n_samples):
        left_caps = [_build_coal_capsule(coal, p0_l[s_idx, i], p1_l[s_idx, i], float(lc.capsule_radii_m[i])) for i in range(6)]
        right_caps = [_build_coal_capsule(coal, p0_r[s_idx, j], p1_r[s_idx, j], float(rc.capsule_radii_m[j])) for j in range(6)]
        for i, (g_l, tf_l) in enumerate(left_caps):
            for j, (g_r, tf_r) in enumerate(right_caps):
                res = coal.DistanceResult()
                dist = float(coal.distance(g_l, tf_l, g_r, tf_r, req, res))
                if dist < min_clr:
                    min_clr = dist
                    best_pair = (i, j)

    return min_clr, (_CAPSULE_NAMES[best_pair[0]], _CAPSULE_NAMES[best_pair[1]])


def prefilter_stages_kinematics(
    stages: list[dict[str, Any]],
    initial_robot_state: dict[str, Any] | None = None,
    *,
    robot_model: Mapping[str, Any] | Serial6RChainSpec | None = None,
    joint_limits_rad: Any = None,
    table_z_m: float | None = None,
    pos_tol_m: float = 0.002,
    rot_tol_rad: float = 0.02,
    bimanual_min_clearance_m: float = 0.0,
    num_interp_samples: int = 7,
    batch_joint_travel_preset: str | None = None,
) -> dict[str, Any]:
    """对整批 `robot_action` 阶段执行 6R 逆运动学、关节行程准入、单臂自碰撞、双臂互碰撞与可选支撑面穿透核验。

    参数:
        stages: 标准 `robot_action` 阶段字典列表（支持 `kind='move' | 'gripper' | 'observe'` 及 `depends_on`）。
        initial_robot_state: 可选起始实测 `robot_state` 字典（含双臂 `q_rad` `rad` 与 `ee_pose` `m`）；
            省略时尝试从沙盒 `_spatial_snapshot.json` 预载。
        robot_model: 可选运行时机器人运动学模型字典或 `Serial6RChainSpec`；省略时使用 ARX X5 URDF 硬件规格。
        joint_limits_rad: 可选 `(6, 2)` 关节角上下限数组，单位 `rad`。
        table_z_m: 可选显式支撑面高度，单位 `m`；默认 `None`（仅显式提供时检查结构连杆潜入支撑面）。
        pos_tol_m: 逆解收敛位置容差，单位 `m`，默认 `0.002 m`。
        rot_tol_rad: 逆解收敛姿态容差，单位 `rad`，默认 `0.02 rad`。
        bimanual_min_clearance_m: 双臂连杆胶囊包络最小安全间距阈值，单位 `m`，默认 `0.0 m`。
        num_interp_samples: 每段关节空间轨迹插值碰撞采样点数（`>= 2`），默认 `7`。
        batch_joint_travel_preset: 批次级默认关节行程预设（`'standard'` | `'extended'` | `'unconstrained'`），默认 `None`。

    返回:
        字典 `{valid, blocking_stage_ids, reasons, min_bimanual_clearance_m, min_self_clearance_m, min_table_clearance_m, max_joint_step_travel_rad, stage_reports, terminal_joint_q_rad}`。

    示例:
        `pf = prefilter_stages_kinematics(stages)`
    """
    if not isinstance(stages, list) or not stages:
        raise ValueError("stages must be a non-empty list of stage dicts")
    n_samples = max(2, int(num_interp_samples))
    eff_table_z: float | None = None
    if table_z_m is not None:
        eff_table_z = float(table_z_m)
        if not math.isfinite(eff_table_z):
            raise ValueError("invalid_table_z_m: expected finite number or None")

    eff_rs = initial_robot_state if initial_robot_state is not None else _autoload_workspace_robot_state()
    chains = {
        arm: resolve_6r_chain_spec(arm, robot_model=robot_model, joint_limits_rad=joint_limits_rad)
        for arm in ARMS
    }
    t_wb = {arm: resolve_arm_base_transform(arm, eff_rs, robot_model=chains[arm]) for arm in ARMS}
    q_cur: dict[str, np.ndarray] = {}
    for arm in ARMS:
        arm_q = (((eff_rs or {}).get("arms") or {}).get(arm) or {}).get("q_rad") if isinstance(eff_rs, dict) else None
        if isinstance(arm_q, (list, tuple)) and len(arm_q) == 6:
            q_cur[arm] = _finite_vec(arm_q, 6, f"{arm}_q_rad")
        else:
            q_cur[arm] = 0.5 * (chains[arm].joint_limits_rad[:, 0] + chains[arm].joint_limits_rad[:, 1])

    alphas = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)[:, None]
    stage_reports: list[dict[str, Any]] = []
    blocking_ids: list[str] = []
    reason_set: list[str] = []
    min_self_all = float("inf")
    min_table_all = float("inf")
    min_bimanual_all = float("inf")
    max_step_travel_all = 0.0

    stage_finish_wave: dict[str, int] = {}
    arm_wave = {"left": 0, "right": 0}
    wave_segments: dict[int, dict[str, tuple[str, np.ndarray, np.ndarray]]] = {}
    wave_start_q: dict[int, dict[str, np.ndarray]] = {}

    for idx, st in enumerate(stages):
        if not isinstance(st, dict):
            raise ValueError(f"stage[{idx}] must be a dict")
        sid = str(st.get("id") or f"stage_{idx + 1}")
        kind = st.get("kind")
        if kind not in ("move", "gripper", "observe"):
            raise ValueError(f"unsupported_stage_kind: {kind!r} in stage {sid!r}")

        deps = st.get("depends_on") or ()
        dep_wave = max((stage_finish_wave.get(str(d), 0) for d in deps), default=0)

        if kind in ("gripper", "observe"):
            arm_opt = st.get("arm")
            if kind == "gripper" and arm_opt not in ARMS:
                raise ValueError(f"gripper stage {sid!r} requires valid arm ('left' or 'right')")
            sync_w = max(dep_wave, arm_wave[arm_opt] if arm_opt in ARMS else max(arm_wave.values()))
            stage_finish_wave[sid] = sync_w
            if arm_opt in ARMS:
                arm_wave[arm_opt] = sync_w
            else:
                arm_wave["left"] = sync_w
                arm_wave["right"] = sync_w
            stage_reports.append(dict(id=sid, kind=kind, arm=arm_opt, valid=True, reason="ok"))
            continue

        arm = st.get("arm")
        if arm not in ARMS:
            raise ValueError(f"move stage {sid!r} requires valid arm ('left' or 'right')")

        wave_idx = max(arm_wave[arm], dep_wave)
        wave_start_q.setdefault(wave_idx, {"left": q_cur["left"].copy(), "right": q_cur["right"].copy()})
        wave_start_q[wave_idx][arm] = q_cur[arm].copy()

        if "grasp_position_m" in st:
            reach = check_pose_reachability(
                arm=arm,
                position_m=st["grasp_position_m"],
                angles_deg=st.get("angles_deg"),
                pose_kind="grasp",
                contact_region=st.get("contact_region", "pad_center"),
                contact_depth_m=st.get("contact_depth_m"),
                q_seed=q_cur[arm],
                robot_model=chains[arm],
                robot_state=eff_rs,
                base_pose_world=t_wb[arm],
                table_z_m=eff_table_z,
                pos_tol_m=pos_tol_m,
                rot_tol_rad=rot_tol_rad,
            )
        elif "position_m" in st and "quaternion_xyzw" in st:
            reach = check_pose_reachability(
                arm=arm,
                position_m=st["position_m"],
                quaternion_xyzw=st["quaternion_xyzw"],
                pose_kind="flange",
                q_seed=q_cur[arm],
                robot_model=chains[arm],
                robot_state=eff_rs,
                base_pose_world=t_wb[arm],
                table_z_m=eff_table_z,
                pos_tol_m=pos_tol_m,
                rot_tol_rad=rot_tol_rad,
            )
        else:
            raise ValueError(f"move stage {sid!r} requires (grasp_position_m, angles_deg) or (position_m, quaternion_xyzw)")

        stage_valid = bool(reach["reachable"])
        stage_reason = str(reach["reason"])
        q_start_arm = q_cur[arm].copy()
        q_goal_arm = np.asarray(reach["joint_q_rad"], dtype=np.float64) if reach["joint_q_rad"] is not None else q_start_arm.copy()

        q_interp = (1.0 - alphas) * q_start_arm[None, :] + alphas * q_goal_arm[None, :]
        p0_path, p1_path = arm_capsules_world(q_interp, t_wb[arm], chain_spec=chains[arm])
        self_path, table_path = _self_and_table_clearances(
            p0_path, p1_path, eff_table_z, capsule_radii_m=chains[arm].capsule_radii_m
        )
        seg_min_self = float(np.min(self_path))
        seg_min_table = float(np.min(table_path))
        min_self_all = min(min_self_all, seg_min_self)
        if eff_table_z is not None:
            min_table_all = min(min_table_all, seg_min_table)

        travel_ok, max_step_t, travel_witnesses = _check_joint_travel_admission(
            q_start_arm,
            q_goal_arm,
            limits=chains[arm].joint_limits_rad,
            joint_names=chains[arm].joint_names,
            joint_travel_preset=st.get("joint_travel_preset", batch_joint_travel_preset),
            max_joint_travel_rad=st.get("max_joint_travel_rad"),
            max_backtracking_rad=st.get("max_backtracking_rad"),
        )
        max_step_travel_all = max(max_step_travel_all, max_step_t)

        if stage_valid and not travel_ok:
            stage_valid = False
            stage_reason = "abnormal_joint_travel"
        elif stage_valid and eff_table_z is not None and seg_min_table < 0.0:
            stage_valid = False
            stage_reason = "arm_body_below_table"
        elif stage_valid and seg_min_self < 0.0:
            stage_valid = False
            stage_reason = "self_collision"

        wave_segments.setdefault(wave_idx, {})[arm] = (sid, q_start_arm, q_goal_arm)
        if stage_valid:
            q_cur[arm] = q_goal_arm
        arm_wave[arm] = wave_idx + 1
        stage_finish_wave[sid] = wave_idx + 1

        rep = dict(
            id=sid,
            kind="move",
            arm=arm,
            valid=stage_valid,
            reason=stage_reason,
            joint_q_rad=reach["joint_q_rad"],
            pos_err_m=reach["pos_err_m"],
            rot_err_rad=reach["rot_err_rad"],
            min_joint_margin_rad=reach["min_joint_margin_rad"],
            self_clearance_m=round(seg_min_self, 5),
            table_clearance_m=round(seg_min_table, 5) if eff_table_z is not None else None,
            max_joint_travel_rad=round(max_step_t, 5),
        )
        if travel_witnesses:
            rep["travel_witnesses"] = travel_witnesses
        stage_reports.append(rep)
        if not stage_valid:
            blocking_ids.append(sid)
            if stage_reason not in reason_set:
                reason_set.append(stage_reason)

    report_by_id = {r["id"]: r for r in stage_reports}
    if not wave_segments:
        biman_clr, _ = bimanual_capsule_clearance_m(
            q_cur["left"][None, :],
            q_cur["right"][None, :],
            t_wb["left"],
            t_wb["right"],
            left_chain=chains["left"],
            right_chain=chains["right"],
        )
        min_bimanual_all = biman_clr
    else:
        running_q = {
            "left": wave_start_q[min(wave_start_q)]["left"].copy(),
            "right": wave_start_q[min(wave_start_q)]["right"].copy(),
        }
        for w_idx in sorted(wave_segments):
            segs = wave_segments[w_idx]
            if "left" in segs:
                _, ql0, ql1 = segs["left"]
                ql_seq = (1.0 - alphas) * ql0[None, :] + alphas * ql1[None, :]
                running_q["left"] = ql1
            else:
                ql_seq = np.broadcast_to(running_q["left"][None, :], (n_samples, 6))
            if "right" in segs:
                _, qr0, qr1 = segs["right"]
                qr_seq = (1.0 - alphas) * qr0[None, :] + alphas * qr1[None, :]
                running_q["right"] = qr1
            else:
                qr_seq = np.broadcast_to(running_q["right"][None, :], (n_samples, 6))

            biman_clr, pair_names = bimanual_capsule_clearance_m(
                ql_seq,
                qr_seq,
                t_wb["left"],
                t_wb["right"],
                left_chain=chains["left"],
                right_chain=chains["right"],
            )
            min_bimanual_all = min(min_bimanual_all, biman_clr)
            for _, (sid_w, _, _) in segs.items():
                report_by_id[sid_w]["bimanual_clearance_m"] = round(biman_clr, 5)
                if biman_clr < float(bimanual_min_clearance_m):
                    report_by_id[sid_w]["valid"] = False
                    if report_by_id[sid_w]["reason"] == "ok":
                        report_by_id[sid_w]["reason"] = "bimanual_arm_collision"
                    report_by_id[sid_w]["colliding_links"] = list(pair_names) if pair_names else []
                    if sid_w not in blocking_ids:
                        blocking_ids.append(sid_w)
                    if "bimanual_arm_collision" not in reason_set:
                        reason_set.append("bimanual_arm_collision")

    return dict(
        valid=len(blocking_ids) == 0,
        blocking_stage_ids=blocking_ids,
        reasons=reason_set,
        min_bimanual_clearance_m=round(min_bimanual_all, 5) if math.isfinite(min_bimanual_all) else None,
        min_self_clearance_m=round(min_self_all, 5) if math.isfinite(min_self_all) else None,
        min_table_clearance_m=round(min_table_all, 5) if (eff_table_z is not None and math.isfinite(min_table_all)) else None,
        max_joint_step_travel_rad=round(max_step_travel_all, 5),
        stage_reports=stage_reports,
        terminal_joint_q_rad={
            "left": [round(float(v), 6) for v in q_cur["left"]],
            "right": [round(float(v), 6) for v in q_cur["right"]],
        },
    )


_JAW_SYMMETRY_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)


def jaw_symmetric_angles_deg(angles_deg: Any) -> list[float]:
    """求解任意三维夹爪姿态角 `angles_deg` 精确的 $SO(3)$ 平行夹爪对称等价姿态（绕局部接近轴 `+X_local` 滚转 `180 deg`）。

    物理含义：
    平行双指夹爪绕局部手指伸出轴 `+X_local` 旋转 `180 deg` 后，两片对称指垫互换位置，
    三维接触点 `grasp_position_m`、法兰中心 `flange_position_m`、接近轴 `approach_dir` 与闭合直线 `closing_dir`
    均保持严格不变，但第 6 关节 (`joint6`) 解耦翻转 `pi rad`。适用于任意水平、侧向或三维倾斜姿态
    （注意当 `pitch != 0` 或 `roll != 0` 时，真实对称姿态并非简单 `yaw +- 180`）。

    参数:
        angles_deg: 夹爪姿态角 `[pitch_deg, roll_deg, yaw_deg]`，单位 `deg`。

    返回:
        规约至标准主值域 `[-180, 180] x [-90, 90] x [-180, 180]` 的等价姿态角列表 `[pitch_deg, roll_deg, yaw_deg]`，单位 `deg`。

    示例:
        `twin = jaw_symmetric_angles_deg([90.0, 0.0, 0.0])  # -> [-90.0, 0.0, -180.0]`
    """
    ang = _finite_vec(angles_deg, 3, "angles_deg")
    q_orig = angles_deg_to_quat_wxyz(float(ang[0]), float(ang[1]), float(ang[2]))
    q_twin = quat_multiply(q_orig, _JAW_SYMMETRY_QUAT_WXYZ)
    p_deg, r_deg, y_deg = quat_wxyz_to_angles_deg(q_twin)
    return [round(float(p_deg), 4), round(float(r_deg), 4), round(float(y_deg), 4)]


def _quat_slerp_midpoint_xyzw(q0_xyzw: np.ndarray, q1_xyzw: np.ndarray) -> list[float]:
    """计算两个单位四元数 `[qx, qy, qz, qw]` 的最短弧球面中点（返回归一化 `[qx, qy, qz, qw]`）。"""
    qa_wxyz = np.array([q0_xyzw[3], q0_xyzw[0], q0_xyzw[1], q0_xyzw[2]], dtype=np.float64)
    qb_wxyz = np.array([q1_xyzw[3], q1_xyzw[0], q1_xyzw[1], q1_xyzw[2]], dtype=np.float64)
    qm_wxyz = quat_slerp_wxyz(qa_wxyz, qb_wxyz, 0.5)
    return [round(float(qm_wxyz[1]), 6), round(float(qm_wxyz[2]), 6), round(float(qm_wxyz[3]), 6), round(float(qm_wxyz[0]), 6)]


DEFAULT_APPROACH_TILT_OFFSETS_DEG: tuple[float, ...] = (
    0.0,
    -20.0,
    20.0,
    -35.0,
    35.0,
    -50.0,
    50.0,
    -65.0,
    65.0,
)


def solve_reachable_pose(
    arm: str,
    position_m: Any,
    angles_deg: Any,
    contact_region: str = "pad_center",
    *,
    include_jaw_symmetry: bool = True,
    approach_tilt_offsets_deg: tuple[float, ...] | list[float] = DEFAULT_APPROACH_TILT_OFFSETS_DEG,
    yaw_offsets_deg: tuple[float, ...] | list[float] = (0.0,),
    candidate_angles_deg: list[Any] | tuple[Any, ...] | None = None,
    contact_depth_m: float | None = None,
    q_seed: Any = None,
    robot_model: Mapping[str, Any] | Serial6RChainSpec | None = None,
    robot_state: dict[str, Any] | None = None,
    base_pose_world: Any = None,
    joint_limits_rad: Any = None,
    table_z_m: float | None = None,
    pos_tol_m: float = 0.002,
    rot_tol_rad: float = 0.02,
    joint_travel_preset: str | None = "standard",
) -> dict[str, Any]:
    """在固定三维抓取接触点 `position_m` 处并行搜索无自碰撞、关节限位与行程合规的最优可达 6R 夹爪姿态。

    物理机制：
    1. 优先评估原始姿态 `angles_deg` 及其精确 $SO(3)$ 平行夹爪对称姿态 `jaw_symmetric_angles_deg(angles_deg)`（零几何偏差）；
    2. 当远端臂展或腕部折叠导致纯俯视/原角度发生连杆自碰 (`self_collision`) 或超限位时，绕局部指间闭合轴 `Y_local`
       施加 `approach_tilt_offsets_deg` 俯仰倾角（保持三维夹持闭合线 `closing_dir` 与指间中面法向 100% 不变，同时将法兰向肩部拉近以展开肘腕连杆）；
    3. 在所有可达候选中优先保留零角度偏差的原始/对称姿态，其次最小化俯仰偏角，并兼顾连杆自碰余量 `self_clearance_m`、
       关节限位余量 `min_joint_margin_rad` 与相对当前实测关节角的行程代价。

    参数:
        arm: 机械臂名称（`'left'` 或 `'right'`）。
        position_m: 目标抓取接触点三维坐标 `[x, y, z]`，单位 `m`（`env_origin` 世界系）。
        angles_deg: 参考夹爪姿态角 `[pitch_deg, roll_deg, yaw_deg]`，单位 `deg`。
        contact_region: 接触截面名称（`'pad_center'` | `'finger_middle'` | `'fingertip'`），默认 `'pad_center'`。
        include_jaw_symmetry: 是否自动纳入绕接近轴 `180 deg` 翻转的平行夹爪对称姿态，默认 `True`。
        approach_tilt_offsets_deg: 绕指间闭合轴的接近角倾角候选序列，单位 `deg`，默认 `(0.0, -20.0, 20.0, -35.0, 35.0, -50.0, 50.0, -65.0, 65.0)`。
        yaw_offsets_deg: 绕世界竖直轴 `Z` 的偏航微调候选序列，单位 `deg`，默认 `(0.0,)`。
        candidate_angles_deg: 可选显式附加评估的姿态角列表 `[[p, r, y], ...]`，单位 `deg`，默认 `None`。
        contact_depth_m: 可选自定义接触截面深度偏移，单位 `m`（范围 `[-0.0075, 0.0625]`），默认 `None`。
        q_seed: 可选起始 6 轴关节角 `[q1..q6]`，单位 `rad`；省略时自动从 `robot_state` 或工作区快照读取。
        robot_model: 可选运行时机器人运动学模型字典或 `Serial6RChainSpec`，默认 `None`。
        robot_state: 可选实测机器人状态字典，默认 `None`。
        base_pose_world: 可选显式 `4x4` `T_world_base` 基座齐次矩阵，单位 `m`，默认 `None`。
        joint_limits_rad: 可选显式 `(6, 2)` 关节上下限，单位 `rad`，默认 `None`。
        table_z_m: 可选显式支撑台面高度，单位 `m`，默认 `None`（不做隐式桌面高度假设）。
        pos_tol_m: 逆解收敛位置容差，单位 `m`（`> 0`），默认 `0.002 m`。
        rot_tol_rad: 逆解收敛姿态容差，单位 `rad`（`> 0`），默认 `0.02 rad`。
        joint_travel_preset: 关节行程准入预设（`'standard'` | `'extended'` | `'unconstrained'`），默认 `'standard'`。

    返回:
        字典 `{reachable, selected_angles_deg, is_jaw_symmetric_twin, approach_tilt_deg, yaw_offset_deg, joint_q_rad, pos_err_m, rot_err_rad, min_joint_margin_rad, self_clearance_m, table_clearance_m, max_joint_travel_rad, joint_travel_admissible, recommended_joint_travel_preset, jaw_symmetric_twin_angles_deg, candidates, reason}`。

    示例:
        `sol = solve_reachable_pose('left', [-0.145, -0.056, 0.90], [0.0, 0.0, -36.0])`
    """
    if arm not in ARMS:
        raise ValueError(f"invalid_arm: {arm!r}; expected 'left' or 'right'")
    pos_in = _finite_vec(position_m, 3, "position_m")
    ang_in = _finite_vec(angles_deg, 3, "angles_deg")

    eff_rs = (
        robot_state
        if robot_state is not None
        else (_autoload_workspace_robot_state() if (base_pose_world is None or q_seed is None) else None)
    )
    chain = resolve_6r_chain_spec(arm, robot_model=robot_model, joint_limits_rad=joint_limits_rad)
    t_wb = (
        np.asarray(base_pose_world, dtype=np.float64)
        if base_pose_world is not None
        else resolve_arm_base_transform(arm, eff_rs, robot_model=chain)
    )
    seed_vec: np.ndarray | None = None
    if q_seed is not None:
        seed_vec = _finite_vec(q_seed, 6, "q_seed")
    elif isinstance(eff_rs, dict):
        arm_q = ((eff_rs.get("arms") or {}).get(arm) or {}).get("q_rad")
        if isinstance(arm_q, (list, tuple)) and len(arm_q) == 6:
            seed_vec = _finite_vec(arm_q, 6, "q_rad")

    q_orig = angles_deg_to_quat_wxyz(float(ang_in[0]), float(ang_in[1]), float(ang_in[2]))
    q_twin_orig = quat_multiply(q_orig, _JAW_SYMMETRY_QUAT_WXYZ)
    twin_ang = jaw_symmetric_angles_deg(ang_in)
    flips = (False, True) if include_jaw_symmetry else (False,)
    tilts = [float(v) for v in approach_tilt_offsets_deg]
    yaws = [float(v) for v in yaw_offsets_deg]
    if 0.0 not in tilts:
        tilts = [0.0, *tilts]
    if 0.0 not in yaws:
        yaws = [0.0, *yaws]

    raw_specs: list[tuple[list[float], np.ndarray, bool, float, float, float]] = []
    seen_quats: list[np.ndarray] = []

    def _add_candidate(
        c_ang: list[float],
        c_quat: np.ndarray,
        is_twin: bool,
        t_deg: float,
        y_deg: float,
        dev_deg_val: float,
    ) -> None:
        qn = c_quat / max(float(np.linalg.norm(c_quat)), 1e-9)
        for prev in seen_quats:
            if abs(float(np.dot(qn, prev))) > 0.99995:
                return
        seen_quats.append(qn)
        raw_specs.append((c_ang, qn, is_twin, t_deg, y_deg, dev_deg_val))

    # 先放入零偏移原姿态与精确平行夹爪对称姿态，再按俯仰/偏航幅度从小到大扩展
    ordered_pairs = sorted(
        [(t_d, y_d) for t_d in tilts for y_d in yaws],
        key=lambda item: (abs(item[0]) + abs(item[1]), abs(item[0]), abs(item[1])),
    )
    for t_d, y_d in ordered_pairs:
        half_t = 0.5 * math.radians(t_d)
        q_tilt = np.array([math.cos(half_t), 0.0, math.sin(half_t), 0.0], dtype=np.float64)
        half_y = 0.5 * math.radians(y_d)
        q_yaw = np.array([math.cos(half_y), 0.0, 0.0, math.sin(half_y)], dtype=np.float64)
        grid_dev = round(math.hypot(t_d, y_d), 3)
        for flip in flips:
            q_base = q_twin_orig if flip else q_orig
            q_cand = quat_multiply(q_yaw, quat_multiply(q_base, q_tilt))
            if t_d == 0.0 and y_d == 0.0:
                c_ang = twin_ang if flip else [round(float(v), 4) for v in ang_in]
            else:
                p_c, r_c, y_c = quat_wxyz_to_angles_deg(q_cand)
                c_ang = [round(float(p_c), 4), round(float(r_c), 4), round(float(y_c), 4)]
            _add_candidate(c_ang, q_cand, flip, t_d, y_d, grid_dev)

    if candidate_angles_deg is not None:
        for extra_ang in candidate_angles_deg:
            ea = _finite_vec(extra_ang, 3, "candidate_angles_deg")
            eq = angles_deg_to_quat_wxyz(float(ea[0]), float(ea[1]), float(ea[2]))
            eq_norm = eq / max(float(np.linalg.norm(eq)), 1e-9)
            dot_orig = min(1.0, abs(float(np.dot(eq_norm, q_orig))))
            dot_twin = min(1.0, abs(float(np.dot(eq_norm, q_twin_orig))))
            is_twin_cand = bool(include_jaw_symmetry and dot_twin > dot_orig + 1e-6)
            best_dot = max(dot_orig, dot_twin) if include_jaw_symmetry else dot_orig
            geodesic_deg = round(math.degrees(2.0 * math.acos(best_dot)), 3)
            _add_candidate([round(float(v), 4) for v in ea], eq_norm, is_twin_cand, 0.0, 0.0, geodesic_deg)

    candidates_out: list[dict[str, Any]] = []
    for c_ang, _, is_twin, t_d, y_d, dev_deg in raw_specs:
        res = check_pose_reachability(
            arm=arm,
            position_m=pos_in,
            angles_deg=c_ang,
            pose_kind="grasp",
            contact_region=contact_region,
            contact_depth_m=contact_depth_m,
            q_seed=seed_vec,
            robot_model=chain,
            robot_state=eff_rs,
            base_pose_world=t_wb,
            table_z_m=table_z_m,
            pos_tol_m=pos_tol_m,
            rot_tol_rad=rot_tol_rad,
        )
        travel_ok = True
        ext_ok = True
        max_travel = res["joint_travel_from_seed_rad"]
        rec_preset = joint_travel_preset or "standard"
        if seed_vec is not None and res["joint_q_rad"] is not None:
            q_goal = np.asarray(res["joint_q_rad"], dtype=np.float64)
            travel_ok, max_t_val, _ = _check_joint_travel_admission(
                seed_vec,
                q_goal,
                limits=chain.joint_limits_rad,
                joint_names=chain.joint_names,
                joint_travel_preset=joint_travel_preset,
            )
            max_travel = round(float(max_t_val), 5)
            if not travel_ok:
                ext_ok, _, _ = _check_joint_travel_admission(
                    seed_vec,
                    q_goal,
                    limits=chain.joint_limits_rad,
                    joint_names=chain.joint_names,
                    joint_travel_preset="extended",
                )
                rec_preset = "extended" if ext_ok else "unconstrained"

        entry = dict(
            angles_deg=c_ang,
            is_jaw_symmetric_twin=bool(is_twin),
            approach_tilt_deg=round(float(t_d), 2),
            yaw_offset_deg=round(float(y_d), 2),
            orientation_deviation_deg=dev_deg,
            reachable=bool(res["reachable"]),
            joint_travel_admissible=bool(travel_ok),
            extended_travel_admissible=bool(ext_ok),
            recommended_joint_travel_preset=rec_preset,
            joint_q_rad=res["joint_q_rad"],
            pos_err_m=res["pos_err_m"],
            rot_err_rad=res["rot_err_rad"],
            min_joint_margin_rad=res["min_joint_margin_rad"],
            self_clearance_m=res["self_clearance_m"],
            table_clearance_m=res["table_clearance_m"],
            max_joint_travel_rad=max_travel,
            reason=res["reason"],
        )
        candidates_out.append(entry)

    feasible = [c for c in candidates_out if c["reachable"]]
    if feasible:
        def _rank_key(item: dict[str, Any]) -> tuple[int, float, float]:
            adm_rank = 0 if item["joint_travel_admissible"] else (1 if item["extended_travel_admissible"] else 2)
            dev = float(item["orientation_deviation_deg"])
            # 零倾角/零偏航的精确姿态（含 180° 平行夹爪对称等价姿态）在 standard 或 extended 行程可达时严格优先于倾斜回退解
            exact_pri = (
                0
                if (dev < 1e-3 and item["joint_travel_admissible"])
                else (1 if (dev < 1e-3 and item["extended_travel_admissible"]) else (2 + adm_rank))
            )
            sc = min(float(item["self_clearance_m"] or 0.0), 0.030)
            mg = min(float(item["min_joint_margin_rad"] or 0.0), 0.30)
            tr = float(item["max_joint_travel_rad"] or 0.0)
            cost = -10.0 * sc - 1.5 * mg + 0.20 * tr
            return (exact_pri, dev, cost)

        feasible_sorted = sorted(feasible, key=_rank_key)
        best = feasible_sorted[0]
        non_feasible = [c for c in candidates_out if not c["reachable"]]
        ordered_all = [*feasible_sorted, *non_feasible]
        return dict(
            reachable=True,
            selected_angles_deg=best["angles_deg"],
            is_jaw_symmetric_twin=best["is_jaw_symmetric_twin"],
            approach_tilt_deg=best["approach_tilt_deg"],
            yaw_offset_deg=best["yaw_offset_deg"],
            joint_q_rad=best["joint_q_rad"],
            pos_err_m=best["pos_err_m"],
            rot_err_rad=best["rot_err_rad"],
            min_joint_margin_rad=best["min_joint_margin_rad"],
            self_clearance_m=best["self_clearance_m"],
            table_clearance_m=best["table_clearance_m"],
            max_joint_travel_rad=best["max_joint_travel_rad"],
            joint_travel_admissible=best["joint_travel_admissible"],
            recommended_joint_travel_preset=best["recommended_joint_travel_preset"],
            jaw_symmetric_twin_angles_deg=twin_ang,
            candidates=ordered_all,
            reason="ok",
        )

    primary = candidates_out[0]
    return dict(
        reachable=False,
        selected_angles_deg=None,
        is_jaw_symmetric_twin=False,
        approach_tilt_deg=0.0,
        yaw_offset_deg=0.0,
        joint_q_rad=primary["joint_q_rad"],
        pos_err_m=primary["pos_err_m"],
        rot_err_rad=primary["rot_err_rad"],
        min_joint_margin_rad=primary["min_joint_margin_rad"],
        self_clearance_m=primary["self_clearance_m"],
        table_clearance_m=primary["table_clearance_m"],
        max_joint_travel_rad=primary["max_joint_travel_rad"],
        joint_travel_admissible=False,
        recommended_joint_travel_preset=joint_travel_preset or "standard",
        jaw_symmetric_twin_angles_deg=twin_ang,
        candidates=candidates_out,
        reason=primary["reason"],
    )


def check_source_to_dest(
    source_pos_m: tuple[float, ...] | list[float] | np.ndarray,
    source_angles_deg: tuple[float, ...] | list[float] | np.ndarray,
    dest_pos_m: tuple[float, ...] | list[float] | np.ndarray,
    dest_angles_deg: tuple[float, ...] | list[float] | np.ndarray | None = None,
    contact_region: str = "pad_center",
    *,
    dest_contact_region: str | None = None,
    arms: tuple[str, ...] | list[str] = ("left", "right"),
    include_jaw_symmetry: bool = True,
    dest_include_jaw_symmetry: bool | None = None,
    approach_tilt_offsets_deg: tuple[float, ...] | list[float] = DEFAULT_APPROACH_TILT_OFFSETS_DEG,
    yaw_offsets_deg: tuple[float, ...] | list[float] = (0.0,),
    contact_depth_m: float | None = None,
    dest_contact_depth_m: float | None = None,
    robot_model: Mapping[str, Any] | Serial6RChainSpec | None = None,
    robot_state: dict[str, Any] | None = None,
    table_z_m: float | None = None,
    pos_tol_m: float = 0.002,
    rot_tol_rad: float = 0.02,
    joint_travel_preset: str | None = "standard",
) -> dict[str, Any]:
    """双臂源点至目标点全链路单臂直达性与选臂核验（纯 6R 正逆运动学、$SO(3)$ 平行夹爪对称与关节行程准入）。

    物理机制：
    对每条候选机械臂依次求解源抓取位姿 `source`（以当前实测 `q_rad` 为初值）与目标放置/插入位姿 `dest`
    （当源点可达时以源点逆解关节角 `q_src` 为初值，精确计量源到目标的真实关节行程 `rad` 并筛选最优防缠绕对称分支），
    在执行任何物理接近前毫秒级判断是否存在单臂同时直达源点与目标点，避免因仅检查源点可达而误选机械臂引发不必要换手。

    参数:
        source_pos_m: 源抓取接触点三维世界坐标 `[x, y, z]`，单位 `m`（`env_origin` 世界系）。
        source_angles_deg: 源抓取参考姿态角 `[pitch_deg, roll_deg, yaw_deg]`，单位 `deg`。
        dest_pos_m: 目标放置/插入接触点三维世界坐标 `[x, y, z]`，单位 `m`（`env_origin` 世界系）。
        dest_angles_deg: 可选目标参考姿态角 `[pitch_deg, roll_deg, yaw_deg]`，单位 `deg`；默认 `None` 表示保持 `source_angles_deg`。
        contact_region: 源点接触截面名称（`'pad_center'` | `'finger_middle'` | `'fingertip'`），默认 `'pad_center'`。
        dest_contact_region: 可选目标点接触截面名称；默认 `None` 表示沿用 `contact_region`。
        arms: 待评估机械臂列表，子集取自 `('left', 'right')`，默认 `('left', 'right')`。
        include_jaw_symmetry: 源点是否自动评估绕接近轴 `180 deg` 翻转的对称等价姿态，默认 `True`。
        dest_include_jaw_symmetry: 目标点是否评估平行夹爪对称等价姿态；默认 `None` 表示沿用 `include_jaw_symmetry`。
        approach_tilt_offsets_deg: 绕指间闭合轴的接近角倾角候选，单位 `deg`，默认 `(0.0, -20.0, 20.0, -35.0, 35.0, -50.0, 50.0, -65.0, 65.0)`。
        yaw_offsets_deg: 绕世界竖直轴 `Z` 的偏航微调候选，单位 `deg`，默认 `(0.0,)`。
        contact_depth_m: 可选源点自定义接触截面深度偏移，单位 `m`（范围 `[-0.0075, 0.0625]`），默认 `None`。
        dest_contact_depth_m: 可选目标点自定义接触截面深度偏移，单位 `m`；默认 `None`（当 `dest_contact_region` 未单独指定时沿用 `contact_depth_m`）。
        robot_model: 可选运行时机器人运动学模型字典或 `Serial6RChainSpec`，默认 `None`。
        robot_state: 可选实测双臂状态字典；省略时自动从沙盒 `_spatial_snapshot.json` 预载。
        table_z_m: 可选显式支撑台面高度，单位 `m`，默认 `None`（不做隐式桌面高度假设）。
        pos_tol_m: 逆解收敛位置容差，单位 `m`（`> 0`），默认 `0.002 m`。
        rot_tol_rad: 逆解收敛姿态容差，单位 `rad`（`> 0`），默认 `0.02 rad`。
        joint_travel_preset: 关节行程准入预设（`'standard'` | `'extended'` | `'unconstrained'`），默认 `'standard'`。

    返回:
        字典 `{single_arm_candidates, recommended_arm, requires_handoff, source_reachable_arms, dest_reachable_arms, arms}`：
        - `single_arm_candidates`: 可单臂同时直达源点与目标点的机械臂列表（按姿态偏差 `deg`、关节行程 `rad` 与自碰净空 `m` 排序）；
        - `recommended_arm`: 最优单臂直达机械臂（`'left'` | `'right'` | `None`）；
        - `requires_handoff`: 布尔值，当无单臂可同时直达源与目标时为 `True`；
        - `source_reachable_arms` / `dest_reachable_arms`: 分别可达源点与目标点的机械臂列表；
        - `arms`: 每条机械臂的详细诊断（含选定 `source_angles_deg`/`dest_angles_deg` `deg`、`source_to_dest_joint_travel_rad` `rad`、`recommended_joint_travel_preset`）。

    示例:
        `route = check_source_to_dest([-0.13, -0.11, 0.87], [0.0, 0.0, -90.0], [0.16, -0.14, 0.96], [0.0, 0.0, 90.0])`
    """
    src_pos = _finite_vec(source_pos_m, 3, "source_pos_m")
    src_ang = _finite_vec(source_angles_deg, 3, "source_angles_deg")
    dst_pos = _finite_vec(dest_pos_m, 3, "dest_pos_m")
    dst_ang = _finite_vec(dest_angles_deg, 3, "dest_angles_deg") if dest_angles_deg is not None else src_ang.copy()

    req_arms: list[str] = []
    for arm_name in arms:
        if arm_name not in ARMS:
            raise ValueError(f"invalid_arm_in_arms: {arm_name!r}; expected subset of ('left', 'right')")
        if arm_name not in req_arms:
            req_arms.append(arm_name)
    if not req_arms:
        raise ValueError("arms must be non-empty")

    eff_rs = robot_state if robot_state is not None else _autoload_workspace_robot_state()
    eff_dst_region = dest_contact_region if dest_contact_region is not None else contact_region
    eff_dst_sym = include_jaw_symmetry if dest_include_jaw_symmetry is None else bool(dest_include_jaw_symmetry)
    eff_dst_depth = (
        dest_contact_depth_m
        if dest_contact_depth_m is not None
        else (contact_depth_m if dest_contact_region is None else None)
    )
    preset_rank_map = {"standard": 0, "extended": 1, "unconstrained": 2}

    def _compact_pose_report(sol: Mapping[str, Any]) -> dict[str, Any]:
        return dict(
            reachable=bool(sol["reachable"]),
            selected_angles_deg=sol["selected_angles_deg"],
            is_jaw_symmetric_twin=bool(sol["is_jaw_symmetric_twin"]),
            approach_tilt_deg=sol["approach_tilt_deg"],
            yaw_offset_deg=sol["yaw_offset_deg"],
            joint_q_rad=sol["joint_q_rad"],
            pos_err_m=sol["pos_err_m"],
            rot_err_rad=sol["rot_err_rad"],
            min_joint_margin_rad=sol["min_joint_margin_rad"],
            self_clearance_m=sol["self_clearance_m"],
            table_clearance_m=sol["table_clearance_m"],
            max_joint_travel_rad=sol["max_joint_travel_rad"],
            joint_travel_admissible=bool(sol["joint_travel_admissible"]),
            recommended_joint_travel_preset=sol["recommended_joint_travel_preset"],
            reason=sol["reason"],
        )

    arm_reports: dict[str, dict[str, Any]] = {}
    source_reachable_arms: list[str] = []
    dest_reachable_arms: list[str] = []
    ranked_single_arms: list[tuple[tuple[int, float, float], str]] = []

    for arm_name in req_arms:
        src_sol = solve_reachable_pose(
            arm=arm_name,
            position_m=src_pos,
            angles_deg=src_ang,
            contact_region=contact_region,
            include_jaw_symmetry=include_jaw_symmetry,
            approach_tilt_offsets_deg=approach_tilt_offsets_deg,
            yaw_offsets_deg=yaw_offsets_deg,
            contact_depth_m=contact_depth_m,
            robot_model=robot_model,
            robot_state=eff_rs,
            table_z_m=table_z_m,
            pos_tol_m=pos_tol_m,
            rot_tol_rad=rot_tol_rad,
            joint_travel_preset=joint_travel_preset,
        )
        dst_seed = src_sol["joint_q_rad"] if src_sol["reachable"] else None
        dst_sol = solve_reachable_pose(
            arm=arm_name,
            position_m=dst_pos,
            angles_deg=dst_ang,
            contact_region=eff_dst_region,
            include_jaw_symmetry=eff_dst_sym,
            approach_tilt_offsets_deg=approach_tilt_offsets_deg,
            yaw_offsets_deg=yaw_offsets_deg,
            contact_depth_m=eff_dst_depth,
            q_seed=dst_seed,
            robot_model=robot_model,
            robot_state=eff_rs,
            table_z_m=table_z_m,
            pos_tol_m=pos_tol_m,
            rot_tol_rad=rot_tol_rad,
            joint_travel_preset=joint_travel_preset,
        )
        src_ok = bool(src_sol["reachable"])
        dst_ok = bool(dst_sol["reachable"])
        both_ok = src_ok and dst_ok
        if src_ok:
            source_reachable_arms.append(arm_name)
        if dst_ok:
            dest_reachable_arms.append(arm_name)

        src_preset = str(src_sol["recommended_joint_travel_preset"] or "standard")
        dst_preset = str(dst_sol["recommended_joint_travel_preset"] or "standard")
        comb_preset = src_preset if preset_rank_map.get(src_preset, 0) >= preset_rank_map.get(dst_preset, 0) else dst_preset
        src_to_dst_travel = dst_sol["max_joint_travel_rad"] if both_ok else None
        min_sc = (
            round(min(float(src_sol["self_clearance_m"] or 0.0), float(dst_sol["self_clearance_m"] or 0.0)), 5)
            if both_ok
            else None
        )
        min_mg = (
            round(min(float(src_sol["min_joint_margin_rad"] or 0.0), float(dst_sol["min_joint_margin_rad"] or 0.0)), 5)
            if both_ok
            else None
        )

        arm_reports[arm_name] = dict(
            both_reachable=both_ok,
            source_reachable=src_ok,
            dest_reachable=dst_ok,
            source_angles_deg=src_sol["selected_angles_deg"],
            dest_angles_deg=dst_sol["selected_angles_deg"],
            source_to_dest_joint_travel_rad=src_to_dst_travel,
            recommended_joint_travel_preset=comb_preset,
            min_self_clearance_m=min_sc,
            min_joint_margin_rad=min_mg,
            source=_compact_pose_report(src_sol),
            dest=_compact_pose_report(dst_sol),
        )

        if both_ok:
            src_dev = math.hypot(float(src_sol["approach_tilt_deg"]), float(src_sol["yaw_offset_deg"]))
            dst_dev = math.hypot(float(dst_sol["approach_tilt_deg"]), float(dst_sol["yaw_offset_deg"]))
            tot_dev = round(src_dev + dst_dev, 3)
            tot_tr = float(src_sol["max_joint_travel_rad"] or 0.0) + float(dst_sol["max_joint_travel_rad"] or 0.0)
            cost = tot_tr - 5.0 * min(float(min_sc or 0.0), 0.030) - 0.5 * min(float(min_mg or 0.0), 0.30)
            zero_dev_pri = (
                0
                if (tot_dev < 1e-3 and comb_preset == "standard")
                else (1 if (tot_dev < 1e-3 and comb_preset == "extended") else (2 + preset_rank_map.get(comb_preset, 2)))
            )
            ranked_single_arms.append(((zero_dev_pri, tot_dev, cost), arm_name))

    ranked_single_arms.sort(key=lambda item: item[0])
    single_arm_candidates = [arm_name for _, arm_name in ranked_single_arms]
    return dict(
        single_arm_candidates=single_arm_candidates,
        recommended_arm=single_arm_candidates[0] if single_arm_candidates else None,
        requires_handoff=bool(len(single_arm_candidates) == 0),
        source_reachable_arms=source_reachable_arms,
        dest_reachable_arms=dest_reachable_arms,
        arms=arm_reports,
    )


def _extract_robot_state_dict(state_or_obs: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(state_or_obs, Mapping):
        return None
    if isinstance(state_or_obs.get("robot_state"), Mapping):
        return dict(state_or_obs["robot_state"])
    return dict(state_or_obs)


def _extract_arm_target_pose(rs_dict: Mapping[str, Any], arm: str) -> dict[str, Any] | None:
    arms_map = rs_dict.get("arms") if isinstance(rs_dict.get("arms"), Mapping) else rs_dict
    if not isinstance(arms_map, Mapping):
        return None
    entry = arms_map.get(arm)
    if not isinstance(entry, Mapping):
        return None
    saved_q: list[float] | None = None
    if isinstance(entry.get("q_rad"), (list, tuple)) and len(entry["q_rad"]) == 6:
        saved_q = [float(v) for v in _finite_vec(entry["q_rad"], 6, f"{arm}_q_rad")]
    ee = entry.get("ee_pose") if isinstance(entry.get("ee_pose"), Mapping) else entry
    if "position_m" in ee and "quaternion_xyzw" in ee:
        pos = [round(float(v), 5) for v in _finite_vec(ee["position_m"], 3, f"{arm}_position_m")]
        quat = [round(float(v), 6) for v in _finite_vec(ee["quaternion_xyzw"], 4, f"{arm}_quaternion_xyzw")]
        return dict(kind="flange", position_m=pos, quaternion_xyzw=quat, saved_q_rad=saved_q)
    if "grasp_position_m" in ee and "angles_deg" in ee:
        pos = [round(float(v), 5) for v in _finite_vec(ee["grasp_position_m"], 3, f"{arm}_grasp_position_m")]
        ang = [round(float(v), 3) for v in _finite_vec(ee["angles_deg"], 3, f"{arm}_angles_deg")]
        return dict(
            kind="grasp",
            grasp_position_m=pos,
            angles_deg=ang,
            contact_region=str(ee.get("contact_region", "pad_center")),
            saved_q_rad=saved_q,
        )
    return None


def plan_joint_safe_return(
    target_robot_state: Mapping[str, Any],
    current_robot_state: Mapping[str, Any] | None = None,
    *,
    arms: tuple[str, ...] | list[str] = ("left", "right"),
    stage_prefix: str = "return_home",
    intermediate_z_offset_m: float = 0.08,
    bimanual_min_clearance_m: float = 0.02,
    robot_model: Mapping[str, Any] | Serial6RChainSpec | None = None,
    table_z_m: float | None = None,
) -> dict[str, Any]:
    """生成经 6R 逆运动学、关节行程缠绕准入与双臂胶囊净空核验的安全复位 `stages` 序列。

    物理机制：
    1. 支持传入完整初始观测 `obs`、`obs['robot_state']` 或单/双臂实测位姿字典，直接生成标准 `move` 阶段；
    2. 若长距离操作或腕部大角度旋转导致单段直返触发 `abnormal_joint_travel`（`> pi rad`），自动按需附加
       `joint_travel_preset='extended'` 或插入抬升中点过渡路点（高度抬升 `intermediate_z_offset_m`，四元数球面最短弧中点）
       解缠绕并避开低处残余障碍；
    3. 若双臂同步回位轨迹之间的连杆胶囊最小净空低于 `bimanual_min_clearance_m`，自动通过 `depends_on` 串行化双臂回位顺序。

    参数:
        target_robot_state: 目标复位状态字典（可为初始观测 `obs_1`、`obs_1['robot_state']` 或 `{arm: {'position_m', 'quaternion_xyzw'}}`；位置单位 `m`）。
        current_robot_state: 可选当前实测状态字典（格式同上）；省略时自动从沙盒工作区 `_spatial_snapshot.json` 预载。
        arms: 需要复位的机械臂列表，子集取自 `('left', 'right')`，默认 `('left', 'right')`。
        stage_prefix: 生成阶段 `id` 的命名前缀，默认 `'return_home'`。
        intermediate_z_offset_m: 当单段直返受阻时自动插入的过渡路点相对起止点最高处的抬升量，单位 `m`（`>= 0`），默认 `0.08 m`。
        bimanual_min_clearance_m: 双臂同步回位允许的最小连杆胶囊净空，单位 `m`，默认 `0.02 m`。
        robot_model: 可选运行时机器人运动学模型字典或 `Serial6RChainSpec`，默认 `None`。
        table_z_m: 可选显式支撑台面高度，单位 `m`，默认 `None`。

    返回:
        字典 `{valid, stages, sequenced_bimanual, recommended_joint_travel_preset, prefilter_report}`。

    示例:
        `ret = plan_joint_safe_return(initial_obs['robot_state'])`
        `if ret['valid']: client.robot_action(mode='preview', observation_id=obs_id, stages=ret['stages'])`
    """
    if not isinstance(target_robot_state, Mapping):
        raise TypeError("target_robot_state must be a mapping (obs, robot_state, or per-arm pose dict)")
    z_lift = float(intermediate_z_offset_m)
    if not math.isfinite(z_lift) or z_lift < 0.0:
        raise ValueError("intermediate_z_offset_m must be finite and >= 0")

    tgt_rs = _extract_robot_state_dict(target_robot_state) or {}
    cur_rs = _extract_robot_state_dict(current_robot_state) if current_robot_state is not None else _autoload_workspace_robot_state()

    req_arms: list[str] = []
    for a in arms:
        if a not in ARMS:
            raise ValueError(f"invalid_arm_in_arms: {a!r}")
        if a not in req_arms:
            req_arms.append(a)
    if not req_arms:
        raise ValueError("arms must be non-empty")

    targets: dict[str, dict[str, Any]] = {}
    for a in req_arms:
        tp = _extract_arm_target_pose(tgt_rs, a)
        if tp is not None:
            targets[a] = tp
    if not targets:
        raise ValueError("target_robot_state contains no valid arm target poses")

    def _make_goal_stage(arm_name: str, preset: str | None = None, deps: list[str] | None = None) -> dict[str, Any]:
        tp = targets[arm_name]
        st: dict[str, Any] = dict(
            id=f"{stage_prefix}_{arm_name}",
            kind="move",
            arm=arm_name,
            tolerance_preset="coarse",
        )
        if tp["kind"] == "flange":
            st["position_m"] = list(tp["position_m"])
            st["quaternion_xyzw"] = list(tp["quaternion_xyzw"])
        else:
            st["grasp_position_m"] = list(tp["grasp_position_m"])
            st["angles_deg"] = list(tp["angles_deg"])
            st["contact_region"] = tp["contact_region"]
        if preset is not None:
            st["joint_travel_preset"] = preset
        if deps:
            st["depends_on"] = list(deps)
        return st

    def _make_arm_sequence(arm_name: str, preset: str | None = None, use_via: bool = False, deps: list[str] | None = None) -> list[dict[str, Any]]:
        cur_pose = _extract_arm_target_pose(cur_rs, arm_name) if isinstance(cur_rs, Mapping) else None
        tp = targets[arm_name]
        if use_via and cur_pose is not None and cur_pose["kind"] == "flange" and tp["kind"] == "flange":
            p0 = np.asarray(cur_pose["position_m"], dtype=np.float64)
            p1 = np.asarray(tp["position_m"], dtype=np.float64)
            q0 = np.asarray(cur_pose["quaternion_xyzw"], dtype=np.float64)
            q1 = np.asarray(tp["quaternion_xyzw"], dtype=np.float64)
            via_pos = [
                round(0.5 * float(p0[0] + p1[0]), 5),
                round(0.5 * float(p0[1] + p1[1]), 5),
                round(max(float(p0[2]), float(p1[2])) + max(z_lift, 0.04), 5),
            ]
            via_quat = _quat_slerp_midpoint_xyzw(q0, q1)
            via_id = f"{stage_prefix}_{arm_name}_lift"
            via_st: dict[str, Any] = dict(
                id=via_id,
                kind="move",
                arm=arm_name,
                position_m=via_pos,
                quaternion_xyzw=via_quat,
                tolerance_preset="coarse",
            )
            if preset is not None:
                via_st["joint_travel_preset"] = preset
            if deps:
                via_st["depends_on"] = list(deps)
            goal_st = _make_goal_stage(arm_name, preset=preset, deps=None)
            return [via_st, goal_st]
        return [_make_goal_stage(arm_name, preset=preset, deps=deps)]

    # 1. 先逐臂检查单段直返，按需升级 extended 或添加抬升过渡中点
    arm_presets: dict[str, str | None] = {a: None for a in targets}
    arm_use_via: dict[str, bool] = {a: False for a in targets}
    rec_preset = "standard"

    for a in targets:
        chain_a = resolve_6r_chain_spec(a, robot_model=robot_model)
        cur_arm_entry = ((cur_rs or {}).get("arms") or {}).get(a) if isinstance(cur_rs, Mapping) else None
        cur_q_raw = cur_arm_entry.get("q_rad") if isinstance(cur_arm_entry, Mapping) else None
        cur_q_vec = _finite_vec(cur_q_raw, 6, f"{a}_cur_q") if isinstance(cur_q_raw, (list, tuple)) and len(cur_q_raw) == 6 else None
        saved_goal_q = np.asarray(targets[a]["saved_q_rad"], dtype=np.float64) if targets[a].get("saved_q_rad") is not None else None

        def _saved_branch_ok(preset_name: str | None) -> bool:
            if cur_q_vec is None or saved_goal_q is None:
                return True
            ok_br, _, _ = _check_joint_travel_admission(
                cur_q_vec,
                saved_goal_q,
                limits=chain_a.joint_limits_rad,
                joint_names=chain_a.joint_names,
                joint_travel_preset=preset_name,
            )
            return ok_br

        st_std = _make_arm_sequence(a, preset=None, use_via=False)
        pf_std = prefilter_stages_kinematics(
            st_std,
            initial_robot_state=cur_rs,
            robot_model=robot_model,
            table_z_m=table_z_m,
            bimanual_min_clearance_m=-1.0,
        )
        if pf_std["valid"] and _saved_branch_ok("standard"):
            continue
        if ("abnormal_joint_travel" in pf_std["reasons"]) or not _saved_branch_ok("standard"):
            st_ext = _make_arm_sequence(a, preset="extended", use_via=False)
            pf_ext = prefilter_stages_kinematics(
                st_ext,
                initial_robot_state=cur_rs,
                robot_model=robot_model,
                table_z_m=table_z_m,
                bimanual_min_clearance_m=-1.0,
            )
            if pf_ext["valid"] and _saved_branch_ok("extended"):
                arm_presets[a] = "extended"
                if rec_preset == "standard":
                    rec_preset = "extended"
                continue
        st_via = _make_arm_sequence(a, preset="extended", use_via=True)
        pf_via = prefilter_stages_kinematics(
            st_via,
            initial_robot_state=cur_rs,
            robot_model=robot_model,
            table_z_m=table_z_m,
            bimanual_min_clearance_m=-1.0,
        )
        if pf_via["valid"] and _saved_branch_ok("extended"):
            arm_presets[a] = "extended"
            arm_use_via[a] = True
            if rec_preset == "standard":
                rec_preset = "extended"
        else:
            st_unc = _make_arm_sequence(a, preset="unconstrained", use_via=True)
            pf_unc = prefilter_stages_kinematics(
                st_unc,
                initial_robot_state=cur_rs,
                robot_model=robot_model,
                table_z_m=table_z_m,
                bimanual_min_clearance_m=-1.0,
            )
            if pf_unc["valid"]:
                arm_presets[a] = "unconstrained"
                arm_use_via[a] = True
                rec_preset = "unconstrained"

    # 2. 组装同步回位批次并检验双臂互碰撞净空；若互碰则自动通过 depends_on 串行化
    sim_stages: list[dict[str, Any]] = []
    for a in req_arms:
        if a in targets:
            sim_stages.extend(_make_arm_sequence(a, preset=arm_presets[a], use_via=arm_use_via[a]))

    pf_sim = prefilter_stages_kinematics(
        sim_stages,
        initial_robot_state=cur_rs,
        robot_model=robot_model,
        table_z_m=table_z_m,
        bimanual_min_clearance_m=bimanual_min_clearance_m,
    )
    if pf_sim["valid"] or len(targets) < 2:
        return dict(
            valid=bool(pf_sim["valid"]),
            stages=sim_stages,
            sequenced_bimanual=False,
            recommended_joint_travel_preset=rec_preset,
            prefilter_report=pf_sim,
        )

    # 尝试左右先后两种串行顺序，选取净空更优且通过核验的方案
    ordered_arms = [a for a in req_arms if a in targets]
    best_seq_stages = sim_stages
    best_seq_pf = pf_sim
    for first_a, second_a in ((ordered_arms[0], ordered_arms[1]), (ordered_arms[1], ordered_arms[0])):
        s1 = _make_arm_sequence(first_a, preset=arm_presets[first_a], use_via=arm_use_via[first_a])
        s2 = _make_arm_sequence(
            second_a,
            preset=arm_presets[second_a],
            use_via=arm_use_via[second_a],
            deps=[s1[-1]["id"]],
        )
        cand_stages = [*s1, *s2]
        pf_seq = prefilter_stages_kinematics(
            cand_stages,
            initial_robot_state=cur_rs,
            robot_model=robot_model,
            table_z_m=table_z_m,
            bimanual_min_clearance_m=bimanual_min_clearance_m,
        )
        if pf_seq["valid"]:
            return dict(
                valid=True,
                stages=cand_stages,
                sequenced_bimanual=True,
                recommended_joint_travel_preset=rec_preset,
                prefilter_report=pf_seq,
            )
        if (pf_seq["min_bimanual_clearance_m"] or -1.0) > (best_seq_pf["min_bimanual_clearance_m"] or -1.0):
            best_seq_stages = cand_stages
            best_seq_pf = pf_seq

    return dict(
        valid=bool(best_seq_pf["valid"]),
        stages=best_seq_stages,
        sequenced_bimanual=best_seq_stages is not sim_stages,
        recommended_joint_travel_preset=rec_preset,
        prefilter_report=best_seq_pf,
    )
