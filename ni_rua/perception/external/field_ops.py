r"""三维刚体李群几何算子、正交标架与手内功能特征精确配准工具 (`field_ops.py`)。

核心职责：
1. 李群 $SO(3)$ / $SE(3)$ 指数-对数映射、左雅可比及其闭式逆、伴随表示（Adjoint）与四元数最短弧球面插值（SLERP）；
2. 三维刚体手内功能特征精确配准 (`align_in_hand_feature_to_target`)：
   将参考夹爪位姿下测得的手内功能特征点与主轴通过精确 $SE(3)$ 刚体变换映射至任意三维目标点与目标轴，
   支持副参考轴防自转唯一锁定与绕目标轴显式滚转角控制，直接输出规范主分支 `[-180, 180] x [-90, 90] x [-180, 180]` (`deg`) 下的夹爪位姿；
3. 通用空间场有符号表面净空与单位外法向梯度查询 (`query_spatial_clearance_and_gradient`)。

本模块仅包含零任务假设的纯数学几何与李群算子，不包含任何动作套路模板或形状分桶启发式。
"""
from __future__ import annotations

import inspect
import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

if __package__:
    from .pose_math import (
        CONTACT_DEPTH_PRESETS_M,
        GRASP_POINT_OFFSET_M,
        angles_deg_to_quat_wxyz,
        matrix_to_quat_wxyz,
        orthonormal_triad_from_axes,
        quat_normalize,
        quat_slerp_wxyz,
        quat_wxyz_to_angles_deg,
        quat_wxyz_to_matrix,
        se3_adjoint,
        se3_exp,
        se3_inverse,
        se3_log,
        so3_exp,
        so3_left_jacobian,
        so3_left_jacobian_inv,
        so3_log,
        so3_skew,
    )
    from ...core.pose_math import _unit_vec3, _vec3, _vec6
else:
    from pose_math import (  # type: ignore[no-redef]
        CONTACT_DEPTH_PRESETS_M,
        GRASP_POINT_OFFSET_M,
        angles_deg_to_quat_wxyz,
        matrix_to_quat_wxyz,
        orthonormal_triad_from_axes,
        quat_normalize,
        quat_slerp_wxyz,
        quat_wxyz_to_angles_deg,
        quat_wxyz_to_matrix,
        se3_adjoint,
        se3_exp,
        se3_inverse,
        se3_log,
        so3_exp,
        so3_left_jacobian,
        so3_left_jacobian_inv,
        so3_log,
        so3_skew,
    )
    from _pose_math_core import _unit_vec3, _vec3, _vec6  # type: ignore[no-redef, import-not-found]

ARMS = ("left", "right")
SPEED_PRESETS = ("slower", "slow", "careful", "normal", "fast", "faster")
STALL_GUARD_PRESETS = ("off", "contact_stop", "strict")
TOLERANCE_PRESETS = (
    "tight",
    "fine",
    "strict",
    "normal",
    "grasp",
    "relaxed",
    "transit",
    "compliant",
    "seating",
    "coarse",
    "unconstrained_orientation",
)
GRIP_GUARD_PRESETS = ("none", "non_empty", "thin_object", "rigid_object", "wide_object")
JOINT_TRAVEL_PRESETS = ("standard", "extended", "unconstrained")
MIN_CONTACT_DEPTH_M = -0.0075
MAX_CONTACT_DEPTH_M = 0.0625
MAX_SCHEMA_POS_TOL_M = 0.10
MAX_SCHEMA_ORI_TOL_RAD = math.pi
MAX_SCHEMA_UPWARD_DZ_M = 0.15
MAX_SCHEMA_DOWNWARD_DZ_M = 0.15
DEFAULT_MAX_APERTURE_M = 0.088


def _validate_arm(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"invalid_arm: {arm!r}; expected 'left' or 'right'")
    return arm


def _validate_contact_spec(contact_region: str = "pad_center", contact_depth_m: float | None = None) -> float:
    """校验接触截面参数并返回接触参考点距 `link6` 沿接近轴 `+X` 的偏移距离（单位 `m`）。"""
    if contact_depth_m is not None:
        if type(contact_depth_m) not in (int, float) or not math.isfinite(contact_depth_m):
            raise ValueError("invalid_contact_depth_m: expected finite number")
        d = float(contact_depth_m)
        if not MIN_CONTACT_DEPTH_M <= d <= MAX_CONTACT_DEPTH_M:
            raise ValueError(f"contact_depth_m_out_of_range: {d}; expected [{MIN_CONTACT_DEPTH_M}, {MAX_CONTACT_DEPTH_M}]")
        return float(GRASP_POINT_OFFSET_M - d)
    if contact_region not in CONTACT_DEPTH_PRESETS_M:
        raise ValueError(f"invalid_contact_region: {contact_region!r}")
    return float(GRASP_POINT_OFFSET_M - CONTACT_DEPTH_PRESETS_M[contact_region])


def _canonicalize_angles_deg(angles_deg: Any) -> list[float]:
    """校验并将姿态角 `[pitch_deg, roll_deg, yaw_deg]` 规约至规范主分支 `[-180,180] x [-90,90] x [-180,180]` (`deg`)。"""
    ang = _vec3(angles_deg, "angles_deg")
    quat = angles_deg_to_quat_wxyz(float(ang[0]), float(ang[1]), float(ang[2]))
    p_deg, r_deg, y_deg = quat_wxyz_to_angles_deg(quat)
    p_deg = float(np.clip(((p_deg + 180.0) % 360.0) - 180.0, -180.0, 180.0))
    r_deg = float(np.clip(r_deg, -90.0, 90.0))
    y_deg = float(np.clip(((y_deg + 180.0) % 360.0) - 180.0, -180.0, 180.0))
    return [round(p_deg, 5), round(r_deg, 5), round(y_deg, 5)]


def _rodrigues_axis_angle(axis_unit: np.ndarray, angle_rad: float) -> np.ndarray:
    """计算绕单位轴 `axis_unit` 右手旋转 `angle_rad` 弧度的 `3x3` 旋转矩阵。"""
    return Rotation.from_rotvec(np.asarray(axis_unit, dtype=np.float64) * float(angle_rad)).as_matrix()


def _minimal_axis_alignment_rotation(u_from: np.ndarray, u_to: np.ndarray, fallback_hint: np.ndarray | None = None) -> np.ndarray:
    """计算将单位向量 `u_from` 旋转至 `u_to` 的测地最短弧 `3x3` 旋转矩阵（含 `180 deg` 反平行确定性正交轴处理）。"""
    dot = float(np.clip(np.dot(u_from, u_to), -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-9:
        cand = fallback_hint if fallback_hint is not None else np.zeros(3, dtype=np.float64)
        proj = cand - float(np.dot(cand, u_from)) * u_from
        if float(np.linalg.norm(proj)) <= 1e-6:
            idx_min = int(np.argmin(np.abs(u_from)))
            basis = np.zeros(3, dtype=np.float64)
            basis[idx_min] = 1.0
            proj = basis - float(np.dot(basis, u_from)) * u_from
        k_axis = proj / float(np.linalg.norm(proj))
        return 2.0 * np.outer(k_axis, k_axis) - np.eye(3, dtype=np.float64)
    cross = np.cross(u_from, u_to)
    k_mat = so3_skew(cross)
    return np.eye(3, dtype=np.float64) + k_mat + (k_mat @ k_mat) / (1.0 + dot)


def _obb_points_sdf_and_grad(
    points_m: np.ndarray,
    center_m: np.ndarray,
    axes: np.ndarray,
    half_extents_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """向量化计算查询点集 `points_m` (`(M, 3)`, 单位 `m`) 到单个有向包围盒 `OBB3D` 的精确有符号距离与单位外法向梯度。"""
    p_loc = (points_m - center_m[None, :]) @ axes
    abs_p = np.abs(p_loc)
    q = abs_p - half_extents_m[None, :]

    outside = np.maximum(q, 0.0)
    out_norm = np.linalg.norm(outside, axis=1)
    inside_dist = np.max(q, axis=1)
    is_out = out_norm > 1e-12
    sdf = np.where(is_out, out_norm, inside_dist)

    signs = np.where(p_loc >= 0.0, 1.0, -1.0)
    grad_out_loc = signs * (outside / np.maximum(out_norm[:, None], 1e-12))

    face_idx = np.argmax(q, axis=1)
    grad_in_loc = np.zeros_like(p_loc)
    rows = np.arange(points_m.shape[0])
    grad_in_loc[rows, face_idx] = signs[rows, face_idx]

    grad_loc = np.where(is_out[:, None], grad_out_loc, grad_in_loc)
    grad_world = grad_loc @ axes.T
    norms = np.linalg.norm(grad_world, axis=1, keepdims=True)
    grad_world = grad_world / np.maximum(norms, 1e-12)
    return sdf, grad_world


def _invoke_compiled_clearance_fn(
    q_fn: Any,
    pts: np.ndarray,
    exclude_set: set[str],
    *,
    include_table: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    """按参数签名调用 `query_clearance_and_gradient`，零异常吞噬。"""
    params = inspect.signature(q_fn).parameters
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    accepts_excl = has_var_kw or ("exclude_object_ids" in params)
    if exclude_set and not accepts_excl:
        return None
    kwargs: dict[str, Any] = {}
    if accepts_excl and "exclude_object_ids" in params:
        kwargs["exclude_object_ids"] = tuple(sorted(exclude_set))
    elif has_var_kw and exclude_set:
        kwargs["exclude_object_ids"] = tuple(sorted(exclude_set))
    if "include_table" in params:
        kwargs["include_table"] = bool(include_table)
    d_arr, g_arr = q_fn(pts, **kwargs)
    m_cnt = pts.shape[0]
    return np.asarray(d_arr, dtype=np.float64).reshape(m_cnt), np.asarray(g_arr, dtype=np.float64).reshape(m_cnt, 3)


def query_spatial_clearance_and_gradient(
    spatial_field: Any,
    query_pts_m: Any,
    *,
    exclude_object_ids: Any = (),
    include_table: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """统一查询任意空间场对象在三维采样点集 `query_pts_m` 处的实测有符号最小表面净空距离与单位排斥梯度。

    参数:
        spatial_field: 空间场实例（`CompiledSpatialField`、`SceneProgram`、`SpatialFieldSnapshot`、
            `PointCloudSpatialField`、`VirtualSpatialField` 或实现了 `.query_clearance_and_gradient(pts_m)` 的鸭子类型对象）。
        query_pts_m: 形状为 `(M, 3)` 或 `(3,)` 的待测三维坐标数组，单位 `m`（`env_origin` 世界系）。
        exclude_object_ids: 显式排除在距离统计之外的物体 ID 序列（如刚体持握物体或允许接触目标），默认 `()`。
        include_table: 对显式持有 `table_z_m` 的空间场是否计入台面半空间净空，默认 `False`。

    返回:
        二元组 `(clearances_m, gradients_unit)`：
        - `clearances_m`: 形状 `(M,)` 的有符号最小表面距离数组，单位 `m`（`> 0` 为表面外净空，`< 0` 为穿透深度；空场为 `+inf`）；
        - `gradients_unit`: 形状 `(M, 3)` 的单位外法向排斥梯度向量（指向远离最近实测表面的方向；空场为 `[0, 0, 1]`）。

    示例:
        `clr_m, grad = query_spatial_clearance_and_gradient(cf, [[0.0, -0.15, 0.82]], exclude_object_ids=['o1'])`
    """
    pts = np.asarray(query_pts_m, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
        raise ValueError("invalid_query_pts_m: expected finite array of shape (M, 3)")
    m_cnt = pts.shape[0]
    exclude_set = {str(oid) for oid in (exclude_object_ids or ()) if oid is not None}

    if spatial_field is None:
        default_grad = np.zeros((m_cnt, 3), dtype=np.float64)
        default_grad[:, 2] = 1.0
        return np.full(m_cnt, np.inf, dtype=np.float64), default_grad

    q_fn = getattr(spatial_field, "query_clearance_and_gradient", None)
    if callable(q_fn):
        direct_res = _invoke_compiled_clearance_fn(q_fn, pts, exclude_set, include_table=include_table)
        if direct_res is not None:
            return direct_res

    objs = getattr(spatial_field, "objects", None)
    if not isinstance(objs, dict):
        raise ValueError(f"unsupported_spatial_field_type: {type(spatial_field).__name__}")

    min_clr = np.full(m_cnt, np.inf, dtype=np.float64)
    best_grad = np.zeros((m_cnt, 3), dtype=np.float64)
    best_grad[:, 2] = 1.0

    local_trees = getattr(spatial_field, "_obj_local_trees", None)
    vox_default = float(getattr(spatial_field, "surface_voxel_radius_m", 0.006))

    for oid, vobj in objs.items():
        if str(oid) in exclude_set:
            continue
        obb_attr = getattr(vobj, "obb", None)
        obb = obb_attr() if callable(obb_attr) else obb_attr
        if obb is None:
            continue
        center_m = np.asarray(obb.center, dtype=np.float64)
        axes = np.asarray(obb.axes, dtype=np.float64)
        half_ext = np.asarray(obb.half_extents, dtype=np.float64)

        tree_obj = local_trees.get(oid) if isinstance(local_trees, dict) else None
        if tree_obj is not None and getattr(tree_obj, "n_points", 0) >= 1:
            p_loc = (pts - center_m[None, :]) @ axes
            raw_dists, nn_idx = tree_obj.tree.query(p_loc, k=1)
            raw_dists = np.asarray(raw_dists, dtype=np.float64).reshape(m_cnt)
            nn_pts = tree_obj.local_points_m[np.asarray(nn_idx, dtype=np.int64).reshape(m_cnt)]
            diff_loc = p_loc - nn_pts
            diff_norm = np.linalg.norm(diff_loc, axis=1)
            fallback_loc = p_loc / np.maximum(np.linalg.norm(p_loc, axis=1, keepdims=True), 1e-12)
            fallback_zero = np.linalg.norm(p_loc, axis=1) <= 1e-12
            if np.any(fallback_zero):
                fallback_loc[fallback_zero] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            unit_loc = np.where(diff_norm[:, None] > 1e-9, diff_loc / np.maximum(diff_norm[:, None], 1e-12), fallback_loc)
            grad_w = unit_loc @ axes.T
            vox_r = float(getattr(tree_obj, "voxel_radius_m", vox_default))
            clr = raw_dists - vox_r
        else:
            clr, grad_w = _obb_points_sdf_and_grad(pts, center_m, axes, half_ext)

        better = clr < min_clr
        if np.any(better):
            min_clr = np.where(better, clr, min_clr)
            best_grad[better] = grad_w[better]

    if include_table:
        tz = getattr(spatial_field, "table_z", getattr(spatial_field, "table_z_m", None))
        if tz is None:
            raise ValueError("table_z_m_not_available_on_spatial_field")
        d_tab = pts[:, 2] - float(tz)
        take_tab = d_tab < min_clr
        if np.any(take_tab):
            min_clr = np.where(take_tab, d_tab, min_clr)
            best_grad = np.where(take_tab[:, None], np.array([[0.0, 0.0, 1.0]], dtype=np.float64), best_grad)

    return min_clr, best_grad


def _build_stage_move_dict(
    *,
    stage_id: str,
    arm: str,
    grasp_pos_m: np.ndarray,
    angles_deg: list[float],
    contact_region: str = "pad_center",
    contact_depth_m: float | None = None,
    depends_on: list[str] | None = None,
    speed_preset: str | None = None,
    position_tolerance_m: float | None = None,
    orientation_tolerance_rad: float | None = None,
    max_upward_dz_m: float | None = None,
    max_downward_dz_m: float | None = None,
    stall_guard_preset: str | None = None,
    tolerance_preset: str | None = None,
    gripper_goal_normalized: float | None = None,
    gripper_motion: dict[str, float] | None = None,
    grip_guard_preset: str | None = None,
    min_aperture_m: float | None = None,
    max_aperture_m: float | None = None,
    joint_travel_preset: str | None = None,
) -> dict[str, Any]:
    """构造严格符合 `robot_action` `_STAGE_SCHEMAS['move']` 契约的单步阶段字典（零未知字段、零互斥冲突）。"""
    _validate_arm(arm)
    _validate_contact_spec(contact_region=contact_region, contact_depth_m=contact_depth_m)
    st: dict[str, Any] = {
        "id": str(stage_id),
        "kind": "move",
        "arm": arm,
        "grasp_position_m": [round(float(v), 5) for v in grasp_pos_m],
        "angles_deg": _canonicalize_angles_deg(angles_deg),
    }
    if contact_depth_m is not None:
        st["contact_depth_m"] = round(float(contact_depth_m), 5)
    elif contact_region != "pad_center":
        st["contact_region"] = str(contact_region)

    if depends_on:
        st["depends_on"] = list(dict.fromkeys(str(d) for d in depends_on))
    if speed_preset is not None:
        if speed_preset not in SPEED_PRESETS:
            raise ValueError(f"invalid_speed_preset: {speed_preset!r}")
        st["speed_preset"] = speed_preset
    if tolerance_preset is not None:
        if tolerance_preset not in TOLERANCE_PRESETS:
            raise ValueError(f"invalid_tolerance_preset: {tolerance_preset!r}")
        st["tolerance_preset"] = str(tolerance_preset)
    if position_tolerance_m is not None:
        tol_p = float(position_tolerance_m)
        if not math.isfinite(tol_p) or not (0.0 < tol_p <= MAX_SCHEMA_POS_TOL_M):
            raise ValueError(f"invalid_position_tolerance_m: {position_tolerance_m!r}")
        st["position_tolerance_m"] = round(tol_p, 5)
    if orientation_tolerance_rad is not None:
        tol_o = float(orientation_tolerance_rad)
        if not math.isfinite(tol_o) or not (0.0 < tol_o <= MAX_SCHEMA_ORI_TOL_RAD):
            raise ValueError(f"invalid_orientation_tolerance_rad: {orientation_tolerance_rad!r}")
        st["orientation_tolerance_rad"] = round(tol_o, 5)
    if max_upward_dz_m is not None:
        up_dz = float(max_upward_dz_m)
        if not math.isfinite(up_dz) or not (0.0 <= up_dz <= MAX_SCHEMA_UPWARD_DZ_M):
            raise ValueError(f"invalid_max_upward_dz_m: {max_upward_dz_m!r}")
        st["max_upward_dz_m"] = round(up_dz, 5)
    if max_downward_dz_m is not None:
        down_dz = float(max_downward_dz_m)
        if not math.isfinite(down_dz) or not (0.0 <= down_dz <= MAX_SCHEMA_DOWNWARD_DZ_M):
            raise ValueError(f"invalid_max_downward_dz_m: {max_downward_dz_m!r}")
        st["max_downward_dz_m"] = round(down_dz, 5)
    if stall_guard_preset is not None:
        if stall_guard_preset not in STALL_GUARD_PRESETS:
            raise ValueError(f"invalid_stall_guard_preset: {stall_guard_preset!r}")
        st["stall_guard_preset"] = stall_guard_preset
    if gripper_goal_normalized is not None:
        g_norm = float(np.clip(float(gripper_goal_normalized), 0.0, 1.0))
        st["gripper_goal_normalized"] = round(g_norm, 5)
        if gripper_motion is not None:
            st["gripper_motion"] = dict(gripper_motion)
    if grip_guard_preset is not None:
        if grip_guard_preset not in GRIP_GUARD_PRESETS:
            raise ValueError(f"invalid_grip_guard_preset: {grip_guard_preset!r}")
        st["grip_guard_preset"] = grip_guard_preset
    if min_aperture_m is not None:
        st["min_aperture_m"] = round(float(np.clip(float(min_aperture_m), 0.0, 0.088)), 5)
    if max_aperture_m is not None:
        st["max_aperture_m"] = round(float(np.clip(float(max_aperture_m), 0.0, 0.088)), 5)
    if joint_travel_preset is not None:
        if joint_travel_preset not in JOINT_TRAVEL_PRESETS:
            raise ValueError(f"invalid_joint_travel_preset: {joint_travel_preset!r}")
        st["joint_travel_preset"] = joint_travel_preset
    return st


def align_in_hand_feature_to_target(
    grasp_pos_m: Any,
    grasp_angles_deg: Any,
    feature_point_m: Any,
    feature_axis: Any,
    target_point_m: Any,
    target_axis: Any,
    *,
    feature_secondary_axis: Any = None,
    target_secondary_axis: Any = None,
    roll_about_target_axis_deg: float = 0.0,
    contact_region: str = "pad_center",
    contact_depth_m: float | None = None,
) -> dict[str, Any]:
    """求解三维刚体手内功能特征点与主轴精确配准到任意三维目标点与目标轴所需的目标夹爪位姿。

    物理原理：
    设夹爪在参考位姿 `(grasp_pos_m, grasp_angles_deg)` 下刚性持握物体，此时在世界系测得手内功能特征参考点
    `feature_point_m` 与特征单位主轴 `feature_axis`。本函数在 $SE(3)$ 群上精确计算将 `(feature_point_m, feature_axis)`
    刚性映射到 `(target_point_m, target_axis)`（支持任意三维倾斜轴向）后的目标夹爪位姿 `(target_pos_m, target_angles_deg)`。
    若同时提供 `feature_secondary_axis` 与 `target_secondary_axis`，则唯一锁定绕目标轴的时钟转角；
    亦可叠加 `roll_about_target_axis_deg` 微调绕目标轴的右手滚转角。

    参数:
        grasp_pos_m: 测量手内特征时的参考夹爪位置 `[x, y, z]`，单位 `m`（`env_origin` 世界系）。
        grasp_angles_deg: 测量手内特征时的参考夹爪姿态角 `[pitch_deg, roll_deg, yaw_deg]`，单位 `deg`。
        feature_point_m: 手内物体功能特征点在世界系下的三维坐标 `[x, y, z]`，单位 `m`。
        feature_axis: 手内物体功能特征主轴在世界系下的三维方向向量 `[ax, ay, az]`（无量纲，内部自动归一化；零向量抛错）。
        target_point_m: 目标配准基准点在世界系下的三维坐标 `[x, y, z]`，单位 `m`。
        target_axis: 目标配准主轴在世界系下的三维方向向量 `[tx, ty, tz]`（无量纲，支持任意三维倾斜角）。
        feature_secondary_axis: 可选手内副参考轴向量 `[sx, sy, sz]`（无量纲，需与 `feature_axis` 不共线），默认 `None`。
        target_secondary_axis: 可选目标副参考轴向量 `[tsx, tsy, tsz]`（无量纲，需与 `target_axis` 不共线），默认 `None`。
        roll_about_target_axis_deg: 在完成主/副轴配准后，绕单位 `target_axis` 附加的右手滚转角，单位 `deg`，默认 `0.0`。
        contact_region: 夹爪接触截面名称（`'pad_center'` | `'finger_middle'` | `'fingertip'`），默认 `'pad_center'`。
        contact_depth_m: 可选自定义接触截面深度偏移，单位 `m`（范围 `[-0.0075, 0.0625]`），默认 `None`。

    返回:
        字典 `{target_pos_m, target_grasp_position_m, target_angles_deg, flange_position_m, quaternion_xyzw, rotation_angle_deg, local_feature_offset_m, local_feature_axis, feature_pos_err_m, feature_axis_err_deg}`：
        - `target_pos_m` / `target_grasp_position_m`: 目标夹爪接触参考点位置 `[x, y, z]`，单位 `m`；
        - `target_angles_deg`: 规约至规范主分支 `[-180,180] x [-90,90] x [-180,180]` 的目标姿态角 `[pitch, roll, yaw]`，单位 `deg`；
        - `flange_position_m`: 对应的 `link6` 法兰中心三维坐标 `[x, y, z]`，单位 `m`；
        - `quaternion_xyzw`: 对应的 `link6` 法兰单位四元数 `[qx, qy, qz, qw]`；
        - `rotation_angle_deg`: 物体从初始姿态到目标姿态的总刚体旋转角，单位 `deg`；
        - `feature_pos_err_m`: 用输出位姿正向反算的配准点残差范数，单位 `m`；
        - `feature_axis_err_deg`: 用输出位姿正向反算的主轴夹角残差，单位 `deg`。

    示例:
        `res = align_in_hand_feature_to_target([0.0, -0.2, 0.85], [0.0, 0.0, 0.0], [0.04, -0.2, 0.83], [1.0, 0.0, 0.0], [0.15, -0.1, 0.92], [0.707, 0.0, -0.707])`
    """
    p_g0 = _vec3(grasp_pos_m, "grasp_pos_m")
    ang_g0 = _vec3(grasp_angles_deg, "grasp_angles_deg")
    p_feat = _vec3(feature_point_m, "feature_point_m")
    u_feat = _unit_vec3(feature_axis, "feature_axis")
    p_targ = _vec3(target_point_m, "target_point_m")
    u_targ = _unit_vec3(target_axis, "target_axis")
    if type(roll_about_target_axis_deg) not in (int, float) or not math.isfinite(roll_about_target_axis_deg):
        raise ValueError("invalid_roll_about_target_axis_deg: expected finite number")
    offset_m = _validate_contact_spec(contact_region=contact_region, contact_depth_m=contact_depth_m)

    if (feature_secondary_axis is None) != (target_secondary_axis is None):
        raise ValueError("feature_secondary_axis and target_secondary_axis must be provided together or both omitted")

    quat_g0 = angles_deg_to_quat_wxyz(float(ang_g0[0]), float(ang_g0[1]), float(ang_g0[2]))
    rot_g0 = quat_wxyz_to_matrix(quat_g0)

    r_local = rot_g0.T @ (p_feat - p_g0)
    u_feat_local = rot_g0.T @ u_feat

    if feature_secondary_axis is not None and target_secondary_axis is not None:
        triad_f = orthonormal_triad_from_axes(u_feat, feature_secondary_axis)
        triad_t = orthonormal_triad_from_axes(u_targ, target_secondary_axis)
        rot_base = triad_t @ triad_f.T
    else:
        rot_base = _minimal_axis_alignment_rotation(u_feat, u_targ, fallback_hint=-rot_g0[:, 1])

    rot_roll = _rodrigues_axis_angle(u_targ, math.radians(float(roll_about_target_axis_deg)))
    rot_align = rot_roll @ rot_base
    rot_g_target = rot_align @ rot_g0
    p_g_target = p_targ - (rot_g_target @ r_local)

    quat_g_wxyz = matrix_to_quat_wxyz(rot_g_target)
    p_deg, r_deg, y_deg = quat_wxyz_to_angles_deg(quat_g_wxyz)
    p_deg = float(np.clip(((p_deg + 180.0) % 360.0) - 180.0, -180.0, 180.0))
    r_deg = float(np.clip(r_deg, -90.0, 90.0))
    y_deg = float(np.clip(((y_deg + 180.0) % 360.0) - 180.0, -180.0, 180.0))
    target_angles = [p_deg, r_deg, y_deg]

    quat_check_wxyz = angles_deg_to_quat_wxyz(p_deg, r_deg, y_deg)
    rot_check = quat_wxyz_to_matrix(quat_check_wxyz)
    feat_pos_after = p_g_target + (rot_check @ r_local)
    feat_axis_after = rot_check @ u_feat_local
    pos_err_m = float(np.linalg.norm(feat_pos_after - p_targ))
    axis_cos = float(np.clip(np.dot(feat_axis_after, u_targ), -1.0, 1.0))
    axis_err_deg = math.degrees(math.acos(axis_cos))

    flange_pos_m = p_g_target - offset_m * rot_check[:, 0]
    quat_xyzw = [float(quat_check_wxyz[1]), float(quat_check_wxyz[2]), float(quat_check_wxyz[3]), float(quat_check_wxyz[0])]
    rot_trace = float(np.clip(0.5 * (np.trace(rot_align) - 1.0), -1.0, 1.0))
    total_rot_deg = math.degrees(math.acos(rot_trace))

    target_pos_list = [float(p_g_target[0]), float(p_g_target[1]), float(p_g_target[2])]
    return dict(
        target_pos_m=target_pos_list,
        target_grasp_position_m=list(target_pos_list),
        target_angles_deg=target_angles,
        flange_position_m=[float(flange_pos_m[0]), float(flange_pos_m[1]), float(flange_pos_m[2])],
        quaternion_xyzw=quat_xyzw,
        rotation_angle_deg=round(total_rot_deg, 5),
        local_feature_offset_m=[round(float(v), 6) for v in r_local],
        local_feature_axis=[round(float(v), 6) for v in u_feat_local],
        feature_pos_err_m=pos_err_m,
        feature_axis_err_deg=axis_err_deg,
    )
