"""无状态三维几何基元库 (`geometry_ops.py`)。

统一收敛 `OBB3D`、`coal` C++ 精确 GJK/EPA 有符号距离与半空间查询、三维多边形与夹爪截面裁剪、倾斜 PCA 标架构造、
多层凸包切片周界采样、`LocalSurfaceKDTree` 局部表面树与实测残差不确定度计算。
所有角度、四元数与旋转矩阵算子统一来自 `ni_rua.core.pose_math` / `ni_rua.perception.external.pose_math`，绝不重复实现。
"""
from __future__ import annotations

import math
from typing import Any

import coal
import numpy as np
from scipy.spatial import cKDTree

if __package__:
    from .pose_math import FINGER_WIDTH_M, PAD_DEPTH_M
else:
    from pose_math import FINGER_WIDTH_M, PAD_DEPTH_M  # type: ignore[no-redef]

# 判定有效凹腔的最小深度阈值，单位 m（4 mm）
MIN_CAVITY_DEPTH_M: float = 0.004
# 默认三维表面体素碰撞半径，单位 m（6 mm）
DEFAULT_SURFACE_VOXEL_RADIUS_M: float = 0.006
# 启用真实三维表面点碰撞替代实心 OBB 的最少非机械臂点数
MIN_OBJECT_SURFACE_POINTS: int = 8
# 单物体表面点碰撞缓存上限（均匀抽稀以保障亚毫秒查询）
MAX_CACHED_OBJECT_POINTS: int = 1600

# 8 个单位立方体顶点符号
_OBB_VERTEX_SIGNS = np.array(
    [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)

# 12 条棱连接的顶点索引对
_OBB_EDGE_PAIRS = np.array(
    [
        [0, 1], [2, 3], [4, 5], [6, 7],
        [0, 2], [1, 3], [4, 6], [5, 7],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ],
    dtype=np.int64,
)

# 6 个面的顶点索引（每面 4 个共面顶点，按逆时针顺序）
_OBB_FACE_QUADS = (
    (0, 1, 3, 2),
    (4, 6, 7, 5),
    (0, 4, 5, 1),
    (2, 3, 7, 6),
    (0, 2, 6, 4),
    (1, 5, 7, 3),
)
_FACE_QUAD_IDX = np.asarray(_OBB_FACE_QUADS, dtype=np.intp)


def _r(value: Any, digits: int = 4) -> float | list[float] | None:
    """数值与数组四舍五入辅助函数。"""
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return [round(float(v), digits) for v in np.asarray(value, dtype=float).reshape(-1)]
    return round(float(value), digits)


def _num(value: Any) -> bool:
    """判断输入是否为有限实数（排除 `bool` 与 `NaN`/`Inf`）。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


class OBB3D:
    """三维定向包围盒 (Oriented Bounding Box)。

    参数:
        center_m: 世界系中心三维坐标 `[x, y, z]`，单位 `m`。
        axes: 形状 `(3, 3)` 的右手正交矩阵（列向量为局部主轴）。
        half_extents_m: 三轴半尺寸 `[hx, hy, hz]`，单位 `m`（内部保底 `1e-5 m`）。
        label: 可选标识字符串。

    示例:
        `box = OBB3D([0.0, 0.0, 0.8], np.eye(3), [0.02, 0.02, 0.02], label="o1")`
    """

    def __init__(self, center_m: Any, axes: Any, half_extents_m: Any, label: str = ""):
        self.center = np.asarray(center_m, dtype=np.float64).reshape(3)
        self.axes = np.asarray(axes, dtype=np.float64).reshape(3, 3)
        self.half_extents = np.maximum(np.asarray(half_extents_m, dtype=np.float64).reshape(3), 1e-5)
        self.label = str(label)

    def vertices(self) -> np.ndarray:
        """返回 OBB 的 8 个三维顶点世界坐标，形状 `(8, 3)`，单位 `m`。"""
        scaled = _OBB_VERTEX_SIGNS * self.half_extents[None, :]
        return self.center[None, :] + scaled @ self.axes.T

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """返回世界系下的轴对齐外包盒 `(min_xyz, max_xyz)`，单位 `m`。"""
        ext = np.abs(self.axes) @ self.half_extents
        return self.center - ext, self.center + ext

    def clamp_point(self, point_m: np.ndarray) -> np.ndarray:
        """将单个世界系三维点（单位 `m`）投影钳位到本 OBB 表面或内部最近点。"""
        local = (point_m - self.center) @ self.axes
        clamped = np.clip(local, -self.half_extents, self.half_extents)
        return self.center + clamped @ self.axes.T

    def clamp_points(self, points_m: np.ndarray) -> np.ndarray:
        """批量将形状 `(N, 3)` 的世界系三维点集（单位 `m`）钳位到本 OBB 表面或内部最近点。"""
        local = (points_m - self.center[None, :]) @ self.axes
        clamped = np.clip(local, -self.half_extents[None, :], self.half_extents[None, :])
        return self.center[None, :] + clamped @ self.axes.T


def obb_signed_distance(obb_a: OBB3D, obb_b: OBB3D) -> float:
    """使用 `coal` C++ GJK/EPA 计算两个 3D OBB 之间的精确有符号距离（单位 `m`）。

    参数:
        obb_a: 第一个三维定向包围盒 `OBB3D`（单位 `m`）。
        obb_b: 第二个三维定向包围盒 `OBB3D`（单位 `m`）。

    返回:
        `> 0` 表示两包围盒分离，返回最小欧氏净空距离（单位 `m`）；
        `<= 0` 表示两包围盒发生穿透，返回负的最小分离平移深度（单位 `m`）。

    示例:
        `d_m = obb_signed_distance(box_a, box_b)`
    """
    sa = 2.0 * obb_a.half_extents
    sb = 2.0 * obb_b.half_extents
    req = coal.DistanceRequest(True)
    res = coal.DistanceResult()
    return float(
        coal.distance(
            coal.Box(float(sa[0]), float(sa[1]), float(sa[2])),
            coal.Transform3s(obb_a.axes, obb_a.center),
            coal.Box(float(sb[0]), float(sb[1]), float(sb[2])),
            coal.Transform3s(obb_b.axes, obb_b.center),
            req,
            res,
        )
    )


def obb_to_halfspace_signed_distance(
    obb: OBB3D,
    normal: Any = (0.0, 0.0, 1.0),
    offset_m: float = 0.0,
) -> float:
    """使用 `coal.Halfspace` 计算三维定向包围盒 `obb` 到实心半空间 `n · x <= offset_m` 的精确有符号距离（单位 `m`）。

    参数:
        obb: 待测三维定向包围盒 `OBB3D`（单位 `m`）。
        normal: 半空间单位外法向三维向量 `[nx, ny, nz]`（无量纲，内部自动归一化），默认 `(0.0, 0.0, 1.0)` 表示水平支撑台面。
        offset_m: 半空间沿外法向的有符号平面偏移 `d`，满足实体区域 `n · x <= d`，单位 `m`，默认 `0.0`（例如 `table_z_m=0.765`）。

    返回:
        有符号距离（单位 `m`）：`> 0` 表示包围盒完全位于半空间外侧的净空距离，`<= 0` 表示穿入半空间的负穿透深度。

    示例:
        `clr_m = obb_to_halfspace_signed_distance(finger_obb, normal=[0.0, 0.0, 1.0], offset_m=0.765)`
    """
    n_vec = np.asarray(normal, dtype=np.float64).reshape(3)
    n_norm = float(np.linalg.norm(n_vec))
    if not math.isfinite(n_norm) or n_norm <= 1e-9:
        raise ValueError("invalid_halfspace_normal")
    n_unit = n_vec / n_norm
    sz = 2.0 * obb.half_extents
    req = coal.DistanceRequest(True)
    res = coal.DistanceResult()
    return float(
        coal.distance(
            coal.Box(float(sz[0]), float(sz[1]), float(sz[2])),
            coal.Transform3s(obb.axes, obb.center),
            coal.Halfspace(n_unit, float(offset_m)),
            coal.Transform3s(),
            req,
            res,
        )
    )


def _clamp_single_point_upper_bound(
    centers_a: np.ndarray,
    axes_a: np.ndarray,
    half_a: np.ndarray,
    centers_b: np.ndarray,
    axes_b: np.ndarray,
    half_b: np.ndarray,
) -> np.ndarray:
    """返回两 OBB 中心点间距作为合法欧氏距离上界 `||c_b - c_a|| >= d(A, B)`，形状 `(M,)`，单位 `m`。"""
    _ = axes_a, half_a, axes_b, half_b
    return np.linalg.norm(np.asarray(centers_b, dtype=np.float64) - np.asarray(centers_a, dtype=np.float64), axis=1)


def obb_signed_distance_batch(
    centers_a: np.ndarray,
    axes_a: np.ndarray,
    half_extents_a: np.ndarray,
    centers_b: np.ndarray,
    axes_b: np.ndarray,
    half_extents_b: np.ndarray,
    *,
    skip_above_m: float | None = None,
) -> np.ndarray:
    """使用 `coal` C++ GJK/EPA 批量计算 `N` 对 3D OBB 的精确有符号欧氏距离（单位 `m`）。

    参数:
        centers_a: 第一组 OBB 中心坐标，形状 `(N, 3)`，单位 `m`。
        axes_a: 第一组 OBB 旋转矩阵（列向量为局部正交主轴），形状 `(N, 3, 3)`。
        half_extents_a: 第一组 OBB 半轴长，形状 `(N, 3)`，单位 `m`（内部保底 `1e-5`）。
        centers_b: 第二组 OBB 中心坐标，形状 `(N, 3)`，单位 `m`。
        axes_b: 第二组 OBB 旋转矩阵，形状 `(N, 3, 3)`。
        half_extents_b: 第二组 OBB 半轴长，形状 `(N, 3)`，单位 `m`。
        skip_above_m: 可选净空截断阈值，单位 `m`；若世界系 AABB 下界已严格大于该阈值，直接返回该合法下界跳过 GJK 求解。

    返回:
        形状为 `(N,)` 的有符号距离数组（`<= 0` 为负穿透深度，`> 0` 为分离净空，单位 `m`）。

    示例:
        `dists_m = obb_signed_distance_batch(ca, aa, ha, cb, ab, hb, skip_above_m=0.05)`
    """
    ca = np.asarray(centers_a, dtype=np.float64).reshape(-1, 3)
    n = ca.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    cb = np.asarray(centers_b, dtype=np.float64).reshape(n, 3)
    aa = np.asarray(axes_a, dtype=np.float64).reshape(n, 3, 3)
    ab = np.asarray(axes_b, dtype=np.float64).reshape(n, 3, 3)
    ha = np.maximum(np.asarray(half_extents_a, dtype=np.float64).reshape(n, 3), 1e-5)
    hb = np.maximum(np.asarray(half_extents_b, dtype=np.float64).reshape(n, 3), 1e-5)

    if skip_above_m is not None:
        cutoff = float(skip_above_m)
        ext_a = np.einsum("nij,nj->ni", np.abs(aa), ha)
        ext_b = np.einsum("nij,nj->ni", np.abs(ab), hb)
        gap_xyz = np.maximum(0.0, np.abs(cb - ca) - ext_a - ext_b)
        aabb_lb = np.linalg.norm(gap_xyz, axis=1)
        out = aabb_lb.copy()
        act = np.flatnonzero(aabb_lb <= cutoff)
        if act.size == 0:
            return out
    else:
        out = np.empty(n, dtype=np.float64)
        act = np.arange(n, dtype=np.intp)

    req = coal.DistanceRequest(True)
    res = coal.DistanceResult()
    sa = 2.0 * ha
    sb = 2.0 * hb
    for idx in act.tolist():
        res.clear()
        out[idx] = float(
            coal.distance(
                coal.Box(float(sa[idx, 0]), float(sa[idx, 1]), float(sa[idx, 2])),
                coal.Transform3s(aa[idx], ca[idx]),
                coal.Box(float(sb[idx, 0]), float(sb[idx, 1]), float(sb[idx, 2])),
                coal.Transform3s(ab[idx], cb[idx]),
                req,
                res,
            )
        )
    return out


def obb_sat_max_gap_batch(
    centers_a: np.ndarray,
    axes_a: np.ndarray,
    half_extents_a: np.ndarray,
    centers_b: np.ndarray,
    axes_b: np.ndarray,
    half_extents_b: np.ndarray,
) -> np.ndarray:
    """兼容重导出门面：委托 `coal` `obb_signed_distance_batch(..., skip_above_m=0.0)` 返回穿透深度或分离下界（单位 `m`）。"""
    return obb_signed_distance_batch(
        centers_a, axes_a, half_extents_a, centers_b, axes_b, half_extents_b, skip_above_m=0.0
    )


def obb_signed_distance_pairs(
    obbs_a: list[OBB3D],
    obbs_b: list[OBB3D],
    *,
    skip_above_m: float | None = None,
) -> np.ndarray:
    """对等长 `OBB3D` 列表对批量计算三维有符号距离（单位 `m`）。

    参数:
        obbs_a: 第一组 `OBB3D` 列表（单位 `m`）。
        obbs_b: 第二组等长 `OBB3D` 列表（单位 `m`）。
        skip_above_m: 可选净空截断上界，单位 `m`。

    返回:
        形状 `(len(obbs_a),)` 的有符号距离数组（单位 `m`）。

    示例:
        `dists = obb_signed_distance_pairs([b1], [b2])`
    """
    if len(obbs_a) != len(obbs_b):
        raise ValueError("obbs_a and obbs_b must have identical length")
    if not obbs_a:
        return np.empty(0, dtype=np.float64)
    ca = np.stack([o.center for o in obbs_a], axis=0)
    aa = np.stack([o.axes for o in obbs_a], axis=0)
    ha = np.stack([o.half_extents for o in obbs_a], axis=0)
    cb = np.stack([o.center for o in obbs_b], axis=0)
    ab = np.stack([o.axes for o in obbs_b], axis=0)
    hb = np.stack([o.half_extents for o in obbs_b], axis=0)
    return obb_signed_distance_batch(ca, aa, ha, cb, ab, hb, skip_above_m=skip_above_m)


def batch_obbs_to_slabs_min_distance(query_obbs: list[OBB3D], slab_obbs: list[OBB3D]) -> float:
    """使用 `coal` C++ GJK/EPA 与 AABB 分支定界批量计算 `query_obbs` 到 `slab_obbs` 的全局最小有符号距离。

    参数:
        query_obbs: 查询部件 `OBB3D` 列表（单位 `m`）。
        slab_obbs: 目标障碍 `OBB3D` 列表（单位 `m`）。

    返回:
        全组合中的最小有符号距离（单位 `m`）；任一列表为空时返回 `+inf`。

    示例:
        `min_d = batch_obbs_to_slabs_min_distance(moving_obbs, [other_obb])`
    """
    if not query_obbs or not slab_obbs:
        return float("inf")
    q_items = [
        (
            coal.Box(*(2.0 * b.half_extents)),
            coal.Transform3s(b.axes, b.center),
            b.center,
            np.abs(b.axes) @ b.half_extents,
        )
        for b in query_obbs
    ]
    s_items = [
        (
            coal.Box(*(2.0 * b.half_extents)),
            coal.Transform3s(b.axes, b.center),
            b.center,
            np.abs(b.axes) @ b.half_extents,
        )
        for b in slab_obbs
    ]
    req = coal.DistanceRequest(True)
    res = coal.DistanceResult()
    best = float("inf")
    for q_box, q_tf, q_c, q_ext in q_items:
        for s_box, s_tf, s_c, s_ext in s_items:
            if best < float("inf"):
                gap = np.maximum(0.0, np.abs(s_c - q_c) - q_ext - s_ext)
                if float(gap @ gap) >= (max(best, 0.0) ** 2) and best > 0.0:
                    continue
                if best <= 0.0 and np.any(gap > 0.0):
                    continue
            res.clear()
            d = float(coal.distance(q_box, q_tf, s_box, s_tf, req, res))
            if d < best:
                best = d
    return best


def obb_to_points_signed_distance(
    query_obb: OBB3D,
    points_m: np.ndarray,
    voxel_radius_m: float = DEFAULT_SURFACE_VOXEL_RADIUS_M,
) -> float:
    """计算单个三维有向包围盒 `query_obb` 到真实三维表面点集 `points_m`（体素半径 `voxel_radius_m`）的有符号距离。

    参数:
        query_obb: 待测运动部件（如左指、右指、手掌或持物）的 `OBB3D`（单位 `m`）。
        points_m: 形状 `(M, 3)` 的真实表面点世界坐标数组（单位 `m`）。
        voxel_radius_m: 表面点体素球半径，单位 `m`，默认 `0.006` (`6 mm`)。

    返回:
        最小有符号距离（单位 `m`）：`> 0` 表示无碰撞净空，`< 0` 表示发生实体干涉穿透。

    示例:
        `d_m = obb_to_points_signed_distance(finger_obb, pts_world, voxel_radius_m=0.006)`
    """
    pts = np.asarray(points_m, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return math.inf
    rel = (pts - query_obb.center[None, :]) @ query_obb.axes
    q = np.abs(rel) - query_obb.half_extents[None, :]
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return float(np.min(outside + inside)) - float(voxel_radius_m)


def obb_to_points_signed_distance_batch(
    centers_w: np.ndarray,
    axes_w: np.ndarray,
    halves: np.ndarray,
    points_m: np.ndarray,
    voxel_radius_m: float = DEFAULT_SURFACE_VOXEL_RADIUS_M,
    *,
    skip_above_m: float | None = None,
) -> np.ndarray:
    """对裸表面点集 `points_m` 批量计算 `Q` 个查询盒的精确有符号距离（单位 `m`）。

    参数:
        centers_w: 查询盒世界系中心坐标数组，形状 `(Q, 3)`，单位 `m`。
        axes_w: 查询盒世界系正交主轴矩阵数组，形状 `(Q, 3, 3)`。
        halves: 查询盒半轴长数组，形状 `(Q, 3)`，单位 `m`。
        points_m: 表面点世界坐标数组，形状 `(M, 3)`，单位 `m`。
        voxel_radius_m: 表面点体素球半径，单位 `m`，默认 `0.006`。
        skip_above_m: 可选净空截断阈值，单位 `m`。

    返回:
        形状 `(Q,)` 的有符号距离数组（单位 `m`）。

    示例:
        `dists = obb_to_points_signed_distance_batch(cw, aw, hw, pts, skip_above_m=0.05)`
    """
    cw = np.asarray(centers_w, dtype=np.float64).reshape(-1, 3)
    q_cnt = cw.shape[0]
    if q_cnt == 0:
        return np.empty(0, dtype=np.float64)
    pts = np.asarray(points_m, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.full(q_cnt, float("inf"), dtype=np.float64)
    aw = np.asarray(axes_w, dtype=np.float64).reshape(q_cnt, 3, 3)
    hw = np.asarray(halves, dtype=np.float64).reshape(q_cnt, 3)
    vox = float(voxel_radius_m)

    if skip_above_m is not None:
        pts_c = np.mean(pts, axis=0)
        pts_r = float(np.max(np.linalg.norm(pts - pts_c[None, :], axis=1)))
        box_r = np.linalg.norm(hw, axis=1)
        sphere_lb = np.linalg.norm(cw - pts_c[None, :], axis=1) - pts_r - box_r - vox
        out = sphere_lb.copy()
        act = np.flatnonzero(sphere_lb <= float(skip_above_m))
        if act.size == 0:
            return out
    else:
        out = np.empty(q_cnt, dtype=np.float64)
        act = np.arange(q_cnt, dtype=np.intp)

    for start in range(0, act.size, 32):
        sub = act[start : start + 32]
        rel = np.einsum("qmd,qdk->qmk", pts[None, :, :] - cw[sub, None, :], aw[sub])
        q_box = np.abs(rel) - hw[sub, None, :]
        outside = np.linalg.norm(np.maximum(q_box, 0.0), axis=2)
        inside = np.minimum(np.max(q_box, axis=2), 0.0)
        out[sub] = np.min(outside + inside, axis=1) - vox
    return out


_obb_to_points_signed_distance_batch = obb_to_points_signed_distance_batch


def obb_points_sdf_and_grad(
    points_m: np.ndarray,
    center_m: np.ndarray,
    axes: np.ndarray,
    half_extents_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """向量化计算查询点集 `points_m` (`(M, 3)`, 单位 `m`) 到单个有向包围盒 `OBB3D` 的精确有符号距离与单位外法向梯度。

    参数:
        points_m: 查询三维点集，形状 `(M, 3)`，单位 `m`。
        center_m: 包围盒世界系中心 `[x, y, z]`，单位 `m`。
        axes: 包围盒 `(3, 3)` 正交主轴矩阵（列向量）。
        half_extents_m: 包围盒半轴长 `[hx, hy, hz]`，单位 `m`。

    返回:
        二元组 `(sdf, grad_world)`：`sdf` 形状 `(M,)` 单位 `m`；`grad_world` 形状 `(M, 3)` 为单位外法向排斥梯度。

    示例:
        `sdf, grad = obb_points_sdf_and_grad(pts, box.center, box.axes, box.half_extents)`
    """
    pts = np.asarray(points_m, dtype=np.float64).reshape(-1, 3)
    c = np.asarray(center_m, dtype=np.float64).reshape(3)
    ax = np.asarray(axes, dtype=np.float64).reshape(3, 3)
    he = np.asarray(half_extents_m, dtype=np.float64).reshape(3)
    if pts.shape[0] == 0:
        return np.empty(0, dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    p_loc = (pts - c[None, :]) @ ax
    abs_p = np.abs(p_loc)
    q = abs_p - he[None, :]

    outside = np.maximum(q, 0.0)
    out_norm = np.linalg.norm(outside, axis=1)
    inside_dist = np.max(q, axis=1)
    is_out = out_norm > 1e-12
    sdf = np.where(is_out, out_norm, inside_dist)

    signs = np.where(p_loc >= 0.0, 1.0, -1.0)
    grad_out_loc = signs * (outside / np.maximum(out_norm[:, None], 1e-12))

    face_idx = np.argmax(q, axis=1)
    grad_in_loc = np.zeros_like(p_loc)
    rows = np.arange(pts.shape[0])
    grad_in_loc[rows, face_idx] = signs[rows, face_idx]

    grad_loc = np.where(is_out[:, None], grad_out_loc, grad_in_loc)
    grad_world = grad_loc @ ax.T
    norms = np.linalg.norm(grad_world, axis=1, keepdims=True)
    grad_world = grad_world / np.maximum(norms, 1e-12)
    return sdf, grad_world


def clip_polygon_nd(poly: np.ndarray, axis: int, bound: float, keep_greater: bool) -> np.ndarray:
    """使用半空间 `poly[:, axis] >= bound`（或 `<= bound`）裁剪任意维度凸多边形 (Sutherland-Hodgman)。

    参数:
        poly: 顶点有序多边形数组，形状 `(V, D)`，单位 `m`。
        axis: 裁剪坐标轴下标 (`0 <= axis < D`)。
        bound: 半空间边界坐标阈值，单位 `m`。
        keep_greater: `True` 保留 `>= bound`，`False` 保留 `<= bound`。

    返回:
        裁剪后的凸多边形顶点数组，形状 `(V', D)`，单位 `m`。

    示例:
        `clipped = clip_polygon_nd(quad, axis=1, bound=-0.009, keep_greater=True)`
    """
    if len(poly) == 0:
        return poly
    out = []
    n = len(poly)
    for i in range(n):
        cur = poly[i]
        prev = poly[(i - 1) % n]
        cur_in = (cur[axis] >= bound) if keep_greater else (cur[axis] <= bound)
        prev_in = (prev[axis] >= bound) if keep_greater else (prev[axis] <= bound)
        if cur_in != prev_in:
            denom = cur[axis] - prev[axis]
            t = (bound - prev[axis]) / denom if abs(denom) > 1e-12 else 0.0
            out.append(prev + np.clip(t, 0.0, 1.0) * (cur - prev))
        if cur_in:
            out.append(cur)
    return np.asarray(out, dtype=np.float64) if out else np.zeros((0, poly.shape[1]), dtype=np.float64)


_clip_polygon_nd = clip_polygon_nd


def clip_obb_faces_to_slab(verts_local: np.ndarray, axis: int, lo: float, hi: float) -> list[np.ndarray]:
    """将局部标架下的 OBB 6 个表面多边形裁剪到指定坐标轴区间 `[lo, hi]` 内（单位 `m`）。

    参数:
        verts_local: 局部标架下 OBB 的 8 个顶点坐标，形状 `(8, 3)`，单位 `m`。
        axis: 裁剪轴索引 (`0`, `1` 或 `2`)。
        lo: 区间下界，单位 `m`。
        hi: 区间上界，单位 `m`。

    返回:
        落在平板 `[lo, hi]` 内的裁剪面多边形列表。

    示例:
        `faces = clip_obb_faces_to_slab(verts_g, axis=1, lo=-0.009, hi=0.009)`
    """
    clipped_faces: list[np.ndarray] = []
    for quad in _OBB_FACE_QUADS:
        poly = verts_local[list(quad)]
        poly = clip_polygon_nd(poly, axis=axis, bound=lo, keep_greater=True)
        poly = clip_polygon_nd(poly, axis=axis, bound=hi, keep_greater=False)
        if len(poly) > 0:
            clipped_faces.append(poly)
    return clipped_faces


_clip_obb_faces_to_slab = clip_obb_faces_to_slab


def clip_target_obb_in_gripper_frames(
    target_obb: OBB3D,
    pad_centers: np.ndarray,
    g_bases: np.ndarray,
    *,
    finger_width_m: float = FINGER_WIDTH_M,
    pad_depth_m: float = PAD_DEPTH_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """在 `G` 个候选夹爪局部正交标架 `(u_close, u_tan, -u_app)` 下精确裁剪目标 3D OBB 咬合窗口。

    与逐面 Sutherland-Hodgman 裁剪 `clip_obb_faces_to_slab` 精确代数等价。

    参数:
        target_obb: 目标物体 `OBB3D`（单位 `m`）。
        pad_centers: 候选指垫中心坐标数组，形状 `(G, 3)`，单位 `m`。
        g_bases: 候选夹爪正交基底 `(u_close, u_tan, -u_app)`，形状 `(G, 3, 3)`。
        finger_width_m: 夹爪指宽，单位 `m`，默认 `0.018`。
        pad_depth_m: 指垫有效接触深度，单位 `m`，默认 `0.015`。

    返回:
        六元组 `(footprint_hit, vertical_bite_m, c_min, c_max, projected_span_m, lateral_eccentricity_m)`，
        其中长度单位均为 `m`。

    示例:
        `hit, bite, c0, c1, span, ecc = clip_target_obb_in_gripper_frames(obb, pad_c, g_bases)`
    """
    half_fw = 0.5 * float(finger_width_m)
    half_pd = 0.5 * float(pad_depth_m)

    target_verts = target_obb.vertices()
    verts_g = np.einsum("gvd,gdk->gvk", target_verts[None, :, :] - pad_centers[:, None, :], g_bases)
    e0 = verts_g[:, _OBB_EDGE_PAIRS[:, 0], :]
    e1 = verts_g[:, _OBB_EDGE_PAIRS[:, 1], :]
    de = e1 - e0

    v_in_w = (verts_g[:, :, 1] >= -half_fw) & (verts_g[:, :, 1] <= half_fw)
    dy = de[:, :, 1]
    valid_dy = np.abs(dy) > 1e-12
    safe_dy = np.where(valid_dy, dy, 1.0)

    cross_ylo = (e0[:, :, 1] >= -half_fw) != (e1[:, :, 1] >= -half_fw)
    t_ylo = np.where(valid_dy, np.minimum(np.maximum((-half_fw - e0[:, :, 1]) / safe_dy, 0.0), 1.0), 0.0)
    p_ylo = e0 + t_ylo[:, :, None] * de

    cross_yhi = (e0[:, :, 1] <= half_fw) != (e1[:, :, 1] <= half_fw)
    t_yhi = np.where(valid_dy, np.minimum(np.maximum((half_fw - e0[:, :, 1]) / safe_dy, 0.0), 1.0), 0.0)
    p_yhi = e0 + t_yhi[:, :, None] * de

    w_pts = np.concatenate([verts_g, p_ylo, p_yhi], axis=1)
    w_mask = np.concatenate([v_in_w, cross_ylo, cross_yhi], axis=1)
    footprint_hit = np.any(w_mask, axis=1)

    z_w_min = np.min(np.where(w_mask, w_pts[:, :, 2], np.inf), axis=1)
    z_w_max = np.max(np.where(w_mask, w_pts[:, :, 2], -np.inf), axis=1)
    vertical_bite_m = np.where(
        footprint_hit,
        np.maximum(0.0, np.minimum(z_w_max, half_pd) - np.maximum(z_w_min, -half_pd)),
        0.0,
    )

    w_in_jaw = w_mask & (w_pts[:, :, 2] >= -half_pd) & (w_pts[:, :, 2] <= half_pd)
    dz = de[:, :, 2]
    valid_dz = np.abs(dz) > 1e-12
    safe_dz = np.where(valid_dz, dz, 1.0)

    cross_zlo = (e0[:, :, 2] >= -half_pd) != (e1[:, :, 2] >= -half_pd)
    t_zlo = np.where(valid_dz, np.minimum(np.maximum((-half_pd - e0[:, :, 2]) / safe_dz, 0.0), 1.0), 0.0)
    p_zlo = e0 + t_zlo[:, :, None] * de
    zlo_in_w = cross_zlo & (p_zlo[:, :, 1] >= -half_fw) & (p_zlo[:, :, 1] <= half_fw)

    cross_zhi = (e0[:, :, 2] <= half_pd) != (e1[:, :, 2] <= half_pd)
    t_zhi = np.where(valid_dz, np.minimum(np.maximum((half_pd - e0[:, :, 2]) / safe_dz, 0.0), 1.0), 0.0)
    p_zhi = e0 + t_zhi[:, :, None] * de
    zhi_in_w = cross_zhi & (p_zhi[:, :, 1] >= -half_fw) & (p_zhi[:, :, 1] <= half_fw)

    corner_g = np.array(
        [
            [0.0, -half_fw, -half_pd],
            [0.0, -half_fw, half_pd],
            [0.0, half_fw, -half_pd],
            [0.0, half_fw, half_pd],
        ],
        dtype=np.float64,
    )
    orig_world = pad_centers[:, None, :] + np.einsum("ck,gdk->gcd", corner_g, g_bases)
    orig_loc = np.einsum("gcd,dk->gck", orig_world - target_obb.center[None, None, :], target_obb.axes)
    dir_loc = np.einsum("gd,dk->gk", g_bases[:, :, 0], target_obb.axes)[:, None, :]
    half_t = target_obb.half_extents[None, None, :]

    par = np.abs(dir_loc) <= 1e-12
    safe_dir = np.where(par, 1.0, dir_loc)
    t1 = (-half_t - orig_loc) / safe_dir
    t2 = (half_t - orig_loc) / safe_dir
    in_par_slab = np.abs(orig_loc) <= half_t
    t_lo_ax = np.where(par, np.where(in_par_slab, -np.inf, np.inf), np.minimum(t1, t2))
    t_hi_ax = np.where(par, np.where(in_par_slab, np.inf, -np.inf), np.maximum(t1, t2))
    x_enter = np.max(t_lo_ax, axis=2)
    x_exit = np.min(t_hi_ax, axis=2)
    ray_hit = x_enter <= x_exit

    jaw_x = np.concatenate([w_pts[:, :, 0], p_zlo[:, :, 0], p_zhi[:, :, 0], x_enter, x_exit], axis=1)
    jaw_mask = np.concatenate([w_in_jaw, zlo_in_w, zhi_in_w, ray_hit, ray_hit], axis=1)
    any_jaw = np.any(jaw_mask, axis=1)

    c_min_jaw = np.min(np.where(jaw_mask, jaw_x, np.inf), axis=1)
    c_max_jaw = np.max(np.where(jaw_mask, jaw_x, -np.inf), axis=1)
    c_min_w = np.min(np.where(w_mask, w_pts[:, :, 0], np.inf), axis=1)
    c_max_w = np.max(np.where(w_mask, w_pts[:, :, 0], -np.inf), axis=1)
    c_min_all = np.min(verts_g[:, :, 0], axis=1)
    c_max_all = np.max(verts_g[:, :, 0], axis=1)

    c_min = np.where(footprint_hit, np.where(any_jaw, c_min_jaw, c_min_w), c_min_all)
    c_max = np.where(footprint_hit, np.where(any_jaw, c_max_jaw, c_max_w), c_max_all)
    projected_span_m = np.maximum(0.0, c_max - c_min)
    lateral_eccentricity_m = np.abs(0.5 * (c_min + c_max))
    return footprint_hit, vertical_bite_m, c_min, c_max, projected_span_m, lateral_eccentricity_m


_clip_target_obb_in_gripper_frames = clip_target_obb_in_gripper_frames


def compute_tilted_obb_frame(
    length: float,
    width: float,
    h_span: float,
    yaw_deg: float,
    tilt_deg: float | None,
    axis_3d: Any = None,
    tilt_sign: float = 1.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """当物体具备物理自洽的三维倾斜主轴（`10 deg <= tilt_deg <= 80 deg` 或显式 `axis_3d`）时，构造右手正交倾斜标架与固有半轴长。

    参数:
        length: 水平主轴包络长度，单位 `m`。
        width: 水平短轴包络宽度，单位 `m`。
        h_span: 竖直高度包络跨度，单位 `m`。
        yaw_deg: 水平偏航角，单位 `deg`。
        tilt_deg: 与世界 `+Z` 轴夹角，单位 `deg`；`None` 时仅当 `axis_3d` 有效才构造。
        axis_3d: 可选三维主轴向量 `[ax, ay, az]`。
        tilt_sign: 倾斜方位符号 (`+1.0` 或 `-1.0`)。

    返回:
        `(axes, half_extents)`：形状 `(3, 3)` 的右手正交矩阵与形状 `(3,)` 的半轴长（单位 `m`）；不倾斜时返回 `(None, None)`。

    示例:
        `axes, half = compute_tilted_obb_frame(0.113, 0.020, 0.113, 0.0, 45.0)`
    """
    if axis_3d is not None:
        vec = np.asarray(axis_3d, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(vec))
        if math.isfinite(norm) and norm > 1e-6:
            u_long = vec / norm
            if u_long[2] < 0.0:
                u_long = -u_long
            sin_t = math.hypot(float(u_long[0]), float(u_long[1]))
            cos_t = abs(float(u_long[2]))
            if sin_t > 1e-3 and cos_t > 1e-3:
                u_short = np.array([-u_long[1] / sin_t, u_long[0] / sin_t, 0.0], dtype=np.float64)
                u_norm = np.cross(u_long, u_short)
                t_3d = max(0.004, min(width, length, h_span))
                l_3d = max(t_3d, (length + h_span) / (sin_t + cos_t) - t_3d)
                half_ext = np.array([0.5 * l_3d, 0.5 * width, 0.5 * t_3d], dtype=np.float64)
                return np.column_stack([u_long, u_short, u_norm]), half_ext

    if not _num(tilt_deg):
        return None, None
    t_deg = float(tilt_deg)
    if not (10.0 <= t_deg <= 80.0):
        return None, None

    theta = math.radians(t_deg)
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)
    t_3d = max(0.004, min(width, length, h_span))
    l_3d = max(t_3d, (length + h_span) / (sin_t + cos_t) - t_3d)
    if l_3d < 1.4 * max(width, t_3d):
        return None, None

    est_xy = l_3d * sin_t + t_3d * cos_t
    est_z = l_3d * cos_t + t_3d * sin_t
    if abs(est_xy - length) > max(0.018, 0.30 * length) or abs(est_z - h_span) > max(0.018, 0.30 * h_span):
        return None, None

    yaw_rad = math.radians(float(yaw_deg))
    sgn = -1.0 if float(tilt_sign) < 0.0 else 1.0
    u_long = np.array([sgn * sin_t * math.cos(yaw_rad), sgn * sin_t * math.sin(yaw_rad), cos_t], dtype=np.float64)
    u_short = np.array([-math.sin(yaw_rad), math.cos(yaw_rad), 0.0], dtype=np.float64)
    u_norm = np.cross(u_long, u_short)
    half_ext = np.array([0.5 * l_3d, 0.5 * width, 0.5 * t_3d], dtype=np.float64)
    return np.column_stack([u_long, u_short, u_norm]), half_ext


_compute_tilted_obb_frame = compute_tilted_obb_frame


def sample_slice_hull_points(
    slices_m: list[dict],
    bottom_z: float,
    top_z: float,
    *,
    cavity_depth_m: float = 0.0,
) -> np.ndarray:
    """将实测多层水平凸包切片 `surface_slices_m` 采样为三维周界表面点集（单位 `m`）。

    凹腔开口绝不封顶盖，仅在实测腔底采样底面。

    参数:
        slices_m: 水平凸包切片列表，每项含 `z_m`（单位 `m`）与 `hull_xy_m`（二维凸包顶点列表，单位 `m`）。
        bottom_z: 物体底面高程，单位 `m`。
        top_z: 物体顶面高程，单位 `m`。
        cavity_depth_m: 实测凹腔深度，单位 `m`，默认 `0.0`。

    返回:
        三维表面采样点数组，形状 `(M, 3)`，单位 `m`。

    示例:
        `pts = sample_slice_hull_points(slices, 0.765, 0.845, cavity_depth_m=0.07)`
    """
    valid_slices = []
    for item in slices_m or []:
        if not isinstance(item, dict) or not _num(item.get("z_m")):
            continue
        hull = item.get("hull_xy_m")
        if not isinstance(hull, (list, tuple)) or len(hull) < 3:
            continue
        poly = np.asarray(hull, dtype=np.float64)
        if poly.ndim == 2 and poly.shape[1] == 2 and np.all(np.isfinite(poly)):
            valid_slices.append((float(item["z_m"]), poly))
    if not valid_slices:
        return np.empty((0, 3), dtype=np.float64)

    valid_slices.sort(key=lambda pair: pair[0])
    pts_out: list[np.ndarray] = []
    n_s = len(valid_slices)
    cav_f = float(cavity_depth_m) if _num(cavity_depth_m) else 0.0
    is_open_cavity = cav_f >= MIN_CAVITY_DEPTH_M
    for idx, (z_mid, poly) in enumerate(valid_slices):
        z_lo = bottom_z if idx == 0 else 0.5 * (valid_slices[idx - 1][0] + z_mid)
        z_hi = top_z if idx == n_s - 1 else 0.5 * (z_mid + valid_slices[idx + 1][0])
        n_z = max(2, int(math.ceil(max(0.004, z_hi - z_lo) / 0.010)) + 1)
        zs = np.linspace(z_lo, z_hi, n_z)
        edge_xy_list = []
        m_v = len(poly)
        for k in range(m_v):
            p0, p1 = poly[k], poly[(k + 1) % m_v]
            seg_len = float(np.linalg.norm(p1 - p0))
            steps = max(1, int(math.ceil(seg_len / 0.008)))
            ts = np.linspace(0.0, 1.0, steps, endpoint=False)[:, None]
            edge_xy_list.append(p0[None, :] + ts * (p1 - p0)[None, :])
        ring_xy = np.vstack(edge_xy_list)
        for zv in zs:
            pts_out.append(np.column_stack([ring_xy, np.full(len(ring_xy), zv)]))
        if idx == n_s - 1 and not is_open_cavity:
            centroid = np.mean(poly, axis=0)
            for alpha in (0.35, 0.70):
                inner_xy = centroid[None, :] + alpha * (poly - centroid[None, :])
                pts_out.append(np.column_stack([inner_xy, np.full(len(inner_xy), z_hi)]))
    if is_open_cavity:
        floor_poly = valid_slices[0][1]
        floor_z = max(float(bottom_z), float(top_z) - cav_f)
        centroid = np.mean(floor_poly, axis=0)
        pts_out.append(np.array([[float(centroid[0]), float(centroid[1]), floor_z]], dtype=np.float64))
        for alpha in (0.35, 0.70):
            inner_xy = centroid[None, :] + alpha * (floor_poly - centroid[None, :])
            pts_out.append(np.column_stack([inner_xy, np.full(len(inner_xy), floor_z)]))
    return np.vstack(pts_out) if pts_out else np.empty((0, 3), dtype=np.float64)


_sample_slice_hull_points = sample_slice_hull_points


def derive_object_spatial_meta(
    length_m: float | None = None,
    width_m: float | None = None,
    top_z_m: float | None = None,
    bottom_z_m: float | None = None,
    *,
    surface_points_m: np.ndarray | None = None,
    depth_std_m: float | None = None,
    residual_std_m: float | None = None,
    **_unused: Any,
) -> dict:
    """仅基于传感器实测离散度或真实三维表面点包络残差标准差客观计算 `uncertainty_m`（不输出任何 `shape_tag`）。

    参数:
        length_m: 可选物体水平主轴长度，单位 `m` (`> 0`)；未传时由 `surface_points_m` 实测主轴跨度计算。
        width_m: 可选物体水平短轴宽度，单位 `m` (`> 0`)；未传时由 `surface_points_m` 实测短轴跨度计算。
        top_z_m: 可选物体世界系顶面高程，单位 `m`；无实测时为 `None`。
        bottom_z_m: 可选物体世界系底面高程，单位 `m`；无实测时为 `None`。
        surface_points_m: 可选非机械臂真实三维表面点数组 `(M, 3)`（单位 `m`）。
        depth_std_m: 可选传感器实测深度标准差，单位 `m` (`>= 0`)。
        residual_std_m: 可选几何拟合实测残差标准差，单位 `m` (`>= 0`)。

    返回:
        字典 `{'uncertainty_m': float | None}`：
        - 当存在实测标准差或 `>= 3` 个三维表面采样点时返回客观残差标准差（单位 `m`）；
        - 无实测点集或离散度输入时诚实返回 `{'uncertainty_m': None}`。

    示例:
        `meta = derive_object_spatial_meta(0.08, 0.06, 0.84, 0.78, residual_std_m=0.0024)`
    """
    for measured in (residual_std_m, depth_std_m):
        if _num(measured) and float(measured) >= 0.0:
            return dict(uncertainty_m=round(float(measured), 4))

    if surface_points_m is not None:
        pts = np.asarray(surface_points_m, dtype=np.float64).reshape(-1, 3)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if len(pts) >= 3:
            lo = np.min(pts, axis=0)
            hi = np.max(pts, axis=0)
            center = 0.5 * (lo + hi)
            centered = pts[:, :2] - center[None, :2]
            cov_xy = centered.T @ centered
            eigvals, eigvecs = np.linalg.eigh(cov_xy)
            major = eigvecs[:, int(np.argmax(eigvals))]
            minor = np.array([-major[1], major[0]], dtype=np.float64)
            local_x = np.abs(centered @ major)
            local_y = np.abs(centered @ minor)
            hx = 0.5 * max(0.0, float(length_m)) if _num(length_m) else float(np.max(local_x))
            hy = 0.5 * max(0.0, float(width_m)) if _num(width_m) else float(np.max(local_y))
            hz = (
                0.5 * max(0.0, float(top_z_m) - float(bottom_z_m))
                if (_num(top_z_m) and _num(bottom_z_m))
                else 0.5 * float(hi[2] - lo[2])
            )
            z_center = 0.5 * (float(top_z_m) + float(bottom_z_m)) if (_num(top_z_m) and _num(bottom_z_m)) else float(center[2])
            local_z = np.abs(pts[:, 2] - z_center)
            face_residuals = np.minimum(
                np.minimum(np.abs(local_x - hx), np.abs(local_y - hy)),
                np.abs(local_z - max(0.0, hz)),
            )
            return dict(uncertainty_m=round(float(np.std(face_residuals)), 4))

    return dict(uncertainty_m=None)


def derive_table_spatial_meta(objects: list[dict], support_z_m: float | None = None) -> dict[str, dict]:
    """Object-keyed measurement dispersion and visible geometry before table-plane rescaling, when available.

    参数:
        objects: 服务端返回的原始物体字典列表（含 `track_id`/`id`、`length_m`、`width_m` 及可选实测残差字段，单位 `m`）。
        support_z_m: 可选实测支撑台面高度，单位 `m`；无实测时保持 `None`，绝不回退默认常数。

    返回:
        `uncertainty_m`: measured scatter in metres when available. For rescaled rows,
        `before_table_rescale.centre_xy_m` is the fitted footprint centre in the table's reference frame
        and `visible_z_range_m` is [p02, p98] of the original visible mask heights (m).

    示例:
        `meta_map = derive_table_spatial_meta(raw_objs, support_z_m=0.765)`
    """
    out: dict[str, dict] = {}
    sup_val = float(support_z_m) if _num(support_z_m) else None
    for idx, item in enumerate(objects or []):
        if not isinstance(item, dict):
            continue
        oid = str(item.get("track_id") or item.get("id") or f"o{idx + 1}")
        length = item.get("length_m") if _num(item.get("length_m")) else item.get("length")
        width = item.get("width_m") if _num(item.get("width_m")) else item.get("width")
        if not (_num(length) and _num(width)):
            continue
        bottom = item.get("bottom_z_m") if _num(item.get("bottom_z_m")) else item.get("bottom_z")
        top = item.get("top_z_m") if _num(item.get("top_z_m")) else item.get("top_z")
        b_val = float(bottom) if _num(bottom) else sup_val
        t_val = float(top) if _num(top) else None
        meta = derive_object_spatial_meta(
            length_m=float(length),
            width_m=float(width),
            top_z_m=t_val,
            bottom_z_m=b_val,
            surface_points_m=item.get("surface_points_m"),
            depth_std_m=item.get("depth_std_m"),
            residual_std_m=item.get("residual_std_m"),
        )
        if meta.get("uncertainty_m") is not None:
            out[oid] = meta
        raw = item.get("raw")
        if isinstance(raw, dict) and "floating_rescaled_to_table" in (item.get("flags") or []):
            xy = raw.get("centre_xy")
            z = [raw.get("low_z_m"), raw.get("top_z_m")]
            if isinstance(xy, (list, tuple)) and len(xy) == 2 and all(_num(v) for v in (*xy, *z)):
                out.setdefault(oid, {})["before_table_rescale"] = {
                    "centre_xy_m": [round(float(v), 3) for v in xy],
                    "visible_z_range_m": [round(float(v), 3) for v in z],
                }
    return out


class LocalSurfaceKDTree:
    """绑定于物体自身局部正交坐标系的三维表面点集 `cKDTree` 加速器。

    属性:
        local_points_m: 物体局部系下的表面采样点数组，形状 `(M, 3)`，单位 `m`。
        tree: 基于 `local_points_m` 一次性构建的 `scipy.spatial.cKDTree`（刚体 `SE(3)` 变换零重建）。
        build_count: 树构建次数计数器（恒为 `1`，用于审计与单测验证零重建契约）。
        local_radius_m: 局部点集距局部原点的最大欧氏半径，单位 `m`。

    示例:
        `ktree = LocalSurfaceKDTree(pts_world, obb0.center, obb0.axes)`
        `d_m = ktree.signed_distance_to_obb(finger_obb, cur_obj_obb, voxel_radius_m=0.006)`
    """

    __slots__ = ("local_points_m", "tree", "build_count", "local_radius_m", "_local_aabb_c", "_local_aabb_h")

    def __init__(self, world_points_m: np.ndarray, origin_m: np.ndarray, axes: np.ndarray):
        pts_w = np.asarray(world_points_m, dtype=np.float64).reshape(-1, 3)
        c0 = np.asarray(origin_m, dtype=np.float64).reshape(3)
        ax0 = np.asarray(axes, dtype=np.float64).reshape(3, 3)
        self.local_points_m: np.ndarray = np.ascontiguousarray((pts_w - c0[None, :]) @ ax0, dtype=np.float64)
        self.tree: cKDTree = cKDTree(self.local_points_m, balanced_tree=True, compact_nodes=True)
        self.build_count: int = 1
        if len(self.local_points_m) > 0:
            self.local_radius_m: float = float(np.max(np.linalg.norm(self.local_points_m, axis=1)))
            lo = np.min(self.local_points_m, axis=0)
            hi = np.max(self.local_points_m, axis=0)
            self._local_aabb_c: np.ndarray = 0.5 * (lo + hi)
            self._local_aabb_h: np.ndarray = 0.5 * (hi - lo)
        else:
            self.local_radius_m = 0.0
            self._local_aabb_c = np.zeros(3, dtype=np.float64)
            self._local_aabb_h = np.zeros(3, dtype=np.float64)

    @property
    def n_points(self) -> int:
        """返回局部树中包含的表面采样点数。"""
        return int(len(self.local_points_m))

    def world_points(self, current_obj_obb: OBB3D) -> np.ndarray:
        """将局部系表面点实时刚体变换回 `current_obj_obb` 对应的世界坐标 `(M, 3)`（单位 `m`，零建树开销）。"""
        return self.local_points_m @ current_obj_obb.axes.T + current_obj_obb.center[None, :]

    def signed_distance_to_obb(
        self,
        query_obb: OBB3D,
        current_obj_obb: OBB3D,
        voxel_radius_m: float = DEFAULT_SURFACE_VOXEL_RADIUS_M,
    ) -> float:
        """精确计算查询盒 `query_obb` 到当前位姿 `current_obj_obb` 下物体表面点集的有符号距离（单位 `m`）。

        参数:
            query_obb: 待测运动部件（手指、手掌或被夹持物）的 `OBB3D`（单位 `m`）。
            current_obj_obb: 当前时刻环境物体的 `OBB3D`（单位 `m`）。
            voxel_radius_m: 表面点体素球半径，单位 `m`，默认 `0.006`。

        返回:
            最小有符号欧氏距离（单位 `m`）：`> 0` 为无碰撞净空，`< 0` 为穿透深度。
        """
        m_pts = len(self.local_points_m)
        if m_pts == 0:
            return math.inf

        c_local = (query_obb.center - current_obj_obb.center) @ current_obj_obb.axes
        rel_axes = current_obj_obb.axes.T @ query_obb.axes
        half = query_obb.half_extents

        _, idx0 = self.tree.query(c_local)
        p0 = self.local_points_m[int(idx0)]
        q0 = np.abs((p0 - c_local) @ rel_axes) - half
        s0 = float(np.linalg.norm(np.maximum(q0, 0.0)) + min(float(np.max(q0)), 0.0))

        r_box = float(np.linalg.norm(half))
        cand_idx = self.tree.query_ball_point(c_local, max(s0, 0.0) + r_box + 1e-9)
        if len(cand_idx) <= 1:
            return s0 - float(voxel_radius_m)

        pts_sub = self.local_points_m[cand_idx]
        rel = (pts_sub - c_local[None, :]) @ rel_axes
        q = np.abs(rel) - half[None, :]
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return float(np.min(outside + inside)) - float(voxel_radius_m)

    def signed_distances_to_obbs(
        self,
        centers_w: np.ndarray,
        axes_w: np.ndarray,
        halves: np.ndarray,
        current_obj_obb: OBB3D,
        voxel_radius_m: float = DEFAULT_SURFACE_VOXEL_RADIUS_M,
        *,
        skip_above_m: float | None = None,
    ) -> np.ndarray:
        """批量计算 `Q` 个查询盒 `(centers_w, axes_w, halves)` 到当前位姿 `current_obj_obb` 表面点集的有符号距离（单位 `m`）。

        参数:
            centers_w: 查询盒世界系中心坐标数组，形状 `(Q, 3)`，单位 `m`。
            axes_w: 查询盒世界系正交主轴矩阵数组，形状 `(Q, 3, 3)`。
            halves: 查询盒半轴长数组，形状 `(Q, 3)`，单位 `m`。
            current_obj_obb: 当前时刻环境物体的 `OBB3D`（单位 `m`）。
            voxel_radius_m: 表面点体素球半径，单位 `m`，默认 `0.006`。
            skip_above_m: 可选净空截断上界，单位 `m`；超过下界直接返回合法下界跳过精化。

        返回:
            形状 `(Q,)` 的有符号距离数组（单位 `m`）。
        """
        cw = np.asarray(centers_w, dtype=np.float64).reshape(-1, 3)
        q_cnt = cw.shape[0]
        if q_cnt == 0:
            return np.empty(0, dtype=np.float64)
        if len(self.local_points_m) == 0:
            return np.full(q_cnt, math.inf, dtype=np.float64)

        aw = np.asarray(axes_w, dtype=np.float64).reshape(q_cnt, 3, 3)
        hw = np.asarray(halves, dtype=np.float64).reshape(q_cnt, 3)
        vox = float(voxel_radius_m)

        obj_c = current_obj_obb.center
        obj_ax = current_obj_obb.axes
        centers_local = (cw - obj_c[None, :]) @ obj_ax
        r_boxes = np.sqrt(np.sum(hw * hw, axis=1))

        if skip_above_m is not None:
            cutoff_s = float(skip_above_m) + vox
            d_orig = np.sqrt(np.sum(centers_local * centers_local, axis=1))
            sphere_lb_s = d_orig - self.local_radius_m - r_boxes
            out_s = sphere_lb_s.copy()
            act = np.flatnonzero(sphere_lb_s <= cutoff_s)
            if act.size == 0:
                return out_s - vox
            rel_axes_act = np.einsum("ji,qjk->qik", obj_ax, aw[act])
            proj_r_local = np.einsum("qik,qk->qi", np.abs(rel_axes_act), hw[act])
            raw_gap = np.abs(centers_local[act] - self._local_aabb_c[None, :]) - self._local_aabb_h[None, :] - proj_r_local
            sep_mask = np.any(raw_gap > 0.0, axis=1)
            gap_pos = np.maximum(raw_gap, 0.0)
            aabb_lb_s = np.where(sep_mask, np.sqrt(np.sum(gap_pos * gap_pos, axis=1)), -np.min(hw[act], axis=1))
            out_s[act] = np.maximum(out_s[act], aabb_lb_s)
            keep_act = np.flatnonzero(aabb_lb_s <= cutoff_s)
            if keep_act.size == 0:
                return out_s - vox
            act = act[keep_act]
            rel_axes_act = rel_axes_act[keep_act]
        else:
            cutoff_s = math.inf
            out_s = np.empty(q_cnt, dtype=np.float64)
            act = np.arange(q_cnt, dtype=np.intp)
            rel_axes_act = np.einsum("ji,qjk->qik", obj_ax, aw)

        c_act = centers_local[act]
        r_act = r_boxes[act]
        h_act = hw[act]

        dists_0, idxs_0 = self.tree.query(c_act)
        dists_0 = np.asarray(dists_0, dtype=np.float64)
        p0 = self.local_points_m[np.asarray(idxs_0, dtype=np.intp)]
        rel_0 = np.einsum("qd,qdk->qk", p0 - c_act, rel_axes_act)
        q_0 = np.abs(rel_0) - h_act
        q_0_pos = np.maximum(q_0, 0.0)
        s_0 = np.sqrt(np.sum(q_0_pos * q_0_pos, axis=1)) + np.minimum(np.max(q_0, axis=1), 0.0)
        lb_0 = np.where(dists_0 > r_act, dists_0 - r_act, -np.min(h_act, axis=1))

        out_s[act] = np.where(lb_0 > cutoff_s, lb_0, s_0)
        need_local = np.flatnonzero((lb_0 <= cutoff_s) & (lb_0 < (s_0 - 1e-12)))
        if need_local.size > 0:
            r_search = np.maximum(s_0[need_local], 0.0) + r_act[need_local] + 1e-9
            cand_lists = self.tree.query_ball_point(c_act[need_local], r_search)
            multi_pos = [j for j, cl in enumerate(cand_lists) if len(cl) > 1]
            if multi_pos:
                counts = np.array([len(cand_lists[j]) for j in multi_pos], dtype=np.intp)
                loc_indices = need_local[multi_pos]
                box_rep = np.repeat(np.arange(len(multi_pos), dtype=np.intp), counts)
                pt_flat = np.concatenate([np.asarray(cand_lists[j], dtype=np.intp) for j in multi_pos])
                rel_q = np.einsum(
                    "pd,pdk->pk",
                    self.local_points_m[pt_flat] - c_act[loc_indices][box_rep],
                    rel_axes_act[loc_indices][box_rep],
                )
                qq = np.abs(rel_q) - h_act[loc_indices][box_rep]
                q_pos = np.maximum(qq, 0.0)
                dists_p = np.sqrt(np.sum(q_pos * q_pos, axis=1)) + np.minimum(np.max(qq, axis=1), 0.0)
                starts = np.empty(len(multi_pos), dtype=np.intp)
                starts[0] = 0
                if len(multi_pos) > 1:
                    np.cumsum(counts[:-1], out=starts[1:])
                out_s[act[loc_indices]] = np.minimum.reduceat(dists_p, starts)

        return out_s - vox

    def batch_stage_clearance(
        self,
        query_obbs: list[OBB3D],
        current_obj_obb: OBB3D,
        voxel_radius_m: float = DEFAULT_SURFACE_VOXEL_RADIUS_M,
    ) -> float:
        """对单阶段内多插值步的所有运动部件 `query_obbs` 一次性批量求解对物体的精确最小有符号距离（单位 `m`）。"""
        if not query_obbs or len(self.local_points_m) == 0:
            return math.inf

        centers_w = np.stack([b.center for b in query_obbs], axis=0)
        axes_w = np.stack([b.axes for b in query_obbs], axis=0)
        halves = np.stack([b.half_extents for b in query_obbs], axis=0)

        obj_c = current_obj_obb.center
        obj_ax = current_obj_obb.axes
        centers_local = (centers_w - obj_c[None, :]) @ obj_ax
        rel_axes_all = np.einsum("ji,qjk->qik", obj_ax, axes_w)
        r_boxes = np.linalg.norm(halves, axis=1)

        dists_0, idxs_0 = self.tree.query(centers_local)
        dists_0 = np.asarray(dists_0, dtype=np.float64)
        p0 = self.local_points_m[np.asarray(idxs_0, dtype=np.intp)]
        rel_0 = np.einsum("qd,qdk->qk", p0 - centers_local, rel_axes_all)
        q_0 = np.abs(rel_0) - halves
        s_0 = np.linalg.norm(np.maximum(q_0, 0.0), axis=1) + np.minimum(np.max(q_0, axis=1), 0.0)

        best_s = float(np.min(s_0))
        lb = np.where(dists_0 > r_boxes, dists_0 - r_boxes, -np.min(halves, axis=1))
        need_mask = lb < (best_s - 1e-12)
        if np.any(need_mask):
            cand_qs = np.flatnonzero(need_mask)
            order = cand_qs[np.argsort(lb[cand_qs])]
            for q_idx in order:
                q_i = int(q_idx)
                if float(lb[q_i]) >= best_s - 1e-12:
                    break
                r_search = max(best_s, 0.0) + float(r_boxes[q_i]) + 1e-9
                cand_pts_idx = self.tree.query_ball_point(centers_local[q_i], r_search)
                if len(cand_pts_idx) > 1:
                    pts_sub = self.local_points_m[cand_pts_idx]
                    rel_q = (pts_sub - centers_local[q_i][None, :]) @ rel_axes_all[q_i]
                    qq = np.abs(rel_q) - halves[q_i][None, :]
                    s_q = float(
                        np.min(np.linalg.norm(np.maximum(qq, 0.0), axis=1) + np.minimum(np.max(qq, axis=1), 0.0))
                    )
                    if s_q < best_s:
                        best_s = s_q

        return best_s - float(voxel_radius_m)
