"""VGGT 三维体素 26 邻域快速连通域滤波、凹腔切片几何与多视角残差门控融合模块。

全部算法采用纯向量化 NumPy + OpenCV 实现（兼容输入 PyTorch 张量），不引入任何任务启发式：
1. `voxel_connected_cluster`: 1cm 三维稀疏体素网格 26 邻域主连通分量提取（单次 < 2ms，替代串行 CPU DBSCAN）。
2. `compute_cavity_depth` / `compute_surface_slices`: 掩码内部未截断有效点的边缘高度 `p90` 减内腔底部 `p5` 凹腔深度，及最多 3 层 2D 凸包切片。
3. `align_and_gate_extra_view` / `voxel_downsample_multiview`: 多视角深度一致性尺度求解、严格残差门控（`<= 0.025m`）与位掩码点云融合。
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

# 26 邻域正半空间 13 个无向偏移向量（利用对称性一次构建全部无向邻接边）
_OFFSETS_13 = np.array(
    [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) > (0, 0, 0)
    ],
    dtype=np.int64,
)
_K3 = np.ones((3, 3), dtype=np.uint8)


def _as_numpy_f64(arr: Any) -> np.ndarray:
    if hasattr(arr, "detach"):
        arr = arr.detach()
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    return np.asarray(arr, dtype=np.float64)


def _union_find_components(n_nodes: int, u_edges: np.ndarray, v_edges: np.ndarray) -> np.ndarray:
    """向量化 Shiloach-Vishkin 连通分量求解，返回每个节点的根代表编号。"""
    parent = np.arange(n_nodes, dtype=np.int32)
    if len(u_edges) == 0:
        return parent
    u = u_edges.astype(np.int32, copy=False)
    v = v_edges.astype(np.int32, copy=False)
    for _ in range(24):
        pu = parent[u]
        pv = parent[v]
        diff = pu != pv
        if not diff.any():
            break
        hi = np.maximum(pu[diff], pv[diff])
        lo = np.minimum(pu[diff], pv[diff])
        # 将较大根挂载到较小根并执行一次路径折叠（指针跳跃）
        np.minimum.at(parent, hi, lo)
        parent = parent[parent]
    while True:
        nxt = parent[parent]
        if np.array_equal(nxt, parent):
            break
        parent = nxt
    return parent


def voxel_connected_cluster(
    points: Any,
    voxel_m: float = 0.01,
    min_samples: int = 5,
) -> tuple[np.ndarray, float, np.ndarray]:
    """基于三维稀疏体素网格 26 邻域连通分量提取主簇，语义对齐 `DBSCAN(eps=voxel_m, min_samples=min_samples)`。

    参数:
        points: `(N, 3)` 三维点坐标（单位 `m`，支持 NumPy 数组或 PyTorch 张量）。
        voxel_m: 体素边长（单位 `m`，默认 `0.01` 即 1cm）。
        min_samples: 核心邻域最少点数阈值（默认 `5`）。

    返回:
        `(main_points, cluster_frac, labels)`:
        - `main_points`: 若存在有效簇则返回最大簇点集 `(M, 3)`，否则返回原始输入点集；
        - `cluster_frac`: 最大有效簇占输入总点数的比例 `0.0..1.0`（无有效簇时为 `0.0`）；
        - `labels`: `(N,)` `int32` 簇标签（噪声为 `-1`，有效簇为 `0..K-1`）。
    """
    P = _as_numpy_f64(points).reshape(-1, 3)
    n_pts = len(P)
    if n_pts < min_samples:
        return P, 0.0, np.full(n_pts, -1, dtype=np.int32)

    valid = np.isfinite(P).all(axis=1)
    if not valid.all():
        sub_main, _, sub_lab = voxel_connected_cluster(P[valid], voxel_m=voxel_m, min_samples=min_samples)
        full_lab = np.full(n_pts, -1, dtype=np.int32)
        full_lab[valid] = sub_lab
        if (full_lab >= 0).any():
            big = int(np.bincount(full_lab[full_lab >= 0]).argmax())
            return P[full_lab == big], float((full_lab == big).mean()), full_lab
        return P, 0.0, full_lab

    q = np.floor(P / float(voxel_m)).astype(np.int64)
    q -= q.min(axis=0)
    # 各轴预留 +1 边界裕量，杜绝 +-1 偏移跨轴回绕伪邻接
    q += 1
    span = q.max(axis=0) + 2

    if float(span[0]) * float(span[1]) * float(span[2]) < 4.0e17:
        stride_x = int(span[1] * span[2])
        stride_y = int(span[2])
        keys = q[:, 0] * stride_x + q[:, 1] * stride_y + q[:, 2]
        uniq_keys, inv, vox_counts = np.unique(keys, return_inverse=True, return_counts=True)
        n_vox = len(uniq_keys)
        delta_keys = _OFFSETS_13[:, 0] * stride_x + _OFFSETS_13[:, 1] * stride_y + _OFFSETS_13[:, 2]
        cand = uniq_keys[:, None] + delta_keys[None, :]
        pos = np.searchsorted(uniq_keys, cand)
        hit = (pos < n_vox) & (uniq_keys[np.minimum(pos, n_vox - 1)] == cand)
        u_idx, k_idx = np.nonzero(hit)
        v_idx = pos[u_idx, k_idx]
    else:
        # 极端离群尺度退化分支：按三元组字典序精确二分查找
        dt_void = np.dtype([("x", "<i8"), ("y", "<i8"), ("z", "<i8")])
        q_struct = np.ascontiguousarray(q).view(dt_void).reshape(-1)
        uniq_struct, inv, vox_counts = np.unique(q_struct, return_inverse=True, return_counts=True)
        n_vox = len(uniq_struct)
        uniq_q = uniq_struct.view(np.int64).reshape(n_vox, 3)
        u_list, v_list = [], []
        for off in _OFFSETS_13:
            cand_q = np.ascontiguousarray(uniq_q + off).view(dt_void).reshape(-1)
            pos = np.searchsorted(uniq_struct, cand_q)
            hit = (pos < n_vox) & (uniq_struct[np.minimum(pos, n_vox - 1)] == cand_q)
            ui = np.flatnonzero(hit)
            if len(ui):
                u_list.append(ui)
                v_list.append(pos[ui])
        u_idx = np.concatenate(u_list) if u_list else np.empty(0, dtype=np.int64)
        v_idx = np.concatenate(v_list) if v_list else np.empty(0, dtype=np.int64)

    # 统计每个非空体素在 3x3x3（26 邻域 + 自身）内的总点数，判定核心体素
    neigh_pts = vox_counts.copy()
    if len(u_idx):
        np.add.at(neigh_pts, u_idx, vox_counts[v_idx])
        np.add.at(neigh_pts, v_idx, vox_counts[u_idx])

    core = neigh_pts >= int(min_samples)
    if not core.any():
        return P, 0.0, np.full(n_pts, -1, dtype=np.int32)

    if len(u_idx):
        both_core = core[u_idx] & core[v_idx]
        comp = _union_find_components(n_vox, u_idx[both_core], v_idx[both_core])
    else:
        comp = np.arange(n_vox, dtype=np.int32)

    vox_label = np.where(core, comp, -1).astype(np.int32)
    # 边界体素（自身非核心但紧邻核心体素）并入邻域密度最高的核心簇
    if len(u_idx) and (~core).any():
        dir_b, dir_c = [], []
        mask_uc = (~core[u_idx]) & core[v_idx]
        mask_vc = (~core[v_idx]) & core[u_idx]
        if mask_uc.any():
            dir_b.append(u_idx[mask_uc])
            dir_c.append(v_idx[mask_uc])
        if mask_vc.any():
            dir_b.append(v_idx[mask_vc])
            dir_c.append(u_idx[mask_vc])
        if dir_b:
            b_nodes = np.concatenate(dir_b)
            c_nodes = np.concatenate(dir_c)
            # 按边界节点升序、核心邻点密度升序排列，末尾覆盖即为最优核心簇
            order = np.lexsort((neigh_pts[c_nodes], b_nodes))
            vox_label[b_nodes[order]] = comp[c_nodes[order]]

    valid_vox = vox_label >= 0
    if not valid_vox.any():
        return P, 0.0, np.full(n_pts, -1, dtype=np.int32)

    uniq_roots, root_inv = np.unique(vox_label[valid_vox], return_inverse=True)
    comp_pts = np.bincount(root_inv, weights=vox_counts[valid_vox]).astype(np.int64)
    keep_comp = comp_pts >= int(min_samples)
    if not keep_comp.any():
        return P, 0.0, np.full(n_pts, -1, dtype=np.int32)

    remap = np.full(len(uniq_roots), -1, dtype=np.int32)
    remap[keep_comp] = np.arange(int(keep_comp.sum()), dtype=np.int32)
    final_vox_label = np.full(n_vox, -1, dtype=np.int32)
    final_vox_label[valid_vox] = remap[root_inv]

    point_labels = final_vox_label[inv]
    valid_counts = comp_pts[keep_comp]
    best_cluster = int(valid_counts.argmax())
    best_mask = point_labels == best_cluster
    cluster_frac = float(best_mask.mean())
    return P[best_mask], cluster_frac, point_labels


def compute_cavity_depth(
    X: np.ndarray,
    conf: np.ndarray,
    conf_p10: float,
    mask: np.ndarray,
    cam_origin: np.ndarray | None = None,
    depth_scale: float = 1.0,
    min_valid_pts: int = 15,
) -> float:
    """计算物体掩码内部未截断有效表面点的凹腔深度 `cavity_depth_m`（单位 `m`，`>= 0.0`）。

    先腐蚀轮廓剥离桌面剪影渗漏，再在掩码内部未按桌面高度截断的有效 VGGT 点上，
    用外圈边缘区域高度 `p90` 减去内腔区域高度 `p5`；有效点不足时返回 `0.0`。
    """
    m_bool = np.asarray(mask, dtype=bool)
    if int(m_bool.sum()) < min_valid_pts:
        return 0.0

    m_u8 = m_bool.astype(np.uint8)
    m_in = cv2.erode(m_u8, _K3, iterations=2).astype(bool)
    if int(m_in.sum()) < min_valid_pts:
        m_in = cv2.erode(m_u8, _K3, iterations=1).astype(bool)
    if int(m_in.sum()) < min_valid_pts:
        m_in = m_bool

    valid = m_in & np.isfinite(X).all(axis=-1) & (conf > float(conf_p10))
    if int(valid.sum()) < min_valid_pts:
        return 0.0

    z_raw = X[..., 2][valid].astype(np.float64)
    if cam_origin is not None and abs(float(depth_scale) - 1.0) > 1e-9:
        cz = float(cam_origin[2])
        z_vals = cz + float(depth_scale) * (z_raw - cz)
    else:
        z_vals = z_raw

    dt = cv2.distanceTransform(m_in.astype(np.uint8), cv2.DIST_L2, 3)
    dt_vals = dt[valid]
    dt_max = float(dt_vals.max()) if len(dt_vals) else 0.0
    if dt_max >= 2.0:
        rim_band = max(2.0, min(8.0, 0.35 * dt_max))
        inner_margin = max(1.5, min(4.0, 0.20 * dt_max))
        rim_sel = dt_vals <= rim_band
        inner_sel = dt_vals > inner_margin
        if int(rim_sel.sum()) >= 5 and int(inner_sel.sum()) >= 5:
            rim_p90 = float(np.percentile(z_vals[rim_sel], 90))
            floor_p5 = float(np.percentile(z_vals[inner_sel], 5))
            return round(max(0.0, rim_p90 - floor_p5), 4)

    rim_p90 = float(np.percentile(z_vals, 90))
    floor_p5 = float(np.percentile(z_vals, 5))
    return round(max(0.0, rim_p90 - floor_p5), 4)


def _simplify_convex_hull(xy_pts: np.ndarray, max_vertices: int = 12) -> list[list[float]]:
    """计算二维点集凸包并将顶点数简化至不超过 `max_vertices`（默认 12）。"""
    if len(xy_pts) < 3:
        return []
    hull = cv2.convexHull(xy_pts.astype(np.float32)).reshape(-1, 2)
    if len(hull) < 3:
        return []
    if len(hull) > max_vertices:
        perim = float(cv2.arcLength(hull.reshape(-1, 1, 2), True))
        if perim > 1e-6:
            lo_eps, hi_eps = 1e-4 * perim, 0.20 * perim
            best = hull
            for _ in range(10):
                mid = 0.5 * (lo_eps + hi_eps)
                approx = cv2.approxPolyDP(hull.reshape(-1, 1, 2), mid, True).reshape(-1, 2)
                if len(approx) > max_vertices:
                    lo_eps = mid
                elif len(approx) < 3:
                    hi_eps = mid
                else:
                    best = approx
                    hi_eps = mid
            hull = best
        if len(hull) > max_vertices:
            idx = np.linspace(0, len(hull) - 1, max_vertices, endpoint=False).round().astype(int)
            hull = hull[idx]
    return [[round(float(x), 4), round(float(y), 4)] for x, y in hull]


def compute_surface_slices(
    points: Any,
    max_slices: int = 3,
    max_vertices: int = 12,
) -> list[dict[str, Any]]:
    """按物体高度切分最多 `max_slices`（默认 3）层二维凸包多边形切片，每层最多 `max_vertices`（默认 12）个顶点。"""
    P = _as_numpy_f64(points).reshape(-1, 3)
    if len(P) < 4 or max_slices <= 0:
        return []
    z_pts = P[:, 2]
    z_lo, z_hi = float(np.percentile(z_pts, 2)), float(np.percentile(z_pts, 98))
    dz = z_hi - z_lo

    if dz < 0.012 or len(P) < 18 or max_slices == 1:
        n_bands = 1
    elif dz < 0.030 or len(P) < 36 or max_slices == 2:
        n_bands = min(2, max_slices)
    else:
        n_bands = min(3, max_slices)

    slices: list[dict[str, Any]] = []
    if n_bands == 1 or dz < 1e-5:
        hull_xy = _simplify_convex_hull(P[:, :2], max_vertices=max_vertices)
        if hull_xy:
            slices.append(dict(z_m=round(float(np.median(z_pts)), 4), hull_xy_m=hull_xy))
        return slices

    edges = np.linspace(z_lo, z_hi, n_bands + 1)
    for bi in range(n_bands):
        b0 = edges[bi] - (1e-4 if bi == 0 else 0.0)
        b1 = edges[bi + 1] + (1e-4 if bi == n_bands - 1 else 0.0)
        sel = (z_pts >= b0) & (z_pts <= b1)
        if int(sel.sum()) < 4:
            continue
        hull_xy = _simplify_convex_hull(P[sel, :2], max_vertices=max_vertices)
        if hull_xy:
            slices.append(dict(z_m=round(float(np.median(z_pts[sel])), 4), hull_xy_m=hull_xy))
    return slices


def build_instance_id_map(instance_masks: list[tuple[int, np.ndarray]] | None, H: int, W: int) -> np.ndarray:
    """将 `[(track_num, mask_bool), ...]` 渲染为 `(H, W)` `np.int16` 实例索引图（面积小者覆盖面积大者以保留容器内物体）。"""
    inst_map = np.zeros((H, W), dtype=np.int16)
    if not instance_masks:
        return inst_map
    ordered = sorted(
        [(int(tid), np.asarray(m, dtype=bool)) for tid, m in instance_masks if int(tid) > 0],
        key=lambda item: -int(item[1].sum()),
    )
    for tid, m in ordered:
        if m.shape == (H, W):
            inst_map[m] = np.int16(min(tid, 32767))
    return inst_map


def compute_conf_rank(conf: np.ndarray) -> np.ndarray:
    """计算图像内每个像素的深度置信度分位秩 `0.0..1.0`。"""
    flat = np.asarray(conf, dtype=np.float64).reshape(-1)
    n_pix = len(flat)
    rank = np.empty(n_pix, dtype=np.float64)
    rank[np.argsort(flat, kind="stable")] = np.arange(n_pix, dtype=np.float64) / max(n_pix - 1, 1)
    return rank


def align_and_gate_extra_view(
    *,
    main_X: np.ndarray,
    main_conf: np.ndarray,
    main_K: np.ndarray,
    main_T: np.ndarray,
    main_inst_map: np.ndarray,
    extra_depth: np.ndarray,
    extra_conf: np.ndarray,
    extra_K: np.ndarray,
    extra_T: np.ndarray,
    support_z: float,
    z_below: float = 0.05,
    z_above: float = 0.60,
    max_range_m: float = 0.40,
    max_residual_m: float = 0.025,
    min_conf_rank: float = 0.25,
) -> dict[str, Any]:
    """求解附加腕部视角的度量深度尺度并执行严格重叠区深度残差门控。

    返回字典包含：
    - `accepted`: `bool`（是否通过重叠数与残差门控并允许融入主点云）
    - `residual_m`: `float | None`（重叠区深度残差中位数，重叠不足时为 `None`）
    - `scale`: `float | None`
    - `n_overlap`: `int`
    - `reason`: `str`
    - 通过门控时额外包含 `xyz` `(M,3)`, `rank` `(M,)`, `uv` `(M,2)`, `obj_id` `(M,)`。
    """
    H_0, W_0 = main_X.shape[:2]
    H_i, W_i = extra_depth.shape[:2]
    min_overlap = min(64, max(12, int(0.005 * H_i * W_i)))

    K_i = np.asarray(extra_K, dtype=np.float64).reshape(3, 3)
    T_i = np.asarray(extra_T, dtype=np.float64).reshape(4, 4)
    R_i = T_i[:3, :3]
    C_i = T_i[:3, 3]

    # 筛选主视角高置信工作空间三维点并投影到附加视角相机坐标系
    X0_flat = main_X.reshape(-1, 3).astype(np.float64)
    c0_flat = main_conf.reshape(-1)
    c0_thr = float(np.percentile(c0_flat, 25))
    valid_0 = (
        np.isfinite(X0_flat).all(axis=1)
        & (c0_flat >= c0_thr)
        & (X0_flat[:, 2] >= support_z - z_below)
        & (X0_flat[:, 2] <= support_z + z_above)
    )
    P0 = X0_flat[valid_0]
    if len(P0) < min_overlap:
        return dict(accepted=False, residual_m=None, scale=None, n_overlap=0, reason="insufficient_main_points")

    # T_world_cam_i 逆变换: P_cam_i = R_i^T (P_world - C_i) = (P_world - C_i) @ R_i
    P_cam_i = (P0 - C_i) @ R_i
    z_proj = P_cam_i[:, 2]
    in_front = (z_proj > 0.03) & (z_proj <= max(float(max_range_m) * 2.5, 1.0))
    if int(in_front.sum()) < min_overlap:
        return dict(accepted=False, residual_m=None, scale=None, n_overlap=int(in_front.sum()), reason="insufficient_overlap")

    P_cam_i = P_cam_i[in_front]
    z_proj = z_proj[in_front]
    u_f = K_i[0, 0] * (P_cam_i[:, 0] / z_proj) + K_i[0, 2]
    v_f = K_i[1, 1] * (P_cam_i[:, 1] / z_proj) + K_i[1, 2]
    ui = np.floor(u_f).astype(np.int64)
    vi = np.floor(v_f).astype(np.int64)
    in_img = (ui >= 0) & (ui < W_i) & (vi >= 0) & (vi < H_i)
    if int(in_img.sum()) < min_overlap:
        return dict(accepted=False, residual_m=None, scale=None, n_overlap=int(in_img.sum()), reason="insufficient_overlap")

    ui = ui[in_img]
    vi = vi[in_img]
    z_proj = z_proj[in_img]
    pix_idx = vi * W_i + ui

    # Z-buffer 消隐：同一副视角像素仅保留距离副视角光心最近的前景主视角投影点
    order_z = np.lexsort((z_proj, pix_idx))
    pix_sorted = pix_idx[order_z]
    first_pix = np.ones(len(order_z), dtype=bool)
    first_pix[1:] = pix_sorted[1:] != pix_sorted[:-1]
    front_sel = order_z[first_pix]

    ov_pix = pix_idx[front_sel]
    z_ref = z_proj[front_sel]

    d_flat = np.asarray(extra_depth, dtype=np.float64).reshape(-1)
    ci_flat = np.asarray(extra_conf, dtype=np.float64).reshape(-1)
    ci_thr = float(np.percentile(ci_flat, 30))
    valid_ov = np.isfinite(d_flat[ov_pix]) & (d_flat[ov_pix] > 1e-6) & (ci_flat[ov_pix] >= ci_thr)
    ov_pix = ov_pix[valid_ov]
    z_ref = z_ref[valid_ov]
    d_raw = d_flat[ov_pix]

    n_overlap = int(len(z_ref))
    if n_overlap < min_overlap:
        return dict(accepted=False, residual_m=None, scale=None, n_overlap=n_overlap, reason="insufficient_overlap")

    # 优先用近场重叠点求解 L1 最优中位数度量尺度，抑制远景漂移干扰
    near_ov = z_ref <= max(float(max_range_m) * 1.5, 0.55)
    if int(near_ov.sum()) >= min_overlap:
        s_i = float(np.median(z_ref[near_ov] / d_raw[near_ov]))
        resid_vec = np.abs(s_i * d_raw[near_ov] - z_ref[near_ov])
    else:
        s_i = float(np.median(z_ref / d_raw))
        resid_vec = np.abs(s_i * d_raw - z_ref)

    if not (math.isfinite(s_i) and s_i > 0.0):
        return dict(accepted=False, residual_m=None, scale=None, n_overlap=n_overlap, reason="invalid_scale")

    residual_m = float(np.median(resid_vec))
    if residual_m > float(max_residual_m):
        return dict(
            accepted=False,
            residual_m=round(residual_m, 5),
            scale=round(s_i, 5),
            n_overlap=n_overlap,
            reason="residual_exceeded",
        )

    # 反投影副视角全图，严格仅保留光心距离 <= max_range_m 的高置信近场工作空间点
    uu, vv = np.meshgrid(np.arange(W_i, dtype=np.float64) + 0.5, np.arange(H_i, dtype=np.float64) + 0.5)
    depth_f64 = np.asarray(extra_depth, dtype=np.float64)
    Pc_i = np.stack(
        [
            (uu - K_i[0, 2]) / K_i[0, 0] * depth_f64,
            (vv - K_i[1, 2]) / K_i[1, 1] * depth_f64,
            depth_f64,
        ],
        axis=-1,
    )
    Xi_world = (C_i + s_i * (Pc_i @ R_i.T)).reshape(-1, 3)
    rank_i = compute_conf_rank(extra_conf)
    dist_cam = np.linalg.norm(Xi_world - C_i, axis=1)

    keep_i = (
        np.isfinite(Xi_world).all(axis=1)
        & (depth_f64.reshape(-1) > 1e-6)
        & (dist_cam <= float(max_range_m))
        & (rank_i >= float(min_conf_rank))
        & (Xi_world[:, 2] >= support_z - z_below)
        & (Xi_world[:, 2] <= support_z + z_above)
    )
    if not keep_i.any():
        return dict(
            accepted=False,
            residual_m=round(residual_m, 5),
            scale=round(s_i, 5),
            n_overlap=n_overlap,
            reason="no_near_field_points",
        )

    P_extra = Xi_world[keep_i]
    rank_extra = rank_i[keep_i]

    # 将副视角三维点投影回主视角 cam_head 以对齐像素 uv 并继承实例掩码编号 obj_id
    K_0 = np.asarray(main_K, dtype=np.float64).reshape(3, 3)
    T_0 = np.asarray(main_T, dtype=np.float64).reshape(4, 4)
    R_0 = T_0[:3, :3]
    C_0 = T_0[:3, 3]
    P_in_0 = (P_extra - C_0) @ R_0
    z0 = np.maximum(P_in_0[:, 2], 1e-6)
    u0 = np.floor(K_0[0, 0] * (P_in_0[:, 0] / z0) + K_0[0, 2]).astype(np.int64)
    v0 = np.floor(K_0[1, 1] * (P_in_0[:, 1] / z0) + K_0[1, 2]).astype(np.int64)
    vis0 = (P_in_0[:, 2] > 0.02) & (u0 >= 0) & (u0 < W_0) & (v0 >= 0) & (v0 < H_0)

    u0_clip = np.clip(u0, 0, max(W_0 - 1, 0))
    v0_clip = np.clip(v0, 0, max(H_0 - 1, 0))
    uv_extra = np.where(vis0[:, None], np.stack([u0_clip, v0_clip], axis=1), 0).astype(np.uint16)

    obj_id_extra = np.zeros(len(P_extra), dtype=np.int16)
    if vis0.any() and main_inst_map is not None and main_inst_map.any():
        vis_idx = np.flatnonzero(vis0)
        uu0, vv0 = u0_clip[vis_idx], v0_clip[vis_idx]
        cand_id = main_inst_map[vv0, uu0]
        # 校验副视角点与主视角对应像素的三维点距离不超过 3.5cm，防止腕部遮挡点误继承背景物体标签
        d3 = np.linalg.norm(P_extra[vis_idx] - main_X[vv0, uu0].astype(np.float64), axis=1)
        obj_id_extra[vis_idx] = np.where(d3 <= 0.035, cand_id, 0).astype(np.int16)

    return dict(
        accepted=True,
        residual_m=round(residual_m, 5),
        scale=round(s_i, 5),
        n_overlap=n_overlap,
        reason="ok",
        xyz=P_extra,
        rank=rank_extra,
        uv=uv_extra,
        obj_id=obj_id_extra,
    )


def voxel_downsample_multiview(
    *,
    xyz: np.ndarray,
    rank: np.ndarray,
    uv: np.ndarray,
    obj_id: np.ndarray,
    view_bit: np.ndarray,
    voxel_m: float,
) -> dict[str, np.ndarray]:
    """对单视角或多视角拼接点集执行体素下采样，聚合最高置信点、像素计数、持久物体编号与视角位掩码。"""
    n_total = len(xyz)
    if n_total == 0:
        return dict(
            xyz=np.zeros((0, 3), dtype=np.float32),
            conf=np.zeros((0,), dtype=np.uint8),
            uv=np.zeros((0, 2), dtype=np.uint16),
            count=np.zeros((0,), dtype=np.uint16),
            obj_id=np.zeros((0,), dtype=np.int16),
            view_mask=np.zeros((0,), dtype=np.uint8),
        )

    q = np.floor(xyz / float(voxel_m)).astype(np.int64)
    q -= q.min(axis=0)
    span = q.max(axis=0) + 1
    key = (q[:, 0] * span[1] + q[:, 1]) * span[2] + q[:, 2]

    order = np.lexsort((-rank, key))
    key_sorted = key[order]
    first = np.ones(n_total, dtype=bool)
    first[1:] = key_sorted[1:] != key_sorted[:-1]
    starts = np.flatnonzero(first)
    counts = np.diff(np.append(starts, n_total))
    winners = order[starts]

    # 位掩码按体素内全部观测来源做按位或（1=主视角, 2/4/...=附加腕部视角）
    merged_view_mask = np.bitwise_or.reduceat(view_bit[order].astype(np.uint8), starts)
    # 持久物体编号优先取最高置信点编号；若该点来自未着色腕部视角而同体素主视角已有非零 obj_id，则继承该非零编号
    win_obj = obj_id[winners].astype(np.int16)
    max_obj = np.maximum.reduceat(obj_id[order].astype(np.int16), starts)
    merged_obj_id = np.where(win_obj > 0, win_obj, max_obj).astype(np.int16)

    return dict(
        xyz=xyz[winners].astype(np.float32),
        conf=np.rint(rank[winners] * 255.0).astype(np.uint8),
        uv=uv[winners].astype(np.uint16),
        count=np.minimum(counts, 65535).astype(np.uint16),
        obj_id=merged_obj_id,
        view_mask=merged_view_mask.astype(np.uint8),
    )
