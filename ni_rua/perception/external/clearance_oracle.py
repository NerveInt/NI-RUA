"""无状态向量化夹爪净空、咬合、轨迹碰撞与凹腔几何评估器 (`clearance_oracle.py`)。

将原先分散在 `virtual_field.py`、`pointcloud_field.py` 与 `vectorized_grasp.py` 中的 4 套抓取评估器与 2 套轨迹推演器
统一收敛为单一无状态向量化物理评估核：
1. `ParallelJawGeometry` / `gripper_obbs`：平行双指夹爪物理包络契约与左指/右指/手掌/闭合扫掠 `OBB3D` 构造；
2. `VirtualObject` / `_parse_virtual_object`：场景物体三维刚体位姿、倾斜标架与夹持绑定状态；
3. `evaluate_grasp_clearance_batch` / `evaluate_grasp_clearance`：统一支持实心 `OBB3D` 与精细 `LocalSurfaceKDTree` 点云/切片几何的向量化抓取净空、咬合与偏心评估（绝不合成 `optimal_grasp_pose`）；
4. `simulate_stage_clearances`：统一的多阶段轨迹净空、持物/释放状态与凹腔落位推演器（绝不生成 `recommended_safe_z_m` 或 `suggested_fixes`）；
5. `inspect_cavity_geometry`：三维点云与实测凹腔深度的开口剖面测量。
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import io
import math
from pathlib import Path
from typing import Any

import numpy as np

from .geometry_ops import (
    DEFAULT_SURFACE_VOXEL_RADIUS_M,
    MIN_CAVITY_DEPTH_M,
    MIN_OBJECT_SURFACE_POINTS,
    LocalSurfaceKDTree,
    OBB3D,
    _clamp_single_point_upper_bound,
    _num,
    _r,
    batch_obbs_to_slabs_min_distance,
    clip_target_obb_in_gripper_frames,
    compute_tilted_obb_frame,
    obb_sat_max_gap_batch,
    obb_signed_distance,
    obb_signed_distance_batch,
    obb_to_points_signed_distance,
    obb_to_points_signed_distance_batch,
)
from .pose_math import (
    CONTACT_DEPTH_PRESETS_M,
    FINGER_LENGTH_M,
    FINGER_THICKNESS_M,
    FINGER_WIDTH_M,
    FINGERTIP_BELOW_PAD_M,
    GRASP_POINT_OFFSET_M,
    MAX_OPENING_M,
    PAD_DEPTH_M,
    PALM_LENGTH_M,
    PALM_THICKNESS_M,
    PALM_WIDTH_M,
    angles_deg_to_matrices_batch,
    flange_pose_to_grasp_pose,
    gripper_frame,
    wrap90,
    wrap180,
)

# 细长物体长宽比阈值
ELONGATED_RATIO: float = 1.5
# 常规刚体最小有效竖直咬合高度上限 (m)；对超薄物体按实测高度自适应缩放以避免误杀
MIN_VERTICAL_BITE_M: float = 0.0025

DEFAULT_ARM_POSES: dict[str, dict[str, list[float]]] = {
    "left": {"position_m": [-0.25, -0.25, 0.92], "angles_deg": [0.0, 0.0, 0.0]},
    "right": {"position_m": [0.25, -0.25, 0.92], "angles_deg": [0.0, 0.0, 0.0]},
}


@dataclass(frozen=True)
class ParallelJawGeometry:
    """末端平行双指夹爪三维包络几何契约（所有线性尺寸单位为 `m`）。"""

    max_opening_m: float = MAX_OPENING_M
    grasp_point_offset_m: float = GRASP_POINT_OFFSET_M
    fingertip_below_pad_m: float = FINGERTIP_BELOW_PAD_M
    pad_depth_m: float = PAD_DEPTH_M
    finger_width_m: float = FINGER_WIDTH_M
    finger_thickness_m: float = FINGER_THICKNESS_M
    finger_length_m: float = FINGER_LENGTH_M
    palm_length_m: float = PALM_LENGTH_M
    palm_width_m: float = PALM_WIDTH_M
    palm_thickness_m: float = PALM_THICKNESS_M
    arms: tuple[str, ...] = ("left", "right")
    contact_depths_m: tuple[tuple[str, float], ...] = tuple(CONTACT_DEPTH_PRESETS_M.items())

    def contact_depth(self, contact_region: str = "pad_center", contact_depth_m: float | None = None) -> float:
        """返回指定接触截面或显式偏移的轴向接触深度，单位 `m`。"""
        if contact_depth_m is not None:
            return float(contact_depth_m)
        return float(dict(self.contact_depths_m).get(contact_region, 0.0))

    @classmethod
    def from_profile(cls, profile: Any = None) -> "ParallelJawGeometry | None":
        """从 `EmbodimentProfile` 提取平行夹爪几何参数；不采用 `grasp_angles` / `tcp_offset` 的异构夹爪返回 `None`。"""
        if profile is None:
            return DEFAULT_JAW_GEOMETRY
        pose_inputs = getattr(profile, "pose_inputs", None) or ()
        grip = getattr(profile, "gripper", None)
        if "grasp_angles" not in pose_inputs or grip is None or getattr(grip, "grasp_reference", None) != "tcp_offset":
            return None
        ft_below = float(getattr(grip, "fingertip_beyond_reference_m", FINGERTIP_BELOW_PAD_M))
        sections = getattr(grip, "contact_sections", None)
        depth_items = (
            tuple((str(k), float(v)) for k, v in sections.items())
            if isinstance(sections, dict) and sections
            else tuple(CONTACT_DEPTH_PRESETS_M.items())
        )
        raw_offset = getattr(grip, "grasp_reference_offset_m", None)
        return cls(
            max_opening_m=float(getattr(grip, "full_open_aperture_m", MAX_OPENING_M)),
            grasp_point_offset_m=GRASP_POINT_OFFSET_M if raw_offset is None else float(raw_offset),
            fingertip_below_pad_m=ft_below,
            pad_depth_m=max(0.002, 2.0 * ft_below) if ft_below > 0.0 else PAD_DEPTH_M,
            arms=tuple(getattr(profile, "arms", None) or ("left", "right")),
            contact_depths_m=depth_items,
        )


DEFAULT_JAW_GEOMETRY = ParallelJawGeometry()


def gripper_obbs(
    position_m: Any,
    angles_deg: Any,
    aperture_m: float = MAX_OPENING_M,
    contact_region: str = "pad_center",
    contact_depth_m: float | None = None,
    *,
    jaw_geometry: ParallelJawGeometry | None = None,
) -> dict[str, Any]:
    """构建指定位姿与开度下夹爪各部件（左指、右指、手掌、指间闭合扫掠区）的 3D OBB。

    参数:
        position_m: 目标位置 `[x, y, z]`，单位 `m` (`env_origin`)。
        angles_deg: 姿态角 `[pitch_deg, roll_deg, yaw_deg]`，单位 `deg`。
        aperture_m: 两指内侧间距，单位 `m`，默认 `0.088`。
        contact_region: 接触截面 (`'pad_center' | 'fingertip' | 'finger_middle'`)。
        contact_depth_m: 可选显式接触深度偏移，单位 `m`。
        jaw_geometry: 可选 `ParallelJawGeometry` 配置。

    返回:
        包含 `left_finger`, `right_finger`, `palm`, `swept_pad` (`OBB3D`) 与 `frame` 字典的结果。
    """
    jg = jaw_geometry if jaw_geometry is not None else DEFAULT_JAW_GEOMETRY
    eff_depth = jg.contact_depth(contact_region, contact_depth_m)
    frame = gripper_frame(
        position_m,
        angles_deg,
        contact_region=contact_region,
        contact_depth_m=eff_depth,
        grasp_point_offset_m=jg.grasp_point_offset_m,
        fingertip_below_pad_m=jg.fingertip_below_pad_m,
    )
    pad_center = frame["pad_center_m"]
    u_app = frame["approach_dir"]
    u_close = frame["closing_dir"]
    axes = frame["axes"]
    ap = float(np.clip(aperture_m, 0.0, jg.max_opening_m))

    finger_axial_center = pad_center + (jg.fingertip_below_pad_m - 0.5 * jg.finger_length_m) * u_app
    finger_half = np.array([0.5 * jg.finger_length_m, 0.5 * jg.finger_thickness_m, 0.5 * jg.finger_width_m])
    lat_offset = 0.5 * ap + 0.5 * jg.finger_thickness_m

    left_finger = OBB3D(finger_axial_center - lat_offset * u_close, axes, finger_half, label="left_finger")
    right_finger = OBB3D(finger_axial_center + lat_offset * u_close, axes, finger_half, label="right_finger")

    palm_center = pad_center - (jg.finger_length_m - jg.fingertip_below_pad_m + 0.5 * jg.palm_length_m) * u_app
    palm_half = np.array([0.5 * jg.palm_length_m, 0.5 * jg.palm_width_m, 0.5 * jg.palm_thickness_m])
    palm = OBB3D(palm_center, axes, palm_half, label="palm")

    swept_half = np.array([0.5 * jg.pad_depth_m, max(0.5 * ap, 5e-4), 0.5 * jg.finger_width_m])
    swept_pad = OBB3D(pad_center, axes, swept_half, label="swept_pad")

    return dict(
        left_finger=left_finger,
        right_finger=right_finger,
        palm=palm,
        swept_pad=swept_pad,
        frame=frame,
    )


class VirtualObject:
    """场景物体的三维体积与动态运动学状态表征（仅承载实测几何事实，不含形状分类标签）。"""

    def __init__(
        self,
        oid: str,
        centre_m: Any,
        top_z: float,
        bottom_z: float,
        length: float,
        width: float,
        yaw_deg: float,
        tilt_deg: float | None = None,
        support: str | None = "table",
        quality: str = "ok",
        held_by_arm: str | None = None,
        occluded: bool = False,
        intrinsic_height_m: float | None = None,
        *,
        occluded_steps: int = 0,
        uncertainty_m: float | None = None,
        axis_3d: Any = None,
        tilt_sign: float = 1.0,
        has_measured_bottom_z: bool = True,
    ):
        self.id = str(oid)
        self.centre_m = [float(v) for v in np.asarray(centre_m, dtype=float).reshape(3)]
        self.top_z = float(top_z)
        self.bottom_z = float(bottom_z)
        self.length = max(float(length), 0.004)
        self.width = max(float(width), 0.004)
        self.intrinsic_height_m = max(
            float(intrinsic_height_m) if _num(intrinsic_height_m) else (self.top_z - self.bottom_z),
            0.001,
        )
        self.yaw_deg = float(yaw_deg)
        self.tilt_deg = float(tilt_deg) if _num(tilt_deg) else None
        self.support = support
        self.quality = str(quality or "ok")
        self.held_by_arm = held_by_arm
        self.occluded = bool(occluded)
        self.occluded_steps = int(occluded_steps) if int(occluded_steps) > 0 else (1 if self.occluded else 0)
        self.uncertainty_m = round(float(uncertainty_m), 4) if _num(uncertainty_m) else None
        self._has_measured_bottom_z = bool(has_measured_bottom_z)
        self.virtual_moved = False
        self.rel_offset_in_gripper: np.ndarray | None = None
        self._rel_centre_in_gripper: np.ndarray | None = None
        self.rel_axes_in_gripper: np.ndarray | None = None
        self._world_axes_override: np.ndarray | None = None
        self._tilted_axes, self._tilted_half_extents = compute_tilted_obb_frame(
            self.length,
            self.width,
            self.intrinsic_height_m,
            self.yaw_deg,
            self.tilt_deg,
            axis_3d=axis_3d,
            tilt_sign=tilt_sign,
        )

    @property
    def height_m(self) -> float:
        """返回世界系下实测竖直跨度，单位 `m`。"""
        return max(self.top_z - self.bottom_z, 0.001)

    @property
    def local_half_extents(self) -> np.ndarray:
        """返回物体局部坐标系下的刚体固有半轴长，单位 `m`。"""
        if self._tilted_half_extents is not None:
            return self._tilted_half_extents.copy()
        return np.array([0.5 * self.length, 0.5 * self.width, 0.5 * self.intrinsic_height_m], dtype=np.float64)

    @property
    def is_elongated(self) -> bool:
        """判断物体在水平面是否呈细长形态（受单爪最大开度约束）。"""
        if self.width <= 0 or self.width > MAX_OPENING_M or (self.length / self.width) < ELONGATED_RATIO:
            return False
        if self._tilted_half_extents is None and self.intrinsic_height_m >= 0.85 * self.length:
            return False
        if _num(self.tilt_deg) and float(self.tilt_deg) < 35.0:
            return False
        return True

    def world_axes(self) -> np.ndarray:
        """返回物体当前在世界坐标系下的右手正交三轴矩阵 `(3, 3)`。"""
        if self._world_axes_override is not None:
            return self._world_axes_override
        if self._tilted_axes is not None:
            return self._tilted_axes
        yaw_rad = math.radians(self.yaw_deg)
        u_long = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0], dtype=np.float64)
        u_short = np.array([-math.sin(yaw_rad), math.cos(yaw_rad), 0.0], dtype=np.float64)
        u_vert = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return np.column_stack([u_long, u_short, u_vert])

    def refine_tilt_from_points(self, points_m: np.ndarray) -> None:
        """当存在三维表面点集且物体具有倾斜标架时，根据点云主协方差精确锁定倾斜方位正负号。"""
        if self._tilted_axes is None or self._world_axes_override is not None:
            return
        pts = np.asarray(points_m, dtype=np.float64).reshape(-1, 3)
        if len(pts) < 6 or not _num(self.tilt_deg):
            return
        yaw_rad = math.radians(self.yaw_deg)
        u_h = np.array([math.cos(yaw_rad), math.sin(yaw_rad)], dtype=np.float64)
        centered = pts - np.mean(pts, axis=0, keepdims=True)
        cov_hz = float(np.dot(centered[:, :2] @ u_h, centered[:, 2]))
        if abs(cov_hz) > 1e-7:
            sgn = 1.0 if cov_hz >= 0.0 else -1.0
            axes_new, half_new = compute_tilted_obb_frame(
                self.length,
                self.width,
                self.intrinsic_height_m,
                self.yaw_deg,
                self.tilt_deg,
                tilt_sign=sgn,
            )
            if axes_new is not None and half_new is not None:
                self._tilted_axes = axes_new
                self._tilted_half_extents = half_new
                self._world_axes_override = axes_new

    def obb(self) -> OBB3D:
        """返回物体当前在世界坐标系下的 3D OBB（始终使用刚体固有半轴长 `local_half_extents`）。"""
        axes = self.world_axes()
        z_mid = 0.5 * (self.top_z + self.bottom_z)
        center = np.array([self.centre_m[0], self.centre_m[1], z_mid], dtype=np.float64)
        return OBB3D(center, axes, self.local_half_extents, label=self.id)

    def footprint_corners_xy(self) -> np.ndarray:
        """返回物体水平截面四个角点的世界坐标 `(4, 3)`，单位 `m`。"""
        axes = self.world_axes()
        u_long, u_short = axes[:, 0], axes[:, 1]
        c = np.asarray(self.centre_m, dtype=np.float64)
        hl, hw = 0.5 * self.length, 0.5 * self.width
        return np.array(
            [
                c - hl * u_long - hw * u_short,
                c + hl * u_long - hw * u_short,
                c + hl * u_long + hw * u_short,
                c - hl * u_long + hw * u_short,
            ],
            dtype=np.float64,
        )

    def attach_to_gripper(self, arm: str, g_frame: dict) -> None:
        """将本物体刚性绑定到指定机械臂的夹爪标架上。"""
        self.held_by_arm = arm
        pad_c = g_frame["pad_center_m"]
        g_axes = g_frame["axes"]
        obb_c = self.obb().center
        self.rel_offset_in_gripper = (obb_c - pad_c) @ g_axes
        self._rel_centre_in_gripper = (np.asarray(self.centre_m, dtype=np.float64) - pad_c) @ g_axes
        self.rel_axes_in_gripper = g_axes.T @ self.world_axes()

    def update_from_gripper(self, g_frame: dict) -> None:
        """当持物机械臂移动时，根据夹爪最新标架实时更新物体的 3D 位姿与世界系 AABB 高度边界（严格幂等）。"""
        if self.rel_offset_in_gripper is None or self.rel_axes_in_gripper is None:
            return
        pad_c = g_frame["pad_center_m"]
        g_axes = g_frame["axes"]
        new_axes = g_axes @ self.rel_axes_in_gripper
        self._world_axes_override = new_axes
        new_centre = pad_c + self._rel_centre_in_gripper @ g_axes.T
        new_obb_c = pad_c + self.rel_offset_in_gripper @ g_axes.T
        z_extent = float(np.abs(new_axes[2, :]) @ self.local_half_extents)
        self.centre_m = [float(v) for v in new_centre]
        self.top_z = float(new_obb_c[2] + z_extent)
        self.bottom_z = float(new_obb_c[2] - z_extent)
        u_long = new_axes[:, 0]
        if math.hypot(float(u_long[0]), float(u_long[1])) > 1e-4:
            self.yaw_deg = float(wrap90(math.degrees(math.atan2(float(u_long[1]), float(u_long[0])))))
        self.tilt_deg = float(math.degrees(math.acos(np.clip(abs(float(u_long[2])), 0.0, 1.0))))
        self.virtual_moved = True

    def detach_from_gripper(self, support_z: float, support_id: str = "table") -> None:
        """开爪释放物体时，保留释放时的 3D 旋转姿态并将物体底部沉降到下方支撑面上（单位 `m`）。"""
        axes = self.world_axes()
        z_extent = float(np.abs(axes[2, :]) @ self.local_half_extents)
        sup_z = float(support_z)
        obb_z = sup_z + z_extent
        old_obb_z = 0.5 * (self.top_z + self.bottom_z)
        dz = obb_z - old_obb_z
        self.centre_m[2] = float(self.centre_m[2] + dz)
        self.bottom_z = sup_z
        self.top_z = float(sup_z + 2.0 * z_extent)
        self.support = support_id
        self.held_by_arm = None
        self.rel_offset_in_gripper = None
        self._rel_centre_in_gripper = None
        self.rel_axes_in_gripper = None
        self.virtual_moved = True

    def to_dict(self) -> dict:
        """导出可序列化的物体几何与持握状态字典（单位 `m` / `deg`）。"""
        return dict(
            id=self.id,
            centre_m=_r(self.centre_m, 4),
            top_z=_r(self.top_z, 4),
            bottom_z=_r(self.bottom_z, 4),
            length=_r(self.length, 4),
            width=_r(self.width, 4),
            yaw_deg=_r(self.yaw_deg, 1),
            tilt_deg=_r(self.tilt_deg, 1) if self.tilt_deg is not None else None,
            support=self.support,
            quality=self.quality,
            held_by_arm=self.held_by_arm,
            virtual_moved=self.virtual_moved,
            occluded=self.occluded,
            occluded_steps=int(self.occluded_steps),
            uncertainty_m=_r(self.uncertainty_m, 4) if self.uncertainty_m is not None else None,
            intrinsic_height_m=_r(self.intrinsic_height_m, 4),
        )


def _min_vertical_bite_m(target: VirtualObject) -> float:
    """根据目标实测高度跨度客观确定最小有效竖直咬合阈值（单位 `m`）。"""
    span_z = max(0.0, float(target.top_z) - float(target.bottom_z))
    if span_z > 0.0:
        return min(MIN_VERTICAL_BITE_M, 0.5 * span_z)
    return MIN_VERTICAL_BITE_M


def _effective_support_z(target: VirtualObject, table_z: float | None) -> float | None:
    """返回目标下方实测支撑高度（单位 `m`）；当既无实测 `table_z` 也无已知底部支撑物时返回 `None`。"""
    vals: list[float] = []
    sup_label = str(target.support or "").strip()
    has_known_support = bool(sup_label) and sup_label != "unknown" and not sup_label.startswith("held_")
    if has_known_support and getattr(target, "_has_measured_bottom_z", True) and _num(target.bottom_z):
        vals.append(float(target.bottom_z))
    if _num(table_z):
        vals.append(float(table_z))
    return max(vals) if vals else None


def _extract_measured_support_z(source: dict | None) -> float | None:
    """从输入字典提取实测支撑平面高程（兼容 `table_z`、`support_z_m`、`support_plane`），单位 `m`；无实测时返回 `None`。"""
    if not isinstance(source, dict):
        return None
    if _num(source.get("table_z")):
        return float(source["table_z"])
    if _num(source.get("support_z_m")):
        return float(source["support_z_m"])
    sp = source.get("support_plane")
    if _num(sp):
        return float(sp)
    if isinstance(sp, dict):
        for key in ("z_m", "table_z", "support_z_m", "height_m"):
            if _num(sp.get(key)):
                return float(sp[key])
        origin = sp.get("origin_m") or sp.get("point_m")
        if isinstance(origin, (list, tuple, np.ndarray)) and len(origin) >= 3 and _num(origin[2]):
            return float(origin[2])
    return None


def _parse_virtual_object(item: Any, index: int = 0, table_z: float | None = None, columns: Any = None) -> VirtualObject | None:
    """将 `object_table` 的行列表或字典解析为 `VirtualObject`（零硬编码台面回退、零形状标签启发式）。"""
    if isinstance(item, VirtualObject):
        return deepcopy(item)
    if isinstance(item, (list, tuple)):
        cols = list(columns or ("id", "centre_m", "top_z", "bottom_z", "length", "width", "yaw_deg", "tilt_deg", "support", "quality"))
        if len(item) < 2:
            return None
        row = dict(zip(cols, item))
    elif isinstance(item, dict):
        row = item
    else:
        return None

    oid = str(row.get("id") or row.get("track_id") or f"o{index + 1}")
    support = row.get("support", "table" if _num(table_z) else None)
    if isinstance(support, int):
        support = f"o{support + 1}"
    quality = str(row.get("quality") or "ok")
    held_by = str(row["held_by_arm"]) if isinstance(row.get("held_by_arm"), str) and row["held_by_arm"] else None
    if held_by is None and isinstance(support, str) and support.startswith("held_"):
        held_by = support.split("held_", 1)[-1]

    centre = row.get("centre_m")
    if not isinstance(centre, (list, tuple, np.ndarray)) or len(centre) != 3 or not all(_num(v) for v in centre):
        return None
    length = row.get("length") if _num(row.get("length")) else row.get("length_m")
    width = row.get("width") if _num(row.get("width")) else row.get("width_m")
    yaw = row.get("yaw_deg")
    if not (_num(length) and _num(width) and _num(yaw)):
        return None
    cz = float(centre[2])
    bottom = row.get("bottom_z") if _num(row.get("bottom_z")) else row.get("bottom_z_m")
    top = row.get("top_z") if _num(row.get("top_z")) else row.get("top_z_m")
    has_measured_bottom = bool(_num(bottom) or _num(table_z))
    if _num(bottom):
        bottom_z = float(bottom)
    elif _num(table_z):
        bottom_z = float(table_z)
    elif _num(top):
        bottom_z = min(float(top), cz - max(float(top) - cz, 0.002))
    else:
        bottom_z = cz - 0.006
    top_z = float(top) if _num(top) else max(cz + (cz - bottom_z), bottom_z + 0.012)
    if top_z < bottom_z:
        top_z, bottom_z = bottom_z, top_z
    tilt = row.get("tilt_deg") if _num(row.get("tilt_deg")) else row.get("axis_tilt_deg")
    occluded = bool(row.get("occluded", False))
    occ_steps = int(row.get("occluded_steps") or (1 if occluded else 0))
    uncertainty_m = float(row["uncertainty_m"]) if _num(row.get("uncertainty_m")) else None
    axis_3d = row.get("axis_3d") if row.get("axis_3d") is not None else row.get("axis_world")
    tilt_sign = float(row["tilt_sign"]) if _num(row.get("tilt_sign")) else 1.0
    return VirtualObject(
        oid=oid,
        centre_m=centre,
        top_z=top_z,
        bottom_z=bottom_z,
        length=float(length),
        width=float(width),
        yaw_deg=float(yaw),
        tilt_deg=float(tilt) if _num(tilt) else None,
        support=support,
        quality=quality,
        held_by_arm=held_by,
        occluded=occluded,
        intrinsic_height_m=row.get("intrinsic_height_m"),
        occluded_steps=occ_steps,
        uncertainty_m=uncertainty_m,
        axis_3d=axis_3d,
        tilt_sign=tilt_sign,
        has_measured_bottom_z=has_measured_bottom,
    )


def _parse_track_num(oid: str | None) -> int | None:
    """解析持久物体 ID（如 `'o3'` -> `3`）；非标准前缀返回 `None`。"""
    if not isinstance(oid, str):
        return None
    s = oid.strip()
    if len(s) >= 2 and s[0] in ("o", "O") and s[1:].isdigit():
        val = int(s[1:])
        return val if val > 0 else None
    return None


def _coerce_cloud_data(cloud_or_npz: Any) -> dict | None:
    """将 `Cloud` 实例、`.npz` 字节流/路径或原始数组字典统一归一化为三维点云字典。"""
    if cloud_or_npz is None:
        return None

    if hasattr(cloud_or_npz, "xyz") and hasattr(cloud_or_npz, "robot"):
        xyz = np.asarray(cloud_or_npz.xyz, dtype=np.float64).reshape(-1, 3)
        n = len(xyz)
        robot = np.asarray(cloud_or_npz.robot, dtype=bool).reshape(n)
        raw_conf = getattr(cloud_or_npz, "conf", None)
        if raw_conf is None:
            conf = np.ones(n, dtype=np.float64)
        else:
            arr_c = np.asarray(raw_conf)
            scale = 255.0 if (arr_c.dtype == np.uint8 or (n > 0 and float(np.max(arr_c)) > 1.001)) else 1.0
            conf = arr_c.astype(np.float64).reshape(n) / scale
        raw_oid = getattr(cloud_or_npz, "obj_id", None)
        obj_id = np.asarray(raw_oid, dtype=np.int16).reshape(n) if raw_oid is not None else np.zeros(n, dtype=np.int16)
        raw_vm = getattr(cloud_or_npz, "view_mask", None)
        view_mask = np.asarray(raw_vm, dtype=np.uint8).reshape(n) if raw_vm is not None else np.ones(n, dtype=np.uint8)
        table_z = float(cloud_or_npz.table) if hasattr(cloud_or_npz, "table") and _num(cloud_or_npz.table) else None
        return dict(xyz=xyz, robot=robot, conf=conf, obj_id=obj_id, view_mask=view_mask, table_z=table_z)

    loaded_npz = None
    if isinstance(cloud_or_npz, (bytes, bytearray, memoryview)):
        loaded_npz = np.load(io.BytesIO(bytes(cloud_or_npz)), allow_pickle=False)
    elif isinstance(cloud_or_npz, (str, Path)):
        loaded_npz = np.load(str(cloud_or_npz), allow_pickle=False)

    if loaded_npz is not None:
        with loaded_npz as z:
            data = {k: z[k] for k in z.files}
    elif isinstance(cloud_or_npz, dict) and "xyz" in cloud_or_npz:
        data = cloud_or_npz
    else:
        return None

    xyz = np.asarray(data["xyz"], dtype=np.float64).reshape(-1, 3)
    n = len(xyz)
    robot = np.asarray(data["robot"], dtype=bool).reshape(n) if "robot" in data else np.zeros(n, dtype=bool)
    if "conf" in data:
        arr_c = np.asarray(data["conf"])
        scale = 255.0 if (arr_c.dtype == np.uint8 or (n > 0 and float(np.max(arr_c)) > 1.001)) else 1.0
        conf = arr_c.astype(np.float64).reshape(n) / scale
    else:
        conf = np.ones(n, dtype=np.float64)
    obj_id = np.asarray(data["obj_id"], dtype=np.int16).reshape(n) if "obj_id" in data else np.zeros(n, dtype=np.int16)
    view_mask = np.asarray(data["view_mask"], dtype=np.uint8).reshape(n) if "view_mask" in data else np.ones(n, dtype=np.uint8)
    sup = data.get("support_z", data.get("table_z"))
    table_z = float(sup) if _num(sup) else None
    return dict(xyz=xyz, robot=robot, conf=conf, obj_id=obj_id, view_mask=view_mask, table_z=table_z)


def _extract_stage_spec(
    stage: dict,
    default_poses: dict,
    default_apertures: dict,
    *,
    jaw_geometry: ParallelJawGeometry | None = None,
) -> dict:
    """解析公开 `robot_action` stage 格式与内部编译后 action 格式为统一阶段规格。"""
    del default_apertures
    jg = jaw_geometry if jaw_geometry is not None else DEFAULT_JAW_GEOMETRY
    stage_id = str(stage.get("id") or stage.get("stage_id") or "stage")
    arm = str(stage.get("arm") or "right")
    kind = str(stage.get("kind") or "move")
    own = stage.get("action") if isinstance(stage.get("action"), dict) else {}

    gp = own.get("input_grasp_pose")
    if isinstance(gp, dict) and arm in gp:
        gp = gp[arm]
    if not isinstance(gp, dict):
        gp = {}

    contact_region = str(
        stage.get("contact_region")
        or gp.get("contact_region")
        or own.get("contact_region")
        or "pad_center"
    )
    raw_depth = (
        stage.get("contact_depth_m")
        if _num(stage.get("contact_depth_m"))
        else (
            gp.get("contact_depth_m")
            if _num(gp.get("contact_depth_m"))
            else own.get("contact_depth_m")
        )
    )
    contact_depth_m = float(raw_depth) if _num(raw_depth) else None
    eff_depth = jg.contact_depth(contact_region, contact_depth_m)

    cur_pose = default_poses.get(arm) or next(iter(default_poses.values()), DEFAULT_ARM_POSES.get(arm, DEFAULT_ARM_POSES["right"]))
    pos = None
    ang = None

    if isinstance(stage.get("grasp_position_m"), (list, tuple)) and len(stage["grasp_position_m"]) == 3:
        pos = [float(v) for v in stage["grasp_position_m"]]
        if isinstance(stage.get("angles_deg"), (list, tuple)) and len(stage["angles_deg"]) == 3:
            ang = [float(v) for v in stage["angles_deg"]]
    elif (
        isinstance(stage.get("position_m"), (list, tuple))
        and len(stage["position_m"]) == 3
        and isinstance(stage.get("quaternion_xyzw"), (list, tuple))
        and len(stage["quaternion_xyzw"]) == 4
    ):
        converted = flange_pose_to_grasp_pose(
            stage["position_m"],
            stage["quaternion_xyzw"],
            contact_region=contact_region,
            contact_depth_m=eff_depth,
            grasp_point_offset_m=jg.grasp_point_offset_m,
            fingertip_below_pad_m=jg.fingertip_below_pad_m,
        )
        pos = converted["grasp_position_m"]
        ang = converted["angles_deg"]
    elif isinstance(stage.get("position_m"), (list, tuple)) and len(stage["position_m"]) == 3:
        pos = [float(v) for v in stage["position_m"]]
        if isinstance(stage.get("angles_deg"), (list, tuple)) and len(stage["angles_deg"]) == 3:
            ang = [float(v) for v in stage["angles_deg"]]

    if pos is None and gp:
        if isinstance(gp.get("position_m"), (list, tuple)) and len(gp["position_m"]) == 3:
            pos = [float(v) for v in gp["position_m"]]
        if isinstance(gp.get("angles_deg"), (list, tuple)) and len(gp["angles_deg"]) == 3:
            ang = [float(v) for v in gp["angles_deg"]]

    waypoints = stage.get("waypoints")
    if not (isinstance(waypoints, list) and waypoints):
        path_dict = own.get("path") if isinstance(own.get("path"), dict) else {}
        waypoints = path_dict.get("waypoints")
    if pos is None and isinstance(waypoints, list) and waypoints:
        wp = waypoints[-1]
        if isinstance(wp, dict) and arm in wp and isinstance(wp[arm], dict):
            wp = wp[arm]
        if isinstance(wp, dict):
            if isinstance(wp.get("grasp_position_m"), (list, tuple)) and len(wp["grasp_position_m"]) == 3:
                pos = [float(v) for v in wp["grasp_position_m"]]
                if isinstance(wp.get("angles_deg"), (list, tuple)) and len(wp["angles_deg"]) == 3:
                    ang = [float(v) for v in wp["angles_deg"]]
            elif (
                isinstance(wp.get("position_m"), (list, tuple))
                and len(wp["position_m"]) == 3
                and isinstance(wp.get("quaternion_xyzw"), (list, tuple))
                and len(wp["quaternion_xyzw"]) == 4
            ):
                converted = flange_pose_to_grasp_pose(
                    wp["position_m"],
                    wp["quaternion_xyzw"],
                    contact_region=contact_region,
                    contact_depth_m=eff_depth,
                    grasp_point_offset_m=jg.grasp_point_offset_m,
                    fingertip_below_pad_m=jg.fingertip_below_pad_m,
                )
                pos = converted["grasp_position_m"]
                ang = converted["angles_deg"]
            elif isinstance(wp.get("position_m"), (list, tuple)) and len(wp["position_m"]) == 3:
                pos = [float(v) for v in wp["position_m"]]
                if isinstance(wp.get("angles_deg"), (list, tuple)) and len(wp["angles_deg"]) == 3:
                    ang = [float(v) for v in wp["angles_deg"]]

    pos_from_state = pos is None
    if pos is None:
        pos = list(cur_pose["position_m"])
    if ang is None:
        ang = list(cur_pose["angles_deg"])

    goal_norm = None
    if _num(stage.get("gripper_goal_normalized")):
        goal_norm = float(stage["gripper_goal_normalized"])
    elif _num(stage.get("goal_normalized")):
        goal_norm = float(stage["goal_normalized"])
    elif _num(stage.get("gripper")):
        goal_norm = float(stage["gripper"])
    else:
        events = own.get("gripper_events") or stage.get("gripper_events") or []
        if isinstance(events, list) and events and isinstance(events[-1], dict):
            if _num(events[-1].get("goal_normalized")):
                goal_norm = float(events[-1]["goal_normalized"])

    return dict(
        stage_id=stage_id,
        arm=arm,
        kind=kind,
        position_m=pos,
        pos_from_state=pos_from_state,
        angles_deg=ang,
        goal_normalized=goal_norm,
        contact_region=contact_region,
        contact_depth_m=contact_depth_m,
        allow_contact_ids=list(stage.get("allow_contact_ids") or ()),
        ignore_object_ids=list(stage.get("ignore_object_ids") or ()),
    )


def _transformed_points(field: Any, obj: VirtualObject, base_pts: np.ndarray) -> np.ndarray:
    """若物体在仿真推演中已发生虚拟刚体移动，将绑定点集同步刚体变换到当前位姿。"""
    init_poses = getattr(field, "_initial_obb_poses", None) or {}
    if not obj.virtual_moved or obj.id not in init_poses:
        return base_pts
    c0, axes0 = init_poses[obj.id]
    cur_obb = obj.obb()
    local = (base_pts - c0[None, :]) @ axes0
    return local @ cur_obb.axes.T + cur_obb.center[None, :]


def _object_surface_points(field: Any, oid: str) -> np.ndarray | None:
    """返回物体 `oid` 当前位姿下的真实非机械臂三维表面点集 `(M, 3)`（单位 `m`）；不足 8 点返回 `None`。"""
    objects = getattr(field, "objects", {})
    obj = objects.get(oid) if isinstance(objects, dict) else None
    local_trees = getattr(field, "_obj_local_trees", None) or {}
    surf_pts = getattr(field, "_obj_surface_points", None) or {}
    ktree = local_trees.get(oid)
    if ktree is not None and ktree.n_points >= MIN_OBJECT_SURFACE_POINTS:
        return ktree.world_points(obj.obb()) if obj is not None else surf_pts.get(oid)
    pts = surf_pts.get(oid)
    if pts is None or len(pts) < MIN_OBJECT_SURFACE_POINTS:
        return None
    return _transformed_points(field, obj, pts) if obj is not None else pts


def _fine_surface_points(field: Any, oid: str) -> np.ndarray | None:
    """返回物体 `oid` 用于局部接触截面计算的三维点集（优先真实点云，其次分层凸包切片点），单位 `m`。"""
    pts = _object_surface_points(field, oid)
    if pts is not None:
        return pts
    objects = getattr(field, "objects", {})
    obj = objects.get(oid) if isinstance(objects, dict) else None
    slice_trees = getattr(field, "_obj_slice_trees", None) or {}
    slice_pts = getattr(field, "_obj_slice_points", None) or {}
    stree = slice_trees.get(oid)
    if stree is not None and stree.n_points >= MIN_OBJECT_SURFACE_POINTS:
        return stree.world_points(obj.obb()) if obj is not None else slice_pts.get(oid)
    s_pts = slice_pts.get(oid)
    if s_pts is not None and len(s_pts) >= MIN_OBJECT_SURFACE_POINTS:
        return _transformed_points(field, obj, s_pts) if obj is not None else s_pts
    return None


def _has_fine_geometry(field: Any, oid: str | None = None) -> bool:
    """判断场内物体是否具备实测局部系表面树或实测分层凸包切片几何。"""
    local_trees = getattr(field, "_obj_local_trees", None) or {}
    surf_pts = getattr(field, "_obj_surface_points", None) or {}
    slice_trees = getattr(field, "_obj_slice_trees", None) or {}
    slice_pts = getattr(field, "_obj_slice_points", None) or {}
    if oid is not None:
        return oid in local_trees or oid in surf_pts or oid in slice_trees or oid in slice_pts
    return bool(local_trees or surf_pts or slice_trees or slice_pts)


def _classify_object_geom(field: Any, obj: VirtualObject) -> tuple:
    """按局部点云树 -> 裸点云 -> 分层切片树 -> 裸切片点 -> 实心 `OBB3D` 优先级归类物体碰撞几何。"""
    oid = obj.id
    local_trees = getattr(field, "_obj_local_trees", None) or {}
    ktree = local_trees.get(oid)
    if ktree is not None and ktree.n_points >= MIN_OBJECT_SURFACE_POINTS:
        return ("kdtree", ktree, obj.obb())

    pts = _object_surface_points(field, oid)
    if pts is not None:
        return ("points", pts)

    slice_trees = getattr(field, "_obj_slice_trees", None) or {}
    stree = slice_trees.get(oid)
    if stree is not None and stree.n_points >= MIN_OBJECT_SURFACE_POINTS:
        return ("kdtree", stree, obj.obb())

    slice_pts = getattr(field, "_obj_slice_points", None) or {}
    s_pts = slice_pts.get(oid)
    if s_pts is not None and len(s_pts) >= MIN_OBJECT_SURFACE_POINTS:
        return ("points", _transformed_points(field, obj, s_pts))

    return ("obbs", [obj.obb()])


def object_signed_distance(field: Any, query_obb: OBB3D | VirtualObject | str, obj: VirtualObject | str | OBB3D) -> float:
    """计算运动部件 `query_obb` 到场内物体 `obj` 的三维有符号距离（单位 `m`，兼容双向参数顺序）。

    参数:
        field: 空间模拟场实例（`VirtualSpatialField` 或 `PointCloudSpatialField`）。
        query_obb: 查询部件 `OBB3D`（或反向传入物体）。
        obj: 目标 `VirtualObject`、物体 ID 字符串（或反向传入 `OBB3D`）。

    返回:
        有符号距离，单位 `m`（`> 0` 分离净空，`<= 0` 穿透）。
    """
    if isinstance(query_obb, (str, VirtualObject)) and isinstance(obj, OBB3D):
        query_obb, obj = obj, query_obb
    if isinstance(obj, str):
        obj = field.objects[obj]
    vox = float(getattr(field, "surface_voxel_radius_m", DEFAULT_SURFACE_VOXEL_RADIUS_M))
    gkind = _classify_object_geom(field, obj)
    if gkind[0] == "kdtree":
        return gkind[1].signed_distance_to_obb(query_obb, gkind[2], voxel_radius_m=vox)
    if gkind[0] == "points":
        return obb_to_points_signed_distance(query_obb, gkind[1], voxel_radius_m=vox)
    return obb_signed_distance(query_obb, gkind[1][0])


def evaluate_grasp_clearance_batch(
    field_or_objects: Any,
    target: VirtualObject,
    positions_m: Any,
    angles_deg: Any,
    *,
    arm: str = "right",
    open_aperture_m: float | None = None,
    contact_region: str = "pad_center",
    contact_depth_m: float | None = None,
    table_z: float | None = None,
    jaw_geometry: ParallelJawGeometry | None = None,
    surface_voxel_radius_m: float | None = None,
) -> list[dict]:
    """对 `G` 组候选抓取位姿 `(positions_m, angles_deg)` 统一批量评估三维可夹持性、咬合高度与邻物净空。

    同时支持实心 `OBB3D` 与精细 `LocalSurfaceKDTree` 点云/切片几何，绝不合成 `optimal_grasp_pose`。

    参数:
        field_or_objects: `VirtualSpatialField` / `PointCloudSpatialField` 实例，或 `VirtualObject` 字典/列表。
        target: 目标 `VirtualObject` 实例。
        positions_m: 候选抓取位置数组 `(G, 3)`，单位 `m` (`env_origin`)。
        angles_deg: 候选姿态角数组 `[pitch_deg, roll_deg, yaw_deg]` `(G, 3)`，单位 `deg`。
        arm: 机械臂名称 (`'left' | 'right'`)，默认 `'right'`。
        open_aperture_m: 下探前夹爪张开宽度，单位 `m`；默认 `None` 自动取 `jaw_geometry.max_opening_m`。
        contact_region: 接触截面 (`'pad_center' | 'fingertip' | 'finger_middle'`)。
        contact_depth_m: 可选自定义接触深度偏移，单位 `m`。
        table_z: 可选实测台面高程覆盖值，单位 `m`。
        jaw_geometry: 可选 `ParallelJawGeometry` 覆盖值。
        surface_voxel_radius_m: 可选表面点体素球半径覆盖值，单位 `m`。

    返回:
        长度为 `G` 的客观几何诊断字典列表（不含 `optimal_grasp_pose`）。

    示例:
        `reports = evaluate_grasp_clearance_batch(vf, vf.objects['o1'], [[0.1, -0.2, 0.78]], [[0, 0, -60]])`
    """
    pos_arr = np.asarray(positions_m, dtype=np.float64).reshape(-1, 3)
    g = pos_arr.shape[0]
    if g == 0:
        return []

    if hasattr(field_or_objects, "objects"):
        objects_dict = field_or_objects.objects
        field_obj = field_or_objects
    elif isinstance(field_or_objects, dict):
        objects_dict = field_or_objects
        field_obj = None
    else:
        objects_dict = {o.id: o for o in (field_or_objects or ())}
        field_obj = None

    eff_table_z = table_z if table_z is not None else getattr(field_obj, "table_z", None)
    jg = jaw_geometry if jaw_geometry is not None else (getattr(field_obj, "jaw_geometry", None) or DEFAULT_JAW_GEOMETRY)
    vox = float(
        surface_voxel_radius_m
        if surface_voxel_radius_m is not None
        else getattr(field_obj, "surface_voxel_radius_m", DEFAULT_SURFACE_VOXEL_RADIUS_M)
    )

    finger_half = np.array([0.5 * jg.finger_length_m, 0.5 * jg.finger_thickness_m, 0.5 * jg.finger_width_m], dtype=np.float64)
    palm_half = np.array([0.5 * jg.palm_length_m, 0.5 * jg.palm_width_m, 0.5 * jg.palm_thickness_m], dtype=np.float64)
    part_halves = np.stack([finger_half, finger_half, palm_half], axis=0)

    ang_arr = np.asarray(angles_deg, dtype=np.float64).reshape(g, 3)
    eff_open = jg.max_opening_m if open_aperture_m is None else float(open_aperture_m)
    ap = float(np.clip(eff_open, 0.0, jg.max_opening_m))
    depth = jg.contact_depth(contact_region, contact_depth_m)

    rot_mats = angles_deg_to_matrices_batch(ang_arr)
    u_app = rot_mats[:, :, 0]
    u_close = -rot_mats[:, :, 1]
    u_tan = rot_mats[:, :, 2]
    axes_g = np.stack([u_app, u_close, u_tan], axis=2)
    g_bases = np.stack([u_close, u_tan, -u_app], axis=2)

    pad_c = pos_arr + depth * u_app
    finger_ax = pad_c + (jg.fingertip_below_pad_m - 0.5 * jg.finger_length_m) * u_app
    lat_offset = 0.5 * ap + 0.5 * jg.finger_thickness_m
    left_c = finger_ax - lat_offset * u_close
    right_c = finger_ax + lat_offset * u_close
    palm_c = pad_c - (jg.finger_length_m - jg.fingertip_below_pad_m + 0.5 * jg.palm_length_m) * u_app
    part_centers = np.stack([left_c, right_c, palm_c], axis=1)
    part_axes_flat = np.repeat(axes_g, 3, axis=0)
    part_half_flat = np.tile(part_halves, (g, 1))

    target_obb = target.obb()
    obj_axes = target.world_axes()
    target_centre = np.asarray(target.centre_m, dtype=np.float64)
    axial_offset_m = np.abs(np.einsum("gd,d->g", target_centre[None, :] - pad_c, obj_axes[:, 0]))

    footprint_hit, vertical_bite_m, c_min, c_max, projected_span_m, lateral_eccentricity_m = (
        clip_target_obb_in_gripper_frames(
            target_obb,
            pad_c,
            g_bases,
            finger_width_m=jg.finger_width_m,
            pad_depth_m=jg.pad_depth_m,
        )
    )

    half_ap = 0.5 * ap
    target_has_fine = field_obj is not None and _has_fine_geometry(field_obj, target.id)
    if target_has_fine:
        t_gkind = _classify_object_geom(field_obj, target)
        if t_gkind[0] == "kdtree":
            d_tgt = t_gkind[1].signed_distances_to_obbs(
                part_centers.reshape(g * 3, 3),
                part_axes_flat,
                part_half_flat,
                t_gkind[2],
                voxel_radius_m=vox,
                skip_above_m=-0.0019,
            ).reshape(g, 3)
        elif t_gkind[0] == "points":
            d_tgt = obb_to_points_signed_distance_batch(
                part_centers.reshape(g * 3, 3),
                part_axes_flat,
                part_half_flat,
                t_gkind[1],
                voxel_radius_m=vox,
                skip_above_m=-0.0019,
            ).reshape(g, 3)
        else:
            d_tgt = np.full((g, 3), 1.0, dtype=np.float64)

        cav_map = getattr(field_obj, "cavity_depths_m", None) or {}
        target_cav = float(cav_map.get(target.id, 0.0))
        target_pts = _fine_surface_points(field_obj, target.id)
        half_fw = 0.5 * jg.finger_width_m
        half_pd = 0.5 * jg.pad_depth_m
        if target_pts is not None:
            cand_mask = np.ones(g, dtype=bool) if target_cav >= MIN_CAVITY_DEPTH_M else (projected_span_m > ap - 0.004)
            cand_idx = np.flatnonzero(cand_mask)
            if cand_idx.size > 0:
                footprint_hit = footprint_hit.copy()
                vertical_bite_m = vertical_bite_m.copy()
                c_min = c_min.copy()
                c_max = c_max.copy()
                projected_span_m = projected_span_m.copy()
                lateral_eccentricity_m = lateral_eccentricity_m.copy()
                for start in range(0, cand_idx.size, 32):
                    sub = cand_idx[start : start + 32]
                    gb_sub = g_bases[sub]
                    pc_sub = pad_c[sub]
                    t_pts_g = (
                        np.einsum("md,gdk->gmk", target_pts, gb_sub)
                        - np.einsum("gd,gdk->gk", pc_sub, gb_sub)[:, None, :]
                    )
                    in_jaw = (
                        (np.abs(t_pts_g[:, :, 1]) <= half_fw + vox)
                        & (np.abs(t_pts_g[:, :, 2]) <= half_pd + vox)
                        & (np.abs(t_pts_g[:, :, 0]) <= half_ap)
                    )
                    any_in = np.any(in_jaw, axis=1)
                    if np.any(any_in):
                        hit_g = sub[any_in]
                        in_sub = in_jaw[any_in]
                        pts_sub = t_pts_g[any_in]
                        z_lo_p = np.maximum(-half_pd, np.min(np.where(in_sub, pts_sub[:, :, 2], np.inf), axis=1) - vox)
                        z_hi_p = np.minimum(half_pd, np.max(np.where(in_sub, pts_sub[:, :, 2], -np.inf), axis=1) + vox)
                        x_min_p = np.min(np.where(in_sub, pts_sub[:, :, 0], np.inf), axis=1) - 0.5 * vox
                        x_max_p = np.max(np.where(in_sub, pts_sub[:, :, 0], -np.inf), axis=1) + 0.5 * vox
                        footprint_hit[hit_g] = True
                        vertical_bite_m[hit_g] = np.maximum(0.0, z_hi_p - z_lo_p)
                        c_min[hit_g] = x_min_p
                        c_max[hit_g] = x_max_p
                        projected_span_m[hit_g] = np.maximum(0.004, x_max_p - x_min_p)
                        lateral_eccentricity_m[hit_g] = np.abs(0.5 * (x_min_p + x_max_p))

        finger_hits_target = (d_tgt[:, 0] < -0.002) | (d_tgt[:, 1] < -0.002)
        palm_hits_target = d_tgt[:, 2] < -0.002
    else:
        tgt_centers = np.broadcast_to(target_obb.center[None, :], (g * 3, 3))
        tgt_axes = np.broadcast_to(target_obb.axes[None, :, :], (g * 3, 3, 3))
        tgt_half = np.broadcast_to(target_obb.half_extents[None, :], (g * 3, 3))
        d_target_sat = obb_sat_max_gap_batch(
            part_centers.reshape(g * 3, 3), part_axes_flat, part_half_flat, tgt_centers, tgt_axes, tgt_half
        ).reshape(g, 3)

        hits_left_slice = (
            footprint_hit
            & (vertical_bite_m > 0.0)
            & (c_min < -half_ap + 0.0015)
            & (c_max > -half_ap - jg.finger_thickness_m - 0.0015)
        )
        hits_right_slice = (
            footprint_hit
            & (vertical_bite_m > 0.0)
            & (c_max > half_ap - 0.0015)
            & (c_min < half_ap + jg.finger_thickness_m + 0.0015)
        )
        finger_hits_target = (
            (d_target_sat[:, 0] < -0.002) | (d_target_sat[:, 1] < -0.002) | hits_left_slice | hits_right_slice
        )
        palm_hits_target = d_target_sat[:, 2] < -0.002

    support_z = _effective_support_z(target, eff_table_z)
    abs_axes_z = np.abs(axes_g[:, 2, :])
    ext_z_f = np.einsum("gd,d->g", abs_axes_z, finger_half)
    ext_z_p = np.einsum("gd,d->g", abs_axes_z, palm_half)
    gripper_min_z = np.minimum(np.minimum(left_c[:, 2] - ext_z_f, right_c[:, 2] - ext_z_f), palm_c[:, 2] - ext_z_p)
    support_clearance_m = (gripper_min_z - support_z) if support_z is not None else None
    min_bite_m = _min_vertical_bite_m(target)
    max_allowed_ecc = np.maximum(0.0, 0.5 * (ap - projected_span_m))

    cos_long = np.clip(np.abs(np.einsum("gd,d->g", u_close, obj_axes[:, 0])), 0.0, 1.0)
    closing_to_long_axis = np.degrees(np.arccos(cos_long))

    neighbors = [o for o in objects_dict.values() if o.id != target.id and o.held_by_arm is None]
    k = len(neighbors)
    if k > 0:
        nb_ids = [o.id for o in neighbors]
        d_min_gk = np.ones((g, k), dtype=np.float64)
        pc_flat = part_centers.reshape(g * 3, 3)
        solid_obbs: list[OBB3D] = []
        solid_owners: list[int] = []

        for j, other in enumerate(neighbors):
            gkind = _classify_object_geom(field_obj, other) if field_obj is not None else ("obbs", [other.obb()])
            if gkind[0] == "kdtree":
                dq = gkind[1].signed_distances_to_obbs(
                    pc_flat, part_axes_flat, part_half_flat, gkind[2], voxel_radius_m=vox, skip_above_m=1.0
                ).reshape(g, 3)
                d_min_gk[:, j] = np.minimum(1.0, np.min(dq, axis=1))
            elif gkind[0] == "points":
                dq = obb_to_points_signed_distance_batch(
                    pc_flat, part_axes_flat, part_half_flat, gkind[1], voxel_radius_m=vox, skip_above_m=1.0
                ).reshape(g, 3)
                d_min_gk[:, j] = np.minimum(1.0, np.min(dq, axis=1))
            else:
                for b in gkind[1]:
                    solid_obbs.append(b)
                    solid_owners.append(j)

        s_cnt = len(solid_obbs)
        if s_cnt > 0:
            sc = np.stack([b.center for b in solid_obbs], axis=0)
            sa = np.stack([b.axes for b in solid_obbs], axis=0)
            sh = np.stack([b.half_extents for b in solid_obbs], axis=0)
            ca_all = np.repeat(pc_flat, s_cnt, axis=0)
            aa_all = np.repeat(part_axes_flat, s_cnt, axis=0)
            ha_all = np.repeat(part_half_flat, s_cnt, axis=0)
            cb_all = np.tile(sc, (g * 3, 1))
            ab_all = np.tile(sa, (g * 3, 1, 1))
            hb_all = np.tile(sh, (g * 3, 1))
            ub_all = _clamp_single_point_upper_bound(ca_all, aa_all, ha_all, cb_all, ab_all, hb_all).reshape(g, 3 * s_cnt)
            cand_ub = np.minimum(np.min(d_min_gk, axis=1), np.min(ub_all, axis=1))
            global_ub = max(float(np.max(cand_ub)), 0.0) + 1e-7
            d_slabs = obb_signed_distance_batch(
                ca_all, aa_all, ha_all, cb_all, ab_all, hb_all, skip_above_m=global_ub
            ).reshape(g, 3, s_cnt)
            min_part_gs = np.min(d_slabs, axis=1)
            for s_i, owner_j in enumerate(solid_owners):
                d_min_gk[:, owner_j] = np.minimum(d_min_gk[:, owner_j], min_part_gs[:, s_i])

        min_nb_clr = np.minimum(1.0, np.min(d_min_gk, axis=1))
    else:
        nb_ids = []
        d_min_gk = np.empty((g, 0), dtype=np.float64)
        min_nb_clr = np.ones(g, dtype=np.float64)

    depth_rounded = _r(depth, 4)
    results: list[dict] = []
    for i in range(g):
        reasons: list[str] = []
        if not footprint_hit[i]:
            reasons.append("grasp_misses_target_footprint")
        if vertical_bite_m[i] <= 0.0 or vertical_bite_m[i] < min_bite_m:
            reasons.append("insufficient_vertical_bite")
        if support_clearance_m is not None and support_clearance_m[i] < -0.006:
            reasons.append("fingertip_penetrates_support")
        if projected_span_m[i] > ap - 0.004:
            reasons.append("projected_span_exceeds_opening")
        if footprint_hit[i] and lateral_eccentricity_m[i] > max_allowed_ecc[i] + 0.0015:
            reasons.append("excessive_lateral_eccentricity")
        if finger_hits_target[i]:
            reasons.append("finger_hits_target_on_descent")
        if palm_hits_target[i]:
            reasons.append("palm_collides_with_target")
        col_ids = [nb_ids[j] for j in range(k) if d_min_gk[i, j] < -0.001] if k > 0 else []
        if col_ids:
            reasons.append("open_finger_collides_with_neighbor")

        results.append(
            dict(
                arm=arm,
                target_id=target.id,
                contact_region=contact_region,
                contact_depth_m=depth_rounded,
                theoretically_graspable=len(reasons) == 0,
                failure_reasons=reasons,
                finger_hits_target=bool(finger_hits_target[i]),
                palm_hits_target=bool(palm_hits_target[i]),
                projected_span_m=_r(projected_span_m[i], 4),
                vertical_bite_m=_r(vertical_bite_m[i], 4),
                lateral_eccentricity_m=_r(lateral_eccentricity_m[i], 4),
                max_allowed_eccentricity_m=_r(max_allowed_ecc[i], 4),
                axial_offset_m=_r(axial_offset_m[i], 4),
                closing_to_long_axis_deg=_r(closing_to_long_axis[i], 1),
                support_clearance_m=_r(support_clearance_m[i], 4) if support_clearance_m is not None else None,
                neighbor_clearance_m=_r(min_nb_clr[i], 4),
                colliding_neighbor_ids=col_ids,
            )
        )
    return results


def evaluate_grasp_clearance(
    field: Any,
    arm: str = "right",
    position_m: Any = None,
    angles_deg: Any = None,
    target_id: str | dict | VirtualObject | None = None,
    open_aperture_m: float | None = None,
    contact_region: str = "pad_center",
    contact_depth_m: float | None = None,
) -> dict:
    """在空间场中评估单次三维抓取位姿的客观几何可行性、咬合高度与周围障碍物净空（不含 `optimal_grasp_pose`）。

    参数:
        field: `VirtualSpatialField` 或 `PointCloudSpatialField` 实例。
        arm: 机械臂名称 (`'left' | 'right'`)，默认 `'right'`。
        position_m: 抓取目标位置 `[x, y, z]`，单位 `m` (`env_origin`)。
        angles_deg: 抓取姿态角 `[pitch_deg, roll_deg, yaw_deg]`，单位 `deg`；默认 `[0, 0, 0]`。
        target_id: 目标物体 ID 或 `VirtualObject` / 物体字典；`None` 时按 `position_m` 就近匹配。
        open_aperture_m: 下探前夹爪张开宽度，单位 `m`；默认 `None` 自动取 `jaw_geometry.max_opening_m`。
        contact_region: 接触截面 (`'pad_center' | 'fingertip' | 'finger_middle'`)。
        contact_depth_m: 可选自定义接触深度偏移，单位 `m`。

    返回:
        客观抓取净空与咬合诊断字典（不含 `optimal_grasp_pose`）。
    """
    if angles_deg is None:
        angles_deg = [0.0, 0.0, 0.0]
    target = field.find_target_object(target_id, position_m=position_m)
    if target is None:
        return dict(
            arm=arm,
            target_id=target_id if isinstance(target_id, str) else None,
            contact_region=contact_region,
            contact_depth_m=_r(contact_depth_m, 4) if contact_depth_m is not None else 0.0,
            theoretically_graspable=False,
            failure_reasons=["no_target_object_in_reach"],
            finger_hits_target=False,
            palm_hits_target=False,
            projected_span_m=None,
            vertical_bite_m=0.0,
            lateral_eccentricity_m=None,
            support_clearance_m=None,
            neighbor_clearance_m=None,
            colliding_neighbor_ids=[],
        )
    if position_m is None:
        position_m = list(target.centre_m)
    return evaluate_grasp_clearance_batch(
        field,
        target,
        [position_m],
        [angles_deg],
        arm=arm,
        open_aperture_m=open_aperture_m,
        contact_region=contact_region,
        contact_depth_m=contact_depth_m,
    )[0]


def inspect_cavity_geometry(
    field_or_cloud_data: Any,
    target_id_or_region: Any,
    radius_m: float = 0.06,
    *,
    objects: dict[str, VirtualObject] | None = None,
    table_z: float | None = None,
    cavity_depths_m: dict[str, float] | None = None,
    surface_voxel_radius_m: float | None = None,
) -> dict:
    """测量目标物体或局部三维区域的凹腔几何剖面（边缘高程、内底高程、凹腔深度、开口中心与最大内切无碰撞半径）。

    参数:
        field_or_cloud_data: `PointCloudSpatialField` 实例或归一化点云字典 `_cloud_data`。
        target_id_or_region: 持久物体 ID（如 `'o2'`）或空间查询区域（`[x, y]`、`{'center': [x, y], 'radius': r}` 或 `(lo, hi)`）。
        radius_m: 水平搜索半径，单位 `m` (`> 0`)，默认 `0.06`。
        objects: 可选物体字典覆盖值。
        table_z: 可选实测台面高程覆盖值，单位 `m`。
        cavity_depths_m: 可选凹腔深度映射字典覆盖值，单位 `m`。
        surface_voxel_radius_m: 可选表面点体素球半径覆盖值，单位 `m`；默认 `None` 取场实例配置或 `0.006`。

    返回:
        字典 `{rim_z_m, floor_z_m, cavity_depth_m, opening_center_xy_m, inner_clearance_radius_m, n_points}`（单位 `m`）。
    """
    r = float(radius_m)
    if not math.isfinite(r) or r <= 0.0:
        raise ValueError("radius_m must be a positive finite number")

    if hasattr(field_or_cloud_data, "objects"):
        field_obj = field_or_cloud_data
        cd = getattr(field_or_cloud_data, "_cloud_data", None)
        objs = objects if objects is not None else field_or_cloud_data.objects
        eff_table_z = table_z if table_z is not None else getattr(field_or_cloud_data, "table_z", None)
        cav_map = (
            cavity_depths_m
            if cavity_depths_m is not None
            else (getattr(field_or_cloud_data, "cavity_depths_m", None) or {})
        )
        vox = float(
            surface_voxel_radius_m
            if surface_voxel_radius_m is not None
            else getattr(field_or_cloud_data, "surface_voxel_radius_m", DEFAULT_SURFACE_VOXEL_RADIUS_M)
        )
    else:
        field_obj = None
        cd = field_or_cloud_data
        objs = objects or {}
        eff_table_z = table_z
        cav_map = cavity_depths_m or {}
        vox = float(surface_voxel_radius_m if surface_voxel_radius_m is not None else DEFAULT_SURFACE_VOXEL_RADIUS_M)

    obj: VirtualObject | None = None
    ref_xy: np.ndarray | None = None
    search_r = r
    pts = np.empty((0, 3), dtype=np.float64)

    has_cloud = isinstance(cd, dict) and len(cd["xyz"]) > 0
    valid_xyz = cd["xyz"][(~cd["robot"]) & (cd["conf"] >= 0.05)] if has_cloud else np.empty((0, 3), dtype=np.float64)

    if isinstance(target_id_or_region, str):
        obj = objs.get(target_id_or_region)
        if obj is None:
            raise ValueError(f"target_object_not_found:{target_id_or_region}")
        ref_xy = np.asarray(obj.centre_m[:2], dtype=np.float64)
        search_r = max(r, 0.5 * max(float(obj.length), float(obj.width)))
        surf_pts = (
            _object_surface_points(field_obj, obj.id)
            if (field_obj is not None and getattr(field_obj, "objects", {}).get(obj.id) is obj)
            else None
        )
        if surf_pts is not None:
            pts = surf_pts
        elif not obj.virtual_moved and len(valid_xyz) > 0:
            obb = obj.obb()
            rel = (valid_xyz - obb.center[None, :]) @ obb.axes
            z_lo_bound = (float(eff_table_z) - 0.005) if eff_table_z is not None else (float(obj.bottom_z) - 0.005)
            in_foot = (
                (np.abs(rel[:, 0]) <= obb.half_extents[0] + 0.008)
                & (np.abs(rel[:, 1]) <= obb.half_extents[1] + 0.008)
                & (valid_xyz[:, 2] >= z_lo_bound)
                & (valid_xyz[:, 2] <= float(obj.top_z) + 0.015)
            )
            pts = valid_xyz[in_foot]
            t_num = _parse_track_num(obj.id)
            if t_num is not None and has_cloud and np.any(cd["obj_id"] == t_num):
                mask_pts = cd["xyz"][(~cd["robot"]) & (cd["conf"] >= 0.05) & (cd["obj_id"] == t_num)]
                if len(mask_pts) > len(pts):
                    pts = mask_pts
    elif isinstance(target_id_or_region, dict) and "center" in target_id_or_region:
        c_raw = np.asarray(target_id_or_region["center"], dtype=np.float64).ravel()
        ref_xy = c_raw[:2]
        search_r = float(target_id_or_region.get("radius", r))
        if len(valid_xyz) > 0:
            d_xy = np.hypot(valid_xyz[:, 0] - ref_xy[0], valid_xyz[:, 1] - ref_xy[1])
            m = d_xy <= search_r
            if "z_min" in target_id_or_region:
                m &= valid_xyz[:, 2] >= float(target_id_or_region["z_min"])
            if "z_max" in target_id_or_region:
                m &= valid_xyz[:, 2] <= float(target_id_or_region["z_max"])
            pts = valid_xyz[m]
    elif (
        isinstance(target_id_or_region, (list, tuple))
        and len(target_id_or_region) == 2
        and isinstance(target_id_or_region[0], (list, tuple, np.ndarray))
    ):
        lo = np.asarray(target_id_or_region[0], dtype=np.float64).reshape(3)
        hi = np.asarray(target_id_or_region[1], dtype=np.float64).reshape(3)
        ref_xy = 0.5 * (lo[:2] + hi[:2])
        search_r = 0.5 * float(max(hi[0] - lo[0], hi[1] - lo[1]))
        if len(valid_xyz) > 0:
            m = np.all((valid_xyz >= lo[None, :]) & (valid_xyz <= hi[None, :]), axis=1)
            pts = valid_xyz[m]
    else:
        c_raw = np.asarray(target_id_or_region, dtype=np.float64).ravel()
        if len(c_raw) < 2:
            raise ValueError("region must be an object id, [x, y], {'center': [x, y], 'radius': r}, or (lo, hi)")
        ref_xy = c_raw[:2]
        if len(valid_xyz) > 0:
            d_xy = np.hypot(valid_xyz[:, 0] - ref_xy[0], valid_xyz[:, 1] - ref_xy[1])
            pts = valid_xyz[d_xy <= search_r]

    if len(pts) == 0:
        if obj is not None:
            cav = float(cav_map.get(obj.id, 0.0))
            rim_z = float(obj.top_z)
            floor_z = max(float(obj.bottom_z), rim_z - cav) if cav >= MIN_CAVITY_DEPTH_M else rim_z
            return dict(
                rim_z_m=_r(rim_z, 4),
                floor_z_m=_r(floor_z, 4),
                cavity_depth_m=_r(max(0.0, rim_z - floor_z), 4),
                opening_center_xy_m=_r(ref_xy, 4),
                inner_clearance_radius_m=0.0,
                n_points=0,
            )
        return dict(
            rim_z_m=None,
            floor_z_m=None,
            cavity_depth_m=0.0,
            opening_center_xy_m=_r(ref_xy, 4) if ref_xy is not None else None,
            inner_clearance_radius_m=0.0,
            n_points=0,
        )

    rim_z = float(np.percentile(pts[:, 2], 95))
    z_p05 = float(np.percentile(pts[:, 2], 5))
    high_band = pts[pts[:, 2] >= rim_z - max(0.008, 0.25 * max(0.0, rim_z - z_p05))]
    if ref_xy is None:
        ref_xy = np.median(high_band[:, :2], axis=0) if len(high_band) > 0 else np.median(pts[:, :2], axis=0)

    if len(high_band) >= 4:
        lo_xy = np.min(high_band[:, :2], axis=0)
        hi_xy = np.max(high_band[:, :2], axis=0)
        span_r = 0.5 * float(np.linalg.norm(hi_xy - lo_xy))
    else:
        span_r = search_r

    inner_r_probe = max(0.010, min(search_r, 0.65 * span_r))
    d_ref = np.hypot(pts[:, 0] - ref_xy[0], pts[:, 1] - ref_xy[1])
    inner_pts = pts[d_ref <= inner_r_probe]
    floor_z = float(np.percentile(inner_pts[:, 2], 10)) if len(inner_pts) >= 3 else z_p05

    min_d_high = float(np.min(np.hypot(high_band[:, 0] - ref_xy[0], high_band[:, 1] - ref_xy[1]))) if len(high_band) > 0 else 0.0
    if rim_z - floor_z < MIN_CAVITY_DEPTH_M:
        if obj is not None and cav_map.get(obj.id, 0.0) >= MIN_CAVITY_DEPTH_M:
            floor_z = max(float(obj.bottom_z), rim_z - float(cav_map[obj.id]))
        elif min_d_high >= 0.008:
            if obj is not None:
                floor_z = float(obj.bottom_z)
            elif eff_table_z is not None:
                floor_z = float(eff_table_z)

    cavity_depth = max(0.0, rim_z - floor_z)
    if cavity_depth < MIN_CAVITY_DEPTH_M:
        return dict(
            rim_z_m=_r(rim_z, 4),
            floor_z_m=_r(rim_z, 4),
            cavity_depth_m=0.0,
            opening_center_xy_m=_r(ref_xy, 4),
            inner_clearance_radius_m=0.0,
            n_points=int(len(pts)),
        )

    wall_z_cut = floor_z + max(MIN_CAVITY_DEPTH_M, 0.35 * cavity_depth)
    wall_xy = pts[pts[:, 2] >= wall_z_cut, :2]
    if len(wall_xy) < 4:
        return dict(
            rim_z_m=_r(rim_z, 4),
            floor_z_m=_r(floor_z, 4),
            cavity_depth_m=_r(cavity_depth, 4),
            opening_center_xy_m=_r(ref_xy, 4),
            inner_clearance_radius_m=0.0,
            n_points=int(len(pts)),
        )

    w_lo = np.maximum(np.min(wall_xy, axis=0) + 0.002, ref_xy - search_r)
    w_hi = np.minimum(np.max(wall_xy, axis=0) - 0.002, ref_xy + search_r)
    if np.all(w_hi > w_lo):
        gx = np.linspace(w_lo[0], w_hi[0], 17)
        gy = np.linspace(w_lo[1], w_hi[1], 17)
        gxx, gyy = np.meshgrid(gx, gy)
        cands = np.vstack([ref_xy[None, :], np.column_stack([gxx.ravel(), gyy.ravel()])])
    else:
        cands = ref_xy[None, :]

    dists = np.linalg.norm(cands[:, None, :] - wall_xy[None, :, :], axis=2)
    min_wall_d = np.min(dists, axis=1)
    center_bias = 0.05 * np.linalg.norm(cands - ref_xy[None, :], axis=1)
    best_idx = int(np.argmax(min_wall_d - center_bias))
    best_xy = cands[best_idx]
    clearance_r = max(0.0, float(min_wall_d[best_idx]) - 0.5 * vox)

    return dict(
        rim_z_m=_r(rim_z, 4),
        floor_z_m=_r(floor_z, 4),
        cavity_depth_m=_r(cavity_depth, 4),
        opening_center_xy_m=_r(best_xy, 4),
        inner_clearance_radius_m=_r(clearance_r, 4),
        n_points=int(len(pts)),
    )


def object_cavity_floor_z(field: Any, obj: VirtualObject) -> float | None:
    """估计凹腔容器物体的内底支撑高程 `floor_z_m`（单位 `m`）；非凹腔物体返回 `None`。"""
    cav_map = getattr(field, "cavity_depths_m", None) or {}
    cav = float(cav_map.get(obj.id, 0.0))
    pts = _object_surface_points(field, obj.id)
    if pts is not None:
        prof = inspect_cavity_geometry(field, obj.id)
        if prof["cavity_depth_m"] >= MIN_CAVITY_DEPTH_M and prof["floor_z_m"] is not None:
            return float(prof["floor_z_m"])
    if cav >= MIN_CAVITY_DEPTH_M:
        return max(float(obj.bottom_z), float(obj.top_z) - cav)
    return None


def simulate_stage_clearances(
    field: Any,
    stages: Any,
    step_m: float = 0.015,
    *,
    allow_contact_ids: list[str] | tuple[str, ...] | set[str] | None = None,
    ignore_object_ids: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict:
    """在虚拟场克隆副本上按顺序动态推演多阶段动作序列 `[stage_1, stage_2, ...]`。

    统一支持实心 `OBB3D` 与精细 `LocalSurfaceKDTree` 点云/切片几何及凹腔内底释放；
    统一在 `pad_center_m` 参考系下插值并纯粹客观回报每阶段最小有符号距离（绝不生成 `recommended_safe_z_m` 或 `suggested_fixes`）。

    参数:
        field: `VirtualSpatialField` 或 `PointCloudSpatialField` 实例。
        stages: 阶段列表 `[stage_1, ...]` 或包含 `'stages'` 的动作字典。
        step_m: 路径线性插值采样步长，单位 `m` (`> 0`)，默认 `0.015`。
        allow_contact_ids: 可选允许接触物体 ID 集合；客观记录距离但不记为碰撞风险。
        ignore_object_ids: 可选完全跳过碰撞与净空计算的物体 ID 集合。

    返回:
        包含 `status` (`'clean' | 'risk_detected'`), `theoretically_valid` (`bool`),
        `min_path_clearance_m` (`m`), `min_clearance_by_object_m` (`dict[str, float]`, `m`),
        `stage_reports` (`list[dict]`), `virtual_world_after` (`list[dict]`) 的客观报告。
    """
    step = float(step_m)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("step_m must be a positive finite number")
    top_allow = set(str(x) for x in (allow_contact_ids or ()))
    top_ignore = set(str(x) for x in (ignore_object_ids or ()))
    if isinstance(stages, dict):
        top_allow.update(str(x) for x in (stages.get("allow_contact_ids") or ()))
        top_ignore.update(str(x) for x in (stages.get("ignore_object_ids") or ()))
        raw_stages = list(stages.get("stages") or [])
    else:
        raw_stages = list(stages or [])

    sim = field.clone()
    vox = float(getattr(sim, "surface_voxel_radius_m", DEFAULT_SURFACE_VOXEL_RADIUS_M))

    stage_reports: list[dict] = []
    overall_min_clearance = math.inf
    overall_clearance_by_obj: dict[str, float] = {}
    has_risk = False

    for raw in raw_stages:
        if not isinstance(raw, dict):
            continue
        jg = sim.jaw_geometry
        spec = _extract_stage_spec(raw, sim.arm_poses, sim.arm_apertures_m, jaw_geometry=jg)
        arm = spec["arm"]
        stage_id = spec["stage_id"]
        kind = spec["kind"]
        stage_allow = top_allow | set(str(x) for x in spec["allow_contact_ids"])
        stage_ignore = top_ignore | set(str(x) for x in spec["ignore_object_ids"])

        start_pose = deepcopy(
            sim.arm_poses.get(arm) or next(iter(sim.arm_poses.values()), DEFAULT_ARM_POSES["right"])
        )
        start_pad_pos = np.asarray(start_pose["position_m"], dtype=np.float64)
        start_ang = np.asarray(start_pose["angles_deg"], dtype=np.float64)
        target_pos = np.asarray(spec["position_m"], dtype=np.float64)
        target_ang = np.asarray(spec["angles_deg"], dtype=np.float64)
        eff_region = "pad_center" if spec["pos_from_state"] else spec["contact_region"]
        eff_depth = 0.0 if spec["pos_from_state"] else jg.contact_depth(eff_region, spec["contact_depth_m"])
        end_frame = gripper_frame(
            target_pos,
            target_ang,
            contact_region=eff_region,
            contact_depth_m=eff_depth,
            grasp_point_offset_m=jg.grasp_point_offset_m,
            fingertip_below_pad_m=jg.fingertip_below_pad_m,
        )
        target_pad_pos = np.asarray(end_frame["pad_center_m"], dtype=np.float64)

        max_open = sim.jaw_geometry.max_opening_m
        prev_ap = sim.arm_apertures_m.get(arm, max_open)
        was_open = prev_ap >= 0.5 * max_open
        goal_norm = spec["goal_normalized"]
        held_obj = next((o for o in sim.objects.values() if o.held_by_arm == arm), None)

        stage_min_clearance = math.inf
        stage_obj_clearances: dict[str, float] = {}
        colliding_ids: list[str] = []
        if kind == "move":
            dist = float(np.linalg.norm(target_pad_pos - start_pad_pos))
            ang_deltas = np.array([wrap180(target_ang[k] - start_ang[k]) for k in range(3)], dtype=np.float64)
            max_ang_deg = float(np.max(np.abs(ang_deltas)))
            ang_step_deg = min(5.0, max(5.0 * (step / 0.015), 0.5))
            n_steps = max(2, int(math.ceil(max(dist / step, max_ang_deg / ang_step_deg))) + 1)
            all_moving_obbs: list[OBB3D] = []
            for s_i in range(1, n_steps + 1):
                alpha = float(s_i) / float(n_steps)
                interp_pad_pos = start_pad_pos + alpha * (target_pad_pos - start_pad_pos)
                interp_ang = start_ang + alpha * ang_deltas
                g_obbs = gripper_obbs(
                    interp_pad_pos,
                    interp_ang,
                    aperture_m=prev_ap,
                    contact_region="pad_center",
                    contact_depth_m=0.0,
                    jaw_geometry=sim.jaw_geometry,
                )
                moving_obbs = [g_obbs["left_finger"], g_obbs["right_finger"], g_obbs["palm"]]
                if held_obj is not None:
                    held_obj.update_from_gripper(g_obbs["frame"])
                    moving_obbs.append(held_obj.obb())
                all_moving_obbs.extend(moving_obbs)

                if sim.table_z is not None:
                    min_moving_z = min(float(m_obb.aabb()[0][2]) for m_obb in moving_obbs)
                    table_clr = min_moving_z - float(sim.table_z)
                    if table_clr < stage_min_clearance:
                        stage_min_clearance = table_clr
                    if table_clr < -0.006 and "table" not in colliding_ids:
                        colliding_ids.append("table")

            if all_moving_obbs:
                mc = np.stack([b.center for b in all_moving_obbs], axis=0)
                ma = np.stack([b.axes for b in all_moving_obbs], axis=0)
                mh = np.stack([b.half_extents for b in all_moving_obbs], axis=0)
                for other in sim.objects.values():
                    if other.held_by_arm == arm or other.id in stage_ignore:
                        continue
                    gkind = _classify_object_geom(sim, other)
                    if gkind[0] == "kdtree":
                        d = gkind[1].batch_stage_clearance(all_moving_obbs, gkind[2], voxel_radius_m=vox)
                    elif gkind[0] == "points":
                        d = float(np.min(obb_to_points_signed_distance_batch(mc, ma, mh, gkind[1], voxel_radius_m=vox)))
                    else:
                        d = batch_obbs_to_slabs_min_distance(all_moving_obbs, gkind[1])

                    if other.id not in stage_obj_clearances or d < stage_obj_clearances[other.id]:
                        stage_obj_clearances[other.id] = d
                    if other.id in stage_allow:
                        continue
                    if d < stage_min_clearance:
                        stage_min_clearance = d
                    if d < -0.002 and other.id not in colliding_ids:
                        colliding_ids.append(other.id)

            sim.arm_poses[arm] = {
                "position_m": [float(v) for v in target_pad_pos],
                "angles_deg": [float(v) for v in target_ang],
            }
            if held_obj is not None:
                held_obj.update_from_gripper(end_frame)

        for oid_k, d_val in stage_obj_clearances.items():
            if oid_k not in overall_clearance_by_obj or d_val < overall_clearance_by_obj[oid_k]:
                overall_clearance_by_obj[oid_k] = d_val

        if stage_min_clearance < overall_min_clearance:
            overall_min_clearance = stage_min_clearance

        reported_stage_clr = _r(stage_min_clearance, 4) if math.isfinite(stage_min_clearance) else None
        stage_entry: dict[str, Any] = dict(
            stage_id=stage_id,
            arm=arm,
            kind=kind,
            target_position_m=_r(target_pos, 4),
            target_angles_deg=_r(target_ang, 1),
            held_object_id=held_obj.id if held_obj is not None else None,
            path_clearance_m=reported_stage_clr,
            min_clearance_by_object_m={k: _r(v, 4) for k, v in stage_obj_clearances.items()},
            colliding_objects=colliding_ids,
            status="ok",
        )

        if colliding_ids:
            has_risk = True
            stage_entry["status"] = "path_collision"

        if was_open and goal_norm is not None and goal_norm < 0.5:
            cmd_ap = float(np.clip(goal_norm, 0.0, 1.0)) * max_open
            grasp_eval = evaluate_grasp_clearance(
                sim,
                arm=arm,
                position_m=target_pos,
                angles_deg=target_ang,
                open_aperture_m=prev_ap,
                contact_region=eff_region,
                contact_depth_m=eff_depth,
            )
            stage_entry["grasp_evaluation"] = grasp_eval
            if grasp_eval["theoretically_graspable"] and grasp_eval["target_id"] in sim.objects:
                grasped = sim.objects[grasp_eval["target_id"]]
                grasped.attach_to_gripper(arm, end_frame)
                sim.arm_apertures_m[arm] = max(0.0, float(grasp_eval.get("projected_span_m") or 0.0))
                stage_entry["grasped_object_id"] = grasped.id
            else:
                has_risk = True
                stage_entry["status"] = "grasp_risk"
                sim.arm_apertures_m[arm] = cmd_ap
        elif goal_norm is not None and goal_norm >= 0.5:
            if held_obj is not None:
                held_centre = np.array(
                    [held_obj.centre_m[0], held_obj.centre_m[1], held_obj.centre_m[2]],
                    dtype=np.float64,
                )
                support_id: str | None = "table" if sim.table_z is not None else None
                support_top: float | None = float(sim.table_z) if sim.table_z is not None else None
                for other in sim.objects.values():
                    if other.id == held_obj.id or other.held_by_arm is not None:
                        continue
                    other_obb = other.obb()
                    delta = held_centre - other_obb.center
                    rel_x = float(np.dot(delta, other_obb.axes[:, 0]))
                    rel_y = float(np.dot(delta, other_obb.axes[:, 1]))
                    if abs(rel_x) <= float(other_obb.half_extents[0]) and abs(rel_y) <= float(other_obb.half_extents[1]):
                        cav_floor = object_cavity_floor_z(sim, other)
                        if cav_floor is not None and held_obj.bottom_z < other.top_z - 0.002:
                            if (support_top is None or cav_floor >= support_top) and cav_floor <= held_obj.bottom_z + 0.002:
                                support_top = cav_floor
                                support_id = other.id
                        elif (support_top is None or other.top_z >= support_top) and other.top_z <= held_obj.bottom_z + 0.002:
                            support_top = other.top_z
                            support_id = other.id
                if support_top is not None:
                    held_obj.detach_from_gripper(support_z=support_top, support_id=support_id or "table")
                else:
                    held_obj.held_by_arm = None
                    held_obj.rel_offset_in_gripper = None
                    held_obj._rel_centre_in_gripper = None
                    held_obj.rel_axes_in_gripper = None
                    held_obj.virtual_moved = True
                stage_entry["released_object_id"] = held_obj.id
                stage_entry["new_object_centre_m"] = _r(held_obj.centre_m, 4)
                stage_entry["new_support"] = support_id
            sim.arm_apertures_m[arm] = float(np.clip(goal_norm, 0.0, 1.0)) * max_open

        stage_reports.append(stage_entry)

    return dict(
        status="risk_detected" if has_risk else "clean",
        theoretically_valid=not has_risk,
        min_path_clearance_m=_r(overall_min_clearance, 4) if math.isfinite(overall_min_clearance) else None,
        min_clearance_by_object_m={k: _r(v, 4) for k, v in overall_clearance_by_obj.items()},
        stage_reports=stage_reports,
        virtual_world_after=[obj.to_dict() for obj in sim.objects.values()],
    )
