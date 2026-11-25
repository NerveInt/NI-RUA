"""点云与分层凹腔空间状态容器 (`PointCloudSpatialField`, 继承 `VirtualSpatialField`)。

保留多视角表面点云缓存与 `LocalSurfaceKDTree` 局部树维护，并将所有几何距离、抓取评估、轨迹推演与凹腔剖面计算
统一委托至 `geometry_ops.py` 与 `clearance_oracle.py`。
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import numpy as np

if __package__:
    from .virtual_field import VirtualSpatialField
    from .geometry_ops import (
        DEFAULT_SURFACE_VOXEL_RADIUS_M,
        MAX_CACHED_OBJECT_POINTS,
        MIN_CAVITY_DEPTH_M,
        MIN_OBJECT_SURFACE_POINTS,
        LocalSurfaceKDTree,
        OBB3D,
        _num,
        derive_object_spatial_meta,
        obb_to_points_signed_distance,
        sample_slice_hull_points,
    )
    from .clearance_oracle import (
        VirtualObject,
        _coerce_cloud_data,
        _fine_surface_points as _oracle_fine_surface_points,
        _has_fine_geometry as _oracle_has_fine_geometry,
        _object_surface_points as _oracle_object_surface_points,
        _parse_track_num,
        evaluate_grasp_clearance,
        inspect_cavity_geometry,
        object_cavity_floor_z as _oracle_cavity_floor_z,
        object_signed_distance as _oracle_object_signed_distance,
        simulate_stage_clearances,
    )
else:
    from virtual_field import VirtualSpatialField  # type: ignore[no-redef]
    from geometry_ops import (  # type: ignore[no-redef]
        DEFAULT_SURFACE_VOXEL_RADIUS_M,
        MAX_CACHED_OBJECT_POINTS,
        MIN_CAVITY_DEPTH_M,
        MIN_OBJECT_SURFACE_POINTS,
        LocalSurfaceKDTree,
        OBB3D,
        _num,
        derive_object_spatial_meta,
        obb_to_points_signed_distance,
        sample_slice_hull_points,
    )
    from clearance_oracle import (  # type: ignore[no-redef]
        VirtualObject,
        _coerce_cloud_data,
        _fine_surface_points as _oracle_fine_surface_points,
        _has_fine_geometry as _oracle_has_fine_geometry,
        _object_surface_points as _oracle_object_surface_points,
        _parse_track_num,
        evaluate_grasp_clearance,
        inspect_cavity_geometry,
        object_cavity_floor_z as _oracle_cavity_floor_z,
        object_signed_distance as _oracle_object_signed_distance,
        simulate_stage_clearances,
    )


class PointCloudSpatialField(VirtualSpatialField):
    """融合真实三维表面点云（含多视角融合点与实例掩码 `obj_id`）及实测分层周界凸包的空间状态容器。"""

    def __init__(
        self,
        objects=None,
        cloud=None,
        table_z: float | None = None,
        robot_state: dict | None = None,
        *,
        cloud_npz=None,
        cavity_depths_m: dict[str, float] | None = None,
        surface_slices_m: dict[str, list[dict]] | None = None,
        raw_objects: list[dict] | None = None,
        surface_voxel_radius_m: float = DEFAULT_SURFACE_VOXEL_RADIUS_M,
        max_occluded_steps: int = 5,
        profile=None,
        jaw_geometry=None,
    ):
        self.surface_voxel_radius_m: float = float(surface_voxel_radius_m)
        self.cavity_depths_m: dict[str, float] = {}
        self.surface_slices_m: dict[str, list[dict]] = {}
        self._cloud_data: dict | None = None
        self._obj_surface_points: dict[str, np.ndarray] = {}
        self._obj_slice_points: dict[str, np.ndarray] = {}
        self._obj_local_trees: dict[str, Any] = {}
        self._obj_slice_trees: dict[str, Any] = {}
        self._initial_obb_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        effective_cloud = cloud if cloud is not None else cloud_npz
        coerced = _coerce_cloud_data(effective_cloud)
        if coerced is not None:
            self._cloud_data = coerced
            if table_z is None and coerced.get("table_z") is not None:
                table_z = float(coerced["table_z"])

        super().__init__(
            objects=None,
            cloud=effective_cloud,
            table_z=table_z,
            robot_state=None,
            max_occluded_steps=max_occluded_steps,
            profile=profile,
            jaw_geometry=jaw_geometry,
        )
        if (
            objects is not None
            or robot_state is not None
            or effective_cloud is not None
            or cavity_depths_m
            or surface_slices_m
            or raw_objects
        ):
            self.update_from_observation(
                objects,
                robot_state=robot_state,
                cloud=effective_cloud,
                cavity_depths_m=cavity_depths_m,
                surface_slices_m=surface_slices_m,
                raw_objects=raw_objects,
            )

    def clone(self) -> "PointCloudSpatialField":
        """创建当前点云空间场的深拷贝副本，共享不可变的物体局部系 `LocalSurfaceKDTree`（零建树开销）。"""
        copied = PointCloudSpatialField(
            table_z=self.table_z,
            surface_voxel_radius_m=self.surface_voxel_radius_m,
            max_occluded_steps=self.max_occluded_steps,
            profile=self.profile,
            jaw_geometry=self.jaw_geometry if self.supported else None,
        )
        copied.supported = self.supported
        copied.observation_id = self.observation_id
        copied.cloud = self.cloud
        copied._cloud_data = self._cloud_data
        copied.objects = deepcopy(self.objects)
        copied.historical_unobserved_objects = deepcopy(self.historical_unobserved_objects)
        copied.arm_poses = deepcopy(self.arm_poses)
        copied.arm_apertures_m = deepcopy(self.arm_apertures_m)
        copied.arm_jaw_goals = dict(self.arm_jaw_goals)
        copied.step_count = self.step_count
        copied.cavity_depths_m = dict(self.cavity_depths_m)
        copied.surface_slices_m = deepcopy(self.surface_slices_m)
        copied._obj_surface_points = dict(self._obj_surface_points)
        copied._obj_slice_points = dict(self._obj_slice_points)
        copied._obj_local_trees = dict(self._obj_local_trees)
        copied._obj_slice_trees = dict(self._obj_slice_trees)
        copied._initial_obb_poses = deepcopy(self._initial_obb_poses)
        return copied

    def _ingest_geometry_metadata(
        self,
        object_table=None,
        *,
        cavity_depths_m: dict[str, float] | None = None,
        surface_slices_m: dict[str, list[dict]] | None = None,
        raw_objects: list[dict] | None = None,
    ) -> None:
        """从 `object_table`、`grasp_hints` 或原始服务端对象列表提取每个物体的凹腔深度与凸包切片。"""
        if object_table is not None:
            self.cavity_depths_m = {}
            self.surface_slices_m = {}

        def _record_item(oid: str, item: dict) -> None:
            cav = item.get("cavity_depth_m")
            if _num(cav) and float(cav) >= MIN_CAVITY_DEPTH_M:
                self.cavity_depths_m[oid] = float(cav)
            slices = item.get("surface_slices_m")
            if isinstance(slices, (list, tuple)) and slices:
                self.surface_slices_m[oid] = [deepcopy(dict(s)) for s in slices if isinstance(s, dict)]

        if isinstance(object_table, dict):
            hints = object_table.get("grasp_hints")
            if isinstance(hints, dict):
                for oid, h in hints.items():
                    if isinstance(h, dict) and _num(h.get("cavity_depth_m")) and float(h["cavity_depth_m"]) >= MIN_CAVITY_DEPTH_M:
                        self.cavity_depths_m[str(oid)] = float(h["cavity_depth_m"])
            for list_key in ("objects", "object_geometries"):
                items = object_table.get(list_key)
                if isinstance(items, list):
                    for idx, item in enumerate(items):
                        if isinstance(item, dict):
                            _record_item(str(item.get("id") or item.get("track_id") or f"o{idx + 1}"), item)
        elif isinstance(object_table, (list, tuple)):
            for idx, item in enumerate(object_table):
                if isinstance(item, dict):
                    _record_item(str(item.get("id") or item.get("track_id") or f"o{idx + 1}"), item)

        if isinstance(raw_objects, (list, tuple)):
            for idx, item in enumerate(raw_objects):
                if isinstance(item, dict):
                    _record_item(str(item.get("id") or item.get("track_id") or f"o{idx + 1}"), item)

        if isinstance(cavity_depths_m, dict):
            for oid, val in cavity_depths_m.items():
                if _num(val) and float(val) >= MIN_CAVITY_DEPTH_M:
                    self.cavity_depths_m[str(oid)] = float(val)

        if isinstance(surface_slices_m, dict):
            for oid, slices in surface_slices_m.items():
                if isinstance(slices, (list, tuple)) and slices:
                    self.surface_slices_m[str(oid)] = [deepcopy(dict(s)) for s in slices if isinstance(s, dict)]

    def _rebuild_surface_caches(
        self,
        *,
        prev_local_trees: dict[str, Any] | None = None,
        prev_slice_trees: dict[str, Any] | None = None,
    ) -> None:
        """为场内每个活动物体在其自身局部坐标系下构建一次 `LocalSurfaceKDTree`，持握随动物体零重建。"""
        self._obj_surface_points = {}
        self._obj_slice_points = {}
        self._obj_local_trees = {}
        self._obj_slice_trees = {}
        self._initial_obb_poses = {}
        prev_local_trees = prev_local_trees or {}
        prev_slice_trees = prev_slice_trees or {}

        cd = self._cloud_data
        has_cloud = cd is not None and len(cd["xyz"]) > 0
        if has_cloud:
            xyz = cd["xyz"]
            valid = (~cd["robot"]) & (cd["conf"] >= 0.05)
            obj_ids = cd["obj_id"]
            has_instance_labels = bool(np.any(obj_ids > 0))
        else:
            xyz = np.empty((0, 3), dtype=np.float64)
            valid = np.empty(0, dtype=bool)
            obj_ids = np.empty(0, dtype=np.int16)
            has_instance_labels = False

        valid_xyz = xyz[valid] if has_cloud else xyz
        valid_oids = obj_ids[valid] if has_cloud else obj_ids

        for oid, obj in self.objects.items():
            if obj.held_by_arm is not None:
                base_id = oid
                for prefix in ("held_left_", "held_right_"):
                    if base_id.startswith(prefix):
                        base_id = base_id[len(prefix):]
                kept_tree = prev_local_trees.get(oid) or prev_local_trees.get(base_id)
                cur_obb = obj.obb()
                self._initial_obb_poses[oid] = (cur_obb.center.copy(), cur_obb.axes.copy())
                if kept_tree is not None:
                    self._obj_local_trees[oid] = kept_tree
                    self._obj_surface_points[oid] = kept_tree.world_points(cur_obb)
                else:
                    kept_slice = prev_slice_trees.get(oid) or prev_slice_trees.get(base_id)
                    if kept_slice is not None:
                        self._obj_slice_trees[oid] = kept_slice
                        self._obj_slice_points[oid] = kept_slice.world_points(cur_obb)
                continue

            obb = obj.obb()
            pts = np.empty((0, 3), dtype=np.float64)
            if has_cloud and len(valid_xyz) > 0:
                t_num = _parse_track_num(oid)
                if has_instance_labels and t_num is not None:
                    mask_k = valid & (obj_ids == t_num)
                    if np.any(mask_k):
                        cand = xyz[mask_k]
                        rel = (cand - obb.center[None, :]) @ obb.axes
                        keep = np.all(np.abs(rel) <= obb.half_extents[None, :] + 1e-5, axis=1)
                        pts = cand[keep]

                if len(pts) < MIN_OBJECT_SURFACE_POINTS:
                    cand_pool = valid_xyz[valid_oids == 0] if has_instance_labels else valid_xyz
                    if len(cand_pool) > 0:
                        rel_all = (cand_pool - obb.center[None, :]) @ obb.axes
                        in_obb = np.all(np.abs(rel_all) <= obb.half_extents[None, :] + 1e-5, axis=1)
                        if self.table_z is not None:
                            in_obb &= cand_pool[:, 2] >= float(self.table_z) + 0.003
                        box_pts = cand_pool[in_obb]
                        if len(box_pts) > len(pts):
                            pts = box_pts

            if len(pts) >= MIN_OBJECT_SURFACE_POINTS:
                if len(pts) > MAX_CACHED_OBJECT_POINTS:
                    stride = int(math.ceil(len(pts) / float(MAX_CACHED_OBJECT_POINTS)))
                    pts = pts[::stride]
                pts_contig = np.ascontiguousarray(pts, dtype=np.float64)
                obj.refine_tilt_from_points(pts_contig)
                obb = obj.obb()
                self._initial_obb_poses[oid] = (obb.center.copy(), obb.axes.copy())
                self._obj_surface_points[oid] = pts_contig
                self._obj_local_trees[oid] = LocalSurfaceKDTree(pts_contig, obb.center, obb.axes)
                meta = derive_object_spatial_meta(
                    float(obj.length),
                    float(obj.width),
                    float(obj.top_z),
                    float(obj.bottom_z),
                    surface_points_m=pts_contig,
                )
                if _num(meta.get("uncertainty_m")):
                    obj.uncertainty_m = meta["uncertainty_m"]
            else:
                self._initial_obb_poses[oid] = (obb.center.copy(), obb.axes.copy())
                if oid in self.surface_slices_m:
                    s_pts = sample_slice_hull_points(
                        self.surface_slices_m[oid],
                        float(obj.bottom_z),
                        float(obj.top_z),
                        cavity_depth_m=self.cavity_depths_m.get(oid, 0.0),
                    )
                    if len(s_pts) >= MIN_OBJECT_SURFACE_POINTS:
                        self._obj_slice_points[oid] = s_pts
                        self._obj_slice_trees[oid] = LocalSurfaceKDTree(s_pts, obb.center, obb.axes)

    def update_from_observation(
        self,
        object_table=None,
        robot_state: dict | None = None,
        cloud=None,
        *,
        cloud_npz=None,
        cavity_depths_m: dict[str, float] | None = None,
        surface_slices_m: dict[str, list[dict]] | None = None,
        raw_objects: list[dict] | None = None,
    ) -> "PointCloudSpatialField":
        """融合最新 `object_table`、实测双臂状态与三维点云更新空间场。"""
        prev_cavities = dict(self.cavity_depths_m)
        prev_slices = dict(self.surface_slices_m)
        prev_local_trees = dict(self._obj_local_trees)
        prev_slice_trees = dict(self._obj_slice_trees)

        effective_cloud = cloud if cloud is not None else cloud_npz
        if effective_cloud is not None:
            coerced = _coerce_cloud_data(effective_cloud)
            if coerced is not None:
                self._cloud_data = coerced
                if coerced.get("table_z") is not None and object_table is None:
                    self.table_z = float(coerced["table_z"])

        super().update_from_observation(object_table=object_table, robot_state=robot_state, cloud=effective_cloud)
        self._ingest_geometry_metadata(
            object_table,
            cavity_depths_m=cavity_depths_m,
            surface_slices_m=surface_slices_m,
            raw_objects=raw_objects,
        )
        for oid, obj in self.objects.items():
            if obj.held_by_arm is not None:
                base_id = oid
                for prefix in ("held_left_", "held_right_"):
                    if base_id.startswith(prefix):
                        base_id = base_id[len(prefix):]
                if oid not in self.cavity_depths_m and base_id in prev_cavities:
                    self.cavity_depths_m[oid] = prev_cavities[base_id]
                if oid not in self.surface_slices_m and base_id in prev_slices:
                    self.surface_slices_m[oid] = prev_slices[base_id]

        self._rebuild_surface_caches(
            prev_local_trees=prev_local_trees if effective_cloud is None else {
                k: v for k, v in prev_local_trees.items()
                if (k in self.objects and self.objects[k].held_by_arm is not None)
            },
            prev_slice_trees=prev_slice_trees,
        )
        return self

    def has_fine_geometry(self, oid: str | None = None) -> bool:
        """判断指定物体（或场内任意物体）是否具备实测局部系表面树或实测分层凸包切片几何。"""
        return _oracle_has_fine_geometry(self, oid)

    def object_surface_points(self, oid: str) -> np.ndarray | None:
        """返回物体 `oid` 当前位姿下的真实非机械臂三维表面点集 `(M, 3)`（单位 `m`）；不足 8 点返回 `None`。"""
        return _oracle_object_surface_points(self, oid)

    def _fine_surface_points(self, oid: str) -> np.ndarray | None:
        """返回物体 `oid` 用于局部接触截面计算的三维点集（优先真实点云，其次分层凸包切片采样点），单位 `m`。"""
        return _oracle_fine_surface_points(self, oid)

    def object_signed_distance(self, query_obb: Any, obj: Any) -> float:
        """计算运动部件 `query_obb` 到环境物体 `obj` 的三维有符号距离（单位 `m`，兼容双向参数顺序）。"""
        return _oracle_object_signed_distance(self, query_obb, obj)

    def _object_cavity_floor_z(self, obj: Any) -> float | None:
        """估计凹腔容器物体的内底支撑高程 `floor_z_m`（单位 `m`）；非凹腔物体返回 `None`。"""
        return _oracle_cavity_floor_z(self, obj)

    def evaluate_grasp(
        self,
        arm: str = "right",
        position_m=None,
        angles_deg=None,
        target_id: str | dict | None = None,
        open_aperture_m: float | None = None,
        contact_region: str = "pad_center",
        contact_depth_m: float | None = None,
    ) -> dict:
        """评估指定三维抓取位姿在真实三维表面点云/凹腔几何下的客观可夹持性与邻物净空（委托 `clearance_oracle`）。"""
        return evaluate_grasp_clearance(
            self,
            arm=arm,
            position_m=position_m,
            angles_deg=angles_deg,
            target_id=target_id,
            open_aperture_m=open_aperture_m,
            contact_region=contact_region,
            contact_depth_m=contact_depth_m,
        )

    def simulate_stages(
        self,
        stages,
        step_m: float = 0.015,
        *,
        allow_contact_ids: list[str] | tuple[str, ...] | set[str] | None = None,
        ignore_object_ids: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> dict:
        """在点云空间场克隆副本上动态推演多阶段动作序列的客观路径净空（委托 `clearance_oracle`）。"""
        return simulate_stage_clearances(
            self,
            stages,
            step_m=step_m,
            allow_contact_ids=allow_contact_ids,
            ignore_object_ids=ignore_object_ids,
        )

    def inspect_cavity(self, target_id_or_region, radius_m: float = 0.06, **kwargs) -> dict:
        """测量目标物体或局部三维区域的凹腔几何剖面（委托 `clearance_oracle.inspect_cavity_geometry`）。"""
        return inspect_cavity_geometry(self, target_id_or_region, radius_m=radius_m, **kwargs)
