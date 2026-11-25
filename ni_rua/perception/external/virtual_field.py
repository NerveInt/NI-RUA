"""虚拟空间状态容器 (`VirtualSpatialField`) 与无状态几何/评估门面。

所有 OBB/SDF 基元收敛至 `geometry_ops.py`，抓取净空评估与多阶段轨迹推演收敛至 `clearance_oracle.py`；
彻底移除策略求解器 (`solve_optimal_grasp` / `plan_pick_and_place` / `optimal_grasp_pose`)。
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import numpy as np

if __package__:
    from .pose_math import (
        MAX_OPENING_M,
        angles_deg_to_quat_wxyz,
        flange_pose_to_grasp_pose,
        gripper_frame,
        quat_wxyz_to_angles_deg,
        quat_wxyz_to_matrix,
    )
    from .geometry_ops import OBB3D, _num, _r, obb_signed_distance
    from .clearance_oracle import (
        DEFAULT_ARM_POSES,
        DEFAULT_JAW_GEOMETRY,
        ParallelJawGeometry,
        VirtualObject,
        _extract_measured_support_z,
        _extract_stage_spec as _oracle_extract_stage_spec,
        _parse_virtual_object,
        evaluate_grasp_clearance,
        gripper_obbs,
        simulate_stage_clearances,
    )
else:
    from pose_math import (  # type: ignore[no-redef]
        MAX_OPENING_M,
        angles_deg_to_quat_wxyz,
        flange_pose_to_grasp_pose,
        gripper_frame,
        quat_wxyz_to_angles_deg,
        quat_wxyz_to_matrix,
    )
    from geometry_ops import OBB3D, _num, _r, obb_signed_distance  # type: ignore[no-redef]
    from clearance_oracle import (  # type: ignore[no-redef]
        DEFAULT_ARM_POSES,
        DEFAULT_JAW_GEOMETRY,
        ParallelJawGeometry,
        VirtualObject,
        _extract_measured_support_z,
        _extract_stage_spec as _oracle_extract_stage_spec,
        _parse_virtual_object,
        evaluate_grasp_clearance,
        gripper_obbs,
        simulate_stage_clearances,
    )


class VirtualSpatialField:
    """基于 `object_table` 与实测机械臂位姿的三维虚拟空间状态容器。"""

    def __init__(
        self,
        objects=None,
        cloud=None,
        table_z: float | None = None,
        robot_state: dict | None = None,
        *,
        max_occluded_steps: int = 5,
        profile: Any = None,
        jaw_geometry: Any = None,
    ):
        self.profile = profile
        resolved_jg = jaw_geometry if jaw_geometry is not None else ParallelJawGeometry.from_profile(profile)
        self.supported: bool = resolved_jg is not None
        self.jaw_geometry = resolved_jg if resolved_jg is not None else DEFAULT_JAW_GEOMETRY
        self.table_z: float | None = float(table_z) if _num(table_z) else None
        self.cloud = cloud
        if cloud is not None and hasattr(cloud, "table") and self.table_z is None and _num(getattr(cloud, "table", None)):
            self.table_z = float(cloud.table)
        self.objects: dict[str, Any] = {}
        self.historical_unobserved_objects: dict[str, Any] = {}
        arms = self.jaw_geometry.arms if self.jaw_geometry is not None else ("left", "right")
        max_open = self.jaw_geometry.max_opening_m if self.jaw_geometry is not None else MAX_OPENING_M
        self.arm_poses: dict[str, dict[str, list[float]]] = {
            arm: deepcopy(DEFAULT_ARM_POSES.get(arm, {"position_m": [0.25, -0.25, 0.92], "angles_deg": [0.0, 0.0, 0.0]}))
            for arm in arms
        }
        self.arm_apertures_m: dict[str, float] = {arm: max_open for arm in arms}
        self.arm_jaw_goals: dict[str, float | None] = {arm: None for arm in arms}
        self.max_occluded_steps: int = max(0, int(max_occluded_steps))
        self.observation_id: str | None = None
        self.step_count: int = 0
        if objects is not None or robot_state is not None:
            self.update_from_observation(objects, robot_state=robot_state, cloud=cloud)

    def clone(self) -> "VirtualSpatialField":
        """创建当前虚拟场的深拷贝副本，用于无副作用的推演。"""
        copied = VirtualSpatialField(
            table_z=self.table_z,
            max_occluded_steps=self.max_occluded_steps,
            profile=self.profile,
            jaw_geometry=self.jaw_geometry if self.supported else None,
        )
        copied.supported = self.supported
        copied.observation_id = self.observation_id
        copied.cloud = self.cloud
        copied.objects = deepcopy(self.objects)
        copied.historical_unobserved_objects = deepcopy(self.historical_unobserved_objects)
        copied.arm_poses = deepcopy(self.arm_poses)
        copied.arm_apertures_m = deepcopy(self.arm_apertures_m)
        copied.arm_jaw_goals = dict(self.arm_jaw_goals)
        copied.step_count = self.step_count
        return copied

    def gripper_frame(self, arm: str = "right") -> dict:
        """返回指定机械臂当前位姿的三维夹爪标架字典（`pad_center_m` 单位 `m` 与正交 `axes`）。"""
        pose = self.arm_poses.get(arm) or next(iter(self.arm_poses.values()), DEFAULT_ARM_POSES["right"])
        jg = self.jaw_geometry
        return gripper_frame(
            pose["position_m"],
            pose["angles_deg"],
            grasp_point_offset_m=jg.grasp_point_offset_m,
            fingertip_below_pad_m=jg.fingertip_below_pad_m,
        )

    def _update_robot_state(self, robot_state: dict | None) -> None:
        if not isinstance(robot_state, dict):
            return
        max_open = self.jaw_geometry.max_opening_m
        grasp_off = self.jaw_geometry.grasp_point_offset_m
        arms = robot_state.get("arms") or {}
        for arm in self.jaw_geometry.arms:
            arm_data = arms.get(arm)
            if not isinstance(arm_data, dict):
                continue
            gp = arm_data.get("grasp_pose")
            if isinstance(gp, dict) and isinstance(gp.get("position_m"), (list, tuple)) and len(gp["position_m"]) == 3:
                ang = gp.get("angles_deg") or [0.0, 0.0, 0.0]
                self.arm_poses[arm] = {
                    "position_m": [float(v) for v in gp["position_m"]],
                    "angles_deg": [float(v) for v in ang[:3]],
                }
            else:
                ee = arm_data.get("ee_pose") or {}
                pos = ee.get("position_m")
                quat_xyzw = ee.get("quaternion_xyzw")
                if isinstance(pos, (list, tuple)) and len(pos) == 3 and isinstance(quat_xyzw, (list, tuple)) and len(quat_xyzw) == 4:
                    conv = flange_pose_to_grasp_pose(
                        pos,
                        quat_xyzw,
                        grasp_point_offset_m=grasp_off,
                        fingertip_below_pad_m=self.jaw_geometry.fingertip_below_pad_m,
                        profile=self.profile,
                    )
                    self.arm_poses[arm] = {
                        "position_m": conv["grasp_position_m"],
                        "angles_deg": conv["angles_deg"],
                    }
            grip = arm_data.get("gripper") or arm_data.get("ee_joint_state") or {}
            if isinstance(grip, dict):
                if _num(grip.get("goal_normalized")):
                    self.arm_jaw_goals[arm] = float(np.clip(grip["goal_normalized"], 0.0, 1.0))
                elif _num(grip.get("command_normalized")):
                    self.arm_jaw_goals[arm] = float(np.clip(grip["command_normalized"], 0.0, 1.0))
                else:
                    self.arm_jaw_goals[arm] = None

                pos_m = grip.get("position_m")
                if isinstance(pos_m, (list, tuple, np.ndarray)) and len(pos_m) >= 2 and _num(pos_m[0]) and _num(pos_m[1]):
                    self.arm_apertures_m[arm] = float(np.clip(float(pos_m[0]) + float(pos_m[1]), 0.0, max_open))
                elif _num(pos_m):
                    self.arm_apertures_m[arm] = float(np.clip(float(pos_m), 0.0, max_open))
                elif _num(grip.get("measured_aperture_m")):
                    self.arm_apertures_m[arm] = float(np.clip(grip["measured_aperture_m"], 0.0, max_open))
                elif _num(grip.get("aperture_m")):
                    self.arm_apertures_m[arm] = float(np.clip(grip["aperture_m"], 0.0, max_open))
                elif _num(grip.get("position_normalized")):
                    self.arm_apertures_m[arm] = float(np.clip(grip["position_normalized"], 0.0, 1.0)) * max_open
                elif _num(grip.get("command_normalized")):
                    self.arm_apertures_m[arm] = float(np.clip(grip["command_normalized"], 0.0, 1.0)) * max_open
                elif _num(grip.get("goal_normalized")):
                    self.arm_apertures_m[arm] = float(np.clip(grip["goal_normalized"], 0.0, 1.0)) * max_open
            elif _num(arm_data.get("aperture_m")):
                self.arm_apertures_m[arm] = float(np.clip(arm_data["aperture_m"], 0.0, max_open))

    def update_from_observation(self, object_table=None, robot_state: dict | None = None, cloud=None) -> "VirtualSpatialField":
        """随真实机械臂移动与新观测实时融合更新虚拟空间场。"""
        if cloud is not None:
            self.cloud = cloud
            if hasattr(cloud, "table") and _num(getattr(cloud, "table", None)):
                self.table_z = float(cloud.table)
        self._update_robot_state(robot_state)

        prev_objects = self.objects
        prev_historical = self.historical_unobserved_objects
        valid_arms = set(self.jaw_geometry.arms) | {"left", "right"}
        prev_held = {
            arm: obj
            for obj in prev_objects.values()
            if obj.held_by_arm in valid_arms
            for arm in (obj.held_by_arm,)
        }

        columns = None
        raw_items = []
        spatial_meta_map = {}
        if isinstance(object_table, dict):
            measured_sup = _extract_measured_support_z(object_table)
            if measured_sup is not None:
                self.table_z = measured_sup
            columns = object_table.get("columns")
            raw_items = list(object_table.get("rows") or object_table.get("objects") or [])
            if isinstance(object_table.get("spatial_meta"), dict):
                spatial_meta_map = object_table["spatial_meta"]
        elif isinstance(object_table, (list, tuple)):
            raw_items = list(object_table)

        new_objects: dict[str, Any] = {}
        for idx, item in enumerate(raw_items):
            parsed = _parse_virtual_object(item, index=idx, table_z=self.table_z, columns=columns)
            if parsed is not None:
                sm = spatial_meta_map.get(parsed.id)
                if isinstance(sm, dict) and _num(sm.get("uncertainty_m")):
                    parsed.uncertainty_m = round(float(sm["uncertainty_m"]), 4)
                if parsed.held_by_arm in valid_arms:
                    pose = self.arm_poses.get(parsed.held_by_arm)
                    if pose is not None:
                        parsed.attach_to_gripper(
                            parsed.held_by_arm,
                            self.gripper_frame(parsed.held_by_arm),
                        )
                new_objects[parsed.id] = parsed

        max_open = self.jaw_geometry.max_opening_m
        release_thresh = min(0.075, 0.85 * max_open)
        for arm, held_obj in prev_held.items():
            ap = self.arm_apertures_m.get(arm, max_open)
            jaw_goal = self.arm_jaw_goals.get(arm)
            jaw_releasing = (ap >= release_thresh) or (jaw_goal is not None and jaw_goal >= 0.5 and ap >= 0.020)
            if not jaw_releasing and not any(o.held_by_arm == arm for o in new_objects.values()):
                pose = self.arm_poses.get(arm)
                if pose is not None:
                    g_frame = self.gripper_frame(arm)
                    held_copy = deepcopy(held_obj)
                    held_copy.update_from_gripper(g_frame)
                    key = held_copy.id if held_copy.id not in new_objects else f"held_{arm}_{held_copy.id}"
                    held_copy.id = key
                    new_objects[key] = held_copy

        if object_table is not None:
            new_historical: dict[str, Any] = {}
            if self.max_occluded_steps > 0:
                seen_hist_ids: set[str] = set()
                for source_dict in (prev_objects, prev_historical):
                    for old_obj in source_dict.values():
                        if old_obj.id in seen_hist_ids:
                            continue
                        seen_hist_ids.add(old_obj.id)
                        if old_obj.held_by_arm is not None or int(old_obj.occluded_steps) >= self.max_occluded_steps:
                            continue
                        ox, oy = old_obj.centre_m[0], old_obj.centre_m[1]
                        matched = (old_obj.id in new_objects) or any(
                            math.hypot(ox - cur.centre_m[0], oy - cur.centre_m[1]) <= max(0.5 * old_obj.length, 0.045)
                            for cur in new_objects.values()
                        )
                        if not matched:
                            hist = deepcopy(old_obj)
                            hist.occluded = True
                            hist.occluded_steps = int(old_obj.occluded_steps) + 1
                            new_historical[hist.id] = hist
                if isinstance(object_table, dict):
                    raw_hist = object_table.get("historical_unobserved_objects")
                    hist_iter = raw_hist.values() if isinstance(raw_hist, dict) else (raw_hist or ())
                    if isinstance(raw_hist, (list, tuple, dict)):
                        for h_idx, h_item in enumerate(hist_iter):
                            h_obj = _parse_virtual_object(h_item, index=h_idx, table_z=self.table_z)
                            if (
                                h_obj is not None
                                and h_obj.id not in new_objects
                                and h_obj.id not in new_historical
                                and int(h_obj.occluded_steps) <= self.max_occluded_steps
                            ):
                                h_obj.occluded = True
                                h_obj.occluded_steps = max(1, int(h_obj.occluded_steps))
                                new_historical[h_obj.id] = h_obj
            self.objects = new_objects
            self.historical_unobserved_objects = new_historical
        elif prev_held:
            for key, obj in new_objects.items():
                self.objects[key] = obj

        self.step_count += 1
        return self

    def find_target_object(self, target_id: str | dict | Any | None, position_m=None):
        """根据 `target_id` 或空间坐标查找目标 `VirtualObject`（显式缺失 ID 绝不静默回退坐标匹配）。"""
        if isinstance(target_id, VirtualObject):
            return target_id
        if isinstance(target_id, dict):
            return _parse_virtual_object(target_id, table_z=self.table_z)
        if isinstance(target_id, str):
            return self.objects.get(target_id)
        if target_id is None and position_m is not None:
            gx, gy = float(position_m[0]), float(position_m[1])
            best_obj, best_dist = None, math.inf
            for obj in self.objects.values():
                if obj.held_by_arm is not None:
                    continue
                dist = math.hypot(gx - obj.centre_m[0], gy - obj.centre_m[1])
                reach = max(0.5 * obj.length + 0.04, 0.5 * obj.width + 0.04, 0.075)
                if dist <= reach and dist < best_dist:
                    best_obj, best_dist = obj, dist
            return best_obj
        return None

    @staticmethod
    def _extract_stage_spec(
        stage: dict,
        default_poses: dict,
        default_apertures: dict,
        *,
        jaw_geometry: ParallelJawGeometry | None = None,
    ) -> dict:
        """委托 `clearance_oracle._extract_stage_spec` 解析阶段规格。"""
        return _oracle_extract_stage_spec(stage, default_poses, default_apertures, jaw_geometry=jaw_geometry)

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
        """评估指定三维抓取位姿的客观几何可夹持性、咬合高度与周围障碍物净空（委托 `clearance_oracle`）。"""
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
        """在虚拟场克隆副本上动态推演多阶段动作序列的客观路径净空（委托 `clearance_oracle`）。"""
        return simulate_stage_clearances(
            self,
            stages,
            step_m=step_m,
            allow_contact_ids=allow_contact_ids,
            ignore_object_ids=ignore_object_ids,
        )
