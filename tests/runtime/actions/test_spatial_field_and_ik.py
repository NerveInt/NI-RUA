"""
- test_virtual_spatial_field.py: 基于 SAM + VGGT 与机械臂物理量的虚拟空间数值模拟场 (VirtualSpatialField) 及局部 KDTree 单元测试。
- test_s2_field_ops_and_ik_prefilter.py: 三维刚体李群算子 (`field_ops.py`)、动态 6R 解析运动学核 (`ik_prefilter.py`) 与沙盒挂载契约单元测试。
"""
from __future__ import annotations

from copy import deepcopy
import io
import math
import time
from types import SimpleNamespace
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from ni_rua.perception.external import clearance_oracle, geometry_ops, object_table, obs3d, scene3d_helpers as s3, vectorized_grasp as vg_mod
from ni_rua.perception.external.local_kdtree_field import (
    LocalSurfaceKDTree,
    derive_object_spatial_meta,
    derive_table_spatial_meta,
)
from ni_rua.perception.external.pointcloud_field import (
    PointCloudSpatialField,
    obb_to_points_signed_distance,
)
from ni_rua.perception.external.virtual_field import (
    OBB3D,
    VirtualSpatialField,
    gripper_frame,
    gripper_obbs,
    obb_signed_distance,
)
from ni_rua.runtime.actions.grasp_preview_3d import audit_batch_grasp_3d
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest.mock
from ni_rua.core.config import Config
from ni_rua.core.contracts import canonical
from ni_rua.core.storage.store import Store
from ni_rua.perception.external import field_ops as fo
from ni_rua.perception.external import ik_prefilter as ikp
from ni_rua.perception.external import pose_math as pm
from ni_rua.perception.external import scene3d
from ni_rua.perception.external import scene3d_helpers as s3
from ni_rua.perception.external.virtual_field import (
    angles_deg_to_quat_wxyz,
    gripper_frame,
    quat_wxyz_to_matrix,
)
from ni_rua.robot.model import geometry
from ni_rua.robot.model.roboprobe_pose import quat_to_matrix
from ni_rua.runtime.actions.grasp_preview_3d import _extract_closing_grasp_stages
from ni_rua.runtime.actions.robot_action_schema import ROBOT_ACTION_TOOL, _validate_stage
from ni_rua.runtime.harness import Harness, _action_focus_roi
from ni_rua.runtime.programs.runner import ProgramRunner
from support.geometry_fixture import IDENTITY
from support.tool_fixture import install_registry
from support.fixture_robot import CalibratedThreeCamRobot
from support.host_capabilities import expect, program_isolation

NI_RUA_TEST_TIER = "component"


# --- Folded from tests/test_virtual_spatial_field.py ---

COLUMNS = ["id", "centre_m", "top_z", "bottom_z", "length", "width", "yaw_deg", "tilt_deg", "support", "quality"]


class VirtualSpatialFieldTest(unittest.TestCase):
    def setUp(self):
        self.base_table = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[
                ["o1", [0.10, -0.20, 0.780], 0.795, 0.765, 0.140, 0.026, 30.0, 88.0, "table", "ok"],
            ],
        )

    def test_obb_signed_distance_and_gripper_frame_alignment(self):
        """验证 3D OBB 有符号距离（分离正距离、穿透负深度）与夹爪三维标架约定。"""
        axes = np.eye(3)
        box_a = OBB3D([0.0, 0.0, 0.0], axes, [0.02, 0.02, 0.02], label="a")
        box_b_sep = OBB3D([0.06, 0.0, 0.0], axes, [0.02, 0.02, 0.02], label="b")
        box_c_overlap = OBB3D([0.03, 0.0, 0.0], axes, [0.02, 0.02, 0.02], label="c")

        self.assertAlmostEqual(obb_signed_distance(box_a, box_b_sep), 0.02, places=3)
        self.assertAlmostEqual(obb_signed_distance(box_a, box_c_overlap), -0.01, places=3)

        # [0, 0, 0] 竖直向下时，接近轴为 [0, 0, -1]，张合轴为 [1, 0, 0]，指尖低于 pad_center 7.5mm
        gf = gripper_frame([0.10, -0.20, 0.780], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(gf["approach_dir"], [0.0, 0.0, -1.0], atol=1e-6)
        np.testing.assert_allclose(gf["closing_dir"], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(gf["fingertip_center_m"], [0.10, -0.20, 0.7725], atol=1e-6)

    def test_evaluate_grasp_single_and_dense_multi_object_scenarios(self):
        """验证 evaluate_grasp 对正常抓取、高度过高咬合不足、偏心空抓/撞击目标、以及张开双指撞邻近物体的检测。"""
        vf = VirtualSpatialField(objects=self.base_table)

        # 1) 正常横跨长轴对准抓取 (yaw=-60 deg 垂直于 30 deg 主轴，z=0.780)
        ok_eval = vf.evaluate_grasp("right", [0.10, -0.20, 0.780], [0.0, 0.0, -60.0], target_id="o1")
        self.assertTrue(ok_eval["theoretically_graspable"], ok_eval)
        self.assertEqual(ok_eval["failure_reasons"], [])
        self.assertFalse(ok_eval["finger_hits_target"])
        self.assertAlmostEqual(ok_eval["projected_span_m"], 0.026, places=3)
        self.assertGreaterEqual(ok_eval["vertical_bite_m"], 0.014)
        self.assertLessEqual(ok_eval["lateral_eccentricity_m"], 0.002)

        # 2) 悬空过高导致 15mm 夹持面无有效竖直咬合 (z=0.815 > top_z=0.795 + 0.0075)
        high_eval = vf.evaluate_grasp("right", [0.10, -0.20, 0.815], [0.0, 0.0, -60.0], target_id="o1")
        self.assertFalse(high_eval["theoretically_graspable"])
        self.assertIn("insufficient_vertical_bite", high_eval["failure_reasons"])
        self.assertEqual(high_eval["vertical_bite_m"], 0.0)

        # 3) 横向偏心过大导致单侧手指下探直接撞在物体上或偏出包络
        # 沿闭合方向 (-60 deg) 偏移 38mm（超过半开度余量 (88.8 - 26)/2 = 31.4mm）
        rad = np.radians(-60.0)
        ecc_pos = [0.10 + 0.038 * float(np.cos(rad)), -0.20 + 0.038 * float(np.sin(rad)), 0.780]
        ecc_eval = vf.evaluate_grasp("right", ecc_pos, [0.0, 0.0, -60.0], target_id="o1")
        self.assertFalse(ecc_eval["theoretically_graspable"])
        self.assertTrue(
            "excessive_lateral_eccentricity" in ecc_eval["failure_reasons"]
            or "finger_hits_target_on_descent" in ecc_eval["failure_reasons"],
            ecc_eval,
        )

        # 4) 密集多物体场景：在 o1 中心右侧闭合轴方向 46mm 处放置紧邻障碍物 o2（恰好位于张开右指 44.4..52.4mm 下探包络内）
        obs_pos = [0.10 + 0.046 * float(np.cos(rad)), -0.20 + 0.046 * float(np.sin(rad)), 0.785]
        dense_table = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[
                ["o1", [0.10, -0.20, 0.780], 0.795, 0.765, 0.140, 0.026, 30.0, 88.0, "table", "ok"],
                ["o2", obs_pos, 0.810, 0.765, 0.030, 0.030, 0.0, None, "table", "ok"],
            ],
        )
        vf_dense = VirtualSpatialField(objects=dense_table)
        col_eval = vf_dense.evaluate_grasp("right", [0.10, -0.20, 0.780], [0.0, 0.0, -60.0], target_id="o1")
        self.assertFalse(col_eval["theoretically_graspable"])
        self.assertIn("open_finger_collides_with_neighbor", col_eval["failure_reasons"])
        self.assertIn("o2", col_eval["colliding_neighbor_ids"])
        self.assertLess(col_eval["neighbor_clearance_m"], 0.0)

        # 5) 沿 o1 长轴移开至无碰撞位置时，clearance_oracle 返回正净空且不含任何预设求解 optimal_grasp_pose
        self.assertNotIn("optimal_grasp_pose", col_eval)
        axis_rad = np.radians(30.0)
        shifted_pos = [0.10 - 0.045 * float(np.cos(axis_rad)), -0.20 - 0.045 * float(np.sin(axis_rad)), 0.780]
        clear_eval = vf_dense.evaluate_grasp("right", shifted_pos, [0.0, 0.0, -60.0], target_id="o1")
        self.assertTrue(clear_eval["theoretically_graspable"], clear_eval)
        self.assertEqual(clear_eval["colliding_neighbor_ids"], [])
        self.assertGreater(clear_eval["neighbor_clearance_m"], 0.0)
        self.assertNotIn("optimal_grasp_pose", clear_eval)

    def test_evaluate_grasp_clearance_detects_neighbor_vs_orthogonal_yaw(self):
        """当紧凑方块物体的 X 轴方向存在紧邻障碍物时，clearance_oracle 准确区分 yaw=0 碰撞与 yaw=90 净空。"""
        compact_table = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[
                ["o1", [0.00, -0.20, 0.780], 0.795, 0.765, 0.040, 0.038, 0.0, None, "table", "ok"],
                # o2 紧贴在 o1 的 +X 侧 (x=0.048)，若以 yaw=0 (沿 X 轴张合) 抓取，右指必撞 o2
                ["o2", [0.048, -0.20, 0.785], 0.810, 0.765, 0.035, 0.035, 0.0, None, "table", "ok"],
            ],
        )
        vf = VirtualSpatialField(objects=compact_table)
        bad = vf.evaluate_grasp("right", [0.00, -0.20, 0.780], [0.0, 0.0, 0.0], target_id="o1")
        self.assertFalse(bad["theoretically_graspable"])
        self.assertIn("o2", bad["colliding_neighbor_ids"])
        self.assertLess(bad["neighbor_clearance_m"], 0.0)
        self.assertNotIn("optimal_grasp_pose", bad)

        good = vf.evaluate_grasp("right", [0.00, -0.20, 0.780], [0.0, 0.0, 90.0], target_id="o1")
        self.assertTrue(good["theoretically_graspable"], good)
        self.assertEqual(good["colliding_neighbor_ids"], [])
        self.assertGreater(good["neighbor_clearance_m"], 0.01)
        self.assertNotIn("optimal_grasp_pose", good)

    def test_simulate_stages_tracks_held_object_detects_path_collision_and_updates_world(self):
        """验证 simulate_stages 多阶段动态推演：抓起绑定 -> 搬运途中碰撞高障碍物预警 -> 抬高后干净通过并更新释放坐标。"""
        scene_table = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[
                # 待抓取物 o1 位于 x=0.0, y=-0.20
                ["o1", [0.00, -0.20, 0.780], 0.795, 0.765, 0.050, 0.030, 0.0, 88.0, "table", "ok"],
                # 中途高耸障碍物 o2 位于 x=0.10, y=-0.20，顶面高达 0.880m
                ["o2", [0.10, -0.20, 0.822], 0.880, 0.765, 0.060, 0.060, 0.0, None, "table", "ok"],
                # 目标容器/垫块 o3 位于 x=0.20, y=-0.20，顶面 0.790m
                ["o3", [0.20, -0.20, 0.778], 0.790, 0.765, 0.080, 0.080, 0.0, None, "table", "ok"],
            ],
        )
        vf = VirtualSpatialField(objects=scene_table)

        # 低空搬运 (z=0.830m < o2.top_z=0.880m) 必在 transport 阶段撞击 o2（且绝不合成 recommended_safe_z_m / suggested_fixes）
        low_stages = [
            {"id": "grasp", "arm": "right", "kind": "move", "grasp_position_m": [0.00, -0.20, 0.780], "angles_deg": [0.0, 0.0, 90.0], "gripper": 0.0},
            {"id": "lift_low", "arm": "right", "kind": "move", "grasp_position_m": [0.00, -0.20, 0.830], "angles_deg": [0.0, 0.0, 90.0]},
            {"id": "transport_low", "arm": "right", "kind": "move", "grasp_position_m": [0.20, -0.20, 0.830], "angles_deg": [0.0, 0.0, 90.0]},
            {"id": "release", "arm": "right", "kind": "move", "grasp_position_m": [0.20, -0.20, 0.810], "angles_deg": [0.0, 0.0, 90.0], "gripper": 1.0},
        ]
        sim_low = vf.simulate_stages(low_stages)
        self.assertEqual(sim_low["status"], "risk_detected")
        self.assertFalse(sim_low["theoretically_valid"])
        self.assertNotIn("suggested_fixes", sim_low)
        transport_rep = sim_low["stage_reports"][2]
        self.assertEqual(transport_rep["status"], "path_collision")
        self.assertIn("o2", transport_rep["colliding_objects"])
        self.assertNotIn("recommended_safe_z_m", transport_rep)

        # 显式抬升至安全搬运高度 z=0.950m：无碰撞通过，释放后 o1 落位到 [0.20, -0.20] 且支撑在 o3 上
        high_stages = [
            {"id": "pre_grasp", "arm": "right", "kind": "move", "grasp_position_m": [0.00, -0.20, 0.950], "angles_deg": [0.0, 0.0, 90.0], "gripper_goal_normalized": 1.0},
            {"id": "grasp", "arm": "right", "kind": "move", "grasp_position_m": [0.00, -0.20, 0.780], "angles_deg": [0.0, 0.0, 90.0], "gripper_goal_normalized": 0.0},
            {"id": "lift", "arm": "right", "kind": "move", "grasp_position_m": [0.00, -0.20, 0.950], "angles_deg": [0.0, 0.0, 90.0]},
            {"id": "pre_place", "arm": "right", "kind": "move", "grasp_position_m": [0.20, -0.20, 0.950], "angles_deg": [0.0, 0.0, 90.0]},
            {"id": "place", "arm": "right", "kind": "move", "grasp_position_m": [0.20, -0.20, 0.808], "angles_deg": [0.0, 0.0, 90.0], "gripper_goal_normalized": 1.0},
            {"id": "retreat", "arm": "right", "kind": "move", "grasp_position_m": [0.20, -0.20, 0.950], "angles_deg": [0.0, 0.0, 90.0]},
        ]
        sim_clean = vf.simulate_stages(high_stages)
        self.assertEqual(sim_clean["status"], "clean", sim_clean)
        self.assertTrue(sim_clean["theoretically_valid"])

        world_after = {item["id"]: item for item in sim_clean["virtual_world_after"]}
        self.assertAlmostEqual(world_after["o1"]["centre_m"][0], 0.20, places=3)
        self.assertAlmostEqual(world_after["o1"]["centre_m"][1], -0.20, places=3)
        self.assertEqual(world_after["o1"]["support"], "o3")
        self.assertTrue(world_after["o1"]["virtual_moved"])
        self.assertIsNone(world_after["o1"]["held_by_arm"])

    def test_update_from_observation_realtime_iteration_with_arm_motion_and_occlusion(self):
        """验证随真实机械臂移动产生新观测时，虚拟场实时更新持物位姿并保留单帧被遮挡物体。"""
        vf = VirtualSpatialField(objects=self.base_table)
        # 模拟右臂在 [0.10, -0.20, 0.780] 抓取了 o1
        gf0 = gripper_frame([0.10, -0.20, 0.780], [0.0, 0.0, -60.0])
        vf.objects["o1"].attach_to_gripper("right", gf0)
        vf.arm_apertures_m["right"] = 0.025

        # 下一帧新观测：右臂移动到 [0.18, -0.12, 0.900] 且旋转到 yaw=30 deg（夹爪仍闭合 aperture=0.025m），
        # 同时桌面上原本在 [-0.15, -0.22] 的物体 o2 在下一帧被遮挡消失
        vf.objects["o2"] = vf.find_target_object(
            dict(id="o2", centre_m=[-0.15, -0.22, 0.780], top_z=0.795, bottom_z=0.765, length=0.04, width=0.04, yaw_deg=0.0)
        )
        next_table = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[
                ["o3", [0.22, -0.25, 0.780], 0.795, 0.765, 0.05, 0.05, 0.0, None, "table", "ok"],
            ],
        )
        next_robot_state = {
            "arms": {
                "right": {
                    "grasp_pose": {"position_m": [0.18, -0.12, 0.900], "angles_deg": [0.0, 0.0, 30.0]},
                    "gripper": {"aperture_m": 0.025},
                }
            }
        }
        vf.update_from_observation(next_table, robot_state=next_robot_state)

        # 1) 持物 o1 随右臂实测位姿实时更新到 [0.18, -0.12, 0.900]
        self.assertIn("o1", vf.objects)
        self.assertEqual(vf.objects["o1"].held_by_arm, "right")
        np.testing.assert_allclose(vf.objects["o1"].centre_m, [0.18, -0.12, 0.900], atol=1e-3)
        # 2) 未被观测且未被持握的桌面物体 o2 绝不留在活动碰撞表 objects 中，仅记入只读 historical_unobserved_objects
        self.assertNotIn("o2", vf.objects)
        self.assertIn("o2", vf.historical_unobserved_objects)
        self.assertTrue(vf.historical_unobserved_objects["o2"].occluded)
        # 3) 新观测中的 o3 正常加入
        self.assertIn("o3", vf.objects)

    def test_integration_with_grasp_hints_grasp_preview_3d_and_clearance_oracle(self):
        """验证虚拟场、object_table.grasp_hints_from、grasp_preview_3d 与 geometry_ops/clearance_oracle 的客观契约集成。"""
        # 1) object_table.grasp_hints_from 仅保留纯几何提示，不再预填泄漏 safe_grasp_pose；geometry_ops + clearance_oracle 客观评估
        raw_objs = [
            dict(centre_m=[0.10, -0.20, 0.780], top_z_m=0.795, bottom_z_m=0.765, length_m=0.140, width_m=0.026, yaw_deg=30.0, axis_tilt_deg=88.0, elongation=2.5),
        ]
        hints = object_table.grasp_hints_from(raw_objs, support_z_m=0.765)
        self.assertNotIn("safe_grasp_pose", hints["o1"])
        self.assertEqual(hints["o1"]["across_yaw_deg"], [-60.0, 120.0])
        vf_single = VirtualSpatialField(
            objects=dict(columns=COLUMNS, table_z=0.765, rows=[["o1", [0.10, -0.20, 0.780], 0.795, 0.765, 0.140, 0.026, 30.0, 88.0, "table", "ok"]]),
        )
        ev_single = clearance_oracle.evaluate_grasp_clearance(vf_single, "right", [0.10, -0.20, 0.780], [0.0, 0.0, -60.0], target_id="o1")
        self.assertTrue(ev_single["theoretically_graspable"])
        self.assertNotIn("optimal_grasp_pose", ev_single)
        obb_o1 = vf_single.objects["o1"].obb()
        sdf_pts, grad_pts = geometry_ops.obb_points_sdf_and_grad(
            np.array([[0.10, -0.20, 0.805], [0.10, -0.20, 0.780]]),
            obb_o1.center,
            obb_o1.axes,
            obb_o1.half_extents,
        )
        self.assertAlmostEqual(float(sdf_pts[0]), 0.010, places=4)
        np.testing.assert_allclose(grad_pts[0], [0.0, 0.0, 1.0], atol=1e-6)
        self.assertLess(float(sdf_pts[1]), 0.0)

        # 2) grasp_preview_3d 在存在邻近障碍物碰撞时客观输出 colliding_neighbor_ids 与 neighbor_clearance_m（绝不合成 optimal_grasp_pose）
        rad = np.radians(-60.0)
        obs_pos = [0.10 + 0.046 * float(np.cos(rad)), -0.20 + 0.046 * float(np.sin(rad)), 0.785]
        crowded_table = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[
                ["o1", [0.10, -0.20, 0.780], 0.795, 0.765, 0.140, 0.026, 30.0, 88.0, "table", "ok"],
                ["o2", obs_pos, 0.815, 0.765, 0.032, 0.032, 0.0, None, "table", "ok"],
            ],
        )
        harness = SimpleNamespace(_obs3d_memo=("obs-crowd", {"object_table": (crowded_table, None, {})}))
        action = {
            "based_on_observation_id": "obs-crowd",
            "stages": [
                {
                    "id": "grasp_crowded",
                    "kind": "move",
                    "arm": "right",
                    "action": {
                        "input_grasp_pose": {
                            "position_m": [0.10, -0.20, 0.780],
                            "angles_deg": [0.0, 0.0, -60.0],
                        },
                        "gripper_events": [{"goal_normalized": 0.0}],
                    },
                }
            ],
        }
        review, _ = audit_batch_grasp_3d(harness, action, "obs-crowd")
        self.assertEqual(review["status"], "warnings_detected")
        self.assertIn("warn_finger_collides_neighbor", review["risk_flags"])
        self.assertIn("virtual_field_simulation", review)
        stage_audit = review["stages"][0]
        self.assertEqual(stage_audit["finger_collision_check"], "warn_finger_collides_neighbor")
        self.assertIn("o2", stage_audit["colliding_neighbor_ids"])
        self.assertNotIn("optimal_grasp_pose", stage_audit)
        for advisory in review["advisory_notes"]:
            self.assertNotIn("optimal_grasp_pose", advisory)
            self.assertNotIn("candidate pose reference", advisory["message"])

        # 3) 显式 PointCloudSpatialField + clearance_oracle.simulate_stage_clearances 直接调用
        vf_pc = PointCloudSpatialField(objects=crowded_table)
        axis_rad = np.radians(30.0)
        clear_pos = [0.10 - 0.045 * float(np.cos(axis_rad)), -0.20 - 0.045 * float(np.sin(axis_rad)), 0.780]
        ev_clear = clearance_oracle.evaluate_grasp_clearance(vf_pc, "right", clear_pos, [0.0, 0.0, -60.0], target_id="o1")
        self.assertTrue(ev_clear["theoretically_graspable"])
        sim_clear = clearance_oracle.simulate_stage_clearances(
            vf_pc,
            [
                {"id": "pre_grasp", "arm": "right", "kind": "move", "grasp_position_m": [clear_pos[0], clear_pos[1], 0.88], "angles_deg": [0.0, 0.0, -60.0], "gripper_goal_normalized": 1.0},
                {"id": "grasp", "arm": "right", "kind": "move", "grasp_position_m": clear_pos, "angles_deg": [0.0, 0.0, -60.0], "gripper_goal_normalized": 0.0},
                {"id": "lift", "arm": "right", "kind": "move", "grasp_position_m": [clear_pos[0], clear_pos[1], 0.88], "angles_deg": [0.0, 0.0, -60.0]},
                {"id": "pre_place", "arm": "right", "kind": "move", "grasp_position_m": [-0.12, -0.20, 0.88], "angles_deg": [0.0, 0.0, -60.0]},
                {"id": "place", "arm": "right", "kind": "move", "grasp_position_m": [-0.12, -0.20, 0.790], "angles_deg": [0.0, 0.0, -60.0], "gripper_goal_normalized": 1.0},
                {"id": "retreat", "arm": "right", "kind": "move", "grasp_position_m": [-0.12, -0.20, 0.88], "angles_deg": [0.0, 0.0, -60.0]},
            ],
        )
        self.assertEqual(len(sim_clear["stage_reports"]), 6)
        self.assertTrue(sim_clear["theoretically_valid"], sim_clear)

    def test_adversarial_counterexamples_from_codex_review(self):
        """针对对抗审查发现的全部 9 类几何、状态与契约反例的回归测试。"""
        from ni_rua.perception.external.virtual_field import (
            angles_deg_to_quat_wxyz,
            flange_pose_to_grasp_pose,
            quat_wxyz_to_matrix,
        )
        from ni_rua.runtime.actions.robot_action_schema import _STAGE_SCHEMAS

        # 反例 1: 86x86mm 不可抓物体不得在 grasp_hints_from 中输出 safe_grasp_pose，且 evaluate_grasp 保留 failure_reasons
        wide_objs = [
            dict(centre_m=[0.0, 0.0, 0.780], top_z_m=0.795, bottom_z_m=0.765, length_m=0.086, width_m=0.086, yaw_deg=0.0),
        ]
        wide_hints = object_table.grasp_hints_from(wide_objs, support_z_m=0.765)
        self.assertNotIn("safe_grasp_pose", wide_hints["o1"])
        vf_wide = VirtualSpatialField(objects=wide_objs, table_z=0.765)
        ev_wide = vf_wide.evaluate_grasp("right", [0.0, 0.0, 0.780], [0.0, 0.0, 0.0], target_id="o1")
        self.assertFalse(ev_wide["theoretically_graspable"])
        self.assertIn("projected_span_exceeds_opening", ev_wide["failure_reasons"])
        self.assertNotIn("optimal_grasp_pose", ev_wide)

        # 反例 2: 符合正式 robot_action schema 属性集且含 gripper_goal_normalized 的公开阶段传入 simulate_stages 能正确抓起并移动物体
        vf = VirtualSpatialField(objects=self.base_table)
        stages = [
            {"id": "pre_grasp", "arm": "right", "kind": "move", "grasp_position_m": [0.10, -0.20, 0.850], "angles_deg": [0.0, 0.0, -60.0], "gripper_goal_normalized": 1.0},
            {"id": "grasp", "arm": "right", "kind": "move", "grasp_position_m": [0.10, -0.20, 0.780], "angles_deg": [0.0, 0.0, -60.0], "gripper_goal_normalized": 0.0},
            {"id": "lift", "arm": "right", "kind": "move", "grasp_position_m": [0.10, -0.20, 0.850], "angles_deg": [0.0, 0.0, -60.0]},
            {"id": "pre_place", "arm": "right", "kind": "move", "grasp_position_m": [0.18, -0.20, 0.850], "angles_deg": [0.0, 0.0, -60.0]},
            {"id": "place", "arm": "right", "kind": "move", "grasp_position_m": [0.18, -0.20, 0.780], "angles_deg": [0.0, 0.0, -60.0], "gripper_goal_normalized": 1.0},
            {"id": "retreat", "arm": "right", "kind": "move", "grasp_position_m": [0.18, -0.20, 0.850], "angles_deg": [0.0, 0.0, -60.0]},
        ]
        allowed_move_keys = set(_STAGE_SCHEMAS["move"]["properties"].keys())
        for st in stages:
            self.assertNotIn("gripper", st)
            self.assertTrue(set(st.keys()).issubset(allowed_move_keys), st.keys())
        sim_schema = vf.simulate_stages(stages)
        self.assertTrue(sim_schema["theoretically_valid"], sim_schema)
        world_after = {o["id"]: o for o in sim_schema["virtual_world_after"]}
        self.assertTrue(world_after["o1"]["virtual_moved"])
        self.assertAlmostEqual(world_after["o1"]["centre_m"][0], 0.18, places=3)

        # 反例 3: 140x26x30mm 刚体在 [0, 45, 0] 倾斜姿态下连续 4 次 update_from_gripper 尺寸严格幂等，绝不膨胀；
        # 且旋转立起释放后保留 3D 姿态
        obj = vf.objects["o1"]
        gf_init = gripper_frame([0.10, -0.20, 0.780], [0.0, 0.0, -60.0])
        obj.attach_to_gripper("right", gf_init)
        gf_tilt = gripper_frame([0.10, -0.20, 0.880], [0.0, 45.0, -60.0])
        heights = []
        for _ in range(4):
            obj.update_from_gripper(gf_tilt)
            heights.append(round(obj.top_z - obj.bottom_z, 6))
        self.assertEqual(heights[0], heights[1])
        self.assertEqual(heights[1], heights[2])
        self.assertEqual(heights[2], heights[3])
        self.assertAlmostEqual(obj.intrinsic_height_m, 0.030, places=6)
        obj.detach_from_gripper(support_z=0.765, support_id="table")
        self.assertAlmostEqual(obj.top_z - obj.bottom_z, heights[0], places=5)
        self.assertAlmostEqual(obj.bottom_z, 0.765, places=6)

        # 反例 4: 30x30x200mm 竖直柱体以 [0, 90, 0] 侧向抓取（闭合轴沿竖直 200mm 方向）被判定为不可抓且手指穿模，
        # 手掌穿入目标本体也被检出
        tall_table = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[
                ["o1", [0.0, 0.0, 0.865], 0.965, 0.765, 0.030, 0.030, 0.0, 2.0, "table", "ok"],
            ],
        )
        vf_tall = VirtualSpatialField(objects=tall_table)
        side_eval = vf_tall.evaluate_grasp("right", [0.0, 0.0, 0.865], [0.0, 90.0, 0.0], target_id="o1")
        self.assertFalse(side_eval["theoretically_graspable"], side_eval)
        self.assertTrue(side_eval["finger_hits_target"])
        self.assertIn("projected_span_exceeds_opening", side_eval["failure_reasons"])
        self.assertIn("finger_hits_target_on_descent", side_eval["failure_reasons"])

        # 下探过深使手掌穿入高柱体顶部 (z=0.865 时手掌中心在 0.865 + 0.0975 = 0.9625，若压低到 z=0.830 则手掌穿入 0.965 以下)
        palm_eval = vf_tall.evaluate_grasp("right", [0.0, 0.0, 0.830], [0.0, 0.0, 0.0], target_id="o1")
        self.assertFalse(palm_eval["theoretically_graspable"])
        self.assertTrue(palm_eval["palm_hits_target"])
        self.assertIn("palm_collides_with_target", palm_eval["failure_reasons"])

        # 反例 5: 原地旋转 0 -> 90 deg 途中 22.5 deg 存在障碍物时被检出 path_collision；下探至桌面以下 z=0.700m 被检出 table 碰撞
        rad22 = math.radians(22.5) if "math" in globals() else np.radians(22.5)
        obs_rot_pos = [0.046 * float(np.cos(rad22)), 0.046 * float(np.sin(rad22)), 0.800]
        rot_table = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[
                ["o_obs", obs_rot_pos, 0.820, 0.765, 0.020, 0.020, 0.0, None, "table", "ok"],
            ],
        )
        vf_rot = VirtualSpatialField(
            objects=rot_table,
            robot_state={"arms": {"right": {"grasp_pose": {"position_m": [0.0, 0.0, 0.790], "angles_deg": [0.0, 0.0, 0.0]}}}},
        )
        sim_rot = vf_rot.simulate_stages([
            {"id": "spin", "arm": "right", "kind": "move", "grasp_position_m": [0.0, 0.0, 0.790], "angles_deg": [0.0, 0.0, 90.0]},
        ])
        self.assertEqual(sim_rot["status"], "risk_detected")
        self.assertIn("o_obs", sim_rot["stage_reports"][0]["colliding_objects"])

        sim_table_hit = vf_rot.simulate_stages([
            {"id": "dive", "arm": "right", "kind": "move", "grasp_position_m": [0.0, -0.20, 0.700], "angles_deg": [0.0, 0.0, 0.0]},
        ])
        self.assertEqual(sim_table_hit["status"], "risk_detected")
        self.assertIn("table", sim_table_hit["stage_reports"][0]["colliding_objects"])

        # 反例 6: 法兰位姿 position_m + quaternion_xyzw 与自定义 contact_depth_m 精确解析
        q_wxyz = angles_deg_to_quat_wxyz(0.0, 0.0, -60.0)
        R_mat = quat_wxyz_to_matrix(q_wxyz)
        flange_pos = np.array([0.10, -0.20, 0.780]) - (0.1501 - 0.05) * R_mat[:, 0]
        quat_xyzw = [float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3]), float(q_wxyz[0])]
        spec = VirtualSpatialField._extract_stage_spec(
            {
                "id": "flange_move",
                "arm": "right",
                "kind": "move",
                "position_m": flange_pos.tolist(),
                "quaternion_xyzw": quat_xyzw,
                "contact_depth_m": 0.05,
            },
            {},
            {},
        )
        np.testing.assert_allclose(spec["position_m"], [0.10, -0.20, 0.780], atol=1e-5)
        np.testing.assert_allclose(spec["angles_deg"], [0.0, 0.0, -60.0], atol=1e-4)
        self.assertAlmostEqual(spec["contact_depth_m"], 0.05, places=5)

        # 反例 7: 真实 RoboDojo gripper: {"position_m": [0.005, 0.005]} 被正确解析为 0.010m 开度
        vf_robo = VirtualSpatialField(
            objects=self.base_table,
            robot_state={
                "arms": {
                    "right": {
                        "grasp_pose": {"position_m": [0.10, -0.20, 0.780], "angles_deg": [0.0, 0.0, -60.0]},
                        "gripper": {"position_m": [0.005, 0.005], "goal_normalized": 0.0},
                    }
                }
            },
        )
        self.assertAlmostEqual(vf_robo.arm_apertures_m["right"], 0.010, places=5)

        # 反例 8: 斜交分离 OBB 的 obb_signed_distance 与网格采样精确欧氏距离一致
        r1 = quat_wxyz_to_matrix(angles_deg_to_quat_wxyz(25.0, -35.0, 40.0))
        r2 = quat_wxyz_to_matrix(angles_deg_to_quat_wxyz(-50.0, 20.0, -70.0))
        b1 = OBB3D([0.0, 0.0, 0.0], r1, [0.04, 0.015, 0.025])
        b2 = OBB3D([0.055, 0.032, 0.018], r2, [0.035, 0.012, 0.020])
        d_obb = obb_signed_distance(b1, b2)
        # 在 b1 表面密集采样并 clamp 到 b2 验证欧氏距离
        grid = np.linspace(-1.0, 1.0, 25)
        u, v, w = np.meshgrid(grid, grid, grid, indexing="ij")
        pts1 = b1.center[None, :] + (np.column_stack([u.ravel(), v.ravel(), w.ravel()]) * b1.half_extents[None, :]) @ b1.axes.T
        clamped2 = b2.clamp_points(pts1)
        clamped1 = b1.clamp_points(clamped2)
        d_ref = float(np.min(np.linalg.norm(clamped1 - clamped2, axis=1)))
        self.assertAlmostEqual(d_obb, d_ref, delta=5e-4)

        # 反例 9: 显式传入不存在的 target_id="o99" 时绝不静默回退到 XY 最近邻物体
        self.assertIsNone(vf.find_target_object("o99", position_m=[0.10, -0.20, 0.780]))
        ev_missing = vf.evaluate_grasp("right", [0.10, -0.20, 0.780], [0.0, 0.0, -60.0], target_id="o99")
        self.assertFalse(ev_missing["theoretically_graspable"])
        self.assertIn("no_target_object_in_reach", ev_missing["failure_reasons"])

    def test_cross_contact_region_zero_offset_interpolation_and_objective_clearance(self):
        """验证跨 contact_region 统一在 pad_center_m 参考系零偏移插值（消除 ~22.5mm 跳变），以及客观 min_clearance_by_object_m 与显式豁免列表。"""
        from ni_rua.perception.external.pointcloud_field import PointCloudSpatialField

        # 在起点 [0.10, -0.20] 手掌正上方 4mm 处放置顶棚障碍 o_shelf（手掌顶面在 pad_z=0.800 时位于 0.800 + 0.1325 = 0.9325m）：
        # o_shelf 底面位于 0.938m（净空 +5.5mm）。若跨 contact_region 发生旧版 +15mm/+22.5mm 起点虚高跳变，s_i=0 手掌必撞穿 o_shelf！
        shelf_table = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[
                ["o_shelf", [0.10, -0.20, 0.958], 0.978, 0.938, 0.100, 0.100, 0.0, None, "unknown", "ok"],
                ["o_peg", [0.32, -0.20, 0.810], 0.855, 0.765, 0.030, 0.030, 0.0, None, "table", "ok"],
            ],
        )
        init_rs = {
            "arms": {
                "right": {
                    "grasp_pose": {"position_m": [0.10, -0.20, 0.800], "angles_deg": [0.0, 0.0, 0.0]},
                    "gripper": {"goal_normalized": 1.0},
                }
            }
        }
        # 阶段 1 为 finger_middle (z=0.8275 -> pad_center z=0.800)，阶段 2 切换为 fingertip (指尖 z=0.7925 -> pad_center z=0.800)
        cross_region_stages = [
            {
                "id": "hold_mid",
                "arm": "right",
                "kind": "move",
                "contact_region": "finger_middle",
                "grasp_position_m": [0.10, -0.20, 0.8275],
                "angles_deg": [0.0, 0.0, 0.0],
            },
            {
                "id": "slide_fingertip",
                "arm": "right",
                "kind": "move",
                "contact_region": "fingertip",
                "grasp_position_m": [0.16, -0.20, 0.7925],
                "angles_deg": [0.0, 0.0, 0.0],
            },
        ]
        shelf_slices = {
            "o_shelf": [
                {
                    "z_m": 0.938,
                    "points_m": [
                        [0.06, -0.24, 0.938], [0.14, -0.24, 0.938], [0.14, -0.16, 0.938], [0.06, -0.16, 0.938],
                        [0.06, -0.24, 0.978], [0.14, -0.24, 0.978], [0.14, -0.16, 0.978], [0.06, -0.16, 0.978],
                    ],
                }
            ]
        }
        for field in (
            VirtualSpatialField(objects=shelf_table, robot_state=init_rs),
            PointCloudSpatialField(objects=shelf_table, robot_state=init_rs, surface_slices_m=shelf_slices),
        ):
            sim = field.simulate_stages(cross_region_stages)
            self.assertEqual(sim["status"], "clean", (type(field).__name__, sim))
            self.assertIn("o_shelf", sim["min_clearance_by_object_m"])
            self.assertGreater(sim["stage_reports"][1]["min_clearance_by_object_m"]["o_shelf"], 0.003)

        # 验证不再靠末步闭爪盲猜 intended_target 掩盖途中扫掠碰撞：低空水平横扫穿过立柱 o_peg 再闭爪时如实检出路径碰撞
        vf = PointCloudSpatialField(objects=shelf_table, robot_state=init_rs, surface_slices_m=shelf_slices)
        lateral_plow_grasp = [
            {
                "id": "plow_peg",
                "arm": "right",
                "kind": "move",
                "grasp_position_m": [0.32, -0.20, 0.810],
                "angles_deg": [0.0, 0.0, 0.0],
                "gripper_goal_normalized": 0.0,
            }
        ]
        sim_crush = vf.simulate_stages(lateral_plow_grasp)
        self.assertEqual(sim_crush["status"], "risk_detected")
        self.assertIn("o_peg", sim_crush["stage_reports"][0]["colliding_objects"])
        self.assertLess(sim_crush["min_clearance_by_object_m"]["o_peg"], -0.002)

        sim_allowed = vf.simulate_stages(lateral_plow_grasp, allow_contact_ids=["o_peg"])
        self.assertEqual(sim_allowed["status"], "clean", sim_allowed)
        self.assertEqual(sim_allowed["stage_reports"][0]["colliding_objects"], [])
        self.assertLess(sim_allowed["min_clearance_by_object_m"]["o_peg"], -0.002)

        sim_ignored = vf.simulate_stages(lateral_plow_grasp, ignore_object_ids=["o_peg"])
        self.assertEqual(sim_ignored["status"], "clean", sim_ignored)
        self.assertNotIn("o_peg", sim_ignored["min_clearance_by_object_m"])

    def test_unmeasured_table_z_remains_none_without_0765_fallback(self):
        """验证无实测支撑面时 table_z 严格保持 None，绝不回退硬编码 0.765m。"""
        raw_objs = [
            dict(
                id="o1",
                centre_m=[0.20, 0.00, 0.520],
                top_z_m=0.540,
                bottom_z_m=0.500,
                length_m=0.040,
                width_m=0.040,
                yaw_deg=0.0,
                support="unknown",
            )
        ]
        vf = VirtualSpatialField(objects=raw_objs, table_z=None)
        self.assertIsNone(vf.table_z)
        ev = vf.evaluate_grasp("right", [0.20, 0.00, 0.520], [0.0, 0.0, 0.0], target_id="o1")
        self.assertIsNone(ev["support_clearance_m"])
        self.assertTrue(ev["theoretically_graspable"], ev)
        sim = vf.simulate_stages([
            {"id": "low_move", "arm": "right", "kind": "move", "grasp_position_m": [0.20, -0.15, 0.650], "angles_deg": [0.0, 0.0, 0.0]},
        ])
        self.assertEqual(sim["status"], "clean", sim)


TABLE_Z = 0.765


def _rot_from_rpy_deg(r_deg: float, p_deg: float, y_deg: float) -> np.ndarray:
    return Rotation.from_euler("xyz", [r_deg, p_deg, y_deg], degrees=True).as_matrix()


def _make_hollow_box_npz(
    center_xy=(0.20, 0.00),
    outer_size_m=0.14,
    wall_thickness_m=0.010,
    bottom_z=TABLE_Z,
    top_z=TABLE_Z + 0.080,
    track_num: int = 2,
) -> bytes:
    """生成带四周外壁与内底板的中空容器 `.npz` 点云字节流。"""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    half = 0.5 * float(outer_size_m)
    inner_half = half - float(wall_thickness_m)
    floor_z = float(bottom_z) + 0.006
    fx = np.arange(cx - inner_half, cx + inner_half + 1e-6, 0.008)
    fy = np.arange(cy - inner_half, cy + inner_half + 1e-6, 0.008)
    fxx, fyy = np.meshgrid(fx, fy)
    floor_pts = np.column_stack([fxx.ravel(), fyy.ravel(), np.full(fxx.size, floor_z)])
    walls = []
    span = np.arange(-half, half + 1e-6, 0.006)
    for zv in np.arange(bottom_z + 0.004, top_z + 1e-6, 0.008):
        for off in (0.0, wall_thickness_m):
            walls.append(np.column_stack([np.full(len(span), cx - half + off), cy + span, np.full(len(span), zv)]))
            walls.append(np.column_stack([np.full(len(span), cx + half - off), cy + span, np.full(len(span), zv)]))
            walls.append(np.column_stack([cx + span, np.full(len(span), cy - half + off), np.full(len(span), zv)]))
            walls.append(np.column_stack([cx + span, np.full(len(span), cy + half - off), np.full(len(span), zv)]))
    xyz = np.vstack([floor_pts, *walls]).astype(np.float32)
    n = len(xyz)
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        xyz=xyz,
        conf=np.full(n, 230, dtype=np.uint8),
        uv=np.zeros((n, 2), dtype=np.uint16),
        count=np.ones(n, dtype=np.uint16),
        robot=np.zeros(n, dtype=bool),
        obj_id=np.full(n, track_num, dtype=np.int16),
        view_mask=np.ones(n, dtype=np.uint8),
        support_z=np.float32(bottom_z),
        voxel_m=np.float32(0.005),
        image_size=np.array([640, 480], dtype=np.int32),
    )
    return buf.getvalue()


class HollowGraspHintsAndKDTreeTest(unittest.TestCase):
    """局部系 `LocalSurfaceKDTree`、三维中空/倾斜几何与双臂对称 `grasp_hints_from` 单测。"""

    def tearDown(self):
        s3.use(None)

    def test_hollow_mugs_o1_grasp_hints_obstacle_switch_and_hybrid_kdtree_parity(self):
        """验证中空杯壁 O(1) 提示无泄漏、邻域阻挡自动换边及点云树/切片树/壳体张量与标量核等价。"""
        def _mug_slices(cx_v: float, cy_v: float) -> list[dict]:
            h_v = 0.043
            hull_v = [
                [round(cx_v - h_v, 4), round(cy_v - h_v, 4)],
                [round(cx_v + h_v, 4), round(cy_v - h_v, 4)],
                [round(cx_v + h_v, 4), round(cy_v + h_v, 4)],
                [round(cx_v - h_v, 4), round(cy_v + h_v, 4)],
            ]
            return [dict(z_m=TABLE_Z + 0.015, hull_xy_m=hull_v), dict(z_m=TABLE_Z + 0.072, hull_xy_m=hull_v)]

        mug_target = dict(
            id="o1", track_id="o1", centre_m=[0.22, 0.00, TABLE_Z + 0.040],
            top_z_m=TABLE_Z + 0.080, bottom_z_m=TABLE_Z, length_m=0.086, width_m=0.086,
            yaw_deg=0.0, tilt_deg=89.0, support="table", quality="ok",
            cavity_depth_m=0.068, surface_slices_m=_mug_slices(0.22, 0.00),
        )
        mug_nb_far = dict(
            id="o2", track_id="o2", centre_m=[0.22, 0.16, TABLE_Z + 0.040],
            top_z_m=TABLE_Z + 0.080, bottom_z_m=TABLE_Z, length_m=0.086, width_m=0.086,
            yaw_deg=0.0, tilt_deg=89.0, support="table", quality="ok",
            cavity_depth_m=0.068, surface_slices_m=_mug_slices(0.22, 0.16),
        )
        t0 = time.perf_counter()
        hints = object_table.grasp_hints_from([mug_target, mug_nb_far], support_z_m=TABLE_Z)
        self.assertLess(time.perf_counter() - t0, 0.05)
        self.assertNotIn("safe_grasp_pose", hints["o1"])
        self.assertEqual(hints["o1"]["cavity_depth_m"], 0.068)

        vf_init = PointCloudSpatialField(objects=[mug_target, mug_nb_far], table_z=TABLE_Z)
        rim_pos = [0.22 - 0.039, 0.00, TABLE_Z + 0.072]
        ev_clear = vf_init.evaluate_grasp("right", rim_pos, [0.0, 0.0, 0.0], target_id="o1")
        self.assertTrue(ev_clear["theoretically_graspable"], ev_clear)
        self.assertNotIn("optimal_grasp_pose", ev_clear)
        self.assertEqual(
            vg_mod.evaluate_grasp_batch(vf_init, vf_init.objects["o1"], [rim_pos], [[0.0, 0.0, 0.0]])[0],
            ev_clear,
        )

        old_parts = gripper_obbs(rim_pos, [0.0, 0.0, 0.0])
        t_xy = np.array([0.22, 0.00], dtype=np.float64)
        lf_xy, rf_xy = old_parts["left_finger"].center[:2], old_parts["right_finger"].center[:2]
        outer_xy = lf_xy if np.linalg.norm(lf_xy - t_xy) > np.linalg.norm(rf_xy - t_xy) else rf_xy
        u_out = (outer_xy - t_xy) / max(1e-6, float(np.linalg.norm(outer_xy - t_xy)))
        block_xy = outer_xy + u_out * 0.038
        mug_nb_blocking = deepcopy(mug_nb_far)
        mug_nb_blocking["centre_m"] = [round(float(block_xy[0]), 4), round(float(block_xy[1]), 4), TABLE_Z + 0.040]
        mug_nb_blocking["surface_slices_m"] = _mug_slices(mug_nb_blocking["centre_m"][0], mug_nb_blocking["centre_m"][1])

        vf_blocked = PointCloudSpatialField(objects=[mug_target, mug_nb_blocking], table_z=TABLE_Z)
        ev_old = vf_blocked.evaluate_grasp("right", rim_pos, [0.0, 0.0, 0.0], target_id="o1")
        self.assertFalse(ev_old["theoretically_graspable"])
        self.assertIn("o2", ev_old["colliding_neighbor_ids"])
        opp_rim_pos = [0.22 + 0.039, 0.00, TABLE_Z + 0.072]
        ev_opp = vf_blocked.evaluate_grasp("right", opp_rim_pos, [0.0, 0.0, 0.0], target_id="o1")
        self.assertTrue(ev_opp["theoretically_graspable"], ev_opp)

        # 点云树 + 切片树 + 倾斜实心杆混合场与 geometry_ops 对拍
        npz_o1 = _make_hollow_box_npz(center_xy=(0.20, -0.06), outer_size_m=0.084, wall_thickness_m=0.008, top_z=TABLE_Z + 0.078, track_num=1)
        raw_objs = [
            dict(id="o1", track_id="o1", centre_m=[0.20, -0.06, TABLE_Z + 0.039], top_z_m=TABLE_Z + 0.078, bottom_z_m=TABLE_Z, length_m=0.084, width_m=0.084, yaw_deg=20.0, cavity_depth_m=0.065),
            dict(id="o2", track_id="o2", centre_m=[0.20, 0.04, TABLE_Z + 0.040], top_z_m=TABLE_Z + 0.080, bottom_z_m=TABLE_Z, length_m=0.086, width_m=0.086, yaw_deg=-15.0, cavity_depth_m=0.068),
            dict(id="o3", track_id="o3", centre_m=[0.30, -0.01, TABLE_Z + 0.025], top_z_m=TABLE_Z + 0.050, bottom_z_m=TABLE_Z, length_m=0.070, width_m=0.045, yaw_deg=35.0,
                 surface_slices_m=[dict(z_m=TABLE_Z + 0.012, hull_xy_m=[[0.27, -0.03], [0.33, -0.03], [0.33, 0.01], [0.27, 0.01]]),
                                   dict(z_m=TABLE_Z + 0.045, hull_xy_m=[[0.28, -0.02], [0.32, -0.02], [0.32, 0.00], [0.28, 0.00]])]),
            dict(id="o4", track_id="o4", centre_m=[0.11, -0.02, TABLE_Z + 0.025], top_z_m=TABLE_Z + 0.050, bottom_z_m=TABLE_Z, length_m=0.120, width_m=0.026, yaw_deg=50.0, tilt_deg=65.0),
        ]
        vf = PointCloudSpatialField(objects=raw_objs, cloud_npz=npz_o1, table_z=TABLE_Z)
        self.assertIn("o1", vf._obj_local_trees)
        self.assertIn("o3", vf._obj_slice_trees)

        rng = np.random.default_rng(123)
        tree = vf._obj_local_trees["o1"]
        o1_obb = vf.objects["o1"].obb()
        q_obbs = [
            OBB3D(o1_obb.center + rng.uniform(-0.08, 0.08, size=3), np.eye(3), rng.uniform(0.006, 0.025, size=3))
            for _ in range(16)
        ]
        cw = np.stack([b.center for b in q_obbs], axis=0)
        aw = np.stack([b.axes for b in q_obbs], axis=0)
        hw = np.stack([b.half_extents for b in q_obbs], axis=0)
        d_vec_full = tree.signed_distances_to_obbs(cw, aw, hw, o1_obb, voxel_radius_m=0.006)
        d_vec_cut = tree.signed_distances_to_obbs(cw, aw, hw, o1_obb, voxel_radius_m=0.006, skip_above_m=0.0501)
        for idx, qb in enumerate(q_obbs):
            d_ref = tree.signed_distance_to_obb(qb, o1_obb, voxel_radius_m=0.006)
            self.assertAlmostEqual(float(d_vec_full[idx]), d_ref, places=10)
            if d_ref <= 0.0501:
                self.assertAlmostEqual(float(d_vec_cut[idx]), d_ref, places=10)
            else:
                self.assertGreater(float(d_vec_cut[idx]), 0.0501)

        for oid in ("o1", "o2", "o3", "o4"):
            obj = vf.objects[oid]
            res_batch = clearance_oracle.evaluate_grasp_clearance_batch(
                vf, obj, [obj.centre_m], [[0.0, 0.0, obj.yaw_deg + 90.0]]
            )[0]
            res_single = vf.evaluate_grasp("right", obj.centre_m, [0.0, 0.0, obj.yaw_deg + 90.0], target_id=oid)
            self.assertEqual(res_batch, res_single, oid)
            self.assertNotIn("optimal_grasp_pose", res_single)

    def test_dual_arm_symmetric_held_by_exclusion_in_grasp_hints_and_field(self):
        """验证左臂持握 (held_left) 与右臂持握 (held_right) 双臂完全对称跳过 grasp_hints 与邻域干涉。"""
        base_target = dict(track_id="o1", centre_m=[0.20, 0.00, TABLE_Z + 0.020], top_z_m=TABLE_Z + 0.040,
                           bottom_z_m=TABLE_Z, length_m=0.120, width_m=0.028, yaw_deg=0.0, tilt_deg=89.0, support="table", quality="ok")
        held_left_obj = dict(track_id="o2", centre_m=[0.20, 0.042, 1.200], top_z_m=1.250, bottom_z_m=1.150,
                             length_m=0.060, width_m=0.060, yaw_deg=0.0, support="held_left", quality="ok")
        held_right_obj = dict(held_left_obj, support="held_right")
        hints_alone = object_table.grasp_hints_from([base_target], support_z_m=TABLE_Z)
        hints_left = object_table.grasp_hints_from([base_target, held_left_obj], support_z_m=TABLE_Z)
        hints_right = object_table.grasp_hints_from([base_target, held_right_obj], support_z_m=TABLE_Z)
        self.assertEqual(hints_left["o1"], hints_right["o1"])
        self.assertEqual(hints_left["o1"], hints_alone["o1"])
        self.assertNotIn("o2", hints_left)
        self.assertNotIn("o2", hints_right)
        vf_l = PointCloudSpatialField(objects=[dict(base_target, id="o1"), dict(held_left_obj, id="o2")], table_z=TABLE_Z)
        vf_r = PointCloudSpatialField(objects=[dict(base_target, id="o1"), dict(held_right_obj, id="o2")], table_z=TABLE_Z)
        self.assertEqual(
            vf_l.evaluate_grasp("right", [0.20, 0.00, TABLE_Z + 0.020], [0.0, 0.0, 90.0], target_id="o1"),
            vf_r.evaluate_grasp("right", [0.20, 0.00, TABLE_Z + 0.020], [0.0, 0.0, 90.0], target_id="o1"),
        )

    def test_local_kdtree_se3_zero_rebuild_and_hollow_container_preflight(self):
        """验证局部系 cKDTree 与暴力计算精确等价、刚体 SE(3) 零重建，且中空容器预检内腔放行、撞壁拦截。"""
        rng = np.random.default_rng(42)
        init_center = np.array([0.18, -0.12, 0.81], dtype=np.float64)
        init_axes = _rot_from_rpy_deg(12.0, -8.0, 35.0)
        local_pts = rng.uniform([-0.06, -0.04, -0.03], [0.06, 0.04, 0.03], size=(300, 3))
        tree = LocalSurfaceKDTree(init_center[None, :] + local_pts @ init_axes.T, init_center, init_axes)
        self.assertEqual(tree.build_count, 1)

        moved_center = np.array([-0.05, 0.22, 0.93], dtype=np.float64)
        moved_axes = _rot_from_rpy_deg(-28.0, 19.0, 115.0)
        moved_obb = OBB3D(moved_center, moved_axes, [0.06, 0.04, 0.03], label="o1")
        world_pts_moved = tree.world_points(moved_obb)

        query_boxes = []
        for idx in range(16):
            offset = rng.uniform(-0.02, 0.02, size=3) if idx % 3 == 0 else rng.uniform(-0.14, 0.14, size=3)
            q_axes = _rot_from_rpy_deg(float(rng.uniform(-45, 45)), float(rng.uniform(-45, 45)), float(rng.uniform(-180, 180)))
            q_obb = OBB3D(moved_center + offset, q_axes, rng.uniform(0.008, 0.035, size=3))
            query_boxes.append(q_obb)
            self.assertAlmostEqual(
                tree.signed_distance_to_obb(q_obb, moved_obb, voxel_radius_m=0.006),
                obb_to_points_signed_distance(q_obb, world_pts_moved, voxel_radius_m=0.006),
                places=9,
            )
        d_batch = tree.batch_stage_clearance(query_boxes, moved_obb, voxel_radius_m=0.006)
        d_min = min(obb_to_points_signed_distance(qb, world_pts_moved, voxel_radius_m=0.006) for qb in query_boxes)
        self.assertAlmostEqual(d_batch, d_min, places=9)
        self.assertEqual(tree.build_count, 1)

        npz_bytes = _make_hollow_box_npz(center_xy=(0.20, 0.00), outer_size_m=0.18, wall_thickness_m=0.010, track_num=2)
        table = {
            "columns": COLUMNS,
            "table_z": TABLE_Z,
            "rows": [
                ["o1", [0.00, -0.18, 0.790], 0.815, 0.765, 0.024, 0.024, 0.0, 89.0, "table", "ok"],
                ["o2", [0.20, 0.00, 0.805], 0.845, 0.765, 0.180, 0.180, 0.0, 89.0, "table", "ok"],
            ],
            "grasp_hints": {
                "o1": {"pad_z_m": 0.790, "opening_check": "ok", "elongated": False},
                "o2": {"opening_check": "too_wide", "elongated": False, "cavity_depth_m": 0.072},
            },
        }
        shared_vf = PointCloudSpatialField(objects=table, cloud_npz=npz_bytes)
        shared_vf.observation_id = "obs_1"
        o2_tree = shared_vf._obj_local_trees["o2"]
        cloned_vf = shared_vf.clone()
        self.assertIs(cloned_vf._obj_local_trees["o2"], o2_tree)
        self.assertEqual(o2_tree.build_count, 1)
        orig_o1_centre = list(shared_vf.objects["o1"].centre_m)
        harness = SimpleNamespace(_obs3d_memo=("obs_1", {"object_table": (table, None, {})}), _virtual_spatial_field=shared_vf)
        insert_action = {
            "based_on_observation_id": "obs_1",
            "stages": [
                {"id": "open_approach", "kind": "move", "arm": "right", "action": {"input_grasp_pose": {"position_m": [0.00, -0.18, 0.870], "angles_deg": [0.0, 0.0, 0.0]}, "gripper_events": [{"goal_normalized": 1.0}]}},
                {"id": "grasp_peg", "kind": "move", "arm": "right", "action": {"input_grasp_pose": {"position_m": [0.00, -0.18, 0.790], "angles_deg": [0.0, 0.0, 0.0]}, "gripper_events": [{"goal_normalized": 0.0}]}},
                {"id": "lift_peg", "kind": "move", "arm": "right", "action": {"input_grasp_pose": {"position_m": [0.00, -0.18, 0.920], "angles_deg": [0.0, 0.0, 0.0]}}},
                {"id": "move_over", "kind": "move", "arm": "right", "action": {"input_grasp_pose": {"position_m": [0.20, 0.00, 0.920], "angles_deg": [0.0, 0.0, 0.0]}}},
                {"id": "lower_cavity", "kind": "move", "arm": "right", "action": {"input_grasp_pose": {"position_m": [0.20, 0.00, 0.815], "angles_deg": [0.0, 0.0, 0.0]}, "gripper_events": [{"goal_normalized": 1.0}]}},
            ],
        }
        review_clean, _ = audit_batch_grasp_3d(harness, insert_action, "obs_1")
        self.assertEqual(review_clean["blocking_risks"], [], review_clean)
        self.assertEqual(shared_vf.objects["o1"].centre_m, orig_o1_centre)
        self.assertEqual(o2_tree.build_count, 1)

        hit_wall = deepcopy(insert_action)
        hit_wall["stages"][-1]["action"]["input_grasp_pose"]["position_m"] = [0.112, 0.00, 0.815]
        review_wall, _ = audit_batch_grasp_3d(harness, hit_wall, "obs_1")
        self.assertIn("warn_transport_collision", review_wall["risk_flags"])

    def test_tilted_obb_thin_cloth_occlusion_and_uncertainty_meta(self):
        """验证 45 度倾斜 OBB、薄布 ap=0.0m 随动持握、未观测物体只读历史不阻断轨迹及实测不确定度推导。"""
        # 1) 45 度斜靠长杆三维正交标架
        table_rod = {"columns": COLUMNS, "table_z": TABLE_Z, "rows": [["o1", [0.10, 0.00, TABLE_Z + 0.0565], TABLE_Z + 0.113, TABLE_Z, 0.113, 0.020, 0.0, 45.0, "table", "ok"]]}
        s_vals = np.linspace(-0.065, 0.065, 60)
        rod_pts = np.column_stack([0.10 + s_vals * np.cos(np.radians(45.0)), np.zeros_like(s_vals), (TABLE_Z + 0.0565) - s_vals * np.sin(np.radians(45.0))])
        vf_rod = PointCloudSpatialField(objects=table_rod)
        vf_rod.objects["o1"].refine_tilt_from_points(rod_pts)
        axes = vf_rod.objects["o1"].world_axes()
        np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-7)
        self.assertGreater(vf_rod.object_signed_distance("o1", OBB3D([0.055, 0.00, TABLE_Z + 0.018], np.eye(3), [0.008, 0.008, 0.008])), 0.020)

        # 2) 超薄布料 ap=0.0m 闭合保持绑定，开爪释放
        vf_cloth = PointCloudSpatialField(objects={"columns": COLUMNS, "table_z": TABLE_Z, "rows": [["o1", [0.15, -0.10, TABLE_Z + 0.002], TABLE_Z + 0.004, TABLE_Z, 0.160, 0.120, 0.0, 89.0, "table", "ok"]]})
        rs_grasp = {"arms": {"right": {"grasp_pose": {"position_m": [0.15, -0.10, TABLE_Z + 0.004], "angles_deg": [90.0, 0.0, 90.0]}, "gripper": {"position_m": [0.0, 0.0], "goal_normalized": 0.0}}}}
        vf_cloth._update_robot_state(rs_grasp)
        vf_cloth.objects["o1"].attach_to_gripper("right", vf_cloth.gripper_frame("right"))
        rs_lifted = {"arms": {"right": {"grasp_pose": {"position_m": [0.15, -0.10, 0.880], "angles_deg": [90.0, 0.0, 90.0]}, "gripper": {"position_m": [0.0, 0.0], "goal_normalized": 0.0}}}}
        vf_cloth.update_from_observation({"columns": COLUMNS, "table_z": TABLE_Z, "rows": []}, robot_state=rs_lifted)
        self.assertEqual(vf_cloth.objects["o1"].held_by_arm, "right")
        rs_opened = {"arms": {"right": {"grasp_pose": {"position_m": [0.15, -0.10, 0.880], "angles_deg": [90.0, 0.0, 90.0]}, "gripper": {"position_m": [0.040, 0.040], "goal_normalized": 1.0}}}}
        vf_cloth.update_from_observation({"columns": COLUMNS, "table_z": TABLE_Z, "rows": []}, robot_state=rs_opened)
        self.assertNotIn("o1", vf_cloth.objects)

        # 3) 多帧未观测物体只读生命周期（1..4 帧递增、幽灵路径零阻断、重现复位与非空实测 uncertainty 更新、第 5 帧淘汰）+ obs3d Copy-on-Write 原子同步
        row_o1 = ["o1", [0.00, -0.15, 0.785], 0.805, 0.765, 0.040, 0.040, 0.0, 89.0, "table", "ok"]
        row_o2 = ["o2", [0.18, 0.05, 0.790], 0.815, 0.765, 0.050, 0.050, 15.0, 89.0, "table", "ok"]
        vf_occ = PointCloudSpatialField(
            objects={
                "columns": COLUMNS,
                "table_z": TABLE_Z,
                "rows": [row_o1, row_o2],
                "spatial_meta": {"o2": {"uncertainty_m": 0.0031}},
            },
            max_occluded_steps=4,
        )
        self.assertAlmostEqual(vf_occ.objects["o2"].uncertainty_m, 0.0031, places=4)
        self.assertEqual(vf_occ.objects["o2"].occluded_steps, 0)
        for step_idx in range(1, 5):
            vf_occ.update_from_observation({"columns": COLUMNS, "table_z": TABLE_Z, "rows": [row_o1]})
            self.assertNotIn("o2", vf_occ.objects)
            self.assertIn("o2", vf_occ.historical_unobserved_objects)
            hist_o2 = vf_occ.historical_unobserved_objects["o2"]
            self.assertTrue(hist_o2.occluded)
            self.assertEqual(hist_o2.occluded_steps, step_idx)
            self.assertAlmostEqual(hist_o2.uncertainty_m, 0.0031, places=4)
        sim_ghost = vf_occ.simulate_stages([
            {"id": "pass_ghost", "arm": "right", "kind": "move", "grasp_position_m": [0.18, 0.05, 0.790], "angles_deg": [0.0, 0.0, 0.0]},
        ])
        self.assertEqual(sim_ghost["status"], "clean", sim_ghost)
        self.assertNotIn("o2", sim_ghost["min_clearance_by_object_m"])

        vf_reappear = vf_occ.clone()
        vf_reappear.update_from_observation({
            "columns": COLUMNS,
            "table_z": TABLE_Z,
            "rows": [row_o1, row_o2],
            "spatial_meta": {"o2": {"uncertainty_m": 0.0087}},
        })
        self.assertIn("o2", vf_reappear.objects)
        self.assertNotIn("o2", vf_reappear.historical_unobserved_objects)
        self.assertFalse(vf_reappear.objects["o2"].occluded)
        self.assertEqual(vf_reappear.objects["o2"].occluded_steps, 0)
        self.assertAlmostEqual(vf_reappear.objects["o2"].uncertainty_m, 0.0087, places=4)

        vf_occ.update_from_observation({"columns": COLUMNS, "table_z": TABLE_Z, "rows": [row_o1]})
        self.assertNotIn("o2", vf_occ.objects)
        self.assertNotIn("o2", vf_occ.historical_unobserved_objects)

        runtime = SimpleNamespace()
        obs3d._sync_virtual_field(runtime, {"observation_id": "obs_10"}, {"columns": COLUMNS, "table_z": TABLE_Z, "rows": [row_o1, row_o2]})
        snap_before = runtime._virtual_spatial_field
        obs3d._sync_virtual_field(runtime, {"observation_id": "obs_11"}, {"columns": COLUMNS, "table_z": TABLE_Z, "rows": [row_o1]})
        snap_mid = runtime._virtual_spatial_field
        self.assertIsNot(snap_before, snap_mid)
        self.assertNotIn("o2", snap_mid.objects)
        self.assertEqual(snap_mid.historical_unobserved_objects["o2"].occluded_steps, 1)
        obs3d._sync_virtual_field(runtime, {"observation_id": "obs_11"}, {"columns": COLUMNS, "table_z": TABLE_Z, "rows": [row_o1]}, cloud_npz=_make_hollow_box_npz(center_xy=(0.00, -0.15), outer_size_m=0.06, track_num=1))
        self.assertIsNot(snap_mid, runtime._virtual_spatial_field)
        self.assertEqual(runtime._virtual_spatial_field.historical_unobserved_objects["o2"].occluded_steps, 1)

        # 4) 纯实测不确定度计算（无 shape_tag）
        rng = np.random.default_rng(7)
        surf_pts = np.column_stack([rng.uniform(-0.04, 0.04, 40), rng.uniform(-0.03, 0.03, 40), np.full(40, 0.805) + rng.normal(0.0, 0.002, 40)])
        meta = derive_table_spatial_meta(
            [
                {"id": "o1", "centre_m": [0.0, 0.0, 0.785], "top_z_m": 0.805, "bottom_z_m": 0.765, "length_m": 0.04, "width_m": 0.04},
                {"id": "o2", "centre_m": [0.1, 0.0, 0.805], "top_z_m": 0.845, "bottom_z_m": 0.765, "length_m": 0.08, "width_m": 0.06, "surface_points_m": surf_pts},
                {"id": "o3", "centre_m": [0.2, 0.0, 0.795], "length_m": 0.18, "width_m": 0.06, "residual_std_m": 0.0031},
            ],
            support_z_m=None,
        )
        self.assertNotIn("o1", meta)
        self.assertGreater(meta["o2"]["uncertainty_m"], 0.0)
        self.assertEqual(meta["o3"], {"uncertainty_m": 0.0031})
        self.assertEqual(derive_object_spatial_meta(0.05, 0.05, 0.80, 0.765, quality="low_conf", occluded_steps=2), {"uncertainty_m": None})

        # 5) 实例掩码点严格限制在物体 OBB 包络内，绝不吸纳 OBB 外 +1.8cm 深度拖尾飞点
        base_npz = _make_hollow_box_npz(center_xy=(0.20, 0.00), outer_size_m=0.14, track_num=2)
        with np.load(io.BytesIO(base_npz), allow_pickle=False) as zf:
            z_dict = {k: zf[k] for k in zf.files}
        fly_pts = np.array([[0.20 + 0.070 + 0.018, 0.00, TABLE_Z + 0.040]] * 12, dtype=np.float32)
        z_dict["xyz"] = np.vstack([z_dict["xyz"], fly_pts])
        z_dict["conf"] = np.concatenate([z_dict["conf"], np.full(12, 230, dtype=np.uint8)])
        z_dict["uv"] = np.vstack([z_dict["uv"], np.zeros((12, 2), dtype=np.uint16)])
        z_dict["count"] = np.concatenate([z_dict["count"], np.ones(12, dtype=np.uint16)])
        z_dict["robot"] = np.concatenate([z_dict["robot"], np.zeros(12, dtype=bool)])
        z_dict["obj_id"] = np.concatenate([z_dict["obj_id"], np.full(12, 2, dtype=np.int16)])
        z_dict["view_mask"] = np.concatenate([z_dict["view_mask"], np.ones(12, dtype=np.uint8)])
        fly_buf = io.BytesIO()
        np.savez_compressed(fly_buf, **z_dict)
        vf_fly = PointCloudSpatialField(
            objects={"columns": COLUMNS, "table_z": TABLE_Z, "rows": [["o2", [0.20, 0.00, 0.805], 0.845, 0.765, 0.140, 0.140, 0.0, 89.0, "table", "ok"]]},
            cloud_npz=fly_buf.getvalue(),
        )
        probe_outside = OBB3D([0.20 + 0.070 + 0.018, 0.00, TABLE_Z + 0.040], np.eye(3), [0.004, 0.004, 0.004])
        self.assertGreater(vf_fly.object_signed_distance("o2", probe_outside), 0.006)

    def test_persist_spatial_snapshot_and_sandbox_zero_boilerplate_preload(self):
        """验证 obs3d._persist_spatial_snapshot、ProgramRunner 沙盒自动预载、轻量四件套隔离与零哈希泄露。"""
        import json
        from pathlib import Path
        import tempfile
        from ni_rua.core.observation_policy import ACTOR_PROFILE
        from ni_rua.core.presentation.actor_presentation import metadata_findings
        from ni_rua.core.storage.store import Store
        from ni_rua.runtime.programs.runner import ProgramRunner, ProgramServices
        from ni_rua.runtime.tools.builtins import create_registry

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        temp = temp_dir.name
        if True:
            identity = dict(run_id="run-0123456789abcdef", episode_id="ep", env_instance_id="env", env_generation=1)
            store = Store(Path(temp) / "run", identity)
            self.addCleanup(store.close)

            npz_bytes = _make_hollow_box_npz(center_xy=(0.20, 0.00), outer_size_m=0.14, track_num=2)
            cloud_ref = store.put_artifact(npz_bytes, "scene3d_points", "application/x-npz")
            raw_obs_id = "observation-fedcba9876543210fedcba98"
            table = {
                "columns": COLUMNS,
                "table_z": TABLE_Z,
                "rows": [
                    ["o1", [0.00, -0.18, 0.785], 0.805, 0.765, 0.030, 0.030, 0.0, 89.0, "table", "ok"],
                    ["o2", [0.20, 0.00, 0.805], 0.845, 0.765, 0.140, 0.140, 0.0, 89.0, "table", "ok"],
                ],
                "grasp_hints": {"o2": {"opening_check": "too_wide", "elongated": False, "cavity_depth_m": 0.070}},
                "spatial_meta": {"o1": {"uncertainty_m": 0.003}, "o2": {"uncertainty_m": 0.003}},
            }
            bundle = {
                "observation_id": raw_obs_id,
                "robot_state": {
                    "arms": {
                        "right": {
                            "ee_pose": {"position_m": [0.05, -0.15, 0.88], "quaternion_xyzw": [0.0, 0.7071068, 0.0, 0.7071068]},
                            "gripper": {"position_m": [0.021, 0.021], "goal_normalized": 0.5},
                        }
                    }
                },
            }
            runtime = SimpleNamespace(store=store)
            obs3d._sync_virtual_field(runtime, bundle, table, cloud_npz=npz_bytes)
            obs3d._persist_spatial_snapshot(runtime, bundle, {
                "observation_id": raw_obs_id,
                "object_table": table,
                "scene3d": {"artifact": cloud_ref, "points": 200, "table_z": TABLE_Z},
            })
            persisted = store.read_state("perception", "spatial_snapshot")
            self.assertEqual(persisted["cloud_artifact"], cloud_ref)

            registry = create_registry()
            services = ProgramServices(
                store=store,
                validate_tool=lambda name: registry.spec(name, role="program"),
                validate_arguments=lambda name, arguments: registry.validate(name, arguments, role="program"),
                invoke=lambda name, arguments, correlation: {"success": True, "contentItems": []},
                effect=lambda name, arguments: registry.spec(name, role="program").effect_for(arguments),
                observation=lambda: {"observation_id": raw_obs_id, "observation_profile": ACTOR_PROFILE},
                action_active=lambda identifier: False,
                cancel_action=lambda identifier: False,
                sandbox_enabled=expect(program_isolation, "program_run runs under the Landlock + seccomp sandbox"),
                workspace_modules=("scene3d_helpers.py",),
            )
            runner = ProgramRunner(services)
            self.addCleanup(runner.close)

            program_src = """import json, os
from pathlib import Path
import scene3d_helpers as s3

info_auto = s3.info()
prof = s3.cavity_profile("o2", radius=0.06)
snap_raw = Path("_spatial_snapshot.json").read_text(encoding="utf-8")
snap = json.loads(snap_raw)
s3.use(None)
try:
    s3.info()
    cleared_raised = False
except ValueError:
    cleared_raised = True
print(json.dumps({
    "workspace_dir": os.getcwd(),
    "obs_id": snap["observation_id"],
    "cloud_points": info_auto["n"],
    "cavity_depth_m": prof["cavity_depth_m"],
    "cleared_raised": cleared_raised,
    "snap_raw": snap_raw,
}))
"""
            res = runner.run({"source": program_src, "timeout_s": 30}, {"thread_id": "t1", "turn_id": "u1", "tool_call_id": "c1"})
            self.assertEqual(res["status"], "succeeded", res.get("stderr_text"))
            out = json.loads(res["stdout_text"])
            self.assertEqual(out["obs_id"], "obs_1")
            self.assertGreater(out["cloud_points"], 100)
            self.assertGreater(out["cavity_depth_m"], 0.05)
            self.assertTrue(out["cleared_raised"])
            self.assertEqual(list(metadata_findings(out["snap_raw"])), [])

            ws_dir = Path(out["workspace_dir"])
            for mod_name in ("scene3d_helpers.py", "pose_math.py", "field_ops.py", "ik_prefilter.py"):
                self.assertTrue((ws_dir / mod_name).is_file(), mod_name)
            for retired_mod in ("virtual_field.py", "pointcloud_field.py", "vectorized_grasp.py", "local_kdtree_field.py"):
                self.assertFalse((ws_dir / retired_mod).exists(), retired_mod)

    def test_redteam_track3_batch_neighbor_parity_cavity_kwargs_moved_pts_and_no_heuristics(self):
        """回归验证 #152 §1.1, §2.4, §2.6, §2.7, §2.8, §2.9, §2.10 的几何与无启发式不变量。"""
        # 1) #152 §1.1: 精细点云邻物 o2 穿透夹爪 (cand_ub < 0) 时，分离实心邻物 o3 (外接球重叠但 OBB 分离 +10mm)
        # 在标量 G=1 与批量 G=2 中均不得被误标入 colliding_neighbor_ids
        rad = math.radians(0.0)
        _ = rad
        # 目标 o1 在原点，yaw=90 (闭合轴沿 X)；精细邻物 o2 紧贴右指 (x=+0.046) 发生穿透；
        # 实心长条邻物 o3 位于 y=+0.062 (手指宽度半长 0.009 + o3 半宽 0.015 -> Y 向分离 ~38mm，但长轴 0.16m 使外接球半径大而 sphere_lb < 0)
        o2_pts = np.column_stack([
            np.random.default_rng(11).uniform(0.040, 0.056, 30),
            np.random.default_rng(12).uniform(-0.010, 0.010, 30),
            np.random.default_rng(13).uniform(TABLE_Z + 0.005, TABLE_Z + 0.035, 30),
        ]).astype(np.float32)
        buf_nb = io.BytesIO()
        np.savez_compressed(
            buf_nb,
            xyz=o2_pts,
            conf=np.full(len(o2_pts), 240, dtype=np.uint8),
            uv=np.zeros((len(o2_pts), 2), dtype=np.uint16),
            count=np.ones(len(o2_pts), dtype=np.uint16),
            robot=np.zeros(len(o2_pts), dtype=bool),
            obj_id=np.full(len(o2_pts), 2, dtype=np.int16),
            view_mask=np.ones(len(o2_pts), dtype=np.uint8),
            support_z=np.float32(TABLE_Z),
        )
        mixed_table = {
            "columns": COLUMNS,
            "table_z": TABLE_Z,
            "rows": [
                ["o1", [0.00, 0.00, TABLE_Z + 0.015], TABLE_Z + 0.030, TABLE_Z, 0.030, 0.024, 90.0, 89.0, "table", "ok"],
                ["o2", [0.048, 0.00, TABLE_Z + 0.020], TABLE_Z + 0.040, TABLE_Z, 0.020, 0.020, 0.0, 89.0, "table", "ok"],
                ["o3", [0.048, 0.032, TABLE_Z + 0.020], TABLE_Z + 0.040, TABLE_Z, 0.160, 0.020, 0.0, 89.0, "table", "ok"],
            ],
        }
        pcf_mix = PointCloudSpatialField(objects=mixed_table, cloud_npz=buf_nb.getvalue(), table_z=TABLE_Z)
        self.assertIn("o2", pcf_mix._obj_local_trees)
        self.assertNotIn("o3", pcf_mix._obj_local_trees)
        ev_g1 = clearance_oracle.evaluate_grasp_clearance(
            pcf_mix, "right", [0.00, 0.00, TABLE_Z + 0.015], [0.0, 0.0, 0.0], target_id="o1"
        )
        ev_g2 = vg_mod.evaluate_grasp_batch(
            field=pcf_mix,
            target=pcf_mix.objects["o1"],
            positions_m=[[0.00, 0.00, TABLE_Z + 0.015], [0.50, 0.00, 1.15]],
            angles_deg=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )
        self.assertEqual(ev_g1["colliding_neighbor_ids"], ["o2"])
        self.assertEqual(ev_g2[0]["colliding_neighbor_ids"], ["o2"])
        self.assertEqual(ev_g1, ev_g2[0])

        # 2) #152 §2.4 & §2.7: inspect_cavity_geometry 显式参数优先 + 容器刚体移动后使用变换后表面点
        pcf_cav_override = PointCloudSpatialField(
            objects={"columns": COLUMNS, "table_z": TABLE_Z, "rows": [["o1", [0.10, 0.00, TABLE_Z + 0.040], TABLE_Z + 0.080, TABLE_Z, 0.090, 0.090, 0.0, 89.0, "table", "ok"]]},
            cavity_depths_m={"o1": 0.020},
            table_z=TABLE_Z,
        )
        prof_override = clearance_oracle.inspect_cavity_geometry(
            pcf_cav_override,
            "o1",
            table_z=TABLE_Z + 0.010,
            cavity_depths_m={"o1": 0.065},
            surface_voxel_radius_m=0.004,
        )
        self.assertAlmostEqual(prof_override["cavity_depth_m"], 0.065, places=4)

        npz_box = _make_hollow_box_npz(center_xy=(0.20, 0.00), outer_size_m=0.10, wall_thickness_m=0.010, bottom_z=TABLE_Z, top_z=TABLE_Z + 0.060, track_num=2)
        pcf_move = PointCloudSpatialField(
            objects={
                "columns": COLUMNS,
                "table_z": TABLE_Z,
                "rows": [["o2", [0.20, 0.00, TABLE_Z + 0.030], TABLE_Z + 0.060, TABLE_Z, 0.100, 0.100, 0.0, 89.0, "table", "ok"]],
                "grasp_hints": {"o2": {"cavity_depth_m": 0.054}},
            },
            cloud_npz=npz_box,
            table_z=TABLE_Z,
        )
        prof_init = pcf_move.inspect_cavity("o2")
        sim_moved = pcf_move.clone()
        gf_attach = gripper_frame([0.20 - 0.045, 0.00, TABLE_Z + 0.052], [0.0, 0.0, 0.0])
        sim_moved.objects["o2"].attach_to_gripper("right", gf_attach)
        gf_dest = gripper_frame([0.40 - 0.045, 0.00, TABLE_Z + 0.152], [0.0, 0.0, 0.0])
        sim_moved.objects["o2"].update_from_gripper(gf_dest)
        prof_moved = sim_moved.inspect_cavity("o2")
        self.assertAlmostEqual(prof_moved["floor_z_m"], prof_init["floor_z_m"] + 0.10, delta=0.003)
        self.assertAlmostEqual(prof_moved["rim_z_m"], prof_init["rim_z_m"] + 0.10, delta=0.003)
        self.assertAlmostEqual(prof_moved["opening_center_xy_m"][0], prof_init["opening_center_xy_m"][0] + 0.20, delta=0.006)
        self.assertAlmostEqual(prof_moved["inner_clearance_radius_m"], prof_init["inner_clearance_radius_m"], delta=0.004)
        self.assertAlmostEqual(clearance_oracle.object_cavity_floor_z(sim_moved, sim_moved.objects["o2"]), prof_moved["floor_z_m"], places=4)

        # 3) #152 §2.8: 5.5mm 浅凹腔 (>= 0.004m) 在 s3.cavity_profile 与 PointCloudSpatialField.inspect_cavity 中一致检出
        shallow_npz = _make_hollow_box_npz(
            center_xy=(0.15, -0.10),
            outer_size_m=0.10,
            wall_thickness_m=0.008,
            bottom_z=TABLE_Z,
            top_z=TABLE_Z + 0.0115,
            track_num=3,
        )
        shallow_table = {
            "columns": COLUMNS,
            "table_z": TABLE_Z,
            "rows": [["o3", [0.15, -0.10, TABLE_Z + 0.006], TABLE_Z + 0.0115, TABLE_Z, 0.100, 0.100, 0.0, 89.0, "table", "ok"]],
            "grasp_hints": {"o3": {"cavity_depth_m": 0.0055}},
        }
        pcf_shallow = PointCloudSpatialField(objects=shallow_table, cloud_npz=shallow_npz, table_z=TABLE_Z)
        s3_shallow = s3.cavity_profile("o3", cloud=s3.from_bytes(shallow_npz), objects=shallow_table)
        pcf_shallow_prof = pcf_shallow.inspect_cavity("o3")
        self.assertGreaterEqual(s3_shallow["cavity_depth_m"], 0.004)
        self.assertAlmostEqual(s3_shallow["cavity_depth_m"], pcf_shallow_prof["cavity_depth_m"], delta=0.0015)

        # 4) #152 §2.9(a) & (b): 释放落位严格使用局部 OBB XY 足迹且绝不向上吸附 (+30mm) 到邻近窄导轨；
        # 60mm 长条在 88mm 张开夹爪内沿长轴闭合物理可抓 (theoretically_graspable=True) 且客观保留 closing_to_long_axis_deg
        bar_and_rail_table = {
            "columns": COLUMNS,
            "table_z": TABLE_Z,
            "rows": [
                ["bar60", [0.00, -0.20, TABLE_Z + 0.010], TABLE_Z + 0.020, TABLE_Z, 0.060, 0.020, 0.0, 89.0, "table", "ok"],
                ["rail", [0.20, 0.025, TABLE_Z + 0.010], TABLE_Z + 0.020, TABLE_Z, 0.200, 0.010, 0.0, 89.0, "table", "ok"],
            ],
        }
        vf_bar = VirtualSpatialField(objects=bar_and_rail_table, table_z=TABLE_Z)
        ev_along_60 = vf_bar.evaluate_grasp("right", [0.00, -0.20, TABLE_Z + 0.010], [0.0, 0.0, 0.0], target_id="bar60")
        self.assertTrue(ev_along_60["theoretically_graspable"], ev_along_60)
        self.assertEqual(ev_along_60["failure_reasons"], [])
        self.assertEqual(ev_along_60["closing_to_long_axis_deg"], 0.0)

        sim_release = vf_bar.simulate_stages([
            {"id": "grasp_bar", "arm": "right", "kind": "move", "grasp_position_m": [0.00, -0.20, TABLE_Z + 0.010], "angles_deg": [0.0, 0.0, 0.0], "gripper_goal_normalized": 0.0},
            {"id": "lift_bar", "arm": "right", "kind": "move", "grasp_position_m": [0.00, -0.20, TABLE_Z + 0.120], "angles_deg": [0.0, 0.0, 0.0]},
            {"id": "over_near_rail", "arm": "right", "kind": "move", "grasp_position_m": [0.20, 0.00, TABLE_Z + 0.120], "angles_deg": [0.0, 0.0, 90.0]},
            {"id": "drop_near_rail", "arm": "right", "kind": "move", "grasp_position_m": [0.20, 0.00, TABLE_Z + 0.015], "angles_deg": [0.0, 0.0, 90.0], "gripper_goal_normalized": 1.0},
        ])
        self.assertEqual(sim_release["stage_reports"][0]["grasped_object_id"], "bar60")
        self.assertEqual(sim_release["stage_reports"][-1]["new_support"], "table")
        after_bar = {o["id"]: o for o in sim_release["virtual_world_after"]}["bar60"]
        self.assertEqual(after_bar["support"], "table")
        self.assertAlmostEqual(after_bar["bottom_z"], TABLE_Z, places=4)

    def test_coal_obb_halfspace_pose_math_parity_and_isolated_program_runtime(self):
        """验证 coal C++ OBB/Halfspace 有符号距离、scipy/pinocchio 李群位姿算子精度与隔离 /harness program_runtime 执行。"""
        import json
        import os
        from pathlib import Path
        import subprocess
        import sys
        import tempfile

        import coal
        from scipy.spatial.transform import RigidTransform, Slerp

        from ni_rua.core import pose_math as core_pm
        from ni_rua.cortex.session.actor_source import snapshot_actor_source
        from ni_rua.runtime.programs.runner import program_runtime

        rng = np.random.default_rng(20261117)

        # 1) scipy.spatial.transform + pinocchio 位姿与李群算子 1e-14 级对拍与入参校验
        for _ in range(25):
            r_mat = Rotation.random(random_state=rng).as_matrix()
            q_wxyz = core_pm.matrix_to_quat_wxyz(r_mat)
            self.assertGreaterEqual(float(q_wxyz[0]), 0.0)
            np.testing.assert_allclose(
                q_wxyz,
                Rotation.from_matrix(r_mat).as_quat(canonical=True, scalar_first=True),
                atol=1e-15,
            )
            np.testing.assert_allclose(core_pm.quat_wxyz_to_matrix(q_wxyz), r_mat, atol=1e-14)
            q_xyzw = core_pm.matrix_to_quat_xyzw(r_mat)
            np.testing.assert_allclose(core_pm.quat_xyzw_to_matrix(q_xyzw), r_mat, atol=1e-14)

            xi = np.r_[rng.uniform(-0.3, 0.3, size=3), rng.uniform(-0.9, 0.9, size=3)]
            t_se3 = core_pm.se3_exp(xi)
            t_rt = RigidTransform.from_exp_coords(np.r_[xi[3:], xi[:3]]).as_matrix()
            np.testing.assert_allclose(t_se3, t_rt, atol=1e-14)
            np.testing.assert_allclose(core_pm.se3_log(t_se3), xi, atol=1e-13)
            np.testing.assert_allclose(core_pm.se3_inverse(t_se3) @ t_se3, np.eye(4), atol=1e-14)

            jl = core_pm.so3_left_jacobian(xi[3:])
            jli = core_pm.so3_left_jacobian_inv(xi[3:])
            np.testing.assert_allclose(jl @ jli, np.eye(3), atol=1e-13)
            eta = rng.uniform(-0.2, 0.2, size=6)
            np.testing.assert_allclose(
                t_se3 @ core_pm.se3_exp(eta) @ core_pm.se3_inverse(t_se3),
                core_pm.se3_exp(core_pm.se3_adjoint(t_se3) @ eta),
                atol=1e-13,
            )

        q_pi = core_pm.matrix_to_quat_wxyz(np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]))
        np.testing.assert_allclose(
            q_pi,
            Rotation.from_matrix([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]).as_quat(
                canonical=True, scalar_first=True
            ),
            atol=1e-15,
        )
        qa = core_pm.angles_deg_to_quat_wxyz(15.0, -25.0, 40.0)
        qb = core_pm.angles_deg_to_quat_wxyz(-30.0, 20.0, -110.0)
        np.testing.assert_allclose(
            core_pm.quat_slerp_wxyz(qa, qb, 0.35),
            Slerp([0.0, 1.0], Rotation.from_quat([qa, qb], scalar_first=True))(0.35).as_quat(
                canonical=True, scalar_first=True
            ),
            atol=1e-15,
        )
        for bad_so3 in (np.eye(2), np.diag([2.0, 1.0, 1.0]), np.diag([-1.0, 1.0, 1.0])):
            with self.assertRaises(ValueError):
                core_pm.so3_log(bad_so3)
        for bad_se3 in (np.eye(3), np.diag([1.0, 1.0, 1.0, 2.0]), np.diag([2.0, 1.0, 1.0, 1.0])):
            with self.assertRaises(ValueError):
                core_pm.se3_log(bad_se3)
            with self.assertRaises(ValueError):
                core_pm.se3_inverse(bad_se3)

        # 2) coal C++ Box / Halfspace 有符号距离与 skip_above_m 门控契约
        req = coal.DistanceRequest(True)
        res = coal.DistanceResult()
        obbs_a, obbs_b = [], []
        for idx in range(48):
            ra = Rotation.random(random_state=rng).as_matrix()
            rb = ra.copy() if idx % 6 == 0 else Rotation.random(random_state=rng).as_matrix()
            ha = np.array([1e-5, 0.02, 0.03]) if idx % 8 == 0 else rng.uniform(0.004, 0.06, size=3)
            hb = rng.uniform(0.004, 0.06, size=3)
            ca = rng.uniform(-0.15, 0.15, size=3)
            cb = ca.copy() if idx == 0 else ca + rng.uniform(-0.09, 0.09, size=3)
            oa, ob = OBB3D(ca, ra, ha), OBB3D(cb, rb, hb)
            obbs_a.append(oa)
            obbs_b.append(ob)
            res.clear()
            d_direct = float(
                coal.distance(
                    coal.Box(*(2.0 * oa.half_extents)),
                    coal.Transform3s(oa.axes, oa.center),
                    coal.Box(*(2.0 * ob.half_extents)),
                    coal.Transform3s(ob.axes, ob.center),
                    req,
                    res,
                )
            )
            self.assertAlmostEqual(obb_signed_distance(oa, ob), d_direct, places=12)
            # 验证 coal.Halfspace 桌面有符号距离等于最低顶点距桌面高度
            hs_d = geometry_ops.obb_to_halfspace_signed_distance(oa, normal=[0.0, 0.0, 1.0], offset_m=TABLE_Z)
            min_vert_z = float(np.min(oa.vertices()[:, 2]))
            self.assertAlmostEqual(hs_d, min_vert_z - TABLE_Z, places=11)

        d_pairs = geometry_ops.obb_signed_distance_pairs(obbs_a, obbs_b)
        d_cut = geometry_ops.obb_signed_distance_pairs(obbs_a, obbs_b, skip_above_m=0.025)
        for idx in range(len(obbs_a)):
            if d_pairs[idx] <= 0.025:
                self.assertAlmostEqual(float(d_cut[idx]), float(d_pairs[idx]), places=12)
            else:
                self.assertGreater(float(d_cut[idx]), 0.025)
        ref_global_min = min(obb_signed_distance(qa_b, sb_b) for qa_b in obbs_a[:8] for sb_b in obbs_b[:6])
        self.assertAlmostEqual(
            geometry_ops.batch_obbs_to_slabs_min_distance(obbs_a[:8], obbs_b[:6]),
            ref_global_min,
            places=12,
        )

        # 3) 隔离 /harness 快照 + runner.program_runtime (-S 零宿主 site-packages) 子进程导入与求解
        tmp_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        harness_snap = snapshot_actor_source(tmp_root / "harness")
        rt_root, rt_packages, _ = program_runtime(tmp_root / "program-runtime", sandbox_enabled=False)
        isolated_env = {
            "PATH": os.defpath,
            "HOME": str(tmp_root),
            "TMPDIR": str(tmp_root),
            "PYTHONHOME": str(rt_root),
            "PYTHONPATH": f"{harness_snap}{os.pathsep}{rt_packages}",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        code = (
            "import json, math, numpy as np, eigenpy, coal, pinocchio, qpsolvers, pink, daqp, ruckig\n"
            "from ni_rua.core import pose_math as pm\n"
            "from ni_rua.perception.external import field_ops as fo, geometry_ops as go, ik_prefilter as ikp\n"
            "b1 = go.OBB3D([0.0, 0.0, 0.80], np.eye(3), [0.02, 0.02, 0.02])\n"
            "b2 = go.OBB3D([0.05, 0.0, 0.80], np.eye(3), [0.02, 0.02, 0.02])\n"
            "d_box = go.obb_signed_distance(b1, b2)\n"
            "d_hs = go.obb_to_halfspace_signed_distance(b1, [0.0, 0.0, 1.0], 0.765)\n"
            "xi = np.array([0.1, -0.05, 0.02, 0.2, -0.1, 0.3])\n"
            "xi_rt = pm.se3_log(pm.se3_exp(xi))\n"
            "jl_err = float(np.max(np.abs(pm.so3_left_jacobian(xi[3:]) @ pm.so3_left_jacobian_inv(xi[3:]) - np.eye(3))))\n"
            "reg = fo.align_in_hand_feature_to_target([0.0, -0.2, 0.85], [0.0, 0.0, 0.0], [0.04, -0.2, 0.83], [1.0, 0.0, 0.0], [0.12, -0.15, 0.88], [0.0, 0.0, -1.0])\n"
            "ik_res = ikp.check_pose_reachability('right', [0.18, -0.22, 0.84], [0.0, 0.0, -20.0])\n"
            "otg = ruckig.Ruckig(1); inp = ruckig.InputParameter(1); tr = ruckig.Trajectory(1)\n"
            "inp.current_position, inp.current_velocity, inp.current_acceleration = [0.0], [0.0], [0.0]\n"
            "inp.target_position, inp.target_velocity, inp.target_acceleration = [1.0], [0.0], [0.0]\n"
            "inp.max_velocity, inp.max_acceleration, inp.max_jerk = [1.0], [10.0], [80.0]\n"
            "otg_ok = int(otg.calculate(inp, tr)) >= 0 and tr.duration > 1.0\n"
            "print(json.dumps({'d_box': d_box, 'd_hs': d_hs, 'xi_err': float(np.max(np.abs(xi - xi_rt))), 'jl_err': jl_err, 'pos_err': reg['feature_pos_err_m'], 'ik_ok': ik_res['reachable'], 'otg_ok': otg_ok}))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-S", "-c", code],
            cwd=str(tmp_root),
            env=isolated_env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        iso_out = json.loads(proc.stdout.strip())
        self.assertAlmostEqual(iso_out["d_box"], 0.01, places=9)
        self.assertAlmostEqual(iso_out["d_hs"], 0.015, places=9)
        self.assertLess(iso_out["xi_err"], 1e-13)
        self.assertLess(iso_out["jl_err"], 1e-13)
        self.assertLess(iso_out["pos_err"], 1e-6)
        self.assertTrue(iso_out["ik_ok"])
        self.assertTrue(iso_out["otg_ok"])

        # 4) 真实 ProgramRunner (`program_run`) 全链路执行核验（严格遵守 LIFO 清理顺序）
        from dataclasses import replace
        from types import SimpleNamespace
        from support.program_fixture import program_services
        from ni_rua.core.storage.store import Store
        from ni_rua.runtime.programs.runner import ProgramRunner

        store = Store(
            tmp_root / "run-store",
            dict(run_id="oss-prog", episode_id="ep1", env_instance_id="env1", env_generation=1),
        )
        self.addCleanup(store.close)
        prog_runner = ProgramRunner(replace(program_services(SimpleNamespace(store=store)), sandbox_enabled=False))
        self.addCleanup(prog_runner.close)
        prog_res = prog_runner.run(
            dict(
                source=(
                    "import eigenpy, coal, pinocchio, qpsolvers, pink, daqp, ruckig\n"
                    "assert coal.__version__ == '3.0.3' and pinocchio.__version__ == '4.1.0'\n"
                    "print('program_run_oss_ok')\n"
                ),
                timeout_s=30,
            ),
            dict(thread_id="t1", turn_id="u1", tool_call_id="c1"),
        )
        self.assertEqual(prog_res["status"], "succeeded", prog_res)
        self.assertEqual(prog_res["returncode"], 0)
        self.assertIn("program_run_oss_ok", prog_res.get("stdout_text", ""))



# --- Folded from tests/unit/test_s2_field_ops_and_ik_prefilter.py ---

class _DummyCompiledSpatialField:
    """模拟实现 `query_clearance_and_gradient(pts_m)` 协议的空间度量场。"""

    def __init__(self, sphere_center_m: list[float], sphere_radius_m: float):
        self.center = np.asarray(sphere_center_m, dtype=np.float64)
        self.radius = float(sphere_radius_m)

    def query_clearance_and_gradient(self, query_pts_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(query_pts_m, dtype=np.float64).reshape(-1, 3)
        diff = pts - self.center[None, :]
        norms = np.linalg.norm(diff, axis=1)
        clr = norms - self.radius
        grad = np.where(norms[:, None] > 1e-9, diff / np.maximum(norms[:, None], 1e-12), np.array([[0.0, 0.0, 1.0]]))
        return clr, grad


def _build_surface_cloud_npz(center_xy=(0.16, -0.14), base_z=0.765) -> bytes:
    """构造无预设魔法高度的实测环形凹腔表面点云 `.npz` 字节流（单位 `m`）。"""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    xs = np.arange(cx - 0.050, cx + 0.0501, 0.008)
    ys = np.arange(cy - 0.050, cy + 0.0501, 0.008)
    xx, yy = np.meshgrid(xs, ys)
    rr = np.hypot(xx.ravel() - cx, yy.ravel() - cy)
    ring_mask = rr >= 0.025
    top_pts = np.column_stack([xx.ravel()[ring_mask], yy.ravel()[ring_mask], np.full(int(np.sum(ring_mask)), base_z + 0.040)])
    floor_pts = np.column_stack([xx.ravel()[~ring_mask], yy.ravel()[~ring_mask], np.full(int(np.sum(~ring_mask)), base_z + 0.006)])
    xyz = np.vstack([top_pts, floor_pts]).astype(np.float32)
    n = len(xyz)
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        xyz=xyz,
        conf=np.full(n, 230, dtype=np.uint8),
        uv=np.zeros((n, 2), dtype=np.uint16),
        count=np.ones(n, dtype=np.uint16),
        robot=np.zeros(n, dtype=bool),
        obj_id=np.ones(n, dtype=np.int16),
        view_mask=np.ones(n, dtype=np.uint8),
        support_z=np.float32(base_z),
        voxel_m=np.float32(0.004),
        image_size=np.array([640, 480], dtype=np.int32),
    )
    return buf.getvalue()


class TestS2FieldOpsAndIKPrefilter(unittest.TestCase):
    """覆盖三维李群代数、功能特征配准、动态 `robot_model` 6R 运动学预筛、沙盒挂载与原生 Schema 零膨胀。"""

    def test_zero_overfitting_macro_templates_removed_and_lie_algebra_exactness(self) -> None:
        """确认过时任务套路宏模板已从 `field_ops` 与 `scene3d_helpers` 彻底清退，且李群算子精度严格达标。"""
        removed_names = (
            "evaluate_funnel_vector_field",
            "funnel_insert_stages",
            "compliant_axis_slide_stages",
            "repulsive_corridor_stages",
        )
        for name in removed_names:
            self.assertFalse(hasattr(fo, name), f"field_ops 仍残留套路模板 {name}")
            self.assertFalse(hasattr(s3, name), f"scene3d_helpers 仍残留套路模板 {name}")

        # 验证 SO(3)/SE(3) 指数-对数互逆（涵盖极小角 1e-9 rad、一般角与近 pi 奇异区）
        test_omegas = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1e-9, -2e-9, 1.5e-9]),
            np.array([0.45, -0.72, 0.91]),
            np.array([0.0, math.pi - 1e-5, 0.0]),
        ]
        for omega in test_omegas:
            rot = fo.so3_exp(omega)
            np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(float(np.linalg.det(rot)), 1.0, places=12)
            omega_rec = fo.so3_log(rot)
            np.testing.assert_allclose(fo.so3_exp(omega_rec), rot, atol=1e-11)

            jl = fo.so3_left_jacobian(omega)
            jl_inv = fo.so3_left_jacobian_inv(omega)
            np.testing.assert_allclose(jl @ jl_inv, np.eye(3), atol=1e-10)

            xi = np.concatenate([[0.12, -0.08, 0.05], omega])
            t_mat = fo.se3_exp(xi)
            xi_rec = fo.se3_log(t_mat)
            np.testing.assert_allclose(fo.se3_exp(xi_rec), t_mat, atol=1e-10)

            # 验证伴随表示恒等式：T * exp(eta^) * T^{-1} == exp((Ad_T * eta)^)
            eta = np.array([-0.03, 0.04, 0.02, 0.15, -0.10, 0.22], dtype=np.float64)
            lhs = t_mat @ fo.se3_exp(eta) @ fo.se3_inverse(t_mat)
            rhs = fo.se3_exp(fo.se3_adjoint(t_mat) @ eta)
            np.testing.assert_allclose(lhs, rhs, atol=1e-11)

        # 验证 query_spatial_clearance_and_gradient 忠实透传 exclude_object_ids / include_table，无悬空剔除黑盒参数
        compiled_mock = _DummyCompiledSpatialField([0.28, -0.14, 0.84], 0.035)
        recorded_kwargs: list[dict] = []

        class _FullCompiledField:
            def query_clearance_and_gradient(self, query_pts_m, *, exclude_object_ids=(), include_table: bool = False):
                recorded_kwargs.append(
                    {
                        "exclude_object_ids": tuple(exclude_object_ids),
                        "include_table": bool(include_table),
                    }
                )
                return compiled_mock.query_clearance_and_gradient(query_pts_m)

        clr_f, grad_f = fo.query_spatial_clearance_and_gradient(
            _FullCompiledField(),
            [[0.28, -0.14, 0.90]],
            exclude_object_ids=["o_held"],
            include_table=True,
        )
        self.assertAlmostEqual(float(clr_f[0]), 0.060 - 0.035, places=6)
        np.testing.assert_allclose(grad_f[0], [0.0, 0.0, 1.0], atol=1e-9)
        self.assertEqual(recorded_kwargs[-1], {"exclude_object_ids": ("o_held",), "include_table": True})

    def test_align_in_hand_feature_tilted_rod_and_dual_axis_sub_micron_accuracy(self) -> None:
        """验证斜置挂杆杯柄孔环与双插脚双轴锁定的三维刚体配准误差 `< 1e-6 m` 且欧拉角处于规范主分支。"""
        grasp_pos = [0.02, -0.22, 0.86]
        grasp_ang = [12.0, -8.0, 25.0]
        handle_center = [0.068, -0.205, 0.848]
        handle_normal = [0.92, 0.35, -0.17]
        handle_cup_down = [0.0, 0.0, -1.0]

        rod_tip = [0.18, -0.12, 0.96]
        rod_axis_into_branch = [0.58, 0.25, -0.775]
        rod_gravity_down = [0.0, 0.0, -1.0]

        res = fo.align_in_hand_feature_to_target(
            grasp_pos_m=grasp_pos,
            grasp_angles_deg=grasp_ang,
            feature_point_m=handle_center,
            feature_axis=handle_normal,
            target_point_m=rod_tip,
            target_axis=rod_axis_into_branch,
            feature_secondary_axis=handle_cup_down,
            target_secondary_axis=rod_gravity_down,
        )
        self.assertLess(res["feature_pos_err_m"], 1e-6)
        self.assertLess(res["feature_axis_err_deg"], 1e-5)

        p_deg, r_deg, y_deg = res["target_angles_deg"]
        self.assertTrue(-180.0 <= p_deg <= 180.0)
        self.assertTrue(-90.0 <= r_deg <= 90.0)
        self.assertTrue(-180.0 <= y_deg <= 180.0)

        rot_g0 = quat_wxyz_to_matrix(angles_deg_to_quat_wxyz(*grasp_ang))
        rot_gt = quat_wxyz_to_matrix(angles_deg_to_quat_wxyz(p_deg, r_deg, y_deg))
        local_pt = rot_g0.T @ (np.asarray(handle_center) - np.asarray(grasp_pos))
        local_ax = rot_g0.T @ (np.asarray(handle_normal) / np.linalg.norm(handle_normal))
        world_pt_after = np.asarray(res["target_pos_m"]) + rot_gt @ local_pt
        world_ax_after = rot_gt @ local_ax
        rod_unit = np.asarray(rod_axis_into_branch) / np.linalg.norm(rod_axis_into_branch)
        np.testing.assert_allclose(world_pt_after, rod_tip, atol=1e-6)
        np.testing.assert_allclose(world_ax_after, rod_unit, atol=1e-6)

        # 180 deg 反平行翻转 + 显式绕轴滚转角控制
        anti = fo.align_in_hand_feature_to_target(
            grasp_pos_m=[0.0, -0.18, 0.88],
            grasp_angles_deg=[0.0, 0.0, 0.0],
            feature_point_m=[0.0, -0.18, 0.84],
            feature_axis=[0.0, 0.0, 1.0],
            target_point_m=[0.12, -0.18, 0.82],
            target_axis=[0.0, 0.0, -1.0],
            roll_about_target_axis_deg=37.5,
        )
        self.assertLess(anti["feature_pos_err_m"], 1e-6)
        self.assertLess(anti["feature_axis_err_deg"], 1e-5)

        with self.assertRaises(ValueError):
            fo.align_in_hand_feature_to_target(grasp_pos, grasp_ang, handle_center, [0.0, 0.0, 0.0], rod_tip, rod_axis_into_branch)
        with self.assertRaises(ValueError):
            fo.align_in_hand_feature_to_target(
                grasp_pos, grasp_ang, handle_center, [1.0, 0.0, 0.0], rod_tip, [0.0, 0.0, -1.0],
                feature_secondary_axis=[2.0, 0.0, 0.0], target_secondary_axis=[0.0, 1.0, 0.0],
            )

    def test_ik_prefilter_dynamic_robot_model_no_default_table_and_collision_rejection(self) -> None:
        """验证动态解析自定义 6R `robot_model`、零默认 `table_z_m` 假设、流形扰动毫秒级 IK 与双臂胶囊碰撞拦截。"""
        # 1) 构造非 ARX X5 参数的自定义 6R 串联机械臂（含斜交旋转轴与收紧关节限位），对照宿主 geometry.fk
        custom_wb = np.eye(4, dtype=np.float64)
        custom_wb[:3, 3] = [0.25, -0.40, 0.60]
        custom_joints = [
            {"name": "right_joint1", "type": "revolute", "parent": "base", "child": "l1", "xyz": [0.0, 0.0, 0.08], "rpy": [0.0, 0.0, 0.0], "axis": [0.0, 0.0, 1.0], "limits_rad": [-2.6, 2.6]},
            {"name": "right_joint2", "type": "revolute", "parent": "l1", "child": "l2", "xyz": [0.03, 0.0, 0.06], "rpy": [0.0, 0.0, 0.0], "axis": [0.0, 1.0, 0.0], "limits_rad": [-2.2, 2.2]},
            {"name": "right_joint3", "type": "revolute", "parent": "l2", "child": "l3", "xyz": [0.28, 0.0, 0.04], "rpy": [0.0, 0.0, 0.0], "axis": [0.0, 1.0, 0.0], "limits_rad": [-2.4, 2.4]},
            {"name": "right_joint4", "type": "revolute", "parent": "l3", "child": "l4", "xyz": [0.26, 0.0, 0.02], "rpy": [0.0, 0.0, 0.0], "axis": [0.0, 0.8, 0.6], "limits_rad": [-2.5, 2.5]},
            {"name": "right_joint5", "type": "revolute", "parent": "l4", "child": "l5", "xyz": [0.09, 0.0, 0.04], "rpy": [0.0, 0.0, 0.0], "axis": [0.0, 0.0, 1.0], "limits_rad": [-2.5, 2.5]},
            {"name": "right_joint6", "type": "revolute", "parent": "l5", "child": "right_link6", "xyz": [0.08, 0.0, 0.03], "rpy": [0.0, 0.0, 0.0], "axis": [1.0, 0.0, 0.0], "limits_rad": [-1.8, 1.8]},
        ]
        for j in custom_joints:
            j["T_parent_joint"] = ikp._rpy_xyz_to_matrix(tuple(j["xyz"]), tuple(j["rpy"])).tolist()
        custom_model = {"base_link": "base", "T_world_base": custom_wb.tolist(), "joints": custom_joints}

        spec = ikp.resolve_6r_chain_spec("right", robot_model=custom_model)
        self.assertEqual(spec.joint_names[0], "right_joint1")
        self.assertAlmostEqual(float(spec.joint_limits_rad[5, 1]), 1.8, places=6)

        rng = np.random.default_rng(17)
        for _ in range(8):
            q_sample = rng.uniform(-0.85, 0.85, size=6)
            pos_b, rot_b, _, _ = ikp.fk_batch_base(q_sample, chain_spec=spec)
            pos_w = custom_wb[:3, :3] @ pos_b[0] + custom_wb[:3, 3]
            rot_w = custom_wb[:3, :3] @ rot_b[0]
            poses_ref = geometry.fk(
                custom_model,
                {"joint_names": [j["name"] for j in custom_joints], "q_rad": q_sample.tolist(), "source_kind": "measurement"},
            )
            t_ref = poses_ref["right_link6"]
            np.testing.assert_allclose(pos_w, t_ref[:3, 3], atol=1e-12)
            np.testing.assert_allclose(rot_w, t_ref[:3, :3], atol=1e-12)

            # 对该自定义 6R 模型目标末端抓取位姿做逆运动学反解验证
            grasp_pos_target = pos_w + rot_w @ np.array([ikp.GRASP_POINT_OFFSET_M, 0.0, 0.0])
            target_ang = fo.quat_wxyz_to_angles_deg(fo.matrix_to_quat_wxyz(rot_w))
            ik_custom = ikp.check_pose_reachability(
                "right",
                grasp_pos_target.tolist(),
                list(target_ang),
                q_seed=(q_sample + 0.25).tolist(),
                robot_model=custom_model,
            )
            self.assertTrue(ik_custom["reachable"], ik_custom)
            self.assertLess(ik_custom["pos_err_m"], 0.002)

        # 2) 验证默认 table_z_m=None 绝不假设 0.765 m 桌面拦截低位架下操作，显式传 table_z_m=0.765 时严格拦截潜台面
        low_pose_pos = [-0.18, -0.20, 0.72]
        low_pose_ang = [180.0, 0.0, 0.0]
        res_no_table = ikp.check_pose_reachability("left", low_pose_pos, low_pose_ang)
        self.assertTrue(res_no_table["reachable"], res_no_table)

        res_with_table = ikp.check_pose_reachability("left", low_pose_pos, low_pose_ang, table_z_m=0.765)
        self.assertFalse(res_with_table["reachable"])
        self.assertEqual(res_with_table["reason"], "arm_body_below_table")

        # 3) 预热并测试单点耗时（并行负载下取 3 组最小均值，确保毫秒级量级）
        ikp.check_pose_reachability("right", [0.16, -0.14, 0.83], [0.0, 0.0, 0.0])
        batch_avgs = []
        ok_pose = None
        for _ in range(3):
            t0 = time.perf_counter()
            n_runs = 15
            for _ in range(n_runs):
                ok_pose = ikp.check_pose_reachability("right", [0.16, -0.14, 0.83], [25.0, -15.0, -35.0])
            batch_avgs.append((time.perf_counter() - t0) * 1000.0 / n_runs)
        avg_single_ms = min(batch_avgs)
        self.assertTrue(ok_pose["reachable"], ok_pose)
        self.assertLess(ok_pose["pos_err_m"], 0.002)
        self.assertLess(ok_pose["rot_err_rad"], 0.02)
        self.assertLess(avg_single_ms, 8.0, f"单次 6R IK 预筛耗时偏高: {avg_single_ms:.3f} ms")

        far_res = ikp.check_pose_reachability("left", [0.95, 0.70, 1.35], [0.0, 0.0, 0.0])
        self.assertFalse(far_res["reachable"])
        self.assertEqual(far_res["reason"], "workspace_out_of_reach")

        oob_ang = ikp.check_pose_reachability("left", [-0.18, -0.20, 0.84], [0.0, 95.0, 0.0])
        self.assertFalse(oob_ang["reachable"])
        self.assertEqual(oob_ang["reason"], "angles_out_of_canonical_bounds")

        # 4) 批次双臂安全协同 vs 交叉碰撞拦截
        clean_stages = [
            {"id": "l1", "kind": "move", "arm": "left", "grasp_position_m": [-0.18, -0.15, 0.86], "angles_deg": [0.0, 0.0, 0.0]},
            {"id": "l2", "kind": "move", "arm": "left", "grasp_position_m": [-0.18, -0.15, 0.80], "angles_deg": [0.0, 0.0, 0.0], "depends_on": ["l1"]},
            {"id": "lg", "kind": "gripper", "arm": "left", "goal_normalized": 0.0, "depends_on": ["l2"]},
            {"id": "r1", "kind": "move", "arm": "right", "grasp_position_m": [0.18, -0.15, 0.86], "angles_deg": [0.0, 0.0, 0.0]},
        ]
        pf_ok = ikp.prefilter_stages_kinematics(clean_stages)
        self.assertTrue(pf_ok["valid"], pf_ok)
        self.assertEqual(pf_ok["blocking_stage_ids"], [])
        self.assertGreater(pf_ok["min_bimanual_clearance_m"], 0.15)

        collide_stages = [
            {"id": "lc", "kind": "move", "arm": "left", "grasp_position_m": [0.11, -0.18, 0.84], "angles_deg": [0.0, 0.0, -30.0]},
            {"id": "rc", "kind": "move", "arm": "right", "grasp_position_m": [-0.11, -0.18, 0.84], "angles_deg": [0.0, 0.0, 30.0]},
        ]
        pf_col = ikp.prefilter_stages_kinematics(collide_stages)
        self.assertFalse(pf_col["valid"])
        self.assertIn("bimanual_arm_collision", pf_col["reasons"])
        self.assertEqual(set(pf_col["blocking_stage_ids"]), {"lc", "rc"})
        self.assertLess(pf_col["min_bimanual_clearance_m"], 0.0)

    def test_sandbox_zero_pythonpath_execution_with_se3_path_and_ik_prefilter(self) -> None:
        """验证 `ProgramRunner` 隔离沙盒在剥离外部 `PYTHONPATH` 下自包含运行 `se3_path` 与 `ik_prefilter`。"""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        run_dir = Path(tmp.name) / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        store = Store(run_dir, dict(IDENTITY))
        self.addCleanup(store.close)
        class _BatchReadyThreeCamRobot(CalibratedThreeCamRobot):
            def start(self) -> dict:
                caps = super().start()
                caps["unified_action"] = {"available": True}
                return caps

        robot = _BatchReadyThreeCamRobot(Path(tmp.name) / "captures")
        cfg = Config.from_dict(
            {
                "features": {
                    "external_perception": True,
                    "scene3d": True,
                    "scene3d_observe": True,
                    "object_table": True,
                    "object_table_observe": False,
                    "motion_review": False,
                    "sandbox_enabled": expect(program_isolation, "program_run runs under the Landlock + seccomp sandbox"),
                }
            }
        )
        harness = Harness(store, robot, config=cfg, observation_profile="rgb_proprio_calibrated_v1")
        install_registry(harness, cfg)
        harness.start()
        obs_bundle = harness.current_observation()
        obs_id = obs_bundle["observation_id"]
        harness.task_planning.call(
            {"operation": "submit", "plan": "测试轻量沙盒与物理执行闭环", "ledger": []},
            harness.identity,
            obs_bundle,
            {},
        )

        self.assertEqual(scene3d.workspace_modules(harness), ("scene3d_helpers.py",))

        cloud_bytes = _build_surface_cloud_npz(center_xy=(0.16, -0.15), base_z=0.765)
        art_ref = store.put_artifact(cloud_bytes, scene3d.ARTIFACT_KIND, scene3d.ARTIFACT_MIME)
        services = replace(harness.program_services(), workspace_modules=("scene3d_helpers.py",))
        runner = ProgramRunner(services)
        self.addCleanup(runner.close)
        harness.programs = runner

        program_src = f"""
import json
import os
import scene3d_helpers as s3
from ni_rua_client import NiRuaClient

client = NiRuaClient()
s3.load({art_ref!r}, client=client)

reg = s3.align_feature_to_target(
    grasp_pos_m=[0.12, -0.22, 0.86],
    grasp_angles_deg=[0.0, 0.0, 0.0],
    feature_point_m=[0.12, -0.22, 0.83],
    feature_axis=[0.0, 0.0, -1.0],
    target_point_m=[0.16, -0.15, 0.807],
    target_axis=[0.0, 0.0, -1.0],
)
wps = s3.se3_screw(
    (reg["target_grasp_position_m"], reg["target_angles_deg"]),
    axis=[0.0, 0.0, -1.0],
    distance_m=0.018,
    n_steps=2,
)
stages = [
    {{"id": "approach", "kind": "move", "arm": "right", "grasp_position_m": [0.16, -0.15, 0.88], "angles_deg": reg["target_angles_deg"]}},
    {{"id": "insert", "kind": "move", "arm": "right", "grasp_position_m": reg["target_pos_m"], "angles_deg": reg["target_angles_deg"], "depends_on": ["approach"], "stall_guard_preset": "contact_stop", "max_upward_dz_m": 0.03}},
]
reach = s3.check_reachability("right", reg["target_pos_m"], reg["target_angles_deg"])
pf = s3.prefilter_stages(stages)

print(json.dumps({{
    "workspace_dir": os.getcwd(),
    "align_err_m": reg["feature_pos_err_m"],
    "has_alias_key": reg["target_grasp_position_m"] == reg["target_pos_m"],
    "wp_count": len(wps),
    "stage_count": len(stages),
    "stages": stages,
    "reach_ok": reach["reachable"],
    "prefilter_valid": pf["valid"],
}}))
"""
        job = runner.run(
            {"source": program_src, "timeout_s": 30.0},
            {"thread_id": "thread-s2", "turn_id": "turn-s2", "tool_call_id": "prog-s2"},
        )
        self.assertEqual(job["status"], "succeeded", job)
        _, stdout_bytes = store.resolve_artifact(job["stdout_ref"])
        payload = json.loads(stdout_bytes.decode("utf-8").strip())
        self.assertLess(payload["align_err_m"], 1e-6)
        self.assertTrue(payload["has_alias_key"])
        self.assertEqual(payload["wp_count"], 2)
        self.assertEqual(payload["stage_count"], 2)
        self.assertTrue(payload["reach_ok"])
        self.assertTrue(payload["prefilter_valid"])

        from ni_rua.core.contracts import new_id
        from ni_rua.runtime.actions.unified_actions import compile_request

        ins_stages = payload["stages"]
        for st in ins_stages:
            _validate_stage(st)
        compiled_action = compile_request(
            harness,
            {"mode": "preview", "observation_id": obs_id, "stages": ins_stages},
        )
        compiled_action["action_id"] = new_id("act")
        exec_res = harness.execute(compiled_action)
        self.assertEqual(exec_res["receipt"]["status"], "completed", exec_res)

        ws_dir = Path(payload["workspace_dir"])
        clean_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import math, scene3d_helpers as s3; "
                    "arc = s3.se3_screw(([0.16, -0.18, 0.84], [0.0, 0.0, 0.0]), "
                    "axis=[1.0, 0.0, 0.0], pivot_m=[0.16, -0.08, 0.84], angle_rad=math.pi/4, n_steps=4); "
                    "stages = [{'id': f's{i}', 'kind': 'move', 'arm': 'right', "
                    "'grasp_position_m': w['grasp_position_m'], 'angles_deg': w['angles_deg']} for i, w in enumerate(arc)]; "
                    "pf = s3.prefilter_stages(stages); "
                    "assert pf['valid'] and len(arc) == 4"
                ),
            ],
            cwd=str(ws_dir),
            env=clean_env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_robot_action_stage_schema_hygiene_and_explainability_docstrings(self) -> None:
        """验证 `ROBOT_ACTION_TOOL` 阶段参数无移除字段残留，公开接口完整写明中文物理单位与调用示例，且返回键与提示词严格对齐。"""
        import re
        from ni_rua.core.config import Config
        from ni_rua.perception.external import obs3d, scene3d

        self.assertEqual(list(s3.API), list(scene3d.API))
        for fn in (
            fo.align_in_hand_feature_to_target,
            fo.query_spatial_clearance_and_gradient,
            ikp.resolve_6r_chain_spec,
            ikp.check_pose_reachability,
            ikp.prefilter_stages_kinematics,
            s3.align_feature_to_target,
            s3.se3_screw,
            s3.cavity_profile,
            s3.in_hand_axis,
            s3.check_reachability,
            s3.solve_reachable_pose,
            s3.plan_joint_safe_return,
            s3.prefilter_stages,
        ):
            doc = fn.__doc__ or ""
            self.assertIn("参数:", doc, f"{fn.__name__} 缺少参数说明")
            self.assertIn("返回:", doc, f"{fn.__name__} 缺少返回说明")
            self.assertIn("示例:", doc, f"{fn.__name__} 缺少调用示例")
            self.assertTrue(("`m`" in doc) or ("`rad`" in doc) or ("`deg`" in doc), f"{fn.__name__} 缺少显式单位")

        stage_props = set(ROBOT_ACTION_TOOL["inputSchema"]["properties"]["stages"]["items"]["properties"].keys())
        self.assertNotIn("funnel_sigma_m", stage_props)
        self.assertNotIn("repulsive_margin_m", stage_props)

        dev_doc = obs3d.developer_instructions(Config().features)
        match = re.search(r"align_feature_to_target\([^)]+\)`\s*\(returns `\{([^}]+)\}`", dev_doc)
        self.assertIsNotNone(match, dev_doc)
        documented_keys = [k.strip() for k in match.group(1).split(",") if k.strip()]
        actual_reg = s3.align_feature_to_target([0.20, -0.15, 0.85], [0.0, 0.0, 0.0], [0.20, -0.15, 0.80], [0.0, 0.0, 1.0], [0.10, 0.10, 0.85], [0.0, 0.0, 1.0])
        for key in documented_keys:
            self.assertIn(key, actual_reg, key)

    def test_jaw_symmetric_angles_safe_return_and_source_to_dest_reachability(self) -> None:
        """覆盖 jaw_symmetric_angles_deg、plan_joint_safe_return、solve_reachable_pose 倾角优先级与 check_source_to_dest。"""
        pos = [0.14, -0.18, 0.86]
        for ang in (
            [0.0, 0.0, -36.0],
            [90.0, 0.0, 0.0],
            [-90.0, 0.0, 45.0],
            [35.0, -22.5, 65.0],
            [0.0, 90.0, 0.0],
            [-179.5, -89.8, 179.5],
        ):
            twin = ikp.jaw_symmetric_angles_deg(ang)
            self.assertEqual(len(twin), 3)
            self.assertTrue(-180.0 - 1e-6 <= twin[0] <= 180.0 + 1e-6)
            self.assertTrue(-90.0 - 1e-6 <= twin[1] <= 90.0 + 1e-6)
            self.assertTrue(-180.0 - 1e-6 <= twin[2] <= 180.0 + 1e-6)
            f_orig = gripper_frame(pos, ang, contact_region="pad_center")
            f_twin = gripper_frame(pos, twin, contact_region="pad_center")
            np.testing.assert_allclose(f_orig["approach_dir"], f_twin["approach_dir"], atol=1e-4)
            np.testing.assert_allclose(f_orig["closing_dir"], -f_twin["closing_dir"], atol=1e-4)
            np.testing.assert_allclose(f_orig["flange_position_m"], f_twin["flange_position_m"], atol=1e-5)

        twin_90 = ikp.jaw_symmetric_angles_deg([90.0, 0.0, 0.0])
        self.assertAlmostEqual(twin_90[0], -90.0, places=3)
        self.assertAlmostEqual(abs(twin_90[2]), 180.0, places=3)
        with self.assertRaises(ValueError):
            ikp.jaw_symmetric_angles_deg([0.0, float("nan"), 0.0])

        # 真实连杆折叠 self_collision -> 绕指间闭合轴微倾逃逸（保持闭合轴严格共线）及超界不可达返回契约
        sc_pos = [-0.145, -0.056, 0.90]
        sc_bad_ang = [-20.0, 0.0, -36.0]
        chk_sc = ikp.check_pose_reachability("left", sc_pos, sc_bad_ang)
        self.assertFalse(chk_sc["reachable"])
        self.assertEqual(chk_sc["reason"], "self_collision")
        sol_sc = ikp.solve_reachable_pose(
            "left",
            sc_pos,
            sc_bad_ang,
            approach_tilt_offsets_deg=(0.0, -20.0, 20.0, -40.0, 40.0),
        )
        self.assertTrue(sol_sc["reachable"])
        self.assertGreater(sol_sc["self_clearance_m"], 0.02)
        f_sc_bad = gripper_frame(sc_pos, sc_bad_ang)
        f_sc_sel = gripper_frame(sc_pos, sol_sc["selected_angles_deg"])
        self.assertAlmostEqual(abs(float(np.dot(f_sc_bad["closing_dir"], f_sc_sel["closing_dir"]))), 1.0, places=4)

        far_unreach = ikp.solve_reachable_pose("right", [2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
        self.assertFalse(far_unreach["reachable"])
        self.assertIsNone(far_unreach["selected_angles_deg"])
        self.assertEqual(far_unreach["reason"], "workspace_out_of_reach")

        def flange_from_q(arm: str, q_rad: list[float]) -> dict:
            q_arr = np.asarray(q_rad, dtype=np.float64)
            t_wb = ikp.resolve_arm_base_transform(arm, None)
            pos_b, rot_b, _, _ = ikp.fk_batch_base(q_arr[None, :])
            pos_w = t_wb[:3, :3] @ pos_b[0] + t_wb[:3, 3]
            rot_w = t_wb[:3, :3] @ rot_b[0]
            qx, qy, qz, qw = Rotation.from_matrix(rot_w).as_quat()
            return {
                "q_rad": [float(v) for v in q_arr],
                "ee_pose": {"position_m": [float(v) for v in pos_w], "quaternion_xyzw": [float(qx), float(qy), float(qz), float(qw)]},
            }

        q_home = [0.0, 0.90, 0.85, -0.55, 0.0, 0.0]
        home_obs = {"robot_state": {"arms": {"left": flange_from_q("left", q_home), "right": flange_from_q("right", q_home)}}}
        cur_clean = {
            "arms": {
                "left": flange_from_q("left", [0.25, 1.05, 0.95, -0.60, 0.15, 0.20]),
                "right": flange_from_q("right", [-0.25, 1.05, 0.95, -0.60, -0.15, -0.20]),
            }
        }
        ret_clean = ikp.plan_joint_safe_return(home_obs, cur_clean)
        self.assertTrue(ret_clean["valid"])
        self.assertFalse(ret_clean["sequenced_bimanual"])
        self.assertEqual(len(ret_clean["stages"]), 2)

        q_wound_r = [float(q_home[0] + 6.283185), *q_home[1:]]
        ret_wound = ikp.plan_joint_safe_return(
            home_obs["robot_state"],
            {"arms": {"left": flange_from_q("left", q_home), "right": flange_from_q("right", q_wound_r)}},
            arms=("right",),
        )
        self.assertTrue(ret_wound["valid"])
        self.assertIn(ret_wound["recommended_joint_travel_preset"], ("extended", "unconstrained"))

        ret_direct = ikp.plan_joint_safe_return(
            {"left": home_obs["robot_state"]["arms"]["left"]["ee_pose"]},
            cur_clean,
            arms=("left",),
        )
        self.assertTrue(ret_direct["valid"])
        self.assertEqual(len(ret_direct["stages"]), 1)

        mid_to_right = ikp.check_source_to_dest([-0.132, -0.110, 0.872], [0.0, 0.0, -90.0], [0.160, -0.140, 0.950], [0.0, 0.0, 90.0])
        self.assertFalse(mid_to_right["requires_handoff"])
        self.assertEqual(mid_to_right["recommended_arm"], "right")

        far_left_to_right = s3.check_source_to_dest([-0.46, -0.05, 0.82], [0.0, 0.0, 0.0], [0.46, -0.05, 0.82])
        self.assertTrue(far_left_to_right["requires_handoff"])
        self.assertIsNone(far_left_to_right["recommended_arm"])
        self.assertEqual(far_left_to_right["single_arm_candidates"], [])

        both_out_of_reach = s3.check_source_to_dest([-0.20, -0.20, 0.80], [0.0, 0.0, 0.0], [2.40, 2.40, 2.10], [0.0, 0.0, 0.0])
        self.assertTrue(both_out_of_reach["requires_handoff"])
        self.assertIsNone(both_out_of_reach["recommended_arm"])
        self.assertEqual(both_out_of_reach["single_arm_candidates"], [])

        cross_slot = s3.check_source_to_dest([-0.30, -0.13, 0.78], [0.0, 0.0, 0.0], [0.2987, -0.1514, 0.898], [0.0, 0.0, 90.0])
        self.assertFalse(cross_slot["requires_handoff"])
        self.assertEqual(cross_slot["recommended_arm"], "left")
        ref_rot = quat_to_matrix(angles_deg_to_quat_wxyz(0.0, 0.0, 90.0))
        sel_ang = cross_slot["arms"]["left"]["dest_angles_deg"]
        sel_rot = quat_to_matrix(angles_deg_to_quat_wxyz(float(sel_ang[0]), float(sel_ang[1]), float(sel_ang[2])))
        self.assertAlmostEqual(abs(float(np.dot(ref_rot[:, 1], sel_rot[:, 1]))), 1.0, places=6)
        with self.assertRaises(ValueError):
            s3.check_source_to_dest([0.0, -0.2, 0.8], [0.0, 0.0, 0.0], [0.1, -0.2, 0.8], arms=())
        with self.assertRaises(ValueError):
            s3.check_source_to_dest([0.0, -0.1, 0.85], [0.0, 0.0, 0.0], [0.1, -0.1, 0.85], arms=("middle",))
        with self.assertRaises(ValueError):
            ikp.solve_reachable_pose("middle", pos, [0.0, 0.0, 0.0])

        seed_q = [0.0, 0.0, 0.0, 0.0, 0.0, -1.5]
        wide_limits = np.array([[-6.0, 6.0]] * 6, dtype=np.float64)

        def fake_pose_reach(arm, position_m, angles_deg=None, **kwargs):
            ang = [float(v) for v in angles_deg]
            is_flat = abs(ang[0]) < 1e-3 and abs(ang[1]) < 1e-3
            q_sol = [0.1, 0.2, 0.3, 0.2, 0.0, 2.7] if is_flat else [0.1, 0.2, 0.3, 0.2, 0.0, -0.3]
            return dict(
                reachable=True,
                reason="ok",
                joint_q_rad=q_sol,
                pos_err_m=0.0001,
                rot_err_rad=0.0001,
                min_joint_margin_rad=0.25,
                self_clearance_m=0.04,
                table_clearance_m=0.05,
                joint_travel_from_seed_rad=4.20 if is_flat else 1.20,
            )

        with unittest.mock.patch.object(ikp, "check_pose_reachability", side_effect=fake_pose_reach):
            solved = ikp.solve_reachable_pose("left", [0.28, 0.18, 0.80], [0.0, 0.0, 90.0], q_seed=seed_q, joint_limits_rad=wide_limits)
            self.assertTrue(solved["reachable"])
            self.assertAlmostEqual(solved["approach_tilt_deg"], 0.0, places=3)
            self.assertEqual(solved["recommended_joint_travel_preset"], "extended")

            wide_chain = ikp.resolve_6r_chain_spec("left", joint_limits_rad=wide_limits)
            rs_seed = dict(arms=dict(left=dict(q_rad=seed_q), right=dict(q_rad=seed_q)))
            s2d = ikp.check_source_to_dest(
                [0.28, 0.18, 0.80],
                [0.0, 0.0, 90.0],
                [0.28, -0.05, 0.82],
                [0.0, 0.0, 90.0],
                robot_model=wide_chain,
                robot_state=rs_seed,
            )
            self.assertFalse(s2d["requires_handoff"])
            best_arm = s2d["recommended_arm"]
            self.assertAlmostEqual(s2d["arms"][best_arm]["source"]["approach_tilt_deg"], 0.0, places=3)

        def fake_only_tilted(arm, position_m, angles_deg=None, **kwargs):
            ang = [float(v) for v in angles_deg]
            is_flat = abs(ang[0]) < 1e-3 and abs(ang[1]) < 1e-3
            if is_flat:
                return dict(
                    reachable=False,
                    reason="ik_failed",
                    joint_q_rad=None,
                    pos_err_m=0.05,
                    rot_err_rad=0.4,
                    min_joint_margin_rad=None,
                    self_clearance_m=None,
                    table_clearance_m=None,
                    joint_travel_from_seed_rad=None,
                )
            return dict(
                reachable=True,
                reason="ok",
                joint_q_rad=[0.1, 0.2, 0.3, 0.2, 0.0, 1.0],
                pos_err_m=0.0001,
                rot_err_rad=0.0001,
                min_joint_margin_rad=0.25,
                self_clearance_m=0.04,
                table_clearance_m=0.05,
                joint_travel_from_seed_rad=1.0,
            )

        with unittest.mock.patch.object(ikp, "check_pose_reachability", side_effect=fake_only_tilted):
            fallback = ikp.solve_reachable_pose("left", [0.45, 0.25, 0.78], [0.0, 0.0, 0.0], q_seed=seed_q)
            self.assertTrue(fallback["reachable"])
            self.assertGreaterEqual(abs(fallback["approach_tilt_deg"]), 15.0)

        # 覆盖贴桌面抓取放行 vs 探底穿桌拦截、真实 SO(3) 测地偏角与 s3.jaw_symmetric_angles_deg
        self.assertEqual(s3.jaw_symmetric_angles_deg([0.0, 0.0, -36.0]), ikp.jaw_symmetric_angles_deg([0.0, 0.0, -36.0]))
        flush_grasp = ikp.check_pose_reachability("right", [0.16, -0.18, 0.775], [0.0, 0.0, -30.0], table_z_m=0.765)
        self.assertTrue(flush_grasp["reachable"], flush_grasp)
        self.assertGreaterEqual(flush_grasp["self_clearance_m"], 0.0)
        self.assertGreaterEqual(flush_grasp["table_clearance_m"], 0.0)
        below_table = ikp.check_pose_reachability("right", [0.16, -0.18, 0.600], [0.0, 0.0, -30.0], table_z_m=0.765)
        self.assertFalse(below_table["reachable"])
        self.assertEqual(below_table["reason"], "arm_body_below_table")
        geod_sol = ikp.solve_reachable_pose(
            arm="left",
            position_m=[-0.18, 0.12, 0.32],
            angles_deg=[0.0, 20.0, -90.0],
            approach_tilt_offsets_deg=(0.0,),
            candidate_angles_deg=[[0.0, 70.0, -90.0]],
        )
        tilted_entries = [c for c in geod_sol["candidates"] if c["angles_deg"] == [0.0, 70.0, -90.0]]
        self.assertEqual(len(tilted_entries), 1)
        self.assertAlmostEqual(tilted_entries[0]["orientation_deviation_deg"], 50.0, delta=0.2)

    def test_pose_math_pure_se3_screw_and_auto_focus_roi_fingertip_center(self) -> None:
        """验证无状态 pose_math.py、纯函数 s3.se3_screw 刚体采样以及 _action_focus_roi 指尖居中。"""
        for preset in ("normal", "tight", "compliant", "coarse", "seating"):
            _validate_stage(
                fo._build_stage_move_dict(
                    stage_id=f"m_{preset}",
                    arm="right",
                    grasp_pos_m=np.array([0.18, -0.15, 0.84]),
                    angles_deg=[0.0, 0.0, -30.0],
                    tolerance_preset=preset,
                )
            )
        q_wxyz = pm.angles_deg_to_quat_wxyz(35.0, -15.0, -40.0)
        q_xyzw = [float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3]), float(q_wxyz[0])]
        flange_pos = [0.16, -0.18, 0.94]
        converted = pm.flange_pose_to_grasp_pose(flange_pos, q_xyzw)
        self.assertAlmostEqual(
            float(np.linalg.norm(np.asarray(converted["grasp_position_m"]) - np.asarray(flange_pos))),
            pm.GRASP_POINT_OFFSET_M,
            places=6,
        )
        np.testing.assert_allclose(converted["angles_deg"], [35.0, -15.0, -40.0], atol=1e-3)

        wps = s3.se3_screw(
            ([0.20, -0.20, 0.80], [0.0, 0.0, 0.0]),
            axis=[1.0, 0.0, 0.0],
            pivot_m=[0.20, -0.10, 0.80],
            angle_rad=-math.pi / 2.0,
            pitch_m_per_rad=0.0,
            n_steps=4,
        )
        self.assertEqual(len(wps), 4)
        for wp_item in wps:
            rad = float(np.linalg.norm(np.asarray(wp_item["grasp_position_m"]) - np.array([0.20, -0.10, 0.80])))
            self.assertAlmostEqual(rad, 0.10, places=3)
        np.testing.assert_allclose(wps[-1]["grasp_position_m"], [0.20, -0.10, 0.90], atol=2e-4)
        with self.assertRaises(ValueError):
            s3.se3_screw(([0.2, -0.2, 0.8], [0, 0, 0]), [0.0, 0.0, 0.0], pivot_m=[0.2, -0.1, 0.8], angle_rad=0.5)

        closing = _extract_closing_grasp_stages(
            {"stages": [{"id": "pick_comp", "kind": "move", "arm": "right", "grasp_position_m": [0.15, -0.10, 0.78], "angles_deg": [0.0, 0.0, 0.0], "gripper_goal_normalized": 0.0, "tolerance_preset": "compliant"}]}
        )
        self.assertTrue(closing[0]["has_vertical_compliance"])

        q_topdown_wxyz = pm.angles_deg_to_quat_wxyz(0.0, 0.0, 0.0)
        q_topdown_xyzw = [float(q_topdown_wxyz[1]), float(q_topdown_wxyz[2]), float(q_topdown_wxyz[3]), float(q_topdown_wxyz[0])]
        t_wc = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, -0.80], [0.0, -1.0, 0.0, 0.80], [0.0, 0.0, 0.0, 1.0]], dtype=float)
        metadata = {
            "camera_calibration": {"cameras": {"cam_head": {"K": [[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]], "T_world_camera": t_wc.tolist()}}},
            "robot_state": {"arms": {"right": {"ee_pose": {"position_m": [0.0, -0.20, 0.80 + pm.GRASP_POINT_OFFSET_M], "quaternion_xyzw": q_topdown_xyzw, "tcp_config_id": "arx-x5-link6-v1"}}}},
        }
        self.assertEqual(_action_focus_roi(metadata, {"camera_id": "cam_head", "width": 640, "height": 480}, {"stages": [{"arm": "right"}]}, patch_px=200), [220, 140, 200, 200])



if __name__ == "__main__":
    unittest.main()
