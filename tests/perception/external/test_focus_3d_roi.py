"""`focus_3d` 三维球域到多相机像素 ROI 精确反投影、坐标互转、无图观测及原生哈希审计单元测试。"""
from __future__ import annotations

NI_RUA_TEST_TIER = "component"

import base64
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

from ni_rua.apps.evaluation.audit import audit
from ni_rua.core.config import Config
from ni_rua.core.contracts import canonical
from ni_rua.core.presentation.actor_output import ActorOutput
from ni_rua.core.presentation.actor_presentation import ActorPresentation, metadata_findings
from ni_rua.core.storage.store import Store
from ni_rua.cortex.session.wire.openai_responses import OpenAIResponsesWire
from ni_rua.perception.external import obs3d
from ni_rua.perception.focus_3d import (
    DEFAULT_PATCH_PX,
    DEFAULT_RADIUS_M,
    full_to_local_uv,
    local_to_full_uv,
    plan_focus_3d_views,
    project_sphere_to_camera,
)
from ni_rua.robot.backends.robodojo.robodojo import FRAME_ID, TCP_CONFIG_ID
from ni_rua.runtime.harness import Harness
from ni_rua.runtime.tools.builtins import create_registry
from ni_rua.runtime.tools.definitions import public_definitions
from support.tool_fixture import install_registry


def _wire_image_labels(content_items: list[dict]) -> list[str | None]:
    """Resolve ``<obs_id>/<cam_id>`` labels via ``OpenAIResponsesWire.part_observation_labels``."""
    wire_parts = [
        {"type": "input_image", "image_url": item["imageUrl"]}
        if item.get("type") == "inputImage"
        else item
        for item in content_items
    ]
    return [
        f"{obs_id}/{cam_id}" if obs_id and cam_id else None
        for obs_id, cam_id in OpenAIResponsesWire.part_observation_labels(wire_parts)
    ]


def _look_at_world_camera(cam_pos_w, target_w, up_w=(0.0, 0.0, 1.0)) -> np.ndarray:
    """构造 OpenCV 光学约定 (+X 右, +Y 下, +Z 光轴向前) 的相机世界位姿矩阵 `T_world_camera`。"""
    c = np.asarray(cam_pos_w, dtype=np.float64)
    t = np.asarray(target_w, dtype=np.float64)
    up = np.asarray(up_w, dtype=np.float64)
    z_axis = t - c
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(z_axis, up)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.cross(z_axis, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, 0] = x_axis
    mat[:3, 1] = y_axis
    mat[:3, 2] = z_axis
    mat[:3, 3] = c
    return mat


def _pinhole_k(fx: float = 520.0, fy: float = 520.0, cx: float = 320.0, cy: float = 240.0) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


class CalibratedThreeCamRobot:
    """确定性三相机（头部固定 + 左右腕部实时关节 FK）合成标定机器人夹具，不依赖仿真器或 GPU。"""

    def __init__(self, root: Path, *, width: int = 640, height: int = 480, marker_world_m=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.executions = 0
        self.q_left = 0.0
        self.q_right = 0.0
        self.marker_world_m = list(marker_world_m) if marker_world_m is not None else [0.03, -0.36, 0.82]

        self.k_mat = _pinhole_k(520.0, 520.0, width * 0.5, height * 0.5)
        # 头部相机固定俯视工作台中心 [0.0, -0.35, 0.80]
        self.t_head = _look_at_world_camera([0.0, -0.05, 1.28], [0.0, -0.35, 0.80])
        # 左腕基座靠近目标，沿世界 +X 单轴平移关节 j1
        self.t_left_base = np.eye(4, dtype=np.float64)
        self.t_left_base[:3, 3] = [-0.14, -0.22, 0.96]
        self.t_left_link_cam = np.linalg.inv(self.t_left_base) @ _look_at_world_camera(
            [-0.14, -0.22, 0.96], [0.0, -0.35, 0.80]
        )
        # 右腕默认朝向远离目标的右侧外部，用于验证视锥外自动过滤与显式报错
        self.t_right_base = np.eye(4, dtype=np.float64)
        self.t_right_base[:3, 3] = [0.42, -0.10, 1.05]
        self.t_right_link_cam = np.linalg.inv(self.t_right_base) @ _look_at_world_camera(
            [0.42, -0.10, 1.05], [0.85, 0.35, 1.10]
        )

    def _wrist_world_pose(self, base_mat: np.ndarray, q_shift_x: float, link_cam: np.ndarray) -> np.ndarray:
        motion = np.eye(4, dtype=np.float64)
        motion[0, 3] = float(q_shift_x)
        return base_mat @ motion @ link_cam

    def _render_sensor_png(self, cam_id: str, t_wc: np.ndarray) -> Path:
        """生成带高频像素网格纹理与精确三维地标投影亮斑的无损传感器 PNG。"""
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        # 1 像素级高频亮度纹理，任何先缩全图再放大的操作都会彻底抹平像素阶跃
        checker = (((xx + yy) % 2) * 110).astype(np.uint8)
        r = np.clip(40 + checker, 0, 255).astype(np.uint8)
        g = np.clip(40 + checker, 0, 255).astype(np.uint8)
        b = np.clip(40 + checker, 0, 255).astype(np.uint8)
        rgb = np.stack([r, g, b], axis=-1)

        # 将已知三维特征点投影至当前相机画面并绘制高对比白色十字中心
        inv = np.linalg.inv(t_wc)
        pt_w = np.asarray(self.marker_world_m, dtype=np.float64)
        pt_c = inv[:3, :3] @ pt_w + inv[:3, 3]
        if pt_c[2] > 0.02:
            uv = cv2.projectPoints(
                pt_c.reshape(1, 1, 3),
                np.zeros((3, 1)),
                np.zeros((3, 1)),
                self.k_mat,
                np.zeros(5),
            )[0].reshape(2)
            u_px, v_px = int(round(float(uv[0]))), int(round(float(uv[1])))
            if 2 <= u_px < self.width - 2 and 2 <= v_px < self.height - 2:
                rgb[v_px - 2 : v_px + 3, u_px - 2 : u_px + 3] = np.array([210, 30, 30], dtype=np.uint8)
                rgb[v_px - 1 : v_px + 2, u_px - 1 : u_px + 2] = np.array([240, 10, 10], dtype=np.uint8)
                rgb[v_px, u_px] = np.array([255, 0, 0], dtype=np.uint8)

        out = self.root / f"{cam_id}_step_{self.executions}_{self.q_left:.3f}.png"
        Image.fromarray(rgb, mode="RGB").save(out, format="PNG")
        return out

    def start(self) -> dict:
        return dict(
            data_origin="synthetic_fixture",
            robot_model_id="arx-x5",
            observation_profile="rgb_proprio_calibrated_v1",
            frame_id=FRAME_ID,
            tcp_config_id=TCP_CONFIG_ID,
            path_kinds=["backend_free"],
            max_waypoints=4,
            dual_arm=True,
            gripper_after_motion=True,
        )

    def observe(self) -> dict:
        t_left = self._wrist_world_pose(self.t_left_base, self.q_left, self.t_left_link_cam)
        t_right = self._wrist_world_pose(self.t_right_base, self.q_right, self.t_right_link_cam)
        k_list = self.k_mat.tolist()
        size = [self.width, self.height]

        cam_defs = [
            ("cam_head", self.t_head, "fixed_calibration"),
            ("cam_left_wrist", t_left, "robot_kinematics"),
            ("cam_right_wrist", t_right, "robot_kinematics"),
        ]
        cameras = []
        pose_cameras = {}
        for cid, t_wc, src in cam_defs:
            png = self._render_sensor_png(cid, t_wc)
            cameras.append(dict(
                camera_id=cid,
                png_path=str(png),
                width=self.width,
                height=self.height,
            ))
            pose_cameras[cid] = dict(
                T_world_camera=t_wc.tolist(),
                source_kind=src,
            )

        calibration = dict(
            profile_id="calibrated_fixture_v1",
            schema_version="0.1",
            reference_frame_id=FRAME_ID,
            cameras=dict(
                cam_head=dict(
                    K=k_list,
                    distortion=[0.0, 0.0, 0.0, 0.0, 0.0],
                    image_size=size,
                    T_world_camera=self.t_head.tolist(),
                ),
                cam_left_wrist=dict(
                    K=k_list,
                    distortion=[0.0, 0.0, 0.0, 0.0, 0.0],
                    image_size=size,
                    arm="left",
                    link_name="link6",
                    T_world_base=self.t_left_base.tolist(),
                    T_link_camera=self.t_left_link_cam.tolist(),
                ),
                cam_right_wrist=dict(
                    K=k_list,
                    distortion=[0.0, 0.0, 0.0, 0.0, 0.0],
                    image_size=size,
                    arm="right",
                    link_name="link6",
                    T_world_base=self.t_right_base.tolist(),
                    T_link_camera=self.t_right_link_cam.tolist(),
                ),
            ),
        )
        robot_model = dict(
            base_link="base",
            T_world_base=np.eye(4, dtype=np.float64).tolist(),
            joints=[
                dict(
                    name="j1",
                    type="prismatic",
                    parent="base",
                    child="link6",
                    T_parent_joint=np.eye(4, dtype=np.float64).tolist(),
                    axis=[1.0, 0.0, 0.0],
                )
            ],
        )
        robot_state = dict(
            data_origin="synthetic_fixture",
            source_kind="measurement",
            sim_step=self.executions,
            step_domain="adapter_control_step",
            arms=dict(
                left=dict(
                    source_kind="measurement",
                    joint_names=["j1"],
                    q_rad=[float(self.q_left)],
                    gripper=dict(joint_names=["g1"], position_m=[0.04]),
                    ee_pose=dict(
                        position_m=[-0.14 + self.q_left, -0.22, 0.96],
                        quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
                        frame_id=FRAME_ID,
                        tcp_config_id=TCP_CONFIG_ID,
                        source_kind="measurement",
                    ),
                ),
                right=dict(
                    source_kind="measurement",
                    joint_names=["j1"],
                    q_rad=[float(self.q_right)],
                    gripper=dict(joint_names=["g1"], position_m=[0.04]),
                    ee_pose=dict(
                        position_m=[0.42 + self.q_right, -0.10, 1.05],
                        quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
                        frame_id=FRAME_ID,
                        tcp_config_id=TCP_CONFIG_ID,
                        source_kind="measurement",
                    ),
                ),
            ),
        )
        return dict(
            data_origin="synthetic_fixture",
            world_kind="observed",
            observation_profile="rgb_proprio_calibrated_v1",
            clock_domain="fixture-env/generation-1",
            sim_step=self.executions,
            step_domain="adapter_control_step",
            sim_time_s=float(self.executions) * 0.04,
            captured_at="2026-11-11T11:30:00Z",
            cameras=cameras,
            camera_calibration=calibration,
            camera_pose_sample=dict(
                sim_step=self.executions,
                step_domain="adapter_control_step",
                reference_frame_id=FRAME_ID,
                cameras=pose_cameras,
            ),
            robot_model=robot_model,
            robot_state=robot_state,
        )

    def execute(self, action, progress_callback=None, cancel_flag=None) -> dict:
        self.executions += 1
        if progress_callback:
            progress_callback(dict(phase="started", sim_step=self.executions, data_origin="synthetic_fixture"))
        obs = self.observe()
        return dict(
            status="completed",
            reason=None,
            executed_steps=1,
            executed_native_steps=1,
            completed_native_steps=1,
            data_origin="synthetic_fixture",
            terminal_observation=obs,
            planned_trace=dict(data_origin="synthetic_fixture", targets=action.get("path")),
            dispatched_trace=dict(data_origin="synthetic_fixture", steps=1),
            actual_trace=[dict(data_origin="synthetic_fixture", measured_fixture_state=obs["robot_state"])],
        )


class Focus3dGeometryUnitTests(unittest.TestCase):
    """纯几何测试：三维球域正轴/离轴透视投影精度、边界钳位与像素可逆换算。"""

    def test_sphere_projection_on_axis_and_off_axis_perspective_accuracy(self):
        k_mat = _pinhole_k(500.0, 500.0, 320.0, 240.0)
        t_wc = np.eye(4, dtype=np.float64)

        # 1. 光轴正前方球体：切锥精确半径 = f * r / sqrt(d^2 - r^2)
        d_m, r_m = 0.50, 0.08
        proj_on = project_sphere_to_camera([0.0, 0.0, d_m], r_m, k_mat, t_wc, 640, 480, patch_px=128)
        self.assertTrue(proj_on["in_frustum"])
        self.assertEqual(proj_on["projected_center_uv_px"], [320.0, 240.0])
        exact_half_px = 500.0 * r_m / math.sqrt(d_m * d_m - r_m * r_m)
        expected_x0 = math.floor(320.0 - exact_half_px)
        expected_w = math.ceil(320.0 + exact_half_px) - expected_x0
        self.assertEqual(proj_on["projected_sphere_xywh_px"][0], expected_x0)
        self.assertEqual(proj_on["projected_sphere_xywh_px"][2], expected_w)
        # 球投影直径 (~162 px) 大于 patch_px=128，裁剪框须自动扩展至完整包络球体
        self.assertGreaterEqual(proj_on["crop_xywh_px"][2], expected_w)

        # 2. 离轴球体（X=+0.22m, Z=0.50m）：透视投影为偏心椭圆，外侧像素跨度显著大于内侧与朴素 f*r/Z
        target_off = np.array([0.22, -0.14, 0.50], dtype=np.float64)
        proj_off = project_sphere_to_camera(target_off, 0.07, k_mat, t_wc, 640, 480, patch_px=128)
        self.assertTrue(proj_off["in_frustum"])
        u_c, v_c = proj_off["projected_center_uv_px"]
        sx, sy, sw, sh = proj_off["projected_sphere_xywh_px"]
        # 外侧 (+u) 拉伸幅度必须严格大于内侧 (-u)
        self.assertGreater((sx + sw) - u_c, u_c - sx)

        # 随机采样球面 1000 个三维点，验证 100% 落在 crop_xywh_px 内部
        rng = np.random.default_rng(42)
        dirs = rng.normal(size=(1000, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        surf_pts = target_off[None, :] + 0.07 * dirs
        uvs = cv2.projectPoints(surf_pts.reshape(-1, 1, 3), np.zeros((3, 1)), np.zeros((3, 1)), k_mat, np.zeros(5))[0].reshape(-1, 2)
        cx, cy, cw, ch = proj_off["crop_xywh_px"]
        self.assertTrue(np.all(uvs[:, 0] >= cx))
        self.assertTrue(np.all(uvs[:, 0] <= cx + cw))
        self.assertTrue(np.all(uvs[:, 1] >= cy))
        self.assertTrue(np.all(uvs[:, 1] <= cy + ch))

    def test_boundary_clamping_and_reversible_uv_helpers(self):
        k_mat = _pinhole_k(500.0, 500.0, 320.0, 240.0)
        t_wc = np.eye(4, dtype=np.float64)
        # 目标靠近左上角边缘 (u_c ≈ 20, v_c ≈ 20)，patch_px=256 会触碰左/上边界并钳位
        proj = project_sphere_to_camera([-0.30, -0.22, 0.50], 0.04, k_mat, t_wc, 640, 480, patch_px=256)
        self.assertTrue(proj["in_frustum"])
        self.assertTrue(proj["clamped_to_bounds"])
        x, y, w, h = proj["crop_xywh_px"]
        self.assertEqual([x, y, w, h], [0, 0, 256, 256])

        # 局部与原图坐标互逆验证（含矩阵 pixel_transform 一致性）
        crop_box = [132, 84, 256, 256]
        local_pt = [77.5, 129.25]
        full_pt = local_to_full_uv(crop_box, local_pt)
        self.assertEqual(full_pt, [209.5, 213.25])
        self.assertEqual(full_to_local_uv(crop_box, full_pt), local_pt)

        proj_mid = project_sphere_to_camera([0.0, 0.0, 0.60], 0.04, k_mat, t_wc, 640, 480, patch_px=256)
        m = np.asarray(proj_mid["pixel_transform"], dtype=np.float64)
        hom_full = np.array([full_pt[0], full_pt[1], 1.0])
        mapped_local = (m @ hom_full)[:2].tolist()
        self.assertEqual(mapped_local, full_to_local_uv(proj_mid["crop_xywh_px"], full_pt))


class Focus3dHarnessAndAuditTests(unittest.TestCase):
    """端到端测试：实时腕部 FK、`object_id` 解析、1:1 无损像素一致性、立体三角化闭环与 `audit(profile='native')`。"""

    def setUp(self):
        self.tmp_dir = self.enterContext(tempfile.TemporaryDirectory())
        self.run_dir = Path(self.tmp_dir) / "run"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.identity = dict(
            run_id="focus3d-run",
            episode_id="ep-1",
            env_instance_id="fixture-env",
            env_generation=1,
        )
        self.store = Store(self.run_dir, dict(self.identity))
        self.addCleanup(self.store.close)
        self.marker_world_m = [0.03, -0.36, 0.82]
        self.robot = CalibratedThreeCamRobot(Path(self.tmp_dir) / "captures", marker_world_m=self.marker_world_m)
        self.config = Config.from_dict(dict(
            features=dict(
                motion_review=False,
                object_table_observe=False,
                scene3d_observe=False,
            ),
            images=dict(storage_jpeg_quality=98, transport_format="webp", webp_quality=95),
        ))
        self.h = Harness(self.store, self.robot, config=self.config, observation_profile="rgb_proprio_calibrated_v1")
        install_registry(self.h, self.config)
        self.h.thread_id = "thread-focus-1"
        self.initial_obs = self.h.start()

    def test_contract_schema_descriptions_and_mutual_exclusion(self):
        registry = create_registry()
        catalog = {d["name"]: d for d in public_definitions()}
        for tool_name in ("robot_observe", "history_view_observation"):
            desc = catalog[tool_name]["description"]
            for keyword in (
                "image_delivery",
                "focus_3d",
                "env_origin",
                "0.02",
                "0.4",
                "128",
                "480",
                "256",
                "full_uv = [crop_x + local_u, crop_y + local_v]",
            ):
                self.assertIn(keyword, desc, f"{tool_name} missing {keyword}")

        valid_observe = [
            {},
            dict(image_delivery="none"),
            dict(image_delivery="head_only"),
            dict(image_delivery="all"),
            dict(focus_3d=dict(target_m=[0.0, -0.35, 0.80])),
            dict(image_delivery="focus_roi", focus_3d=dict(object_id="o2", radius_m=0.06, patch_px=192)),
            dict(focus_3d=dict(target_m=[0.0, -0.35, 0.80], cameras=["cam_left_wrist", "cam_head"])),
        ]
        invalid_observe = [
            dict(focus_3d=dict(target_m=[0.0, -0.35, 0.80]), roi=[10, 10, 100, 100], camera_id="cam_head"),
            dict(focus_3d=dict(target_m=[0.0, -0.35, 0.80]), scale=2, camera_id="cam_head"),
            dict(image_delivery="focus_roi"),
            dict(image_delivery="none", focus_3d=dict(target_m=[0.0, -0.35, 0.80])),
            dict(image_delivery="head_only", focus_3d=dict(target_m=[0.0, -0.35, 0.80])),
            dict(image_delivery="all", camera_id="cam_head"),
            dict(image_delivery="none", roi=[0, 0, 50, 50], camera_id="cam_head"),
            dict(focus_3d={}),
            dict(focus_3d=dict(target_m=[0.0, -0.35, 0.80], radius_m=0.005)),
            dict(focus_3d=dict(target_m=[0.0, -0.35, 0.80], radius_m=0.55)),
            dict(focus_3d=dict(target_m=[0.0, -0.35, 0.80], patch_px=64)),
            dict(focus_3d=dict(target_m=[0.0, -0.35, 0.80], patch_px=512)),
        ]
        for role in ("actor", "program"):
            for args in valid_observe:
                registry.validate("robot_observe", args, role=role)
            for args in invalid_observe:
                with self.subTest(role=role, invalid=args), self.assertRaises(ValueError):
                    registry.validate("robot_observe", args, role=role)

    def test_auto_camera_selection_and_live_wrist_fk_tracking(self):
        target = [0.0, -0.35, 0.80]
        # 未指定 cameras：cam_right_wrist 朝外不可见，应自动选择距离更近的 cam_left_wrist + cam_head（共 2 个）
        planned_0 = plan_focus_3d_views(self.h, self.h.latest, dict(target_m=target))
        self.assertEqual(planned_0["selected_camera_ids"], ["cam_left_wrist", "cam_head"])
        self.assertEqual(planned_0["radius_m"], DEFAULT_RADIUS_M)
        self.assertEqual(planned_0["patch_px"], DEFAULT_PATCH_PX)
        wrist_v0 = planned_0["views"][0]
        head_v0 = planned_0["views"][1]
        self.assertEqual(wrist_v0["pose_source"], "robot_kinematics")
        self.assertEqual(head_v0["pose_source"], "fixed_calibration")

        # 改变左臂关节位移 q_left 并采集新观测，验证左腕相机 FK 实时更新投影像素而头部固定不变
        self.robot.q_left = 0.06
        obs_moved = self.h.observe()
        planned_1 = plan_focus_3d_views(self.h, obs_moved, dict(target_m=target))
        wrist_v1 = next(v for v in planned_1["views"] if v["camera_id"] == "cam_left_wrist")
        head_v1 = next(v for v in planned_1["views"] if v["camera_id"] == "cam_head")
        self.assertNotEqual(wrist_v1["projected_center_uv_px"], wrist_v0["projected_center_uv_px"])
        self.assertEqual(head_v1["projected_center_uv_px"], head_v0["projected_center_uv_px"])

        # 显式指定视锥外相机 cam_right_wrist 应立即清晰报错
        with self.assertRaisesRegex(ValueError, "focus_3d_target_outside_camera_frustum:cam_right_wrist"):
            plan_focus_3d_views(self.h, obs_moved, dict(target_m=target, cameras=["cam_right_wrist"]))

        # 所有相机后方的目标应报不可见错误
        with self.assertRaisesRegex(ValueError, "focus_3d_target_not_visible_in_any_camera"):
            plan_focus_3d_views(self.h, obs_moved, dict(target_m=[0.0, 2.5, 3.0]))

    def test_object_id_resolution_and_explicit_parameter_preservation(self):
        obs_id = self.h.latest["observation_id"]
        # 未提供物体表时应直接报错
        with self.assertRaisesRegex(ValueError, "focus_3d_object_table_unavailable"):
            plan_focus_3d_views(self.h, self.h.latest, dict(object_id="o2"))

        fake_table = dict(
            columns=["id", "centre_m", "top_z", "bottom_z", "length", "width", "yaw_deg", "tilt_deg", "support", "quality"],
            rows=[
                ["o1", [-0.12, -0.34, 0.79], 0.82, 0.76, 0.05, 0.04, 0.0, 90.0, "table", "ok"],
                ["o2", [0.02, -0.35, 0.81], 0.86, 0.76, 0.12, 0.08, 15.0, 90.0, "table", "ok"],
            ],
            table_z=0.76,
        )
        obs3d._remember(self.h, obs_id, "object_table", (fake_table, None, {"status": "conditional"}))

        resolved = plan_focus_3d_views(self.h, self.h.latest, dict(object_id="o2"))
        self.assertEqual(resolved["object_id"], "o2")
        self.assertEqual(resolved["target_m"], [0.02, -0.35, 0.81])
        self.assertEqual(resolved["target_source"], "object_table")
        self.assertEqual(resolved["radius_source"], "object_envelope")
        expected_env_r = round(1.25 * 0.5 * math.sqrt(0.12**2 + 0.08**2 + 0.10**2), 6)
        self.assertAlmostEqual(resolved["radius_m"], expected_env_r, places=5)

        # 显式传入 radius_m 与 target_m 必须 100% 尊重调用方设定，严禁后台覆盖
        explicit = plan_focus_3d_views(
            self.h,
            self.h.latest,
            dict(object_id="o2", target_m=[0.01, -0.33, 0.80], radius_m=0.05),
        )
        self.assertEqual(explicit["target_m"], [0.01, -0.33, 0.80])
        self.assertEqual(explicit["target_source"], "explicit_target_m")
        self.assertEqual(explicit["radius_m"], 0.05)
        self.assertEqual(explicit["radius_source"], "explicit")

        with self.assertRaisesRegex(ValueError, "focus_3d_unknown_object_id:o99"):
            plan_focus_3d_views(self.h, self.h.latest, dict(object_id="o99"))

        # 跨观测溯源隔离：另一帧观测无对齐 object_table 时必须明确报 unavailable，绝不静默借用旧帧 o2 的陈旧坐标
        other_bundle = {**deepcopy(self.h.latest), "observation_id": "obs-unaligned-other"}
        with self.assertRaisesRegex(ValueError, "focus_3d_object_table_unavailable:obs-unaligned-other"):
            plan_focus_3d_views(self.h, other_bundle, dict(object_id="o2"))

    def test_lossless_1_to_1_crop_and_stereo_triangulation_from_local_uv(self):
        # 通过真实工具入口裁剪双视角特写，验证 1:1 原生分辨率像素无缩放模糊，并点选局部亮斑反算三维坐标
        resp = self.h.invoke(
            "robot_observe",
            dict(focus_3d=dict(target_m=[0.02, -0.35, 0.81], radius_m=0.05, patch_px=160)),
            dict(tool_call_id="call-tri"),
            role="actor",
        )
        focus_meta = resp.data["focus_3d"]
        self.assertEqual(focus_meta["selected_camera_ids"], ["cam_left_wrist", "cam_head"])
        image_parts = [p for p in resp.content if isinstance(p, dict) and p.get("type") == "inputImage"]
        self.assertEqual(len(image_parts), 2)

        recovered_full_uvs = {}
        cam_entries = {c["camera_id"]: c for c in self.h.latest["cameras"]}
        for view_info, img_part in zip(focus_meta["views"], image_parts):
            cid = view_info["camera_id"]
            cx, cy, cw, ch = view_info["crop_xywh_px"]
            self.assertGreaterEqual(cw, 160)
            self.assertGreaterEqual(ch, 160)
            if cid == "cam_head":
                self.assertEqual((cw, ch), (160, 160))
            self.assertEqual(view_info["display_scale"], 1)

            raw_b64 = img_part["imageUrl"].split(",", 1)[1]
            with Image.open(io.BytesIO(base64.b64decode(raw_b64))) as crop_pil:
                crop_arr = np.asarray(crop_pil.convert("RGB"), dtype=np.float32)
            self.assertEqual(crop_arr.shape[:2], (ch, cw))

            # 直接与原始全分辨率传感器图对应 1:1 切片及模拟先缩全图再放大的模糊对照相比，证明零缩放模糊
            _, orig_bytes = self.store.resolve_image(cam_entries[cid]["rgb_ref"], allowed_scopes=("actor",))
            with Image.open(io.BytesIO(orig_bytes)) as orig_pil:
                orig_rgb = orig_pil.convert("RGB")
                direct_slice = np.asarray(orig_rgb.crop((cx, cy, cx + cw, cy + ch)), dtype=np.float32)
                down_up_full = orig_rgb.resize((orig_rgb.width // 4, orig_rgb.height // 4), Image.Resampling.BILINEAR).resize(
                    (orig_rgb.width, orig_rgb.height), Image.Resampling.BILINEAR
                )
                blurry_slice = np.asarray(down_up_full.crop((cx, cy, cx + cw, cy + ch)), dtype=np.float32)
            mean_diff = float(np.mean(np.abs(crop_arr - direct_slice)))
            blurry_diff = float(np.mean(np.abs(blurry_slice - direct_slice)))
            self.assertLess(mean_diff, 8.0)
            self.assertGreater(blurry_diff, mean_diff * 3.0)

            # 在局部特写图上检测红色特征点质心 [local_u, local_v]（背景灰阶 R==G，红色靶标 R-G 唯一显著）
            redness = np.clip(crop_arr[:, :, 0] - crop_arr[:, :, 1] - 80.0, 0.0, None)
            total_w = float(np.sum(redness))
            self.assertGreater(total_w, 0.0)
            ys, xs = np.indices(redness.shape, dtype=np.float64)
            local_u = float(np.sum(xs * redness) / total_w)
            local_v = float(np.sum(ys * redness) / total_w)
            full_uv = local_to_full_uv(view_info["crop_xywh_px"], [local_u, local_v])
            recovered_full_uvs[cid] = full_uv

        # 将局部特写换算出的原图 [u, v] 直接喂给 geometry_query(kind='triangulate')，验证三维重建误差 < 2mm
        tri_resp = self.h.invoke(
            "geometry_query",
            dict(
                request=dict(
                    kind="triangulate",
                    observation_ids=[self.h.latest["observation_id"], self.h.latest["observation_id"]],
                    camera_id="cam_left_wrist",
                    camera_id_b="cam_head",
                    pixels_a=[recovered_full_uvs["cam_left_wrist"]],
                    pixels_b=[recovered_full_uvs["cam_head"]],
                )
            ),
            dict(tool_call_id="call-geo"),
            role="actor",
        )
        self.assertEqual(tri_resp.data["status"], "conditional")
        self.assertEqual(tri_resp.data["value"]["accepted"], 1)
        recon_pt = np.asarray(tri_resp.data["value"]["points_m"][0], dtype=np.float64)
        err_m = float(np.linalg.norm(recon_pt - np.asarray(self.marker_world_m, dtype=np.float64)))
        self.assertLess(err_m, 0.0025, f"triangulated 3D error too large: {err_m:.5f} m")

    def test_image_delivery_modes_zero_hash_leak_and_native_audit_pass(self):
        # 完整端到端构造 manifest.json 与 codex-raw.jsonl，验证无图观测、多视角 focus_3d 观测与物理动作 100% 通过 native 审计
        manifest = dict(
            run_id=self.identity["run_id"],
            thread_id=self.h.thread_id,
            terminal_session_id="term-focus-1",
            data_origin="robodojo_runtime",
        )
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        raw_lines: list[dict] = []

        def _call_actor_tool(call_id: str, tool_name: str, arguments: dict) -> dict:
            self.h.call_items[call_id] = f"item-{call_id}"
            raw_lines.append(dict(direction="in", message=dict(method="item/started", params=dict(item=dict(id=f"item-{call_id}")))))
            wire = self.h.handle_tool(
                dict(
                    threadId=self.h.thread_id,
                    turnId="turn-1",
                    callId=call_id,
                    tool=tool_name,
                    arguments=arguments,
                )
            )
            self.assertTrue(wire["success"], wire)
            for item in wire["contentItems"]:
                if item.get("type") in ("inputText", "input_text", "text"):
                    self.assertEqual(metadata_findings(item["text"]), [], item["text"])
            raw_lines.append(dict(direction="out", message=dict(id=call_id, result=wire)))
            return wire

        # 1. image_delivery="none"：0 张图像，快速刷新当前观测并作为后续动作合法输入
        wire_none = _call_actor_tool("call-1", "robot_observe", dict(image_delivery="none"))
        imgs_none = [c for c in wire_none["contentItems"] if c.get("type") == "inputImage"]
        self.assertEqual(len(imgs_none), 0)
        obs_after_none = self.h.latest["observation_id"]
        self.assertNotEqual(obs_after_none, self.initial_obs["observation_id"])

        # 基于无图观测执行一次真实物理动作（验证空图 delivery 同样通过 audit 的 input was delivered 校验）
        act_corr_1 = dict(thread_id=self.h.thread_id, turn_id="turn-1", tool_call_id="call-act-1", item_id="item-act-1")
        raw_lines.append(dict(direction="in", message=dict(method="item/started", params=dict(item=dict(id="item-act-1")))))
        act_1 = dict(
            schema_version="0.1",
            **self.identity,
            world_kind="observed",
            action_id="action-after-none",
            based_on_observation_id=obs_after_none,
            frame_id=FRAME_ID,
            tcp_config_id=TCP_CONFIG_ID,
            mode="absolute",
            limits_profile="simulation-slow-v1",
            timeout_s=10,
            path=dict(
                kind="backend_free",
                orientation="backend_free",
                blend_radius_m=0.0,
                position_tolerance_m=0.005,
                waypoints=[dict(left=dict(position_m=[-0.12, -0.22, 0.95], quaternion_xyzw=[0.0, 0.0, 0.0, 1.0]))],
            ),
            gripper_events=[],
        )
        res_1 = self.h.execute(act_1, correlation=act_corr_1)
        self.assertEqual(res_1["receipt"]["status"], "completed")

        # 2. image_delivery="head_only"：仅返回 1 张 cam_head 图像
        wire_head = _call_actor_tool("call-2", "robot_observe", dict(image_delivery="head_only"))
        imgs_head = [c for c in wire_head["contentItems"] if c.get("type") == "inputImage"]
        self.assertEqual(len(imgs_head), 1)
        labels_head = _wire_image_labels(wire_head["contentItems"])
        self.assertTrue(any(lbl and lbl.endswith("/cam_head") for lbl in labels_head), labels_head)

        # 3. focus_3d 多视角特写观测（自动选中 cam_left_wrist + cam_head 共 2 张 1:1 裁剪图）并验证标签对齐
        obs_before_focus = self.h.latest["observation_id"]
        wire_focus = _call_actor_tool(
            "call-3",
            "robot_observe",
            dict(focus_3d=dict(target_m=[0.0, -0.35, 0.80], radius_m=0.06, patch_px=160)),
        )
        # focus_3d 特写不替换当前 observation_id
        self.assertEqual(self.h.latest["observation_id"], obs_before_focus)
        imgs_focus = [c for c in wire_focus["contentItems"] if c.get("type") == "inputImage"]
        self.assertEqual(len(imgs_focus), 2)
        labels_focus = [lbl for lbl in _wire_image_labels(wire_focus["contentItems"]) if lbl is not None]
        self.assertEqual(len(labels_focus), 2)
        self.assertTrue(labels_focus[0].endswith("/cam_left_wrist"), labels_focus)
        self.assertTrue(labels_focus[1].endswith("/cam_head"), labels_focus)

        # 4. history_view_observation：覆盖短 ID 还原 + image_delivery="none" 与 image_delivery="focus_roi"
        short_obs = json.loads(wire_focus["contentItems"][0]["text"])["bundle"]["observation_id"]
        wire_hist_none = _call_actor_tool(
            "call-4",
            "history_view_observation",
            dict(observation_id=short_obs, image_delivery="none"),
        )
        self.assertEqual([c for c in wire_hist_none["contentItems"] if c.get("type") == "inputImage"], [])

        wire_hist_focus = _call_actor_tool(
            "call-5",
            "history_view_observation",
            dict(
                observation_id=short_obs,
                image_delivery="focus_roi",
                focus_3d=dict(target_m=[0.0, -0.35, 0.80], cameras=["cam_head"], patch_px=144),
            ),
        )
        imgs_hist_focus = [c for c in wire_hist_focus["contentItems"] if c.get("type") == "inputImage"]
        self.assertEqual(len(imgs_hist_focus), 1)

        # 同时验证 program 角色调用契约通过
        prog_resp = self.h.invoke(
            "history_view_observation",
            dict(
                observation_id=obs_before_focus,
                focus_3d=dict(target_m=[0.0, -0.35, 0.80], cameras=["cam_left_wrist", "cam_head"], patch_px=160),
            ),
            dict(program_id="job-test", tool_call_id="prog-call-1"),
            role="program",
        )
        self.assertIn("focus_3d", prog_resp.data)
        self.store.set_state(
            "program_call",
            "prog-call-1",
            dict(result=dict(contentItems=[c for c in prog_resp.content if isinstance(c, dict) and c.get("type") == "inputImage"])),
        )

        # 执行第二次真实动作并写入完整 turn/completed 记录
        act_corr_2 = dict(thread_id=self.h.thread_id, turn_id="turn-1", tool_call_id="call-act-2", item_id="item-act-2")
        raw_lines.append(dict(direction="in", message=dict(method="item/started", params=dict(item=dict(id="item-act-2")))))
        act_2 = deepcopy(act_1)
        act_2["action_id"] = "action-after-focus"
        act_2["based_on_observation_id"] = self.h.latest["observation_id"]
        res_2 = self.h.execute(act_2, correlation=act_corr_2)
        self.assertEqual(res_2["receipt"]["status"], "completed")

        raw_lines.append(
            dict(
                direction="in",
                message=dict(
                    method="turn/completed",
                    params=dict(
                        threadId=self.h.thread_id,
                        turn=dict(id="turn-1", status="completed", items=[dict(type="agentMessage", text="NI_RUA_CONTINUITY_OK")]),
                    ),
                ),
            )
        )
        (self.run_dir / "codex-raw.jsonl").write_text(
            "\n".join(canonical(line) for line in raw_lines) + "\n",
            encoding="utf-8",
        )

        # 5. 运行官方 integrity 与 native 双模式审计，断言 100% 零失败通过！
        integrity_report = audit(self.run_dir, profile="integrity")
        self.assertTrue(integrity_report["passed"], integrity_report)
        self.assertEqual(integrity_report["failures"], [])

        native_report = audit(self.run_dir, profile="native")
        self.assertTrue(native_report["passed"], native_report)
        self.assertEqual(native_report["failures"], [])
        self.assertGreaterEqual(native_report["delivered_image_digests"], 4)
        self.assertEqual(native_report["program_image_digests"], 2)


if __name__ == "__main__":
    unittest.main()
