"""P2 三维点云空间场 `PointCloudSpatialField`、持久物体 `track_id`、腕部多视角外参字典解析与 `reanchor_tracks` 契约单测。"""
from __future__ import annotations

NI_RUA_TEST_TIER = "component"

import base64
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

import numpy as np

from ni_rua.apps.evaluation.audit import audit
from ni_rua.core.config import Config
from ni_rua.core.contracts import canonical
from ni_rua.core.presentation.actor_presentation import metadata_findings
from ni_rua.core.storage.store import Store
from ni_rua.perception.external import object_table, obs3d, reanchor_tracks, scene3d
from ni_rua.perception.external import scene3d_helpers as s3
from ni_rua.perception.external.pointcloud_field import PointCloudSpatialField
from ni_rua.perception.external.virtual_field import VirtualSpatialField
from ni_rua.perception.scene.store import SceneStore
from ni_rua.robot.backends.robodojo.robodojo import FRAME_ID, TCP_CONFIG_ID
from ni_rua.runtime.harness import Harness
from ni_rua.runtime.programs.runner import ProgramRunner
from ni_rua.runtime.tools import handlers
from ni_rua.runtime.tools.builtins import create_registry
from ni_rua.runtime.tools.definitions import GEOMETRY_REQUEST_DESCRIPTIONS, public_definitions
from support.geometry_fixture import IDENTITY
from support.perception_fixtures import K, T_HEAD, _branch, two_camera_observation
from support.tool_fixture import install_registry
from support.fixture_robot import CalibratedThreeCamRobot
from support.host_capabilities import expect, program_isolation

TABLE_Z = 0.765
HEX64_RE = re.compile(r"\b[0-9a-f]{64}\b")


def build_hollow_container_cloud_npz(
    center_xy=(0.20, 0.00),
    outer_size_m=0.16,
    wall_thickness_m=0.010,
    bottom_z=TABLE_Z,
    floor_thickness_m=0.006,
    top_z=TABLE_Z + 0.080,
    track_num: int = 5,
) -> bytes:
    """构造带真实内底面与四周侧壁、中心中空的容器三维表面点云 `.npz`（含 `obj_id`、`view_mask` 与机械臂干扰点）。"""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    half = 0.5 * float(outer_size_m)
    inner_half = half - float(wall_thickness_m)
    floor_z = float(bottom_z) + float(floor_thickness_m)

    # 1. 台面背景网格（排除容器外轮廓遮挡区）
    gx = np.arange(-0.10, 0.4501, 0.015)
    gy = np.arange(-0.25, 0.2501, 0.015)
    tx, ty = np.meshgrid(gx, gy)
    table_pts = np.column_stack([tx.ravel(), ty.ravel(), np.full(tx.size, bottom_z)])
    keep_t = (np.abs(table_pts[:, 0] - cx) > half + 0.005) | (np.abs(table_pts[:, 1] - cy) > half + 0.005)
    table_pts = table_pts[keep_t]

    # 2. 容器内腔底板表面点（z = floor_z）
    fx = np.arange(cx - inner_half, cx + inner_half + 1e-6, 0.008)
    fy = np.arange(cy - inner_half, cy + inner_half + 1e-6, 0.008)
    fxx, fyy = np.meshgrid(fx, fy)
    floor_pts = np.column_stack([fxx.ravel(), fyy.ravel(), np.full(fxx.size, floor_z)])

    # 3. 容器四周侧壁与上沿（内腔中心完全中空无点）
    wall_list = []
    zs = np.arange(bottom_z + 0.004, top_z + 1e-6, 0.008)
    span_1d = np.arange(-half, half + 1e-6, 0.006)
    thick_offsets = (0.0, 0.5 * wall_thickness_m, wall_thickness_m)
    for zv in zs:
        for off in thick_offsets:
            wall_list.append(np.column_stack([np.full(len(span_1d), cx - half + off), cy + span_1d, np.full(len(span_1d), zv)]))
            wall_list.append(np.column_stack([np.full(len(span_1d), cx + half - off), cy + span_1d, np.full(len(span_1d), zv)]))
            wall_list.append(np.column_stack([cx + span_1d, np.full(len(span_1d), cy - half + off), np.full(len(span_1d), zv)]))
            wall_list.append(np.column_stack([cx + span_1d, np.full(len(span_1d), cy + half - off), np.full(len(span_1d), zv)]))
    container_pts = np.vstack([floor_pts, *wall_list])

    # 4. 在中空腔体正中心放置少量 robot=True 点，验证机械臂点绝不会被误判为环境碰撞障碍
    robot_pts = np.column_stack([
        np.full(12, cx),
        np.full(12, cy),
        np.linspace(floor_z + 0.02, top_z + 0.04, 12),
    ])

    xyz = np.vstack([table_pts, container_pts, robot_pts]).astype(np.float32)
    n_total = len(xyz)
    robot_mask = np.zeros(n_total, dtype=bool)
    robot_mask[-len(robot_pts):] = True

    obj_id = np.zeros(n_total, dtype=np.int16)
    obj_id[len(table_pts):len(table_pts) + len(container_pts)] = int(track_num)

    view_mask = np.ones(n_total, dtype=np.uint8)
    view_mask[len(table_pts):len(table_pts) + len(floor_pts)] = 3  # 头部 + 左腕双视角均可见内底

    conf = np.full(n_total, 230, dtype=np.uint8)
    uv = np.column_stack([
        np.clip(np.round(320.0 + 500.0 * xyz[:, 0]), 0, 639),
        np.clip(np.round(240.0 - 500.0 * xyz[:, 1]), 0, 479),
    ]).astype(np.uint16)

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        xyz=xyz,
        conf=conf,
        uv=uv,
        count=np.ones(n_total, dtype=np.uint16),
        robot=robot_mask,
        obj_id=obj_id,
        view_mask=view_mask,
        support_z=np.float64(bottom_z),
        voxel_m=np.float64(0.004),
        image_size=np.asarray([640, 480], dtype=np.int32),
    )
    return buf.getvalue()


class FakeP2PerceptionServer:
    """模拟 `:8331` 持久物体跟踪（`track_id`、`cavity_depth_m`、`reanchor`）与多视角点云（`/points`）服务。"""

    def __init__(self, cloud_npz: bytes):
        self.cloud_npz = cloud_npz
        self.requests: list[tuple[str, dict]] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append((self.path, body))
                if self.path == "/object_table":
                    reanchor = body.get("reanchor")
                    is_reanchored = isinstance(reanchor, dict) and not reanchor.get("reset")
                    target_tid = reanchor.get("target_track_id") if isinstance(reanchor, dict) else None
                    o3_reanchored = is_reanchored and target_tid in (None, "o3")
                    o2_reanchored = is_reanchored and target_tid in (None, "o2")
                    objects = [
                        dict(
                            track_id="o3",
                            centre_m=[0.08, -0.18, TABLE_Z + 0.035],
                            top_z_m=TABLE_Z + 0.070,
                            bottom_z_m=TABLE_Z,
                            length_m=0.065,
                            width_m=0.065,
                            yaw_deg=15.0,
                            tilt_deg=None,
                            elongated=False,
                            support="table",
                            quality="ok",
                            quality_flags=[],
                            cavity_depth_m=0.058,
                            track_confidence=0.96 if o3_reanchored else 0.91,
                            track_status="reanchored" if o3_reanchored else "tracked",
                            bbox_px=[260, 290, 320, 350],
                            label_px=[290, 320],
                            contour_px=[[260, 290], [320, 290], [320, 350], [260, 350]],
                        ),
                        dict(
                            track_id="o5",
                            centre_m=[0.20, 0.00, TABLE_Z + 0.040],
                            top_z_m=TABLE_Z + 0.080,
                            bottom_z_m=TABLE_Z,
                            length_m=0.140,
                            width_m=0.140,
                            yaw_deg=0.0,
                            tilt_deg=None,
                            elongated=False,
                            support=0,  # 指向下标 0 对象（即持久 ID 'o3'）以验证索引映射不洗牌
                            quality="ok",
                            quality_flags=[],
                            cavity_depth_m=0.074,
                            surface_slices_m=[
                                dict(z_m=TABLE_Z + 0.015, hull_xy_m=[[0.13, -0.07], [0.27, -0.07], [0.27, 0.07], [0.13, 0.07]]),
                                dict(z_m=TABLE_Z + 0.075, hull_xy_m=[[0.13, -0.07], [0.27, -0.07], [0.27, 0.07], [0.13, 0.07]]),
                            ],
                            track_confidence=0.94,
                            track_status="tracked",
                            bbox_px=[380, 200, 460, 280],
                            label_px=[420, 240],
                            contour_px=[[380, 200], [460, 200], [460, 280], [380, 280]],
                        ),
                        dict(
                            track_id="o2",
                            centre_m=[-0.12, -0.15, TABLE_Z + 0.002],
                            top_z_m=TABLE_Z + 0.004,
                            bottom_z_m=TABLE_Z,
                            length_m=0.080,
                            width_m=0.050,
                            yaw_deg=-30.0,
                            tilt_deg=89.0,
                            elongated=True,
                            support="table",
                            quality="ok",
                            quality_flags=[],
                            cavity_depth_m=0.002,  # < 0.004m，不应暴露在 grasp_hints
                            track_confidence=0.88 if o2_reanchored else 0.42,
                            track_status="reanchored" if o2_reanchored else "low_confidence",
                            bbox_px=[180, 300, 230, 340],
                            label_px=[205, 320],
                            contour_px=[[180, 300], [230, 300], [230, 340], [180, 340]],
                        ),
                    ]
                    low_ids = [] if o2_reanchored else ["o2"]
                    min_conf = 0.88 if o2_reanchored else 0.42
                    answer = dict(
                        status="ok",
                        version="p2-sam31-vggt",
                        support_z_m=body["support_z_m"],
                        n_objects=len(objects),
                        objects=objects,
                        tracking=dict(
                            session_id=body.get("session_id"),
                            persistent_ids=bool(body.get("session_id")),
                            active_tracks=len(objects),
                            low_conf_ids=low_ids,
                            min_track_conf=min_conf,
                        ),
                        timing=dict(total_s=0.012),
                    )
                elif self.path == "/points":
                    extra = body.get("extra_views") or []
                    fused = ["cam_head", *[str(v["camera_id"]) for v in extra if isinstance(v, dict) and v.get("camera_id")]]
                    answer = dict(
                        status="ok",
                        version="p2-sam31-vggt",
                        n_points=1200,
                        n_above_support=820,
                        n_robot=12,
                        voxel_m=body["voxel_m"],
                        support_z_m=body["support_z_m"],
                        bbox_m=[[-0.15, -0.25, TABLE_Z], [0.45, 0.25, TABLE_Z + 0.12]],
                        bbox_above_support_m=[[0.13, -0.07, TABLE_Z + 0.004], [0.27, 0.07, TABLE_Z + 0.08]],
                        npz_b64=base64.b64encode(owner.cloud_npz).decode("ascii"),
                        npz_bytes=len(owner.cloud_npz),
                        fused_views=fused,
                        multiview_residual_m=0.0028 if len(fused) > 1 else None,
                        timing=dict(total_s=0.015),
                        support_fit=dict(tilt_deg=0.3),
                        scale=1.0,
                    )
                else:
                    answer = dict(status="error", error="unknown_endpoint")
                raw = json.dumps(answer).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class PointCloudFieldAndReanchorTests(unittest.TestCase):
    """覆盖持久 `track_id`、`PointCloudSpatialField` 凹腔碰撞净空、`program_run` 零外部依赖导入、腕部外参字典与 `reanchor_tracks`。"""

    def setUp(self):
        self.cloud_npz = build_hollow_container_cloud_npz()
        s3.use(None)

    def tearDown(self):
        s3.use(None)

    def test_persistent_track_ids_cavity_hints_and_track_meta_contract(self):
        """验证持久 `track_id` 跳号与乱序不重编、`COLUMNS` 10 列不变、`too_wide` 保留 `cavity_depth_m` 及 `track_meta` 对齐。"""
        tmp_dir = self.enterContext(tempfile.TemporaryDirectory())
        store = Store(tmp_dir, IDENTITY)
        self.addCleanup(store.close)
        scene = SceneStore(Path(tmp_dir) / "scene", store.resolve_artifact, store.put_artifact)
        observation, _ = two_camera_observation(store)
        scene.import_observation(observation)

        fake = FakeP2PerceptionServer(self.cloud_npz)
        self.addCleanup(fake.close)

        cfg = Config().with_overrides({
            "features": {
                "external_perception": True,
                "object_table": True,
                "object_table_observe": True,
                "persistent_object_tracking": True,
                "scene3d_observe": False,
                "vggt_endpoint": fake.url,
                "vggt_support_z_m": TABLE_Z,
            }
        })
        runtime = SimpleNamespace(
            config=cfg,
            scene=scene,
            image_settings=None,
            identity=dict(IDENTITY),
            store=store,
            features=cfg.features,
        )

        resp = handlers.geometry_query(runtime, {"request": {"kind": "object_table", "observation_id": "obs-ot"}}, {})
        val = resp.data["value"]

        # 1. COLUMNS 暴露纯传感器像素坐标 label_px（11 列），且行首 ID 直接采用服务端持久 track_id（绝不重编为 o1, o2, o3）
        self.assertEqual(
            object_table.COLUMNS,
            ["id", "centre_m", "top_z", "bottom_z", "length", "width", "yaw_deg", "tilt_deg", "support", "quality", "label_px"],
        )
        self.assertEqual(val["columns"], object_table.COLUMNS)
        self.assertEqual([r[0] for r in val["rows"]], ["o3", "o5", "o2"])
        self.assertEqual([r[10] for r in val["rows"]], [[290, 320], [420, 240], [205, 320]])
        # support=0 精确映射至下标 0 对象的持久 ID 'o3'
        self.assertEqual(val["rows"][1][8], "o3")

        # 2. 请求体自动附带 episode_id 作为 session_id
        self.assertEqual(fake.requests[0][0], "/object_table")
        self.assertEqual(fake.requests[0][1]["session_id"], str(IDENTITY["episode_id"]))

        # 3. 凹腔深度 >= 0.004m 时在 grasp_hints 中暴露 cavity_depth_m（包含 opening_check == 'too_wide' 的大容器 o5）
        hints = val["grasp_hints"]
        self.assertEqual(hints["o3"]["cavity_depth_m"], 0.058)
        self.assertEqual(hints["o5"]["opening_check"], "too_wide")
        self.assertEqual(hints["o5"]["cavity_depth_m"], 0.074)
        self.assertNotIn("cavity_depth_m", hints["o2"])

        # 4. track_meta 精确对齐服务端 tracking 与 #65 compute_object_table_delta 读取契约
        expected_meta = {
            "persistent_ids": True,
            "min_track_conf": 0.42,
            "low_conf_ids": ["o2"],
            "active_tracks": 3,
        }
        self.assertEqual(val["track_meta"], expected_meta)

        # 5. obs3d 自动挂载时同样在 shown["object_table"]["track_meta"] 中输出紧凑元数据
        parts = obs3d.attachments(runtime, observation, {"tool_call_id": "obs-call-1"}, reply="observe")
        shown = parts[0].value
        self.assertEqual(shown["object_table"]["track_meta"], expected_meta)
        self.assertEqual([r[0] for r in shown["object_table"]["rows"]], ["o3", "o5", "o2"])

    def test_pointcloud_spatial_field_hollow_cavity_vs_wall_collision_and_fallback(self):
        """验证 `PointCloudSpatialField` 消除中空容器内腔假阳性碰撞、保留盒壁真实干涉报警、支持壳体分解与实心回退。"""
        # 构造桌面待抓取小圆柱 'o1' 与中空收纳盒 'o5'（16cm x 16cm, 壁厚 1cm, 内腔净宽 14cm, 内底 0.771m, 顶高 0.845m）
        ot = dict(
            columns=object_table.COLUMNS,
            table_z=TABLE_Z,
            rows=[
                ["o1", [0.00, -0.20, TABLE_Z + 0.025], TABLE_Z + 0.050, TABLE_Z, 0.028, 0.028, 0.0, None, "table", "ok"],
                ["o5", [0.20, 0.00, TABLE_Z + 0.040], TABLE_Z + 0.080, TABLE_Z, 0.160, 0.160, 0.0, None, "table", "ok"],
            ],
            grasp_hints={
                "o5": {"opening_check": "too_wide", "elongated": False, "cavity_depth_m": 0.074},
            },
        )
        robot_state_holding = {
            "arms": {
                "right": {
                    "grasp_pose": {"position_m": [0.00, -0.20, TABLE_Z + 0.160], "angles_deg": [0.0, 0.0, 0.0]},
                    "gripper": {"measured_aperture_m": 0.028},
                }
            }
        }
        # 先在两套场中让右臂竖直向下夹持住小圆柱 o1
        ot_held = deepcopy(ot)
        ot_held["rows"][0][1] = [0.00, -0.20, TABLE_Z + 0.160]
        ot_held["rows"][0][2] = TABLE_Z + 0.185
        ot_held["rows"][0][3] = TABLE_Z + 0.135
        ot_held["rows"][0][8] = "held_right"

        # 动作序列 A：高空平移至收纳盒中心 (0.20, 0.00) 上方 -> 竖直向下伸入中空内腔 z=0.810m (< top_z 0.845m!) -> 张爪释放 -> 撤离
        insert_center_stages = [
            {"id": "hover", "arm": "right", "kind": "move", "grasp_position_m": [0.20, 0.00, TABLE_Z + 0.160], "angles_deg": [0.0, 0.0, 0.0]},
            {"id": "insert_cavity", "arm": "right", "kind": "move", "grasp_position_m": [0.20, 0.00, TABLE_Z + 0.045], "angles_deg": [0.0, 0.0, 0.0]},
            {"id": "release", "arm": "right", "kind": "gripper", "goal_normalized": 1.0},
            {"id": "retreat", "arm": "right", "kind": "move", "grasp_position_m": [0.20, 0.00, TABLE_Z + 0.160], "angles_deg": [0.0, 0.0, 0.0]},
        ]

        # 对照组：旧版纯实心 OBB 场把中空盒当成实心砖块，在伸入内腔时误报碰撞
        solid_vf = VirtualSpatialField(objects=ot_held, robot_state=robot_state_holding)
        sim_solid = solid_vf.simulate_stages(insert_center_stages)
        self.assertEqual(sim_solid["status"], "risk_detected")
        self.assertIn("o5", sim_solid["stage_reports"][1]["colliding_objects"])

        # 实验组 1：绑定真实三维表面点云的 PointCloudSpatialField 准确识别中空内腔净空，且释放后支撑在容器内底而非浮在盒顶
        pc_vf = PointCloudSpatialField(objects=ot_held, robot_state=robot_state_holding, cloud_npz=self.cloud_npz)
        sim_clean = pc_vf.simulate_stages(insert_center_stages)
        self.assertEqual(sim_clean["status"], "clean", sim_clean)
        self.assertTrue(sim_clean["theoretically_valid"])
        self.assertEqual(sim_clean["stage_reports"][1]["colliding_objects"], [])
        self.assertEqual(sim_clean["stage_reports"][2]["new_support"], "o5")
        released_o1 = next(item for item in sim_clean["virtual_world_after"] if item["id"] == "o1")
        self.assertLess(released_o1["bottom_z"], TABLE_Z + 0.015)

        # 实验组 2：横向偏移撞击真实容器右侧壁 (x = 0.20 + 0.075 = 0.275m)，准确触发路径碰撞报警
        hit_wall_stages = [
            {"id": "hover", "arm": "right", "kind": "move", "grasp_position_m": [0.275, 0.00, TABLE_Z + 0.160], "angles_deg": [0.0, 0.0, 0.0]},
            {"id": "hit_wall", "arm": "right", "kind": "move", "grasp_position_m": [0.275, 0.00, TABLE_Z + 0.055], "angles_deg": [0.0, 0.0, 0.0]},
        ]
        sim_wall = pc_vf.simulate_stages(hit_wall_stages)
        self.assertEqual(sim_wall["status"], "risk_detected")
        self.assertIn("o5", sim_wall["stage_reports"][1]["colliding_objects"])

        # 实验组 3：竖直向下沿 X 轴闭合夹持中空容器 1cm 左侧壁边缘（x = 0.20 - 0.075 = 0.125m, angles_deg=[0,0,0]）可正常通过可夹持性评估
        rim_eval = pc_vf.evaluate_grasp(
            arm="right",
            position_m=[0.125, 0.00, TABLE_Z + 0.072],
            angles_deg=[0.0, 0.0, 0.0],
            target_id="o5",
        )
        self.assertTrue(rim_eval["theoretically_graspable"], rim_eval)
        self.assertLess(rim_eval["projected_span_m"], 0.030)

        # 实验组 4：带实测 surface_slices_m 的中空容器无点云时同样支持中心插入并拦截撞壁；
        # 既无点云又无实测切片时绝不凭空脑补 0.12 * min_dim 五板壁厚，诚实回退实心 OBB3D
        ot_with_slices = deepcopy(ot_held)
        hull_o5 = [[0.12, -0.08], [0.28, -0.08], [0.28, 0.08], [0.12, 0.08]]
        slice_vf = PointCloudSpatialField(
            objects=ot_with_slices,
            robot_state=robot_state_holding,
            cloud_npz=None,
            surface_slices_m={
                "o5": [
                    dict(z_m=TABLE_Z + 0.012, hull_xy_m=hull_o5),
                    dict(z_m=TABLE_Z + 0.075, hull_xy_m=hull_o5),
                ]
            },
        )
        self.assertEqual(slice_vf.simulate_stages(insert_center_stages)["status"], "clean")
        self.assertEqual(slice_vf.simulate_stages(hit_wall_stages)["status"], "risk_detected")

        no_cloud_no_slice_vf = PointCloudSpatialField(objects=ot_held, robot_state=robot_state_holding, cloud_npz=None)
        self.assertFalse(no_cloud_no_slice_vf.has_fine_geometry("o5"))
        self.assertEqual(no_cloud_no_slice_vf.inspect_cavity("o5")["inner_clearance_radius_m"], 0.0)
        self.assertEqual(no_cloud_no_slice_vf.simulate_stages(insert_center_stages)["status"], "risk_detected")

        solid_fallback_ot = dict(
            columns=object_table.COLUMNS,
            table_z=TABLE_Z,
            rows=[["o9", [0.20, 0.00, TABLE_Z + 0.040], TABLE_Z + 0.080, TABLE_Z, 0.160, 0.160, 0.0, None, "table", "ok"]],
        )
        fallback_vf = PointCloudSpatialField(objects=solid_fallback_ot, cloud_npz=None)
        sim_fb = fallback_vf.simulate_stages([
            {"id": "into_solid", "arm": "right", "kind": "move", "grasp_position_m": [0.20, 0.00, TABLE_Z + 0.045], "angles_deg": [0.0, 0.0, 0.0]},
        ])
        self.assertEqual(sim_fb["status"], "risk_detected")
        self.assertIn("o9", sim_fb["stage_reports"][0]["colliding_objects"])

        # 实验组 5：inspect_cavity 与 s3.cavity_profile 几何剖面测量精度
        pc_prof = PointCloudSpatialField(objects=ot, cloud_npz=self.cloud_npz).inspect_cavity("o5", radius_m=0.09)
        s3.load(self.cloud_npz)
        prof = s3.cavity_profile("o5", radius=0.09, objects=ot)
        for p_res in (pc_prof, prof):
            self.assertAlmostEqual(p_res["rim_z_m"], TABLE_Z + 0.080, delta=0.008)
            self.assertAlmostEqual(p_res["floor_z_m"], TABLE_Z + 0.006, delta=0.006)
            self.assertAlmostEqual(p_res["cavity_depth_m"], 0.074, delta=0.010)
            np.testing.assert_allclose(p_res["opening_center_xy_m"], [0.20, 0.00], atol=0.012)
            self.assertGreater(p_res["inner_clearance_radius_m"], 0.050)
            self.assertGreater(p_res["n_points"], 100)

    def test_program_run_sandbox_zero_external_pythonpath_import_and_execution(self):
        """验证 `ProgramRunner` 工作目录拷贝齐备，沙箱子进程在零项目 `PYTHONPATH` 下真实导入并运行 `PointCloudSpatialField` 与 `cavity_profile`。"""
        tmp_dir = self.enterContext(tempfile.TemporaryDirectory())
        run_dir = Path(tmp_dir) / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        store = Store(run_dir, dict(IDENTITY))
        self.addCleanup(store.close)
        robot = CalibratedThreeCamRobot(Path(tmp_dir) / "captures")
        cfg = Config.from_dict({
            "features": {
                "external_perception": True,
                "scene3d": True,
                "scene3d_observe": True,
                "object_table": True,
                "object_table_observe": False,
                "motion_review": False,
                "sandbox_enabled": expect(program_isolation, "program_run runs under the Landlock + seccomp sandbox"),
            }
        })
        harness = Harness(store, robot, config=cfg, observation_profile="rgb_proprio_calibrated_v1")
        install_registry(harness, cfg)
        harness.start()

        # workspace_modules 接口保持 ("scene3d_helpers.py",) 单项不变
        self.assertEqual(scene3d.workspace_modules(harness), ("scene3d_helpers.py",))

        art_ref = store.put_artifact(self.cloud_npz, scene3d.ARTIFACT_KIND, scene3d.ARTIFACT_MIME)
        runner = ProgramRunner(harness.program_services())
        self.addCleanup(runner.close)
        harness.programs = runner

        program_src = f"""
import json
import scene3d_helpers as s3
from ni_rua_client import NiRuaClient

client = NiRuaClient()
cloud = s3.load({art_ref!r}, client=client)
ot = {{
    "columns": ["id", "centre_m", "top_z", "bottom_z", "length", "width", "yaw_deg", "tilt_deg", "support", "quality"],
    "table_z": {TABLE_Z},
    "rows": [["o5", [0.20, 0.00, {TABLE_Z + 0.040}], {TABLE_Z + 0.080}, {TABLE_Z}, 0.160, 0.160, 0.0, None, "table", "ok"]],
    "grasp_hints": {{"o5": {{"opening_check": "too_wide", "elongated": False, "cavity_depth_m": 0.074}}}},
}}
prof = s3.cavity_profile("o5", radius=0.09, objects=ot, cloud=cloud)
pf = s3.prefilter_stages([
    {{"id": "hover", "arm": "right", "kind": "move", "grasp_position_m": [0.20, -0.14, {TABLE_Z + 0.140}], "angles_deg": [0.0, 0.0, 0.0]}},
    {{"id": "into_cavity", "arm": "right", "kind": "move", "grasp_position_m": [0.20, -0.14, {TABLE_Z + 0.055}], "angles_deg": [0.0, 0.0, 0.0], "depends_on": ["hover"]}},
])
print(json.dumps({{
    "cavity_depth_m": prof["cavity_depth_m"],
    "inner_r_m": prof["inner_clearance_radius_m"],
    "pf_valid": pf["valid"],
    "workspace_dir": __import__("os").getcwd(),
}}))
"""
        job = runner.run(
            {"source": program_src, "timeout_s": 30.0},
            {"thread_id": "thread-prog", "turn_id": "turn-prog", "tool_call_id": "prog-1"},
        )
        self.assertEqual(job["status"], "succeeded", job)
        _, stdout_bytes = store.resolve_artifact(job["stdout_ref"])
        payload = json.loads(stdout_bytes.decode("utf-8").strip())
        self.assertGreater(payload["cavity_depth_m"], 0.06)
        self.assertGreater(payload["inner_r_m"], 0.045)
        self.assertTrue(payload["pf_valid"])

        # 额外在完全剥离项目根目录 PYTHONPATH 的纯净独立进程中验证轻量四件套工作目录零外部依赖可导入，且重型空间场不再污染沙盒
        ws_dir = Path(payload["workspace_dir"])
        for fname in ("scene3d_helpers.py", "pose_math.py", "field_ops.py", "ik_prefilter.py"):
            self.assertTrue((ws_dir / fname).is_file(), f"missing {fname} in program workspace")
        for retired_fname in ("pointcloud_field.py", "virtual_field.py", "vectorized_grasp.py", "se3_path.py"):
            self.assertFalse((ws_dir / retired_fname).exists(), f"unexpected heavy module {retired_fname} in sandbox")
        clean_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import scene3d_helpers as s3; wps = s3.se3_screw(([0.2, -0.1, 0.85], [0, 0, 0]), [1, 0, 0], pivot_m=[0.2, 0.0, 0.85], angle_rad=0.7854, n_steps=3); assert len(wps) == 3",
            ],
            cwd=str(ws_dir),
            env=clean_env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_wrist_extra_views_dict_extrinsics_voxel_0002_and_obs3d_field_upgrade(self):
        """验证 `ObservationBundle` 字典结构腕部动态外参提取、`voxel_m=0.002` 下限及 `obs3d` 升级 `PointCloudSpatialField`。"""
        tmp_dir = self.enterContext(tempfile.TemporaryDirectory())
        run_dir = Path(tmp_dir) / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        store = Store(run_dir, dict(IDENTITY))
        self.addCleanup(store.close)
        fake = FakeP2PerceptionServer(self.cloud_npz)
        self.addCleanup(fake.close)

        robot = CalibratedThreeCamRobot(Path(tmp_dir) / "captures")
        # 驱动左腕移动 +0.05m，使动态关节 FK 位姿显著异于初始零位
        robot.q_left = 0.05
        cfg = Config.from_dict({
            "features": {
                "external_perception": True,
                "object_table": True,
                "object_table_observe": True,
                "scene3d": True,
                "scene3d_observe": True,
                "persistent_object_tracking": True,
                "multiview_scene3d": True,
                "vggt_endpoint": fake.url,
                "vggt_support_z_m": TABLE_Z,
                "motion_review": False,
            }
        })
        harness = Harness(store, robot, config=cfg, observation_profile="rgb_proprio_calibrated_v1")
        install_registry(harness, cfg)
        harness.start()
        obs_resp = harness.invoke("robot_observe", {}, {"tool_call_id": "obs-init"}, role="actor")
        obs = obs_resp.data

        # 1. 验证 obs3d 默认内联挂载时：/object_table 先于 /points 到达服务端，纯几何视锥预筛仅保留工作区内近场的 cam_left_wrist 并剔除朝向台外的空闲 cam_right_wrist
        paths = [p for p, _ in fake.requests]
        self.assertEqual(paths[:2], ["/object_table", "/points"])
        points_req = fake.requests[1][1]
        self.assertEqual(points_req["session_id"], str(IDENTITY["episode_id"]))
        self.assertIn("extra_views", points_req)
        extra_by_id = {v["camera_id"]: v for v in points_req["extra_views"]}
        self.assertEqual(set(extra_by_id), {"cam_left_wrist"})
        expected_t_left = robot._wrist_world_pose(robot.t_left_base, 0.05, robot.t_left_link_cam)
        np.testing.assert_allclose(extra_by_id["cam_left_wrist"]["T_world_camera"], expected_t_left, atol=1e-5)

        self.assertIsInstance(harness._virtual_spatial_field, PointCloudSpatialField)
        self.assertIsNotNone(harness._virtual_spatial_field._cloud_data)
        self.assertIn("o5", harness._virtual_spatial_field._obj_surface_points)

        # 1b. 当双侧机械臂均停泊在高处且朝向远离桌面时，obs3d 自动跳过全部空闲腕部视角以直接命中单图缓存
        orig_left_base = robot.t_left_base.copy()
        robot.t_left_base = robot.t_right_base.copy()
        try:
            n_before = len(fake.requests)
            harness.invoke("robot_observe", {}, {"tool_call_id": "obs-idle-both"}, role="actor")
            idle_points_req = fake.requests[n_before + 1][1]
            self.assertNotIn("extra_views", idle_points_req)
        finally:
            robot.t_left_base = orig_left_base

        # 2. 验证显式 geometry_query(kind='scene3d', voxel_m=0.002, include_wrist_views=True) 双角色校验通过，且显式请求始终包含全部可用腕部视角
        registry = create_registry(runtime=harness)
        req_0002 = {"kind": "scene3d", "observation_id": obs["observation_id"], "voxel_m": 0.002, "include_wrist_views": True}
        for role in ("actor", "program"):
            registry.validate("geometry_query", {"request": req_0002}, role=role)
        s3_resp = harness.invoke("geometry_query", {"request": req_0002}, {"tool_call_id": "s3-0002"}, role="actor")
        self.assertEqual(s3_resp.data["status"], "conditional")
        self.assertEqual(s3_resp.data["value"]["voxel_m"], 0.002)
        self.assertEqual(s3_resp.data["value"]["fused_views"], ["cam_head", "cam_left_wrist", "cam_right_wrist"])
        self.assertAlmostEqual(s3_resp.data["value"]["multiview_residual_m"], 0.0028, places=4)

        # 3. 验证直接构造内联 dict 结构 camera_calibration["cameras"] 与 camera_pose_sample["cameras"]（含回退 camera_calibration）
        t_left_dyn = np.eye(4, dtype=float)
        t_left_dyn[:3, 3] = [0.12, -0.18, 0.94]
        t_right_cal_fallback = np.eye(4, dtype=float)
        t_right_cal_fallback[:3, 3] = [0.22, 0.14, 0.98]
        inline_bundle = {
            "observation_id": "obs-inline-dict",
            "cameras": [
                {"camera_id": "cam_head", "image_b64": base64.b64encode(b"head").decode("ascii")},
                {"camera_id": "cam_left_wrist", "image_b64": base64.b64encode(b"left").decode("ascii")},
                {"camera_id": "cam_right_wrist", "image_b64": base64.b64encode(b"right").decode("ascii")},
            ],
            "camera_calibration": {
                "cameras": {
                    "cam_left_wrist": {"K": K, "distortion": {"model": "opencv", "coefficients": [0, 0, 0, 0, 0]}, "T_world_camera": T_HEAD},
                    "cam_right_wrist": {"K": K, "distortion": {"model": "opencv", "coefficients": [0, 0, 0, 0, 0]}, "T_world_camera": t_right_cal_fallback.tolist()},
                }
            },
            "camera_pose_sample": {
                "cameras": {
                    "cam_left_wrist": {"T_world_camera": t_left_dyn.tolist()},
                }
            },
        }
        left_view = scene3d._extract_wrist_camera_from_bundle(harness, inline_bundle, "cam_left_wrist")
        right_view = scene3d._extract_wrist_camera_from_bundle(harness, inline_bundle, "cam_right_wrist")
        np.testing.assert_allclose(left_view["T_world_camera"], t_left_dyn, atol=1e-6)
        np.testing.assert_allclose(right_view["T_world_camera"], t_right_cal_fallback, atol=1e-6)

    def test_reanchor_tracks_dual_role_schema_audit_and_zero_hash_leak(self):
        """验证 `reanchor_tracks` Actor+Program 双入口、双向 `target_id`/`target_track_id`、`audit(profile='native')` 及零摘要泄漏。"""
        tmp_dir = self.enterContext(tempfile.TemporaryDirectory())
        run_dir = Path(tmp_dir) / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        identity = dict(
            run_id="p2-reanchor-audit",
            episode_id="ep-reanchor-1",
            env_instance_id="fixture-env",
            env_generation=1,
        )
        store = Store(run_dir, dict(identity))
        self.addCleanup(store.close)
        fake = FakeP2PerceptionServer(self.cloud_npz)
        self.addCleanup(fake.close)

        robot = CalibratedThreeCamRobot(Path(tmp_dir) / "captures")
        cfg = Config.from_dict({
            "features": {
                "external_perception": True,
                "vggt_height": True,
                "vggt_height_mode": "default",
                "object_table": True,
                "object_table_observe": True,
                "scene3d": True,
                "scene3d_observe": True,
                "persistent_object_tracking": True,
                "vggt_endpoint": fake.url,
                "vggt_support_z_m": TABLE_Z,
                "motion_review": False,
            }
        })
        harness = Harness(store, robot, config=cfg, observation_profile="rgb_proprio_calibrated_v1")
        install_registry(harness, cfg)
        harness.thread_id = "thread-p2-1"
        harness.start()

        registry = create_registry(runtime=harness)
        obs_id = harness.latest["observation_id"]

        # 1. Actor + Program 双角色输入契约正反例校验
        valid_reqs = [
            {"kind": "reanchor_tracks", "observation_id": obs_id, "target_id": "o3", "pos_pixels": [[290, 320]], "neg_pixels": [[200, 200]]},
            {"kind": "reanchor_tracks", "observation_id": obs_id, "target_track_id": "o2", "phrases": ["thin card"], "include_overlay": True},
            {"kind": "reanchor_tracks", "observation_id": obs_id, "reset": True},
        ]
        for role in ("actor", "program"):
            for r_item in valid_reqs:
                registry.validate("geometry_query", {"request": r_item}, role=role)
            for bad_item in (
                {"kind": "reanchor_tracks", "observation_id": obs_id},  # minProperties=3
                {"kind": "reanchor_tracks", "observation_id": obs_id, "unknown_field": 1},
                {"kind": "reanchor_tracks", "observation_id": obs_id, "pos_pixels": [[10]]},
            ):
                with self.assertRaises(Exception):
                    registry.validate("geometry_query", {"request": bad_item}, role=role)

        # 冲突的 target_id 与 target_track_id 显式报错
        conflict_resp = harness.invoke(
            "geometry_query",
            {"request": {"kind": "reanchor_tracks", "observation_id": obs_id, "target_id": "o2", "target_track_id": "o3"}},
            {"tool_call_id": "call-conflict"},
            role="program",
        )
        self.assertEqual(conflict_resp.data["status"], "invalid_input")
        self.assertIn("conflicting_target_ids", conflict_resp.data["reason"])

        # Program 角色执行 reanchor_tracks（target_track_id 映射 + include_overlay 默认 False 返回 0 张图）
        prog_resp = harness.invoke(
            "geometry_query",
            {
                "request": {
                    "kind": "reanchor_tracks",
                    "observation_id": obs_id,
                    "target_track_id": "o3",
                    "phrases": ["ceramic mug"],
                    "pos_pixels": [[290, 320]],
                    "neg_pixels": [[420, 240]],
                }
            },
            {"program_id": "job-reanchor", "tool_call_id": "prog-reanchor-1"},
            role="program",
        )
        self.assertEqual(prog_resp.data["status"], "conditional")
        self.assertEqual(prog_resp.data["query_type"], "reanchor_tracks")
        self.assertEqual(prog_resp.data["value"]["reanchored_ids"], ["o3"])
        self.assertEqual(prog_resp.data["value"]["track_meta"]["low_conf_ids"], ["o2"])
        self.assertEqual([c for c in prog_resp.content if isinstance(c, dict) and c.get("type") == "inputImage"], [])
        last_sent = fake.requests[-1][1]
        self.assertEqual(last_sent["session_id"], identity["episode_id"])
        self.assertEqual(
            last_sent["reanchor"],
            {
                "reset": False,
                "target_track_id": "o3",
                "phrases": ["ceramic mug"],
                "pos_pixels": [[290.0, 320.0]],
                "neg_pixels": [[420.0, 240.0]],
            },
        )
        store.set_state("program_call", "prog-reanchor-1", {"result": {"contentItems": []}})

        # 2. Actor 端真实记录 codex-raw.jsonl（测试 include_overlay=False 与 include_overlay=True）并运行 audit(profile="native") + 零 hash 检查
        manifest = {
            "run_id": identity["run_id"],
            "thread_id": harness.thread_id,
            "terminal_session_id": "term-p2-1",
            "data_origin": "robodojo_runtime",
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        raw_lines: list[dict] = []

        def _call_actor(call_id: str, tool_name: str, arguments: dict) -> dict:
            harness.call_items[call_id] = f"item-{call_id}"
            raw_lines.append({"direction": "in", "message": {"method": "item/started", "params": {"item": {"id": f"item-{call_id}"}}}})
            wire = harness.handle_tool({
                "threadId": harness.thread_id,
                "turnId": "turn-p2",
                "callId": call_id,
                "tool": tool_name,
                "arguments": arguments,
            })
            self.assertTrue(wire["success"], wire)
            for item in wire["contentItems"]:
                if item.get("type") in ("inputText", "input_text", "text"):
                    self.assertEqual(metadata_findings(item["text"]), [], item["text"])
                    self.assertEqual(HEX64_RE.findall(item["text"]), [], item["text"])
            raw_lines.append({"direction": "out", "message": {"id": call_id, "result": wire}})
            return wire

        wire_obs = _call_actor("call-obs-1", "robot_observe", {})
        short_obs = json.loads(wire_obs["contentItems"][0]["text"])["bundle"]["observation_id"]

        wire_reanchor_no_img = _call_actor(
            "call-reanchor-0",
            "geometry_query",
            {"request": {"kind": "reanchor_tracks", "observation_id": short_obs, "target_id": "o3", "pos_pixels": [[290, 320]]}},
        )
        self.assertEqual([c for c in wire_reanchor_no_img["contentItems"] if c.get("type") == "inputImage"], [])

        wire_reanchor_img = _call_actor(
            "call-reanchor-1",
            "geometry_query",
            {"request": {"kind": "reanchor_tracks", "observation_id": short_obs, "phrases": ["mug"], "include_overlay": True}},
        )
        self.assertEqual(len([c for c in wire_reanchor_img["contentItems"] if c.get("type") == "inputImage"]), 1)

        act_corr = {"thread_id": harness.thread_id, "turn_id": "turn-p2", "tool_call_id": "call-act-1", "item_id": "item-act-1"}
        raw_lines.append({"direction": "in", "message": {"method": "item/started", "params": {"item": {"id": "item-act-1"}}}})
        act_res = harness.execute(
            {
                "schema_version": "0.1",
                **identity,
                "world_kind": "observed",
                "action_id": "action-p2-ee-1",
                "based_on_observation_id": harness.latest["observation_id"],
                "frame_id": FRAME_ID,
                "tcp_config_id": TCP_CONFIG_ID,
                "mode": "absolute",
                "limits_profile": "simulation-slow-v1",
                "timeout_s": 10,
                "path": {
                    "kind": "backend_free",
                    "orientation": "backend_free",
                    "blend_radius_m": 0.0,
                    "position_tolerance_m": 0.005,
                    "waypoints": [{"left": {"position_m": [-0.12, -0.22, 0.95], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]}}],
                },
                "gripper_events": [],
            },
            correlation=act_corr,
        )
        self.assertEqual(act_res["receipt"]["status"], "completed")

        raw_lines.append({
            "direction": "in",
            "message": {
                "method": "turn/completed",
                "params": {
                    "threadId": harness.thread_id,
                    "turn": {"id": "turn-p2", "status": "completed", "items": [{"type": "agentMessage", "text": "done"}]},
                },
            },
        })
        (run_dir / "codex-raw.jsonl").write_text("\n".join(canonical(line) for line in raw_lines) + "\n", encoding="utf-8")

        native_audit = audit(run_dir, profile="native")
        self.assertTrue(native_audit["passed"], native_audit)
        self.assertEqual(native_audit["failures"], [])

        # 非法像素入参报 invalid_input，而内部 TypeError 不被伪装为调用方入参错误
        from unittest import mock
        from ni_rua.perception.external import reanchor_tracks as rt_mod
        bad_px_res, _ = rt_mod.reanchor_tracks_query(
            harness,
            {"kind": "reanchor_tracks", "observation_id": obs_id, "pos_pixels": ["bad"]},
        )
        self.assertEqual(bad_px_res["status"], "invalid_input")
        with mock.patch.object(rt_mod, "build_reanchor_payload", side_effect=TypeError("internal_reanchor_bug")):
            with self.assertRaisesRegex(TypeError, "internal_reanchor_bug"):
                rt_mod.reanchor_tracks_query(harness, {"kind": "reanchor_tracks", "observation_id": obs_id, "reset": True})


if __name__ == "__main__":
    unittest.main()
