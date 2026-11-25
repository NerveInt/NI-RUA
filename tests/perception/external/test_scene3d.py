"""features.scene3d: geometry_query kind scene3d (fake /points service) and the scene3d_helpers queries."""
NI_RUA_TEST_TIER = "component"

import base64
import hashlib
import io
import json
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from ni_rua.core.config import Config
from ni_rua.core.contracts import canonical
from ni_rua.perception.scene.store import SceneStore
from ni_rua.core.storage.store import Store
from ni_rua.runtime.tools import handlers
from ni_rua.runtime.tools.builtins import create_registry
from ni_rua.runtime.tools.definitions import public_definitions
from ni_rua.perception.external import scene3d
from ni_rua.perception.external import scene3d_helpers as s3
from support.geometry_fixture import IDENTITY
from support.perception_fixtures import K, T_HEAD, two_camera_observation, _branch

TABLE = 0.765


def synthetic_cloud(seed=0):
    """Table grid (1 cm) at z=TABLE over [-0.3, 0.3]^2 with two objects and a few robot points.
    box: 6 x 4 cm footprint centred (0.10, 0.00), top TABLE+0.04 (top face + side walls);
    cylinder: r 2 cm at (-0.10, 0.10), top TABLE+0.10; robot: a vertical line at (0.2, 0.2) up to +0.3 m.
    uv: a top-down pinhole with 500 px/m about (320, 240)."""
    rng = np.random.default_rng(seed)
    g = np.arange(-0.30, 0.3001, 0.01)
    X, Y = np.meshgrid(g, g)
    table = np.stack([X.ravel(), Y.ravel(), np.full(X.size, TABLE)], 1)
    box_top = [[x, y, TABLE + 0.04] for x in np.arange(0.07, 0.1301, 0.005) for y in np.arange(-0.02, 0.0201, 0.005)]
    box_side = [[0.07, y, z] for y in np.arange(-0.02, 0.0201, 0.005) for z in np.arange(TABLE + 0.005, TABLE + 0.0301, 0.005)]
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    cyl = [[-0.10 + 0.02 * np.cos(a), 0.10 + 0.02 * np.sin(a), z] for a in ang for z in np.arange(TABLE + 0.01, TABLE + 0.1001, 0.01)]
    cyl += [[-0.10 + r * np.cos(a), 0.10 + r * np.sin(a), TABLE + 0.10] for r in (0.0, 0.01) for a in ang[::3]]
    robot = [[0.2, 0.2, TABLE + z] for z in np.arange(0.05, 0.3, 0.01)]
    objects = np.asarray(box_top + box_side + cyl, float)
    # objects occlude the table under them
    keep = np.ones(len(table), bool)
    keep &= ~((np.abs(table[:, 0] - 0.10) <= 0.03) & (np.abs(table[:, 1]) <= 0.02))
    keep &= np.hypot(table[:, 0] + 0.10, table[:, 1] - 0.10) > 0.02
    xyz = np.concatenate([table[keep], objects, np.asarray(robot)])
    robot_flag = np.zeros(len(xyz), bool)
    robot_flag[-len(robot):] = True
    conf = rng.integers(60, 256, len(xyz)).astype(np.uint8)
    conf[5] = 0                                    # one low-confidence table point
    uv = np.stack([320 + 500 * xyz[:, 0], 240 - 500 * xyz[:, 1]], 1).round().astype(np.uint16)
    buf = io.BytesIO()
    np.savez_compressed(buf, xyz=xyz.astype(np.float32), conf=conf, uv=uv, count=np.ones(len(xyz), np.uint16), robot=robot_flag,
                        support_z=np.float64(TABLE), voxel_m=np.float64(0.01), image_size=np.asarray([640, 480], np.int32))
    return buf.getvalue()


class HelperQueries(unittest.TestCase):
    def setUp(self):
        self.cloud = s3.use(s3.from_bytes(synthetic_cloud()))
        self.addCleanup(s3.use, None)

    def test_info_and_table(self):
        info = s3.info()
        self.assertEqual(info["table_z"], TABLE)
        self.assertEqual(info["near_robot"], 25)
        self.assertEqual(s3.table_z(), TABLE)
        local = s3.table_z([0.0, -0.2], radius=0.05)
        self.assertAlmostEqual(local["z"], TABLE, places=4)
        self.assertGreater(local["n"], 20)

    def test_top_z_regions(self):
        box = s3.top_z(([0.06, -0.03, 0.0], [0.14, 0.03, 2.0]))
        self.assertAlmostEqual(box["top_z"], TABLE + 0.04, places=3)
        self.assertAlmostEqual(box["height_above_table"], 0.04, places=3)
        cyl = s3.top_z({"center": [-0.10, 0.10], "radius": 0.03})
        self.assertAlmostEqual(cyl["top_z"], TABLE + 0.10, places=3)
        poly = s3.top_z({"polygon": [[0.06, -0.03], [0.14, -0.03], [0.14, 0.03], [0.06, 0.03]]})
        self.assertEqual(poly["n"], box["n"])
        clipped = s3.top_z({"center": [-0.10, 0.10], "radius": 0.03, "z_max": TABLE + 0.05})
        self.assertLess(clipped["top_z"], TABLE + 0.051)
        empty = s3.top_z({"center": [0.0, -0.2], "radius": 0.03})
        self.assertEqual((empty["n"], empty["top_z"]), (0, None))
        # robot points are excluded unless asked for
        self.assertEqual(s3.top_z({"center": [0.2, 0.2], "radius": 0.01})["n"], 0)
        self.assertAlmostEqual(s3.top_z({"center": [0.2, 0.2], "radius": 0.01}, robot=True, q=100)["top_z"], TABLE + 0.29, places=3)

    def test_points_in_box_and_cylinder(self):
        inside = s3.points_in_box([0.06, -0.03, TABLE + 0.035], [0.14, 0.03, TABLE + 0.05], xyz=True)
        self.assertEqual(inside["n"], 13 * 9)
        self.assertEqual(inside["xyz"].shape, (117, 3))
        self.assertAlmostEqual(inside["centroid"][0], 0.10, places=3)
        self.assertNotIn("xyz", s3.points_in_box([0, 0, 0], [1, 1, 1]))
        cyl = s3.points_in_cylinder([-0.10, 0.10], 0.025, z_min=TABLE + 0.004)
        self.assertGreater(cyl["n"], 200)
        self.assertAlmostEqual(cyl["max"][2], TABLE + 0.10, places=3)
        # confidence filter: the one conf-0 table point is dropped by default
        everything = s3.points_in_box([-1, -1, 0], [1, 1, 2], min_conf=0.0, robot=True)["n"]
        self.assertEqual(everything, len(self.cloud.xyz))
        self.assertEqual(s3.points_in_box([-1, -1, 0], [1, 1, 2], robot=True)["n"], everything - 1)

    def test_nearest(self):
        near = s3.nearest([0.10, 0.0, TABLE + 0.04], k=3)
        self.assertEqual(len(near["xyz"]), 3)
        self.assertLessEqual(near["dist"][0], 1e-6)
        self.assertEqual(near["uv"][0], [370, 240])
        horizontal = s3.nearest([-0.10, 0.10], k=1)
        self.assertLess(horizontal["dist"][0], 0.011)

    def test_cluster_at_pixel_separates_objects(self):
        box = s3.cluster_at_pixel(370, 240)
        self.assertAlmostEqual(box["top_z"], TABLE + 0.04, places=3)
        self.assertLess(box["max"][0], 0.131)
        self.assertGreater(box["min"][0], 0.069)
        self.assertEqual(box["seed"]["uv"], [370, 240])
        cyl = s3.cluster_at_pixel(270, 190)
        self.assertAlmostEqual(cyl["max"][2], TABLE + 0.10, places=3)
        self.assertLess(abs(cyl["centroid"][0] + 0.10), 0.01)
        self.assertNotIn("xyz", cyl)

    def test_not_loaded(self):
        s3.use(None)
        with self.assertRaises(ValueError):
            s3.info()
        with self.assertRaises(RuntimeError):
            s3.info()

    def test_workspace_cloud_autoload_reload_invalidation_and_pinning(self):
        def _pack_cloud(xyz_pts):
            pts = np.asarray(xyz_pts, dtype=np.float32).reshape(-1, 3)
            n = len(pts)
            uv = np.stack([320 + 500 * pts[:, 0], 240 - 500 * pts[:, 1]], axis=1).round().astype(np.uint16)
            buf = io.BytesIO()
            np.savez_compressed(
                buf,
                xyz=pts,
                conf=np.full(n, 220, dtype=np.uint8),
                uv=uv,
                count=np.ones(n, dtype=np.uint16),
                robot=np.zeros(n, dtype=bool),
                support_z=np.float64(TABLE),
                voxel_m=np.float64(0.01),
                image_size=np.asarray([640, 480], dtype=np.int32),
            )
            return buf.getvalue()

        ws = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(mock.patch.object(s3, "_workspace_dir", return_value=ws))
        s3.use(None)

        cloud_path = ws / "_spatial_cloud.npz"
        snap_path = ws / "_spatial_snapshot.json"

        # 1) Auto-loads workspace _spatial_cloud.npz + _spatial_snapshot.json when _DEFAULT is unset
        pts_v1 = [[0.10 + 0.005 * i, 0.0, TABLE + 0.04] for i in range(12)]
        cloud_path.write_bytes(_pack_cloud(pts_v1))
        snap_path.write_text(json.dumps({"observation_id": "obs_1"}), encoding="utf-8")

        self.assertEqual(s3.API, tuple(scene3d.API))
        info_v1 = s3.info()
        self.assertEqual(info_v1["n"], 12)
        self.assertEqual(s3._DEFAULT_SOURCE, "workspace")
        self.assertEqual(s3._DEFAULT_STAMP[0], "obs_1")
        self.assertAlmostEqual(s3.top_z({"center": [0.10, 0.0], "radius": 0.08})["top_z"], TABLE + 0.04, places=4)

        # 2) Automatically reloads when _spatial_cloud.npz and/or observation_id updates after an action
        pts_v2 = [[0.10 + 0.004 * i, 0.0, TABLE + 0.11] for i in range(25)]
        cloud_path.write_bytes(_pack_cloud(pts_v2))
        snap_path.write_text(json.dumps({"observation_id": "obs_2"}), encoding="utf-8")

        info_v2 = s3.info()
        self.assertEqual(info_v2["n"], 25)
        self.assertEqual(s3._DEFAULT_STAMP[0], "obs_2")
        self.assertAlmostEqual(s3.top_z({"center": [0.10, 0.0], "radius": 0.08})["top_z"], TABLE + 0.11, places=4)

        # Updating observation_id alone also updates stamp cleanly
        snap_path.write_text(json.dumps({"observation_id": "obs_3"}), encoding="utf-8")
        self.assertEqual(s3.info()["n"], 25)
        self.assertEqual(s3._DEFAULT_STAMP[0], "obs_3")

        # 3) s3.use(cloud) pins an explicit cloud; s3.use(None) unpins back to workspace auto-load
        pinned = s3.from_bytes(synthetic_cloud())
        s3.use(pinned)
        self.assertEqual(s3._DEFAULT_SOURCE, "pinned")
        self.assertEqual(s3.info()["n"], len(pinned.xyz))

        s3.use(None)
        self.assertIsNone(s3._DEFAULT_SOURCE)
        self.assertEqual(s3.info()["n"], 25)
        self.assertEqual(s3._DEFAULT_SOURCE, "workspace")

        # 4) Action invalidation removes _spatial_cloud.npz -> raises RuntimeError instead of serving stale points
        cloud_path.unlink()
        snap_path.unlink()
        with self.assertRaises(RuntimeError):
            s3.info()
        self.assertIsNone(s3._DEFAULT)

    def test_triangulate_rays_undistorts_strong_radial_tangential_distortion(self):
        def _look_at(eye, target):
            fwd = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
            fwd /= np.linalg.norm(fwd)
            right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
            right /= np.linalg.norm(right)
            down = np.cross(fwd, right)
            T = np.eye(4, dtype=np.float64)
            T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, down, fwd, eye
            return T

        K1 = np.array([[420.0, 0.0, 320.0], [0.0, 420.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        K2 = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        D = np.array([-0.32, 0.14, 0.002, -0.001, -0.03], dtype=np.float64)
        true_pt = np.array([0.35, -0.12, 0.78], dtype=np.float64)

        # Aim cameras off-axis from true_pt so lens distortion induces > 2 cm pinhole error
        T1 = _look_at([0.05, -0.45, 1.18], [0.10, 0.12, 0.78])
        T2 = _look_at([0.62, 0.22, 1.12], [0.10, 0.12, 0.78])

        inv1, inv2 = np.linalg.inv(T1), np.linalg.inv(T2)
        uv1 = cv2.projectPoints(true_pt.reshape(1, 3), cv2.Rodrigues(inv1[:3, :3])[0], inv1[:3, 3], K1, D)[0].reshape(2)
        uv2 = cv2.projectPoints(true_pt.reshape(1, 3), cv2.Rodrigues(inv2[:3, :3])[0], inv2[:3, 3], K2, D)[0].reshape(2)

        tri = s3.triangulate_rays(uv1[0], uv1[1], K1, T1, uv2[0], uv2[1], K2, T2, D1=D, D2=D)
        self.assertEqual(tri["status"], "ok")
        self.assertLess(float(np.linalg.norm(np.asarray(tri["point_m"]) - true_pt)), 1e-4)
        self.assertLess(tri["skew_gap_m"], 1e-5)

        # Ignoring D incurs multi-cm error (> 2 cm)
        tri_pinhole = s3.triangulate_rays(uv1[0], uv1[1], K1, T1, uv2[0], uv2[1], K2, T2)
        self.assertGreater(float(np.linalg.norm(np.asarray(tri_pinhole["point_m"]) - true_pt)), 0.02)

        # Camera calibration dict convenience (K/D and intrinsics/dist_coeffs) + keyword aliases
        tri_dict = s3.triangulate_rays(
            uv1[0], uv1[1], {"K": K1, "D": D.tolist()}, T1,
            uv2[0], uv2[1], {"intrinsics": K2.tolist(), "dist_coeffs": D.tolist()}, T2,
        )
        np.testing.assert_allclose(tri_dict["point_m"], tri["point_m"], atol=1e-6)
        tri_kw = s3.triangulate_rays(
            uv1[0], uv1[1], K1, T1, uv2[0], uv2[1], K2, T2, dist_coeffs1=D, dist_coeffs2=D
        )
        np.testing.assert_allclose(tri_kw["point_m"], tri["point_m"], atol=1e-6)

        # D=None backward compatibility even when cv2 is absent
        uv1_p = cv2.projectPoints(true_pt.reshape(1, 3), cv2.Rodrigues(inv1[:3, :3])[0], inv1[:3, 3], K1, np.zeros(5))[0].reshape(2)
        uv2_p = cv2.projectPoints(true_pt.reshape(1, 3), cv2.Rodrigues(inv2[:3, :3])[0], inv2[:3, 3], K2, np.zeros(5))[0].reshape(2)
        with mock.patch.dict(sys.modules, {"cv2": None}):
            tri_none = s3.triangulate_rays(uv1_p[0], uv1_p[1], K1, T1, uv2_p[0], uv2_p[1], K2, T2)
        self.assertEqual(tri_none["status"], "ok")
        self.assertLess(float(np.linalg.norm(np.asarray(tri_none["point_m"]) - true_pt)), 1e-4)

    def test_cluster_at_pixel_kdtree_exact_connectivity_and_10k_speed(self):
        rng = np.random.default_rng(19)
        radius = 0.015
        # Part 1: Exact parity against reference Euclidean dist <= radius connectivity (including exact boundary link)
        seed_pt = np.array([[0.0, 0.0, TABLE + 0.03]], dtype=np.float64)
        bridge_pt = np.array([[0.009, 0.012, TABLE + 0.03]], dtype=np.float64)  # exact 0.015 m 3-4-5 link
        cluster_a = rng.uniform([0.008, 0.012, TABLE + 0.022], [0.042, 0.046, TABLE + 0.052], size=(180, 3))
        cluster_a[0] = [0.012, 0.016, TABLE + 0.03]
        distractor_b = rng.uniform([0.065, 0.012, TABLE + 0.022], [0.105, 0.046, TABLE + 0.052], size=(150, 3))  # > 0.023 m gap
        table_pts = np.column_stack([rng.uniform(-0.2, 0.2, 100), rng.uniform(-0.2, 0.2, 100), np.full(100, TABLE)])
        xyz_small = np.vstack([seed_pt, bridge_pt, cluster_a, distractor_b, table_pts])
        uv_small = np.stack([320 + 500 * xyz_small[:, 0], 240 - 500 * xyz_small[:, 1]], axis=1).round().astype(np.uint16)
        uv_small[0] = [320, 240]
        small_cloud = s3.Cloud({
            "xyz": xyz_small,
            "conf": np.full(len(xyz_small), 230, dtype=np.uint8),
            "uv": uv_small,
            "support_z": TABLE,
        })

        # Reference brute-force Euclidean dist <= radius BFS
        keep = np.flatnonzero(small_cloud.mask(0.1, False) & (small_cloud.xyz[:, 2] > TABLE + s3.ABOVE_M))
        pix = np.linalg.norm(small_cloud.uv[keep] - np.array([320.0, 240.0]), axis=1)
        seed_idx = keep[int(np.argmin(pix))]
        Q = small_cloud.xyz[keep]
        bf_vis = np.zeros(len(Q), dtype=bool)
        bf_vis[int(np.flatnonzero(keep == seed_idx)[0])] = True
        front = bf_vis.copy()
        while front.any():
            grown = np.zeros(len(Q), dtype=bool)
            for p in Q[front]:
                grown |= np.linalg.norm(Q - p, axis=1) <= radius
            front = grown & ~bf_vis
            bf_vis |= grown
        ref_pts = Q[bf_vis]

        got = s3.cluster_at_pixel(320, 240, radius=radius, cloud=small_cloud, xyz=True)
        self.assertEqual(got["n"], len(ref_pts))
        self.assertEqual(got["n"], 1 + 1 + 180)
        np.testing.assert_allclose(got["xyz"], ref_pts, atol=1e-12)
        self.assertEqual(got["min"], [round(float(v), 4) for v in ref_pts.min(0)])
        self.assertEqual(got["max"], [round(float(v), 4) for v in ref_pts.max(0)])
        self.assertEqual(got["centroid"], [round(float(v), 4) for v in ref_pts.mean(0)])

        # Part 2: Dense 10,000-point cloud completes in < 0.2s and cleanly separates adjacent distractor cluster
        dense_table = np.column_stack([rng.uniform(-0.3, 0.3, 4000), rng.uniform(-0.3, 0.3, 4000), np.full(4000, TABLE)])
        dense_target = rng.uniform([0.05, -0.03, TABLE + 0.008], [0.11, 0.03, TABLE + 0.068], size=(3500, 3))
        dense_target[0] = [0.08, 0.00, TABLE + 0.035]
        dense_distractor = rng.uniform([0.132, -0.03, TABLE + 0.008], [0.19, 0.03, TABLE + 0.068], size=(2500, 3))
        xyz_10k = np.vstack([dense_table, dense_target, dense_distractor])
        self.assertEqual(len(xyz_10k), 10000)
        uv_10k = np.stack([320 + 500 * xyz_10k[:, 0], 240 - 500 * xyz_10k[:, 1]], axis=1).round().astype(np.uint16)
        uv_10k[4000] = [360, 240]
        dense_cloud = s3.Cloud({
            "xyz": xyz_10k,
            "conf": np.full(10000, 230, dtype=np.uint8),
            "uv": uv_10k,
            "support_z": TABLE,
        })

        t0 = time.perf_counter()
        dense_res = s3.cluster_at_pixel(360, 240, radius=radius, cloud=dense_cloud)
        elapsed_s = time.perf_counter() - t0
        self.assertEqual(dense_res["n"], 3500)
        self.assertLess(dense_res["max"][0], 0.111)
        self.assertLess(elapsed_s, 0.2, f"cluster_at_pixel on 10k points took {elapsed_s:.3f}s >= 0.2s")

    def test_api_signatures_defaults_and_docstrings_match_implementation(self):
        """校验 scene3d.API / s3.API 中每一项函数名、形参名、显式默认值均与真实 inspect.signature 及进程内查询默认一致。"""
        import ast
        import inspect
        import re

        self.assertEqual(tuple(scene3d.API), s3.API)
        for entry in scene3d.API:
            if entry.startswith("kwargs:"):
                continue
            match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\(([^)]*)\)", entry)
            self.assertIsNotNone(match, f"无法解析 API 条目签名: {entry}")
            fn_name, arg_str = match.group(1), match.group(2)
            self.assertTrue(hasattr(s3, fn_name), f"scene3d_helpers 缺少 API 声明的函数: {fn_name}")
            fn = getattr(s3, fn_name)
            real_sig = inspect.signature(fn)
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in real_sig.parameters.values())
            # 用 AST 解析 API 字符串中的形参与默认值（跳过含 "..." 的省略占位）
            clean_args = [a.strip() for a in arg_str.split(",") if a.strip() and a.strip() != "..."]
            parsed = ast.parse(f"def _f({', '.join(clean_args)}): pass").body[0].args
            all_args = list(parsed.args)
            defaults = [None] * (len(all_args) - len(parsed.defaults)) + list(parsed.defaults)
            for arg_node, default_node in zip(all_args, defaults):
                name = arg_node.arg
                if name not in real_sig.parameters:
                    self.assertTrue(has_var_kw, f"{fn_name}: API 形参 {name!r} 不在真实签名 {real_sig} 中")
                    continue
                if default_node is not None:
                    expected_default = ast.literal_eval(default_node)
                    actual_default = real_sig.parameters[name].default
                    self.assertEqual(
                        actual_default,
                        expected_default,
                        f"{fn_name}.{name}: API 默认值 {expected_default!r} != 真实默认值 {actual_default!r}",
                    )

        # 校验进程内 _execute_one_cloud_query 对 in_hand_axis / surface_extrema 的默认 min_conf 与 helpers 对齐
        fake_cloud = s3.Cloud({"xyz": np.zeros((1, 3), dtype=np.float32), "conf": np.array([255], dtype=np.uint8),
                               "uv": np.zeros((1, 2), dtype=np.uint16), "support_z": TABLE})
        captured = {}
        with mock.patch.object(fake_cloud, "in_hand_axis", side_effect=lambda *a, **kw: captured.setdefault("ih", kw) or {"n": 0}), \
                mock.patch.object(fake_cloud, "surface_extrema", side_effect=lambda *a, **kw: captured.setdefault("se", kw) or {"n": 0}):
            scene3d._execute_one_cloud_query(
                SimpleNamespace(), fake_cloud, {"op": "in_hand_axis", "grasp_center_m": [0.0, 0.0, 0.8]},
                observation_id=None,
            )
            scene3d._execute_one_cloud_query(
                SimpleNamespace(), fake_cloud, {"op": "surface_extrema", "center": [0.0, 0.0]},
                observation_id=None,
            )
        self.assertEqual(captured["ih"]["min_conf"], 0.1)
        self.assertEqual(captured["se"]["min_conf"], 0.05)

        # 校验 scene3d_helpers.py 源码中无错误示例 client.robot_action(
        helpers_src = Path(s3.__file__).read_text(encoding="utf-8")
        self.assertNotIn("client.robot_action(", helpers_src)

    def test_xyzw_quaternion_helpers_in_package_and_isolated_sandbox(self):
        """验证核心层、external 层与隔离沙盒工作区均导出统一 xyzw 四元数换算函数且与 wxyz/旋转矩阵往返一致。"""
        import os
        import subprocess
        from ni_rua.core import pose_math as core_pm
        from ni_rua.perception.external import pose_math as ext_pm

        angles = (18.5, -12.25, 67.5)
        for mod in (core_pm, ext_pm, s3):
            q_xyzw = mod.angles_deg_to_quat_xyzw(*angles)
            q_wxyz = mod.angles_deg_to_quat_wxyz(*angles)
            self.assertEqual(len(q_xyzw), 4)
            np.testing.assert_allclose(mod.quat_wxyz_to_xyzw(q_wxyz), q_xyzw, atol=1e-12)
            np.testing.assert_allclose(mod.quat_xyzw_to_wxyz(q_xyzw), q_wxyz, atol=1e-12)
            np.testing.assert_allclose(mod.quat_xyzw_to_angles_deg(q_xyzw), angles, atol=1e-6)
            R_ref = mod.quat_wxyz_to_matrix(q_wxyz)
            np.testing.assert_allclose(mod.quat_xyzw_to_matrix(q_xyzw), R_ref, atol=1e-12)
            q_back = mod.matrix_to_quat_xyzw(R_ref)
            # q 与 -q 表示同一旋转
            self.assertAlmostEqual(abs(float(np.dot(q_back, q_xyzw))), 1.0, places=10)

        # 在完全无 ni_rua 包路径的隔离沙盒工作区验证 pose_math 与 scene3d_helpers 的 xyzw 导出
        workspace = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.assertTrue(scene3d.copy_workspace_modules((scene3d.WORKSPACE_MODULE,), workspace))
        probe = (
            "import json, numpy as np, pose_math as pm, scene3d_helpers as s3\n"
            "ang = (18.5, -12.25, 67.5)\n"
            "q1 = s3.angles_deg_to_quat_xyzw(*ang)\n"
            "q2 = pm.angles_deg_to_quat_xyzw(*ang)\n"
            "assert np.allclose(q1, q2, atol=1e-12)\n"
            "assert np.allclose(s3.quat_xyzw_to_angles_deg(q1), ang, atol=1e-6)\n"
            "q_w = s3.angles_deg_to_quat_wxyz(*ang)\n"
            "assert np.allclose(s3.quat_xyzw_to_wxyz(q1), q_w, atol=1e-12)\n"
            "assert np.allclose(s3.quat_wxyz_to_xyzw(q_w), q1, atol=1e-12)\n"
            "R = s3.quat_xyzw_to_matrix(q1)\n"
            "assert np.allclose(R, s3.quat_wxyz_to_matrix(q_w), atol=1e-12)\n"
            "q_m = s3.matrix_to_quat_xyzw(R)\n"
            "assert abs(float(np.dot(q_m, q1))) > 1.0 - 1e-10\n"
            "print(json.dumps(q1))\n"
        )
        done = subprocess.run(
            [sys.executable, "-P", "-c", probe],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "PYTHONPATH": str(workspace), "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        np.testing.assert_allclose(json.loads(done.stdout), core_pm.angles_deg_to_quat_xyzw(*angles), atol=1e-12)


class FakePoints:
    def __init__(self, status="ok"):
        self.requests, owner = [], self
        data = synthetic_cloud()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append((self.path, body))
                if status != "ok":
                    answer, code = dict(status="error", error=status), 422
                else:
                    answer, code = dict(status="ok", version="fake-points", n_points=3000, n_above_support=600, n_robot=25,
                                        voxel_m=body["voxel_m"], support_z_m=body["support_z_m"],
                                        bbox_m=[[-0.3, -0.3, 0.765], [0.3, 0.3, 1.055]],
                                        bbox_above_support_m=[[-0.12, -0.02, 0.77], [0.13, 0.12, 0.865]],
                                        npz_b64=base64.b64encode(data).decode(), npz_bytes=len(data),
                                        timing=dict(total_s=0.01), support_fit=dict(tilt_deg=0.5), scale=1.0), 200
                raw = json.dumps(answer).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.data = data
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class Scene3dTool(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = self.enterContext(tempfile.TemporaryDirectory())
        self.artifacts = Store(self.tmp_dir, IDENTITY); self.addCleanup(self.artifacts.close)
        self.scene = SceneStore(Path(self.tmp_dir) / "scene", self.artifacts.resolve_artifact, self.artifacts.put_artifact)
        observation, self.images = two_camera_observation(self.artifacts)
        self.scene.import_observation(observation)

    def runtime(self, on=True, endpoint="http://127.0.0.1:9", object_table=False):
        config = Config().with_overrides({"features": {"tool_catalog_profile": "full_v1", "external_perception": False, "scene3d": on, "object_table": object_table, "vggt_endpoint": endpoint,
                                                       "vggt_support_z_m": TABLE, "vggt_timeout_s": 5.0}})
        return SimpleNamespace(config=config, scene=self.scene, image_settings=None, identity=dict(IDENTITY), store=self.artifacts,
                               features=config.features)

    def catalog_runtime(self, **kw):
        runtime = self.runtime(**kw)
        return SimpleNamespace(disabled_tools=(), observation_profile="rgb_proprio_calibrated_v1", motion_review=True,
                               enable_wait=True, vision_compare=True, visual_reviewer=True, action_policy=None,
                               capabilities=None, programs=True, config=runtime.config, features=runtime.config.features)

    def request(self, **changes):
        return dict(kind="scene3d", observation_id="obs-ot") | changes

    def test_off_catalog_identical_and_request_rejected(self):
        self.assertEqual(canonical(public_definitions(scene3d=False)), canonical(public_definitions()))
        digest = lambda runtime: hashlib.sha256(canonical(create_registry(runtime=runtime).catalog(runtime, role="actor")).encode()).hexdigest()
        self.assertEqual(digest(self.catalog_runtime(on=False)), hashlib.sha256(canonical(
            create_registry().catalog(self.catalog_runtime(on=False), role="actor")).encode()).hexdigest())
        self.assertNotIn("scene3d", canonical(create_registry(runtime=self.runtime(on=False)).catalog(self.catalog_runtime(on=False), role="actor")))
        from ni_rua.core.tool_errors import ToolInputError
        with self.assertRaises(ToolInputError):
            create_registry(runtime=self.runtime(on=False)).validate("geometry_query", {"request": self.request()}, role="actor")
        self.assertEqual(scene3d.workspace_modules(self.runtime(on=False)), ())
        self.assertEqual(scene3d.workspace_modules(self.runtime()), ("scene3d_helpers.py",))
        from ni_rua.runtime.programs.runner import ProgramServices
        self.assertEqual(ProgramServices.__dataclass_fields__["workspace_modules"].default, ())

    def test_on_definition_adds_only_the_scene3d_branch(self):
        off = {d["name"]: d for d in public_definitions()}
        on = {d["name"]: d for d in public_definitions(scene3d=True)}
        for name in off:
            if name != "geometry_query":
                self.assertEqual(canonical(on[name]), canonical(off[name]), name)
        geometry = on["geometry_query"]
        branch = _branch(geometry, "scene3d")
        self.assertEqual(branch["properties"]["kind"]["description"], scene3d.KIND_SENTENCE)
        self.assertEqual(branch["required"], ["kind"])
        self.assertEqual(canonical(_branch(geometry, "pixel_plane")), canonical(_branch(off["geometry_query"], "pixel_plane")))
        added = len(canonical(geometry)) - len(canonical(off["geometry_query"]))
        self.assertLess(added, 2450)   # 含 voxel_m (0.002..0.05)、include_wrist_views 与 include_progress_wrist_views 说明
        both = {d["name"]: d for d in public_definitions(object_table=True, scene3d=True)}
        self.assertIsNotNone(_branch(both["geometry_query"], "object_table"))
        self.assertIsNotNone(_branch(both["geometry_query"], "scene3d"))
        registry = create_registry(runtime=self.runtime())
        registry.validate("geometry_query", {"request": self.request(camera_id="cam_head", voxel_m=0.02)}, role="actor")
        registry.validate("geometry_query", {"request": self.request()}, role="program")
        from ni_rua.core.tool_errors import ToolInputError
        for bad in (self.request(extra=1), self.request(voxel_m=0.5), {"kind": "scene3d"}):
            with self.assertRaises(ToolInputError):
                registry.validate("geometry_query", {"request": bad}, role="actor")

    def test_query_stores_artifact_and_returns_summary(self):
        fake = FakePoints(); self.addCleanup(fake.close)
        response = handlers.geometry_query(self.runtime(endpoint=fake.url), {"request": self.request()}, {})
        result = response.data
        self.assertEqual(result["status"], "conditional", result)
        value = result["value"]
        self.assertTrue(value["artifact"].startswith("artifact:"))
        info, content = self.artifacts.resolve_artifact(value["artifact"])
        self.assertEqual((info["kind"], info["mime_type"], content), ("scene3d_points", "application/x-npz", fake.data))
        self.assertEqual((value["points"], value["points_above_table"], value["near_robot"], value["table_z"]), (3000, 600, 25, TABLE))
        self.assertEqual(value["api"], scene3d.API)
        text = json.dumps(response.content, default=str)
        self.assertNotIn("npz_b64", text)
        self.assertNotIn("diagnostics", text)
        self.assertLess(len(text), 4000)
        path, sent = fake.requests[0]
        self.assertEqual(path, "/points")
        self.assertEqual(base64.b64decode(sent["image_b64"]), self.images["cam_head"])
        np.testing.assert_allclose(sent["K"], K); np.testing.assert_allclose(sent["T_world_camera"], T_HEAD)
        self.assertEqual((sent["voxel_m"], sent["support_z_m"], sent["params"]), (0.01, TABLE, dict(z_above=0.6, z_below=0.05)))
        cloud = s3.from_bytes(content)
        self.assertAlmostEqual(s3.top_z({"center": [-0.10, 0.10], "radius": 0.03}, cloud=cloud)["top_z"], TABLE + 0.10, places=3)

    def test_api_lists_only_what_the_embodiment_workspace_can_call(self):
        """An embodiment whose workspace gets no ik_prefilter.py (SO-101: no grasp_angles input) is told of no IK
        function, by the scene3d reply or by s3.API in its program workspace; ARX-X5 keeps the full list."""
        import os
        import subprocess
        from dataclasses import replace
        from ni_rua.robot.backends.so101.deployment import embodiment_profile
        from ni_rua.robot.model import embodiments
        from robot.so101.test_so101_backend import deployment

        x5 = embodiments.ARX_X5_PROFILE
        so101 = embodiment_profile(deployment())
        # The ARX-X5 fixture scene without the grasp_angles input: the same staging decision as SO-101.
        no_ik = replace(x5, pose_inputs=x5.pose_inputs - {"grasp_angles"})
        self.assertEqual([scene3d.ik_prefilter_staged(profile) for profile in (None, x5, so101, no_ik)],
                         [True, True, False, False])

        def workspace_api(profile):
            """(s3.API, public functions that call the IK prefilter) as a program job of ``profile`` sees them."""
            workspace = Path(self.enterContext(tempfile.TemporaryDirectory()))
            self.assertTrue(scene3d.copy_workspace_modules((scene3d.WORKSPACE_MODULE,), workspace, profile=profile))
            probe = ("import json, scene3d_helpers as s3\n"
                     "ik = sorted(name for name, value in vars(s3).items() if not name.startswith('_') and any("
                     "used.startswith('_ik_') for used in getattr(getattr(value, '__code__', None), 'co_names', ())))\n"
                     "print(json.dumps([list(s3.API), ik]))\n")
            done = subprocess.run([sys.executable, "-P", "-c", probe], cwd=str(workspace), capture_output=True,
                                  text=True, timeout=60,
                                  env={**os.environ, "PYTHONPATH": str(workspace), "PYTHONDONTWRITEBYTECODE": "1"})
            self.assertEqual(done.returncode, 0, done.stderr)
            return json.loads(done.stdout)

        full, ik_functions = workspace_api(x5)
        self.assertEqual(full, scene3d.API)
        self.assertTrue(ik_functions)
        names_ik = lambda entry: any(name + "(" in entry for name in ik_functions)
        # IK_API is exactly the entries that name a function calling the IK prefilter.
        self.assertEqual([entry for entry in full if names_ik(entry)], list(s3.IK_API))
        self.assertEqual(list(s3.api_for(True)), scene3d.API)
        reduced = [entry for entry in scene3d.API if entry not in s3.IK_API]
        for profile in (so101, no_ik):
            listed, unstaged_ik = workspace_api(profile)
            self.assertEqual((listed, unstaged_ik), (reduced, ik_functions), profile.robot_model_id)
        fake = FakePoints(); self.addCleanup(fake.close)
        for profile, expected in ((no_ik, reduced), (x5, scene3d.API)):
            runtime = self.runtime(endpoint=fake.url)
            runtime.embodiment = profile
            value = handlers.geometry_query(runtime, {"request": self.request()}, {}).data["value"]
            self.assertEqual(value["api"], expected)

    def test_unavailable(self):
        result = handlers.geometry_query(self.runtime(), {"request": self.request()}, {}).data
        self.assertEqual(result["status"], "unknown")
        self.assertTrue(result["reason"].startswith("scene3d_unavailable:"), result)
        failing = FakePoints(status="no_horizontal_support_plane"); self.addCleanup(failing.close)
        failed = handlers.geometry_query(self.runtime(endpoint=failing.url), {"request": self.request()}, {}).data
        self.assertIn("no_horizontal_support_plane", failed["reason"])

    def test_in_hand_axis_and_stereo_triangulate_rays(self):
        from ni_rua.perception.external.object_table import triangulate_object_centres
        from support.perception_fixtures import T_SIDE
        # Build a synthetic vertical rod held in the gripper at [0.0, -0.25, 0.88] with robot=True
        zs = np.linspace(0.84, 0.94, 40)
        rod_xyz = np.column_stack([np.zeros(40), np.full(40, -0.25), zs]).astype(np.float32)
        cloud = s3.Cloud({
            "xyz": rod_xyz,
            "conf": np.full(40, 220, dtype=np.uint8),
            "uv": np.zeros((40, 2), dtype=np.uint16),
            "robot": np.ones(40, dtype=bool),
            "support_z": TABLE,
        })
        ih = s3.in_hand_axis([0.0, -0.25, 0.89], radius=0.065, cloud=cloud)
        self.assertEqual(ih["n"], 40)
        self.assertEqual(ih["posture"], "upright")
        self.assertLess(ih["tilt_deg"], 5.0)
        self.assertGreater(ih["elongation"], 5.0)

        # Stereo triangulation between cam_head (T_HEAD) and cam_side (T_SIDE) for true 3D point [0.1, -0.05, 0.2]
        true_pt = np.array([0.10, -0.05, 0.20])
        from ni_rua.robot.model.geometry import project
        cam1 = dict(K=np.asarray(K, float), D=np.zeros(5), T=np.asarray(T_HEAD, float), width=640, height=480)
        cam2 = dict(K=np.asarray(K, float), D=np.zeros(5), T=np.asarray(T_SIDE, float), width=640, height=480)
        uv1, _, _ = project(cam1, [true_pt.tolist()])
        uv2, _, _ = project(cam2, [true_pt.tolist()])
        tri = s3.triangulate_rays(uv1[0][0], uv1[0][1], K, T_HEAD, uv2[0][0], uv2[0][1], K, T_SIDE)
        self.assertEqual(tri["status"], "ok")
        np.testing.assert_allclose(tri["point_m"], true_pt, atol=1e-3)
        self.assertLess(tri["skew_gap_m"], 1e-4)

        anchored = triangulate_object_centres(
            cam1, cam2,
            [dict(centre_m=[0.10, -0.05, 0.175], top_z_m=0.215, bottom_z_m=0.135, label_px=uv1[0].tolist())],
            {"o1": uv2[0].tolist()},
        )
        self.assertEqual(len(anchored), 1)
        self.assertAlmostEqual(anchored[0]["dz_correction_m"], 0.025, places=3)
        self.assertAlmostEqual(anchored[0]["stereo_top_z_m"], 0.240, places=3)
        self.assertAlmostEqual(ih["bottom_point_m"][2], 0.84, places=2)
        self.assertAlmostEqual(ih["tip_offset_from_grasp_m"][2], -0.05, places=2)

        surf = s3.surface_extrema({"center": [0.10, 0.0], "radius": 0.05}, cloud=s3.from_bytes(synthetic_cloud()))
        self.assertAlmostEqual(surf["floor_z"], TABLE, places=3)
        self.assertAlmostEqual(surf["rim_z"], TABLE + 0.04, places=3)

    def test_in_process_scene3d_queries_and_batch_geometry_query(self):
        fake = FakePoints(); self.addCleanup(fake.close)
        runtime = self.runtime(endpoint=fake.url)
        registry = create_registry(runtime=runtime)
        first = handlers.geometry_query(runtime, {"request": self.request()}, {}).data
        artifact_ref = first["value"]["artifact"]
        self.assertEqual(len(fake.requests), 1)
        # In-process queries against stored artifact without any additional HTTP round-trip
        q_req = {
            "kind": "scene3d",
            "artifact": artifact_ref,
            "queries": [
                {"op": "top_z", "region": {"center": [-0.10, 0.10], "radius": 0.03}},
                {"op": "surface_extrema", "center": [0.10, 0.0], "radius": 0.05},
                {"op": "cluster_at_pixel", "pixel": [370, 240]},
                {"op": "in_hand_axis", "grasp_center_m": [0.0, -0.25, 0.25], "radius": 0.065},
                {"op": "in_hand_axis", "arm": "left", "radius": 0.065},
                {"op": "triangulate_rays", "pixel_a": [370, 240], "pixel_b": [120, 240], "camera_id_b": "cam_side"},
            ],
        }
        runtime.latest = {
            **self.scene.observation(runtime.identity, "obs-ot"),
            "robot_state": {
                "arms": {
                    "left": {
                        "ee_pose": {
                            "position_m": [0.0, -0.25, 0.40],
                            "quaternion_xyzw": [0.0, 0.70710678, 0.0, 0.70710678],
                        }
                    }
                }
            },
        }
        registry.validate("geometry_query", {"request": q_req}, role="actor")
        q_resp = handlers.geometry_query(runtime, {"request": q_req}, {}).data
        self.assertEqual(q_resp["status"], "conditional")
        self.assertEqual(len(fake.requests), 1)
        results = q_resp["value"]["query_results"]
        self.assertEqual(len(results), 6)
        self.assertAlmostEqual(results[0]["top_z"], TABLE + 0.10, places=3)
        self.assertAlmostEqual(results[1]["floor_z"], TABLE, places=3)
        self.assertAlmostEqual(results[1]["rim_z"], TABLE + 0.04, places=3)
        self.assertAlmostEqual(results[2]["top_z"], TABLE + 0.04, places=3)
        self.assertEqual(results[3]["op"], "in_hand_axis")
        self.assertNotEqual(results[3].get("status"), "error")
        self.assertEqual(results[4]["op"], "in_hand_axis")
        self.assertNotEqual(results[4].get("status"), "error")
        self.assertEqual(results[5]["op"], "triangulate_rays")
        self.assertEqual(results[5]["status"], "ok")
        # Top-level batch query combining pixel_ray and scene3d in one call
        batch_req = {
            "kind": "batch",
            "observation_id": "obs-ot",
            "artifact": artifact_ref,
            "queries": [
                {"kind": "pixel_ray", "camera_id": "cam_head", "pixel": [320, 240]},
                {"op": "top_z", "region": {"center": [-0.10, 0.10], "radius": 0.03}},
            ],
        }
        registry.validate("geometry_query", {"request": batch_req}, role="actor")
        b_resp = handlers.geometry_query(runtime, {"request": batch_req}, {}).data
        self.assertEqual(b_resp["query_type"], "batch")
        self.assertEqual(b_resp["value"]["count"], 2)
        self.assertEqual(b_resp["value"]["batch_results"][0]["query_type"], "pixel_ray")
        self.assertAlmostEqual(
            b_resp["value"]["batch_results"][1]["value"]["query_result"]["top_z"], TABLE + 0.10, places=3
        )

    def test_program_job_loads_the_artifact_through_the_helper(self):
        """Real ProgramRunner (sandbox off): geometry_query scene3d inside the job, then the helpers on the alias."""
        from ni_rua.runtime.programs.runner import ProgramRunner, ProgramServices
        fake = FakePoints(); self.addCleanup(fake.close)
        runtime = self.runtime(endpoint=fake.url)
        registry = create_registry(runtime=runtime)
        services = ProgramServices(store=self.artifacts, validate_tool=lambda name: registry.spec(name, role="program"),
            validate_arguments=lambda name, arguments: registry.validate(name, arguments, role="program"),
            invoke=lambda name, arguments, correlation: handlers.geometry_query(runtime, arguments, correlation),
            effect=lambda name, arguments: registry.spec(name, role="program").effect_for(arguments),
            observation=lambda: dict(observation_id="obs-ot", observation_profile="rgb_proprio_calibrated_v1"),
            action_active=lambda identifier: False, cancel_action=lambda identifier: False, sandbox_enabled=False,
            workspace_modules=scene3d.workspace_modules(runtime))
        runner = ProgramRunner(services); self.addCleanup(runner.close)
        source = ("import json\nfrom ni_rua_client import NiRuaClient\nimport scene3d_helpers as s3\n"
                  "r = NiRuaClient().call('geometry_query', {'request': {'kind': 'scene3d', 'observation_id': 'obs-ot'}})\n"
                  "a = r['data']['value']['artifact']\ns3.load(a)\n"
                  "print(json.dumps(dict(alias=a, info=s3.info(), top=s3.top_z({'center': [-0.10, 0.10], 'radius': 0.03}),"
                  " cl=s3.cluster_at_pixel(370, 240)['top_z'])))\n")
        job = runner.run(dict(source=source, timeout_s=60), dict(thread_id="t", turn_id="u", tool_call_id="c1"))
        self.assertEqual(job["status"], "succeeded", job)
        log = Path(self.artifacts.root) / "private" / "program-logs" / (job["job_id"] + ".stdout")
        out = json.loads(log.read_text().strip().splitlines()[-1])
        self.assertTrue(out["alias"].startswith("artifact:ref_"), out)
        self.assertAlmostEqual(out["top"]["top_z"], TABLE + 0.10, places=3)
        self.assertAlmostEqual(out["cl"], TABLE + 0.04, places=3)

    def test_high_elevation_wrist_views_preserved_without_table_corridor_clip(self):
        """验证高塔、立柜、高架操作时（z = 1.45m / 1.65m）腕部相机视角不再被台面高度走廊硬裁剪，非法外参仍正确拒绝。"""
        k_head = np.asarray(K, dtype=np.float64)
        # 构造斜俯视工作区的头部相机与位于 z=1.48m 水平观察立柜上层 (z=1.48m) 的腕部相机
        t_head_oblique = np.array([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -0.848, 0.530, -0.05],
            [0.0, 0.530, 0.848, 1.28],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        # 使 z_axis 指向 [0, -0.35, 0.80]：即 [0, -0.530, -0.848]
        t_head_oblique[:3, 1] = [0.0, 0.848, -0.530]
        t_head_oblique[:3, 2] = [0.0, -0.530, -0.848]

        # 高处腕部相机位于 [-0.08, -0.18, 1.48]，光轴 (+Z) 朝向立柜高层 [0.02, -0.36, 1.50]（远超旧版 sup_z + 0.42 = 1.185m 走廊！）
        fwd = np.array([0.10, -0.18, 0.02], dtype=np.float64)
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        t_wrist_high = np.eye(4, dtype=np.float64)
        t_wrist_high[:3, 0] = right
        t_wrist_high[:3, 1] = down
        t_wrist_high[:3, 2] = fwd
        t_wrist_high[:3, 3] = [-0.08, -0.18, 1.48]

        self.assertTrue(
            scene3d._wrist_view_overlaps_workspace(t_wrist_high, k_head, t_head_oblique, (480, 640), support_z_m=TABLE)
        )
        # 纯竖直俯视主相机 (T_HEAD, z=1.0m) 下，高于主相机的腕部相机 (z=1.48m) 同样保留
        t_wrist_topdown = t_wrist_high.copy()
        t_wrist_topdown[:3, 3] = [0.10, -0.05, 1.48]
        self.assertTrue(
            scene3d._wrist_view_overlaps_workspace(
                t_wrist_topdown, k_head, np.asarray(T_HEAD, dtype=np.float64), (480, 640), support_z_m=None
            )
        )

        # 背离工作区朝向后外侧的空闲腕部相机仍被过滤
        t_wrist_away = np.eye(4, dtype=np.float64)
        t_wrist_away[:3, 3] = [0.45, 0.20, 1.10]
        t_wrist_away[:3, 0] = [-1.0, 0.0, 0.0]
        t_wrist_away[:3, 1] = [0.0, 0.0, -1.0]
        t_wrist_away[:3, 2] = [0.0, 1.0, 0.0]
        self.assertFalse(
            scene3d._wrist_view_overlaps_workspace(t_wrist_away, k_head, t_head_oblique, (480, 640), support_z_m=TABLE)
        )

        # 非有限外参或退化旋转矩阵拒绝通过
        t_bad = t_wrist_high.copy()
        t_bad[0, 3] = np.nan
        self.assertFalse(scene3d._wrist_view_overlaps_workspace(t_bad))

    def test_grasp_center_safety_historical_obs_isolation_and_all_queries_failed(self):
        import io
        import urllib.error
        from unittest import mock
        from ni_rua.perception.external import ik_prefilter, scene3d_helpers as s3

        fake = FakePoints(); self.addCleanup(fake.close)
        runtime = self.runtime(endpoint=fake.url)

        # 1) NaN TCP / 缺失四元数的 X5 link6 姿态绝不静默返回 -15cm 法兰位置，历史 observation_id 绝不借用 runtime.latest
        runtime.latest = {
            "observation_id": "obs-latest",
            "robot_state": {
                "arms": {
                    "left": {
                        "ee_pose": {
                            "position_m": [0.10, -0.20, 0.85],
                            "quaternion_xyzw": [0.0, 0.70710678, 0.0, 0.70710678],
                        }
                    }
                }
            },
        }
        # 请求不存在/不同的历史帧 obs-ot（其存储 robot_state 无 arms），严禁借用 obs-latest 的 robot_state
        self.assertIsNone(scene3d._resolve_arm_grasp_center(runtime, "left", observation_id="obs-ot"))
        self.assertIsNone(scene3d._resolve_arm_grasp_center(runtime, "left", observation_id="obs-nonexistent"))

        # 同观测下：NaN 坐标、缺失四元数（X5 需 15.01cm 指尖前移）或 tcp_config_id 不匹配一律返回 None
        for bad_ee in (
            {"position_m": [float("nan"), 0.0, 0.85], "quaternion_xyzw": [0.0, 0.70710678, 0.0, 0.70710678]},
            {"position_m": [0.10, -0.20, 0.85]},  # 无四元数无法外推 +15.01cm 指尖中心
            {"position_m": [0.10, -0.20, 0.85], "quaternion_xyzw": [0.0, 0.0, 0.0, 0.0]},
            {"position_m": [0.10, -0.20, 0.85], "quaternion_xyzw": [0.0, 0.70710678, 0.0, 0.70710678], "tcp_config_id": "other-tcp"},
        ):
            runtime.latest = {"observation_id": "obs-latest", "robot_state": {"arms": {"left": {"ee_pose": bad_ee}}}}
            self.assertIsNone(scene3d._resolve_arm_grasp_center(runtime, "left", observation_id="obs-latest"))

        # 2) 所有子查询失败时顶层 status 诚实置为 "unknown"（reason="all_cloud_queries_failed:..."）
        first = handlers.geometry_query(runtime, {"request": {"kind": "scene3d", "observation_id": "obs-ot"}}, {}).data
        art = first["value"]["artifact"]
        all_bad = handlers.geometry_query(
            runtime,
            {"request": {"kind": "scene3d", "artifact": art, "queries": [{"op": "no_such_op_xyz"}]}},
            {},
        ).data
        self.assertEqual(all_bad["status"], "unknown")
        self.assertTrue(all_bad["reason"].startswith("all_cloud_queries_failed:"))
        self.assertEqual(all_bad["value"]["query_results"][0]["status"], "error")
        self.assertTrue(bool(all_bad["value"]["query_results"][0]["error"]))

        # 3) 工作区损坏的 _spatial_cloud.npz 与 _spatial_snapshot.json 清空陈旧缓存而不残留旧帧
        tmp_ws = Path(self.enterContext(tempfile.TemporaryDirectory()))
        valid_npz = tmp_ws / "_spatial_cloud.npz"
        valid_npz.write_bytes(scene3d._resolve_cloud_artifact(runtime, art))
        with mock.patch.object(s3, "_cloud_path", return_value=valid_npz), \
                mock.patch.object(s3, "_snapshot_path", return_value=tmp_ws / "_spatial_snapshot.json"):
            s3._DEFAULT = None
            s3._DEFAULT_SOURCE = None
            s3._DEFAULT_STAMP = None
            self.assertIsNotNone(s3._autoload_workspace_cloud())
            valid_npz.write_bytes(b"corrupt_not_a_zip")
            self.assertIsNone(s3._autoload_workspace_cloud())
            self.assertIsNone(s3._DEFAULT)

        bad_snap = tmp_ws / "_spatial_snapshot.json"
        bad_snap.write_text('{"robot_state": {"arms": {"left": {}}}}', encoding="utf-8")
        with mock.patch.object(ik_prefilter, "_WORKSPACE_SNAP_PATH", bad_snap):
            ik_prefilter._SNAPSHOT_CACHE = None
            self.assertIsNotNone(ik_prefilter._autoload_workspace_robot_state())
            bad_snap.write_text("{truncated_json", encoding="utf-8")
            ik_prefilter._SNAPSHOT_CACHE = (0, {"arms": {"left": {}}})
            self.assertIsNone(ik_prefilter._autoload_workspace_robot_state())

        # 4) scene3d.query_service 在 HTTPError 时关闭响应流
        closed_flag = {"closed": False}

        class _ClosableFP(io.BytesIO):
            def close(self):
                closed_flag["closed"] = True
                super().close()

        http_err = urllib.error.HTTPError("http://127.0.0.1:9/points", 503, "Unavailable", {}, _ClosableFP(b"busy"))
        with mock.patch("urllib.request.urlopen", side_effect=http_err):
            with self.assertRaisesRegex(scene3d.ServiceError, "http_503"):
                scene3d.query_service("http://127.0.0.1:9", {}, 1.0)
        self.assertTrue(closed_flag["closed"])

    def test_points_sha256_cache_hit_miss_retry_legacy_server_fallback_and_keepalive(self):
        from ni_rua.perception.external import object_table

        scene3d.close_pool()
        self.addCleanup(scene3d.close_pool)

        cloud_bytes = synthetic_cloud()
        srv_state = {"mode": "modern_hit", "requests": [], "client_ports": []}

        class DualKeepAliveHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            timeout = 5.0

            def log_message(self, *a):
                pass

            def _reply(self, code: int, obj: dict):
                raw = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                srv_state["requests"].append((self.path, body))
                srv_state["client_ports"].append(self.client_address[1])
                if self.path == "/object_table":
                    caps = ["points_image_sha256"] if srv_state["mode"].startswith("modern") else []
                    return self._reply(
                        200,
                        dict(
                            status="ok",
                            version="fake-ot",
                            caps=caps,
                            objects=[],
                            support_z_m=body["support_z_m"],
                            timing=dict(total_s=0.01),
                        ),
                    )
                mode = srv_state["mode"]
                if "image_b64" not in body:
                    if mode == "modern_miss_once":
                        srv_state["mode"] = "modern_hit"
                        return self._reply(
                            404,
                            dict(status="error", reason="image_cache_miss", error="image_cache_miss", http_status=404),
                        )
                    if mode == "legacy_422":
                        return self._reply(
                            422,
                            dict(
                                status="error",
                                reason="invalid_input",
                                error="ValueError: missing_required_field:image_b64",
                                http_status=422,
                            ),
                        )
                    if mode == "legacy_500":
                        return self._reply(
                            500,
                            dict(
                                status="error",
                                reason="internal_error",
                                error="KeyError: 'image_b64'",
                                http_status=500,
                            ),
                        )
                    if mode == "domain_422":
                        return self._reply(422, dict(status="error", error="no_horizontal_support_plane"))
                    if mode == "unrelated_500":
                        return self._reply(
                            500,
                            dict(status="error", reason="internal_error", error="ZeroDivisionError: boom", http_status=500),
                        )
                return self._reply(
                    200,
                    dict(
                        status="ok",
                        version="fake-points",
                        n_points=3000,
                        n_above_support=600,
                        n_robot=25,
                        voxel_m=body["voxel_m"],
                        support_z_m=body["support_z_m"],
                        bbox_m=[[-0.3, -0.3, 0.765], [0.3, 0.3, 1.055]],
                        bbox_above_support_m=[[-0.12, -0.02, 0.77], [0.13, 0.12, 0.865]],
                        npz_b64=base64.b64encode(cloud_bytes).decode(),
                        npz_bytes=len(cloud_bytes),
                        cache_hit=("image_b64" not in body),
                    ),
                )

        srv = ThreadingHTTPServer(("127.0.0.1", 0), DualKeepAliveHandler)
        srv.daemon_threads = True
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        ep = f"http://127.0.0.1:{srv.server_address[1]}"
        rt = self.runtime(endpoint=ep, object_table=True)
        expected_sha = hashlib.sha256(self.images["cam_head"]).hexdigest()

        # 1) /object_table uploads image_b64 -> subsequent /points omits image_b64, sends image_sha256, and reuses socket
        ot_res, _ = object_table.object_table_query(rt, {"kind": "object_table", "observation_id": "obs-ot"})
        self.assertEqual(ot_res["status"], "conditional")
        s3_res, s3_npz = scene3d.scene3d_query(rt, {"kind": "scene3d", "observation_id": "obs-ot"})
        self.assertEqual(s3_res["status"], "conditional")
        self.assertIsNotNone(s3_npz)
        self.assertEqual([p for p, _ in srv_state["requests"]], ["/object_table", "/points"])
        pts_req_1 = srv_state["requests"][1][1]
        self.assertNotIn("image_b64", pts_req_1)
        self.assertEqual(pts_req_1["image_sha256"], expected_sha)
        self.assertEqual(srv_state["client_ports"][0], srv_state["client_ports"][1])

        # 2) Typed 404 image_cache_miss retries ONCE with image_b64 and bounded remaining timeout
        srv_state["requests"].clear()
        srv_state["mode"] = "modern_miss_once"
        s3_miss_res, _ = scene3d.scene3d_query(rt, {"kind": "scene3d", "observation_id": "obs-ot"}, timeout_s=4.0)
        self.assertEqual(s3_miss_res["status"], "conditional")
        self.assertEqual([p for p, _ in srv_state["requests"]], ["/points", "/points"])
        self.assertNotIn("image_b64", srv_state["requests"][0][1])
        self.assertIn("image_b64", srv_state["requests"][1][1])
        self.assertLessEqual(srv_state["requests"][1][1]["timeout_s"], 4.0)

        # 3) Domain geometry error (422) and unrelated JSON 500 do NOT retry with image_b64 or flip sha_mode
        srv_state["requests"].clear()
        srv_state["mode"] = "domain_422"
        s3_dom_res, _ = scene3d.scene3d_query(rt, {"kind": "scene3d", "observation_id": "obs-ot"})
        self.assertEqual(s3_dom_res["status"], "unknown")
        self.assertIn("no_horizontal_support_plane", s3_dom_res["reason"])
        self.assertEqual(len(srv_state["requests"]), 1)

        srv_state["requests"].clear()
        srv_state["mode"] = "unrelated_500"
        s3_500_res, _ = scene3d.scene3d_query(rt, {"kind": "scene3d", "observation_id": "obs-ot"})
        self.assertEqual(s3_500_res["status"], "unknown")
        self.assertIn("ZeroDivisionError", s3_500_res["reason"])
        self.assertEqual(len(srv_state["requests"]), 1)
        self.assertEqual(scene3d.get_endpoint_sha_mode(ep), "supported")

        # 3b) Concurrent inflight upload counter from object_table is untouched by standalone scene3d upload
        from ni_rua.perception.external import _service_http
        inflight_key = _service_http.make_image_cache_key("a" * 64, K, T_HEAD, TABLE)
        _service_http.note_image_upload_started(ep, inflight_key)
        _service_http.note_image_uploaded(ep, inflight_key)
        self.assertEqual(_service_http._ENDPOINT_STATES[ep].inflight.get(inflight_key), 1)
        _service_http.note_image_upload_finished(ep, inflight_key, ok=True)
        self.assertNotIn(inflight_key, _service_http._ENDPOINT_STATES[ep].inflight)

        # 4) Legacy servers (422 missing_required_field:image_b64, 500 KeyError: 'image_b64', and generic JSON 500
        #    while sha_mode == "unknown") fall back seamlessly on first call and mark the endpoint unsupported
        for leg_mode in ("legacy_422", "legacy_500", "unrelated_500"):
            with self.subTest(legacy_mode=leg_mode):
                scene3d.close_pool()
                srv_state["requests"].clear()
                srv_state["mode"] = leg_mode
                object_table.object_table_query(rt, {"kind": "object_table", "observation_id": "obs-ot"})
                self.assertEqual(scene3d.get_endpoint_sha_mode(ep), "unknown")
                r_first, _ = scene3d.scene3d_query(rt, {"kind": "scene3d", "observation_id": "obs-ot"})
                self.assertEqual(r_first["status"], "conditional")
                self.assertEqual([p for p, _ in srv_state["requests"]], ["/object_table", "/points", "/points"])
                self.assertNotIn("image_b64", srv_state["requests"][1][1])
                self.assertIn("image_b64", srv_state["requests"][2][1])
                self.assertEqual(scene3d.get_endpoint_sha_mode(ep), "unsupported")

                srv_state["requests"].clear()
                r_second, _ = scene3d.scene3d_query(rt, {"kind": "scene3d", "observation_id": "obs-ot"})
                self.assertEqual(r_second["status"], "conditional")
                self.assertEqual([p for p, _ in srv_state["requests"]], ["/points"])
                self.assertIn("image_b64", srv_state["requests"][0][1])


if __name__ == "__main__":
    unittest.main()
