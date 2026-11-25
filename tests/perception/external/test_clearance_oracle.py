"""三维 OBB 15 轴分离轴距离、无状态向量化净空评估核 (`clearance_oracle`) 与多阶段轨迹推演回归测试。"""
from __future__ import annotations

NI_RUA_TEST_TIER = "fast"

import io
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ni_rua.perception.external.clearance_oracle import (
    evaluate_grasp_clearance,
    evaluate_grasp_clearance_batch,
    simulate_stage_clearances,
)
from ni_rua.perception.external.object_table import grasp_hints_from
from ni_rua.perception.external.pointcloud_field import PointCloudSpatialField
from ni_rua.perception.external.vectorized_grasp import (
    angles_deg_to_matrices_batch,
    batch_obbs_to_slabs_min_distance,
    evaluate_grasp_batch,
    obb_signed_distance_pairs,
)
from ni_rua.perception.external.virtual_field import (
    OBB3D,
    VirtualSpatialField,
    angles_deg_to_quat_wxyz,
    obb_signed_distance,
    quat_wxyz_to_matrix,
)
from ni_rua.runtime.actions.grasp_preview_3d import audit_batch_grasp_3d

TABLE_COLUMNS = [
    "id",
    "centre_m",
    "top_z",
    "bottom_z",
    "length",
    "width",
    "yaw_deg",
    "tilt_deg",
    "support",
    "quality",
]


def _random_orthonormal_axes(rng: np.random.Generator) -> np.ndarray:
    """生成随机右手正交旋转矩阵 `(3, 3)`。"""
    mat = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(mat)
    q = q @ np.diag(np.where(np.diag(r) >= 0.0, 1.0, -1.0))
    if np.linalg.det(q) < 0.0:
        q[:, 0] = -q[:, 0]
    return q


def _hollow_container_npz(
    center_xy: tuple[float, float] = (0.28, -0.10),
    table_z: float = 0.765,
    outer_size_m: float = 0.072,
    wall_thickness_m: float = 0.008,
    height_m: float = 0.055,
    track_num: int = 1,
) -> bytes:
    """构造带真实薄壁与内底的三维点云 NPZ 字节流，用于验证精细 `LocalSurfaceKDTree` 向量化评估。"""
    cx, cy = center_xy
    half_out = 0.5 * outer_size_m
    half_in = half_out - wall_thickness_m
    xs = np.linspace(cx - half_out, cx + half_out, 15)
    ys = np.linspace(cy - half_out, cy + half_out, 15)
    zs = np.linspace(table_z + 0.006, table_z + height_m, 8)

    pts_list: list[np.ndarray] = []
    for z in zs:
        gxx, gyy = np.meshgrid(xs, ys)
        ring = (np.abs(gxx - cx) >= half_in) | (np.abs(gyy - cy) >= half_in)
        pts_list.append(np.column_stack([gxx[ring], gyy[ring], np.full(int(np.sum(ring)), z)]))
    fxx, fyy = np.meshgrid(xs, ys)
    pts_list.append(np.column_stack([fxx.ravel(), fyy.ravel(), np.full(fxx.size, table_z + 0.006)]))
    xyz = np.vstack(pts_list).astype(np.float32)

    buf = io.BytesIO()
    np.savez(
        buf,
        xyz=xyz,
        conf=np.full(len(xyz), 0.9, dtype=np.float32),
        robot=np.zeros(len(xyz), dtype=bool),
        obj_id=np.full(len(xyz), track_num, dtype=np.int16),
        table_z=np.float64(table_z),
    )
    return buf.getvalue()


class VectorizedGraspTest(unittest.TestCase):
    """验证向量化 OBB 距离、实心/精细表面抓取净空评估、多阶段轨迹推演与 12 物体耗时基准。"""

    obbs_a: list[OBB3D]
    obbs_b: list[OBB3D]
    dense_scenes: list[tuple[int, float, list[dict]]]
    fine_pcf: PointCloudSpatialField
    bench_12_objects: list[dict]

    @classmethod
    def setUpClass(cls) -> None:
        rng_obb = np.random.default_rng(42)
        obbs_a: list[OBB3D] = []
        obbs_b: list[OBB3D] = []
        for i in range(320):
            ax_a = _random_orthonormal_axes(rng_obb)
            # 覆盖平行棱叉积退化 (norm <= 1e-6) 与一般斜交姿态
            if i % 5 == 0:
                ax_b = ax_a.copy()
            elif i % 7 == 0:
                ax_b = ax_a[:, [1, 2, 0]]
            else:
                ax_b = _random_orthonormal_axes(rng_obb)

            ha = rng_obb.uniform(0.002, 0.09, size=3)
            hb = rng_obb.uniform(0.002, 0.09, size=3)
            if i % 11 == 0:
                ha[2] = 1e-5  # 覆盖极薄退化半轴
            ca = rng_obb.uniform(-0.25, 0.25, size=3)
            spread = 0.02 if i < 120 else (0.08 if i < 240 else 0.30)
            cb = ca + rng_obb.uniform(-spread, spread, size=3)
            obbs_a.append(OBB3D(ca, ax_a, ha))
            obbs_b.append(OBB3D(cb, ax_b, hb))
        cls.obbs_a = obbs_a
        cls.obbs_b = obbs_b

        dense_scenes: list[tuple[int, float, list[dict]]] = []
        for seed in (200, 201):
            rng = np.random.default_rng(seed)
            n_obj = 3
            table_z = float(np.round(rng.uniform(0.73, 0.78), 3))
            objs: list[dict] = []
            base_x = float(rng.uniform(0.24, 0.34))
            base_y = float(rng.uniform(-0.08, 0.08))
            for j in range(n_obj):
                elong = bool(rng.random() < 0.7)
                length = float(np.round(rng.uniform(0.085, 0.175) if elong else rng.uniform(0.024, 0.072), 4))
                width = float(np.round(rng.uniform(0.016, 0.044) if elong else rng.uniform(0.022, min(length, 0.072)), 4))
                height = float(np.round(rng.uniform(0.006, 0.014) if j % 2 == 0 else rng.uniform(0.018, 0.075), 4))
                bottom_z = table_z + (0.0 if rng.random() < 0.8 else float(np.round(rng.uniform(0.01, 0.04), 4)))
                top_z = round(bottom_z + height, 4)
                centre_z = round(0.5 * (bottom_z + top_z), 4)
                support = "held_left" if (j == 0 and seed % 2 == 0) else "table"
                objs.append(
                    dict(
                        id=f"o{j + 1}",
                        centre_m=[
                            round(base_x + float(rng.uniform(-0.075, 0.075)), 4),
                            round(base_y + float(rng.uniform(-0.075, 0.075)), 4),
                            centre_z,
                        ],
                        top_z_m=top_z,
                        bottom_z_m=round(bottom_z, 4),
                        length_m=max(length, width),
                        width_m=min(length, width),
                        yaw_deg=round(float(rng.uniform(-89.5, 89.5)), 1),
                        tilt_deg=round(float(rng.uniform(25.0, 90.0)), 1) if rng.random() < 0.75 else None,
                        support=support,
                    )
                )
            dense_scenes.append((seed, table_z, objs))
        cls.dense_scenes = dense_scenes

        fine_table = dict(
            columns=TABLE_COLUMNS,
            table_z=0.765,
            rows=[
                ["o1", [0.28, -0.10, 0.7925], 0.820, 0.765, 0.072, 0.072, 0.0, 90.0, "table", "ok"],
                ["o2", [0.28, -0.025, 0.785], 0.805, 0.765, 0.120, 0.028, 15.0, 86.0, "table", "ok"],
            ],
        )
        cls.fine_pcf = PointCloudSpatialField(
            objects=fine_table,
            table_z=0.765,
            cloud_npz=_hollow_container_npz(),
            cavity_depths_m={"o1": 0.048},
        )

        rng_bench = np.random.default_rng(2026)
        bench_objs: list[dict] = []
        for i in range(12):
            gx = 0.24 + (i % 4) * 0.095 + float(rng_bench.uniform(-0.012, 0.012))
            gy = -0.16 + (i // 4) * 0.105 + float(rng_bench.uniform(-0.012, 0.012))
            yaw = float(rng_bench.uniform(-89.0, 89.0))
            tilt = round(90.0 - float(rng_bench.uniform(0.0, 28.0)), 1)
            bench_objs.append(
                dict(
                    id=f"o{i + 1}",
                    centre_m=[round(gx, 4), round(gy, 4), 0.780],
                    top_z_m=0.798,
                    bottom_z_m=0.762,
                    length_m=0.140,
                    width_m=0.030,
                    yaw_deg=round(yaw, 1),
                    tilt_deg=tilt,
                )
            )
        cls.bench_12_objects = bench_objs

    def _assert_grasp_dict_close(self, vec_dict: dict, sca_dict: dict, tol: float = 1e-6) -> None:
        """严格断言批量与单次返回字典键集一致、离散字段相等且全部数值误差 `< tol`。"""
        self.assertNotIn("optimal_grasp_pose", vec_dict)
        self.assertNotIn("optimal_grasp_pose", sca_dict)
        self.assertEqual(set(vec_dict.keys()), set(sca_dict.keys()))
        for key in sca_dict:
            v_val = vec_dict[key]
            s_val = sca_dict[key]
            if isinstance(s_val, bool):
                self.assertIs(v_val, s_val, msg=f"布尔字段不一致: {key}: {v_val} != {s_val}")
            elif isinstance(s_val, (int, float)):
                self.assertLess(
                    abs(float(v_val) - float(s_val)),
                    tol,
                    msg=f"标量字段误差超限: {key}: {v_val} vs {s_val}",
                )
            elif isinstance(s_val, list) and s_val and isinstance(s_val[0], (int, float)):
                self.assertEqual(len(v_val), len(s_val), msg=f"向量长度不一致: {key}")
                for idx, (vv, sv) in enumerate(zip(v_val, s_val)):
                    self.assertLess(
                        abs(float(vv) - float(sv)),
                        tol,
                        msg=f"向量字段误差超限: {key}[{idx}]: {vv} vs {sv}",
                    )
            else:
                self.assertEqual(v_val, s_val, msg=f"结构字段不一致: {key}: {v_val} != {s_val}")
        self.assertEqual(vec_dict, sca_dict)

    def test_angles_deg_to_matrices_batch_matches_scalar(self) -> None:
        """验证批量欧拉角转法兰旋转矩阵与逐个四元数构造精度一致。"""
        rng = np.random.default_rng(7)
        angles = rng.uniform(-180.0, 180.0, size=(64, 3))
        batch_mats = angles_deg_to_matrices_batch(angles)
        for i in range(angles.shape[0]):
            ref = quat_wxyz_to_matrix(angles_deg_to_quat_wxyz(angles[i, 0], angles[i, 1], angles[i, 2]))
            np.testing.assert_allclose(batch_mats[i], ref, atol=1e-12, rtol=0.0)

    def test_obb_signed_distance_batch_matches_scalar(self) -> None:
        """验证穿透、分离、平行退化轴、超薄片 OBB 与多块最小值核的 15 轴有符号距离完全一致。"""
        scalar_d = np.array([obb_signed_distance(a, b) for a, b in zip(self.obbs_a, self.obbs_b)], dtype=np.float64)
        vec_d = obb_signed_distance_pairs(self.obbs_a, self.obbs_b)
        max_err = float(np.max(np.abs(vec_d - scalar_d)))
        self.assertLess(max_err, 1e-9)

        # 验证 skip_above_m 阈值内 100% 精确、阈值外严格大于阈值
        cutoff = 0.05
        culled_d = obb_signed_distance_pairs(self.obbs_a, self.obbs_b, skip_above_m=cutoff)
        within = scalar_d <= cutoff
        np.testing.assert_allclose(culled_d[within], scalar_d[within], atol=1e-9, rtol=0.0)
        self.assertTrue(np.all(culled_d[~within] > cutoff))

        # 验证 batch_obbs_to_slabs_min_distance 与逐对全局最小值完全一致
        moving_sub = self.obbs_a[:12]
        slabs_sub = self.obbs_b[:7]
        ref_min = min(obb_signed_distance(m_obb, s_obb) for m_obb in moving_sub for s_obb in slabs_sub)
        self.assertLess(abs(batch_obbs_to_slabs_min_distance(moving_sub, slabs_sub) - ref_min), 1e-9)

    def test_batch_and_fine_clearance_equivalence_across_dense_tilted_scenes(self) -> None:
        """在多随机种子密集多物体、细长倾斜、薄片贴桌、持物及精细 KDTree 凹腔场景下断言批量与单次评估误差 `< 1e-6`。"""
        eval_count = 0
        seen_regions: set[str] = set()
        seen_neg_depth = False
        combo_specs = (
            ("pad_center", -0.006),
            ("fingertip", None),
            ("finger_middle", 0.012),
        )
        for scene_idx, (seed, table_z, objs) in enumerate(self.dense_scenes):
            rng = np.random.default_rng(seed + 1000)
            vf = VirtualSpatialField(objects=objs, table_z=table_z)
            oids = list(vf.objects.keys())
            selected_oids = [oids[0], oids[1]] if scene_idx == 0 else [oids[2]]
            for sub_idx, oid in enumerate(selected_oids):
                region, custom_depth = combo_specs[(scene_idx * 2 + sub_idx) % len(combo_specs)]
                seen_regions.add(region)
                if custom_depth is not None and custom_depth < 0.0:
                    seen_neg_depth = True
                arm = "left" if ((scene_idx + sub_idx) % 2 == 0) else "right"
                aperture = round(float(rng.uniform(0.042, 0.088)), 4)

                target = vf.find_target_object(oid)
                assert target is not None
                base_c = np.asarray(target.centre_m, dtype=np.float64)
                pos_samples = np.vstack(
                    [
                        base_c,
                        base_c + np.array([0.0, 0.0, -0.012]),
                        base_c + rng.uniform(-0.03, 0.03, size=(4, 3)),
                    ]
                )
                ang_samples = np.vstack(
                    [
                        np.array([[0.0, 0.0, float(target.yaw_deg) + 90.0], [0.0, 0.0, float(target.yaw_deg)]]),
                        rng.uniform([-30.0, -30.0, -180.0], [30.0, 30.0, 180.0], size=(4, 3)),
                    ]
                )
                batch_evals = evaluate_grasp_batch(
                    vf,
                    target,
                    pos_samples,
                    ang_samples,
                    arm=arm,
                    open_aperture_m=aperture,
                    contact_region=region,
                    contact_depth_m=custom_depth,
                )
                self.assertEqual(len(batch_evals), 6)
                for idx in range(len(batch_evals)):
                    sca_eval = vf.evaluate_grasp(
                        arm=arm,
                        position_m=pos_samples[idx],
                        angles_deg=ang_samples[idx],
                        target_id=oid,
                        open_aperture_m=aperture,
                        contact_region=region,
                        contact_depth_m=custom_depth,
                    )
                    self._assert_grasp_dict_close(batch_evals[idx], sca_eval, tol=1e-6)
                    eval_count += 1
        self.assertEqual(seen_regions, {"pad_center", "fingertip", "finger_middle"})
        self.assertTrue(seen_neg_depth)

        # 覆盖精细 LocalSurfaceKDTree + 凹腔边缘抓取 (rim grasp) 与邻物碰撞的批量==单次一致性
        fine_target = self.fine_pcf.objects["o1"]
        fine_positions = np.array(
            [
                [0.28, -0.132, 0.812],  # 对准薄壁边缘抓取
                [0.28, -0.100, 0.792],  # 宽口容器正中心跨抓（超开度）
                [0.28, -0.068, 0.812],  # 靠近邻物 o2 的另一侧薄壁
                [0.28, -0.132, 0.760],  # 穿透桌面支撑面
                [0.35, -0.200, 0.812],  # 偏离目标足迹
                [0.28, -0.132, 0.812],  # 斜向滚转姿态
            ],
            dtype=np.float64,
        )
        fine_angles = np.array(
            [
                [0.0, 0.0, 90.0],  # 沿 Y 轴横跨 8mm 薄壁抓取
                [0.0, 0.0, 0.0],   # 容器正中心沿 X 轴跨抓（72mm > 60mm 超开度）
                [0.0, 0.0, 90.0],  # 靠近邻物 o2 的另一侧薄壁
                [0.0, 0.0, 90.0],  # 穿透桌面支撑面
                [0.0, 0.0, 90.0],  # 偏离目标足迹
                [12.0, -8.0, 75.0],  # 斜向滚转姿态
            ],
            dtype=np.float64,
        )
        fine_batch = evaluate_grasp_clearance_batch(
            self.fine_pcf,
            fine_target,
            fine_positions,
            fine_angles,
            arm="right",
            open_aperture_m=0.060,
            contact_region="fingertip",
        )
        self.assertTrue(fine_batch[0]["theoretically_graspable"], fine_batch[0])
        self.assertFalse(fine_batch[1]["theoretically_graspable"])
        self.assertIn("projected_span_exceeds_opening", fine_batch[1]["failure_reasons"])
        for idx in range(len(fine_batch)):
            sca_fine = self.fine_pcf.evaluate_grasp(
                arm="right",
                position_m=fine_positions[idx],
                angles_deg=fine_angles[idx],
                target_id="o1",
                open_aperture_m=0.060,
                contact_region="fingertip",
            )
            self._assert_grasp_dict_close(fine_batch[idx], sca_fine, tol=1e-6)
            eval_count += 1

        self.assertGreaterEqual(eval_count, 22)

    def test_object_table_skips_solve_for_single_compact_object(self) -> None:
        """验证 `grasp_hints_from` 纯粹输出客观几何测量（不触发批量穷举求解，不注入 `safe_grasp_pose`）。"""
        compact_a = dict(
            centre_m=[0.28, -0.10, 0.785],
            top_z_m=0.805,
            bottom_z_m=0.765,
            length_m=0.045,
            width_m=0.040,
            yaw_deg=15.0,
        )
        compact_b = dict(
            centre_m=[0.28, 0.08, 0.785],
            top_z_m=0.805,
            bottom_z_m=0.765,
            length_m=0.046,
            width_m=0.038,
            yaw_deg=-20.0,
        )
        elongated_single = dict(
            centre_m=[0.28, -0.10, 0.780],
            top_z_m=0.795,
            bottom_z_m=0.765,
            length_m=0.140,
            width_m=0.026,
            yaw_deg=30.0,
            tilt_deg=88.0,
        )

        with patch(
            "ni_rua.perception.external.clearance_oracle.evaluate_grasp_clearance_batch",
            wraps=evaluate_grasp_clearance_batch,
        ) as spy_batch:
            hints_single_compact = grasp_hints_from([compact_a], support_z_m=0.765)
            self.assertEqual(spy_batch.call_count, 0)
            self.assertNotIn("safe_grasp_pose", hints_single_compact["o1"])
            self.assertNotIn("optimal_grasp_pose", hints_single_compact["o1"])

            hints_single_elong = grasp_hints_from([elongated_single], support_z_m=0.765)
            self.assertEqual(spy_batch.call_count, 0)
            self.assertNotIn("safe_grasp_pose", hints_single_elong["o1"])
            self.assertTrue(hints_single_elong["o1"]["elongated"])
            self.assertEqual(len(hints_single_elong["o1"]["across_yaw_deg"]), 2)

            hints_two_compact = grasp_hints_from([compact_a, compact_b], support_z_m=0.765)
            self.assertEqual(spy_batch.call_count, 0)
            self.assertNotIn("safe_grasp_pose", hints_two_compact["o1"])
            self.assertNotIn("safe_grasp_pose", hints_two_compact["o2"])

    def test_preview_audit_and_clearance_respond_to_geometry_change(self) -> None:
        """验证 `grasp_preview_3d` 与无状态 `clearance_oracle` 客观报告确定性，且邻物位移立即反映至净空测量。"""
        table = dict(
            columns=TABLE_COLUMNS,
            table_z=0.765,
            rows=[
                ["o1", [0.28, -0.12, 0.780], 0.795, 0.765, 0.135, 0.028, 0.0, 90.0, "table", "ok"],
                ["o2", [0.28, -0.04, 0.780], 0.795, 0.765, 0.135, 0.028, 0.0, 90.0, "table", "ok"],
            ],
        )
        harness = SimpleNamespace(_obs3d_memo=("obs-cache-1", {"object_table": (table, None, {})}))
        action = {
            "based_on_observation_id": "obs-cache-1",
            "stages": [
                {
                    "id": "grasp_o1",
                    "kind": "move",
                    "arm": "right",
                    "action": {
                        "input_grasp_pose": {
                            "position_m": [0.28, -0.12, 0.780],
                            "angles_deg": [0.0, 0.0, 90.0],
                        },
                        "gripper_events": [{"goal_normalized": 0.0}],
                    },
                }
            ],
        }

        rev1, targets1 = audit_batch_grasp_3d(harness, action, "obs-cache-1")
        rev2, targets2 = audit_batch_grasp_3d(harness, action, "obs-cache-1")
        self.assertEqual(rev1["blocking_risks"], [])
        self.assertEqual(targets1, targets2)
        self.assertEqual(rev1, rev2)
        self.assertNotIn("optimal_grasp_pose", str(rev1))
        self.assertNotIn("suggested_fixes", rev1["virtual_field_simulation"])

        vf = VirtualSpatialField(objects=table, table_z=0.765)
        r_before = vf.evaluate_grasp("right", [0.28, -0.12, 0.780], [0.0, 0.0, 90.0], target_id="o1")
        sim_before = simulate_stage_clearances(vf, action["stages"])
        vf.objects["o2"].centre_m[1] = -0.062
        r_after = vf.evaluate_grasp("right", [0.28, -0.12, 0.780], [0.0, 0.0, 90.0], target_id="o1")
        sim_after = simulate_stage_clearances(vf, action["stages"])
        self.assertLess(r_after["neighbor_clearance_m"], r_before["neighbor_clearance_m"])
        self.assertLess(
            sim_after["min_clearance_by_object_m"]["o2"],
            sim_before["min_clearance_by_object_m"]["o2"],
        )

    def test_stateless_dict_list_and_stage_simulation_consistency(self) -> None:
        """验证 `evaluate_grasp_clearance_batch` 在场实例、字典与对象列表输入下结果一致，且多阶段抓取-搬运-释放推演无副作用。"""
        objs = [
            dict(
                id="o1",
                centre_m=[0.28, -0.10, 0.780],
                top_z_m=0.795,
                bottom_z_m=0.765,
                length_m=0.130,
                width_m=0.028,
                yaw_deg=-35.0,
                tilt_deg=88.0,
            ),
            dict(
                id="o2",
                centre_m=[0.33, -0.02, 0.782],
                top_z_m=0.800,
                bottom_z_m=0.765,
                length_m=0.050,
                width_m=0.040,
                yaw_deg=10.0,
            ),
        ]
        vf = VirtualSpatialField(objects=objs, table_z=0.765)
        target = vf.objects["o1"]
        pose_p = [0.28, -0.10, 0.780]
        pose_a = [0.0, 0.0, 55.0]

        from_field = evaluate_grasp_clearance_batch(vf, target, [pose_p], [pose_a])[0]
        from_dict = evaluate_grasp_clearance_batch(vf.objects, target, [pose_p], [pose_a], table_z=0.765)[0]
        from_list = evaluate_grasp_clearance_batch(list(vf.objects.values()), target, [pose_p], [pose_a], table_z=0.765)[0]
        from_scalar = evaluate_grasp_clearance(vf, "right", pose_p, pose_a, target_id="o1")

        self._assert_grasp_dict_close(from_field, from_dict, tol=1e-9)
        self._assert_grasp_dict_close(from_field, from_list, tol=1e-9)
        self._assert_grasp_dict_close(from_field, from_scalar, tol=1e-9)

        stages = [
            {"id": "grasp", "arm": "right", "position_m": [0.28, -0.10, 0.780], "angles_deg": [0.0, 0.0, 55.0], "gripper": 0.0},
            {"id": "lift", "arm": "right", "position_m": [0.28, -0.10, 0.860], "angles_deg": [0.0, 0.0, 55.0]},
            {"id": "place", "arm": "right", "position_m": [0.20, -0.18, 0.782], "angles_deg": [0.0, 0.0, 55.0], "gripper": 1.0},
        ]
        sim_res = vf.simulate_stages(stages)
        self.assertEqual(sim_res["stage_reports"][0]["grasped_object_id"], "o1")
        self.assertEqual(sim_res["stage_reports"][1]["held_object_id"], "o1")
        self.assertEqual(sim_res["stage_reports"][2]["released_object_id"], "o1")
        self.assertEqual(sim_res["stage_reports"][2]["new_support"], "table")
        self.assertNotIn("recommended_safe_z_m", sim_res)
        self.assertNotIn("suggested_fixes", sim_res)
        # 源场实例保持零副作用
        self.assertIsNone(vf.objects["o1"].held_by_arm)
        self.assertEqual(vf.objects["o1"].centre_m, [0.28, -0.10, 0.780])

    def test_benchmark_12_elongated_objects_grasp_hints_under_150ms(self) -> None:
        """基准断言：12 个细长倾斜物体的 `grasp_hints_from()` 与 64 姿态批量净空评估耗时均严格 `< 0.15s`。"""
        grasp_hints_from(self.bench_12_objects[:2], support_z_m=0.762)

        t0 = time.perf_counter()
        hints = grasp_hints_from(self.bench_12_objects, support_z_m=0.762)
        elapsed_hints_s = time.perf_counter() - t0

        self.assertEqual(len(hints), 12)
        self.assertTrue(all(hints[f"o{i + 1}"]["elongated"] for i in range(12)))
        self.assertLess(
            elapsed_hints_s,
            0.15,
            msg=f"12 细长物体 grasp_hints_from 耗时 {elapsed_hints_s:.4f}s 超过 0.15s 预算",
        )

        vf = VirtualSpatialField(objects=self.bench_12_objects, table_z=0.762)
        target = vf.objects["o1"]
        rng = np.random.default_rng(99)
        cand_pos = np.asarray(target.centre_m, dtype=np.float64)[None, :] + rng.uniform(-0.02, 0.02, size=(64, 3))
        cand_ang = rng.uniform([-15.0, -15.0, -180.0], [15.0, 15.0, 180.0], size=(64, 3))
        evaluate_grasp_clearance_batch(vf, target, cand_pos[:2], cand_ang[:2])

        t1 = time.perf_counter()
        batch_out = evaluate_grasp_clearance_batch(vf, target, cand_pos, cand_ang)
        elapsed_batch_s = time.perf_counter() - t1
        self.assertEqual(len(batch_out), 64)
        self.assertLess(
            elapsed_batch_s,
            0.15,
            msg=f"12 物体场景 64 候选批量净空评估耗时 {elapsed_batch_s:.4f}s 超过 0.15s 预算",
        )

    def test_vectorized_matches_scalar_with_none_table_z_and_distant_tall_object(self) -> None:
        """验证无实测 `table_z=None` 及远端高耸障碍物 (1.30m) 场景下批量与单次评估 100% 一致，且轨迹推演无天际线污染。"""
        objs = [
            dict(
                id="o1",
                centre_m=[0.18, -0.16, 0.785],
                top_z_m=0.805,
                bottom_z_m=0.765,
                length_m=0.130,
                width_m=0.028,
                yaw_deg=15.0,
                tilt_deg=88.0,
                support="unknown",
            ),
            dict(
                id="o_tall_far",
                centre_m=[0.42, 0.25, 1.030],
                top_z_m=1.300,
                bottom_z_m=0.765,
                length_m=0.060,
                width_m=0.060,
                yaw_deg=0.0,
                support="unknown",
            ),
        ]
        vf = VirtualSpatialField(objects=objs, table_z=None)
        vec_ev = evaluate_grasp_batch(vf, vf.objects["o1"], [[0.18, -0.16, 0.785]], [[0.0, 0.0, -75.0]])[0]
        sca_ev = vf.evaluate_grasp("right", [0.18, -0.16, 0.785], [0.0, 0.0, -75.0], target_id="o1")
        self._assert_grasp_dict_close(vec_ev, sca_ev, tol=1e-6)
        self.assertIsNone(vec_ev["support_clearance_m"])
        self.assertTrue(vec_ev["theoretically_graspable"], vec_ev)

        sim = simulate_stage_clearances(
            vf,
            [
                {"id": "grasp_o1", "arm": "right", "position_m": [0.18, -0.16, 0.785], "angles_deg": [0.0, 0.0, -75.0], "gripper": 0.0},
                {"id": "lift_local", "arm": "right", "position_m": [0.18, -0.16, 0.860], "angles_deg": [0.0, 0.0, -75.0]},
            ],
        )
        self.assertEqual(sim["status"], "clean")
        self.assertTrue(sim["theoretically_valid"])
        self.assertGreater(sim["min_clearance_by_object_m"]["o_tall_far"], 0.15)
        self.assertNotIn("safe_transit_z_m", sim)
        self.assertNotIn("recommended_safe_z_m", sim)


if __name__ == "__main__":
    unittest.main()
