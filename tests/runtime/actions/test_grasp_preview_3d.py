"""3D 感知驱动的零回合税 grasp_hints 与 robot_action(mode='preview') grasp_3d_review 单元测试。"""
import math
from types import SimpleNamespace
import unittest
import numpy as np
from ni_rua.robot.motion.arx_actions import compile_action
from ni_rua.runtime.actions.grasp_preview_3d import ADVISORY_NOTE_CODES, audit_batch_grasp_3d
from ni_rua.perception.external.object_table import grasp_hints_from
from ni_rua.perception.overlays.pose_axis_overlay import draw_object_long_axis_overlay

NI_RUA_TEST_TIER = "fast"


COLUMNS = ["id", "centre_m", "top_z", "bottom_z", "length", "width", "yaw_deg", "tilt_deg", "support", "quality"]
TABLE = dict(
    columns=COLUMNS,
    table_z=0.765,
    rows=[
        ["o1", [0.10, -0.20, 0.780], 0.792, 0.765, 0.140, 0.025, 30.0, 88.0, "table", "ok"],
    ],
)


class GraspPreview3DTest(unittest.TestCase):
    def test_zero_tax_grasp_hints_from_objects(self):
        objects = [
            dict(centre_m=[0.10, -0.20, 0.780], top_z_m=0.792, bottom_z_m=0.765, length_m=0.140, width_m=0.025, yaw_deg=30.0, axis_tilt_deg=88.0, elongation=2.5),
            dict(centre_m=[-0.21, -0.23, 0.778], top_z_m=0.791, bottom_z_m=0.765, length_m=0.045, width_m=0.038, yaw_deg=89.0),
            dict(centre_m=[-0.33, -0.02, 0.834], top_z_m=0.875, bottom_z_m=0.793, length_m=0.116, width_m=0.116, yaw_deg=13.0),
        ]
        hints = grasp_hints_from(objects, support_z_m=0.765)
        self.assertIn("o1", hints)
        self.assertEqual(hints["o1"]["across_yaw_deg"], [-60.0, 120.0])
        self.assertEqual(hints["o1"]["pad_z_m"], 0.78)
        self.assertEqual(hints["o1"]["opening_check"], "ok")
        self.assertTrue(hints["o1"]["elongated"])
        self.assertEqual(hints["o1"]["stand_upright_angles_deg"], [90.0, 0.0, -60.0])
        # Compact near-square object (45x38mm) omits noisy across_yaw_deg
        self.assertNotIn("across_yaw_deg", hints["o2"])
        self.assertEqual(hints["o2"]["pad_z_m"], 0.778)
        self.assertFalse(hints["o2"]["elongated"])
        # Too-wide container/merged blob (116x116mm > 88.8mm) omits across_yaw_deg and pad_z_m
        self.assertEqual(hints["o3"], {"opening_check": "too_wide", "elongated": False})
        # Support facts are reported beside pad_z_m, never folded into it
        self.assertAlmostEqual(hints["o1"]["fingertip_support_clearance_m"], 0.0075, delta=0.0006)
        self.assertNotIn("pad_z_floor_m", hints["o1"])

    def test_thin_block_pad_z_is_the_centre_not_the_support_floor(self):
        """A 15 mm thick block on the table: pad_z_m stays at the centre (7.5 mm above the table), the fingertips would
        reach the support (clearance about 0), and no "safe floor" is offered beside it."""
        block = dict(centre_m=[0.0, -0.20, 0.7725], top_z_m=0.780, bottom_z_m=0.765, length_m=0.080, width_m=0.060, yaw_deg=10.0)
        hints = grasp_hints_from([block], support_z_m=0.765)
        self.assertAlmostEqual(hints["o1"]["pad_z_m"], 0.7725, delta=0.0006)
        self.assertAlmostEqual(hints["o1"]["fingertip_support_clearance_m"], 0.0, places=3)
        self.assertNotIn("pad_z_floor_m", hints["o1"])
        thin = dict(centre_m=[0.0, -0.20, 0.768], top_z_m=0.771, bottom_z_m=0.765, length_m=0.060, width_m=0.040, yaw_deg=0.0)
        hints = grasp_hints_from([thin], support_z_m=0.765)
        self.assertEqual(hints["o1"]["pad_z_m"], 0.768)
        self.assertLess(hints["o1"]["fingertip_support_clearance_m"], 0)

    def test_audit_batch_grasp_3d_flags_along_axis_and_wide_guard_and_passes_aligned(self):
        harness = SimpleNamespace(_obs3d_memo=("obs-1", {"object_table": (TABLE, None, {})}))
        # Bad grasp: yaw=25 deg (closing along the 30 deg long axis!), z=0.820 (28 mm above top_z), wide_object guard (15 mm) on 25 mm object is ok, but min_aperture_m=0.040 > 0.025
        bad_action = {
            "based_on_observation_id": "obs-1",
            "stages": [
                {
                    "id": "grasp_tube",
                    "kind": "move",
                    "arm": "right",
                    "min_aperture_m": 0.040,
                    "action": {
                        "input_grasp_pose": {
                            "position_m": [0.105, -0.198, 0.820],
                            "angles_deg": [0.0, 0.0, 25.0],
                        },
                        "gripper_events": [{"goal_normalized": 0.0}],
                    },
                }
            ],
        }
        review, targets = audit_batch_grasp_3d(harness, bad_action, "obs-1")
        self.assertIsNotNone(review)
        self.assertEqual(review["status"], "warnings_detected")
        self.assertEqual(review["blocking_risks"], [])
        self.assertEqual(review["severity"], "advisory")
        stage = review["stages"][0]
        self.assertEqual(stage["blocking_risks"], [])
        self.assertEqual(stage["matched_object_id"], "o1")
        self.assertEqual(stage["axis_alignment"], "warn_along_long_axis")
        self.assertEqual(stage["recommended_across_yaw_deg"], -60.0)
        self.assertEqual(stage["height_check"], "warn_above_object_top")
        self.assertEqual(stage["grip_guard_check"], "warn_guard_exceeds_object_width")
        self.assertEqual(
            stage["risk_flags"],
            ["warn_along_long_axis", "warn_above_object_top", "warn_guard_exceeds_object_width"],
        )
        self.assertEqual(
            review["risk_flags"],
            ["warn_along_long_axis", "warn_above_object_top", "warn_guard_exceeds_object_width"],
        )
        self.assertEqual(len(targets), 1)

        # Aligned grasp: yaw=-60 deg (perpendicular to 30 deg), z=0.780, rigid_object guard
        good_action = {
            "based_on_observation_id": "obs-1",
            "stages": [
                {
                    "id": "grasp_tube_ok",
                    "kind": "move",
                    "arm": "right",
                    "grip_guard_preset": "rigid_object",
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
        review_ok, _ = audit_batch_grasp_3d(harness, good_action, "obs-1")
        self.assertEqual(review_ok["status"], "aligned")
        self.assertEqual(review_ok["warnings"], [])
        self.assertEqual(review_ok["risk_flags"], [])

        # 当 object_table 中包含 held_fk_depth 夹持物行（含 None 几何字段）时，grasp_hints_from 与 audit_batch_grasp_3d 均应跳过该行
        held_objects = [
            dict(centre_m=[0.10, -0.20, 0.850], top_z_m=0.890, bottom_z_m=0.810, length_m=0.140, width_m=0.025,
                 yaw_deg=30.0, support="held_left", quality="held_fk_depth"),
        ]
        self.assertEqual(grasp_hints_from(held_objects, support_z_m=0.765), {})
        table_with_held = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[
                ["o0", [0.10, -0.20, 0.850], None, None, None, None, None, None, "held_left", "held_fk_depth"],
                ["o1", [0.10, -0.20, 0.780], 0.792, 0.765, 0.140, 0.025, 30.0, 88.0, "table", "ok"],
            ],
        )
        harness_held = SimpleNamespace(_obs3d_memo=("obs-1", {"object_table": (table_with_held, None, {})}))
        review_held, _ = audit_batch_grasp_3d(harness_held, good_action, "obs-1")
        self.assertEqual(review_held["status"], "aligned")
        self.assertEqual(review_held["stages"][0]["matched_object_id"], "o1")

    def test_draw_object_long_axis_overlay_renders_on_panel(self):
        camera = dict(
            width=320,
            height=240,
            K=np.array([[250.0, 0.0, 160.0], [0.0, 250.0, 120.0], [0.0, 0.0, 1.0]]),
            D=np.zeros(5),
            T=np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 1.2], [0, 0, 0, 1]], dtype=float),
        )
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        draw_object_long_axis_overlay(
            img,
            camera,
            [dict(id="o1", centre_m=[0.0, 0.0, 0.78], length_m=0.14, width_m=0.03, yaw_deg=30.0)],
            320,
            240,
        )
        self.assertGreater(int(img.sum()), 0)

    def test_compiled_eccentric_rim_grasp_and_obstacle_collisions_without_prescriptive_safe_z(self):
        def compiled_move(stage_id, arm, pos_m, angles_deg):
            own = compile_action(
                "grasp",
                dict(arm=arm, observation_id="obs-1", grasp_position_m=list(pos_m), angles_deg=list(angles_deg)),
                dict(backend_id="robodojo-sim", robot_id="arx-x5-bimanual-sim"),
                dict(frame_id="env_origin", tcp_config_id="arx-x5-link6-v1"),
            )
            return dict(id=stage_id, kind="move", arm=arm, action=own)

        def compiled_grip(stage_id, arm, goal):
            return dict(
                id=stage_id,
                kind="gripper",
                arm=arm,
                action=dict(
                    path=dict(kind="hold", waypoints=[]),
                    gripper_events=[dict(arm=arm, phase="after_motion", goal_normalized=float(goal))],
                ),
            )

        table_cup = dict(
            columns=COLUMNS,
            rows=[
                ["o2", [0.18, -0.15, 0.815], 0.865, 0.765, 0.080, 0.080, 0.0, 0.0, "table", "ok"],
                ["o5", [0.18, 0.02, 0.880], 0.995, 0.765, 0.050, 0.050, 0.0, 0.0, "table", "ok"],
            ],
        )
        cup_stages = [
            compiled_move("approach", "right", [0.226, -0.150, 0.930], [0.0, 0.0, 0.0]),
            compiled_move("grasp/motion", "right", [0.226, -0.150, 0.852], [0.0, 0.0, 0.0]),
            compiled_grip("grasp", "right", 0.0),
            compiled_move("lift", "right", [0.226, -0.150, 0.960], [0.0, 0.0, 0.0]),
            compiled_move("carry", "right", [0.300, -0.150, 0.960], [0.0, 0.0, 0.0]),
        ]
        review_cup, _ = audit_batch_grasp_3d(
            SimpleNamespace(_obs3d_memo=("obs-1", {"object_table": (table_cup, None, {})})),
            dict(based_on_observation_id="obs-1", stages=cup_stages),
            "obs-1",
        )
        self.assertIsNotNone(review_cup)
        note_codes = [n["code"] for n in review_cup["advisory_notes"]]
        self.assertNotIn("warn_lateral_eccentricity", ADVISORY_NOTE_CODES)
        self.assertNotIn("warn_above_object_top", note_codes)
        self.assertNotIn("warn_transport_collision", note_codes)
        self.assertGreater(max(review_cup["stages"][0]["lateral_eccentricity_m"], math.hypot(0.226 - 0.18, 0.0)), 0.044)

        table_hit = dict(
            columns=COLUMNS,
            rows=[
                ["o2", [0.18, -0.15, 0.815], 0.865, 0.765, 0.10, 0.10, 0.0, 0.0, "table", "ok"],
                ["o5", [0.18, 0.00, 0.880], 0.995, 0.765, 0.08, 0.08, 0.0, 0.0, "table", "ok"],
            ],
        )
        hit_stages = [
            compiled_move("grasp/motion", "right", [0.18, -0.195, 0.850], [0.0, 0.0, 0.0]),
            compiled_grip("grasp", "right", 0.0),
            compiled_move("hit_post", "right", [0.18, 0.08, 0.860], [0.0, 0.0, 0.0]),
        ]
        review_hit, _ = audit_batch_grasp_3d(
            SimpleNamespace(_obs3d_memo=("obs-1", {"object_table": (table_hit, None, {})})),
            dict(based_on_observation_id="obs-1", stages=hit_stages),
            "obs-1",
        )
        self.assertIsNotNone(review_hit)
        coll_notes = [n for n in review_hit["advisory_notes"] if n["code"] == "warn_transport_collision"]
        self.assertTrue(any(n.get("stage_id") == "hit_post" and "o5" in n.get("colliding_objects", []) for n in coll_notes))
        for n in coll_notes:
            self.assertNotIn("o2", n.get("colliding_objects", []))
            self.assertNotIn("recommended_safe_z_m", n)
        self.assertNotIn("suggested_fixes", review_hit["virtual_field_simulation"])

    def test_warn_finger_collides_neighbor_is_objective_without_optimal_grasp_pose(self):
        """验证 warn_finger_collides_neighbor 与 stage_audits 仅客观回报碰撞邻物与净空，绝不合成 optimal_grasp_pose 且不依赖 pose_skill。"""
        import ast
        from pathlib import Path
        from ni_rua.runtime.actions import grasp_preview_3d

        src = Path(grasp_preview_3d.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        imported_mods = [
            node.module for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        ]
        self.assertFalse(any("pose_skill" in mod for mod in imported_mods), imported_mods)

        rad = math.radians(-60.0)
        obs_pos = [0.10 + 0.046 * float(math.cos(rad)), -0.20 + 0.046 * float(math.sin(rad)), 0.785]
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
        self.assertIsNotNone(review)
        self.assertIn("warn_finger_collides_neighbor", review["risk_flags"])
        stage_audit = review["stages"][0]
        self.assertEqual(stage_audit["finger_collision_check"], "warn_finger_collides_neighbor")
        self.assertIn("o2", stage_audit["colliding_neighbor_ids"])
        self.assertNotIn("optimal_grasp_pose", stage_audit)
        finger_notes = [n for n in review["advisory_notes"] if n["code"] == "warn_finger_collides_neighbor"]
        self.assertEqual(len(finger_notes), 1)
        advisory = finger_notes[0]
        self.assertNotIn("optimal_grasp_pose", advisory)
        self.assertNotIn("candidate pose reference", advisory["message"])
        self.assertIn("o2", advisory["colliding_neighbor_ids"])
        self.assertLess(advisory["neighbor_clearance_m"], -0.005)

    def test_non_x5_grasp_angles_profile_forwards_jaw_geometry_into_audit_field_and_waypoints(self):
        """回归验证 #152 §2.10: grasp_preview_3d 向 PointCloudSpatialField、evaluate_grasp 与法兰航点透传非 X5 jaw_geometry。"""
        from dataclasses import replace
        from ni_rua.core.embodiment import FrozenDict, RobotDescription
        from ni_rua.perception.external.pose_math import angles_deg_to_quat_wxyz, quat_wxyz_to_matrix
        from ni_rua.robot.model.embodiments import arx_x5_profile

        base = arx_x5_profile()
        narrow_gripper = replace(
            base.gripper,
            full_open_aperture_m=0.050,
            grasp_reference_offset_m=0.105,
            fingertip_beyond_reference_m=0.012,
        )
        narrow_profile = replace(
            base,
            robot=RobotDescription(robot_model_id="custom-6r", frame_id="custom_base", tcp_config_id="custom-tcp-v1"),
            arms=("main",),
            max_simultaneous_arms=1,
            arm_joint_names=FrozenDict({"main": tuple(f"j{i}" for i in range(1, 7))}),
            jaw_joint_names=FrozenDict({"main": ("jaw",)}),
            gripper=narrow_gripper,
            pose_inputs=frozenset({"grasp_angles", "tcp_quaternion"}),
            supports=frozenset({"stall_guard"}),
            cameras=("cam_head", "cam_main_wrist"),
            overview_camera="cam_head",
            wrist_cameras=FrozenDict({"main": "cam_main_wrist"}),
            camera_examples=("cam_head", "cam_main_wrist"),
            handoff=False,
            home_check=None,
            policy_step=False,
        )

        # o1 宽 24mm，邻物 o2 位于 +X 方向 46mm：对 X5 (88mm 张角，指中心 x=+48mm) 必碰撞，
        # 但对 narrow_profile (50mm 张角，指外侧仅到 x=25+8=33mm，o2 左边缘在 46-12=34mm) 完全无碰撞！
        table = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[
                ["o1", [0.10, -0.20, 0.780], 0.795, 0.765, 0.024, 0.024, 90.0, 89.0, "table", "ok"],
                ["o2", [0.146, -0.20, 0.785], 0.810, 0.765, 0.024, 0.024, 0.0, None, "table", "ok"],
            ],
        )
        q_wxyz = angles_deg_to_quat_wxyz(0.0, 0.0, 0.0)
        r_mat = quat_wxyz_to_matrix(q_wxyz)
        flange_pos = (np.array([0.10, -0.20, 0.780]) - 0.105 * r_mat[:, 0]).tolist()
        quat_xyzw = [float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3]), float(q_wxyz[0])]
        action = {
            "based_on_observation_id": "obs-narrow",
            "stages": [
                {
                    "id": "grasp_main",
                    "kind": "move",
                    "arm": "main",
                    "action": {
                        "path": {"waypoints": [{"position_m": flange_pos, "quaternion_xyzw": quat_xyzw}]},
                        "gripper_events": [{"goal_normalized": 0.0}],
                    },
                }
            ],
        }
        harness = SimpleNamespace(
            embodiment=narrow_profile,
            _obs3d_memo=("obs-narrow", {"object_table": (table, None, {})}),
        )
        review, _ = audit_batch_grasp_3d(harness, action, "obs-narrow")
        self.assertIsNotNone(review)
        stage_audit = review["stages"][0]
        self.assertEqual(stage_audit["matched_object_id"], "o1")
        self.assertAlmostEqual(stage_audit["planned_z_m"], 0.780, places=4)
        self.assertEqual(stage_audit["finger_collision_check"], "ok")
        self.assertEqual(stage_audit["colliding_neighbor_ids"], [])



if __name__ == "__main__":
    unittest.main()
