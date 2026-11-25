"""SO-101 gripper geometry derived from the packaged URDF and STL meshes."""

NI_RUA_TEST_TIER = "fast"

# === tests/test_so101_gripper_geometry.py ===

import json
import math
import unittest
from support.paths import CONFIGS_ROOT, REPO_ROOT

import numpy as np

from ni_rua.robot.model import so101_gripper_geometry as gg
from ni_rua.robot.model import so101_kinematics as kin

LO, HI = kin.joint_limits_rad(kin.load_model())[kin.GRIPPER_JOINT]


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


class SO101GripperGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.facts = gg.derive_gripper_geometry()

    def test_facts_are_finite_json_and_in_tcp_frame(self):
        self.assertTrue(finite(self.facts))
        json.dumps(self.facts, allow_nan=False)
        self.assertEqual(self.facts["frame"], kin.TCP_LINK)
        self.assertEqual(self.facts["approach_axis_tcp"], [0.0, 0.0, 1.0])

    def test_fixed_jaw_on_plus_x_and_moving_jaw_opens_along_minus_x(self):
        fixed = self.facts["fixed_jaw"]
        moving = self.facts["moving_jaw"]
        self.assertEqual(fixed["link"], kin.WRIST_CAMERA_PARENT_LINK)
        self.assertEqual(moving["link"], "moving_jaw_so101_v1_link")
        self.assertAlmostEqual(abs(fixed["fingertip_inner_point_m"][0]), 0.0, delta=0.002)
        self.assertLess(moving["fingertip_inner_point_at_zero_m"][0], fixed["fingertip_inner_point_m"][0])
        self.assertLess(moving["opening_direction_tcp"][0], -0.9)
        np.testing.assert_allclose(np.abs(moving["hinge_axis_tcp"]), [0, 1, 0], atol=1e-3)
        self.assertTrue(0.04 < fixed["finger_length_m"] < 0.09)

    def test_opening_limits_and_monotonic(self):
        self.assertLessEqual(gg.opening_width_m(LO), 0.005)
        self.assertGreaterEqual(gg.opening_width_m(HI), 0.03)
        widths = [gg.opening_width_m(q) for q in np.linspace(LO, HI, 200)]
        self.assertTrue(np.all(np.diff(widths) > 0))
        table = [row["opening_m"] for row in self.facts["opening_table"]]
        self.assertEqual(table, sorted(table))
        with self.assertRaises(ValueError):
            gg.opening_width_m(float("nan"))

    def test_goal_mapping_uses_deployment_jaw_angles(self):
        closed, opened = -0.1, 1.2
        self.assertAlmostEqual(gg.opening_width_for_goal_m(0, closed, opened), gg.opening_width_m(closed))
        self.assertAlmostEqual(gg.opening_width_for_goal_m(1, closed, opened), gg.opening_width_m(opened))
        goal = gg.goal_normalized_for_opening(0.03, closed, opened)
        self.assertAlmostEqual(gg.opening_width_for_goal_m(goal, closed, opened), 0.03, places=5)
        self.assertIsNone(gg.goal_normalized_for_opening(0.5, closed, opened))
        self.assertIsNone(gg.goal_normalized_for_opening(0.13, closed, opened))
        self.assertAlmostEqual(gg.opening_width_m(gg.jaw_rad_for_opening(0.05)), 0.05, places=5)

    def test_protrusion_and_bounding_boxes(self):
        protrusion = self.facts["protrusion_beyond_tcp_m"]
        self.assertTrue(0.003 <= protrusion["max"] <= 0.015)
        self.assertAlmostEqual(protrusion["moving_jaw_at_zero"], 0.0073, delta=0.001)
        boxes = self.facts["bbox_tcp_m"]
        self.assertAlmostEqual(boxes["overall"]["max_m"][2], protrusion["max"], places=4)
        camera = boxes["wrist_camera"]
        self.assertGreater(camera["min_m"][1], 0.01)  # beside the jaws on +Y
        self.assertLess(camera["max_m"][2], 0.0)  # behind the TCP
        for box in boxes.values():
            self.assertTrue(all(a <= b for a, b in zip(box["min_m"], box["max_m"])))

    def test_centered_grasp_offset_grows_toward_fixed_jaw(self):
        self.assertAlmostEqual(gg.centered_grasp_tcp_offset_m(0.0), 0.0, delta=0.002)
        offsets = [gg.centered_grasp_tcp_offset_m(w) for w in (0.0, 0.01, 0.03, 0.06)]
        self.assertTrue(np.all(np.diff(offsets) > 0))
        self.assertAlmostEqual(offsets[2] - offsets[0], 0.015, places=6)
        with self.assertRaises(ValueError):
            gg.centered_grasp_tcp_offset_m(-0.01)

    def test_payload_and_text_are_compact_and_model_readable(self):
        payload = gg.gripper_geometry_payload(-0.174533, 1.74533)
        json.dumps(payload, allow_nan=False)
        self.assertNotIn("source", payload)
        self.assertEqual([row["goal_normalized"] for row in payload["goal_to_opening"]], list(gg.GOAL_TABLE))
        self.assertEqual(set(payload["goal_to_opening"][0]),
                         {"goal_normalized", "jaw_rad", "opening_m", "moving_tip_x_m", "moving_tip_z_m"})
        text = gg.gripper_geometry_text(-0.174533, 1.74533)
        self.assertTrue(text.startswith("gripper_geometry"))
        self.assertIn("Fixed jaw (+X, rigid)", text)
        self.assertNotIn("between the jaw tips", text)
        self.assertNotRegex(text, r"[-+] [-+]0\.0000")
        self.assertIn("Wrist camera", text)
        self.assertNotIn("sha", text.lower())
        self.assertLess(len(text), 3200)

# === tests/test_so101_ik_plan.py ===

import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from ni_rua.robot.backends.so101 import ik_plan
from ni_rua.robot.backends.so101.ik_plan import (
    POSITION_PRIORITY_TILT_SCALE, SOLVER_CONVERGENCE, arrival_verdict, axis_weights, curobo_solve, local_solve,
    matrix_to_quaternion, orientation_components, orientation_within, pose_matrix, pose_residual, quaternion_to_matrix,
    ruckig_joint_interpolation, split_tolerances)
from ni_rua.robot.model import so101_kinematics as kin

MODEL = kin.load_model()
IK_PLAN_LIMITS = {name: (lo + 0.05, hi - 0.05) for name, (lo, hi) in list(kin.joint_limits_rad(MODEL).items())[:5]}
SEED = [0.0, -0.5, 0.8, 0.4, 0.0]
Q_TRUE = [0.3, -0.4, 0.9, 0.5, 0.2]


def rot(axis, angle):
    vec = np.asarray(axis, dtype=float)
    return Rotation.from_rotvec((vec / np.linalg.norm(vec)) * float(angle)).as_matrix()


DOWN = np.diag([1.0, -1.0, -1.0])  # TCP +Z pointing down


def pose_of(q):
    T = np.asarray(kin.fk(MODEL, q))
    return dict(position_m=T[:3, 3].tolist(), quaternion_xyzw=matrix_to_quaternion(T[:3, :3]))


def candidate(q, success=True, feasible=None, **reported):
    out = dict(q_rad=list(q), success=success, position_error_m=reported.get("position_error_m", 0.0),
               rotation_error_rad=reported.get("rotation_error_rad", 0.0))
    if feasible is not None:
        out["feasible"] = feasible
    return out


def shifted(delta):
    """Q_TRUE with the base joint moved by ``delta`` rad (TCP ~0.3 m from the axis: ~0.3*delta m position error)."""
    q = list(Q_TRUE)
    q[0] += delta
    return q


class FakeClient:
    """Answers each ``solve`` from a script: ``answers`` is a callable (poses, weights) -> candidate lists."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def solve(self, poses, seed, weights, pos_tol, ori_tol):
        self.calls.append(dict(poses=poses, seed=seed, weights=weights, pos_tol=pos_tol, ori_tol=ori_tol))
        return self.answers(poses, weights)


def _ik_plan_solve(client, requests, seed=SEED):
    return curobo_solve(client, kin, MODEL, IK_PLAN_LIMITS, requests, seed)


class GeometryTests(unittest.TestCase):
    def test_quaternion_round_trip(self):
        R = rot([0.3, -0.5, 0.8] / np.linalg.norm([0.3, -0.5, 0.8]), 1.1)
        np.testing.assert_allclose(quaternion_to_matrix(matrix_to_quaternion(R)), R, atol=1e-9)
        # 180 degree rotation exercises the non-positive-trace branch
        for R in (DOWN, np.diag([-1.0, 1.0, -1.0]), np.diag([-1.0, -1.0, 1.0])):
            np.testing.assert_allclose(quaternion_to_matrix(matrix_to_quaternion(R)), R, atol=1e-9)

    def test_pose_matrix(self):
        T = pose_matrix(dict(position_m=[1, 2, 3], quaternion_xyzw=[0, 0, 0, 2]))
        np.testing.assert_allclose(T, np.array([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]))

    def test_tilt_of_30_deg_about_base_y(self):
        approach, roll = orientation_components(rot([0, 1, 0], math.radians(30)) @ DOWN, DOWN)
        self.assertAlmostEqual(approach, 0.5236, places=4)
        self.assertAlmostEqual(roll, 0.0, places=6)

    def test_pure_rotation_about_tcp_z(self):
        for angle in (0.4, -0.4, 2.5):
            approach, roll = orientation_components(DOWN @ rot([0, 0, 1], angle), DOWN)
            self.assertAlmostEqual(approach, 0.0, places=6)
            self.assertAlmostEqual(roll, abs(angle), places=6)

    def test_combined_tilt_and_roll(self):
        achieved = rot([0, 1, 0], math.radians(30)) @ DOWN @ rot([0, 0, 1], 0.7)
        approach, roll = orientation_components(achieved, DOWN)
        self.assertAlmostEqual(approach, math.radians(30), places=6)
        self.assertAlmostEqual(roll, 0.7, places=6)

    def test_anti_parallel_axes(self):
        approach, roll = orientation_components(np.eye(3), DOWN)  # achieved +Z up, target +Z down
        self.assertAlmostEqual(approach, math.pi, places=6)
        self.assertTrue(0.0 <= roll <= math.pi)
        self.assertTrue(math.isfinite(roll))

    def test_symmetry_and_ranges(self):
        rng = np.random.default_rng(3)
        for _ in range(40):
            a = rot(*(lambda v: (v / np.linalg.norm(v), rng.uniform(0, math.pi)))(rng.normal(size=3)))
            b = rot(*(lambda v: (v / np.linalg.norm(v), rng.uniform(0, math.pi)))(rng.normal(size=3)))
            approach, roll = orientation_components(a, b)
            self.assertTrue(0.0 <= approach <= math.pi and 0.0 <= roll <= math.pi + 1e-9)
            # the approach angle is symmetric in the two frames
            self.assertAlmostEqual(approach, orientation_components(b, a)[0], places=6)

    def test_residual_split_adds_components(self):
        target = pose_matrix(dict(position_m=[0.2, 0.0, 0.1], quaternion_xyzw=matrix_to_quaternion(DOWN)))
        measured = target.copy()
        measured[:3, :3] = rot([0, 1, 0], 0.2) @ DOWN @ rot([0, 0, 1], 0.1)
        measured[:3, 3] += [0.001, 0, -0.002]
        plain = pose_residual(measured, target)
        self.assertNotIn("approach_error_rad", plain)
        split = pose_residual(measured, target, split=True)
        self.assertAlmostEqual(split["approach_error_rad"], 0.2, places=4)
        self.assertAlmostEqual(split["roll_error_rad"], 0.1, places=4)
        for key, value in plain.items():
            self.assertEqual(split[key], value)
        self.assertAlmostEqual(plain["dz_m"], -0.002)


class VerdictTests(unittest.TestCase):
    def residual(self, **kw):
        base = dict(dx_m=0.0, dy_m=0.0, dz_m=0.0, position_error_m=0.0, orientation_error_rad=0.0,
                    approach_error_rad=0.0, roll_error_rad=0.0)
        base.update(kw)
        return base

    def test_split_approach_ok_roll_outside_fails(self):
        split = dict(approach_tolerance_rad=0.1, roll_tolerance_rad=0.1)
        residual = self.residual(approach_error_rad=0.05, roll_error_rad=0.5, orientation_error_rad=0.5)
        self.assertEqual(arrival_verdict(residual, 0.01, 0.15, split=split), (False, False))
        residual["roll_error_rad"] = 0.05
        self.assertEqual(arrival_verdict(residual, 0.01, 0.15, split=split), (True, False))

    def test_free_component_ignored(self):
        split = dict(approach_tolerance_rad=0.1, roll_tolerance_rad=math.pi)
        residual = self.residual(approach_error_rad=0.05, roll_error_rad=3.0)
        self.assertTrue(arrival_verdict(residual, 0.01, 0.15, split=split)[0])
        split = dict(approach_tolerance_rad=math.pi, roll_tolerance_rad=0.1)
        self.assertTrue(arrival_verdict(self.residual(approach_error_rad=3.0, roll_error_rad=0.05), 0.01, 0.15,
                                        split=split)[0])

    def test_split_ignores_total_angle(self):
        split = dict(approach_tolerance_rad=0.1, roll_tolerance_rad=0.1)
        residual = self.residual(approach_error_rad=0.05, roll_error_rad=0.05, orientation_error_rad=2.0)
        self.assertTrue(arrival_verdict(residual, 0.01, 0.15, split=split)[0])

    def test_vertical_window_with_split(self):
        split = dict(approach_tolerance_rad=0.1, roll_tolerance_rad=0.1)
        residual = self.residual(dz_m=0.04, position_error_m=0.04)
        self.assertEqual(arrival_verdict(residual, 0.01, 0.15, 0.05, split=split), (True, True))
        self.assertEqual(arrival_verdict(residual, 0.01, 0.15, 0.0, split=split), (False, False))
        residual["roll_error_rad"] = 0.5
        self.assertEqual(arrival_verdict(residual, 0.01, 0.15, 0.05, split=split), (False, False))

    def test_legacy_positional_signature(self):
        residual = self.residual(position_error_m=0.005, orientation_error_rad=0.1)
        self.assertEqual(arrival_verdict(residual, 0.01, 0.15), (True, False))
        self.assertEqual(arrival_verdict(residual, 0.01, 0.05), (False, False))
        self.assertEqual(arrival_verdict(residual, 0.01, math.pi), (True, False))
        self.assertTrue(orientation_within(residual, 0.15))
        self.assertFalse(orientation_within(residual, 0.05))

    def test_split_tolerances(self):
        self.assertIsNone(split_tolerances(dict(orientation_tolerance_rad=0.15)))
        self.assertEqual(split_tolerances(dict(orientation_tolerance_rad=0.15, approach_tolerance_rad=0.05)),
                         dict(approach_tolerance_rad=0.05, roll_tolerance_rad=0.15))
        self.assertEqual(split_tolerances(dict(orientation_tolerance_rad=0.15, roll_tolerance_rad=0.3)),
                         dict(approach_tolerance_rad=0.15, roll_tolerance_rad=0.3))
        self.assertEqual(split_tolerances(dict(orientation_tolerance_rad=0.15, approach_tolerance_rad=0.1,
                                               roll_tolerance_rad=3.2)),
                         dict(approach_tolerance_rad=0.1, roll_tolerance_rad=3.2))


class AxisWeightTests(unittest.TestCase):
    def test_single_angle_and_free(self):
        self.assertEqual(axis_weights(0.15), [1.0] * 6)
        self.assertEqual(axis_weights(math.pi), [1, 1, 1, 0, 0, 0])
        self.assertEqual(axis_weights(3.14159), [1, 1, 1, 0, 0, 0])

    def test_split_normalised_by_inverse_tolerance(self):
        weights = axis_weights(0.15, dict(approach_tolerance_rad=0.1, roll_tolerance_rad=0.4))
        self.assertEqual(weights[:3], [1.0] * 3)
        self.assertAlmostEqual(weights[3], 1.0)
        self.assertAlmostEqual(weights[4], 1.0)
        self.assertAlmostEqual(weights[5], 0.25)
        weights = axis_weights(0.15, dict(approach_tolerance_rad=0.4, roll_tolerance_rad=0.1))
        self.assertAlmostEqual(weights[3], 0.25)
        self.assertAlmostEqual(weights[5], 1.0)
        self.assertLessEqual(max(weights), 1.0)

    def test_free_roll_zeroes_rot_z(self):
        weights = axis_weights(0.15, dict(approach_tolerance_rad=0.1, roll_tolerance_rad=math.pi))
        self.assertEqual(weights, [1, 1, 1, 1.0, 1.0, 0.0])
        weights = axis_weights(0.15, dict(approach_tolerance_rad=math.pi, roll_tolerance_rad=0.1))
        self.assertEqual(weights, [1, 1, 1, 0.0, 0.0, 1.0])
        self.assertEqual(axis_weights(0.15, dict(approach_tolerance_rad=math.pi, roll_tolerance_rad=math.pi)),
                         [1, 1, 1, 0, 0, 0])

    def test_position_priority_scales_tilt_only_and_keeps_roll(self):
        weights = axis_weights(0.15, position_priority=True)
        self.assertEqual(weights[:3], [1.0] * 3)
        self.assertEqual(POSITION_PRIORITY_TILT_SCALE, 0.1)
        self.assertAlmostEqual(weights[3], POSITION_PRIORITY_TILT_SCALE)
        self.assertAlmostEqual(weights[4], POSITION_PRIORITY_TILT_SCALE)
        self.assertEqual(weights[5], 1.0)
        split = axis_weights(0.15, dict(approach_tolerance_rad=0.1, roll_tolerance_rad=0.4), position_priority=True)
        self.assertAlmostEqual(split[3], POSITION_PRIORITY_TILT_SCALE)
        self.assertAlmostEqual(split[5], 0.25)  # roll weight untouched
        free_roll = axis_weights(0.15, dict(approach_tolerance_rad=0.1, roll_tolerance_rad=math.pi),
                                 position_priority=True)
        self.assertEqual(free_roll[5], 0.0)


class CuroboSolveTests(unittest.TestCase):
    TOL = (0.005, 0.1)

    def setUp(self):
        self.pose = pose_of(Q_TRUE)

    def test_picks_least_travel_valid_candidate(self):
        # two precise joint sets (orientation free, so no retry); the seed is nearer to `nearer` by largest joint travel
        nearer, farther = shifted(0.0005), shifted(-0.0005)
        seed = shifted(0.05)  # nearer travels 0.0495, farther 0.0505
        for order in ([nearer, farther], [farther, nearer]):
            client = FakeClient(lambda poses, w: [[candidate(c) for c in order]])
            (q, record), = _ik_plan_solve(client, [(self.pose, (0.02, math.pi), None)], seed)
            self.assertTrue(record["ok"])
            self.assertEqual(q, nearer)

    def test_rejects_service_failure_even_if_fk_matches(self):
        client = FakeClient(lambda poses, w: [[candidate(Q_TRUE, success=False)]])
        (q, record), = _ik_plan_solve(client, [(self.pose, self.TOL, None)])
        self.assertFalse(record["ok"])
        self.assertLessEqual(record["position_error_m"], 1e-4)  # FK matches, yet refused
        self.assertEqual(record["in_collision_candidates"], 1)

    def test_rejects_out_of_limit_and_non_finite(self):
        lo, hi = list(IK_PLAN_LIMITS.values())[1]
        out = list(Q_TRUE)
        out[1] = hi + 0.01
        bad = list(Q_TRUE)
        bad[2] = float("nan")
        short = Q_TRUE[:4]
        client = FakeClient(lambda poses, w: [[candidate(out), candidate(bad), candidate(short)]])
        (q, record), = _ik_plan_solve(client, [(self.pose, (0.5, math.pi), None)])
        self.assertFalse(record["ok"])

    def test_nonfinite_only_falls_back_to_seed_residual(self):
        bad = list(Q_TRUE)
        bad[0] = float("inf")
        client = FakeClient(lambda poses, w: [[candidate(bad)]])
        (q, record), = _ik_plan_solve(client, [(self.pose, (0.02, math.pi), None)])
        self.assertEqual(q, SEED)
        self.assertFalse(record["ok"])
        self.assertTrue(math.isfinite(record["position_error_m"]))

    def test_record_names_the_active_client_solver(self):
        client = FakeClient(lambda poses, w: [[candidate(Q_TRUE)]])
        client.solver = "pink"
        (q, record), = _ik_plan_solve(client, [(self.pose, self.TOL, None)])
        self.assertTrue(record["ok"])
        self.assertEqual(record["solver"], "pink")

    def test_residuals_come_from_local_fk(self):
        client = FakeClient(lambda poses, w: [[candidate(Q_TRUE, position_error_m=0.9, rotation_error_rad=2.0)]])
        (q, record), = _ik_plan_solve(client, [(self.pose, self.TOL, None)])
        self.assertTrue(record["ok"])
        self.assertEqual(record["solver"], "curobo")
        self.assertLess(record["position_error_m"], 1e-4)
        self.assertLess(record["orientation_error_rad"], 1e-3)
        # and a candidate that the fake claims is perfect but FK says is far off is rejected
        wrong = [q + 0.3 for q in Q_TRUE]
        client = FakeClient(lambda poses, w: [[candidate(wrong, position_error_m=0.0, rotation_error_rad=0.0)]])
        (q, record), = _ik_plan_solve(client, [(self.pose, self.TOL, None)])
        self.assertFalse(record["ok"])
        self.assertGreater(record["position_error_m"], 0.005)

    def test_position_priority_second_request_used_when_valid(self):
        def answers(poses, weights):
            if weights[3] == 1.0:
                return [[candidate(Q_TRUE, success=False)]]
            return [[candidate(Q_TRUE)]]
        client = FakeClient(answers)
        (q, record), = _ik_plan_solve(client, [(self.pose, self.TOL, None)])
        self.assertTrue(record["ok"])
        self.assertEqual(record["solve"], "position_priority")
        self.assertEqual(len(client.calls), 2)
        self.assertAlmostEqual(client.calls[1]["weights"][3], POSITION_PRIORITY_TILT_SCALE)
        self.assertEqual(client.calls[1]["weights"][5], 1.0)

    def test_no_second_request_when_first_valid_or_orientation_free(self):
        client = FakeClient(lambda poses, w: [[candidate(Q_TRUE)]])
        (_, record), = _ik_plan_solve(client, [(self.pose, self.TOL, None)])
        self.assertNotIn("solve", record)
        self.assertEqual(len(client.calls), 1)
        client = FakeClient(lambda poses, w: [[candidate(Q_TRUE, success=False)]])
        _ik_plan_solve(client, [(self.pose, (0.005, math.pi), None)])
        self.assertEqual(len(client.calls), 1)

    def test_failure_record_carries_position_priority_hint(self):
        # position reached, orientation off by a roll: the second request reaches the position with a wrong
        # orientation, which becomes the hint (quaternion of the reached pose).
        q_off = list(Q_TRUE)
        q_off[4] += 0.8  # wrist roll: orientation, and the TCP sits ~6 mm off the roll axis
        client = FakeClient(lambda poses, w: [[candidate(q_off)]])
        (q, record), = _ik_plan_solve(client, [(self.pose, (0.01, 0.1), None)])
        self.assertFalse(record["ok"])
        hint = record["position_priority"]
        np.testing.assert_allclose(hint["quaternion_xyzw"], matrix_to_quaternion(np.asarray(kin.fk(MODEL, q_off))[:3, :3]),
                                   atol=1e-5)
        self.assertLessEqual(hint["position_error_m"], 0.01)
        self.assertGreater(hint["orientation_error_rad"], 0.1)
        self.assertGreater(record["orientation_error_rad"], 0.1)

    def test_no_hint_when_position_not_reached(self):
        wrong = [q + 0.3 for q in Q_TRUE]
        client = FakeClient(lambda poses, w: [[candidate(wrong)]])
        (_, record), = _ik_plan_solve(client, [(self.pose, self.TOL, None)])
        self.assertNotIn("position_priority", record)

    def test_in_collision_candidates_counts_fk_valid_infeasible(self):
        # counted from local FK, not from the service's reported errors: within tolerances but feasible False
        far = [q + 0.3 for q in Q_TRUE]
        candidates = [candidate(Q_TRUE, feasible=False, position_error_m=0.9, rotation_error_rad=2.0),
                      candidate(shifted(0.001), feasible=False),
                      candidate(far, feasible=False, position_error_m=0.0, rotation_error_rad=0.0),
                      candidate(Q_TRUE, success=False)]  # no feasible key: success=False is infeasible too
        client = FakeClient(lambda poses, w: [candidates])
        (_, record), = _ik_plan_solve(client, [(self.pose, self.TOL, None)])
        self.assertFalse(record["ok"])
        self.assertEqual(record["in_collision_candidates"], 3)

    def test_in_collision_ignores_orientation_misses(self):
        rolled = list(Q_TRUE)
        rolled[4] += 0.8
        client = FakeClient(lambda poses, w: [[candidate(rolled, feasible=False)]])
        (_, record), = _ik_plan_solve(client, [(self.pose, (0.01, 0.1), None)])
        self.assertEqual(record["in_collision_candidates"], 0)

    def test_split_request_records_components_and_service_bound(self):
        split = dict(approach_tolerance_rad=0.1, roll_tolerance_rad=math.pi)
        rolled = list(Q_TRUE)
        rolled[4] += 0.8
        client = FakeClient(lambda poses, w: [[candidate(rolled)]])
        (q, record), = _ik_plan_solve(client, [(self.pose, (0.01, 0.15), split)])
        self.assertTrue(record["ok"])  # a free roll is not checked
        self.assertGreater(record["roll_error_rad"], 0.5)
        self.assertLess(record["approach_error_rad"], 0.05)
        self.assertEqual(client.calls[0]["weights"], axis_weights(0.15, split))
        self.assertEqual((client.calls[0]["pos_tol"], client.calls[0]["ori_tol"]), SOLVER_CONVERGENCE)
        self.assertTrue(record["orientation_checked"])
        # tight roll bound fails the same candidate
        tight = dict(approach_tolerance_rad=0.1, roll_tolerance_rad=0.2)
        client = FakeClient(lambda poses, w: [[candidate(rolled)]])
        (_, record), = _ik_plan_solve(client, [(self.pose, (0.01, 0.15), tight)])
        self.assertFalse(record["ok"])
        self.assertIn("approach_error_rad", record)
        self.assertGreater(record["roll_error_rad"], 0.2)

    def test_all_free_split_not_orientation_checked_and_no_retry(self):
        split = dict(approach_tolerance_rad=math.pi, roll_tolerance_rad=math.pi)
        client = FakeClient(lambda poses, w: [[candidate(Q_TRUE, success=False)]])
        (_, record), = _ik_plan_solve(client, [(self.pose, (0.005, 0.15), split)])
        self.assertFalse(record["orientation_checked"])
        self.assertEqual(len(client.calls), 1)

    def test_equal_weight_requests_share_one_call(self):
        other = pose_of([0.1, -0.3, 0.7, 0.3, 0.0])
        qs = [Q_TRUE, [0.1, -0.3, 0.7, 0.3, 0.0]]
        client = FakeClient(lambda poses, w: [[candidate(qs[0])], [candidate(qs[1])]])
        results = _ik_plan_solve(client, [(self.pose, self.TOL, None), (other, self.TOL, None)])
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(client.calls[0]["poses"]), 2)
        self.assertEqual([r[0] for r in results], qs)
        self.assertTrue(all(r[1]["ok"] for r in results))

    def test_different_weights_split_into_calls_and_keep_order(self):
        other = pose_of([0.1, -0.3, 0.7, 0.3, 0.0])
        def answers(poses, weights):
            q = Q_TRUE if poses[0] is self.pose else [0.1, -0.3, 0.7, 0.3, 0.0]
            return [[candidate(q)] for _ in poses]
        client = FakeClient(answers)
        results = _ik_plan_solve(client, [(self.pose, self.TOL, None), (other, (0.005, math.pi), None)])
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(results[0][0], Q_TRUE)
        self.assertEqual(results[1][0], [0.1, -0.3, 0.7, 0.3, 0.0])

    def test_partial_failure_in_batch(self):
        other = pose_of([0.1, -0.3, 0.7, 0.3, 0.0])
        client = FakeClient(lambda poses, w: [[candidate(Q_TRUE)], [candidate([0.0] * 5, success=False)]]
                            if len(poses) == 2 else [[candidate([0.0] * 5, success=False)]])
        results = _ik_plan_solve(client, [(self.pose, self.TOL, None), (other, self.TOL, None)])
        self.assertTrue(results[0][1]["ok"])
        self.assertFalse(results[1][1]["ok"])

    def test_feasible_flag_not_success_decides_validity(self):
        client = FakeClient(lambda poses, w: [[candidate(Q_TRUE, success=False, feasible=True)]])
        (q, record), = _ik_plan_solve(client, [(self.pose, self.TOL, None)])
        self.assertTrue(record["ok"])
        self.assertEqual(q, Q_TRUE)
        # feasible False is rejected although FK matches and the service says success
        client = FakeClient(lambda poses, w: [[candidate(Q_TRUE, success=True, feasible=False)]])
        (_, record), = _ik_plan_solve(client, [(self.pose, self.TOL, None)])
        self.assertFalse(record["ok"])
        self.assertEqual(record["in_collision_candidates"], 1)
        # without a feasible key, success is the fallback
        client = FakeClient(lambda poses, w: [[candidate(Q_TRUE, success=True)]])
        self.assertTrue(_ik_plan_solve(client, [(self.pose, self.TOL, None)])[0][1]["ok"])

    def test_every_request_sends_solver_convergence(self):
        client = FakeClient(lambda poses, w: [[candidate(Q_TRUE, success=False)]])
        _ik_plan_solve(client, [(self.pose, (0.02, 0.3), None)])
        # first request, position-first retry, then the roll-relaxed request for the hint (nothing feasible here)
        self.assertEqual(len(client.calls), 3)
        for call in client.calls:
            self.assertEqual((call["pos_tol"], call["ori_tol"]), SOLVER_CONVERGENCE)
        tilt = ik_plan.POSITION_PRIORITY_TILT_SCALE
        self.assertEqual([round(v, 6) for v in client.calls[2]["weights"]], [1, 1, 1, tilt, tilt, tilt])

    def test_roll_blocked_failure_names_the_limit_joint_and_hints_the_reachable_orientation(self):
        # A pose whose roll is beyond wrist_roll: the full and position-first requests return a solution pressing
        # wrist_roll against its upper limit; the roll-relaxed request reaches the position with another jaw roll.
        q_press = list(Q_TRUE)
        q_press[4] = IK_PLAN_LIMITS["wrist_roll"][1] - 0.001
        relaxed_target = pose_matrix(self.pose)

        def answer(poses, w):
            if w[5] < 1.0:  # roll-relaxed request: reaches the pose (orientation within tolerance or not)
                return [[candidate(Q_TRUE, feasible=True)]]
            return [[candidate(q_press, feasible=True)]]
        bad = dict(position_m=self.pose["position_m"], quaternion_xyzw=self.flip_roll(self.pose["quaternion_xyzw"]))
        client = FakeClient(answer)
        (q, record), = _ik_plan_solve(client, [(bad, (0.002, 0.05), None)])
        self.assertFalse(record["ok"])
        self.assertEqual(record["limit_joints"], {"wrist_roll": "upper"})
        self.assertIn("position_priority", record)
        self.assertLessEqual(record["position_priority"]["position_error_m"], 0.002)
        self.assertEqual(len(client.calls), 3)

    @staticmethod
    def flip_roll(q):
        """The quaternion turned half a turn about its own +Z (the approach axis): same approach, opposite jaw."""
        from ni_rua.robot.backends.so101.ik_plan import quaternion_to_matrix
        R = quaternion_to_matrix(q) @ np.diag([-1.0, -1.0, 1.0])
        return ik_plan.matrix_to_quaternion(R)

    def test_precise_tier_preferred_over_closer_travel_imprecise(self):
        imprecise, precise = shifted(0.01), shifted(0.0)
        seed = shifted(0.01)  # the imprecise candidate has zero travel, the precise one 0.01
        client = FakeClient(lambda poses, w: [[candidate(imprecise), candidate(precise)]])
        (q, record), = _ik_plan_solve(client, [(self.pose, self.TOL, None)], seed)
        residual = pose_residual(np.asarray(kin.fk(MODEL, imprecise)), pose_matrix(self.pose))
        self.assertTrue(SOLVER_CONVERGENCE[0] < residual["position_error_m"] <= self.TOL[0])
        self.assertEqual(q, precise)
        self.assertLessEqual(record["position_error_m"], SOLVER_CONVERGENCE[0])
        self.assertNotIn("solve", record)
        self.assertEqual(len(client.calls), 1)

    def test_imprecise_first_precise_retry_is_used(self):
        imprecise = shifted(0.01)
        client = FakeClient(lambda poses, w: [[candidate(imprecise)]] if w[3] == 1.0 else [[candidate(Q_TRUE)]])
        (q, record), = _ik_plan_solve(client, [(self.pose, self.TOL, None)])
        self.assertEqual(len(client.calls), 2)
        self.assertTrue(record["ok"])
        self.assertEqual(record["solve"], "position_priority")
        self.assertEqual(q, Q_TRUE)

    def test_imprecise_first_imprecise_retry_keeps_first(self):
        first, second = shifted(0.01), shifted(-0.012)
        client = FakeClient(lambda poses, w: [[candidate(first)]] if w[3] == 1.0 else [[candidate(second)]])
        (q, record), = _ik_plan_solve(client, [(self.pose, self.TOL, None)])
        self.assertEqual(len(client.calls), 2)
        self.assertTrue(record["ok"])
        self.assertNotIn("solve", record)
        self.assertEqual(q, first)

    def test_imprecise_first_retry_invalid_keeps_first(self):
        first = shifted(0.01)
        client = FakeClient(lambda poses, w: [[candidate(first)]] if w[3] == 1.0 else [[candidate(Q_TRUE, feasible=False)]])
        (q, record), = _ik_plan_solve(client, [(self.pose, self.TOL, None)])
        self.assertTrue(record["ok"])
        self.assertNotIn("solve", record)
        self.assertEqual(q, first)

    def test_imprecise_with_free_orientation_is_accepted_without_retry(self):
        client = FakeClient(lambda poses, w: [[candidate(shifted(0.01))]])
        (q, record), = _ik_plan_solve(client, [(self.pose, (0.02, math.pi), None)])
        self.assertTrue(record["ok"])
        self.assertEqual(len(client.calls), 1)

    def test_imprecise_retry_in_batch_only_for_those_requests(self):
        other_q = [0.1, -0.3, 0.7, 0.3, 0.0]
        other = pose_of(other_q)

        def answers(poses, w):
            if w[3] == 1.0:  # first request: batched together, first pose imprecise, second precise
                return [[candidate(shifted(0.01))], [candidate(other_q)]]
            self.assertEqual(poses, [self.pose])
            return [[candidate(Q_TRUE)]]
        client = FakeClient(answers)
        results = _ik_plan_solve(client, [(self.pose, self.TOL, None), (other, self.TOL, None)])
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(results[0][1]["solve"], "position_priority")
        self.assertNotIn("solve", results[1][1])

    def test_service_error_propagates(self):
        from ni_rua.robot.backends.so101.ik_service import IKServiceError

        class Down:
            def solve(self, *a):
                raise IKServiceError("ik_service_unreachable")
        with self.assertRaises(IKServiceError):
            _ik_plan_solve(Down(), [(self.pose, self.TOL, None)])

# === tests/test_so101_ik_service.py ===

import io
import json
import socket
import unittest
import math
import urllib.error
from pathlib import Path

from ni_rua.robot.backends.so101 import contract
from ni_rua.robot.backends.so101.deployment import embodiment_profile, validate_deployment, validate_ik_service
from ni_rua.robot.backends.so101.ik_service import ARM_JOINTS, CuroboClient, IKServiceError

from ni_rua.robot.model import embodiments
from ni_rua.robot.motion import motion_tolerance as mt

CONFIGS = CONFIGS_ROOT
SETTINGS = dict(solver="curobo", url="http://127.0.0.1:8341/", timeout_s=7.5, num_solutions=4, table_top_z_m=-0.02)
IK_SERVICE_LIMITS = {name: (-1.0 - i, 1.0 + i) for i, name in enumerate(ARM_JOINTS)}


class Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.payload if isinstance(self.payload, bytes) else json.dumps(self.payload).encode()


def pose(x):
    return dict(position_m=[x, 0.0, 0.1], quaternion_xyzw=[0.0, 1.0, 0.0, 0.0])


def solution():
    return [dict(q_rad=[0.0] * 5, success=True, position_error_m=0.0, rotation_error_rad=0.0)]


class FakeServer:
    """Records requests; ``solve_answer`` builds the /ik answer from the request body."""

    def __init__(self, max_batch=3, health=None, solve_answer=None, error=None):
        self.requests = []
        self.max_batch = max_batch
        self.health = health
        self.solve_answer = solve_answer
        self.error = error

    def __call__(self, request, timeout=None):
        body = None if request.data is None else json.loads(request.data)
        self.requests.append(dict(method=request.get_method(), url=request.full_url, body=body, timeout=timeout,
                                  headers=dict(request.header_items())))
        if self.error is not None:
            raise self.error
        if request.full_url.endswith("/health"):
            return Response(self.health if self.health is not None else dict(status="ok", max_batch=self.max_batch))
        if self.solve_answer is not None:
            return Response(self.solve_answer(body))
        return Response(dict(status="ok", solutions=[solution() for _ in body["poses"]]))


def client(server, **overrides):
    return CuroboClient(dict(SETTINGS, **overrides), IK_SERVICE_LIMITS, opener=server)


def _ik_service_solve(c, poses):
    return c.solve(poses, [0.0, -0.5, 0.5, 0.2, 0.0], [1, 1, 1, 1, 1, 0.5], 0.005, 0.1)


def http_error(code, payload):
    return urllib.error.HTTPError("http://x/ik", code, "err", {}, io.BytesIO(payload if isinstance(payload, bytes)
                                                                              else json.dumps(payload).encode()))


class CuroboClientTests(unittest.TestCase):
    def test_health_then_solve_request_body(self):
        server = FakeServer(max_batch=8)
        out = _ik_service_solve(client(server), [pose(0.2), pose(0.3)])
        self.assertEqual(len(out), 2)
        health, ik = server.requests
        self.assertEqual((health["method"], health["url"]), ("GET", "http://127.0.0.1:8341/health"))
        self.assertEqual((ik["method"], ik["url"]), ("POST", "http://127.0.0.1:8341/ik"))
        self.assertEqual(ik["timeout"], 7.5)
        self.assertEqual(ik["headers"]["Content-type"], "application/json")
        body = ik["body"]
        self.assertEqual(body["poses"], [pose(0.2), pose(0.3)])
        self.assertEqual(body["seed_q_rad"], [0.0, -0.5, 0.5, 0.2, 0.0])
        self.assertEqual(body["axis_weights"], [1, 1, 1, 1, 1, 0.5])
        self.assertEqual(body["position_tolerance_m"], 0.005)
        self.assertEqual(body["orientation_tolerance_rad"], 0.1)
        self.assertEqual(body["num_solutions"], 4)
        self.assertEqual(body["table_top_z_m"], -0.02)
        self.assertEqual(body["joint_limits_rad"], {name: list(IK_SERVICE_LIMITS[name]) for name in ARM_JOINTS})

    def test_table_none_is_sent_as_null_and_health_asked_once(self):
        server = FakeServer()
        settings = dict(SETTINGS)
        settings.pop("table_top_z_m")
        c = CuroboClient(settings, IK_SERVICE_LIMITS, opener=server)
        _ik_service_solve(c, [pose(0.2)])
        _ik_service_solve(c, [pose(0.2)])
        self.assertIsNone(server.requests[1]["body"]["table_top_z_m"])
        self.assertEqual([r["method"] for r in server.requests], ["GET", "POST", "POST"])

    def test_chunking_by_max_batch_keeps_order(self):
        def answer(body):
            return dict(status="ok", solutions=[[dict(q_rad=[p["position_m"][0]] * 5, success=True)]
                                                 for p in body["poses"]])
        server = FakeServer(max_batch=2, solve_answer=answer)
        out = _ik_service_solve(client(server), [pose(i / 10) for i in range(1, 6)])
        posts = [r for r in server.requests if r["method"] == "POST"]
        self.assertEqual([len(r["body"]["poses"]) for r in posts], [2, 2, 1])
        self.assertEqual([item[0]["q_rad"][0] for item in out], [0.1, 0.2, 0.3, 0.4, 0.5])

    def test_http_409_joint_limits_mismatch(self):
        server = FakeServer(error=http_error(409, dict(status="error", reason="joint_limits_mismatch")))
        with self.assertRaises(IKServiceError) as caught:
            client(server).health()
        self.assertTrue(caught.exception.reason.startswith("ik_service_http_409:joint_limits_mismatch"))

    def test_http_error_with_unparseable_body(self):
        server = FakeServer(error=http_error(500, b"<html>boom"))
        with self.assertRaises(IKServiceError) as caught:
            client(server).health()
        self.assertEqual(caught.exception.reason, "ik_service_http_500:")

    def test_http_reason_is_truncated(self):
        server = FakeServer(error=http_error(400, dict(reason="x" * 500)))
        with self.assertRaises(IKServiceError) as caught:
            client(server).health()
        self.assertLessEqual(len(caught.exception.reason), 200)

    def test_unreachable(self):
        for error in (urllib.error.URLError(ConnectionRefusedError(61, "refused")), ConnectionRefusedError(),
                      socket.timeout("timed out"), TimeoutError(), OSError("no route")):
            with self.assertRaises(IKServiceError) as caught:
                _ik_service_solve(client(FakeServer(error=error)), [pose(0.2)])
            self.assertEqual(caught.exception.reason, "ik_service_unreachable", repr(error))

    def test_malformed_json(self):
        server = FakeServer(health=b"{not json")
        with self.assertRaises(IKServiceError) as caught:
            client(server).health()
        self.assertEqual(caught.exception.reason, "ik_service_bad_response")

    def test_non_ok_status_or_non_object(self):
        for payload in (dict(status="error"), [1, 2], dict(max_batch=3)):
            with self.assertRaises(IKServiceError) as caught:
                client(FakeServer(health=payload)).health()
            self.assertEqual(caught.exception.reason, "ik_service_bad_response")

    def test_bad_max_batch(self):
        for value in (0, -1, "3", None, 2.5):
            with self.assertRaises(IKServiceError) as caught:
                client(FakeServer(health=dict(status="ok", max_batch=value))).health()
            self.assertEqual(caught.exception.reason, "ik_service_bad_response")

    def test_wrong_solution_count_or_shape(self):
        bad_answers = (dict(status="ok", solutions=[solution()]),            # one for two poses
                       dict(status="ok", solutions=[solution(), []]),          # empty candidate list
                       dict(status="ok", solutions=[solution(), ["q"]]),       # entries not objects
                       dict(status="ok", solutions="nope"),
                       dict(status="ok"))
        for answer in bad_answers:
            server = FakeServer(solve_answer=lambda body, answer=answer: answer)
            with self.assertRaises(IKServiceError) as caught:
                _ik_service_solve(client(server), [pose(0.2), pose(0.3)])
            self.assertEqual(caught.exception.reason, "ik_service_bad_response", answer)

    def test_failure_in_later_chunk_raises_without_partial_result(self):
        calls = []

        def answer(body):
            calls.append(1)
            return dict(status="ok", solutions=[solution() for _ in body["poses"]]) if len(calls) == 1 else dict(
                status="ok", solutions=[])
        server = FakeServer(max_batch=1, solve_answer=answer)
        with self.assertRaises(IKServiceError):
            _ik_service_solve(client(server), [pose(0.2), pose(0.3)])

    def test_url_trailing_slash_stripped(self):
        self.assertEqual(client(FakeServer()).url, "http://127.0.0.1:8341")


# ----------------------------------------------------------------------- deployment section / profile tags ---

def document(**ik_service):
    data = json.loads((CONFIGS / "so101.example.json").read_text(encoding="utf-8"))
    if ik_service:
        data["ik_service"] = ik_service
    return data


def deployment(ik=None):
    data = document()
    if ik is not None:
        data["ik_service"] = ik
    return validate_deployment(data, base=CONFIGS)


GOOD = dict(solver="curobo", url="http://127.0.0.1:8341")


class IKServiceSectionTests(unittest.TestCase):
    def test_absent_or_null_is_none(self):
        self.assertIsNone(validate_ik_service(None))
        self.assertIsNone(deployment()["ik_service"])
        self.assertIsNone(deployment(None)["ik_service"])

    def test_defaults(self):
        self.assertEqual(validate_ik_service(GOOD), dict(GOOD, timeout_s=10.0, num_solutions=8, table_top_z_m=None,
                                                         path_check_step_rad=0.01))
        self.assertEqual(validate_ik_service(dict(GOOD, url="http://localhost:9/"))["url"], "http://localhost:9")

    def test_explicit_values_and_comment_keys(self):
        out = validate_ik_service(dict(GOOD, timeout_s=60, num_solutions=32, table_top_z_m=-0.3, _comment="x"))
        self.assertEqual((out["timeout_s"], out["num_solutions"], out["table_top_z_m"]), (60.0, 32, -0.3))

    def test_refused(self):
        for bad in (dict(GOOD, url="http://10.0.0.5:8341"), dict(GOOD, url="http://0.0.0.0:8341"),
                    dict(GOOD, url="https://127.0.0.1:8341"), dict(GOOD, url="http://127.0.0.1"),
                    dict(GOOD, url="http://127.0.0.1.evil.com:8341"), dict(GOOD, url=None), dict(GOOD, url=8341),
                    dict(GOOD, extra=1), dict(GOOD, solver="local"), dict(url=GOOD["url"]),
                    dict(GOOD, timeout_s=0), dict(GOOD, timeout_s=61), dict(GOOD, timeout_s=-1),
                    dict(GOOD, timeout_s=float("nan")), dict(GOOD, timeout_s="10"), dict(GOOD, num_solutions=0),
                    dict(GOOD, num_solutions=33), dict(GOOD, num_solutions=2.5), dict(GOOD, num_solutions=True),
                    dict(GOOD, table_top_z_m=0.31), dict(GOOD, table_top_z_m=-0.31),
                    dict(GOOD, table_top_z_m=float("inf")), "http://127.0.0.1:1", [GOOD]):
            with self.assertRaises(ValueError, msg=repr(bad)) as caught:
                validate_ik_service(bad)
            self.assertIn("so101_deployment_", str(caught.exception))

    def test_deployment_load_keeps_other_fields_identical(self):
        without, with_ik = deployment(), deployment(GOOD)
        self.assertEqual({k: v for k, v in with_ik.items() if k != "ik_service"},
                         {k: v for k, v in without.items() if k != "ik_service"})
        self.assertEqual(with_ik["ik_service"]["url"], GOOD["url"])

    def test_deployment_refuses_bad_section(self):
        with self.assertRaises(ValueError):
            deployment(dict(GOOD, url="http://example.com:80"))

    def test_profile_support_tags(self):
        plain = embodiment_profile(deployment()).supports
        with_ik = embodiment_profile(deployment(GOOD)).supports
        self.assertIn("reach", plain)
        self.assertNotIn("approach_roll", plain)
        self.assertIn("reach", with_ik)
        self.assertIn("approach_roll", with_ik)
        # Tags that exist only with the cuRobo service: split bounds, the path check and its collision world.
        service_only = {"approach_roll", "path_check", "collision_world", "obstacle_object_ids"}
        self.assertEqual(with_ik - plain, service_only)
        self.assertEqual(with_ik - service_only, plain)
        self.assertIn("arrival_correction", plain)
        self.assertFalse((service_only | {"arrival_correction"}) - {"obstacle_object_ids"} & embodiments.ARX_X5_PROFILE.supports)
        self.assertTrue({"stall_guard", "settle", "execute_if_clean", "gripper_motion", "max_upward_dz"} <= plain)
        self.assertTrue({"reach", "approach_roll"} <= embodiments.SUPPORT_TAGS)
        self.assertFalse({"reach", "approach_roll"} & embodiments.ARX_X5_PROFILE.supports)


class SplitToleranceValidationTests(unittest.TestCase):
    def setUp(self):
        self.plain = embodiment_profile(deployment())
        self.split = embodiment_profile(deployment(GOOD))
        self.x5 = embodiments.ARX_X5_PROFILE

    def test_refused_without_approach_roll(self):
        for profile in (self.plain, self.x5):
            for field in mt.SPLIT_ORIENTATION_FIELDS:
                with self.assertRaises(ValueError) as caught:
                    mt.validate_tolerance_fields({field: 0.1}, profile=profile)
                self.assertTrue(str(caught.exception).startswith("unsupported_capability"))
                with self.assertRaises(ValueError):
                    mt.resolve_stage_tolerance({field: 0.1}, profile=profile)
                with self.assertRaises(ValueError):
                    mt.resolve_stage_tolerance({}, {field: 0.1}, profile=profile)

    def test_none_is_not_a_split_field(self):
        mt.validate_tolerance_fields({"approach_tolerance_rad": None, "roll_tolerance_rad": None}, profile=self.plain)

    def test_bounds(self):
        for field in mt.SPLIT_ORIENTATION_FIELDS:
            for good in (1e-6, 0.1, 3, math.pi):
                mt.validate_tolerance_fields({field: good}, profile=self.split)
            for bad in (0, -0.1, math.pi + 0.01, float("nan"), float("inf"), True, "0.1"):
                with self.assertRaises(ValueError, msg=repr(bad)) as caught:
                    mt.validate_tolerance_fields({field: bad}, profile=self.split)
                self.assertEqual(str(caught.exception), "invalid_" + field)

    def test_resolve_stage_then_batch(self):
        out = mt.resolve_stage_tolerance({"roll_tolerance_rad": 0.3}, {"approach_tolerance_rad": 0.05,
                                                                      "roll_tolerance_rad": 0.9}, profile=self.split)
        self.assertEqual((out["approach_tolerance_rad"], out["roll_tolerance_rad"]), (0.05, 0.3))
        out = mt.resolve_stage_tolerance({}, {"approach_tolerance_rad": 0.05}, profile=self.split)
        self.assertEqual(out["approach_tolerance_rad"], 0.05)
        self.assertNotIn("roll_tolerance_rad", out)
        out = mt.resolve_stage_tolerance({"tolerance_preset": "tight"}, profile=self.split)
        self.assertNotIn("approach_tolerance_rad", out)
        self.assertNotIn("roll_tolerance_rad", out)

    def test_path_fields_in_contract(self):
        path = {"kind": "line", "approach_tolerance_rad": 0.05, "roll_tolerance_rad": math.pi}
        contract._validate_path_fields(dict(path), self.split)
        for profile in (self.plain, self.x5):
            for field in contract.SPLIT_ORIENTATION_PATH_FIELDS:
                with self.assertRaises(ValueError) as caught:
                    contract._validate_path_fields({"kind": "line", field: 0.1}, profile)
                self.assertIn("unsupported_capability: " + field, str(caught.exception))
        for field in contract.SPLIT_ORIENTATION_PATH_FIELDS:
            contract._validate_path_fields({"kind": "line", field: None}, self.split)  # None = unset, allowed
            for bad in (0, -1, 3.2, True, float("nan"), "0.1"):
                with self.assertRaises(ValueError, msg=repr(bad)):
                    contract._validate_path_fields({"kind": "line", field: bad}, self.split)

    def test_offered_unsupported_batch_fields(self):
        self.assertTrue(set(contract.SPLIT_ORIENTATION_PATH_FIELDS) <=
                        set(contract._unsupported_batch_fields(self.plain)))
        self.assertFalse(set(contract.SPLIT_ORIENTATION_PATH_FIELDS) &
                         set(contract._unsupported_batch_fields(self.split)))

# === tests/test_so101_kinematics.py ===
# SO-101 & N-DoF kinematics, Pinocchio RNEA gravity, Pink+DAQP IK, and Coal collision tests.

import math
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ni_rua.robot.embodiments.arx_x5 import arx_x5_chain
from ni_rua.robot.embodiments.so101 import gravity as so101_gravity, gripper_cad as gg
from ni_rua.robot.model import kinematics as generic_kin
from ni_rua.robot.model import so101_kinematics as kin
from ni_rua.robot.model.pinocchio_backend import (
    pinocchio_gravity_torques,
    so101_collision_engine,
    solve_so101_pink_ik,
)
from ni_rua.robot.model.robot_model import urdf_manifest


def rotation_angle(a, b):
    return math.acos(max(-1.0, min(1.0, (np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0)))


def _reference_chain_fk_and_jacobian(chain: generic_kin.KinematicChainSpec, q: np.ndarray, T_world_base: np.ndarray | None = None):
    """Independent pure-NumPy reference forward kinematics and geometric Jacobian from ``KinematicChainSpec``."""
    cur = T_world_base.copy() if T_world_base is not None else np.eye(4, dtype=np.float64)
    poses = {chain.base_link: cur.copy()}
    origins = np.empty((chain.dof, 3), dtype=np.float64)
    axes_world = np.empty((chain.dof, 3), dtype=np.float64)
    for idx in range(chain.dof):
        frame = cur @ chain.parent_transforms[idx]
        axis_loc = chain.joint_axes[idx]
        origins[idx] = frame[:3, 3]
        axes_world[idx] = frame[:3, :3] @ axis_loc
        motion = np.eye(4, dtype=np.float64)
        if chain.joint_types[idx] in ("revolute", "continuous"):
            motion[:3, :3] = Rotation.from_rotvec(axis_loc * float(q[idx])).as_matrix()
        else:
            motion[:3, 3] = axis_loc * float(q[idx])
        cur = frame @ motion
        poses[chain.link_names[idx]] = cur.copy()
    tip = cur @ chain.tip_transform
    poses[chain.tip_link] = tip.copy()
    jac = np.zeros((6, chain.dof), dtype=np.float64)
    for idx in range(chain.dof):
        if chain.joint_types[idx] in ("revolute", "continuous"):
            jac[:3, idx] = np.cross(axes_world[idx], tip[:3, 3] - origins[idx])
            jac[3:, idx] = axes_world[idx]
        else:
            jac[:3, idx] = axes_world[idx]
    return poses, tip, jac


class SO101KinematicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = kin.load_model()
        limits = kin.joint_limits_rad(cls.model)
        cls.lower = np.array([limits[name][0] for name in kin.ARM_JOINTS])
        cls.upper = np.array([limits[name][1] for name in kin.ARM_JOINTS])
        cls.limits_map = {name: limits[name] for name in kin.ARM_JOINTS}

    def test_joint_names_match_lerobot_driver_and_urdf(self):
        self.assertEqual(kin.ARM_JOINTS, ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"))
        names = {joint["name"]: joint for joint in self.model["joints"]}
        for name in (*kin.ARM_JOINTS, kin.GRIPPER_JOINT):
            self.assertEqual(names[name]["type"], "revolute")
        self.assertEqual(self.model["base_link"], kin.BASE_LINK)
        self.assertEqual(names["gripper_frame_joint"]["parent"], kin.WRIST_CAMERA_PARENT_LINK)
        self.assertEqual(names["wrist_camera_mount_joint"]["parent"], kin.WRIST_CAMERA_PARENT_LINK)
        limits = kin.joint_limits_rad(self.model)
        self.assertEqual(limits["gripper"], (-0.174533, 1.74533))
        self.assertEqual(limits["wrist_roll"], (-2.74385, 2.84121))
        self.assertTrue(all(lo < 0 < hi for lo, hi in limits.values()))

    def test_zero_configuration_tcp_is_ahead_and_above_base(self):
        tcp = kin.fk(self.model, [0.0] * 5)
        np.testing.assert_allclose(tcp[:3, 3], [0.3914, 0.0, 0.2265], atol=5e-4)
        # Approach axis (local +Z) points forward along base +X at zero.
        np.testing.assert_allclose(tcp[:3, 2], [1.0, 0.0, 0.0], atol=1e-3)
        poses = kin.fk_all(self.model, [0.0] * 5)
        for link in (
            "base_link",
            "shoulder_link",
            "gripper_link",
            "gripper_frame_link",
            "moving_jaw_so101_v1_link",
            "wrist_camera_mount_link",
            "wrist_camera_link",
        ):
            self.assertIn(link, poses)
        np.testing.assert_allclose(poses["gripper_frame_link"], tcp)
        np.testing.assert_allclose(kin.fk(self.model, [0.0] * 5, link="gripper_link"), poses["gripper_link"])

    def test_shoulder_pan_rotates_tcp_about_base_z(self):
        zero = kin.fk(self.model, [0.0] * 5)[:3, 3]
        turned = kin.fk(self.model, [0.5, 0, 0, 0, 0])[:3, 3]
        # URDF rpy uses 3.14159 rather than pi, so the pan axis is tilted by ~3e-6 rad.
        self.assertAlmostEqual(zero[2], turned[2], places=5)
        self.assertNotAlmostEqual(zero[1], turned[1], places=3)

    def test_gripper_angle_moves_only_the_jaw(self):
        closed = kin.fk_all(self.model, [0.1] * 5, gripper_rad=-0.17)
        opened = kin.fk_all(self.model, [0.1] * 5, gripper_rad=1.7)
        np.testing.assert_allclose(closed["gripper_frame_link"], opened["gripper_frame_link"])
        self.assertFalse(np.allclose(closed["moving_jaw_so101_v1_link"], opened["moving_jaw_so101_v1_link"]))

    def test_ik_round_trip_on_random_reachable_configurations(self):
        rng = np.random.default_rng(7)
        position_errors, orientation_errors = [], []
        for _ in range(60):
            q = rng.uniform(self.lower * 0.9, self.upper * 0.9)
            target = kin.fk(self.model, q)
            seed = np.clip(q + rng.normal(0.0, 0.5, 5), self.lower, self.upper)
            result = kin.ik(self.model, target, seed)
            self.assertEqual(set(result), {"q_rad", "position_error_m", "orientation_error_rad", "converged", "iterations"})
            self.assertTrue(result["converged"], result)
            achieved = kin.fk(self.model, result["q_rad"])
            # Reported residuals are the true achieved residuals.
            self.assertAlmostEqual(np.linalg.norm(achieved[:3, 3] - target[:3, 3]), result["position_error_m"], places=9)
            self.assertAlmostEqual(rotation_angle(achieved, target), result["orientation_error_rad"], places=6)
            self.assertTrue(np.all(np.asarray(result["q_rad"]) >= self.lower - 1e-12))
            self.assertTrue(np.all(np.asarray(result["q_rad"]) <= self.upper + 1e-12))
            position_errors.append(result["position_error_m"])
            orientation_errors.append(result["orientation_error_rad"])
        self.assertLess(max(position_errors), 1e-3)
        self.assertLess(max(orientation_errors), 1e-2)

    def test_unreachable_target_reports_honest_residuals(self):
        target = np.eye(4)
        target[:3, 3] = [2.0, 0.0, 0.5]
        result = kin.ik(self.model, target, [0.0] * 5)
        self.assertFalse(result["converged"])
        self.assertGreater(result["position_error_m"], 1.0)
        achieved = kin.fk(self.model, result["q_rad"])
        self.assertAlmostEqual(np.linalg.norm(achieved[:3, 3] - target[:3, 3]), result["position_error_m"], places=9)

    def test_five_dof_orientation_conflict_is_not_masked(self):
        # Reachable position, but an orientation the 5-DOF wrist cannot produce there.
        target = kin.fk(self.model, [0.6, 0.2, 0.3, 0.4, 0.0]).copy()
        twist = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])  # 90 deg about base X
        target[:3, :3] = twist @ target[:3, :3]
        result = kin.ik(self.model, target, [0.6, 0.2, 0.3, 0.4, 0.0], restarts=2)
        achieved = kin.fk(self.model, result["q_rad"])
        self.assertAlmostEqual(rotation_angle(achieved, target), result["orientation_error_rad"], places=6)
        if result["orientation_error_rad"] > 1e-2 or result["position_error_m"] > 1e-3:
            self.assertFalse(result["converged"])
        position_only = kin.ik(self.model, target, [0.0] * 5, orientation_weight=0.0)
        self.assertTrue(position_only["converged"])
        self.assertLess(position_only["position_error_m"], 1e-3)

    def yaw_conflict_target(self, q=(0.0, -1.0, 1.0, 0.6, 0.0), dy=0.1):
        # Real failure (run 6d8a8065): the measured quaternion was copied to a laterally shifted
        # position. Gripper yaw is tied to shoulder_pan, so that 6D pose is outside the 5-DOF manifold.
        target = kin.fk(self.model, list(q)).copy()
        target[1, 3] += dy
        target[2, 3] -= 0.05
        return target

    def test_position_priority_reaches_position_with_smallest_orientation_error(self):
        target = self.yaw_conflict_target()
        seed = [0.0, -1.0, 1.0, 0.6, 0.0]
        weighted = kin.ik(self.model, target, seed)
        position_only = kin.ik(self.model, target, seed, orientation_weight=0.0)
        priority = kin.ik_position_priority(
            self.model,
            target,
            seed,
            position_tolerance_m=0.01,
            orientation_tolerance_rad=0.6,
        )
        self.assertEqual(set(priority), {"q_rad", "position_error_m", "orientation_error_rad", "converged", "iterations"})
        # The weighted cost trades centimetres of position for orientation; position priority does not.
        self.assertGreater(weighted["position_error_m"], 0.02)
        self.assertLess(priority["position_error_m"], 1e-4)
        self.assertLess(priority["orientation_error_rad"], position_only["orientation_error_rad"] - 0.05)
        self.assertTrue(priority["converged"])
        achieved = kin.fk(self.model, priority["q_rad"])
        self.assertAlmostEqual(np.linalg.norm(achieved[:3, 3] - target[:3, 3]), priority["position_error_m"], places=9)
        self.assertAlmostEqual(rotation_angle(achieved, target), priority["orientation_error_rad"], places=6)
        # Same pose, stricter orientation tolerance: the position is still reached but convergence is not claimed.
        strict = kin.ik_position_priority(
            self.model,
            target,
            seed,
            position_tolerance_m=0.01,
            orientation_tolerance_rad=0.15,
            restarts=0,
        )
        self.assertFalse(strict["converged"])
        self.assertLess(strict["position_error_m"], 1e-4)
        self.assertAlmostEqual(strict["orientation_error_rad"], priority["orientation_error_rad"], places=3)
        # The orientation it reports is exactly reachable: a weighted solve of that pose converges.
        reachable = target.copy()
        reachable[:3, :3] = achieved[:3, :3]
        self.assertTrue(kin.ik(self.model, reachable, seed, position_tolerance_m=1e-3)["converged"])

    def test_position_priority_matches_reachable_targets_and_respects_limits(self):
        rng = np.random.default_rng(11)
        for _ in range(20):
            q = rng.uniform(self.lower * 0.9, self.upper * 0.9)
            target = kin.fk(self.model, q)
            seed = np.clip(q + rng.normal(0.0, 0.3, 5), self.lower, self.upper)
            result = kin.ik_position_priority(
                self.model,
                target,
                seed,
                position_tolerance_m=1e-3,
                orientation_tolerance_rad=1e-2,
            )
            self.assertLess(result["position_error_m"], 1e-3, result)
            self.assertTrue(np.all(np.asarray(result["q_rad"]) >= self.lower - 1e-12))
            self.assertTrue(np.all(np.asarray(result["q_rad"]) <= self.upper + 1e-12))
            if result["converged"]:
                self.assertLessEqual(result["orientation_error_rad"], 1e-2)

    def test_position_priority_reports_unreachable_position_honestly(self):
        target = np.eye(4)
        target[:3, 3] = [2.0, 0.0, 0.5]
        result = kin.ik_position_priority(self.model, target, [0.0] * 5, orientation_tolerance_rad=math.pi, restarts=2)
        self.assertFalse(result["converged"])
        self.assertGreater(result["position_error_m"], 1.0)
        achieved = kin.fk(self.model, result["q_rad"])
        self.assertAlmostEqual(np.linalg.norm(achieved[:3, 3] - target[:3, 3]), result["position_error_m"], places=9)
        bad = np.eye(4)
        bad[0, 0] = 2.0
        with self.assertRaises(ValueError):
            kin.ik_position_priority(self.model, bad, [0.0] * 5)
        with self.assertRaises(ValueError):
            kin.ik_position_priority(self.model, target, [0.0] * 5, limits={"gripper": (0, 1)})

    def test_explicit_limits_are_enforced(self):
        target = kin.fk(self.model, [0.8, 0.0, 0.0, 0.0, 0.0])
        result = kin.ik(self.model, target, [0.0] * 5, limits={"shoulder_pan": (-0.2, 0.2)})
        self.assertLessEqual(result["q_rad"][0], 0.2)
        self.assertFalse(result["converged"])
        with self.assertRaises(ValueError):
            kin.ik(self.model, target, [0.0] * 5, limits={"gripper": (0, 1)})
        with self.assertRaises(ValueError):
            kin.ik(self.model, target, [0.0] * 5, limits={"shoulder_pan": (1, -1)})

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            kin.fk(self.model, [0.0] * 6)
        with self.assertRaises(ValueError):
            kin.fk(self.model, [0.0, 0.0, math.nan, 0.0, 0.0])
        with self.assertRaises(ValueError):
            kin.fk(self.model, [0.0] * 5, link="no_such_link")
        bad = np.eye(4)
        bad[0, 0] = 2.0
        with self.assertRaises(ValueError):
            kin.ik(self.model, bad, [0.0] * 5)
        with self.assertRaises(ValueError):
            kin.ik(self.model, np.eye(4), [0.0] * 5, position_weight=0.0)

    def test_pinocchio_fk_and_jacobian_parity_across_so101_and_arx_x5(self):
        rng = np.random.default_rng(2026)
        t_wb = generic_kin.rpy_xyz_to_matrix((0.15, -0.22, 0.76), (0.12, -0.08, 0.45))
        for chain in (kin.so101_chain(self.model), arx_x5_chain()):
            lo = chain.joint_limits_rad[:, 0] * 0.85
            hi = chain.joint_limits_rad[:, 1] * 0.85
            q_samples = rng.uniform(lo, hi, size=(25, chain.dof))
            pos_b, rot_b, jac_b, _ = generic_kin.fk_batch(chain, q_samples)
            for idx in range(len(q_samples)):
                q_row = q_samples[idx]
                ref_poses, ref_tip, ref_jac = _reference_chain_fk_and_jacobian(chain, q_row)
                poses_pin = generic_kin.fk_links(chain, q_row)
                tip_pin, jac_pin = generic_kin.fk_jacobian(chain, q_row)
                for link_name, ref_mat in ref_poses.items():
                    np.testing.assert_allclose(poses_pin[link_name], ref_mat, atol=1e-12)
                np.testing.assert_allclose(tip_pin, ref_tip, atol=1e-12)
                np.testing.assert_allclose(jac_pin, ref_jac, atol=1e-12)
                np.testing.assert_allclose(pos_b[idx], ref_tip[:3, 3], atol=1e-12)
                np.testing.assert_allclose(rot_b[idx], ref_tip[:3, :3], atol=1e-12)
                np.testing.assert_allclose(jac_b[idx], ref_jac, atol=1e-12)

                _, ref_tip_w, ref_jac_w = _reference_chain_fk_and_jacobian(chain, q_row, T_world_base=t_wb)
                tip_w, jac_w = generic_kin.fk_jacobian(chain, q_row, T_world_base=t_wb)
                np.testing.assert_allclose(tip_w, ref_tip_w, atol=1e-12)
                np.testing.assert_allclose(jac_w, ref_jac_w, atol=1e-12)

    def test_chain_from_manifest_imports_zero_native_deps_at_import_time(self):
        repo_root = REPO_ROOT
        code = (
            "import sys; "
            "from ni_rua.robot.model.kinematics import KinematicChainSpec, chain_from_manifest; "
            "assert 'pinocchio' not in sys.modules, list(sys.modules); "
            "assert 'pink' not in sys.modules, list(sys.modules); "
            "assert 'coal' not in sys.modules, list(sys.modules)"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_pinocchio_rnea_gravity_torques_match_custom_tree_walk_to_1e_14_nm(self):
        def _ref_tree_walk_gravity(manifest, inertials_map, q_vec, jaw_rad=0.0):
            poses, axes = kin._chain(manifest, kin._arm_vector(q_vec), float(jaw_rad))
            children = {}
            for joint in manifest["joints"]:
                children.setdefault(joint["parent"], []).append(joint["child"])
            weights = {}
            for link, info in inertials_map.items():
                if link in poses and info["mass_kg"] > 0:
                    centre = (poses[link] @ np.append(np.asarray(info["com_m"], dtype=float), 1.0))[:3]
                    weights[link] = (centre, np.array([0.0, 0.0, -info["mass_kg"] * so101_gravity.GRAVITY_M_S2]))
            out = np.zeros(len(kin.ARM_JOINTS), dtype=np.float64)
            for idx_arm, j_name in enumerate(kin.ARM_JOINTS):
                origin, axis = axes[j_name]
                joint_entry = next(j for j in manifest["joints"] if j["name"] == j_name)
                queue = [joint_entry["child"]]
                while queue:
                    cur_link = queue.pop()
                    if cur_link in weights:
                        c_pt, w_vec = weights[cur_link]
                        out[idx_arm] += float(np.dot(axis, np.cross(c_pt - origin, w_vec)))
                    queue.extend(children.get(cur_link, []))
            return out

        inertials = so101_gravity.load_inertials()
        rng = np.random.default_rng(99)
        for _ in range(40):
            q_arm = rng.uniform(self.lower, self.upper)
            jaw = float(rng.uniform(-0.1745, 1.7453))
            tau_pin = pinocchio_gravity_torques(None, q_arm, gripper_rad=jaw)
            tau_delegated = so101_gravity.gravity_torques(self.model, inertials, q_arm, gripper_rad=jaw)
            tau_ref = _ref_tree_walk_gravity(self.model, inertials, q_arm, jaw_rad=jaw)
            self.assertEqual(tau_pin.shape, (5,))
            self.assertEqual(tau_pin.dtype, np.float64)
            np.testing.assert_allclose(tau_pin, tau_ref, atol=1e-14, rtol=0.0)
            np.testing.assert_allclose(tau_delegated, tau_ref, atol=1e-14, rtol=0.0)

        with tempfile.TemporaryDirectory() as td:
            links, parent = [], "base_link"
            for index, name in enumerate(kin.ARM_JOINTS):
                child = f"l{index}"
                inertial = '<inertial><origin xyz="0.12 0 0"/><mass value="0.25"/></inertial>' if name == "elbow_flex" else ""
                links.append(
                    f'<link name="{child}">{inertial}</link>'
                    f'<joint name="{name}" type="revolute"><parent link="{parent}"/><child link="{child}"/>'
                    f'<axis xyz="{"0 0 1" if index == 0 else "0 1 0"}"/><limit lower="-3" upper="3"/></joint>'
                )
                parent = child
            urdf_file = Path(td) / "point_mass.urdf"
            urdf_file.write_text(
                f'<robot name="pm"><link name="base_link"/>{"".join(links)}'
                '<link name="gripper_frame_link"/><joint name="gf" type="fixed">'
                '<parent link="l4"/><child link="gripper_frame_link"/></joint></robot>',
                encoding="utf-8",
            )
            syn_manifest = urdf_manifest(urdf_file)
            syn_inertials = so101_gravity.load_inertials(urdf_file)
            q_syn = [0.4, math.radians(25.0), math.radians(35.0), -0.2, 0.5]
            tau_urdf = pinocchio_gravity_torques(urdf_file, q_syn)
        # Verify synthetic in-memory (manifest, inertials) works AFTER TemporaryDirectory deletion!
        self.assertFalse(urdf_file.exists())
        tau_after_delete = so101_gravity.gravity_torques(syn_manifest, syn_inertials, q_syn)
        tau_syn_ref = _ref_tree_walk_gravity(syn_manifest, syn_inertials, q_syn)
        np.testing.assert_allclose(tau_after_delete, tau_urdf, atol=1e-14, rtol=0.0)
        np.testing.assert_allclose(tau_after_delete, tau_syn_ref, atol=1e-14, rtol=0.0)

    def test_solve_so101_pink_ik_100_percent_convergence_and_candidate_modes(self):
        rng = np.random.default_rng(314)
        for _ in range(30):
            q_true = rng.uniform(self.lower * 0.88, self.upper * 0.88)
            target = kin.fk(self.model, q_true)
            seed = np.clip(q_true + rng.normal(0.0, 0.35, size=5), self.lower, self.upper)

            full_sol = solve_so101_pink_ik(target, seed, self.limits_map)
            self.assertTrue(full_sol["converged"], full_sol)
            self.assertLessEqual(full_sol["position_error_m"], 1e-3)
            self.assertLessEqual(full_sol["orientation_error_rad"], 1e-2)

            # Approach-only (+Z) with free roll around local Z: cost=(0.3, 0.3, 0.0)
            rolled_target = target.copy()
            rolled_target[:3, :3] = target[:3, :3] @ Rotation.from_euler("z", 1.1, degrees=False).as_matrix()
            app_sol = solve_so101_pink_ik(
                rolled_target,
                seed,
                self.limits_map,
                orientation_costs=(0.3, 0.3, 0.0),
            )
            self.assertTrue(app_sol["converged"], app_sol)
            self.assertLessEqual(app_sol["position_error_m"], 1e-3)
            T_app = kin.fk(self.model, app_sol["q_rad"])
            approach_angle = math.acos(max(-1.0, min(1.0, float(T_app[:3, 2] @ rolled_target[:3, 2]))))
            self.assertLessEqual(approach_angle, 1e-2)

        all_cands = solve_so101_pink_ik(
            kin.fk(self.model, [0.0, -0.95, 0.9, 1.0, 0.0]),
            [0.1, -0.8, 0.8, 0.9, -0.1],
            self.limits_map,
            restarts=5,
            return_all_candidates=True,
        )
        self.assertIsInstance(all_cands, list)
        self.assertEqual(len(all_cands), 6)
        self.assertTrue(all_cands[0]["converged"])

    def test_coal_collision_engine_open_jaw_pocket_table_obstacles_and_25_step_speed(self):
        engine = so101_collision_engine()
        q_ready = [0.0, -0.95, 0.9, 1.0, 0.0]
        jaw_open = 1.0

        # Commissioned ready pose above z=0 table is completely collision-free.
        clear_state = engine.check_state(q_ready, jaw_rad=jaw_open, table_top_z_m=0.0)
        self.assertTrue(clear_state["self_and_table_free"], clear_state)
        self.assertTrue(clear_state["obstacle_free"], clear_state)
        self.assertEqual(clear_state["self"], [])
        self.assertEqual(clear_state["world"], [])

        # Risk #4 Open-Jaw Primitives: a 24 mm object centered between the open fingers must NOT collide!
        jaw_grasp_open = 0.50  # ~56 mm fingertip gap; moving jaw inner face at x <= -0.055 m, fixed jaw at x >= 0.0 m
        T_tcp = kin.fk(self.model, q_ready)
        grasp_ref = gg.grasp_centre_reference(jaw_grasp_open)
        obj_centre_tcp = np.array([grasp_ref["centre_m"][0], 0.0, -0.008], dtype=np.float64)
        obj_centre_base = T_tcp[:3, :3] @ obj_centre_tcp + T_tcp[:3, 3]
        between_fingers_world = {
            "obstacles": [
                {
                    "id": "target_cube",
                    "min_m": (obj_centre_base - np.array([0.012, 0.012, 0.012])).tolist(),
                    "max_m": (obj_centre_base + np.array([0.012, 0.012, 0.012])).tolist(),
                }
            ]
        }
        open_jaw_verdict = engine.check_state(q_ready, jaw_rad=jaw_grasp_open, table_top_z_m=0.0, world=between_fingers_world)
        self.assertTrue(open_jaw_verdict["self_and_table_free"], open_jaw_verdict)
        self.assertTrue(open_jaw_verdict["obstacle_free"], open_jaw_verdict)

        # Closing the moving jaw onto that same box (jaw=0.0 -> moving finger at x in [-0.028, -0.016]) MUST detect collision!
        closed_jaw_verdict = engine.check_state(q_ready, jaw_rad=0.0, table_top_z_m=0.0, world=between_fingers_world)
        self.assertTrue(closed_jaw_verdict["self_and_table_free"])
        self.assertFalse(closed_jaw_verdict["obstacle_free"])
        self.assertTrue(any("target_cube" in hit["obstacles"] for hit in closed_jaw_verdict["world"]))

        # Raising table_top_z_m above TCP height flags table collision with positive penetration_m.
        high_table = float(T_tcp[2, 3]) + 0.03
        table_hit = engine.check_state(q_ready, jaw_rad=0.0, table_top_z_m=high_table)
        self.assertFalse(table_hit["self_and_table_free"])
        self.assertFalse(table_hit["obstacle_free"])
        self.assertTrue(any("table" in hit["obstacles"] and hit["penetration_m"] > 0 for hit in table_hit["world"]))

        # Held attached_box colliding with an obstacle keeps self_and_table_free=True while obstacle_free=False.
        held_world = {
            "obstacles": [
                {
                    "id": "wall",
                    "min_m": (T_tcp[:3, 3] + np.array([0.01, -0.02, -0.02])).tolist(),
                    "max_m": (T_tcp[:3, 3] + np.array([0.05, 0.02, 0.02])).tolist(),
                }
            ],
            "attached_box": {
                "centre_tcp_m": [0.0, 0.0, 0.02],
                "half_extents_m": [0.03, 0.03, 0.03],
                "rotation_tcp": np.eye(3).tolist(),
            },
        }
        held_hit = engine.check_state(q_ready, jaw_rad=0.2, table_top_z_m=0.0, world=held_world)
        self.assertTrue(held_hit["self_and_table_free"])
        self.assertFalse(held_hit["obstacle_free"])
        self.assertTrue(any(hit["link"] == "attached_object" and "wall" in hit["obstacles"] for hit in held_hit["world"]))

        # Warm 25-sample check_path must execute in < 10 ms and match CuroboClient.check_path contract.
        seg_25 = [
            {
                "start_q_rad": q_ready,
                "end_q_rad": [0.24, -0.85, 0.85, 0.95, 0.15],
                "start_jaw_rad": 0.2,
                "end_jaw_rad": 0.4,
                "attached": True,
            }
        ]
        engine.check_path(seg_25, table_top_z_m=0.0, max_joint_step_rad=0.01)
        t0 = time.perf_counter()
        path_res = engine.check_path(seg_25, table_top_z_m=0.0, max_joint_step_rad=0.01)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.assertEqual(path_res["samples"], 25)
        self.assertEqual(path_res["segments"][0]["samples"], 25)
        self.assertEqual(path_res["segments"][0]["colliding_samples"], 0)
        self.assertIsNone(path_res["segments"][0]["first_collision"])
        self.assertLess(elapsed_ms, 10.0, f"25-sample check_path took {elapsed_ms:.2f} ms")

    def test_pure_numpy_fk_and_endpoint_polish_fallback_when_pinocchio_and_pink_absent(self):
        repo_root = REPO_ROOT
        script = """
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation

# Block native C++ kinematics/IK modules to simulate isolated RoboDojo robot_worker.py
sys.modules["pinocchio"] = None
sys.modules["pink"] = None

from ni_rua.robot.embodiments.arx_x5 import arx_x5_chain
from ni_rua.robot.model import geometry, kinematics as k
from ni_rua.robot.motion.endpoint_ik_polish import polish_trajectory_endpoint

# 1. fk_manifest / geometry.fk with revolute, prismatic, fixed, and mimic joints
manifest = {
    "base_link": "base_link",
    "joints": [
        {
            "name": "j1",
            "type": "revolute",
            "parent": "base_link",
            "child": "link1",
            "T_parent_joint": k.rpy_xyz_to_matrix((0.0, 0.0, 0.1), (0.0, 0.0, 0.0)).tolist(),
            "axis": [0.0, 0.0, 1.0],
        },
        {
            "name": "j2_slide",
            "type": "prismatic",
            "parent": "link1",
            "child": "link2",
            "T_parent_joint": np.eye(4).tolist(),
            "axis": [1.0, 0.0, 0.0],
        },
        {
            "name": "j3_mimic",
            "type": "revolute",
            "parent": "link2",
            "child": "link3",
            "T_parent_joint": k.rpy_xyz_to_matrix((0.2, 0.0, 0.0), (0.0, 0.0, 0.0)).tolist(),
            "axis": [0.0, 1.0, 0.0],
            "mimic": {"joint": "j1", "multiplier": -0.5, "offset": 0.1},
        },
        {
            "name": "tip_fixed",
            "type": "fixed",
            "parent": "link3",
            "child": "tip_link",
            "T_parent_joint": k.rpy_xyz_to_matrix((0.05, 0.0, 0.0), (0.0, 0.0, 0.0)).tolist(),
        },
    ],
}
state = {"joint_names": ["j1", "j2_slide"], "q_rad": [0.4, 0.15], "source_kind": "estimate"}
poses = geometry.fk(manifest, state)
assert set(poses) == {"base_link", "link1", "link2", "link3", "tip_link"}

# 2. fk_links on KinematicChainSpec with T_world_base
chain = arx_x5_chain()
base = k.rpy_xyz_to_matrix((0.1, -0.2, 0.75), (0.0, 0.0, 0.3))
q_goal = np.array([0.2, 0.8, 0.9, -0.4, 0.3, -0.2], dtype=float)
chain_poses = k.fk_links(chain, q_goal, T_world_base=base)
T_goal = chain_poses[chain.tip_link]
quat_xyzw = Rotation.from_matrix(T_goal[:3, :3]).as_quat(scalar_first=False).tolist()
target = {"position_m": T_goal[:3, 3].tolist(), "quaternion_xyzw": quat_xyzw}

# 3. polish_trajectory_endpoint with BOTH pose_error.chain and pose_error.world_link6 (robot_worker.py:760-766)
q_sloppy = q_goal + np.array([0.006, -0.005, 0.004, -0.005, 0.003, 0.0], dtype=float)
raw_traj = np.linspace(np.zeros(6), q_sloppy, 12)

def world_link6(q):
    return k.fk_links(chain, q, T_world_base=base)[chain.tip_link]

def pose_err(q):
    return k.pose_error(world_link6(q), T_goal)

pose_err.world_link6 = world_link6
pose_err.chain = chain
pose_err.T_world_base = base

diag_worker = {}
polished_worker = polish_trajectory_endpoint(
    raw_traj,
    limits=chain.joint_limits_rad,
    target=target,
    pose_error=pose_err,
    diagnostic=diag_worker,
)
assert diag_worker.get("selected") is True, diag_worker
assert diag_worker["after_position_error_m"] < diag_worker["before_position_error_m"] * 0.25, diag_worker
assert np.array_equal(polished_worker[0], raw_traj[0])

# Also verify chain-only call (world_link6=None) falls back cleanly via fk_links
def bare_pose_err(q):
    return k.pose_error(world_link6(q), T_goal)

diag_chain_only = {}
polished_chain_only = polish_trajectory_endpoint(
    raw_traj,
    limits=chain.joint_limits_rad,
    target=target,
    pose_error=bare_pose_err,
    chain=chain,
    T_world_base=base,
    diagnostic=diag_chain_only,
)
assert diag_chain_only.get("selected") is True, diag_chain_only
assert diag_chain_only["after_position_error_m"] < diag_chain_only["before_position_error_m"] * 0.25, diag_chain_only
print(json.dumps({"tip": poses["tip_link"].tolist(), "chain_tip": T_goal.tolist()}))
"""
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        import json as _json

        payload = _json.loads(proc.stdout.strip().splitlines()[-1])
        # Cross-check pure-NumPy FK subprocess output against in-process Pinocchio FK!
        manifest = {
            "base_link": "base_link",
            "joints": [
                {
                    "name": "j1",
                    "type": "revolute",
                    "parent": "base_link",
                    "child": "link1",
                    "T_parent_joint": generic_kin.rpy_xyz_to_matrix((0.0, 0.0, 0.1), (0.0, 0.0, 0.0)).tolist(),
                    "axis": [0.0, 0.0, 1.0],
                },
                {
                    "name": "j2_slide",
                    "type": "prismatic",
                    "parent": "link1",
                    "child": "link2",
                    "T_parent_joint": np.eye(4).tolist(),
                    "axis": [1.0, 0.0, 0.0],
                },
                {
                    "name": "j3_mimic",
                    "type": "revolute",
                    "parent": "link2",
                    "child": "link3",
                    "T_parent_joint": generic_kin.rpy_xyz_to_matrix((0.2, 0.0, 0.0), (0.0, 0.0, 0.0)).tolist(),
                    "axis": [0.0, 1.0, 0.0],
                    "mimic": {"joint": "j1", "multiplier": -0.5, "offset": 0.1},
                },
                {
                    "name": "tip_fixed",
                    "type": "fixed",
                    "parent": "link3",
                    "child": "tip_link",
                    "T_parent_joint": generic_kin.rpy_xyz_to_matrix((0.05, 0.0, 0.0), (0.0, 0.0, 0.0)).tolist(),
                },
            ],
        }
        pin_tip = generic_kin.fk_manifest(
            manifest,
            {"joint_names": ["j1", "j2_slide"], "q_rad": [0.4, 0.15], "source_kind": "estimate"},
        )["tip_link"]
        np.testing.assert_allclose(np.asarray(payload["tip"]), pin_tip, atol=1e-12)
        pin_chain_tip = generic_kin.fk_links(
            arx_x5_chain(),
            [0.2, 0.8, 0.9, -0.4, 0.3, -0.2],
            T_world_base=generic_kin.rpy_xyz_to_matrix((0.1, -0.2, 0.75), (0.0, 0.0, 0.3)),
        )["link6"]
        np.testing.assert_allclose(np.asarray(payload["chain_tip"]), pin_chain_tip, atol=1e-12)

    def test_solve_ik_and_solve_ik_batch_honor_explicit_iteration_caps(self):
        one_dof = generic_kin.KinematicChainSpec(
            joint_names=("j1",),
            joint_types=("revolute",),
            parent_transforms=np.eye(4, dtype=np.float64)[None, :, :],
            joint_axes=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
            joint_limits_rad=np.array([[-3.0, 3.0]], dtype=np.float64),
            tip_transform=generic_kin.rpy_xyz_to_matrix((0.30, 0.0, 0.0), (0.0, 0.0, 0.0)),
            base_link="base_link",
            tip_link="tip_link",
            link_names=("link1",),
        )
        # 1. solve_ik(max_iterations=200, max_step_rad=0.001) on a 0.18 rad move requires >= 180 steps (previously capped at 120!)
        T_18 = generic_kin.fk_links(one_dof, [0.18])["tip_link"]
        res_200 = generic_kin.solve_ik(
            one_dof,
            T_18,
            [0.0],
            generic_kin.IKOptions(
                tasks=("pose",),
                max_iterations=200,
                max_step_rad=0.001,
                position_tolerance_m=1e-4,
                orientation_tolerance_rad=1e-3,
            ),
        )
        self.assertTrue(res_200.converged, res_200)
        self.assertGreaterEqual(res_200.iterations, 179)
        self.assertAlmostEqual(float(res_200.q_rad[0]), 0.18, places=3)

        # Also verify ("pose", "posture") honors both high (max_iterations=200) and low (max_iterations=10) budgets honestly
        res_posture_200 = generic_kin.solve_ik(
            one_dof,
            generic_kin.IKTarget(T_base_tip=T_18, posture_rad=np.array([0.18])),
            [0.0],
            generic_kin.IKOptions(
                tasks=("pose", "posture"),
                posture_weight=0.1,
                max_iterations=200,
                max_step_rad=0.001,
                position_tolerance_m=1e-4,
                orientation_tolerance_rad=1e-3,
            ),
        )
        self.assertTrue(res_posture_200.converged, res_posture_200)
        self.assertGreaterEqual(res_posture_200.iterations, 179)

        res_posture_10 = generic_kin.solve_ik(
            one_dof,
            generic_kin.IKTarget(T_base_tip=T_18, posture_rad=np.array([0.18])),
            [0.0],
            generic_kin.IKOptions(
                tasks=("pose", "posture"),
                posture_weight=0.1,
                max_iterations=10,
                max_step_rad=0.001,
                position_tolerance_m=1e-4,
                orientation_tolerance_rad=1e-3,
            ),
        )
        self.assertLessEqual(res_posture_10.iterations, 10)
        self.assertFalse(res_posture_10.converged)

        # 2. solve_ik_batch(max_iters=36, max_step_rad=0.01) on a 0.25 rad move reaches full convergence instead of halting at 0.20 rad
        T_25 = generic_kin.fk_links(one_dof, [0.25])["tip_link"]
        q_batch = generic_kin.solve_ik_batch(
            one_dof,
            T_25[:3, 3],
            T_25[:3, :3],
            np.array([[0.0]], dtype=np.float64),
            pos_tol_m=1e-4,
            rot_tol_rad=1e-3,
            max_iters=36,
            max_step_rad=0.01,
        )
        self.assertAlmostEqual(float(q_batch[0, 0]), 0.25, places=3)
        T_ach = generic_kin.fk_links(one_dof, q_batch[0])["tip_link"]
        pe, oe = generic_kin.pose_error(T_ach, T_25)
        self.assertLessEqual(pe, 1e-4)
        self.assertLessEqual(oe, 1e-3)


class LocalPinkSolveTests(unittest.TestCase):
    IK_CONFIG = dict(max_iterations=60, restarts=8, orientation_weight_m_per_rad=0.3)

    def test_100pct_convergence_on_reachable_5dof_poses(self):
        rng = np.random.default_rng(17)
        lows = np.array([lo + 0.1 for lo, _ in IK_PLAN_LIMITS.values()], dtype=float)
        highs = np.array([hi - 0.1 for _, hi in IK_PLAN_LIMITS.values()], dtype=float)
        for index in range(30):
            q_target = rng.uniform(lows, highs)
            seed = np.clip(q_target + rng.normal(0.0, 0.35, size=5),
                           [lo for lo, _ in IK_PLAN_LIMITS.values()],
                           [hi for _, hi in IK_PLAN_LIMITS.values()])
            q, record = local_solve(kin, MODEL, pose_of(q_target.tolist()), seed.tolist(),
                                    (1e-3, 1e-2), self.IK_CONFIG, IK_PLAN_LIMITS)
            self.assertTrue(record["ok"], f"pose {index} failed: {record}")
            self.assertLessEqual(record["position_error_m"], 1e-3)
            self.assertLessEqual(record["orientation_error_rad"], 1e-2)
            for value, (lo, hi) in zip(q, IK_PLAN_LIMITS.values()):
                self.assertTrue(lo - 1e-6 <= value <= hi + 1e-6)

    def test_split_approach_and_roll_tolerances(self):
        q_near_roll_hi = [0.2, -0.5, 0.9, 0.4, IK_PLAN_LIMITS["wrist_roll"][1] - 0.05]
        base_pose = pose_of(q_near_roll_hi)
        # Flip jaw roll by pi when wrist_roll is already near its upper limit (+2.8 rad) AND check if -0.34 rad is
        # reachable vs an out-of-range roll. Instead construct a roll beyond both limits if possible, or test a
        # tilted out-of-plane roll vs free roll!
        R_base = quaternion_to_matrix(base_pose["quaternion_xyzw"])
        # Rotate target by 25 deg around TCP X (out-of-plane tilt unreachable on 5-DoF at fixed position) vs pure roll:
        R_rolled = R_base @ rot([0, 0, 1], 1.2)
        rolled_pose = dict(position_m=base_pose["position_m"], quaternion_xyzw=matrix_to_quaternion(R_rolled))
        # With free roll, approach +Z is kept within 0.01 rad and position within 1 mm even with restarts=0 from seed!
        single_restarts0 = dict(self.IK_CONFIG, restarts=0)
        q_free, rec_free = local_solve(
            kin, MODEL, rolled_pose, q_near_roll_hi, (1e-3, 0.05), single_restarts0, IK_PLAN_LIMITS,
            split=dict(approach_tolerance_rad=0.01, roll_tolerance_rad=math.pi),
        )
        self.assertTrue(rec_free["ok"], rec_free)
        self.assertTrue(rec_free["orientation_checked"])
        self.assertLessEqual(rec_free["position_error_m"], 1e-3)
        self.assertLessEqual(rec_free["approach_error_rad"], 0.01)
        # Tightenwrist_roll limits so the +1.2 rad roll is strictly outside planning limits -> tight roll fails, free roll passes!
        narrow_limits = dict(IK_PLAN_LIMITS, wrist_roll=(q_near_roll_hi[4] - 0.2, IK_PLAN_LIMITS["wrist_roll"][1]))
        _, rec_tight = local_solve(
            kin, MODEL, rolled_pose, q_near_roll_hi, (1e-3, 0.05), self.IK_CONFIG, narrow_limits,
            split=dict(approach_tolerance_rad=0.02, roll_tolerance_rad=0.05),
        )
        self.assertFalse(rec_tight["ok"])
        self.assertIn("approach_error_rad", rec_tight)
        self.assertIn("roll_error_rad", rec_tight)
        self.assertIn("position_priority", rec_tight)
        self.assertIn("approach_error_rad", rec_tight["position_priority"])

    def test_position_priority_fallback_and_hint_on_5dof_attitude_tradeoff(self):
        home = [0.0, -1.5, 1.45, 0.9, 0.0]
        home_pose = pose_of(home)
        shifted_pose = dict(
            position_m=[home_pose["position_m"][0], home_pose["position_m"][1] + 0.08, home_pose["position_m"][2] - 0.05],
            quaternion_xyzw=home_pose["quaternion_xyzw"],
        )
        _, ok_rec = local_solve(kin, MODEL, shifted_pose, home, (0.01, 0.6), self.IK_CONFIG, IK_PLAN_LIMITS)
        self.assertTrue(ok_rec["ok"], ok_rec)
        self.assertEqual(ok_rec.get("solve"), "position_priority")
        self.assertLess(ok_rec["position_error_m"], 1e-3)
        _, fail_rec = local_solve(kin, MODEL, shifted_pose, home, (0.01, 0.15), self.IK_CONFIG, IK_PLAN_LIMITS)
        self.assertFalse(fail_rec["ok"])
        hint = fail_rec["position_priority"]
        self.assertEqual(set(hint), {"quaternion_xyzw", "position_error_m", "orientation_error_rad"})
        self.assertLess(hint["position_error_m"], 1e-3)


class RuckigAndCollisionGeometryTests(unittest.TestCase):
    def test_rest_to_rest_s_curve_lies_on_straight_joint_line_and_matches_step_count(self):
        start = [0.0, -1.0, 1.0, 0.6, 0.0]
        target = [0.45, -0.7, 1.2, 0.4, -0.3]
        steps = ik_plan.interpolation_steps(
            start, target, speed_rad_s=0.3, rate_hz=20.0,
            max_acceleration_rad_s2=12.0, max_jerk_rad_s3=80.0,
        )
        rows = ik_plan.joint_interpolation(
            start, target, speed_rad_s=0.3, rate_hz=20.0,
            max_acceleration_rad_s2=12.0, max_jerk_rad_s3=80.0,
        )
        self.assertEqual(len(rows), steps)
        np.testing.assert_allclose(rows[-1], target, atol=0.0)
        start_arr, delta = np.asarray(start), np.asarray(target) - np.asarray(start)
        denom = float(delta @ delta)
        s_prev = 0.0
        prev = start_arr
        for row in rows:
            s = float(((row - start_arr) @ delta) / denom)
            self.assertGreaterEqual(s, s_prev - 1e-12)
            self.assertLessEqual(s, 1.0 + 1e-12)
            np.testing.assert_allclose(row, start_arr + s * delta, atol=1e-12)
            self.assertLessEqual(float(np.abs(row - prev).max()), 0.3 / 20.0 + 1e-9)
            s_prev, prev = s, row
        # S-curve starts and ends gentler than peak cruise step
        step_norms = [float(np.linalg.norm(b - a)) for a, b in zip([start_arr, *rows[:-1]], rows)]
        self.assertLess(step_norms[0], max(step_norms))
        self.assertEqual(
            ik_plan.interpolation_steps(start, start, 0.3, 20.0, max_acceleration_rad_s2=12.0, max_jerk_rad_s3=80.0),
            1,
        )
        with self.assertRaises(ValueError):
            ik_plan.interpolation_steps(start, target, 0.3, 20.0, max_acceleration_rad_s2=12.0)
        with self.assertRaises(ValueError):
            ik_plan.joint_interpolation(start, target, max_acceleration_rad_s2=12.0, max_jerk_rad_s3=80.0)

    def test_mid_motion_online_retargeting_from_nonzero_state(self):
        start = [0.1, -0.8, 0.9, 0.5, 0.0]
        target = [-0.2, -0.6, 1.0, 0.4, 0.2]
        v0 = [0.25, -0.1, 0.05, 0.0, -0.15]
        a0 = [1.5, -0.5, 0.2, 0.0, -0.8]
        steps = ik_plan.interpolation_steps(
            start, target, speed_rad_s=0.5, rate_hz=50.0,
            max_acceleration_rad_s2=12.0, max_jerk_rad_s3=80.0,
            current_velocity=v0, current_acceleration=a0,
        )
        rows = ruckig_joint_interpolation(
            start, target, 0.5, 50.0, 12.0, 80.0,
            current_velocity=v0, current_acceleration=a0,
        )
        self.assertEqual(len(rows), steps)
        np.testing.assert_allclose(rows[-1], target, atol=0.0)
        # Positive initial velocity on joint 0 moves forward initially before braking toward target[0] < start[0]
        self.assertGreater(rows[0][0], start[0])

    def test_bounded_resample_ruckig_and_exact_coal_segment_box_distance(self):
        from ni_rua.robot.motion.motion_speed import bounded_resample
        from ni_rua.robot.motion.scene_obstacles import _segment_xy_distance_to_box

        knots = np.array([[0.0, 0.0], [0.2, 0.0], [0.2, 0.25]])
        resampled = bounded_resample(
            knots, [0.02, 0.02], max_acceleration_rad_s2=10.0, max_jerk_rad_s3=80.0, rate_hz=20.0
        )
        self.assertTrue(np.any(np.all(np.isclose(resampled, [0.2, 0.0], atol=1e-12), axis=1)))
        np.testing.assert_allclose(resampled[-1], [0.2, 0.25], atol=0.0)
        diffs = np.abs(np.diff(np.vstack([knots[0], resampled]), axis=0))
        self.assertLessEqual(float(diffs.max()), 0.02 + 1e-9)
        with self.assertRaises(ValueError):
            bounded_resample(knots, [0.02, 0.02], max_acceleration_rad_s2=10.0, max_jerk_rad_s3=80.0)

        # Segment crossing a small box strictly between 1/16th sample points (t = 1/32 = 0.03125):
        # discrete 17-point sampling missed this intersection (> 0.02 m error), while exact coal distance returns 0.0!
        dist_hit = _segment_xy_distance_to_box([0.0, 0.0], [1.0, 0.0], [0.030, -0.001], [0.032, 0.001])
        self.assertEqual(dist_hit, 0.0)
        dist_sep = _segment_xy_distance_to_box([0.0, 0.05], [1.0, 0.05], [0.2, -0.02], [0.3, 0.01])
        self.assertAlmostEqual(dist_sep, 0.04, places=9)

        # collision_world attached_box exact TCP-frame transform round-trip
        T_tcp = np.asarray(kin.fk(MODEL, Q_TRUE))
        world = ik_plan.collision_world(
            dict(obstacles=[], held_objects=dict(right=dict(min_m=[0.18, -0.02, 0.08], max_m=[0.22, 0.02, 0.12], margin_m=0.005))),
            "right",
            T_tcp,
        )
        box = world["attached_box"]
        self.assertIsNotNone(box)
        np.testing.assert_allclose(box["half_extents_m"], [0.025, 0.025, 0.025], atol=1e-9)
        centre_base = T_tcp[:3, :3] @ np.asarray(box["centre_tcp_m"]) + T_tcp[:3, 3]
        np.testing.assert_allclose(centre_base, [0.20, 0.0, 0.10], atol=1e-9)
        np.testing.assert_allclose(T_tcp[:3, :3] @ np.asarray(box["rotation_tcp"]), np.eye(3), atol=1e-9)


class PinkDeploymentAndMotionTimingTests(unittest.TestCase):
    def test_pink_solver_deployment_validation_and_local_profile_tags(self):
        pink_cfg = validate_ik_service(dict(solver="pink"))
        self.assertEqual(
            pink_cfg,
            dict(solver="pink", url=None, timeout_s=10.0, num_solutions=8, table_top_z_m=None, path_check_step_rad=0.01),
        )
        explicit = validate_ik_service(
            dict(solver="pink", url=None, timeout_s=5.0, num_solutions=6, table_top_z_m=-0.015, path_check_step_rad=0.008)
        )
        self.assertEqual((explicit["solver"], explicit["url"], explicit["num_solutions"], explicit["table_top_z_m"]),
                         ("pink", None, 6, -0.015))
        with_loopback = validate_ik_service(dict(solver="pink", url="http://127.0.0.1:8341/"))
        self.assertEqual(with_loopback["url"], "http://127.0.0.1:8341")
        for bad_url in ("http://10.0.0.5:8341", "https://127.0.0.1:8341", 8341, ""):
            with self.assertRaisesRegex(ValueError, "so101_deployment_invalid:ik_service.url"):
                validate_ik_service(dict(solver="pink", url=bad_url))
        profile = embodiment_profile(deployment(dict(solver="pink")))
        self.assertTrue({"approach_roll", "path_check", "collision_world", "obstacle_object_ids"} <= profile.supports)

    def test_motion_acceleration_and_jerk_validation_and_public_limits(self):
        from ni_rua.robot.backends.so101.deployment import FIELDS, public_limits

        default_dep = deployment()
        self.assertIsNone(default_dep["motion"]["max_acceleration_rad_s2"])
        self.assertIsNone(default_dep["motion"]["max_jerk_rad_s3"])
        default_pub = public_limits(default_dep)
        self.assertNotIn("max_acceleration_rad_s2", default_pub)
        self.assertNotIn("max_jerk_rad_s3", default_pub)
        self.assertNotIn("acceleration", default_pub["units"])
        self.assertNotIn("jerk", default_pub["units"])

        for partial in (dict(max_acceleration_rad_s2=12.0), dict(max_jerk_rad_s3=80.0)):
            raw = document()
            raw["motion"].update(partial)
            with self.assertRaisesRegex(ValueError, "must both be set or both null"):
                validate_deployment(raw, base=CONFIGS)

        for bad_acc in (0.0, -1.0, 100.1, float("nan"), True, "12.0"):
            raw = document()
            raw["motion"].update(max_acceleration_rad_s2=bad_acc, max_jerk_rad_s3=80.0)
            with self.assertRaises(ValueError):
                validate_deployment(raw, base=CONFIGS)
        for bad_jerk in (0.0, -5.0, 1000.1, float("inf"), True, "80.0"):
            raw = document()
            raw["motion"].update(max_acceleration_rad_s2=12.0, max_jerk_rad_s3=bad_jerk)
            with self.assertRaises(ValueError):
                validate_deployment(raw, base=CONFIGS)

        raw = document()
        raw["motion"].update(max_acceleration_rad_s2=12.0, max_jerk_rad_s3=80.0)
        configured = validate_deployment(raw, base=CONFIGS)
        pub = public_limits(configured)
        self.assertEqual(pub["max_acceleration_rad_s2"], 12.0)
        self.assertEqual(pub["max_jerk_rad_s3"], 80.0)
        self.assertEqual(pub["units"]["acceleration"], "rad/s^2")
        self.assertEqual(pub["units"]["jerk"], "rad/s^3")
        for key, unit in (("max_acceleration_rad_s2", "rad/s^2"), ("max_jerk_rad_s3", "rad/s^3")):
            spec = FIELDS["motion"][key]
            self.assertEqual(spec["unit"], unit)
            self.assertIn("ruckig", spec["doc"])
            self.assertIn("e.g.", spec["doc"])


class PinkLocalClientAndAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ni_rua.robot.model import so101_kinematics as kin

        cls.kin = kin
        cls.model = kin.load_model()
        urdf_limits = kin.joint_limits_rad(cls.model)
        cls.real_limits = {name: (urdf_limits[name][0] + 0.05, urdf_limits[name][1] - 0.05) for name in ARM_JOINTS}

    @staticmethod
    def _forbid_network(*args, **kwargs):
        raise AssertionError("solver='pink' must never invoke network opener")

    def test_pink_client_zero_network_solve_and_fast_open_jaw_collision_check(self):
        from ni_rua.robot.backends.so101.ik_plan import matrix_to_quaternion

        settings = validate_ik_service(dict(solver="pink", table_top_z_m=-0.03, num_solutions=6, path_check_step_rad=0.01))
        pink_client = CuroboClient(settings, self.real_limits, opener=self._forbid_network)
        health = pink_client.health()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["solver"], "pink")
        self.assertEqual(pink_client.max_batch, 64)
        self.assertEqual(pink_client.max_path_samples, 4096)

        q_ref = [0.1, -0.8, 1.1, 0.7, 0.0]
        T_ref = self.kin.fk(self.model, q_ref)
        target_pose = dict(position_m=T_ref[:3, 3].tolist(), quaternion_xyzw=matrix_to_quaternion(T_ref[:3, :3]))
        (candidates,) = pink_client.solve([target_pose], [0.0, -1.0, 1.0, 0.6, 0.0], [1, 1, 1, 1, 1, 1], 5e-4, 5e-3)
        self.assertTrue(any(c["success"] and c["feasible"] and c["position_error_m"] <= 5e-4 for c in candidates))

        # Warm up cached convex-hull engine once, then assert 25-sample check_path runs in < 10 ms CPU time
        import time

        seg_25 = dict(
            start_q_rad=[0.0, -1.1, 1.1, 0.6, 0.0],
            end_q_rad=[0.24, -0.95, 1.0, 0.6, 0.1],
            start_jaw_rad=1.2,
            end_jaw_rad=1.2,
            attached=False,
        )
        pink_client.check_path([seg_25])
        best_cpu_ms = float("inf")
        timed = None
        for _ in range(5):
            t0 = time.process_time()
            cand = pink_client.check_path([seg_25])
            best_cpu_ms = min(best_cpu_ms, (time.process_time() - t0) * 1000.0)
            if timed is None or cand["check_ms"] < timed["check_ms"]:
                timed = cand
        self.assertGreaterEqual(timed["samples"], 25)
        self.assertLess(best_cpu_ms, 10.0)
        self.assertGreater(timed["check_ms"], 0.0)
        self.assertEqual(timed["segments"][0]["colliding_samples"], 0)

        # Open-jaw object-between-fingers is clear, whereas closed-jaw or fixed-finger overlap collides!
        # Construct a pose and transform a small cube inside the open jaw gap (TCP frame x in [-0.024, -0.012],
        # y in [-0.006, 0.006], z in [-0.008, 0.002]) into so101_base:
        q_grasp = [0.0, -0.9, 1.0, 1.47, 0.0]
        T_grasp = self.kin.fk(self.model, q_grasp)
        # TCP origin is on the fixed jaw's inner face (x=0), jaw opens toward -X.
        p_between = T_grasp[:3, :3] @ [-0.020, 0.0, -0.004] + T_grasp[:3, 3]
        between_box = dict(
            id="target_between_jaws",
            min_m=(p_between - [0.005, 0.005, 0.005]).tolist(),
            max_m=(p_between + [0.005, 0.005, 0.005]).tolist(),
        )
        static_open = dict(start_q_rad=q_grasp, end_q_rad=q_grasp, start_jaw_rad=1.2, end_jaw_rad=1.2, attached=False)
        clear_open = pink_client.check_path([static_open], world=dict(obstacles=[between_box], attached_spheres=[]))
        self.assertEqual(clear_open["segments"][0]["colliding_samples"], 0, clear_open)

        # Closing the jaw sweeps the moving finger across x = -0.020 m -> collides with the object!
        closing_seg = dict(start_q_rad=q_grasp, end_q_rad=q_grasp, start_jaw_rad=1.2, end_jaw_rad=-0.15, attached=False)
        hit_closed = pink_client.check_path([closing_seg], world=dict(obstacles=[between_box], attached_spheres=[]))
        self.assertGreater(hit_closed["segments"][0]["colliding_samples"], 0, hit_closed)
        self.assertIsNotNone(hit_closed["segments"][0]["first_collision"])

        # Raising table_top_z_m above the TCP height detects table collision cleanly:
        high_table_client = CuroboClient(
            validate_ik_service(dict(solver="pink", table_top_z_m=float(T_grasp[2, 3]) + 0.03)),
            self.real_limits,
            opener=self._forbid_network,
        )
        table_hit = high_table_client.check_path([static_open])
        self.assertGreater(table_hit["segments"][0]["colliding_samples"], 0)
        first_world = table_hit["segments"][0]["first_collision"]["world"]
        self.assertTrue(any("table" in entry["obstacles"] for entry in first_world), first_world)

    def test_pink_client_attached_box_only_default_rotation_and_partial_orientation_success(self):
        from scipy.spatial.transform import Rotation
        from ni_rua.robot.backends.so101.ik_plan import matrix_to_quaternion
        from ni_rua.robot.model.pinocchio_backend import so101_collision_engine

        pink_client = CuroboClient(
            validate_ik_service(dict(solver="pink", table_top_z_m=0.0, num_solutions=6, path_check_step_rad=0.01)),
            self.real_limits,
            opener=self._forbid_network,
        )
        q_ready = [0.0, -0.95, 0.9, 1.0, 0.0]
        seg_ready = dict(start_q_rad=q_ready, end_q_rad=q_ready, start_jaw_rad=0.2, end_jaw_rad=0.2, attached=True)

        # 1 & 2. attached_box-only collision world (with rotation_tcp omitted, None, or I_3, and with or without
        # attached_spheres=[]) MUST detect table collision identically to direct SO101CoalCollisionEngine!
        box_no_rot = dict(centre_tcp_m=[0.0, 0.0, 0.14], half_extents_m=[0.04, 0.04, 0.04])
        normalized = CuroboClient._local_world(dict(attached_box=box_no_rot))
        self.assertNotIn("attached_spheres", normalized)
        self.assertEqual(normalized["attached_box"]["rotation_tcp"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        direct_hit = so101_collision_engine().check_path(
            [seg_ready],
            table_top_z_m=0.0,
            world=dict(attached_box=dict(box_no_rot, rotation_tcp=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])),
        )
        self.assertGreater(direct_hit["segments"][0]["colliding_samples"], 0, direct_hit)

        for world_variant in (
            dict(attached_box=box_no_rot),
            dict(attached_box=dict(box_no_rot, rotation_tcp=None)),
            dict(attached_box=box_no_rot, attached_spheres=[]),
        ):
            client_hit = pink_client.check_path([seg_ready], world=world_variant)
            self.assertEqual(
                client_hit["segments"][0]["colliding_samples"],
                direct_hit["segments"][0]["colliding_samples"],
                world_variant,
            )
            world_hits = client_hit["segments"][0]["first_collision"]["world"]
            self.assertTrue(
                any(item["link"] == "attached_object" and "table" in item["obstacles"] for item in world_hits),
                world_hits,
            )

        # 3. Position-only ([1,1,1,0,0,0]) and roll-free ([1,1,1,1,1,0]) solve() report success=True when converged!
        T_ready = self.kin.fk(self.model, q_ready)
        # Rotate target orientation by 120 deg around base X (unreachable 6D orientation at T_ready[:3, 3], large rot_err)
        R_unreach = Rotation.from_euler("x", 2.1, degrees=False).as_matrix() @ T_ready[:3, :3]
        pos_only_pose = dict(position_m=T_ready[:3, 3].tolist(), quaternion_xyzw=matrix_to_quaternion(R_unreach))
        (pos_cands,) = pink_client.solve([pos_only_pose], q_ready, [1, 1, 1, 0, 0, 0], 5e-4, 5e-3)
        self.assertTrue(
            any(c["success"] and c["feasible"] and c["position_error_m"] <= 5e-4 and c["rotation_error_rad"] > 0.5 for c in pos_cands),
            pos_cands,
        )

        # Rotate target orientation by 1.8 rad purely around local TCP +Z (free roll axis, unreachable if wrist_roll locked or far)
        R_rolled = T_ready[:3, :3] @ Rotation.from_euler("z", 1.8, degrees=False).as_matrix()
        roll_free_pose = dict(position_m=T_ready[:3, 3].tolist(), quaternion_xyzw=matrix_to_quaternion(R_rolled))
        # Restrict wrist_roll so full 6D rotation_error_rad stays ~1.8 rad > 5e-3, while approach (+Z) converges to < 5e-3!
        tight_roll_limits = dict(self.real_limits, wrist_roll=(-0.05, 0.05))
        roll_client = CuroboClient(
            validate_ik_service(dict(solver="pink", table_top_z_m=0.0, num_solutions=4)),
            tight_roll_limits,
            opener=self._forbid_network,
        )
        (roll_cands,) = roll_client.solve([roll_free_pose], q_ready, [1, 1, 1, 1, 1, 0], 5e-4, 5e-3)
        self.assertTrue(
            any(c["success"] and c["feasible"] and c["position_error_m"] <= 5e-4 and c["rotation_error_rad"] > 1.0 for c in roll_cands),
            roll_cands,
        )

    def test_so101_adapter_preview_reach_and_execute_with_local_pink_solver_zero_ssh(self):
        import tempfile
        import threading
        from ni_rua.core.config import Config
        from ni_rua.robot.backends.so101.driver import FakeArmDriver, FakeCameraSource, FakeClock
        from ni_rua.robot.backends.so101.worker import SO101Adapter

        temp_dir = self.enterContext(tempfile.TemporaryDirectory())
        raw = document()
        raw["ik_service"] = dict(solver="pink", table_top_z_m=-0.06, num_solutions=6, path_check_step_rad=0.01)
        deploy = validate_deployment(raw, base=CONFIGS)
        for cam in ("fixed", "wrist"):
            deploy["cameras"][cam].update(width=64, height=48)
        clock = FakeClock()
        driver = FakeArmDriver(deploy, clock)
        cameras = {cam: FakeCameraSource(cam, clock=clock, width=64, height=48) for cam in ("fixed", "wrist")}
        adapter = SO101Adapter(
            deploy,
            temp_dir,
            "rgb_proprio_v1",
            driver=driver,
            cameras=cameras,
            clock=clock,
            image_settings=Config().images.to_dict(),
            start_telemetry=False,
        )
        self.addCleanup(adapter.close)
        adapter.ik_client.opener = self._forbid_network

        identity = dict(run_id="r", episode_id="e", env_instance_id="i", env_generation=0)
        ee = adapter.robot_state(adapter.read())["arms"]["right"]["ee_pose"]
        caps = adapter.capabilities()

        def move_stage(sid, dx=0.0, dz=0.0, **path_extra):
            leaf = contract.compile_action(
                "move",
                dict(
                    arm="right",
                    observation_id="obs-0",
                    position_m=[ee["position_m"][0] + dx, ee["position_m"][1], ee["position_m"][2] + dz],
                    quaternion_xyzw=ee["quaternion_xyzw"],
                    position_tolerance_m=0.005,
                    orientation_tolerance_rad=0.1,
                    **path_extra,
                ),
                identity,
                caps,
            )
            return dict(id=sid, kind="move", arm="right", action=leaf)

        def make_batch(*stages, **extra):
            return dict(
                schema_version="0.1",
                type="BatchAction",
                **identity,
                world_kind="observed",
                action_id="batch-pink",
                based_on_observation_id="obs-0",
                frame_id=caps["frame_id"],
                tcp_config_id=caps["tcp_config_id"],
                limits_profile=caps["limits_profile"],
                timeout_s=30.0,
                stages=list(stages),
                **extra,
            )

        reach_res = adapter.reach_batch(
            make_batch(
                move_stage("s1", dx=0.03),
                move_stage("s2", dx=0.04, dz=-0.02, approach_tolerance_rad=0.05, roll_tolerance_rad=math.pi),
            )
        )
        self.assertEqual(reach_res["status"], "solved", reach_res)
        self.assertEqual(reach_res["reachable_stage_count"], 2)
        self.assertTrue(reach_res["stage_results"]["s2"]["ok"])

        action = make_batch(move_stage("m1", dx=0.03), move_stage("m2", dx=0.05, dz=0.01))
        preview = adapter.preview_batch(action)
        self.assertEqual(preview["status"], "planned", preview)
        self.assertEqual(preview["path_check"]["status"], "clear", preview["path_check"])
        receipt = adapter.execute(dict(action, batch_plan_id=preview["plan_id"]), lambda _: None, threading.Event())
        self.assertEqual(receipt["status"], "completed", receipt)
        motion_rows = [k for k in receipt["planned_trace"]["row_kinds"] if k == "motion"]
        self.assertEqual(len(motion_rows), preview["planned_native_steps"])

    def test_pink_solve_ik_reraises_internal_code_bugs_instead_of_returning_seed(self):
        from unittest.mock import patch
        import pink

        model = kin.load_model()
        limits_map = {name: bounds for name, bounds in kin.joint_limits_rad(model).items() if name in kin.ARM_JOINTS}
        q_seed = np.array([0.0, -0.8, 0.8, 0.9, 0.0], dtype=np.float64)
        target = kin.fk(model, [0.2, -0.9, 0.85, 0.95, 0.1])
        chain = kin.so101_chain(model)
        # Probe with the real solver: the probe caches per process and itself re-raises an injected bug, so calling
        # it under the patch made this test depend on an earlier test having filled the cache.
        daqp_usable = generic_kin._pink_daqp_usable()

        for bug in (TypeError("bad pink signature"), KeyError("missing task"), ZeroDivisionError("div by zero")):
            with self.subTest(bug=repr(bug)), patch.object(pink, "solve_ik", side_effect=bug):
                with self.assertRaises(type(bug)):
                    solve_so101_pink_ik(target, q_seed, limits_map)
                if daqp_usable:
                    with self.assertRaises(type(bug)):
                        generic_kin.solve_ik(chain, target, q_seed)


if __name__ == "__main__":
    unittest.main()
