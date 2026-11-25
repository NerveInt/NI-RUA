"""SO-101 grasp-centre move input (``grasp_centre_m`` + ``object_width_m``; pose input tag ``jaw_grasp_centre``).

* CAD conversion (``so101_gripper_geometry``, ``so101.contract.grasp_centre_pose``): the fixed fingertip face ends
  ``object_width_m / 2 + fixed_jaw_clearance_m`` from the commanded centre on the TCP +X side for several widths, yaws
  and tilted orientations; the shift is toward the fixed jaw; the target does not depend on the jaw opening;
  ``grasp_depth_m`` shifts along the approach; the jaw-gap functions invert each other.
* Contract: form and bound errors are refused by name, a compiled leaf whose target is not the conversion of its
  request is refused, the ARX-X5 grasp fields stay refused and name the SO-101 input.
* Schema gating per profile: ARX-X5 carries none of the fields (its golden catalogue: ``test_embodiment_x5_golden``).
* Harness: compile -> preview reports the conversion in ``stage_plans``, execution in ``stage_results``, ``mode=reach``
  answers grasp-centre poses through the real SO-101 worker IK, ``robot_state`` carries the measured ``jaw_grasp``.
* ``grasp_3d_review`` warnings and the execute_if_clean downgrade; object-table ``grasp_hints`` for the input.

No model, network, GPU, robot or simulator (the fake SO-101 of ``support.embodiment_golden``).
Run: ``PYTHONPATH=.:tests mamba run -n ni-rua python -m unittest test_so101_grasp_centre -v``.
"""

NI_RUA_TEST_TIER = "fast"

# === tests/test_so101_grasp_centre.py ===

from copy import deepcopy
import dataclasses
import json
import math
import tempfile
import unittest

import numpy as np

from support.motion_fixtures import _close, _so101_harness, _so101_profile

from ni_rua.perception.external import object_table
from ni_rua.robot.backends.so101 import contract
from ni_rua.robot.backends.so101.ik_plan import matrix_to_quaternion, quaternion_to_matrix
from ni_rua.robot.model import embodiments as e
from ni_rua.robot.model import so101_gripper_geometry as cad
from ni_rua.runtime.actions.grasp_preview_3d import audit_batch_grasp_3d
from ni_rua.runtime.actions.robot_action_schema import build
from ni_rua.runtime.actions.unified_actions import _preview_downgrade_reasons, compile_request
from ni_rua.runtime.tools.registry import validator_for

FIELDS = ("grasp_centre_m", "object_width_m", "fixed_jaw_clearance_m", "grasp_depth_m", "fixed_jaw_yaw_deg")
COLUMNS = ["id", "centre_m", "top_z", "bottom_z", "length", "width", "yaw_deg", "tilt_deg", "support", "quality"]
CLOSED_RAD, OPEN_RAD = -0.174533, 1.74533  # jaw angles of goal 0 and 1 in configs/so101.example.json


def _request(**fields):
    return contract.grasp_centre_request({"grasp_centre_m": [0.24, -0.06, 0.015], "object_width_m": 0.03, **fields})


# ---- CAD conversion ---------------------------------------------------------------------------------------------
class GraspCentreGeometryTests(unittest.TestCase):
    def test_fixed_face_stands_half_width_plus_clearance_from_the_centre_for_widths_and_yaws(self):
        fixed_tip = np.asarray(cad.fingertip_reference(0.0)["fixed_tip_m"])
        for width in (0.0, 0.012, 0.03, 0.055, 0.09):
            for clearance in (0.0, 0.004):
                for yaw in (-180, -90, -35, 0, 60, 90, 180):
                    with self.subTest(width=width, clearance=clearance, yaw=yaw):
                        request = _request(object_width_m=width, fixed_jaw_clearance_m=clearance, fixed_jaw_yaw_deg=yaw)
                        target = contract.grasp_centre_pose(request)
                        R = quaternion_to_matrix(target["quaternion_xyzw"])
                        direction = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0])
                        np.testing.assert_allclose(R[:, 0], direction, atol=1e-12)        # TCP +X at the yaw
                        np.testing.assert_allclose(R[:, 2], [0, 0, -1], atol=1e-12)       # approach straight down
                        np.testing.assert_allclose(R[:, 1], [direction[1], -direction[0], 0], atol=1e-12)
                        centre = np.asarray(request["grasp_centre_m"])
                        tcp = np.asarray(target["position_m"])
                        # The fixed fingertip face (CAD point) is width/2 + clearance from the centre along the jaw line.
                        fixed_world = tcp + R @ fixed_tip
                        self.assertAlmostEqual(float((fixed_world - centre) @ direction), width / 2 + clearance, places=6)
                        # Sign: the TCP is displaced from the centre toward the fixed jaw, never toward the moving jaw.
                        self.assertGreaterEqual(float((tcp - centre) @ direction), -1e-7)   # CAD face x = 1e-8 m
                        self.assertAlmostEqual(float((tcp - centre) @ direction), width / 2 + clearance, places=6)
                        self.assertAlmostEqual(tcp[2], centre[2], places=12)              # depth 0: same height
                        # At the jaw angle whose gap is width + clearance the moving fingertip is at the far face.
                        jaw = cad.jaw_rad_for_gap(width + clearance)
                        if jaw is not None:
                            moving_world = tcp + R @ np.asarray(cad.fingertip_reference(jaw)["moving_tip_m"])
                            self.assertAlmostEqual(float((moving_world - centre) @ direction), -width / 2, places=5)

    def test_offset_is_the_cad_centring_offset_and_does_not_depend_on_the_jaw(self):
        for width in (0.01, 0.03, 0.06):
            offset = cad.grasp_centre_tcp_offset_m(width, 0.003, 0.02)
            self.assertEqual(offset, [cad.centered_grasp_tcp_offset_m(width) + 0.003, 0.0, 0.02])
        # The measured centre moves with the jaw (only the -X jaw moves); the commanded conversion has no jaw input.
        centres = [cad.grasp_centre_reference(jaw)["centre_m"][0] for jaw in (0.0, 0.3, 0.6)]
        self.assertTrue(centres[0] > centres[1] > centres[2])
        for jaw in (0.0, 0.3, 0.6):
            reference = cad.grasp_centre_reference(jaw)
            self.assertAlmostEqual(reference["centre_m"][0], -reference["gap_m"] / 2, places=6)
            # An object as wide as the present gap: the commanded offset puts the TCP back on the fixed face.
            self.assertAlmostEqual(cad.grasp_centre_tcp_offset_m(reference["gap_m"])[0] + reference["centre_m"][0], 0.0,
                                   places=6)
        with self.assertRaises(ValueError):
            cad.grasp_centre_tcp_offset_m(0.03, -0.001)
        with self.assertRaises(ValueError):
            cad.grasp_centre_tcp_offset_m(0.03, 0.0, float("nan"))

    def test_depth_and_tilted_orientations_rotate_the_offset(self):
        tilt = matrix_to_quaternion(quaternion_to_matrix(contract.straight_down_quaternion(30.0))
                                    @ quaternion_to_matrix([0.0, math.sin(0.2), 0.0, math.cos(0.2)]))
        request = _request(quaternion_xyzw=tilt, grasp_depth_m=0.02, fixed_jaw_clearance_m=0.002)
        target = contract.grasp_centre_pose(request)
        R = quaternion_to_matrix(tilt)
        np.testing.assert_allclose(np.asarray(target["position_m"]) - request["grasp_centre_m"],
                                   R @ np.array([0.017, 0.0, 0.02]), atol=1e-7)
        # Depth alone moves the TCP along the approach axis: straight down, the TCP goes 0.02 m below the centre.
        down = contract.grasp_centre_pose(_request(fixed_jaw_yaw_deg=0, grasp_depth_m=0.02, object_width_m=0.0))
        np.testing.assert_allclose(down["position_m"], [0.24, -0.06, -0.005], atol=1e-7)

    def test_jaw_gap_functions(self):
        angles = np.linspace(CLOSED_RAD, 1.7, 120)
        gaps = [cad.jaw_gap_m(angle) for angle in angles]
        self.assertTrue(np.all(np.diff(gaps) > 0))
        for gap in (0.01, 0.03, 0.06, 0.1):
            self.assertAlmostEqual(cad.jaw_gap_m(cad.jaw_rad_for_gap(gap)), gap, places=6)
        self.assertIsNone(cad.jaw_rad_for_gap(0.2))
        self.assertIsNone(cad.jaw_rad_for_gap(0.0))
        # The moving fingertip is beyond the TCP plane when nearly closed and behind it when wide open.
        self.assertLess(cad.moving_fingertip_depth_m(0.0), 0.0)
        self.assertGreater(cad.moving_fingertip_depth_m(1.2), 0.03)
        plane = cad.jaw_gap_reaching_depth_m(0.0)
        self.assertAlmostEqual(cad.moving_fingertip_depth_m(cad.jaw_rad_for_gap(plane)), 0.0, places=5)
        self.assertGreater(cad.jaw_gap_reaching_depth_m(0.02), plane)
        facts = cad.jaw_grasp_facts(OPEN_RAD)
        self.assertEqual(facts, dict(fixed_face_x_m=0.0, min_grasp_depth_m=-0.0053, max_grasp_depth_m=0.0611,
                                     tcp_plane_gap_m=0.0618, max_gap_m=0.1101))
        self.assertLess(cad.jaw_grasp_facts(0.6)["max_gap_m"], facts["max_gap_m"])     # a deployment that opens less
        geometry = cad.derive_gripper_geometry()
        self.assertEqual(facts["max_grasp_depth_m"], -geometry["pocket_floor_z_m"])
        self.assertEqual(facts["min_grasp_depth_m"], -geometry["fixed_jaw"]["fingertip_inner_point_m"][2])


# ---- contract -----------------------------------------------------------------------------------------------------
class GraspCentreContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="ni-rua-grasp-centre-")
        from support import embodiment_golden as golden
        from ni_rua.robot.backends.so101.deployment import load_deployment
        cls.deployment = load_deployment(str(golden.so101_deployment(cls.directory.name, empty_close_aperture_m=0.0051)))
        cls.profile = contract.embodiment(cls.deployment)
        cls.capabilities = dict(arms=["right"], frame_id=cls.profile.frame_id, tcp_config_id=cls.profile.tcp_config_id,
                                default_timeout_s=60.0, gripper_phases=["after_motion", "during_motion"],
                                motion_tolerance_defaults=dict(position_tolerance_m=0.01, orientation_tolerance_rad=0.15))
        cls.identity = dict(run_id="r", episode_id="e", env_instance_id="i", env_generation=0)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def compile(self, kind="move", **fields):
        return contract.compile_action(kind, dict(arm="right", observation_id="obs", **fields), self.identity,
                                       self.capabilities)

    def test_compiles_to_the_tcp_target_and_keeps_the_request(self):
        action = self.compile(grasp_centre_m=[0.2, 0.05, 0.03], object_width_m=0.03, fixed_jaw_clearance_m=0.004,
                              fixed_jaw_yaw_deg=90)
        pose = action["path"]["waypoints"][0]["right"]
        np.testing.assert_allclose(pose["position_m"], [0.2, 0.069, 0.03], atol=1e-7)   # shifted toward +y (fixed jaw)
        np.testing.assert_allclose(pose["quaternion_xyzw"], [math.sqrt(0.5), math.sqrt(0.5), 0, 0], atol=1e-12)
        self.assertEqual(action["input_grasp_centre"], dict(
            grasp_centre_m=[0.2, 0.05, 0.03], object_width_m=0.03, fixed_jaw_clearance_m=0.004, grasp_depth_m=0.0,
            quaternion_xyzw=pose["quaternion_xyzw"], fixed_jaw_yaw_deg=90.0))
        self.assertEqual(contract.validate_leaf(action, self.deployment), "right")
        # The jaw goal of the stage does not change the target: only the -X jaw moves.
        for goal in (0.0, 0.4, 1.0):
            other = self.compile(grasp_centre_m=[0.2, 0.05, 0.03], object_width_m=0.03, fixed_jaw_clearance_m=0.004,
                                 fixed_jaw_yaw_deg=90, gripper_goal_normalized=goal)
            self.assertEqual(other["path"]["waypoints"], action["path"]["waypoints"])
        # A plain TCP move carries no request.
        self.assertNotIn("input_grasp_centre", self.compile(position_m=[0.2, 0, 0.05], quaternion_xyzw=[1, 0, 0, 0]))

    def test_form_errors_are_refused_by_name(self):
        centre = dict(grasp_centre_m=[0.2, 0.0, 0.03], object_width_m=0.03)
        cases = {
            "give grasp_centre_m or position_m": dict(centre, position_m=[0.2, 0, 0.03], quaternion_xyzw=[1, 0, 0, 0]),
            "exactly one orientation": dict(centre),
            "exactly one orientation: ": dict(centre, quaternion_xyzw=[1, 0, 0, 0], fixed_jaw_yaw_deg=0),
            "object_width_m (m, the object's extent along the jaw line) is required": dict(
                grasp_centre_m=[0.2, 0, 0.03], fixed_jaw_yaw_deg=0),
            "object_width_m, fixed_jaw_yaw_deg require grasp_centre_m": dict(
                position_m=[0.2, 0, 0.03], quaternion_xyzw=[1, 0, 0, 0], object_width_m=0.03, fixed_jaw_yaw_deg=0),
            "grasp_centre_m must be three finite numbers": dict(centre, grasp_centre_m=[0.2, 0.0], fixed_jaw_yaw_deg=0),
            "object_width_m must be a finite number in metres, >= 0": dict(centre, object_width_m=-0.01, fixed_jaw_yaw_deg=0),
            "fixed_jaw_clearance_m must be a finite number": dict(centre, fixed_jaw_clearance_m=-0.001, fixed_jaw_yaw_deg=0),
            "fixed_jaw_yaw_deg must be in [-180, 180]": dict(centre, fixed_jaw_yaw_deg=270),
            "quaternion_xyzw must be a unit quaternion": dict(centre, quaternion_xyzw=[1, 1, 0, 0]),
        }
        for message, fields in cases.items():
            with self.subTest(message=message):
                with self.assertRaises(ValueError) as caught:
                    self.compile(**fields)
                self.assertTrue(str(caught.exception).startswith("invalid_grasp_centre: "), caught.exception)
                self.assertIn(message, str(caught.exception))
        with self.assertRaisesRegex(ValueError, "invalid_grasp_centre: .*belong to a move stage"):
            self.compile("gripper", goal_normalized=0.0, object_width_m=0.03)

    def test_cad_bounds_are_refused_never_clamped(self):
        bounds = self.profile.gripper.jaw_grasp
        ok = self.compile(grasp_centre_m=[0.2, 0, 0.05], object_width_m=bounds.max_gap_m - 0.004,
                          fixed_jaw_clearance_m=0.004, fixed_jaw_yaw_deg=0)
        contract.validate_leaf(ok, self.deployment)
        wide = self.compile(grasp_centre_m=[0.2, 0, 0.05], object_width_m=bounds.max_gap_m - 0.004,
                            fixed_jaw_clearance_m=0.0041, fixed_jaw_yaw_deg=0)
        with self.assertRaisesRegex(ValueError, "exceeds the widest jaw gap 0.1101 m; refused, never clamped"):
            contract.validate_leaf(wide, self.deployment)
        for depth, accepted in ((bounds.min_grasp_depth_m, True), (bounds.max_grasp_depth_m, True),
                                (bounds.min_grasp_depth_m - 0.0001, False), (bounds.max_grasp_depth_m + 0.0001, False)):
            leaf = self.compile(grasp_centre_m=[0.2, 0, 0.1], object_width_m=0.03, grasp_depth_m=depth, fixed_jaw_yaw_deg=0)
            with self.subTest(depth=depth):
                if accepted:
                    contract.validate_leaf(leaf, self.deployment)
                else:
                    with self.assertRaisesRegex(ValueError, r"grasp_depth_m must be in \[-0.0053, 0.0611\] m"):
                        contract.validate_leaf(leaf, self.deployment)
        # The converted TCP target, not the centre, must lie in the workspace box; the refusal says the target was
        # converted from grasp_centre_m and names the request (the centre itself is inside the box here).
        outside = self.compile(grasp_centre_m=[0.2, 0.44, 0.05], object_width_m=0.03, fixed_jaw_yaw_deg=90)
        with self.assertRaises(ValueError) as caught:
            contract.validate_leaf(outside, self.deployment)
        message = str(caught.exception)
        self.assertTrue(message.startswith("out_of_workspace: target [0.2, 0.45"), message)
        for part in ("gripper_frame_link target converted from grasp_centre_m [0.2, 0.44, 0.05]",
                     "object_width_m 0.03", "fixed_jaw_clearance_m 0", "grasp_depth_m 0"):
            self.assertIn(part, message)
        # The request's own bounds are checked before the workspace box: a too-wide request outside the box is refused
        # for its width, naming the field the model set.
        both = self.compile(grasp_centre_m=[0.2, 0.44, 0.05], object_width_m=bounds.max_gap_m, fixed_jaw_clearance_m=0.002,
                            fixed_jaw_yaw_deg=90)
        with self.assertRaisesRegex(ValueError, "exceeds the widest jaw gap"):
            contract.validate_leaf(both, self.deployment)
        # A plain TCP target outside the box keeps the plain message.
        plain = self.compile(position_m=[0.2, 0.46, 0.05], quaternion_xyzw=[1, 0, 0, 0])
        with self.assertRaises(ValueError) as caught:
            contract.validate_leaf(plain, self.deployment)
        self.assertNotIn("grasp_centre_m", str(caught.exception))

    def test_a_leaf_whose_target_is_not_the_conversion_of_its_request_is_refused(self):
        action = self.compile(grasp_centre_m=[0.2, 0.05, 0.03], object_width_m=0.03, fixed_jaw_yaw_deg=0)
        moved = deepcopy(action)
        moved["path"]["waypoints"][0]["right"]["position_m"][0] += 0.002
        with self.assertRaisesRegex(ValueError, "the move target is not the conversion of input_grasp_centre"):
            contract.validate_leaf(moved, self.deployment)
        wider = deepcopy(action)
        wider["input_grasp_centre"]["object_width_m"] = 0.05
        with self.assertRaisesRegex(ValueError, "not the conversion"):
            contract.validate_leaf(wider, self.deployment)
        extra = deepcopy(action)
        extra["input_grasp_centre"]["jaw_rad"] = 0.3
        with self.assertRaisesRegex(ValueError, "carries only the grasp-centre fields"):
            contract.validate_leaf(extra, self.deployment)
        # A JSON round trip (the worker pipe) keeps a compiled leaf valid.
        contract.validate_leaf(json.loads(json.dumps(action)), self.deployment)

    def test_x5_grasp_fields_stay_refused_and_name_the_so101_input(self):
        with self.assertRaisesRegex(ValueError, "unsupported_capability: angles_deg .*grasp_centre_m \\+ object_width_m"):
            self.compile(grasp_position_m=[0.2, 0, 0.05], angles_deg=[0, 0, 0])
        for field, alternative in (("contact_region", "grasp_depth_m"), ("contact_depth_m", "grasp_depth_m"),
                                   ("via_grasp_poses", "successive move stages")):
            with self.assertRaisesRegex(ValueError, f"unsupported_capability: {field} \\(on SO-101 .*{alternative}"):
                self.compile(position_m=[0.2, 0, 0.05], quaternion_xyzw=[1, 0, 0, 0], **{field: "pad_center"})
        with self.assertRaisesRegex(ValueError, "^unsupported_capability: dual_arm_progression$"):
            self.compile(position_m=[0.2, 0, 0.05], quaternion_xyzw=[1, 0, 0, 0], dual_arm_progression="x")

    def test_report_names_the_target_offset_and_jaw_goals(self):
        request = _request(fixed_jaw_clearance_m=0.004, fixed_jaw_yaw_deg=90)
        report = contract.grasp_centre_report(request, self.deployment,
                                              planned_tcp=([0.24, -0.041, 0.015], request["quaternion_xyzw"]))
        self.assertEqual(report["tcp_target_position_m"], [0.24, -0.041, 0.015])
        self.assertEqual(report["tcp_offset_tcp_m"], [0.019, 0.0, 0.0])
        self.assertEqual(report["fixed_jaw_direction"], [0.0, 1.0, 0.0])
        self.assertEqual(report["approach_direction"], [0.0, 0.0, -1.0])
        self.assertEqual(report["object_shift_on_closing_m"], 0.004)
        self.assertEqual(report["planned"], dict(grasp_centre_m=[0.24, -0.06, 0.015], centre_error_m=0.0))
        jaw = report["jaw"]
        self.assertEqual(jaw["straddle_gap_m"], 0.034)
        span = OPEN_RAD - CLOSED_RAD
        self.assertAlmostEqual(jaw["straddle_goal_normalized"], (cad.jaw_rad_for_gap(0.034) - CLOSED_RAD) / span, places=3)
        self.assertAlmostEqual(jaw["closed_on_object_goal_normalized"], (cad.jaw_rad_for_gap(0.03) - CLOSED_RAD) / span,
                               places=3)
        self.assertLess(jaw["closed_on_object_goal_normalized"], jaw["straddle_goal_normalized"])
        self.assertAlmostEqual(jaw["moving_fingertip_depth_m"], cad.moving_fingertip_depth_m(jaw["closed_on_object_jaw_rad"]),
                               places=4)
        # An accepted orientation residual swings the centre about the TCP: reported in metres.
        rolled = matrix_to_quaternion(quaternion_to_matrix(request["quaternion_xyzw"])
                                      @ quaternion_to_matrix([0.0, 0.0, math.sin(0.1), math.cos(0.1)]))
        swung = contract.grasp_centre_report(request, self.deployment, planned_tcp=([0.24, -0.041, 0.015], rolled))
        self.assertAlmostEqual(swung["planned"]["centre_error_m"], 2 * 0.019 * math.sin(0.1), places=4)
        compact = contract.grasp_centre_report(request, self.deployment, compact=True)
        self.assertEqual(set(compact), {"grasp_centre_m", "object_width_m", "fixed_jaw_clearance_m", "grasp_depth_m",
                                        "fixed_jaw_yaw_deg", "tcp_target_position_m"})
        # Zero width: no jaw angle closes to it (the empty close keeps a gap); the goals are null, not invented.
        thin = contract.grasp_centre_report(_request(object_width_m=0.0, fixed_jaw_yaw_deg=0), self.deployment)
        self.assertIsNone(thin["jaw"]["closed_on_object_goal_normalized"])
        self.assertIsNone(thin["jaw"]["moving_fingertip_depth_m"])

    def test_motion_instructions_explain_the_input_with_units_defaults_and_one_example(self):
        text = contract.actor_motion_instructions(self.deployment)
        sentence = contract._grasp_sentence(self.deployment, self.profile)
        self.assertIn(sentence, text)
        # The conversion, units, defaults and bounds are the field help's (one full explanation, checked in
        # test_so101_schema_publishes_the_fields_with_units_bounds_defaults_and_example); the motion text points to
        # it and keeps what the field help does not carry: the worked example, the reports and jaw_grasp.
        for present in ("FIXED jaw", "grasp_centre_m (where the object's centre sits between the jaws) + object_width_m",
                        "quaternion_xyzw or fixed_jaw_yaw_deg instead of position_m",
                        "field help of grasp_centre_m, object_width_m, fixed_jaw_clearance_m, grasp_depth_m and "
                        "fixed_jaw_yaw_deg is the contract", "an accepted orientation residual moves the grasp centre",
                        "stage_plans[<stage>].grasp_centre", "robot_state.arms.right.jaw_grasp", "grasp_3d_review"):
            self.assertIn(present, sentence)
        for repeated in ("R(orientation) * [object_width_m/2", "may not exceed 0.1101 m", "range [-0.0053, 0.0611]",
                         "[cos(yaw/2), sin(yaw/2), 0, 0]"):
            self.assertNotIn(repeated, sentence)
        self.assertEqual(sentence.count("Example: "), 1)
        # The example's numbers are the conversion of the example's request.
        report = contract.grasp_centre_report(_request(fixed_jaw_clearance_m=0.004, fixed_jaw_yaw_deg=90), self.deployment)
        self.assertIn("targets gripper_frame_link at [0.24, -0.041, 0.015]", sentence)
        self.assertIn(f"jaw goal above {report['jaw']['straddle_goal_normalized']:g}", sentence)
        self.assertIn(f"at goal {report['jaw']['closed_on_object_goal_normalized']:g}", sentence)
        self.assertIsNone(__import__("re").search(r"\bleft\b", sentence))
        without = dataclasses.replace(self.profile, pose_inputs=frozenset({"tcp_quaternion"}),
                                      gripper=dataclasses.replace(self.profile.gripper, jaw_grasp=None))
        self.assertEqual(contract._grasp_sentence(self.deployment, without), "")


# ---- profile and schema gating ------------------------------------------------------------------------------------
class GraspCentreProfileSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.so101_profile = _so101_profile()
        cls.x5, cls.so101 = build(None), build(cls.so101_profile)

    def test_profile_tag_and_cad_bounds_go_together(self):
        profile = self.so101_profile
        self.assertIn("jaw_grasp_centre", profile.pose_inputs)
        self.assertEqual(dataclasses.asdict(profile.gripper.jaw_grasp), cad.jaw_grasp_facts(OPEN_RAD))
        self.assertNotIn("jaw_grasp_centre", e.ARX_X5_PROFILE.pose_inputs)
        self.assertIsNone(e.ARX_X5_PROFILE.gripper.jaw_grasp)
        with self.assertRaisesRegex(ValueError, "invalid_embodiment_profile:pose_inputs"):
            dataclasses.replace(profile, pose_inputs=frozenset({"tcp_quaternion"}))
        with self.assertRaisesRegex(ValueError, "invalid_embodiment_profile:pose_inputs"):
            dataclasses.replace(e.ARX_X5_PROFILE, pose_inputs=e.ARX_X5_PROFILE.pose_inputs | {"jaw_grasp_centre"})
        # Archive record: SO-101 round-trips with jaw_grasp; a record without the key (ARX-X5, older archives) restores.
        record = json.loads(json.dumps(e.profile_record(profile)))
        self.assertEqual(record["profile"]["gripper"]["jaw_grasp"]["max_gap_m"], 0.1101)
        self.assertEqual(e.profile_from_record(record), profile)
        x5_record = json.loads(json.dumps(e.profile_record(e.ARX_X5_PROFILE)))
        self.assertNotIn("jaw_grasp", x5_record["profile"]["gripper"])
        self.assertEqual(e.profile_from_record(x5_record), e.ARX_X5_PROFILE)
        old = deepcopy(record)
        del old["profile"]["gripper"]["jaw_grasp"]
        old["profile"]["pose_inputs"] = ["tcp_quaternion"]
        self.assertIsNone(e.profile_from_record(old).gripper.jaw_grasp)

    def test_x5_schema_has_none_of_the_fields(self):
        text = json.dumps(self.x5.tool)
        for field in (*FIELDS, "grasp_centre", "jaw_grasp"):
            self.assertNotIn(field, text)
            self.assertNotIn(field, self.x5.stage_schemas["move"]["properties"])
        stage = dict(id="a", kind="move", arm="left", grasp_centre_m=[0, 0, 0.8], object_width_m=0.03,
                     quaternion_xyzw=[1, 0, 0, 0])
        self.assertFalse(validator_for(self.x5.stage_schemas["move"]).is_valid(stage))

    def test_so101_schema_publishes_the_fields_with_units_bounds_defaults_and_example(self):
        move = self.so101.stage_schemas["move"]["properties"]
        bounds = self.so101_profile.gripper.jaw_grasp
        self.assertEqual({key: move["object_width_m"][key] for key in ("minimum", "maximum")},
                         dict(minimum=0, maximum=bounds.max_gap_m))
        self.assertEqual({key: move["grasp_depth_m"][key] for key in ("minimum", "maximum")},
                         dict(minimum=bounds.min_grasp_depth_m, maximum=bounds.max_grasp_depth_m))
        self.assertEqual({key: move["fixed_jaw_yaw_deg"][key] for key in ("minimum", "maximum")},
                         dict(minimum=-180, maximum=180))
        for field, words in (("grasp_centre_m", ("so101_base metres", "Example: {", "stage_plans[<stage>].grasp_centre")),
                             ("object_width_m", ("required", "metres", "[0, 0.1101]", "no default")),
                             ("fixed_jaw_clearance_m", ("metres", "default 0")),
                             ("grasp_depth_m", ("metres [-0.0053, 0.0611]", "default 0", "0.0618 m")),
                             ("fixed_jaw_yaw_deg", ("degrees [-180, 180]", "[cos(yaw/2), sin(yaw/2), 0, 0]"))):
            for word in words:
                with self.subTest(field=field, word=word):
                    self.assertIn(word, move[field]["description"])
            self.assertIn(field, self.so101.stage["properties"])
        description = self.so101.tool["description"]
        self.assertIn("move may instead give grasp_centre_m", description)
        self.assertIn("runs grasp_3d_review", description)
        # The X5 grasp-angle phrase is not listed as absent: this robot has its own grasp-centre input.
        self.assertNotIn("a grasp centre plus top-down angles", description)
        self.assertNotIn("contact-section selection", description)
        # The example in the field help is admitted by the schema it documents and compiles.
        example = json.loads(move["grasp_centre_m"]["description"].split("Example: ")[1].rstrip("."))
        self.assertTrue(validator_for(self.so101.stage_schemas["move"]).is_valid(example))

    def test_so101_schema_admits_exactly_one_complete_pose(self):
        validator = validator_for(self.so101.stage_schemas["move"])
        base = dict(id="a", kind="move", arm="right")
        centre = dict(grasp_centre_m=[0.2, 0, 0.03], object_width_m=0.03)
        tcp = dict(position_m=[0.2, 0, 0.03], quaternion_xyzw=[1, 0, 0, 0])
        for fields, valid in (
                (tcp, True),
                (dict(centre, fixed_jaw_yaw_deg=0), True),
                (dict(centre, quaternion_xyzw=[1, 0, 0, 0], fixed_jaw_clearance_m=0.004, grasp_depth_m=0.01), True),
                (dict(centre), False),                                               # no orientation
                (dict(centre, quaternion_xyzw=[1, 0, 0, 0], fixed_jaw_yaw_deg=0), False),   # two orientations
                (dict(centre, position_m=[0.2, 0, 0.03], quaternion_xyzw=[1, 0, 0, 0]), False),
                (dict(grasp_centre_m=[0.2, 0, 0.03], fixed_jaw_yaw_deg=0), False),   # no width
                (dict(tcp, object_width_m=0.03), False),
                (dict(tcp, grasp_depth_m=0.01), False),
                (dict(position_m=[0.2, 0, 0.03], fixed_jaw_yaw_deg=0), False),
                (dict(centre, fixed_jaw_yaw_deg=181), False),
                (dict(centre, fixed_jaw_yaw_deg=0, object_width_m=0.12), False),
                (dict(centre, fixed_jaw_yaw_deg=0, grasp_depth_m=0.07), False)):
            with self.subTest(fields=sorted(fields)):
                self.assertEqual(validator.is_valid(dict(base, **fields)), valid)


# ---- harness: compile, preview, execute, reach, robot_state --------------------------------------------------------
class GraspCentreHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="ni-rua-grasp-centre-")
        cls.harness = _so101_harness(cls.directory.name)
        cls.harness.task_planning.required = False
        cls.arm = cls.harness.embodiment.arms[0]
        adapter = cls.harness.robot.adapter
        T = adapter.tcp(adapter.read()["q_rad"])
        # A reachable pose: the home TCP pose. An object 0.03 m wide gripped there has its centre 0.019 m (width/2 +
        # 0.004 m clearance) from the TCP toward the moving jaw.
        cls.home_position, cls.home_rotation = T[:3, 3].copy(), T[:3, :3].copy()
        cls.quaternion = matrix_to_quaternion(T[:3, :3])
        cls.centre = [float(v) for v in T[:3, 3] - T[:3, :3] @ np.array([0.019, 0.0, 0.0])]

    @classmethod
    def tearDownClass(cls):
        _close(cls.harness)
        cls.directory.cleanup()

    def observation_id(self):
        return self.harness.current_observation()["observation_id"]

    def grasp_stage(self, identifier="descend", **fields):
        return {**dict(id=identifier, kind="move", arm=self.arm, grasp_centre_m=self.centre, object_width_m=0.03,
                       fixed_jaw_clearance_m=0.004, quaternion_xyzw=self.quaternion), **fields}

    def invoke(self, args, call):
        return self.harness.invoke("robot_action", deepcopy(args), dict(tool_call_id=call), role="actor")

    def set_table(self, observation_id, rows, table_z=0.0):
        self.harness._obs3d_memo = (observation_id, {"object_table": (dict(columns=COLUMNS, table_z=table_z, rows=rows),
                                                                      None, {})})

    def test_compile_request_converts_and_splits_the_after_arrival_jaw(self):
        action = compile_request(self.harness, dict(observation_id=self.observation_id(), stages=[
            self.grasp_stage(gripper_goal_normalized=0.0), self.grasp_stage("carry", fixed_jaw_clearance_m=0.0)]))
        motion, jaw, carry = action["stages"]
        self.assertEqual((motion["id"], jaw["id"], jaw["kind"]), ("descend/motion", "descend", "gripper"))
        pose = motion["action"]["path"]["waypoints"][0][self.arm]
        np.testing.assert_allclose(pose["position_m"], self.home_position, atol=1e-7)
        self.assertEqual(motion["action"]["input_grasp_centre"]["object_width_m"], 0.03)
        self.assertNotIn("input_grasp_centre", jaw["action"])
        # Without the clearance the same centre puts the TCP 0.004 m nearer along TCP +X.
        np.testing.assert_allclose(carry["action"]["path"]["waypoints"][0][self.arm]["position_m"],
                                   self.home_position - self.home_rotation @ np.array([0.004, 0, 0]), atol=1e-7)
        # Tolerances resolve into the converted move as on any move stage.
        self.assertEqual(motion["action"]["path"]["position_tolerance_m"], 0.01)

    def test_preview_reports_the_conversion_and_execution_repeats_it(self):
        observation_id = self.observation_id()
        response = self.invoke(dict(observation_id=observation_id, stages=[
            dict(id="open", kind="gripper", arm=self.arm, goal_normalized=0.5),
            self.grasp_stage(gripper_goal_normalized=0.0),
            dict(id="release", kind="gripper", arm=self.arm, goal_normalized=0.5)]), "preview")
        self.assertIsNone(response.contract_error)
        plans = response.data["stage_plans"]
        self.assertNotIn("grasp_centre", plans["open"])
        self.assertNotIn("grasp_centre", plans["descend"])              # the jaw stage of the split move
        report = plans["descend/motion"]["grasp_centre"]
        np.testing.assert_allclose(report["tcp_target_position_m"], self.home_position, atol=1e-4)
        np.testing.assert_allclose(report["grasp_centre_m"], self.centre, atol=1e-4)
        self.assertEqual(report["tcp_offset_tcp_m"], [0.019, 0.0, 0.0])
        np.testing.assert_allclose(report["fixed_jaw_direction"], self.home_rotation[:, 0], atol=1e-4)
        np.testing.assert_allclose(report["approach_direction"], self.home_rotation[:, 2], atol=1e-4)
        self.assertEqual(report["jaw"]["straddle_gap_m"], 0.034)
        self.assertEqual(report["object_shift_on_closing_m"], 0.004)
        # The planned joints are the home joints: the planned centre is the requested one.
        np.testing.assert_allclose(report["planned"]["grasp_centre_m"], self.centre, atol=2e-4)
        self.assertLess(report["planned"]["centre_error_m"], 2e-4)
        # The jaw arrives at goal 0.5 (opened in the batch), above the straddle goal: the review is clean.
        review = response.data["grasp_3d_review"]
        self.assertEqual((review["status"], review["risk_flags"]), ("aligned", []))
        self.assertEqual(review["stages"][0]["arrival_jaw_goal_normalized"], 0.5)
        self.assertEqual(_preview_downgrade_reasons(response.data), [])
        executed = self.invoke(dict(mode="execute", plan_id=response.data["plan_id"]), "execute")
        self.assertIsNone(executed.contract_error)
        results = executed.data["receipt"]["stage_results"]
        self.assertEqual(results["descend/motion"]["status"], "completed")
        self.assertEqual(set(results["descend/motion"]["grasp_centre"]),
                         {"grasp_centre_m", "object_width_m", "fixed_jaw_clearance_m", "grasp_depth_m",
                          "tcp_target_position_m"})
        self.assertNotIn("grasp_centre", results["open"])

    def test_default_reply_delivery_keeps_every_so101_value_and_states_no_x5_note(self):
        """Preview (grasp_3d_review, stage_plans), execution, mode=reach and robot_observe of the SO-101 through the
        default reply settings: nothing fails, and compared with the legacy presentation of the same typed result
        every value arrives (apart from what the settings document: six-decimal display, blocks by reference)."""
        from ni_rua.core.presentation.actor_output import ActorOutput
        from ni_rua.core.presentation.reply_compaction import dangling_same_as
        from support.reply_defaults import DEFAULT_REPLY_FEATURES, reply_losses
        harness = self.harness
        self.assertEqual({name: getattr(harness.config.features, name) for name in DEFAULT_REPLY_FEATURES},
                         DEFAULT_REPLY_FEATURES)
        plan = harness.tool_registry.static_once_plan(harness, role="actor")
        # No native step budget, joint-travel admission or grasp-contact fields on this robot: none of those notes.
        self.assertEqual(set(plan.segments) & {"budget", "steps", "grasp_contact", "joint_travel", "trajectory_link6"}, set())
        legacy = ActorOutput(harness.store, harness.actor_presentation, "lean_v1")
        correlation = dict(thread_id="t", turn_id="turn", tool_call_id="so101-reply", item_id="so101-reply")
        preview = self.invoke(dict(observation_id=self.observation_id(), stages=[
            dict(id="open", kind="gripper", arm=self.arm, goal_normalized=0.5),
            self.grasp_stage(gripper_goal_normalized=0.0)]), "reply-preview")
        results = dict(
            preview=("robot_action", preview),
            execute=("robot_action", self.invoke(dict(mode="execute", plan_id=preview.data["plan_id"]), "reply-execute")),
            reach=("robot_action", self.invoke(dict(mode="reach", observation_id=self.observation_id(), stages=[
                self.grasp_stage("here"), self.grasp_stage("far", grasp_centre_m=[0.44, 0.3, 0.3])]), "reply-reach")),
            observe=("robot_observe", harness.invoke("robot_observe", {}, dict(tool_call_id="reply-observe"), role="actor")))
        self.assertEqual(results["execute"][1].data["receipt"]["status"], "completed")
        for name, (tool, result) in results.items():
            with self.subTest(name):
                presented = harness.actor_output.present(result, tool=tool)
                delivered = harness._shape_actor_reply(presented, getattr(result, "kind", None), correlation)
                self.assertTrue(delivered["success"])
                self.assertEqual(dangling_same_as(delivered["contentItems"]), [])
                before = legacy.present(result, tool=tool)["contentItems"]
                # every image arrives (the layout moves the trajectory overlay behind the trajectory text)
                images = lambda items: sorted(item["imageUrl"] for item in items if item["type"] == "inputImage")
                self.assertEqual(images(delivered["contentItems"]), images(before))
                self.assertEqual(reply_losses(before, delivered["contentItems"], plan, tool=tool), [])
        # The execution reply is laid out (result first) although this host has no known text limit.
        executed = harness._shape_actor_reply(harness.actor_output.present(results["execute"][1], tool="robot_action"),
                                              "action", correlation)["contentItems"]
        self.assertIn('"receipt"', executed[0]["text"])
        self.assertIsNone(harness.action_reply_budget)

    def test_reach_answers_grasp_centre_poses_through_the_worker_ik(self):
        far = self.grasp_stage("far", grasp_centre_m=[0.44, 0.3, 0.3])
        yaw = dict(id="yaw", kind="move", arm=self.arm, grasp_centre_m=[0.2, 0.0, 0.05], object_width_m=0.03,
                   fixed_jaw_yaw_deg=180, tolerance_preset="unconstrained_orientation")
        response = self.invoke(dict(mode="reach", observation_id=self.observation_id(),
                                    stages=[self.grasp_stage("here"), far, yaw]), "reach")
        self.assertIsNone(response.contract_error)
        results = response.data["stage_results"]
        self.assertTrue(results["here"]["ok"])
        np.testing.assert_allclose(results["here"]["reached"]["position_m"], self.home_position, atol=2e-3)
        self.assertLess(results["here"]["grasp_centre"]["planned"]["centre_error_m"], 2e-3)
        self.assertFalse(results["far"]["ok"])
        self.assertIn("planned", results["far"]["grasp_centre"])          # the closest pose, honestly far off
        self.assertGreater(results["far"]["grasp_centre"]["planned"]["centre_error_m"], 0.05)
        # fixed_jaw_yaw_deg 180: the TCP target is 0.015 m toward -x of the centre, straight down.
        self.assertEqual(results["yaw"]["grasp_centre"]["tcp_target_position_m"], [0.185, 0.0, 0.05])
        self.assertEqual(results["yaw"]["grasp_centre"]["fixed_jaw_direction"], [-1.0, 0.0, 0.0])
        self.assertEqual(results["yaw"]["grasp_centre"]["fixed_jaw_yaw_deg"], 180.0)
        # With the orientation left free the position is reached but the centre is wherever the reached orientation
        # puts it: the report shows that distance instead of hiding it behind a met position tolerance.
        if results["yaw"]["ok"]:
            self.assertIn("centre_error_m", results["yaw"]["grasp_centre"]["planned"])
        # Fields outside the reach set stay refused on a grasp-centre stage.
        with self.assertRaisesRegex(ValueError, "robot_action_reach_accepts_move_poses_only"):
            self.invoke(dict(mode="reach", observation_id=self.observation_id(),
                             stages=[self.grasp_stage("x", gripper_goal_normalized=0.0)]), "reach-refused")

    def test_robot_state_reports_the_measured_jaw_grasp(self):
        harness = self.harness
        response = harness.invoke("robot_observe", {}, dict(tool_call_id="state"), role="actor")
        shown = next(item.value for item in response.content
                     if isinstance(getattr(item, "value", None), dict) and "robot_state" in item.value)
        state = shown["robot_state"]["arms"][self.arm]
        measured = state["jaw_grasp"]
        self.assertEqual(set(measured), {"grasp_centre_m", "gap_m", "fixed_tip_m", "moving_tip_m", "fixed_jaw_direction",
                                         "source_kind"})
        self.assertEqual(measured["source_kind"], "derived_from_measured_tcp_and_jaw_angle")
        position, R = np.asarray(state["ee_pose"]["position_m"]), quaternion_to_matrix(state["ee_pose"]["quaternion_xyzw"])
        jaw_rad = state["q_rad"][-1]          # q_rad = five arm joints, then the jaw joint "gripper"
        reference = cad.grasp_centre_reference(jaw_rad)
        np.testing.assert_allclose(measured["grasp_centre_m"], position + R @ reference["centre_m"], atol=1e-4)
        np.testing.assert_allclose(measured["fixed_tip_m"], position + R @ reference["fixed_tip_m"], atol=1e-4)
        np.testing.assert_allclose(measured["moving_tip_m"], position + R @ reference["moving_tip_m"], atol=1e-4)
        np.testing.assert_allclose(measured["fixed_jaw_direction"], R[:, 0], atol=1e-4)
        self.assertAlmostEqual(measured["gap_m"], cad.jaw_gap_m(jaw_rad), places=4)
        # Commanding the measured centre with the measured gap as the width targets the measured TCP: the measured
        # value and the command input are the same convention.
        target = contract.grasp_centre_pose(contract.grasp_centre_request(dict(
            grasp_centre_m=measured["grasp_centre_m"], object_width_m=measured["gap_m"],
            quaternion_xyzw=state["ee_pose"]["quaternion_xyzw"])))
        np.testing.assert_allclose(target["position_m"], position, atol=2e-4)
        # ARX-X5 states get no such block.
        from ni_rua.robot.model.geometry import measured_jaw_grasp
        names = [*harness.embodiment.arm_joint_names[self.arm], *harness.embodiment.jaw_joint_names[self.arm]]
        # The delivered state omits fields already sent as static conventions; restore the ones the derivation reads.
        full = dict(state, joint_names=names, ee_pose=dict(state["ee_pose"], source_kind="measurement",
                                                           tcp_config_id=harness.embodiment.tcp_config_id))
        self.assertEqual(measured_jaw_grasp(harness.embodiment, self.arm, full), measured)
        # A state without joint_names (legacy actor profile) is read in the profile's joint order.
        self.assertEqual(measured_jaw_grasp(harness.embodiment, self.arm,
                                            {key: value for key, value in full.items() if key != "joint_names"}), measured)
        self.assertIsNone(measured_jaw_grasp(e.ARX_X5_PROFILE, "right", full))
        # No measured jaw angle, or a pose that is not a measurement: nothing is derived.
        self.assertIsNone(measured_jaw_grasp(harness.embodiment, self.arm, dict(full, joint_names=names[:-1],
                                                                                 q_rad=state["q_rad"][:-1])))
        self.assertIsNone(measured_jaw_grasp(harness.embodiment, self.arm,
                                             dict(full, ee_pose=dict(full["ee_pose"], source_kind="command"))))

    def test_review_warns_when_the_jaw_arrives_closed_and_execute_if_clean_does_not_execute(self):
        observation_id = self.observation_id()
        args = dict(mode="execute_if_clean", observation_id=observation_id, stages=[
            dict(id="shut", kind="gripper", arm=self.arm, goal_normalized=0.1),
            self.grasp_stage(gripper_goal_normalized=0.0, min_aperture_m=0.04)])
        response = self.invoke(args, "if-clean")
        data = response.data
        self.assertNotIn("receipt", data)
        self.assertTrue(data["downgraded_from_execute_if_clean"])
        self.assertIn("grasp_3d_review_warnings", data["downgrade_reasons"])
        review = data["grasp_3d_review"]
        self.assertEqual(review["status"], "warnings_detected")
        self.assertEqual(review["risk_flags"], ["warn_jaw_not_open_enough", "warn_guard_exceeds_object_width"])
        stage = review["stages"][0]
        self.assertEqual((stage["stage_id"], stage["arrival_jaw_goal_normalized"]), ("descend/motion", 0.1))
        self.assertIn("arrives with jaw goal 0.1, below the", review["warnings"][0])
        self.assertIn("34.0 mm", review["warnings"][0])
        self.assertIn("min_aperture_m=40.0 mm exceeds object_width_m=30.0 mm", review["warnings"][1])
        self.assertEqual(self.harness.current_observation()["observation_id"], observation_id)   # nothing moved

    def review(self, stages, rows=None, **batch):
        observation_id = self.observation_id()
        self.harness._obs3d_memo = None
        if rows is not None:
            self.set_table(observation_id, rows)
        action = compile_request(self.harness, dict(observation_id=observation_id, stages=stages, **batch))
        plans = self.harness.robot.preview_batch(action)["stage_plans"]
        return audit_batch_grasp_3d(self.harness, action, observation_id, stage_plans=plans)

    def test_review_jaw_checks_need_no_object_table(self):
        opened = dict(id="open", kind="gripper", arm=self.arm, goal_normalized=0.6)
        # Closing only to a goal above the one that meets the object: it would not grip.
        review, targets = self.review([opened, self.grasp_stage(gripper_goal_normalized=0.3)])
        self.assertEqual((review["risk_flags"], targets), (["warn_closing_goal_above_object_width"], []))
        self.assertIsNone(review["stages"][0]["matched_object_id"])
        # A carry: the jaw opens after arrival, so the stage is no grasp and is not audited.
        self.assertEqual(self.review([self.grasp_stage(gripper_goal_normalized=1.0)]), (None, []))
        self.assertEqual(self.review([self.grasp_stage()]), (None, []))
        # A plain TCP move followed by a close is not a grasp-centre stage.
        tcp = dict(id="tcp", kind="move", arm=self.arm, position_m=[float(v) for v in self.home_position],
                   quaternion_xyzw=self.quaternion, gripper_goal_normalized=0.0)
        self.assertEqual(self.review([opened, tcp]), (None, []))
        # Free orientation: the planned centre may miss by more than the stage's position tolerance.
        swung, _ = self.review([opened, self.grasp_stage(
            quaternion_xyzw=matrix_to_quaternion(self.home_rotation @ quaternion_to_matrix(
                [0.0, 0.0, math.sin(0.4), math.cos(0.4)])), grasp_centre_m=self.centre,
            tolerance_preset="unconstrained_orientation", position_tolerance_m=0.005, gripper_goal_normalized=0.0)])
        stage = swung["stages"][0]
        self.assertEqual(stage["centre_check"] == "warn_planned_centre_outside_position_tolerance",
                         stage["planned_centre_error_m"] > 0.005)

    def test_review_against_the_object_table(self):
        opened = dict(id="open", kind="gripper", arm=self.arm, goal_normalized=0.6)
        grasp = self.grasp_stage(gripper_goal_normalized=0.0)
        direction = self.home_rotation[:, 0]
        centre = np.asarray(self.centre)

        def row(**overrides):
            values = dict(id="o1", centre_m=self.centre, top_z=self.centre[2] + 0.03, bottom_z=self.centre[2] - 0.03,
                          length=0.012, width=0.012, yaw_deg=0.0, tilt_deg=None, support="table", quality="ok")
            values.update(overrides)
            return [values[column] for column in COLUMNS]
        # A small object around the commanded centre: nothing to report. (The home pose is tilted, so the box extent
        # on the jaw line includes part of the height; the row is chosen so the declared width covers it.)
        from ni_rua.runtime.actions.grasp_preview_3d import _box_extent_m
        small = row()
        extent = _box_extent_m([float(v) for v in direction], dict(zip(COLUMNS, small)))
        clean, targets = self.review([opened, self.grasp_stage(object_width_m=round(extent, 4), gripper_goal_normalized=0.0)],
                                     [small])
        self.assertEqual((clean["status"], clean["risk_flags"]), ("aligned", []))
        self.assertEqual(clean["stages"][0]["matched_object_id"], "o1")
        self.assertAlmostEqual(clean["stages"][0]["object_extent_on_jaw_line_m"], extent, places=4)
        self.assertAlmostEqual(clean["stages"][0]["fixed_jaw_gap_to_object_m"], 0.004, places=3)
        self.assertEqual(targets[0]["id"], "o1")
        # The object is wider along the jaw line than declared: the fixed fingertip face lands inside it.
        review, _ = self.review([opened, grasp], [row(length=0.06, width=0.06)])
        self.assertIn("warn_fixed_jaw_inside_object", review["risk_flags"])
        self.assertLess(review["stages"][0]["fixed_jaw_gap_to_object_m"], 0)
        self.assertIn("inside o1 along the jaw line", " ".join(review["warnings"]))
        # The commanded centre is above the object's top / below its bottom.
        review, _ = self.review([opened, grasp], [row(top_z=self.centre[2] - 0.005, bottom_z=self.centre[2] - 0.03)])
        self.assertIn("warn_centre_above_object_top", review["risk_flags"])
        review, _ = self.review([opened, grasp], [row(top_z=self.centre[2] + 0.05, bottom_z=self.centre[2] + 0.004)])
        self.assertIn("warn_centre_below_object_bottom", review["risk_flags"])
        # The fingertips reach below the support under the object.
        review, _ = self.review([opened, grasp], [row(bottom_z=self.centre[2] - 0.001)])
        self.assertEqual(review["stages"][0]["height_check"], "warn_fingertip_below_support")
        # An elongated lying object whose long axis is along the jaw line.
        jaw_yaw = math.degrees(math.atan2(direction[1], direction[0]))
        review, _ = self.review([opened, grasp], [row(length=0.09, width=0.02, yaw_deg=round((jaw_yaw + 90) % 180 - 90, 1),
                                                    top_z=self.centre[2] + 0.01, bottom_z=self.centre[2] - 0.01,
                                                    tilt_deg=88.0)])
        self.assertEqual(review["stages"][0]["axis_alignment"], "warn_along_long_axis")
        self.assertLess(review["stages"][0]["jaw_line_to_long_axis_deg"], 1.0)
        # A row far from the commanded centre is not matched: only the jaw checks remain.
        review, targets = self.review([opened, grasp], [row(centre_m=[float(v) for v in centre + [0.0, 0.2, 0.0]])])
        self.assertEqual((review["stages"][0]["matched_object_id"], targets, review["status"]), (None, [], "aligned"))


# ---- object-table grasp hints -------------------------------------------------------------------------------------
class GraspCentreHintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = _so101_profile()
        cls.bounds = cls.profile.gripper.jaw_grasp

    def test_opening_check_uses_the_jaw_gap_range_not_the_cad_full_open(self):
        b = self.bounds
        self.assertLess(b.max_gap_m, self.profile.gripper.full_open_aperture_m)
        for width, verdict in ((0.03, "ok"), (b.tcp_plane_gap_m - 0.0101, "ok"), (b.tcp_plane_gap_m - 0.0099, "tight"),
                               (b.tcp_plane_gap_m, "tight"), (b.tcp_plane_gap_m + 0.0001, "beyond_parallel_range"),
                               (b.max_gap_m, "beyond_parallel_range"), (b.max_gap_m + 0.0001, "too_wide")):
            with self.subTest(width=width):
                self.assertEqual(object_table.jaw_opening_check(width, b), verdict)

    def test_hints_are_the_inputs_of_a_straight_down_grasp_centre_move(self):
        cube = dict(centre_m=[0.236, -0.06, -0.005], top_z_m=0.007, bottom_z_m=-0.016, length_m=0.033, width_m=0.028,
                    yaw_deg=73.0)
        hints = object_table.grasp_hints_from([cube], -0.016, self.profile)["o1"]
        self.assertEqual(hints, dict(
            grasp_centre_z_m=-0.005, opening_check="ok", elongated=False, fingertip_support_clearance_m=0.003,
            jaw_lines=[dict(fixed_jaw_yaw_deg=[163.0, -17.0], object_width_m=0.028),     # across the long axis
                       dict(fixed_jaw_yaw_deg=[73.0, -107.0], object_width_m=0.033)]))   # along it
        # Each entry compiles as given, with the row centre: reachability is a separate question (mode=reach).
        for line in hints["jaw_lines"]:
            for yaw in line["fixed_jaw_yaw_deg"]:
                request = contract.grasp_centre_request(dict(grasp_centre_m=cube["centre_m"],
                                                             object_width_m=line["object_width_m"], fixed_jaw_yaw_deg=yaw))
                target = contract.grasp_centre_pose(request)
                shift = np.asarray(target["position_m"]) - cube["centre_m"]
                self.assertAlmostEqual(float(np.linalg.norm(shift)), line["object_width_m"] / 2, places=6)
                self.assertAlmostEqual(math.degrees(math.atan2(shift[1], shift[0])) % 360, yaw % 360, places=4)
        # A line the widest gap cannot span is omitted; a row too wide on its narrow side gets no lines at all.
        plank = dict(cube, length_m=0.2, width_m=0.04, yaw_deg=0.0)
        self.assertEqual(object_table.grasp_hints_from([plank], -0.016, self.profile)["o1"]["jaw_lines"],
                         [dict(fixed_jaw_yaw_deg=[90.0, -90.0], object_width_m=0.04)])
        slab = dict(cube, length_m=0.2, width_m=0.12)
        self.assertEqual(object_table.grasp_hints_from([slab], -0.016, self.profile)["o1"],
                         dict(opening_check="too_wide", elongated=False))
        self.assertEqual(object_table.height_hint_field(self.profile), "grasp_centre_z_m")
        self.assertEqual(object_table.height_hint_field(e.ARX_X5_PROFILE), "pad_z_m")

    def test_continuation_override_switches_the_pose_form(self):
        from ni_rua.runtime.actions.plan_continuation import apply_stage_override
        stage = dict(id="a", kind="move", arm="right", grasp_centre_m=[0.2, 0, 0.03], object_width_m=0.03,
                     fixed_jaw_yaw_deg=0)
        tcp = apply_stage_override(deepcopy(stage), dict(position_m=[0.2, 0, 0.05], quaternion_xyzw=[1, 0, 0, 0]))
        self.assertEqual(set(tcp), {"id", "kind", "arm", "position_m", "quaternion_xyzw"})
        moved = apply_stage_override(deepcopy(stage), dict(target_pose=[0.21, 0.0, 0.03]))
        self.assertEqual((moved["grasp_centre_m"], moved["fixed_jaw_yaw_deg"]), ([0.21, 0.0, 0.03], 0))
        turned = apply_stage_override(deepcopy(stage), dict(quaternion_xyzw=[0, 1, 0, 0]))
        self.assertNotIn("fixed_jaw_yaw_deg", turned)
        back = apply_stage_override(dict(id="a", kind="move", arm="right", position_m=[0.2, 0, 0.05],
                                         quaternion_xyzw=[1, 0, 0, 0]),
                                    dict(grasp_centre_m=[0.2, 0, 0.03], object_width_m=0.03, fixed_jaw_yaw_deg=90))
        self.assertEqual(set(back), {"id", "kind", "arm", "grasp_centre_m", "object_width_m", "fixed_jaw_yaw_deg"})

# === tests/test_so101_touch_off.py ===

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ni_rua.robot.calibration.workflow import touch_off
from ni_rua.robot.backends.so101.driver import FakeArmDriver
from ni_rua.robot.model import so101_kinematics as Kin
from robot.so101.test_so101_backend import deployment, make_adapter

MODEL = Kin.load_model()


class TableArm(FakeArmDriver):
    """Fake arm whose TCP z cannot go below ``table_tcp_z_m``: joints stop following there (contact).

    ``load_on_contact``: raw Present_Load rise reported by ``arm_load`` while blocked by the table.
    """

    def __init__(self, *args, table_tcp_z_m=0.0, load_on_contact=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.table_tcp_z_m, self.load_on_contact, self.blocked_by_table = table_tcp_z_m, load_on_contact, False

    def _advance(self):
        before = list(self.q)
        super()._advance()
        if Kin.fk(MODEL, self.q)[2, 3] < self.table_tcp_z_m:
            self.q = before
        # Pressed against the table while the goal lies below it (sag offset aside).
        self.blocked_by_table = Kin.fk(MODEL, self.goal)[2, 3] < self.table_tcp_z_m - 0.002

    def arm_load(self, names):
        self._advance()
        return {name: 100.0 + (self.load_on_contact if self.blocked_by_table else 0.0) for name in names}


def touch_adapter(directory, **arm):
    deploy = deployment()
    adapter, _, clock = make_adapter(directory, deploy=deploy)
    driver = TableArm(deploy, clock, q_rad=deploy["home"]["q_rad"], **arm)
    adapter.driver = driver
    adapter.arm_goal = list(driver.q)
    return adapter, driver


class TouchOffTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.directory, True)

    def settings(self, adapter, **changes):
        return dict(adapter.deployment["calibration"], **changes)

    def test_one_transient_in_each_joint_cannot_latch_a_contact(self):
        """Original wrist branch falsely confirmed the single 0.04 rad transient at z=+52 mm."""
        for joint in range(5):
            with self.subTest(joint=joint):
                adapter, _, clock = make_adapter(self.directory)
                settings = self.settings(adapter, touch_contact_load_delta=0., touch_min_z_m=.03,
                                         touch_step_m=.002, touch_warmup_steps=3)
                hover = adapter.kin.ik(adapter.model, touch_off.tcp_target([.22, .03, settings["touch_hover_z_m"]]),
                                       adapter.deployment["home"]["q_rad"], orientation_weight=0., limits=adapter.limits)["q_rad"]

                class TransientArm(FakeArmDriver):
                    def read(self):
                        return dict(q_rad=list(self.q), gripper_rad=self.jaw)

                    def send_goal(self, q_rad, gripper_normalized):
                        self.sends += 1
                        self.goal = list(q_rad)
                        self.q = list(q_rad)
                        if self.sends == 4:
                            self.q[joint] -= .04

                driver = TransientArm(adapter.deployment, clock, q_rad=hover)
                adapter.driver, adapter.arm_goal = driver, list(hover)
                report = dict(physics_advanced=False, status="completed", reason=None)
                result = touch_off._descend(adapter, settings, .22, .03, adapter.stop_generation,
                                            clock.now() + 60, lambda _: None, 0, 1, report)
                self.assertIsNone(result)
                trace = report["descent_traces"][0]
                self.assertIsNotNone(trace["first_candidate_read"])
                self.assertIsNone(trace["confirmation_read"])
                self.assertEqual(max(r["consecutive_hits"] for r in trace["reads"]), 1)
                first = trace["first_candidate_read"]
                self.assertEqual(trace["reads"][first]["q_command_rad"], trace["reads"][first+1]["q_command_rad"])

    def test_bounded_slide_records_both_directions_without_asserting_contact(self):
        adapter, _ = touch_adapter(self.directory, table_tcp_z_m=.01, load_on_contact=200.)
        settings = self.settings(adapter, touch_slide_distance_m=.003, touch_slide_step_m=.001)
        report = touch_off.touch_points(adapter, [[.22, .0]], settings)
        self.assertEqual(report["status"], "completed", report.get("reason"))
        trajectory = report["contact_trajectories"][0]
        self.assertEqual(trajectory["status"], "completed")
        self.assertEqual(trajectory["contact_evidence"], "unknown")
        self.assertIsNone(trajectory["contact_link"])
        samples = trajectory["samples"]
        self.assertEqual(len(samples), 7)
        self.assertEqual([r["phase"] for r in samples[1:]], ["slide_out"]*3 + ["slide_return"]*3)
        self.assertEqual(samples[-1]["ik_target_tcp_xyz_m"][:2], samples[0]["commanded_tcp_xyz_m"][:2])
        self.assertEqual(len(set(r["ik_target_tcp_xyz_m"][2] for r in samples[1:])), 1)
        for row in samples:
            np.testing.assert_allclose(np.subtract(row["q_command_rad"], report["descent_traces"][0]["sag_offset_rad"]),
                                       row["q_expected_rad"])
            self.assertLessEqual(row["command_sent_s"], row["joint_read_interval_s"][0])
            self.assertLessEqual(row["joint_read_interval_s"][1], row["load_read_interval_s"][0])

    def test_slide_loss_missing_load_and_bounds_stop_without_lift_or_park(self):
        from unittest.mock import patch
        for failure in ("missing", "lost", "load", "tracking", "timeout", "stop"):
            with self.subTest(failure=failure):
                adapter, _ = touch_adapter(self.directory, table_tcp_z_m=.01, load_on_contact=200.)
                settings = self.settings(adapter, touch_slide_distance_m=.004)
                original = touch_off._arm_load

                def load(a, enabled):
                    # During _slide's first load read, supply the requested fault.
                    import inspect
                    sliding = any(f.function == "_slide" for f in inspect.stack())
                    if sliding and failure == "missing":
                        return None
                    if sliding and failure == "load":
                        return np.full(3, 1000.)
                    return original(a, enabled)

                if failure == "tracking":
                    settings["touch_max_tracking_error_rad"] = .001
                if failure == "timeout":
                    settings["touch_slide_max_duration_s"] = .01
                if failure == "lost":
                    settings.update(touch_contact_lag_rad=.4)
                    def load(a, enabled):
                        import inspect
                        if any(f.function == "_slide" for f in inspect.stack()):
                            return np.full(3, 100.)
                        return original(a, enabled)
                def progress(update):
                    if failure == "stop" and update["phase"] == "slide_out":
                        adapter.operator("hold")
                with patch.object(touch_off, "_arm_load", side_effect=load):
                    report = touch_off.touch_points(adapter, [[.22, 0.]], settings, progress=progress)
                self.assertEqual(report["status"], "stopped" if failure == "stop" else "failed", report.get("reason"))
                self.assertIsNone(report["park"])
                self.assertEqual(report["contacts"], 0)
                self.assertTrue(report["descent_traces"][0]["reads"])
                if failure != "stop":
                    self.assertTrue(report["stop_confirmation"]["confirmed"])

    def test_timeout_does_not_erase_unknown_stop(self):
        from unittest.mock import patch
        adapter, _ = touch_adapter(self.directory, table_tcp_z_m=-1.)
        original = touch_off._descend
        def timeout(*args, **kwargs):
            args = list(args)
            args[5] = adapter.clock.now()  # deadline expires before first descent command
            return original(*args, **kwargs)
        with patch.object(touch_off, "_descend", side_effect=timeout), \
                patch.object(adapter, "_stop", return_value={"confirmed": False, "state": "unknown"}):
            report = touch_off.touch_points(adapter, [[.22, 0.]], self.settings(adapter))
        self.assertEqual(report["status"], "unknown")
        self.assertFalse(report["stop_confirmation"]["confirmed"])
        self.assertIsNone(report["park"])

    def test_invalid_load_and_tiny_slide_step_fail_closed(self):
        from unittest.mock import patch
        adapter, driver = touch_adapter(self.directory, table_tcp_z_m=.01)
        with patch.object(driver, "arm_load", return_value={name: float("nan") for name in touch_off.LOAD_JOINTS}):
            self.assertIsNone(touch_off._arm_load(adapter, True))
        report = touch_off.touch_points(adapter, [[.22, 0.]],
            self.settings(adapter, touch_slide_distance_m=.01, touch_slide_step_m=1e-300))
        self.assertEqual(report["reason"], "slide_timeout")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["contact_trajectories"][0]["samples"]), 1)

    def test_warmup_cannot_learn_away_load_guard(self):
        from unittest.mock import patch
        adapter, _ = touch_adapter(self.directory, table_tcp_z_m=-1.)
        loads = iter([np.zeros(3), np.full(3, 60.)])
        with patch.object(touch_off, "_arm_load", side_effect=lambda *args: next(loads)):
            report = touch_off.touch_points(adapter, [[.22, 0.]],
                self.settings(adapter, touch_contact_load_delta=120., touch_max_load_rise=50.))
        self.assertEqual(report["reason"], "touch_load_limit")
        row = report["descent_traces"][0]["reads"][0]
        self.assertEqual(row["guard_load_rise"], 60.)
        self.assertEqual(row["load_rise"], 0.)  # trigger baseline learned; guard baseline did not

    def test_failed_operator_hold_and_relax_never_claim_confirmed_stop(self):
        from unittest.mock import patch
        for command in ("hold", "relax"):
            with self.subTest(command=command):
                adapter, _ = touch_adapter(self.directory, table_tcp_z_m=-1.)
                def progress(update):
                    if update["phase"] == "descending":
                        adapter.operator(command)
                with patch.object(adapter, "_stop", return_value={"confirmed": False, "state": "unknown"}) as stop:
                    report = touch_off.touch_points(adapter, [[.22, 0.]], self.settings(adapter), progress=progress)
                self.assertEqual(report["status"], "unknown")
                self.assertFalse(report["stop_confirmation"]["confirmed"])
                self.assertIsNone(report["park"])
                if command == "relax":
                    self.assertEqual(stop.call_count, 0)
                    self.assertEqual(report["stop_confirmation"]["state"], "torque_off")

    def test_progress_exception_preserves_partial_slide_and_confirms_stop(self):
        from unittest.mock import patch
        adapter, _ = touch_adapter(self.directory, table_tcp_z_m=.01, load_on_contact=200.)
        def progress(update):
            if update["phase"] == "slide_out":
                raise RuntimeError("progress sink closed")
        with patch.object(adapter, "_stop", wraps=adapter._stop) as stop:
            report = touch_off.touch_points(adapter, [[.22, 0.]],
                self.settings(adapter, touch_slide_distance_m=.003), progress=progress)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(stop.call_count, 1)
        self.assertTrue(report["stop_confirmation"]["confirmed"])
        self.assertEqual(report["contact_trajectories"][0]["status"], "incomplete")
        self.assertIsNone(report["park"])

    def test_camera_extra_uses_post_hold_read_and_rejects_endpoint_drift(self):
        from unittest.mock import patch
        from scripts.so101_touch_table import camera_measurement
        from ni_rua.robot.calibration.workflow import charuco as cal
        adapter, _ = touch_adapter(self.directory, table_tcp_z_m=.01)
        before = adapter.read()
        after = dict(before, q_rad=[q + .1 for q in before["q_rad"]], t=before["t"] + .1)
        document = {"cameras": {"wrist": {}}, "board": cal.DEFAULT_BOARD_SPEC}
        with patch.object(cal, "read_calibration_file", return_value=document), \
                patch.object(adapter, "capture_frames", return_value={"wrist": np.zeros((32,32,3), np.uint8)}), \
                patch.object(adapter, "read", side_effect=[before, after]), \
                patch.object(cal, "detect_charuco") as detect:
            result = camera_measurement(adapter, "unused")({}, [9.]*5)
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["q_before_rad"], before["q_rad"])
        self.assertEqual(result["measurement_phase"], "post_hold")
        detect.assert_not_called()

    def test_partial_write_failure_retains_attempt_and_never_parks(self):
        from unittest.mock import patch
        from ni_rua.robot.backends.so101.worker import DriverFault
        adapter, _ = touch_adapter(self.directory, table_tcp_z_m=.01)
        original = touch_off._send_traced
        def fail(a, command, generation, trace, origin):
            with patch.object(a, "_send", side_effect=DriverFault("uncertain write")):
                return original(a, command, generation, trace, origin)
        with patch.object(touch_off, "_send_traced", side_effect=fail):
            report = touch_off.touch_points(adapter, [[.22, 0.]], self.settings(adapter))
        self.assertEqual(report["status"], "unknown")
        self.assertEqual(report["descent_traces"][0]["commands"][0]["delivery"], "unknown")
        self.assertIsNone(report["park"])

    def test_resampled_actual_goals_respect_limits_and_evidence_is_command_consistent(self):
        adapter, _ = touch_adapter(self.directory, table_tcp_z_m=.01, sag_rad=(0, .04, -.03, .02, 0))
        settings = self.settings(adapter, touch_contact_load_delta=0.)
        report = touch_off.touch_points(adapter, [[.22, 0.]], settings)
        self.assertEqual(report["status"], "completed", report.get("reason"))
        trace = report["descent_traces"][0]
        prior = np.asarray(trace["hover_q_measured_rad"]) + trace["sag_offset_rad"]
        self.assertTrue(any(r["ik_endpoint_pending"] for r in trace["reads"]))
        for row in trace["reads"]:
            command = np.asarray(row["q_command_rad"])
            for name, q in zip(touch_off.ARM_JOINTS, command):
                self.assertGreaterEqual(q, adapter.limits[name][0])
                self.assertLessEqual(q, adapter.limits[name][1])
            self.assertLessEqual(float(np.max(np.abs(command-prior))),
                                 adapter.deployment["motion"]["max_command_delta_rad"] + 1e-12)
            self.assertLessEqual(float(np.max(np.abs(command-prior))) / settings["touch_step_period_s"],
                                 adapter.effective_limits()["effective_max_joint_speed_rad_s"] + 1e-12)
            np.testing.assert_allclose(command - trace["sag_offset_rad"], row["q_expected_rad"])
            np.testing.assert_allclose(adapter.kin.fk(adapter.model, row["q_expected_rad"])[:3,3], row["commanded_tcp_xyz_m"])
            prior = command

    def test_lag_contact_records_table_height_lifts_and_parks(self):
        adapter, driver = touch_adapter(self.directory, table_tcp_z_m=0.01)
        settings = self.settings(adapter, touch_contact_load_delta=0.0)
        updates = []
        report = touch_off.touch_points(adapter, [[0.22, 0.0], [0.20, 0.05]], settings, progress=updates.append)
        self.assertEqual((report["status"], report["contacts"], report["planned_points"]), ("completed", 2, 2), report)
        for entry in report["touches"]:
            self.assertEqual((entry["status"], entry["trigger"]), ("contact", "lag"))
            # The commanded TCP stopped within a few steps below the table; the measured TCP sits on it.
            self.assertAlmostEqual(entry["tcp_measured_m"][2], 0.01, delta=0.003)
            self.assertLess(entry["commanded_tcp_z_m"], 0.01)
            self.assertIn(entry["contact_link"], ("gripper_link", "moving_jaw_so101_v1_link"))
            self.assertEqual(len(entry["contact_point_in_link_m"]), 3)
        summary = report["table_z_nominal_m"]
        self.assertEqual(summary["count"], 2)
        self.assertEqual(report["park"], "completed")
        np.testing.assert_allclose(driver.q, adapter.deployment["home"]["q_rad"], atol=settings["touch_arrival_tolerance_rad"])
        self.assertIn("contact", [update["phase"] for update in updates])
        self.assertTrue(report["physics_advanced"])
        self.assertEqual(len(report["motion_segments"]), 6)  # home, two hover/lift pairs, park
        self.assertTrue(report["motion_segments"][0]["result"]["trace"])
        self.assertIn("terminal_hold", report["descent_traces"][0])

    def test_load_trigger_fires_before_lag(self):
        adapter, _ = touch_adapter(self.directory, table_tcp_z_m=0.01, load_on_contact=500.0)
        settings = self.settings(adapter, touch_contact_lag_rad=0.4, touch_max_load_rise=800.0)  # load-only
        report = touch_off.touch_points(adapter, [[0.22, 0.0]], settings)
        self.assertEqual(report["touches"][0]["trigger"], "load")
        self.assertGreater(report["touches"][0]["load_rise"], settings["touch_contact_load_delta"])

    def test_trace_and_contact_keep_first_hit_separate_from_confirmation(self):
        from unittest.mock import patch
        adapter, _ = touch_adapter(self.directory, table_tcp_z_m=0.01)
        settings = self.settings(adapter, touch_contact_lag_rad=0.4, touch_confirm_reads=2, touch_max_load_rise=800.0)
        loads = iter([np.zeros(3), np.full(3, 500.), np.full(3, 700.)])
        with patch.object(touch_off, "_arm_load", side_effect=lambda *args: next(loads)):
            report = touch_off.touch_points(adapter, [[.22, 0.]], settings)
        self.assertEqual(report["status"], "completed")
        entry = report["touches"][0]
        trace = report["descent_traces"][0]
        first, last = trace["reads"]
        self.assertEqual(entry["load_rise"], 500.)
        self.assertEqual(entry["confirmation"]["load_rise"], 700.)
        self.assertEqual(entry["confirmation"]["reads"], 2)
        self.assertEqual(entry["q_measured_rad"], first["q_measured_rad"])
        self.assertEqual(entry["confirmation"]["q_measured_rad"], last["q_measured_rad"])
        self.assertEqual(first["commanded_tcp_z_m"], last["commanded_tcp_z_m"])
        self.assertEqual(last["q_command_rad"], first["q_command_rad"])
        self.assertEqual(entry["contact_evidence"], "unknown")
        for read in trace["reads"]:
            lag = np.subtract(read["q_expected_rad"], read["q_measured_rad"])
            np.testing.assert_allclose(read["lag_by_joint_rad"], lag)
            self.assertAlmostEqual(read["lag_rad"], np.max(np.abs(lag)))
            self.assertAlmostEqual(read["load_rise"], np.max(np.subtract(read["load_abs"], read["load_reference"])))

    def test_no_table_reaches_min_z_and_unreachable_point_is_skipped(self):
        adapter, _ = touch_adapter(self.directory, table_tcp_z_m=-1.0)
        settings = self.settings(adapter, touch_min_z_m=0.03, touch_contact_load_delta=0.0)
        report = touch_off.touch_points(adapter, [[0.22, 0.0], [0.9, 0.0]], settings)
        self.assertEqual([t["status"] for t in report["touches"]], ["no_contact_before_min_z", "unreachable"])
        self.assertEqual(report["contacts"], 0)
        self.assertIsNone(report["table_z_nominal_m"])

    def test_hold_during_descent_stops_without_park(self):
        adapter, _ = touch_adapter(self.directory, table_tcp_z_m=-1.0)
        settings = self.settings(adapter, touch_contact_load_delta=0.0)

        def progress(update):
            if update["phase"] == "descending":
                adapter.operator("hold")
        report = touch_off.touch_points(adapter, [[0.22, 0.0], [0.20, 0.05]], settings, progress=progress)
        self.assertEqual((report["status"], report["reason"]), ("stopped", "operator_hold"))
        self.assertIsNone(report["park"])
        self.assertEqual(len(report["touches"]), 1)

    def test_elbow_up_reachability_and_hover_residual_immunity_across_s1_and_s6_points(self):
        adapter, driver = touch_adapter(self.directory, table_tcp_z_m=0.01)
        settings = self.settings(adapter, touch_contact_load_delta=0.0)
        all_points = list(settings["touch_points_xy_m"]) + list(settings["verify_points_xy_m"])
        for x, y in all_points:
            target = touch_off.tcp_target([x, y, settings["touch_hover_z_m"]])
            np.testing.assert_allclose(target[:3, 2], [0.0, 0.0, -1.0], atol=1e-9)
            self.assertAlmostEqual(float(np.linalg.det(target[:3, :3])), 1.0, places=9)
            sol = adapter.kin.ik(adapter.model, target, adapter.deployment["home"]["q_rad"],
                                 orientation_weight=0.0, limits=adapter.limits)
            self.assertLess(sol["position_error_m"], settings["touch_ik_max_position_error_m"])
            # Must stay in natural elbow-up branch (never near-singular straight-elbow / cocked-wrist)
            self.assertLess(sol["q_rad"][1], -0.2)
            self.assertGreater(sol["q_rad"][2], 0.5)
        report = touch_off.touch_points(adapter, all_points, settings)
        self.assertEqual(report["contacts"], len(all_points))
        self.assertIsNotNone(report["gripper_closed_rad"])
        # Even if hover stops 0.045 rad short of IK target (within 0.05 rad arrival tolerance),
        # _descend must track incremental dq from q_start and never false-trigger in mid-air.
        hover_sol = adapter.kin.ik(adapter.model, touch_off.tcp_target([0.22, 0.03, settings["touch_hover_z_m"]]),
                                   adapter.deployment["home"]["q_rad"], orientation_weight=0.0, limits=adapter.limits)
        q_hover = list(hover_sol["q_rad"])
        driver.q = [q_hover[0], q_hover[1], q_hover[2] + 0.045, q_hover[3] - 0.045, q_hover[4]]
        driver.goal = list(driver.q)
        adapter.arm_goal = list(driver.q)
        contact = touch_off._descend(adapter, settings, 0.22, 0.03, adapter.stop_generation,
                                     adapter.clock.now() + 60.0, lambda u: None, 0, 1,
                                     {"physics_advanced": False, "status": "completed", "reason": None})
        self.assertIsInstance(contact, dict)
        self.assertGreater(contact["descent_steps"], 20)
        self.assertLess(contact["commanded_tcp_z_m"], 0.015)

    def test_sag_offset_preserved_on_contact_hold(self):
        sag = (0.0, 0.04, -0.03, 0.02, 0.0)
        adapter, driver = touch_adapter(self.directory, table_tcp_z_m=0.01, sag_rad=sag)
        settings = self.settings(adapter, touch_contact_load_delta=0.0)
        generation = adapter.stop_generation
        deadline = adapter.clock.now() + 60.0
        hover_sol = adapter.kin.ik(adapter.model, touch_off.tcp_target([0.22, 0.0, settings["touch_hover_z_m"]]),
                                   adapter.deployment["home"]["q_rad"], orientation_weight=0.0, limits=adapter.limits)
        adapter.move_joints(hover_sol["q_rad"], 0.0, settings["touch_arrival_tolerance_rad"],
                            speed_scale=settings["touch_speed_scale"], action_id="hover", correct_sag=True)
        expected_offset = np.subtract(adapter.arm_goal, adapter.read()["q_rad"])
        self.assertGreater(np.max(np.abs(expected_offset)), 0.01)
        report = {"physics_advanced": False, "status": "completed", "reason": None}
        contact = touch_off._descend(adapter, settings, 0.22, 0.0, generation, deadline,
                                     lambda u: None, 0, 1, report)
        self.assertIsInstance(contact, dict)
        # Hold command sent at end of _descend must include the sag offset above measured q
        np.testing.assert_allclose(np.subtract(driver.goal, contact["q_measured_rad"]), expected_offset, atol=0.01)

    def test_early_contact_during_warmup_steps_does_not_poison_load_ref(self):
        # Contact occurs while steps <= touch_warmup_steps=30; collision load must NOT be absorbed into load_ref
        adapter, _ = touch_adapter(self.directory, table_tcp_z_m=0.01, load_on_contact=500.0)
        settings = self.settings(adapter, touch_warmup_steps=30, touch_confirm_reads=2,
                                 touch_contact_lag_rad=0.4, touch_max_load_rise=800.0)
        report = touch_off.touch_points(adapter, [[0.22, 0.0]], settings)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["contacts"], 1)
        entry = report["touches"][0]
        self.assertEqual((entry["status"], entry["trigger"]), ("contact", "load"))
        self.assertLessEqual(entry["descent_steps"], settings["touch_warmup_steps"])
        self.assertAlmostEqual(entry["load_rise"], 500.0, delta=1.0)

    def test_timeout_sets_failed_status_and_timeout_entry(self):
        adapter, _ = touch_adapter(self.directory, table_tcp_z_m=-1.0)
        settings = self.settings(adapter, touch_timeout_s=0.5, touch_contact_load_delta=0.0)
        report = touch_off.touch_points(adapter, [[0.22, 0.0], [0.20, 0.05]], settings)
        self.assertEqual((report["status"], report["reason"]), ("failed", "touch_timeout"))
        self.assertIsNone(report["park"])
        self.assertTrue(all(t["status"] == "timeout" for t in report["touches"]))

    def test_fit_table_plane_and_write_gripper_limits(self):
        from ni_rua.robot.backends.so101.deployment import load_deployment, write_gripper_limits
        from robot.so101.test_so101_backend import EXAMPLE
        # Exact plane z = 0.05 * x - 0.02 * y - 0.015
        pts = [[0.20, 0.0], [0.24, 0.06], [0.24, -0.06], [0.17, -0.05]]
        touches = [{"status": "contact", "xy": [x, y], "contact_z_m": 0.05 * x - 0.02 * y - 0.015} for x, y in pts]
        summary = touch_off.table_summary(touches)
        plane = summary["plane"]
        self.assertIsNotNone(plane)
        self.assertAlmostEqual(plane["slope_x"], 0.05, places=6)
        self.assertAlmostEqual(plane["slope_y"], -0.02, places=6)
        self.assertAlmostEqual(plane["z_at_origin_m"], -0.015, places=6)
        self.assertLess(plane["residual_rms_m"], 1e-9)
        self.assertGreater(plane["tilt_deg"], 0.0)
        self.assertEqual(plane["coordinate_source"], "legacy_requested_tcp_xy_proxy")
        for touch in touches:
            x, y = touch["xy"]
            touch["contact_position_m"] = [x, y, touch["contact_z_m"]]
            touch["xy"] = [9., 9.]  # request coordinates cannot replace the modeled feature XY
        modeled = touch_off.fit_table_plane(touches)
        self.assertAlmostEqual(modeled["slope_x"], .05, places=6)
        self.assertEqual(modeled["coordinate_source"], "modeled_contact_xyz")
        self.assertFalse(modeled["physical_plane_verified"])
        # Collinear points or < 3 points return plane=None
        collinear = [{"status": "contact", "xy": [0.2, 0.0], "contact_z_m": 0.0},
                     {"status": "contact", "xy": [0.22, 0.0], "contact_z_m": 0.0},
                     {"status": "contact", "xy": [0.24, 0.0], "contact_z_m": 0.0}]
        self.assertIsNone(touch_off.fit_table_plane(collinear))
        self.assertIsNone(touch_off.fit_table_plane(touches[:2]))
        # write_gripper_limits atomic update + validation
        dep_path = self.directory / "so101.local.json"
        dep_path.write_text(EXAMPLE.read_text())
        res = write_gripper_limits(dep_path, jaw_closed_rad=-0.12, jaw_open_rad=1.55, stamp="20261109T000000Z")
        self.assertEqual(res["gripper"], {"jaw_closed_rad": -0.12, "jaw_open_rad": 1.55})
        self.assertEqual(load_deployment(dep_path)["gripper"]["jaw_closed_rad"], -0.12)
        with self.assertRaises(ValueError):
            write_gripper_limits(dep_path)


if __name__ == "__main__":
    unittest.main()
