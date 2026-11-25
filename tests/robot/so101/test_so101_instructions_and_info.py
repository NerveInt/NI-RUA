"""SO-101 model-facing instructions: rendered from the run's embodiment profile, X5 text untouched."""

NI_RUA_TEST_TIER = "fast"

# === tests/test_so101_actor_instructions.py ===

from pathlib import Path
from support.paths import CONFIGS_ROOT
import unittest
from unittest import mock

from ni_rua.cortex.context import actor_instructions as module
from ni_rua.cortex.context.actor_instructions import (CALIBRATED_OBSERVATION_INSTRUCTIONS, INSTRUCTIONS,
    LEGACY_OBSERVATION_INSTRUCTIONS, instructions, legacy_observation_instructions)
from ni_rua.cortex.context.task_planning import PLANNING_INSTRUCTIONS
from ni_rua.robot.backends.so101.contract import actor_motion_instructions
from ni_rua.robot.backends.so101.deployment import embodiment_profile, load_deployment

EXAMPLE = CONFIGS_ROOT / "so101.example.json"
X5_ONLY = ("ARX", "0.088 m", "link6", "env_origin", "both arms", "angles_deg=[pitch", "25 Hz")
# Shared-section text that is true only for a simulated, two-arm, grasp-angle robot (design 13 G1-G3).
SIMULATED_TWO_ARM_ONLY = ("simulat", "native-step", "native step", "grasp_position_m", "handoff", "cross-arm",
                          "To move both arms together", "do not advance the world")


class SO101ActorInstructionTests(unittest.TestCase):
    def setUp(self):
        self.deployment = load_deployment(EXAMPLE)
        self.profile = embodiment_profile(self.deployment)
        self.motion = actor_motion_instructions(self.deployment)
        self.text = instructions(legacy_observation_instructions(self.profile), None,
                                 embodiment_motion=self.motion, profile=self.profile)
        # Everything except the backend-owned motion section (robot/backends/so101/contract.py).
        self.shared = self.text.replace(self.motion.rstrip("\n"), "")

    def test_no_x5_specific_text_reaches_an_so101_actor(self):
        for phrase in X5_ONLY:
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, self.text)

    def test_shared_sections_carry_no_simulated_or_two_arm_fact(self):
        self.assertEqual(self.text.count(self.motion.rstrip("\n")), 1)
        for phrase in SIMULATED_TWO_ARM_ONLY:
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, self.shared)
        for phrase in ("simulation", "native-step", "## Run handoff policy"):
            with self.subTest(whole_text=phrase):
                self.assertNotIn(phrase, self.text)

    def test_real_time_clock_and_tcp_targets_are_stated(self):
        self.assertIn(module.WORLD_CLOCK_SENTENCES["real_time"], self.text)
        for _, real_time in module.PLANNING_REAL_TIME_SENTENCES:
            self.assertIn(real_time, self.text)
        self.assertIn(module.GRIPPER_RELATIVE_TARGET_SENTENCES["tcp_quaternion"], self.text)
        self.assertIn(module.BUDGET_SENTENCES["real_time"], self.text)
        self.assertIn("measured arm joints and gripper_frame_link poses", self.text)
        self.assertIn("hidden object state", legacy_observation_instructions(self.profile))

    def test_numbers_come_from_the_deployment_the_worker_enforces(self):
        motion = self.deployment["motion"]
        self.assertIn(f"{motion['max_joint_speed_rad_s']} rad/s", self.text)
        self.assertIn(f"{motion['default_position_tolerance_m'] * 1000:.0f} mm", self.text)
        self.assertIn(str(self.deployment["workspace"]["min_m"]), self.text)
        self.assertIn('only arm id is "right"', self.text)
        self.assertIn("so101_base", self.text)

    def test_shared_sections_are_kept(self):
        for heading in ("## Task and evidence", "## Task plan and current concerns", "## Tools and execution",
                        "## Feedback and next decision", "## Programs and reviews", "## Reusable experience",
                        "## Observation contract"):
            self.assertEqual(self.text.count(heading), 1, heading)
        self.assertEqual(self.text.count("## Motion planning"), 1)
        self.assertEqual(self.text.count("## Coordinates and timing"), 1)
        self.assertEqual(self.text.count("## Run handoff policy"), 0)
        self.assertTrue(self.text.endswith(legacy_observation_instructions(self.profile)))

    def test_profile_refuses_text_it_cannot_honour(self):
        with self.assertRaisesRegex(ValueError, "handoff_policy_requires_cross_arm_embodiment"):
            instructions(CALIBRATED_OBSERVATION_INSTRUCTIONS, "handoff", embodiment_motion=self.motion,
                         profile=self.profile)
        # The ARX-X5 motion sections describe angles_deg grasp poses: a profile without them needs its own section.
        with self.assertRaisesRegex(ValueError, "embodiment_motion_section_required"):
            instructions(CALIBRATED_OBSERVATION_INSTRUCTIONS, None, profile=self.profile)

    def test_planning_text_drift_fails_loudly_instead_of_keeping_simulated_wording(self):
        for simulated, _ in module.PLANNING_REAL_TIME_SENTENCES:
            self.assertEqual(PLANNING_INSTRUCTIONS.count(simulated), 1, simulated)
        edited = PLANNING_INSTRUCTIONS.replace("Waiting advances simulation", "Waiting advances the simulation")
        with mock.patch.object(module, "PLANNING_INSTRUCTIONS", edited):
            self.assertEqual(module.planning_instructions("simulated_steps"), edited)
            with self.assertRaisesRegex(ValueError, "planning_clock_sentence_changed"):
                module.planning_instructions("real_time")
        with self.assertRaisesRegex(ValueError, "invalid_clock"):
            module.planning_instructions("paused")

    def test_x5_default_is_unchanged(self):
        from ni_rua.robot.model.embodiments import ARX_X5_PROFILE, arx_x5_profile
        for contract in (LEGACY_OBSERVATION_INSTRUCTIONS, CALIBRATED_OBSERVATION_INSTRUCTIONS):
            default = instructions(contract, "handoff")
            self.assertEqual(default, instructions(contract, "handoff", embodiment_motion=None))
            self.assertEqual(default, instructions(contract, "handoff", profile=ARX_X5_PROFILE))
            # Feedback mode (one arm at a time) keeps the X5 text: no X5 change is approved for it.
            self.assertEqual(default, instructions(contract, "handoff",
                                                   profile=arx_x5_profile(motion_execution_mode="feedback")))
            self.assertIn("ARX-X5 full open is 0.088 m", default)
            self.assertIn(INSTRUCTIONS[INSTRUCTIONS.index("## Motion planning"):INSTRUCTIONS.index("## Feedback")], default)
            self.assertTrue(default.endswith("\n\n## Run handoff policy\nhandoff"))
        self.assertEqual(legacy_observation_instructions(), LEGACY_OBSERVATION_INSTRUCTIONS)
        self.assertEqual(legacy_observation_instructions(ARX_X5_PROFILE), LEGACY_OBSERVATION_INSTRUCTIONS)
        self.assertEqual(module.planning_instructions("simulated_steps"), PLANNING_INSTRUCTIONS)


class GripperGeometryWiringTests(unittest.TestCase):
    def test_instructions_carry_gripper_geometry_and_table_status(self):
        text = actor_motion_instructions(load_deployment(EXAMPLE))
        self.assertIn("gripper_geometry (TCP frame", text)
        self.assertIn("the only finger that moves", text)
        self.assertIn("goal_normalized -> jaw angle", text)
        self.assertIn("lies ON the fixed jaw's inner face", text)
        self.assertNotIn("between the jaw tips", text)
        self.assertIn("Table height is not measured", text)  # example deployment has no touch calibration

    def test_capabilities_and_gripper_receipt_report_geometry(self):
        import tempfile, threading
        from robot.so101.test_so101_backend import make_adapter, IDENTITY
        from ni_rua.robot.backends.so101 import contract
        from ni_rua.robot.model import so101_gripper_geometry as gg
        with tempfile.TemporaryDirectory() as directory:
            adapter, _, _ = make_adapter(directory)
            geometry = adapter.capabilities()["tool_geometry"]["gripper_geometry"]
            self.assertEqual(geometry["frame"], "gripper_frame_link")
            self.assertEqual([row["goal_normalized"] for row in geometry["goal_to_opening"]], list(gg.GOAL_TABLE))
            self.assertNotIn("between the jaw tips", adapter.capabilities()["tool_geometry"]["conversion"])
            action = contract.compile_action("gripper", dict(arm="right", goal_normalized=1.0, observation_id="obs"),
                                             IDENTITY, adapter.capabilities())
            receipt = adapter.execute(action, lambda update: None, threading.Event())
            jaw = receipt["gripper_execution"]["right"]
            self.assertGreater(jaw["measured_opening_m"], 0.03)

# === tests/test_so101_legacy_settings.py ===

import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import so101_calibrate as legacy


class LegacySettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / 'robot.json'
        self.config.write_text('{}')

    def resolve(self, step='capture', flags=()):
        required = {'capture': ['--config', str(self.config), '--output-root', str(self.root)],
                    'capture-handheld': ['--config', str(self.config), '--output-root', str(self.root)],
                    'solve': ['--config', str(self.config), '--session', str(self.root)],
                    'verify': ['--session', str(self.root), '--calibration', 'unused.json']}
        args = legacy.build_parser().parse_args([step, *required[step], *flags])
        legacy.resolve_settings(args)
        return args

    def test_historical_defaults_and_units_are_unchanged(self):
        args = self.resolve()
        self.assertEqual(args.n_poses, 20)
        self.assertEqual(args.z_floor_m, .05)
        self.assertEqual(args.rate_hz, 20.)
        self.assertEqual(args.max_joint_speed_rad_s, math.radians(15))
        self.assertEqual(args.max_command_delta_rad, math.radians(2))
        self.assertEqual(args.settle_tolerance_deg, .3)
        self.assertIsNone(self.resolve('solve').max_fixed_reprojection_px)
        self.assertEqual(self.resolve('solve').hand_eye_score_lever_m, .1)

    def test_deployment_values_override_omitted_cli_defaults(self):
        self.config.write_text(json.dumps({'so101': {'motion': {'max_joint_speed_rad_s': .13,
            'max_command_delta_rad': .02, 'command_rate_hz': 15.}, 'max_relative_target_deg': 6.,
            'calibration': {'settle_timeout_s': 3., 'z_floor_m': None, 'handheld_target_views': 35,
                'handheld_timeout_s': 100., 'min_board_corners': 12, 'handheld_still_px': .7,
                'handheld_min_novelty': .15}}}))
        args = self.resolve()
        self.assertEqual((args.max_joint_speed_rad_s, args.max_command_delta_rad, args.rate_hz), (.13, .02, 15.))
        self.assertEqual((args.max_relative_target_deg, args.settle_timeout_s, args.z_floor_m), (6., 3., None))
        self.assertEqual(args.settle_tolerance_deg, .3)  # no conversion of unrelated velocity/arrival semantics
        handheld = self.resolve('capture-handheld')
        self.assertEqual((handheld.views, handheld.timeout_s, handheld.min_corners, handheld.still_px,
                          handheld.min_novelty), (35, 100., 12, .7, .15))

    def test_user_file_and_explicit_flags_have_documented_precedence(self):
        self.config.write_text(json.dumps({'motion': {'command_rate_hz': 15.}}))
        settings = self.root / 'legacy.json'
        settings.write_text(json.dumps({'capture': {'rate_hz': 12., 'n_poses': 23, 'board_yaw_deg': 25.}}))
        args = self.resolve(flags=['--settings', str(settings), '--rate-hz', '18', '--board-yaw-deg', '0'])
        self.assertEqual((args.rate_hz, args.n_poses, args.board_yaw_deg), (18., 23, 0.))
        self.assertEqual(args.effective_settings['sources']['n_poses'], 'settings')
        self.assertEqual(args.effective_settings['sources']['rate_hz'], 'cli')
        self.assertEqual(self.resolve().rate_hz, 15.)  # no mutation of defaults

    def test_invalid_settings_fail_before_hardware_or_solve(self):
        path = self.root / 'bad.json'
        for value in ({'captur': {}}, {'capture': {'n_pose': 2}}, {'capture': {'rate_hz': 0}},
                      {'capture': {'n_poses': True}}, {'solve': {'max_pnp_reprojection_px': float('nan')}},
                      {'capture': {'distance_min_m': .4, 'distance_max_m': .2}}):
            path.write_text(json.dumps(value))
            with self.subTest(value=value), patch.object(legacy, 'make_robot') as robot, \
                 self.assertRaises(ValueError):
                self.resolve(flags=['--settings', str(path)])
            robot.assert_not_called()

    def test_zero_motion_flag_is_rejected_not_replaced_by_config(self):
        self.config.write_text('{"motion":{"max_joint_speed_rad_s":0.1}}')
        with self.assertRaises(ValueError):
            self.resolve(flags=['--max-joint-speed-rad-s', '0'])

    def test_print_settings_never_calls_physical_entry(self):
        out = io.StringIO()
        with patch.object(legacy, 'cmd_capture', side_effect=AssertionError('no hardware')), contextlib.redirect_stdout(out):
            self.assertEqual(legacy.main(['capture', '--config', str(self.config), '--output-root', str(self.root),
                                          '--print-settings']), 0)
        self.assertEqual(json.loads(out.getvalue())['values']['n_poses'], 20)

    def test_solve_uses_deployment_gates_and_archives_effective_values_even_on_gate_failure(self):
        self.config.write_text(json.dumps({'calibration': {'max_pnp_reprojection_px': 1.2,
            'max_intrinsics_rms_px': .7, 'max_hand_eye_translation_m': .003,
            'max_hand_eye_rotation_rad': .01, 'max_fixed_reprojection_px': 4.}}))
        args = self.resolve('solve', ['--max-intrinsics-rms-px', '.9'])
        with patch.object(legacy, 'solve_session', side_effect=legacy.cal.CalibrationGateError('fixture')) as solve, \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(legacy.cmd_solve(args), 2)
        self.assertEqual(solve.call_args.kwargs['max_pnp_reprojection_px'], 1.2)
        self.assertEqual(solve.call_args.kwargs['thresholds'], {'max_intrinsics_rms_px': .9,
            'max_hand_eye_translation_m': .003, 'max_hand_eye_rotation_rad': .01, 'max_fixed_reprojection_px': 4.})
        saved = json.loads((self.root / 'legacy_solve_settings.json').read_text())
        self.assertEqual(saved, args.effective_settings)

    def test_session_record_retains_effective_settings(self):
        args = self.resolve()
        record = legacy._session_record(args, {}, 'automatic', {}, {})
        self.assertEqual(record['legacy_effective_settings'], args.effective_settings)

    def test_hand_eye_ranking_coefficient_preserves_zero_and_each_override_layer(self):
        self.config.write_text('{"calibration":{"hand_eye_score_lever_m":0.2}}')
        self.assertEqual(self.resolve('solve').hand_eye_score_lever_m, .2)
        path = self.root / 'ranking.json'
        path.write_text('{"solve":{"hand_eye_score_lever_m":0.3}}')
        self.assertEqual(self.resolve('solve', ['--settings', str(path)]).hand_eye_score_lever_m, .3)
        args = self.resolve('solve', ['--settings', str(path), '--hand-eye-score-lever-m', '0'])
        with patch.object(legacy, 'solve_session', side_effect=legacy.cal.CalibrationGateError('fixture')) as solve, \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(legacy.cmd_solve(args), 2)
        self.assertEqual(solve.call_args.kwargs['hand_eye_score_lever_m'], 0.)
        self.assertNotIn('hand_eye_score_lever_m', solve.call_args.kwargs['thresholds'])
        saved = json.loads((self.root / 'legacy_solve_settings.json').read_text())
        self.assertEqual(saved['values']['hand_eye_score_lever_m'], 0.)
        self.assertEqual(saved['sources']['hand_eye_score_lever_m'], 'cli')
        for bad in ('-0.1', '10.1'):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                self.resolve('solve', ['--hand-eye-score-lever-m', bad])

# === tests/test_so101_model_facts.py ===

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ni_rua.robot.backends.so101 import contract
from ni_rua.robot.backends.so101.deployment import embodiment_profile, load_deployment
from ni_rua.robot.model import so101_gripper_geometry as gg
from ni_rua.robot.model import so101_kinematics as kin
from ni_rua.robot.model.robot_information import robot_information
from ni_rua.runtime.actions.robot_action_schema import build
from robot.so101.test_so101_backend import deployment
from robot.so101.test_so101_motion_features import Case

X5_ONLY_SCHEMA = ("exact trajectory", "exact cached rows", "blocking_risks is empty", "vertical contact/load compliance")


def straight_down(a):
    """Rotation matrix of quaternion_xyzw [cos(a/2), sin(a/2), 0, 0]."""
    x, y = math.cos(a / 2), math.sin(a / 2)
    return np.array([[1 - 2 * y * y, 2 * x * y, 0.0], [2 * x * y, 1 - 2 * x * x, 0.0], [0.0, 0.0, -1.0]])


def planner_limits(deploy):
    margin = deploy["motion"]["joint_limit_margin_rad"]
    return {name: (low + margin, high - margin) for name, (low, high) in kin.joint_limits_rad(kin.load_model()).items()
            if name in kin.ARM_JOINTS}


class MotionTextFactsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deployment = load_deployment(EXAMPLE)
        cls.text = contract.actor_motion_instructions(cls.deployment)

    def test_tcp_is_named_on_the_fixed_jaw_face_everywhere(self):
        self.assertIn("on the fixed jaw's inner fingertip face, not between the jaws", self.text)
        briefing = robot_information("so101")["briefing"]
        reference = json.dumps(robot_information("so101", "all")["reference"])
        for name, text in (("motion", self.text), ("briefing", briefing), ("physical", reference)):
            with self.subTest(text=name):
                self.assertNotIn("between the jaw tips", text)
                self.assertNotIn("between the fixed and moving jaw tips", text)
                self.assertNotIn("tied to the shoulder_pan azimuth", text)
        self.assertIn("fixed jaw's inner fingertip face", briefing)
        self.assertNotIn("roll about the approach axis and the remaining orientation are coupled", briefing)
        self.assertNotIn("to be verified on hardware", briefing)

    def test_packaged_moving_tip_number_is_the_cad_inner_point(self):
        axes = robot_information("so101", "all")["reference"]["kinematics"]["pose_interface"]["tcp_axes"]
        tip = gg.derive_gripper_geometry()["moving_jaw"]["fingertip_inner_point_at_zero_m"][0]
        self.assertIn(f"{abs(tip):.4f} m along local -X", axes)
        midpoint = gg.fingertip_reference(0.0)["midpoint_m"][0]
        self.assertIn(f"CAD: {1000 * abs(midpoint):.1f} mm along -X at jaw 0 rad", robot_information("so101")["briefing"])

    def test_top_down_band_is_derived_from_the_deployment_limits(self):
        limits = planner_limits(self.deployment)["wrist_roll"]
        facts = gg.top_down_roll_facts(limits)
        centre, half = facts["unreachable_centre_rad"], facts["unreachable_half_width_rad"]
        self.assertIn(f"[{limits[0]:.4f}, {limits[1]:.4f}] rad", self.text)
        self.assertIn(f"({math.degrees(centre - half):+.1f}, {math.degrees(centre + half):+.1f}) deg", self.text)
        self.assertIn(f"a = phi + {facts['yaw_at_zero_roll_rad']:.4f} rad + wrist_roll", self.text)
        self.assertIn("[cos(a/2), sin(a/2), 0, 0]", self.text)
        self.assertIn("(sin a, -cos a, 0)", self.text)
        self.assertIn("radially away from the base", self.text)
        # A wider margin narrows the reachable roll range and widens the band by the same amount.
        wider = gg.top_down_roll_facts((limits[0] + 0.1, limits[1] - 0.1))
        self.assertAlmostEqual(wider["unreachable_half_width_rad"] - half, 0.1, places=9)

    def test_fingertip_contact_heights_come_from_the_cad(self):
        reach = gg.jaw_reach_facts(self.deployment["gripper"]["jaw_closed_rad"], self.deployment["gripper"]["jaw_open_rad"])
        self.assertIn(f"fixed fingertip is {1000 * reach['fixed_reach_m']:.1f} mm below the TCP", self.text)
        with tempfile.TemporaryDirectory() as directory:
            calibration = Path(directory) / "camera_calibration.json"
            calibration.write_text(json.dumps({"joint_calibration": {"table_z_m": -0.016}}))
            line = contract._table_line(dict(self.deployment, camera_calibration_file=str(calibration)),
                                        self.deployment["workspace"])
        self.assertIn(f"touches the table at TCP z={-0.016 + reach['fixed_reach_m']:.4f} m", line)
        self.assertIn(f"up to TCP z={-0.016 + reach['peak_reach_m']:.4f} m", line)
        self.assertNotIn("~6-8 mm", line)

    def test_readings_name_cad_and_measured_values_and_both_field_names(self):
        measured = deployment(**{"gripper.empty_close_aperture_m": 0.0051, "gripper.full_open_aperture_m": 0.1361})
        text = contract._gripper_text(measured)
        self.assertIn("gripper_execution.measured_opening_m", text)
        self.assertIn("measured_aperture_m (grip_guard, grip_facts) are one reading", text)
        self.assertIn("empty goal-0 close reads 0.0051 m on this unit", text)
        self.assertIn(f"CAD at the commanded closed limit {contract._metres(contract.closed_limit_opening_m(measured))}",
                      text)
        self.assertIn("Goal 1 reads 0.1361 m (deployment gripper.full_open_aperture_m; CAD 0.1372 m).",
                      contract.grip_guard_text(measured))
        self.assertIn("Goal 1 reads 0.1372 m (CAD opening at gripper.jaw_open_rad).", contract.grip_guard_text(self.deployment))
        unmeasured = contract._gripper_text(self.deployment)
        self.assertIn("not measured in this deployment", unmeasured)

    def test_centred_grasp_formula_is_clean_and_profile_is_labelled(self):
        self.assertNotRegex(self.text, r"[-+] [-+]0\.0000")
        # The conversion formula lives once, in the robot_action field help of grasp_centre_m; the gripper geometry
        # text points there and keeps the CAD fact the field help does not carry (the face steps toward +X with depth).
        self.assertNotIn("TCP = object_centre + (w/2 - x_face)", self.text)
        self.assertIn("robot_action grasp_centre_m computes that TCP target (its field help is the conversion)", self.text)
        profile = gg.derive_gripper_geometry()["fixed_jaw"]["inner_face_profile_m"]
        self.assertTrue(all(set(row) == {"z_m", "x_m"} for row in profile))
        self.assertIn(f"z {profile[0]['z_m']:+.4f}: x {profile[0]['x_m']:+.4f}", self.text)
        self.assertIn(f"up to x {max(row['x_m'] for row in profile):+.4f}), so an object gripped deeper", self.text)

    def test_release_and_opening_facts_follow_the_cad(self):
        g = self.deployment["gripper"]
        reach = gg.jaw_reach_facts(g["jaw_closed_rad"], g["jaw_open_rad"])
        self.assertIn("fixed jaw's inner face stays on the held object's\n+X face", self.text)
        self.assertIn(f"retracts behind the TCP plane above goal {reach['behind_tcp_plane_above_goal']:.2f}", self.text)
        self.assertIn(f"wholly behind the TCP plane above goal {reach['behind_tcp_plane_above_goal']:.2f}", self.text)

    def test_continuation_is_named(self):
        self.assertIn("source_plan_id and start_from_stage_id", self.text)
        self.assertIn("skip_failed_stage=true", self.text)
        measured = contract.grip_guard_text(deployment(**{"gripper.empty_close_aperture_m": 0.0051}))
        self.assertIn("Choose the guard from the object", measured)
        self.assertIn("rigid_object, wide_object", measured)
        self.assertIn("Choose the guard from the object", contract.grip_guard_text(self.deployment))


class GeometryAgainstKinematicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = kin.load_model()
        cls.limits = planner_limits(load_deployment(EXAMPLE))
        cls.facts = gg.top_down_roll_facts(cls.limits["wrist_roll"])

    def down_config(self, pan, roll, lift=0.3, elbow=0.5):
        """Arm joints with the approach axis straight down (wrist_flex solved by bisection on approach x)."""
        flex = [-1.6, 1.6]
        for _ in range(60):
            mid = 0.5 * sum(flex)
            tilt = kin.fk(self.model, [0, lift, elbow, mid, 0])[0, 2]
            flex[0 if tilt > 0 else 1] = mid  # approach x shrinks as wrist_flex grows at this lift/elbow
        return [pan, lift, elbow, 0.5 * sum(flex), roll]

    def test_tcp_is_on_the_fixed_inner_face_and_the_midpoint_moves(self):
        facts = gg.derive_gripper_geometry()
        self.assertAlmostEqual(facts["fixed_jaw"]["fingertip_inner_point_m"][0], 0.0, delta=1e-4)
        mid0 = gg.fingertip_reference(0.0)["midpoint_m"][0]
        self.assertAlmostEqual(mid0, 0.5 * facts["moving_jaw"]["fingertip_inner_point_at_zero_m"][0], delta=1e-4)
        self.assertLess(gg.fingertip_reference(0.3)["midpoint_m"][0], mid0 - 0.01)
        self.assertAlmostEqual(facts["wrist_roll_axis_tcp"]["tcp_distance_m"], self.facts["roll_axis_offset_m"], places=3)

    def test_straight_down_quaternion_matches_fk(self):
        for pan, roll in ((0.0, 0.0), (0.4, -1.0), (-0.7, 2.0)):
            with self.subTest(pan=pan, roll=roll):
                T = kin.fk(self.model, self.down_config(pan, roll))
                self.assertLess(T[2, 2], -1 + 1e-6)
                a = math.atan2(T[1, 0], T[0, 0])
                np.testing.assert_allclose(T[:3, :3], straight_down(a), atol=2e-4)
                np.testing.assert_allclose(T[:3, 1], [math.sin(a), -math.cos(a), 0.0], atol=2e-4)
                # a = phi + yaw_at_zero_roll_rad + roll_sign * wrist_roll, phi from the pan axis to the roll axis.
                _, axes = kin._chain(self.model, np.array(self.down_config(pan, roll)), 0.0)
                origin = axes["wrist_roll"][0]
                px, py = self.facts["pan_axis_xy_m"]
                phi = math.atan2(origin[1] - py, origin[0] - px)
                expected = phi + self.facts["yaw_at_zero_roll_rad"] + self.facts["roll_sign"] * roll
                self.assertAlmostEqual(math.remainder(a - expected, 2 * math.pi), 0.0, places=4)

    def test_band_agrees_with_ik(self):
        centre, half = self.facts["unreachable_centre_rad"], self.facts["unreachable_half_width_rad"]
        px, py = self.facts["pan_axis_xy_m"]
        position = np.array([0.25, 0.06, 0.03])
        r = math.hypot(position[0] - px, position[1] - py)
        phi = math.atan2(position[1] - py, position[0] - px)
        shift = math.atan(self.facts["roll_axis_offset_m"] / r)
        seed = [0.0, -0.95, 0.90, 1.00, 0.0]
        for relative, reachable in ((centre, False), (centre + half - shift - 0.03, False),
                                    (centre - half + shift + 0.03, False), (centre + half + shift + 0.03, True),
                                    (centre - half - shift - 0.03, True), (centre + math.pi, True)):
            with self.subTest(relative_deg=round(math.degrees(relative), 1)):
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = straight_down(phi + relative), position
                result = kin.ik(self.model, T, seed, orientation_weight=1.0, limits=self.limits, restarts=24,
                                position_tolerance_m=1e-3, orientation_tolerance_rad=2e-3)
                self.assertEqual(result["converged"], reachable, result)

    def test_reach_facts_are_consistent_with_the_opening_table(self):
        g = load_deployment(EXAMPLE)["gripper"]
        reach = gg.jaw_reach_facts(g["jaw_closed_rad"], g["jaw_open_rad"])
        behind = reach["behind_tcp_plane_above_goal"]
        jaw = g["jaw_closed_rad"] + behind * (g["jaw_open_rad"] - g["jaw_closed_rad"])
        self.assertAlmostEqual(gg.moving_jaw_reach_m(jaw), 0.0, delta=1e-6)
        self.assertLess(gg.moving_jaw_reach_m(jaw + 0.01), 0.0)
        self.assertGreater(reach["peak_reach_m"], reach["fixed_reach_m"])
        low, high = reach["beyond_fixed_jaw_goals"]
        self.assertTrue(low < reach["peak_goal"] < high < behind)
        rows = gg.gripper_geometry_payload(g["jaw_closed_rad"], g["jaw_open_rad"])["goal_to_opening"]
        for row in rows:
            if row["goal_normalized"] > behind + 0.01:
                self.assertLess(row["moving_tip_z_m"], 0.0, row)


class SchemaWordingTests(unittest.TestCase):
    def test_real_time_catalogue_has_no_x5_only_wording(self):
        profile = embodiment_profile(deployment(**{"gripper.empty_close_aperture_m": 0.0051}))
        contract_text = json.dumps(build(profile).tool)
        # SO-101 runs one preview review (grasp_3d_review on grasp-centre stages, package B), so execute_if_clean's
        # "zero warnings/risks" sentence is accurate there and names that review; the other X5-only phrases stay out.
        self.assertIn("grasp_3d_review", profile.pose_inputs and contract_text)
        for phrase in X5_ONLY_SCHEMA:
            if phrase == "zero warnings/risks":
                continue
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, contract_text)
        self.assertIn("grasp_3d_review (grasp-centre stages only; TCP-pose stages are not reviewed)", contract_text)
        self.assertIn("no downgrade_reasons", contract_text)
        self.assertIn("closed_jaw_before_grasp", contract_text)
        self.assertIn("re-interpolated from the joints measured at execution", contract_text)
        unmeasured = json.dumps(build(embodiment_profile(load_deployment(EXAMPLE))).tool)
        self.assertNotIn("closed_jaw_before_grasp", unmeasured)

    def test_x5_catalogue_keeps_its_wording(self):
        text = json.dumps(build(None).tool)
        for phrase in X5_ONLY_SCHEMA:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)


class ContinuationTests(Case):
    """Recovery the SO-101 motion text names, through the shared robot_action path (unified_actions.dispatch)."""

    def dispatch(self, harness, args, call):
        from ni_rua.runtime.actions.unified_actions import dispatch
        with patch("ni_rua.runtime.actions.inspection._trajectory_image", return_value=({"status": "unavailable"}, [])):
            return dispatch(harness, args, {"tool_call_id": call}).data

    def test_paused_batch_continues_from_a_named_stage(self):
        harness, adapter, _ = self.harness()
        stages = [self.public_move(adapter, "up", dz=0.01, position_tolerance_m=0.005, orientation_tolerance_rad=0.1),
                  dict(id="look", kind="observe", reason="check the arm"),
                  self.public_move(adapter, "higher", dz=0.02, position_tolerance_m=0.005, orientation_tolerance_rad=0.1)]
        preview = self.dispatch(harness, dict(observation_id=harness.latest["observation_id"], stages=stages), "p1")
        self.assertEqual(preview["status"], "planned", preview)
        paused = self.dispatch(harness, dict(mode="execute", plan_id=preview["plan_id"]), "e1")["receipt"]
        self.assertEqual(paused["status"], "stopped", paused)
        z_paused = self.pose(adapter)["position_m"][2]
        target = stages[2]["position_m"]
        resumed = self.dispatch(harness, dict(observation_id=harness.latest["observation_id"],
                                              source_plan_id=preview["plan_id"], start_from_stage_id="higher",
                                              stage_overrides={"higher": {"position_tolerance_m": 0.006}}), "p2")
        self.assertEqual(resumed["status"], "planned", resumed)
        self.assertEqual(list(resumed["stage_plans"]), ["higher"])
        receipt = self.dispatch(harness, dict(mode="execute", plan_id=resumed["plan_id"]), "e2")["receipt"]
        self.assertEqual(receipt["status"], "completed", receipt)
        self.assertAlmostEqual(self.pose(adapter)["position_m"][2], target[2], delta=0.006)
        self.assertGreater(self.pose(adapter)["position_m"][2], z_paused + 0.005)

    def test_failed_stage_is_skipped_on_request(self):
        harness, adapter, driver = self.harness()
        start = self.pose(adapter)["position_m"]
        stages = [self.public_move(adapter, "blocked", dz=0.02, position_tolerance_m=0.005, orientation_tolerance_rad=0.1),
                  self.public_move(adapter, "next", dz=0.01, position_tolerance_m=0.005, orientation_tolerance_rad=0.1)]
        preview = self.dispatch(harness, dict(observation_id=harness.latest["observation_id"], stages=stages), "p1")
        self.assertEqual(preview["status"], "planned", preview)
        driver.blocked = set(range(5))  # every arm joint stalls: the first move fails with tracking_error
        failed = self.dispatch(harness, dict(mode="execute", plan_id=preview["plan_id"]), "e1")["receipt"]
        driver.blocked = set()
        self.assertNotEqual(failed["status"], "completed", failed)
        self.assertEqual(failed.get("failed_stage_id") or (failed.get("summary") or {}).get("failed_stage_id"),
                         "blocked", failed)
        repaired = self.dispatch(harness, dict(observation_id=harness.latest["observation_id"],
                                               plan_id=preview["plan_id"], skip_failed_stage=True), "p2")
        self.assertEqual(repaired["status"], "planned", repaired)
        self.assertEqual(list(repaired["stage_plans"]), ["next"])
        receipt = self.dispatch(harness, dict(mode="execute", plan_id=repaired["plan_id"]), "e2")["receipt"]
        self.assertEqual(receipt["status"], "completed", receipt)
        self.assertAlmostEqual(self.pose(adapter)["position_m"][2], start[2] + 0.01, delta=0.005)

# === tests/test_so101_robot_information.py ===

import json
import struct
import unittest
import xml.etree.ElementTree as ET
import zipfile

from ni_rua.robot.model.robot_information import (
    ROBOT_INFORMATION_SECTIONS,
    robot_geometry_path,
    robot_information,
    robot_information_text,
    robot_resource_path,
    robot_resources,
)
from ni_rua.robot.model import so101_kinematics as kin


class SO101RobotInformationTests(unittest.TestCase):
    def test_registered_and_every_section_available(self):
        for section in ROBOT_INFORMATION_SECTIONS:
            with self.subTest(section=section):
                result = robot_information("so101", section)
                self.assertTrue(result["available"], result)
                self.assertEqual(result["robot_id"], "so101")
        self.assertIsNone(robot_geometry_path("so101"))
        self.assertFalse(robot_information("so-101")["available"])

    def test_default_projection_has_no_x5_fields(self):
        default = robot_information("so101")
        compact = default["reference"]
        self.assertEqual(set(compact), {"identity", "kinematics", "geometry", "control"})
        self.assertEqual(compact["identity"]["native_action_dimensions"], 6)
        pose = compact["kinematics"]["pose_interface"]
        self.assertEqual(pose["tcp_link"], kin.TCP_LINK)
        self.assertIn("quaternion_xyzw", pose["pose_units"])
        self.assertNotIn("tool_axis_link6", pose)
        self.assertNotIn("grasp_contact_offsets_m", pose)
        control = compact["control"]
        self.assertNotIn("preview_gripper_target_m", control)
        self.assertNotIn("native_joint_target_m", control["gripper"])
        self.assertIn("native_joint_target_rad", control["gripper"])
        self.assertEqual(control["gripper"]["normalized_command_range"], [0, 1])
        self.assertIn("real time", control["timing"]["clock"])
        text = json.dumps(compact)
        self.assertNotIn("link6", text)
        self.assertNotIn("joints", compact["kinematics"])

    def test_briefing_states_conventions_and_unknowns_without_task_content(self):
        briefing = robot_information("so101")["briefing"]
        for phrase in ("five serial revolute arm joints", "STS3215", "4096 ticks", "gripper_frame_link",
                       "quaternion_xyzw", "angles_deg poses are not supported", "0 requests closed",
                       "1 requests open", "public arm slot is right", "not reachable",
                       "nothing is paused", "joint_mapping", "not between the jaws", "No payload rating"):
            self.assertIn(phrase, briefing)
        self.assertNotRegex(briefing, r"(?i)\b[0-9a-f]{32,}\b|https?://|github|commit")

    def test_native_geometry_archive_is_complete_and_matches_kinematics(self):
        default = robot_information("so101")
        member = default["reference"]["geometry"]["assembly_member"]
        self.assertEqual(member, "so101_new_calib_camera.urdf")
        catalog = robot_resources("so101")
        self.assertEqual([entry["resource_id"] for entry in catalog["resources"]], ["geometry_model"])
        path = robot_resource_path("so101", "geometry_model")
        self.assertIsNotNone(path)
        self.assertLess(path.stat().st_size, 16*1024*1024)
        with zipfile.ZipFile(path) as archive:
            self.assertIsNone(archive.testzip())
            urdf = archive.read(member)
            self.assertEqual(default["urdf"].encode("utf-8"), urdf)
            self.assertNotIn(b"<!--", urdf)
            root = ET.fromstring(urdf)
            meshes = {node.get("filename") for node in root.findall(".//mesh")}
            self.assertEqual(set(archive.namelist()), meshes | {member})
            for mesh in meshes:
                self.assertTrue(mesh.startswith("assets/") and ".." not in mesh and mesh.endswith(".stl"))
                data = archive.read(mesh)
                self.assertEqual(data[:80], b"\0"*80)
                count = struct.unpack_from("<I", data, 80)[0]
                self.assertGreater(count, 0)
                self.assertEqual(len(data), 84 + 50*count)
            links = {link.get("name") for link in root.findall("link")}
            self.assertTrue({"base_link", "gripper_link", "gripper_frame_link", "wrist_camera_mount_link",
                             "wrist_camera_link"} <= links)
            joints = {joint.get("name") for joint in root.findall("joint")}
            self.assertTrue(set(kin.ARM_JOINTS) | {kin.GRIPPER_JOINT} <= joints)
        full = robot_information("so101", "all")["reference"]
        self.assertEqual(full["kinematics"]["arm_joint_order"], list(kin.ARM_JOINTS))
        self.assertEqual(full["kinematics"]["pose_interface"]["tcp_link"], kin.TCP_LINK)
        self.assertEqual(full["kinematics"]["wrist_camera_parent_link"], kin.WRIST_CAMERA_PARENT_LINK)
        limits = kin.joint_limits_rad(kin.load_model())
        for joint in full["kinematics"]["joints"]:
            if joint["type"] == "revolute":
                self.assertEqual(tuple(joint["declared_limits_rad"]), limits[joint["name"]])

    def test_startup_text_and_audit_sidecar_stay_out_of_model_output(self):
        record = robot_information("so101")
        text = robot_information_text(record)
        self.assertEqual(text.count(record["urdf"]), 1)
        for section in ROBOT_INFORMATION_SECTIONS:
            encoded = json.dumps(robot_information("so101", section))
            self.assertNotRegex(encoded, r"(?i)upstream|sha256|a758567c|github")


if __name__ == "__main__":
    unittest.main()
