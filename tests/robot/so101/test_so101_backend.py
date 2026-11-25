"""SO-101 real-hardware backend on CPU: fake driver, fake cameras, real kinematics.

No serial port or camera is opened. These tests prove the software contract
(compilation, validation, receipts, stop/unknown semantics, telemetry and
operator commands), not physical motion quality.
"""

NI_RUA_TEST_TIER = "component"

# === tests/test_so101_backend.py ===

from copy import deepcopy
import json
import math
from pathlib import Path
import functools
from support.paths import CONFIGS_ROOT, REPO_ROOT, TESTS_ROOT
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from ni_rua.core.config import Config
from ni_rua.core.contracts import validate_observation
from ni_rua.robot.backends.factory import create_backend
from ni_rua.robot.backends.so101 import contract
from ni_rua.robot.backends.so101.backend import RobotBackend
from ni_rua.robot.backends.so101.deployment import embodiment_profile, load_deployment, validate_deployment
from ni_rua.robot.backends.so101.driver import FakeArmDriver, FakeCameraSource, FakeClock
from ni_rua.robot.backends.so101.worker import SO101Adapter, pose_matrix

ROOT = REPO_ROOT
EXAMPLE = CONFIGS_ROOT / "so101.example.json"
IDENTITY = dict(run_id="run", episode_id="ep", env_instance_id="env", env_generation=1)


@functools.lru_cache(maxsize=None)
def _example_document_text():
    return EXAMPLE.read_text()


@functools.lru_cache(maxsize=None)
def _default_deployment():
    return validate_deployment(json.loads(_example_document_text()), base=EXAMPLE.parent)


def deployment(**updates):
    if not updates:
        return deepcopy(_default_deployment())
    document = json.loads(_example_document_text())
    for dotted, value in updates.items():
        section, key = dotted.split(".")
        document[section][key] = value
    return validate_deployment(document, base=EXAMPLE.parent)


def make_adapter(directory, *, profile="rgb_proprio_v1", deploy=None, start_telemetry=False, **driver_options):
    deploy = deploy or deployment()
    clock = FakeClock()
    driver = FakeArmDriver(deploy, clock, **driver_options)
    cameras = {name: FakeCameraSource(name, clock=clock, width=deploy["cameras"][name]["width"] // 10,
                                      height=deploy["cameras"][name]["height"] // 10) for name in ("fixed", "wrist")}
    adapter = SO101Adapter(deploy, directory, profile, driver=driver, cameras=cameras, clock=clock,
                           image_settings=Config().images.to_dict(), start_telemetry=start_telemetry)
    return adapter, driver, clock


def move_action(adapter, dx=0.02, **path):
    pose = adapter.robot_state(adapter.read())["arms"]["right"]["ee_pose"]
    args = dict(arm="right", position_m=[pose["position_m"][0] + dx, *pose["position_m"][1:]],
                quaternion_xyzw=pose["quaternion_xyzw"], observation_id="obs")
    action = contract.compile_action("move", args, IDENTITY, adapter.capabilities())
    action["path"].update(path)
    return action


class DeploymentTests(unittest.TestCase):
    def test_example_loads_and_every_field_is_required(self):
        loaded = load_deployment(EXAMPLE)
        self.assertEqual(loaded["schema"], "ni_rua.so101.deployment.v1")
        document = json.loads(EXAMPLE.read_text())
        del document["motion"]["max_joint_speed_rad_s"]
        with self.assertRaisesRegex(ValueError, "motion: missing=\\['max_joint_speed_rad_s'\\]"):
            validate_deployment(document, base=EXAMPLE.parent)

    def test_units_and_bounds_are_enforced(self):
        for dotted, value, message in (("motion.max_joint_speed_rad_s", 0, "out_of_range"),
                                       ("telemetry.joint_hz", -1, "out_of_range"),
                                       ("home.q_rad", [0, 0, 0], "home.q_rad"),
                                       ("workspace.min_m", [1, 1, 1], "workspace"),
                                       ("ik.max_iterations", 1.5, "integer")):
            with self.subTest(field=dotted), self.assertRaisesRegex(ValueError, message):
                deployment(**{dotted: value})


class SelectionAndCompilationTests(unittest.TestCase):
    def config(self, **backend):
        return Config.from_dict({"backend": {"environment": "so101_real", "robot_model_id": "so101",
                                             "deployment": str(EXAMPLE), **backend},
                                 "features": {"observation_profile": "rgb_proprio_v1"}})

    def test_factory_accepts_so101_pair_and_rejects_others(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(RobotBackend, "start_process", side_effect=AssertionError("unexpected launch")):
            backend = create_backend(config=self.config(), run_dir=directory, task="ignored", gpu_id=0, seed=1, layout_id=2)
            self.assertIsInstance(backend, RobotBackend)
            self.assertEqual(backend.python, str(Path("~/miniforge3/envs/so101/bin/python").expanduser()))
            for backend_section, error in (({"robot_model_id": "arx-x5"}, "unsupported_so101_real_robot"),
                                           ({"deployment": None}, "so101_deployment_required")):
                with self.subTest(backend=backend_section), self.assertRaisesRegex(ValueError, error):
                    create_backend(config=self.config(**backend_section), run_dir=directory)
            with self.assertRaisesRegex(ValueError, "unsupported_robodojo_robot"):
                create_backend(config=Config.from_dict({"backend": {"environment": "robodojo", "robot_model_id": "so101"}}))
            calibrated = Config.from_dict({"backend": self.config().backend.to_dict(),
                                           "features": {"observation_profile": "rgb_proprio_calibrated_v1"}})
            with self.assertRaisesRegex(ValueError, "so101_camera_calibration_missing"):
                create_backend(config=calibrated, run_dir=directory)

    def test_worker_command_uses_deployment_interpreter(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = create_backend(config=self.config(), run_dir=directory)
            with patch.object(backend, "start_process", return_value={"ready": True}) as launch:
                backend.start()
            command = launch.call_args.args[0]
            self.assertEqual(command[0], backend.python)
            self.assertEqual(command[command.index("-m") + 1], "ni_rua.robot.backends.so101.worker")
            self.assertEqual(command[command.index("--observation-profile") + 1], "rgb_proprio_v1")

    def test_compile_rejects_left_arm_and_x5_grasp_convention(self):
        caps = dict(frame_id="so101_base", tcp_config_id="so101-gripper-frame-v1", limits_profile=contract.LIMITS_PROFILE,
                    default_timeout_s=60., motion_tolerance_defaults=dict(position_tolerance_m=.01, orientation_tolerance_rad=.15))
        pose = dict(position_m=[.25, 0, .1], quaternion_xyzw=[0, 0, 0, 1], observation_id="obs")
        with self.assertRaisesRegex(ValueError, "unsupported_arm:left"):
            contract.compile_action("move", dict(pose, arm="left"), IDENTITY, caps)
        with self.assertRaisesRegex(ValueError, "unsupported_capability: angles_deg"):
            contract.compile_action("grasp", dict(arm="right", grasp_position_m=[.2, 0, .1], angles_deg=[0, 0, 0],
                                                  observation_id="obs"), IDENTITY, caps)
        with self.assertRaisesRegex(ValueError, "unsupported_capability: angles_deg"):
            contract.compile_action("move", dict(arm="right", grasp_position_m=[.2, 0, .1], angles_deg=[0, 0, 0],
                                                 observation_id="obs"), IDENTITY, caps)
        action = contract.compile_action("move", dict(pose, arm="right"), IDENTITY, caps)
        deploy = deployment()
        self.assertEqual(contract.validate_action(action, deploy), ("right", None))
        for change, error in ((lambda a: a["path"]["waypoints"][0].update(left=a["path"]["waypoints"][0].pop("right")), "unsupported_arm"),
                              (lambda a: a["path"]["waypoints"][0]["right"].update(position_m=[5, 0, 0]), "out_of_workspace"),
                              (lambda a: a["path"]["waypoints"][0]["right"].update(quaternion_xyzw=[0, 0, 0, 2]), "normalized"),
                              (lambda a: a.update(frame_id="env_origin"), "absolute pose in so101_base"),
                              (lambda a: a.update(tcp_config_id="arx-x5-link6-v1"), "TCP must be"),
                              (lambda a: a.update(timeout_s=10_000), "timeout_s")):
            candidate = deepcopy(action)
            change(candidate)
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                contract.validate_action(candidate, deploy)


class WorkerExecutionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def test_capabilities_are_single_arm_real_time(self):
        adapter, _, _ = make_adapter(self.directory.name)
        caps = adapter.capabilities()
        self.assertEqual((caps["arms"], caps["dual_arm"], caps["max_simultaneous_arms"]), (["right"], False, 1))
        self.assertEqual(caps["world_kind"], "observed")
        self.assertIn("real time", caps["clock_policy"])
        self.assertTrue(caps["observer_video"])
        self.assertIsNone(caps["native_evaluator"])
        self.assertFalse(caps["motion_preview"]["available"])
        self.assertEqual(caps["unified_action"]["requires_camera_calibration"], False)
        self.assertEqual(caps["runtime_limits"]["speed_scale_cap"], 0.9)
        from ni_rua.core.observation_policy import require_actor_observation
        require_actor_observation(caps, "rgb_proprio_v1")

    def test_move_completes_with_measured_residuals_and_effective_tolerances(self):
        adapter, driver, _ = make_adapter(self.directory.name)
        action = move_action(adapter, position_tolerance_m=0.004, orientation_tolerance_rad=0.1)
        receipt = adapter.execute(action, lambda update: None, threading.Event())
        self.assertEqual(receipt["status"], "completed", receipt.get("reason"))
        self.assertLessEqual(receipt["position_error_m"], 0.004)
        self.assertEqual(receipt["effective_tolerances"]["position_tolerance_m"], 0.004)
        self.assertEqual(receipt["effective_limits"]["speed_scale_cap"], 0.9)
        result = receipt["waypoint_results"][0]
        self.assertTrue(result["within_tolerance"] and result["settled"])
        self.assertGreater(receipt["executed_native_steps"], 1)
        self.assertGreater(receipt["wall_clock_duration_s"], 0)
        speed = adapter.deployment["motion"]["max_joint_speed_rad_s"] * adapter.speed_scale_cap
        rate = adapter.deployment["motion"]["command_rate_hz"]
        goals = np.asarray([goal for goal, _ in driver.goals[1:]])
        self.assertLessEqual(np.abs(np.diff(goals, axis=0)).max(), speed / rate + 1e-9)
        self.assertIsNotNone(receipt["terminal_observation"])


    def test_gravity_sag_is_corrected_reported_and_bounded(self):
        sag = (0., 0., 0.04, 0., 0.)  # elbow settles 0.04 rad short, as measured on hardware
        deploy = deployment(**{"motion.arrival_correction_rounds": 0})
        adapter, _, _ = make_adapter(self.directory.name, deploy=deploy, sag_rad=sag)
        action = move_action(adapter, position_tolerance_m=0.004, orientation_tolerance_rad=0.1)
        uncorrected = adapter.execute(action, lambda update: None, threading.Event())
        self.assertEqual(uncorrected["reason"], "tracking_error")
        self.assertEqual(uncorrected["waypoint_results"][0]["arrival_correction"]["rounds"], [])

        adapter, driver, _ = make_adapter(tempfile.mkdtemp(dir=self.directory.name), sag_rad=sag)
        action = move_action(adapter, position_tolerance_m=0.004, orientation_tolerance_rad=0.1)
        receipt = adapter.execute(action, lambda update: None, threading.Event())
        self.assertEqual(receipt["status"], "completed", receipt.get("reason"))
        rounds = receipt["waypoint_results"][0]["arrival_correction"]["rounds"]
        self.assertTrue(rounds)
        correction = receipt["waypoint_results"][0]["arrival_correction"]
        self.assertGreater(correction["initial_tcp_residual"]["position_error_m"],
                           rounds[-1]["tcp_residual"]["position_error_m"])
        self.assertIn(correction["termination_reason"], ("round_limit", "joint_error_below_correction_threshold"))
        self.assertAlmostEqual(rounds[-1]["goal_offset_rad"][2], 0.04, delta=0.006)
        self.assertLessEqual(abs(rounds[-1]["settled_joint_error_rad"][2]), 0.006)
        # Hold re-applies the learned offset instead of letting the elbow sag again.
        before = driver.read()["q_rad"][2]
        stop = adapter._stop()
        self.assertTrue(stop["confirmed"])
        self.assertAlmostEqual(stop["hold_offset_rad"][2], 0.04, delta=0.006)
        self.assertAlmostEqual(driver.read()["q_rad"][2], before, delta=0.006)

    def test_sag_correction_offset_is_capped_and_failure_stays_honest(self):
        deploy = deployment(**{"motion.arrival_correction_max_offset_rad": 0.01})
        adapter, _, _ = make_adapter(self.directory.name, deploy=deploy, sag_rad=(0., 0., 0.04, 0., 0.))
        action = move_action(adapter, position_tolerance_m=0.004, orientation_tolerance_rad=0.1)
        receipt = adapter.execute(action, lambda update: None, threading.Event())
        self.assertEqual(receipt["reason"], "tracking_error")
        rounds = receipt["waypoint_results"][0]["arrival_correction"]["rounds"]
        self.assertTrue(all(abs(v) <= 0.01 + 1e-9 for r in rounds for v in r["goal_offset_rad"]))

    def test_unreachable_tolerance_fails_planning_before_any_command(self):
        adapter, driver, _ = make_adapter(self.directory.name)
        pose = adapter.robot_state(adapter.read())["arms"]["right"]["ee_pose"]
        # Rotate 90 deg about the base x axis: generally outside the 5-DOF manifold here.
        target = pose_matrix(pose)
        rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) @ target[:3, :3]
        from ni_rua.robot.backends.so101.worker import matrix_to_quaternion
        action = move_action(adapter, dx=0.0, orientation_tolerance_rad=0.01)
        action["path"]["waypoints"][0]["right"]["quaternion_xyzw"] = matrix_to_quaternion(rotation)
        sends = driver.sends
        receipt = adapter.execute(action, lambda update: None, threading.Event())
        self.assertEqual((receipt["status"], receipt["reason"], receipt["physics_advanced"]), ("failed", "planning_failed", False))
        self.assertGreater(receipt["ik_result"]["orientation_error_rad"], 0.01)
        self.assertEqual(driver.sends, sends)

    @staticmethod
    def copied_quaternion_pose(adapter, dy=0.08):
        # Real failure (run 6d8a8065): the measured quaternion reused at a laterally shifted target.
        # Gripper yaw follows shoulder_pan on the 5-DOF arm, so this exact 6D pose is unreachable.
        pose = adapter.robot_state(adapter.read())["arms"]["right"]["ee_pose"]
        return dict(position_m=[pose["position_m"][0], pose["position_m"][1] + dy, pose["position_m"][2] - 0.05],
                    quaternion_xyzw=list(pose["quaternion_xyzw"]))

    def test_every_free_orientation_spelling_solves_position_only(self):
        # Regression: the schema maximum (3.14159265) and the unconstrained_orientation preset (3.14159)
        # were below math.pi, so position-only IK was unreachable and the plan stayed ~24 mm off target.
        from ni_rua.robot.motion.motion_tolerance import MAX_ORIENTATION_TOLERANCE_RAD, TOLERANCE_PRESETS
        adapter, driver, _ = make_adapter(self.directory.name)
        target = self.copied_quaternion_pose(adapter)
        seed = adapter.read()["q_rad"]
        preset = TOLERANCE_PRESETS["unconstrained_orientation"]["orientation_tolerance_rad"]
        for tolerance in (MAX_ORIENTATION_TOLERANCE_RAD, preset, 3.14159265, math.pi):
            _, ik = adapter._ik(target, seed, (0.015, tolerance))
            self.assertTrue(ik["ok"], (tolerance, ik))
            self.assertFalse(ik["orientation_checked"])
            self.assertLess(ik["position_error_m"], 1e-3)
            effective = adapter._effective_tolerances(dict(position_tolerance_m=0.015, orientation_tolerance_rad=tolerance))
            self.assertFalse(effective["orientation_checked"], tolerance)
        action = move_action(adapter, dx=0.0, position_tolerance_m=0.015, orientation_tolerance_rad=3.14159265)
        action["path"]["waypoints"][0]["right"] = target
        receipt = adapter.execute(action, lambda update: None, threading.Event())
        self.assertEqual(receipt["status"], "completed", receipt)
        self.assertLessEqual(receipt["position_error_m"], 0.015)
        self.assertGreater(driver.sends, 0)

    def test_position_priority_fallback_honours_requested_orientation_tolerance(self):
        adapter, driver, _ = make_adapter(self.directory.name)
        target = self.copied_quaternion_pose(adapter)
        seed = adapter.read()["q_rad"]
        weighted = adapter.kin.ik(adapter.model, pose_matrix(target), seed, orientation_weight=0.3,
                                  limits=adapter.limits, position_tolerance_m=0.01, orientation_tolerance_rad=0.6)
        self.assertGreater(weighted["position_error_m"], 0.01, "fixture must reproduce the weighted-cost trade-off")
        # A relaxed but bounded tolerance used to change nothing; now the position-first solve meets it.
        _, ik = adapter._ik(target, seed, (0.01, 0.6))
        self.assertEqual((ik["ok"], ik["solve"], ik["orientation_checked"]), (True, "position_priority", True), ik)
        self.assertLess(ik["position_error_m"], 1e-3)
        self.assertLessEqual(ik["orientation_error_rad"], 0.6)
        self.assertNotIn("position_priority", ik)
        # Too strict for any orientation at this position: fail before any command, report what is reachable.
        sends = driver.sends
        action = move_action(adapter, dx=0.0, position_tolerance_m=0.01, orientation_tolerance_rad=0.15)
        action["path"]["waypoints"][0]["right"] = target
        receipt = adapter.execute(action, lambda update: None, threading.Event())
        self.assertEqual((receipt["status"], receipt["reason"], receipt["physics_advanced"]), ("failed", "planning_failed", False))
        self.assertEqual(driver.sends, sends)
        reachable = receipt["ik_result"]["position_priority"]
        self.assertEqual(set(reachable), {"quaternion_xyzw", "position_error_m", "orientation_error_rad"})
        self.assertLess(reachable["position_error_m"], 1e-3)
        self.assertGreater(reachable["orientation_error_rad"], 0.15)
        self.assertNotIn("solve", receipt["ik_result"])
        # Reusing the reported quaternion at the same position plans and executes with the default tolerance.
        action["path"]["waypoints"][0]["right"] = dict(position_m=target["position_m"], quaternion_xyzw=reachable["quaternion_xyzw"])
        receipt = adapter.execute(action, lambda update: None, threading.Event())
        self.assertEqual(receipt["status"], "completed", receipt)

    def test_short_move_under_sag_is_not_a_stall(self):
        # Real run: the elbow was commanded 0.013 rad in the stall window but sagged so it measured 0.006 rad;
        # that is servo deadband/sag (fixed by arrival correction), not a blocked joint.
        adapter, driver, _ = make_adapter(self.directory.name, sag_rad=(0., 0., 0.01, 0., 0.), tau_s=0.3)
        action = move_action(adapter, dx=0.004, position_tolerance_m=0.004, orientation_tolerance_rad=0.2)
        receipt = adapter.execute(action, lambda update: None, threading.Event())
        self.assertEqual(receipt["status"], "completed", receipt.get("waypoint_results"))
        self.assertNotEqual(receipt["waypoint_results"][0].get("kind"), "stall")

    def test_stall_is_tracking_error_never_completed(self):
        adapter, _, _ = make_adapter(self.directory.name, blocked=(1, 2))
        receipt = adapter.execute(move_action(adapter, dx=0.06), lambda update: None, threading.Event())
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["reason"], "tracking_error")
        self.assertIn(receipt["waypoint_results"][0]["kind"], ("stall", "joint_tracking_limit"))
        self.assertTrue(receipt["stop_confirmation"]["confirmed"])

    def test_driver_exception_mid_motion_is_unknown(self):
        adapter, driver, _ = make_adapter(self.directory.name)
        driver.fail_on_send = driver.sends + 4
        receipt = adapter.execute(move_action(adapter, dx=0.05), lambda update: None, threading.Event())
        self.assertEqual(receipt["status"], "unknown")
        self.assertTrue(receipt["execution_uncertain"])
        self.assertIn("driver_write_failed", receipt["reason"])
        from ni_rua.core.presentation.feedback_delivery import execution_unconfirmed
        self.assertTrue(execution_unconfirmed(receipt))

    def test_cancellation_stops_with_measured_confirmation(self):
        adapter, _, _ = make_adapter(self.directory.name)
        cancel = threading.Event()

        def progress(update):
            if update.get("phase") == "executing":
                cancel.set()
        receipt = adapter.execute(move_action(adapter, dx=0.05), progress, cancel)
        self.assertEqual((receipt["status"], receipt["reason"]), ("stopped", "cancelled"))
        self.assertTrue(receipt["stop_confirmation"]["confirmed"])
        self.assertEqual(receipt["stop_confirmation"]["method"], "measured_position_goal_then_measured_settling")

    def test_unsettled_stop_is_unknown(self):
        adapter, driver, _ = make_adapter(self.directory.name)
        cancel = threading.Event()
        original = driver.read

        def noisy():
            value = original()
            value["q_rad"][0] += 0.05 * math.sin(driver.clock.now() * 40)
            return value
        driver.read = noisy
        cancel.set()
        receipt = adapter.execute(move_action(adapter, dx=0.05), lambda update: None, cancel)
        self.assertEqual(receipt["status"], "stopped")
        self.assertFalse(receipt["stop_confirmation"]["confirmed"])
        self.assertEqual(receipt["stop_state"], "unknown")
        from ni_rua.core.presentation.feedback_delivery import execution_unconfirmed
        self.assertTrue(execution_unconfirmed(receipt))

    def test_gripper_and_wait_report_measurements_and_wall_clock(self):
        adapter, _, _ = make_adapter(self.directory.name, jaw_min_normalized=0.3)
        caps = adapter.capabilities()
        jaw = contract.compile_action("gripper", dict(arm="right", goal_normalized=0, observation_id="o"), IDENTITY, caps)
        receipt = adapter.execute(jaw, lambda update: None, threading.Event())
        self.assertEqual(receipt["status"], "completed")
        execution = receipt["gripper_execution"]["right"]
        self.assertTrue(execution["command_complete"])
        self.assertAlmostEqual(execution["measured_normalized"], 0.3, places=2)
        wait = contract.compile_action("wait", dict(native_steps=10, observation_id="o"), IDENTITY, caps)
        receipt = adapter.execute(wait, lambda update: None, threading.Event())
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["wait"]["executed_steps"], 10)
        self.assertAlmostEqual(receipt["wait"]["measured_wall_clock_s"], 10 / caps["native_control_hz"], places=3)

    def test_reset_moves_home_only(self):
        adapter, driver, _ = make_adapter(self.directory.name)
        result = adapter.reset(seed=3, layout_id=4)
        self.assertEqual(result["reset_strategy"], "arm_home_only_world_unchanged")
        self.assertFalse(result["world_reset"])
        np.testing.assert_allclose(driver.q, adapter.deployment["home"]["q_rad"], atol=adapter.deployment["home"]["arrival_tolerance_rad"])


class ObservationTests(unittest.TestCase):
    def publish(self, adapter, profile):
        from ni_rua.core.storage.store import Store
        from ni_rua.runtime.harness import Harness
        directory = tempfile.mkdtemp(dir=adapter.run_dir)
        store = Store(Path(directory) / "store", IDENTITY)
        self.addCleanup(store.close)
        harness = Harness(store, InProcessBackend(adapter), observation_profile=profile)
        bundle = harness.start()
        self.addCleanup(adapter.close)
        validate_observation(bundle)
        return harness, bundle

    def test_uncalibrated_observation_validates_and_declares_async_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter, _, _ = make_adapter(directory)
            raw = adapter.observe()
            self.assertNotIn("camera_calibration", raw)
            self.assertEqual(raw["step_domain"], "so101_control_step")
            self.assertIn("asynchronous", raw["cameras"][0]["capture_alignment"])
            _, bundle = self.publish(adapter, "rgb_proprio_v1")
            self.assertIsNone(bundle["calibration_id"])
            self.assertEqual(bundle["robot_model_id"], "so101")

    def test_calibrated_observation_publishes_kinematic_wrist_pose(self):
        from ni_rua.robot.calibration.workflow import charuco as cal
        with tempfile.TemporaryDirectory() as directory:
            deploy = deployment()
            K = [[50., 0, 32], [0, 50., 24], [0, 0, 1]]
            mount = np.eye(4)
            mount[:3, 3] = [0.01, 0.02, 0.03]
            fixed = np.eye(4)
            fixed[:3, 3] = [0.5, 0, 0.4]
            path = Path(directory) / "camera_calibration.json"
            size = {name: [deploy["cameras"][name]["width"] // 10, deploy["cameras"][name]["height"] // 10] for name in ("wrist", "fixed")}
            cal.write_calibration_file(path, board=cal.DEFAULT_BOARD_SPEC, samples=20, cameras={
                "wrist": dict(K=K, distortion=[0] * 5, image_size=size["wrist"], T_link_camera=mount.tolist(),
                              intrinsics_rms_px=.3, hand_eye_residual=dict(translation_m=.001, rotation_rad=.002)),
                "fixed": dict(K=K, distortion=[0] * 5, image_size=size["fixed"], T_base_camera=fixed.tolist(),
                              intrinsics_rms_px=.3, extrinsic_residual=dict(reprojection_px=.5))})
            deploy["camera_calibration_file"] = str(path)
            for name in ("wrist", "fixed"):
                deploy["cameras"][name].update(width=size[name][0], height=size[name][1])
            adapter, _, _ = make_adapter(directory, profile="rgb_proprio_calibrated_v1", deploy=deploy)
            adapter.cameras = {name: FakeCameraSource(name, clock=adapter.clock, width=size[name][0], height=size[name][1])
                               for name in ("wrist", "fixed")}
            raw = adapter.observe()
            sample = raw["camera_pose_sample"]["cameras"]
            q = raw["robot_state"]["arms"]["right"]["q_rad"][:5]
            expected = np.asarray(adapter.kin.fk(adapter.model, q, link="gripper_link")) @ mount
            np.testing.assert_allclose(sample["wrist"]["T_world_camera"], expected, atol=1e-9)
            np.testing.assert_allclose(sample["fixed"]["T_world_camera"], fixed, atol=1e-12)
            harness, bundle = self.publish(adapter, "rgb_proprio_calibrated_v1")
            self.assertTrue(bundle["calibration_id"])
            from ni_rua.robot.model.robot_calibration_information import robot_calibration_information
            info = robot_calibration_information(harness)
            self.assertTrue(info["available"], info)
            self.assertEqual(info["reference_frame_id"], "so101_base")

    def test_calibrated_observation_end_to_end_downstream_geometry_and_perception(self):
        from ni_rua.perception.external.object_table import default_camera, effective_support_z_m, object_table_query
        from ni_rua.perception.overlays.trajectory_overlay import deployable_camera, render_trajectory_overlay
        from ni_rua.robot.calibration.workflow import charuco as cal
        from ni_rua.robot.model import so101_kinematics as kin
        from ni_rua.runtime.actions.unified_actions import dispatch
        from ni_rua.runtime.tools.plugins.stereo_guide import make_stereo_guide_tool, read_stereo_guide
        from support.perception_fixtures import FakeService, fake_object
        with tempfile.TemporaryDirectory() as directory:
            deploy = deployment()
            K = [[510., 0., 320.], [0., 509., 240.], [0., 0., 1.]]
            D = [-0.015, 0.224, -0.010, 0.015, -0.386]
            model = kin.load_model()
            t_gripper = np.asarray(kin.fk(model, FakeArmDriver(deploy, FakeClock()).q, link="gripper_link"))
            r_cam = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
            fixed = np.eye(4); fixed[:3, :3] = r_cam; fixed[:3, 3] = [0.25, -0.45, 0.25]
            wrist_world = np.eye(4); wrist_world[:3, :3] = r_cam; wrist_world[:3, 3] = [0.10, -0.45, 0.25]
            mount = np.linalg.inv(t_gripper) @ wrist_world
            path = Path(directory) / "camera_calibration.json"
            size = {name: [640, 480] for name in ("wrist", "fixed")}
            cal.write_calibration_file(path, board=cal.DEFAULT_BOARD_SPEC, samples=20, cameras={
                "wrist": dict(K=K, distortion=D, distortion_model="opencv_plumb_bob", image_size=size["wrist"],
                              T_link_camera=mount.tolist(), intrinsics_rms_px=.3,
                              hand_eye_residual=dict(translation_m=.001, rotation_rad=.002)),
                "fixed": dict(K=K, distortion=D, distortion_model="opencv_plumb_bob", image_size=size["fixed"],
                              T_base_camera=fixed.tolist(), intrinsics_rms_px=.3,
                              extrinsic_residual=dict(reprojection_px=.5))})
            doc = json.loads(path.read_text())
            doc["joint_calibration"] = {"table_z_m": -0.018}
            path.write_text(json.dumps(doc))
            deploy["camera_calibration_file"] = str(path)
            for name in ("wrist", "fixed"):
                deploy["cameras"][name].update(width=640, height=480)
            adapter, _, _ = make_adapter(directory, profile="rgb_proprio_calibrated_v1", deploy=deploy)
            adapter.cameras = {name: FakeCameraSource(name, clock=adapter.clock, width=640, height=480)
                               for name in ("wrist", "fixed")}

            from ni_rua.core.storage.store import Store
            from ni_rua.runtime.harness import Harness
            store = Store(Path(directory) / "store", IDENTITY)
            self.addCleanup(store.close)
            fake = FakeService(objects=[fake_object((0.25, 0.0, -0.005), 0.03, 0.0, label=(320, 380))])
            self.addCleanup(fake.close)
            config = Config.from_dict({
                "backend": {"environment": "so101_real", "robot_model_id": "so101", "deployment": str(EXAMPLE)},
                "features": {"observation_profile": "rgb_proprio_calibrated_v1", "object_table": True,
                             "vggt_endpoint": fake.url, "vggt_timeout_s": 5.0, "vggt_support_z_m": None},
            })
            stereo_spec = make_stereo_guide_tool(lambda args, corr: read_stereo_guide(harness, args, corr))
            harness = Harness(store, InProcessBackend(adapter), observation_profile="rgb_proprio_calibrated_v1",
                              config=config, tool_specs=(stereo_spec,))
            harness.thread_id = "so101-e2e"
            harness.task_planning.required = False
            bundle = harness.start()
            self.addCleanup(adapter.close)
            oid = bundle["observation_id"]

            # 1. harness.geometry.camera on both "fixed" and "wrist"
            cam_fixed = harness.geometry.camera(IDENTITY, oid, "fixed")
            cam_wrist = harness.geometry.camera(IDENTITY, oid, "wrist")
            self.assertEqual(cam_fixed["joint_calibration"]["table_z_m"], -0.018)
            np.testing.assert_allclose(cam_fixed["T"], fixed, atol=1e-9)
            np.testing.assert_allclose(cam_wrist["T"], wrist_world, atol=1e-6)

            # 2. geometry_query (pixel_ray, pixel_plane, project_points, triangulate)
            ray_res = harness.geometry.execute(dict(IDENTITY, kind="pixel_ray", observation_id=oid,
                                                    camera_id="fixed", pixel=[320., 380.]))
            self.assertEqual(ray_res["status"], "supported", ray_res)

            plane_res = harness.geometry.execute(dict(IDENTITY, kind="pixel_plane", observation_id=oid,
                                                      camera_id="fixed", pixel=[320., 380.],
                                                      plane_origin_m=[0., 0., -0.018], plane_normal=[0., 0., 1.],
                                                      target_camera_id="wrist"))
            self.assertEqual(plane_res["status"], "conditional", plane_res)
            pt3d = plane_res["value"]["point_m"]
            self.assertAlmostEqual(pt3d[2], -0.018, places=6)

            proj_fixed = harness.geometry.execute(dict(IDENTITY, kind="project_points", observation_id=oid,
                                                       camera_id="fixed", points_m=[pt3d]))
            proj_wrist = harness.geometry.execute(dict(IDENTITY, kind="project_points", observation_id=oid,
                                                       camera_id="wrist", points_m=[pt3d]))
            self.assertEqual(proj_fixed["status"], "supported", proj_fixed)
            self.assertEqual(proj_wrist["status"], "supported", proj_wrist)

            tri_res = harness.geometry.execute(dict(IDENTITY, kind="triangulate",
                                                    observation_ids=[oid], camera_id="fixed",
                                                    pixels_a=proj_fixed["value"]["pixels"],
                                                    camera_id_b="wrist",
                                                    pixels_b=proj_wrist["value"]["pixels"]))
            self.assertEqual(tri_res["status"], "conditional", tri_res)
            np.testing.assert_allclose(tri_res["value"]["points_m"][0], pt3d, atol=1e-4)

            # 3. deployable_camera and render_trajectory_overlay on both "fixed" and "wrist"
            pose = adapter.robot_state(adapter.read())["arms"]["right"]["ee_pose"]
            target_pos = [pose["position_m"][0] + 0.02, *pose["position_m"][1:]]
            preview = dispatch(harness, dict(
                observation_id=oid,
                stages=[dict(id="reach", kind="move", arm="right", position_m=target_pos,
                             quaternion_xyzw=pose["quaternion_xyzw"],
                             position_tolerance_m=0.005, orientation_tolerance_rad=0.1)],
            ), {"tool_call_id": "preview-e2e"})
            self.assertEqual(preview.data["status"], "planned", preview.data)
            self.assertEqual(preview.data["trajectory_overlay"]["status"], "available", preview.data["trajectory_overlay"])
            self.assertEqual(preview.data["trajectory_overlay"]["camera_id"], "fixed")

            import cv2
            trajectory_data = json.loads(store.resolve_artifact(preview.data["trajectory_ref"])[1])
            for cam_id in ("fixed", "wrist"):
                dep_cam = deployable_camera(store, IDENTITY, oid, cam_id, trajectory_data)
                self.assertEqual(dep_cam["frame"]["camera_id"], cam_id)
                _, raw_rgb = store.resolve_artifact(dep_cam["frame"]["rgb_ref"])
                img = cv2.imdecode(np.frombuffer(raw_rgb, np.uint8), cv2.IMREAD_COLOR)
                rendered_img, overlay_meta = render_trajectory_overlay(dep_cam, img, trajectory_data)
                self.assertEqual(overlay_meta["camera_id"], cam_id)
                self.assertEqual(rendered_img.ndim, 3)

            # 4. stereo_guide between "fixed" and "wrist"
            guide = harness.invoke("stereo_guide", dict(
                observation_id=oid, source_camera_id="fixed", target_camera_id="wrist",
                source_pixel=proj_fixed["value"]["pixels"][0],
            ), dict(tool_call_id="stereo-e2e"), role="actor")
            self.assertTrue(guide.success, guide.contract_error)
            self.assertEqual(guide.data["status"], "guided")

            # 5. effective_support_z_m & object_table_query auto-selecting fixed and support_z_m == -0.018
            self.assertEqual(default_camera(harness, bundle), "fixed")
            self.assertEqual(effective_support_z_m(harness.config.features, cam_fixed), -0.018)
            ot_res, ot_overlays = object_table_query(harness, {"kind": "object_table", "observation_id": oid})
            self.assertEqual(ot_res["status"], "conditional", ot_res)
            self.assertEqual(ot_res["value"]["camera_id"], "fixed")
            self.assertEqual(ot_res["value"]["support_z_m"], -0.018)
            self.assertEqual(fake.requests[0][1]["support_z_m"], -0.018)
            self.assertEqual([name for name, _, _ in ot_overlays], ["fixed"])


class InProcessBackend:
    """Harness-facing wrapper around the adapter, as the JSONL parent would be."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.observation_profile = adapter.profile
        self.compile_action = staticmethod(contract.compile_action)

    def validate_action(self, action):
        return contract.validate_action(action, self.adapter.deployment)

    def embodiment_profile(self):
        """The run's ``EmbodimentProfile``, as the real backend hook builds it (without it the Harness would fall back
        to the ARX-X5 default robot)."""
        return embodiment_profile(self.adapter.deployment)

    def start(self):
        return self.adapter.capabilities()

    def observe(self):
        return self.adapter.observe()

    def record(self, enabled=True):
        return self.adapter.motion_video.record(enabled)

    def save_video(self):
        return self.adapter.motion_video.save()

    def preview_batch(self, action):
        self.validate_action(action)
        return self.adapter.preview_batch(action)

    def execute(self, action, progress_callback=None, cancel_flag=None):
        self.validate_action(action)
        return self.adapter.execute(action, progress_callback or (lambda update: None), cancel_flag or threading.Event())

    def reset(self, seed=0, layout_id=0):
        return self.adapter.reset(seed=seed, layout_id=layout_id)


class HarnessRobotActionTests(unittest.TestCase):
    def harness(self, **driver_options):
        from ni_rua.core.storage.store import Store
        from ni_rua.runtime.harness import Harness
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        adapter, driver, _ = make_adapter(directory.name, **driver_options)
        store = Store(Path(directory.name) / "store", IDENTITY)
        self.addCleanup(store.close)
        config = Config.from_dict({"backend": {"environment": "so101_real", "robot_model_id": "so101", "deployment": str(EXAMPLE)},
                                   "features": {"observation_profile": "rgb_proprio_v1", "record_video": False}})
        harness = Harness(store, InProcessBackend(adapter), observation_profile="rgb_proprio_v1", config=config)
        harness.start()
        self.addCleanup(adapter.close)
        return harness, adapter, driver

    def stages(self, harness, adapter, dx=0.02):
        pose = adapter.robot_state(adapter.read())["arms"]["right"]["ee_pose"]
        target = [pose["position_m"][0] + dx, *pose["position_m"][1:]]
        return [dict(id="reach", kind="move", arm="right", position_m=target, quaternion_xyzw=pose["quaternion_xyzw"],
                     position_tolerance_m=0.005, orientation_tolerance_rad=0.1),
                dict(id="look", kind="observe", reason="verify arrival before closing"),
                dict(id="close", kind="gripper", arm="right", goal_normalized=0.0)]

    def test_robot_action_is_available_uncalibrated_and_rejects_left(self):
        harness, adapter, _ = self.harness()
        spec = harness.tool_registry.spec("robot_action", role="actor")
        self.assertIsNone(harness.tool_registry.unavailable(spec, harness, {}))
        from ni_rua.runtime.actions.unified_actions import compile_request
        stages = self.stages(harness, adapter)
        stages[0]["arm"] = "left"
        with self.assertRaisesRegex(ValueError, "robot_action_stage_requires_arm|unsupported_arm"):
            compile_request(harness, dict(observation_id=harness.latest["observation_id"], stages=stages))

    def test_batch_preview_checkpoint_resume_and_complete(self):
        from ni_rua.runtime.actions.unified_actions import dispatch
        harness, adapter, driver = self.harness()
        harness.task_planning.required = False
        request = dict(observation_id=harness.latest["observation_id"], stages=self.stages(harness, adapter))
        sends = driver.sends
        with patch("ni_rua.runtime.actions.inspection._trajectory_image", return_value=({"status": "unavailable"}, [])):
            preview = dispatch(harness, request, {"tool_call_id": "preview"})
            self.assertEqual(preview.data["status"], "planned", preview.data)
            self.assertEqual(driver.sends, sends, "preview must not command the arm")
            first = dispatch(harness, dict(mode="execute", plan_id=preview.data["plan_id"]), {"tool_call_id": "execute-1"})
            receipt = first.data["receipt"]
            self.assertEqual((receipt["status"], receipt["reason"]), ("stopped", "observation_checkpoint"), receipt)
            self.assertTrue(receipt["stop_confirmation"]["confirmed"])
            self.assertEqual(receipt["completed_stage_ids"], ["reach", "look"])
            self.assertLessEqual(receipt["stage_results"]["reach"]["position_error_m"], 0.005)
            second = dispatch(harness, dict(mode="execute", plan_id=preview.data["plan_id"]), {"tool_call_id": "execute-2"})
        receipt = second.data["receipt"]
        self.assertEqual(receipt["status"], "completed", receipt)
        self.assertEqual(receipt["completed_stage_ids"], ["reach", "look", "close"])

    def test_preview_free_orientation_and_reachable_quaternion_through_robot_action(self):
        # End to end through the actor tool path (shared schema, tolerance presets, SO-101 contract, worker).
        from ni_rua.runtime.actions.unified_actions import dispatch
        harness, adapter, driver = self.harness()
        harness.task_planning.required = False
        target = WorkerExecutionTests.copied_quaternion_pose(adapter)

        def preview(call_id, **tolerance):
            stage = dict(id="above", kind="move", arm="right", **target, **tolerance)
            request = dict(observation_id=harness.latest["observation_id"], stages=[stage])
            with patch("ni_rua.runtime.actions.inspection._trajectory_image", return_value=({"status": "unavailable"}, [])):
                response = dispatch(harness, request, {"tool_call_id": call_id})
            responses[call_id] = response
            return response.data

        responses = {}

        sends = driver.sends
        # The exact value the model sent in the failed run, and the named preset.
        for call_id, tolerance in (("max", dict(position_tolerance_m=0.015, orientation_tolerance_rad=3.14159265)),
                                   ("preset", dict(tolerance_preset="unconstrained_orientation"))):
            data = preview(call_id, **tolerance)
            self.assertEqual(data["status"], "planned", data)
            self.assertFalse(data["stage_plans"]["above"]["ik"]["orientation_checked"])
            self.assertLess(data["stage_plans"]["above"]["ik"]["position_error_m"], 1e-3)
        failed = preview("normal", tolerance_preset="normal")
        self.assertEqual((failed["status"], failed["reason"]), ("failed", "planning_failed"), failed)
        reachable = failed["failure_feedback"]["ik"]["position_priority"]
        self.assertLess(reachable["position_error_m"], 1e-3)
        # The actor-visible payload after presentation carries the same reachable quaternion.
        from ni_rua.core.presentation.actor_presentation import ActorPresentation
        wire = ActorPresentation(harness.store).present_response(responses["normal"])
        shown = [json.loads(item["text"]) for item in wire["contentItems"] if item["type"] == "inputText"]
        self.assertIn(reachable["quaternion_xyzw"], [item.get("failure_feedback", {}).get("ik", {}).get("position_priority", {})
                                                     .get("quaternion_xyzw") for item in shown], shown)
        target["quaternion_xyzw"] = reachable["quaternion_xyzw"]
        retried = preview("reuse", tolerance_preset="normal")
        self.assertEqual(retried["status"], "planned", retried)
        self.assertEqual(driver.sends, sends, "preview must not command the arm")

    def test_batch_stale_start_requires_new_preview(self):
        from ni_rua.core.presentation.actor_presentation import ActorPresentation
        from ni_rua.runtime.actions.unified_actions import dispatch
        for reference in ("preview", "checkpoint"):
            with self.subTest(reference=reference):
                harness, adapter, driver = self.harness()
                harness.task_planning.required = False
                request = dict(observation_id=harness.latest["observation_id"], stages=self.stages(harness, adapter))
                with patch("ni_rua.runtime.actions.inspection._trajectory_image", return_value=({"status": "unavailable"}, [])):
                    preview = dispatch(harness, request, {"tool_call_id": "preview"})
                    execute = dict(mode="execute", plan_id=preview.data["plan_id"])
                    if reference == "checkpoint":
                        paused = dispatch(harness, execute, {"tool_call_id": "pause"})
                        self.assertTrue(paused.data["receipt"]["stop_confirmation"]["confirmed"])
                    before = adapter.tcp(driver.q)[:3, 3]
                    driver.q[0] += 0.2  # Someone moved the arm before (re)execution.
                    after = adapter.tcp(driver.q)[:3, 3]
                    sends = driver.sends
                    result = dispatch(harness, execute, {"tool_call_id": "execute"})
                receipt = result.data["receipt"]
                self.assertEqual((receipt["status"], receipt["reason"]), ("rejected", "batch_plan_stale_state_requires_new_preview"))
                self.assertEqual(driver.sends, sends)
                self.assertFalse(receipt["physics_advanced"])
                wire = ActorPresentation(harness.store).present_response(result)
                shown = next(value["receipt"] for item in wire["contentItems"] if item["type"] == "inputText"
                             and "receipt" in (value := json.loads(item["text"])))
                for drift in (receipt["tcp_drift"], shown["tcp_drift"]):
                    self.assertEqual((drift["frame_id"], drift["reference"]), (contract.FRAME_ID, reference))
                    np.testing.assert_allclose(drift["reference_position_m"], before, atol=1e-6)
                    np.testing.assert_allclose(drift["measured_position_m"], after, atol=1e-6)
                    np.testing.assert_allclose(drift["delta_position_m"], after - before, atol=1e-6)

    def test_unknown_receipt_closes_harness_admission(self):
        harness, adapter, driver = self.harness()
        harness.task_planning.required = False
        driver.fail_on_send = driver.sends + 3
        action = harness._action_from_args("robot_wait", dict(native_steps=10, observation_id=harness.latest["observation_id"],
                                                              action_id="wait-1"))
        receipt = harness.execute(action)["receipt"]
        self.assertEqual(receipt["status"], "unknown")
        self.assertFalse(harness.accepting)


class OperatorAndTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def test_telemetry_schema_and_atomic_files(self):
        adapter, _, _ = make_adapter(self.directory.name)
        adapter.cameras["wrist"].read_after(0, 1)
        state = adapter.telemetry_tick()
        directory = Path(self.directory.name) / "telemetry"
        stored = json.loads((directory / "state.json").read_text())
        self.assertEqual(stored, json.loads(json.dumps(state)))
        self.assertEqual(stored["schema"], "ni_rua.so101.telemetry.v1")
        for key in ("seq", "captured_at", "monotonic_s", "connected", "torque_enabled", "q_rad", "q_goal_rad",
                    "gripper_normalized", "tcp_position_m", "tcp_quaternion_xyzw", "motor", "motion", "cameras", "faults"):
            self.assertIn(key, stored)
        self.assertEqual(set(stored["motion"]), {"active_action_id", "phase", "tracking_error_rad"})
        self.assertTrue((directory / "wrist.jpg").is_file())
        self.assertFalse(list(directory.glob(".*.tmp")))

    def test_telemetry_read_failure_is_reported_not_raised(self):
        adapter, driver, clock = make_adapter(self.directory.name)
        clock.sleep(1)
        driver.read = lambda: (_ for _ in ()).throw(OSError("bus gone"))
        state = adapter.telemetry_tick(frames=False)
        self.assertFalse(state["connected"])
        self.assertIsNone(state["q_rad"])
        self.assertTrue(any("telemetry_read_failed" in fault for fault in state["faults"]))

    def test_hold_preempts_running_execution(self):
        adapter, _, _ = make_adapter(self.directory.name)
        held = {}

        def progress(update):
            if update.get("phase") == "executing" and not held:
                held.update(adapter.operator("hold"))
        receipt = adapter.execute(move_action(adapter, dx=0.05), progress, threading.Event())
        self.assertEqual(held["status"], "held")
        self.assertEqual((receipt["status"], receipt["reason"]), ("stopped", "operator_hold"))

    def test_relax_latches_gate_until_torque_on_and_fresh_observation(self):
        adapter, driver, _ = make_adapter(self.directory.name)
        result = adapter.operator("relax")
        self.assertEqual((result["status"], result["torque_enabled"]), ("relaxed", False))
        self.assertFalse(driver.torque)
        caps = adapter.capabilities()
        wait = contract.compile_action("wait", dict(native_steps=2, observation_id="o"), IDENTITY, caps)
        self.assertEqual(adapter.execute(wait, lambda u: None, threading.Event())["reason"], "operator_relaxed_requires_torque_on")
        self.assertEqual(adapter.operator("go_home")["reason"], "operator_relaxed_requires_torque_on")
        on = adapter.operator("torque_on")
        self.assertEqual(on["status"], "torque_on")
        np.testing.assert_allclose(driver.goal, driver.q)
        self.assertEqual(adapter.execute(wait, lambda u: None, threading.Event())["reason"],
                         "fresh_observation_required_after_torque_on")
        adapter.observe()
        self.assertEqual(adapter.execute(wait, lambda u: None, threading.Event())["status"], "completed")

    def test_speed_cap_only_lowers_and_is_reported(self):
        adapter, driver, _ = make_adapter(self.directory.name)
        for bad in (1.5, 0.0, adapter.deployment["operator"]["min_speed_scale_cap"] / 2):
            with self.subTest(cap=bad), self.assertRaisesRegex(ValueError, "invalid_speed_scale_cap"):
                adapter.operator("set_runtime_limits", speed_scale_cap=bad)
        applied = adapter.operator("set_runtime_limits", speed_scale_cap=0.5)
        self.assertEqual(applied["runtime_limits"]["effective_max_joint_speed_rad_s"],
                         adapter.deployment["motion"]["max_joint_speed_rad_s"] * 0.5)
        self.assertEqual(adapter.capabilities()["runtime_limits"]["speed_scale_cap"], 0.5)
        start = len(driver.goals)
        receipt = adapter.execute(move_action(adapter, dx=0.02), lambda u: None, threading.Event())
        self.assertEqual(receipt["effective_limits"]["speed_scale_cap"], 0.5)
        goals = np.asarray([goal for goal, _ in driver.goals[start:]])
        rate = adapter.deployment["motion"]["command_rate_hz"]
        self.assertLessEqual(np.abs(np.diff(goals, axis=0)).max(),
                             adapter.deployment["motion"]["max_joint_speed_rad_s"] * 0.5 / rate + 1e-9)

    def test_go_home_matches_reset_motion(self):
        adapter, driver, _ = make_adapter(self.directory.name)
        result = adapter.operator("go_home")
        self.assertEqual((result["status"], result["world"]), ("completed", "objects_not_reset"))
        np.testing.assert_allclose(driver.q, adapter.deployment["home"]["q_rad"], atol=adapter.deployment["home"]["arrival_tolerance_rad"])


FAKE_WORKER = """
import sys, threading, time
sys.path.insert(0, {root!r}); sys.path.insert(0, {tests!r})
from ni_rua.robot.backends.so101 import worker
from ni_rua.robot.backends.so101.driver import Clock, FakeArmDriver, FakeCameraSource
from ni_rua.robot.backends.so101.deployment import load_deployment
from ni_rua.core.config import Config

class RealtimeFake(FakeArmDriver):
    pass

def factory(arguments):
    deploy = load_deployment({example!r})
    clock = Clock()
    return worker.SO101Adapter(deploy, {run_dir!r}, "rgb_proprio_v1", driver=RealtimeFake(deploy, clock, tau_s=0.02),
        cameras={{n: FakeCameraSource(n, clock=clock) for n in ("wrist", "fixed")}}, clock=clock,
        image_settings=Config().images.to_dict(), start_telemetry=True)
worker.serve(factory, None, worker.emit)
"""


class ProcessProtocolTests(unittest.TestCase):
    """Real JSONL subprocess with the fake driver: hold reaches a running execute."""

    def test_hold_over_ipc_while_execute_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "fake_worker.py"
            script.write_text(FAKE_WORKER.format(root=str(REPO_ROOT), tests=str(TESTS_ROOT), example=str(EXAMPLE),
                                                 run_dir=str(Path(directory) / "worker")))
            config = Config.from_dict({"backend": {"environment": "so101_real", "robot_model_id": "so101",
                                                   "deployment": str(EXAMPLE)},
                                       "features": {"observation_profile": "rgb_proprio_v1"}})
            backend = create_backend(config=config, run_dir=Path(directory) / "worker")
            self.addCleanup(backend.close)
            caps = backend.start_process([sys.executable, "-u", str(script)], cwd=directory, environment=None)
            self.assertEqual(caps["arms"], ["right"])
            backend.capabilities = caps
            self.assertTrue(backend.record()["enabled"])
            observation = backend.observe()
            pose = observation["robot_state"]["arms"]["right"]["ee_pose"]
            action = contract.compile_action("wait", dict(native_steps=50, observation_id="o"), IDENTITY, caps)
            outcome = {}
            started = threading.Event()
            runner = threading.Thread(target=lambda: outcome.update(
                backend.execute(action, progress_callback=lambda u: u.get("phase") == "started" and started.set())))
            runner.start()
            import time
            deadline = time.monotonic() + 5
            while not (Path(directory) / "worker" / "telemetry" / "state.json").exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(started.wait(timeout=10))
            held = backend.operator("hold")
            runner.join(timeout=20)
            self.assertFalse(runner.is_alive())
            self.assertEqual(held["status"], "held")
            self.assertEqual((outcome["status"], outcome["reason"]), ("stopped", "operator_hold"))
            self.assertLess(outcome["wait"]["executed_steps"], 50)
            state = json.loads((Path(directory) / "worker" / "telemetry" / "state.json").read_text())
            self.assertEqual(state["schema"], "ni_rua.so101.telemetry.v1")
            self.assertIsNotNone(pose)
            video = backend.save_video()
            self.assertEqual(video["mode"], "action_execution_only")
            self.assertEqual(len(video["segments"]), 1)
            self.assertIsNotNone(video["segments"][0]["end_monotonic_s"])
            self.assertEqual({item["camera_id"] for item in video["cameras"]}, {"fixed", "wrist"})
            from ni_rua.robot.backends.so101 import motion_video_composite
            self.assertIn("composite", video)
            self.assertEqual(video["composite"]["timeline_hud_height"], motion_video_composite.FOOTER_HEIGHT)
            self.assertEqual(video["composite"]["hud_render_version"], motion_video_composite.HUD_RENDER_VERSION)
            self.assertEqual(backend.save_video(), video)
            report = backend.close()
            self.assertTrue(report["process_exit_confirmed"])

class LeRobotConnectOrderTests(unittest.TestCase):
    """Torque must never come on before Goal_Position equals the present pose."""

    def run_connect(self, *, calibrated=True):
        from types import ModuleType, SimpleNamespace
        from ni_rua.robot.backends.so101.driver import LeRobotSO101Driver
        log = []

        class Bus:
            def connect(self, handshake=True): log.append(f"bus.connect:handshake={handshake}")
            def sync_read(self, name, normalize=True, num_retry=0):
                log.append(f"read:{name}:{normalize}")
                return {"shoulder_pan": 2130, "gripper": 2214}
            def sync_write(self, name, values, normalize=True):
                log.append(f"write:{name}:{normalize}:{sorted(values.items())}")
            def disconnect(self, disable_torque=True): log.append(f"bus.disconnect:{disable_torque}")

        class Follower:
            def __init__(self, config): self.bus, self.is_calibrated = Bus(), calibrated
            def connect(self, calibrate=True): raise AssertionError("SOFollower.connect must not be used")
            def configure(self): log.append("configure(torque on)")

        module = ModuleType("lerobot.robots.so_follower")
        module.SO101Follower, module.SO101FollowerConfig = Follower, lambda **kw: SimpleNamespace(**kw)
        driver = LeRobotSO101Driver(deployment())
        from ni_rua.robot.backends.so101 import preflight

        def fake_preflight(bus, *, calibrated, **_):
            log.append("preflight")
            if not calibrated():
                raise preflight.PreflightError("motor_calibration_mismatch", "differs", {"ok": False})
            return {"ok": True}
        with patch("os.path.exists", return_value=True), patch.object(preflight, "run_preflight", fake_preflight), \
                patch.dict(sys.modules, {"lerobot": ModuleType("lerobot"), "lerobot.robots": ModuleType("lerobot.robots"),
                                      "lerobot.robots.so_follower": module}):
            try:
                driver.connect()
            except RuntimeError as error:  # PreflightError is a RuntimeError
                return log, error
        return log, None

    def test_goal_pinned_to_raw_present_before_and_after_torque_enable(self):
        log, error = self.run_connect()
        self.assertIsNone(error)
        pin = "write:Goal_Position:False:[('gripper', 2214), ('shoulder_pan', 2130)]"
        self.assertEqual(log, ["bus.connect:handshake=False", "preflight", "read:Present_Position:False", pin, "configure(torque on)",
                               "read:Present_Position:False", pin])

    def test_uncalibrated_motors_are_refused_without_torque_or_writes(self):
        log, error = self.run_connect(calibrated=False)
        self.assertIn("motor_calibration_mismatch", str(error))
        self.assertEqual(log, ["bus.connect:handshake=False", "preflight", "bus.disconnect:False"])

# === tests/test_so101_worker_signals.py ===

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest



HUP_WORKER = r"""
import sys
sys.path[:0] = [{root!r}, {tests!r}]
from ni_rua.robot.backends.so101 import worker
from ni_rua.robot.backends.so101.driver import Clock, FakeArmDriver, FakeCameraSource
from ni_rua.robot.backends.so101.deployment import load_deployment
from ni_rua.core.config import Config

def factory(arguments):
    deploy = load_deployment({example!r})
    clock = Clock()
    return worker.SO101Adapter(deploy, {run_dir!r}, "rgb_proprio_v1", driver=FakeArmDriver(deploy, clock),
        cameras={{n: FakeCameraSource(n, clock=clock) for n in ("wrist", "fixed")}}, clock=clock,
        image_settings=Config().images.to_dict(), start_telemetry=False)
worker._factory = factory
sys.argv = ["worker", "--deployment", {example!r}, "--run-dir", {run_dir!r}, "--observation-profile", "rgb_proprio_v1"]
worker.main()
"""


def default_dispositions():
    """``preexec_fn``: SIGINT and SIGHUP at their defaults in the worker, as under a gateway started from an interactive
    terminal. Ignored dispositions are inherited across fork and exec, so a test run started under ``nohup`` or as a
    background job of a non-interactive shell would hand SIG_IGN to the worker: the standalone worker would then keep
    ignoring Ctrl-C (Python's convention for an inherited SIG_IGN), and the supervised checks would pass without the
    worker ignoring anything itself."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGHUP, signal.SIG_DFL)


class WorkerSignalTests(unittest.TestCase):
    def script(self, directory):
        script = Path(directory) / "hup_worker.py"
        script.write_text(HUP_WORKER.format(root=str(REPO_ROOT), tests=str(TESTS_ROOT), example=str(EXAMPLE),
                                            run_dir=str(Path(directory) / "worker")))
        return script

    def supervised(self, directory):
        """The worker as the gateway starts it: stdin and stdout are pipes from the parent."""
        process = subprocess.Popen([sys.executable, "-u", str(self.script(directory))], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
                                   preexec_fn=default_dispositions)
        self.addCleanup(lambda: (process.poll() is None and process.kill(), process.stdout.close(), process.stderr.close()))
        return process

    def test_sighup_is_ignored_and_stdin_eof_still_ends_the_worker(self):
        from ni_rua.robot.backends.robodojo.worker_protocol import WIRE_PREFIX
        with tempfile.TemporaryDirectory() as directory:
            process = self.supervised(directory)

            def request(op, identifier):
                process.stdin.write(json.dumps({"id": identifier, "op": op}) + "\n")
                process.stdin.flush()
                for line in process.stdout:
                    if line.startswith(WIRE_PREFIX) and json.loads(line[len(WIRE_PREFIX):]).get("id") == identifier:
                        return json.loads(line[len(WIRE_PREFIX):])
                return None
            self.assertEqual(request("ready", "r1")["result"]["arms"], ["right"])
            os.kill(process.pid, signal.SIGHUP)  # what a closing terminal sends to its process group
            time.sleep(0.1)
            self.assertIsNone(process.poll(), "SIGHUP must not kill the worker before the gateway closes it")
            self.assertIn("result", request("ready", "r2"))
            process.stdin.close()  # owner gone: the worker holds the arm and exits on its own
            self.assertEqual(process.wait(timeout=20), 0)
            self.assertIn("so101_worker_start", process.stderr.read())

    def test_sigint_is_ignored_when_supervised_by_a_pipe(self):
        # Ctrl-C in the gateway's terminal reaches the whole process group; the gateway handles it and sends close.
        from ni_rua.robot.backends.robodojo.worker_protocol import WIRE_PREFIX
        with tempfile.TemporaryDirectory() as directory:
            process = self.supervised(directory)
            process.stdin.write(json.dumps({"id": "r1", "op": "ready"}) + "\n")
            process.stdin.flush()
            for line in process.stdout:
                if line.startswith(WIRE_PREFIX):
                    break
            os.kill(process.pid, signal.SIGINT)
            time.sleep(0.1)
            self.assertIsNone(process.poll(), "Ctrl-C must not kill a supervised worker before the gateway closes it")
            process.stdin.close()
            self.assertEqual(process.wait(timeout=20), 0)
            self.assertNotIn("KeyboardInterrupt", process.stderr.read())

    def test_standalone_worker_still_stops_on_ctrl_c(self):
        # Started by hand from a terminal (stdin is a TTY, not a pipe): Ctrl-C keeps its default effect.
        import pty
        with tempfile.TemporaryDirectory() as directory:
            master, slave = pty.openpty()
            self.addCleanup(os.close, master)
            process = subprocess.Popen([sys.executable, "-u", str(self.script(directory))], stdin=slave,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                       preexec_fn=default_dispositions)
            os.close(slave)
            self.addCleanup(lambda: (process.poll() is None and process.kill(), process.stdout.close(), process.stderr.close()))
            deadline = time.monotonic() + 20
            seen = ""
            while "so101_worker_start" not in seen and time.monotonic() < deadline:
                seen += process.stderr.readline()
            self.assertIn("so101_worker_start", seen)  # signal dispositions are set before this line
            os.kill(process.pid, signal.SIGINT)
            self.assertNotEqual(process.wait(timeout=20), 0)

    def test_supervision_probe(self):
        from ni_rua.robot.backends.so101.worker import supervised_by_pipe
        import pty
        read, write = os.pipe()
        master, slave = pty.openpty()
        try:
            self.assertTrue(supervised_by_pipe(read))
            self.assertFalse(supervised_by_pipe(slave))
        finally:
            for fd in (read, write, master, slave):
                os.close(fd)

    def test_so101_motion_non_driver_exceptions_pause_hold_faults_camera_grab_and_init_rollback(self):
        from unittest.mock import MagicMock, patch
        from ni_rua.core.config import Config
        from ni_rua.robot.backends.so101 import worker
        from ni_rua.robot.backends.so101.deployment import load_deployment
        from ni_rua.robot.backends.so101.driver import Clock, FakeArmDriver, FakeCameraSource, OpenCVCameraSource

        run_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        deploy = load_deployment(str(EXAMPLE))
        clock = Clock()
        arm_driver = FakeArmDriver(deploy, clock)
        cams = {n: FakeCameraSource(n, clock=clock) for n in ("wrist", "fixed")}
        adapter = worker.SO101Adapter(
            deploy,
            str(run_dir / "adapter"),
            "rgb_proprio_v1",
            driver=arm_driver,
            cameras=cams,
            clock=clock,
            image_settings=Config().images.to_dict(),
            start_telemetry=False,
        )
        self.addCleanup(adapter.close)
        obs = adapter.observe()
        obs_id = obs["raw_observation_id"]
        caps = adapter.capabilities()
        identity = dict(run_id="r", episode_id="e", env_instance_id="i", env_generation=1)

        def jaw_stage(sid, goal):
            return {
                "id": sid,
                "kind": "gripper",
                "arm": "right",
                "checkpoint_after_gripper": True,
                "action": contract.compile_action(
                    "gripper",
                    dict(arm="right", goal_normalized=goal, observation_id=obs_id, action_id=f"act-{sid}"),
                    identity,
                    caps,
                ),
            }

        batch = {
            "schema_version": "0.1",
            "type": "BatchAction",
            **identity,
            "world_kind": "observed",
            "action_id": "batch-exc-1",
            "based_on_observation_id": obs_id,
            "frame_id": caps["frame_id"],
            "tcp_config_id": caps["tcp_config_id"],
            "limits_profile": caps["limits_profile"],
            "timeout_s": 15.0,
            "stages": [jaw_stage("g1", 0.2), {"id": "look", "kind": "observe", "reason": "check"}, jaw_stage("g2", 0.8)],
        }

        # DEF-04: unconfirmed hold() in _pause_batch must NOT call self.read() on faulted bus and keeps status='unknown'.
        prev1 = adapter.preview_batch(batch)
        self.assertEqual(prev1["status"], "planned", prev1)
        with patch.object(adapter, "hold", return_value={"confirmed": False, "state": "unknown", "reason": "serial_fault"}):
            orig_read = adapter.read
            read_after_hold = {"called": False}

            def guarded_read():
                if adapter._plans[prev1["plan_id"]].get("status") == "failed":
                    read_after_hold["called"] = True
                    raise worker.DriverFault("bus dead after hold")
                return orig_read()

            with patch.object(adapter, "read", side_effect=guarded_read):
                rec_pause = adapter.execute(dict(batch, batch_plan_id=prev1["plan_id"]), lambda _: None, threading.Event())
        from ni_rua.core.presentation.feedback_delivery import execution_unconfirmed
        self.assertFalse(read_after_hold["called"])
        self.assertEqual(rec_pause["status"], "stopped")
        self.assertEqual(rec_pause["stop_state"], "unknown")
        self.assertFalse(rec_pause["stop_confirmation"]["confirmed"])
        self.assertTrue(execution_unconfirmed(rec_pause))
        self.assertIn("g1", rec_pause["completed_stage_ids"])
        self.assertIsNone(adapter.active_action_id)

        # DEF-03: mid-batch non-DriverFault exception after control_step advances invokes _stop(), clears active_action_id,
        # preserves completed stages, and returns status='unknown' + execution_uncertain=True.
        batch2 = dict(batch, action_id="batch-exc-2")
        batch2["stages"] = [dict(jaw_stage("g1", 0.4), checkpoint_after_gripper=False), jaw_stage("g2", 0.9)]
        prev2 = adapter.preview_batch(batch2)
        self.assertEqual(prev2["status"], "planned", prev2)
        stop_spy = MagicMock(wraps=adapter._stop)
        orig_gripper = adapter._gripper
        calls = {"n": 0}

        def crash_second_gripper(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise TypeError("simulated mid-batch guard bug")
            return orig_gripper(*args, **kwargs)

        with patch.object(adapter, "_stop", stop_spy), patch.object(adapter, "_gripper", side_effect=crash_second_gripper):
            rec_crash = adapter.execute(dict(batch2, batch_plan_id=prev2["plan_id"]), lambda _: None, threading.Event())
        self.assertEqual(stop_spy.call_count, 1)
        self.assertIsNone(adapter.active_action_id)
        self.assertNotEqual(adapter.motion_phase, "moving")
        self.assertEqual(rec_crash["status"], "unknown")
        self.assertTrue(rec_crash.get("execution_uncertain"))
        self.assertEqual(rec_crash["stage_results"]["g1"]["status"], "completed")
        self.assertIn("g1", rec_crash["completed_stage_ids"])

        # DEF-14: SO101Adapter.__init__ rolls back started cameras and disconnects driver on partial init failure.
        rb_driver = FakeArmDriver(deploy, clock)
        started_cam = FakeCameraSource("fixed", clock=clock)
        started_cam.close = MagicMock(wraps=started_cam.close)
        failing_cam = FakeCameraSource("wrist", clock=clock)
        failing_cam.start = MagicMock(side_effect=RuntimeError("so101_camera_unavailable:wrist"))
        with self.assertRaisesRegex(RuntimeError, "so101_camera_unavailable"):
            worker.SO101Adapter(
                deploy,
                str(run_dir / "rollback"),
                "rgb_proprio_v1",
                driver=rb_driver,
                cameras={"fixed": started_cam, "wrist": failing_cam},
                clock=clock,
                image_settings=Config().images.to_dict(),
                start_telemetry=False,
            )
        started_cam.close.assert_called_once()
        self.assertFalse(rb_driver.connected)

        # DEF-16 (#2) & DEF-24 (#1): _write_atomic cleans up .tmp on replace error; MotionVideoRecorder.record(True) aborts on mkdir OSError.
        target_file = run_dir / "atomic_target.bin"
        with patch("ni_rua.robot.backends.so101.worker.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                adapter._write_atomic(target_file, b"payload")
        self.assertEqual(list(run_dir.glob(".atomic_target.bin.*.tmp")), [])

        with patch.object(Path, "mkdir", side_effect=OSError("readonly videos dir")):
            rec_vid = adapter.motion_video.record(True)
        self.assertFalse(rec_vid["enabled"])
        self.assertEqual(adapter.motion_video.streams, {})

        # DEF-13: OpenCVCameraSource._grab isolates subscriber errors and close() skips release() if thread is still alive.
        import numpy as np
        cam_src = object.__new__(OpenCVCameraSource)
        cam_src.camera_id = "wrist"
        cam_src.config = {"width": 8, "height": 6, "rotate_deg": 0}
        cam_src.clock = clock
        cam_src.condition = threading.Condition()
        cam_src.frame = None
        cam_src.stamp = None
        cam_src.read_finished = None
        cam_src.stamps = []
        cam_src.error = None
        cam_src.running = True
        cam_src.frame_sink = lambda *_a: (_ for _ in ()).throw(RuntimeError("sink boom"))

        class OneShotCapture:
            def __init__(self):
                self.released = False

            def read(self):
                cam_src.running = False
                return True, np.zeros((6, 8, 3), dtype=np.uint8)

            def release(self):
                self.released = True

        cap = OneShotCapture()
        cam_src.capture = cap
        cam_src.thread = None
        cam_src._grab()
        self.assertIsNotNone(cam_src.frame)
        self.assertIsNone(cam_src.error)

        alive_evt = threading.Event()
        self.addCleanup(alive_evt.set)
        stuck_thread = threading.Thread(target=alive_evt.wait, args=(10.0,), daemon=True)
        stuck_thread.start()
        cam_src.thread = stuck_thread
        with patch.object(stuck_thread, "join"):
            cam_src.close()
        self.assertFalse(cap.released)
        self.assertIs(cam_src._abandoned_capture, cap)


if __name__ == "__main__":
    unittest.main()
