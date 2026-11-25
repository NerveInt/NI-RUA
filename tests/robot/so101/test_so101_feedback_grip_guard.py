"""SO-101 trajectory feedback and grip guards on CPU (fake driver, fake cameras, real kinematics).

Regression for two real-arm defects (2026-11-13):
- every executed SO-101 action returned ``trajectory_feedback.available=false``
  (``invalid_artifact_reference``): the worker receipt had no ``before_state`` and the
  feedback iterated a fixed ARX-X5 left/right arm layout;
- SO-101 rejected ``grip_guard_preset`` / ``min_aperture_m`` / ``max_aperture_m`` /
  ``checkpoint_after_gripper``, so an empty grasp was never caught before the lift.
These tests prove the software contract, not physical grasp quality.
"""

NI_RUA_TEST_TIER = "component"

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ni_rua.core.config import Config
from ni_rua.core.contracts import canonical
from ni_rua.core.storage.images import image_settings
from ni_rua.core.storage.store import Store
from ni_rua.robot.backends.so101 import contract
from ni_rua.robot.backends.so101.deployment import validate_deployment
from ni_rua.robot.backends.so101.driver import FakeCameraSource, gripper_rad_from_normalized
from ni_rua.robot.model.so101_gripper_geometry import jaw_rad_for_opening, opening_width_m
from ni_rua.robot.motion.grip_guard import preset_aperture_bounds
from ni_rua.runtime.actions.trajectory_feedback import build_action_trajectory_feedback
from robot.so101.test_so101_backend import EXAMPLE, IDENTITY, InProcessBackend, deployment, make_adapter

PROFILE = "rgb_proprio_calibrated_v1"
EMPTY_CLOSE_M = 0.0035  # fake jaw closes to the CAD closed limit (about 0.0031 m)
CUBE_M = 0.029


def calibrated_adapter(directory, deploy=None, **driver_options):
    """Fake SO-101 with a synthetic camera calibration file (calibrated profile keeps joint names)."""
    from ni_rua.robot.calibration.workflow import charuco as cal
    deploy = deploy or deployment(**{"gripper.empty_close_aperture_m": EMPTY_CLOSE_M})
    size = {name: [deploy["cameras"][name]["width"] // 10, deploy["cameras"][name]["height"] // 10] for name in ("wrist", "fixed")}
    K = [[50., 0, 32], [0, 50., 24], [0, 0, 1]]
    fixed = np.eye(4)
    fixed[:3, 3] = [0.5, 0, 0.4]
    path = Path(directory) / "camera_calibration.json"
    cal.write_calibration_file(path, board=cal.DEFAULT_BOARD_SPEC, samples=20, cameras={
        "wrist": dict(K=K, distortion=[0] * 5, image_size=size["wrist"], T_link_camera=np.eye(4).tolist(),
                      intrinsics_rms_px=.3, hand_eye_residual=dict(translation_m=.001, rotation_rad=.002)),
        "fixed": dict(K=K, distortion=[0] * 5, image_size=size["fixed"], T_base_camera=fixed.tolist(),
                      intrinsics_rms_px=.3, extrinsic_residual=dict(reprojection_px=.5))})
    deploy["camera_calibration_file"] = str(path)
    for name in ("wrist", "fixed"):
        deploy["cameras"][name].update(width=size[name][0], height=size[name][1])
    adapter, driver, clock = make_adapter(directory, profile=PROFILE, deploy=deploy, **driver_options)
    adapter.cameras = {name: FakeCameraSource(name, clock=clock, width=size[name][0], height=size[name][1])
                       for name in ("wrist", "fixed")}
    return adapter, driver, clock


class HarnessCase(unittest.TestCase):
    def harness(self, deploy=None, **driver_options):
        from ni_rua.runtime.harness import Harness
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        adapter, driver, _ = calibrated_adapter(directory.name, deploy=deploy, **driver_options)
        store = Store(Path(directory.name) / "store", IDENTITY)
        self.addCleanup(store.close)
        config = Config.from_dict({"backend": {"environment": "so101_real", "robot_model_id": "so101", "deployment": str(EXAMPLE)},
                                   "features": {"observation_profile": PROFILE, "record_video": False}})
        harness = Harness(store, InProcessBackend(adapter), observation_profile=PROFILE, config=config)
        harness.start()
        harness.task_planning.required = False
        self.addCleanup(adapter.close)
        return harness, adapter, driver

    @staticmethod
    def pose(adapter):
        return adapter.robot_state(adapter.read())["arms"]["right"]["ee_pose"]

    def move(self, adapter, identifier, dx=0.0, dz=0.0, **fields):
        pose = self.pose(adapter)
        return dict(id=identifier, kind="move", arm="right",
                    position_m=[pose["position_m"][0] + dx, pose["position_m"][1], pose["position_m"][2] + dz],
                    quaternion_xyzw=pose["quaternion_xyzw"], position_tolerance_m=0.005, orientation_tolerance_rad=0.1, **fields)

    def run_batch(self, harness, stages, **batch):
        from ni_rua.runtime.actions.unified_actions import dispatch
        request = dict(observation_id=harness.latest["observation_id"], stages=stages, **batch)
        with patch("ni_rua.runtime.actions.inspection._trajectory_image", return_value=({"status": "unavailable"}, [])):
            preview = dispatch(harness, request, {"tool_call_id": "preview"})
        self.assertEqual(preview.data["status"], "planned", preview.data)
        return preview, self.execute(harness, preview.data["plan_id"])

    @staticmethod
    def execute(harness, plan_id, call="execute"):
        from ni_rua.runtime.actions.unified_actions import dispatch
        return dispatch(harness, dict(mode="execute", plan_id=plan_id), {"tool_call_id": call + plan_id})

    @staticmethod
    def shown(harness, response):
        """The actor-visible JSON parts after presentation."""
        from ni_rua.core.presentation.actor_presentation import ActorPresentation
        wire = ActorPresentation(harness.store).present_response(response)
        values = []
        for item in wire["contentItems"]:
            if item["type"] == "inputText":
                try:
                    values.append(json.loads(item["text"]))
                except ValueError:
                    pass
        return values

    def feedback(self, harness, response):
        found = [value["trajectory_feedback"] for value in self.shown(harness, response)
                 if isinstance(value, dict) and "trajectory_feedback" in value]
        self.assertEqual(len(found), 1, found)
        return found[0]


class TrajectoryFeedbackTests(HarnessCase):
    def test_executed_move_returns_measured_and_planned_single_arm_feedback(self):
        harness, adapter, _ = self.harness()
        _, response = self.run_batch(harness, [self.move(adapter, "reach", dx=0.02)])
        receipt = response.data["receipt"]
        self.assertEqual(receipt["status"], "completed", receipt)
        stored = harness.store.read("action_receipt", receipt["action_id"] if "action_id" in receipt else response.data["action_id"])
        self.assertIsNotNone(stored["before_state_ref"])
        feedback = self.feedback(harness, response)
        self.assertTrue(feedback["available"], feedback)
        self.assertTrue(feedback["native_boundary_alignment"])
        self.assertEqual(feedback["joint_names"], {"right": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
                                                             "wrist_roll", "gripper"]})
        self.assertTrue(feedback["planned"]["available"])
        for path in ("measured", "planned"):
            self.assertEqual({arm for sample in feedback[path]["samples"] for arm in sample["arms"]}, {"right"})
        self.assertIn("measured_command_error", feedback)
        # E3 (WP9): the worker captures the execution-start observation, so FK geometry is available.
        self.assertTrue(stored["execution_start_observation_id"])
        self.assertEqual(feedback["geometry"]["status"], "available", feedback["geometry"])
        # The full artifact has one measured sample per dispatched control step plus the start.
        reference = harness.store.read_state("unified_action_feedback", response.data["action_id"])["trajectory_ref"]
        full = json.loads(harness.store.resolve_artifact(reference, ("actor",))[1])
        measured = full["paths"]["measured"]
        self.assertEqual(len(measured["samples"]), stored["completed_native_steps"] + 1)
        self.assertEqual(measured["measurement_timing"]["source_kind"], "recorded_host_read_time")
        times = [sample["elapsed_time_s"] for sample in measured["samples"]]
        self.assertEqual(times[0], 0.0)
        self.assertTrue(all(b > a for a, b in zip(times, times[1:])))
        final = measured["samples"][-1]["arms"]["right"]["measured_ee_pose"]["position_m"]
        target = self.move(adapter, "x")["position_m"]
        self.assertLess(np.linalg.norm(np.subtract(final, target)), 0.005)

    def test_worker_receipt_shape_feeds_feedback_without_robot_access(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        adapter, _, _ = calibrated_adapter(directory.name)
        self.addCleanup(adapter.close)
        pose = self.pose(adapter)
        action = contract.compile_action("move", dict(arm="right", position_m=[pose["position_m"][0] + 0.01, *pose["position_m"][1:]],
                                                      quaternion_xyzw=pose["quaternion_xyzw"], observation_id="obs"),
                                         IDENTITY, adapter.capabilities())
        raw = adapter.execute(action, lambda update: None, __import__("threading").Event())
        self.assertEqual(raw["status"], "completed", raw)
        self.assertTrue(raw["trace_sampling"]["actual_boundary_samples"])
        self.assertEqual(raw["before_state"], raw["actual_trace"][0]["state"])
        self.assertEqual([entry["sim_step"] for entry in raw["actual_trace"]],
                         list(range(raw["execution_interval"]["start_sim_step"], raw["execution_interval"]["end_sim_step"] + 1)))
        self.assertEqual(len(raw["planned_trace"]["q_rad"]), raw["completed_native_steps"])
        # Store the evidence like Harness.execute and build feedback from the artifacts alone.
        store = Store(Path(directory.name) / "store", IDENTITY)
        self.addCleanup(store.close)
        receipt = {key: value for key, value in raw.items()
                   if key not in ("planned_trace", "dispatched_trace", "actual_trace", "before_state", "terminal_observation")}
        for name in ("planned_trace", "dispatched_trace", "actual_trace", "before_state"):
            receipt[name + "_ref"] = store.put_artifact(canonical(raw[name]).encode(), name, "application/json")
        receipt["result_observation_id"] = "terminal"
        harness = type("H", (), dict(store=store, identity=IDENTITY, latest=None, observation_profile=PROFILE,
                                     image_settings=image_settings(), config=None))()
        action.update(IDENTITY, action_id="a1")
        feedback, images = build_action_trajectory_feedback(harness, action, receipt, {})
        self.assertTrue(feedback["available"], feedback)
        self.assertEqual(images, [])
        self.assertEqual(feedback["measured"]["total_samples"], raw["completed_native_steps"] + 1)

    def raw_move(self, adapter, dx=0.01, around_arrive=None):
        """Execute one move directly on the worker adapter; ``around_arrive(original)`` may inject events."""
        import threading
        pose = self.pose(adapter)
        action = contract.compile_action("move", dict(arm="right", position_m=[pose["position_m"][0] + dx, *pose["position_m"][1:]],
                                                      quaternion_xyzw=pose["quaternion_xyzw"], observation_id="obs"),
                                         IDENTITY, adapter.capabilities())
        if around_arrive is None:
            return adapter.execute(action, lambda update: None, threading.Event()), action
        with patch.object(adapter, "_arrive", side_effect=around_arrive(adapter._arrive)):
            return adapter.execute(action, lambda update: None, threading.Event()), action

    @staticmethod
    def feedback_from_raw(directory, raw, action):
        """Store the receipt evidence like Harness.execute and build feedback from the artifacts alone."""
        store = Store(Path(directory) / "store", IDENTITY)
        receipt = {key: value for key, value in raw.items()
                   if key not in ("planned_trace", "dispatched_trace", "actual_trace", "before_state", "terminal_observation")}
        for name in ("planned_trace", "dispatched_trace", "actual_trace", "before_state"):
            receipt[name + "_ref"] = store.put_artifact(canonical(raw[name]).encode(), name, "application/json")
        receipt["result_observation_id"] = "terminal"
        harness = type("H", (), dict(store=store, identity=IDENTITY, latest=None, observation_profile=PROFILE,
                                     image_settings=image_settings(), config=None))()
        try:
            return build_action_trajectory_feedback(harness, dict(action, **IDENTITY, action_id="a1"), receipt, {})[0]
        finally:
            store.close()

    def adapter(self, **driver_options):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        adapter, driver, _ = calibrated_adapter(directory.name, **driver_options)
        self.addCleanup(adapter.close)
        return adapter, driver, directory.name

    def test_foreign_command_inside_the_action_is_not_complete_sampling(self):
        # An operator write from outside this run (run=None) advances control_step during settling.
        adapter, _, directory = self.adapter()
        def foreign(original):
            def wrapped(run, q_target, pose, tolerances, **options):
                adapter._send(adapter.arm_goal, adapter.jaw_goal)
                return original(run, q_target, pose, tolerances, **options)
            return wrapped
        raw, action = self.raw_move(adapter, around_arrive=foreign)
        self.assertEqual(raw["status"], "completed", raw)
        self.assertEqual(len(raw["planned_trace"]["q_rad"]) + 1, raw["completed_native_steps"])
        self.assertFalse(raw["trace_sampling"]["actual_boundary_samples"])
        feedback = self.feedback_from_raw(directory, raw, action)
        self.assertTrue(feedback["available"], feedback)
        self.assertFalse(feedback["native_boundary_alignment"])
        self.assertFalse(feedback["planned"]["available"])
        self.assertNotIn("measured_command_error", feedback)

    def test_operator_hold_and_faulted_write_are_not_complete_sampling(self):
        adapter, driver, directory = self.adapter()
        def hold(original):
            def wrapped(run, q_target, pose, tolerances, **options):
                adapter.operator("hold")  # what the console hold does from another thread
                return original(run, q_target, pose, tolerances, **options)
            return wrapped
        raw, action = self.raw_move(adapter, around_arrive=hold)
        self.assertEqual((raw["status"], raw["reason"]), ("stopped", "operator_hold"), raw)
        self.assertFalse(raw["trace_sampling"]["actual_boundary_samples"])
        self.assertTrue(self.feedback_from_raw(directory, raw, action)["available"])
        adapter.observe_required = False
        driver.fail_on_send = driver.sends + 5
        raw, action = self.raw_move(adapter, dx=-0.01)
        self.assertEqual(raw["status"], "unknown", raw)
        self.assertFalse(raw["trace_sampling"]["actual_boundary_samples"])
        self.assertIn("partially executed", raw["trace_sampling"]["incomplete_reason"])
        feedback = self.feedback_from_raw(directory, raw, action)
        self.assertTrue(feedback["available"], feedback)
        self.assertFalse(feedback["planned"]["available"])

    def test_resumed_batch_returns_feedback_for_each_execute(self):
        harness, adapter, _ = self.harness()
        stages = [self.move(adapter, "reach", dx=0.02), dict(id="look", kind="observe", reason="check arrival"),
                  self.move(adapter, "back", dx=0.0)]
        preview, first = self.run_batch(harness, stages)
        self.assertEqual(first.data["receipt"]["reason"], "observation_checkpoint")
        second = self.execute(harness, preview.data["plan_id"], call="resume")
        self.assertEqual(second.data["receipt"]["status"], "completed", second.data["receipt"])
        for response in (first, second):
            feedback = self.feedback(harness, response)
            self.assertTrue(feedback["available"], feedback)
            self.assertTrue(feedback["native_boundary_alignment"])
            self.assertTrue(feedback["planned"]["available"])

    def test_trace_artifacts_are_compact_and_still_feed_feedback(self):
        jaw = (jaw_rad_for_opening(CUBE_M) + 0.174533) / (1.74533 + 0.174533)
        harness, adapter, _ = self.harness(jaw_min_normalized=jaw)
        _, response = self.run_batch(harness, [dict(id="open", kind="gripper", arm="right", goal_normalized=0.5),
                                               dict(id="close", kind="gripper", arm="right", goal_normalized=0.0),
                                               self.move(adapter, "lift", dz=0.03)])
        stored = harness.store.read("action_receipt", response.data["action_id"])
        read = lambda key: harness.store.resolve_artifact(stored[key], ("actor",))[1]
        actual, planned = json.loads(read("actual_trace_ref")), json.loads(read("planned_trace_ref"))
        dispatched = json.loads(read("dispatched_trace_ref"))
        steps = stored["completed_native_steps"]
        self.assertIn("joint_names", actual[0]["state"]["arms"]["right"])  # the start sample is complete
        self.assertNotIn("joint_names", actual[-1]["state"]["arms"]["right"])
        self.assertNotIn("contact_feedback_reason", actual[-1]["state"]["arms"]["right"]["gripper"])
        self.assertEqual(dispatched["steps"], list(range(stored["execution_interval"]["start_sim_step"] + 1,
                                                         stored["execution_interval"]["end_sim_step"] + 1)))
        self.assertNotIn("q_goal_rad", read("dispatched_trace_ref").decode())  # rows are stored once, in planned_trace
        self.assertEqual(len(planned["q_rad"]), steps)
        self.assertLess(len(read("actual_trace_ref")) / (steps + 1), 500)  # bytes per control step (§4.2 size bound)
        feedback = self.feedback(harness, response)
        self.assertTrue(feedback["available"] and feedback["planned"]["available"], feedback)
        # The blocked jaw is named; with the worker's start frame (E3) the measured overlay image is delivered too.
        self.assertEqual(feedback["measured_command_error"]["joint"], "gripper")
        self.assertEqual(feedback["delivery"], "automatic_sampled_joints_and_image")

    def test_read_fault_at_batch_start_leaves_the_plan_executable(self):
        harness, adapter, driver = self.harness()
        from ni_rua.runtime.actions.unified_actions import dispatch
        with patch("ni_rua.runtime.actions.inspection._trajectory_image", return_value=({"status": "unavailable"}, [])):
            preview = dispatch(harness, dict(observation_id=harness.latest["observation_id"],
                                             stages=[self.move(adapter, "reach", dx=0.01)]), {"tool_call_id": "p"})
        plan_id = preview.data["plan_id"]
        worker_plan = next(iter(adapter._plans.values()))
        from ni_rua.robot.backends.so101.worker import DriverFault
        with patch.object(adapter, "_start_sample", side_effect=DriverFault("driver_read_failed: fake serial read failure")):
            first = self.execute(harness, plan_id, call="fault")
        self.assertEqual((first.data["receipt"]["status"], worker_plan["status"], worker_plan["revision"]),
                         ("rejected", "planned", 0), first.data["receipt"])
        self.assertIn("backend_error", first.data["receipt"]["reason"])
        harness.accepting = True  # the operator re-opens admission; the plan itself was not consumed
        second = self.execute(harness, plan_id, call="retry")
        self.assertEqual(second.data["receipt"]["status"], "completed", second.data["receipt"])

    def test_single_arm_state_without_left_arm_is_accepted(self):
        store = Store(Path(tempfile.mkdtemp()), IDENTITY)
        self.addCleanup(store.close)
        names = ["j1", "j2", "j3", "jaw"]
        def state(step, q):
            return dict(arms={"right": dict(joint_names=names, q_rad=q, source_kind="measurement",
                                            ee_pose=dict(position_m=[0.2, 0, 0.1 + q[0] / 10], quaternion_xyzw=[0, 0, 0, 1],
                                                         frame_id="base", tcp_config_id="tcp", source_kind="measurement"))},
                        sim_step=step, step_domain="so101_control_step")
        before = state(5, [0., 0., 0., 0.])
        actual = [dict(sim_step=5, t_s=0.0, state=before)] + [
            dict(sim_step=5 + index, t_s=0.05 * index, state=state(5 + index, [0.1 * index] * 4)) for index in (1, 2)]
        plan = dict(arm="right", joint_names=names, q_rad=[[0.1] * 4, [0.2] * 4], completed_native_steps=2)
        put = lambda value, kind: store.put_artifact(canonical(value).encode(), kind, "application/json")
        receipt = dict(status="completed", completed_native_steps=2, executed_steps=2, result_observation_id="t",
                       trace_sampling=dict(actual_every_native_actions=1, actual_boundary_samples=True),
                       before_state_ref=put(before, "before_state"), actual_trace_ref=put(actual, "actual_trace"),
                       planned_trace_ref=put(plan, "planned_trace"), dispatched_trace_ref=put([], "dispatched_trace"))
        harness = type("H", (), dict(store=store, identity=IDENTITY, latest=None, observation_profile=PROFILE,
                                     image_settings=image_settings(), config=None))()
        action = dict(IDENTITY, action_id="a", frame_id="base", tcp_config_id="tcp")
        feedback, _ = build_action_trajectory_feedback(harness, action, receipt, {})
        self.assertTrue(feedback["available"], feedback)
        self.assertEqual(feedback["joint_names"], {"right": names})
        self.assertEqual(feedback["planned"]["samples"][-1]["arms"]["right"]["q_rad"], [0.2] * 4)
        self.assertAlmostEqual(feedback["measured_command_error"]["max_abs_joint_error_rad"], 0.0)
        self.assertEqual(feedback["measured"]["measurement_timing"]["source_kind"], "recorded_host_read_time")
        self.assertEqual([sample["elapsed_time_s"] for sample in feedback["measured"]["samples"]], [0.0, 0.05, 0.1])


class GripGuardTests(HarnessCase):
    def grasp(self, adapter, close_fields=None, lift_fields=None, open_goal=0.5):
        return [dict(id="open", kind="gripper", arm="right", goal_normalized=open_goal),
                dict(id="close", kind="gripper", arm="right", goal_normalized=0.0, **(close_fields or {})),
                self.move(adapter, "lift", dz=0.03, **(lift_fields or {}))]

    def test_empty_close_pauses_before_lift_with_typed_receipt(self):
        harness, adapter, driver = self.harness()
        preview, response = self.run_batch(harness, self.grasp(adapter, dict(grip_guard_preset="non_empty")))
        minimum = preset_aperture_bounds("non_empty", empty_close_aperture_m=EMPTY_CLOSE_M)[0]
        planned_guard = preview.data["stage_plans"]["close"]["grip_guard"]
        self.assertEqual(planned_guard, dict(grip_guard_preset="non_empty", min_aperture_m=minimum, max_aperture_m=None,
                                             empty_close_aperture_m=EMPTY_CLOSE_M, source="stage", detects_empty_close=True,
                                             post_grasp_recheck={"available": True}))
        receipt = response.data["receipt"]
        self.assertEqual((receipt["status"], receipt["reason"], receipt["checkpoint_reason"]),
                         ("stopped", "observation_checkpoint", "aperture_below_min_m"), receipt)
        self.assertTrue(receipt["paused_at_checkpoint"])
        self.assertTrue(receipt["stop_confirmation"]["confirmed"])
        self.assertEqual(receipt["checkpoint_stage_ids"], ["close"])
        self.assertEqual(receipt["pending_stage_ids"], ["lift"])
        close = receipt["stage_results"]["close"]
        self.assertEqual(close["status"], "paused_at_checkpoint")
        guard = close["grip_guard"]
        self.assertTrue(guard["triggered"])
        self.assertEqual(guard["min_aperture_m"], minimum)
        self.assertAlmostEqual(guard["measured_aperture_m"], opening_width_m(-0.174533), places=4)
        self.assertLess(guard["measured_aperture_m"], minimum)
        self.assertEqual(close["grip_facts"]["measured_aperture_m"], guard["measured_aperture_m"])
        self.assertEqual(receipt["stage_results"]["lift"]["status"], "pending")
        # No lift command was dispatched after the close.
        stored = harness.store.read("action_receipt", response.data["action_id"])
        rows = json.loads(harness.store.resolve_artifact(stored["planned_trace_ref"], ("actor",))[1])
        self.assertEqual(set(rows["row_kinds"]) - {"motion", "hold"}, set())
        self.assertTrue(all(np.allclose(row[:5], rows["q_rad"][0][:5]) for row in rows["q_rad"]))
        # The model may still resume after inspecting the images; the lift then runs.
        resumed = self.execute(harness, preview.data["plan_id"], call="resume")
        self.assertEqual(resumed.data["receipt"]["status"], "completed", resumed.data["receipt"])

    def test_real_grasp_with_same_guard_passes_and_lifts(self):
        jaw = (jaw_rad_for_opening(CUBE_M) + 0.174533) / (1.74533 + 0.174533)
        harness, adapter, _ = self.harness(jaw_min_normalized=jaw)
        _, response = self.run_batch(harness, self.grasp(adapter, dict(grip_guard_preset="non_empty")))
        receipt = response.data["receipt"]
        self.assertEqual(receipt["status"], "completed", receipt)
        guard = receipt["stage_results"]["close"]["grip_guard"]
        self.assertFalse(guard["triggered"])
        self.assertAlmostEqual(guard["measured_aperture_m"], CUBE_M, delta=0.001)
        self.assertEqual(receipt["stage_results"]["lift"]["status"], "completed")

    def test_explicit_bounds_batch_default_and_above_max(self):
        jaw = (jaw_rad_for_opening(CUBE_M) + 0.174533) / (1.74533 + 0.174533)
        harness, adapter, _ = self.harness(jaw_min_normalized=jaw)
        # Batch-level bounds apply to jaw stages with goal < 0.85 (shared rule), not to a goal-0.9 open.
        preview, response = self.run_batch(harness, self.grasp(adapter, open_goal=0.9), min_aperture_m=0.01, max_aperture_m=0.02)
        self.assertNotIn("grip_guard", preview.data["stage_plans"]["open"])
        self.assertEqual(preview.data["stage_plans"]["close"]["grip_guard"]["source"], "batch_default")
        receipt = response.data["receipt"]
        self.assertEqual(receipt["checkpoint_reason"], "aperture_above_max_m", receipt)
        self.assertEqual(receipt["stage_results"]["close"]["grip_guard"]["max_aperture_m"], 0.02)

    def test_post_grasp_recheck_fires_once_on_first_lift_after_slip(self):
        jaw = (jaw_rad_for_opening(CUBE_M) + 0.174533) / (1.74533 + 0.174533)
        harness, adapter, driver = self.harness(jaw_min_normalized=jaw)
        stages = self.grasp(adapter, dict(grip_guard_preset="rigid_object")) + [self.move(adapter, "carry", dx=0.02, dz=0.03)]
        original = adapter._arrive

        def slip_during_lift(run, q_target, pose, tolerances, **options):
            if pose["position_m"][2] > self.pose(adapter)["position_m"][2] - 1e-6 and driver.jaw_min is not None:
                driver.jaw_min = None  # the object slips out while lifting; the jaw closes on air
            return original(run, q_target, pose, tolerances, **options)
        with patch.object(adapter, "_arrive", side_effect=slip_during_lift):
            _, response = self.run_batch(harness, stages)
        receipt = response.data["receipt"]
        self.assertEqual((receipt["status"], receipt["checkpoint_reason"]), ("stopped", "aperture_below_min_m"), receipt)
        self.assertEqual(receipt["checkpoint_stage_ids"], ["lift"])
        self.assertEqual(receipt["stage_results"]["close"]["status"], "completed")
        self.assertEqual(receipt["stage_results"]["carry"]["status"], "pending")

    def test_checkpoint_after_gripper_pauses_before_dependent_move(self):
        harness, adapter, _ = self.harness()
        preview, response = self.run_batch(harness, self.grasp(adapter, dict(checkpoint_after_gripper=True)))
        self.assertEqual([(item["stage_id"], item["pause_before_stage_id"]) for item in preview.data["checkpoints"]],
                         [("close", "lift")])
        receipt = response.data["receipt"]
        self.assertEqual((receipt["status"], receipt["reason"]), ("stopped", "observation_checkpoint"), receipt)
        self.assertEqual(receipt["checkpoint_stage_ids"], ["close"])
        self.assertNotIn("checkpoint_reason", receipt)
        self.assertEqual(receipt["pending_stage_ids"], ["lift"])
        resumed = self.execute(harness, preview.data["plan_id"], call="resume")
        self.assertEqual(resumed.data["receipt"]["status"], "completed", resumed.data["receipt"])

    def test_actor_text_and_capabilities_state_measured_zero_point(self):
        deploy = deployment(**{"gripper.empty_close_aperture_m": 0.0051, "gripper.full_open_aperture_m": 0.136})
        text = contract.actor_motion_instructions(deploy)
        self.assertIn("An EMPTY full close on this unit reads 0.0051 m", text)
        self.assertIn("non_empty >= 0.0066 m", text)
        self.assertNotIn("grip_guard_*", text)
        # Decision Q3: one sentence names what this arm does not offer, in plain words (no X5 field names). It is the
        # generated sentence of the run's robot_action description; the motion section does not repeat it.
        from ni_rua.runtime.actions.robot_action_schema import build
        self.assertNotIn("Not available on this", text)
        description = build(contract.embodiment(deploy)).tool["description"]
        self.assertEqual(description.count("Not available on this robot, so absent from this schema:"), 1)
        unavailable = description.split("Not available on this robot")[1].split(". ")[0]
        self.assertIn("grasp-angle pose input", unavailable)
        # WP9: the vertical arrival window and the during-motion jaw ramp are offered (and described) on SO-101.
        self.assertNotIn("a vertical arrival window", unavailable)
        self.assertNotIn("jaw motion during a move", unavailable)
        self.assertIn("Vertical arrival window: max_upward_dz_m (m, 0..0.15; default 0 = no window", text)
        self.assertIn("Jaw during a move: a move stage with gripper_goal_normalized and gripper_motion", text)
        for field in ("grasp_position_m", "angles_deg", "max_upward_dz_m"):
            self.assertNotIn(field, unavailable)
        self.assertNotIn("grasp_position_m", text)
        capability = contract.grip_guard_capability(deploy)
        self.assertEqual(capability["empty_close_aperture_m"], 0.0051)
        self.assertEqual(capability["presets"]["non_empty"]["min_aperture_m"], 0.0066)
        # R4: the explicit bound is this jaw's full opening, not the X5 0.088 m.
        self.assertEqual(capability["explicit_bounds_m"], [0.0, 0.136])
        self.assertIn("[0, 0.1360] m", text)

    def test_unmeasured_zero_point_reports_a_bound_that_cannot_catch_an_empty_close(self):
        harness, adapter, _ = self.harness(deploy=deployment())
        floor = contract.closed_limit_opening_m(adapter.deployment)
        self.assertAlmostEqual(floor, opening_width_m(-0.174533), places=5)
        # A bound at or below the CAD closed-limit opening: kept as given, reported as blind, never fires.
        preview, response = self.run_batch(harness, self.grasp(adapter, dict(min_aperture_m=0.003)))
        planned = preview.data["stage_plans"]["close"]["grip_guard"]
        self.assertEqual((planned["min_aperture_m"], planned["detects_empty_close"]), (0.003, False))
        self.assertEqual(planned["post_grasp_recheck"], {"available": False, "reason": "gripper.empty_close_aperture_m not measured"})
        receipt = response.data["receipt"]
        self.assertEqual(receipt["status"], "completed")
        self.assertFalse(receipt["stage_results"]["close"]["grip_guard"]["triggered"])
        # Above the CAD floor but with no measured empty-close gap: unknown, not claimed.
        self.assertIsNone(contract.detects_empty_close(0.004, adapter.deployment))
        self.assertFalse(contract.detects_empty_close(None, adapter.deployment))
        measured = deployment(**{"gripper.empty_close_aperture_m": 0.0051})
        self.assertFalse(contract.detects_empty_close(0.005, measured))
        self.assertTrue(contract.detects_empty_close(0.0066, measured))
        text = contract.grip_guard_text(adapter.deployment)
        self.assertIn(f"CAD closed-limit opening {floor:.4f} m", text)
        self.assertIn("A min_aperture_m at or below", text)
        self.assertIn("no post-grasp slip re-check", text)
        capability = adapter.capabilities()["grip_guard"]
        self.assertEqual(capability["post_grasp_recheck"]["available"], False)
        self.assertEqual(capability["closed_limit_cad_opening_m"], floor)

    def test_unmeasured_zero_point_never_arms_the_slip_recheck(self):
        jaw = (jaw_rad_for_opening(CUBE_M) + 0.174533) / (1.74533 + 0.174533)
        for deploy, expected in ((deployment(), "completed"),
                                 (deployment(**{"gripper.empty_close_aperture_m": EMPTY_CLOSE_M}), "stopped")):
            with self.subTest(empty_close=deploy["gripper"]["empty_close_aperture_m"]):
                harness, adapter, driver = self.harness(deploy=deploy, jaw_min_normalized=jaw)
                stages = self.grasp(adapter, dict(min_aperture_m=0.02)) + [self.move(adapter, "carry", dx=0.02, dz=0.03)]
                original = adapter._arrive

                def slip(run, q_target, pose, tolerances, **options):
                    driver.jaw_min = None  # the object slips out during the lift; the jaw closes on air
                    return original(run, q_target, pose, tolerances, **options)
                with patch.object(adapter, "_arrive", side_effect=slip):
                    preview, response = self.run_batch(harness, stages)
                receipt = response.data["receipt"]
                self.assertEqual(receipt["status"], expected, receipt)
                recheck = preview.data["stage_plans"]["close"]["grip_guard"]["post_grasp_recheck"]["available"]
                self.assertEqual(recheck, expected == "stopped")
                self.assertEqual(adapter.capabilities()["grip_guard"]["post_grasp_recheck"]["available"], recheck)
                if expected == "stopped":
                    self.assertEqual((receipt["checkpoint_reason"], receipt["checkpoint_stage_ids"]), ("aperture_below_min_m", ["lift"]))

    def test_unsettled_jaw_pauses_a_guarded_close(self):
        harness, adapter, driver = self.harness(gripper_normalized=0.5)
        advance = driver._advance

        def slow_jaw():
            # The jaw keeps closing at 0.05 rad/s (above settle_velocity 0.02 rad/s) past settle_timeout_s.
            start, before = driver.last, driver.jaw
            advance()
            limit = 0.05 * max(driver.last - start, 0.0)
            driver.jaw = before + float(np.clip(driver.jaw - before, -limit, limit))
        driver._advance = slow_jaw
        _, response = self.run_batch(harness, [dict(id="close", kind="gripper", arm="right", goal_normalized=0.0,
                                                     grip_guard_preset="non_empty"), self.move(adapter, "lift", dz=0.03)])
        receipt = response.data["receipt"]
        self.assertEqual((receipt["status"], receipt["checkpoint_reason"]), ("stopped", "jaw_not_settled"), receipt)
        close = receipt["stage_results"]["close"]
        self.assertFalse(close["gripper_execution"]["jaw_settled"])
        self.assertFalse(close["grip_guard"]["jaw_settled"])
        self.assertTrue(close["grip_guard"]["triggered"])
        # The unsettled reading is above the minimum: without this rule the lift would have run.
        self.assertGreater(close["grip_guard"]["measured_aperture_m"], close["grip_guard"]["min_aperture_m"])
        self.assertEqual(receipt["pending_stage_ids"], ["lift"])

    def test_model_text_qualifies_reading_spread_and_recheck_reach(self):
        deploy = deployment(**{"gripper.empty_close_aperture_m": 0.0051, "gripper.full_open_aperture_m": 0.1361})
        text, capability = contract.grip_guard_text(deploy), contract.grip_guard_capability(deploy)
        # The shared schema numbers are this robot's (design 13 section 5.2): no "another robot" disclaimer remains.
        self.assertNotIn("another robot", text)
        self.assertNotIn("shared_schema_numbers", capability)
        self.assertNotIn("another robot", contract.actor_motion_instructions(deploy))
        self.assertIn("verified at one point only: a 0.029 m cube read\n0.0289 m", text)
        self.assertIn("0.0042-0.0051 m (0.9 mm), more than thin_object's", text)
        width = contract.recheck_min_object_width_m(deploy)
        self.assertAlmostEqual(width, 0.0172, delta=0.0005)
        self.assertIn(f"narrower than about\n{width:.3f} m can slip out unnoticed", text)
        self.assertEqual(capability["post_grasp_recheck"]["min_object_width_m"], width)
        self.assertAlmostEqual(capability["post_grasp_recheck"]["min_jaw_drop_m_near_empty_close"], 0.012, delta=0.001)
        self.assertEqual(capability["empty_close_spread_m"], [0.0042, 0.0051])
        self.assertIsNone(contract.recheck_min_object_width_m(deployment()))

    def test_contract_rejections_and_unmeasured_zero_point(self):
        deploy = deployment()
        self.assertIsNone(deploy["gripper"]["empty_close_aperture_m"])
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        adapter, _, _ = make_adapter(directory.name, deploy=deploy)
        self.addCleanup(adapter.close)
        caps = adapter.capabilities()
        self.assertFalse(caps["grip_guard"]["presets_available"])
        jaw = contract.compile_action("gripper", dict(arm="right", goal_normalized=0.0, observation_id="o"), IDENTITY, caps)
        move = contract.compile_action("move", dict(arm="right", position_m=[0.25, 0, 0.1], quaternion_xyzw=[0, 1, 0, 0],
                                                    observation_id="o"), IDENTITY, caps)
        def batch(*stages, **fields):
            return dict(type="BatchAction", frame_id=contract.FRAME_ID, tcp_config_id=contract.TCP_CONFIG_ID,
                        limits_profile=contract.LIMITS_PROFILE, timeout_s=30, stages=list(stages), **fields)
        close = dict(id="close", kind="gripper", arm="right", action=jaw)
        with self.assertRaisesRegex(ValueError, "grip_guard_preset requires the measured empty-close"):
            contract.validate_action(batch(dict(close, grip_guard_preset="non_empty")), deploy)
        contract.validate_action(batch(dict(close, grip_guard_preset="none")), deploy)
        contract.validate_action(batch(dict(close, min_aperture_m=0.008)), deploy)
        with self.assertRaisesRegex(ValueError, r"invalid_min_aperture_m: 0.2 m is outside \[0, 0.1372\] m \(full "
                                                r"opening of this SO-101 jaw, CAD opening at gripper.jaw_open_rad\)"):
            contract.validate_action(batch(dict(close, min_aperture_m=0.2)), deploy)
        with self.assertRaisesRegex(ValueError, "not observe stages"):
            contract.validate_action(batch(dict(id="look", kind="observe", reason="r", min_aperture_m=0.01), close), deploy)
        with self.assertRaisesRegex(ValueError, "applies to jaw stages"):
            contract.validate_action(batch(dict(id="m", kind="move", arm="right", action=move, min_aperture_m=0.01)), deploy)
        for field, value in (("joint_travel_preset", "standard"), ("review_proxies", {})):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, f"unsupported_capability: {field}"):
                contract.validate_action(batch(close, **{field: value}), deploy)
        # R6 (WP9): a batch-wide vertical window is honoured (the Harness resolves it into every move path), bounded.
        contract.validate_action(batch(close, max_upward_dz_m=0.02), deploy)
        with self.assertRaisesRegex(ValueError, r"max_upward_dz_m must be in \[0.0, 0.15\]"):
            contract.validate_action(batch(close, max_upward_dz_m=0.2), deploy)
        with self.assertRaisesRegex(ValueError, "invalid_checkpoint_after_gripper"):
            contract.validate_action(batch(close, checkpoint_after_gripper="yes"), deploy)

    def test_deployment_aperture_fields_validate(self):
        document = json.loads(EXAMPLE.read_text())
        document["gripper"].update(empty_close_aperture_m=0.006, full_open_aperture_m=0.005)
        with self.assertRaisesRegex(ValueError, "empty_close_aperture_m must be below full_open_aperture_m"):
            validate_deployment(document, base=EXAMPLE.parent)
        document["gripper"].update(empty_close_aperture_m=0.06, full_open_aperture_m=None)
        with self.assertRaisesRegex(ValueError, "out_of_range:gripper.empty_close_aperture_m"):
            validate_deployment(document, base=EXAMPLE.parent)


def batch_action(*stages, **fields):
    """A compiled SO-101 BatchAction around already compiled stages (bypasses the shared robot_action schema)."""
    return dict(type="BatchAction", frame_id=contract.FRAME_ID, tcp_config_id=contract.TCP_CONFIG_ID,
                limits_profile=contract.LIMITS_PROFILE, timeout_s=30, stages=list(stages), **fields)


class So101ProfileContractTests(HarnessCase):
    """Design 13 WP6: the SO-101 validator, worker and text read the run's EmbodimentProfile (deployment file)."""

    def compiled(self, adapter, kind, **args):
        return contract.compile_action(kind, dict(args, arm="right", observation_id="o"), IDENTITY, adapter.capabilities())

    def move_stage(self, adapter, identifier="m", dx=0.02, **path):
        pose = self.pose(adapter)
        leaf = self.compiled(adapter, "move", position_m=[pose["position_m"][0] + dx, *pose["position_m"][1:]],
                             quaternion_xyzw=pose["quaternion_xyzw"])
        leaf["path"].update(path)
        return dict(id=identifier, kind="move", arm="right", action=leaf)

    def test_explicit_aperture_bound_is_the_full_opening_of_this_jaw(self):
        """R4 / A7: explicit bounds up to this jaw's full opening (CAD 0.1372 m, measured 0.1361 m), not X5 0.088 m."""
        for gripper, full in (({}, 0.1372), ({"gripper.full_open_aperture_m": 0.1361}, 0.1361)):
            deploy = deployment(**{"gripper.empty_close_aperture_m": EMPTY_CLOSE_M, **gripper})
            with self.subTest(full_open=full):
                self.assertEqual(contract.embodiment(deploy).gripper.full_open_aperture_m, full)
                self.assertEqual(contract.explicit_aperture_limit_m(deploy), full)
                self.assertEqual(contract.grip_guard_capability(deploy)["explicit_bounds_m"], [0.0, full])
                close = dict(id="c", kind="gripper", arm="right", action=dict(
                    contract.compile_action("gripper", dict(arm="right", goal_normalized=0.0, observation_id="o"),
                                            IDENTITY, dict(frame_id=contract.FRAME_ID, tcp_config_id=contract.TCP_CONFIG_ID,
                                                           default_timeout_s=30.0))))
                for field in ("min_aperture_m", "max_aperture_m"):
                    contract.validate_action(batch_action(dict(close, **{field: full})), deploy)
                    with self.assertRaisesRegex(ValueError, rf"invalid_{field}: {full + 1e-4:g} m is outside \[0, {full}\] m"):
                        contract.validate_action(batch_action(dict(close, **{field: round(full + 1e-4, 4)})), deploy)
        # The worker plans and arms a guard with a 0.10 m maximum (a wide object the X5 schema bound could not express).
        harness, adapter, _ = self.harness()
        stage = dict(id="close", kind="gripper", arm="right", action=self.compiled(adapter, "gripper", goal_normalized=0.0),
                     max_aperture_m=0.10)
        preview = adapter.preview_batch(batch_action(stage))
        self.assertEqual(preview["status"], "planned", preview)
        self.assertEqual(preview["stage_plans"]["close"]["grip_guard"]["max_aperture_m"], 0.10)

    def test_presets_are_the_deployment_presets(self):
        """Q4: the SO-101 offers none + the four schema presets with the deployment margins; X5 legacy names refused."""
        margins = dict(contract.embodiment(deployment()).gripper.preset_margins_m)
        self.assertEqual(list(margins), ["none", *contract.SCHEMA_GRIP_GUARD_PRESETS])
        document_margins = {name: list(value) for name, value in margins.items() if name != "none"}
        deploy = deployment(**{"gripper.empty_close_aperture_m": 0.005,
                               "gripper.preset_margins_m": dict(document_margins, rigid_object=[0.004, None])})
        capability, text = contract.grip_guard_capability(deploy), contract.grip_guard_text(deploy)
        self.assertEqual(capability["presets"]["rigid_object"], {"min_aperture_m": 0.009, "max_aperture_m": None})
        self.assertIn("rigid_object >= 0.0090 m", text)
        close = dict(id="c", kind="gripper", arm="right", action={})
        self.assertAlmostEqual(contract.validate_grip_guard(dict(grip_guard_preset="rigid_object"), deploy)["min_aperture_m"],
                               0.009, places=12)
        for legacy in ("normal", "thin", "solid", "empty_only"):
            with self.subTest(preset=legacy), self.assertRaisesRegex(
                    ValueError, rf"invalid_grip_guard_preset: {legacy} \(presets on this SO-101: none, non_empty"):
                contract.validate_grip_guard(dict(close, grip_guard_preset=legacy), deploy)

    def test_ignored_path_fields_are_refused(self):
        """A14 and the C-finding: fields the worker would ignore are refused by name, never accepted and ignored."""
        from jsonschema import ValidationError
        harness, adapter, _ = self.harness()
        # Through robot_action the run's schema already refuses them by name (decision Q3, WP2: the fields are absent
        # from the SO-101 schema), before anything is compiled or stored.
        for stage, batch in (({"clearance_height_m": 0.1}, {}), ({"clearance_preset": "high"}, {}),
                             ({}, {"target_exemption_preset": "wide"})):
            field = next(iter(stage or batch))
            with self.subTest(field=field), self.assertRaisesRegex(ValidationError, rf"'{field}' was unexpected"):
                self.run_batch(harness, [self.move(adapter, "m", dz=0.01, **stage)], **batch)
        # A compiled action that reaches the SO-101 contract with them is refused by name with the reason.
        for path, message in (({"clearance_height_m": 0.1},
                               r"unsupported_capability: clearance_height_m \(no clearance arc is planned on SO-101"),
                              ({"target_exemption_radius_m": 0.05},
                               r"unsupported_capability: target_exemption_radius_m \(no obstacle review"),
                              ({"foo": 1}, r"unsupported_capability: foo \(not read by the SO-101 worker\)"),
                              ({"max_upward_dz_m": 0.2}, r"max_upward_dz_m must be in \[0.0, 0.15\]"),
                              ({"tolerance_preset": "seating"}, r"tolerance_preset seating \(presets of this deployment: "
                                                                r"tight, normal, relaxed, compliant, coarse"),
                              ({"stall_guard_preset": "bogus"}, r"stall_guard_preset bogus"),
                              ({"max_tracking_divergence_m": 0.2}, r"max_tracking_divergence_m must be in \[0.005, 0.15\]")):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, message):
                contract.validate_action(batch_action(self.move_stage(adapter, **path)), adapter.deployment)
        # The plain move after a refusal still plans and runs (nothing is stored by a refused request).
        _, response = self.run_batch(harness, [self.move(adapter, "m", dz=0.01)])
        self.assertEqual(response.data["receipt"]["status"], "completed", response.data["receipt"])

    def test_speed_and_timeouts_follow_the_profile(self):
        """R2: speed ranges are the profile's [0.25, 1]; a wait without timeout_s uses the deployment wait deadline."""
        harness, adapter, _ = self.harness(deploy=deployment(**{"gripper.empty_close_aperture_m": EMPTY_CLOSE_M,
                                                                "motion.default_wait_timeout_s": 12.5}))
        profile = harness.embodiment
        self.assertEqual(profile.robot_model_id, "so101")
        speed = adapter.capabilities()["action_speed"]
        self.assertEqual((speed["speed_scale"]["min"], speed["speed_scale"]["max"]), (profile.speed_scale.min, profile.speed_scale.max))
        for value, accepted in ((profile.speed_scale.min, True), (profile.speed_scale.min - 1e-9, False),
                                (profile.speed_scale.max, True), (profile.speed_scale.max + 1e-9, False)):
            with self.subTest(speed_scale=value):
                stage = self.move_stage(adapter)
                stage["action"]["speed_scale"] = value
                if accepted:
                    contract.validate_action(batch_action(stage), adapter.deployment)
                else:
                    with self.assertRaisesRegex(ValueError, r"speed_scale must be in \[0.25, 1.0\]"):
                        contract.validate_action(batch_action(stage), adapter.deployment)
        wait = self.compiled(adapter, "wait", native_steps=3)
        self.assertEqual(wait["timeout_s"], 12.5)
        self.assertEqual(adapter.capabilities()["bounded_wait"]["default_timeout_s"], profile.default_wait_timeout_s)
        self.assertEqual(self.compiled(adapter, "gripper", goal_normalized=1.0)["timeout_s"], profile.default_action_timeout_s)
        receipt = harness.invoke("robot_wait", dict(observation_id=harness.latest["observation_id"], native_steps=3),
                                 {"tool_call_id": "w"}, role="actor").data["receipt"]
        self.assertEqual(receipt["status"], "completed", receipt)

    def test_deployment_default_tolerance_reaches_the_worker_plan(self):
        """C3 (SO-101 side): the deployment default tolerance is what the compiled stage and the plan carry."""
        harness, adapter, _ = self.harness(deploy=deployment(**{"gripper.empty_close_aperture_m": EMPTY_CLOSE_M,
                                                                "motion.default_position_tolerance_m": 0.02}))
        self.assertEqual(harness.embodiment.default_tolerance.position_m, 0.02)
        self.assertEqual(adapter.capabilities()["motion_tolerance_defaults"]["position_tolerance_m"], 0.02)
        stage = self.move_stage(adapter)
        self.assertEqual(stage["action"]["path"]["position_tolerance_m"], 0.02)
        preview = adapter.preview_batch(batch_action(stage))
        self.assertEqual(preview["stage_plans"]["m"]["tolerances"]["position_tolerance_m"], 0.02)
        self.assertIn("defaults are 20 mm", contract.actor_motion_instructions(adapter.deployment))

    def test_deployment_default_tolerance_end_to_end_after_wp2(self):
        """C3 end to end through robot_action: the shared resolver (resolve_stage_tolerance(profile=harness.embodiment))
        applies the deployment default tolerance, not the ARX-X5 'normal' 0.01 m."""
        harness, adapter, _ = self.harness(deploy=deployment(**{"gripper.empty_close_aperture_m": EMPTY_CLOSE_M,
                                                                "motion.default_position_tolerance_m": 0.02}))
        pose = self.pose(adapter)
        stage = dict(id="m", kind="move", arm="right", quaternion_xyzw=pose["quaternion_xyzw"],
                     position_m=[pose["position_m"][0], pose["position_m"][1], pose["position_m"][2] + 0.01])
        preview, _ = self.run_batch(harness, [stage])
        self.assertEqual(preview.data["stage_plans"]["m"]["tolerances"]["position_tolerance_m"], 0.02)

    def test_max_aperture_0_10_end_to_end_after_wp2(self):
        """R4 end to end through robot_action: the run's schema bound and the shared grip-guard validation come from the
        profile (the X5 schema maximum 0.088 m would refuse 0.10 before the SO-101 contract sees it)."""
        harness, adapter, _ = self.harness()
        preview, _ = self.run_batch(harness, [dict(id="c", kind="gripper", arm="right", goal_normalized=0.0,
                                                   max_aperture_m=0.10), self.move(adapter, "lift", dz=0.02)])
        self.assertEqual(preview.data["stage_plans"]["c"]["grip_guard"]["max_aperture_m"], 0.10)

    def test_stall_guard_stop_fails_the_stage_and_the_gripper_stage_does_not_run(self):
        """A13 / R7: a contact_stop stop holds the measured pose and judges it by the arrival rule (WP9 phase 2); a
        hold far from the target (a blocked shoulder, no vertical window) is a failed move (tracking_error, hold
        confirmed once), the attached gripper stage does not run, and the text says exactly that."""
        harness, adapter, _ = self.harness(blocked=(1,))  # shoulder_lift held: the measured TCP falls behind
        stages = [self.move(adapter, "descend", dx=0.06, stall_guard_preset="contact_stop"),
                  dict(id="close", kind="gripper", arm="right", goal_normalized=0.0)]
        preview, response = self.run_batch(harness, stages)
        threshold = harness.embodiment.stall_guard_presets_m["contact_stop"]
        planned = preview.data["stage_plans"]["descend"]["tolerances"]
        self.assertEqual((planned["stall_guard_preset"], planned["max_tracking_divergence_m"]), ("contact_stop", threshold))
        receipt = response.data["receipt"]
        self.assertEqual((receipt["status"], receipt["reason"], receipt["failed_stage_id"]),
                         ("failed", "tracking_error", "descend"), receipt)
        descend = receipt["stage_results"]["descend"]
        self.assertEqual((descend["kind"], descend["status"]), ("move", "failed"))
        self.assertEqual(descend["tracking_stop"]["kind"], "tcp_divergence")
        self.assertEqual(descend["tracking_stop"]["max_tracking_divergence_m"], threshold)
        self.assertGreater(descend["tracking_stop"]["divergence_m"], threshold)
        self.assertEqual(receipt["stage_results"]["close"]["status"], "pending")
        self.assertTrue(receipt["stop_confirmation"]["confirmed"])
        self.assertTrue(descend["contact_stopped_early"])
        self.assertFalse(descend["within_tolerance"])
        self.assertEqual(receipt["stop_confirmation"], descend["hold_confirmation"])  # one hold, not a second one
        self.assertEqual(receipt["contact_block_summary"], descend["contact_block_summary"])
        text = contract.actor_motion_instructions(adapter.deployment)
        self.assertIn(harness.embodiment.stall_guard_semantics, text)
        self.assertIn("With contact_stop any of these stops holds the measured pose", text)
        # Only the names the schema offers (off, contact_stop, strict), each with its threshold.
        self.assertIn(f"stall_guard_preset (default off; contact_stop {threshold:g} m; strict ", text)

    def test_capabilities_and_preview_come_from_the_profile(self):
        """A20 (execute_if_clean in modes), C16 (no forged link6 key), R8 (joint travel reported), Q3 wording."""
        harness, adapter, _ = self.harness()
        profile, capabilities = harness.embodiment, adapter.capabilities()
        self.assertIn("execute_if_clean", capabilities["unified_action"]["modes"])
        self.assertIn("execute_if_clean", profile.supports)
        self.assertEqual((capabilities["unified_action"]["max_stages"], capabilities["unified_action"]["max_stages_per_arm"]),
                         (profile.max_stages, profile.stages_per_arm))
        self.assertNotIn("grasp_position_m", json.dumps(capabilities))
        self.assertNotIn("another robot", json.dumps(capabilities))
        self.assertEqual(capabilities["arms"], list(profile.arms))
        stage = self.move_stage(adapter, dx=0.03)
        preview = adapter.preview_batch(batch_action(stage))
        for sample in preview["trajectory_data"]["samples"]:
            links = sample["arms"]["right"]["link_positions_m"]
            self.assertNotIn("link6", links)
            self.assertIn(profile.tcp_link, links)
        start = adapter.read()["q_rad"]
        plan = preview["stage_plans"]["m"]
        np.testing.assert_allclose(plan["joint_travel_rad"], np.abs(np.subtract(plan["q_rad"], start)), atol=1e-4)
        # execute_if_clean is honoured: a planned SO-101 preview executes in the same call.
        from ni_rua.runtime.actions.unified_actions import dispatch
        with patch("ni_rua.runtime.actions.inspection._trajectory_image", return_value=({"status": "unavailable"}, [])):
            response = dispatch(harness, dict(mode="execute_if_clean", observation_id=harness.latest["observation_id"],
                                              stages=[self.move(adapter, "m", dz=0.01)]), {"tool_call_id": "clean"})
        self.assertEqual(response.data["receipt"]["status"], "completed", response.data)

    def test_so101_tick_level_carry_slip_guard_scopes_streak_release_and_validation(self):
        import threading
        from ni_rua.robot.motion.batch_scheduler import MotionControllerPort

        jaw_held = (jaw_rad_for_opening(CUBE_M) + 0.174533) / (1.74533 + 0.174533)

        def make_compiled_batch(adapter, close_fields, *, include_release=False, extra_moves=True):
            open_leaf = self.compiled(adapter, "gripper", goal_normalized=0.5)
            close_leaf = self.compiled(adapter, "gripper", goal_normalized=0.0)
            stages = [
                dict(id="open", kind="gripper", arm="right", action=open_leaf),
                dict(id="close", kind="gripper", arm="right", action=close_leaf, **close_fields),
                self.move_stage(adapter, "lift", dx=0.04),
            ]
            if include_release:
                rel_leaf = self.compiled(adapter, "gripper", goal_normalized=0.7)
                stages.append(dict(id="release", kind="gripper", arm="right", action=rel_leaf))
            if extra_moves:
                stages.append(self.move_stage(adapter, "carry", dx=0.0))
            return batch_action(*stages)

        # 1) Validation rejections on SO-101 preview_batch
        _, adapter_v, _ = self.harness(jaw_min_normalized=jaw_held)
        self.assertIsInstance(adapter_v, MotionControllerPort)
        bad = adapter_v.preview_batch(make_compiled_batch(adapter_v, dict(grip_guard_preset="none", carry_slip_guard=True)))
        self.assertEqual(bad["status"], "failed")
        self.assertIn("invalid_carry_slip_requires_min_aperture", bad["reason"])

        bad_scope = adapter_v.preview_batch(
            make_compiled_batch(adapter_v, dict(grip_guard_preset="rigid_object", carry_slip_scope="next_move"))
        )
        self.assertEqual(bad_scope["status"], "failed")
        self.assertIn("invalid_carry_slip_guard_disabled", bad_scope["reason"])

        # 2) Mid-segment drop on control tick 1 of lift aborts remaining ticks immediately & skips dependent carry
        _, adapter1, driver1 = self.harness(jaw_min_normalized=jaw_held)
        act1 = make_compiled_batch(adapter1, dict(grip_guard_preset="rigid_object", carry_slip_guard=True))
        prev1 = adapter1.preview_batch(act1)
        self.assertEqual(prev1["status"], "planned", prev1)
        lift_planned = prev1["stage_costs"]["lift"]["native_steps"]
        self.assertGreaterEqual(lift_planned, 4)
        orig_tick1 = adapter1._tick
        plan1_state = adapter1._plans[prev1["plan_id"]]
        lift_ticks = 0

        def drop_on_second_lift_tick(run, period):
            nonlocal lift_ticks
            sample = orig_tick1(run, period)
            if plan1_state["status"] == "executing" and plan1_state["order"][plan1_state["index"]] == "lift":
                lift_ticks += 1
                if lift_ticks >= 2:
                    driver1.jaw_min = None
                    return dict(sample, gripper_rad=driver1.deployment["gripper"]["jaw_closed_rad"])
            return sample

        with patch.object(adapter1, "_tick", side_effect=drop_on_second_lift_tick):
            rcpt1 = adapter1.execute(dict(act1, batch_plan_id=prev1["plan_id"]), lambda _: None, threading.Event())
        self.assertEqual(rcpt1["status"], "stopped", rcpt1)
        self.assertEqual(rcpt1["reason"], "observation_checkpoint")
        self.assertEqual(rcpt1["checkpoint_stage_id"], "lift")
        self.assertEqual(rcpt1["checkpoint_reason"], "carry_slip_aperture_below_min_m")
        self.assertEqual(rcpt1["checkpoint_reason_code"], "carry_slip_aperture_below_min_m")
        self.assertEqual(rcpt1["stage_results"]["lift"]["status"], "paused_at_checkpoint")
        self.assertEqual(rcpt1["stage_results"]["carry"]["status"], "pending")
        self.assertEqual(rcpt1["pending_stage_ids"], ["lift", "carry"])
        self.assertEqual(rcpt1["slip_detected_at"]["stage_id"], "lift")
        self.assertEqual(rcpt1["slip_detected_at"]["arm"], "right")
        self.assertEqual(rcpt1["slip_detected_at"]["step_index"], 1)
        self.assertEqual(rcpt1["slip_detected_at"]["consecutive_steps"], 1)
        self.assertEqual(rcpt1["effective_grip_guard"]["carry_slip_guard"], True)

        # 3) Scope "next_move" vs "until_release" on 2nd move ("carry"), plus transient dip absorbed with slip_consecutive_steps=2
        for scope, expected in (("next_move", "completed"), ("until_release", "stopped")):
            with self.subTest(scope=scope):
                _, ad_s, drv_s = self.harness(jaw_min_normalized=jaw_held)
                act_s = make_compiled_batch(
                    ad_s,
                    dict(grip_guard_preset="rigid_object", carry_slip_guard=True, carry_slip_scope=scope, slip_consecutive_steps=2),
                )
                prev_s = ad_s.preview_batch(act_s)
                plan_s = ad_s._plans[prev_s["plan_id"]]
                orig_tick_s = ad_s._tick
                lift_tick = 0
                carry_tick = 0

                def transient_lift_then_carry_drop(run, period):
                    nonlocal lift_tick, carry_tick
                    sample = orig_tick_s(run, period)
                    if plan_s["status"] != "executing" or plan_s["index"] >= len(plan_s["order"]):
                        return sample
                    active_stage = plan_s["order"][plan_s["index"]]
                    if active_stage == "lift":
                        lift_tick += 1
                        if lift_tick == 1:
                            # Single-tick transient dip during lift -> absorbed by slip_consecutive_steps=2!
                            return dict(sample, gripper_rad=drv_s.deployment["gripper"]["jaw_closed_rad"])
                    elif active_stage == "carry" and ad_s.motion_phase == "moving":
                        carry_tick += 1
                        if carry_tick >= 1:
                            drv_s.jaw_min = None
                            return dict(sample, gripper_rad=drv_s.deployment["gripper"]["jaw_closed_rad"])
                    return sample

                with patch.object(ad_s, "_tick", side_effect=transient_lift_then_carry_drop):
                    rcpt_s = ad_s.execute(dict(act_s, batch_plan_id=prev_s["plan_id"]), lambda _: None, threading.Event())
                self.assertEqual(rcpt_s["status"], expected, rcpt_s)
                if expected == "stopped":
                    self.assertEqual(rcpt_s["checkpoint_stage_id"], "carry")
                    self.assertEqual(rcpt_s["slip_detected_at"]["consecutive_steps"], 2)

        # 4) Release stage (goal=0.7 >= 0.5) disarms until_release before subsequent move
        _, ad_r, drv_r = self.harness(jaw_min_normalized=jaw_held)
        act_r = make_compiled_batch(
            ad_r,
            dict(grip_guard_preset="rigid_object", carry_slip_guard=True, carry_slip_scope="until_release"),
            include_release=True,
        )
        prev_r = ad_r.preview_batch(act_r)
        plan_r = ad_r._plans[prev_r["plan_id"]]
        orig_gripper = ad_r._gripper
        orig_tick_r = ad_r._tick

        def drop_on_release(run, goal_normalized, gripper_speed_scale, squeeze_rad=None):
            if goal_normalized >= 0.6:
                drv_r.jaw_min = None
            return orig_gripper(run, goal_normalized, gripper_speed_scale, squeeze_rad)

        def zero_aperture_on_carry(run, period):
            sample = orig_tick_r(run, period)
            if plan_r["status"] == "executing" and plan_r["index"] < len(plan_r["order"]) and plan_r["order"][plan_r["index"]] == "carry":
                return dict(sample, gripper_rad=drv_r.deployment["gripper"]["jaw_closed_rad"])
            return sample

        with patch.object(ad_r, "_gripper", side_effect=drop_on_release), patch.object(
            ad_r, "_tick", side_effect=zero_aperture_on_carry
        ):
            rcpt_r = ad_r.execute(dict(act_r, batch_plan_id=prev_r["plan_id"]), lambda _: None, threading.Event())
        self.assertEqual(rcpt_r["status"], "completed", rcpt_r)
        self.assertEqual(rcpt_r["completed_stage_ids"], ["open", "close", "lift", "release", "carry"])

        # 5) Carry-slip during _arrive settling AND ArrivalCorrection ticks pauses before finish_stage_carry_slip
        for subcase, sag in (("settle_ticks", (0.0, 0.0, 0.0, 0.0, 0.0)), ("arrival_correction_ticks", (0.0, 0.03, 0.0, 0.0, 0.0))):
            with self.subTest(arrive_phase=subcase):
                _, ad_a, drv_a = self.harness(jaw_min_normalized=jaw_held, sag_rad=sag)
                act_a = make_compiled_batch(
                    ad_a,
                    dict(grip_guard_preset="rigid_object", carry_slip_guard=True, slip_consecutive_steps=2),
                )
                prev_a = ad_a.preview_batch(act_a)
                self.assertEqual(prev_a["status"], "planned", prev_a)
                lift_steps = prev_a["stage_costs"]["lift"]["native_steps"]
                orig_tick_a = ad_a._tick
                in_arrive = {"active": False, "correction_sent": False}
                orig_arrive = ad_a._arrive
                orig_send = ad_a._send

                def spy_send(q_rad, jaw_normalized, generation=None, run=None, kind=None, **kw):
                    if kind == "arrival_correction":
                        in_arrive["correction_sent"] = True
                    return orig_send(q_rad, jaw_normalized, generation=generation, run=run, kind=kind, **kw)

                def spy_arrive(run, q_target, pose, tolerances, **kw):
                    in_arrive["active"] = True
                    try:
                        return orig_arrive(run, q_target, pose, tolerances, **kw)
                    finally:
                        in_arrive["active"] = False

                def slip_in_arrive_ticks(run, period):
                    sample = orig_tick_a(run, period)
                    trigger_now = (
                        in_arrive["active"]
                        and ad_a.motion_phase == "moving"
                        and (subcase == "settle_ticks" or in_arrive["correction_sent"])
                    )
                    if trigger_now:
                        drv_a.jaw_min = None
                        return dict(sample, gripper_rad=drv_a.deployment["gripper"]["jaw_closed_rad"])
                    return sample

                with patch.object(ad_a, "_send", side_effect=spy_send), patch.object(
                    ad_a, "_arrive", side_effect=spy_arrive
                ), patch.object(ad_a, "_tick", side_effect=slip_in_arrive_ticks):
                    rcpt_a = ad_a.execute(dict(act_a, batch_plan_id=prev_a["plan_id"]), lambda _: None, threading.Event())
                if subcase == "arrival_correction_ticks":
                    self.assertTrue(in_arrive["correction_sent"])
                self.assertEqual(rcpt_a["status"], "stopped", rcpt_a)
                self.assertEqual(rcpt_a["checkpoint_stage_id"], "lift")
                self.assertEqual(rcpt_a["checkpoint_reason_code"], "carry_slip_aperture_below_min_m")
                self.assertGreaterEqual(rcpt_a["slip_detected_at"]["step_index"], lift_steps)
                self.assertEqual(rcpt_a["slip_detected_at"]["consecutive_steps"], 2)
                self.assertEqual(rcpt_a["stage_results"]["carry"]["status"], "pending")

        # 6) until_release survives carry-slip pause + resume on SO-101 and catches slip on subsequent move
        _, ad_u, drv_u = self.harness(jaw_min_normalized=jaw_held)
        act_u = make_compiled_batch(
            ad_u,
            dict(grip_guard_preset="rigid_object", carry_slip_guard=True, carry_slip_scope="until_release"),
        )
        prev_u = ad_u.preview_batch(act_u)
        plan_u = ad_u._plans[prev_u["plan_id"]]
        orig_tick_u = ad_u._tick
        run_num = {"n": 1}

        def slip_lift_then_carry(run, period):
            sample = orig_tick_u(run, period)
            if plan_u["status"] != "executing" or plan_u["index"] >= len(plan_u["order"]):
                return sample
            sid = plan_u["order"][plan_u["index"]]
            if run_num["n"] == 1 and sid == "lift" and ad_u.motion_phase == "moving":
                return dict(sample, gripper_rad=drv_u.deployment["gripper"]["jaw_closed_rad"])
            if run_num["n"] == 2 and sid == "carry" and ad_u.motion_phase == "moving":
                drv_u.jaw_min = None
                return dict(sample, gripper_rad=drv_u.deployment["gripper"]["jaw_closed_rad"])
            return sample

        with patch.object(ad_u, "_tick", side_effect=slip_lift_then_carry):
            rcpt_u1 = ad_u.execute(dict(act_u, batch_plan_id=prev_u["plan_id"]), lambda _: None, threading.Event())
            self.assertEqual((rcpt_u1["status"], rcpt_u1["checkpoint_stage_id"]), ("stopped", "lift"), rcpt_u1)
            self.assertIn("right", plan_u["active_carry_slip_monitors"])
            run_num["n"] = 2
            rcpt_u2 = ad_u.execute(dict(act_u, batch_plan_id=prev_u["plan_id"]), lambda _: None, threading.Event())
        self.assertEqual((rcpt_u2["status"], rcpt_u2["checkpoint_stage_id"]), ("stopped", "carry"), rcpt_u2)
        self.assertEqual(rcpt_u2["checkpoint_reason_code"], "carry_slip_aperture_below_min_m")
        self.assertIn("lift", rcpt_u2["completed_stage_ids"])

    def test_so101_batch_pause_and_carry_slip_driver_fault_attribution_and_hygiene(self):
        import threading
        from ni_rua.robot.backends.so101.worker import DriverFault

        jaw_held = (jaw_rad_for_opening(CUBE_M) + 0.174533) / (1.74533 + 0.174533)

        # 1) DriverFault in _pause_batch (self.read()) during an observe stage blames "look", not "back"
        _, ad_obs, _ = self.harness()
        act_obs = batch_action(
            self.move_stage(ad_obs, "reach", dx=0.01),
            dict(id="look", kind="observe", reason="verify"),
            self.move_stage(ad_obs, "back", dx=0.0),
        )
        prev_obs = ad_obs.preview_batch(act_obs)
        orig_pause_obs = ad_obs._pause_batch

        def fault_in_pause(run, plan, receipt, stage_ids, reasons, **kw):
            ad_obs.hold(run)
            raise DriverFault("driver_read_failed: pause read fault")

        with patch.object(ad_obs, "_pause_batch", side_effect=fault_in_pause):
            rcpt_obs = ad_obs.execute(dict(act_obs, batch_plan_id=prev_obs["plan_id"]), lambda _: None, threading.Event())
        self.assertEqual(rcpt_obs["status"], "unknown", rcpt_obs)
        self.assertFalse(rcpt_obs["paused_at_checkpoint"])
        self.assertEqual(rcpt_obs["failed_stage_id"], "look")
        self.assertEqual(rcpt_obs["completed_stage_ids"], ["reach"])
        self.assertEqual(rcpt_obs["pending_stage_ids"], ["look", "back"])
        self.assertEqual(rcpt_obs["stage_results"]["look"]["status"], "unknown")
        self.assertEqual(rcpt_obs["stage_results"]["back"]["status"], "pending")

        # 2) DriverFault in _pause_batch (self.read()) during post-stage grip_guard blames "close", not "lift"
        _, ad_gg, _ = self.harness(gripper_normalized=1.0)
        act_gg = batch_action(
            dict(id="close", kind="gripper", arm="right", action=self.compiled(ad_gg, "gripper", goal_normalized=0.0), grip_guard_preset="rigid_object"),
            self.move_stage(ad_gg, "lift", dx=0.02),
        )
        prev_gg = ad_gg.preview_batch(act_gg)
        orig_read_gg = ad_gg.read
        in_pause_gg = {"active": False}
        orig_pause_gg = ad_gg._pause_batch

        def wrapped_pause_gg(run, plan, receipt, stage_ids, reasons, **kw):
            in_pause_gg["active"] = True
            try:
                return orig_pause_gg(run, plan, receipt, stage_ids, reasons, **kw)
            finally:
                in_pause_gg["active"] = False

        def fault_read_in_pause():
            if in_pause_gg["active"]:
                raise DriverFault("driver_read_failed: post-grip-guard read fault")
            return orig_read_gg()

        with patch.object(ad_gg, "_pause_batch", side_effect=wrapped_pause_gg), patch.object(
            ad_gg, "read", side_effect=fault_read_in_pause
        ):
            rcpt_gg = ad_gg.execute(dict(act_gg, batch_plan_id=prev_gg["plan_id"]), lambda _: None, threading.Event())
        self.assertEqual(rcpt_gg["status"], "unknown", rcpt_gg)
        self.assertFalse(rcpt_gg["paused_at_checkpoint"])
        self.assertEqual(rcpt_gg["failed_stage_id"], "close")
        self.assertEqual(rcpt_gg["completed_stage_ids"], [])
        self.assertEqual(rcpt_gg["pending_stage_ids"], ["close", "lift"])
        self.assertEqual(rcpt_gg["stage_results"]["close"]["status"], "unknown")
        self.assertEqual(rcpt_gg["stage_results"]["lift"]["status"], "pending")

        # 3) DriverFault inside carry-slip _sync() clears paused_at_checkpoint=False and attributes "lift"
        _, ad_cs, _ = self.harness(jaw_min_normalized=jaw_held)
        act_cs = batch_action(
            dict(id="open", kind="gripper", arm="right", action=self.compiled(ad_cs, "gripper", goal_normalized=0.5)),
            dict(
                id="close",
                kind="gripper",
                arm="right",
                action=self.compiled(ad_cs, "gripper", goal_normalized=0.0),
                grip_guard_preset="rigid_object",
                carry_slip_guard=True,
            ),
            self.move_stage(ad_cs, "lift", dx=0.04),
        )
        prev_cs = ad_cs.preview_batch(act_cs)
        plan_cs = ad_cs._plans[prev_cs["plan_id"]]
        orig_tick_cs = ad_cs._tick
        orig_read_cs = ad_cs.read
        slip_triggered = {"done": False}

        def trigger_slip_tick(run, period):
            sample = orig_tick_cs(run, period)
            if (
                plan_cs["status"] == "executing"
                and plan_cs.get("active_stage_id") == "lift"
                and ad_cs.motion_phase == "moving"
            ):
                slip_triggered["done"] = True
                return dict(sample, gripper_rad=ad_cs.deployment["gripper"]["jaw_closed_rad"])
            return sample

        def fault_on_sync_read():
            if slip_triggered["done"] and ad_cs.motion_phase == "holding":
                slip_triggered["done"] = False
                raise DriverFault("driver_read_failed: carry-slip sync read fault")
            return orig_read_cs()

        with patch.object(ad_cs, "_tick", side_effect=trigger_slip_tick), patch.object(
            ad_cs, "read", side_effect=fault_on_sync_read
        ):
            rcpt_cs = ad_cs.execute(dict(act_cs, batch_plan_id=prev_cs["plan_id"]), lambda _: None, threading.Event())
        self.assertEqual(rcpt_cs["status"], "unknown", rcpt_cs)
        self.assertFalse(rcpt_cs["paused_at_checkpoint"])
        self.assertNotIn("checkpoint_reason", rcpt_cs)
        self.assertNotIn("slip_detected_at", rcpt_cs)
        self.assertEqual(rcpt_cs["failed_stage_id"], "lift")
        self.assertEqual(rcpt_cs["completed_stage_ids"], ["open", "close"])
        self.assertEqual(rcpt_cs["pending_stage_ids"], ["lift"])
        self.assertEqual(rcpt_cs["stage_results"]["lift"]["status"], "unknown")
        self.assertFalse(plan_cs["executable"])


if __name__ == "__main__":
    unittest.main()
