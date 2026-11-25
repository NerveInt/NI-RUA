"""SO-101 bounded holding squeeze on CPU (fake driver, fake cameras, real kinematics).

Regression for a real-arm defect (2026-11-16): a closing jaw stage left the jaw goal at the closed limit, so the
position-controlled STS3215 pushed at its torque limit for as long as the object (or the other fingertip) held the jaw
open; after about ten carry cycles the gripper servo read 46 degC and stopped answering. With deployment
``gripper.hold_squeeze_rad`` the worker sets the jaw goal to the measured contact angle minus the squeeze.
These tests prove the software contract (which goal is commanded and what is reported), not the holding force.
"""

NI_RUA_TEST_TIER = "component"

from copy import deepcopy
import dataclasses
import json
import tempfile
import unittest
from unittest.mock import patch

import jsonschema

from ni_rua.robot.backends.so101 import contract
from ni_rua.robot.backends.so101.deployment import embodiment_profile, hold_squeeze, validate_deployment
from ni_rua.robot.backends.so101.driver import (
    ARM_JOINTS,
    LeRobotSO101Driver,
    STS_STATUS_ADDR,
    STS_STATUS_LENGTH,
    decode_sts_status_block,
    gripper_normalized_from_rad,
    lerobot_max_relative_target,
    sts_current_ma,
    sts_load_pct,
)
from ni_rua.robot.model import embodiments
from ni_rua.robot.model.so101_gripper_geometry import jaw_rad_for_opening
from ni_rua.robot.motion.grip_guard import validate_grip_squeeze
from ni_rua.runtime.actions import robot_action_schema
from robot.so101.test_so101_backend import EXAMPLE, IDENTITY, deployment, make_adapter
from robot.so101.test_so101_feedback_grip_guard import CUBE_M, EMPTY_CLOSE_M, HarnessCase

SQUEEZE = 0.02  # rad of the jaw joint: the deployment default of these tests
JAW_RANGE = 1.74533 + 0.174533
CUBE_RAD = jaw_rad_for_opening(CUBE_M)
CUBE_JAW = (CUBE_RAD + 0.174533) / JAW_RANGE  # normalized jaw opening at which the fake cube stops the jaw


def squeeze_deployment(**updates):
    return deployment(**{"gripper.empty_close_aperture_m": EMPTY_CLOSE_M, "gripper.hold_squeeze_rad": SQUEEZE, **updates})


REFUSED = (ValueError, jsonschema.ValidationError)  # the published schema or the validators behind it


class SqueezeCase(HarnessCase):
    calls = 0

    def run_batch(self, harness, stages, **batch):
        """Preview + execute with a tool call id of its own, so one harness can run several batches."""
        from ni_rua.runtime.actions.unified_actions import dispatch
        SqueezeCase.calls += 1
        request = dict(observation_id=harness.latest["observation_id"], stages=stages, **batch)
        with patch("ni_rua.runtime.actions.inspection._trajectory_image", return_value=({"status": "unavailable"}, [])):
            preview = dispatch(harness, request, {"tool_call_id": f"preview-{SqueezeCase.calls}"})
        self.assertEqual(preview.data["status"], "planned", preview.data)
        return preview, self.execute(harness, preview.data["plan_id"])

    def grasp(self, adapter, close_fields=None, lift_fields=None):
        return [dict(id="open", kind="gripper", arm="right", goal_normalized=0.5),
                dict(id="close", kind="gripper", arm="right", goal_normalized=0.0, **(close_fields or {})),
                self.move(adapter, "lift", dz=0.03, **(lift_fields or {}))]

    def held(self, **driver_options):
        """A harness whose fake jaw is stopped by a cube, on a deployment with the default squeeze."""
        return self.harness(deploy=driver_options.pop("deploy", None) or squeeze_deployment(),
                            jaw_min_normalized=CUBE_JAW, **driver_options)


class DeploymentDefaultOffTests(SqueezeCase):
    def test_example_deployment_applies_no_squeeze_and_offers_no_field(self):
        deploy = deployment(**{"gripper.empty_close_aperture_m": EMPTY_CLOSE_M})
        self.assertIsNone(deploy["gripper"]["hold_squeeze_rad"])
        self.assertIsNone(deploy["gripper"]["max_hold_squeeze_rad"])
        self.assertIsNone(hold_squeeze(deploy))
        profile = embodiment_profile(deploy)
        self.assertIsNone(profile.gripper.hold_squeeze)
        built = robot_action_schema.build(profile)
        for kind in ("move", "gripper"):
            self.assertNotIn("grip_squeeze_rad", built.stage_schemas[kind]["properties"])
        self.assertNotIn("grip_squeeze_rad", built.stage_field_help)
        # The archived profile record of a deployment without the key is the record written before the key existed.
        self.assertNotIn("hold_squeeze", embodiments.profile_record(profile)["profile"]["gripper"])
        self.assertNotIn("Holding squeeze", contract.actor_motion_instructions(deploy))
        self.assertEqual(contract.hold_squeeze_capability(deploy), {})
        self.assertTrue(contract.post_grasp_recheck_available(deploy))

    def test_a_deployment_dict_from_before_the_keys_is_off(self):
        deploy = deployment()
        for key in ("hold_squeeze_rad", "max_hold_squeeze_rad"):
            del deploy["gripper"][key]
        self.assertIsNone(hold_squeeze(deploy))
        self.assertIsNone(embodiment_profile(deploy).gripper.hold_squeeze)

    def test_without_the_key_a_close_on_an_object_keeps_the_commanded_goal(self):
        harness, adapter, driver = self.harness(jaw_min_normalized=CUBE_JAW)
        _, response = self.run_batch(harness, self.grasp(adapter))
        receipt = response.data["receipt"]
        self.assertEqual(receipt["status"], "completed", receipt)
        self.assertNotIn("hold_squeeze", receipt["stage_results"]["close"]["gripper_execution"])
        self.assertNotIn("grip_hold", receipt["stage_results"]["lift"])
        self.assertEqual(adapter.jaw_goal, 0.0)
        self.assertAlmostEqual(driver.jaw_goal, -0.174533)
        self.assertIsNone(adapter.jaw_hold)
        self.assertNotIn("hold_squeeze", adapter.capabilities()["gripper_semantics"])

    def test_without_the_key_the_stage_field_is_refused(self):
        harness, adapter, _ = self.harness(jaw_min_normalized=CUBE_JAW)
        from ni_rua.runtime.actions.unified_actions import dispatch
        stages = self.grasp(adapter, dict(grip_squeeze_rad=0.01))
        with self.assertRaisesRegex(REFUSED, "grip_squeeze_rad"):
            dispatch(harness, dict(observation_id=harness.latest["observation_id"], stages=stages), {"tool_call_id": "p"})
        with self.assertRaisesRegex(ValueError, "unsupported_robot_action_stage_fields"):
            from ni_rua.runtime.actions.unified_actions import compile_request
            compile_request(harness, dict(observation_id=harness.latest["observation_id"], stages=stages))
        with self.assertRaisesRegex(ValueError, "unsupported_capability: grip_squeeze_rad"):
            validate_grip_squeeze({"grip_squeeze_rad": 0.01}, profile=adapter.embodiment)
        with self.assertRaisesRegex(ValueError, "unsupported_capability: grip_squeeze_rad"):
            validate_grip_squeeze({"grip_squeeze_rad": 0.01})  # the default robot (ARX-X5)

    def test_the_default_robot_schema_has_no_squeeze_field(self):
        built = robot_action_schema.build(None)
        self.assertNotIn("grip_squeeze_rad", json.dumps(built.tool))
        self.assertIsNone(embodiments.ARX_X5_PROFILE.gripper.hold_squeeze)


class DeploymentValidationTests(unittest.TestCase):
    def document(self, **gripper):
        document = json.loads(EXAMPLE.read_text())
        document["gripper"].update(gripper)
        return document

    def test_ranges(self):
        loaded = validate_deployment(self.document(hold_squeeze_rad=0.02), base=EXAMPLE.parent)
        self.assertEqual(hold_squeeze(loaded), (0.02, JAW_RANGE))
        loaded = validate_deployment(self.document(hold_squeeze_rad=0.0, max_hold_squeeze_rad=0.1), base=EXAMPLE.parent)
        self.assertEqual(hold_squeeze(loaded), (0.0, 0.1))
        self.assertEqual(embodiment_profile(loaded).gripper.hold_squeeze, embodiments.HoldSqueeze(0.0, 0.1))
        for gripper, message in (
                (dict(hold_squeeze_rad=-0.01), "out_of_range:gripper.hold_squeeze_rad"),
                (dict(hold_squeeze_rad=2.0), "out_of_range:gripper.hold_squeeze_rad"),
                (dict(hold_squeeze_rad="firm"), "gripper.hold_squeeze_rad: finite number required"),
                (dict(hold_squeeze_rad=0.02, max_hold_squeeze_rad=0.0), "out_of_range:gripper.max_hold_squeeze_rad"),
                (dict(hold_squeeze_rad=0.05, max_hold_squeeze_rad=0.02), "max_hold_squeeze_rad requires"),
                (dict(max_hold_squeeze_rad=0.1), "max_hold_squeeze_rad requires hold_squeeze_rad"),
                (dict(hold_squeeze_rad=0.6, jaw_open_rad=0.3), "hold_squeeze_rad exceeds the jaw range"),
                (dict(hold_squeeze_rad=0.02, max_hold_squeeze_rad=0.6, jaw_open_rad=0.3), "max_hold_squeeze_rad requires")):
            with self.subTest(gripper=gripper), self.assertRaisesRegex(ValueError, message):
                validate_deployment(self.document(**gripper), base=EXAMPLE.parent)

    def test_profile_bounds_and_record_round_trip(self):
        profile = embodiment_profile(squeeze_deployment(**{"gripper.max_hold_squeeze_rad": 0.08}))
        self.assertEqual(profile.gripper.hold_squeeze, embodiments.HoldSqueeze(SQUEEZE, 0.08))
        record = embodiments.profile_record(profile)
        self.assertEqual(record["profile"]["gripper"]["hold_squeeze"], dict(default_rad=SQUEEZE, max_rad=0.08))
        self.assertEqual(embodiments.profile_from_record(json.loads(json.dumps(record))), profile)
        for bad in (embodiments.HoldSqueeze(0.1, 0.05), embodiments.HoldSqueeze(-0.01, 0.05),
                    embodiments.HoldSqueeze(0.0, 0.0), (0.02, 0.05)):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "invalid_embodiment_profile:gripper.hold_squeeze"):
                dataclasses.replace(profile, gripper=dataclasses.replace(profile.gripper, hold_squeeze=bad))

    def test_documentation_states_unit_default_and_fingertip_travel(self):
        deploy = squeeze_deployment()
        # The documented fingertip equivalents are the CAD ones (so101_gripper_geometry).
        self.assertAlmostEqual(contract.squeeze_fingertip_travel_m(0.01, 0.029), 0.0008, delta=0.00005)
        self.assertAlmostEqual(contract.squeeze_fingertip_travel_m(0.01, 0.005), 0.00065, delta=0.00005)
        self.assertAlmostEqual(contract.squeeze_fingertip_travel_m(2 * 3.141592653589793 / 4096, 0.029), 0.00012, delta=0.00002)
        self.assertIsNone(contract.squeeze_fingertip_travel_m(0.01, 0.5))
        from ni_rua.robot.backends.so101.deployment import FIELDS
        for key in ("hold_squeeze_rad", "max_hold_squeeze_rad"):
            spec = FIELDS["gripper"][key]
            self.assertEqual((spec["unit"], spec["default"], spec["nullable"]), ("rad", None, True))
            self.assertIn("rad of the jaw joint", spec["doc"])
            self.assertIn("Default null", spec["doc"])
        self.assertIn("0.8 mm", FIELDS["gripper"]["hold_squeeze_rad"]["doc"])
        example = json.loads(EXAMPLE.read_text())["gripper"]
        self.assertIsNone(example["hold_squeeze_rad"])
        self.assertIn("rad", example["_hold_squeeze_rad"])
        text = contract.actor_motion_instructions(deploy)
        self.assertIn("Holding squeeze", text)
        self.assertIn("Default 0.02 rad of the jaw joint (about 1.6 mm of fingertip gap near a 29 mm opening", text)
        self.assertIn("There is no post-grasp slip re-check with the holding squeeze", text)
        self.assertEqual(contract.hold_squeeze_capability(deploy)["hold_squeeze"]["default_rad"], SQUEEZE)
        self.assertAlmostEqual(contract.hold_squeeze_capability(deploy)["hold_squeeze"]["default_fingertip_travel_m"],
                               0.0016, delta=0.0001)
        help_text = robot_action_schema.build(embodiment_profile(deploy)).stage_field_help["grip_squeeze_rad"]
        for expected in ("radians of the jaw joint", "default 0.02", "[0, 1.91986]", "Example:"):
            self.assertIn(expected, help_text)


class SqueezeGoalTests(SqueezeCase):
    def test_close_on_an_object_sets_the_goal_to_contact_minus_squeeze(self):
        harness, adapter, driver = self.held()
        driver.arm_load = lambda names: {name: 190.0 for name in names}  # raw 0.1 % units
        preview, response = self.run_batch(harness, self.grasp(adapter))
        self.assertEqual(preview.data["stage_plans"]["close"]["hold_squeeze_rad"], SQUEEZE)
        self.assertNotIn("hold_squeeze_rad", preview.data["stage_plans"]["lift"])
        receipt = response.data["receipt"]
        self.assertEqual(receipt["status"], "completed", receipt)
        jaw = receipt["stage_results"]["close"]["gripper_execution"]
        squeeze = jaw["hold_squeeze"]
        self.assertTrue(squeeze["applied"])
        self.assertEqual(squeeze["squeeze_rad"], SQUEEZE)
        self.assertAlmostEqual(squeeze["contact_jaw_rad"], CUBE_RAD, delta=1e-3)
        self.assertAlmostEqual(squeeze["contact_opening_m"], CUBE_M, delta=2e-4)
        self.assertAlmostEqual(squeeze["hold_goal_rad"], squeeze["contact_jaw_rad"] - SQUEEZE, delta=1e-4)
        self.assertAlmostEqual(squeeze["hold_goal_normalized"],
                               gripper_normalized_from_rad(adapter.deployment, squeeze["hold_goal_rad"]), delta=1e-4)
        self.assertEqual((squeeze["contact_load_pct"], squeeze["hold_load_pct"]), (19.0, 19.0))
        # The commanded goal of the stage is still reported; the reading is the one after settling at the hold goal.
        self.assertEqual(jaw["goal_normalized"], 0.0)
        self.assertTrue(jaw["jaw_settled"])
        self.assertAlmostEqual(jaw["measured_opening_m"], CUBE_M, delta=2e-4)
        self.assertAlmostEqual(receipt["stage_results"]["close"]["grip_facts"]["measured_aperture_m"], CUBE_M, delta=2e-4)
        # The servo goal the driver holds, in rad: the measured contact angle minus the squeeze, not the closed limit.
        self.assertAlmostEqual(driver.jaw_goal, CUBE_RAD - SQUEEZE, delta=1e-3)
        self.assertAlmostEqual(adapter.jaw_goal, squeeze["hold_goal_normalized"], delta=1e-4)
        self.assertEqual(adapter.jaw_command, 0.0)
        # One dispatched command sets it, labelled in the planned trace.
        stored = harness.store.read("action_receipt", response.data["action_id"])
        rows = json.loads(harness.store.resolve_artifact(stored["planned_trace_ref"], ("actor",))[1])
        self.assertEqual(rows["row_kinds"].count("hold_squeeze"), 1)

    def test_receipt_without_a_load_reading_omits_the_load_fields(self):
        harness, adapter, driver = self.held()
        self.assertFalse(hasattr(driver, "arm_load"))
        _, response = self.run_batch(harness, self.grasp(adapter))
        squeeze = response.data["receipt"]["stage_results"]["close"]["gripper_execution"]["hold_squeeze"]
        self.assertTrue(squeeze["applied"])
        self.assertNotIn("contact_load_pct", squeeze)
        self.assertNotIn("hold_load_pct", squeeze)
        driver.arm_load = lambda names: (_ for _ in ()).throw(OSError("bus"))
        self.assertIsNone(adapter._jaw_load())  # a failed read is "not measured", never an exception

    def test_no_squeeze_without_contact_and_none_on_opening(self):
        harness, adapter, driver = self.harness(deploy=squeeze_deployment())  # nothing stops the fake jaw
        _, response = self.run_batch(harness, self.grasp(adapter))
        receipt = response.data["receipt"]
        self.assertEqual(receipt["status"], "completed", receipt)
        # An opening command reports nothing; a close that reached its goal says why no squeeze was applied.
        self.assertNotIn("hold_squeeze", receipt["stage_results"]["open"]["gripper_execution"])
        self.assertEqual(receipt["stage_results"]["close"]["gripper_execution"]["hold_squeeze"],
                         dict(squeeze_rad=SQUEEZE, applied=False, reason="jaw_within_squeeze_of_goal"))
        self.assertEqual(adapter.jaw_goal, 0.0)
        self.assertAlmostEqual(driver.jaw_goal, -0.174533)
        self.assertIsNone(adapter.jaw_hold)
        self.assertNotIn("grip_hold", receipt["stage_results"]["lift"])

    def test_partial_close_that_reaches_its_goal_is_unchanged(self):
        # The cube stops the jaw at CUBE_JAW; a goal above that is reached without contact.
        harness, adapter, driver = self.held(gripper_normalized=1.0)
        goal = CUBE_JAW + 0.1
        _, response = self.run_batch(harness, [dict(id="close", kind="gripper", arm="right", goal_normalized=goal),
                                               self.move(adapter, "lift", dz=0.03)])
        jaw = response.data["receipt"]["stage_results"]["close"]["gripper_execution"]
        self.assertEqual(jaw["hold_squeeze"]["reason"], "jaw_within_squeeze_of_goal")
        self.assertAlmostEqual(adapter.jaw_goal, goal)

    def test_opening_against_an_obstruction_is_never_squeezed(self):
        harness, adapter, driver = self.held(gripper_normalized=CUBE_JAW)
        driver.jaw_min = None
        original = driver._advance

        def blocked_open():  # the jaw cannot open beyond 0.4 normalized
            original()
            driver.jaw = min(driver.jaw, -0.174533 + 0.4 * JAW_RANGE)
        driver._advance = blocked_open
        _, response = self.run_batch(harness, [dict(id="open", kind="gripper", arm="right", goal_normalized=1.0),
                                               self.move(adapter, "lift", dz=0.03)])
        jaw = response.data["receipt"]["stage_results"]["open"]["gripper_execution"]
        self.assertNotIn("hold_squeeze", jaw)
        self.assertEqual(adapter.jaw_goal, 1.0)

    def test_unsettled_jaw_keeps_the_commanded_goal_and_the_guard_pauses(self):
        import numpy as np
        harness, adapter, driver = self.harness(deploy=squeeze_deployment(), gripper_normalized=0.5)
        advance = driver._advance

        def slow_jaw():  # still closing at 0.05 rad/s when gripper.settle_timeout_s expires
            start, before = driver.last, driver.jaw
            advance()
            limit = 0.05 * max(driver.last - start, 0.0)
            driver.jaw = before + float(np.clip(driver.jaw - before, -limit, limit))
        driver._advance = slow_jaw
        _, response = self.run_batch(harness, [dict(id="close", kind="gripper", arm="right", goal_normalized=0.0,
                                                     grip_guard_preset="non_empty"), self.move(adapter, "lift", dz=0.03)])
        receipt = response.data["receipt"]
        self.assertEqual((receipt["status"], receipt["checkpoint_reason"]), ("stopped", "jaw_not_settled"), receipt)
        jaw = receipt["stage_results"]["close"]["gripper_execution"]
        self.assertEqual(jaw["hold_squeeze"], dict(squeeze_rad=SQUEEZE, applied=False, reason="jaw_not_settled"))
        self.assertEqual(adapter.jaw_goal, 0.0)


class StageOverrideTests(SqueezeCase):
    def test_stage_value_replaces_the_deployment_default(self):
        for value in (0.05, 0.0):
            with self.subTest(grip_squeeze_rad=value):
                harness, adapter, driver = self.held()
                preview, response = self.run_batch(harness, self.grasp(adapter, dict(grip_squeeze_rad=value)))
                self.assertEqual(preview.data["stage_plans"]["close"]["hold_squeeze_rad"], value)
                squeeze = response.data["receipt"]["stage_results"]["close"]["gripper_execution"]["hold_squeeze"]
                self.assertEqual((squeeze["applied"], squeeze["squeeze_rad"]), (True, value))
                self.assertAlmostEqual(driver.jaw_goal, CUBE_RAD - value, delta=1e-3)

    def test_move_with_an_after_arrival_jaw_goal_carries_the_field_to_its_jaw_stage(self):
        harness, adapter, driver = self.held(gripper_normalized=0.5)
        stage = self.move(adapter, "grasp", dx=0.01, gripper_goal_normalized=0.0, grip_squeeze_rad=0.04)
        preview, response = self.run_batch(harness, [stage, self.move(adapter, "lift", dz=0.03)])
        self.assertEqual(preview.data["stage_plans"]["grasp"]["hold_squeeze_rad"], 0.04)
        squeeze = response.data["receipt"]["stage_results"]["grasp"]["gripper_execution"]["hold_squeeze"]
        self.assertEqual((squeeze["applied"], squeeze["squeeze_rad"]), (True, 0.04))
        self.assertAlmostEqual(driver.jaw_goal, CUBE_RAD - 0.04, delta=1e-3)

    def test_validation_refuses_out_of_range_and_misplaced_values(self):
        from ni_rua.runtime.actions.unified_actions import dispatch
        harness, adapter, _ = self.held(deploy=squeeze_deployment(**{"gripper.max_hold_squeeze_rad": 0.08}))

        from ni_rua.runtime.actions.unified_actions import compile_request
        calls = iter(range(100))

        def preview(stages):
            return dispatch(harness, dict(observation_id=harness.latest["observation_id"], stages=stages),
                            {"tool_call_id": f"p{next(calls)}"})

        def compiled(stages):  # the validators behind the published schema (helper programs reach them directly)
            return compile_request(harness, dict(observation_id=harness.latest["observation_id"], stages=stages))
        for value in (0.081, -0.001, "firm", True, float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(REFUSED):
                    preview(self.grasp(adapter, dict(grip_squeeze_rad=value)))
                with self.assertRaisesRegex(ValueError, "invalid_grip_squeeze_rad|invalid_robot_action_stage"):
                    compiled(self.grasp(adapter, dict(grip_squeeze_rad=value)))
        misplaced = [self.move(adapter, "lift", dz=0.03, grip_squeeze_rad=0.01)]  # a move without a jaw command
        with self.assertRaises(REFUSED):
            preview(misplaced)
        with self.assertRaisesRegex(ValueError, "invalid_robot_action_stage|grip_squeeze_rad"):
            compiled(misplaced)
        with self.assertRaises(REFUSED):
            preview([dict(id="look", kind="observe", reason="r", grip_squeeze_rad=0.01)])
        self.assertEqual(preview(self.grasp(adapter, dict(grip_squeeze_rad=0.08))).data["status"], "planned")
        # The schema publishes the deployment's bound as the field maximum.
        schema = robot_action_schema.build(adapter.embodiment).stage_schemas["gripper"]["properties"]["grip_squeeze_rad"]
        self.assertEqual((schema["minimum"], schema["maximum"]), (0, 0.08))
        self.assertEqual(validate_grip_squeeze({"grip_squeeze_rad": 0.08}, profile=adapter.embodiment), 0.08)
        self.assertIsNone(validate_grip_squeeze({}, profile=adapter.embodiment))

    def test_backend_validator_refuses_what_the_harness_would(self):
        deploy = squeeze_deployment(**{"gripper.max_hold_squeeze_rad": 0.08})
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        adapter, _, _ = make_adapter(directory.name, deploy=deploy)
        self.addCleanup(adapter.close)
        caps = adapter.capabilities()
        jaw = contract.compile_action("gripper", dict(arm="right", goal_normalized=0.0, observation_id="o"), IDENTITY, caps)
        move = contract.compile_action("move", dict(arm="right", position_m=[0.25, 0, 0.1], quaternion_xyzw=[0, 1, 0, 0],
                                                    observation_id="o"), IDENTITY, caps)

        def batch(*stages):
            return dict(type="BatchAction", frame_id=contract.FRAME_ID, tcp_config_id=contract.TCP_CONFIG_ID,
                        limits_profile=contract.LIMITS_PROFILE, timeout_s=30, stages=list(stages))
        close = dict(id="close", kind="gripper", arm="right", action=jaw)
        contract.validate_action(batch(dict(close, grip_squeeze_rad=0.08)), deploy)
        with self.assertRaisesRegex(ValueError, r"invalid_grip_squeeze_rad: must be a number in \[0, 0.08\] rad"):
            contract.validate_action(batch(dict(close, grip_squeeze_rad=0.09)), deploy)
        with self.assertRaisesRegex(ValueError, "applies to jaw stages"):
            contract.validate_action(batch(dict(id="m", kind="move", arm="right", action=move, grip_squeeze_rad=0.01)), deploy)
        with self.assertRaisesRegex(ValueError, "not observe stages"):
            contract.validate_action(batch(dict(id="look", kind="observe", reason="r", grip_squeeze_rad=0.01), close), deploy)
        with self.assertRaisesRegex(ValueError, "unsupported_capability: grip_squeeze_rad"):
            contract.validate_action(batch(dict(close, grip_squeeze_rad=0.01)), deployment())
        # A changed squeeze is a changed plan: the execute digest covers the stage field.
        first = adapter._batch_digest(batch(dict(close, grip_squeeze_rad=0.01)))
        self.assertNotEqual(first, adapter._batch_digest(batch(dict(close, grip_squeeze_rad=0.02))))


class GuardInterplayTests(SqueezeCase):
    def test_guard_judges_the_reading_after_the_squeeze_settled(self):
        harness, adapter, driver = self.held()
        events = []
        original_send, original_measure = adapter._send, adapter._jaw_measurement

        def send(*args, **kwargs):
            if kwargs.get("kind") == "hold_squeeze":
                events.append("hold_squeeze")
            return original_send(*args, **kwargs)

        def measure(run, goal):
            events.append("measure")
            return original_measure(run, goal)
        with patch.object(adapter, "_send", side_effect=send), patch.object(adapter, "_jaw_measurement", side_effect=measure):
            preview, response = self.run_batch(harness, self.grasp(adapter, dict(grip_guard_preset="rigid_object")))
        self.assertEqual(events, ["measure", "measure", "hold_squeeze", "measure"])  # open, close, squeeze, re-measure
        receipt = response.data["receipt"]
        self.assertEqual(receipt["status"], "completed", receipt)
        close = receipt["stage_results"]["close"]
        self.assertFalse(close["grip_guard"]["triggered"])
        self.assertEqual(close["grip_guard"]["measured_aperture_m"], close["grip_facts"]["measured_aperture_m"])
        self.assertAlmostEqual(close["grip_guard"]["measured_aperture_m"], close["gripper_execution"]["measured_opening_m"],
                               delta=1e-4)
        # The preview says the re-check does not exist on this deployment, and why.
        recheck = preview.data["stage_plans"]["close"]["grip_guard"]["post_grasp_recheck"]
        self.assertEqual(recheck, {"available": False, "reason": contract.RECHECK_HOLD_SQUEEZE})
        capability = adapter.capabilities()["grip_guard"]["post_grasp_recheck"]
        self.assertEqual(capability, {"available": False, "reason": contract.RECHECK_HOLD_SQUEEZE})

    def test_guard_pauses_when_the_squeezed_reading_is_below_the_minimum(self):
        # A soft object: once the goal is the hold goal the jaw closes onto it (the fake object gives way).
        harness, adapter, driver = self.held()
        original = adapter._send

        def send(*args, **kwargs):
            if kwargs.get("kind") == "hold_squeeze":
                driver.jaw_min = None
            return original(*args, **kwargs)
        with patch.object(adapter, "_send", side_effect=send):
            _, response = self.run_batch(harness, self.grasp(adapter, dict(min_aperture_m=CUBE_M - 0.0005)))
        receipt = response.data["receipt"]
        self.assertEqual((receipt["status"], receipt["checkpoint_reason"]), ("stopped", "aperture_below_min_m"), receipt)
        close = receipt["stage_results"]["close"]
        squeeze = close["gripper_execution"]["hold_squeeze"]
        self.assertAlmostEqual(squeeze["contact_opening_m"], CUBE_M, delta=2e-4)  # at contact it was above the minimum
        self.assertLess(close["grip_guard"]["measured_aperture_m"], CUBE_M - 0.0005)
        self.assertAlmostEqual(close["gripper_execution"]["measured_jaw_rad"], squeeze["hold_goal_rad"], delta=1e-3)

    def test_empty_close_on_the_other_fingertip_is_squeezed_and_still_caught_by_the_guard(self):
        # Real jaw: an empty close stops on the opposite fingertip short of the closed limit (here 0.03 rad short).
        stop = 0.03 / JAW_RANGE
        harness, adapter, driver = self.harness(deploy=squeeze_deployment(), jaw_min_normalized=stop)
        _, response = self.run_batch(harness, self.grasp(adapter, dict(grip_guard_preset="rigid_object")))
        receipt = response.data["receipt"]
        self.assertEqual((receipt["status"], receipt["checkpoint_reason"]), ("stopped", "aperture_below_min_m"), receipt)
        self.assertTrue(receipt["stage_results"]["close"]["gripper_execution"]["hold_squeeze"]["applied"])
        self.assertAlmostEqual(driver.jaw_goal, -0.174533 + 0.03 - SQUEEZE, delta=1e-3)
        self.assertGreater(driver.jaw_goal, -0.174533)  # no longer the closed limit

    def test_slip_after_a_squeezed_close_is_reported_not_paused(self):
        harness, adapter, driver = self.held()
        stages = self.grasp(adapter, dict(grip_guard_preset="rigid_object")) + [self.move(adapter, "carry", dx=0.02, dz=0.03)]
        original = adapter._arrive

        def slip_during_lift(run, q_target, pose, tolerances, **options):
            if driver.jaw_min is not None:
                driver.jaw_min = None  # the object slips out while lifting
            return original(run, q_target, pose, tolerances, **options)
        with patch.object(adapter, "_arrive", side_effect=slip_during_lift):
            _, response = self.run_batch(harness, stages)
        receipt = response.data["receipt"]
        self.assertEqual(receipt["status"], "completed", receipt)
        self.assertEqual(adapter._plans[receipt["batch_plan_id"]]["armed_grip_guard"], {})
        lift = receipt["stage_results"]["lift"]
        # The jaw travelled the squeeze and stopped at its hold goal: nothing resists it any more.
        self.assertAlmostEqual(lift["grip_hold"]["squeeze_remaining_rad"], 0.0, delta=2e-3)
        self.assertAlmostEqual(lift["grip_facts"]["measured_jaw_after"], lift["grip_hold"]["hold_goal_normalized"], delta=2e-3)
        self.assertGreater(lift["grip_facts"]["measured_aperture_m"], EMPTY_CLOSE_M + 0.002)

    def test_held_object_reports_the_remaining_squeeze_on_later_moves(self):
        harness, adapter, _ = self.held()
        stages = self.grasp(adapter) + [self.move(adapter, "carry", dx=0.02, dz=0.03)]
        _, response = self.run_batch(harness, stages)
        results = response.data["receipt"]["stage_results"]
        for stage in ("lift", "carry"):
            self.assertAlmostEqual(results[stage]["grip_hold"]["squeeze_remaining_rad"], SQUEEZE, delta=1e-3)
        self.assertNotIn("grip_hold", results["close"])

    def test_unmeasured_zero_point_keeps_its_own_reason(self):
        deploy = deployment(**{"gripper.hold_squeeze_rad": SQUEEZE})
        self.assertEqual(contract.post_grasp_recheck_unavailable_reason(deploy), contract.RECHECK_NOT_MEASURED)
        self.assertIsNone(contract.post_grasp_recheck_unavailable_reason(
            deployment(**{"gripper.empty_close_aperture_m": EMPTY_CLOSE_M})))


class HeldGoalTests(SqueezeCase):
    def hold_goal(self, receipt, stage="close"):
        return receipt["stage_results"][stage]["gripper_execution"]["hold_squeeze"]["hold_goal_normalized"]

    def test_later_moves_resend_the_hold_goal(self):
        harness, adapter, driver = self.held()
        stages = self.grasp(adapter) + [self.move(adapter, "carry", dx=0.02, dz=0.03)]
        _, response = self.run_batch(harness, stages)
        receipt = response.data["receipt"]
        self.assertEqual(receipt["status"], "completed", receipt)
        hold = adapter.jaw_goal
        self.assertAlmostEqual(hold, self.hold_goal(receipt), delta=1e-4)
        jaws = [jaw for _, jaw in driver.goals]
        first = jaws.index(hold)
        self.assertGreater(len(jaws) - first, 20)  # the lift and the carry
        self.assertEqual(set(jaws[first:]), {hold})
        state = adapter.robot_state(adapter.read())["arms"]["right"]["gripper"]
        # The observed state keeps the commanded goal (what "closed" means to its readers); the servo holds the hold goal.
        self.assertEqual((state["goal_normalized"], state["command_normalized"]), (0.0, 0.0))
        self.assertGreater(hold, 0.0)
        self.assertTrue(adapter._in_hand(adapter.read()))

    def test_wait_keeps_the_hold_goal(self):
        harness, adapter, driver = self.held()
        _, response = self.run_batch(harness, self.grasp(adapter))
        hold = adapter.jaw_goal
        action = harness._action_from_args("robot_wait", dict(native_steps=5, observation_id=harness.latest["observation_id"],
                                                             action_id="wait-1"))
        self.assertEqual(harness.execute(action)["receipt"]["status"], "completed")
        self.assertEqual({jaw for _, jaw in driver.goals[-5:]}, {hold})
        self.assertEqual(adapter.jaw_hold["squeeze_rad"], SQUEEZE)

    def test_checkpoint_pause_and_resume_keep_the_hold_goal(self):
        harness, adapter, driver = self.held()
        preview, response = self.run_batch(harness, self.grasp(adapter, dict(checkpoint_after_gripper=True)))
        receipt = response.data["receipt"]
        self.assertEqual((receipt["status"], receipt["reason"]), ("stopped", "observation_checkpoint"), receipt)
        self.assertTrue(receipt["stop_confirmation"]["confirmed"])
        hold = self.hold_goal(receipt)
        self.assertAlmostEqual(adapter.jaw_goal, hold, delta=1e-4)
        self.assertAlmostEqual(driver.goals[-1][1], hold, delta=1e-4)  # the pause's hold command
        sent = len(driver.goals)
        resumed = self.execute(harness, preview.data["plan_id"], call="resume")
        self.assertEqual(resumed.data["receipt"]["status"], "completed", resumed.data["receipt"])
        self.assertEqual({jaw for _, jaw in driver.goals[sent:]}, {adapter.jaw_goal})
        self.assertAlmostEqual(adapter.jaw_goal, hold, delta=1e-4)
        self.assertAlmostEqual(resumed.data["receipt"]["stage_results"]["lift"]["grip_hold"]["squeeze_remaining_rad"],
                               SQUEEZE, delta=1e-3)

    def test_operator_hold_keeps_the_hold_goal_and_torque_on_ends_it(self):
        harness, adapter, driver = self.held()
        self.run_batch(harness, self.grasp(adapter))
        hold = adapter.jaw_goal
        self.assertLess(hold, CUBE_JAW)
        held = adapter.operator("hold")
        self.assertTrue(held["stop_confirmation"]["confirmed"], held)
        self.assertEqual(adapter.jaw_goal, hold)
        self.assertEqual(driver.goals[-1][1], hold)
        self.assertIsNotNone(adapter.jaw_hold)
        adapter.operator("relax")
        on = adapter.operator("torque_on")
        self.assertEqual(on["status"], "torque_on", on)
        # torque_on pins the jaw goal to the measured jaw: no squeeze is in effect any more.
        self.assertAlmostEqual(adapter.jaw_goal, CUBE_JAW, delta=1e-3)
        self.assertIsNone(adapter.jaw_hold)
        self.assertFalse(adapter.jaw_closing)

    def test_next_jaw_command_starts_from_the_hold_goal_and_ends_the_squeeze(self):
        harness, adapter, driver = self.held()
        self.run_batch(harness, self.grasp(adapter))
        hold, sent = adapter.jaw_goal, len(driver.goals)
        driver.jaw_min = None  # released
        preview, response = self.run_batch(harness, [dict(id="release", kind="gripper", arm="right", goal_normalized=1.0),
                                                     self.move(adapter, "up", dz=0.01)])
        receipt = response.data["receipt"]
        self.assertEqual(receipt["status"], "completed", receipt)
        ramp = [jaw for _, jaw in driver.goals[sent:]]
        step = adapter._jaw_step_normalized(1.0)
        self.assertLessEqual(ramp[0] - hold, step + 1e-9)  # no jump: the ramp starts at the goal the servo holds
        self.assertGreater(ramp[0], hold)
        self.assertEqual(preview.data["stage_costs"]["release"]["native_steps"],
                         receipt["stage_results"]["release"]["gripper_execution"]["planned_steps"])
        self.assertNotIn("hold_squeeze", receipt["stage_results"]["release"]["gripper_execution"])
        self.assertIsNone(adapter.jaw_hold)
        self.assertNotIn("grip_hold", receipt["stage_results"]["up"])
        self.assertEqual((adapter.jaw_goal, adapter.jaw_command), (1.0, 1.0))

    def test_checkpoint_rule_compares_with_the_commanded_goal_not_the_hold_goal(self):
        # After a squeezed close the commanded goal is 0: a batch-level checkpoint_after_gripper must not treat a new
        # goal-0 stage as a jaw change just because the servo holds a hold goal above 0.
        harness, adapter, _ = self.held()
        self.run_batch(harness, self.grasp(adapter))
        self.assertGreater(adapter.jaw_goal, 0.0)
        preview, _ = self.run_batch(harness, [dict(id="again", kind="gripper", arm="right", goal_normalized=0.0),
                                              self.move(adapter, "up", dz=0.01)], checkpoint_after_gripper=True)
        self.assertEqual(preview.data["checkpoints"], [])

    def test_single_gripper_action_uses_the_deployment_squeeze(self):
        harness, adapter, driver = self.held(gripper_normalized=0.5)
        action = contract.compile_action("gripper", dict(arm="right", goal_normalized=0.0,
                                                         observation_id=harness.latest["observation_id"]),
                                         harness.identity, harness.capabilities)
        receipt = harness.execute(action)["receipt"]
        self.assertEqual(receipt["status"], "completed", receipt)
        squeeze = receipt["gripper_execution"]["right"]["hold_squeeze"]
        self.assertEqual((squeeze["applied"], squeeze["squeeze_rad"]), (True, SQUEEZE))
        self.assertAlmostEqual(driver.jaw_goal, CUBE_RAD - SQUEEZE, delta=1e-3)

    def test_stop_during_the_hold_settle_fails_the_stage_with_the_hold_goal_kept(self):
        harness, adapter, driver = self.held()
        original = adapter._send

        def send(*args, **kwargs):
            sent = original(*args, **kwargs)
            if kwargs.get("kind") == "hold_squeeze":
                adapter.stop_generation += 1
                adapter.stop_reason = "operator_hold"
            return sent
        with patch.object(adapter, "_send", side_effect=send):
            _, response = self.run_batch(harness, self.grasp(adapter))
        receipt = response.data["receipt"]
        self.assertEqual((receipt["status"], receipt["reason"], receipt["failed_stage_id"]),
                         ("stopped", "operator_hold", "close"), receipt)
        self.assertTrue(receipt["stage_results"]["close"]["gripper_execution"]["hold_squeeze"]["applied"])
        self.assertAlmostEqual(driver.jaw_goal, CUBE_RAD - SQUEEZE, delta=1e-3)
        self.assertEqual(receipt["stage_results"]["lift"]["status"], "pending")


class DuringMotionRampTests(SqueezeCase):
    def test_ramp_inside_a_move_is_squeezed_after_arrival(self):
        harness, adapter, driver = self.held(gripper_normalized=0.5)
        stage = self.move(adapter, "close_moving", dx=0.02, gripper_goal_normalized=0.0,
                          gripper_motion=dict(start_fraction=0.2, end_fraction=0.8), grip_squeeze_rad=0.03)
        preview, response = self.run_batch(harness, [stage, self.move(adapter, "lift", dz=0.03)])
        self.assertEqual(preview.data["stage_plans"]["close_moving"]["hold_squeeze_rad"], 0.03)
        receipt = response.data["receipt"]
        self.assertEqual(receipt["status"], "completed", receipt)
        result = receipt["stage_results"]["close_moving"]
        jaw = result["gripper_execution"]
        self.assertEqual(jaw["phase"], "during_motion")
        self.assertEqual((jaw["hold_squeeze"]["applied"], jaw["hold_squeeze"]["squeeze_rad"]), (True, 0.03))
        self.assertAlmostEqual(driver.jaw_goal, CUBE_RAD - 0.03, delta=1e-3)
        # During the move the jaw followed its ramp to the commanded goal; the hold goal came after the arrival.
        jaws = [jaw_goal for _, jaw_goal in driver.goals]
        self.assertIn(0.0, jaws)
        self.assertGreater(jaws.index(adapter.jaw_goal), max(i for i, value in enumerate(jaws) if value == 0.0))
        self.assertNotIn("grip_hold", result)  # the stage commanded the jaw itself
        self.assertIn("grip_hold", receipt["stage_results"]["lift"])

    def test_opening_ramp_inside_a_move_is_unchanged(self):
        harness, adapter, driver = self.harness(deploy=squeeze_deployment())
        stage = self.move(adapter, "open_moving", dx=0.02, gripper_goal_normalized=1.0,
                          gripper_motion=dict(start_fraction=0.2, end_fraction=0.8))
        _, response = self.run_batch(harness, [stage])
        jaw = response.data["receipt"]["stage_results"]["open_moving"]["gripper_execution"]
        self.assertNotIn("hold_squeeze", jaw)
        self.assertEqual(adapter.jaw_goal, 1.0)


class DriverTelemetryTests(unittest.TestCase):
    def test_sts_current_and_load_decoding(self):
        self.assertEqual(sts_current_ma(0), 0.0)
        self.assertEqual(sts_current_ma(16), 104.0)
        self.assertEqual(sts_current_ma(0x8000 | 16), -104.0)
        self.assertEqual(sts_load_pct(0), 0.0)
        self.assertEqual(sts_load_pct(202), 20.2)
        self.assertEqual(sts_load_pct(0x0400 | 202), -20.2)

    def test_lerobot_max_relative_target_keeps_arm_cap_without_clamping_gripper_range(self):
        # Regression: SO101Follower(use_degrees=True) normalizes ARM_JOINTS in degrees, while normalizing "gripper" in
        # RANGE_0_100 (0..100). Passing a scalar float clamped "gripper" to +-10 % (~0.19 rad), preventing full close.
        deploy = deployment()
        deploy["max_relative_target_deg"] = 10.0
        target = lerobot_max_relative_target(deploy)
        self.assertEqual({name: target[name] for name in ARM_JOINTS}, {name: 10.0 for name in ARM_JOINTS})
        self.assertEqual(target["gripper"], 100.0)
        deploy["max_relative_target_deg"] = None
        self.assertIsNone(lerobot_max_relative_target(deploy))

    def test_motor_status_reads_11_byte_block_and_rejects_stale_2_byte_position_frames(self):
        # Regression for real-arm false 48..120 degC temperature aborts: scservo_sdk.readRx slices rxpacket[5:5+length]
        # without checking PKT_LENGTH, so a delayed 2-byte Present_Position reply (raw tick 0x0B78 -> low byte 0x78=120)
        # was decoded as 120 degC on a 1-byte Present_Temperature read, and slices to 3 bytes on an 11-byte read.
        self.assertIsNone(decode_sts_status_block([0x78, 0x0B, 0x6A]))  # 3-byte slice of a 2-byte status packet
        load_word = 0x0400 | 202          # -20.2 %
        current_word = 0x8000 | 19        # -123.5 mA
        valid_frame = [
            load_word & 0xFF, (load_word >> 8) & 0xFF,  # 60..61 Present_Load
            121,                                        # 62 Present_Voltage -> 12.1 V
            37,                                         # 63 Present_Temperature -> 37.0 degC
            0, 0, 0, 0, 0,                              # 64..68 intermediate registers
            current_word & 0xFF, (current_word >> 8) & 0xFF,  # 69..70 Present_Current
        ]
        self.assertEqual(len(valid_frame), STS_STATUS_LENGTH)

        class FakePort:
            def __init__(self):
                self.cleared = 0

            def clearPort(self):
                self.cleared += 1

        class FakeReader:
            def __init__(self, frames):
                self.frames = list(frames)
                self.port = FakePort()
                self.data_dict = {}
                self.calls = 0

            def txRxPacket(self):
                self.calls += 1
                self.data_dict = self.frames.pop(0)
                return 0

        class FakeMotor:
            def __init__(self, motor_id):
                self.id = motor_id

        class FakeBus:
            def __init__(self, reader):
                self.motors = {"shoulder_pan": FakeMotor(1), "gripper": FakeMotor(6)}
                self.sync_reader = reader
                self.setups = []

            def _setup_sync_reader(self, ids, addr, length):
                self.setups.append((list(ids), addr, length))

            def _is_comm_success(self, comm):
                return comm == 0

        deploy = deployment()
        deploy["num_read_retries"] = 2
        drv = LeRobotSO101Driver(deploy)
        # Attempt 1 returns a stale 3-byte slice on gripper (raw pos 0x0B78 -> 0x78=120); attempt 2 returns 11 bytes.
        reader = FakeReader([
            {1: valid_frame, 6: [0x78, 0x0B, 0x6A]},
            {1: valid_frame, 6: valid_frame},
        ])
        drv.robot = type("R", (), {"bus": FakeBus(reader)})()
        rows = drv.motor_status()
        self.assertEqual(drv.robot.bus.setups[0], ([1, 6], STS_STATUS_ADDR, STS_STATUS_LENGTH))
        self.assertEqual(reader.calls, 2)
        self.assertEqual(reader.port.cleared, 1)
        self.assertEqual(rows["gripper"], dict(temperature_c=37.0, load=-20.2, voltage_v=12.1, current_ma=-123.5))

        # When every retry still carries a truncated frame on gripper, report None (unknown), never 120 degC.
        bad_reader = FakeReader([{1: valid_frame, 6: [0x78, 0x0B, 0x6A]}] * 3)
        drv.robot = type("R", (), {"bus": FakeBus(bad_reader)})()
        bad_rows = drv.motor_status()
        self.assertEqual(bad_rows["gripper"], dict(temperature_c=None, load=None, voltage_v=None, current_ma=None))
        self.assertEqual(bad_rows["shoulder_pan"]["temperature_c"], 37.0)

    def test_telemetry_motor_rows_carry_current_when_the_driver_reads_it(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        adapter, driver, _ = make_adapter(directory.name)
        self.addCleanup(adapter.close)
        state = adapter.telemetry_tick(frames=False)
        self.assertEqual(state["motor"]["gripper"], dict(temperature_c=30.0, load=0.0, voltage_v=12.0, current_ma=None))
        status = deepcopy(driver.motor_status())
        status["gripper"]["current_ma"] = 130.0
        driver.motor_status = lambda: status
        adapter._last_motor_read = None
        self.assertEqual(adapter.telemetry_tick(frames=False)["motor"]["gripper"]["current_ma"], 130.0)


class ZeroDeltaJawCommandTests(SqueezeCase):
    def test_jaw_command_equal_to_current_hold_goal_clears_squeeze_and_updates_commanded_goal(self):
        # Regression: when a jaw stage explicitly commands goal_normalized == adapter.jaw_goal (the held goal of a
        # previous squeezed close), _send must still record the new jaw_command, set jaw_closing=False and clear the
        # previous jaw_hold so later moves do not report a stale grip_hold.
        harness, adapter, _ = self.held()
        _, first = self.run_batch(harness, self.grasp(adapter))
        held_goal = first.data["receipt"]["stage_results"]["close"]["gripper_execution"]["hold_squeeze"]["hold_goal_normalized"]
        self.assertEqual(adapter.jaw_command, 0.0)
        self.assertIsNotNone(adapter.jaw_hold)

        _, second = self.run_batch(harness, [
            dict(id="match_hold", kind="gripper", arm="right", goal_normalized=adapter.jaw_goal),
            self.move(adapter, "carry", dx=0.01),
        ])
        receipt = second.data["receipt"]
        self.assertEqual(receipt["status"], "completed", receipt)
        self.assertAlmostEqual(adapter.jaw_command, held_goal, places=4)
        self.assertFalse(adapter.jaw_closing)
        self.assertIsNone(adapter.jaw_hold)
        self.assertNotIn("hold_squeeze", receipt["stage_results"]["match_hold"]["gripper_execution"])
        self.assertNotIn("grip_hold", receipt["stage_results"]["carry"])


if __name__ == "__main__":
    unittest.main()
