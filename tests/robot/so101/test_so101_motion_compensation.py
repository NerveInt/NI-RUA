"""SO-101 in-transit sag work on CPU (fake servo models over a rigid table, fake cameras, real kinematics).

- M1 ``motion_compensation.carry_arrival_offset``: the last free-space arrival's offset rides along the next move;
  zero on contact / compliant / correction-disabled moves and after a contact hold; guards judge without it.
- M2 ``transit``: the measured TCP against the commanded path while a move streams (always reported).
- M3 ``motion_compensation.gravity_feedforward``: URDF gravity torque over identified servo stiffness on every goal.
- M5 ``motion_compensation.no_response_stop`` / ``arrival_breakaway_rad``: a correction round the joints do not answer.
- Deployment validation of every new key; everything off = the receipts and commands of before.

These tests prove the software contract on servo models, not the behaviour of the real arm.
"""

NI_RUA_TEST_TIER = "fast"

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np

from ni_rua.core.config import Config
from ni_rua.robot.backends.so101 import contract
from ni_rua.robot.backends.so101.deployment import (MOTION_COMPENSATION_FIELDS, active_motion_compensation,
                                                     public_limits, validate_deployment)
from ni_rua.robot.backends.so101.driver import ARM_JOINTS, ENCODER_RESOLUTION_RAD, FakeCameraSource, FakeClock
from ni_rua.robot.backends.so101.motion_compensation import contact_stage, fit_stiffness, transit_deviation
from ni_rua.robot.backends.so101.worker import SO101Adapter
from ni_rua.robot.embodiments.so101 import gravity as so101_gravity
from ni_rua.robot.model import so101_kinematics
from ni_rua.robot.model.robot_model import urdf_manifest
from robot.so101.test_so101_backend import EXAMPLE
from robot.so101.test_so101_motion_features import EMPTY_CLOSE_M, Case, TableArmDriver, batch

SAG = (0.0, -0.03, -0.02, 0.0, 0.0)  # goal - measured (rad): the loaded shoulder and elbow rest past their goals
STIFFNESS = {"shoulder_lift": 10.0, "elbow_flex": 8.0, "wrist_flex": 6.0}  # N*m/rad of the modelled servos


class ServoModelDriver(TableArmDriver):
    """``TableArmDriver`` with two optional servo effects.

    ``stiffness``: ``{joint: N*m/rad}``; such a joint is a pure-P loop under gravity and is drawn toward
    ``goal + gravity_torque(q) / stiffness`` (rad) instead of ``goal``. ``static_band_rad``: a joint at rest starts
    moving only when that target is at least this far away (rad; gear static friction), then runs to it; one value
    for every joint or five values."""

    def __init__(self, deployment, clock, table_z_m=None, *, stiffness=None, static_band_rad=0.0, **options):
        super().__init__(deployment, clock, table_z_m, **options)
        self.inertials = so101_gravity.load_inertials()
        self.stiffness = [None if stiffness is None else stiffness.get(name) for name in ARM_JOINTS]
        self.static_band_rad, self.moving = static_band_rad, [False] * 5

    def _advance(self):
        now = self.clock.now()
        dt, self.last = now - self.last, now
        if not self.torque or dt <= 0:
            return
        before, alpha = list(self.q), 1 - math.exp(-dt / self.tau_s)
        torque = so101_gravity.gravity_torques(self.model, self.inertials, self.q)
        for index in range(5):
            target = self.goal[index] - self.sag[index]
            if self.stiffness[index] is not None:
                target += torque[index] / self.stiffness[index]
            delta = target - self.q[index]
            band = self.static_band_rad[index] if isinstance(self.static_band_rad, (list, tuple)) else self.static_band_rad
            if not self.moving[index] and abs(delta) < band:
                continue
            self.q[index] += delta * alpha
            self.moving[index] = abs(delta) > 1e-5
        jaw = self.jaw_goal if self.jaw_min is None else max(self.jaw_goal, self.jaw_min)
        self.jaw += (jaw - self.jaw) * alpha
        if self.table_z_m is not None and so101_kinematics.fk(self.model, self.q)[2, 3] < self.table_z_m:
            self.q = before


def document(compensation=None, **motion):
    raw = json.loads(EXAMPLE.read_text())
    raw["gripper"]["empty_close_aperture_m"] = EMPTY_CLOSE_M
    raw["motion"].update(motion)
    if compensation is not None:
        raw["motion_compensation"] = compensation
        if isinstance(compensation, dict) and isinstance(compensation.get("gravity_feedforward"), dict):
            raw["max_relative_target_deg"] = 30.0  # room for arrival offset + feedforward + one command step
    return raw


class CompensationCase(Case):
    def adapter(self, compensation=None, *, motion=None, table_z_m=None, **driver_options):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        deploy = validate_deployment(document(compensation, **(motion or {})), base=EXAMPLE.parent)
        for name in ("fixed", "wrist"):
            deploy["cameras"][name].update(width=64, height=48)
        clock = FakeClock()
        driver = ServoModelDriver(deploy, clock, table_z_m, **driver_options)
        cameras = {name: FakeCameraSource(name, clock=clock, width=64, height=48) for name in ("fixed", "wrist")}
        adapter = SO101Adapter(deploy, directory.name, "rgb_proprio_v1", driver=driver, cameras=cameras, clock=clock,
                               image_settings=Config().images.to_dict(), start_telemetry=False)
        self.addCleanup(adapter.close)
        # Let the loaded arm take its resting pose under the pinned start goal before anything is planned.
        for _ in range(40):
            clock.sleep(0.05)
            adapter.read()
        return adapter, driver

    def one(self, adapter, stage):
        """Preview and execute one stage as its own batch; returns its stage result."""
        _, receipt = self.run_batch(adapter, batch(stage))
        self.assertEqual(receipt["status"], "completed", receipt)
        return receipt["stage_results"][stage["id"]]

    @staticmethod
    def streamed(driver, receipt):
        """The arm goals the servos received for the ``motion`` rows of one receipt: the planned row plus, with motion
        compensation, the offset sent on top of it (``planned_trace.goal_offset_rad``)."""
        trace = receipt["planned_trace"]
        offsets = trace.get("goal_offset_rad") or [[0.0] * 5] * len(trace["q_rad"])
        return [np.add(row[:5], offset).tolist() for row, offset, kind in zip(trace["q_rad"], offsets, trace["row_kinds"])
                if kind == "motion"]

    @staticmethod
    def planned(receipt):
        """The planned (offset-free) arm rows of the ``motion`` commands of one receipt."""
        trace = receipt["planned_trace"]
        return [row[:5] for row, kind in zip(trace["q_rad"], trace["row_kinds"]) if kind == "motion"]


# ---- M1 -------------------------------------------------------------------------------------------------------------
class CarryTests(CompensationCase):
    CARRY = dict(carry_arrival_offset=True)

    def test_the_arrival_offset_rides_along_the_next_move(self):
        results = {}
        for label, compensation in (("off", None), ("carry", self.CARRY)):
            adapter, driver = self.adapter(compensation, sag_rad=SAG)
            first = self.one(adapter, self.move(adapter, "a", dx=0.03))
            start = np.asarray(adapter.read()["q_rad"])
            stage = self.move(adapter, "b", dx=0.03)
            _, receipt = self.run_batch(adapter, batch(stage))
            self.assertEqual(receipt["status"], "completed", receipt)
            results[label] = (first, receipt["stage_results"]["b"], np.asarray(self.streamed(driver, receipt)[0]) - start)
        first, second, first_goal = results["off"]
        self.assertNotIn("path_compensation", first)
        self.assertNotIn("path_compensation", second)
        self.assertNotIn("start_goal_offset_rad", second["arrival_correction"])
        # Without the carry the first goal of the move is next to the measured (sagged) pose: the offset is gone.
        self.assertLess(float(np.abs(first_goal).max()), 0.02)
        first, second, first_goal = results["carry"]
        self.assertEqual(first["path_compensation"], dict(carried_offset_rad=[0.0] * 5, carry="unavailable"))
        offset = first["arrival_correction"]["final_goal_offset_rad"]
        np.testing.assert_allclose(offset, SAG, atol=2e-3)
        self.assertEqual(second["path_compensation"], dict(carried_offset_rad=offset, carry="last_arrival"))
        self.assertEqual(second["arrival_correction"]["start_goal_offset_rad"], offset)
        # The first goal of the move keeps the offset on top of one interpolation step.
        np.testing.assert_allclose(first_goal[1:3], SAG[1:3], atol=0.016)
        # The arm stays on the commanded path instead of dropping back into its sag at the start of the move.
        self.assertGreater(results["off"][1]["transit"]["max_deviation_m"], 0.003)
        self.assertLess(second["transit"]["max_deviation_m"], 0.25 * results["off"][1]["transit"]["max_deviation_m"])
        # The arrival then has nothing left to correct.
        self.assertEqual(second["arrival_correction"]["rounds"], [])
        self.assertLess(second["arrival_correction"]["initial_tcp_residual"]["position_error_m"], 0.001)

    def test_carried_offset_is_bounded_by_the_moves_own_limit(self):
        adapter, _ = self.adapter(self.CARRY, sag_rad=SAG)
        self.one(adapter, self.move(adapter, "a", dx=0.03))
        result = self.one(adapter, self.move(adapter, "b", dx=0.03, tolerance=0.02, arrival_correction_max_offset_rad=0.01))
        self.assertEqual(result["path_compensation"]["carried_offset_rad"], [0.0, -0.01, -0.01, 0.0, 0.0])
        self.assertEqual(result["arrival_correction"]["max_offset_rad"], 0.01)

    def test_no_carry_into_or_out_of_contact_compliant_and_uncorrected_moves(self):
        cases = (("contact_stage", dict(stall_guard_preset="contact_stop", max_tracking_divergence_m=0.05)),
                 ("contact_stage", dict(max_upward_dz_m=0.02)),
                 ("contact_stage", dict(tolerance_preset="compliant")),
                 ("correction_disabled", dict(arrival_correction_rounds=0)))
        for source, path in cases:
            with self.subTest(path=path):
                adapter, _ = self.adapter(self.CARRY, sag_rad=SAG)
                self.one(adapter, self.move(adapter, "a", dx=0.03))
                result = self.one(adapter, self.move(adapter, "b", dx=0.02, tolerance=0.03, **path))
                self.assertEqual(result["path_compensation"], dict(carried_offset_rad=[0.0] * 5, carry=source))
                self.assertEqual(result["arrival_correction"]["start_goal_offset_rad"], [0.0] * 5)
                # Whatever that move's own arrival ended with is not carried on: it may be a contact offset.
                after = self.one(adapter, self.move(adapter, "c", dx=0.02))
                self.assertEqual(after["path_compensation"], dict(carried_offset_rad=[0.0] * 5, carry="unavailable"))

    def test_no_carry_after_a_contact_hold(self):
        adapter, driver = self.adapter(self.CARRY, sag_rad=SAG, gripper_normalized=1.0)
        self.one(adapter, self.move(adapter, "a", dx=0.02))
        driver.table_z_m = self.pose(adapter)["position_m"][2] - 0.03
        down = self.one(adapter, self.move(adapter, "down", dz=-0.06, tolerance=0.01, stall_guard_preset="contact_stop",
                                           max_tracking_divergence_m=0.02, max_upward_dz_m=0.04))
        self.assertTrue(down["contact_stopped_early"])
        self.assertEqual(down["path_compensation"]["carry"], "contact_stage")
        lift = self.one(adapter, self.move(adapter, "lift", dz=0.02))
        self.assertEqual(lift["path_compensation"], dict(carried_offset_rad=[0.0] * 5, carry="unavailable"))

    def test_a_pause_between_stages_keeps_the_carry_and_a_jaw_stage_does_not_end_it(self):
        adapter, _ = self.adapter(self.CARRY, sag_rad=SAG, gripper_normalized=1.0)
        action = batch(self.move(adapter, "a", dx=0.03), self.jaw(adapter, "close", 0.5),
                       dict(id="look", kind="observe", reason="check"), self.move(adapter, "b", dx=0.05))
        preview = adapter.preview_batch(action)
        self.assertEqual(preview["status"], "planned", preview)
        request = dict(action, batch_plan_id=preview["plan_id"])
        paused = adapter.execute(request, lambda update: None, threading.Event())
        self.assertEqual((paused["status"], paused["reason"]), ("stopped", "observation_checkpoint"), paused)
        resumed = adapter.execute(request, lambda update: None, threading.Event())
        self.assertEqual(resumed["status"], "completed", resumed)
        result = resumed["stage_results"]["b"]
        self.assertEqual(result["path_compensation"]["carry"], "last_arrival")
        np.testing.assert_allclose(result["path_compensation"]["carried_offset_rad"], SAG, atol=2e-3)

    def test_a_foreign_goal_write_ends_the_carry(self):
        adapter, _ = self.adapter(self.CARRY, sag_rad=SAG)
        self.one(adapter, self.move(adapter, "a", dx=0.03))
        adapter.operator("torque_on")  # pins the goal to the measured pose, like any operator motion would change it
        adapter.observe()
        self.assertEqual(self.one(adapter, self.move(adapter, "b", dx=0.02))["path_compensation"]["carry"], "unavailable")

    def test_guards_judge_the_commanded_path_without_the_offset(self):
        sag = (0.0, -0.09, 0.0, 0.0, 0.0)
        adapter, _ = self.adapter(self.CARRY, sag_rad=sag)
        self.one(adapter, self.move(adapter, "a", dx=0.02))
        # Tighter than the carried offset: a guard comparing the SENT goal (path + 0.09 rad) with the measured joints
        # would stop every command of the next move.
        adapter.deployment["motion"]["tracking_error_limit_rad"] = 0.06
        stage = self.move(adapter, "b", dx=0.03, stall_guard_preset="strict", max_tracking_divergence_m=0.01)
        stage["action"]["speed_scale"] = 0.25
        result = self.one(adapter, stage)
        self.assertEqual(result["path_compensation"]["carry"], "last_arrival")
        self.assertAlmostEqual(result["path_compensation"]["carried_offset_rad"][1], -0.09, delta=2e-3)
        self.assertNotIn("tracking_stop", result)
        self.assertLess(adapter.last_tracking_error, 0.06)

    def test_effective_limits_name_what_is_switched_on(self):
        adapter, _ = self.adapter(self.CARRY)
        _, receipt = self.run_batch(adapter, batch(self.move(adapter, "a", dx=0.02)))
        self.assertEqual(receipt["effective_limits"]["motion_compensation"], dict(carry_arrival_offset=True))
        self.assertEqual(adapter.capabilities()["runtime_limits"]["motion_compensation"], dict(carry_arrival_offset=True))
        adapter, _ = self.adapter()
        _, receipt = self.run_batch(adapter, batch(self.move(adapter, "a", dx=0.02)))
        self.assertNotIn("motion_compensation", receipt["effective_limits"])


class SafetyReviewTests(CompensationCase):
    """Regressions of the independent safety review of the compensation mechanisms."""
    CARRY = dict(carry_arrival_offset=True)
    FEEDFORWARD = dict(gravity_feedforward=dict(joint_stiffness_nm_per_rad=STIFFNESS, max_offset_rad=0.1,
                                                provenance="unit test servo model"))

    def test_checkpoint_confirms_the_last_hold_goal_when_the_load_changes(self):
        adapter, driver = self.adapter(sag_rad=SAG)
        self.one(adapter, self.move(adapter, "approach", dx=0.03))
        measured = np.asarray(adapter.read()["q_rad"])
        offset = np.asarray(adapter.hold_offset)
        self.assertGreater(np.abs(offset).max(), 0.02)
        driver.sag = [0.0] * 5  # the load that produced the measured offset has gone
        action = batch(dict(id="look", kind="observe", reason="check"), self.jaw(adapter, "close", 0.0))
        _, receipt = self.run_batch(adapter, action)
        self.assertEqual((receipt["status"], receipt["reason"]), ("stopped", "observation_checkpoint"))
        self.assertTrue(receipt["stop_confirmation"]["confirmed"])
        terminal = receipt["actual_trace"][-1]["state"]["arms"]["right"]["q_rad"][:5]
        np.testing.assert_allclose(driver.goal, measured + offset, atol=1e-6)
        adapter.clock.sleep(1.0)
        np.testing.assert_allclose(adapter.read()["q_rad"], terminal, atol=0.001)

    def test_checkpoint_cannot_confirm_a_hold_after_operator_relax(self):
        adapter, driver = self.adapter()
        action = batch(dict(id="look", kind="observe", reason="check"), self.jaw(adapter, "close", 0.0))
        preview = adapter.preview_batch(action)
        sends = driver.sends
        receipt = adapter.execute(dict(action, batch_plan_id=preview["plan_id"]),
                                  lambda update: adapter.operator("relax"), threading.Event())
        self.assertFalse(receipt["stop_confirmation"]["confirmed"])
        self.assertEqual(receipt["stop_confirmation"]["state"], "torque_off")
        self.assertEqual(receipt["batch_status"], "failed")
        self.assertEqual(driver.sends, sends)

    def test_a_failed_arrival_against_a_support_leaves_no_carried_offset(self):
        """A free-space descent that ends pressed on a table: the correction pushes its offset to the bound, the
        stage fails, and the recovery lift must not carry that offset toward the obstacle."""
        adapter, driver = self.adapter(self.CARRY, sag_rad=SAG, gripper_normalized=1.0)
        self.one(adapter, self.move(adapter, "a", dx=0.02))
        self.assertIsNotNone(adapter.path_offset_goal)
        driver.table_z_m = self.pose(adapter)["position_m"][2] - 0.015
        _, receipt = self.run_batch(adapter, batch(self.move(adapter, "down", dz=-0.022, tolerance=0.003)))
        self.assertEqual((receipt["status"], receipt["reason"]), ("failed", "tracking_error"), receipt)
        down = receipt["stage_results"]["down"]
        self.assertNotIn("tracking_stop", down)  # no guard fired: the arrival verdict failed it
        self.assertEqual(down["path_compensation"]["carry"], "last_arrival")
        self.assertGreater(float(np.abs(down["arrival_correction"]["final_goal_offset_rad"]).max()), 0.04)  # contact, not sag
        self.assertTrue(receipt["stop_confirmation"]["confirmed"])
        self.assertIsNone(adapter.path_offset_goal)
        lift = self.one(adapter, self.move(adapter, "lift", dz=0.02))
        self.assertEqual(lift["path_compensation"], dict(carried_offset_rad=[0.0] * 5, carry="unavailable"))

    def test_an_arrival_that_used_its_whole_offset_bound_is_not_carried(self):
        adapter, _ = self.adapter(self.CARRY, sag_rad=SAG)
        first = self.one(adapter, self.move(adapter, "a", dx=0.03, tolerance=0.02, arrival_correction_max_offset_rad=0.01))
        self.assertTrue(first["arrival_correction"]["offset_limit_reached"])
        second = self.one(adapter, self.move(adapter, "b", dx=0.03))
        self.assertEqual(second["path_compensation"]["carry"], "unavailable")

    def test_hold_offset_is_the_residual_and_the_hold_adds_the_feedforward_of_its_own_pose(self):
        softer = {name: value * 0.8 for name, value in STIFFNESS.items()}
        adapter, driver = self.adapter(self.FEEDFORWARD, stiffness=softer)
        result = self.one(adapter, self.move(adapter, "a", dx=0.04))
        feedforward = np.asarray(result["path_compensation"]["gravity_feedforward_rad"])
        # Stored: the residual after the feedforward, inside the arrival bound (not bound + feedforward bound).
        limit = adapter.deployment["motion"]["arrival_correction_max_offset_rad"]
        self.assertLessEqual(float(np.abs(adapter.hold_offset).max()), limit)
        self.assertLess(float(np.abs(adapter.hold_offset).max()), 0.5 * float(np.abs(feedforward).max()))
        residual = np.asarray(adapter.hold_offset)
        # Someone pushes the arm to another pose; the hold there uses that pose's feedforward plus the residual.
        driver.q = np.add(driver.q, [0.0, 0.3, -0.3, 0.0, 0.0]).tolist()
        start = np.asarray(adapter.read()["q_rad"])
        stop = adapter.operator("hold")["stop_confirmation"]
        self.assertTrue(stop["confirmed"])
        expected = adapter._feedforward(start) + residual
        np.testing.assert_allclose(np.subtract(adapter.arm_goal, start), expected, atol=1e-6)
        self.assertGreater(float(np.abs(adapter._feedforward(start) - feedforward).max()), 0.01)  # the pose mattered
        np.testing.assert_allclose(stop["hold_offset_rad"], expected, atol=1e-5)
        # torque_on pins the goal to the present pose: nothing learned before describes the arm any more.
        adapter.operator("torque_on")
        self.assertEqual((adapter.hold_offset, adapter.hold_feedforward, adapter.path_offset_goal), ([0.0] * 5, False, None))
        adapter.hold_offset, adapter.hold_feedforward = [0.01] * 5, True
        adapter.operator("relax")
        self.assertEqual((adapter.hold_offset, adapter.hold_feedforward), ([0.0] * 5, False))

    def test_a_contact_hold_is_not_lifted_by_feedforward_at_a_later_pause(self):
        adapter, driver = self.adapter(self.FEEDFORWARD, stiffness=STIFFNESS, gripper_normalized=1.0)
        driver.table_z_m = self.pose(adapter)["position_m"][2] - 0.03
        action = batch(self.move(adapter, "down", dz=-0.06, tolerance=0.01, stall_guard_preset="contact_stop",
                                 max_tracking_divergence_m=0.02, max_upward_dz_m=0.04),
                       dict(id="look", kind="observe", reason="check"), self.jaw(adapter, "close", 0.0))
        preview = adapter.preview_batch(action)
        receipt = adapter.execute(dict(action, batch_plan_id=preview["plan_id"]), lambda update: None, threading.Event())
        self.assertEqual((receipt["status"], receipt["reason"]), ("stopped", "observation_checkpoint"), receipt)
        self.assertFalse(adapter.hold_feedforward)
        self.assertLess(float(np.abs(receipt["stop_confirmation"]["hold_offset_rad"]).max()), 0.005)

    def test_planned_trace_is_the_offset_free_plan_and_the_offsets_are_reported_beside_it(self):
        adapter, driver = self.adapter(dict(self.FEEDFORWARD, carry_arrival_offset=True),
                                       stiffness={name: value * 0.8 for name, value in STIFFNESS.items()})
        self.one(adapter, self.move(adapter, "a", dx=0.03))
        sends = len(driver.goals)
        start = adapter.read()["q_rad"]
        preview, receipt = self.run_batch(adapter, batch(self.move(adapter, "b", dx=0.03)))
        trace, target = receipt["planned_trace"], preview["stage_plans"]["b"]["q_rad"]
        rows = self.planned(receipt)
        from ni_rua.robot.backends.so101.ik_plan import joint_interpolation
        np.testing.assert_allclose(rows, joint_interpolation(np.asarray(start), target, len(rows)), atol=1e-9)
        self.assertEqual(rows[-1], [float(v) for v in target])
        self.assertEqual(len(trace["goal_offset_rad"]), len(trace["q_rad"]))
        self.assertIn("goal_offset_rad", trace["semantics"])
        # row + offset is what the servos received.
        sent = [goal for goal, _ in driver.goals[sends:sends + len(rows)]]
        np.testing.assert_allclose(np.add(rows, trace["goal_offset_rad"][:len(rows)]), sent, atol=1e-6)
        self.assertGreater(float(np.abs(trace["goal_offset_rad"][len(rows) - 1]).max()), 0.02)
        # With every mechanism off the trace is the one of before: sent goals, no offset column.
        adapter, driver = self.adapter(sag_rad=SAG)
        sends = len(driver.goals)
        _, receipt = self.run_batch(adapter, batch(self.move(adapter, "a", dx=0.03)))
        self.assertNotIn("goal_offset_rad", receipt["planned_trace"])
        self.assertNotIn("goal_offset_rad", receipt["planned_trace"]["semantics"])
        np.testing.assert_allclose([row[:5] for row in receipt["planned_trace"]["q_rad"]],
                                   [goal for goal, _ in driver.goals[sends:]], atol=0)

    def test_offsets_slew_in_under_the_published_command_step(self):
        """After a contact hold the goal is the measured pose; the next move's feedforward (about 0.06 rad on the
        elbow) may not arrive in one command."""
        adapter, driver = self.adapter(self.FEEDFORWARD, stiffness=STIFFNESS, gripper_normalized=1.0)
        driver.table_z_m = self.pose(adapter)["position_m"][2] - 0.03
        self.one(adapter, self.move(adapter, "down", dz=-0.06, tolerance=0.01, stall_guard_preset="contact_stop",
                                    max_tracking_divergence_m=0.02, max_upward_dz_m=0.04))
        held_goal, sends = np.asarray(adapter.arm_goal), len(driver.goals)
        stage = self.move(adapter, "lift", dz=0.03)
        _, receipt = self.run_batch(adapter, batch(stage))
        self.assertEqual(receipt["status"], "completed", receipt)
        count = len(self.planned(receipt))
        sent = np.array([held_goal] + [goal for goal, _ in driver.goals[sends:sends + count]])
        limit = adapter.deployment["motion"]["max_command_delta_rad"]
        self.assertLessEqual(float(np.abs(np.diff(sent, axis=0)).max()), limit + 1e-12)
        # The full feedforward is in force by the end of the move.
        target = receipt["planned_trace"]["ik_targets"][0]["q_rad"]
        np.testing.assert_allclose(sent[-1], np.add(target, adapter._feedforward(target)), atol=1e-9)
        self.assertGreater(float(np.abs(adapter._feedforward(target)).max()), limit)  # one step could not hold it

    def test_no_response_takes_back_only_the_joints_that_did_not_answer(self):
        # shoulder_lift: 0.007 rad short inside a 0.03 rad static band (does not answer); elbow_flex: 0.02 rad short,
        # free (answers).
        adapter, driver = self.adapter(dict(no_response_stop=True), sag_rad=(0.0, -0.007, -0.02, 0.0, 0.0))
        stage = self.move(adapter, "a", dx=0.03, tolerance=0.001)
        driver.static_band_rad = [0.0, 0.03, 0.0, 0.0, 0.0]
        _, receipt = self.run_batch(adapter, batch(stage))
        correction = receipt["stage_results"]["a"]["arrival_correction"]
        first = correction["rounds"][0]
        self.assertIs(first["responded"], True)
        self.assertEqual(first["unanswered_joints"], ["shoulder_lift"])
        self.assertAlmostEqual(first["goal_offset_rad"][1], -0.007, delta=1e-3)   # tried
        self.assertEqual(correction["final_goal_offset_rad"][1], 0.0)             # taken back
        self.assertAlmostEqual(correction["final_goal_offset_rad"][2], -0.02, delta=2e-3)  # the answered joint keeps it
        self.assertIn("arrival_correction_revert", receipt["planned_trace"]["row_kinds"])
        self.assertAlmostEqual(adapter.hold_offset[1], -0.007, delta=1e-3)

    def test_feedforward_must_fit_inside_the_driver_clamp(self):
        raw = document(self.FEEDFORWARD)
        raw["max_relative_target_deg"] = 10.0   # 0.1745 rad < 0.1 + 0.1 + 0.035
        with self.assertRaisesRegex(ValueError, "gravity_feedforward.*exceeds max_relative_target_deg"):
            validate_deployment(raw, base=EXAMPLE.parent)
        raw["max_relative_target_deg"] = 13.5   # 0.2356 rad >= 0.235
        validate_deployment(raw, base=EXAMPLE.parent)
        raw = document(dict(carry_arrival_offset=True))
        raw["max_relative_target_deg"] = 10.0   # deployments without feedforward keep loading as before
        validate_deployment(raw, base=EXAMPLE.parent)
        text = contract.actor_motion_instructions(validate_deployment(document(self.FEEDFORWARD), base=EXAMPLE.parent))
        self.assertIn("a stopped press pushes harder", text)


# ---- M2 -------------------------------------------------------------------------------------------------------------
class TransitTests(CompensationCase):
    @staticmethod
    def tcp(q):
        pose = np.eye(4)
        pose[:3, 3] = q[:3]
        return pose

    def test_deviation_is_measured_from_the_commanded_path_not_from_the_lagging_goal(self):
        # A straight 0.1 m path along x in ten commands; the measured point lags one command behind and runs lower.
        rows = [[0.01 * (index + 1), 0.0, 0.0, 0.0, 0.0] for index in range(10)]
        drop = [0.0135, 0.012, 0.010, 0.008, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001]
        measured = [[0.01 * index, 0.0, -drop[index], 0.0, 0.0] for index in range(10)]
        facts = transit_deviation(self.tcp, [0.0] * 5, rows, measured, planned_steps=10)
        self.assertEqual(facts, dict(samples=10, max_below_path_m=0.0135, max_below_at_fraction=0.1,
                                     max_deviation_m=0.0135, max_deviation_at_fraction=0.1))

    def test_sideways_and_upward_deviation_is_not_reported_as_below(self):
        rows = [[0.01 * (index + 1), 0.0, 0.0, 0.0, 0.0] for index in range(4)]
        measured = [[0.01, 0.0, 0.002, 0.0, 0.0], [0.02, 0.004, 0.0, 0.0, 0.0], [0.03, 0.0, -0.001, 0.0, 0.0],
                    [0.04, 0.0, 0.0, 0.0, 0.0]]
        facts = transit_deviation(self.tcp, [0.0] * 5, rows, measured, planned_steps=8)
        self.assertEqual((facts["max_below_path_m"], facts["max_below_at_fraction"]), (0.001, 0.375))
        self.assertEqual((facts["max_deviation_m"], facts["max_deviation_at_fraction"]), (0.004, 0.25))
        self.assertIsNone(transit_deviation(self.tcp, [0.0] * 5, [], [], planned_steps=8))

    def test_every_executed_move_reports_transit_and_duration(self):
        adapter, _ = self.adapter(sag_rad=SAG)
        start = adapter.read()["q_rad"]
        result = self.one(adapter, self.move(adapter, "a", dx=0.04))
        transit = result["transit"]
        self.assertEqual(set(transit), {"samples", "max_below_path_m", "max_below_at_fraction", "max_deviation_m",
                                        "max_deviation_at_fraction"})
        # The modelled sag puts the TCP below the commanded path by about its FK displacement.
        expected = float(so101_kinematics.fk(adapter.model, start)[2, 3]
                         - so101_kinematics.fk(adapter.model, np.subtract(start, SAG))[2, 3])
        self.assertGreater(expected, 0.003)
        self.assertAlmostEqual(transit["max_below_path_m"], expected, delta=0.002)
        self.assertGreaterEqual(transit["max_deviation_m"], transit["max_below_path_m"])
        self.assertGreater(result["duration_s"], 0.0)

    def test_a_stopped_move_still_reports_the_transit_it_had(self):
        adapter, driver = self.adapter(gripper_normalized=1.0)
        driver.table_z_m = self.pose(adapter)["position_m"][2] - 0.02
        _, receipt = self.run_batch(adapter, batch(self.move(adapter, "down", dz=-0.06, tolerance=0.01,
                                                             stall_guard_preset="strict", max_tracking_divergence_m=0.02)))
        self.assertEqual((receipt["status"], receipt["reason"]), ("failed", "tracking_error"), receipt)
        down = receipt["stage_results"]["down"]
        self.assertEqual(down["transit"]["samples"], down["tracking_stop"]["completed_steps"])
        self.assertLess(down["transit"]["max_below_path_m"], 0.001)  # held up by the table, never below the path

    def test_identified_servo_values_never_reach_the_model(self):
        """Stiffness, breakaway bands and provenance are tuning data: only switches and the feedforward bound are
        model-visible (capabilities, receipts, motion text)."""
        stiffness = dict(shoulder_lift=9.4321, elbow_flex=7.6543, wrist_flex=5.8765)
        section = dict(carry_arrival_offset=True, no_response_stop=True, stop_correction_within_tolerance=True,
                       arrival_breakaway_rad=dict(shoulder_lift=0.0234, elbow_flex=0.0187),
                       gravity_feedforward=dict(joint_stiffness_nm_per_rad=stiffness, max_offset_rad=0.15,
                                                provenance="identify-run-XYZ p16"))
        adapter, _ = self.adapter(section, stiffness=stiffness)
        preview, receipt = self.run_batch(adapter, batch(self.move(adapter, "a", dx=0.03)))
        self.assertEqual(receipt["status"], "completed", receipt)
        expected = dict(carry_arrival_offset=True, no_response_stop=True, arrival_breakaway=True,
                        stop_correction_within_tolerance=True, gravity_feedforward=dict(max_offset_rad=0.15))
        self.assertEqual(receipt["effective_limits"]["motion_compensation"], expected)
        self.assertEqual(adapter.capabilities()["runtime_limits"]["motion_compensation"], expected)
        evidence = ("planned_trace", "actual_trace", "dispatched_trace", "before_state", "before_observation",
                    "terminal_observation")
        visible = json.dumps([adapter.capabilities(), preview["effective_limits"], preview["stage_plans"],
                              {key: value for key, value in receipt.items() if key not in evidence}], default=str)
        visible += contract.actor_motion_instructions(adapter.deployment)
        for secret in ("9.4321", "7.6543", "5.8765", "0.0234", "0.0187", "identify-run-XYZ", "stiffness_nm_per_rad",
                       "provenance", "arrival_breakaway_rad"):
            self.assertNotIn(secret, visible)
        self.assertIn("at most 0.15 rad per joint", visible)
        # The worker still uses the full deployment values.
        self.assertEqual(adapter.deployment["motion_compensation"]["gravity_feedforward"]["joint_stiffness_nm_per_rad"], stiffness)

    def test_model_text_says_the_preview_path_is_the_commanded_one(self):
        deploy = validate_deployment(document(), base=EXAMPLE.parent)
        text = contract.actor_motion_instructions(deploy)
        self.assertIn("transit: max_below_path_m", text)
        self.assertIn("The preview describes the COMMANDED path", text)
        self.assertNotIn("Motion compensation on this deployment", text)
        deploy = validate_deployment(document(dict(carry_arrival_offset=True, no_response_stop=True)), base=EXAMPLE.parent)
        text = contract.actor_motion_instructions(deploy)
        self.assertIn("path_compensation.carried_offset_rad", text)
        self.assertIn("termination_reason no_response", text)
        self.assertNotIn("gravity_feedforward_rad", text)


# ---- M3 -------------------------------------------------------------------------------------------------------------
SYNTHETIC_URDF = """<robot name="synthetic">
  <link name="base_link"/>
  {links}
  <link name="gripper_frame_link"/>
  <joint name="gripper_frame_joint" type="fixed"><parent link="l4"/><child link="gripper_frame_link"/></joint>
</robot>"""


def synthetic_urdf(directory, mass_kg, radius_m):
    """Five revolute joints at the base origin: shoulder_pan about z, the others about y. Only the link below
    elbow_flex has mass: ``mass_kg`` at ``radius_m`` along its x axis."""
    links, parent = [], "base_link"
    for index, name in enumerate(ARM_JOINTS):
        child = f"l{index}"
        inertial = (f'<inertial><origin xyz="{radius_m} 0 0"/><mass value="{mass_kg}"/></inertial>'
                    if name == "elbow_flex" else "")
        links.append(f'<link name="{child}">{inertial}</link>'
                     f'<joint name="{name}" type="revolute"><parent link="{parent}"/><child link="{child}"/>'
                     f'<axis xyz="{"0 0 1" if index == 0 else "0 1 0"}"/><limit lower="-3" upper="3"/></joint>')
        parent = child
    path = Path(directory) / "synthetic.urdf"
    path.write_text(SYNTHETIC_URDF.format(links="\n  ".join(links)))
    return path


class GravityTests(CompensationCase):
    FEEDFORWARD = dict(gravity_feedforward=dict(joint_stiffness_nm_per_rad=STIFFNESS, max_offset_rad=0.2,
                                                provenance="unit test servo model"))

    def test_gravity_torque_of_a_point_mass_matches_the_hand_computation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = synthetic_urdf(directory, mass_kg=0.2, radius_m=0.1)
            model, inertials = urdf_manifest(path), so101_gravity.load_inertials(path)
        self.assertEqual(inertials, {"l2": dict(mass_kg=0.2, com_m=[0.1, 0.0, 0.0])})
        weight_arm = 0.2 * so101_gravity.GRAVITY_M_S2 * 0.1  # N*m with the mass level with the joint: 0.196133
        self.assertAlmostEqual(weight_arm, 0.196133, places=6)
        # Level: gravity turns +x toward -z, the positive direction about +y, for both joints above the mass.
        np.testing.assert_allclose(so101_gravity.gravity_torques(model, inertials, [0, 0, 0, 0, 0]),
                                   [0, weight_arm, weight_arm, 0, 0], atol=1e-12)
        # Shoulder 20 deg + elbow 40 deg below level: the lever arm is r cos(60 deg), pan does not change it.
        q = [0.7, math.radians(20), math.radians(40), 0.3, -0.2]
        np.testing.assert_allclose(so101_gravity.gravity_torques(model, inertials, q),
                                   [0, weight_arm * 0.5, weight_arm * 0.5, 0, 0], atol=1e-12)
        # Hanging straight down: no torque.
        np.testing.assert_allclose(so101_gravity.gravity_torques(model, inertials, [0, math.pi / 2, 0, 0, 0]),
                                   np.zeros(5), atol=1e-12)

    def test_packaged_arm_torques_are_the_gradient_of_its_potential_energy(self):
        model, inertials = so101_kinematics.load_model(), so101_gravity.load_inertials()
        self.assertAlmostEqual(sum(item["mass_kg"] for item in inertials.values()), 0.667006, places=5)

        def energy(q):
            poses = so101_kinematics.fk_all(model, q)
            return sum(item["mass_kg"] * so101_gravity.GRAVITY_M_S2 * (poses[link] @ np.append(item["com_m"], 1.0))[2]
                       for link, item in inertials.items())
        for q in ([0.0, -1.0, 1.0, 0.6, 0.0], [0.5, 0.3, -0.2, 0.8, 0.4]):
            numeric = [-(energy(np.add(q, step)) - energy(np.subtract(q, step))) / 2e-6 for step in np.eye(5) * 1e-6]
            np.testing.assert_allclose(so101_gravity.gravity_torques(model, inertials, q), numeric, atol=1e-6)
        # A level base: the pan joint carries no gravity torque; the stretched arm loads the shoulder most.
        stretched = so101_gravity.gravity_torques(model, inertials, [0.0] * 5)
        self.assertAlmostEqual(stretched[0], 0.0, places=5)
        self.assertGreater(stretched[1], stretched[2])
        self.assertGreater(stretched[2], stretched[3])

    def test_feedforward_offset_formula_and_bound(self):
        offsets = so101_gravity.feedforward_offset_rad([0.1, -0.4, 0.8, 0.3, 0.0], [None, 10.0, 8.0, 1.0, 5.0], 0.2)
        np.testing.assert_allclose(offsets, [0.0, 0.04, -0.1, -0.2, 0.0])

    def test_feedforward_is_added_to_every_goal_and_cancels_the_modelled_sag(self):
        results = {}
        for label, compensation in (("off", None), ("feedforward", self.FEEDFORWARD)):
            adapter, driver = self.adapter(compensation, stiffness=STIFFNESS)
            stage = self.move(adapter, "a", dx=0.04)
            preview, receipt = self.run_batch(adapter, batch(stage))
            self.assertEqual(receipt["status"], "completed", receipt)
            results[label] = (adapter, preview, receipt, self.streamed(driver, receipt), driver)
        adapter, preview, receipt, sent, driver = results["off"]
        self.assertIsNone(adapter._feedforward([0.0] * 5))
        result = receipt["stage_results"]["a"]
        self.assertNotIn("path_compensation", result)
        self.assertGreater(result["arrival_correction"]["initial_tcp_residual"]["position_error_m"], 0.004)
        self.assertGreater(result["transit"]["max_below_path_m"], 0.004)
        adapter, preview, receipt, sent, driver = results["feedforward"]
        result = receipt["stage_results"]["a"]
        target = preview["stage_plans"]["a"]["q_rad"]
        expected = so101_gravity.feedforward_offset_rad(
            so101_gravity.gravity_torques(adapter.model, so101_gravity.load_inertials(), target),
            [STIFFNESS.get(name) for name in ARM_JOINTS], 0.2)
        np.testing.assert_allclose(result["path_compensation"]["gravity_feedforward_rad"], expected, atol=1e-5)
        self.assertNotIn("carry", result["path_compensation"])
        # The last streamed goal is the IK target plus the feedforward of that row; joints without stiffness get none.
        np.testing.assert_allclose(np.subtract(sent[-1], target), expected, atol=1e-6)
        self.assertEqual((sent[-1][0], sent[-1][4]), (target[0], target[4]))
        np.testing.assert_allclose(driver.goals[-1][0], np.add(target, adapter._feedforward(target)), atol=1e-12)
        self.assertLess(expected[1], -0.01)  # gravity pulls the shoulder toward larger angles: the goal leads back
        # The modelled arm now follows the commanded path and arrives without correction.
        self.assertLess(result["transit"]["max_below_path_m"], 0.001)
        self.assertLess(result["arrival_correction"]["initial_tcp_residual"]["position_error_m"], 0.001)
        self.assertEqual(result["arrival_correction"]["rounds"], [])
        self.assertEqual(result["arrival_correction"]["final_goal_offset_rad"], [0.0] * 5)
        self.assertEqual(receipt["effective_limits"]["motion_compensation"], dict(gravity_feedforward=dict(max_offset_rad=0.2)))

    def test_carry_holds_only_the_residual_the_feedforward_leaves(self):
        # The servos are 25 % softer than the deployment says: the feedforward under-compensates.
        softer = {name: value * 0.8 for name, value in STIFFNESS.items()}
        adapter, _ = self.adapter(dict(self.FEEDFORWARD, carry_arrival_offset=True), stiffness=softer)
        first = self.one(adapter, self.move(adapter, "a", dx=0.03))
        feedforward = np.asarray(first["path_compensation"]["gravity_feedforward_rad"])
        offset = np.asarray(first["arrival_correction"]["final_goal_offset_rad"])
        # Residual = a quarter of the feedforward, far inside the arrival-correction bound.
        np.testing.assert_allclose(offset[1:3], 0.25 * feedforward[1:3], atol=3e-3)
        second = self.one(adapter, self.move(adapter, "b", dx=0.03))
        self.assertEqual(second["path_compensation"]["carry"], "last_arrival")
        self.assertEqual(second["path_compensation"]["carried_offset_rad"], offset.tolist())
        self.assertLess(second["transit"]["max_below_path_m"], 0.002)
        # The hold offset is the residual alone (bounded by the arrival bound); the feedforward is added per pose.
        self.assertLess(float(np.abs(adapter.hold_offset).max()), 0.5 * float(np.abs(feedforward).max()))
        self.assertTrue(adapter.hold_feedforward)

    def test_stiffness_fit_recovers_the_modelled_servos(self):
        model, inertials = so101_kinematics.load_model(), so101_gravity.load_inertials()
        rng = np.random.default_rng(7)
        poses = [[0.0, -1.0 + 0.25 * a, 1.0 - 0.2 * b, 0.6 + 0.15 * c, 0.0] for a in range(3) for b in range(3) for c in range(2)]
        torques = np.array([so101_gravity.gravity_torques(model, inertials, q) for q in poses])
        stiffness = np.array([50.0, 10.0, 8.0, 6.0, 4.0])
        errors = -torques / stiffness + 0.002 + rng.normal(0.0, 2e-4, torques.shape)
        fit = fit_stiffness(errors, torques, min_torque_range_nm=0.05)
        for name, expected in zip(ARM_JOINTS[1:4], stiffness[1:4]):
            self.assertAlmostEqual(fit[name]["stiffness_nm_per_rad"], expected, delta=0.06 * expected)
            self.assertAlmostEqual(fit[name]["intercept_rad"], 0.002, delta=1e-3)
            self.assertGreater(fit[name]["r_squared"], 0.98)
        for name in ("shoulder_pan", "wrist_roll"):
            self.assertIsNone(fit[name]["stiffness_nm_per_rad"])
            self.assertEqual(fit[name]["reason"], "torque_range_below_minimum")
        self.assertEqual(fit_stiffness(errors[:2], torques[:2], 0.05)["elbow_flex"]["reason"], "too_few_samples")
        self.assertEqual(fit_stiffness(-errors, torques, 0.05)["elbow_flex"]["reason"], "non_positive_compliance")


# ---- M5 -------------------------------------------------------------------------------------------------------------
class NoResponseTests(CompensationCase):
    """A settled error of 0.007 rad on a joint whose gear needs 0.03 rad of goal change to move from rest."""
    SAG = (0.0, -0.007, 0.0, 0.0, 0.0)

    def run_move(self, compensation, band_rad=0.03, tolerance=0.005):
        adapter, driver = self.adapter(compensation, sag_rad=self.SAG)
        stage = self.move(adapter, "a", dx=0.03, tolerance=tolerance)
        driver.static_band_rad = band_rad
        preview, receipt = self.run_batch(adapter, batch(stage))
        if tolerance == 0.005:
            self.assertEqual(receipt["status"], "completed", receipt)
        return adapter, driver, preview["stage_plans"]["a"]["q_rad"], receipt

    def test_without_the_switch_every_round_accumulates_and_the_hold_keeps_it(self):
        adapter, driver, target, receipt = self.run_move(None)
        correction = receipt["stage_results"]["a"]["arrival_correction"]
        self.assertEqual((len(correction["rounds"]), correction["termination_reason"]), (3, "round_limit"))
        self.assertNotIn("responded", correction["rounds"][0])
        self.assertNotIn("final_goal_offset_rad", correction)
        self.assertAlmostEqual(correction["rounds"][-1]["goal_offset_rad"][1], -0.021, delta=1e-3)
        self.assertAlmostEqual(adapter.hold_offset[1], -0.028, delta=1e-3)  # three unanswered steps stored for holds

    def test_an_unanswered_round_stops_the_correction_and_is_taken_back(self):
        adapter, driver, target, receipt = self.run_move(dict(no_response_stop=True))
        correction = receipt["stage_results"]["a"]["arrival_correction"]
        self.assertEqual(correction["termination_reason"], "no_response")
        (entry,) = correction["rounds"]
        self.assertIs(entry["responded"], False)
        self.assertAlmostEqual(entry["goal_offset_rad"][1], -0.007, delta=5e-4)  # what was tried
        self.assertEqual(correction["final_goal_offset_rad"], [0.0] * 5)  # what is in force
        self.assertFalse(correction["offset_limit_reached"])
        # The goal is back at the planned target and the hold offset is the settled shortfall alone.
        np.testing.assert_allclose(driver.goal, target, atol=1e-12)
        self.assertEqual(receipt["planned_trace"]["row_kinds"][-2:], ["arrival_correction", "arrival_correction_revert"])
        self.assertAlmostEqual(adapter.hold_offset[1], -0.007, delta=5e-4)
        self.assertEqual(receipt["effective_limits"]["motion_compensation"], dict(no_response_stop=True))

    def test_a_round_the_joints_answer_continues_as_before(self):
        adapter, _ = self.adapter(dict(no_response_stop=True), sag_rad=SAG)
        correction = self.one(adapter, self.move(adapter, "a", dx=0.03))["arrival_correction"]
        self.assertEqual(correction["termination_reason"], "joint_error_below_correction_threshold")
        self.assertEqual([entry["responded"] for entry in correction["rounds"]], [True])
        np.testing.assert_allclose(correction["final_goal_offset_rad"], SAG, atol=2e-3)

    def test_one_encoder_tick_of_motion_is_not_a_response(self):
        self.assertAlmostEqual(ENCODER_RESOLUTION_RAD, 2 * math.pi / 4096)
        adapter, driver, _, _ = self.run_move(dict(no_response_stop=True))
        original, state = driver.read, {"nudged": False}

        def one_tick_later():
            sample = original()
            return dict(sample, q_rad=[value - (ENCODER_RESOLUTION_RAD if state["nudged"] and index == 1 else 0.0)
                                       for index, value in enumerate(sample["q_rad"])])
        driver.read = one_tick_later
        original_send = adapter._send

        def send(q_rad, jaw, generation=None, run=None, kind=None, **options):
            if kind == "arrival_correction":
                state["nudged"] = True  # the reading flickers by one tick after the correction command
            return original_send(q_rad, jaw, generation, run=run, kind=kind, **options)
        adapter._send = send
        stage = self.move(adapter, "b", dx=0.02)
        correction = self.one(adapter, stage)["arrival_correction"]
        self.assertEqual(correction["termination_reason"], "no_response")
        self.assertIs(correction["rounds"][0]["responded"], False)

    def test_no_carried_offset_is_built_from_an_unanswered_round(self):
        adapter, _, _, receipt = self.run_move(dict(no_response_stop=True, carry_arrival_offset=True))
        self.assertEqual(receipt["stage_results"]["a"]["arrival_correction"]["final_goal_offset_rad"], [0.0] * 5)
        stage = self.move(adapter, "b", dx=0.02)
        result = self.one(adapter, stage)
        self.assertEqual(result["path_compensation"], dict(carried_offset_rad=[0.0] * 5, carry="last_arrival"))

    def test_breakaway_band_enlarges_a_step_that_would_stay_inside_static_friction(self):
        # The identified band (0.03 rad) is just above what this gear needs (0.029 rad).
        # The settled error (about 1.6 mm at the TCP) is outside this move's 1 mm tolerance, so a round is sent.
        adapter, driver, target, receipt = self.run_move(
            dict(arrival_breakaway_rad=dict(shoulder_lift=0.03), stop_correction_within_tolerance=True), band_rad=0.029,
            tolerance=0.001)
        correction = receipt["stage_results"]["a"]["arrival_correction"]
        first = correction["rounds"][0]
        self.assertEqual(first["breakaway_joints"], ["shoulder_lift"])
        self.assertAlmostEqual(first["goal_offset_rad"][1], -0.03, places=5)
        self.assertEqual(first["goal_offset_rad"][2], 0.0)  # a joint without an identified band keeps its own rule
        # The joint moved this time: its settled error changed by more than the step it was short of.
        self.assertGreater(abs(first["settled_joint_error_rad"][1] - (-0.007)), 0.01)
        self.assertIn("final_goal_offset_rad", correction)
        self.assertEqual(receipt["effective_limits"]["motion_compensation"],
                         dict(arrival_breakaway=True, stop_correction_within_tolerance=True))
        # The step is the band itself, never more: the goal passes the measured error by band - error.
        self.assertLessEqual(abs(first["goal_offset_rad"][1]), 0.03)


# ---- M6 -------------------------------------------------------------------------------------------------------------
class WithinToleranceTests(CompensationCase):
    """The live case (real arm 2026-11-16, stage blue_mid): tolerance_preset normal with 0.015 m / 0.15 rad; the IK
    target met the orientation bound by a hair (position-priority solve); the arm first settled well inside both
    tolerances; the correction then drove the joints to the IK target and the orientation ended just outside."""
    ON = dict(stop_correction_within_tolerance=True)
    PATH = dict(position_tolerance_m=0.015, orientation_tolerance_rad=0.15, tolerance_preset="normal")
    # goal - measured (rad): the shoulder and the wrist roll rest short of the target (both corrected); the wrist flex
    # error is below arrival_correction_min_error_rad and is never corrected.
    SAG = (0.0, -0.045, 0.0, 0.004, 0.045)

    def live_move(self, compensation, **path):
        from ni_rua.robot.backends.so101.worker import matrix_to_quaternion
        adapter, _ = self.adapter(compensation, sag_rad=self.SAG)
        wanted = np.array([0.0, -0.9, 0.9, 0.8, 0.3])
        pose = adapter.tcp(wanted)
        pose = dict(position_m=pose[:3, 3].tolist(), quaternion_xyzw=list(matrix_to_quaternion(pose[:3, :3])))
        # The planned joints: the wanted pose rolled 0.14998 rad about the approach axis, inside the 0.15 rad bound.
        target = (wanted + [0.0, 0.0, 0.0, 0.0, 0.14998]).tolist()
        planned = adapter.tcp(target)
        self.assertLess(float(np.linalg.norm(planned[:3, 3] - np.asarray(pose["position_m"]))), 0.002)
        run = adapter._new_run(threading.Event(), 60.0, "live-case")
        ok, reason, fields = adapter._move(run, target, pose, dict(self.PATH, **path))
        return adapter, ok, reason, fields

    def test_without_the_switch_correction_turns_an_in_tolerance_arrival_into_a_failure(self):
        _, ok, reason, fields = self.live_move(None)
        first = fields["arrival_correction"]["initial_tcp_residual"]
        self.assertLess(first["position_error_m"], 0.015)
        self.assertLess(first["orientation_error_rad"], 0.15)
        self.assertEqual((ok, reason), (False, "tracking_error"))
        self.assertEqual(len(fields["arrival_correction"]["rounds"]), 1)
        self.assertLess(fields["position_error_m"], first["position_error_m"])   # the position did improve
        self.assertGreater(fields["orientation_error_rad"], 0.15)                # and the orientation left the bound
        self.assertNotIn("final_goal_offset_rad", fields["arrival_correction"])

    def test_with_the_switch_the_in_tolerance_arrival_is_kept_and_judged(self):
        adapter, ok, reason, fields = self.live_move(self.ON)
        self.assertEqual((ok, reason), (True, None))
        correction = fields["arrival_correction"]
        self.assertEqual((correction["termination_reason"], correction["rounds"]), ("within_tolerance", []))
        self.assertEqual(correction["final_goal_offset_rad"], [0.0] * 5)
        # The verdict judges the very pose that was found inside the tolerance: nothing moved in between.
        self.assertEqual(fields["position_error_m"], correction["initial_tcp_residual"]["position_error_m"])
        self.assertEqual(fields["orientation_error_rad"], correction["initial_tcp_residual"]["orientation_error_rad"])
        self.assertTrue(fields["within_tolerance"])
        # The trade-off, reported: the arm stays where it settled, several mm from the target but inside 0.015 m.
        self.assertGreater(fields["position_error_m"], 0.005)
        self.assertEqual(adapter.capabilities()["runtime_limits"]["motion_compensation"],
                         dict(stop_correction_within_tolerance=True))
        self.assertIn("termination_reason within_tolerance", contract.actor_motion_instructions(adapter.deployment))

    def test_an_arrival_outside_the_tolerance_is_still_corrected_until_it_is_inside(self):
        _, ok, reason, fields = self.live_move(self.ON, position_tolerance_m=0.005, orientation_tolerance_rad=0.2)
        self.assertEqual((ok, reason), (True, None))
        correction = fields["arrival_correction"]
        self.assertGreater(correction["initial_tcp_residual"]["position_error_m"], 0.005)
        self.assertEqual(len(correction["rounds"]), 1)
        self.assertLess(fields["position_error_m"], 0.005)
        self.assertIn(correction["termination_reason"], ("within_tolerance", "joint_error_below_correction_threshold"))

    def test_the_vertical_window_counts_as_inside_so_a_resting_arm_is_not_pushed_harder(self):
        results = {}
        for label, compensation in (("off", None), ("on", self.ON)):
            adapter, driver = self.adapter(compensation, gripper_normalized=1.0)
            driver.table_z_m = self.pose(adapter)["position_m"][2] - 0.01  # the support is 10 mm above the target
            stage = self.move(adapter, "down", dz=-0.02, tolerance=0.005, max_upward_dz_m=0.02)
            _, receipt = self.run_batch(adapter, batch(stage))
            self.assertEqual(receipt["status"], "completed", receipt)
            results[label] = receipt["stage_results"]["down"]["arrival_correction"]
        self.assertGreater(len(results["off"]["rounds"]), 0)  # goals shifted toward the target under the table
        self.assertEqual((results["on"]["rounds"], results["on"]["termination_reason"]), ([], "within_tolerance"))

    def test_operator_joint_moves_keep_their_full_correction(self):
        adapter, _ = self.adapter(self.ON, sag_rad=SAG)
        target = np.add(adapter.read()["q_rad"], [0.1, 0.0, 0.0, 0.0, 0.0]).tolist()
        result = adapter.move_joints(target, None, 0.02)
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(len(result["arrival_correction"]["rounds"]), 1)
        self.assertLess(result["max_joint_error_rad"], 0.005)

    def test_preview_shows_the_ik_residual_next_to_the_tolerance_it_must_meet(self):
        """The margin is readable without a new field: the planned residual and its bound sit side by side."""
        adapter, _ = self.adapter()
        preview = adapter.preview_batch(batch(self.move(adapter, "a", dx=0.03, tolerance=0.015)))
        plan = preview["stage_plans"]["a"]
        for residual, bound in (("position_error_m", "position_tolerance_m"), ("orientation_error_rad", "orientation_tolerance_rad")):
            self.assertLessEqual(plan["ik"][residual], plan["tolerances"][bound])
        self.assertNotIn("margin", json.dumps(plan))


# ---- deployment ------------------------------------------------------------------------------------------------------
class DeploymentTests(unittest.TestCase):
    OFF = dict(carry_arrival_offset=False, no_response_stop=False, stop_correction_within_tolerance=False,
               arrival_breakaway_rad=None, gravity_feedforward=None)
    FEEDFORWARD = dict(joint_stiffness_nm_per_rad=dict(shoulder_lift=9.5), max_offset_rad=0.1, provenance="bench 2026-11-17")

    def load(self, compensation="absent"):
        return validate_deployment(document(None if compensation == "absent" else compensation), base=EXAMPLE.parent)

    def test_absent_null_and_empty_sections_switch_everything_off(self):
        raw = json.loads(EXAMPLE.read_text())
        self.assertNotIn("motion_compensation", raw)  # the template keeps the behaviour of before
        for compensation in ("absent", {}):
            deploy = self.load(compensation)
            self.assertEqual(deploy["motion_compensation"], self.OFF)
            self.assertIsNone(active_motion_compensation(deploy))
            self.assertNotIn("motion_compensation", public_limits(deploy))
        raw["motion_compensation"] = None
        self.assertEqual(validate_deployment(raw, base=EXAMPLE.parent)["motion_compensation"], self.OFF)
        legacy = self.load()
        del legacy["motion_compensation"]  # a deployment dict built before the section existed
        self.assertIsNone(active_motion_compensation(legacy))

    def test_every_documented_key_is_accepted_and_normalized(self):
        section = dict(carry_arrival_offset=True, no_response_stop=True, stop_correction_within_tolerance=True,
                       arrival_breakaway_rad=dict(elbow_flex=0.02, shoulder_lift=0.015),
                       gravity_feedforward=dict(self.FEEDFORWARD, _note="comments are ignored"), _comment="x")
        self.assertEqual(set(section) - {"_comment"}, set(MOTION_COMPENSATION_FIELDS))
        deploy = self.load(section)
        self.assertEqual(deploy["motion_compensation"], dict(
            carry_arrival_offset=True, no_response_stop=True, stop_correction_within_tolerance=True,
            arrival_breakaway_rad=dict(shoulder_lift=0.015, elbow_flex=0.02), gravity_feedforward=self.FEEDFORWARD))
        self.assertEqual(public_limits(deploy)["motion_compensation"], dict(
            carry_arrival_offset=True, no_response_stop=True, stop_correction_within_tolerance=True, arrival_breakaway=True,
            gravity_feedforward=dict(max_offset_rad=0.1)))
        for text in MOTION_COMPENSATION_FIELDS.values():
            self.assertIn("default", text)

    def test_invalid_values_are_rejected_with_the_field_name(self):
        feedforward = self.FEEDFORWARD
        cases = (
            ([], "motion_compensation: object or null"),
            (dict(carry=True), "motion_compensation: unknown carry"),
            (dict(carry_arrival_offset=1), "motion_compensation.carry_arrival_offset: bool"),
            (dict(no_response_stop="yes"), "motion_compensation.no_response_stop: bool"),
            (dict(stop_correction_within_tolerance=0), "motion_compensation.stop_correction_within_tolerance: bool"),
            (dict(arrival_breakaway_rad=dict(elbow_flex=0.02)), "arrival_breakaway_rad requires stop_correction_within_tolerance"),
            (dict(arrival_breakaway_rad={}), "arrival_breakaway_rad: object with arm joint names"),
            (dict(arrival_breakaway_rad=dict(gripper=0.01)), "arrival_breakaway_rad: object with arm joint names"),
            (dict(arrival_breakaway_rad=dict(elbow_flex=0)), "arrival_breakaway_rad.elbow_flex: 0 not in (0, 0.1] rad"),
            (dict(arrival_breakaway_rad=dict(elbow_flex=0.2)), "arrival_breakaway_rad.elbow_flex"),
            (dict(gravity_feedforward=True), "gravity_feedforward: object with exactly"),
            (dict(gravity_feedforward={k: v for k, v in feedforward.items() if k != "provenance"}), "gravity_feedforward: object with exactly"),
            (dict(gravity_feedforward=dict(feedforward, extra=1)), "gravity_feedforward: object with exactly"),
            (dict(gravity_feedforward=dict(feedforward, provenance=" ")), "gravity_feedforward.provenance"),
            (dict(gravity_feedforward=dict(feedforward, max_offset_rad=0)), "gravity_feedforward.max_offset_rad"),
            (dict(gravity_feedforward=dict(feedforward, max_offset_rad=0.5)), "gravity_feedforward.max_offset_rad"),
            (dict(gravity_feedforward=dict(feedforward, joint_stiffness_nm_per_rad={})), "joint_stiffness_nm_per_rad: object"),
            (dict(gravity_feedforward=dict(feedforward, joint_stiffness_nm_per_rad=dict(elbow_flex=0))),
             "joint_stiffness_nm_per_rad.elbow_flex: 0 not in (0, 1000] N*m/rad"),
            (dict(gravity_feedforward=dict(feedforward, joint_stiffness_nm_per_rad=dict(elbow_flex=None))),
             "joint_stiffness_nm_per_rad.elbow_flex"),
            (dict(gravity_feedforward=dict(feedforward, joint_stiffness_nm_per_rad=dict(jaw=3.0))), "joint_stiffness_nm_per_rad: object"),
        )
        for compensation, message in cases:
            with self.subTest(compensation=compensation):
                with self.assertRaises(ValueError) as raised:
                    self.load(compensation)
                self.assertIn("so101_deployment_", str(raised.exception))
                self.assertIn(message, str(raised.exception))

    def test_servo_gains_are_validated_and_the_integral_gain_documents_its_contact_risk(self):
        from ni_rua.robot.backends.so101.deployment import FIELDS
        text = FIELDS["servo"]["i_coefficient"]["doc"]
        for phrase in ("LeRobot default 0", "integral windup", "contact_stop", "so101_motion_bench"):
            self.assertIn(phrase, text)
        raw = document()
        self.assertEqual(raw["servo"]["i_coefficient"], 0)  # the template default is unchanged
        raw["servo"]["i_coefficient"] = 4
        self.assertEqual(validate_deployment(raw, base=EXAMPLE.parent)["servo"]["i_coefficient"], 4)
        for bad in (-1, 255, 1.5, True):
            raw["servo"]["i_coefficient"] = bad
            with self.subTest(i_coefficient=bad), self.assertRaisesRegex(ValueError, "servo.i_coefficient"):
                validate_deployment(raw, base=EXAMPLE.parent)

    def test_contact_stage_rule(self):
        self.assertFalse(contact_stage({}))
        self.assertFalse(contact_stage(dict(stall_guard_preset="normal", tolerance_preset="tight", max_upward_dz_m=0)))
        for path in (dict(stall_guard_preset="contact_stop"), dict(tolerance_preset="compliant"), dict(max_upward_dz_m=0.01)):
            self.assertTrue(contact_stage(path), path)


if __name__ == "__main__":
    unittest.main()
