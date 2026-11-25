"""Conditional trajectory constraints and data identifiability, with no hardware or image replay."""

NI_RUA_TEST_TIER = "component"

# === tests/test_so101_contact_trajectory.py ===

from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.optimize import brentq

from ni_rua.robot.model import so101_kinematics as kin
from ni_rua.robot.calibration.evidence.contact_trajectory import ContactSegment, fit_contact_trajectories, reviewed_capture_fit


class ContactTrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = kin.load_model()
        cls.point = np.array([0.01, -0.01, -0.03])
        cls.plane = np.array([-0.01, 0.02, -0.01])
        cls.delta = np.array([0, 0.02, -0.03, 0.015, 0])
        rng, rows = np.random.default_rng(22), []
        # Exact rigid-point/table intersections in the actual packaged SO101 chain. This is
        # mathematical ground truth only; it does not assert this arbitrary feature exists physically.
        for _ in range(150):
            q = rng.uniform([-.5, -1.5, -.7, -.7, -1], [.5, 1.5, 1.5, 1.5, 1])
            def gap(lift):
                qt = q.copy(); qt[1] = lift
                T = kin.fk_all(cls.model, qt)["gripper_link"]
                p = T[:3, :3] @ cls.point + T[:3, 3]
                return p[2] - cls.plane[0] - cls.plane[1] * p[0] - cls.plane[2] * p[1]
            grid = np.linspace(-1.5, 1.5, 25)
            for a, b in zip(grid[:-1], grid[1:]):
                if gap(a) * gap(b) < 0:
                    q[1] = brentq(gap, a, b)
                    rows.append(q - cls.delta)
                    break
            if len(rows) >= 40:
                break
        cls.rows = np.asarray(rows)

    def fk(self, q, jaw):
        return kin.fk_all(self.model, q, gripper_rad=jaw)

    def segments(self):
        return [ContactSegment(str(i), "mapping-A", self.rows[i:i + 10], np.zeros(10), np.arange(10),
                               "gripper_link", self.point, "supported", "synthetic rigid-point truth")
                for i in range(0, 40, 10)]

    def test_joint_and_plane_recovery_with_identified_fixed_geometry(self):
        r = fit_contact_trajectories(self.segments(), self.fk,
                                    estimated_joints=("shoulder_lift", "elbow_flex", "wrist_flex"))
        self.assertEqual(r["status"], "conditional_fit")
        self.assertEqual(r["data_rank"], 6)
        np.testing.assert_allclose(list(r["candidate"]["joint_delta_rad"].values()), self.delta, atol=1e-7)
        plane = r["candidate"]["plane"]
        np.testing.assert_allclose([plane["height_at_base_origin_m"], plane["slope_x"], plane["slope_y"]],
                                   self.plane, atol=1e-7)
        self.assertIsNone(r["physical_acceptance"])
        self.assertIsNone(r["uncertainty_m"])
        json.dumps(r, allow_nan=False)

    def test_contact_geometry_recovery_requires_fixed_zeros_and_plane(self):
        correction = np.array([.001, -.002, .0007])
        segments = [replace(s, q_rad=np.asarray(s.q_rad) + self.delta,
                            point_in_link_m=self.point - correction) for s in self.segments()]
        r = fit_contact_trajectories(segments, self.fk, initial_plane=self.plane, estimated_plane=(),
                                    estimated_contact_axes=("x", "y", "z"))
        self.assertEqual(r["status"], "conditional_fit")
        np.testing.assert_allclose(r["candidate"]["contact_point_delta_m"], correction, atol=1e-7)

    def test_encoder_candidate_and_old_trace_are_not_supported_contact(self):
        unknown = [replace(s, evidence="unknown", evidence_note="persistent encoder lag") for s in self.segments()]
        r = fit_contact_trajectories(unknown, self.fk)
        self.assertEqual(r["status"], "insufficient_evidence")
        self.assertIsNone(r["candidate"])
        self.assertEqual(len(r["excluded_segments"]), 4)
        with self.assertRaises(ValueError):
            fit_contact_trajectories([replace(self.segments()[0], evidence_note="")], self.fk)

    def test_single_pose_and_collinear_sliding_do_not_identify_plane(self):
        for rows in ([[0, 0, 0, 0, 0]] * 8, [[v, 0, 0, 0, 0] for v in np.linspace(-.1, .1, 8)]):
            s = ContactSegment("line", "A", rows, [0]*8, range(8), "p", [0,0,0],
                               "supported", "synthetic plane contact")
            def fk(q, jaw):
                T = np.eye(4); T[:3, 3] = [q[0], q[1], q[2]]
                return {"p": T}
            r = fit_contact_trajectories([s], fk)
            self.assertEqual(r["status"], "unidentifiable")
            self.assertIsNone(r["candidate"])
            self.assertGreater(len(r["scaled_null_vectors"]), 0)

    def test_free_pan_with_plane_orientation_exposes_gauge(self):
        r = fit_contact_trajectories(self.segments(), self.fk,
                                    estimated_joints=("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"))
        self.assertEqual(r["status"], "unidentifiable")
        self.assertIsNone(r["candidate"])
        self.assertLess(r["data_rank"], r["parameter_count"])

    @staticmethod
    def xyz_fk(q, jaw):
        T = np.eye(4); T[:3, 3] = q[:3]
        return {"p": T}

    def linear_segment(self, name, z, count=4, role="fit"):
        q = np.array([[0,0,z,0,0], [.1,0,z,0,0], [0,.1,z,0,0], [.1,.1,z,0,0]])
        q = np.tile(q, (count // 4, 1))
        return ContactSegment(name, "A", q, np.zeros(count), np.arange(count), "p", [0,0,0],
                              "supported", "synthetic plane", role)

    def test_plane_height_and_unoriented_tip_height_are_confounded(self):
        r = fit_contact_trajectories([self.linear_segment("a", 0)], self.xyz_fk,
                                    estimated_contact_axes=("z",))
        self.assertEqual(r["status"], "unidentifiable")
        self.assertEqual(r["data_rank"], 3)

    def test_holdout_not_used_for_fit_and_errors_not_hidden(self):
        r = fit_contact_trajectories([self.linear_segment("fit", 0), self.linear_segment("held", .007, role="holdout")],
                                    self.xyz_fk)
        self.assertAlmostEqual(r["candidate"]["plane"]["height_at_base_origin_m"], 0)
        held = next(s for s in r["segments"] if s["role"] == "holdout")
        self.assertAlmostEqual(held["max_abs_m"], .007)
        self.assertIsNone(r["physical_acceptance"])

    def test_holdout_cannot_repair_training_rank(self):
        line = self.linear_segment("line", 0)
        q = np.asarray(line.q_rad).copy(); q[:, 1] = 0
        r = fit_contact_trajectories([replace(line, q_rad=q),
                                     self.linear_segment("held", 0, role="holdout")], self.xyz_fk)
        self.assertEqual(r["status"], "unidentifiable")
        self.assertEqual(r["data_rank"], 2)

    def test_more_samples_from_one_segment_do_not_change_weight(self):
        a, b = self.linear_segment("a", 0), self.linear_segment("b", .004)
        first = fit_contact_trajectories([a, b], self.xyz_fk)
        second = fit_contact_trajectories([self.linear_segment("a", 0, 400), b], self.xyz_fk)
        self.assertAlmostEqual(first["candidate"]["plane"]["height_at_base_origin_m"],
                               second["candidate"]["plane"]["height_at_base_origin_m"], places=9)

    def test_bound_limited_solution_is_not_applicable(self):
        r = fit_contact_trajectories([self.linear_segment("a", .04)], self.xyz_fk,
                                    estimated_plane=("height",), max_plane_height_delta_m=.03)
        self.assertEqual(r["status"], "bound_limited")
        self.assertIsNone(r["candidate"])

    def test_capture_review_preserves_raw_and_requires_explicit_interval(self):
        raw = {"contact_evidence": "supported", "samples": [
            {"q_measured_rad": [0]*5, "gripper_rad": 0, "elapsed_s": i*.1, "phase": "slide"} for i in range(4)]}
        saved = copy.deepcopy(raw)
        s = ContactSegment.from_capture(raw, segment_id="review", basis_id="A", link="p",
                                        point_in_link_m=[0,0,0], sample_indices=[1,2])
        self.assertEqual(s.evidence, "unknown")
        self.assertEqual(s.elapsed_s, [.1,.2])
        self.assertEqual(raw, saved)
        with self.assertRaises(ValueError):
            ContactSegment.from_capture(raw, segment_id="review", basis_id="A", link="p",
                                        point_in_link_m=[0,0,0], sample_indices=[2,1])

    def test_mixed_basis_feature_switch_and_invalid_timestamps_rejected(self):
        s = self.segments()[0]
        invalid = [replace(s, segment_id="other", basis_id="B"),
                   replace(s, segment_id="other", point_in_link_m=[0,0,0]),
                   replace(s, segment_id="other", elapsed_s=[0]*10)]
        for other in invalid:
            with self.subTest(other=other.segment_id), self.assertRaises(ValueError):
                fit_contact_trajectories([s, other], self.fk)
        with self.assertRaises(ValueError):
            fit_contact_trajectories([replace(s, jaw_rad=[float("nan")]*10)], self.fk)

    def capture_review(self, directory, *, evidence="supported"):
        """Actual touch-report shape, with synthetic actual-SO101 rigid contact samples."""
        capture_path, review_path = Path(directory) / "capture.json", Path(directory) / "review.json"
        mapping = {name: {"sign": 1, "offset_rad": 0} for name in kin.ARM_JOINTS}
        capture = {"schema": "ni_rua.so101.touch_table.v1", "contact_trajectories": []}
        annotations = []
        for s in self.segments():
            capture["contact_trajectories"].append({
                "trajectory_id": s.segment_id, "joint_mapping": copy.deepcopy(mapping),
                "contact_evidence": "unknown", "contact_link": None, "point_in_link_m": None,
                "samples": [{"q_measured_rad": q.tolist(), "gripper_rad": float(jaw), "elapsed_s": float(t),
                             "phase": "slide_out"} for q, jaw, t in zip(s.q_rad, s.jaw_rad, s.elapsed_s)]})
            annotations.append({"segment_id": "review-" + s.segment_id, "trajectory_id": s.segment_id,
                                "sample_indices": list(range(10)), "link": s.link,
                                "point_in_link_m": s.point_in_link_m.tolist(), "evidence": evidence,
                                "evidence_note": "synthetic supported contact truth", "role": "fit"})
        capture_path.write_text(json.dumps(capture))
        review = {"schema": "ni_rua.so101.contact_trajectory_review.v1",
                  "capture_sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
                  "segments": annotations,
                  "fit_settings": {"estimated_joints": ["shoulder_lift", "elbow_flex", "wrist_flex"]}}
        review_path.write_text(json.dumps(review))
        return capture_path, review_path, capture, review

    def test_cli_capture_review_fit_preserves_inputs_and_records_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            capture, review, _, _ = self.capture_review(directory)
            original = capture.read_bytes(), review.read_bytes()
            output = Path(directory) / "result.json"
            process = subprocess.run([sys.executable, "-m", "ni_rua.robot.calibration.evidence.contact_trajectory",
                                      "--capture", str(capture), "--review", str(review), "--output", str(output)],
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "conditional_fit")
            np.testing.assert_allclose(list(result["candidate"]["joint_delta_rad"].values()), self.delta, atol=1e-7)
            self.assertIn("effective_fit_settings", result["operator_input_binding"])
            self.assertEqual((capture.read_bytes(), review.read_bytes()), original)
            with self.assertRaises(FileExistsError):
                reviewed_capture_fit(capture, review, output)
            with self.assertRaises(ValueError):
                reviewed_capture_fit(capture, review, capture)
            self.assertEqual((capture.read_bytes(), review.read_bytes()), original)

    def test_cli_unknown_has_no_candidate_and_describe_never_fits(self):
        with tempfile.TemporaryDirectory() as directory:
            capture, review, _, _ = self.capture_review(directory, evidence="unknown")
            output = Path(directory) / "unknown.json"
            process = subprocess.run([sys.executable, "-m", "ni_rua.robot.calibration.evidence.contact_trajectory",
                                      "--capture", str(capture), "--review", str(review), "--output", str(output)],
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 2, process.stderr)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "insufficient_evidence")
            self.assertIsNone(result["candidate"])
            described = subprocess.run([sys.executable, "-m", "ni_rua.robot.calibration.evidence.contact_trajectory",
                                       "--capture", str(capture), "--describe"], capture_output=True, text=True)
            self.assertEqual(described.returncode, 0, described.stderr)
            self.assertEqual(json.loads(described.stdout)["capture_sha256"],
                             hashlib.sha256(capture.read_bytes()).hexdigest())

    def test_review_binding_mapping_and_whole_acquisition_holdout_are_enforced(self):
        cases = ("wrong_binding", "mixed_mapping", "missing_mapping", "forged_basis", "overlap", "split_role")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                capture_path, review_path, capture, review = self.capture_review(directory)
                if case == "wrong_binding":
                    review["capture_sha256"] = "incorrect"
                elif case in ("mixed_mapping", "missing_mapping"):
                    row = capture["contact_trajectories"][1]["joint_mapping"]
                    if case == "mixed_mapping":
                        row["wrist_flex"]["offset_rad"] = .01
                    else:
                        del row["wrist_flex"]
                    capture_path.write_text(json.dumps(capture))
                    review["capture_sha256"] = hashlib.sha256(capture_path.read_bytes()).hexdigest()
                elif case == "forged_basis":
                    review["segments"][0]["basis_id"] = "pretend-same"
                else:
                    extra = copy.deepcopy(review["segments"][0]); extra["segment_id"] = "extra"
                    if case == "split_role":
                        review["segments"][0]["sample_indices"] = [0,1,2,3,4]
                        extra.update(sample_indices=[5,6,7,8,9], role="holdout")
                    review["segments"].append(extra)
                review_path.write_text(json.dumps(review))
                output = Path(directory) / "result.json"
                with self.assertRaises(ValueError):
                    reviewed_capture_fit(capture_path, review_path, output)
                self.assertFalse(output.exists())

# === tests/test_so101_motion_features.py ===

import dataclasses
import math
from pathlib import Path
import re
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from ni_rua.core.config import Config
from ni_rua.robot.backends.so101 import contract
from ni_rua.robot.backends.so101.driver import (FakeArmDriver, FakeCameraSource, FakeClock,
                                                 gripper_normalized_from_rad)
from ni_rua.robot.backends.so101.worker import SO101Adapter, arrival_verdict, quaternion_to_matrix
from ni_rua.robot.model import so101_kinematics
from robot.so101.test_so101_backend import EXAMPLE, IDENTITY, InProcessBackend, deployment

EMPTY_CLOSE_M = 0.0035  # the fake jaw closes to the CAD closed limit (about 0.0031 m), as in the grip-guard tests
OBJECT_JAW = 0.15  # normalized jaw opening at which a held object stops the fake jaw (about 0.03 m)


class TableArmDriver(FakeArmDriver):
    """``FakeArmDriver`` above a rigid table at ``table_z_m`` (m, so101_base; None = no table): a measured joint update
    that would put the TCP below the table is not applied, so the arm rests on it while its goal keeps going."""

    def __init__(self, deployment, clock, table_z_m=None, **options):
        super().__init__(deployment, clock, **options)
        self.model, self.table_z_m = so101_kinematics.load_model(), table_z_m

    def _advance(self):
        before = list(self.q)
        super()._advance()
        if self.table_z_m is not None and so101_kinematics.fk(self.model, self.q)[2, 3] < self.table_z_m:
            self.q = before


def table_adapter(directory, *, table_z_m=None, profile="rgb_proprio_v1", **driver_options):
    deploy = deployment(**{"gripper.empty_close_aperture_m": EMPTY_CLOSE_M})
    for name in ("fixed", "wrist"):
        deploy["cameras"][name].update(width=64, height=48)
    clock = FakeClock()
    driver = TableArmDriver(deploy, clock, table_z_m, **driver_options)
    cameras = {name: FakeCameraSource(name, clock=clock, width=64, height=48) for name in ("fixed", "wrist")}
    adapter = SO101Adapter(deploy, directory, profile, driver=driver, cameras=cameras, clock=clock,
                           image_settings=Config().images.to_dict(), start_telemetry=False)
    return adapter, driver


def batch(*stages, timeout_s=30, **fields):
    return dict(type="BatchAction", frame_id=contract.FRAME_ID, tcp_config_id=contract.TCP_CONFIG_ID,
                limits_profile=contract.LIMITS_PROFILE, timeout_s=timeout_s, stages=list(stages), **fields)


class Case(unittest.TestCase):
    def adapter(self, **options):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        adapter, driver = table_adapter(directory.name, **options)
        self.addCleanup(adapter.close)
        return adapter, driver

    @staticmethod
    def pose(adapter):
        return adapter.robot_state(adapter.read())["arms"]["right"]["ee_pose"]

    def move(self, adapter, identifier, dz=0.0, dx=0.0, *, tolerance=0.005, args=None, stage=None, **path):
        """A compiled SO-101 move stage relative to the measured TCP (m), keeping its orientation."""
        pose = self.pose(adapter)
        leaf = contract.compile_action("move", dict(arm="right", observation_id="o", quaternion_xyzw=pose["quaternion_xyzw"],
                                                    position_m=[pose["position_m"][0] + dx, pose["position_m"][1],
                                                                pose["position_m"][2] + dz], **(args or {})),
                                       IDENTITY, adapter.capabilities())
        leaf["path"].update(position_tolerance_m=tolerance, orientation_tolerance_rad=0.1, **path)
        return dict(id=identifier, kind="move", arm="right", action=leaf, **(stage or {}))

    @staticmethod
    def jaw(adapter, identifier, goal, **stage):
        leaf = contract.compile_action("gripper", dict(arm="right", goal_normalized=goal, observation_id="o"), IDENTITY,
                                       adapter.capabilities())
        return dict(id=identifier, kind="gripper", arm="right", action=leaf, **stage)

    def run_batch(self, adapter, action):
        preview = adapter.preview_batch(action)
        self.assertEqual(preview["status"], "planned", preview)
        receipt = adapter.execute(dict(action, batch_plan_id=preview["plan_id"]), lambda update: None, threading.Event())
        return preview, receipt

    def harness(self, **options):
        """A Harness over the table fake (in-process backend with the profile hook), as the actor reaches it."""
        from ni_rua.core.storage.store import Store
        from ni_rua.runtime.harness import Harness
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        adapter, driver = table_adapter(directory.name, **options)
        self.addCleanup(adapter.close)
        store = Store(Path(directory.name) / "store", IDENTITY)
        self.addCleanup(store.close)
        config = Config.from_dict({"backend": {"environment": "so101_real", "robot_model_id": "so101",
                                               "deployment": str(EXAMPLE)},
                                   "features": {"observation_profile": "rgb_proprio_v1", "record_video": False}})
        harness = Harness(store, InProcessBackend(adapter), observation_profile="rgb_proprio_v1", config=config)
        harness.start()
        harness.task_planning.required = False
        return harness, adapter, driver

    def robot_action(self, harness, stages, **request):
        """Preview then execute one robot_action request through the shared tool path (unified_actions.dispatch)."""
        from ni_rua.runtime.actions.unified_actions import dispatch
        with patch("ni_rua.runtime.actions.inspection._trajectory_image", return_value=({"status": "unavailable"}, [])):
            preview = dispatch(harness, dict(observation_id=harness.latest["observation_id"], stages=stages, **request),
                               {"tool_call_id": "preview"})
            self.assertEqual(preview.data["status"], "planned", preview.data)
            response = dispatch(harness, dict(mode="execute", plan_id=preview.data["plan_id"]), {"tool_call_id": "execute"})
        return preview.data, response.data["receipt"]

    def public_move(self, adapter, identifier, dz=0.0, **fields):
        """A public robot_action move stage relative to the measured TCP (m), keeping its orientation."""
        pose = self.pose(adapter)
        return dict(id=identifier, kind="move", arm="right", quaternion_xyzw=pose["quaternion_xyzw"],
                    position_m=[pose["position_m"][0], pose["position_m"][1], pose["position_m"][2] + dz], **fields)


# ---- R5 -------------------------------------------------------------------------------------------------------------
class JawRampTests(Case):
    RAMP = dict(start_fraction=0.25, end_fraction=0.75)

    def ramp_move(self, adapter, identifier="m", goal=0.0, ramp=None, dz=0.01, stage=None, **args):
        return self.move(adapter, identifier, dz=dz, stage=stage,
                         args=dict(gripper_goal_normalized=goal, gripper_motion=dict(ramp or self.RAMP), **args))

    def test_profile_and_capabilities_offer_the_during_motion_phase(self):
        adapter, _ = self.adapter()
        self.assertIn("gripper_motion", adapter.embodiment.supports)
        capabilities = adapter.capabilities()
        self.assertEqual(capabilities["gripper_phases"], ["after_motion", "during_motion"])
        self.assertIn("during_motion", capabilities["gripper_semantics"])
        stage = self.ramp_move(adapter)
        self.assertEqual(stage["action"]["gripper_events"], [dict(arm="right", phase="during_motion", goal_normalized=0.0,
                                                                  start_fraction=0.25, end_fraction=0.75)])

    def test_preview_cost_and_schedule_respect_the_jaw_speed_limit(self):
        adapter, _ = self.adapter(gripper_normalized=1.0)
        plain = adapter.preview_batch(batch(self.move(adapter, "m", dz=0.01)))["stage_costs"]["m"]["native_steps"]
        step = adapter._jaw_step_normalized(1.0)
        for scale in (1.0, 0.5):
            with self.subTest(gripper_speed_scale=scale):
                preview = adapter.preview_batch(batch(self.ramp_move(adapter, gripper_speed_scale=scale)))
                self.assertEqual(preview["status"], "planned", preview)
                plan, cost = preview["stage_plans"]["m"], preview["stage_costs"]["m"]["native_steps"]
                schedule = plan["gripper_motion"]
                self.assertEqual((schedule["initial_goal_normalized"], schedule["goal_normalized"]), (1.0, 0.0))
                self.assertEqual((schedule["start_fraction"], schedule["end_fraction"]), (0.25, 0.75))
                self.assertEqual(schedule["original_motion_steps"], plain)
                self.assertEqual(schedule["scheduled_motion_steps"], cost)
                self.assertEqual(cost, plain * schedule["subdivision_factor"])
                self.assertAlmostEqual(schedule["max_goal_step"], round(step * scale, 6), places=6)
                # The ramp covers half of the move's commands: it needs at least 1 / max step commands there.
                self.assertGreaterEqual(cost * 0.5, 1.0 / (step * scale) - 1)
                self.assertNotIn("goal_normalized_by_step", schedule)
                self.assertEqual(plan["goal_normalized"], 0.0)
        slow = adapter.preview_batch(batch(self.ramp_move(adapter, gripper_speed_scale=0.5)))
        fast = adapter.preview_batch(batch(self.ramp_move(adapter)))
        self.assertGreater(slow["planned_native_steps"], fast["planned_native_steps"])

    def test_execution_ramps_the_jaw_inside_the_move_then_measures_it(self):
        adapter, _ = self.adapter()  # jaw starts closed (0), opens during the move
        preview, receipt = self.run_batch(adapter, batch(self.ramp_move(adapter, goal=1.0)))
        self.assertEqual(receipt["status"], "completed", receipt)
        result = receipt["stage_results"]["m"]
        schedule = preview["stage_plans"]["m"]["gripper_motion"]
        self.assertEqual(result["gripper_motion"]["completed_steps"], schedule["scheduled_motion_steps"])
        execution = result["gripper_execution"]
        self.assertEqual((execution["phase"], execution["goal_normalized"], execution["jaw_settled"]),
                         ("during_motion", 1.0, True))
        self.assertAlmostEqual(execution["measured_normalized"], 1.0, places=2)
        # The commanded jaw column of this move: latched until 25 %, linear, at the goal from 75 % of the commands.
        kinds = receipt["planned_trace"]["row_kinds"]
        rows = [row for row, kind in zip(receipt["planned_trace"]["q_rad"], kinds) if kind == "motion"]
        jaws = [gripper_normalized_from_rad(adapter.deployment, row[5]) for row in rows]
        count = len(jaws)
        self.assertEqual(count, schedule["scheduled_motion_steps"])
        for index, value in enumerate(jaws, start=1):
            progress = index / count
            if progress <= 0.25:
                self.assertAlmostEqual(value, 0.0, places=6)
            elif progress >= 0.75:
                self.assertAlmostEqual(value, 1.0, places=6)
        steps = np.abs(np.diff([0.0, *jaws]))
        self.assertLessEqual(float(steps.max()), adapter._jaw_step_normalized(1.0) + 1e-9)
        self.assertTrue(all(b >= a - 1e-12 for a, b in zip(jaws, jaws[1:])))
        self.assertNotIn("gripper", kinds)  # no separate jaw stage: the jaw moved inside the move

    def test_guard_on_a_closing_ramp_pauses_on_an_empty_close_and_passes_on_an_object(self):
        for jaw_min, expected in ((None, "aperture_below_min_m"), (OBJECT_JAW, None)):
            with self.subTest(object_in_jaw=jaw_min is not None):
                adapter, _ = self.adapter(gripper_normalized=1.0, jaw_min_normalized=jaw_min)
                stages = [self.ramp_move(adapter, stage=dict(grip_guard_preset="rigid_object")),
                          self.move(adapter, "lift", dz=0.03)]
                preview, receipt = self.run_batch(adapter, batch(*stages))
                guard = preview["stage_plans"]["m"]["grip_guard"]
                self.assertEqual((guard["grip_guard_preset"], guard["min_aperture_m"]), ("rigid_object", EMPTY_CLOSE_M + 0.002))
                evaluation = receipt["stage_results"]["m"]["grip_guard"]
                self.assertEqual(evaluation["jaw_settled"], True)
                if expected:
                    self.assertEqual((receipt["status"], receipt["checkpoint_reason"]), ("stopped", expected), receipt)
                    self.assertEqual(receipt["checkpoint_stage_ids"], ["m"])
                    self.assertEqual(receipt["stage_results"]["lift"]["status"], "pending")
                    self.assertTrue(receipt["stop_confirmation"]["confirmed"])
                else:
                    self.assertEqual(receipt["status"], "completed", receipt)
                    self.assertFalse(evaluation["triggered"])
                    self.assertGreater(evaluation["measured_aperture_m"], evaluation["min_aperture_m"])

    def test_checkpoint_after_gripper_on_a_ramp_move_pauses_before_the_next_move(self):
        adapter, _ = self.adapter(gripper_normalized=1.0, jaw_min_normalized=OBJECT_JAW)
        stages = [self.ramp_move(adapter, stage=dict(checkpoint_after_gripper=True)), self.move(adapter, "lift", dz=0.03)]
        preview, receipt = self.run_batch(adapter, batch(*stages))
        self.assertIn({"stage_id": "m", "cursor": preview["stage_costs"]["lift"]["start_native_step"],
                       "reason": receipt["checkpoint_reasons"]["m"], "pause_before_stage_id": "lift"}, preview["checkpoints"])
        self.assertEqual((receipt["status"], receipt["reason"]), ("stopped", "observation_checkpoint"), receipt)
        self.assertEqual(receipt["completed_stage_ids"], ["m"])

    def test_schedule_longer_than_the_timeout_is_refused_at_preview(self):
        adapter, _ = self.adapter(gripper_normalized=1.0)
        preview = adapter.preview_batch(batch(self.ramp_move(adapter), timeout_s=1.0))
        self.assertEqual((preview["status"], preview["reason"], preview["failed_stage_id"]),
                         ("failed", "gripper_motion_schedule_exceeds_timeout", "m"), preview)
        self.assertEqual(preview["failure_feedback"]["max_commands"], 20)  # 1.0 s x 20 Hz
        self.assertFalse(preview["physics_advanced"])

    def test_contract_refuses_the_ramp_without_the_profile_tag_and_bad_events(self):
        adapter, _ = self.adapter()
        stage = self.ramp_move(adapter)
        contract.validate_action(batch(stage), adapter.deployment)
        profile = contract.embodiment(adapter.deployment)
        without = dataclasses.replace(profile, supports=profile.supports - {"gripper_motion"})
        with patch.object(contract, "embodiment", return_value=without):
            with self.assertRaisesRegex(ValueError, "unsupported_capability:gripper_events:during_motion"):
                contract.validate_action(batch(stage), adapter.deployment)
        capabilities = dict(adapter.capabilities(), gripper_phases=["after_motion"])
        pose = self.pose(adapter)
        args = dict(arm="right", observation_id="o", position_m=pose["position_m"], quaternion_xyzw=pose["quaternion_xyzw"],
                    gripper_goal_normalized=0.0, gripper_motion=dict(self.RAMP))
        with self.assertRaisesRegex(ValueError, "unsupported_capability: gripper_motion"):
            contract.compile_action("move", args, IDENTITY, capabilities)
        with self.assertRaisesRegex(ValueError, "invalid_gripper_motion: requires a move and gripper_goal_normalized"):
            contract.compile_action("move", {key: value for key, value in args.items() if key != "gripper_goal_normalized"},
                                    IDENTITY, adapter.capabilities())
        for change, error in ((dict(start_fraction=0.8), "invalid_gripper_motion"),
                              (dict(extra=1), "a during_motion event has exactly"),
                              (dict(phase="after_motion"), "batch_after_motion_jaw_requires_separate_stage")):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, error):
                bad = self.ramp_move(adapter)
                bad["action"]["gripper_events"][0].update(change)
                contract.validate_action(batch(bad), adapter.deployment)
        # Grip-guard fields on a move need its jaw ramp.
        with self.assertRaisesRegex(ValueError, "applies to jaw stages"):
            contract.validate_action(batch(self.move(adapter, "m", dz=0.01, stage=dict(min_aperture_m=0.01))),
                                     adapter.deployment)

    def test_robot_action_gripper_motion_end_to_end(self):
        """The public fields (gripper_goal_normalized + gripper_motion + grip guard on the move) through the shared
        tool path: one move stage with a during_motion event, planned and executed by the worker."""
        harness, adapter, _ = self.harness(gripper_normalized=1.0, jaw_min_normalized=OBJECT_JAW)
        stages = [self.public_move(adapter, "grasp", dz=0.01, gripper_goal_normalized=0.0, gripper_motion=dict(self.RAMP),
                                   grip_guard_preset="rigid_object", position_tolerance_m=0.005, orientation_tolerance_rad=0.1),
                  self.public_move(adapter, "lift", dz=0.03, position_tolerance_m=0.005, orientation_tolerance_rad=0.1)]
        preview, receipt = self.robot_action(harness, stages)
        self.assertEqual(set(preview["stage_plans"]), {"grasp", "lift"})  # no generated "/motion" jaw stage
        self.assertEqual(preview["stage_plans"]["grasp"]["gripper_motion"]["goal_normalized"], 0.0)
        self.assertEqual(preview["stage_plans"]["grasp"]["grip_guard"]["grip_guard_preset"], "rigid_object")
        self.assertEqual(receipt["status"], "completed", receipt)
        self.assertFalse(receipt["stage_results"]["grasp"]["grip_guard"]["triggered"])

    def test_single_request_ramp_runs_on_the_final_waypoint(self):
        adapter, _ = self.adapter()
        action = self.ramp_move(adapter, goal=1.0)["action"]
        receipt = adapter.execute(action, lambda update: None, threading.Event())
        self.assertEqual(receipt["status"], "completed", receipt)
        self.assertEqual(receipt["gripper_execution"]["right"]["phase"], "during_motion")
        self.assertEqual(receipt["waypoint_results"][0]["gripper_motion"]["completed_steps"],
                         receipt["waypoint_results"][0]["gripper_motion"]["scheduled_motion_steps"])
        self.assertNotIn("gripper", receipt["planned_trace"]["row_kinds"])


# ---- R6 -------------------------------------------------------------------------------------------------------------
class VerticalWindowTests(Case):
    def test_arrival_rule(self):
        residual = dict(dx_m=0.003, dy_m=0.0, dz_m=0.012, position_error_m=math.hypot(0.003, 0.012), orientation_error_rad=0.05)
        self.assertEqual(arrival_verdict(residual, 0.005, 0.1), (False, False))  # default: no window
        self.assertEqual(arrival_verdict(residual, 0.005, 0.1, 0.015), (True, True))
        self.assertEqual(arrival_verdict(residual, 0.005, 0.1, 0.010), (False, False))  # above the window
        self.assertEqual(arrival_verdict(residual, 0.005, 0.01, 0.015), (False, False))  # orientation still checked
        self.assertEqual(arrival_verdict(dict(residual, dx_m=0.006), 0.005, 0.1, 0.015), (False, False))  # horizontal
        low = dict(residual, dz_m=-0.004, position_error_m=0.005)
        self.assertEqual(arrival_verdict(low, 0.005, 0.1, 0.015), (True, False))
        self.assertEqual(arrival_verdict(dict(low, dz_m=-0.006, position_error_m=0.0067), 0.005, 0.1, 0.015), (False, False))
        within = dict(residual, dx_m=0.001, dz_m=0.002, position_error_m=0.0022)
        self.assertEqual(arrival_verdict(within, 0.005, 0.1, 0.015), (True, False))  # plain 3D arrival first

    def test_default_window_is_zero_and_reported(self):
        adapter, _ = self.adapter()
        self.assertIn("max_upward_dz", adapter.embodiment.supports)
        preview, receipt = self.run_batch(adapter, batch(self.move(adapter, "m", dz=0.01)))
        self.assertEqual(preview["stage_plans"]["m"]["tolerances"]["max_upward_dz_m"], 0.0)
        self.assertEqual(receipt["stage_results"]["m"]["effective_tolerances"]["max_upward_dz_m"], 0.0)
        explicit, _ = self.run_batch(adapter, batch(self.move(adapter, "m", dz=0.01, max_upward_dz_m=0.03)))
        self.assertEqual(explicit["stage_plans"]["m"]["tolerances"]["max_upward_dz_m"], 0.03)

    def test_robot_action_stage_and_batch_windows_reach_the_worker(self):
        harness, adapter, driver = self.harness()
        driver.table_z_m = self.pose(adapter)["position_m"][2] - 0.02 + 0.008
        common = dict(position_tolerance_m=0.005, orientation_tolerance_rad=0.1)
        preview, receipt = self.robot_action(harness, [self.public_move(adapter, "down", dz=-0.02, **common)],
                                             max_upward_dz_m=0.015)
        self.assertEqual(preview["stage_plans"]["down"]["tolerances"]["max_upward_dz_m"], 0.015)
        self.assertEqual(receipt["status"], "completed", receipt)
        self.assertTrue(receipt["stage_results"]["down"]["vertical_compliance_accepted"])
        harness, adapter, _ = self.harness()
        preview, _ = self.robot_action(harness, [self.public_move(adapter, "m", dz=0.01, max_upward_dz_m=0.02, **common)],
                                       max_upward_dz_m=0.05)
        self.assertEqual(preview["stage_plans"]["m"]["tolerances"]["max_upward_dz_m"], 0.02)  # the stage value wins

    def test_resting_above_the_target_arrives_only_inside_an_explicit_window(self):
        for window, completed in ((0.015, True), (None, False)):
            with self.subTest(max_upward_dz_m=window):
                adapter, driver = self.adapter()
                driver.table_z_m = self.pose(adapter)["position_m"][2] - 0.02 + 0.008  # rests 8 mm above the target
                path = {} if window is None else dict(max_upward_dz_m=window)
                _, receipt = self.run_batch(adapter, batch(self.move(adapter, "down", dz=-0.02, **path)))
                result = receipt["stage_results"]["down"]
                self.assertGreater(result["dz_m"], 0.005)
                self.assertNotIn("tracking_stop", result)  # the segment ran to its end; the arrival rule decided
                if completed:
                    self.assertEqual(receipt["status"], "completed", receipt)
                    self.assertTrue(result["vertical_compliance_accepted"] and result["within_tolerance"])
                    self.assertLessEqual(result["xy_error_m"], 0.005)
                    self.assertEqual(result["effective_tolerances"]["max_upward_dz_m"], window)
                    self.assertNotIn("contact_block_summary", receipt)
                else:
                    self.assertEqual((receipt["status"], receipt["reason"]), ("failed", "tracking_error"))
                    summary = receipt["contact_block_summary"]
                    self.assertEqual(summary, result["contact_block_summary"])
                    self.assertTrue(summary["blocked_above_target"])
                    self.assertAlmostEqual(summary["vertical_shortfall_m"], result["dz_m"], delta=2e-4)
                    self.assertEqual(summary["completed_steps"], summary["planned_steps"])


# ---- R7 phase 2 -----------------------------------------------------------------------------------------------------
class ContactStopTests(Case):
    """A descent whose target is 30 mm below the table (TCP z): the commanded path keeps going after the contact."""

    def descent(self, adapter, driver, *, below_table=0.03, depth=0.06, **path):
        driver.table_z_m = self.pose(adapter)["position_m"][2] - depth + below_table
        return self.move(adapter, "down", dz=-depth, tolerance=0.01, **path)

    def test_contact_stop_inside_the_window_holds_and_the_gripper_stage_runs(self):
        adapter, driver = self.adapter(gripper_normalized=1.0)
        stage = self.descent(adapter, driver, stall_guard_preset="contact_stop", max_tracking_divergence_m=0.02,
                             max_upward_dz_m=0.04)
        target = np.asarray(stage["action"]["path"]["waypoints"][0]["right"]["position_m"])
        rotation = quaternion_to_matrix(stage["action"]["path"]["waypoints"][0]["right"]["quaternion_xyzw"])
        _, receipt = self.run_batch(adapter, batch(stage, self.jaw(adapter, "close", 0.0), self.move(adapter, "lift", dz=0.0)))
        self.assertEqual(receipt["status"], "completed", receipt)
        down = receipt["stage_results"]["down"]
        self.assertEqual(down["status"], "completed")
        self.assertTrue(down["contact_stopped_early"] and down["within_tolerance"] and down["vertical_compliance_accepted"])
        self.assertEqual(down["tracking_stop"]["kind"], "tcp_divergence")
        self.assertLess(down["tracking_stop"]["completed_steps"], down["tracking_stop"]["planned_steps"])
        self.assertTrue(down["hold_confirmation"]["confirmed"])
        self.assertNotIn("arrival_correction", down)  # a contact hold is never corrected toward the target
        summary = down["contact_block_summary"]
        measured = np.asarray(summary["measured_position_m"])
        self.assertTrue(summary["blocked_above_target"])
        self.assertAlmostEqual(summary["vertical_shortfall_m"], 0.03, delta=0.003)
        self.assertAlmostEqual(summary["approach_shortfall_m"], float(np.dot(target - measured, rotation[:, 2])), delta=2e-4)
        self.assertNotIn("contact_block_summary", receipt)  # top level only on a failed batch
        self.assertEqual(receipt["stage_results"]["close"]["status"], "completed")
        kinds = receipt["planned_trace"]["row_kinds"]
        self.assertEqual(kinds.count("hold"), 1)  # one hold command: the contact hold is not re-pinned
        # The hold rows have their measured samples, so the trace stays aligned row by row.
        self.assertTrue(receipt["trace_sampling"]["actual_boundary_samples"], receipt["trace_sampling"])
        # The jaw closes at the held pose: its rows keep the hold's arm goal.
        rows = receipt["planned_trace"]["q_rad"]
        held = rows[len(kinds) - 1 - kinds[::-1].index("hold")][:5]
        first_jaw = next(index for index, (row, kind) in enumerate(zip(rows, kinds))
                         if kind == "motion" and index > kinds.index("hold") and row[5] != rows[index - 1][5])
        self.assertEqual(rows[first_jaw][:5], held)

    def test_contact_hold_pins_the_measured_pose_without_the_sag_offset(self):
        """Real-arm finding: re-applying the free-space sag offset at a contact lifted the arm off the support. The
        contact hold commands the measured joints exactly; other stops keep the sag-compensated hold."""
        adapter, driver = self.adapter()
        adapter.hold_offset = [0.0, -0.05, -0.04, 0.0, 0.0]  # a learned free-space sag offset (rad)
        stage = self.descent(adapter, driver, stall_guard_preset="contact_stop", max_tracking_divergence_m=0.02,
                             max_upward_dz_m=0.04)
        _, receipt = self.run_batch(adapter, batch(stage))
        down = receipt["stage_results"]["down"]
        self.assertTrue(down["contact_stopped_early"])
        self.assertEqual(down["hold_confirmation"]["hold_offset_rad"], [0.0] * 5)
        kinds, rows = receipt["planned_trace"]["row_kinds"], receipt["planned_trace"]["q_rad"]
        holds = [row[:5] for row, kind in zip(rows, kinds) if kind == "hold"]
        stop = receipt["actual_trace"][kinds.index("hold")]["state"]["arms"]["right"]["q_rad"][:5]
        np.testing.assert_allclose(holds[0], stop, atol=1e-6)  # the first hold goal is the measured stop pose
        # A later pause at the held pose (observe stage) re-commands the held goal: no stale free-space offset.
        adapter, driver = self.adapter()
        adapter.hold_offset = [0.0, -0.05, -0.04, 0.0, 0.0]
        stage = self.descent(adapter, driver, stall_guard_preset="contact_stop", max_tracking_divergence_m=0.02,
                             max_upward_dz_m=0.04)
        _, receipt = self.run_batch(adapter, batch(stage, dict(id="look", kind="observe", reason="check the contact"),
                                                   self.move(adapter, "lift", dz=0.0)))
        self.assertEqual((receipt["status"], receipt["reason"]), ("stopped", "observation_checkpoint"), receipt)
        self.assertTrue(receipt["stage_results"]["down"]["contact_stopped_early"])
        self.assertEqual(receipt["stop_confirmation"]["hold_offset_rad"], [0.0] * 5)
        kinds, rows = receipt["planned_trace"]["row_kinds"], receipt["planned_trace"]["q_rad"]
        holds = [row[:5] for row, kind in zip(rows, kinds) if kind == "hold"]
        self.assertEqual(len(holds), 2)  # one verified hold goal at contact and one at the observe pause
        np.testing.assert_allclose(holds[1], holds[0], atol=1e-3)  # the pause re-commands the held goal
        # A stop without contact_stop keeps the sag-compensated hold (offset reported).
        adapter, driver = self.adapter()
        adapter.hold_offset = [0.0, -0.05, -0.04, 0.0, 0.0]
        _, receipt = self.run_batch(adapter, batch(self.descent(adapter, driver)))
        self.assertEqual(receipt["stop_confirmation"]["hold_offset_rad"], [0.0, -0.05, -0.04, 0.0, 0.0])

    def test_a_free_space_contact_hold_stays_at_the_reported_pose(self):
        """A contact_stop stop without a support (a lagging, sagging servo trips the stall guard in free space): the
        hold keeps the measured stop pose as its goal and is not re-pinned, so the arm stays at the pose the receipt
        reports. A re-pin at the settled joints (no sag offset) would let it sag one more step after the report."""
        adapter, driver = self.adapter(tau_s=0.15, sag_rad=(0.0, 0.04, 0.04, 0.0, 0.0))
        stage = self.move(adapter, "down", dz=-0.03, tolerance=0.01, stall_guard_preset="contact_stop",
                          max_tracking_divergence_m=0.005, max_upward_dz_m=0.04)
        _, receipt = self.run_batch(adapter, batch(stage))
        down = receipt["stage_results"]["down"]
        self.assertTrue(down["contact_stopped_early"] and down["hold_confirmation"]["confirmed"], down)
        self.assertEqual(down["tracking_stop"]["kind"], "tcp_divergence")
        adapter.clock.sleep(2.0)  # let the fake servo converge at its last goal
        reported = np.asarray(down["contact_block_summary"]["measured_position_m"])
        now = adapter.tcp(adapter.read()["q_rad"])[:3, 3]
        self.assertLess(float(np.linalg.norm(now - reported)), 0.001)
        self.assertEqual(receipt["planned_trace"]["row_kinds"].count("hold"), 1)

    def test_a_ramp_cut_short_by_a_contact_stop_finishes_at_the_held_pose(self):
        """ARX-X5 keeps the jaw schedule of held rows; on SO-101 the rest of the ramp runs at the held pose."""
        adapter, driver = self.adapter(gripper_normalized=1.0)
        driver.table_z_m = self.pose(adapter)["position_m"][2] - 0.06 + 0.03  # the target is 30 mm below the table
        leaf = self.move(adapter, "down", dz=-0.06, tolerance=0.01, args=dict(
            gripper_goal_normalized=0.0, gripper_motion=dict(start_fraction=0.0, end_fraction=1.0)),
            stall_guard_preset="contact_stop", max_tracking_divergence_m=0.02, max_upward_dz_m=0.04)
        _, receipt = self.run_batch(adapter, batch(leaf))
        self.assertEqual(receipt["status"], "completed", receipt)
        down = receipt["stage_results"]["down"]
        self.assertTrue(down["contact_stopped_early"])
        self.assertLess(down["gripper_motion"]["completed_steps"], down["gripper_motion"]["scheduled_motion_steps"])
        execution = down["gripper_execution"]
        self.assertTrue(execution["finished_at_held_pose"] and execution["command_complete"])
        self.assertEqual(execution["phase"], "during_motion")
        self.assertAlmostEqual(execution["measured_normalized"], 0.0, places=2)
        kinds, rows = receipt["planned_trace"]["row_kinds"], receipt["planned_trace"]["q_rad"]
        last_hold = len(kinds) - 1 - kinds[::-1].index("hold")
        for row in rows[last_hold + 1:]:  # the remaining jaw rows keep the held arm goal
            self.assertEqual(row[:5], rows[last_hold][:5])

    def test_contact_stop_outside_the_window_fails_with_one_hold(self):
        adapter, driver = self.adapter()
        stage = self.descent(adapter, driver, stall_guard_preset="contact_stop", max_tracking_divergence_m=0.02)
        _, receipt = self.run_batch(adapter, batch(stage, self.jaw(adapter, "close", 0.0)))
        self.assertEqual((receipt["status"], receipt["reason"], receipt["failed_stage_id"]), ("failed", "tracking_error", "down"))
        down = receipt["stage_results"]["down"]
        self.assertEqual(down["status"], "failed")
        self.assertTrue(down["contact_stopped_early"])
        self.assertFalse(down["within_tolerance"])
        self.assertEqual(down["effective_tolerances"]["max_upward_dz_m"], 0.0)
        self.assertEqual(receipt["contact_block_summary"], down["contact_block_summary"])
        self.assertEqual(receipt["stop_confirmation"], down["hold_confirmation"])
        self.assertEqual(receipt["planned_trace"]["row_kinds"].count("hold"), 1)  # no second hold, no re-pin
        self.assertEqual(receipt["stage_results"]["close"]["status"], "pending")

    def test_uncompliant_descent_stall_fails_even_inside_a_window(self):
        """Law 6: without contact_stop a stop above the target is a failed stage with contact_block_summary."""
        adapter, driver = self.adapter()
        _, receipt = self.run_batch(adapter, batch(self.descent(adapter, driver, max_upward_dz_m=0.04)))
        self.assertEqual((receipt["status"], receipt["reason"]), ("failed", "tracking_error"), receipt)
        down = receipt["stage_results"]["down"]
        self.assertIn(down["tracking_stop"]["kind"], ("stall", "joint_tracking_limit"))
        self.assertNotIn("contact_stopped_early", down)
        summary = receipt["contact_block_summary"]
        self.assertTrue(summary["blocked_above_target"])
        self.assertLess(summary["completed_steps"], summary["planned_steps"])
        self.assertTrue(receipt["stop_confirmation"]["confirmed"])

    def test_robot_action_contact_stop_descent_then_close(self):
        """Public fields through the shared tool path: stall_guard_preset contact_stop + max_upward_dz_m on a descent,
        gripper_goal_normalized after arrival (a generated jaw stage), all in one batch."""
        harness, adapter, driver = self.harness(gripper_normalized=1.0)
        driver.table_z_m = self.pose(adapter)["position_m"][2] - 0.06 + 0.03
        stage = self.public_move(adapter, "down", dz=-0.06, stall_guard_preset="contact_stop", max_upward_dz_m=0.04,
                                 position_tolerance_m=0.01, orientation_tolerance_rad=0.1, gripper_goal_normalized=0.0)
        preview, receipt = self.robot_action(harness, [stage])
        tolerances = preview["stage_plans"]["down/motion"]["tolerances"]
        self.assertEqual((tolerances["stall_guard_preset"], tolerances["max_upward_dz_m"]), ("contact_stop", 0.04))
        self.assertEqual(tolerances["max_tracking_divergence_m"], harness.embodiment.stall_guard_presets_m["contact_stop"])
        self.assertEqual(receipt["status"], "completed", receipt)
        self.assertTrue(receipt["stage_results"]["down/motion"]["contact_stopped_early"])
        self.assertEqual(receipt["stage_results"]["down"]["status"], "completed")

    def test_motion_text_states_the_rule(self):
        adapter, _ = self.adapter()
        text = contract.actor_motion_instructions(adapter.deployment)
        self.assertIn("With contact_stop any of these stops holds the measured pose", text)
        self.assertIn("A contact_stop hold pins the joint goals to the measured pose (no sag offset", text)
        self.assertIn("a later pause at the held pose (observe stage, checkpoint) keeps that goal", text)
        self.assertIn("the rest of the ramp runs at the held pose (finished_at_held_pose)", text)
        self.assertIn("blocked_above_target (vertical_shortfall_m > position_tolerance_m)", text)
        self.assertIn("approach_shortfall_m (target minus measured along the target's TCP +Z approach axis", text)
        self.assertIn(f"gripper.max_speed_rad_s ({adapter.deployment['gripper']['max_speed_rad_s']:g} rad/s)", text)
        self.assertNotIn("does not hold at a contact and continue", text)
        # The window changes only the verdict: the sag correction still runs before it. Its bounds (deployment numbers)
        # are explained once, in the robot_action field help; the window sentence points there.
        self.assertIn("before the verdict the arrival correction (robot_action arrival_correction_rounds) shifts the "
                      "joint goals further toward the target", text)
        self.assertIn("only a contact_stop hold (a stop signal during the move) or arrival_correction_rounds=0 on the "
                      "stage skips the correction", text)
        self.assertNotIn("(arrival_correction: up to", text)
        # A deployment that corrects nothing says so, and offers no per-stage bound.
        uncorrected = dict(adapter.deployment, motion=dict(adapter.deployment["motion"], arrival_correction_rounds=0))
        text = contract.actor_motion_instructions(uncorrected)
        self.assertIn("this deployment applies no arrival correction", text)
        self.assertNotIn("arrival_correction_rounds=0 on the stage", text)
        self.assertNotIn("arrival_correction", contract.embodiment(uncorrected).supports)


class StallGuardNameTests(Case):
    """The stall-guard names the SO-101 offers are the schema's (off, contact_stop, strict): the motion text, the stage
    enum and the contract agree, and a batch-wide legacy alias (firm, normal, sensitive, compliant) is refused too."""

    def test_the_contract_list_is_the_schema_list(self):
        from ni_rua.runtime.actions import robot_action_schema
        self.assertEqual(contract.SCHEMA_STALL_GUARD_PRESETS, robot_action_schema.SCHEMA_STALL_GUARD_PRESETS)
        adapter, _ = self.adapter()
        offered = contract.offered_stall_guard_presets(adapter.embodiment)
        self.assertEqual(list(offered), ["off", "contact_stop", "strict"])
        schema = robot_action_schema.build(adapter.embodiment).stage_schemas["move"]["properties"]["stall_guard_preset"]
        self.assertEqual(schema["enum"], list(offered))
        text = contract.actor_motion_instructions(adapter.deployment)
        self.assertIn(f"stall_guard_preset (default off; contact_stop {offered['contact_stop']:g} m; "
                      f"strict {offered['strict']:g} m)", text)
        sentence = text.split("Stall guard (move stages):")[1].split("\n")[0]
        for legacy in ("firm", "sensitive", "compliant"):
            self.assertIsNone(re.search(rf"\b{legacy}\b", sentence), legacy)

    def test_legacy_names_are_refused_at_stage_and_batch_level(self):
        from ni_rua.runtime.actions.unified_actions import dispatch
        harness, adapter, _ = self.harness()
        for legacy in ("firm", "normal", "sensitive", "compliant"):
            with self.subTest(preset=legacy):
                self.assertIn(legacy, adapter.embodiment.stall_guard_presets_m)  # defined by the profile, not offered
                with self.assertRaisesRegex(ValueError, rf"unsupported_capability: stall_guard_preset {legacy} "
                                                        r"\(presets on this SO-101: off, contact_stop, strict\)"):
                    contract.validate_action(batch(self.move(adapter, "m", dz=0.01, stall_guard_preset=legacy)),
                                             adapter.deployment)
                # Batch-wide: the Harness resolves it into the move stage's path, where the contract refuses it.
                request = dict(observation_id=harness.latest["observation_id"], stall_guard_preset=legacy,
                               stages=[self.public_move(adapter, "m", dz=0.01)])
                with self.assertRaisesRegex(ValueError, rf"stall_guard_preset {legacy}"):
                    dispatch(harness, request, {"tool_call_id": "preview-" + legacy})
        preview, receipt = self.robot_action(harness, [self.public_move(adapter, "m", dz=0.01)], stall_guard_preset="strict")
        self.assertEqual(preview["stage_plans"]["m"]["tolerances"]["max_tracking_divergence_m"],
                         adapter.embodiment.stall_guard_presets_m["strict"])
        self.assertEqual(receipt["status"], "completed", receipt)


# ---- E3 -------------------------------------------------------------------------------------------------------------
class StartFrameTests(Case):
    def test_batch_and_wait_receipts_carry_the_start_observation_at_the_start_step(self):
        adapter, _ = self.adapter()
        _, receipt = self.run_batch(adapter, batch(self.move(adapter, "m", dz=0.01)))
        wait = adapter.execute(contract.compile_action("wait", dict(native_steps=2, observation_id="o"), IDENTITY,
                                                       adapter.capabilities()), lambda update: None, threading.Event())
        for name, value in (("batch", receipt), ("wait", wait)):
            with self.subTest(action=name):
                self.assertEqual(value["status"], "completed", value)
                start = value["before_observation"]
                self.assertEqual(start["sim_step"], value["execution_interval"]["start_sim_step"])
                self.assertEqual(start["robot_state"]["sim_step"], value["execution_interval"]["start_sim_step"])
                self.assertEqual(sorted(camera["camera_id"] for camera in start["cameras"]), ["fixed", "wrist"])
                self.assertNotEqual(start["raw_observation_id"], value["terminal_observation"]["raw_observation_id"])
                # Same joint read: equal within the overlay's camera-pose tolerance (atol 1e-8), so the post-action
                # overlay of the arm-mounted wrist camera is available (WP11 defect D2: before_state was rounded).
                np.testing.assert_allclose(start["robot_state"]["arms"]["right"]["q_rad"][:5],
                                           value["before_state"]["arms"]["right"]["q_rad"][:5], rtol=0, atol=1e-8)
                self.assertNotIn("execution_start_observation_error", value)

    def test_a_failed_capture_never_blocks_the_motion(self):
        adapter, _ = self.adapter()
        stage = self.move(adapter, "m", dz=0.01)
        preview = adapter.preview_batch(batch(stage))
        for camera in adapter.cameras.values():
            camera.fail = True
        receipt = adapter.execute(dict(batch(stage), batch_plan_id=preview["plan_id"]), lambda update: None, threading.Event())
        self.assertEqual(receipt["status"], "completed", receipt)
        self.assertNotIn("before_observation", receipt)
        self.assertIn("so101_camera_frame_timeout", receipt["execution_start_observation_error"])

    def test_a_rejected_plan_captures_nothing(self):
        adapter, driver = self.adapter()
        stage = self.move(adapter, "m", dz=0.01)
        preview = adapter.preview_batch(batch(stage))
        driver.q[0] += 0.2  # the arm was moved by hand since preview
        frames = {name: camera.count for name, camera in adapter.cameras.items()}
        receipt = adapter.execute(dict(batch(stage), batch_plan_id=preview["plan_id"]), lambda update: None, threading.Event())
        self.assertEqual((receipt["status"], receipt["reason"]), ("rejected", "batch_plan_stale_state_requires_new_preview"))
        self.assertNotIn("before_observation", receipt)
        self.assertEqual({name: camera.count for name, camera in adapter.cameras.items()}, frames)

    def test_harness_publishes_the_start_frame_as_a_historical_observation(self):
        harness, adapter, _ = self.harness()
        _, receipt = self.robot_action(harness, [self.public_move(adapter, "m", dz=0.01, position_tolerance_m=0.005,
                                                                  orientation_tolerance_rad=0.1)])
        self.assertEqual(receipt["status"], "completed", receipt)
        start = harness.store.read("observation", receipt["execution_start_observation_id"])
        self.assertEqual(start["sim_step"], receipt["execution_interval"]["start_sim_step"])
        self.assertNotEqual(harness.latest["observation_id"], receipt["execution_start_observation_id"])
        self.assertNotIn("before_observation", receipt)

    def test_unsettled_transient_arrival_blocks_dependent_gripper_stage_in_batch(self):
        adapter, _ = self.adapter()
        approach = self.move(adapter, "approach", dz=0.01)
        close = self.jaw(adapter, "close", 0.0)
        preview = adapter.preview_batch(batch(approach, close))
        self.assertEqual(preview["status"], "planned", preview)
        target_q = list(preview["stage_plans"]["approach"]["q_rad"])
        orig_settle = adapter._settle
        arm_settle_calls = 0

        def transient_unsettled_arm(run, timeout_s, *, joints="arm", interruptible=True):
            nonlocal arm_settle_calls
            if joints == "arm" and interruptible:
                arm_settle_calls += 1
                if arm_settle_calls == 1:
                    sample = dict(adapter.read(), q_rad=list(target_q))
                    return False, sample, "settle_timeout"
            return orig_settle(run, timeout_s, joints=joints, interruptible=interruptible)

        with patch.object(adapter, "_settle", side_effect=transient_unsettled_arm), \
                patch.object(adapter, "_gripper", wraps=adapter._gripper) as gripper_spy:
            receipt = adapter.execute(
                dict(batch(approach, close), batch_plan_id=preview["plan_id"]),
                lambda update: None,
                threading.Event(),
            )
        self.assertEqual((receipt["status"], receipt["reason"], receipt["failed_stage_id"]),
                         ("failed", "not_settled", "approach"), receipt)
        approach_res = receipt["stage_results"]["approach"]
        self.assertEqual(approach_res["status"], "failed")
        self.assertFalse(approach_res["settled"])
        self.assertTrue(approach_res["within_tolerance"])
        self.assertIn("endpoint_error", approach_res)
        self.assertFalse(approach_res["endpoint_error"]["settled"])
        self.assertTrue(approach_res["endpoint_error"]["within_tolerance"])
        self.assertEqual(receipt["stage_results"]["close"]["status"], "pending")
        self.assertEqual(receipt["pending_stage_ids"], ["approach", "close"])
        gripper_spy.assert_not_called()

    def test_ruckig_scurve_acceleration_and_jerk_bounds_match_preview_and_execute(self):
        from ni_rua.robot.backends.so101.ik_plan import joint_interpolation

        def build_adapter(accel, jerk):
            directory = self.enterContext(tempfile.TemporaryDirectory())
            deploy = deployment(
                **{
                    "gripper.empty_close_aperture_m": EMPTY_CLOSE_M,
                    "operator.default_speed_scale_cap": 1.0,
                    "motion.max_joint_speed_rad_s": 1.0,
                    "motion.command_rate_hz": 50.0,
                    "motion.max_command_delta_rad": 0.05,
                    "motion.max_acceleration_rad_s2": accel,
                    "motion.max_jerk_rad_s3": jerk,
                }
            )
            for name in ("fixed", "wrist"):
                deploy["cameras"][name].update(width=64, height=48)
            clock = FakeClock()
            driver = TableArmDriver(deploy, clock, tau_s=0.01)
            cameras = {name: FakeCameraSource(name, clock=clock, width=64, height=48) for name in ("fixed", "wrist")}
            ad = SO101Adapter(
                deploy,
                directory,
                "rgb_proprio_v1",
                driver=driver,
                cameras=cameras,
                clock=clock,
                image_settings=Config().images.to_dict(),
                start_telemetry=False,
            )
            self.addCleanup(ad.close)
            return ad

        # 1) Configured Ruckig S-curve: 1 rad move at 1 rad/s, 50 Hz, accel=12 rad/s^2, jerk=80 rad/s^3
        adapter_scurve = build_adapter(12.0, 80.0)
        start_q = adapter_scurve.read()["q_rad"]
        target_q_nom = [start_q[0] + 1.0, *start_q[1:]]
        target_pose = adapter_scurve._pose_dict(adapter_scurve.tcp(target_q_nom))
        leaf = contract.compile_action(
            "move",
            dict(
                arm="right",
                observation_id="o",
                position_m=target_pose["position_m"],
                quaternion_xyzw=target_pose["quaternion_xyzw"],
            ),
            IDENTITY,
            adapter_scurve.capabilities(),
        )
        leaf["path"].update(position_tolerance_m=0.01, orientation_tolerance_rad=0.1, arrival_correction_rounds=0)
        act = batch(dict(id="pan_1rad", kind="move", arm="right", action=leaf))

        preview = adapter_scurve.preview_batch(act)
        self.assertEqual(preview["status"], "planned", preview)
        self.assertEqual(preview["effective_limits"]["max_acceleration_rad_s2"], 12.0)
        self.assertEqual(preview["effective_limits"]["max_jerk_rad_s3"], 80.0)
        planned_steps = preview["stage_costs"]["pan_1rad"]["native_steps"]
        self.assertGreater(planned_steps, 50)
        self.assertEqual(preview["stage_plans"]["pan_1rad"]["path"]["samples"], planned_steps + 1)

        solved_q = preview["stage_plans"]["pan_1rad"]["q_rad"]
        expected_rows = joint_interpolation(
            start_q,
            solved_q,
            planned_steps,
            speed_rad_s=1.0,
            rate_hz=50.0,
            max_acceleration_rad_s2=12.0,
            max_jerk_rad_s3=80.0,
        )
        self.assertEqual(len(expected_rows), planned_steps)
        first_step_delta = float(np.max(np.abs(np.asarray(expected_rows[0]) - np.asarray(start_q))))
        self.assertLess(first_step_delta, 0.001)

        receipt = adapter_scurve.execute(
            dict(act, batch_plan_id=preview["plan_id"]), lambda update: None, threading.Event()
        )
        self.assertEqual(receipt["status"], "completed", receipt)
        motion_rows = [
            row[:5]
            for row, kind in zip(receipt["planned_trace"]["q_rad"], receipt["planned_trace"]["row_kinds"])
            if kind == "motion"
        ]
        self.assertEqual(len(motion_rows), planned_steps)
        exec_first_delta = float(np.max(np.abs(np.asarray(motion_rows[0]) - np.asarray(start_q))))
        self.assertLess(exec_first_delta, 0.001)
        np.testing.assert_allclose(motion_rows, expected_rows, rtol=0.0, atol=1e-9)

        # 2) Unconfigured (None) preserves 50-step constant-velocity linear interpolation (0.02 rad first step)
        adapter_linear = build_adapter(None, None)
        start_lin = adapter_linear.read()["q_rad"]
        pose_lin = adapter_linear._pose_dict(adapter_linear.tcp([start_lin[0] + 1.0, *start_lin[1:]]))
        leaf_lin = contract.compile_action(
            "move",
            dict(
                arm="right",
                observation_id="o",
                position_m=pose_lin["position_m"],
                quaternion_xyzw=pose_lin["quaternion_xyzw"],
            ),
            IDENTITY,
            adapter_linear.capabilities(),
        )
        leaf_lin["path"].update(position_tolerance_m=0.01, orientation_tolerance_rad=0.1, arrival_correction_rounds=0)
        act_lin = batch(dict(id="pan_lin", kind="move", arm="right", action=leaf_lin))
        prev_lin, rcpt_lin = self.run_batch(adapter_linear, act_lin)
        self.assertEqual(prev_lin["stage_costs"]["pan_lin"]["native_steps"], 50)
        lin_rows = [
            row[:5]
            for row, kind in zip(rcpt_lin["planned_trace"]["q_rad"], rcpt_lin["planned_trace"]["row_kinds"])
            if kind == "motion"
        ]
        self.assertEqual(len(lin_rows), 50)
        self.assertAlmostEqual(abs(lin_rows[0][0] - start_lin[0]), 0.02, places=3)


if __name__ == "__main__":
    unittest.main()
