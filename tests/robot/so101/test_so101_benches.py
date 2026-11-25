"""The SO-101 developer benches on CPU: metric computation on recorded data, and both scripts end to end on servo
models (``scripts/so101_motion_bench.py`` through a Harness over the in-process adapter, as the actor reaches it;
``scripts/so101_gravity_identify.py`` on the adapter). No serial port, camera or service is opened; this proves the
protocol code and its numbers, not the real arm.
"""

NI_RUA_TEST_TIER = "component"

# === tests/test_so101_motion_bench.py ===

import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ni_rua.core.config import Config
from ni_rua.robot.backends.so101.deployment import validate_deployment
from ni_rua.robot.backends.so101.driver import ARM_JOINTS, ENCODER_RESOLUTION_RAD, FakeCameraSource, FakeClock
from ni_rua.robot.backends.so101.worker import SO101Adapter
from ni_rua.robot.model import so101_kinematics
from scripts import so101_gravity_identify as identify
from scripts import so101_motion_bench as bench
from robot.so101.test_so101_backend import EXAMPLE, IDENTITY, InProcessBackend
from robot.so101.test_so101_motion_compensation import STIFFNESS, ServoModelDriver, document


def fake_adapter(deploy, directory, **driver_options):
    for name in ("fixed", "wrist"):
        deploy["cameras"][name].update(width=64, height=48)
    clock = FakeClock()
    driver = ServoModelDriver(deploy, clock, **driver_options)
    cameras = {name: FakeCameraSource(name, clock=clock, width=64, height=48) for name in ("fixed", "wrist")}
    adapter = SO101Adapter(deploy, directory, "rgb_proprio_v1", driver=driver, cameras=cameras, clock=clock,
                           image_settings=Config().images.to_dict(), start_telemetry=False)
    return adapter, driver


def bench_document(compensation=None):
    """The test deployment with its home at the servo model's start pose (TCP z 0.11 m, above the modelled table)."""
    raw = document(compensation)
    raw["home"]["q_rad"] = [0.0, -1.0, 1.0, 0.6, 0.0]
    return raw


STAGE_RESULT = {  # a move stage result as the worker reports it (carry on, one answered and one unanswered round)
    "kind": "move", "arm": "right", "status": "completed", "dx_m": 0.0004, "dy_m": -0.0002, "dz_m": -0.0011,
    "position_error_m": 0.0012, "orientation_error_rad": 0.01, "settled": True, "within_tolerance": True,
    "path_compensation": {"carried_offset_rad": [0.0, -0.03, -0.02, 0.0, 0.0], "carry": "last_arrival"},
    "arrival_correction": {"rounds": [{"round": 1, "responded": True}, {"round": 2, "responded": False}],
                           "max_rounds": 3, "initial_tcp_residual": {"position_error_m": 0.0071, "dz_m": -0.0066},
                           "termination_reason": "no_response", "offset_limit_reached": False, "max_offset_rad": 0.1,
                           "final_goal_offset_rad": [0.0, -0.036, -0.02, 0.0, 0.0]},
    "transit": {"samples": 42, "max_below_path_m": 0.0135, "max_below_at_fraction": 0.048, "max_deviation_m": 0.0141,
                "max_deviation_at_fraction": 0.048},
    "duration_s": 3.21}


class MotionBenchMetricTests(unittest.TestCase):
    def test_stage_metrics_from_a_recorded_stage_result(self):
        metrics = bench.stage_metrics(STAGE_RESULT)
        self.assertEqual({key: metrics[key] for key in (*bench.NUMERIC, *bench.CATEGORICAL)}, dict(
            max_below_path_m=0.0135, max_below_at_fraction=0.048, max_deviation_m=0.0141, max_deviation_at_fraction=0.048,
            residual_before_m=0.0071, residual_before_dz_m=-0.0066, residual_after_m=0.0012, residual_after_dz_m=-0.0011,
            correction_rounds=2, unanswered_rounds=1, duration_s=3.21, status="completed",
            termination_reason="no_response", tracking_stop=None, carry="last_arrival"))
        self.assertEqual(metrics["final_goal_offset_rad"], [0.0, -0.036, -0.02, 0.0, 0.0])
        stopped = bench.stage_metrics({"kind": "move", "status": "failed", "tracking_stop": {"kind": "stall"},
                                       "transit": {"max_below_path_m": 0.002}, "duration_s": 1.0})
        self.assertEqual((stopped["tracking_stop"], stopped["max_below_path_m"]), ("stall", 0.002))
        # No arrival was judged: nothing is invented.
        self.assertIsNone(stopped["correction_rounds"])
        self.assertIsNone(stopped["residual_after_m"])
        jaw = bench.stage_metrics({"kind": "gripper", "status": "completed"})
        self.assertTrue(all(jaw[name] is None for name in bench.NUMERIC))

    def test_contact_metrics_from_recorded_telemetry(self):
        def sample(t, lift, z):
            return {"t": t, "load": {"shoulder_lift": lift, "elbow_flex": -lift / 2, "wrist_flex": None}, "tcp_z_m": z}
        samples = [sample(0.5, 4.0, 0.030), sample(0.9, 6.0, 0.030),            # free-air hover before the press
                   sample(1.2, 8.0, 0.010), sample(1.6, -31.0, -0.0145), sample(1.9, 22.0, -0.0140),  # descent and press
                   sample(2.2, 12.0, -0.0138), sample(2.8, 14.0, -0.0136),      # dwell on the table
                   sample(3.5, 5.0, 0.020)]                                     # lift, outside both windows
        facts = bench.contact_metrics(samples, press_window=(1.0, 2.0), hold_window=(2.0, 3.0), contact_tcp_z_m=-0.013)
        self.assertEqual(facts["samples"], dict(baseline=2, press=3, hold=2))
        self.assertEqual(facts["baseline_load_pct"], dict(shoulder_lift=5.0, elbow_flex=-2.5, wrist_flex=None))
        self.assertEqual(facts["peak_load_pct"], dict(shoulder_lift=-31.0, elbow_flex=15.5, wrist_flex=None))
        self.assertEqual(facts["hold_load_pct"], dict(shoulder_lift=13.0, elbow_flex=-6.5, wrist_flex=None))
        self.assertEqual(facts["lowest_tcp_z_m"], -0.0145)
        self.assertEqual(facts["below_contact_m"], 0.0015)          # 1.5 mm below the given contact height
        self.assertEqual(facts["held_above_contact_m"], -0.0007)    # resting 0.7 mm below it during the dwell
        empty = bench.contact_metrics([], (1.0, 2.0), (2.0, 3.0), -0.013)
        self.assertIsNone(empty["below_contact_m"])
        self.assertIsNone(empty["peak_load_pct"]["shoulder_lift"])

    def test_summary_is_mean_and_sample_stdev_over_repetitions(self):
        def repetition(below, rounds, stop=None, peak=20.0):
            stage = dict(bench.stage_metrics(STAGE_RESULT), max_below_path_m=below, correction_rounds=rounds,
                         tracking_stop=stop)
            return {"tests": {"transit": [{"name": "sweep_z60_r180", "stages": {"sweep_out": stage}}],
                              "contact": [{"name": "contact", "stages": {},
                                           "contact": {"samples": {"press": 3}, "below_contact_m": below / 10,
                                                       "peak_load_pct": {"shoulder_lift": peak, "wrist_flex": None}}}]}}
        summary = bench.summarise([repetition(0.010, 1), repetition(0.014, 3, peak=30.0), repetition(0.012, 2, stop="stall")])
        stage = summary["transit/sweep_z60_r180/sweep_out"]
        self.assertEqual(stage["max_below_path_m"], dict(mean=0.012, stdev=0.002, min=0.01, max=0.014, n=3))
        self.assertEqual(stage["correction_rounds"], dict(mean=2.0, stdev=1.0, min=1.0, max=3.0, n=3))
        self.assertEqual(stage["counts"]["tracking_stop"], {"None": 2, "stall": 1})
        self.assertEqual(stage["counts"]["termination_reason"], {"no_response": 3})
        self.assertEqual(summary["contact"]["peak_load_pct.shoulder_lift"]["mean"], round((20 + 30 + 20) / 3, 6))
        self.assertEqual(summary["contact"]["below_contact_m"]["n"], 3)
        self.assertNotIn("peak_load_pct.wrist_flex", summary["contact"])
        self.assertNotIn("samples.press", summary["contact"])
        single = bench.summarise([repetition(0.010, 1)])
        self.assertIsNone(single["transit/sweep_z60_r180/sweep_out"]["max_below_path_m"]["stdev"])

    def test_protocol_geometry(self):
        model = so101_kinematics.load_model()
        x0, y0 = bench.pan_axis_xy(model)
        self.assertAlmostEqual(x0, 0.0388353, places=6)
        point = bench.polar_point(model, 0.24, 30.0, 0.06)
        self.assertAlmostEqual(math.hypot(point[0] - x0, point[1] - y0), 0.24, places=4)
        self.assertEqual(point[2], 0.06)
        arguments = bench.build_parser().parse_args(["run", "--deployment", "d", "--condition", "c", "--out", "o"])
        actions = bench.transit_actions(arguments, model)
        self.assertEqual([name for name, _ in actions],
                         ["sweep_z60_r180", "sweep_z60_r240", "sweep_z60_r300", "radial_z60", "gravity_sign",
                          "sweep_z120_r180", "sweep_z120_r240", "sweep_z120_r300", "radial_z120"])
        self.assertEqual([stage["id"] for stage in actions[0][1]], ["approach", "sweep_out", "sweep_back"])
        self.assertEqual(actions[0][1][0]["position_m"], actions[0][1][2]["position_m"])


class MotionBenchRunTests(unittest.TestCase):
    """The whole protocol through Harness.execute on a sagging servo model above a rigid table."""
    CONTACT_Z = 0.045

    def session_factory(self, compensation):
        def factory(arguments, out):
            from ni_rua.core.storage.store import Store
            from ni_rua.runtime.harness import Harness
            deploy = validate_deployment(bench_document(compensation), base=EXAMPLE.parent)
            adapter, driver = fake_adapter(deploy, out / "worker", stiffness=STIFFNESS, table_z_m=self.CONTACT_Z,
                                           gripper_normalized=1.0)
            store = Store(out / "store", IDENTITY)
            config = Config.from_dict({"backend": {"environment": "so101_real", "robot_model_id": "so101",
                                                   "deployment": str(EXAMPLE)},
                                       "features": {"observation_profile": "rgb_proprio_v1", "external_perception": False, "record_video": False}})
            harness = Harness(store, InProcessBackend(adapter), observation_profile="rgb_proprio_v1", config=config)
            harness.start()
            harness.task_planning.required = False
            self.sessions.append((adapter, driver))

            def close():
                adapter.close()
                store.close()
                return []
            return bench.Session(harness, lambda: adapter.telemetry_tick(frames=False), close)
        return factory

    def run_bench(self, directory, condition, compensation, *extra):
        self.sessions = []
        deployment_path = Path(directory) / f"{condition}.json"
        deployment_path.write_text(json.dumps(bench_document(compensation)))
        arguments = bench.build_parser().parse_args([
            "run", "--deployment", str(deployment_path), "--condition", condition, "--out", str(Path(directory) / condition),
            "--repetitions", "2", "--transit-heights-m", "0.08", "--transit-reaches-m", "0.18", "0.26",
            "--contact-xy-m", "0.20", "0.0", "--contact-tcp-z-m", str(self.CONTACT_Z), "--contact-dwell-s", "0.2", *extra])
        with (patch("ni_rua.runtime.actions.inspection._trajectory_image", return_value=({"status": "unavailable"}, [])),
              contextlib.redirect_stdout(io.StringIO())):
            report = bench.run(arguments, session_factory=self.session_factory(compensation))
        written = json.loads((Path(directory) / condition / f"bench-{condition}.json").read_text())
        self.assertEqual(written["summary"], json.loads(json.dumps(report["summary"])))
        return report

    def test_protocol_runs_and_the_conditions_differ_only_by_the_deployment(self):
        feedforward = dict(gravity_feedforward=dict(joint_stiffness_nm_per_rad=STIFFNESS, max_offset_rad=0.2,
                                                    provenance="servo model"), carry_arrival_offset=True)
        with tempfile.TemporaryDirectory() as directory:
            carry = ["--cube-xy-m", "0.22", "0.05", "--cube-centre-z-m", "0.065", "--carry-over-xy-m", "0.22", "-0.06",
                     "--carry-height-m", "0.08"]
            baseline = self.run_bench(directory, "baseline", None, *carry)
            compensated = self.run_bench(directory, "m1-m3", feedforward, *carry)
            for report in (baseline, compensated):
                self.assertIsNone(report["aborted"])
                self.assertEqual((report["schema"], report["tests"], report["skipped_tests"]),
                                 (bench.SCHEMA, ["transit", "carry", "contact"], {}))
                self.assertEqual(len(report["repetitions"]), 2)
                self.assertEqual(report["settings"]["servo"], dict(p_coefficient=16, i_coefficient=0, d_coefficient=32))
                self.assertIn("revision", report["source"])
                for repetition in report["repetitions"]:
                    names = [action["name"] for action in repetition["tests"]["transit"]]
                    self.assertEqual(names, ["sweep_z80_r180", "sweep_z80_r260", "radial_z80", "gravity_sign"])
                    # The pair crosses the sign change of the shoulder's gravity torque, as the protocol promises.
                    crossing = repetition["tests"]["transit"][3]["gravity_torque"]
                    self.assertEqual(crossing["cross_out"]["sign_change_joints"], ["shoulder_lift"])
                    self.assertLess(crossing["cross_out"]["start_nm"]["shoulder_lift"], 0.0)
                    self.assertGreater(crossing["cross_out"]["end_nm"]["shoulder_lift"], 0.0)
                    self.assertEqual(repetition["tests"]["transit"][0]["gravity_torque"]["sweep_out"]["sign_change_joints"], [])
                    for action in (*repetition["tests"]["transit"], *repetition["tests"]["carry"],
                                   *repetition["tests"]["contact"]):
                        self.assertEqual(action["status"], "completed", action)
                    carried = repetition["tests"]["carry"][0]["stages"]
                    self.assertEqual(list(carried), ["open", "above_cube", "down_to_cube", "close", "lift", "carry_over",
                                                     "carry_back", "set_down", "release", "retreat"])
                    self.assertIsNotNone(carried["carry_over"]["max_below_path_m"])
                    press = repetition["tests"]["contact"][0]
                    self.assertTrue(press["stages"]["press"]["contact_stopped_early"])
                    self.assertIn(press["stages"]["press"]["tracking_stop"], ("tcp_divergence", "stall"))
                    self.assertEqual(set(press["contact"]), {"samples", "baseline_load_pct", "peak_load_pct",
                                                             "hold_load_pct", "lowest_tcp_z_m", "below_contact_m",
                                                             "held_above_contact_m"})
            self.assertIsNone(baseline["settings"]["motion_compensation"])
            self.assertTrue(compensated["settings"]["motion_compensation"]["carry_arrival_offset"])
            self.assertEqual(compensated["settings"]["motion_compensation"]["gravity_feedforward"]["joint_stiffness_nm_per_rad"],
                             STIFFNESS)
            key = "transit/radial_z80/reach_out"
            self.assertEqual(baseline["summary"][key]["max_below_path_m"]["n"], 2)
            self.assertEqual(baseline["summary"][key]["counts"]["carry"], {"None": 2})
            self.assertEqual(compensated["summary"][key]["counts"]["carry"], {"last_arrival": 2})
            # The modelled sag is what the bench is built to show, and what the compensation removes.
            self.assertGreater(baseline["summary"][key]["max_below_path_m"]["mean"], 0.004)
            self.assertLess(compensated["summary"][key]["max_below_path_m"]["mean"],
                            0.3 * baseline["summary"][key]["max_below_path_m"]["mean"])
            self.assertGreater(baseline["summary"]["carry/carry/carry_over"]["residual_before_m"]["mean"],
                               compensated["summary"]["carry/carry/carry_over"]["residual_before_m"]["mean"])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                rows = bench.main(["compare", str(Path(directory) / "baseline" / "bench-baseline.json"),
                                   str(Path(directory) / "m1-m3" / "bench-m1-m3.json")])
            self.assertIn((key, "max_below_path_m"), [(row[0], row[1]) for row in rows])
            self.assertIn("baseline", output.getvalue().splitlines()[0])
            self.assertIn("m1-m3", output.getvalue().splitlines()[0])
            # A used directory is refused: every run has its own.
            with self.assertRaises(SystemExit):
                self.run_bench(directory, "baseline", None)

    def test_tests_without_their_positions_are_skipped_and_said_so(self):
        with tempfile.TemporaryDirectory() as directory:
            self.sessions = []
            deployment_path = Path(directory) / "x.json"
            deployment_path.write_text(json.dumps(bench_document()))
            arguments = bench.build_parser().parse_args([
                "run", "--deployment", str(deployment_path), "--condition", "x", "--out", str(Path(directory) / "x"),
                "--repetitions", "1", "--transit-heights-m", "0.08", "--transit-reaches-m", "0.2"])
            with (patch("ni_rua.runtime.actions.inspection._trajectory_image", return_value=({"status": "unavailable"}, [])),
                  contextlib.redirect_stdout(io.StringIO())):
                report = bench.run(arguments, session_factory=self.session_factory(None))
        self.assertEqual(report["tests"], ["transit"])
        self.assertEqual(set(report["skipped_tests"]), {"carry", "contact"})
        self.assertEqual([action["name"] for action in report["repetitions"][0]["tests"]["transit"]],
                         ["sweep_z80_r200", "gravity_sign"])
        self.assertEqual(report["protocol"]["speed_scale"], 0.5)


class IdentifyTests(unittest.TestCase):
    def run_identify(self, directory, *extra, sabotage_after_holds=None, hold_at_breakaway=False, **driver_options):
        made = {}

        def factory(deploy, run_dir):
            adapter = build(deploy, run_dir)
            if sabotage_after_holds is not None:
                original, calls = adapter.move_joints, {"count": 0}

                def move_joints(q_target, *args, **options):
                    calls["count"] += 1
                    if calls["count"] == sabotage_after_holds + 1:
                        return {"status": "failed", "reason": "tracking_error", "diagnostic": {"kind": "stall"}}
                    return original(q_target, *args, **options)
                adapter.move_joints = move_joints
            if hold_at_breakaway:
                original_send = adapter._send

                def send(q_rad, jaw, generation=None, **options):
                    if generation is not None and "run" not in options and not adapter.relaxed:
                        with adapter.state_lock:  # what an operator hold does to the stop generation
                            adapter.stop_generation += 1
                            adapter.stop_reason = "operator_hold"
                    return original_send(q_rad, jaw, generation, **options)
                adapter._send = send
            return adapter

        def build(deploy, run_dir):
            self.assertEqual(deploy["motion_compensation"], dict(carry_arrival_offset=False, no_response_stop=False,
                                                                 stop_correction_within_tolerance=False,
                                                                 arrival_breakaway_rad=None, gravity_feedforward=None))
            made["adapter"], made["driver"] = fake_adapter(deploy, run_dir, **driver_options)
            return made["adapter"]
        out = Path(directory) / "proposal.json"
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            report = identify.main(["--deployment", str(EXAMPLE), "--out", str(out), "--dwell-s", "0.2", *extra],
                                   adapter_factory=factory)
        return report, out, printed.getvalue(), made

    def test_stiffness_is_recovered_from_uncorrected_holds_and_only_a_proposal_is_written(self):
        before = EXAMPLE.read_text()
        with tempfile.TemporaryDirectory() as directory:
            report, out, printed, made = self.run_identify(directory, stiffness=STIFFNESS)
            written = json.loads(out.read_text())
        self.assertEqual(EXAMPLE.read_text(), before)  # the deployment file is never edited
        self.assertTrue(report["ok"], report.get("error"))
        self.assertEqual(written["schema"], identify.SCHEMA)
        usable = [item for item in report["poses"] if item["reason"] is None]
        self.assertGreaterEqual(len(usable), 8)
        self.assertEqual(len(report["holds"]), 2 * len(usable))
        self.assertEqual({entry["approach"] for entry in report["holds"]}, {"forward", "reverse"})
        feedforward = report["proposal"]["motion_compensation"]["gravity_feedforward"]
        for name, expected in STIFFNESS.items():
            self.assertAlmostEqual(feedforward["joint_stiffness_nm_per_rad"][name], expected, delta=0.03 * expected)
            self.assertGreater(report["fit"][name]["r_squared"], 0.99)
        self.assertEqual(set(feedforward["joint_stiffness_nm_per_rad"]), set(STIFFNESS))
        for name in ("shoulder_pan", "wrist_roll"):
            self.assertEqual(report["fit"][name]["reason"], "torque_range_below_minimum")
        self.assertIn("servo p/i/d 16/0/32", feedforward["provenance"])
        self.assertNotIn("arrival_breakaway_rad", report["proposal"]["motion_compensation"])
        # The proposal is a valid deployment section as written.
        raw = document(report["proposal"]["motion_compensation"])
        self.assertEqual(validate_deployment(raw, base=EXAMPLE.parent)["motion_compensation"]["gravity_feedforward"], feedforward)
        self.assertEqual(report["park_home"]["status"], "completed")
        # Holds were uncorrected: no arrival-correction command was ever sent.
        self.assertGreater(report["largest_feedforward_offset_rad"]["elbow_flex"], 0.03)

    def test_breakaway_band_is_found_in_encoder_ticks(self):
        with tempfile.TemporaryDirectory() as directory:
            report, _, _, _ = self.run_identify(directory, "--breakaway", "--breakaway-joints", "elbow_flex",
                                                "--breakaway-period-s", "0.5", static_band_rad=0.02)
        self.assertTrue(report["ok"], report.get("error"))
        band = report["breakaway"]["elbow_flex"]
        for key in ("plus_rad", "minus_rad"):
            self.assertAlmostEqual(band[key], 0.02, delta=2 * ENCODER_RESOLUTION_RAD)
            self.assertAlmostEqual(band[key] / ENCODER_RESOLUTION_RAD, round(band[key] / ENCODER_RESOLUTION_RAD), places=2)
        self.assertEqual(report["proposal"]["motion_compensation"]["arrival_breakaway_rad"],
                         {"elbow_flex": max(band.values())})

    def test_dry_run_integral_gain_and_existing_output_never_open_the_bus(self):
        def refuse(deploy, run_dir):
            raise AssertionError("the adapter must not be created")
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertIsNone(identify.main(["--deployment", str(EXAMPLE), "--dry-run"], adapter_factory=refuse))
        self.assertIn("usable poses", printed.getvalue())
        self.assertIn("lowest_point_below_min_z", printed.getvalue())
        with tempfile.TemporaryDirectory() as directory:
            raw = document()
            raw["servo"]["i_coefficient"] = 4
            path = Path(directory) / "integral.json"
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(SystemExit, "i_coefficient must be 0"):
                identify.main(["--deployment", str(path), "--out", str(Path(directory) / "p.json")], adapter_factory=refuse)
            existing = Path(directory) / "existing.json"
            existing.write_text("{}")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(SystemExit, "exists"):
                identify.main(["--deployment", str(EXAMPLE), "--out", str(existing)], adapter_factory=refuse)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(SystemExit, "exists"):
                identify.main(["--deployment", str(EXAMPLE), "--out", str(EXAMPLE)], adapter_factory=refuse)

    def test_pose_screen(self):
        model, deploy = so101_kinematics.load_model(), validate_deployment(document(), base=EXAMPLE.parent)
        from ni_rua.robot.embodiments.so101.gravity import load_inertials
        inertials = load_inertials()
        self.assertIsNone(identify.pose_facts(model, inertials, deploy, [0.0, -0.9, 0.7, 1.0, 0.0], 0.06)["reason"])
        self.assertEqual(identify.pose_facts(model, inertials, deploy, [0.0, -0.1, 1.2, 0.4, 0.0], 0.06)["reason"],
                         "tcp_outside_deployment_workspace")
        self.assertEqual(identify.pose_facts(model, inertials, deploy, [0.0, -0.9, 0.7, 1.0, 0.0], 0.2)["reason"],
                         "lowest_point_below_min_z")
        self.assertEqual(identify.pose_facts(model, inertials, deploy, [0.0, -1.74, 0.7, 1.0, 0.0], 0.06)["reason"],
                         "outside_joint_limits_with_margin")
        self.assertEqual(len(identify.candidate_poses(None)), 24)

    def test_height_screen_covers_fingertips_travel_lines_and_the_way_home(self):
        model, deploy = so101_kinematics.load_model(), validate_deployment(document(), base=EXAMPLE.parent)
        jaws = (deploy["gripper"]["jaw_closed_rad"], deploy["gripper"]["jaw_open_rad"])
        q = [0.0, -0.9, 0.7, 1.0, 0.0]
        tcp_z = float(so101_kinematics.fk(model, q)[2, 3])
        self.assertLess(identify.lowest_z(model, q, jaws), tcp_z)  # a fingertip reaches below the TCP
        # Two poses that are each fine, with a joint-space line between them that dips lower than both.
        high, far = [0.0, 0.3, 0.2, 1.4, 0.0], [0.0, 0.3, 1.2, -1.0, 0.0]
        ends = min(identify.lowest_z(model, high, jaws), identify.lowest_z(model, far, jaws))
        line = identify.line_facts(model, high, far, jaws, 0.02, ends - 0.001, "high -> far")
        self.assertLess(line["lowest_z_m"], ends - 0.04)  # 4 cm lower than either end pose
        self.assertFalse(line["ok"])
        self.assertTrue(0.0 < line["lowest_at_fraction"] < 1.0)
        self.assertGreaterEqual(line["samples"], int(2.4 / 0.02))
        # Dry run prints every pose and line; home below the floor is a given endpoint, not a failure.
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            identify.main(["--deployment", str(EXAMPLE), "--dry-run", "--breakaway"])
        text = printed.getvalue()
        for needle in ("height screen from home.q_rad (assumed start)", '"line": "start -> pose 0"', '-> home"',
                       '"line": "first pose -> breakaway pose"', '"line": "breakaway pose -> home"', "problems: none"):
            self.assertIn(needle, text)
        # A pose whose inbound line dips under the floor is left out by its line, although the pose itself is fine.
        import argparse
        arguments = argparse.Namespace(screen_step_rad=0.02, min_z_m=-0.01, breakaway=False, breakaway_pose=None)
        deploy["home"]["q_rad"] = list(high)
        facts = [dict(q_rad=list(high), reason=None), dict(q_rad=list(far), reason=None)]
        usable, lines, problems = identify.screen_lines(model, deploy, facts, high, arguments)
        self.assertEqual([item["q_rad"] for item in usable], [list(high)])
        self.assertEqual(facts[1]["reason"], "line_below_min_z")
        self.assertEqual([item["line"] for item in lines if not item["ok"]], ["pose 0 -> pose 1", "pose 1 -> home"])

    def test_poor_fits_are_marked_rejected_and_left_out_of_the_proposal(self):
        with tempfile.TemporaryDirectory() as directory:
            # Gear friction on the shoulder (0.01 rad static band) blurs its fit; the other joints stay clean.
            report, _, _, _ = self.run_identify(directory, "--min-r-squared", "0.995", stiffness=STIFFNESS,
                                                static_band_rad=[0.0, 0.01, 0.0, 0.0, 0.0])
        self.assertTrue(report["ok"], report.get("error"))
        shoulder = report["fit"]["shoulder_lift"]
        self.assertFalse(shoulder["accepted"])
        self.assertEqual(shoulder["rejected"], ["r_squared below 0.995"])
        self.assertIsNotNone(shoulder["stiffness_nm_per_rad"])  # the fit is still shown
        self.assertTrue(report["fit"]["elbow_flex"]["accepted"])
        self.assertEqual(set(report["proposal"]["motion_compensation"]["gravity_feedforward"]["joint_stiffness_nm_per_rad"]),
                         {"elbow_flex", "wrist_flex"})
        # Friction as large as the gravity signal: the two approach directions disagree and the joint is rejected.
        with tempfile.TemporaryDirectory() as directory:
            report, _, _, _ = self.run_identify(directory, "--max-approach-spread-ratio", "0.0", stiffness=STIFFNESS,
                                                static_band_rad=0.004)
        rejected = [name for name in STIFFNESS if not report["fit"][name]["accepted"]]
        self.assertTrue(rejected)
        self.assertIn("approach_spread_ratio above 0.0", report["fit"][rejected[0]]["rejected"])

    def test_abort_returns_home_and_a_dropping_arm_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            # The third hold is knocked out of tolerance: the script aborts, confirms the stop and goes home.
            report, _, _, made = self.run_identify(directory, "--arrival-tolerance-rad", "0.2", stiffness=STIFFNESS,
                                                   sabotage_after_holds=2)
            self.assertFalse(report["ok"])
            self.assertIn("Abort", report["error"])
            self.assertEqual(len(report["holds"]), 2)
            self.assertTrue(report["abort_stop"]["confirmed"] and report["return_home"])
            self.assertEqual(report["park_home"]["status"], "completed")
            home = validate_deployment(document(), base=EXAMPLE.parent)["home"]["q_rad"]
            np.testing.assert_allclose(made["driver"].q, home, atol=0.11)  # at home, within the uncorrected sag
            raw = document()
            raw["disable_torque_on_disconnect"] = True
            path = Path(directory) / "drop.json"
            path.write_text(json.dumps(raw))

            def refuse(deploy, run_dir):
                raise AssertionError("the adapter must not be created")
            with self.assertRaisesRegex(SystemExit, "disable_torque_on_disconnect is true"):
                identify.main(["--deployment", str(path), "--out", str(Path(directory) / "p.json")], adapter_factory=refuse)

    def test_breakaway_steps_obey_the_gate_and_a_hold(self):
        with tempfile.TemporaryDirectory() as directory:
            report, _, _, made = self.run_identify(directory, "--breakaway", "--breakaway-joints", "elbow_flex",
                                                   "--breakaway-period-s", "0.5", static_band_rad=0.02, hold_at_breakaway=True)
        self.assertFalse(report["ok"])
        self.assertIn("breakaway elbow_flex: stopped", report["error"])
        self.assertEqual(ARM_JOINTS[1], "shoulder_lift")

# === tests/test_so101_grip_bench.py ===

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ni_rua.core.config import Config
from ni_rua.robot.backends.so101.deployment import validate_deployment
from ni_rua.robot.model.so101_gripper_geometry import jaw_rad_for_opening
from scripts import so101_grip_bench as grip_bench
from scripts import so101_motion_bench as motion_bench
from robot.so101.test_so101_backend import EXAMPLE, IDENTITY, InProcessBackend

CUBE_M = 0.029
JAW_RANGE = 1.74533 + 0.174533
CUBE_JAW = (jaw_rad_for_opening(CUBE_M) + 0.174533) / JAW_RANGE


def sample(t, load=None, current=None, temperature=None, jaw=None, goal=None, aperture=None):
    return {"t": t, "load_pct": load, "current_ma": current, "temperature_c": temperature, "jaw_rad": jaw,
            "jaw_goal_rad": goal, "aperture_m": aperture}


class GripBenchMetricTests(unittest.TestCase):
    def test_window_metrics_from_recorded_telemetry(self):
        samples = [sample(0.5, 2.0, 10.0, 31.0, 0.5, 0.5, 0.06),                       # before the hold
                   sample(1.0, -20.0, -130.0, 31.0, 0.161, 0.141, 0.0290),
                   sample(2.0, -18.0, -117.0, 32.0, 0.160, 0.141, 0.0289),
                   sample(3.0, -16.0, None, 34.0, 0.159, 0.141, 0.0288),
                   sample(4.0, 1.0, 5.0, 34.0, 0.9, 0.9, 0.09)]                         # after it
        facts = grip_bench.window_metrics(samples, (1.0, 3.0))
        self.assertEqual(facts["samples"], 3)
        self.assertEqual(facts["duration_s"], 2.0)
        self.assertEqual(facts["load_pct"], dict(mean=-18.0, peak=-20.0, last=-16.0))
        self.assertEqual(facts["current_ma"], dict(mean=-123.5, peak=-130.0, last=-117.0))
        self.assertEqual(facts["temperature_c"], dict(start=31.0, end=34.0, max=34.0, rise=3.0))
        self.assertEqual(facts["aperture_m"], dict(start=0.029, end=0.0288, min=0.0288, change=-0.0002))
        self.assertEqual(facts["squeeze_remaining_rad"], dict(mean=0.019, last=0.018))
        empty = grip_bench.window_metrics(samples, (10.0, 11.0))
        self.assertEqual(empty["samples"], 0)
        for key in ("load_pct", "current_ma", "temperature_c", "aperture_m", "squeeze_remaining_rad"):
            self.assertIsNone(empty[key])  # nothing is invented without samples

    def test_sampler_entry_reads_the_gripper_rows_of_the_telemetry_document(self):
        sampler = grip_bench.GripSampler(lambda: None)
        state = {"motor": {"gripper": {"temperature_c": 33.0, "load": -19.5, "voltage_v": 12.0, "current_ma": -110.5}},
                 "q_rad": {"gripper": jaw_rad_for_opening(CUBE_M)}, "q_goal_rad": {"gripper": 0.14}}
        entry = sampler.entry(state)
        self.assertEqual((entry["load_pct"], entry["current_ma"], entry["temperature_c"], entry["jaw_goal_rad"]),
                         (-19.5, -110.5, 33.0, 0.14))
        self.assertAlmostEqual(entry["aperture_m"], CUBE_M, delta=1e-5)
        old = sampler.entry({"motor": {"gripper": {"temperature_c": 30.0, "load": 0.0}}, "q_rad": None, "q_goal_rad": None})
        self.assertEqual((old["current_ma"], old["jaw_rad"], old["aperture_m"]), (None, None, None))

    def test_stage_facts_keep_the_jaw_evidence(self):
        result = {"kind": "gripper", "arm": "right", "status": "completed",
                  "gripper_execution": {"goal_normalized": 0.0, "phase": "after_motion", "planned_steps": 12,
                                        "measured_opening_m": 0.029, "measured_jaw_rad": 0.161, "jaw_settled": True,
                                        "hold_squeeze": {"squeeze_rad": 0.02, "applied": True, "contact_opening_m": 0.029}},
                  "grip_facts": {"measured_aperture_m": 0.029, "aperture_delta_m": -0.04}}
        facts = grip_bench.grip_stage_facts(result)
        self.assertEqual(facts["gripper_execution"]["hold_squeeze"]["squeeze_rad"], 0.02)
        self.assertNotIn("planned_steps", facts["gripper_execution"])
        self.assertEqual(facts["grip_facts"]["measured_aperture_m"], 0.029)
        self.assertNotIn("grip_hold", facts)
        paused = grip_bench.grip_stage_facts({"kind": "move", "status": "paused_at_checkpoint",
                                         "checkpoint_reason": "aperture_below_min_m",
                                         "grip_hold": {"squeeze_remaining_rad": 0.001}, "position_error_m": 0.002})
        self.assertEqual((paused["checkpoint_reason"], paused["grip_hold"], paused["position_error_m"]),
                         ("aperture_below_min_m", {"squeeze_remaining_rad": 0.001}, 0.002))

    def test_stage_facts_recorded_at_a_pause_survive_the_resumed_receipt(self):
        # A resumed receipt lists the stages completed before the pause with kind, arm and status only.
        paused = {"kind": "gripper", "arm": "right", "status": "paused_at_checkpoint",
                  "checkpoint_reason": "aperture_below_min_m", "grip_facts": {"measured_aperture_m": 0.004},
                  "gripper_execution": {"measured_opening_m": 0.004, "hold_squeeze": {"applied": True, "squeeze_rad": 0.02}}}
        receipts = iter([
            {"status": "stopped", "paused_at_checkpoint": True, "wall_clock_duration_s": 1.0,
             "stage_results": {"close": paused, "lift": {"kind": "move", "arm": "right", "status": "pending"}}},
            {"status": "completed", "wall_clock_duration_s": 2.0,
             "stage_results": {"close": {"kind": "gripper", "arm": "right", "status": "completed"},
                               "lift": {"kind": "move", "arm": "right", "status": "completed", "position_error_m": 0.001}}}])

        class Response:
            def __init__(self, data):
                self.data = data

        def dispatch(harness, request, correlation):
            if request.get("mode") == "execute":
                return Response({"receipt": next(receipts)})
            return Response({"status": "planned", "plan_id": "p1", "stage_plans": {}})
        session = motion_bench.Session(type("H", (), {"latest": {"observation_id": "obs1"}})(), lambda: None, lambda: [])
        stages = [dict(id="close", kind="gripper"), dict(id="lift", kind="move")]
        with patch("ni_rua.runtime.actions.unified_actions.dispatch", dispatch):
            record = motion_bench.run_action(session, "grasp", stages, metrics=grip_bench.grip_stage_facts)
        self.assertEqual((record["status"], record["worker_duration_s"]), ("completed", 3.0))
        self.assertEqual(record["stages"]["close"]["checkpoint_reason"], "aperture_below_min_m")
        self.assertTrue(record["stages"]["close"]["gripper_execution"]["hold_squeeze"]["applied"])
        self.assertEqual(record["stages"]["lift"], dict(status="completed", kind="move", position_error_m=0.001))

    def test_slip_facts(self):
        def move(delta, remaining=None, reason=None):
            return {"status": "completed", "grip_facts": {"aperture_delta_m": delta},
                    **({"grip_hold": {"squeeze_remaining_rad": remaining}} if remaining is not None else {}),
                    **({"checkpoint_reason": reason} if reason else {})}
        grasp = {"stages": {"close": {"grip_facts": {"measured_aperture_m": 0.0290}}}}
        release = {"stages": {"release": {"grip_facts": {"measured_aperture_m": 0.1361, "aperture_delta_m": 0.1072}}}}
        kept = {"stages": {"lift": move(-0.0001, 0.0199), "carry_over": move(0.0, 0.0199), "carry_back": move(0.0, 0.0198),
                           "set_down": move(0.0, 0.0198)}}
        facts = grip_bench.slip_facts(grasp, kept, release, 0.001)
        self.assertEqual((facts["aperture_at_close_m"], facts["aperture_before_release_m"]), (0.029, 0.0289))
        self.assertEqual(facts["jaw_closed_further_m"], 0.0001)
        self.assertIs(facts["jaw_closed_beyond_threshold"], False)
        self.assertEqual(facts["transport_aperture_change_m"], -0.0001)
        self.assertEqual(facts["squeeze_remaining_rad"]["set_down"], 0.0198)
        self.assertEqual(facts["recheck"], [])
        # The jaw closed by the squeeze's travel during the lift: reported, with the stage that paused.
        lost = {"stages": {"lift": move(-0.0016, 0.0002, "aperture_below_min_m"), "carry_over": move(0.0, 0.0001),
                           "carry_back": move(0.0, 0.0001), "set_down": move(0.0, 0.0001)}}
        released = {"stages": {"release": {"grip_facts": {"measured_aperture_m": 0.1361, "aperture_delta_m": 0.1087}}}}
        facts = grip_bench.slip_facts(grasp, lost, released, 0.001)
        self.assertEqual(facts["jaw_closed_further_m"], 0.0016)
        self.assertIs(facts["jaw_closed_beyond_threshold"], True)
        self.assertEqual(facts["recheck"], [{"stage": "lift", "reason": "aperture_below_min_m"}])
        # Missing readings stay unknown.
        unknown = grip_bench.slip_facts({"stages": {}}, {"stages": {}}, None, 0.001)
        self.assertIsNone(unknown["jaw_closed_further_m"])
        self.assertIsNone(unknown["jaw_closed_beyond_threshold"])
        self.assertIsNone(unknown["transport_aperture_change_m"])

    def test_summary_over_repetitions(self):
        def repetition(load, rise):
            return {"empty": {"close": {"status": "completed", "gripper_execution": {"measured_opening_m": 0.004,
                                                                                    "jaw_settled": True}},
                              "hold": {"samples": 60, "load_pct": {"mean": load}, "temperature_c": {"rise": rise},
                                       "current_ma": None}}}
        summary = grip_bench.summarise([repetition(-20.0, 3.0), repetition(-18.0, 5.0)])
        self.assertEqual(summary["empty.hold.load_pct.mean"], dict(mean=-19.0, stdev=round(2 ** 0.5, 6), min=-20.0,
                                                                   max=-18.0, n=2))
        self.assertEqual(summary["empty.hold.temperature_c.rise"]["mean"], 4.0)
        self.assertNotIn("empty.close.gripper_execution.jaw_settled", summary)  # booleans are not averaged
        self.assertNotIn("empty.close.status", summary)
        self.assertIsNone(grip_bench.summarise([repetition(-20.0, 3.0)])["empty.hold.load_pct.mean"]["stdev"])


class GateTests(unittest.TestCase):
    def clock(self):
        state = {"t": 0.0}
        return (lambda seconds: state.__setitem__("t", state["t"] + seconds)), (lambda: state["t"])

    def test_gate_waits_until_the_gripper_has_cooled(self):
        readings = iter([44.0, 40.0, 36.5, 35.5])
        sleep, now = self.clock()
        lines = []
        result = grip_bench.cool_gate(lambda: {"motor": {"gripper": {"temperature_c": next(readings)}}}, 36.0, 600.0,
                                 period_s=5.0, sleep=sleep, now=now, report=lines.append)
        self.assertEqual(result, dict(limit_c=36.0, start_c=44.0, end_c=35.5, waited_s=15.0, ok=True, reason=None))
        self.assertEqual(len(lines), 3)

    def test_gate_times_out_and_never_guesses_a_missing_temperature(self):
        sleep, now = self.clock()
        hot = grip_bench.cool_gate(lambda: {"motor": {"gripper": {"temperature_c": 45.0}}}, 36.0, 20.0, period_s=5.0,
                              sleep=sleep, now=now, report=lambda _: None)
        self.assertEqual((hot["ok"], hot["reason"], hot["end_c"]), (False, "cool_timeout", 45.0))
        sleep, now = self.clock()
        for telemetry in ({"motor": {"gripper": {"temperature_c": None}}}, {"motor": {}}, None):
            unknown = grip_bench.cool_gate(lambda: telemetry, 36.0, 10.0, period_s=5.0, sleep=sleep, now=now,
                                      report=lambda _: None)
            self.assertEqual((unknown["ok"], unknown["reason"]), (False, "temperature_unknown"))

        def unreadable():
            raise OSError("no telemetry yet")
        self.assertEqual(grip_bench.cool_gate(unreadable, 36.0, 0.0, sleep=sleep, now=now, report=lambda _: None)["reason"],
                         "temperature_unknown")

    def test_hold_ends_early_at_the_abort_temperature(self):
        sleep, now = self.clock()
        sampler = grip_bench.GripSampler(lambda: None)
        temperatures = iter([35.0, 38.0, 42.5, 50.0])

        def step(seconds):
            sleep(seconds)
            sampler.samples.append(sample(now(), temperature=next(temperatures)))
        window, hot = grip_bench.hold(sampler, 60.0, 42.0, period_s=1.0, sleep=step, now=now)
        self.assertEqual((window, hot), ((0.0, 3.0), 42.5))
        cool = grip_bench.GripSampler(lambda: None)
        window, hot = grip_bench.hold(cool, 2.5, 42.0, period_s=1.0, sleep=sleep, now=now)
        self.assertEqual((round(window[1] - window[0], 6), hot), (2.5, None))

    def test_hold_confirm_samples_ignores_single_sample_spike_and_aborts_on_sustained_heat(self):
        # Regression: a single transient temperature spike must not false-abort a 60 s hold when confirm_samples=2,
        # while two consecutive readings >= abort_c abort immediately.
        sleep, now = self.clock()
        spike_sampler = grip_bench.GripSampler(lambda: None)
        spike_series = iter([36.0, 52.0, 36.5, 36.5])

        def step_spike(seconds):
            sleep(seconds)
            spike_sampler.samples.append(sample(now(), temperature=next(spike_series, 36.5)))
        window, hot = grip_bench.hold(spike_sampler, 4.0, 42.0, confirm_samples=2, period_s=1.0, sleep=step_spike, now=now)
        self.assertEqual((round(window[1] - window[0], 6), hot), (4.0, None))

        sleep2, now2 = self.clock()
        sustained_sampler = grip_bench.GripSampler(lambda: None)
        sustained_series = iter([36.0, 42.5, 43.0, 50.0])

        def step_sustained(seconds):
            sleep2(seconds)
            sustained_sampler.samples.append(sample(now2(), temperature=next(sustained_series)))
        window2, hot2 = grip_bench.hold(sustained_sampler, 60.0, 42.0, confirm_samples=2, period_s=1.0,
                                   sleep=step_sustained, now=now2)
        self.assertEqual((window2, hot2), ((0.0, 3.0), 43.0))


class GripBenchRunTests(unittest.TestCase):
    """The whole protocol through Harness.execute on the fake arm, the jaw stopped by a 29 mm cube."""
    ARGUMENTS = ["--repetitions", "2", "--hold-s", "0.4", "--grasp-centre-m", "0.22", "0.05", "0.065",
                 "--carry-over-xy-m", "0.22", "-0.06", "--carry-height-m", "0.08"]

    def document(self, squeeze):
        raw = bench_document()
        raw["gripper"]["hold_squeeze_rad"] = squeeze
        return raw

    def session_factory(self, squeeze, temperature=30.0, cube=True):
        def factory(arguments, out):
            from ni_rua.core.storage.store import Store
            from ni_rua.runtime.harness import Harness
            deploy = validate_deployment(self.document(squeeze), base=EXAMPLE.parent)
            adapter, driver = fake_adapter(deploy, out / "worker", gripper_normalized=1.0,
                                           jaw_min_normalized=CUBE_JAW if cube else 0.03 / JAW_RANGE)
            status = driver.motor_status

            def motor_status():  # a gripper servo that reports a temperature, a load and a current
                rows = status()
                pushing = driver.jaw - driver.jaw_goal
                rows["gripper"].update(temperature_c=temperature() if callable(temperature) else temperature, load=round(-1000 * pushing, 1),
                                       current_ma=round(-6500 * pushing, 1))
                return rows
            driver.motor_status = motor_status
            store = Store(out / "store", IDENTITY)
            config = Config.from_dict({"backend": {"environment": "so101_real", "robot_model_id": "so101",
                                                   "deployment": str(EXAMPLE)},
                                       "features": {"observation_profile": "rgb_proprio_v1", "external_perception": False, "record_video": False}})
            harness = Harness(store, InProcessBackend(adapter), observation_profile="rgb_proprio_v1", config=config)
            harness.start()
            harness.task_planning.required = False
            self.sessions.append((adapter, driver))

            def read_telemetry():
                adapter._last_motor_read = None  # a motor row with every telemetry read (the fake clock barely moves)
                return adapter.telemetry_tick(frames=False)

            def close():
                adapter.close()
                store.close()
                return []
            return motion_bench.Session(harness, read_telemetry, close)
        return factory

    def run_bench(self, directory, condition, squeeze, *extra, arguments=None, **session):
        self.sessions = []
        deployment_path = Path(directory) / f"{condition}.json"
        deployment_path.write_text(json.dumps(self.document(squeeze)))
        parsed = grip_bench.build_parser().parse_args([
            "run", "--deployment", str(deployment_path), "--condition", condition, "--out", str(Path(directory) / condition),
            *(self.ARGUMENTS if arguments is None else arguments), *extra])
        with (patch("ni_rua.runtime.actions.inspection._trajectory_image", return_value=({"status": "unavailable"}, [])),
              contextlib.redirect_stdout(io.StringIO())):
            report = grip_bench.run(parsed, session_factory=self.session_factory(squeeze, **session))
        written = json.loads((Path(directory) / condition / f"grip-{condition}.json").read_text())
        self.assertEqual(written["summary"], json.loads(json.dumps(report["summary"])))
        return report

    def test_protocol_runs_and_the_conditions_differ_only_by_the_squeeze(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = self.run_bench(directory, "baseline", None)
            squeezed = self.run_bench(directory, "sq-020", 0.02)
            for report, squeeze in ((baseline, None), (squeezed, 0.02)):
                self.assertIsNone(report["aborted"], report["aborted"])
                self.assertEqual((report["schema"], report["tests"], report["skipped_tests"]),
                                 (grip_bench.SCHEMA, ["empty", "cube"], {}))
                self.assertEqual(report["settings"]["gripper"]["hold_squeeze_rad"], squeeze)
                self.assertIn("revision", report["source"])
                self.assertEqual(len(report["repetitions"]), 2)
                for repetition in report["repetitions"]:
                    self.assertEqual({name: gate["ok"] for name, gate in repetition["gates"].items()},
                                     {"empty": True, "cube": True})
                    cube = repetition["cube"]
                    self.assertEqual(list(cube["grasp"]["stages"]), list(grip_bench.GRASP_STAGES))
                    self.assertEqual(list(cube["transport_action"]["stages"]), list(grip_bench.TRANSPORT_STAGES))
                    self.assertEqual(list(cube["release"]["stages"]), list(grip_bench.RELEASE_STAGES))
                    for part in ("grasp", "transport_action", "release"):
                        self.assertEqual(cube[part]["status"], "completed", cube[part])
                    self.assertAlmostEqual(cube["close"]["gripper_execution"]["measured_opening_m"], CUBE_M, delta=3e-4)
                    self.assertGreater(cube["hold"]["samples"], 1)
                    self.assertGreater(cube["transport"]["samples"], 1)
                    self.assertEqual(cube["hold"]["temperature_c"]["rise"], 0.0)
                    # The fake cube never leaves the jaw.
                    self.assertAlmostEqual(cube["slip"]["jaw_closed_further_m"], 0.0, delta=2e-4)
                    self.assertIs(cube["slip"]["jaw_closed_beyond_threshold"], False)
                    self.assertEqual(cube["slip"]["recheck"], [])
                    # The transport ran at the agent's normal speed, the approach at the bench's slower one.
                    self.assertEqual(report["protocol"]["transport_speed_scale"], 1.0)
            # Baseline: the servo pushes against the cube toward the closed limit for the whole hold and transport.
            base_cube = baseline["repetitions"][0]["cube"]
            self.assertNotIn("hold_squeeze", base_cube["close"]["gripper_execution"])
            self.assertAlmostEqual(base_cube["hold"]["squeeze_remaining_rad"]["mean"], CUBE_JAW * JAW_RANGE, delta=2e-3)
            self.assertEqual(base_cube["slip"]["squeeze_remaining_rad"], {})
            # Squeezed: the error the servo holds is the squeeze, in the hold, in the transport and per stage.
            cube = squeezed["repetitions"][0]["cube"]
            hold_squeeze = cube["close"]["gripper_execution"]["hold_squeeze"]
            self.assertEqual((hold_squeeze["applied"], hold_squeeze["squeeze_rad"]), (True, 0.02))
            for window in ("hold", "transport"):
                self.assertAlmostEqual(cube[window]["squeeze_remaining_rad"]["mean"], 0.02, delta=1e-3)
            self.assertEqual(set(cube["slip"]["squeeze_remaining_rad"]), set(grip_bench.TRANSPORT_STAGES))
            self.assertAlmostEqual(cube["slip"]["squeeze_remaining_rad"]["carry_over"], 0.02, delta=1e-3)
            # The modelled load follows the held error: the squeezed hold is far below the baseline hold.
            key = "cube.hold.load_pct.mean"
            self.assertEqual(baseline["summary"][key]["n"], 2)
            self.assertLess(abs(squeezed["summary"][key]["mean"]), 0.2 * abs(baseline["summary"][key]["mean"]))
            self.assertLess(abs(squeezed["summary"]["cube.hold.current_ma.mean"]["mean"]),
                            0.2 * abs(baseline["summary"]["cube.hold.current_ma.mean"]["mean"]))
            # The empty close (stopped by the other fingertip 0.03 rad short of the limit) is squeezed as well.
            empty = squeezed["repetitions"][0]["empty"]
            self.assertEqual(empty["grasp"]["status"], "completed")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                rows = grip_bench.main(["compare", str(Path(directory) / "baseline" / "grip-baseline.json"),
                                   str(Path(directory) / "sq-020" / "grip-sq-020.json")])
            names = [name for name, _ in rows]
            self.assertIn("cube.hold.load_pct.mean", names)
            self.assertIn("cube.slip.jaw_closed_further_m", names)
            self.assertIn("cube.hold.temperature_c.rise", names)
            header = output.getvalue().splitlines()[0]
            self.assertIn("baseline (None rad)", header)
            self.assertIn("sq-020 (0.02 rad)", header)
            # A used directory is refused: every run has its own.
            with self.assertRaises(SystemExit):
                self.run_bench(directory, "baseline", None)

    def test_empty_close_hold_alone_and_the_stage_override(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.run_bench(directory, "sq-stage", 0.02, "--grip-squeeze-rad", "0.01", cube=False,
                                    arguments=["--repetitions", "1", "--hold-s", "0.3"])
            self.assertEqual((report["tests"], set(report["skipped_tests"])), (["empty"], {"cube"}))
            self.assertIsNone(report["aborted"], report["aborted"])
            empty = report["repetitions"][0]["empty"]
            squeeze = empty["close"]["gripper_execution"]["hold_squeeze"]
            self.assertEqual((squeeze["applied"], squeeze["squeeze_rad"]), (True, 0.01))
            self.assertAlmostEqual(empty["hold"]["squeeze_remaining_rad"]["mean"], 0.01, delta=1e-3)
            self.assertEqual(empty["release"]["status"], "completed")
            self.assertEqual(self.sessions[0][0].jaw_goal, 1.0)  # the jaw is left open
            # The stage field needs a deployment that offers it: refused before anything is opened.
            with self.assertRaises(SystemExit):
                self.run_bench(directory, "off", None, "--grip-squeeze-rad", "0.01",
                               arguments=["--repetitions", "1", "--hold-s", "0.3"])
            self.assertEqual(self.sessions, [])

    def test_a_guard_on_the_close_is_recorded_without_pausing(self):
        # No cube between the fingers and a guard on the close. The close is the last stage of the grasp batch, so the
        # shared rule evaluates the guard without pausing: the bench records the reading against the bound.
        with tempfile.TemporaryDirectory() as directory:
            report = self.run_bench(directory, "guarded", 0.02, "--grip-guard-preset", "rigid_object", "--tests", "cube",
                                    cube=False, arguments=["--repetitions", "1", "--hold-s", "0.3", *self.ARGUMENTS[4:]])
            self.assertIsNone(report["aborted"], report["aborted"])
            close = report["repetitions"][0]["cube"]["close"]
            self.assertEqual(close["status"], "completed")
            self.assertLess(close["grip_guard"]["measured_aperture_m"], close["grip_guard"]["min_aperture_m"])
            self.assertTrue(close["gripper_execution"]["hold_squeeze"]["applied"])

    def test_a_hot_gripper_is_never_held(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.run_bench(directory, "hot", 0.02, "--cool-timeout-s", "0", temperature=45.0,
                                    arguments=["--repetitions", "1", "--hold-s", "0.3"])
            self.assertIn("temperature gate before empty: cool_timeout", report["aborted"])
            repetition = report["repetitions"][0]
            self.assertEqual(repetition["gates"]["empty"]["ok"], False)
            self.assertNotIn("grasp", repetition["empty"])  # no close was commanded
            adapter, driver = self.sessions[0]
            self.assertEqual(adapter.jaw_command, 1.0)

    def test_reaching_the_abort_temperature_opens_the_jaw_and_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            # At the start limit and below the abort limit the hold runs to its end.
            report = self.run_bench(directory, "warm", 0.02, "--start-temperature-c", "41", "--abort-temperature-c", "41.5",
                                    temperature=41.0, cube=False, arguments=["--repetitions", "1", "--hold-s", "0.3"])
            self.assertIsNone(report["aborted"])
            reading = {"c": 30.0}
            original = grip_bench.hold

            def heating(sampler, hold_s, abort_c, **options):
                reading["c"] = abort_c + 0.5  # the servo heats up as soon as the hold starts
                return original(sampler, hold_s, abort_c, period_s=0.3, **options)
            with patch.object(grip_bench, "hold", heating):
                report = self.run_bench(directory, "abort", 0.02, temperature=lambda: reading["c"], cube=False,
                                        arguments=["--repetitions", "1", "--hold-s", "30"])
            self.assertIn("empty hold: gripper reached 42.5 degC", report["aborted"])
            empty = report["repetitions"][0]["empty"]
            self.assertEqual(empty["hold"]["aborted_at_temperature_c"], 42.5)
            self.assertLess(empty["hold"]["duration_s"], 5.0)
            self.assertEqual(empty["release"]["status"], "completed")
            self.assertEqual(self.sessions[0][0].jaw_goal, 1.0)  # opened before the bench stopped

    def test_invalid_arguments_are_refused_before_anything_opens(self):
        with tempfile.TemporaryDirectory() as directory:
            for extra in (["--hold-s", "0"], ["--transport-speed-scale", "1.5"],
                          ["--start-temperature-c", "45", "--abort-temperature-c", "42"],
                          ["--abort-confirm-samples", "0"]):
                with self.subTest(extra=extra), self.assertRaises(SystemExit):
                    self.run_bench(directory, "bad", None, *extra, arguments=["--repetitions", "1"])
                self.assertEqual(self.sessions, [])


if __name__ == "__main__":
    unittest.main()
