"""Non-instantaneous tracking through the real native worker dispatch loop."""

from copy import deepcopy
import io
import threading
import unittest
from unittest.mock import patch

import numpy as np

from ni_rua.robot.motion.feedback_motion import ArrivalMonitor, pose_error
from support.motion_fixtures import replan_adapter, motion_action, periodic_pose
from support.motion_fixtures import wait_action
from support.paths import REPO_ROOT

NI_RUA_TEST_TIER = "fast"


def feedback_adapter(*, max_delta=.04, speed_override=None, after_step=None, **options):
    adapter, robot, planner, dispatched = replan_adapter(enabled=False, wound=False, **options)
    adapter.motion_execution_mode = "feedback"
    q = {side: np.zeros(6) for side in ("left", "right")}
    speed = {side: np.zeros(6) for side in q}
    native = adapter.env.take_action

    def take_action(action):
        native(action)
        for side in q:
            delta = np.clip(np.asarray(action[side + "_arm_joint_state"]) - q[side], -max_delta, max_delta)
            q[side] += delta
            speed[side] = delta * 25 if speed_override is None else np.full(6, speed_override)
        if after_step:
            after_step(adapter.env.take_action_cnt[0])

    adapter.env.take_action = take_action
    adapter.state = lambda: {"arms": {side: dict(q_rad=value.tolist(), qdot_rad_s=speed[side].tolist(),
                                                ee_pose=periodic_pose(value)) for side, value in q.items()}}
    adapter.robot.get_joint = lambda robot, ids: [q[robot.arm_name.split("_")[0]].copy()]
    return adapter, planner, dispatched


class FeedbackMotionTests(unittest.TestCase):
    def execute(self, adapter, action=None, cancellation=None, progress=None):
        return adapter.execute(action or motion_action(), progress or (lambda value: None),
                               cancellation or threading.Event())

    def test_fixed_target_repeats_until_measured_arrival_and_velocity_settle(self):
        for arm in ("left", "right"):
            adapter, planner, rows = feedback_adapter(arm=arm)
            receipt = self.execute(adapter, motion_action(arm=arm))
            self.assertEqual(receipt["status"], "completed")
            self.assertEqual(len(planner.pose_calls), 1)
            self.assertEqual(len(rows), 6)
            self.assertEqual(receipt["executed_native_steps"], 6)
            self.assertEqual(receipt["executed_control_steps"], 60)
            np.testing.assert_allclose([row[arm + "_arm_joint_state"][0] for row in rows], [-.125] * 6)
            trace = receipt["planned_trace"]
            self.assertEqual(trace["reference_resampled_steps"], 3)
            self.assertEqual(trace["executed_motion_steps"], 6)
            self.assertEqual(trace["feedback_samples"][-1]["settled_samples"], 2)
            self.assertGreater(trace["feedback_samples"][-3]["max_joint_velocity_rad_s"], .1)

    def test_each_waypoint_plans_from_measured_state_then_jaw_dwell_only_after_last(self):
        adapter, planner, rows = feedback_adapter()
        receipt = self.execute(adapter, motion_action(angles=(-.125, -.25), gripper=True))
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(len(planner.pose_calls), 2)
        self.assertAlmostEqual(planner.pose_calls[1][0][0], -.125)
        self.assertEqual(receipt["executed_native_steps"], 18)
        self.assertEqual([r["left_ee_joint_state"][0] for r in rows], [1.] * 12 + [0.] * 6)
        np.testing.assert_allclose([r["left_arm_joint_state"][0] for r in rows[-6:]], [-.25] * 6)
        self.assertTrue(all(r["right_ee_joint_state"] == [.3] for r in rows))

    def test_stall_spends_bounded_steps_and_never_executes_gripper(self):
        adapter, _, rows = feedback_adapter(max_delta=0)
        receipt = self.execute(adapter, motion_action(gripper=True))
        self.assertEqual((receipt["status"], receipt["reason"]), ("failed", "feedback_stalled"))
        self.assertEqual(receipt["executed_native_steps"], 15)
        self.assertTrue(all(r["left_ee_joint_state"] == [1.] for r in rows))

    def test_slow_real_progress_still_obeys_unconditional_action_cap(self):
        adapter, _, rows = feedback_adapter(step_limit=1000, max_delta=.001)
        receipt = self.execute(adapter, motion_action(angles=(-.9,), gripper=True))
        self.assertEqual(receipt["reason"], "motion_duration_limit")
        self.assertEqual(receipt["executed_native_steps"], 250)
        self.assertTrue(all(row["left_ee_joint_state"] == [1.] for row in rows))

    def test_high_reported_speed_without_progress_stops_and_preserves_grip(self):
        # Real failures included both 3.588 rad/s and a marginal 0.103888 rad/s.
        for speed in (3.588, .103888):
            adapter, _, rows = feedback_adapter(max_delta=0, speed_override=speed)
            receipt = self.execute(adapter, motion_action(gripper=True))
            self.assertEqual((receipt["status"], receipt["reason"]), ("failed", "feedback_stalled"))
            self.assertEqual(receipt["executed_native_steps"], 15)
            self.assertTrue(all(row["left_ee_joint_state"] == [1.] for row in rows))

    def test_pose_arrived_but_never_settled_is_bounded_without_releasing(self):
        adapter, _, rows = feedback_adapter(max_delta=1, speed_override=1)
        receipt = self.execute(adapter, motion_action(gripper=True))
        self.assertEqual(receipt["reason"], "feedback_stalled")
        self.assertEqual(receipt["executed_native_steps"], 16)
        self.assertLess(receipt["position_error_m"], .005)
        self.assertTrue(all(row["left_ee_joint_state"] == [1.] for row in rows))

    def test_high_velocity_with_real_progress_can_run_beyond_stall_window(self):
        adapter, _, _ = feedback_adapter(max_delta=.005, step_limit=1000)
        receipt = self.execute(adapter, motion_action(angles=(-.9,)))
        self.assertEqual(receipt["status"], "completed")
        self.assertGreater(receipt["executed_native_steps"], 100)

    def test_periodic_motion_without_best_error_improvement_is_bounded(self):
        target = periodic_pose([-.9])
        initial = dict(q_rad=[0.] * 6, qdot_rad_s=[2.] * 6, ee_pose=periodic_pose([0.]))
        monitor = ArrivalMonitor(target, [-.9] + [0.] * 5, initial)
        for index in range(15):
            state = deepcopy(initial)
            state["q_rad"][0] = .02 if index % 2 else 0.
            state["ee_pose"] = periodic_pose(state["q_rad"])
            outcome, _ = monitor.observe(state)
            self.assertEqual(outcome, "stalled" if index == 14 else "continue")

    def test_native_terminal_retains_last_completed_sample_and_no_extra_jaw(self):
        for success_at in (None, 2):
            adapter, _, rows = feedback_adapter(step_limit=3, success_at=success_at)
            receipt = self.execute(adapter, motion_action(gripper=True))
            count = success_at or 3
            self.assertEqual((receipt["status"], receipt["reason"]), ("stopped", "episode_ended"))
            self.assertEqual(receipt["executed_native_steps"], count)
            self.assertEqual(receipt["planned_trace"]["executed_motion_steps"], count)
            self.assertEqual(len(receipt["planned_trace"]["feedback_samples"]), count)
            self.assertTrue(all(row["left_ee_joint_state"] == [1.] for row in rows))
            again = self.execute(adapter)
            self.assertEqual(again["executed_native_steps"], 0)

    def test_cancel_during_execution_keeps_counts_and_does_not_release(self):
        cancellation = threading.Event()
        adapter, _, rows = feedback_adapter(after_step=lambda n: cancellation.set() if n == 3 else None)
        receipt = self.execute(adapter, motion_action(gripper=True), cancellation)
        self.assertEqual((receipt["status"], receipt["reason"]), ("stopped", "cancelled"))
        self.assertEqual(receipt["executed_native_steps"], 3)
        self.assertEqual(receipt["planned_trace"]["executed_motion_steps"], 3)
        self.assertEqual(len(rows), 3)

    def test_deadline_after_planning_or_after_native_step_prevents_more_motion(self):
        for planning in (True, False):
            adapter, planner, rows = feedback_adapter()
            expired = [False]
            if planning:
                original = planner.plan_path
                def plan(*args):
                    result = original(*args)
                    expired[0] = True
                    return result
                planner.plan_path = plan
            else:
                original_native = adapter.env.take_action
                def execute(action):
                    original_native(action)
                    expired[0] = True
                adapter.env.take_action = execute
            with patch("ni_rua.robot.backends.robodojo.robot_worker.time.monotonic", side_effect=lambda: 100 if expired[0] else 0):
                receipt = self.execute(adapter, motion_action(timeout_s=1, gripper=True))
            self.assertEqual(receipt["reason"], "planning_timeout" if planning else "execution_timeout")
            self.assertEqual(len(rows), 0 if planning else 1)

    def test_partial_native_fault_preserves_started_but_incomplete_count(self):
        adapter, _, rows = feedback_adapter()
        def failure(action):
            adapter.env.take_action_cnt[0] += 1
            adapter.control_step += 4
            raise RuntimeError("partial feedback native fault")
        adapter.env.take_action = failure
        with patch("sys.stderr", new=io.StringIO()):
            receipt = self.execute(adapter)
        self.assertEqual(receipt["status"], "unknown")
        self.assertEqual((receipt["executed_native_steps"], receipt["completed_native_steps"], receipt["executed_control_steps"]), (1, 0, 4))

    def test_unsupported_preview_and_synchronized_gripper_reject_without_motion(self):
        adapter, planner, rows = feedback_adapter()
        self.assertFalse(adapter.capabilities()["motion_preview"]["available"])
        self.assertEqual(adapter.capabilities()["gripper_phases"], ["after_motion"])
        with self.assertRaisesRegex(ValueError, "feedback_motion_preview_unsupported"):
            adapter.preview_motion(motion_action())
        synchronized = motion_action(gripper=True)
        synchronized["gripper_events"][0].update(phase="during_motion", start_fraction=0, end_fraction=1)
        for action in (dict(motion_action(), preview_plan_id="existing-plan"), synchronized):
            receipt = self.execute(adapter, action)
            self.assertEqual(receipt["status"], "rejected")
            self.assertEqual(receipt["executed_native_steps"], 0)
        self.assertEqual(rows, [])
        self.assertEqual(planner.pose_calls, [])

    def test_wait_keeps_explicit_native_step_semantics_in_feedback_mode(self):
        adapter, _, rows = feedback_adapter()
        receipt = self.execute(adapter, wait_action(7))
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["executed_native_steps"], 7)

    def test_rotation_sign_and_finite_measured_velocity(self):
        target = periodic_pose([.3])
        negative = deepcopy(target)
        negative["quaternion_xyzw"] = [-v for v in target["quaternion_xyzw"]]
        self.assertAlmostEqual(pose_error(negative, target)[1], 0.)
        state = dict(q_rad=[0.] * 6, qdot_rad_s=[0.] * 6, ee_pose=periodic_pose([0.]))
        monitor = ArrivalMonitor(target, [.3] + [0.] * 5, state)
        state["qdot_rad_s"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite_measured"):
            monitor.observe(state)


# ======================================================================
# 闭环反馈模式配置与端点合法性测试 (迁自 test_feedback_config.py / test_feedback_endpoint.py)
# ======================================================================


class FeedbackConfigTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.case = dict(
            cwd=str(REPO_ROOT),
            run_dir=str(self.root / "run"),
            task="move_bottle",
            seed=0,
            layout_id=0,
            title="feedback config",
            tag=[],
        )

    def test_default_and_archived_configs_preserve_execution_policy(self):
        import json
        from ni_rua.core.config import Config
        from scripts import benchmark_queue as queue

        self.assertEqual(Config.from_dict({}).features.motion_execution_mode, "resampled")
        for mode in ("resampled", "feedback"):
            with self.subTest(mode=mode):
                config = Config.from_dict({"features": {"motion_execution_mode": mode, "motion_review": False}})
                manifest = queue.configured_manifest({"cases": [self.case]}, config)
                with patch.object(Config, "load", side_effect=AssertionError("ambient reload")):
                    restored = queue.queue_config(json.loads(json.dumps(manifest)))
                self.assertEqual(restored.features.motion_execution_mode, mode)

    def test_invalid_modes_fail_at_config_backend_and_queue_boundaries(self):
        from ni_rua.core.config import Config, ConfigError
        from ni_rua.robot.backends.robodojo.robodojo import RobotBackend
        from scripts import benchmark_queue as queue

        for value in (None, False, 0, "", "closed_loop", ["feedback"]):
            with self.subTest(value=value):
                with self.assertRaises(ConfigError):
                    Config.from_dict({"features": {"motion_execution_mode": value}})
                with self.assertRaisesRegex(ValueError, "invalid_motion_execution_mode"):
                    RobotBackend(config=Config(), motion_execution_mode=value)
                with self.assertRaisesRegex(ValueError, "Invalid motion_execution_mode"):
                    queue.command({**self.case, "motion_execution_mode": value}, 0)

    def test_queue_case_override_reaches_gateway_resolved_config_and_backend(self):
        import json
        import sys
        from unittest.mock import AsyncMock
        from ni_rua.apps.gateway import cli as gateway
        from ni_rua.core.config import Config, ConfigError
        from ni_rua.robot.backends.robodojo.robodojo import RobotBackend
        from scripts import benchmark_queue as queue

        snapshot = self.root / "config.json"
        for configured, override in (("resampled", None), ("feedback", None), ("resampled", "feedback"), ("feedback", "resampled")):
            with self.subTest(configured=configured, override=override):
                config = Config.from_dict({"features": {"motion_execution_mode": configured, "motion_review": False}})
                snapshot.write_text(json.dumps(config.to_dict()))
                case = {**self.case, **({"motion_execution_mode": override} if override else {})}
                if (override or configured) == "feedback":
                    with self.assertRaisesRegex(ConfigError, "feedback_is_diagnostic_only"):
                        queue.command(case, 0, config=config, config_path=snapshot)
                    continue
                cmd = queue.command(case, 0, config=config, config_path=snapshot)
                with patch.object(gateway, "serve", new_callable=AsyncMock) as serve, patch.object(sys, "argv", ["gateway", *cmd[4:]]):
                    gateway.main()
                resolved = serve.call_args.args[0].resolved_config
                self.assertEqual(resolved.features.motion_execution_mode, override or configured)
                with patch.object(Config, "load", side_effect=AssertionError("ambient reload")):
                    backend = RobotBackend(config=resolved)
                self.assertEqual(backend.motion_execution_mode, override or configured)

    def test_feedback_diagnostics_and_archives_remain_readable_but_benchmark_cannot_launch(self):
        import sys
        from unittest.mock import AsyncMock
        from ni_rua.apps.gateway import cli as gateway
        from ni_rua.core.config import Config, ConfigError

        config = Config.from_dict({"features": {"motion_execution_mode": "feedback", "motion_review": False}})
        self.assertIs(config.validate_launch(), config)
        archived = config.with_overrides({"run": {"replay": True, "run_dir": str(self.root / "archive")}})
        self.assertIs(archived.validate_launch(), archived)
        with self.assertRaisesRegex(ConfigError, "feedback_is_diagnostic_only"):
            config.with_overrides({"run": {"benchmark": True}}).validate_launch()
        with patch.object(gateway, "serve", new_callable=AsyncMock) as serve, \
                patch.object(sys, "argv", ["gateway", "--benchmark", "--motion-execution-mode", "feedback", "--no-motion-review"]), \
                self.assertRaises(SystemExit) as stopped:
            gateway.main()
        self.assertEqual(stopped.exception.code, 2)
        serve.assert_not_called()

    def test_backend_constructor_override_and_worker_cli(self):
        from unittest.mock import MagicMock
        from ni_rua.core.config import Config
        from ni_rua.robot.backends.robodojo.robodojo import RobotBackend

        for configured, override in (("resampled", None), ("feedback", None), ("feedback", "resampled")):
            with self.subTest(configured=configured, override=override):
                config = Config.from_dict({"features": {"motion_execution_mode": configured, "motion_review": False}})
                backend = RobotBackend(config=config, run_dir=self.root / configured, **({"motion_execution_mode": override} if override else {}))
                process = MagicMock(pid=999999999)
                with patch("ni_rua.robot.backends.robodojo.robot_process.subprocess.Popen", return_value=process) as popen, \
                        patch("ni_rua.robot.backends.robodojo.robot_process.threading.Thread"), \
                        patch.object(backend, "_request", return_value={"ready": True}):
                    try:
                        self.assertEqual(backend.start(), {"ready": True})
                    finally:
                        backend.process = None
                        if getattr(backend, "log", None) is not None:
                            backend.log.close()
                args = popen.call_args.args[0]
                self.assertEqual(args.count("--motion-execution-mode"), 1)
                self.assertEqual(args[args.index("--motion-execution-mode") + 1], override or configured)

    def test_feedback_conflicts_rejected_by_snapshots_files_and_overrides(self):
        import json
        from ni_rua.core.config import Config, ConfigError
        from scripts import benchmark_queue as queue

        for feature in ("motion_review", "tracking_guard"):
            valid = {"motion_execution_mode": "feedback", "motion_review": False}
            invalid = {**valid, feature: True}
            expected = "requires_" + feature + "_disabled"
            with self.subTest(feature=feature):
                with self.assertRaisesRegex(ConfigError, expected):
                    Config.from_dict({"features": invalid})
                config = Config.from_dict({"features": valid})
                with self.assertRaisesRegex(ConfigError, expected):
                    config.with_overrides({"features": {feature: True}})
                snapshot = self.root / "invalid.json"
                snapshot.write_text(json.dumps({"features": invalid}))
                with self.assertRaisesRegex(ConfigError, expected):
                    Config.load(snapshot, environ={})
                with self.assertRaisesRegex(ConfigError, expected):
                    queue.command({**self.case, **invalid}, 0)
        switched = Config().with_overrides({"features": {"motion_execution_mode": "feedback", "motion_review": False}})
        self.assertEqual(switched.features.motion_execution_mode, "feedback")
        self.assertTrue(switched.with_overrides({"features": {"motion_execution_mode": "resampled", "motion_review": True}}).features.motion_review)


class FeedbackEndpointTests(unittest.TestCase):
    def execute(self, adapter, action=None):
        return adapter.execute(action or motion_action(gripper=True), lambda event: None, threading.Event())

    def test_winding_pose_plan_selects_legal_short_endpoint_once_on_both_arms(self):
        import math

        for arm in ("left", "right"):
            with self.subTest(arm=arm):
                adapter, planner, rows = feedback_adapter(arm=arm)
                planner.wound = True
                receipt = self.execute(adapter, motion_action(arm=arm, gripper=True))
                self.assertEqual(receipt["status"], "completed")
                self.assertEqual(len(planner.pose_calls), 1)
                self.assertEqual(planner.joint_calls, [])
                self.assertEqual(receipt["planned_trace"]["executed_motion_steps"], 6)
                self.assertEqual(receipt["executed_native_steps"], 12 if arm == "left" else 8)
                np.testing.assert_allclose([row[arm + "_arm_joint_state"][0] for row in rows], -0.125)
                diagnostic = receipt["planned_trace"]["feedback_endpoint"]
                self.assertEqual(diagnostic["turn_shifts"], [-1, 0, 0, 0, 0, 0])
                self.assertTrue(diagnostic["validated"])
                self.assertTrue(diagnostic["changed"])
                self.assertLess(diagnostic["all_link_fk_max_abs_error"], 1e-12)
                self.assertGreater(receipt["planned_trace"]["reference_resampled_steps"], 100)
                self.assertAlmostEqual(planner.originals[0][-1, 0], math.tau - 0.125)

    def test_nearest_endpoint_uses_measured_unwrapped_state_and_respects_joint_six_limit(self):
        import math

        adapter, robot, _, _ = replan_adapter()
        current = np.array([6.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        goal = np.array([-0.125, 0.0, 0.0, 0.0, 0.0, 2.8])
        diagnostic = {}
        selected = adapter._feedback_endpoint(robot, periodic_pose(goal), current, goal, 0.005, diagnostic)
        self.assertAlmostEqual(selected[0], math.tau - 0.125)
        self.assertEqual(selected[5], 2.8)
        self.assertEqual(diagnostic["turn_shifts"], [1, 0, 0, 0, 0, 0])

    def test_geometry_failure_rejects_before_any_native_step_or_jaw_command(self):
        adapter, planner, rows = feedback_adapter()
        planner.wound = True

        def invalid(*args):
            raise ValueError("unsupported_joint_replan_geometry")

        adapter._joint_replan_geometry = invalid
        receipt = self.execute(adapter)
        self.assertEqual((receipt["status"], receipt["reason"]), ("failed", "feedback_endpoint_invalid"))
        self.assertEqual(receipt["executed_native_steps"], 0)
        self.assertEqual(rows, [])
        self.assertFalse(getattr(adapter, "evaluator_failure", None))

    def test_fk_mismatch_nonrevolute_and_invalid_pose_reject_without_fallback(self):
        for failure in ("fk", "nonrevolute", "position", "orientation", "nan"):
            with self.subTest(failure=failure):
                adapter, planner, rows = feedback_adapter()
                planner.wound = True
                original = adapter._joint_replan_geometry

                def geometry(*args):
                    kinds, transforms, err_fn = original(*args)
                    if failure == "nonrevolute":
                        kinds[0] = "prismatic"
                    if failure == "fk":
                        def transforms(q):
                            result = np.eye(4)
                            result[0, 3] = q[0]
                            return {"tool": result}
                    if failure in {"position", "orientation", "nan"}:
                        err_fn = lambda q: {"position": (0.0051, 0), "orientation": (0, 0.101), "nan": (float("nan"), 0)}[failure]
                    return kinds, transforms, err_fn

                adapter._joint_replan_geometry = geometry
                receipt = self.execute(adapter)
                self.assertEqual(receipt["reason"], "feedback_endpoint_invalid")
                self.assertEqual(receipt["executed_native_steps"], 0)
                self.assertEqual(rows, [])

    def test_requested_tighter_position_tolerance_is_validated(self):
        adapter, _, rows = feedback_adapter()
        original = adapter._joint_replan_geometry

        def geometry(*args):
            kinds, transforms, _ = original(*args)
            return kinds, transforms, lambda q: (0.002, 0.0)

        adapter._joint_replan_geometry = geometry
        action = motion_action()
        action["path"]["position_tolerance_m"] = 0.001
        self.assertEqual(self.execute(adapter, action)["reason"], "feedback_endpoint_invalid")
        self.assertEqual(rows, [])

    def test_resampled_mode_keeps_original_wound_trajectory_without_geometry_lookup(self):
        import math

        adapter, _, _, rows = replan_adapter(enabled=False, wound=True, step_limit=1000)
        adapter.motion_execution_mode = "resampled"
        receipt = self.execute(adapter, motion_action())
        self.assertEqual(receipt["status"], "completed")
        self.assertGreater(receipt["executed_native_steps"], 100)
        self.assertAlmostEqual(rows[-1]["left_arm_joint_state"][0], math.tau - 0.125)
        self.assertEqual(adapter.geometry_calls, [])


if __name__ == "__main__":
    unittest.main()
