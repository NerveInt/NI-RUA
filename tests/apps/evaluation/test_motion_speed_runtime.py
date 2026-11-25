"""Speed requests through real worker compilation and native dispatch fixtures."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

from ni_rua.core.config import Config, ConfigError
from ni_rua.core.storage.store import Store
from ni_rua.robot.motion.arx_actions import compile_action
from ni_rua.robot.motion.joint_replan import select_joint_replan
from ni_rua.robot.motion.motion_speed import bounded_resample
from ni_rua.runtime.harness import Harness
from scripts import benchmark_queue as queue
from apps.evaluation.test_feedback_motion import feedback_adapter
from support.fixture_robot import FixtureRobot
from support.motion_fixtures import motion_action, replan_adapter
from support.motion_fixtures import preview_adapter
from support.motion_fixtures import wait_action
from support.motion_fixtures import synchronized
from support.paths import REPO_ROOT

NI_RUA_TEST_TIER = "component"


def speed_adapter(*, preview=False, measured_opening=.6, **kwargs):
    if preview:
        adapter, planner, dispatched = preview_adapter(**kwargs)
    else:
        adapter, _, planner, dispatched = replan_adapter(enabled=False, wound=False, **kwargs)
    adapter.arm_velocity_step_limits = {side: np.array([.2] * 6) for side in ("left", "right")}
    robot = adapter.robot.robot_list[0]
    robot.gripper_joints_name = ["jaw1", "jaw2"]
    robot.gripper_move = {"base": "jaw1", "sign": 1}
    robot.gripper_scale = [-.01, .044]
    original_state = adapter.state

    def state():
        value = original_state()
        for side in value["arms"]:
            value["arms"][side]["gripper"] = dict(position_source_kind="measurement",
                position_m=[-.01 + .054 * measured_opening] * 2)
        return value

    adapter.state = state
    return adapter, planner, dispatched


class MotionSpeedRuntimeTests(unittest.TestCase):
    def execute(self, adapter, action):
        return adapter.execute(action, lambda event: None, threading.Event())

    def test_scale_one_is_exact_legacy_and_slower_faster_preserve_endpoint(self):
        counts = []
        for scale in (.5, 1., 2.):
            adapter, _, rows = speed_adapter()
            action = motion_action(angles=(-.4,))
            action["speed_scale"] = scale
            result = self.execute(adapter, action)
            self.assertEqual(result["status"], "completed")
            counts.append(len(rows))
            q = np.array([row["left_arm_joint_state"] for row in rows])
            self.assertLessEqual(np.abs(np.diff(np.vstack([np.zeros(6), q]), axis=0)).max(), .05 * scale + 1e-12)
            np.testing.assert_allclose(q[-1], [-.4, 0, 0, 0, 0, 0])
            self.assertEqual(result["speed_scale"], scale)
            self.assertEqual(result["executed_control_steps"], 10 * len(rows))
            self.assertEqual(result["planned_trace"]["completed_motion_steps"], len(rows))
            if scale == 1:
                np.testing.assert_array_equal(q, adapter._resample_plan(np.array([[0.] * 6, [-.4, 0, 0, 0, 0, 0]]), "left"))
        self.assertEqual(counts, [16, 8, 4])

    def test_native_velocity_cap_and_low_velocity_joint_slowdown(self):
        adapter, _, _ = speed_adapter()
        adapter.arm_step_limits["left"][0] = .02
        adapter.arm_velocity_step_limits["left"][0] = .02
        self.assertAlmostEqual(adapter._speed_step_limits("left", 2)[0], .02)
        self.assertAlmostEqual(adapter._speed_step_limits("left", .5)[0], .01)
        del adapter.arm_velocity_step_limits
        np.testing.assert_allclose(adapter._speed_step_limits("left", 2), adapter.arm_step_limits["left"])

    def test_bounded_resampling_retains_corner_reversal_and_exact_endpoint(self):
        points = np.array([[0, 0], [.01, 0], [.2, 0], [.2, .15], [.2, .3], [0, .3], [0, 0]])
        rows = bounded_resample(points, [.1, .1])
        for knot in points[[2, 4, 5, 6]]:
            self.assertTrue(np.any(np.all(rows == knot, axis=1)))
        self.assertLessEqual(np.abs(np.diff(np.vstack([points[0], rows]), axis=0)).max(), .1 + 1e-12)
        np.testing.assert_array_equal(rows[-1], points[-1])
        loop = bounded_resample(np.array([[0.], [.2], [0.]]), [.1])
        np.testing.assert_allclose(loop.ravel(), [.1, .2, .1, 0.])

    def test_slow_gripper_ramps_from_measured_not_latched_then_old_dwell(self):
        adapter, _, rows = speed_adapter(measured_opening=.6)
        action = wait_action()
        del action["hold_native_steps"]
        action.update(gripper_events=[dict(arm="left", phase="after_motion", goal_normalized=0.)],
                      gripper_speed_scale=.5)
        result = self.execute(adapter, action)
        self.assertEqual(result["status"], "completed")
        jaw = np.array([row["left_ee_joint_state"][0] for row in rows])
        self.assertEqual(len(jaw), 7 + 6)
        np.testing.assert_allclose(jaw[:7], np.linspace(.6, 0, 8)[1:])
        np.testing.assert_array_equal(jaw[7:], 0.)
        self.assertTrue(all(row["right_ee_joint_state"] == [.3] for row in rows))
        self.assertLessEqual(np.abs(np.diff(np.r_[.6, jaw])).max(), .09)
        self.assertEqual(result["planned_trace"]["completed_gripper_steps"], len(rows))

    def test_default_gripper_remains_old_immediate_goal_and_dwell(self):
        adapter, _, rows = speed_adapter()
        result = self.execute(adapter, motion_action(gripper=True))
        self.assertEqual(result["status"], "completed")
        self.assertEqual([row["left_ee_joint_state"][0] for row in rows], [1.] * 3 + [0.] * 6)

    def test_slow_opening_supports_reversed_native_jaw_axis(self):
        adapter, _, rows = speed_adapter(measured_opening=.6)
        robot = adapter.robot.robot_list[0]
        robot.gripper_move["sign"] = -1
        # Identical measured position means opening .4 on the reversed axis.
        action = wait_action()
        del action["hold_native_steps"]
        action.update(gripper_events=[dict(arm="left", phase="after_motion", goal_normalized=1.)],
                      gripper_speed_scale=.25)
        result = self.execute(adapter, action)
        self.assertEqual(result["status"], "completed")
        values = np.array([row["left_ee_joint_state"][0] for row in rows])
        self.assertGreater(values[0], .4)
        self.assertLessEqual(np.abs(np.diff(np.r_[.4, values])).max(), .045 + 1e-12)
        self.assertEqual(values[-1], 1.)

    def test_waypoints_share_speed_and_each_endpoint_is_retained(self):
        adapter, planner, rows = speed_adapter()
        result = self.execute(adapter, motion_action(angles=(-.2, -.4)) | dict(speed_scale=2.))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(planner.pose_calls), 2)
        self.assertEqual(len(rows), 4)
        np.testing.assert_allclose([row["left_arm_joint_state"][0] for row in rows], [-.1, -.2, -.3, -.4])
        self.assertEqual([segment["speed_scale"] for segment in result["planned_trace"]["segments"]], [2., 2.])

    def test_synchronized_slow_gripper_bounds_ramp_from_measured_opening(self):
        adapter, _, rows = speed_adapter()
        action = synchronized(motion_action())
        action.update(speed_scale=2., gripper_speed_scale=.25)
        result = self.execute(adapter, action)
        self.assertEqual(result["status"], "completed")
        values = np.array([row["left_ee_joint_state"][0] for row in rows])
        self.assertLessEqual(np.abs(np.diff(np.r_[.6, values])).max(), .045 + 1e-12)
        self.assertEqual(values[-1], 0.)
        self.assertEqual(result["planned_trace"]["gripper_motion"]["initial_goal_normalized"], .6)

    def test_preview_execution_uses_identical_speed_rows_and_binds_both_fields(self):
        for field in ("speed_scale", "gripper_speed_scale"):
            adapter, _, rows = speed_adapter(preview=True)
            action = motion_action(gripper=True)
            action.update(speed_scale=2., gripper_speed_scale=.5)
            preview = adapter.preview_motion(action)
            self.assertEqual(preview["status"], "planned")
            altered = deepcopy(action)
            altered.update(preview_plan_id=preview["plan_id"])
            altered[field] = 1.
            result = self.execute(adapter, altered)
            self.assertEqual(result["reason"], "preview_plan_action_mismatch")
            self.assertEqual(rows, [])
        adapter, _, rows = speed_adapter(preview=True)
        action = motion_action(gripper=True) | dict(speed_scale=2., gripper_speed_scale=.5)
        preview = adapter.preview_motion(action)
        result = self.execute(adapter, action | dict(preview_plan_id=preview["plan_id"]))
        self.assertEqual(result["status"], "completed")
        actual = [[*row["left_arm_joint_state"], *row["left_ee_joint_state"],
                   *row["right_arm_joint_state"], *row["right_ee_joint_state"]] for row in rows]
        np.testing.assert_array_equal(actual, preview["native_rows"])

    def test_feedback_rejects_speed_without_planning_or_steps(self):
        adapter, planner, rows = speed_adapter()
        adapter.motion_execution_mode = "feedback"
        result = self.execute(adapter, motion_action() | dict(speed_scale=2.))
        self.assertEqual(result["reason"], "feedback_action_speed_unsupported")
        self.assertEqual(result["executed_native_steps"], 0)
        self.assertFalse(adapter.capabilities()["action_speed"]["available"])
        self.assertEqual(planner.pose_calls, [])
        self.assertEqual(rows, [])

    def test_default_digest_treats_omitted_integer_float_defaults_identically(self):
        adapter, _, _ = speed_adapter()
        action = motion_action(gripper=True)
        digest = adapter._preview_action_digest(action)
        self.assertEqual(digest, adapter._preview_action_digest(action | dict(speed_scale=1, gripper_speed_scale=1)))
        self.assertEqual(digest, adapter._preview_action_digest(action | dict(speed_scale=1., gripper_speed_scale=1.)))

    def test_exact_preview_moved_jaw_start_stops_before_release_then_can_retry(self):
        for cached in (True,):
            adapter, _, rows = speed_adapter(preview=True, measured_opening=.6)
            original_state = adapter.state

            def state():
                value = original_state()
                # A previously opening jaw finishes moving during arm motion.
                if adapter.env.take_action_cnt[0]:
                    value["arms"]["left"]["gripper"]["position_m"] = [.044, .044]
                return value

            adapter.state = state
            action = motion_action(gripper=True) | dict(speed_scale=2., gripper_speed_scale=.5)
            if cached:
                preview = adapter.preview_motion(action)
                action["preview_plan_id"] = preview["plan_id"]
            result = self.execute(adapter, action)
            self.assertEqual(result["reason"], "gripper_start_state_changed")
            self.assertEqual(result["executed_native_steps"], 2)
            self.assertEqual(result["planned_trace"]["completed_motion_steps"], 2)
            self.assertNotIn("completed_gripper_steps", result["planned_trace"])
            self.assertTrue(all(row["left_ee_joint_state"] == [1.] for row in rows))
            self.assertIn("separate gripper request", result["gripper_start_state"]["recovery"])
            retry = wait_action()
            del retry["hold_native_steps"]
            retry.update(gripper_events=[dict(arm="left", phase="after_motion", goal_normalized=0.)],
                         gripper_speed_scale=.5)
            resumed = self.execute(adapter, retry)
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(resumed["executed_native_steps"], 12 + 6)
            self.assertGreater(rows[2]["left_ee_joint_state"][0], .9)

    def test_missing_planner_start_knot_is_bounded_from_measured_start(self):
        adapter, _, _ = speed_adapter()
        path = np.array([[-.25] + [0.] * 5, [-.4] + [0.] * 5])
        rows = adapter._resample_action_plan(path, "left", dict(speed_scale=2.))
        self.assertLessEqual(np.abs(np.diff(np.vstack([np.zeros(6), rows]), axis=0)).max(), .1 + 1e-12)

    def test_slow_request_exceeding_cap_does_not_dispatch(self):
        adapter, _, rows = speed_adapter(step_limit=1000)
        result = self.execute(adapter, motion_action(angles=(-3.13,)) | dict(speed_scale=.25))
        self.assertEqual(result["reason"], "motion_duration_limit")
        self.assertEqual(rows, [])


class MotionSpeedIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def harness(self, default=1.25):
        store = Store(self.root / "store", dict(run_id="speed", episode_id="ep", env_instance_id="env", env_generation=1))
        self.addCleanup(store.close)
        robot = FixtureRobot(self.root)
        original_observe = robot.observe
        robot.observe = lambda: dict(original_observe(),
            camera_calibration=dict(reference_frame_id="env_origin", cameras={}),
            camera_pose_sample=dict(cameras={}), robot_model=dict(base_link="base", joints=[]))
        caps = dict(robot.start(), path_kinds=["backend_free", "hold"],
                    action_speed=dict(available=True, speed_scale=dict(default=1.)))
        robot.start = lambda: caps
        config = Config.from_dict(dict(features=dict(motion_review=False, default_motion_speed_scale=default)))
        harness = Harness(store, robot, config=config)
        harness.start()
        return harness, robot, caps

    def move(self, harness, **values):
        return dict(observation_id=harness.latest["observation_id"], arm="left",
                    position_m=[.1, 0, 1.], quaternion_xyzw=[0, 0, 0, 1], **values)

    def test_actual_default_capability_is_published_without_mutating_backend(self):
        h, _, caps = self.harness(1.5)
        self.assertEqual(caps["action_speed"]["speed_scale"]["default"], 1.)
        expected = h.capabilities["action_speed"]
        self.assertEqual(expected["speed_scale"]["default"], 1.5)
        self.assertEqual(expected["speed_presets"], dict(slower=.5, slow=.8, careful=.9, normal=1., fast=1.25, faster=1.5))
        self.assertEqual(h.store.read_state("runtime", "capabilities")["action_speed"], expected)

    def test_equivalent_preset_and_numeric_retry_is_idempotent(self):
        h, robot, _ = self.harness()
        args = self.move(h, speed_preset="normal", action_id="same")
        h.execute(compile_action("move", args, h.identity, h.capabilities), {"tool_call_id": "one"})
        equivalent = {key: value for key, value in args.items() if key != "speed_preset"}
        equivalent["speed_scale"] = 1
        h.execute(compile_action("move", equivalent, h.identity, h.capabilities), {"tool_call_id": "two"})
        self.assertEqual(robot.executions, 1)

    def test_queue_cli_config_and_archived_snapshot_preserve_default(self):
        from ni_rua.apps.gateway import cli as gateway
        snapshot = self.root / "config.json"
        config = Config.from_dict(dict(features=dict(default_motion_speed_scale=1.25)))
        snapshot.write_text(json.dumps(config.to_dict()))
        case = dict(cwd=str(REPO_ROOT), run_dir=str(self.root / "run"),
                    task="put_bottles_into_dustbin", seed=0, layout_id=0, title="default speed", tag=[])
        for override in (None, 1.5):
            value = {**case, **({"default_motion_speed_scale": override} if override else {})}
            command = queue.command(value, 0, config=config, config_path=snapshot)
            with patch.object(gateway, "serve", new_callable=AsyncMock) as serve, patch.object(sys, "argv", ["gateway", *command[4:]]):
                gateway.main()
            resolved = serve.call_args.args[0].resolved_config
            self.assertEqual(resolved.features.default_motion_speed_scale, override or 1.25)
            self.assertEqual(Config.from_dict(json.loads(json.dumps(resolved.to_dict()))).features.default_motion_speed_scale, override or 1.25)

    def test_invalid_defaults_and_feedback_combination_fail_before_launch(self):
        for value in (True, None, 0, .24, 2.01, float("nan"), float("inf"), "fast"):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                Config.from_dict(dict(features=dict(default_motion_speed_scale=value)))
        with self.assertRaisesRegex(ConfigError, "requires_default_speed_1"):
            Config.from_dict(dict(features=dict(motion_execution_mode="feedback", motion_review=False, default_motion_speed_scale=1.25)))


class WaypointSpeedRuntimeTests(unittest.TestCase):
    def execute(self, adapter, action):
        return adapter.execute(action, lambda event: None, threading.Event())

    def test_fast_approach_slow_contact_share_one_request_and_preserve_knots(self):
        adapter, planner, rows = speed_adapter()
        action = motion_action(angles=(-.3, -.6), gripper=True)
        action.update(speed_scale=.6, waypoint_speed_scales=[1.5, .6], gripper_speed_scale=.6)
        result = self.execute(adapter, action)
        self.assertEqual(result["status"], "completed")
        segments = result["planned_trace"]["segments"]
        self.assertEqual([part["speed_scale"] for part in segments], [1.5, .6])
        self.assertEqual(result["waypoint_speed_scales"], [1.5, .6])
        self.assertEqual([part["speed_scale"] for part in result["waypoint_results"]], [1.5, .6])
        self.assertEqual(segments[0]["planned_motion_steps"], 4)
        self.assertEqual(segments[0]["planned_gripper_steps"], 0)
        self.assertGreaterEqual(segments[1]["planned_motion_steps"], 10)
        self.assertGreater(segments[1]["completed_gripper_steps"], 0)
        self.assertEqual(len(planner.pose_calls), 2)
        self.assertAlmostEqual(planner.pose_calls[1][0][0], -.3)
        start = np.zeros(6)
        for part in segments:
            q = np.array(part["q_rad"])
            bound = np.array(part["command_step_limits_rad"])
            self.assertTrue((np.abs(np.diff(np.vstack([start, q]), axis=0)) <= bound + 1e-12).all())
            start = q[-1]
        self.assertAlmostEqual(segments[0]["q_rad"][-1][0], -.3)
        self.assertAlmostEqual(segments[1]["q_rad"][-1][0], -.6)
        self.assertEqual(result["executed_native_steps"], len(rows))
        self.assertEqual(result["executed_control_steps"], len(rows) * 10)

    def test_omitted_list_matches_explicit_uniform_list_and_preview_binds_segment_speed(self):
        for top in (1., .6, 1.5):
            adapter, _, implicit = speed_adapter()
            other, _, explicit = speed_adapter()
            action = motion_action(angles=(-.2, -.4)) | dict(speed_scale=top)
            expected = self.execute(adapter, action)
            actual = self.execute(other, action | dict(waypoint_speed_scales=[top, top]))
            self.assertEqual(expected["status"], "completed")
            self.assertEqual(actual["status"], "completed")
            self.assertEqual(implicit, explicit)
            self.assertEqual(expected["waypoint_speed_scales"], [top, top])

        bad_adapter, _, bad_rows = speed_adapter(preview=True)
        bad_action = motion_action() | dict(speed_scale=.6, waypoint_speed_scales=[1.5])
        bad_preview = bad_adapter.preview_motion(bad_action)
        changed = deepcopy(bad_action) | dict(preview_plan_id=bad_preview["plan_id"], waypoint_speed_scales=[.6])
        self.assertEqual(self.execute(bad_adapter, changed)["reason"], "preview_plan_action_mismatch")
        self.assertEqual(bad_rows, [])

        adapter, _, rows = speed_adapter(preview=True)
        action = motion_action() | dict(speed_scale=.6, waypoint_speed_scales=[1.5])
        preview = adapter.preview_motion(action)
        self.assertEqual(preview["status"], "planned")
        self.assertEqual(preview["planned_trace"]["speed_scale"], 1.5)
        result = self.execute(adapter, action | dict(preview_plan_id=preview["plan_id"]))
        self.assertEqual(result["status"], "completed")
        actual_rows = [[*r["left_arm_joint_state"], *r["left_ee_joint_state"],
                        *r["right_arm_joint_state"], *r["right_ee_joint_state"]] for r in rows]
        np.testing.assert_array_equal(actual_rows, preview["native_rows"])

    def test_digest_feedback_rejection_and_shared_duration_cap(self):
        adapter, _, _ = speed_adapter()
        action = motion_action()
        motion = {key: action.get(key) for key in (
            "type", "mode", "frame_id", "tcp_config_id", "limits_profile", "path", "gripper_events")}
        motion.update(speed_scale=1., gripper_speed_scale=1.)
        previous = hashlib.sha256(json.dumps(motion, sort_keys=True, allow_nan=False).encode()).hexdigest()
        self.assertEqual(adapter._preview_action_digest(action), previous)

        for scales in ([1.], [.6]):
            fb_adapter, planner, rows = speed_adapter()
            fb_adapter.motion_execution_mode = "feedback"
            result = self.execute(fb_adapter, motion_action() | dict(waypoint_speed_scales=scales))
            self.assertEqual(result["reason"], "feedback_action_speed_unsupported")
            self.assertEqual(planner.pose_calls, [])
            self.assertEqual(rows, [])

        cap_adapter, _, cap_rows = speed_adapter(step_limit=1000)
        res = self.execute(cap_adapter, motion_action(angles=(-.5, 2.9), gripper=True) |
                           dict(waypoint_speed_scales=[2., .25]))
        self.assertEqual(res["reason"], "motion_duration_limit")
        self.assertEqual(res["failed_waypoint_index"], 1)
        self.assertEqual(res["executed_native_steps"], 5)
        self.assertTrue(all(row["left_ee_joint_state"] == [1.] for row in cap_rows))


class JointReplanSpeedTests(unittest.TestCase):
    @staticmethod
    def capture(diagnostics):
        def select(**kwargs):
            result = select_joint_replan(**kwargs)
            diagnostics.append(kwargs["diagnostic"])
            return result
        return select

    @staticmethod
    def velocity_limits(adapter):
        adapter.arm_velocity_step_limits = {side: np.full(6, .2) for side in ("left", "right")}

    def test_each_waypoint_cost_matches_its_dispatched_rate_and_step_count(self):
        for arm in ("left", "right"):
            for overrides, speeds, counts in (
                    ({"speed_scale": .8}, [.8, .8], [5, 6]),
                    ({"speed_scale": .8, "waypoint_speed_scales": [1.25, .8]}, [1.25, .8], [3, 6])):
                with self.subTest(arm=arm, speeds=speeds):
                    adapter, _, planner, dispatched = replan_adapter(arm)
                    self.velocity_limits(adapter)
                    action = motion_action(arm, (-.18, -.41)) | overrides
                    diagnostics = []
                    with patch("ni_rua.robot.motion.joint_replan.select_joint_replan", side_effect=self.capture(diagnostics)):
                        result = adapter.execute(action, lambda event: None, threading.Event())
                    self.assertEqual(result["status"], "completed")
                    self.assertEqual(len(planner.joint_calls), 2)
                    segments = result["planned_trace"]["segments"]
                    self.assertEqual([d["speed_scale"] for d in diagnostics], speeds)
                    self.assertEqual([d["candidate"]["resampled_rows"] for d in diagnostics], counts)
                    self.assertEqual([s["planned_motion_steps"] for s in segments], counts)
                    self.assertEqual(len(dispatched), sum(counts))
                    self.assertEqual(result["executed_native_steps"], sum(counts))
                    for d, speed in zip(diagnostics, speeds):
                        self.assertTrue(d["selected"])
                        self.assertLessEqual(d["candidate"]["max_dispatch_delta_rad"], .05 * speed + 1e-12)
                    np.testing.assert_allclose(planner.joint_calls[1][0], [-.18] + [0.] * 5)

    def test_preview_uses_override_for_replan_and_feedback_skips_replan(self):
        adapter, planner, dispatched = preview_adapter()
        adapter.joint_replan_enabled = True
        planner.wound = True
        self.velocity_limits(adapter)
        action = motion_action(angles=(-.18,)) | {"speed_scale": .8, "waypoint_speed_scales": [1.25]}
        diagnostics = []
        with patch("ni_rua.robot.motion.joint_replan.select_joint_replan", side_effect=self.capture(diagnostics)):
            preview = adapter.preview_motion(action)
        self.assertEqual(preview["status"], "planned")
        self.assertEqual(diagnostics[0]["speed_scale"], 1.25)
        self.assertEqual(preview["planned_native_steps"], 3)
        result = adapter.execute(action | {"preview_plan_id": preview["plan_id"]},
                                 lambda event: None, threading.Event())
        self.assertEqual(result["status"], "completed")
        np.testing.assert_array_equal([r["left_arm_joint_state"] for r in dispatched],
                                      np.array(preview["native_rows"])[:, :6])

        fb_adapter, fb_planner, fb_dispatched = feedback_adapter()
        fb_adapter.joint_replan_enabled = True
        fb_planner.wound = True
        fb_res = fb_adapter.execute(motion_action(), lambda event: None, threading.Event())
        self.assertEqual(fb_res["status"], "completed")
        self.assertEqual(fb_planner.joint_calls, [])
        self.assertTrue(fb_res["planned_trace"]["feedback_endpoint"]["changed"])
        np.testing.assert_allclose([r["left_arm_joint_state"][0] for r in fb_dispatched], -.125)


if __name__ == "__main__":
    unittest.main()
