"""Preview planning and exact reuse through the real CPU native-loop fixture."""

NI_RUA_TEST_TIER = "fast"

# === tests/test_motion_preview_backend.py ===

from copy import deepcopy
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock, PropertyMock, patch

import numpy as np

from ni_rua.robot.backends.robodojo.robodojo import RobotBackend, validate_preview_action
from support.motion_fixtures import replan_adapter, motion_action


def preview_adapter(**kwargs):
    adapter, robot, planner, dispatched = replan_adapter(enabled=False, wound=False, **kwargs)
    robot.entity_origin_pose = [0., 0., 0., 1., 0., 0., 0.]
    robot.ee_link_name = "tool"
    adapter.root = Path("/nonexistent/preview-fixture")
    # Matches periodic_pose: joint1 turns a unit-radius tool at z=1.
    joints = []
    for index in range(1, 7):
        origin = np.eye(4)
        if index == 2:
            origin[:3, 3] = [1., 0., 1.]
        joints.append(dict(name=f"joint{index}", type="continuous" if index == 1 else "fixed",
            parent="base" if index == 1 else f"link{index-1}",
            child="tool" if index == 6 else f"link{index}",
            T_parent_joint=origin.tolist(), axis=[0., 0., 1.]))
    adapter.robot_model = {"base_link": "base", "joints": joints, "geometry_scope": "fixture"}
    native_state = adapter.state

    def state():
        value = native_state()
        for arm in value["arms"].values():
            arm.update(joint_names=list(robot.arm_joints_name), source_kind="measurement")
            arm["ee_pose"].update(source_kind="measurement", frame_id="env_origin",
                                  tcp_config_id="arx-x5-link6-v1")
        return value

    adapter.state = state
    planner.robot_cfg = {"robot_cfg": {"kinematics": {
        "collision_spheres": {"link1": [{"center": [0., 0., 0.], "radius": .03}]},
        "self_collision_ignore": {"base": ["link1"]}}}}
    planner.scene_model = {"cuboid": {"table": {"dims": [3., 3., .05]}}}
    return adapter, planner, dispatched


class MotionPreviewBackendTests(unittest.TestCase):
    def test_preview_has_no_physics_or_dispatch_and_exposes_both_arm_geometry(self):
        adapter, planner, dispatched = preview_adapter()
        before = deepcopy(adapter.state())
        result = adapter.preview_motion(motion_action(gripper=True))
        self.assertEqual(result["status"], "planned")
        self.assertEqual(adapter.state(), before)
        self.assertEqual((adapter.control_step, adapter.env.take_action_cnt[0]), (0, 0))
        self.assertEqual(dispatched, [])
        self.assertEqual(len(planner.pose_calls), 1)
        self.assertEqual(set(result["before_state"]["arms"]), {"left", "right"})
        self.assertEqual(set(result["T_world_bases"]), {"left", "right"})
        self.assertTrue(result["preview_geometry"]["available"])
        self.assertEqual(result["T_world_bases_source_kind"], "derived_from_public_proprioception_and_robot_model")
        self.assertNotIn("planner_scene_model", result)
        self.assertEqual(result["robot_collision_model"]["spheres"]["link1"][0]["radius"], .03)
        self.assertEqual(result["planned_native_steps"], len(result["native_rows"]))
        self.assertGreater(result["planned_native_steps"], len(result["planned_trace"]["q_rad"]))

    def test_preview_derives_bases_from_public_pose_without_simulator_sources(self):
        from ni_rua.robot.model.geometry import fk
        adapter, planner, dispatched = preview_adapter()
        robot = adapter.robot.robot_list[0]

        class PoisonBase:
            def __array__(self, *args, **kwargs):
                raise AssertionError("simulator base must not be converted")

        robot.entity_origin_pose = PoisonBase()
        original_state = adapter.state
        expected = {}

        def state():
            value = original_state()
            for index, (side, measured) in enumerate(value["arms"].items()):
                measured["q_rad"][0] = .25 + index * .1
                base = np.eye(4)
                base[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
                base[:3, 3] = [index + .2, -.3, .4]
                tool = fk({**adapter.robot_model, "T_world_base": base}, measured)["tool"]
                angle = np.pi / 2 + measured["q_rad"][0]
                measured["ee_pose"].update(position_m=tool[:3, 3].tolist(),
                    quaternion_xyzw=[0., 0., float(np.sin(angle / 2)), float(np.cos(angle / 2))])
                expected[side] = base
            return value

        adapter.state = state
        with patch.object(type(planner), "scene_model", create=True, new_callable=PropertyMock,
                          side_effect=AssertionError("simulator scene must not be read")) as scene:
            result = adapter.preview_motion(motion_action())
        scene.assert_not_called()
        self.assertTrue(result["preview_geometry"]["available"])
        self.assertNotIn("planner_scene_model", result)
        for side, base in expected.items():
            np.testing.assert_allclose(result["T_world_bases"][side], base, atol=1e-12)
        self.assertEqual(dispatched, [])

    def test_execute_uses_exact_cached_rows_once_without_replanning_or_resampling(self):
        for arm in ("left", "right"):
            with self.subTest(arm=arm):
                adapter, planner, dispatched = preview_adapter(arm=arm)
                action = motion_action(arm=arm, gripper=True)
                preview = adapter.preview_motion(action)
                expected = deepcopy(preview["native_rows"])
                # A caller cannot mutate the native cache through preview output.
                preview["planned_trace"]["q_rad"][0][0] = 999
                preview["native_rows"][0][0] = 999
                adapter._plan_segment = Mock(side_effect=AssertionError("must not replan"))
                adapter._resample_plan = Mock(side_effect=AssertionError("must not resample"))
                executable = dict(action, preview_plan_id=preview["plan_id"], action_id="execute-preview",
                                  based_on_observation_id="refreshed-observation", timeout_s=60)
                result = adapter.execute(executable, lambda event: None, threading.Event())
                self.assertEqual(result["status"], "completed")
                actual = [row["left_arm_joint_state"] + row["left_ee_joint_state"] +
                          row["right_arm_joint_state"] + row["right_ee_joint_state"] for row in dispatched]
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(result["preview_plan_id"], preview["plan_id"])
                self.assertEqual(result["executed_native_steps"], len(expected))
                rejected = adapter.execute(executable, lambda event: None, threading.Event())
                self.assertEqual(rejected["status"], "rejected")
                self.assertIn("unknown_expired_or_consumed", rejected["reason"])
                self.assertEqual(rejected["executed_native_steps"], 0)
                self.assertEqual(len(dispatched), len(expected))

    def test_state_or_command_change_rejects_with_zero_motion_and_consumes_reference(self):
        for mutation in (lambda a: setattr(a, "control_step", 1),
                         lambda a: a.env.take_action_cnt.__setitem__(0, 1),
                         lambda a: a.last_dispatched_goals.__setitem__("right", .9),
                         lambda a: a.env.end_flag.__setitem__(0, True)):
            adapter, _, dispatched = preview_adapter()
            action = motion_action()
            preview = adapter.preview_motion(action)
            executable = dict(action, preview_plan_id=preview["plan_id"])
            mutation(adapter)
            rejected = adapter.execute(executable, lambda event: None, threading.Event())
            self.assertEqual(rejected["status"], "rejected")
            self.assertEqual(rejected["executed_native_steps"], 0)
            self.assertIn("stale_state", rejected["reason"])
            repeated = adapter.execute(executable, lambda event: None, threading.Event())
            self.assertEqual(repeated["status"], "rejected")
            self.assertIn("unknown_expired_or_consumed", repeated["reason"])
            self.assertEqual(dispatched, [])

    def test_goal_or_gripper_change_cannot_reuse_cached_plan(self):
        for changed in (motion_action(angles=(-.2,)), motion_action(gripper=True)):
            adapter, _, dispatched = preview_adapter()
            preview = adapter.preview_motion(motion_action())
            rejected = adapter.execute(dict(changed, preview_plan_id=preview["plan_id"]), lambda event: None, threading.Event())
            self.assertEqual(rejected["status"], "rejected")
            self.assertIn("action_mismatch", rejected["reason"])
            self.assertEqual(dispatched, [])

    def test_failed_planning_has_no_executable_plan_and_no_movement(self):
        adapter, _, dispatched = preview_adapter(fail_pose_at=1)
        result = adapter.preview_motion(motion_action())
        self.assertEqual((result["status"], result["reason"], result["plan_id"]), ("failed", "path_infeasible", None))
        self.assertEqual(result["planner_failure_stage"], "path_planning")
        self.assertEqual(dispatched, [])
        self.assertFalse(getattr(adapter, "_motion_previews", {}))

    def test_preview_boundaries_and_bounded_cache(self):
        adapter, _, dispatched = preview_adapter()
        with self.assertRaisesRegex(ValueError, "one_backend_free_waypoint"):
            adapter.preview_motion(motion_action(angles=(-.1, -.2)))
        adapter.observation_profile = "rgb_proprio_v1"
        with self.assertRaisesRegex(ValueError, "requires_calibrated_profile"):
            adapter.preview_motion(motion_action())
        adapter.observation_profile = "rgb_proprio_calibrated_v1"
        first = adapter.preview_motion(motion_action())
        for _ in range(16):
            adapter.preview_motion(motion_action())
        self.assertEqual(len(adapter._motion_previews), 16)
        rejected = adapter.execute(dict(motion_action(), preview_plan_id=first["plan_id"]), lambda event: None, threading.Event())
        self.assertEqual(rejected["status"], "rejected")
        self.assertIn("unknown_expired_or_consumed", rejected["reason"])
        self.assertEqual(dispatched, [])

    def test_backend_routes_validated_read_only_rpc(self):
        backend = object.__new__(RobotBackend)
        backend.operation_lock = threading.Lock()
        backend._request = Mock(return_value={"status": "planned"})
        action = motion_action()
        self.assertEqual(backend.preview_motion(action), {"status": "planned"})
        backend._request.assert_called_once_with("preview_motion", action, timeout=90)
        invalid = motion_action(angles=(-.1, -.2))
        with self.assertRaises(ValueError):
            validate_preview_action(invalid)
        with self.assertRaises(ValueError):
            backend.preview_motion(invalid)
        self.assertEqual(backend._request.call_count, 1)

# === tests/test_motion_speed_presets.py ===

from copy import deepcopy
import unittest

from ni_rua.core.contracts import digest, validate_action_speed
from ni_rua.robot.backends.robodojo.robodojo import validate_action as validate_native
from ni_rua.robot.motion.motion_speed import ARM_PRESETS, GRIPPER_SPEED_PRESETS, resolve_action_speed
from support.motion_fixtures import motion_action
from support.motion_fixtures import wait_action


def move(**selection):
    return dict(type="EEAction", path=dict(kind="backend_free", waypoints=[{"left": {}}]),
                gripper_events=[], **selection)


def grip(**selection):
    return dict(type="EEAction", path=dict(kind="hold", waypoints=[]),
                gripper_events=[dict(arm="left", phase="after_motion", goal_normalized=0.)],
                **selection)


class MotionSpeedPresetTests(unittest.TestCase):
    def test_fixed_arm_presets_override_run_default_and_resolve_to_float(self):
        self.assertEqual(ARM_PRESETS, dict(slower=.5, slow=.8, careful=.9, normal=1., fast=1.25, faster=1.5))
        for preset, scale in ARM_PRESETS.items():
            for default in (1., 1.25, 1.5, 2.):
                result = resolve_action_speed(move(speed_preset=preset), default)
                self.assertEqual(result["speed_scale"], scale)
                self.assertIs(type(result["speed_scale"]), float)
                self.assertNotIn("speed_preset", result)
                self.assertNotIn("gripper_speed_scale", result)

    def test_jaw_presets_are_independent_from_arm_numeric_or_preset(self):
        self.assertEqual(GRIPPER_SPEED_PRESETS, dict(slower=.5, slow=.8, normal=1.))
        for preset, scale in GRIPPER_SPEED_PRESETS.items():
            action = move(speed_preset="faster", gripper_speed_preset=preset)
            action["gripper_events"] = grip()["gripper_events"]
            result = resolve_action_speed(action, 2.)
            self.assertEqual(result["speed_scale"], 1.5)
            self.assertEqual(result["gripper_speed_scale"], scale)
            self.assertNotIn("gripper_speed_preset", result)
            action.pop("speed_preset")
            action["speed_scale"] = .4
            self.assertEqual(resolve_action_speed(action)["speed_scale"], .4)

    def test_default_only_applies_to_omitted_arm_selection(self):
        self.assertEqual(resolve_action_speed(move(), 1.5)["speed_scale"], 1.5)
        self.assertEqual(resolve_action_speed(move(speed_scale=1), 1.5)["speed_scale"], 1.)
        self.assertEqual(resolve_action_speed(move(speed_preset="normal"), 1.5)["speed_scale"], 1.)
        action = move()
        action["gripper_events"] = grip()["gripper_events"]
        result = resolve_action_speed(action, 2.)
        self.assertEqual((result["speed_scale"], result["gripper_speed_scale"]), (2., 1.))

    def test_numeric_ranges_and_float_canonicalization(self):
        for number in (.25, .6, 1, 1.5, 2):
            result = resolve_action_speed(move(speed_scale=number))
            self.assertEqual(result["speed_scale"], number)
            self.assertIs(type(result["speed_scale"]), float)
        for number in (.25, .6, 1):
            result = resolve_action_speed(grip(gripper_speed_scale=number), 2.)
            self.assertEqual(result["gripper_speed_scale"], number)
            self.assertIs(type(result["gripper_speed_scale"]), float)
            self.assertNotIn("speed_scale", result)

    def test_same_axis_conflict_rejected_even_if_values_agree(self):
        for action in (move(speed_preset="normal", speed_scale=1),
                       grip(gripper_speed_preset="normal", gripper_speed_scale=1)):
            with self.assertRaisesRegex(ValueError, "conflicting_"):
                resolve_action_speed(action)

    def test_explicit_invalid_preset_and_numeric_never_fall_back(self):
        for invalid in (None, True, 1, [], {}, "", "FAST", "invalid"):
            with self.subTest(preset=invalid), self.assertRaisesRegex(ValueError, "invalid_speed_preset"):
                resolve_action_speed(move(speed_preset=invalid), 1.5)
            with self.subTest(jaw_preset=invalid), self.assertRaisesRegex(ValueError, "invalid_gripper_speed_preset"):
                resolve_action_speed(grip(gripper_speed_preset=invalid), 1.5)
        for invalid in (None, True, False, "1", float("nan"), float("inf"), 0, .24, 2.01):
            with self.subTest(scale=invalid), self.assertRaisesRegex(ValueError, "invalid_speed_scale"):
                resolve_action_speed(move(speed_scale=invalid), 1.5)
        for invalid in (None, True, "1", float("nan"), .24, 1.01):
            with self.subTest(jaw_scale=invalid), self.assertRaisesRegex(ValueError, "invalid_gripper_speed_scale"):
                resolve_action_speed(grip(gripper_speed_scale=invalid), 1.5)
        for invalid in ("fast", "faster"):
            with self.assertRaisesRegex(ValueError, "invalid_gripper_speed_preset"):
                resolve_action_speed(grip(gripper_speed_preset=invalid))

    def test_invalid_default_is_rejected(self):
        for invalid in (None, True, "1", float("nan"), float("inf"), .24, 2.01):
            with self.subTest(default=invalid), self.assertRaisesRegex(ValueError, "invalid_default_motion_speed_scale"):
                resolve_action_speed(move(), invalid)

    def test_wait_and_policy_unchanged_and_reject_all_explicit_selection_fields(self):
        wait = dict(type="EEAction", path=dict(kind="hold", waypoints=[]), hold_native_steps=5)
        prefix = dict(type="JointPrefixAction", actions=[[0.] * 14])
        for action in (wait, prefix):
            self.assertEqual(resolve_action_speed(action, 2.), action)
            for field, value in (("speed_scale", 1), ("speed_preset", "normal"),
                                 ("gripper_speed_scale", 1), ("gripper_speed_preset", "normal")):
                with self.subTest(kind=action["type"], field=field), self.assertRaisesRegex(ValueError, "unsupported_for_wait_or_joint_prefix"):
                    resolve_action_speed(action | {field: value}, 2.)

    def test_hold_has_only_jaw_default_and_empty_hold_is_unchanged(self):
        action = grip()
        self.assertEqual(resolve_action_speed(action, 2.), action | dict(gripper_speed_scale=1.))
        empty = dict(type="EEAction", path=dict(kind="hold", waypoints=[]), gripper_events=[])
        self.assertEqual(resolve_action_speed(empty, 2.), empty)
        for action in (grip(speed_preset="slow"), grip(speed_scale=.6)):
            with self.assertRaisesRegex(ValueError, "speed_scale_requires_motion_path"):
                resolve_action_speed(action)
        for action in (move(gripper_speed_preset="slow"), move(gripper_speed_scale=.6)):
            with self.assertRaisesRegex(ValueError, "gripper_speed_scale_requires_gripper_event"):
                resolve_action_speed(action)

    def test_no_mutation_and_repeated_resolution_preserves_effective_speed(self):
        action = move(speed_preset="fast")
        before = deepcopy(action)
        result = resolve_action_speed(action, 2.)
        self.assertEqual(action, before)
        self.assertIsNot(result, action)
        # The helper never mutates nested structures and deliberately copies
        # only the top-level selection fields; it is not a general deep clone.
        self.assertIs(result["path"], action["path"])
        self.assertEqual(resolve_action_speed(result, 1.), result)

    def test_equivalent_preset_numeric_default_have_identical_canonical_digest(self):
        explicit = resolve_action_speed(move(speed_scale=1))
        preset = resolve_action_speed(move(speed_preset="normal"), 2.)
        omitted = resolve_action_speed(move(), 1.)
        self.assertEqual(digest(explicit), digest(preset))
        self.assertEqual(digest(explicit), digest(omitted))

    def test_numeric_backend_boundary_rejects_unresolved_presets(self):
        for action in (move(speed_preset="normal"), grip(gripper_speed_preset="normal")):
            with self.assertRaisesRegex(ValueError, "unresolved_action_speed_preset"):
                validate_action_speed(action)
            validate_action_speed(resolve_action_speed(action))


def _waypoint_motion(count=2, **selections):
    return motion_action(angles=tuple(-.1 * (index + 1) for index in range(count))) | selections


class ActionSpeedInterfaceTests(unittest.TestCase):
    def test_invalid_rates_and_contexts_rejected_before_execution(self):
        for field, bad_values in (("speed_scale", [True, None, "fast", 0, .24, 2.01, float("nan"), float("inf")]),
                                  ("gripper_speed_scale", [False, None, 0, 1.01, float("nan")])):
            for value in bad_values:
                with self.subTest(field=field, value=value):
                    action = motion_action(gripper=True)
                    action[field] = value
                    with self.assertRaises(ValueError):
                        validate_action_speed(action)
                    with self.assertRaises(ValueError):
                        validate_native(action)
        for action in (dict(wait_action(1), speed_scale=1),
                       dict(wait_action(1), gripper_speed_scale=.5),
                       dict(type="JointPrefixAction", speed_scale=.5),
                       dict(motion_action(), gripper_speed_scale=.5)):
            with self.assertRaises(ValueError):
                validate_action_speed(action)

    def test_unsupported_backend_refuses_nondefault_speed(self):
        action = dict(motion_action(), speed_scale=.5)
        for caps in ({}, {"action_speed": {"available": False}}):
            with self.assertRaisesRegex(ValueError, "unsupported_capability:action_speed"):
                validate_action_speed(action, caps)
        action["speed_scale"] = 1
        validate_action_speed(action, {})


class WaypointSpeedResolutionTests(unittest.TestCase):
    def test_presets_follow_waypoint_order_with_fixed_values_and_null_inheritance(self):
        for base, inherited in (({}, 1.25), ({"speed_preset": "normal"}, 1.),
                                ({"speed_scale": .8}, .8)):
            action = _waypoint_motion(4, waypoint_speed_presets=["faster", None, "slow", "normal"], **base)
            result = resolve_action_speed(action, 1.25)
            self.assertEqual(result["waypoint_speed_scales"], [1.5, inherited, .8, 1.])
            self.assertTrue(all(type(value) is float for value in result["waypoint_speed_scales"]))
            self.assertNotIn("waypoint_speed_presets", result)
            validate_native(result)

    def test_numeric_boundaries_and_null_inherit_without_mutating_inputs(self):
        action = _waypoint_motion(4, speed_scale=.75, waypoint_speed_scales=[.25, None, 1, 2])
        original = deepcopy(action)
        result = resolve_action_speed(action, 1.5)
        self.assertEqual(result["waypoint_speed_scales"], [.25, .75, 1., 2.])
        self.assertEqual(action, original)
        result["waypoint_speed_scales"][0] = 1.
        self.assertEqual(action, original)
        canonical = resolve_action_speed(action, 1.5)
        self.assertEqual(resolve_action_speed(canonical, 2.), canonical)
        same = resolve_action_speed(_waypoint_motion(4, speed_scale=.75,
                                                     waypoint_speed_scales=[.25, .75, 1., 2.]))
        self.assertEqual(digest(canonical), digest(same))

    def test_omission_preserves_legacy_shape_and_jaw_selection_is_independent(self):
        result = resolve_action_speed(_waypoint_motion(), 1.25)
        self.assertNotIn("waypoint_speed_scales", result)
        self.assertEqual(result["speed_scale"], 1.25)
        action = _waypoint_motion(2, waypoint_speed_presets=["fast", "slow"], gripper_speed_preset="slow")
        action["gripper_events"] = [dict(arm="left", phase="after_motion", goal_normalized=0.)]
        result = resolve_action_speed(action, 1.5)
        self.assertEqual(result["waypoint_speed_scales"], [1.25, .8])
        self.assertEqual(result["gripper_speed_scale"], .8)

    def test_conflicts_lengths_container_types_and_invalid_entries_fail_closed(self):
        with self.assertRaises(ValueError):
            resolve_action_speed(_waypoint_motion(2, waypoint_speed_presets=["normal", "normal"],
                                                  waypoint_speed_scales=[1., 1.]))
        for field in ("waypoint_speed_presets", "waypoint_speed_scales"):
            for value in (None, True, "slow", {}, (), [], [None], [None] * 3):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    resolve_action_speed(_waypoint_motion(**{field: value}))
        for field, invalid in (
                ("waypoint_speed_presets", (True, 1, .6, "", "SLOW", "unknown", [], {})),
                ("waypoint_speed_scales", (True, False, "slow", "1", .24, 2.01, float("nan"), float("inf"), [], {}))):
            for value in invalid:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    resolve_action_speed(_waypoint_motion(**{field: [None, value]}))

    def test_malformed_waypoint_container_and_unsupported_actions_fail_closed(self):
        for points in (None, 1, "invalid", {}, ()):
            action = _waypoint_motion(1, waypoint_speed_scales=[1.])
            action["path"]["waypoints"] = points
            for check in (resolve_action_speed, validate_action_speed, validate_native):
                with self.subTest(points=points, boundary=check.__name__), self.assertRaises(ValueError):
                    check(action)
        gripper = wait_action()
        gripper.pop("hold_native_steps")
        gripper["gripper_events"] = [dict(arm="left", phase="after_motion", goal_normalized=0.)]
        for action in (wait_action(1), dict(type="JointPrefixAction", actions=[[0.] * 14]), gripper):
            for field in ("waypoint_speed_presets", "waypoint_speed_scales"):
                with self.subTest(kind=action.get("type"), field=field), self.assertRaises(ValueError):
                    resolve_action_speed(action | {field: [None]})

    def test_numeric_boundary_requires_resolved_entries_and_explicit_capability(self):
        for invalid in (_waypoint_motion(1, waypoint_speed_presets=["slow"]),
                        _waypoint_motion(1, waypoint_speed_scales=[None]),
                        _waypoint_motion(1, waypoint_speed_scales=[True]),
                        _waypoint_motion(1, waypoint_speed_scales=[float("nan")])):
            with self.assertRaises(ValueError):
                validate_native(invalid)
        action = resolve_action_speed(_waypoint_motion(1, waypoint_speed_presets=["normal"]))
        for caps in ({}, {"action_speed": {"available": True}},
                     {"action_speed": {"available": True, "waypoint_speed_scales": {"available": False}}}):
            with self.assertRaises(ValueError):
                validate_action_speed(action, caps)
        validate_action_speed(action, {"action_speed": {"available": True,
                                                       "waypoint_speed_scales": {"available": True}}})
        validate_action_speed(resolve_action_speed(_waypoint_motion(1)), {})

# === tests/test_native_runtime.py ===

from types import SimpleNamespace
import threading
import unittest

import numpy as np

from ni_rua.robot.backends.robodojo.robodojo import validate_joint_prefix
from ni_rua.robot.backends.robodojo.robot_worker import RoboDojoAdapter, host_articulation_properties


class NativeRuntimeCheck(unittest.TestCase):
    def test_articulation_numpy_metadata_reads_copy_to_host_and_restore(self):
        class DeviceArray:
            def __init__(self, value): self.value = np.asarray(value)
            def __array__(self, *args, **kwargs): raise TypeError("CUDA tensor requires cpu()")
            def detach(self): return self
            def cpu(self): return SimpleNamespace(numpy=lambda: self.value)

        for device_values in (False, True):
            for fail in (False, True):
                with self.subTest(device_values=device_values, fail=fail):
                    wrap = DeviceArray if device_values else np.asarray
                    gains = (wrap([[4.]]), wrap([[5.]]))
                    view = SimpleNamespace(**{name: (lambda: wrap([[3.]])) for name in (
                        "get_dof_types", "get_dof_limits", "get_drive_types", "get_max_efforts", "get_joint_max_velocities")},
                        get_gains=lambda: gains)
                    class NativeObject:
                        _articulation_view = view
                        @property
                        def dof_properties(self):
                            result = np.zeros(1, dtype=[("effort", float), ("stiffness", float), ("damping", float)])
                            result["effort"] = np.asarray(self._articulation_view.get_max_efforts())[0]
                            stiffness, damping = self._articulation_view.get_gains()
                            result["stiffness"], result["damping"] = np.asarray(stiffness)[0], np.asarray(damping)[0]
                            if fail: raise RuntimeError("native getter failed")
                            return result
                    class SceneObject(NativeObject): pass
                    original = dict(vars(view))
                    if device_values:
                        with self.assertRaisesRegex(TypeError, "CUDA"):
                            SceneObject().dof_properties
                    with host_articulation_properties(SceneObject):
                        if fail:
                            with self.assertRaisesRegex(RuntimeError, "native getter failed"):
                                SceneObject().dof_properties
                        else:
                            self.assertEqual(SceneObject().dof_properties.tolist(), [(3., 4., 5.)])
                        self.assertEqual(vars(view), original)
                    self.assertNotIn("dof_properties", vars(SceneObject))
                    self.assertEqual(vars(view), original)

    def adapter(self, *, step_limit=3, success_at=None):
        adapter = object.__new__(RoboDojoAdapter)
        adapter.np, adapter.control_hz = np, 25
        adapter.control_step = adapter.evaluator_actions = 0
        adapter.versions = {}
        adapter.observation_profile = "rgb_proprio_calibrated_v1"
        adapter.goals = adapter.last_dispatched_goals = {"left": 1., "right": 1.}
        adapter.robot = SimpleNamespace(robot_list=[], robot_key=[])
        adapter.env = SimpleNamespace(end_flag=[False], success=[True], take_action_cnt=[0], step_lim=step_limit)
        adapter.state = lambda: {"arms": {side: {"q_rad": [0.] * 6} for side in ("left", "right")}}
        adapter.observe = lambda: {"robot_state": adapter.state()}
        def take_action(action):
            self.assertFalse(adapter.env.end_flag[0], "no native call may follow terminal")
            adapter.env.take_action_cnt[0] += 1
            adapter.control_step += 10
            count = adapter.env.take_action_cnt[0]
            if count == success_at or count == step_limit:
                adapter.env.end_flag[0] = True
                adapter.env.success[0] = count == success_at
        adapter.env.take_action = take_action
        return adapter

    def test_native_terminal_mid_chunk_and_already_terminal(self):
        for success_at, count, reason in ((None, 3, "native_step_limit"), (2, 2, "native_success")):
            with self.subTest(success_at=success_at):
                adapter = self.adapter(success_at=success_at)
                action = dict(type="JointPrefixAction", actions=[[0.] * 14] * 50,
                              policy_provenance={"fixture": "terminal_semantics"})
                result = adapter.execute(action, lambda value: None, threading.Event())
                self.assertEqual(result["status"], "stopped")
                self.assertEqual(result["reason"], "episode_ended")
                self.assertEqual(result["executed_native_steps"], count)
                self.assertEqual(result["executed_control_steps"], 10 * count)
                self.assertEqual(result["episode_terminal"]["reason"], reason)
                self.assertEqual(result["stop_confirmation"]["settling_steps"], 0)
                again = adapter.execute(action, lambda value: None, threading.Event())
                self.assertEqual(again["executed_native_steps"], 0)
                self.assertEqual(again["executed_control_steps"], 0)

    def test_polyline_variation_preserves_detour_and_final_target(self):
        adapter = self.adapter()
        adapter.arm_step_limits = {"left": np.array([.05])}
        result = adapter._resample_plan(np.array([[0.], [.1], [0.]]), "left")
        np.testing.assert_allclose(result[:, 0], [.05, .1, .05, 0.])

    def test_partial_native_fault_preserves_started_action_and_ticks(self):
        adapter = self.adapter()
        def fail_after_motion(action):
            adapter.env.take_action_cnt[0] += 1
            adapter.control_step += 4
            raise RuntimeError("injected native error after four physics ticks")
        adapter.env.take_action = fail_after_motion
        result = adapter.execute(dict(type="JointPrefixAction", actions=[[0.] * 14],
                                     policy_provenance={"fixture": "partial_fault"}), lambda value: None, threading.Event())
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["completed_native_steps"], 0)
        self.assertEqual(result["executed_native_steps"], 1)
        self.assertEqual(result["executed_control_steps"], 4)
        self.assertIn("native_action_failed", adapter.evaluator_failure)

    def test_exhausted_ik_retries_without_plan_status_are_classified(self):
        for results, expected in (([[[False]]] * 5, "inverse_kinematics"),
                                  ([[[False]], [[True]], [[False]]], "path_planning"),
                                  ([[[False]], None], "path_planning"),
                                  ([[]], "path_planning"), ([], "path_planning")):
            with self.subTest(results=results):
                adapter = self.adapter()
                adapter.robot.get_robot_by_arm_name = lambda arm: object()
                adapter.capabilities = lambda: {"collision_scope": "fixture"}
                def no_plan(robot, target, diagnostics, waypoint_index):
                    # A previous waypoint must not contaminate this failure stage.
                    diagnostics.append(dict(waypoint_index=-1, stage="plan_pose", status="IK_FAIL"))
                    diagnostics.extend(dict(waypoint_index=waypoint_index, stage="ik", success=value)
                                       for value in results)
                    diagnostics.append(dict(waypoint_index=waypoint_index, stage="plan_pose",
                                            status=None, returned_none=True))
                    return None
                adapter._plan_segment = no_plan
                action = dict(mode="absolute", frame_id="env_origin", tcp_config_id="arx-x5-link6-v1",
                              limits_profile="robodojo-native-v1", timeout_s=10, gripper_events=[],
                              path=dict(kind="backend_free", waypoints=[dict(left=dict(
                                  position_m=[-.3, 0, 1], quaternion_xyzw=[0, 0, 0, 1]))]))
                receipt = adapter.execute(action, lambda value: None, threading.Event())
                self.assertEqual(receipt["status"], "failed")
                self.assertEqual(receipt["reason"], "path_infeasible")
                self.assertEqual(receipt["planner_failure_stage"], expected)
                self.assertEqual(receipt["executed_native_steps"], 0)
                self.assertEqual(receipt["executed_control_steps"], 0)
                self.assertNotIn("planning_diagnostics", receipt)

    def test_prefix_boundary_rejects_base_output_and_empty_chunks(self):
        for rows in ([], [[0.] * 32], [[0.] * 14] * 51, [[float("nan")] * 14]):
            with self.assertRaises(ValueError):
                validate_joint_prefix(dict(type="JointPrefixAction", actions=rows, policy_provenance={"source": "test"}))

# === tests/test_native_validity.py ===

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call

from ni_rua.robot.backends.robodojo.robot_worker import RoboDojoAdapter


class NativeEnvironment(SimpleNamespace):
    def mark_env_unstable(self, env_idx):
        # EvalEnv records this independently of success and reward, then
        # excludes the environment from its aggregate evaluation results.
        self.unstable_envs.add(env_idx)


class NativeValidityTests(unittest.TestCase):
    def adapter(self, directory, *, success=False, ended=True):
        adapter = object.__new__(RoboDojoAdapter)
        adapter.directory = Path(directory)
        adapter.task, adapter.seed, adapter.layout_id = "fixture", 0, 4
        adapter.episode_index = adapter.evaluator_actions = adapter.control_step = 0
        adapter.versions, adapter.evaluator_failure = {}, None
        adapter.layout_acceptance = dict(valid=True, stable=True, unstable_envs=[],
                                         authority="accepted reset fixture")
        reward = float(success)
        reward_manager = SimpleNamespace(
            get_reward=Mock(return_value=[reward]),
            get_score=Mock(return_value=[100. if success else 50.]),
            step=Mock(side_effect=AssertionError("validity must not advance predicates")),
            check_once=Mock(side_effect=AssertionError("validity must not query predicates")))
        layout_manager = SimpleNamespace(layout_valid=[True],
            check_layout_stability=Mock(side_effect=AssertionError("validity must not rerun stability")))
        adapter.env = NativeEnvironment(end_flag=[ended], success=[success],
            take_action_cnt=[1], step_lim=1600, unstable_envs=set(),
            scene_manager=SimpleNamespace(layout_manager=layout_manager),
            reward_manager=reward_manager, get_score=Mock())
        return adapter

    def test_delayed_native_instability_invalidates_accepted_reset_without_rescoring(self):
        for success in (False, True):
            with self.subTest(success=success), tempfile.TemporaryDirectory() as directory:
                adapter = self.adapter(directory, success=success)
                reset_acceptance = deepcopy(adapter.layout_acceptance)
                before = adapter.evaluate()
                self.assertTrue(before["valid_episode"])
                adapter.env.mark_env_unstable(0)
                after = adapter.evaluate()
                self.assertFalse(after["valid_episode"])
                self.assertEqual(adapter.layout_acceptance, reset_acceptance)
                self.assertEqual(after["layout_acceptance"], reset_acceptance)
                for key in ("task_success", "native_reward", "native_process_score", "normalized_score",
                            "episode_terminal", "success_authority", "final_check"):
                    self.assertEqual(after[key], before[key], key)
                self.assertEqual(after["native_validity"]["unstable_envs"], [0])
                self.assertTrue(after["native_validity"]["unstable"])
                self.assertFalse(before["native_validity"]["unstable"])
                saved = json.loads(Path(after["evaluation_path"]).read_text())
                self.assertFalse(saved["valid_episode"])
                self.assertEqual(saved["native_validity"], after["native_validity"])
                # Exactly the pre-existing evaluator calls, with the same
                # terminal-safe arguments. Reading validity adds no calls.
                rm = adapter.env.reward_manager
                self.assertEqual(rm.get_reward.call_args_list, [call(final_check=False)] * 2)
                self.assertEqual(rm.get_score.call_args_list, [call([float(success)])] * 2)
                rm.step.assert_not_called()
                rm.check_once.assert_not_called()
                adapter.env.get_score.assert_not_called()
                adapter.env.scene_manager.layout_manager.check_layout_stability.assert_not_called()

    def test_live_layout_invalidation_and_reset_rejection_both_exclude_sample(self):
        for reset_valid, reset_stable, live_valid in (
                (True, True, False), (False, True, True), (True, False, True)):
            with self.subTest(reset_valid=reset_valid, reset_stable=reset_stable, live_valid=live_valid), \
                    tempfile.TemporaryDirectory() as directory:
                adapter = self.adapter(directory)
                adapter.layout_acceptance.update(valid=reset_valid, stable=reset_stable)
                adapter.env.scene_manager.layout_manager.layout_valid[0] = live_valid
                result = adapter.evaluate()
                self.assertFalse(result["valid_episode"])
                self.assertEqual(result["native_validity"]["layout_valid"], live_valid)

    def test_live_snapshot_is_read_only_and_scoped_to_native_environment_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.adapter(directory)
            adapter.env.mark_env_unstable(4)
            adapter.env.mark_env_unstable(2)
            first = adapter._native_validity()
            self.assertEqual(first["unstable_envs"], [2, 4])
            self.assertTrue(first["valid"])
            adapter.env.mark_env_unstable(0)
            second = adapter._native_validity()
            self.assertFalse(second["valid"])
            self.assertEqual(first["unstable_envs"], [2, 4])
            self.assertEqual(adapter.env.unstable_envs, {0, 2, 4})
            adapter.env.reward_manager.get_reward.assert_not_called()
            adapter.env.reward_manager.get_score.assert_not_called()
            adapter.env.reward_manager.step.assert_not_called()
            adapter.env.reward_manager.check_once.assert_not_called()
            adapter.env.scene_manager.layout_manager.check_layout_stability.assert_not_called()

    def test_nonterminal_evaluation_preserves_existing_final_check_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.adapter(directory, ended=False)
            adapter.env.mark_env_unstable(0)
            result = adapter.evaluate(final_check=True)
            self.assertFalse(result["valid_episode"])
            self.assertTrue(result["final_check"])
            self.assertEqual(result["success_authority"], "native_final_check")
            adapter.env.reward_manager.get_reward.assert_called_once_with(final_check=True)
            adapter.env.reward_manager.get_score.assert_called_once_with([0.])

# === tests/test_robot_process_deadline.py ===

import os
import sys
import tempfile
import time
import unittest

from ni_rua.robot.backends.robodojo.robot_process import RobotProcess
from ni_rua.robot.backends.robodojo.worker_protocol import WIRE_PREFIX


class RobotProcessDeadlineTest(unittest.TestCase):
    def worker(self, directory, *, stall=False):
        source = f'''
import json, sys, time
prefix = {WIRE_PREFIX!r}
executions = 0
for line in sys.stdin:
    message = json.loads(line)
    operation = message['op']
    if operation == 'execute': executions += 1
    result = {{'closed': True}} if operation == 'close' else {{'size': len((message.get('payload') or {{}}).get('text', ''))}}
    if operation == 'execute': result['executions'] = executions
    print(prefix + json.dumps({{'id': message['id'], 'result': result}}), flush=True)
    if operation == 'close': break
    if operation == 'ready' and {stall!r}: time.sleep(30)
'''
        robot = RobotProcess(directory)
        robot.start_process([sys.executable, '-u', '-c', source], cwd=directory, environment=os.environ.copy())
        def cleanup():
            if robot.process.poll() is None:
                robot.process.kill()
                robot.process.wait(timeout=3)
            robot.close()
        self.addCleanup(cleanup)
        return robot

    def test_request_before_worker_start_is_a_typed_error_not_a_failure(self):
        # Regression: the gateway console polled /v1/calibration/summary during startup before the
        # SO-101 worker was spawned; _send dereferenced process=None and raised AttributeError.
        with tempfile.TemporaryDirectory() as directory:
            robot = RobotProcess(directory)
            with self.assertRaisesRegex(RuntimeError, '^robot_worker_not_running$'):
                robot._request('operator', {'command': 'calib_workflow_status'}, timeout=1)
            self.assertFalse(robot.pending)
            self.assertIsNone(robot.failure)
            self.assertTrue(robot.close()['not_started'])

    def test_full_pipe_times_out_and_close_reaps_without_replaying_request(self):
        with tempfile.TemporaryDirectory() as directory:
            robot = self.worker(directory, stall=True)
            started = time.monotonic()
            with self.assertRaisesRegex(TimeoutError, 'status unknown: worker request write deadline; do not replay action'):
                robot._request('execute', {'text': 'x' * (1024 * 1024)}, timeout=.1)
            self.assertLess(time.monotonic() - started, 1)
            self.assertFalse(robot.pending)
            self.assertIsNotNone(robot.failure)
            self.assertIsNone(robot.process.poll())
            started = time.monotonic()
            report = robot.close()
            self.assertLess(time.monotonic() - started, 4)
            self.assertTrue(report['process_exit_confirmed'])
            self.assertFalse(report['native_close_acknowledged'])
            self.assertEqual(report['forced_termination'], 'SIGTERM')

    def test_waiting_for_another_writer_uses_the_same_request_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            robot = self.worker(directory)
            with robot.write_lock:
                started = time.monotonic()
                with self.assertRaisesRegex(TimeoutError, 'worker request write deadline'):
                    robot._request('observe', timeout=.05)
                self.assertLess(time.monotonic() - started, .5)
            self.assertFalse(robot.pending)
            self.assertIsNotNone(robot.failure)

    def test_large_request_is_written_once_and_native_close_still_acknowledges(self):
        with tempfile.TemporaryDirectory() as directory:
            robot = self.worker(directory)
            self.assertEqual(robot._request('execute', {'text': 'x' * (1024 * 1024)}, timeout=2),
                             {'size': 1024 * 1024, 'executions': 1})
            self.assertIsNone(robot.failure)
            report = robot.close()
            self.assertTrue(report['native_close_acknowledged'])
            self.assertTrue(report['process_exit_confirmed'])
            self.assertIsNone(report['forced_termination'])

# === tests/test_synchronized_gripper.py ===

from copy import deepcopy
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ni_rua.core.contracts import validate_action as validate_contract_action
from ni_rua.robot.backends.robodojo.robodojo import validate_action
from support.motion_fixtures import motion_action, replan_adapter
from support.motion_fixtures import preview_adapter as _sync_preview_adapter
from support.motion_fixtures import IDENTITY, fake_adapter, wait_action
from robot.motion.test_motion_guards_and_wait import ee_action, tracking_adapter


def synchronized(action, *, goal=0., start=0., end=1.):
    action = deepcopy(action)
    arm = next(iter(action["path"]["waypoints"][-1]))
    action["gripper_events"] = [dict(arm=arm, phase="during_motion", goal_normalized=goal,
                                     start_fraction=start, end_fraction=end)]
    return action


def jaw_commands(rows, arm):
    return np.array([row[f"{arm}_ee_joint_state"][0] for row in rows])


class SynchronizedGripperRuntimeTests(unittest.TestCase):
    def execute(self, adapter, action, cancellation=None):
        return adapter.execute(action, lambda event: None, cancellation or threading.Event())

    def assert_schedule(self, rows, trace, arm, initial, goal, start, end, original_steps):
        count = len(rows)
        fractions = np.arange(1, count + 1) / count
        expected = initial + (goal - initial) * np.clip((fractions - start) / (end - start), 0, 1)
        actual = jaw_commands(rows, arm)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        self.assertLessEqual(np.max(np.abs(np.diff(np.r_[initial, actual]))), .18 + 1e-12)
        self.assertAlmostEqual(actual[-1], goal)
        self.assertEqual(len(trace["q_rad"]), count)
        metadata = trace["gripper_motion"]
        np.testing.assert_allclose(metadata["goal_normalized_by_step"], expected, atol=1e-12)
        self.assertEqual(metadata["original_motion_steps"], original_steps)
        self.assertEqual(metadata["scheduled_motion_steps"], count)
        self.assertEqual(metadata["timing_basis"], "final_segment_native_steps")

    def test_left_right_open_close_interpolate_latched_command_without_extra_rows(self):
        for arm in ("left", "right"):
            for initial, goal in ((1., 0.), (0., 1.)):
                with self.subTest(arm=arm, initial=initial):
                    adapter, rows = tracking_adapter(positions=np.zeros((20, 6)))
                    adapter.last_dispatched_goals[arm] = initial
                    action = synchronized(ee_action(arm), goal=goal)
                    result = self.execute(adapter, action)
                    self.assertEqual((result["status"], result["reason"]), ("completed", None))
                    self.assertEqual(len(rows), 20)
                    self.assertEqual(result["executed_native_steps"], 20)
                    self.assert_schedule(rows, result["planned_trace"], arm, initial, goal, 0., 1., 20)
                    inactive = "right" if arm == "left" else "left"
                    np.testing.assert_array_equal(jaw_commands(rows, inactive), .3 if inactive == "right" else 1.)
                    self.assertEqual(adapter.last_dispatched_goals[arm], goal)

    def test_delayed_interval_retains_then_ramps_then_holds_target(self):
        adapter, rows = tracking_adapter(positions=np.zeros((20, 6)))
        result = self.execute(adapter, synchronized(ee_action(), start=.25, end=.75))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(rows), 20)
        self.assert_schedule(rows, result["planned_trace"], "left", 1., 0., .25, .75, 20)
        np.testing.assert_array_equal(jaw_commands(rows[:5], "left"), 1.)
        np.testing.assert_array_equal(jaw_commands(rows[14:], "left"), 0.)

    def test_short_motion_retimes_uniformly_and_preserves_all_original_knots(self):
        original = np.array([[0.] * 6, [.1, 0., .2, 0., 0., 0.], [.2, .1, 0., 0., 0., 0.]])
        adapter, rows = tracking_adapter(positions=original)
        result = self.execute(adapter, synchronized(ee_action(), start=.2, end=.8))
        self.assertEqual(result["status"], "completed")
        self.assertGreater(len(rows), len(original))
        self.assert_schedule(rows, result["planned_trace"], "left", 1., 0., .2, .8, len(original))
        positions = np.array([row["left_arm_joint_state"] for row in rows])
        # The measured start to first commanded row is also an original interval.
        factor, remainder = divmod(len(positions), len(original))
        self.assertEqual(remainder, 0)
        np.testing.assert_allclose(positions[factor - 1::factor], original)
        knots = np.vstack([np.zeros((1, 6)), original])
        expected = np.vstack([np.linspace(a, b, factor + 1)[1:] for a, b in zip(knots, knots[1:])])
        np.testing.assert_allclose(positions, expected)

    def test_real_resampling_and_replanning_still_meet_synchronized_rate(self):
        for arm in ("left", "right"):
            with self.subTest(arm=arm):
                adapter, _, planner, rows = replan_adapter(arm=arm)
                initial = adapter.last_dispatched_goals[arm]
                action = synchronized(motion_action(arm=arm), goal=0.)
                result = self.execute(adapter, action)
                self.assertEqual(result["status"], "completed")
                self.assertEqual(len(planner.joint_calls), 1)
                self.assert_schedule(rows, result["planned_trace"], arm, initial, 0., 0., 1., 3)
                np.testing.assert_allclose(rows[-1][f"{arm}_arm_joint_state"], [-.125] + [0.] * 5)

    def test_single_arm_row_can_schedule_bounded_gripper_commands(self):
        original = np.array([[.1, .2, .1, 0., 0., 0.]])
        adapter, rows = tracking_adapter(positions=original)
        result = self.execute(adapter, synchronized(ee_action()))
        self.assertEqual(result["status"], "completed")
        self.assert_schedule(rows, result["planned_trace"], "left", 1., 0., 0., 1., 1)
        np.testing.assert_allclose([row["left_arm_joint_state"] for row in rows],
                                   np.linspace(np.zeros(6), original[0], len(rows) + 1)[1:])

    def test_unchanged_gripper_goal_needs_no_retiming_even_for_narrow_interval(self):
        adapter, rows = tracking_adapter(positions=np.zeros((3, 6)))
        result = self.execute(adapter, synchronized(ee_action(), goal=1., start=.999))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(rows), 3)
        self.assert_schedule(rows, result["planned_trace"], "left", 1., 1., .999, 1., 3)

    def test_only_final_waypoint_segment_changes_jaw_command(self):
        adapter, rows = tracking_adapter(positions=np.zeros((10, 6)))
        result = self.execute(adapter, synchronized(ee_action(waypoints=3)))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(rows), 30)
        np.testing.assert_array_equal(jaw_commands(rows[:20], "left"), 1.)
        segments = result["planned_trace"]["segments"]
        self.assertEqual(len(segments), 3)
        for segment in segments[:-1]:
            self.assertNotIn("gripper_motion", segment)
        self.assert_schedule(rows[20:], segments[-1], "left", 1., 0., 0., 1., 10)

    def test_duration_limit_rejects_entire_oversized_segment_before_dispatch(self):
        adapter, rows = tracking_adapter(positions=np.zeros((3, 6)), step_limit=1000)
        result = self.execute(adapter, synchronized(ee_action(), start=.999, end=1.))
        self.assertEqual((result["status"], result["reason"]), ("failed", "motion_duration_limit"))
        self.assertEqual(result["executed_native_steps"], 0)
        self.assertEqual(rows, [])

    def test_duration_cap_counts_preceding_waypoint_segments(self):
        adapter, rows = tracking_adapter(positions=np.zeros((125, 6)), step_limit=1000)
        result = self.execute(adapter, synchronized(ee_action(waypoints=2), start=.98, end=1.))
        self.assertEqual((result["status"], result["reason"]), ("failed", "motion_duration_limit"))
        self.assertEqual(result["failed_waypoint_index"], 1)
        self.assertEqual(len(rows), 125)
        self.assertEqual(result["executed_native_steps"], 125)
        np.testing.assert_array_equal(jaw_commands(rows, "left"), 1.)

    def test_native_terminal_stops_mid_schedule_without_hidden_gripper_steps(self):
        for success_at, reason, count in ((None, "native_step_limit", 3), (2, "native_success", 2)):
            with self.subTest(reason=reason):
                adapter, rows = tracking_adapter(positions=np.zeros((20, 6)), step_limit=3, success_at=success_at)
                result = self.execute(adapter, synchronized(ee_action()))
                self.assertEqual((result["status"], result["reason"]), ("stopped", "episode_ended"))
                self.assertEqual(result["episode_terminal"]["reason"], reason)
                self.assertEqual(len(rows), count)
                self.assertEqual(result["executed_native_steps"], count)
                self.assertEqual(result["executed_control_steps"], count * 10)
                self.assertEqual(result["stop_confirmation"]["settling_steps"], 0)
                self.assertAlmostEqual(adapter.last_dispatched_goals["left"], 1. - count / 20)

    def test_cancel_before_and_during_schedule_never_finishes_jaw_motion(self):
        for count in (0, 3):
            with self.subTest(count=count):
                cancellation = threading.Event()
                if count == 0:
                    cancellation.set()
                adapter, rows = tracking_adapter(positions=np.zeros((20, 6)),
                    after_step=lambda index, row, measured: cancellation.set() if index == count else None)
                result = self.execute(adapter, synchronized(ee_action()), cancellation)
                self.assertEqual((result["status"], result["reason"]), ("stopped", "cancelled"))
                self.assertEqual(len(rows), count)
                self.assertEqual(result["executed_native_steps"], count)
                self.assertEqual(result["stop_confirmation"]["settling_steps"], 0)
                self.assertAlmostEqual(adapter.last_dispatched_goals["left"], 1. - count / 20)

    def test_timeout_mid_schedule_stops_without_hidden_jaw_rows(self):
        clock = [0.]
        adapter, rows = tracking_adapter(positions=np.zeros((20, 6)),
            after_step=lambda count, row, measured: clock.__setitem__(0, float(count)))
        action = synchronized(ee_action())
        action["timeout_s"] = 2.5
        with patch("ni_rua.robot.backends.robodojo.robot_worker.time.monotonic", side_effect=lambda: clock[0]):
            result = self.execute(adapter, action)
        self.assertEqual((result["status"], result["reason"]), ("failed", "execution_timeout"))
        self.assertEqual(len(rows), 3)
        self.assertEqual(result["executed_native_steps"], 3)
        self.assertEqual(result["completed_native_steps"], 3)
        self.assertEqual(result["executed_control_steps"], 30)
        self.assertEqual(result["stop_confirmation"]["settling_steps"], 0)
        np.testing.assert_allclose(jaw_commands(rows, "left"), [.95, .9, .85])
        self.assertAlmostEqual(adapter.last_dispatched_goals["left"], .85)

    def test_legacy_after_motion_keeps_jaw_constant_until_arm_finishes(self):
        adapter, rows = tracking_adapter(positions=np.zeros((20, 6)))
        result = self.execute(adapter, ee_action(gripper=True))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(rows), 26)
        np.testing.assert_array_equal(jaw_commands(rows[:20], "left"), 1.)
        self.assertAlmostEqual(jaw_commands(rows, "left")[-1], 0.)
        self.assertNotIn("gripper_motion", result["planned_trace"])


class SynchronizedGripperPreviewTests(unittest.TestCase):
    def test_preview_caches_exact_retimed_rows_and_execution_reuses_them(self):
        for arm in ("left", "right"):
            with self.subTest(arm=arm):
                adapter, _, rows = _sync_preview_adapter(arm=arm)
                action = synchronized(motion_action(arm=arm), goal=0., start=.2, end=.8)
                preview = adapter.preview_motion(action)
                self.assertEqual(preview["status"], "planned")
                self.assertEqual(rows, [])
                expected = deepcopy(preview["native_rows"])
                self.assertEqual(len(expected), len(preview["planned_trace"]["q_rad"]))
                adapter._plan_segment = Mock(side_effect=AssertionError("cached plan must not replan"))
                adapter._resample_plan = Mock(side_effect=AssertionError("cached plan must not resample"))
                preview["native_rows"][0][0] = 999.
                result = adapter.execute(dict(action, preview_plan_id=preview["plan_id"]), lambda event: None, threading.Event())
                self.assertEqual(result["status"], "completed")
                actual = [row["left_arm_joint_state"] + row["left_ee_joint_state"] +
                          row["right_arm_joint_state"] + row["right_ee_joint_state"] for row in rows]
                np.testing.assert_array_equal(actual, expected)

    def test_preview_rejects_changed_schedule_and_duration_overflow_without_motion(self):
        adapter, _, rows = _sync_preview_adapter()
        action = synchronized(motion_action())
        preview = adapter.preview_motion(action)
        changed = synchronized(motion_action(), start=.25)
        result = adapter.execute(dict(changed, preview_plan_id=preview["plan_id"]), lambda event: None, threading.Event())
        self.assertEqual(result["status"], "rejected")
        self.assertIn("action_mismatch", result["reason"])
        overflow = adapter.preview_motion(synchronized(motion_action(), start=.999))
        self.assertEqual((overflow["status"], overflow["reason"], overflow["plan_id"]), ("failed", "motion_duration_limit", None))
        self.assertEqual(rows, [])


class SynchronizedGripperValidationTests(unittest.TestCase):
    def test_capability_and_both_validation_boundaries(self):
        caps = fake_adapter()[0].capabilities()
        self.assertIn("during_motion", caps["gripper_phases"])
        action = synchronized(ee_action())
        validate_action(action)
        validate_contract_action(action, IDENTITY, "observed", caps)
        caps["gripper_phases"] = ["after_motion"]
        with self.assertRaises(ValueError):
            validate_contract_action(action, IDENTITY, "observed", caps)

    def test_invalid_fractions_and_event_context_reject_before_execution(self):
        valid = synchronized(ee_action())
        invalid = []
        for field in ("start_fraction", "end_fraction"):
            for value in (None, True, False, "0.5", float("nan"), float("inf"), -.1, 1.1):
                action = deepcopy(valid)
                action["gripper_events"][0][field] = value
                invalid.append(action)
            action = deepcopy(valid)
            del action["gripper_events"][0][field]
            invalid.append(action)
        invalid.extend([synchronized(ee_action(), start=.5, end=.5), synchronized(ee_action(), start=.75, end=.25)])
        wrong_arm = deepcopy(valid)
        wrong_arm["gripper_events"][0]["arm"] = "right"
        invalid.append(wrong_arm)
        hold = wait_action()
        del hold["hold_native_steps"]
        hold["gripper_events"] = valid["gripper_events"]
        invalid.append(hold)
        legacy = ee_action(gripper=True)
        legacy["gripper_events"][0]["start_fraction"] = 0.
        invalid.append(legacy)
        caps = fake_adapter()[0].capabilities()
        for action in invalid:
            for boundary in (validate_action, lambda value: validate_contract_action(value, IDENTITY, "observed", caps)):
                with self.subTest(event=action["gripper_events"], boundary=boundary), self.assertRaises(ValueError):
                    boundary(action)


if __name__ == "__main__":
    unittest.main()
