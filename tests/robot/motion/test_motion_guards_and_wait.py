"""CPU-only tests of bounded waits through the real native dispatch loop."""

NI_RUA_TEST_TIER = "fast"

# === tests/test_native_wait.py ===

from copy import deepcopy
import io
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ni_rua.core.contracts import validate_action as validate_contract_action
from ni_rua.robot.motion.execution_evidence import ExecutionEvidence
from ni_rua.robot.backends.robodojo.robodojo import validate_action, validate_joint_prefix
from ni_rua.robot.backends.robodojo.robot_worker import RoboDojoAdapter


IDENTITY = dict(run_id="run", episode_id="episode", env_instance_id="env", env_generation=1)


def wait_action(steps=7, **values):
    return dict(schema_version="0.1", **IDENTITY, type="EEAction", action_id="wait",
                based_on_observation_id="observed", mode="absolute", frame_id="env_origin",
                tcp_config_id="arx-x5-link6-v1", limits_profile="robodojo-native-v1",
                timeout_s=10, path=dict(kind="hold", waypoints=[]), gripper_events=[],
                hold_native_steps=steps) | values


def fake_adapter(*, step_limit=100, success_at=None, after_step=None):
    adapter = object.__new__(RoboDojoAdapter)
    adapter.np, adapter.control_hz = np, 25
    adapter.control_step = adapter.evaluator_actions = 0
    adapter.native_step_ticks = 10
    adapter.versions = {}
    adapter.observation_profile = "rgb_proprio_calibrated_v1"
    adapter.goals = {"left": .2, "right": .6}  # Wait must use last dispatched goals instead.
    adapter.last_dispatched_goals = {"left": 1., "right": .3}
    adapter.arm_step_limits = {side: np.array([.05] * 6) for side in ("left", "right")}
    adapter.arm_joint_limits, adapter.arm_joint_names = {}, {}
    adapter.gripper_step_limit, adapter.eef_gripper_dwell_step = .25, .18
    adapter.robot = SimpleNamespace(robot_list=[], robot_key=[])
    adapter.env = SimpleNamespace(end_flag=[False], success=[True], take_action_cnt=[0],
                                  step_lim=step_limit, dt=.004)
    measured = {"left": [.1] * 6, "right": [-.2] * 6}
    adapter.state = lambda: {"arms": {side: {"q_rad": list(q)} for side, q in measured.items()}}
    adapter.observe = lambda: {"robot_state": adapter.state(), "sim_step": adapter.control_step}
    dispatched = []

    def take_action(action):
        if adapter.env.end_flag[0]:
            raise AssertionError("dispatch after native termination")
        dispatched.append(deepcopy(action))
        adapter.env.take_action_cnt[0] += 1
        adapter.control_step += 10
        count = adapter.env.take_action_cnt[0]
        measured["left"][0] += .001  # Targets must not follow measured drift during the wait.
        if count == success_at or count == step_limit:
            adapter.env.end_flag[0] = True
            adapter.env.success[0] = count == success_at
        if after_step:
            after_step(count)

    adapter.env.take_action = take_action
    return adapter, dispatched


class NativeWaitTests(unittest.TestCase):
    def validate_both(self, action, capabilities=None):
        if capabilities is None:
            capabilities = fake_adapter()[0].capabilities()
        validate_contract_action(action, IDENTITY, "observed", capabilities)
        return validate_action(action)

    def test_capability_and_both_validation_boundaries(self):
        caps = fake_adapter()[0].capabilities()
        self.assertEqual(caps["bounded_wait"]["max_native_steps"], 50)
        self.assertEqual(caps["bounded_wait"]["native_control_hz"], 25)
        for count in (1, 50):
            with self.subTest(count=count):
                self.assertEqual(self.validate_both(wait_action(count), caps), (None, None))
        for count in (0, -1, 51, True, False, 1., "1", None, float("nan")):
            for boundary in (validate_action, lambda a: validate_contract_action(a, IDENTITY, "observed", caps)):
                with self.subTest(count=count, boundary=boundary), self.assertRaisesRegex(ValueError, "hold_native_steps"):
                    boundary(wait_action(count))
        without_wait = {key: value for key, value in caps.items() if key != "bounded_wait"}
        with self.assertRaisesRegex(ValueError, "unsupported_capability:bounded_wait"):
            validate_contract_action(wait_action(), IDENTITY, "observed", without_wait)
        caps["bounded_wait"]["max_native_steps"] = 3
        with self.assertRaisesRegex(ValueError, "hold_native_steps"):
            validate_contract_action(wait_action(4), IDENTITY, "observed", caps)

    def test_wait_rejects_mixed_motion_gripper_and_joint_prefix_semantics(self):
        pose = dict(left=dict(position_m=[0, 0, 1], quaternion_xyzw=[0, 0, 0, 1]))
        invalid = [
            wait_action(path=dict(kind="backend_free", waypoints=[pose])),
            wait_action(path=dict(kind="hold", waypoints=[pose])),
            wait_action(path=dict(kind="hold", waypoints=None)),
            wait_action(path=dict(kind="hold", waypoints=[], orientation="fixed")),
            wait_action(path=dict(kind="hold", waypoints=[], blend_radius_m=.01)),
            wait_action(gripper_events=[dict(arm="left", phase="after_motion", goal_normalized=0)]),
            wait_action(gripper_events=None),
            wait_action(type="JointPrefixAction", actions=[[0.] * 14], policy_provenance={"source": "test"}),
        ]
        caps = fake_adapter()[0].capabilities()
        for action in invalid:
            for boundary in (validate_action, lambda a: validate_contract_action(a, IDENTITY, "observed", caps)):
                with self.subTest(action=action, boundary=boundary), self.assertRaises(ValueError):
                    boundary(action)
        with self.assertRaisesRegex(ValueError, "bounded wait requires EEAction"):
            validate_joint_prefix(invalid[-1])
        adapter, dispatched = fake_adapter()
        with self.assertRaises(ValueError):
            adapter.execute(invalid[3], lambda event: None, threading.Event())
        self.assertEqual(dispatched, [])

    def test_wait_exact_counts_preserves_measured_arms_and_latched_grippers(self):
        for count in (1, 7, 50):
            with self.subTest(count=count):
                adapter, dispatched = fake_adapter()
                progress = []
                result = adapter.execute(wait_action(count), progress.append, threading.Event())
                self.assertEqual((result["status"], result["reason"]), ("completed", None))
                self.assertEqual(len(dispatched), count)
                self.assertEqual(result["completed_native_steps"], count)
                self.assertEqual(result["executed_native_steps"], count)
                self.assertEqual(result["executed_control_steps"], count * 10)
                self.assertEqual(result["planned_trace"]["dt_s"], 1 / 25)
                self.assertEqual(result["planned_trace"]["actions"], [[.1] * 6 + [1.] + [-.2] * 6 + [.3]] * count)
                for action in dispatched:
                    self.assertEqual(action, dict(left_arm_joint_state=[.1] * 6, left_ee_joint_state=[1.],
                                                  right_arm_joint_state=[-.2] * 6, right_ee_joint_state=[.3]))
                observed = [e for e in progress if "observation" in e]
                self.assertEqual([e["executed_steps"] for e in observed], [i for i in range(1, count + 1) if i == 1 or i % 5 == 0])
                self.assertEqual(result["terminal_observation"]["sim_step"], count * 10)
                self.assertEqual(adapter.last_dispatched_goals, dict(left=1., right=.3))

    def test_wait_stops_at_native_limit_or_success_without_hidden_steps(self):
        for success_at, expected, reason in ((None, 3, "native_step_limit"), (2, 2, "native_success")):
            with self.subTest(success_at=success_at):
                adapter, dispatched = fake_adapter(step_limit=3, success_at=success_at)
                result = adapter.execute(wait_action(50), lambda event: None, threading.Event())
                self.assertEqual((result["status"], result["reason"]), ("stopped", "episode_ended"))
                self.assertEqual(result["episode_terminal"]["reason"], reason)
                self.assertEqual(result["executed_native_steps"], expected)
                self.assertEqual(len(dispatched), expected)
                self.assertEqual(result["stop_confirmation"]["settling_steps"], 0)
                again = adapter.execute(wait_action(), lambda event: None, threading.Event())
                self.assertEqual(again["executed_native_steps"], 0)
                self.assertEqual(again["executed_control_steps"], 0)
                self.assertEqual(len(dispatched), expected)

    def test_wait_cancellation_before_and_during_dispatch(self):
        for cancel_at in (0, 2):
            with self.subTest(cancel_at=cancel_at):
                cancellation = threading.Event()
                adapter, dispatched = fake_adapter(after_step=lambda count: cancellation.set() if count == cancel_at else None)
                if cancel_at == 0:
                    cancellation.set()
                result = adapter.execute(wait_action(), lambda event: None, cancellation)
                self.assertEqual((result["status"], result["reason"]), ("stopped", "cancelled"))
                self.assertEqual(len(dispatched), cancel_at)
                self.assertEqual(result["completed_native_steps"], cancel_at)
                self.assertEqual(result["executed_native_steps"], cancel_at)
                self.assertEqual(result["stop_confirmation"]["settling_steps"], 0)

    def test_wait_timeout_preserves_partial_native_counts(self):
        clock = [0.]
        adapter, dispatched = fake_adapter(after_step=lambda count: clock.__setitem__(0, float(count)))
        with patch("ni_rua.robot.backends.robodojo.robot_worker.time.monotonic", side_effect=lambda: clock[0]):
            result = adapter.execute(wait_action(timeout_s=2.5), lambda event: None, threading.Event())
        self.assertEqual((result["status"], result["reason"]), ("failed", "execution_timeout"))
        self.assertEqual(len(dispatched), 3)
        self.assertEqual(result["executed_native_steps"], 3)
        self.assertEqual(result["completed_native_steps"], 3)
        self.assertEqual(result["executed_control_steps"], 30)
        self.assertEqual(result["stop_confirmation"]["settling_steps"], 0)

    def test_wait_timeout_before_dispatch_spends_no_native_steps(self):
        adapter, dispatched = fake_adapter()
        with patch("ni_rua.robot.backends.robodojo.robot_worker.time.monotonic", side_effect=[0., 1.]):
            result = adapter.execute(wait_action(timeout_s=.1), lambda event: None, threading.Event())
        self.assertEqual((result["status"], result["reason"]), ("failed", "planning_timeout"))
        self.assertEqual(dispatched, [])
        self.assertEqual(result["executed_native_steps"], 0)
        self.assertEqual(result["executed_control_steps"], 0)

    def test_wait_native_fault_retains_started_and_completed_step_counts(self):
        def fail_on_third_step(count):
            if count == 3:
                raise RuntimeError("native action failed after advancing physics")
        adapter, dispatched = fake_adapter(after_step=fail_on_third_step)
        with patch("ni_rua.robot.backends.robodojo.robot_worker.sys.stderr", new=io.StringIO()):
            result = adapter.execute(wait_action(), lambda event: None, threading.Event())
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(len(dispatched), 3)
        self.assertEqual(result["completed_native_steps"], 2)
        self.assertEqual(result["executed_native_steps"], 3)
        self.assertEqual(result["executed_control_steps"], 30)
        self.assertIn("native_action_failed", adapter.evaluator_failure)
        self.assertEqual(result["stop_confirmation"]["settling_steps"], 0)

    def test_final_state_sample_failure_preserves_execution_outcome_and_measurements(self):
        for outcome in ("completed", "stopped", "unknown"):
            with self.subTest(outcome=outcome):
                cancellation = threading.Event()
                def after_step(count):
                    if outcome == "stopped":
                        cancellation.set()
                    elif outcome == "unknown" and count == 2:
                        raise RuntimeError("native action failed after advancing physics")
                adapter, dispatched = fake_adapter(after_step=after_step)
                finish = ExecutionEvidence.finish
                saved = {}
                def fail_final_sample(evidence, **kwargs):
                    finish(evidence, **kwargs)
                    saved.update(trace=deepcopy(evidence.receipt["actual_trace"]))
                    adapter.state = Mock(side_effect=OSError("terminal state unavailable"))
                with patch.object(ExecutionEvidence, "finish", fail_final_sample), \
                        patch("ni_rua.robot.backends.robodojo.robot_worker.sys.stderr", new=io.StringIO()):
                    result = adapter.execute(wait_action(1 if outcome == "completed" else 3),
                        lambda _: None, cancellation)
                expected_native = 2 if outcome == "unknown" else 1
                self.assertEqual(result["status"], outcome)
                self.assertEqual(result["completed_native_steps"], 1)
                self.assertEqual(result["executed_native_steps"], expected_native)
                self.assertEqual(result["executed_control_steps"], expected_native * 10)
                self.assertEqual(len(dispatched), expected_native)
                self.assertEqual(result["actual_trace"], saved["trace"])
                self.assertIsNone(result["terminal_observation"])
                self.assertEqual(result["terminal_observation_error"], "terminal state unavailable")
                self.assertFalse(result["trace_sampling"]["actual_boundary_samples"])
                self.assertEqual(result["trace_sampling"]["terminal_state_sample_error"], "terminal state unavailable")

    def test_terminal_image_failure_does_not_discard_successful_boundary_samples(self):
        adapter, dispatched = fake_adapter()
        finish = ExecutionEvidence.finish
        def fail_terminal_image(evidence, **kwargs):
            finish(evidence, **kwargs)
            adapter.observe = Mock(side_effect=OSError("terminal camera unavailable"))
        with patch.object(ExecutionEvidence, "finish", fail_terminal_image):
            result = adapter.execute(wait_action(1), lambda _: None, threading.Event())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(dispatched), 1)
        self.assertIsNone(result["terminal_observation"])
        self.assertTrue(result["trace_sampling"]["actual_boundary_samples"])
        self.assertNotIn("terminal_state_sample_error", result["trace_sampling"])
        self.assertEqual(result["actual_trace"][-1]["sim_step"], adapter.control_step)

    def test_legacy_standalone_gripper_hold_keeps_its_existing_dwell(self):
        action = wait_action()
        del action["hold_native_steps"]
        with self.assertRaisesRegex(ValueError, "hold requires a standalone gripper event"):
            validate_action(action)
        action["gripper_events"] = [dict(arm="left", phase="after_motion", goal_normalized=0.)]
        self.assertEqual(self.validate_both(action), (None, None))
        adapter, dispatched = fake_adapter()
        result = adapter.execute(action, lambda event: None, threading.Event())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(dispatched), 6)  # ceil(abs(0 - 1) / .18), as before.
        for native in dispatched:
            self.assertEqual(native["left_ee_joint_state"], [0.])
            self.assertEqual(native["right_ee_joint_state"], [.3])
            self.assertEqual(native["left_arm_joint_state"], [.1] * 6)
            self.assertEqual(native["right_arm_joint_state"], [-.2] * 6)

# === tests/test_planner_witness.py ===

from contextlib import contextmanager, nullcontext
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from ni_rua.robot.motion.planner_witness import capture_candidate_witness, diagnose_candidate_geometry


@contextmanager
def native_fk_fixture(joint_names=("joint1", "joint2")):
    """Run the real native adapter on CPU without importing torch or cuRobo."""
    names = list(joint_names)
    calls = dict(scene_reads=0, fk=[], tensors=[])
    limits = SimpleNamespace(joint_names=names, position=np.array([[-1.] * len(names), [1.] * len(names)]),
        velocity=np.array([[-2.] * len(names), [2.] * len(names)]),
        acceleration=np.array([[-5.] * len(names), [5.] * len(names)]),
        jerk=np.array([[-10.] * len(names), [10.] * len(names)]))
    cfg = SimpleNamespace(kinematics_config=SimpleNamespace(
        link_sphere_idx_map=np.array([0, 1]), link_name_to_idx_map={"link1": 0, "link2": 1}),
        self_collision_config=SimpleNamespace(collision_pairs=np.array([[0, 1]]),
                                              sphere_padding=np.array([.01, .02])))

    def compute_kinematics(state):
        calls["fk"].append(state)
        samples = len(state.position)
        return SimpleNamespace(robot_spheres=np.tile([[[0., 0., .04, .1], [.15, 0., .04, .1]]], (samples, 1, 1)),
            tool_poses=SimpleNamespace(tool_frames=["link6"],
                position=np.tile([[[1., 2., 3.]]], (samples, 1, 1)),
                quaternion=np.tile([[[1., 0., 0., 0.]]], (samples, 1, 1))))

    motion = SimpleNamespace(joint_names=names, compute_kinematics=compute_kinematics,
        kinematics=SimpleNamespace(config=cfg, get_joint_limits=lambda: limits))

    class Planner:
        motion_planner = motion
        device_cfg = SimpleNamespace(device="cpu", dtype="float64")

        @property
        def scene_model(self):
            calls["scene_reads"] += 1
            raise AssertionError("native environment geometry is not deployment evidence")

    class JointState:
        @staticmethod
        def from_position(position, joint_names):
            return SimpleNamespace(position=position, joint_names=list(joint_names))

    def as_tensor(q, **options):
        calls["tensors"].append(options)
        return np.array(q, copy=True)

    torch = ModuleType("torch")
    torch.as_tensor, torch.no_grad = as_tensor, nullcontext
    curobo, types = ModuleType("curobo"), ModuleType("curobo.types")
    types.JointState = JointState
    curobo.types = types
    with patch.dict(sys.modules, {"torch": torch, "curobo": curobo, "curobo.types": types}):
        yield Planner(), calls


class PlannerWitnessTests(unittest.TestCase):
    def fixture(self, **overrides):
        values = dict(q_rad=[[0., 0.], [1.2, -.98]], joint_names=["joint1", "joint2"],
            joint_limits=[[-1., 1.], [-1., 1.]],
            spheres=[[[0., 0., 1., .1], [.5, 0., 1., .1]],
                     [[0., 0., .04, .1], [.15, 0., .04, .1]]],
            sphere_link_names=["link1", "link2"], self_collision_pairs=[[0, 1]],
            self_collision_padding=[.01, .02])
        values.update(overrides)
        return values

    def test_sample_self_collision_pair_and_joint_evidence(self):
        result = diagnose_candidate_geometry(**self.fixture())
        pair, = result["collision_model_witnesses"]
        self.assertEqual(pair["kind"], "self_collision")
        self.assertEqual(pair["trajectory_sample_index"], 1)
        self.assertEqual(pair["link_names"], ["link1", "link2"])
        self.assertAlmostEqual(pair["penetration_m"], .08)
        violation = result["joint_limit_violations"][0]
        self.assertEqual(violation["joint_name"], "joint1")
        self.assertEqual(violation["nearest_bound"], "upper")
        self.assertAlmostEqual(violation["margin_rad"], -.2)
        self.assertEqual(result["joints_near_limits"][0]["joint_name"], "joint2")
        self.assertTrue(result["candidate_only"])
        self.assertEqual(result["frame_id"], "planner_base")
        self.assertNotIn("_full_collision_model_witnesses", result)

    def test_ignored_pairs_and_disabled_spheres_have_no_witness(self):
        result = diagnose_candidate_geometry(**self.fixture(self_collision_pairs=np.empty((0, 2)),
            spheres=[[[0, 0, 1, -.1], [0, 0, 1, -.1]], [[0, 0, 0, -.1], [0, 0, 0, -.1]]]))
        self.assertEqual(result["collision_model_witnesses"], [])

    def test_bounds_per_kind_and_reports_omissions_without_mutating_inputs(self):
        args = self.fixture()
        args["q_rad"] = np.repeat(args["q_rad"], 4, axis=0)
        args["spheres"] = np.repeat(args["spheres"], 4, axis=0)
        original = args["spheres"].copy()
        result = diagnose_candidate_geometry(**args, max_witnesses=1)
        self.assertEqual(len(result["collision_model_witnesses"]), 1)
        complete = result["_full_collision_model_witnesses"]
        self.assertEqual([hit["trajectory_sample_index"] for hit in complete], [4, 5, 6, 7])
        self.assertEqual(complete[:1], result["collision_model_witnesses"])
        self.assertEqual(result["omitted_witness_count"], 3)
        np.testing.assert_array_equal(args["spheres"], original)

    def test_all_joint_limits_survive_a_small_collision_preview_limit(self):
        result = diagnose_candidate_geometry([[2., 3., 4., 5.]], ['a', 'b', 'c', 'd'],
            joint_limits=[[-1., 1.]] * 4, max_witnesses=1)
        self.assertEqual({item['joint_name'] for item in result['joint_limit_violations']},
                         {'a', 'b', 'c', 'd'})
        self.assertTrue(result['collision_geometry_unavailable'])

    def test_complete_witnesses_keep_all_pairs_and_summary_keeps_deepest(self):
        result = diagnose_candidate_geometry([[0.], [.1]], ["joint1"],
            spheres=[[[0, 0, 0, .2], [.1, 0, 0, .2], [.2, 0, 0, .2]],
                     [[0, 0, 0, .2], [.05, 0, 0, .2], [.1, 0, 0, .2]]],
            sphere_link_names=["a", "b", "c"], self_collision_pairs=[[0, 1], [0, 2], [1, 2]],
            self_collision_padding=[0., 0., 0.])
        complete = result["_full_collision_model_witnesses"]
        self.assertEqual(len(complete), 6)
        self.assertEqual(result["collision_model_witnesses"], complete[:3])
        self.assertEqual([hit["penetration_m"] for hit in complete],
                         sorted([hit["penetration_m"] for hit in complete], reverse=True))
        self.assertEqual(result["omitted_witness_count"], 3)

    def test_rejects_wrong_shapes_nonfinite_and_invalid_pair_mapping(self):
        for changed in (dict(q_rad=[float("nan"), 0.]), dict(sphere_link_names=["link1"]),
                        dict(self_collision_pairs=[[0, 9]]), dict(self_collision_pairs=[[0, .5]]),
                        dict(joint_limits=[[1., -1.], [-1., 1.]])):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                diagnose_candidate_geometry(**self.fixture(**changed))

    def test_unavailable_geometry_is_not_negative_collision_evidence(self):
        result = diagnose_candidate_geometry([0], ["joint1"], joint_limits=[[-1, 1]])
        self.assertTrue(result["collision_geometry_unavailable"])
        self.assertEqual(result["collision_model_witnesses"], [])

    def test_native_adapter_name_mismatch_is_fail_open_and_does_not_run_fk(self):
        class Motion:
            joint_names = ["joint1"]
            def compute_kinematics(self, state):
                raise AssertionError("must not compute FK")
        result = capture_candidate_witness(SimpleNamespace(motion_planner=Motion()), [0], ["other_joint"])
        self.assertFalse(result["available"])
        self.assertEqual(result["diagnostic_error"], "ValueError")

    def test_native_adapter_preserves_limits_when_fk_api_is_unavailable(self):
        limits = SimpleNamespace(joint_names=["joint1"], position=np.array([[-1.], [1.]]))
        kinematics = SimpleNamespace(config=SimpleNamespace(), get_joint_limits=lambda: limits)
        motion = SimpleNamespace(joint_names=["joint1"], kinematics=kinematics)
        result = capture_candidate_witness(SimpleNamespace(motion_planner=motion), [1.2], ["joint1"])
        self.assertTrue(result["available"])
        self.assertEqual(result["joint_limit_violations"][0]["joint_name"], "joint1")
        self.assertIn("collision_diagnostic_error", result)

    def test_native_fk_self_witness_and_limits_succeed_without_reading_scene(self):
        with native_fk_fixture() as (planner, calls):
            result = capture_candidate_witness(planner, [[0., 0.], [1.2, -.98]], ["joint1", "joint2"],
                joint_derivatives={"velocity": [[0., 0.], [3., 0.]]})
        self.assertEqual(calls["scene_reads"], 0)
        self.assertEqual(len(calls["fk"]), 1)
        self.assertEqual(calls["fk"][0].joint_names, ["joint1", "joint2"])
        self.assertEqual(calls["tensors"], [dict(device="cpu", dtype="float64")])
        self.assertTrue(result["available"])
        self.assertNotIn("collision_diagnostic_error", result)
        self.assertEqual({hit["kind"] for hit in result["collision_model_witnesses"]}, {"self_collision"})
        self.assertAlmostEqual(result["collision_model_witnesses"][0]["penetration_m"], .08)
        self.assertEqual(result["joint_limit_violations"][0]["joint_name"], "joint1")
        self.assertEqual(result["dynamic_joint_limit_violations"][0]["exceed"], 1.)
        self.assertEqual(result["frame_id"], "planner_base")
        self.assertEqual(result["last_sample_tool_poses"][0]["position_m"], [1., 2., 3.])

    def test_native_derivatives_identify_joint_sample_bounds_and_units(self):
        result = diagnose_candidate_geometry(**self.fixture(),
            joint_derivatives={"velocity": [[0, 0], [2., -3.]],
                               "acceleration": [[4., 0], [0, 0]],
                               "jerk": [[0, 0], [0, 15.]]},
            derivative_limits={"velocity": [[-1, 1], [-2, 2]],
                               "acceleration": [[-3, 3], [-3, 3]],
                               "jerk": [[-10, 10], [-10, 10]]})
        findings = result["dynamic_joint_limit_violations"]
        self.assertEqual(len(findings), 4)
        velocity = [item for item in findings if item["derivative"] == "velocity"]
        self.assertEqual(velocity[0]["trajectory_sample_index"], 1)
        self.assertEqual((velocity[0]["joint_name"], velocity[0]["units"], velocity[0]["exceed"]),
                         ("joint1", "rad/s", 1.))
        self.assertEqual(velocity[1]["violated_bound"], "lower")
        self.assertEqual(findings[-1]["units"], "rad/s^3")
        self.assertEqual(result["dynamic_diagnostic_unavailable"], {})

    def test_bad_derivative_does_not_discard_other_dynamics_or_geometry(self):
        result = diagnose_candidate_geometry(**self.fixture(),
            joint_derivatives={"velocity": [[float("nan"), 0], [0, 0]],
                               "acceleration": [[5, 0], [0, 0]], "jerk": [[0, 0]]},
            derivative_limits={kind: [[-1, 1], [-1, 1]] for kind in ("velocity", "acceleration", "jerk")})
        self.assertTrue(result["collision_model_witnesses"])
        self.assertTrue(result["joint_limit_violations"])
        self.assertEqual(result["dynamic_joint_limit_violations"][0]["derivative"], "acceleration")
        self.assertEqual(set(result["dynamic_diagnostic_unavailable"]), {"velocity", "jerk"})

    def test_all_joint_derivative_violations_and_explicit_missing_information(self):
        values = dict(q_rad=[[0]*5], joint_names=[f"joint{i}" for i in range(5)],
            joint_derivatives={kind: [[i+2 for i in range(5)]] for kind in ("velocity", "acceleration", "jerk")},
            derivative_limits={kind: [[-1, 1]]*5 for kind in ("velocity", "acceleration", "jerk")},
            max_witnesses=2)
        result = diagnose_candidate_geometry(**values)
        self.assertEqual(len(result["dynamic_joint_limit_violations"]), 15)
        self.assertEqual(result["omitted_dynamic_witness_count"], 0)
        self.assertEqual(result["dynamic_joint_limit_violations"][0]["joint_name"], "joint4")
        values["derivative_limits"].pop("velocity")
        values["joint_derivatives"].pop("acceleration")
        result = diagnose_candidate_geometry(**values)
        self.assertEqual(result["dynamic_diagnostic_unavailable"], {
            "velocity": "native_derivative_limits_unavailable", "acceleration": "native_derivative_not_returned"})

    def test_adapter_preserves_dynamic_evidence_when_fk_fails(self):
        limits = SimpleNamespace(joint_names=["joint1"], position=np.array([[-1.], [1.]]),
            velocity=np.array([[-2.], [2.]]), acceleration=np.array([[-5.], [5.]]), jerk=np.array([[-10.], [10.]]))
        kinematics = SimpleNamespace(config=SimpleNamespace(), get_joint_limits=lambda: limits)
        motion = SimpleNamespace(joint_names=["joint1"], kinematics=kinematics)
        result = capture_candidate_witness(SimpleNamespace(motion_planner=motion), [[0.], [.1]], ["joint1"],
            joint_derivatives={"velocity": [[0.], [3.]]})
        finding, = result["dynamic_joint_limit_violations"]
        self.assertEqual((finding["trajectory_sample_index"], finding["exceed"]), (1, 1.))
        self.assertIn("collision_diagnostic_error", result)

# === tests/test_tracking_guard_runtime.py ===

from copy import deepcopy
from types import SimpleNamespace
import threading
import unittest

import numpy as np

from support.motion_fixtures import fake_adapter as _guard_fake_adapter, wait_action as _guard_wait_action


POSE = dict(position_m=[0., 0., 1.], quaternion_xyzw=[0., 0., 0., 1.])


def ee_action(arm="left", *, waypoints=1, gripper=False):
    action = _guard_wait_action()
    del action["hold_native_steps"]
    action["path"] = dict(kind="backend_free", waypoints=[{arm: deepcopy(POSE)} for _ in range(waypoints)])
    if gripper:
        action["gripper_events"] = [dict(arm=arm, phase="after_motion", goal_normalized=0.)]
    return action


def tracking_adapter(*, positions=None, enabled=True, after_step=None, **native_options):
    measured = {side: [0.] * 6 for side in ("left", "right")}

    def advance(count):
        if after_step is not None:
            after_step(count, dispatched[-1], measured)

    adapter, dispatched = _guard_fake_adapter(after_step=advance, **native_options)
    adapter.tracking_guard_enabled = enabled
    adapter.state = lambda: {"arms": {side: dict(q_rad=list(q), ee_pose=deepcopy(POSE))
                                     for side, q in measured.items()}}
    adapter.robot.get_robot_by_arm_name = lambda name: SimpleNamespace(arm_joints_name=[f"q{i}" for i in range(6)])
    adapter.fixture_positions = np.ones((10, 6)) if positions is None else np.asarray(positions, dtype=float)
    # Bypass GPU planning only. Validation, native dispatch, state sampling,
    # endpoint checks, hold behavior and receipt accounting remain production code.
    adapter._plan_segment = lambda robot, target, diagnostics, index: (adapter.fixture_positions.copy(), 1)
    adapter._resample_plan = lambda positions, arm: positions
    return adapter, dispatched


class TrackingGuardRuntimeTests(unittest.TestCase):
    def execute(self, adapter, action=None):
        return adapter.execute(action or ee_action(), lambda event: None, threading.Event())

    def assert_counts(self, result, count):
        self.assertEqual(result["executed_steps"], count)
        self.assertEqual(result["executed_native_steps"], count)
        self.assertEqual(result["completed_native_steps"], count)
        self.assertEqual(result["executed_control_steps"], count * 10)

    def test_fifth_sustained_row_stops_without_gripper_or_settling_commands(self):
        for arm in ("left", "right"):
            with self.subTest(arm=arm):
                positions = np.zeros((12, 6))
                positions[:, 3] = .9
                adapter, dispatched = tracking_adapter(positions=positions)
                result = self.execute(adapter, ee_action(arm, gripper=True))
                self.assertEqual((result["status"], result["reason"]), ("failed", "tracking_divergence"))
                self.assert_counts(result, 5)
                self.assertEqual(len(dispatched), 5)
                self.assertEqual(result["failed_waypoint_index"], 0)
                self.assertEqual(result["waypoint_results"], [])
                self.assertEqual(result["tracking_guard"], dict(
                    arm=arm, max_joint_error_rad=.9, threshold_rad=.5, consecutive_steps=5,
                    joint_index=3, commanded_q_rad=positions[4].tolist(), measured_q_rad=[0.] * 6))
                self.assertEqual(result["stop_confirmation"]["settling_steps"], 0)
                self.assertEqual(result["terminal_observation"]["sim_step"], 50)
                self.assertEqual(result["execution_interval"], dict(
                    start_sim_step=0, end_sim_step=50, step_domain="adapter_control_step"))
                self.assertEqual(len(result["planned_trace"]["q_rad"]), 12)
                self.assertEqual({sample["sim_step"] for sample in result["actual_trace"]}, set(range(0, 51, 10)))
                for row in dispatched:
                    self.assertEqual(row["left_ee_joint_state"], [1.])
                    self.assertEqual(row["right_ee_joint_state"], [.3])

    def test_disabled_or_absent_flag_dispatches_entire_plan(self):
        for absent in (False, True):
            with self.subTest(absent=absent):
                adapter, dispatched = tracking_adapter(enabled=False)
                if absent:
                    del adapter.tracking_guard_enabled
                result = self.execute(adapter)
                self.assertEqual((result["status"], result["reason"]), ("completed", None))
                self.assert_counts(result, 10)
                self.assertEqual(len(dispatched), 10)
                self.assertNotIn("tracking_guard", result)

    def test_native_success_or_step_limit_wins_on_fifth_divergent_row(self):
        for success_at, reason in ((None, "native_step_limit"), (5, "native_success")):
            with self.subTest(success_at=success_at):
                adapter, dispatched = tracking_adapter(step_limit=5, success_at=success_at)
                result = self.execute(adapter)
                self.assertEqual((result["status"], result["reason"]), ("stopped", "episode_ended"))
                self.assertEqual(result["episode_terminal"]["reason"], reason)
                self.assert_counts(result, 5)
                self.assertEqual(len(dispatched), 5)
                self.assertNotIn("tracking_guard", result)
                self.assertEqual(result["stop_confirmation"]["settling_steps"], 0)

    def test_inactive_arm_large_error_is_ignored_for_both_active_arms(self):
        for arm in ("left", "right"):
            with self.subTest(arm=arm):
                inactive = "right" if arm == "left" else "left"

                def follow_active(count, row, measured):
                    measured[arm] = list(row[f"{arm}_arm_joint_state"])
                    measured[inactive] = [10.] * 6

                # Current commands initially differ substantially from the final
                # target; following each dispatched row must remain acceptable.
                positions = np.repeat(np.linspace(.05, 1.2, 12)[:, None], 6, axis=1)
                adapter, dispatched = tracking_adapter(positions=positions, after_step=follow_active)
                result = self.execute(adapter, ee_action(arm))
                self.assertEqual(result["status"], "completed")
                self.assert_counts(result, 12)
                self.assertEqual(len(dispatched), 12)
                self.assertNotIn("tracking_guard", result)
                self.assertEqual(result["actual_trace"][-1]["state"]["arms"][inactive]["q_rad"], [10.] * 6)

    def test_fresh_guard_for_each_action_including_after_a_trigger(self):
        adapter, dispatched = tracking_adapter(positions=np.ones((4, 6)))
        for _ in range(2):
            result = self.execute(adapter)
            self.assertEqual(result["status"], "completed")
            self.assert_counts(result, 4)
            self.assertNotIn("tracking_guard", result)
        adapter.fixture_positions = np.ones((10, 6))
        for _ in range(2):
            result = self.execute(adapter)
            self.assertEqual(result["reason"], "tracking_divergence")
            self.assert_counts(result, 5)
            self.assertEqual(result["tracking_guard"]["consecutive_steps"], 5)
        self.assertEqual(len(dispatched), 18)

    def test_recovery_at_threshold_resets_native_row_streak(self):
        positions = np.ones((12, 6))
        positions[4] = .5
        adapter, dispatched = tracking_adapter(positions=positions)
        result = self.execute(adapter)
        self.assertEqual(result["reason"], "tracking_divergence")
        self.assert_counts(result, 10)
        self.assertEqual(len(dispatched), 10)
        self.assertEqual(result["tracking_guard"]["consecutive_steps"], 5)

    def test_failure_retains_completed_waypoint_and_partial_action_count(self):
        adapter, dispatched = tracking_adapter()
        adapter._plan_segment = lambda robot, target, diagnostics, index: (
            np.zeros((3, 6)) if index == 0 else np.ones((10, 6)), 1)
        result = self.execute(adapter, ee_action(waypoints=3, gripper=True))
        self.assertEqual(result["reason"], "tracking_divergence")
        self.assertEqual(result["failed_waypoint_index"], 1)
        self.assertEqual([value["waypoint_index"] for value in result["waypoint_results"]], [0])
        self.assertEqual(len(result["planned_trace"]["segments"]), 2)
        self.assert_counts(result, 8)
        self.assertEqual(len(dispatched), 8)
        self.assertEqual(result["stop_confirmation"]["settling_steps"], 0)

    def test_wait_standalone_gripper_and_pi_prefix_ignore_tracking_guard(self):
        gripper = _guard_wait_action()
        del gripper["hold_native_steps"]
        gripper["gripper_events"] = [dict(arm="left", phase="after_motion", goal_normalized=0.)]
        prefix = dict(type="JointPrefixAction", actions=[[0.] * 14 for _ in range(10)],
                      policy_provenance={"source": "pi05"})

        def diverge(count, row, measured):
            for side in measured:
                measured[side] = [10.] * 6

        for name, action, expected in (("wait", _guard_wait_action(10), 10), ("gripper", gripper, 6)):
            with self.subTest(name=name):
                adapter, dispatched = tracking_adapter(after_step=diverge)
                result = self.execute(adapter, action)
                self.assertEqual((result["status"], result["reason"]), ("completed", None))
                self.assert_counts(result, expected)
                self.assertEqual(len(dispatched), expected)
                self.assertNotIn("tracking_guard", result)
                for row in dispatched:
                    self.assertEqual(row["left_arm_joint_state"], [0.] * 6)
                    self.assertEqual(row["right_arm_joint_state"], [0.] * 6)

        # P0-3: JointPrefixAction always enforces TrackingGuard across active arms (even when tracking_guard_enabled=False)
        for enabled in (True, False):
            with self.subTest(pi_prefix_tracking_guard_enabled=enabled):
                adapter, dispatched = tracking_adapter(enabled=enabled, after_step=diverge)
                result = self.execute(adapter, prefix)
                self.assertEqual((result["status"], result["reason"]), ("stopped", "joint_tracking_divergence"))
                self.assert_counts(result, 5)
                self.assertEqual(len(dispatched), 5)
                self.assertEqual(result["tracking_guard"]["arm"], "left")
                self.assertEqual(result["tracking_guard"]["consecutive_steps"], 5)
                self.assertEqual(result["tracking_guard"]["max_joint_error_rad"], 10.0)
                guard_summary = result["policy_chunk_guard"]
                self.assertTrue(guard_summary["tracking_guard_enabled"])
                self.assertTrue(guard_summary["triggered"])
                self.assertEqual(guard_summary["triggered_arm"], "left")
                self.assertEqual(guard_summary["threshold_rad"], 0.5)
                self.assertEqual(guard_summary["consecutive_steps"], 5)
                self.assertEqual(guard_summary["divergence"]["max_joint_error_rad"], 10.0)

        # Nominal JointPrefixAction within threshold completes and reports policy_chunk_guard(triggered=False)
        adapter_ok, dispatched_ok = tracking_adapter(enabled=False)
        res_ok = self.execute(adapter_ok, prefix)
        self.assertEqual((res_ok["status"], res_ok["reason"]), ("completed", None))
        self.assert_counts(res_ok, 10)
        self.assertEqual(len(dispatched_ok), 10)
        self.assertEqual(
            res_ok["policy_chunk_guard"],
            {
                "motion_safety_enabled": False,
                "motion_safety_status": "disabled",
                "tracking_guard_enabled": True,
                "threshold_rad": 0.5,
                "consecutive_steps": 5,
                "triggered": False,
            },
        )

        # JointPrefixAction routes through admit_motion when motion_safety_guard_enabled=True and rejects before physics
        adapter_rej, dispatched_rej = tracking_adapter(enabled=True)
        adapter_rej.motion_safety_guard_enabled = True
        adapter_rej.robot_model = {
            "joints": [
                {"name": f"q{i}", "type": "revolute", "limits": {"lower": -3.14, "upper": 3.14}}
                for i in range(6)
            ]
        }
        bad_rows = [[0.0] * 14 for _ in range(4)]
        bad_rows[0][0] = 2.6
        bad_rows[1][0] = 0.0
        bad_rows[2][0] = 2.6
        bad_prefix = dict(type="JointPrefixAction", actions=bad_rows, policy_provenance={"source": "pi05"})
        res_rej = self.execute(adapter_rej, bad_prefix)
        self.assertEqual((res_rej["status"], res_rej["reason"]), ("rejected", "motion_safety_rejected"))
        self.assertEqual(res_rej["executed_native_steps"], 0)
        self.assertFalse(res_rej["physics_advanced"])
        self.assertEqual(len(dispatched_rej), 0)
        self.assertEqual(res_rej["motion_safety"]["status"], "reject")
        self.assertEqual(
            res_rej["policy_chunk_guard"],
            {
                "motion_safety_enabled": True,
                "motion_safety_status": "reject",
                "tracking_guard_enabled": True,
                "threshold_rad": 0.5,
                "consecutive_steps": 5,
                "triggered": False,
            },
        )


class TrackingGuardTest(unittest.TestCase):
    """TrackingGuard 纯数学跟随误差状态机测试 (迁自 test_tracking_guard.py)。"""

    def test_sustained_error_triggers_on_fifth_native_row(self):
        from ni_rua.robot.motion.tracking_guard import TrackingGuard

        guard = TrackingGuard()
        command = [0, 0, 0, 0, 0, 0]
        measured = [0, 0, 0, -0.6, 0, 0]
        for _ in range(4):
            self.assertIsNone(guard.observe(command, measured))
        self.assertEqual(guard.observe(command, measured), {
            "max_joint_error_rad": 0.6,
            "threshold_rad": 0.5,
            "consecutive_steps": 5,
            "joint_index": 3,
            "commanded_q_rad": command,
            "measured_q_rad": measured,
        })

    def test_transient_and_exact_threshold_reset_consecutive_count(self):
        from ni_rua.robot.motion.tracking_guard import TrackingGuard

        for recovery in (0.1, 0.5):
            with self.subTest(recovery=recovery):
                guard = TrackingGuard()
                for _ in range(4):
                    self.assertIsNone(guard.observe([0] * 6, [0.6] * 6))
                self.assertIsNone(guard.observe([0] * 6, [recovery] * 6))
                for _ in range(4):
                    self.assertIsNone(guard.observe([0] * 6, [0.6] * 6))
                self.assertEqual(guard.observe([0] * 6, [0.6] * 6)["consecutive_steps"], 5)

    def test_compares_same_row_target_and_does_not_wrap_joint_coordinates(self):
        from ni_rua.robot.motion.tracking_guard import TrackingGuard

        guard = TrackingGuard(consecutive_steps=1)
        self.assertIsNone(guard.observe([2.5] * 6, [2.4] * 6))
        result = guard.observe([3.1] * 6, [-3.1] * 6)
        self.assertAlmostEqual(result["max_joint_error_rad"], 6.2)
        self.assertEqual(result["joint_index"], 0)

    def test_invalid_configuration(self):
        from ni_rua.robot.motion.tracking_guard import TrackingGuard

        for value in (0, -1, True, '0.5', float('nan'), float('inf')):
            with self.subTest(threshold=value), self.assertRaises(ValueError):
                TrackingGuard(threshold_rad=value)
        for value in (0, -1, True, 1.5, '5', float('inf')):
            with self.subTest(steps=value), self.assertRaises(ValueError):
                TrackingGuard(consecutive_steps=value)

    def test_invalid_samples_raise_without_updating_streak(self):
        from ni_rua.robot.motion.tracking_guard import TrackingGuard

        invalid = (
            None, [], [0] * 5, [0] * 7, [0, 0, 0, 0, 0, True],
            [0, 0, 0, 0, 0, '1'], [0, 0, 0, 0, 0, float('nan')],
            [0, 0, 0, 0, 0, float('inf')],
        )
        for sample in invalid:
            for side in ('command', 'measured'):
                with self.subTest(sample=sample, side=side):
                    guard = TrackingGuard(consecutive_steps=2)
                    self.assertIsNone(guard.observe([0] * 6, [0.6] * 6))
                    arguments = {'command': [0] * 6, 'measured': [0.6] * 6, side: sample}
                    with self.assertRaises(ValueError):
                        guard.observe(**arguments)
                    self.assertIsNotNone(guard.observe([0] * 6, [0.6] * 6))
        with self.assertRaises(ValueError):
            TrackingGuard().observe([1e308] * 6, [-1e308] * 6)

    def test_inputs_and_diagnostic_arrays_are_independent(self):
        from ni_rua.robot.motion.tracking_guard import TrackingGuard

        command, measured = [0] * 6, [0.6] * 6
        guard = TrackingGuard(consecutive_steps=1)
        result = guard.observe(command, measured)
        self.assertEqual(command, [0] * 6)
        self.assertEqual(measured, [0.6] * 6)
        result['commanded_q_rad'][0] = 99
        result['measured_q_rad'][0] = 99
        self.assertEqual(command, [0] * 6)
        self.assertEqual(measured, [0.6] * 6)
        self.assertIsNone(TrackingGuard().observe(command, measured))


if __name__ == "__main__":
    unittest.main()
