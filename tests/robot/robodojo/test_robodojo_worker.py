"""RoboDojo setup keeps task authority and worker compatibility after extraction."""

NI_RUA_TEST_TIER = "component"

# === tests/test_environment_adapter.py ===

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from support.paths import REPO_ROOT
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ni_rua.robot.backends.robodojo import robot_worker
from ni_rua.robot.model.embodiments import get_robodojo_robot
from ni_rua.robot.backends.robodojo import environment as robodojo
from ni_rua.robot.backends.robodojo.native_evaluator import HostPoseView


class EnvironmentAdapterTests(unittest.TestCase):
    def test_import_does_not_start_or_import_simulator(self):
        result = subprocess.run([sys.executable, '-c',
            'import sys; import ni_rua.robot.backends.robodojo.environment; '
            'assert not any(name.startswith(("isaacsim", "isaaclab", "env.")) for name in sys.modules)'],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_native_sim_settings_are_preserved_without_an_installed_simulator(self):
        for device_settings in ({}, {"device": "cpu", "use_fabric": False},
                                {"device": "cuda:1", "use_fabric": True}):
            with self.subTest(settings=device_settings):
                sim = {**device_settings, "dt": .004, "render_interval": 5,
                       "scene": {"num_envs": 10, "env_spacing": 7},
                       "physx": {"enable_stabilization": True}, "render": {"antialiasing_mode": "TAA"}}
                files = {"arx_x5.yml": {"config": {"sim": "sim_config"}},
                         "sim_config.yml": sim, "fixture.yml": {}}
                captured = {}
                def create(sections):
                    captured.update(sections)
                    return SimpleNamespace(camera=SimpleNamespace(),
                        eval_cfg=SimpleNamespace(observation=SimpleNamespace(collect_freq=25)))
                modules = {
                    "omegaconf": SimpleNamespace(OmegaConf=SimpleNamespace(create=create)),
                    "utils.load_file": SimpleNamespace(load_yaml=lambda path: deepcopy(files[path.name])),
                    "utils.pipeline_utils": SimpleNamespace(process_randomization=lambda config: config,
                        process_config=lambda config, task_name: (config, 25)),
                }
                with patch.dict(sys.modules, modules), patch.object(robodojo, "configure_depth_capture"):
                    robodojo.native_eval_config('/native', 'fixture', seed=3, layout_id=2)
                expected = deepcopy(sim)
                expected["seed"] = [2]
                expected["scene"]["num_envs"] = 1
                self.assertEqual(captured["sim"], expected)

    def test_worker_preserves_config_override_and_rejects_unknown_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = SimpleNamespace(gpu=0, run_dir=directory, robodojo_root=directory,
                task='fixture', seed=3, layout_id=2, observation_profile='rgb_proprio_v1')
            with patch.object(robot_worker, 'launch_application') as launch, \
                    patch.object(robot_worker, 'native_eval_config') as config, \
                    patch.object(robot_worker, 'create_environment', side_effect=RuntimeError('setup reached')) as create:
                with self.assertRaisesRegex(RuntimeError, 'setup reached'):
                    robot_worker.RoboDojoAdapter(arguments)
                config.assert_called_once_with(Path(directory), 'fixture', 3, 2, 'rgb_proprio_v1',
                                               robot_binding=get_robodojo_robot())
                self.assertIs(create.call_args.args[3], config.return_value)
                launch.reset_mock()
                arguments.robot_model_id = 'unsupported-arm'
                with self.assertRaisesRegex(ValueError, 'unsupported'):
                    robot_worker.RoboDojoAdapter(arguments)
                launch.assert_not_called()

    def environment(self, *, mismatch=False):
        binding = replace(get_robodojo_robot(), layout_directory=Path('custom-layouts'))
        arguments = SimpleNamespace(seed=3, layout_id=2, task='fixture')
        layout = '/assets/custom-layouts/3/fixture_2.json'
        manager = SimpleNamespace(layout_valid=[True], check_layout_stability=Mock(return_value=(True, [])))
        parser = SimpleNamespace(init_state=Mock())
        native = SimpleNamespace(scene_manager=SimpleNamespace(layout_manager=manager),
            seed_manager=SimpleNamespace(seed_info={2: {'scene_layout': '/wrong' if mismatch else layout}}),
            reward_manager=SimpleNamespace(func_parser=parser))
        def reset(*, seed):
            self.assertEqual(seed, [2])
            manager.check_layout_stability()
            parser.init_state()
        native.reset = Mock(side_effect=reset)
        external_policy = object()
        eval_env = SimpleNamespace(WsModelClient=external_policy)
        def create(config, app):
            policy = eval_env.WsModelClient()
            policy.call(func_name='reset')
            with self.assertRaisesRegex(ValueError, 'policy_inference_is_owned_by_Harness'):
                policy.call(func_name='predict')
            return native
        eval_env.create_eval_env = create
        modules = {'env.global_configs': SimpleNamespace(ASSETS_PATH='/assets'),
                   'src.eval_client': SimpleNamespace(eval_env=eval_env),
                   'env.scene_manager.objects.articulation': SimpleNamespace(ArticulationObject=object)}
        return arguments, binding, native, eval_env, external_policy, modules

    def test_native_reset_records_admission_and_restores_external_policy(self):
        arguments, binding, native, eval_env, policy, modules = self.environment()
        with patch.dict(sys.modules, modules), patch.object(robodojo, 'host_articulation_properties', return_value=nullcontext()):
            result, acceptance = robodojo.create_environment(arguments, object(), Path('/run'), object(), binding)
        self.assertIs(result, native)
        self.assertIs(eval_env.WsModelClient, policy)
        self.assertEqual(native.save_dir, '/run/native-eval')
        self.assertEqual(native._stream_dir, '/run/native-eval/_stream')
        self.assertEqual(acceptance, dict(valid=True, stable=True, unstable_envs=[],
            authority='official EvalEnv.reset/setup_scene stability check'))
        self.assertIsInstance(native.reward_manager.func_parser.layout_manager, HostPoseView)
        native.reset.assert_called_once_with(seed=[2])

    def test_wrong_layout_fails_before_reset(self):
        arguments, binding, native, eval_env, policy, modules = self.environment(mismatch=True)
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(ValueError, 'native_seed_manager_layout_identity_mismatch'):
                robodojo.create_environment(arguments, object(), Path('/run'), object(), binding)
        native.reset.assert_not_called()
        self.assertIs(eval_env.WsModelClient, policy)

# === tests/test_robot.py ===

from pathlib import Path
import hashlib
import json
import os
import subprocess
import signal
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ni_rua.robot.backends.robodojo.robodojo import FRAME_ID, TCP_CONFIG_ID, RobotBackend, validate_action
from ni_rua.robot.backends.robodojo.robot_process import native_gpu_oom_signature
from ni_rua.robot.backends.robodojo.robot_worker import RoboDojoAdapter


class RobotTests(unittest.TestCase):
    def test_machine_paths_can_be_overridden_without_starting_a_worker(self):
        with patch.dict(os.environ, NI_RUA_ROBODOJO_ROOT="~/simulation/robodojo",
                        NI_RUA_ROBODOJO_PYTHON="/opt/robot-env/bin/python"):
            backend = RobotBackend()
            self.assertEqual(backend.root, str((Path.home() / "simulation/robodojo").resolve()))
            self.assertEqual(backend.python, "/opt/robot-env/bin/python")
            explicit = RobotBackend(robodojo_root="/opt/explicit", python="/opt/explicit/python")
            self.assertEqual((explicit.root, explicit.python), ("/opt/explicit", "/opt/explicit/python"))
            self.assertIsNone(backend.process)

    def action(self):
        return {"mode": "absolute", "frame_id": FRAME_ID, "tcp_config_id": TCP_CONFIG_ID,
            "limits_profile": "simulation-slow-v1", "timeout_s": 10,
            "path": {"kind": "backend_free", "waypoints": [{"left": {"position_m": [-0.3, 0, 1],
                       "quaternion_xyzw": [0, 0, 0, 1]}}], "position_tolerance_m": 0.005}, "gripper_events": []}

    def test_capability_refusal(self):
        action = self.action()
        self.assertEqual(validate_action(action)[0], "left")
        for kind in ("linear", "polyline", "arc", "spline"):
            action["path"]["kind"] = kind
            with self.assertRaisesRegex(ValueError, "unsupported_capability"):
                validate_action(action)
        action = self.action()
        action["path"]["waypoints"][0]["left"]["position_m"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid_pose"):
            validate_action(action)
        action = self.action()
        action["path"]["waypoints"][0]["right"] = action["path"]["waypoints"][0]["left"]
        with self.assertRaisesRegex(ValueError, "^unsupported_capability:dual_arm_motion$"):
            validate_action(action)
        action["path"]["waypoints"].append({"left": action["path"]["waypoints"][0]["left"]})
        with self.assertRaisesRegex(ValueError, "^unsupported_capability:dual_arm_motion$"):
            validate_action(action)

    def planning_fixture(self, outcome):
        import numpy as np
        adapter = object.__new__(RoboDojoAdapter)
        adapter.np = np
        adapter.env = SimpleNamespace(dt=1.0, reward_manager=SimpleNamespace(step=lambda _: None),
                                      end_flag=[False], take_action_cnt=[0], step_lim=200, success=[True])
        robot = SimpleNamespace(robot_name="fixture", arm_joints_name=["joint"], arm_joint_indices=[0], entity_origin_pose=object())
        calls, failure = [], RuntimeError("synthetic native failure")

        class BadStatus:
            def __str__(self): raise ValueError("synthetic diagnostic conversion failure")

        result = SimpleNamespace(status=BadStatus() if outcome == "diagnostic_error" else "SUCCESS" if outcome == "success" else "IK_FAIL_PRIVATE",
                                 success=np.array([outcome == "success"]))

        class NativePlanner:
            def plan_pose(self, *args, **kwargs):
                calls.append((args, kwargs))
                if outcome == "exception": raise failure
                return result

        native = NativePlanner()
        received = []
        def plan_path(current, target, origin):
            received.append((current, target, origin))
            actual = native.plan_pose(goal_tool_poses=target, current_state=current,
                                      max_attempts=5, enable_graph_attempt=5)
            self.assertIs(actual, result)
            return {"status": "Success", "position": [[0.0], [0.1]]} if bool(actual.success[0]) else {"status": "Fail"}

        class Limits:
            def __getitem__(self, key): return self
            def cpu(self): return self
            def numpy(self): return np.array([[-1.0, 1.0]])

        adapter.robot = SimpleNamespace(planner={robot.robot_name: SimpleNamespace(motion_planner=native, plan_path=plan_path)},
            get_joint=lambda *args: [np.array([0.0])], robot_list=[robot],
            robot_key=[SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=Limits()))],
            get_robot_by_arm_name=lambda name: robot)
        return adapter, robot, native, calls, received, failure

    def test_native_planning_diagnostics_preserve_call_result_and_restore_instance(self):
        for outcome in ("success", "failure", "exception", "diagnostic_error"):
            for instance_override in (False, True):
                with self.subTest(outcome=outcome, instance_override=instance_override):
                    adapter, robot, native, calls, received, failure = self.planning_fixture(outcome)
                    original = native.plan_pose
                    if instance_override: native.plan_pose = original
                    before = dict(vars(native))
                    diagnostic = []
                    target = self.action()["path"]["waypoints"][0]["left"]
                    if outcome == "exception":
                        with self.assertRaises(RuntimeError) as raised:
                            adapter._plan_segment(robot, target, diagnostic, waypoint_index=3)
                        self.assertIs(raised.exception, failure)
                    else:
                        segment = adapter._plan_segment(robot, target, diagnostic, waypoint_index=3)
                        if outcome == "success":
                            self.assertEqual(segment[0].tolist(), [[0.0], [0.1]])
                            self.assertEqual(segment[1], 1)
                        else: self.assertIsNone(segment)
                    self.assertEqual(vars(native), before)
                    self.assertEqual(native.plan_pose, original)
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(len(received), 1)
                    self.assertEqual(calls[0][0], ())
                    self.assertIs(calls[0][1]["current_state"], received[0][0])
                    self.assertIs(calls[0][1]["goal_tool_poses"], received[0][1])
                    self.assertIs(received[0][2], robot.entity_origin_pose)
                    self.assertEqual(calls[0][1]["max_attempts"], 5)
                    self.assertEqual(calls[0][1]["enable_graph_attempt"], 5)
                    self.assertEqual(len(diagnostic), 1)
                    self.assertEqual(diagnostic[0]["waypoint_index"], 3)
                    self.assertEqual(diagnostic[0]["stage"], "plan_pose")
                    self.assertIs(diagnostic[0]["returned_none"], None if outcome == "exception" else False)
                    if outcome == "diagnostic_error":
                        self.assertEqual(diagnostic[0]["diagnostic_error"], "ValueError")
                    else:
                        self.assertGreaterEqual(diagnostic[0]["elapsed_wall_s"], 0)
                        self.assertEqual(diagnostic[0]["exception_type"], "RuntimeError" if outcome == "exception" else None)
                        self.assertEqual(diagnostic[0]["success"], None if outcome == "exception" else [outcome == "success"])
                        self.assertEqual(diagnostic[0]["status"], {"success": "SUCCESS", "failure": "IK_FAIL_PRIVATE", "exception": None}[outcome])

    def test_native_planning_stages_preserve_retries_and_optional_results(self):
        import numpy as np
        for outcome in ("success", "ik_failure", "graph_failure", "trajopt_failure", "exception"):
            with self.subTest(outcome=outcome):
                adapter, robot, native, _, _, failure = self.planning_fixture("success")
                ik_result = SimpleNamespace(success=np.array([[outcome != "ik_failure"]]))
                trajopt_result = SimpleNamespace(success=np.array([[outcome == "success"]]))
                graph_result = None if outcome == "graph_failure" else np.zeros((1, 2, 1))
                seed, calls = object(), []

                class Solver:
                    def __init__(self, stage, result): self.stage, self.result = stage, result
                    def solve_pose(self, *args, **kwargs):
                        calls.append((self.stage, args, kwargs))
                        if self.stage == "trajopt" and outcome == "exception": raise failure
                        return self.result

                native.ik_solver, native.trajopt_solver = Solver("ik", ik_result), Solver("trajopt", trajopt_result)
                def graph(*args, **kwargs):
                    calls.append(("graph_seed", args, kwargs))
                    return graph_result
                native._get_graph_seed_trajectories = graph
                def plan_pose(goal_tool_poses, current_state, max_attempts, enable_graph_attempt):
                    result = None
                    for _ in range(max_attempts):
                        ik = native.ik_solver.solve_pose(goal_tool_poses, current_state=current_state, return_seeds=2)
                        self.assertIs(ik, ik_result)
                        if not ik.success.any(): continue
                        graph_seed = native._get_graph_seed_trajectories(current_state, seed_config=seed)
                        self.assertIs(graph_seed, graph_result)
                        if graph_seed is None: continue
                        result = native.trajopt_solver.solve_pose(goal_tool_poses, current_state,
                                                                 seed_config=seed, seed_traj=graph_seed)
                        self.assertIs(result, trajopt_result)
                        if result.success.any(): break
                    return result
                native.plan_pose = plan_pose
                def plan_path(current, target, origin):
                    result = native.plan_pose(target, current, max_attempts=5, enable_graph_attempt=0)
                    return {"status": "Success", "position": [[0.1]]} if result and result.success.any() else {"status": "Fail"}
                adapter.robot.planner[robot.robot_name].plan_path = plan_path
                owners = (native, native.ik_solver, native.trajopt_solver)
                before = [dict(vars(owner)) for owner in owners]
                diagnostic = []
                target = self.action()["path"]["waypoints"][0]["left"]
                if outcome == "exception":
                    with self.assertRaises(RuntimeError) as raised:
                        adapter._plan_segment(robot, target, diagnostic)
                    self.assertIs(raised.exception, failure)
                else:
                    result = adapter._plan_segment(robot, target, diagnostic)
                    self.assertEqual(result is not None, outcome == "success")
                self.assertEqual([vars(owner) for owner in owners], before)
                attempt = ["ik"] if outcome == "ik_failure" else ["ik", "graph_seed"] if outcome == "graph_failure" else ["ik", "graph_seed", "trajopt"]
                expected = attempt * (1 if outcome in ("success", "exception") else 5)
                self.assertEqual([item[0] for item in calls], expected)
                self.assertEqual([item["stage"] for item in diagnostic], expected + ["plan_pose"])
                self.assertTrue(all(item["status"] is None for item in diagnostic))
                for item, (_, args, kwargs) in zip(diagnostic, calls):
                    if item["stage"] == "ik": self.assertEqual(item["success"], ik_result.success.tolist())
                    if item["stage"] == "graph_seed": self.assertIs(item["returned_none"], graph_result is None)
                    if item["stage"] == "trajopt":
                        self.assertIs(kwargs["seed_config"], seed)
                        self.assertIs(kwargs["seed_traj"], graph_result)
                self.assertIs(diagnostic[-1]["returned_none"], None if outcome == "exception" else outcome in ("ik_failure", "graph_failure"))

    def test_planning_diagnostics_only_enter_private_development_file(self):
        for outcome in ("success", "failure", "exception"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as directory:
                adapter, robot, native, calls, received, failure = self.planning_fixture(outcome)
                original_plan = adapter._plan_segment
                def plan_with_private_debug(*args, **kwargs):
                    try:
                        return original_plan(*args, **kwargs)
                    finally:
                        for item in args[2]:
                            item["private_debug"] = "PRIVATE_DIAGNOSTIC_SENTINEL"
                adapter._plan_segment = plan_with_private_debug
                adapter.directory = Path(directory)
                adapter.observation_profile = "rgb_proprio_v1"
                adapter.control_step = adapter.evaluator_actions = 0
                adapter.versions = {}
                target = self.action()["path"]["waypoints"][0]["left"]
                adapter.state = lambda: {"arms": {side: {"q_rad": [0.0] * 6, "ee_pose": target}
                                                   for side in ("left", "right")}}
                adapter.observe = lambda: {"robot_state": adapter.state()}
                adapter.capabilities = lambda: {"collision_scope": "synthetic fixture"}
                adapter.control_hz, adapter.gripper_step_limit, adapter.eef_gripper_dwell_step = 25, 0.25, 0.18
                adapter.arm_step_limits = {"left": __import__("numpy").full(6, .05)}
                adapter.last_dispatched_goals = {"left": 1.0, "right": 1.0}
                adapter._resample_plan = lambda positions, arm: __import__("numpy").zeros((1, 6))
                def native_action(row):
                    adapter.control_step += 10
                    adapter.env.take_action_cnt[0] += 1
                    adapter.evaluator_actions += 1
                    return True
                adapter._native_action = native_action
                adapter._hold = lambda: {"clock": "paused"}
                progress = []
                action = {**self.action(), "action_id": "private-boundary"}
                receipt = adapter.execute(action, progress.append, threading.Event())
                private_path, = (Path(directory) / "development-diagnostics").glob("action-*.json")
                private = json.loads(private_path.read_text())
                self.assertEqual(private["audience"], "observer_only")
                self.assertEqual(private["action_id"], "private-boundary")
                self.assertEqual(len(private["planning_diagnostics"]), 1)
                self.assertEqual(private["planning_diagnostics"][0]["private_debug"], "PRIVATE_DIAGNOSTIC_SENTINEL")
                self.assertNotIn("planning_diagnostics", private["receipt"])
                published = json.dumps({"receipt": receipt, "progress": progress, "observe": adapter.observe()})
                self.assertNotIn("planning_diagnostics", published)
                self.assertNotIn("PRIVATE_DIAGNOSTIC_SENTINEL", published)
                # Native status is now an explicitly public solver summary;
                # arbitrary raw diagnostic fields remain private in all profiles.
                without_public_status = json.loads(published)
                if outcome == "failure":
                    summary, = without_public_status["receipt"]["failure_feedback"]["solver_summary"]
                    self.assertEqual(summary.pop("native_statuses"), ["IK_FAIL_PRIVATE"])
                self.assertNotIn("IK_FAIL_PRIVATE", json.dumps(without_public_status))
                self.assertEqual(receipt["status"], "completed" if outcome == "success" else "failed")
                self.assertEqual(len(calls), 1)
                self.assertNotIn("plan_pose", vars(native))

                # A diagnostic archive failure must not replace the actuator receipt.
                with patch.object(Path, "write_text", side_effect=ValueError("synthetic archive error")):
                    second = adapter.execute(action, lambda _: None, threading.Event())
                self.assertEqual(second["status"], receipt["status"])
                self.assertEqual(second["reason"], receipt["reason"])
                self.assertEqual(len(calls), 2)
        action = self.action()
        action["gripper_events"] = [{"arm": "left", "phase": "during_motion", "goal_normalized": 0}]
        with self.assertRaisesRegex(ValueError, "unsupported_capability"):
            validate_action(action)

    def test_stop_does_not_dispatch_pending_gripper_event(self):
        # Reproduce cancellation after the arm path ends but before the first gripper dispatch.
        adapter = object.__new__(RoboDojoAdapter)
        adapter.goals = {"left": 0.0, "right": 1.0}
        adapter.last_dispatched_goals = {"left": 0.5, "right": 1.0}
        adapter.state = lambda: {"arms": {"left": {"q_rad": [0.0] * 6, "qdot_rad_s": [0.0] * 6}}}
        dispatched = []
        adapter._control = lambda holds: dispatched.append(dict(adapter.goals))
        stopped = adapter._hold()
        self.assertEqual(dispatched, [])
        self.assertEqual(stopped["settling_steps"], 0)
        self.assertEqual(adapter.goals, adapter.last_dispatched_goals)
        self.assertEqual(stopped["clock"], "paused")

    def test_multiple_waypoints_and_standalone_gripper(self):
        action = self.action()
        action["path"]["waypoints"] *= 8
        self.assertEqual(validate_action(action)[0], "left")
        action["path"]["waypoints"].append(action["path"]["waypoints"][0])
        with self.assertRaisesRegex(ValueError, "one to eight"):
            validate_action(action)

        action = self.action()
        action["path"]["waypoints"].append({"right": action["path"]["waypoints"][0]["left"]})
        with self.assertRaisesRegex(ValueError, "same arm"):
            validate_action(action)
        action = self.action()
        action["path"] = {"kind": "hold", "waypoints": []}
        with self.assertRaisesRegex(ValueError, "requires a standalone"):
            validate_action(action)
        action["gripper_events"] = [{"arm": "right", "phase": "after_motion", "goal_normalized": 0.5}]
        self.assertEqual(validate_action(action), (None, None))
        action["gripper_events"][0]["goal_normalized"] = True
        with self.assertRaisesRegex(ValueError, "invalid_gripper"):
            validate_action(action)

    def test_standalone_gripper_freezes_measured_joints(self):
        # Real executor and a native-action sink; cancellation spends no extra ticks.
        from copy import deepcopy
        import numpy as np
        for cancel_at in (None, 0, 1):
            with self.subTest(cancel_at=cancel_at):
                adapter = object.__new__(RoboDojoAdapter)
                adapter.np = np
                adapter.observation_profile = "rgb_proprio_calibrated_v1"
                measured = {"left": [0.1] * 6, "right": [-0.2] * 6}
                adapter.goals = {"left": 1.0, "right": 0.3}
                adapter.last_dispatched_goals = dict(adapter.goals)
                adapter.control_step = adapter.evaluator_actions = 0
                adapter.control_hz, adapter.gripper_step_limit, adapter.eef_gripper_dwell_step = 25, 0.25, 0.18
                adapter.versions = {}
                adapter.env = SimpleNamespace(end_flag=[False], take_action_cnt=[0], step_lim=200, success=[True])
                adapter.capabilities = lambda: {"collision_scope": "synthetic_fixture"}
                adapter.state = lambda: {"arms": {name: {"q_rad": list(q), "qdot_rad_s": [0.0] * 6}
                                                   for name, q in measured.items()}}
                adapter.observe = lambda: {"robot_state": adapter.state()}
                cancellation = threading.Event()
                dispatched = []

                def native_action(row):
                    dispatched.append(deepcopy(row))
                    adapter.goals = {"left": row[6], "right": row[13]}
                    adapter.last_dispatched_goals = dict(adapter.goals)
                    adapter.control_step += 10
                    adapter.env.take_action_cnt[0] += 1
                    adapter.evaluator_actions += 1
                    measured["left"][0] += 0.001
                    if len(dispatched) == cancel_at:
                        cancellation.set()
                    return True

                adapter._native_action = native_action
                if cancel_at == 0:
                    cancellation.set()
                initial = deepcopy(measured)
                action = self.action()
                action["path"] = {"kind": "hold", "waypoints": []}
                action["gripper_events"] = [{"arm": "left", "phase": "after_motion", "goal_normalized": 0.0}]
                receipt = adapter.execute(action, lambda progress: None, cancellation)
                self.assertEqual(receipt["before_state"]["arms"]["left"]["q_rad"], initial["left"])
                self.assertEqual(receipt["status"], "completed" if cancel_at is None else "stopped")
                self.assertIsNotNone(receipt["terminal_observation"])
                self.assertEqual(len(dispatched), {None: 6, 0: 0, 1: 1}[cancel_at])
                for row in dispatched:
                    self.assertEqual(row, initial["left"] + [0.0] + initial["right"] + [0.3])
                self.assertEqual(receipt["executed_control_steps"], 10 * len(dispatched))
                if cancel_at == 1:
                    self.assertEqual(receipt["stop_confirmation"]["actuator_target"], "last_native_command_retained")
                    self.assertEqual(receipt["stop_confirmation"]["settling_steps"], 0)

    def test_bootstrap_cuda_fault_restarts_the_worker_once(self):
        cuda = "bootstrap_failed: RuntimeError: CUDA error: operation failed due to a previous error during capture"
        for errors, expected_starts, raises in (
                ([cuda], 2, False),
                ([cuda, cuda], 2, True),
                (["bootstrap_failed: OutOfMemoryError: CUDA out of memory"], 1, True),
                (["bootstrap_failed: ValueError: bad layout"], 1, True)):
            with self.subTest(errors=errors), tempfile.TemporaryDirectory() as directory:
                backend = RobotBackend(robodojo_root=directory, run_dir=Path(directory) / "worker")
                pending, starts, closes = list(errors), [], []
                def start_process(command, **kwargs):
                    starts.append(command)
                    backend.process = object()
                    if pending:
                        raise RuntimeError(pending.pop(0))
                    backend.capabilities = {"ready": True}
                    return backend.capabilities
                backend.start_process = start_process
                backend.close = lambda: closes.append(1) or {"process_exit_confirmed": True}
                if raises:
                    with self.assertRaises(RuntimeError):
                        backend.start()
                else:
                    self.assertEqual(backend.start(), {"ready": True})
                    record = json.loads((Path(directory) / "worker" / "bootstrap-retry.json").read_text())
                    self.assertEqual(record["attempt"], 2)
                self.assertEqual(len(starts), expected_starts)
                self.assertEqual(len(closes), expected_starts - 1)

    def test_reset_preflight_refuses_without_worker_or_world_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = RobotBackend(robodojo_root=directory, run_dir=Path(directory) / "worker")
            requests = []
            backend._request = lambda operation, payload=None, **kwargs: requests.append((operation, payload)) or {"reset": True}
            backend.close = lambda: self.fail("preflight must not close the current worker")
            for seed, layout_id in ((True, 0), (0, False), (-1, 0), (0, -1), (0.0, 0), (0, "0")):
                with self.subTest(seed=seed, layout_id=layout_id), self.assertRaisesRegex(ValueError, "nonnegative integers"):
                    backend.validate_reset(seed, layout_id)
            with self.assertRaisesRegex(ValueError, "reset_layout_unavailable"):
                backend.reset(seed=0, layout_id=999)
            self.assertEqual(requests, [])
            self.assertEqual((backend.seed, backend.layout_id), (0, 0))
            self.assertIsNone(backend.process)
            self.assertFalse(backend.run_dir.exists())

            layout = Path(directory) / "Assets/Eval_Layout/RoboDojo/arx_x5/0/stack_bowls_0.json"
            layout.parent.mkdir(parents=True)
            layout.write_text("{}")
            self.assertEqual(backend.validate_reset(0, 0), layout)
            self.assertEqual(requests, [])
            backend._request = lambda operation, payload=None, **kwargs: requests.append((operation, payload)) or {"robot_state": {"arms": {}}}
            lifecycle = []
            backend.close = lambda: lifecycle.append("close") or {"process_exit_confirmed": True}
            backend.start = lambda: lifecycle.append("start")
            result = backend.reset()
            self.assertEqual(result["reset_strategy"], "fresh_worker_official_eval_reset")
            self.assertEqual(lifecycle, ["close", "start"])
            self.assertEqual(requests, [("observe", None), ("observe", None)])

    def test_pinned_reference_tcp_and_orientation(self):
        import numpy as np
        from ni_rua.robot.model import roboprobe_pose as pose
        provenance = json.loads((REPO_ROOT / "third_party/roboprobe/provenance.json").read_text())
        self.assertEqual(hashlib.sha256(Path(pose.__file__).read_bytes()).hexdigest(), provenance["source_files"]["pose.py"]["sha256"])
        quaternion = pose.angles_to_quat(0, 0, 0)
        np.testing.assert_allclose(pose.tool_axis(quaternion), [0, 0, -1], atol=1e-12)
        original = np.r_[[0.1, 0.2, 1.0], quaternion]
        values = pose.pose_to_values(original)
        self.assertAlmostEqual(values["z"], 1.0 - 0.1501)
        np.testing.assert_allclose(pose.values_to_pose(values), original, atol=1e-12)
        # Hold the grasp point while yaw changes: the inverse uses the target tool axis.
        values["pitch_deg"], values["yaw_deg"] = 30, 45
        rebuilt = pose.values_to_pose(values)
        np.testing.assert_allclose(rebuilt[:3] + 0.1501 * pose.tool_axis(rebuilt[3:]), [values[axis] for axis in ("x", "y", "z")], atol=1e-12)

    def test_moving_camera_uses_measured_link_instead_of_stale_usd(self):
        import numpy as np
        adapter = object.__new__(RoboDojoAdapter)
        adapter.np = np
        def matrix(pose):
            result = np.eye(4)
            result[:3, 3] = pose[:3]
            return result
        adapter._matrix = matrix
        camera = SimpleNamespace(get_world_pose=lambda camera_axes: (np.zeros(3), np.array([1, 0, 0, 0])))
        manager = SimpleNamespace(camera_names=[["cam_left_wrist"]], cameras=[[camera]],
            camera_config={"cam_left_wrist": SimpleNamespace(camera={"mount_link": "robot0/camera"})})
        origin = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.zeros(3)))
        adapter.env = SimpleNamespace(camera_manager=manager, env_origins=[origin])
        robot = SimpleNamespace()
        adapter.robot = SimpleNamespace(robot_list=[robot], robot_key=[SimpleNamespace(body_names=["camera", "base_link"])],
            get_robot_by_arm_name=lambda name: robot,
            get_link_pose=lambda robot, link, ids, is_relative: {0: np.array([1 if link == "camera" else 0, 0, 0, 1, 0, 0, 0])})
        adapter.robot_model = {"base_link": "base_link", "joints": []}
        mount = np.eye(4)
        mount[1, 3] = 0.2
        adapter.camera_calibration = {"cameras": {"cam_left_wrist": {"T_link_camera": mount.tolist()}}}
        sample = adapter._camera_pose("cam_left_wrist")
        self.assertEqual(sample["source_kind"], "robot_kinematics")
        np.testing.assert_allclose(np.array(sample["T_world_camera"])[:3, 3], [1, 0.2, 0])
        self.assertGreater(sample["native_usd_pose_diagnostic"]["translation_residual_m"], 1)

    def test_camera_calibration_enforces_square_pixel_vertical_focal_length(self):
        import numpy as np
        adapter = object.__new__(RoboDojoAdapter)
        adapter.np = np
        adapter._camera_pose = lambda camera_id: {"T_world_camera": np.eye(4).tolist()}
        head_k = np.array([[288.13253784, 0.0, 320.0], [0.0, 336.46432495, 240.0], [0.0, 0.0, 1.0]])
        wrist_k = np.array([[397.04125977, 0.0, 320.0], [0.0, 397.04125977, 240.0], [0.0, 0.0, 1.0]])
        head = SimpleNamespace(get_intrinsics_matrix=lambda: head_k, get_resolution=lambda: (640, 480))
        wrist = SimpleNamespace(get_intrinsics_matrix=lambda: wrist_k, get_resolution=lambda: (640, 480))
        adapter.env = SimpleNamespace(camera_manager=SimpleNamespace(
            camera_names=[["cam_head", "cam_left_wrist"]], cameras=[[head, wrist]]))
        calibration = adapter._camera_calibration()
        np.testing.assert_allclose(calibration["cameras"]["cam_head"]["K"],
            [[288.13253784, 0.0, 320.0], [0.0, 288.13253784, 240.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(calibration["cameras"]["cam_left_wrist"]["K"], wrist_k)
        self.assertEqual(head_k[1, 1], 336.46432495)

    def test_ipc_separates_native_logs_and_cancellation(self):
        # This fixture verifies transport only; it does not claim simulated motion.
        code = '''import json, sys
for line in sys.stdin:
    request = json.loads(line)
    if request['op'] == 'execute':
        print('native simulator log', flush=True)
        print('NI_RUA_JSON ' + json.dumps({'id': request['id'], 'progress': {'phase': 'started'}}), flush=True)
        cancel = json.loads(sys.stdin.readline())
        assert cancel['op'] == 'cancel' and cancel['target'] == request['id']
        print('NI_RUA_JSON ' + json.dumps({'id': request['id'], 'result': {'status': 'stopped'}}), flush=True)
    else:
        print('NI_RUA_JSON ' + json.dumps({'id': request['id'], 'result': {'closed': True}}), flush=True)
        break
'''
        with tempfile.TemporaryDirectory() as directory:
            backend = RobotBackend(run_dir=directory)
            backend.log = (Path(directory) / "robot-worker.log").open("w", buffering=1)
            backend.process = subprocess.Popen([sys.executable, "-u", "-c", code], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=backend.log, text=True)
            threading.Thread(target=backend._read, daemon=True).start()
            cancel = threading.Event()
            updates = []
            def progress(update):
                updates.append(update)
                cancel.set()
            try:
                receipt = backend.execute(self.action(), progress, cancel)
                self.assertEqual(receipt["status"], "stopped")
                self.assertEqual(updates, [{"phase": "started"}])
            finally:
                backend.close()
            self.assertIn("native simulator log", (Path(directory) / "robot-worker.log").read_text())

    def test_native_gpu_oom_interrupts_pending_request_without_exposing_logs(self):
        # Exercise real file-backed stderr and IPC: OOM need not kill the worker.
        signatures = ["ERROR_OUT_OF_DEVICE_MEMORY", "Out of GPU memory allocating resource '/private/scene.usd'",
                      "CUDA error: out of memory", "PxgCudaDeviceMemoryAllocator failed to allocate memory 268435456 bytes! Result = 2",
                      "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 200.00 MiB. GPU 0 has /private"]
        for signature in signatures:
            self.assertIsNotNone(native_gpu_oom_signature(signature), signature)
        self.assertIsNone(native_gpu_oom_signature("Warning: failed to open display; GPU memory statistics unavailable"))
        popen = subprocess.Popen
        signature = signatures[1]
        with tempfile.TemporaryDirectory() as directory:
            code = '''import json, sys
for line in sys.stdin:
    request = json.loads(line)
    if request['op'] == 'ready':
        print('Warning: failed to open display; GPU memory statistics unavailable', file=sys.stderr, flush=True)
        print('NI_RUA_JSON ' + json.dumps({'id': request['id'], 'result': {'ready': True}}), flush=True)
    elif request['op'] == 'close':
        print('NI_RUA_JSON ' + json.dumps({'id': request['id'], 'result': {'closed': True}}), flush=True)
        break
    else:
        print(sys.argv[1], file=sys.stderr, flush=True)
'''
            backend = RobotBackend(robodojo_root=directory, run_dir=directory)
            def launch(*args, **kwargs):
                return popen([sys.executable, "-u", "-c", code, signature], **kwargs)
            started = time.monotonic()
            try:
                with patch("ni_rua.robot.backends.robodojo.robot_process.subprocess.Popen", side_effect=launch):
                    self.assertEqual(backend.start(), {"ready": True})
                with self.assertRaisesRegex(RuntimeError, "robot_gpu_out_of_memory.*robot status unknown") as raised:
                    backend.execute(self.action())
                self.assertNotIn(signature, str(raised.exception))
                self.assertEqual(backend.pending, {})
                with self.assertRaisesRegex(RuntimeError, "robot_gpu_out_of_memory"):
                    backend.observe()
            finally:
                backend.close()
            self.assertLess(time.monotonic() - started, 5)
            self.assertIn(signature, (Path(directory) / "robot-worker.log").read_text())

    def test_planner_oom_reply_marks_robot_failed_without_log_evidence(self):
        # cuRobo/torch OOM is caught by the worker: a failed receipt or error reply, nothing on stderr.
        popen = subprocess.Popen
        replies = {
            "receipt": ("execute", {"result": {"status": "failed", "reason": "backend_error: OutOfMemoryError: CUDA out of memory. /private"}}),
            "error": ("observe", {"error": "OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB"}),
            "quoted": ("execute", {"result": {"status": "failed", "reason": "backend_error: ValueError: label 'OutOfMemoryError: CUDA out of memory'"}}),
        }
        for name, (operation, reply) in replies.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                code = '''import json, sys
reply = json.loads(sys.argv[1])
for line in sys.stdin:
    request = json.loads(line)
    body = {'result': {'ready': True}} if request['op'] in ('ready', 'close') else reply
    print('NI_RUA_JSON ' + json.dumps({'id': request['id'], **body}), flush=True)
    if request['op'] == 'close': break
'''
                backend = RobotBackend(robodojo_root=directory, run_dir=directory)
                def launch(*args, **kwargs):
                    return popen([sys.executable, "-u", "-c", code, json.dumps(reply)], **kwargs)
                try:
                    with patch("ni_rua.robot.backends.robodojo.robot_process.subprocess.Popen", side_effect=launch):
                        backend.start()
                    call = (lambda: backend.execute(self.action())) if operation == "execute" else backend.observe
                    if name == "quoted":
                        self.assertEqual(call()["status"], "failed")
                        self.assertIsNone(backend.failure)
                        continue
                    with self.assertRaisesRegex(RuntimeError, "^robot_gpu_out_of_memory:.*do not replay action") as raised:
                        call()
                    self.assertNotIn("/private", str(raised.exception))
                    self.assertTrue(backend.failure.startswith("robot_gpu_out_of_memory:"))
                    with self.assertRaisesRegex(RuntimeError, "robot_gpu_out_of_memory"):
                        backend.observe()
                finally:
                    backend.close()

    def test_native_oom_precedes_bootstrap_error_or_eof(self):
        popen = subprocess.Popen
        code = '''import json, sys
request = json.loads(sys.stdin.readline())
print("Out of GPU memory allocating resource '/private/scene.usd'", file=sys.stderr, flush=True)
if sys.argv[1] == 'reply':
    print('NI_RUA_JSON ' + json.dumps({'id': request['id'], 'error': 'bootstrap_failed: downstream NoneType'}), flush=True)
'''
        for mode in ("reply", "eof"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                backend = RobotBackend(robodojo_root=directory, run_dir=directory)
                def launch(*args, **kwargs):
                    return popen([sys.executable, "-u", "-c", code, mode], **kwargs)
                try:
                    with patch("ni_rua.robot.backends.robodojo.robot_process.subprocess.Popen", side_effect=launch):
                        with self.assertRaisesRegex(RuntimeError, "^robot_gpu_out_of_memory:.*robot status unknown") as raised:
                            backend.start()
                    self.assertNotIn("NoneType", str(raised.exception))
                    self.assertNotIn("scene.usd", str(raised.exception))
                    self.assertEqual(backend.pending, {})
                finally:
                    backend.close()

    def test_previous_worker_oom_does_not_poison_new_worker(self):
        popen = subprocess.Popen
        code = '''import json, sys
for line in sys.stdin:
    request = json.loads(line)
    print('NI_RUA_JSON ' + json.dumps({'id': request['id'], 'result': {'healthy': True}}), flush=True)
    if request['op'] == 'close': break
'''
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "robot-worker.log"
            log.write_text("previous worker: CUDA error: out of memory\n")
            backend = RobotBackend(robodojo_root=directory, run_dir=directory)
            def launch(*args, **kwargs):
                return popen([sys.executable, "-u", "-c", code], **kwargs)
            try:
                with patch("ni_rua.robot.backends.robodojo.robot_process.subprocess.Popen", side_effect=launch):
                    self.assertEqual(backend.start(), {"healthy": True})
                self.assertEqual(backend.observe(), {"healthy": True})
                self.assertIsNone(backend.failure)
            finally:
                backend.close()
            self.assertIn("previous worker", log.read_text())

    def test_cleanup_does_not_wait_for_descendant_inheriting_stderr(self):
        popen = subprocess.Popen
        code = '''import json, subprocess, sys
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
for line in sys.stdin:
    request = json.loads(line)
    print('NI_RUA_JSON ' + json.dumps({'id': request['id'], 'result': {'child': child.pid}}), flush=True)
    if request['op'] == 'close': break
'''
        with tempfile.TemporaryDirectory() as directory:
            backend = RobotBackend(robodojo_root=directory, run_dir=directory)
            descendant, closer = None, None
            def launch(*args, **kwargs):
                return popen([sys.executable, "-u", "-c", code], **kwargs)
            try:
                with patch("ni_rua.robot.backends.robodojo.robot_process.subprocess.Popen", side_effect=launch):
                    descendant = backend.start()["child"]
                closer = threading.Thread(target=backend.close)
                closer.start()
                closer.join(timeout=2)
                self.assertFalse(closer.is_alive(), "cleanup waited for an inherited stderr descriptor")
                self.assertIsNotNone(backend.process.poll())
                os.kill(descendant, 0)  # The descendant remains alive; only its parent was closed.
            finally:
                if descendant is not None:
                    try: os.kill(descendant, signal.SIGTERM)
                    except ProcessLookupError: pass
                if closer is not None: closer.join(timeout=5)
                else: backend.close()

# === tests/test_scene_truth.py ===

import ast
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ni_rua.core.config import Config
from ni_rua.core.contracts import canonical
from ni_rua.core.observation_policy import reject_privileged
from ni_rua.robot.backends.robodojo.scene_truth import FILENAME, SceneTruthRecorder, truth_path
from ni_rua.core.storage.store import Store

_spec = importlib.util.spec_from_file_location("score_geometry_truth", ROOT / "scripts" / "score_geometry_truth.py")
scorer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scorer)

RAW = "raw-%08d-0000-4000-8000-000000000000"
IDENTITY = dict(run_id="r", episode_id="ep-1", env_instance_id="env-1", env_generation=1)


class FakeRecords:
    def __init__(self, records, metadata):
        self.layout_records_by_env, self.metadata_by_env = [records], [metadata]


class FakeLayout:
    """LayoutManager accessors used by the recorder (RoboDojo env/scene_manager/layout_manager.py)."""

    def __init__(self, poses, fail=False):
        self.poses, self.fail = poses, fail
        box = dict(geometry=dict(oriented_bbox=dict(vertices=[[x, y, z] for x in (-.04, .04) for y in (-.03, .03) for z in (-.02, .02)])))
        self.object_records_by_type = {
            "Rigid": FakeRecords([dict(inst_name="obj_0", label="red_cup", scale=[1, 1, 1]),
                                  dict(inst_name="obj_1", label=None, scale=[2, 2, 2])],
                                 {"obj_0": dict(box, model_name="cup", model_id="m0"), "obj_1": dict(box, model_name="block")})}
        self.table_info = [dict(pos=[0, 0, .74], size=[-.6, -.4, .6, .4], height=.765)]
        self.tables = [["table_0"]]

    def get_scene_object(self, env_idx, inst):
        return None

    def get_instance_pose(self, env_idx, inst_name):
        if self.fail:
            raise RuntimeError("pose unavailable")
        return self.poses[inst_name]


class FakeEnv:
    def __init__(self, layout):
        self.scene_manager = type("SM", (), {"layout_manager": layout})()


STATE = {"arms": {"left": {"ee_pose": {"position_m": [0.1, 0.2, 0.9], "quaternion_xyzw": [0, 0, 0, 1],
                                       "tcp_config_id": "arx-x5-link6-v1"}, "gripper": {"position_m": [0.02, 0.02]}}}}


class WriterTest(unittest.TestCase):
    def test_truth_path_under_private(self):
        self.assertEqual(truth_path(Path("/x/run/private/robot")), Path("/x/run/private") / FILENAME)
        self.assertEqual(truth_path(Path("/x/run")), Path("/x/run/private") / FILENAME)

    def test_record_fields_and_write_only(self):
        poses = {"obj_0": ([0.1, -0.2, 0.785], [1, 0, 0, 0]),                       # wxyz identity
                 "obj_1": ([0.3, 0.0, 0.805], [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)])}  # yaw 90
        with tempfile.TemporaryDirectory() as tmp:
            path = truth_path(Path(tmp) / "private" / "robot")
            recorder = SceneTruthRecorder(FakeEnv(FakeLayout(poses)), path, grasp_offset_m=0.15, usd_bbox=None)
            self.assertIsNone(recorder.record(RAW % 1, 7, STATE, 0.28))
            self.assertIsNone(recorder.record(RAW % 2, 9, STATE, 0.36))
            lines = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(lines), 2)
        line = lines[0]
        self.assertEqual((line["raw_observation_id"], line["sim_step"], line["audience"], line["errors"]), (RAW % 1, 7, "analysis_only", []))
        cup, block = line["objects"]
        self.assertEqual((cup["name"], cup["category"], cup["labelled"]), ("red_cup", "cup", True))
        self.assertEqual(cup["quaternion_xyzw"], [0, 0, 0, 1])
        self.assertAlmostEqual(cup["top_z_m"], 0.805)
        self.assertAlmostEqual(cup["bottom_z_m"], 0.765)          # resting on the table top
        self.assertEqual(cup["bbox_size_m"], [0.08, 0.06, 0.04])
        self.assertEqual(block["name"], "obj_1")                  # unlabelled: instance name
        self.assertEqual(block["bbox_size_m"], [0.16, 0.12, 0.08])  # scale applied
        self.assertAlmostEqual(block["top_z_m"], 0.845)
        self.assertAlmostEqual(line["table"]["top_z_m"], 0.765)
        self.assertEqual(line["robot"]["left"]["grasp_center_m"], [0.25, 0.2, 0.9])
        self.assertEqual(line["robot"]["left"]["gripper_position_m"], [0.02, 0.02])

    def test_failures_never_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "private" / FILENAME
            recorder = SceneTruthRecorder(FakeEnv(FakeLayout({}, fail=True)), path, grasp_offset_m=0.15, usd_bbox=None)
            self.assertIsNone(recorder.record(RAW % 1, 0, None))
            line = json.loads(path.read_text())
            self.assertTrue(line["errors"] and line["objects"][0]["position_m"] is None)
            broken = SceneTruthRecorder(object(), Path(tmp) / "file_as_dir" / "x", grasp_offset_m=0.15, usd_bbox=None)
            (Path(tmp) / "file_as_dir").write_text("")
            self.assertIsNone(broken.record(RAW % 2, 0, STATE))
            self.assertIsNotNone(broken.last_error)


class IsolationTest(unittest.TestCase):
    """The truth log is write-only: no model-facing module can reach it."""

    ALLOWED = {"robot/backends/robodojo/scene_truth.py": "writer",
               "robot/backends/robodojo/robot_worker.py": "calls the writer, discards its value",
               "robot/backends/robodojo/robodojo.py": "passes the output path to the worker",
               "core/config/sections.py": "switch",
               "core/observation_policy.py": "rejects a scene_truth key in actor payloads",
               "perception/external/obs3d.py": "reads the switch to log the attached (model-visible) table as an internal event; no truth"}

    def test_only_writer_side_modules_mention_it(self):
        found = {path.relative_to(ROOT / "ni_rua").as_posix() for path in (ROOT / "ni_rua").rglob("*.py")
                 if "scene_truth" in path.read_text(encoding="utf-8")}
        self.assertTrue(found.issubset(set(self.ALLOWED)), f"unexpected modules: {found - set(self.ALLOWED)}")
        readers = {path.relative_to(ROOT).as_posix() for folder in ("ni_rua", "scripts") for path in (ROOT / folder).rglob("*.py")
                   if FILENAME in path.read_text(encoding="utf-8")}
        self.assertTrue(readers.issubset({"ni_rua/robot/backends/robodojo/scene_truth.py", "scripts/score_geometry_truth.py",
                                          "scripts/probes/scene_truth_probe.py"}))

    def test_worker_discards_record_value(self):
        tree = ast.parse((ROOT / "ni_rua" / "robot/backends/robodojo/robot_worker.py").read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr == "record" and isinstance(node.func.value, ast.Attribute)
                 and node.func.value.attr == "scene_truth"]
        self.assertEqual(len(calls), 1)
        statements = [node for node in ast.walk(tree) if isinstance(node, ast.Expr) and node.value is calls[0]]
        self.assertEqual(len(statements), 1, "the recorder's return value must be discarded")

    def test_privileged_key_rejected_and_profile_unchanged(self):
        with self.assertRaises(ValueError):
            reject_privileged({"robot_state": {}, "scene_truth": {"objects": []}})
        with self.assertRaises(ValueError):
            reject_privileged({"value": [{"scene_truth": 1}]})
        on = Config.from_dict({"analysis": {"scene_truth_log": True}})
        off = Config.from_dict({})
        self.assertFalse(off.analysis.scene_truth_log)
        self.assertEqual(on.features, off.features)               # observation profile, tools, payload: unchanged


# ------------------------------------------------------------------ scorer
def _truth_line(raw, step, cup_xy=(0.1, -0.2)):
    def obj(name, p, size):
        half = [s / 2 for s in size]
        return dict(name=name, inst=name, category=name, position_m=list(p), quaternion_xyzw=[0, 0, 0, 1],
                    centroid_m=list(p), top_z_m=p[2] + half[2], bottom_z_m=p[2] - half[2],
                    bbox_local_min_m=[-h for h in half], bbox_local_max_m=half, bbox_size_m=list(size))
    return dict(raw_observation_id=raw, sim_step=step, objects=[
        obj("cup", (cup_xy[0], cup_xy[1], 0.785), (0.08, 0.08, 0.04)),
        obj("block", (0.3, 0.0, 0.79), (0.05, 0.05, 0.05)),
        obj("pen_a", (-0.2, 0.1, 0.77), (0.1, 0.01, 0.01)),
        obj("pen_b", (-0.2, 0.125, 0.77), (0.1, 0.01, 0.01)),
        dict(obj("camera_stand", (0.33, 0.0, 1.0), (0.3, 0.5, 0.1)), category="camera_stand")], table=dict(top_z_m=0.765))


V5_COLUMNS = ["id", "centre_m", "top_z", "bottom_z", "length", "width", "yaw_deg", "tilt_deg", "support", "quality"]
V5_ROWS = [["o1", [0.104, -0.197, 0.79], 0.80, 0.765, 0.08, 0.08, 0.0, None, "table", "ok"],
           ["o2", [0.3, 0.01, 0.80], 0.82, 0.765, 0.05, 0.05, 10.0, None, "table", "low:floating_rescaled_to_table"]]


def obs3d_text(observation_id, table):
    return canonical({"object_table": table, "observation_id": observation_id})


def build_run(root, with_truth=True, object_table=False, obs3d=None):
    """obs3d: None, 'reply_text' (tables only inside reply wire_response, as before the analysis event) or 'event'."""
    store = Store(root, IDENTITY)
    for k, step in ((1, 0), (2, 40)):
        store.save("observation", f"obs-{k}", dict(observation_id=f"obs-{k}", sim_step=step,
                   cameras=[dict(camera_id=c, frame_id=(RAW % k) + "-" + c) for c in ("cam_head", "cam_left_wrist")]))

    def tool(identifier, data, role="actor"):
        ref = store.put_artifact(canonical(dict(data=data, kind="json", success=True)).encode(), "tool_result", "application/json", "observer")
        store.save("tool_result", identifier, dict(tool="geometry_query", role=role, raw_ref=ref, correlation={}))
    head = (RAW % 1) + "-cam_head"
    # VGGT default-mode pixel_plane on the cup: applied plane = VGGT centre 0.79, caller said 0.765
    tool("t1", dict(query_type="pixel_plane", status="conditional", frame_ids=[head], observation_ids=["obs-1"],
                    value=dict(point_m=[0.11, -0.2, 0.79], plane_origin_m=[0, 0, 0.79], plane_normal=[0, 0, 1],
                               plane_origin_alt_m=[0, 0, 0.765], alternate_point_m=[0.13, -0.19, 0.765],
                               plane_estimate=dict(status="ok", center_z_m=0.79, top_z_m=0.81, alternate_plane="caller_plane_origin_m"))))
    # caller-only pixel_plane on the block (no plane_estimate), table plane assumed
    tool("t2", dict(query_type="pixel_plane", status="conditional", frame_ids=[head], observation_ids=["obs-1"],
                    value=dict(point_m=[0.33, 0.0, 0.765], plane_origin_m=[0, 0, 0.765], plane_normal=[0, 0, 1])), role="program")
    # triangulate: one point near pen_a (ambiguous with pen_b), one near the block; joined via observation id only
    tool("t3", dict(query_type="triangulate", status="conditional", frame_ids=[], observation_ids=["obs-2"],
                    value=dict(points_m=[[-0.2, 0.11, 0.775], [0.3, 0.005, 0.81]], accepted_indices=[0, 2],
                               reprojection_error_px=[1.0, 9.0, 2.0])))
    request = dict(action_id="a1", based_on_observation_id="obs-1", stages=[
        dict(id="cup_above", kind="move", arm="left", action={}),
        dict(id="cup_pick", kind="move", arm="left", action={}),
        dict(id="close", kind="gripper", arm="left", action=dict(gripper_events=[dict(arm="left", goal_normalized=0)])),
        dict(id="cup_lift", kind="move", arm="left", action={})])
    def feedback(target):
        return dict(grasp_contact_feedback=dict(arms=dict(left=dict(requested_contact_position_m=target, available=True, position_error_m=0.004))))
    receipt = dict(status="completed", execution_start_observation_id="obs-1", stage_results=dict(
        cup_above=feedback([0.1, -0.2, 0.9]), cup_pick=dict(motion=feedback([0.102, -0.203, 0.80])), close={},
        cup_lift=feedback([0.1, -0.2, 0.95])))
    store.save("action_request", "a1", request)
    store.save("action_receipt", "a1", receipt)
    store.save("tool_call", "c1", dict(params=dict(arguments=dict(code="print(1)"))))
    if object_table:   # object table service v5 with caller labels: extra 'label' column and 'labels' field
        tool("t4", dict(query_type="object_table", status="conditional", frame_ids=[head], observation_ids=["obs-1"],
                        value=dict(source="vggt_object_table", measured=False, camera_id="cam_head", support_z_m=0.765,
                                   columns=V5_COLUMNS + ["label"], rows=[r + [l] for r, l in zip(V5_ROWS, ("cup", "block"))],
                                   labels=[dict(label="cup", status="ok", ids=["o1"]), dict(label="block", status="ok", ids=["o2"])],
                                   omitted=0, overlay_cameras=["cam_head"])))
    if obs3d:
        table = dict(columns=V5_COLUMNS, rows=V5_ROWS, table_z=0.765, note="head camera")
        later = dict(table, rows=[["o1", [0.25, 0.0, 0.79], 0.815, 0.765, 0.05, 0.05, 0.0, None, "table", "ok"]])

        def reply(identifier, tool_name, observation_id, part, call):
            items = [dict(type="inputText", text=canonical({"bundle": {"observation_id": observation_id}})),
                     dict(type="inputText", text=obs3d_text(observation_id, part))]
            ref = store.put_artifact(canonical(dict(data={}, kind="observation", success=True,
                                     wire_response=dict(contentItems=items, success=True))).encode(),
                                     "tool_result", "application/json", "observer")
            store.save("tool_result", identifier, dict(tool=tool_name, role="actor", raw_ref=ref,
                                                       correlation=dict(tool_call_id=call)))
        # action reply attaches obs-1; robot_observe of obs-1 reuses it; the action reply of obs-2 is unavailable first
        reply("r1", "robot_action", "obs-1", table, "k1")
        reply("r2", "robot_observe", "obs-1", table, "k2")
        reply("r3", "robot_wait", "obs-2", "unavailable: timeout", "k3")
        reply("r4", "robot_observe", "obs-2", later, "k4")
        if obs3d == "event":
            for call, observation_id, frame, value, kind, reused in (
                    ("k1", "obs-1", head, table, "action", False), ("k2", "obs-1", head, table, "observe", True),
                    ("k3", "obs-2", (RAW % 2) + "-cam_head", None, "action", False),
                    ("k4", "obs-2", (RAW % 2) + "-cam_head", later, "observe", False)):
                payload = dict(observation_id=observation_id, observation_ids=[observation_id], camera_id="cam_head",
                               reply=kind, reused=reused, frame_ids=[frame], available=value is not None)
                if value is None:
                    payload["unavailable"] = "unavailable: timeout"
                else:
                    payload.update(columns=value["columns"], rows=value["rows"], table_z=0.765, omitted=0)
                store.event("analysis.obs3d_object_table", payload, dict(tool_call_id=call), world_kind="internal")
    store.close()
    if with_truth:
        private = Path(root) / "private"
        private.mkdir(exist_ok=True)
        (private / FILENAME).write_text("\n".join(json.dumps(line) for line in (_truth_line(RAW % 1, 0), _truth_line(RAW % 2, 40))) + "\n")


class ScorerTest(unittest.TestCase):
    def test_scores_geometry_and_grasp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "case"
            build_run(root)
            before = sorted(p.name for p in root.iterdir())
            result = scorer.score_run(root)
            self.assertEqual(sorted(p.name for p in root.iterdir()), before)     # read-only on the run
        rows = {(r["tool_result_id"], r.get("point_index")): r for r in result["geometry_rows"]}
        vggt = rows[("t1", None)]
        self.assertEqual((vggt["object"], vggt["plane_source"], vggt["truth_join"], vggt["unambiguous"]), ("cup", "vggt", "frame_id", True))
        self.assertAlmostEqual(vggt["xy_error_m"], 0.01)
        self.assertAlmostEqual(vggt["z_error_centre_m"], 0.005)
        self.assertAlmostEqual(vggt["z_error_top_m"], -0.015)
        self.assertAlmostEqual(vggt["caller_plane_minus_truth_centre_z_m"], -0.02)
        self.assertAlmostEqual(vggt["vggt_centre_minus_truth_centre_z_m"], 0.005)
        self.assertAlmostEqual(vggt["vggt_top_minus_truth_top_z_m"], 0.005)
        self.assertTrue(vggt["inside_xy_footprint"])
        caller = rows[("t2", None)]
        self.assertEqual((caller["object"], caller["plane_source"], caller["role"]), ("block", "caller", "program"))
        self.assertAlmostEqual(caller["plane_applied_minus_truth_centre_z_m"], -0.025)
        self.assertFalse(caller["inside_xy_footprint"])
        pen = rows[("t3", 0)]
        self.assertEqual((pen["object"], pen["unambiguous"], pen["truth_join"], pen["truth_sim_step"]), ("pen_a", False, "observation_raw_id", 40))
        self.assertEqual(rows[("t3", 2)]["reprojection_error_px"], 2.0)
        grasp = [r for r in result["grasp_rows"] if r["is_grasp"]]
        self.assertEqual([(r["stage_id"], r["object"]) for r in grasp], [("cup_pick", "cup")])
        self.assertAlmostEqual(grasp[0]["xy_error_m"], math.hypot(0.002, 0.003), places=4)
        self.assertAlmostEqual(grasp[0]["z_error_top_m"], -0.005)
        self.assertEqual(len(result["grasp_rows"]), 3)
        summary = result["geometry"]
        self.assertEqual((summary["all"]["n"], summary["all"]["unambiguous"]), (4, 3))
        self.assertEqual(set(summary["by_plane_source"]), {"vggt", "caller", "triangulate"})
        self.assertEqual(summary["by_kind"]["triangulate"]["n"], 2)
        self.assertEqual(result["leak_audit"]["tool_call_arguments"], 0)
        text = scorer.markdown([result])
        self.assertIn("| case | plane=vggt | 1 |", text)

    def test_v5_object_table_rows_scored_as_before(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "case"
            build_run(root, object_table=True)
            result = scorer.score_run(root)
        rows = {r["object_id"]: r for r in result["geometry_rows"] if r["kind"] == "object_table"}
        self.assertEqual(set(rows), {"o1", "o2"})
        cup, block = rows["o1"], rows["o2"]
        self.assertEqual((cup["object"], cup["ot_label"], cup["plane_source"], cup["truth_join"]), ("cup", "cup", "object_table", "frame_id"))
        self.assertAlmostEqual(cup["xy_error_m"], 0.005)
        self.assertAlmostEqual(cup["z_error_centre_m"], 0.005)
        self.assertAlmostEqual(cup["z_error_top_m"], -0.015)
        self.assertAlmostEqual(cup["ot_top_minus_truth_top_z_m"], -0.005)
        self.assertEqual((block["object"], block["ot_quality"]), ("block", "low:floating_rescaled_to_table"))
        self.assertAlmostEqual(block["xy_error_m"], 0.01)
        self.assertAlmostEqual(block["z_error_centre_m"], 0.01)
        self.assertAlmostEqual(block["ot_top_minus_truth_top_z_m"], 0.005)
        summary = result["geometry"]
        self.assertEqual((summary["all"]["n"], summary["by_kind"]["object_table"]["n"]), (6, 2))
        self.assertEqual(summary["by_kind"]["object_table"]["ot_top_minus_truth_top_z_m"]["p50"], 0.005)
        text = scorer.markdown([result])
        line = next(l for l in text.splitlines() if "| kind=object_table |" in l)
        header = next(l for l in text.splitlines() if l.startswith("| episode | group |"))
        self.assertEqual(line.count("|"), header.count("|"))                    # one header cell per metric
        self.assertIn("table top - top z", header)

    def test_v1_columns_without_label_unchanged(self):
        points = scorer.table_points(V5_COLUMNS, V5_ROWS, "object_table")
        self.assertEqual([(p[0], p[1]["object_id"], "ot_label" in p[1]) for p in points],
                         [([0.104, -0.197, 0.79], "o1", False), ([0.3, 0.01, 0.80], "o2", False)])

    def check_obs3d(self, mode):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "case"
            build_run(root, obs3d=mode)
            result = scorer.score_run(root)
        rows = [r for r in result["geometry_rows"] if r["kind"] == "object_table_obs"]
        self.assertEqual([(r["observation_id"], r["object_id"], r["object"]) for r in rows],
                         [("obs-1", "o1", "cup"), ("obs-1", "o2", "block"), ("obs-2", "o1", "block")])
        cup = rows[0]
        self.assertAlmostEqual(cup["xy_error_m"], 0.005)
        self.assertAlmostEqual(cup["z_error_centre_m"], 0.005)
        self.assertAlmostEqual(cup["z_error_top_m"], -0.015)
        self.assertAlmostEqual(cup["ot_top_minus_truth_top_z_m"], -0.005)
        self.assertEqual((cup["obs3d_deliveries"], cup["role"], cup["obs3d_source"]), (2, "actor", mode))
        self.assertEqual((rows[2]["truth_sim_step"], rows[2]["obs3d_reply"]), (40, "observe"))
        self.assertAlmostEqual(rows[2]["xy_error_m"], 0.05)
        obs = result["obs3d"]
        self.assertEqual((obs["source"], obs["attachments"], obs["observations"], obs["unavailable_observations"],
                          obs["scored_rows"]), (mode, 4, 2, 0, 3))
        self.assertEqual(obs["reused_deliveries"], 1 if mode == "event" else None)
        summary = result["geometry"]
        self.assertEqual(summary["all"]["n"], 4)                                  # geometry_query calls only
        self.assertEqual(summary["by_kind"]["object_table_obs"]["n"], 3)
        self.assertEqual(summary["by_plane_source"]["object_table_obs"]["n"], 3)
        text = scorer.markdown([result])
        self.assertIn("| case | kind=object_table_obs | 3 |", text)
        self.assertIn(f"| case | {mode} | 4 | 2 | 0 |", text)
        return result

    def test_obs3d_attachments_from_events(self):
        result = self.check_obs3d("event")
        self.assertEqual({r["truth_join"] for r in result["geometry_rows"] if r["kind"] == "object_table_obs"}, {"frame_id"})

    def test_obs3d_attachments_from_reply_text_fallback(self):
        result = self.check_obs3d("reply_text")
        self.assertEqual({r["truth_join"] for r in result["geometry_rows"] if r["kind"] == "object_table_obs"},
                         {"observation_raw_id"})

    def test_footprint_uses_horizontal_axes(self):
        # y-up asset lying with local y vertical: footprint is local x/z
        item = dict(position_m=[0, 0, 0.8], quaternion_xyzw=[math.sin(math.pi / 4), 0, 0, math.cos(math.pi / 4)],
                    bbox_local_min_m=[-0.05, -0.01, -0.02], bbox_local_max_m=[0.05, 0.01, 0.02])
        self.assertTrue(scorer.inside_footprint(item, [0.04, 0.015, 0.81]))
        self.assertFalse(scorer.inside_footprint(item, [0.04, 0.03, 0.81]))

    def test_missing_truth_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "case"
            build_run(root, with_truth=False)
            result = scorer.score_run(root)
            self.assertIn("warning", result)
            self.assertEqual(result["geometry"]["all"]["matched"], 0)
            self.assertEqual(scorer.find_runs([tmp]), [root])

    def test_cli_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs" / "case"
            root.parent.mkdir()
            build_run(root)
            out = Path(tmp) / "out"
            out.mkdir()
            self.assertEqual(scorer.main([str(root.parent), "--out-json", str(out / "s.json"), "--out-md", str(out / "s.md")]), 0)
            document = json.loads((out / "s.json").read_text())
            self.assertEqual(document["pooled"]["geometry"]["n"], 4)
            self.assertEqual(document["pooled"]["grasp"]["n"], 1)


class WorkerIpcAndMotionExceptionContractTests(unittest.TestCase):
    def test_ipc_structured_error_round_trip_read_input_dispatch_error_and_start_cleanup(self):
        from ni_rua.core.presentation.actor_presentation import ActorPresentation
        from ni_rua.core.storage.store import Store
        from ni_rua.core.tool_errors import WorkerRemoteError, error_payload
        from ni_rua.robot.ipc.process import WorkerProcess

        worker_script = (
            "import json, sys\n"
            f"sys.path[:0] = [{str(REPO_ROOT)!r}]\n"
            "from ni_rua.core.tool_errors import ToolInputError\n"
            "from ni_rua.robot.ipc.protocol import WIRE_PREFIX, serve_worker\n"
            "class IKServiceError(RuntimeError):\n"
            "    pass\n"
            "class Adapter:\n"
            "    def capabilities(self): return {'ready': True}\n"
            "    def observe(self): raise TypeError('bad operand in observe')\n"
            "    def preview_batch(self, p):\n"
            "        mode = p.get('case')\n"
            "        if mode == 'stale': raise ValueError('stale_observation')\n"
            "        if mode == 'ik_down': raise IKServiceError('ik_service_unreachable')\n"
            "        if mode == 'key_bug': raise KeyError('internal_missing_slot')\n"
            "        if mode == 'unpack_bug': raise ValueError('not enough values to unpack (expected 3, got 2)')\n"
            "        if mode == 'tool_input': raise ToolInputError('invalid_stage', field_path='/stages/0', suggestion='fix stage')\n"
            "        return {'ok': True}\n"
            "    def close(self, progress=None): pass\n"
            "serve_worker(lambda _: Adapter(), None, lambda m: print(WIRE_PREFIX + json.dumps(m), flush=True))\n"
        )
        run_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        proc = WorkerProcess(run_dir)
        self.addCleanup(proc.close)
        caps = proc.start_process([sys.executable, "-u", "-c", worker_script], cwd=REPO_ROOT, environment=dict(os.environ))
        self.assertEqual(caps, {"ready": True})

        with self.assertRaises(WorkerRemoteError) as cm_type:
            proc._request("observe")
        self.assertIsInstance(cm_type.exception, RuntimeError)
        self.assertEqual(cm_type.exception.category, "internal_runtime_error")
        self.assertEqual(cm_type.exception.remote_type, "TypeError")

        for case, expected_cat, expected_code, expected_type in (
            ("stale", "input_error", "stale_observation", "ValueError"),
            ("ik_down", "infrastructure_error", "ik_service_unreachable", "IKServiceError"),
            ("key_bug", "internal_runtime_error", "internal_runtime_error", "KeyError"),
            ("unpack_bug", "internal_runtime_error", "internal_runtime_error", "ValueError"),
            ("tool_input", "input_error", "invalid_stage", "ToolInputError"),
        ):
            with self.subTest(case=case), self.assertRaises(WorkerRemoteError) as cm:
                proc._request("preview_batch", {"case": case})
            self.assertEqual(cm.exception.category, expected_cat)
            self.assertEqual(cm.exception.code, expected_code)
            self.assertEqual(cm.exception.remote_type, expected_type)

        # DEF-11: malformed request with id but missing 'op' returns an error frame immediately instead of hanging.
        req_id = "missing-op-req"
        done_evt = threading.Event()
        reply_box = {}
        with proc.lock:
            proc.pending[req_id] = (done_evt, reply_box, None)
        try:
            proc._send({"id": req_id}, time.monotonic() + 5.0)
            self.assertTrue(done_evt.wait(timeout=5.0), "read_input swallowed error without replying")
            self.assertIn("error", reply_box)
            self.assertEqual(reply_box["error_detail"]["category"], "internal_runtime_error")
        finally:
            with proc.lock:
                proc.pending.pop(req_id, None)

        # DEF-17: worker disconnect error exposes zero host <run_dir> path characters.
        proc.process.kill()
        proc.process.wait(timeout=5)
        proc.reader.join(timeout=5)
        self.assertNotIn(str(run_dir), proc.failure or "")
        store_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        store = Store(store_dir, dict(run_id="r", episode_id="e", env_instance_id="i", env_generation=1))
        self.addCleanup(store.close)
        pres = ActorPresentation(store)
        with self.assertRaises(WorkerRemoteError) as cm_disc:
            proc._request("observe")
        presented_err = pres.present(error_payload(cm_disc.exception, tool="robot_observe", arguments={}))
        self.assertNotIn(str(run_dir), json.dumps(presented_err))

        # DEF-12: start_process cleans up live child & log when _request('ready') times out, allowing clean restart.
        retry_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        retry_proc = WorkerProcess(retry_dir)
        self.addCleanup(retry_proc.close)
        hang_script = "import time\ntime.sleep(30)\n"
        orig_req = retry_proc._request
        with patch.object(retry_proc, "_request", side_effect=lambda op, **kw: orig_req(op, timeout=0.15) if op == "ready" else orig_req(op, **kw)):
            with self.assertRaises(TimeoutError):
                retry_proc.start_process([sys.executable, "-u", "-c", hang_script], cwd=REPO_ROOT, environment=dict(os.environ))
        self.assertIsNone(retry_proc.process)
        self.assertIsNone(retry_proc.capabilities)
        self.assertTrue(retry_proc.close_report["process_exit_confirmed"])
        ok_caps = retry_proc.start_process([sys.executable, "-u", "-c", worker_script], cwd=REPO_ROOT, environment=dict(os.environ))
        self.assertEqual(ok_caps, {"ready": True})

    def test_gpu_oom_propagates_from_planner_witness_motion_failure_and_tilt_probe(self):
        from types import SimpleNamespace
        from ni_rua.robot.motion.batch_motion import _probe_feasible_approach_tilt_angles
        from ni_rua.robot.motion.motion_failure import failure_feedback
        from ni_rua.robot.motion.planner_witness import capture_candidate_witness

        class OutOfMemoryError(RuntimeError):
            pass

        class OomMotionPlanner:
            joint_names = ["j1"]

            @property
            def kinematics(self):
                raise OutOfMemoryError("CUDA out of memory")

        with self.assertRaises(OutOfMemoryError):
            capture_candidate_witness(SimpleNamespace(motion_planner=OomMotionPlanner()), [[0.0]], ["j1"])

        diag = [{
            "stage": "ik",
            "waypoint_index": 0,
            "candidate": {"q_rad": [[0.0] * 6], "joint_names": [f"j{i}" for i in range(6)], "selection_score": 0.1},
        }]
        action = {"path": {"waypoints": [{"left": {"position_m": [0.2, 0.0, 0.8], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]}}]}}
        with self.assertRaises(OutOfMemoryError):
            failure_feedback(
                action,
                "left",
                0,
                diag,
                "self_only",
                witness=lambda *a, **kw: (_ for _ in ()).throw(OutOfMemoryError("CUDA out of memory")),
            )

        with patch(
            "ni_rua.robot.motion.batch_motion._open_jaw_tilted_grasp_candidates",
            return_value=[({"gripper_events": [], "path": {"waypoints": [{"left": {}}]}}, {}, [0.0, 0.0, 0.0])],
        ), patch(
            "ni_rua.robot.motion.scene_obstacles.plan_move_with_obstacle_clearance",
            side_effect=OutOfMemoryError("CUDA out of memory"),
        ):
            with self.assertRaises(OutOfMemoryError):
                _probe_feasible_approach_tilt_angles(
                    adapter=SimpleNamespace(),
                    robot=SimpleNamespace(),
                    arm="left",
                    slot_dof=6,
                    orig_own={},
                    orig_target={},
                    start_pos_m=[0.0, 0.0, 0.0],
                    expected_q=[0.0] * 6,
                    start_jaw=1.0,
                    obstacles=[],
                    contact_entity_id=None,
                    before_state={"arms": {"left": {"q_rad": [0.0] * 6}}},
                    rows_so_far=0,
                    cap=100,
                    deadline=time.monotonic() + 5.0,
                    joint_travel_options={},
                    options={},
                )

    def test_joint_replan_geometry_and_motion_guards_propagate_internal_bugs_and_preserve_unknown_status(self):
        import numpy as np
        from types import SimpleNamespace
        from ni_rua.core.tool_errors import InternalToolRuntimeError
        from ni_rua.robot.model.geometry import GeometryTools
        from ni_rua.robot.motion.batch_geometry import review_batch
        from ni_rua.robot.motion.batch_motion import _populate_planned_positions
        from ni_rua.robot.motion.batch_scheduler import summarize_batch
        from ni_rua.robot.motion.grip_guard import _measured_physical_jaw
        from ni_rua.robot.motion.joint_replan import replan_from_ik_candidate, select_joint_replan
        from ni_rua.robot.motion.trajectory_evidence import build_trajectory_evidence
        from ni_rua.robot.backends.robodojo.scene_truth import SceneTruthRecorder

        # DEF-05: joint_replan propagates TypeError in callbacks, while None position is domain validation failure.
        limits = [[-6.28, 6.28], [-6.28, 6.28]]
        current = [0.0, 0.0]
        orig = [[0.0, 0.0], [6.20, 0.0]]
        eye_links = lambda _q: {"l1": np.eye(4)}
        with self.assertRaises(TypeError):
            select_joint_replan(
                original=orig,
                current=current,
                limits=limits,
                joint_types=["revolute", "revolute"],
                transforms=lambda _q: (_ for _ in ()).throw(TypeError("broken fk callback")),
                pose_error=lambda _q: [0.0, 0.0],
                resample=lambda v: v,
                plan_joint=lambda c, g: {"status": "Success", "position": [c.tolist(), g.tolist()]},
                deadline=time.monotonic() + 30.0,
                diagnostic={},
            )
        diag_none = {}
        kept = replan_from_ik_candidate(
            ik_q=[6.20, 0.0],
            current=current,
            limits=limits,
            joint_types=["revolute", "revolute"],
            transforms=eye_links,
            pose_error=lambda _q: [0.0, 0.0],
            resample=lambda v: v,
            plan_joint=lambda _c, _g: {"status": "Success", "position": None},
            deadline=time.monotonic() + 30.0,
            diagnostic=diag_none,
        )
        self.assertIsNone(kept)
        self.assertEqual(diag_none["reason"], "candidate_validation_failed")

        # DEF-08: summarize_batch preserves status='unknown' when execution_uncertain is True.
        plan = {"status": "executing", "completed": [], "cursor": 1, "revision": 1}
        stage_results = {"s1": {"status": "running", "execution_uncertain": True}}
        receipt = {}
        summarize_batch("p1", plan, ["s1"], stage_results, receipt)
        self.assertEqual(stage_results["s1"]["status"], "unknown")

        # DEF-01 (#5): GeometryTools.execute maps LinAlgError to status='unknown'/numerical_failure and propagates KeyError/TypeError.
        class StubSceneStore:
            def __init__(self, mode):
                self.mode = mode

            def observation(self, _ident, _obs_id):
                if self.mode == "linalg":
                    raise np.linalg.LinAlgError("Singular matrix")
                if self.mode == "key_bug":
                    raise KeyError("unexpected_internal_key")
                raise TypeError("unexpected_internal_type")

        ident_map = dict(run_id="r", episode_id="e", env_instance_id="i", env_generation=1)
        base_req = {**ident_map, "schema_version": "0.1", "kind": "camera_pose", "camera_id": "head", "observation_id": "obs-1"}
        linalg_res = GeometryTools(StubSceneStore("linalg")).execute(base_req)
        self.assertEqual(linalg_res["status"], "unknown")
        self.assertEqual(linalg_res["failure_kind"], "numerical_failure")
        self.assertNotIn("diagnostics", linalg_res)
        with self.assertRaises(InternalToolRuntimeError):
            GeometryTools(StubSceneStore("key_bug")).execute(base_req)
        with self.assertRaises(TypeError):
            GeometryTools(StubSceneStore("type_bug")).execute(base_req)

        # DEF-18: trajectory_evidence and batch_geometry scrub host paths on OSError and re-raise TypeError.
        tmp_assets = Path(self.enterContext(tempfile.TemporaryDirectory()))
        fake_adapter = SimpleNamespace(
            root=tmp_assets,
            control_hz=20.0,
            last_dispatched_goals={"left": 1.0, "right": 1.0},
            robot=SimpleNamespace(get_robot_by_arm_name=lambda _n: SimpleNamespace(arm_joints_name=[f"j{i}" for i in range(6)])),
        )
        compiled = {"left": (None, [], [[0.0] * 14], {})}
        before_st = {
            "arms": {
                arm: {
                    "joint_names": [f"j{i}" for i in range(6)],
                    "q_rad": [0.0] * 6,
                    "source_kind": "measurement",
                    "ee_pose": {"position_m": [0.2, 0.0, 0.8], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                                "frame_id": "env_origin", "tcp_config_id": "arx-x5-link6-v1", "source_kind": "measurement"},
                    "gripper": {"goal_normalized": 1.0, "joint_names": ["j7", "j8"], "position_m": [0.04, 0.04], "position_source_kind": "measurement"},
                }
                for arm in ("left", "right")
            }
        }
        with patch("ni_rua.robot.motion.trajectory_evidence._hardware", side_effect=FileNotFoundError(f"missing {tmp_assets / 'X5A.urdf'}")):
            _, full_ev = build_trajectory_evidence(fake_adapter, before_st, compiled, "independent", {"frame_id": "env_origin", "tcp_config_id": "arx-x5-link6-v1"})
            self.assertEqual(full_ev["geometry"]["reason"], "robot_geometry_asset_unavailable")
            self.assertNotIn(str(tmp_assets), json.dumps(full_ev))
            rev = review_batch(fake_adapter, before_st, {"frame_id": "env_origin", "tcp_config_id": "arx-x5-link6-v1", "samples": []}, {})
            self.assertNotIn(str(tmp_assets), json.dumps(rev))
        with patch("ni_rua.robot.motion.trajectory_evidence._hardware", side_effect=TypeError("internal_type_bug")):
            with self.assertRaises(TypeError):
                build_trajectory_evidence(fake_adapter, before_st, compiled, "independent", {"frame_id": "env_origin", "tcp_config_id": "arx-x5-link6-v1"})

        # DEF-19: _populate_planned_positions marks linear_fallback on domain ValueError and propagates TypeError.
        rec_map = {"m1": {"kind": "move", "arm": "left", "start": 0, "end": 2, "rows": [[0.0] * 14, [0.0] * 14], "target": {"position_m": [0.3, 0.0, 0.8]}}}
        _populate_planned_positions(SimpleNamespace(robot_model=None), before_st, rec_map, {})
        self.assertEqual(rec_map["m1"]["planned_positions_source"], "linear_fallback")
        with patch("ni_rua.robot.model.geometry.fk", side_effect=TypeError("fk_type_bug")):
            with self.assertRaises(TypeError):
                _populate_planned_positions(SimpleNamespace(robot_model={"joints": [{"name": "j1"}]}, robot=fake_adapter.robot), before_st, rec_map, {})

        # DEF-20: _measured_physical_jaw propagates TypeError from broken get_robot_by_arm_name.
        bad_adapter = SimpleNamespace(robot=SimpleNamespace(get_robot_by_arm_name=lambda _n: (_ for _ in ()).throw(TypeError("broken robot lookup"))))
        with self.assertRaises(TypeError):
            _measured_physical_jaw(bad_adapter, before_st, "left")

        # DEF-24 (#2): SceneTruthRecorder serializes before os.open so NaN never creates an empty file.
        truth_file = Path(self.enterContext(tempfile.TemporaryDirectory())) / "scene_truth.jsonl"
        recorder = SceneTruthRecorder(SimpleNamespace(), truth_file)
        recorder.objects = lambda _errs: [{"nan_val": float("nan")}]
        recorder.table = lambda _errs: {}
        recorder.robot = lambda _st: {}
        recorder.record("step", 1, 0.1, {})
        self.assertFalse(truth_file.exists())
        self.assertIsNotNone(recorder.last_error)


if __name__ == "__main__":
    unittest.main()
