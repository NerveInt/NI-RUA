"""Shared motion, harness, program-review, and trajectory fixtures across robot, runtime, and apps zones."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
from types import MethodType, ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ni_rua.core.config import Config
from ni_rua.core.contracts import canonical
from ni_rua.core.storage.images import image_settings
from ni_rua.core.storage.store import Store
from ni_rua.core.tool_output import ToolOutput
from ni_rua.robot.backends.robodojo.robodojo import FRAME_ID, TCP_CONFIG_ID
from ni_rua.robot.backends.robodojo.robot_worker import RoboDojoAdapter
from ni_rua.runtime.actions.unified_actions import compile_request
from ni_rua.runtime.harness import Harness
from ni_rua.runtime.programs.runner import ProgramRunner
from support import embodiment_golden as golden
from support.fixture_robot import FixtureRobot
from support.program_fixture import program_services
from support.tool_fixture import install_registry

IDENTITY = dict(run_id="run", episode_id="episode", env_instance_id="env", env_generation=1)
DUAL_ARM_IDENTITY = dict(run_id="dual-interface", episode_id="ep", env_instance_id="env", env_generation=1)

SPLIT = ("approach_tolerance_rad", "roll_tolerance_rad")
DOWN = [1.0, 0.0, 0.0, 0.0]

RETIRED_TOOLS = frozenset({
    "robot_move_ee", "robot_move_grasp", "robot_move_dual_grasp",
    "robot_execute_ee", "robot_gripper", "robot_execute_queue",
    "robot_preview_cost", "robot_preview_motion", "robot_execute_plan",
    "robot_inspect_trajectory", "grasp_plan", "grasp_memory",
    "manipulation_run", "skill_run",
    "operation_verify", "depth_move",
})


def wait_action(steps=7, **values):
    return dict(
        schema_version="0.1",
        **IDENTITY,
        type="EEAction",
        action_id="wait",
        based_on_observation_id="observed",
        mode="absolute",
        frame_id="env_origin",
        tcp_config_id="arx-x5-link6-v1",
        limits_profile="robodojo-native-v1",
        timeout_s=10,
        path=dict(kind="hold", waypoints=[]),
        gripper_events=[],
        hold_native_steps=steps,
    ) | values


def fake_adapter(*, step_limit=100, success_at=None, after_step=None):
    adapter = object.__new__(RoboDojoAdapter)
    adapter.np, adapter.control_hz = np, 25
    adapter.control_step = adapter.evaluator_actions = 0
    adapter.native_step_ticks = 10
    adapter.versions = {}
    adapter.observation_profile = "rgb_proprio_calibrated_v1"
    adapter.goals = {"left": 0.2, "right": 0.6}
    adapter.last_dispatched_goals = {"left": 1.0, "right": 0.3}
    adapter.arm_step_limits = {side: np.array([0.05] * 6) for side in ("left", "right")}
    adapter.arm_joint_limits, adapter.arm_joint_names = {}, {}
    adapter.gripper_step_limit, adapter.eef_gripper_dwell_step = 0.25, 0.18
    adapter.robot = SimpleNamespace(robot_list=[], robot_key=[])
    adapter.env = SimpleNamespace(
        end_flag=[False],
        success=[True],
        take_action_cnt=[0],
        step_lim=step_limit,
        dt=0.004,
    )
    measured = {"left": [0.1] * 6, "right": [-0.2] * 6}
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
        measured["left"][0] += 0.001
        if count == success_at or count == step_limit:
            adapter.env.end_flag[0] = True
            adapter.env.success[0] = count == success_at
        if after_step:
            after_step(count)

    adapter.env.take_action = take_action
    return adapter, dispatched


def periodic_pose(q):
    angle = float(q[0])
    return dict(
        position_m=[math.cos(angle), math.sin(angle), 1.0],
        quaternion_xyzw=[0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)],
    )


def motion_action(arm="left", angles=(-0.125,), *, gripper=False, timeout_s=30):
    action = wait_action(timeout_s=timeout_s)
    del action["hold_native_steps"]
    action["path"] = dict(
        kind="backend_free",
        waypoints=[{arm: periodic_pose([angle])} for angle in angles],
    )
    if gripper:
        action["gripper_events"] = [dict(arm=arm, phase="after_motion", goal_normalized=0.0)]
    return action


class Limits:
    def __getitem__(self, key):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.array([[-10.0, 10.0]] * 5 + [[-3.14, 3.14]])


class Planner:
    """A periodic geometric fixture; native collision solving is not simulated."""

    def __init__(self, *, wound=True, candidate="success", after_candidate=None, fail_pose_at=None):
        self.wound, self.candidate = wound, candidate
        self.after_candidate, self.fail_pose_at = after_candidate, fail_pose_at
        self.motion_planner = SimpleNamespace()
        self.pose_calls, self.joint_calls, self.originals = [], [], []

    def plan_path(self, current, target, origin):
        self.pose_calls.append((np.array(current, copy=True), deepcopy(target), origin))
        if len(self.pose_calls) == self.fail_pose_at:
            return {"status": "Fail"}
        goal = np.array(current, copy=True)
        goal[0] = math.atan2(target[1], target[0]) + (math.tau if self.wound else 0.0)
        original = np.stack([current, goal])
        self.originals.append(original.copy())
        return {"status": "Success", "position": original}

    def plan_joint(self, current, goal):
        self.joint_calls.append((current.copy(), goal.copy()))
        if self.after_candidate is not None:
            self.after_candidate()
        if self.candidate == "exception":
            raise RuntimeError("synthetic native planner failure")
        if self.candidate == "fail":
            return {"status": "Fail"}
        positions = np.stack([current, goal])
        if self.candidate == "bad_start":
            positions[0, 1] += 0.2
        return {"status": "Success", "position": positions}


def replan_adapter(arm="left", *, enabled=True, wound=True, candidate="success",
                   after_candidate=None, fail_pose_at=None, **native_options):
    measured = {"left": [0.0] * 6, "right": [0.0] * 6}

    def follow_dispatched(count):
        for side in measured:
            measured[side] = list(dispatched[-1][f"{side}_arm_joint_state"])

    adapter, dispatched = fake_adapter(after_step=follow_dispatched, **native_options)
    adapter.joint_replan_enabled = enabled
    adapter.state = lambda: {
        "arms": {side: dict(q_rad=list(q), ee_pose=periodic_pose(q)) for side, q in measured.items()}
    }
    robot = SimpleNamespace(
        robot_name="fixture",
        arm_name=arm + "_arm",
        type="target",
        arm_joints_name=[f"joint{i}" for i in range(1, 7)],
        arm_joint_indices=list(range(6)),
        entity_origin_pose=object(),
    )
    planner = Planner(wound=wound, candidate=candidate, after_candidate=after_candidate, fail_pose_at=fail_pose_at)
    adapter.robot = SimpleNamespace(
        planner={robot.robot_name: planner},
        robot_list=[robot],
        robot_key=[SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=Limits()))],
        get_robot_by_arm_name=lambda name: robot,
        get_joint=lambda selected, env_ids: [np.array(measured[arm], dtype=float)],
    )
    adapter.geometry_calls = []

    def geometry(selected_robot, selected_planner, target):
        adapter.geometry_calls.append((selected_robot, selected_planner, deepcopy(target)))

        def transforms(q):
            angle = float(q[0])
            c, s = math.cos(angle), math.sin(angle)
            tool = np.array([[c, -s, 0.0, c], [s, c, 0.0, s], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
            return {"base": np.eye(4), "tool": tool}

        def pose_error(q):
            actual = periodic_pose(q)
            distance = np.linalg.norm(np.array(actual["position_m"]) - target["position_m"])
            dot = abs(np.dot(actual["quaternion_xyzw"], target["quaternion_xyzw"]))
            return float(distance), 2 * math.acos(float(np.clip(dot, 0.0, 1.0)))

        return ["revolute"] * 6, transforms, pose_error

    adapter._joint_replan_geometry = geometry
    return adapter, robot, planner, dispatched


def preview_adapter(**kwargs):
    adapter, robot, planner, dispatched = replan_adapter(enabled=False, wound=False, **kwargs)
    robot.entity_origin_pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    robot.ee_link_name = "tool"
    adapter.root = Path("/nonexistent/preview-fixture")
    joints = []
    for index in range(1, 7):
        origin = np.eye(4)
        if index == 2:
            origin[:3, 3] = [1.0, 0.0, 1.0]
        joints.append(
            dict(
                name=f"joint{index}",
                type="continuous" if index == 1 else "fixed",
                parent="base" if index == 1 else f"link{index - 1}",
                child="tool" if index == 6 else f"link{index}",
                T_parent_joint=origin.tolist(),
                axis=[0.0, 0.0, 1.0],
            )
        )
    adapter.robot_model = {"base_link": "base", "joints": joints, "geometry_scope": "fixture"}
    native_state = adapter.state

    def state():
        value = native_state()
        for arm in value["arms"].values():
            arm.update(joint_names=list(robot.arm_joints_name), source_kind="measurement")
            arm["ee_pose"].update(source_kind="measurement", frame_id="env_origin", tcp_config_id="arx-x5-link6-v1")
        return value

    adapter.state = state
    planner.robot_cfg = {
        "robot_cfg": {
            "kinematics": {
                "collision_spheres": {"link1": [{"center": [0.0, 0.0, 0.0], "radius": 0.03}]},
                "self_collision_ignore": {"base": ["link1"]},
            }
        }
    }
    planner.scene_model = {"cuboid": {"table": {"dims": [3.0, 3.0, 0.05]}}}
    return adapter, planner, dispatched


def move(identifier, arm="left", angle=-0.125, **values):
    return dict(id=identifier, kind="move", arm=arm, action=motion_action(arm, (angle,)), **values)


def batch(stages, **values):
    own = motion_action()
    return {
        key: value for key, value in own.items() if key not in ("path", "gripper_events")
    } | dict(type="BatchAction", stages=stages, **values)


def jaw(identifier="close", arm="left", goal=0.0, **values):
    own = motion_action(arm)
    own.update(
        path=dict(kind="hold", waypoints=[]),
        gripper_events=[dict(arm=arm, phase="after_motion", goal_normalized=goal)],
    )
    return dict(id=identifier, kind="gripper", arm=arm, action=own, **values)


def jaws(adapter):
    robot = adapter.robot.robot_list[0]
    robot.gripper_joints_name = ["jaw"]
    robot.gripper_move = dict(base="jaw", sign=1)
    robot.gripper_scale = [0.0, 1.0]
    original = adapter.state

    def state():
        result = original()
        for arm in ("left", "right"):
            result["arms"][arm]["gripper"] = dict(
                position_source_kind="measurement",
                position_m=[adapter.last_dispatched_goals[arm]],
                joint_names=["jaw"],
            )
        return result

    adapter.state = state


def synchronized(action, *, goal=0.0, start=0.0, end=1.0):
    action = deepcopy(action)
    arm = next(iter(action["path"]["waypoints"][-1]))
    action["gripper_events"] = [
        dict(arm=arm, phase="during_motion", goal_normalized=goal, start_fraction=start, end_fraction=end)
    ]
    return action


def jaw_commands(rows, arm):
    return np.array([row[f"{arm}_ee_joint_state"][0] for row in rows])


@contextmanager
def native_fk_fixture(joint_names=("joint1", "joint2")):
    """Run the real native adapter on CPU without importing torch or cuRobo."""
    names = list(joint_names)
    calls = dict(scene_reads=0, fk=[], tensors=[])
    limits = SimpleNamespace(
        joint_names=names,
        position=np.array([[-1.0] * len(names), [1.0] * len(names)]),
        velocity=np.array([[-2.0] * len(names), [2.0] * len(names)]),
        acceleration=np.array([[-5.0] * len(names), [5.0] * len(names)]),
        jerk=np.array([[-10.0] * len(names), [10.0] * len(names)]),
    )
    cfg = SimpleNamespace(
        kinematics_config=SimpleNamespace(
            link_sphere_idx_map=np.array([0, 1]),
            link_name_to_idx_map={"link1": 0, "link2": 1},
        ),
        self_collision_config=SimpleNamespace(
            collision_pairs=np.array([[0, 1]]),
            sphere_padding=np.array([0.01, 0.02]),
        ),
    )

    def compute_kinematics(state):
        calls["fk"].append(state)
        samples = len(state.position)
        return SimpleNamespace(
            robot_spheres=np.tile([[[0.0, 0.0, 0.04, 0.1], [0.15, 0.0, 0.04, 0.1]]], (samples, 1, 1)),
            tool_poses=SimpleNamespace(
                tool_frames=["link6"],
                position=np.tile([[[1.0, 2.0, 3.0]]], (samples, 1, 1)),
                quaternion=np.tile([[[1.0, 0.0, 0.0, 0.0]]], (samples, 1, 1)),
            ),
        )

    motion = SimpleNamespace(
        joint_names=names,
        compute_kinematics=compute_kinematics,
        kinematics=SimpleNamespace(config=cfg, get_joint_limits=lambda: limits),
    )

    class _FkPlanner:
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
        yield _FkPlanner(), calls


def camera_fixture():
    step, domain = 30, "adapter_control_step"
    obs = dict(world_kind="observed", observation_id="observed", sim_step=step, step_domain=domain)
    frame = dict(
        camera_id="head",
        frame_id="head-frame",
        sim_step=step,
        step_domain=domain,
        rgb_ref="original-actor-rgb",
        pixel_transform="identity",
    )
    state = dict(sim_step=step, step_domain=domain, arms={})
    for side, x in (("left", 0.1), ("right", -0.4)):
        state["arms"][side] = dict(
            ee_pose=dict(
                position_m=[x, -0.05, 2.0],
                quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
                source_kind="measurement",
                frame_id="env_origin",
                tcp_config_id="arx-x5-link6-v1",
            )
        )
    camera = dict(
        observation=obs,
        frame=frame,
        state=state,
        width=640,
        height=480,
        K=np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]]),
        D=np.zeros(5),
        T=np.eye(4),
        reference_frame_id="env_origin",
        pose_source="fixed_calibration",
    )
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(640, dtype=np.uint16)[None, :] % 256
    return camera, image


def path_fixture():
    return dict(
        source_kind="candidate_command_trajectory_not_measurement",
        frame_id="env_origin",
        tcp_config_id="arx-x5-link6-v1",
        joint_names={a: ["joint1"] for a in ("left", "right")},
        native_dt_s=0.05,
        robot_model=dict(joints=[dict(parent="base", child="link6")]),
        samples=[
            dict(
                native_step=i,
                arms={
                    a: dict(
                        q_rad=[i * 0.01] * 6,
                        phase="motion",
                        link_positions_m={"base": [x, 0, 2], "link6": [x + i * 0.03, 0.1, 2]},
                    )
                    for a, x in [("left", -0.1), ("right", 0.1)]
                },
            )
            for i in range(4)
        ],
    )


def dual_arm_capabilities():
    return dict(
        frame_id=FRAME_ID,
        tcp_config_id=TCP_CONFIG_ID,
        path_kinds=["backend_free", "hold"],
        max_waypoints=8,
        dual_arm=True,
        gripper_after_motion=True,
        gripper_phases=["after_motion", "during_motion"],
        action_speed=dict(available=True, waypoint_speed_scales=dict(available=True)),
    )


def setup_dual_arm_harness(testcase):
    """Install a started dual-arm Harness + FixtureRobot + Store on ``testcase`` with LIFO cleanup."""
    temporary = tempfile.TemporaryDirectory()
    testcase.addCleanup(temporary.cleanup)
    testcase.store = Store(Path(temporary.name) / "store", dict(DUAL_ARM_IDENTITY))
    testcase.addCleanup(testcase.store.close)
    testcase.robot = FixtureRobot(temporary.name)
    observe = testcase.robot.observe
    testcase.robot.observe = lambda: dict(
        observe(),
        camera_calibration=dict(reference_frame_id="env_origin", cameras={}),
        camera_pose_sample=dict(cameras={}),
        robot_model=dict(base_link="base", joints=[]),
    )
    caps = {**testcase.robot.start(), **dual_arm_capabilities()}
    testcase.robot.start = lambda: caps
    testcase.h = Harness(
        testcase.store,
        testcase.robot,
        config=Config.from_dict(dict(features=dict(motion_review=False, default_motion_speed_scale=1.25))),
    )
    testcase.h.start()


class DualArmHarnessBase(unittest.TestCase):
    """Base TestCase providing dual-arm Harness setup without defining any test_* methods."""

    def setUp(self):
        setup_dual_arm_harness(self)


class _JointLimits:
    def __getitem__(self, index):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.array([[-10.0, 10.0]] * 6)


class _Planner:
    def __init__(self, side):
        self.side = side
        self.calls = []
        self.motion_planner = SimpleNamespace()

    def plan_path(self, start, target, origin):
        if origin != self.side:
            raise AssertionError("planner received the other arm base")
        self.calls.append(dict(start=start.tolist(), target=list(target)))
        goal = np.array(start, copy=True)
        goal[0] = target[0]
        return dict(status="Success", position=np.stack([start, goal]))


class _DualRuntime:
    def __init__(self):
        adapter = self.adapter = object.__new__(RoboDojoAdapter)
        adapter.np = np
        adapter.control_hz, adapter.native_step_ticks = 25, 10
        adapter.control_step = adapter.evaluator_actions = 0
        adapter.versions = {}
        adapter.root = Path("/nonexistent/unified-dual-cpu-fixture")
        adapter.observation_profile = "rgb_proprio_calibrated_v1"
        adapter.motion_execution_mode = "resampled"
        adapter.tracking_guard_enabled = adapter.motion_safety_guard_enabled = True
        adapter.last_dispatched_goals = dict(left=1.0, right=1.0)
        adapter.goals = dict(adapter.last_dispatched_goals)
        adapter.arm_step_limits = {side: np.full(6, 0.05) for side in ("left", "right")}
        adapter.arm_joint_limits = {}
        adapter.eef_gripper_dwell_step, adapter.gripper_step_limit = 0.18, 0.25
        self.initial = dict(left=[0.0, 0.2, 0.0, 0.0, 0.0, 0.0], right=[0.0, -0.4, 0.0, 0.0, 0.0, 0.0])
        self.measured = deepcopy(self.initial)
        self.jaws = dict(adapter.last_dispatched_goals)
        self.robots = {
            side: SimpleNamespace(
                arm_name=side + "_arm",
                robot_name=side,
                type="target",
                arm_joints_name=["joint" + str(i) for i in range(6)],
                arm_joint_indices=list(range(6)),
                gripper_joints_name=["jaw"],
                gripper_move=dict(base="jaw", sign=1),
                gripper_scale=[0.0, 1.0],
                entity_origin_pose=side,
            )
            for side in self.initial
        }
        self.planners = {side: _Planner(side) for side in self.initial}
        adapter.robot = SimpleNamespace(
            robot_list=list(self.robots.values()),
            planner=self.planners,
            robot_key=[SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=_JointLimits())) for _ in self.initial],
            get_robot_by_arm_name=lambda name: self.robots[name.split("_")[0]],
            get_joint=lambda robot, env_ids: [np.array(self.measured[robot.robot_name])],
        )
        adapter.arm_joint_names = {side: list(robot.arm_joints_name) for side, robot in self.robots.items()}
        adapter.state = self.state
        adapter.observe = lambda: dict(robot_state=self.state(), sim_step=adapter.control_step)
        adapter.env = SimpleNamespace(
            end_flag=[False],
            success=[False],
            take_action_cnt=[0],
            step_lim=1000,
            dt=0.004,
            take_action=self.take_action,
        )
        self.commands = []
        self.harness = SimpleNamespace(
            identity=dict(run_id="dual-runtime", episode_id="episode", env_instance_id="env", env_generation=1),
            capabilities=adapter.capabilities(),
            config=SimpleNamespace(features=SimpleNamespace(default_motion_speed_scale=1.0)),
        )
        self.harness._action_from_args = MethodType(Harness._action_from_args, self.harness)

    def state(self):
        return dict(
            arms={
                side: dict(
                    q_rad=list(q),
                    joint_names=list(self.robots[side].arm_joints_name),
                    ee_pose=dict(position_m=[q[0], 0.0, 1.0], quaternion_xyzw=[0.0, 0.0, 0.0, 1.0]),
                    gripper=dict(position_source_kind="measurement", position_m=[self.jaws[side]]),
                )
                for side, q in self.measured.items()
            }
        )

    def take_action(self, command):
        self.commands.append(deepcopy(command))
        for side in self.measured:
            self.measured[side] = list(command[side + "_arm_joint_state"])
            self.jaws[side] = float(command[side + "_ee_joint_state"][0])
        self.adapter.control_step += self.adapter.native_step_ticks
        self.adapter.env.take_action_cnt[0] += 1

    def preview(self, stages):
        request = dict(action_id="dual", observation_id="obs", stages=stages)
        original = deepcopy(request)
        action = compile_request(self.harness, request)
        if request != original:
            raise AssertionError("compilation mutated the public request")
        plan = self.adapter.preview_batch(action)
        if plan["status"] != "planned":
            raise AssertionError(plan)
        return action, plan

    def execute(self, action, plan):
        return self.adapter.execute(dict(action, batch_plan_id=plan["plan_id"]), lambda _: None, threading.Event())

    def simultaneous_rows(self):
        previous = deepcopy(self.initial)
        count = 0
        for command in self.commands:
            current = {side: command[side + "_arm_joint_state"] for side in previous}
            count += all(not np.array_equal(current[side], previous[side]) for side in previous)
            previous = current
        return count


def _so101_profile(ik_service=False):
    from ni_rua.robot.backends.so101.deployment import embodiment_profile, validate_deployment

    document = json.loads((golden.REPOSITORY / "configs" / "so101.example.json").read_text())
    if ik_service:
        document["ik_service"] = {"solver": "curobo", "url": "http://127.0.0.1:9"}
    return embodiment_profile(validate_deployment(document, base=golden.REPOSITORY / "configs"))


def _so101_harness(root, *, ik_service=False):
    root = Path(root)
    deployment = golden.so101_deployment(root / "deployment", empty_close_aperture_m=golden.SO101_EMPTY_CLOSE_M)
    if ik_service:
        document = json.loads(deployment.read_text())
        document["ik_service"] = {"solver": "curobo", "url": "http://127.0.0.1:9"}
        deployment.write_text(json.dumps(document, indent=2))
    config = golden.load_config(golden.SO101_CONFIGS["so101_gateway"], overrides={"backend": {"deployment": str(deployment)}})
    robot = golden._so101_robot_class()(config=config, run_dir=root / "robot")
    features = config.features
    harness = Harness(
        Store(root / "store", dict(golden.IDENTITY)),
        robot,
        actor_payload_profile=features.actor_payload_profile,
        enable_wait=features.enable_wait,
        observation_profile=features.observation_profile,
        motion_review=features.motion_review,
        vision_compare=features.vision_compare,
        config=config,
        task_plan_required=True,
    )
    harness.start()
    return harness


def _close(harness):
    harness.store.close()
    close = getattr(harness.robot, "close", None)
    if callable(close):
        close()


def observation(identifier="obs-1", generation=1):
    from ni_rua.core.presentation.observation_output import observation_metadata

    matrix = [[1.0, 0.0, 0.0, 0.123456789012345], [0.0, 1.0, 0.0, 0.2], [0.0, 0.0, 1.0, 0.3], [0.0, 0.0, 0.0, 1.0]]
    calibration = dict(
        K=[[397.041259765625, 0.0, 320.0], [0.0, 397.041259765625, 240.0], [0.0, 0.0, 1.0]],
        distortion=[0.0, 0.0, 0.0, 0.0, 0.0],
        distortion_model="pinhole_no_distortion",
        image_size=[640, 480],
        optical_axes="+x right,+y down,+z forward",
        calibration_source="public sensor calibration",
    )
    arm = dict(
        joint_names=["joint" + str(i) for i in range(1, 7)],
        source_kind="measurement",
        q_rad=[0.123456789012345] * 6,
        qdot_rad_s=[0.00000123456789] * 6,
        ee_pose=dict(
            position_m=[0.1, 0.2, 0.3],
            quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
            frame_id="env_origin",
            tcp_config_id="arx-x5-link6-v1",
            source_kind="measurement",
        ),
        gripper=dict(
            joint_names=["joint7", "joint8"],
            position_m=[0.012345678901, 0.02],
            position_source_kind="measurement",
            goal_normalized=0.5,
            goal_source_kind="command",
        ),
        grasp_pose=dict(
            position_m=[0.1, 0.2, 0.4],
            angles_deg=[0.0, 0.0, 90.0],
            frame_id="env_origin",
            source_kind="derived_from_measured_link6",
            convention="robot_move_grasp",
            contact_region="pad_center",
            contact_depth_m=0.0,
            contact_point_meaning="center_between_fingers_at_distal_15mm_pad_center",
        ),
    )
    value = dict(
        bundle=dict(
            observation_id=identifier,
            run_id="run",
            episode_id="episode",
            env_instance_id="env",
            env_generation=generation,
            sim_step=10,
            action_budget=dict(native_steps_remaining=700),
        ),
        robot_state=dict(
            arms=dict(left=deepcopy(arm), right=deepcopy(arm)),
            sim_step=10,
            frame_id="env_origin",
            tcp_config_id="arx-x5-link6-v1",
            units=dict(position="m", joints="rad"),
        ),
        camera_calibration=dict(
            profile_id="public-cal-v1",
            reference_frame_id="env_origin",
            source_kind="public_calibration",
            full_ref="artifact:calibration",
            cameras={"head": deepcopy(calibration), "left": deepcopy(calibration), "right": deepcopy(calibration)},
        ),
        camera_pose_sample=dict(
            sim_step=10,
            reference_frame_id="env_origin",
            full_ref="artifact:pose-" + identifier,
            cameras={
                name: dict(T_world_camera=deepcopy(matrix), frame_id=identifier + "-" + name, source_kind="measurement")
                for name in ("head", "left", "right")
            },
        ),
    )
    return observation_metadata(
        value,
        current_observation_id=identifier,
        current_action_budget=dict(native_steps_remaining=700),
        compact_geometry=True,
    )


def clocked_observation(identifier="obs-1"):
    value = observation(identifier)
    bundle = value["bundle"]
    bundle.update(
        schema_version="0.1",
        robot_model_id="arx-x5",
        observation_profile="rgb_proprio_calibrated_v1",
        clock_domain="env/generation-1",
        step_domain="adapter_control_step",
        calibration_id="a" * 64,
        robot_model_ref="artifact:model",
        data_origin="robodojo_runtime",
        task_instruction="Keep the task visible.",
    )
    bundle["cameras"] = [
        dict(
            camera_id=name,
            width=640,
            height=480,
            pixel_transform="identity",
            control_dt_s=0.004,
            step_domain="adapter_control_step",
            frame_id=identifier + "-" + name,
            capture_time=identifier + "-time",
            sim_step=10,
            clock_domain=None,
            rgb_available=True,
            calibration_available=True,
            pose_sample_available=True,
            synchronization="physics paused; camera fence",
        )
        for name in ("head", "left", "right")
    ]
    for budget in (bundle["action_budget"], value["current_action_budget"]):
        budget.update(
            semantics="official native steps; separate from tool requests and model-call budget; reserves are not included",
            source="native_execution_accounting",
            unknown_values_are_null=True,
            native_steps_limit=1000,
            native_steps_used=300,
            per_request_native_limit=250,
        )
    value["robot_state"]["step_domain"] = "adapter_control_step"
    value["camera_pose_sample"]["step_domain"] = "adapter_control_step"
    return value


def hardware():
    joints = []
    for i in range(1, 7):
        t = np.eye(4)
        t[0, 3] = 0.1
        joints.append(
            dict(
                name=f"joint{i}",
                type="revolute",
                parent="base_link" if i == 1 else f"link{i - 1}",
                child=f"link{i}",
                axis=[0, 0, 1],
                T_parent_joint=t.tolist(),
            )
        )
    for i in (7, 8):
        joints.append(
            dict(
                name=f"joint{i}",
                type="prismatic",
                parent="link6",
                child=f"link{i}",
                axis=[0, 1 if i == 7 else -1, 0],
                T_parent_joint=np.eye(4).tolist(),
            )
        )
    return dict(
        model=dict(base_link="base_link", joints=joints, source_sha256="public-fixture"),
        spheres={f"link{i}": [dict(center=[0, 0, 0], radius=0.03)] for i in range(1, 7)},
        sphere_buffer_m=0,
        jaws=dict(
            gripper_scale=[-0.01, 0.044],
            gripper_move=dict(base="joint7", sign=1, mimic=["joint8", 1.0, 0.0]),
            gripper_joints_name=["joint7", "joint8"],
        ),
        provenance={"kind": "public-hardware-fixture"},
    )


def public_fixture(separation=3.0):
    h = hardware()
    state = {"arms": {}}
    for arm, base in [("left", 0), ("right", separation)]:
        state["arms"][arm] = dict(
            source_kind="measurement",
            joint_names=[f"joint{i}" for i in range(1, 7)],
            q_rad=[0.0] * 6,
            ee_pose=dict(
                frame_id="env_origin",
                tcp_config_id="arx-x5-link6-v1",
                source_kind="measurement",
                position_m=[base + 0.6, 0, 0],
                quaternion_xyzw=[0, 0, 0, 1],
            ),
            gripper=dict(joint_names=["joint7", "joint8"], position_source_kind="measurement", position_m=[0.044, 0.044]),
        )
    adapter = SimpleNamespace(control_hz=25, last_dispatched_goals={"left": 1.0, "right": 1.0})
    row = [0.0] * 6 + [1.0] + [0.0] * 6 + [1.0]
    compiled = {a: (np.zeros((2, 6)), [list(row), list(row)], [], None) for a in ("left", "right")}
    action = dict(frame_id="env_origin", tcp_config_id="arx-x5-link6-v1")
    return adapter, state, compiled, action, h


class ActionTrajectoryFeedbackFixture:
    """Reusable trajectory-feedback Store/Harness builder without TestCase test_* side effects."""

    def __init__(self, embodiment=None):
        self._stack = ExitStack()
        tmp = self._stack.enter_context(tempfile.TemporaryDirectory())
        self.identity = dict(run_id="run", episode_id="episode", env_instance_id="environment", env_generation=1)
        self.store = Store(Path(tmp), self.identity)
        self._stack.callback(self.store.close)
        _, self.before, _, self.action, hw = public_fixture(0.5)
        self.action.update(self.identity, action_id="action")
        self.before.update(sim_step=30, step_domain="adapter_control_step")
        self.harness = SimpleNamespace(
            store=self.store,
            identity=self.identity,
            latest={"observation_id": "terminal"},
            observation_profile="rgb_proprio_calibrated_v1",
            image_settings=image_settings(),
            config=SimpleNamespace(features=SimpleNamespace(action_trajectory_delivery="diagnostic")),
        )
        if embodiment is not None:
            self.harness.embodiment = embodiment
        self.model_ref = self.put(hw["model"], "robot_model")
        camera = np.eye(4)
        camera[2, 3] = -2
        calibration = dict(
            reference_frame_id="env_origin",
            cameras={
                "cam_head": dict(
                    image_size=[640, 480],
                    K=[[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]],
                    distortion=[0.0] * 5,
                    T_world_camera=camera.tolist(),
                )
            },
        )
        calibration_ref = self.put(calibration, "calibration")
        buffer = io.BytesIO()
        Image.new("RGB", (640, 480), (80, 120, 150)).save(buffer, format="PNG")
        self.rgb_ref = self.store.put_artifact(buffer.getvalue(), "rgb", "image/png")
        self.observation = dict(
            self.identity,
            observation_id="start",
            world_kind="observed",
            sim_step=30,
            step_domain="adapter_control_step",
            robot_model_ref=self.model_ref,
            robot_state_ref=self.put(self.before, "robot_state"),
            cameras=[
                dict(
                    camera_id="cam_head",
                    frame_id="head",
                    width=640,
                    height=480,
                    sim_step=30,
                    step_domain="adapter_control_step",
                    calibration_ref=calibration_ref,
                    rgb_ref=self.rgb_ref,
                    pixel_transform="identity",
                    pose_sample_ref="FORBIDDEN",
                )
            ],
        )
        self.store.save("observation", "start", self.observation)

    def put(self, value, kind, scope="actor"):
        return self.store.put_artifact(canonical(value).encode(), kind, "application/json", scope)

    def receipt(self, plan, counts=2, measured_error=0.0):
        actual = []
        for index in range(counts + 1):
            state = deepcopy(self.before)
            state["sim_step"] = 30 + index * 10
            for arm in ("left", "right"):
                state["arms"][arm]["q_rad"] = [index * 0.1 + (measured_error if index else 0.0)] * 6
            actual.append(dict(sim_step=state["sim_step"], state=state))
        actual.append(deepcopy(actual[-1]))
        return dict(
            status="completed",
            executed_steps=counts,
            completed_native_steps=counts,
            execution_start_observation_id="start",
            result_observation_id="terminal",
            trace_sampling=dict(actual_every_native_actions=1, actual_boundary_samples=True),
            planned_trace_ref=self.put(plan, "planned_trace"),
            actual_trace_ref=self.put(actual, "actual_trace"),
            before_state_ref=self.put(self.before, "before_state"),
            dispatched_trace_ref=self.put({"samples": []}, "dispatched_trace"),
        )

    @staticmethod
    def rows(count):
        return [[i * 0.1] * 6 + [1.0] + [i * 0.1] * 6 + [1.0] for i in range(1, count + 1)]

    def close(self):
        self._stack.close()


def measured_state(offset=0):
    return {
        "frame_id": "env_origin",
        "tcp_config_id": "arx-x5-link6-v1",
        "sim_step": 0,
        "step_domain": "adapter_control_step",
        "arms": {
            side: {
                "ee_pose": {
                    "position_m": [offset, 0, 1],
                    "quaternion_xyzw": [0, 0, 0, 1],
                    "frame_id": "env_origin",
                    "tcp_config_id": "arx-x5-link6-v1",
                    "source_kind": "measurement",
                }
            }
            for side in ("left", "right")
        },
    }


def saved_observation(store, identifier, offset=0):
    return {
        **store.identity,
        "schema_version": "0.1",
        "world_kind": "observed",
        "observation_id": identifier,
        "sim_step": 0,
        "step_domain": "adapter_control_step",
        "robot_state_ref": store.put_artifact(
            json.dumps(measured_state(offset)).encode(), "robot_state", "application/json"
        ),
        "task_instruction": "Arrange the visible digits.",
    }


class ProgramReviewFixture:
    """Isolated ProgramRunner + Store review fixture shared by runtime and apps tests."""

    def __init__(self):
        self._stack = ExitStack()
        tmp = self._stack.enter_context(tempfile.TemporaryDirectory())
        self.store = Store(
            Path(tmp) / "run",
            dict(run_id="review", episode_id="ep", env_instance_id="env", env_generation=1),
        )
        self._stack.callback(self.store.close)
        self.calls = []
        self.plan_count = 0
        self.action_count = 0
        self.stale = False
        self.plan_cursor = 0
        self.plan_revision = 0
        self.harness = SimpleNamespace(
            store=self.store,
            lock=threading.RLock(),
            cancel=threading.Event(),
            active_context=None,
            dispatch=self.dispatch,
        )
        self.runner = ProgramRunner(program_services(self.harness))
        self._stack.callback(self.runner.close)
        self.identity = dict(thread_id="thread", turn_id="start", tool_call_id="start-call")

    @staticmethod
    def response(body):
        kind = "action" if "receipt" in body else "plan" if "plan_id" in body else "data"
        return ToolOutput(body, kind=kind).response()

    def dispatch(self, name, args, correlation):
        self.calls.append((name, dict(args), dict(correlation)))
        if name == "robot_wait":
            self.action_count += 1
            return self.response(
                dict(
                    receipt=dict(
                        action_id="wait-action",
                        status="completed",
                        result_observation_id="obs-after-wait",
                        executed_native_steps=args["native_steps"],
                    )
                )
            )
        if name != "robot_action":
            return self.response(dict(test=True))
        mode = args.get("mode", "preview")
        if mode == "preview":
            self.plan_count += 1
            result = self.response(
                dict(
                    status="planned",
                    plan_id="plan-" + str(self.plan_count),
                    observation_id=args["observation_id"],
                    trajectory_evidence={"stages": args["stages"]},
                )
            )
            result.content.append(dict(type="inputImage", imageUrl="data:image/png;base64,AA=="))
            return result
        if mode == "inspect":
            return self.response(
                dict(
                    status="planned",
                    plan_id=args["plan_id"],
                    executed_batch_native_steps=self.plan_cursor,
                    plan_revision=self.plan_revision,
                    trajectory_evidence={"inspected": True},
                )
            )
        self.action_count += 1
        status = "rejected" if self.stale else "completed"
        return self.response(
            dict(
                plan_id=args.get("plan_id"),
                receipt=dict(
                    action_id="action-" + str(self.action_count),
                    status=status,
                    reason="stale_plan" if self.stale else None,
                    result_observation_id=None if self.stale else "obs-final-" + str(self.action_count),
                    executed_native_steps=0 if self.stale else 2,
                    stop_confirmed=True,
                ),
            )
        )

    def start(self, source, **args):
        return self.runner.run(dict(source=source, timeout_s=10, **args), self.identity)

    def resume(self, job, decision="approve", **extra):
        return self.runner.run(
            dict(
                operation="resume",
                job_id=job["job_id"],
                review_id=job["pending_review"]["review_id"],
                decision=decision,
                **extra,
            ),
            dict(self.identity, turn_id="review", tool_call_id="resume-call"),
        )

    def close(self):
        self._stack.close()


def queue_case_fixture(root):
    """Minimal validated checkout + queue case dictionary for CLI/flag checks."""
    cwd = Path(root) / "checkout"
    (cwd / "ni_rua/apps/gateway").mkdir(parents=True, exist_ok=True)
    (cwd / "ni_rua/apps/gateway/cli.py").write_text("parser.add_argument('--config')\nConfig.load()\n")
    return dict(
        cwd=str(cwd),
        run_dir=str(Path(root) / "run-1"),
        task="general_pickup",
        seed=0,
        layout_id=2,
        title="paired baseline",
        tag=["paired", "baseline"],
    )
