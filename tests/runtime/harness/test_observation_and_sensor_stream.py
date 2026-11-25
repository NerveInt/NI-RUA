"""
- test_observation_policy.py: Official actor inputs and development evidence must never share a publication path.
- test_sensor_stream_p0.py: 阶段一（P0）传感器契约、动作期 5Hz 腕部流门控缓存与 SAM 3.1 单次特征提取单元测试。

覆盖范围：
1. 通用固定外部视角（如 `cam_third_person`、`cam_overhead`）与随动腕部视角拓扑判定，及无 `cam_head` 时的 `scene3d` 查询支持；
2. `DepthFrameSpec`、`StereoPairSpec`、`StreamBudgetConfig`（24GB 默认与 40GB 扩展）物理单位与边界严格报错校验；
3. `Harness.execute` 运动中途 5Hz `progress` 观测 `make_current=False` 锚点不变性与终态缺失时的安全清零；
4. `WristProgressCache` 准静态关节角速度门控（`<= 0.8 rad/s`）、原地悬停去重、`scene3d` 多视角自动注入与跨局重置隔离；
5. `video_session_manager` 单帧多候选 `RGB2HSV` 与 `backbone_feat` GPU->CPU 单次提取数学等价性及 `MAX_GPU_HOT_SESSIONS` 环境变量配置。
"""
from __future__ import annotations

from copy import deepcopy
import asyncio
import sqlite3
from unittest.mock import patch
import json
from pathlib import Path
import tempfile
import unittest
from ni_rua.apps.gateway.workbench import Workbench
from ni_rua.runtime.harness import Harness
from ni_rua.core.config import Config
from ni_rua.core.observation_policy import (ACTOR_PROFILE, LEGACY_ACTOR_PROFILE, ACTOR_POLICY,
    DEVELOPMENT_TOOLS, actor_robot_state, actor_policy, require_actor_observation, reject_privileged)
from ni_rua.core.storage.store import Store
from ni_rua.runtime.tools.definitions import TOOLS
from support.fixture_robot import FixtureRobot
import os
from unittest import mock
import cv2
import numpy as np
from ni_rua.perception.external.sensor_frame_contract import (
    CAMERA_ROLE_FIXED_EXTERNAL,
    CAMERA_ROLE_WRIST_MOUNTED,
    DepthFrameSpec,
    StereoPairSpec,
    StreamBudgetConfig,
    classify_camera_role,
    list_fixed_camera_ids,
    list_wrist_camera_ids,
    resolve_camera_arm,
    select_primary_fixed_camera_id,
)
from ni_rua.perception.external.wrist_progress_cache import (
    WristProgressCache,
    measure_arm_joint_speed_rad_s,
)
from ni_rua.robot.backends.robodojo.robodojo import FRAME_ID, TCP_CONFIG_ID
from services.vggt_height.sam31 import video_session_manager as vsm
from support.perception_fixtures import FakePoints
from support.tool_fixture import install_registry
from support.fixture_robot import CalibratedThreeCamRobot

NI_RUA_TEST_TIER = "component"


# --- Folded from tests/test_observation_policy.py ---

class ObservationPolicyTest(unittest.TestCase):
    def test_fail_closed_publication_history_and_derived_tools(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        store = Store(Path(directory) / "run", dict(run_id="policy", episode_id="ep", env_instance_id="env", env_generation=1))
        self.addCleanup(store.close)
        robot = FixtureRobot(directory)
        harness = Harness(store, robot, observation_profile=LEGACY_ACTOR_PROFILE, motion_review=False)
        initial = harness.start()
        before = len(store.records("observation"))
        for changes in ({"observation_profile": "rgb_proprio_calibrated_v1"}, {"camera_calibration": {}},
                        {"camera_pose_sample": {}}, {"robot_model": {}}, {"extra": {"native_usd_pose_diagnostic": {}}},
                        {"extra": [{"planning_diagnostics": [{"status": "native_failure"}]}]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                harness.publish_observation({**robot.observe(), **changes})
        raw = robot.observe()
        raw["robot_state"]["arms"]["left"]["camera_poses"] = {}
        with self.assertRaises(ValueError):
            harness.publish_observation(raw)
        self.assertEqual(len(store.records("observation")), before)
        self.assertTrue(DEVELOPMENT_TOOLS.issubset(item["name"] for item in TOOLS))
        for name in DEVELOPMENT_TOOLS:
            with self.subTest(tool=name), self.assertRaisesRegex(ValueError, "tool_unavailable"):
                harness.dispatch(name, {}, {})
        # A historical calibrated record must not be re-published, even if it exists on disk.
        historical = {**initial, "observation_id": "old-calibrated", "observation_profile": "rgb_proprio_calibrated_v1"}
        store.save("observation", "old-calibrated", historical)
        with self.assertRaisesRegex(ValueError, "actor_requires"):
            harness.dispatch("history_view_observation", {"observation_id": "old-calibrated"}, {})
        # Developer evidence can still be saved and explicitly read by the observer.
        ref = store.put_artifact(b'{"K":"development-only"}', "calibration", "application/json", "observer")
        self.assertIn(b"development-only", store.resolve_artifact(ref, allowed_scopes=("observer",))[1])
        with self.assertRaisesRegex(ValueError, "access_scope_denied"):
            store.resolve_artifact(ref)

    def test_native_proprioception_projection_and_clean_session_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            robot = FixtureRobot(directory)
            state = robot.observe()["robot_state"]
            arm = state["arms"]["left"]
            arm.update(joint_names=["joint1"], qdot_rad_s=[17.0], gripper=dict(
                joint_names=["finger"], position_m=[0.123456], position_source_kind="measurement",
                command_normalized=.25, command_source_kind="command", goal_normalized=.25,
                goal_source_kind="command", contact_feedback=None, contact_feedback_reason="unavailable"))
            original = deepcopy(state)
            projected = actor_robot_state(state, LEGACY_ACTOR_PROFILE)["arms"]["left"]
            self.assertEqual(projected["q_rad"], arm["q_rad"])
            self.assertEqual(projected["ee_pose"], arm["ee_pose"])
            self.assertNotIn("qdot_rad_s", projected)
            self.assertNotIn("joint_names", projected)
            self.assertNotIn("position_m", projected["gripper"])
            self.assertEqual(projected["gripper"]["command_normalized"], .25)
            self.assertEqual(state, original)
            arm["unexpected_scene_measurement"] = 1
            with self.assertRaisesRegex(ValueError, "non_actor"):
                actor_robot_state(state)
            path = Path(directory) / "forbidden-run"
            with self.assertRaisesRegex(ValueError, "actor_requires"):
                Workbench(path, observation_profile="unknown_profile")
            self.assertFalse(path.exists())
            path.mkdir()
            (path / "manifest.json").write_text(json.dumps({"observation_profile": ACTOR_PROFILE}))
            with self.assertRaisesRegex(ValueError, "clean_actor_history"):
                Workbench(path, resume=True)

    def test_calibration_is_permitted_without_object_or_contact_truth(self):
        public = dict(observation_profile=ACTOR_PROFILE,
                      camera_calibration={"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
                      camera_pose_sample={"cameras": {"wrist": {"T_world_camera": []}}},
                      robot_model={"joints": []})
        require_actor_observation(public, ACTOR_PROFILE)
        self.assertEqual(actor_policy(), ACTOR_POLICY)
        with self.assertRaisesRegex(ValueError, "matching_observation_profile"):
            require_actor_observation(public, LEGACY_ACTOR_PROFILE)
        for field in ("object_poses", "depth", "contact_forces", "evaluator_state", "native_usd_pose_diagnostic"):
            for profile in (ACTOR_PROFILE, LEGACY_ACTOR_PROFILE):
                with self.subTest(field=field, profile=profile), self.assertRaises(ValueError):
                    reject_privileged({"nested": [{field: {}}]}, profile)
        with self.assertRaises(ValueError):
            require_actor_observation({**public, "observation_profile": LEGACY_ACTOR_PROFILE})

    def test_calibrated_proprioception_keeps_fk_inputs_but_not_contacts(self):
        with tempfile.TemporaryDirectory() as directory:
            state = FixtureRobot(directory).observe()["robot_state"]
            state["arms"]["left"].update(joint_names=["joint1"], qdot_rad_s=[0.2],
                gripper=dict(joint_names=["finger"], position_m=[0.02],
                             position_source_kind="measurement", contact_feedback=None))
            projected = actor_robot_state(state, ACTOR_PROFILE)["arms"]["left"]
            self.assertEqual(projected["joint_names"], ["joint1"])
            self.assertEqual(projected["gripper"]["position_m"], [0.02])
            self.assertNotIn("contact_feedback", projected["gripper"])
            state["arms"]["left"]["gripper"]["contact_feedback"] = {"object": "hidden"}
            with self.assertRaisesRegex(ValueError, "contact_truth"):
                actor_robot_state(state, ACTOR_PROFILE)

    def test_gateway_filters_controls_and_owns_shared_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = Path(directory).resolve() / "memory"  # configured paths are canonical
            with patch("ni_rua.core.config.LOCAL_CONFIG_PATH", Path(directory) / "absent.local.json"):
                config = Config.load(environ={}, overrides={"features": {"grasp_memory_dir": str(memory)}})
                workbench = Workbench(Path(directory) / "calibrated", robot=FixtureRobot(directory), config=config)
                try:
                    self.assertEqual(workbench.observation_profile, ACTOR_PROFILE)
                    names = {tool["name"] for tool in workbench.actor_tools()}
                    self.assertIn("geometry_query", names)
                    self.assertIn("robot_action", names)
                    self.assertIn("operation_memory", names)
                    self.assertFalse({"robot_preview_motion", "grasp_memory", "skill_run", "manipulation_run"} & names)
                    self.assertFalse(any(name.startswith("scene_") for name in names))
                    self.assertNotIn("operation_verify", names)
                    self.assertNotIn("vision_review", names)
                    self.assertEqual(workbench.grasp_memory_store.root, memory)
                    shared = workbench.grasp_memory_store
                finally:
                    asyncio.run(workbench.close())
                with self.assertRaises(sqlite3.ProgrammingError):
                    shared.db.execute("SELECT 1")
                for profile in (ACTOR_PROFILE, LEGACY_ACTOR_PROFILE):
                    control = Workbench(Path(directory) / profile, robot=FixtureRobot(directory), config=config,
                                        observation_profile=profile, motion_review=False)
                    try:
                        names = {tool["name"] for tool in control.actor_tools()}
                        self.assertNotIn("robot_preview_motion", names)
                        self.assertEqual("robot_action" in names, profile == ACTOR_PROFILE)
                        self.assertIn("operation_memory", names)
                        self.assertFalse({"skill_run", "manipulation_run", "grasp_memory"} & names)
                        self.assertFalse(any(name.startswith("scene_") for name in names))
                        self.assertEqual("geometry_query" in names, profile == ACTOR_PROFILE)
                        self.assertEqual(control.grasp_memory_store.root, memory)
                        self.assertIs(control.harness.operation_memory.shared_store, control.grasp_memory_store)
                        shared = control.grasp_memory_store
                    finally:
                        asyncio.run(control.close())
                    with self.assertRaises(sqlite3.ProgrammingError):
                        shared.db.execute("SELECT 1")

    def test_calibrated_harness_publishes_exact_camera_and_model_closure(self):
        from ni_rua.robot.model.geometry import GeometryTools
        from ni_rua.perception.scene.store import SceneStore
        directory = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(directory)
        identity = dict(run_id="calibrated", episode_id="ep", env_instance_id="env", env_generation=1)
        store, scene_artifacts = Store(root/"run", identity), Store(root/"scene-artifacts", identity)
        self.addCleanup(store.close)
        self.addCleanup(scene_artifacts.close)
        robot = FixtureRobot(directory)
        harness = Harness(store, robot, observation_profile=ACTOR_PROFILE, motion_review=False)
        harness.scene_artifacts = scene_artifacts
        harness.scene = SceneStore(root/"separate-scene", scene_artifacts.resolve_artifact, scene_artifacts.put_artifact)
        raw = robot.observe()
        raw["cameras"][0]["frame_id"] = "frame-calibrated"
        raw["robot_state"].update(sim_step=0, step_domain=raw["step_domain"])
        matrix = [[1.,0.,0.,0.], [0.,1.,0.,0.], [0.,0.,1.,0.], [0.,0.,0.,1.]]
        camera_pose = deepcopy(matrix)
        camera_pose[0][3] = 0.2
        intrinsic = [[100.,0.,0.5], [0.,100.,0.5], [0.,0.,1.]]
        raw.update(observation_profile=ACTOR_PROFILE,
            camera_calibration=dict(reference_frame_id="env_origin", cameras={"fixture_camera": dict(
                K=intrinsic, distortion={"model":"opencv", "coefficients":[]}, image_size=[1,1],
                T_link_camera=matrix)}),
            camera_pose_sample=dict(reference_frame_id="env_origin", sim_step=0, step_domain=raw["step_domain"],
                cameras={"fixture_camera": dict(T_world_camera=camera_pose, frame_id="frame-calibrated", source_kind="measurement")}),
            robot_model=dict(base_link="base", joints=[], source_kind="public_robot_model"))
        bundle = harness.publish_observation(raw)
        metadata = json.loads(harness.observation_content(bundle, {}, detail="full")[0]["text"])
        self.assertEqual(metadata["camera_calibration"], raw["camera_calibration"])
        self.assertEqual(metadata["camera_pose_sample"], raw["camera_pose_sample"])
        frame = bundle["cameras"][0]
        for ref in (frame["rgb_ref"], frame["calibration_ref"], frame["pose_sample_ref"],
                    bundle["robot_model_ref"], bundle["robot_state_ref"]):
            self.assertEqual(store.resolve_artifact(ref)[1], scene_artifacts.resolve_artifact(ref)[1])
        self.assertEqual(json.loads(store.resolve_artifact(bundle["robot_model_ref"])[1]), raw["robot_model"])
        camera = GeometryTools(harness.scene).camera(identity, bundle["observation_id"], "fixture_camera")
        self.assertEqual(camera["K"].tolist(), intrinsic)
        self.assertEqual(camera["T"].tolist(), camera_pose)
        with self.assertRaisesRegex(ValueError, "missing_geometry"):
            harness.publish_observation({**robot.observe(), "observation_profile": ACTOR_PROFILE})

    def test_native_capability_contact_unavailable_is_not_contact_truth(self):
        for profile in (ACTOR_PROFILE, LEGACY_ACTOR_PROFILE):
            capabilities = dict(observation_profile=profile, gripper_semantics=dict(
                command="normalized last command", contact_feedback="unavailable"))
            require_actor_observation(capabilities, profile)
            for feedback in ({"contacts": []}, "measured", False, 0, []):
                with self.subTest(profile=profile, feedback=feedback), self.assertRaisesRegex(ValueError, "contact_truth"):
                    require_actor_observation({**capabilities, "gripper_semantics": {"contact_feedback": feedback}}, profile)



# --- Folded from tests/unit/test_sensor_stream_p0.py ---

TABLE_Z = 0.765


class _CountingGpuTensor:
    """记录 `.cpu()` 卸载次数的模拟 CUDA 张量，用于验证单帧骨干特征零重复拷贝。"""

    def __init__(self, arr: np.ndarray, device_type: str = "cuda") -> None:
        self._arr = np.asarray(arr, dtype=np.float32)
        self.device = type("_Dev", (), {"type": device_type})()
        self.cpu_calls = 0

    def detach(self) -> "_CountingGpuTensor":
        return self

    def float(self) -> "_CountingGpuTensor":
        return self

    def cpu(self) -> "_CountingGpuTensor":
        self.cpu_calls += 1
        out = _CountingGpuTensor(self._arr.copy(), device_type="cpu")
        return out

    def numpy(self) -> np.ndarray:
        return self._arr


class _ThirdPersonAndProgressRobot(CalibratedThreeCamRobot):
    """支持自定义主固定相机名与动作期 5Hz 过程帧序列注入的标定机器人夹具。"""

    def __init__(
        self,
        root: Path,
        *,
        fixed_camera_id: str = "cam_head",
        left_wrist_id: str = "cam_left_wrist",
        omit_terminal: bool = False,
    ) -> None:
        super().__init__(root)
        self.fixed_camera_id = fixed_camera_id
        self.left_wrist_id = left_wrist_id
        self.omit_terminal = omit_terminal
        self.qdot_left: float | None = 0.0
        self.jaw_left: float = 0.04
        self.terminal_q_left: float | None = None
        self.terminal_jaw_left: float | None = None
        self.scripted_steps: list[dict] = []
        self.latest_ids_during_progress: list[str | None] = []
        self.harness_ref: Harness | None = None

    def observe(self) -> dict:
        raw = super().observe()
        # 支持将默认 cam_head / cam_left_wrist 替换为第三方固定机位或自定义手眼相机 ID
        for cam in raw["cameras"]:
            if cam["camera_id"] == "cam_head":
                cam["camera_id"] = self.fixed_camera_id
            elif cam["camera_id"] == "cam_left_wrist":
                cam["camera_id"] = self.left_wrist_id

        cal_cams = raw["camera_calibration"]["cameras"]
        if self.fixed_camera_id != "cam_head":
            cal_cams[self.fixed_camera_id] = cal_cams.pop("cam_head")
        if self.left_wrist_id != "cam_left_wrist":
            cal_cams[self.left_wrist_id] = cal_cams.pop("cam_left_wrist")

        pose_cams = raw["camera_pose_sample"]["cameras"]
        if self.fixed_camera_id != "cam_head":
            pose_cams[self.fixed_camera_id] = pose_cams.pop("cam_head")
        if self.left_wrist_id != "cam_left_wrist":
            pose_cams[self.left_wrist_id] = pose_cams.pop("cam_left_wrist")

        raw["robot_state"]["arms"]["left"]["gripper"]["position_m"] = [float(self.jaw_left)]
        if self.qdot_left is not None:
            raw["robot_state"]["arms"]["left"]["qdot_rad_s"] = [float(self.qdot_left)]
        raw["robot_state"]["arms"]["right"]["qdot_rad_s"] = [0.0]
        return raw

    def execute(self, action, progress_callback=None, cancel_flag=None) -> dict:
        orig_left_base = self.t_left_base.copy()
        for step_cfg in self.scripted_steps:
            self.executions += 1
            self.q_left = float(step_cfg.get("q_left", self.q_left))
            self.qdot_left = step_cfg.get("qdot_left", 0.0)
            if "jaw_left" in step_cfg:
                self.jaw_left = float(step_cfg["jaw_left"])
            if "t_left_base" in step_cfg:
                self.t_left_base = step_cfg["t_left_base"].copy()
            obs = self.observe()
            if progress_callback is not None:
                progress_callback(
                    dict(
                        phase=step_cfg.get("phase", "progress"),
                        sim_step=self.executions,
                        executed_steps=self.executions,
                        data_origin="synthetic_fixture",
                        observation=obs,
                    )
                )
            if self.harness_ref is not None:
                curr = self.harness_ref.latest
                self.latest_ids_during_progress.append(curr["observation_id"] if curr else None)

        # 终态采样：恢复或切换至指定终态机位与终态夹爪开度
        self.executions += 1
        self.t_left_base = orig_left_base
        if self.terminal_q_left is not None:
            self.q_left = float(self.terminal_q_left)
        if self.terminal_jaw_left is not None:
            self.jaw_left = float(self.terminal_jaw_left)
        self.qdot_left = 0.0
        terminal_obs = None if self.omit_terminal else self.observe()
        state_doc = (terminal_obs or self.observe())["robot_state"]
        return dict(
            status="completed",
            reason=None,
            executed_steps=max(1, len(self.scripted_steps) + 1),
            executed_native_steps=max(1, len(self.scripted_steps) + 1),
            completed_native_steps=max(1, len(self.scripted_steps) + 1),
            data_origin="synthetic_fixture",
            terminal_observation=terminal_obs,
            planned_trace=dict(data_origin="synthetic_fixture", targets=action.get("path")),
            dispatched_trace=dict(data_origin="synthetic_fixture", steps=1),
            actual_trace=[dict(data_origin="synthetic_fixture", state=state_doc)],
        )


def _make_action(identity: dict, based_on_obs_id: str, action_id: str) -> dict:
    """构造最小合法观测世界绝对轨迹动作请求。"""
    return dict(
        schema_version="0.1",
        **identity,
        world_kind="observed",
        action_id=action_id,
        based_on_observation_id=based_on_obs_id,
        frame_id=FRAME_ID,
        tcp_config_id=TCP_CONFIG_ID,
        mode="absolute",
        limits_profile="simulation-slow-v1",
        timeout_s=10,
        path=dict(
            kind="backend_free",
            orientation="backend_free",
            blend_radius_m=0,
            position_tolerance_m=0.005,
            waypoints=[
                dict(
                    left=dict(
                        position_m=[-0.10, -0.22, 0.96],
                        quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
                    )
                )
            ],
        ),
        gripper_events=[],
    )


class SensorFrameContractTests(unittest.TestCase):
    """验证通用相机拓扑角色判定、第三方固定视角 `scene3d` 查询及 RGB-D/双目/显存预算契约。"""

    def test_camera_role_topology_and_third_person_scene3d_query(self) -> None:
        # 1. 第三方三脚架/龙门架固定机位与历史 cam_head 均归类为 fixed_external
        for fixed_id in ("cam_head", "cam_third_person", "cam_overhead", "tripod_front_rgb"):
            self.assertEqual(classify_camera_role(fixed_id, {"T_world_camera": np.eye(4).tolist()}), CAMERA_ROLE_FIXED_EXTERNAL)
            self.assertIsNone(resolve_camera_arm(fixed_id, {"T_world_camera": np.eye(4).tolist()}))

        # 2. 运动学连杆手眼标定相机（无论名称是否含 wrist）与历史腕部命名均归类为 wrist_mounted
        ee_cal = {"arm": "right", "link_name": "link6", "T_link_camera": np.eye(4).tolist()}
        self.assertEqual(classify_camera_role("cam_ee_aux", ee_cal), CAMERA_ROLE_WRIST_MOUNTED)
        self.assertEqual(resolve_camera_arm("cam_ee_aux", ee_cal), "right")
        self.assertEqual(classify_camera_role("cam_left_wrist"), CAMERA_ROLE_WRIST_MOUNTED)
        self.assertEqual(resolve_camera_arm("cam_left_wrist"), "left")

        # 3. 矛盾挂载元数据与非法角色名必须直接抛 ValueError，严禁静默容错
        with self.assertRaisesRegex(ValueError, "contradictory_camera_mount_metadata"):
            classify_camera_role("cam_bad", {"mount_role": "fixed_external", "arm": "left"})
        with self.assertRaisesRegex(ValueError, "unsupported_camera_role"):
            classify_camera_role("cam_bad", {"mount_role": "floating_drone"})
        with self.assertRaisesRegex(ValueError, "conflicting_explicit_camera_roles"):
            classify_camera_role("cam_bad", {"mount_role": "fixed_external"}, {"mount_role": "wrist_mounted"})

        # 4. 完全不含 cam_head、仅含 cam_third_person + cam_ee_aux 的观测包可直接驱动 scene3d 查询
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        identity = dict(run_id="r-tp", episode_id="ep-tp", env_instance_id="env-tp", env_generation=1)
        store = Store(Path(tmp) / "store", identity)
        self.addCleanup(store.close)
        fake_points = FakePoints()
        self.addCleanup(fake_points.close)

        robot = _ThirdPersonAndProgressRobot(
            Path(tmp) / "caps",
            fixed_camera_id="cam_third_person",
            left_wrist_id="cam_ee_aux",
        )
        cfg = Config.from_dict(
            {
                "features": {
                    "external_perception": False,
                    "scene3d": True,
                    "vggt_endpoint": fake_points.url,
                    "vggt_support_z_m": TABLE_Z,
                    "motion_review": False,
                }
            }
        )
        harness = Harness(store, robot, config=cfg, observation_profile="rgb_proprio_calibrated_v1")
        install_registry(harness, cfg)
        obs = harness.start()

        resolve_json = harness.scene.resolve_json
        self.assertEqual(list_fixed_camera_ids(obs, resolve_json=resolve_json), ("cam_third_person",))
        self.assertEqual(list_wrist_camera_ids(obs, resolve_json=resolve_json), ("cam_ee_aux", "cam_right_wrist"))
        self.assertEqual(select_primary_fixed_camera_id(obs, resolve_json=resolve_json), "cam_third_person")

        resp = harness.invoke(
            "geometry_query",
            {"request": {"kind": "scene3d", "observation_id": obs["observation_id"], "include_wrist_views": True}},
            {"tool_call_id": "tp-scene3d"},
            role="actor",
        )
        self.assertEqual(resp.data["status"], "conditional", resp.data)
        self.assertEqual(resp.data["value"]["camera_id"], "cam_third_person")
        self.assertEqual(len(fake_points.requests), 1)
        sent_body = fake_points.requests[0][1]
        np.testing.assert_allclose(sent_body["T_world_camera"], robot.t_head, atol=1e-6)
        extra_ids = [v["camera_id"] for v in sent_body.get("extra_views", [])]
        self.assertIn("cam_ee_aux", extra_ids)

    def test_depth_stereo_and_stream_budget_contracts(self) -> None:
        # 1. DepthFrameSpec 合法构造与 from_camera_frame 解析
        depth = DepthFrameSpec(
            depth_ref="artifact:depth-001",
            camera_id="cam_third_person",
            image_size_px=(640, 480),
            unit="m",
            depth_scale=0.001,
            valid_range_m=(0.08, 2.0),
            timestamp_ns=1731398400123456789,
        )
        self.assertEqual(depth.unit, "m")
        self.assertAlmostEqual(depth.depth_scale, 0.001)
        self.assertEqual(depth.valid_range_m, (0.08, 2.0))

        from_cam = DepthFrameSpec.from_camera_frame(
            {
                "camera_id": "cam_right_wrist",
                "depth_ref": "artifact:depth-wrist",
                "width": 320,
                "height": 240,
                "depth_unit": "m",
                "depth_clip_m": [0.05, 1.5],
            }
        )
        self.assertEqual(from_cam.image_size_px, (320, 240))
        self.assertEqual(from_cam.valid_range_m, (0.05, 1.5))

        for bad_kwargs in (
            dict(depth_ref="", camera_id="c", image_size_px=(640, 480)),
            dict(depth_ref="artifact:x", camera_id="c", image_size_px=(640, 480), unit="mm"),
            dict(depth_ref="artifact:x", camera_id="c", image_size_px=(640, 480), depth_scale=0.0),
            dict(depth_ref="artifact:x", camera_id="c", image_size_px=(640, 480), valid_range_m=(1.5, 0.5)),
            dict(depth_ref="artifact:x", camera_id="c", image_size_px=(640, 480), timestamp_ns=-1),
            dict(depth_ref="artifact:x", camera_id="c", image_size_px=(True, 480)),
        ):
            with self.assertRaises(ValueError):
                DepthFrameSpec(**bad_kwargs)

        # 2. StereoPairSpec 合法构造与非法基线/同名相机拦截
        stereo = StereoPairSpec(
            left_camera_id="cam_stereo_left",
            right_camera_id="cam_stereo_right",
            baseline_m=0.065,
            is_rectified=True,
            image_size_px=(1280, 720),
            disparity_range_px=(0.0, 192.0),
        )
        self.assertAlmostEqual(stereo.baseline_m, 0.065)
        self.assertTrue(stereo.is_rectified)

        for bad_stereo in (
            dict(left_camera_id="cam_same", right_camera_id="cam_same", baseline_m=0.06),
            dict(left_camera_id="cam_l", right_camera_id="cam_r", baseline_m=0.001),
            dict(left_camera_id="cam_l", right_camera_id="cam_r", baseline_m=0.06, is_rectified=1),  # type: ignore[arg-type]
            dict(left_camera_id="cam_l", right_camera_id="cam_r", baseline_m=0.06, disparity_range_px=(128.0, 64.0)),
        ):
            with self.assertRaises(ValueError):
                StereoPairSpec(**bad_stereo)

        # 3. StreamBudgetConfig 单卡 24GB 默认预算与 40GB 扩展预算
        b24 = StreamBudgetConfig.for_vram_gb(24.0)
        self.assertEqual((b24.vram_budget_gb, b24.max_gpu_hot_sessions, b24.max_keyframes, b24.max_wrist_progress_frames), (24.0, 4, 6, 4))
        self.assertAlmostEqual(b24.target_progress_hz, 5.0)
        self.assertAlmostEqual(b24.max_wrist_angular_speed_rad_s, 0.8)

        b40 = StreamBudgetConfig.for_vram_gb(40.0)
        self.assertEqual((b40.vram_budget_gb, b40.max_gpu_hot_sessions, b40.max_keyframes, b40.max_wrist_progress_frames), (40.0, 8, 10, 6))

        with self.assertRaises(ValueError):
            StreamBudgetConfig(vram_budget_gb=4.0)
        with self.assertRaises(ValueError):
            StreamBudgetConfig(max_wrist_angular_speed_rad_s=0.0)


class HarnessAnchorAndWristStreamTests(unittest.TestCase):
    """验证 `Harness.execute` 动作期 `make_current=False` 锚点不变性与 5Hz 腕部准静态流门控缓存。"""

    def test_progress_frames_never_overwrite_latest_anchor_and_missing_terminal_clears_latest(self) -> None:
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        identity = dict(run_id="r-anchor", episode_id="ep-anchor", env_instance_id="env-anchor", env_generation=1)
        store = Store(Path(tmp) / "store", identity)
        self.addCleanup(store.close)
        robot = _ThirdPersonAndProgressRobot(Path(tmp) / "caps")
        harness = Harness(store, robot, observation_profile="rgb_proprio_calibrated_v1")
        robot.harness_ref = harness

        init_obs = harness.start()
        pre_obs_id = init_obs["observation_id"]

        # 注入 2 帧 5Hz 运动中途观测包
        robot.scripted_steps = [
            dict(q_left=0.02, qdot_left=0.4, phase="started"),
            dict(q_left=0.05, qdot_left=0.2, phase="progress"),
        ]
        out = harness.execute(_make_action(identity, pre_obs_id, "act-with-terminal"))
        self.assertEqual(out["receipt"]["status"], "completed")
        # 运动过程中 harness.latest 始终锚定在动作前的正式观测，绝不被中间 5Hz 帧覆写
        self.assertEqual(robot.latest_ids_during_progress, [pre_obs_id, pre_obs_id])
        # 动作完成后才正式推进为终态观测
        term_obs_id = out["receipt"]["result_observation_id"]
        self.assertNotEqual(term_obs_id, pre_obs_id)
        self.assertEqual(harness.latest["observation_id"], term_obs_id)

        obs_events = [e for e in store.events() if e["type"] == "observation.available"]
        # 顺序：初始观测(False) -> 过程帧1(True) -> 过程帧2(True) -> 终态观测(False)
        self.assertEqual([e["payload"]["historical"] for e in obs_events], [False, True, True, False])

        # 验证若后端终态观测缺失（omit_terminal=True），即使中途发过 5Hz 帧，harness.latest 也必须置为 None
        robot.omit_terminal = True
        robot.latest_ids_during_progress.clear()
        robot.scripted_steps = [dict(q_left=0.08, qdot_left=0.1, phase="progress")]
        out_no_term = harness.execute(_make_action(identity, term_obs_id, "act-missing-terminal"))
        self.assertEqual(robot.latest_ids_during_progress, [term_obs_id])
        self.assertIsNone(out_no_term["receipt"]["result_observation_id"])
        self.assertIsNone(harness.latest)

    def test_wrist_progress_quasistatic_gating_and_scene3d_extra_view_fusion(self) -> None:
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        identity = dict(run_id="r-wrist", episode_id="ep-wrist", env_instance_id="env-wrist", env_generation=1)
        store = Store(Path(tmp) / "store", identity)
        self.addCleanup(store.close)
        fake_points = FakePoints()
        self.addCleanup(fake_points.close)

        robot = _ThirdPersonAndProgressRobot(Path(tmp) / "caps")
        cfg = Config.from_dict(
            {
                "features": {
                    "external_perception": False,
                    "scene3d": True,
                    "vggt_endpoint": fake_points.url,
                    "vggt_support_z_m": TABLE_Z,
                    "motion_review": False,
                }
            }
        )
        harness = Harness(store, robot, config=cfg, observation_profile="rgb_proprio_calibrated_v1")
        install_registry(harness, cfg)
        init_obs = harness.start()

        near_base = robot.t_left_base.copy()
        # 终态将左臂停泊至与右臂相同的高处台外视角，且终态关节位移设为 0.25m（异于动作中途的 0.005m 与 0.08m）
        robot.t_left_base = robot.t_right_base.copy()
        robot.terminal_q_left = 0.25

        # 动作中途依次经历：
        # step 1: 高速挥臂（2.4 rad/s > 0.8 rad/s） -> 被快门防模糊门控拦截
        # step 2: 近场准静态机位 A（0.15 rad/s, q_left=0.00） -> 入列缓存
        # step 3: 机位 A 原地微移 5mm（0.10 rad/s, q_left=0.005 < 2cm） -> 原地去重替换
        # step 4: 近场准静态机位 B（0.18 rad/s, q_left=0.08 >= 4cm 平移基线） -> 入列新视角
        robot.scripted_steps = [
            dict(t_left_base=near_base, q_left=0.00, qdot_left=2.40, phase="started"),
            dict(t_left_base=near_base, q_left=0.00, qdot_left=0.15, phase="progress"),
            dict(t_left_base=near_base, q_left=0.005, qdot_left=0.10, phase="progress"),
            dict(t_left_base=near_base, q_left=0.08, qdot_left=0.18, phase="progress"),
        ]
        exec_out = harness.execute(_make_action(identity, init_obs["observation_id"], "act-approach-hover"))
        term_obs_id = exec_out["receipt"]["result_observation_id"]

        cache: WristProgressCache = harness._wrist_progress_cache
        # 右腕始终为 0 rad/s 共 4 帧（原地悬停去重为 1 条），左腕 step 1 被拒、step 2&3 去重合并为 1 条、step 4 新增 1 条
        self.assertEqual(cache.rejected_motion_blur, 1)
        left_cached = [f for f in cache.snapshot() if f.camera_id == "cam_left_wrist"]
        self.assertEqual(len(left_cached), 2)
        # 原地去重保留了 step 3（而非 step 2）与 step 4
        self.assertEqual([f.sim_step for f in left_cached], [3, 4])

        scene3d_mod = __import__("ni_rua.perception.external.scene3d", fromlist=["scene3d_query"])
        from ni_rua.runtime.tools.builtins import create_registry
        from ni_rua.runtime.tools.definitions import public_definitions

        # 校验公开工具契约同时支持 actor 与 program 角色传入 include_progress_wrist_views
        registry = create_registry(runtime=harness)
        for role in ("actor", "program"):
            registry.validate(
                "geometry_query",
                {
                    "request": {
                        "kind": "scene3d",
                        "observation_id": term_obs_id,
                        "include_wrist_views": True,
                        "include_progress_wrist_views": True,
                    }
                },
                role=role,
            )
        s3_defs = {d["name"]: d for d in public_definitions(scene3d=True)}
        self.assertIn("include_progress_wrist_views", s3_defs["geometry_query"]["description"])

        # 1) 默认 include_wrist_views=True（不传 include_progress_wrist_views，模拟 obs3d 默认路径）：
        #    严禁后台隐式混入历史 5Hz 过程帧！终态实时腕部朝向台外，故 extra_views 为空
        res_default, _ = scene3d_mod.scene3d_query(
            harness,
            {"observation_id": term_obs_id, "include_wrist_views": True},
            auto_filter_wrist_views=True,
        )
        self.assertEqual(res_default["status"], "conditional", res_default)
        default_body = fake_points.requests[-1][1]
        self.assertNotIn("extra_views", default_body)
        terminal_only_segs = default_body["robot_segments_m"]

        # 2) 显式指定 include_progress_wrist_views=True：
        #    正常注入缓存的清晰腕部帧 cam_left_wrist@4 与 cam_left_wrist@3，并将历史帧机械臂连杆合并进 robot_segments_m
        res_prog, _ = scene3d_mod.scene3d_query(
            harness,
            {
                "observation_id": term_obs_id,
                "include_wrist_views": True,
                "include_progress_wrist_views": True,
            },
            auto_filter_wrist_views=True,
        )
        self.assertEqual(res_prog["status"], "conditional", res_prog)
        points_body = fake_points.requests[-1][1]
        extra_ids = [v["camera_id"] for v in points_body.get("extra_views", [])]
        self.assertEqual(extra_ids, ["cam_left_wrist@4", "cam_left_wrist@3"])
        merged_segs = points_body["robot_segments_m"]
        self.assertGreater(len(merged_segs), len(terminal_only_segs))

        # 3) 单独调用 robot_observe（last_action_id=None）即使显式传 include_progress_wrist_views=True 也不得跨动作误带旧帧
        fresh_obs = harness.observe()
        self.assertIsNone(fresh_obs["last_action_id"])
        res_fresh, _ = scene3d_mod.scene3d_query(
            harness,
            {
                "observation_id": fresh_obs["observation_id"],
                "include_wrist_views": True,
                "include_progress_wrist_views": True,
            },
            auto_filter_wrist_views=True,
        )
        self.assertEqual(res_fresh["status"], "conditional")
        self.assertNotIn("extra_views", fake_points.requests[-1][1])

        # 验证无 qdot_rad_s 时有限差分回退与跨 env_generation 自动隔离清空
        diff_speed = measure_arm_joint_speed_rad_s(
            {"q_rad": [0.20, -0.10]},
            prev_q_rad=np.array([0.00, -0.10]),
            dt_s=0.20,
        )
        self.assertAlmostEqual(diff_speed, 1.0, places=6)

        next_gen_obs = deepcopy(fresh_obs)
        next_gen_obs["env_generation"] = 2
        cache.seed_pre_action_state(next_gen_obs)
        self.assertEqual(cache.snapshot(), [])

    def test_gripper_aperture_jump_invalidates_pre_grasp_and_transport_ghost_views(self) -> None:
        """验证 `[下探开爪 -> 闭爪抓取 -> 搬运 -> 开爪释放]` 与终态闭爪跃迁均严格剔除旧接触相位过程帧。"""
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        identity = dict(run_id="r-jaw", episode_id="ep-jaw", env_instance_id="env-jaw", env_generation=1)
        store = Store(Path(tmp) / "store", identity)
        self.addCleanup(store.close)
        fake_points = FakePoints()
        self.addCleanup(fake_points.close)

        robot = _ThirdPersonAndProgressRobot(Path(tmp) / "caps")
        cfg = Config.from_dict(
            {
                "features": {
                    "external_perception": False,
                    "scene3d": True,
                    "vggt_endpoint": fake_points.url,
                    "vggt_support_z_m": TABLE_Z,
                    "motion_review": False,
                }
            }
        )
        harness = Harness(store, robot, config=cfg, observation_profile="rgb_proprio_calibrated_v1")
        install_registry(harness, cfg)
        init_obs = harness.start()

        near_base = robot.t_left_base.copy()
        robot.t_left_base = robot.t_right_base.copy()
        robot.terminal_jaw_left = 0.040

        # 动作 A：完整搬运序列（起终点均为开爪 0.040m，中途闭爪 0.005m 搬运后在 q_left=0.12m 处释放）
        # - step 1 (q=0.00, jaw=0.040): 搬运前下探悬停（拍到旧位置物体）
        # - step 2 (q=0.00, jaw=0.005): 闭爪抓取（开度跳变 35mm > 4mm -> 立即清空 step 1！）
        # - step 3 (q=0.06, jaw=0.005): 持物搬运准静态帧
        # - step 4 (q=0.12, jaw=0.040): 开爪释放后悬停（开度跳变 35mm > 4mm -> 立即清空 step 2 & 3！）
        robot.scripted_steps = [
            dict(t_left_base=near_base, q_left=0.00, qdot_left=0.10, jaw_left=0.040, phase="progress"),
            dict(t_left_base=near_base, q_left=0.00, qdot_left=0.05, jaw_left=0.005, phase="progress"),
            dict(t_left_base=near_base, q_left=0.06, qdot_left=0.12, jaw_left=0.005, phase="progress"),
            dict(t_left_base=near_base, q_left=0.12, qdot_left=0.08, jaw_left=0.040, phase="progress"),
        ]
        out_a = harness.execute(_make_action(identity, init_obs["observation_id"], "act-pick-place"))
        term_a_id = out_a["receipt"]["result_observation_id"]

        cache: WristProgressCache = harness._wrist_progress_cache
        self.assertGreaterEqual(cache.invalidated_by_gripper_jump, 2)
        scene3d_mod = __import__("ni_rua.perception.external.scene3d", fromlist=["scene3d_query"])
        res_a, _ = scene3d_mod.scene3d_query(
            harness,
            {
                "observation_id": term_a_id,
                "include_wrist_views": True,
                "include_progress_wrist_views": True,
            },
            auto_filter_wrist_views=True,
        )
        self.assertEqual(res_a["status"], "conditional")
        # 仅保留释放后同世界状态的 step 4（executions==4），绝不混入搬运前 step 1 或搬运中 step 2/3
        extra_a = [v["camera_id"] for v in fake_points.requests[-1][1].get("extra_views", [])]
        self.assertEqual(extra_a, ["cam_left_wrist@4"])

        # 动作 B：过程帧采样于开爪 0.040m，终态才完成闭爪至 0.004m（差值 36mm > 4mm，无闭爪后过程帧）
        # 验证 select_extra_views 终态开度一致性校验直接剔除闭爪前过程帧
        robot.scripted_steps = [
            dict(t_left_base=near_base, q_left=0.02, qdot_left=0.10, jaw_left=0.040, phase="progress"),
        ]
        robot.terminal_jaw_left = 0.004
        out_b = harness.execute(_make_action(identity, term_a_id, "act-terminal-grasp"))
        term_b_id = out_b["receipt"]["result_observation_id"]

        res_b, _ = scene3d_mod.scene3d_query(
            harness,
            {
                "observation_id": term_b_id,
                "include_wrist_views": True,
                "include_progress_wrist_views": True,
            },
            auto_filter_wrist_views=True,
        )
        self.assertEqual(res_b["status"], "conditional")
        self.assertNotIn("extra_views", fake_points.requests[-1][1])


class Sam31SingleCopyAndHotSessionTests(unittest.TestCase):
    """验证 SAM 3.1 锁外单次特征提取数学等价性、零重复 GPU->CPU 拷贝风暴与热会话配置。"""

    def test_single_frame_feature_extraction_equivalence_and_zero_redundant_copies(self) -> None:
        rng = np.random.default_rng(42)
        H, W = 128, 160
        img_rgb = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
        raw_feat = rng.standard_normal((1, 16, 16, 20)).astype(np.float32)

        # 生成 12 个互不重叠的候选掩码（分布在左上 88×116 区域）
        candidates = []
        for idx in range(12):
            r0 = (idx // 4) * 28 + 4
            c0 = (idx % 4) * 28 + 4
            m = np.zeros((H, W), dtype=bool)
            m[r0 : r0 + 20, c0 : c0 + 22] = True
            candidates.append(dict(mask=m, score=0.92 - idx * 0.01, labels=[[f"obj_{idx}", 0.9]]))

        # 先准备第 0 帧种入一个将在第 1 帧完全漏检需走 predict_fn 传播的目标（位于右下角独立无重叠区域），以及一个需走 reanchor 精化的目标
        mgr = vsm.VideoSessionManager(max_gpu_hot_sessions=4)
        extra_mask_missed = np.zeros((H, W), dtype=bool)
        extra_mask_missed[98:122, 128:154] = True
        mgr.update_tracks(
            session_id="s-equiv",
            img_rgb=img_rgb,
            image_sha256="frame-0",
            candidates=[
                dict(mask=candidates[0]["mask"], score=0.95, labels=[["obj_0", 0.95]]),
                dict(mask=extra_mask_missed, score=0.93, labels=[["missed_later", 0.93]]),
            ],
            now=100.0,
        )

        # 逐掩码调用原始独立路径计算期望外观特征基准
        ref_backbone = _CountingGpuTensor(raw_feat, device_type="cuda")
        expected_by_first_cand = vsm.extract_appearance_feature(img_rgb, candidates[1]["mask"], ref_backbone)
        expected_reanchored = vsm.extract_appearance_feature(img_rgb, candidates[0]["mask"], ref_backbone)
        expected_propagated = vsm.extract_appearance_feature(img_rgb, extra_mask_missed, ref_backbone)

        # 在第 1 帧同时触发：12 个候选掩码 + reanchor(o1) + predict_fn 补回漏检(o2)
        shared_gpu_feat = _CountingGpuTensor(raw_feat, device_type="cuda")
        real_cvt_color = cv2.cvtColor
        rgb2hsv_calls = 0
        predict_calls = 0

        def _counting_cvt_color(src, code, *args, **kwargs):
            nonlocal rgb2hsv_calls
            if code == cv2.COLOR_RGB2HSV:
                rgb2hsv_calls += 1
            return real_cvt_color(src, code, *args, **kwargs)

        def _fake_predict_fn(*, point_coords=None, point_labels=None, box=None, mask_input=None):
            nonlocal predict_calls
            predict_calls += 1
            if box is not None and float(box[0]) > 100.0:
                m_out = extra_mask_missed
            else:
                m_out = candidates[0]["mask"]
            return m_out[None, :, :], np.array([0.91], dtype=np.float32), np.zeros((1, 288, 288), dtype=np.float32)

        with mock.patch("services.vggt_height.sam31.video_session_manager.cv2.cvtColor", side_effect=_counting_cvt_color):
            res = mgr.update_tracks(
                session_id="s-equiv",
                img_rgb=img_rgb,
                image_sha256="frame-1",
                candidates=candidates,
                reanchor={"target_track_id": "o1", "pos_pixels": [[14, 14]]},
                predict_fn=_fake_predict_fn,
                backbone_feat=shared_gpu_feat,
                now=101.0,
            )

        # 关键断言：无论多少候选掩码、reanchor 与 predict_fn，整帧 GPU->CPU 拷贝与 RGB2HSV 转换均严格只执行 1 次！
        self.assertEqual(predict_calls, 2)
        self.assertEqual(shared_gpu_feat.cpu_calls, 1)
        self.assertEqual(rgb2hsv_calls, 1)
        self.assertEqual(len(res["instances"]), 13)

        # 验证新分配目标（无 EMA 历史混合）及重锚定/传播目标的外观特征与逐个独立提取 100% 数学等价
        tracks = mgr._sessions["s-equiv"].tracks
        # track 3 对应第 1 帧新出现的 candidates[1]
        np.testing.assert_allclose(tracks[3].appearance, expected_by_first_cand, atol=1e-7)
        # 验证直接显式传预计算 img_hsv + backbone_feat_np 对全部 12 个掩码均与独立计算逐元素一致
        pre_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        pre_feat_np = vsm.prepare_backbone_feature_np(_CountingGpuTensor(raw_feat, "cuda"))
        for cand in candidates:
            f_single = vsm.extract_appearance_feature(img_rgb, cand["mask"], _CountingGpuTensor(raw_feat, "cuda"))
            f_shared = vsm.extract_appearance_feature(
                img_rgb, cand["mask"], img_hsv=pre_hsv, backbone_feat_np=pre_feat_np
            )
            np.testing.assert_allclose(f_shared, f_single, atol=1e-7)
        self.assertEqual(expected_reanchored.shape, expected_propagated.shape)

    def test_max_gpu_hot_sessions_default_4_and_env_override(self) -> None:
        self.assertEqual(vsm.DEFAULT_MAX_GPU_HOT_SESSIONS, 4)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NI_RUA_SAM31_MAX_HOT_SESSIONS", None)
            self.assertEqual(vsm.resolve_max_gpu_hot_sessions(), 4)

        with mock.patch.dict(os.environ, {"NI_RUA_SAM31_MAX_HOT_SESSIONS": "6"}):
            self.assertEqual(vsm.resolve_max_gpu_hot_sessions(), 6)
            mgr6 = vsm.VideoSessionManager(max_gpu_hot_sessions=vsm.resolve_max_gpu_hot_sessions())
            self.assertEqual(mgr6.max_gpu_hot_sessions, 6)

        with mock.patch.dict(os.environ, {"NI_RUA_SAM31_MAX_HOT_SESSIONS": "0"}):
            self.assertEqual(vsm.resolve_max_gpu_hot_sessions(), 0)

        for invalid_val in ("-1", "abc"):
            with mock.patch.dict(os.environ, {"NI_RUA_SAM31_MAX_HOT_SESSIONS": invalid_val}):
                with self.assertRaises(ValueError):
                    vsm.resolve_max_gpu_hot_sessions()



if __name__ == "__main__":
    unittest.main()
