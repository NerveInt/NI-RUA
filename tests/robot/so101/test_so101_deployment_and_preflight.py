"""Deployment contract regressions for bounded SO-101 contact trajectories (CPU only)."""

NI_RUA_TEST_TIER = "fast"

# === tests/test_so101_deployment.py ===

from copy import deepcopy
import ast
import asyncio
import json
import math
import os
from pathlib import Path
import sys
from support.paths import CONFIGS_ROOT, REPO_ROOT, isolate_home
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np
from aiohttp.test_utils import TestClient, TestServer

from ni_rua.apps.gateway.workbench import Workbench
from ni_rua.core.config import Config
from ni_rua.cortex.agents.factory import backend_requirements
from ni_rua.robot.backends.so101.backend import RobotBackend
from ni_rua.robot.backends.so101.deployment import FIELDS, load_deployment, validate_deployment, write_joint_offsets
from ni_rua.robot.backends.so101.driver import Clock, FakeArmDriver
from ni_rua.robot.backends.so101.ik_plan import matrix_to_quaternion, pose_matrix, pose_residual
from ni_rua.robot.backends.so101.worker import SO101Adapter
from ni_rua.robot.model import so101_kinematics
from support import so101_scene
from support.fake_mcp_client import FakeMcpClient


def setUpModule():
    # 将主目录隔离至临时目录，避免配置中的波浪号路径触碰宿主状态
    isolate_home()


EXAMPLE = CONFIGS_ROOT / "so101.example.json"
TRAJECTORY_FIELDS = (
    "touch_slide_distance_m", "touch_slide_direction_rad", "touch_slide_step_m",
    "touch_max_tracking_error_rad", "touch_max_load_rise", "touch_slide_max_duration_s",
)


class ContactTrajectoryDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.document = json.loads(EXAMPLE.read_text())

    def test_legacy_deployment_loads_without_enabling_slide_or_mutating_input(self):
        for key in TRAJECTORY_FIELDS:
            del self.document["calibration"][key]
        before = deepcopy(self.document)
        normalized = validate_deployment(self.document)
        self.assertEqual(normalized["calibration"]["touch_slide_distance_m"], 0.0)
        self.assertEqual(normalized["calibration"]["touch_max_tracking_error_rad"], 0.15)
        self.assertEqual(normalized["calibration"]["touch_max_load_rise"], 300.0)
        self.assertEqual(self.document, before)
        for key in TRAJECTORY_FIELDS:
            self.assertIn(key, normalized["calibration"])

    def test_explicit_controls_are_preserved_exactly(self):
        explicit = dict(touch_slide_distance_m=0.01, touch_slide_direction_rad=-math.pi / 2,
                        touch_slide_step_m=0.0005, touch_max_tracking_error_rad=0.08,
                        touch_max_load_rise=180.0, touch_slide_max_duration_s=6.0)
        self.document["calibration"].update(explicit)
        normalized = validate_deployment(self.document)["calibration"]
        self.assertEqual({key: normalized[key] for key in explicit}, explicit)

    def test_invalid_controls_rejected_before_a_worker_can_start(self):
        invalid = dict(touch_slide_distance_m=[-0.001, 0.051],
                       touch_slide_direction_rad=[-math.pi - 0.001, math.pi + 0.001],
                       touch_slide_step_m=[0, -0.001, 0.006],
                       touch_max_tracking_error_rad=[0, -0.001, 0.501],
                       touch_max_load_rise=[0, -1, 1001],
                       touch_slide_max_duration_s=[0, -1, 61])
        for key, values in invalid.items():
            for value in values + [float("nan"), float("inf"), True, "0.01", None]:
                with self.subTest(key=key, value=value):
                    document = deepcopy(self.document)
                    document["calibration"][key] = value
                    with self.assertRaises(ValueError):
                        validate_deployment(document)

    def test_missing_existing_control_and_unknown_new_control_still_rejected(self):
        del self.document["calibration"]["touch_step_period_s"]
        with self.assertRaisesRegex(ValueError, "touch_step_period_s"):
            validate_deployment(self.document)
        self.document = json.loads(EXAMPLE.read_text())
        self.document["calibration"]["touch_slide_force_n"] = 2.0
        with self.assertRaisesRegex(ValueError, "touch_slide_force_n"):
            validate_deployment(self.document)

    def test_template_and_published_default_contract_match(self):
        for key in TRAJECTORY_FIELDS:
            with self.subTest(key=key):
                spec = FIELDS["calibration"][key]
                self.assertEqual(self.document["calibration"][key], spec["default"])
                self.assertTrue(spec["unit"])
                self.assertIn("template", spec["doc"].lower())

    def test_joint_offset_install_preserves_exact_float_basis_across_reload(self):
        # Solver outputs are not limited to microradians. Installing a rounded
        # value would invalidate strict active/live calibration version binding.
        offsets = dict(shoulder_pan=0.01234567890123456, wrist_flex=-0.000000123456789)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deployment.json"
            original = EXAMPLE.read_text()
            path.write_text(original)
            installed = write_joint_offsets(path, offsets, stamp="precision-test")
            reloaded = load_deployment(path)
            self.assertEqual(Path(installed["backup"]).read_text(), original)
            for joint, value in offsets.items():
                self.assertEqual(installed["offsets_rad"][joint], value)
                self.assertEqual(installed["deployment"]["joint_mapping"][joint]["offset_rad"], value)
                self.assertEqual(reloaded["joint_mapping"][joint]["offset_rad"], value)

# === tests/test_so101_preflight.py ===

import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from ni_rua.robot.backends.so101.deployment import load_deployment
from ni_rua.robot.backends.so101.preflight import PreflightError, run_preflight

NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
LIMITS = {name: (-1.7, 1.7) for name in NAMES[:5]}


class FakeBus:
    model_number_table = {"sts3215": 777}

    def __init__(self, **overrides):
        self.motors = {name: SimpleNamespace(id=i + 1, model="sts3215") for i, name in enumerate(NAMES)}
        self.calibration = {name: SimpleNamespace(range_min=900, range_max=3200) for name in NAMES}
        self.ping = {i + 1: 777 for i in range(6)}
        self.registers = {"Present_Voltage": dict.fromkeys(NAMES, 123), "Present_Temperature": dict.fromkeys(NAMES, 31),
                          "Status": dict.fromkeys(NAMES, 0), "Present_Position": dict.fromkeys(NAMES, 2048)}
        self.degrees = dict.fromkeys(NAMES, 0.0)
        self.writes = []
        for key, value in overrides.items():
            setattr(self, key, value)

    def broadcast_ping(self, num_retry=0):
        return self.ping

    def sync_read(self, register, normalize=True, num_retry=0):
        return dict(self.degrees) if register == "Present_Position" and normalize else dict(self.registers[register])

    def sync_write(self, *arguments, **keywords):
        self.writes.append(arguments)


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.deployment = load_deployment(EXAMPLE)

    def run_check(self, bus, calibrated=True, port_exists=True):
        import math
        return run_preflight(bus, calibrated=lambda: calibrated, deployment=self.deployment, limits_rad=LIMITS,
                             to_rad=lambda name, degrees: math.radians(degrees), port_exists=port_exists)

    def assert_code(self, code, bus, **arguments):
        with self.assertRaises(PreflightError) as caught:
            self.run_check(bus, **arguments)
        self.assertEqual(caught.exception.code, code)
        self.assertFalse(caught.exception.report["ok"])
        self.assertEqual(bus.writes if bus else [], [])
        return caught.exception

    def test_healthy_arm_passes_without_any_write(self):
        bus = FakeBus()
        report = self.run_check(bus)
        self.assertTrue(report["ok"])
        self.assertTrue(all(check["ok"] for check in report["checks"]))
        self.assertEqual(bus.writes, [])
        json.dumps(report)

    def test_relaxed_arm_on_an_end_stop_just_past_the_calibration_range_starts(self):
        # 2026-11-16 on the real arm: after a power cycle the forearm rested on its stop at raw 3106 while the hand-
        # recorded range ended at 3104 (2 ticks, 0.003 rad); the strict check refused to start.
        bus = FakeBus()
        bus.registers["Present_Position"]["elbow_flex"] = 3200 + 2
        report = self.run_check(bus)
        self.assertTrue(report["ok"])
        joint_range = next(check for check in report["checks"] if check["name"] == "joint_range")
        self.assertEqual(joint_range["beyond_calibration_ticks"], {"elbow_flex": 2})
        self.assertAlmostEqual(joint_range["calibration_range_tolerance_ticks"], 0.05 * 4096 / (2 * 3.141592653589793),
                               places=1)
        # Beyond the documented allowance it still stops startup, and 0 restores the strict check.
        bus = FakeBus()
        bus.registers["Present_Position"]["elbow_flex"] = 900 - 40
        self.assert_code("joint_outside_calibration_range", bus)
        self.deployment["preflight"]["calibration_range_tolerance_rad"] = 0.0
        bus = FakeBus()
        bus.registers["Present_Position"]["elbow_flex"] = 3200 + 2
        self.assert_code("joint_outside_calibration_range", bus)

    def test_each_fault_gets_its_own_diagnosis(self):
        self.assert_code("serial_port_missing", None, port_exists=False)
        self.assert_code("servo_power_off", FakeBus(ping=None))
        error = self.assert_code("servo_missing", FakeBus(ping={i: 777 for i in (1, 2, 3, 4)}))
        self.assertIn("5,6", error.detail)
        self.assert_code("servo_model_mismatch", FakeBus(ping={**{i + 1: 777 for i in range(6)}, 3: 1}))
        self.assert_code("motor_calibration_mismatch", FakeBus(), calibrated=False)
        for register, value, code in (("Present_Voltage", 70, "supply_voltage_out_of_range"),
                                      ("Present_Temperature", 75, "servo_overtemperature"),
                                      ("Status", 32, "servo_error_status"),
                                      ("Present_Position", 4000, "joint_outside_calibration_range")):
            bus = FakeBus()
            bus.registers[register]["elbow_flex"] = value
            with self.subTest(code=code):
                self.assert_code(code, bus)
        bus = FakeBus()
        bus.degrees["elbow_flex"] = 120.0  # 2.09 rad > 1.7 + tolerance: wrong mapping sign/offset
        self.assert_code("joint_outside_urdf_limits", bus)


def _parse_requirement_names(path: Path) -> set[str]:
    """解析 requirements 文件中的标准化包名（小写、下划线统一为连字符）。"""
    names: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        pkg = line.split(";", 1)[0].strip()
        for sep in ("==", ">=", "<=", "~=", "!=", ">", "<", "["):
            pkg = pkg.split(sep, 1)[0].strip()
        if pkg:
            names.add(pkg.lower().replace("_", "-"))
    return names


def _parse_conda_dependencies(path: Path) -> set[str]:
    """解析 environment.yml 中的 conda 依赖名称（去除版本号）。"""
    deps: set[str] = set()
    in_deps = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped == "dependencies:":
            in_deps = True
            continue
        if in_deps and raw and not raw.startswith((" ", "\t", "-")):
            break
        if in_deps and stripped.startswith("- ") and not stripped.startswith("- pip:"):
            item = stripped[2:].strip().split("=", 1)[0].split("<", 1)[0].split(">", 1)[0].strip()
            if item:
                deps.add(item.lower().replace("_", "-"))
    return deps


def _parse_mcp_json(result) -> dict:
    """解析 MCP tools/call 返回的首段 JSON 文本。"""
    if result.error is not None:
        raise AssertionError(f"MCP 调用返回 JSON-RPC 错误: {result.error}")
    payload = result.result
    if payload is None or payload.get("isError"):
        raise AssertionError(f"MCP 工具调用失败: {payload}")
    return json.loads(payload["content"][0]["text"])


class _UncalibratedSceneBackend(RobotBackend):
    """用于无标定冷启动验证的真机后端替身：保留真实 SO101Adapter 与 /calibration 链路，仅替换总线与相机句柄。"""

    def __init__(self, *, scene, **kwargs):
        super().__init__(**kwargs)
        self.scene = scene
        self.adapter = None
        self.close_report = None

    def start(self):
        if self.adapter is not None:
            return self.capabilities
        self.run_dir.mkdir(parents=True, exist_ok=True)
        clock = Clock()
        home = self.deployment["home"]
        driver = FakeArmDriver(self.deployment, clock, q_rad=home["q_rad"], gripper_normalized=home["gripper_normalized"])
        # 预置通过的上电自检报告，使 S0 设备自检能够真实核验 preflight.json
        driver.preflight_report = {"ok": True, "checks": []}
        intrinsics = [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        cameras = {
            "wrist": so101_scene.SceneCamera(
                "wrist", self.deployment["cameras"]["wrist"], clock, self.scene, intrinsics,
                lambda _q: np.eye(4), joints=lambda: list(driver.q),
            ),
            "fixed": so101_scene.SceneCamera(
                "fixed", self.deployment["cameras"]["fixed"], clock, self.scene, intrinsics,
                lambda _q: so101_scene.look_at(so101_scene.FIXED_POSITION_M, so101_scene.FIXED_TARGET_M),
                joints=lambda: list(driver.q),
            ),
        }
        self.adapter = SO101Adapter(
            self.deployment, self.run_dir, self.observation_profile,
            driver=driver, cameras=cameras, clock=clock,
            image_settings=self.image_settings, deployment_path=self.deployment_path,
        )
        self.capabilities = self.adapter.capabilities()
        return self.capabilities

    def _request(self, operation, payload=None, timeout=180, progress=None, cancel=None):
        payload = dict(payload or {})
        if operation == "observe":
            return self.adapter.observe()
        if operation == "record":
            return self.adapter.motion_video.record(**payload)
        if operation == "save_video":
            return self.adapter.motion_video.save()
        if operation == "operator":
            command = payload.pop("command")
            payload.pop("progress", None)
            return self.adapter.operator(command, progress=progress, **payload)
        raise ValueError(f"unsupported_operation:{operation}")

    def close(self):
        if self.adapter is not None:
            self.adapter.close()
        return {"native_close_acknowledged": True, "process_exit_confirmed": True, "reader_exit_confirmed": True}


class SO101EnvironmentAndConfigTests(unittest.TestCase):
    def test_so101_environment_dependencies_cover_required_third_party_imports(self):
        # 扫描 SO-101 worker、标定工作流、运动学与图像编码模块的第三方导入，核验 so101 环境清单完备
        scan_targets = (
            REPO_ROOT / "ni_rua" / "robot" / "backends" / "so101",
            REPO_ROOT / "ni_rua" / "robot" / "calibration" / "workflow",
            REPO_ROOT / "ni_rua" / "robot" / "model",
            REPO_ROOT / "ni_rua" / "robot" / "motion",
            REPO_ROOT / "ni_rua" / "core" / "storage" / "images.py",
        )
        # 仅用于 GPU planner 异常诊断的可选延迟导入（planner_witness.py）以及 ARX-X5/沙盒专用回退分支除外
        optional_excluded = {"torch", "curobo", "yaml", "kinematics"}
        import_to_package = {
            "pil": "pillow",
            "cv2": "opencv-python-headless",
            "pink": "pin-pink",
            "pinocchio": "pin",
        }
        discovered_imports: set[str] = set()
        for target in scan_targets:
            files = [target] if target.is_file() else sorted(target.rglob("*.py"))
            for path in files:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            top = alias.name.split(".", 1)[0]
                            if top not in sys.stdlib_module_names and top != "ni_rua":
                                discovered_imports.add(top)
                    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                        top = node.module.split(".", 1)[0]
                        if top not in sys.stdlib_module_names and top != "ni_rua":
                            discovered_imports.add(top)

        required_pip = {
            import_to_package.get(name.lower(), name.lower().replace("_", "-"))
            for name in discovered_imports - optional_excluded
        }
        # Pink + DAQP 求解器及 LeRobot 串口舵机运行时所需的间接显式依赖
        required_pip.update({"qpsolvers", "daqp", "feetech-servo-sdk", "pyserial"})

        declared_pip = _parse_requirement_names(REPO_ROOT / "requirements-so101.txt")
        declared_conda = _parse_conda_dependencies(REPO_ROOT / "environment.so101.yml")

        missing_pip = sorted(required_pip - declared_pip)
        self.assertEqual(missing_pip, [], f"requirements-so101.txt 缺失依赖: {missing_pip}")
        self.assertIn("ffmpeg", declared_conda, "environment.so101.yml 必须声明 ffmpeg 供 H.264 视频录制使用")

    def test_so101_standalone_and_remote_example_configs(self):
        standalone_path = CONFIGS_ROOT / "so101_standalone.json"
        remote_example_path = CONFIGS_ROOT / "so101.remote.example.json"
        for path in (standalone_path, remote_example_path):
            text = path.read_text(encoding="utf-8")
            for forbidden in ("/home/", "/Users/", ".gemini"):
                self.assertNotIn(forbidden, text, f"{path.name} 不得包含宿主私有绝对路径或内部目录: {forbidden}")

        # 核验单机无 GPU 默认配置：关闭全部远程 3D 感知依赖，同时通过托管启动校验
        standalone = Config.load(standalone_path, ignore_local=True).validate_launch()
        self.assertIsNone(backend_requirements(standalone))
        self.assertEqual(standalone.backend.environment, "so101_real")
        self.assertEqual(standalone.backend.robot_model_id, "so101")
        self.assertEqual(standalone.features.observation_profile, "rgb_proprio_calibrated_v1")
        self.assertFalse(standalone.features.sandbox_enabled)
        self.assertFalse(standalone.features.motion_review)
        self.assertFalse(standalone.remote.enabled)
        for flag in (
            "external_perception", "vggt_height", "object_table", "scene3d",
            "object_table_observe", "scene3d_observe", "held_perception",
        ):
            self.assertFalse(getattr(standalone.features, flag), f"单机配置 {flag} 必须为 False")

        # 核验脱敏后的远程 GPU 服务覆盖配置合法
        remote_cfg = Config.load(remote_example_path, ignore_local=True)
        self.assertTrue(remote_cfg.remote.enabled)
        self.assertEqual(remote_cfg.remote.host, "your-gpu-host")
        self.assertEqual(remote_cfg.remote.services[0].effective_local_port, 8331)
        self.assertEqual(
            remote_cfg.remote.services[0].env,
            ("VGGT_HOME=~/models/vggt", "VGGT_RUN_DIR=~/ni-rua-perception/run"),
        )


class SO101StandaloneCpuCalibrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_uncalibrated_cold_start_calibration_state_and_s0_self_check(self):
        # 强制无 GPU 环境并最先注册临时目录（遵守 LIFO 清理顺序）
        self.enterContext(mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}))
        root = Path(self.enterContext(tempfile.TemporaryDirectory(prefix="ni-rua-so101-uncal-"))).resolve()

        dep_doc = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        dep_doc["camera_calibration_file"] = str(root / "missing_camera_calibration.json")
        dep_path = root / "so101.local.json"
        dep_path.write_text(json.dumps(dep_doc), encoding="utf-8")

        config = Config.load(CONFIGS_ROOT / "so101_standalone.json", ignore_local=True).with_overrides({
            "backend": {"deployment": str(dep_path)},
            "features": {
                "observation_profile": "rgb_proprio_v1",
                "terminal": False,
                "grasp_memory_dir": str(root / "grasp-memory"),
            },
            "agent": {"backend": "external", "external": {"mcp_port": 0}},
            "run": {"experiment_root": str(root / "exp"), "port": 0},
        }).validate_launch()
        self.assertIsNone(backend_requirements(config))

        scene = so101_scene.scene_document("red")
        robot = _UncalibratedSceneBackend(scene=scene, config=config, run_dir=root / "run" / "private" / "robot")
        workbench = Workbench(root / "run", robot=robot, visual_review_model=None, config=config)
        self.addAsyncCleanup(workbench.close)
        await workbench.start(None)

        client = TestClient(TestServer(workbench.app()))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        origin = {"Origin": f"http://{client.host}:{client.port}"}

        state_resp = await client.get("/v1/calibration/state")
        self.assertEqual(state_resp.status, 200)
        state = await state_resp.json()
        self.assertTrue(state["available"])
        self.assertFalse(state["bundle"]["present"])

        summary_resp = await client.get("/v1/calibration/summary")
        self.assertEqual(summary_resp.status, 200)
        summary = await summary_resp.json()
        self.assertEqual(summary["status"], "needed")
        self.assertEqual(summary["reason"], "no_calibration_loaded")

        claim = await (await client.post(
            "/v1/control",
            json={"command": "claim_input", "request_id": "claim-cal", "client_id": "cal-ui"},
            headers=origin,
        )).json()
        lease = {"client_id": "cal-ui", "writer_token": claim["writer_token"]}

        torque = await (await client.post(
            "/v1/control",
            json={"command": "torque_on", "request_id": "torque-cal", **lease},
            headers=origin,
        )).json()
        self.assertEqual(torque["status"], "torque_on")

        begin = await (await client.post(
            "/v1/calibration/begin",
            json={"request_id": "begin-cal", "mode": "first_run", **lease},
            headers=origin,
        )).json()
        self.assertEqual(begin["status"], "started")

        # 轮询等待相机首帧与关节遥测就绪，避免固定长 sleep
        for _ in range(50):
            if robot.adapter.last_sample is not None and all(
                cam.latest() is not None for cam in robot.adapter.cameras.values()
            ):
                break
            await asyncio.sleep(0.05)

        s0_resp = await client.post(
            "/v1/calibration/steps/S0/start",
            json={"request_id": "s0-cal", **lease},
            headers=origin,
        )
        self.assertEqual(s0_resp.status, 200)
        s0 = await s0_resp.json()
        self.assertEqual(s0["status"], "passed", s0)
        self.assertTrue(s0["operator"]["result"]["verdict"]["passed"])
        self.assertTrue(s0["operator"]["result"]["detail"]["solver_dependencies_ok"])
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("curobo", sys.modules)


class SO101StandaloneCpuClosedLoopTests(unittest.TestCase):
    def test_standalone_cpu_external_mcp_closed_loop(self):
        # 强制无 GPU 环境且最先注册临时目录（LIFO 清理顺序）
        self.enterContext(mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}))
        root = Path(self.enterContext(tempfile.TemporaryDirectory(prefix="ni-rua-so101-cpu-e2e-"))).resolve()

        scene = so101_scene.scene_document("red")
        dep_path, _ = so101_scene.write_scene(root / "scene", scene)
        config = Config.load(CONFIGS_ROOT / "so101_standalone.json", ignore_local=True).with_overrides({
            "backend": {"deployment": str(dep_path)},
            "agent": {"backend": "external", "external": {"mcp_port": 0}},
            "features": {"terminal": False, "grasp_memory_dir": str(root / "grasp-memory")},
            "run": {"experiment_root": str(root / "exp"), "port": 0},
        }).validate_launch()
        self.assertIsNone(backend_requirements(config))

        robot = so101_scene.SceneBackend(scene=scene, config=config, run_dir=root / "run" / "private" / "robot")
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, name="so101-cpu-e2e-loop", daemon=True)
        thread.start()

        def stop_loop():
            loop.call_soon_threadsafe(loop.stop)
            thread.join(10)
            loop.close()

        self.addCleanup(stop_loop)
        workbench = Workbench(root / "run", robot=robot, visual_review_model=None, config=config)
        self.addCleanup(lambda: asyncio.run_coroutine_threadsafe(workbench.close(), loop).result(30))
        asyncio.run_coroutine_threadsafe(workbench.start(None), loop).result(30)

        client = FakeMcpClient(workbench.session.bridge.url, protocol_version="2025-11-25")
        client.handshake()

        begin = client.call_tool("run_begin", {}).result
        self.assertFalse(begin.get("isError"), begin)

        obs_res = client.call_tool("robot_observe", {})
        obs_raw = obs_res.result["content"][0]["text"]
        obs = _parse_mcp_json(obs_res)
        obs_id = obs["current_observation_id"]
        self.assertNotIn("object_table", obs)
        self.assertNotIn("scene3d", obs)
        self.assertNotIn("vggt", obs_raw.lower())
        self.assertNotIn("unavailable", obs_raw.lower())

        plan = _parse_mcp_json(client.call_tool("task_plan", {
            "plan": "Observe scene, project pixel onto table plane, reach/preview/execute target pose via CPU IK.",
            "ledger": [],
        }))
        self.assertEqual(plan["status"], "saved")

        plane = _parse_mcp_json(client.call_tool("geometry_query", {"request": {
            "kind": "pixel_plane",
            "observation_id": obs_id,
            "camera_id": "fixed",
            "pixel": [320, 240],
            "plane_origin_m": [0.0, 0.0, 0.0],
            "plane_normal": [0.0, 0.0, 1.0],
        }}))
        self.assertEqual(plane["status"], "conditional")
        self.assertIn("right", plane["value"]["gripper_relative"])

        # 基于正运动学构造可达目标位姿，由纯 CPU Pink+DAQP IK 求解并验证残差
        kin_model = so101_kinematics.load_model()
        target_T = np.asarray(so101_kinematics.fk(kin_model, [0.1, -0.9, 0.9, 0.8, 0.0]))
        stages = [{
            "id": "m1",
            "kind": "move",
            "arm": "right",
            "position_m": target_T[:3, 3].tolist(),
            "quaternion_xyzw": matrix_to_quaternion(target_T[:3, :3]),
            "tolerance_preset": "relaxed",
        }]

        reach = _parse_mcp_json(client.call_tool("robot_action", {
            "mode": "reach", "observation_id": obs_id, "stages": stages, "timeout_s": 30,
        }))
        self.assertTrue(reach["stage_results"]["m1"]["ok"])
        self.assertLessEqual(reach["stage_results"]["m1"]["ik"]["position_error_m"], 0.005)

        preview = _parse_mcp_json(client.call_tool("robot_action", {
            "mode": "preview", "observation_id": obs_id, "stages": stages, "timeout_s": 30,
        }))
        self.assertEqual(preview["status"], "planned")
        plan_id = preview["plan_id"]

        executed = _parse_mcp_json(client.call_tool("robot_action", {
            "mode": "execute", "plan_id": plan_id, "timeout_s": 30,
        }))
        self.assertEqual(executed["receipt"]["status"], "completed")
        self.assertEqual(executed["receipt"]["batch_status"], "completed")
        self.assertEqual(executed["receipt"]["completed_stage_ids"], ["m1"])

        # 核验执行后的实测末端位姿残差落入 relaxed 容差范围内
        post_obs = _parse_mcp_json(client.call_tool("robot_observe", {}))
        measured_ee = post_obs["robot_state"]["arms"]["right"]["ee_pose"]
        residual = pose_residual(pose_matrix(measured_ee), target_T)
        self.assertLessEqual(residual["position_error_m"], 0.015)
        self.assertLessEqual(residual["orientation_error_rad"], 0.2)

        # 通过控制台指令触发 save_video，验证 ffmpeg libx264 编码与双摄合流 MP4 产出
        claim = asyncio.run_coroutine_threadsafe(
            workbench.execute_control({"command": "claim_input", "request_id": "claim-e2e", "client_id": "e2e"}),
            loop,
        ).result(30)
        video = asyncio.run_coroutine_threadsafe(
            workbench.execute_control({
                "command": "save_video",
                "request_id": "video-e2e",
                "client_id": "e2e",
                "writer_token": claim["writer_token"],
            }),
            loop,
        ).result(30)
        self.assertEqual(video["status"], "complete", video)
        self.assertEqual(video["composite"]["status"], "complete", video)
        self.assertTrue(Path(video["composite"]["video_path"]).is_file())

        finish = client.call_tool("run_finish", {"summary": "CPU standalone E2E closed loop succeeded."}).result
        self.assertFalse(finish.get("isError"), finish)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("curobo", sys.modules)


class SO101PackagingCliAndActionableErrorTests(unittest.TestCase):
    def test_bundled_config_presets_and_user_deployment_fallback(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory(prefix="ni-rua-preset-"))).resolve()
        # 源码仓内预设解析应与直接传入配置文件等价
        via_preset = Config.load("so101", environ={}, ignore_local=True)
        via_alias = Config.load("so101_standalone", environ={}, ignore_local=True)
        via_path = Config.load(CONFIGS_ROOT / "so101_standalone.json", environ={}, ignore_local=True)
        self.assertEqual(via_preset, via_alias)
        self.assertEqual(via_preset, via_path)
        self.assertEqual(
            Config.load("ni-rua", environ={}, ignore_local=True),
            Config.load(CONFIGS_ROOT / "ni-rua.json", environ={}, ignore_local=True),
        )

        # 模拟无 git 源码仓的已安装 wheel 布局（仅含 core 内置 JSON）
        pkg_core = root / "site-packages" / "ni_rua" / "core"
        pkg_core.mkdir(parents=True)
        (pkg_core / "default_config.json").write_text((CONFIGS_ROOT / "ni-rua.json").read_text(encoding="utf-8"), encoding="utf-8")
        for name in ("so101_standalone.json", "so101.example.json", "so101_board.json"):
            (pkg_core / name).write_text((CONFIGS_ROOT / name).read_text(encoding="utf-8"), encoding="utf-8")

        fake_home = root / "home"
        fake_home.mkdir()
        work_dir = root / "empty-cwd"
        work_dir.mkdir()
        orig_cwd = os.getcwd()
        self.addCleanup(os.chdir, orig_cwd)
        os.chdir(work_dir)

        with mock.patch("ni_rua.core.config.__file__", str(pkg_core / "config" / "__init__.py")), \
                mock.patch("ni_rua.core.config.LOCAL_CONFIG_PATH", root / "absent.local.json"), \
                mock.patch.dict(os.environ, {"HOME": str(fake_home)}):
            wheel_so101 = Config.load("so101", environ={}, ignore_local=True)
            self.assertEqual(wheel_so101.backend.environment, "so101_real")
            self.assertEqual(wheel_so101.backend.robot_model_id, "so101")
            self.assertEqual(
                wheel_so101.backend.deployment,
                str((fake_home / ".config/ni-rua/so101.local.json").resolve()),
            )
            # 当 CWD 存在 configs/so101.local.json 时优先使用 CWD 路径
            cwd_dep = work_dir / "configs" / "so101.local.json"
            cwd_dep.parent.mkdir(parents=True)
            cwd_dep.write_text("{}", encoding="utf-8")
            wheel_so101_cwd = Config.load("so101", environ={}, ignore_local=True)
            self.assertEqual(wheel_so101_cwd.backend.deployment, str(cwd_dep.resolve()))

    def test_so101_cli_detect_ports_init_config_cameras_and_charuco_board(self):
        import io
        from contextlib import redirect_stdout
        from ni_rua.apps import so101_cli

        # 串口探测：macOS 与 Linux by-id 去重
        darwin_ports = so101_cli.detect_serial_ports(
            platform="darwin",
            glob_fn=lambda pat: ["/dev/tty.usbmodem58760431541"] if "usbmodem" in pat else [],
        )
        self.assertEqual(darwin_ports, [{"port": "/dev/tty.usbmodem58760431541", "source": "usbmodem"}])

        root = Path(self.enterContext(tempfile.TemporaryDirectory(prefix="ni-rua-so101-cli-"))).resolve()
        target_dev = root / "ttyACM0"
        target_dev.write_text("", encoding="utf-8")
        by_id_link = root / "usb-feetech"
        by_id_link.symlink_to(target_dev)
        linux_ports = so101_cli.detect_serial_ports(
            platform="linux",
            glob_fn=lambda pat: [str(by_id_link)] if "by-id" in pat else ([str(target_dev)] if "ACM" in pat else []),
        )
        self.assertEqual(linux_ports, [{"port": str(by_id_link), "source": "by-id"}])

        # init-config 非交互模式：优先 USB 相机、校验并写出合法部署配置、支持 --force 备份
        fake_inventory = {
            "platform": "darwin",
            "opencv_version": "4.13.0",
            "index_rule": "avfoundation_sorted_unique_id",
            "index_rule_error": None,
            "error": None,
            "devices": [
                {"index": 0, "path": None, "name": "FaceTime HD Camera", "model": "Apple", "transport": "bltn", "unique_ids": ["builtin-uuid"]},
                {"index": 1, "path": None, "name": "USB Wrist Cam", "model": "UVC", "transport": "usb", "unique_ids": ["0x11000001"]},
                {"index": 2, "path": None, "name": "USB Fixed Cam", "model": "UVC", "transport": "usb", "unique_ids": ["0x12000002"]},
            ],
        }
        out_path = root / "so101.local.json"
        parser = so101_cli.build_parser()

        # 无可用串口且未传 --port 时应快速报错
        with self.assertRaises(SystemExit) as caught:
            so101_cli.cmd_init_config(
                parser.parse_args(["init-config", "--non-interactive", "--output", str(out_path)]),
                list_devices=lambda: fake_inventory,
                list_ports=lambda: [],
            )
        self.assertIn("so101_serial_port_not_found", str(caught.exception))

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = so101_cli.cmd_init_config(
                parser.parse_args(["init-config", "--non-interactive", "--output", str(out_path), "--json"]),
                list_devices=lambda: fake_inventory,
                list_ports=lambda: darwin_ports,
            )
        self.assertEqual(rc, 0)
        summary = json.loads(buf.getvalue())
        self.assertEqual(summary["port"], "/dev/tty.usbmodem58760431541")
        loaded = load_deployment(out_path)
        self.assertEqual(loaded["cameras"]["wrist"]["unique_id"], "0x11000001")
        self.assertEqual(loaded["cameras"]["fixed"]["unique_id"], "0x12000002")

        # 未加 --force 拒绝覆盖；加 --force 后保存 .bak 备份
        with self.assertRaises(SystemExit) as exists_err:
            so101_cli.cmd_init_config(
                parser.parse_args(["init-config", "--non-interactive", "--output", str(out_path)]),
                list_devices=lambda: fake_inventory,
                list_ports=lambda: darwin_ports,
            )
        self.assertIn("so101_config_exists", str(exists_err.exception))

        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            rc2 = so101_cli.cmd_init_config(
                parser.parse_args(["init-config", "--non-interactive", "--force", "--output", str(out_path), "--json"]),
                list_devices=lambda: fake_inventory,
                list_ports=lambda: darwin_ports,
            )
        self.assertEqual(rc2, 0)
        self.assertTrue(Path(json.loads(buf2.getvalue())["backup"]).is_file())

        # cameras 子命令核验部署绑定（匹配成功返回 0，缺失返回 1）
        cam_out = io.StringIO()
        with mock.patch.object(so101_cli, "list_video_devices", return_value=fake_inventory), redirect_stdout(cam_out):
            self.assertEqual(so101_cli.main(["cameras", "--deployment", str(out_path), "--json"]), 0)
        missing_inventory = {**fake_inventory, "devices": []}
        with mock.patch.object(so101_cli, "list_video_devices", return_value=missing_inventory), redirect_stdout(io.StringIO()):
            self.assertEqual(so101_cli.main(["cameras", "--deployment", str(out_path)]), 1)

        # charuco-board 子命令可直接使用内置 so101_board 预设渲染 PNG + PDF
        board_dir = root / "board"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(
                so101_cli.main(["charuco-board", "--board-config", "so101_board", "--px-per-mm", "4", "--output-dir", str(board_dir)]),
                0,
            )
        self.assertTrue((board_dir / "so101_charuco_board.png").is_file())
        self.assertTrue((board_dir / "so101_charuco_board.pdf").is_file())

    def test_so101_cold_start_actionable_errors(self):
        from ni_rua.robot.backends.so101 import contract
        from ni_rua.robot.backends.so101.driver import OpenCVCameraSource

        root = Path(self.enterContext(tempfile.TemporaryDirectory(prefix="ni-rua-so101-err-"))).resolve()
        dep_doc = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        dep_doc["camera_calibration_file"] = str(root / "missing_camera_calibration.json")
        dep_path = root / "so101.local.json"
        dep_path.write_text(json.dumps(dep_doc), encoding="utf-8")

        # 1. 冷启动缺少 camera_calibration.json：明确提示 --observation-profile rgb_proprio_v1 与 /calibration
        cfg = Config.from_dict({
            "backend": {"environment": "so101_real", "robot_model_id": "so101", "deployment": str(dep_path)},
            "features": {"observation_profile": "rgb_proprio_calibrated_v1"},
        })
        with self.assertRaises(ValueError) as cal_err:
            RobotBackend(config=cfg, run_dir=root / "run")
        msg = str(cal_err.exception)
        self.assertIn("so101_camera_calibration_missing", msg)
        self.assertIn("--observation-profile rgb_proprio_v1", msg)
        self.assertIn("/calibration", msg)

        # 2. 12V 舵机断电与部分电机 ID 失联：明确提示 12V 供电/断电排查
        deployment = load_deployment(dep_path)
        with self.assertRaises(PreflightError) as power_err:
            run_preflight(FakeBus(ping=None), calibrated=lambda: True, deployment=deployment,
                          limits_rad=LIMITS, to_rad=lambda _n, d: math.radians(d))
        self.assertEqual(power_err.exception.code, "servo_power_off")
        self.assertIn("12V", power_err.exception.detail)
        self.assertIn("1,2,3,4,5,6", power_err.exception.detail)

        with self.assertRaises(PreflightError) as missing_err:
            run_preflight(FakeBus(ping={1: 777, 2: 777}), calibrated=lambda: True, deployment=deployment,
                          limits_rad=LIMITS, to_rad=lambda _n, d: math.radians(d))
        self.assertEqual(missing_err.exception.code, "servo_missing")
        self.assertIn("12V", missing_err.exception.detail)

        # 3. macOS 相机打开失败：明确提示终端 Camera 权限与 ni-rua-so101 cameras
        cam = OpenCVCameraSource("wrist", deployment["cameras"]["wrist"], Clock())
        cam.binding = {"mode": "index", "opened": 0, "api": "avfoundation"}
        fake_cap = mock.MagicMock()
        fake_cap.isOpened.return_value = False
        with mock.patch("ni_rua.robot.backends.so101.driver.sys.platform", "darwin"), \
                mock.patch("cv2.VideoCapture", return_value=fake_cap):
            with self.assertRaises(RuntimeError) as cam_err:
                cam.start()
        self.assertIn("so101_camera_unavailable:wrist", str(cam_err.exception))
        self.assertIn("Camera permission", str(cam_err.exception))
        self.assertIn("ni-rua-so101 cameras", str(cam_err.exception))

        # 4. 未测量 empty_close_aperture_m 时使用 grip_guard_preset：提示在 so101.local.json 实测配置
        with self.assertRaises(ValueError) as guard_err:
            contract.validate_grip_guard({"grip_guard_preset": "non_empty"}, deployment)
        self.assertIn("empty_close_aperture_m", str(guard_err.exception))
        self.assertIn("so101.local.json", str(guard_err.exception))


if __name__ == "__main__":
    unittest.main()
