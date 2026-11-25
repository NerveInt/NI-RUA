"""Merged cortex tests:
- tests/test_self_briefing.py: Retirement contract for ``features.self_briefing`` and historical v1 archive loading.
- tests/unit/test_startup_inline_grounding.py: 首轮内联场景接地与默认关闭 self_briefing 单元测试。
"""

from __future__ import annotations

NI_RUA_TEST_TIER = "component"

import asyncio
import base64
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
import cv2
import numpy as np
from PIL import Image
from ni_rua.apps.evaluation.audit import audit
from ni_rua.apps.gateway.workbench import Workbench
from ni_rua.core.config import Config
from ni_rua.core.observation_policy import ACTOR_PROFILE
from ni_rua.core.presentation.actor_presentation import ActorPresentation, metadata_findings
from ni_rua.core.storage.store import Store
from ni_rua.cortex.agents.codex.session import CodexSession
from ni_rua.cortex.context import startup_grounding
from ni_rua.cortex.context.public_task_rules import actor_task_rules_text, public_task_rules
from ni_rua.cortex.session.codex_evidence_agent import BASE_INSTRUCTIONS as EVIDENCE_BASE_INSTRUCTIONS, CodexEvidenceAgent
from ni_rua.perception.external import obs3d
from ni_rua.robot.backends.robodojo.robodojo import FRAME_ID, TCP_CONFIG_ID
from ni_rua.runtime.harness import Harness
from support.paths import isolate_home


def setUpModule():
    isolate_home()   # a fresh Workbench opens the grasp memory store at its '~' default


# ==== tests/test_self_briefing.py ====

TASK = "pour_balls_into_vase"


def saved_observation(store, identifier):
    state = {"frame_id": "env_origin", "sim_step": 0, "step_domain": "adapter_control_step", "arms": {}}
    return {
        **store.identity,
        "schema_version": "0.1",
        "world_kind": "observed",
        "observation_id": identifier,
        "sim_step": 0,
        "step_domain": "adapter_control_step",
        "robot_state_ref": store.put_artifact(json.dumps(state).encode(), "robot_state", "application/json"),
        "task_instruction": "Fixture native instruction.",
        "cameras": [],
    }


class SelfBriefingRetirementTests(unittest.IsolatedAsyncioTestCase):
    def test_module_and_config_flag_are_retired(self):
        self.assertIsNone(importlib.util.find_spec("ni_rua.cortex.context.self_briefing"))
        default_cfg = Config()
        self.assertFalse(default_cfg.features.self_briefing)
        self.assertNotIn("self_briefing", default_cfg.to_dict()["features"])

        for raw_flag in (True, False):
            with self.subTest(raw_flag=raw_flag):
                migrated = Config.from_dict(
                    {"version": 1, "run": {"task": TASK}, "features": {"self_briefing": raw_flag}}
                )
                self.assertEqual(migrated.version, 2)
                self.assertFalse(migrated.features.self_briefing)
                self.assertNotIn("self_briefing", migrated.to_dict()["features"])

    async def test_workbench_startup_never_runs_pre_action_briefing_turn(self):
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        robot = SimpleNamespace(close=Mock(), execute=Mock(), reset=Mock(), evaluate=Mock())
        config = Config.from_dict({"version": 1, "run": {"task": TASK}, "features": {"self_briefing": True}})
        workbench = Workbench(Path(temporary) / "run", robot=robot, visual_review_model=None, config=config)
        self.addAsyncCleanup(workbench.close)
        initial = saved_observation(workbench.store, "initial-observation")
        done = threading.Event()
        done.set()
        harness = SimpleNamespace(
            accepting=True,
            active_action=None,
            states={},
            action_count=0,
            latest=initial,
            home_check=None,
            capabilities={},
            episode_done=threading.Event(),
            episode_terminal=None,
            action_budget=lambda: dict(native_steps_used=0, native_steps_limit=None, native_steps_remaining=None),
            native_action_count=0,
            started_at=time.monotonic(),
            last_native_progress_at=time.monotonic(),
            scene=SimpleNamespace(capabilities=lambda: {}),
            start=Mock(),
        )
        for field in (
            "tool_registry",
            "config",
            "disabled_tools",
            "observation_profile",
            "motion_review",
            "enable_wait",
            "vision_compare",
            "visual_reviewer",
            "action_policy",
            "programs",
        ):
            setattr(harness, field, getattr(workbench.harness, field))
        harness.request_stop = lambda: setattr(harness, "accepting", False)
        session = SimpleNamespace(
            thread_id="fixture-thread",
            turn_id="first-turn",
            state="completed",
            turn_done=done,
            model="gpt-6-astra",
            provider="fixture",
            effort="high",
            price_card=None,
            cli_executable=Path("/fixture/codex"),
            command_prefix=["fixture-sandbox"],
            actor_dir=workbench.root,
            socket_path=workbench.root / "fixture.sock",
            env={},
            process=SimpleNamespace(pid=-1),
            start=Mock(return_value="fixture-thread"),
            run_turn=Mock(),
            close=Mock(),
            model_proxy=SimpleNamespace(limit_notified=False, request_count=1),
        )
        workbench.harness, workbench.session = harness, session
        terminal = SimpleNamespace(
            start=AsyncMock(), close=AsyncMock(), session_id="fixture-terminal", process=SimpleNamespace(pid=-1)
        )
        evidence_helper = Mock(side_effect=AssertionError("retired self_briefing must never instantiate helper"))
        with (
            patch("ni_rua.apps.gateway.workbench.Terminal", return_value=terminal),
            patch("ni_rua.cortex.session.codex_evidence_agent.CodexEvidenceAgent", evidence_helper),
        ):
            await workbench.start("unused", benchmark=True)

        evidence_helper.assert_not_called()
        session.run_turn.assert_called_once()
        self.assertIsNone(workbench.self_briefing)
        prompt = session.run_turn.call_args.args[0]
        self.assertNotIn("Operating notes", prompt)
        manifest = json.loads((workbench.root / "manifest.json").read_text())
        self.assertNotIn("self_briefing", manifest)
        events = [event["type"] for event in workbench.store.events()]
        self.assertFalse([ev for ev in events if ev.startswith("self_briefing.")])

    def test_evidence_agent_defaults_are_unchanged(self):
        agent = CodexEvidenceAgent(SimpleNamespace(store=None))
        self.assertEqual(
            (agent.base_instructions, agent.developer_instructions, agent.source),
            (EVIDENCE_BASE_INSTRUCTIONS, "Use only the supplied evidence packet.", "sol_visual_review"),
        )

# ==== tests/unit/test_startup_inline_grounding.py ====

TABLE_Z = 0.765
CAMERA_K = [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
T_HEAD = [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.35], [0.0, 0.0, 0.0, 1.0]]


def _translation(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> list[list[float]]:
    return [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, y], [0.0, 0.0, 1.0, z], [0.0, 0.0, 0.0, 1.0]]


def _synthetic_npz_bytes() -> bytes:
    """构造最小合法 scene3d 压缩点云字节流。"""
    xyz = np.array(
        [[0.0, 0.0, TABLE_Z], [0.05, 0.0, TABLE_Z], [0.08, 0.02, TABLE_Z + 0.04]],
        dtype=np.float32,
    )
    conf = np.array([200, 210, 230], dtype=np.uint8)
    uv = np.array([[320, 240], [345, 240], [360, 230]], dtype=np.uint16)
    count = np.ones(len(xyz), dtype=np.uint16)
    robot = np.zeros(len(xyz), dtype=bool)
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        xyz=xyz,
        conf=conf,
        uv=uv,
        count=count,
        robot=robot,
        support_z=np.float64(TABLE_Z),
        voxel_m=np.float64(0.01),
        image_size=np.asarray([640, 480], dtype=np.int32),
    )
    return buf.getvalue()


def _fake_object_row(centre: tuple[float, float, float] = (0.08, 0.02, TABLE_Z + 0.02)) -> dict:
    """构造带轮廓与标签坐标的合成 3D 物体记录，支持触发 overlay 绘制与 grasp_hints。"""
    u, v = 360, 230
    return dict(
        centre_m=list(centre),
        top_z_m=centre[2] + 0.02,
        bottom_z_m=TABLE_Z,
        length_m=0.09,
        width_m=0.03,
        yaw_deg=25.0,
        axis_tilt_deg=89.0,
        elongation=3.0,
        support="table",
        flags=[],
        quality="ok",
        label_px=[u, v],
        contour_px=[[u - 12, v - 6], [u + 12, v - 6], [u + 12, v + 6], [u - 12, v + 6]],
        mask_px=240,
        n_points=160,
        iou_pred=0.92,
        conf_rank_p50=0.6,
        contact_gap_m=0.0,
        cluster_frac=1.0,
    )


class FakeObs3dServer:
    """本地轻量 HTTP 假服务，模拟 VGGT /object_table 与 /points 端点。"""

    def __init__(self, *, status: str = "ok") -> None:
        self.status = status
        self.requests: list[tuple[str, dict]] = []
        self.objects = [_fake_object_row()]
        npz_bytes = _synthetic_npz_bytes()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append((self.path, body))
                if owner.status != "ok":
                    answer, code = {"status": "error", "error": owner.status}, 503
                elif self.path == "/object_table":
                    answer, code = (
                        dict(
                            status="ok",
                            version="fake-ot-v5",
                            objects=deepcopy(owner.objects),
                            requested=[],
                            omitted=0,
                            labels=[],
                            support_z_m=body["support_z_m"],
                            dropped={"robot": 0},
                            n_prompts=4,
                            timing={"total_s": 0.01},
                            support_fit={"tilt_deg": 0.2},
                            scale=1.0,
                        ),
                        200,
                    )
                elif self.path == "/points":
                    answer, code = (
                        dict(
                            status="ok",
                            version="fake-points-v1",
                            n_points=3,
                            n_above_support=1,
                            n_robot=0,
                            voxel_m=body["voxel_m"],
                            support_z_m=body["support_z_m"],
                            bbox_m=[[-0.2, -0.2, TABLE_Z], [0.2, 0.2, TABLE_Z + 0.05]],
                            bbox_above_support_m=[[0.05, 0.0, TABLE_Z], [0.10, 0.04, TABLE_Z + 0.04]],
                            npz_b64=base64.b64encode(npz_bytes).decode(),
                            npz_bytes=len(npz_bytes),
                            timing={"total_s": 0.01},
                            support_fit={"tilt_deg": 0.2},
                            scale=1.0,
                        ),
                        200,
                    )
                else:
                    answer, code = {"status": "error", "error": "unknown_path"}, 404
                payload = json.dumps(answer).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class ThreeCameraDualArmRobot:
    """三相机（cam_head + 左右腕部相机）双臂标定测试机器人夹具。"""

    def __init__(self, root: Path, instruction: str = "Stack all bowls onto the target coaster.") -> None:
        self.root = Path(root)
        self.instruction = instruction
        self.observations = 0
        self.executions = 0
        self.closed = False
        self.image_paths: dict[str, Path] = {}
        rng = np.random.default_rng(7)
        for idx, cam_id in enumerate(("cam_head", "cam_left_wrist", "cam_right_wrist")):
            img = rng.integers(30 + idx * 20, 200, size=(480, 640, 3), dtype=np.uint8)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            path = self.root / f"{cam_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(cv2.imencode(".png", img)[1].tobytes())
            self.image_paths[cam_id] = path

    def start(self) -> dict:
        return dict(
            data_origin="synthetic_fixture",
            robot_model_id="arx-x5",
            frame_id=FRAME_ID,
            tcp_config_id=TCP_CONFIG_ID,
            path_kinds=["backend_free"],
            max_waypoints=4,
            dual_arm=True,
            gripper_after_motion=True,
        )

    def observe(self) -> dict:
        self.observations += 1
        frames, calibration, poses = [], {}, {}
        specs = (
            ("cam_head", None, T_HEAD),
            ("cam_left_wrist", "left", _translation(0.22, 0.18, 0.92)),
            ("cam_right_wrist", "right", _translation(0.22, -0.18, 0.92)),
        )
        for cam_id, arm, transform in specs:
            frame_id = f"{cam_id}-{self.observations}"
            frames.append(
                dict(
                    camera_id=cam_id,
                    frame_id=frame_id,
                    png_path=str(self.image_paths[cam_id]),
                    width=640,
                    height=480,
                )
            )
            calibration[cam_id] = dict(
                K=CAMERA_K,
                distortion=[0.0, 0.0, 0.0, 0.0, 0.0],
                distortion_model="opencv",
                image_size=[640, 480],
                optical_axes="+x right,+y down,+z forward",
            )
            poses[cam_id] = dict(
                frame_id=frame_id,
                T_world_camera=transform,
                source_kind="robot_kinematics" if arm else "simulator_camera_transform",
            )
            if arm:
                base_y = 0.25 if arm == "left" else -0.25
                calibration[cam_id].update(
                    arm=arm,
                    link_name="link6",
                    T_link_camera=_translation(0.04, 0.0, 0.03),
                    T_world_base=_translation(0.0, base_y, TABLE_Z),
                )
                poses[cam_id].update(
                    arm=arm,
                    link_name="link6",
                    T_world_link=_translation(0.20, base_y * 0.6, 0.90),
                    T_world_base=_translation(0.0, base_y, TABLE_Z),
                )
            else:
                calibration[cam_id]["T_world_camera"] = T_HEAD

        arms = {}
        for arm, y_pos in (("left", 0.20), ("right", -0.20)):
            arms[arm] = dict(
                q_rad=[0.0, 0.4, -0.4, 0.0, 0.0, 0.0],
                ee_pose=dict(
                    position_m=[0.18, y_pos, 0.92],
                    quaternion_xyzw=[0.0, 0.70710678, 0.0, 0.70710678],
                    frame_id=FRAME_ID,
                    tcp_config_id=TCP_CONFIG_ID,
                    source_kind="measurement",
                ),
                gripper=dict(
                    goal_normalized=1.0,
                    goal_source_kind="command_state",
                    position_m=[0.044, 0.044],
                    position_source_kind="measurement",
                ),
            )
        return dict(
            data_origin="synthetic_fixture",
            world_kind="observed",
            observation_profile=ACTOR_PROFILE,
            instruction=self.instruction,
            sim_step=self.executions,
            step_domain="adapter_control_step",
            sim_time_s=0.0,
            captured_at="2026-11-11T08:00:00Z",
            cameras=frames,
            camera_calibration=dict(reference_frame_id=FRAME_ID, cameras=calibration),
            camera_pose_sample=dict(
                reference_frame_id=FRAME_ID,
                sim_step=self.executions,
                step_domain="adapter_control_step",
                cameras=poses,
            ),
            robot_model=dict(base_link="base_link", joints=[]),
            robot_state=dict(
                data_origin="synthetic_fixture",
                source_kind="measurement",
                frame_id=FRAME_ID,
                tcp_config_id=TCP_CONFIG_ID,
                sim_step=self.executions,
                step_domain="adapter_control_step",
                arms=arms,
            ),
        )

    def execute(self, action: dict, progress_callback=None, cancel_flag=None) -> dict:
        self.executions += 1
        if progress_callback:
            progress_callback({"phase": "started", "sim_step": self.executions, "data_origin": "synthetic_fixture"})
        terminal_obs = self.observe()
        return dict(
            status="completed",
            reason="synthetic_fixture",
            data_origin="synthetic_fixture",
            executed_steps=1,
            position_error_m=0.001,
            orientation_error_rad=0.01,
            terminal_observation=terminal_obs,
            planned_trace={"data_origin": "synthetic_fixture", "targets": action.get("path")},
            dispatched_trace={"data_origin": "synthetic_fixture", "steps": 1},
            actual_trace=[{"data_origin": "synthetic_fixture", "measured_fixture_state": terminal_obs["robot_state"]}],
        )

    def close(self) -> dict:
        self.closed = True
        return {"process_exited": True}


class FakeSessionClient:
    """记录 turn_start 真实调用入参并同步写入 codex-raw.jsonl 的轻量假客户端。"""

    def __init__(self, store: Store | None = None) -> None:
        self.store = store
        self.turn_calls: list[tuple[str, list[dict], dict | None]] = []

    def turn_start(self, thread_id: str, input_items: list[dict], params: dict | None = None):
        self.turn_calls.append((thread_id, deepcopy(input_items), deepcopy(params)))
        turn_id = f"turn-{len(self.turn_calls)}"
        if self.store is not None:
            self.store.raw_codex(
                "sent",
                dict(
                    method="turn/start",
                    params=dict(threadId=thread_id, input=deepcopy(input_items), **(params or {})),
                ),
            )
            self.store.raw_codex(
                "received",
                dict(
                    method="turn/started",
                    params=dict(threadId=thread_id, turn=dict(id=turn_id, status="inProgress")),
                ),
            )
        return SimpleNamespace(turn=SimpleNamespace(id=turn_id))


def _attach_fake_codex_runtime(workbench: Workbench) -> FakeSessionClient:
    """将 Workbench.session 绑定至离线假客户端，保留真实 CodexSession.run_turn 与呈现脱敏管道。"""
    fake_client = FakeSessionClient(store=workbench.store)

    def _start_session(tools, instructions, resume_thread_id=None):
        thread_id = resume_thread_id or "thread-startup-1"
        workbench.session.client = fake_client
        workbench.session.thread_id = thread_id
        workbench.session.state = "idle"
        workbench.session.provider = "test-provider"
        workbench.session.cli_executable = "/usr/bin/codex"
        workbench.session.process = SimpleNamespace(pid=43210)
        workbench.session.model_proxy = SimpleNamespace(transport_id="transport-1", image_context_epoch=1)
        workbench.store.set_state(
            "runtime",
            "codex",
            {"thread_id": thread_id, "app_server_instance_id": "app-instance-1"},
        )
        return thread_id

    workbench.session.start = _start_session
    workbench.session.close = lambda: None
    return fake_client


class StartupInlineGroundingTests(unittest.TestCase):
    """验证首轮内联接地、单张概览图投递、平稳回退与跨回合状态一致性。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _make_workbench(self, run_name: str, overrides: dict | None = None) -> tuple[Workbench, ThreeCameraDualArmRobot]:
        run_dir = self.root / run_name
        robot = ThreeCameraDualArmRobot(self.root / f"{run_name}-assets")
        base_overrides = {
            "run": {"task": "stack_bowls", "benchmark": True},
            "features": {
                "terminal": False,
                "motion_review": False,
                "grasp_memory_dir": str(self.root / f"{run_name}-mem"),
            },
        }
        if overrides:
            for section, values in overrides.items():
                base_overrides.setdefault(section, {}).update(values)
        config = Config().with_overrides(base_overrides)
        workbench = Workbench(run_dir, robot=robot, config=config)
        self.addCleanup(self._close_workbench_stores, workbench)
        return workbench, robot

    @staticmethod
    def _close_workbench_stores(workbench: Workbench) -> None:
        if getattr(workbench, "grasp_memory_store", None) is not None:
            workbench.grasp_memory_store.close()
        if getattr(workbench, "store", None) is not None:
            workbench.store.close()
        for attr in ("run_lock", "lock", "lock_file"):
            handle = getattr(workbench, attr, None)
            if handle is not None and hasattr(handle, "close"):
                handle.close()

    def test_default_config_disables_self_briefing_and_enables_startup_grounding(self) -> None:
        """默认配置已退役阻塞式 self_briefing 并开启首轮内联接地。"""
        import importlib.util

        self.assertIsNone(importlib.util.find_spec("ni_rua.cortex.context.self_briefing"))
        default_cfg = Config()
        self.assertFalse(default_cfg.features.self_briefing)
        self.assertTrue(default_cfg.features.startup_inline_grounding)

        with patch("ni_rua.core.config.LOCAL_CONFIG_PATH", self.root / "nonexistent.local.json"):
            loaded_cfg = Config.load(environ={})
        self.assertFalse(loaded_cfg.features.self_briefing)
        self.assertTrue(loaded_cfg.features.startup_inline_grounding)

    def test_benchmark_startup_inlines_3d_state_and_single_overlay_image(self) -> None:
        """外部 3D 服务可用时，首轮内联结构化状态 + 3D 表，且仅附带 1 张 cam_head ID 标注图。"""
        server = FakeObs3dServer(status="ok")
        self.addCleanup(server.close)

        workbench, robot = self._make_workbench(
            "run-3d-ok",
            {
                "features": {
                    "external_perception": True,
                    "vggt_endpoint": server.url,
                    "vggt_support_z_m": TABLE_Z,
                    "held_perception": False,
                }
            },
        )
        fake_client = _attach_fake_codex_runtime(workbench)
        asyncio.run(workbench.start(None, benchmark=True))

        # 仅在 harness.start() 采集 1 次初始观测，首轮无额外仿真步消耗
        self.assertEqual(robot.observations, 1)
        self.assertIsNone(workbench.self_briefing)
        self.assertIsNotNone(workbench.startup_grounding)
        self.assertEqual(workbench.startup_grounding["image_kind"], "obs3d_overlay")
        self.assertEqual(workbench.startup_grounding["image_count"], 1)
        self.assertEqual(workbench.startup_grounding["overview_camera_id"], "cam_head")
        self.assertTrue(workbench.startup_grounding["obs3d_included"])

        # 验证首轮 turn_start 收到 [文本, 单张 cam_head 标注概览图]，绝无多余腕部图或重复裸原图
        self.assertEqual(len(fake_client.turn_calls), 1)
        _, input_items, _ = fake_client.turn_calls[0]
        self.assertEqual([item["type"] for item in input_items], ["text", "image"])
        self.assertTrue(input_items[1]["url"].startswith("data:image/"))

        prompt_text = input_items[0]["text"]
        # 严格校验信息脱敏边界：零原始 hash、零内部长 ID 泄露，obs_1 与 artifact:ref_N 可逆还原
        self.assertEqual(metadata_findings(prompt_text), [])
        self.assertNotIn(workbench.initial_observation_id, prompt_text)
        self.assertEqual(workbench.harness.actor_output.restore("obs_1"), workbench.initial_observation_id)

        self.assertIn("Initial scene grounding (inline obs_1; 1 overview image attached: cam_head with 3D object IDs):", prompt_text)
        self.assertIn('"grasp_pose"', prompt_text)
        self.assertIn('"camera_calibration"', prompt_text)
        self.assertIn('"object_table"', prompt_text)
        self.assertIn('"grasp_hints"', prompt_text)
        self.assertIn('"scene3d"', prompt_text)
        self.assertIn("artifact:ref_", prompt_text)
        self.assertIn('camera_id="cam_left_wrist"', prompt_text)
        self.assertIn('camera_id="cam_right_wrist"', prompt_text)

        # 验证 obs3d_overlay 模式下 delivery 审计记录不误记未投递的裸原图，delivery_channel 为 startup_turn
        deliveries = workbench.store.records("delivery")
        self.assertEqual(len(deliveries), 1)
        self.assertEqual(deliveries[0]["images"], [])
        self.assertEqual(deliveries[0]["delivery_channel"], "startup_turn")

        manifest = json.loads((workbench.root / "manifest.json").read_text())
        self.assertIn("startup_grounding", manifest)
        self.assertEqual(manifest["startup_grounding"]["image_kind"], "obs3d_overlay")
        self.assertIn("cortex/context/startup_grounding.py", manifest["source_sha256"])

    def test_fallback_to_single_raw_cam_head_when_external_perception_off_or_unavailable(self) -> None:
        """外部 3D 服务关闭或报错时，平稳回退为附带 1 张 cam_head 原图。"""
        # 场景 A：external_perception = False
        wb_off, _ = self._make_workbench(
            "run-3d-off",
            {"features": {"external_perception": False}},
        )
        client_off = _attach_fake_codex_runtime(wb_off)
        asyncio.run(wb_off.start(None, benchmark=True))

        _, items_off, _ = client_off.turn_calls[0]
        self.assertEqual([item["type"] for item in items_off], ["text", "image"])
        self.assertTrue(items_off[1]["url"].startswith("data:image/"))
        self.assertEqual(wb_off.startup_grounding["image_kind"], "raw_rgb")
        self.assertFalse(wb_off.startup_grounding["obs3d_included"])
        self.assertIn("1 overview image attached: cam_head raw RGB", items_off[0]["text"])
        self.assertNotIn("3D scene perception", items_off[0]["text"])
        self.assertEqual(metadata_findings(items_off[0]["text"]), [])

        # 场景 B：external_perception = True 但服务端返回 503 错误（无 overlay 图）
        down_server = FakeObs3dServer(status="gpu_oom")
        self.addCleanup(down_server.close)
        wb_err, _ = self._make_workbench(
            "run-3d-down",
            {
                "features": {
                    "external_perception": True,
                    "vggt_endpoint": down_server.url,
                    "held_perception": False,
                }
            },
        )
        client_err = _attach_fake_codex_runtime(wb_err)
        asyncio.run(wb_err.start(None, benchmark=True))

        _, items_err, _ = client_err.turn_calls[0]
        self.assertEqual([item["type"] for item in items_err], ["text", "image"])
        self.assertEqual(wb_err.startup_grounding["image_kind"], "raw_rgb")
        self.assertTrue(wb_err.startup_grounding["obs3d_included"])
        self.assertIn("unavailable:", items_err[0]["text"])
        self.assertEqual(metadata_findings(items_err[0]["text"]), [])

    def test_cross_turn_state_sync_metadata_delivery_virtual_field_and_ledger(self) -> None:
        """验证首轮内联 obs_1 后，后续回合 MetadataDelivery、按需腕部看图、虚拟空间场与动作账本无缝衔接。"""
        server = FakeObs3dServer(status="ok")
        self.addCleanup(server.close)

        workbench, robot = self._make_workbench(
            "run-continuity",
            {
                "images": {"feedback_policy": "adaptive"},
                "features": {
                    "external_perception": True,
                    "vggt_endpoint": server.url,
                    "vggt_support_z_m": TABLE_Z,
                    "held_perception": False,
                },
            },
        )
        _attach_fake_codex_runtime(workbench)
        asyncio.run(workbench.start(None, benchmark=True))
        harness: Harness = workbench.harness

        # 1. 启动完毕即已同步预热 _virtual_spatial_field 与 _obs3d_memo，且未把未发送的裸原图误记入 _last_delivered_views
        self.assertIsNotNone(harness._virtual_spatial_field)
        self.assertEqual(harness._obs3d_memo[0], workbench.initial_observation_id)
        self.assertIn(workbench.initial_observation_id, harness._obs3d_history)
        self.assertEqual(harness._last_delivered_views, {})

        # 2. 按提示按需读取左腕相机：直接读取已保存的 obs_1 腕部视图，不推进仿真步（observations 保持为 1）
        wrist_reply = harness.handle_tool(
            {
                "threadId": harness.thread_id,
                "turnId": "turn-1",
                "callId": "turn1-wrist-check",
                "tool": "robot_observe",
                "arguments": {"camera_id": "cam_left_wrist"},
            }
        )
        self.assertTrue(wrist_reply["success"])
        self.assertEqual(robot.observations, 1)
        wrist_obs = json.loads(wrist_reply["contentItems"][0]["text"])
        self.assertEqual(wrist_obs["current_observation_id"], "obs_1")
        # 静态相机标定已在首轮内联时登记至真实会话 epoch，此处自动输出 static_refs -> obs_1
        self.assertEqual(wrist_obs["metadata_delivery"]["mode"], "unchanged_static_fields_referenced")
        self.assertEqual(
            wrist_obs["metadata_delivery"]["static_refs"]["camera_calibration"]["observation_id"],
            "obs_1",
        )

        # 3. 模拟首动作执行后产生新观测 obs_2（目标物体位移 6cm）：验证直接基于首轮缓存的 obs_1 产出 object_state_delta 并写入动作账本
        server.objects = [_fake_object_row(centre=(0.14, 0.02, TABLE_Z + 0.02))]
        obs2_bundle = harness.observe({"tool_call_id": "post-action-obs2", "thread_id": harness.thread_id})
        self.assertEqual(robot.observations, 2)

        action_parts = obs3d.action_attachments(
            harness,
            obs2_bundle,
            {"tool_call_id": "action-1", "thread_id": harness.thread_id},
            before_observation_id=workbench.initial_observation_id,
        )
        presented_action_obs3d = harness.actor_output.present(
            startup_grounding.ToolResponse([action_parts[0]], kind="observation"),
            tool="robot_action",
        )
        obs3d_after = json.loads(presented_action_obs3d["contentItems"][0]["text"])
        self.assertIn("object_state_delta", obs3d_after)
        self.assertEqual(len(obs3d_after["object_state_delta"]["moved"]), 1)
        self.assertIsInstance(harness._action_state_ledger, dict)
        self.assertEqual(
            harness._action_state_ledger["last_object_delta"]["before_observation_id"],
            workbench.initial_observation_id,
        )
        self.assertEqual(
            harness._action_state_ledger["last_object_delta"]["after_observation_id"],
            obs2_bundle["observation_id"],
        )

    def test_non_benchmark_and_error_restoration_isolation(self) -> None:
        """非 benchmark 模式不注入首轮内联接地，且 run_initial_turn 异常时严格恢复 turn_start。"""
        wb_smoke, _ = self._make_workbench(
            "run-smoke",
            {"run": {"benchmark": False}, "features": {"external_perception": False}},
        )
        client_smoke = _attach_fake_codex_runtime(wb_smoke)
        asyncio.run(wb_smoke.start("Inspect joint state only.", benchmark=False))

        self.assertIsNone(wb_smoke.startup_grounding)
        _, smoke_items, _ = client_smoke.turn_calls[0]
        self.assertEqual(smoke_items, [{"type": "text", "text": "Inspect joint state only."}])

        # 验证旧归档（不含 startup_inline_grounding 字段）在 replay 模式下默认保持 startup_inline_grounding = False
        legacy_manifest_path = wb_smoke.root / "manifest.json"
        legacy_manifest = json.loads(legacy_manifest_path.read_text())
        legacy_manifest["effective_config"]["features"].pop("startup_inline_grounding", None)
        legacy_manifest_path.write_text(json.dumps(legacy_manifest, indent=2))
        self._close_workbench_stores(wb_smoke)
        wb_replay = Workbench(wb_smoke.root, replay=True)
        self.addCleanup(self._close_workbench_stores, wb_replay)
        self.assertFalse(wb_replay.config.features.startup_inline_grounding)

        # 验证 startup_grounding 已移除 codex 专用补丁入口，且 CodexSession.run_turn 直接接受 images
        self.assertFalse(hasattr(startup_grounding, "supports_startup_grounding"))
        self.assertFalse(hasattr(startup_grounding, "run_initial_turn"))
        store = Store(self.root / "err-store", {"run_id": "r", "episode_id": "e", "env_instance_id": "env", "env_generation": 1})
        self.addCleanup(store.close)
        session = CodexSession.__new__(CodexSession)
        session.store = store
        session.presentation = ActorPresentation(store)
        session.projection_lock = threading.Lock()
        session.turn_done = threading.Event()
        session.state = "idle"
        session.turn_id = None
        session.turn_result = None
        session.turn_revision = 0
        session.thread_id = "thread-err"
        session.model = "gpt-6.1-sol"
        session.effort = "high"
        session._disconnected = Mock()

        boom = Mock(side_effect=RuntimeError("rpc_closed"))
        session.client = SimpleNamespace(turn_start=boom)
        with self.assertRaisesRegex(RuntimeError, "rpc_closed"):
            session.run_turn(
                "Test prompt",
                images=({"type": "image", "url": "data:image/webp;base64,AAAA"},),
            )
        boom.assert_called_once()
        self.assertEqual([item["type"] for item in boom.call_args.args[1]], ["text", "image"])

    def test_unified_startup_grounding_across_hosted_backends_embodiments_and_rollback(self) -> None:
        """验证非 Codex 托管后端、纯文本模型、SO-101 构型动态相机与 run_turn 失败基线回滚。"""
        from ni_rua.cortex.agents.base import AgentCapabilities

        so101_embodiment = SimpleNamespace(overview_camera="fixed", wrist_cameras={"arm": "wrist"})

        # 1. 非 Codex 托管后端（claude_code）在 benchmark 启动时统一启用首轮多模态接地
        wb_cc, _ = self._make_workbench(
            "run-hosted-cc",
            {"features": {"external_perception": False, "startup_inline_grounding": True}},
        )
        run_turn_cc = Mock(return_value="turn-cc-1")
        wb_cc.session = SimpleNamespace(
            backend_kind="claude_code",
            capabilities=AgentCapabilities(turn_images=True),
            actor_dir=wb_cc.root / "actor",
            model="claude-test",
            provider="anthropic",
            effort="high",
            price_card=None,
            context_delivery="delta",
            state="idle",
            thread_id="thr-cc",
            turn_id=None,
            turn_done=threading.Event(),
            start=Mock(return_value="thr-cc"),
            run_turn=run_turn_cc,
            supports_turn_images=Mock(return_value=True),
            host_tool_output_truncation_limit=Mock(return_value=(500_000, None, True)),
            close=Mock(),
            model_proxy=None,
            provenance=Mock(return_value={"kind": "claude_code", "capabilities": {}}),
        )
        with patch("ni_rua.apps.gateway.workbench.Terminal", return_value=SimpleNamespace(start=AsyncMock(), close=AsyncMock(), session_id="t", process=SimpleNamespace(pid=-1))):
            asyncio.run(wb_cc.start(None, benchmark=True))
        run_turn_cc.assert_called_once()
        self.assertEqual(len(run_turn_cc.call_args.kwargs["images"]), 1)
        self.assertEqual(wb_cc.startup_grounding["image_kind"], "raw_rgb")
        self.assertEqual(wb_cc.startup_grounding["image_count"], 1)

        # 2. 纯文本模型（supports_turn_images=False）：不附图片且不谎称已附 cam_head
        wb_text, _ = self._make_workbench(
            "run-hosted-text-only",
            {"features": {"external_perception": False, "startup_inline_grounding": True}},
        )
        run_turn_text = Mock(return_value="turn-oc-1")
        wb_text.session = SimpleNamespace(
            backend_kind="opencode",
            capabilities=AgentCapabilities(turn_images=True),
            actor_dir=wb_text.root / "actor",
            model="text-only-model",
            provider="relay",
            effort="high",
            price_card=None,
            context_delivery="delta",
            state="idle",
            thread_id="thr-oc",
            turn_id=None,
            turn_done=threading.Event(),
            start=Mock(return_value="thr-oc"),
            run_turn=run_turn_text,
            supports_turn_images=Mock(return_value=False),
            close=Mock(),
            model_proxy=None,
            provenance=Mock(return_value={"kind": "opencode", "capabilities": {}}),
        )
        with patch("ni_rua.apps.gateway.workbench.Terminal", return_value=SimpleNamespace(start=AsyncMock(), close=AsyncMock(), session_id="t", process=SimpleNamespace(pid=-1))):
            asyncio.run(wb_text.start(None, benchmark=True))
        run_turn_text.assert_called_once()
        self.assertNotIn("images", run_turn_text.call_args.kwargs)
        self.assertEqual(wb_text.startup_grounding["image_kind"], "none")
        self.assertEqual(wb_text.startup_grounding["image_count"], 0)
        self.assertNotIn("cam_head is already attached above", run_turn_text.call_args.args[0])

        # 3. SO-101 构型动态相机选择（fixed / wrist），不写死 ARX-X5 相机名
        so101_bundle = {"cameras": [{"camera_id": "wrist"}, {"camera_id": "fixed"}]}
        self.assertEqual(
            startup_grounding.select_overview_camera(so101_bundle, embodiment=so101_embodiment),
            "fixed",
        )
        self.assertEqual(
            startup_grounding._wrist_camera_ids(so101_bundle, so101_embodiment, "fixed"),
            ("wrist",),
        )

        # 4. 首轮 run_turn 抛异常时立即回滚 startup-grounding 已提交基线
        wb_fail, _ = self._make_workbench(
            "run-hosted-rollback",
            {"features": {"external_perception": False, "startup_inline_grounding": True}},
        )
        wb_fail.session = SimpleNamespace(
            backend_kind="claude_code",
            capabilities=AgentCapabilities(turn_images=True),
            actor_dir=wb_fail.root / "actor",
            model="claude-test",
            provider="anthropic",
            effort="high",
            price_card=None,
            context_delivery="delta",
            state="idle",
            thread_id="thr-fail",
            turn_id=None,
            turn_done=threading.Event(),
            start=Mock(return_value="thr-fail"),
            run_turn=Mock(side_effect=RuntimeError("submit_boom")),
            supports_turn_images=Mock(return_value=True),
            close=Mock(),
            model_proxy=None,
            provenance=Mock(return_value={"kind": "claude_code", "capabilities": {}}),
        )
        with patch.object(wb_fail.harness, "_invalidate_actor_delivery", wraps=wb_fail.harness._invalidate_actor_delivery) as inv:
            with self.assertRaisesRegex(RuntimeError, "submit_boom"):
                asyncio.run(wb_fail.start(None, benchmark=True))
        inv.assert_called_once_with("startup-grounding")

    def test_native_audit_passes_for_both_obs3d_overlay_and_raw_rgb_startup(self) -> None:
        """验证首轮直接基于内联 obs_1 执行动作时，obs3d_overlay 与 raw_rgb 两种模式均通过 native audit。"""
        server = FakeObs3dServer(status="ok")
        self.addCleanup(server.close)

        def _run_and_execute_turn1_action(run_name: str, feature_overrides: dict) -> Workbench:
            wb, robot = self._make_workbench(run_name, {"features": feature_overrides})
            orig_start = robot.start
            robot.start = lambda: {**orig_start(), "data_origin": "robodojo_runtime"}
            _attach_fake_codex_runtime(wb)
            asyncio.run(wb.start(None, benchmark=True))
            harness: Harness = wb.harness
            plan_reply = harness.handle_tool(
                {
                    "threadId": harness.thread_id,
                    "turnId": "turn-1",
                    "callId": "call-plan-1",
                    "tool": "task_plan",
                    "arguments": {
                        "operation": "submit",
                        "observation_id": "obs_1",
                        "plan": "Pick and place the outer bowl onto the target coaster.",
                        "ledger": [],
                    },
                }
            )
            self.assertTrue(plan_reply["success"])
            wb.store.raw_codex(
                "received",
                {
                    "method": "item/started",
                    "params": {"threadId": harness.thread_id, "item": {"id": "item-act-1", "type": "dynamicToolCall"}},
                },
            )
            action = dict(
                schema_version="0.1",
                **wb.identity,
                world_kind="observed",
                action_id="turn1-act-1",
                based_on_observation_id=wb.initial_observation_id,
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
                    waypoints=[{"left": {"position_m": [0.18, 0.20, 0.90], "quaternion_xyzw": [0.0, 0.70710678, 0.0, 0.70710678]}}],
                ),
                gripper_events=[],
            )
            result = harness.execute(
                action,
                correlation={"thread_id": harness.thread_id, "turn_id": "turn-1", "item_id": "item-act-1"},
            )
            self.assertEqual(result["receipt"]["status"], "completed")
            wb.store.raw_codex(
                "received",
                {
                    "method": "turn/completed",
                    "params": {"threadId": harness.thread_id, "turn": {"id": "turn-1", "status": "completed", "items": []}},
                },
            )
            return wb

        # 路径 1：obs3d_overlay 模式（delivered_image_digests == 0, passed == True）
        wb_overlay = _run_and_execute_turn1_action(
            "audit-overlay",
            {"external_perception": True, "vggt_endpoint": server.url, "vggt_support_z_m": TABLE_Z, "held_perception": False},
        )
        overlay_audit = audit(wb_overlay.root, profile="native")
        self.assertTrue(overlay_audit["passed"], overlay_audit)
        self.assertEqual(overlay_audit["failures"], [])
        self.assertEqual(overlay_audit["delivered_image_digests"], 0)

        # 路径 2：raw_rgb 回退模式（delivered_image_digests == 1, passed == True）
        wb_raw = _run_and_execute_turn1_action(
            "audit-raw-rgb",
            {"external_perception": False},
        )
        raw_audit = audit(wb_raw.root, profile="native")
        self.assertTrue(raw_audit["passed"], raw_audit)
        self.assertEqual(raw_audit["failures"], [])
        self.assertEqual(raw_audit["delivered_image_digests"], 1)

        # 反向防篡改断言：若抹除 turn/start 中的内联图片，native audit 必须检出未证明交付
        raw_log_path = wb_raw.root / "codex-raw.jsonl"
        tampered = []
        for line in raw_log_path.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("message", {}).get("method") == "turn/start":
                rec["message"]["params"]["input"] = [
                    item for item in rec["message"]["params"]["input"] if item.get("type") != "image"
                ]
            tampered.append(json.dumps(rec))
        raw_log_path.write_text("\n".join(tampered) + "\n")
        tampered_audit = audit(wb_raw.root, profile="native")
        self.assertFalse(tampered_audit["passed"])
        self.assertIn("delivery not proved by actual SDK tool response", tampered_audit["failures"])

    def test_startup_grounding_views_all_cameras_preserves_default_and_attaches_clean_multiview(self) -> None:
        """验证默认 overview 保持 1 张图兼容，all_cameras 同时内联 3 路无遮挡原图 + 3D ID 标注图且零元数据泄露。"""
        from ni_rua.core.tool_output import StructuredText
        from ni_rua.cortex.context import startup_grounding

        self.assertEqual(Config().features.startup_grounding_views, "overview")
        cfg = Config().with_overrides(
            {"features": {"external_perception": False, "startup_inline_grounding": True, "startup_grounding_views": "all_cameras"}}
        )
        store = Store(
            self.root / "all-cams-store",
            dict(run_id="all-cams", episode_id="ep", env_instance_id="env", env_generation=1),
        )
        self.addCleanup(store.close)
        robot = ThreeCameraDualArmRobot(self.root / "all-cams-imgs")
        harness = Harness(store, robot, 10, observation_profile="rgb_proprio_calibrated_v1", config=cfg)
        harness.start()
        fake_overlay = {
            "type": "inputImage",
            "imageUrl": "data:image/webp;base64,UklGRiIAAABXRUJQVlA4IBYAAAAwAQCdASoBAAEADsD+JaQAA3AAAAAA",
        }
        fake_ot = StructuredText(
            {"observation_id": harness.latest["observation_id"], "object_table": {"columns": ["id"], "rows": [["o1"]]}},
            "data",
        )
        with patch("ni_rua.cortex.context.startup_grounding.obs3d_attachments", return_value=[fake_ot, fake_overlay]):
            grounded = startup_grounding.build_startup_grounding(harness)
        self.assertEqual(grounded.record["image_kind"], "all_cameras_with_obs3d_overlay")
        self.assertEqual(grounded.record["image_count"], 4)
        self.assertEqual(len(grounded.image_items), 4)
        self.assertEqual(metadata_findings(grounded.section_text), [])
        self.assertIn("All calibrated raw RGB camera views", grounded.section_text)

if __name__ == "__main__":
    unittest.main()
