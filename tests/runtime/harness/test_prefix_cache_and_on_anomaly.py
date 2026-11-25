"""
- test_on_anomaly_receipt_and_single_panel.py: Phase 1-2 单元测试：默认无图紧凑 3D 回执 on_anomaly、单面板 640×480 复合轨迹图、program_run 3D 事实开放与 native audit 闭环。

运行方式：
PYTHONPATH=tests:. mamba run -n ni-rua python -m unittest tests/unit/test_on_anomaly_receipt_and_single_panel.py -v
- test_prefix_cache_and_verbosity.py: 前缀缓存字节幂等锁定、默认输出详略度下调与同仿真步双倍渲染去重回归测试。
"""
from __future__ import annotations

import asyncio
import base64
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
import cv2
import numpy as np
from PIL import Image
from ni_rua.apps.evaluation.audit import audit
from ni_rua.apps.gateway.workbench import Workbench
from ni_rua.core.config import Config
from ni_rua.core.contracts import canonical
from ni_rua.core.observation_policy import ACTOR_PROFILE
from ni_rua.core.storage.store import Store
from ni_rua.perception.external import obs3d
from ni_rua.perception.overlays.trajectory_overlay import render_trajectory_overlay
from ni_rua.robot.backends.robodojo.robodojo import FRAME_ID, TCP_CONFIG_ID
from ni_rua.robot.motion.batch_geometry import annotate_overlay
from ni_rua.runtime.actions.compact_trajectory_summary import build_compact_trajectory_summary
from ni_rua.runtime.actions.trajectory_feedback import build_action_trajectory_feedback
from ni_rua.runtime.actions.unified_actions import dispatch as unified_dispatch
from ni_rua.runtime.harness import Harness
from ni_rua.runtime.programs.runner import ProgramRunner
import tomllib
from unittest import mock
from ni_rua.core.observation_policy import DEPTH_ACTOR_PROFILE, require_actor_observation
from ni_rua.core.presentation.actor_presentation import ActorPresentation, metadata_findings
from ni_rua.core.storage.images import camera_path, encode_image, image_settings
from ni_rua.cortex.context.task_planning import TaskPlanning
from ni_rua.cortex.agents.codex.session import CodexSession, DEFAULT_MODEL_VERBOSITY
from ni_rua.robot.backends.robodojo.robot_worker import RoboDojoAdapter
from support.host_capabilities import expect, program_isolation

NI_RUA_TEST_TIER = "component"


# --- Folded from tests/unit/test_on_anomaly_receipt_and_single_panel.py ---

RUN_IDENTITY = {
    "run_id": "run-p1-unit",
    "episode_id": "ep-p1-unit",
    "env_instance_id": "env-p1-unit",
    "env_generation": 1,
}


TABLE_Z = 0.765
CAMERA_K = [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
T_HEAD = [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.35], [0.0, 0.0, 0.0, 1.0]]


def _translation(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> list[list[float]]:
    return [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, y], [0.0, 0.0, 1.0, z], [0.0, 0.0, 0.0, 1.0]]


def _urdf_6dof_model() -> dict:
    """构造带双指夹爪（joint7/joint8）的标准 6 自由度公开运动学模型。"""
    joints = []
    for idx in range(1, 7):
        t_mat = np.eye(4)
        t_mat[0, 3] = 0.06
        joints.append(
            dict(
                name=f"joint{idx}",
                type="revolute",
                parent="base_link" if idx == 1 else f"link{idx - 1}",
                child=f"link{idx}",
                axis=[0.0, 0.0, 1.0],
                T_parent_joint=t_mat.tolist(),
            )
        )
    for idx in (7, 8):
        joints.append(
            dict(
                name=f"joint{idx}",
                type="prismatic",
                parent="link6",
                child=f"link{idx}",
                axis=[0.0, 1.0 if idx == 7 else -1.0, 0.0],
                T_parent_joint=np.eye(4).tolist(),
            )
        )
    return dict(base_link="base_link", joints=joints, source_sha256="unit-test-model")


def _synthetic_npz_bytes() -> bytes:
    """构造最小合法 scene3d 压缩点云字节流。"""
    xyz = np.array(
        [[0.0, 0.0, TABLE_Z], [0.06, 0.0, TABLE_Z], [0.12, 0.04, TABLE_Z + 0.035]],
        dtype=np.float32,
    )
    conf = np.array([210, 220, 240], dtype=np.uint8)
    uv = np.array([[320, 240], [350, 240], [380, 220]], dtype=np.uint16)
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


def _fake_object_row(centre: tuple[float, float, float] = (0.12, 0.04, TABLE_Z + 0.02)) -> dict:
    """构造合成 3D 物体表行数据。"""
    u, v = 380, 220
    return dict(
        centre_m=list(centre),
        top_z_m=centre[2] + 0.02,
        bottom_z_m=TABLE_Z,
        length_m=0.08,
        width_m=0.03,
        yaw_deg=15.0,
        axis_tilt_deg=89.0,
        elongation=2.6,
        support="table",
        flags=[],
        quality="ok",
        label_px=[u, v],
        contour_px=[[u - 10, v - 6], [u + 10, v - 6], [u + 10, v + 6], [u - 10, v + 6]],
        mask_px=220,
        n_points=140,
        iou_pred=0.93,
        conf_rank_p50=0.7,
        contact_gap_m=0.0,
        cluster_frac=1.0,
    )


class FakeObs3dServer:
    """本地轻量 HTTP 假服务，模拟 VGGT /object_table 与 /points 端点。"""

    def __init__(self, *, status: str = "ok") -> None:
        self.status = status
        self.requests: list[tuple[str, dict]] = []
        self.objects = [_fake_object_row()]
        self.object_sequence: list[list[dict]] = []
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
                    current_objs = owner.object_sequence.pop(0) if owner.object_sequence else owner.objects
                    answer, code = (
                        dict(
                            status="ok",
                            version="fake-ot-v5",
                            objects=deepcopy(current_objs),
                            requested=[],
                            omitted=0,
                            labels=[],
                            support_z_m=body["support_z_m"],
                            dropped={"robot": 0},
                            n_prompts=4,
                            timing={"total_s": 0.005},
                            support_fit={"tilt_deg": 0.1},
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
                            bbox_above_support_m=[[0.06, 0.0, TABLE_Z], [0.12, 0.04, TABLE_Z + 0.035]],
                            npz_b64=base64.b64encode(npz_bytes).decode(),
                            npz_bytes=len(npz_bytes),
                            timing={"total_s": 0.005},
                            support_fit={"tilt_deg": 0.1},
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


class BatchCapableDualArmRobot:
    """支持真实多步 BatchAction 执行、settle 保持动作与可配置异常/开度的双臂测试机器人。"""

    def __init__(self, root: Path, *, data_origin: str = "robodojo_runtime") -> None:
        self.root = Path(root)
        self.data_origin = data_origin
        self.sim_step = 0
        self.observations = 0
        self.executions = 0
        self.next_status = "completed"
        self.next_reason = None
        self.next_jaw_position_m = [0.022, 0.022]
        self.calibration_has_extrinsics = True
        self._plans: dict[str, dict] = {}
        self.model = _urdf_6dof_model()
        self.q_state = {"left": [0.0] * 6, "right": [0.0] * 6}
        self.jaw_state = {"left": [0.044, 0.044], "right": [0.044, 0.044]}
        self.image_paths: dict[str, Path] = {}
        rng = np.random.default_rng(19)
        for idx, cam_id in enumerate(("cam_head", "cam_left_wrist", "cam_right_wrist")):
            img = rng.integers(35 + idx * 15, 205, size=(480, 640, 3), dtype=np.uint8)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            path = self.root / f"{cam_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(cv2.imencode(".png", img)[1].tobytes())
            self.image_paths[cam_id] = path

    def _fk_link6(self, arm: str, q_rad: list[float]) -> tuple[list[float], list[list[float]]]:
        from ni_rua.robot.model.geometry import fk

        base_y = 0.25 if arm == "left" else -0.25
        jaw = list(self.jaw_state[arm])
        poses = fk(
            {**self.model, "T_world_base": np.asarray(_translation(0.0, base_y, TABLE_Z + 0.12))},
            dict(
                joint_names=[f"joint{i}" for i in range(1, 9)],
                q_rad=list(q_rad) + [jaw[0], -jaw[1]],
                source_kind="measurement",
            ),
        )
        mat = poses["link6"]
        return mat[:3, 3].tolist(), mat[:3, :3].tolist()

    def _state_snapshot(self) -> dict:
        from scipy.spatial.transform import Rotation

        arms = {}
        for arm in ("left", "right"):
            pos_m, rot_m = self._fk_link6(arm, self.q_state[arm])
            quat_xyzw = Rotation.from_matrix(np.asarray(rot_m, dtype=float)).as_quat().tolist()
            arms[arm] = dict(
                source_kind="measurement",
                joint_names=[f"joint{i}" for i in range(1, 7)],
                q_rad=list(self.q_state[arm]),
                qdot_rad_s=[0.0] * 6,
                ee_pose=dict(
                    position_m=pos_m,
                    quaternion_xyzw=quat_xyzw,
                    frame_id=FRAME_ID,
                    tcp_config_id=TCP_CONFIG_ID,
                    source_kind="measurement",
                ),
                gripper=dict(
                    joint_names=["joint7", "joint8"],
                    goal_normalized=1.0,
                    goal_source_kind="command",
                    command_normalized=1.0,
                    command_source_kind="command",
                    position_m=list(self.jaw_state[arm]),
                    position_source_kind="measurement",
                ),
            )
        return dict(
            data_origin=self.data_origin,
            source_kind="measurement",
            frame_id=FRAME_ID,
            tcp_config_id=TCP_CONFIG_ID,
            sim_step=self.sim_step,
            step_domain="adapter_control_step",
            arms=arms,
        )

    def start(self) -> dict:
        return dict(
            data_origin=self.data_origin,
            robot_model_id="arx-x5",
            frame_id=FRAME_ID,
            tcp_config_id=TCP_CONFIG_ID,
            limits_profile="robodojo-native-v1",
            path_kinds=["backend_free", "hold"],
            max_waypoints=4,
            dual_arm=True,
            gripper_after_motion=True,
            bounded_wait=dict(max_native_steps=50),
            action_budget=dict(native_step_limit=200),
            batch_actions=dict(available=True, max_stages=32),
        )

    def observe(self) -> dict:
        self.observations += 1
        frames, calibration, poses = [], {}, {}
        specs = (
            ("cam_head", None, T_HEAD),
            ("cam_left_wrist", "left", _translation(0.15, 0.20, 0.95)),
            ("cam_right_wrist", "right", _translation(0.15, -0.20, 0.95)),
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
                    control_dt_s=0.04,
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
                    T_link_camera=_translation(0.03, 0.0, 0.02),
                    T_world_base=_translation(0.0, base_y, TABLE_Z + 0.12),
                )
            elif self.calibration_has_extrinsics:
                calibration[cam_id]["T_world_camera"] = T_HEAD
        return dict(
            data_origin=self.data_origin,
            world_kind="observed",
            observation_profile=ACTOR_PROFILE,
            instruction="Pick the cylinder and place it upright into the slot.",
            sim_step=self.sim_step,
            step_domain="adapter_control_step",
            sim_time_s=round(self.sim_step * 0.04, 4),
            control_dt_s=0.04,
            captured_at="2026-11-11T10:00:00Z",
            cameras=frames,
            camera_calibration=dict(reference_frame_id=FRAME_ID, cameras=calibration),
            camera_pose_sample=dict(
                reference_frame_id=FRAME_ID,
                sim_step=self.sim_step,
                step_domain="adapter_control_step",
                cameras=poses,
            ),
            robot_model=deepcopy(self.model),
            robot_state=self._state_snapshot(),
        )

    def preview_batch(self, action: dict) -> dict:
        plan_id = f"batch-plan-{len(self._plans) + 1}"
        stages = action.get("stages", [])
        rows, contexts = [], []
        q_curr = deepcopy(self.q_state)
        for idx, stage in enumerate(stages, 1):
            arm = stage["arm"]
            q_next = [round(v + 0.08 * idx, 5) for v in q_curr[arm]]
            q_curr[arm] = q_next
            row = [*q_curr["left"], 1.0, *q_curr["right"], 1.0]
            rows.append(row)
            contexts.append(
                {
                    "left": dict(phase=stage["kind"] if arm == "left" else "hold", stage_id=stage["id"] if arm == "left" else None),
                    "right": dict(phase=stage["kind"] if arm == "right" else "hold", stage_id=stage["id"] if arm == "right" else None),
                }
            )
        samples = [
            dict(
                native_step=0,
                arms={a: dict(q_rad=list(self.q_state[a]), jaw_goal_normalized=1.0, phase="initial") for a in ("left", "right")},
            )
        ]
        for step_idx, row in enumerate(rows, 1):
            samples.append(
                dict(
                    native_step=step_idx,
                    arms={
                        "left": dict(q_rad=row[:6], jaw_goal_normalized=row[6], phase="move"),
                        "right": dict(q_rad=row[7:13], jaw_goal_normalized=row[13], phase="move"),
                    },
                )
            )
        traj = dict(
            schema_version="candidate-trajectory-v1",
            source_kind="candidate_command_trajectory_not_measurement",
            frame_id=FRAME_ID,
            tcp_config_id=TCP_CONFIG_ID,
            joint_names={a: [f"joint{i}" for i in range(1, 7)] for a in ("left", "right")},
            samples=samples,
            geometry=dict(status="unknown"),
            inter_arm_proximity=dict(status="unknown"),
            execution_binding="exact_cached_rows; no replanning or resampling",
        )
        self._plans[plan_id] = dict(stages=deepcopy(stages), rows=rows, contexts=contexts)
        return dict(
            status="planned",
            plan_id=plan_id,
            planned_native_steps=len(rows),
            trajectory_data=traj,
            trajectory_evidence=dict(available=True),
            stage_costs={},
        )

    def execute(self, action: dict, progress_callback=None, cancel_flag=None) -> dict:
        self.executions += 1
        before_state = self._state_snapshot()
        before_obs = self.observe()
        # 处理 bounded_settle 或 robot_wait 的纯保持（hold）动作
        if isinstance(action.get("path"), dict) and action["path"].get("kind") == "hold":
            steps = int(action.get("hold_native_steps", 2))
            actual_trace = []
            rows = []
            for _ in range(steps):
                self.sim_step += 1
                rows.append([*self.q_state["left"], 1.0, *self.q_state["right"], 1.0])
                actual_trace.append(dict(sim_step=self.sim_step, state=self._state_snapshot()))
            terminal_obs = self.observe()
            return dict(
                status="completed",
                reason=None,
                data_origin=self.data_origin,
                executed_steps=steps,
                completed_native_steps=steps,
                position_error_m=0.0,
                orientation_error_rad=0.0,
                before_observation=before_obs,
                terminal_observation=terminal_obs,
                before_state=before_state,
                trace_sampling=dict(actual_every_native_actions=1, actual_boundary_samples=True),
                planned_trace=dict(segments=[dict(actions=rows, completed_native_steps=steps)]),
                dispatched_trace=dict(steps=steps),
                actual_trace=actual_trace,
            )

        plan_info = self._plans.get(action.get("batch_plan_id"))
        rows = deepcopy(plan_info["rows"]) if plan_info else [[0.08] * 6 + [1.0] + [0.0] * 6 + [1.0]]
        contexts = deepcopy(plan_info["contexts"]) if plan_info else []
        stages = plan_info["stages"] if plan_info else action.get("stages", [])
        status = self.next_status
        reason = self.next_reason
        completed_steps = len(rows) if status == "completed" else max(1, len(rows) - 1)
        actual_trace = []
        for idx in range(completed_steps):
            self.sim_step += 1
            row = rows[idx]
            err = 0.04 if (status != "completed" and idx == completed_steps - 1) else 0.0
            self.q_state["left"] = [v + err for v in row[:6]]
            self.q_state["right"] = list(row[7:13])
            self.jaw_state["left"] = list(self.next_jaw_position_m)
            actual_trace.append(dict(sim_step=self.sim_step, state=self._state_snapshot()))
        terminal_obs = self.observe()
        stage_results = {}
        for s_idx, st in enumerate(stages):
            done = s_idx < completed_steps and (status == "completed" or s_idx < completed_steps - 1)
            stage_results[st["id"]] = dict(
                kind=st["kind"],
                arm=st["arm"],
                status="completed" if done else status,
                planned_steps=1,
                completed_steps=1 if s_idx < completed_steps else 0,
            )
        return dict(
            status=status,
            reason=reason,
            data_origin=self.data_origin,
            executed_steps=completed_steps,
            completed_native_steps=completed_steps if status == "completed" else max(0, completed_steps - 1),
            position_error_m=0.001 if status == "completed" else 0.032,
            orientation_error_rad=0.01 if status == "completed" else 0.08,
            stage_results=stage_results,
            before_observation=before_obs,
            terminal_observation=terminal_obs,
            before_state=before_state,
            trace_sampling=dict(actual_every_native_actions=1, actual_boundary_samples=True),
            planned_trace=dict(
                segments=[
                    dict(
                        actions=rows,
                        row_contexts=contexts,
                        completed_native_steps=completed_steps if status == "completed" else max(0, completed_steps - 1),
                    )
                ]
            ),
            dispatched_trace=dict(steps=completed_steps),
            actual_trace=actual_trace,
        )

    def close(self) -> dict:
        return {"process_exited": True}


class FakeSessionClient:
    """记录 turn_start 并将请求与事件真实写入 codex-raw.jsonl 以供 native audit 检验。"""

    def __init__(self, store: Store) -> None:
        self.store = store
        self.turn_calls: list[tuple[str, list[dict], dict | None]] = []

    def turn_start(self, thread_id: str, input_items: list[dict], params: dict | None = None):
        self.turn_calls.append((thread_id, deepcopy(input_items), deepcopy(params)))
        turn_id = f"turn-{len(self.turn_calls)}"
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


def _attach_fake_codex_session(workbench: Workbench) -> FakeSessionClient:
    fake_client = FakeSessionClient(workbench.store)

    def _start_session(tools, instructions, resume_thread_id=None):
        thread_id = resume_thread_id or "thread-p1-audit"
        workbench.session.client = fake_client
        workbench.session.thread_id = thread_id
        workbench.session.state = "idle"
        workbench.session.provider = "test-provider"
        workbench.session.cli_executable = "/usr/bin/codex"
        workbench.session.process = SimpleNamespace(pid=54321)
        workbench.session.model_proxy = SimpleNamespace(transport_id="transport-p1", image_context_epoch=1)
        workbench.store.set_state(
            "runtime",
            "codex",
            {"thread_id": thread_id, "app_server_instance_id": "app-instance-p1"},
        )
        return thread_id

    workbench.session.start = _start_session
    workbench.session.close = lambda: None
    return fake_client


class OnAnomalyReceiptAndSinglePanelTests(unittest.TestCase):
    """验证紧凑 3D 轨迹摘要、640×480 单面板复合轨迹图、on_anomaly 零图回执、program_run 3D 贯通与 native audit。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _make_harness(self, name: str, *, server_url: str, extra_overrides: dict | None = None) -> tuple[Harness, BatchCapableDualArmRobot, Store]:
        run_dir = self.root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        store = Store(run_dir, dict(RUN_IDENTITY))
        self.addCleanup(store.close)
        robot = BatchCapableDualArmRobot(run_dir / "captures")
        overrides = {
            "features": {
                "external_perception": True,
                "vggt_endpoint": server_url,
                "vggt_support_z_m": TABLE_Z,
                "held_perception": False,
                "motion_review": False,
                "task_scheduling": False,
                "sandbox_enabled": expect(program_isolation, "program_run runs under the Landlock + seccomp sandbox"),
                "action_trajectory_delivery": "diagnostic",
            },
            "images": {
                "action_feedback_policy": "on_anomaly",
            },
        }
        if extra_overrides:
            for sec, vals in extra_overrides.items():
                overrides.setdefault(sec, {}).update(vals)
        config = Config().with_overrides(overrides)
        harness = Harness(store, robot, config=config)
        harness.thread_id = "thread-unit"
        harness.start()
        harness.task_planning.call(
            dict(operation="submit", observation_id=harness.latest["observation_id"], plan="Execute manipulation stages.", ledger=[]),
            harness.identity,
            harness.latest,
            dict(thread_id=harness.thread_id, turn_id="t0"),
        )
        return harness, robot, store

    def test_default_configs_and_legacy_archive_compat(self) -> None:
        """默认配置归正为 automatic + all，并与旧归档保持兼容。"""
        cfg_cls = Config()
        cfg_file = Config.load(environ={})
        for cfg in (cfg_cls, cfg_file):
            self.assertEqual(cfg.features.action_trajectory_delivery, "automatic")
            self.assertEqual(cfg.images.action_feedback_policy, "all")
            self.assertEqual(cfg.images.feedback_policy, "all")

        # 验证不含 action_feedback_policy 的旧归档恢复时保持原有 feedback_policy
        legacy_dir = self.root / "legacy-store"
        legacy_store = Store(legacy_dir, dict(RUN_IDENTITY))
        self.addCleanup(legacy_store.close)
        legacy_store.set_state(
            "runtime",
            "image_settings",
            dict(
                feedback_policy="all",
                transport_format="webp",
                webp_quality=85,
                webp_method=4,
                jpeg_quality=95,
                storage_jpeg_quality=95,
            ),
        )
        robot = BatchCapableDualArmRobot(legacy_dir / "cap")
        reloaded = Harness(legacy_store, robot, config=Config())
        self.assertEqual(reloaded.image_settings["action_feedback_policy"], "all")

    def test_compact_3d_trajectory_summary_accuracy_and_stages(self) -> None:
        """验证纯数值 3D 轨迹摘要精确提取起止坐标、峰值偏差步、实测夹爪开度、双臂净空与各阶段摘要。"""
        rot = np.eye(3).tolist()
        measured_samples = []
        planned_samples = []
        # 构造 4 个边界步（0, 1, 2, 3），在第 2 步注入精确 0.015m 的 Z 向跟踪偏差
        meas_x = [0.18, 0.22, 0.26, 0.26]
        meas_z = [0.90, 0.85, 0.815, 0.88]
        plan_z = [0.90, 0.85, 0.800, 0.88]
        jaws = [[0.044, 0.044], [0.044, 0.044], [0.006, 0.006], [0.006, 0.006]]
        stage_ids = [None, "approach", "grasp", "lift"]
        for step in range(4):
            l_pos_m = [meas_x[step], 0.15, meas_z[step]]
            l_pos_p = [meas_x[step], 0.15, plan_z[step]]
            r_pos = [0.18, -0.25, 0.90]
            measured_samples.append(
                dict(
                    native_step=step,
                    sim_step=step * 5,
                    arms=dict(
                        left=dict(
                            link_positions_m=dict(link5=[l_pos_m[0] - 0.05, 0.15, l_pos_m[2]], link6=l_pos_m),
                            link6_pose=dict(position_m=l_pos_m, rotation_matrix=rot),
                            measured_jaw=dict(joint_names=["joint7", "joint8"], position_m=jaws[step]),
                        ),
                        right=dict(
                            link_positions_m=dict(link5=[0.13, -0.25, 0.90], link6=r_pos),
                            link6_pose=dict(position_m=r_pos, rotation_matrix=rot),
                            measured_jaw=dict(joint_names=["joint7", "joint8"], position_m=[0.044, 0.044]),
                        ),
                    ),
                )
            )
            planned_samples.append(
                dict(
                    native_step=step,
                    execution_state="initial_measurement" if step == 0 else "completed_command",
                    arms=dict(
                        left=dict(
                            link_positions_m=dict(link6=l_pos_p),
                            link6_pose=dict(position_m=l_pos_p, rotation_matrix=rot),
                            phase="move",
                            **({"stage_id": stage_ids[step]} if stage_ids[step] else {}),
                        ),
                        right=dict(
                            link_positions_m=dict(link6=r_pos),
                            link6_pose=dict(position_m=r_pos, rotation_matrix=rot),
                            phase="hold",
                        ),
                    ),
                )
            )
        paths = dict(
            measured=dict(frame_id=FRAME_ID, executed_native_steps=3, samples=measured_samples),
            planned=dict(frame_id=FRAME_ID, samples=planned_samples),
        )
        action = dict(
            stages=[
                dict(id="approach", arm="left", kind="move"),
                dict(id="grasp", arm="left", kind="gripper"),
                dict(id="lift", arm="left", kind="move"),
            ]
        )
        receipt = dict(
            status="completed",
            executed_steps=3,
            completed_native_steps=3,
            stage_results=dict(
                approach=dict(arm="left", kind="move", status="completed"),
                grasp=dict(arm="left", kind="gripper", status="completed"),
                lift=dict(arm="left", kind="move", status="completed"),
            ),
        )
        summary = build_compact_trajectory_summary(paths, action, receipt)
        self.assertIsNotNone(summary)
        self.assertNotIn("q_rad", canonical(summary))
        left = summary["arms"]["left"]
        self.assertEqual(left["start_link6_m"], [0.18, 0.15, 0.9])
        self.assertEqual(left["end_link6_m"], [0.26, 0.15, 0.88])
        self.assertAlmostEqual(left["max_divergence_m"], 0.015, places=4)
        self.assertEqual(left["peak_divergence_step"], 2)
        self.assertEqual(left["peak_divergence_position_m"], [0.26, 0.15, 0.815])
        self.assertEqual(
            left["measured_aperture_m"],
            dict(start_m=0.088, end_m=0.012, min_m=0.012, max_m=0.088),
        )
        self.assertAlmostEqual(summary["min_inter_arm_clearance_m"], 0.4, places=3)
        self.assertEqual([s["stage_id"] for s in summary["stages"]], ["approach", "grasp", "lift"])
        self.assertAlmostEqual(summary["stages"][1]["max_divergence_m"], 0.015, places=4)
        self.assertEqual(summary["stages"][1]["measured_aperture_end_m"], 0.012)

    def test_single_panel_640x480_composite_overlay_and_proxy_annotation(self) -> None:
        """验证轨迹叠加图升级为固定 640×480 单面板复合图，并与 batch_geometry.annotate_overlay 完整兼容。"""
        obs_doc = dict(world_kind="observed", observation_id="obs-panel", sim_step=1, step_domain="adapter_control_step")
        frame = dict(
            camera_id="cam_head",
            frame_id="head",
            sim_step=1,
            step_domain="adapter_control_step",
            rgb_ref="rgb-head",
            pixel_transform="identity",
        )
        camera = dict(
            observation=obs_doc,
            frame=frame,
            width=640,
            height=480,
            K=np.asarray(CAMERA_K, dtype=float),
            D=np.zeros(5),
            T=np.eye(4),
            reference_frame_id=FRAME_ID,
            pose_source="fixed_calibration",
        )
        base_img = np.full((480, 640, 3), 40, dtype=np.uint8)
        samples = []
        for step in range(4):
            x_val = -0.15 + 0.10 * step
            y_val = -0.05 + 0.04 * step
            samples.append(
                dict(
                    native_step=step,
                    execution_state="completed_command",
                    arms=dict(
                        left=dict(
                            phase="move",
                            link_positions_m=dict(
                                link5=[x_val - 0.06, y_val, 1.4],
                                link6=[x_val, y_val, 1.4],
                            ),
                            link6_pose=dict(position_m=[x_val, y_val, 1.4], rotation_matrix=np.eye(3).tolist()),
                        )
                    ),
                )
            )
        trajectory = dict(
            source_kind="measured_action_trajectory",
            frame_id=FRAME_ID,
            label="single-panel-check",
            executed_native_steps=3,
            robot_model=dict(joints=[dict(parent="link5", child="link6")]),
            samples=samples,
            batch_geometry_review=dict(
                obstacles=[dict(id="obs_box", min_m=[-0.05, -0.05, 1.35], max_m=[0.05, 0.05, 1.48])],
                held_objects={"left": dict(min_m=[-0.02, -0.02, 1.36], max_m=[0.04, 0.02, 1.44])},
            ),
        )
        rendered, metadata = render_trajectory_overlay(camera, base_img, trajectory)
        # 核心断言：输出尺寸严格等于 640×480 单面板（H=480, W=640），绝不再生成 1920×480 三联图
        self.assertEqual(rendered.shape, (480, 640, 3))
        self.assertEqual((metadata["panel_width"], metadata["panel_height"]), (640, 480))
        self.assertEqual([p["native_step"] for p in metadata["panels"]], [0, 2, 3])
        # 中央可视区域应绘制出连续轨迹色带与半透明骨架像素变化
        self.assertTrue(np.any(rendered[80:400, 80:560] != base_img[80:400, 80:560]))

        annotated, annotated_meta = annotate_overlay(camera, rendered, metadata, trajectory)
        self.assertEqual(annotated.shape, (480, 640, 3))
        self.assertEqual(annotated_meta["proxy_overlay"]["obstacle_boxes"], 1)
        self.assertEqual(annotated_meta["proxy_overlay"]["held_object_arms"], ["left"])

    def test_post_action_on_anomaly_zero_image_and_anomaly_fallbacks(self) -> None:
        """验证真实多步动作默认 0 图回执、0.0m 薄布开度不误判异常、受阻/3D 故障自动附图及显式覆盖。"""
        server = FakeObs3dServer(status="ok")
        self.addCleanup(server.close)
        harness, robot, store = self._make_harness("run-on-anomaly", server_url=server.url)

        def _count_images(wire_reply: dict) -> list[tuple[int, int]]:
            sizes = []
            for item in wire_reply["contentItems"]:
                if item.get("type") == "inputImage":
                    raw = base64.b64decode(item["imageUrl"].split(",", 1)[1])
                    with Image.open(io.BytesIO(raw)) as im:
                        sizes.append(im.size)
            return sizes

        # 1) 正常完成的多阶段动作 + 有效 obs3d -> 0 张轨迹图 + 0 张原图 + 0 张叠图，但完整包含 3D 结构化状态与 delivery 记录
        obs1_id = harness.latest["observation_id"]
        stages = [
            dict(id="approach", kind="move", arm="left", grasp_position_m=[0.20, 0.15, 0.88], angles_deg=[0, 0, 0]),
            dict(id="lift", kind="move", arm="left", depends_on=["approach"], grasp_position_m=[0.20, 0.15, 0.94], angles_deg=[0, 0, 0]),
        ]
        typed_ok = unified_dispatch(
            harness,
            dict(mode="direct", observation_id=obs1_id, stages=stages),
            dict(thread_id=harness.thread_id, turn_id="t1", tool_call_id="c1", item_id="i1"),
        )
        wire_ok = harness.actor_output.present(typed_ok, tool="robot_action")
        self.assertTrue(wire_ok["success"])
        self.assertEqual(_count_images(wire_ok), [])
        action_header = json.loads(wire_ok["contentItems"][0]["text"])
        self.assertEqual(action_header["trajectory_feedback"]["delivery"], "routine_execution_summary")
        self.assertIn("compact_3d_trajectory", action_header["trajectory_feedback"])
        obs_header = json.loads(wire_ok["contentItems"][1]["text"])
        self.assertEqual(obs_header["image_feedback"]["policy"], "on_anomaly")
        self.assertEqual(obs_header["image_feedback"]["included_camera_ids"], [])
        obs3d_part = json.loads(wire_ok["contentItems"][-1]["text"])
        self.assertEqual(obs3d_part["object_table"]["rows"][0][0], "o1")
        self.assertIn("scene3d", obs3d_part)
        obs2_id = typed_ok.data["receipt"]["result_observation_id"]
        # 验证 SQLite deliveries 表确实为 0 图观测写入了 images=[] 记录，且原始三相机 RGB 归档完好
        deliveries = [d for d in store.records("delivery") if d["observation_id"] == obs2_id]
        self.assertEqual(len(deliveries), 1)
        self.assertEqual(deliveries[0]["images"], [])
        self.assertEqual(len(store.read("observation", obs2_id)["cameras"]), 3)

        # 2) AGENTS.md 红线验证：夹爪完全闭合 0.0m（如夹持极薄布料），只要 status == "completed"，严禁误判为异常附图！
        robot.next_jaw_position_m = [0.0, 0.0]
        server.objects = [_fake_object_row((0.16, 0.08, TABLE_Z + 0.06))]
        typed_thin = unified_dispatch(
            harness,
            dict(mode="direct", observation_id=obs2_id, stages=[stages[0]]),
            dict(thread_id=harness.thread_id, turn_id="t2", tool_call_id="c2", item_id="i2"),
        )
        wire_thin = harness.actor_output.present(typed_thin, tool="robot_action")
        self.assertEqual(_count_images(wire_thin), [])
        thin_header = json.loads(wire_thin["contentItems"][0]["text"])
        self.assertEqual(
            thin_header["trajectory_feedback"]["compact_3d_trajectory"]["arms"]["left"]["measured_aperture_m"]["end_m"],
            0.0,
        )
        thin_obs3d = json.loads(wire_thin["contentItems"][-1]["text"])
        self.assertIn("object_state_delta", thin_obs3d)

        # 3) 契约级动作异常（status == "stopped", reason == "tracking_error"）-> 自动附单面板 640×480 诊断轨迹图 + 3 张原图
        obs3_id = typed_thin.data["receipt"]["result_observation_id"]
        robot.next_status = "stopped"
        robot.next_reason = "tracking_error"
        typed_err = unified_dispatch(
            harness,
            dict(mode="direct", observation_id=obs3_id, stages=stages),
            dict(thread_id=harness.thread_id, turn_id="t3", tool_call_id="c3", item_id="i3"),
        )
        wire_err = harness.actor_output.present(typed_err, tool="robot_action")
        err_sizes = _count_images(wire_err)
        # 2 张单面板轨迹图（planned + measured，各 640×480）+ 3 张相机原图（各 640×480）
        self.assertGreaterEqual(len(err_sizes), 4)
        for size in err_sizes:
            self.assertEqual(size, (640, 480))

        # 4) 外部 3D 感知故障（status == "completed"，但 obs3d 服务 503 不可用）-> 自动回退交付 3 张相机原图
        robot.next_status = "completed"
        robot.next_reason = None
        server.status = "gpu_busy"
        obs4_id = typed_err.data["receipt"]["result_observation_id"]
        typed_s3_down = unified_dispatch(
            harness,
            dict(mode="direct", observation_id=obs4_id, stages=[stages[0]]),
            dict(thread_id=harness.thread_id, turn_id="t4", tool_call_id="c4", item_id="i4"),
        )
        wire_s3_down = harness.actor_output.present(typed_s3_down, tool="robot_action")
        self.assertEqual(_count_images(wire_s3_down), [(640, 480)] * 3)

        # 5) 显式 image_delivery 覆盖（head_only / focus_roi / none / all）及主动 robot_observe 默认正常返回图像
        server.status = "ok"
        robot.q_state["left"] = [-1.3, 0.0, 0.0, 0.0, 0.0, 0.0]
        latest_bundle = harness.observe({"tool_call_id": "obs-check"})
        clean_receipt = dict(status="completed", reason=None)
        head_parts = harness.observation_content(
            latest_bundle,
            {"thread_id": harness.thread_id, "tool_call_id": "ovr-head"},
            feedback_action={"type": "BatchAction", "image_delivery": "head_only"},
            feedback_receipt=clean_receipt,
        )
        self.assertEqual(sum(p["type"] == "inputImage" for p in head_parts), 1)

        roi_parts = harness.observation_content(
            latest_bundle,
            {"thread_id": harness.thread_id, "tool_call_id": "ovr-roi"},
            feedback_action={"type": "BatchAction", "image_delivery": "focus_roi", "stages": [{"arm": "left"}]},
            feedback_receipt=clean_receipt,
        )
        roi_imgs = [p for p in roi_parts if p["type"] == "inputImage"]
        self.assertEqual(len(roi_imgs), 1)
        with Image.open(io.BytesIO(base64.b64decode(roi_imgs[0]["imageUrl"].split(",", 1)[1]))) as roi_im:
            self.assertEqual(roi_im.size, (256, 256))

        none_traj, none_traj_imgs = build_action_trajectory_feedback(
            harness,
            dict(typed_err.data["request"], image_delivery="none"),
            typed_err.data["receipt"],
            {"thread_id": harness.thread_id},
        )
        self.assertTrue(none_traj["available"])
        self.assertEqual(none_traj_imgs, [])

        # 模型主动调用 robot_observe 仍默认返回 3 张相机原图 + 1 张 obs3d ID 叠图
        explicit_obs = harness.handle_tool(
            dict(threadId=harness.thread_id, turnId="t5", callId="c-obs", tool="robot_observe", arguments={})
        )
        self.assertTrue(explicit_obs["success"])
        self.assertEqual(len(_count_images(explicit_obs)), 4)

    def test_program_run_receives_live_obs3d_without_main_model_image_spam(self) -> None:
        """验证 program_run 沙箱内调用 robot_observe 与 robot_action 实时获取最新 object_table 与 scene3d，且不刷屏多余图片。"""
        server = FakeObs3dServer(status="ok")
        self.addCleanup(server.close)
        harness, _, _ = self._make_harness("run-program-obs3d", server_url=server.url)
        harness.programs = ProgramRunner(harness.program_services())
        self.addCleanup(harness.programs.close)

        script = """
import json
import scene3d_helpers as s3
from ni_rua_client import NiRuaClient

client = NiRuaClient()
obs_reply = client.call("robot_observe", {})
obs_data = obs_reply["data"]
assert "object_table" in obs_data, f"missing object_table in data: {list(obs_data)}"
assert "scene3d" in obs_data, f"missing scene3d in data: {list(obs_data)}"
cloud = s3.load(obs_data["scene3d"])
cloud_info = s3.info(cloud)

act_reply = client.action(
    observation_id=obs_data["observation_id"],
    stages=[{"id": "step1", "kind": "move", "arm": "left", "grasp_position_m": [0.20, 0.16, 0.89], "angles_deg": [0, 0, 0]}],
    require_completed=True,
)
obs3d_texts = [json.loads(item["text"]) for item in act_reply["contentItems"] if item["type"] == "inputText" and "object_table" in item["text"]]
assert len(obs3d_texts) == 1
print(json.dumps({
    "rows": len(obs_data["object_table"]["rows"]),
    "points": cloud_info["n"],
    "after_rows": len(obs3d_texts[0]["object_table"]["rows"]),
}))
"""
        prog_wire = harness.handle_tool(
            dict(
                threadId=harness.thread_id,
                turnId="t-prog",
                callId="call-prog-1",
                tool="program_run",
                arguments=dict(source=script, allow_direct_actions=True, timeout_s=15),
            )
        )
        self.assertTrue(prog_wire["success"], prog_wire)
        self.assertEqual(sum(i["type"] == "inputImage" for i in prog_wire["contentItems"]), 0)
        job_summary = json.loads(prog_wire["contentItems"][0]["text"])
        self.assertEqual(job_summary["status"], "succeeded", job_summary)
        # 沙箱内部 obs3d 不生成 overlay 图片，故不产生多余 visual_feedback
        self.assertEqual(job_summary.get("visual_feedback", []), [])
        stdout_raw = job_summary.get("stdout") if "stdout" in job_summary else job_summary["stdout_text"]
        printed = json.loads(stdout_raw.strip())
        self.assertEqual(printed, {"rows": 1, "points": 3, "after_rows": 1})
        self.assertIsNotNone(getattr(harness, "_virtual_spatial_field", None))

    def test_prefetch_cache_isolation_and_pose_sample_focus_roi(self) -> None:
        """验证预取缓存跨调用/跨受众严格隔离、obs3d_action_overlay 下显式 none 零图及 camera_pose_sample 外参 focus_roi。"""
        server = FakeObs3dServer(status="ok")
        self.addCleanup(server.close)
        harness, robot, _ = self._make_harness(
            "run-cache-isolation",
            server_url=server.url,
            extra_overrides={"features": {"obs3d_action_overlay": True}},
        )
        harness.programs = ProgramRunner(harness.program_services())
        self.addCleanup(harness.programs.close)

        # (a) 两步动作 program_run（obs_a -> a1 -> obs_b -> a2 -> obs_c）：最终交付给主模型的增量必须保留完整跨度 obs_a -> obs_c
        server.object_sequence = [
            [_fake_object_row((0.08, 0.02, TABLE_Z + 0.02))],  # obs_a 初始位置
            [_fake_object_row((0.14, 0.02, TABLE_Z + 0.02))],  # a1 后 obs_b (+0.06m)
            [_fake_object_row((0.22, 0.02, TABLE_Z + 0.02))],  # a2 后 obs_c (单步 +0.08m，较 obs_a 累计 +0.14m)
        ]
        init_obs_wire = harness.handle_tool(
            dict(threadId=harness.thread_id, turnId="t-iso-0", callId="call-iso-obs0", tool="robot_observe", arguments={})
        )
        self.assertTrue(init_obs_wire["success"])
        obs_a_id = harness.latest["observation_id"]

        two_step_script = f"""
from ni_rua_client import NiRuaClient
client = NiRuaClient()
r1 = client.action(
    observation_id={obs_a_id!r},
    stages=[{{"id": "s1", "kind": "move", "arm": "left", "grasp_position_m": [0.20, 0.16, 0.89], "angles_deg": [0, 0, 0]}}],
    require_completed=True,
)
obs_b_id = r1["data"]["receipt"]["result_observation_id"]
r2 = client.action(
    observation_id=obs_b_id,
    stages=[{{"id": "s2", "kind": "move", "arm": "left", "grasp_position_m": [0.24, 0.16, 0.89], "angles_deg": [0, 0, 0]}}],
    require_completed=True,
)
print(r2["data"]["receipt"]["result_observation_id"])
"""
        prog_two_wire = harness.handle_tool(
            dict(
                threadId=harness.thread_id,
                turnId="t-iso-1",
                callId="call-iso-prog2",
                tool="program_run",
                arguments=dict(source=two_step_script, allow_direct_actions=True, timeout_s=15),
            )
        )
        self.assertTrue(prog_two_wire["success"], prog_two_wire)
        # 即便开启 obs3d_action_overlay=True，正常完成且 on_anomaly 下依然严格 0 图
        self.assertEqual(sum(i["type"] == "inputImage" for i in prog_two_wire["contentItems"]), 0)
        final_obs3d_item = [
            json.loads(i["text"])
            for i in prog_two_wire["contentItems"]
            if i.get("type") == "inputText" and "object_state_delta" in i.get("text", "")
        ]
        self.assertEqual(len(final_obs3d_item), 1)
        delta_doc = final_obs3d_item[0]["object_state_delta"]
        self.assertEqual(harness._action_state_ledger["last_object_delta"]["before_observation_id"], obs_a_id)
        moved_entry = delta_doc["moved"][0]
        disp_m = float(moved_entry.get("displacement_m", moved_entry.get("delta_norm_m", 0.0)))
        self.assertAlmostEqual(disp_m, 0.14, places=3)

        # (b) program_run 内调用 history_view_observation 后，主模型对同一观测调用 obs3d.attachments(reply="observe") 仍正常获得 3D 叠加图
        shared_bundle = harness.observe({"tool_call_id": "shared-obs"})
        shared_obs_id = shared_bundle["observation_id"]
        hist_script = f"""
from ni_rua_client import NiRuaClient
client = NiRuaClient()
res = client.call("history_view_observation", {{"observation_id": {shared_obs_id!r}}})
assert res["success"]
"""
        hist_prog_wire = harness.handle_tool(
            dict(
                threadId=harness.thread_id,
                turnId="t-iso-2",
                callId="call-iso-hist",
                tool="program_run",
                arguments=dict(source=hist_script, allow_direct_actions=False, timeout_s=15),
            )
        )
        self.assertTrue(hist_prog_wire["success"])
        self.assertIsNone(getattr(harness, "_pending_observe_obs3d", None))
        model_obs_parts = obs3d.attachments(
            harness,
            shared_bundle,
            {"thread_id": harness.thread_id, "tool_call_id": "main-observe-after-prog"},
            reply="observe",
        )
        self.assertEqual(sum(isinstance(p, dict) and p.get("type") == "inputImage" for p in model_obs_parts), 1)

        # (c) 开启 obs3d_action_overlay=True 时，显式指定 image_delivery="none" 依然严格保持 0 原图 + 0 叠加图
        after_none_bundle = harness.observe({"tool_call_id": "after-none"})
        corr_none = {"thread_id": harness.thread_id, "tool_call_id": "act-explicit-none"}
        none_obs_parts = harness.observation_content(
            after_none_bundle,
            corr_none,
            feedback_action={
                "type": "BatchAction",
                "based_on_observation_id": shared_obs_id,
                "image_delivery": "none",
            },
            feedback_receipt={"status": "completed", "reason": None},
            structured=True,
        )
        none_3d_parts = obs3d.action_attachments(
            harness,
            after_none_bundle,
            corr_none,
            before_observation_id=shared_obs_id,
        )
        combined_none_images = [
            p for p in [*none_obs_parts, *none_3d_parts]
            if isinstance(p, dict) and p.get("type") == "inputImage"
        ]
        self.assertEqual(combined_none_images, [])

        # (d) 当 T_world_camera 仅存在于 camera_pose_sample（camera_calibration 中无外参）时，image_delivery="focus_roi" 正常输出 256×256 裁剪
        robot.calibration_has_extrinsics = False
        robot.q_state["left"] = [-1.3, 0.0, 0.0, 0.0, 0.0, 0.0]
        pose_only_bundle = harness.observe({"tool_call_id": "pose-only-obs"})
        cal_stored = json.loads(harness.store.resolve_artifact(pose_only_bundle["cameras"][0]["calibration_ref"])[1])
        self.assertNotIn("T_world_camera", cal_stored["cameras"]["cam_head"])
        pose_roi_parts = harness.observation_content(
            pose_only_bundle,
            {"thread_id": harness.thread_id, "tool_call_id": "act-pose-only-roi"},
            feedback_action={"type": "BatchAction", "image_delivery": "focus_roi", "stages": [{"arm": "left"}]},
            feedback_receipt={"status": "completed", "reason": None},
        )
        pose_roi_imgs = [p for p in pose_roi_parts if p.get("type") == "inputImage"]
        self.assertEqual(len(pose_roi_imgs), 1)
        with Image.open(io.BytesIO(base64.b64decode(pose_roi_imgs[0]["imageUrl"].split(",", 1)[1]))) as crop_im:
            self.assertEqual(crop_im.size, (256, 256))

    def test_native_audit_passes_for_zero_image_multistep_settle_and_anomaly_turns(self) -> None:
        """端到端验证：0 图回执跨轮引用 + 带多步 bounded_settle 保持动作 + 异常附图全链路 100% 通过 native audit。"""
        server = FakeObs3dServer(status="ok")
        self.addCleanup(server.close)
        run_dir = self.root / "native-audit-run"
        robot = BatchCapableDualArmRobot(self.root / "native-audit-captures", data_origin="robodojo_runtime")
        config = Config().with_overrides(
            {
                "run": {"task": "stack_bowls", "benchmark": True},
                "images": {"action_feedback_policy": "on_anomaly"},
                "features": {
                    "terminal": False,
                    "external_perception": True,
                    "vggt_endpoint": server.url,
                    "vggt_support_z_m": TABLE_Z,
                    "held_perception": False,
                    "action_trajectory_delivery": "diagnostic",
                    "motion_review": False,
                    "task_scheduling": False,
                    "grasp_memory_dir": str(self.root / "native-audit-mem"),
                },
            }
        )
        wb = Workbench(run_dir, config=config, robot=robot)
        self.addCleanup(lambda: wb.store.close() if getattr(wb, "store", None) else None)
        self.addCleanup(lambda: wb.grasp_memory_store.close() if getattr(wb, "grasp_memory_store", None) else None)
        self.addCleanup(lambda: wb.run_lock.close() if getattr(wb, "run_lock", None) else None)
        _attach_fake_codex_session(wb)
        asyncio.run(wb.start(None, benchmark=True))
        harness: Harness = wb.harness

        def _invoke_native_tool(turn_id: str, call_id: str, item_id: str, tool: str, arguments: dict) -> dict:
            wb.store.raw_codex(
                "received",
                {
                    "method": "item/started",
                    "params": {"threadId": harness.thread_id, "item": {"id": item_id, "type": "dynamicToolCall"}},
                },
            )
            harness.call_items[call_id] = item_id
            wire = harness.handle_tool(
                {"threadId": harness.thread_id, "turnId": turn_id, "callId": call_id, "tool": tool, "arguments": arguments}
            )
            wb.store.raw_codex("sent", {"id": f"rpc-{call_id}", "result": wire})
            return wire

        # Turn 1：提交计划 + 基于首轮内联 obs_1 执行正常多步动作 -> 产出 0 图回执 obs_2
        plan_wire = _invoke_native_tool(
            "turn-1",
            "call-plan",
            "item-plan",
            "task_plan",
            {"operation": "submit", "observation_id": "obs_1", "plan": "Grasp and settle.", "ledger": []},
        )
        self.assertTrue(plan_wire["success"])

        stages_t1 = [
            dict(id="reach", kind="move", arm="left", grasp_position_m=[0.20, 0.16, 0.88], angles_deg=[0, 0, 0]),
            dict(id="descend", kind="move", arm="left", depends_on=["reach"], grasp_position_m=[0.20, 0.16, 0.84], angles_deg=[0, 0, 0]),
        ]
        wire_t1 = _invoke_native_tool(
            "turn-1",
            "call-act-1",
            "item-act-1",
            "robot_action",
            {"mode": "direct", "observation_id": "obs_1", "stages": stages_t1},
        )
        self.assertTrue(wire_t1["success"])
        self.assertEqual(sum(i["type"] == "inputImage" for i in wire_t1["contentItems"]), 0)
        from support.reply_defaults import observation_value
        t1_obs_text = observation_value(wire_t1["contentItems"])
        obs_after_t1 = t1_obs_text["bundle"]["observation_id"]
        wb.store.raw_codex(
            "received",
            {
                "method": "turn/completed",
                "params": {"threadId": harness.thread_id, "turn": {"id": "turn-1", "status": "completed", "items": []}},
            },
        )

        # Turn 2：引用 0 图回执的 obs_after_t1，并附带多步 settle（触发 2 次内部 hold 子动作产生未直接交付的过渡观测！）
        wire_t2 = _invoke_native_tool(
            "turn-2",
            "call-act-2",
            "item-act-2",
            "robot_action",
            {
                "mode": "direct",
                "observation_id": obs_after_t1,
                "stages": [dict(id="lift", kind="move", arm="left", grasp_position_m=[0.20, 0.16, 0.92], angles_deg=[0, 0, 0])],
                "settle": {
                    "camera_id": "cam_head",
                    "roi": [120, 100, 200, 180],
                    "max_native_steps": 10,
                    "sample_steps": 5,
                    "stable_samples": 2,
                    "max_motion_px": 0.5,
                },
            },
        )
        self.assertTrue(wire_t2["success"])
        self.assertEqual(sum(i["type"] == "inputImage" for i in wire_t2["contentItems"]), 0)
        t2_header = json.loads(wire_t2["contentItems"][0]["text"])
        self.assertGreaterEqual(len(t2_header["settling"]["wait_action_ids"]), 2)
        t2_obs_text = observation_value(wire_t2["contentItems"])
        obs_after_t2 = t2_obs_text["bundle"]["observation_id"]

        # Turn 3：引用 settle 后的最终 0 图观测 obs_after_t2 执行动作并遭遇受阻 -> 自动附单面板 640×480 轨迹图与原图
        robot.next_status = "stopped"
        robot.next_reason = "tracking_error"
        wire_t3 = _invoke_native_tool(
            "turn-3",
            "call-act-3",
            "item-act-3",
            "robot_action",
            {
                "mode": "direct",
                "observation_id": obs_after_t2,
                "stages": [dict(id="insert", kind="move", arm="left", grasp_position_m=[0.22, 0.16, 0.82], angles_deg=[0, 0, 0])],
            },
        )
        self.assertTrue(wire_t3["success"])
        self.assertGreaterEqual(sum(i["type"] == "inputImage" for i in wire_t3["contentItems"]), 4)
        wb.store.raw_codex(
            "received",
            {
                "method": "turn/completed",
                "params": {
                    "threadId": harness.thread_id,
                    "turn": {
                        "id": "turn-2",
                        "status": "completed",
                        "items": [{"type": "agentMessage", "text": "NI_RUA_CONTINUITY_OK"}],
                    },
                },
            },
        )

        # 执行完整 native 证据链审计（含连续性校验）
        report = audit(run_dir, profile="native", require_continuity=True)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["failures"], [])
        # 总物理动作数：Turn1(1) + Turn2(1 主动作 + >=2 内部 settle hold) + Turn3(1) >= 5
        self.assertGreaterEqual(len(report["actions"]), 5)

        # 反向防逃逸校验：若新工具调用（item-illegal）跨轮执行了引用未交付 settle 过渡观测的动作，audit 必须严格报错拦截！
        all_requests = wb.store.record_items("action_request")
        all_receipts = wb.store.record_items("action_receipt")
        settle_hold_ids = [
            aid
            for aid, req in all_requests.items()
            if isinstance(req.get("path"), dict) and req["path"].get("kind") == "hold"
        ]
        self.assertGreaterEqual(len(settle_hold_ids), 2)
        template_hold_id = settle_hold_ids[0]
        undelivered_intermediate_obs = all_requests[template_hold_id]["based_on_observation_id"]
        wb.store.raw_codex(
            "received",
            {
                "method": "item/started",
                "params": {"threadId": harness.thread_id, "item": {"id": "item-illegal", "type": "dynamicToolCall"}},
            },
        )
        illegal_req = dict(
            deepcopy(all_requests[template_hold_id]),
            action_id="illegal-cross-turn-hold",
            based_on_observation_id=undelivered_intermediate_obs,
        )
        illegal_rec = deepcopy(all_receipts[template_hold_id])
        wb.store.save("action_request", "illegal-cross-turn-hold", illegal_req)
        wb.store.save("action_receipt", "illegal-cross-turn-hold", illegal_rec)
        wb.store.event(
            "action.requested",
            illegal_req,
            {"action_id": "illegal-cross-turn-hold", "thread_id": harness.thread_id, "turn_id": "turn-4", "item_id": "item-illegal"},
        )
        tampered_report = audit(run_dir, profile="native")
        self.assertFalse(tampered_report["passed"])
        self.assertIn("input was not delivered:illegal-cross-turn-hold", tampered_report["failures"])



# --- Folded from tests/unit/test_prefix_cache_and_verbosity.py ---

def _canonical_bytes(value) -> bytes:
    """按确定性紧凑 JSON 编码，用于逐字节比对请求前缀一致性。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _raw_wire_bytes(value) -> bytes:
    """按实际键顺序紧凑编码，直接核验线上发送字节无冷/暖漂移。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


class PrefixCacheStabilityTests(unittest.TestCase):
    """验证跨轮新注册 actor_id、上下文压缩及会话恢复不会回溯改写早期请求字节。"""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.identity = dict(
            run_id="prefix-cache-test",
            episode_id="ep-01",
            env_instance_id="env-0123456789abcdef",
            env_generation=1,
        )
        self.store = Store(self.root / "run", self.identity)
        self.addCleanup(self.store.close)
        self.codex_home = self.root / "codex-source"
        self.codex_home.mkdir()
        (self.codex_home / "config.toml").write_text(
            'model_provider = "fixture"\n'
            '[model_providers.fixture]\n'
            'base_url = "http://127.0.0.1:9/v1"\n'
        )
        self.config = Config.from_dict({"model": {"codex_home": str(self.codex_home)}})

    def _build_turn1_request(self, artifact_ref: str) -> dict:
        raw_sha = "a" * 64
        obs_long = "observation-1122334455667788"
        act_long = "action-99aabbccddeeff00"
        # 包含未知前缀裸词（12 位与 32 位 hex）、已知前缀长 ID、artifact 引用、clock_domain 及原始哈希
        return {
            "model": "gpt-6.1-sol",
            "instructions": f"System instructions sha256={raw_sha} with frame-1234abcd1234.",
            "tools": [
                {
                    "type": "function",
                    "name": "robot_observe",
                    "description": f"Capture observation build_commit={raw_sha} ref token frame-1234abcd1234",
                    "parameters": {
                        "type": "object",
                        "properties": {"observation_id": {"type": "string"}},
                        "additionalProperties": False,
                    },
                }
            ],
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                f"Inspect {obs_long} in env-0123456789abcdef/generation-1 "
                                f"candidate frame-1234abcd1234 and stage-deadbeefcafe01 "
                                f"and token-0123456789abcdef0123456789abcdef sha256={raw_sha} {artifact_ref}"
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": "data:image/webp;base64," + ("QUJD" * 512),
                        },
                    ],
                },
                {
                    "type": "function_call",
                    "call_id": "call_turn1",
                    "name": "robot_observe",
                    "arguments": json.dumps({"note": "check frame-1234abcd1234"}),
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_turn1",
                    "output": json.dumps({
                        "observation_id": obs_long,
                        "last_action_id": act_long,
                        "source_url": "https://internal.example/repo.py",
                        "sha256": raw_sha,
                        "note": "captured frame-1234abcd1234 and stage-deadbeefcafe01",
                    }),
                },
            ],
        }

    def test_late_registered_actor_ids_never_mutate_historical_prefix_or_resumed_egress(self):
        artifact_ref = self.store.put_artifact(b"payload", "fixture", "text/plain")
        planning = TaskPlanning(self.store)
        planning.call(
            {"plan": "Grasp and verify target.", "ledger": []},
            self.identity,
            dict(self.identity, observation_id="obs_1"),
            {},
        )
        session = CodexSession(self.store.root, self.store, lambda _: {}, config=self.config, task_planning=planning)
        session.thread_id = "actor-thread-1"
        self.addCleanup(session.close)

        turn1_req = self._build_turn1_request(artifact_ref)
        turn1_out = session._present_model_request(turn1_req)
        self.assertEqual(metadata_findings(json.dumps(turn1_out, ensure_ascii=False)), [])

        turn1_prefix_bytes = [_canonical_bytes(item) for item in turn1_out["input"]]
        turn1_instructions_bytes = _canonical_bytes(turn1_out["instructions"])
        turn1_tools_bytes = _canonical_bytes(turn1_out["tools"])

        # 模拟后续轮次工具回执将先前出现过的未知前缀裸词正式注册进 actor_id 表
        presented_receipt = session.presentation.present({
            "frame_id": "frame-1234abcd1234",
            "stages": {
                "stage-deadbeefcafe01": {"status": "completed"},
                "token-0123456789abcdef0123456789abcdef": {"status": "completed"},
            },
        })
        self.assertRegex(presented_receipt["frame_id"], r"^id_\d+$")
        # 确认沙箱代码侧直接调用 present_identifiers 仍保持短 ID 绑定兼容
        self.assertEqual(
            ActorPresentation(self.store).present_identifiers("use frame-1234abcd1234"),
            f"use {presented_receipt['frame_id']}",
        )

        # 第 2 轮请求扩展历史：验证前 N 项命中幂等缓存（跳过重复文本清洗）且字节 100% 恒等
        turn2_req = deepcopy(turn1_req)
        turn2_req["input"].append({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": f"Follow up sha256={'b' * 64} on action-ffeeddccbbaa9988"}],
        })
        with mock.patch.object(
            session.presentation,
            "present_egress_text",
            wraps=session.presentation.present_egress_text,
        ) as spy_egress_text:
            turn2_out = session._present_model_request(turn2_req)
            # 仅新追加的第 4 个条目触发 1 次文本清洗，前 3 个历史条目及 instructions/tools 全部直接命中缓存
            self.assertEqual(spy_egress_text.call_count, 1)

        self.assertEqual([_canonical_bytes(item) for item in turn2_out["input"][:3]], turn1_prefix_bytes)
        self.assertEqual(_canonical_bytes(turn2_out["instructions"]), turn1_instructions_bytes)
        self.assertEqual(_canonical_bytes(turn2_out["tools"]), turn1_tools_bytes)
        self.assertEqual(metadata_findings(json.dumps(turn2_out, ensure_ascii=False)), [])

        # 验证原地修改返回结果不会污染内部条目缓存
        turn2_out["input"][0]["content"][0]["text"] = "MUTATED_BY_CALLER"
        turn2_repeat = session._present_model_request(turn2_req)
        self.assertEqual([_canonical_bytes(item) for item in turn2_repeat["input"][:3]], turn1_prefix_bytes)

        # 验证 /responses/compact 路径同样保持历史前缀字节 100% 恒等且无哈希泄露
        compact_identity = dict(source="actor", native_thread_id=session.thread_id, endpoint="/responses/compact")
        compact_req = session._task_decision_context(turn2_req, compact_identity)
        compact_out = session._present_model_request(compact_req)
        self.assertEqual([_canonical_bytes(item) for item in compact_out["input"][:3]], turn1_prefix_bytes)
        self.assertEqual(metadata_findings(json.dumps(compact_out, ensure_ascii=False)), [])

        # 验证网关重启后的恢复会话（全新 CodexSession 冷缓存 + 已持久化的 actor_id 表）仍保持第 1 轮字节 100% 恒等
        resumed_session = CodexSession(
            self.root / "resumed-session-dir",
            self.store,
            lambda _: {},
            config=self.config,
            task_planning=planning,
        )
        self.addCleanup(resumed_session.close)
        resumed_out = resumed_session._present_model_request(turn2_req)
        self.assertEqual([_canonical_bytes(item) for item in resumed_out["input"][:3]], turn1_prefix_bytes)
        self.assertEqual(_canonical_bytes(resumed_out["instructions"]), turn1_instructions_bytes)
        self.assertEqual(_canonical_bytes(resumed_out["tools"]), turn1_tools_bytes)
        self.assertEqual(metadata_findings(json.dumps(resumed_out, ensure_ascii=False)), [])

    def test_egress_security_guards_reject_forbidden_payloads_without_caching(self):
        session = CodexSession(self.store.root, self.store, lambda _: {}, config=self.config)
        self.addCleanup(session.close)
        bad_media = {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "https://evil.invalid/x.png"}]}]}
        with self.assertRaisesRegex(ValueError, "sandbox_remote_media_url_disabled"):
            session._present_model_request(bad_media)
        egress_cache = getattr(session, "_egress_cache", None)
        if egress_cache is not None:
            self.assertEqual(egress_cache.bucket_size("items"), 0)
            self.assertEqual(egress_cache.text_entries, 0)
        else:
            self.assertEqual(len(session._egress_item_cache), 0)

    def test_unsorted_wire_bytes_key_order_schema_determinism_and_lru_hot_prefix(self):
        from ni_rua.cortex.session.egress_cache import EgressProjectionCache
        from ni_rua.cortex.session.wire.openai_responses import present_responses_request

        presentation = ActorPresentation(self.store)
        cache = EgressProjectionCache(max_item_entries=2)
        item_ab = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}], "extra": {"a": 1, "b": 2}}
        item_ba = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}], "extra": {"b": 2, "a": 1}}
        tools = [
            {
                "type": "function",
                "name": "robot_observe",
                "description": "Observe frame-1234abcd1234",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "string", "description": "Ref frame-1234abcd1234"}},
                },
            }
        ]
        req1 = {"model": "gpt-6.1-sol", "tools": deepcopy(tools), "input": [deepcopy(item_ab), deepcopy(item_ba)]}
        out1 = present_responses_request(req1, presentation, deferred_tools=frozenset(), namespace="ni_rua", cache=cache)
        # 未排序字节直比：键顺序必须原样保留，不得因缓存命中串成同一键序
        self.assertEqual(list(out1["input"][0]["extra"]), ["a", "b"])
        self.assertEqual(list(out1["input"][1]["extra"]), ["b", "a"])
        self.assertNotEqual(_raw_wire_bytes(out1["input"][0]), _raw_wire_bytes(out1["input"][1]))

        # 后续轮次注册未知前缀裸词，冷缓存重算也必须与暖缓存字节 100% 恒等
        presentation.present({"frame_id": "frame-1234abcd1234"})
        out_cold = present_responses_request(req1, presentation, deferred_tools=frozenset(), namespace="ni_rua", cache=None)
        self.assertEqual(_raw_wire_bytes(out1), _raw_wire_bytes(out_cold))

        # 验证条目缓存为 LRU：反复命中的热前缀 item_ab 在新条目涌入时不被淘汰，而是淘汰较冷的 item_ba
        item_c = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "cold-c"}]}
        present_responses_request({"input": [deepcopy(item_ab)]}, presentation, deferred_tools=frozenset(), namespace="ni_rua", cache=cache)
        present_responses_request({"input": [deepcopy(item_c)]}, presentation, deferred_tools=frozenset(), namespace="ni_rua", cache=cache)
        with mock.patch.object(presentation, "present_egress_text", wraps=presentation.present_egress_text) as spy_text:
            present_responses_request({"input": [deepcopy(item_ab)]}, presentation, deferred_tools=frozenset(), namespace="ni_rua", cache=cache)
            self.assertEqual(spy_text.call_count, 0)


class ModelVerbosityConfigTests(unittest.TestCase):
    """验证默认 model_verbosity 与推理深度均保持 high，忽略宿主配置且支持显式与配置覆盖。"""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "host-codex-home"
        self.source.mkdir()
        # 模拟宿主配置写有 model_verbosity = "low" 与 model_reasoning_effort = "low"
        (self.source / "config.toml").write_text(
            'model_provider = "fixture"\n'
            'model_reasoning_effort = "low"\n'
            'model_verbosity = "low"\n'
            '[model_providers.fixture]\n'
            'base_url = "http://127.0.0.1:9/v1"\n'
        )
        self.store = Store(
            self.root / "run",
            dict(run_id="verbosity-test", episode_id="ep", env_instance_id="env", env_generation=1),
        )
        self.addCleanup(self.store.close)
        relay = mock.MagicMock(base_url="http://127.0.0.1:54321")
        relay.start.return_value = relay
        patcher = mock.patch("ni_rua.cortex.agents.codex.session.ModelRequestProxy", return_value=relay)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_default_verbosity_is_high_while_preserving_high_reasoning_effort(self):
        self.assertEqual(DEFAULT_MODEL_VERBOSITY, "high")
        config = Config.from_dict({
            "features": {"sandbox_enabled": False},
            "model": {"name": "gpt-6.1-sol", "effort": "high", "codex_home": str(self.source)},
        })
        session = CodexSession(self.root / "default-session", self.store, lambda _: {}, config=config)
        self.addCleanup(session.close)
        self.assertEqual(session.verbosity, "high")
        self.assertEqual(session.effort, "high")

        session._prepare()
        generated = tomllib.loads((session.codex_home / "config.toml").read_text())
        self.assertEqual(generated["model_verbosity"], "high")
        self.assertEqual(generated["model_reasoning_effort"], "high")

    def test_explicit_and_configured_verbosity_overrides_and_validation(self):
        # 1. 通过统一 Config.from_dict 指定 model.verbosity = "medium"
        base_config = Config.from_dict({
            "features": {"sandbox_enabled": False},
            "model": {
                "name": "gpt-6.1-sol",
                "effort": "high",
                "verbosity": "medium",
                "codex_home": str(self.source),
            },
        })
        self.assertEqual(base_config.model.verbosity, "medium")
        configured_session = CodexSession(self.root / "configured-session", self.store, lambda _: {}, config=base_config)
        self.addCleanup(configured_session.close)
        configured_session._prepare()
        configured_toml = tomllib.loads((configured_session.codex_home / "config.toml").read_text())
        self.assertEqual(configured_toml["model_verbosity"], "medium")

        # 2. 显式构造参数优先于配置项
        explicit_session = CodexSession(
            self.root / "explicit-session",
            self.store,
            lambda _: {},
            config=base_config,
            verbosity="low",
        )
        self.addCleanup(explicit_session.close)
        explicit_session._prepare()
        explicit_toml = tomllib.loads((explicit_session.codex_home / "config.toml").read_text())
        self.assertEqual(explicit_toml["model_verbosity"], "low")
        self.assertEqual(explicit_toml["model_reasoning_effort"], "high")

        # 3. 非法详略度枚举直接报错
        with self.assertRaisesRegex(ValueError, "invalid_model_verbosity"):
            CodexSession(self.root / "bad-session", self.store, lambda _: {}, config=base_config, verbosity="verbose")


class SameSimStepRenderDedupTests(unittest.TestCase):
    """验证运动期间同一 sim_step 录屏与进度/终态观测共享单次 GPU 渲染与图像编码。"""

    def _make_adapter(self, root: Path, profile: str = "rgb_proprio_v1"):
        adapter = object.__new__(RoboDojoAdapter)
        adapter.np = np
        adapter.directory = root / "private/robot"
        adapter.directory.mkdir(parents=True, exist_ok=True)
        adapter.observation_profile = profile
        adapter.image_settings = image_settings({})
        adapter.legacy_png_ipc = False
        adapter.recording = False
        adapter.recorded_frames = []
        adapter._step_capture_cache = None
        adapter.control_step = 0
        adapter.camera_calibration = None
        adapter.versions = {}
        adapter.scene_truth = mock.MagicMock()

        counters = {"render": 0, "get_obs": 0}
        rng = np.random.default_rng(42)
        color = rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
        depth = np.full((32, 48), 0.65, dtype=np.float32)

        def render_for_capture():
            counters["render"] += 1

        def get_obs(_indices):
            counters["get_obs"] += 1
            return [{
                "instruction": "Synthetic motion capture dedup test.",
                "vision": {
                    name: {"color": color, "depth": depth}
                    for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")
                },
            }]

        adapter.obs = SimpleNamespace(render_for_capture=render_for_capture, get_obs=get_obs)
        adapter.env = SimpleNamespace(
            dt=0.004,
            sim=SimpleNamespace(sim=SimpleNamespace(current_time=0.0)),
            camera_manager=SimpleNamespace(
                camera_names=[["cam_head", "cam_left_wrist", "cam_right_wrist"]],
                cameras=[[SimpleNamespace(get_clipping_range=lambda: [0.05, 5.0]) for _ in range(3)]],
            ),
        )
        arm_state = {
            "joint_names": [f"j{i}" for i in range(6)],
            "q_rad": [0.0] * 6,
            "qdot_rad_s": [0.0] * 6,
            "source_kind": "measurement",
            "ee_pose": {
                "position_m": [0.2, 0.0, 0.3],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "frame_id": "base_link",
                "tcp_config_id": "arx_x5_tcp",
                "source_kind": "measurement",
            },
            "gripper": {
                "goal_normalized": 1.0,
                "goal_source_kind": "command",
                "command_normalized": 1.0,
                "command_source_kind": "command",
                "joint_names": ["g1", "g2"],
                "position_m": [0.04, 0.04],
                "position_source_kind": "measurement",
                "contact_feedback": None,
                "contact_feedback_reason": "none",
            },
        }
        adapter.state = lambda: {
            "arms": {"left": deepcopy(arm_state), "right": deepcopy(arm_state)},
            "frame_id": "base_link",
            "tcp_config_id": "arx_x5_tcp",
            "sim_step": adapter.control_step,
            "step_domain": "adapter_control_step",
        }
        return adapter, counters

    def test_tracked_step_and_dispatch_share_single_render_and_encoding_per_sim_step(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(directory)
        store = Store(root / "run", dict(run_id="dedup", episode_id="ep", env_instance_id="env", env_generation=1))
        self.addCleanup(store.close)
        adapter, counters = self._make_adapter(store.root)
        harness = Harness(store, adapter)

        with mock.patch(
            "ni_rua.robot.backends.robodojo.robot_worker.encode_image",
            wraps=encode_image,
        ) as spy_encode:
            # 1. 启动录屏（sim_step=0 触发 video_only=True）紧接动作前正式观测（sim_step=0）
            adapter.record(True)
            before_obs = adapter.observe()
            harness.publish_observation(before_obs)
            self.assertEqual(counters["render"], 1)
            self.assertEqual(counters["get_obs"], 1)
            # 3 个相机仅编码 1 轮（共 3 次），消除第 2 轮重复编码
            self.assertEqual(spy_encode.call_count, 3)

            # 2. 模拟运动推进 5 个原生步至 sim_step=50：tracked_step 先调 video_only=True，dispatch 再调进度 observe()，随后动作结束调终态 observe()
            for _ in range(5):
                adapter.control_step += 10
                adapter.env.sim.sim.current_time = adapter.control_step * 0.004
                if adapter.recording and adapter.control_step % 50 == 0:
                    adapter.observe(video_only=True)
            self.assertEqual(adapter.control_step, 50)
            self.assertEqual(counters["render"], 2)

            progress_obs = adapter.observe()
            terminal_obs = adapter.observe()
            harness.publish_observation(progress_obs)
            harness.publish_observation(terminal_obs)

            # sim_step=50 全程（录屏帧 + 进度观测 + 终态观测）仅发生 1 次 GPU 渲染与 1 轮 JPEG 编码！
            self.assertEqual(counters["render"], 2)
            self.assertEqual(counters["get_obs"], 2)
            self.assertEqual(spy_encode.call_count, 6)

        # 验证导出录屏后临时录屏路径按原契约清理，而正式进度/终态观测图像文件完整保留在磁盘
        video_manifest = adapter.save_video()
        self.assertEqual(video_manifest["temporary_frame_cleanup_errors"], [])
        for obs in (before_obs, progress_obs, terminal_obs):
            for cam in obs["cameras"]:
                self.assertTrue(camera_path(cam).is_file())

    def test_rgbd_upgrade_from_video_frame_preserves_depth_and_scene_truth_without_second_render(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter, counters = self._make_adapter(Path(directory), profile=DEPTH_ACTOR_PROFILE)
            adapter.recording = True
            adapter.control_step = 50

            # video_only=True 只保存 RGB 不写深度与真值
            adapter.observe(video_only=True)
            self.assertEqual(counters["render"], 1)
            adapter.scene_truth.record.assert_not_called()

            # 同一步正式观测复用已渲染缓存，不二次渲染，但补全深度归档与 scene_truth 记录
            formal = adapter.observe(video_only=False)
            self.assertEqual(counters["render"], 1)
            self.assertEqual(counters["get_obs"], 1)
            require_actor_observation(formal, DEPTH_ACTOR_PROFILE)
            for cam in formal["cameras"]:
                self.assertTrue(Path(cam["depth_path"]).is_file())
            adapter.scene_truth.record.assert_called_once()



if __name__ == "__main__":
    unittest.main()
