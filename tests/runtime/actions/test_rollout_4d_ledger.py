"""Phase 2 单元测试：运动期 4D 过程轨迹（共动/滑脱/原地旋拧/关键帧）、持久 ID 跨桌面差分、
on_anomaly 滑脱自动附图、会话回收 /session_close 与 codex-cli <5000 字节红线校验。

运行方式：
PYTHONPATH=tests:. mamba run -n ni-rua python -m unittest tests/unit/test_p2_rollout_4d_ledger.py -v
"""
from __future__ import annotations

import asyncio
import base64
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import re
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from PIL import Image
from ni_rua.apps.evaluation.audit import audit
from ni_rua.apps.gateway.workbench import Workbench
from ni_rua.core.config import Config
from ni_rua.core.contracts import INSPECT_STEP_LIMIT, canonical
from ni_rua.core.presentation.actor_presentation import metadata_findings
from ni_rua.core.presentation.reply_budget import fit_reply_report
from ni_rua.core.presentation.reply_compaction import DETAIL_KEY, unify_detail_entries
from ni_rua.core.storage.store import Store
from ni_rua.cortex.context.actor_instructions import (
    CALIBRATED_OBSERVATION_INSTRUCTIONS,
    INSTRUCTIONS,
    OPERATION_NOTES,
    instructions,
)
from ni_rua.runtime.actions.rollout_tracker import (
    ProcessRolloutTracker,
    _fingertip_local_flow_px,
    build_process_trajectory,
)
from ni_rua.robot.backends.robodojo.robodojo import FRAME_ID, TCP_CONFIG_ID
from ni_rua.robot.model import roboprobe_pose
from ni_rua.runtime.actions.action_state_ledger import compute_object_table_delta
from ni_rua.runtime.actions.compact_trajectory_summary import build_compact_trajectory_summary
from ni_rua.runtime.actions.robot_action_schema import ROBOT_ACTION_TOOL
from ni_rua.runtime.harness import Harness
from ni_rua.runtime.programs.runner import ProgramRunner
from support.host_capabilities import expect, program_isolation

NI_RUA_TEST_TIER = "component"


TABLE_Z = 0.765
CAMERA_K = [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
T_HEAD = [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.35], [0.0, 0.0, 0.0, 1.0]]

RUN_IDENTITY = {
    "run_id": "run-p2-ledger",
    "episode_id": "ep-p2-initial",
    "env_instance_id": "env-p2-ledger",
    "env_generation": 1,
}


def _urdf_6dof_model() -> dict:
    """构造带双指夹爪（joint7/joint8）的 6 自由度公开运动学模型。"""
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
    return dict(base_link="base_link", joints=joints, source_sha256="unit-test-p2-model")


def _synthetic_npz_bytes() -> bytes:
    """构造最小合法 scene3d 压缩点云字节流。"""
    xyz = np.array([[0.0, 0.0, TABLE_Z], [0.12, 0.04, TABLE_Z + 0.03]], dtype=np.float32)
    conf = np.array([220, 240], dtype=np.uint8)
    uv = np.array([[320, 240], [380, 220]], dtype=np.uint16)
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


def _quat_xyzw_from_angles_deg(pitch_deg: float, roll_deg: float, yaw_deg: float) -> list[float]:
    """按 roboprobe_pose 外旋约定将 (pitch, roll, yaw) 度数转为 link6 单位四元数 [x, y, z, w]。"""
    pose_wxyz = roboprobe_pose.values_to_pose(
        dict(x=0.0, y=0.0, z=0.85, pitch_deg=pitch_deg, roll_deg=roll_deg, yaw_deg=yaw_deg)
    )
    _, _, _, qw, qx, qy, qz = pose_wxyz
    return [float(qx), float(qy), float(qz), float(qw)]


class FakePerceptionServer:
    """本地轻量 HTTP 假服务，模拟 VGGT /object_table、/points 与 /session_close 端点。"""

    def __init__(self, *, session_close_delay_s: float = 0.0) -> None:
        self.session_close_delay_s = session_close_delay_s
        self.closed_sessions: list[str] = []
        self._lock = threading.Lock()
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(raw.decode("utf-8") or "{}")
                if self.path == "/session_close":
                    if parent.session_close_delay_s > 0:
                        time.sleep(parent.session_close_delay_s)
                    with parent._lock:
                        sid = payload.get("session_id")
                        if isinstance(sid, str):
                            parent.closed_sessions.append(sid)
                    body = json.dumps({"status": "ok", "closed": True}).encode("utf-8")
                    try:
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    return
                if self.path == "/object_table":
                    obj = dict(
                        centre_m=[0.12, 0.04, TABLE_Z + 0.02],
                        top_z_m=TABLE_Z + 0.04,
                        bottom_z_m=TABLE_Z,
                        length_m=0.06,
                        width_m=0.04,
                        yaw_deg=10.0,
                        axis_tilt_deg=89.0,
                        elongation=1.5,
                        support="table",
                        flags=[],
                        quality="ok",
                        label_px=[380, 220],
                        contour_px=[[370, 214], [390, 214], [390, 226], [370, 226]],
                        mask_px=200,
                        n_points=120,
                        iou_pred=0.92,
                        conf_rank_p50=0.75,
                        contact_gap_m=0.0,
                        cluster_frac=1.0,
                    )
                    body = json.dumps(
                        dict(
                            status="ok",
                            support_z_m=TABLE_Z,
                            table_inliers=4000,
                            n_objects=1,
                            objects=[obj],
                            tracking=dict(session_id=payload.get("session_id"), persistent_ids=True),
                        )
                    ).encode("utf-8")
                else:
                    body = json.dumps(
                        dict(
                            status="ok",
                            npz_b64=base64.b64encode(_synthetic_npz_bytes()).decode("ascii"),
                            n_points=2,
                            support_z_m=TABLE_Z,
                            voxel_m=0.01,
                        )
                    ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args) -> None:
                return

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_port}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


class RolloutCapableRobot:
    """支持多步实测夹爪孔径序列、过程观测帧回调与 reset 的双臂仿真假后端。"""

    observation_profile = "rgb_proprio_calibrated_v1"

    def __init__(self, work_dir: Path, *, data_origin: str = "unit_test") -> None:
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.data_origin = data_origin
        self.sim_step = 0
        self.q_state = {"left": [0.05] * 6, "right": [-0.05] * 6}
        self.jaw_half_m = {"left": 0.044, "right": 0.044}
        self.jaw_goal = {"left": 1.0, "right": 1.0}
        self.ee_pos = {"left": [0.18, 0.14, 0.88], "right": [0.18, -0.14, 0.88]}
        self.ee_angles = {"left": (0.0, 0.0, 0.0), "right": (0.0, 0.0, 0.0)}
        self._plans: dict[str, dict] = {}
        self._plan_seq = 0
        # 每次 execute 可注入逐原生步的左臂实测全开度序列（单位 m）与可选过程观测步
        self.scripted_left_apertures_m: list[float] | None = None
        self.emit_progress_at_step: int | None = None
        self.raise_Reset_error: bool = False
        self.steps_per_stage = 2  # native steps each stage of a planned batch takes

    def start(self) -> dict:
        return dict(
            data_origin=self.data_origin,
            backend="robodojo_curobo",
            robot_model_id="arx-x5",
            frame_id=FRAME_ID,
            tcp_config_id=TCP_CONFIG_ID,
            limits_profile="robodojo-native-v1",
            path_kinds=["backend_free", "hold"],
            max_waypoints=4,
            dual_arm=True,
            gripper_after_motion=True,
            reset=True,
            observer_video=False,
            bounded_wait=dict(max_native_steps=50),
            action_budget=dict(native_step_limit=200),
            batch_actions=dict(available=True, max_stages=32),
            unified_action=dict(available=True),
            action_speed=dict(
                available=True,
                speed_scale=dict(min=0.25, max=2.0, default=1.0),
                gripper_speed_scale=dict(min=0.25, max=1.0, default=1.0),
            ),
        )

    def _state_snapshot(self) -> dict:
        arms = {}
        for arm in ("left", "right"):
            p, r, y = self.ee_angles[arm]
            half_ap = max(0.0, float(self.jaw_half_m[arm]))
            arms[arm] = dict(
                source_kind="measurement",
                joint_names=[f"joint{i}" for i in range(1, 7)],
                q_rad=list(self.q_state[arm]),
                qdot_rad_s=[0.0] * 6,
                ee_pose=dict(
                    source_kind="measurement",
                    frame_id=FRAME_ID,
                    tcp_config_id=TCP_CONFIG_ID,
                    position_m=list(self.ee_pos[arm]),
                    quaternion_xyzw=_quat_xyzw_from_angles_deg(p, r, y),
                ),
                gripper=dict(
                    joint_names=["joint7", "joint8"],
                    position_source_kind="measurement",
                    position_m=[half_ap, half_ap],
                    goal_source_kind="command",
                    goal_normalized=float(self.jaw_goal[arm]),
                    command_source_kind="command",
                    command_normalized=float(self.jaw_goal[arm]),
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

    def observe(self) -> dict:
        frames, calibration, poses = [], {}, {}
        specs = (
            ("cam_head", None, T_HEAD),
            ("cam_left_wrist", "left", [[1.0, 0.0, 0.0, 0.15], [0.0, 1.0, 0.0, 0.20], [0.0, 0.0, 1.0, 0.95], [0.0, 0.0, 0.0, 1.0]]),
            ("cam_right_wrist", "right", [[1.0, 0.0, 0.0, 0.15], [0.0, 1.0, 0.0, -0.20], [0.0, 0.0, 1.0, 0.95], [0.0, 0.0, 0.0, 1.0]]),
        )
        for idx, (cam_id, arm, transform) in enumerate(specs):
            img = np.full((480, 640, 3), 35 + idx * 15, dtype=np.uint8)
            cx = 300 + (self.sim_step * 6) % 60
            cv2.circle(img, (cx, 230), 16, (80, 190, 240), -1)
            if cam_id == "cam_head":
                # 在 cam_head 上按真实外参投影左臂指尖夹持中心标记点，使局部 48×48 邻域存在真实像素运动
                t_cw = np.linalg.inv(np.asarray(T_HEAD, dtype=np.float64))
                k_mat = np.asarray(CAMERA_K, dtype=np.float64)
                tip_world = np.asarray(self.ee_pos["left"], dtype=np.float64) + np.array(
                    [0.0, 0.0, -roboprobe_pose.GRASP_POINT_OFFSET_M], dtype=np.float64
                )
                p_cam = t_cw[:3, :3] @ tip_world + t_cw[:3, 3]
                if p_cam[2] > 0.05:
                    fu = int(round(float(k_mat[0, 0] * p_cam[0] / p_cam[2] + k_mat[0, 2])))
                    fv = int(round(float(k_mat[1, 1] * p_cam[1] / p_cam[2] + k_mat[1, 2])))
                    cv2.circle(img, (fu, fv), 7, (235, 220, 90), -1)
            ok, buf = cv2.imencode(".png", img)
            assert ok
            p = self.work_dir / f"{cam_id}_{self.sim_step}.png"
            p.write_bytes(buf.tobytes())
            frame_id = f"{cam_id}-{self.sim_step}-{idx}"
            frames.append(
                dict(
                    camera_id=cam_id,
                    frame_id=frame_id,
                    png_path=str(p),
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
                    T_link_camera=[[1.0, 0.0, 0.0, 0.03], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.02], [0.0, 0.0, 0.0, 1.0]],
                    T_world_base=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, base_y], [0.0, 0.0, 1.0, TABLE_Z + 0.12], [0.0, 0.0, 0.0, 1.0]],
                )
            else:
                calibration[cam_id]["T_world_camera"] = T_HEAD
        return dict(
            data_origin=self.data_origin,
            world_kind="observed",
            observation_profile=self.observation_profile,
            instruction="Pick the block and place it onto the stand.",
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
            robot_model=_urdf_6dof_model(),
            robot_state=self._state_snapshot(),
        )

    def reset(self, seed: int = 0, layout_id: int = 0) -> dict:
        if self.raise_Reset_error:
            raise RuntimeError("simulated_reset_failure")
        self.sim_step = 0
        self.jaw_half_m = {"left": 0.044, "right": 0.044}
        self.jaw_goal = {"left": 1.0, "right": 1.0}
        return dict(observation=self.observe(), seed=seed, layout_id=layout_id)

    def preview_batch(self, request: dict) -> dict:
        self._plan_seq += 1
        plan_id = f"batch-p2-{self._plan_seq}"
        stages = request["stages"]
        rows = []
        contexts = []
        cur_q_l = list(self.q_state["left"])
        cur_q_r = list(self.q_state["right"])
        cur_g_l = float(self.jaw_goal["left"])
        cur_g_r = float(self.jaw_goal["right"])
        steps_per_stage = self.steps_per_stage
        for idx, stage in enumerate(stages):
            arm = stage.get("arm", "left")
            kind = stage["kind"]
            if kind == "gripper":
                own = stage.get("action") or stage
                events = own.get("gripper_events") or []
                goal = float(events[-1]["goal_normalized"]) if events else float(own.get("goal_normalized", 0.0))
                if arm == "left":
                    cur_g_l = goal
                else:
                    cur_g_r = goal
            elif kind == "move":
                delta = 0.03 * (idx + 1)
                if arm == "left":
                    cur_q_l = [round(v + delta, 5) for v in self.q_state["left"]]
                else:
                    cur_q_r = [round(v - delta, 5) for v in self.q_state["right"]]
            for _ in range(steps_per_stage):
                rows.append(cur_q_l + [cur_g_l] + cur_q_r + [cur_g_r])
                contexts.append(
                    {
                        "left": dict(stage_id=stage["id"] if arm == "left" else None, phase=kind if arm == "left" else "hold"),
                        "right": dict(stage_id=stage["id"] if arm == "right" else None, phase=kind if arm == "right" else "hold"),
                    }
                )
        samples = [
            dict(
                native_step=0,
                arms={a: dict(q_rad=list(self.q_state[a]), jaw_goal_normalized=float(self.jaw_goal[a]), phase="initial") for a in ("left", "right")},
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
        self._plans[plan_id] = dict(rows=rows, contexts=contexts, stages=deepcopy(stages))
        return dict(
            status="planned",
            plan_id=plan_id,
            planned_native_steps=len(rows),
            trajectory_data=traj,
            trajectory_evidence=dict(available=True),
            stage_costs={},
        )

    def execute(self, action: dict, progress_callback=None, cancel_flag=None) -> dict:
        del cancel_flag
        before_obs = self.observe()
        before_state = self._state_snapshot()
        plan_info = self._plans.get(action.get("batch_plan_id"))
        rows = deepcopy(plan_info["rows"]) if plan_info else [[0.05] * 6 + [1.0] + [-0.05] * 6 + [1.0]]
        contexts = deepcopy(plan_info["contexts"]) if plan_info else []
        stages = plan_info["stages"] if plan_info else action.get("stages", [])
        total_steps = len(rows)
        scripted_ap = list(self.scripted_left_apertures_m) if self.scripted_left_apertures_m else None
        actual_trace = []
        for idx in range(total_steps):
            self.sim_step += 1
            row = rows[idx]
            self.q_state["left"] = list(row[:6])
            self.jaw_goal["left"] = float(row[6])
            self.q_state["right"] = list(row[7:13])
            self.jaw_goal["right"] = float(row[13])
            self.ee_pos["left"][2] = 0.84 + 0.015 * idx
            if scripted_ap and idx < len(scripted_ap):
                self.jaw_half_m["left"] = max(0.0, float(scripted_ap[idx]) * 0.5)
            else:
                self.jaw_half_m["left"] = 0.008 if self.jaw_goal["left"] < 0.5 else 0.044
            actual_trace.append(dict(sim_step=self.sim_step, state=self._state_snapshot()))
            if progress_callback and self.emit_progress_at_step == (idx + 1):
                ctx_l = contexts[idx]["left"] if idx < len(contexts) else {}
                progress_callback(
                    dict(
                        phase="checkpoint_progress",
                        stage_id=ctx_l.get("stage_id"),
                        executed_steps=idx + 1,
                        sim_step=self.sim_step,
                        observation=self.observe(),
                    )
                )
        terminal_obs = self.observe()
        stage_results = {
            st["id"]: dict(kind=st["kind"], arm=st.get("arm"), status="completed", planned_steps=2, completed_steps=2)
            for st in stages
        }
        return dict(
            status="completed",
            reason=None,
            data_origin=self.data_origin,
            executed_steps=total_steps,
            completed_native_steps=total_steps,
            position_error_m=0.001,
            orientation_error_rad=0.01,
            stage_results=stage_results,
            before_observation=before_obs,
            terminal_observation=terminal_obs,
            before_state=before_state,
            trace_sampling=dict(actual_every_native_actions=1, actual_boundary_samples=True),
            planned_trace=dict(
                segments=[dict(actions=rows, row_contexts=contexts, completed_native_steps=total_steps)]
            ),
            dispatched_trace=dict(steps=total_steps),
            actual_trace=actual_trace,
        )


class FakeSessionClient:
    """记录 turn_start 并写入 codex-raw.jsonl 以供 native audit 校验。"""

    def __init__(self, store: Store) -> None:
        self.store = store
        self.turn_calls: list[tuple[str, list[dict], dict | None]] = []

    def turn_start(self, thread_id: str, input_items: list[dict], params: dict | None = None):
        self.turn_calls.append((thread_id, deepcopy(input_items), deepcopy(params)))
        turn_id = f"turn-{len(self.turn_calls)}"
        self.store.raw_codex(
            "sent",
            dict(method="turn/start", params=dict(threadId=thread_id, input=deepcopy(input_items), **(params or {}))),
        )
        self.store.raw_codex(
            "received",
            dict(method="turn/started", params=dict(threadId=thread_id, turn=dict(id=turn_id, status="inProgress"))),
        )
        return SimpleNamespace(turn=SimpleNamespace(id=turn_id))


def _attach_fake_codex_session(workbench: Workbench) -> FakeSessionClient:
    fake_client = FakeSessionClient(workbench.store)

    def _start_session(tools, instructions_text, resume_thread_id=None):
        del tools, instructions_text
        thread_id = resume_thread_id or "thread-p2-audit"
        workbench.session.client = fake_client
        workbench.session.thread_id = thread_id
        workbench.session.state = "idle"
        workbench.session.provider = "test-provider"
        workbench.session.cli_executable = "/usr/bin/codex"
        workbench.session.process = SimpleNamespace(pid=54321)
        workbench.session.model_proxy = SimpleNamespace(transport_id="transport-p2", image_context_epoch=1)
        workbench.store.set_state(
            "runtime",
            "codex",
            {"thread_id": thread_id, "app_server_instance_id": "app-instance-p2"},
        )
        return thread_id

    workbench.session.start = _start_session
    workbench.session.close = lambda: None
    return fake_client


from support.reply_defaults import LEGACY_REPLY_FEATURES


class P2Rollout4DLedgerTests(unittest.TestCase):
    """全面验证 Phase 2 运动期 4D 过程轨迹、持久 ID 差分、on_anomaly 滑脱联动与 Schema <5000 字节红线。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_co_motion_slip_detection_thin_cloth_immunity_and_release_exclusion(self) -> None:
        """验证刚体搬运中途孔径塌陷检出 slip_detected，极薄布料捏持与正常主动开爪释放零误报。"""
        rot = np.eye(3).tolist()

        def _make_sample(step: int, ap_m: float, z_m: float, goal: float, stage_id: str) -> tuple[dict, dict]:
            half = ap_m * 0.5
            m = dict(
                native_step=step,
                arms=dict(
                    left=dict(
                        link6_pose=dict(position_m=[0.20, 0.12, z_m], rotation_matrix=rot),
                        measured_jaw=dict(joint_names=["joint7", "joint8"], position_m=[half, half]),
                        jaw_goal_normalized=goal,
                    ),
                    right=dict(
                        link6_pose=dict(position_m=[0.20, -0.15, 0.88], rotation_matrix=rot),
                        measured_jaw=dict(joint_names=["joint7", "joint8"], position_m=[0.044, 0.044]),
                        jaw_goal_normalized=1.0,
                    ),
                ),
            )
            p = dict(
                native_step=step,
                execution_state="initial_measurement" if step == 0 else "completed_command",
                arms=dict(
                    left=dict(
                        link6_pose=dict(position_m=[0.20, 0.12, z_m], rotation_matrix=rot),
                        jaw_goal_normalized=goal,
                        stage_id=stage_id if step > 0 else None,
                    ),
                    right=dict(
                        link6_pose=dict(position_m=[0.20, -0.15, 0.88], rotation_matrix=rot),
                        jaw_goal_normalized=1.0,
                    ),
                ),
            )
            return m, p

        # (a) 刚体抓取稳态 0.012m 后，在闭爪抬升阶段第 4 步孔径塌陷至 0.0004m -> slip_detected=True
        slip_specs = [
            (0, 0.088, 0.86, 1.0, ""),
            (1, 0.088, 0.82, 1.0, "approach"),
            (2, 0.012, 0.82, 0.0, "grasp"),
            (3, 0.012, 0.86, 0.0, "lift"),
            (4, 0.0004, 0.90, 0.0, "lift"),
            (5, 0.0004, 0.94, 0.0, "carry"),
        ]
        m_samples, p_samples = zip(*(_make_sample(*sp) for sp in slip_specs))
        action_slip = dict(
            stages=[
                dict(id="approach", arm="left", kind="move"),
                dict(id="grasp", arm="left", kind="gripper", goal_normalized=0.0),
                dict(id="lift", arm="left", kind="move"),
                dict(id="carry", arm="left", kind="move"),
            ]
        )
        receipt_slip = dict(
            status="completed",
            executed_steps=5,
            completed_native_steps=5,
            execution_start_observation_id="obs-start-1",
            result_observation_id="obs-end-1",
        )
        summary_slip = build_compact_trajectory_summary(
            dict(measured=dict(frame_id=FRAME_ID, executed_native_steps=5, samples=list(m_samples)),
                 planned=dict(frame_id=FRAME_ID, samples=list(p_samples))),
            action_slip,
            receipt_slip,
        )
        pt_slip = summary_slip["process_trajectory"]
        self.assertIn("left", pt_slip["co_motion"])
        self.assertNotIn("right", pt_slip["co_motion"])
        co_left = pt_slip["co_motion"]["left"]
        self.assertEqual(co_left["aperture_start_m"], 0.088)
        self.assertEqual(co_left["aperture_min_m"], 0.0004)
        self.assertEqual(co_left["aperture_end_m"], 0.0004)
        self.assertAlmostEqual(co_left["transport_aperture_drop_m"], 0.0116, places=4)
        self.assertTrue(co_left["slip_detected"])
        self.assertEqual(co_left["slip_onset_step"], 4)
        self.assertEqual(co_left["slip_observation_id"], "obs-end-1")

        # (b) 极薄布料捏持（稳态开度 0.0006m < 0.0035m，搬运中 0.0002m）-> slip_detected=False
        cloth_specs = [
            (0, 0.088, 0.84, 1.0, ""),
            (1, 0.0006, 0.80, 0.0, "grasp"),
            (2, 0.0003, 0.86, 0.0, "lift"),
            (3, 0.0002, 0.90, 0.0, "fold"),
        ]
        m_cloth, p_cloth = zip(*(_make_sample(*sp) for sp in cloth_specs))
        pt_cloth = build_process_trajectory(
            list(m_cloth),
            action=dict(stages=[dict(id="grasp", arm="left", kind="gripper", goal_normalized=0.0),
                                dict(id="lift", arm="left", kind="move"),
                                dict(id="fold", arm="left", kind="move")]),
            planned_samples=list(p_cloth),
        )
        self.assertFalse(pt_cloth["co_motion"]["left"]["slip_detected"])
        self.assertIsNone(pt_cloth["co_motion"]["left"]["slip_onset_step"])

        # (c) 刚体正常闭爪搬运后主动开爪释放（goal_normalized=1.0 -> 0.088m）-> slip_detected=False
        release_specs = [
            (0, 0.088, 0.86, 1.0, ""),
            (1, 0.015, 0.82, 0.0, "grasp"),
            (2, 0.015, 0.90, 0.0, "lift"),
            (3, 0.088, 0.90, 1.0, "release"),
        ]
        m_rel, p_rel = zip(*(_make_sample(*sp) for sp in release_specs))
        pt_rel = build_process_trajectory(
            list(m_rel),
            action=dict(stages=[
                dict(id="grasp", arm="left", kind="gripper", goal_normalized=0.0),
                dict(id="lift", arm="left", kind="move"),
                dict(id="release", arm="left", kind="gripper", goal_normalized=1.0),
            ]),
            planned_samples=list(p_rel),
        )
        self.assertEqual(pt_rel["co_motion"]["left"]["transport_aperture_drop_m"], 0.0)
        self.assertFalse(pt_rel["co_motion"]["left"]["slip_detected"])

    def test_in_place_motion_yaw_unwrap_ratchet_and_vertical_lever_press(self) -> None:
        """验证闭爪啮合期间跨越 ±180° 分支的偏航旋拧积分、开爪回退隔离及垂直轴向行程积分。"""
        # 模拟棘轮拧螺母 + 下压拨杆：
        # 步 0: 张爪 0.088m, yaw=-150°, z=0.840m
        # 步 1: 闭爪啮合 0.020m, yaw=-150°, z=0.840m
        # 步 2..4: 闭爪旋拧跨越 -180/+180 边界 (-150 -> -175 -> +160 -> +120，共 -90°) 且下压 dz=-0.035m (0.840 -> 0.805)
        # 步 5..6: 张爪 0.088m 回转复位 (+120 -> -150，开爪态不计入啮合积分！)
        # 步 7..8: 再次闭爪 0.020m 旋拧 -45° (-150 -> -175 -> +165)
        trace_steps = [
            (0, 0.088, 1.0, -150.0, 0.840, None),
            (1, 0.020, 0.0, -150.0, 0.840, "close_1"),
            (2, 0.020, 0.0, -175.0, 0.830, "twist_1"),
            (3, 0.020, 0.0, 160.0, 0.815, "twist_1"),
            (4, 0.020, 0.0, 120.0, 0.805, "twist_1"),
            (5, 0.088, 1.0, 120.0, 0.805, "open_1"),
            (6, 0.088, 1.0, -150.0, 0.805, "unwind"),
            (7, 0.020, 0.0, -150.0, 0.805, "close_2"),
            (8, 0.020, 0.0, 165.0, 0.805, "twist_2"),
        ]
        measured_samples = []
        planned_samples = []
        for step, ap_m, goal, yaw_deg, z_m, sid in trace_steps:
            quat_xyzw = _quat_xyzw_from_angles_deg(0.0, 0.0, yaw_deg)
            # 保持法兰 +x（竖直向下）方向不变，使 TCP 水平坐标 (0.15, -0.10) 恒定为零水平位移
            measured_samples.append(
                dict(
                    native_step=step,
                    arms=dict(
                        right=dict(
                            ee_pose=dict(
                                source_kind="measurement",
                                frame_id=FRAME_ID,
                                tcp_config_id=TCP_CONFIG_ID,
                                position_m=[0.15, -0.10, z_m],
                                quaternion_xyzw=quat_xyzw,
                            ),
                            measured_jaw=dict(joint_names=["joint7", "joint8"], position_m=[ap_m * 0.5, ap_m * 0.5]),
                            jaw_goal_normalized=goal,
                        ),
                        left=dict(
                            measured_jaw=dict(joint_names=["joint7", "joint8"], position_m=[0.044, 0.044]),
                            jaw_goal_normalized=1.0,
                        ),
                    ),
                )
            )
            planned_samples.append(
                dict(
                    native_step=step,
                    arms=dict(
                        right=dict(jaw_goal_normalized=goal, stage_id=sid),
                        left=dict(jaw_goal_normalized=1.0),
                    ),
                )
            )
        action = dict(
            stages=[
                dict(id="close_1", arm="right", kind="gripper", goal_normalized=0.0),
                dict(id="twist_1", arm="right", kind="move"),
                dict(id="open_1", arm="right", kind="gripper", goal_normalized=1.0),
                dict(id="unwind", arm="right", kind="move"),
                dict(id="close_2", arm="right", kind="gripper", goal_normalized=0.0),
                dict(id="twist_2", arm="right", kind="move"),
            ]
        )
        pt = build_process_trajectory(measured_samples, action=action, planned_samples=planned_samples)
        ip = pt["in_place_motion"]["right"]
        # 第一轮啮合旋拧 -90° + 第二轮啮合旋拧 -45° = -135°；开爪回转 +90° 被精确排除
        self.assertAlmostEqual(ip["engaged_wrist_yaw_delta_deg"], -135.0, delta=0.5)
        self.assertAlmostEqual(ip["engaged_dz_m"], -0.035, places=4)
        self.assertAlmostEqual(ip["engaged_xy_travel_m"], 0.0, places=4)
        self.assertGreaterEqual(ip["engaged_steps"], 4)

    def test_event_keyframes_short_id_mapping_zero_sha256_and_program_visibility(self) -> None:
        """验证 event_keyframes 最多 4 帧、observation_id 全部映射为短 ID obs_N、零 sha256 泄露，且 Actor 与 program_run 双端可见。"""
        server = FakePerceptionServer()
        self.addCleanup(server.close)
        run_dir = self.root / "run-short-id"
        run_dir.mkdir(parents=True, exist_ok=True)
        store = Store(run_dir, dict(RUN_IDENTITY))
        self.addCleanup(store.close)
        robot = RolloutCapableRobot(run_dir / "captures")
        config = Config().with_overrides(
            {
                "features": {
                    "external_perception": True,
                    "vggt_endpoint": server.url,
                    "vggt_support_z_m": TABLE_Z,
                    "held_perception": False,
                    "motion_review": False,
                    "task_scheduling": False,
                    "sandbox_enabled": expect(program_isolation, "program_run runs under the Landlock + seccomp sandbox"),
                    # the presented reply before any reply shaping: this test applies the shaping functions itself
                    **LEGACY_REPLY_FEATURES,
                }
            }
        )
        harness = Harness(store, robot, config=config)
        harness.thread_id = "thread-short-id"
        harness.start()
        harness.programs = ProgramRunner(harness.program_services())
        self.addCleanup(harness.programs.close)

        obs_wire = harness.handle_tool(
            dict(threadId=harness.thread_id, turnId="t-obs", callId="c-obs", tool="robot_observe", arguments={})
        )
        self.assertTrue(obs_wire["success"], obs_wire)
        init_short_obs = json.loads(obs_wire["contentItems"][0]["text"])["bundle"]["observation_id"]

        plan_wire = harness.handle_tool(
            dict(
                threadId=harness.thread_id,
                turnId="t-plan",
                callId="c-plan",
                tool="task_plan",
                arguments=dict(operation="submit", observation_id=init_short_obs, plan="Grasp and lift.", ledger=[]),
            )
        )
        self.assertTrue(plan_wire["success"], plan_wire)

        # 3 个阶段 × 每阶段 2 步 = 6 步：grasp 结束在步 4（稳态 0.014m），lift 第 5 步滑脱塌陷至 0.0004m，并在第 5 步产生正式过程观测帧
        robot.scripted_left_apertures_m = [0.088, 0.088, 0.030, 0.014, 0.0004, 0.0004]
        robot.emit_progress_at_step = 5
        stages = [
            dict(id="approach", kind="move", arm="left", grasp_position_m=[0.18, 0.14, 0.84], angles_deg=[0, 0, 0]),
            dict(id="grasp", kind="gripper", arm="left", depends_on=["approach"], goal_normalized=0.0),
            dict(id="lift", kind="move", arm="left", depends_on=["grasp"], grasp_position_m=[0.18, 0.14, 0.92], angles_deg=[0, 0, 0]),
        ]
        wire = harness.handle_tool(
            dict(
                threadId=harness.thread_id,
                turnId="t-act",
                callId="c-act",
                tool="robot_action",
                arguments=dict(mode="direct", observation_id=init_short_obs, stages=stages),
            )
        )
        self.assertTrue(wire["success"], wire)

        # 校验零元数据/哈希泄露（无任何 64 位十六进制 sha256 出现在文本载荷中）
        text_blobs = [item["text"] for item in wire["contentItems"] if item.get("type") == "inputText"]
        joined_text = "\n".join(text_blobs)
        self.assertEqual(metadata_findings(joined_text), [])
        self.assertIsNone(re.search(r"\b[0-9a-f]{64}\b", joined_text))

        header = json.loads(text_blobs[0])
        pt_actor = header["trajectory_feedback"]["compact_3d_trajectory"]["process_trajectory"]
        self.assertEqual(header["receipt"]["process_trajectory"], pt_actor)
        self.assertTrue(pt_actor["co_motion"]["left"]["slip_detected"])
        self.assertEqual(pt_actor["co_motion"]["left"]["slip_onset_step"], 5)
        self.assertRegex(pt_actor["co_motion"]["left"]["slip_observation_id"], r"^obs_\d+$")
        flow_px = pt_actor["co_motion"]["left"].get("fingertip_flow_px")
        self.assertIsNotNone(flow_px)
        self.assertIsInstance(flow_px, float)
        self.assertGreater(flow_px, 0.0)
        # 验证收窄异常捕获后不会静默吞掉 AttributeError 代码缺陷
        with self.assertRaises(AttributeError):
            _fingertip_local_flow_px(object(), harness.latest, harness.latest, np.array([0.18, 0.14, 0.90]))

        keyframes = pt_actor["event_keyframes"]
        self.assertLessEqual(len(keyframes), 4)
        self.assertEqual([kf["event"] for kf in keyframes], ["start", "grasp_settled", "peak_error_or_slip", "terminal"])
        required_keys = {"event", "stage_id", "executed_step", "observation_id", "tcp_position_m", "aperture_m"}
        for kf in keyframes:
            self.assertEqual(set(kf.keys()), required_keys)
            self.assertRegex(kf["observation_id"], r"^obs_\d+$")
            # 验证每个短 ID 均能被 ActorOutput 精确反解回正式注册的真实观测 ID
            restored = harness.actor_output.restore(kf["observation_id"])
            self.assertTrue(restored.startswith("obs-"))
            self.assertIsNotNone(store.read("observation", restored))

        # 验证 program_run 同样可直接读取 reply["data"]["receipt"]["process_trajectory"]
        latest_obs_id = harness.latest["observation_id"]
        robot.scripted_left_apertures_m = [0.088, 0.088, 0.020, 0.012, 0.012, 0.012]
        robot.emit_progress_at_step = None
        prog_script = f"""
import json
from ni_rua_client import NiRuaClient
client = NiRuaClient()
res = client.call("robot_action", {{
    "mode": "direct",
    "observation_id": {latest_obs_id!r},
    "stages": [
        {{"id": "reach", "kind": "move", "arm": "left", "grasp_position_m": [0.20, 0.14, 0.85], "angles_deg": [0, 0, 0]}},
        {{"id": "close", "kind": "gripper", "arm": "left", "depends_on": ["reach"], "goal_normalized": 0.0}},
        {{"id": "carry", "kind": "move", "arm": "left", "depends_on": ["close"], "grasp_position_m": [0.22, 0.14, 0.90], "angles_deg": [0, 0, 0]}},
    ],
}})
assert res["success"], res
pt = res["data"]["receipt"]["process_trajectory"]
print(json.dumps({{"has_co_motion": "left" in pt["co_motion"], "slip": pt["co_motion"]["left"]["slip_detected"], "kf_len": len(pt["event_keyframes"])}}))
"""
        prog_wire = harness.handle_tool(
            dict(
                threadId=harness.thread_id,
                turnId="t-prog",
                callId="c-prog",
                tool="program_run",
                arguments=dict(source=prog_script, allow_direct_actions=True, timeout_s=15),
            )
        )
        self.assertTrue(prog_wire["success"], prog_wire)
        job_doc = json.loads(prog_wire["contentItems"][0]["text"])
        stdout_txt = job_doc.get("stdout") if "stdout" in job_doc else job_doc["stdout_text"]
        self.assertEqual(json.loads(stdout_txt.strip()), {"has_co_motion": True, "slip": False, "kf_len": 3})

        # 预算器把轨迹块换成按引用后，回复里写明的现有入口能取回与原回复逐值相同的内容
        from ni_rua.core.presentation.reply_budget import fit_reply_report

        fitted, report = fit_reply_report(wire["contentItems"], 40)
        self.assertEqual(
            report["by_reference"],
            ["planned_samples", "stages", "measured_samples", "compact_3d_trajectory"],
        )
        self.assertEqual(sum(item["type"] == "inputImage" for item in fitted),
                         sum(item["type"] == "inputImage" for item in wire["contentItems"]))
        parts = [json.loads(item["text"]) for item in fitted if item["type"] == "inputText"]
        self.assertTrue(parts[0]["receipt"]["process_trajectory"]["co_motion"]["left"]["slip_detected"])
        feedback = next(part for part in parts if "trajectory_feedback" in part)["trajectory_feedback"]
        self.assertEqual(feedback["anomaly_reason"], "slip_detected")
        note = feedback["compact_3d_trajectory"]
        read_wire = harness.handle_tool(
            dict(
                threadId=harness.thread_id,
                turnId="t-read",
                callId="c-read",
                tool="program_run",
                arguments=dict(
                    source=(
                        "import json\nfrom ni_rua_client import NiRuaClient\n"
                        f"document = json.loads(NiRuaClient().artifact({note['read']['reference']!r}))\n"
                        "print(json.dumps(document['summary']['compact_3d_trajectory'], sort_keys=True))\n"
                    ),
                    timeout_s=15,
                ),
            )
        )
        self.assertTrue(read_wire["success"], read_wire)
        read_doc = json.loads(read_wire["contentItems"][0]["text"])
        read_text = read_doc.get("stdout") if "stdout" in read_doc else read_doc["stdout_text"]
        self.assertEqual(note["at"], "summary.compact_3d_trajectory")
        self.assertEqual(
            json.loads(read_text.strip())["process_trajectory"]["co_motion"]["left"]["slip_onset_step"],
            header["trajectory_feedback"]["compact_3d_trajectory"]["process_trajectory"]["co_motion"]["left"]["slip_onset_step"],
        )
        for path_kind in ("measured", "planned"):
            read = dict(feedback[path_kind]["samples"]["read"])
            inspected = harness.handle_tool(
                dict(threadId=harness.thread_id, turnId="t-read", callId="c-inspect-" + path_kind,
                     tool=read.pop("tool"), arguments=read)
            )
            self.assertTrue(inspected["success"], inspected)
            shown = json.loads(inspected["contentItems"][0]["text"])
            self.assertEqual((shown["path_kind"], shown["total_samples"]),
                             (path_kind, header["trajectory_feedback"][path_kind]["total_samples"]))
            # exactly the rows the shaped reply left out, each holding the values the unshaped reply had inline
            inline = header["trajectory_feedback"][path_kind]["samples"]
            self.assertEqual([row["native_step"] for row in shown["samples"]], [row["native_step"] for row in inline])
            self.assertEqual(read["native_steps"], [row["native_step"] for row in inline])
            for row, original_row in zip(shown["samples"], inline):
                self._assert_row_holds(row, original_row)

        # reply_static_once 把同一批取用入口统一成 {"detail": {"tool", "arguments"}}：robot_action_status 常驻加载，入口直接点名它，
        # 按回复里给出的参数调用即可取回被替换的阶段明细（at 是其 data 内的路径）
        from ni_rua.core.presentation.reply_compaction import unify_detail_entries

        loaded = harness.tool_registry.loaded_tool_names(harness, role="actor")
        self.assertIn("robot_action_status", loaded)
        unified = unify_detail_entries(fitted, loaded)
        stages_note = next(part for part in (json.loads(item["text"]) for item in unified if item["type"] == "inputText")
                           if "stages_by_reference" in part)["stages_by_reference"]
        self.assertEqual(stages_note["detail"]["tool"], "robot_action_status")
        self.assertEqual(stages_note["at"], "receipt.stage_results")
        stage_wire = harness.handle_tool(
            dict(threadId=harness.thread_id, turnId="t-read", callId="c-detail",
                 tool=stages_note["detail"]["tool"], arguments=stages_note["detail"]["arguments"])
        )
        self.assertTrue(stage_wire["success"], stage_wire)
        stage_doc = json.loads(stage_wire["contentItems"][0]["text"])
        self.assertEqual(sorted(stage_doc["receipt"]["stage_results"]), sorted(stages_note["stage_ids"]))
        # 未被预算器替换的原回复：trajectory_feedback 的取用入口按回复里写的参数直接可调，取回实测路径
        whole = unify_detail_entries(wire["contentItems"], loaded)
        entry = next(part for part in (json.loads(item["text"]) for item in whole if item["type"] == "inputText")
                     if "trajectory_feedback" in part)["trajectory_feedback"]
        self.assertNotIn("detail_access", entry)
        # detail_access names the artifact holding every exact row: the entry reads that artifact (program_run loaded)
        self.assertEqual(entry["detail"]["tool"], "program_run")
        path_wire = harness.handle_tool(
            dict(threadId=harness.thread_id, turnId="t-read", callId="c-detail-path",
                 tool=entry["detail"]["tool"], arguments=entry["detail"]["arguments"])
        )
        self.assertTrue(path_wire["success"], path_wire)
        path_doc = json.loads(path_wire["contentItems"][0]["text"])
        path_text = path_doc.get("stdout") if "stdout" in path_doc else path_doc["stdout_text"]
        document = json.loads(path_text.strip())
        self.assertEqual(len(document["paths"]["measured"]["samples"]),
                         header["trajectory_feedback"]["measured"]["total_samples"])
        # 滑脱回复属于异常：固定说明句全部保留；登记表里的轨迹类说明句与真实生产者写出的逐字一致（改写后会重新送达）
        from ni_rua.core.presentation.reply_compaction import drop_static_once_report

        kept, static_report = drop_static_once_report(wire["contentItems"])
        self.assertIs(kept, wire["contentItems"])
        self.assertEqual(static_report, dict(exception=True, dropped={}))
        carried = drop_static_once_report(wire["contentItems"], routine_only=False)[1]["dropped"]
        self.assertEqual({key: carried.get(key) for key in (
            "calibration_scope", "jaw_scope", "image_scope", "limitations", "derivatives", "timing", "unexecuted_scope")},
            dict(calibration_scope=1, jaw_scope=1, image_scope=1, limitations=1, derivatives=1, timing=2,
                 unexecuted_scope=1))

        # summary_first：滑脱属于异常，固定形态之外内联滑脱起点附近的实测采样；compact_3d_trajectory 始终内联
        from ni_rua.core.presentation import reply_budget
        from ni_rua.core.presentation.reply_encoding import DERIVED_PARENT, DERIVED_QUANTITIES, encode_compact_items
        from core.presentation.test_actor_payload import compact_encoding_loss, decode_compact_value

        shaped, shape_report = fit_reply_report(wire["contentItems"], None, profile="summary_first")
        self.assertFalse(shape_report["routine"])
        self.assertEqual(shape_report["fixed"], [name for name in reply_budget._SUMMARY_EXCEPTION
                                                 if name in shape_report["fixed"]])
        self.assertIn("planned_samples", shape_report["fixed"])
        self.assertNotIn("measured_samples", shape_report["by_reference"])
        shaped_feedback = next(part for part in (json.loads(item["text"]) for item in shaped if item["type"] == "inputText")
                               if "trajectory_feedback" in part)["trajectory_feedback"]
        self.assertEqual(shaped_feedback["compact_3d_trajectory"], header["trajectory_feedback"]["compact_3d_trajectory"])
        original_samples = header["trajectory_feedback"]["measured"]["samples"]
        self.assertTrue(shaped_feedback["measured"]["samples"])
        self.assertTrue(all(sample in original_samples for sample in shaped_feedback["measured"]["samples"]))
        onset = header["trajectory_feedback"]["compact_3d_trajectory"]["process_trajectory"]["co_motion"]["left"]["slip_onset_step"]
        kept_steps = [sample["native_step"] for sample in shaped_feedback["measured"]["samples"]]
        self.assertTrue(min(kept_steps) <= onset <= max(kept_steps), (kept_steps, onset))
        # reply_compact_encoding：真实回复编码后可解码回逐值相同的内容（仅派生量按 4 位有效数字舍入）；
        # 派生量表与真实生产者写出的 derived_motion 字段一致
        loss = None
        for before, after in zip(wire["contentItems"], encode_compact_items(wire["contentItems"])):
            if before["type"] == "inputText":
                loss = compact_encoding_loss(json.loads(before["text"]), decode_compact_value(json.loads(after["text"])), loss)
        self.assertLessEqual(loss["derived_relative"], 5e-4)
        produced = set()
        for sample in original_samples:
            for entry in sample["arms"].values():
                produced |= {key for key, value in entry.get(DERIVED_PARENT, {}).items()
                             if isinstance(value, list) and key != "interval_sim_steps"}
        self.assertTrue(produced)
        self.assertLessEqual(produced, set(DERIVED_QUANTITIES))

    def _executed_reply(self, name: str, *, steps_per_stage: int = 2, slip: bool = False) -> tuple[Harness, list[dict]]:
        """A real harness (the reply as it was before any shaping) after one direct three-stage action; its execution
        reply items. ``slip`` collapses the jaw during the last stage, which makes the reply an exception."""
        server = FakePerceptionServer()
        self.addCleanup(server.close)
        run_dir = self.root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        store = Store(run_dir, dict(RUN_IDENTITY))
        self.addCleanup(store.close)
        robot = RolloutCapableRobot(run_dir / "captures")
        robot.steps_per_stage = steps_per_stage
        config = Config().with_overrides({"features": {
            "external_perception": True, "vggt_endpoint": server.url, "vggt_support_z_m": TABLE_Z,
            "held_perception": False, "motion_review": False, "task_scheduling": False,
            "sandbox_enabled": expect(program_isolation, "program_run runs under the Landlock + seccomp sandbox"),
            **LEGACY_REPLY_FEATURES}})
        harness = Harness(store, robot, config=config)
        harness.thread_id = "thread-" + name
        harness.start()
        harness.programs = ProgramRunner(harness.program_services())
        self.addCleanup(harness.programs.close)
        observation = self._call(harness, "robot_observe", {})
        self._call(harness, "task_plan", dict(operation="submit", observation_id=observation["bundle"]["observation_id"],
                                              plan="Grasp and lift.", ledger=[]))
        if slip:
            robot.scripted_left_apertures_m = [0.088, 0.088, 0.030, 0.014, 0.0004, 0.0004]
            robot.emit_progress_at_step = 5
        stages = [
            dict(id="approach", kind="move", arm="left", grasp_position_m=[0.18, 0.14, 0.84], angles_deg=[0, 0, 0]),
            dict(id="grasp", kind="gripper", arm="left", depends_on=["approach"], goal_normalized=0.0),
            dict(id="lift", kind="move", arm="left", depends_on=["grasp"], grasp_position_m=[0.18, 0.14, 0.92], angles_deg=[0, 0, 0]),
        ]
        wire = harness.handle_tool(dict(threadId=harness.thread_id, turnId="t-act", callId="c-act", tool="robot_action",
                                        arguments=dict(mode="direct", observation_id=observation["bundle"]["observation_id"],
                                                       stages=stages)))
        self.assertTrue(wire["success"], wire)
        return harness, wire["contentItems"]

    _calls = 0

    def _call(self, harness: Harness, tool: str, arguments: dict) -> dict:
        """The first text document of a tool call made as the actor does (schema validation included)."""
        P2Rollout4DLedgerTests._calls += 1
        wire = harness.handle_tool(dict(threadId=harness.thread_id, turnId="t-read", callId=f"c-read-{self._calls}",
                                        tool=tool, arguments=arguments))
        self.assertTrue(wire["success"], wire)
        return json.loads(wire["contentItems"][0]["text"])

    @staticmethod
    def _feedback(items: list[dict]) -> dict:
        return next(part for part in (json.loads(item["text"]) for item in items if item["type"] == "inputText")
                    if "trajectory_feedback" in part)["trajectory_feedback"]

    def _assert_row_holds(self, shown: dict, inline: dict) -> None:
        """Every value the unshaped reply carried inline for this row is in the row a call returns, equal to it (the
        call may return more: the reply drops fields it states once elsewhere, such as the jaw of all but the last row)."""
        self.assertEqual({key: shown[key] for key in inline if key != "arms"}, {key: inline[key] for key in inline if key != "arms"})
        self.assertEqual(set(inline["arms"]), set(shown["arms"]))
        for arm, state in inline["arms"].items():
            for key, value in state.items():
                if key.endswith("_position_m") and key not in shown["arms"][arm]:
                    continue  # the reply adds the rounded tool position; the call returns the measured pose itself
                self.assertEqual(shown["arms"][arm][key], value, (shown["native_step"], arm, key))

    def _assert_call_returns_rows(self, harness: Harness, call: dict, inline: list[dict], steps: list[int]) -> None:
        """Making ``call`` exactly as written returns the rows of ``steps``, each holding the values of the row the
        unshaped reply carried inline."""
        arguments = dict(call)
        shown = self._call(harness, arguments.pop("tool"), arguments)
        by_step = {row["native_step"]: row for row in inline}
        self.assertEqual([row["native_step"] for row in shown["samples"]], steps)
        for row in shown["samples"]:
            self._assert_row_holds(row, by_step[row["native_step"]])

    def test_calls_named_by_a_shaped_reply_return_exactly_the_rows_it_left_out_value_for_value(self) -> None:
        harness, items = self._executed_reply("reply-routine")
        inline = self._feedback(items)
        shaped, report = fit_reply_report(items, None, profile="summary_first")
        self.assertTrue(report["routine"])
        feedback = self._feedback(shaped)
        for path_kind in ("measured", "planned"):
            with self.subTest(path_kind):
                read = feedback[path_kind]["samples"]["read"]
                steps = [row["native_step"] for row in inline[path_kind]["samples"]]
                self.assertGreater(len(steps), 3)  # more than the first, middle and last row inspect returns by default
                self.assertEqual(read["native_steps"], steps)
                self._assert_call_returns_rows(harness, read, inline[path_kind]["samples"], steps)

    def test_calls_named_for_the_rows_outside_an_exception_window_return_exactly_those_rows(self) -> None:
        harness, items = self._executed_reply("reply-exception", slip=True)
        inline = self._feedback(items)
        shaped, report = fit_reply_report(items, None, profile="summary_first")
        self.assertFalse(report["routine"])
        measured = self._feedback(shaped)["measured"]
        kept = [row["native_step"] for row in measured["samples"]]
        outside = [row["native_step"] for row in inline["measured"]["samples"] if row["native_step"] not in kept]
        self.assertTrue(kept and outside)
        self.assertEqual(measured["samples_by_reference"]["count"], len(outside))
        self.assertEqual(measured["samples_by_reference"]["read"]["native_steps"], outside)
        self._assert_call_returns_rows(harness, measured["samples_by_reference"]["read"], inline["measured"]["samples"], outside)

    def test_a_path_longer_than_one_inspect_call_is_read_in_pages_that_continue_to_its_last_row(self) -> None:
        harness, items = self._executed_reply("reply-pages", steps_per_stage=12)
        inline = self._feedback(items)
        # program_run is not loaded: the detail entry is the paged inspect read of the whole measured path
        entry = self._feedback(unify_detail_entries(items, frozenset({"robot_action"})))[DETAIL_KEY]
        total = inline["measured"]["total_samples"]
        self.assertGreater(total, INSPECT_STEP_LIMIT)
        arguments, steps, pages = dict(entry["arguments"]), [], 0
        while True:
            shown = self._call(harness, entry["tool"], dict(arguments))
            pages += 1
            self.assertLessEqual(len(shown["samples"]), INSPECT_STEP_LIMIT)
            steps += [row["native_step"] for row in shown["samples"]]
            if shown["next_native_step"] is None:
                break
            self.assertEqual(shown["next_native_step"], steps[-1] + 1)
            arguments["start_native_step"] = shown["next_native_step"]
        self.assertEqual((steps, pages), (list(range(total)), -(-total // INSPECT_STEP_LIMIT)))

    def test_the_trajectory_detail_entry_reads_the_artifact_that_holds_every_row(self) -> None:
        harness, items = self._executed_reply("reply-artifact")
        inline = self._feedback(items)
        reference = inline["trajectory_feedback_ref"]
        # program_run loaded: the entry is the artifact read detail_access describes, and returns every exact row
        entry = self._feedback(unify_detail_entries(items, frozenset({"robot_action", "program_run"})))[DETAIL_KEY]
        self.assertEqual(entry["tool"], "program_run")
        arguments = dict(entry["arguments"])
        job = self._call(harness, entry["tool"], arguments)
        document = json.loads(job.get("stdout", job.get("stdout_text")).strip())
        stored = json.loads(harness.store.resolve_artifact(harness.actor_output.restore(reference), allowed_scopes=("actor",))[1])
        for path_kind in ("measured", "planned"):
            self.assertEqual([row["native_step"] for row in document["paths"][path_kind]["samples"]],
                             [row["native_step"] for row in stored["paths"][path_kind]["samples"]])
        for path_kind in ("measured", "planned"):
            self.assertEqual(len(document["paths"][path_kind]["samples"]), inline[path_kind]["total_samples"])
        # program_run not loaded: the inspect call that pages from the first row (more than the default three rows)
        entry = self._feedback(unify_detail_entries(items, frozenset({"robot_action"})))[DETAIL_KEY]
        self.assertEqual(entry["arguments"], dict(mode="inspect", trajectory_ref=reference, path_kind="measured",
                                                  start_native_step=0, max_samples=INSPECT_STEP_LIMIT))
        shown = self._call(harness, entry["tool"], dict(entry["arguments"]))
        self.assertEqual([row["native_step"] for row in shown["samples"]],
                         [row["native_step"] for row in stored["paths"]["measured"]["samples"]][:INSPECT_STEP_LIMIT])

    def test_on_anomaly_slip_upgrades_to_single_640x480_image_and_native_audit_passes(self) -> None:
        """验证默认 on_anomaly 下正常抓取保持 0 图、中途滑脱自动附单幅 640×480 状态图、显式 none 保持 0 图，且通过 native audit。"""
        server = FakePerceptionServer()
        self.addCleanup(server.close)
        run_dir = self.root / "native-audit-p2"
        robot = RolloutCapableRobot(self.root / "native-audit-captures", data_origin="robodojo_runtime")
        config = Config().with_overrides(
            {
                "run": {"task": "place_object_stand", "benchmark": True},
                "features": {
                    "terminal": False,
                    "external_perception": True,
                    "vggt_endpoint": server.url,
                    "vggt_support_z_m": TABLE_Z,
                    "held_perception": False,
                    "motion_review": False,
                    "task_scheduling": False,
                    "action_trajectory_delivery": "diagnostic",
                    "grasp_memory_dir": str(self.root / "native-audit-mem"),
                },
                "images": {
                    "action_feedback_policy": "on_anomaly",
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
                {"method": "item/started", "params": {"threadId": harness.thread_id, "item": {"id": item_id, "type": "dynamicToolCall"}}},
            )
            harness.call_items[call_id] = item_id
            wire = harness.handle_tool(
                {"threadId": harness.thread_id, "turnId": turn_id, "callId": call_id, "tool": tool, "arguments": arguments}
            )
            wb.store.raw_codex("sent", {"id": f"rpc-{call_id}", "result": wire})
            return wire

        stages_template = [
            dict(id="approach", kind="move", arm="left", grasp_position_m=[0.18, 0.14, 0.85], angles_deg=[0, 0, 0]),
            dict(id="grasp", kind="gripper", arm="left", depends_on=["approach"], goal_normalized=0.0),
            dict(id="lift", kind="move", arm="left", depends_on=["grasp"], grasp_position_m=[0.18, 0.14, 0.92], angles_deg=[0, 0, 0]),
        ]

        # Turn 1：正常稳态抓取搬运（0.016m 保持不变），默认省略 image_delivery -> 0 图极速闭环
        _invoke_native_tool(
            "turn-1", "call-plan", "item-plan", "task_plan",
            {"operation": "submit", "observation_id": "obs_1", "plan": "Grasp and transport.", "ledger": []},
        )
        robot.scripted_left_apertures_m = [0.088, 0.088, 0.030, 0.016, 0.016, 0.016]
        wire_clean = _invoke_native_tool(
            "turn-1", "call-clean", "item-clean", "robot_action",
            {"mode": "direct", "observation_id": "obs_1", "stages": stages_template},
        )
        self.assertTrue(wire_clean["success"], wire_clean)
        self.assertEqual(sum(i["type"] == "inputImage" for i in wire_clean["contentItems"]), 0)
        from support.reply_defaults import observation_value
        obs_after_clean = observation_value(wire_clean["contentItems"])["bundle"]["observation_id"]
        wb.store.raw_codex(
            "received",
            {
                "method": "turn/completed",
                "params": {"threadId": harness.thread_id, "turn": {"id": "turn-1", "status": "completed", "items": []}},
            },
        )

        # Turn 2：中途滑脱（grasp 稳态 0.016m -> lift 塌陷至 0.0004m），默认省略 image_delivery -> 自动附单幅 640×480 图
        robot.scripted_left_apertures_m = [0.088, 0.088, 0.030, 0.016, 0.0004, 0.0004]
        wire_slip = _invoke_native_tool(
            "turn-2", "call-slip", "item-slip", "robot_action",
            {"mode": "direct", "observation_id": obs_after_clean, "stages": stages_template},
        )
        self.assertTrue(wire_slip["success"], wire_slip)
        slip_imgs = [i for i in wire_slip["contentItems"] if i.get("type") == "inputImage"]
        self.assertEqual(len(slip_imgs), 1)
        raw_img = base64.b64decode(slip_imgs[0]["imageUrl"].split(",", 1)[1])
        with Image.open(io.BytesIO(raw_img)) as im:
            self.assertEqual(im.size, (640, 480))
        from support.reply_defaults import action_result
        slip_header = action_result(wire_slip["contentItems"])
        self.assertEqual(slip_header["trajectory_feedback"].get("anomaly_reason"), "slip_detected")
        slip_obs_meta = observation_value(wire_slip["contentItems"])
        self.assertEqual(slip_obs_meta["image_feedback"]["reason"], "slip_detected")
        obs_after_slip = slip_obs_meta["bundle"]["observation_id"]

        # Turn 3：显式传 image_delivery="none" 时，即便检出 slip_detected 仍严格抑制图像（0 图）
        robot.scripted_left_apertures_m = [0.088, 0.088, 0.030, 0.016, 0.0004, 0.0004]
        wire_none = _invoke_native_tool(
            "turn-3", "call-none", "item-none", "robot_action",
            {"mode": "direct", "image_delivery": "none", "observation_id": obs_after_slip, "stages": stages_template},
        )
        self.assertTrue(wire_none["success"], wire_none)
        self.assertEqual(sum(i["type"] == "inputImage" for i in wire_none["contentItems"]), 0)

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
        report = audit(run_dir, profile="native", require_continuity=True)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["failures"], [])

    def test_harness_reset_and_close_notify_session_close_non_blocking_and_receipt_isolation(self) -> None:
        """验证 Harness.reset / close 发起短超时非阻塞 /session_close，且诊断异常绝不吞没已确认的物理回执。"""
        fast_server = FakePerceptionServer()
        self.addCleanup(fast_server.close)
        run_dir = self.root / "run-session-close"
        run_dir.mkdir(parents=True, exist_ok=True)
        store = Store(run_dir, dict(RUN_IDENTITY))
        self.addCleanup(store.close)
        robot = RolloutCapableRobot(run_dir / "captures")
        config = Config().with_overrides(
            {"features": {"external_perception": True, "vggt_endpoint": fast_server.url, "task_scheduling": False}}
        )
        harness = Harness(store, robot, config=config)
        harness.start()
        first_ep = harness.identity["episode_id"]

        # 1. reset 切换新 Episode 时通知关闭旧会话 first_ep
        harness.reset(seed=1, layout_id=0)
        second_ep = harness.identity["episode_id"]
        self.assertNotEqual(first_ep, second_ep)
        self.assertIn(first_ep, fast_server.closed_sessions)

        # 2. request_stop()（workbench.stop / 超时停止权威入口）与 close() 均幂等通知关闭当前会话
        harness.request_stop()
        harness.request_stop()
        harness.close()
        self.assertEqual(fast_server.closed_sessions.count(second_ep), 1)

        # reset 开启新 Episode 后再次单独调用 close() 亦精确通知关闭一次
        harness.reset(seed=3, layout_id=0)
        third_ep = harness.identity["episode_id"]
        harness.close()
        harness.close()
        self.assertEqual(fast_server.closed_sessions.count(third_ep), 1)

        # 2b. 模拟终态动作 _native_progress 已关闭会话后，尾步 obs3d 再次携带 third_ep 请求外部感知 -> 收尾 close() 必须再次发送 /session_close 杜绝僵尸会话
        harness.mark_perception_session_active(third_ep)
        harness.close()
        harness.close()
        self.assertEqual(fast_server.closed_sessions.count(third_ep), 2)

        # 3. 当外部感知服务慢响应（阻塞 1.5s）时，reset 与 close 必须在 <0.5s 内非阻塞返回且零异常
        slow_server = FakePerceptionServer(session_close_delay_s=1.5)
        self.addCleanup(slow_server.close)
        slow_dir = self.root / "run-slow-close"
        slow_dir.mkdir(parents=True, exist_ok=True)
        slow_store = Store(slow_dir, dict(RUN_IDENTITY, episode_id="ep-slow-1"))
        self.addCleanup(slow_store.close)
        slow_harness = Harness(
            slow_store,
            RolloutCapableRobot(slow_dir / "captures"),
            config=Config().with_overrides(
                {"features": {"external_perception": True, "vggt_endpoint": slow_server.url, "task_scheduling": False}}
            ),
        )
        slow_harness.start()
        t0 = time.monotonic()
        slow_harness.reset(seed=2, layout_id=0)
        slow_harness.close()
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 0.50)

        # 4. 诊断异常隔离：模拟 ProcessRolloutTracker.finalize 抛出异常，物理 receipt 与 result_observation_id 仍完整保留
        iso_dir = self.root / "run-diag-isolation"
        iso_dir.mkdir(parents=True, exist_ok=True)
        iso_store = Store(iso_dir, dict(RUN_IDENTITY, episode_id="ep-iso-1"))
        self.addCleanup(iso_store.close)
        iso_robot = RolloutCapableRobot(iso_dir / "captures")
        iso_harness = Harness(
            iso_store,
            iso_robot,
            config=Config().with_overrides({"features": {"external_perception": False, "task_scheduling": False}}),
        )
        iso_harness.thread_id = "thread-iso"
        iso_harness.start()
        iso_harness.task_planning.call(
            dict(operation="submit", observation_id=iso_harness.latest["observation_id"], plan="Move.", ledger=[]),
            iso_harness.identity,
            iso_harness.latest,
            dict(thread_id="thread-iso", turn_id="t0"),
        )
        with patch.object(ProcessRolloutTracker, "finalize", side_effect=RuntimeError("synthetic_tracker_boom")):
            act_wire = iso_harness.handle_tool(
                dict(
                    threadId="thread-iso",
                    turnId="t1",
                    callId="c1",
                    tool="robot_action",
                    arguments=dict(
                        mode="direct",
                        observation_id=iso_harness.latest["observation_id"],
                        stages=[dict(id="m1", kind="move", arm="left", grasp_position_m=[0.18, 0.14, 0.86], angles_deg=[0, 0, 0])],
                    ),
                )
            )
        self.assertTrue(act_wire["success"], act_wire)
        stored_receipts = list(iso_store.records("action_receipt"))
        self.assertEqual(stored_receipts[-1]["status"], "completed")
        self.assertIsNotNone(stored_receipts[-1]["result_observation_id"])
        self.assertEqual(stored_receipts[-1]["process_trajectory"]["available"], False)

    def test_persistent_id_cross_table_object_state_delta_and_fallback(self) -> None:
        """验证 track_meta.persistent_ids=True 时 >0.50m 跨全桌面搬运精确配对、主分支 yaw_delta_deg 与旧表 0.14m 回退。"""
        columns = ["id", "centre_m", "top_z", "bottom_z", "length", "width", "yaw_deg", "tilt_deg", "support", "quality"]
        before_persistent = dict(
            columns=columns,
            table_z=TABLE_Z,
            track_meta=dict(persistent_ids=True, session_id="ep-1"),
            rows=[
                ["o1", [-0.20, 0.15, 0.785], 0.805, 0.765, 0.05, 0.05, 10.0, 89.0, "table", "ok"],
                # o3 从左后方 [-0.28, -0.25, 0.785] 横跨全桌面搬运 0.62m 到右前方 [+0.30, -0.03, 0.785]，偏航角从 -85° 跨边界转至 +85°（最短差 -10°）
                ["o3", [-0.28, -0.25, 0.785], 0.815, 0.765, 0.08, 0.03, -85.0, 89.0, "table", "ok"],
            ],
        )
        after_persistent = dict(
            columns=columns,
            table_z=TABLE_Z,
            track_meta=dict(persistent_ids=True, session_id="ep-1"),
            rows=[
                ["o1", [-0.20, 0.15, 0.785], 0.805, 0.765, 0.05, 0.05, 10.0, 89.0, "table", "ok"],
                ["o3", [0.30, -0.03, 0.785], 0.815, 0.765, 0.08, 0.03, 85.0, 89.0, "table", "ok"],
            ],
        )
        delta_p = compute_object_table_delta(before_persistent, after_persistent)
        self.assertEqual(delta_p["vacated"], [])
        self.assertEqual(delta_p["appeared"], [])
        self.assertEqual(delta_p["unchanged_count"], 1)
        self.assertEqual(len(delta_p["moved"]), 1)
        m = delta_p["moved"][0]
        self.assertEqual(m["before_id"], "o3")
        self.assertEqual(m["after_id"], "o3")
        self.assertGreater(m["displacement_m"], 0.50)
        self.assertAlmostEqual(m["displacement_m"], math.hypot(0.58, 0.22), places=3)
        self.assertEqual(m["yaw_delta_deg"], -10.0)
        self.assertTrue(m["persistent_match"])

        # 无持久 ID 的旧表：0.62m 跨度超出 0.14m 半径 -> 保持原有回退逻辑分裂为 vacated + appeared；近距 <=0.14m 仍正常配对
        before_legacy = dict(before_persistent, track_meta=dict(persistent_ids=False))
        after_legacy = dict(after_persistent, track_meta=dict(persistent_ids=False))
        delta_legacy = compute_object_table_delta(before_legacy, after_legacy)
        self.assertEqual(delta_legacy["moved"], [])
        self.assertEqual(len(delta_legacy["vacated"]), 1)
        self.assertEqual(len(delta_legacy["appeared"]), 1)

        after_short = dict(
            columns=columns,
            table_z=TABLE_Z,
            rows=[
                ["o1", [-0.14, 0.15, 0.785], 0.805, 0.765, 0.05, 0.05, 25.0, 89.0, "table", "ok"],
            ],
        )
        before_short = dict(
            columns=columns,
            table_z=TABLE_Z,
            rows=[
                ["o2", [-0.20, 0.15, 0.785], 0.805, 0.765, 0.05, 0.05, 10.0, 89.0, "table", "ok"],
            ],
        )
        delta_short = compute_object_table_delta(before_short, after_short)
        self.assertEqual(len(delta_short["moved"]), 1)
        self.assertEqual(delta_short["yaw_delta_deg"] if "yaw_delta_deg" in delta_short else delta_short["moved"][0]["yaw_delta_deg"], 15.0)
        self.assertNotIn("persistent_match", delta_short["moved"][0])

    def test_half_open_preshaping_to_empty_close_no_false_slip_and_tight_grasp_slip(self) -> None:
        """验证 [pre_grasp(jaw=0.4->0.035m), grasp_miss(jaw=0.0->0.0m), lift] 绝不误报滑脱，而 [grasp_hit(0.028m), lift_slip(0.0004m)] 精准检出。"""
        rot = np.eye(3).tolist()

        def _sample_pair(step: int, ap_m: float, z_m: float, goal: float, sid: str | None) -> tuple[dict, dict]:
            half = ap_m * 0.5
            m = dict(
                native_step=step,
                arms=dict(
                    left=dict(
                        link6_pose=dict(position_m=[0.20, 0.10, z_m], rotation_matrix=rot),
                        measured_jaw=dict(joint_names=["joint7", "joint8"], position_m=[half, half]),
                        jaw_goal_normalized=goal,
                    ),
                    right=dict(
                        link6_pose=dict(position_m=[0.20, -0.15, 0.88], rotation_matrix=rot),
                        measured_jaw=dict(joint_names=["joint7", "joint8"], position_m=[0.044, 0.044]),
                        jaw_goal_normalized=1.0,
                    ),
                ),
            )
            p = dict(
                native_step=step,
                execution_state="initial_measurement" if step == 0 else "completed_command",
                arms=dict(
                    left=dict(
                        link6_pose=dict(position_m=[0.20, 0.10, z_m], rotation_matrix=rot),
                        jaw_goal_normalized=goal,
                        stage_id=sid if step > 0 else None,
                    ),
                    right=dict(
                        link6_pose=dict(position_m=[0.20, -0.15, 0.88], rotation_matrix=rot),
                        jaw_goal_normalized=1.0,
                    ),
                ),
            )
            return m, p

        # 1) 反例（根因 3 核心回归）：半开预缩爪 jaw=0.4 (0.035m) -> 主动空抓闭合 jaw=0.0 (0.0m) -> 抬升 (0.0m)：绝不误判为搬运滑脱
        miss_specs = [
            (0, 0.088, 0.88, 1.0, None),
            (1, 0.035, 0.84, 0.4, "pre_grasp"),
            (2, 0.035, 0.81, 0.4, "pre_grasp"),
            (3, 0.014, 0.81, 0.0, "grasp_miss"),
            (4, 0.000, 0.81, 0.0, "grasp_miss"),
            (5, 0.000, 0.87, 0.0, "lift"),
            (6, 0.000, 0.92, 0.0, "lift"),
        ]
        m_miss, p_miss = zip(*(_sample_pair(*sp) for sp in miss_specs))
        action_miss = dict(
            stages=[
                dict(id="pre_grasp", arm="left", kind="move", jaw_goal_normalized=0.4),
                dict(id="grasp_miss", arm="left", kind="gripper", goal_normalized=0.0),
                dict(id="lift", arm="left", kind="move"),
            ]
        )
        pt_miss = build_process_trajectory(
            list(m_miss),
            action=action_miss,
            receipt=dict(status="completed", execution_start_observation_id="obs-s", result_observation_id="obs-e"),
            planned_samples=list(p_miss),
        )
        self.assertFalse(pt_miss["co_motion"]["left"]["slip_detected"])
        self.assertEqual(pt_miss["co_motion"]["left"]["transport_aperture_drop_m"], 0.0)
        self.assertIsNone(pt_miss["co_motion"]["left"]["slip_onset_step"])

        # 无显式 stage_id 时仅凭 jaw_goal_normalized 序列同样不得误报
        pt_miss_untagged = build_process_trajectory(
            list(m_miss),
            receipt=dict(status="completed", execution_start_observation_id="obs-s", result_observation_id="obs-e"),
        )
        self.assertFalse(pt_miss_untagged["co_motion"]["left"]["slip_detected"])

        # 2) 正例：[grasp_hit(jaw=0.0 -> 0.028m), lift_slip(jaw=0.0 -> 0.0004m)] 精准检出 slip_detected=True
        hit_slip_specs = [
            (0, 0.088, 0.88, 1.0, None),
            (1, 0.035, 0.82, 0.4, "pre_grasp"),
            (2, 0.028, 0.81, 0.0, "grasp_hit"),
            (3, 0.028, 0.86, 0.0, "lift_slip"),
            (4, 0.0004, 0.91, 0.0, "lift_slip"),
        ]
        m_hit, p_hit = zip(*(_sample_pair(*sp) for sp in hit_slip_specs))
        action_hit = dict(
            stages=[
                dict(id="pre_grasp", arm="left", kind="move", jaw_goal_normalized=0.4),
                dict(id="grasp_hit", arm="left", kind="gripper", goal_normalized=0.0),
                dict(id="lift_slip", arm="left", kind="move"),
            ]
        )
        pt_hit = build_process_trajectory(
            list(m_hit),
            action=action_hit,
            receipt=dict(status="completed", execution_start_observation_id="obs-s2", result_observation_id="obs-e2"),
            planned_samples=list(p_hit),
        )
        self.assertTrue(pt_hit["co_motion"]["left"]["slip_detected"])
        self.assertEqual(pt_hit["co_motion"]["left"]["slip_onset_step"], 4)
        self.assertAlmostEqual(pt_hit["co_motion"]["left"]["transport_aperture_drop_m"], 0.0276, places=4)

    def test_anti_head_only_guidance_and_stationary_track_id_swap_guard(self) -> None:
        """验证指令清除劝退看图话术并保留完整枚举说明，且 persistent_ids=True 下两静止物体互换 ID 不谎报位移。"""
        rendered_inst = instructions(CALIBRATED_OBSERVATION_INSTRUCTIONS, "Autonomous run.")
        img_desc = ROBOT_ACTION_TOOL["inputSchema"]["properties"]["image_delivery"]["description"]
        for mode_token in ("on_anomaly", "head_only", "all", "slip_detected"):
            self.assertIn(mode_token, img_desc)
        for doc in (INSTRUCTIONS, rendered_inst, OPERATION_NOTES):
            self.assertIn("object_state_delta", doc)
            self.assertNotIn("0-image closed loop", doc)
            self.assertNotIn("do not reflexively", doc)
            self.assertNotIn("s3.virtual_field", doc)

        cols = ["id", "centre_m", "top_z", "bottom_z", "length", "width", "yaw_deg", "tilt_deg", "support", "quality"]
        before_swap = dict(
            columns=cols,
            track_meta=dict(persistent_ids=True),
            rows=[
                ["o1", [0.395, 0.086, 0.790], 0.840, 0.765, 0.16, 0.12, 0.0, None, "table", "ok"],
                ["o3", [0.118, 0.229, 0.780], 0.805, 0.765, 0.20, 0.20, 10.0, None, "table", "ok"],
            ],
        )
        after_swap = dict(
            columns=cols,
            track_meta=dict(persistent_ids=True),
            rows=[
                ["o3", [0.397, 0.085, 0.790], 0.840, 0.765, 0.16, 0.12, 0.0, None, "table", "ok"],
                ["o1", [0.116, 0.230, 0.780], 0.805, 0.765, 0.20, 0.20, 10.0, None, "table", "ok"],
            ],
        )
        delta_swap = compute_object_table_delta(before_swap, after_swap)
        self.assertEqual(delta_swap["moved"], [])
        self.assertEqual(delta_swap["unchanged_count"], 2)

        after_transport = dict(
            columns=cols,
            track_meta=dict(persistent_ids=True),
            rows=[
                ["o1", [-0.210, -0.050, 0.790], 0.840, 0.765, 0.16, 0.12, 45.0, None, "table", "ok"],
                ["o3", [0.118, 0.229, 0.780], 0.805, 0.765, 0.20, 0.20, 10.0, None, "table", "ok"],
            ],
        )
        delta_transport = compute_object_table_delta(before_swap, after_transport)
        self.assertEqual(len(delta_transport["moved"]), 1)
        self.assertEqual(delta_transport["moved"][0]["before_id"], "o1")
        self.assertTrue(delta_transport["moved"][0]["persistent_match"])
        self.assertGreater(delta_transport["moved"][0]["displacement_m"], 0.60)

    def test_compact_summary_reads_measured_aperture_m_with_legacy_fallback(self) -> None:
        """验证未进入采样步数的阶段优先从 grip_facts.measured_aperture_m 读取终态孔径，并向后兼容 aperture_after_m。"""
        rot = np.eye(3).tolist()
        sample_0 = dict(
            native_step=0,
            arms=dict(
                left=dict(
                    link6_pose=dict(position_m=[0.20, 0.12, 0.85], rotation_matrix=rot),
                    measured_jaw=dict(joint_names=["joint7", "joint8"], position_m=[0.044, 0.044]),
                    jaw_goal_normalized=1.0,
                )
            ),
        )
        planned_0 = dict(
            native_step=0,
            execution_state="initial_measurement",
            arms=dict(
                left=dict(
                    link6_pose=dict(position_m=[0.20, 0.12, 0.85], rotation_matrix=rot),
                    jaw_goal_normalized=1.0,
                )
            ),
        )
        action = dict(
            stages=[
                dict(id="grasp_measured", arm="left", kind="gripper", goal_normalized=0.0),
                dict(id="grasp_legacy", arm="left", kind="gripper", goal_normalized=0.0),
                dict(id="grasp_both", arm="left", kind="gripper", goal_normalized=0.0),
            ]
        )
        receipt = dict(
            status="stopped",
            executed_steps=0,
            completed_native_steps=0,
            stage_results=dict(
                grasp_measured=dict(kind="gripper", status="completed", grip_facts=dict(measured_aperture_m=0.01856)),
                grasp_legacy=dict(kind="gripper", status="completed", grip_facts=dict(aperture_after_m=0.01424)),
                grasp_both=dict(
                    kind="gripper",
                    status="completed",
                    grip_facts=dict(measured_aperture_m=0.02111, aperture_after_m=0.00999),
                ),
            ),
        )
        summary = build_compact_trajectory_summary(
            dict(
                measured=dict(frame_id=FRAME_ID, executed_native_steps=0, samples=[sample_0]),
                planned=dict(frame_id=FRAME_ID, samples=[planned_0]),
            ),
            action,
            receipt,
        )
        by_stage = {st["stage_id"]: st for st in summary.get("stages", [])}
        self.assertEqual(by_stage["grasp_measured"]["measured_aperture_end_m"], 0.0186)
        self.assertEqual(by_stage["grasp_legacy"]["measured_aperture_end_m"], 0.0142)
        self.assertEqual(by_stage["grasp_both"]["measured_aperture_end_m"], 0.0211)



if __name__ == "__main__":
    unittest.main()
