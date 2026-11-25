"""Shared CPU robot doubles extracted from runtime and perception test suites."""
from __future__ import annotations

import json
from pathlib import Path
import struct
import zlib

import cv2
import numpy as np
from PIL import Image

from ni_rua.robot.backends.robodojo.robodojo import FRAME_ID, TCP_CONFIG_ID


class FixtureRobot:
    """Deterministic protocol fixture; no physics, IK, task state, or GPU."""

    def __init__(self, root, outcome="completed"):
        self.executions = 0
        self.outcome = outcome
        self.path = Path(root) / "synthetic-rgb.png"
        self.path.parent.mkdir(parents=True, exist_ok=True)

        def chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

        self.path.write_bytes(
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00"))
            + chunk(b"IEND", b"")
        )

    def start(self):
        return dict(
            data_origin="synthetic_fixture",
            frame_id=FRAME_ID,
            tcp_config_id=TCP_CONFIG_ID,
            path_kinds=["backend_free"],
            max_waypoints=1,
            dual_arm=False,
            gripper_after_motion=True,
        )

    def observe(self):
        return dict(
            data_origin="synthetic_fixture",
            world_kind="observed",
            sim_step=self.executions,
            step_domain="adapter_control_step",
            sim_time_s=None,
            captured_at="2026-09-30T00:00:00Z",
            cameras=[dict(camera_id="fixture_camera", png_path=str(self.path), width=1, height=1)],
            robot_state=dict(
                data_origin="synthetic_fixture",
                source_kind="measurement",
                arms=dict(
                    left=dict(
                        q_rad=[0.0],
                        ee_pose=dict(
                            position_m=[0.009, 0, 1],
                            quaternion_xyzw=[0, 0, 0, 1],
                            source_kind="measurement",
                        ),
                    )
                ),
            ),
        )

    def execute(self, action, progress_callback=None, cancel_flag=None):
        self.executions += 1
        if progress_callback:
            progress_callback(dict(phase="started", sim_step=self.executions, data_origin="synthetic_fixture"))
        return dict(
            status=self.outcome,
            reason="synthetic_fixture",
            data_origin="synthetic_fixture",
            terminal_observation=self.observe(),
            planned_trace=dict(data_origin="synthetic_fixture", targets=action["path"]),
            dispatched_trace=dict(data_origin="synthetic_fixture", steps=1),
            actual_trace=[dict(data_origin="synthetic_fixture", measured_fixture_state=self.observe()["robot_state"])],
        )


class WaitFixtureRobot(FixtureRobot):
    """A deterministic native-step/observation producer, without simulator or model."""

    def __init__(self, root):
        super().__init__(root)
        self.native_count = 0
        self.cancel_at = None

    def start(self):
        return super().start() | dict(
            path_kinds=["backend_free", "hold"],
            bounded_wait=dict(max_native_steps=50, native_control_hz=25),
        )

    def observe(self):
        observation = super().observe()
        observation.update(
            sim_step=self.native_count * 10,
            sim_time_s=self.native_count / 25,
            instruction="Observe the visible scene before acting.",
        )
        observation["cameras"][0]["camera_id"] = "cam_head"
        observation["robot_state"]["arms"]["left"]["q_rad"] = [0.123456789] * 6
        return observation

    def execute(self, action, progress_callback=None, cancel_flag=None):
        self.executions += 1
        start = self.native_count
        before = self.observe()
        count = 0
        for count in range(1, action["hold_native_steps"] + 1):
            self.native_count += 1
            if progress_callback and (count == 1 or count % 5 == 0):
                progress_callback(
                    dict(
                        phase="executing",
                        executed_steps=count,
                        observation=self.observe(),
                        episode_terminal=self.terminal(),
                    )
                )
            if count == self.cancel_at:
                cancel_flag.set()
                break
        cancelled = cancel_flag.is_set()
        return dict(
            status="stopped" if cancelled else "completed",
            reason="cancelled" if cancelled else None,
            before_observation=before,
            before_state=before["robot_state"],
            terminal_observation=self.observe(),
            episode_terminal=self.terminal(),
            executed_native_steps=count,
            completed_native_steps=count,
            executed_control_steps=count * 10,
            execution_interval=dict(start_sim_step=start * 10, end_sim_step=self.native_count * 10),
            versions=dict(fixture="full-precision-source"),
            step_semantics=dict(executed_native_steps="native action count"),
            planned_trace=dict(actions=action["hold_native_steps"]),
            dispatched_trace=dict(actions=count),
            actual_trace=[dict(state=self.observe()["robot_state"])],
        )

    def terminal(self):
        return dict(ended=False, success=None, reason=None, native_action_count=self.native_count, native_step_limit=1000)


def text_values(reply):
    return [json.loads(item["text"]) for item in reply["contentItems"] if item["type"] == "inputText"]


def _look_at_world_camera(cam_pos_w, target_w, up_w=(0.0, 0.0, 1.0)) -> np.ndarray:
    """构造 OpenCV 光学约定 (+X 右, +Y 下, +Z 光轴向前) 的相机世界位姿矩阵 `T_world_camera`。"""
    c = np.asarray(cam_pos_w, dtype=np.float64)
    t = np.asarray(target_w, dtype=np.float64)
    up = np.asarray(up_w, dtype=np.float64)
    z_axis = t - c
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(z_axis, up)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.cross(z_axis, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, 0] = x_axis
    mat[:3, 1] = y_axis
    mat[:3, 2] = z_axis
    mat[:3, 3] = c
    return mat


def _pinhole_k(fx: float = 520.0, fy: float = 520.0, cx: float = 320.0, cy: float = 240.0) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


class CalibratedThreeCamRobot:
    """确定性三相机（头部固定 + 左右腕部实时关节 FK）合成标定机器人夹具，不依赖仿真器或 GPU。"""

    def __init__(self, root: Path, *, width: int = 640, height: int = 480, marker_world_m=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.executions = 0
        self.q_left = 0.0
        self.q_right = 0.0
        self.marker_world_m = list(marker_world_m) if marker_world_m is not None else [0.03, -0.36, 0.82]

        self.k_mat = _pinhole_k(520.0, 520.0, width * 0.5, height * 0.5)
        self.t_head = _look_at_world_camera([0.0, -0.05, 1.28], [0.0, -0.35, 0.80])
        self.t_left_base = np.eye(4, dtype=np.float64)
        self.t_left_base[:3, 3] = [-0.14, -0.22, 0.96]
        self.t_left_link_cam = np.linalg.inv(self.t_left_base) @ _look_at_world_camera(
            [-0.14, -0.22, 0.96], [0.0, -0.35, 0.80]
        )
        self.t_right_base = np.eye(4, dtype=np.float64)
        self.t_right_base[:3, 3] = [0.42, -0.10, 1.05]
        self.t_right_link_cam = np.linalg.inv(self.t_right_base) @ _look_at_world_camera(
            [0.42, -0.10, 1.05], [0.85, 0.35, 1.10]
        )

    def _wrist_world_pose(self, base_mat: np.ndarray, q_shift_x: float, link_cam: np.ndarray) -> np.ndarray:
        motion = np.eye(4, dtype=np.float64)
        motion[0, 3] = float(q_shift_x)
        return base_mat @ motion @ link_cam

    def _render_sensor_png(self, cam_id: str, t_wc: np.ndarray) -> Path:
        """生成带高频像素网格纹理与精确三维地标投影亮斑的无损传感器 PNG。"""
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        checker = (((xx + yy) % 2) * 110).astype(np.uint8)
        r = np.clip(40 + checker, 0, 255).astype(np.uint8)
        g = np.clip(40 + checker, 0, 255).astype(np.uint8)
        b = np.clip(40 + checker, 0, 255).astype(np.uint8)
        rgb = np.stack([r, g, b], axis=-1)

        inv = np.linalg.inv(t_wc)
        pt_w = np.asarray(self.marker_world_m, dtype=np.float64)
        pt_c = inv[:3, :3] @ pt_w + inv[:3, 3]
        if pt_c[2] > 0.02:
            uv = cv2.projectPoints(
                pt_c.reshape(1, 1, 3),
                np.zeros((3, 1)),
                np.zeros((3, 1)),
                self.k_mat,
                np.zeros(5),
            )[0].reshape(2)
            u_px, v_px = int(round(float(uv[0]))), int(round(float(uv[1])))
            if 2 <= u_px < self.width - 2 and 2 <= v_px < self.height - 2:
                rgb[v_px - 2 : v_px + 3, u_px - 2 : u_px + 3] = np.array([210, 30, 30], dtype=np.uint8)
                rgb[v_px - 1 : v_px + 2, u_px - 1 : u_px + 2] = np.array([240, 10, 10], dtype=np.uint8)
                rgb[v_px, u_px] = np.array([255, 0, 0], dtype=np.uint8)

        out = self.root / f"{cam_id}_step_{self.executions}_{self.q_left:.3f}.png"
        Image.fromarray(rgb, mode="RGB").save(out, format="PNG")
        return out

    def start(self) -> dict:
        return dict(
            data_origin="synthetic_fixture",
            robot_model_id="arx-x5",
            observation_profile="rgb_proprio_calibrated_v1",
            frame_id=FRAME_ID,
            tcp_config_id=TCP_CONFIG_ID,
            path_kinds=["backend_free"],
            max_waypoints=4,
            dual_arm=True,
            gripper_after_motion=True,
        )

    def observe(self) -> dict:
        t_left = self._wrist_world_pose(self.t_left_base, self.q_left, self.t_left_link_cam)
        t_right = self._wrist_world_pose(self.t_right_base, self.q_right, self.t_right_link_cam)
        k_list = self.k_mat.tolist()
        size = [self.width, self.height]

        cam_defs = [
            ("cam_head", self.t_head, "fixed_calibration"),
            ("cam_left_wrist", t_left, "robot_kinematics"),
            ("cam_right_wrist", t_right, "robot_kinematics"),
        ]
        cameras = []
        pose_cameras = {}
        for cid, t_wc, src in cam_defs:
            png = self._render_sensor_png(cid, t_wc)
            cameras.append(
                dict(
                    camera_id=cid,
                    png_path=str(png),
                    width=self.width,
                    height=self.height,
                )
            )
            pose_cameras[cid] = dict(
                T_world_camera=t_wc.tolist(),
                source_kind=src,
            )

        calibration = dict(
            profile_id="calibrated_fixture_v1",
            schema_version="0.1",
            reference_frame_id=FRAME_ID,
            cameras=dict(
                cam_head=dict(
                    K=k_list,
                    distortion=[0.0, 0.0, 0.0, 0.0, 0.0],
                    image_size=size,
                    T_world_camera=self.t_head.tolist(),
                ),
                cam_left_wrist=dict(
                    K=k_list,
                    distortion=[0.0, 0.0, 0.0, 0.0, 0.0],
                    image_size=size,
                    arm="left",
                    link_name="link6",
                    T_world_base=self.t_left_base.tolist(),
                    T_link_camera=self.t_left_link_cam.tolist(),
                ),
                cam_right_wrist=dict(
                    K=k_list,
                    distortion=[0.0, 0.0, 0.0, 0.0, 0.0],
                    image_size=size,
                    arm="right",
                    link_name="link6",
                    T_world_base=self.t_right_base.tolist(),
                    T_link_camera=self.t_right_link_cam.tolist(),
                ),
            ),
        )
        robot_model = dict(
            base_link="base",
            T_world_base=np.eye(4, dtype=np.float64).tolist(),
            joints=[
                dict(
                    name="j1",
                    type="prismatic",
                    parent="base",
                    child="link6",
                    T_parent_joint=np.eye(4, dtype=np.float64).tolist(),
                    axis=[1.0, 0.0, 0.0],
                )
            ],
        )
        robot_state = dict(
            data_origin="synthetic_fixture",
            source_kind="measurement",
            sim_step=self.executions,
            step_domain="adapter_control_step",
            arms=dict(
                left=dict(
                    source_kind="measurement",
                    joint_names=["j1"],
                    q_rad=[float(self.q_left)],
                    gripper=dict(joint_names=["g1"], position_m=[0.04]),
                    ee_pose=dict(
                        position_m=[-0.14 + self.q_left, -0.22, 0.96],
                        quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
                        frame_id=FRAME_ID,
                        tcp_config_id=TCP_CONFIG_ID,
                        source_kind="measurement",
                    ),
                ),
                right=dict(
                    source_kind="measurement",
                    joint_names=["j1"],
                    q_rad=[float(self.q_right)],
                    gripper=dict(joint_names=["g1"], position_m=[0.04]),
                    ee_pose=dict(
                        position_m=[0.42 + self.q_right, -0.10, 1.05],
                        quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
                        frame_id=FRAME_ID,
                        tcp_config_id=TCP_CONFIG_ID,
                        source_kind="measurement",
                    ),
                ),
            ),
        )
        return dict(
            data_origin="synthetic_fixture",
            world_kind="observed",
            observation_profile="rgb_proprio_calibrated_v1",
            clock_domain="fixture-env/generation-1",
            sim_step=self.executions,
            step_domain="adapter_control_step",
            sim_time_s=float(self.executions) * 0.04,
            captured_at="2026-11-11T11:30:00Z",
            cameras=cameras,
            camera_calibration=calibration,
            camera_pose_sample=dict(
                sim_step=self.executions,
                step_domain="adapter_control_step",
                reference_frame_id=FRAME_ID,
                cameras=pose_cameras,
            ),
            robot_model=robot_model,
            robot_state=robot_state,
        )

    def execute(self, action, progress_callback=None, cancel_flag=None) -> dict:
        self.executions += 1
        if progress_callback:
            progress_callback(dict(phase="started", sim_step=self.executions, data_origin="synthetic_fixture"))
        obs = self.observe()
        return dict(
            status="completed",
            reason=None,
            executed_steps=1,
            executed_native_steps=1,
            completed_native_steps=1,
            data_origin="synthetic_fixture",
            terminal_observation=obs,
            planned_trace=dict(data_origin="synthetic_fixture", targets=action.get("path")),
            dispatched_trace=dict(data_origin="synthetic_fixture", steps=1),
            actual_trace=[dict(data_origin="synthetic_fixture", measured_fixture_state=obs["robot_state"])],
        )
