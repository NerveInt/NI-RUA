"""阶段二支柱三单元测试：无标定板在线手眼自标定、硬件曝光时钟域、三维阴影雕刻与走廊安全监护。"""
from __future__ import annotations

NI_RUA_TEST_TIER = "component"

from copy import deepcopy
from pathlib import Path
import struct
import tempfile
import time
import unittest
import zlib

import cv2
import numpy as np

from ni_rua.core.contracts import canonical, digest
from ni_rua.core.observation_policy import reject_recorded_action_privileged
from ni_rua.core.storage.store import Store
from ni_rua.robot.model.calibration_correction import (
    CalibrationCorrection,
    HARDWARE_MONOTONIC_STEP_DOMAIN,
    rodrigues_exp_rad,
)
from ni_rua.perception.external.sensor_frame_contract import (
    DepthFrameSpec,
    StereoPairSpec,
    StreamBudgetConfig,
)
from ni_rua.perception.scene.scene_contracts import validate_response
from ni_rua.perception.scene.store import SceneStore
from ni_rua.robot.backends.robodojo.robodojo import FRAME_ID, TCP_CONFIG_ID
from ni_rua.robot.model.geometry import GeometryTools
from ni_rua.runtime.harness import Harness

TEST_IDENTITY = {
    "run_id": "test-s2-calib-stream",
    "episode_id": "ep-s2-1",
    "env_instance_id": "real-rig-sim",
    "env_generation": 1,
}


def _make_minimal_png_bytes(width: int = 640, height: int = 480) -> bytes:
    """生成合法纯色 PNG 字节流用于合成相机观测工件。"""
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _look_at_camera_pose(cam_pos: list[float] | np.ndarray, target_pos: list[float] | np.ndarray) -> np.ndarray:
    """构造光学坐标系（右+X / 下+Y / 前+Z）看向目标点的标准 4×4 刚体位姿矩阵 `T_world_camera`。"""
    c = np.asarray(cam_pos, dtype=np.float64).reshape(3)
    t = np.asarray(target_pos, dtype=np.float64).reshape(3)
    z_axis = t - c
    z_axis /= np.linalg.norm(z_axis)
    up_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(z_axis, up_world))) > 0.98:
        up_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = np.cross(z_axis, up_world)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    T = np.eye(4, dtype=np.float64)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[:3, 3] = c
    return T


def _project_points_pinhole(
    K: np.ndarray,
    T_world_cam: np.ndarray,
    points_world: np.ndarray,
) -> np.ndarray:
    """在零畸变针孔相机模型下将世界系三维点投影为像素坐标 `(N, 2)`。"""
    R_wc = T_world_cam[:3, :3]
    p_wc = T_world_cam[:3, 3]
    pts_cam = (points_world - p_wc[None, :]) @ R_wc
    u = K[0, 0] * (pts_cam[:, 0] / pts_cam[:, 2]) + K[0, 2]
    v = K[1, 1] * (pts_cam[:, 1] / pts_cam[:, 2]) + K[1, 2]
    return np.column_stack((u, v))


def _build_two_joint_robot_model() -> dict:
    """构造含基座、右臂两自由度旋转关节与 `right_link6` 末端的最小可解析机器人运动学清单。"""
    T_j1 = np.eye(4, dtype=np.float64)
    T_j1[:3, 3] = [0.0, -0.20, 0.80]
    T_j2 = np.eye(4, dtype=np.float64)
    T_j2[:3, 3] = [0.25, 0.0, 0.05]
    return {
        "base_link": "base",
        "T_world_base": np.eye(4, dtype=np.float64).tolist(),
        "joints": [
            {
                "name": "right_joint1",
                "parent": "base",
                "child": "right_link1",
                "type": "revolute",
                "axis": [0.0, 0.0, 1.0],
                "T_parent_joint": T_j1.tolist(),
            },
            {
                "name": "right_joint2",
                "parent": "right_link1",
                "child": "right_link6",
                "type": "revolute",
                "axis": [0.0, 1.0, 0.0],
                "T_parent_joint": T_j2.tolist(),
            },
        ],
    }


class GeometryAndSceneOnlineCalibrationTests(unittest.TestCase):
    """验证 `geometry.py` 与 `scene/store.py` 的在线外参修正接线及 `hardware_monotonic_ns` 时钟域契约。"""

    def setUp(self) -> None:
        self.tmpdir = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(self.tmpdir)
        self.artifact_store = Store(root / "store", TEST_IDENTITY)
        self.addCleanup(self.artifact_store.close)
        self.scene = SceneStore(
            root / "scene",
            self.artifact_store.resolve_artifact,
            self.artifact_store.put_artifact,
        )
        self.geometry = GeometryTools(self.scene)

        self.K = np.array([[520.0, 0.0, 320.0], [0.0, 520.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.T_fixed_nom = _look_at_camera_pose([0.55, -0.50, 1.25], [0.35, 0.0, 0.80])
        # 腕部相机相对 right_link6 的固定安装矩阵（看向工作台下方）
        self.T_link_cam = _look_at_camera_pose([0.05, 0.0, 0.06], [0.22, 0.0, -0.18])
        self.robot_model = _build_two_joint_robot_model()

        self.calibration_doc = {
            "reference_frame_id": "env_origin",
            "cameras": {
                "cam_third_person": {
                    "K": self.K.tolist(),
                    "distortion": {"model": "opencv", "coefficients": [0.0, 0.0, 0.0, 0.0, 0.0]},
                    "image_size": [640, 480],
                    "T_world_camera": self.T_fixed_nom.tolist(),
                },
                "cam_right_wrist": {
                    "K": self.K.tolist(),
                    "distortion": {"model": "opencv", "coefficients": [0.0, 0.0, 0.0, 0.0, 0.0]},
                    "image_size": [640, 480],
                    "arm": "right",
                    "link_name": "right_link6",
                    "T_link_camera": self.T_link_cam.tolist(),
                },
            },
        }
        self.calibration_id = digest(self.calibration_doc)
        self.calibration_ref = self.artifact_store.put_artifact(
            canonical(self.calibration_doc).encode(), "camera_calibration", "application/json"
        )
        self.model_ref = self.artifact_store.put_artifact(
            canonical(self.robot_model).encode(), "robot_model", "application/json"
        )
        self.rgb_ref = self.artifact_store.put_artifact(
            _make_minimal_png_bytes(640, 480), "rgb", "image/png"
        )

    def _import_hw_observation(
        self,
        obs_id: str,
        *,
        obs_ns: int,
        cam_skew_ns: int = 6_000_000,
        state_skew_ns: int = -3_500_000,
        q_rad: tuple[float, float] = (0.15, 0.25),
        correction: CalibrationCorrection | dict | None = None,
        step_domain: str = HARDWARE_MONOTONIC_STEP_DOMAIN,
        use_corrected_sample_pose: bool = False,
    ) -> dict:
        from ni_rua.robot.model.geometry import fk as eval_fk

        state_step = obs_ns + state_skew_ns if step_domain == HARDWARE_MONOTONIC_STEP_DOMAIN else obs_ns
        cam_step = obs_ns + cam_skew_ns if step_domain == HARDWARE_MONOTONIC_STEP_DOMAIN else obs_ns

        robot_state = {
            "source_kind": "measurement",
            "sim_step": state_step,
            "step_domain": step_domain,
            "arms": {
                "right": {
                    "source_kind": "measurement",
                    "joint_names": ["right_joint1", "right_joint2"],
                    "q_rad": list(q_rad),
                    "ee_pose": {
                        "position_m": [0.30, -0.15, 0.85],
                        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                        "frame_id": "env_origin",
                        "tcp_config_id": TCP_CONFIG_ID,
                        "source_kind": "measurement",
                    },
                    "gripper": {"joint_names": [], "position_m": []},
                }
            },
        }
        fk_poses = eval_fk(
            self.robot_model,
            {"source_kind": "measurement", "joint_names": ["right_joint1", "right_joint2"], "q_rad": list(q_rad)},
        )
        T_wrist_nom = fk_poses["right_link6"] @ self.T_link_cam

        corr_obj = CalibrationCorrection.from_mapping(correction) if correction is not None else None
        T_fixed_sample = (
            corr_obj.apply_to_camera_pose(self.T_fixed_nom, "cam_third_person")
            if (corr_obj is not None and use_corrected_sample_pose)
            else self.T_fixed_nom
        )
        T_wrist_sample = (
            corr_obj.apply_to_camera_pose(T_wrist_nom, "cam_right_wrist")
            if (corr_obj is not None and use_corrected_sample_pose)
            else T_wrist_nom
        )

        pose_sample = {
            "sim_step": cam_step,
            "step_domain": step_domain,
            "cameras": {
                "cam_third_person": {
                    "T_world_camera": T_fixed_sample.tolist(),
                    "frame_id": f"{obs_id}-tp",
                    "source_kind": "online_calibrated_fixed" if use_corrected_sample_pose else "fixed_calibration",
                },
                "cam_right_wrist": {
                    "T_world_camera": T_wrist_sample.tolist(),
                    "T_world_link": fk_poses["right_link6"].tolist(),
                    "frame_id": f"{obs_id}-rw",
                    "source_kind": "online_calibrated_kinematics" if use_corrected_sample_pose else "robot_kinematics",
                },
            },
        }
        state_ref = self.artifact_store.put_artifact(
            canonical(robot_state).encode(), "robot_state", "application/json"
        )
        pose_ref = self.artifact_store.put_artifact(
            canonical(pose_sample).encode(), "camera_pose_sample", "application/json"
        )

        bundle: dict = {
            "schema_version": "0.1",
            **TEST_IDENTITY,
            "world_kind": "observed",
            "data_origin": "synthetic_fixture",
            "observation_id": obs_id,
            "sim_step": obs_ns,
            "step_domain": step_domain,
            "sim_time_s": float(obs_ns) * 1e-9 if step_domain == HARDWARE_MONOTONIC_STEP_DOMAIN else float(obs_ns),
            "clock_domain": "real-rig-sim/generation-1",
            "observation_profile": "rgb_proprio_calibrated_v1",
            "calibration_id": self.calibration_id,
            "robot_model_id": "arx-x5",
            "robot_model_ref": self.model_ref,
            "robot_state_ref": state_ref,
            "cameras": [
                {
                    "camera_id": "cam_third_person",
                    "frame_id": f"{obs_id}-tp",
                    "width": 640,
                    "height": 480,
                    "sim_step": cam_step,
                    "step_domain": step_domain,
                    "rgb_ref": self.rgb_ref,
                    "calibration_ref": self.calibration_ref,
                    "pose_sample_ref": pose_ref,
                    "pixel_transform": "identity",
                },
                {
                    "camera_id": "cam_right_wrist",
                    "frame_id": f"{obs_id}-rw",
                    "width": 640,
                    "height": 480,
                    "sim_step": cam_step,
                    "step_domain": step_domain,
                    "rgb_ref": self.rgb_ref,
                    "calibration_ref": self.calibration_ref,
                    "pose_sample_ref": pose_ref,
                    "pixel_transform": "identity",
                },
            ],
        }
        if corr_obj is not None:
            bundle["calibration_correction"] = corr_obj.to_dict()
        self.scene.import_observation(bundle)
        return bundle

    def test_fixed_third_person_and_wrist_online_correction_preserves_calibration_id(self) -> None:
        """固定第三方视角与随动腕部相机注入 `calibration_correction` 后投影精确修正，且跨帧 `calibration_id` 恒定。"""
        obs0 = self._import_hw_observation("obs-hw-0", obs_ns=1_000_000_000)
        corr1 = CalibrationCorrection(
            calib_rev=1,
            delta_t_base_cam_m={
                "cam_third_person": (0.015, -0.010, 0.008),
                "cam_right_wrist": (-0.012, 0.009, 0.006),
            },
            delta_rot_base_cam_deg={
                "cam_third_person": (1.2, -0.8, 1.5),
                "cam_right_wrist": (-0.9, 1.1, -0.7),
            },
            time_offset_ms=3.5,
            reprojection_rmse_px=0.48,
            position_sigma_m=0.0013,
        )
        obs1 = self._import_hw_observation(
            "obs-hw-1",
            obs_ns=1_100_000_000,
            correction=corr1,
            use_corrected_sample_pose=True,
        )
        corr2 = CalibrationCorrection(
            calib_rev=2,
            delta_t_base_cam_m={
                "cam_third_person": (0.016, -0.009, 0.007),
                "cam_right_wrist": (-0.011, 0.010, 0.005),
            },
            delta_rot_base_cam_deg={
                "cam_third_person": (1.1, -0.7, 1.4),
                "cam_right_wrist": (-0.8, 1.0, -0.6),
            },
            time_offset_ms=3.6,
            reprojection_rmse_px=0.35,
            position_sigma_m=0.0011,
        )
        obs2 = self._import_hw_observation(
            "obs-hw-2",
            obs_ns=1_200_000_000,
            correction=corr2,
            use_corrected_sample_pose=False,
        )

        # 1. 静态 calibration_id 跨在线标定修订保持 100% 恒定
        self.assertEqual(obs0["calibration_id"], self.calibration_id)
        self.assertEqual(obs1["calibration_id"], self.calibration_id)
        self.assertEqual(obs2["calibration_id"], self.calibration_id)

        # 2. 校验第三方固定相机与腕部相机的修正位姿与像素重投影精度
        target_pt = np.array([[0.34, -0.02, 0.78]], dtype=np.float64)
        for cam_id, expected_source in (
            ("cam_third_person", "online_calibrated_fixed"),
            ("cam_right_wrist", "online_calibrated_kinematics"),
        ):
            cam_uncorr = self.geometry.camera(TEST_IDENTITY, "obs-hw-0", cam_id)
            pose_res = self.geometry.execute(
                {**TEST_IDENTITY, "kind": "camera_pose", "observation_id": "obs-hw-1", "camera_id": cam_id}
            )
            validate_response("geometry/query", pose_res, method="POST")
            self.assertEqual(pose_res["status"], "supported")
            self.assertEqual(pose_res["calibration_id"], self.calibration_id)
            self.assertEqual(pose_res["camera_pose_source"], expected_source)

            expected_T = corr1.apply_to_camera_pose(cam_uncorr["T"], cam_id)
            np.testing.assert_allclose(np.asarray(pose_res["value"]["T_world_camera"]), expected_T, atol=1e-9)

            diag = pose_res["camera_pose_diagnostics"]["calibration_correction"]
            self.assertEqual(diag["calib_rev"], 1)
            self.assertTrue(diag["applied"])
            self.assertAlmostEqual(diag["reprojection_rmse_px"], 0.48, places=4)
            self.assertAlmostEqual(diag["position_sigma_m"], 0.0013, places=6)

            proj_res = self.geometry.execute(
                {
                    **TEST_IDENTITY,
                    "kind": "project_points",
                    "observation_id": "obs-hw-1",
                    "camera_id": cam_id,
                    "points_m": target_pt.tolist(),
                }
            )
            validate_response("geometry/query", proj_res, method="POST")
            expected_uv = _project_points_pinhole(self.K, expected_T, target_pt)
            np.testing.assert_allclose(np.asarray(proj_res["value"]["pixels"]), expected_uv, atol=1e-6)

            # 确认与未修正相机像素相比确实产生了显著毫米级物理修正（> 5 px）
            uncorr_uv = _project_points_pinhole(self.K, cam_uncorr["T"], target_pt)
            self.assertGreater(float(np.linalg.norm(expected_uv - uncorr_uv)), 5.0)

        # 3. 验证 SceneStore 跨在线修正版本增量追加补丁无需 rebuild=True，且拒绝回退与乱序时钟
        base_op = {
            "op": "upsert_geometry",
            "entity_id": "mug_0",
            "source_kind": "observation_inference",
            "geometry": {"kind": "box", "min_m": [0.30, -0.05, 0.76], "max_m": [0.38, 0.03, 0.86]},
            "assumptions": [],
        }
        s1 = self.scene.apply_patch(
            {
                "schema_version": "0.1",
                **TEST_IDENTITY,
                "world_kind": "internal",
                "base_scene_version": 0,
                "frame_id": "env_origin",
                "observation_ids": ["obs-hw-1"],
                "builder_digest": "builder-s2-v1",
                "operations": [base_op],
            }
        )
        self.assertEqual(s1["calib_rev"], 1)

        s2 = self.scene.apply_patch(
            {
                "schema_version": "0.1",
                **TEST_IDENTITY,
                "world_kind": "internal",
                "base_scene_version": 1,
                "frame_id": "env_origin",
                "observation_ids": ["obs-hw-2"],
                "builder_digest": "builder-s2-v1",
                "operations": [base_op],
            }
        )
        self.assertEqual(s2["calib_rev"], 2)

        # 拒绝旧 calib_rev=1 在无 rebuild 情况下覆盖 calib_rev=2 场景
        with self.assertRaisesRegex(ValueError, "stale_calibration_correction_revision"):
            self.scene.apply_patch(
                {
                    "schema_version": "0.1",
                    **TEST_IDENTITY,
                    "world_kind": "internal",
                    "base_scene_version": 2,
                    "frame_id": "env_origin",
                    "observation_ids": ["obs-hw-1"],
                    "builder_digest": "builder-s2-v1",
                    "operations": [{"op": "mark_unknown", "source_kind": "observation_inference", "region": "back", "reason": "test"}],
                }
            )

        # 拒绝旧硬件曝光时间戳覆盖较新实体状态
        with self.assertRaisesRegex(ValueError, "late_observation_cannot_overwrite_entity"):
            self.scene.apply_patch(
                {
                    "schema_version": "0.1",
                    **TEST_IDENTITY,
                    "world_kind": "internal",
                    "base_scene_version": 2,
                    "frame_id": "env_origin",
                    "observation_ids": ["obs-hw-0"],
                    "builder_digest": "builder-s2-v1",
                    "operations": [base_op],
                }
            )

    def test_clock_domain_strictness_and_corrupted_calibration_rejection(self) -> None:
        """验证超窗硬件时间戳、离散步时钟不等、无修正外参篡改与非法修正契约均直接报错。"""
        # 1. hardware_monotonic_ns 超出默认 25ms 同步窗口直接报错
        with self.assertRaisesRegex(ValueError, "unsynchronized_observation"):
            self._import_hw_observation("obs-skew-bad", obs_ns=2_000_000_000, cam_skew_ns=40_000_000)

        # 2. adapter_control_step 保持严格相等校验
        obs_sim = self._import_hw_observation("obs-sim-ok", obs_ns=10, step_domain="adapter_control_step")
        bad_sim = deepcopy(obs_sim)
        bad_sim["observation_id"] = "obs-sim-mismatch"
        bad_sim["cameras"][0]["sim_step"] = 11
        self.scene.import_observation(bad_sim)
        res = self.geometry.execute(
            {**TEST_IDENTITY, "kind": "camera_pose", "observation_id": "obs-sim-mismatch", "camera_id": "cam_third_person"}
        )
        self.assertEqual(res["reason"], "unsynchronized_observation")

        # 3. 无 calibration_correction 时，sample 位姿偏移 15mm 仍严格触发 fixed_camera_calibration_inconsistent
        shifted_T = self.T_fixed_nom.copy()
        shifted_T[0, 3] += 0.015
        tampered_pose = {
            "sim_step": 10,
            "step_domain": "adapter_control_step",
            "cameras": {
                "cam_third_person": {
                    "T_world_camera": shifted_T.tolist(),
                    "frame_id": "obs-tampered-tp",
                    "source_kind": "fixed_calibration",
                }
            },
        }
        tampered_pose_ref = self.artifact_store.put_artifact(
            canonical(tampered_pose).encode(), "camera_pose_sample", "application/json"
        )
        tampered_obs = deepcopy(obs_sim)
        tampered_obs["observation_id"] = "obs-tampered"
        tampered_obs["cameras"] = [deepcopy(obs_sim["cameras"][0])]
        tampered_obs["cameras"][0]["frame_id"] = "obs-tampered-tp"
        tampered_obs["cameras"][0]["pose_sample_ref"] = tampered_pose_ref
        self.scene.import_observation(tampered_obs)
        res_tampered = self.geometry.execute(
            {**TEST_IDENTITY, "kind": "camera_pose", "observation_id": "obs-tampered", "camera_id": "cam_third_person"}
        )
        self.assertEqual(res_tampered["reason"], "fixed_camera_calibration_inconsistent")

        # 4. 非法标定契约（calib_rev=0 却非零偏移，或平移超界）直接抛 ValueError
        with self.assertRaisesRegex(ValueError, "calib_rev_zero_with_nonzero_delta"):
            CalibrationCorrection(calib_rev=0, delta_t_base_cam_m={"cam_third_person": (0.01, 0.0, 0.0)})
        with self.assertRaisesRegex(ValueError, "translation_out_of_bounds"):
            CalibrationCorrection(calib_rev=1, delta_t_base_cam_m={"cam_third_person": (0.45, 0.0, 0.0)})


if __name__ == "__main__":
    unittest.main()
