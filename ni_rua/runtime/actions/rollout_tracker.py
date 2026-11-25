"""运动期 4D 过程轨迹追踪、共动/滑脱客观验证与关键事件帧索引模块。

仅读取已公开授权的机器人本体状态（25Hz actual_trace 关节/TCP 位姿与实测夹爪孔径）、
已正式发布的观测帧序列及公开标定参数，零特权仿真状态读取，零哈希摘要暴露。
所有诊断指标均为纯客观运动学/视觉测量事实，绝不在底层隐式篡改参数或强停后续阶段。
"""
from __future__ import annotations

import json
import math
import threading
import urllib.error
import urllib.request
from typing import Any

import cv2
import numpy as np

from ...core.pose_math import matrix_to_quat_wxyz, quat_wxyz_to_matrix, wrap180
from ...robot.model import roboprobe_pose
from ...robot.motion.batch_scheduler import ArmLayout
from .inspection import embodiment_of

_DEFAULT_ARMS: tuple[str, ...] = ("left", "right")

# 中途滑脱客观运动学判定物理常量（单位：米 m）
# 1. 抓取闭合稳态须保持非空有效开度 >= 3.5mm（避免将 ~0mm 薄布料/薄片抓取误判为滑脱）
SLIP_MIN_SETTLED_APERTURE_M = 0.0035
# 2. 后续闭爪搬运阶段实测孔径突降塌陷至接近全闭 < 1.2mm
SLIP_COLLAPSED_APERTURE_M = 0.0012
# 3. 相对稳态抓取开度的塌陷降幅 >= 2.5mm
SLIP_MIN_APERTURE_DROP_M = 0.0025
# 紧闭夹持指令归一化上限（区分 jaw_goal <= 0.15 的紧闭抓取与 jaw_goal > 0.15 的半开预缩爪整形）
TIGHT_CLOSE_MAX_GOAL = 0.15
# 外部感知会话回收通知默认超时上界（<= 1.0s）
SESSION_CLOSE_TIMEOUT_S = 1.0


def _round_scalar(val: float | None, digits: int = 4) -> float | None:
    if val is None or not math.isfinite(float(val)):
        return None
    return round(float(val), digits)


def _round_vec(vec: Any, digits: int = 4) -> list[float] | None:
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    return [round(float(v), digits) for v in arr]


def external_perception_endpoint(runtime: Any) -> str | None:
    """当开启外部感知时返回配置的 vggt_endpoint，否则返回 None。"""
    cfg = getattr(runtime, "config", runtime)
    features = getattr(cfg, "features", cfg)
    if features is None:
        return None
    switches = (
        "external_perception",
        "object_table_observe",
        "scene3d_observe",
        "vggt_height",
        "object_table",
        "scene3d",
    )
    if not any(bool(getattr(features, name, False)) for name in switches):
        return None
    endpoint = getattr(features, "vggt_endpoint", None)
    if isinstance(endpoint, str) and endpoint.strip():
        return endpoint.strip()
    return None


def notify_session_close(
    endpoint: str | None,
    session_id: str | None,
    *,
    timeout_s: float = SESSION_CLOSE_TIMEOUT_S,
) -> None:
    """发起短超时（<=1.0s）非阻塞 POST /session_close 回收旧 Episode 显存会话，失败静默忽略。"""
    if not (isinstance(endpoint, str) and endpoint.strip() and isinstance(session_id, str) and session_id.strip()):
        return
    bounded_timeout = max(0.05, min(float(timeout_s), 1.0))
    url = endpoint.rstrip("/") + "/session_close"
    payload = json.dumps({"session_id": session_id.strip()}).encode("utf-8")

    def _send() -> None:
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=bounded_timeout) as resp:
                resp.read(256)
        except (urllib.error.URLError, OSError, ValueError):
            return

    worker = threading.Thread(target=_send, name="ni-rua-vggt-session-close", daemon=True)
    worker.start()
    # 服务端 /session_close 绕过 GPU 锁在 HTTP 线程即时返回（通常 <3ms），等待至多 0.15s 兼顾确定性送达与零阻塞
    worker.join(timeout=min(0.15, bounded_timeout))


def has_process_slip(receipt: dict | None, trajectory_summary: dict | None = None) -> bool:
    """检查动作回执或轨迹摘要的 process_trajectory 中是否检出中途滑脱（slip_detected == True）。"""
    candidates = []
    if isinstance(receipt, dict):
        pt = receipt.get("process_trajectory")
        if isinstance(pt, dict):
            candidates.append(pt)
    if isinstance(trajectory_summary, dict):
        c3d = trajectory_summary.get("compact_3d_trajectory")
        if isinstance(c3d, dict) and isinstance(c3d.get("process_trajectory"), dict):
            candidates.append(c3d["process_trajectory"])
        if isinstance(trajectory_summary.get("process_trajectory"), dict):
            candidates.append(trajectory_summary["process_trajectory"])
    for pt in candidates:
        co_motion = pt.get("co_motion")
        if isinstance(co_motion, dict):
            for arm_info in co_motion.values():
                if isinstance(arm_info, dict) and arm_info.get("slip_detected") is True:
                    return True
    return False


def _measured_aperture_from_arm(arm_state: dict | None) -> float | None:
    """从原始 robot_state 单臂字典或 trajectory_feedback 归一化单臂字典提取实测孔径（单位 m）。"""
    if not isinstance(arm_state, dict):
        return None
    jaw = arm_state.get("measured_jaw")
    if not isinstance(jaw, dict):
        raw_jaw = arm_state.get("gripper")
        if isinstance(raw_jaw, dict) and raw_jaw.get("position_source_kind") in (None, "measurement"):
            jaw = raw_jaw
    if not isinstance(jaw, dict):
        return None
    pos = jaw.get("position_m")
    if not isinstance(pos, (list, tuple, np.ndarray)) or len(pos) == 0:
        return None
    arr = np.asarray(pos, dtype=float).reshape(-1)
    if not np.isfinite(arr).all():
        return None
    if arr.size == 1:
        return max(0.0, float(arr[0])) * 2.0
    return float(sum(max(0.0, float(v)) for v in arr[:2]))


def _commanded_jaw_goal(arm_state: dict | None) -> float | None:
    """提取单步夹爪归一化目标开度 [0, 1]（0 为全闭，1 为全开）。"""
    if not isinstance(arm_state, dict):
        return None
    for key in ("jaw_goal_normalized", "jaw_command_normalized"):
        val = arm_state.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val)):
            return float(val)
    raw_jaw = arm_state.get("gripper")
    if isinstance(raw_jaw, dict):
        for key in ("goal_normalized", "command_normalized"):
            val = raw_jaw.get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val)):
                return float(val)
    return None


def _extract_tcp_and_angles(
    arm_state: dict | None,
    *,
    profile: Any = None,
) -> tuple[np.ndarray | None, tuple[float, float, float] | None]:
    """提取单步单臂 TCP 位置（优先指间夹持中心，单位 m）与外旋角 (pitch_deg, roll_deg, yaw_deg)。"""
    if not isinstance(arm_state, dict):
        return None, None

    default_prof = embodiment_of(None)
    prof = profile if profile is not None else default_prof
    grip_spec = getattr(prof, "gripper", None) or default_prof.gripper
    offset_m = (
        float(grip_spec.grasp_reference_offset_m)
        if grip_spec.grasp_reference == "tcp_offset" and grip_spec.grasp_reference_offset_m is not None
        else 0.0
    )
    raw_app_axis = getattr(grip_spec, "approach_axis_local", None)
    app_axis_local = (
        np.asarray(raw_app_axis, dtype=np.float64).reshape(3)
        if isinstance(raw_app_axis, (list, tuple, np.ndarray)) and len(raw_app_axis) == 3
        else np.array([1.0, 0.0, 0.0], dtype=np.float64)
    )
    tcp_link = getattr(prof, "tcp_link", None) or default_prof.tcp_link
    tcp_cfg_id = getattr(prof, "tcp_config_id", None) or default_prof.tcp_config_id

    # 1. 优先从公开 FK link6_pose 提取位置与旋转矩阵
    l6 = arm_state.get("link6_pose") or arm_state.get("tcp_pose")
    if isinstance(l6, dict) and "position_m" in l6:
        pos = np.asarray(l6["position_m"], dtype=float).reshape(-1)
        if pos.size == 3 and np.isfinite(pos).all():
            rot = l6.get("rotation_matrix")
            if rot is not None:
                rot_arr = np.asarray(rot, dtype=float)
                if rot_arr.shape == (3, 3) and np.isfinite(rot_arr).all():
                    tcp = pos + offset_m * (rot_arr @ app_axis_local)
                    w, x, y, z = (float(v) for v in matrix_to_quat_wxyz(rot_arr))
                    try:
                        vals = roboprobe_pose.pose_to_values([float(pos[0]), float(pos[1]), float(pos[2]), w, x, y, z])
                        angles = (float(vals["pitch_deg"]), float(vals["roll_deg"]), float(vals["yaw_deg"]))
                    except (ValueError, KeyError, ArithmeticError):
                        angles = None
                    return tcp, angles
            return pos, None

    # 2. 从实测 ee_pose / measured_ee_pose 提取
    ee = arm_state.get("measured_ee_pose") or arm_state.get("ee_pose")
    if isinstance(ee, dict) and "position_m" in ee:
        pos = np.asarray(ee["position_m"], dtype=float).reshape(-1)
        if pos.size == 3 and np.isfinite(pos).all():
            quat = ee.get("quaternion_xyzw")
            if isinstance(quat, (list, tuple, np.ndarray)) and len(quat) == 4:
                q_arr = np.asarray(quat, dtype=float).reshape(4)
                norm = float(np.linalg.norm(q_arr))
                if np.isfinite(q_arr).all() and norm > 1e-12:
                    qx, qy, qz, qw = (float(v / norm) for v in q_arr)
                    rot_mat = quat_wxyz_to_matrix(np.array([qw, qx, qy, qz], dtype=np.float64))
                    app_world = rot_mat @ app_axis_local
                    tcp = (
                        pos + offset_m * app_world
                        if offset_m > 0.0 and ee.get("tcp_config_id") in (None, tcp_cfg_id)
                        else pos
                    )
                    try:
                        vals = roboprobe_pose.pose_to_values([float(pos[0]), float(pos[1]), float(pos[2]), qw, qx, qy, qz])
                        angles = (float(vals["pitch_deg"]), float(vals["roll_deg"]), float(vals["yaw_deg"]))
                    except (ValueError, KeyError, ArithmeticError):
                        angles = None
                    return tcp, angles
            return pos, None

    links = arm_state.get("link_positions_m")
    if isinstance(links, dict) and tcp_link in links:
        pos = np.asarray(links[tcp_link], dtype=float).reshape(-1)
        if pos.size == 3 and np.isfinite(pos).all():
            return pos, None
    return None, None


def _select_nearest_observation_id(
    target_step: int,
    published_observations: list[tuple[int, str]],
    *,
    prefer_at_or_after: bool = False,
) -> str | None:
    """从已正式注册的过程/起止观测帧中选取步数最接近 target_step 的合法 observation_id。"""
    valid = [
        (int(step), obs_id)
        for step, obs_id in published_observations
        if isinstance(obs_id, str) and obs_id
    ]
    if not valid:
        return None
    if prefer_at_or_after:
        after = [(step - target_step, step, obs_id) for step, obs_id in valid if step >= target_step]
        if after:
            after.sort(key=lambda item: (item[0], item[1]))
            return after[0][2]
    ranked = sorted(valid, key=lambda item: (abs(item[0] - target_step), -item[0]))
    return ranked[0][1]


def _extract_planned_step_map(
    planned_samples: list[dict] | None,
    planned_trace: dict | None,
    *,
    arms_order: tuple[str, ...] | None = None,
    profile: Any = None,
) -> dict[int, dict[str, dict[str, Any]]]:
    """构建 native_step -> {arm: {stage_id, jaw_goal_normalized, phase}} 规划步上下文索引。"""
    if arms_order is None:
        arms_order = tuple(getattr(profile, "arms", None) or getattr(embodiment_of(None), "arms", None) or _DEFAULT_ARMS)
    step_map: dict[int, dict[str, dict[str, Any]]] = {}
    if planned_samples:
        for sample in planned_samples:
            step = sample.get("native_step")
            if not isinstance(step, int) or isinstance(step, bool):
                continue
            arms = sample.get("arms") or {}
            step_map[step] = {}
            for arm in arms_order:
                info = arms.get(arm) or {}
                step_map[step][arm] = dict(
                    stage_id=info.get("stage_id"),
                    jaw_goal_normalized=info.get("jaw_goal_normalized"),
                    phase=info.get("phase"),
                )
        return step_map

    if not isinstance(planned_trace, dict):
        return step_map
    if profile is not None and tuple(getattr(profile, "arms", ())) == tuple(arms_order):
        layout = ArmLayout.from_profile(profile)
    else:
        layout = ArmLayout.from_specs({arm: 6 for arm in arms_order}, with_jaw=True)
    segments = planned_trace.get("segments", [planned_trace])
    cursor = 0
    for seg in segments:
        if not isinstance(seg, dict) or "actions" not in seg:
            continue
        rows = seg.get("actions") or []
        contexts = seg.get("row_contexts") or []
        completed_rows = seg.get("completed_native_steps", len(rows))
        limit = min(len(rows), int(completed_rows) if isinstance(completed_rows, int) else len(rows))
        for idx in range(limit):
            step = cursor + idx + 1
            row = rows[idx]
            step_map[step] = {}
            for arm in arms_order:
                ctx = (contexts[idx].get(arm) or {}) if idx < len(contexts) and isinstance(contexts[idx], dict) else {}
                jaw_col = layout.jaw_index(arm) if arm in layout.slots else None
                goal = (
                    float(row[jaw_col])
                    if jaw_col is not None and isinstance(row, (list, tuple)) and len(row) >= layout.row_width
                    else None
                )
                step_map[step][arm] = dict(
                    stage_id=ctx.get("stage_id"),
                    jaw_goal_normalized=goal,
                    phase=ctx.get("phase"),
                )
        cursor += limit
    return step_map


def _stage_jaw_goals(stage_spec: dict | None) -> list[float]:
    """同时兼容原始阶段参数与编译后 BatchAction stage['action']['gripper_events'] 提取目标开度列表。"""
    if not isinstance(stage_spec, dict):
        return []
    goals: list[float] = []
    for key in ("goal_normalized", "gripper_goal_normalized"):
        val = stage_spec.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val)):
            goals.append(float(val))
    own_action = stage_spec.get("action")
    if isinstance(own_action, dict):
        for key in ("goal_normalized", "gripper_goal_normalized"):
            val = own_action.get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val)):
                goals.append(float(val))
        for ev in own_action.get("gripper_events") or []:
            if isinstance(ev, dict):
                val = ev.get("goal_normalized")
                if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val)):
                    goals.append(float(val))
    return goals


def _is_closing_stage_spec(stage_spec: dict | None) -> bool:
    """判断阶段声明是否为下达缩爪/闭爪目标（jaw_goal < 0.5）的阶段。"""
    goals = _stage_jaw_goals(stage_spec)
    return bool(goals) and goals[-1] < 0.5


def _is_tight_closing_stage_spec(stage_spec: dict | None) -> bool:
    """判断阶段声明是否为下达紧闭夹持目标（jaw_goal <= 0.15，排除半开预缩爪）的抓取阶段。"""
    goals = _stage_jaw_goals(stage_spec)
    return bool(goals) and goals[-1] <= TIGHT_CLOSE_MAX_GOAL


def _is_opening_stage_spec(stage_spec: dict | None) -> bool:
    """判断阶段声明是否为下达开爪释放目标的阶段。"""
    goals = _stage_jaw_goals(stage_spec)
    return bool(goals) and goals[-1] >= 0.5


def _lookup_camera_entry(doc: Any, camera_id: str, direct_key: str) -> dict | None:
    """从标准相机映射字典（{"cam_head": {...}}）、列表结构或单相机字典中提取目标相机条目。"""
    if not isinstance(doc, dict):
        return None
    cameras = doc.get("cameras")
    if isinstance(cameras, dict) and direct_key not in doc:
        entry = cameras.get(camera_id)
        return entry if isinstance(entry, dict) else None
    if isinstance(cameras, list):
        return next(
            (item for item in cameras if isinstance(item, dict) and item.get("camera_id") == camera_id),
            None,
        )
    return doc if direct_key in doc else None


def _fingertip_local_flow_px(
    store: Any,
    start_obs: dict | None,
    terminal_obs: dict | None,
    tcp_pos_m: np.ndarray | None,
    *,
    arm: str = "left",
    overview_camera: str | None = None,
    profile: Any = None,
) -> float | None:
    """在本地已落盘起止帧的指尖投影 48×48 邻域计算轻量光流幅值（单位 px，零外部网络调用）。"""
    if store is None or not isinstance(start_obs, dict) or not isinstance(terminal_obs, dict):
        return None
    start_cameras = start_obs.get("cameras")
    term_cameras = terminal_obs.get("cameras")
    if not isinstance(start_cameras, list) or not isinstance(term_cameras, list):
        return None
    from ...perception.external.sensor_frame_contract import LEGACY_HEAD_CAMERA_ID, list_fixed_camera_ids
    resolve_json = lambda ref: json.loads(store.resolve_artifact(ref, allowed_scopes=("actor",))[1])
    fixed_ids = list_fixed_camera_ids(start_obs, resolve_json=resolve_json)
    pref_cid = overview_camera or LEGACY_HEAD_CAMERA_ID
    primary_cid = pref_cid if pref_cid in fixed_ids else (fixed_ids[0] if fixed_ids else pref_cid)
    s_cam = next((c for c in start_cameras if isinstance(c, dict) and c.get("camera_id") == primary_cid), None)
    t_cam = next((c for c in term_cameras if isinstance(c, dict) and c.get("camera_id") == primary_cid), None)
    if (
        not isinstance(s_cam, dict)
        or not isinstance(t_cam, dict)
        or not isinstance(s_cam.get("calibration_ref"), str)
        or not isinstance(s_cam.get("pose_sample_ref"), str)
        or not isinstance(s_cam.get("rgb_ref"), str)
        or not isinstance(t_cam.get("rgb_ref"), str)
    ):
        return None

    try:
        if tcp_pos_m is None and isinstance(terminal_obs.get("robot_state_ref"), str):
            state_doc = json.loads(
                store.resolve_artifact(terminal_obs["robot_state_ref"], allowed_scopes=("actor",))[1]
            )
            if isinstance(state_doc, dict) and isinstance(state_doc.get("arms"), dict):
                tcp_pos_m = _extract_tcp_and_angles(state_doc["arms"].get(arm), profile=profile)[0]
        if tcp_pos_m is None:
            return None

        cal = json.loads(store.resolve_artifact(s_cam["calibration_ref"], allowed_scopes=("actor",))[1])
        pose_doc = json.loads(store.resolve_artifact(s_cam["pose_sample_ref"], allowed_scopes=("actor",))[1])
        cam_cal = _lookup_camera_entry(cal, primary_cid, "K")
        cam_pose = _lookup_camera_entry(pose_doc, primary_cid, "T_world_camera")
        if not isinstance(cam_cal, dict) or "K" not in cam_cal:
            return None
        t_wc_raw = (
            cam_pose.get("T_world_camera")
            if isinstance(cam_pose, dict) and "T_world_camera" in cam_pose
            else cam_cal.get("T_world_camera")
        )
        if t_wc_raw is None:
            return None

        k_mat = np.asarray(cam_cal["K"], dtype=np.float64).reshape(3, 3)
        t_cw = np.linalg.inv(np.asarray(t_wc_raw, dtype=np.float64).reshape(4, 4))
        p_cam = t_cw[:3, :3] @ tcp_pos_m + t_cw[:3, 3]
        if not np.isfinite(p_cam).all() or p_cam[2] <= 0.05:
            return None
        u = int(round(float(k_mat[0, 0] * p_cam[0] / p_cam[2] + k_mat[0, 2])))
        v = int(round(float(k_mat[1, 1] * p_cam[1] / p_cam[2] + k_mat[1, 2])))
        s_bytes = store.resolve_artifact(s_cam["rgb_ref"], allowed_scopes=("actor",))[1]
        t_bytes = store.resolve_artifact(t_cam["rgb_ref"], allowed_scopes=("actor",))[1]
        img0 = cv2.imdecode(np.frombuffer(s_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imdecode(np.frombuffer(t_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img0 is None or img1 is None or img0.shape != img1.shape:
            return None
        h, w = img0.shape
        half = 24
        x0, x1 = max(0, u - half), min(w, u + half)
        y0, y1 = max(0, v - half), min(h, v + half)
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        patch0 = img0[y0:y1, x0:x1]
        patch1 = img1[y0:y1, x0:x1]
        flow = cv2.calcOpticalFlowFarneback(patch0, patch1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        return _round_scalar(float(np.percentile(mag, 90)), 2)
    except (KeyError, ValueError, OSError, np.linalg.LinAlgError, cv2.error):
        return None


def build_process_trajectory(
    measured_samples: list[dict],
    *,
    action: dict | None = None,
    receipt: dict | None = None,
    planned_samples: list[dict] | None = None,
    planned_trace: dict | None = None,
    published_observations: list[tuple[int, str]] | None = None,
    peak_divergence_by_arm: dict[str, tuple[float | None, int | None]] | None = None,
    store: Any = None,
    start_observation: dict | None = None,
    terminal_observation: dict | None = None,
    profile: Any = None,
) -> dict | None:
    """从实测轨迹与正式观测索引纯函数构建运动期 4D 过程轨迹字典（process_trajectory）。

    输出结构：
    - co_motion: 按本批次曾闭合或持物的活跃手臂输出开度包络、搬运开度塌陷量与滑脱事实。
    - in_place_motion: 按闭爪啮合手臂积分偏航/俯仰/横滚角位移（deg）与垂直/水平行程（m）。
    - event_keyframes: 最多 4 个关键事件帧索引（start, grasp_settled, peak_error_or_slip, terminal）。
    """
    if not measured_samples:
        return None

    default_prof = embodiment_of(None)
    prof = profile if profile is not None else default_prof
    arms_order: tuple[str, ...] = (
        tuple(prof.arms)
        if getattr(prof, "arms", None)
        else tuple(getattr(default_prof, "arms", None) or _DEFAULT_ARMS)
    )
    grip_spec = getattr(prof, "gripper", None) or default_prof.gripper
    empty_close_m = float(grip_spec.empty_close_aperture_m or 0.0)
    full_open_m = float(grip_spec.full_open_aperture_m)
    slip_min_settled_m = empty_close_m + SLIP_MIN_SETTLED_APERTURE_M
    slip_collapsed_m = empty_close_m + SLIP_COLLAPSED_APERTURE_M
    engaged_max_ap_m = max(empty_close_m + 0.01, full_open_m - 0.013)
    overview_cam = getattr(prof, "overview_camera", None)

    action = action if isinstance(action, dict) else {}
    receipt = receipt if isinstance(receipt, dict) else {}
    obs_list = list(published_observations or [])
    if not obs_list:
        start_obs_id = receipt.get("execution_start_observation_id")
        if isinstance(start_obs_id, str) and start_obs_id:
            obs_list.append((0, start_obs_id))
        end_obs_id = receipt.get("result_observation_id")
        last_step_val = int(measured_samples[-1].get("native_step", len(measured_samples) - 1))
        if isinstance(end_obs_id, str) and end_obs_id:
            obs_list.append((last_step_val, end_obs_id))

    step_plan_map = _extract_planned_step_map(planned_samples, planned_trace, arms_order=arms_order, profile=prof)
    action_stages = action.get("stages") if isinstance(action.get("stages"), list) else []
    stage_spec_by_id = {
        s["id"]: s for s in action_stages if isinstance(s, dict) and isinstance(s.get("id"), str)
    }
    stage_results = receipt.get("stage_results") if isinstance(receipt.get("stage_results"), dict) else {}
    for sid, sr in stage_results.items():
        if sid not in stage_spec_by_id and isinstance(sr, dict):
            stage_spec_by_id[sid] = sr

    explicit_arms = {
        s.get("arm")
        for s in action_stages
        if isinstance(s, dict) and s.get("arm") in arms_order
    }
    if not explicit_arms and action.get("arm") in arms_order:
        explicit_arms.add(action["arm"])

    steps = [int(s.get("native_step", idx)) for idx, s in enumerate(measured_samples)]
    last_step = steps[-1]

    co_motion: dict[str, dict[str, Any]] = {}
    in_place_motion: dict[str, dict[str, Any]] = {}
    arm_Key_events: dict[str, dict[str, Any]] = {}
    per_arm_series: dict[str, dict[str, list]] = {}

    for arm in arms_order:
        ap_list: list[float | None] = []
        goal_list: list[float | None] = []
        stage_id_list: list[str | None] = []
        tcp_list: list[np.ndarray | None] = []
        ang_list: list[tuple[float, float, float] | None] = []
        q_list: list[np.ndarray | None] = []

        for idx, sample in enumerate(measured_samples):
            step = steps[idx]
            arm_state = (sample.get("arms") or {}).get(arm) or {}
            plan_ctx = (step_plan_map.get(step) or {}).get(arm) or {}
            ap = _measured_aperture_from_arm(arm_state)
            goal = plan_ctx.get("jaw_goal_normalized")
            if goal is None:
                goal = _commanded_jaw_goal(arm_state)
            sid = plan_ctx.get("stage_id") or arm_state.get("stage_id")
            tcp, angles = _extract_tcp_and_angles(arm_state, profile=prof)
            q_raw = arm_state.get("q_rad")
            q_arr = (
                np.asarray(q_raw, dtype=float).reshape(-1)
                if isinstance(q_raw, (list, tuple, np.ndarray)) and len(q_raw) >= 6
                else None
            )

            ap_list.append(ap)
            goal_list.append(goal)
            stage_id_list.append(sid)
            tcp_list.append(tcp)
            ang_list.append(angles)
            q_list.append(q_arr)

        per_arm_series[arm] = dict(
            ap=ap_list, goal=goal_list, stage_id=stage_id_list, tcp=tcp_list, angles=ang_list
        )

        # 判断该手臂在本批次是否为活跃手臂（显式声明了阶段，或实际发生了关节/位姿/夹爪运动）
        valid_aps = [a for a in ap_list if a is not None]
        valid_tcps = [p for p in tcp_list if p is not None]
        tcp_span = (
            max(float(np.linalg.norm(p - valid_tcps[0])) for p in valid_tcps[1:])
            if len(valid_tcps) >= 2
            else 0.0
        )
        valid_qs = [q for q in q_list if q is not None and np.isfinite(q).all()]
        q_span = (
            max(float(np.max(np.abs(q - valid_qs[0]))) for q in valid_qs[1:])
            if len(valid_qs) >= 2
            else 0.0
        )
        ap_span = (max(valid_aps) - min(valid_aps)) if len(valid_aps) >= 2 else 0.0
        arm_active = (arm in explicit_arms) or (
            not explicit_arms and (tcp_span > 0.002 or q_span > 0.01 or ap_span > 0.0015)
        )
        if not arm_active or not valid_aps:
            continue

        arm_stages = [s for s in action_stages if isinstance(s, dict) and s.get("arm") == arm]
        has_closing_cmd = any(_is_closing_stage_spec(s) for s in arm_stages) or any(
            g is not None and g < 0.5 for g in goal_list[1:]
        )
        started_holding = (ap_list[0] is not None and ap_list[0] < engaged_max_ap_m)
        closed_during_batch = min(valid_aps) < engaged_max_ap_m and (has_closing_cmd or started_holding or ap_span > 0.002)
        if not closed_during_batch:
            continue

        # 定位紧闭抓取稳态完成步索引 grasp_settled_idx（严格排除 jaw_goal > 0.15 的半开预缩爪整形阶段）
        settled_idx: int | None = None
        tight_closing_stage_ids = [
            s["id"] for s in arm_stages if isinstance(s.get("id"), str) and _is_tight_closing_stage_spec(s)
        ]
        half_closing_stage_ids = {
            s["id"]
            for s in arm_stages
            if isinstance(s.get("id"), str) and _is_closing_stage_spec(s) and not _is_tight_closing_stage_spec(s)
        }
        if tight_closing_stage_ids:
            first_close_sid = tight_closing_stage_ids[0]
            matching_indices = [i for i, sid in enumerate(stage_id_list) if sid == first_close_sid and ap_list[i] is not None]
            if matching_indices:
                settled_idx = matching_indices[-1]

        if settled_idx is None:
            # 若批次起始即已紧闭持物（ap_0 >= slip_min_settled_m 且非半开/开爪指令）且第一步非主动快速闭合瞬态，则起始步即为稳态持物起点
            if (
                ap_list[0] is not None
                and slip_min_settled_m <= ap_list[0] < engaged_max_ap_m
                and (goal_list[0] is None or goal_list[0] <= TIGHT_CLOSE_MAX_GOAL)
                and (len(ap_list) == 1 or ap_list[1] is None or abs(ap_list[0] - ap_list[1]) <= 0.002)
                and not tight_closing_stage_ids
            ):
                settled_idx = 0
            else:
                # 仅在紧闭目标步（goal is None 或 goal <= 0.15，且非半开预整形阶段）中寻找首次闭爪到达稳态平台或转入平移搬运的分界步
                for i in range(1, len(measured_samples)):
                    curr_ap = ap_list[i]
                    prev_ap = ap_list[i - 1]
                    curr_goal = goal_list[i]
                    curr_sid = stage_id_list[i]
                    if curr_ap is None or curr_ap >= engaged_max_ap_m:
                        continue
                    if (curr_goal is not None and curr_goal > TIGHT_CLOSE_MAX_GOAL) or (curr_sid in half_closing_stage_ids):
                        continue
                    if prev_ap is not None and (prev_ap - curr_ap) > 0.002:
                        # 仍在紧闭驱动进程中，检查下一步是否开始平稳或手臂开始搬运位移
                        if i + 1 < len(measured_samples):
                            next_ap = ap_list[i + 1]
                            tcp_curr, tcp_next = tcp_list[i], tcp_list[i + 1]
                            translating = (
                                tcp_curr is not None
                                and tcp_next is not None
                                and float(np.linalg.norm(tcp_next - tcp_curr)) > 0.002
                            )
                            plateau = next_ap is not None and abs(next_ap - curr_ap) <= 0.001
                            if plateau or translating:
                                settled_idx = i
                                break
                        else:
                            settled_idx = i
                            break
                    elif prev_ap is not None and abs(curr_ap - prev_ap) <= 0.001:
                        prev_goal = goal_list[i - 1]
                        prev_sid = stage_id_list[i - 1]
                        prev_tight = (
                            ap_list[i - 1] is not None
                            and ap_list[i - 1] < engaged_max_ap_m
                            and (prev_goal is None or prev_goal <= TIGHT_CLOSE_MAX_GOAL)
                            and (prev_sid not in half_closing_stage_ids)
                        )
                        settled_idx = i - 1 if prev_tight else i
                        break

        # 统计紧闭稳态后的持物/抬升/搬运区间孔径塌陷与中途滑脱（仅监测非闭爪驱动且保持紧闭 goal<=0.15 的步）
        transport_drop = 0.0
        slip_detected = False
        slip_onset_step: int | None = None
        slip_obs_id: str | None = None

        if settled_idx is not None and ap_list[settled_idx] is not None:
            settled_ap = float(ap_list[settled_idx])
            settled_sid = stage_id_list[settled_idx]
            active_reclose_untagged = False
            for j in range(settled_idx + 1, len(measured_samples)):
                sid_j = stage_id_list[j]
                spec_j = stage_spec_by_id.get(sid_j or "")
                goal_j = goal_list[j]
                goal_prev = goal_list[j - 1]
                ap_j = ap_list[j]
                ap_prev = ap_list[j - 1]
                # 一旦进入主动张爪释放阶段，立即结束闭爪搬运滑脱监测窗口
                if _is_opening_stage_spec(spec_j):
                    break
                if goal_j is not None and goal_j >= 0.5:
                    break
                if goal_j is None and sid_j is None and ap_j is not None and ap_prev is not None and (ap_j - ap_prev) > 0.006:
                    break
                # 若同一批次后续存在再次紧闭抓取阶段，该阶段内属于主动执行器闭爪驱动而非滑脱，结束后更新稳态开度
                if sid_j is not None and sid_j != settled_sid and _is_tight_closing_stage_spec(spec_j):
                    if ap_j is not None:
                        settled_ap = float(ap_j)
                    continue
                if sid_j is None and goal_j is not None and goal_j <= TIGHT_CLOSE_MAX_GOAL:
                    if goal_prev is not None and goal_prev > TIGHT_CLOSE_MAX_GOAL:
                        active_reclose_untagged = True
                    if active_reclose_untagged:
                        if ap_j is not None:
                            settled_ap = float(ap_j)
                            if ap_prev is not None and abs(ap_j - ap_prev) <= 0.001 and not (goal_prev is not None and goal_prev > TIGHT_CLOSE_MAX_GOAL):
                                active_reclose_untagged = False
                        continue
                else:
                    active_reclose_untagged = False
                # 半开预整形指令（0.15 < goal_j < 0.5）并非紧闭持物搬运，不计入滑脱监测
                if (goal_j is not None and goal_j > TIGHT_CLOSE_MAX_GOAL) or (sid_j in half_closing_stage_ids):
                    continue
                if ap_j is None:
                    continue
                drop = max(0.0, settled_ap - float(ap_j))
                if drop > transport_drop:
                    transport_drop = drop
                if (
                    not slip_detected
                    and settled_ap >= slip_min_settled_m
                    and float(ap_j) < slip_collapsed_m
                    and drop >= SLIP_MIN_APERTURE_DROP_M
                ):
                    slip_detected = True
                    slip_onset_step = steps[j]
                    slip_obs_id = _select_nearest_observation_id(
                        slip_onset_step, obs_list, prefer_at_or_after=True
                    )

        co_entry: dict[str, Any] = dict(
            aperture_start_m=_round_scalar(ap_list[0] if ap_list[0] is not None else valid_aps[0]),
            aperture_min_m=_round_scalar(min(valid_aps)),
            aperture_end_m=_round_scalar(ap_list[-1] if ap_list[-1] is not None else valid_aps[-1]),
            transport_aperture_drop_m=_round_scalar(transport_drop) or 0.0,
            slip_detected=bool(slip_detected),
            slip_onset_step=slip_onset_step,
            slip_observation_id=slip_obs_id,
        )
        flow_px = _fingertip_local_flow_px(
            store,
            start_observation,
            terminal_observation,
            tcp_list[-1] if tcp_list else None,
            arm=arm,
            overview_camera=overview_cam,
            profile=prof,
        )
        if flow_px is not None:
            co_entry["fingertip_flow_px"] = flow_px
        co_motion[arm] = co_entry

        # 积分闭爪啮合期间的原地旋拧角位移（deg）与垂直/水平行程（m）
        yaw_sum = 0.0
        pitch_sum = 0.0
        roll_sum = 0.0
        dz_sum = 0.0
        xy_travel = 0.0
        engaged_steps = 0
        for j in range(1, len(measured_samples)):
            ap_prev, ap_curr = ap_list[j - 1], ap_list[j]
            if ap_prev is None or ap_curr is None:
                continue
            if ap_prev >= engaged_max_ap_m or ap_curr >= engaged_max_ap_m:
                continue
            goal_curr = goal_list[j]
            sid_curr = stage_id_list[j]
            if _is_opening_stage_spec(stage_spec_by_id.get(sid_curr or "")) or (goal_curr is not None and goal_curr >= 0.5):
                continue
            # 排除空行程快速合拢瞬态，仅在夹爪闭合到位持物/啮合区间积分工具端运动
            if (ap_prev - ap_curr) > 0.0025 and settled_idx is not None and j <= settled_idx:
                continue

            engaged_steps += max(1, steps[j] - steps[j - 1])
            ang_prev, ang_curr = ang_list[j - 1], ang_list[j]
            if ang_prev is not None and ang_curr is not None:
                pitch_sum += wrap180(float(ang_curr[0]) - float(ang_prev[0]))
                roll_sum += wrap180(float(ang_curr[1]) - float(ang_prev[1]))
                yaw_sum += wrap180(float(ang_curr[2]) - float(ang_prev[2]))
            elif q_list[j - 1] is not None and q_list[j] is not None:
                # 缺失笛卡尔旋转时，回退使用腕部末端关节角增量（joint6 绕工具轴旋转）
                dq5 = float(q_list[j][5] - q_list[j - 1][5])
                yaw_sum += -math.degrees(dq5)

            tcp_prev, tcp_curr = tcp_list[j - 1], tcp_list[j]
            if tcp_prev is not None and tcp_curr is not None:
                dz_sum += float(tcp_curr[2] - tcp_prev[2])
                xy_travel += float(np.linalg.norm(tcp_curr[:2] - tcp_prev[:2]))

        in_place_motion[arm] = dict(
            engaged_wrist_yaw_delta_deg=_round_scalar(yaw_sum, 2) or 0.0,
            engaged_pitch_delta_deg=_round_scalar(pitch_sum, 2) or 0.0,
            engaged_roll_delta_deg=_round_scalar(roll_sum, 2) or 0.0,
            engaged_dz_m=_round_scalar(dz_sum, 4) or 0.0,
            engaged_xy_travel_m=_round_scalar(xy_travel, 4) or 0.0,
            engaged_steps=int(engaged_steps),
        )
        arm_Key_events[arm] = dict(
            settled_idx=settled_idx,
            slip_onset_step=slip_onset_step,
            slip_obs_id=slip_obs_id,
        )

    # 选择主参考手臂以生成最多 4 个关键事件帧索引（start, grasp_settled, peak_error_or_slip, terminal）
    default_arm = arms_order[0] if arms_order else _DEFAULT_ARMS[0]
    primary_arm = next(
        (a for a in arms_order if a in co_motion),
        next((a for a in arms_order if a in explicit_arms), default_arm),
    )

    def _make_keyframe(
        event_name: str,
        sample_idx: int,
        arm_name: str,
        override_obs_id: str | None = None,
    ) -> dict[str, Any]:
        series = per_arm_series[arm_name]
        exec_step = steps[sample_idx]
        obs_id = override_obs_id or _select_nearest_observation_id(
            exec_step,
            obs_list,
            prefer_at_or_after=(event_name in ("peak_error_or_slip", "terminal")),
        )
        return dict(
            event=event_name,
            stage_id=series["stage_id"][sample_idx],
            executed_step=int(exec_step),
            observation_id=obs_id,
            tcp_position_m=_round_vec(series["tcp"][sample_idx]),
            aperture_m=_round_scalar(series["ap"][sample_idx]),
        )

    step_to_idx = {st: idx for idx, st in enumerate(steps)}
    keyframes: list[dict[str, Any]] = [_make_keyframe("start", 0, primary_arm)]

    # 2. grasp_settled 关键帧
    settled_candidates = [
        (arm_Key_events[a]["settled_idx"], a)
        for a in arms_order
        if a in arm_Key_events and arm_Key_events[a]["settled_idx"] is not None and arm_Key_events[a]["settled_idx"] > 0
    ]
    if settled_candidates:
        settled_candidates.sort(key=lambda item: item[0])
        s_idx, s_arm = settled_candidates[0]
        keyframes.append(_make_keyframe("grasp_settled", s_idx, s_arm))

    # 3. peak_error_or_slip 关键帧（滑脱优先，否则取最大跟踪偏差步）
    slip_candidates = [
        (arm_Key_events[a]["slip_onset_step"], a, arm_Key_events[a]["slip_obs_id"])
        for a in arms_order
        if a in arm_Key_events and arm_Key_events[a]["slip_onset_step"] is not None
    ]
    if slip_candidates:
        slip_candidates.sort(key=lambda item: item[0])
        slip_step, slip_arm, slip_obs = slip_candidates[0]
        if slip_step in step_to_idx:
            keyframes.append(
                _make_keyframe("peak_error_or_slip", step_to_idx[slip_step], slip_arm, override_obs_id=slip_obs)
            )
    elif peak_divergence_by_arm:
        best_div = 0.0
        best_step: int | None = None
        best_arm = primary_arm
        for arm_name, (div_val, div_step) in peak_divergence_by_arm.items():
            if div_val is not None and div_step is not None and float(div_val) > best_div and div_step in step_to_idx:
                best_div = float(div_val)
                best_step = int(div_step)
                best_arm = arm_name
        if best_step is not None and best_div > 0.001 and best_step not in (0, last_step):
            keyframes.append(_make_keyframe("peak_error_or_slip", step_to_idx[best_step], best_arm))

    # 4. terminal 关键帧（确保总条目数 <= 4）
    if len(measured_samples) > 1 or not keyframes:
        keyframes.append(_make_keyframe("terminal", len(measured_samples) - 1, primary_arm))

    return dict(
        co_motion=co_motion,
        in_place_motion=in_place_motion,
        event_keyframes=keyframes[:4],
    )


class ProcessRolloutTracker:
    """在单次物理动作执行期间轻量记录过程观测帧，并在终态产出 4D 过程轨迹诊断摘要。"""

    def __init__(self, action: dict | None = None, *, runtime: Any = None, store: Any = None, profile: Any = None) -> None:
        self.action = action if isinstance(action, dict) else {}
        self.runtime = runtime
        self.profile = profile if profile is not None else embodiment_of(runtime)
        self.store = store if store is not None else (getattr(runtime, "store", None) if runtime is not None else None)
        self.progress_observations: list[dict[str, Any]] = []
        if self.runtime is not None:
            from ...perception.external.wrist_progress_cache import ensure_wrist_progress_cache, make_artifact_json_resolver
            cache = ensure_wrist_progress_cache(self.runtime)
            cache.seed_pre_action_state(
                getattr(self.runtime, "latest", None),
                resolve_json=make_artifact_json_resolver(self.runtime, self.store),
            )

    def record_progress_observation(self, progress_value: dict, bundle: dict) -> None:
        """记录已通过 publish_observation 正式注册的过程观测帧摘要，并同步更新准静态腕部视角缓存。"""
        obs_id = bundle.get("observation_id") if isinstance(bundle, dict) else None
        if not isinstance(obs_id, str) or not obs_id:
            return
        raw_steps = progress_value.get("executed_steps", 0) if isinstance(progress_value, dict) else 0
        exec_steps = int(raw_steps) if isinstance(raw_steps, (int, float)) and not isinstance(raw_steps, bool) else 0
        self.progress_observations.append(
            dict(
                observation_id=obs_id,
                executed_steps=max(0, exec_steps),
                sim_step=bundle.get("sim_step", progress_value.get("sim_step") if isinstance(progress_value, dict) else None),
                phase=(progress_value.get("phase") or "progress") if isinstance(progress_value, dict) else "progress",
                stage_id=progress_value.get("stage_id") if isinstance(progress_value, dict) else None,
            )
        )
        if self.runtime is not None:
            from ...perception.external.wrist_progress_cache import ensure_wrist_progress_cache, make_artifact_json_resolver
            ensure_wrist_progress_cache(self.runtime).ingest_progress_bundle(
                bundle,
                action_id=bundle.get("last_action_id"),
                resolve_json=make_artifact_json_resolver(self.runtime, self.store),
            )

    def finalize(
        self,
        *,
        before_state: dict | None,
        actual_trace: list[dict] | None,
        planned_trace: dict | None,
        receipt: dict | None,
        start_observation: dict | None = None,
        terminal_observation: dict | None = None,
        store: Any = None,
    ) -> dict | None:
        """结合起止状态、25Hz actual_trace 与已发布观测帧序列产出客观 process_trajectory。"""
        if not isinstance(before_state, dict) or not isinstance(before_state.get("arms"), dict):
            return None
        records = [dict(sim_step=before_state.get("sim_step", 0), state=before_state)] + list(actual_trace or [])
        measured_samples: list[dict] = []
        last_sim_step = None
        for rec in records:
            if not isinstance(rec, dict) or not isinstance(rec.get("state"), dict):
                continue
            sim_step = rec.get("sim_step", rec["state"].get("sim_step"))
            if last_sim_step is not None and sim_step is not None and sim_step == last_sim_step:
                continue
            last_sim_step = sim_step
            measured_samples.append(
                dict(
                    native_step=len(measured_samples),
                    sim_step=sim_step,
                    arms=rec["state"].get("arms") or {},
                )
            )
        if not measured_samples:
            return None

        published_obs: list[tuple[int, str]] = []
        if isinstance(start_observation, dict) and isinstance(start_observation.get("observation_id"), str):
            published_obs.append((0, start_observation["observation_id"]))
        elif isinstance(receipt, dict) and isinstance(receipt.get("execution_start_observation_id"), str):
            published_obs.append((0, receipt["execution_start_observation_id"]))
        for item in self.progress_observations:
            published_obs.append((int(item["executed_steps"]), str(item["observation_id"])))
        last_step = measured_samples[-1]["native_step"]
        if isinstance(terminal_observation, dict) and isinstance(terminal_observation.get("observation_id"), str):
            published_obs.append((last_step, terminal_observation["observation_id"]))
        elif isinstance(receipt, dict) and isinstance(receipt.get("result_observation_id"), str):
            published_obs.append((last_step, receipt["result_observation_id"]))

        return build_process_trajectory(
            measured_samples,
            action=self.action,
            receipt=receipt,
            planned_trace=planned_trace,
            published_observations=published_obs,
            store=store if store is not None else self.store,
            start_observation=start_observation,
            terminal_observation=terminal_observation,
            profile=self.profile,
        )
