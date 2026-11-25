"""动作执行期 5Hz 随动腕部多视角准静态门控与有界缓存。

在物理动作执行期间（`Harness.execute` 的 5Hz `progress` 回调），对已落盘的过程观测包
按实测关节角速度执行准静态防运动模糊门控（默认 `<= 0.8 rad/s`），剔除高速挥臂快门拖影帧，
仅保留清晰的移动腕部新视角帧供后续 `scene3d` 多视角点云融合按需读取。
"""
from __future__ import annotations

import base64
import collections
import json
import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...core.episode_state import StateSlot, slot_peek, slot_set
from .sensor_frame_contract import (
    StreamBudgetConfig,
    extract_camera_entries,
    list_wrist_camera_ids,
    resolve_camera_arm,
)

WRIST_PROGRESS_CACHE_SLOT: StateSlot = StateSlot("wrist_progress_cache", lambda: None, scope="episode")

# 同一相机在原地悬停时的去重阈值（平移 < 2cm 且光轴转角 < 5 度视为重复机位，原地刷新而非挤占滑窗）
IN_PLACE_DEDUP_TRANSLATION_M = 0.02
IN_PLACE_DEDUP_ANGLE_DEG = 5.0


@dataclass(frozen=True)
class CachedWristFrame:
    """经准静态门控筛选后的单帧腕部多视角几何与图像句柄快照。

    字段说明：
    - `episode_id`: 所属 Episode 标识符，跨局自动隔离。
    - `env_generation`: 环境重置代数（整数，`reset` 后自动失效旧缓存）。
    - `action_id`: 采集该过程帧时正在执行的动作 ID。
    - `observation_id`: 对应的历史过程观测 ID（`make_current=False`）。
    - `camera_id`: 腕部相机标识符（如 `'cam_left_wrist'`、`'cam_right_wrist'`）。
    - `arm`: 所属机械臂侧标识（`'left'` 或 `'right'`）。
    - `sim_step`: 采集时刻的控制步计数（整数或 `None`）。
    - `rgb_ref`: `Store` 中的不可变图像引用句柄（优先使用）。
    - `image_b64`: 内存内联图像 Base64（仅无 `rgb_ref` 的纯内存测试结构使用）。
    - `K`: 3×3 无畸变相机内参矩阵，单位 `px`。
    - `T_world_camera`: 4×4 世界/基座参考系下的相机位姿矩阵，平移单位 `m`。
    - `joint_speed_rad_s`: 采样时刻该手臂的最大关节角速度绝对值，单位 `rad/s`。
    - `gripper_apertures_m`: 采样时刻双臂实测夹爪开度字典（取 `position_m[0]`，单位 `m`），用于抓取/释放因果门控。
    """

    episode_id: str
    env_generation: int
    action_id: str | None
    observation_id: str
    camera_id: str
    arm: str | None
    sim_step: int | None
    rgb_ref: str | None
    image_b64: str | None
    K: tuple[tuple[float, ...], ...]
    T_world_camera: tuple[tuple[float, ...], ...]
    joint_speed_rad_s: float
    gripper_apertures_m: dict[str, float] = field(default_factory=dict, hash=False)


def _observation_time_s(observation: Mapping[str, Any]) -> float | None:
    """提取观测帧的物理时钟秒数（单位 `s`）：优先取 `sim_time_s`，否则由 `sim_step * control_dt_s` 推导。"""
    sim_t = observation.get("sim_time_s")
    if isinstance(sim_t, (int, float)) and not isinstance(sim_t, bool) and math.isfinite(float(sim_t)):
        return float(sim_t)
    step = observation.get("sim_step")
    dt = observation.get("control_dt_s")
    if dt is None:
        for cam in observation.get("cameras") or []:
            if isinstance(cam, Mapping) and cam.get("control_dt_s") is not None:
                dt = cam.get("control_dt_s")
                break
    if (
        isinstance(step, int)
        and not isinstance(step, bool)
        and isinstance(dt, (int, float))
        and not isinstance(dt, bool)
        and float(dt) > 0.0
    ):
        return float(step) * float(dt)
    return None


def measure_arm_joint_speed_rad_s(
    arm_state: Mapping[str, Any] | None,
    *,
    prev_q_rad: np.ndarray | None = None,
    dt_s: float | None = None,
) -> float | None:
    """计算单臂最大关节角速度绝对值（单位 `rad/s`）。

    优先读取实测关节角速度 `qdot_rad_s`；若当前观测画像不含 `qdot_rad_s`，
    且存在同局前一采样步 `(prev_q_rad, dt_s)`（`dt_s > 1e-6 s`），则计算有限差分角速度。
    """
    if not isinstance(arm_state, Mapping):
        return None
    qdot = arm_state.get("qdot_rad_s")
    if isinstance(qdot, (list, tuple, np.ndarray)) and len(qdot) > 0:
        arr = np.asarray(qdot, dtype=np.float64).reshape(-1)
        if np.isfinite(arr).all():
            return float(np.max(np.abs(arr)))

    q_raw = arm_state.get("q_rad")
    if (
        isinstance(q_raw, (list, tuple, np.ndarray))
        and len(q_raw) > 0
        and prev_q_rad is not None
        and dt_s is not None
        and float(dt_s) > 1e-6
    ):
        q_curr = np.asarray(q_raw, dtype=np.float64).reshape(-1)
        if q_curr.shape == prev_q_rad.shape and np.isfinite(q_curr).all():
            return float(np.max(np.abs(q_curr - prev_q_rad)) / float(dt_s))
    return None


def _se3_pose_delta(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    """计算两个 4×4 相机位姿之间的光心欧氏距离（`m`）与旋转夹角（`deg`）。"""
    trans_m = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))
    rot_rel = T_a[:3, :3].T @ T_b[:3, :3]
    cos_val = float(np.clip((np.trace(rot_rel) - 1.0) * 0.5, -1.0, 1.0))
    angle_deg = math.degrees(math.acos(cos_val))
    return trans_m, angle_deg


def _extract_valid_pinhole_and_pose(
    cal_entry: Mapping[str, Any] | None,
    pose_entry: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """校验并提取无畸变 3×3 内参 `K` 与合法 4×4 刚体变换 `T_world_camera`。"""
    if not isinstance(cal_entry, Mapping) or cal_entry.get("K") is None:
        return None
    dist = cal_entry.get("distortion")
    d_coeffs = (
        dist.get("coefficients", [])
        if isinstance(dist, Mapping)
        else (dist if isinstance(dist, (list, tuple, np.ndarray)) else [])
    )
    if len(d_coeffs) > 0 and np.any(np.abs(np.asarray(d_coeffs, dtype=np.float64)) > 1e-12):
        return None

    K = np.asarray(cal_entry["K"], dtype=np.float64).reshape(3, 3)
    if not np.isfinite(K).all() or K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
        return None

    t_raw = pose_entry.get("T_world_camera") if isinstance(pose_entry, Mapping) else None
    if t_raw is None:
        t_raw = cal_entry.get("T_world_camera")
    if t_raw is None:
        return None

    T = np.asarray(t_raw, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(T).all() or not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-4):
        return None
    if abs(float(np.linalg.det(T[:3, :3])) - 1.0) > 1e-2:
        return None
    return K, T


def _extract_gripper_apertures_m(arms_doc: Mapping[str, Any] | None) -> dict[str, float]:
    """提取各机械臂实测夹爪位置字典（取 `arm_doc["gripper"]["position_m"][0]`，单位 `m`）。"""
    if not isinstance(arms_doc, Mapping):
        return {}
    apertures: dict[str, float] = {}
    for arm_name, arm_doc in arms_doc.items():
        if not isinstance(arm_doc, Mapping):
            continue
        jaw = arm_doc.get("gripper")
        if not isinstance(jaw, Mapping):
            continue
        pos = jaw.get("position_m")
        if isinstance(pos, (list, tuple, np.ndarray)) and len(pos) > 0:
            v0 = float(pos[0])
            if math.isfinite(v0):
                apertures[str(arm_name)] = v0
    return apertures


def _has_gripper_aperture_jump(
    a_map: Mapping[str, float],
    b_map: Mapping[str, float],
    threshold_m: float,
) -> bool:
    """判断两份实测夹爪开度字典间是否存在任一公共手臂发生超过 `threshold_m`（单位 `m`）的开闭跃迁。"""
    for arm_name, val_a in a_map.items():
        if arm_name in b_map and abs(float(val_a) - float(b_map[arm_name])) > threshold_m:
            return True
    return False


class WristProgressCache:
    """线程安全的 Episode 级 5Hz 动作期腕部准静态多视角缓存。"""

    def __init__(self, budget: StreamBudgetConfig | None = None) -> None:
        self.budget = budget if budget is not None else StreamBudgetConfig()
        self._lock = threading.RLock()
        self._episode_key: tuple[str, int] | None = None
        self._frames: collections.deque[CachedWristFrame] = collections.deque(
            maxlen=self.budget.max_wrist_progress_frames
        )
        self._last_arm_q: dict[str, tuple[np.ndarray, float]] = {}
        self._last_gripper_apertures_m: dict[str, float] = {}
        self.accepted_frames = 0
        self.rejected_motion_blur = 0
        self.skipped_missing_velocity = 0
        self.invalidated_by_gripper_jump = 0

    def _sync_episode_unlocked(self, observation: Mapping[str, Any]) -> tuple[str, int]:
        ep_id = str(observation.get("episode_id") or "")
        gen_raw = observation.get("env_generation", 1)
        gen = int(gen_raw) if isinstance(gen_raw, int) and not isinstance(gen_raw, bool) else 1
        key = (ep_id, gen)
        if self._episode_key != key:
            self._episode_key = key
            self._frames.clear()
            self._last_arm_q.clear()
            self._last_gripper_apertures_m.clear()
        return key

    def _resolve_robot_state(
        self,
        observation: Mapping[str, Any],
        resolve_json: Callable[[str], Any] | None,
    ) -> Mapping[str, Any] | None:
        state = observation.get("robot_state")
        if isinstance(state, Mapping):
            return state
        ref = observation.get("robot_state_ref")
        if isinstance(ref, str) and ref and callable(resolve_json):
            loaded = resolve_json(ref)
            if isinstance(loaded, Mapping):
                return loaded
        return None

    def seed_pre_action_state(
        self,
        observation: Mapping[str, Any] | None,
        *,
        resolve_json: Callable[[str], Any] | None = None,
    ) -> None:
        """在动作启动前同步 Episode 代数并记录起始关节角与实测夹爪开度，供有限差分速度与因果跃迁门控使用。"""
        if not isinstance(observation, Mapping):
            return
        with self._lock:
            self._sync_episode_unlocked(observation)
        state = self._resolve_robot_state(observation, resolve_json)
        if not isinstance(state, Mapping):
            return
        arms = state.get("arms")
        if not isinstance(arms, Mapping):
            return
        obs_t = _observation_time_s(observation)
        apertures = _extract_gripper_apertures_m(arms)
        with self._lock:
            self._last_gripper_apertures_m = dict(apertures)
            if obs_t is None:
                return
            for arm_name, arm_doc in arms.items():
                if not isinstance(arm_doc, Mapping):
                    continue
                q_raw = arm_doc.get("q_rad")
                if isinstance(q_raw, (list, tuple, np.ndarray)) and len(q_raw) > 0:
                    q_arr = np.asarray(q_raw, dtype=np.float64).reshape(-1)
                    if np.isfinite(q_arr).all():
                        self._last_arm_q[str(arm_name)] = (q_arr, obs_t)

    def ingest_progress_bundle(
        self,
        observation: Mapping[str, Any],
        *,
        action_id: str | None = None,
        resolve_json: Callable[[str], Any] | None = None,
    ) -> list[CachedWristFrame]:
        """对单帧 5Hz 动作期观测包执行准静态角速度与夹爪开度跃迁因果门控，缓存合格的清晰腕部视角。"""
        if not isinstance(observation, Mapping):
            raise ValueError("progress_observation_must_be_mapping")
        obs_id = observation.get("observation_id")
        if not isinstance(obs_id, str) or not obs_id:
            return []

        wrist_ids = list_wrist_camera_ids(observation, resolve_json=resolve_json)
        if not wrist_ids:
            return []

        cal_by_id, pose_by_id = extract_camera_entries(observation, resolve_json=resolve_json)
        state = self._resolve_robot_state(observation, resolve_json)
        arms_doc = state.get("arms") if isinstance(state, Mapping) and isinstance(state.get("arms"), Mapping) else {}
        obs_t = _observation_time_s(observation)
        eff_action_id = action_id if action_id is not None else observation.get("last_action_id")
        act_key = str(eff_action_id) if eff_action_id is not None else None
        sim_step = observation.get("sim_step") if isinstance(observation.get("sim_step"), int) else None
        curr_apertures = _extract_gripper_apertures_m(arms_doc)
        jump_thresh = self.budget.max_gripper_aperture_jump_m

        frames_by_id = {
            str(item["camera_id"]): item
            for item in (observation.get("cameras") or [])
            if isinstance(item, Mapping) and isinstance(item.get("camera_id"), str)
        }

        admitted: list[CachedWristFrame] = []
        with self._lock:
            ep_id, gen = self._sync_episode_unlocked(observation)
            # 夹爪开闭跃迁因果门控：一旦发生抓取或释放（> 4mm 跳变或累积偏移），立即清空该动作在跃迁前已缓存的旧世界状态帧
            if _has_gripper_aperture_jump(curr_apertures, self._last_gripper_apertures_m, jump_thresh):
                kept = [f for f in self._frames if f.action_id != act_key]
            elif curr_apertures:
                kept = [
                    f
                    for f in self._frames
                    if f.action_id != act_key
                    or not _has_gripper_aperture_jump(curr_apertures, f.gripper_apertures_m, jump_thresh)
                ]
            else:
                kept = None
            if kept is not None and len(kept) < len(self._frames):
                self.invalidated_by_gripper_jump += len(self._frames) - len(kept)
                self._frames.clear()
                self._frames.extend(kept)
            if curr_apertures:
                self._last_gripper_apertures_m.update(curr_apertures)

            arm_speeds: dict[str, float | None] = {}
            for arm_name, arm_doc in arms_doc.items():
                if not isinstance(arm_doc, Mapping):
                    continue
                prev = self._last_arm_q.get(str(arm_name))
                prev_q = prev[0] if prev is not None else None
                dt_s = (obs_t - prev[1]) if (prev is not None and obs_t is not None) else None
                arm_speeds[str(arm_name)] = measure_arm_joint_speed_rad_s(
                    arm_doc, prev_q_rad=prev_q, dt_s=dt_s
                )
                q_raw = arm_doc.get("q_rad")
                if obs_t is not None and isinstance(q_raw, (list, tuple, np.ndarray)) and len(q_raw) > 0:
                    q_arr = np.asarray(q_raw, dtype=np.float64).reshape(-1)
                    if np.isfinite(q_arr).all():
                        self._last_arm_q[str(arm_name)] = (q_arr, obs_t)

            for wrist_id in wrist_ids:
                cam_frame = frames_by_id.get(wrist_id)
                if cam_frame is None:
                    continue
                rgb_ref = cam_frame.get("rgb_ref") if isinstance(cam_frame.get("rgb_ref"), str) else None
                img_b64 = cam_frame.get("image_b64") if isinstance(cam_frame.get("image_b64"), str) else None
                if not rgb_ref and not img_b64:
                    continue

                arm = resolve_camera_arm(wrist_id, cal_by_id.get(wrist_id), pose_by_id.get(wrist_id))
                speed = arm_speeds.get(arm) if arm is not None else None
                if speed is None:
                    self.skipped_missing_velocity += 1
                    continue
                if speed > self.budget.max_wrist_angular_speed_rad_s:
                    self.rejected_motion_blur += 1
                    continue

                kt = _extract_valid_pinhole_and_pose(cal_by_id.get(wrist_id), pose_by_id.get(wrist_id))
                if kt is None:
                    continue
                K, T = kt
                entry = CachedWristFrame(
                    episode_id=ep_id,
                    env_generation=gen,
                    action_id=act_key,
                    observation_id=obs_id,
                    camera_id=wrist_id,
                    arm=arm,
                    sim_step=sim_step,
                    rgb_ref=rgb_ref,
                    image_b64=img_b64 if rgb_ref is None else None,
                    K=tuple(tuple(float(v) for v in row) for row in K),
                    T_world_camera=tuple(tuple(float(v) for v in row) for row in T),
                    joint_speed_rad_s=round(float(speed), 6),
                    gripper_apertures_m=dict(curr_apertures),
                )

                # 若与同一动作同一腕部相机的最近缓存帧机位几乎重合，则原地更新而非重复占满滑窗
                replaced = False
                for idx in range(len(self._frames) - 1, -1, -1):
                    prev_entry = self._frames[idx]
                    if prev_entry.camera_id == wrist_id and prev_entry.action_id == entry.action_id:
                        t_dist, r_deg = _se3_pose_delta(np.asarray(prev_entry.T_world_camera), T)
                        if t_dist < IN_PLACE_DEDUP_TRANSLATION_M and r_deg < IN_PLACE_DEDUP_ANGLE_DEG:
                            self._frames[idx] = entry
                            replaced = True
                        break
                if not replaced:
                    self._frames.append(entry)
                self.accepted_frames += 1
                admitted.append(entry)
        return admitted

    def snapshot(self) -> list[CachedWristFrame]:
        """返回当前缓存中全部准静态腕部视角快照副本（旧到新排序）。"""
        with self._lock:
            return list(self._frames)

    def select_extra_views(
        self,
        observation: Mapping[str, Any],
        *,
        resolve_bytes: Callable[[str], bytes] | None = None,
        resolve_json: Callable[[str], Any] | None = None,
        main_camera_id: str | None = None,
        existing_views: list[dict[str, Any]] | None = None,
        action_id: str | None = None,
        overlap_fn: Callable[[np.ndarray], bool] | None = None,
        max_views: int = 6,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """筛选与目标观测同局同动作、满足夹爪开度一致性、基线间距与视锥交集的缓存腕部视图及源 `observation_id` 列表。"""
        if not isinstance(observation, Mapping) or max_views <= 0:
            return [], []
        target_action_id = action_id if action_id is not None else observation.get("last_action_id")
        if not isinstance(target_action_id, str) or not target_action_id:
            return [], []

        ep_id = str(observation.get("episode_id") or "")
        gen_raw = observation.get("env_generation", 1)
        gen = int(gen_raw) if isinstance(gen_raw, int) and not isinstance(gen_raw, bool) else 1
        curr_obs_id = observation.get("observation_id")

        with self._lock:
            if self._episode_key != (ep_id, gen):
                return [], []
            candidates = [
                f
                for f in self._frames
                if f.action_id == target_action_id
                and f.observation_id != curr_obs_id
                and f.camera_id != main_camera_id
            ]

        if not candidates:
            return [], []

        term_state = self._resolve_robot_state(observation, resolve_json)
        term_arms = term_state.get("arms") if isinstance(term_state, Mapping) else None
        term_apertures = _extract_gripper_apertures_m(term_arms)
        jump_thresh = self.budget.max_gripper_aperture_jump_m

        accepted_poses: list[np.ndarray] = [
            np.asarray(v["T_world_camera"], dtype=np.float64).reshape(4, 4)
            for v in (existing_views or [])
            if isinstance(v, Mapping) and v.get("T_world_camera") is not None
        ]
        min_trans = self.budget.min_view_translation_m
        selected: list[dict[str, Any]] = []
        selected_obs_ids: list[str] = []

        # 优先选择最近采集的清晰视角，并剔除与终态夹爪开度差 > 4mm 的跨接触相位旧帧
        for entry in reversed(candidates):
            if len(selected) >= max_views:
                break
            if term_apertures and _has_gripper_aperture_jump(entry.gripper_apertures_m, term_apertures, jump_thresh):
                continue
            T_cand = np.asarray(entry.T_world_camera, dtype=np.float64).reshape(4, 4)
            if callable(overlap_fn) and not overlap_fn(T_cand):
                continue
            if any(_se3_pose_delta(T_cand, T_prev)[0] < min_trans for T_prev in accepted_poses):
                continue

            if entry.rgb_ref is not None:
                if not callable(resolve_bytes):
                    continue
                raw_bytes = resolve_bytes(entry.rgb_ref)
                b64 = base64.b64encode(raw_bytes).decode("ascii")
            elif entry.image_b64:
                b64 = entry.image_b64
            else:
                continue

            tag = f"{entry.camera_id}@{entry.sim_step}" if entry.sim_step is not None else f"{entry.camera_id}@progress"
            selected.append(
                dict(
                    camera_id=tag,
                    image_b64=b64,
                    K=[list(row) for row in entry.K],
                    T_world_camera=[list(row) for row in entry.T_world_camera],
                )
            )
            accepted_poses.append(T_cand)
            if entry.observation_id not in selected_obs_ids:
                selected_obs_ids.append(entry.observation_id)
        return selected, selected_obs_ids


def ensure_wrist_progress_cache(runtime: Any, *, budget: StreamBudgetConfig | None = None) -> WristProgressCache:
    """获取或初始化挂载在运行时状态账本上的线程安全 `WristProgressCache`。"""
    cache = slot_peek(runtime, WRIST_PROGRESS_CACHE_SLOT, None, legacy_attr="_wrist_progress_cache")
    if not isinstance(cache, WristProgressCache):
        cache = WristProgressCache(budget=budget)
        slot_set(runtime, WRIST_PROGRESS_CACHE_SLOT, cache, legacy_attr="_wrist_progress_cache")
    return cache


def make_artifact_json_resolver(runtime: Any, store: Any = None) -> Callable[[str], Any] | None:
    """构造统一的 `actor` 作用域 JSON 产物解析闭包。"""
    scene = getattr(runtime, "scene", None) if runtime is not None else None
    resolve_json = getattr(scene, "resolve_json", None)
    if callable(resolve_json):
        return resolve_json
    eff_store = store if store is not None else getattr(runtime, "store", None)
    resolve_art = getattr(eff_store, "resolve_artifact", None)
    if callable(resolve_art):
        return lambda ref: json.loads(resolve_art(ref, allowed_scopes=("actor",))[1])
    return None


def make_artifact_bytes_resolver(runtime: Any, store: Any = None) -> Callable[[str], bytes] | None:
    """构造统一的 `actor` 作用域二进制产物解析闭包。"""
    scene = getattr(runtime, "scene", None) if runtime is not None else None
    resolve_scene_art = getattr(scene, "resolve_artifact", None)
    if callable(resolve_scene_art):
        return lambda ref: resolve_scene_art(ref)[1]
    eff_store = store if store is not None else getattr(runtime, "store", None)
    resolve_store_art = getattr(eff_store, "resolve_artifact", None)
    if callable(resolve_store_art):
        return lambda ref: resolve_store_art(ref, allowed_scopes=("actor",))[1]
    return None
