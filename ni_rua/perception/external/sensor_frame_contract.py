"""传感器拓扑角色判定与深度/双目/流式预算数据契约。

提供与具体相机命名解耦的拓扑分类（通用固定外部视角 `fixed_external` 与随动腕部视角 `wrist_mounted`），
同时 100% 兼容历史默认名称（`cam_head`、`cam_left_wrist`、`cam_right_wrist`）；
并定义面向真机接入的 RGB-D 深度帧契约 `DepthFrameSpec`、双目相机对契约 `StereoPairSpec`
及单卡 24GB–40GB 流式会话显存预算配置 `StreamBudgetConfig`。
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

CAMERA_ROLE_FIXED_EXTERNAL = "fixed_external"
CAMERA_ROLE_WRIST_MOUNTED = "wrist_mounted"
VALID_CAMERA_ROLES = frozenset({CAMERA_ROLE_FIXED_EXTERNAL, CAMERA_ROLE_WRIST_MOUNTED})

LEGACY_HEAD_CAMERA_ID = "cam_head"
LEGACY_WRIST_CAMERA_IDS = ("cam_left_wrist", "cam_right_wrist")


def _is_real_number(val: Any) -> bool:
    """检查数值是否为非布尔类型的有限实数。"""
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))


def _is_valid_int(val: Any) -> bool:
    """检查数值是否为非布尔类型的整数。"""
    return isinstance(val, int) and not isinstance(val, bool)


def _has_kinematic_mount(entry: Mapping[str, Any] | None) -> bool:
    """判断标定或位姿字典是否声明了机械臂运动学连杆挂载字段。"""
    if not isinstance(entry, Mapping):
        return False
    for key in ("arm", "link_name"):
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            return True
    return entry.get("T_link_camera") is not None


def classify_camera_role(
    camera_id: str,
    calibration_entry: Mapping[str, Any] | None = None,
    pose_entry: Mapping[str, Any] | None = None,
) -> str:
    """判定相机的物理安装拓扑角色（`fixed_external` 或 `wrist_mounted`）。

    判定优先级：
    1. 若标定或位姿字典显式声明 `mount_role` / `camera_role`，校验合法性与无矛盾性后直接采用；
    2. 若标定或位姿字典包含 `arm`、`link_name` 或手眼矩阵 `T_link_camera`，判定为 `wrist_mounted`；
    3. 若 `camera_id` 属于历史腕部命名（如 `cam_left_wrist` / `cam_right_wrist` 或含 `wrist`），判定为 `wrist_mounted`；
    4. 其余固定在头部立柱或第三方三脚架/龙门架上的外部相机（如 `cam_head`、`cam_third_person`、`cam_overhead`），
       统一判定为 `fixed_external`。
    """
    if not isinstance(camera_id, str) or not camera_id.strip():
        raise ValueError("invalid_camera_id")
    if calibration_entry is not None and not isinstance(calibration_entry, Mapping):
        raise ValueError("calibration_entry_must_be_mapping")
    if pose_entry is not None and not isinstance(pose_entry, Mapping):
        raise ValueError("pose_entry_must_be_mapping")

    has_mount = _has_kinematic_mount(calibration_entry) or _has_kinematic_mount(pose_entry)

    explicit_role: str | None = None
    for entry in (pose_entry, calibration_entry):
        if isinstance(entry, Mapping):
            raw_role = entry.get("mount_role") if "mount_role" in entry else entry.get("camera_role")
            if raw_role is not None:
                if not isinstance(raw_role, str) or raw_role not in VALID_CAMERA_ROLES:
                    raise ValueError(f"unsupported_camera_role:{raw_role}")
                if explicit_role is not None and explicit_role != raw_role:
                    raise ValueError("conflicting_explicit_camera_roles")
                explicit_role = raw_role

    if explicit_role is not None:
        if explicit_role == CAMERA_ROLE_FIXED_EXTERNAL and has_mount:
            raise ValueError("contradictory_camera_mount_metadata")
        return explicit_role

    if has_mount:
        return CAMERA_ROLE_WRIST_MOUNTED

    cid_norm = camera_id.strip().lower()
    if camera_id in LEGACY_WRIST_CAMERA_IDS or "wrist" in cid_norm:
        return CAMERA_ROLE_WRIST_MOUNTED
    return CAMERA_ROLE_FIXED_EXTERNAL


def resolve_camera_arm(
    camera_id: str,
    calibration_entry: Mapping[str, Any] | None = None,
    pose_entry: Mapping[str, Any] | None = None,
) -> str | None:
    """解析随动腕部相机所属的机械臂侧标识（如 `'left'` 或 `'right'`）；固定外部视角返回 `None`。"""
    role = classify_camera_role(camera_id, calibration_entry, pose_entry)
    if role != CAMERA_ROLE_WRIST_MOUNTED:
        return None
    for entry in (pose_entry, calibration_entry):
        if isinstance(entry, Mapping):
            arm = entry.get("arm")
            if isinstance(arm, str) and arm.strip():
                return arm.strip()
    cid_low = camera_id.strip().lower()
    if "left" in cid_low:
        return "left"
    if "right" in cid_low:
        return "right"
    return None


def _normalize_camera_map(doc: Any) -> dict[str, dict[str, Any]]:
    """将标准相机字典映射或相机列表统一规整为 `{camera_id: dict}`。"""
    if not isinstance(doc, Mapping):
        return {}
    cams = doc.get("cameras")
    if isinstance(cams, Mapping):
        return {str(k): dict(v) for k, v in cams.items() if isinstance(k, str) and k and isinstance(v, Mapping)}
    if isinstance(cams, list):
        return {
            str(item["camera_id"]): dict(item)
            for item in cams
            if isinstance(item, Mapping) and isinstance(item.get("camera_id"), str) and item["camera_id"]
        }
    return {}


def extract_camera_entries(
    observation: Mapping[str, Any] | None,
    *,
    resolve_json: Callable[[str], Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """从 `ObservationBundle` 提取各相机的 `(calibration_by_id, pose_sample_by_id)` 字典。"""
    if not isinstance(observation, Mapping):
        return {}, {}
    cal_by_id = _normalize_camera_map(observation.get("camera_calibration"))
    pose_by_id = _normalize_camera_map(observation.get("camera_pose_sample"))

    if callable(resolve_json):
        if not cal_by_id and isinstance(observation.get("camera_calibration_ref"), str):
            cal_by_id = _normalize_camera_map(resolve_json(observation["camera_calibration_ref"]))
        if not pose_by_id and isinstance(observation.get("camera_pose_sample_ref"), str):
            pose_by_id = _normalize_camera_map(resolve_json(observation["camera_pose_sample_ref"]))

    resolved_cal_refs: dict[str, dict[str, dict[str, Any]]] = {}
    resolved_pose_refs: dict[str, dict[str, dict[str, Any]]] = {}
    for frame in observation.get("cameras") or []:
        if not isinstance(frame, Mapping):
            continue
        cid = frame.get("camera_id")
        if not isinstance(cid, str) or not cid:
            continue
        if isinstance(frame.get("calibration"), Mapping) and cid not in cal_by_id:
            frame_cal = _normalize_camera_map(frame["calibration"])
            cal_by_id[cid] = frame_cal.get(cid, dict(frame["calibration"]))
        elif callable(resolve_json) and isinstance(frame.get("calibration_ref"), str) and cid not in cal_by_id:
            cref = frame["calibration_ref"]
            if cref not in resolved_cal_refs:
                resolved_cal_refs[cref] = _normalize_camera_map(resolve_json(cref))
            if cid in resolved_cal_refs[cref]:
                cal_by_id[cid] = resolved_cal_refs[cref][cid]

        if isinstance(frame.get("pose_sample"), Mapping) and cid not in pose_by_id:
            frame_pose = _normalize_camera_map(frame["pose_sample"])
            pose_by_id[cid] = frame_pose.get(cid, dict(frame["pose_sample"]))
        elif callable(resolve_json) and isinstance(frame.get("pose_sample_ref"), str) and cid not in pose_by_id:
            pref = frame["pose_sample_ref"]
            if pref not in resolved_pose_refs:
                resolved_pose_refs[pref] = _normalize_camera_map(resolve_json(pref))
            if cid in resolved_pose_refs[pref]:
                pose_by_id[cid] = resolved_pose_refs[pref][cid]
    return cal_by_id, pose_by_id


def list_fixed_camera_ids(
    observation: Mapping[str, Any] | None,
    *,
    resolve_json: Callable[[str], Any] | None = None,
) -> tuple[str, ...]:
    """按观测帧内顺序返回全部 `fixed_external` 固定外部视角相机 ID 元组。"""
    if not isinstance(observation, Mapping):
        return ()
    cal_by_id, pose_by_id = extract_camera_entries(observation, resolve_json=resolve_json)
    fixed_ids: list[str] = []
    seen: set[str] = set()
    for frame in observation.get("cameras") or []:
        if not isinstance(frame, Mapping):
            continue
        cid = frame.get("camera_id")
        if not isinstance(cid, str) or not cid or cid in seen:
            continue
        seen.add(cid)
        if classify_camera_role(cid, cal_by_id.get(cid), pose_by_id.get(cid)) == CAMERA_ROLE_FIXED_EXTERNAL:
            fixed_ids.append(cid)
    return tuple(fixed_ids)


def list_wrist_camera_ids(
    observation: Mapping[str, Any] | None,
    *,
    resolve_json: Callable[[str], Any] | None = None,
) -> tuple[str, ...]:
    """按观测帧内顺序返回全部 `wrist_mounted` 随动腕部相机 ID 元组。"""
    if not isinstance(observation, Mapping):
        return ()
    cal_by_id, pose_by_id = extract_camera_entries(observation, resolve_json=resolve_json)
    wrist_ids: list[str] = []
    seen: set[str] = set()
    for frame in observation.get("cameras") or []:
        if not isinstance(frame, Mapping):
            continue
        cid = frame.get("camera_id")
        if not isinstance(cid, str) or not cid or cid in seen:
            continue
        seen.add(cid)
        if classify_camera_role(cid, cal_by_id.get(cid), pose_by_id.get(cid)) == CAMERA_ROLE_WRIST_MOUNTED:
            wrist_ids.append(cid)
    return tuple(wrist_ids)


def select_primary_fixed_camera_id(
    observation: Mapping[str, Any] | None,
    *,
    preferred_camera_id: str | None = None,
    resolve_json: Callable[[str], Any] | None = None,
) -> str:
    """选择主固定外部视角相机 ID（兼容 `cam_head`，不存在时自动选用第三方固定机位如 `cam_third_person`）。"""
    if preferred_camera_id is not None:
        if not isinstance(preferred_camera_id, str) or not preferred_camera_id.strip():
            raise ValueError("invalid_preferred_camera_id")
        return preferred_camera_id.strip()
    if not isinstance(observation, Mapping) or not observation.get("cameras"):
        return LEGACY_HEAD_CAMERA_ID
    fixed_ids = list_fixed_camera_ids(observation, resolve_json=resolve_json)
    if LEGACY_HEAD_CAMERA_ID in fixed_ids:
        return LEGACY_HEAD_CAMERA_ID
    if fixed_ids:
        return fixed_ids[0]
    raise ValueError("no_fixed_external_camera_in_observation")


@dataclass(frozen=True)
class DepthFrameSpec:
    """RGB-D 对齐深度帧数据契约（仅声明物理接口与严格校验，不含未接入硬件死逻辑）。

    字段说明：
    - `depth_ref`: 不可变深度数组产物引用句柄（对应单通道公制 Z 深度 NPZ 产物，非空字符串）。
    - `camera_id`: 与该深度图严格像素对齐的相机标识符（如 `'cam_third_person'` 或 `'cam_right_wrist'`）。
    - `image_size_px`: 深度图分辨率 `(width_px, height_px)`，单位 `px`，每维整数范围 `[1, 4096]`。
    - `unit`: 转换后的物理深度单位，固定为 `'m'`（公制米）。
    - `depth_scale`: 传感器原始数值到公制米（`m`）的线性乘性比例因子（默认 `1.0`；如 16 位毫米深度计为 `0.001`）。
    - `valid_range_m`: 传感器有效物理量程 `(min_depth_m, max_depth_m)`，单位 `m`（默认 `(0.05, 2.50)`，需满足 `0.0 <= min < max <= 20.0`）。
    - `timestamp_ns`: 传感器硬件曝光中点单调时钟戳，单位 `ns`（纳秒，默认 `None` 表示仿真步时钟或未采集硬件纳秒戳）。
    """

    depth_ref: str
    camera_id: str
    image_size_px: tuple[int, int]
    unit: str = "m"
    depth_scale: float = 1.0
    valid_range_m: tuple[float, float] = (0.05, 2.50)
    timestamp_ns: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.depth_ref, str) or not self.depth_ref.strip():
            raise ValueError("invalid_depth_ref")
        if not isinstance(self.camera_id, str) or not self.camera_id.strip():
            raise ValueError("invalid_depth_camera_id")
        if (
            not isinstance(self.image_size_px, (tuple, list))
            or len(self.image_size_px) != 2
            or not all(_is_valid_int(v) and 1 <= v <= 4096 for v in self.image_size_px)
        ):
            raise ValueError("invalid_depth_image_size_px")
        object.__setattr__(self, "image_size_px", (int(self.image_size_px[0]), int(self.image_size_px[1])))
        if self.unit != "m":
            raise ValueError("depth_unit_must_be_m")
        if not _is_real_number(self.depth_scale) or float(self.depth_scale) <= 0.0:
            raise ValueError("invalid_depth_scale")
        object.__setattr__(self, "depth_scale", float(self.depth_scale))
        if (
            not isinstance(self.valid_range_m, (tuple, list))
            or len(self.valid_range_m) != 2
            or not all(_is_real_number(v) for v in self.valid_range_m)
        ):
            raise ValueError("invalid_depth_valid_range_m")
        near_m, far_m = float(self.valid_range_m[0]), float(self.valid_range_m[1])
        if not (0.0 <= near_m < far_m <= 20.0):
            raise ValueError("depth_valid_range_out_of_bounds")
        object.__setattr__(self, "valid_range_m", (near_m, far_m))
        if self.timestamp_ns is not None:
            if not _is_valid_int(self.timestamp_ns) or self.timestamp_ns < 0:
                raise ValueError("invalid_depth_timestamp_ns")

    @classmethod
    def from_camera_frame(cls, camera: Mapping[str, Any]) -> DepthFrameSpec:
        """从 `ObservationBundle` 相机条目构造并校验 `DepthFrameSpec`。"""
        if not isinstance(camera, Mapping):
            raise ValueError("camera_frame_must_be_mapping")
        clip = camera.get("depth_clip_m", (0.05, 2.50))
        if not isinstance(clip, (tuple, list)) or len(clip) != 2:
            raise ValueError("invalid_depth_clip_m")
        return cls(
            depth_ref=camera.get("depth_ref", ""),
            camera_id=camera.get("camera_id", ""),
            image_size_px=(camera.get("width"), camera.get("height")),
            unit=camera.get("depth_unit", "m"),
            depth_scale=camera.get("depth_scale", 1.0),
            valid_range_m=(clip[0], clip[1]),
            timestamp_ns=camera.get("timestamp_ns"),
        )


@dataclass(frozen=True)
class StereoPairSpec:
    """极简双目相机对预留数据契约（声明双目光心基线与极线校正状态）。

    字段说明：
    - `left_camera_id`: 左目相机唯一标识符（非空字符串，不可与右目相同）。
    - `right_camera_id`: 右目相机唯一标识符（非空字符串，不可与左目相同）。
    - `baseline_m`: 左右目光心之间的刚性物理基线距离，单位 `m`（必须满足 `0.005 <= baseline_m <= 2.0`）。
    - `is_rectified`: 双目图像对是否已完成极线共面行对齐校正（默认 `True`；为 `True` 时视差仅沿水平 `u` 轴搜索）。
    - `image_size_px`: 可选同步双目图像尺寸 `(width_px, height_px)`，单位 `px`（默认 `None`）。
    - `disparity_range_px`: 有效水平视差搜索区间 `(min_disp_px, max_disp_px)`，单位 `px`（默认 `(0.0, 256.0)`）。
    """

    left_camera_id: str
    right_camera_id: str
    baseline_m: float
    is_rectified: bool = True
    image_size_px: tuple[int, int] | None = None
    disparity_range_px: tuple[float, float] = (0.0, 256.0)

    def __post_init__(self) -> None:
        if not isinstance(self.left_camera_id, str) or not self.left_camera_id.strip():
            raise ValueError("invalid_left_camera_id")
        if not isinstance(self.right_camera_id, str) or not self.right_camera_id.strip():
            raise ValueError("invalid_right_camera_id")
        if self.left_camera_id.strip() == self.right_camera_id.strip():
            raise ValueError("stereo_pair_requires_distinct_cameras")
        if not _is_real_number(self.baseline_m) or not (0.005 <= float(self.baseline_m) <= 2.0):
            raise ValueError("invalid_stereo_baseline_m")
        object.__setattr__(self, "baseline_m", float(self.baseline_m))
        if not isinstance(self.is_rectified, bool):
            raise ValueError("stereo_is_rectified_must_be_bool")
        if self.image_size_px is not None:
            if (
                not isinstance(self.image_size_px, (tuple, list))
                or len(self.image_size_px) != 2
                or not all(_is_valid_int(v) and 1 <= v <= 4096 for v in self.image_size_px)
            ):
                raise ValueError("invalid_stereo_image_size_px")
            object.__setattr__(self, "image_size_px", (int(self.image_size_px[0]), int(self.image_size_px[1])))
        if (
            not isinstance(self.disparity_range_px, (tuple, list))
            or len(self.disparity_range_px) != 2
            or not all(_is_real_number(v) for v in self.disparity_range_px)
        ):
            raise ValueError("invalid_disparity_range_px")
        d_min, d_max = float(self.disparity_range_px[0]), float(self.disparity_range_px[1])
        if not (0.0 <= d_min < d_max <= 4096.0):
            raise ValueError("disparity_range_px_out_of_bounds")
        object.__setattr__(self, "disparity_range_px", (d_min, d_max))


@dataclass(frozen=True)
class StreamBudgetConfig:
    """面向单张 24GB 级显卡（可扩展至 40GB）的流式视觉会话与滑窗预算契约。

    字段说明：
    - `vram_budget_gb`: 单卡感知总显存目标预算上限，单位 `GB`（默认 `24.0`，合法范围 `[8.0, 128.0]`）。
    - `max_gpu_hot_sessions`: 常驻 GPU 显存的活跃分割跟踪会话数上限（默认 `4`，40GB 预设为 `8`）。
    - `max_keyframes`: 单会话滑窗保留的最大关键帧摘要数，单位 `帧`（默认 `6`，40GB 预设为 `10`）。
    - `max_wrist_progress_frames`: 动作执行期准静态门控后缓存的清晰腕部多视角帧上限，单位 `帧`（默认 `4`）。
    - `target_progress_hz`: 动作执行期过程观测标称采样频率，单位 `Hz`（默认 `5.0 Hz`）。
    - `max_wrist_angular_speed_rad_s`: 腕部过程帧防快门运动模糊的最大允许关节角速度，单位 `rad/s`（默认 `0.8 rad/s`）。
    - `min_view_translation_m`: 触发新腕部视角入列融合所需的最小三维光心平移基线，单位 `m`（默认 `0.04 m`）。
    - `max_gripper_aperture_jump_m`: 判定发生夹爪抓取或释放的实测开度跳变阈值，单位 `m`（默认 `0.004 m`；超阈时自动失效跳变前过程帧）。
    """

    vram_budget_gb: float = 24.0
    max_gpu_hot_sessions: int = 4
    max_keyframes: int = 6
    max_wrist_progress_frames: int = 4
    target_progress_hz: float = 5.0
    max_wrist_angular_speed_rad_s: float = 0.8
    min_view_translation_m: float = 0.04
    max_gripper_aperture_jump_m: float = 0.004

    def __post_init__(self) -> None:
        if not _is_real_number(self.vram_budget_gb) or not (8.0 <= float(self.vram_budget_gb) <= 128.0):
            raise ValueError("invalid_vram_budget_gb")
        object.__setattr__(self, "vram_budget_gb", float(self.vram_budget_gb))
        if not _is_valid_int(self.max_gpu_hot_sessions) or self.max_gpu_hot_sessions < 0:
            raise ValueError("invalid_max_gpu_hot_sessions")
        if not _is_valid_int(self.max_keyframes) or self.max_keyframes < 1:
            raise ValueError("invalid_max_keyframes")
        if not _is_valid_int(self.max_wrist_progress_frames) or self.max_wrist_progress_frames < 1:
            raise ValueError("invalid_max_wrist_progress_frames")
        if not _is_real_number(self.target_progress_hz) or float(self.target_progress_hz) <= 0.0:
            raise ValueError("invalid_target_progress_hz")
        object.__setattr__(self, "target_progress_hz", float(self.target_progress_hz))
        if not _is_real_number(self.max_wrist_angular_speed_rad_s) or float(self.max_wrist_angular_speed_rad_s) <= 0.0:
            raise ValueError("invalid_max_wrist_angular_speed_rad_s")
        object.__setattr__(self, "max_wrist_angular_speed_rad_s", float(self.max_wrist_angular_speed_rad_s))
        if not _is_real_number(self.min_view_translation_m) or float(self.min_view_translation_m) < 0.0:
            raise ValueError("invalid_min_view_translation_m")
        object.__setattr__(self, "min_view_translation_m", float(self.min_view_translation_m))
        if not _is_real_number(self.max_gripper_aperture_jump_m) or float(self.max_gripper_aperture_jump_m) <= 0.0:
            raise ValueError("invalid_max_gripper_aperture_jump_m")
        object.__setattr__(self, "max_gripper_aperture_jump_m", float(self.max_gripper_aperture_jump_m))

    @classmethod
    def for_vram_gb(cls, vram_gb: float = 24.0) -> StreamBudgetConfig:
        """按目标单卡显存容量（如 24GB 或 40GB）生成匹配的流式滑窗预算配置。"""
        if not _is_real_number(vram_gb):
            raise ValueError("invalid_vram_budget_gb")
        budget = float(vram_gb)
        if budget >= 36.0:
            return cls(
                vram_budget_gb=budget,
                max_gpu_hot_sessions=8,
                max_keyframes=10,
                max_wrist_progress_frames=6,
            )
        return cls(vram_budget_gb=budget)
