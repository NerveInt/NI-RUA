"""SAM 3.1 跨帧持久身份与有界滑窗视频会话管理器。

提供按 Episode (`session_id`) 隔离的多目标持久身份跟踪、显式重锚定 (`reanchor`)、
基于 `MODEL.predict_inst` 的交互掩码传播，以及高并发显存安全治理：
- `MAX_SESSIONS = 24`（超出按 LRU 立即淘汰并释放全部引用）
- `SESSION_TTL_S = 900`（超时自动回收）
- `MAX_KEYFRAMES = 6`（每会话最多保留最近 6 个关键帧摘要/状态）
- `MAX_GPU_HOT_SESSIONS = 4`（默认保留 4 个 GPU 热会话适配 24GB 单卡，支持 `NI_RUA_SAM31_MAX_HOT_SESSIONS` 环境变量配置，其余张量卸载至 CPU）
"""
from __future__ import annotations

import collections
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

MAX_SESSIONS = 24
SESSION_TTL_S = 900
MAX_KEYFRAMES = 6
DEFAULT_MAX_GPU_HOT_SESSIONS = 4


def resolve_max_gpu_hot_sessions(default: int = DEFAULT_MAX_GPU_HOT_SESSIONS) -> int:
    """读取环境变量 `NI_RUA_SAM31_MAX_HOT_SESSIONS` 解析最大 GPU 热会话数（默认 `4`）。"""
    raw = os.environ.get("NI_RUA_SAM31_MAX_HOT_SESSIONS")
    if raw is None or not raw.strip():
        return int(default)
    val = int(raw.strip())
    if val < 0:
        raise ValueError("invalid_NI_RUA_SAM31_MAX_HOT_SESSIONS")
    return val


MAX_GPU_HOT_SESSIONS = resolve_max_gpu_hot_sessions()
LOW_TRACK_CONF_THRESHOLD = 0.55
MAX_RELOCATION_MISSED_FRAMES = 4
MAX_TRACK_RETENTION_FRAMES = 16


def is_gpu_tensor(obj: Any) -> bool:
    """判断对象是否为驻留在 CUDA 设备上的张量。"""
    dev = getattr(obj, "device", None)
    return getattr(dev, "type", None) == "cuda"


def has_any_gpu_tensor(obj: Any) -> bool:
    """递归检查嵌套结构中是否包含任何 CUDA 张量。"""
    if is_gpu_tensor(obj):
        return True
    if isinstance(obj, dict):
        return any(has_any_gpu_tensor(v) for v in obj.values())
    if isinstance(obj, (list, tuple, collections.deque)):
        return any(has_any_gpu_tensor(v) for v in obj)
    return False


def offload_to_cpu(obj: Any) -> Any:
    """递归将所有具备 `.detach().cpu()` 接口的设备张量迁移至 CPU 内存。"""
    if hasattr(obj, "detach") and hasattr(obj, "cpu"):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: offload_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(offload_to_cpu(v) for v in obj)
    if isinstance(obj, list):
        return [offload_to_cpu(v) for v in obj]
    return obj


def parse_track_num(track_id: Any) -> int | None:
    """解析 `'o3'` 或 `'3'` 形式的持久跟踪编号，返回正整数编号。"""
    if isinstance(track_id, (int, np.integer)):
        val = int(track_id)
        return val if val >= 1 else None
    if not isinstance(track_id, str):
        return None
    s = track_id.strip().lower()
    if s.startswith("o"):
        s = s[1:]
    if not s.isdigit():
        return None
    val = int(s)
    return val if val >= 1 else None


def format_track_id(track_num: int) -> str:
    """将正整数跟踪编号格式化为标准短引用 `'o{k}'`。"""
    return f"o{int(track_num)}"


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """计算两个布尔掩码的交并比 IoU。"""
    inter = int(np.count_nonzero(mask_a & mask_b))
    if inter == 0:
        return 0.0
    union = int(np.count_nonzero(mask_a | mask_b))
    return float(inter / union) if union > 0 else 0.0


def mask_geometry(mask: np.ndarray) -> dict[str, Any]:
    """提取布尔掩码的包围盒、面积、质心与最大内切圆中心点（用于交互传播点击）。"""
    m_u8 = np.asarray(mask, dtype=np.uint8)
    ys, xs = np.nonzero(m_u8)
    if len(xs) == 0:
        return dict(area=0, bbox_xyxy=[0.0, 0.0, 0.0, 0.0], centroid_uv=np.zeros(2, dtype=np.float64), interior_uv=[0.0, 0.0])
    area = int(len(xs))
    bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    centroid = np.array([float(xs.mean()), float(ys.mean())], dtype=np.float64)
    # 在最大连通分量上取距离变换峰值，确保提示点严格落在物体掩码内部而非凹形空洞中
    n_cc, labels_cc, stats_cc, _ = cv2.connectedComponentsWithStats(m_u8, connectivity=8)
    core = (labels_cc == (1 + int(np.argmax(stats_cc[1:, cv2.CC_STAT_AREA])))) if n_cc > 2 else m_u8
    dt = cv2.distanceTransform(np.pad(core.astype(np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    iv, iu = np.unravel_index(int(dt.argmax()), dt.shape)
    return dict(area=area, bbox_xyxy=bbox, centroid_uv=centroid, interior_uv=[float(iu) + 0.5, float(iv) + 0.5])


# SAM 3.1 图像分辨率 1008×1008、ViT patch stride 14 -> 72×72 骨干特征网格，Prompt Encoder 掩码下采样倍率 4 -> 288×288
DEFAULT_LOW_RES_LOGITS_HW: tuple[int, int] = (288, 288)


def to_low_res_logits(
    mask_or_logits: np.ndarray | None,
    mask_bool: np.ndarray,
    target_hw: tuple[int, int] = DEFAULT_LOW_RES_LOGITS_HW,
) -> np.ndarray:
    """生成存储在 CPU 上的 `(1, 288, 288)` float16 低分辨率掩码 Logits，供跨帧 `predict_inst` 复用。"""
    th, tw = int(target_hw[0]), int(target_hw[1])
    if mask_or_logits is not None:
        arr = np.asarray(mask_or_logits, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] >= 1:
            arr2d = arr[0]
        elif arr.ndim == 2:
            arr2d = arr
        else:
            arr2d = None
        if arr2d is not None and arr2d.size > 0:
            # 若输入本身已是目标低分辨率尺寸（如 SAM 3.1 原生 288×288），直接原样保留避免重采样损耗
            if arr2d.shape != (th, tw):
                arr2d = cv2.resize(arr2d, (tw, th), interpolation=cv2.INTER_LINEAR)
            return np.clip(arr2d, -12.0, 12.0).astype(np.float16)[None, :, :]
    soft = cv2.resize(mask_bool.astype(np.float32), (tw, th), interpolation=cv2.INTER_LINEAR)
    logits = (soft * 2.0 - 1.0) * 6.0
    return logits.astype(np.float16)[None, :, :]


def prepare_backbone_feature_np(backbone_feat: Any) -> np.ndarray | None:
    """将单帧 ViT 骨干特征一次性从设备卸载并转换为 `(C, H_f, W_f)` float32 NumPy 数组。"""
    if backbone_feat is None:
        return None
    feat_cpu = offload_to_cpu(backbone_feat)
    if hasattr(feat_cpu, "float") and hasattr(feat_cpu, "numpy"):
        feat_np = feat_cpu.float().numpy()
    else:
        feat_np = np.asarray(feat_cpu, dtype=np.float32)
    if feat_np.ndim == 4:
        feat_np = feat_np[0]
    if feat_np.ndim == 3 and feat_np.shape[0] > 0:
        return feat_np
    return None


def extract_appearance_feature(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    backbone_feat: Any = None,
    *,
    img_hsv: np.ndarray | None = None,
    backbone_feat_np: np.ndarray | None = None,
) -> np.ndarray:
    """提取掩码内部的位置/旋转无关外观特征（支持复用单帧预计算的 `img_hsv` 与 `backbone_feat_np`）。"""
    m_u8 = mask.astype(np.uint8)
    eroded = cv2.erode(m_u8, np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    sel = eroded if int(eroded.sum()) >= 12 else mask.astype(bool)
    if not sel.any():
        return np.zeros(32, dtype=np.float32)

    pix_rgb = img_rgb[sel].astype(np.float32) / 255.0
    hsv_frame = img_hsv if img_hsv is not None else cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hsv = hsv_frame[sel].astype(np.float32)
    h_rad = hsv[:, 0] * (2.0 * math.pi / 180.0)
    s_norm = hsv[:, 1] / 255.0
    v_norm = hsv[:, 2] / 255.0

    # 32 维颜色与色调圆统计描述子（对平移、倾倒、90/180 度旋转完全不变）
    rgb_q = np.percentile(pix_rgb, [15, 50, 85], axis=0).reshape(-1)  # 9
    rgb_std = pix_rgb.std(axis=0)  # 3
    h_cos_sin = np.array(
        [
            float(np.mean(np.cos(h_rad) * s_norm)),
            float(np.mean(np.sin(h_rad) * s_norm)),
            float(np.mean(np.cos(2.0 * h_rad) * s_norm)),
            float(np.mean(np.sin(2.0 * h_rad) * s_norm)),
        ],
        dtype=np.float32,
    )  # 4
    sv_q = np.concatenate([np.percentile(s_norm, [20, 50, 80]), np.percentile(v_norm, [20, 50, 80])]).astype(np.float32)  # 6
    h_hist, _ = np.histogram(hsv[:, 0], bins=10, range=(0.0, 180.0), weights=s_norm + 0.05)
    h_hist = (h_hist / max(float(h_hist.sum()), 1e-6)).astype(np.float32)  # 10
    color_vec = np.concatenate([rgb_q, rgb_std, h_cos_sin, sv_q, h_hist]).astype(np.float32)
    c_norm = float(np.linalg.norm(color_vec))
    if c_norm > 1e-8:
        color_vec = color_vec / c_norm

    feat_np = backbone_feat_np if backbone_feat_np is not None else prepare_backbone_feature_np(backbone_feat)
    if feat_np is not None:
        fh, fw = feat_np.shape[1], feat_np.shape[2]
        m_small = cv2.resize(sel.astype(np.float32), (fw, fh), interpolation=cv2.INTER_AREA)
        w_sum = float(m_small.sum())
        if w_sum > 1e-4:
            pooled = (feat_np * m_small[None, :, :]).sum(axis=(1, 2)) / w_sum
            p_norm = float(np.linalg.norm(pooled))
            if p_norm > 1e-8:
                pooled = (pooled / p_norm).astype(np.float32)
                joint = np.concatenate([color_vec * 0.75, pooled * 0.65]).astype(np.float32)
                j_norm = float(np.linalg.norm(joint))
                return joint / max(j_norm, 1e-8)

    return color_vec


def cosine_sim(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """计算两个一维特征向量的余弦相似度（维度不同时按头部公共颜色子空间对齐）。"""
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 0.0
    if len(a) != len(b):
        k = min(len(a), len(b), 32)
        a_sub = a[:k]
        b_sub = b[:k]
        na = float(np.linalg.norm(a_sub))
        nb = float(np.linalg.norm(b_sub))
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.clip(np.dot(a_sub, b_sub) / (na * nb), -1.0, 1.0))
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


@dataclass
class TrackRecord:
    """单个持久目标在会话中的紧凑 CPU 跟踪状态。"""

    track_num: int
    packed_mask: np.ndarray
    image_hw: tuple[int, int]
    bbox_xyxy: list[float]
    centroid_uv: np.ndarray
    velocity_uv: np.ndarray
    interior_uv: list[float]
    area: int
    appearance: np.ndarray
    low_res_logits: np.ndarray
    labels: list[list[Any]] = field(default_factory=list)
    conf: float = 1.0
    last_frame_seq: int = 0
    missed_frames: int = 0

    @property
    def track_id(self) -> str:
        return format_track_id(self.track_num)

    def clone(self) -> TrackRecord:
        """深拷贝当前轨迹记录，供单帧影子状态事务在提交前隔离推演。"""
        return TrackRecord(
            track_num=int(self.track_num),
            packed_mask=self.packed_mask.copy(),
            image_hw=(int(self.image_hw[0]), int(self.image_hw[1])),
            bbox_xyxy=list(self.bbox_xyxy),
            centroid_uv=self.centroid_uv.copy(),
            velocity_uv=self.velocity_uv.copy(),
            interior_uv=list(self.interior_uv),
            area=int(self.area),
            appearance=self.appearance.copy(),
            low_res_logits=self.low_res_logits.copy(),
            labels=[list(item) for item in self.labels],
            conf=float(self.conf),
            last_frame_seq=int(self.last_frame_seq),
            missed_frames=int(self.missed_frames),
        )

    def unpack_mask(self, target_hw: tuple[int, int] | None = None) -> np.ndarray:
        """按需从紧凑位数组还原布尔掩码，尺寸变化时线性重采样。"""
        h, w = self.image_hw
        m = np.unpackbits(self.packed_mask)[: h * w].reshape(h, w).astype(bool)
        if target_hw is not None and target_hw != (h, w):
            th, tw = target_hw
            m = cv2.resize(m.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST).astype(bool)
        return m


@dataclass
class SessionState:
    """单个 Episode 的跨帧跟踪会话状态（滑窗上限 6 帧）。"""

    session_id: str
    created_at: float
    last_access: float
    max_keyframes: int = MAX_KEYFRAMES
    next_track_num: int = 1
    frame_seq: int = 0
    last_image_sha256: str = ""
    tracks: dict[int, TrackRecord] = field(default_factory=dict)
    keyframes: collections.deque = field(init=False)
    hot_gpu_state: Any = None

    def __post_init__(self) -> None:
        self.keyframes = collections.deque(maxlen=self.max_keyframes)

    def allocate_track_num(self) -> int:
        tid = self.next_track_num
        self.next_track_num += 1
        return tid

    def reserve_track_num(self, track_num: int) -> None:
        if track_num >= self.next_track_num:
            self.next_track_num = track_num + 1

    def offload_gpu_tensors(self) -> None:
        """将当前会话持有的全部设备张量卸载至 CPU。"""
        if self.hot_gpu_state is not None:
            self.hot_gpu_state = offload_to_cpu(self.hot_gpu_state)
        for kf in self.keyframes:
            if "device_state" in kf and kf["device_state"] is not None:
                kf["device_state"] = offload_to_cpu(kf["device_state"])

    def clear(self) -> None:
        """立即清空全部引用，防止任何 CPU/GPU 内存滞留。"""
        self.tracks.clear()
        self.keyframes.clear()
        self.hot_gpu_state = None


class VideoSessionManager:
    """有界滑窗持久跨帧跟踪会话管理器。"""

    def __init__(
        self,
        max_sessions: int = MAX_SESSIONS,
        session_ttl_s: float = SESSION_TTL_S,
        max_keyframes: int = MAX_KEYFRAMES,
        max_gpu_hot_sessions: int | None = None,
    ) -> None:
        self.max_sessions = int(max_sessions)
        self.session_ttl_s = float(session_ttl_s)
        self.max_keyframes = int(max_keyframes)
        resolved_hot = (
            resolve_max_gpu_hot_sessions(MAX_GPU_HOT_SESSIONS)
            if max_gpu_hot_sessions is None
            else int(max_gpu_hot_sessions)
        )
        if resolved_hot < 0:
            raise ValueError("invalid_max_gpu_hot_sessions")
        self.max_gpu_hot_sessions = resolved_hot
        self._lock = threading.RLock()
        self._sessions: collections.OrderedDict[str, SessionState] = collections.OrderedDict()
        self._gpu_hot_lru: collections.OrderedDict[str, None] = collections.OrderedDict()

    def purge_expired(self, now: float | None = None) -> int:
        """清理超过 `SESSION_TTL_S` 未访问的过期会话。"""
        ts = time.time() if now is None else float(now)
        with self._lock:
            expired = [sid for sid, st in self._sessions.items() if ts - st.last_access > self.session_ttl_s]
            for sid in expired:
                self._evict_unlocked(sid)
            return len(expired)

    def _evict_unlocked(self, session_id: str) -> bool:
        self._gpu_hot_lru.pop(session_id, None)
        st = self._sessions.pop(session_id, None)
        if st is None:
            return False
        st.clear()
        return True

    def close_session(self, session_id: str) -> bool:
        """显式关闭指定会话并立即释放其全部 CPU/GPU 引用。"""
        if not session_id:
            return False
        with self._lock:
            return self._evict_unlocked(str(session_id))

    def offload_all_gpu_sessions(self) -> int:
        """将全部活跃会话持有的设备张量卸载至 CPU（用于 CUDA OOM 显存紧急回收）。"""
        with self._lock:
            offloaded = len(self._gpu_hot_lru)
            self._gpu_hot_lru.clear()
            for st in self._sessions.values():
                st.offload_gpu_tensors()
            return offloaded

    def _get_or_create_unlocked(self, session_id: str, now: float, hot_gpu_state: Any = None) -> SessionState:
        self.purge_expired(now)
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
            st = self._sessions[session_id]
            st.last_access = now
        else:
            while len(self._sessions) >= self.max_sessions:
                oldest_sid = next(iter(self._sessions))
                self._evict_unlocked(oldest_sid)
            st = SessionState(
                session_id=session_id,
                created_at=now,
                last_access=now,
                max_keyframes=self.max_keyframes,
            )
            self._sessions[session_id] = st

        if hot_gpu_state is not None:
            if self.max_gpu_hot_sessions > 0 and has_any_gpu_tensor(hot_gpu_state):
                st.hot_gpu_state = hot_gpu_state
                self._gpu_hot_lru.pop(session_id, None)
                self._gpu_hot_lru[session_id] = None
                while len(self._gpu_hot_lru) > self.max_gpu_hot_sessions:
                    cold_sid, _ = self._gpu_hot_lru.popitem(last=False)
                    cold_st = self._sessions.get(cold_sid)
                    if cold_st is not None:
                        cold_st.offload_gpu_tensors()
            else:
                st.hot_gpu_state = offload_to_cpu(hot_gpu_state)
                self._gpu_hot_lru.pop(session_id, None)
        return st

    def stats(self, now: float | None = None) -> dict[str, Any]:
        """返回当前活跃会话数与 GPU 热会话数（供 `/health` 暴露监控）。"""
        with self._lock:
            self.purge_expired(now)
            # 二次核验热会话持有的张量确实在 GPU 上
            stale_hot = [
                sid
                for sid in self._gpu_hot_lru
                if sid not in self._sessions or not has_any_gpu_tensor(self._sessions[sid].hot_gpu_state)
            ]
            for sid in stale_hot:
                self._gpu_hot_lru.pop(sid, None)
            return dict(
                active_sessions=len(self._sessions),
                gpu_hot_sessions=len(self._gpu_hot_lru),
                max_sessions=self.max_sessions,
                session_ttl_s=self.session_ttl_s,
                max_keyframes=self.max_keyframes,
            )

    def get_session_snapshot(self, session_id: str) -> dict[str, Any] | None:
        """只读获取指定会话的诊断快照（用于测试与状态核验）。"""
        with self._lock:
            st = self._sessions.get(session_id)
            if st is None:
                return None
            return dict(
                session_id=st.session_id,
                next_track_num=st.next_track_num,
                frame_seq=st.frame_seq,
                n_keyframes=len(st.keyframes),
                active_track_ids=[t.track_id for t in sorted(st.tracks.values(), key=lambda x: x.track_num)],
                is_gpu_hot=session_id in self._gpu_hot_lru,
            )

    def update_tracks(
        self,
        *,
        session_id: str,
        img_rgb: np.ndarray,
        image_sha256: str,
        candidates: list[dict[str, Any]],
        reanchor: dict[str, Any] | None = None,
        predict_fn: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray | None]] | None = None,
        backbone_feat: Any = None,
        hot_gpu_state: Any = None,
        min_area: int = 16,
        max_area_frac: float = 0.25,
        min_next_track_num: int | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        """执行单帧多目标持久身份关联、交互掩码传播与显式重锚定。

        `candidates` 每项包含：`mask` (H,W bool), `score` (float), `labels` ([[phrase, score], ...]),
        可选 `low_res_logits` 与 `reanchor_match` (bool)。
        `predict_fn(point_coords, point_labels, box, mask_input)` 调用底层 `MODEL.predict_inst` 返回
        `(masks_bool_CxHxW, ious_C, low_res_logits_Cx288x288)`。
        """
        ts = time.time() if now is None else float(now)
        H, W = img_rgb.shape[:2]
        diag = max(math.hypot(W, H), 1.0)
        max_area = int(max_area_frac * H * W)
        reanchor = reanchor if isinstance(reanchor, dict) else {}

        frame_hsv: np.ndarray | None = None
        frame_feat_np: np.ndarray | None = None
        frame_ctx_ready = False

        def _frame_appearance(mask_arr: np.ndarray) -> np.ndarray:
            """单帧内惰性提取至多一次全图 HSV 与 CPU 骨干特征并在全部候选掩码间复用。"""
            nonlocal frame_hsv, frame_feat_np, frame_ctx_ready
            if not frame_ctx_ready:
                frame_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
                frame_feat_np = prepare_backbone_feature_np(backbone_feat)
                frame_ctx_ready = True
            return extract_appearance_feature(
                img_rgb,
                mask_arr,
                img_hsv=frame_hsv,
                backbone_feat_np=frame_feat_np,
            )

        # 在会话锁外完成候选检测面积门控与单次共享特征提取，缩短全局锁临界区耗时
        det_list: list[dict[str, Any]] = []
        for raw in candidates:
            m = np.asarray(raw["mask"], dtype=bool)
            if m.shape != (H, W):
                continue
            area = int(m.sum())
            if area < min_area or area > max_area:
                continue
            geo = mask_geometry(m)
            feat = _frame_appearance(m)
            labs = list(raw.get("labels") or [])
            det_list.append(
                dict(
                    mask=m,
                    score=float(np.clip(raw.get("score", 0.8), 0.0, 1.0)),
                    labels=labs,
                    label_names={x[0] for x in labs if isinstance(x, (list, tuple)) and x},
                    low_res_logits=to_low_res_logits(raw.get("low_res_logits"), m),
                    appearance=feat,
                    reanchor_match=bool(raw.get("reanchor_match", False)),
                    **geo,
                )
            )

        with self._lock:
            reset_flag = bool(reanchor.get("reset", False))
            target_tid_num = parse_track_num(reanchor.get("target_track_id"))
            pos_pixels = [
                [float(u), float(v)]
                for u, v in np.asarray(reanchor.get("pos_pixels") or [], dtype=np.float64).reshape(-1, 2)
                if 0 <= u < W and 0 <= v < H
            ]
            neg_pixels = [
                [float(u), float(v)]
                for u, v in np.asarray(reanchor.get("neg_pixels") or [], dtype=np.float64).reshape(-1, 2)
                if 0 <= u < W and 0 <= v < H
            ]
            reanchor_phrases = {str(p).strip() for p in (reanchor.get("phrases") or []) if str(p).strip()}
            has_reanchor = bool(reset_flag or target_tid_num is not None or pos_pixels or reanchor_phrases)

            # 在影子状态上执行阶段 0~4 推演，确保 predict_fn 等中途抛错时原会话状态 100% 原子回滚零污染
            existing_st = self._sessions.get(session_id)
            if existing_st is not None and (ts - existing_st.last_access > self.session_ttl_s):
                existing_st = None
            full_reset = bool(reset_flag and target_tid_num is None)
            if existing_st is None or full_reset:
                working_tracks: dict[int, TrackRecord] = {}
                working_next_track_num = 1
                working_frame_seq = 0
                working_last_sha = ""
            else:
                working_tracks = {tid: trk.clone() for tid, trk in existing_st.tracks.items()}
                working_next_track_num = int(existing_st.next_track_num)
                working_frame_seq = int(existing_st.frame_seq)
                working_last_sha = str(existing_st.last_image_sha256)

            def _reserve_track_num(track_num: int) -> None:
                nonlocal working_next_track_num
                if track_num >= working_next_track_num:
                    working_next_track_num = track_num + 1

            def _allocate_track_num() -> int:
                nonlocal working_next_track_num
                tid = working_next_track_num
                working_next_track_num += 1
                return tid

            if not full_reset and min_next_track_num is not None and int(min_next_track_num) > 1:
                _reserve_track_num(int(min_next_track_num) - 1)

            same_frame = bool(working_last_sha and working_last_sha == image_sha256 and not has_reanchor)
            if not same_frame:
                working_frame_seq += 1

            emitted: list[dict[str, Any]] = []
            claimed_dets: set[int] = set()
            matched_tracks: set[int] = set()

            def _commit_track(
                tid_num: int,
                det: dict[str, Any],
                status: str,
                conf_val: float,
            ) -> None:
                prev = working_tracks.get(tid_num)
                _reserve_track_num(tid_num)
                if prev is not None and not same_frame:
                    dt_frames = max(1, working_frame_seq - prev.last_frame_seq)
                    inst_vel = (det["centroid_uv"] - prev.centroid_uv) / float(dt_frames)
                    # 大跨度搬运后速度归零，避免下一帧向桌面外过冲外推
                    if np.linalg.norm(inst_vel) > 0.20 * diag:
                        vel = np.zeros(2, dtype=np.float64)
                    else:
                        vel = 0.6 * inst_vel + 0.4 * prev.velocity_uv
                elif prev is not None:
                    vel = prev.velocity_uv.copy()
                else:
                    vel = np.zeros(2, dtype=np.float64)

                merged_labels = list(det["labels"])
                if prev is not None and prev.labels and not merged_labels:
                    merged_labels = list(prev.labels)

                working_tracks[tid_num] = TrackRecord(
                    track_num=tid_num,
                    packed_mask=np.packbits(det["mask"].reshape(-1)),
                    image_hw=(H, W),
                    bbox_xyxy=det["bbox_xyxy"],
                    centroid_uv=det["centroid_uv"],
                    velocity_uv=vel,
                    interior_uv=det["interior_uv"],
                    area=det["area"],
                    appearance=det["appearance"] if prev is None else (
                        (0.65 * det["appearance"] + 0.35 * prev.appearance)
                        / max(float(np.linalg.norm(0.65 * det["appearance"] + 0.35 * prev.appearance)), 1e-8)
                        if len(det["appearance"]) == len(prev.appearance)
                        else det["appearance"]
                    ),
                    low_res_logits=det["low_res_logits"],
                    labels=merged_labels,
                    conf=float(np.clip(conf_val, 0.0, 1.0)),
                    last_frame_seq=working_frame_seq,
                    missed_frames=0,
                )
                matched_tracks.add(tid_num)
                emitted.append(
                    dict(
                        track_num=tid_num,
                        track_id=format_track_id(tid_num),
                        track_conf=round(float(np.clip(conf_val, 0.0, 1.0)), 4),
                        track_status=status,
                        score=round(float(det["score"]), 4),
                        box_xyxy=det["bbox_xyxy"],
                        area=int(det["area"]),
                        labels=merged_labels,
                        mask=det["mask"],
                    )
                )

            # ---------------- 阶段 0：显式重锚定 (reanchor) ----------------
            if pos_pixels and predict_fn is not None:
                pts_np = np.asarray(pos_pixels + neg_pixels, dtype=np.float32).reshape(-1, 2)
                lbl_np = np.asarray([1] * len(pos_pixels) + [0] * len(neg_pixels), dtype=np.int32)
                pmasks, pious, plogits = predict_fn(
                    point_coords=pts_np,
                    point_labels=lbl_np,
                    box=None,
                    mask_input=working_tracks[target_tid_num].low_res_logits if (target_tid_num in working_tracks and not reset_flag) else None,
                )
                best_k, best_q = None, -1.0
                for k in range(len(pmasks)):
                    mk = np.asarray(pmasks[k], dtype=bool)
                    ak = int(mk.sum())
                    if ak < min_area or ak > max_area:
                        continue
                    pos_ok = all(mk[int(v), int(u)] for u, v in pos_pixels)
                    neg_ok = all(not mk[int(v), int(u)] for u, v in neg_pixels)
                    qk = float(pious[k]) + (0.25 if pos_ok else 0.0) + (0.15 if neg_ok else 0.0)
                    if qk > best_q:
                        best_k, best_q = k, qk
                if best_k is not None:
                    rmask = np.asarray(pmasks[best_k], dtype=bool)
                    rconf = float(np.clip(pious[best_k], 0.0, 1.0))
                    rlogits = plogits[best_k : best_k + 1] if plogits is not None and len(plogits) > best_k else None
                    rdet = dict(
                        mask=rmask,
                        score=rconf,
                        labels=[[p, round(rconf, 3)] for p in sorted(reanchor_phrases)],
                        label_names=set(reanchor_phrases),
                        low_res_logits=to_low_res_logits(rlogits, rmask),
                        appearance=_frame_appearance(rmask),
                        **mask_geometry(rmask),
                    )
                    if target_tid_num is not None:
                        chosen_tid = target_tid_num
                    else:
                        # 未指定 target_track_id 时，若与已有活跃轨迹重叠则重锚定该轨迹，否则分配新递增 ID
                        best_tid, best_ov = None, 0.30
                        for tid_c, trk_c in working_tracks.items():
                            ov = mask_iou(rmask, trk_c.unpack_mask((H, W)))
                            if ov >= best_ov:
                                best_tid, best_ov = tid_c, ov
                        chosen_tid = best_tid if best_tid is not None else _allocate_track_num()
                    _commit_track(chosen_tid, rdet, "reanchored", max(rconf, 0.75))
                    # 抑制与重锚定掩码高度重合的重复检测
                    for di, d in enumerate(det_list):
                        if mask_iou(rmask, d["mask"]) >= 0.45:
                            claimed_dets.add(di)

            elif target_tid_num is not None and (reanchor_phrases or det_list):
                # 仅通过 target_track_id + phrases 指定重锚定目标
                cand_indices = [
                    di
                    for di, d in enumerate(det_list)
                    if di not in claimed_dets and (not reanchor_phrases or bool(d["label_names"] & reanchor_phrases) or d["reanchor_match"])
                ]
                if cand_indices:
                    prev_trk = working_tracks.get(target_tid_num)
                    if prev_trk is not None and not reset_flag:
                        prev_m = prev_trk.unpack_mask((H, W))
                        best_di = max(
                            cand_indices,
                            key=lambda di: (
                                0.5 * mask_iou(prev_m, det_list[di]["mask"])
                                + 0.3 * cosine_sim(prev_trk.appearance, det_list[di]["appearance"])
                                + 0.2 * det_list[di]["score"]
                            ),
                        )
                    else:
                        best_di = max(cand_indices, key=lambda di: det_list[di]["score"])
                    claimed_dets.add(best_di)
                    _commit_track(target_tid_num, det_list[best_di], "reanchored", det_list[best_di]["score"])

            # ---------------- 阶段 1：空间 + 运动外推匈牙利匹配 ----------------
            avail_tracks = [
                trk
                for tid, trk in sorted(working_tracks.items())
                if tid not in matched_tracks and trk.missed_frames <= MAX_TRACK_RETENTION_FRAMES
            ]
            avail_det_indices = [di for di in range(len(det_list)) if di not in claimed_dets]

            if avail_tracks and avail_det_indices:
                sim_mat = np.full((len(avail_tracks), len(avail_det_indices)), -1e6, dtype=np.float64)
                prev_masks = [t.unpack_mask((H, W)) for t in avail_tracks]
                for ti, trk in enumerate(avail_tracks):
                    damp = 0.0 if same_frame else (0.5 ** trk.missed_frames)
                    pred_uv = trk.centroid_uv + damp * trk.velocity_uv
                    pred_uv[0] = float(np.clip(pred_uv[0], 0.0, W - 1.0))
                    pred_uv[1] = float(np.clip(pred_uv[1], 0.0, H - 1.0))
                    trk_labels = {x[0] for x in trk.labels if isinstance(x, (list, tuple)) and x}
                    for dj, di in enumerate(avail_det_indices):
                        det = det_list[di]
                        iou_val = mask_iou(prev_masks[ti], det["mask"])
                        dist_norm = float(np.linalg.norm(det["centroid_uv"] - pred_uv)) / diag
                        app_sim = cosine_sim(trk.appearance, det["appearance"])
                        area_ratio = float(min(trk.area, det["area"])) / float(max(trk.area, det["area"], 1))
                        # 空间门控：掩码直接相交，或位于局部邻域且外观与面积尺度一致（支持原地倾倒、90度翻转）
                        spatial_ok = (iou_val >= 0.08) or (
                            trk.missed_frames <= 3 and dist_norm <= 0.22 and app_sim >= 0.48 and area_ratio >= 0.22
                        )
                        if not spatial_ok:
                            continue
                        label_bonus = 0.08 if (trk_labels and det["label_names"] and (trk_labels & det["label_names"])) else 0.0
                        prox = max(0.0, 1.0 - dist_norm / 0.25)
                        sim_mat[ti, dj] = 0.45 * iou_val + 0.25 * prox + 0.20 * max(0.0, app_sim) + 0.10 * area_ratio + label_bonus

                row_ind, col_ind = linear_sum_assignment(-sim_mat)
                for ti, dj in zip(row_ind, col_ind):
                    sc = float(sim_mat[ti, dj])
                    if sc < 0.28:
                        continue
                    trk = avail_tracks[ti]
                    di = avail_det_indices[dj]
                    det = det_list[di]
                    claimed_dets.add(di)
                    status = "reanchored" if (reset_flag or (reanchor_phrases and (det["label_names"] & reanchor_phrases))) else "tracked"
                    conf = float(np.clip(0.65 * det["score"] + 0.35 * min(1.0, sc + 0.25), 0.0, 1.0))
                    _commit_track(trk.track_num, det, status, conf)

            # ---------------- 阶段 2：跨全桌面大跨度重定位匹配 ----------------
            rem_tracks = [
                trk
                for tid, trk in sorted(working_tracks.items())
                if tid not in matched_tracks and trk.missed_frames <= MAX_RELOCATION_MISSED_FRAMES
            ]
            rem_det_indices = [di for di in range(len(det_list)) if di not in claimed_dets]

            if rem_tracks and rem_det_indices:
                reloc_mat = np.full((len(rem_tracks), len(rem_det_indices)), -1e6, dtype=np.float64)
                for ti, trk in enumerate(rem_tracks):
                    trk_labels = {x[0] for x in trk.labels if isinstance(x, (list, tuple)) and x}
                    for dj, di in enumerate(rem_det_indices):
                        det = det_list[di]
                        app_sim = cosine_sim(trk.appearance, det["appearance"])
                        area_ratio = float(min(trk.area, det["area"])) / float(max(trk.area, det["area"], 1))
                        if app_sim < 0.72 or area_ratio < 0.32:
                            continue
                        # 若双方均有明确的非通用语义短语且完全冲突，则禁止跨类别串号
                        if trk_labels and det["label_names"] and not (trk_labels & det["label_names"]):
                            continue
                        label_bonus = 0.10 if (trk_labels and (trk_labels & det["label_names"])) else 0.0
                        reloc_mat[ti, dj] = 0.72 * app_sim + 0.28 * area_ratio + label_bonus

                row_ind, col_ind = linear_sum_assignment(-reloc_mat)
                for ti, dj in zip(row_ind, col_ind):
                    sc = float(reloc_mat[ti, dj])
                    if sc < 0.65:
                        continue
                    trk = rem_tracks[ti]
                    di = rem_det_indices[dj]
                    det = det_list[di]
                    claimed_dets.add(di)
                    status = "reanchored" if (reanchor_phrases and (det["label_names"] & reanchor_phrases)) else "tracked"
                    conf = float(np.clip(0.60 * det["score"] + 0.40 * min(1.0, sc), 0.0, 1.0))
                    _commit_track(trk.track_num, det, status, conf)

            # ---------------- 阶段 3：SAM 3.1 `predict_inst` 交互传播补全 ----------------
            if predict_fn is not None and not same_frame:
                missed_recent = [
                    trk
                    for tid, trk in sorted(working_tracks.items())
                    if tid not in matched_tracks and trk.missed_frames <= 2
                ]
                for trk in missed_recent:
                    damp = 0.5 ** trk.missed_frames
                    shift = damp * trk.velocity_uv
                    pu = float(np.clip(trk.interior_uv[0] + shift[0], 0.0, W - 1.0))
                    pv = float(np.clip(trk.interior_uv[1] + shift[1], 0.0, H - 1.0))
                    bx0, by0, bx1, by1 = trk.bbox_xyxy
                    pad_x = max(4.0, 0.12 * (bx1 - bx0))
                    pad_y = max(4.0, 0.12 * (by1 - by0))
                    ext_box = np.array(
                        [
                            np.clip(bx0 + shift[0] - pad_x, 0.0, W - 1.0),
                            np.clip(by0 + shift[1] - pad_y, 0.0, H - 1.0),
                            np.clip(bx1 + shift[0] + pad_x, 0.0, W - 1.0),
                            np.clip(by1 + shift[1] + pad_y, 0.0, H - 1.0),
                        ],
                        dtype=np.float32,
                    )
                    pmasks, pious, plogits = predict_fn(
                        point_coords=np.array([[pu, pv]], dtype=np.float32),
                        point_labels=np.array([1], dtype=np.int32),
                        box=ext_box,
                        mask_input=trk.low_res_logits,
                    )
                    best_cand = None
                    for k in range(len(pmasks)):
                        mk = np.asarray(pmasks[k], dtype=bool)
                        ak = int(mk.sum())
                        qk = float(pious[k])
                        if qk < 0.62 or ak < min_area or ak > max_area:
                            continue
                        area_ratio = float(min(trk.area, ak)) / float(max(trk.area, ak, 1))
                        if area_ratio < 0.30:
                            continue
                        if any(mask_iou(mk, em["mask"]) >= 0.45 for em in emitted):
                            continue
                        app_vec = _frame_appearance(mk)
                        app_sim = cosine_sim(trk.appearance, app_vec)
                        if app_sim < 0.62:
                            continue
                        score_comb = 0.55 * qk + 0.45 * app_sim
                        if best_cand is None or score_comb > best_cand[0]:
                            lgt = plogits[k : k + 1] if plogits is not None and len(plogits) > k else None
                            best_cand = (score_comb, mk, qk, app_vec, lgt)
                    if best_cand is not None:
                        _, mk, qk, app_vec, lgt = best_cand
                        pdet = dict(
                            mask=mk,
                            score=qk,
                            labels=list(trk.labels),
                            label_names={x[0] for x in trk.labels if isinstance(x, (list, tuple)) and x},
                            low_res_logits=to_low_res_logits(lgt, mk),
                            appearance=app_vec,
                            **mask_geometry(mk),
                        )
                        _commit_track(trk.track_num, pdet, "tracked", float(np.clip(qk * 0.92, 0.0, 1.0)))

            # ---------------- 阶段 4：新入场实例分配单调递增新 ID ----------------
            new_dets = [det_list[di] for di in range(len(det_list)) if di not in claimed_dets]
            # 先按是否含显式标签、置信度与像素坐标确定性排序，仅在首次入场时决定新递增编号
            new_dets.sort(key=lambda d: (not bool(d["labels"]), -d["score"], d["bbox_xyxy"][0], d["bbox_xyxy"][1]))
            for det in new_dets:
                tid_new = _allocate_track_num()
                status_new = (
                    "reanchored"
                    if (reset_flag or det["reanchor_match"] or (reanchor_phrases and (det["label_names"] & reanchor_phrases)))
                    else "new"
                )
                _commit_track(tid_new, det, status_new, det["score"])

            # 更新未出现轨迹的丢失计数，淘汰长期离场轨迹（但绝不回退 next_track_num）
            if not same_frame:
                stale_tids = []
                for tid, trk in working_tracks.items():
                    if tid not in matched_tracks:
                        trk.missed_frames += 1
                        trk.velocity_uv *= 0.5
                        if trk.missed_frames > MAX_TRACK_RETENTION_FRAMES:
                            stale_tids.append(tid)
                for tid in stale_tids:
                    working_tracks.pop(tid, None)

            # 全部阶段无异常完成后一次性原子提交至 SessionState
            st = self._get_or_create_unlocked(session_id, ts, hot_gpu_state=hot_gpu_state)
            if full_reset:
                st.keyframes.clear()
            st.tracks = working_tracks
            st.next_track_num = working_next_track_num
            st.frame_seq = working_frame_seq
            st.last_image_sha256 = image_sha256
            emitted.sort(key=lambda x: x["track_num"])

            # 维护最多 6 帧有界滑窗关键帧摘要，历史帧设备张量一律保持在 CPU
            if not same_frame or not st.keyframes:
                for kf in st.keyframes:
                    if kf.get("device_state") is not None:
                        kf["device_state"] = offload_to_cpu(kf["device_state"])
                st.keyframes.append(
                    dict(
                        frame_seq=st.frame_seq,
                        timestamp=ts,
                        image_sha256=image_sha256,
                        image_size=[W, H],
                        tracks={
                            item["track_id"]: dict(
                                bbox_xyxy=item["box_xyxy"],
                                area=item["area"],
                                track_conf=item["track_conf"],
                                track_status=item["track_status"],
                            )
                            for item in emitted
                        },
                        device_state=st.hot_gpu_state if session_id in self._gpu_hot_lru else offload_to_cpu(hot_gpu_state),
                    )
                )

            confs = [item["track_conf"] for item in emitted]
            low_conf_ids = [item["track_id"] for item in emitted if item["track_conf"] < LOW_TRACK_CONF_THRESHOLD]
            return dict(
                session_id=session_id,
                frame_seq=st.frame_seq,
                next_track_num=st.next_track_num,
                instances=emitted,
                active_tracks=len(st.tracks),
                min_track_conf=round(min(confs), 4) if confs else None,
                low_conf_ids=low_conf_ids,
                keyframes_kept=len(st.keyframes),
                gpu_hot_sessions=len(self._gpu_hot_lru),
                active_sessions=len(self._sessions),
            )
