"""Phase 2 感知服务端升级单元测试与本地端到端 HTTP 契约测试。

覆盖范围：
1. `VideoSessionManager`: 跨帧持久 ID 锁定（倾倒/置信度翻转/跨全桌面搬运不串号、漏检交互传播、新目标递增分配）、
   显式 `reanchor`、`MAX_SESSIONS=24` LRU、`SESSION_TTL_S=900`、`MAX_KEYFRAMES=6`、`gpu_hot_sessions <= 2` 显存卸载与 `session_close`。
2. `multiview_cluster`: 1cm 体素 26 邻域快速主簇提取（精度对齐与 `N=4000` 单次 `< 2ms`）、未截断凹腔深度 `cavity_depth_m`、
   最多 3 层 <= 12 顶点 2D 凸包切片 `surface_slices_m`。
3. `sam31_service` + `vggt_height_service` 本地临时端口 HTTP 联调（不触碰宿主机 `:8331`/`:8333`）：
   `/object_table` -> 同图缓存 `entry["instance_masks"]` -> `/points` `.npz` (`obj_id`, `view_mask`)、
   多视角 `[S, 3, H, W]` 单次 Batch 前向与 `<= 0.025m` 严格残差门控正反例、级联 `/session_close` 及无状态回退向后兼容。
"""
from __future__ import annotations

NI_RUA_TEST_TIER = "component"

import base64
import io
import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from typing import Any

import cv2
import numpy as np
from PIL import Image

from services.vggt_height import multiview_cluster as mc
from services.vggt_height import vggt_height_service as vggt_svc
from services.vggt_height.sam31 import sam31_service as sam31_svc
from services.vggt_height.sam31 import video_session_manager as vsm


class _FakeDevice:
    def __init__(self, dev_type: str) -> None:
        self.type = dev_type


class _FakeTensor:
    """模拟 PyTorch CUDA/CPU 张量，用于验证非热会话张量自动卸载至 CPU。"""

    def __init__(self, arr: np.ndarray, device_type: str = "cuda") -> None:
        self._arr = np.asarray(arr, dtype=np.float32)
        self.device = _FakeDevice(device_type)

    def detach(self) -> "_FakeTensor":
        return self

    def float(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return _FakeTensor(self._arr.copy(), device_type="cpu")

    def numpy(self) -> np.ndarray:
        return self._arr


def _encode_png_b64(img_rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _rect_mask(h: int, w: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)] = True
    return m


def _paint_masked(img: np.ndarray, mask: np.ndarray, rgb: tuple[int, int, int]) -> None:
    # 叠加细微纹理使颜色统计与色调分布具备真实方差
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return
    base = np.array(rgb, dtype=np.int32)
    mod = ((xs * 3 + ys * 5) % 11 - 5)[:, None]
    img[ys, xs] = np.clip(base[None, :] + mod, 0, 255).astype(np.uint8)


def _look_at_camera(cam_pos: list[float], target: list[float], up_hint: list[float] | None = None) -> np.ndarray:
    """构造 OpenCV 相机外参 `T_world_camera`（右 X、下 Y、前 +Z）。"""
    c = np.asarray(cam_pos, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    up = np.asarray(up_hint if up_hint is not None else [0.0, 0.0, 1.0], dtype=np.float64)
    z_axis = tgt - c
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(z_axis, up)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    T = np.eye(4, dtype=np.float64)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[:3, 3] = c
    return T


def _render_synthetic_depth(
    H: int,
    W: int,
    K: np.ndarray,
    T: np.ndarray,
    support_z: float,
    boxes: list[dict[str, Any]],
    scale_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """对桌面 + 若干轴对齐凹腔/实心盒体进行精确射线求交，生成合成相对深度与置信度图。"""
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64) + 0.5, np.arange(H, dtype=np.float64) + 0.5)
    dirs_cam = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], axis=-1)
    R = T[:3, :3]
    C = T[:3, 3]
    dirs_world = dirs_cam @ R.T  # 注意 z_cam = 1 时，射线参数 t 直接等于相机坐标系深度 z_cam

    # 与桌面 z = support_z 求交
    dz = np.where(np.abs(dirs_world[..., 2]) > 1e-8, dirs_world[..., 2], -1e-8)
    t_table = (support_z - C[2]) / dz
    z_metric = np.where(t_table > 0.05, t_table, 3.5)
    hit_pts = C + z_metric[..., None] * dirs_world

    # 对每个盒体（支持外框高度 top_z 与可选内腔高度 floor_z）近似顶面覆盖求交
    for bx in boxes:
        x0, x1 = bx["x_range"]
        y0, y1 = bx["y_range"]
        top_z = float(bx["top_z"])
        floor_z = bx.get("floor_z")
        wall = float(bx.get("wall_m", 0.012))

        t_top = (top_z - C[2]) / dz
        p_top = C + t_top[..., None] * dirs_world
        in_outer = (
            (t_top > 0.05)
            & (p_top[..., 0] >= x0)
            & (p_top[..., 0] <= x1)
            & (p_top[..., 1] >= y0)
            & (p_top[..., 1] <= y1)
        )
        if floor_z is not None:
            t_flr = (float(floor_z) - C[2]) / dz
            p_flr = C + t_flr[..., None] * dirs_world
            in_cavity = (
                (t_flr > 0.05)
                & (p_flr[..., 0] >= x0 + wall)
                & (p_flr[..., 0] <= x1 - wall)
                & (p_flr[..., 1] >= y0 + wall)
                & (p_flr[..., 1] <= y1 - wall)
            )
            rim_hit = in_outer & ~in_cavity
            z_metric = np.where(rim_hit & (t_top < z_metric), t_top, z_metric)
            z_metric = np.where(in_cavity & (t_flr < z_metric), t_flr, z_metric)
        else:
            z_metric = np.where(in_outer & (t_top < z_metric), t_top, z_metric)
        hit_pts = C + z_metric[..., None] * dirs_world
        _ = hit_pts

    rng = np.random.default_rng(42)
    conf = (0.85 + 0.12 * rng.random((H, W))).astype(np.float32)
    rel_depth = (z_metric / float(scale_factor)).astype(np.float32)
    return rel_depth, conf


def _http_json(url: str, body: dict[str, Any] | None = None, method: str = "POST") -> tuple[int, dict[str, Any]]:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


class TestVideoSessionManager(unittest.TestCase):
    """验证 SAM 3.1 会话管理器的持久身份锁定、重锚定与 16 并发显存红线。"""

    def test_persistent_track_ids_across_score_inversion_topple_and_relocation(self) -> None:
        mgr = vsm.VideoSessionManager()
        H, W = 96, 128

        # 第 1 帧：三个不同颜色物体入场，按初始置信度分配 o1, o2, o3
        img1 = np.full((H, W, 3), 180, dtype=np.uint8)
        m_cup_1 = _rect_mask(H, W, 12, 16, 30, 38)
        m_peg_1 = _rect_mask(H, W, 50, 14, 62, 46)
        m_blk_1 = _rect_mask(H, W, 88, 20, 104, 52)
        _paint_masked(img1, m_cup_1, (220, 40, 40))
        _paint_masked(img1, m_peg_1, (35, 90, 230))
        _paint_masked(img1, m_blk_1, (235, 195, 25))

        r1 = mgr.update_tracks(
            session_id="ep_track",
            img_rgb=img1,
            image_sha256="sha_f1",
            candidates=[
                dict(mask=m_cup_1, score=0.95, labels=[["cup", 0.95]]),
                dict(mask=m_peg_1, score=0.88, labels=[["peg", 0.88]]),
                dict(mask=m_blk_1, score=0.79, labels=[["block", 0.79]]),
            ],
            now=100.0,
        )
        self.assertEqual([x["track_id"] for x in r1["instances"]], ["o1", "o2", "o3"])
        self.assertEqual([x["track_status"] for x in r1["instances"]], ["new", "new", "new"])

        # 第 2 帧：o3 原地倾倒（由竖直长条变成水平长条）且单帧检测分跃升至最高 0.98，o1 分数跌至最低 0.58
        img2 = np.full((H, W, 3), 180, dtype=np.uint8)
        m_cup_2 = _rect_mask(H, W, 14, 17, 32, 39)
        m_peg_2 = _rect_mask(H, W, 51, 15, 63, 47)
        m_blk_2 = _rect_mask(H, W, 80, 32, 114, 46)  # 90 度倾倒侧立
        _paint_masked(img2, m_cup_2, (220, 40, 40))
        _paint_masked(img2, m_peg_2, (35, 90, 230))
        _paint_masked(img2, m_blk_2, (235, 195, 25))

        r2 = mgr.update_tracks(
            session_id="ep_track",
            img_rgb=img2,
            image_sha256="sha_f2",
            candidates=[
                dict(mask=m_blk_2, score=0.98, labels=[["block", 0.98]]),
                dict(mask=m_peg_2, score=0.86, labels=[["peg", 0.86]]),
                dict(mask=m_cup_2, score=0.58, labels=[["cup", 0.58]]),
            ],
            now=101.0,
        )
        by_id_2 = {x["track_id"]: x for x in r2["instances"]}
        self.assertEqual(list(by_id_2.keys()), ["o1", "o2", "o3"])
        self.assertEqual([by_id_2[k]["track_status"] for k in ("o1", "o2", "o3")], ["tracked", "tracked", "tracked"])
        self.assertGreater(vsm.mask_iou(by_id_2["o3"]["mask"], m_blk_2), 0.95)
        self.assertGreater(vsm.mask_iou(by_id_2["o1"]["mask"], m_cup_2), 0.95)

        # 第 3 帧：o2 被夹爪横跨全桌面搬运至右下角（零 IoU 重叠），同时新入场绿色物体应分配递增 o4
        img3 = np.full((H, W, 3), 180, dtype=np.uint8)
        m_peg_moved = _rect_mask(H, W, 96, 66, 108, 92)
        m_green_new = _rect_mask(H, W, 12, 64, 30, 86)
        _paint_masked(img3, m_cup_2, (220, 40, 40))
        _paint_masked(img3, m_peg_moved, (35, 90, 230))
        _paint_masked(img3, m_blk_2, (235, 195, 25))
        _paint_masked(img3, m_green_new, (30, 190, 75))

        r3 = mgr.update_tracks(
            session_id="ep_track",
            img_rgb=img3,
            image_sha256="sha_f3",
            candidates=[
                dict(mask=m_green_new, score=0.92, labels=[]),
                dict(mask=m_peg_moved, score=0.90, labels=[["peg", 0.90]]),
                dict(mask=m_blk_2, score=0.85, labels=[["block", 0.85]]),
                dict(mask=m_cup_2, score=0.80, labels=[["cup", 0.80]]),
            ],
            now=102.0,
        )
        by_id_3 = {x["track_id"]: x for x in r3["instances"]}
        self.assertEqual(list(by_id_3.keys()), ["o1", "o2", "o3", "o4"])
        self.assertGreater(vsm.mask_iou(by_id_3["o2"]["mask"], m_peg_moved), 0.95)
        self.assertEqual(by_id_3["o2"]["track_status"], "tracked")
        self.assertGreater(vsm.mask_iou(by_id_3["o4"]["mask"], m_green_new), 0.95)
        self.assertEqual(by_id_3["o4"]["track_status"], "new")

    def test_predict_inst_propagation_and_explicit_reanchor(self) -> None:
        mgr = vsm.VideoSessionManager()
        H, W = 80, 100
        img1 = np.full((H, W, 3), 170, dtype=np.uint8)
        m1 = _rect_mask(H, W, 10, 12, 28, 32)
        m2 = _rect_mask(H, W, 52, 20, 72, 44)
        _paint_masked(img1, m1, (210, 50, 60))
        _paint_masked(img1, m2, (40, 120, 220))

        mgr.update_tracks(
            session_id="ep_prop",
            img_rgb=img1,
            image_sha256="p1",
            candidates=[dict(mask=m1, score=0.91, labels=[]), dict(mask=m2, score=0.87, labels=[])],
        )

        # 第 2 帧：文本检测器漏检 o2，由 predict_fn（模拟 MODEL.predict_inst）根据掩码内点传播找回 o2
        img2 = np.full((H, W, 3), 170, dtype=np.uint8)
        m2_shift = _rect_mask(H, W, 54, 22, 74, 46)
        _paint_masked(img2, m1, (210, 50, 60))
        _paint_masked(img2, m2_shift, (40, 120, 220))

        def _fake_predict(point_coords, point_labels, box=None, mask_input=None):
            u, v = float(point_coords[0, 0]), float(point_coords[0, 1])
            if 50 <= u <= 76 and 18 <= v <= 48:
                return m2_shift[None, :, :], np.array([0.89], dtype=np.float64), np.ones((1, 256, 256), dtype=np.float32)
            return np.zeros((1, H, W), dtype=bool), np.array([0.1], dtype=np.float64), None

        r2 = mgr.update_tracks(
            session_id="ep_prop",
            img_rgb=img2,
            image_sha256="p2",
            candidates=[dict(mask=m1, score=0.90, labels=[])],
            predict_fn=_fake_predict,
        )
        self.assertEqual([x["track_id"] for x in r2["instances"]], ["o1", "o2"])
        self.assertEqual(r2["instances"][1]["track_status"], "tracked")

        # 第 3 帧：显式 reanchor 指定 target_track_id="o2" 与正负像素提示
        m2_reanchor = _rect_mask(H, W, 58, 24, 78, 48)
        img3 = np.full((H, W, 3), 170, dtype=np.uint8)
        _paint_masked(img3, m1, (210, 50, 60))
        _paint_masked(img3, m2_reanchor, (40, 120, 220))

        def _reanchor_predict(point_coords, point_labels, box=None, mask_input=None):
            self.assertEqual(list(point_labels), [1, 0])
            return m2_reanchor[None, :, :], np.array([0.93], dtype=np.float64), None

        r3 = mgr.update_tracks(
            session_id="ep_prop",
            img_rgb=img3,
            image_sha256="p3",
            candidates=[dict(mask=m1, score=0.90, labels=[])],
            reanchor={"target_track_id": "o2", "pos_pixels": [[65, 35]], "neg_pixels": [[15, 20]]},
            predict_fn=_reanchor_predict,
        )
        by_id_3 = {x["track_id"]: x for x in r3["instances"]}
        self.assertEqual(by_id_3["o2"]["track_status"], "reanchored")
        self.assertGreater(vsm.mask_iou(by_id_3["o2"]["mask"], m2_reanchor), 0.99)

        # reset=True 重置整个会话
        r4 = mgr.update_tracks(
            session_id="ep_prop",
            img_rgb=img3,
            image_sha256="p4",
            candidates=[dict(mask=m2_reanchor, score=0.92, labels=[])],
            reanchor={"reset": True},
        )
        self.assertEqual([x["track_id"] for x in r4["instances"]], ["o1"])

    def test_16_concurrency_vram_guard_lru_ttl_and_keyframe_cap(self) -> None:
        mgr = vsm.VideoSessionManager()
        H, W = 48, 64
        img = np.full((H, W, 3), 150, dtype=np.uint8)
        m = _rect_mask(H, W, 10, 10, 24, 24)
        _paint_masked(img, m, (200, 80, 40))

        # 单会话写入 10 帧，滑窗最多保留 6 个关键帧摘要
        for f_idx in range(10):
            res = mgr.update_tracks(
                session_id="s_window",
                img_rgb=img,
                image_sha256=f"win_{f_idx}",
                candidates=[dict(mask=m, score=0.9, labels=[])],
                hot_gpu_state={"sem": _FakeTensor(np.ones((4, 4)), "cuda")},
                now=1000.0 + f_idx,
            )
        self.assertEqual(res["keyframes_kept"], vsm.MAX_KEYFRAMES)
        self.assertEqual(vsm.MAX_KEYFRAMES, 6)

        # 16 并发会话同时打入 GPU 张量：gpu_hot_sessions 严格受 MAX_GPU_HOT_SESSIONS 上限约束，其余全部已卸载至 CPU
        for i in range(16):
            mgr.update_tracks(
                session_id=f"conc_{i}",
                img_rgb=img,
                image_sha256=f"c_{i}",
                candidates=[dict(mask=m, score=0.88, labels=[])],
                hot_gpu_state={"sem": _FakeTensor(np.ones((8, 8)), "cuda")},
                now=2000.0 + i,
            )
        st_16 = mgr.stats(now=2020.0)
        self.assertLessEqual(st_16["gpu_hot_sessions"], vsm.MAX_GPU_HOT_SESSIONS)
        gpu_holders = [
            sid for sid, s_obj in mgr._sessions.items() if vsm.has_any_gpu_tensor(s_obj.hot_gpu_state)
        ]
        self.assertLessEqual(len(gpu_holders), vsm.MAX_GPU_HOT_SESSIONS)

        # 写入至 28 个会话：严格受 MAX_SESSIONS=24 LRU 淘汰约束
        for i in range(16, 28):
            mgr.update_tracks(
                session_id=f"conc_{i}",
                img_rgb=img,
                image_sha256=f"c_{i}",
                candidates=[dict(mask=m, score=0.88, labels=[])],
                now=2030.0 + i,
            )
        self.assertEqual(mgr.stats(now=2060.0)["active_sessions"], 24)
        self.assertIsNone(mgr.get_session_snapshot("s_window"))

        # close_session 立即释放指定热会话
        self.assertTrue(mgr.close_session("conc_27"))
        self.assertEqual(mgr.stats(now=2060.0)["active_sessions"], 23)

        # 推进时间超过 SESSION_TTL_S=900 自动清理全部过期会话
        self.assertEqual(mgr.stats(now=2060.0 + vsm.SESSION_TTL_S + 5.0)["active_sessions"], 0)

    def test_update_tracks_atomic_rollback_on_mid_frame_predict_exception(self) -> None:
        mgr = vsm.VideoSessionManager()
        H, W = 64, 80
        img1 = np.full((H, W, 3), 170, dtype=np.uint8)
        m1 = _rect_mask(H, W, 8, 10, 24, 28)
        m2 = _rect_mask(H, W, 44, 16, 62, 38)
        _paint_masked(img1, m1, (215, 45, 55))
        _paint_masked(img1, m2, (40, 115, 225))

        mgr.update_tracks(
            session_id="ep_atomic",
            img_rgb=img1,
            image_sha256="sha_1",
            candidates=[dict(mask=m1, score=0.92, labels=[["cup", 0.92]]), dict(mask=m2, score=0.88, labels=[["peg", 0.88]])],
            now=10.0,
        )
        snap_before = mgr.get_session_snapshot("ep_atomic")
        self.assertEqual(snap_before["frame_seq"], 1)
        self.assertEqual(snap_before["next_track_num"], 3)
        self.assertEqual(snap_before["active_track_ids"], ["o1", "o2"])

        img2 = np.full((H, W, 3), 170, dtype=np.uint8)
        m2_shift = _rect_mask(H, W, 46, 18, 64, 40)
        _paint_masked(img2, m1, (215, 45, 55))
        _paint_masked(img2, m2_shift, (40, 115, 225))
        m_new = _rect_mask(H, W, 8, 42, 24, 58)
        _paint_masked(img2, m_new, (35, 195, 70))

        def _oom_predict(*_a, **_kw):
            raise RuntimeError("CUDA out of memory")

        with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
            mgr.update_tracks(
                session_id="ep_atomic",
                img_rgb=img2,
                image_sha256="sha_2",
                candidates=[dict(mask=m1, score=0.90, labels=[["cup", 0.90]]), dict(mask=m_new, score=0.85, labels=[])],
                predict_fn=_oom_predict,
                now=11.0,
            )

        # 中途 predict_fn 抛错后，会话帧序号、递增 ID 分配器、轨迹表与关键帧必须 100% 保持第 1 帧状态
        snap_after_err = mgr.get_session_snapshot("ep_atomic")
        self.assertEqual(snap_after_err, snap_before)
        st_obj = mgr._sessions["ep_atomic"]
        self.assertEqual(st_obj.last_image_sha256, "sha_1")
        self.assertEqual(st_obj.tracks[2].missed_frames, 0)

        # 恢复正常后重试同一帧：平滑推进至 frame_seq=2，o2 被找回，新物体分配 o3（无跳号）
        def _ok_predict(*_a, **_kw):
            return m2_shift[None, :, :], np.array([0.89], dtype=np.float64), None

        r_retry = mgr.update_tracks(
            session_id="ep_atomic",
            img_rgb=img2,
            image_sha256="sha_2",
            candidates=[dict(mask=m1, score=0.90, labels=[["cup", 0.90]]), dict(mask=m_new, score=0.85, labels=[])],
            predict_fn=_ok_predict,
            now=12.0,
        )
        self.assertEqual(r_retry["frame_seq"], 2)
        self.assertEqual([x["track_id"] for x in r_retry["instances"]], ["o1", "o2", "o3"])


class TestMultiviewClusterAndGeometry(unittest.TestCase):
    """验证 1cm 体素 26 邻域连通域速度/精度、凹腔深度与 3 层凸包切片。"""

    def test_voxel_connected_cluster_accuracy_and_speed_under_2ms(self) -> None:
        rng = np.random.default_rng(7)
        main_pts = rng.normal(loc=[0.35, -0.10, 0.80], scale=0.008, size=(3200, 3))
        sec_pts = rng.normal(loc=[0.55, 0.15, 0.85], scale=0.006, size=(600, 3))
        outliers = rng.uniform(low=[-0.2, -0.4, 0.76], high=[0.9, 0.4, 1.25], size=(200, 3))
        P = np.vstack([main_pts, sec_pts, outliers])

        # 预热一次后统计 25 次中位耗时
        P_main, c_frac, labels = mc.voxel_connected_cluster(P, voxel_m=0.01, min_samples=5)
        timings_ms = []
        for _ in range(25):
            t0 = time.perf_counter()
            mc.voxel_connected_cluster(P, voxel_m=0.01, min_samples=5)
            timings_ms.append((time.perf_counter() - t0) * 1000.0)

        median_ms = float(np.median(timings_ms))
        self.assertLess(median_ms, 2.0, f"体素连通域耗时 {median_ms:.3f}ms 超出 2ms 红线")
        self.assertGreater(c_frac, 0.75)
        self.assertLess(c_frac, 0.84)
        self.assertLess(np.linalg.norm(P_main.mean(axis=0) - np.array([0.35, -0.10, 0.80])), 0.005)
        # 稀疏离群噪声绝大部分应标记为 -1
        self.assertGreater(int((labels[3800:] == -1).sum()), 170)

    def test_cavity_depth_and_surface_slices(self) -> None:
        H, W = 64, 64
        X = np.zeros((H, W, 3), dtype=np.float32)
        uu, vv = np.meshgrid(np.linspace(-0.1, 0.1, W), np.linspace(-0.1, 0.1, H))
        X[..., 0] = uu
        X[..., 1] = vv
        X[..., 2] = 0.76  # 桌面高度
        conf = np.full((H, W), 0.9, dtype=np.float32)

        # 1) 中空收纳盒：外沿高度 0.84m，内腔底面紧贴桌面 0.762m（未按 support_z+4mm 截断！）
        box_mask = _rect_mask(H, W, 14, 14, 50, 50)
        cavity_mask = _rect_mask(H, W, 21, 21, 43, 43)
        X[box_mask & ~cavity_mask, 2] = 0.84
        X[cavity_mask, 2] = 0.762
        cav_box = mc.compute_cavity_depth(X, conf, conf_p10=0.2, mask=box_mask)
        self.assertAlmostEqual(cav_box, 0.078, delta=0.01)

        # 2) 实心平顶积木：全表面 0.81m -> 凹腔深度接近 0
        X_flat = X.copy()
        X_flat[box_mask, 2] = 0.81
        self.assertLessEqual(mc.compute_cavity_depth(X_flat, conf, conf_p10=0.2, mask=box_mask), 0.002)

        # 3) 有效点不足（掩码极小） -> 返回 0.0
        tiny_mask = _rect_mask(H, W, 2, 2, 5, 5)
        self.assertEqual(mc.compute_cavity_depth(X, conf, conf_p10=0.2, mask=tiny_mask), 0.0)

        # 4) 三维圆台点集的最多 3 层二维凸包切片，每层顶点数严格 <= 12
        angles = np.linspace(0.0, 2.0 * np.pi, 60, endpoint=False)
        ring_layers = []
        for z_val, rad in ((0.77, 0.05), (0.81, 0.04), (0.85, 0.03)):
            ring_layers.append(np.column_stack([rad * np.cos(angles), rad * np.sin(angles), np.full_like(angles, z_val)]))
        pts_3d = np.vstack(ring_layers)
        slices = mc.compute_surface_slices(pts_3d, max_slices=3, max_vertices=12)
        self.assertEqual(len(slices), 3)
        for sl in slices:
            self.assertIn("z_m", sl)
            self.assertGreaterEqual(len(sl["hull_xy_m"]), 3)
            self.assertLessEqual(len(sl["hull_xy_m"]), 12)


class _FakeSam31Runtime:
    """在本地临时测试服务中确定性识别红/蓝/黄颜色连通域并模拟 SAM 3.1 处理器与交互预测器（强制校验 288×288 掩码尺寸）。"""

    def __init__(self) -> None:
        from types import SimpleNamespace

        self.score_overrides: dict[str, float] = {}
        self.drop_text_names: set[str] = set()
        self.predict_inst_mask_shapes: list[tuple[int, ...] | None] = []
        self.inst_interactive_predictor = SimpleNamespace(
            model=SimpleNamespace(sam_prompt_encoder=SimpleNamespace(mask_input_size=(288, 288)))
        )

    def set_image(self, img: Image.Image) -> dict[str, Any]:
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        return {"backbone_out": {"rgb": arr, "backbone_fpn": [np.zeros((1, 8, 72, 72), dtype=np.float32)]}}

    def _detect_color_blobs(self, rgb: np.ndarray) -> list[tuple[str, np.ndarray, float]]:
        r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
        specs = [
            ("box", (r > 180) & (g < 80) & (b < 80), self.score_overrides.get("box", 0.93)),
            ("peg", (b > 180) & (r < 80) & (g < 120), self.score_overrides.get("peg", 0.86)),
        ]
        out = []
        for name, m, sc in specs:
            if int(m.sum()) >= 16:
                out.append((name, m, sc))
        return out

    def set_text_prompt(self, prompt: str, state: dict[str, Any]) -> dict[str, Any]:
        rgb = state["backbone_out"]["rgb"]
        H, W = rgb.shape[:2]
        blobs = self._detect_color_blobs(rgb)
        masks, boxes, scores = [], [], []
        for name, m, sc in blobs:
            if name in self.drop_text_names:
                continue
            if prompt in ("object", name):
                ys, xs = np.nonzero(m)
                masks.append(m[None, :, :])
                boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
                scores.append(sc)
        st = dict(state)
        st["masks"] = np.stack(masks, axis=0) if masks else np.zeros((0, 1, H, W), dtype=bool)
        st["boxes"] = np.asarray(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        st["scores"] = np.asarray(scores, dtype=np.float32) if scores else np.zeros((0,), dtype=np.float32)
        return st

    def predict_inst(
        self,
        state: dict[str, Any],
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        box: np.ndarray | None = None,
        mask_input: np.ndarray | None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        exp_h, exp_w = self.inst_interactive_predictor.model.sam_prompt_encoder.mask_input_size
        if mask_input is not None:
            self.predict_inst_mask_shapes.append(tuple(mask_input.shape))
            if tuple(mask_input.shape) != (1, exp_h, exp_w):
                raise RuntimeError(
                    f"The size of tensor a ({exp_h // 4}) must match the size of tensor b ({mask_input.shape[-1] // 4}) at non-singleton dimension 3"
                )
        else:
            self.predict_inst_mask_shapes.append(None)
        rgb = state["backbone_out"]["rgb"]
        H, W = rgb.shape[:2]
        u, v = int(np.clip(point_coords[0, 0], 0, W - 1)), int(np.clip(point_coords[0, 1], 0, H - 1))
        chosen = np.zeros((H, W), dtype=bool)
        for _, m, _ in self._detect_color_blobs(rgb):
            if m[v, u]:
                chosen = m
                break
        if not chosen.any():
            chosen[max(0, v - 5) : min(H, v + 6), max(0, u - 5) : min(W, u + 6)] = True
        lgt = np.where(chosen, 4.0, -4.0).astype(np.float32)[None, :, :]
        out_m = lgt if return_logits else (lgt > 0)
        low_res = cv2.resize(chosen.astype(np.float32) * 8.0 - 4.0, (exp_w, exp_h), interpolation=cv2.INTER_LINEAR)[None, :, :]
        return out_m, np.array([0.91], dtype=np.float32), low_res.astype(np.float32)


class TestEndToEndServicesHTTP(unittest.TestCase):
    """在本地 127.0.0.1 随机空闲端口启动两个真实 HTTP Handler，验证全链路接口契约。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.fake_sam = _FakeSam31Runtime()
        sam31_svc.MODEL = cls.fake_sam
        sam31_svc.PROC = cls.fake_sam
        sam31_svc.CACHE.clear()
        sam31_svc.SESSION_MANAGER = vsm.VideoSessionManager()

        cls.sam_srv = ThreadingHTTPServer(("127.0.0.1", 0), sam31_svc.H)
        cls.sam_srv.daemon_threads = True
        cls.sam_port = cls.sam_srv.server_address[1]
        threading.Thread(target=cls.sam_srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

        vggt_svc.args.sam31 = f"http://127.0.0.1:{cls.sam_port}"
        vggt_svc.CACHE.clear()
        vggt_svc.MULTIVIEW_CACHE.clear()
        vggt_svc.SESSION_CACHE_KEYS.clear()
        vggt_svc.SESSION_MAX_TRACK_NUM.clear()
        vggt_svc.SESSION_KNOWN_TRACKS.clear()

        cls.vggt_calls: list[tuple[int, ...]] = []
        cls.next_vggt_outputs: list[tuple[np.ndarray, np.ndarray]] = []

        def _fake_vggt_forward(ims: Any) -> dict[str, np.ndarray]:
            arr = np.asarray(ims, dtype=np.float32)
            cls.vggt_calls.append(tuple(arr.shape))
            s_cnt, _, h, w = arr.shape
            depths, confs = [], []
            for idx in range(s_cnt):
                d_i, c_i = cls.next_vggt_outputs.pop(0)
                assert d_i.shape == (h, w)
                depths.append(d_i)
                confs.append(c_i)
            return {
                "depth": np.stack(depths, axis=0)[None, ..., None],
                "depth_conf": np.stack(confs, axis=0)[None, ...],
            }

        vggt_svc.VGGT = _fake_vggt_forward
        cls.vggt_srv = ThreadingHTTPServer(("127.0.0.1", 0), vggt_svc.H)
        cls.vggt_srv.daemon_threads = True
        cls.vggt_port = cls.vggt_srv.server_address[1]
        threading.Thread(target=cls.vggt_srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.vggt_srv.shutdown()
        cls.vggt_srv.server_close()
        cls.sam_srv.shutdown()
        cls.sam_srv.server_close()

    def setUp(self) -> None:
        self.fake_sam.score_overrides.clear()
        self.fake_sam.drop_text_names.clear()
        self.fake_sam.predict_inst_mask_shapes.clear()
        self.fake_sam.inst_interactive_predictor.model.sam_prompt_encoder.mask_input_size = (288, 288)
        sam31_svc.CACHE.clear()
        sam31_svc.STATS["errors"] = 0
        sam31_svc.SESSION_MANAGER = vsm.VideoSessionManager()
        vggt_svc.CACHE.clear()
        vggt_svc.MULTIVIEW_CACHE.clear()
        vggt_svc.SESSION_CACHE_KEYS.clear()
        vggt_svc.SESSION_MAX_TRACK_NUM.clear()
        vggt_svc.SESSION_KNOWN_TRACKS.clear()
        vggt_svc.args.sam31 = f"http://127.0.0.1:{self.sam_port}"
        self.vggt_calls.clear()
        self.next_vggt_outputs.clear()

    def _build_main_scene(self, peg_toppled: bool = False) -> tuple[str, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        H, W = 96, 128
        K = np.array([[140.0, 0.0, 64.0], [0.0, 140.0, 48.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        T = _look_at_camera([0.40, 0.0, 1.26], [0.40, 0.0, 0.76], up_hint=[0.0, -1.0, 0.0])
        support_z = 0.76

        # 盒体 o1（带 6cm 真实凹腔）与柱体 o2
        box_3d = dict(x_range=(0.24, 0.34), y_range=(-0.06, 0.04), top_z=0.83, floor_z=0.765, wall_m=0.015)
        if not peg_toppled:
            peg_3d = dict(x_range=(0.46, 0.51), y_range=(-0.03, 0.03), top_z=0.85, floor_z=None)
        else:
            peg_3d = dict(x_range=(0.44, 0.54), y_range=(-0.02, 0.02), top_z=0.80, floor_z=None)

        depth, conf = _render_synthetic_depth(H, W, K, T, support_z, [box_3d, peg_3d], scale_factor=2.0)

        # 用精确投影生成与三维盒体严格对齐的 RGB 图像供 SAM3.1 分割
        uu, vv = np.meshgrid(np.arange(W, dtype=np.float64) + 0.5, np.arange(H, dtype=np.float64) + 0.5)
        dirs_cam = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], axis=-1)
        pts_w = T[:3, 3] + (depth * 2.0)[..., None] * (dirs_cam @ T[:3, :3].T)

        img = np.full((H, W, 3), 175, dtype=np.uint8)
        m_box = (
            (pts_w[..., 0] >= box_3d["x_range"][0])
            & (pts_w[..., 0] <= box_3d["x_range"][1])
            & (pts_w[..., 1] >= box_3d["y_range"][0])
            & (pts_w[..., 1] <= box_3d["y_range"][1])
        )
        m_peg = (
            (pts_w[..., 0] >= peg_3d["x_range"][0])
            & (pts_w[..., 0] <= peg_3d["x_range"][1])
            & (pts_w[..., 1] >= peg_3d["y_range"][0])
            & (pts_w[..., 1] <= peg_3d["y_range"][1])
        )
        _paint_masked(img, m_box, (225, 45, 45))
        _paint_masked(img, m_peg, (35, 75, 225))
        return _encode_png_b64(img), K, T, support_z, depth, conf

    def test_sam31_segment_backward_compat_and_health_metrics(self) -> None:
        img_b64, _, _, _, _, _ = self._build_main_scene()
        code, seg = _http_json(
            f"http://127.0.0.1:{self.sam_port}/segment",
            {"image_b64": img_b64, "phrases": ["box", "peg"], "threshold": 0.4},
        )
        self.assertEqual(code, 200)
        self.assertEqual(seg["status"], "ok")
        self.assertEqual([p["phrase"] for p in seg["phrases"]], ["box", "peg"])
        self.assertEqual(len(seg["phrases"][0]["instances"]), 1)

        code_h, hlth = _http_json(f"http://127.0.0.1:{self.sam_port}/health", method="GET")
        self.assertEqual(code_h, 200)
        self.assertEqual(hlth["active_sessions"], 0)
        self.assertEqual(hlth["gpu_hot_sessions"], 0)

    def test_object_table_persistent_tracking_cavity_points_obj_id_and_close(self) -> None:
        img1_b64, K, T, support_z, d1, c1 = self._build_main_scene(peg_toppled=False)
        self.next_vggt_outputs.append((d1, c1))

        code1, ot1 = _http_json(
            f"http://127.0.0.1:{self.vggt_port}/object_table",
            {
                "session_id": "ep_live_1",
                "image_b64": img1_b64,
                "K": K.tolist(),
                "T_world_camera": T.tolist(),
                "support_z_m": support_z,
                "labels": ["box", "peg"],
            },
        )
        self.assertEqual(code1, 200)
        self.assertTrue(ot1["tracking"]["persistent_ids"])
        self.assertEqual(ot1["tracking"]["session_id"], "ep_live_1")
        self.assertEqual(ot1["tracking"]["active_tracks"], 2)
        self.assertEqual([o["track_id"] for o in ot1["objects"]], ["o1", "o2"])
        self.assertEqual([o["track_status"] for o in ot1["objects"]], ["new", "new"])

        box_obj, peg_obj = ot1["objects"][0], ot1["objects"][1]
        self.assertGreater(box_obj["cavity_depth_m"], 0.045)
        self.assertLess(peg_obj["cavity_depth_m"], 0.01)
        self.assertGreaterEqual(len(box_obj["surface_slices_m"]), 1)
        self.assertLessEqual(len(box_obj["surface_slices_m"]), 3)

        # 紧随其后的同图 /points 必须直接复用缓存 entry["instance_masks"] 写出 obj_id 与 view_mask
        code_p, pts_ans = _http_json(
            f"http://127.0.0.1:{self.vggt_port}/points",
            {
                "image_b64": img1_b64,
                "K": K.tolist(),
                "T_world_camera": T.tolist(),
                "support_z_m": support_z,
                "voxel_m": 0.01,
            },
        )
        self.assertEqual(code_p, 200)
        self.assertTrue(pts_ans["cache_hit"])
        self.assertEqual(pts_ans["fused_views"], [])
        self.assertIsNone(pts_ans["multiview_residual_m"])

        npz = np.load(io.BytesIO(base64.b64decode(pts_ans["npz_b64"])), allow_pickle=False)
        for req_key in ("xyz", "conf", "uv", "count", "robot", "support_z", "voxel_m", "image_size", "obj_id", "view_mask"):
            self.assertIn(req_key, npz.files)
        self.assertEqual(npz["obj_id"].dtype, np.int16)
        self.assertEqual(npz["view_mask"].dtype, np.uint8)
        self.assertEqual(len(npz["obj_id"]), len(npz["xyz"]))
        self.assertTrue(np.all(npz["view_mask"] == 1))
        self.assertTrue(np.any(npz["obj_id"] == 1))
        self.assertTrue(np.any(npz["obj_id"] == 2))
        self.assertTrue(np.any(npz["obj_id"] == 0))

        # 第 2 帧：peg (o2) 倾倒且置信度反转超过 box (o1)，身份必须严格保持 o1, o2 不串号
        self.fake_sam.score_overrides = {"peg": 0.99, "box": 0.62}
        img2_b64, _, _, _, d2, c2 = self._build_main_scene(peg_toppled=True)
        self.next_vggt_outputs.append((d2, c2))
        code2, ot2 = _http_json(
            f"http://127.0.0.1:{self.vggt_port}/object_table",
            {
                "session_id": "ep_live_1",
                "image_b64": img2_b64,
                "K": K.tolist(),
                "T_world_camera": T.tolist(),
                "support_z_m": support_z,
                "labels": ["box", "peg"],
            },
        )
        self.assertEqual(code2, 200)
        self.assertEqual([o["track_id"] for o in ot2["objects"]], ["o1", "o2"])
        self.assertEqual([o["label"] for o in ot2["objects"]], ["box", "peg"])
        self.assertEqual([o["track_status"] for o in ot2["objects"]], ["tracked", "tracked"])

        # 级联关闭会话：:8331 清理本地掩码并转发至 :8333/session_close
        code_c, close_ans = _http_json(
            f"http://127.0.0.1:{self.vggt_port}/session_close",
            {"session_id": "ep_live_1"},
        )
        self.assertEqual(code_c, 200)
        self.assertEqual(close_ans["status"], "ok")
        self.assertTrue(close_ans["closed"])
        _, sam_hlth = _http_json(f"http://127.0.0.1:{self.sam_port}/health", method="GET")
        self.assertEqual(sam_hlth["active_sessions"], 0)

    def test_multiview_points_batch_forward_and_strict_residual_gating(self) -> None:
        img_main_b64, K_main, T_main, support_z, d_main, c_main = self._build_main_scene()
        H, W = d_main.shape

        # 构造距桌面目标 ~0.28m 的腕部视角相机（重叠区一致，尺度因子 3.0 由服务端自动求解）
        T_wrist = _look_at_camera([0.22, -0.18, 0.96], [0.34, 0.0, 0.78])
        K_wrist = K_main.copy()
        boxes = [
            dict(x_range=(0.24, 0.34), y_range=(-0.06, 0.04), top_z=0.83, floor_z=0.765, wall_m=0.015),
            dict(x_range=(0.46, 0.51), y_range=(-0.03, 0.03), top_z=0.85, floor_z=None),
        ]
        d_wrist, c_wrist = _render_synthetic_depth(H, W, K_wrist, T_wrist, support_z, boxes, scale_factor=3.0)
        wrist_b64 = _encode_png_b64(np.full((H, W, 3), 160, dtype=np.uint8))

        # 1) 正例：单次 [2, 3, H, W] Batch 前向，残差接近 0 <= 0.025m，且严格裁剪 <= 0.40m 近场点
        self.vggt_calls.clear()
        self.next_vggt_outputs.extend([(d_main, c_main), (d_wrist, c_wrist)])
        code_ok, ans_ok = _http_json(
            f"http://127.0.0.1:{self.vggt_port}/points",
            {
                "image_b64": img_main_b64,
                "K": K_main.tolist(),
                "T_world_camera": T_main.tolist(),
                "support_z_m": support_z,
                "voxel_m": 0.01,
                "extra_views": [
                    {
                        "camera_id": "cam_left_wrist",
                        "image_b64": wrist_b64,
                        "K": K_wrist.tolist(),
                        "T_world_camera": T_wrist.tolist(),
                        "max_range_m": 0.40,
                    }
                ],
            },
        )
        self.assertEqual(code_ok, 200)
        self.assertEqual(self.vggt_calls, [(2, 3, H, W)])
        self.assertEqual(ans_ok["fused_views"], ["cam_left_wrist"])
        self.assertIsNotNone(ans_ok["multiview_residual_m"])
        self.assertLessEqual(ans_ok["multiview_residual_m"], 0.025)

        npz_ok = np.load(io.BytesIO(base64.b64decode(ans_ok["npz_b64"])), allow_pickle=False)
        wrist_pts = npz_ok["xyz"][(npz_ok["view_mask"] & 2) != 0]
        self.assertGreater(len(wrist_pts), 10)
        self.assertTrue(np.any(npz_ok["view_mask"] == 3))
        # 验证所有仅来自腕部视角的点距腕部相机光心严格 <= 0.40m + 体素量化误差
        only_wrist = npz_ok["xyz"][npz_ok["view_mask"] == 2]
        if len(only_wrist):
            dists = np.linalg.norm(only_wrist - T_wrist[:3, 3], axis=1)
            self.assertLessEqual(float(dists.max()), 0.40 + 1e-4)

        # 1b) 相同主图 + 相同腕部 extra_views 重复请求必须直接命中 MULTIVIEW_CACHE，零额外 VGGT 前向
        code_hit, ans_hit = _http_json(
            f"http://127.0.0.1:{self.vggt_port}/points",
            {
                "image_b64": img_main_b64,
                "K": K_main.tolist(),
                "T_world_camera": T_main.tolist(),
                "support_z_m": support_z,
                "voxel_m": 0.01,
                "extra_views": [
                    {
                        "camera_id": "cam_left_wrist",
                        "image_b64": wrist_b64,
                        "K": K_wrist.tolist(),
                        "T_world_camera": T_wrist.tolist(),
                        "max_range_m": 0.40,
                    }
                ],
            },
        )
        self.assertEqual(code_hit, 200)
        self.assertTrue(ans_hit["cache_hit"])
        self.assertEqual(self.vggt_calls, [(2, 3, H, W)])
        self.assertEqual(ans_hit["fused_views"], ["cam_left_wrist"])

        # 2) 反例 A：新腕部图像深度扭曲导致重叠区残差中位数 > 0.025m -> 严禁融合污染主点云
        rng = np.random.default_rng(99)
        d_bad = d_wrist * rng.uniform(0.55, 1.55, size=d_wrist.shape).astype(np.float32)
        wrist_bad_b64 = _encode_png_b64(np.full((H, W, 3), 161, dtype=np.uint8))
        self.next_vggt_outputs.extend([(d_main, c_main), (d_bad, c_wrist)])
        code_bad, ans_bad = _http_json(
            f"http://127.0.0.1:{self.vggt_port}/points",
            {
                "image_b64": img_main_b64,
                "K": K_main.tolist(),
                "T_world_camera": T_main.tolist(),
                "support_z_m": support_z,
                "voxel_m": 0.01,
                "extra_views": [
                    {
                        "camera_id": "cam_left_wrist",
                        "image_b64": wrist_bad_b64,
                        "K": K_wrist.tolist(),
                        "T_world_camera": T_wrist.tolist(),
                    }
                ],
            },
        )
        self.assertEqual(code_bad, 200)
        self.assertEqual(ans_bad["fused_views"], [])
        self.assertGreater(ans_bad["multiview_residual_m"], 0.025)
        npz_bad = np.load(io.BytesIO(base64.b64decode(ans_bad["npz_b64"])), allow_pickle=False)
        self.assertTrue(np.all(npz_bad["view_mask"] == 1))

        # 3) 反例 B：附加视角朝向远离工作台无重叠 -> fused_views=[], multiview_residual_m=None
        T_away = _look_at_camera([5.0, 5.0, 1.5], [6.0, 6.0, 1.5])
        self.next_vggt_outputs.extend([(d_main, c_main), (d_wrist, c_wrist)])
        code_no_ov, ans_no_ov = _http_json(
            f"http://127.0.0.1:{self.vggt_port}/points",
            {
                "image_b64": img_main_b64,
                "K": K_main.tolist(),
                "T_world_camera": T_main.tolist(),
                "support_z_m": support_z,
                "voxel_m": 0.01,
                "extra_views": [
                    {
                        "camera_id": "cam_right_wrist",
                        "image_b64": wrist_b64,
                        "K": K_wrist.tolist(),
                        "T_world_camera": T_away.tolist(),
                    }
                ],
            },
        )
        self.assertEqual(code_no_ov, 200)
        self.assertEqual(ans_no_ov["fused_views"], [])
        self.assertIsNone(ans_no_ov["multiview_residual_m"])

    def test_stateless_compat_and_track_endpoint_fallback(self) -> None:
        img_b64, K, T, support_z, d, c = self._build_main_scene()
        self.next_vggt_outputs.append((d, c))

        # 不传 session_id：走原有 /segment 路径，状态为 stateless
        code, ot_stateless = _http_json(
            f"http://127.0.0.1:{self.vggt_port}/object_table",
            {
                "image_b64": img_b64,
                "K": K.tolist(),
                "T_world_camera": T.tolist(),
                "support_z_m": support_z,
            },
        )
        self.assertEqual(code, 200)
        self.assertFalse(ot_stateless["tracking"]["persistent_ids"])
        self.assertTrue(all(o["track_status"] == "stateless" for o in ot_stateless["objects"]))

        # 模拟旧版 SAM3.1 服务无 /track 路由（返回 404）但有 /segment：平滑回退并记录诊断
        orig_do_post = sam31_svc.H.do_POST

        def _legacy_post(handler_self):
            if handler_self.path.startswith("/track"):
                return handler_self._send(404, {"status": "error", "error": "not_found"})
            return orig_do_post(handler_self)

        sam31_svc.H.do_POST = _legacy_post
        try:
            code_fb, ot_fb = _http_json(
                f"http://127.0.0.1:{self.vggt_port}/object_table",
                {
                    "session_id": "ep_fallback",
                    "image_b64": img_b64,
                    "K": K.tolist(),
                    "T_world_camera": T.tolist(),
                    "support_z_m": support_z,
                    "labels": ["box", "peg"],
                },
            )
            self.assertEqual(code_fb, 200)
            self.assertFalse(ot_fb["tracking"]["persistent_ids"])
            self.assertIn("fallback_error", ot_fb["tracking"])
            self.assertEqual(len(ot_fb["objects"]), 2)
            self.assertTrue(all(o["track_status"] == "stateless" for o in ot_fb["objects"]))
        finally:
            sam31_svc.H.do_POST = orig_do_post

    def test_sam31_288x288_mask_input_fidelity_dynamic_alignment_and_lock_free_session_close(self) -> None:
        """验证 SAM 3.1 288×288 low_res_logits 保真、非匹配尺寸动态对齐、多帧遮挡传播零报错及 GPU 锁繁忙下 /session_close 即时返回。"""
        # 1) 验证 to_low_res_logits 默认输出 (1, 288, 288) 且对已是 288×288 的输入原样保留零重采样损耗
        raw_288 = np.linspace(-6.0, 6.0, 288 * 288, dtype=np.float32).reshape(288, 288)
        dummy_mask = np.zeros((48, 64), dtype=bool)
        dummy_mask[10:20, 12:24] = True
        preserved = vsm.to_low_res_logits(raw_288, dummy_mask)
        self.assertEqual(preserved.shape, (1, 288, 288))
        np.testing.assert_array_equal(preserved[0], raw_288.astype(np.float16))
        from_bool = vsm.to_low_res_logits(None, dummy_mask)
        self.assertEqual(from_bool.shape, (1, 288, 288))

        # 2) 验证 _align_mask_input 将旧 (1, 256, 256) 或二维掩码动态对齐到模型真实 mask_input_size
        aligned_288 = sam31_svc._align_mask_input(np.ones((1, 256, 256), dtype=np.float16), (288, 288))
        self.assertEqual(aligned_288.shape, (1, 288, 288))
        self.assertEqual(aligned_288.dtype, np.float32)
        aligned_320 = sam31_svc._align_mask_input(np.ones((256, 256), dtype=np.float32), (320, 320))
        self.assertEqual(aligned_320.shape, (1, 320, 320))

        # 3) 3 帧跨帧运动+局部遮挡（第 2 帧文本检测漏检 peg，强制触发 _predict_inst_cb(..., mask_input=...)）：3/3 persistent_ids=True 且 errors=0
        img1_b64, K, T, support_z, d1, c1 = self._build_main_scene(peg_toppled=False)
        img2_b64, _, _, _, d2, c2 = self._build_main_scene(peg_toppled=True)
        img3 = np.asarray(Image.open(io.BytesIO(base64.b64decode(img1_b64))).convert("RGB")).copy()
        img3[0, 0] = (176, 175, 175)
        img3_b64 = _encode_png_b64(img3)
        self.next_vggt_outputs.extend([(d1, c1), (d2, c2), (d1, c1)])

        for idx_f, (b64_f, drop_peg) in enumerate(((img1_b64, False), (img2_b64, True), (img3_b64, False)), start=1):
            self.fake_sam.drop_text_names = {"peg"} if drop_peg else set()
            code_f, ot_f = _http_json(
                f"http://127.0.0.1:{self.vggt_port}/object_table",
                {
                    "session_id": "ep_occ_288",
                    "image_b64": b64_f,
                    "K": K.tolist(),
                    "T_world_camera": T.tolist(),
                    "support_z_m": support_z,
                    "labels": ["box", "peg"],
                },
            )
            self.assertEqual(code_f, 200, f"frame {idx_f} failed: {ot_f}")
            self.assertTrue(ot_f["tracking"]["persistent_ids"])
            self.assertEqual([o["track_id"] for o in ot_f["objects"]], ["o1", "o2"])

        self.assertIn((1, 288, 288), self.fake_sam.predict_inst_mask_shapes)
        _, sam_h = _http_json(f"http://127.0.0.1:{self.sam_port}/health", method="GET")
        self.assertEqual(sam_h["errors"], 0)
        self.assertEqual(sam_h["active_sessions"], 1)

        # 4) 模拟 GPU 锁与队列被耗时推理占满，验证 POST :8331/session_close 绕过队列与 GPU 锁即时清理并返回 200
        vggt_svc.ensure_worker_started()
        worker_entered = threading.Event()
        release_worker = threading.Event()
        orig_handle = vggt_svc.handle

        def _slow_handle(_req):
            worker_entered.set()
            release_worker.wait(timeout=2.0)
            return {"status": "ok"}

        vggt_svc.handle = _slow_handle
        sam31_svc.LOCK.acquire()
        try:
            slow_job = dict(req={}, done=threading.Event(), t_enq=time.time(), kind="height")
            vggt_svc.Q.put_nowait(slow_job)
            self.assertTrue(worker_entered.wait(timeout=1.0))

            t0 = time.monotonic()
            code_cls, ans_cls = _http_json(
                f"http://127.0.0.1:{self.vggt_port}/session_close",
                {"session_id": "ep_occ_288", "timeout_s": 0.5},
            )
            dt = time.monotonic() - t0
            self.assertEqual(code_cls, 200)
            self.assertTrue(ans_cls["closed"])
            self.assertLess(dt, 0.20)
        finally:
            sam31_svc.LOCK.release()
            release_worker.set()
            slow_job["done"].wait(timeout=1.0)
            vggt_svc.handle = orig_handle

        _, vggt_h_after = _http_json(f"http://127.0.0.1:{self.vggt_port}/health", method="GET")
        _, sam_h_after = _http_json(f"http://127.0.0.1:{self.sam_port}/health", method="GET")
        self.assertEqual(vggt_h_after["active_sessions"], 0)
        self.assertEqual(sam_h_after["active_sessions"], 0)

    def test_sam31_outage_and_mid_episode_track_failure_return_503_instead_of_empty_table(self) -> None:
        """验证 SAM 3.1 不可达/503 或中途跨帧 /track 失败时返回 HTTP 503，绝不静默伪造 200 空物体表或洗牌 ID。"""
        img1_b64, K, T, support_z, d1, c1 = self._build_main_scene(peg_toppled=False)
        img2_b64, _, _, _, d2, c2 = self._build_main_scene(peg_toppled=True)
        self.next_vggt_outputs.extend([(d1, c1), (d2, c2)])

        # 1) SAM 3.1 完全宕机（503）：/object_table 必须返回 503 错误而非 status="ok", objects=[]
        orig_sam_post = sam31_svc.H.do_POST
        sam31_svc.H.do_POST = lambda s: s._send(503, {"status": "error", "reason": "cuda_out_of_memory", "error": "OOM"})
        try:
            code_down, ans_down = _http_json(
                f"http://127.0.0.1:{self.vggt_port}/object_table",
                {
                    "image_b64": img1_b64,
                    "K": K.tolist(),
                    "T_world_camera": T.tolist(),
                    "support_z_m": support_z,
                    "labels": ["box", "peg"],
                },
            )
            self.assertEqual(code_down, 503, ans_down)
            self.assertEqual(ans_down["status"], "error")
            self.assertEqual(ans_down["reason"], "upstream_unavailable")
        finally:
            sam31_svc.H.do_POST = orig_sam_post

        # 2) 会话第 1 帧已建立持久轨迹后，第 2 帧 /track 故障时严禁降级为无状态 /segment 重编 o1..oN
        code_f1, ans_f1 = _http_json(
            f"http://127.0.0.1:{self.vggt_port}/object_table",
            {
                "session_id": "ep_mid_fail",
                "image_b64": img1_b64,
                "K": K.tolist(),
                "T_world_camera": T.tolist(),
                "support_z_m": support_z,
                "labels": ["box", "peg"],
            },
        )
        self.assertEqual(code_f1, 200)
        self.assertTrue(ans_f1["tracking"]["persistent_ids"])

        def _fail_track_mid_episode(handler_self):
            if handler_self.path.startswith("/track"):
                return handler_self._send(404, {"status": "error", "error": "track_crashed"})
            return orig_sam_post(handler_self)

        sam31_svc.H.do_POST = _fail_track_mid_episode
        try:
            code_f2, ans_f2 = _http_json(
                f"http://127.0.0.1:{self.vggt_port}/object_table",
                {
                    "session_id": "ep_mid_fail",
                    "image_b64": img2_b64,
                    "K": K.tolist(),
                    "T_world_camera": T.tolist(),
                    "support_z_m": support_z,
                    "labels": ["box", "peg"],
                },
            )
            self.assertEqual(code_f2, 503, ans_f2)
            self.assertEqual(ans_f2["status"], "error")
            self.assertIn("sam31_track_failed", ans_f2["error"])
        finally:
            sam31_svc.H.do_POST = orig_sam_post

    def test_cross_session_points_mask_isolation_and_tempfile_cleanup(self) -> None:
        """验证不同 session_id 同图请求 /points 绝不串用彼此掩码，且 _run_vggt_batch 非 16 倍数临时文件异常时零残留。"""
        img_b64, K, T, support_z, d, c = self._build_main_scene(peg_toppled=False)
        self.next_vggt_outputs.append((d, c))

        code_ot, ans_ot = _http_json(
            f"http://127.0.0.1:{self.vggt_port}/object_table",
            {
                "session_id": "ep_owner",
                "image_b64": img_b64,
                "K": K.tolist(),
                "T_world_camera": T.tolist(),
                "support_z_m": support_z,
                "labels": ["box", "peg"],
            },
        )
        self.assertEqual(code_ot, 200)
        self.assertEqual(len(ans_ot["objects"]), 2)

        # 另一会话 ep_other 尚未执行 /object_table，直接对同图调 /points：obj_id 必须全 0（绝不泄漏 ep_owner 掩码）
        code_pts_other, ans_pts_other = _http_json(
            f"http://127.0.0.1:{self.vggt_port}/points",
            {
                "session_id": "ep_other",
                "image_b64": img_b64,
                "K": K.tolist(),
                "T_world_camera": T.tolist(),
                "support_z_m": support_z,
            },
        )
        self.assertEqual(code_pts_other, 200)
        with np.load(io.BytesIO(base64.b64decode(ans_pts_other["npz_b64"])), allow_pickle=False) as npz_other:
            self.assertTrue(np.all(npz_other["obj_id"] == 0))

        # 匿名调用（无 session_id）仍兼容使用同图最新掩码
        code_pts_anon, ans_pts_anon = _http_json(
            f"http://127.0.0.1:{self.vggt_port}/points",
            {
                "image_b64": img_b64,
                "K": K.tolist(),
                "T_world_camera": T.tolist(),
                "support_z_m": support_z,
            },
        )
        self.assertEqual(code_pts_anon, 200)
        with np.load(io.BytesIO(base64.b64decode(ans_pts_anon["npz_b64"])), allow_pickle=False) as npz_anon:
            self.assertTrue(np.any(npz_anon["obj_id"] > 0))

        # 验证 _run_vggt_batch 对非 16 倍数图像在预处理抛错时立即清理临时 PNG
        import os
        seen_tmp_paths: list[str] = []
        orig_load_fn = vggt_svc.load_and_preprocess_images
        orig_torch = vggt_svc.torch

        class _DummyTorch:
            @staticmethod
            def inference_mode():
                import contextlib
                return contextlib.nullcontext()

            class cuda:
                @staticmethod
                def is_available():
                    return False

        def _failing_load(paths, **_kw):
            seen_tmp_paths.extend(paths)
            self.assertTrue(os.path.exists(paths[0]))
            raise RuntimeError("preprocess_boom")

        vggt_svc.load_and_preprocess_images = _failing_load
        vggt_svc.torch = _DummyTorch
        try:
            with self.assertRaisesRegex(RuntimeError, "preprocess_boom"):
                vggt_svc._run_vggt_batch([np.zeros((50, 70, 3), dtype=np.uint8)])
        finally:
            vggt_svc.load_and_preprocess_images = orig_load_fn
            vggt_svc.torch = orig_torch
        self.assertEqual(len(seen_tmp_paths), 1)
        self.assertFalse(os.path.exists(seen_tmp_paths[0]))

    def test_gpu_oom_sticky_cuda_content_length_abandoned_queue_and_curobo_http(self) -> None:
        """验证三项后端服务 OOM 503 缓存回收、粘滞 CUDA /health 503、Content-Length 护栏、超时废单跳过及 cuRobo HTTP 分类。"""
        from services.curobo_ik import curobo_ik_service as curobo_svc

        # 1) Content-Length 护栏（负值 -> 400，超上限 -> 413）
        for port in (self.vggt_port, self.sam_port):
            path = "/height" if port == self.vggt_port else "/segment"
            for bad_cl, expected_code in (("-5", 400), (str(vggt_svc.MAX_REQUEST_BYTES + 10), 413)):
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}{path}",
                    data=b"{}",
                    headers={"Content-Type": "application/json", "Content-Length": bad_cl},
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=5.0) as resp:
                        got_code = resp.status
                except urllib.error.HTTPError as err:
                    err.close()
                    got_code = err.code
                self.assertEqual(got_code, expected_code)

        # 2) SAM 3.1 OOM -> 503 + CACHE 清空 + 热会话卸载；粘滞 CUDA -> 503 且 /health -> 503
        sam31_svc.CACHE["dummy"] = ("img", {})
        sam31_svc.SESSION_MANAGER.update_tracks(
            session_id="hot_ep",
            img_rgb=np.zeros((32, 32, 3), dtype=np.uint8),
            image_sha256="h1",
            candidates=[],
            hot_gpu_state={"sem": _FakeTensor(np.ones((2, 2)), "cuda")},
        )
        orig_seg = sam31_svc.segment
        try:
            sam31_svc.segment = lambda _req: (_ for _ in ()).throw(RuntimeError("CUDA out of memory"))
            code_oom, ans_oom = _http_json(f"http://127.0.0.1:{self.sam_port}/segment", {"image_b64": "e30="})
            self.assertEqual(code_oom, 503)
            self.assertEqual(ans_oom["reason"], "cuda_out_of_memory")
            self.assertEqual(len(sam31_svc.CACHE), 0)
            self.assertEqual(sam31_svc.SESSION_MANAGER.stats()["gpu_hot_sessions"], 0)

            sam31_svc.segment = lambda _req: (_ for _ in ()).throw(KeyError("internal_bug_key"))
            code_500, ans_500 = _http_json(f"http://127.0.0.1:{self.sam_port}/segment", {"image_b64": "e30="})
            self.assertEqual(code_500, 500)
            self.assertEqual(ans_500["reason"], "internal_error")

            sam31_svc.segment = lambda _req: (_ for _ in ()).throw(RuntimeError("CUDA error: misaligned address"))
            code_sticky, _ = _http_json(f"http://127.0.0.1:{self.sam_port}/segment", {"image_b64": "e30="})
            self.assertEqual(code_sticky, 503)
            code_sh, body_sh = _http_json(f"http://127.0.0.1:{self.sam_port}/health", method="GET")
            self.assertEqual(code_sh, 503)
            self.assertEqual(body_sh["status"], "error")
        finally:
            sam31_svc.segment = orig_seg
            sam31_svc.CUDA_FATAL_ERROR = None

        # 3) VGGT 队列超时废单跳过：请求 1 占住 worker，请求 2 在队列内 504 超时后绝不再消耗 worker 执行
        entered_1 = threading.Event()
        release_1 = threading.Event()
        executed_ids: list[str] = []
        orig_handle = vggt_svc.handle

        def _controlled_handle(req_in):
            rid = req_in.get("req_id", "unknown")
            executed_ids.append(rid)
            if rid == "first":
                entered_1.set()
                release_1.wait(timeout=2.0)
            return {"status": "ok"}

        vggt_svc.handle = _controlled_handle
        abandoned_before = vggt_svc.STATS["abandoned"]
        first_box: dict[str, Any] = {}
        t_first = threading.Thread(
            target=lambda: first_box.setdefault(
                "res",
                _http_json(f"http://127.0.0.1:{self.vggt_port}/height", {"req_id": "first", "timeout_s": 5.0}),
            ),
            daemon=True,
        )
        try:
            t_first.start()
            self.assertTrue(entered_1.wait(timeout=1.0))
            code_to, ans_to = _http_json(
                f"http://127.0.0.1:{self.vggt_port}/height",
                {"req_id": "second_abandoned", "timeout_s": 0.03},
            )
            self.assertEqual(code_to, 504)
            self.assertEqual(ans_to["error"], "timeout_in_queue")
            release_1.set()
            t_first.join(timeout=2.0)
            code_after, _ = _http_json(
                f"http://127.0.0.1:{self.vggt_port}/height",
                {"req_id": "third_live", "timeout_s": 5.0},
            )
            self.assertEqual(code_after, 200)
            self.assertEqual(executed_ids, ["first", "third_live"])
            self.assertGreaterEqual(vggt_svc.STATS["abandoned"], abandoned_before + 1)

            # VGGT 粘滞 CUDA 异常 -> 503 且 /health -> 503
            vggt_svc.handle = lambda _r: (_ for _ in ()).throw(RuntimeError("CUDA error: device-side assert triggered"))
            code_vc, _ = _http_json(f"http://127.0.0.1:{self.vggt_port}/height", {"timeout_s": 2.0})
            self.assertEqual(code_vc, 503)
            code_vh, body_vh = _http_json(f"http://127.0.0.1:{self.vggt_port}/health", method="GET")
            self.assertEqual(code_vh, 503)
            self.assertEqual(body_vh["status"], "error")
        finally:
            release_1.set()
            vggt_svc.handle = orig_handle
            vggt_svc.CUDA_FATAL_ERROR = None

        # 4) cuRobo HTTP 处理器分类：坏 JSON -> 400，RequestError -> 409，内部 TypeError -> 500，OOM -> 503 + empty_cache，粘滞 CUDA -> /health 503
        empty_cache_calls = []

        class _FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def empty_cache():
                empty_cache_calls.append(1)

        class _FakeCuroboSolver:
            torch = type("_FakeTorchMod", (), {"cuda": _FakeCuda})
            cuda_fatal = None
            mode = "ok"

            def health(self):
                return {"status": "ok", "solver": "curobo"}

            def solve(self, _req):
                if self.mode == "limits":
                    raise curobo_svc.RequestError(409, "joint_limits_mismatch")
                if self.mode == "bug":
                    raise TypeError("unexpected_none_tensor")
                if self.mode == "oom":
                    raise RuntimeError("CUDA error 2: out of memory")
                if self.mode == "sticky":
                    raise RuntimeError("CUDA error: an illegal memory access was encountered")
                return {"status": "ok", "solutions": []}

            def path_check(self, _req):
                return {"status": "ok", "segments": []}

        fake_solver = _FakeCuroboSolver()
        ik_srv = ThreadingHTTPServer(("127.0.0.1", 0), curobo_svc.make_http_handler(fake_solver))
        ik_srv.daemon_threads = True
        ik_port = ik_srv.server_address[1]
        ik_thread = threading.Thread(target=ik_srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        ik_thread.start()
        try:
            bad_req = urllib.request.Request(
                f"http://127.0.0.1:{ik_port}/ik",
                data=b"{not_json",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx_400:
                urllib.request.urlopen(bad_req, timeout=5.0)
            ctx_400.exception.close()
            self.assertEqual(ctx_400.exception.code, 400)

            fake_solver.mode = "limits"
            c_409, b_409 = _http_json(f"http://127.0.0.1:{ik_port}/ik", {})
            self.assertEqual(c_409, 409)
            self.assertEqual(b_409["reason"], "joint_limits_mismatch")

            fake_solver.mode = "bug"
            c_500, b_500 = _http_json(f"http://127.0.0.1:{ik_port}/ik", {})
            self.assertEqual(c_500, 500)
            self.assertTrue(b_500["reason"].startswith("solver_error:TypeError:"))

            fake_solver.mode = "oom"
            c_oom, b_oom = _http_json(f"http://127.0.0.1:{ik_port}/ik", {})
            self.assertEqual(c_oom, 503)
            self.assertEqual(b_oom["reason"], "cuda_out_of_memory")
            self.assertEqual(len(empty_cache_calls), 1)
            c_hok, _ = _http_json(f"http://127.0.0.1:{ik_port}/health", method="GET")
            self.assertEqual(c_hok, 200)

            fake_solver.mode = "sticky"
            c_st, _ = _http_json(f"http://127.0.0.1:{ik_port}/ik", {})
            self.assertEqual(c_st, 503)
            c_hbad, b_hbad = _http_json(f"http://127.0.0.1:{ik_port}/health", method="GET")
            self.assertEqual(c_hbad, 503)
            self.assertEqual(b_hbad["status"], "error")
        finally:
            ik_srv.shutdown()
            ik_srv.server_close()
            ik_thread.join(timeout=2.0)

    def test_points_image_sha256_cache_hit_miss_multiview_and_keepalive_early_errors(self) -> None:
        """验证 `/points` 的 `image_sha256` 缓存复用、404 miss 零错误计数、多视角复用、真实 HTTP/1.1 Keep-Alive 与早退关闭连接。"""
        import hashlib
        import http.client
        import socket

        img_main_b64, K_main, T_main, support_z, d_main, c_main = self._build_main_scene()
        img_sha = hashlib.sha256(base64.b64decode(img_main_b64)).hexdigest()
        with vggt_svc.CLOCK:
            vggt_svc.CACHE.clear()
            vggt_svc.MULTIVIEW_CACHE.clear()
        orig_errors = vggt_svc.STATS["errors"]
        self.addCleanup(vggt_svc.STATS.__setitem__, "errors", orig_errors)
        vggt_svc.STATS["errors"] = 0

        # 1) 缓存尚未写入时仅传 image_sha256 -> 返回 HTTP 404 reason="image_cache_miss"，且绝不计入 STATS["errors"]
        code_miss, ans_miss = _http_json(
            f"http://127.0.0.1:{self.vggt_port}/points",
            {
                "image_sha256": img_sha,
                "K": K_main.tolist(),
                "T_world_camera": T_main.tolist(),
                "support_z_m": support_z,
                "voxel_m": 0.01,
            },
        )
        self.assertEqual(code_miss, 404, ans_miss)
        self.assertEqual(ans_miss["status"], "error")
        self.assertEqual(ans_miss["reason"], "image_cache_miss")
        self.assertEqual(ans_miss["error"], "image_cache_miss")
        self.assertEqual(vggt_svc.STATS["errors"], 0)

        # 2) 非法 image_sha256 -> HTTP 422 invalid_input
        code_bad_sha, ans_bad_sha = _http_json(
            f"http://127.0.0.1:{self.vggt_port}/points",
            {
                "image_sha256": "not-a-64-char-hex-digest",
                "K": K_main.tolist(),
                "T_world_camera": T_main.tolist(),
                "support_z_m": support_z,
            },
        )
        self.assertEqual(code_bad_sha, 422)
        self.assertEqual(ans_bad_sha["reason"], "invalid_input")

        # 3) /object_table 填充 CACHE 与实例掩码后，仅传 image_sha256 的 /points 零重传直接命中 CACHE 并保留 obj_id
        self.vggt_calls.clear()
        self.next_vggt_outputs.append((d_main, c_main))
        code_ot, ans_ot = _http_json(
            f"http://127.0.0.1:{self.vggt_port}/object_table",
            {
                "image_b64": img_main_b64,
                "K": K_main.tolist(),
                "T_world_camera": T_main.tolist(),
                "support_z_m": support_z,
                "labels": ["box", "peg"],
            },
        )
        self.assertEqual(code_ot, 200)
        self.assertIn("points_image_sha256", ans_ot["caps"])
        self.assertEqual(len(self.vggt_calls), 1)

        code_hit, ans_hit = _http_json(
            f"http://127.0.0.1:{self.vggt_port}/points",
            {
                "image_sha256": img_sha.upper(),  # 大小写十六进制均兼容归一化
                "K": K_main.tolist(),
                "T_world_camera": T_main.tolist(),
                "support_z_m": support_z,
                "voxel_m": 0.01,
            },
        )
        self.assertEqual(code_hit, 200, ans_hit)
        self.assertTrue(ans_hit["cache_hit"])
        self.assertEqual(ans_hit["image_sha256"], img_sha)
        self.assertEqual(len(self.vggt_calls), 1)
        with np.load(io.BytesIO(base64.b64decode(ans_hit["npz_b64"])), allow_pickle=False) as npz_hit:
            self.assertTrue(np.any(npz_hit["obj_id"] == 1))
            self.assertTrue(np.any(npz_hit["obj_id"] == 2))

        # 4) 主图仅传 image_sha256 + 附加腕部视角传 image_b64：复用 cached_entry["img"] 融合多视角并写入 MULTIVIEW_CACHE
        H, W = d_main.shape
        T_wrist = _look_at_camera([0.22, -0.18, 0.96], [0.34, 0.0, 0.78])
        boxes = [
            dict(x_range=(0.24, 0.34), y_range=(-0.06, 0.04), top_z=0.83, floor_z=0.765, wall_m=0.015),
            dict(x_range=(0.46, 0.51), y_range=(-0.03, 0.03), top_z=0.85, floor_z=None),
        ]
        d_wrist, c_wrist = _render_synthetic_depth(H, W, K_main, T_wrist, support_z, boxes, scale_factor=3.0)
        wrist_b64 = _encode_png_b64(np.full((H, W, 3), 160, dtype=np.uint8))
        self.next_vggt_outputs.extend([(d_main, c_main), (d_wrist, c_wrist)])

        mv_body = {
            "image_sha256": img_sha,
            "K": K_main.tolist(),
            "T_world_camera": T_main.tolist(),
            "support_z_m": support_z,
            "voxel_m": 0.01,
            "extra_views": [
                {
                    "camera_id": "cam_left_wrist",
                    "image_b64": wrist_b64,
                    "K": K_main.tolist(),
                    "T_world_camera": T_wrist.tolist(),
                    "max_range_m": 0.40,
                }
            ],
        }
        code_mv1, ans_mv1 = _http_json(f"http://127.0.0.1:{self.vggt_port}/points", mv_body)
        self.assertEqual(code_mv1, 200, ans_mv1)
        self.assertEqual(ans_mv1["fused_views"], ["cam_left_wrist"])
        calls_after_mv1 = len(self.vggt_calls)
        code_mv2, ans_mv2 = _http_json(f"http://127.0.0.1:{self.vggt_port}/points", mv_body)
        self.assertEqual(code_mv2, 200, ans_mv2)
        self.assertTrue(ans_mv2["cache_hit"])
        self.assertEqual(len(self.vggt_calls), calls_after_mv1)

        # 5) 真实 vggt_svc HTTP/1.1 Keep-Alive 正例：单条持久连接连续完成 GET /health 与 POST /points（相同本地端口、will_close=False）
        ka_conn = http.client.HTTPConnection("127.0.0.1", self.vggt_port, timeout=5.0)
        try:
            ka_conn.request("GET", "/health", headers={"Connection": "keep-alive"})
            ka_r1 = ka_conn.getresponse()
            ka_b1 = json.loads(ka_r1.read())
            sock_name_1 = ka_conn.sock.getsockname()
            self.assertEqual(ka_r1.status, 200)
            self.assertFalse(ka_r1.will_close)
            self.assertEqual(ka_b1["status"], "ok")

            ka_pts_payload = json.dumps(
                {
                    "image_sha256": img_sha,
                    "K": K_main.tolist(),
                    "T_world_camera": T_main.tolist(),
                    "support_z_m": support_z,
                    "voxel_m": 0.01,
                }
            ).encode()
            ka_conn.request(
                "POST",
                "/points",
                body=ka_pts_payload,
                headers={"Content-Type": "application/json", "Content-Length": str(len(ka_pts_payload)), "Connection": "keep-alive"},
            )
            ka_r2 = ka_conn.getresponse()
            ka_b2 = json.loads(ka_r2.read())
            sock_name_2 = ka_conn.sock.getsockname()
            self.assertEqual(ka_r2.status, 200)
            self.assertFalse(ka_r2.will_close)
            self.assertTrue(ka_b2["cache_hit"])
            self.assertEqual(sock_name_1, sock_name_2)
        finally:
            ka_conn.close()

        # 6) HTTP/1.1 Keep-Alive 早退错误（404 未知路由、413 超大载荷、400 短读未完整请求体）强制返回 Connection: close
        unused_body = b'{"unused": 1}'
        for bad_path, bad_headers, bad_body, half_close, exp_status in (
            ("/unknown_route", {"Content-Type": "application/json", "Content-Length": str(len(unused_body))}, unused_body, False, 404),
            (
                "/points",
                {"Content-Type": "application/json", "Content-Length": str(vggt_svc.MAX_REQUEST_BYTES + 8)},
                b"",
                False,
                413,
            ),
            ("/points", {"Content-Type": "application/json", "Content-Length": "32"}, b'{"truncated":', True, 400),
        ):
            conn = http.client.HTTPConnection("127.0.0.1", self.vggt_port, timeout=5.0)
            try:
                conn.putrequest("POST", bad_path)
                for hk, hv in bad_headers.items():
                    conn.putheader(hk, hv)
                conn.endheaders()
                if bad_body:
                    conn.send(bad_body)
                if half_close and conn.sock is not None:
                    conn.sock.shutdown(socket.SHUT_WR)
                resp = conn.getresponse()
                _ = resp.read()
                self.assertEqual(resp.status, exp_status)
                self.assertTrue(resp.will_close)
                self.assertEqual((resp.getheader("Connection") or "").lower(), "close")
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
