"""VGGT-Omega object-height service (binds 127.0.0.1 by default; packaging and install: services/vggt_height/README.md).

Input per request: one RGB image (PNG/JPEG bytes, base64), the camera's calibrated intrinsics K (3x3),
camera-to-world T_world_camera (4x4, OpenCV optical axes right/down/forward) and the support (table)
height in the same world frame, plus the pixels of interest. Nothing else (no object names, no truth).

Method (see docs/vggt_height.md):
  1. one VGGT-Omega forward pass on the native image -> relative depth + confidence;
  2. back-project the depth with the CALIBRATED K and the known camera rotation ("unit-scale" points),
     RANSAC a near-horizontal plane (the table), scale s so the plane sits at support_z (per image);
  3. per pixel: object mask = SAM 2.1 point prompt at the pixel (default; fallback window), VGGT points in the
     mask (eroded 2 px, conf > p10, above support + 4 mm, DBSCAN 1 cm largest cluster) -> top = p98 z,
     center = (top + support) / 2.  method="window": 21x21 px window around the pixel, p98 z as top.
v2 (2026-10-20, object_table line): POST /object_table (automatic SAM 2.1 masks -> per-object VGGT geometry, see
object_table()) and plausibility flags on /height pixels (height above max, low depth confidence, contact
mismatch); /height values are unchanged.
v3 (2026-10-20, scene3d line): POST /points (voxel-downsampled VGGT point cloud of the image in the world frame, per point
confidence rank, source pixel and a near-robot flag, as a compressed npz) for NI-RUA features.scene3d; /height and
/object_table code and numerics are unchanged.
One shared process; requests are served FIFO by a single GPU worker thread; results per image are cached.
"""
import argparse
import atexit
import base64
import binascii
import collections
import contextlib
import gc
import hashlib
import http.client
import io
import json
import math
import os
import queue
import socket
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

try:
    import torch
except ImportError:  # 允许在无 PyTorch 的纯 CPU 单测环境中导入与桩测试
    torch = None

try:
    from .multiview_cluster import (
        align_and_gate_extra_view,
        build_instance_id_map,
        compute_cavity_depth,
        compute_conf_rank,
        compute_surface_slices,
        voxel_connected_cluster,
        voxel_downsample_multiview,
    )
except ImportError:
    from multiview_cluster import (  # type: ignore[no-redef]
        align_and_gate_extra_view,
        build_instance_id_map,
        compute_cavity_depth,
        compute_conf_rank,
        compute_surface_slices,
        voxel_connected_cluster,
        voxel_downsample_multiview,
    )


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("VGGT_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("VGGT_PORT", "8331")))
    ap.add_argument("--ckpt", default=os.environ.get("VGGT_CKPT", "weights/vggt_omega_1b_512.pt"))
    ap.add_argument("--sam", default=os.environ.get("SAM2_MODEL", "facebook/sam2.1-hiera-large"))
    ap.add_argument("--cache", type=int, default=48, help="images kept (reconstruction + SAM embedding)")
    ap.add_argument("--max_queue", type=int, default=64)
    ap.add_argument("--status", default="")
    ap.add_argument("--sam31", default=os.environ.get("SAM31_ENDPOINT", "http://127.0.0.1:8333"), help="SAM 3.1 text-prompt service (sam31/sam31_service.py)")
    return ap


args = build_arg_parser().parse_args([])
DEV = "cuda"
DEBUG_DIR = os.environ.get("OT_DEBUG_DIR", "")   # dev only: dump the object masks of each /object_table call
VERSION = "vggt-height-v2"
SERVICE_CAPS = ["points_image_sha256"]
MAX_REQUEST_BYTES = 64 * 1024 * 1024
CUDA_FATAL_ERROR: str | None = None
WIN = 10                    # window half width (px), fixed a priori in section 9


class UpstreamServiceError(RuntimeError):
    """SAM 3.1 等上游依赖服务不可达、超时或返回 5xx 时抛出的显式基础设施错误。"""


def _is_valid_sha256_hex(val) -> bool:
    return isinstance(val, str) and len(val) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in val)


def _parse_cam_inputs(req):
    """校验并解码请求中的 `image_b64`、`K`、`T_world_camera` 与 `support_z_m`，非法入参统一抛出 `ValueError`。"""
    if not isinstance(req, dict):
        raise ValueError("request_must_be_object")
    for field in ("image_b64", "K", "T_world_camera", "support_z_m"):
        if field not in req:
            raise ValueError(f"missing_required_field:{field}")
    if not isinstance(req["image_b64"], str) or not req["image_b64"]:
        raise ValueError("invalid_image_b64")
    data = base64.b64decode(req["image_b64"])
    try:
        K = np.asarray(req["K"], np.float64).reshape(3, 3)
        T = np.asarray(req["T_world_camera"], np.float64).reshape(4, 4)
        support_z = float(req["support_z_m"])
    except TypeError as exc:
        raise ValueError("invalid_camera_geometry") from exc
    if not (np.isfinite(K).all() and np.isfinite(T).all() and math.isfinite(support_z)):
        raise ValueError("nonfinite_camera_geometry")
    return data, K, T, support_z


def _parse_points_inputs(req):
    """校验 `/points` 入参：支持传 `image_b64`（字节优先）或仅传 64 位十六进制 `image_sha256` 命中 `CACHE`。"""
    if not isinstance(req, dict):
        raise ValueError("request_must_be_object")
    for field in ("K", "T_world_camera", "support_z_m"):
        if field not in req:
            raise ValueError(f"missing_required_field:{field}")
    has_sha = "image_sha256" in req and req["image_sha256"] is not None
    if has_sha and not _is_valid_sha256_hex(req["image_sha256"]):
        raise ValueError("invalid_image_sha256")
    if "image_b64" in req and req["image_b64"] is not None:
        if not isinstance(req["image_b64"], str) or not req["image_b64"]:
            raise ValueError("invalid_image_b64")
        data = base64.b64decode(req["image_b64"])
        img_sha = hashlib.sha256(data).hexdigest()
    else:
        if not has_sha:
            raise ValueError("missing_required_field:image_b64")
        data = None
        img_sha = str(req["image_sha256"]).lower()
    try:
        K = np.asarray(req["K"], np.float64).reshape(3, 3)
        T = np.asarray(req["T_world_camera"], np.float64).reshape(4, 4)
        support_z = float(req["support_z_m"])
    except TypeError as exc:
        raise ValueError("invalid_camera_geometry") from exc
    if not (np.isfinite(K).all() and np.isfinite(T).all() and math.isfinite(support_z)):
        raise ValueError("nonfinite_camera_geometry")
    return data, img_sha, K, T, support_z
ABOVE = 0.004               # points above the support plane (m)
MAX_MASK_FRAC = 0.15        # SAM masks larger than this fraction of the image are support/background
VGGT = None
load_and_preprocess_images = None
LOAD_S = 0.0
MEM_LOADED_MB = 0.0


def log(*a):
    print(time.strftime("%F %T"), *a, flush=True)


def _cuda_ready() -> bool:
    return bool(torch is not None and torch.cuda.is_available())


def _inference_ctx():
    return torch.inference_mode() if torch is not None else contextlib.nullcontext()


def _sync_cuda():
    if _cuda_ready():
        torch.cuda.synchronize()


def _reset_peak_mem():
    if _cuda_ready():
        torch.cuda.reset_peak_memory_stats()


def _peak_mem_mb() -> float:
    return float(torch.cuda.max_memory_allocated() / 2**20) if _cuda_ready() else 0.0


def _to_numpy_f32(val) -> np.ndarray:
    if hasattr(val, "detach"):
        val = val.detach()
    if hasattr(val, "float"):
        val = val.float()
    if hasattr(val, "cpu"):
        val = val.cpu()
    if hasattr(val, "numpy"):
        val = val.numpy()
    return np.asarray(val, dtype=np.float32)


def load_runtime_model(parsed_args):
    """加载 VGGT-Omega 模型权重并记录显存指标。"""
    global VGGT, load_and_preprocess_images, LOAD_S, MEM_LOADED_MB
    from vggt_omega.models import VGGTOmega  # noqa: E402
    from vggt_omega.utils.load_fn import load_and_preprocess_images as _load_fn  # noqa: E402

    load_and_preprocess_images = _load_fn
    t0 = time.time()
    VGGT = VGGTOmega().to(DEV).eval()
    log("vggt load_state_dict", VGGT.load_state_dict(torch.load(parsed_args.ckpt, map_location="cpu", weights_only=True)))
    torch.cuda.synchronize()
    LOAD_S = time.time() - t0
    MEM_LOADED_MB = torch.cuda.memory_allocated() / 2**20
    log(f"models loaded in {LOAD_S:.1f}s, allocated {MEM_LOADED_MB:.0f} MB (VGGT-Omega only; segmentation via SAM 3.1)")


# ------------------------------------------------------------------ geometry (analyze.py, generalized to any T)
def fit_table(X1, conf, cam_z, support_z, rng, thr_rel=0.004, iters=400):
    """RANSAC near-horizontal plane on unit-scale world-oriented points X1 (origin at the camera centre).

    Only rays pointing into the half-space of ``z = support_z`` relative to ``cam_z`` (``(support_z - cam_z) * dz > 0``)
    can intersect ``z = support_z`` at a positive metric scale ``s > 0``; restricting candidates to that half-space
    prevents an upward-tilted wrist camera (e.g. at a ``clear_view`` pose) from fitting a horizontal ceiling in the
    upper image instead of the table in the lower image.
    """
    dz_support = float(support_z - cam_z)
    if not math.isfinite(dz_support) or abs(dz_support) <= 1e-6:
        raise ValueError("invalid_support_scale")
    valid = (conf > np.percentile(conf, 10)) & ((X1[..., 2] * dz_support) > 0)
    idx = np.flatnonzero(valid.reshape(-1))
    if len(idx) < 3:
        raise ValueError("no_horizontal_support_plane")
    pts = X1.reshape(-1, 3)
    thr = thr_rel * np.median(np.linalg.norm(pts[idx], axis=1))
    sub = rng.choice(idx, size=min(len(idx), 20000), replace=False)
    pts_sub = np.ascontiguousarray(pts[sub], dtype=np.float64)
    cos_lim = math.cos(math.radians(25))
    cand_n = []
    cand_d = []
    for _ in range(iters):
        a, b, c = pts[rng.choice(sub, 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        if n[2] < 0:
            n = -n
        if n[2] < cos_lim:
            continue
        cand_n.append(n)
        cand_d.append(float(-n @ a))
    if not cand_n:
        raise ValueError("no_horizontal_support_plane")
    n_mat = np.asarray(cand_n, dtype=np.float64)
    d_vec = np.asarray(cand_d, dtype=np.float64)
    best_idx, best_n = -1, -1
    for st in range(0, len(n_mat), 64):
        n_blk = n_mat[st : st + 64]
        d_blk = d_vec[st : st + 64]
        cnts = (np.abs(pts_sub @ n_blk.T + d_blk) < thr).sum(axis=0)
        loc_i = int(np.argmax(cnts))
        loc_c = int(cnts[loc_i])
        if loc_c > best_n:
            best_n = loc_c
            best_idx = st + loc_i
    n = n_mat[best_idx]
    d = float(d_vec[best_idx])
    inl = (np.abs(pts @ n + d) < thr) & valid.reshape(-1)
    Q = pts[inl]
    cen = Q.mean(0)
    qc = np.asarray(Q - cen, dtype=np.float64)
    _, vecs = np.linalg.eigh(qc.T @ qc)
    v = vecs[:, 0]
    n = v if v[2] > 0 else -v
    d = -n @ cen
    inl = (np.abs(pts @ n + d) < thr) & valid.reshape(-1)
    zs = pts[inl, 2]
    s = float(np.median((support_z - cam_z) / zs))
    if not (math.isfinite(s) and s > 0):
        raise ValueError("invalid_support_scale")
    tilt = math.degrees(math.acos(min(1.0, abs(n[2]))))
    resid = np.abs(cam_z + s * zs - support_z)
    return s, dict(tilt_deg=tilt, residual_p50_m=float(np.median(resid)), residual_p90_m=float(np.percentile(resid, 90)),
                   inlier_frac=float(inl.mean()))


def _run_vggt_batch(imgs_rgb):
    """将单视角或多视角 RGB 图像组装为 `[S, 3, H, W]` Batch 一次送入 `VGGTOmega` 前向并还原至各自原始分辨率。"""
    s_views = len(imgs_rgb)
    H0, W0 = imgs_rgb[0].shape[:2]
    t = time.time()
    if s_views == 1 and (H0 % 16 != 0 or W0 % 16 != 0) and load_and_preprocess_images is not None and torch is not None:
        with tempfile.NamedTemporaryFile(prefix="vggt_svc_", suffix=".png", delete=True) as tmp:
            Image.fromarray(imgs_rgb[0]).save(tmp.name)
            ims = load_and_preprocess_images([tmp.name], mode="balanced", image_resolution=512).to(DEV)
        mode = "balanced"
    else:
        h_net = H0 if H0 % 16 == 0 else max(16, int(round(H0 / 16.0)) * 16)
        w_net = W0 if W0 % 16 == 0 else max(16, int(round(W0 / 16.0)) * 16)
        mode = "native" if (h_net == H0 and w_net == W0) else "resized"
        aligned = [
            im if im.shape[:2] == (h_net, w_net) else cv2.resize(im, (w_net, h_net), interpolation=cv2.INTER_LINEAR)
            for im in imgs_rgb
        ]
        batch_np = np.stack(aligned, axis=0).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        if torch is not None:
            ims = torch.from_numpy(batch_np).to(DEV if _cuda_ready() else "cpu")
        else:
            ims = batch_np

    with _inference_ctx():
        pr = VGGT(ims)
    _sync_cuda()
    dt = time.time() - t
    h_out, w_out = int(ims.shape[-2]), int(ims.shape[-1])
    depths_raw = _to_numpy_f32(pr["depth"]).reshape(s_views, h_out, w_out)
    confs_raw = _to_numpy_f32(pr["depth_conf"]).reshape(s_views, h_out, w_out)
    depths, confs = [], []
    for idx_v, im in enumerate(imgs_rgb):
        Hi, Wi = im.shape[:2]
        d_i, c_i = depths_raw[idx_v], confs_raw[idx_v]
        if (h_out, w_out) != (Hi, Wi):
            d_i = cv2.resize(d_i, (Wi, Hi), interpolation=cv2.INTER_LINEAR)
            c_i = cv2.resize(c_i, (Wi, Hi), interpolation=cv2.INTER_LINEAR)
        depths.append(d_i)
        confs.append(c_i)
    return depths, confs, mode, dt


def _reconstruct_from_depth(depth, conf, K, T, support_z, dt, mode, allow_no_table=False):
    H, W = depth.shape[:2]
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    Pc = np.stack([(uu - K[0, 2]) / K[0, 0] * depth, (vv - K[1, 2]) / K[1, 1] * depth, depth], -1)
    X1 = Pc @ T[:3, :3].T
    rng = np.random.default_rng(0)       # deterministic per image
    try:
        s, table = fit_table(X1, conf, T[2, 3], support_z, rng)
    except ValueError as ex:
        if not allow_no_table:
            raise
        s, table = 1.0, dict(error=str(ex))   # hp: unscaled points (held rows rescale them to the FK depth anyway)
    X = T[:3, 3] + s * X1
    return dict(X=X.astype(np.float32), conf=conf.astype(np.float32), conf_p10=float(np.percentile(conf, 10)),
                s=s, table=table, vggt_s=dt, mode=mode, hw=(H, W))


def reconstruct(img_rgb, K, T, support_z, allow_no_table=False):
    depths, confs, mode, dt = _run_vggt_batch([img_rgb])
    return _reconstruct_from_depth(depths[0], confs[0], K, T, support_z, dt, mode, allow_no_table=allow_no_table)


def reconstruct_multiview(main_img_rgb, K, T, support_z, extra_imgs_rgb, fallback_rec=None):
    """将主视角与附加腕部视角拼成 `[S, 3, H, W]` Batch 一次送入 `VGGTOmega` 前向重建。"""
    depths, confs, mode, dt = _run_vggt_batch([main_img_rgb, *extra_imgs_rgb])
    try:
        rec_0 = _reconstruct_from_depth(depths[0], confs[0], K, T, support_z, dt, mode, allow_no_table=False)
    except ValueError:
        if fallback_rec is None:
            raise
        rec_0 = dict(fallback_rec, vggt_s=dt)
    extra_preds = list(zip(depths[1:], confs[1:]))
    return rec_0, extra_preds


K3 = np.ones((3, 3), np.uint8)


def estimate(rec, mask, support_z):
    """analyze.py estimate(): VGGT points of a mask -> top (p98 z) and centre height."""
    m2 = cv2.erode(mask.astype(np.uint8), K3, iterations=2).astype(bool)
    if m2.sum() < 15:
        m2 = mask
    sel = m2 & (rec["conf"] > rec["conf_p10"])
    if sel.sum() < 5:
        sel = m2
    P = rec["X"][sel].astype(np.float64)
    if len(P) < 3:
        return None
    above = P[:, 2] > support_z + ABOVE
    if above.sum() >= 10:
        P = P[above]
    rng = np.random.default_rng(0)
    if len(P) > 4000:
        P = P[rng.choice(len(P), 4000, replace=False)]
    P_main, c_frac, _ = voxel_connected_cluster(P, voxel_m=0.01, min_samples=5)
    if c_frac > 0.0:
        P = P_main
    z98 = float(np.percentile(P[:, 2], 98))
    lo, hi = np.percentile(P[:, :2], 2, axis=0), np.percentile(P[:, :2], 98, axis=0)
    return dict(top_z_m=z98, n_points=int(len(P)), xy_extent_m=[float(v) for v in (hi - lo)],
                xy_center_m=[float(v) for v in (lo + hi) / 2])


def window(rec, u, v, support_z):
    H, W = rec["hw"]
    ui, vi = int(u), int(v)
    sl = (slice(max(0, vi - WIN), vi + WIN + 1), slice(max(0, ui - WIN), ui + WIN + 1))
    Pw = rec["X"][sl].reshape(-1, 3)
    cw = rec["conf"][sl].reshape(-1) > rec["conf_p10"]
    Pw = Pw[cw & (Pw[:, 2] > support_z + ABOVE)]
    if len(Pw) < 10:
        return None
    return dict(top_z_m=float(np.percentile(Pw[:, 2], 98)), n_points=int(len(Pw)))


_SAM31_CONN_LOCK = threading.Lock()
_SAM31_CONN_POOL: dict[tuple[str, str, int], list[tuple[http.client.HTTPConnection, float]]] = {}
_SAM31_CONN_IDLE_TTL_S = 60.0


def _sam31_close_all_conns() -> None:
    with _SAM31_CONN_LOCK:
        buckets = list(_SAM31_CONN_POOL.values())
        _SAM31_CONN_POOL.clear()
    for bucket in buckets:
        for conn, _ in bucket:
            with contextlib.suppress(Exception):
                conn.close()


atexit.register(_sam31_close_all_conns)


def _sam31_acquire_conn(scheme: str, host: str, port: int, timeout_s: float) -> tuple[http.client.HTTPConnection, bool]:
    key = (scheme, host, port)
    now = time.monotonic()
    stale: list[http.client.HTTPConnection] = []
    chosen: http.client.HTTPConnection | None = None
    with _SAM31_CONN_LOCK:
        bucket = _SAM31_CONN_POOL.get(key)
        while bucket:
            conn, ts = bucket.pop()
            if (now - ts) > _SAM31_CONN_IDLE_TTL_S:
                stale.append(conn)
                continue
            chosen = conn
            break
    for old_conn in stale:
        with contextlib.suppress(Exception):
            old_conn.close()
    if chosen is not None:
        chosen.timeout = timeout_s
        if chosen.sock is not None:
            with contextlib.suppress(OSError):
                chosen.sock.settimeout(timeout_s)
        return chosen, True
    if scheme == "https":
        conn = http.client.HTTPSConnection(host, port, timeout=timeout_s)
    else:
        conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
    return conn, False


def _sam31_release_conn(scheme: str, host: str, port: int, conn: http.client.HTTPConnection) -> None:
    key = (scheme, host, port)
    evicted: list[http.client.HTTPConnection] = []
    with _SAM31_CONN_LOCK:
        if key not in _SAM31_CONN_POOL and len(_SAM31_CONN_POOL) >= 8:
            oldest_key = next(iter(_SAM31_CONN_POOL))
            evicted.extend(c for c, _ in _SAM31_CONN_POOL.pop(oldest_key, ()))
        bucket = _SAM31_CONN_POOL.setdefault(key, [])
        if len(bucket) < 4:
            bucket.append((conn, time.monotonic()))
            conn = None  # type: ignore[assignment]
    for old_conn in evicted:
        with contextlib.suppress(Exception):
            old_conn.close()
    if conn is not None:
        with contextlib.suppress(Exception):
            conn.close()


def _sam31_post_json(path: str, body: dict, timeout_s: float) -> dict:
    """调用 SAM 3.1 服务 JSON 接口，复用 HTTP/1.1 Keep-Alive 连接并将不可达/超时/5xx 显式转为 `UpstreamServiceError`。"""
    base = args.sam31.rstrip("/")
    parsed = urlsplit(base)
    scheme = (parsed.scheme or "http").lower()
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if scheme == "https" else 80)
    req_path = (parsed.path.rstrip("/") + path) if parsed.path else path
    payload = json.dumps(body).encode()
    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(payload)),
        "Connection": "keep-alive",
    }

    for attempt in range(2):
        conn, reused = _sam31_acquire_conn(scheme, host, port, float(timeout_s))
        try:
            conn.request("POST", req_path, body=payload, headers=headers)
            if conn.sock is not None:
                with contextlib.suppress(OSError):
                    conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            resp = conn.getresponse()
            raw = resp.read()
            status = int(resp.status)
            close_hdr = str(resp.getheader("Connection") or "").lower()
            if resp.will_close or close_hdr == "close" or status >= 400:
                with contextlib.suppress(Exception):
                    conn.close()
            else:
                _sam31_release_conn(scheme, host, port, conn)
        except (http.client.RemoteDisconnected, http.client.CannotSendRequest, BrokenPipeError, ConnectionResetError) as err:
            with contextlib.suppress(Exception):
                conn.close()
            if reused and attempt == 0:
                continue
            raise UpstreamServiceError(f"sam31_unreachable: {type(err).__name__}: {err}"[:200]) from err
        except (http.client.HTTPException, TimeoutError, OSError) as err:
            with contextlib.suppress(Exception):
                conn.close()
            raise UpstreamServiceError(f"sam31_unreachable: {type(err).__name__}: {err}"[:200]) from err

        if status >= 400:
            try:
                ans = json.loads(raw or b"{}")
                if not isinstance(ans, dict):
                    ans = {"status": "error", "error": f"http_{status}"}
            except ValueError:
                ans = {"status": "error", "error": f"http_{status}"}
            if status >= 500:
                raise UpstreamServiceError(f"sam31_http_{status}: {ans.get('error', '')}"[:200]) from None
            return ans

        ans = json.loads(raw)
        if not isinstance(ans, dict):
            raise UpstreamServiceError(f"sam31_non_object_response:{path}")
        return ans

    raise UpstreamServiceError("sam31_unreachable: keepalive_retry_exhausted")


def sam31_points(image_b64, points, H, W, timeout_s=60.0):
    """SAM 3.1 interactive point prompts on the SAM 3.1 service -> [(masks (3,H,W) bool, ious (3,))]."""
    if not len(points):
        return []
    body = dict(image_b64=image_b64, points=[[float(u), float(v)] for u, v in points], phrases=[])
    ans = _sam31_post_json("/segment", body, timeout_s)
    if ans.get("status") != "ok":
        raise RuntimeError(str(ans.get("error", "sam31_error"))[:200])
    if list(ans["image_size"]) != [W, H]:
        raise RuntimeError(f"sam31_image_size {ans['image_size']} != {[W, H]}")
    out = []
    for pt_res in ans.get("points") or []:
        ms = [np.unpackbits(np.frombuffer(base64.b64decode(m["mask_b64"]), np.uint8))[:H * W].reshape(H, W).astype(bool)
              for m in pt_res["masks"]]
        masks = np.stack(ms, axis=0) if ms else np.zeros((0, H, W), bool)
        ious = np.asarray([float(m["iou"]) for m in pt_res["masks"]], np.float32)
        out.append((masks, ious))
    return out


def pick_mask(masks, iou, u, v):
    """Highest predicted IoU among masks that contain the pixel and are not support/background-sized."""
    H, W = masks.shape[-2:]
    ui, vi = min(int(u), W - 1), min(int(v), H - 1)
    cand = [(float(q), i) for i, q in enumerate(iou) if masks[i][vi, ui] and masks[i].sum() <= MAX_MASK_FRAC * H * W]
    if not cand:
        return None, None
    q, i = max(cand)
    return masks[i], q


# ------------------------------------------------------------------ object table (NI-RUA features.object_table)
# POST /object_table {image_b64, K, T_world_camera, support_z_m, pixels?, robot_segments_m?, max_objects?,
#                     max_object_height_m?, params?, timeout_s?}
# Automatic SAM 2.1 masks (point-prompt grid restricted to pixels whose VGGT point lies above the support plane)
# -> drop support / background / robot masks -> per mask VGGT points -> centre, top, bottom, footprint (PCA yaw,
# length, width), principal-axis tilt, support relation, depth confidence, plausibility flags. No names, no truth.
OT_VERSION = "object-table-v5-hp1"
OT = dict(stride=8, min_above=0.004, bg_height=0.45, max_area_frac=0.08, min_area=40, iou=0.50, stab=0.80, min_relief=0.004,
          above_frac=0.6, overlap=0.3, batch=64, robot_radius=0.05, robot_frac=0.5, cluster_frac=0.6,
          contact_margin=0.03, max_object_height=0.30, low_conf_rank=0.25, support_gap=0.02, order="area",
          floating=0.03, reanchor=1, reanchor_mode="thickness", reanchor_margin=0.0, reuse_iou=0.5, mask_core=0,
          seg_backend="sam31", label_thr=0.4, label_max=20, label_iou=0.5, text_max_area_frac=0.08, auto_with_labels=0,
          auto_phrase="object", part_frac=0.5)
# object-table-v5 (2026-10-20): optional request field labels:[str] (phrases chosen by the CALLER, never by this service).
# seg_backend "sam31" (default): each phrase -> SAM 3.1 text-prompted instance masks (sam31_service.py, label_thr) -> the
# same filters and geometry as the automatic masks; an instance overlapping an already chosen object (IoU >= label_iou)
# adds its phrase to that object. With labels the automatic SAM 2.1 grid is skipped (auto_with_labels=1 keeps it).
# Every object carries label / label_score / labels; the response lists per phrase the object indices and a status.
# Without labels, or with seg_backend "sam21", or when the SAM 3.1 service fails: the v4 automatic SAM 2.1 path.
# Requested pixels always use the SAM 2.1 point prompt with the v4 reuse rule.
# Automatic part (seg_backend sam31): the class-agnostic prompt auto_phrase ("object"; a fixed generic word, not an
# object name) runs with the caller's phrases; its instances are added after the labelled ones, unlabelled, unless they
# repeat a chosen object (IoU >= label_iou) or lie mostly (> part_frac of their area) inside one. auto_phrase "" = labels only.
# reanchor_mode: "always" = rescale every floating object (lowest visible point > floating above the support plane,
# no supporting object) onto the table (object-table-v1); "thickness" = rescale only when the floating height
# (low_z - support_z) also exceeds the visible thickness of the raw point set (top_z - low_z) + reanchor_margin.
# Objects whose bottom is hidden (spheres, drawers) are then kept as measured and flagged "floating".
# object-table-v4 (2026-10-20): a requested pixel reuses an automatic object only when the pixel lies in that mask's core (its largest
# connected component; a SAM mask can contain detached parts, e.g. the rack mask also covering a separate white tube)
# AND the point-prompt mask at the pixel overlaps the object's mask by IoU >= reuse_iou; otherwise the point-prompt
# mask becomes a new object. Requested pixels that resolve to the same object are marked (same_object_as, status
# "ok_same_object"). mask_core=1 (default off) also cuts every automatic mask to its core before the greedy selection.


def _span_px(mask):
    ys, xs = np.nonzero(mask)
    return float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min())) if len(xs) else 0.0


def _erode(mask, *, thin=False):
    u8 = mask.astype(np.uint8)
    if thin:
        total = int(u8.sum())
        span0 = _span_px(mask)
        for it in (2, 1):
            m2 = cv2.erode(u8, K3, iterations=it).astype(bool)
            if m2.sum() >= max(15, int(0.40 * total)) and _span_px(m2) >= 0.78 * span0:
                return m2
        return mask
    m2 = cv2.erode(u8, K3, iterations=2).astype(bool)
    return m2 if m2.sum() >= 15 else mask


def _seg_dist(P, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    t = np.clip(((P - a) @ d) / max(float(d @ d), 1e-12), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * d), axis=1)


def _shape(P):
    """Geometry of a cleaned point set (world, metres): footprint centre/extent/yaw, top/low, 3D axis tilt."""
    top, low = float(np.percentile(P[:, 2], 98)), float(np.percentile(P[:, 2], 2))
    xy = P[:, :2]
    c0 = xy.mean(0)
    w, v = np.linalg.eigh(np.cov((xy - c0).T) + 1e-12 * np.eye(2))
    major, minor = v[:, 1], v[:, 0]
    yaw = math.degrees(math.atan2(major[1], major[0]))
    yaw = (yaw + 90.0) % 180.0 - 90.0
    a1, a2 = (xy - c0) @ major, (xy - c0) @ minor
    l_lo, l_hi = np.percentile(a1, [2, 98])
    w_lo, w_hi = np.percentile(a2, [2, 98])
    centre_xy = c0 + major * (l_lo + l_hi) / 2 + minor * (w_lo + w_hi) / 2
    w3, v3 = np.linalg.eigh(np.cov((P - P.mean(0)).T) + 1e-12 * np.eye(3))
    return dict(centre_xy=[float(x) for x in centre_xy], top_z_m=top, low_z_m=low, yaw_deg=float(yaw),
                length_m=float(l_hi - l_lo), width_m=float(w_hi - w_lo),
                axis_tilt_deg=float(math.degrees(math.acos(min(1.0, abs(v3[2, 2]))))),
                elongation=float(math.sqrt(max(w3[2], 0) / max(w3[1], 1e-12))))


def object_stats(rec, mask, support_z, T, K, floating=None, *, explicit=False, return_points=False):
    """VGGT points of one mask -> geometry + quality (no names)."""
    m2 = _erode(mask)
    conf = rec["conf"]
    thin_explicit = bool(explicit and int((m2 & (rec["X"][..., 2] > support_z + ABOVE)).sum()) < 45)
    if thin_explicit:
        m2 = _erode(mask, thin=True)
    sel = m2 & (conf > rec["conf_p10"])
    if sel.sum() < 5:
        sel = m2
    if thin_explicit and sel.sum() < 5:
        sel = mask
    P = rec["X"][sel].astype(np.float64)
    if len(P) < 5:
        return None
    above = P[:, 2] > support_z + ABOVE
    if thin_explicit:
        if above.mean() >= 0.5 and above.sum() >= 20:
            P = P[above]
    elif above.sum() >= 10:
        P = P[above]
    rng = np.random.default_rng(0)
    if len(P) > 4000:
        P = P[rng.choice(len(P), 4000, replace=False)]
    P_main, cluster_frac, _ = voxel_connected_cluster(P, voxel_m=0.01, min_samples=5)
    if cluster_frac > 0.0 and (not explicit or len(P_main) >= 5):
        P = P_main
    if len(P) < 5:
        return None
    shape = _shape(P)
    # floating point set (lowest visible point well above the table): the same points rescaled along the camera rays
    # so that the lowest one lies on the table plane (depth offset under the mask; the shape is kept)
    reanchored = None
    if floating is not None and shape["low_z_m"] - support_z > floating:
        C = T[:3, 3]
        lam = (support_z - C[2]) / (shape["low_z_m"] - C[2])
        if 0 < lam < 10:
            reanchored = dict(_shape(C + lam * (P - C)), depth_scale=float(lam))
    # depth confidence of the mask pixels as a rank within this image (0 = least confident pixel of the image)
    cm = conf[m2]
    flat = np.sort(conf.reshape(-1))
    rank = np.searchsorted(flat, cm) / flat.size
    # contact consistency: the lowest mask pixels back-projected onto the support plane should not lie farther from
    # the camera than the object's estimated centre by more than a margin; a centre much nearer the camera than its
    # own contact means the depth under the mask places the object too high.
    ys, xs = np.nonzero(mask)
    vmax = ys.max()
    bot = xs[ys >= vmax - 2]
    ub, vb = float(np.median(bot)) + 0.5, float(vmax) + 0.5
    d = T[:3, :3] @ np.array([(ub - K[0, 2]) / K[0, 0], (vb - K[1, 2]) / K[1, 1], 1.0])
    contact = None
    if d[2] < -1e-6:
        s = (support_z - T[2, 3]) / d[2]
        contact = (T[:3, 3] + s * d)[:2]
    fwd = T[:3, 2][:2]
    fwd = fwd / max(np.linalg.norm(fwd), 1e-9)
    gap = (lambda xy: float(fwd @ (np.asarray(xy) - contact)) if contact is not None else None)
    if reanchored is not None:
        reanchored["contact_gap_m"] = gap(reanchored["centre_xy"])
    out = dict(shape, contact_gap_m=gap(shape["centre_xy"]), reanchored=reanchored,
               n_points=int(len(P)), cluster_frac=cluster_frac, conf_rank_p50=float(np.median(rank)),
               conf_rank_p10=float(np.percentile(rank, 10)), conf_p50=float(np.median(cm)), mask_px=int(mask.sum()))
    if return_points:
        out["_clean_points"] = P
    return out


def relief(rec, mask, support_z):
    """p75 height of the mask minus the median height of a ring around it (both VGGT, metres)."""
    m = mask.astype(np.uint8)
    ring = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=6).astype(bool) & ~cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=2).astype(bool)
    h = rec["X"][..., 2] - support_z
    if ring.sum() < 10:
        return None
    return float(np.percentile(h[_erode(mask)], 75) - np.median(h[ring]))


def _contour(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return []
    c = max(cs, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2)
    if len(c) > 40:
        c = c[np.linspace(0, len(c) - 1, 40).astype(int)]
    return c.astype(int).tolist()


def _core(mask):
    """Largest 8-connected component of a mask (the part that carries the object's points)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask
    return lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))


def _label_px(mask):
    dt = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    v, u = np.unravel_index(int(dt.argmax()), dt.shape)
    return [int(u), int(v)]


def sam31_segment(image_b64, phrases, cfg, H, W, timeout_s=60.0):
    """SAM 3.1 text prompts on the second process -> [(mask HxW bool, phrase index, score)], per-phrase counts."""
    body = dict(image_b64=image_b64, phrases=phrases, threshold=float(cfg["label_thr"]), max_per_phrase=int(cfg["label_max"]))
    ans = _sam31_post_json("/segment", body, timeout_s)
    if ans.get("status") != "ok":
        raise RuntimeError(str(ans.get("error", "sam31_error"))[:200])
    if list(ans["image_size"]) != [W, H]:
        raise RuntimeError(f"sam31_image_size {ans['image_size']} != {[W, H]}")
    out, counts = [], []
    for li, ph in enumerate(ans["phrases"]):
        counts.append(len(ph["instances"]))
        for inst in ph["instances"]:
            m = np.unpackbits(np.frombuffer(base64.b64decode(inst["mask_b64"]), np.uint8))[:H * W].reshape(H, W).astype(bool)
            out.append((m, li, float(inst["score"])))
    return out, counts, ans.get("timing"), ans.get("gpu_peak_allocated_mb")


def sam31_track(
    session_id,
    image_b64,
    phrases,
    generic_phrases,
    points,
    reanchor,
    cfg,
    H,
    W,
    min_next_track_num=None,
    timeout_s=60.0,
):
    """调用 `:8333/track` 获取跨帧持久身份掩码、短语统计与可选点提示分割结果。"""
    has_explicit = bool(len(phrases) > len(generic_phrases) or reanchor)
    body = dict(
        session_id=session_id,
        image_b64=image_b64,
        phrases=phrases,
        generic_phrases=generic_phrases,
        points=[[float(u), float(v)] for u, v in points],
        threshold=float(cfg["label_thr"]),
        max_per_phrase=int(cfg["label_max"]),
        dedup_iou=float(cfg["label_iou"]),
        min_area=int(min(cfg["min_area"], 8)) if has_explicit else int(cfg["min_area"]),
        max_area_frac=float(max(cfg["text_max_area_frac"], 0.25)) if has_explicit else float(cfg["text_max_area_frac"]),
    )
    if reanchor is not None:
        body["reanchor"] = reanchor
    if min_next_track_num is not None:
        body["min_next_track_num"] = int(min_next_track_num)
    ans = _sam31_post_json("/track", body, timeout_s)
    if ans.get("status") != "ok":
        raise RuntimeError(str(ans.get("error", "sam31_track_error"))[:200])
    if list(ans["image_size"]) != [W, H]:
        raise RuntimeError(f"sam31_image_size {ans['image_size']} != {[W, H]}")

    tracked_insts = []
    for inst in ans.get("instances") or []:
        m = np.unpackbits(np.frombuffer(base64.b64decode(inst["mask_b64"]), np.uint8))[: H * W].reshape(H, W).astype(bool)
        tnum = int(inst["track_num"])
        tracked_insts.append(
            dict(
                mask=m,
                track_num=tnum,
                track_id=str(inst.get("track_id") or f"o{tnum}"),
                track_conf=float(inst.get("track_conf", inst.get("score", 0.8))),
                track_status=str(inst.get("track_status") or "tracked"),
                score=float(inst.get("score", inst.get("track_conf", 0.8))),
                labels=[[str(x[0]), float(x[1])] for x in (inst.get("labels") or [])],
            )
        )

    ph_map = {str(ph.get("phrase")): int(ph.get("n_instances", 0)) for ph in (ans.get("phrases") or [])}
    counts = [ph_map.get(p, 0) for p in phrases]

    pt_out = []
    for pt_res in ans.get("points") or []:
        ms = [
            np.unpackbits(np.frombuffer(base64.b64decode(m["mask_b64"]), np.uint8))[: H * W].reshape(H, W).astype(bool)
            for m in pt_res["masks"]
        ]
        masks = np.stack(ms, axis=0) if ms else np.zeros((0, H, W), bool)
        ious = np.asarray([float(m["iou"]) for m in pt_res["masks"]], np.float32)
        pt_out.append((masks, ious))

    return tracked_insts, counts, pt_out, ans.get("tracking") or {}, ans.get("timing"), ans.get("gpu_peak_allocated_mb")


def object_table(req):
    t_all = time.time()
    data, K, T, support_z = _parse_cam_inputs(req)
    cfg = dict(OT, **{k: v for k, v in (req.get("params") or {}).items() if k in OT})
    max_objects = int(req.get("max_objects", 20))
    max_h = float(req.get("max_object_height_m", cfg["max_object_height"]))
    segments = [np.asarray(s, float).reshape(2, 3) for s in req.get("robot_segments_m") or []]
    radius = float(req.get("robot_radius_m", cfg["robot_radius"]))
    exclude = [[int(v) for v in b] for b in req.get("exclude_boxes_px") or []]
    session_id = str(req.get("session_id") or "").strip() or None
    track_reanchor = req.get("reanchor") if isinstance(req.get("reanchor"), dict) else None
    if session_id and track_reanchor and bool(track_reanchor.get("reset")) and not track_reanchor.get("target_track_id"):
        with CLOCK:
            SESSION_MAX_TRACK_NUM.pop(session_id, None)
            SESSION_KNOWN_TRACKS.pop(session_id, None)

    _reset_peak_mem()
    entry_key = image_key(data, K, T, support_z)
    entry, hit = _entry(data, K, T, support_z)
    rec = entry["rec"]
    H, W = rec["hw"]
    # robot links drawn into the image (capsules of the same radius projected with the calibration): a mask counts as
    # robot only when it is near the links both in 3D (VGGT points) and in the image, so a depth error under an object
    # cannot move it onto a link
    robot_img = np.zeros((H, W), np.uint8)
    Ti = np.linalg.inv(T)
    for a, b in segments:
        for q in np.linspace(0.0, 1.0, 12):
            pc = Ti[:3, :3] @ (a + q * (b - a)) + Ti[:3, 3]
            if pc[2] <= 0.05:
                continue
            u, v = K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]
            rad = int(np.ceil(K[0, 0] * radius / pc[2]))
            if -rad <= u < W + rad and -rad <= v < H + rad:
                cv2.circle(robot_img, (int(u), int(v)), rad, 1, -1)
    robot_img = robot_img.astype(bool)
    # hp: a caller-supplied robot pixel mask (FK render of the robot meshes, already dilated) is used by held_estimate
    # to strip gripper fingers from held objects; table objects keep the 3D+2D segment check when segments are supplied
    pix_mask = _robot_mask(req, H, W)
    if pix_mask is not None and not segments:
        robot_img = pix_mask
    # caller-supplied phrases -> SAM 3.1 text-prompted instances (second process); failure falls back to the SAM 2.1 grid
    labels = [str(x).strip()[:80] for x in (req.get("labels") or []) if str(x).strip()][:16]
    auto_phrase = str(cfg["auto_phrase"] or "").strip()
    phrases = labels + ([auto_phrase] if auto_phrase and auto_phrase not in labels else [])
    generic_phrases = [auto_phrase] if auto_phrase else []
    req_pixels = np.asarray(req.get("pixels") or [], np.float64).reshape(-1, 2)
    valid_req_pts = [(float(u), float(v)) for u, v in req_pixels if 0 <= u < W and 0 <= v < H]
    timeout_s = float(req.get("timeout_s", 60))

    persistent_ids = False
    tracked_insts, track_meta, track_fallback_err = [], None, None
    pt_masks_list = None
    text_cands, text_counts, text_err, text_timing, text_peak = [], [], None, None, None

    t = time.time()
    if session_id and cfg["seg_backend"] == "sam31" and bool(phrases or track_reanchor):
        with CLOCK:
            min_next = SESSION_MAX_TRACK_NUM.get(session_id, 0) + 1 if session_id in SESSION_MAX_TRACK_NUM else None
            has_known_tracks = bool(SESSION_KNOWN_TRACKS.get(session_id))
        try:
            tracked_insts, text_counts, pt_masks_list, track_meta, text_timing, text_peak = sam31_track(
                session_id,
                req["image_b64"],
                phrases,
                generic_phrases,
                valid_req_pts,
                track_reanchor,
                cfg,
                H,
                W,
                min_next_track_num=min_next,
                timeout_s=timeout_s,
            )
            persistent_ids = True
            next_from_svc = int(track_meta.get("next_track_num", 1)) - 1
            if next_from_svc > 0:
                with CLOCK:
                    SESSION_MAX_TRACK_NUM[session_id] = max(SESSION_MAX_TRACK_NUM.get(session_id, 0), next_from_svc)
        except Exception as ex:
            track_fallback_err = f"{type(ex).__name__}: {ex}"[:200]
            # 会话已有持久轨迹或 SAM 3.1 服务不可达/5xx 时严禁静默降级为无状态 o1..oN 洗牌
            if has_known_tracks or isinstance(ex, UpstreamServiceError):
                raise UpstreamServiceError(f"sam31_track_failed: {track_fallback_err}") from ex
            persistent_ids = False
            pt_masks_list = None

    use_text = bool(phrases) and cfg["seg_backend"] == "sam31"
    if not persistent_ids and use_text:
        try:
            text_cands, text_counts, text_timing, text_peak = sam31_segment(
                req["image_b64"], phrases, cfg, H, W, timeout_s
            )
        except Exception as ex:
            text_err = f"{type(ex).__name__}: {ex}"[:200]
            use_text = False
    if cfg["seg_backend"] == "sam31" and not persistent_ids and not use_text and (track_fallback_err or text_err):
        raise UpstreamServiceError(f"sam31_unavailable: {text_err or track_fallback_err}")
    t_text = time.time() - t
    if persistent_ids:
        backend = "sam31_track" + (":labels" if labels else "") + (f":{auto_phrase}" if auto_phrase else "")
    else:
        backend = ("sam31_text" + (":labels" if labels else "") + (f":{auto_phrase}" if auto_phrase else "")) if use_text else "sam31_points"
    t_embed = 0.0
    h = rec["X"][..., 2] - support_z
    above = (h > cfg["min_above"]) & (h < cfg["bg_height"]) & (rec["conf"] > rec["conf_p10"])
    grid = []
    border_np = np.zeros((H, W), dtype=bool)
    border_np[:2, :] = True; border_np[-2:, :] = True; border_np[:, :2] = True; border_np[:, -2:] = True
    exclude_np = np.zeros((H, W), dtype=bool)
    for x0, y0, x1, y1 in exclude:      # a mask reaching into an excluded box is treated like one touching the border
        border_np[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
        exclude_np[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
    reject = collections.Counter()
    t_sam = 0.0
    chosen, dropped = [], collections.Counter()
    n_cands = 0
    t = time.time()

    def screen(mask, *, explicit=False):
        """Object geometry of one mask, or (None, reason, occupy) for support / background / fragmented / robot masks."""
        if pix_mask is not None and not segments:
            if pix_mask[mask].mean() >= cfg["robot_frac"]:
                return None, "robot", True
        rel = relief(rec, mask, support_z)          # cheap test first: plain table texture has no relief
        screen_flags = []
        if rel is not None and rel < cfg["min_relief"]:
            if not explicit:
                return None, "no_relief", False
            screen_flags.append("low_relief")
        st = object_stats(rec, mask, support_z, T, K, cfg["floating"], explicit=explicit, return_points=True)
        if st is None:
            return None, "no_points", False
        if st["top_z_m"] - support_z > cfg["bg_height"]:
            return None, "background_height", True
        if st["cluster_frac"] < cfg["cluster_frac"]:
            if not explicit:
                return None, "fragmented", False
            screen_flags.append("fragmented")
        st["relief_m"] = rel
        if screen_flags:
            st["screen_flags"] = screen_flags
        if segments:
            Pm = rec["X"][_erode(mask)].astype(np.float64)
            near = np.min([_seg_dist(Pm, a, b) for a, b in segments], axis=0) < radius
            if near.mean() >= cfg["robot_frac"] and robot_img[mask].mean() >= cfg["robot_frac"]:
                return None, "robot", True
        return st, None, True

    label_drop = [collections.Counter() for _ in labels]
    label_idx_map = {lab_name: idx_l for idx_l, lab_name in enumerate(labels)}

    if persistent_ids:
        # 持久跨帧模式：具名或显式重锚定实例优先筛选，严格锁定原始 track_num / track_id 身份
        ordered_tracked = sorted(
            tracked_insts,
            key=lambda it: (it["track_status"] != "reanchored" and not bool(it["labels"]), it["track_num"]),
        )
        for inst in ordered_tracked:
            m_np = inst["mask"]
            inst_labs = [[nm, round(float(sc), 3)] for nm, sc in inst["labels"] if nm in label_idx_map]
            is_reanchored = inst["track_status"] == "reanchored"
            generic = (not inst_labs) and (not is_reanchored)
            area = float(m_np.sum())
            min_a = cfg["min_area"] if generic else min(cfg["min_area"], 8)
            max_a = (cfg["text_max_area_frac"] if generic else max(cfg["text_max_area_frac"], 0.25)) * H * W
            why = (
                "small" if area < min_a
                else "large" if area > max_a
                else "not_above_support" if (generic and (m_np & above).sum() / max(area, 1.0) < cfg["above_frac"])
                else "border_or_excluded" if ((m_np & border_np).any() if generic else (m_np & exclude_np).any())
                else None
            )
            if why is None:
                dup, best_j = None, 0.0
                for i, c in enumerate(chosen):
                    inter = float((m_np & c["mask"]).sum())
                    if inter:
                        j = inter / float((m_np | c["mask"]).sum())
                        if j >= cfg["label_iou"] and j > best_j:
                            dup, best_j = i, j
                if dup is None and generic and any((m_np & c["mask"]).sum() > cfg["part_frac"] * area for c in chosen):
                    dropped["text_part_of_chosen"] += 1
                    continue
                if dup is not None:
                    if not generic:
                        lab = chosen[dup].setdefault("labels", [])
                        for nm, sc in inst_labs:
                            if nm not in [x[0] for x in lab]:
                                lab.append([nm, sc])
                    continue
                st, why, _ = screen(m_np, explicit=not generic)
                if st is not None:
                    if not generic and (m_np & above).sum() / max(area, 1.0) < cfg["above_frac"]:
                        st.setdefault("screen_flags", []).append("thin_or_on_support")
                    if not generic and (m_np & border_np).any():
                        st.setdefault("screen_flags", []).append("touches_border")
                    chosen.append(
                        dict(
                            mask=m_np,
                            iou_pred=inst["score"],
                            stability=None,
                            auto=False,
                            text=True,
                            labels=inst_labs,
                            track_num=int(inst["track_num"]),
                            track_id=inst["track_id"],
                            track_conf=round(float(np.clip(inst["track_conf"], 0.0, 1.0)), 4),
                            track_status=inst["track_status"],
                            **st,
                        )
                    )
                    continue
            for nm, _ in inst_labs:
                label_drop[label_idx_map[nm]][why] += 1
            dropped["text_" + str(why)] += 1
    else:
        # text-prompted instances, highest score first: same mask filters (text_max_area_frac for size),
        # same screening; an instance that is the same mask as a chosen object (IoU >= label_iou) only adds its phrase
        for m_np, li, score in sorted(text_cands, key=lambda c: (c[1] >= len(labels), -c[2])):   # labelled first
            generic = li >= len(labels)
            area = float(m_np.sum())
            min_a = cfg["min_area"] if generic else min(cfg["min_area"], 8)
            max_a = (cfg["text_max_area_frac"] if generic else max(cfg["text_max_area_frac"], 0.25)) * H * W
            why = ("small" if area < min_a else "large" if area > max_a
                   else "not_above_support" if (generic and (m_np & above).sum() / max(area, 1.0) < cfg["above_frac"])
                   else "border_or_excluded" if ((m_np & border_np).any() if generic else (m_np & exclude_np).any()) else None)
            if why is None:
                dup, best_j = None, 0.0
                for i, c in enumerate(chosen):
                    inter = float((m_np & c["mask"]).sum())
                    if inter:
                        j = inter / float((m_np | c["mask"]).sum())
                        if j >= cfg["label_iou"] and j > best_j:
                            dup, best_j = i, j
                if dup is None and generic and any((m_np & c["mask"]).sum() > cfg["part_frac"] * area for c in chosen):
                    dropped["text_part_of_chosen"] += 1
                    continue
                if dup is not None:
                    if generic:
                        continue
                    lab = chosen[dup].setdefault("labels", [])
                    if labels[li] not in [x[0] for x in lab]:
                        lab.append([labels[li], round(score, 3)])
                    continue
                st, why, _ = screen(m_np, explicit=not generic)
                if st is not None:
                    if not generic and (m_np & above).sum() / max(area, 1.0) < cfg["above_frac"]:
                        st.setdefault("screen_flags", []).append("thin_or_on_support")
                    if not generic and (m_np & border_np).any():
                        st.setdefault("screen_flags", []).append("touches_border")
                    chosen.append(dict(mask=m_np, iou_pred=score, stability=None, auto=False, text=True,
                                       labels=[] if generic else [[labels[li], round(score, 3)]], **st))
                    continue
            if not generic:
                label_drop[li][why] += 1
            dropped["text_" + why] += 1

    # model-requested pixels: SAM 3.1 point prompt each; reuse an existing object only when the pixel lies in its mask core and
    # the point-prompt mask is the same object (IoU >= reuse_iou); otherwise the point-prompt mask is a new object
    requested = []
    if pt_masks_list is None:
        t_pt0 = time.time()
        pt_masks_list = sam31_points(req["image_b64"], valid_req_pts, H, W, timeout_s) if valid_req_pts else []
        t_sam += time.time() - t_pt0
    pt_idx = 0
    for u, v in req_pixels:
        if not (0 <= u < W and 0 <= v < H):
            requested.append(dict(pixel=[float(u), float(v)], status="pixel_out_of_bounds"))
            continue
        ui, vi = int(u), int(v)
        masks, iou = pt_masks_list[pt_idx]
        pt_idx += 1
        pmask, q = pick_mask(masks, iou, u, v)
        found, best, why = None, 0.0, None
        for i, c in enumerate(chosen):
            if not c["mask"][vi, ui]:
                continue
            if "core" not in c:
                c["core"] = _core(c["mask"])
            if not c["core"][vi, ui]:
                why = "pixel_in_detached_part"
                continue
            if pmask is None:        # no usable point mask: the core containment alone decides
                found, best = i, None
                break
            j = float((pmask & c["mask"]).sum()) / max(float((pmask | c["mask"]).sum()), 1.0)
            if j >= cfg["reuse_iou"] and j > best:
                found, best = i, j
            elif found is None:
                why = "low_iou"
        if found is not None:
            chosen[found].setdefault("requested", []).append([float(u), float(v)])
            requested.append(dict(pixel=[float(u), float(v)], status="ok", index=found, match="reused",
                                  reuse_iou=None if best is None else round(best, 3)))
            continue
        if pmask is not None and pix_mask is not None and not segments:
            if pix_mask[pmask].mean() >= cfg["robot_frac"]:
                pmask = None
        st = object_stats(rec, pmask, support_z, T, K, cfg["floating"], explicit=True, return_points=True) if pmask is not None else None
        if st is None:
            requested.append(dict(pixel=[float(u), float(v)], status="no_object_mask"))
            continue
        st["relief_m"] = relief(rec, pmask, support_z)
        pt_extra = {}
        if persistent_ids and session_id:
            with CLOCK:
                new_tid = max([SESSION_MAX_TRACK_NUM.get(session_id, 0)] + [int(c.get("track_num", 0)) for c in chosen]) + 1
                SESSION_MAX_TRACK_NUM[session_id] = new_tid
            pt_extra = dict(
                track_num=new_tid,
                track_id=f"o{new_tid}",
                track_conf=round(float(np.clip(q if q is not None else 0.8, 0.0, 1.0)), 4),
                track_status="new",
            )
        chosen.append(dict(mask=pmask, iou_pred=q, stability=None, auto=False, point_only=True,
                           requested=[[float(u), float(v)]], **pt_extra, **st))
        requested.append(dict(pixel=[float(u), float(v)], status="ok", index=len(chosen) - 1, match="new_point_mask",
                              **({"not_reused": why} if why else {})))
    if DEBUG_DIR:
        np.savez_compressed(os.path.join(DEBUG_DIR, f"ot_{hashlib.sha256(data).hexdigest()[:12]}_{int(time.time()*1000)}.npz"),
                            masks=np.stack([c["mask"] for c in chosen]) if chosen else np.zeros((0, H, W), bool),
                            auto=np.array([c["auto"] for c in chosen]), requested=json.dumps(requested))
    # cap: requested objects always kept; then by predicted mask quality
    order = sorted(
        range(len(chosen)),
        key=lambda i: (
            not chosen[i].get("requested"),
            chosen[i].get("track_status") != "reanchored",
            not chosen[i].get("labels"),
            -(chosen[i]["iou_pred"] or 0),
        ),
    )
    cap_n = max(max_objects, sum(1 for c in chosen if c.get("requested")))
    if persistent_ids:
        keep = sorted(order[:cap_n], key=lambda i: int(chosen[i]["track_num"]))
    else:
        keep = sorted(order[:cap_n])
    omitted = len(chosen) - len(keep)
    remap = {old: new for new, old in enumerate(keep)}
    chosen = [chosen[i] for i in keep]
    for r in requested:
        if "index" in r:
            r["index"] = remap.get(r["index"])
    by_index = collections.defaultdict(list)
    for k, r in enumerate(requested):
        if r.get("index") is not None:
            by_index[r["index"]].append(k)
    for ks in by_index.values():
        if len(ks) > 1:
            for k in ks:
                requested[k]["status"] = "ok_same_object"
                requested[k]["same_object_as"] = [j for j in ks if j != k]   # positions in the request's pixels list
    # support relation and bottom height
    for i, c in enumerate(chosen):
        c["support"], c["bottom_z_m"] = "unknown", c["low_z_m"]
        if c["low_z_m"] - support_z <= cfg["support_gap"]:
            c["support"], c["bottom_z_m"] = "table", support_z
            continue
        best = None
        c_area = c["length_m"] * c["width_m"]
        for j, d in enumerate(chosen):
            if j == i or d["top_z_m"] <= support_z + 0.01:
                continue
            on_top = (d["top_z_m"] < c["low_z_m"] + cfg["support_gap"]
                      and d["top_z_m"] < c["top_z_m"] - 0.004
                      and d["low_z_m"] < c["low_z_m"])
            in_container = (c["low_z_m"] <= d["top_z_m"] + cfg["support_gap"]
                            and c["top_z_m"] >= d["low_z_m"] - 0.01
                            and d["length_m"] * d["width_m"] > c_area * 1.25)
            if not (on_top or in_container):
                continue
            rel = np.asarray(c["centre_xy"]) - np.asarray(d["centre_xy"])
            ax = np.array([math.cos(math.radians(d["yaw_deg"])), math.sin(math.radians(d["yaw_deg"]))])
            inside = (abs(rel @ ax) <= d["length_m"] / 2 + 0.015 and abs(rel @ np.array([-ax[1], ax[0]])) <= d["width_m"] / 2 + 0.015)
            if inside and (best is None or d["top_z_m"] > chosen[best]["top_z_m"]):
                best = j
        if best is not None:
            c["support"] = best
            c["bottom_z_m"] = chosen[best]["top_z_m"] if (
                chosen[best]["top_z_m"] < c["low_z_m"] + cfg["support_gap"]
                and chosen[best]["top_z_m"] < c["top_z_m"] - 0.004
            ) else c["low_z_m"]
    for c in chosen:
        c["floating"] = c["support"] == "unknown" and c["low_z_m"] - support_z > cfg["floating"]
        rescale = c["floating"] and cfg["reanchor"] and c.get("reanchored") and not c.get("point_only")
        if rescale and cfg["reanchor_mode"] == "thickness":
            rescale = c["low_z_m"] - support_z > (c["top_z_m"] - c["low_z_m"]) + float(cfg["reanchor_margin"])
        if rescale:
            raw = {k: c[k] for k in ("centre_xy", "top_z_m", "low_z_m", "yaw_deg", "length_m", "width_m", "axis_tilt_deg",
                                     "elongation", "contact_gap_m")}
            c.update({k: v for k, v in c["reanchored"].items() if k != "depth_scale"})
            c["raw"], c["support"], c["bottom_z_m"] = raw, "table", support_z
    t_held = time.time()
    held_out = [held_estimate(entry, req["image_b64"], K, T, support_z, pix_mask, item, [c["mask"] for c in chosen])
                for item in (req.get("held") or [])[:2]] if req.get("held") else None
    t_held = time.time() - t_held
    objects = []
    instance_masks_list = []
    sorted_labs = [sorted(c.get("labels") or [], key=lambda x: -x[1]) for c in chosen]
    primary_counts = collections.Counter(lab[0][0] for lab in sorted_labs if lab)
    for idx_c, lab in enumerate(sorted_labs):
        if len(lab) > 1 and primary_counts[lab[0][0]] > 1:
            for alt_idx in range(1, len(lab)):
                alt_p, _ = lab[alt_idx]
                if primary_counts[alt_p] == 0:
                    primary_counts[lab[0][0]] -= 1
                    primary_counts[alt_p] += 1
                    sorted_labs[idx_c] = [lab[alt_idx], *lab[:alt_idx], *lab[alt_idx + 1:]]
                    break
    cam_origin = T[:3, 3]
    for idx_c, c in enumerate(chosen):
        flags = list(c.pop("screen_flags", None) or [])
        if c["floating"]:
            flags.append("floating_rescaled_to_table" if c.get("raw") else "floating")
        if c["top_z_m"] - support_z > max_h:
            flags.append("height_above_max")
        if c["conf_rank_p50"] < cfg["low_conf_rank"]:
            flags.append("low_depth_confidence")
        if (not c["floating"] and c["contact_gap_m"] is not None and not isinstance(c["support"], int)
                and c["contact_gap_m"] < -cfg["contact_margin"]):
            flags.append("contact_mismatch")
        mask = c.pop("mask")
        reanc_info = c.pop("reanchored", None)
        clean_pts = c.pop("_clean_points", None)
        c.pop("core", None)
        applied_scale = float(reanc_info["depth_scale"]) if (c.get("raw") and reanc_info and "depth_scale" in reanc_info) else 1.0
        if clean_pts is not None and abs(applied_scale - 1.0) > 1e-9:
            clean_pts = cam_origin + applied_scale * (clean_pts - cam_origin)

        cavity_depth_m = compute_cavity_depth(
            rec["X"],
            rec["conf"],
            rec["conf_p10"],
            mask,
            cam_origin=cam_origin,
            depth_scale=applied_scale,
        )
        surface_slices_m = compute_surface_slices(clean_pts, max_slices=3, max_vertices=12) if clean_pts is not None else []

        if persistent_ids:
            t_num = int(c.pop("track_num"))
            t_id = str(c.pop("track_id"))
            t_conf = round(float(np.clip(c.pop("track_conf"), 0.0, 1.0)), 4)
            t_status = str(c.pop("track_status"))
        else:
            c.pop("track_num", None)
            c.pop("track_id", None)
            c.pop("track_conf", None)
            c.pop("track_status", None)
            t_num = idx_c + 1
            t_id = f"o{t_num}"
            raw_q = c.get("iou_pred")
            t_conf = round(float(np.clip(raw_q if raw_q is not None else 0.8, 0.0, 1.0)), 4)
            t_status = "stateless"

        instance_masks_list.append((int(t_num), mask))
        cols, rows = np.nonzero(mask.any(0))[0], np.nonzero(mask.any(1))[0]
        lab = sorted_labs[idx_c]
        objects.append(
            dict(
                c,
                track_id=t_id,
                track_conf=t_conf,
                track_status=t_status,
                cavity_depth_m=cavity_depth_m,
                surface_slices_m=surface_slices_m,
                centre_m=[*c["centre_xy"], (c["top_z_m"] + c["bottom_z_m"]) / 2],
                height_m=c["top_z_m"] - c["bottom_z_m"],
                flags=flags,
                quality="low" if flags else "ok",
                label_px=_label_px(mask),
                contour_px=_contour(mask),
                labels=lab,
                label=lab[0][0] if lab else None,
                label_score=lab[0][1] if lab else None,
                bbox_px=[int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])],
            )
        )

    # 在当前图像缓存 entry["instance_masks"] 保存持久物体掩码列表，供紧随其后的同图 /points 复用
    with CLOCK:
        entry["instance_masks"] = instance_masks_list
        if session_id:
            _record_session_masks(session_id, entry_key, entry, instance_masks_list, persistent_ids=persistent_ids)
        active_tracks = len(SESSION_KNOWN_TRACKS.get(session_id, ())) if (persistent_ids and session_id) else len(objects)

    obj_confs = [float(o["track_conf"]) for o in objects]
    low_conf_ids = [o["track_id"] for o in objects if o["track_conf"] < 0.55]
    tracking_out = dict(
        session_id=session_id,
        persistent_ids=bool(persistent_ids),
        min_track_conf=round(min(obj_confs), 4) if obj_confs else None,
        low_conf_ids=low_conf_ids,
        active_tracks=max(active_tracks, len(objects)),
        **({"fallback_error": track_fallback_err} if track_fallback_err else {}),
    )

    # per phrase: which returned objects carry it (an object can carry several phrases; a phrase can match several objects)
    label_report = []
    for li, p in enumerate(labels):
        idx = [i for i, o in enumerate(objects) if p in [x[0] for x in o["labels"]]]
        n_inst = text_counts[li] if li < len(text_counts) else 0
        status = ("backend_unavailable" if text_err else "unsupported_backend" if cfg["seg_backend"] != "sam31"
                  else "ok" if idx else "no_match" if n_inst == 0 else "filtered")
        label_report.append(dict(label=p, status=status, indices=idx, n_instances=n_inst,
                                 **({"filtered": dict(label_drop[li])} if label_drop and label_drop[li] else {})))
    t_obj = time.time() - t
    total = time.time() - t_all
    peak = _peak_mem_mb()
    STATS["peak_mb"] = max(STATS["peak_mb"], peak)
    STATS["total_s"].append(total)
    diag_err = text_err or track_fallback_err
    return dict(status="ok", version=OT_VERSION, source="vggt_estimate", image_sha256=hashlib.sha256(data).hexdigest(),
                caps=list(SERVICE_CAPS),
                cache_hit=hit, image_size=[W, H], scale=rec["s"], support_fit=rec["table"], support_z_m=support_z,
                objects=objects, tracking=tracking_out, requested=requested, omitted=omitted, dropped=dict(dropped),
                rejected=dict(reject), n_prompts=len(grid), n_candidates=n_cands,
                params=dict(cfg, max_objects=max_objects, max_object_height=max_h),
                backend=backend, labels=label_report, **({"backend_error": diag_err} if diag_err else {}),
                **({"held": held_out, "held_s": t_held} if held_out is not None else {}),
                **({"robot_mask": "pixels"} if pix_mask is not None else {}),
                timing=dict(vggt_s=None if hit else rec["vggt_s"], embed_s=t_embed, sam_s=t_sam, text_s=t_text,
                            sam31=text_timing, objects_s=t_obj, total_s=total),
                gpu_peak_allocated_mb=peak, sam31_gpu_peak_allocated_mb=text_peak)


def plausibility(rec, mask, support_z, T, K, top_z, max_h):
    """Flags for /height (pixel_plane plane_source vggt): the same checks as the object table."""
    flags, info = [], {}
    if mask is not None:
        st = object_stats(rec, mask, support_z, T, K)
        if st is not None:
            info = dict(conf_rank_p50=st["conf_rank_p50"], contact_gap_m=st["contact_gap_m"], low_z_m=st["low_z_m"])
            if st["low_z_m"] - support_z > OT["floating"]:
                flags.append("floating")
            if st["conf_rank_p50"] < OT["low_conf_rank"]:
                flags.append("low_depth_confidence")
            if st["contact_gap_m"] is not None and st["contact_gap_m"] < -OT["contact_margin"]:
                flags.append("contact_mismatch")
    if top_z - support_z > max_h:
        flags.append("height_above_max")
    return flags, info


# ------------------------------------------------------------------ hp: held objects (NI-RUA features.held_perception)
# The robot's own kinematics give the grasp centre (FK link6 + R [0.1501, 0, 0]); camera model and FK are exact in this sim,
# VGGT depth of a held object is not (pick_anatomy/cal_grip). Per held item {arm, grasp_centre_m, phrases?}:
#   mask  = SAM 3.1 text instance (caller phrases + "object", threshold text_thr) whose mask contains the projected grasp
#           centre, else the nearest within max_dist_px, minus the robot pixel mask; fallback: SAM 3.1 point prompt at the
#           nearest non-robot pixel to the projected grasp centre;
#   centre (Kfk) = ray through the median mask pixel at the FK grasp-centre camera depth;
#   shape (GK)   = VGGT points of the mask rescaled along the camera rays so that the points imaged within near_px of the
#           grasp centre sit at the FK depth; cleaned (1 cm DBSCAN largest cluster, z99 trim); footprint as _shape.
# Never the floating -> table rescale. No names, no truth.
HELD = dict(text_thr=0.2, max_per_phrase=10, max_dist_px=40.0, min_px=20, max_area_frac=0.25, near_px=25.0,
            min_points=20, point_fallback=1, max_phrases=3, generic="object")


def _robot_mask(req, H, W):
    b64 = req.get("robot_mask_b64")
    if not b64:
        return None
    shape = [int(v) for v in req.get("robot_mask_shape") or [H, W]]
    if shape != [H, W]:
        raise ValueError(f"robot_mask_shape {shape} != {[H, W]}")
    bits = np.unpackbits(np.frombuffer(base64.b64decode(b64), np.uint8))
    if bits.size < H * W:
        raise ValueError("robot_mask_too_short")
    return bits[:H * W].reshape(H, W).astype(bool)


def _held_pick(cands, robot, pu, pv, H, W, cfg):
    best = None
    ui, vi = int(math.floor(pu)), int(math.floor(pv))
    for m, src, phrase, score in cands:
        mm = m & ~robot if robot is not None else m
        n = int(mm.sum())
        if n < cfg["min_px"] or n > cfg["max_area_frac"] * H * W:
            continue
        inside = 0 <= vi < H and 0 <= ui < W and bool(m[vi, ui])
        if inside:
            dd = 0.0
        else:
            yy, xx = np.nonzero(mm)
            dd = float(np.min(np.hypot(xx + 0.5 - pu, yy + 0.5 - pv)))
        key = (dd, -score)
        if dd <= cfg["max_dist_px"] and (best is None or key < best[0]):
            best = (key, mm, src, phrase, score, dd)
    return best


def held_estimate(entry, image_b64, K, T, support_z, robot, item, chosen_masks=()):
    cfg = dict(HELD, **{k: v for k, v in (item.get("params") or {}).items() if k in HELD})
    rec = entry["rec"]
    H, W = rec["hw"]
    out = dict(arm=item.get("arm"), status="no_mask")
    G = np.asarray(item["grasp_centre_m"], float).reshape(3)
    Ti = np.linalg.inv(T)
    pc = Ti[:3, :3] @ G + Ti[:3, 3]
    if pc[2] <= 0.02:
        out["status"] = "grasp_centre_behind_camera"
        return out
    zg = float(pc[2])
    pu, pv = K[0, 0] * pc[0] / zg + K[0, 2], K[1, 1] * pc[1] / zg + K[1, 2]
    out.update(grasp_px=[float(pu), float(pv)], grasp_depth_m=zg)
    t = time.time()
    phrases = [str(p).strip()[:80] for p in (item.get("phrases") or []) if str(p).strip()][:cfg["max_phrases"]]
    if cfg["generic"] and cfg["generic"] not in phrases:
        phrases.append(cfg["generic"])
    cands, err = [], None
    try:
        tc, _, _, _ = sam31_segment(image_b64, phrases, dict(OT, label_thr=cfg["text_thr"], label_max=cfg["max_per_phrase"]),
                                    H, W, float(item.get("timeout_s", 30)))
        cands = [(m, "sam31_text", phrases[li], s) for m, li, s in tc]
    except Exception as ex:
        err = f"{type(ex).__name__}: {ex}"[:160]
    best = _held_pick(cands, robot, pu, pv, H, W, cfg)
    out["text_s"] = time.time() - t
    if best is None and cfg["point_fallback"]:
        # nearest non-robot pixel to the projected grasp centre (within max_dist_px) as a SAM 3.1 point prompt
        # (main has no SAM 2.1 predictor; PR #35 used a SAM 2.1 point prompt here)
        free = ~robot if robot is not None else np.ones((H, W), bool)
        r = int(cfg["max_dist_px"])
        u0, v0 = int(math.floor(pu)), int(math.floor(pv))
        win = free[max(0, v0 - r):max(0, v0 + r + 1), max(0, u0 - r):max(0, u0 + r + 1)]
        if win.any():
            yy, xx = np.nonzero(win)
            xx, yy = xx + max(0, u0 - r), yy + max(0, v0 - r)
            k = int(np.argmin(np.hypot(xx + 0.5 - pu, yy + 0.5 - pv)))
            qu, qv = float(xx[k]) + 0.5, float(yy[k]) + 0.5
            try:
                answers = sam31_points(image_b64, [(qu, qv)], H, W, float(item.get("timeout_s", 30)))
            except Exception as ex:
                answers, err = [], err or f"{type(ex).__name__}: {ex}"[:160]
            pm, q = pick_mask(*answers[0], qu, qv) if answers and len(answers[0][0]) else (None, None)
            if pm is not None:
                best = _held_pick([(pm, "sam31_point", None, float(q))], robot, pu, pv, H, W, cfg)
                out["point_prompt_px"] = [qu, qv]
    if best is None:
        if err:
            out["text_error"] = err
        return out
    _, mm, src, phrase, score, dd = best
    yy, xx = np.nonzero(mm)
    cu, cv_ = float(np.median(xx)) + 0.5, float(np.median(yy)) + 0.5
    C, R = T[:3, 3], T[:3, :3]
    d = np.array([(cu - K[0, 2]) / K[0, 0], (cv_ - K[1, 2]) / K[1, 1], 1.0])
    centre = C + zg * (R @ d)
    cols, rows = np.nonzero(mm.any(0))[0], np.nonzero(mm.any(1))[0]
    out.update(status="ok", mask_source=src, phrase=phrase, score=float(score), mask_px=int(mm.sum()), dist_px=dd,
               centroid_px=[cu, cv_], centre_m=[float(v) for v in centre], contour_px=_contour(mm), label_px=_label_px(mm),
               bbox_px=[int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])],
               border=bool(mm[:2].any() or mm[-2:].any() or mm[:, :2].any() or mm[:, -2:].any()),
               table_fit=bool(isinstance(rec.get("table"), dict) and "error" not in rec["table"]))
    # GK shape
    sel = _erode(mm) & (rec["conf"] > rec["conf_p10"])
    if sel.sum() < cfg["min_points"]:
        sel = mm & (rec["conf"] > rec["conf_p10"])
    vs, us = np.nonzero(sel)
    X = rec["X"][sel].astype(np.float64)
    gk = None
    if len(X) >= cfg["min_points"]:
        zc = (X - C) @ R[:, 2]
        dpx = np.hypot(us + 0.5 - pu, vs + 0.5 - pv)
        near = dpx < cfg["near_px"]
        if near.sum() < 5:
            near = dpx <= np.percentile(dpx, 20)
        kk = zg / float(np.median(zc[near]))
        P = C + kk * (X - C)
        P = P[P[:, 2] > support_z + 0.003] if (P[:, 2] > support_z + 0.003).sum() >= cfg["min_points"] else P
        if len(P) > 4000:
            P = P[np.random.default_rng(0).choice(len(P), 4000, replace=False)]
        P_main, c_frac, _ = voxel_connected_cluster(P, voxel_m=0.01, min_samples=5)
        if c_frac > 0.0:
            P = P_main
        if len(P) >= 5:
            P = P[P[:, 2] <= np.percentile(P[:, 2], 99) + 0.002]
        if len(P) >= cfg["min_points"]:
            sh = _shape(P)
            gk = dict(sh, k=float(kk), n_points=int(len(P)), centre_m=[*sh["centre_xy"], (sh["top_z_m"] + sh["low_z_m"]) / 2])
    out["gk"] = gk
    # the chosen object masks of the same call that are this held mask (IoU >= 0.3 or mostly inside it)
    match, best_j = None, 0.0
    for i, cm in enumerate(chosen_masks):
        inter = float((cm & mm).sum())
        if not inter:
            continue
        j = inter / float((cm | mm).sum())
        if (j >= 0.3 or inter >= 0.5 * float(cm.sum())) and j > best_j:
            match, best_j = i, j
    out["object_index"], out["object_iou"] = match, best_j if match is not None else None
    return out


def held(req):
    """POST /held {image_b64, K, T_world_camera, support_z_m, robot_mask_b64?, robot_mask_shape?, held:[{arm,
    grasp_centre_m, phrases?}]}: held rows of one image (no object table). A missing support plane is tolerated."""
    t_all = time.time()
    data, K, T, support_z = _parse_cam_inputs(req)
    entry, hit = _entry(data, K, T, support_z, allow_no_table=True)
    H, W = entry["rec"]["hw"]
    robot = _robot_mask(req, H, W)
    res = [held_estimate(entry, req["image_b64"], K, T, support_z, robot, item) for item in (req.get("held") or [])[:2]]
    total = time.time() - t_all
    STATS["total_s"].append(total)
    return dict(status="ok", version=OT_VERSION, cache_hit=hit, image_size=[W, H], scale=entry["rec"]["s"],
                held=res, timing=dict(vggt_s=None if hit else entry["rec"]["vggt_s"], total_s=total))


# ------------------------------------------------------------------ scene3d points (NI-RUA features.scene3d)
# POST /points {image_b64, K, T_world_camera, support_z_m, voxel_m?, robot_segments_m?, robot_radius_m?, params?, timeout_s?}
# Same reconstruction and cache as /height: every pixel's VGGT point (calibrated K + T, scaled to the support plane),
# kept within [support_z - z_below, support_z + z_above] and max_range of the camera, one point per voxel (the most
# confident pixel of that voxel). Returned as npz (base64): xyz float32 (N,3) world frame, conf uint8 (depth-confidence
# rank of that pixel in the image, 0..255 = 0..1), uv uint16 (N,2) source pixel (column, row), count uint16 pixels in
# the voxel, robot bool (within robot_radius of the robot link segments in 3D). No per-object correction.
PTS_VERSION = "scene3d-points-v1"
PTS = dict(voxel=0.01, z_below=0.05, z_above=0.60, max_range=3.0, robot_radius=0.06)


def points(req):
    t_all = time.time()
    data, img_sha, K, T, support_z = _parse_points_inputs(req)
    cfg = dict(PTS, **{k: float(v) for k, v in (req.get("params") or {}).items() if k in PTS})
    voxel = float(req.get("voxel_m", cfg["voxel"]))
    if not (0.002 <= voxel <= 0.05):
        raise ValueError("voxel_m_out_of_range")
    radius = float(req.get("robot_radius_m", cfg["robot_radius"]))
    segments = [np.asarray(s, float).reshape(2, 3) for s in req.get("robot_segments_m") or []]
    session_id = str(req.get("session_id") or "").strip() or None
    extra_views = [ev for ev in (req.get("extra_views") or []) if isinstance(ev, dict)][:6]
    c_key = image_key_from_sha(img_sha, K, T, support_z)

    if data is None:
        with CLOCK:
            cached_entry = CACHE.get(c_key)
            if cached_entry is not None:
                CACHE.move_to_end(c_key)
        if cached_entry is None:
            return dict(
                status="error",
                reason="image_cache_miss",
                error="image_cache_miss",
                http_status=404,
                caps=list(SERVICE_CAPS),
            )
        if extra_views:
            ev_bytes_list = [base64.b64decode(ev["image_b64"]) for ev in extra_views]
            mv_key = multiview_key(c_key, extra_views, ev_bytes_list=ev_bytes_list)
            with CLOCK:
                cached_mv = MULTIVIEW_CACHE.get(mv_key)
                if cached_mv is not None:
                    MULTIVIEW_CACHE.move_to_end(mv_key)
            if cached_mv is not None:
                rec, extra_preds = cached_mv
                entry = cached_entry
                hit = True
                STATS["cache_hits"] += 1
            else:
                main_img = cached_entry["img"]
                extra_imgs = [
                    np.asarray(Image.open(io.BytesIO(ev_b)).convert("RGB"))
                    for ev_b in ev_bytes_list
                ]
                rec, extra_preds = reconstruct_multiview(
                    main_img,
                    K,
                    T,
                    support_z,
                    extra_imgs,
                    fallback_rec=cached_entry["rec"],
                )
                with CLOCK:
                    MULTIVIEW_CACHE[mv_key] = (rec, extra_preds)
                    while len(MULTIVIEW_CACHE) > args.cache:
                        MULTIVIEW_CACHE.popitem(last=False)
                entry = cached_entry
                hit = False
                STATS["images"] += len(extra_imgs)
                STATS["vggt_s"].append(rec["vggt_s"])
        else:
            entry = cached_entry
            rec = entry["rec"]
            extra_preds = []
            hit = True
            STATS["cache_hits"] += 1
    elif extra_views:
        ev_bytes_list = [base64.b64decode(ev["image_b64"]) for ev in extra_views]
        mv_key = multiview_key(c_key, extra_views, ev_bytes_list=ev_bytes_list)
        with CLOCK:
            cached_entry = CACHE.get(c_key)
            if cached_entry is not None:
                CACHE.move_to_end(c_key)
            cached_mv = MULTIVIEW_CACHE.get(mv_key)
            if cached_mv is not None:
                MULTIVIEW_CACHE.move_to_end(mv_key)
        if cached_mv is not None:
            rec, extra_preds = cached_mv
            if cached_entry is None:
                main_img = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
                with CLOCK:
                    entry = CACHE.get(c_key)
                    if entry is None:
                        entry = dict(rec=rec, img=main_img)
                        CACHE[c_key] = entry
                        while len(CACHE) > args.cache:
                            CACHE.popitem(last=False)
            else:
                entry = cached_entry
            hit = True
            STATS["cache_hits"] += 1
        else:
            main_img = (
                cached_entry["img"]
                if cached_entry is not None
                else np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
            )
            extra_imgs = [
                np.asarray(Image.open(io.BytesIO(ev_b)).convert("RGB"))
                for ev_b in ev_bytes_list
            ]
            rec, extra_preds = reconstruct_multiview(
                main_img,
                K,
                T,
                support_z,
                extra_imgs,
                fallback_rec=cached_entry["rec"] if cached_entry is not None else None,
            )
            with CLOCK:
                MULTIVIEW_CACHE[mv_key] = (rec, extra_preds)
                while len(MULTIVIEW_CACHE) > args.cache:
                    MULTIVIEW_CACHE.popitem(last=False)
                entry = CACHE.get(c_key)
                if entry is None:
                    entry = dict(rec=rec, img=main_img)
                    CACHE[c_key] = entry
                    while len(CACHE) > args.cache:
                        CACHE.popitem(last=False)
            hit = False
            STATS["images"] += 1 + len(extra_imgs)
            STATS["vggt_s"].append(rec["vggt_s"])
    else:
        entry, hit = _entry(data, K, T, support_z)
        rec = entry["rec"]
        extra_preds = []

    H, W = rec["hw"]
    t = time.time()
    with CLOCK:
        sess_masks_map = dict(entry.get("session_instance_masks") or {})
        fallback_masks = entry.get("instance_masks")
    # 显式传入 session_id 时严禁回退使用其他会话写入的 entry["instance_masks"]，仅匿名调用允许读取 fallback_masks
    inst_masks = sess_masks_map.get(session_id) if session_id else fallback_masks
    inst_map = build_instance_id_map(inst_masks, H, W)

    X = rec["X"].reshape(-1, 3).astype(np.float64)
    conf = rec["conf"].reshape(-1)
    n_pix = len(conf)
    rank = compute_conf_rank(conf)
    cam = T[:3, 3]
    keep = (
        np.isfinite(X).all(1)
        & (X[:, 2] >= support_z - cfg["z_below"])
        & (X[:, 2] <= support_z + cfg["z_above"])
        & (np.linalg.norm(X - cam, axis=1) <= cfg["max_range"])
    )
    idx = np.flatnonzero(keep)

    xyz_parts = [X[idx]]
    rank_parts = [rank[idx]]
    uv_parts = [np.stack([idx % W, idx // W], axis=1).astype(np.uint16)]
    obj_parts = [inst_map.reshape(-1)[idx].astype(np.int16)]
    vbit_parts = [np.ones(len(idx), dtype=np.uint8)]

    fused_views = []
    accepted_residuals = []
    rejected_residuals = []
    extra_views_report = []

    for vi_idx, (ev, (ex_depth, ex_conf)) in enumerate(zip(extra_views, extra_preds)):
        cam_id = str(ev.get("camera_id") or f"extra_{vi_idx + 1}")
        ex_K = np.asarray(ev["K"], np.float64).reshape(3, 3)
        ex_T = np.asarray(ev["T_world_camera"], np.float64).reshape(4, 4)
        ex_max_range = float(ev.get("max_range_m", 0.40))
        gate = align_and_gate_extra_view(
            main_X=rec["X"],
            main_conf=rec["conf"],
            main_K=K,
            main_T=T,
            main_inst_map=inst_map,
            extra_depth=ex_depth,
            extra_conf=ex_conf,
            extra_K=ex_K,
            extra_T=ex_T,
            support_z=support_z,
            z_below=cfg["z_below"],
            z_above=cfg["z_above"],
            max_range_m=ex_max_range,
            max_residual_m=0.025,
        )
        extra_views_report.append(
            dict(
                camera_id=cam_id,
                accepted=bool(gate["accepted"]),
                residual_m=gate["residual_m"],
                scale=gate["scale"],
                n_overlap=int(gate["n_overlap"]),
                reason=str(gate["reason"]),
            )
        )
        if gate["accepted"]:
            fused_views.append(cam_id)
            if gate["residual_m"] is not None:
                accepted_residuals.append(float(gate["residual_m"]))
            bit_val = np.uint8(1 << (vi_idx + 1))
            xyz_parts.append(gate["xyz"])
            rank_parts.append(gate["rank"])
            uv_parts.append(gate["uv"])
            obj_parts.append(gate["obj_id"])
            vbit_parts.append(np.full(len(gate["xyz"]), bit_val, dtype=np.uint8))
        elif gate["residual_m"] is not None:
            rejected_residuals.append(float(gate["residual_m"]))

    ds = voxel_downsample_multiview(
        xyz=np.concatenate(xyz_parts, axis=0) if len(xyz_parts) > 1 else xyz_parts[0],
        rank=np.concatenate(rank_parts, axis=0) if len(rank_parts) > 1 else rank_parts[0],
        uv=np.concatenate(uv_parts, axis=0) if len(uv_parts) > 1 else uv_parts[0],
        obj_id=np.concatenate(obj_parts, axis=0) if len(obj_parts) > 1 else obj_parts[0],
        view_bit=np.concatenate(vbit_parts, axis=0) if len(vbit_parts) > 1 else vbit_parts[0],
        voxel_m=voxel,
    )
    P = ds["xyz"].astype(np.float64)
    robot = np.zeros(len(P), bool)
    for a, b in segments:
        robot |= _seg_dist(P, a, b) < radius

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        xyz=ds["xyz"],
        conf=ds["conf"],
        uv=ds["uv"],
        count=ds["count"],
        robot=robot,
        obj_id=ds["obj_id"],
        view_mask=ds["view_mask"],
        support_z=np.float64(support_z),
        voxel_m=np.float64(voxel),
        image_size=np.asarray([W, H], np.int32),
    )
    raw = buf.getvalue()
    above = P[(P[:, 2] > support_z + ABOVE) & ~robot]
    bbox = [P.min(0).round(4).tolist(), P.max(0).round(4).tolist()] if len(P) else None
    bbox_above = [above.min(0).round(4).tolist(), above.max(0).round(4).tolist()] if len(above) else None

    if accepted_residuals:
        multiview_residual_m = round(float(max(accepted_residuals)), 5)
    elif rejected_residuals:
        multiview_residual_m = round(float(min(rejected_residuals)), 5)
    else:
        multiview_residual_m = None

    t_pts = time.time() - t
    total = time.time() - t_all
    STATS["total_s"].append(total)
    return dict(
        status="ok",
        version=PTS_VERSION,
        source="vggt_estimate",
        image_sha256=img_sha,
        caps=list(SERVICE_CAPS),
        cache_hit=hit,
        image_size=[W, H],
        scale=rec["s"],
        support_fit=rec["table"],
        support_z_m=support_z,
        voxel_m=voxel,
        z_range_m=[support_z - cfg["z_below"], support_z + cfg["z_above"]],
        max_range_m=cfg["max_range"],
        n_points=int(len(P)),
        n_pixels_kept=int(len(idx)),
        n_pixels=int(n_pix),
        n_robot=int(robot.sum()),
        n_above_support=int(len(above)),
        bbox_m=bbox,
        bbox_above_support_m=bbox_above,
        fused_views=fused_views,
        multiview_residual_m=multiview_residual_m,
        **({"extra_views_report": extra_views_report} if extra_views_report else {}),
        npz_bytes=len(raw),
        npz_b64=base64.b64encode(raw).decode("ascii"),
        timing=dict(vggt_s=None if hit else rec["vggt_s"], points_s=t_pts, total_s=total),
    )


MAX_SESSION_META = 24


def _evict_local_session(session_id: str) -> bool:
    """释放单个 `session_id` 在本进程图像缓存中的全部掩码与编号元数据。"""
    local_closed = False
    if SESSION_MAX_TRACK_NUM.pop(session_id, None) is not None:
        local_closed = True
    if SESSION_KNOWN_TRACKS.pop(session_id, None) is not None:
        local_closed = True
    keys = SESSION_CACHE_KEYS.pop(session_id, None)
    if keys is not None:
        local_closed = True
        for ck in keys:
            ent = CACHE.get(ck)
            if ent is not None:
                s_map = ent.get("session_instance_masks")
                if isinstance(s_map, dict):
                    s_map.pop(session_id, None)
                if ent.get("last_session_id") == session_id:
                    ent.pop("instance_masks", None)
                    ent.pop("last_session_id", None)
    return local_closed


def _record_session_masks(session_id: str, entry_key: str, entry: dict, instance_masks_list: list, *, persistent_ids: bool):
    if session_id in SESSION_CACHE_KEYS:
        SESSION_CACHE_KEYS.move_to_end(session_id)
    else:
        while len(SESSION_CACHE_KEYS) >= MAX_SESSION_META:
            oldest_sid = next(iter(SESSION_CACHE_KEYS))
            _evict_local_session(oldest_sid)
        SESSION_CACHE_KEYS[session_id] = set()
    SESSION_CACHE_KEYS[session_id].add(entry_key)
    entry.setdefault("session_instance_masks", {})[session_id] = instance_masks_list
    entry["last_session_id"] = session_id
    if persistent_ids:
        known = SESSION_KNOWN_TRACKS.setdefault(session_id, set())
        for t_num, _ in instance_masks_list:
            known.add(int(t_num))


def session_close(req):
    """清理本服务缓存中的会话实例掩码状态并级联转发至 `:8333/session_close`。"""
    session_id = str((req or {}).get("session_id") or "").strip()
    if not session_id:
        raise ValueError("missing_session_id")

    with CLOCK:
        local_closed = _evict_local_session(session_id)
    remote_closed = False
    remote_err = None
    body = json.dumps({"session_id": session_id}).encode()
    rq = urllib.request.Request(
        args.sam31.rstrip("/") + "/session_close",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(rq, timeout=float((req or {}).get("timeout_s", 10.0))) as r:
            ans = json.loads(r.read())
            remote_closed = bool(ans.get("closed", False))
    except urllib.error.HTTPError as err:
        with contextlib.closing(err):
            raw_err = err.read()
        remote_err = f"HTTPError {err.code}: {raw_err[:120]!r}"[:160]
    except Exception as ex:
        remote_err = f"{type(ex).__name__}: {ex}"[:160]

    return dict(
        status="ok",
        closed=bool(local_closed or remote_closed),
        **({"sam31_error": remote_err} if remote_err else {}),
    )


# ------------------------------------------------------------------ cache + worker
CACHE = collections.OrderedDict()
MULTIVIEW_CACHE = collections.OrderedDict()
SESSION_CACHE_KEYS = collections.OrderedDict()
SESSION_MAX_TRACK_NUM = collections.OrderedDict()
SESSION_KNOWN_TRACKS = collections.OrderedDict()
CLOCK = threading.Lock()
STATS = dict(requests=0, images=0, cache_hits=0, errors=0, abandoned=0, busy_s=0.0, vggt_s=[], total_s=[], peak_mb=0.0,
             started=time.strftime("%F %T %Z"))
Q = queue.Queue(maxsize=args.max_queue)
_WORKER_STARTED = False
_WORKER_LOCK = threading.Lock()


def image_key_from_sha(image_sha256: str, K, T, support_z) -> str:
    h = hashlib.sha256(str(image_sha256).lower().encode("ascii"))
    h.update(np.asarray(K, np.float64).tobytes())
    h.update(np.asarray(T, np.float64).tobytes())
    h.update(np.float64(support_z).tobytes())
    return h.hexdigest()


def image_key(data, K, T, support_z):
    return image_key_from_sha(hashlib.sha256(data).hexdigest(), K, T, support_z)


def multiview_key(c_key: str, extra_views: list[dict], *, ev_bytes_list: list[bytes] | None = None) -> tuple:
    """构建主视角键与全部附加视角（图像 SHA-256、内外参字节、距离门限）的不可变多视角缓存键。"""
    decoded = ev_bytes_list if ev_bytes_list is not None else [base64.b64decode(ev["image_b64"]) for ev in extra_views]
    return (
        c_key,
        tuple(
            (
                hashlib.sha256(ev_b).hexdigest(),
                np.asarray(ev["K"], np.float64).reshape(3, 3).tobytes(),
                np.asarray(ev["T_world_camera"], np.float64).reshape(4, 4).tobytes(),
                round(float(ev.get("max_range_m", 0.40)), 6),
            )
            for ev, ev_b in zip(extra_views, decoded)
        ),
    )


def _entry(data, K, T, support_z, allow_no_table=False):
    key = image_key(data, K, T, support_z)
    with CLOCK:
        in_cache = key in CACHE
    if allow_no_table and not in_cache:
        try:
            return _entry(data, K, T, support_z)
        except ValueError:
            key = key + ":no_table"
    with CLOCK:
        entry = CACHE.get(key)
        if entry is not None:
            CACHE.move_to_end(key)
            STATS["cache_hits"] += 1
            return entry, True
    img = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    rec = reconstruct(img, K, T, support_z, allow_no_table=key.endswith(":no_table"))
    with CLOCK:
        existing = CACHE.get(key)
        if existing is not None:
            CACHE.move_to_end(key)
            entry = existing
        else:
            entry = dict(rec=rec, img=img)
            CACHE[key] = entry
            while len(CACHE) > args.cache:
                CACHE.popitem(last=False)
    STATS["images"] += 1
    STATS["vggt_s"].append(rec["vggt_s"])
    return entry, False


def handle(req):
    t_all = time.time()
    data, K, T, support_z = _parse_cam_inputs(req)
    method = req.get("method", "sam_point")
    if method not in ("sam_point", "window"):
        raise ValueError("unsupported_method")
    pixels = np.asarray(req.get("pixels", []), np.float64).reshape(-1, 2)
    _reset_peak_mem()
    entry, hit = _entry(data, K, T, support_z)
    max_h = float(req.get("max_object_height_m", OT["max_object_height"]))
    rec = entry["rec"]
    H, W = rec["hw"]
    t_sam = 0.0
    valid_pts = [(float(u), float(v)) for u, v in pixels if 0 <= u < W and 0 <= v < H]
    pt_masks_list = []
    if method == "sam_point" and valid_pts:
        t = time.time()
        pt_masks_list = sam31_points(req["image_b64"], valid_pts, H, W, float(req.get("timeout_s", 60)))
        t_sam += time.time() - t
    pt_idx = 0
    out = []
    for u, v in pixels:
        if not (0 <= u < W and 0 <= v < H):
            out.append(dict(pixel=[u, v], status="pixel_out_of_bounds"))
            continue
        ui, vi = int(u), int(v)
        surf = rec["X"][vi, ui].astype(float)
        conf_rank = float((rec["conf"] < rec["conf"][vi, ui]).mean())
        est, used, mask_px, mask_iou, mask = None, None, 0, None, None
        if method == "sam_point":
            masks, iou = pt_masks_list[pt_idx]
            pt_idx += 1
            mask, mask_iou = pick_mask(masks, iou, u, v)
            if mask is not None:
                est = estimate(rec, mask, support_z)
                mask_px = int(mask.sum())
                used = "sam_point_mask" if est is not None else None
        if est is None:
            est = window(rec, u, v, support_z)
            used = "window" if est is not None else None
        if est is None:
            out.append(dict(pixel=[u, v], status="no_points_above_support", surface_point_m=surf.tolist(),
                            depth_conf_rank=conf_rank))
            continue
        top = est["top_z_m"]
        flags, check = plausibility(rec, mask if used == "sam_point_mask" else None, support_z, T, K, top, max_h)
        out.append(dict(pixel=[float(u), float(v)], status="ok", method=used, top_z_m=top, flags=flags, **check,
                        center_z_m=(top + support_z) / 2.0, support_z_m=support_z, height_m=top - support_z,
                        surface_point_m=surf.tolist(), n_points=est["n_points"], mask_px=mask_px,
                        mask_iou_pred=mask_iou, depth_conf_rank=conf_rank,
                        **({"xy_extent_m": est["xy_extent_m"]} if "xy_extent_m" in est else {})))
    total = time.time() - t_all
    peak = _peak_mem_mb()
    STATS["peak_mb"] = max(STATS["peak_mb"], peak)
    STATS["total_s"].append(total)
    return dict(status="ok", version=VERSION, source="vggt_estimate", image_sha256=hashlib.sha256(data).hexdigest(),
                cache_hit=hit, image_size=[W, H], vggt_mode=rec["mode"], scale=rec["s"], support_fit=rec["table"],
                pixels=out, timing=dict(vggt_s=None if hit else rec["vggt_s"], sam_s=t_sam, total_s=total),
                gpu_peak_allocated_mb=peak)


def _is_cuda_oom(exc: BaseException) -> bool:
    oom_cls = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None) if torch is not None else None
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    msg = str(exc).lower()
    return isinstance(exc, (RuntimeError, MemoryError)) and ("out of memory" in msg or "cuda error 2" in msg)


def _is_sticky_cuda_error(exc: BaseException) -> bool:
    if _is_cuda_oom(exc):
        return False
    msg = str(exc).lower()
    return isinstance(exc, RuntimeError) and ("cuda error" in msg or "cudnn error" in msg)


def _classify_and_recover_exception(exc: BaseException) -> tuple[int, dict]:
    """区分入参/领域几何错误(422)、上游/CUDA OOM/粘滞错误(503)与未预期内部错误(500)。"""
    global CUDA_FATAL_ERROR
    msg = f"{type(exc).__name__}: {exc}"[:500]
    if _is_cuda_oom(exc):
        with CLOCK:
            CACHE.clear()
            MULTIVIEW_CACHE.clear()
        gc.collect()
        if _cuda_ready():
            torch.cuda.empty_cache()
        return 503, dict(status="error", reason="cuda_out_of_memory", error=msg, http_status=503)
    if _is_sticky_cuda_error(exc):
        CUDA_FATAL_ERROR = msg[:300]
        return 503, dict(status="error", reason="cuda_runtime_error", error=msg, http_status=503)
    if isinstance(exc, (UpstreamServiceError, urllib.error.URLError, TimeoutError)):
        return 503, dict(status="error", reason="upstream_unavailable", error=msg, http_status=503)
    if isinstance(exc, (ValueError, binascii.Error, UnidentifiedImageError)):
        return 422, dict(status="error", reason="invalid_input", error=msg, http_status=422)
    return 500, dict(status="error", reason="internal_error", error=msg, http_status=500)


def worker():
    while True:
        job = Q.get()
        t = time.time()
        cancelled = job.get("cancelled")
        deadline = job.get("deadline")
        if (cancelled is not None and cancelled.is_set()) or (deadline is not None and (deadline - t) <= 0.0):
            STATS["abandoned"] += 1
            job["result"] = dict(status="error", error="abandoned_in_queue", http_status=504, queue_wait_s=t - job["t_enq"])
            job["done"].set()
            continue
        req_run = dict(job["req"], timeout_s=deadline - t) if deadline is not None else job["req"]
        fn = {
            "object_table": object_table,
            "points": points,
            "held": held,
            "session_close": session_close,
        }.get(job.get("kind"), handle)
        try:
            job["result"] = fn(req_run)
        except Exception as e:     # reported to the caller, never crashes the worker
            STATS["errors"] += 1
            _, job["result"] = _classify_and_recover_exception(e)
            log("error", traceback.format_exc()[-800:])
        STATS["busy_s"] += time.time() - t
        STATS["requests"] += 1
        job["result"]["queue_wait_s"] = t - job["t_enq"]
        job["done"].set()


def ensure_worker_started():
    global _WORKER_STARTED
    with _WORKER_LOCK:
        if not _WORKER_STARTED:
            threading.Thread(target=worker, daemon=True).start()
            _WORKER_STARTED = True


def health():
    nv = STATS["vggt_s"][-200:]
    tv = STATS["total_s"][-200:]
    with CLOCK:
        cached_images = len(CACHE)
        multiview_cached = len(MULTIVIEW_CACHE)
        active_sessions = len(SESSION_CACHE_KEYS)
    try:
        with urllib.request.urlopen(args.sam31.rstrip("/") + "/health", timeout=1.5) as r:
            s31 = json.loads(r.read()).get("version")
    except urllib.error.HTTPError as err:
        with contextlib.closing(err):
            err.read()
        s31 = None
    except Exception:
        s31 = None
    out = dict(status="error" if CUDA_FATAL_ERROR else "ok", version=VERSION, object_table=OT_VERSION, points=PTS_VERSION, held="held-v1",
               caps=list(SERVICE_CAPS),
               object_table_backend=OT["seg_backend"], sam31=s31, sam31_endpoint=args.sam31,
               pid=os.getpid(), device=torch.cuda.get_device_name(0) if _cuda_ready() else "cpu",
               cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), queue_depth=Q.qsize(),
               cached_images=cached_images, multiview_cached_images=multiview_cached,
               active_sessions=active_sessions, load_s=LOAD_S,
               allocated_mb=torch.cuda.memory_allocated() / 2**20 if _cuda_ready() else 0.0,
               reserved_mb=torch.cuda.memory_reserved() / 2**20 if _cuda_ready() else 0.0,
               peak_request_mb=STATS["peak_mb"],
               requests=STATS["requests"], images=STATS["images"], cache_hits=STATS["cache_hits"], errors=STATS["errors"],
               abandoned=STATS["abandoned"],
               vggt_s_p50=float(np.median(nv)) if nv else None, total_s_p50=float(np.median(tv)) if tv else None,
               started=STATS["started"])
    if CUDA_FATAL_ERROR:
        out["error"] = CUDA_FATAL_ERROR
    return out


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 60.0

    def log_message(self, fmt, *a):
        pass

    def _send(self, code, obj, *, close=False):
        body = json.dumps(obj).encode()
        if close:
            self.close_connection = True
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if close:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            h = health()
            return self._send(200 if h.get("status") == "ok" else 503, h)
        self._send(404, dict(status="error", error="not_found"))

    def do_POST(self):
        if self.path.startswith("/object_table"):
            kind = "object_table"
        elif self.path.startswith("/points"):
            kind = "points"
        elif self.path.startswith("/held"):
            kind = "held"
        elif self.path.startswith("/session_close"):
            kind = "session_close"
        elif self.path.startswith("/height"):
            kind = "height"
        else:
            return self._send(404, dict(status="error", error="not_found"), close=True)
        try:
            cl = int(self.headers.get("Content-Length", 0))
        except (ValueError, TypeError):
            return self._send(400, dict(status="error", error="invalid_content_length"), close=True)
        if cl < 0:
            return self._send(400, dict(status="error", error="invalid_content_length"), close=True)
        if cl > MAX_REQUEST_BYTES:
            return self._send(413, dict(status="error", error="payload_too_large"), close=True)
        raw = b""
        if cl > 0:
            try:
                raw = self.rfile.read(cl)
            except TimeoutError:
                return self._send(408, dict(status="error", error="request_body_timeout"), close=True)
            except OSError:
                return self._send(400, dict(status="error", error="request_body_read_error"), close=True)
            if len(raw) != cl:
                return self._send(400, dict(status="error", error="incomplete_request_body"), close=True)
        try:
            req = json.loads(raw) if cl > 0 else {}
        except Exception:
            return self._send(400, dict(status="error", error="invalid_json"))
        if not isinstance(req, dict):
            return self._send(400, dict(status="error", error="invalid_json"))
        # /session_close 仅走 CLOCK 轻量字典锁并转发 :8333/session_close，无需排入 GPU 队列 Q
        if kind == "session_close":
            code = 200
            try:
                res = session_close(req)
            except Exception as e:
                STATS["errors"] += 1
                log("error", traceback.format_exc()[-800:])
                code, res = _classify_and_recover_exception(e)
            STATS["requests"] += 1
            res["queue_wait_s"] = 0.0
            return self._send(code, res)
        try:
            raw_timeout = float(req.get("timeout_s", 120))
            if not math.isfinite(raw_timeout) or raw_timeout <= 0:
                raise ValueError("invalid_timeout_s")
            timeout_s = min(max(0.01, raw_timeout), 600.0)
        except (ValueError, TypeError, OverflowError):
            return self._send(422, dict(status="error", reason="invalid_input", error="invalid_timeout_s"))
        ensure_worker_started()
        t_enq = time.time()
        job = dict(
            req=req,
            done=threading.Event(),
            cancelled=threading.Event(),
            t_enq=t_enq,
            deadline=t_enq + timeout_s,
            kind=kind,
        )
        try:
            Q.put_nowait(job)
        except queue.Full:
            return self._send(503, dict(status="error", error="queue_full"))
        if not job["done"].wait(timeout_s):
            job["cancelled"].set()
            return self._send(504, dict(status="error", error="timeout_in_queue"))
        res = job["result"]
        code = 200 if res.get("status") == "ok" else int(res.get("http_status", 422))
        self._send(code, res)


def _probe_bind_port(host: str, port: int) -> None:
    """Fail fast on EADDRINUSE before loading multi-GB GPU weights."""
    if int(port) <= 0:
        return
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, int(port)))


def main(argv=None):
    global args, Q
    args = build_arg_parser().parse_args(argv)
    _probe_bind_port(args.host, args.port)
    Q = queue.Queue(maxsize=args.max_queue)
    load_runtime_model(args)
    ensure_worker_started()
    srv = ThreadingHTTPServer((args.host, args.port), H)
    srv.daemon_threads = True
    if args.status:
        with open(args.status, "w") as fh:
            fh.write(f"UP pid={os.getpid()} port={args.port} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')} "
                     f"since={time.strftime('%F %T %Z')} load_s={LOAD_S:.1f} allocated_mb={MEM_LOADED_MB:.0f}\n")
    log(f"listening on {args.host}:{args.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
