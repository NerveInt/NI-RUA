"""SAM 3.1 text-prompt segmentation service (binds 127.0.0.1 by default; own venv, see services/vggt_height/README.md).

Official facebookresearch/sam3 package (7f473cfd98), image model (detector only, build_sam3_image_model with
enable_inst_interactivity=False) loaded from facebook/sam3.1 sam3.1_multiplex.pt: the "detector.*" keys are the image
grounding model (same key mapping as sam3/model_builder.py _load_checkpoint); tracker keys are not used for single images.
  GET  /health
  POST /segment {image_b64, phrases:[str], threshold?: 0.5, max_per_phrase?: 20}
    -> {status, image_size [W,H], phrases:[{phrase, instances:[{score, box_xyxy, area, mask_b64}]}], timing}
       mask_b64 = base64 of np.packbits(mask HxW bool, row-major); unpack with np.unpackbits(...)[:H*W].reshape(H, W).
  POST /track {session_id, image_b64, phrases?:[str], points?:[[u,v],...], reanchor?:dict, threshold?:0.4, max_per_phrase?:20}
  POST /session_close {session_id: str}
One GPU worker (lock), per-image backbone cache (LRU). Phrases are whatever the caller sends; this service never names objects.
"""
import argparse
import base64
import binascii
import collections
import contextlib
import gc
import hashlib
import io
import json
import os
import socket
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

try:
    import torch
except ImportError:  # 允许在无 PyTorch 的纯 CPU 单测环境中导入与桩测试
    torch = None

try:
    from .video_session_manager import (
        DEFAULT_LOW_RES_LOGITS_HW,
        MAX_GPU_HOT_SESSIONS,
        MAX_KEYFRAMES,
        MAX_SESSIONS,
        SESSION_TTL_S,
        VideoSessionManager,
        mask_iou,
    )
except ImportError:
    from video_session_manager import (  # type: ignore[no-redef]
        DEFAULT_LOW_RES_LOGITS_HW,
        MAX_GPU_HOT_SESSIONS,
        MAX_KEYFRAMES,
        MAX_SESSIONS,
        SESSION_TTL_S,
        VideoSessionManager,
        mask_iou,
    )

VERSION = "sam31-text-v1"
MAX_REQUEST_BYTES = 64 * 1024 * 1024
CUDA_FATAL_ERROR: str | None = None


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("SAM31_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("SAM31_PORT", "8333")))
    ap.add_argument("--ckpt", default=os.environ.get("SAM31_CKPT", "weights/sam3.1/sam3.1_multiplex.pt"))
    ap.add_argument("--cache", type=int, default=16)
    ap.add_argument("--status", default="")
    return ap


args = build_arg_parser().parse_args([])
MODEL = None
PROC = None
LOAD = {}
LOAD_S = 0.0

CACHE = collections.OrderedDict()
SESSION_MANAGER = VideoSessionManager(
    max_sessions=MAX_SESSIONS,
    session_ttl_s=SESSION_TTL_S,
    max_keyframes=MAX_KEYFRAMES,
    max_gpu_hot_sessions=MAX_GPU_HOT_SESSIONS,
)
LOCK = threading.Lock()
STATS = dict(requests=0, images=0, errors=0, total_s=[], peak_mb=0.0, started=time.strftime("%F %T %Z"))


def log(*a):
    print(time.strftime("%F %T"), *a, flush=True)


def _cuda_ready() -> bool:
    return bool(torch is not None and torch.cuda.is_available())


def _inference_ctx():
    return torch.inference_mode() if torch is not None else contextlib.nullcontext()


def _autocast_ctx():
    if _cuda_ready():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _sync_cuda():
    if _cuda_ready():
        torch.cuda.synchronize()


def _reset_peak_mem():
    if _cuda_ready():
        torch.cuda.reset_peak_memory_stats()


def _peak_mem_mb() -> float:
    return float(torch.cuda.max_memory_allocated() / 2**20) if _cuda_ready() else 0.0


def _to_numpy(val, dtype=None):
    if hasattr(val, "detach"):
        val = val.detach()
    if hasattr(val, "float") and (
        dtype in (None, np.float32, np.float64, float) or "bfloat16" in str(getattr(val, "dtype", ""))
    ):
        val = val.float()
    if hasattr(val, "cpu"):
        val = val.cpu()
    if hasattr(val, "numpy"):
        val = val.numpy()
    return np.asarray(val, dtype=dtype) if dtype is not None else np.asarray(val)


def _encode_mask_b64(mask_bool: np.ndarray) -> str:
    return base64.b64encode(np.packbits(np.asarray(mask_bool, dtype=bool).reshape(-1)).tobytes()).decode("ascii")


def _decode_mask_b64(b64_str: str | None, H: int, W: int) -> np.ndarray | None:
    if not b64_str:
        return None
    bits = np.unpackbits(np.frombuffer(base64.b64decode(b64_str), dtype=np.uint8))
    if bits.size < H * W:
        raise ValueError("mask_b64_too_short")
    return bits[: H * W].reshape(H, W).astype(bool)


def load_runtime_model(parsed_args):
    """加载 SAM 3.1 图像接地与交互分割模型权重。"""
    global MODEL, PROC, LOAD, LOAD_S
    import sam3  # noqa: E402
    from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402
    from sam3.model_builder import build_sam3_image_model  # noqa: E402

    t0 = time.time()
    bpe = os.path.join(os.path.dirname(sam3.__file__), "assets", "bpe_simple_vocab_16e6.txt.gz")
    model = build_sam3_image_model(
        bpe_path=bpe,
        device="cuda",
        eval_mode=True,
        checkpoint_path=None,
        load_from_HF=False,
        enable_inst_interactivity=True,
    )
    ck = torch.load(parsed_args.ckpt, map_location="cpu", weights_only=True)
    if "model" in ck and isinstance(ck["model"], dict):
        ck = ck["model"]
    sd = {}
    for k, v in ck.items():
        if k.startswith("detector."):
            k2 = k.replace("detector.", "", 1)
            sd[k2] = v
            if k2.startswith("backbone.vision_backbone.interactive_convs."):
                sd[k2.replace("backbone.vision_backbone.interactive_convs.", "backbone.vision_backbone.sam2_convs.", 1)] = v
        elif k == "tracker.model.interactivity_no_mem_embed":
            sd["inst_interactive_predictor.model.no_mem_embed"] = v
        elif k.startswith("tracker.model.interactive_sam_prompt_encoder."):
            sd[k.replace("tracker.model.interactive_sam_prompt_encoder.", "inst_interactive_predictor.model.sam_prompt_encoder.", 1)] = v
        elif k.startswith("tracker.model.interactive_sam_mask_decoder."):
            sd[k.replace("tracker.model.interactive_sam_mask_decoder.", "inst_interactive_predictor.model.sam_mask_decoder.", 1)] = v
        elif k.startswith("tracker."):
            sd[k.replace("tracker.", "inst_interactive_predictor.model.", 1)] = v
    res = model.load_state_dict(sd, strict=False)
    LOAD = dict(
        ckpt=os.path.basename(parsed_args.ckpt),
        n_ckpt_keys=len(ck),
        n_loaded_keys=len(sd),
        missing=len(res.missing_keys),
        unexpected=len(res.unexpected_keys),
        missing_non_rope=len([k for k in res.missing_keys if "freqs_cis" not in k]),
        missing_sample=[k for k in res.missing_keys if "freqs_cis" not in k][:8],
        unexpected_sample=list(res.unexpected_keys)[:8],
    )
    del ck, sd
    MODEL = model.cuda().eval()
    PROC = Sam3Processor(MODEL, confidence_threshold=0.0)
    torch.cuda.synchronize()
    LOAD_S = time.time() - t0
    log("loaded", json.dumps(LOAD), f"{LOAD_S:.1f}s", f"{torch.cuda.memory_allocated() / 2**20:.0f} MB")


def _clone_bb(bb):
    out = {}
    for k, v in bb.items():
        if isinstance(v, dict):
            out[k] = {sk: (list(sv) if isinstance(sv, list) else sv) for sk, sv in v.items()}
        elif isinstance(v, list):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _get_image_and_backbone(data: bytes):
    """按图像 SHA-256 命中或计算骨干特征（LRU 缓存）。"""
    key = hashlib.sha256(data).hexdigest()
    t = time.time()
    hit = key in CACHE
    if hit:
        CACHE.move_to_end(key)
        img, bb = CACHE[key]
    else:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        st = PROC.set_image(img)
        bb = _clone_bb(st["backbone_out"])
        CACHE[key] = (img, bb)
        while len(CACHE) > args.cache:
            CACHE.popitem(last=False)
        STATS["images"] += 1
        _sync_cuda()
    return key, hit, img, bb, time.time() - t


def _predict_points(points, bb, H: int, W: int):
    """在当前骨干特征上执行点提示交互分割及低分通用词兜底。"""
    pts_out = []
    obj_masks, obj_scores = None, None
    for u, v in points:
        state = dict(original_height=H, original_width=W, backbone_out=_clone_bb(bb))
        pmasks, pious, _ = MODEL.predict_inst(
            state,
            point_coords=np.array([[u, v]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )
        pmasks_bool = np.asarray(pmasks > 0, dtype=bool)
        pious_f = np.asarray(pious, dtype=np.float64).reshape(-1)
        m_list = [
            dict(
                iou=float(pious_f[k]),
                area=int(pmasks_bool[k].sum()),
                mask_b64=_encode_mask_b64(pmasks_bool[k]),
            )
            for k in range(len(pmasks_bool))
        ]
        ui, vi = min(max(int(u), 0), W - 1), min(max(int(v), 0), H - 1)
        if len(pious_f) == 0 or float(pious_f.max()) < 0.70:
            if obj_masks is None:
                ost = dict(original_height=H, original_width=W, backbone_out=_clone_bb(bb))
                ost = PROC.set_text_prompt(prompt="object", state=ost)
                osc = _to_numpy(ost["scores"], np.float32).reshape(-1)
                keep = [int(i) for i in np.argsort(-osc) if osc[i] >= 0.4][:20]
                obj_masks = _to_numpy(ost["masks"][keep, 0], bool) if keep else np.zeros((0, H, W), bool)
                obj_scores = osc[keep] if keep else np.zeros((0,), np.float32)
            for om, osc_val in zip(obj_masks, obj_scores):
                if om[vi, ui]:
                    m_list.append(
                        dict(
                            iou=float(max(osc_val, 0.75)),
                            area=int(om.sum()),
                            mask_b64=_encode_mask_b64(om),
                        )
                    )
        pts_out.append(dict(point=[u, v], masks=m_list))
    return pts_out


def segment(req):
    if not isinstance(req, dict):
        raise ValueError("request_must_be_object")
    if not isinstance(req.get("image_b64"), str) or not req["image_b64"]:
        raise ValueError("missing_image_b64")
    t_all = time.time()
    data = base64.b64decode(req["image_b64"])
    try:
        phrases = [str(p).strip()[:120] for p in req.get("phrases") or [] if str(p).strip()][:32]
        points = [[float(u), float(v)] for u, v in np.asarray(req.get("points") or [], np.float64).reshape(-1, 2)]
        thr = float(req.get("threshold", 0.5))
        cap = int(req.get("max_per_phrase", 20))
    except TypeError as exc:
        raise ValueError("invalid_segment_params") from exc
    _reset_peak_mem()
    with _inference_ctx(), _autocast_ctx():
        key, hit, img, bb, t_img = _get_image_and_backbone(data)
        W, H = img.size
        out = []
        t = time.time()
        for p in phrases:
            state = dict(original_height=H, original_width=W, backbone_out=_clone_bb(bb))
            state = PROC.set_text_prompt(prompt=p, state=state)
            sc = _to_numpy(state["scores"], np.float32).reshape(-1)
            order = [int(i) for i in np.argsort(-sc) if sc[i] >= thr][:cap]
            masks = _to_numpy(state["masks"][:, 0], bool) if len(sc) else np.zeros((0, H, W), bool)
            boxes = _to_numpy(state["boxes"], np.float32)
            inst = []
            for i in order:
                m = masks[i].astype(bool)
                inst.append(
                    dict(
                        score=float(sc[i]),
                        box_xyxy=[float(x) for x in boxes[i]],
                        area=int(m.sum()),
                        mask_b64=_encode_mask_b64(m),
                    )
                )
            out.append(dict(phrase=p, instances=inst, n_scored=int(len(sc)), n_above_0_3=int((sc >= 0.3).sum())))
        pts_out = _predict_points(points, bb, H, W)
        t_txt = time.time() - t
    peak = _peak_mem_mb()
    STATS["peak_mb"] = max(STATS["peak_mb"], peak)
    total = time.time() - t_all
    STATS["total_s"].append(total)
    return dict(
        status="ok",
        version=VERSION,
        image_sha256=key,
        cache_hit=hit,
        image_size=[W, H],
        threshold=thr,
        phrases=out,
        points=pts_out,
        timing=dict(image_s=t_img, text_s=t_txt, total_s=total),
        gpu_peak_allocated_mb=peak,
    )


def _expected_mask_input_hw(model, bb) -> tuple[int, int]:
    """读取交互预测器 Prompt Encoder 的真实低分辨率掩码输入尺寸，缺失时由骨干特征图尺寸×4 推导或回退 `(288, 288)`。"""
    prompt_enc = getattr(
        getattr(getattr(model, "inst_interactive_predictor", None), "model", None),
        "sam_prompt_encoder",
        None,
    )
    mask_size = getattr(prompt_enc, "mask_input_size", None)
    if isinstance(mask_size, (tuple, list)) and len(mask_size) == 2 and int(mask_size[0]) > 0 and int(mask_size[1]) > 0:
        return (int(mask_size[0]), int(mask_size[1]))
    fpn = bb.get("backbone_fpn") if isinstance(bb, dict) else None
    if isinstance(fpn, (list, tuple)) and fpn:
        feat_shape = getattr(fpn[-1], "shape", None)
        if feat_shape is not None and len(feat_shape) >= 4 and int(feat_shape[-2]) >= 16 and int(feat_shape[-1]) >= 16:
            return (int(feat_shape[-2]) * 4, int(feat_shape[-1]) * 4)
    return DEFAULT_LOW_RES_LOGITS_HW


def _align_mask_input(mask_input, expected_hw: tuple[int, int]) -> np.ndarray | None:
    """将传入 `mask_input` 对齐至 `(1, exp_h, exp_w)` float32，杜绝 Prompt Encoder 下采样 4 倍后的特征网格尺寸失配。"""
    if mask_input is None:
        return None
    arr = np.asarray(mask_input, dtype=np.float32)
    if arr.ndim == 2:
        arr2d = arr
    elif arr.ndim == 3 and arr.shape[0] >= 1:
        arr2d = arr[0]
    else:
        return None
    exp_h, exp_w = int(expected_hw[0]), int(expected_hw[1])
    if arr2d.shape != (exp_h, exp_w):
        arr2d = cv2.resize(arr2d, (exp_w, exp_h), interpolation=cv2.INTER_LINEAR)
    return np.asarray(arr2d, dtype=np.float32)[None, :, :]


def track(req):
    """按 `session_id` 执行跨帧持久身份分割跟踪与显式重锚定。"""
    if not isinstance(req, dict):
        raise ValueError("request_must_be_object")
    session_id = str(req.get("session_id") or "").strip()
    if not session_id:
        raise ValueError("missing_session_id")
    if not isinstance(req.get("image_b64"), str) or not req["image_b64"]:
        raise ValueError("missing_image_b64")
    t_all = time.time()
    data = base64.b64decode(req["image_b64"])
    raw_phrases = req.get("phrases") if "phrases" in req else ["object"]
    phrases = [str(p).strip()[:120] for p in (raw_phrases or []) if str(p).strip()][:32]
    generic_set = {str(p).strip() for p in (req.get("generic_phrases") or ["object"]) if str(p).strip()}
    reanchor = req.get("reanchor") if isinstance(req.get("reanchor"), dict) else None
    reanchor_phrases = [str(p).strip()[:120] for p in ((reanchor or {}).get("phrases") or []) if str(p).strip()][:16]
    all_phrases = list(phrases)
    for rp in reanchor_phrases:
        if rp not in all_phrases:
            all_phrases.append(rp)

    try:
        points = [[float(u), float(v)] for u, v in np.asarray(req.get("points") or [], np.float64).reshape(-1, 2)]
        thr = float(req.get("threshold", 0.4))
        cap = int(req.get("max_per_phrase", 20))
        dedup_iou = float(req.get("dedup_iou", 0.5))
        min_area = int(req.get("min_area", 16))
        max_area_frac = float(req.get("max_area_frac", 0.25))
    except TypeError as exc:
        raise ValueError("invalid_track_params") from exc

    _reset_peak_mem()
    with _inference_ctx(), _autocast_ctx():
        key, hit, img, bb, t_img = _get_image_and_backbone(data)
        W, H = img.size
        img_rgb = np.asarray(img, dtype=np.uint8)
        ignore_mask = _decode_mask_b64(req.get("ignore_mask_b64"), H, W)
        ignore_frac = float(req.get("ignore_frac", 0.5))

        t = time.time()
        phrase_summaries = []
        raw_dets = []
        for p in all_phrases:
            state = dict(original_height=H, original_width=W, backbone_out=_clone_bb(bb))
            state = PROC.set_text_prompt(prompt=p, state=state)
            sc = _to_numpy(state["scores"], np.float32).reshape(-1)
            order = [int(i) for i in np.argsort(-sc) if sc[i] >= thr][:cap]
            masks = _to_numpy(state["masks"][:, 0], bool) if len(sc) else np.zeros((0, H, W), bool)
            logits = _to_numpy(state.get("masks_logits"), np.float32) if "masks_logits" in state and len(sc) else None
            if p in phrases:
                phrase_summaries.append(dict(phrase=p, n_instances=len(order), n_scored=int(len(sc))))
            is_generic = (p in generic_set) and (p not in reanchor_phrases)
            for i in order:
                m = masks[i].astype(bool)
                area = int(m.sum())
                if area < min_area or area > int(max_area_frac * H * W):
                    continue
                if ignore_mask is not None and float(ignore_mask[m].mean()) >= ignore_frac:
                    continue
                lgt = logits[i, 0] if (logits is not None and logits.ndim == 4 and i < len(logits)) else None
                raw_dets.append(
                    dict(
                        mask=m,
                        score=float(sc[i]),
                        phrase=p,
                        is_generic=is_generic,
                        is_reanchor=(p in reanchor_phrases),
                        low_res_logits=lgt,
                    )
                )

        # 合并同帧跨短语重复检测（具名短语优先，保留全部匹配标签供上游 label_report 对齐）
        raw_dets.sort(key=lambda d: (d["is_generic"], -d["score"]))
        merged_cands = []
        for d in raw_dets:
            dup_idx, best_j = None, 0.0
            for idx_m, mc in enumerate(merged_cands):
                j = mask_iou(d["mask"], mc["mask"])
                if j >= dedup_iou and j > best_j:
                    dup_idx, best_j = idx_m, j
            if dup_idx is not None:
                if not d["is_generic"]:
                    existing_names = [x[0] for x in merged_cands[dup_idx]["labels"]]
                    if d["phrase"] not in existing_names:
                        merged_cands[dup_idx]["labels"].append([d["phrase"], round(d["score"], 3)])
                if d["is_reanchor"]:
                    merged_cands[dup_idx]["reanchor_match"] = True
                if d["score"] > merged_cands[dup_idx]["score"] and (not d["is_generic"] or not merged_cands[dup_idx]["labels"]):
                    merged_cands[dup_idx]["score"] = d["score"]
                continue
            merged_cands.append(
                dict(
                    mask=d["mask"],
                    score=d["score"],
                    labels=[] if d["is_generic"] else [[d["phrase"], round(d["score"], 3)]],
                    low_res_logits=d["low_res_logits"],
                    reanchor_match=d["is_reanchor"],
                )
            )

        pts_out = _predict_points(points, bb, H, W) if points else []
        t_txt = time.time() - t

        expected_mask_hw = _expected_mask_input_hw(MODEL, bb)

        def _predict_inst_cb(point_coords, point_labels, box=None, mask_input=None):
            st_p = dict(original_height=H, original_width=W, backbone_out=_clone_bb(bb))
            pmasks_lgt, pious_arr, low_res_arr = MODEL.predict_inst(
                st_p,
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                mask_input=_align_mask_input(mask_input, expected_mask_hw),
                multimask_output=True,
                return_logits=True,
            )
            return np.asarray(pmasks_lgt > 0.0, dtype=bool), np.asarray(pious_arr, dtype=np.float64).reshape(-1), low_res_arr

        backbone_fpn_list = bb.get("backbone_fpn") if isinstance(bb, dict) else None
        sem_feat = backbone_fpn_list[-1] if (isinstance(backbone_fpn_list, (list, tuple)) and backbone_fpn_list) else None
        hot_state = {"backbone_sem": sem_feat} if sem_feat is not None else None

        t_t0 = time.time()
        track_res = SESSION_MANAGER.update_tracks(
            session_id=session_id,
            img_rgb=img_rgb,
            image_sha256=key,
            candidates=merged_cands,
            reanchor=reanchor,
            predict_fn=_predict_inst_cb if MODEL is not None else None,
            backbone_feat=sem_feat,
            hot_gpu_state=hot_state,
            min_area=min_area,
            max_area_frac=max_area_frac,
            min_next_track_num=int(req["min_next_track_num"]) if req.get("min_next_track_num") is not None else None,
        )
        t_trk = time.time() - t_t0

    peak = _peak_mem_mb()
    STATS["peak_mb"] = max(STATS["peak_mb"], peak)
    total = time.time() - t_all
    STATS["total_s"].append(total)
    return dict(
        status="ok",
        version=VERSION,
        session_id=session_id,
        image_sha256=key,
        cache_hit=hit,
        image_size=[W, H],
        threshold=thr,
        instances=[
            dict(
                track_id=inst["track_id"],
                track_num=inst["track_num"],
                track_conf=inst["track_conf"],
                track_status=inst["track_status"],
                score=inst["score"],
                box_xyxy=inst["box_xyxy"],
                area=inst["area"],
                labels=inst["labels"],
                mask_b64=_encode_mask_b64(inst["mask"]),
            )
            for inst in track_res["instances"]
        ],
        phrases=phrase_summaries,
        points=pts_out,
        tracking=dict(
            session_id=session_id,
            persistent_ids=True,
            min_track_conf=track_res["min_track_conf"],
            low_conf_ids=track_res["low_conf_ids"],
            active_tracks=track_res["active_tracks"],
            next_track_num=track_res["next_track_num"],
            keyframes_kept=track_res["keyframes_kept"],
        ),
        timing=dict(image_s=t_img, text_s=t_txt, track_s=t_trk, total_s=total),
        gpu_peak_allocated_mb=peak,
    )


def session_close(req):
    """关闭指定 `session_id` 的跨帧跟踪会话并立即释放显存/内存引用。"""
    session_id = str((req or {}).get("session_id") or "").strip()
    if not session_id:
        raise ValueError("missing_session_id")
    closed = SESSION_MANAGER.close_session(session_id)
    st = SESSION_MANAGER.stats()
    return dict(status="ok", session_id=session_id, closed=bool(closed), **st)


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
    """区分客户端入参错误(422)、GPU 显存溢出/粘滞异常(503)与内部服务异常(500)，并在 OOM 时清理缓存恢复。"""
    global CUDA_FATAL_ERROR
    msg = f"{type(exc).__name__}: {exc}"[:500]
    if _is_cuda_oom(exc):
        CACHE.clear()
        SESSION_MANAGER.offload_all_gpu_sessions()
        gc.collect()
        if _cuda_ready():
            torch.cuda.empty_cache()
        return 503, dict(status="error", reason="cuda_out_of_memory", error=msg)
    if _is_sticky_cuda_error(exc):
        CUDA_FATAL_ERROR = msg[:300]
        return 503, dict(status="error", reason="cuda_runtime_error", error=msg)
    if isinstance(exc, TimeoutError):
        return 503, dict(status="error", reason="timeout", error=msg)
    if isinstance(exc, (ValueError, binascii.Error, UnidentifiedImageError)):
        return 422, dict(status="error", reason="invalid_input", error=msg)
    return 500, dict(status="error", reason="internal_error", error=msg)


def health():
    tv = STATS["total_s"][-200:]
    sess_stats = SESSION_MANAGER.stats()
    out = dict(
        status="error" if CUDA_FATAL_ERROR else "ok",
        version=VERSION,
        load=LOAD,
        load_s=LOAD_S,
        pid=os.getpid(),
        device=torch.cuda.get_device_name(0) if _cuda_ready() else "cpu",
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        allocated_mb=torch.cuda.memory_allocated() / 2**20 if _cuda_ready() else 0.0,
        reserved_mb=torch.cuda.memory_reserved() / 2**20 if _cuda_ready() else 0.0,
        peak_request_mb=STATS["peak_mb"],
        requests=STATS["requests"],
        images=STATS["images"],
        errors=STATS["errors"],
        total_s_p50=float(np.median(tv)) if tv else None,
        cached_images=len(CACHE),
        active_sessions=sess_stats["active_sessions"],
        gpu_hot_sessions=sess_stats["gpu_hot_sessions"],
        started=STATS["started"],
    )
    if CUDA_FATAL_ERROR:
        out["error"] = CUDA_FATAL_ERROR
    return out


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            h = health()
            return self._send(200 if h.get("status") == "ok" else 503, h)
        self._send(404, dict(status="error", error="not_found"))

    def do_POST(self):
        if self.path.startswith("/segment"):
            fn = segment
        elif self.path.startswith("/track"):
            fn = track
        elif self.path.startswith("/session_close"):
            fn = session_close
        else:
            return self._send(404, dict(status="error", error="not_found"))
        try:
            cl = int(self.headers.get("Content-Length", 0))
        except (ValueError, TypeError):
            return self._send(400, dict(status="error", error="invalid_content_length"))
        if cl < 0:
            return self._send(400, dict(status="error", error="invalid_content_length"))
        if cl > MAX_REQUEST_BYTES:
            return self._send(413, dict(status="error", error="payload_too_large"))
        try:
            req = json.loads(self.rfile.read(cl)) if cl > 0 else {}
        except Exception:
            return self._send(400, dict(status="error", error="invalid_json"))
        if not isinstance(req, dict):
            return self._send(400, dict(status="error", error="invalid_json"))
        # /session_close 仅走线程安全的 SESSION_MANAGER._lock 字典清理，绕过 GPU 推理互斥锁即时返回
        if fn is session_close:
            code = 200
            try:
                r = session_close(req)
            except Exception as e:
                STATS["errors"] += 1
                log("error", traceback.format_exc()[-800:])
                code, r = _classify_and_recover_exception(e)
            STATS["requests"] += 1
            return self._send(code, r)
        code = 200
        with LOCK:
            try:
                r = fn(req)
            except Exception as e:  # reported to the caller, never crashes the server
                STATS["errors"] += 1
                log("error", traceback.format_exc()[-800:])
                code, r = _classify_and_recover_exception(e)
            STATS["requests"] += 1
        self._send(code, r)


def _probe_bind_port(host: str, port: int) -> None:
    """Fail fast on EADDRINUSE before loading multi-GB GPU weights."""
    if int(port) <= 0:
        return
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, int(port)))


def main(argv=None):
    global args
    args = build_arg_parser().parse_args(argv)
    _probe_bind_port(args.host, args.port)
    load_runtime_model(args)
    srv = ThreadingHTTPServer((args.host, args.port), H)
    srv.daemon_threads = True
    if args.status:
        with open(args.status, "w") as fh:
            fh.write(
                f"UP pid={os.getpid()} port={args.port} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')} "
                f"since={time.strftime('%F %T %Z')} load_s={LOAD_S:.1f} missing_non_rope={LOAD['missing_non_rope']} "
                f"unexpected={LOAD['unexpected']}\n"
            )
    log(f"listening on {args.host}:{args.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
