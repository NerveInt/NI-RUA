"""慢→快双系统语义重锚定分支 `geometry_query(request={"kind": "reanchor_tracks", ...})`。

对接 `:8331/object_table` 流式视频跟踪会话（SAM 3.1 Streaming Video Predictor）的 `session_id` 与 `reanchor` 接口，
支持按持久物体 ID（`target_id` / `target_track_id`）、开放词汇短语（`phrases`）、正负像素提示点（`pos_pixels` / `neg_pixels`，单位 `px`）
或会话级重置（`reset`）修正跨帧身份漂移，并默认以零图像开销（`include_overlay=False`）刷新结构化 `object_table` 与虚拟空间场。
"""
from __future__ import annotations

from . import embodiment_of
from .object_table import (
    DEFAULT_CAMERA,
    MAX_LABELS,
    MAX_PIXELS,
    camera_of,
    enabled as object_table_enabled,
    object_table_query,
)

KIND = "reanchor_tracks"


def _kind_sentence(profile=None) -> str:
    cam = (profile.overview_camera if profile is not None else None) or DEFAULT_CAMERA
    return (
        f"reanchor_tracks (camera_id default {cam}): re-anchor persistent SAM 3.1 tracks on saved observation_id "
        "via phrases, pos_pixels/neg_pixels (px) or reset; returns updated object_table + track_meta (0 images by default)."
    )


KIND_SENTENCE = _kind_sentence()

FIELD_DESCRIPTIONS = {
    "target_id": "Optional persistent track ID (1..32 characters, e.g. 'o3') to re-bind or drop; alias of target_track_id.",
    "target_track_id": "Optional persistent track ID (1..32 characters, e.g. 'o3') to re-bind or drop; alias of target_id.",
    "phrases": "Optional 1..16 open-vocabulary noun phrases (each 1..80 characters) for SAM 3.1 text re-anchoring.",
    "pos_pixels": "Optional 1..16 foreground 2-number [u,v] prompt coordinates in px in camera_id's image.",
    "neg_pixels": "Optional 1..16 background 2-number [u,v] prompt coordinates in px to exclude neighbors.",
    "reset": "Boolean (default false): reset target_id track, or reset episode session when target_id is omitted.",
    "include_overlay": "Boolean (default false): false returns text only (0 images); true adds camera_id ID overlay image.",
}


def enabled(runtime) -> bool:
    """当 `object_table` 与 `persistent_object_tracking` 同时开启时启用 `reanchor_tracks`。"""
    if not object_table_enabled(runtime):
        return False
    cfg = getattr(runtime, "config", runtime)
    features = getattr(cfg, "features", cfg)
    return bool(getattr(features, "persistent_object_tracking", False))


def applies(runtime, request) -> bool:
    """判断 `geometry_query` 请求是否命中 `reanchor_tracks` 分支。"""
    return enabled(runtime) and isinstance(request, dict) and request.get("kind") == KIND


def schema_branch() -> dict:
    """构建 `geometry_query` 中 `kind='reanchor_tracks'` 的输入契约 Schema。"""
    v2 = dict(type="array", items=dict(type="number"), minItems=2, maxItems=2, description="2-number [u,v] pixel in px.")
    return dict(
        type="object",
        properties=dict(
            kind=dict(type="string", const=KIND),
            observation_id=dict(type="string", minLength=1),
            camera_id=dict(type="string", minLength=1),
            target_id=dict(type="string", minLength=1, maxLength=32),
            target_track_id=dict(type="string", minLength=1, maxLength=32),
            phrases=dict(
                type="array",
                items=dict(type="string", minLength=1, maxLength=80, description="Nonempty phrase, 1..80 characters."),
                minItems=1,
                maxItems=MAX_LABELS,
            ),
            pos_pixels=dict(type="array", items=v2, minItems=1, maxItems=MAX_PIXELS),
            neg_pixels=dict(type="array", items=v2, minItems=1, maxItems=MAX_PIXELS),
            reset=dict(type="boolean", default=False),
            include_overlay=dict(type="boolean", default=False),
        ),
        required=["kind", "observation_id"],
        minProperties=3,
        additionalProperties=False,
    )


def extend_definition(geometry_schema: dict, descriptions: dict) -> dict:
    """当开关开启时向 `geometry_query` 注册 `reanchor_tracks` 分支。"""
    geometry_schema["oneOf"].append(schema_branch())
    return descriptions


def describe_branch(geometry_schema: dict, profile=None) -> None:
    """填充 `reanchor_tracks` 分支各参数的物理意义、单位与默认行为说明。"""
    sentence = _kind_sentence(profile)
    for branch in geometry_schema["oneOf"]:
        if branch.get("properties", {}).get("kind", {}).get("const") == KIND:
            branch["properties"]["kind"]["description"] = sentence
            for field, text in FIELD_DESCRIPTIONS.items():
                branch["properties"][field]["description"] = text


def _parse_pixel_pairs(raw_pixels, field_name: str) -> list[list[float]]:
    if raw_pixels is None:
        return []
    if not isinstance(raw_pixels, (list, tuple)):
        raise ValueError(f"invalid_{field_name}:expected_list")
    out: list[list[float]] = []
    for item in raw_pixels:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in item)
        ):
            raise ValueError(f"invalid_{field_name}:expected_2_numbers")
        u, v = float(item[0]), float(item[1])
        if not (u == u and v == v and abs(u) != float("inf") and abs(v) != float("inf")):
            raise ValueError(f"invalid_{field_name}:nonfinite")
        out.append([u, v])
    return out


def build_reanchor_payload(request: dict) -> dict:
    """将客户端请求参数双向兼容归一化为 `:8331/object_table` 所需的 `reanchor` 字典。

    支持 `target_id` 与 `target_track_id` 双向映射为服务端字段 `target_track_id`，
    并规范化 `phrases`、`pos_pixels`（单位 `px`）、`neg_pixels`（单位 `px`）及 `reset`。
    """
    if not isinstance(request, dict):
        raise ValueError("reanchor_request_must_be_object")
    tid_a = request.get("target_id")
    tid_b = request.get("target_track_id")
    if isinstance(tid_a, str) and isinstance(tid_b, str) and tid_a.strip() and tid_b.strip() and tid_a.strip() != tid_b.strip():
        raise ValueError(f"conflicting_target_ids:{tid_a.strip()}_vs_{tid_b.strip()}")
    target_track_id = (tid_b if isinstance(tid_b, str) and tid_b.strip() else tid_a)
    target_track_id = target_track_id.strip() if isinstance(target_track_id, str) and target_track_id.strip() else None

    raw_phrases = request.get("phrases")
    if raw_phrases is not None and not isinstance(raw_phrases, (list, tuple)):
        raise ValueError("invalid_phrases:expected_list")
    phrases = [str(p).strip()[:80] for p in (raw_phrases or []) if str(p).strip()]
    pos_pixels = _parse_pixel_pairs(request.get("pos_pixels"), "pos_pixels")
    neg_pixels = _parse_pixel_pairs(request.get("neg_pixels"), "neg_pixels")
    reset = bool(request.get("reset", False))

    if target_track_id is None and not phrases and not pos_pixels and not neg_pixels and not reset:
        raise ValueError("reanchor_requires_phrases_pixels_target_id_or_reset")

    payload: dict = {"reset": reset}
    if target_track_id is not None:
        payload["target_track_id"] = target_track_id
    if phrases:
        payload["phrases"] = phrases
    if pos_pixels:
        payload["pos_pixels"] = pos_pixels
    if neg_pixels:
        payload["neg_pixels"] = neg_pixels
    return payload


def reanchor_tracks_query(runtime, request: dict, *, timeout_s: float | None = None, slack_s: float = 5.0):
    """执行慢→快语义重锚定请求并返回 `(result, overlays)`。"""
    req = dict(request)
    camera_id = req.get("camera_id") or camera_of(runtime)
    include_overlay = bool(req.get("include_overlay", False))
    try:
        reanchor_payload = build_reanchor_payload(req)
    except ValueError as error:
        result = {
            "schema_version": "0.1",
            **runtime.identity,
            "world_kind": "internal",
            "query_type": KIND,
            "observation_ids": [req.get("observation_id")],
            "status": "invalid_input",
            "reason": str(error)[:200],
            "frame_id": embodiment_of(runtime).frame_id,
            "unit": "m",
            "value": None,
        }
        return result, []

    ot_req = {
        "kind": "object_table",
        "observation_id": req["observation_id"],
        "camera_id": camera_id,
    }
    if reanchor_payload.get("phrases"):
        ot_req["labels"] = list(reanchor_payload["phrases"])

    result, overlays = object_table_query(
        runtime,
        ot_req,
        timeout_s=timeout_s,
        slack_s=slack_s,
        reanchor=reanchor_payload,
        include_overlay=include_overlay,
        query_type=KIND,
    )
    val = result.get("value")
    if isinstance(val, dict):
        diag_objs = (result.get("diagnostics") or {}).get("objects") or []
        val["reanchored_ids"] = [
            str(o["id"]) for o in diag_objs
            if isinstance(o, dict) and o.get("track_status") == "reanchored" and o.get("id")
        ]
        val["reanchor_applied"] = {
            k: v for k, v in reanchor_payload.items()
            if k in ("target_track_id", "phrases", "reset") or (k in ("pos_pixels", "neg_pixels") and v)
        }
    return result, overlays
