"""三维目标/物体球域到多相机像素 ROI 的精确反投影与无损局部视觉投递模块。

本模块支持通过 `env_origin` 世界系三维坐标 `target_m` 或受控观测 `object_table` 中的物体短 ID `object_id`，
利用相机标定内参 `K`（含畸变 `D`）与实测外参 `T_world_camera`（头部固定标定外参与腕部相机实时关节 FK 外参），
精确计算三维感兴趣球域在各相机传感器上的透视投影像素包围盒 `crop_xywh_px = [x, y, w, h]`，
并直接从原始传感器图像进行 1:1 无损像素裁剪（绝不先缩小全图再放大），同时透明返回局部像素与原图 `[u, v]` 的可逆换算关系。
"""
from __future__ import annotations

from copy import deepcopy
import io
import math

import cv2
import numpy as np
from PIL import Image

from ..core.contracts import new_id
from ..core.storage.images import encode_image, image_settings, image_url
from ..robot.model.geometry import GeometryTools, distortion_monotonic_limit

# ------------------------------------------------------------------ 公开契约常量与 JSON Schema
DEFAULT_RADIUS_M: float = 0.08
MIN_RADIUS_M: float = 0.02
MAX_RADIUS_M: float = 0.40

DEFAULT_PATCH_PX: int = 256
MIN_PATCH_PX: int = 128
MAX_PATCH_PX: int = 480

MAX_AUTO_CAMERAS: int = 2
IMAGE_DELIVERY_MODES: tuple[str, ...] = ("none", "focus_roi", "head_only", "all")
FULL_UV_FORMULA: str = "full_uv = [crop_xywh_px[0] + local_u, crop_xywh_px[1] + local_v]"


def _profile_or_default(profile=None):
    if profile is None:
        from ..robot.model.embodiments import ARX_X5_PROFILE
        return ARX_X5_PROFILE
    return profile


def image_delivery_description(profile=None) -> str:
    """Render ``image_delivery`` help for ``profile`` (None = ARX-X5)."""
    p = _profile_or_default(profile)
    overview = p.overview_camera or (p.camera_examples or p.cameras or ("camera_1",))[0]
    has_wrists = bool(p.wrist_cameras)
    skip_note = ", skipping wrist images" if has_wrists else ""
    return (
        "Optional visual delivery policy: 'none' (0 images; fast numeric state/budget refresh without sensor image transport), "
        "'focus_roi' (1:1 native-sensor local crops around focus_3d; requires focus_3d), "
        f"'head_only' (deliver only {overview} RGB image{skip_note}), or "
        "'all' (deliver all calibrated camera RGB views, bypassing adaptive view skipping). "
        "Omission uses focus_roi when focus_3d is supplied, single-view crop when camera_id/roi is supplied, "
        "or the run's standard observation image policy."
    )


def focus_3d_description(profile=None) -> str:
    """Render ``focus_3d`` help for ``profile`` (None = ARX-X5)."""
    p = _profile_or_default(profile)
    frame = p.frame_id
    return (
        f"Optional 3D-to-2D back-projected native-resolution local ROI request in the {frame} frame; "
        "mutually exclusive with manual pixel roi and scale!=1. "
        "Supply target_m (3-element [x,y,z] in m) and/or detected 3D object_id (1..64 characters, such as 'o2'). "
        "Projects the 3D sphere (center target_m, radius radius_m in m, range 0.02..0.40, default 0.08) through calibrated K/D and measured T_world_camera "
        "(including live wrist-camera joint FK) and crops 1:1 native sensor pixels (patch_px 128..480, default 256) without downscale-upscale blur. "
        "Returns focus_3d.views[] with crop_xywh_px=[crop_x,crop_y,w,h], projected_center_uv_px=[u,v], "
        "local_center_uv_px=[local_u,local_v], full_uv_offset_px=[crop_x,crop_y] and pixel_transform; "
        "convert any pixel [local_u,local_v] picked on a returned crop back to original full-image coordinates via "
        "full_uv = [crop_x + local_u, crop_y + local_v] for direct use in geometry_query (triangulate / pixel_plane)."
    )


def focus_3d_schema(profile=None) -> dict:
    """Render ``FOCUS_3D_SCHEMA`` for ``profile`` (None = ARX-X5)."""
    p = _profile_or_default(profile)
    frame = p.frame_id
    overview = p.overview_camera or (p.camera_examples or p.cameras or ("camera_1",))[0]
    cam_examples = list(p.camera_examples or p.cameras)[:2]
    max_cams = max(1, min(3, len(p.cameras))) if p.cameras else 3
    cam_example_str = repr(cam_examples) if cam_examples else "['camera_1']"
    return dict(
        type="object",
        description=focus_3d_description(p),
        additionalProperties=False,
        anyOf=[{"required": ["target_m"]}, {"required": ["object_id"]}],
        properties=dict(
            target_m=dict(
                type="array",
                minItems=3,
                maxItems=3,
                items=dict(type="number"),
                description=(
                    f"Optional 3-element 3D target center [x, y, z] in {frame} metres (m). "
                    "Required when object_id is omitted; when both target_m and object_id are supplied, "
                    "this explicit target_m is used verbatim without background override."
                ),
            ),
            object_id=dict(
                type="string",
                minLength=1,
                maxLength=64,
                description=(
                    f"Optional detected 3D object ID (1..64 characters, such as 'o2') from the current or latest permitted 3D observation in this episode. "
                    f"Resolves the object's 3D centre_m (in {frame} m) when target_m is omitted, and resolves an adaptive "
                    "3D bounding-sphere radius clamped to 0.02..0.40 m when radius_m is omitted."
                ),
            ),
            radius_m=dict(
                type="number",
                minimum=MIN_RADIUS_M,
                maximum=MAX_RADIUS_M,
                default=DEFAULT_RADIUS_M,
                description=(
                    "Optional 3D sphere radius of interest around the target center in metres (m), range 0.02..0.40, "
                    "default 0.08 (or the object's 3D bounding-sphere radius clamped to 0.02..0.40 when object_id is supplied "
                    "and radius_m is omitted). Explicit values are respected verbatim."
                ),
            ),
            cameras=dict(
                type="array",
                minItems=1,
                maxItems=max_cams,
                uniqueItems=True,
                items=dict(
                    type="string",
                    minLength=1,
                    description="Nonempty calibrated camera ID string (at least 1 character).",
                ),
                description=(
                    f"Optional ordered list of 1..{max_cams} unique calibrated camera IDs (each at least 1 character) to crop (e.g. {cam_example_str}). "
                    f"When omitted (and top-level camera_id is omitted), automatically selects at most {min(MAX_AUTO_CAMERAS, max_cams)} frustum-visible cameras "
                    f"closest to the 3D target (preferring the closest visible wrist camera plus {overview} for stereo coverage)."
                ),
            ),
            patch_px=dict(
                type="integer",
                minimum=MIN_PATCH_PX,
                maximum=MAX_PATCH_PX,
                default=DEFAULT_PATCH_PX,
                description=(
                    "Optional minimum reference side length in native sensor pixels (px) for the local crop box, "
                    "range 128..480, default 256. Crops 1:1 from the original full-resolution sensor image (display_scale=1, "
                    "never downscaled then upscaled) and clamps [x, y, width, height] strictly within sensor bounds."
                ),
            ),
        ),
    )


def observation_view_schema_rules(profile=None) -> list[dict]:
    """Render conditional JSON Schema rules for ``robot_observe`` / ``history_view_observation``."""
    p = _profile_or_default(profile)
    overview = p.overview_camera
    head_disallow: list[dict] = [{"required": ["focus_3d"]}]
    if overview:
        head_disallow.append({"properties": {"camera_id": {"not": {"const": overview}}}, "required": ["camera_id"]})
    return [
        {"not": {"required": ["focus_3d", "roi"]}},
        {
            "if": {"properties": {"image_delivery": {"const": "focus_roi"}}, "required": ["image_delivery"]},
            "then": {"required": ["focus_3d"]},
        },
        {
            "not": {
                "properties": {"image_delivery": {"enum": ["none", "head_only", "all"]}},
                "required": ["focus_3d", "image_delivery"],
            }
        },
        {
            "not": {
                "properties": {"scale": {"not": {"const": 1}}},
                "required": ["focus_3d", "scale"],
            }
        },
        {
            "if": {"properties": {"image_delivery": {"enum": ["none", "all"]}}, "required": ["image_delivery"]},
            "then": {
                "not": {
                    "anyOf": [
                        {"required": ["roi"]},
                        {"required": ["focus_3d"]},
                        {"required": ["camera_id"]},
                        {"properties": {"scale": {"not": {"const": 1}}}, "required": ["scale"]},
                    ]
                }
            },
        },
        {
            "if": {"properties": {"image_delivery": {"const": "head_only"}}, "required": ["image_delivery"]},
            "then": {"not": {"anyOf": head_disallow}},
        },
    ]


IMAGE_DELIVERY_DESCRIPTION: str = image_delivery_description()
FOCUS_3D_DESCRIPTION: str = focus_3d_description()
FOCUS_3D_SCHEMA: dict = focus_3d_schema()
OBSERVATION_VIEW_SCHEMA_RULES: list[dict] = observation_view_schema_rules()


# ------------------------------------------------------------------ 原图与局部裁剪坐标可逆换算纯函数
def _is_real_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def local_to_full_uv(crop_xywh_px, local_uv) -> list[float]:
    """将 1:1 局部裁剪图内的像素坐标 `[local_u, local_v]` 换算回原始全图坐标 `[u, v]`（单位：px）。"""
    if not isinstance(crop_xywh_px, (list, tuple)) or len(crop_xywh_px) != 4:
        raise ValueError("invalid_crop_xywh_px")
    if not isinstance(local_uv, (list, tuple)) or len(local_uv) != 2 or not all(_is_real_number(v) for v in local_uv):
        raise ValueError("invalid_local_uv")
    crop_x, crop_y = float(crop_xywh_px[0]), float(crop_xywh_px[1])
    return [crop_x + float(local_uv[0]), crop_y + float(local_uv[1])]


def full_to_local_uv(crop_xywh_px, full_uv) -> list[float]:
    """将原始全图像素坐标 `[u, v]` 换算为 1:1 局部裁剪图内的相对像素坐标 `[local_u, local_v]`（单位：px）。"""
    if not isinstance(crop_xywh_px, (list, tuple)) or len(crop_xywh_px) != 4:
        raise ValueError("invalid_crop_xywh_px")
    if not isinstance(full_uv, (list, tuple)) or len(full_uv) != 2 or not all(_is_real_number(v) for v in full_uv):
        raise ValueError("invalid_full_uv")
    crop_x, crop_y = float(crop_xywh_px[0]), float(crop_xywh_px[1])
    return [float(full_uv[0]) - crop_x, float(full_uv[1]) - crop_y]


# ------------------------------------------------------------------ 参数校验
def validate_focus_3d_spec(focus_spec) -> dict:
    """校验并标准化 `focus_3d` 输入字典，保留调用方显式指定字段标记。"""
    if not isinstance(focus_spec, dict):
        raise ValueError("focus_3d_must_be_object")
    extra = set(focus_spec) - {"target_m", "object_id", "radius_m", "cameras", "patch_px"}
    if extra:
        raise ValueError("unsupported_focus_3d_fields:" + ",".join(sorted(extra)))
    has_target = "target_m" in focus_spec and focus_spec["target_m"] is not None
    has_object = "object_id" in focus_spec and focus_spec["object_id"] is not None
    if not has_target and not has_object:
        raise ValueError("focus_3d_requires_target_m_or_object_id")

    target_m = None
    if has_target:
        raw_t = focus_spec["target_m"]
        if not isinstance(raw_t, (list, tuple)) or len(raw_t) != 3 or not all(_is_real_number(v) for v in raw_t):
            raise ValueError("invalid_focus_3d_target_m")
        target_m = [float(v) for v in raw_t]

    object_id = None
    if has_object:
        raw_oid = focus_spec["object_id"]
        if not isinstance(raw_oid, str) or not raw_oid.strip() or len(raw_oid) > 64:
            raise ValueError("invalid_focus_3d_object_id")
        object_id = raw_oid.strip()

    explicit_radius = "radius_m" in focus_spec and focus_spec["radius_m"] is not None
    radius_m = None
    if explicit_radius:
        raw_r = focus_spec["radius_m"]
        if not _is_real_number(raw_r) or not (MIN_RADIUS_M <= float(raw_r) <= MAX_RADIUS_M):
            raise ValueError("invalid_focus_3d_radius_m")
        radius_m = float(raw_r)

    cameras = None
    if "cameras" in focus_spec and focus_spec["cameras"] is not None:
        raw_cams = focus_spec["cameras"]
        if (
            not isinstance(raw_cams, (list, tuple))
            or not (1 <= len(raw_cams) <= 3)
            or any(not isinstance(c, str) or not c.strip() for c in raw_cams)
            or len(set(raw_cams)) != len(raw_cams)
        ):
            raise ValueError("invalid_focus_3d_cameras")
        cameras = [c.strip() for c in raw_cams]

    patch_px = DEFAULT_PATCH_PX
    if "patch_px" in focus_spec and focus_spec["patch_px"] is not None:
        raw_p = focus_spec["patch_px"]
        if type(raw_p) is not int or not (MIN_PATCH_PX <= raw_p <= MAX_PATCH_PX):
            raise ValueError("invalid_focus_3d_patch_px")
        patch_px = raw_p

    return dict(
        target_m=target_m,
        object_id=object_id,
        radius_m=radius_m,
        explicit_radius=explicit_radius,
        cameras=cameras,
        patch_px=patch_px,
    )


def validate_observation_view_args(args: dict, *, overview_camera: str | None = "cam_head") -> tuple[str | None, dict | None]:
    """统一校验 `robot_observe` 与 `history_view_observation` 的视觉投递与 `focus_3d` 互斥规则。"""
    image_delivery = args.get("image_delivery")
    focus_raw = args.get("focus_3d")
    roi = args.get("roi")
    camera_id = args.get("camera_id")
    scale = args.get("scale", 1)

    if type(scale) is not int or not (1 <= scale <= 4):
        raise ValueError("invalid_image_scale")
    if image_delivery is not None and (not isinstance(image_delivery, str) or image_delivery not in IMAGE_DELIVERY_MODES):
        raise ValueError("invalid_image_delivery_mode")
    if focus_raw is not None and roi is not None:
        raise ValueError("focus_3d_conflicts_with_manual_roi")
    if focus_raw is not None and scale != 1:
        raise ValueError("focus_3d_forbids_display_scale")
    if image_delivery == "focus_roi" and focus_raw is None:
        raise ValueError("focus_roi_requires_focus_3d")
    if focus_raw is not None and image_delivery in ("none", "head_only", "all"):
        raise ValueError("focus_3d_incompatible_with_image_delivery")

    focus_spec = validate_focus_3d_spec(focus_raw) if focus_raw is not None else None
    if focus_spec is not None and camera_id is not None and focus_spec["cameras"] is not None:
        if focus_spec["cameras"] != [camera_id]:
            raise ValueError("focus_3d_cameras_conflict_with_camera_id")

    effective_mode = "focus_roi" if focus_spec is not None else image_delivery
    if effective_mode in ("none", "all"):
        if roi is not None or camera_id is not None or scale != 1:
            raise ValueError(f"image_delivery_{effective_mode}_forbids_single_view_selector")
    elif effective_mode == "head_only":
        if camera_id is not None and overview_camera is not None and camera_id != overview_camera:
            raise ValueError("image_delivery_head_only_requires_overview_camera")
    else:
        if (roi is not None or scale != 1) and camera_id is None and focus_spec is None:
            raise ValueError("image_view_requires_camera_id")

    return effective_mode, focus_spec


# ------------------------------------------------------------------ 受控观测物体表解析
def _envelope_radius_from_dims(length_m, width_m, height_m) -> float:
    """根据物体三维包围盒长宽高计算自适应感兴趣球域半径（1.25 倍外接球半径并钳位至 [0.02, 0.40] m）。"""
    l = max(float(length_m) if _is_real_number(length_m) else 0.04, 0.004)
    w = max(float(width_m) if _is_real_number(width_m) else 0.04, 0.004)
    h = max(float(height_m) if _is_real_number(height_m) else 0.04, 0.004)
    half_diag = 0.5 * math.sqrt(l * l + w * w + h * h)
    return float(np.clip(1.25 * half_diag, MIN_RADIUS_M, MAX_RADIUS_M))


def _lookup_in_table_dict(table: dict, object_id: str) -> dict | None:
    """在单张 `object_table` 字典（支持行列表+列名或行字典格式）中查找指定 `object_id`。"""
    rows = table.get("rows")
    if not isinstance(rows, list):
        return None
    cols = list(
        table.get("columns")
        or ("id", "centre_m", "top_z", "bottom_z", "length", "width", "yaw_deg", "tilt_deg", "support", "quality")
    )
    for row in rows:
        item = dict(zip(cols, row)) if isinstance(row, (list, tuple)) and len(row) >= 2 else (row if isinstance(row, dict) else None)
        if not isinstance(item, dict) or str(item.get("id")) != object_id:
            continue
        centre = item.get("centre_m")
        if not isinstance(centre, (list, tuple)) or len(centre) != 3 or not all(_is_real_number(v) for v in centre):
            raise ValueError(f"focus_3d_object_missing_centre_m:{object_id}")
        top_z, bottom_z = item.get("top_z"), item.get("bottom_z")
        height_m = (float(top_z) - float(bottom_z)) if (_is_real_number(top_z) and _is_real_number(bottom_z)) else 0.04
        return dict(
            object_id=object_id,
            centre_m=[float(v) for v in centre],
            envelope_radius_m=_envelope_radius_from_dims(item.get("length"), item.get("width"), height_m),
        )
    return None


def _extract_cached_tables(runtime, target_obs_id: str | None) -> list[tuple[str | None, dict]]:
    """严格提取与 `target_obs_id` 精确对齐的受控观测 `object_table`（绝不混入其他观测帧缓存）。"""
    if not target_obs_id:
        return []
    candidates: list[tuple[str | None, dict]] = []
    seen_ids: set[str] = set()

    def _push_entry(obs_id: str | None, entry):
        if obs_id != target_obs_id or not isinstance(entry, dict) or "object_table" not in entry:
            return
        payload = entry["object_table"]
        table = payload[0] if isinstance(payload, tuple) and payload else payload
        if isinstance(table, dict) and isinstance(table.get("rows"), list):
            if obs_id not in seen_ids:
                seen_ids.add(obs_id)
                candidates.append((obs_id, table))

    from .external.obs3d import OBS3D_HISTORY_SLOT, OBS3D_MEMO_SLOT
    from ..core.episode_state import slot_peek

    memo = slot_peek(runtime, OBS3D_MEMO_SLOT, None, legacy_attr="_obs3d_memo")
    history = slot_peek(runtime, OBS3D_HISTORY_SLOT, None, legacy_attr="_obs3d_history")

    if isinstance(memo, tuple) and len(memo) == 2 and memo[0] == target_obs_id:
        _push_entry(target_obs_id, memo[1])
    if isinstance(history, dict) and target_obs_id in history:
        _push_entry(target_obs_id, history[target_obs_id])

    return candidates


def resolve_object_target(runtime, bundle: dict, object_id: str) -> dict:
    """严格从与 `bundle['observation_id']` 同帧对齐的 `WorldObservation3D` / `object_table` / 虚拟空间场中解析 `object_id`。"""
    from ..core.episode_state import slot_peek
    from .external.obs3d import VIRTUAL_FIELD_SLOT
    from .world_model import get_world_observation

    obs_id = bundle.get("observation_id") if isinstance(bundle, dict) else None
    if not isinstance(obs_id, str) or not obs_id:
        raise ValueError("focus_3d_object_table_unavailable:missing_observation_id")

    exact_world = get_world_observation(runtime, obs_id)
    if exact_world is not None and exact_world.observation_id != obs_id:
        exact_world = None
    candidates = _extract_cached_tables(runtime, obs_id)

    # 若目标观测尚未缓存且启用了 object_table_observe，则按需计算同帧受控物体表并写入缓存
    has_exact = (exact_world is not None and exact_world.has_table) or bool(candidates)
    if not has_exact:
        from .external import obs3d
        from .external.object_table import camera_of as ot_camera_of
        object_table_on, _ = obs3d.flags(runtime)
        ot_cam_id = ot_camera_of(runtime, bundle)
        if object_table_on and obs3d._head_camera(bundle, ot_cam_id) is not None:
            features = getattr(getattr(runtime, "config", None), "features", None)
            timeout_s = float(getattr(features, "obs3d_timeout_s", 10.0))
            max_objects = int(getattr(features, "obs3d_max_objects", 10))
            table, overlay, result = obs3d._object_table(runtime, obs_id, timeout_s, max_objects, camera_id=ot_cam_id)
            if isinstance(table, dict) and isinstance(table.get("rows"), list):
                obs3d._remember(runtime, obs_id, "object_table", (table, overlay, result), bundle=bundle)
                candidates.insert(0, (obs_id, table))
                exact_world = get_world_observation(runtime, obs_id)

    vf = slot_peek(runtime, VIRTUAL_FIELD_SLOT, None, legacy_attr="_virtual_spatial_field")
    vf_aligned = vf is not None and getattr(vf, "observation_id", None) == obs_id
    vf_objects = getattr(vf, "objects", None) if vf_aligned else None

    has_any_aligned_source = False
    if exact_world is not None and exact_world.has_table:
        has_any_aligned_source = True
        found = exact_world.envelope_target(object_id, min_radius_m=MIN_RADIUS_M, max_radius_m=MAX_RADIUS_M)
        if found is not None:
            return {**found, "source_observation_id": obs_id, "source": "object_table"}

    for cand_obs_id, table in candidates:
        if cand_obs_id != obs_id:
            continue
        has_any_aligned_source = True
        found = _lookup_in_table_dict(table, object_id)
        if found is not None:
            return {**found, "source_observation_id": obs_id, "source": "object_table"}

    if isinstance(vf_objects, dict) and vf_aligned:
        has_any_aligned_source = has_any_aligned_source or bool(vf_objects)
        if object_id in vf_objects:
            obj = vf_objects[object_id]
            return dict(
                object_id=object_id,
                centre_m=[float(v) for v in obj.centre_m],
                envelope_radius_m=_envelope_radius_from_dims(obj.length, obj.width, obj.height_m),
                source_observation_id=obs_id,
                source="virtual_spatial_field",
            )

    if has_any_aligned_source:
        raise ValueError(f"focus_3d_unknown_object_id:{object_id}")

    raise ValueError(f"focus_3d_object_table_unavailable:{obs_id}")


# ------------------------------------------------------------------ 三维球域透视投影与裁剪框精确求解
def _orthonormal_tangent_basis(unit_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """构造与单位视线向量 `unit_vec` 正交的两个标准正交基向量 `(e1, e2)`。"""
    anchor = np.array([1.0, 0.0, 0.0], dtype=np.float64) if abs(float(unit_vec[0])) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e1 = np.cross(unit_vec, anchor)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(unit_vec, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def project_sphere_to_camera(
    target_m,
    radius_m: float,
    K,
    T_world_camera,
    width: int,
    height: int,
    *,
    D=None,
    patch_px: int = DEFAULT_PATCH_PX,
    camera_id: str | None = None,
    pose_source: str | None = None,
    is_wrist: bool = False,
) -> dict:
    """计算世界系三维球域 `(target_m, radius_m)` 在单台标定相机上的精确透视投影与传感器裁剪框。"""
    k_mat = np.asarray(K, dtype=np.float64).reshape(3, 3)
    t_wc = np.asarray(T_world_camera, dtype=np.float64).reshape(4, 4)
    d_coeffs = np.asarray(D if D is not None and len(D) > 0 else [], dtype=np.float64).reshape(-1)
    if d_coeffs.size == 0:
        d_coeffs = np.zeros(5, dtype=np.float64)

    w_px, h_px = int(width), int(height)
    if w_px <= 0 or h_px <= 0:
        raise ValueError("invalid_camera_dimensions")

    r_m = float(radius_m)
    center_w = np.asarray(target_m, dtype=np.float64).reshape(3)
    # 变换到相机光学坐标系 (+X 右, +Y 下, +Z 光轴向前)
    r_wc = t_wc[:3, :3]
    p_cam = r_wc.T @ (center_w - t_wc[:3, 3])
    z_cam = float(p_cam[2])
    dist_m = float(np.linalg.norm(p_cam))

    zero_vec = np.zeros((3, 1), dtype=np.float64)
    if z_cam <= 1e-3:
        return dict(
            camera_id=camera_id,
            in_frustum=False,
            center_in_full_image=False,
            depth_m=round(z_cam, 4),
            camera_distance_m=round(dist_m, 4),
            pose_source=pose_source,
            is_wrist=bool(is_wrist),
        )

    limit = distortion_monotonic_limit(d_coeffs)
    r_center = math.hypot(float(p_cam[0]), float(p_cam[1])) / z_cam
    if limit < math.inf and not (r_center <= limit):
        c_clamp = p_cam.copy()
        c_clamp[:2] *= (limit * (1.0 - 1e-6)) / max(r_center, 1e-12)
        center_px = cv2.projectPoints(c_clamp.reshape(1, 1, 3), zero_vec, zero_vec, k_mat, d_coeffs)[0].reshape(2)
        u_c, v_c = float(center_px[0]), float(center_px[1])
        center_in_image = False
    else:
        center_px = cv2.projectPoints(p_cam.reshape(1, 1, 3), zero_vec, zero_vec, k_mat, d_coeffs)[0].reshape(2)
        u_c, v_c = float(center_px[0]), float(center_px[1])
        center_in_image = bool(0.0 <= u_c < w_px and 0.0 <= v_c < h_px)

    if dist_m <= r_m * 1.001:
        # 相机光心位于球域内部或球面上，可见球面覆盖全幅传感器
        sx0, sy0, sx1, sy1 = 0.0, 0.0, float(w_px), float(h_px)
    else:
        # 求解视线切锥在球面上的精确三维切圆（捕获离轴透视椭圆偏心拉伸与镜头畸变）
        unit_dir = p_cam / dist_m
        ratio_sq = (r_m / dist_m) ** 2
        tangent_center = p_cam * (1.0 - ratio_sq)
        tangent_radius = r_m * math.sqrt(max(0.0, 1.0 - ratio_sq))
        e1, e2 = _orthonormal_tangent_basis(unit_dir)
        angles = np.linspace(0.0, 2.0 * math.pi, 180, endpoint=False, dtype=np.float64)
        circle_pts = (
            tangent_center[None, :]
            + tangent_radius * (np.cos(angles)[:, None] * e1[None, :] + np.sin(angles)[:, None] * e2[None, :])
        )
        # 补充球心及近端/轴向极值点，兼顾大畸变镜头下的内部极值
        cardinal_pts = np.vstack([
            p_cam.reshape(1, 3),
            (p_cam - r_m * unit_dir).reshape(1, 3),
            (p_cam + np.array([r_m, 0.0, 0.0])).reshape(1, 3),
            (p_cam - np.array([r_m, 0.0, 0.0])).reshape(1, 3),
            (p_cam + np.array([0.0, r_m, 0.0])).reshape(1, 3),
            (p_cam - np.array([0.0, r_m, 0.0])).reshape(1, 3),
        ])
        sample_pts = np.vstack([circle_pts, cardinal_pts])
        front_mask = sample_pts[:, 2] >= 1e-3
        if limit < math.inf:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                raw_radii = np.hypot(sample_pts[:, 0], sample_pts[:, 1]) / np.maximum(sample_pts[:, 2], 1e-12)
            valid_cone = front_mask & (raw_radii <= limit)
        else:
            valid_cone = front_mask
        if not np.any(valid_cone):
            return dict(
                camera_id=camera_id,
                in_frustum=False,
                center_in_full_image=False,
                depth_m=round(z_cam, 4),
                camera_distance_m=round(dist_m, 4),
                pose_source=pose_source,
                is_wrist=bool(is_wrist),
            )
        sample_pts = sample_pts[valid_cone]
        proj = cv2.projectPoints(sample_pts.reshape(-1, 1, 3), zero_vec, zero_vec, k_mat, d_coeffs)[0].reshape(-1, 2)
        finite_proj = proj[np.all(np.isfinite(proj), axis=1)]
        if len(finite_proj) == 0:
            return dict(
                camera_id=camera_id,
                in_frustum=False,
                center_in_full_image=False,
                depth_m=round(z_cam, 4),
                camera_distance_m=round(dist_m, 4),
                pose_source=pose_source,
                is_wrist=bool(is_wrist),
            )
        sx0, sy0 = float(np.min(finite_proj[:, 0])), float(np.min(finite_proj[:, 1]))
        sx1, sy1 = float(np.max(finite_proj[:, 0])), float(np.max(finite_proj[:, 1]))

    ix0, iy0 = int(math.floor(sx0)), int(math.floor(sy0))
    ix1, iy1 = max(ix0 + 1, int(math.ceil(sx1))), max(iy0 + 1, int(math.ceil(sy1)))

    # 与图像传感器有效范围 [0, w_px] x [0, h_px] 求交集
    vis_x0, vis_y0 = max(0, ix0), max(0, iy0)
    vis_x1, vis_y1 = min(w_px, ix1), min(h_px, iy1)
    if vis_x1 <= vis_x0 or vis_y1 <= vis_y0:
        return dict(
            camera_id=camera_id,
            in_frustum=False,
            center_in_full_image=False,
            projected_center_uv_px=[round(u_c, 2), round(v_c, 2)],
            depth_m=round(z_cam, 4),
            camera_distance_m=round(dist_m, 4),
            pose_source=pose_source,
            is_wrist=bool(is_wrist),
        )

    vis_w, vis_h = vis_x1 - vis_x0, vis_y1 - vis_y0
    target_w = min(w_px, max(vis_w, int(patch_px)))
    target_h = min(h_px, max(vis_h, int(patch_px)))

    # 在保证 100% 包容可见球域 [vis_x0, vis_x1] 且不越出图像边界的前提下，尽量以投影球心 u_c / v_c 居中
    raw_x0 = int(round(u_c - 0.5 * target_w))
    raw_y0 = int(round(v_c - 0.5 * target_h))
    lo_x, hi_x = max(0, vis_x1 - target_w), min(vis_x0, w_px - target_w)
    lo_y, hi_y = max(0, vis_y1 - target_h), min(vis_y0, h_px - target_h)
    crop_x = int(np.clip(raw_x0, lo_x, hi_x))
    crop_y = int(np.clip(raw_y0, lo_y, hi_y))

    clamped = bool(
        ix0 < 0
        or iy0 < 0
        or ix1 > w_px
        or iy1 > h_px
        or raw_x0 < 0
        or raw_x0 + target_w > w_px
        or raw_y0 < 0
        or raw_y0 + target_h > h_px
    )
    crop_xywh = [crop_x, crop_y, target_w, target_h]
    return dict(
        camera_id=camera_id,
        in_frustum=True,
        center_in_full_image=center_in_image,
        crop_xywh_px=crop_xywh,
        projected_sphere_xywh_px=[vis_x0, vis_y0, vis_w, vis_h],
        full_image_size_px=[w_px, h_px],
        projected_center_uv_px=[round(u_c, 2), round(v_c, 2)],
        local_center_uv_px=[round(u_c - crop_x, 2), round(v_c - crop_y, 2)],
        display_scale=1,
        full_uv_offset_px=[crop_x, crop_y],
        full_uv_formula=FULL_UV_FORMULA,
        pixel_transform=[[1, 0, float(-crop_x)], [0, 1, float(-crop_y)], [0, 0, 1]],
        depth_m=round(z_cam, 4),
        camera_distance_m=round(dist_m, 4),
        pose_source=pose_source,
        clamped_to_bounds=clamped,
        is_wrist=bool(is_wrist),
    )


def _select_auto_views(candidates: list[dict]) -> list[dict]:
    """从未显式指定相机的候选投影中，自动挑选视锥内可见且距离三维目标最近的不超过 2 个相机（兼顾最近腕部与全局头部视角）。"""
    in_frustum = [c for c in candidates if c.get("in_frustum")]
    if not in_frustum:
        raise ValueError("focus_3d_target_not_visible_in_any_camera")
    center_vis = [c for c in in_frustum if c.get("center_in_full_image")]
    pool = center_vis if center_vis else in_frustum

    wrists = sorted([c for c in pool if c.get("is_wrist")], key=lambda c: c["camera_distance_m"])
    fixed = sorted([c for c in pool if not c.get("is_wrist")], key=lambda c: c["camera_distance_m"])

    if wrists and fixed:
        chosen = sorted([wrists[0], fixed[0]], key=lambda c: c["camera_distance_m"])
        return chosen[:MAX_AUTO_CAMERAS]
    ordered = sorted(pool, key=lambda c: c["camera_distance_m"])
    return ordered[:MAX_AUTO_CAMERAS]


def _public_view_record(view: dict) -> dict:
    """生成模型与程序可读的单视角元数据副本（移除内部筛选字段，不含任何哈希摘要）。"""
    keys = (
        "camera_id",
        "crop_xywh_px",
        "projected_sphere_xywh_px",
        "full_image_size_px",
        "projected_center_uv_px",
        "local_center_uv_px",
        "display_scale",
        "full_uv_offset_px",
        "full_uv_formula",
        "pixel_transform",
        "depth_m",
        "camera_distance_m",
        "pose_source",
        "clamped_to_bounds",
    )
    return {k: deepcopy(view[k]) for k in keys if k in view}


def plan_focus_3d_views(runtime, bundle: dict, focus_spec: dict, *, camera_id: str | None = None) -> dict:
    """结合标定内参、实时腕部 FK 外参及 `object_table` 规划 `focus_3d` 多视角裁剪窗口。"""
    spec = validate_focus_3d_spec(focus_spec)
    obs_id = bundle["observation_id"]
    resolved_obj = None
    if spec["object_id"] is not None and (spec["target_m"] is None or not spec["explicit_radius"]):
        resolved_obj = resolve_object_target(runtime, bundle, spec["object_id"])

    if spec["target_m"] is not None:
        target_m = list(spec["target_m"])
        target_source = "explicit_target_m"
    else:
        target_m = list(resolved_obj["centre_m"])
        target_source = "object_table"

    if spec["explicit_radius"]:
        radius_m = float(spec["radius_m"])
        radius_source = "explicit"
    elif resolved_obj is not None:
        radius_m = float(resolved_obj["envelope_radius_m"])
        radius_source = "object_envelope"
    else:
        radius_m = DEFAULT_RADIUS_M
        radius_source = "default"

    bundle_cams = {cam["camera_id"]: cam for cam in bundle.get("cameras", []) if isinstance(cam, dict) and cam.get("camera_id")}
    explicit_cams = spec["cameras"] if spec["cameras"] is not None else ([camera_id] if camera_id is not None else None)
    if explicit_cams is not None:
        for cid in explicit_cams:
            if cid not in bundle_cams:
                raise ValueError("unknown_camera")

    from .external import embodiment_of as run_embodiment
    profile = run_embodiment(runtime)
    geometry = getattr(runtime, "geometry", None) or GeometryTools(runtime.scene, settings=runtime.image_settings, profile=profile)
    identity = {k: bundle[k] for k in ("run_id", "episode_id", "env_instance_id", "env_generation") if k in bundle}
    if len(identity) != 4:
        identity = runtime.identity

    eval_ids = explicit_cams if explicit_cams is not None else list(bundle_cams.keys())
    evaluated: list[dict] = []
    for cid in eval_ids:
        cam_entry = bundle_cams[cid]
        if not cam_entry.get("calibration_ref"):
            if explicit_cams is not None:
                raise ValueError("calibrated_camera_requires_reference")
            continue
        cam_model = geometry.camera(identity, obs_id, cid)
        cal = cam_model.get("calibration") or {}
        is_wrist = bool(cal.get("arm") in profile.arms or cid in profile.wrist_cameras or "wrist" in cid)
        proj = project_sphere_to_camera(
            target_m=target_m,
            radius_m=radius_m,
            K=cam_model["K"],
            T_world_camera=cam_model["T"],
            width=cam_model["width"],
            height=cam_model["height"],
            D=cam_model.get("D"),
            patch_px=spec["patch_px"],
            camera_id=cid,
            pose_source=cam_model.get("pose_source"),
            is_wrist=is_wrist,
        )
        evaluated.append(proj)

    if explicit_cams is not None:
        for proj in evaluated:
            if not proj.get("in_frustum"):
                raise ValueError(f"focus_3d_target_outside_camera_frustum:{proj['camera_id']}")
        selected = evaluated
        camera_selection = "explicit"
    else:
        selected = _select_auto_views(evaluated)
        camera_selection = "auto_closest_frustum"

    result = dict(
        frame_id=profile.frame_id,
        unit="m",
        observation_id=obs_id,
        target_m=[round(float(v), 6) for v in target_m],
        target_source=target_source,
        radius_m=round(float(radius_m), 6),
        radius_source=radius_source,
        patch_px=int(spec["patch_px"]),
        camera_selection=camera_selection,
        selected_camera_ids=[v["camera_id"] for v in selected],
        coordinate_conversion=dict(
            local_to_full_uv=FULL_UV_FORMULA,
            full_to_local_uv="local_uv = [u - crop_xywh_px[0], v - crop_xywh_px[1]]",
            semantics="Crops are 1:1 native sensor pixels (display_scale=1, no resampling blur); add full_uv_offset_px=[crop_x,crop_y] to any crop pixel before passing [u,v] to geometry_query or triangulate_rays.",
        ),
        views=[_public_view_record(v) for v in selected],
    )
    if spec["object_id"] is not None:
        result["object_id"] = spec["object_id"]
        if resolved_obj is not None and resolved_obj.get("source_observation_id"):
            result["object_table_observation_id"] = resolved_obj["source_observation_id"]
    return result


# ------------------------------------------------------------------ 无损裁剪图像生成与审计落库投递
def _deliver_extra_crop(runtime, bundle: dict, cam_entry: dict, crop_xywh: list[int], correlation: dict) -> tuple[dict, dict]:
    """对第 2/3 个选中视角执行与 `Harness.observation_content` 完全同构的 1:1 裁剪、存储落盘与哈希审计登记。"""
    x, y, width, height = (int(v) for v in crop_xywh)
    settings = runtime.image_settings if runtime.image_settings is not None else image_settings(None)
    artifact, raw_bytes = runtime.store.resolve_image(cam_entry["rgb_ref"], allowed_scopes=("actor",))
    with Image.open(io.BytesIO(raw_bytes)) as original:
        cropped_image = original.crop((x, y, x + width, y + height))
        crop_bytes, _ = encode_image(cropped_image, settings, storage=True)
    reference = runtime.store.put_artifact(crop_bytes, "rgb_crop", "image/jpeg")
    cropped_info, _ = runtime.store.artifact(reference)
    url, encoding = image_url(crop_bytes, settings)

    delivery = dict(
        camera_id=cam_entry["camera_id"],
        rgb_ref=reference,
        original_rgb_ref=cam_entry["rgb_ref"],
        preprocessing="crop_resize_lanczos",
        source_roi=[x, y, width, height],
        display_scale=1,
        width=width,
        height=height,
        pixel_transform=[[1, 0, float(-x)], [0, 1, float(-y)], [0, 0, 1]],
        source_sha256=cropped_info["sha256"],
        sha256=encoding["sha256"],
        encoding=encoding,
    )
    if artifact.get("reconstruction"):
        delivery["reconstruction"] = artifact["reconstruction"]

    is_model = not bool(correlation.get("program_id"))
    audience = "actor_program" if correlation.get("program_id") else "astra_model"
    delivery_channel = (
        "program_fd"
        if correlation.get("program_id")
        else "startup_turn"
        if correlation.get("source") == "startup_grounding"
        else getattr(runtime, "delivery_channel", "codex_tool")
    )
    metrics = dict(
        image_count=1,
        transport_bytes=encoding["byte_size"],
        omitted_image_count=0,
        omitted_transport_bytes=0,
    )
    runtime.store.save(
        "delivery",
        new_id("delivery"),
        dict(
            observation_id=bundle["observation_id"],
            correlation=correlation,
            image_selection=None,
            image_metrics=metrics,
            images=[delivery],
            delivered_to_actor=is_model,
            audience=audience,
            delivery_channel=delivery_channel,
        ),
    )
    runtime.store.event(
        "observation.delivered",
        dict(
            images=[delivery],
            image_selection=None,
            image_metrics=metrics,
            audience=audience,
            delivered_to_model=is_model,
        ),
        {**correlation, "observation_id": bundle["observation_id"]},
    )
    return dict(type="inputImage", imageUrl=url), delivery


def deliver_focus_3d_observation(
    runtime,
    bundle: dict,
    focus_spec: dict,
    correlation: dict,
    *,
    camera_id: str | None = None,
    detail: str | None = None,
) -> tuple[list, dict]:
    """执行 `focus_3d` 三维反投影裁剪并完成 `delivery` 哈希审计落盘，返回 `(content, focus_result)`。"""
    focus_result = plan_focus_3d_views(runtime, bundle, focus_spec, camera_id=camera_id)
    views = focus_result["views"]
    first = views[0]

    # 首视角直接复用 Harness.observation_content(scale=1) 完成 1:1 裁剪、元数据精简投影及首帧 delivery 审计落库
    content = runtime.observation_content(
        bundle,
        correlation,
        camera_id=first["camera_id"],
        roi=first["crop_xywh_px"],
        scale=1,
        detail=detail,
        structured=True,
    )

    cam_map = {cam["camera_id"]: cam for cam in bundle.get("cameras", []) if isinstance(cam, dict) and cam.get("camera_id")}
    for extra_view in views[1:]:
        image_part, _ = _deliver_extra_crop(
            runtime,
            bundle,
            cam_map[extra_view["camera_id"]],
            extra_view["crop_xywh_px"],
            correlation,
        )
        content.append(image_part)

    if content and hasattr(content[0], "value") and isinstance(content[0].value, dict):
        meta = content[0].value
        meta["focus_3d"] = deepcopy(focus_result)
        if len(views) > 1:
            # 多视角裁剪时移除单视角 image_view，改由 image_feedback 声明完整相机顺序以对齐 lean_projection 图像标签
            meta.pop("image_view", None)
            all_ids = [cam["camera_id"] for cam in bundle.get("cameras", []) if isinstance(cam, dict) and cam.get("camera_id")]
            included = [v["camera_id"] for v in views]
            meta["image_feedback"] = dict(
                policy="focus_roi",
                observation_id=bundle["observation_id"],
                included_camera_ids=included,
                omitted_camera_ids=[cid for cid in all_ids if cid not in included],
            )
    return content, focus_result
