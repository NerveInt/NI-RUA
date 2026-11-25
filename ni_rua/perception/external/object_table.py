"""features.object_table (on by default with external_perception): geometry_query kind 'object_table'.

With the switch on, geometry_query accepts {kind:'object_table', observation_id, camera_id?, pixels?, labels?,
target_camera_ids?}. labels are noun phrases chosen by the model; they are forwarded verbatim and the service (object
table v5, SAM 3.1 text prompts) segments each phrase; the harness never names objects itself.
The harness sends the saved RGB image of camera_id (default: default_camera, the embodiment's overview camera), its
calibrated K and T_world_camera, the configured support (table) height and the robot link segments of the same
observation (public FK of the measured joint state, plus the profile's gripper mask segment; used only to drop robot
masks) to the local VGGT service (features.vggt_endpoint, POST /object_table).
The service segments the image automatically (SAM 2.1, point-prompt grid restricted to pixels whose VGGT point lies
above the support plane), drops support / background / robot masks and returns, per object mask, VGGT-based geometry in the
observation's reference frame (EmbodimentProfile.frame_id):
centre, top and bottom height, oriented footprint (PCA yaw, length, width), principal-axis tilt, support (table or
the object below) and quality flags. The tool reports this table with ids o1..oN (no names: the model matches ids to
what it sees) and one overlay image per camera: ids at the mask in camera_id, and at the projected centres in each
target_camera_ids view (same ids; a projection, not a detection in that view). Inputs are permitted RGB, calibration
and proprioception only; no simulator truth is read. All values are estimates from monocular depth, not measurements.

Embodiment (design 13 section 5.3): the grasp hints, the default camera, the gripper mask segment, the frame label and
the held-object switch are read from the run's EmbodimentProfile (perception.external.embodiment_of); the hint fields
are documented at grasp_hints_from.

Cross-view check (embodiments whose profile lists ``cross_view`` in ``supports``; SO-101): with target_camera_ids
the result also carries ``value.cross_view[target camera]``. The rows of this table are matched to the object masks
of the target camera's table of the same observation (the table already attached to the observation when it is of
that camera, else one more /object_table call for that camera) by their monocular centres, and each pair is
triangulated from the two mask-centre rays (``cross_view_rows``). Per paired id it reports the target view's own
monocular centre and its offset from the row, the two-view point and its offset, the ray gap and the parallax; ids
without a pair are in ``unmatched`` and ids that were not compared (held rows, rows without a centre or a mask pixel)
in ``skipped`` with the reason. Rows are never replaced or corrected: the monocular row, the other view's monocular
estimate and the two-view point are three estimates with their provenance, and the model decides which to use.

Arm occlusion (profile tag ``robot_occlusion``; SO-101 deployment ``perception.robot_occlusion_report``, default on):
every model-facing table also carries ``value.robot_occlusion`` (``robot_view.occlusion``): the share of the table
within the arm's reach that the arm's FK silhouette covers in this camera, and the rows of the camera's previous table
now under it. Two-view top height (profile tag ``contact_top``; deployment ``perception.contact_top_estimate``, default
on): cross_view rows that stand on the table, or on rows that do, also get ``contact_top_z`` (``contact_rows``,
``contact_height.contact_top``). Both are reports: no row is added, removed or changed.

With the switch off only enabled() is consulted: tool definitions and handler behaviour are unchanged.
"""
from __future__ import annotations

import base64
import contextlib
import http.client
import json
import math
import time
import urllib.error
import urllib.request
from copy import deepcopy

import cv2
import numpy as np

from ...core.config import SUPPORT_Z_FROM_PROFILE
from ...core.pose_math import (
    closure_yaws_deg,
    nearest_yaw_deg,
    posture_transition_angles_deg,
    wrap90,
    wrap180,
)
from ...robot.model.embodiments import X5_MASK_REACH_M, X5_OVERVIEW_CAMERA, EmbodimentProfile
from . import embodiment_of
from ._service_http import (
    close_pool,
    make_image_cache_key,
    note_image_upload_finished,
    note_image_upload_started,
    parse_http_endpoint,
    post_json_keepalive,
)
from .rectify import Rectified, rectified_cached

KIND = "object_table"
SOURCE = "vggt_object_table"
# features.vggt_camera_id default (the ARX-X5 head camera); the X5 head-camera mount box
# (features.object_table_head_exclude_px) applies only to this camera id.
DEFAULT_CAMERA = X5_OVERVIEW_CAMERA
MAX_PIXELS = 16
MAX_LABELS = 16
MAX_TARGETS = 3
# ARX-X5 gripper mask reach (m, link6 origin -> beyond the fingertips); the code reads profile.gripper.mask_segment.
GRIPPER_REACH_M = X5_MASK_REACH_M
LINK_RADIUS_M = 0.06            # robot capsule radius (m) sent to the service around every robot segment
# opening_check: "tight" when the visible width is within this of the gripper's full opening (m).
OPENING_TIGHT_MARGIN_M = 0.010
ELONGATED_RATIO = 1.5
COLUMNS = ["id", "centre_m", "top_z", "bottom_z", "length", "width", "yaw_deg", "tilt_deg", "support", "quality", "label_px"]
# cross-view check (profile support tag CROSS_VIEW_TAG): matching bounds of cross_view_rows, reported with the result
CROSS_VIEW_TAG = "cross_view"
#: Profile tag of value.robot_occlusion (robot_view.TAG; deployment perception.robot_occlusion_report).
ROBOT_OCCLUSION_TAG = "robot_occlusion"
CROSS_VIEW_MAX_CENTRE_M = 0.06        # m: two views' monocular centres of one object lie within this of each other
CROSS_VIEW_MIN_PARALLAX_DEG = 5.0     # deg: smaller angles between the two rays give no usable depth
CROSS_VIEW_FIELD = (" On this robot the result also carries cross_view per target camera: each row paired with the "
                    "same object in that camera's own table (target_view_centre_m, target_view_top_z and the offset "
                    "from the row) and triangulated from the two mask-centre rays (stereo_centre_m, its offset from the "
                    "row, ray_gap_m, parallax_deg); rows are not replaced.")
#: Appended to CROSS_VIEW_FIELD with the profile tag ``contact_top`` (contact_height.contact_top).
CONTACT_TOP_TAG = "contact_top"
CONTACT_TOP_FIELD = (" Paired rows standing on the table (or on rows that do) also get contact_top_z, "
                     "contact_top_z_range and contact_fit_iou: the top height of a prism standing on the table fitted "
                     "to both views' outlines.")
# What exact calibrations leave (ratios to the object's longest side): the two mask-centre rays go through the area
# centroids of two masks that show different faces of the object, so they need not meet. Synthetic check with exact
# calibrations (boxes 30 mm to 120 mm, overview camera plus a wrist-like camera 0.10-0.25 m away at 20-70 deg
# elevation; 895 pairs in tests/test_so101_held_perception.py CrossView.test_exact_calibrations_leave_a_size_
# dependent_ray_gap, 2385 in a larger run; docs/object_table.md#cross-view-check): ray_gap_m up to about 0.25 x
# (p90 0.08 x; a 30 mm cube <= 2 mm, a 120 x 30 mm bar 22-25 mm); stereo_centre_m within 0.16 x of the true centre at
# p90, further at small parallax. Reported facts for the model's judgement; no code decides with them.
CROSS_VIEW_EXACT_GAP_MAX = 0.25
CROSS_VIEW_EXACT_GAP_P90 = 0.08
CROSS_VIEW_EXACT_MIDPOINT_P90 = 0.16
CROSS_VIEW_ASSUMPTION = ("cross_view: a row is paired with the row of the target camera's own table whose monocular "
                         "centre is its mutual nearest within %g m (ambiguous: a second one is that close). "
                         "target_view_centre_m and target_view_top_z are that camera's monocular estimates of the same "
                         "object. stereo_centre_m "
                         "is the midpoint of the closest approach of the two mask-centre rays, independent of "
                         "monocular depth and of the table plane; ray_gap_m is the rays' closest distance. The two "
                         "masks show different faces of the object, so the rays need not meet even with exact "
                         "calibrations: in synthetic exact-calibration checks ray_gap_m was up to about %g x the "
                         "object's longest side (p90 %g x) and the midpoint lay within %g x of the object's centre at "
                         "p90, further at small parallax_deg. A gap beyond what the object's size explains can also "
                         "come from a camera calibration that no longer holds, a mask covering only part of the object "
                         "or more than it, a wrong pairing, or the arm moving between the two images. Large "
                         "target_view_minus_row_m or stereo_minus_row_m means the views disagree about this object. "
                         "No row was changed."
                         % (CROSS_VIEW_MAX_CENTRE_M, CROSS_VIEW_EXACT_GAP_MAX, CROSS_VIEW_EXACT_GAP_P90,
                            CROSS_VIEW_EXACT_MIDPOINT_P90))

_KIND_SENTENCE = ("object_table (this run; {camera}): estimated table of the objects on the "
                  "support with ids o1..oN (centre, top/bottom height, footprint length/width/yaw, axis tilt, support, quality; "
                  "VGGT monocular depth + automatic SAM masks, estimates not measurements) and an id overlay image per camera.")
KIND_SENTENCE = _KIND_SENTENCE.format(camera="camera_id optional, default " + DEFAULT_CAMERA)   # kind_sentence(None)
FIELD_DESCRIPTIONS = {
    "pixels": "Optional for object_table: up to 16 [u,v] pixels on objects you want included even if the automatic "
              "segmentation missed them (one mask per pixel).",
    "target_camera_ids": "Optional for object_table: up to 3 other cameras of the same observation (such as wrist views) "
                         "on which the same ids are drawn at the projected object centres.",
    "labels": "Optional for object_table: noun phrases for the objects you need; each is segmented by text and its rows "
              "get column label.",
}


def kind_sentence(profile=None):
    """Catalogue help of kind object_table for ``profile`` (None = the default ARX-X5 robot): the default camera is
    the profile's ``overview_camera``; an embodiment without one needs camera_id."""
    profile = profile if profile is not None else embodiment_of(None)
    camera = profile.overview_camera
    return _KIND_SENTENCE.format(camera="camera_id optional, default " + camera if camera else "camera_id required")


def _observed_robot_model_id(runtime, observation):
    """robot_model_id of ``observation``, else of the latest observation, else of the capabilities; None."""
    for source in (observation, getattr(runtime, "latest", None), getattr(runtime, "capabilities", None)):
        if isinstance(source, dict) and source.get("robot_model_id"):
            return source.get("robot_model_id")
    return None


def _legacy_robot_model_id(runtime, cfg, observation):
    """The robot id rule of default_camera before EmbodimentProfile (configured id, overridden by the observation, and
    by the latest observation / capabilities when the configured id is the default robot's)."""
    model_id = getattr(getattr(cfg, "backend", None), "robot_model_id", None)
    if isinstance(observation, dict) and observation.get("robot_model_id"):
        return observation.get("robot_model_id")
    if model_id in (None, "arx-x5"):
        return _observed_robot_model_id(runtime, None) or model_id
    return model_id


def default_camera(runtime, observation=None) -> str:
    """features.vggt_camera_id: the camera used when a request omits camera_id and for the obs3d attachments.

    The configured id wins, except the shared default ``DEFAULT_CAMERA`` (the ARX-X5 head camera) on an embodiment
    that has no camera of that id: then the profile's ``overview_camera`` (SO-101 ``fixed``). The profile is
    ``runtime.embodiment`` when it describes the observed robot (same robot_model_id as ``observation`` / the latest
    observation / the capabilities). Offline recomputation of an archived run restores the run's archived profile
    (``apps.evaluation.checkpoint.archived_embodiment``). Two cases keep the rule that predates the profile, the
    SO-101 overview camera ``fixed`` for an ``so101`` observation: a runtime without a profile (an archive written
    before the profile record existed, a bare config), and a profile that is the default robot of a backend without
    an ``embodiment_profile`` hook while another robot is observed (test doubles)."""
    cfg = getattr(runtime, "config", runtime)
    features = getattr(cfg, "features", cfg)
    cam = str(getattr(features, "vggt_camera_id", None) or DEFAULT_CAMERA)
    profile = getattr(runtime, "embodiment", None)
    if isinstance(profile, EmbodimentProfile) and _observed_robot_model_id(runtime, observation) in (
            None, profile.robot_model_id):
        if cam == DEFAULT_CAMERA and cam not in profile.cameras and profile.overview_camera:
            return profile.overview_camera
        return cam
    if cam == DEFAULT_CAMERA and _legacy_robot_model_id(runtime, cfg, observation) == "so101":   # (docstring)
        return "fixed"
    return cam


camera_of = default_camera


def effective_support_z_m(features, camera=None, profile=None) -> float:
    """Resolve support-plane height in m in the frozen observation's reference frame.

    ``features.vggt_support_z_m`` (core.config): ``"profile"`` (the default; also when ``features`` carries no such
    field) = ``profile.support_surface_z_m`` (ARX-X5 0.765 m), or the camera's joint calibration when the embodiment
    has no fixed support height (SO-101: None); an explicit finite number in [-3, 3] m (authoritative for every
    robot); None = the supplied camera's joint calibration.
    ``profile``: the run's EmbodimentProfile (None = the default ARX-X5 robot).
    ``camera``: GeometryTools.camera payload from the same archived observation as RGB/K/T;
    its joint_calibration.table_z_m is used only in the calibration form. No current active-version lookup.
    Returns the effective scalar m sent to the service; raises ValueError if unavailable/invalid.
    Example: effective_support_z_m(Config.from_dict({'features': {'vggt_support_z_m': None}}).features,
                                  {'joint_calibration': {'table_z_m': -0.016}}) == -0.016.
    """
    configured = getattr(features, "vggt_support_z_m", SUPPORT_Z_FROM_PROFILE)
    if configured == SUPPORT_Z_FROM_PROFILE:
        configured = (profile if profile is not None else embodiment_of(None)).support_surface_z_m
    if configured is not None:
        if type(configured) not in (int, float) or not np.isfinite(configured) or not -3 <= configured <= 3:
            raise ValueError("invalid_configured_support_z_m")
        return float(configured)
    jc = camera.get("joint_calibration") if isinstance(camera, dict) else None
    if jc is None and isinstance(camera, dict) and isinstance(camera.get("calibration"), dict):
        jc = camera["calibration"].get("joint_calibration")
    table_z = jc.get("table_z_m") if isinstance(jc, dict) else None
    if type(table_z) not in (int, float) or not np.isfinite(table_z) or not -3 <= table_z <= 3:
        raise ValueError("observation_support_z_unavailable_or_invalid")
    return float(table_z)


def enabled(runtime) -> bool:
    cfg = getattr(runtime, "config", runtime)
    features = getattr(cfg, "features", cfg)
    return bool(getattr(features, "object_table", False))


def persistent_tracking_enabled(runtime) -> bool:
    """判定是否启用跨帧流式持久物体身份跟踪（默认关闭，需显式开启）。"""
    cfg = getattr(runtime, "config", runtime)
    features = getattr(cfg, "features", cfg)
    return bool(getattr(features, "persistent_object_tracking", False))


def held_perception_enabled(runtime) -> bool:
    """features.held_perception on an embodiment that supports it (``"held_perception" in profile.supports``: an FK
    render of the arm meshes, ``held_perception.X5Body`` / ``SO101Body``); off for every other robot whatever the
    switch says."""
    cfg = getattr(runtime, "config", runtime)
    features = getattr(cfg, "features", cfg)
    return bool(getattr(features, "held_perception", False)) and "held_perception" in embodiment_of(runtime).supports


def applies(runtime, request):
    return enabled(runtime) and isinstance(request, dict) and request.get("kind") == KIND


def schema_branch():
    v2 = dict(type="array", items=dict(type="number"), minItems=2, maxItems=2)
    return dict(type="object", properties=dict(
        kind=dict(type="string", const=KIND), observation_id=dict(type="string", minLength=1),
        camera_id=dict(type="string", minLength=1),
        pixels=dict(type="array", items=v2, minItems=1, maxItems=MAX_PIXELS),
        labels=dict(type="array", items=dict(type="string"), maxItems=MAX_LABELS),
        target_camera_ids=dict(type="array", items=dict(type="string", minLength=1), minItems=1, maxItems=MAX_TARGETS,
                               uniqueItems=True)),
        required=["kind", "observation_id"], additionalProperties=False)


def extend_definition(geometry_schema, descriptions):
    """Model-facing schema/description changes, applied only when the switch is on (after other extensions)."""
    geometry_schema["oneOf"].append(schema_branch())
    return descriptions


def describe_branch(geometry_schema, profile=None):
    """Branch-specific field help (called after the shared descriptions were applied); ``profile`` = the run's
    EmbodimentProfile (None = the default ARX-X5 robot)."""
    for branch in geometry_schema["oneOf"]:
        if branch.get("properties", {}).get("kind", {}).get("const") == KIND:
            branch["properties"]["kind"]["description"] = kind_sentence(profile)
            for field, text in FIELD_DESCRIPTIONS.items():
                branch["properties"][field]["description"] = text
            supports = (profile if profile is not None else embodiment_of(None)).supports
            if CROSS_VIEW_TAG in supports:
                branch["properties"]["target_camera_ids"]["description"] += CROSS_VIEW_FIELD
                if CONTACT_TOP_TAG in supports:
                    branch["properties"]["target_camera_ids"]["description"] += CONTACT_TOP_FIELD


class ServiceError(RuntimeError):
    pass


_STDLIB_URLOPEN = urllib.request.urlopen


def _validate_service_object_table_payload(answer: dict) -> None:
    """校验上游 `/object_table` 返回载荷结构与有限数值字段，防止畸形载荷穿透成假 `KeyError`/`TypeError`。"""
    objs = answer.get("objects")
    if objs is not None:
        if not isinstance(objs, list):
            raise ServiceError("malformed_service_payload:objects_must_be_list")
        for idx, item in enumerate(objs):
            if not isinstance(item, dict):
                raise ServiceError(f"malformed_service_payload:object_{idx}_not_dict")
            cm = item.get("centre_m")
            if (
                not isinstance(cm, (list, tuple))
                or len(cm) != 3
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in cm)
            ):
                raise ServiceError(f"malformed_service_payload:object_{idx}_centre_m")
            for num_field in ("top_z_m", "bottom_z_m", "length_m", "width_m", "yaw_deg"):
                val = item.get(num_field)
                if val is not None and (isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val))):
                    raise ServiceError(f"malformed_service_payload:object_{idx}_{num_field}")
            if any(item.get(req_f) is None for req_f in ("top_z_m", "bottom_z_m", "length_m", "width_m", "yaw_deg")):
                raise ServiceError(f"malformed_service_payload:object_{idx}_missing_geometry")
    for list_field in ("held", "requested", "labels"):
        val = answer.get(list_field)
        if val is not None and (not isinstance(val, list) or any(not isinstance(x, dict) for x in val)):
            raise ServiceError(f"malformed_service_payload:{list_field}")


def query_service(endpoint, body, timeout_s, slack_s=5.0, *, on_sent=None):
    payload = json.dumps(body).encode()
    target_info = (
        parse_http_endpoint(endpoint, "/object_table")
        if urllib.request.urlopen is _STDLIB_URLOPEN and isinstance(endpoint, str)
        else None
    )
    if target_info is not None:
        host, port, target = target_info
        _, answer = post_json_keepalive(
            host,
            port,
            target,
            payload,
            timeout_s + slack_s,
            on_sent=on_sent,
            service_error_cls=ServiceError,
        )
    else:
        request = urllib.request.Request(
            endpoint.rstrip("/") + "/object_table",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        sent_fired = False

        def _fire_sent():
            nonlocal sent_fired
            if not sent_fired and callable(on_sent):
                sent_fired = True
                on_sent()

        _fire_sent()
        try:
            with urllib.request.urlopen(request, timeout=timeout_s + slack_s) as response:
                _fire_sent()
                answer = json.loads(response.read())
        except urllib.error.HTTPError as error:
            _fire_sent()
            with contextlib.closing(error):
                raw_err = error.read()
            try:
                answer = json.loads(raw_err)
            except ValueError:
                raise ServiceError("http_" + str(error.code)) from None
        except (urllib.error.URLError, OSError, ValueError) as error:
            _fire_sent()
            raise ServiceError(type(error).__name__ + ":" + str(getattr(error, "reason", error))[:200]) from None
        finally:
            _fire_sent()
    if not isinstance(answer, dict) or answer.get("status") != "ok":
        raise ServiceError(str((answer if isinstance(answer, dict) else {}).get("error", "service_error"))[:200])
    _validate_service_object_table_payload(answer)
    return answer


_DEFAULT_QUERY_SERVICE = query_service


_DEFAULT_ROBOT = object()


class _RobotSegments(list):
    """兼容原生 `list` 的机械臂胶囊线段表，额外记录 FK 求解降级的相机映射 `degraded_cameras`。"""

    def __init__(self, iterable=(), *, degraded_cameras=None):
        super().__init__(iterable)
        self.degraded_cameras = dict(degraded_cameras or {})


def robot_segments(tools, full, observation, mask_segment=_DEFAULT_ROBOT):
    """Line segments of the robot links (public FK of the measured joint state) for dropping robot masks.

    ``mask_segment``: the profile's ``gripper.mask_segment`` (``LinkSegment``: from URDF link ``link`` along
    ``axis_local`` for ``length_m`` m), added so the gripper beyond the last joint is covered; None (SO-101: the
    joint chain ends at the TCP between the fingertips) adds nothing; omitted = the default ARX-X5 robot's
    (link6 -> 0.19 m along link6 +x)."""
    if mask_segment is _DEFAULT_ROBOT:
        mask_segment = embodiment_of(None).gripper.mask_segment
    segments = []
    degraded_cameras = {}
    model = None
    for frame in observation.get("cameras", []):
        cam_id = frame.get("camera_id") if isinstance(frame, dict) else None
        try:
            result = tools.execute({**full, "kind": "fk", "camera_id": cam_id})
        except Exception as exc:
            if cam_id:
                degraded_cameras[str(cam_id)] = f"{type(exc).__name__}:{str(exc)[:120]}"
            continue
        poses = (result.get("value") or {}).get("link_poses") if isinstance(result, dict) else None
        if not poses:
            reason = result.get("reason") if isinstance(result, dict) else None
            if cam_id and reason and reason != "missing_robot_link_calibration":
                degraded_cameras[str(cam_id)] = str(reason)[:120]
            continue
        if model is None:
            model = tools.store.resolve_json(observation["robot_model_ref"])
        poses = {name: np.asarray(item, dtype=np.float64) for name, item in poses.items()}
        for joint in model.get("joints", []):
            parent, child = poses.get(joint.get("parent")), poses.get(joint.get("child"))
            if parent is not None and child is not None and np.linalg.norm(parent[:3, 3] - child[:3, 3]) > 1e-4:
                segments.append([parent[:3, 3].tolist(), child[:3, 3].tolist()])
        if mask_segment is not None and mask_segment.link in poses:
            pose = poses[mask_segment.link]
            reach = mask_segment.length_m * np.asarray(mask_segment.axis_local, dtype=np.float64)
            tip = pose[:3, 3] + pose[:3, :3] @ reach
            segments.append([pose[:3, 3].tolist(), tip.tolist()])
    unique = {json.dumps(np.round(np.asarray(s), 4).tolist()): s for s in segments}
    return _RobotSegments(unique.values(), degraded_cameras=degraded_cameras)


def _r(value, digits=3):
    return None if value is None else round(float(value), digits)


def _object_id(item, index: int) -> str:
    """获取物体的持久 `track_id`（如 `'o3'`）；无持久 ID 时按列表下标回退 `'o1'..'oN'`。"""
    tid = item.get("track_id") if isinstance(item, dict) else None
    if isinstance(tid, str) and tid.strip():
        return tid.strip()
    return "o%d" % (index + 1)


def _normalize_persistent_ids(objects: list[dict], held_meta: dict | None = None) -> list[dict]:
    """在持久跟踪模式下为追加的夹持物行分配不冲突的持久 ID，严禁重编已有 `track_id`。"""
    if not any(isinstance(o, dict) and isinstance(o.get("track_id"), str) and o["track_id"].strip() for o in objects):
        return objects
    used: set[str] = set()
    max_num = 0
    for o in objects:
        if isinstance(o, dict) and isinstance(o.get("track_id"), str) and o["track_id"].strip():
            tid = o["track_id"].strip()
            used.add(tid)
            if len(tid) >= 2 and tid[0] in ("o", "O") and tid[1:].isdigit():
                max_num = max(max_num, int(tid[1:]))
    out = []
    for idx, o in enumerate(objects):
        if not isinstance(o, dict):
            out.append(o)
            continue
        tid = o.get("track_id")
        if isinstance(tid, str) and tid.strip():
            out.append(o)
            continue
        meta = (held_meta or {}).get(idx) if isinstance(held_meta, dict) else None
        pre_id = meta.get("pregrasp_row_id") if isinstance(meta, dict) else None
        if isinstance(pre_id, str) and pre_id.strip() and pre_id.strip() not in used:
            chosen = pre_id.strip()
        else:
            max_num = max(max_num, idx) + 1
            while ("o%d" % max_num) in used:
                max_num += 1
            chosen = "o%d" % max_num
        used.add(chosen)
        copied = dict(o)
        copied["track_id"] = chosen
        out.append(copied)
    return out


def _build_track_meta(answer: dict, objects: list[dict]) -> dict:
    """提取与 4D 动作账本严格对齐的紧凑跨帧跟踪元数据 `track_meta`。"""
    raw = answer.get("tracking") if isinstance(answer.get("tracking"), dict) else {}
    has_track_ids = any(isinstance(o, dict) and isinstance(o.get("track_id"), str) and o["track_id"].strip() for o in objects)
    persistent_ids = bool(raw["persistent_ids"]) if "persistent_ids" in raw else has_track_ids
    min_tc = raw.get("min_track_conf")
    min_track_conf = _r(min_tc, 3) if isinstance(min_tc, (int, float)) and not isinstance(min_tc, bool) else None
    low_conf_ids = [str(x) for x in (raw.get("low_conf_ids") or []) if isinstance(x, str)]
    active_raw = raw.get("active_tracks")
    active_tracks = (
        int(active_raw)
        if isinstance(active_raw, int) and not isinstance(active_raw, bool)
        else (len(objects) if persistent_ids else 0)
    )
    return dict(
        persistent_ids=persistent_ids,
        min_track_conf=min_track_conf,
        low_conf_ids=low_conf_ids,
        active_tracks=active_tracks,
    )


def rows_from(objects, with_label=False):
    rows = []
    for index, item in enumerate(objects):
        support = item.get("support")
        if isinstance(support, int):
            support = _object_id(objects[support], support) if 0 <= support < len(objects) else ("o%d" % (support + 1))
        tilt = _r(item.get("axis_tilt_deg"), 0) if (item.get("elongation") or 0) >= 1.8 else None
        quality = item.get("quality", "ok")
        if item.get("flags"):
            quality = "low:" + ",".join(item["flags"])
        raw_px = item.get("label_px")
        label_px = (
            [int(round(float(v))) for v in raw_px]
            if isinstance(raw_px, (list, tuple)) and len(raw_px) == 2 and all(_finite_num(v) for v in raw_px)
            else None
        )
        row = [_object_id(item, index), [_r(v) for v in item["centre_m"]], _r(item["top_z_m"]), _r(item["bottom_z_m"]),
               _r(item["length_m"]), _r(item["width_m"]), _r(item["yaw_deg"], 0), tilt, support, quality, label_px]
        if with_label:
            row.append(item.get("label"))
        rows.append(row)
    return rows


def is_horizontally_elongated_graspable(length, width, top_z=None, bottom_z=None, tilt_deg=None, *, max_opening_m=None):
    """判定物体是否为适合单爪横跨长轴抓取的水平细长单体（排除超宽容器/连体块、近似等轴物体及竖直高耸物体）。

    ``length`` / ``width`` (m): footprint of the table row; ``top_z`` / ``bottom_z`` (m) and ``tilt_deg`` (deg from
    vertical) optional. ``max_opening_m`` (m): the gripper's full opening (``profile.gripper.full_open_aperture_m``;
    None = the default ARX-X5 robot, 0.088 m): a wider object does not fit across the jaws."""
    if max_opening_m is None:
        max_opening_m = embodiment_of(None).gripper.full_open_aperture_m
    if width <= 0 or width > max_opening_m or (length / width) < ELONGATED_RATIO:
        return False
    if isinstance(top_z, (int, float)) and isinstance(bottom_z, (int, float)):
        if (float(top_z) - float(bottom_z)) >= 0.85 * float(length):
            return False
    if isinstance(tilt_deg, (int, float)) and float(tilt_deg) < 35.0:
        return False
    return True


MIN_CAVITY_DEPTH_M: float = 0.004


def _finite_num(val) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))


def height_hint_field(profile):
    """Name of the grasp-height hint for ``profile``: ``pad_z_m`` when the embodiment accepts grasp-point poses
    (``grasp_angles``: the commanded ``grasp_position_m`` is the ARX-X5 pad centre), ``grasp_centre_z_m`` when it
    accepts the grasp centre between a fixed and a moving jaw (``jaw_grasp_centre``: the commanded ``grasp_centre_m``),
    else ``tcp_z_m`` (the commanded ``position_m`` is the TCP link, ``profile.tcp_label``)."""
    if "grasp_angles" in profile.pose_inputs:
        return "pad_z_m"
    return "grasp_centre_z_m" if "jaw_grasp_centre" in profile.pose_inputs else "tcp_z_m"


def jaw_opening_check(width_m, jaw_grasp):
    """``opening_check`` of an object ``width_m`` (m) wide against a fixed-jaw / moving-jaw gripper
    (``gripper.jaw_grasp``): ``too_wide`` above ``max_gap_m`` (the widest gap between the fingertips along the jaw
    line); ``beyond_parallel_range`` above ``tcp_plane_gap_m`` (the moving fingertip has swung behind the TCP plane
    there, so it meets the object behind a centre commanded at grasp_depth_m 0); ``tight`` within
    ``OPENING_TIGHT_MARGIN_M`` (0.010 m) below ``tcp_plane_gap_m``; else ``ok``."""
    if width_m > jaw_grasp.max_gap_m:
        return "too_wide"
    if width_m > jaw_grasp.tcp_plane_gap_m:
        return "beyond_parallel_range"
    return "tight" if width_m > jaw_grasp.tcp_plane_gap_m - OPENING_TIGHT_MARGIN_M else "ok"


def jaw_line_hints(yaw_deg, length_m, width_m, jaw_grasp):
    """The two horizontal jaw lines along an object's face axes, as inputs of a straight-down ``grasp_centre_m``
    move: ``[{"fixed_jaw_yaw_deg": [a, a +/- 180], "object_width_m": w}, ...]``, first the line across the long
    axis (``yaw_deg + 90``; the object spans its ``width_m`` there), then the line along it (``yaw_deg``;
    ``length_m``). The two yaws of a line put the fixed jaw on opposite sides (degrees about frame +z from +x, in
    [-180, 180)); a line the widest jaw gap (``jaw_grasp.max_gap_m``) cannot span is left out. Facts about the
    object's box only: whether the arm reaches a pose is not implied."""
    lines = []
    for line_yaw, span in ((float(yaw_deg) + 90.0, float(width_m)), (float(yaw_deg), float(length_m))):
        if span <= jaw_grasp.max_gap_m:
            first = round(wrap180(line_yaw), 1)
            lines.append({"fixed_jaw_yaw_deg": [first, round(wrap180(first + 180.0), 1)], "object_width_m": _r(span, 3)})
    return lines


def grasp_hints_from(objects, support_z_m=None, profile=None):
    """为每个识别到的物体预计算零回合税抓取几何提示（仅对水平细长可抓单体输出 across_yaw_deg 及平躺立起 3D 姿态）。

    Per object id (rows of held objects and rows without numeric yaw / width / length get none), computed from the
    table row and ``profile`` (the run's EmbodimentProfile; None = the default ARX-X5 robot):

    * ``opening_check`` (``ok`` | ``tight`` | ``too_wide``): visible width against ``gripper.full_open_aperture_m``
      (m); ``tight`` within ``OPENING_TIGHT_MARGIN_M`` (0.010 m) of it. A ``too_wide`` row gets ``elongated: false``
      (plus ``cavity_depth_m`` when >= 0.004 m).
    * height (``height_hint_field``: ``pad_z_m`` with ``grasp_angles``, else ``tcp_z_m``; m in frame_id): the object's
      centre z, the height of the commanded grasp point (X5 pad centre / TCP) level with the object's centre.
    * ``elongated`` (bool): horizontally elongated and narrower than the opening (``is_horizontally_elongated_graspable``).
    * ``fingertip_support_clearance_m`` (m, when a support height is known): height - ``gripper.
      fingertip_beyond_reference_m`` - support, i.e. how far the fingertips stay above the support with the approach
      axis pointing down; negative = they reach the support first.
    * elongated rows, with ``grasp_angles`` (the RoboProbe ``angles_deg`` convention, ARX-X5): ``across_yaw_deg`` (the
      two angles_deg yaws closing across the long axis) and, for lying objects, ``stand_upright_angles_deg``.
    * elongated rows, without ``grasp_angles`` (decision Q5: convention-free facts): ``jaw_line_yaw_deg`` (deg in
      [-90, 90), about frame +z from frame +x): the horizontal line across the long axis (row ``yaw_deg`` + 90,
      undirected), along which the fingertips close to span the narrow side. No tool angle convention is implied.
    * with ``jaw_grasp_centre`` (a fixed and a moving jaw, SO-101; replaces the two items above): the height field is
      ``grasp_centre_z_m``; ``opening_check`` is ``jaw_opening_check`` of the row width (``ok`` | ``tight`` |
      ``beyond_parallel_range`` | ``too_wide``); every row that is not ``too_wide`` gets ``jaw_lines``
      (``jaw_line_hints``: per face axis the two ``fixed_jaw_yaw_deg`` and the ``object_width_m`` along it), which
      together with the row's ``centre_m`` are the inputs of a straight-down ``grasp_centre_m`` move."""
    profile = profile if profile is not None else embodiment_of(None)
    max_opening = profile.gripper.full_open_aperture_m
    fingertip_m = profile.gripper.fingertip_beyond_reference_m
    height_field = height_hint_field(profile)
    angles = "grasp_angles" in profile.pose_inputs
    jaw_grasp = profile.gripper.jaw_grasp if "jaw_grasp_centre" in profile.pose_inputs else None
    hints = {}
    for index, item in enumerate(objects):
        oid = _object_id(item, index)
        if str(item.get("support") or "").startswith("held_") or item.get("quality") == "held_fk_depth":
            continue
        if (
            not isinstance(item.get("yaw_deg"), (int, float))
            or not isinstance(item.get("width_m"), (int, float))
            or not isinstance(item.get("length_m"), (int, float))
        ):
            continue
        yaw = float(item["yaw_deg"])
        width = float(item["width_m"])
        length = float(item["length_m"])
        centre_z = float(item["centre_m"][2])
        top = item.get("top_z_m")
        bottom = item.get("bottom_z_m")
        tilt = _r(item.get("axis_tilt_deg"), 0) if (item.get("elongation") or 0) >= 1.8 else item.get("tilt_deg")
        support = bottom if isinstance(bottom, (int, float)) else support_z_m
        cav_raw = item.get("cavity_depth_m")
        has_cavity = (
            isinstance(cav_raw, (int, float))
            and not isinstance(cav_raw, bool)
            and float(cav_raw) >= MIN_CAVITY_DEPTH_M
        )
        # The height hint is the object's centre height, the grasp-point height that engages the whole gripping face. It
        # is not raised to keep the fingertips clear of the support: for a thin object (such as a 15 mm block, centre
        # 7.5 mm above the table) that would lift the pads above the object's half height and leave less of it under
        # the face, which slips. The two facts the model needs to decide are given beside it instead.
        pad_z = centre_z
        fit = "too_wide" if width > max_opening else ("tight" if width > max_opening - OPENING_TIGHT_MARGIN_M else "ok")
        if jaw_grasp is not None:
            max_opening, fit = jaw_grasp.max_gap_m, jaw_opening_check(width, jaw_grasp)
        if fit == "too_wide":
            too_wide_hint = dict(opening_check="too_wide", elongated=False)
            if has_cavity:
                too_wide_hint["cavity_depth_m"] = _r(cav_raw, 3)
            hints[oid] = too_wide_hint
            continue
        elongated = is_horizontally_elongated_graspable(length, width, top, bottom, tilt, max_opening_m=max_opening)
        hint = {height_field: _r(pad_z, 3), "opening_check": fit, "elongated": bool(elongated)}
        if has_cavity:
            hint["cavity_depth_m"] = _r(cav_raw, 3)
        if isinstance(support, (int, float)):
            # fingertips reach fingertip_beyond_reference_m beyond the grasp point; negative clearance means they reach
            # the support before the grasp point reaches the centre, which is normal for thin objects. No "safe floor"
            # is offered: a floor value is read as the height to use and lifts the pads above a thin object's half
            # height.
            hint["fingertip_support_clearance_m"] = _r(pad_z - fingertip_m - float(support), 3)
        if jaw_grasp is not None:
            hint["jaw_lines"] = jaw_line_hints(yaw, length, width, jaw_grasp)
        elif elongated and not angles:
            hint["jaw_line_yaw_deg"] = round(wrap90(yaw + 90.0), 1)
        if elongated and angles:
            yaws = sorted(closure_yaws_deg(yaw), key=lambda value: (abs(value), value))
            hint["across_yaw_deg"] = yaws
            if isinstance(tilt, (int, float)) and float(tilt) >= 55.0:
                up = posture_transition_angles_deg(nearest_yaw_deg(yaws, 0.0), "lying", "upright")
                hint["stand_upright_angles_deg"] = up["release_angles_deg"]
        hints[oid] = hint
    return hints


#: Overlay drawing size: label font scale 0.45 and 1 px lines on images up to OVERLAY_REFERENCE_WIDTH_PX wide; a wider
#: image (a 1280 or 1920 px camera mode) scales both with width / 640, so the ids keep their share of the image when a
#: model or viewer sees the overlay downscaled. Images up to 640 px wide are drawn exactly as before.
OVERLAY_REFERENCE_WIDTH_PX = 640
LABEL_SCALE = 0.45


def overlay_scale(image):
    """Drawing scale of an overlay image (H, W[, 3]): max(1, W / OVERLAY_REFERENCE_WIDTH_PX)."""
    return max(1.0, float(image.shape[1]) / OVERLAY_REFERENCE_WIDTH_PX)


def _draw_label(image, text, u, v, scale=LABEL_SCALE):
    grow = overlay_scale(image)
    scale, thickness, pad = scale * grow, max(1, int(round(grow))), int(round(2 * grow))
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = int(round(u - w / 2)), int(round(v + h / 2))
    cv2.rectangle(image, (x - pad, y - h - pad), (x + w + pad - 1, y + pad + 1), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def render_source(image, objects):
    line = max(1, int(round(overlay_scale(image))))
    for index, item in enumerate(objects):
        contour = np.asarray(item.get("contour_px") or [], dtype=np.int32).reshape(-1, 1, 2)
        if len(contour) >= 3:
            cv2.polylines(image, [contour], True, (0, 255, 255) if item.get("flags") else (0, 255, 0), line, cv2.LINE_AA)
    for index, item in enumerate(objects):
        u, v = item["label_px"]
        _draw_label(image, _object_id(item, index), u, v)
    return image


def render_target(image, camera, objects):
    from ...robot.model.geometry import project
    if not objects:
        return image, []
    pixels, depths, visible = project(camera, np.asarray([o["centre_m"] for o in objects], dtype=np.float64))
    shown = []
    grow = overlay_scale(image)
    for index, (pixel, ok) in enumerate(zip(pixels, visible)):
        if ok:
            oid = _object_id(objects[index], index)
            cv2.circle(image, tuple(np.rint(pixel).astype(int)), int(round(3 * grow)), (0, 255, 255), -1)
            _draw_label(image, oid, pixel[0], pixel[1] - 12 * grow)
            shown.append(oid)
    return image, shown


def triangulate_object_centres(camera_src, camera_dst, objects, dst_pixels, *, max_skew_gap_m=0.035):
    """利用副视角（如腕部相机）匹配像素对主视角 object_table 的三维中心与高度执行双目三角化深度锚定修正。"""
    from .scene3d_helpers import triangulate_rays
    K1, T1 = camera_src["K"], camera_src["T"]
    K2, T2 = camera_dst["K"], camera_dst["T"]
    anchored = []
    for index, obj in enumerate(objects):
        oid = _object_id(obj, index)
        dst_uv = (dst_pixels or {}).get(oid) if isinstance(dst_pixels, dict) else (
            dst_pixels[index] if isinstance(dst_pixels, (list, tuple)) and index < len(dst_pixels) else None
        )
        src_uv = obj.get("label_px")
        if dst_uv is None or src_uv is None or len(dst_uv) != 2 or len(src_uv) != 2:
            continue
        tri = triangulate_rays(src_uv[0], src_uv[1], K1, T1, dst_uv[0], dst_uv[1], K2, T2)
        if tri.get("status") != "ok" or tri.get("skew_gap_m") is None or tri["skew_gap_m"] > max_skew_gap_m:
            continue
        pt = tri["point_m"]
        old_c = [float(v) for v in obj["centre_m"]]
        dz = float(pt[2]) - old_c[2]
        anchored.append(
            dict(
                id=oid,
                vggt_centre_m=[_r(v) for v in old_c],
                stereo_centre_m=pt,
                dz_correction_m=_r(dz, 4),
                stereo_top_z_m=_r(float(obj["top_z_m"]) + dz) if obj.get("top_z_m") is not None else None,
                stereo_bottom_z_m=_r(float(obj["bottom_z_m"]) + dz) if obj.get("bottom_z_m") is not None else None,
                skew_gap_m=tri["skew_gap_m"],
            )
        )
    return anchored


def mask_centre_px(item):
    """[u, v] (original-image px) of an object's mask: the area centroid of its outline ``contour_px``; ``label_px``
    (the mask's most interior pixel) when the outline has fewer than 3 points or no area; None when neither exists."""
    contour = np.asarray(item.get("contour_px") or [], dtype=np.float32).reshape(-1, 2)
    if len(contour) >= 3:
        moments = cv2.moments(contour.reshape(-1, 1, 2))
        if abs(moments["m00"]) > 1e-9:
            return [float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])]
    label = item.get("label_px")
    return [float(label[0]), float(label[1])] if isinstance(label, (list, tuple)) and len(label) == 2 else None


def _pinhole_px(camera, pixel):
    """An original-image pixel as the pixel of the same ray in the ideal pinhole image of ``camera["K"]`` (a camera
    without ``D`` or with zero distortion: unchanged)."""
    D = np.asarray(camera.get("D", []), dtype=np.float64).reshape(-1)
    if not D.size or not np.any(np.abs(D) > 1e-12):
        return [float(pixel[0]), float(pixel[1])]
    K = np.asarray(camera["K"], dtype=np.float64)
    out = cv2.undistortPoints(np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2), K, D, P=K).reshape(2)
    return [float(out[0]), float(out[1])]


def _is_held_row(item):
    return str(item.get("support") or "").startswith("held_") or item.get("quality") == "held_fk_depth"


def cross_view_rows(camera_src, objects_src, camera_dst, objects_dst, *, max_centre_distance_m=CROSS_VIEW_MAX_CENTRE_M,
                    min_parallax_deg=CROSS_VIEW_MIN_PARALLAX_DEG, pairs_out=None):
    """Two-view check of object-table rows against the object table of a second calibrated camera of the same
    observation. Returns ``(rows, unmatched, skipped)``."""
    from .scene3d_helpers import triangulate_rays
    K1, T1 = np.asarray(camera_src["K"], dtype=np.float64), np.asarray(camera_src["T"], dtype=np.float64)
    K2, T2 = np.asarray(camera_dst["K"], dtype=np.float64), np.asarray(camera_dst["T"], dtype=np.float64)

    def direction(K, T, pixel):
        ray = T[:3, :3] @ np.linalg.inv(K) @ np.array([pixel[0], pixel[1], 1.0])
        return ray / max(float(np.linalg.norm(ray)), 1e-12)

    def usable(camera, objects, skipped):
        out = []
        for index, item in enumerate(objects or []):
            oid = _object_id(item, index)
            centre = item.get("centre_m")
            pixel = mask_centre_px(item)
            if _is_held_row(item):
                skipped[oid] = "held_row"
            elif not isinstance(centre, (list, tuple)) or len(centre) != 3:
                skipped[oid] = "no_centre"
            elif pixel is None:
                skipped[oid] = "no_mask_pixel"
            else:
                out.append((index, oid, _pinhole_px(camera, pixel), np.asarray(centre, dtype=np.float64)))
        return out

    skipped = {}
    source, target = usable(camera_src, objects_src, skipped), usable(camera_dst, objects_dst, {})
    rows, used = [], set()
    if source and target:
        distance = np.array([[float(np.linalg.norm(a[3] - b[3])) for b in target] for a in source])
        for si, (i, oid, src_px, src_centre) in enumerate(source):
            ti = int(np.argmin(distance[si]))
            if distance[si, ti] > max_centre_distance_m or int(np.argmin(distance[:, ti])) != si:
                continue
            j, _, dst_px, dst_centre = target[ti]
            used.add(i)
            if pairs_out is not None:
                pairs_out.append((i, j))
            found = triangulate_rays(src_px[0], src_px[1], K1, T1, dst_px[0], dst_px[1], K2, T2)
            cosine = float(np.clip(direction(K1, T1, src_px) @ direction(K2, T2, dst_px), -1.0, 1.0))
            parallax = float(np.degrees(np.arccos(cosine)))
            ok = found.get("status") == "ok" and found.get("point_m") is not None and parallax >= min_parallax_deg
            point = np.asarray(found["point_m"], dtype=np.float64) if ok else None
            row = dict(id=oid, target_view_centre_m=[_r(v) for v in dst_centre],
                       target_view_top_z=_r(objects_dst[j].get("top_z_m")),
                       target_view_minus_row_m=[_r(v) for v in dst_centre - src_centre],
                       stereo_centre_m=None if point is None else [_r(v) for v in point],
                       stereo_minus_row_m=None if point is None else [_r(v) for v in point - src_centre],
                       ray_gap_m=_r(found["skew_gap_m"], 4) if ok else None, parallax_deg=_r(parallax, 1))
            if int((distance[si] <= max_centre_distance_m).sum()) > 1:
                row["ambiguous"] = True
            rows.append(row)
    unmatched = [oid for i, oid, _, _ in source if i not in used]
    return rows, unmatched, skipped


def _robot_occlusion(runtime, tools, full, observation, camera, camera_id, prepared, support_z):
    """``(value, diagnostics)`` of ``value.robot_occlusion`` (profile tag ``robot_occlusion``; robot_view.occlusion)."""
    from . import held_perception as hp
    from . import robot_view
    started = time.monotonic()
    try:
        if prepared is not None:
            body, meshes, poses, robot = prepared.body, prepared.meshes, prepared.poses, prepared.mask
            wrist = prepared.wrist_cameras
        else:
            body, body_profile = hp.body_and_profile(runtime)
            meshes = body.meshes(runtime)
            poses = hp.arm_link_poses(tools, full, full["observation_id"], observation)
            robot, wrist = None, body_profile.wrist_cameras
        if not meshes or not poses:
            return dict(unavailable="robot_meshes_or_fk_unavailable"), dict(error="robot_meshes_or_fk_unavailable")
    except Exception as error:
        return dict(unavailable="robot_meshes_or_fk_unavailable"), dict(error=type(error).__name__ + ":" + str(error)[:120])
    try:
        robot, base = robot_view.silhouettes(body, meshes, poses, camera, camera_id, wrist, robot=robot)
        value, diagnostics = robot_view.occlusion(runtime, camera, camera_id, robot, base, support_z,
                                                  full["observation_id"])
    except Exception as error:
        return dict(unavailable="render_failed"), dict(error=type(error).__name__ + ":" + str(error)[:120])
    diagnostics["latency_s"] = round(time.monotonic() - started, 4)
    return value, diagnostics


def _support_chain(objects, index):
    """Indices from row ``index`` down through its supporting rows to the one standing on the table, or None."""
    chain, seen = [], set()
    while isinstance(index, int) and 0 <= index < len(objects) and index not in seen:
        seen.add(index)
        chain.append(index)
        support = objects[index].get("support")
        if support == "table":
            return chain
        index = support
    return None


def contact_rows(camera_src, objects_src, camera_dst, objects_dst, rows, pairs, support_z_m, max_height_m):
    """Adds ``contact_top_z`` / ``contact_top_z_range`` / ``contact_fit_iou`` (m, m, 0..1), or ``contact_reason``,
    to each paired cross-view row in ``rows``."""
    from .contact_height import contact_top, pinhole_polygon
    partner = dict(pairs)
    by_id = {row["id"]: row for row in rows}

    def polygons(camera, items, indices):
        return [pinhole_polygon(camera, items[k]["contour_px"]) for k in indices
                if len(items[k].get("contour_px") or []) >= 3]
    for i, j in pairs:
        oid = _object_id(objects_src[i], i) if 0 <= i < len(objects_src) else ("o%d" % (i + 1))
        row = by_id.get(oid)
        if row is None:
            continue
        chain = _support_chain(objects_src, i)
        if chain is None:
            row["contact_reason"] = "not_on_table"
            continue
        targets = sorted({partner[k] for k in chain if k in partner})
        try:
            found = contact_top([(camera_src["K"], camera_src["T"], polygons(camera_src, objects_src, chain),
                                  (int(camera_src["width"]), int(camera_src["height"]))),
                                 (camera_dst["K"], camera_dst["T"], polygons(camera_dst, objects_dst, targets),
                                  (int(camera_dst["width"]), int(camera_dst["height"])))], support_z_m, max_height_m)
        except Exception:
            found = dict(contact_reason="fit_failed")
        unpaired = [_object_id(objects_src[k], k) if 0 <= k < len(objects_src) else ("o%d" % (k + 1))
                    for k in chain if k not in partner]
        if unpaired and "contact_top_z" in found:
            found["contact_unpaired_support"] = unpaired
        row.update(found)
    return rows


def attached_table(runtime, observation_id):
    """Object table attached to ``observation_id`` (delegates to ``obs3d.attached_table`` without importing ``pose_skill``)."""
    from .obs3d import attached_table as _attached_table
    return _attached_table(runtime, observation_id)


def _target_view_objects(runtime, request, target_id, labels, timeout_s, max_objects, slack_s):
    """(rows, origin) of the object table of camera ``target_id`` for the request's observation."""
    from .obs3d import _memo
    from ..world_model import get_world_observation
    obs_id = request["observation_id"]
    wobs = get_world_observation(runtime, obs_id)
    if wobs is not None and wobs.camera_id == target_id:
        raw = wobs.raw_objects()
        if raw is not None:
            return raw, "attached"
    entry = _memo(runtime, obs_id).get("object_table")
    result = entry[2] if isinstance(entry, tuple) and len(entry) == 3 and isinstance(entry[2], dict) else {}
    rows = (result.get("diagnostics") or {}).get("objects")
    if (result.get("value") or {}).get("camera_id") == target_id and isinstance(rows, list):
        return rows, "attached"
    query = {"kind": KIND, "observation_id": request["observation_id"], "camera_id": target_id}
    if labels:
        query["labels"] = list(labels)
    result, _ = object_table_query(runtime, query, timeout_s=timeout_s, max_objects=max_objects, slack_s=slack_s,
                                   track_held=False)
    if result.get("status") != "conditional":
        raise ServiceError(str(result.get("reason") or "object_table_unavailable")[:120])
    return (result.get("diagnostics") or {}).get("objects") or [], "computed"


def object_table_query(
    runtime,
    request,
    *,
    timeout_s=None,
    max_objects=None,
    slack_s=5.0,
    reanchor=None,
    include_overlay=True,
    query_type=KIND,
    on_before_send=None,
    track_held=True,
):
    """Returns (result, overlays) where overlays is a list of (camera_id, encoded JPEG bytes, original rgb_ref).

    timeout_s / max_objects default to features.vggt_timeout_s / object_table_max_objects (obs3d passes its own).
    ``track_held`` (features.held_perception only): True (default) for a table the model receives (geometry_query,
    the obs3d attachment): held perception keeps it as a candidate pre-grasp table and numbers it (``object table
    #<n>`` in held-row notes). False for a table the model never receives (the cross-view check's target-camera
    table): held rows are still computed, but the table is not kept and takes no number."""
    from ...robot.model.geometry import GeometryTools
    from ...core.storage.images import encode_image
    from PIL import Image
    features = runtime.config.features
    profile = embodiment_of(runtime)
    request = dict(request)
    camera_id = request.get("camera_id") or default_camera(runtime)
    full = {**request, "camera_id": camera_id, **runtime.identity, "schema_version": "0.1"}
    tools = GeometryTools(runtime.scene, settings=runtime.image_settings, profile=profile)
    started = time.monotonic()
    limit = float(features.vggt_timeout_s if timeout_s is None else timeout_s)
    result = {"schema_version": "0.1", **runtime.identity, "world_kind": "internal", "query_type": query_type,
              "observation_ids": [request.get("observation_id")], "status": "supported", "frame_id": profile.frame_id,
              "unit": "m", "value": None}
    notified_before_send = False

    def _notify_before_send():
        nonlocal notified_before_send
        if not notified_before_send and callable(on_before_send):
            notified_before_send = True
            on_before_send()

    try:
        obs_id_req = request.get("observation_id")
        if not isinstance(obs_id_req, str) or not obs_id_req:
            raise ValueError("invalid_observation_id")
        try:
            observation = runtime.scene.observation(runtime.identity, obs_id_req)
        except KeyError as exc:
            raise ValueError("unknown_observation") from exc
        if not request.get("camera_id"):
            camera_id = default_camera(runtime, observation)
            full["camera_id"] = camera_id
        camera = tools.camera(full, full["observation_id"], camera_id)
        targets = []
        for target_id in request.get("target_camera_ids") or []:
            if target_id == camera_id:
                continue
            targets.append((target_id, tools.camera(full, full["observation_id"], target_id)))
        try:
            _, image_bytes = runtime.scene.resolve_artifact(camera["frame"]["rgb_ref"])
        except KeyError as exc:
            raise ValueError("unknown_artifact") from exc
        rectified = rectified_cached(camera, image_bytes)   # pinhole service: distorted cameras are sent undistorted
        image_bytes = rectified.image_bytes
        prepared, held_error = None, None
        if held_perception_enabled(runtime):   # features.held_perception: FK robot pixel mask + held rows
            try:
                from .held_perception import Prepared
                prepared = Prepared(runtime, tools, full, observation, camera, camera_id)
                if not prepared.meshes or not prepared.poses:
                    prepared, held_error = None, "robot_meshes_or_fk_unavailable"
            except Exception as error:   # the capsule path below stays available
                prepared, held_error = None, type(error).__name__ + ":" + str(error)[:120]
        segments = robot_segments(tools, full, observation, profile.gripper.mask_segment)
        support_z = effective_support_z_m(features, camera, profile)
        cache_key = make_image_cache_key(rectified.sha256, camera["K"], camera["T"], support_z)
        body = dict(image_b64=base64.b64encode(image_bytes).decode("ascii"), K=np.asarray(camera["K"]).tolist(),
                    T_world_camera=np.asarray(camera["T"]).tolist(), support_z_m=support_z,
                    robot_segments_m=segments, robot_radius_m=LINK_RADIUS_M,
                    max_objects=int(features.object_table_max_objects if max_objects is None else max_objects),
                    max_object_height_m=float(features.object_table_max_height_m), timeout_s=limit)
        ep_id = (getattr(runtime, "identity", None) or {}).get("episode_id")
        eff_reanchor = reanchor if isinstance(reanchor, dict) and reanchor else (
            request.get("reanchor") if isinstance(request.get("reanchor"), dict) and request.get("reanchor") else None
        )
        if persistent_tracking_enabled(runtime) and ep_id:
            body["session_id"] = str(ep_id)
        if eff_reanchor is not None:
            body["reanchor"] = eff_reanchor
            if "session_id" not in body:
                body["session_id"] = str(ep_id or "default")
        box = str(getattr(features, "object_table_head_exclude_px", "") or "")
        if camera_id == DEFAULT_CAMERA and box.strip():   # the X5 head-camera mount; never another camera
            body["exclude_boxes_px"] = [[int(float(v)) for v in box.split(",")]]
        if prepared is not None:
            body.update(prepared.body_fields())
            # 若任一臂 FK/网格不完整，同步保留胶囊线段掩码兜底
            if segments and (prepared.keep_segments or getattr(prepared, "mask_incomplete", False)):
                body["robot_segments_m"] = segments
        if request.get("pixels"):
            raw_pixels = request["pixels"]
            if not isinstance(raw_pixels, (list, tuple)) or any(
                not isinstance(pt, (list, tuple)) or len(pt) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in pt)
                for pt in raw_pixels
            ):
                raise ValueError("invalid_pixels")
            body["pixels"] = [[float(u), float(v)] for u, v in rectified.to_rectified(raw_pixels)]
        labels = [str(text).strip()[:80] for text in request.get("labels") or [] if str(text).strip()]
        if labels:
            body["labels"] = labels
        if body.get("session_id"):
            mark_active = getattr(runtime, "mark_perception_session_active", None)
            if callable(mark_active):
                mark_active(body["session_id"])
        track_upload = urllib.request.urlopen is _STDLIB_URLOPEN and query_service is _DEFAULT_QUERY_SERVICE
        upload_started = False
        upload_ok = False

        def _on_ot_sent():
            nonlocal upload_started
            if track_upload and not upload_started:
                upload_started = True
                note_image_upload_started(features.vggt_endpoint, cache_key)
            _notify_before_send()

        try:
            if callable(on_before_send):
                answer = query_service(features.vggt_endpoint, body, limit, slack_s, on_sent=_on_ot_sent)
            else:
                if track_upload:
                    upload_started = True
                    note_image_upload_started(features.vggt_endpoint, cache_key)
                answer = query_service(features.vggt_endpoint, body, limit, slack_s)
            if prepared is not None and answer.get("version") == "object-table-v5" and not answer.get("robot_mask"):
                # Backward-compatibility with an un-restarted object-table-v5 service: re-run with capsule segments
                # if available so robot links are still filtered while keeping client-side held rows.
                segments = robot_segments(tools, full, observation, profile.gripper.mask_segment)
                if segments:
                    fallback_body = {
                        k: v for k, v in body.items() if k not in ("robot_mask_b64", "robot_mask_shape", "held")
                    }
                    fallback_body["robot_segments_m"] = segments
                    answer = query_service(features.vggt_endpoint, fallback_body, limit, slack_s)
            upload_ok = True
            if track_upload:
                note_image_upload_finished(features.vggt_endpoint, cache_key, ok=True, answer=answer)
        finally:
            if upload_started and not upload_ok:
                note_image_upload_finished(features.vggt_endpoint, cache_key, ok=False)
        # 纵深防御：若服务端携带 backend_error 且分割未产出任何跟踪/文本结果与物体（无显式 pixels），拒绝把服务故障伪装为空桌面
        if (
            answer.get("backend_error")
            and not (answer.get("objects") or [])
            and not body.get("pixels")
            and not str(answer.get("backend") or "").startswith(("sam31_track", "sam31_text"))
        ):
            raise ServiceError(f"backend_error:{answer['backend_error']}")
    except ServiceError as error:
        _notify_before_send()
        result.update(status="unknown", reason="object_table_unavailable:" + str(error)[:200])
        result["duration_s"] = time.monotonic() - started
        return result, []
    except ValueError as error:
        _notify_before_send()
        err_text = str(error)[:200]
        if err_text.startswith("invalid_") or err_text == "pixel_out_of_bounds":
            result.update(status="invalid_input", reason=err_text)
        else:
            result.update(status="unknown", reason="object_table_unavailable:" + err_text)
        result["duration_s"] = time.monotonic() - started
        return result, []
    finally:
        _notify_before_send()
    objects = rectified.map_objects(answer.get("objects") or [])
    held_meta = {}
    if prepared is not None:
        rectified.map_objects([item for item in answer.get("held") or [] if isinstance(item, dict)])
        objects, held_meta = prepared.apply(objects, answer, features.vggt_endpoint, limit, record=track_held)
    objects = _normalize_persistent_ids(objects, held_meta=held_meta)
    occlusion = occlusion_diagnostics = None
    if track_held and ROBOT_OCCLUSION_TAG in profile.supports:   # a model-facing table: report the arm's silhouette
        occlusion, occlusion_diagnostics = _robot_occlusion(runtime, tools, full, observation, camera, camera_id,
                                                            prepared, support_z)
    overlays = []
    projected = {}
    if include_overlay:
        source = render_source(tools.image(camera), objects)
        encoded, _ = encode_image(Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB)), tools.image_settings, storage=True)
        overlays.append((camera_id, encoded, camera["frame"]["rgb_ref"]))
        for target_id, target in targets:
            drawn, shown = render_target(tools.image(target), target, objects)
            encoded, _ = encode_image(Image.fromarray(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)), tools.image_settings, storage=True)
            overlays.append((target_id, encoded, target["frame"]["rgb_ref"]))
            projected[target_id] = shown
    requested = []
    for item in answer.get("requested") or []:
        index = item.get("index")
        pixel = item.get("pixel")
        if rectified.active and isinstance(pixel, (list, tuple)) and len(pixel) == 2:
            pixel = [round(float(v), 1) for v in rectified.to_original([pixel])[0]]
        req_oid = (
            _object_id(objects[index], index)
            if isinstance(index, int) and 0 <= index < len(objects)
            else ("o%d" % (index + 1) if isinstance(index, int) else None)
        )
        requested.append(dict(pixel=pixel, status=item.get("status"), id=req_oid))
    with_label = bool(body.get("labels"))
    track_meta = _build_track_meta(answer, objects)
    from .local_kdtree_field import derive_table_spatial_meta
    spatial_meta = derive_table_spatial_meta(objects, answer.get("support_z_m"))
    value = dict(source=SOURCE, measured=False, camera_id=camera_id, support_z_m=_r(answer.get("support_z_m")),
                 columns=COLUMNS + ["label"] if with_label else COLUMNS, rows=rows_from(objects, with_label),
                 grasp_hints=grasp_hints_from(objects, answer.get("support_z_m"), profile),
                 track_meta=track_meta,
                 omitted=answer.get("omitted", 0), overlay_cameras=[name for name, _, _ in overlays])
    if spatial_meta:
        value["spatial_meta"] = spatial_meta
    if requested:
        value["requested"] = requested
    if with_label:
        value["labels"] = [dict(label=item.get("label"), status=item.get("status"),
                                ids=[_object_id(objects[index], index) if 0 <= index < len(objects) else ("o%d" % (index + 1))
                                     for index in item.get("indices") or [] if isinstance(index, int)],
                                **({"filtered": item["filtered"]} if item.get("filtered") else {}))
                           for item in answer.get("labels") or []]
    if projected:
        value["projected_ids"] = projected
    if held_meta:
        value["held"] = {
            (_object_id(objects[index], index) if 0 <= index < len(objects) else ("o%d" % (index + 1))): meta
            for index, meta in sorted(held_meta.items())
        }
    if occlusion is not None:
        value["robot_occlusion"] = occlusion
    mask_degraded = {}
    seg_degraded = getattr(segments, "degraded_cameras", None)
    if seg_degraded:
        mask_degraded["segment_fk_cameras"] = dict(seg_degraded)
    if prepared is not None and getattr(prepared, "mask_incomplete", False):
        if getattr(prepared, "degraded_arms", None):
            mask_degraded["degraded_arms"] = dict(prepared.degraded_arms)
        if getattr(prepared, "degraded_meshes", None):
            mask_degraded["degraded_meshes"] = list(prepared.degraded_meshes)
    elif held_error and held_perception_enabled(runtime):
        mask_degraded["held_mask_error"] = held_error
    if mask_degraded:
        value["robot_mask_degraded"] = mask_degraded
    id_semantics = "Ids are persistent episode tracks when track_meta.persistent_ids is true, else per call"
    assumptions = [
        "Values are VGGT-Omega monocular-depth estimates scaled to the configured table plane "
        "(top = 98th-percentile height of the visible mask points; length/width/yaw from the "
        "visible points); tilt is given for elongated objects only; low:<flags> marks implausible "
        f"depth. {id_semantics}; target views show projected centres, not detections."
    ]
    cross_diagnostics = {}
    if targets and CROSS_VIEW_TAG in profile.supports:   # two-view check against each target camera's own masks
        cross = {}
        for target_id, target in targets:
            t0 = time.monotonic()
            try:
                rows_dst, origin = _target_view_objects(runtime, request, target_id, labels, limit,
                                                        body["max_objects"], slack_s)
                pairs = []
                matched, unmatched, skipped = cross_view_rows(camera, objects, target, rows_dst, pairs_out=pairs)
                if CONTACT_TOP_TAG in profile.supports:
                    contact_rows(camera, objects, target, rows_dst, matched, pairs, support_z,
                                 float(features.object_table_max_height_m))
                cross[target_id] = dict(method="mask_centre_rays", target_table=origin, rows=matched,
                                        unmatched=unmatched, skipped=skipped)
                cross_diagnostics[target_id] = dict(target_objects=len(rows_dst), matched=len(matched), origin=origin,
                                                    latency_s=round(time.monotonic() - t0, 3))
            except Exception as error:   # the table itself stays valid without the optional check
                cross[target_id] = dict(unavailable=str(error)[:120] or type(error).__name__)
                cross_diagnostics[target_id] = dict(error=type(error).__name__ + ":" + str(error)[:120])
        value["cross_view"] = cross
        if CONTACT_TOP_TAG in profile.supports:
            from .contact_height import ASSUMPTION as CONTACT_ASSUMPTION
            assumptions.append(CONTACT_ASSUMPTION)
        assumptions.append(CROSS_VIEW_ASSUMPTION)   # the last entry states the pairing bounds (docs)
    result.update(status="conditional", value=value, frame_ids=[camera["frame"]["frame_id"]], assumptions=assumptions)
    diagnostics_objects = []
    for idx, o in enumerate(objects):
        supp = o.get("support")
        supp_id = (_object_id(objects[supp], supp) if 0 <= supp < len(objects) else ("o%d" % (supp + 1))) if isinstance(supp, int) else supp
        item_diag = {
            "id": _object_id(o, idx),
            "label": o.get("label"),
            "label_score": _r(o.get("label_score")),
            "labels": o.get("labels"),
            "bbox_px": o.get("bbox_px"),
            "contour_px": o.get("contour_px"),
            "label_px": o.get("label_px"),
            "centre_m": [_r(v) for v in o.get("centre_m", [])] if o.get("centre_m") is not None else None,
            "top_z_m": _r(o.get("top_z_m")),
            "bottom_z_m": _r(o.get("bottom_z_m")),
            "height_m": _r(o.get("height_m")),
            "length_m": _r(o.get("length_m")),
            "width_m": _r(o.get("width_m")),
            "yaw_deg": _r(o.get("yaw_deg"), 0),
            "axis_tilt_deg": _r(o.get("axis_tilt_deg"), 0),
            "support": supp_id,
            "quality": o.get("quality", "ok"),
            "flags": o.get("flags") or [],
            "mask_px": o.get("mask_px"),
            "n_points": o.get("n_points"),
            "iou_pred": _r(o.get("iou_pred")),
            "conf_rank_p50": _r(o.get("conf_rank_p50")),
            "contact_gap_m": _r(o.get("contact_gap_m")),
            "cluster_frac": _r(o.get("cluster_frac")),
            "elongation": _r(o.get("elongation")),
            "track_conf": o.get("track_conf"),
            "track_status": o.get("track_status"),
            "cavity_depth_m": o.get("cavity_depth_m"),
            "surface_slices_m": o.get("surface_slices_m"),
        }
        diagnostics_objects.append(item_diag)
    result["diagnostics"] = dict(service_version=answer.get("version"), backend=answer.get("backend"),
                                 latency_s=round(time.monotonic() - started, 3),
                                 service_timing=answer.get("timing"), dropped=answer.get("dropped"),
                                 n_prompts=answer.get("n_prompts"), robot_segments=len(segments),
                                 support_fit=answer.get("support_fit"), scale=_r(answer.get("scale")),
                                 objects=diagnostics_objects)
    if cross_diagnostics:
        result["diagnostics"]["cross_view"] = cross_diagnostics
    if occlusion_diagnostics is not None:
        result["diagnostics"]["robot_occlusion"] = occlusion_diagnostics
    if prepared is not None or held_error:
        result["diagnostics"]["held_perception"] = (dict(prepared.diagnostics, service_held_s=answer.get("held_s"))
                                                    if prepared is not None else dict(error=held_error))
    if mask_degraded:
        result["diagnostics"]["robot_mask_degraded"] = mask_degraded
    result["duration_s"] = time.monotonic() - started
    return result, overlays


def _is_held_table_row(row):
    """判断 object_table 行是否为随夹爪运动的夹持物合成行。"""
    return str(row.get("support") or "").startswith("held_") or row.get("quality") == "held_fk_depth"


def _table_has_persistent_ids(table):
    """检查 object_table 是否携带持久追踪身份标志（兼容 track_meta 与 tracking 元信息）。"""
    if not isinstance(table, dict):
        return False
    if bool((table.get("track_meta") or {}).get("persistent_ids")):
        return True
    return bool((table.get("tracking") or {}).get("persistent_ids"))


def _principal_yaw_delta_deg(before_yaw, after_yaw):
    """计算主分支 [-90, 90) 上的最短有向偏航角差（单位 deg，180° 对称主轴）。"""
    if (
        not isinstance(before_yaw, (int, float))
        or isinstance(before_yaw, bool)
        or not isinstance(after_yaw, (int, float))
        or isinstance(after_yaw, bool)
    ):
        return None
    b_val, a_val = float(before_yaw), float(after_yaw)
    if not (math.isfinite(b_val) and math.isfinite(a_val)):
        return None
    diff = ((a_val - b_val + 90.0) % 180.0) - 90.0
    rounded = round(diff, 2)
    return 0.0 if abs(rounded) < 1e-9 else rounded


def compute_object_table_delta(before_table, after_table, *, before_observation_id=None,
                               move_threshold_m=0.015, match_radius_m=0.14):
    """对比动作前后两帧 3D object_table，识别发生明显位移/抬升、从原位消失或新出现的 3D 物体。

    当前后表均具备持久追踪身份（track_meta.persistent_ids == True）时，优先按相同持久 id 精确配对，
    彻底解除 0.14m 最近邻距离上限；未命中持久 id 的剩余行或旧表回退使用 0.14m 最近邻匹配。
    未匹配的前帧行若出现在本次手臂遮挡报告中，单列其投影覆盖比例；原位与当前位置均未得到确认。
    """
    if not isinstance(before_table, dict) or not isinstance(after_table, dict):
        return None
    b_cols, a_cols = list(before_table.get("columns") or []), list(after_table.get("columns") or [])
    b_rows = [
        row for r in (before_table.get("rows") or [])
        if isinstance(r, (list, tuple)) and len(r) >= len(b_cols)
        and not _is_held_table_row(row := dict(zip(b_cols, r)))
    ]
    a_rows = [
        row for r in (after_table.get("rows") or [])
        if isinstance(r, (list, tuple)) and len(r) >= len(a_cols)
        and not _is_held_table_row(row := dict(zip(a_cols, r)))
    ]
    if not b_rows and not a_rows:
        return None

    persistent_mode = _table_has_persistent_ids(before_table) and _table_has_persistent_ids(after_table)
    matched_b = {}
    used_after = set()

    if persistent_mode:
        stationary_tol_m = 0.035
        id_pairs = []
        for b_idx, b in enumerate(b_rows):
            bid = b.get("id")
            bc = b.get("centre_m")
            if not isinstance(bid, str) or not bid or not isinstance(bc, list) or len(bc) != 3:
                continue
            bx, by, bz = float(bc[0]), float(bc[1]), float(bc[2])
            for a_idx, a in enumerate(a_rows):
                ac = a.get("centre_m")
                if a.get("id") == bid and isinstance(ac, list) and len(ac) == 3:
                    ax, ay, az = float(ac[0]), float(ac[1]), float(ac[2])
                    d3 = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2)
                    if d3 > match_radius_m:
                        origin_still_occupied = any(
                            oa.get("id") != bid
                            and isinstance(oa.get("centre_m"), list)
                            and len(oa["centre_m"]) == 3
                            and math.dist((bx, by, bz), (float(oa["centre_m"][0]), float(oa["centre_m"][1]), float(oa["centre_m"][2]))) <= stationary_tol_m
                            for oa in a_rows
                        )
                        dest_previously_occupied = any(
                            ob.get("id") != bid
                            and isinstance(ob.get("centre_m"), list)
                            and len(ob["centre_m"]) == 3
                            and math.dist((ax, ay, az), (float(ob["centre_m"][0]), float(ob["centre_m"][1]), float(ob["centre_m"][2]))) <= stationary_tol_m
                            for ob in b_rows
                        )
                        if origin_still_occupied and dest_previously_occupied:
                            continue
                    id_pairs.append((d3, b_idx, a_idx))
        id_pairs.sort(key=lambda item: (round(item[0], 6), item[1], item[2]))
        for d3, b_idx, a_idx in id_pairs:
            if b_idx in matched_b or a_idx in used_after:
                continue
            matched_b[b_idx] = (a_idx, d3, True)
            used_after.add(a_idx)

    # IDs may be local even when the same observation is queried with different cameras or limits.
    # Both producers export the same rounded centre for the source row.
    hidden = {
        (item.get("id"), tuple(item["centre_m"])): item["covered"]
        for item in (after_table.get("robot_occlusion") or {}).get("hidden", [])
        if isinstance(item, dict) and before_observation_id and item.get("seen_in") == before_observation_id
        and isinstance(item.get("centre_m"), list)
        and type(item.get("covered")) in (float, int) and 0 < item["covered"] <= 1
    }
    moved, unchanged, vacated, unmatched_under_arm = [], [], [], []
    for b_idx, b in enumerate(b_rows):
        bc = b.get("centre_m")
        if not isinstance(bc, list) or len(bc) != 3:
            continue
        bx, by, bz = float(bc[0]), float(bc[1]), float(bc[2])
        if b_idx in matched_b:
            best_idx, best_d, is_persistent = matched_b[b_idx]
        else:
            best_idx, best_d, is_persistent = None, math.inf, False
            for idx, a in enumerate(a_rows):
                if idx in used_after:
                    continue
                ac = a.get("centre_m")
                if not isinstance(ac, list) or len(ac) != 3:
                    continue
                d3 = math.sqrt((bx - float(ac[0])) ** 2 + (by - float(ac[1])) ** 2 + (bz - float(ac[2])) ** 2)
                if d3 <= match_radius_m and d3 < best_d:
                    best_idx, best_d = idx, d3
            if best_idx is None:
                covered = hidden.get((b["id"], tuple(bc)))
                if covered is None:
                    vacated.append(dict(before_id=b["id"], before_centre_m=[round(bx, 3), round(by, 3), round(bz, 3)]))
                else:
                    unmatched_under_arm.append(dict(before_id=b["id"], covered=covered))
                continue
            used_after.add(best_idx)
        a = a_rows[best_idx]
        ax, ay, az = (float(v) for v in a["centre_m"])
        dx, dy, dz = round(ax - bx, 4), round(ay - by, 4), round(az - bz, 4)
        if best_d >= move_threshold_m or abs(dz) >= move_threshold_m:
            moved_item = dict(
                before_id=b["id"],
                after_id=a["id"],
                from_centre_m=[round(bx, 3), round(by, 3), round(bz, 3)],
                to_centre_m=[round(ax, 3), round(ay, 3), round(az, 3)],
                delta_m=[dx, dy, dz],
                displacement_m=round(best_d, 4),
                lifted=bool(dz >= 0.020),
            )
            yaw_delta = _principal_yaw_delta_deg(b.get("yaw_deg"), a.get("yaw_deg"))
            if is_persistent:
                moved_item["yaw_delta_deg"] = yaw_delta
                moved_item["persistent_match"] = True
            elif yaw_delta is not None:
                moved_item["yaw_delta_deg"] = yaw_delta
            moved.append(moved_item)
        else:
            unchanged.append(dict(before_id=b["id"], after_id=a["id"]))
    appeared = [
        dict(after_id=a["id"], centre_m=[round(float(v), 3) for v in a["centre_m"]])
        for idx, a in enumerate(a_rows)
        if idx not in used_after and isinstance(a.get("centre_m"), list) and len(a["centre_m"]) == 3
    ]
    delta = dict(moved=moved, vacated=vacated, appeared=appeared, unchanged_count=len(unchanged))
    if unmatched_under_arm:
        delta["unmatched_under_arm"] = unmatched_under_arm
    return delta
