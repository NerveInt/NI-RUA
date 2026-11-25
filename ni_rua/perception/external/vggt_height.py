"""features.vggt_height (on by default with external_perception): estimated object height for pixel_plane.

With the switch on, a pixel_plane request may set plane_source="vggt" instead of supplying the plane
height itself. The harness sends the saved RGB image of that camera, its calibrated K and
T_world_camera, and the configured support (table) height to a local VGGT-Omega service
(features.vggt_endpoint). The service returns the estimated top height of the object under the pixel
(monocular depth, scaled per image so that the fitted support plane lies at the support height;
see docs/vggt_height.md). The plane is then the horizontal plane at the estimated centre
height (midway between the estimated top and the support), and the alternate plane is the caller's
plane_origin_alt_m, else the caller's plane_origin_m, else the estimated top plane. Everything else is
the unchanged GeometryTools pixel_plane result (point_m, assumption_delta_m, cross_camera_projections,
overlay). Inputs are permitted RGB and calibration only; no simulator truth is read.

With the switch off only enabled() is consulted: tool definitions and handler behaviour are unchanged.
"""
from __future__ import annotations

import base64
import contextlib
import json
import math
import time
import urllib.error
import urllib.request
from copy import deepcopy

import numpy as np

SOURCE = "vggt_estimate"
PLANE_SOURCES = ("vggt",)

FIELD_DESCRIPTION = ("Optional only for pixel_plane in this run: 'caller' (default) uses your plane; 'vggt' sets the plane "
                     "to the estimated height of the object under the pixel (VGGT monocular depth from this RGB image, "
                     "scaled by the calibration and the table plane; an estimate, not measured depth) and makes "
                     "plane_origin_m/plane_normal optional. value.plane_estimate reports the estimated top/centre "
                     "heights and quality.")
DESCRIPTION_SENTENCE = (" In this run pixel_plane also accepts plane_source:'vggt', which replaces your plane height with "
                        "an image-based estimate of the object's centre height at that pixel (alternate: its top); "
                        "check value.plane_estimate.quality.")
_PLANE_ORIGIN_DESCRIPTION = ("Required for pixel_plane unless plane_source is 'vggt': an explicit [x,y,z] point on your "
                            "hypothesized infinite plane, in calibration-frame metres (normally {frame_id}). With "
                            "plane_source 'vggt' it becomes the alternate hypothesis. An intersection is not an observed object surface.")
PLANE_NORMAL_DESCRIPTION_PREFIX = "Required for pixel_plane unless plane_source is 'vggt' (then [0,0,1]): "

# features.vggt_height_mode = "default": every pixel_plane uses the estimate unless plane_source is 'caller'.
DEFAULT_DESCRIPTION_SENTENCE = (" In this run the pixel_plane plane height is estimated from the image (VGGT, centre height "
                                "of the object at the pixel; value.plane_estimate) unless plane_source is 'caller'; a "
                                "plane_origin_m you pass is kept as the alternate hypothesis (assumption_delta_m).")
DEFAULT_FIELD_DESCRIPTION = ("Optional only for pixel_plane: 'vggt' (default in this run) sets the plane to the estimated "
                             "centre height of the object under the pixel (VGGT monocular depth from this RGB image, scaled "
                             "by the calibration and the table plane; an estimate, not measured depth); 'caller' uses your "
                             "plane_origin_m/plane_normal unchanged.")
_DEFAULT_PLANE_ORIGIN_DESCRIPTION = ("Optional for pixel_plane in this run (required with plane_source 'caller'): an explicit "
                                    "[x,y,z] point on your hypothesized infinite plane, in calibration-frame metres (normally "
                                    "{frame_id}). Without plane_source 'caller' it is reported as the alternate plane against "
                                    "the estimated height. An intersection is not an observed object surface.")
# Default-robot (ARX-X5, env_origin) renderings; extend_definition renders the run profile's frame_id.
PLANE_ORIGIN_DESCRIPTION = _PLANE_ORIGIN_DESCRIPTION.format(frame_id="env_origin")
DEFAULT_PLANE_ORIGIN_DESCRIPTION = _DEFAULT_PLANE_ORIGIN_DESCRIPTION.format(frame_id="env_origin")
HORIZONTAL_COS = 0.95  # default mode replaces only near-horizontal caller planes (|n_z| >= this)


def _features(runtime):
    cfg = getattr(runtime, "config", runtime)
    return getattr(cfg, "features", cfg)


def enabled(runtime) -> bool:
    features = _features(runtime)
    return bool(getattr(features, "vggt_height", False))


def mode(runtime):
    """None (switch off), 'optional' or 'default'."""
    if not enabled(runtime):
        return None
    return getattr(_features(runtime), "vggt_height_mode", "optional")


def applies(runtime, request):
    """True when the handler must route this geometry_query request through pixel_plane_with_estimate."""
    selected = mode(runtime)
    if selected is None or not isinstance(request, dict):
        return False
    return "plane_source" in request or (selected == "default" and request.get("kind") == "pixel_plane")


def extend_definition(geometry_schema, descriptions, selected="optional", profile=None):
    """Model-facing schema/description changes, applied only when the switch is on. ``profile``: the run's
    EmbodimentProfile (None = the default ARX-X5 robot) names the usual calibration frame (``frame_id``)."""
    from . import embodiment_of
    frame_id = (profile if profile is not None else embodiment_of(None)).frame_id
    descriptions = dict(descriptions)
    default = selected == "default"
    descriptions["plane_source"] = DEFAULT_FIELD_DESCRIPTION if default else FIELD_DESCRIPTION
    descriptions["plane_origin_m"] = (_DEFAULT_PLANE_ORIGIN_DESCRIPTION if default
                                      else _PLANE_ORIGIN_DESCRIPTION).format(frame_id=frame_id)
    descriptions["plane_normal"] = PLANE_NORMAL_DESCRIPTION_PREFIX + descriptions["plane_normal"].split(": ", 1)[1]
    for branch in geometry_schema["oneOf"]:
        if branch.get("properties", {}).get("kind", {}).get("const") != "pixel_plane":
            continue
        # Two values, so the input field guide does not read a one-value enum as a branch selector.
        branch["properties"]["plane_source"] = dict(type="string", enum=["caller", *PLANE_SOURCES],
                                                    default="vggt" if default else "caller")
        branch["required"] = [key for key in branch["required"] if key not in ("plane_origin_m", "plane_normal")]
        if not default:
            branch.setdefault("anyOf", []).extend([dict(required=["plane_origin_m", "plane_normal"]),
                                                   dict(required=["plane_source"])])
    return descriptions


class ServiceError(RuntimeError):
    pass


def query_service(endpoint, image_bytes, K, T, support_z, pixel, timeout_s, method="sam_point"):
    body = json.dumps(dict(image_b64=base64.b64encode(image_bytes).decode("ascii"), K=np.asarray(K).tolist(),
                           T_world_camera=np.asarray(T).tolist(), support_z_m=float(support_z),
                           pixels=[[float(pixel[0]), float(pixel[1])]], method=method,
                           timeout_s=float(timeout_s))).encode()
    request = urllib.request.Request(endpoint.rstrip("/") + "/height", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s + 5) as response:
            answer = json.loads(response.read())
    except urllib.error.HTTPError as error:
        with contextlib.closing(error):
            raw_err = error.read()
        try:
            answer = json.loads(raw_err)
        except ValueError:
            raise ServiceError("http_" + str(error.code)) from None
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise ServiceError(type(error).__name__ + ":" + str(getattr(error, "reason", error))[:200]) from None
    if not isinstance(answer, dict) or answer.get("status") != "ok":
        raise ServiceError(str((answer if isinstance(answer, dict) else {}).get("error", "service_error"))[:200])
    pixels_out = answer.get("pixels")
    if not isinstance(pixels_out, list) or not pixels_out or not isinstance(pixels_out[0], dict):
        raise ServiceError("malformed_service_payload:pixels")
    item = pixels_out[0]
    if item.get("status") != "ok":
        raise ServiceError(str(item.get("status", "no_estimate")))
    for num_key in ("center_z_m", "top_z_m", "support_z_m", "height_m"):
        val = item.get(num_key)
        if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(float(val)):
            raise ServiceError(f"malformed_service_payload:{num_key}")
    return answer, item


def quality(answer, item):
    fit = answer.get("support_fit", {})
    reasons = []
    if item.get("method") != "sam_point_mask":
        reasons.append("no_object_mask_window_estimate")
    if item.get("n_points", 0) < 30:
        reasons.append("few_points")
    if fit.get("tilt_deg", 0) > 5:
        reasons.append("support_plane_tilted")
    if fit.get("inlier_frac", 1) < 0.1:
        reasons.append("small_support_region")
    if item.get("height_m", 0) < 0.004:
        reasons.append("no_height_above_support")
    # plausibility flags of VGGT service v2 (floating point set, height above max, low depth confidence, contact
    # mismatch); absent from v1 answers, so quality is unchanged there
    reasons.extend(str(flag) for flag in item.get("flags") or [] if str(flag) not in reasons)
    return ("low" if reasons else "ok"), reasons


def pixel_plane_with_estimate(runtime, request):
    """Run pixel_plane with the estimated plane. request is the model's request (identity not yet merged).
    optional mode: only plane_source='vggt'; default mode: every pixel_plane unless plane_source='caller'."""
    from ...robot.model.geometry import GeometryTools
    features = runtime.config.features
    default = getattr(features, "vggt_height_mode", "optional") == "default"
    request = dict(request)
    explicit_source = request.pop("plane_source", None)
    source = explicit_source if explicit_source is not None else ("vggt" if default else None)
    full = {**request, **runtime.identity, "schema_version": "0.1"}
    from . import run_embodiment
    profile = run_embodiment(runtime)   # None: the packaged TCPs are recognised by their tcp_config_id
    tools = GeometryTools(runtime.scene, settings=runtime.image_settings, profile=profile)
    if request.get("kind") != "pixel_plane" or source not in PLANE_SOURCES:
        return tools.execute(full)
    if default and "plane_normal" in full:
        try:
            normal = np.asarray(full["plane_normal"], dtype=np.float64).reshape(3)
            horizontal = bool(np.isfinite(normal).all() and np.linalg.norm(normal) > 0
                              and abs(normal[2]) / np.linalg.norm(normal) >= HORIZONTAL_COS)
        except (ValueError, TypeError):
            horizontal = False
        if not horizontal:  # a caller wall/slope plane is not an object height; leave it to the caller
            result = tools.execute(full)
            if isinstance(result.get("value"), dict):
                result["value"]["plane_estimate"] = dict(source=SOURCE, status="not_applied",
                                                         reason="caller_plane_not_horizontal")
            return result
    started = time.monotonic()
    from . import embodiment_of
    frame_id_default = (profile if profile is not None else embodiment_of(None)).frame_id
    try:
        try:
            camera = tools.camera(full, full["observation_id"], full["camera_id"])
        except KeyError as exc:
            raise ValueError("unknown_observation") from exc
        raw_px = full.get("pixel")
        if (
            not isinstance(raw_px, (list, tuple))
            or len(raw_px) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in raw_px)
        ):
            raise ValueError("invalid_pixel")
        pixel = np.asarray(raw_px, dtype=np.float64).reshape(2)
        if not (np.isfinite(pixel).all() and 0 <= pixel[0] < camera["width"] and 0 <= pixel[1] < camera["height"]):
            raise ValueError("pixel_out_of_bounds")
    except ValueError as input_err:
        reason = str(input_err)[:200]
        is_invalid = reason.startswith("invalid_") or reason == "pixel_out_of_bounds"
        return {
            "schema_version": "0.1",
            **runtime.identity,
            "world_kind": "internal",
            "query_type": "pixel_plane",
            "observation_ids": [full.get("observation_id")],
            "status": "invalid_input" if is_invalid else "unknown",
            "reason": reason,
            "frame_id": frame_id_default,
            "unit": "m",
            "value": None,
            "duration_s": time.monotonic() - started,
        }
    estimate, error = None, None
    try:
        _, image = runtime.scene.resolve_artifact(camera["frame"]["rgb_ref"])
        from .object_table import effective_support_z_m
        from .rectify import rectified_cached
        rectified = rectified_cached(camera, image)   # pinhole service: distorted cameras are sent undistorted
        support_z = effective_support_z_m(features, camera, profile)
        answer, item = query_service(features.vggt_endpoint, rectified.image_bytes, camera["K"], camera["T"],
                                     support_z, rectified.to_rectified([pixel])[0], features.vggt_timeout_s)
        estimate = (answer, item)
    except (ServiceError, ValueError) as caught:
        error = str(caught)[:200]
    latency = time.monotonic() - started
    if estimate is None:
        if "plane_origin_m" in full and "plane_normal" in full:
            result = tools.execute(full)
            if isinstance(result.get("value"), dict):
                result["value"]["plane_estimate"] = dict(source=SOURCE, status="unavailable", error=error,
                                                         note="caller plane used unchanged")
            else:
                result["plane_estimate"] = dict(source=SOURCE, status="unavailable", error=error)
            return result
        return {
            "schema_version": "0.1",
            **runtime.identity,
            "world_kind": "internal",
            "query_type": "pixel_plane",
            "observation_ids": [full.get("observation_id")],
            "status": "unknown",
            "reason": "vggt_estimate_unavailable:" + str(error),
            "frame_id": camera["frame"]["frame_id"],
            "unit": "m",
            "value": None,
            "plane_estimate": dict(source=SOURCE, status="unavailable", error=error),
            "duration_s": latency,
        }
    answer, item = estimate
    center, top, support = float(item["center_z_m"]), float(item["top_z_m"]), float(item["support_z_m"])
    caller_origin = full.get("plane_origin_m")
    xy = [float(caller_origin[0]), float(caller_origin[1])] if isinstance(caller_origin, (list, tuple)) and len(caller_origin) == 3 else [0.0, 0.0]
    unmasked_default = (
        default
        and caller_origin is not None
        and explicit_source != "vggt"
        and (
            item.get("method") != "sam_point_mask"
            or float(item.get("height_m") or 0.0) < 0.004
            or "floating" in (item.get("flags") or [])
        )
    )
    if unmasked_default:
        # 未命中物体掩码且调用方显式提供了水平平面时，保留调用方平面为主平面并诚实回报
        plane = dict(full, plane_origin_m=deepcopy(caller_origin), plane_normal=[0.0, 0.0, 1.0],
                     plane_origin_alt_m=[*xy, center])
        used_plane = "caller_plane_origin_m"
        alternate = "estimated_centre_plane"
    else:
        plane = dict(full, plane_origin_m=[*xy, center], plane_normal=[0.0, 0.0, 1.0])
        used_plane = "estimated_centre_height"
        if default and caller_origin is not None:  # the model's own plane is the hypothesis to compare against
            plane["plane_origin_alt_m"], alternate = deepcopy(caller_origin), "caller_plane_origin_m"
        elif "plane_origin_alt_m" in full:
            alternate = "caller_plane_origin_alt_m"
        elif caller_origin is not None:
            plane["plane_origin_alt_m"], alternate = deepcopy(caller_origin), "caller_plane_origin_m"
        else:
            plane["plane_origin_alt_m"], alternate = [*xy, top], "estimated_top_plane"
    result = tools.execute(plane)
    level, reasons = quality(answer, item)
    summary = dict(source=SOURCE, status="ok", measured=False, plane=used_plane,
                   mode="default" if default else "optional",
                   alternate_plane=alternate, center_z_m=center, top_z_m=top, support_z_m=support,
                   object_height_m=float(item["height_m"]), method=item.get("method"), quality=level,
                   quality_reasons=reasons, n_points=item.get("n_points"), mask_px=item.get("mask_px"),
                   depth_scale=answer.get("scale"), support_fit=answer.get("support_fit"),
                   latency_s=latency, service_version=answer.get("version"))
    if isinstance(result.get("value"), dict):
        result["value"]["plane_estimate"] = summary
        if unmasked_default:
            assumption = (
                "plane_origin_m uses the caller-supplied plane; alternate_point_m uses an RGB-based VGGT-Omega "
                "estimate (centre height of the object under the pixel, scaled to the configured table plane; "
                "not measured depth)."
            )
        else:
            assumption = (
                "plane_origin_m height is an RGB-based VGGT-Omega estimate (centre height of the object under the "
                "pixel, scaled to the configured table plane); it is an estimate, not measured depth."
            )
        result["assumptions"] = [assumption, *result.get("assumptions", [])]
    else:
        result["plane_estimate"] = summary
    return result
