"""Offline SO101 diagnostics for the existing table-anchored VGGT /points response.

The service's metric point cloud depends on supplied camera calibration and support height. It must
not certify those inputs. Inverting its recorded similarity recovers only *retained camera-depth
samples*, not an independent raw reconstruction. An explicit external length reference can provide a
separate conditional scale; held-out lengths remain separate from fitted lengths. No network or robot
calls, calibration install, or automatic contact acceptance occur in this module.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np

from .config import evidence_settings

DEFAULTS = evidence_settings("vggt_evidence")

SCHEMA = "ni_rua.so101.vggt_evidence.v1"
REVIEW_SCHEMA = "ni_rua.so101.vggt_evidence_review.v1"


def _finite(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    return array


def decode_points_response(request, response):
    """Validate and invert one recorded existing-service /points exchange, without calling it.

    request: actual JSON body containing image_b64, pinhole K (px), T_world_camera (rigid 4x4, m in
        so101_base), support_z_m and optional voxel/filter settings; image must already be rectified.
    response: unmodified ``status=ok, version=scene3d-points-v1`` response including npz_b64, scale,
        image_sha256 and echoed support/image/voxel metadata. Do not use client-remapped NPZ UV arrays.
    Returns arrays xyz_m (service world frame), camera_depth_points (unvalidated model-depth units),
        uv_center_px (reported integer UV + .5, as the service's backprojection), confidence_rank
        (0..1 percentile, not probability), and metadata. Camera points equal (xyz-C) @ R / scale.
    Input/response image, support, dimensions, voxel and NPZ metadata are checked. K/T are not echoed
        by this service, so their association remains the recorded caller's assertion, not a service
        attestation. Existing voxel/range/support filtering cannot be undone. Raises ValueError for
        missing/invalid facts; no guessed scale, coordinate conversion or fallback is permitted.
    """
    if not isinstance(request, dict) or not isinstance(response, dict):
        raise ValueError("recorded request/response must be objects")
    if response.get("status") != "ok" or response.get("version") != "scene3d-points-v1":
        raise ValueError("requires a successful scene3d-points-v1 response")
    raw = base64.b64decode(request["image_b64"], validate=True)
    digest = hashlib.sha256(raw).hexdigest()
    if response.get("image_sha256") != digest:
        raise ValueError("VGGT response does not bind the submitted image bytes")
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("submitted service image cannot be decoded")
    height, width = image.shape[:2]
    if response.get("image_size") != [width, height]:
        raise ValueError("response image size differs from submitted service image")
    K = _finite(request["K"], (3, 3), "K")
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
        raise ValueError("positive pinhole focal lengths and homogeneous K required")
    T = _finite(request["T_world_camera"], (4, 4), "T_world_camera")
    R, C = T[:3, :3], T[:3, 3]
    if not np.allclose(T[3], [0, 0, 0, 1]) or not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or np.linalg.det(R) <= 0:
        raise ValueError("T_world_camera must be rigid")
    support, scale = float(request["support_z_m"]), float(response["scale"])
    if not np.isfinite([support, scale]).all() or scale <= 0 or response.get("support_z_m") != support:
        raise ValueError("support echo and finite positive recorded scale required")
    raw_npz = base64.b64decode(response["npz_b64"], validate=True)
    with np.load(io.BytesIO(raw_npz), allow_pickle=False) as archive:
        required = {"xyz", "uv", "conf", "count", "robot", "support_z", "voxel_m", "image_size"}
        if not required <= set(archive.files):
            raise ValueError("incomplete original /points NPZ")
        arrays = {name: np.array(archive[name], copy=True) for name in required}
    n = response.get("n_points")
    if type(n) is not int or n < 0:
        raise ValueError("response n_points must be a nonnegative integer")
    xyz = _finite(arrays["xyz"], (n, 3), "xyz")
    uv = arrays["uv"]
    if uv.shape != (n, 2) or not np.issubdtype(uv.dtype, np.integer) or np.any(uv < 0) or np.any(uv >= [width, height]):
        raise ValueError("NPZ UV must be integer indices inside service image")
    if len(np.unique(uv, axis=0)) != n:
        raise ValueError("original /points response cannot contain duplicate source pixels")
    if arrays["conf"].shape != (n,) or arrays["conf"].dtype != np.uint8:
        raise ValueError("NPZ confidence must be uint8 image-wide rank")
    if arrays["robot"].shape != (n,) or arrays["robot"].dtype != np.bool_:
        raise ValueError("NPZ robot flags must be boolean")
    if arrays["count"].shape != (n,) or not np.issubdtype(arrays["count"].dtype, np.integer) or np.any(arrays["count"] < 1):
        raise ValueError("NPZ voxel sample counts must be positive integers")
    voxel = float(response["voxel_m"])
    requested_voxel = float(request.get("voxel_m", request.get("params", {}).get("voxel", .01)))
    if not .002 <= voxel <= .05 or voxel != requested_voxel or arrays["voxel_m"].shape != () or float(arrays["voxel_m"]) != voxel:
        raise ValueError("voxel metadata mismatch or unsupported voxel size")
    if arrays["support_z"].shape != () or float(arrays["support_z"]) != support or not np.array_equal(arrays["image_size"], [width, height]):
        raise ValueError("NPZ support/image metadata mismatch")
    camera = (xyz - C) @ R / scale
    if not np.isfinite(camera).all() or np.any(camera[:, 2] <= 0):
        raise ValueError("retained camera-depth samples must be finite and in front of the camera")
    projected = np.column_stack([K[0, 0] * camera[:, 0] / camera[:, 2] + K[0, 2],
                                 K[1, 1] * camera[:, 1] / camera[:, 2] + K[1, 2]]) if n else np.zeros((0, 2))
    # This is a reported correspondence consistency diagnostic, not an invented acceptance gate.
    ray_error = np.linalg.norm(projected - (uv.astype(float) + .5), axis=1)
    return dict(xyz_m=xyz, camera_depth_points=camera, uv=uv, uv_center_px=uv.astype(float) + .5,
                confidence_rank=arrays["conf"].astype(float) / 255, robot=arrays["robot"],
                image_size=[width, height], K=K, T=T, scale=scale, support_z_m=support,
                voxel_m=voxel, image_sha256=digest, npz_sha256=hashlib.sha256(raw_npz).hexdigest(),
                service_ray_error_max_px=float(ray_error.max()) if n else None,
                support_fit=response.get("support_fit"),
                request_camera_binding="caller_recorded_not_echoed_by_service")


def _plane(points, rank_rtol):
    if len(points) < 3:
        return {"status": "unknown", "reason": "fewer_than_three_selected_points"}
    center = points.mean(axis=0)
    _, singular, vectors = np.linalg.svd(points - center, full_matrices=False)
    if singular[0] <= 0 or singular[1] <= rank_rtol * singular[0]:
        return {"status": "unknown", "reason": "selected_points_collinear", "singular_values_depth_units": singular.tolist()}
    normal = vectors[-1]
    if normal @ center < 0:
        normal = -normal
    residual = (points - center) @ normal
    rms = float(np.sqrt(np.mean(residual ** 2)))
    return {"status": "conditional_shape_fit", "samples": len(points), "normal_camera": normal.tolist(),
            "centroid_camera_depth_units": center.tolist(), "distance_camera_depth_units": float(normal @ center),
            "residual_rms_depth_units": rms, "residual_max_depth_units": float(np.max(np.abs(residual))),
            "normalized_residual_rms": rms / float(np.median(points[:, 2])),
            "singular_values_depth_units": singular.tolist()}


def _edges(landmarks, role, scale=None):
    selected = [row for row in landmarks if row["status"] == "matched" and row["role"] == role]
    edges = []
    for i, a in enumerate(selected):
        for b in selected[i + 1:]:
            model = float(np.linalg.norm(np.asarray(a["camera_depth_point"]) - b["camera_depth_point"]))
            length = float(np.linalg.norm(np.asarray(a["reference_point_m"]) - b["reference_point_m"]))
            if model == 0 or length == 0:
                continue
            edge = {"ids": [a["id"], b["id"]], "model_length_depth_units": model, "reference_length_m": length}
            if scale is not None:
                edge.update(estimated_length_m=scale * model, error_m=scale * model - length)
            edges.append(edge)
    return edges


def analyze_points(request, response, review, *, min_confidence_rank=DEFAULTS["min_confidence_rank"],
                   max_landmark_pixel_distance_px=DEFAULTS["max_landmark_pixel_distance_px"], rank_rtol=DEFAULTS["rank_rtol"]):
    """Analyze existing /points data with an explicit surface ROI and optional external scale.

    request/response: original saved exchange described by decode_points_response; never a raw VGGT
        reconstruction. review: schema=ni_rua.so101.vggt_evidence_review.v1, image_sha256 matching the
        service image, pixel_coordinates='service_image', and plane_roi_px polygon (>=3 [u,v] vertices,
        original *rectified service image* px). User identifies table-only pixels; no automatic table
        semantics. Optional metric_landmarks have {id,pixel_px,reference_point_m,role:'fit'|'holdout'};
        reference_point_m is a measured/known point in one common external metric frame, e.g. board
        corner grid coordinates. Provide reference_note and reference_uncertainty_m (nonnegative m
        or null) when using landmarks. Geometry must not be generated from active FK/table/hand-eye.
    min_confidence_rank: dimensionless percentile cutoff [0,1], default .1; not model probability.
    max_landmark_pixel_distance_px: positive nearest-retained-pixel radius, default 3 px; actual offset
        and selected pixel are reported. No extrapolation/interpolation or hidden correspondence.
    rank_rtol: relative numerical rank cutoff (0,1), default 1e-6 for selected geometry, not accuracy.

    Fits plane SHAPE in retained camera-depth units. A scalar camera metric scale is fit only from
        >=3 matched, noncollinear fit landmarks via pairwise lengths; all held-out-only edge errors
        are reported separately and never used for scale. Pairwise edges share correlated endpoints.
        Base-frame plane outputs still depend on the request's extrinsics and are not independent
        validation of those extrinsics. Returns diagnostics only; no accepted calibration/contact.
    Example: analyze_points(request, response, {'schema': REVIEW_SCHEMA, 'image_sha256': digest,
        'pixel_coordinates':'service_image', 'plane_roi_px': [[10,10],[300,10],[300,200],[10,200]]}).
    """
    if not np.isfinite([min_confidence_rank, max_landmark_pixel_distance_px, rank_rtol]).all() or not 0 <= min_confidence_rank <= 1 or max_landmark_pixel_distance_px <= 0 or not 0 < rank_rtol < 1:
        raise ValueError("invalid confidence rank, pixel match radius or numerical rank cutoff")
    cloud = decode_points_response(request, response)
    if not isinstance(review, dict) or review.get("schema") != REVIEW_SCHEMA or review.get("image_sha256") != cloud["image_sha256"] or review.get("pixel_coordinates") != "service_image":
        raise ValueError("review must bind this rectified service image and coordinate convention")
    polygon = np.asarray(review["plane_roi_px"], float)
    if polygon.ndim != 2 or polygon.shape[1:] != (2,) or len(polygon) < 3 or not np.isfinite(polygon).all() or np.any(polygon < 0) or np.any(polygon > cloud["image_size"]):
        raise ValueError("plane_roi_px requires >=3 finite polygon vertices within image extent")
    if abs(cv2.contourArea(polygon.astype(np.float32))) == 0:
        raise ValueError("plane ROI has zero area")
    eligible = (cloud["confidence_rank"] >= min_confidence_rank) & ~cloud["robot"]
    inside = np.array([cv2.pointPolygonTest(polygon.astype(np.float32), tuple(p), False) >= 0
                       for p in cloud["uv_center_px"]], bool)
    plane = _plane(cloud["camera_depth_points"][eligible & inside], rank_rtol)
    if plane["status"] == "conditional_shape_fit":
        base_normal = cloud["T"][:3, :3] @ np.asarray(plane["normal_camera"])
        if base_normal[2] < 0:
            base_normal = -base_normal
        plane["normal_base_from_request_rotation"] = base_normal.tolist()
        plane["angle_to_request_base_z_deg"] = float(np.degrees(np.arccos(np.clip(base_normal[2], -1., 1.))))
    result = {"schema": SCHEMA, "status": "diagnostic", "source": "existing_support_anchored_vggt_points",
              "independence": {"absolute_height_from_service": False, "input_support_height_removed_algebraically": True,
                  "full_raw_reconstruction": False, "selection_depends_on_support_and_extrinsics": True,
                  "camera_intrinsics_independent": None, "camera_extrinsics_independent": None,
                  "metric_scale_source": "none", "request_camera_binding": cloud["request_camera_binding"]},
              "physical_acceptance": None, "contact_evidence": "unknown", "use_in_joint_solver": "diagnostic_only",
              "submitted_support_z_m": cloud["support_z_m"], "service_scale": cloud["scale"],
              "service_support_fit": cloud["support_fit"], "retained_points": len(cloud["uv"]),
              "selected_plane_points": int((eligible & inside).sum()), "service_ray_error_max_px": cloud["service_ray_error_max_px"],
              "plane_camera_depth_units": plane, "external_metric_scale": None, "metric_plane": None,
              "external_metric_scale_status": "not_requested", "external_metric_scale_reason": None,
              "pixel_coordinates": "service_image", "plane_roi_px": polygon.tolist(),
              "landmarks": [], "training_edges": [], "heldout_edges": [],
              "settings": dict(min_confidence_rank=min_confidence_rank,
                  max_landmark_pixel_distance_px=max_landmark_pixel_distance_px, rank_rtol=rank_rtol)}
    landmarks = review.get("metric_landmarks", [])
    if not isinstance(landmarks, list):
        raise ValueError("metric_landmarks must be a list")
    if landmarks:
        result.update(external_metric_scale_status="unknown", external_metric_scale_reason="insufficient_matched_noncollinear_fit_landmarks")
        uncertainty = review.get("reference_uncertainty_m")
        if not isinstance(review.get("reference_note"), str) or not review["reference_note"].strip() or "reference_uncertainty_m" not in review:
            raise ValueError("external landmarks require a reference_note and explicit uncertainty (m or null)")
        if uncertainty is not None and (not np.isfinite(uncertainty) or uncertainty < 0):
            raise ValueError("reference uncertainty must be nonnegative m or null")
        result.update(reference_note=review["reference_note"], reference_uncertainty_m=uncertainty)
    seen, seen_references, used_pixels = set(), set(), set()
    candidates = np.flatnonzero(eligible)
    for landmark in landmarks:
        if not isinstance(landmark, dict) or set(landmark) != {"id", "pixel_px", "reference_point_m", "role"}:
            raise ValueError("landmark requires exactly id/pixel_px/reference_point_m/role")
        identity, role = landmark["id"], landmark["role"]
        if not isinstance(identity, str) or not identity or identity in seen or role not in ("fit", "holdout"):
            raise ValueError("landmark IDs must be unique and roles fit/holdout")
        seen.add(identity)
        pixel = _finite(landmark["pixel_px"], (2,), "landmark pixel")
        reference = _finite(landmark["reference_point_m"], (3,), "reference metric point")
        if tuple(reference) in seen_references:
            raise ValueError("reference landmarks must be distinct physical points across fit/holdout")
        seen_references.add(tuple(reference))
        if np.any(pixel < 0) or np.any(pixel >= cloud["image_size"]):
            raise ValueError("landmark pixel outside service image")
        row = {"id": identity, "role": role, "pixel_px": pixel.tolist(), "reference_point_m": reference.tolist(),
               "status": "unknown", "reason": "no_retained_pixel_within_radius"}
        if len(candidates):
            distances = np.linalg.norm(cloud["uv_center_px"][candidates] - pixel, axis=1)
            at = int(candidates[np.argmin(distances)])
            distance = float(distances.min())
            row["nearest_pixel_distance_px"] = distance
            if distance <= max_landmark_pixel_distance_px:
                if at in used_pixels:
                    raise ValueError("landmarks cannot reuse a retained source pixel, including across fit/holdout")
                used_pixels.add(at)
                row.update(status="matched", reason=None, sampled_uv_center_px=cloud["uv_center_px"][at].tolist(),
                           camera_depth_point=cloud["camera_depth_points"][at].tolist(),
                           confidence_rank=float(cloud["confidence_rank"][at]))
        result["landmarks"].append(row)
    fit = [row for row in result["landmarks"] if row["status"] == "matched" and row["role"] == "fit"]
    raw_edges = _edges(result["landmarks"], "fit")
    if len(fit) >= 3 and raw_edges:
        ref = np.asarray([row["reference_point_m"] for row in fit])
        sv = np.linalg.svd(ref - ref.mean(axis=0), compute_uv=False)
        modeled = np.asarray([row["camera_depth_point"] for row in fit])
        mv = np.linalg.svd(modeled - modeled.mean(axis=0), compute_uv=False)
        if sv[0] > 0 and sv[1] > rank_rtol * sv[0] and mv[0] > 0 and mv[1] > rank_rtol * mv[0]:
            a = np.array([e["model_length_depth_units"] for e in raw_edges])
            b = np.array([e["reference_length_m"] for e in raw_edges])
            metric_scale = float(a @ b / (a @ a))
            if not np.isfinite(metric_scale) or metric_scale <= 0:
                raise ValueError("external length fit produced an invalid metric scale")
            result["external_metric_scale"] = metric_scale
            result.update(external_metric_scale_status="estimated", external_metric_scale_reason=None)
            result["independence"]["metric_scale_source"] = "caller_external_reference_lengths"
            result["training_edges"] = _edges(result["landmarks"], "fit", metric_scale)
            result["heldout_edges"] = _edges(result["landmarks"], "holdout", metric_scale)
            if plane["status"] == "conditional_shape_fit":
                normal = np.asarray(plane["normal_camera"])
                center_m = metric_scale * np.asarray(plane["centroid_camera_depth_units"])
                R, C = cloud["T"][:3, :3], cloud["T"][:3, 3]
                base_normal, base_point = R @ normal, C + R @ center_m
                if base_normal[2] < 0:
                    base_normal = -base_normal
                base_height = float(base_normal @ base_point / base_normal[2]) if abs(base_normal[2]) > 1e-12 else None
                result["metric_plane"] = {"status": "conditional_on_reference_intrinsics_and_extrinsics",
                    "normal_camera": normal.tolist(), "distance_camera_m": metric_scale * plane["distance_camera_depth_units"],
                    "residual_rms_m": metric_scale * plane["residual_rms_depth_units"],
                    "normal_base": base_normal.tolist(), "height_at_base_origin_m": base_height,
                    "external_validation": None}
    result["edge_residual_summary"] = {}
    for name in ("training_edges", "heldout_edges"):
        errors = np.asarray([row["error_m"] for row in result[name]])
        result["edge_residual_summary"][name] = {"count": len(errors),
            "rms_m": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None,
            "max_abs_m": float(np.max(np.abs(errors))) if len(errors) else None}
    return result


def compare_support_hypotheses(request_a, response_a, request_b, response_b):
    """Compare same-image /points calls differing only in assumed support height (m); no service call.

    Each request/response is as decode_points_response. K, T and all geometric filtering settings must
    match. Timeout may differ. Returns scale response and retained common-UV camera-depth differences;
    differing selected pixels stay in the denominator. This is sensitivity, never a new table estimate.
    """
    a, b = decode_points_response(request_a, response_a), decode_points_response(request_b, response_b)
    qa = {k: v for k, v in request_a.items() if k not in ("support_z_m", "timeout_s")}
    qb = {k: v for k, v in request_b.items() if k not in ("support_z_m", "timeout_s")}
    if qa != qb:
        raise ValueError("support sensitivity requires identical image/K/T/filter inputs except support_z_m")
    ai = {tuple(pixel): i for i, pixel in enumerate(a["uv"])}
    bi = {tuple(pixel): i for i, pixel in enumerate(b["uv"])}
    common = sorted(set(ai) & set(bi))
    delta = np.array([a["camera_depth_points"][ai[p]] - b["camera_depth_points"][bi[p]] for p in common])
    denominator = a["support_z_m"] - a["T"][2, 3]
    return {"status": "sensitivity_only", "support_delta_m": b["support_z_m"] - a["support_z_m"],
            "retained_a": len(ai), "retained_b": len(bi), "common_uv_count": len(common),
            "only_a_count": len(set(ai) - set(bi)), "only_b_count": len(set(bi) - set(ai)),
            "actual_service_scale_ratio": b["scale"] / a["scale"],
            "support_formula_scale_ratio": (b["support_z_m"] - b["T"][2, 3]) / denominator if denominator else None,
            "common_camera_depth_rms_difference": float(np.sqrt(np.mean(np.sum(delta ** 2, axis=1)))) if common else None,
            "common_camera_depth_max_difference": float(np.max(np.linalg.norm(delta, axis=1))) if common else None,
            "table_height_independently_validated": False}


def main(argv=None):
    """Offline CLI consumes saved service exchanges/review and creates a new diagnostics JSON file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="recorded /points JSON request with submitted image bytes")
    parser.add_argument("--response", required=True, help="unmodified /points JSON response with original npz_b64")
    parser.add_argument("--review", required=True, help="bound service-image ROI and optional metric landmark review JSON")
    parser.add_argument("--output", required=True, help="new diagnostics JSON; never overwrite existing artifacts")
    parser.add_argument("--compare-request", help="optional same-image support-perturbed request JSON")
    parser.add_argument("--compare-response", help="paired support-perturbed response JSON")
    parser.add_argument("--min-confidence-rank", type=float, default=argparse.SUPPRESS, help="image-relative confidence percentile cutoff [0,1]")
    parser.add_argument("--max-landmark-pixel-distance-px", type=float, default=argparse.SUPPRESS, help="nearest retained pixel radius, px (default 3)")
    parser.add_argument("--rank-rtol", type=float, default=argparse.SUPPRESS, help="relative numerical rank cutoff (default 1e-6)")
    parser.add_argument("--config", help="shared offline evidence JSON; explicit CLI settings override the file; omitted uses packaged defaults")
    args = parser.parse_args(argv)
    settings = evidence_settings("vggt_evidence", args.config,
        {key: getattr(args, key) for key in DEFAULTS if hasattr(args, key)})
    if bool(args.compare_request) != bool(args.compare_response):
        parser.error("compare request and response must be supplied together")
    paths = {name: Path(getattr(args, name)) for name in ("request", "response", "review")}
    if args.compare_request:
        paths.update(compare_request=Path(args.compare_request), compare_response=Path(args.compare_response))
    source = {name: path.read_bytes() for name, path in paths.items()}
    documents = {name: json.loads(data) for name, data in source.items()}
    report = analyze_points(documents["request"], documents["response"], documents["review"],
        **settings)
    if args.compare_request:
        report["support_sensitivity"] = compare_support_hypotheses(documents["request"], documents["response"],
            documents["compare_request"], documents["compare_response"])
    report["operator_input_binding"] = {name: {"path": str(paths[name].resolve()), "sha256": hashlib.sha256(data).hexdigest()}
                                        for name, data in source.items()}
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with Path(args.output).open("x") as stream:
        stream.write(serialized)
    print(json.dumps({"status": report["status"], "plane_status": report["plane_camera_depth_units"]["status"],
                      "external_metric_scale": report["external_metric_scale"], "physical_acceptance": None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
