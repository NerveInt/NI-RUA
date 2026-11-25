"""Offline SO101 video evidence: SAM mask association plus explicitly seeded optical-flow features.

SAM masks are object regions, not material keypoints. This module uses separate OpenCV LK tracking,
checks mask membership and forward/backward consistency, and never reacquires a lost feature under
the same identity. Host camera times are not exposure times. Outputs aid operator review; they
never certify contact, modify joint offsets, or automatically admit samples to a calibration fit.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import uuid

import cv2
import numpy as np

from .config import evidence_settings

DEFAULTS = evidence_settings("video_evidence")

SCHEMA = "ni_rua.so101.video_evidence.v1"


def _inside(root, name):
    """Resolve an existing relative artifact beneath root; never read an escaped archive path."""
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("artifact path escapes archive or is missing")
    return path


def _snapshot(path):
    """Read one JSON byte snapshot at path and its operator-only byte binding."""
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def _image(path, digest, flags=cv2.IMREAD_COLOR):
    """Decode bound original pixels; flags selects BGR (default) or grayscale mask pixels."""
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("artifact bytes differ from recorded binding")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), flags)
    if image is None:
        raise ValueError("recorded image cannot be decoded")
    return image


def _joint_samples(manifest, touch_report):
    """Measured q/jaw at host monotonic s; never use goals or infer missing trace origins."""
    samples = []
    for camera in manifest["cameras"].values():
        samples.extend(frame["joint_sample"] for frame in camera["frames"] if frame.get("joint_sample"))
    if touch_report is not None:
        if touch_report.get("joint_mapping") != manifest.get("joint_mapping"):
            raise ValueError("touch/video joint mappings differ")
        for trace in touch_report.get("descent_traces", []):
            if trace.get("clock_domain") != "adapter_monotonic":
                raise ValueError("touch trace clock domain differs from video")
            origin = trace.get("started_monotonic_s")
            if origin is None:
                continue  # Old relative traces cannot be synchronized after the fact.
            if trace.get("joint_mapping") != manifest.get("joint_mapping"):
                raise ValueError("trace/video joint mappings differ")
            for row in [*trace["reads"], *([trace["terminal_hold"]] if trace.get("terminal_hold") else [])]:
                samples.append(dict(t=float(origin) + row["elapsed_s"], q_rad=row["q_measured_rad"],
                                    gripper_rad=row["gripper_rad"]))
    by_time, conflicts = {}, set()
    for sample in samples:
        t, q, jaw = float(sample["t"]), np.asarray(sample["q_rad"], float), float(sample["gripper_rad"])
        if q.shape != (5,) or not np.isfinite([t, jaw]).all() or not np.isfinite(q).all():
            raise ValueError("joint samples must contain finite time, five measured rad and jaw rad")
        row = dict(t=t, q_rad=q.tolist(), gripper_rad=jaw)
        if t in by_time and by_time[t] != row:
            conflicts.add(t)
        by_time[t] = row
    return [by_time[t] for t in sorted(by_time) if t not in conflicts], sorted(conflicts)


def align_joints(samples, time_s, max_gap_s, conflicts=()):
    """Interpolate measured joints at a HOST timestamp, with brackets and no extrapolation.

    samples: time-sorted dicts {t: monotonic s, q_rad: five rad, gripper_rad: rad}.
    time_s: camera host-read start in the same clock. max_gap_s: positive permitted bracket span (s).
    conflicts: optional inconsistent sample timestamps (s); brackets spanning one are rejected.
    Returns q_estimate_rad/jaw_estimate_rad or unknown; exposure alignment always remains false.
    Example: align_joints(samples, 12.5, .2). An interpolated q is never labelled a measured exposure q.
    """
    out = dict(status="unknown", exposure_aligned=False, query_host_monotonic_s=float(time_s),
               q_estimate_rad=None, jaw_estimate_rad=None, bracket_monotonic_s=None)
    if not math.isfinite(time_s) or not math.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("finite host time and positive max_gap_s required")
    if time_s in conflicts:
        return dict(out, reason="conflicting_joint_samples")
    times = [r["t"] for r in samples]
    i = int(np.searchsorted(times, time_s))
    if i < len(samples) and times[i] == time_s:
        return dict(out, status="host_sample", q_estimate_rad=samples[i]["q_rad"],
                    jaw_estimate_rad=samples[i]["gripper_rad"], bracket_monotonic_s=[time_s, time_s])
    if i == 0 or i == len(samples):
        return dict(out, reason="no_joint_bracket")
    a, b = samples[i-1], samples[i]
    if b["t"] - a["t"] > max_gap_s:
        return dict(out, reason="joint_bracket_too_wide", bracket_monotonic_s=[a["t"], b["t"]])
    if any(a["t"] <= t <= b["t"] for t in conflicts):
        return dict(out, reason="conflicting_joint_samples")
    delta = np.asarray(b["q_rad"]) - np.asarray(a["q_rad"])
    if np.any(np.abs(delta) >= np.pi) or abs(b["gripper_rad"] - a["gripper_rad"]) >= np.pi:
        return dict(out, reason="angular_branch_discontinuity", joint_delta_rad=delta.tolist())
    ratio = (time_s - a["t"]) / (b["t"] - a["t"])
    return dict(out, status="host_interpolated", joint_delta_rad=delta.tolist(), bracket_monotonic_s=[a["t"], b["t"]],
                q_estimate_rad=(np.asarray(a["q_rad"]) * (1-ratio) + np.asarray(b["q_rad"]) * ratio).tolist(),
                jaw_estimate_rad=a["gripper_rad"] * (1-ratio) + b["gripper_rad"] * ratio)


def _features(features, frame_count, image_size):
    seen = set()
    for f in features:
        required = {"feature_id", "frame_index", "object_id", "point_px"}
        if not isinstance(f, dict) or not required <= set(f) or set(f) - required - {"link", "point_in_link_m"}:
            raise ValueError("feature requires id/frame/object/point_px and optional link+point_in_link_m")
        if not isinstance(f["feature_id"], str) or not f["feature_id"] or f["feature_id"] in seen:
            raise ValueError("feature IDs must be unique nonempty strings")
        seen.add(f["feature_id"])
        if type(f["frame_index"]) is not int or not 0 <= f["frame_index"] < frame_count:
            raise ValueError("feature frame_index out of range")
        if type(f["object_id"]) is not int or f["object_id"] < 0:
            raise ValueError("feature object_id must be a SAM nonnegative integer mask ID")
        p = np.asarray(f["point_px"], float)
        if p.shape != (2,) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p >= image_size):
            raise ValueError("feature seed must be an original-image pixel inside the frame")
        if ("link" in f) != ("point_in_link_m" in f):
            raise ValueError("feature projection needs both link and point_in_link_m")
        if "link" in f:
            point = np.asarray(f["point_in_link_m"], float)
            if not isinstance(f["link"], str) or point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError("identified link and finite material point in metres required")
    return features


def _projection(snapshot, camera_id, alignment, feature, observed, model, image_size):
    """Conditional raw pixel residual using the captured calibration; no fit or physical verdict."""
    if "link" not in feature or alignment["q_estimate_rad"] is None:
        return None
    from ...embodiments.so101 import profile as kin
    spec = (snapshot or {}).get("cameras", {}).get(camera_id)
    if not spec:
        return None
    if spec.get("image_size") != image_size:
        return dict(status="unknown", reason="camera_calibration_image_size_unverified")
    poses = kin.fk_all(model, alignment["q_estimate_rad"], gripper_rad=alignment["jaw_estimate_rad"])
    if feature["link"] not in poses:
        raise ValueError("identified material point link is not in the SO101 model")
    if "T_base_camera" in spec:
        T = np.asarray(spec["T_base_camera"], float)
    else:
        T = poses[kin.WRIST_CAMERA_PARENT_LINK] @ np.asarray(spec["T_link_camera"], float)
    K = np.asarray(spec["K"], float)
    if T.shape != (4, 4) or K.shape != (3, 3) or not np.isfinite(T).all() or not np.isfinite(K).all():
        raise ValueError("finite camera transform and intrinsics required")
    if not np.allclose(T[3], [0, 0, 0, 1]) or not np.allclose(T[:3,:3].T @ T[:3,:3], np.eye(3), atol=1e-6) or np.linalg.det(T[:3,:3]) <= 0:
        raise ValueError("camera transform must be rigid")
    distortion = np.asarray(spec.get("distortion", []), float)
    if distortion.ndim != 1 or distortion.size not in (0, 4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
        raise ValueError("finite OpenCV distortion vector required")
    if K[0,0] <= 0 or K[1,1] <= 0 or not np.allclose(K[2], [0,0,1]):
        raise ValueError("positive focal lengths and homogeneous camera intrinsics required")
    world = poses[feature["link"]] @ np.r_[feature["point_in_link_m"], 1.]
    point = np.linalg.inv(T) @ world
    if point[2] <= 0:
        return dict(status="unknown", reason="point_behind_camera")
    uv, _ = cv2.projectPoints(point[:3], np.zeros(3), np.zeros(3), np.asarray(spec["K"], float),
                              distortion)
    predicted = uv.reshape(2)
    if not np.isfinite(predicted).all():
        raise ValueError("camera projection produced nonfinite pixels")
    return dict(status="conditional_pixel_residual", predicted_px=predicted.tolist(),
                observed_minus_predicted_px=(np.asarray(observed)-predicted).tolist(),
                error_px=float(np.linalg.norm(np.asarray(observed)-predicted)),
                physical_acceptance=None, timing_uncertainty="exposure_latency_unknown")


def analyze_video(video_manifest, camera_id, tracking_path, features, output_dir, *, touch_report_path=None,
                  max_frame_gap_s=DEFAULTS["max_frame_gap_s"], max_joint_gap_s=DEFAULTS["max_joint_gap_s"],
                  max_forward_backward_error_px=DEFAULTS["max_forward_backward_error_px"],
                  window_size_px=DEFAULTS["window_size_px"], pyramid_levels=DEFAULTS["pyramid_levels"],
                  min_eigenvalue=DEFAULTS["min_eigenvalue"]):
    """Join recorded video, SAM masks, seeded LK features and measured joints into operator evidence.

    video_manifest: calibration_video manifest.json path (immutable PNG frames and host times).
    camera_id: recorded camera name, e.g. fixed; no inferred camera selection.
    tracking_path: SAM3.1 tracking.json bound to the same original frame bytes; no current-image calls.
    features: list of {feature_id, frame_index, object_id, point_px:[u,v]} with explicit original-image
        pixel seeds. Optional link+point_in_link_m[3] supplies a caller-identified material feature for
        conditional projection against frozen calibration. Default CLI list is empty (mask review only).
    output_dir: required new directory, never overwrite inputs/results; contains evidence.json and
        standalone review.html with rendered overlays. touch_report_path: optional matching S1/S6 JSON
        with video.manifest_sha256; adds measured descent samples, never goals.
    max_frame_gap_s/max_joint_gap_s: positive allowed frame interval / joint bracket span, defaults
        .25 s; gaps stay unknown and a lost feature is not silently reacquired.
    max_forward_backward_error_px: positive LK round-trip bound, default 1 px; not metric precision.
    window_size_px: odd LK window diameter, default 21 px, range 3..101. pyramid_levels: 0..6, default 3.
    min_eigenvalue: positive LK feature conditioning cutoff, default OpenCV 1e-4 (dimensionless).
    Returns diagnostics with feature identities, mask support, forward/backward pixel errors, host-time
        joint brackets and optional pixel residuals. Material identity/contact remain unverified.
    Example: analyze_video('video/manifest.json','fixed','sam/tracking.json',seeds,'review').
    """
    for value in (max_frame_gap_s, max_joint_gap_s, max_forward_backward_error_px, min_eigenvalue):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError("positive finite timing/flow thresholds required")
    if type(window_size_px) is not int or not 3 <= window_size_px <= 101 or window_size_px % 2 != 1:
        raise ValueError("window_size_px must be odd in 3..101")
    if type(pyramid_levels) is not int or not 0 <= pyramid_levels <= 6:
        raise ValueError("pyramid_levels must be in 0..6")
    video_path, tracking_path, output = Path(video_manifest), Path(tracking_path), Path(output_dir)
    manifest, video_digest = _snapshot(video_path)
    tracking, tracking_digest = _snapshot(tracking_path)
    if manifest.get("schema") != "ni_rua.so101.calibration_video.v1":
        raise ValueError("unsupported calibration video schema")
    if tracking.get("schema") != "ni_rua.sam31.video_tracking.v1":
        raise ValueError("unsupported SAM video tracking schema")
    if manifest.get("clock_domain") != "adapter_monotonic":
        raise ValueError("unsupported video clock domain")
    if manifest.get("status") not in ("completed", "truncated", "failed", "unknown") or tracking.get("status") not in ("completed", "incomplete", "failed", "cancelled") :
        raise ValueError("finish acquisition and inference before evidence analysis")
    frames = manifest["cameras"][camera_id]["frames"]
    if not frames or len(frames) != len(tracking["frames"]):
        raise ValueError("SAM output must inventory every recorded frame, including unknown results")
    raw_paths = [_inside(video_path.parent, row["path"]) for row in frames]
    shape = None
    for index, (path, row, sam) in enumerate(zip(raw_paths, frames, tracking["frames"])):
        if type(row["index"]) is not int or type(sam["frame_index"]) is not int or row["index"] != index or sam["frame_index"] != index or sam["source_file"] != path.name or row.get("source_sha256") != sam["source_sha256"]:
            raise ValueError("SAM frame identity/bytes do not match recorded calibration video")
        image = _image(path, sam["source_sha256"])
        if [row["width"], row["height"]] != [image.shape[1], image.shape[0]] or tracking.get("image_size_px") != [image.shape[1], image.shape[0]]:
            raise ValueError("recorded image dimensions disagree with metadata")
        ids = [obj["object_id"] for obj in sam["objects"]]
        if any(type(oid) is not int or oid < 0 for oid in ids) or len(set(ids)) != len(ids):
            raise ValueError("SAM object IDs must be unique nonnegative integers")
        if sam["status"] not in ("unknown", "mask_output"):
            raise ValueError("unsupported SAM frame status")
        for obj in sam["objects"]:
            visibility = obj.get("mask_visibility")
            if visibility not in ("mask_present", "empty_mask", "not_returned", "unknown"):
                raise ValueError("unsupported SAM mask visibility")
            if visibility == "mask_present" and (sam["status"] != "mask_output" or not obj.get("mask_file")):
                raise ValueError("present mask requires a returned frame and mask artifact")
        if shape is not None and image.shape != shape:
            raise ValueError("recorded video dimensions changed")
        shape = image.shape
    times = [float(f["capture_monotonic_s"]) for f in frames]
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("recorded frame host times must strictly increase")
    height, width = shape[:2]
    features = _features(list(features), len(frames), [width, height])
    touch = None if touch_report_path is None else json.loads(Path(touch_report_path).read_text())
    if touch is not None and touch.get("video", {}).get("manifest_sha256") != video_digest:
        raise ValueError("touch report does not bind this exact video manifest")
    joints, conflicts = _joint_samples(manifest, touch)
    model = None
    if any("link" in f for f in features):
        from ...embodiments.so101.profile import load_model
        model = load_model()
    output.mkdir(parents=True, exist_ok=False)
    (output / "overlays").mkdir()
    report = dict(schema=SCHEMA, status="completed", contact_evidence="unknown", material_identity_verified=None,
        acquisition_status=manifest.get("status"), visual_evidence_status=manifest.get("visual_evidence_status", "unknown"),
        coverage=manifest.get("coverage"), sam_inference_status=tracking.get("status"),
        camera_id=camera_id, method="SAM3.1 object masks plus separate OpenCV pyramidal LK",
        video_sha256=video_digest, tracking_sha256=tracking_digest,
        features=features, joint_conflicts_monotonic_s=conflicts,
        settings=dict(max_frame_gap_s=max_frame_gap_s, max_joint_gap_s=max_joint_gap_s,
                      max_forward_backward_error_px=max_forward_backward_error_px,
                      window_size_px=window_size_px, pyramid_levels=pyramid_levels, min_eigenvalue=min_eigenvalue),
        physical_acceptance=None, frames=[])
    active, lost, previous_gray = {}, set(), None
    lk = dict(winSize=(window_size_px, window_size_px), maxLevel=pyramid_levels,
              minEigThreshold=min_eigenvalue)
    try:
        for i, (frame, sam, path) in enumerate(zip(frames, tracking["frames"], raw_paths)):
            image = _image(path, sam["source_sha256"])
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            masks = {}
            for obj in sam["objects"]:
                if obj.get("mask_file") and obj.get("mask_visibility") == "mask_present":
                    mask_path = _inside(tracking_path.parent, obj["mask_file"])
                    mask = _image(mask_path, obj.get("mask_sha256"), cv2.IMREAD_GRAYSCALE)
                    if mask is None or mask.shape != gray.shape:
                        raise ValueError("SAM mask dimensions differ from recorded frame")
                    masks[obj["object_id"]] = mask > 0
            alignment = align_joints(joints, times[i], max_joint_gap_s, conflicts)
            row = dict(frame_index=i, source_file=frame["path"], host_monotonic_s=times[i],
                       host_read_interval_s=frame.get("host_read_interval_s"), exposure_monotonic_s=None,
                       joint_alignment=alignment, mask_status=sam["status"], features=[], contact_evidence="unknown")
            for f in features:
                fid = f["feature_id"]
                if i < f["frame_index"]:
                    continue
                reason, point, fb = None, None, None
                mask = masks.get(f["object_id"])
                if fid in lost:
                    reason = "previously_lost_requires_new_feature_identity"
                elif mask is None:
                    reason = "sam_object_mask_unavailable"
                elif i == f["frame_index"]:
                    point = np.asarray(f["point_px"], np.float32)
                elif times[i] - times[i-1] > max_frame_gap_s:
                    reason = "frame_gap_exceeds_limit"
                else:
                    old = active[fid].reshape(1,1,2).astype(np.float32)
                    forward, ok1, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, old, None, **lk)
                    if forward is None or not bool(ok1[0,0]) or not np.isfinite(forward).all():
                        reason = "forward_flow_failed"
                    else:
                        backward, ok2, _ = cv2.calcOpticalFlowPyrLK(gray, previous_gray, forward, None, **lk)
                        if backward is None or not bool(ok2[0,0]) or not np.isfinite(backward).all():
                            reason = "backward_flow_failed"
                        else:
                            fb = float(np.linalg.norm(backward-old))
                            if fb > max_forward_backward_error_px:
                                reason = "forward_backward_inconsistent"
                            else:
                                point = forward.reshape(2)
                if point is not None:
                    u, v = (int(round(float(c))) for c in point)
                    if np.any(point < 0) or np.any(point >= [width, height]) or not 0 <= u < width or not 0 <= v < height or not mask[v,u]:
                        reason, point = "feature_outside_sam_object", None
                if reason:
                    lost.add(fid)
                else:
                    active[fid] = point
                observed = point.tolist() if point is not None else None
                row["features"].append(dict(feature_id=fid, object_id=f["object_id"],
                    status="unknown" if reason else "seed" if i == f["frame_index"] else "tracked",
                    reason=reason, pixel=observed, forward_backward_error_px=fb,
                    material_identity_verified=None,
                    projection=_projection(manifest.get("camera_calibration"), camera_id, alignment, f, observed, model, [width, height])
                        if observed is not None else None))
            overlay = image.copy()
            for mask in masks.values():
                overlay[mask] = (overlay[mask].astype(float)*.7 + np.array([0,255,0])*.3).astype(np.uint8)
            for f in row["features"]:
                if f["pixel"] is not None:
                    cv2.circle(overlay, tuple(int(round(v)) for v in f["pixel"]), 4, (0,0,255), 1)
            relative = f"overlays/{i:06d}.png"
            if not cv2.imwrite(str(output / relative), overlay):
                raise OSError("failed to save review overlay")
            row["overlay"] = relative
            report["frames"].append(row)
            previous_gray = gray
    except Exception as error:
        report.update(status="failed", reason=f"{type(error).__name__}: {error}")
        evidence_path = output / "evidence.json"
        tmp_path = evidence_path.with_name(f".{evidence_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            tmp_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            os.replace(tmp_path, evidence_path)
        except (OSError, ValueError, TypeError) as write_err:
            tmp_path.unlink(missing_ok=True)
            error.add_note(f"evidence_report_write_failed: {type(write_err).__name__}: {write_err}")
        raise
    evidence_path = output / "evidence.json"
    tmp_path = evidence_path.with_name(f".{evidence_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(tmp_path, evidence_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    _review_html(output, report)
    return report


def _review_html(output, report):
    """Standalone operator review; no network requests, control links or implied contact verdict."""
    payload = json.dumps(report["frames"], ensure_ascii=False).replace("<", "\\u003c")
    page = '''<!doctype html><meta charset="utf-8"><title>Calibration video evidence</title>
<style>body{font:16px system-ui;max-width:1100px;margin:24px auto;padding:0 16px}img{max-width:100%}
input{width:100%}pre{white-space:pre-wrap}p{line-height:1.5}</style>
<h1>Calibration video evidence — CAMERA</h1><p>SAM object masks are green; independently seeded LK features are red.
Contact and material-point identity remain unverified. Host timestamps are not exposure timestamps.
Lost points stay unknown. These results do not change S5 inputs or active calibration.</p>
<input id="slider" type="range" min="0" value="0"><p id="label"></p><img id="frame"><pre id="facts"></pre>
<script>const frames=DATA;const slider=document.getElementById('slider');slider.max=frames.length-1;
function draw(){const f=frames[Number(slider.value)];document.getElementById('frame').src=f.overlay;
document.getElementById('label').textContent=`Frame ${f.frame_index} — host time ${f.host_monotonic_s} s`;
document.getElementById('facts').textContent=JSON.stringify(f,null,2)}slider.addEventListener('input',draw);draw();</script>'''
    (output / "review.html").write_text(page.replace("CAMERA", html.escape(report["camera_id"])).replace("const frames=DATA;", "const frames=" + payload + ";"))


def main(argv=None):
    """Offline CLI; required paths are immutable inputs and a new output directory; no hardware/model calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-manifest", required=True, help='sealed recording manifest.json with original PNG bindings and host seconds')
    parser.add_argument("--camera", required=True, help='explicit recorded camera ID, e.g. fixed; no default camera')
    parser.add_argument("--tracking", required=True, help='offline SAM tracking.json for the exact same frame bytes')
    parser.add_argument("--features", help="JSON list of explicitly seeded features; omitted means mask-only review")
    parser.add_argument("--touch-report", help='optional matching S1/S6 report; adds measured descent/sliding samples')
    parser.add_argument("--output", required=True, help='new output directory for evidence.json, review.html and overlays; never overwrite')
    parser.add_argument("--max-frame-gap-s", type=float, default=argparse.SUPPRESS, help='positive maximum consecutive frame gap in s (default .25); larger gaps lose the feature')
    parser.add_argument("--max-joint-gap-s", type=float, default=argparse.SUPPRESS, help='positive maximum measured-joint bracket span in s (default .25); no extrapolation')
    parser.add_argument("--max-forward-backward-error-px", type=float, default=argparse.SUPPRESS, help='positive LK round-trip error limit in pixels (default 1); not metric accuracy')
    parser.add_argument("--window-size-px", type=int, default=argparse.SUPPRESS, help='odd LK window diameter in pixels, 3..101 (default 21)')
    parser.add_argument("--pyramid-levels", type=int, default=argparse.SUPPRESS, help='LK scale-pyramid depth, integer 0..6 (default 3)')
    parser.add_argument("--min-eigenvalue", type=float, default=argparse.SUPPRESS, help='positive dimensionless LK conditioning cutoff (default .0001)')
    parser.add_argument("--config", help="shared offline evidence JSON; explicit CLI settings override the file; omitted uses packaged defaults")
    args = parser.parse_args(argv)
    settings = evidence_settings("video_evidence", args.config,
        {key: getattr(args, key) for key in DEFAULTS if hasattr(args, key)})
    try:
        result = analyze_video(args.video_manifest, args.camera, args.tracking,
            [] if args.features is None else json.loads(Path(args.features).read_text()), args.output,
            touch_report_path=args.touch_report, **settings)
        print(json.dumps(dict(status=result["status"], frames=len(result["frames"]), contact_evidence="unknown")))
        return 0
    except (ValueError, OSError, KeyError, TypeError, cv2.error) as error:
        print(json.dumps(dict(status="failed", reason=str(error))))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
