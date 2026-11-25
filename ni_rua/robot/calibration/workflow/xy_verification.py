"""Operator-only camera/arm XY consistency from observed fixed-fingertip pixels.

This compares an independently selected image feature with CAD/FK; it neither
fits calibration nor moves the arm. The wrist pose still depends on FK, so this
is not an independent metrology certificate for absolute base-frame accuracy.
"""
import numpy as np

from ...model.geometry import array, transform, triangulate
from ...embodiments.so101.gripper_cad import fingertip_reference

XY_SETTING_KEYS = {"max_xy_error_m": "xy_max_error_m", "max_reprojection_error_px": "xy_max_reprojection_error_px",
                   "min_parallax_deg": "xy_min_parallax_deg"}


def validate_xy_request(params, *, defaults=None):
    """Validate S6 xy_capture/xy_evaluate operator parameters (no motion).

    xy_capture needs only action. xy_evaluate requires snapshot_id (opaque
    frozen capture), fixed_pixel/wrist_pixel ([u,v] in original image pixels),
    max_xy_error_m (positive lateral tolerance in m, e.g. .005),
    max_reprojection_error_px (positive stereo threshold in px, e.g. 3), and
    min_parallax_deg (0..90 deg, e.g. 1). Omitted thresholds are filled only from
    ``defaults`` (same three keys, provided by the worker's deployment); without
    defaults they remain unresolved for gateway validation. Explicit values win. The two
    pixels must identify the same fixed inner fingertip used by the CAD model.
    """
    action = params.get("action")
    keys = {"action"} if action == "xy_capture" else {
        "action", "snapshot_id", "fixed_pixel", "wrist_pixel", "max_xy_error_m",
        "max_reprojection_error_px", "min_parallax_deg"}
    required = keys - set(XY_SETTING_KEYS) if action == "xy_evaluate" else keys
    if action not in ("xy_capture", "xy_evaluate") or not required <= set(params) or set(params) - keys:
        raise ValueError("invalid_xy_parameters")
    if action == "xy_capture":
        return dict(params)
    params = {**({key: defaults[key] for key in XY_SETTING_KEYS if key in defaults} if defaults is not None else {}), **params}
    if not isinstance(params["snapshot_id"], str) or not params["snapshot_id"]:
        raise ValueError("invalid_xy_snapshot_id")
    for key in ("fixed_pixel", "wrist_pixel"):
        array(params[key], (2,))
    for key, bound in (("max_xy_error_m", .1), ("max_reprojection_error_px", 100), ("min_parallax_deg", 90)):
        if key not in params:
            continue
        v = params[key]
        if type(v) not in (float, int) or not 0 < v < bound:
            raise ValueError("invalid_xy_threshold:" + key)
    return dict(params)


def evaluate_xy(snapshot, params):
    """Compare fixed-tip image triangulation with captured CAD/FK, all lengths m.

    snapshot contains frozen camera K/D/T/width/height and T_base_tcp (4x4).
    Params follow validate_xy_request; no movement, fitting or offsets applied.
    Returns status unknown for rejected stereo, otherwise passed/failed for
    this single sample's XY norm against the explicitly requested tolerance.
    Selection uncertainty and shared calibration errors remain unbounded.
    """
    p = validate_xy_request(params)
    if not set(XY_SETTING_KEYS) <= set(p):
        raise ValueError("xy_thresholds_unresolved")
    if p["snapshot_id"] != snapshot["snapshot_id"]:
        raise ValueError("xy_snapshot_mismatch")
    cameras = []
    for name in ("fixed", "wrist"):
        c = dict(snapshot["cameras"][name])
        c.update(K=array(c["K"], (3, 3)), D=array(c["D"]), T=transform(c["T"]))
        cameras.append(c)
    stereo = triangulate(*cameras, [p["fixed_pixel"]], [p["wrist_pixel"]],
                         min_angle_deg=p["min_parallax_deg"], max_error_px=p["max_reprojection_error_px"])
    out = dict(status="unknown", passed=None, snapshot_id=p["snapshot_id"],
               frame_id="so101_base", reference="fixed_inner_fingertip", unit="m",
               parameters=p, stereo=stereo, annotation_uncertainty_m=None,
               semantics="Single-sample camera/arm consistency, conditional on matching the same CAD fixed inner fingertip. Shared calibration and annotation errors are not bounded; not absolute accuracy certification.")
    if not stereo["accepted"]:
        return dict(out, reason=stereo["point_status"][0])
    T = transform(snapshot["T_base_tcp"])
    tip = np.asarray(fingertip_reference(0.)["fixed_tip_m"])
    predicted = T[:3, :3] @ tip + T[:3, 3]
    observed = np.asarray(stereo["points_m"][0])
    delta = observed - predicted
    xy = float(np.linalg.norm(delta[:2]))
    passed = xy <= p["max_xy_error_m"]
    return dict(out, status="passed" if passed else "failed", passed=passed,
                predicted_tip_m=predicted.tolist(), observed_tip_m=observed.tolist(),
                observed_minus_predicted_m=delta.tolist(), xy_error_m=xy)
