"""Explicit LIBERO Franka Emika Panda finger contact references (simulation-only: robosuite PandaGripper)."""
from __future__ import annotations

import math

import numpy as np

from . import pose_convention

CONTACT_FIELDS = frozenset({"contact_region", "contact_depth_m"})
#: MuJoCo-measured axial depths relative to ``gripper0_grip_site`` (positive toward palm along local ``-Z``).
CONTACT_DEPTH_PRESETS_M = {
    "pad_center": 0.0,
    "finger_middle": 0.0176,
    "fingertip": -0.0093,
}
MIN_CONTACT_DEPTH_M = -0.0093
MAX_CONTACT_DEPTH_M = 0.0445
TCP_CONFIG_ID = "libero-panda-grip-site-v1"
FRAME_ID = "franka_base"


def _depth(value) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("invalid_contact_depth_m: requires a finite number")
    if not MIN_CONTACT_DEPTH_M <= value <= MAX_CONTACT_DEPTH_M:
        raise ValueError("invalid_contact_depth_m: outside exposed inner finger span")
    return float(value)


def _validated_spec(spec: dict) -> dict:
    depth = _depth(spec["contact_depth_m"])
    region = spec["contact_region"]
    if not isinstance(region, str) or (
        region != "custom" and (region not in CONTACT_DEPTH_PRESETS_M or depth != CONTACT_DEPTH_PRESETS_M[region])
    ):
        raise ValueError("invalid_contact_region_spec")
    offset = pose_convention.GRASP_POINT_OFFSET_M - depth
    if (
        type(spec.get("offset_m")) not in (int, float)
        or not math.isfinite(spec["offset_m"])
        or abs(spec["offset_m"] - offset) > 1e-12
    ):
        raise ValueError("invalid_contact_offset_spec")
    return dict(contact_region=region, contact_depth_m=depth, offset_m=offset)


def resolve_contact_spec(selection: dict, inherited: dict | None = None) -> dict:
    """Resolve a grasp pose's explicit contact selection, otherwise inherit the request selection."""
    if not isinstance(selection, dict):
        raise ValueError("invalid_contact_selection")
    selected = CONTACT_FIELDS.intersection(selection)
    if len(selected) > 1:
        raise ValueError("contact_region_and_contact_depth_m_are_mutually_exclusive")
    if "contact_region" in selected:
        region = selection["contact_region"]
        if not isinstance(region, str) or region not in CONTACT_DEPTH_PRESETS_M:
            raise ValueError("invalid_contact_region")
        depth = CONTACT_DEPTH_PRESETS_M[region]
    elif "contact_depth_m" in selected:
        region, depth = "custom", _depth(selection["contact_depth_m"])
    elif inherited is not None:
        return _validated_spec(inherited)
    else:
        region, depth = "pad_center", 0.0
    return dict(
        contact_region=region,
        contact_depth_m=depth,
        offset_m=pose_convention.GRASP_POINT_OFFSET_M - depth,
    )


def values_to_contact_pose(values: dict[str, float], spec: dict) -> np.ndarray:
    """Convert the selected contact cross-section's world target into ``gripper0_grip_site`` pose."""
    spec = _validated_spec(spec)
    pose = pose_convention.values_to_pose(values)
    if spec["contact_depth_m"]:
        pose[:3] += spec["contact_depth_m"] * pose_convention.tool_axis(pose[3:])
    return pose


def contact_capabilities() -> dict:
    """Static disclosed simulation gripper geometry; no object state or grip guarantee."""
    return dict(
        default_reference="pad_center",
        regions={name: resolve_contact_spec({"contact_region": name}) for name in CONTACT_DEPTH_PRESETS_M},
        contact_depth_m=dict(
            min=MIN_CONTACT_DEPTH_M,
            max=MAX_CONTACT_DEPTH_M,
            positive_direction="from pad_center (gripper0_grip_site) toward palm along local -z",
        ),
        inner_face_axial_span_from_tcp_m=[-0.04447, 0.0093],
        distal_pad_box_axial_span_from_tcp_m=[-0.0116, 0.0044],
        simulation_only=True,
        semantics=(
            "grasp_position_m targets the selected cross-section between PandaGripper fingers; omitted selection "
            "uses pad_center (coincident with gripper0_grip_site, offset 0.0 m). finger_middle sits 17.6 mm deeper "
            "toward the palm; fingertip sits 9.3 mm beyond gripper0_grip_site. References do not certify clearance, "
            "contact or retention."
        ),
        observation_reference=(
            "robot_state.arms['arm0'].grasp_pose always reports pad_center; receipt.grasp_contact_feedback "
            "reports the selected reference from the terminal measured gripper0_grip_site pose."
        ),
    )


def _vector(value, size: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (size,) or not np.isfinite(arr).all():
        raise ValueError("invalid_contact_feedback_pose")
    return arr


def contact_target_feedback(action: dict, robot_state: dict) -> dict:
    """Compare final requested contact references to terminal measured ``gripper0_grip_site`` pose."""
    metadata = action.get("input_grasp_pose")
    if not isinstance(metadata, dict):
        return {}
    points = action.get("path", {}).get("waypoints", [])
    if not points:
        return {}
    output = {}
    for arm in dict.fromkeys(arm for point in points for arm in point):
        grasp = metadata.get(arm) if arm in metadata else metadata
        if not isinstance(grasp, dict) or "position_m" not in grasp:
            continue
        frames = grasp.get("waypoint_contact_frames")
        if frames:
            spec = _validated_spec(frames[-1])
        else:
            spec = resolve_contact_spec({})
            if abs(grasp.get("offset_m", spec["offset_m"]) - spec["offset_m"]) > 1e-12:
                raise ValueError("contact_feedback_missing_selected_frame")
        target = _vector(grasp["position_m"], 3)
        angles = _vector(grasp["angles_deg"], 3)
        wanted_q = pose_convention.angles_to_quat(*angles)
        target_axis = pose_convention.tool_axis(wanted_q)
        entry = dict(**spec, requested_contact_position_m=target.tolist(), available=False)
        pose = robot_state.get("arms", {}).get(arm, {}).get("ee_pose", {})
        if (
            pose.get("source_kind") != "measurement"
            or pose.get("tcp_config_id") != TCP_CONFIG_ID
            or pose.get("frame_id") != action.get("frame_id", FRAME_ID)
        ):
            entry["unavailable_reason"] = "terminal_measured_gripper0_grip_site_unavailable"
            output[arm] = entry
            continue
        ee_position = _vector(pose["position_m"], 3)
        xyzw = _vector(pose["quaternion_xyzw"], 4)
        actual_q = np.array([xyzw[3], *xyzw[:3]])
        actual = ee_position + spec["offset_m"] * pose_convention.tool_axis(actual_q)
        error = actual - target
        entry.update(
            available=True,
            actual_contact_position_m=actual.tolist(),
            position_error_vector_m=error.tolist(),
            position_error_m=float(np.linalg.norm(error)),
            orientation_error_rad=math.radians(pose_convention.orientation_error_deg(wanted_q, actual_q)),
            approach_shortfall_m=float(np.dot(-error, target_axis)),
        )
        output[arm] = entry
    if not output:
        return {}
    return dict(
        target_scope="requested_final_pose_not_completion_claim",
        source_kind="derived_from_terminal_measured_gripper0_grip_site",
        frame_id=action.get("frame_id", FRAME_ID),
        arms=output,
        interpretation=(
            "Positive approach_shortfall_m means the selected reference remains behind its requested final target "
            "along the requested approach axis. The final target may not have been attempted after failure/stop. "
            "Robot geometry and pose feedback do not establish object contact or retention."
        ),
    )
