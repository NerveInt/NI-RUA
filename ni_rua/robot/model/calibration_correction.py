"""Pure SE(3) online camera extrinsic correction & exposure synchronization contract.

Stdlib + NumPy ONLY (zero ``perception``, ``scipy``, or ``cv2`` imports), shared cleanly
between ``ni_rua.robot.model.geometry``, ``ni_rua.perception.scene.store``, and
``ni_rua.perception.external.online_self_calibration``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

# 真机硬件单调曝光时钟步域标识
HARDWARE_MONOTONIC_STEP_DOMAIN = "hardware_monotonic_ns"
# 默认最大允许的多传感器硬件曝光对齐时间窗（25 ms = 25,000,000 ns）
DEFAULT_MAX_HARDWARE_SYNC_SKEW_NS = 25_000_000
# 硬件曝光同步窗口物理硬上界（100 ms = 100,000,000 ns）
MAX_HARDWARE_SYNC_SKEW_BOUND_NS = 100_000_000

# 在线标定物理合理性安全硬边界（防止异常输入伪造离谱外参）
MAX_CALIB_TRANSLATION_NORM_M = 0.25
MAX_CALIB_ROTATION_NORM_DEG = 25.0
MAX_CALIB_TIME_OFFSET_ABS_MS = 100.0


def _is_finite_real(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))


def _is_valid_int(val: Any) -> bool:
    return isinstance(val, int) and not isinstance(val, bool)


def _vec3_tuple(val: Any, field_name: str) -> tuple[float, float, float]:
    if not isinstance(val, (list, tuple, np.ndarray)) or len(val) != 3:
        raise ValueError(f"invalid_calibration_correction:{field_name}_requires_3_elements")
    out = tuple(float(x) for x in val)
    if any(isinstance(x, bool) for x in val) or not all(math.isfinite(x) for x in out):
        raise ValueError(f"invalid_calibration_correction:{field_name}_nonfinite")
    return (out[0], out[1], out[2])


def _skew(vec: np.ndarray) -> np.ndarray:
    vx, vy, vz = float(vec[0]), float(vec[1]), float(vec[2])
    return np.array(
        [[0.0, -vz, vy], [vz, 0.0, -vx], [-vy, vx, 0.0]],
        dtype=np.float64,
    )


def rodrigues_exp_rad(omega_rad: np.ndarray) -> np.ndarray:
    """Map a 3D axis-angle vector (unit: ``rad``) to a ``3x3`` orthogonal rotation matrix via ``SO(3)`` ``exp``."""
    w = np.asarray(omega_rad, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(w))
    if theta < 1e-10:
        return np.eye(3, dtype=np.float64) + _skew(w)
    K = _skew(w / theta)
    return np.eye(3, dtype=np.float64) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def so3_left_jacobian(omega_rad: np.ndarray) -> np.ndarray:
    """Compute the ``SO(3)`` left Jacobian ``J_l(omega)`` for ``omega_rad`` in ``rad``."""
    w = np.asarray(omega_rad, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(w))
    W = _skew(w)
    if theta < 1e-7:
        return np.eye(3, dtype=np.float64) + 0.5 * W + (1.0 / 6.0) * (W @ W)
    theta2 = theta * theta
    c1 = (1.0 - math.cos(theta)) / theta2
    c2 = (theta - math.sin(theta)) / (theta2 * theta)
    return np.eye(3, dtype=np.float64) + c1 * W + c2 * (W @ W)


@dataclass(frozen=True)
class CalibrationCorrection:
    """Versioned online camera extrinsic & exposure-latency correction snapshot (base-frame fixed-point contract).

    Fields & units:
    - ``calib_rev``: Non-negative integer revision (``0`` = identity / zero correction).
    - ``delta_t_base_cam_m``: ``{camera_id: (dx_m, dy_m, dz_m)}`` optical-centre translation offset in base frame (``m``).
    - ``delta_rot_base_cam_deg``: ``{camera_id: (rx_deg, ry_deg, rz_deg)}`` axis-angle rotation offset in base frame (``deg``).
    - ``time_offset_ms``: Global camera timestamp lag ``t_stamp - t_exp`` in ``ms``.
    - ``reprojection_rmse_px``: FK silhouette/keypoint reprojection RMS residual in ``px``.
    - ``position_sigma_m``: 1-sigma 3D position uncertainty at characteristic workspace depth in ``m``.
    - ``camera_time_offset_ms``: Optional per-camera timestamp lag ``{camera_id: offset_ms}`` in ``ms``.
    - ``observable_dof_mask``: Optional per-camera 6-bool observability mask ``(tx, ty, tz, rx, ry, rz)``.
    """

    calib_rev: int = 0
    delta_t_base_cam_m: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    delta_rot_base_cam_deg: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    time_offset_ms: float = 0.0
    reprojection_rmse_px: float = 0.0
    position_sigma_m: float = 0.0
    camera_time_offset_ms: dict[str, float] = field(default_factory=dict)
    observable_dof_mask: dict[str, tuple[bool, bool, bool, bool, bool, bool]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _is_valid_int(self.calib_rev) or self.calib_rev < 0:
            raise ValueError("invalid_calibration_correction:calib_rev")
        if not isinstance(self.delta_t_base_cam_m, Mapping):
            raise ValueError("invalid_calibration_correction:delta_t_base_cam_m")
        if not isinstance(self.delta_rot_base_cam_deg, Mapping):
            raise ValueError("invalid_calibration_correction:delta_rot_base_cam_deg")

        norm_t: dict[str, tuple[float, float, float]] = {}
        for cid, vec in self.delta_t_base_cam_m.items():
            if not isinstance(cid, str) or not cid.strip():
                raise ValueError("invalid_calibration_correction:empty_camera_id")
            t_tup = _vec3_tuple(vec, f"delta_t_base_cam_m[{cid}]")
            if math.hypot(*t_tup) > MAX_CALIB_TRANSLATION_NORM_M:
                raise ValueError("invalid_calibration_correction:translation_out_of_bounds")
            norm_t[cid.strip()] = t_tup

        norm_r: dict[str, tuple[float, float, float]] = {}
        for cid, vec in self.delta_rot_base_cam_deg.items():
            if not isinstance(cid, str) or not cid.strip():
                raise ValueError("invalid_calibration_correction:empty_camera_id")
            r_tup = _vec3_tuple(vec, f"delta_rot_base_cam_deg[{cid}]")
            if math.hypot(*r_tup) > MAX_CALIB_ROTATION_NORM_DEG:
                raise ValueError("invalid_calibration_correction:rotation_out_of_bounds")
            norm_r[cid.strip()] = r_tup

        if not _is_finite_real(self.time_offset_ms) or abs(float(self.time_offset_ms)) > MAX_CALIB_TIME_OFFSET_ABS_MS:
            raise ValueError("invalid_calibration_correction:time_offset_ms")
        if not _is_finite_real(self.reprojection_rmse_px) or float(self.reprojection_rmse_px) < 0.0:
            raise ValueError("invalid_calibration_correction:reprojection_rmse_px")
        if not _is_finite_real(self.position_sigma_m) or float(self.position_sigma_m) < 0.0:
            raise ValueError("invalid_calibration_correction:position_sigma_m")

        if not isinstance(self.camera_time_offset_ms, Mapping):
            raise ValueError("invalid_calibration_correction:camera_time_offset_ms")
        norm_cam_dt: dict[str, float] = {}
        for cid, val in self.camera_time_offset_ms.items():
            if not isinstance(cid, str) or not cid.strip() or not _is_finite_real(val):
                raise ValueError("invalid_calibration_correction:camera_time_offset_entry")
            fval = float(val)
            if abs(fval) > MAX_CALIB_TIME_OFFSET_ABS_MS:
                raise ValueError("invalid_calibration_correction:camera_time_offset_out_of_bounds")
            norm_cam_dt[cid.strip()] = fval

        if not isinstance(self.observable_dof_mask, Mapping):
            raise ValueError("invalid_calibration_correction:observable_dof_mask")
        norm_mask: dict[str, tuple[bool, bool, bool, bool, bool, bool]] = {}
        for cid, mval in self.observable_dof_mask.items():
            if (
                not isinstance(cid, str)
                or not cid.strip()
                or not isinstance(mval, (list, tuple))
                or len(mval) != 6
                or not all(isinstance(b, bool) for b in mval)
            ):
                raise ValueError("invalid_calibration_correction:observable_dof_mask_entry")
            norm_mask[cid.strip()] = (
                bool(mval[0]),
                bool(mval[1]),
                bool(mval[2]),
                bool(mval[3]),
                bool(mval[4]),
                bool(mval[5]),
            )

        if self.calib_rev == 0:
            has_nonzero = any(any(abs(x) > 1e-12 for x in v) for v in norm_t.values()) or any(
                any(abs(x) > 1e-12 for x in v) for v in norm_r.values()
            )
            if has_nonzero:
                raise ValueError("invalid_calibration_correction:calib_rev_zero_with_nonzero_delta")

        object.__setattr__(self, "delta_t_base_cam_m", norm_t)
        object.__setattr__(self, "delta_rot_base_cam_deg", norm_r)
        object.__setattr__(self, "time_offset_ms", float(self.time_offset_ms))
        object.__setattr__(self, "reprojection_rmse_px", float(self.reprojection_rmse_px))
        object.__setattr__(self, "position_sigma_m", float(self.position_sigma_m))
        object.__setattr__(self, "camera_time_offset_ms", norm_cam_dt)
        object.__setattr__(self, "observable_dof_mask", norm_mask)

    def has_camera(self, camera_id: str) -> bool:
        return camera_id in self.delta_t_base_cam_m or camera_id in self.delta_rot_base_cam_deg

    def time_offset_ms_for(self, camera_id: str | None = None) -> float:
        if camera_id and camera_id in self.camera_time_offset_ms:
            return self.camera_time_offset_ms[camera_id]
        return self.time_offset_ms

    def apply_to_camera_pose(self, T_nominal: np.ndarray | Sequence[Sequence[float]], camera_id: str) -> np.ndarray:
        T = np.asarray(T_nominal, dtype=np.float64)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            raise ValueError("invalid_nominal_camera_transform")
        out = T.copy()
        if not self.has_camera(camera_id):
            return out
        dt = np.asarray(self.delta_t_base_cam_m.get(camera_id, (0.0, 0.0, 0.0)), dtype=np.float64)
        dr_rad = np.deg2rad(np.asarray(self.delta_rot_base_cam_deg.get(camera_id, (0.0, 0.0, 0.0)), dtype=np.float64))
        out[:3, :3] = rodrigues_exp_rad(dr_rad) @ T[:3, :3]
        out[:3, 3] = T[:3, 3] + dt
        return out

    def camera_summary(self, camera_id: str) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "calib_rev": self.calib_rev,
            "applied": self.has_camera(camera_id) and self.calib_rev > 0,
            "delta_t_base_cam_m": list(self.delta_t_base_cam_m.get(camera_id, (0.0, 0.0, 0.0))),
            "delta_rot_base_cam_deg": list(self.delta_rot_base_cam_deg.get(camera_id, (0.0, 0.0, 0.0))),
            "time_offset_ms": round(self.time_offset_ms_for(camera_id), 4),
            "reprojection_rmse_px": round(self.reprojection_rmse_px, 4),
            "position_sigma_m": round(self.position_sigma_m, 6),
        }
        if camera_id in self.observable_dof_mask:
            summary["observable_dof_mask"] = list(self.observable_dof_mask[camera_id])
        return summary

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "calib_rev": self.calib_rev,
            "delta_t_base_cam_m": {k: list(v) for k, v in self.delta_t_base_cam_m.items()},
            "delta_rot_base_cam_deg": {k: list(v) for k, v in self.delta_rot_base_cam_deg.items()},
            "time_offset_ms": self.time_offset_ms,
            "reprojection_rmse_px": self.reprojection_rmse_px,
            "position_sigma_m": self.position_sigma_m,
        }
        if self.camera_time_offset_ms:
            out["camera_time_offset_ms"] = dict(self.camera_time_offset_ms)
        if self.observable_dof_mask:
            out["observable_dof_mask"] = {k: list(v) for k, v in self.observable_dof_mask.items()}
        return out

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | CalibrationCorrection) -> CalibrationCorrection:
        if isinstance(data, cls):
            return data
        if not isinstance(data, Mapping):
            raise ValueError("invalid_calibration_correction:must_be_mapping")
        allowed = {
            "calib_rev",
            "delta_t_base_cam_m",
            "delta_rot_base_cam_deg",
            "time_offset_ms",
            "reprojection_rmse_px",
            "position_sigma_m",
            "camera_time_offset_ms",
            "observable_dof_mask",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"invalid_calibration_correction:unexpected_fields:{sorted(unknown)}")
        if "calib_rev" not in data:
            raise ValueError("invalid_calibration_correction:missing_calib_rev")

        raw_time = data.get("time_offset_ms", 0.0)
        cam_time_map = dict(data.get("camera_time_offset_ms") or {})
        if isinstance(raw_time, Mapping):
            for k, v in raw_time.items():
                cam_time_map[str(k)] = v
            scalar_time = float(next(iter(raw_time.values()), 0.0)) if raw_time else 0.0
        else:
            scalar_time = raw_time

        return cls(
            calib_rev=data["calib_rev"],
            delta_t_base_cam_m=data.get("delta_t_base_cam_m") or {},
            delta_rot_base_cam_deg=data.get("delta_rot_base_cam_deg") or {},
            time_offset_ms=scalar_time,
            reprojection_rmse_px=data.get("reprojection_rmse_px", 0.0),
            position_sigma_m=data.get("position_sigma_m", 0.0),
            camera_time_offset_ms=cam_time_map,
            observable_dof_mask=data.get("observable_dof_mask") or {},
        )


def validate_hardware_exposure_sync(
    observation: Mapping[str, Any],
    *,
    frame: Mapping[str, Any] | None = None,
    sample: Mapping[str, Any] | None = None,
    state: Mapping[str, Any] | None = None,
    correction: CalibrationCorrection | None = None,
    camera_id: str | None = None,
    max_skew_ns: int | None = None,
) -> dict[str, int]:
    """Validate bounded nanosecond exposure synchronization in ``hardware_monotonic_ns`` step domain."""
    if observation.get("step_domain") != HARDWARE_MONOTONIC_STEP_DOMAIN:
        raise ValueError("unsynchronized_observation")
    obs_ns = observation.get("sim_step")
    if not _is_valid_int(obs_ns) or obs_ns < 0:
        raise ValueError("unsynchronized_observation")

    raw_skew = observation.get("max_sync_skew_ns", DEFAULT_MAX_HARDWARE_SYNC_SKEW_NS) if max_skew_ns is None else max_skew_ns
    if not _is_valid_int(raw_skew) or not (0 <= raw_skew <= MAX_HARDWARE_SYNC_SKEW_BOUND_NS):
        raise ValueError("invalid_max_sync_skew_ns")
    effective_skew_ns = int(raw_skew)

    frame_ns = obs_ns
    if frame is not None:
        if frame.get("step_domain") != HARDWARE_MONOTONIC_STEP_DOMAIN:
            raise ValueError("unsynchronized_observation")
        f_step = frame.get("sim_step")
        if not _is_valid_int(f_step) or f_step < 0 or abs(f_step - obs_ns) > effective_skew_ns:
            raise ValueError("unsynchronized_observation")
        frame_ns = int(f_step)

    if sample is not None:
        if sample.get("step_domain") != HARDWARE_MONOTONIC_STEP_DOMAIN:
            raise ValueError("unsynchronized_observation")
        s_step = sample.get("sim_step")
        if not _is_valid_int(s_step) or s_step < 0 or abs(s_step - frame_ns) > effective_skew_ns:
            raise ValueError("unsynchronized_observation")

    state_skew_ns = 0
    if state is not None:
        if state.get("step_domain") != HARDWARE_MONOTONIC_STEP_DOMAIN:
            raise ValueError("unsynchronized_observation")
        st_step = state.get("sim_step")
        if not _is_valid_int(st_step) or st_step < 0:
            raise ValueError("unsynchronized_observation")
        offset_ns = round(correction.time_offset_ms_for(camera_id) * 1e6) if correction is not None else 0
        expected_compensated_ns = frame_ns - offset_ns
        skew_raw = abs(st_step - frame_ns)
        skew_comp = abs(st_step - expected_compensated_ns)
        state_skew_ns = min(skew_raw, skew_comp)
        if state_skew_ns > effective_skew_ns:
            raise ValueError("unsynchronized_observation")

    return {
        "observation_ns": int(obs_ns),
        "frame_ns": int(frame_ns),
        "state_sync_skew_ns": int(state_skew_ns),
        "max_sync_skew_ns": effective_skew_ns,
    }


def validate_exposure_sync(
    observation: Mapping[str, Any],
    *,
    frame: Mapping[str, Any] | None = None,
    sample: Mapping[str, Any] | None = None,
    state: Mapping[str, Any] | None = None,
    correction: CalibrationCorrection | None = None,
    camera_id: str | None = None,
    max_skew_ns: int | None = None,
) -> dict[str, int] | None:
    """Validate observation clock alignment across both ``hardware_monotonic_ns`` and discrete step domains.

    On ``hardware_monotonic_ns``, enforces bounded exposure skew (``validate_hardware_exposure_sync``).
    On discrete domains (such as ``adapter_control_step`` or ``so101_control_step``), enforces exact
    ``sim_step`` and ``step_domain`` equality and rejects non-zero ``time_offset_ms``.
    """
    domain = observation.get("step_domain")
    if domain == HARDWARE_MONOTONIC_STEP_DOMAIN:
        return validate_hardware_exposure_sync(
            observation,
            frame=frame,
            sample=sample,
            state=state,
            correction=correction,
            camera_id=camera_id,
            max_skew_ns=max_skew_ns,
        )
    step = observation.get("sim_step")
    if step is None:
        raise ValueError("unsynchronized_observation")
    for part in (frame, sample, state):
        if part is not None and (part.get("sim_step") != step or part.get("step_domain") != domain):
            raise ValueError("unsynchronized_observation")
    if correction is not None and abs(correction.time_offset_ms_for(camera_id)) > 1e-12:
        raise ValueError("discrete_step_domain_forbids_nonzero_time_offset")
    return None


__all__ = [
    "CalibrationCorrection",
    "DEFAULT_MAX_HARDWARE_SYNC_SKEW_NS",
    "HARDWARE_MONOTONIC_STEP_DOMAIN",
    "MAX_CALIB_ROTATION_NORM_DEG",
    "MAX_CALIB_TIME_OFFSET_ABS_MS",
    "MAX_CALIB_TRANSLATION_NORM_M",
    "MAX_HARDWARE_SYNC_SKEW_BOUND_NS",
    "rodrigues_exp_rad",
    "so3_left_jacobian",
    "validate_exposure_sync",
    "validate_hardware_exposure_sync",
]
