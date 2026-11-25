"""Pure helpers of the SO-101 in-transit sag work: the measured deviation of a move from its commanded path, which
moves may carry a sag offset, and the fits of the identification bench.

No bus, camera or clock: numpy over joint rows and a forward-kinematics callable, so the worker, the developer
benches (``scripts/so101_motion_bench.py``, ``scripts/so101_gravity_identify.py``) and CPU tests share one
implementation. Units: m, rad, N·m. The deployment switches are ``deployment.MOTION_COMPENSATION_FIELDS``.
"""
from __future__ import annotations

import numpy as np

from .driver import ARM_JOINTS


def contact_stage(path) -> bool:
    """True when a compiled move path asks for contact or compliance: ``stall_guard_preset`` ``contact_stop``,
    ``tolerance_preset`` ``compliant`` or a vertical arrival window (``max_upward_dz_m`` > 0 m). On such a move a
    settled joint error may come from the support, not from gravity, so no sag offset is carried into it or out of it
    (``SO101Adapter._carry``)."""
    return (path.get("stall_guard_preset") == "contact_stop" or path.get("tolerance_preset") == "compliant"
            or float(path.get("max_upward_dz_m") or 0.0) > 0.0)


def transit_deviation(tcp, start_q, commanded_rows, measured_rows, planned_steps):
    """How far the measured TCP left the commanded path while a move was streaming, or None without samples.

    ``tcp(q_rad)``: forward kinematics returning the 4x4 TCP pose (m). ``start_q``: the measured joints the
    interpolation started from (rad). ``commanded_rows``: the interpolated joint goals sent so far, WITHOUT any sag
    or gravity offset (rad, one row per command). ``measured_rows``: the joints read one command period after each
    row (same length). ``planned_steps``: commands of the whole move, the denominator of the fractions.

    The commanded path is the TCP polyline through ``start_q`` and the rows. Each measured TCP is compared with the
    nearest point of the part of that polyline already commanded, so servo lag along the path is not counted as
    deviation. Returns (m, rounded to 0.1 mm; fractions 0 = start, 1 = last planned command):
    ``max_below_path_m`` (largest amount the measured TCP was lower in Z than its nearest path point; 0 when never
    below), ``max_below_at_fraction``, ``max_deviation_m`` (largest 3D distance to the path),
    ``max_deviation_at_fraction`` and ``samples``. Example: a 13.5 mm drop right after the first command gives
    ``max_below_path_m`` 0.0135 at fraction 0.02."""
    count = min(len(commanded_rows), len(measured_rows))
    if count == 0:
        return None
    path = np.array([np.asarray(tcp(q))[:3, 3] for q in (start_q, *commanded_rows[:count])], dtype=float)
    below, distance = np.zeros(count), np.zeros(count)
    for index in range(count):
        point = np.asarray(tcp(measured_rows[index]))[:3, 3]
        begin, end = path[:index + 1], path[1:index + 2]
        span = end - begin
        length = np.einsum("ij,ij->i", span, span)
        along = np.clip(np.einsum("ij,ij->i", point - begin, span) / np.where(length > 0, length, 1.0), 0.0, 1.0)
        nearest = begin + along[:, None] * span
        gaps = np.linalg.norm(point - nearest, axis=1)
        closest = int(gaps.argmin())
        distance[index] = gaps[closest]
        below[index] = max(0.0, float(nearest[closest][2] - point[2]))
    total = max(int(planned_steps), count)
    worst_below, worst = int(below.argmax()), int(distance.argmax())
    return {"samples": count,
            "max_below_path_m": round(float(below[worst_below]), 4),
            "max_below_at_fraction": round((worst_below + 1) / total, 3),
            "max_deviation_m": round(float(distance[worst]), 4),
            "max_deviation_at_fraction": round((worst + 1) / total, 3)}


def fit_stiffness(errors_rad, torques_nm, min_torque_range_nm):
    """Per-joint servo position stiffness from steady-state holds (``scripts/so101_gravity_identify.py``).

    ``errors_rad``: N x 5 settled ``goal - measured`` (rad) of the arm joints, one row per hold. ``torques_nm``:
    N x 5 gravity torques at the measured joints (N·m; ``so101_gravity.gravity_torques``). A pure-P servo rests at
    ``error = -torque / stiffness``, so each joint is fitted by least squares as ``error = compliance * (-torque) +
    intercept`` (the intercept absorbs a constant bias such as a zero-offset error; it is reported, not used).
    ``min_torque_range_nm``: a joint whose gravity torque varied less than this over the holds (N·m) is not fitted:
    the data cannot separate its stiffness from friction (shoulder_pan and wrist_roll on a level base).

    Returns ``{joint: {...}}`` with ``samples``, ``torque_range_nm`` and either ``stiffness_nm_per_rad`` (N·m/rad),
    ``compliance_rad_per_nm``, ``intercept_rad``, ``residual_rms_rad`` and ``r_squared``, or ``stiffness_nm_per_rad``
    None with ``reason`` (``torque_range_below_minimum``, ``too_few_samples``, ``non_positive_compliance``)."""
    errors, torques = np.asarray(errors_rad, dtype=float), np.asarray(torques_nm, dtype=float)
    if errors.ndim != 2 or errors.shape != torques.shape or errors.shape[1] != len(ARM_JOINTS):
        raise ValueError("fit_stiffness_requires_matching_N_by_5_arrays")
    result = {}
    for index, name in enumerate(ARM_JOINTS):
        load, error = -torques[:, index], errors[:, index]
        span = float(load.max() - load.min()) if len(load) else 0.0
        entry = {"samples": int(len(load)), "torque_range_nm": round(span, 4), "stiffness_nm_per_rad": None}
        result[name] = entry
        if len(load) < 3:
            entry["reason"] = "too_few_samples"
            continue
        if span < min_torque_range_nm:
            entry["reason"] = "torque_range_below_minimum"
            continue
        design = np.column_stack([load, np.ones(len(load))])
        (compliance, intercept), *_ = np.linalg.lstsq(design, error, rcond=None)
        residual = error - design @ np.array([compliance, intercept])
        scatter = float(((error - error.mean()) ** 2).sum())
        entry.update(compliance_rad_per_nm=round(float(compliance), 6), intercept_rad=round(float(intercept), 5),
                     residual_rms_rad=round(float(np.sqrt((residual ** 2).mean())), 5),
                     r_squared=round(1.0 - float((residual ** 2).sum()) / scatter, 4) if scatter > 0 else None)
        if compliance <= 0:
            entry["reason"] = "non_positive_compliance"
            continue
        entry["stiffness_nm_per_rad"] = round(1.0 / float(compliance), 3)
    return result
