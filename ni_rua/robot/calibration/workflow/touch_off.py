"""SO-101 table touch-off: find the table height by lightly touching it with the closed fingertips.

Importable core of ``scripts/so101_touch_table.py`` and of calibration workflow steps S1 (touch-off)
and S6 (verification touches). It runs inside the worker interpreter against a live ``SO101Adapter``;
every move uses the adapter's speed cap, stop generation and bus lock. Operator-only.

Like a CNC touch-off: hover above a target ``[x, y]`` (m, so101_base), then descend in small TCP steps
pointing down. Low-gain servos sag under gravity, so the settled hover sag offset (goal - measured) stays
applied to every descent goal; in free air measured q then tracks the intended q. Contact is declared
when, after ``touch_warmup_steps``, on ``touch_confirm_reads`` consecutive reads either

* the largest |intended - measured| arm joint lag exceeds ``touch_contact_lag_rad`` (rad), or
* |Present_Load| of shoulder_lift/elbow_flex/wrist_flex rises more than ``touch_contact_load_delta``
  (raw STS3215 units) above its free-air value learned during the warm-up (0 disables).

These are obstruction candidates, not proof of table contact. Confirmation retains the same tested
goal for every joint. Optional bounded out-and-back XY motion at that goal's base-frame z records
possible sliding; it does not infer contact force or chase a lost surface. Lowest CAD vertices are
geometric hypotheses only. Independent review is required before using candidates as constraints.
Every control bound comes from the deployment ``calibration`` section (``touch_*`` fields).
"""
from __future__ import annotations

from datetime import datetime, timezone
from copy import deepcopy
import math
import sys
import threading

import numpy as np

from ni_rua.robot.backends.so101.driver import ARM_JOINTS
from .defaults import calibration_default

SCHEMA = "ni_rua.so101.touch_table.v1"
DOWN = np.diag([-1.0, 1.0, -1.0])  # TCP +Z (approach) pointing to -Z of the base at azimuth 0
LOAD_JOINTS = ("shoulder_lift", "elbow_flex", "wrist_flex")


def tcp_target(xyz):
    """4x4 TCP target at ``xyz`` (m, so101_base) with the approach axis pointing down and radially aligned."""
    x, y, z = (float(v) for v in xyz)
    psi = math.atan2(y, x) if (x or y) else 0.0
    c, s = math.cos(psi), math.sin(psi)
    T = np.eye(4)
    T[:3, 0] = [-c, -s, 0.0]
    T[:3, 1] = [-s, c, 0.0]
    T[:3, 2] = [0.0, 0.0, -1.0]
    T[:3, 3] = [x, y, z]
    return T


def _arm_load(adapter, enabled):
    """|Present_Load| of LOAD_JOINTS (raw) via ``driver.arm_load``; None when unavailable or disabled."""
    reader = getattr(adapter.driver, "arm_load", None)
    if not enabled or not callable(reader):
        return None
    try:
        with adapter.bus_lock:
            values = reader(LOAD_JOINTS)
        measured = np.abs([float(values[name]) for name in LOAD_JOINTS])
        return measured if np.all(np.isfinite(measured)) else None
    except Exception:
        return None


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def fit_table_plane(touches):
    """Fit ``z = a*x + b*y + c`` (m) across >= 3 non-collinear contact points; ``None`` when underconstrained."""
    contacts = [t for t in touches if t.get("status", "contact") == "contact" and "xy" in t and "contact_z_m" in t]
    if len(contacts) < 3:
        return None
    modeled = all(t.get("contact_position_m") is not None for t in contacts)
    pts = np.array([t["contact_position_m"] if modeled else
                    [float(t["xy"][0]), float(t["xy"][1]), float(t["contact_z_m"])] for t in contacts], dtype=float)
    xy_centered = pts[:, :2] - pts[:, :2].mean(axis=0)
    if np.linalg.svd(xy_centered, compute_uv=False)[-1] < 1e-6:
        return None
    A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
    coeffs, _, _, _ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
    a, b, c = (float(v) for v in coeffs)
    raw_normal = np.array([-a, -b, 1.0], dtype=float)
    normal = raw_normal / np.linalg.norm(raw_normal)
    tilt_rad = float(math.acos(max(-1.0, min(1.0, float(normal[2])))))
    residuals = pts[:, 2] - (A @ coeffs)
    return {
        "coordinate_source": "modeled_contact_xyz" if modeled else "legacy_requested_tcp_xy_proxy",
        "physical_plane_verified": False,
        "z_at_origin_m": c,
        "slope_x": a,
        "slope_y": b,
        "normal": [float(v) for v in normal],
        "tilt_rad": tilt_rad,
        "tilt_deg": float(math.degrees(tilt_rad)),
        "pitch_deg": float(math.degrees(math.atan2(a, 1.0))),
        "roll_deg": float(math.degrees(math.atan2(b, 1.0))),
        "residual_rms_m": float(np.sqrt(np.mean(residuals ** 2))),
        "residual_max_m": float(np.max(np.abs(residuals))),
    }


def table_summary(touches):
    """``{"mean", "min", "max", "count", "plane"}`` of contact-point heights (m) of the contacts, or None."""
    heights = [t["contact_z_m"] for t in touches if t.get("status") == "contact"]
    if not heights:
        return None
    return {"mean": float(np.mean(heights)), "min": float(np.min(heights)), "max": float(np.max(heights)),
            "count": len(heights), "plane": fit_table_plane(touches)}


def touch_points(adapter, points_xy_m, settings, *, progress=None, label="touch", on_contact=None):
    """Touch the table at each ``[x, y]`` (m) of ``points_xy_m``; returns the touch report.

    ``settings``: deployment ``calibration`` section (``touch_*`` keys). ``progress(dict)``: phase updates
    (``moving``/``descending``/``contact``/``parking``) with ``point_index``/``point_count``.
    ``on_contact(entry, q_rad)``: optional developer measurement at the current post-hold pose (rad),
    after any sliding. Its entry is a copy with current geometry; return value and its measured pose
    are stored separately under ``entry["extra"]``, never paired with the first candidate's joints.

    Report: ``{"schema", "started_at", "finished_at", "status": completed|stopped|failed|unknown, "reason",
    "planned_points", "contacts", "touches": [...], "table_z_nominal_m", "physics_advanced", "park"}``. Each
    contact entry uses the FIRST confirmed-hit sample for height, lag and trigger; ``confirmation``
    retains the final read separately (same units, reads=count). ``descent_traces`` archives every read
    with expected/measured joints, joint-read-end elapsed seconds, separate read/write intervals, lag and load.
    ``commands`` retains uncertain writes even if the next read fails. Optional ``contact_trajectories``
    are unreviewed samples, not fit constraints. Each
    contact entry has ``q_measured_rad`` (rad), ``gripper_rad``, ``contact_link``/``contact_point_in_link_m``
    (lowest gripper mesh vertex, m), ``contact_z_m`` (its height in the nominal model, m), ``fingertip_z_m``,
    ``commanded_tcp_z_m``, ``lag_rad``, ``load_rise`` and ``trigger`` (``lag`` or ``load``). After a stop,
    failed or unknown move the arm stays held where it stopped (no lift or park).
    """
    from ni_rua.robot.backends.so101.worker import DriverFault
    from ...embodiments.so101 import gripper_cad as geometry
    from ...embodiments.so101.contact import lowest_gripper_point
    progress = progress or (lambda update: None)
    kin, model, clock = adapter.kin, adapter.model, adapter.clock
    settings = dict(settings)
    settings.setdefault("touch_orientation_weight", calibration_default("touch_orientation_weight"))
    protrusion = geometry.derive_gripper_geometry()["protrusion_beyond_tcp_m"]["fixed_jaw"]
    points = [[float(v) for v in point] for point in points_xy_m]
    report = {"schema": SCHEMA, "started_at": _now(), "finished_at": None, "status": "completed", "reason": None,
              "label": label, "planned_points": len(points), "contacts": 0, "touches": [],
              "joint_mapping": deepcopy(adapter.deployment["joint_mapping"]),
              "effective_motion_limits": adapter.effective_limits(),
              "motion_segments": [],
              "fingertip_beyond_tcp_m": protrusion, "settings": {k: v for k, v in settings.items() if k.startswith("touch_")},
              "gripper_closed_rad": None, "table_z_nominal_m": None, "physics_advanced": False, "park": None}
    home = list(adapter.deployment["home"]["q_rad"])
    jaw, speed, tolerance = settings["touch_jaw_normalized"], settings["touch_speed_scale"], settings["touch_arrival_tolerance_rad"]
    orientation_weight = float(settings["touch_orientation_weight"])
    generation = adapter.stop_generation
    deadline = clock.now() + settings["touch_timeout_s"]

    def move(q, action_id, jaw_goal=None, correct_sag=True):
        started = clock.now()
        result = adapter.move_joints(q, jaw_goal, tolerance, speed_scale=speed, action_id=action_id,
                                    correct_sag=correct_sag, expected_generation=generation, deadline_s=deadline)
        report["motion_segments"].append({"action_id": action_id, "started_monotonic_s": started,
            "finished_monotonic_s": clock.now(), "q_requested_rad": list(q), "gripper_goal_normalized": jaw_goal,
            "clock_domain": "adapter_monotonic", "result": result})
        report["physics_advanced"] = report["physics_advanced"] or bool(result.get("physics_advanced"))
        if result["status"] != "completed":
            status = result["status"] if result["status"] in ("stopped", "unknown") else "failed"
            reason = "touch_timeout" if clock.now() >= deadline else f"{action_id}:{result.get('reason')}"
            report.update(status=status, reason=reason)
            if result.get("stop_confirmation") is not None:
                report["stop_confirmation"] = result["stop_confirmation"]
            return False
        return True

    def stopped():
        if adapter.stop_generation != generation:
            confirmation = _confirm_stop(adapter)
            report.update(status="stopped" if confirmation.get("confirmed") else "unknown",
                          reason=adapter.stop_reason or "operator_hold", stop_confirmation=confirmation)
            return True
        return False

    try:
        progress({"phase": "moving", "point_index": 0, "point_count": len(points), "target": "home"})
        if move(home, f"{label}_home", jaw):
            report["gripper_closed_rad"] = float(adapter.read()["gripper_rad"])
            for index, (x, y) in enumerate(points):
                if stopped():
                    break
                if clock.now() > deadline:
                    report.update(status="failed", reason="touch_timeout")
                    break
                entry = {"index": index, "xy": [x, y], "status": None}
                report["touches"].append(entry)
                sol = kin.ik(model, tcp_target([x, y, settings["touch_hover_z_m"]]), home,
                             orientation_weight=orientation_weight, limits=adapter.limits)
                if sol["position_error_m"] > settings["touch_ik_max_position_error_m"]:
                    entry.update(status="unreachable", ik_position_error_m=float(sol["position_error_m"]))
                    continue
                hover = [float(v) for v in sol["q_rad"]]
                progress({"phase": "moving", "point_index": index + 1, "point_count": len(points), "xy": [x, y]})
                if not move(hover, f"{label}_hover_{index}"):
                    entry["status"] = "hover_" + report["status"]
                    break
                contact = _descend(adapter, settings, x, y, generation, deadline, progress, index, len(points), report)
                if contact == "stopped" or adapter.stop_generation != generation:
                    if not stopped():
                        confirmation = _confirm_stop(adapter)
                        report.update(status="stopped" if confirmation.get("confirmed") else "unknown",
                                      reason="operator_hold", stop_confirmation=confirmation)
                    entry["status"] = report["status"]
                    break
                if contact == "timeout":
                    entry["status"] = "timeout"
                    if report["status"] != "unknown":
                        report.update(status="failed", reason="touch_timeout")
                    break
                if contact == "failed":
                    entry["status"] = "failed"
                    break
                if contact is None:
                    entry["status"] = "no_contact_before_min_z"
                else:
                    q = contact["q_measured_rad"]
                    low = lowest_gripper_point(model, q, contact["gripper_rad"])
                    T = kin.fk(model, q)
                    entry.update(contact, status="contact", contact_evidence="unknown",
                                 measurement_kind="obstruction_candidate", geometry_assumption="lowest_cad_vertex",
                                 tcp_measured_m=[float(v) for v in T[:3, 3]],
                                 fingertip_z_m=float(T[2, 3] + protrusion * T[2, 2]), contact_link=low["link"],
                                 contact_point_in_link_m=[float(v) for v in low["point_in_link_m"]],
                                 contact_position_m=list(low["position_m"]),
                                 contact_z_m=float(low["z_m"]))
                    report["contacts"] += 1
                    progress({"phase": "contact", "point_index": index + 1, "point_count": len(points),
                              "contact_z_m": entry["contact_z_m"], "trigger": entry["trigger"]})
                    if on_contact is not None:
                        try:
                            # A post-hold image cannot be paired with the first candidate's q.
                            current = adapter.read()
                            current_low = lowest_gripper_point(model, current["q_rad"], current["gripper_rad"])
                            extra_pose = dict(entry, q_measured_rad=list(current["q_rad"]),
                                              gripper_rad=current["gripper_rad"], contact_link=current_low["link"],
                                              contact_point_in_link_m=current_low["point_in_link_m"],
                                              contact_z_m=current_low["z_m"])
                            entry["extra"] = {"measurement_phase": "post_hold", "joint_read_monotonic_s": current["t"],
                                              "q_measured_rad": list(current["q_rad"]),
                                              "result": on_contact(extra_pose, current["q_rad"])}
                        except Exception as error:  # a developer measurement never aborts the sequence
                            entry["extra"] = {"error": f"{type(error).__name__}: {error}"}
                if not move(hover, f"{label}_lift_{index}", correct_sag=False):
                    break
            if report["status"] == "completed":
                progress({"phase": "parking", "point_count": len(points)})
                parked = move(home, f"{label}_park")
                report["park"] = "completed" if parked else report["status"]
    except DriverFault as error:
        report.update(status="unknown", reason=f"backend_error: {error}", stop_confirmation=_confirm_stop(adapter))
    except Exception as error:
        confirmation = _confirm_stop(adapter)
        report.update(status="failed" if confirmation.get("confirmed") else "unknown",
                      reason=f"touch_processing_error: {type(error).__name__}: {error}", stop_confirmation=confirmation)
    if (report.get("stop_confirmation") or {}).get("confirmed") is False:
        report["status"] = "unknown"
    for trajectory in report.get("contact_trajectories", []):
        if trajectory["status"] == "running":
            trajectory.update(status="incomplete", termination_reason=report.get("reason") or report["status"])
    report["table_z_nominal_m"] = table_summary(report["touches"])
    report["finished_at"] = _now()
    return report


def _confirm_stop(adapter):
    """Reuse measured hold confirmation under the operator lock; never hold a relaxed arm."""
    return adapter._confirm_stop()


def _guard_failure(adapter, report, trace, reason):
    """Stop via the existing measured-settling owner and retain partial trace; no lift/park on failure."""
    confirmation = _confirm_stop(adapter)
    report.update(status="failed" if confirmation.get("confirmed") else "unknown",
                  reason=reason, stop_confirmation=confirmation)
    trace["termination_reason"] = reason
    return "failed"


def _send_traced(adapter, command, generation, trace, origin):
    """Archive a goal attempt (five rad, active mapping) before I/O; origin is adapter monotonic s.

    A failed write remains unknown, never a fabricated successful send; False means suppressed by stop.
    """
    attempt = {"q_command_rad": list(command), "started_s": adapter.clock.now() - origin,
               "finished_s": None, "delivery": "unknown", "effective_motion_limits": adapter.effective_limits()}
    trace.setdefault("commands", []).append(attempt)
    try:
        sent = adapter._send(command, adapter.jaw_goal, generation)
        attempt["delivery"] = "sent" if sent else "suppressed_by_stop"
        return sent
    finally:
        attempt["finished_s"] = adapter.clock.now() - origin


def _slide(adapter, settings, x, y, z, q_anchor, sag, load_ref, load_guard_ref, generation, deadline,
           report, trace, trace_started, progress):
    """Record one bounded out-and-back XY trial after an obstruction candidate.

    x/y/z are commanded TCP coordinates (m, so101_base); q_anchor and fixed sag are rad in the
    recorded mapping. load_ref is raw unsigned servo load, never N. deadline/trace_started use
    adapter monotonic seconds. Settings supply distance, direction, step, time and guards, with
    distance=0 disabling this path at the caller. Returns None on completion or stopped/failed.
    No normal-force control or surface-following is inferred. See touch_points for an example API.
    """
    clock, kin, model = adapter.clock, adapter.kin, adapter.model
    trajectory = {"trajectory_id": f"{report.get('label', 'touch')}-{trace['point_index']}",
                  "contact_evidence": "unknown", "evidence_note": None,
                  "contact_link": None, "point_in_link_m": None,
                  "joint_mapping": deepcopy(adapter.deployment["joint_mapping"]),
                  "clock_domain": "adapter_monotonic", "status": "running", "samples": []}
    report.setdefault("contact_trajectories", []).append(trajectory)
    trajectory["samples"].append(dict(trace["reads"][-1]))

    def fail(reason):
        trajectory.update(status="incomplete", termination_reason=reason)
        return _guard_failure(adapter, report, trace, reason)

    if load_ref is None or load_guard_ref is None:
        return fail("slide_load_unavailable")
    start = clock.now()
    distance = settings["touch_slide_distance_m"]
    ratio = distance / settings["touch_slide_step_m"]
    available = min(deadline - start, settings["touch_slide_max_duration_s"])
    if not math.isfinite(ratio) or 2 * math.ceil(ratio) * settings["touch_step_period_s"] > available:
        return fail("slide_timeout")
    n = int(math.ceil(ratio))
    direction = np.array([math.cos(settings["touch_slide_direction_rad"]), math.sin(settings["touch_slide_direction_rad"])])
    weight = settings.get("touch_orientation_weight", calibration_default("touch_orientation_weight"))
    limits = _sag_safe_limits(adapter, sag)
    initial = kin.ik(model, tcp_target([x, y, z]), q_anchor.tolist(), orientation_weight=weight, limits=limits)
    if initial["position_error_m"] > settings["touch_ik_max_position_error_m"]:
        return fail("slide_ik_residual")
    q0 = np.asarray(initial["q_rad"])
    seed = q0.copy()
    for i in range(2 * n):
        along = distance * (i + 1 if i < n else 2 * n - i - 1) / n
        if adapter.stop_generation != generation:
            trajectory.update(status="incomplete", termination_reason="operator_stop")
            return "stopped"
        if clock.now() + settings["touch_step_period_s"] > min(deadline, start + settings["touch_slide_max_duration_s"]):
            return fail("slide_timeout")
        xy = np.array([x, y]) + direction * along
        step = kin.ik(model, tcp_target([*xy, z]), seed.tolist(), orientation_weight=weight, limits=limits)
        if step["position_error_m"] > settings["touch_ik_max_position_error_m"]:
            return fail("slide_ik_residual")
        seed = np.asarray(step["q_rad"])
        expected = seed
        command = expected + sag
        if any(not adapter.limits[name][0] <= v <= adapter.limits[name][1] for name, v in zip(ARM_JOINTS, command)):
            return fail("slide_joint_limit")
        speed = float(np.max(np.abs(command - np.asarray(adapter.arm_goal)))) / settings["touch_step_period_s"]
        if speed > adapter.effective_limits()["effective_max_joint_speed_rad_s"]:
            return fail("slide_command_speed_limit")
        if float(np.max(np.abs(command - np.asarray(adapter.arm_goal)))) > adapter.deployment["motion"]["max_command_delta_rad"]:
            return fail("slide_command_delta_limit")
        sent_at = clock.now()
        if not _send_traced(adapter, command.tolist(), generation, trace, trace_started):
            trajectory.update(status="incomplete", termination_reason="operator_stop")
            return "stopped"
        report["physics_advanced"] = True
        clock.sleep(settings["touch_step_period_s"])
        if adapter.stop_generation != generation:
            trajectory.update(status="incomplete", termination_reason="operator_stop")
            return "stopped"
        read_start = clock.now()
        sample = adapter.read()
        read_end = clock.now()
        load = _arm_load(adapter, True)
        load_end = clock.now()
        lag_by_joint = expected - np.asarray(sample["q_rad"])
        lag = float(np.max(np.abs(lag_by_joint)))
        rise = float(np.max(load - load_ref)) if load is not None else None
        guard_rise = float(np.max(load - load_guard_ref)) if load is not None else None
        support = lag > settings["touch_contact_lag_rad"] or (
            settings["touch_contact_load_delta"] > 0 and rise is not None and rise > settings["touch_contact_load_delta"])
        row = dict(phase="slide_out" if i < n else "slide_return", elapsed_s=read_end - trace_started,
                   command_index=len(trace["commands"]) - 1,
                   command_sent_s=sent_at - trace_started,
                   joint_read_interval_s=[read_start - trace_started, read_end - trace_started],
                   load_read_interval_s=[read_end - trace_started, load_end - trace_started],
                   commanded_tcp_xyz_m=kin.fk(model, expected)[:3, 3].tolist(),
                   ik_target_tcp_xyz_m=[float(xy[0]), float(xy[1]), float(z)],
                   q_command_rad=command.tolist(), q_expected_rad=expected.tolist(),
                   q_measured_rad=list(sample["q_rad"]), gripper_rad=float(sample["gripper_rad"]),
                   lag_by_joint_rad=lag_by_joint.tolist(), lag_rad=lag, load_rise=rise,
                   load_abs=load.tolist() if load is not None else None, load_reference=load_ref.tolist(),
                   guard_load_rise=guard_rise, load_guard_reference=load_guard_ref.tolist(),
                   obstruction_supported=bool(support), contact_evidence="unknown")
        trajectory["samples"].append(row)
        trace["reads"].append(row)
        if load is None:
            return fail("slide_load_unavailable")
        if lag > settings["touch_max_tracking_error_rad"]:
            return fail("touch_tracking_limit")
        if guard_rise > settings["touch_max_load_rise"]:
            return fail("touch_load_limit")
        if not support:
            return fail("slide_obstruction_evidence_lost")
        progress({"phase": row["phase"], "point_index": trace["point_index"] + 1,
                  "sample_count": len(trajectory["samples"]), "contact_evidence": "unknown"})
    trajectory.update(status="completed", termination_reason="bounded_path_complete")
    return None


def _sag_safe_limits(adapter, sag):
    """IK bounds (rad) for which both expected joints and sag-compensated commands stay legal."""
    return {name: (max(adapter.limits[name][0], adapter.limits[name][0] - float(d)),
                   min(adapter.limits[name][1], adapter.limits[name][1] - float(d)))
            for name, d in zip(ARM_JOINTS, sag)}


def _descend(adapter, settings, x, y, generation, deadline, progress, index, count, report):
    """Step down from the hover pose until contact; returns the contact dict, None, ``"stopped"`` or ``"timeout"``."""
    kin, model, clock = adapter.kin, adapter.model, adapter.clock
    q_start = np.asarray(adapter.read()["q_rad"], dtype=float)
    offset = np.subtract(adapter.arm_goal, q_start)  # settled hover sag compensation
    limits = _sag_safe_limits(adapter, offset)
    orientation_weight = float(settings.get("touch_orientation_weight", calibration_default("touch_orientation_weight")))
    hover_measured = kin.fk(model, q_start)
    # Start the Cartesian path at the measured hover. Adding an IK joint delta to a
    # different measured q can put the sent goal outside the very limits used by IK.
    target_x, target_y = (float(v) for v in hover_measured[:2, 3])
    q_ik = q_start.copy()
    q_expected = q_start.copy()
    pending = False
    first_hit_sample = first_hit_diagnostics = None
    load_enabled = True  # load guarding is independent of whether load triggers candidates
    load_ref = _arm_load(adapter, load_enabled)
    load_guard_ref = None if load_ref is None else load_ref.copy()  # never learn away a protective limit
    z, steps, above = float(hover_measured[2, 3]), 0, 0
    run = adapter._new_run(threading.Event(), settings["touch_timeout_s"], "touch_descent")
    contact = None
    trace = {"point_index": index, "xy_m": [float(x), float(y)],
             "schema": "ni_rua.so101.contact_trace.v2", "clock_domain": "adapter_monotonic",
             "joint_mapping": deepcopy(adapter.deployment["joint_mapping"]),
             "contact_evidence": "unknown", "first_candidate_read": None, "confirmation_read": None,
             "hover_q_measured_rad": q_start.tolist(), "sag_offset_rad": offset.tolist(),
             "units": {"elapsed_s": "s since descent start", "commanded_tcp_z_m": "m in so101_base",
                       "q_expected_rad": "rad, actual sent goal minus fixed hover sag offset", "q_measured_rad": "rad, measured joints",
                       "q_command_rad": "rad, actual sent goal including fixed hover sag compensation",
                       "gripper_rad": "rad, measured jaw", "load_abs": "raw STS3215 absolute load, LOAD_JOINTS order",
                       "load_reference": "raw STS3215 warm-up baseline, LOAD_JOINTS order",
                       "lag_by_joint_rad": "rad, expected minus measured", "lag_rad": "rad, max absolute joint lag",
                       "load_rise": "raw STS3215 load units", "consecutive_hits": "count",
                       "guard_load_rise": "raw STS3215 load units above immutable measured-hover load reference",
                       "commanded_tcp_xyz_m": "m, FK of sag-adjusted expected goal in so101_base",
                       "ik_target_tcp_xyz_m": "m, requested endpoint; not necessarily reached",
                       "joint_read_interval_s": "s since descent start; host read interval, not atomic encoder exposure",
                       "load_read_interval_s": "s since descent start; separate host load read interval"}, "reads": []}
    report.setdefault("descent_traces", []).append(trace)
    trace_started = clock.now()
    trace["started_monotonic_s"] = trace_started
    adapter.motion_phase = "moving"
    try:
        while above > 0 or pending or z - settings["touch_step_m"] >= settings["touch_min_z_m"]:
            if adapter.stop_generation != generation:
                return "stopped"
            if clock.now() + settings["touch_step_period_s"] > deadline:
                _guard_failure(adapter, report, trace, "touch_timeout")
                return "timeout"
            if above == 0:
                if not pending:
                    z -= settings["touch_step_m"]
                    steps += 1
                    step = kin.ik(model, tcp_target([target_x, target_y, z]), q_ik.tolist(), orientation_weight=orientation_weight, limits=limits)
                    if step["position_error_m"] > settings["touch_ik_max_position_error_m"]:
                        return _guard_failure(adapter, report, trace, "descent_ik_residual")
                    q_ik = np.asarray(step["q_rad"], dtype=float)
                # Resample IK endpoints at the existing physical command limits, without
                # silently changing the requested endpoint or accumulating pressure on a hit.
                delta = q_ik - q_expected
                cap = min(adapter.deployment["motion"]["max_command_delta_rad"],
                          adapter.effective_limits()["effective_max_joint_speed_rad_s"] * settings["touch_step_period_s"])
                fraction = min(1.0, cap / float(np.max(np.abs(delta)))) if np.any(delta) else 1.0
                q_expected = q_expected + fraction * delta
                pending = fraction < 1.0
            q_cmd = q_expected.copy()
            # Keep the goal being tested. Freezing a joint at its first measured lag while
            # comparing to the old goal manufactures evidence even in an unloaded servo.
            sent_at = clock.now()
            command = q_cmd + offset
            if any(not adapter.limits[name][0] <= v <= adapter.limits[name][1] for name, v in zip(ARM_JOINTS, command)):
                return _guard_failure(adapter, report, trace, "descent_joint_limit")
            speed = float(np.max(np.abs(command - np.asarray(adapter.arm_goal)))) / settings["touch_step_period_s"]
            if speed > np.nextafter(adapter.effective_limits()["effective_max_joint_speed_rad_s"], np.inf) * (1 + 1e-12):
                return _guard_failure(adapter, report, trace, "descent_command_speed_limit")
            if not _send_traced(adapter, (q_cmd + offset).tolist(), generation, trace, trace_started):
                return "stopped"
            run["commands"] += 1
            report["physics_advanced"] = True
            clock.sleep(settings["touch_step_period_s"])
            if adapter.stop_generation != generation:
                return "stopped"
            read_started = clock.now()
            sample = adapter.read()
            read_finished = clock.now()
            lag = float(np.abs(q_expected - np.asarray(sample["q_rad"], dtype=float)).max())
            load = _arm_load(adapter, load_enabled)
            load_finished = clock.now()
            if load is not None and steps <= settings["touch_warmup_steps"] and lag <= settings["touch_contact_lag_rad"]:
                if load_ref is None or float((load - load_ref).max()) <= settings["touch_contact_load_delta"]:
                    load_ref = load if load_ref is None else np.maximum(load_ref, load)
            rise = float((load - load_ref).max()) if load is not None and load_ref is not None else None
            guard_rise = float((load - load_guard_ref).max()) if load is not None and load_guard_ref is not None else None
            by_load = settings["touch_contact_load_delta"] > 0 and rise is not None and rise > settings["touch_contact_load_delta"]
            hit = by_load or (steps > settings["touch_warmup_steps"] and lag > settings["touch_contact_lag_rad"])
            diagnostics = {"lag_rad": lag, "load_rise": rise, "trigger": "load" if by_load else "lag" if hit else None}
            if hit and above == 0:
                first_hit_sample = sample
                first_hit_diagnostics = dict(diagnostics)
                if trace["first_candidate_read"] is None:
                    trace["first_candidate_read"] = len(trace["reads"])
            elif not hit:
                first_hit_sample = first_hit_diagnostics = None
            above = above + 1 if hit else 0
            expected_tcp = kin.fk(model, q_expected)[:3, 3].tolist()
            trace["reads"].append({"elapsed_s": float(read_finished - trace_started), "commanded_tcp_z_m": float(expected_tcp[2]),
                "command_index": len(trace["commands"]) - 1,
                "commanded_tcp_xyz_m": expected_tcp, "ik_target_tcp_xyz_m": [target_x, target_y, float(z)],
                "ik_endpoint_pending": pending,
                "phase": "confirm" if above > 1 else "descend", "command_sent_s": sent_at - trace_started,
                "joint_read_interval_s": [read_started - trace_started, read_finished - trace_started],
                "load_read_interval_s": [read_finished - trace_started, load_finished - trace_started],
                "contact_evidence": "unknown", "obstruction_supported": bool(hit),
                "q_expected_rad": q_expected.tolist(), "q_measured_rad": [float(v) for v in sample["q_rad"]],
                "q_command_rad": (q_cmd + offset).tolist(), "gripper_rad": float(sample["gripper_rad"]),
                "load_abs": load.tolist() if load is not None else None,
                "load_reference": load_ref.tolist() if load_ref is not None else None,
                "load_guard_reference": load_guard_ref.tolist() if load_guard_ref is not None else None,
                "guard_load_rise": guard_rise,
                "lag_by_joint_rad": (q_expected - np.asarray(sample["q_rad"], dtype=float)).tolist(),
                **diagnostics, "consecutive_hits": above})
            if lag > settings["touch_max_tracking_error_rad"]:
                return _guard_failure(adapter, report, trace, "touch_tracking_limit")
            if guard_rise is not None and guard_rise > settings["touch_max_load_rise"]:
                return _guard_failure(adapter, report, trace, "touch_load_limit")
            if steps % 5 == 0:
                progress({"phase": "descending", "point_index": index + 1, "point_count": count,
                          "commanded_tcp_z_m": round(expected_tcp[2], 4), "lag_rad": round(lag, 4)})
            if above >= settings["touch_confirm_reads"]:
                trace["confirmation_read"] = len(trace["reads"]) - 1
                rec = first_hit_sample if first_hit_sample is not None else sample
                contact = {"commanded_tcp_z_m": float(expected_tcp[2]), **(first_hit_diagnostics or diagnostics),
                           "confirmation": {**diagnostics, "q_measured_rad": [float(v) for v in sample["q_rad"]],
                                            "gripper_rad": float(sample["gripper_rad"]), "reads": above},
                           "descent_steps": steps,
                           "q_measured_rad": [float(v) for v in rec["q_rad"]], "gripper_rad": float(rec["gripper_rad"])}
                if settings["touch_slide_distance_m"] > 0:
                    outcome = _slide(adapter, settings, *expected_tcp, q_expected, offset, load_ref, load_guard_ref,
                                     generation, deadline, report, trace, trace_started, progress)
                    if outcome is not None:
                        return outcome
                break
        measured = adapter.read()["q_rad"]
        if not _send_traced(adapter, (np.asarray(measured) + offset).tolist(), generation, trace, trace_started):
            return "stopped"
        clock.sleep(settings["touch_settle_s"])
        if adapter.stop_generation != generation:
            return "stopped"
        if clock.now() > deadline:
            _guard_failure(adapter, report, trace, "touch_timeout")
            return "timeout"
        read_start = clock.now()
        terminal = adapter.read()
        trace["terminal_hold"] = {"phase": "post_hold", "elapsed_s": clock.now() - trace_started,
            "joint_read_interval_s": [read_start - trace_started, clock.now() - trace_started],
            "q_measured_rad": list(terminal["q_rad"]), "gripper_rad": float(terminal["gripper_rad"]),
            "contact_evidence": "unknown"}
    finally:
        if adapter.motion_phase == "moving":
            adapter.motion_phase = "holding"
    return contact
