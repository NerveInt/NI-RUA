"""SO-101 camera calibration capture and one-click recalibration (operator-only).

Runs inside the worker interpreter (``so101`` env) against a live :class:`SO101Adapter`; every arm
motion goes through ``adapter.move_joints`` (same speed cap, tracking/stall guards and sag
correction as ``robot_action``). Nothing here is model-visible: progress, sessions, reports and the
calibration history are operator/audit records (docs/development/so101-integration.md section 6.4).

Flow of :func:`run_calibration` (operator command ``calibrate``):

1. **Board check, no motion.** Fresh frames of every camera must show at least
   ``calibration.min_board_corners`` ChArUco corners; otherwise ``status="board_not_visible"`` with
   per-camera counts and the arm never moves.
2. **Capture.** Visit ``home.q_rad`` plus the joint-offset grid of :func:`pose_grid`; after measured
   settling keep a sample when the wrist camera sees ``sample_min_wrist_corners``. The session
   directory (``session.json``, ``samples.jsonl``, ``images/``) has the format of
   ``scripts/so101_calibrate.py solve``. The arm parks at home unless a move stopped, failed or is
   unknown (then it stays held where the move ended).
3. **Solve + gates.** :func:`ni_rua.robot.calibration.workflow.hand_eye_solve.solve_session` writes a new
   timestamped file next to the active one, only when every residual gate passes.
4. **Install.** Only after a pass: the candidate is validated exactly like a reload, the previous
   active file is copied to ``<stem>-previous-<UTC>.json``, the active path is atomically replaced
   (``os.replace``) and the adapter reloads it. On any failure the old calibration stays active.
"""
from __future__ import annotations

from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import shutil
import threading
import uuid

import numpy as np

from ni_rua.robot.backends.so101.driver import ARM_JOINTS
from .defaults import calibration_default

SESSION_SCHEMA = "ni_rua.so101.calibration_session.v1"


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def pose_grid(ready, pan, flex, roll, lift, lift_elbow_coupling=None):
    """Joint-offset grid around ``ready`` (5 arm joints, rad).

    ``pan``/``flex``/``roll``: offsets (rad) of shoulder_pan, wrist_flex and wrist_roll; every
    combination is one pose. ``lift``: extra poses with shoulder_lift + d and
    elbow_flex - ``lift_elbow_coupling`` * d (changes height/distance). Returns a list of 5-lists.
    ``lift_elbow_coupling``: dimensionless counter-rotation ratio; None reads the packaged legacy
    default 0.6. Production passes its validated deployment value explicitly (range 0..2).
    Example: ``pose_grid([0, -.95, .9, 1, 0], [-.2, .2], [0], [-.35, .35], [.12])`` -> 5 poses.
    """
    if lift_elbow_coupling is None:
        lift_elbow_coupling = calibration_default("lift_elbow_coupling")
    poses = []
    for dp, df, dr in itertools.product(pan, flex, roll):
        q = [float(v) for v in ready]
        q[0] += dp
        q[3] += df
        q[4] += dr
        poses.append(q)
    for dl in lift:
        q = [float(v) for v in ready]
        q[1] += dl
        q[2] -= dl * lift_elbow_coupling
        poses.append(q)
    return poses


def calibration_poses(deployment):
    """Planned poses of the one-click calibration: home first, then the deployment grid."""
    c, ready = deployment["calibration"], list(deployment["home"]["q_rad"])
    return [ready] + pose_grid(ready, c["pan_offsets_rad"], c["flex_offsets_rad"], c["roll_offsets_rad"],
                               c["lift_offsets_rad"], c["lift_elbow_coupling"])


def count_corners(image_rgb, board):
    """Number of ChArUco inner corners detected in one RGB uint8 image (0 if none/degenerate)."""
    import cv2
    from . import charuco as cal
    image = np.asarray(image_rgb, dtype=np.uint8)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
    detection = cal.detect_charuco(bgr, board, min_corners=4)
    return 0 if detection is None else int(len(detection["ids"]))


def _joint_mapping(deployment):
    from . import charuco as cal
    return cal.normalize_joint_mapping({name: {"sign": deployment["joint_mapping"][name]["sign"],
                                               "offset_rad": deployment["joint_mapping"][name]["offset_rad"]}
                                        for name in ARM_JOINTS})


def _camera_devices(adapter):
    """Physical device each camera was bound to (camera_devices.binding_summary), or None for test fakes."""
    from ni_rua.robot.backends.so101.camera_devices import binding_summary
    bindings = (getattr(adapter, "camera_binding", None) or {}).get("cameras") or {}
    return {name: binding_summary(binding) for name, binding in bindings.items()} or None


def _pose_summary(index, result):
    return {"index": index, "status": result.get("status"), "reason": result.get("reason"),
            "max_joint_error_rad": result.get("max_joint_error_rad")}


def capture_session(adapter, session_dir, *, settings=None, poses=None, progress=None, board_spec=None):
    """Visit ``poses`` (default :func:`calibration_poses`) and save samples into a fresh ``session_dir``.

    ``settings``: the deployment ``calibration`` section (or a dict with the same keys).
    ``board_spec``: ChArUco board spec (``so101_calibration.validate_board_spec``); default the packaged board.
    ``progress(dict)``: optional callback per phase. Checks the original stop generation and absolute ``capture_timeout_s`` deadline before and after
    every move, settle, capture, progress callback and park. No physical retry occurs after failure. Returns
    ``{"status": completed|stopped|failed|unknown, "reason", "kept", "pose_count", "poses": [...],
    "park": {...}|None, "physics_advanced", "stop_confirmation"?, "evidence_persisted"}``.
    Sample timing fields use adapter monotonic seconds: joint-before/after, host capture interval and
    per-camera host stamps/read intervals. Unknown exposure times stay null; camera reads are not synchronous.
    """
    import cv2
    from . import charuco as cal
    from .hand_eye_solve import source_revision
    settings = dict(settings or adapter.deployment["calibration"])
    drift_limit_rad = settings.get("capture_max_drift_rad")
    if drift_limit_rad is None:
        drift_limit_rad = settings["arrival_tolerance_rad"] * calibration_default("capture_legacy_drift_multiplier")
    drift_limit_rad = float(drift_limit_rad)
    generation = adapter.stop_generation
    deadline = adapter.clock.now() + settings["capture_timeout_s"]
    progress = progress or (lambda update: None)
    session_dir = Path(session_dir)
    (session_dir / "images").mkdir(parents=True, exist_ok=False)
    raw_board = board_spec or adapter.deployment.get("board") or cal.DEFAULT_BOARD_SPEC
    board_spec = cal.validate_board_spec({k: v for k, v in raw_board.items() if k != "thickness_m"})
    board = cal.make_board(board_spec)
    mapping = _joint_mapping(adapter.deployment)
    poses = calibration_poses(adapter.deployment) if poses is None else [list(q) for q in poses]
    session = {"schema": SESSION_SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
               "mode": "auto_grid_worker", "board": board_spec, "joint_mapping": mapping,
               "cameras": adapter.deployment["cameras"], "opencv_version": cv2.__version__,
               "camera_devices": _camera_devices(adapter),
               "settings": dict(settings), "effective_capture_max_drift_rad": drift_limit_rad,
               "planned_q_rad": poses, "source": source_revision()}
    (session_dir / "session.json").write_text(json.dumps(session, indent=2))
    start_step = adapter.control_step
    motion_attempted = False
    out = {"status": "completed", "reason": None, "kept": 0, "pose_count": len(poses), "poses": [], "park": None,
           "physics_advanced": False, "evidence_persisted": True,
           "sequence_generation": generation, "deadline_monotonic_s": deadline}

    def fail(reason, *, operator_stop=False, uncertain=False, confirmation=None):
        from .touch_off import _confirm_stop
        confirmation = _confirm_stop(adapter) if confirmation is None else confirmation
        out.update(status="unknown" if uncertain or not confirmation.get("confirmed") else
                   "stopped" if operator_stop else "failed", reason=reason, stop_confirmation=confirmation)
        out["physics_advanced"] = out["physics_advanced"] or adapter.control_step != start_step

    def boundary():
        if adapter.stop_generation != generation:
            fail(adapter.stop_reason or "operator_hold", operator_stop=True)
            return False
        if adapter.clock.now() >= deadline:
            fail("capture_timeout")
            return False
        return True

    def move(q, action_id):
        nonlocal motion_attempted
        motion_attempted = True
        result = adapter.move_joints(q, None, settings["arrival_tolerance_rad"], speed_scale=settings["speed_scale"],
                                    action_id=action_id, expected_generation=generation, deadline_s=deadline)
        out["physics_advanced"] = out["physics_advanced"] or bool(result.get("physics_advanced"))
        if result["status"] != "completed":
            fail(result.get("reason") or "move_failed", operator_stop=result["status"] == "stopped",
                 uncertain=result["status"] == "unknown", confirmation=result.get("stop_confirmation"))
        return result

    try:
        for index, q in enumerate(poses):
            if not boundary():
                break
            if any(not adapter.limits[name][0] <= value <= adapter.limits[name][1] for name, value in zip(ARM_JOINTS, q)):
                out["poses"].append({"index": index, "status": "skipped", "reason": "outside_joint_limits_with_margin"})
                continue
            if settings.get("z_floor_m") is not None:
                violation = cal.path_floor_violation(adapter.kin, adapter.model, adapter.read()["q_rad"], q,
                                                     float(settings["z_floor_m"]))
                if violation is not None:
                    out["poses"].append({"index": index, "status": "skipped", "reason": "below_z_floor",
                                         "violation": violation})
                    continue
            progress({"phase": "moving", "pose_index": index + 1, "pose_count": len(poses), "kept_count": out["kept"]})
            if not boundary():
                break
            result = move(q, f"calibration_pose_{index}")
            summary = _pose_summary(index, result)
            out["poses"].append(summary)
            if out["status"] != "completed" or not boundary():
                break
            settle_run = adapter._new_run(threading.Event(), settings["settle_timeout_s"] * 2, "calibration_settle")
            settle_run.update(generation=generation, deadline=min(deadline, settle_run["deadline"]))
            settled, before, settle_reason = adapter._settle(settle_run, settings["settle_timeout_s"])
            if not boundary():
                break
            if not settled:
                summary.update(settled=False, kept=False, reason="arm_not_settled")
                fail(str(settle_reason or "settle_unconfirmed"))
                break
            capture_started = adapter.clock.now()
            frames = adapter.capture_frames(before["t"])
            capture_finished = adapter.clock.now()
            if not boundary():
                break
            after = adapter.read()
            if not boundary():
                break
            max_drift_rad = float(np.abs(np.subtract(after["q_rad"], before["q_rad"])).max())
            drift_ok = max_drift_rad <= drift_limit_rad
            detected = {name: count_corners(image, board) for name, image in frames.items()}
            if not boundary():
                break
            corners_ok = detected.get("wrist", 0) >= settings["sample_min_wrist_corners"]
            keep = drift_ok and corners_ok
            summary.update(settled=True, max_drift_rad=max_drift_rad, effective_capture_max_drift_rad=drift_limit_rad,
                           corners=detected, kept=keep)
            if keep:
                images = {}
                for name, image in frames.items():
                    relative = f"images/{index:04d}_{name}.png"
                    if not cv2.imwrite(str(session_dir / relative), cv2.cvtColor(np.asarray(image, dtype=np.uint8),
                                                                                  cv2.COLOR_RGB2BGR)):
                        raise OSError(f"failed to write {relative}")
                    images[name] = relative
                if not boundary():
                    summary.update(kept=False, reason="interrupted_before_sample_commit")
                    break
                record = {"index": index, "captured_at": datetime.now(timezone.utc).isoformat(),
                          "joints_deg": cal.urdf_rad_to_lerobot_deg(before["q_rad"], mapping),
                          "joints_deg_after": cal.urdf_rad_to_lerobot_deg(after["q_rad"], mapping),
                          "joint_before_monotonic_s": before["t"], "joint_after_monotonic_s": after["t"],
                          "max_endpoint_drift_rad": max_drift_rad, "effective_capture_max_drift_rad": drift_limit_rad,
                          "capture_host_interval_s": [capture_started, capture_finished],
                          "camera_timing": getattr(adapter, "last_capture_timing", {}),
                          "capture_alignment": "Separate host-timed camera reads between joint samples; exposure latency unknown. Endpoint drift does not prove continuous stillness.",
                          "settled": True, "images": images, "planned_q_rad": [float(v) for v in q],
                          "detected_corners_at_capture": detected}
                with open(session_dir / "samples.jsonl", "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                out["kept"] += 1
            elif not drift_ok:
                summary["reason"] = "joint_drift_during_capture"
            else:
                summary["reason"] = "wrist_corners_below_sample_min"
            progress({"phase": "sample", "pose_index": index + 1, "pose_count": len(poses), "corners": detected,
                      "kept": keep, "kept_count": out["kept"], "settled": True})
            if not boundary():
                break
        if out["status"] == "completed" and boundary():
            progress({"phase": "parking", "kept_count": out["kept"]})
            if boundary():
                park = move(list(adapter.deployment["home"]["q_rad"]), "calibration_park")
                out["park"] = _pose_summary("park", park)
                if out["status"] == "completed":
                    boundary()
    except Exception as error:
        from ni_rua.robot.backends.so101.worker import DriverFault
        out["physics_advanced"] = out["physics_advanced"] or adapter.control_step != start_step
        if isinstance(error, DriverFault) and motion_attempted:
            # A failed write may have reached the bus without a counter increment.
            # Preserve uncertainty separately; an attempted call is not proof of motion.
            out["execution_uncertain"] = True
        fail(f"capture_error: {type(error).__name__}: {error}", uncertain=isinstance(error, DriverFault))
    out["physics_advanced"] = out["physics_advanced"] or adapter.control_step != start_step
    try:
        (session_dir / "capture.json").write_text(json.dumps(out, indent=2, default=str))
    except Exception as error:
        # Persistence failure never erases a physical receipt or its stop uncertainty.
        out.update(evidence_persisted=False, archive_error=f"{type(error).__name__}: {error}")
        if out["status"] == "completed":
            out.update(status="failed", reason="capture_archive_failed")
    return out


def solve_for_adapter(adapter, session_dir, output, settings):
    """Default solve step: gated offline solve with the adapter's kinematics; returns the report."""
    from . import charuco as cal
    from .hand_eye_solve import solve_session
    thresholds = {key: settings[key] for key in ("max_intrinsics_rms_px", "max_hand_eye_translation_m",
                                                 "max_hand_eye_rotation_rad", "max_fixed_reprojection_px")}
    fixed_K, fixed_dist, fixed_rms = None, None, None
    if adapter.calibration_path and Path(adapter.calibration_path).is_file():
        try:
            active_doc = cal.read_calibration_file(adapter.calibration_path)
            active_fixed = (active_doc.get("cameras") or {}).get("fixed")
            cfg_fixed = (adapter.deployment.get("cameras") or {}).get("fixed")
            if (active_fixed is not None and active_fixed.get("intrinsics_source") == "handheld"
                    and cfg_fixed is not None
                    and list(active_fixed.get("image_size") or []) == [int(cfg_fixed["width"]), int(cfg_fixed["height"])]):
                fixed_K = active_fixed["K"]
                fixed_dist = active_fixed["distortion"]
                fixed_rms = active_fixed.get("intrinsics_rms_px")
        except Exception:
            pass
    link = adapter.kin.WRIST_CAMERA_PARENT_LINK
    return solve_session(session_dir, output=output, thresholds=thresholds,
                         max_pnp_reprojection_px=settings["max_pnp_reprojection_px"],
                         hand_eye_score_lever_m=settings["hand_eye_score_lever_m"],
                         fixed_intrinsics_from_wrist=True,
                         fixed_intrinsics=fixed_K, fixed_distortion=fixed_dist,
                         fixed_intrinsics_rms_px=fixed_rms,
                         fk_link=lambda q: np.asarray(adapter.kin.fk(adapter.model, list(q), link=link)))


def report_residuals(report):
    """Operator summary of a solve report: px, m, rad; ``None`` where the solve did not get that far."""
    report = report or {}
    cameras = report.get("cameras") or {}
    hand_eye, fixed = report.get("hand_eye") or {}, report.get("fixed") or {}
    return {"wrist_intrinsics_rms_px": (cameras.get("wrist") or {}).get("intrinsics_rms_px"),
            "fixed_intrinsics_rms_px": (cameras.get("fixed") or {}).get("intrinsics_rms_px"),
            "fixed_intrinsics_source": (cameras.get("fixed") or {}).get("intrinsics_source"),
            "hand_eye_translation_rms_m": hand_eye.get("translation_rms_m"),
            "hand_eye_rotation_rms_rad": hand_eye.get("rotation_rms_rad"),
            "hand_eye_samples": hand_eye.get("samples"),
            "fixed_chain_reprojection_px": fixed.get("chain_reprojection_px"),
            "skipped_samples": report.get("skipped")}


def _carry_joint_calibration(candidate, active):
    """Keep the touch-off record (table height, joint offsets) of the active file in a new candidate.

    Camera recalibration does not re-measure the table; dropping the record would silently remove the
    measured table height from the model's instructions. A candidate that carries its own record wins.
    """
    if not Path(active).exists():
        return
    previous = json.loads(Path(active).read_text()).get("joint_calibration")
    document = json.loads(Path(candidate).read_text())
    if previous is None or "joint_calibration" in document:
        return
    document["joint_calibration"] = previous
    temporary = Path(candidate).with_name(f".{Path(candidate).name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(document, indent=2) + "\n")
        os.replace(temporary, candidate)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _install(adapter, candidate, active, stamp):
    """Atomically make ``candidate`` the active file and reload; restore the previous file on failure."""
    _carry_joint_calibration(candidate, active)
    adapter.validate_calibration_file(candidate)  # same checks as reload, before touching the active file
    backup = None
    if active.exists():
        backup = active.with_name(f"{active.stem}-previous-{stamp}{active.suffix}")
        shutil.copy2(active, backup)
    workflow = getattr(adapter, "workflow", None)
    bundle = getattr(workflow, "bundle", None)
    prev_bundle_active, new_bundle_version = None, None
    if bundle is not None:
        try:
            prev_bundle_active = bundle.active_version()
            offsets = {name: float(adapter.deployment["joint_mapping"][name]["offset_rad"]) for name in ARM_JOINTS}
            if prev_bundle_active is None and backup is not None and backup.is_file():
                prev_doc = json.loads(backup.read_text(encoding="utf-8"))
                prev_joint = prev_doc.get("joint_calibration")
                prev_bundle_active = bundle.create_version(
                    camera_file=backup, stamp=f"{stamp}-imported",
                    manifest={"source": "imported_existing_file", "mode": None, "changes": [], "steps": {},
                              "calibration_created_at": prev_doc.get("created_at"), "joint_offsets_rad": dict(offsets),
                              "table_z_m": (prev_joint or {}).get("table_z_m"), "previous": None},
                    files={"joint_calibration.json": prev_joint} if prev_joint else None)
                bundle.set_active(prev_bundle_active)
            doc = json.loads(Path(candidate).read_text(encoding="utf-8"))
            joint = doc.get("joint_calibration")
            # This installation observes the effective mapping now. The imported backup above
            # deliberately has no full mapping: current signs are not evidence about its history.
            new_bundle_version = bundle.create_version(
                camera_file=candidate, stamp=stamp,
                manifest={"source": "calibrate", "mode": None, "changes": [], "steps": {},
                          "calibration_created_at": doc.get("created_at"), "joint_offsets_rad": dict(offsets),
                          "joint_mapping": {name: dict(adapter.deployment["joint_mapping"][name]) for name in ARM_JOINTS},
                          "table_z_m": (joint or {}).get("table_z_m"), "previous": prev_bundle_active},
                files={"joint_calibration.json": joint} if joint else None)
            bundle.set_active(new_bundle_version)
        except Exception:
            pass
    temporary = active.with_name(f".{active.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(candidate, temporary)
        os.replace(temporary, active)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    reload = adapter.reload_calibration()
    if reload["status"] != "reloaded":
        if bundle is not None and new_bundle_version is not None:
            try:
                if prev_bundle_active is not None:
                    bundle.set_active(prev_bundle_active)
                else:
                    (bundle.root / "active").unlink(missing_ok=True)
            except Exception:
                pass
        if backup is not None:
            temporary = active.with_name(f".{active.name}.{uuid.uuid4().hex}.tmp")
            try:
                shutil.copy2(backup, temporary)
                os.replace(temporary, active)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        else:
            active.unlink(missing_ok=True)
        raise ValueError(f"calibration_reload_failed_after_install: {reload.get('reason')}")
    return reload, backup


def run_calibration(adapter, *, progress=None, capture=capture_session, solve=solve_for_adapter):
    """One-click recalibration. ``capture``/``solve`` are injectable for CPU tests.

    Returns an operator receipt. ``status``: ``calibrated`` (new file active), ``board_not_visible``
    (no motion), ``rejected``, ``stopped``/``failed``/``unknown`` (capture motion), ``insufficient_samples``,
    ``gate_failed``, ``solve_failed`` or ``install_failed``. Every status except ``calibrated`` keeps the
    previous calibration active.
    """
    from . import charuco as cal
    progress = progress or (lambda update: None)
    settings = adapter.deployment["calibration"]
    previous = adapter.calibration_identity
    base = {"command": "calibrate", "physics_advanced": False, "previous": previous,
            "min_board_corners": settings["min_board_corners"]}
    if not adapter.calibration_path:
        return dict(base, status="rejected", reason="camera_calibration_file_not_configured")
    gate = adapter._gate()
    if gate:
        return dict(base, status="rejected", reason=gate)
    check = adapter.board_check(fresh=True)
    progress({"phase": "board_check", "corners": check["corners"], "min_board_corners": check["min_board_corners"]})
    if not check["ok"]:
        return dict(base, status="board_not_visible", corners=check["corners"], reason=check.get("reason"))
    stamp = utc_stamp()
    active = Path(adapter.calibration_path)
    session_dir = adapter.run_dir / "calibration" / f"session-{stamp}-{uuid.uuid4().hex[:6]}"
    result = dict(base, corners=check["corners"], session_dir=str(session_dir))
    captured = capture(adapter, session_dir, settings=settings, progress=progress)
    result.update(physics_advanced=bool(captured.get("physics_advanced")),
                  capture={key: captured.get(key) for key in ("status", "reason", "kept", "pose_count", "poses", "park")})
    if captured.get("stop_confirmation") is not None:
        result["stop_confirmation"] = captured["stop_confirmation"]
    if captured["status"] != "completed":
        return dict(result, status=captured["status"], reason=captured.get("reason"))
    if captured["kept"] < settings["min_samples"]:
        return dict(result, status="insufficient_samples",
                    reason=f"kept {captured['kept']} samples < calibration.min_samples={settings['min_samples']}")
    progress({"phase": "solving", "kept_count": captured["kept"]})
    candidate = active.with_name(f"{active.stem}-{stamp}{active.suffix}")
    try:
        report = solve(adapter, session_dir, candidate, settings)
    except cal.CalibrationGateError as error:
        report = _read_report(session_dir)
        return dict(result, status="gate_failed", reason=str(error), gate_failures=report.get("gate_failures"),
                    residuals=report_residuals(report), thresholds=report.get("thresholds"))
    except Exception as error:
        return dict(result, status="solve_failed", reason=f"{type(error).__name__}: {error}",
                    residuals=report_residuals(_read_report(session_dir)))
    result.update(residuals=report_residuals(report), thresholds=report.get("thresholds"),
                  gate_failures=report.get("gate_failures"), candidate_file=str(candidate))
    progress({"phase": "installing"})
    try:
        reload, backup = _install(adapter, candidate, active, stamp)
    except Exception as error:
        return dict(result, status="install_failed", reason=f"{type(error).__name__}: {error}",
                    calibration=adapter.calibration_identity)
    sidecar_tmp = None
    try:  # Lets a later hand-held fixed-intrinsics solve re-solve T_base_camera (operator record only).
        sidecar = robot_session_sidecar(active)
        sidecar_tmp = sidecar.with_name(f".{sidecar.name}.{uuid.uuid4().hex}.tmp")
        sidecar_tmp.write_text(json.dumps(_plain({"session_dir": str(session_dir), "T_base_board": np.asarray(
            report["T_base_board"], float).tolist(), "joint_mapping": _joint_mapping(adapter.deployment),
            "calibrated_at": (reload.get("calibration") or {}).get("created_at")}), indent=2))
        os.replace(sidecar_tmp, sidecar)
    except (KeyError, TypeError, ValueError, OSError):
        if sidecar_tmp is not None:
            sidecar_tmp.unlink(missing_ok=True)
        robot_session_sidecar(active).unlink(missing_ok=True)  # never leave a record of an older session
    except BaseException:
        if sidecar_tmp is not None:
            sidecar_tmp.unlink(missing_ok=True)
        raise
    return dict(result, status="calibrated", reason=None, calibration=reload["calibration"],
                previous_file_backup=None if backup is None else str(backup), active_file=str(active))


def _read_report(session_dir):
    try:
        return json.loads((Path(session_dir) / "solve_report.json").read_text())
    except (OSError, ValueError):
        return {}


# ----------------------------------------------------------------------------------------------
# Hand-held intrinsics capture: still-and-novel view selection (pure, no camera access)
# ----------------------------------------------------------------------------------------------

GRID_CELL_NAMES = (("画面左上角", "画面上方中间", "画面右上角"),
                   ("画面左侧中间", "画面中央", "画面右侧中间"),
                   ("画面左下角", "画面下方中间", "画面右下角"))
DISTANCE_BUCKETS = ("far", "mid", "near")
TILT_BUCKETS = ("frontal", "small", "large")


def _board_reference_projection(detection, board):
    """Project fixed inner corners and outline (px) from detected ID/pixel correspondences.

    ``detection`` contains ``ids`` (board corner indices) and ``corners`` (N, 2 pixels);
    ``board`` supplies the corresponding planar points in metres. Returns two arrays
    ``(all_inner_corners_px, outline_px)`` using one fitted homography. No intrinsics are
    assumed; this is a view-selection approximation, not a distortion/pose measurement.
    Degenerate correspondences or a horizon crossing the board raise ``ValueError``.
    Example: ``reference, outline = _board_reference_projection(detection, board)``.
    """
    import cv2
    from . import charuco as cal
    ids = np.asarray(detection["ids"]).reshape(-1)
    pixels = np.asarray(detection["corners"], float).reshape(-1, 2)
    reference = cal.board_corner_points(board)[:, :2]
    if (len(ids) < 4 or len(ids) != len(pixels) or len(np.unique(ids)) != len(ids)
            or not np.all(np.isfinite(pixels)) or not np.all(np.isfinite(ids))
            or np.any(ids != ids.astype(int)) or np.any(ids < 0) or np.any(ids >= len(reference))):
        raise ValueError("invalid_board_correspondences")
    plane = reference[ids.astype(int)]
    if np.linalg.matrix_rank(plane - plane.mean(axis=0)) < 2 or np.linalg.matrix_rank(pixels - pixels.mean(axis=0)) < 2:
        raise ValueError("degenerate_board_projection")
    H, _ = cv2.findHomography(plane, pixels)
    if H is None or not np.all(np.isfinite(H)) or np.linalg.matrix_rank(H) < 3:
        raise ValueError("degenerate_board_projection")
    outline = cal.board_outline_points(board)[:, :2]
    points = np.vstack([reference, outline])
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ H.T
    denominator = homogeneous[:, 2]
    # A finite planar board must lie on one side of the homography's horizon.
    # This checks projective validity only; it introduces no pose/novelty threshold.
    if not (np.all(denominator > 0) or np.all(denominator < 0)):
        raise ValueError("board_projection_crosses_horizon")
    projected = homogeneous[:, :2] / denominator[:, None]
    if not np.all(np.isfinite(projected)):
        raise ValueError("nonfinite_board_projection")
    return projected[:-4], projected[-4:]


def view_signature(detection, image_size, *, board=None):
    """Board view descriptor for novelty (dimensionless, 6 values).

    ``detection``: ``{"corners": (N, 2) px, "ids": (N,)}``; ``image_size``: ``(width, height)`` px.
    ``board``: known ChArUco board, used to project its complete fixed corner set so changing detected
    IDs alone cannot create novelty. Default ``None`` retains the legacy detected-cloud descriptor
    for callers without board geometry; production capture always supplies a board.
    Returns ``[cx, cy, 2*sx, 2*sy, 0.3*cos(2a), 0.3*sin(2a)]``: reference-corner centroid and spread
    normalised by image size, and in-plane orientation ``a`` (rad, folded to 180 deg symmetry).
    Example: ``view_signature(detection, (640, 480), board=board)``. Invalid projections raise ValueError.
    """
    corners = (np.asarray(detection["corners"], float) if board is None else
               _board_reference_projection(detection, board)[0])
    w, h = image_size
    centre = corners.mean(axis=0) / [w, h]
    spread = corners.std(axis=0) / [w, h]
    cov = np.cov((corners - corners.mean(axis=0)).T)
    angle = 0.5 * np.arctan2(2 * cov[0, 1], cov[0, 0] - cov[1, 1])
    return np.array([*centre, *spread * 2, np.cos(2 * angle) * 0.3, np.sin(2 * angle) * 0.3])


class HandheldViewSelector:
    """Keeps a hand-held board view only when it is held still and differs from every kept view.

    Pure bookkeeping: feed one detection per camera frame to :meth:`offer`; it never touches a camera.

    Args:
        target_views: number of kept views at which capture counts as complete (default 20).
        min_corners: ChArUco corners two consecutive frames must share to measure motion (default 10).
        still_px: mean corner motion between consecutive frames below which the board counts as still
            (px, default 1.5).
        min_novelty: minimum :func:`view_signature` distance to every kept view (dimensionless, default 0.08;
            ~8 % of the image size in centroid/spread).
        near_fraction / far_fraction: projected full inner-corner bounding-box extent as a fraction of the image (max of width and
            height ratios) above which a view counts as ``near`` / below which as ``far`` (default 0.55 / 0.3).
        tilt_small / tilt_large: foreshortening ratio of opposite board edges (>= 1, 1 = frontal) at which a
            view counts as ``small`` / ``large`` tilt (default 1.06 / 1.15; ~15 / ~35 deg at 0.4 m). Needs
            ``board`` (``cv2.aruco.CharucoBoard``); without it tilt is ``None``.
        min_kept: kept views needed before solving is offered (default 8, the intrinsics minimum).
        still_frames: consecutive detections in the stillness window (count, default 2; deployment
            range 2..100). E.g. 3 requires two quiet transitions.
        min_sharpness: lower variance bound of a full grayscale frame's discrete Laplacian response
            (squared 8-bit gray levels, default 0 disables). A positive value requires a finite
            ``sharpness`` supplied to offer; production measures OpenCV Laplacian with ksize=1.
            E.g. 50 rejects variance below 50; this is camera/exposure dependent.

    None selects the corresponding entry in calibration_defaults.json. Production supplies every
    setting explicitly; the standalone min_corners=10/min_kept=8 defaults retain historical behavior.

    Example::

        selector = HandheldViewSelector(board=board)
        verdict = selector.offer(cal.detect_charuco(frame, board, min_corners=10), (640, 480))
        if verdict["kept"]: save(frame)
        selector.guidance()  # ["还缺：画面左上角", "倾斜一点"]
    """

    def __init__(self, *, target_views=None, min_corners=None, still_px=None, min_novelty=None, board=None,
                 near_fraction=None, far_fraction=None, tilt_small=None, tilt_large=None, min_kept=None,
                 still_frames=None, min_sharpness=None):
        def resolved(value, name):
            return calibration_default(name) if value is None else value
        target_views = resolved(target_views, "handheld_target_views")
        min_corners = resolved(min_corners, "selector_min_corners")
        still_px = resolved(still_px, "handheld_still_px")
        min_novelty = resolved(min_novelty, "handheld_min_novelty")
        near_fraction = resolved(near_fraction, "handheld_near_fraction")
        far_fraction = resolved(far_fraction, "handheld_far_fraction")
        tilt_small = resolved(tilt_small, "handheld_tilt_small")
        tilt_large = resolved(tilt_large, "handheld_tilt_large")
        min_kept = resolved(min_kept, "selector_min_kept")
        still_frames = resolved(still_frames, "handheld_still_frames")
        min_sharpness = resolved(min_sharpness, "handheld_min_sharpness")
        self.target_views, self.min_corners, self.still_px = int(target_views), int(min_corners), float(still_px)
        self.min_novelty, self.board, self.min_kept = float(min_novelty), board, int(min_kept)
        self.near_fraction, self.far_fraction = float(near_fraction), float(far_fraction)
        self.tilt_small, self.tilt_large = float(tilt_small), float(tilt_large)
        self.still_frames, self.min_sharpness = max(2, int(still_frames)), float(min_sharpness)
        self.kept, self.previous, self.last = [], None, None
        self._still_streak = 0
        self.grid = [[0] * 3 for _ in range(3)]
        self.distance = dict.fromkeys(DISTANCE_BUCKETS, 0)
        self.tilt = dict.fromkeys(TILT_BUCKETS, 0)
        self.frames = 0

    def _tilt_ratio(self, outline):
        if outline is None:
            return None
        edges = [np.linalg.norm(outline[(i + 1) % 4] - outline[i]) for i in range(4)]
        if min(edges) <= 0:
            return None
        return float(max(max(edges[0], edges[2]) / min(edges[0], edges[2]),
                         max(edges[1], edges[3]) / min(edges[1], edges[3])))

    def offer(self, detection, image_size, *, sharpness=None):
        """Judge one frame. ``detection``: :func:`detect_charuco` result or ``None``; ``image_size``: (w, h) px.

        Returns ``{"kept", "still", "novel", "detected", "motion_px", "corners", "cell", "distance", "tilt",
        "tilt_ratio", "sharpness", "sharpness_ok"}``; ``motion_px`` is ``None`` without a comparable
        previous frame. ``sharpness`` is optional full-frame Laplacian variance (squared gray levels),
        not metres; missing/nonfinite measurements fail an enabled sharpness gate.
        """
        self.frames += 1
        verdict = {"kept": False, "still": False, "novel": False, "detected": detection is not None,
                   "motion_px": None, "corners": 0 if detection is None else int(len(detection["ids"])),
                   "cell": None, "distance": None, "tilt": None, "tilt_ratio": None,
                   "sharpness": None, "sharpness_ok": False}
        if detection is None:
            self.previous, self._still_streak, self.last = None, 0, verdict
            return verdict
        w, h = image_size
        scale = max(w, h) / 640.0
        effective_still_px = self.still_px * max(1.0, scale)
        sharpness_val = sharpness if sharpness is not None else detection.get("sharpness")
        if sharpness_val is not None:
            sharpness_val = float(sharpness_val)
            if not np.isfinite(sharpness_val):
                sharpness_val = None
        sharp_ok = self.min_sharpness <= 0.0 or (sharpness_val is not None and sharpness_val >= self.min_sharpness)
        verdict.update(sharpness=sharpness_val, sharpness_ok=bool(sharp_ok))
        previous, self.previous = self.previous, detection
        if previous is None:
            self._still_streak = 0
        else:
            common, a, b = np.intersect1d(previous["ids"], detection["ids"], return_indices=True)
            if len(common) >= self.min_corners:
                motion = float(np.linalg.norm(np.asarray(previous["corners"])[a]
                                              - np.asarray(detection["corners"])[b], axis=1).mean())
                if motion < effective_still_px and sharp_ok:
                    self._still_streak += 1
                else:
                    self._still_streak = 0
                verdict.update(motion_px=motion, still=bool(self._still_streak >= max(1, self.still_frames - 1)))
            else:
                self._still_streak = 0
        outline = None
        try:
            corners = np.asarray(detection["corners"], float)
            if self.board is not None:
                corners, outline = _board_reference_projection(detection, self.board)
        except ValueError as exc:
            verdict["descriptor_error"] = str(exc)
            self.previous, self._still_streak, self.last = None, 0, verdict
            return verdict
        cx, cy = corners.mean(axis=0) / [w, h]
        verdict["cell"] = [min(2, max(0, int(cy * 3))), min(2, max(0, int(cx * 3)))]
        extent = max(np.ptp(corners[:, 0]) / w, np.ptp(corners[:, 1]) / h)
        verdict["distance"] = ("near" if extent >= self.near_fraction else
                               "far" if extent < self.far_fraction else "mid")
        if verdict["still"]:
            signature = view_signature({"corners": corners}, image_size)
            verdict["novel"] = bool(not self.kept or min(np.linalg.norm(signature - k) for k in self.kept) >= self.min_novelty)
            if verdict["novel"]:
                ratio = self._tilt_ratio(outline)
                verdict["tilt_ratio"] = ratio
                if ratio is not None:
                    verdict["tilt"] = ("large" if ratio >= self.tilt_large else
                                       "small" if ratio >= self.tilt_small else "frontal")
                    self.tilt[verdict["tilt"]] += 1
                self.kept.append(signature)
                self.grid[verdict["cell"][0]][verdict["cell"][1]] += 1
                self.distance[verdict["distance"]] += 1
                verdict["kept"] = True
        self.last = verdict
        return verdict

    @property
    def kept_count(self):
        return len(self.kept)

    @property
    def complete(self):
        return len(self.kept) >= self.target_views

    def coverage(self):
        """Kept-view coverage: ``grid`` 3x3 counts (row 0 = image top), ``distance`` far/mid/near counts,
        ``tilt`` frontal/small/large counts (all zero without a board), plus kept/target/min counts."""
        return {"grid": [list(row) for row in self.grid], "distance": dict(self.distance), "tilt": dict(self.tilt),
                "kept": len(self.kept), "target": self.target_views, "min_kept": self.min_kept}

    def guidance(self):
        """Chinese operator hints, most urgent first (empty list = nothing to say)."""
        hints, last = [], self.last
        if self.complete:
            return ["采集已足够，可以求解"]
        if last is not None and not last["detected"]:
            hints.append("未检测到标定板，请把标定板完整放到固定相机前")
        elif last is not None and not last["still"]:
            hints.append("板子拿稳")
        elif last is not None and last["still"] and not last["novel"]:
            hints.append("这个位置已采过，换个位置")
        missing = [GRID_CELL_NAMES[r][c] for r in range(3) for c in range(3) if self.grid[r][c] == 0]
        if missing:
            hints.append("还缺：" + "、".join(missing))
        if self.kept:
            if self.distance["near"] == 0:
                hints.append("再靠近一些")
            if self.distance["far"] == 0:
                hints.append("再远一些")
            if self.board is not None and self.tilt["small"] + self.tilt["large"] < max(2, len(self.kept) // 3):
                hints.append("倾斜一点")
        return hints


# ----------------------------------------------------------------------------------------------
# Hand-held fixed-camera intrinsics: in-worker capture (existing camera source) and solve
# ----------------------------------------------------------------------------------------------

HANDHELD_CAMERA = "fixed"
HANDHELD_STATUS_FILE = "handheld.json"        # telemetry_dir: operator console progress (never model context)
HANDHELD_PREVIEW_FILE = "fixed_corners.jpg"   # telemetry_dir: latest fixed frame with detected corners drawn


def robot_session_sidecar(active):
    """Path of the operator record of the last robot calibration session next to the active file."""
    active = Path(active)
    return active.with_name(f"{active.stem}.robot_session.json")


class HandheldCapture:
    """Background sampler of the fixed camera's existing :class:`CameraSource` (never opens a camera).

    Each new frame received after the previous one is checked against the configured capture size
    (other sizes are dropped and counted), ChArUco-detected, offered to a :class:`HandheldViewSelector`
    and, when kept, saved as ``<session>/handheld/fixed/NNNN.png`` (same layout as
    ``scripts/so101_calibrate.py capture-handheld``). Progress goes to ``telemetry/handheld.json`` and
    the corner overlay to ``telemetry/fixed_corners.jpg``.
    """

    def __init__(self, adapter, session_dir, board_spec=None):
        import cv2
        from . import charuco as cal
        from .hand_eye_solve import source_revision
        self.adapter, self.session_dir = adapter, Path(session_dir)
        settings = adapter.deployment["calibration"]
        config = adapter.deployment["cameras"][HANDHELD_CAMERA]
        self.size = (int(config["width"]), int(config["height"]))
        self.board_spec = cal.validate_board_spec(board_spec or cal.DEFAULT_BOARD_SPEC)
        self.board = cal.make_board(self.board_spec)
        self.min_corners = settings["min_board_corners"]
        self.selector = HandheldViewSelector(target_views=settings["handheld_target_views"], min_corners=self.min_corners,
                                             still_px=settings["handheld_still_px"],
                                             min_novelty=settings["handheld_min_novelty"], board=self.board,
                                             min_kept=settings["handheld_min_views"],
                                             near_fraction=settings["handheld_near_fraction"],
                                             far_fraction=settings["handheld_far_fraction"],
                                             tilt_small=settings["handheld_tilt_small"],
                                             tilt_large=settings["handheld_tilt_large"],
                                             still_frames=settings["handheld_still_frames"],
                                             min_sharpness=settings["handheld_min_sharpness"])
        self.effective_settings = {key: settings[key] for key in settings if key.startswith("handheld_")}
        self.effective_settings["min_board_corners"] = self.min_corners
        self.timeout_s = settings["handheld_timeout_s"]
        self.out = self.session_dir / "handheld" / HANDHELD_CAMERA
        self.out.mkdir(parents=True, exist_ok=False)
        (self.session_dir / "session.json").write_text(json.dumps({
            "schema": SESSION_SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(), "mode": "handheld_worker",
            "board": self.board_spec, "joint_mapping": _joint_mapping(adapter.deployment),
            "cameras": adapter.deployment["cameras"], "opencv_version": cv2.__version__,
            "camera_devices": _camera_devices(adapter),
            "settings": dict(self.effective_settings),
            "source": source_revision()}, indent=2))
        self.stop_event, self.lock = threading.Event(), threading.Lock()
        self.status, self.reason, self.frames, self.dropped, self.errors = "running", None, 0, 0, 0
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        self.finished_at, self._last_preview = None, None
        self._stop_reason, self._writer_finished = None, False
        self.thread = threading.Thread(target=self._run, daemon=True, name="so101-handheld-capture")

    def start(self):
        self._publish()
        self.thread.start()
        return self.snapshot()

    def stop(self, reason="operator_stop"):
        """Request stop, waiting at most 5 s; live writers stay unknown and cannot be solved."""
        self._stop_reason = reason
        self.stop_event.set()
        if self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=5)
        if self.thread.is_alive():
            # No blocking lock or disk publication after the bounded join. The
            # writer will publish its own final state when current I/O returns.
            self.status, self.reason = "unknown", "writer_shutdown_pending"
            return self._snapshot()
        with self.lock:
            if self.status in ("running", "unknown"):
                self._finish("stopped", reason)
            self._writer_finished = True
        return self.snapshot()

    @property
    def running(self):
        return self.status == "running" or self.thread.is_alive()

    def kept_images(self):
        if self.running or not self._writer_finished:
            raise RuntimeError("handheld_writer_not_stopped")
        return sorted(self.out.glob("*.png"))

    def snapshot(self):
        with self.lock:
            return self._snapshot()

    def _snapshot(self):
        """Advisory fields; stop_confirmed/sealed require the writer thread to have exited."""
        stopped = self._writer_finished and not self.thread.is_alive()
        return {"camera": HANDHELD_CAMERA, "status": self.status, "reason": self.reason,
                    "stop_confirmed": stopped, "writer_stopped": stopped, "sealed": stopped,
                    "session_dir": str(self.session_dir), "started_at": self.started_at, "finished_at": self.finished_at,
                    "kept": self.selector.kept_count, "target_views": self.selector.target_views,
                    "min_views": self.selector.min_kept, "ready_to_solve": self.selector.kept_count >= self.selector.min_kept,
                    "frames": self.frames, "dropped_wrong_size": self.dropped, "frame_errors": self.errors,
                    "image_size": list(self.size), "guidance": self.selector.guidance() if self.frames else [],
                    "coverage": self.selector.coverage(), "last": _plain(self.selector.last),
                    "timeout_s": self.timeout_s, "effective_settings": dict(self.effective_settings)}

    def _finish(self, status, reason):
        self.status, self.reason = status, reason
        self.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    def _publish(self):
        try:
            directory = self.adapter.telemetry_dir
            self.adapter._write_atomic(directory / HANDHELD_STATUS_FILE,
                                       json.dumps(self.snapshot(), allow_nan=False).encode())
        except Exception:
            pass

    def _preview(self, bgr, detection, verdict):
        import cv2
        now = self.adapter.clock.now()
        period = 1 / self.adapter.deployment["telemetry"]["frame_hz"]
        if self._last_preview is not None and now - self._last_preview < period and not verdict["kept"]:
            return
        self._last_preview = now
        image = bgr.copy()
        if detection is not None:
            colour = (0, 200, 0) if verdict["still"] else (0, 165, 255)
            for x, y in np.asarray(detection["corners"], float):
                cv2.circle(image, (int(round(x)), int(round(y))), 4, colour, 2)
        side = self.adapter.deployment["telemetry"]["preview_max_side_px"]
        scale = min(1.0, side / max(image.shape[:2]))
        if scale < 1.0:
            image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            self.adapter._write_atomic(self.adapter.telemetry_dir / HANDHELD_PREVIEW_FILE, buffer.tobytes())

    def _run(self):
        import cv2
        from . import charuco as cal
        camera = self.adapter.cameras[HANDHELD_CAMERA]
        clock, timeout = self.adapter.clock, self.adapter.deployment["camera"]["frame_timeout_s"]
        deadline, last = clock.now() + self.timeout_s, clock.now()
        try:
            while not self.stop_event.is_set():
                if clock.now() > deadline:
                    with self.lock:
                        self._finish("timeout", f"handheld_timeout_s={self.timeout_s} reached")
                    break
                try:
                    image, stamp = camera.read_after(last, timeout)
                except TimeoutError:
                    with self.lock:
                        self.errors += 1
                    continue
                if self.stop_event.is_set():
                    break
                last = stamp
                image = np.asarray(image, dtype=np.uint8)
                with self.lock:
                    self.frames += 1
                    if image.ndim != 3 or (image.shape[1], image.shape[0]) != self.size:
                        self.dropped += 1
                        continue
                bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                detection = cal.detect_charuco(bgr, self.board, min_corners=self.min_corners)
                sharpness = (float(cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                             if self.selector.min_sharpness > 0 else None)
                if self.stop_event.is_set():
                    break
                with self.lock:
                    verdict = self.selector.offer(detection, self.size, sharpness=sharpness)
                    path = self.out / f"{self.selector.kept_count - 1:04d}.png" if verdict["kept"] else None
                if path is not None:
                    # Disk I/O never holds the selector lock or blocks stop's snapshot.
                    if not cv2.imwrite(str(path), bgr):
                        raise OSError(f"failed to write {path}")
                with self.lock:
                    if self.selector.complete:
                        self._finish("completed", None)
                self._preview(bgr, detection, verdict)
                self._publish()
                if self.status != "running":
                    break
        except Exception as error:
            with self.lock:
                self._finish("failed", f"{type(error).__name__}: {error}")
        finally:
            with self.lock:
                if self.stop_event.is_set() and self.status in ("running", "unknown"):
                    self._finish("stopped", self._stop_reason or "operator_stop")
                self._writer_finished = True
            self._publish()


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def handheld_capture_command(adapter, action):
    """Operator ``handheld_capture``: ``action`` start / status / stop (no motion, no bus access)."""
    base = {"command": "handheld_capture", "action": action, "physics_advanced": False}
    current = getattr(adapter, "handheld", None)
    if action == "status":
        return dict(base, status="idle", capture=None) if current is None else dict(base, **current.snapshot(),
                                                                                   capture_status=current.status)
    if action == "stop":
        if current is None:
            return dict(base, status="idle", capture=None)
        snapshot = current.stop()
        return dict(base, **snapshot)
    if action != "start":
        raise ValueError("handheld_capture_action: expected start, status or stop")
    if adapter.active_action_id is not None:
        return dict(base, status="rejected", reason="action_active")
    if HANDHELD_CAMERA not in adapter.cameras or HANDHELD_CAMERA not in adapter.deployment["cameras"]:
        return dict(base, status="rejected", reason="fixed_camera_not_configured")
    if current is not None and current.running:
        current.stop("restarted")
    session_dir = adapter.run_dir / "calibration" / f"handheld-{utc_stamp()}-{uuid.uuid4().hex[:6]}"
    adapter.handheld = HandheldCapture(adapter, session_dir)
    return dict(base, **adapter.handheld.start())


def fixed_extrinsic_from_session(session_dir, wrist, K, dist, board, fk_link, *, mapping=None, T_base_board=None):
    """Fixed-camera extrinsic from one robot capture session with a static board (no hardware access).

    ``wrist``: ``{"K", "distortion", "T_link_camera"}`` of the wrist camera (px, m). ``K``/``dist``: fixed-camera
    intrinsics. ``fk_link(q_rad) -> T_base_gripper_link``. ``mapping``: joint mapping of the samples' LeRobot
    degrees (default the session's). ``T_base_board``: board pose in base (4x4, m); default the mean wrist-chain
    estimate ``fk(q) @ T_link_camera @ T_camera_board`` over the samples.
    Returns ``{"T_base_camera", "T_base_board", "chain_reprojection_px", "fixed_samples", "wrist_samples",
    "board_spread": {...}}`` or raises ``LookupError`` with the reason (no usable views).
    """
    from . import charuco as cal
    from .hand_eye_solve import _detect, load_session
    session_dir = Path(session_dir)
    try:
        session, samples = load_session(session_dir)
    except (OSError, ValueError) as error:
        raise LookupError(f"session_unavailable: {type(error).__name__}: {error}") from error
    mapping = mapping or session.get("joint_mapping")
    fixed_T, rows, chain = [], [], []
    for sample in samples:
        if not sample.get("settled", True):
            continue
        images = sample.get("images") or {}
        if "fixed" not in images or "wrist" not in images:
            continue
        det_f, _ = _detect(session_dir / images["fixed"], board)
        det_w, _ = _detect(session_dir / images["wrist"], board)
        if det_w is not None:
            T_wb, _ = cal.board_pose(det_w["corners"], det_w["ids"], wrist["K"], wrist["distortion"], board)
            q = cal.lerobot_deg_to_urdf_rad(sample["joints_deg"], mapping)
            T_chain = np.asarray(fk_link(q)) @ np.asarray(wrist["T_link_camera"], float) @ T_wb
            chain.append(T_chain)
            if det_f is not None:
                rows.append((det_f, T_chain))
        if det_f is not None:
            fixed_T.append(cal.board_pose(det_f["corners"], det_f["ids"], K, dist, board)[0])
    if not fixed_T:
        raise LookupError("session_has_no_fixed_board_views")
    if T_base_board is None:
        if not chain:
            raise LookupError("session_has_no_wrist_board_views")
        T_base_board = cal.mean_transform(chain)
    spread = cal.transform_spread(chain) if len(chain) > 1 else None
    T_base_camera = cal.solve_fixed_camera(np.asarray(T_base_board, float), fixed_T)["T_base_camera"]
    T_cam_base = cal.invert_transform(T_base_camera)
    errors = []
    for det, T_base_board_i in rows:
        predicted = cal.project_board_corners(T_cam_base @ T_base_board_i, det["ids"], K, dist, board)
        errors.extend(np.sum((predicted - det["corners"]) ** 2, axis=1).tolist())
    return {"T_base_camera": T_base_camera, "T_base_board": np.asarray(T_base_board, float),
            "chain_reprojection_px": float(np.sqrt(np.mean(errors))) if errors else None,
            "fixed_samples": len(fixed_T), "wrist_samples": len(chain),
            "board_spread": None if spread is None else {k: spread[k] for k in ("translation_rms_m", "rotation_rms_rad")}}


def _resolve_extrinsic(adapter, document, K, dist, board, fk_link):
    """Re-solve ``T_base_camera`` of the fixed camera with new intrinsics from the last robot session.

    Needs the sidecar written by :func:`run_calibration` (``T_base_board`` and the session's fixed images).
    Returns ``(T_base_camera, chain_reprojection_px, samples)`` or raises ``LookupError`` with the reason.
    """
    from .hand_eye_solve import load_session
    sidecar = robot_session_sidecar(adapter.calibration_path)
    try:
        record = json.loads(sidecar.read_text())
        T_base_board = np.asarray(record["T_base_board"], float).reshape(4, 4)
        session_dir = Path(record["session_dir"])
        load_session(session_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise LookupError(f"last_robot_session_unavailable: {type(error).__name__}: {error}") from error
    wrist = document["cameras"].get("wrist")
    if wrist is None:
        raise LookupError("active_calibration_has_no_wrist_camera")
    mapping = record.get("joint_mapping") or _joint_mapping(adapter.deployment)
    try:
        solved = fixed_extrinsic_from_session(session_dir, wrist, K, dist, board, fk_link, mapping=mapping,
                                              T_base_board=T_base_board)
    except LookupError as error:
        raise LookupError(f"last_robot_{error}") from error
    return solved["T_base_camera"], solved["chain_reprojection_px"], solved["fixed_samples"]


def handheld_intrinsics(capture, min_views):
    """Fixed-camera intrinsics from the kept views of a :class:`HandheldCapture` (stopped or running).

    Returns ``(K 3x3 px, distortion [k1,k2,p1,p2,k3], rms_px, used_views)``; raises ``ValueError`` with fewer
    than ``min_views`` usable detections. Kept images whose size differs from the capture size are ignored.
    """
    import cv2
    from . import charuco as cal
    detections = []
    for path in capture.kept_images():
        image = cv2.imread(str(path))
        if image is None or (image.shape[1], image.shape[0]) != capture.size:
            continue
        detection = cal.detect_charuco(image, capture.board, min_corners=capture.min_corners)
        if detection is not None:
            detections.append(detection)
    K, dist, rms = cal.calibrate_intrinsics(detections, capture.size, capture.board, min_views=min_views)
    return K, dist, float(rms), len(detections)


def run_handheld_solve(adapter, *, fk_link=None):
    """Operator ``handheld_solve``: fixed-camera intrinsics from the kept hand-held views.

    Gate: ``calibration.max_handheld_intrinsics_rms_px``; when the extrinsic is re-solved (last robot
    session available) also ``calibration.max_fixed_reprojection_px`` on the chain reprojection. On pass a
    new file (fixed K/distortion, ``intrinsics_source="handheld"``, wrist entry unchanged) is installed like
    ``calibrate`` does; every other status keeps the active file.
    """
    from . import charuco as cal
    settings = adapter.deployment["calibration"]
    base = {"command": "handheld_solve", "physics_advanced": False, "previous": adapter.calibration_identity,
            "gate_px": settings["max_handheld_intrinsics_rms_px"], "extrinsic_updated": False}
    capture = getattr(adapter, "handheld", None)
    if not adapter.calibration_path or not Path(adapter.calibration_path).is_file():
        return dict(base, status="rejected", reason="active_camera_calibration_file_required")
    if capture is None:
        return dict(base, status="rejected", reason="no_handheld_capture")
    if capture.running:
        capture.stop("stopped_for_solve")
    images = capture.kept_images()
    result = dict(base, session_dir=str(capture.session_dir), views=len(images))
    if len(images) < settings["handheld_min_views"]:
        return dict(result, status="insufficient_views",
                    reason=f"kept {len(images)} views < calibration.handheld_min_views={settings['handheld_min_views']}")
    board = capture.board
    try:
        K, dist, rms, used = handheld_intrinsics(capture, settings["handheld_min_views"])
        document = cal.read_calibration_file(adapter.calibration_path)
    except Exception as error:
        return dict(result, status="solve_failed", reason=f"{type(error).__name__}: {error}")
    result.update(rms_px=rms, used_views=used, K=np.asarray(K).tolist(),
                  distortion=np.asarray(dist).reshape(-1).tolist())
    old_fixed = document["cameras"].get("fixed")
    if old_fixed is None:
        return dict(result, status="rejected", reason="active_calibration_has_no_fixed_camera")
    failures = []
    if rms > settings["max_handheld_intrinsics_rms_px"]:
        failures.append(f"fixed.intrinsics_rms_px={rms:.4g} > max_handheld_intrinsics_rms_px="
                        f"{settings['max_handheld_intrinsics_rms_px']:.4g}")
    fixed = {"K": K, "distortion": dist, "image_size": list(capture.size), "intrinsics_rms_px": rms,
             "intrinsics_source": "handheld", "T_base_camera": old_fixed["T_base_camera"],
             "extrinsic_residual": dict(old_fixed["extrinsic_residual"])}
    if fk_link is None:
        link = adapter.kin.WRIST_CAMERA_PARENT_LINK
        fk_link = lambda q: np.asarray(adapter.kin.fk(adapter.model, list(q), link=link))  # noqa: E731
    try:
        T_base_camera, chain, samples = _resolve_extrinsic(adapter, document, K, dist, board, fk_link)
    except LookupError as error:
        result.update(extrinsic_reason=str(error), extrinsic_note="T_base_camera kept from the active file "
                      "(solved with the previous intrinsics)")
    else:
        result.update(extrinsic_updated=True, extrinsic_samples=samples, fixed_chain_reprojection_px=chain,
                      extrinsic_gate_px=settings["max_fixed_reprojection_px"])
        fixed["T_base_camera"] = T_base_camera
        if chain is not None:
            fixed["extrinsic_residual"] = {"reprojection_px": chain}
            if chain > settings["max_fixed_reprojection_px"]:
                failures.append(f"fixed.extrinsic_residual.reprojection_px={chain:.4g} > max_fixed_reprojection_px="
                                f"{settings['max_fixed_reprojection_px']:.4g}")
        else:
            result["extrinsic_note"] = "no robot sample with both cameras: chain residual kept from the active file"
    result["gate_failures"] = failures
    result["passed"] = not failures
    if failures:
        return dict(result, status="gate_failed", reason="; ".join(failures), extrinsic_updated=False)
    stamp = utc_stamp()
    active = Path(adapter.calibration_path)
    candidate = active.with_name(f"{active.stem}-handheld-{stamp}{active.suffix}")
    cameras = {name: dict(camera) for name, camera in document["cameras"].items()}
    cameras["fixed"] = fixed
    try:
        cal.write_calibration_file(candidate, board=document["board"], cameras=cameras, samples=document["samples"])
        (capture.session_dir / "handheld_solve.json").write_text(json.dumps(_plain(result), indent=2, default=str))
        reload, backup = _install(adapter, candidate, active, stamp)
    except Exception as error:
        return dict(result, status="install_failed", reason=f"{type(error).__name__}: {error}", extrinsic_updated=False,
                    calibration=adapter.calibration_identity)
    return dict(result, status="calibrated", reason=None, calibration=reload["calibration"], candidate_file=str(candidate),
                previous_file_backup=None if backup is None else str(backup), active_file=str(active))
