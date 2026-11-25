"""Offline SO-101 camera calibration solve/verify (CPU only; no serial port or camera).

Importable core of ``scripts/so101_calibrate.py solve|verify`` and of the worker's one-click
recalibration (``ni_rua/robot/backends/so101/calibration_capture.py``). Input is a capture session
directory (``session.json`` schema ``ni_rua.so101.calibration_session.v1``, ``samples.jsonl``,
``images/``, optional ``handheld/<camera>/*.png``); output is a gated ``camera_calibration.json``
(schema ``ni_rua.so101.camera_calibration.v1``) plus ``solve_report.json`` with per-sample residuals.
Depends on numpy + cv2 (aruco); kinematics is imported lazily or injected as ``fk_link``.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Callable

import cv2
import numpy as np

from . import charuco as cal
from .defaults import calibration_default

REPO = Path(__file__).resolve().parents[3]
SESSION_SCHEMA = "ni_rua.so101.calibration_session.v1"
REPORT_SCHEMA = "ni_rua.so101.calibration_solve_report.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def board_spec_with_overrides(spec: dict, square_m: float | None, marker_m: float | None) -> dict:
    """Apply a measured printed square edge (m); the marker edge is scaled by the same ratio unless given."""
    spec = cal.validate_board_spec(spec)
    if square_m is not None:
        ratio = float(square_m) / spec["square_m"]
        spec["square_m"] = float(square_m)
        spec["marker_m"] = float(marker_m) if marker_m is not None else spec["marker_m"] * ratio
    elif marker_m is not None:
        spec["marker_m"] = float(marker_m)
    return cal.validate_board_spec(spec)


def source_revision() -> dict[str, Any]:
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True,
                                    text=True, timeout=5).stdout.strip())
    except Exception:
        head, dirty = None, None
    return {"git_head": head, "worktree_dirty": dirty}


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, default=_json_default) + "\n", encoding="utf-8")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value).__name__)


def load_session(session_dir: str | Path) -> tuple[dict, list[dict]]:
    session_dir = Path(session_dir)
    session = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    if session.get("schema") != SESSION_SCHEMA:
        raise ValueError(f"{session_dir}: not a calibration session ({session.get('schema')!r})")
    samples = []
    path = session_dir / "samples.jsonl"
    if path.exists():
        samples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return session, samples


def handheld_images(session_dir: Path, camera: str) -> list[Path]:
    return sorted((session_dir / "handheld" / camera).glob("*.png"))


def load_kinematics():
    try:
        from ...embodiments.so101 import profile as kin
    except ImportError as error:
        raise RuntimeError(f"so101_kinematics is required for this step: {error}") from error
    return kin, kin.load_model()


def default_fk_link() -> Callable[[np.ndarray], np.ndarray]:
    """``fk(q_rad) -> T_base_gripper_link`` from the packaged URDF."""
    kin, model = load_kinematics()
    return lambda q: kin.fk(model, q, kin.WRIST_CAMERA_PARENT_LINK)


def _detect(path: Path, board):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"cannot read {path}")
    return cal.detect_charuco(image, board), (image.shape[1], image.shape[0])


def solve_session(session_dir: str | Path, *, output: str | Path | None = None, square_m: float | None = None,
                  marker_m: float | None = None, joint_mapping: dict | None = None,
                  thresholds: dict | None = None, force: bool = False, overwrite: bool = False,
                  intrinsics_from: str | None = None, exclude: set[int] | None = None,
                  include_unsettled: bool = False,
                  max_pnp_reprojection_px: float = calibration_default("max_pnp_reprojection_px"),
                  hand_eye_score_lever_m: float = calibration_default("hand_eye_score_lever_m"),
                  fixed_intrinsics_from_wrist: bool = False,
                  fixed_intrinsics: Any = None, fixed_distortion: Any = None,
                  fixed_intrinsics_rms_px: float | None = None,
                  fixed_intrinsics_source: str = "handheld",
                  fk_link: Callable[[np.ndarray], np.ndarray] | None = None) -> dict:
    """Offline solve of one capture session. Returns the solve report (also written to solve_report.json).

    ``fk_link(q_rad) -> T_base_gripper_link`` defaults to the packaged URDF kinematics. Samples are skipped
    (with the reason recorded) when unsettled, excluded, undetected, or their wrist PnP RMS exceeds
    ``max_pnp_reprojection_px`` (px). The calibration file is written only if every gate passes or ``force``.
    ``hand_eye_score_lever_m``: m/rad coefficient in method-selection score (translation RMS plus
    coefficient times angular RMS), default 0.1 from packaged data; 0 ranks by translation only.
    Example: ``solve_session(directory, hand_eye_score_lever_m=0.2)``. It changes ranking, not gates.
    """
    session_dir = Path(session_dir)
    session, samples = load_session(session_dir)
    spec = board_spec_with_overrides(session["board"], square_m, marker_m)
    board = cal.make_board(spec)
    mapping = joint_mapping if joint_mapping is not None else session.get("joint_mapping")
    fk_link = fk_link or default_fk_link()
    exclude = exclude or set()
    prior = cal.read_calibration_file(intrinsics_from)["cameras"] if intrinsics_from else {}

    detections: dict[str, dict[int, Any]] = {"wrist": {}, "fixed": {}}
    sizes: dict[str, tuple[int, int]] = {}
    for s in samples:
        for name, rel in s["images"].items():
            det, size = _detect(session_dir / rel, board)
            detections[name][s["index"]] = det
            sizes[name] = size
    handheld = {}
    for name in ("wrist", "fixed"):
        handheld[name] = []
        for path in handheld_images(session_dir, name):
            det, size = _detect(path, board)
            sizes.setdefault(name, size)
            if det is not None:
                handheld[name].append(det)

    report: dict[str, Any] = {"schema": REPORT_SCHEMA, "created_at": utc_now(), "session": str(session_dir),
                              "board": spec, "joint_mapping": cal.normalize_joint_mapping(mapping), "cameras": {},
                              "skipped": {}, "source": source_revision(),
                              "effective_settings": {"max_pnp_reprojection_px": max_pnp_reprojection_px,
                                                     "hand_eye_score_lever_m": hand_eye_score_lever_m}}
    intrinsics = {}
    for name in ("wrist", "fixed"):
        if name not in sizes:
            continue
        if name in prior:
            K, dist, rms = prior[name]["K"], prior[name]["distortion"], prior[name].get("intrinsics_rms_px")
            if list(prior[name]["image_size"]) != list(sizes[name]):
                raise ValueError(f"{name}: --intrinsics-from image_size {prior[name]['image_size']} != {sizes[name]}")
            source = f"file:{intrinsics_from}"
        elif name == "fixed" and fixed_intrinsics is not None and not handheld[name]:
            if isinstance(fixed_intrinsics, dict):
                K, dist = fixed_intrinsics["K"], fixed_intrinsics["distortion"]
                rms = fixed_intrinsics.get("intrinsics_rms_px", fixed_intrinsics_rms_px)
                source = fixed_intrinsics.get("intrinsics_source", fixed_intrinsics_source)
            else:
                K = fixed_intrinsics
                dist = fixed_distortion if fixed_distortion is not None else np.zeros(5)
                rms = fixed_intrinsics_rms_px
                source = fixed_intrinsics_source
        else:
            views = list(handheld[name])
            if name == "wrist":  # wrist views vary with the arm; the fixed camera sees one static view
                views += [d for d in detections[name].values() if d is not None]
            if len(views) < 8 and name == "fixed" and fixed_intrinsics_from_wrist and "wrist" in intrinsics \
                    and list(sizes["fixed"]) == intrinsics["wrist"]["image_size"]:
                # Static board => one fixed-camera viewpoint, intrinsics unobservable. Same camera model:
                # borrow the wrist intrinsics; the fixed chain reprojection error below checks the result.
                intrinsics[name] = {**intrinsics["wrist"], "intrinsics_source": "copied_from_wrist_same_model_unverified"}
                continue
            if len(views) < 8:
                report["cameras"][name] = {"error": f"only {len(views)} usable intrinsics views (need 8); "
                                                    "use capture-handheld or --intrinsics-from"}
                continue
            K, dist, rms = cal.calibrate_intrinsics(views, sizes[name], board)
            source = f"{len(views)} views"
        intrinsics[name] = {"K": np.asarray(K), "distortion": np.asarray(dist), "intrinsics_rms_px": rms,
                            "image_size": list(sizes[name]), "intrinsics_source": source}

    if "wrist" not in intrinsics:
        write_json(session_dir / "solve_report.json", report)
        raise ValueError("wrist intrinsics unavailable: " + json.dumps(report["cameras"].get("wrist")))

    used, A, B, per_sample = [], [], [], []
    for s in samples:
        idx = s["index"]
        reason = None
        if idx in exclude:
            reason = "excluded"
        elif not s.get("settled", True) and not include_unsettled:
            reason = "not_settled"
        elif detections["wrist"].get(idx) is None:
            reason = "wrist_board_not_detected"
        if reason is None:
            det = detections["wrist"][idx]
            T_cb, rms_px = cal.board_pose(det["corners"], det["ids"], intrinsics["wrist"]["K"],
                                          intrinsics["wrist"]["distortion"], board)
            if rms_px > max_pnp_reprojection_px:
                reason = f"wrist_pnp_rms_px={rms_px:.3f}"
        if reason:
            report["skipped"][str(idx)] = reason
            continue
        q = cal.lerobot_deg_to_urdf_rad(s["joints_deg"], mapping)
        used.append(idx)
        A.append(np.asarray(fk_link(q)))
        B.append(T_cb)
        per_sample.append({"index": idx, "wrist_pnp_rms_px": rms_px, "wrist_corners": int(len(det["ids"]))})
    he = cal.solve_eye_in_hand(A, B, score_lever_m=hand_eye_score_lever_m)
    for row, t_err, r_err in zip(per_sample, he["translation_errors_m"], he["rotation_errors_rad"]):
        row["hand_eye_translation_error_m"], row["hand_eye_rotation_error_rad"] = t_err, r_err
    report["T_base_board"] = he["T_base_board"]
    report["hand_eye"] = {k: he[k] for k in ("method", "methods", "samples", "translation_rms_m",
                                             "translation_max_m", "rotation_rms_rad", "rotation_max_rad", "score_m", "score_lever_m")}
    cameras = {"wrist": {**intrinsics["wrist"], "T_link_camera": he["T_gripper_camera"],
                         "hand_eye_residual": {"translation_m": he["translation_rms_m"],
                                               "rotation_rad": he["rotation_rms_rad"]}}}

    if "fixed" in intrinsics:
        fixed_rows, fixed_T, chain_pred = [], [], []
        Kf, df = intrinsics["fixed"]["K"], intrinsics["fixed"]["distortion"]
        for row, T_bg, T_cb in zip(per_sample, A, B):
            det = detections["fixed"].get(row["index"])
            if det is None:
                continue
            T_fb, rms_px = cal.board_pose(det["corners"], det["ids"], Kf, df, board)
            fixed_T.append(T_fb)
            fixed_rows.append((row, det, T_bg @ he["T_gripper_camera"] @ T_cb))
            row["fixed_pnp_rms_px"] = rms_px
        if fixed_T:
            fx = cal.solve_fixed_camera(he["T_base_board"], fixed_T)
            T_cam_base = cal.invert_transform(fx["T_base_camera"])
            sq = []
            for row, det, T_base_board_i in fixed_rows:
                pred = cal.project_board_corners(T_cam_base @ T_base_board_i, det["ids"], Kf, df, board)
                err = np.sum((pred - det["corners"]) ** 2, axis=1)
                row["fixed_chain_reprojection_px"] = float(np.sqrt(err.mean()))
                sq.extend(err.tolist())
            reproj = float(np.sqrt(np.mean(sq)))
            report["fixed"] = {k: fx[k] for k in ("samples", "translation_rms_m", "translation_max_m",
                                                  "rotation_rms_rad", "rotation_max_rad")}
            report["fixed"]["chain_reprojection_px"] = reproj
            cameras["fixed"] = {**intrinsics["fixed"], "T_base_camera": fx["T_base_camera"],
                                "extrinsic_residual": {"reprojection_px": reproj}}
        else:
            report["cameras"]["fixed"] = {"error": "board never detected by the fixed camera in robot samples"}
    for name, cam in intrinsics.items():
        report["cameras"].setdefault(name, {}).update(
            {"intrinsics_rms_px": cam["intrinsics_rms_px"], "intrinsics_source": cam["intrinsics_source"],
             "K": cam["K"], "distortion": cam["distortion"], "image_size": cam["image_size"]})
    report["per_sample"] = per_sample
    report["thresholds"] = thresholds or {}
    failures = cal.check_residual_gates(cameras, thresholds or {})
    report["gate_failures"] = failures
    output = Path(output) if output else session_dir / "camera_calibration.json"
    report["calibration_file"] = None
    try:
        cal.write_calibration_file(output, board=spec, cameras=cameras, samples=len(used),
                                   thresholds=thresholds, force=force, overwrite=overwrite)
        report["calibration_file"] = str(output)
        report["forced"] = bool(failures and force)
    finally:
        write_json(session_dir / "solve_report.json", report)
    return report


def verify_session(session_dir: str | Path, calibration: str | Path, *, joint_mapping: dict | None = None,
                   fk_link: Callable[[np.ndarray], np.ndarray] | None = None) -> dict:
    """Cross-camera reprojection check of a calibration file on a session (ideally a fresh, held-out capture).

    The board pose in base is not in the calibration file, so it is re-estimated: from the fixed camera
    (``T_base_camera @ T_fixed_board_i``) to predict the wrist view, and leave-one-out from the wrist chain
    (mean of the other samples) to predict wrist and fixed views. Reports RMS / max px per camera.
    """
    session_dir = Path(session_dir)
    session, samples = load_session(session_dir)
    doc = cal.read_calibration_file(calibration)
    board = cal.make_board(doc["board"])
    mapping = joint_mapping if joint_mapping is not None else session.get("joint_mapping")
    fk_link = fk_link or default_fk_link()
    wrist, fixed = doc["cameras"].get("wrist"), doc["cameras"].get("fixed")
    if wrist is None:
        raise ValueError("verify needs a wrist camera in the calibration file")
    rows = []
    for s in samples:
        det_w, _ = _detect(session_dir / s["images"]["wrist"], board) if "wrist" in s["images"] else (None, None)
        if det_w is None:
            continue
        det_f = None
        if fixed is not None and "fixed" in s["images"]:
            det_f, _ = _detect(session_dir / s["images"]["fixed"], board)
        T_cb, _ = cal.board_pose(det_w["corners"], det_w["ids"], wrist["K"], wrist["distortion"], board)
        T_base_cam = np.asarray(fk_link(cal.lerobot_deg_to_urdf_rad(s["joints_deg"], mapping))) @ wrist["T_link_camera"]
        rows.append({"index": s["index"], "det_w": det_w, "det_f": det_f, "T_base_cam": T_base_cam,
                     "T_base_board_wrist": T_base_cam @ T_cb})
    if len(rows) < 2:
        raise ValueError("verify needs at least 2 samples with the board detected by the wrist camera")

    def rms(pred, det):
        return np.sum((pred - det["corners"]) ** 2, axis=1)

    out = {"session": str(session_dir), "calibration": str(calibration), "per_sample": []}
    acc = {"wrist_from_wrist_loo": [], "wrist_from_fixed": [], "fixed_from_wrist_loo": []}
    for i, r in enumerate(rows):
        loo = cal.mean_transform([o["T_base_board_wrist"] for j, o in enumerate(rows) if j != i])
        entry = {"index": r["index"]}
        e = rms(cal.project_board_corners(cal.invert_transform(r["T_base_cam"]) @ loo, r["det_w"]["ids"],
                                          wrist["K"], wrist["distortion"], board), r["det_w"])
        acc["wrist_from_wrist_loo"].extend(e)
        entry["wrist_from_wrist_loo_px"] = float(np.sqrt(e.mean()))
        if r["det_f"] is not None:
            T_fb, _ = cal.board_pose(r["det_f"]["corners"], r["det_f"]["ids"], fixed["K"], fixed["distortion"], board)
            T_base_board_fixed = fixed["T_base_camera"] @ T_fb
            e = rms(cal.project_board_corners(cal.invert_transform(r["T_base_cam"]) @ T_base_board_fixed,
                                              r["det_w"]["ids"], wrist["K"], wrist["distortion"], board), r["det_w"])
            acc["wrist_from_fixed"].extend(e)
            entry["wrist_from_fixed_px"] = float(np.sqrt(e.mean()))
            e = rms(cal.project_board_corners(cal.invert_transform(fixed["T_base_camera"]) @ loo,
                                              r["det_f"]["ids"], fixed["K"], fixed["distortion"], board), r["det_f"])
            acc["fixed_from_wrist_loo"].extend(e)
            entry["fixed_from_wrist_loo_px"] = float(np.sqrt(e.mean()))
        out["per_sample"].append(entry)
    out["summary"] = {k: ({"rms_px": float(np.sqrt(np.mean(v))), "max_px": float(np.sqrt(np.max(v))),
                           "corners": len(v)} if v else None) for k, v in acc.items()}
    out["samples"] = len(rows)
    return out
