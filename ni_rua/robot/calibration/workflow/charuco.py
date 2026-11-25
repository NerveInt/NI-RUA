"""SO-101 camera calibration math: ChArUco detection, intrinsics, hand-eye, fixed extrinsic, file schema.

Pure module: depends only on ``numpy`` and ``cv2`` (with ``cv2.aruco``). It never opens a serial port,
camera, or imports LeRobot. Kinematics is injected as callables so this module does not depend on
``so101_kinematics``.

Conventions (all poses are 4x4 homogeneous transforms, metres and radians):

- ``T_a_b`` maps points expressed in frame ``b`` into frame ``a`` (``p_a = T_a_b @ p_b``).
- Camera frames use the OpenCV optical convention: +x right, +y down, +z forward (optical axis).
- Board frame is the OpenCV ChArUco frame: origin at the outer corner of the first square, +x along
  ``squares_x``, +y along ``squares_y``, +z = x cross y, which points *into* the board (away from a camera
  looking at the printed face). Board corner coordinates come from ``board.getChessboardCorners()``.
- ``base`` is the URDF ``base_link`` frame (``so101_base``); ``gripper``/``link`` is the link the wrist
  camera is mounted on (``gripper_link``).

Hand-eye: the board is static in the base frame, so for every sample ``i``
``T_base_board = T_base_gripper_i @ T_gripper_camera @ T_camera_board_i``. The *consistency residual*
is the spread of that product across samples (translation in m, rotation in rad); it is the quality
metric used to pick among solver methods and to gate saving.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .defaults import calibration_default


SCHEMA = "ni_rua.so101.camera_calibration.v1"
REFERENCE_FRAME_ID = "so101_base"
DISTORTION_MODEL = "opencv_plumb_bob"
OPTICAL_AXES = "opencv: +x right, +y down, +z forward"
WRIST_LINK_NAME = "gripper_link"
WRIST_ARM = "right"
# Arm joint order shared with so101_kinematics.ARM_JOINTS and LeRobot SOFollower motor names.
ARM_JOINTS: tuple[str, ...] = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

DEFAULT_BOARD_SPEC: dict[str, Any] = calibration_default("default_board")

# Characteristic lever arm (m per rad) used to fold the rotation spread into one hand-eye score:
# score_m = translation_rms_m + score_lever_m * rotation_rms_rad. This is a configurable
# ranking tradeoff, not a measured camera distance. The constant retains the standalone default.
HAND_EYE_SCORE_LEVER_M = calibration_default("hand_eye_score_lever_m")

# OpenCV calibrateHandEye solver suite evaluated by solve_eye_in_hand.
_CV2_HAND_EYE_METHODS = ("TSAI", "PARK", "HORAUD", "DANIILIDIS", "ANDREFF")


class CalibrationGateError(RuntimeError):
    """Raised when residuals exceed the caller's thresholds and saving was not forced."""

    def __init__(self, failures: list[str]):
        super().__init__("calibration residuals exceed thresholds: " + "; ".join(failures))
        self.failures = failures


# ----------------------------------------------------------------------------------------------
# Small rigid-transform helpers
# ----------------------------------------------------------------------------------------------

def make_transform(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    """Build a 4x4 transform from a 3x3 rotation matrix and a translation vector (m)."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid 4x4 transform (uses R^T, not a general matrix inverse)."""
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    return make_transform(R.T, -R.T @ T[:3, 3])


def rotvec_to_matrix(rotvec: Sequence[float]) -> np.ndarray:
    """Axis-angle vector (rad, direction = axis, norm = angle) -> 3x3 rotation matrix."""
    return Rotation.from_rotvec(np.asarray(rotvec, dtype=float).reshape(3)).as_matrix()


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> axis-angle vector (rad)."""
    return Rotation.from_matrix(np.asarray(R, dtype=float).reshape(3, 3)).as_rotvec()


def rotation_angle(R: np.ndarray) -> float:
    """Rotation angle (rad, in [0, pi]) of a 3x3 rotation matrix."""
    c = (np.trace(np.asarray(R, dtype=float)) - 1.0) / 2.0
    return float(math.acos(max(-1.0, min(1.0, c))))


def project_to_rotation(M: np.ndarray) -> np.ndarray:
    """Nearest proper rotation (Frobenius norm) to a 3x3 matrix, via SVD."""
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=float))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt)) or 1.0])
    return U @ D @ Vt


def mean_transform(transforms: Sequence[np.ndarray]) -> np.ndarray:
    """Average of rigid transforms: arithmetic mean translation, chordal (SVD-projected) mean rotation."""
    Ts = np.asarray(transforms, dtype=float)
    return make_transform(project_to_rotation(Ts[:, :3, :3].sum(axis=0)), Ts[:, :3, 3].mean(axis=0))


def transform_spread(transforms: Sequence[np.ndarray]) -> dict[str, Any]:
    """Spread of a set of transforms that should all be equal.

    Returns ``mean`` (4x4), ``translation_rms_m`` / ``translation_max_m`` (distance of each translation from
    the mean, m) and ``rotation_rms_rad`` / ``rotation_max_rad`` (angle between each rotation and the mean
    rotation, rad), plus the per-sample lists ``translation_errors_m`` and ``rotation_errors_rad``.
    """
    Ts = np.asarray(transforms, dtype=float)
    mean = mean_transform(Ts)
    t_err = np.linalg.norm(Ts[:, :3, 3] - mean[:3, 3], axis=1)
    r_err = np.array([rotation_angle(mean[:3, :3].T @ T[:3, :3]) for T in Ts])
    return {
        "mean": mean,
        "translation_rms_m": float(np.sqrt(np.mean(t_err ** 2))),
        "translation_max_m": float(t_err.max()),
        "rotation_rms_rad": float(np.sqrt(np.mean(r_err ** 2))),
        "rotation_max_rad": float(r_err.max()),
        "translation_errors_m": t_err.tolist(),
        "rotation_errors_rad": r_err.tolist(),
    }


# ----------------------------------------------------------------------------------------------
# Board
# ----------------------------------------------------------------------------------------------

def validate_board_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalise a ChArUco board spec.

    Keys: ``type`` ("charuco"), ``squares_x`` / ``squares_y`` (int, number of chessboard squares along
    board x / y), ``square_m`` (float, printed square edge, m), ``marker_m`` (float, printed ArUco marker
    edge, m, strictly smaller than ``square_m``), ``dictionary`` (name of a ``cv2.aruco.DICT_*`` constant).
    """
    out = dict(DEFAULT_BOARD_SPEC)
    out.update(dict(spec))
    if out["type"] != "charuco":
        raise ValueError(f"unsupported board type: {out['type']!r}")
    out["squares_x"], out["squares_y"] = int(out["squares_x"]), int(out["squares_y"])
    out["square_m"], out["marker_m"] = float(out["square_m"]), float(out["marker_m"])
    if out["squares_x"] < 3 or out["squares_y"] < 3:
        raise ValueError("board needs at least 3x3 squares")
    if not (0 < out["marker_m"] < out["square_m"]):
        raise ValueError("board requires 0 < marker_m < square_m")
    if not hasattr(cv2.aruco, str(out["dictionary"])):
        raise ValueError(f"unknown ArUco dictionary: {out['dictionary']!r}")
    return out


def make_board(spec: Mapping[str, Any] | None = None):
    """Construct a ``cv2.aruco.CharucoBoard`` from a board spec (see :func:`validate_board_spec`).

    Requires OpenCV >= 4.7 (``CharucoBoard((sx, sy), square_m, marker_m, dictionary)`` API).
    The returned board's object points are in metres.
    """
    spec = validate_board_spec(spec or DEFAULT_BOARD_SPEC)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec["dictionary"]))
    return cv2.aruco.CharucoBoard((spec["squares_x"], spec["squares_y"]), spec["square_m"], spec["marker_m"],
                                  dictionary)


def board_corner_points(board, ids: Sequence[int] | None = None) -> np.ndarray:
    """Board-frame 3D coordinates (N, 3) in metres of the inner chessboard corners (all, or by ``ids``)."""
    pts = np.asarray(board.getChessboardCorners(), dtype=float).reshape(-1, 3)
    return pts if ids is None else pts[np.asarray(ids, dtype=int).reshape(-1)]


def board_outline_points(board) -> np.ndarray:
    """Board-frame 3D coordinates (4, 3) in metres of the outer board rectangle corners."""
    sx, sy = board.getChessboardSize()
    s = float(board.getSquareLength())
    return np.array([[0, 0, 0], [sx * s, 0, 0], [sx * s, sy * s, 0], [0, sy * s, 0]], dtype=float)


# ----------------------------------------------------------------------------------------------
# Detection, intrinsics, board pose
# ----------------------------------------------------------------------------------------------

def detect_charuco(image_bgr: np.ndarray, board, *, min_corners: int = 6) -> dict[str, np.ndarray] | None:
    """Detect ChArUco inner corners in one image.

    Args:
        image_bgr: HxWx3 BGR uint8 image (or HxW grayscale).
        board: board from :func:`make_board`.
        min_corners: minimum number of interpolated chessboard corners required (default 6); fewer, or
            all corners on one line (degenerate for PnP/calibration), returns ``None``.

    Returns:
        ``{"corners": (N, 2) float64 pixel coordinates (x right, y down), "ids": (N,) int32 corner ids}``
        or ``None`` when the board is not reliably visible.
    """
    if not hasattr(cv2.aruco, "CharucoDetector"):
        raise RuntimeError("OpenCV >= 4.7 with cv2.aruco.CharucoDetector is required")
    image = np.asarray(image_bgr)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners, ids, _marker_corners, _marker_ids = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    if corners is None or ids is None:
        return None
    corners = np.asarray(corners, dtype=float).reshape(-1, 2)
    ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    if len(ids) < max(4, int(min_corners)):
        return None
    if board.checkCharucoCornersCollinear(ids.reshape(-1, 1)):
        return None
    return {"corners": corners, "ids": ids}


def calibrate_intrinsics(detections: Sequence[Mapping[str, np.ndarray]], image_size: Sequence[int], board, *,
                         flags: int = 0, min_views: int = 8) -> tuple[np.ndarray, np.ndarray, float]:
    """Pinhole + plumb-bob intrinsics from ChArUco detections of one camera.

    Args:
        detections: list of :func:`detect_charuco` results from distinct views (board/camera relative pose
            must vary; identical views add no information).
        image_size: ``(width, height)`` in pixels.
        board: board from :func:`make_board`.
        flags: ``cv2.calibrateCamera`` flags (default 0: estimate fx, fy, cx, cy, k1, k2, p1, p2, k3).
        min_views: minimum number of detections required (default 8).

    Returns:
        ``(K, dist, rms_px)``: K is 3x3 (pixels), dist is ``[k1, k2, p1, p2, k3]`` (dimensionless),
        rms_px is the overall RMS reprojection error in pixels.
    """
    detections = [d for d in detections if d is not None]
    if len(detections) < min_views:
        raise ValueError(f"need at least {min_views} board views for intrinsics, got {len(detections)}")
    obj = [board_corner_points(board, d["ids"]).astype(np.float32) for d in detections]
    img = [np.asarray(d["corners"], dtype=np.float32).reshape(-1, 1, 2) for d in detections]
    w, h = int(image_size[0]), int(image_size[1])
    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        [o.reshape(-1, 1, 3) for o in obj], img, (w, h), None, None, flags=flags)
    dist = np.asarray(dist, dtype=float).reshape(-1)
    dist = np.concatenate([dist, np.zeros(max(0, 5 - len(dist)))])[:5]
    return np.asarray(K, dtype=float), dist, float(rms)


def project_board_corners(T_camera_board: np.ndarray, ids: Sequence[int], K: np.ndarray, dist: Sequence[float],
                          board) -> np.ndarray:
    """Project board corners ``ids`` through ``T_camera_board`` (4x4, m) and intrinsics into pixels (N, 2)."""
    T = np.asarray(T_camera_board, dtype=float)
    pts, _ = cv2.projectPoints(board_corner_points(board, ids).reshape(-1, 1, 3), matrix_to_rotvec(T[:3, :3]),
                               T[:3, 3].reshape(3, 1), np.asarray(K, dtype=float), np.asarray(dist, dtype=float))
    return pts.reshape(-1, 2)


def reprojection_rms_px(T_camera_board: np.ndarray, corners: np.ndarray, ids: Sequence[int], K: np.ndarray,
                        dist: Sequence[float], board) -> float:
    """RMS pixel distance between detected ``corners`` (N, 2) and corners projected via ``T_camera_board``."""
    pred = project_board_corners(T_camera_board, ids, K, dist, board)
    return float(np.sqrt(np.mean(np.sum((pred - np.asarray(corners, dtype=float).reshape(-1, 2)) ** 2, axis=1))))


def board_pose(corners: np.ndarray, ids: Sequence[int], K: np.ndarray, dist: Sequence[float],
               board) -> tuple[np.ndarray, float]:
    """Board pose in a camera from detected corners (planar PnP: IPPE + Levenberg-Marquardt refinement).

    Args:
        corners: (N, 2) detected corner pixels; ids: (N,) corner ids (N >= 4, not collinear).
        K: 3x3 intrinsics (pixels); dist: plumb-bob distortion ``[k1, k2, p1, p2, k3]``.
        board: board from :func:`make_board`.

    Returns:
        ``(T_camera_board, reprojection_px)``: 4x4 transform (m) and RMS reprojection error (px).
    """
    ids = np.asarray(ids, dtype=int).reshape(-1)
    if len(ids) < 4:
        raise ValueError("board_pose needs at least 4 corners")
    obj = board_corner_points(board, ids).reshape(-1, 1, 3)
    img = np.asarray(corners, dtype=float).reshape(-1, 1, 2)
    K = np.asarray(K, dtype=float)
    dist = np.asarray(dist, dtype=float)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise ValueError("solvePnP failed")
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    T = make_transform(rotvec_to_matrix(rvec), tvec.reshape(3))
    if T[2, 3] <= 0:
        raise ValueError("solvePnP returned a board behind the camera")
    return T, reprojection_rms_px(T, corners, ids, K, dist, board)


# ----------------------------------------------------------------------------------------------
# Hand-eye (eye-in-hand) and fixed camera
# ----------------------------------------------------------------------------------------------

def hand_eye_consistency(T_base_gripper_list, T_gripper_camera, T_camera_board_list, *,
                         score_lever_m=HAND_EYE_SCORE_LEVER_M) -> dict[str, Any]:
    """Spread of ``T_base_board_i = T_base_gripper_i @ T_gripper_camera @ T_camera_board_i`` across samples.

    Returns :func:`transform_spread` keys, with ``mean`` renamed to ``T_base_board`` (4x4, m), plus
    ``score_m = translation_rms_m + score_lever_m * rotation_rms_rad`` (m).
    ``score_lever_m``: ranking coefficient in m/rad, packaged default 0.1, range 0..10; 0 ignores
    angular spread for ranking. It is not an estimated camera distance or accuracy gate.
    Example: ``hand_eye_consistency(A, X, B, score_lever_m=0.2)``. The effective coefficient is returned.
    """
    if type(score_lever_m) not in (int, float) or not math.isfinite(score_lever_m) or not 0 <= score_lever_m <= 10:
        raise ValueError("invalid_hand_eye_score_lever_m")
    X = np.asarray(T_gripper_camera, dtype=float)
    chain = [np.asarray(A, dtype=float) @ X @ np.asarray(B, dtype=float)
             for A, B in zip(T_base_gripper_list, T_camera_board_list)]
    spread = transform_spread(chain)
    spread["T_base_board"] = spread.pop("mean")
    spread["score_lever_m"] = float(score_lever_m)
    spread["score_m"] = spread["translation_rms_m"] + score_lever_m * spread["rotation_rms_rad"]
    return spread


def solve_eye_in_hand(T_base_gripper_list: Sequence[np.ndarray], T_camera_board_list: Sequence[np.ndarray], *,
                      methods: Iterable[str] | None = None,
                      score_lever_m=HAND_EYE_SCORE_LEVER_M) -> dict[str, Any]:
    """Eye-in-hand calibration with a board static in the base frame via ``cv2.calibrateHandEye``.

    Args:
        T_base_gripper_list: per-sample 4x4 pose (m) of the camera's parent link in the base frame, from FK
            of the *measured* joints at capture time.
        T_camera_board_list: per-sample 4x4 board pose (m) in the camera frame, from :func:`board_pose`.
        methods: subset of ``cv2.calibrateHandEye`` solver names to try. Default:
            ``("TSAI", "PARK", "HORAUD", "DANIILIDIS", "ANDREFF")``.
        score_lever_m: nonnegative m/rad coefficient trading angular versus translation spread for
            method selection, default 0.1 from the packaged defaults, range 0..10. E.g. 0 ranks by
            translation alone; 0.2 doubles the default angular contribution. Returned in diagnostics.

    Returns:
        dict with ``T_gripper_camera`` (4x4, m), ``method`` (winner, lowest ``score_m``),
        ``T_base_board`` (mean board pose, 4x4), ``translation_rms_m``/``translation_max_m``,
        ``rotation_rms_rad``/``rotation_max_rad`` (consistency spread of the winner), per-sample error lists,
        ``samples`` (count) and ``methods`` (per-method residuals or error string).

    Needs >= 3 samples with gripper rotations about at least two non-parallel axes.
    """
    if not hasattr(cv2, "calibrateHandEye"):
        raise RuntimeError("opencv_calibrate_hand_eye_unavailable")
    A = [np.asarray(T, dtype=float) for T in T_base_gripper_list]
    B = [np.asarray(T, dtype=float) for T in T_camera_board_list]
    if len(A) != len(B):
        raise ValueError("T_base_gripper_list and T_camera_board_list must have equal length")
    if len(A) < 3:
        raise ValueError("eye-in-hand calibration needs at least 3 samples")
    rel_rotvecs = Rotation.from_matrix([A[0][:3, :3].T @ T[:3, :3] for T in A[1:]]).as_rotvec()
    if np.linalg.svd(rel_rotvecs, compute_uv=False)[1] <= 1e-6:
        raise ValueError("degenerate rotations: need rotation about at least two non-parallel axes")
    if methods is None:
        methods = _CV2_HAND_EYE_METHODS
    R_g2b = [T[:3, :3] for T in A]
    t_g2b = [T[:3, 3].reshape(3, 1) for T in A]
    R_t2c = [T[:3, :3] for T in B]
    t_t2c = [T[:3, 3].reshape(3, 1) for T in B]
    results: dict[str, Any] = {}
    best = None
    for name in methods:
        try:
            if name not in _CV2_HAND_EYE_METHODS:
                raise ValueError(f"unknown hand-eye method {name!r}")
            R, t = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=getattr(cv2, "CALIB_HAND_EYE_" + name))
            if not (np.all(np.isfinite(R)) and np.all(np.isfinite(t))):
                raise ValueError("non-finite solution")
            X = make_transform(project_to_rotation(R), np.asarray(t, dtype=float).reshape(3))
            cons = hand_eye_consistency(A, X, B, score_lever_m=score_lever_m)
        except Exception as error:  # a failing method is recorded, others still compete
            results[name] = {"error": f"{type(error).__name__}: {error}"}
            continue
        results[name] = {k: cons[k] for k in ("translation_rms_m", "rotation_rms_rad", "score_m")}
        if best is None or cons["score_m"] < best[2]["score_m"]:
            best = (name, X, cons)
    if best is None:
        raise ValueError("every hand-eye method failed: " + json.dumps(results))
    name, X, cons = best
    out = {"T_gripper_camera": X, "method": name, "samples": len(A), "methods": results}
    out.update(cons)
    return out


def solve_fixed_camera(T_base_board: np.ndarray, T_fixedcam_board_list: Sequence[np.ndarray]) -> dict[str, Any]:
    """Extrinsic of a camera static w.r.t. the base, from a board static in the base frame.

    Args:
        T_base_board: 4x4 board pose (m) in base, e.g. ``solve_eye_in_hand(...)["T_base_board"]``.
        T_fixedcam_board_list: per-sample 4x4 board pose (m) seen by the fixed camera (:func:`board_pose`).

    Returns:
        dict with ``T_base_camera`` (mean of ``T_base_board @ inv(T_fixedcam_board_i)``, 4x4, m),
        ``translation_rms_m``/``translation_max_m``/``rotation_rms_rad``/``rotation_max_rad`` (spread of the
        per-sample estimates; measures fixed-camera PnP repeatability only, not hand-eye error) and
        ``samples``. End-to-end pixel error should be measured with :func:`reprojection_rms_px` on
        ``inv(T_base_camera) @ T_base_board``.
    """
    T_bb = np.asarray(T_base_board, dtype=float)
    estimates = [T_bb @ invert_transform(np.asarray(T, dtype=float)) for T in T_fixedcam_board_list]
    if not estimates:
        raise ValueError("fixed camera needs at least one board observation")
    spread = transform_spread(estimates)
    spread["T_base_camera"] = spread.pop("mean")
    spread["samples"] = len(estimates)
    return spread


# ----------------------------------------------------------------------------------------------
# Joint mapping (LeRobot degrees <-> URDF radians)
# ----------------------------------------------------------------------------------------------

def default_joint_mapping() -> dict[str, dict[str, float]]:
    """Identity mapping: ``q_urdf_rad = sign * radians(q_lerobot_deg) + offset_rad`` with sign 1, offset 0.

    LeRobot ``SOFollower(use_degrees=True)`` reports arm joints in degrees relative to *its own* calibration
    midpoint; whether that equals the URDF zero and direction is hardware/calibration dependent and must be
    verified on the robot (e.g. pose the arm at URDF zero and read LeRobot degrees). Fix it per joint via
    ``sign`` (+1 or -1, dimensionless) and ``offset_rad`` (rad, added after the sign).
    """
    return {name: {"sign": 1.0, "offset_rad": 0.0} for name in ARM_JOINTS}


def normalize_joint_mapping(mapping: Mapping[str, Mapping[str, float]] | None) -> dict[str, dict[str, float]]:
    """Validate a joint mapping (see :func:`default_joint_mapping`); missing joints default to identity."""
    out = default_joint_mapping()
    for name, entry in (mapping or {}).items():
        if name.startswith("_"):  # deployment-file comment keys
            continue
        if name not in out:
            raise ValueError(f"joint_mapping has unknown joint {name!r}")
        sign = float(entry.get("sign", 1.0))
        if sign not in (1.0, -1.0):
            raise ValueError(f"joint_mapping[{name!r}].sign must be +1 or -1")
        out[name] = {"sign": sign, "offset_rad": float(entry.get("offset_rad", 0.0))}
    return out


def lerobot_deg_to_urdf_rad(q_deg: Mapping[str, float], mapping: Mapping[str, Mapping[str, float]] | None = None
                            ) -> np.ndarray:
    """LeRobot joint dict (``"shoulder_pan"`` or ``"shoulder_pan.pos"`` -> deg) to URDF arm vector (5,) in rad."""
    m = normalize_joint_mapping(mapping)
    values = {k.removesuffix(".pos"): float(v) for k, v in q_deg.items()}
    return np.array([m[n]["sign"] * math.radians(values[n]) + m[n]["offset_rad"] for n in ARM_JOINTS])


def urdf_rad_to_lerobot_deg(q_rad: Sequence[float], mapping: Mapping[str, Mapping[str, float]] | None = None
                            ) -> dict[str, float]:
    """Inverse of :func:`lerobot_deg_to_urdf_rad`: URDF arm vector (5,) rad -> ``{joint: deg}``."""
    m = normalize_joint_mapping(mapping)
    q = np.asarray(q_rad, dtype=float).reshape(len(ARM_JOINTS))
    return {n: math.degrees((q[i] - m[n]["offset_rad"]) * m[n]["sign"]) for i, n in enumerate(ARM_JOINTS)}


# ----------------------------------------------------------------------------------------------
# Calibration pose generation
# ----------------------------------------------------------------------------------------------

def board_frame_from_center(center_m: Sequence[float], board, *, yaw_rad: float = 0.0) -> np.ndarray:
    """Approximate ``T_base_board`` for a board lying face-up and flat on the table.

    Args:
        center_m: board centre ``[x, y, z]`` in base (m); z is the table/print surface height.
        board: board from :func:`make_board` (size taken from it).
        yaw_rad: rotation (rad) of the board x axis about base +z (0: board x along base +x).

    The board +z axis points down into the table (OpenCV ChArUco convention, printed face up).
    """
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    R = np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])  # x=(c,s,0), y=(s,-c,0), z=(0,0,-1)
    outline = board_outline_points(board)
    half = (outline[2] - outline[0]) / 2.0
    return make_transform(R, np.asarray(center_m, dtype=float) - R @ half)


def _look_rotation(forward: np.ndarray, roll_rad: float) -> np.ndarray:
    """Camera rotation (OpenCV axes) whose +z is ``forward``; image 'down' (+y) points toward world -z
    as much as possible before applying ``roll_rad`` about the optical axis."""
    z = forward / np.linalg.norm(forward)
    ref = np.array([0.0, 0.0, -1.0])
    y = ref - z * float(ref @ z)
    if np.linalg.norm(y) < 1e-6:  # looking straight down: pick base +x as image 'down'
        y = np.array([1.0, 0.0, 0.0]) - z * z[0]
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    R = np.column_stack([x, y, z])
    return R @ rotvec_to_matrix([0.0, 0.0, roll_rad])


def generate_calibration_poses(
    T_base_board: np.ndarray,
    board,
    ik: Callable[[np.ndarray, np.ndarray], Mapping[str, Any]],
    fk: Callable[[np.ndarray], np.ndarray],
    q_seed_rad: Sequence[float],
    joint_limits_rad: Sequence[tuple[float, float]],
    *,
    T_link_camera_guess: np.ndarray | None = None,
    n_poses: int = 20,
    tilt_range_rad: tuple[float, float] = (math.radians(15.0), math.radians(35.0)),
    distance_range_m: tuple[float, float] = (0.20, 0.32),
    azimuth_span_rad: float = math.radians(75.0),
    roll_candidates: int = 12,
    joint_margin_rad: float = math.radians(5.0),
    z_floor_m: float = 0.05,
    max_ik_position_error_m: float = 0.03,
    max_view_angle_rad: float = math.radians(50.0),
    board_margin_m: float = 0.01,
    min_view_separation_rad: float = math.radians(4.0),
    fk_points: Callable[[np.ndarray], Iterable[np.ndarray]] | None = None,
    K_guess: np.ndarray | None = None,
    image_size: Sequence[int] | None = None,
    seed: int = 0,
    max_candidates: int = 60,
    rejections: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Generate joint-limit-safe arm configurations where the wrist camera sees the board from diverse views.

    Candidate camera viewpoints are sampled on a spherical cap above the board centre: tilt from the board
    normal in ``tilt_range_rad``, distance in ``distance_range_m``, azimuth within ``+-azimuth_span_rad`` of
    the direction from the board toward the robot base (keeps views reachable). The arm is 5-DOF, so the
    roll about the optical axis is not free: for every viewpoint ``roll_candidates`` rolls are tried and the
    one with the lowest IK cost (``position_error_m + 0.1 m/rad * orientation_error_rad``) is kept; roll
    diversity then follows from azimuth diversity. The kept solution is *re-checked with ``fk``* (IK may not
    hit the exact 6D target; only the achieved pose counts).

    Args:
        T_base_board: approximate 4x4 board pose in base (m), e.g. :func:`board_frame_from_center`.
        board: board from :func:`make_board`.
        ik: ``ik(T_base_link_target, q_seed_rad) -> {"q_rad", "position_error_m", "orientation_error_rad",
            "converged"}`` for the camera's parent link (adapt TCP IK by composing the fixed link->TCP
            transform). The adapter should itself respect the shrunk joint limits.
        fk: ``fk(q_rad) -> 4x4 T_base_link`` of the camera's parent link.
        q_seed_rad: current measured arm joints (5,) rad; first IK seed and start of the visit order.
        joint_limits_rad: per-joint ``(lower, upper)`` rad, in arm joint order.
        T_link_camera_guess: rough 4x4 camera pose (m, OpenCV axes) in the parent link; default identity
            (camera at the link origin looking along link +z). Only used for aiming; the solve estimates it.
        n_poses: number of poses to return (default 20; 15-25 recommended).
        tilt_range_rad: viewing tilt from the board normal, rad (default 15-35 deg); diversity of tilt is
            what makes hand-eye and intrinsics well-conditioned.
        distance_range_m: camera-to-board-centre distance, m (default 0.20-0.32).
        azimuth_span_rad: half-width of allowed azimuths around the board->base direction, rad (default 75 deg).
        roll_candidates: number of evenly spaced rolls over 360 deg (random phase per viewpoint) tried per
            viewpoint (default 12). Each costs one ``ik`` call.
        joint_margin_rad: joint limits are shrunk by this margin on both sides, rad (default 5 deg).
        z_floor_m: minimum base-frame z (m) of the camera and of every point from ``fk_points`` (default 0.05,
            measured from the base frame origin; set it to table height + clearance).
        max_ik_position_error_m: reject IK results whose reported position error exceeds this (m, default 0.03).
        max_view_angle_rad: reject achieved poses where the optical axis is more than this from the board
            normal (rad, default 50 deg) - steep views detect poorly.
        board_margin_m: achieved optical axis must hit the board rectangle shrunk by this margin (m, default 0.01).
        min_view_separation_rad: reject a pose whose viewing direction is within this angle of an accepted
            pose with similar roll (rad, default 4 deg) to keep samples diverse.
        fk_points: optional ``fk_points(q_rad) -> iterable of 3-vectors or 4x4 transforms`` (base frame, m),
            e.g. all link origins, checked against ``z_floor_m``.
        K_guess, image_size: optional approximate 3x3 intrinsics (px) and ``(width, height)``; when given,
            every outer board corner must project inside the image.
        seed: RNG seed (deterministic output for identical inputs).
        max_candidates: number of candidate viewpoints tried before giving up (default 60).
        rejections: optional dict filled with per-reason rejection counts (``ik_joint_limits``,
            ``ik_position_error``, ``view_angle``, ``axis_misses_board``, ``z_floor``, ``board_outside_image``,
            ``too_similar``) so the caller can report why fewer than ``n_poses`` were found.

    Returns:
        List (length <= ``n_poses``; fewer only if not enough candidates passed) ordered by greedy
        nearest-neighbour in joint space from ``q_seed_rad``. Each entry: ``q_rad`` (5,), ``T_base_link``,
        ``T_base_camera`` (achieved, from fk), ``view_angle_rad`` (optical axis vs board normal),
        ``distance_m`` (camera to optical-axis/board intersection), ``ik_position_error_m``,
        ``ik_orientation_error_rad``.
    """
    T_bb = np.asarray(T_base_board, dtype=float)
    T_lc = np.eye(4) if T_link_camera_guess is None else np.asarray(T_link_camera_guess, dtype=float)
    T_cl = invert_transform(T_lc)
    limits = np.asarray(joint_limits_rad, dtype=float).reshape(-1, 2)
    lo, hi = limits[:, 0] + joint_margin_rad, limits[:, 1] - joint_margin_rad
    if np.any(lo >= hi):
        raise ValueError("joint_margin_rad leaves an empty joint range")
    outline = board_outline_points(board)
    size = outline[2, :2]
    center_board = np.array([size[0] / 2, size[1] / 2, 0.0])
    center = (T_bb @ np.append(center_board, 1.0))[:3]
    normal_up = -T_bb[:3, 2]  # toward the viewer (board +z points into the table)
    toward_base = -center.copy()
    toward_base -= normal_up * float(toward_base @ normal_up)
    if np.linalg.norm(toward_base) < 1e-6:
        toward_base = T_bb[:3, 0] - normal_up * float(T_bb[:3, 0] @ normal_up)
    e1 = toward_base / np.linalg.norm(toward_base)
    e2 = np.cross(normal_up, e1)
    rng = np.random.default_rng(seed)
    n_roll = max(1, int(roll_candidates))

    def candidate(k: int):
        # stratified: golden-ratio azimuth sequence + jittered tilt/distance/roll for coverage
        u = (k * 0.6180339887498949 + rng.uniform(0, 0.1)) % 1.0
        az = (2 * u - 1) * azimuth_span_rad
        tilt = rng.uniform(*tilt_range_rad)
        dist = rng.uniform(*distance_range_m)
        phase = rng.uniform(0, 2 * math.pi)
        direction = (math.cos(tilt) * normal_up
                     + math.sin(tilt) * (math.cos(az) * e1 + math.sin(az) * e2))
        pos = center + dist * direction
        return [make_transform(_look_rotation(center - pos, phase + 2 * math.pi * r / n_roll), pos)
                for r in range(n_roll)]

    def view_check(T_base_camera: np.ndarray, q: np.ndarray):
        axis = T_base_camera[:3, 2]
        origin = T_base_camera[:3, 3]
        view_angle = math.acos(max(-1.0, min(1.0, float(axis @ -normal_up))))
        if view_angle > max_view_angle_rad:
            return None, "view_angle"
        denom = float(axis @ normal_up)
        if denom >= -1e-9:
            return None, "axis_misses_board"
        s = float((center - origin) @ normal_up) / denom
        if s <= 0:
            return None, "axis_misses_board"
        hit_board = (invert_transform(T_bb) @ np.append(origin + s * axis, 1.0))[:3]
        if not (board_margin_m <= hit_board[0] <= size[0] - board_margin_m
                and board_margin_m <= hit_board[1] <= size[1] - board_margin_m):
            return None, "axis_misses_board"
        if origin[2] < z_floor_m:
            return None, "z_floor"
        if fk_points is not None:
            for p in fk_points(q):
                p = np.asarray(p, dtype=float)
                z = p[2, 3] if p.shape == (4, 4) else p.reshape(-1)[2]
                if z < z_floor_m:
                    return None, "z_floor"
        if K_guess is not None and image_size is not None:
            T_cb = invert_transform(T_base_camera) @ T_bb
            pts = (T_cb @ np.c_[outline, np.ones(4)].T)[:3].T
            if np.any(pts[:, 2] <= 0):
                return None, "board_outside_image"
            uv = (np.asarray(K_guess, dtype=float) @ (pts / pts[:, 2:3]).T)[:2].T
            w, h = image_size
            if np.any(uv < 0) or np.any(uv[:, 0] >= w) or np.any(uv[:, 1] >= h):
                return None, "board_outside_image"
        return {"view_angle_rad": view_angle, "distance_m": s}, None

    accepted: list[dict[str, Any]] = []
    rejected = rejections if rejections is not None else {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    q_seed = np.asarray(q_seed_rad, dtype=float).reshape(-1)
    for k in range(max_candidates):
        if len(accepted) >= n_poses:
            break
        seed_q = accepted[-1]["q_rad"] if accepted else q_seed
        result, best_cost = None, math.inf
        for T_target_cam in candidate(k):
            r = ik(T_target_cam @ T_cl, seed_q)
            cost = (float(r.get("position_error_m", math.inf))
                    + HAND_EYE_SCORE_LEVER_M * float(r.get("orientation_error_rad", math.inf)))
            if cost < best_cost:
                result, best_cost = r, cost
        if result is None:
            reject("ik_failed")
            continue
        q = np.asarray(result["q_rad"], dtype=float).reshape(-1)
        if not np.all(np.isfinite(q)) or np.any(q < lo - 1e-9) or np.any(q > hi + 1e-9):
            reject("ik_joint_limits")
            continue
        if float(result.get("position_error_m", np.inf)) > max_ik_position_error_m:
            reject("ik_position_error")
            continue
        T_link = np.asarray(fk(q), dtype=float)
        T_cam = T_link @ T_lc
        info, reason = view_check(T_cam, q)
        if info is None:
            reject(reason)
            continue
        direction = T_cam[:3, 3] - center
        direction /= np.linalg.norm(direction)
        too_close = False
        for other in accepted:
            od = other["T_base_camera"][:3, 3] - center
            od /= np.linalg.norm(od)
            rel_roll = rotation_angle(other["T_base_camera"][:3, :3].T @ T_cam[:3, :3])
            if math.acos(max(-1.0, min(1.0, float(direction @ od)))) < min_view_separation_rad \
                    and rel_roll < 2 * min_view_separation_rad:
                too_close = True
                break
        if too_close:
            reject("too_similar")
            continue
        accepted.append({"q_rad": q, "T_base_link": T_link, "T_base_camera": T_cam,
                         "ik_position_error_m": float(result.get("position_error_m", float("nan"))),
                         "ik_orientation_error_rad": float(result.get("orientation_error_rad", float("nan"))),
                         **info})
    # greedy nearest-neighbour visiting order in joint space (max-abs joint delta)
    ordered, current, remaining = [], q_seed, list(accepted)
    while remaining:
        i = int(np.argmin([np.max(np.abs(p["q_rad"] - current)) for p in remaining]))
        ordered.append(remaining.pop(i))
        current = ordered[-1]["q_rad"]
    return ordered


def path_floor_violation(kin, model, q_start: Sequence[float], q_end: Sequence[float], z_floor_m: float,
                         steps: int | None = None, *, step_rad: float = math.radians(2.0)) -> dict[str, Any] | None:
    """First interpolated joint configuration along ``q_start -> q_end`` where a moving link origin goes below
    ``z_floor_m`` (m); ``None`` if the whole segment is clear."""
    q_from, q_to = np.asarray(q_start, dtype=float), np.asarray(q_end, dtype=float)
    n_steps = max(1, int(steps) if steps is not None else math.ceil(float(np.max(np.abs(q_to - q_from))) / step_rad))
    base_links = {getattr(kin, "BASE_LINK", "base_link"), "shoulder_link"}
    for k in range(n_steps + 1):
        q = q_from + (q_to - q_from) * k / n_steps
        for name, T in kin.fk_all(model, q).items():
            if name not in base_links and float(T[2, 3]) < z_floor_m:
                return {"q_rad": q.tolist(), "link": name, "z_m": float(T[2, 3])}
    return None


# ----------------------------------------------------------------------------------------------
# Calibration file (schema ni_rua.so101.camera_calibration.v1)
# ----------------------------------------------------------------------------------------------

def _matrix(value, shape: tuple[int, int], name: str) -> np.ndarray:
    M = np.asarray(value, dtype=float)
    if M.shape != shape:
        raise ValueError(f"{name} must be {shape[0]}x{shape[1]}, got shape {M.shape}")
    if not np.all(np.isfinite(M)):
        raise ValueError(f"{name} contains non-finite values")
    return M


def _rigid(value, name: str) -> np.ndarray:
    T = _matrix(value, (4, 4), name)
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-9):
        raise ValueError(f"{name} last row must be [0, 0, 0, 1]")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or np.linalg.det(R) <= 0:
        raise ValueError(f"{name} rotation block is not a proper rotation")
    return T


def _intrinsics(value, name: str) -> np.ndarray:
    K = _matrix(value, (3, 3), name)
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]) or K[1, 0] != 0 or K[2, 0] != 0:
        raise ValueError(f"{name} is not a valid pinhole intrinsics matrix")
    return K


def _nonneg(value, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return v


def _validate_camera_common(name: str, cam: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(cam)
    out["K"] = _intrinsics(cam["K"], f"cameras.{name}.K")
    dist = np.asarray(cam["distortion"], dtype=float).reshape(-1)
    if dist.shape != (5,) or not np.all(np.isfinite(dist)):
        raise ValueError(f"cameras.{name}.distortion must be 5 finite values [k1, k2, p1, p2, k3]")
    out["distortion"] = dist
    size = [int(v) for v in cam["image_size"]]
    if len(size) != 2 or min(size) <= 0:
        raise ValueError(f"cameras.{name}.image_size must be [width, height] > 0")
    out["image_size"] = size
    if cam.get("distortion_model", DISTORTION_MODEL) != DISTORTION_MODEL:
        raise ValueError(f"cameras.{name}.distortion_model must be {DISTORTION_MODEL!r}")
    out["distortion_model"] = DISTORTION_MODEL
    out["optical_axes"] = OPTICAL_AXES
    if "intrinsics_rms_px" in cam:
        out["intrinsics_rms_px"] = _nonneg(cam["intrinsics_rms_px"], f"cameras.{name}.intrinsics_rms_px")
    if "intrinsics_source" in cam and not isinstance(cam["intrinsics_source"], str):
        raise ValueError(f"cameras.{name}.intrinsics_source must be a string")
    return out


def validate_calibration_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a calibration document; returns a copy with matrices as numpy arrays.

    Raises ``ValueError`` on wrong schema/frame, non-3x3/4x4 or non-finite matrices, non-rigid transforms,
    invalid intrinsics, missing residuals, or no cameras.
    """
    if document.get("schema") != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA!r}")
    if document.get("reference_frame_id") != REFERENCE_FRAME_ID:
        raise ValueError(f"reference_frame_id must be {REFERENCE_FRAME_ID!r}")
    out = dict(document)
    out["board"] = validate_board_spec(document["board"])
    cameras = document.get("cameras") or {}
    if not cameras or set(cameras) - {"wrist", "fixed"}:
        raise ValueError("cameras must be a non-empty subset of {'wrist', 'fixed'}")
    out["cameras"] = {}
    if "wrist" in cameras:
        cam = _validate_camera_common("wrist", cameras["wrist"])
        if cam.get("arm", WRIST_ARM) != WRIST_ARM:
            raise ValueError("cameras.wrist.arm must be 'right'")
        cam["arm"] = WRIST_ARM
        cam["link_name"] = str(cam.get("link_name", WRIST_LINK_NAME))
        cam["T_link_camera"] = _rigid(cameras["wrist"]["T_link_camera"], "cameras.wrist.T_link_camera")
        res = cameras["wrist"]["hand_eye_residual"]
        cam["hand_eye_residual"] = {
            "translation_m": _nonneg(res["translation_m"], "hand_eye_residual.translation_m"),
            "rotation_rad": _nonneg(res["rotation_rad"], "hand_eye_residual.rotation_rad")}
        cam["intrinsics_rms_px"] = _nonneg(cameras["wrist"]["intrinsics_rms_px"], "cameras.wrist.intrinsics_rms_px")
        out["cameras"]["wrist"] = cam
    if "fixed" in cameras:
        cam = _validate_camera_common("fixed", cameras["fixed"])
        cam["T_base_camera"] = _rigid(cameras["fixed"]["T_base_camera"], "cameras.fixed.T_base_camera")
        cam["extrinsic_residual"] = {"reprojection_px": _nonneg(
            cameras["fixed"]["extrinsic_residual"]["reprojection_px"], "extrinsic_residual.reprojection_px")}
        out["cameras"]["fixed"] = cam
    out["samples"] = int(document.get("samples", 0))
    if out["samples"] < 0:
        raise ValueError("samples must be >= 0")
    return out


def fit_pixel_mapping(source_px: Sequence[Sequence[float]], target_px: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Per-axis scale and offset between two capture modes of one camera: ``target = scale * source + offset``.

    ``source_px`` / ``target_px``: (N, 2) pixels (original, distorted images) of the same N physical points seen in
    both modes (e.g. ChArUco corners of one static board, matched by id), N >= 3. A UVC camera usually delivers a
    lower mode by scaling (and possibly cropping) the same sensor pixels, so the map is affine per axis; a mode made
    by a different optical path or a non-affine ISP warp shows up as a large residual. Returns
    ``dict(scale=[sx, sy], offset_px=[ox, oy], rms_px, max_px, points)``: least-squares scale (dimensionless) and
    offset (target px), and the RMS / largest residual (target px) over the points. Example: 640x480 -> 1280x960
    full-sensor scaling gives scale [2, 2] and offset [0.5, 0.5] (pixel centres)."""
    source = np.asarray(source_px, dtype=float).reshape(-1, 2)
    target = np.asarray(target_px, dtype=float).reshape(-1, 2)
    if len(source) != len(target) or len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("fit_pixel_mapping needs the same >= 3 finite points in both modes")
    scale, offset, residual = [], [], []
    for axis in range(2):
        A = np.column_stack([source[:, axis], np.ones(len(source))])
        (s_axis, o_axis), *_ = np.linalg.lstsq(A, target[:, axis], rcond=None)
        scale.append(float(s_axis))
        offset.append(float(o_axis))
        residual.append(target[:, axis] - A @ np.array([s_axis, o_axis]))
    error = np.hypot(*residual)
    return dict(scale=scale, offset_px=offset, rms_px=float(np.sqrt(np.mean(error ** 2))), max_px=float(error.max()),
                points=int(len(source)))


def remap_camera_intrinsics(camera: Mapping[str, Any], mapping: Mapping[str, Any], image_size: Sequence[int],
                            *, from_size: Sequence[int]) -> dict[str, Any]:
    """One calibration camera entry for another capture mode of the same camera (``fit_pixel_mapping`` result).

    ``K' = [[sx fx, 0, sx cx + ox], [0, sy fy, sy cy + oy], [0, 0, 1]]`` (px); the OpenCV distortion is applied in
    normalised coordinates, so it is unchanged under a per-axis pixel scale and offset; the extrinsic
    (``T_base_camera`` / ``T_link_camera``, m) is the physical camera pose and is kept; ``image_size`` becomes
    ``image_size`` ([width, height] px). Pixel residuals are carried into the new pixel grid (multiplied by the mean
    scale) and increased by the mapping fit's RMS in quadrature; metric and angular residuals are kept.
    ``intrinsics_source`` records the derivation (``from_size``: the old [width, height] px). The caller writes the
    result to a new calibration file; nothing is changed in place."""
    sx, sy = (float(v) for v in mapping["scale"])
    ox, oy = (float(v) for v in mapping["offset_px"])
    K = np.asarray(camera["K"], dtype=float)
    out = {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in dict(camera).items()}
    out["K"] = [[sx * K[0, 0], 0.0, sx * K[0, 2] + ox], [0.0, sy * K[1, 1], sy * K[1, 2] + oy], [0.0, 0.0, 1.0]]
    out["image_size"] = [int(image_size[0]), int(image_size[1])]
    mean, fit = (sx + sy) / 2.0, float(mapping["rms_px"])
    if out.get("intrinsics_rms_px") is not None:
        out["intrinsics_rms_px"] = float(math.hypot(float(out["intrinsics_rms_px"]) * mean, fit))
    if isinstance(out.get("extrinsic_residual"), Mapping):
        out["extrinsic_residual"] = {"reprojection_px": float(math.hypot(
            float(out["extrinsic_residual"]["reprojection_px"]) * mean, fit))}
    out["intrinsics_source"] = (f"remapped from {int(from_size[0])}x{int(from_size[1])}: scale {sx:.5f}, {sy:.5f}; "
                                f"offset {ox:.2f}, {oy:.2f} px; fit rms {fit:.3f} px on {int(mapping['points'])} points")
    return out


def check_residual_gates(cameras: Mapping[str, Mapping[str, Any]], thresholds: Mapping[str, float | None]
                         ) -> list[str]:
    """Compare recorded residuals with thresholds; returns human-readable failures (empty = pass).

    Threshold keys (``None``/missing = not gated): ``max_intrinsics_rms_px`` (px, every camera),
    ``max_hand_eye_translation_m`` (m, wrist), ``max_hand_eye_rotation_rad`` (rad, wrist),
    ``max_fixed_reprojection_px`` (px, fixed camera end-to-end reprojection).
    """
    failures = []

    def gate(value, key, label):
        limit = thresholds.get(key)
        if limit is not None and value is not None and not (float(value) <= float(limit)):
            failures.append(f"{label}={float(value):.6g} > {key}={float(limit):.6g}")

    for name, cam in cameras.items():
        gate(cam.get("intrinsics_rms_px"), "max_intrinsics_rms_px", f"{name}.intrinsics_rms_px")
    if "wrist" in cameras:
        res = cameras["wrist"]["hand_eye_residual"]
        gate(res["translation_m"], "max_hand_eye_translation_m", "wrist.hand_eye_residual.translation_m")
        gate(res["rotation_rad"], "max_hand_eye_rotation_rad", "wrist.hand_eye_residual.rotation_rad")
    if "fixed" in cameras:
        gate(cameras["fixed"]["extrinsic_residual"]["reprojection_px"], "max_fixed_reprojection_px",
             "fixed.extrinsic_residual.reprojection_px")
    return failures


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _joint_calibration(value: Mapping[str, Any]) -> dict[str, Any]:
    """Optional touch-off/joint calibration record kept with the camera calibration (m, rad, px)."""
    if not isinstance(value, Mapping):
        raise ValueError("joint_calibration must be an object")
    table = value.get("table_z_m")
    if type(table) not in (int, float) or not math.isfinite(table):
        raise ValueError("joint_calibration.table_z_m must be a finite number (m)")
    out = {key: _jsonable(item) for key, item in value.items()}
    out["table_z_m"] = float(table)
    return out


def build_calibration_document(*, board: Mapping[str, Any], cameras: Mapping[str, Mapping[str, Any]],
                               samples: int, created_at: str | None = None,
                               joint_calibration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Assemble a schema ``ni_rua.so101.camera_calibration.v1`` document (validated, JSON-serialisable).

    ``cameras["wrist"]`` needs ``K``, ``distortion``, ``image_size``, ``T_link_camera``, ``intrinsics_rms_px``,
    ``hand_eye_residual`` ``{"translation_m", "rotation_rad"}``; ``cameras["fixed"]`` needs ``K``,
    ``distortion``, ``image_size``, ``T_base_camera``, ``extrinsic_residual`` ``{"reprojection_px"}`` and may
    carry ``intrinsics_rms_px`` and ``intrinsics_source`` (str: ``"handheld"`` after the console hand-held
    solve; absent in older files). Constant fields (distortion model, optical axes, arm, link) are filled in.
    """
    order = {"wrist": ("K", "distortion", "distortion_model", "image_size", "optical_axes", "arm", "link_name",
                       "T_link_camera", "intrinsics_rms_px", "intrinsics_source", "hand_eye_residual"),
             "fixed": ("K", "distortion", "distortion_model", "image_size", "optical_axes", "T_base_camera",
                       "intrinsics_rms_px", "intrinsics_source", "extrinsic_residual")}
    document = {
        "schema": SCHEMA,
        "reference_frame_id": REFERENCE_FRAME_ID,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "board": validate_board_spec(board),
        "cameras": {},
        "samples": int(samples),
    }
    validated = validate_calibration_document({**document, "cameras": cameras})
    for name, cam in validated["cameras"].items():
        document["cameras"][name] = {k: _jsonable(cam[k]) for k in order[name] if k in cam}
    if joint_calibration is not None:
        # Extrinsics are only valid together with the deployment joint offsets recorded here.
        document["joint_calibration"] = _joint_calibration(joint_calibration)
    return document


def write_calibration_file(path: str | os.PathLike, *, board: Mapping[str, Any],
                           cameras: Mapping[str, Mapping[str, Any]], samples: int,
                           thresholds: Mapping[str, float | None] | None = None, force: bool = False,
                           overwrite: bool = False, created_at: str | None = None,
                           joint_calibration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate, gate and atomically write ``camera_calibration.json``.

    Args:
        path: destination file.
        board, cameras, samples, created_at: see :func:`build_calibration_document`.
        thresholds: residual limits, see :func:`check_residual_gates` (``None`` = no gating).
        force: write even when residuals exceed thresholds (the true residuals are still recorded).
        overwrite: allow replacing an existing file (default False: refuse).

    Returns the written document. Raises :class:`CalibrationGateError` when gated and not forced,
    ``FileExistsError`` when the file exists and ``overwrite`` is False, ``ValueError`` on invalid content.
    """
    document = build_calibration_document(board=board, cameras=cameras, samples=samples, created_at=created_at,
                                          joint_calibration=joint_calibration)
    failures = check_residual_gates(document["cameras"], thresholds or {})
    if failures and not force:
        raise CalibrationGateError(failures)
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2)
            handle.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return document


def read_calibration_file(path: str | os.PathLike) -> dict[str, Any]:
    """Read and validate a calibration file; matrices (K, distortion, T_*) are returned as numpy arrays."""
    with open(path, "r", encoding="utf-8") as handle:
        return validate_calibration_document(json.load(handle))
