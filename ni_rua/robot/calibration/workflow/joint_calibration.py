"""Joint SO-101 calibration: joint zero offsets + wrist hand-eye + board pose, one least-squares problem.

Approach after ROS ``robot_calibration`` (Fetch Robotics): minimise ChArUco corner reprojection error in
the wrist camera over every sample, with the kinematic chain evaluated at ``q_measured + joint_offset``,
plus physical priors:

* the board lies flat on the table, its top ``board_thickness_m`` above ``table_z_m``;
* joint offsets are small (Gaussian prior ``offset_prior_rad``).

Gauge freedoms are not estimated, because board-only data cannot observe them; they are reported:

* ``shoulder_pan``: rotating the whole arm about base z is indistinguishable from moving the board
  (needs an external reference such as a known touch point);
* ``wrist_roll``: the last joint before the camera parent link, absorbed by the hand-eye transform.

Units: metres, radians, pixels. Pure numpy/scipy/cv2; no hardware access.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares

from . import charuco as cal
from .defaults import calibration_default

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
DEFAULT_ESTIMATED = ("shoulder_lift", "elbow_flex", "wrist_flex")


@dataclass
class Sample:
    """One capture: measured arm joints (rad, URDF convention) and the wrist-camera ChArUco detection."""
    q_rad: np.ndarray
    corners_px: np.ndarray  # (N, 2)
    ids: np.ndarray  # (N,)
    board: int = 0  # index into T_base_board0 when the board was placed more than once (one static pose each)


@dataclass
class JointCalibrationResult:
    joint_offsets_rad: dict[str, float]
    T_gripper_camera: np.ndarray
    T_base_board: np.ndarray
    reprojection_rms_px: float
    initial_reprojection_rms_px: float
    board_top_z_m: float
    board_tilt_rad: float
    estimated_joints: tuple[str, ...]
    unobservable_joints: tuple[str, ...]
    per_sample_rms_px: list[float] = field(default_factory=list)
    focal_scale: float = 1.0
    success: bool = True
    message: str = ""
    table_z_m: float | None = None
    touch_residuals_m: list[float] = field(default_factory=list)
    T_base_boards: list[np.ndarray] = field(default_factory=list)
    board_top_z_list_m: list[float] = field(default_factory=list)
    max_board_corner_height_error_m: float = 0.0
    T_base_fixed: np.ndarray | None = None
    fixed_reprojection_rms_px: float | None = None
    effective_solver_settings: dict[str, Any] = field(default_factory=dict)
    nfev: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"joint_offsets_rad": self.joint_offsets_rad,
                "T_gripper_camera": self.T_gripper_camera.tolist(), "T_base_board": self.T_base_board.tolist(),
                "reprojection_rms_px": self.reprojection_rms_px,
                "initial_reprojection_rms_px": self.initial_reprojection_rms_px,
                "board_top_z_m": self.board_top_z_m, "board_tilt_rad": self.board_tilt_rad,
                "estimated_joints": list(self.estimated_joints), "unobservable_joints": list(self.unobservable_joints),
                "per_sample_rms_px": self.per_sample_rms_px, "focal_scale": self.focal_scale,
                "success": self.success, "message": self.message, "table_z_m": self.table_z_m,
                "touch_residuals_m": self.touch_residuals_m,
                "T_base_boards": [T.tolist() for T in self.T_base_boards],
                "board_top_z_list_m": self.board_top_z_list_m,
                "max_board_corner_height_error_m": self.max_board_corner_height_error_m,
                "T_base_fixed": None if self.T_base_fixed is None else self.T_base_fixed.tolist(),
                "fixed_reprojection_rms_px": self.fixed_reprojection_rms_px,
                "effective_solver_settings": self.effective_solver_settings, "nfev": self.nfev}


def _pose(vec6: np.ndarray) -> np.ndarray:
    return cal.make_transform(cal.rotvec_to_matrix(vec6[:3]), vec6[3:])


def _vec(T: np.ndarray) -> np.ndarray:
    return np.concatenate([cal.matrix_to_rotvec(T[:3, :3]), T[:3, 3]])


def _perturb(T0: np.ndarray, delta6: np.ndarray) -> np.ndarray:
    return np.asarray(T0, float) @ cal.make_transform(cal.rotvec_to_matrix(delta6[:3]), delta6[3:])


def _robust_pixel_residual(r: np.ndarray, f_scale: float) -> np.ndarray:
    """Smooth pseudo-Huber / soft-L1 scaling applied only to vision reprojection residuals."""
    r = np.asarray(r, float)
    return r * np.sqrt(2.0 / (np.sqrt(1.0 + (r / f_scale) ** 2) + 1.0))


def _normalize_table_normal(table_normal: Sequence[float] | None) -> np.ndarray:
    if table_normal is None:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    n_vec = np.asarray(table_normal, dtype=float).reshape(3)
    norm = float(np.linalg.norm(n_vec))
    if norm <= 0.0 or n_vec[2] <= 0.0:
        raise ValueError("table_normal must be a non-zero 3-vector with positive z component")
    return n_vec / norm


def board_corner_height_error_m(T_base_boards: np.ndarray | Sequence[np.ndarray], board, table_z_m: float,
                                board_thickness_m: float, *, table_normal: Sequence[float] | None = None,
                                table_xy_origin: Sequence[float] = (0.0, 0.0)) -> float:
    """Maximum |corner_z - (table_z_at_corner + board_thickness_m)| (m) across all 4 outer corners of every board."""
    boards = [np.asarray(T, float) for T in ([T_base_boards] if np.ndim(T_base_boards) == 2 else T_base_boards)]
    if not boards:
        return 0.0
    outline = cal.board_outline_points(board)
    n_vec = _normalize_table_normal(table_normal)
    x0, y0 = (float(v) for v in np.asarray(table_xy_origin, dtype=float).reshape(2))
    errors = []
    for B in boards:
        corners = (B[:3, :3] @ outline.T).T + B[:3, 3]
        expected_z = float(table_z_m) - (n_vec[0] * (corners[:, 0] - x0) + n_vec[1] * (corners[:, 1] - y0)) / n_vec[2] + float(board_thickness_m)
        errors.extend(np.abs(corners[:, 2] - expected_z).tolist())
    return float(max(errors, default=0.0))


def solve_joint_calibration(samples: Sequence[Sample], board, K: np.ndarray, dist: Sequence[float],
                            fk_link: Callable[[np.ndarray], np.ndarray], *,
                            T_gripper_camera0: np.ndarray, T_base_board0: np.ndarray,
                            table_z_m: float, board_thickness_m: float,
                            estimated_joints: Sequence[str] = DEFAULT_ESTIMATED,
                            offset_prior_rad: float = 0.2, plane_sigma_m: float = 0.002,
                            pixel_sigma: float = 0.5, estimate_focal: bool = False,
                            touches: Sequence[tuple] = (), fk_all: Callable[[np.ndarray], dict] | None = None,
                            touch_sigma_m: float = 0.001,
                            table_prior_sigma_m: float | None = None,
                            initial_offsets_rad: Mapping[str, float] | Sequence[float] | None = None,
                            lowest_point_fn: Callable[[np.ndarray, float], dict] | None = None,
                            table_normal: Sequence[float] | None = None,
                            table_xy_origin: Sequence[float] = (0.0, 0.0),
                            freeze_gripper_camera: bool = False,
                            fixed_samples: Sequence[Sample] = (),
                            fixed_K: np.ndarray | None = None,
                            fixed_dist: Sequence[float] | None = None,
                            T_base_fixed0: np.ndarray | None = None,
                            max_nfev: int = calibration_default("joint_solver_max_nfev"),
                            pixel_robust_scale: float = calibration_default("joint_pixel_robust_scale"),
                            ftol: float = calibration_default("joint_solver_ftol"),
                            xtol: float = calibration_default("joint_solver_xtol"),
                            gtol: float = calibration_default("joint_solver_gtol"),
                            x_scale: float = calibration_default("joint_solver_x_scale")) -> JointCalibrationResult:
    """Joint least-squares solve.

    Args:
        samples: captures with measured q (rad) and wrist ChArUco detections (px).
        board: ChArUco board from ``so101_calibration.make_board``.
        K, dist: wrist intrinsics (px) and plumb-bob distortion.
        fk_link: ``q_rad -> T_base_camera_parent_link`` (4x4, m).
        T_gripper_camera0, T_base_board0: initial guesses (e.g. the classic hand-eye solve). ``T_base_board0``
            may be a list of 4x4 poses, one per board placement; ``Sample.board`` indexes it and the plane
            prior applies to every placement. The result's ``T_base_board`` is placement 0.
        table_z_m: table surface height in so101_base (m) at ``table_xy_origin``.
        board_thickness_m: board top above the table (m).
        estimated_joints: joints whose zero offsets are estimated; the rest stay 0 (see module docstring).
        offset_prior_rad: 1-sigma Gaussian prior on each total offset from URDF mechanical zero (rad).
        plane_sigma_m: 1-sigma tolerance of the "board top on table + thickness" prior (m).
        pixel_sigma: 1-sigma corner noise (px) used to weight reprojection residuals.
        touches: ``(q_rad, link_name, point_in_link_m[, jaw_rad])`` per table contact:
            the contact point is the lowest gripper mesh vertex at contact. Each adds the
            camera-independent constraint "contact point z = table_z_at_xy", which breaks the
            joint-offset / board-height ambiguity of board-only data.
        fk_all: ``q_rad -> {link: T_base_link}``; required with ``touches`` when ``lowest_point_fn`` is None.
        touch_sigma_m: 1-sigma tolerance of a touch constraint (m).
        estimate_focal: also refine one focal-length scale (fx, fy together).
        table_prior_sigma_m: when set (m), the table height is estimated too, with a Gaussian prior of this
            1-sigma centred on ``table_z_m``; ``None`` keeps it fixed.
        initial_offsets_rad: active joint offsets (rad) already included in ``Sample.q_rad`` / ``touches``;
            when provided, the Gaussian prior penalizes the total offset ``initial_offsets_rad + delta`` from
            URDF mechanical zero rather than incremental changes.
        lowest_point_fn: optional ``(q_rad, jaw_rad) -> {"z_m": float, ...}`` that dynamically re-evaluates
            the lowest gripper mesh vertex as joint offsets vary during optimization.
        table_normal: optional upward unit normal ``[nx, ny, nz]`` (with ``nz > 0``) of a tilted table plane.
        table_xy_origin: ``(x0, y0)`` reference point (m) in ``so101_base`` where table height equals ``table_z_m``.
        freeze_gripper_camera: when True, keep ``T_gripper_camera`` fixed at ``T_gripper_camera0``.
        fixed_samples, fixed_K, fixed_dist, T_base_fixed0: optional fixed-camera ChArUco observations and
            initial extrinsic pose ``T_base_fixed0`` (4x4, m) for simultaneous dual-camera bundle adjustment.
        max_nfev: positive objective-evaluation budget (count, default 400, range 1..100000),
            excluding numerical-Jacobian evaluations; e.g. 800 allows a longer offline solve.
        pixel_robust_scale: positive soft-L1 transition scale on normalized visual residuals
            (dimensionless, default 3, range (0,100]); e.g. 2 downweights outliers earlier.
            Residuals already divide by pixel_sigma and the existing per-view corner normalization.
        ftol, xtol, gtol: relative cost, parameter-step and scaled-gradient termination tolerances
            (dimensionless, defaults 1e-8, range (1e-15,1]); e.g. 1e-9 requires tighter convergence.
        x_scale: uniform optimization-coordinate scale (dimensionless multiplier, default 1,
            range (0,1000]); e.g. 2 rescales the trust-region coordinates uniformly. The existing
            trf method, two-point numerical Jacobian and linear outer loss remain unchanged.
            All effective controls and the actual nfev are returned for offline audit.
    """
    if type(max_nfev) is not int or not 1 <= max_nfev <= 100000:
        raise ValueError("invalid_joint_solver_max_nfev")
    for name, value, lower, upper in (("pixel_robust_scale", pixel_robust_scale, 0, 100),
                                     ("ftol", ftol, 1e-15, 1), ("xtol", xtol, 1e-15, 1),
                                     ("gtol", gtol, 1e-15, 1), ("x_scale", x_scale, 0, 1000)):
        if type(value) not in (int, float) or not np.isfinite(value) or not lower < value <= upper:
            raise ValueError("invalid_joint_solver_" + name)
    effective_solver_settings = dict(method="trf", jac="2-point", loss="linear", max_nfev=max_nfev,
                                     pixel_robust_scale=float(pixel_robust_scale), ftol=float(ftol),
                                     xtol=float(xtol), gtol=float(gtol), x_scale=float(x_scale),
                                     diff_step=None, tr_solver=None,
                                     visual_corner_normalization="sqrt_count_when_touches")
    K = np.asarray(K, float); dist = np.asarray(dist, float)
    T_gripper_camera0 = np.asarray(T_gripper_camera0, float)
    index = [JOINTS.index(name) for name in estimated_joints]
    outline = cal.board_outline_points(board)  # board corners in board frame (m)
    objects = [cal.board_corner_points(board, s.ids) for s in samples]
    n = len(index)
    if initial_offsets_rad is None:
        init_vec = np.zeros(n, dtype=float)
    elif isinstance(initial_offsets_rad, Mapping):
        init_vec = np.array([float(initial_offsets_rad.get(name, 0.0)) for name in estimated_joints], dtype=float)
    else:
        seq = np.asarray(initial_offsets_rad, dtype=float).reshape(-1)
        init_vec = np.array([float(seq[i]) for i in index], dtype=float) if len(seq) == len(JOINTS) else seq[:n].astype(float)
    n_vec = _normalize_table_normal(table_normal)
    origin_x, origin_y = (float(v) for v in np.asarray(table_xy_origin, dtype=float).reshape(2))
    boards0 = [np.asarray(T, float) for T in ([T_base_board0] if np.ndim(T_base_board0) == 2 else T_base_board0)]
    m = len(boards0)
    if any(not 0 <= s.board < m for s in samples):
        raise ValueError("sample board index outside T_base_board0")
    use_fixed = bool(fixed_samples) and T_base_fixed0 is not None
    if use_fixed and any(not 0 <= s.board < m for s in fixed_samples):
        raise ValueError("fixed_sample board index outside T_base_board0")
    fixed_K_arr = np.asarray(K if fixed_K is None else fixed_K, float)
    fixed_dist_arr = np.asarray(dist if fixed_dist is None else fixed_dist, float)
    fixed_objects = [cal.board_corner_points(board, s.ids) for s in fixed_samples] if use_fixed else []
    T_base_fixed0_arr = None if T_base_fixed0 is None else np.asarray(T_base_fixed0, float)

    estimate_table = table_prior_sigma_m is not None
    hand_eye_dim = 0 if freeze_gripper_camera else 6
    b_at = n + hand_eye_dim
    focal_at = b_at + 6 * m
    table_at = focal_at + (1 if estimate_focal else 0)
    fixed_at = table_at + (1 if estimate_table else 0)
    x0 = np.concatenate([
        np.zeros(n),
        np.zeros(hand_eye_dim),
        np.zeros(6 * m),
        [0.0] if estimate_focal else [],
        [float(table_z_m)] if estimate_table else [],
        np.zeros(6) if use_fixed else [],
    ])

    def table(x):
        return x[table_at] if estimate_table else float(table_z_m)

    def table_z_at(x, px, py):
        return table(x) - (n_vec[0] * (px - origin_x) + n_vec[1] * (py - origin_y)) / n_vec[2]

    def _eval_touch(x, offsets, touch):
        q, link, point = touch[:3]
        jaw = float(touch[3]) if len(touch) > 3 else 0.0
        q_cal = np.asarray(q, float) + offsets
        if lowest_point_fn is not None:
            low = lowest_point_fn(q_cal, jaw)
            z = float(low["z_m"])
            if table_normal is not None:
                if "point_in_base_m" in low:
                    px, py = float(low["point_in_base_m"][0]), float(low["point_in_base_m"][1])
                elif fk_all is not None:
                    p_base = fk_all(q_cal)[low.get("link", link)] @ np.append(
                        np.asarray(low.get("point_in_link_m", point), float), 1.0)
                    px, py = float(p_base[0]), float(p_base[1])
                else:
                    px, py = origin_x, origin_y
            else:
                px, py = origin_x, origin_y
        else:
            p_base = fk_all(q_cal)[link] @ np.append(np.asarray(point, float), 1.0)
            px, py, z = float(p_base[0]), float(p_base[1]), float(p_base[2])
        return z - float(table_z_at(x, px, py))

    def touch_residuals(x):
        offsets = unpack(x)[0]
        return [float(_eval_touch(x, offsets, touch)) for touch in touches]

    def unpack(x):
        offsets = np.zeros(5); offsets[index] = x[:n]
        X = T_gripper_camera0 if freeze_gripper_camera else _perturb(T_gripper_camera0, x[n:n + 6])
        boards = [_perturb(boards0[k], x[b_at + 6 * k:b_at + 6 * (k + 1)]) for k in range(m)]
        T_fixed = _perturb(T_base_fixed0_arr, x[fixed_at:fixed_at + 6]) if use_fixed else T_base_fixed0_arr
        return offsets, X, boards, T_fixed

    def intrinsics(x):
        if not estimate_focal:
            return K
        scaled = K.copy(); scaled[0, 0] *= np.exp(x[focal_at]); scaled[1, 1] *= np.exp(x[focal_at])
        return scaled

    def residuals(x, per_sample=False):
        offsets, X, boards, T_fixed = unpack(x)
        Kx = intrinsics(x)
        X_inv = cal.invert_transform(X)
        out, rms, fixed_rms = [], [], []
        for sample, obj in zip(samples, objects):
            T_camera_board = X_inv @ cal.invert_transform(fk_link(sample.q_rad + offsets)) @ boards[sample.board]
            rvec = cv2.Rodrigues(T_camera_board[:3, :3])[0]
            projected, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rvec, T_camera_board[:3, 3], Kx, dist)
            err = (projected.reshape(-1, 2) - sample.corners_px)
            corner_scale = np.sqrt(max(1.0, float(len(sample.ids)))) if touches else 1.0
            out.append(_robust_pixel_residual(err.ravel() / (pixel_sigma * corner_scale), pixel_robust_scale))
            rms.append(float(np.sqrt(np.mean(np.sum(err ** 2, axis=1)))))
        if use_fixed:
            T_fixed_inv = cal.invert_transform(T_fixed)
            for sample, obj in zip(fixed_samples, fixed_objects):
                T_camera_board = T_fixed_inv @ boards[sample.board]
                rvec = cv2.Rodrigues(T_camera_board[:3, :3])[0]
                projected, _ = cv2.projectPoints(
                    obj.reshape(-1, 1, 3), rvec, T_camera_board[:3, 3], fixed_K_arr, fixed_dist_arr)
                err = (projected.reshape(-1, 2) - sample.corners_px)
                out.append(_robust_pixel_residual(err.ravel() / pixel_sigma, pixel_robust_scale))
                fixed_rms.append(float(np.sqrt(np.mean(np.sum(err ** 2, axis=1)))))
        for B in boards:
            corners = (B[:3, :3] @ outline.T).T + B[:3, 3]
            expected_z = table_z_at(x, corners[:, 0], corners[:, 1]) + board_thickness_m
            out.append((corners[:, 2] - expected_z) / plane_sigma_m)
        for touch in touches:
            out.append(np.array([_eval_touch(x, offsets, touch) / touch_sigma_m]))
        if n:
            out.append((x[:n] + init_vec) / offset_prior_rad)
        if estimate_table:
            out.append(np.array([(x[table_at] - table_z_m) / table_prior_sigma_m]))
        flat = np.concatenate(out) if out else np.zeros(0, dtype=float)
        return (flat, rms, fixed_rms) if per_sample else flat

    initial = float(np.sqrt(np.mean(np.square(residuals(x0, True)[1]))))
    fit = least_squares(residuals, x0, method="trf", jac="2-point", loss="linear", max_nfev=max_nfev,
                        ftol=ftol, xtol=xtol, gtol=gtol, x_scale=x_scale) if len(x0) else type(
        "_Fit", (), {"x": x0, "success": True, "message": "frozen"})()
    offsets, X, boards, T_fixed = unpack(fit.x)
    B = boards[0]
    _, per, fixed_per = residuals(fit.x, True)
    normal = B[:3, 2]
    table_val = float(table(fit.x))
    return JointCalibrationResult(
        joint_offsets_rad={name: float(offsets[JOINTS.index(name)]) for name in JOINTS},
        T_gripper_camera=X, T_base_board=B,
        reprojection_rms_px=float(np.sqrt(np.mean(np.square(per)))), initial_reprojection_rms_px=initial,
        board_top_z_m=float(np.mean((B[:3, :3] @ outline.T).T[:, 2] + B[2, 3])),
        board_tilt_rad=float(np.arccos(np.clip(float(np.dot(normal, -n_vec)), -1.0, 1.0))),
        estimated_joints=tuple(estimated_joints),
        unobservable_joints=tuple(name for name in JOINTS if name not in estimated_joints),
        per_sample_rms_px=per, focal_scale=float(np.exp(fit.x[focal_at])) if estimate_focal else 1.0,
        success=bool(fit.success), message=str(fit.message), table_z_m=table_val,
        effective_solver_settings=effective_solver_settings, nfev=getattr(fit, "nfev", None),
        touch_residuals_m=touch_residuals(fit.x) if touches else [], T_base_boards=boards,
        board_top_z_list_m=[float(np.mean((T[:3, :3] @ outline.T).T[:, 2] + T[2, 3])) for T in boards],
        max_board_corner_height_error_m=board_corner_height_error_m(
            boards, board, table_val, board_thickness_m, table_normal=table_normal, table_xy_origin=table_xy_origin),
        T_base_fixed=T_fixed,
        fixed_reprojection_rms_px=float(np.sqrt(np.mean(np.square(fixed_per)))) if fixed_per else None)
