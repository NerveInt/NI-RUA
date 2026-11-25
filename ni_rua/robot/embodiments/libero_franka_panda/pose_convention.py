"""Top-down ``grasp_angles`` convention for the simulated LIBERO Franka Emika Panda (simulation-only).

``gripper0_grip_site`` sits directly at the nominal grasp pad centre (``GRASP_POINT_OFFSET_M = 0.0``),
with local ``+Z`` pointing outward along the fingers (approach direction) and local ``+X`` along the
parallel-jaw opening line (``panda_rightfinger`` opens toward ``+X``, ``panda_leftfinger`` toward ``-X``).

``TOP_DOWN_QUAT_WXYZ`` maps ``gripper0_grip_site`` to the exact same world grasp frame as NI-RUA's
global ``angles_deg = [pitch_deg, roll_deg, yaw_deg]`` convention:
- at ``angles_deg = [0, 0, 0]`` (``quaternion_xyzw == [0, 1, 0, 0]``), local ``+Z`` (approach) points straight down
  ``[0, 0, -1]`` in ``franka_base`` and local ``+X`` (jaw opening) points along ``[-1, 0, 0]`` (world ``-X``);
- at ``angles_deg = [0, 0, -90]``, ``quaternion_xyzw == [0.707107, 0.707107, 0, 0]`` (LIBERO's nominal spawn
  orientation of ``gripper0_grip_site``, opening the jaws along world ``+-Y``).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from ....core.pose_math import quat_normalize, quat_wxyz_to_matrix

#: Straight-down grasp reference (w, x, y, z): R_ref = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]].
TOP_DOWN_QUAT_WXYZ: tuple[float, float, float, float] = (
    0.0,
    0.0,
    1.0,
    0.0,
)
_REFERENCE_ROT = Rotation.from_quat(np.asarray(TOP_DOWN_QUAT_WXYZ, dtype=np.float64), scalar_first=True)

#: ``gripper0_grip_site`` lies directly at ``pad_center`` between the fingers (0.0 m offset).
GRASP_POINT_OFFSET_M: float = 0.0

POSITION_AXES: tuple[str, ...] = ("x", "y", "z")
ANGLE_AXES: tuple[str, ...] = ("pitch_deg", "roll_deg", "yaw_deg")
ANGLE_BOUNDS: dict[str, tuple[float, float]] = {
    "pitch_deg": (-180.0, 180.0),
    "roll_deg": (-90.0, 90.0),
    "yaw_deg": (-180.0, 180.0),
}
_GIMBAL_EPS = 1e-7


def angles_to_quat(pitch_deg: float, roll_deg: float, yaw_deg: float) -> np.ndarray:
    """Convert extrinsic world ``[pitch_deg, roll_deg, yaw_deg]`` (deg) relative to ``TOP_DOWN_QUAT_WXYZ`` into ``(w, x, y, z)``."""
    rot = Rotation.from_euler("xyz", [float(pitch_deg), float(roll_deg), float(yaw_deg)], degrees=True) * _REFERENCE_ROT
    return rot.as_quat(canonical=True, scalar_first=True)


def quat_to_angles(quaternion_wxyz: np.ndarray) -> tuple[float, float, float]:
    """Recover ``(pitch_deg, roll_deg, yaw_deg)`` (deg) relative to ``TOP_DOWN_QUAT_WXYZ`` from ``(w, x, y, z)``."""
    arr = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(-1)
    if arr.size != 4 or not np.isfinite(arr).all() or float(np.linalg.norm(arr)) <= 1e-8:
        raise ValueError("a quaternion needs four finite non-zero numbers")
    mat = (Rotation.from_quat(quat_normalize(arr), scalar_first=True) * _REFERENCE_ROT.inv()).as_matrix()
    roll = math.asin(float(np.clip(-mat[2, 0], -1.0, 1.0)))
    if abs(math.cos(roll)) > _GIMBAL_EPS:
        pitch = math.atan2(float(mat[2, 1]), float(mat[2, 2]))
        yaw = math.atan2(float(mat[1, 0]), float(mat[0, 0]))
    else:
        yaw = 0.0
        sign = 1.0 if mat[2, 0] < 0 else -1.0
        pitch = math.atan2(sign * float(mat[0, 1]), float(mat[1, 1]))
    return (math.degrees(pitch) + 0.0, math.degrees(roll) + 0.0, math.degrees(yaw) + 0.0)


def tool_axis(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """World approach direction of ``gripper0_grip_site`` (local ``+Z`` axis)."""
    return quat_wxyz_to_matrix(quaternion_wxyz)[:, 2].copy()


def pose_to_values(pose: np.ndarray) -> dict[str, float]:
    """Convert ``[x, y, z, qw, qx, qy, qz]`` of ``gripper0_grip_site`` into ``grasp_pose`` axis values."""
    arr = np.asarray(pose, dtype=np.float64).reshape(-1)
    if arr.size < 7 or not np.isfinite(arr[:7]).all():
        raise ValueError("a TCP pose needs seven finite numbers")
    pitch, roll, yaw = quat_to_angles(arr[3:7])
    position = arr[:3] + GRASP_POINT_OFFSET_M * tool_axis(arr[3:7])
    return {
        "x": float(position[0]),
        "y": float(position[1]),
        "z": float(position[2]),
        "pitch_deg": pitch,
        "roll_deg": roll,
        "yaw_deg": yaw,
    }


def values_to_pose(values: dict[str, float]) -> np.ndarray:
    """Convert ``{x, y, z, pitch_deg, roll_deg, yaw_deg}`` into ``[x, y, z, qw, qx, qy, qz]`` of ``gripper0_grip_site``."""
    quat = angles_to_quat(values["pitch_deg"], values["roll_deg"], values["yaw_deg"])
    position = np.asarray([values["x"], values["y"], values["z"]], dtype=np.float64)
    if GRASP_POINT_OFFSET_M:
        position = position - GRASP_POINT_OFFSET_M * tool_axis(quat)
    return np.concatenate([position, quat])


def orientation_error_deg(first_wxyz: np.ndarray, second_wxyz: np.ndarray) -> float:
    """Geodesic SO(3) rotation angle (deg) between two ``(w, x, y, z)`` quaternions."""
    q1 = quat_normalize(np.asarray(first_wxyz, dtype=np.float64))
    q2 = quat_normalize(np.asarray(second_wxyz, dtype=np.float64))
    dot = min(1.0, abs(float(np.dot(q1, q2))))
    return math.degrees(2.0 * math.acos(dot))
