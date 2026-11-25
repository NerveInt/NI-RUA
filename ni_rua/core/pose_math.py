"""Stateless SE(3), SO(3), quaternion (w, x, y, z), and gripper angle helpers backed by SciPy & Pinocchio.

Contains zero ``ni_rua.*`` imports and no embodiment-specific physical constants so it can serve
``ni_rua.core``, ``ni_rua.robot``, ``ni_rua.perception``, ``ni_rua.runtime``, and standalone sandboxed workspaces.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# Reference top-down orientation quaternion (w, x, y, z): tool +x along world -Z, jaw axis along world +X.
TOP_DOWN_QUAT_WXYZ: tuple[float, float, float, float] = (0.5, -0.5, 0.5, 0.5)
_REFERENCE_QUAT = np.asarray(TOP_DOWN_QUAT_WXYZ, dtype=np.float64)
_REFERENCE_ROT = Rotation.from_quat(_REFERENCE_QUAT, scalar_first=True)


def _rigid_transform():
    """SciPy's ``RigidTransform`` (SciPy >= 1.16), imported on first SE(3) exp/log use.

    Robot workers run in the simulator's own interpreter, whose SciPy may predate it; they use the
    quaternion and rotation helpers of this module, which must stay importable there.
    """
    from scipy.spatial.transform import RigidTransform
    return RigidTransform


def _vec3(values: Any, name: str) -> np.ndarray:
    """Validate and return a finite length-3 float64 vector."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise ValueError(f"invalid_{name}: expected 3 finite numbers")
    return arr


def _vec6(values: Any, name: str) -> np.ndarray:
    """Validate and return a finite length-6 float64 twist vector ``[vx, vy, vz, wx, wy, wz]``."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (6,) or not np.isfinite(arr).all():
        raise ValueError(f"invalid_{name}: expected 6 finite numbers")
    return arr


def _unit_vec3(values: Any, name: str) -> np.ndarray:
    """Validate and normalize a non-zero length-3 direction vector."""
    vec = _vec3(values, name)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9:
        raise ValueError(f"zero_norm_{name}: axis vector must be non-zero")
    return vec / norm


def _so3_matrix(rot: Any) -> np.ndarray:
    """Validate a finite right-handed 3x3 orthogonal rotation matrix."""
    m = np.asarray(rot, dtype=np.float64)
    if m.shape != (3, 3) or not np.isfinite(m).all():
        raise ValueError("invalid_rotation_matrix_shape")
    if not np.allclose(m.T @ m, np.eye(3), atol=1e-4) or float(np.linalg.det(m)) <= 0.0:
        raise ValueError("invalid_rotation_matrix")
    return m


def _se3_matrix(T_4x4: Any) -> np.ndarray:
    """Validate a finite 4x4 homogeneous SE(3) rigid transformation matrix."""
    mat = np.asarray(T_4x4, dtype=np.float64)
    if mat.shape != (4, 4) or not np.isfinite(mat).all():
        raise ValueError("invalid_se3_matrix")
    if not np.allclose(mat[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
        raise ValueError("invalid_se3_matrix")
    rot = mat[:3, :3]
    if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-4) or float(np.linalg.det(rot)) <= 0.0:
        raise ValueError("invalid_se3_matrix")
    clean = mat.copy()
    clean[3] = [0.0, 0.0, 0.0, 1.0]
    return clean


def wrap180(angle_deg: float) -> float:
    """Wrap an angle in degrees into ``[-180, 180)``."""
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def wrap90(angle_deg: float) -> float:
    """Wrap an undirected axis angle in degrees into ``[-90, 90)``."""
    return (float(angle_deg) + 90.0) % 180.0 - 90.0


def axis_difference_deg(a_deg: float, b_deg: float) -> float:
    """Compute the undirected angular difference in degrees between two horizontal axes in ``[0, 90]``."""
    return abs(wrap90(float(a_deg) - float(b_deg)))


axis_difference = axis_difference_deg


def closure_yaws_deg(object_yaw_deg: float) -> list[float]:
    """Return the two equivalent gripper jaw yaws (deg) orthogonal to ``object_yaw_deg``."""
    return [round(wrap180(float(object_yaw_deg) + 90.0), 1), round(wrap180(float(object_yaw_deg) - 90.0), 1)]


closure_yaws = closure_yaws_deg


def nearest_yaw_deg(candidates: list[float] | tuple[float, ...], reference: float | None = None) -> float:
    """Return the candidate yaw (deg) closest to ``reference`` (or smallest ``|yaw|`` when ``reference`` is ``None``)."""
    if reference is None:
        return min(candidates, key=lambda value: (abs(value), value))
    ref = float(reference)
    return min(candidates, key=lambda value: (abs(wrap180(float(value) - ref)), abs(value)))


nearest = nearest_yaw_deg


def _axis_quat(axis: int, radians: float) -> np.ndarray:
    """Unit ``(w, x, y, z)`` quaternion for an elementary rotation of ``radians`` (rad) about coordinate axis ``axis``."""
    return Rotation.from_euler("xyz"[int(axis)], float(radians)).as_quat(scalar_first=True)


def quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply two ``(w, x, y, z)`` quaternions."""
    r_left = Rotation.from_quat(np.asarray(left, dtype=np.float64).reshape(4), scalar_first=True)
    r_right = Rotation.from_quat(np.asarray(right, dtype=np.float64).reshape(4), scalar_first=True)
    return (r_left * r_right).as_quat(scalar_first=True)


def quat_multiply_batch(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply batches of ``(w, x, y, z)`` quaternions ``(N, 4) * (N, 4) -> (N, 4)``."""
    l_arr = np.asarray(left, dtype=np.float64).reshape(-1, 4)
    if l_arr.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float64)
    r_arr = np.asarray(right, dtype=np.float64).reshape(l_arr.shape[0], 4)
    return (Rotation.from_quat(l_arr, scalar_first=True) * Rotation.from_quat(r_arr, scalar_first=True)).as_quat(
        scalar_first=True
    )


def quat_conjugate(quat_wxyz: np.ndarray) -> np.ndarray:
    """Conjugate (inverse for unit norm) of a ``(w, x, y, z)`` quaternion."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    return np.asarray([w, -x, -y, -z], dtype=np.float64)


def quat_normalize(quat_wxyz: np.ndarray) -> np.ndarray:
    """Normalize a ``(w, x, y, z)`` quaternion to unit length."""
    arr = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(arr))
    return arr / max(norm, 1e-12)


def quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert a ``(w, x, y, z)`` quaternion to a 3x3 rotation matrix."""
    return Rotation.from_quat(quat_normalize(quat_wxyz), scalar_first=True).as_matrix()


def quat_xyzw_to_matrix(quat_xyzw: Any) -> np.ndarray:
    """Convert an ``(x, y, z, w)`` quaternion to a 3x3 rotation matrix."""
    arr = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    return Rotation.from_quat(arr / max(float(np.linalg.norm(arr)), 1e-12), scalar_first=False).as_matrix()


def matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    """将 `3x3` 正交旋转矩阵转换为规范半球单位四元数 `[w, x, y, z]`（标量在前，满足 `w >= 0`）。

    参数:
        rot: 形状为 `(3, 3)` 的正交旋转矩阵。

    返回:
        长度为 4 的单位四元数数组 `[w, x, y, z]`（无量纲）。

    示例:
        `q_wxyz = matrix_to_quat_wxyz(np.eye(3))`
    """
    m = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    return Rotation.from_matrix(m).as_quat(canonical=True, scalar_first=True)


def matrix_to_quat_xyzw(rotation: np.ndarray) -> list[float]:
    """Convert a 3x3 rotation matrix into a canonical ``[x, y, z, w]`` unit quaternion (``w >= 0``)."""
    m = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    q = Rotation.from_matrix(m).as_quat(canonical=True, scalar_first=False)
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def quat_wxyz_to_xyzw(quat_wxyz: Any) -> list[float]:
    """将 `[w, x, y, z]` 四元数转换为与 `robot_action.quaternion_xyzw` 同序的 `[x, y, z, w]` 列表。"""
    w, x, y, z = (float(v) for v in np.asarray(quat_wxyz, dtype=np.float64).reshape(4))
    return [x, y, z, w]


def quat_xyzw_to_wxyz(quat_xyzw: Any) -> np.ndarray:
    """将 `[x, y, z, w]` 四元数转换为 `[w, x, y, z]` 浮点数组。"""
    x, y, z, w = (float(v) for v in np.asarray(quat_xyzw, dtype=np.float64).reshape(4))
    return np.asarray([w, x, y, z], dtype=np.float64)


def quat_slerp_wxyz(q0_wxyz: Any, q1_wxyz: Any, t: float) -> np.ndarray:
    r"""在 $S^3 / \{\pm 1\}$ 上执行单位四元数最短测地弧球面线性插值（SLERP）。

    参数:
        q0_wxyz: 起始单位四元数 `[w, x, y, z]`（无量纲）。
        q1_wxyz: 终止单位四元数 `[w, x, y, z]`（无量纲）。
        t: 插值相位参数（无量纲，范围 `[0.0, 1.0]`，`0.0` 返回 `q0`，`1.0` 返回 `q1`）。

    返回:
        插值后的单位四元数 `[w, x, y, z]`（满足 `w >= 0`）。

    示例:
        `q_mid = quat_slerp_wxyz([1.0, 0.0, 0.0, 0.0], [0.7071, 0.0, 0.0, 0.7071], 0.5)`
    """
    tau = float(t)
    if not math.isfinite(tau) or not (0.0 - 1e-9 <= tau <= 1.0 + 1e-9):
        raise ValueError("slerp_parameter_t_must_be_in_0_1")
    tau = float(np.clip(tau, 0.0, 1.0))
    qa = quat_normalize(np.asarray(q0_wxyz, dtype=np.float64).reshape(4))
    qb = quat_normalize(np.asarray(q1_wxyz, dtype=np.float64).reshape(4))
    key_rots = Rotation.from_quat(np.stack([qa, qb], axis=0), scalar_first=True)
    return Slerp([0.0, 1.0], key_rots)(tau).as_quat(canonical=True, scalar_first=True)


def angles_deg_to_quat_wxyz(pitch_deg: float, roll_deg: float, yaw_deg: float) -> np.ndarray:
    """Convert extrinsic ``[pitch_deg, roll_deg, yaw_deg]`` (deg) relative to ``TOP_DOWN_QUAT_WXYZ`` into ``(w, x, y, z)``."""
    rot = Rotation.from_euler("xyz", [float(pitch_deg), float(roll_deg), float(yaw_deg)], degrees=True) * _REFERENCE_ROT
    return rot.as_quat(scalar_first=True)


def angles_deg_to_quat_xyzw(pitch_deg: float, roll_deg: float, yaw_deg: float) -> list[float]:
    """将相对竖直俯抓基准的外旋姿态角 `[pitch_deg, roll_deg, yaw_deg]`（单位 `deg`）转换为规范单位四元数 `[x, y, z, w]`（与 `robot_action.quaternion_xyzw` 同序）。"""
    rot = Rotation.from_euler("xyz", [float(pitch_deg), float(roll_deg), float(yaw_deg)], degrees=True) * _REFERENCE_ROT
    q = rot.as_quat(canonical=True, scalar_first=False)
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def quat_wxyz_to_angles_deg(quat_wxyz: np.ndarray) -> tuple[float, float, float]:
    """Recover ``(pitch_deg, roll_deg, yaw_deg)`` (deg) relative to ``TOP_DOWN_QUAT_WXYZ`` from ``(w, x, y, z)``."""
    mat = (Rotation.from_quat(quat_normalize(quat_wxyz), scalar_first=True) * _REFERENCE_ROT.inv()).as_matrix()
    roll = math.asin(float(np.clip(-mat[2, 0], -1.0, 1.0)))
    if abs(math.cos(roll)) > 1e-7:
        pitch = math.atan2(float(mat[2, 1]), float(mat[2, 2]))
        yaw = math.atan2(float(mat[1, 0]), float(mat[0, 0]))
    else:
        yaw = 0.0
        sign = 1.0 if mat[2, 0] < 0 else -1.0
        pitch = math.atan2(sign * float(mat[0, 1]), float(mat[1, 1]))
    return (math.degrees(pitch) + 0.0, math.degrees(roll) + 0.0, math.degrees(yaw) + 0.0)


def quat_xyzw_to_angles_deg(quat_xyzw: Any) -> tuple[float, float, float]:
    """将 `[x, y, z, w]` 单位四元数还原为相对竖直俯抓基准的 `(pitch_deg, roll_deg, yaw_deg)`（单位 `deg`）。"""
    return quat_wxyz_to_angles_deg(quat_xyzw_to_wxyz(quat_xyzw))


def angles_deg_to_matrices_batch(angles_deg: np.ndarray) -> np.ndarray:
    """Convert shape ``(N, 3)`` ``[pitch_deg, roll_deg, yaw_deg]`` (deg) into ``(N, 3, 3)`` rotation matrices."""
    ang = np.asarray(angles_deg, dtype=np.float64).reshape(-1, 3)
    if ang.shape[0] == 0:
        return np.empty((0, 3, 3), dtype=np.float64)
    return (Rotation.from_euler("xyz", ang, degrees=True) * _REFERENCE_ROT).as_matrix()


def posture_transition_angles_deg(
    release_yaw_deg: float,
    source_posture: str | None = None,
    target_posture: str | None = None,
) -> dict[str, Any]:
    """Compute ``angles_deg=[pitch_deg, roll_deg, yaw_deg]`` (deg) for ``lying <-> upright`` 90-deg jaw-axis transitions."""
    src = source_posture or ("lying" if target_posture == "upright" else "upright")
    dst = target_posture or src
    if src == dst:
        alt_yaw = round(wrap180(float(release_yaw_deg) + 180.0), 1)
        return dict(
            posture_transition="%s->%s" % (src, dst),
            release_angles_deg=[0.0, 0.0, round(float(release_yaw_deg), 1)],
            alternative_angles_deg=[0.0, 0.0, alt_yaw],
        )
    q_base = angles_deg_to_quat_wxyz(0.0, 0.0, float(release_yaw_deg))
    candidates = []
    for sign in (1.0, -1.0):
        q_rot = quat_multiply(q_base, _axis_quat(1, sign * (math.pi / 2.0)))
        pitch, roll, yaw = quat_wxyz_to_angles_deg(q_rot)
        t_axis = quat_wxyz_to_matrix(q_rot)[:, 0]
        candidates.append((
            float(t_axis[1]),
            [round(wrap180(pitch), 1) + 0.0, round(float(roll), 1) + 0.0, round(wrap180(yaw), 1) + 0.0],
            [round(float(v), 3) + 0.0 for v in t_axis],
        ))
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_angles, best_axis = candidates[0][1], candidates[0][2]
    alt_angles = candidates[1][1]
    return dict(
        posture_transition="%s->%s" % (src, dst),
        release_angles_deg=best_angles,
        alternative_angles_deg=alt_angles,
        tool_approach_axis_world=best_axis,
    )


def so3_skew(vec: Any) -> np.ndarray:
    """构造三维向量 `vec` 的 `3x3` 反对称叉乘矩阵 `[v]_x`（满足 `[v]_x @ u == cross(v, u)`）。

    参数:
        vec: 三维向量 `[vx, vy, vz]`（单位由输入物理量决定，如 `m` 或 `rad`）。

    返回:
        形状为 `(3, 3)` 的浮点反对称矩阵。

    示例:
        `K = so3_skew([0.0, 0.0, 1.0])`
    """
    vx, vy, vz = (float(x) for x in _vec3(vec, "vec"))
    return np.array([[0.0, -vz, vy], [vz, 0.0, -vx], [-vy, vx, 0.0]], dtype=np.float64)


def so3_exp(omega_rad: Any) -> np.ndarray:
    r"""计算 $SO(3)$ 李代数指数映射 $\exp([\boldsymbol{\omega}]_\times)$。

    参数:
        omega_rad: 三维旋转向量（轴角表示）`[wx, wy, wz]`，模长为转角，单位 `rad`。

    返回:
        形状为 `(3, 3)` 的右手正交旋转矩阵。

    示例:
        `R = so3_exp([0.0, 0.0, math.pi / 2.0])`
    """
    return Rotation.from_rotvec(_vec3(omega_rad, "omega_rad")).as_matrix()


def so3_log(rot: Any) -> np.ndarray:
    r"""计算 $SO(3)$ 正交旋转矩阵的主分支对数映射 $\log(\mathbf{R}) \in \mathbb{R}^3$（模长 $\in [0, \pi]$，单位 `rad`）。

    参数:
        rot: 形状为 `(3, 3)` 的右手正交旋转矩阵（无量纲）。

    返回:
        长度为 3 的轴角向量 `[wx, wy, wz]`，单位 `rad`。

    示例:
        `omega = so3_log(np.eye(3))`
    """
    return Rotation.from_matrix(_so3_matrix(rot)).as_rotvec()


def so3_left_jacobian(omega_rad: Any) -> np.ndarray:
    r"""计算 $SO(3)$ 左雅可比矩阵 $\mathbf{J}_l(\boldsymbol{\omega})$（无量纲 `3x3` 矩阵）。

    参数:
        omega_rad: 三维旋转轴角向量 `[wx, wy, wz]`，单位 `rad`。

    返回:
        形状为 `(3, 3)` 的左雅可比矩阵。

    示例:
        `Jl = so3_left_jacobian([0.0, 0.0, 0.5])`
    """
    w = _vec3(omega_rad, "omega_rad")
    return np.ascontiguousarray(
        _rigid_transform().from_exp_coords(np.column_stack([np.broadcast_to(w, (3, 3)), np.eye(3, dtype=np.float64)])).translation.T
    )


def so3_left_jacobian_inv(omega_rad: Any) -> np.ndarray:
    r"""计算 $SO(3)$ 左雅可比逆矩阵 $\mathbf{J}_l^{-1}(\boldsymbol{\omega})$（满足 $\mathbf{J}_l \mathbf{J}_l^{-1} = \mathbf{I}$）。

    参数:
        omega_rad: 三维旋转轴角向量 `[wx, wy, wz]`，单位 `rad`（模长 `< 2*pi`）。

    返回:
        形状为 `(3, 3)` 的左雅可比逆矩阵。

    示例:
        `Jl_inv = so3_left_jacobian_inv([0.1, -0.2, 0.3])`
    """
    w = _vec3(omega_rad, "omega_rad")
    if float(np.linalg.norm(w)) >= 2.0 * math.pi - 1e-7:
        raise ValueError("singular_so3_left_jacobian_inv")
    return np.linalg.inv(so3_left_jacobian(w))


def se3_exp(xi_6d: Any) -> np.ndarray:
    r"""计算 $SE(3)$ 旋量李代数指数映射 $\exp(\hat{\boldsymbol{\xi}}) \in SE(3)$。

    参数:
        xi_6d: 长度为 6 的旋量向量 `[vx, vy, vz, wx, wy, wz]`，单位 `[m, m, m, rad, rad, rad]`。

    返回:
        形状为 `(4, 4)` 的齐次刚体变换矩阵（平移单位 `m`）。

    示例:
        `T = se3_exp([0.10, 0.0, 0.0, 0.0, 0.0, math.pi / 2.0])`
    """
    xi = _vec6(xi_6d, "xi_6d")
    return _rigid_transform().from_exp_coords(np.r_[xi[3:], xi[:3]]).as_matrix()


def se3_log(T_4x4: Any) -> np.ndarray:
    r"""计算 $SE(3)$ 齐次刚体矩阵的主分支对数映射 $\log(\mathbf{T}) = [\mathbf{v}; \boldsymbol{\omega}] \in \mathbb{R}^6$。

    参数:
        T_4x4: 形状为 `(4, 4)` 的齐次刚体变换矩阵，平移分量单位 `m`。

    返回:
        长度为 6 的旋量向量 `[vx, vy, vz, wx, wy, wz]`，单位 `[m, m, m, rad, rad, rad]`。

    示例:
        `xi = se3_log(np.eye(4))`
    """
    ec = _rigid_transform().from_matrix(_se3_matrix(T_4x4)).as_exp_coords()
    return np.r_[ec[3:], ec[:3]]


def se3_inverse(T_4x4: Any) -> np.ndarray:
    r"""计算 `4x4` 齐次刚体变换矩阵的精确解析逆 $\mathbf{T}^{-1}$（平移单位 `m`）。

    参数:
        T_4x4: 形状为 `(4, 4)` 的齐次刚体变换矩阵，平移单位 `m`。

    返回:
        形状为 `(4, 4)` 的逆齐次刚体变换矩阵。

    示例:
        `T_inv = se3_inverse(T)`
    """
    return _rigid_transform().from_matrix(_se3_matrix(T_4x4)).inv().as_matrix()


def se3_adjoint(T_4x4: Any) -> np.ndarray:
    r"""计算旋量顺序 $[\mathbf{v}; \boldsymbol{\omega}]$ 下的 `6x6` 伴随变换矩阵 $\text{Ad}_{\mathbf{T}}$。

    参数:
        T_4x4: 形状为 `(4, 4)` 的齐次刚体变换矩阵，平移单位 `m`。

    返回:
        形状为 `(6, 6)` 的浮点伴随变换矩阵。

    示例:
        `Ad = se3_adjoint(T)`
    """
    mat = _se3_matrix(T_4x4)
    rot, trans = mat[:3, :3], mat[:3, 3]
    out = np.zeros((6, 6), dtype=np.float64)
    out[:3, :3] = rot
    out[:3, 3:] = so3_skew(trans) @ rot
    out[3:, 3:] = rot
    return out


def orthonormal_triad_from_axes(primary_axis: Any, secondary_axis: Any | None = None) -> np.ndarray:
    """由主方向轴与可选副参考方向轴执行 Gram-Schmidt 正交归一化，构造 `3x3` 右手正交标架矩阵。

    参数:
        primary_axis: 标架第一列（`+X` 主轴）非零方向向量 `[ax, ay, az]`（无量纲）。
        secondary_axis: 可选位于 `+X-+Y` 半平面内的非共线副参考向量 `[sx, sy, sz]`（无量纲），默认 `None` 自动取最小分量正交基。

    返回:
        形状为 `(3, 3)` 的右手正交矩阵 `[e_x, e_y, e_z]`（满足 `det(R) == +1`）。

    示例:
        `R = orthonormal_triad_from_axes([0.0, 0.0, -1.0], [1.0, 0.0, 0.0])`
    """
    ex = _unit_vec3(primary_axis, "primary_axis")
    if secondary_axis is not None:
        s = _unit_vec3(secondary_axis, "secondary_axis")
        vy = s - float(np.dot(s, ex)) * ex
        ny = float(np.linalg.norm(vy))
        if ny <= 1e-6:
            raise ValueError("secondary_axis_collinear_with_primary_axis")
        ey = vy / ny
    else:
        ref = np.zeros(3, dtype=np.float64)
        ref[int(np.argmin(np.abs(ex)))] = 1.0
        vy = ref - float(np.dot(ref, ex)) * ex
        ey = vy / float(np.linalg.norm(vy))
    ez = np.cross(ex, ey)
    return np.column_stack([ex, ey, ez])
