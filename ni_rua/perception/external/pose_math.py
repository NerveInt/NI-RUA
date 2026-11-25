"""无状态三维位姿、四元数、李群算子与平行夹爪 TCP 几何纯函数底座 (`pose_math.py`)。

纯粹 SE(3)/SO(3)/四元数/角度算子唯一定义在 ``ni_rua/core/pose_math.py``（在脱敏沙盒中由 ``_pose_math_core.py`` 提供），
本模块保持全部原有符号与 ARX X5 几何常量重导出，确保包内与独立沙盒调用方 100% 兼容。
"""
from __future__ import annotations

import numpy as np

if __package__:
    from ...core.pose_math import (
        TOP_DOWN_QUAT_WXYZ,
        _REFERENCE_QUAT,
        _axis_quat,
        angles_deg_to_matrices_batch,
        angles_deg_to_quat_wxyz,
        angles_deg_to_quat_xyzw,
        axis_difference,
        axis_difference_deg,
        closure_yaws,
        closure_yaws_deg,
        matrix_to_quat_wxyz,
        matrix_to_quat_xyzw,
        nearest,
        nearest_yaw_deg,
        orthonormal_triad_from_axes,
        posture_transition_angles_deg,
        quat_conjugate,
        quat_multiply,
        quat_multiply_batch,
        quat_normalize,
        quat_slerp_wxyz,
        quat_wxyz_to_angles_deg,
        quat_wxyz_to_matrix,
        quat_wxyz_to_xyzw,
        quat_xyzw_to_angles_deg,
        quat_xyzw_to_matrix,
        quat_xyzw_to_wxyz,
        se3_adjoint,
        se3_exp,
        se3_inverse,
        se3_log,
        so3_exp,
        so3_left_jacobian,
        so3_left_jacobian_inv,
        so3_log,
        so3_skew,
        wrap90,
        wrap180,
    )
else:
    try:
        from _pose_math_core import (  # type: ignore[no-redef, import-not-found]
            TOP_DOWN_QUAT_WXYZ,
            _REFERENCE_QUAT,
            _axis_quat,
            angles_deg_to_matrices_batch,
            angles_deg_to_quat_wxyz,
            angles_deg_to_quat_xyzw,
            axis_difference,
            axis_difference_deg,
            closure_yaws,
            closure_yaws_deg,
            matrix_to_quat_wxyz,
            matrix_to_quat_xyzw,
            nearest,
            nearest_yaw_deg,
            orthonormal_triad_from_axes,
            posture_transition_angles_deg,
            quat_conjugate,
            quat_multiply,
            quat_multiply_batch,
            quat_normalize,
            quat_slerp_wxyz,
            quat_wxyz_to_angles_deg,
            quat_wxyz_to_matrix,
            quat_wxyz_to_xyzw,
            quat_xyzw_to_angles_deg,
            quat_xyzw_to_matrix,
            quat_xyzw_to_wxyz,
            se3_adjoint,
            se3_exp,
            se3_inverse,
            se3_log,
            so3_exp,
            so3_left_jacobian,
            so3_left_jacobian_inv,
            so3_log,
            so3_skew,
            wrap90,
            wrap180,
        )
    except ImportError:
        from ni_rua.core.pose_math import (  # type: ignore[no-redef]
            TOP_DOWN_QUAT_WXYZ,
            _REFERENCE_QUAT,
            _axis_quat,
            angles_deg_to_matrices_batch,
            angles_deg_to_quat_wxyz,
            angles_deg_to_quat_xyzw,
            axis_difference,
            axis_difference_deg,
            closure_yaws,
            closure_yaws_deg,
            matrix_to_quat_wxyz,
            matrix_to_quat_xyzw,
            nearest,
            nearest_yaw_deg,
            orthonormal_triad_from_axes,
            posture_transition_angles_deg,
            quat_conjugate,
            quat_multiply,
            quat_multiply_batch,
            quat_normalize,
            quat_slerp_wxyz,
            quat_wxyz_to_angles_deg,
            quat_wxyz_to_matrix,
            quat_wxyz_to_xyzw,
            quat_xyzw_to_angles_deg,
            quat_xyzw_to_matrix,
            quat_xyzw_to_wxyz,
            se3_adjoint,
            se3_exp,
            se3_inverse,
            se3_log,
            so3_exp,
            so3_left_jacobian,
            so3_left_jacobian_inv,
            so3_log,
            so3_skew,
            wrap90,
            wrap180,
        )

# ------------------------------------------------------------------ 机械臂末端与夹爪物理常量 (单位: m, deg)
GRASP_POINT_OFFSET_M: float = 0.1501
PAD_DEPTH_M: float = 0.015
FINGERTIP_BELOW_PAD_M: float = 0.0075
FINGER_WIDTH_M: float = 0.018
FINGER_THICKNESS_M: float = 0.008
FINGER_LENGTH_M: float = 0.070
PALM_LENGTH_M: float = 0.065
PALM_WIDTH_M: float = 0.096
PALM_THICKNESS_M: float = 0.056
MAX_OPENING_M: float = 0.088
CONTACT_DEPTH_PRESETS_M: dict[str, float] = {
    "pad_center": 0.0,
    "finger_middle": 0.0275,
    "fingertip": -0.0075,
}


def flange_pose_to_grasp_pose(
    position_m,
    quaternion_xyzw,
    contact_region: str = "pad_center",
    contact_depth_m: float | None = None,
    grasp_point_offset_m: float = GRASP_POINT_OFFSET_M,
    fingertip_below_pad_m: float = FINGERTIP_BELOW_PAD_M,
    *,
    profile=None,
) -> dict:
    """将末端法兰位姿 `(position_m, quaternion_xyzw)` 转换为指垫目标位姿 `(grasp_position_m, angles_deg)`。

    参数:
        position_m: 法兰中心三维坐标 `[x, y, z]`，单位 `m`。
        quaternion_xyzw: 法兰单位四元数 `[qx, qy, qz, qw]`。
        contact_region: 接触截面名称（`'pad_center'` | `'fingertip'` | `'finger_middle'`），默认 `'pad_center'`。
        contact_depth_m: 可选显式轴向接触深度偏移，单位 `m`；提供时覆盖 `contact_region`。
        grasp_point_offset_m: 法兰沿接近轴到 `pad_center` 的基准距离，单位 `m`，默认 `0.1501`。
        fingertip_below_pad_m: 指尖相对 `pad_center` 沿接近轴向前的突出距离，单位 `m`，默认 `0.0075`。

    返回:
        字典 `{'grasp_position_m': [x, y, z], 'angles_deg': [pitch_deg, roll_deg, yaw_deg]}`（单位 `m` / `deg`）。
    """
    flange_pos = np.asarray(position_m, dtype=np.float64).reshape(3)
    qx, qy, qz, qw = (float(v) for v in np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4))
    quat_wxyz = np.array([qw, qx, qy, qz], dtype=np.float64)
    presets = CONTACT_DEPTH_PRESETS_M
    if profile is not None and getattr(profile, "robot_model_id", "arx-x5") == "libero-franka-panda":
        from ...robot.embodiments.libero_franka_panda import grasp_contact as panda_contact, pose_convention as panda_pose

        p_deg, r_deg, y_deg = panda_pose.quat_to_angles(quat_wxyz)
        u_app = panda_pose.tool_axis(quat_wxyz)
        presets = panda_contact.CONTACT_DEPTH_PRESETS_M
    else:
        p_deg, r_deg, y_deg = quat_wxyz_to_angles_deg(quat_wxyz)
        u_app = quat_wxyz_to_matrix(quat_wxyz)[:, 0]
    if contact_depth_m is not None:
        depth = float(contact_depth_m)
    elif contact_region == "fingertip":
        depth = -float(fingertip_below_pad_m)
    else:
        depth = float(presets.get(contact_region, 0.0))
    grasp_pos = flange_pos + (float(grasp_point_offset_m) - depth) * u_app
    return dict(
        grasp_position_m=[float(grasp_pos[0]), float(grasp_pos[1]), float(grasp_pos[2])],
        angles_deg=[float(p_deg), float(r_deg), float(y_deg)],
    )


def gripper_frame(
    position_m,
    angles_deg,
    contact_region: str = "pad_center",
    contact_depth_m: float | None = None,
    grasp_point_offset_m: float = GRASP_POINT_OFFSET_M,
    fingertip_below_pad_m: float = FINGERTIP_BELOW_PAD_M,
) -> dict:
    """根据请求抓取位姿计算夹爪在世界系下的正交标架与关键几何参考点。

    参数:
        position_m: 长度为 3 的目标位置 [x, y, z]，单位 m (env_origin)。
        angles_deg: 长度为 3 的姿态角 [pitch_deg, roll_deg, yaw_deg]，单位 deg。
        contact_region: 接触截面名称 ('pad_center' | 'fingertip' | 'finger_middle')，默认 'pad_center'。
        contact_depth_m: 可选自定义接触深度偏移，单位 m；若提供则覆盖 contact_region。
        grasp_point_offset_m: 法兰沿接近轴到 pad_center 的基准距离，单位 m，默认 0.1501。
        fingertip_below_pad_m: 指尖相对 pad_center 沿接近轴向前的突出距离，单位 m，默认 0.0075。

    返回:
        包含 pad_center_m, fingertip_center_m, flange_position_m, approach_dir, closing_dir, tangent_dir, axes 的字典。
    """
    pos = np.asarray(position_m, dtype=np.float64).reshape(3)
    ang = np.asarray(angles_deg, dtype=np.float64).reshape(3)
    quat = angles_deg_to_quat_wxyz(ang[0], ang[1], ang[2])
    R = quat_wxyz_to_matrix(quat)
    approach_dir = R[:, 0].copy()
    closing_dir = -R[:, 1].copy()
    tangent_dir = R[:, 2].copy()
    if contact_depth_m is not None:
        depth = float(contact_depth_m)
    elif contact_region == "fingertip":
        depth = -float(fingertip_below_pad_m)
    else:
        depth = float(CONTACT_DEPTH_PRESETS_M.get(contact_region, 0.0))
    pad_center = pos + depth * approach_dir
    fingertip_center = pad_center + float(fingertip_below_pad_m) * approach_dir
    flange_pos = pad_center - float(grasp_point_offset_m) * approach_dir
    axes = np.column_stack([approach_dir, closing_dir, tangent_dir])
    return dict(
        pad_center_m=pad_center,
        fingertip_center_m=fingertip_center,
        flange_position_m=flange_pos,
        approach_dir=approach_dir,
        closing_dir=closing_dir,
        tangent_dir=tangent_dir,
        axes=axes,
        quat_wxyz=quat,
        contact_region=contact_region,
        contact_depth_m=depth,
    )
