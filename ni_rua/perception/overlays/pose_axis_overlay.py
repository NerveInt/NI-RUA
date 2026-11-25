"""3D 夹爪抓取参考点、前向逼近轴与开合方向横杠的投影叠加辅助模块。

The gripper geometry comes from the run's ``EmbodimentProfile`` (``profile``; None = ``ARX_X5_PROFILE``), through
``grasp_frame``, for both gripper references the profile vocabulary defines:

* ``tcp_offset`` (ARX-X5): the grasp centre lies ``grasp_reference_offset_m`` (0.1501 m, distal pad centre) minus
  the ``finger_middle`` contact depth (0.0275 m) along the TCP approach axis (link6 +X), i.e. 0.1226 m; the pad
  centre at 0.1501 m; the fingertip ``fingertip_beyond_reference_m`` (0.0075 m) beyond the pad, 0.1576 m. The
  jaw crossbar spans ``JAW_CROSSBAR_HALF_M`` (0.036 m) each way along the jaw opening axis (link6 +Y).
* ``jaw_fingertip_midpoint`` (SO-101): the CAD inner fixed and moving fingertips at the sample's jaw angle (rad,
  ``so101_gripper_geometry.fingertip_reference``); the grasp centre is their midpoint, the crossbar joins the two
  tips, the approach axis is the TCP approach axis (gripper_frame_link +Z).

Offsets along the approach axis are rounded to 1 um (display geometry). TCP poses are read from the stored
``<tcp_link>_pose`` (``position_m`` m, ``rotation_matrix``; X5 ``link6_pose``). All points are in the trajectory
frame (m); nothing here is a measured object contact.
"""
import cv2
import numpy as np

from ...robot.model.geometry import project

#: Display length (m) of the approach arrow beyond the fingertips.
APPROACH_ARROW_EXTRA_M = 0.045
#: Display half-length (m) of the jaw-opening crossbar of a ``tcp_offset`` gripper (X5).
JAW_CROSSBAR_HALF_M = 0.036
#: Grasp-path colours (BGR) by the arm's index in ``profile.arms``; approach-arrow colours likewise.
GRASP_PATH_COLORS = ((255, 235, 60), (60, 210, 255), (120, 255, 120), (255, 120, 255))
ARROW_COLORS = ((255, 240, 50), (50, 230, 255), (120, 255, 120), (255, 120, 255))


def _profile(profile):
    if profile is None:
        from ...robot.model.embodiments import ARX_X5_PROFILE as profile
    return profile


def _arm_index(profile, arm):
    return profile.arms.index(arm) if arm in profile.arms else len(profile.arms)


def tcp_axis(rotation, local):
    """World direction of the TCP-frame unit vector ``local`` under ``rotation`` (3x3). A basis vector selects the
    rotation column exactly (``rotation[:, k] * sign``), so signed zeros survive as in a column read."""
    local = np.asarray(local, dtype=float)
    hits = np.flatnonzero(local)
    if len(hits) == 1 and abs(local[hits[0]]) == 1.0:
        return rotation[:, hits[0]] * local[hits[0]]
    return rotation @ local


def _tcp_pose(arm_data, tcp):
    pose = (arm_data or {}).get(tcp + "_pose")
    if not isinstance(pose, dict) or "position_m" not in pose or "rotation_matrix" not in pose:
        return None, None
    origin = np.asarray(pose["position_m"], dtype=float)
    rot = np.asarray(pose["rotation_matrix"], dtype=float)
    if origin.shape != (3,) or rot.shape != (3, 3) or not (np.isfinite(origin).all() and np.isfinite(rot).all()):
        return None, None
    return origin, rot


def jaw_angle_rad(profile, arm, arm_data, joint_names=None):
    """Jaw angle (rad) of a sample whose jaw is one of its named joints (SO-101 ``gripper``), else None.

    ``joint_names``: the names of ``arm_data['q_rad']`` (a trajectory's ``joint_names[arm]``); when absent, a
    ``q_rad`` of exactly the profile's arm joints followed by its jaw joints is read in that order."""
    q = (arm_data or {}).get("q_rad")
    jaws = list(profile.jaw_joint_names.get(arm, ()))
    if not isinstance(q, list) or len(jaws) != 1:
        return None
    names = list(joint_names) if joint_names else list(profile.arm_joint_names.get(arm, ())) + jaws
    if len(names) != len(q) or jaws[0] not in names:
        return None
    value = q[names.index(jaws[0])]
    return float(value) if type(value) in (int, float) and np.isfinite(value) else None


def grasp_frame(profile, origin, rotation, jaw_rad=None):
    """World-frame grasp references of one gripper from its TCP pose (``origin`` m, ``rotation`` 3x3).

    Returns None when the gripper needs a jaw angle (``jaw_fingertip_midpoint``) and ``jaw_rad`` is None. Keys:
    ``grasp_center`` (m), ``approach_axis`` and ``jaw_axis`` (unit, world), ``fingertips`` (list of m),
    ``jaw_ends`` (crossbar end points, m), ``reference`` (text), ``pad_center`` (m, ``tcp_offset`` only)."""
    gripper = profile.gripper
    approach = tcp_axis(rotation, gripper.approach_axis_local)
    if gripper.grasp_reference == "tcp_offset":
        offset = gripper.grasp_reference_offset_m
        sections = gripper.contact_sections or {}
        grasp_offset = round(offset - sections.get("finger_middle", 0.0), 6)
        pad_offset = round(offset, 6)
        tip_offset = round(offset + gripper.fingertip_beyond_reference_m, 6)
        jaw_axis = tcp_axis(rotation, gripper.jaw_opening_axis_local)
        grasp_center = origin + grasp_offset * approach
        return dict(reference="tcp_offset_finger_middle", grasp_center=grasp_center,
                    pad_center=origin + pad_offset * approach, fingertips=[origin + tip_offset * approach],
                    approach_axis=approach, jaw_axis=jaw_axis,
                    jaw_ends=[grasp_center + JAW_CROSSBAR_HALF_M * jaw_axis, grasp_center - JAW_CROSSBAR_HALF_M * jaw_axis])
    if gripper.grasp_reference == "jaw_fingertip_midpoint":
        if jaw_rad is None:
            return None
        from ...robot.model.so101_gripper_geometry import fingertip_reference
        tips = fingertip_reference(jaw_rad)
        fixed = origin + rotation @ np.asarray(tips["fixed_tip_m"], dtype=float)
        moving = origin + rotation @ np.asarray(tips["moving_tip_m"], dtype=float)
        normal = rotation @ np.asarray(tips["normal_tcp"], dtype=float) if tips["normal_tcp"] is not None else None
        return dict(reference="cad_inner_fingertip_midpoint_at_jaw_angle", grasp_center=(fixed + moving) / 2,
                    fingertips=[fixed, moving], approach_axis=approach, jaw_axis=normal, jaw_ends=[moving, fixed],
                    opening_m=tips["opening_m"])
    raise ValueError("unsupported_grasp_reference:" + str(gripper.grasp_reference))


def _sample_frame(profile, arm, arm_data, joint_names):
    origin, rot = _tcp_pose(arm_data, profile.tcp_link)
    if origin is None:
        return None, None
    try:
        frame = grasp_frame(profile, origin, rot, jaw_angle_rad(profile, arm, arm_data, joint_names))
    except (ValueError, TypeError):
        return origin, None
    return origin, frame


def draw_grasp_center_path(background, camera, samples, width, height, *, profile=None, joint_names=None):
    """在轨迹预览底图上绘制各臂夹爪抓取参考点 (``grasp_frame`` 的 grasp_center) 的连续投影路径。

    ``joint_names``: ``{arm: names of q_rad}`` of the trajectory (needed for a jaw-angle gripper)."""
    profile = _profile(profile)
    for index, arm in enumerate(profile.arms):
        color = GRASP_PATH_COLORS[index % len(GRASP_PATH_COLORS)]
        points = []
        for sample in samples:
            _, frame = _sample_frame(profile, arm, (sample.get("arms") or {}).get(arm), (joint_names or {}).get(arm))
            points.append(None if frame is None else frame["grasp_center"].tolist())
        previous = None
        for start in range(0, len(points), 2000):
            chunk = points[start : start + 2000]
            valid = [pt for pt in chunk if pt is not None]
            if not valid:
                previous = None
                continue
            pixels, depth, _ = project(camera, valid)
            iterator = iter(zip(pixels, depth))
            for pt in chunk:
                if pt is None:
                    previous = None
                    continue
                pixel, z = next(iterator)
                if not np.isfinite(pixel).all():
                    previous = None
                    continue
                pixel_i = tuple(np.clip(np.rint(pixel), -100000, 100000).astype(int))
                if previous is not None and z > 0 and previous[1] > 0:
                    visible, a, b = cv2.clipLine((0, 0, width, height), previous[0], pixel_i)
                    if visible:
                        cv2.line(background, a, b, color, 1, cv2.LINE_AA)
                previous = pixel_i, z


def draw_pose_axis_frame(output, camera, arm, arm_data, arm_color, width, height, *, profile=None, joint_names=None):
    """在轨迹关键帧面板上绘制夹爪抓取参考点、指尖、前向逼近箭头与开合方向横杠 (``grasp_frame``)。

    ``joint_names``: names of ``arm_data['q_rad']`` (needed for a jaw-angle gripper). Returns the world points
    (m, rounded to 0.1 mm) or None when the sample has no TCP pose or no usable jaw angle."""
    profile = _profile(profile)
    origin, frame = _sample_frame(profile, arm, arm_data, joint_names)
    if frame is None:
        return None
    grasp_center = frame["grasp_center"]
    fingertips = frame["fingertips"]
    axis_x = frame["approach_axis"]
    if "pad_center" in frame:
        approach_tip = fingertips[0] + APPROACH_ARROW_EXTRA_M * axis_x
    else:
        # The arrow starts at the grasp centre and ends beyond the fingertip that reaches furthest.
        reach = max(float(np.dot(tip - grasp_center, axis_x)) for tip in fingertips)
        approach_tip = grasp_center + (reach + APPROACH_ARROW_EXTRA_M) * axis_x
    jaw_pos, jaw_neg = frame["jaw_ends"]

    pts_3d = [
        origin.tolist(),
        grasp_center.tolist(),
        approach_tip.tolist(),
        jaw_pos.tolist(),
        jaw_neg.tolist(),
        *[tip.tolist() for tip in fingertips],
    ]
    pixels, depths, _ = project(camera, pts_3d)
    # A point without an image location (beyond the lens model range, NaN pixel) is skipped exactly like a point
    # behind the camera: no marker, no segment through it. The other elements are still drawn.
    located = np.isfinite(pixels).all(axis=1)
    depths = np.where(located, depths, -np.inf)
    px = [tuple(np.clip(np.rint(p), -100000, 100000).astype(int)) if ok else (0, 0) for p, ok in zip(pixels, located)]
    p_origin, p_grasp, p_arrow, p_jaw_pos, p_jaw_neg = px[:5]
    d_origin, d_grasp, d_arrow, d_jaw_pos, d_jaw_neg = [float(d) for d in depths[:5]]

    # 1. TCP 到 grasp_center 的延伸连杆
    if d_origin > 0 and d_grasp > 0:
        vis, a, b = cv2.clipLine((0, 0, width, height), p_origin, p_grasp)
        if vis:
            cv2.line(output, a, b, arm_color, 2, cv2.LINE_AA)

    # 2. 夹爪开合方向横杠 (绿色)
    crossbar_color = (90, 235, 110)
    if d_jaw_pos > 0 and d_jaw_neg > 0:
        vis, a, b = cv2.clipLine((0, 0, width, height), p_jaw_neg, p_jaw_pos)
        if vis:
            cv2.line(output, a, b, crossbar_color, 2, cv2.LINE_AA)

    # 3. 前向逼近/插入方向箭头 (高亮青/黄色)
    index = _arm_index(profile, arm)
    arrow_color = ARROW_COLORS[index % len(ARROW_COLORS)]
    if d_grasp > 0 and d_arrow > 0:
        vis, a, b = cv2.clipLine((0, 0, width, height), p_grasp, p_arrow)
        if vis:
            cv2.arrowedLine(output, a, b, arrow_color, 2, line_type=cv2.LINE_AA, tipLength=0.28)

    # 4. 抓取参考点圆环与指尖标记
    if d_grasp > 0 and 0 <= p_grasp[0] < width and 0 <= p_grasp[1] < height:
        cv2.circle(output, p_grasp, 5, arrow_color, 2, cv2.LINE_AA)
        cv2.putText(
            output,
            f"{arm[0]}:grasp",
            (min(width - 48, max(4, p_grasp[0] + 6)), min(height - 42, max(64, p_grasp[1] - 6))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            arrow_color,
            1,
            cv2.LINE_AA,
        )
    for p_tip, d_tip in zip(px[5:], depths[5:]):
        if d_tip > 0 and 0 <= p_tip[0] < width and 0 <= p_tip[1] < height:
            cv2.circle(output, p_tip, 3, (255, 255, 255), -1, cv2.LINE_AA)

    jaw_axis = frame["jaw_axis"].round(4).tolist() if frame["jaw_axis"] is not None else None
    if "pad_center" in frame:
        return dict(grasp_center_m=grasp_center.round(4).tolist(), pad_center_m=frame["pad_center"].round(4).tolist(),
                    fingertip_m=fingertips[0].round(4).tolist(), approach_axis_world=axis_x.round(4).tolist(),
                    jaw_closing_axis_world=jaw_axis)
    return dict(reference=frame["reference"], grasp_center_m=grasp_center.round(4).tolist(),
                fingertip_m=fingertips[0].round(4).tolist(), moving_fingertip_m=fingertips[1].round(4).tolist(),
                approach_axis_world=axis_x.round(4).tolist(), jaw_closing_axis_world=jaw_axis)


def draw_contact_jaw_crossbar(output, camera, selected_pos, rotation_matrix, width, height):
    """在接触点叠加图上补充绘制 ±Y 夹爪开合平面横杠，辅助判断旋转对齐。"""
    axis_y = rotation_matrix[:, 1]
    jaw_pos = (selected_pos + JAW_CROSSBAR_HALF_M * axis_y).tolist()
    jaw_neg = (selected_pos - JAW_CROSSBAR_HALF_M * axis_y).tolist()
    pixels, depths, in_img = project(camera, [jaw_neg, jaw_pos])
    if (
        len(depths) == 2
        and depths[0] > 0
        and depths[1] > 0
        and in_img[0]
        and in_img[1]
        and np.isfinite(pixels).all()
    ):
        a = tuple(np.rint(pixels[0]).astype(int))
        b = tuple(np.rint(pixels[1]).astype(int))
        if not (a[0] < 55 and a[1] < 55) and not (b[0] < 55 and b[1] < 55):
            cv2.line(output, a, b, (90, 210, 70), 2, cv2.LINE_AA)
    return axis_y.tolist()


def draw_object_long_axis_overlay(output, camera, targets_3d, width, height):
    """在轨迹预览面板上绘制匹配目标物体的 3D 主长轴与有向包围盒底面，辅助与 ±Y 夹爪闭合横杠垂直比对。"""
    if not targets_3d:
        return
    seen = set()
    for item in targets_3d:
        if not isinstance(item, dict):
            continue
        oid = item.get("id")
        if oid in seen:
            continue
        seen.add(oid)
        centre = np.asarray(item.get("centre_m", [0.0, 0.0, 0.0]), dtype=float)
        length = max(float(item.get("length_m") or 0.04), 0.02)
        obj_w = max(float(item.get("width_m") or 0.02), 0.01)
        yaw_rad = float(np.radians(float(item.get("yaw_deg") or 0.0)))
        u = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0.0])
        v = np.array([-np.sin(yaw_rad), np.cos(yaw_rad), 0.0])
        p_neg = centre - 0.5 * length * u
        p_pos = centre + 0.5 * length * u
        corners = [
            (centre - 0.5 * length * u - 0.5 * obj_w * v).tolist(),
            (centre + 0.5 * length * u - 0.5 * obj_w * v).tolist(),
            (centre + 0.5 * length * u + 0.5 * obj_w * v).tolist(),
            (centre - 0.5 * length * u + 0.5 * obj_w * v).tolist(),
        ]
        pts = [p_neg.tolist(), p_pos.tolist(), *corners]
        pixels, depths, _ = project(camera, pts)
        if not np.isfinite(pixels).all():
            continue
        px = [tuple(np.clip(np.rint(p), -100000, 100000).astype(int)) for p in pixels]
        for idx in range(4):
            c0, c1 = px[2 + idx], px[2 + (idx + 1) % 4]
            if depths[2 + idx] > 0 and depths[2 + (idx + 1) % 4] > 0:
                vis, a, b = cv2.clipLine((0, 0, width, height), c0, c1)
                if vis:
                    cv2.line(output, a, b, (220, 110, 255), 1, cv2.LINE_AA)
        if depths[0] > 0 and depths[1] > 0:
            vis, a, b = cv2.clipLine((0, 0, width, height), px[0], px[1])
            if vis:
                cv2.line(output, a, b, (255, 60, 220), 2, cv2.LINE_AA)
                if oid:
                    cv2.putText(
                        output,
                        f"{oid}:axis",
                        (min(width - 52, max(4, a[0] + 4)), min(height - 42, max(64, a[1] - 4))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.34,
                        (255, 120, 240),
                        1,
                        cv2.LINE_AA,
                    )
