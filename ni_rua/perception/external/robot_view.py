"""The SO-101 arm as one calibrated camera sees it: the object table's occlusion report and the clear-view pose search.

Both use the FK silhouette of ``held_perception`` (the packaged SO-101 meshes posed by forward kinematics of the
measured joints, rastered with the camera's ``K`` / ``T_world_camera`` and dilated by ``SO101Body.dilate_px``).

1. ``occlusion`` (profile support tag ``robot_occlusion``; deployment ``perception.robot_occlusion_report``, default
   true): every model-facing object table of a camera carries ``value.robot_occlusion``:

   * ``table_covered`` (fraction 0..1, 2 decimals, or null when the camera sees no table inside the reach): the share of
     the table ROI under the arm silhouette. Table ROI: the image pixels whose ray meets the table plane
     (``support_z_m`` of the table, m) in front of the camera at a point whose horizontal distance from the
     shoulder_pan axis is at most the arm's horizontal TCP reach (``reach_facts``: m, from the packaged URDF), minus the
     pixels of the static robot base (``base_link``: always in view, nothing can be moved out of it).
   * ``hidden`` (only when non-empty): the rows of the newest earlier table of the same camera attached to an observation
     reply (obs3d keeps the last 8; on-demand geometry_query tables are not kept), and the entries that table itself
     reported hidden unless a row of it lists the same object (an entry's centre inside a row's footprint), whose
     estimated box now lies at least partly under the arm silhouette,
     most covered first: ``id`` (its id in the table that listed it), ``seen_in`` (that table's observation id),
     ``centre_m`` and ``top_z`` (m, that table's estimate) and ``covered`` (fraction 0..1, 2 decimals, of the box's
     projected area under the silhouette). Held rows (support ``held_<arm>``) are never listed: they move with the arm.
     An entry is listed when its covered fraction rounds above 0.00; it is a geometric fact about the old estimate, not
     a detection: the object may have moved, and it can also be a row of the current table.

   No row is added, removed or changed. Without meshes, FK or a table height the field is
   ``{"unavailable": <reason>}``.

2. ``clear_view_search``: offline search over joint configurations (``scripts/so101_clear_view_pose.py``) for a pose
   whose silhouette leaves the table ROI of a camera visible, within the joint limits (minus the deployment margin),
   with the TCP inside the deployment workspace box and no moving link mesh below the table plane. The operator writes
   the chosen joints into deployment ``perception.clear_view_q_rad``; the motion instructions then name its TCP pose
   (``contract.clear_view_sentence``). Nothing moves on its own.

Pixel space: the ideal pinhole image of ``K`` (for a distorted camera the rectified image, where the robot mask is
rendered and the service segments); boxes and rays use ``K`` only. Units: m, rad, px.
"""
from __future__ import annotations

import functools
import itertools
import math
import time

import numpy as np

from . import held_perception as hp

TAG = "robot_occlusion"
#: Samples per joint of the reach grid (shoulder_lift, elbow_flex, wrist_flex over their URDF limits; shoulder_pan and
#: wrist_roll do not change the horizontal distance from the pan axis). 13 samples put a grid point within about
#: 0.15 rad of every configuration; the reach of the packaged URDF is 0.44 m.
REACH_GRID = 13
#: Box corners of a previous row: the oriented footprint (centre, length along yaw_deg, width across it) from its
#: bottom_z to its top_z, m.
_CORNERS = tuple(itertools.product((-0.5, 0.5), (-0.5, 0.5), (0, 1)))


def _model():
    from ...robot.model import so101_kinematics as kin
    return kin.load_model()


def _root_link(model):
    """The link no joint moves (the robot base): parent of a joint and child of none."""
    children = {joint["child"] for joint in model["joints"]}
    roots = [joint["parent"] for joint in model["joints"] if joint["parent"] not in children]
    return roots[0]


@functools.lru_cache(maxsize=2)
def reach_facts():
    """``dict(pan_axis_xy_m=[x, y], reach_m=r, base_link=name)`` of the packaged SO-101 URDF (m, so101_base).

    ``pan_axis_xy_m``: where the shoulder_pan axis meets the table plane (xy of the shoulder_pan child link origin).
    ``reach_m``: the largest horizontal distance of the TCP from that axis over a ``REACH_GRID``^3 grid of
    shoulder_lift, elbow_flex and wrist_flex inside their URDF limits (wrist_roll and shoulder_pan at 0). Cached."""
    from ...robot.model import so101_kinematics as kin
    model = _model()
    pan = next(joint for joint in model["joints"] if joint["name"] == kin.ARM_JOINTS[0])
    axis = kin.fk_all(model, [0.0] * len(kin.ARM_JOINTS))[pan["child"]][:2, 3]
    limits = kin.joint_limits_rad(model)
    grids = [np.linspace(*limits[name], REACH_GRID) for name in kin.ARM_JOINTS[1:4]]
    reach = 0.0
    for lift, elbow, flex in itertools.product(*grids):
        tcp = kin.fk(model, [0.0, lift, elbow, flex, 0.0])[:2, 3]
        reach = max(reach, float(np.hypot(*(tcp - axis))))
    return dict(pan_axis_xy_m=[float(v) for v in axis], reach_m=round(reach, 4), base_link=_root_link(model))


def table_roi(K, T, width, height, table_z_m, facts=None):
    """bool (height, width): pinhole pixels whose ray meets the plane z = ``table_z_m`` (m) in front of the camera
    within ``facts["reach_m"]`` of ``facts["pan_axis_xy_m"]`` (``reach_facts``). ``K`` 3x3 px, ``T`` 4x4
    T_world_camera (m)."""
    facts = facts or reach_facts()
    K, T = np.asarray(K, float), np.asarray(T, float)
    u, v = np.meshgrid(np.arange(int(width)) + 0.5, np.arange(int(height)) + 0.5)
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1) @ T[:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (float(table_z_m) - T[2, 3]) / rays[..., 2]
    x, y = T[0, 3] + s * rays[..., 0], T[1, 3] + s * rays[..., 1]
    axis = facts["pan_axis_xy_m"]
    return (s > 0) & (np.hypot(x - axis[0], y - axis[1]) <= facts["reach_m"])


def silhouettes(body, meshes, poses, camera, camera_id=None, wrist_cameras=None, base_link=None, robot=None):
    """(robot, base) bool masks (pinhole image of ``camera``: K, T, width, height) of every meshed link of ``poses``
    ({arm: {link: 4x4}}) and of the static base link alone; the wrist camera ``camera_id`` leaves out its own housing
    (``held_perception.posed_links``). ``robot``: an already rendered robot mask of the same camera and poses
    (``held_perception.Prepared.mask``), which is then returned instead of a second render."""
    base_link = base_link or reach_facts()["base_link"]
    if robot is None:
        skip = hp.arm_of_camera(camera_id, wrist_cameras) if camera_id is not None else None
        robot = body.mask(camera, hp.posed_links(meshes, poses, skip_camera_of=skip, camera_links=body.camera_links))
    base = [(V, F, arm_poses[base_link]) for arm_poses in poses.values() for link, (V, F) in meshes.items()
            if link == base_link and base_link in arm_poses]
    return robot, body.mask(camera, base[:1]) if base else np.zeros_like(robot)


def coverage(robot, base, roi):
    """Fraction of the table ROI outside the base silhouette that the robot silhouette covers (None: empty ROI)."""
    free = roi & ~base
    total = int(free.sum())
    return None if not total else float((robot & free).sum()) / total


def box_corners(box):
    """(8, 3) corners (m) of ``box`` = dict(centre_m [x, y, z], length_m, width_m, yaw_deg, bottom_z_m, top_z_m)."""
    yaw = math.radians(float(box["yaw_deg"] or 0.0))
    along, across = np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])
    centre = np.asarray(box["centre_m"][:2], float)
    out = []
    for a, b, up in _CORNERS:
        xy = centre + a * float(box["length_m"]) * along + b * float(box["width_m"]) * across
        out.append([xy[0], xy[1], float(box["top_z_m"] if up else box["bottom_z_m"])])
    return np.asarray(out)


def box_covered(K, T, robot, box):
    """Fraction (0..1) of the convex image outline of ``box`` (pinhole ``K``, ``T``) under the ``robot`` mask; None
    when a corner is behind the camera or the outline has no pixel inside the image."""
    import cv2
    K, T = np.asarray(K, float), np.asarray(T, float)
    local = (box_corners(box) - T[:3, 3]) @ T[:3, :3]
    if not (local[:, 2] > 1e-6).all():
        return None
    # continuous image coordinates (pixel i spans [i, i + 1), as render_mask and table_roi) -> raster indices
    uv = np.stack([K[0, 0] * local[:, 0] / local[:, 2] + K[0, 2], K[1, 1] * local[:, 1] / local[:, 2] + K[1, 2]], 1) - 0.5
    height, width = robot.shape
    low = np.maximum(np.floor(uv.min(0)).astype(int) - 1, 0)
    high = np.minimum(np.ceil(uv.max(0)).astype(int) + 2, [width, height])
    if (high <= low).any():
        return None
    canvas = np.zeros((high[1] - low[1], high[0] - low[0]), np.uint8)
    hull = cv2.convexHull(np.rint((uv - low) * 4).astype(np.int32))
    cv2.fillConvexPoly(canvas, hull, 1, lineType=cv2.LINE_8, shift=2)
    inside = canvas.astype(bool)
    total = int(inside.sum())
    return None if not total else float((inside & robot[low[1]:high[1], low[0]:high[0]]).sum()) / total


def _r(value, digits=3):
    return None if value is None else round(float(value), digits)


def row_box(item):
    """The box of one table row (service / diagnostics object: centre_m, length_m, width_m, yaw_deg, top_z_m,
    bottom_z_m) or None when a value is missing."""
    keys = ("centre_m", "length_m", "width_m", "top_z_m", "bottom_z_m")
    if any(item.get(key) is None for key in keys) or not isinstance(item.get("centre_m"), (list, tuple)):
        return None
    return dict(centre_m=[float(v) for v in item["centre_m"]], length_m=float(item["length_m"]),
                width_m=float(item["width_m"]), yaw_deg=float(item.get("yaw_deg") or 0.0),
                bottom_z_m=float(item["bottom_z_m"]), top_z_m=float(item["top_z_m"]))


def inside_footprint(point_xy, box):
    """True when ``point_xy`` (m) lies inside the oriented footprint (centre, length along yaw_deg, width across it)
    of ``box``."""
    yaw = math.radians(float(box["yaw_deg"] or 0.0))
    d = np.asarray(point_xy[:2], float) - np.asarray(box["centre_m"][:2], float)
    along, across = d @ [math.cos(yaw), math.sin(yaw)], d @ [-math.sin(yaw), math.cos(yaw)]
    return abs(along) <= float(box["length_m"]) / 2 and abs(across) <= float(box["width_m"]) / 2


def previous_entries(runtime, observation_id, camera_id):
    """(reference observation id, [dict(id, seen_in, box)]) of the reference table of ``camera_id``: the newest table
    of that camera attached to an observation reply and kept by obs3d (``_obs3d_history``) that comes before
    ``observation_id`` (any kept table when ``observation_id`` is not kept itself). Entries: its rows that are not held
    rows, then the entries that table itself reported hidden, each keeping the id and observation of the table that
    listed it, except a carried entry whose footprint centre lies inside the footprint of a row or entry already taken
    (the newer listing of the same object). (None, []) when there is none."""
    from ...core.episode_state import slot_peek
    from .obs3d import OBS3D_HISTORY_SLOT, OBS3D_MEMO_SLOT
    history = slot_peek(runtime, OBS3D_HISTORY_SLOT, None, legacy_attr="_obs3d_history")
    memo = slot_peek(runtime, OBS3D_MEMO_SLOT, None, legacy_attr="_obs3d_memo")
    entries = dict(history) if isinstance(history, dict) else {}
    if isinstance(memo, tuple) and len(memo) == 2 and isinstance(memo[1], dict):
        entries.setdefault(memo[0], memo[1])
    order = list(entries)
    if observation_id in order:
        order = order[:order.index(observation_id)]
    for previous in reversed(order):
        entry = (entries[previous] or {}).get("object_table")
        result = entry[2] if isinstance(entry, tuple) and len(entry) == 3 and isinstance(entry[2], dict) else {}
        if (result.get("value") or {}).get("camera_id") != camera_id:
            continue
        diagnostics = result.get("diagnostics") or {}
        out = []
        for item in diagnostics.get("objects") or []:
            if str(item.get("support") or "").startswith("held_") or item.get("quality") == "held_fk_depth":
                continue
            box = row_box(item)
            if box is not None:
                out.append(dict(id=item.get("id"), seen_in=previous, box=box))
        for item in (diagnostics.get("robot_occlusion") or {}).get("hidden") or []:
            box = item.get("box")
            if not isinstance(box, dict) or any(inside_footprint(box["centre_m"], kept["box"]) for kept in out):
                continue
            out.append(dict(id=item["id"], seen_in=item["seen_in"], box=box))
        return previous, out
    return None, []


@functools.lru_cache(maxsize=8)
def _cached_roi(K, T, width, height, table_z_m):
    roi = table_roi(np.asarray(K).reshape(3, 3), np.asarray(T).reshape(4, 4), width, height, table_z_m)
    roi.setflags(write=False)
    return roi


def cached_table_roi(K, T, width, height, table_z_m):
    """``table_roi`` cached by exact camera and table values (a static camera's ROI is computed once per process)."""
    return _cached_roi(tuple(np.asarray(K, float).ravel().tolist()), tuple(np.asarray(T, float).ravel().tolist()),
                       int(width), int(height), float(table_z_m))


def occlusion(runtime, camera, camera_id, robot, base, support_z_m, observation_id):
    """``(value, diagnostics)`` of one table's ``robot_occlusion`` (module docstring). ``camera``: GeometryTools
    camera payload (K, T, width, height); ``robot`` / ``base``: its silhouettes (``silhouettes``); ``support_z_m``:
    the table plane (m) of this table."""
    started = time.monotonic()
    K, T = camera["K"], camera["T"]
    roi = cached_table_roi(K, T, robot.shape[1], robot.shape[0], support_z_m)
    covered = coverage(robot, base, roi)
    reference, entries = previous_entries(runtime, observation_id, camera_id)
    hidden = []
    for entry in entries:
        fraction = box_covered(K, T, robot, entry["box"])
        if fraction is not None and round(fraction, 2) > 0:
            hidden.append(dict(entry, covered=round(fraction, 2)))
    hidden.sort(key=lambda item: -item["covered"])
    value = dict(table_covered=_r(covered, 2))
    if hidden:
        value["hidden"] = [dict(id=item["id"], seen_in=item["seen_in"],
                                centre_m=[_r(v) for v in item["box"]["centre_m"]], top_z=_r(item["box"]["top_z_m"]),
                                covered=item["covered"]) for item in hidden]
    diagnostics = dict(roi_px=int(roi.sum()), base_px=int((roi & base).sum()), covered_px=int((roi & ~base & robot).sum()),
                       reference_observation_id=reference, candidates=len(entries),
                       hidden=[dict(id=item["id"], seen_in=item["seen_in"], box=item["box"], covered=item["covered"])
                               for item in hidden],
                       latency_s=round(time.monotonic() - started, 4))
    return value, diagnostics


# ------------------------------------------------------------------ clear-view pose search (offline)
def moving_min_z_m(meshes, poses, base_link):
    """Lowest vertex z (m) of every meshed link except the base in ``poses`` ({link: 4x4})."""
    low = math.inf
    for link, (V, _) in meshes.items():
        if link == base_link or link not in poses:
            continue
        pose = np.asarray(poses[link], float)
        low = min(low, float((V @ pose[:3, :3].T + pose[:3, 3])[:, 2].min()))
    return low


def clear_view_search(camera, table_z_m, *, limits_rad, home_q_rad, workspace_min_m, workspace_max_m, gripper_rad,
                      grid=(13, 9, 9, 7), wrist_roll_rad=0.0, render_scale=0.25):
    """Joint configurations ranked by how much of a camera's table ROI the arm silhouette covers.

    ``camera``: dict(K 3x3 px (pinhole), T 4x4 T_world_camera m, width, height px). ``table_z_m``: table plane (m).
    ``limits_rad``: {joint: (lower, upper)} rad, the planner's limits (URDF limits minus the deployment
    ``motion.joint_limit_margin_rad``). ``home_q_rad``: 5 joints (rad) the joint travel is measured from (deployment
    ``home.q_rad``). ``workspace_min_m`` / ``workspace_max_m``: the deployment's accepted TCP box (m). ``gripper_rad``:
    jaw angle the silhouette is rendered with (rad). ``grid``: samples of shoulder_pan, shoulder_lift, elbow_flex and
    wrist_flex across their limits; wrist_roll is held at ``wrist_roll_rad``. ``render_scale``: raster scale of the
    ranking pass (the front is re-rendered at the default held-perception scale).

    A configuration is kept when its TCP lies inside the workspace box and no moving link vertex is below the table
    plane (no penetration; the path between poses is not checked). Returns the Pareto front over (coverage, travel):
    a list of dict(q_rad, coverage (fraction of the ROI outside the base), travel_rad (largest |q - home| of a joint),
    tcp_position_m, tcp_quaternion_xyzw, min_link_z_m) with coverage ascending and travel strictly descending, so the
    first entry is the least-covering configuration with the least travel among those, and each later one trades
    coverage for travel. Example: ``clear_view_search(fixed, -0.016, limits_rad=..., home_q_rad=[0, -0.95, 0.9, 1, 0],
    workspace_min_m=[-0.45, -0.45, -0.014], workspace_max_m=[0.45, 0.45, 0.45], gripper_rad=0.0)[0]``."""
    from ...core.pose_math import matrix_to_quat_xyzw as _quaternion
    from ...robot.model import so101_kinematics as kin
    model, body = _model(), hp.SO101Body()
    meshes, facts = body.meshes(None), reach_facts()
    base_link = facts["base_link"]
    names = kin.ARM_JOINTS
    width, height = int(camera["width"]), int(camera["height"])
    roi = table_roi(camera["K"], camera["T"], width, height, table_z_m, facts)
    lo, hi = np.asarray(workspace_min_m, float), np.asarray(workspace_max_m, float)

    bases = {}   # the base link does not move: one silhouette per raster scale

    def render(q, scale):
        poses = {"right": kin.fk_all(model, q, gripper_rad=gripper_rad)}
        posed = hp.posed_links(meshes, poses, camera_links=body.camera_links)
        robot = hp.render_mask(camera["K"], camera["T"], width, height, posed, scale=scale,
                               dilate_px=body.dilate_for(width))
        if scale not in bases:
            bases[scale] = hp.render_mask(camera["K"], camera["T"], width, height,
                                          [(meshes[base_link][0], meshes[base_link][1], poses["right"][base_link])],
                                          scale=scale, dilate_px=body.dilate_for(width))
        return poses["right"], coverage(robot, bases[scale], roi)

    axes = [np.linspace(*limits_rad[name], count) for name, count in zip(names[:4], grid)]
    ranked = []
    for values in itertools.product(*axes):
        q = [*map(float, values), float(wrist_roll_rad)]
        poses = kin.fk_all(model, q, gripper_rad=gripper_rad)
        tcp = poses[kin.TCP_LINK][:3, 3]
        if not ((tcp >= lo).all() and (tcp <= hi).all()) or moving_min_z_m(meshes, poses, base_link) < table_z_m:
            continue
        _, covered = render(q, render_scale)
        if covered is not None:
            ranked.append((covered, float(np.max(np.abs(np.subtract(q, home_q_rad)))), q))

    def front(items):   # Pareto front over (coverage, travel), travel compared at the reported 0.1 mrad
        out, best = [], math.inf
        for covered, travel, q in sorted(items, key=lambda item: (item[0], item[1])):
            if round(travel, 4) < best:
                out.append((covered, travel, q))
                best = round(travel, 4)
        return out
    refined = []
    for _, travel, q in front(ranked):
        poses, covered = render(q, hp.RENDER_SCALE)
        refined.append((covered, travel, q))
    def inward(q):   # 4 decimals, rounded toward the inside of the limits (the deployment check is exact)
        out = []
        for name, value in zip(names, q):
            low, high = limits_rad.get(name, (-math.inf, math.inf))
            rounded = round(float(value), 4)
            out.append(math.floor(high * 1e4) / 1e4 if rounded > high else
                       math.ceil(low * 1e4) / 1e4 if rounded < low else rounded)
        return out
    result = []
    for covered, travel, q in front(refined):
        q = inward(q)
        poses = kin.fk_all(model, q, gripper_rad=gripper_rad)
        tcp = poses[kin.TCP_LINK]
        result.append(dict(q_rad=q, coverage=round(covered, 4), travel_rad=round(travel, 4),
                           tcp_position_m=[round(float(v), 4) for v in tcp[:3, 3]],
                           tcp_quaternion_xyzw=[round(float(v), 6) for v in _quaternion(tcp[:3, :3])],
                           min_link_z_m=round(moving_min_z_m(meshes, poses, base_link), 4)))
    return result


def pose_coverage(camera, table_z_m, q_rad, gripper_rad):
    """Coverage (``coverage``) of the table ROI of ``camera`` with the arm at ``q_rad`` (5 joints, rad) and the jaw at
    ``gripper_rad`` (rad), at the default render scale; None for an empty ROI."""
    from ...robot.model import so101_kinematics as kin
    body, facts = hp.SO101Body(), reach_facts()
    meshes = body.meshes(None)
    poses = {"right": kin.fk_all(_model(), list(q_rad), gripper_rad=gripper_rad)}
    robot, base = silhouettes(body, meshes, poses, camera, base_link=facts["base_link"])
    return coverage(robot, base, table_roi(camera["K"], camera["T"], camera["width"], camera["height"], table_z_m, facts))
