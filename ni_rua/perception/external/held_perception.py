"""features.held_perception (on by default with external_perception): FK robot pixel mask and a kinematic held-object row in object tables.

With the switch on, every object_table request (geometry_query kind object_table and the attached obs3d table) changes
in two ways; everything else (service call, table-plane scale of table objects, columns, ids, overlay) is unchanged.

1. Robot pixels. Instead of the robot link segments (6 cm capsules), the harness renders the robot meshes (STL links
   base_link..link8 and the GLB camera mount of both arms, from the public RoboDojo robot asset directory) with the
   calibrated K and T_world_camera of the requested camera and the forward kinematics of the measured joint state
   (fingers at the measured opening), dilates the silhouette by DILATE_PX and sends it as robot_mask_b64. The service
   (hp variant, object-table-v5-hp1) then drops a mask as robot when >= 50 % of its pixels are robot pixels and cuts the
   robot pixels out of every other mask. Hence a held object is no longer deleted for lying within 6 cm of the gripper.
   The wrist camera of an arm does not render that arm's own camera housing (a camera does not see itself).

2. Held row. For each arm whose jaws are commanded closed and stopped on something (_holding: goal < 0.5 and
   0.002 m < opening < full opening - 0.004 m), the grasp centre is
   G = FK link6 + R [0.1501, 0, 0]. The service segments the held object (SAM 3.1 text instances of the phrase
   "object", plus the label of the arm's pose_plan grasp row when it has one; the instance containing the projected
   grasp centre, else the nearest within 40 px; robot pixels removed; SAM 3.1 point prompt at the nearest non-robot
   pixel as a fallback) in the requested camera and in the holding arm's wrist camera, and returns per view the
   mask-centroid ray at the FK grasp-centre camera depth (pick_anatomy/cal_grip "Kfk") and the VGGT points of the mask
   rescaled so that the points next to the grasp centre sit at that depth ("GK"). Never the floating -> table rescale.
   Centre: per group (elongated = grasp-row length/width or GK elongation >= 1.8, else compact) the estimator in
   CENTRE_MODE; shipped 'grasp' (= G) for both groups, chosen by the offline truth check (UNCERTAINTY_MM); 'kfk',
   'clamp' (Kfk within 2 cm of G) and 'axis' (G slid along the long axis to the mask centre) are kept selectable.
   Extents (length, width, height, footprint yaw) from GK of the wrist view when its mask is usable, else the
   requested camera. The row replaces the object row that is the same mask (else it is appended), with support
   'held_<arm>' and quality 'held_fk_depth'; value.held[id] gives source, arm, group, estimator, camera, a one-line
   method note with the measured uncertainty, and the long-axis direction for elongated objects (static prior: the
   link6 z axis, perpendicular to the jaws in the pad plane; held_object() also accepts a measured axis);
   compact objects carry orientation_prior up_axis_unchanged_since_grasp.

3. Held row v2 (EXTENT_MODE 'pregrasp', default; 'gk' restores the v1 row above). The harness follows the measured
   gripper state of every saved observation (sim_step order). When an arm's jaws close on something (the same holding
   test), the closure observation and its link6 pose are kept; at the first object table that needs the held row, the
   arm's object row is looked up in the newest pre-closure object tables (at most PREGRASP_TABLES, newest first): the
   row whose centre lies within 6 cm (xy) of the grasp centre at the closure, preferring the row whose mask contains the
   projected grasp centre. Its extents (length, width, height = top - bottom), centre, yaw and tilt are stored with the
   held state, with the object-centre offset in the link6 frame o = R6^T (c_row - p_link6) at the closure (its
   across-jaw component set to the jaw centre) and the row's long axis in the link6 frame. Rows of a held object then
   use the pre-grasp extents always; the centre per pre-grasp group (CENTRE_MODE_V2: elongated = length >= 2.5 x width,
   tube-like = length >= 0.09 m and width <= 0.025 m, else compact) is the carried offset p_link6 + R6 o or the FK grasp
   centre; the long axis of elongated objects is the carried row axis or the static prior (AXIS_MODE; both are given
   in value.held), with the v2.1 axis-consistency gate (AXIS_CONFLICT_DEG = 45.0) switching to the carried row axis and
   FK grasp centre when the carried row axis differs by more than 45 deg from the cross-jaw static prior. No pre-grasp
   row within 6 cm (or no closure observation) -> the v1 row, and the note says so. A release (jaws no longer holding)
   drops the carried state; a regrasp captures again. value.held[id] also names the pre-grasp table
   (pregrasp_table_source: per-run table number, request kind, observation, camera) and row centre.

Inputs: RGB, calibration, measured joints and gripper state, robot meshes, the harness's own earlier object tables; no
simulator truth. With the switch off only enabled() is consulted.

Embodiment: the text above describes the ARX-X5 RoboDojo arm (its STL/GLB meshes, link6 and the two finger joints),
whose behaviour is unchanged. The feature runs only on an embodiment whose profile lists ``held_perception`` in
``supports`` and that has a body below (``body_of``); on any other robot enabled() is False whatever the switch says.
Everything robot-specific is one ``*Body`` object:

* ``X5Body``: the constants of this module (``MESH_DIR``, ``LINKS``, ``CAMERA_LINKS``, ``GRASP_OFFSET_M``, the link6
  axes, ``DILATE_PX``, the jaw rule ``_holding``) and the packaged ARX-X5 profile (``embodiments.ARX_X5_PROFILE``, the
  one source of arms, wrist cameras and the full jaw opening 0.088 m; a run-threshold override of
  ``max_gripper_aperture_m`` does not reach this jaw rule). The capsule segments stay in the request beside the mask.
* ``SO101Body``: the run's profile (``tcp_link`` gripper_frame_link, ``gripper.approach_axis_local`` /
  ``jaw_opening_axis_local``, ``jaw_joint_names``, ``wrist_cameras``, ``gripper.empty_close_aperture_m`` /
  ``full_open_aperture_m`` / ``preset_margins_m``) and the packaged SO-101 URDF + STL archive
  (``robot_information_data/so101/geometry_model.zip``, every link's visual meshes). Differences from the X5 text:
  the mask is sent WITHOUT capsule segments, so the service's pixel rule decides (a mask is robot when >= 50 % of its
  pixels are robot pixels) and an object beside or under the gripper is kept; the silhouette is dilated by
  ``SO101Body.dilate_px`` = 5 px (real-arm FK and hand-eye error; chosen on the archived real observations, see
  docs/held_perception.md); the grasp centre is ``geometry.grasp_reference`` (the CAD fingertip
  midpoint at the measured jaw angle); the jaws hold something when the commanded goal is < 0.5 and the measured
  fingertip opening exceeds the commanded opening by more than the profile's ``non_empty`` grip-guard margin
  (``SO101Body.holding``; no held row without a measured ``empty_close_aperture_m``; when the rule cannot decide,
  ``diagnostics.held_perception.holding_undecided`` names why, ``SO101Body.undecided``); the centre-error statistics of
  the X5 truth checks are not quoted (``uncertainty_p50_mm`` null, the note says "not measured on this robot").
  Distorted cameras are sent rectified (``rectify.Rectified``); the mask is rendered in the rectified image.
"""
from __future__ import annotations

import base64

from ...core.episode_state import StateSlot, slot_get, slot_peek, slot_set
from ...core.pose_math import quat_xyzw_to_matrix as quat_matrix

HELD_PERCEPTION_SLOT: StateSlot[dict] = StateSlot(
    "held_perception",
    lambda: dict(cursor=0, step=None, arms={}, tables=[], carried={}, episode_id=None),
    scope="run",
)
_POSE_SKILL_GRASPS_SLOT: StateSlot[dict] = StateSlot("pose_skill_grasps", dict, scope="episode")
import json
import math
import struct
import time
from pathlib import Path

import numpy as np

from ...robot.model.roboprobe_pose import GRASP_POINT_OFFSET_M
from . import embodiment_of, run_embodiment

FLAG = "held_perception"
MESH_DIR = "Assets/Robots/x5/meshes"
LINKS = ("base_link", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "link8", "camera_base", "camera")
CAMERA_LINKS = ("camera_base", "camera")
DILATE_PX = 3
RENDER_SCALE = 0.5            # raster at half resolution, nearest-neighbour upsampled, then dilated at full resolution
NEAR_M = 0.01
DECIMATE_M = 0.002            # vertex clustering cell for the link meshes (silhouette error well below a pixel at 0.3 m)
GRASP_OFFSET_M = GRASP_POINT_OFFSET_M   # m, ARX-X5 gripping-face centre in front of link6 (= gripper.grasp_reference_offset_m)
WRIST_MIN_MASK_PX = 100
HEAD_MIN_MASK_PX = 20
ELONGATED = 1.8               # object_table tilt rule: elongation >= 1.8
TILT_FOR_YAW_DEG = 15.0       # below this tilt the footprint yaw is taken from the points, not from the axis
CLAMP_RADIUS_M = 0.02
AXIS_MAX_SHIFT_M = 0.08
# centre estimator per group (set from the offline truth check, bevmap/hp/offline): 'grasp' | 'kfk' | 'clamp' | 'axis'
AXIS_REF = "gk"               # 'axis' mode: along-axis offset taken from the GK point centre (else the Kfk centre)
# 2026-10-26 offline check (profile_val, 373 held objects, harness-observable groups; xy p50/p90 mm): compact grasp
# 4.8/28.3 vs kfk 8.3/47.3, clamp 8.2/26.9; elongated grasp 10.8/34.7 vs axis 14.2/49.5, clamp 16.9/36.3, kfk 24.2/56.8.
# Per truth class only test tubes favour 'axis' (14.9/21.2 vs grasp 26.0/32.6); the harness has no class names.
CENTRE_MODE = {"elongated": "grasp", "compact": "grasp"}
# (group, estimator[_view]) -> (p50, p90) centre error in mm from the offline truth check
UNCERTAINTY_MM = {("compact", "grasp"): (5, 28), ("elongated", "grasp"): (11, 35),
                  ("all", "kfk_wrist"): (10, 36), ("all", "kfk_head"): (32, 109), ("all", "clamp_wrist"): (10, 36),
                  ("all", "clamp_head"): (18, 36), ("all", "axis_wrist"): (11, 29), ("all", "axis_head"): (20, 79)}
SUPPORT = "held_%s"
QUALITY = "held_fk_depth"
SOURCE = "held_fk_depth"
INSTRUCTION = ("Object-table rows with quality held_fk_depth are candidate held-object geometry estimates, "
               "conditioned on rigid attachment to the gripper. A closed jaw or a carried pre-grasp row does not "
               "confirm retention. The held entry distinguishes a current mask from prediction_only; even a current "
               "mask is not a co-motion check. Use existing RGB and action evidence to assess actual retention, "
               "slip and support before relying on the predicted pose.")


def _retention_evidence(meta, results):
    """Expose evidence scope without inferring physical retention from jaw/FK geometry."""
    views = [view for view in ("head", "wrist") if usable(results.get(view), view)]
    meta.update(retention_status="unverified", attachment_assumption="rigid_since_grasp",
                geometry_evidence="current_mask" if views else "prediction_only",
                current_mask_views=views)
    meta["note"] += ("; retention unverified: " +
                     ("current mask available, no co-motion confirmation" if views else
                      "no usable current mask; pose predicted under rigid attachment"))
    return meta



def enabled(runtime) -> bool:
    """features.held_perception on an embodiment that supports it (``"held_perception" in profile.supports``)."""
    features = getattr(getattr(runtime, "config", None), "features", None)
    return bool(getattr(features, FLAG, False)) and FLAG in embodiment_of(runtime).supports


def instruction(features, profile=None) -> str:
    """The developer-instruction sentence (``INSTRUCTION``) when features.held_perception is on and ``profile``
    (None = the default ARX-X5 robot) supports held perception; '' otherwise."""
    profile = profile if profile is not None else embodiment_of(None)
    return INSTRUCTION if bool(getattr(features, FLAG, False)) and FLAG in profile.supports else ""


# ------------------------------------------------------------------ meshes
def load_stl(path):
    return stl_mesh(Path(path).read_bytes())


def stl_mesh(data):
    """(V, F) of a binary or ASCII STL given as bytes (vertices in the file's unit, three per triangle)."""
    n = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else 0
    if n and len(data) == 84 + 50 * n:
        rec = np.frombuffer(data, dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("x", "<u2")]), count=n, offset=84)
        V = rec["v"].reshape(-1, 3).astype(np.float64)
    else:
        V = np.array([[float(t) for t in line.split()[1:4]] for line in data.decode(errors="ignore").splitlines()
                      if line.strip().startswith("vertex")], dtype=np.float64).reshape(-1, 3)
    return V, np.arange(len(V)).reshape(-1, 3)


def _node_matrix(node):
    if "matrix" in node:
        return np.array(node["matrix"], float).reshape(4, 4).T
    M = np.eye(4)
    if "scale" in node:
        M = np.diag(list(node["scale"]) + [1.0]) @ M
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        R = np.eye(4)
        R[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        M = R @ M
    if "translation" in node:
        Tm = np.eye(4)
        Tm[:3, 3] = node["translation"]
        M = Tm @ M
    return M


def load_glb(path):
    """Triangles of a binary glTF in its own node frames (no axis conversion; as validated in pick_anatomy/cal_grip)."""
    data = Path(path).read_bytes()
    l0 = struct.unpack_from("<I", data, 12)[0]
    js = json.loads(data[20:20 + l0])
    off = 20 + l0
    l1 = struct.unpack_from("<I", data, off)[0]
    blob = data[off + 8: off + 8 + l1]

    def accessor(i):
        a = js["accessors"][i]
        view = js["bufferViews"][a["bufferView"]]
        dtype = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[a["componentType"]]
        nc = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
        start = view.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = view.get("byteStride")
        size = nc * np.dtype(dtype).itemsize
        if stride and stride != size:
            raw = np.frombuffer(blob, np.uint8, count=stride * a["count"], offset=start).reshape(a["count"], stride)
            return raw[:, :size].copy().view(dtype).reshape(a["count"], nc)
        return np.frombuffer(blob, dtype, count=a["count"] * nc, offset=start).reshape(a["count"], nc)

    verts, faces, base = [], [], 0

    def walk(index, M):
        nonlocal base
        node = js["nodes"][index]
        M2 = M @ _node_matrix(node)
        if "mesh" in node:
            for prim in js["meshes"][node["mesh"]]["primitives"]:
                V = accessor(prim["attributes"]["POSITION"]).astype(np.float64)
                F = (accessor(prim["indices"]).reshape(-1, 3).astype(np.int64) if "indices" in prim
                     else np.arange(len(V)).reshape(-1, 3))
                verts.append(V @ M2[:3, :3].T + M2[:3, 3])
                faces.append(F + base)
                base += len(V)
        for child in node.get("children", []):
            walk(child, M2)

    for root in js["scenes"][js.get("scene", 0)]["nodes"]:
        walk(root, np.eye(4))
    return np.vstack(verts), np.vstack(faces)


def decimate(V, F, cell=DECIMATE_M):
    """Vertex clustering: vertices merged per cubic cell, degenerate and repeated triangles removed."""
    keys = np.floor(V / cell).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    sums = np.zeros((len(first), 3))
    np.add.at(sums, inverse, V)
    counts = np.bincount(inverse, minlength=len(first)).astype(float)
    V2 = sums / counts[:, None]
    F2 = inverse[F]
    ok = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
    F2 = np.unique(np.sort(F2[ok], axis=1), axis=0)
    return V2, F2


_MESHES = {}


class _MeshDict(dict):
    """兼容原生 `dict` 的网格表，额外携带加载失败的连杆列表 `degraded_links`。"""

    def __init__(self, *args, degraded_links=()):
        super().__init__(*args)
        self.degraded_links = list(degraded_links)


class _ArmLinkPoses(dict):
    """兼容原生 `dict` 的双臂连杆位姿表，额外携带 FK 求解失败的手臂映射 `degraded_arms`。"""

    def __init__(self, *args, degraded_arms=None):
        super().__init__(*args)
        self.degraded_arms = dict(degraded_arms or {})


def robot_meshes(mesh_dir):
    """{link: (V, F)} decimated, cached per directory; links without a mesh file are left out."""
    key = str(mesh_dir)
    if key not in _MESHES:
        raw_out = {}
        degraded = []
        had_io_error = False
        for link in LINKS:
            stl, glb = Path(mesh_dir) / ("%s.STL" % link), Path(mesh_dir) / ("%s.glb" % link)
            has_glb, has_stl = glb.exists(), stl.exists()
            if not (has_glb or has_stl):
                continue
            try:
                V, F = load_glb(glb) if has_glb else load_stl(stl)
            except OSError:
                had_io_error = True
                degraded.append(link)
                V, F = None, None
            except (ValueError, KeyError, struct.error):
                degraded.append(link)
                V, F = None, None
            if V is not None and len(F):
                raw_out[link] = decimate(V, F)
        out = _MeshDict(raw_out, degraded_links=degraded)
        if not had_io_error:
            _MESHES[key] = out
        return out
    return _MESHES[key]


def mesh_dir(runtime):
    root = getattr(getattr(getattr(runtime, "config", None), "backend", None), "root", None)
    return Path(str(root)).expanduser() / MESH_DIR if root else None


def _rpy_matrix(roll, pitch, yaw):
    """URDF fixed-axis roll / pitch / yaw (rad) -> 3x3 rotation (Rz(yaw) Ry(pitch) Rx(roll))."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw),
                              math.sin(yaw))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def urdf_archive_meshes(archive_path, urdf_member, cell=DECIMATE_M):
    """{link: (V, F)} of a zipped URDF: every link's ``<visual>`` STL meshes (scale and origin applied, link frame, m),
    merged per link and decimated to ``cell`` (m); cached per archive. Links without a mesh are left out.

    ``archive_path``: zip holding the URDF ``urdf_member`` and the STL files its ``<mesh filename>`` paths name
    (relative to the URDF). Example: ``urdf_archive_meshes(so101_dir / "geometry_model.zip",
    "so101_new_calib_camera.urdf")["moving_jaw_so101_v1_link"]``."""
    import xml.etree.ElementTree as ElementTree
    import zipfile
    key = (str(archive_path), str(urdf_member), float(cell))
    if key not in _MESHES:
        out = {}
        with zipfile.ZipFile(archive_path) as archive:
            root = ElementTree.fromstring(archive.read(urdf_member))
            for node in root.findall("link"):
                verts, faces, base = [], [], 0
                for element in node.findall("visual"):
                    mesh = element.find("geometry/mesh")
                    if mesh is None:
                        continue
                    V, F = stl_mesh(archive.read(mesh.attrib["filename"]))
                    V = V * np.array([float(v) for v in mesh.get("scale", "1 1 1").split()])
                    origin = element.find("origin")
                    if origin is not None:
                        R = _rpy_matrix(*(float(v) for v in origin.get("rpy", "0 0 0").split()))
                        V = V @ R.T + np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
                    verts.append(V)
                    faces.append(F + base)
                    base += len(V)
                if verts:
                    out[node.get("name")] = decimate(np.vstack(verts), np.vstack(faces), cell)
        _MESHES[key] = out
    return _MESHES[key]


# ------------------------------------------------------------------ FK robot pixel mask
def render_mask(K, T, width, height, posed, *, scale=RENDER_SCALE, dilate_px=DILATE_PX):
    """posed: [(V, F, T_world_link)]. Union of the projected triangles in front of the camera (no z-buffer: a pixel is
    robot when any robot triangle covers it), rastered at `scale`, upsampled, dilated by dilate_px. bool (H, W)."""
    import cv2
    K, T = np.asarray(K, float), np.asarray(T, float)
    w2, h2 = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    sx, sy = w2 / float(width), h2 / float(height)
    Ti = np.linalg.inv(T)
    canvas = np.zeros((h2, w2), np.uint8)
    for V, F, T_link in posed:
        M = Ti @ np.asarray(T_link, float)
        Vc = V @ M[:3, :3].T + M[:3, 3]
        z = Vc[:, 2]
        tri = F[(z[F] > NEAR_M).all(axis=1)]
        if not len(tri):
            continue
        u = (K[0, 0] * Vc[:, 0] / np.maximum(z, 1e-9) + K[0, 2]) * sx - 0.5
        v = (K[1, 1] * Vc[:, 1] / np.maximum(z, 1e-9) + K[1, 2]) * sy - 0.5
        pu, pv = u[tri], v[tri]
        keep = (pu.max(1) >= -1) & (pu.min(1) <= w2) & (pv.max(1) >= -1) & (pv.min(1) <= h2)
        if not keep.any():
            continue
        pts = np.stack([pu[keep], pv[keep]], axis=-1)
        pts = np.clip(np.rint(pts * 4.0), -1e6, 1e6).astype(np.int32)       # shift=2: quarter-pixel vertices
        cv2.fillPoly(canvas, list(pts), 1, lineType=cv2.LINE_8, shift=2)
    if (w2, h2) != (width, height):
        canvas = cv2.resize(canvas, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        canvas = cv2.dilate(canvas, k)
    return canvas.astype(bool)


def posed_links(meshes, poses_by_arm, *, skip_camera_of=None, camera_links=None):
    """[(V, F, T_world_link)] of every meshed link of every arm; the arm ``skip_camera_of`` leaves out its own
    wrist-camera housing links ``camera_links`` (default: the ARX-X5 ``CAMERA_LINKS``)."""
    camera_links = CAMERA_LINKS if camera_links is None else camera_links
    out = []
    for arm, poses in poses_by_arm.items():
        for link, (V, F) in meshes.items():
            if link not in poses or (arm == skip_camera_of and link in camera_links):
                continue
            out.append((V, F, poses[link]))
    return out


def pack_mask(mask):
    mask = np.asarray(mask, bool)
    return base64.b64encode(np.packbits(mask.reshape(-1)).tobytes()).decode("ascii"), [int(mask.shape[0]),
                                                                                         int(mask.shape[1])]


def arm_of_camera(camera_id, wrist_cameras=None):
    """Arm whose wrist camera is ``camera_id`` (``wrist_cameras``: ``profile.wrist_cameras``, default ARX-X5)."""
    wrist_cameras = wrist_cameras if wrist_cameras is not None else embodiment_of(None).wrist_cameras
    for arm, camera in wrist_cameras.items():
        if camera_id == camera:
            return arm
    return None


def arm_link_poses(tools, identity, observation_id, observation):
    """{arm: {link: 4x4}} from the FK of the measured joints (one wrist calibration per arm gives its base)."""
    from ...robot.model.geometry import camera_fk
    out, degraded, model = {}, {}, None
    for frame in observation.get("cameras", []):
        cam_id = frame.get("camera_id") if isinstance(frame, dict) else None
        try:
            camera = tools.camera(identity, observation_id, cam_id)
        except Exception as exc:
            arm_hint = arm_of_camera(cam_id, getattr(getattr(tools, "profile", None), "wrist_cameras", None))
            if arm_hint and arm_hint not in out:
                degraded[arm_hint] = f"{type(exc).__name__}:{str(exc)[:120]}"
            continue
        calibration = camera.get("calibration") or {}
        arm = calibration.get("arm")
        if not arm or arm in out or not calibration.get("link_name"):
            continue
        try:
            if model is None:
                model = tools.store.resolve_json(observation["robot_model_ref"])
            _, poses = camera_fk(model, camera["state"], calibration, camera["sample"])
        except Exception as exc:
            degraded[arm] = f"{type(exc).__name__}:{str(exc)[:120]}"
            continue
        degraded.pop(arm, None)
        out[arm] = {name: np.asarray(item, dtype=np.float64) for name, item in poses.items()}
    return _ArmLinkPoses(out, degraded_arms=degraded)


def grasp_centre(T_link6):
    T_link6 = np.asarray(T_link6, float)
    return T_link6[:3, 3] + T_link6[:3, :3] @ np.array([GRASP_OFFSET_M, 0.0, 0.0])


# jaw state of a measured arm (ARX X5): sum of both finger positions and the commanded goal (the margins of
# held_end.py in PR #35, copied so that held_perception does not depend on held_end). The full opening (sum of both
# finger positions at the maximum opening, m) is the profile's gripper.full_open_aperture_m (ARX-X5 0.088 m =
# joint7 + joint8 travel; design 13 V-D).
OPEN_MARGIN_M = 0.004           # m: within this of the maximum the gripper counts as open
EMPTY_CLOSED_M = 0.002          # m: fingers closed on nothing read at most this
JAW_CLOSED_GOAL = 0.5           # goal_normalized below this = jaws commanded closed


def _holding(arm_state, full_open_m=None):
    """The jaws are commanded closed and stopped on something (neither empty nor fully open).

    ``full_open_m`` (m): the sum of both finger positions at the maximum opening (``profile.gripper.
    full_open_aperture_m``; None = the ARX-X5 profile, 0.088 m)."""
    full_open = full_open_m if full_open_m is not None else embodiment_of(None).gripper.full_open_aperture_m
    gripper = arm_state.get("gripper") if isinstance(arm_state, dict) else None
    if not isinstance(gripper, dict):
        return False
    positions = gripper.get("position_m")
    opening = float(sum(positions)) if isinstance(positions, list) and all(map(_num, positions)) else None
    goal = gripper.get("goal_normalized")
    if opening is None or not (_num(goal) and goal < JAW_CLOSED_GOAL):
        return False
    min_opening = max(EMPTY_CLOSED_M, float(goal) * full_open + EMPTY_CLOSED_M)
    return bool(min_opening < opening < full_open - OPEN_MARGIN_M)


def _read_grasp_memory(runtime) -> dict:
    memory = slot_peek(runtime, _POSE_SKILL_GRASPS_SLOT, None, legacy_attr="_pose_skill_grasps")
    return memory if isinstance(memory, dict) else {}


def held_record(runtime, arm, object_id=None):
    """The pose_plan grasp record of object_id, else the latest one of this arm (None when there is none)."""
    memory = _read_grasp_memory(runtime)
    if object_id is not None and isinstance(memory.get(str(object_id)), dict):
        return dict(memory[str(object_id)], object=str(object_id))
    records = [(v.get("held_seq", 0), k, v) for k, v in memory.items() if isinstance(v, dict) and v.get("arm") == arm]
    if not records:
        return None
    _, key, value = max(records, key=lambda item: item[0])
    return dict(value, object=key or None)


def holding_arms(state, profile=None):
    """Arms of ``profile`` (None = the ARX-X5 profile) whose jaws hold something in ``state`` (the jaw rule of the
    profile's body: ``_holding`` on the ARX-X5, ``SO101Body.holding`` on the SO-101; no arm for a robot without one)."""
    profile = profile if profile is not None else embodiment_of(None)
    body = body_of(profile)
    arms = (state or {}).get("arms") or {}
    if body is None:
        return []
    return [arm for arm in profile.arms if isinstance(arms.get(arm), dict) and body.holding(profile, arm, arms[arm])]


def grasp_label(runtime, arm):
    """Label of the arm's latest pose_plan grasp row (only when that row came from a labelled table), else None."""
    try:
        records = [(v.get("held_seq", 0), v) for v in _read_grasp_memory(runtime).values()
                   if isinstance(v, dict) and v.get("arm") == arm and v.get("label")]
    except Exception:
        return None
    return str(max(records, key=lambda item: item[0])[1]["label"])[:80] if records else None


def record_elongated(runtime, arm):
    """True / False from the length / width of the arm's latest pose_plan grasp row (pose_skill grasp memory), else None."""
    try:
        record = held_record(runtime, arm)
    except Exception:
        return None
    length, width = (record or {}).get("length"), (record or {}).get("width")
    if isinstance(length, (int, float)) and isinstance(width, (int, float)) and width > 0:
        return bool(length / width >= ELONGATED)
    return None


# ------------------------------------------------------------------ v2: pre-grasp table row carried in the link6 frame
# EXTENT_MODE 'pregrasp' (v2): at the jaw closure the arm's row of the newest pre-grasp object table is captured and
# carried rigidly with link6; 'gk' (v1): no capture, extents from the VGGT points of the held mask.
EXTENT_MODE = "pregrasp"
PREGRASP_RADIUS_M = 0.06      # pre-grasp row centre within this xy distance of the grasp centre at the closure
PREGRASP_TABLES = 4           # newest pre-closure tables searched, newest first
TABLES_KEPT = 8
TRACK_PAGE = 500
ELONG_RATIO = 2.5             # elongated: length >= 2.5 x width (or tube-like)
TUBE_MIN_LENGTH_M = 0.09      # tube-like: length >= 0.09 m and width <= 0.025 m
TUBE_MAX_WIDTH_M = 0.025
UPRIGHT_TILT_DEG = 45.0       # pre-grasp long axis within 45 deg of vertical: static prior = approach axis
GRASP_L = np.array([GRASP_OFFSET_M, 0.0, 0.0])
APPROACH_L = np.array([1.0, 0.0, 0.0])    # link6 x: flange -> fingers
CLOSING_L = np.array([0.0, 1.0, 0.0])     # link6 y: jaw closing line (held_end.gripper_frame)
THIRD_L = np.array([0.0, 0.0, 1.0])       # link6 z: perpendicular to the jaws in the pad plane
ROW_KEYS = ("centre_m", "top_z_m", "bottom_z_m", "length_m", "width_m", "yaw_deg", "axis_tilt_deg", "elongation",
            "contour_px", "flags", "quality", "support", "label")
# v2 centre per pre-grasp group: 'grasp' (FK grasp centre), 'carried' (pre-grasp row centre carried rigidly from the
# closure, across-jaw component at the jaw centre), 'carried_axis' (grasp centre moved along the long axis by the
# carried along-axis offset), 'axis' (v1 mode: grasp centre slid along the long axis to the GK point centre)
# 2026-10-26 offline replay (bevmap/hp/v2/offline, 511 captured held observations, xy p50/p90 mm): compact grasp 5.8/28.2
# vs carried 6.6/46.3; elongated carried_axis 2.5/10.1 vs carried 2.9/11.5, grasp 14.9/29.7; tube-like carried_axis
# 4.1/10.6 vs carried 5.8/13.8, grasp 22.7/32.9 (truth test tubes 3.1/10.7 vs v1 22.0/31.4)
CENTRE_MODE_V2 = {"compact": "grasp", "elongated": "carried_axis", "tube": "carried_axis"}
# v2 long axis per group: 'carried' (pre-grasp row axis carried rigidly) | 'prior' (link6 z; link6 x when the pre-grasp
# row was upright)
# offline: prior 2.8/19.6 deg (elongated), 2.2/6.8 (tube-like) vs carried 9.6/31.7, 9.0/19.6
AXIS_MODE = {"elongated": "prior", "tube": "prior"}
# (group, centre mode) -> (p50, p90) centre xy error in mm; (group, axis source) -> (p50, p90) axis error in deg
UNCERTAINTY_V2_MM = {("compact", "grasp"): (6, 28), ("compact", "carried"): (7, 46),
                     ("elongated", "carried_axis"): (3, 10), ("elongated", "carried"): (3, 12),
                     ("elongated", "grasp"): (15, 30), ("tube", "carried_axis"): (4, 11), ("tube", "carried"): (6, 14),
                     ("tube", "grasp"): (23, 33)}
AXIS_UNCERTAINTY_DEG = {("elongated", "static_prior"): (3, 20), ("tube", "static_prior"): (2, 7),
                        ("elongated", "pregrasp_row_carried"): (10, 32), ("tube", "pregrasp_row_carried"): (9, 20)}
SOURCE_V2 = "held_pregrasp_row"
# extents of a held row: 'box' = the pre-grasp row box (length x width footprint along its yaw, height) carried
# rigidly with link6 and re-measured in the world frame (long side along the shipped axis for elongated objects);
# 'row' = the pre-grasp row values unchanged
EXTENT_CARRY = "box"
# v2.1 (2026-10-27, hp21): two guards on the carried pre-grasp row; None / False restores v2 behaviour.
# Offline replay (bevmap/hp/v21/offline, 511 captured held rows with truth): the carried-vs-prior angle is p50/p90
# 8/20 deg and never above 45 deg there; at 30 deg the gate fired on 21 rows, all test tubes whose prior axis was right
# (centre xy p50 3.2 -> 20.7 mm), so the shipped threshold is 45 deg (no change on that set; the hp2h2 W1 frames,
# organize l2 figurine, are at 76 deg).
# W1 axis-consistency gate: when the pre-grasp row's long axis (carried rigidly with link6) is more than
# AXIS_CONFLICT_DEG from the static-prior axis, the object was not grasped across its long axis as the prior assumes:
# centre per AXIS_CONFLICT_CENTRE ('grasp' = FK grasp centre, no shift; 'carried_axis' = grasp centre moved along the
# carried axis by the carried offset), long axis reported = the carried axis, note axis_conflict=<deg>.
AXIS_CONFLICT_DEG = 45.0
AXIS_CONFLICT_CENTRE = "grasp"
# W2 partial-row guard: a pre-grasp row shorter than PARTIAL_ROW_RATIO x the reference length, or (PARTIAL_ROW_CONTAINS)
# whose mask outline exists and does not contain the projected grasp centre, is treated as a partial row: v1 extents
# and group (GK), extent_source=gk_partial_row. Reference length = max of the sources in PARTIAL_ROW_REF:
# 'mask' = the row's own mask outline intersected with the horizontal plane at the row mid-height (min-area rectangle
# long side), 'gk' = the held mask's VGGT extent max(length, height) at the first held request of the streak.
# Offline replay: the GK closure length does not predict a short row (correlation 0.00 with row/truth length, median
# 0.63 x truth); the mask outline does weakly (0.36, median 1.10 x truth). With ratio 0.8 + both references + the
# outline test the guard fired on 261/511 rows and the GK fallback was worse than the row even for the rows that were
# really short (rows < 0.7 x truth length: height |err| p50/p90 27/62 mm with the row, 41/150 with GK); test tubes
# 2.9/10.6 -> 5.0/24.8 mm, compact height 15/61 -> 28/110 mm. The fallback is therefore off in the shipped setting
# (PARTIAL_ROW_RATIO None, PARTIAL_ROW_CONTAINS False); the task-sheet setting is 0.8 / True / ('mask', 'gk').
PARTIAL_ROW_RATIO = None
PARTIAL_ROW_CONTAINS = False
PARTIAL_ROW_REF = ("mask",)
# group of a non-partial row from max(row length, reference length) with the row width (v2: row length). Offline it
# changed 59/511 groups (compact -> elongated, mostly tall objects whose mask outline is long): compact height
# |err| p90 61 -> 77 mm, toy vehicles 81 -> 127 mm, tubes unchanged; it fixed the hp2h2 insert l0 tube2 row (lower end
# 36 -> 10 mm). Off in the shipped setting (task-sheet setting: on).
GROUP_FROM_MAX = False


def _num(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _unit_up(v):
    v = np.asarray(v, float)
    v = v / max(float(np.linalg.norm(v)), 1e-12)
    return -v if v[2] < 0 else v


def _ee_matrix(ee_pose):
    T = np.eye(4)
    T[:3, :3] = quat_matrix(ee_pose["quaternion_xyzw"])
    T[:3, 3] = [float(v) for v in ee_pose["position_m"]]
    return T


def axis_from(yaw_deg, tilt_deg):
    """Unit long axis from a table row's footprint yaw and axis tilt from vertical (tilt unknown -> horizontal)."""
    if not _num(yaw_deg):
        return None
    theta, psi = math.radians(float(tilt_deg) if _num(tilt_deg) else 90.0), math.radians(float(yaw_deg))
    return np.array([math.sin(theta) * math.cos(psi), math.sin(theta) * math.sin(psi), math.cos(theta)])


def pregrasp_group(length, width):
    """'tube' (length >= 0.09 m, width <= 0.025 m), 'elongated' (length >= 2.5 x width) or 'compact'."""
    if not (_num(length) and _num(width)):
        return "compact"
    if length >= TUBE_MIN_LENGTH_M and width <= TUBE_MAX_WIDTH_M:
        return "tube"
    if width <= 0 or length >= ELONG_RATIO * width:
        return "elongated"
    return "compact"


_IDENTITY_KEYS = ("run_id", "env_instance_id", "episode_id", "env_generation")


def tracker(runtime):
    """Per-run v2 state: observation cursor, per-arm holding streaks, recent object tables, carried rows."""
    memory = slot_get(runtime, HELD_PERCEPTION_SLOT, legacy_attr="_held_perception")
    if not isinstance(memory, dict):
        memory = dict(cursor=0, step=None, arms={}, tables=[], carried={}, episode_id=None, identity_key=None)
        slot_set(runtime, HELD_PERCEPTION_SLOT, memory, legacy_attr="_held_perception")
    ident = getattr(runtime, "identity", None)
    if isinstance(ident, dict):
        cur_ep = ident.get("episode_id")
        cur_ident_key = tuple(ident.get(k) for k in _IDENTITY_KEYS)
        prev_ident_key = memory.get("identity_key")
        prev_ep = memory.get("episode_id")
        if (prev_ident_key is not None and any(cur_ident_key) and prev_ident_key != cur_ident_key) or (
            prev_ep is not None and cur_ep is not None and prev_ep != cur_ep
        ):
            memory.update(step=None, arms={}, tables=[], carried={})
        if cur_ep is not None:
            memory["episode_id"] = cur_ep
        if any(cur_ident_key):
            memory["identity_key"] = cur_ident_key
    return memory


def _arms_of(runtime, bundle):
    _, data = runtime.store.resolve_artifact(bundle["robot_state_ref"], allowed_scopes=("actor",))
    return json.loads(data).get("arms") or {}


def track(runtime, max_pages=40):
    """Reads the observation records saved since the last call (sim_step order; a record at an earlier or the same
    step, such as a replayed historical frame, is skipped) and updates each arm's holding streak: the first holding
    observation of a streak is its closure (observation id, sim_step, measured link6 pose); a non-holding observation
    ends the streak and drops the carried row; a new streak (regrasp) drops it too, so the next table captures again."""
    memory = tracker(runtime)
    store = getattr(runtime, "store", None)
    if store is None or not hasattr(store, "record_page"):
        return memory
    ident = getattr(runtime, "identity", None)
    ident_dict = ident if isinstance(ident, dict) else {}
    body, profile = body_and_profile(runtime)   # ARX-X5: the packaged profile (module docstring)
    for _ in range(max_pages):
        page = store.record_page("observation", after=int(memory["cursor"]), limit=TRACK_PAGE)
        if page.get("reset") and memory["cursor"]:
            memory.update(cursor=0, step=None, arms={}, tables=[], carried={})
        for entry in page.get("entries") or []:
            record = entry.get("value") or {}
            if ident_dict and any(
                ident_dict.get(k) is not None and record.get(k) is not None and record.get(k) != ident_dict.get(k)
                for k in _IDENTITY_KEYS
            ):
                continue
            step = record.get("sim_step")
            if not _num(step) or (memory["step"] is not None and step <= memory["step"]):
                continue
            try:
                arms = _arms_of(runtime, record)
            except Exception as exc:
                # 中途帧状态读取失败形成夹爪连续性断口：清空 carried 与活跃持握链，杜绝跨断口幽灵持物残留
                memory["carried"].clear()
                for streak in memory["arms"].values():
                    if isinstance(streak, dict):
                        streak["holding"] = False
                        streak["closure"] = None
                memory["last_state_gap_error"] = f"{type(exc).__name__}:{str(exc)[:120]}"
                continue
            memory["step"] = step
            for arm in profile.arms:
                state = arms.get(arm)
                if not isinstance(state, dict):
                    continue
                holding = bool(body.holding(profile, arm, state))
                streak = memory["arms"].setdefault(arm, dict(holding=False, closure=None))
                if holding and not streak["holding"]:
                    try:
                        T6 = _ee_matrix(state["ee_pose"]).tolist()
                    except Exception:
                        T6 = None
                    streak["closure"] = (dict(observation_id=record.get("observation_id") or entry.get("id"),
                                              sim_step=step, T6=T6, **body.closure_facts(profile, arm, state))
                                         if T6 is not None else None)
                    memory["carried"].pop(arm, None)
                elif not holding and streak["holding"]:
                    streak["closure"] = None
                    memory["carried"].pop(arm, None)
                streak["holding"] = holding
        memory["cursor"] = int(page.get("cursor") or memory["cursor"])
        if not page.get("has_more"):
            break
    return memory


def record_table(runtime, observation_id, step, camera_id, camera, objects, poses, free_arms, source=None,
                 tcp_link="link6"):
    """Keeps the rows of an object table computed while an arm was free (its candidate pre-grasp rows). `source`
    identifies the object-table request (per-run sequence number, request kind), kept with the table. ``tcp_link``:
    the link whose pose is kept per free arm (stored under the historical key ``link6``; the profile's ``tcp_link``)."""
    if not free_arms or not _num(step):
        return
    memory = tracker(runtime)
    rows = []
    for index, item in enumerate(objects):
        if (not isinstance(item, dict) or not isinstance(item.get("centre_m"), list) or item.get("quality") == QUALITY
                or str(item.get("support") or "").startswith("held_")):
            continue
        rows.append(dict(index=index, **{key: item.get(key) for key in ROW_KEYS}))
    tables = [t for t in memory["tables"] if (t["observation_id"], t["camera_id"]) != (observation_id, camera_id)]
    tables.append(dict(observation_id=observation_id, sim_step=step, camera_id=camera_id,
                       K=np.asarray(camera["K"], float).tolist(), T=np.asarray(camera["T"], float).tolist(),
                       rows=rows, free_arms=sorted(free_arms), source=dict(source or {}),
                       size=[int(camera["width"]), int(camera["height"])] if camera.get("width") else None,
                       link6={arm: np.asarray(poses[arm][tcp_link], float).tolist() for arm in free_arms
                              if tcp_link in poses.get(arm, {})}))
    tables.sort(key=lambda t: t["sim_step"])
    memory["tables"] = tables[-TABLES_KEPT:]


def _project(K, T, point):
    T = np.asarray(T, float)
    Ti = np.linalg.inv(T)
    pc = Ti[:3, :3] @ np.asarray(point, float) + Ti[:3, 3]
    if pc[2] <= NEAR_M:
        return None
    uv = np.asarray(K, float) @ pc
    return float(uv[0] / uv[2]), float(uv[1] / uv[2])


def _contains(contour, pixel):
    import cv2
    if pixel is None or not contour or len(contour) < 3:
        return False
    return cv2.pointPolygonTest(np.asarray(contour, np.float32).reshape(-1, 1, 2), pixel, False) >= 0


def mask_length(K, T, contour, z):
    """Long side (m) of the min-area rectangle of the mask outline `contour` (pixels) intersected with the horizontal
    plane at height z; None when the outline is missing or a ray does not reach the plane in front of the camera."""
    if not contour or len(contour) < 3 or not _num(z):
        return None
    T = np.asarray(T, float)
    uv1 = np.hstack([np.asarray(contour, float).reshape(-1, 2), np.ones((len(contour), 1))])
    d = (T[:3, :3] @ (np.linalg.inv(np.asarray(K, float)) @ uv1.T)).T
    o = T[:3, 3]
    if np.any(np.abs(d[:, 2]) < 1e-9):
        return None
    t = (float(z) - o[2]) / d[:, 2]
    if np.any(t <= 0):
        return None
    points = o + t[:, None] * d
    return float(_footprint(points[:, :2])[0])


def pregrasp_row(table, grasp):
    """(row, contains, xy distance) of the table row nearest the grasp centre within PREGRASP_RADIUS_M, rows whose mask
    contains the projected grasp centre first; None when no row is that close."""
    pixel = _project(table["K"], table["T"], grasp)
    best = None
    for row in table["rows"]:
        d = float(np.linalg.norm(np.asarray(row["centre_m"][:2], float) - np.asarray(grasp[:2], float)))
        if d > PREGRASP_RADIUS_M:
            continue
        inside = _contains(row.get("contour_px"), pixel)
        key = (0 if inside else 1, d)
        if best is None or key < best[0]:
            best = (key, row, inside, d)
    return None if best is None else best[1:]


def _frame_of(source):
    """(grasp point, approach axis, jaw closing line, third axis) in the TCP frame kept in a closure / carried dict
    (``X5Body`` / ``SO101Body.closure_facts``); the ARX-X5 link6 constants when the dict carries none."""
    return tuple(np.asarray(source.get(key) if source.get(key) is not None else default, float)
                 for key, default in (("grasp_local", GRASP_L), ("approach_local", APPROACH_L),
                                      ("closing_local", CLOSING_L), ("third_local", THIRD_L)))


def capture(memory, arm, closure):
    """Carried state of arm's grasp from the newest pre-closure tables, or (None, reason)."""
    T6 = np.asarray(closure["T6"], float)
    R6, p6 = T6[:3, :3], T6[:3, 3]
    grasp_l, approach_l, closing_l, third_l = _frame_of(closure)
    frame = {key: closure[key] for key in ("grasp_local", "approach_local", "closing_local", "third_local")
             if closure.get(key) is not None}
    grasp = p6 + R6 @ grasp_l
    examined = 0
    for table in reversed(memory["tables"]):
        if table["sim_step"] >= closure["sim_step"] or arm not in table["free_arms"]:
            continue
        examined += 1
        if examined > PREGRASP_TABLES:
            break
        found = pregrasp_row(table, grasp)
        if found is None:
            continue
        row, inside, distance = found
        contour = row.get("contour_px")
        pixel = _project(table["K"], table["T"], grasp)
        size = table.get("size")       # (width, height) when known: the outline test needs the grasp point in view
        has_contour = (bool(contour) and len(contour) >= 3 and pixel is not None
                       and (not size or (0 <= pixel[0] < size[0] and 0 <= pixel[1] < size[1])))
        centre = np.asarray(row["centre_m"], float)
        offset = R6.T @ (centre - p6)
        offset_jaw = offset - float((offset - grasp_l) @ closing_l) * closing_l
        length, width = row.get("length_m"), row.get("width_m")
        top, bottom = row.get("top_z_m"), row.get("bottom_z_m")
        height = float(top) - float(bottom) if _num(top) and _num(bottom) else None
        try:
            outline = mask_length(table["K"], table["T"], contour,
                                  (float(top) + float(bottom)) / 2 if _num(top) and _num(bottom) else centre[2])
        except Exception:
            outline = None
        yaw, tilt = row.get("yaw_deg"), row.get("axis_tilt_deg")
        axis = axis_from(yaw, tilt)
        upright = _num(tilt) and float(tilt) < UPRIGHT_TILT_DEG
        footprint = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0]) if _num(yaw) else None
        return dict(closure_step=closure["sim_step"], closure_observation_id=closure["observation_id"],
                    observation_id=table["observation_id"], camera_id=table["camera_id"], sim_step=table["sim_step"],
                    row_id="o%d" % (int(row["index"]) + 1), contains=bool(inside), has_contour=has_contour,
                    mask_length_m=outline, closure_length_m=None, table_source=dict(table.get("source") or {}),
                    distance_m=round(distance, 4),
                    tables_back=examined - 1, centre_m=centre.tolist(), length=length, width=width, height=height,
                    top_z=top, bottom_z=bottom, yaw_deg=yaw, tilt_deg=tilt, elongation=row.get("elongation"),
                    flags=list(row.get("flags") or []), group=pregrasp_group(length, width), upright=bool(upright),
                    offset_local=offset_jaw.tolist(), offset_row_local=offset.tolist(),
                    axis_local=None if axis is None else (R6.T @ axis).tolist(),
                    prior_local=(approach_l if upright else third_l).tolist(), **frame,
                    footprint_local=None if footprint is None else (R6.T @ footprint).tolist(),
                    box_local=None if footprint is None else [(R6.T @ v).tolist() for v in (
                        footprint, np.cross([0.0, 0.0, 1.0], footprint), np.array([0.0, 0.0, 1.0]))],
                    link6_table=table["link6"].get(arm), link6_closure=T6.tolist()), None
    return None, ("no_pregrasp_table" if examined == 0 else "no_pregrasp_row_within_%dcm" % round(100 * PREGRASP_RADIUS_M))


def _footprint(points_xy, step_deg=0.5):
    """(long side, short side, yaw of the long side in deg [-90, 90)) of the minimum-area rectangle of 2D points."""
    best = None
    for a in np.radians(np.arange(0.0, 90.0, step_deg)):
        c, s = math.cos(a), math.sin(a)
        u, v = points_xy @ np.array([c, s]), points_xy @ np.array([-s, c])
        du, dv = float(u.max() - u.min()), float(v.max() - v.min())
        if best is None or du * dv < best[0] - 1e-12:
            best = (du * dv, du, dv, a)
    _, du, dv, a = best
    yaw = math.degrees(a) if du >= dv else math.degrees(a) + 90.0
    return max(du, dv), min(du, dv), (yaw + 90.0) % 180.0 - 90.0


def carried_box(carried, R6, axis=None):
    """World (length, width, height, footprint yaw) of the pre-grasp row box carried rigidly with link6 (R6 = link6
    rotation now); with `axis` the long side follows it. Row values unchanged when the box is unknown."""
    dims = [carried.get(k) for k in ("length", "width", "height")]
    if EXTENT_CARRY != "box" or carried.get("box_local") is None or not all(_num(d) for d in dims):
        return dims[0], dims[1], dims[2], None
    A = [np.asarray(R6, float) @ np.asarray(v, float) for v in carried["box_local"]]
    if axis is not None:
        u = np.asarray(axis, float) / max(float(np.linalg.norm(axis)), 1e-12)
        u = u if float(u @ A[0]) >= 0 else -u
        g = A[1] - float(A[1] @ u) * u
        if float(np.linalg.norm(g)) < 1e-6:
            g = A[2] - float(A[2] @ u) * u
        g = g / max(float(np.linalg.norm(g)), 1e-12)
        A = [u, g, np.cross(u, g)]
    signs = np.array([[a, b, c] for a in (-0.5, 0.5) for b in (-0.5, 0.5) for c in (-0.5, 0.5)])
    corners = (signs * np.asarray(dims, float)) @ np.vstack(A)
    length, width, yaw = _footprint(corners[:, :2])
    return length, width, float(corners[:, 2].max() - corners[:, 2].min()), yaw


def _gk_of(results):
    """(view, GK extent dict) of the held mask as the v1 row uses it (wrist preferred), or (None, None)."""
    view = "wrist" if usable(results.get("wrist"), "wrist") else "head" if usable(results.get("head"), "head") else None
    if view is None:
        return None, None
    other = results.get("head" if view == "wrist" else "wrist") or {}
    gk = results[view].get("gk") or (other.get("gk") if usable(other, "head") else None)
    return view, gk


def note_closure_length(carried, results):
    """Sets carried['closure_length_m'] once per streak: the held mask's VGGT extent max(length, height) at the first
    held request with a GK extent."""
    if carried.get("closure_length_m") is not None:
        return
    _, gk = _gk_of(results)
    if not gk:
        return
    values = [gk.get("length_m")]
    if _num(gk.get("top_z_m")) and _num(gk.get("low_z_m")):
        values.append(float(gk["top_z_m"]) - float(gk["low_z_m"]))
    values = [float(v) for v in values if _num(v)]
    if values:
        carried["closure_length_m"] = max(values)


def reference_length(carried):
    """Reference length of the grasped object for the W2 guard (max of the PARTIAL_ROW_REF sources), or None."""
    values = [carried.get({"mask": "mask_length_m", "gk": "closure_length_m"}[k]) for k in PARTIAL_ROW_REF or ()]
    values = [float(v) for v in values if _num(v)]
    return max(values) if values else None


def partial_row(carried):
    """None when the pre-grasp row is taken as complete, else dict(reasons, length_m, reference_m)."""
    if PARTIAL_ROW_RATIO is None:
        return None
    length, ref = carried.get("length"), reference_length(carried)
    reasons = []
    if _num(length) and ref is not None and float(length) < PARTIAL_ROW_RATIO * ref:
        reasons.append("short")
    if PARTIAL_ROW_CONTAINS and carried.get("has_contour") and not carried.get("contains"):
        reasons.append("outline")
    return dict(reasons=reasons, length_m=length, reference_m=ref) if reasons else None


def carried_group(carried):
    """Group of a carried row: from max(row length, reference length) with GROUP_FROM_MAX, else the captured group."""
    ref = reference_length(carried) if GROUP_FROM_MAX else None
    length = carried.get("length")
    if ref is None or not _num(length) or ref <= float(length):
        return carried["group"]
    return pregrasp_group(max(float(length), ref), carried.get("width"))


def _angle_deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = abs(float(a @ b)) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
    return math.degrees(math.acos(min(1.0, c)))


def carried_for(runtime, arm, step=None):
    """(carried state, None) for the arm's current holding streak, captured once per streak; else (None, reason)."""
    memory = tracker(runtime)
    streak = memory["arms"].get(arm) or {}
    closure = streak.get("closure")
    if not streak.get("holding") or not closure:
        return None, "no_closure_observation"
    if _num(step) and step < closure["sim_step"]:
        return None, "observation_before_closure"
    cached = memory["carried"].get(arm)
    if cached and cached.get("closure_step") == closure["sim_step"]:
        return (None, cached["failed"]) if cached.get("failed") else (cached, None)
    made, reason = capture(memory, arm, closure)
    memory["carried"][arm] = made if made is not None else dict(closure_step=closure["sim_step"], failed=reason)
    return made, reason


def _table_source(carried):
    """The object-table request the pre-grasp row comes from: per-run table number, request kind, observation, camera."""
    source = dict(carried.get("table_source") or {})
    source.update(observation_id=carried.get("observation_id"), camera_id=carried.get("camera_id"))
    return source


def _source_text(carried):
    source = carried.get("table_source") or {}
    parts = []
    if source.get("seq") is not None:
        parts.append("object table #%s" % source["seq"])
    if source.get("request"):
        parts.append("%s request" % source["request"])
    centre = carried.get("centre_m")
    if centre:
        parts.append("row centre (%.3f, %.3f, %.3f)" % tuple(float(v) for v in centre[:3]))
    return (", " + ", ".join(parts)) if parts else ""


def held_object_carried(results, arm, T_link6, *, camera_ids, carried, axis_measured=None, modes=None, axis_modes=None,
                        grasp_local=None, uncertainty=True):
    """v2 object dict + held meta from the carried pre-grasp row (extents always from it). ``grasp_local`` /
    ``uncertainty``: as ``held_object``."""
    modes = dict(CENTRE_MODE_V2, **(modes or {}))
    axis_modes = dict(AXIS_MODE, **(axis_modes or {}))
    group = carried_group(carried)
    T6 = np.asarray(T_link6, float)
    R6, p6 = T6[:3, :3], T6[:3, 3]
    grasp = p6 + R6 @ (_frame_of(carried)[0] if grasp_local is None else np.asarray(grasp_local, float))
    carried_centre = p6 + R6 @ np.asarray(carried["offset_local"], float)
    axes, axis, axis_source = {}, None, None
    if group != "compact":
        if carried.get("axis_local") is not None:
            axes["carried"] = _unit_up(R6 @ np.asarray(carried["axis_local"], float))
        axes["prior"] = _unit_up(R6 @ np.asarray(carried["prior_local"], float))
        if axis_measured is not None:
            axis, axis_source = _unit_up(axis_measured), "held_end"
        else:
            choice = axis_modes.get(group, "prior")
            choice = choice if choice in axes else "prior"
            axis, axis_source = axes[choice], ("pregrasp_row_carried" if choice == "carried" else "static_prior")
    conflict = None
    if AXIS_CONFLICT_DEG is not None and axis_source == "static_prior" and "carried" in axes:
        theta = _angle_deg(axes["carried"], axes["prior"])
        if theta > AXIS_CONFLICT_DEG:      # W1: not grasped across the long axis the prior assumes
            conflict, axis, axis_source = theta, axes["carried"], "pregrasp_row_carried"
    view = "wrist" if usable(results.get("wrist"), "wrist") else "head" if usable(results.get("head"), "head") else None
    chosen = results[view] if view else {}
    mode = modes.get(group, "grasp") if conflict is None else (
        "carried_axis" if AXIS_CONFLICT_CENTRE == "carried_axis" else "grasp")
    if mode in ("carried_axis", "axis") and axis is None:
        mode = "carried"
    if mode == "axis" and not chosen.get("gk"):
        mode = "carried_axis"
    if mode == "grasp":
        centre = grasp
    elif mode == "carried":
        centre = carried_centre
    elif mode == "carried_axis":
        t = float(np.clip(np.dot(carried_centre - grasp, axis), -AXIS_MAX_SHIFT_M, AXIS_MAX_SHIFT_M))
        centre = grasp + t * axis
    else:
        centre = centre_estimate("axis", chosen["gk"]["centre_m"], grasp, axis=axis)
    centre = [float(v) for v in centre]
    length, width, height, box_yaw = carried_box(carried, R6, axis)
    tilt = yaw = None
    if axis is not None:
        tilt = math.degrees(math.acos(min(1.0, abs(float(axis[2])))))
        if tilt >= TILT_FOR_YAW_DEG:
            yaw = (math.degrees(math.atan2(axis[1], axis[0])) + 90.0) % 180.0 - 90.0
    if yaw is None:
        yaw = box_yaw
    if yaw is None and carried.get("footprint_local") is not None:
        f = R6 @ np.asarray(carried["footprint_local"], float)
        if math.hypot(f[0], f[1]) > 1e-6:
            yaw = (math.degrees(math.atan2(f[1], f[0])) + 90.0) % 180.0 - 90.0
    head_ok = usable(results.get("head"), "head")
    obj = dict(centre_m=centre, top_z_m=centre[2] + height / 2 if height is not None else None,
               bottom_z_m=centre[2] - height / 2 if height is not None else None,
               length_m=length, width_m=width, yaw_deg=yaw, axis_tilt_deg=tilt,
               elongation=ELONGATED if axis is not None else min(float(carried.get("elongation") or 0.0), 1.0),
               support=SUPPORT % arm, flags=[], quality=QUALITY, label=None,
               contour_px=results["head"].get("contour_px") if head_ok else [],
               label_px=results["head"].get("label_px") if head_ok else None, held_arm=arm)
    camera = camera_ids.get(view) if view else None
    p50, p90 = UNCERTAINTY_V2_MM.get((group, mode), (None, None)) if uncertainty else (None, None)
    kind = {"tube": "tube-like", "elongated": "elongated", "compact": "compact"}[group]
    method = {"grasp": "the gripper-kinematics grasp centre",
              "carried": "the pre-grasp row centre carried with the gripper since the closure (across the jaws: the "
                         "jaw centre)",
              "carried_axis": "the gripper-kinematics grasp centre moved along the long axis by the pre-grasp row's "
                              "offset from the grasp point",
              "axis": "the gripper-kinematics grasp centre shifted along the long axis to the %s mask centre" % camera}[mode]
    note = ("in the %s gripper (%s object): centre = %s; extents from the pre-grasp table row%s "
            "(pregrasp_row_id=%s of %s%s, extent_source=pregrasp_row)" % (
                arm, kind, method, " turned with the gripper" if box_yaw is not None else "", carried["row_id"],
                carried["observation_id"], _source_text(carried)))
    if conflict is not None:
        note += ("; axis_conflict=%.0f deg (pre-grasp row axis vs the static prior): %s" % (
            conflict, "no shift along the axis" if mode == "grasp" else "shift along the pre-grasp row axis"))
    if p50 is not None:
        note += "; centre error p50 ~%d mm, p90 ~%d mm" % (p50, p90)
    elif not uncertainty:
        note += UNMEASURED_NOTE
    if axis is not None:
        prior_text = "along the approach axis" if carried.get("upright") else "perpendicular to the jaws in the pad plane"
        note += "; long axis: %s" % {"held_end": "held_end measurement",
                                     "pregrasp_row_carried": "pre-grasp row axis carried with the gripper",
                                     "static_prior": "static prior (%s)" % prior_text}[axis_source]
        a50 = AXIS_UNCERTAINTY_DEG.get((group, axis_source)) if uncertainty else None
        if a50:
            note += " (error p50 ~%d deg, p90 ~%d deg)" % a50
    else:
        note += "; orientation prior: up axis as before the grasp (assumed unchanged, not measured)"
    meta = dict(source=SOURCE_V2, arm=arm, group=group, centre=mode, camera_id=camera, mask=chosen.get("mask_source"),
                note=note, uncertainty_p50_mm=p50, extents="pregrasp_row", extent_source="pregrasp_row",
                pregrasp_row_id=carried["row_id"], pregrasp_observation_id=carried["observation_id"],
                closure_observation_id=carried["closure_observation_id"],
                pregrasp_row_centre_m=[_r(v, 3) for v in carried["centre_m"]],
                pregrasp_table_source=_table_source(carried))
    if conflict is not None:
        meta["axis_conflict_deg"] = round(conflict, 1)
    meta["long_axis"] = (dict(direction=[_r(v, 3) for v in axis], source=axis_source,
                              carried=[_r(v, 3) for v in axes["carried"]] if "carried" in axes else None,
                              prior=[_r(v, 3) for v in axes["prior"]]) if axis is not None else None)
    if axis is None:
        meta["orientation_prior"] = "up_axis_unchanged_since_grasp"
    return obj, _retention_evidence(meta, results)


# ------------------------------------------------------------------ embodiment bodies
class X5Body:
    """ARX-X5 (RoboDojo) robot facts of held perception: the historical constants of this module, unchanged."""

    camera_links = CAMERA_LINKS          # wrist-camera housing links, not rendered in that arm's own wrist view
    segments_with_mask = True            # the capsule segments stay in the object-table request beside the pixel mask
    measured_uncertainty = True          # UNCERTAINTY_* come from the RoboDojo truth checks of this robot

    def profile(self, runtime):
        """The packaged ARX-X5 profile (module docstring: a run-threshold override does not reach the jaw rule)."""
        return embodiment_of(None)

    def meshes(self, runtime):
        directory = mesh_dir(runtime)
        return robot_meshes(directory) if directory is not None else {}

    def mask(self, camera, posed):
        return render_mask(camera["K"], camera["T"], int(camera["width"]), int(camera["height"]), posed)

    def holding(self, profile, arm, arm_state):
        return _holding(arm_state, profile.gripper.full_open_aperture_m)

    def undecided(self, profile, arm, arm_state):
        """None: the ARX-X5 jaw rule ``_holding`` is unchanged and reports no undecided case."""
        return None

    def grasp_local(self, profile, arm, arm_state):
        return GRASP_L

    def third_local(self, profile):
        return THIRD_L

    def closure_facts(self, profile, arm, arm_state):
        return {}                        # capture() and held_object_carried() default to the link6 constants


class SO101Body:
    """SO-101 robot facts of held perception, from the run's profile and the packaged URDF + STL archive.

    ``dilate_px`` (px at ``dilate_reference_width_px`` = 640 px image width): dilation of the FK silhouette, covering
    how far the real arm's outline lies from the posed CAD mesh in the image (FK of the measured joints, hand-eye
    calibration, printed parts). On the archived real 640 x 480 observations 5 px covers the jaw outline in the wrist
    view as well as 8 px does without removing objects beside the gripper (docs/held_perception.md). That error is an
    angle, so a wider image of the same camera needs proportionally more pixels: ``dilate_for(width)`` = round(5 x
    width / 640) px (5 at 640, 10 at 1280, 15 at 1920; unchanged at the 640 x 480 default)."""

    camera_links = ("wrist_camera_mount_link", "wrist_camera_link")
    segments_with_mask = False           # mask only: the service's pixel rule decides which masks are robot
    measured_uncertainty = False         # no truth check on this robot: no centre-error statistics are quoted
    dilate_px = 5
    dilate_reference_width_px = 640

    def dilate_for(self, width):
        """Silhouette dilation (px) of an image ``width`` px wide (class docstring)."""
        return int(round(self.dilate_px * float(width) / self.dilate_reference_width_px))

    def profile(self, runtime):
        return embodiment_of(runtime)

    def meshes(self, runtime):
        from ...robot.model import embodiments
        native = embodiments.physical_facts("so101")["geometry"]["native_model"]
        directory = Path(embodiments.__file__).with_name("robot_information_data") / "so101"
        return urdf_archive_meshes(directory / (native["resource_id"] + ".zip"), native["assembly_member"])

    def mask(self, camera, posed):
        width = int(camera["width"])
        return render_mask(camera["K"], camera["T"], width, int(camera["height"]), posed,
                           dilate_px=self.dilate_for(width))

    @staticmethod
    def jaw_rad(profile, arm, arm_state):
        """Measured jaw joint angle (rad) of ``arm`` in a robot-state arm entry, or None."""
        names, values = arm_state.get("joint_names"), arm_state.get("q_rad")
        jaw = (profile.jaw_joint_names.get(arm) or (None,))[0]
        if not isinstance(names, list) or not isinstance(values, list) or jaw not in names:
            return None
        value = values[names.index(jaw)] if names.index(jaw) < len(values) else None
        return float(value) if _num(value) else None

    def _opening(self, profile, arm, arm_state):
        """(excess, measured) in m as ``opening_excess_m``, or the reason (str) why it is unknown."""
        from ...robot.model import so101_gripper_geometry as cad
        gripper = arm_state.get("gripper") if isinstance(arm_state, dict) else None
        goal = gripper.get("goal_normalized") if isinstance(gripper, dict) else None
        if not _num(goal):
            return "no_goal_normalized"
        if self.jaw_rad(profile, arm, arm_state) is None:
            return "no_measured_jaw_angle"
        empty = profile.gripper.empty_close_aperture_m
        if empty is None:
            return "empty_close_aperture_m_not_measured"
        closed = cad.jaw_rad_for_opening(empty)
        if closed is None:
            return "empty_close_aperture_m_outside_cad_opening_range"
        opened = cad.jaw_rad_for_opening(profile.gripper.full_open_aperture_m)
        if opened is None:
            return "full_open_aperture_m_outside_cad_opening_range"
        commanded = cad.opening_width_m(closed + min(1.0, max(0.0, float(goal))) * (opened - closed))
        measured = cad.opening_width_m(self.jaw_rad(profile, arm, arm_state))
        return measured - commanded, measured

    def opening_excess_m(self, profile, arm, arm_state):
        """(measured fingertip opening - commanded opening, measured opening) in m, or None when unknown.

        Measured opening: the CAD fingertip gap at the measured jaw angle (``gripper.aperture_reading``). Commanded
        opening: the CAD gap at the jaw angle ``closed + goal_normalized x (open - closed)``, ``closed`` / ``open`` =
        the jaw angles whose CAD gap equals the profile's measured ``empty_close_aperture_m`` /
        ``full_open_aperture_m``. None without a goal, a jaw angle or a measured empty close, or when either
        measured opening lies outside the CAD opening range of the declared jaw limits (``undecided`` names which)."""
        found = self._opening(profile, arm, arm_state)
        return found if isinstance(found, tuple) else None

    def undecided(self, profile, arm, arm_state):
        """Why ``holding`` cannot decide for this arm entry, or None when it can: a jaw that is not commanded closed
        (goal_normalized >= JAW_CLOSED_GOAL) is decided (not holding); a jaw commanded closed (or without a goal)
        whose ``opening_excess_m`` is unknown is not, and ``holding`` then returns False. Reasons:
        ``no_goal_normalized``, ``no_measured_jaw_angle``, ``empty_close_aperture_m_not_measured``,
        ``empty_close_aperture_m_outside_cad_opening_range``, ``full_open_aperture_m_outside_cad_opening_range``.
        Reported in the object table's ``diagnostics.held_perception.holding_undecided`` (operator diagnostics)."""
        found = self._opening(profile, arm, arm_state)
        if isinstance(found, tuple):
            return None
        goal = (arm_state.get("gripper") or {}).get("goal_normalized") if isinstance(arm_state, dict) else None
        return None if _num(goal) and goal >= JAW_CLOSED_GOAL else found

    def holding(self, profile, arm, arm_state):
        """The jaws are commanded closed (goal_normalized < JAW_CLOSED_GOAL) and stopped on something: the measured
        opening exceeds the commanded opening by more than the profile's ``non_empty`` grip-guard margin
        (``gripper.preset_margins_m``, the margin of the grip guard's ``non_empty`` preset above the empty close;
        EMPTY_CLOSED_M when the profile has none) and is not fully open (< full_open_aperture_m - OPEN_MARGIN_M).
        False when ``opening_excess_m`` is unknown (``undecided`` gives the reason). On the 404 archived real readings
        with the jaw commanded closed (tests/fixtures/so101_jaw_readings.json, docs/held_perception.md) the excess was
        -4.3 to 0.0 mm where the jaw closed to its command and +9.2 to +24.9 mm where it stopped on an object."""
        found = self.opening_excess_m(profile, arm, arm_state)
        if found is None or not arm_state["gripper"]["goal_normalized"] < JAW_CLOSED_GOAL:
            return False
        excess, measured = found
        margin = (profile.gripper.preset_margins_m.get("non_empty") or (None,))[0]
        margin = EMPTY_CLOSED_M if margin is None else float(margin)
        return bool(excess > margin and measured < profile.gripper.full_open_aperture_m - OPEN_MARGIN_M)

    def grasp_local(self, profile, arm, arm_state):
        """Grasp centre in the TCP frame (m): ``geometry.grasp_reference`` at the measured jaw angle, or None."""
        from ...robot.model.geometry import grasp_reference
        jaw = self.jaw_rad(profile, arm, arm_state)
        found = grasp_reference(profile.gripper, np.zeros(3), np.eye(3), jaw) if jaw is not None else None
        return None if found is None else np.asarray(found[0], float)

    def third_local(self, profile):
        approach = np.asarray(profile.gripper.approach_axis_local, float)
        closing = np.asarray(profile.gripper.jaw_opening_axis_local, float)
        third = np.cross(approach, closing)
        return third / max(float(np.linalg.norm(third)), 1e-12)

    def closure_facts(self, profile, arm, arm_state):
        grasp = self.grasp_local(profile, arm, arm_state)
        return dict(grasp_local=None if grasp is None else grasp.tolist(),
                    approach_local=[float(v) for v in profile.gripper.approach_axis_local],
                    closing_local=[float(v) for v in profile.gripper.jaw_opening_axis_local],
                    third_local=self.third_local(profile).tolist())


_BODIES = {"arx-x5": X5Body(), "so101": SO101Body()}


def body_of(profile):
    """The ``*Body`` of ``profile.robot_model_id`` (ARX-X5, SO-101), or None for a robot without one."""
    return _BODIES.get(getattr(profile, "robot_model_id", None))


def body_and_profile(runtime):
    """(body, profile) held perception computes with: the run's profile and its body; the ARX-X5 body with the
    packaged ARX-X5 profile for a runtime without a profile or with a robot that has no body (the behaviour before
    bodies existed: ``enabled`` keeps such a robot off)."""
    body = body_of(run_embodiment(runtime))
    body = body if body is not None else _BODIES["arx-x5"]
    return body, body.profile(runtime)


# ------------------------------------------------------------------ held row
def usable(result, view):
    return (isinstance(result, dict) and result.get("status") == "ok" and isinstance(result.get("centre_m"), list)
            and int(result.get("mask_px") or 0) >= (WRIST_MIN_MASK_PX if view == "wrist" else HEAD_MIN_MASK_PX))


def _r(value, digits=3):
    return None if value is None else round(float(value), digits)


def centre_estimate(mode, kfk, grasp, radius=CLAMP_RADIUS_M, axis=None):
    """Held-object centre: 'grasp' = FK grasp centre; 'kfk' = mask-centroid ray at the FK grasp depth; 'clamp' = kfk
    moved onto the sphere of `radius` around the grasp centre when it lies outside; 'axis' = the grasp centre moved
    along the long axis by the along-axis component of the mask-based centre `kfk` (across the axis the jaws centre
    the object; along it the grasp point can be anywhere), limited to AXIS_MAX_SHIFT_M."""
    grasp = np.asarray(grasp, float)
    if mode == "grasp" or kfk is None:
        return grasp
    kfk = np.asarray(kfk, float)
    if mode == "axis":
        if axis is None:
            return grasp
        a = np.asarray(axis, float) / max(float(np.linalg.norm(axis)), 1e-12)
        t = float(np.clip(np.dot(kfk - grasp, a), -AXIS_MAX_SHIFT_M, AXIS_MAX_SHIFT_M))
        return grasp + t * a
    if mode == "clamp":
        d = kfk - grasp
        n = float(np.linalg.norm(d))
        return grasp + d * (radius / n) if n > radius else kfk
    return kfk


UNMEASURED_NOTE = "; centre error not measured on this robot"


def held_object(results, arm, T_link6, *, camera_ids, axis_measured=None, record_elongated=None, modes=None,
                carried=None, fallback=None, grasp_local=None, third_local=None, uncertainty=True):
    """Object dict (object_table service format) + held meta for one arm, from {view: service held result} with view in
    ('wrist', 'head'). The head entry stands for the requested camera. With a carried pre-grasp row (v2) the row comes
    from held_object_carried. Otherwise (v1): group = elongated (grasp-row length/width or GK elongation >= 1.8) or
    compact; its centre estimator comes from `modes` (default CENTRE_MODE). None when no view has a usable mask, unless
    the group's estimator is 'grasp' (then the row has the grasp centre and unknown extents). `fallback` is the reason
    why v2 had no pre-grasp row (said in the note).

    ``T_link6``: 4x4 pose of the TCP link (the profile's ``tcp_link``; ARX-X5 link6) in the table frame (m).
    ``grasp_local`` (3, m) / ``third_local`` (unit 3-vector): the grasp centre and the static-prior long axis
    (perpendicular to the approach axis and to the jaw closing line) in that link's frame; None = the ARX-X5 link6
    values ([0.1501, 0, 0] and link6 z). ``uncertainty``: False when the centre-error statistics (measured on the
    ARX-X5 in simulation) do not apply to this robot: ``uncertainty_p50_mm`` is then None and the note says so."""
    partial = None
    if carried is not None:
        note_closure_length(carried, results)
        partial = partial_row(carried)
        if partial is None:
            return held_object_carried(results, arm, T_link6, camera_ids=camera_ids, carried=carried,
                                       axis_measured=axis_measured, grasp_local=grasp_local, uncertainty=uncertainty)
        fallback = "gk_partial_row"        # W2: v1 extents and group
    modes = dict(CENTRE_MODE, **(modes or {}))
    view = "wrist" if usable(results.get("wrist"), "wrist") else "head" if usable(results.get("head"), "head") else None
    chosen = results[view] if view else {}
    other = (results.get("head" if view == "wrist" else "wrist") or {}) if view else {}
    gk = chosen.get("gk") or (other.get("gk") if usable(other, "head") else None)
    elongation = gk.get("elongation") if gk else None
    elongated = bool(record_elongated) or (elongation is not None and elongation >= ELONGATED)
    group = "elongated" if elongated else "compact"
    mode = modes[group]
    if view is None and mode != "grasp":
        return None
    T_tcp = np.asarray(T_link6, float)
    grasp = grasp_centre(T_link6) if grasp_local is None else T_tcp[:3, 3] + T_tcp[:3, :3] @ np.asarray(grasp_local, float)
    height = (gk["top_z_m"] - gk["low_z_m"]) if gk else None
    axis, axis_source = None, None
    if elongated:
        if axis_measured is not None:
            axis, axis_source = np.asarray(axis_measured, float), "held_end"
        else:
            axis, axis_source = (T_tcp[:3, 2].copy() if third_local is None
                                 else T_tcp[:3, :3] @ np.asarray(third_local, float)), "static_prior"
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        if axis[2] < 0:
            axis = -axis
    if mode == "axis" and axis is None:          # compact objects: no long axis to slide along
        mode = "grasp"
    ref = chosen.get("centre_m")
    if mode == "axis" and AXIS_REF == "gk" and gk:
        ref = gk["centre_m"]
    centre = [float(v) for v in centre_estimate(mode, ref, grasp, axis=axis)]
    tilt = yaw = None
    if axis is not None:
        tilt = math.degrees(math.acos(min(1.0, abs(float(axis[2])))))
        if tilt >= TILT_FOR_YAW_DEG:
            yaw = (math.degrees(math.atan2(axis[1], axis[0])) + 90.0) % 180.0 - 90.0
    if yaw is None and gk:
        yaw = gk.get("yaw_deg")
    head_ok = usable(results.get("head"), "head")
    obj = dict(centre_m=centre, top_z_m=centre[2] + height / 2 if height is not None else None,
               bottom_z_m=centre[2] - height / 2 if height is not None else None,
               length_m=gk.get("length_m") if gk else None, width_m=gk.get("width_m") if gk else None,
               yaw_deg=yaw, axis_tilt_deg=tilt, elongation=ELONGATED if axis is not None else (elongation or 0.0),
               support=SUPPORT % arm, flags=[], quality=QUALITY, label=None,
               contour_px=results["head"].get("contour_px") if head_ok else [],
               label_px=results["head"].get("label_px") if head_ok else None, held_arm=arm)
    camera = camera_ids.get(view) if view else None
    key = mode if mode == "grasp" else "%s_%s" % (mode, view)
    p50, p90 = (UNCERTAINTY_MM.get((group, key), UNCERTAINTY_MM.get(("all", key), (None, None))) if uncertainty
                else (None, None))
    method = {"grasp": "the gripper-kinematics grasp centre",
              "kfk": "the %s mask centroid ray at the gripper-kinematics grasp depth" % camera,
              "clamp": "the %s mask centroid ray at the gripper-kinematics grasp depth, kept within %.0f cm of the grasp "
                       "centre" % (camera, 100 * CLAMP_RADIUS_M),
              "axis": "the gripper-kinematics grasp centre shifted along the long axis to the %s mask centre"
                      % camera}[mode]
    extents = ("extents from VGGT points rescaled to the grasp depth" if gk else "extents unknown")
    note = "in the %s gripper (%s object): centre = %s; %s" % (arm, group, method, extents)
    if partial is not None:
        why = []
        if "short" in partial["reasons"]:
            why.append("length %.3f m < %.1f x %.3f m measured" % (float(partial["length_m"]), PARTIAL_ROW_RATIO,
                                                                    partial["reference_m"]))
        if "outline" in partial["reasons"]:
            why.append("its mask outline does not contain the grasp point")
        note += (" (v1 estimate: pre-grasp row %s of %s%s is partial (%s), extent_source=gk_partial_row)"
                 % (carried["row_id"], carried["observation_id"], _source_text(carried), "; ".join(why)))
    elif fallback:
        note += " (v1 estimate: %s, pregrasp_row_id=none, extent_source=%s)" % (
            fallback.replace("_", " "), ("gk_" + (view if chosen.get("gk") else ("head" if view == "wrist" else "wrist")))
            if gk else "none")
    if p50 is not None:
        note += "; centre error p50 ~%d mm, p90 ~%d mm" % (p50, p90)
    elif not uncertainty:
        note += UNMEASURED_NOTE
    if elongated:
        note += "; along the long axis the object centre can be off the grasp point (grasped near one end)"
    else:
        note += "; orientation prior: up axis as before the grasp (assumed unchanged, not measured)"
    meta = dict(source=SOURCE, arm=arm, group=group, centre=mode, camera_id=camera,
                mask=chosen.get("mask_source"), note=note, uncertainty_p50_mm=p50,
                extents=("gk_" + (view if chosen.get("gk") else ("head" if view == "wrist" else "wrist"))) if gk else None)
    meta["long_axis"] = (dict(direction=[_r(v, 3) for v in axis], source=axis_source) if axis is not None else None)
    if not elongated:
        meta["orientation_prior"] = "up_axis_unchanged_since_grasp"
    if partial is not None:
        meta.update(extent_source="gk_partial_row", pregrasp_row_id=None, fallback=fallback,
                    partial_pregrasp_row=dict(row_id=carried["row_id"], observation_id=carried["observation_id"],
                                              centre_m=[_r(v, 3) for v in carried["centre_m"]],
                                              table_source=_table_source(carried), reasons=partial["reasons"],
                                              length_m=_r(partial["length_m"]), reference_m=_r(partial["reference_m"])))
    elif fallback:
        meta.update(extent_source=meta["extents"], pregrasp_row_id=None, fallback=fallback)
    return obj, _retention_evidence(meta, results)


def _same_3d_held_object(target, held_obj):
    """判断 2D 掩码匹配的候选是否与手中夹持物同属一物；若候选底面贴在台面而夹持物已明显悬空，则保留台面物体并追加夹持物。"""
    if not isinstance(target, dict) or not isinstance(held_obj, dict):
        return False
    tb, hb = target.get("bottom_z_m"), held_obj.get("bottom_z_m")
    if isinstance(tb, (int, float)) and isinstance(hb, (int, float)) and (float(hb) - float(tb)) > 0.045:
        return False
    return True


def merge(objects, held, head_results):
    """Insert held objects into the service object list: replace the object that is the same mask (service
    object_index of the requested-camera result), else append. Returns (objects, {index: meta})."""
    objects = list(objects)
    metas = {}
    for arm, (obj, meta) in held.items():
        index = (head_results.get(arm) or {}).get("object_index")
        if (
            isinstance(index, int)
            and 0 <= index < len(objects)
            and index not in metas
            and _same_3d_held_object(objects[index], obj)
        ):
            if not obj.get("contour_px"):
                obj = dict(obj, contour_px=objects[index].get("contour_px"), label_px=objects[index].get("label_px"))
            objects[index] = obj
        else:
            index = len(objects)
            objects.append(obj)
        if obj.get("label_px") is None:
            obj["label_px"] = [0, 0]
        metas[index] = meta
    return objects, metas


def query_held(endpoint, image_bytes, camera, robot_mask, items, timeout_s, support_z_m=0.765):
    """POST /held on one camera image -> list of results (one per item), or raise ServiceError."""
    import contextlib
    from .object_table import ServiceError
    from .rectify import rectified_cached
    import urllib.error
    import urllib.request
    b64, shape = pack_mask(robot_mask)
    # pinhole service: a distorted camera is sent undistorted and its pixels are mapped back (no-op when D = 0)
    rectified = rectified_cached(camera, image_bytes) if "D" in camera else None
    if rectified is not None:
        image_bytes = rectified.image_bytes
    body = dict(image_b64=base64.b64encode(image_bytes).decode("ascii"), K=np.asarray(camera["K"]).tolist(),
                T_world_camera=np.asarray(camera["T"]).tolist(), support_z_m=float(support_z_m), robot_mask_b64=b64,
                robot_mask_shape=shape, held=items, timeout_s=timeout_s)
    request = urllib.request.Request(endpoint.rstrip("/") + "/held", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s + 5.0) as response:
            answer = json.loads(response.read())
    except urllib.error.HTTPError as error:
        with contextlib.closing(error):
            raw_err = error.read()
        detail = ""
        try:
            err_doc = json.loads(raw_err or b"{}")
            if isinstance(err_doc, dict) and err_doc.get("error"):
                detail = ":" + str(err_doc["error"])[:120]
        except ValueError:
            pass
        raise ServiceError(f"http_{error.code}{detail}") from None
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise ServiceError(type(error).__name__ + ":" + str(getattr(error, "reason", error))[:160]) from None
    if not isinstance(answer, dict) or answer.get("status") != "ok":
        raise ServiceError(str((answer or {}).get("error", "service_error"))[:160])
    held = answer.get("held") or []
    if rectified is not None:
        rectified.map_objects([item for item in held if isinstance(item, dict)])
    return held


class Prepared:
    """Per-request state of the flag-on path (robot mask of the requested camera and the held items)."""

    def _facts(self):
        if "_body_profile" not in self.__dict__:
            self._body_profile = body_and_profile(getattr(self, "runtime", None))
        return self._body_profile

    @property
    def body(self):
        """The ``*Body`` of the run's robot (``body_and_profile``)."""
        return self._facts()[0]

    @property
    def profile(self):
        """The profile the body computes with (ARX-X5: the packaged profile, module docstring)."""
        return self._facts()[1]

    @property
    def wrist_cameras(self):
        """{arm: wrist camera id} of the body's profile."""
        return self.profile.wrist_cameras

    @property
    def keep_segments(self):
        """True when the capsule segments stay in the request beside the pixel mask (``*Body.segments_with_mask``)."""
        return self.body.segments_with_mask

    def holding(self):
        """Arms holding something in this observation's state (ARX-X5: ``holding_arms(state)``, the packaged profile)."""
        if isinstance(self.body, X5Body):
            return holding_arms(self.state)
        return holding_arms(self.state, self.profile)

    def robot_mask(self, camera, camera_id):
        """FK robot pixel mask of one camera (pinhole K: the rectified image of a distorted camera)."""
        posed = posed_links(self.meshes, self.poses, skip_camera_of=arm_of_camera(camera_id, self.wrist_cameras),
                            camera_links=self.body.camera_links)
        return self.body.mask(camera, posed)

    def __init__(self, runtime, tools, full, observation, camera, camera_id):
        started = time.monotonic()
        self.runtime, self.tools, self.full, self.camera_id = runtime, tools, full, camera_id
        self.tcp_link = self.profile.tcp_link
        self.observation_id = full["observation_id"]
        self.poses = arm_link_poses(tools, full, self.observation_id, observation)
        self.meshes = self.body.meshes(runtime)
        self.mask = self.robot_mask(camera, camera_id)
        self.mask_s = time.monotonic() - started
        self.camera = dict(K=camera["K"], T=camera["T"], width=camera.get("width"), height=camera.get("height"))
        self.step = observation.get("sim_step")
        self.track_error = None
        if EXTENT_MODE == "pregrasp":   # v2: follow the gripper state of the saved observations
            try:
                track(runtime)
            except Exception as error:
                self.track_error = type(error).__name__ + ":" + str(error)[:120]
        self.state = camera.get("state") or {}
        self.items, self.grasp_local = [], {}
        arms = self.state.get("arms") or {}
        for arm in self.holding():
            local = self.body.grasp_local(self.profile, arm, arms.get(arm) or {})
            if self.tcp_link not in self.poses.get(arm, {}) or local is None:
                continue
            label = grasp_label(runtime, arm)
            tcp = self.poses[arm][self.tcp_link]
            self.grasp_local[arm] = local
            self.items.append(dict(arm=arm, grasp_centre_m=[float(v) for v in tcp[:3, 3] + tcp[:3, :3] @ local],
                                   **({"phrases": [label]} if label else {})))
        self.wrist_s = 0.0
        degraded_arms = dict(getattr(self.poses, "degraded_arms", None) or {})
        obs_cam_ids = {
            c.get("camera_id") for c in (observation.get("cameras") or []) if isinstance(c, dict)
        }
        for arm_name, wcam_id in self.wrist_cameras.items():
            if wcam_id in obs_cam_ids and arm_name not in self.poses and arm_name not in degraded_arms:
                degraded_arms[arm_name] = "missing_arm_fk"
        degraded_meshes = list(getattr(self.meshes, "degraded_links", None) or [])
        self.degraded_arms = degraded_arms
        self.degraded_meshes = degraded_meshes
        self.mask_incomplete = bool(degraded_arms or degraded_meshes)
        self.diagnostics = dict(robot_mask_px=int(self.mask.sum()), robot_mask_s=round(self.mask_s, 4),
                                mesh_links=len(self.meshes), arms=sorted(self.poses), holding=[i["arm"] for i in self.items])
        if degraded_arms:
            self.diagnostics["degraded_arms"] = degraded_arms
        if degraded_meshes:
            self.diagnostics["degraded_meshes"] = degraded_meshes
        if self.mask_incomplete:
            self.diagnostics["mask_incomplete"] = True
        undecided = {arm: self.body.undecided(self.profile, arm, arms[arm]) for arm in self.profile.arms
                     if isinstance(arms.get(arm), dict)}
        undecided = {arm: reason for arm, reason in undecided.items() if reason}
        if undecided:   # the jaw rule could not decide (no held row for that arm); body.undecided names why
            self.diagnostics["holding_undecided"] = undecided

    def body_fields(self):
        b64, shape = pack_mask(self.mask)
        fields = dict(robot_segments_m=[], robot_mask_b64=b64, robot_mask_shape=shape)
        if self.items:
            fields["held"] = self.items
        return fields

    def wrist_results(self, endpoint, timeout_s):
        """{arm: result} of the holding wrist cameras (not the requested camera)."""
        out = {}
        started = time.monotonic()
        for item in self.items:
            camera_id = self.wrist_cameras.get(item["arm"])
            if camera_id is None or camera_id == self.camera_id:
                continue
            try:
                camera = self.tools.camera(self.full, self.observation_id, camera_id)
                _, image = self.runtime.scene.resolve_artifact(camera["frame"]["rgb_ref"])
                mask = self.robot_mask(camera, camera_id)
                from .object_table import effective_support_z_m
                features = getattr(getattr(self.runtime, "config", None), "features", None)
                support_z = effective_support_z_m(features, camera, embodiment_of(self.runtime))
                out[item["arm"]] = (query_held(endpoint, image, camera, mask, [item], timeout_s, support_z) or [None])[0]
            except Exception as error:   # the wrist view is optional; the requested camera's result remains
                out[item["arm"]] = dict(status="unavailable", reason=str(error)[:120])
        self.wrist_s = time.monotonic() - started
        return out

    def apply(self, objects, answer, endpoint, timeout_s, *, record=True):
        """(objects, {index: meta}) with the held rows merged; objects unchanged when no arm is holding.

        ``record``: True when the model receives this table: it takes the next per-run table number, its rows are
        kept as candidate pre-grasp rows of the free arms (``record_table``) and its held mask may set the carried
        row's ``closure_length_m``. False for a table the model never receives (``object_table._target_view_objects``):
        none of these, so a later closure cannot carry a row, or name a table number, the model has not seen."""
        if EXTENT_MODE == "pregrasp" and record:   # v2: candidate pre-grasp rows for the arms that are not holding
            holding = {item["arm"] for item in self.items} | set(self.holding())
            try:
                memory = tracker(self.runtime)
                memory["table_seq"] = int(memory.get("table_seq") or 0) + 1
                kind = "labelled" if self.full.get("labels") else "pixels" if self.full.get("pixels") else "plain"
                record_table(self.runtime, self.observation_id, self.step, self.camera_id, self.camera, objects,
                             self.poses, [arm for arm in sorted(self.poses) if arm not in holding],
                             source=dict(seq=memory["table_seq"], request=kind), tcp_link=self.tcp_link)
            except Exception as error:
                self.track_error = self.track_error or type(error).__name__ + ":" + str(error)[:120]
        if self.track_error:
            self.diagnostics["held_track_error"] = self.track_error
        if not self.items:
            return objects, {}
        head = {r.get("arm"): r for r in answer.get("held") or [] if isinstance(r, dict)}
        legacy_v5 = answer.get("version") == "object-table-v5" and not answer.get("robot_mask")
        if legacy_v5:
            wrist = {
                item["arm"]: dict(status="unavailable", reason="service_version_object-table-v5")
                for item in self.items
                if self.wrist_cameras.get(item["arm"]) != self.camera_id
            }
            self.diagnostics["service_fallback"] = "object-table-v5_no_hp1"
        else:
            wrist = self.wrist_results(endpoint, timeout_s)
        held, views, carried_rows = {}, {}, {}
        for item in self.items:
            arm = item["arm"]
            own_wrist = self.camera_id == self.wrist_cameras.get(arm)
            results = dict(head=None if own_wrist else head.get(arm), wrist=head.get(arm) if own_wrist else wrist.get(arm))
            carried, why = carried_for(self.runtime, arm, self.step) if EXTENT_MODE == "pregrasp" else (None, None)
            if carried is not None and not record:   # held_object notes the closure length once per streak: a
                carried = dict(carried)              # table the model never receives leaves the carried state as is
            x5 = isinstance(self.body, X5Body)   # ARX-X5: the link6 defaults of held_object, unchanged
            made = held_object(results, arm, self.poses[arm][self.tcp_link],
                               camera_ids=dict(head=self.camera_id, wrist=self.wrist_cameras.get(arm)),
                               axis_measured=None,
                               record_elongated=record_elongated(self.runtime, arm), carried=carried, fallback=why,
                               grasp_local=None if x5 else self.grasp_local[arm],
                               third_local=None if x5 else self.body.third_local(self.profile),
                               uncertainty=self.body.measured_uncertainty)
            views[arm] = {k: (v or {}).get("status") for k, v in results.items()}
            if EXTENT_MODE == "pregrasp":
                carried_rows[arm] = (dict(row=carried["row_id"], observation_id=carried["observation_id"],
                                          group=carried["group"], tables_back=carried["tables_back"],
                                          contains=carried["contains"], length=_r(carried.get("length")),
                                          mask_length=_r(carried.get("mask_length_m")),
                                          closure_length=_r(carried.get("closure_length_m")),
                                          table=carried.get("table_source"))
                                     if carried else dict(fallback=why))
                if made is not None and carried:
                    carried_rows[arm].update(extent_source=made[1].get("extent_source"),
                                             group_used=made[1].get("group"),
                                             axis_conflict_deg=made[1].get("axis_conflict_deg"))
            if made is not None:
                held[arm] = made
        self.diagnostics.update(held_views=views, wrist_s=round(self.wrist_s, 3), held_rows=sorted(held))
        if carried_rows:
            self.diagnostics["held_pregrasp"] = carried_rows
        return merge(objects, held, head)
