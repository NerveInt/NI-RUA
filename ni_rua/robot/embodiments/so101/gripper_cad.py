"""SO-101 gripper geometry facts derived from the packaged URDF and STL meshes.

Pure numpy; no serial bus, camera or simulator. Every number is expressed in the
TCP frame ``gripper_frame_link`` (metres, radians):

* ``+Z`` is the approach axis (outward along the fingers); points with ``z > 0``
  lie beyond the TCP.
* The TCP origin lies ON the fixed jaw's inner fingertip face (``x = 0``), not
  between the jaws: the fixed jaw is rigid ``gripper_link`` geometry on the
  ``+X`` side, and the whole jaw gap is on ``-X``, where the single moving jaw
  (``moving_jaw_so101_v1_link``) swings away along ``-X`` (and, past about
  0.6 rad, back behind the TCP plane) as the ``gripper`` joint angle grows. The
  fingertip midpoint therefore moves with the jaw angle.
* ``Y`` is the jaw hinge direction (finger width); the wrist camera sits on ``+Y``.

The facts are CAD/URDF-derived, not measured on a given printed unit. Opening
widths are geometric: on hardware the moving jaw stops on contact with a held
object, so the measured jaw angle maps to the object width through
``opening_width_m``. ``top_down_roll_facts`` relates a straight-down TCP yaw to
``wrist_roll`` by FK of the same URDF.
"""
from __future__ import annotations

import copy
import functools
import json
import math
import struct
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from . import profile as kin

_PACKAGE = Path(__file__).resolve().parents[2] / "model" / "robot_information_data" / "so101"
_ZERO_ARM = (0.0,) * len(kin.ARM_JOINTS)
# Numerical resolution of the derivation (not task thresholds): depth bins for
# face profiles, the tip band that defines a fingertip contact point, and the
# tolerance separating "between the jaws" from "on the fixed inner face".
_BIN_M = 0.001
_TIP_BAND_M = 0.002
_FACE_EPS_M = 1e-4
_SWEEP_SAMPLES = 97
_TABLE_SAMPLES = 12
_BISECT_STEPS = 50
# goal_normalized rows of the model-facing opening table (deployment-independent sampling of the command range).
GOAL_TABLE = tuple(round(0.1*i, 1) for i in range(11))


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp, cp*sr, cp*cr]])


def _stl_triangles(data: bytes) -> np.ndarray:
    """(N, 3, 3) triangle vertices from a binary or ASCII STL."""
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        if count and len(data) == 84 + 50*count:
            record = np.dtype([("normal", "<3f4"), ("vertices", "<9f4"), ("attribute", "<u2")])
            return np.frombuffer(data, dtype=record, count=count, offset=84)["vertices"].reshape(-1, 3, 3).astype(np.float64)
    vertices = [line.split()[1:4] for line in data.decode("ascii", "strict").splitlines()
                if line.strip().startswith("vertex")]
    if not vertices or len(vertices) % 3:
        raise ValueError("so101_gripper_mesh_unreadable_stl")
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)


def _link_meshes(root, archive, link_name):
    """Collision meshes (visual if none) of one link in its link frame, (N, 3, 3)."""
    for node in root.findall("link"):
        if node.get("name") != link_name:
            continue
        elements = node.findall("collision") or node.findall("visual")
        parts = []
        for element in elements:
            mesh = element.find("geometry/mesh")
            if mesh is None:
                continue
            triangles = _stl_triangles(archive.read(mesh.attrib["filename"]))
            triangles = triangles * np.array([float(v) for v in mesh.get("scale", "1 1 1").split()])
            origin = element.find("origin")
            if origin is not None:
                rotation = _rpy(*(float(v) for v in origin.get("rpy", "0 0 0").split()))
                triangles = triangles @ rotation.T + np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
            parts.append(triangles)
        return np.concatenate(parts) if parts else np.zeros((0, 3, 3))
    raise ValueError(f"so101_gripper_missing_link:{link_name}")


def _transform(T, triangles):
    return triangles @ T[:3, :3].T + T[:3, 3]


def _section_x(triangles, z):
    """X values where the mesh surface crosses plane ``z`` (exact triangle edges)."""
    values = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        za, zb = triangles[:, a, 2], triangles[:, b, 2]
        hit = (za - z)*(zb - z) < 0
        s = (z - za[hit])/(zb[hit] - za[hit])
        values.append(triangles[hit, a, 0] + (triangles[hit, b, 0] - triangles[hit, a, 0])*s)
    return np.concatenate(values)


def _bbox(points):
    points = np.asarray(points).reshape(-1, 3)
    return {"min_m": [round(float(v), 4) for v in points.min(axis=0)],
            "max_m": [round(float(v), 4) for v in points.max(axis=0)]}


def _r(values, digits=4):
    return [round(float(v), digits) + 0.0 for v in values]  # + 0.0 drops negative zero


class _Geometry:
    """Cached derivation state; use the module functions, not this class."""

    def __init__(self, archive_path=None):
        if archive_path is None:
            native = json.loads((_PACKAGE / "physical.json").read_text(encoding="utf-8"))["geometry"]["native_model"]
            archive_path, member = _PACKAGE / f"{native['resource_id']}.zip", native["assembly_member"]
        else:
            member = None
        with zipfile.ZipFile(archive_path) as archive:
            if member is None:
                member = next(name for name in archive.namelist() if name.endswith(".urdf"))
            urdf = archive.read(member)
            if archive_path == _PACKAGE / "geometry_model.zip":
                self.model = kin.load_model()
            else:
                import tempfile
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / Path(member).name
                    path.write_bytes(urdf)
                    self.model = kin.load_model(path)
            root = ET.fromstring(urdf)
            joints = {joint["name"]: joint for joint in self.model["joints"]}
            self.moving_link = joints[kin.GRIPPER_JOINT]["child"]
            self.jaw_axis_local = np.asarray(joints[kin.GRIPPER_JOINT]["axis"], dtype=np.float64)
            self.jaw_limits = kin.joint_limits_rad(self.model)[kin.GRIPPER_JOINT]
            parent = joints[kin.GRIPPER_JOINT]["parent"]
            # The revolute joint carrying the jaw link (wrist_roll): its axis in the child link frame.
            self.roll_joint = next(joint for joint in self.model["joints"]
                                   if joint["child"] == parent and joint["type"] != "fixed")
            # Rigid attachments of the jaw-carrying link (camera mount, camera).
            attached, frontier = [], [parent]
            while frontier:
                link = frontier.pop()
                for joint in self.model["joints"]:
                    if joint["parent"] == link and joint["type"] == "fixed" and joint["child"] != kin.TCP_LINK:
                        attached.append(joint["child"]); frontier.append(joint["child"])
            poses = self._poses(0.0)
            to_tcp = np.linalg.inv(poses[kin.TCP_LINK])
            roll_frame = to_tcp @ poses[parent]
            roll_axis = roll_frame[:3, :3] @ np.asarray(self.roll_joint["axis"], dtype=np.float64)
            self.roll_axis = roll_axis/np.linalg.norm(roll_axis)
            # Where the roll axis crosses the TCP z = 0 plane (it is parallel to the approach axis on SO-101).
            origin = roll_frame[:3, 3]
            self.roll_point = (origin - self.roll_axis*origin[2]/self.roll_axis[2] if abs(self.roll_axis[2]) > 1e-9
                               else origin)
            self.body = _transform(to_tcp @ poses[parent], _link_meshes(root, archive, parent))
            self.moving_local = _link_meshes(root, archive, self.moving_link)
            self.attachments = {link: _transform(to_tcp @ poses[link], _link_meshes(root, archive, link))
                                for link in attached}
            self.attachments = {k: v for k, v in self.attachments.items() if len(v)}
        self._derive()

    def _poses(self, jaw_rad):
        return kin.fk_all(self.model, _ZERO_ARM, gripper_rad=float(jaw_rad))

    def T_tcp_moving(self, jaw_rad):
        poses = self._poses(jaw_rad)
        return np.linalg.inv(poses[kin.TCP_LINK]) @ poses[self.moving_link]

    def moving(self, jaw_rad):
        return _transform(self.T_tcp_moving(jaw_rad), self.moving_local)

    def _derive(self):
        body_v = self.body.reshape(-1, 3)
        zero = self.T_tcp_moving(0.0)
        moving0 = _transform(zero, self.moving_local)
        moving_v = moving0.reshape(-1, 3)
        # Fixed fingertip: inner (minimum X) point within the tip band of the rigid body.
        fixed_top = body_v[:, 2].max()
        band = body_v[body_v[:, 2] >= fixed_top - _TIP_BAND_M]
        x_in = band[:, 0].min()
        near = band[band[:, 0] <= x_in + 0.5*_BIN_M]
        self.fixed_tip = np.array([x_in, near[:, 1].mean(), near[:, 2].mean()])
        # Moving fingertip (at jaw 0): inner (maximum X) point in its tip band, kept in link frame.
        moving_top = moving_v[:, 2].max()
        band = moving_v[moving_v[:, 2] >= moving_top - _TIP_BAND_M]
        x_in = band[:, 0].max()
        near = band[band[:, 0] >= x_in - 0.5*_BIN_M]
        tip_tcp = np.array([x_in, near[:, 1].mean(), near[:, 2].mean()])
        self.moving_tip_local = (np.linalg.inv(zero) @ np.append(tip_tcp, 1.0))[:3]
        # Pocket floor: highest rigid-body point lying between the two inner tip faces.
        between = body_v[(body_v[:, 0] < self.fixed_tip[0] - _FACE_EPS_M) & (body_v[:, 0] > tip_tcp[0])
                         & (np.abs(body_v[:, 1] - self.fixed_tip[1]) <= np.abs(moving_v[:, 1] - tip_tcp[1]).max())]
        self.palm_z = float(between[:, 2].max())
        # Inner-face profiles between the pocket floor and the tips (exact sections).
        depths = np.arange(self.fixed_tip[2] - _BIN_M, self.palm_z, -_BIN_M)
        fixed_profile = [(z, _section_x(self.body, z)) for z in depths]
        self.fixed_profile = np.array([(x.min(), z) for z, x in fixed_profile if len(x)])
        moving_profile = [(z, _section_x(moving0, z)) for z in depths]
        moving_profile = np.array([(x.max(), z) for z, x in moving_profile if len(x)])
        root_tcp = np.array([moving_profile[-1, 0], tip_tcp[1], moving_profile[-1, 1]])
        self.moving_root_local = (np.linalg.inv(zero) @ np.append(root_tcp, 1.0))[:3]
        # Sweep of the moving jaw over the declared limits.
        self.sweep = np.linspace(*self.jaw_limits, _SWEEP_SAMPLES)
        swept = [self.moving(q).reshape(-1, 3) for q in self.sweep]
        self.moving_min = np.min([s.min(axis=0) for s in swept], axis=0)
        self.moving_max = np.max([s.max(axis=0) for s in swept], axis=0)
        self.moving_top = np.array([s[:, 2].max() for s in swept])
        self.openings = np.array([self.opening(q) for q in self.sweep])

    def moving_point(self, local, jaw_rad):
        return (self.T_tcp_moving(jaw_rad) @ np.append(local, 1.0))[:3]

    def opening(self, jaw_rad):
        return float(np.linalg.norm(self.moving_point(self.moving_tip_local, jaw_rad) - self.fixed_tip))

    def moving_reach(self, jaw_rad):
        """Largest TCP-frame z (m) of the moving jaw mesh at ``jaw_rad``."""
        return float(self.moving(jaw_rad)[..., 2].max())

    def bisect(self, value, lo, hi, target):
        """Jaw angle (rad) in [lo, hi] where ``value(jaw) - target`` changes sign (bisection; assumes one crossing)."""
        below = value(lo) < target
        for _ in range(_BISECT_STEPS):
            mid = 0.5*(lo + hi)
            lo, hi = (mid, hi) if (value(mid) < target) == below else (lo, mid)
        return 0.5*(lo + hi)


@functools.lru_cache(maxsize=4)
def _geometry(archive_path=None) -> _Geometry:
    return _Geometry(archive_path)


@functools.lru_cache(maxsize=4)
def _reach_facts(archive_path=None) -> dict:
    """Moving-jaw reach beyond the TCP over the declared jaw range (rad, m): its peak, where it exceeds the fixed jaw's
    and where the whole moving jaw is behind the TCP plane (sweep, then bisection on each monotonic branch)."""
    g = _geometry(archive_path)
    lo, hi = g.jaw_limits
    index = int(np.argmax(g.moving_top))
    left, right = g.sweep[max(index - 1, 0)], g.sweep[min(index + 1, len(g.sweep) - 1)]
    fine = np.linspace(left, right, 81)
    peak = float(fine[int(np.argmax([g.moving_reach(q) for q in fine]))])
    fixed = float(g.body[..., 2].max())
    facts = {"peak_jaw_rad": peak, "peak_reach_m": g.moving_reach(peak), "fixed_reach_m": fixed,
             "beyond_fixed_jaw_rad": None, "behind_tcp_plane_from_jaw_rad": None}
    if g.moving_reach(peak) > fixed:
        start = lo if g.moving_reach(lo) >= fixed else g.bisect(g.moving_reach, lo, peak, fixed)
        end = hi if g.moving_reach(hi) >= fixed else g.bisect(g.moving_reach, peak, hi, fixed)
        facts["beyond_fixed_jaw_rad"] = [start, end]
    if g.moving_reach(hi) <= 0.0 < g.moving_reach(peak):
        facts["behind_tcp_plane_from_jaw_rad"] = g.bisect(g.moving_reach, peak, hi, 0.0)
    return facts


def _jaw_rad(goal_normalized, jaw_closed_rad, jaw_open_rad):
    return float(jaw_closed_rad) + float(goal_normalized)*(float(jaw_open_rad) - float(jaw_closed_rad))


def opening_width_m(jaw_rad, *, archive_path=None) -> float:
    """Distance (m) between the moving and fixed fingertip inner contact points.

    ``jaw_rad``: measured or commanded URDF ``gripper`` joint angle in rad
    (declared limits -0.174533..1.74533; values outside are extrapolated
    geometrically). Increases monotonically with ``jaw_rad``. Near the closed
    limit it can be slightly below the true printed-part clearance; a negative
    model penetration cannot occur because it is a point-to-point distance.
    Example: ``opening_width_m(0.0)`` -> about 0.016.
    """
    value = float(jaw_rad)
    if not math.isfinite(value):
        raise ValueError("so101_jaw_rad_must_be_finite")
    return _geometry(archive_path).opening(value)


def fingertip_reference(jaw_rad, *, archive_path=None) -> dict:
    """Return CAD inner-tip points and their midpoint in TCP coordinates (m).

    ``jaw_rad`` is the measured URDF gripper angle (rad, normally
    -0.174533..1.74533); required, with no commanded-angle fallback.
    ``archive_path`` optionally selects an alternate URDF/mesh zip; None uses
    the packaged SO-101 geometry. Example: ``fingertip_reference(0.0)``.
    The midpoint moves with the single swinging jaw; it is a geometric
    reference, not an object contact estimate. ``normal_tcp`` is the unit
    vector from moving to fixed tip, not a claim that the faces are parallel.
    """
    angle = float(jaw_rad)
    if not math.isfinite(angle):
        raise ValueError("so101_jaw_rad_must_be_finite")
    g = _geometry(archive_path)
    fixed, moving = g.fixed_tip, g.moving_point(g.moving_tip_local, angle)
    delta = fixed - moving
    width = float(np.linalg.norm(delta))
    return dict(fixed_tip_m=fixed.tolist(), moving_tip_m=moving.tolist(),
                midpoint_m=((fixed + moving) / 2).tolist(),
                normal_tcp=(delta / width).tolist() if width > 0 else None,
                opening_m=width)


def opening_width_for_goal_m(goal_normalized, jaw_closed_rad, jaw_open_rad, *, archive_path=None) -> float:
    """Fingertip opening (m) for ``goal_normalized`` (0 closed, 1 open).

    ``jaw_closed_rad`` / ``jaw_open_rad``: the deployment's URDF jaw angles (rad)
    for goal 0 and 1; the jaw target is linear between them.
    """
    return opening_width_m(_jaw_rad(goal_normalized, jaw_closed_rad, jaw_open_rad), archive_path=archive_path)


def jaw_rad_for_opening(width_m, *, archive_path=None):
    """Smallest URDF jaw angle (rad) whose fingertip opening reaches ``width_m`` (m).

    Returns ``None`` when ``width_m`` is outside the opening range over the
    declared jaw limits.
    """
    geometry = _geometry(archive_path)
    width = float(width_m)
    if not geometry.openings[0] <= width <= geometry.openings[-1]:
        return None
    lo, hi = geometry.jaw_limits
    for _ in range(60):
        mid = 0.5*(lo + hi)
        lo, hi = (mid, hi) if geometry.opening(mid) < width else (lo, mid)
    return 0.5*(lo + hi)


def goal_normalized_for_opening(width_m, jaw_closed_rad, jaw_open_rad, *, archive_path=None):
    """``goal_normalized`` giving a fingertip opening of ``width_m`` (m), or ``None``.

    ``None`` when the width is not reachable between the deployment's
    ``jaw_closed_rad`` and ``jaw_open_rad``.
    """
    jaw = jaw_rad_for_opening(width_m, archive_path=archive_path)
    span = float(jaw_open_rad) - float(jaw_closed_rad)
    if jaw is None or span == 0:
        return None
    goal = (jaw - float(jaw_closed_rad))/span
    return goal if 0.0 <= goal <= 1.0 else None


def centered_grasp_tcp_offset_m(object_width_m, *, archive_path=None) -> float:
    """TCP shift (m) along TCP ``+X`` (toward the fixed jaw) for a centred grasp.

    Only the ``-X`` jaw moves, so the object's ``+X`` face rests on the fixed
    inner face (at ``x_fixed``). To centre an object of width ``object_width_m``
    (m, measured along TCP X) the TCP goes to
    ``object_centre + (object_width_m/2 - x_fixed) * tcp_x_axis``.
    Example: a 0.03 m cube -> about +0.015 m toward the fixed jaw.
    """
    width = float(object_width_m)
    if not math.isfinite(width) or width < 0:
        raise ValueError("so101_object_width_must_be_finite_nonnegative_m")
    return 0.5*width - float(_geometry(archive_path).fixed_tip[0])


def moving_jaw_reach_m(jaw_rad, *, archive_path=None) -> float:
    """How far (m) the moving jaw reaches beyond the TCP along the approach axis at URDF jaw angle ``jaw_rad`` (rad).

    The largest TCP-frame ``z`` of any moving-jaw mesh point: positive = beyond the TCP (with the approach pointing
    down: below it, toward a support), negative = the whole moving jaw is behind the TCP plane. It first grows as the
    jaw opens from the closed limit, peaks, then falls as the jaw swings back. Example: ``moving_jaw_reach_m(0.0)`` ->
    about 0.0073; ``moving_jaw_reach_m(1.0)`` -> about -0.020."""
    value = float(jaw_rad)
    if not math.isfinite(value):
        raise ValueError("so101_jaw_rad_must_be_finite")
    return _geometry(archive_path).moving_reach(value)


def _goal(jaw_rad, jaw_closed_rad, jaw_open_rad):
    """goal_normalized of a URDF jaw angle on the deployment's linear map (unclamped), or None without a span."""
    span = float(jaw_open_rad) - float(jaw_closed_rad)
    return None if jaw_rad is None or span == 0 else (float(jaw_rad) - float(jaw_closed_rad))/span


def jaw_reach_facts(jaw_closed_rad, jaw_open_rad, *, archive_path=None) -> dict:
    """Where the fingers reach along the approach axis over the command range, in ``goal_normalized`` units (CAD).

    ``jaw_closed_rad`` / ``jaw_open_rad``: the deployment's URDF jaw angles (rad) for goal 0 and 1. Returns
    ``fixed_reach_m`` (m beyond the TCP; the fixed jaw, the same at every opening), ``moving_reach_at_goal0_m`` and
    ``moving_reach_at_goal1_m`` (m), ``peak_reach_m`` / ``peak_goal`` / ``peak_jaw_rad`` (the moving jaw's furthest
    reach over the declared jaw limits and where), ``beyond_fixed_jaw_goals`` ([low, high] goals over which the
    moving jaw reaches further than the fixed jaw, or None) and ``behind_tcp_plane_above_goal`` /
    ``behind_tcp_plane_opening_m`` (above that goal no moving-jaw point is beyond the TCP plane: only the fixed jaw
    is at TCP depth; None if never). Goals outside [0, 1] are reported as computed (unclamped).
    Example: ``jaw_reach_facts(-0.174533, 1.74533)["behind_tcp_plane_above_goal"]`` -> about 0.40."""
    g, facts = _geometry(archive_path), _reach_facts(archive_path)
    goal = lambda jaw: _goal(jaw, jaw_closed_rad, jaw_open_rad)  # noqa: E731
    behind = facts["behind_tcp_plane_from_jaw_rad"]
    band = facts["beyond_fixed_jaw_rad"]
    return {"fixed_reach_m": facts["fixed_reach_m"],
            "moving_reach_at_goal0_m": g.moving_reach(float(jaw_closed_rad)),
            "moving_reach_at_goal1_m": g.moving_reach(float(jaw_open_rad)),
            "peak_reach_m": facts["peak_reach_m"], "peak_jaw_rad": facts["peak_jaw_rad"],
            "peak_goal": goal(facts["peak_jaw_rad"]),
            "beyond_fixed_jaw_goals": None if band is None else [goal(band[0]), goal(band[1])],
            "behind_tcp_plane_above_goal": goal(behind),
            "behind_tcp_plane_opening_m": None if behind is None else g.opening(behind)}


def _wrap(angle):
    """Angle (rad) wrapped to (-pi, pi]."""
    value = math.remainder(float(angle), 2*math.pi)
    return math.pi if value == -math.pi else value


def top_down_roll_facts(wrist_roll_limits_rad, *, archive_path=None) -> dict:
    """Straight-down TCP yaw versus ``wrist_roll`` for this arm, derived from the URDF by forward kinematics.

    With the approach axis (TCP +Z) pointing at world -z, the yaw ``a`` of TCP +X (rad about so101_base +z from +x)
    is ``a = phi + yaw_at_zero_roll_rad + roll_sign * wrist_roll`` (wrapped), where ``phi`` is the azimuth (rad) of
    the wrist_roll axis seen from the shoulder_pan axis. Pan rotates the whole arm about that vertical axis, so the
    relation holds at every position where a straight-down pose is reachable.

    ``wrist_roll_limits_rad``: ``(lower, upper)`` rad the planner may use (the URDF limits shrunk by the deployment's
    ``motion.joint_limit_margin_rad``, as the worker does); ``upper - lower`` must be in (0, 2 pi).
    Returns ``pan_axis_xy_m`` ([x, y] m in so101_base), ``yaw_at_zero_roll_rad``, ``roll_sign`` (+1 or -1),
    ``roll_axis_offset_m`` (horizontal distance of the TCP from the wrist_roll axis, m),
    ``reachable_yaw_minus_azimuth_rad`` ([low, high], unwrapped: ``a - phi`` reachable straight down),
    ``unreachable_centre_rad`` (wrapped centre of the gap ``a - phi`` cannot reach) and
    ``unreachable_half_width_rad`` (half its width). Raises ``ValueError`` when the shoulder_pan or the wrist_roll axis
    is not vertical in a straight-down pose or no straight-down pose exists. Example:
    ``top_down_roll_facts((-2.69385, 2.79121))["unreachable_half_width_rad"]`` -> about 0.399 (22.9 deg)."""
    lower, upper = (float(value) for value in wrist_roll_limits_rad)
    if not (math.isfinite(lower) and math.isfinite(upper) and 0 < upper - lower < 2*math.pi):
        raise ValueError("so101_wrist_roll_limits_must_span_less_than_a_turn")
    return copy.deepcopy(_top_down_roll_facts(lower, upper, archive_path))


@functools.lru_cache(maxsize=8)
def _top_down_roll_facts(lower, upper, archive_path):
    g = _geometry(archive_path)
    model, names = g.model, list(kin.ARM_JOINTS)
    flex, roll = names.index("wrist_flex"), names.index(g.roll_joint["name"])
    flex_limits = kin.joint_limits_rad(model)["wrist_flex"]

    def pose(flex_rad, roll_rad=0.0):
        q = np.zeros(len(names)); q[flex], q[roll] = flex_rad, roll_rad
        poses, axes = kin._chain(model, q, 0.0)
        return poses[kin.TCP_LINK], axes

    # Straight down at shoulder_pan = shoulder_lift = elbow_flex = 0: the wrist_flex angle with the lowest approach z.
    grid = np.linspace(*flex_limits, 721)
    best = int(np.argmin([pose(q)[0][2, 2] for q in grid]))
    lo, hi = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
    for _ in range(_BISECT_STEPS):  # golden-section refinement of the unimodal approach z
        a, b = lo + 0.382*(hi - lo), lo + 0.618*(hi - lo)
        lo, hi = (lo, b) if pose(a)[0][2, 2] < pose(b)[0][2, 2] else (a, hi)
    T, axes = pose(0.5*(lo + hi))
    pan_origin, pan_axis = axes["shoulder_pan"]
    roll_origin, roll_axis = axes[g.roll_joint["name"]]
    if T[2, 2] > -1 + 1e-6 or abs(pan_axis[2]) < 1 - 1e-6 or abs(roll_axis[2]) < 1 - 1e-6:
        raise ValueError("so101_no_vertical_straight_down_pose")
    azimuth = math.atan2(roll_origin[1] - pan_origin[1], roll_origin[0] - pan_origin[0])
    yaw0 = math.atan2(T[1, 0], T[0, 0])
    turned = pose(0.5*(lo + hi), 0.1)[0]
    sign = 1 if _wrap(math.atan2(turned[1, 0], turned[0, 0]) - yaw0) > 0 else -1
    offset = _wrap(yaw0 - azimuth)
    reachable = sorted((offset + sign*lower, offset + sign*upper))
    gap = 2*math.pi - (upper - lower)
    return {"pan_axis_xy_m": [float(pan_origin[0]), float(pan_origin[1])], "yaw_at_zero_roll_rad": offset,
            "roll_sign": sign,
            "roll_axis_offset_m": float(math.hypot(T[0, 3] - roll_origin[0], T[1, 3] - roll_origin[1])),
            "reachable_yaw_minus_azimuth_rad": reachable,
            "unreachable_centre_rad": _wrap(reachable[1] + 0.5*gap), "unreachable_half_width_rad": 0.5*gap}


def derive_gripper_geometry(*, archive_path=None) -> dict:
    """Full JSON-able gripper facts in the TCP frame (m, rad); deployment-independent.

    ``fixed_jaw.inner_face_profile_m`` lists ``{"z_m", "x_m"}`` points of the fixed inner face (depth z along the
    approach axis, x across the jaws); ``wrist_roll_axis_tcp`` is the wrist_roll axis in the TCP frame (unit
    ``direction``, ``point_at_tcp_plane_m`` where it crosses z = 0, ``tcp_distance_m`` from the TCP origin)."""
    g = _geometry(archive_path)
    lo, hi = g.jaw_limits
    axis = g.T_tcp_moving(0.0)[:3, :3] @ (g.jaw_axis_local/np.linalg.norm(g.jaw_axis_local))
    eps = 1e-3
    direction = g.moving_point(g.moving_tip_local, eps) - g.moving_point(g.moving_tip_local, -eps)
    direction /= np.linalg.norm(direction)
    attachments = np.concatenate(list(g.attachments.values())) if g.attachments else np.zeros((0, 3, 3))
    everything = [g.body.reshape(-1, 3), g.moving_min[None], g.moving_max[None], attachments.reshape(-1, 3)]
    protrusion = max(float(g.body[..., 2].max()), float(g.moving_top.max()))
    table_q = np.linspace(lo, hi, _TABLE_SAMPLES)
    return {
        "frame": kin.TCP_LINK,
        "units": "m, rad",
        "tcp_definition": ("gripper_frame_link origin lies on the fixed jaw's inner fingertip face (x = 0), not between "
                           "the jaws; the whole jaw gap is on -X and the fingertip midpoint moves with the jaw angle."),
        "approach_axis_tcp": [0.0, 0.0, 1.0],
        "wrist_roll_axis_tcp": {"direction": _r(g.roll_axis, 3), "point_at_tcp_plane_m": _r(g.roll_point),
                                "tcp_distance_m": round(float(np.hypot(*g.roll_point[:2])), 4)},
        "fixed_jaw": {
            "link": kin.WRIST_CAMERA_PARENT_LINK, "side": "+X",
            "fingertip_inner_point_m": _r(g.fixed_tip),
            "inner_face_profile_m": [{"z_m": round(float(z), 4) + 0.0, "x_m": round(float(x), 4) + 0.0}
                                     for x, z in g.fixed_profile[::10]],
            "finger_length_m": round(float(g.fixed_tip[2] - g.palm_z), 4),
        },
        "moving_jaw": {
            "link": g.moving_link, "side": "-X", "joint": kin.GRIPPER_JOINT,
            "hinge_axis_tcp": _r(axis, 3),
            "opening_direction_tcp": _r(direction, 3),
            "fingertip_inner_point_at_closed_limit_m": _r(g.moving_point(g.moving_tip_local, lo)),
            "fingertip_inner_point_at_zero_m": _r(g.moving_point(g.moving_tip_local, 0.0)),
            "inner_face_root_point_at_zero_m": _r(g.moving_point(g.moving_root_local, 0.0)),
        },
        "pocket_floor_z_m": round(g.palm_z, 4),
        "protrusion_beyond_tcp_m": {
            "max": round(protrusion, 4),
            "fixed_jaw": round(float(g.body[..., 2].max()), 4),
            "moving_jaw_at_closed_limit": round(float(g.moving_top[0]), 4),
            "moving_jaw_at_zero": round(float(g.moving(0.0)[..., 2].max()), 4),
            "moving_jaw_at_open_limit": round(float(g.moving_top[-1]), 4),
        },
        "bbox_tcp_m": {
            "gripper_body": _bbox(g.body),
            "moving_jaw_swept": {"min_m": _r(g.moving_min), "max_m": _r(g.moving_max)},
            "wrist_camera": _bbox(attachments) if len(attachments) else None,
            "overall": _bbox(np.concatenate([p.reshape(-1, 3) for p in everything if len(p)])),
        },
        "jaw_limits_rad": [lo, hi],
        "opening_table": [{"jaw_rad": round(float(q), 4), "opening_m": round(g.opening(q), 4),
                           "moving_tip_m": _r(g.moving_point(g.moving_tip_local, q))} for q in table_q],
        "opening_definition": "Distance between the moving and fixed fingertip inner contact points; monotonic in jaw_rad.",
        "centered_grasp": ("TCP = object_centre + (w/2 - x_face) * (TCP +X in the world); w: object width along TCP X "
                           "(m); x_face: fixed inner-face x (m) at the depth where the object touches it "
                           "(fixed_jaw.inner_face_profile_m, 0 at the tip). Only the -X jaw moves, so closing pushes "
                           "the object along +X onto the fixed face."),
        "source": "packaged SO-101 URDF and STL meshes (CAD, not measured on this unit)",
    }


def gripper_geometry_payload(jaw_closed_rad, jaw_open_rad, *, archive_path=None) -> dict:
    """Compact model-facing ``gripper_geometry`` dict for one deployment (m, rad).

    ``jaw_closed_rad`` / ``jaw_open_rad``: the deployment's URDF jaw angles for
    ``goal_normalized`` 0 and 1. ``goal_to_opening`` rows (``GOAL_TABLE``): ``goal_normalized``, ``jaw_rad``,
    ``opening_m`` (tip-to-tip distance), ``moving_tip_x_m`` / ``moving_tip_z_m`` (moving inner fingertip in the TCP
    frame). ``moving_jaw_reach`` is ``jaw_reach_facts`` (m, goals).
    """
    full = derive_gripper_geometry(archive_path=archive_path)
    g = _geometry(archive_path)
    payload = {key: copy.deepcopy(full[key]) for key in (
        "frame", "units", "tcp_definition", "approach_axis_tcp", "wrist_roll_axis_tcp", "fixed_jaw", "moving_jaw",
        "pocket_floor_z_m", "protrusion_beyond_tcp_m", "bbox_tcp_m", "opening_definition", "centered_grasp")}
    payload["fixed_jaw"].pop("link"); payload["moving_jaw"].pop("link")
    rows = []
    for goal in GOAL_TABLE:
        jaw = _jaw_rad(goal, jaw_closed_rad, jaw_open_rad)
        tip = g.moving_point(g.moving_tip_local, jaw)
        rows.append({"goal_normalized": goal, "jaw_rad": round(jaw, 4), "opening_m": round(g.opening(jaw), 4),
                     "moving_tip_x_m": round(float(tip[0]), 4) + 0.0, "moving_tip_z_m": round(float(tip[2]), 4) + 0.0})
    payload["goal_to_opening"] = rows
    reach = jaw_reach_facts(jaw_closed_rad, jaw_open_rad, archive_path=archive_path)
    payload["moving_jaw_reach"] = {key: (None if value is None else [round(v, 4) for v in value]
                                         if isinstance(value, list) else round(value, 4))
                                   for key, value in reach.items()}
    return payload


def gripper_geometry_text(jaw_closed_rad, jaw_open_rad, *, archive_path=None) -> str:
    """Compact model-readable ``gripper_geometry`` text block (m, rad; CAD, deployment jaw range)."""
    p = gripper_geometry_payload(jaw_closed_rad, jaw_open_rad, archive_path=archive_path)
    f, m, box, reach, roll = (p["fixed_jaw"], p["moving_jaw"], p["bbox_tcp_m"], p["moving_jaw_reach"],
                              p["wrist_roll_axis_tcp"])
    mm = lambda v: f"{1000*v:.1f} mm"  # noqa: E731
    xyz = lambda v: "(" + ", ".join(f"{c:+.4f}" for c in v) + ")"  # noqa: E731
    span = lambda b: f"x {b['min_m'][0]:+.4f}..{b['max_m'][0]:+.4f}, y {b['min_m'][1]:+.4f}..{b['max_m'][1]:+.4f}, z {b['min_m'][2]:+.4f}..{b['max_m'][2]:+.4f}"  # noqa: E731
    tip = f["fingertip_inner_point_m"]
    profile = ", ".join(f"z {row['z_m']:+.4f}: x {row['x_m']:+.4f}" for row in f["inner_face_profile_m"])
    table = "; ".join(f"{row['goal_normalized']:g} -> {row['jaw_rad']:+.3f} rad, {mm(row['opening_m'])}, "
                      f"({1000*row['moving_tip_x_m']:+.1f}, {1000*row['moving_tip_z_m']:+.1f})"
                      for row in p["goal_to_opening"])
    zero = m["fingertip_inner_point_at_zero_m"]
    lines = [
        "gripper_geometry (TCP frame gripper_frame_link, metres; CAD of the packaged URDF meshes, not measured on this "
        "unit):",
        f"- TCP: gripper_frame_link lies ON the fixed jaw's inner face at the fingertip (x {tip[0]:+.4f}), not between "
        "the jaws; the whole jaw gap is on TCP -X. Axes: +Z approach (out along the fingers), +X from the gap toward "
        "the fixed jaw, Y along the jaw hinge (finger width; the wrist camera is on +Y). The wrist_roll axis runs along "
        f"Z through x {roll['point_at_tcp_plane_m'][0]:+.4f}, y {roll['point_at_tcp_plane_m'][1]:+.4f}, so turning "
        f"wrist_roll swings the TCP on a {mm(roll['tcp_distance_m'])} radius.",
        f"- Fixed jaw (+X, rigid): inner fingertip point (x {tip[0]:+.4f}, y {tip[1]:+.4f}, z {tip[2]:+.4f}); inner "
        f"face x by depth z: {profile}; finger length {mm(f['finger_length_m'])} to the pocket floor at z "
        f"{p['pocket_floor_z_m']:+.4f}.",
        f"- Moving jaw (-X, the only finger that moves; hinge axis {xyz(m['hinge_axis_tcp'])}). goal_normalized -> "
        f"jaw angle, opening (tip-to-tip distance), moving inner fingertip (x, z) in mm: {table}. Goal 0 and 1 are "
        "CAD values at the commanded jaw limits; the real jaw stops short of them (readings of this unit below).",
        f"- Fingertip midpoint = (fixed tip + moving tip) / 2: it moves with the opening (x {0.5*(tip[0] + zero[0]):+.4f} "
        "at jaw 0 rad, about -opening/2 while both tips are at the same depth), so the TCP is not the grasp centre.",
        f"- Reach beyond the TCP along +Z: fixed jaw {mm(reach['fixed_reach_m'])} at every opening; moving jaw "
        f"{mm(reach['moving_reach_at_goal0_m'])} at goal 0, up to {mm(reach['peak_reach_m'])} near goal "
        f"{reach['peak_goal']:.2f}"
        + (f" (further than the fixed jaw for goal {reach['beyond_fixed_jaw_goals'][0]:.2f}.."
           f"{reach['beyond_fixed_jaw_goals'][1]:.2f})" if reach["beyond_fixed_jaw_goals"] else "")
        + (f", and wholly behind the TCP plane above goal {reach['behind_tcp_plane_above_goal']:.2f} (opening "
           f"{mm(reach['behind_tcp_plane_opening_m'])}): there only the fixed jaw is at TCP depth, it meets a support "
           "or an object top first on a downward approach, and the opening is no longer a gap at TCP depth."
           if reach["behind_tcp_plane_above_goal"] is not None else "."),
        "- Grasping an object of width w (m, along TCP X): closing pushes it along +X until its +X face rests on the "
        "fixed inner face, so it is grasped where it stands only with that face already on it; open to more than w "
        "first. robot_action grasp_centre_m computes that TCP target (its field help is the conversion) with the face "
        f"at the tip (x {f['inner_face_profile_m'][0]['x_m']:+.4f}); deeper the face lies further toward +X (profile "
        f"above, up to x {max(row['x_m'] for row in f['inner_face_profile_m']):+.4f}), so an object gripped deeper is "
        "pushed that much further along +X as the jaw closes.",
        f"- Gripper body bbox: {span(box['gripper_body'])}; moving jaw swept bbox: {span(box['moving_jaw_swept'])}.",
    ]
    if box["wrist_camera"]:
        lines.append(f"- Wrist camera + mount bbox: {span(box['wrist_camera'])} (beside the jaws on +Y, behind the TCP).")
    lines.append(f"- Overall bbox: {span(box['overall'])}.")
    return "\n".join(lines)


# ---- grasp centre between the fixed and the single moving jaw (robot_action grasp_centre_m; pose input tag
# ``jaw_grasp_centre``). Pure functions of the CAD above; TCP frame, metres and radians. ---------------------------
def jaw_gap_m(jaw_rad, *, archive_path=None) -> float:
    """Gap (m) across the jaw line: fixed fingertip inner face x minus the moving fingertip inner point x (TCP X).

    ``jaw_rad``: URDF ``gripper`` joint angle (rad). Unlike ``opening_width_m`` (the 3D fingertip distance) this is
    the width along TCP X that fits between the fingertips, also when the moving fingertip has swung back along -Z.
    Increases with ``jaw_rad`` up to about 1.72 rad (0.110 m) and is flat within 0.1 mm from there to the open limit.
    Example: ``jaw_gap_m(0.0)`` -> about 0.0158."""
    angle = float(jaw_rad)
    if not math.isfinite(angle):
        raise ValueError("so101_jaw_rad_must_be_finite")
    g = _geometry(archive_path)
    return float(g.fixed_tip[0] - g.moving_point(g.moving_tip_local, angle)[0])


def jaw_rad_for_gap(gap_m, *, archive_path=None):
    """Smallest URDF jaw angle (rad) whose ``jaw_gap_m`` reaches ``gap_m`` (m); ``None`` outside the gap range of the
    declared jaw limits. Example: ``jaw_rad_for_gap(0.04)`` -> about 0.30."""
    g = _geometry(archive_path)
    gap = float(gap_m)
    # The gap grows up to its widest sweep sample (about 1.72 rad); the search stays on that monotonic part.
    lo, hi = float(g.jaw_limits[0]), float(max(g.sweep, key=lambda q: jaw_gap_m(q, archive_path=archive_path)))
    if not math.isfinite(gap) or not jaw_gap_m(lo, archive_path=archive_path) <= gap <= jaw_gap_m(hi, archive_path=archive_path):
        return None
    for _ in range(60):
        mid = 0.5*(lo + hi)
        lo, hi = (mid, hi) if jaw_gap_m(mid, archive_path=archive_path) < gap else (lo, mid)
    return 0.5*(lo + hi)


def moving_fingertip_depth_m(jaw_rad, *, archive_path=None) -> float:
    """How far (m) the moving fingertip inner point lies behind the TCP plane (toward the palm, TCP -Z) at
    ``jaw_rad`` (rad); negative = beyond the TCP plane. The same axis and sign as ``grasp_depth_m``.
    Example: about -0.006 at 0 rad, 0 at about 0.58 rad, +0.034 at 1.2 rad (the tip swings back as the jaw opens)."""
    angle = float(jaw_rad)
    if not math.isfinite(angle):
        raise ValueError("so101_jaw_rad_must_be_finite")
    g = _geometry(archive_path)
    return float(-g.moving_point(g.moving_tip_local, angle)[2])


def jaw_gap_reaching_depth_m(grasp_depth_m=0.0, *, archive_path=None) -> float:
    """Largest jaw gap (m, ``jaw_gap_m``) at which the moving fingertip still reaches the plane ``grasp_depth_m``
    (m) behind the TCP plane, i.e. ``moving_fingertip_depth_m <= grasp_depth_m``. A wider object centred at that
    depth is closed on behind its centre plane. The gap at the open limit when the fingertip reaches the plane over
    the whole range; the gap at the closed limit when it never does. Example: about 0.063 for depth 0 (the TCP
    plane), about 0.079 for 0.010."""
    g = _geometry(archive_path)
    depth = float(grasp_depth_m)
    if not math.isfinite(depth):
        raise ValueError("so101_grasp_depth_must_be_finite_m")
    reaches = [moving_fingertip_depth_m(q, archive_path=archive_path) <= depth for q in g.sweep]
    if not any(reaches):
        return jaw_gap_m(g.sweep[0], archive_path=archive_path)
    last = max(index for index, ok in enumerate(reaches) if ok)
    if last == len(g.sweep) - 1:
        return jaw_gap_m(g.sweep[-1], archive_path=archive_path)
    lo, hi = float(g.sweep[last]), float(g.sweep[last + 1])
    for _ in range(60):
        mid = 0.5*(lo + hi)
        lo, hi = (mid, hi) if moving_fingertip_depth_m(mid, archive_path=archive_path) <= depth else (lo, mid)
    return jaw_gap_m(lo, archive_path=archive_path)


def grasp_centre_tcp_offset_m(object_width_m, fixed_jaw_clearance_m=0.0, grasp_depth_m=0.0, *, archive_path=None) -> list:
    """TCP origin minus grasp centre, in TCP axes ``[x, y, z]`` (m): the vector the Harness adds (rotated into the
    base frame) to a commanded ``grasp_centre_m`` to get the ``gripper_frame_link`` target.

    The grasp centre is where the centre of the object sits between the jaws: ``object_width_m / 2`` (m, the
    object's extent along TCP X) plus ``fixed_jaw_clearance_m`` (m, >= 0, default 0: gap left between the fixed
    fingertip face and the object) from the fixed fingertip inner face toward the moving jaw, on the finger
    centreline (TCP y = 0), ``grasp_depth_m`` (m, default 0; positive toward the palm) behind the TCP plane.
    Only the -X jaw moves, so the offset does not depend on the jaw angle:
    ``[centered_grasp_tcp_offset_m(object_width_m) + fixed_jaw_clearance_m, 0, grasp_depth_m]``.
    Example: ``grasp_centre_tcp_offset_m(0.03, 0.004)`` -> about ``[0.019, 0.0, 0.0]``."""
    clearance, depth = float(fixed_jaw_clearance_m), float(grasp_depth_m)
    if not math.isfinite(clearance) or clearance < 0:
        raise ValueError("so101_fixed_jaw_clearance_must_be_finite_nonnegative_m")
    if not math.isfinite(depth):
        raise ValueError("so101_grasp_depth_must_be_finite_m")
    return [centered_grasp_tcp_offset_m(object_width_m, archive_path=archive_path) + clearance, 0.0, depth]


def grasp_centre_reference(jaw_rad, *, archive_path=None) -> dict:
    """The measured counterpart of ``grasp_centre_tcp_offset_m`` in TCP coordinates (m) at jaw angle ``jaw_rad``
    (rad, measured): ``centre_m`` = midway between the fixed fingertip face and the moving fingertip along TCP X, on
    the finger centreline and the TCP plane (the commanded grasp centre of an object as wide as the present gap, with
    clearance 0 and depth 0), ``fixed_tip_m`` / ``moving_tip_m`` (CAD inner fingertip points, ``fingertip_reference``)
    and ``gap_m`` (``jaw_gap_m``). Example: ``grasp_centre_reference(0.3)["centre_m"]`` -> about ``[-0.020, 0, 0]``."""
    tips = fingertip_reference(jaw_rad, archive_path=archive_path)
    fixed, moving = tips["fixed_tip_m"], tips["moving_tip_m"]
    return dict(centre_m=[0.5*(fixed[0] + moving[0]), 0.0, 0.0], fixed_tip_m=fixed, moving_tip_m=moving,
                gap_m=fixed[0] - moving[0])


def jaw_grasp_facts(jaw_open_rad=None, *, archive_path=None) -> dict:
    """CAD numbers of the grasp-centre input (m, TCP frame), rounded to 0.1 mm:

    ``fixed_face_x_m``: x of the fixed fingertip inner face. ``min_grasp_depth_m`` / ``max_grasp_depth_m``: the
    ``grasp_depth_m`` of the fixed fingertip inner point (negative: beyond the TCP plane) and of the pocket floor.
    ``tcp_plane_gap_m``: ``jaw_gap_reaching_depth_m(0)``, the jaw gap above which the moving fingertip is behind the
    TCP plane. ``max_gap_m``: the widest ``jaw_gap_m`` up to ``jaw_open_rad`` (rad; None = the URDF open limit),
    i.e. the widest ``object_width_m + fixed_jaw_clearance_m`` the jaw can open to."""
    g = _geometry(archive_path)
    opened = float(g.jaw_limits[1] if jaw_open_rad is None else jaw_open_rad)
    widest = max(jaw_gap_m(q, archive_path=archive_path) for q in (*(q for q in g.sweep if q < opened), opened))
    return {"fixed_face_x_m": round(float(g.fixed_tip[0]), 4) + 0.0,
            "min_grasp_depth_m": round(float(-g.fixed_tip[2]), 4), "max_grasp_depth_m": round(float(-g.palm_z), 4),
            "tcp_plane_gap_m": round(jaw_gap_reaching_depth_m(0.0, archive_path=archive_path), 4),
            "max_gap_m": round(widest, 4)}


class SO101JawFingertipHook:
    """``GraspReferenceHook`` implementation for SO-101's CAD swinging-jaw fingertip midpoint."""

    def locate(
        self,
        gripper: Any,
        origin: np.ndarray,
        rotation: np.ndarray,
        jaw_rad: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None] | None:
        if jaw_rad is None:
            return None
        tips = fingertip_reference(jaw_rad)
        if tips["normal_tcp"] is None:
            return None
        return origin + rotation @ np.asarray(tips["midpoint_m"]), rotation @ np.asarray(tips["normal_tcp"]), tips

    def aperture_m(self, gripper: Any, jaw_rad: float) -> float | None:
        return opening_width_m(jaw_rad)
