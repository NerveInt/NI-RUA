"""Queries on a scene3d point cloud inside program_run (features.scene3d; copied into the job workspace).

    import scene3d_helpers as s3
    s3.load("artifact:ref_3")              # handle returned by geometry_query kind scene3d
    s3.top_z({"center": [0.1, -0.2], "radius": 0.03})

The cloud (npz: xyz float32 m in the observation's reference frame, the frame_id of the scene3d reply; conf uint8
depth-confidence rank 0..255, uv uint16 source pixel (column, row), count uint16, robot bool, support_z, voxel_m,
image_size) is a VGGT-Omega monocular estimate scaled to the table plane. Functions return small dicts of numbers; the points themselves only with xyz=True. No per-object depth
correction is applied here: thin, white or transparent parts can sit above or in front of their true place.
Only numpy is required; load() reads the artifact through NiRuaClient.
"""
import io
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np

# API entries whose functions need the 6R IK prefilter (ik_prefilter.py, staged only for an embodiment that takes
# grasp_angles and has packaged kinematics). The only place that says which entries those are.
IK_API = (
    "check_reachability(arm,position_m,angles_deg,contact_region='pad_center') -> reachable,joint_q_rad,pos_err_m,rot_err_rad,min_joint_margin_rad (<1ms 6R IK; yaw+-180 deg)",
    "prefilter_stages(stages) -> valid,blocking_stage_ids,reasons,min_bimanual_clearance_m (<3ms 6R IK, travel rad, clearance m)",
    "solve_reachable_pose(arm,position_m,angles_deg)->selected_angles_deg; check_source_to_dest(source_pos_m,source_angles_deg,dest_pos_m,dest_angles_deg=None)->recommended_arm,requires_handoff; plan_joint_safe_return(target_robot_state)->valid,stages",
)
API = (  # 与 ni_rua/perception/external/scene3d.py API 逐字符一致（由 tests/test_obs3d.py 严格校验）
    "load(artifact)->cloud; info()->n,bbox,table_z,voxel_m; table_z(xy=None,radius=0.08)->z (m)",
    "points_in_box(lo,hi) / points_in_cylinder(center_xy,radius,z_min=None,z_max=None) -> n,centroid,min,max,top_z,conf_p50",
    "top_z(region,q=98) -> pct z (m) above table; region=(lo,hi)|{center:[x,y],radius:r}|{polygon:[[x,y],...]} (+z_min,z_max)",
    "nearest(p,k=5) -> xyz,dist,conf,uv; cluster_at_pixel(u,v,radius=0.015) -> n,centroid,min,max,top_z,bottom_z,seed",
    "surface_extrema(region,radius=0.04,min_conf=0.05) -> floor_z,rim_z,lowest_point_m,highest_point_m; triangulate_rays(u1,v1,K1,T1,u2,v2,K2,T2) -> point_m,skew_gap_m",
    "cavity_profile(region_or_obj_id,radius=0.06,min_conf=0.05) -> rim_z_m,floor_z_m,cavity_depth_m,opening_center_xy_m,inner_clearance_radius_m,n_points",
    "in_hand_axis(grasp_center_m,radius=0.065,cloud=None,min_conf=0.1) -> yaw_deg,tilt_deg,posture,elongation,bottom_point_m,tip_offset_from_grasp_m (robot=True)",
    *IK_API,
    "align_feature_to_target(grasp_pos_m,grasp_angles_deg,feature_point_m,feature_axis,target_point_m,target_axis) -> target_grasp_position_m,target_pos_m,target_angles_deg,feature_pos_err_m,feature_axis_err_deg (m,deg)",
    "se3_screw(start_pose,axis,pivot_m=None,angle_rad=0.0,distance_m=0.0,pitch_m_per_rad=0.0,n_steps=4) -> list[{s,grasp_position_m,angles_deg}] (m,rad,deg)",
    "kwargs: min_conf=0.1 (surface_extrema/cavity_profile=0.05; 0-1), robot=False, xyz=False",
)
_API_ALL = API   # API itself loses IK_API below when ik_prefilter is not staged in this workspace


def api_for(ik_staged):
    """API entries callable in a workspace: all of ``API`` with the IK prefilter staged, else without ``IK_API``.
    The host (geometry_query kind scene3d) and this module's own ``API`` in a program workspace both use it."""
    return _API_ALL if ik_staged else tuple(entry for entry in _API_ALL if entry not in IK_API)


ABOVE_M = 0.004          # "above the table" threshold, same as the VGGT service
NEAR_TABLE_M = 0.015     # |z - table| for the local table height
_MIN_CAVITY_DEPTH_M = 0.004
_SURFACE_VOXEL_RADIUS_M = 0.006
_SPATIAL_SNAPSHOT_FILE = "_spatial_snapshot.json"
_SPATIAL_CLOUD_FILE = "_spatial_cloud.npz"
_DEFAULT_WORKSPACE_DIR = Path(__file__).resolve().parent
_CLOUD_FILE = _DEFAULT_WORKSPACE_DIR / _SPATIAL_CLOUD_FILE
_SNAPSHOT_FILE = _DEFAULT_WORKSPACE_DIR / _SPATIAL_SNAPSHOT_FILE
_DEFAULT = None
_DEFAULT_SOURCE: str | None = None
_DEFAULT_STAMP: tuple | None = None
_CLEARED_STAMP: tuple | None = None
_DEFAULT_FIELD = None


class CloudNotLoadedError(ValueError, RuntimeError):
    """未加载默认点云且工作区无有效 `_spatial_cloud.npz` 时抛出的异常（兼容 `ValueError` 与 `RuntimeError`）。"""


class Cloud:
    def __init__(self, data):
        self.xyz = np.asarray(data["xyz"], dtype=np.float64).reshape(-1, 3)
        self.conf = np.asarray(data["conf"], dtype=np.float64).reshape(-1) / 255.0
        self.uv = np.asarray(data["uv"], dtype=np.int64).reshape(-1, 2)
        self.count = np.asarray(data["count"]).reshape(-1) if "count" in data else np.ones(len(self.xyz), np.int64)
        self.robot = np.asarray(data["robot"], dtype=bool).reshape(-1) if "robot" in data else np.zeros(len(self.xyz), bool)
        self.obj_id = np.asarray(data["obj_id"], dtype=np.int16).reshape(-1) if "obj_id" in data else np.zeros(len(self.xyz), np.int16)
        self.view_mask = np.asarray(data["view_mask"], dtype=np.uint8).reshape(-1) if "view_mask" in data else np.ones(len(self.xyz), np.uint8)
        self.table = float(data["support_z"])
        self.voxel = float(data["voxel_m"]) if "voxel_m" in data else None
        self.image_size = [int(v) for v in data["image_size"]] if "image_size" in data else None

    @classmethod
    def from_npz(cls, data):
        if isinstance(data, cls):
            return data
        if isinstance(data, (str, Path)):
            data = Path(data).read_bytes()
        return from_bytes(bytes(data))

    def mask(self, min_conf=0.1, robot=False):
        keep = self.conf >= min_conf
        if not robot:
            keep &= ~self.robot
        return keep


def _r(value, digits=4):
    if value is None:
        return None
    if np.ndim(value):
        return [round(float(v), digits) for v in np.asarray(value).reshape(-1)]
    return round(float(value), digits)


def from_bytes(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        return Cloud({key: npz[key] for key in npz.files})


def _workspace_dir() -> Path:
    return Path(__file__).resolve().parent


def _cloud_path() -> Path:
    ws = _workspace_dir()
    if ws != _DEFAULT_WORKSPACE_DIR:
        return ws / _SPATIAL_CLOUD_FILE
    return Path(_CLOUD_FILE)


def _snapshot_path() -> Path:
    ws = _workspace_dir()
    if ws != _DEFAULT_WORKSPACE_DIR:
        return ws / _SPATIAL_SNAPSHOT_FILE
    return Path(_SNAPSHOT_FILE)


def _workspace_cloud_stamp() -> tuple | None:
    """读取工作区 `_spatial_cloud.npz` 与 `_spatial_snapshot.json` 的版本戳；点云文件不存在时返回 `None`。"""
    cloud_path = _cloud_path()
    if not cloud_path.is_file():
        return None
    try:
        st = cloud_path.stat()
    except OSError:
        return None
    obs_id = None
    snap_mtime_ns = None
    snap_path = _snapshot_path()
    if snap_path.is_file():
        try:
            snap_st = snap_path.stat()
            snap_mtime_ns = snap_st.st_mtime_ns
            snap = json.loads(snap_path.read_text(encoding="utf-8"))
            if isinstance(snap, dict):
                obs_id = snap.get("observation_id")
                if snap.get("snapshot_mtime_ns") is not None:
                    snap_mtime_ns = snap.get("snapshot_mtime_ns")
        except (OSError, ValueError, TypeError):
            pass
    return (obs_id, st.st_mtime_ns, st.st_size, snap_mtime_ns)


def _is_workspace_cleared() -> bool:
    if _CLEARED_STAMP is None:
        return False
    return _workspace_cloud_stamp() == _CLEARED_STAMP


def _autoload_workspace_cloud():
    """在沙盒工作区内按 `(observation_id, mtime)` 自动加载或热重载 `_spatial_cloud.npz`；失效删除后清空缓存。"""
    global _DEFAULT, _DEFAULT_SOURCE, _DEFAULT_STAMP, _CLEARED_STAMP
    if _DEFAULT_SOURCE == "pinned" and _DEFAULT is not None:
        return _DEFAULT
    stamp = _workspace_cloud_stamp()
    if stamp is None or (_CLEARED_STAMP is not None and stamp == _CLEARED_STAMP):
        _DEFAULT = None
        _DEFAULT_SOURCE = None
        _DEFAULT_STAMP = None
        return None
    _CLEARED_STAMP = None
    if _DEFAULT is None or _DEFAULT_SOURCE != "workspace" or stamp != _DEFAULT_STAMP:
        import zipfile
        _DEFAULT = None
        _DEFAULT_SOURCE = None
        _DEFAULT_STAMP = None
        try:
            raw = _cloud_path().read_bytes()
            cloud = Cloud.from_npz(raw)
        except (OSError, ValueError, zipfile.BadZipFile, EOFError):
            return None
        _DEFAULT = cloud
        _DEFAULT_SOURCE = "workspace"
        _DEFAULT_STAMP = stamp
        if _DEFAULT_FIELD is not None:
            _DEFAULT_FIELD.update_from_observation(cloud=_DEFAULT)
    return _DEFAULT


def load(artifact, client=None):
    """加载 scene3d 点云 artifact 句柄（或原始字节流 / Cloud / 路径 / dict），并固定为后续查询的默认点云。"""
    global _DEFAULT, _DEFAULT_SOURCE, _DEFAULT_STAMP, _CLEARED_STAMP, _DEFAULT_FIELD
    _CLEARED_STAMP = None
    if isinstance(artifact, Cloud):
        _DEFAULT = artifact
    elif isinstance(artifact, (bytes, bytearray, memoryview)):
        _DEFAULT = Cloud.from_npz(bytes(artifact))
    elif isinstance(artifact, Path) or (
        isinstance(artifact, str) and not artifact.startswith("artifact:") and Path(artifact).is_file()
    ):
        _DEFAULT = Cloud.from_npz(Path(artifact).read_bytes())
    else:
        if isinstance(artifact, dict) and "artifact" in artifact:
            artifact = artifact["artifact"]
        if client is None:
            from ni_rua_client import NiRuaClient
            client = NiRuaClient()
        data = client.artifact(artifact)
        _DEFAULT = data if isinstance(data, Cloud) else Cloud.from_npz(data)
    _DEFAULT_SOURCE = "pinned"
    _DEFAULT_STAMP = None
    if _DEFAULT_FIELD is not None:
        _DEFAULT_FIELD.update_from_observation(cloud=_DEFAULT)
    return _DEFAULT


def use(cloud):
    global _DEFAULT, _DEFAULT_SOURCE, _DEFAULT_STAMP, _CLEARED_STAMP, _DEFAULT_FIELD
    was_workspace = _DEFAULT_SOURCE == "workspace"
    prev_stamp = _DEFAULT_STAMP
    _DEFAULT = cloud
    _DEFAULT_STAMP = None
    if cloud is None:
        _DEFAULT_SOURCE = None
        _DEFAULT_FIELD = None
        _CLEARED_STAMP = prev_stamp if was_workspace else None
    else:
        _DEFAULT_SOURCE = "pinned"
        _CLEARED_STAMP = None
        if _DEFAULT_FIELD is not None:
            _DEFAULT_FIELD.update_from_observation(cloud=cloud)
    return cloud


def _cloud(cloud=None):
    if isinstance(cloud, (str, bytes, bytearray, memoryview, dict, Path)):
        return load(cloud)
    if cloud is not None:
        return cloud
    if _DEFAULT_SOURCE == "pinned" and _DEFAULT is not None:
        return _DEFAULT
    loaded = _autoload_workspace_cloud()
    if loaded is None:
        raise CloudNotLoadedError(
            "no default point cloud loaded (scene3d_cloud_not_loaded): call load(artifact) first"
        )
    return loaded


def info(cloud=None):
    c = _cloud(cloud)
    return dict(n=int(len(c.xyz)), bbox=[_r(c.xyz.min(0)), _r(c.xyz.max(0))] if len(c.xyz) else None,
                table_z=_r(c.table), voxel_m=c.voxel, near_robot=int(c.robot.sum()), image_size=c.image_size)


def _summary(c, keep, xyz=False):
    P = c.xyz[keep]
    out = dict(n=int(len(P)))
    if len(P):
        out.update(centroid=_r(P.mean(0)), min=_r(P.min(0)), max=_r(P.max(0)),
                   top_z=_r(np.percentile(P[:, 2], 98)), conf_p50=_r(np.median(c.conf[keep]), 2))
    else:
        out.update(centroid=None, min=None, max=None, top_z=None, conf_p50=None)
    if xyz:
        out["xyz"] = P
    return out


def _region_mask(c, region):
    """region = (lo, hi) | {'center': [x, y], 'radius': r} | {'polygon': [[x, y], ...]}, optional 'z_min'/'z_max'."""
    P = c.xyz
    if isinstance(region, (tuple, list)) and len(region) == 2 and np.size(region[0]) == 3:
        lo, hi = np.asarray(region[0], float), np.asarray(region[1], float)
        return np.all((P >= lo) & (P <= hi), axis=1)
    if not isinstance(region, dict):
        raise ValueError("region must be (lo, hi), {'center','radius'} or {'polygon'}")
    if "center" in region:
        centre = np.asarray(region["center"], float)[:2]
        keep = np.linalg.norm(P[:, :2] - centre, axis=1) <= float(region["radius"])
    elif "polygon" in region:
        keep = _in_polygon(P[:, :2], np.asarray(region["polygon"], float).reshape(-1, 2))
    elif "min_m" in region and "max_m" in region:
        lo, hi = np.asarray(region["min_m"], float), np.asarray(region["max_m"], float)
        keep = np.all((P[:, :len(lo)] >= lo) & (P[:, :len(hi)] <= hi), axis=1)
    elif "lo" in region and "hi" in region:
        lo, hi = np.asarray(region["lo"], float), np.asarray(region["hi"], float)
        keep = np.all((P[:, :len(lo)] >= lo) & (P[:, :len(hi)] <= hi), axis=1)
    elif "min" in region and "max" in region:
        lo, hi = np.asarray(region["min"], float), np.asarray(region["max"], float)
        keep = np.all((P[:, :len(lo)] >= lo) & (P[:, :len(hi)] <= hi), axis=1)
    else:
        raise ValueError("region dict needs 'center' + 'radius', 'polygon', or 'min_m' + 'max_m'")
    if region.get("z_min") is not None:
        keep &= P[:, 2] >= float(region["z_min"])
    if region.get("z_max") is not None:
        keep &= P[:, 2] <= float(region["z_max"])
    return keep


def _in_polygon(xy, poly):
    """Even-odd rule, vectorised over points."""
    inside = np.zeros(len(xy), bool)
    x, y = xy[:, 0], xy[:, 1]
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        crosses = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (x < xc)
    return inside


def points_in_box(lo, hi=None, z_or_hi=None, cloud=None, min_conf=0.1, robot=False, xyz=False):
    if isinstance(lo, Cloud):
        cloud, lo, hi = lo, hi, z_or_hi
    c = _cloud(cloud)
    return _summary(c, _region_mask(c, (lo, hi)) & c.mask(min_conf, robot), xyz)


def points_in_cylinder(center_xy, radius=0.05, z_min=None, z_max=None, cloud=None, min_conf=0.1, robot=False, xyz=False):
    if isinstance(center_xy, Cloud):
        cloud, center_xy, radius, z_min = center_xy, radius, (0.05 if z_min is None else z_min), z_max
        z_max = None
    c = _cloud(cloud)
    region = dict(center=center_xy, radius=radius, z_min=z_min, z_max=z_max)
    return _summary(c, _region_mask(c, region) & c.mask(min_conf, robot), xyz)


def top_z(region, q=98, cloud=None, min_conf=0.1, robot=False, above=ABOVE_M, xyz=False, radius=None, z_min=None, z_max=None):
    """q-th percentile z of the points above the table (table_z + above) in region; top_z None when there are none."""
    if isinstance(region, Cloud):
        cloud, region, q = region, q, 98
    c = _cloud(cloud)
    if not isinstance(region, dict) and not (isinstance(region, (tuple, list)) and len(region) == 2 and np.size(region[0]) == 3):
        pt = np.asarray(region, dtype=float).reshape(-1)
        if len(pt) in (2, 3):
            rad = float(radius) if radius is not None else (float(q) if isinstance(q, float) and q < 1.0 else 0.05)
            if isinstance(q, float) and q < 1.0 and radius is None:
                q = 98
            region = dict(center=pt[:2].tolist(), radius=rad, z_min=z_min, z_max=z_max)
    elif isinstance(region, dict) and (z_min is not None or z_max is not None):
        region = {**region, **({"z_min": z_min} if z_min is not None else {}), **({"z_max": z_max} if z_max is not None else {})}
    keep = _region_mask(c, region) & c.mask(min_conf, robot) & (c.xyz[:, 2] > c.table + above)
    P = c.xyz[keep]
    out = dict(n=int(len(P)), top_z=_r(np.percentile(P[:, 2], q)) if len(P) else None,
               height_above_table=_r(np.percentile(P[:, 2], q) - c.table) if len(P) else None,
               conf_p50=_r(np.median(c.conf[keep]), 2) if len(P) else None)
    if xyz:
        out["xyz"] = P
    return out


def table_z(xy=None, radius=0.08, cloud=None, min_conf=0.1):
    """Without xy: the table plane z the cloud was scaled to. With xy: median z of the near-table points
    (|z - table| < 1.5 cm) within radius of xy, i.e. the local table height of this reconstruction."""
    if isinstance(xy, Cloud):
        cloud = xy
        xy = radius if isinstance(radius, (list, tuple, np.ndarray)) else None
        if xy is not None:
            radius = 0.08
    c = _cloud(cloud)
    if xy is None:
        return c.table
    keep = (c.mask(min_conf, False) & (np.abs(c.xyz[:, 2] - c.table) < NEAR_TABLE_M)
            & (np.linalg.norm(c.xyz[:, :2] - np.asarray(xy, float)[:2], axis=1) <= radius))
    z = c.xyz[keep, 2]
    return dict(n=int(len(z)), z=_r(np.median(z)) if len(z) else None,
                z_p10_p90=[_r(np.percentile(z, 10)), _r(np.percentile(z, 90))] if len(z) else None)


def nearest(p, k=5, cloud=None, min_conf=0.1, robot=False):
    if isinstance(p, Cloud):
        cloud, p, k = p, k, (cloud if isinstance(cloud, int) else 5)
    c = _cloud(cloud)
    keep = np.flatnonzero(c.mask(min_conf, robot))
    p = np.asarray(p, float).reshape(-1)
    dims = 2 if len(p) == 2 else 3
    d = np.linalg.norm(c.xyz[keep, :dims] - p[:dims], axis=1)
    order = keep[np.argsort(d)[:int(k)]]
    dist = np.linalg.norm(c.xyz[order, :dims] - p[:dims], axis=1)
    return dict(xyz=[_r(v) for v in c.xyz[order]], dist=_r(dist), conf=_r(c.conf[order], 2),
                uv=c.uv[order].tolist())


def cluster_at_pixel(u, v=None, radius=0.015, cloud=None, min_conf=0.1, robot=False, above=ABOVE_M, max_extent=0.4,
                     xyz=False):
    """Points above the table connected (link distance <= radius) to the cloud point whose source pixel is nearest to
    (u, v). Growth stops at max_extent from the seed. Returns n, centroid, min, max, top_z, bottom_z, seed."""
    if isinstance(u, Cloud):
        cloud = u
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 2:
            u, v = v[0], v[1]
        else:
            u, v, radius = v, radius, 0.015
    elif isinstance(u, (list, tuple, np.ndarray)) and len(u) == 2 and v is None:
        u, v = u[0], u[1]
    c = _cloud(cloud)
    keep = np.flatnonzero(c.mask(min_conf, robot) & (c.xyz[:, 2] > c.table + above))
    if not len(keep):
        return dict(n=0, seed=None)
    pix = np.linalg.norm(c.uv[keep] - np.asarray([u, v], float), axis=1)
    seed = keep[int(np.argmin(pix))]
    P = c.xyz[keep]
    local = np.linalg.norm(P - c.xyz[seed], axis=1) <= max_extent
    ids, Q = keep[local], P[local]
    visited = np.zeros(len(Q), bool)
    if len(Q):
        from scipy.spatial import cKDTree

        r_val = float(radius)
        seed_local = int(np.flatnonzero(ids == seed)[0])
        visited[seed_local] = True
        if len(Q) > 1 and r_val >= 0.0:
            tree = cKDTree(Q)
            frontier = np.array([seed_local], dtype=np.intp)
            ub = r_val * (1.0 + 1e-12) + 1e-15
            while len(frontier):
                if len(frontier) <= 16:
                    neighbor_lists = tree.query_ball_point(Q[frontier], r=r_val, return_sorted=False)
                    flat = np.fromiter(itertools.chain.from_iterable(neighbor_lists), dtype=np.intp)
                    if not len(flat):
                        break
                    next_mask = np.zeros(len(Q), bool)
                    next_mask[flat] = True
                    next_mask &= ~visited
                    if not next_mask.any():
                        break
                    visited |= next_mask
                    frontier = np.flatnonzero(next_mask)
                else:
                    unv = np.flatnonzero(~visited)
                    if not len(unv):
                        break
                    _, nn = cKDTree(Q[frontier]).query(Q[unv], k=1, distance_upper_bound=ub)
                    cand = nn < len(frontier)
                    if not cand.any():
                        break
                    unv_c = unv[cand]
                    newly = unv_c[np.linalg.norm(Q[unv_c] - Q[frontier[nn[cand]]], axis=1) <= r_val]
                    if not len(newly):
                        break
                    visited[newly] = True
                    frontier = newly
    sel = np.zeros(len(c.xyz), bool)
    sel[ids[visited]] = True
    out = _summary(c, sel, xyz)
    out["bottom_z"] = _r(np.percentile(c.xyz[sel, 2], 2))
    out["seed"] = dict(xyz=_r(c.xyz[seed]), uv=c.uv[seed].tolist(), pixel_distance=_r(pix.min(), 1))
    return out


def in_hand_axis(grasp_center_m, radius=0.065, cloud=None, min_conf=0.1, above=0.015, robot=True, xyz=False,
                 z_clearance_above_table=None):
    """基于 `robot=True` 的指尖邻域 3D 点云簇做主成分分析 (PCA)，估计夹爪中持握物体的实际 3D 主轴姿态与最低端坐标。

    参数:
        grasp_center_m: 当前夹爪接触中心世界坐标 `[x, y, z]`，单位 `m`。
        radius: 指尖邻域球体裁剪半径，单位 `m`，默认 `0.065`。
        cloud: 可选点云对象或工件引用；默认使用当前已加载点云。
        min_conf: 最小点云深度置信度阈值（`[0, 1]`），默认 `0.1`。
        above: 高于支撑台面 `table_z` 的最小净空高度，单位 `m`，默认 `0.015`。
        robot: 是否纳入机械臂连杆邻域（指尖内持握物）点云，默认 `True`。
        xyz: 是否附带裁剪后的原始点坐标数组，默认 `False`。
        z_clearance_above_table: `above` 的兼容别名，单位 `m`，默认 `None`。

    返回:
        字典 `{n, centroid, axis_world, yaw_deg, tilt_deg, elongation, posture, bottom_point_m, tip_offset_from_grasp_m}`，
        其中坐标与偏移单位为 `m`，`yaw_deg` / `tilt_deg` 单位为 `deg`。

    示例:
        `s3.in_hand_axis([0.22, -0.14, 0.86], radius=0.065, min_conf=0.1, robot=True)`
    """
    if z_clearance_above_table is not None:
        above = float(z_clearance_above_table)
    if isinstance(grasp_center_m, Cloud):
        cloud = grasp_center_m
        grasp_center_m = radius
        radius = float(min_conf) if isinstance(min_conf, (int, float)) and min_conf != 0.1 else 0.065
    c = _cloud(cloud)
    center = np.asarray(grasp_center_m, dtype=float).reshape(3)
    keep = (
        c.mask(min_conf, robot=bool(robot))
        & (c.xyz[:, 2] > c.table + above)
        & (np.linalg.norm(c.xyz - center[None, :], axis=1) <= float(radius))
    )
    P = c.xyz[keep]
    if len(P) < 5:
        return dict(n=int(len(P)), centroid=None, yaw_deg=None, tilt_deg=None, posture="unknown", elongation=None)
    mean = P.mean(0)
    centered = P - mean
    cov = (centered.T @ centered) / max(len(P) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 1e-12)
    eigvecs = eigvecs[:, order]
    axis = eigvecs[:, 0].copy()
    if axis[2] < 0 or (abs(axis[2]) < 1e-6 and axis[0] < 0):
        axis = -axis
    tilt_deg = float(np.degrees(np.arccos(np.clip(abs(axis[2]), 0.0, 1.0))))
    yaw_deg = float((np.degrees(np.arctan2(axis[1], axis[0])) + 90.0) % 180.0 - 90.0)
    elongation = float(np.sqrt(eigvals[0] / eigvals[1]))
    posture = "upright" if tilt_deg < 35.0 else ("lying" if tilt_deg > 55.0 else "tilted")
    z_low_thr = float(np.percentile(P[:, 2], 5))
    low_pts = P[P[:, 2] <= z_low_thr + 1e-6]
    bottom_pt = low_pts.mean(0) if len(low_pts) else P[int(np.argmin(P[:, 2]))]
    tip_offset = bottom_pt - center
    out = _summary(c, keep, xyz)
    out.update(
        axis_world=_r(axis, 4),
        yaw_deg=_r(yaw_deg, 1),
        tilt_deg=_r(tilt_deg, 1),
        elongation=_r(elongation, 2),
        posture=posture,
        bottom_point_m=_r(bottom_pt, 4),
        tip_offset_from_grasp_m=_r(tip_offset, 4),
    )
    return out


in_hand_geometry = in_hand_axis


def surface_extrema(region, radius=0.04, cloud=None, min_conf=0.05, robot=False, xyz=False):
    """无固定高度截断的局部表面 3D 高程极值分析（直接返回槽底/孔位 lowest_point_m 与边缘/最高点 highest_point_m）。"""
    if isinstance(region, Cloud):
        cloud, region = region, radius
        radius = 0.04
    c = _cloud(cloud)
    if not isinstance(region, dict) and not (isinstance(region, (tuple, list)) and len(region) == 2 and np.size(region[0]) == 3):
        pt = np.asarray(region, dtype=float).reshape(-1)
        region = dict(center=pt[:2].tolist(), radius=float(radius))
    keep = _region_mask(c, region) & c.mask(min_conf, robot)
    P = c.xyz[keep]
    if not len(P):
        return dict(n=0, floor_z=None, rim_z=None, relief_m=None, lowest_point_m=None, highest_point_m=None)
    floor_z = float(np.percentile(P[:, 2], 5))
    rim_z = float(np.percentile(P[:, 2], 95))
    low_band = P[P[:, 2] <= floor_z + 0.003]
    high_band = P[P[:, 2] >= rim_z - 0.003]
    low_pt = low_band.mean(0) if len(low_band) else P[int(np.argmin(P[:, 2]))]
    high_pt = high_band.mean(0) if len(high_band) else P[int(np.argmax(P[:, 2]))]
    out = dict(
        n=int(len(P)),
        floor_z=_r(floor_z, 4),
        rim_z=_r(rim_z, 4),
        relief_m=_r(rim_z - floor_z, 4),
        lowest_point_m=_r(low_pt, 4),
        highest_point_m=_r(high_pt, 4),
        table_z=_r(c.table, 4),
    )
    if xyz:
        out["xyz"] = P
    return out


def _unpack_camera_calib(K_or_cam, T_val, D_val, dist_kw):
    """提取单目相机 `K (3, 3)`、`T_world_camera (4, 4)` 与 OpenCV 畸变向量 `D`（支持传入相机标定字典）。"""
    eff_D = D_val if D_val is not None else dist_kw
    eff_T = T_val
    if isinstance(K_or_cam, dict):
        cal = K_or_cam.get("calibration") if isinstance(K_or_cam.get("calibration"), dict) else K_or_cam
        intr = K_or_cam.get("intrinsics") if "intrinsics" in K_or_cam else cal.get("intrinsics")
        K_raw = None
        for cand in (K_or_cam.get("K"), cal.get("K"), K_or_cam.get("camera_matrix"), cal.get("camera_matrix")):
            if cand is not None:
                K_raw = cand
                break
        if K_raw is None and intr is not None:
            if isinstance(intr, dict):
                if intr.get("K") is not None:
                    K_raw = intr["K"]
                elif intr.get("camera_matrix") is not None:
                    K_raw = intr["camera_matrix"]
                elif "fx" in intr and "fy" in intr:
                    K_raw = [
                        [float(intr["fx"]), float(intr.get("skew", 0.0)), float(intr.get("cx", 0.0))],
                        [0.0, float(intr["fy"]), float(intr.get("cy", 0.0))],
                        [0.0, 0.0, 1.0],
                    ]
            else:
                K_raw = intr
        if K_raw is None:
            raise ValueError("camera dict for triangulate_rays must contain 'K' or 'intrinsics'")
        if eff_D is None:
            sources = [K_or_cam, cal]
            if isinstance(intr, dict):
                sources.append(intr)
            for src in sources:
                for d_key in ("D", "dist_coeffs", "distortion_coefficients", "distortion"):
                    if src.get(d_key) is not None:
                        eff_D = src[d_key]
                        break
                if eff_D is not None:
                    break
        if eff_T is None:
            for src in (K_or_cam, cal):
                for t_key in ("T", "T_world_camera"):
                    if src.get(t_key) is not None:
                        eff_T = src[t_key]
                        break
                if eff_T is None and isinstance(src.get("extrinsics"), dict):
                    ext = src["extrinsics"]
                    eff_T = ext.get("T_world_camera") if ext.get("T_world_camera") is not None else ext.get("T")
                elif eff_T is None and src.get("extrinsics") is not None:
                    eff_T = src["extrinsics"]
                if eff_T is not None:
                    break
    else:
        K_raw = K_or_cam
    K_arr = np.asarray(K_raw, dtype=np.float64).reshape(3, 3)
    T_arr = np.asarray(eff_T, dtype=np.float64).reshape(4, 4)
    D_arr = None
    if eff_D is not None:
        d_flat = np.asarray(eff_D, dtype=np.float64).reshape(-1)
        if d_flat.size > 0 and np.any(np.abs(d_flat) > 0.0):
            D_arr = d_flat
    return K_arr, T_arr, D_arr


def _camera_unit_ray_world(u, v, K_arr: np.ndarray, T_arr: np.ndarray, D_arr: np.ndarray | None) -> np.ndarray:
    """求原始图像像素 `(u, v)`（单位 `px`）经镜头去畸变后的世界系单位视线方向向量。"""
    u_f, v_f = float(u), float(v)
    if D_arr is not None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("cv2 is required for non-zero lens distortion D in triangulate_rays") from exc
        xy_n = cv2.undistortPoints(
            np.array([[[u_f, v_f]]], dtype=np.float64),
            K_arr,
            D_arr,
        ).reshape(2)
        d_cam = np.array([float(xy_n[0]), float(xy_n[1]), 1.0], dtype=np.float64)
    elif K_arr[0, 1] == 0.0 and K_arr[2, 0] == 0.0 and K_arr[2, 1] == 0.0 and K_arr[2, 2] == 1.0:
        d_cam = np.array(
            [(u_f - K_arr[0, 2]) / K_arr[0, 0], (v_f - K_arr[1, 2]) / K_arr[1, 1], 1.0],
            dtype=np.float64,
        )
    else:
        d_cam = np.linalg.solve(K_arr, np.array([u_f, v_f, 1.0], dtype=np.float64))
    d_cam = d_cam / max(float(np.linalg.norm(d_cam)), 1e-12)
    d_world = T_arr[:3, :3] @ d_cam
    return d_world / max(float(np.linalg.norm(d_world)), 1e-12)


def triangulate_rays(
    u1,
    v1,
    K1,
    T1,
    u2,
    v2,
    K2,
    T2,
    D1=None,
    D2=None,
    *,
    dist_coeffs1=None,
    dist_coeffs2=None,
):
    """双目校准相机像素射线去畸变三角化：精确反解镜头径向/切向畸变并求两相机视线最小二乘三维中点。

    参数:
        u1, v1: 相机 1 原始图像像素坐标 `(column, row)`，单位 `px`（典型范围 `0..width-1`, `0..height-1`）。
        K1: 相机 1 内参矩阵 `3x3`（焦距与主点单位 `px`），或相机标定字典（如 `{'K': K1, 'D': D1}` / `{'intrinsics': ..., 'dist_coeffs': ...}`；未显式传 `D1` 时自动从中提取畸变系数）。
        T1: 相机 1 光心坐标系到基座/世界坐标系的刚体变换矩阵 `4x4`（平移分量单位 `m`，相机光学轴 `+X` 右、`+Y` 下、`+Z` 前；若 `K1` 为含 `'T'`/`'T_world_camera'` 的标定字典可传 `None`）。
        u2, v2: 相机 2 原始图像像素坐标 `(column, row)`，单位 `px`。
        K2: 相机 2 内参矩阵 `3x3`（单位 `px`），或含内参与畸变系数的相机标定字典。
        T2: 相机 2 光心坐标系到同一基座/世界坐标系的刚体变换矩阵 `4x4`（平移分量单位 `m`）。
        D1, D2: 可选相机 1 / 相机 2 的 OpenCV 镜头畸变系数向量 `[k1, k2, p1, p2, k3, ...]`（无量纲，支持 4/5/8/12/14 项径向与切向模型）；默认 `None` 表示零畸变针孔模型。
        dist_coeffs1, dist_coeffs2: `D1` / `D2` 的关键字兼容别名，默认 `None`。

    返回:
        字典 `{status, point_m, ray1_point_m, ray2_point_m, skew_gap_m, depth1_m, depth2_m}`：
        - `status`: `'ok'`（正深度交会）、`'behind_camera'`（交点位于相机后方）或 `'parallel_rays'`（近似平行射线）。
        - `point_m`: 双视线公垂线段三维中点 `[x, y, z]`，单位 `m`。
        - `ray1_point_m`, `ray2_point_m`: 射线 1 与射线 2 上的最近点三维坐标，单位 `m`。
        - `skew_gap_m`: 双射线异面最小间距 `||ray1_point_m - ray2_point_m||_2`，单位 `m`（标定良好且匹配一致时通常 `< 0.005 m`）。
        - `depth1_m`, `depth2_m`: 最近点沿各相机单位视线的正投影距离，单位 `m`。

    示例:
        `tri = s3.triangulate_rays(390.5, 253.3, K1, T1, 237.7, 226.4, K2, T2, D1=[-0.32, 0.14, 0.002, -0.001, -0.03], D2=D2)`
        `tri = s3.triangulate_rays(390.5, 253.3, {'K': K1, 'D': D1}, T1, 237.7, 226.4, {'K': K2, 'D': D2}, T2)`
    """
    K1_arr, T1_arr, D1_arr = _unpack_camera_calib(K1, T1, D1, dist_coeffs1)
    K2_arr, T2_arr, D2_arr = _unpack_camera_calib(K2, T2, D2, dist_coeffs2)
    o1, o2 = T1_arr[:3, 3], T2_arr[:3, 3]
    d1 = _camera_unit_ray_world(u1, v1, K1_arr, T1_arr, D1_arr)
    d2 = _camera_unit_ray_world(u2, v2, K2_arr, T2_arr, D2_arr)
    w0 = o1 - o2
    b = float(np.dot(d1, d2))
    denom = 1.0 - b * b
    if denom < 1e-8:
        return dict(status="parallel_rays", point_m=None, skew_gap_m=None)
    d = float(np.dot(d1, w0))
    e = float(np.dot(d2, w0))
    s = (b * e - d) / denom
    t = (e - b * d) / denom
    p1 = o1 + s * d1
    p2 = o2 + t * d2
    mid = 0.5 * (p1 + p2)
    gap = float(np.linalg.norm(p1 - p2))
    return dict(
        status="ok" if (s > 0 and t > 0) else "behind_camera",
        point_m=_r(mid, 4),
        ray1_point_m=_r(p1, 4),
        ray2_point_m=_r(p2, 4),
        skew_gap_m=_r(gap, 4),
        depth1_m=_r(s, 4),
        depth2_m=_r(t, 4),
    )


Cloud.info = lambda self: info(cloud=self)
Cloud.points_in_box = lambda self, *a, **kw: points_in_box(*a, cloud=self, **kw)
Cloud.points_in_cylinder = lambda self, *a, **kw: points_in_cylinder(*a, cloud=self, **kw)
Cloud.top_z = lambda self, *a, **kw: top_z(*a, cloud=self, **kw)
Cloud.table_z = lambda self, *a, **kw: table_z(*a, cloud=self, **kw)
Cloud.nearest = lambda self, *a, **kw: nearest(*a, cloud=self, **kw)
Cloud.cluster_at_pixel = lambda self, *a, **kw: cluster_at_pixel(*a, cloud=self, **kw)
Cloud.in_hand_axis = lambda self, *a, **kw: in_hand_axis(*a, cloud=self, **kw)
Cloud.in_hand_geometry = Cloud.in_hand_axis
Cloud.surface_extrema = lambda self, *a, **kw: surface_extrema(*a, cloud=self, **kw)

# ------------------------------------------------------------------ 轻量纯数学底座、特征配准与 6R 运动学预筛入口
if __package__:
    from .pose_math import (
        angles_deg_to_quat_wxyz,
        angles_deg_to_quat_xyzw,
        gripper_frame,
        matrix_to_quat_wxyz,
        matrix_to_quat_xyzw,
        quat_wxyz_to_angles_deg,
        quat_wxyz_to_matrix,
        quat_wxyz_to_xyzw,
        quat_xyzw_to_angles_deg,
        quat_xyzw_to_matrix,
        quat_xyzw_to_wxyz,
        wrap180,
    )
    from .field_ops import (
        align_in_hand_feature_to_target as _fo_align_feature,
        query_spatial_clearance_and_gradient as _fo_query_clearance_grad,
        so3_exp as _fo_so3_exp,
    )
    from .ik_prefilter import (
        check_pose_reachability as _ik_check_pose,
        check_source_to_dest as _ik_check_source_to_dest,
        jaw_symmetric_angles_deg as _ik_jaw_symmetric,
        plan_joint_safe_return as _ik_plan_return,
        prefilter_stages_kinematics as _ik_prefilter_stages,
        solve_reachable_pose as _ik_solve_reachable,
    )
else:
    from pose_math import (  # type: ignore[no-redef]
        angles_deg_to_quat_wxyz,
        angles_deg_to_quat_xyzw,
        gripper_frame,
        matrix_to_quat_wxyz,
        matrix_to_quat_xyzw,
        quat_wxyz_to_angles_deg,
        quat_wxyz_to_matrix,
        quat_wxyz_to_xyzw,
        quat_xyzw_to_angles_deg,
        quat_xyzw_to_matrix,
        quat_xyzw_to_wxyz,
        wrap180,
    )
    from field_ops import (  # type: ignore[no-redef]
        align_in_hand_feature_to_target as _fo_align_feature,
        query_spatial_clearance_and_gradient as _fo_query_clearance_grad,
        so3_exp as _fo_so3_exp,
    )
    try:
        from ik_prefilter import (  # type: ignore[no-redef]
            check_pose_reachability as _ik_check_pose,
            check_source_to_dest as _ik_check_source_to_dest,
            jaw_symmetric_angles_deg as _ik_jaw_symmetric,
            plan_joint_safe_return as _ik_plan_return,
            prefilter_stages_kinematics as _ik_prefilter_stages,
            solve_reachable_pose as _ik_solve_reachable,
        )
    except ImportError:
        def _missing_ik(*_a, **_kw):
            raise RuntimeError("6R grasp_angles IK prefilter is not staged for this embodiment profile")

        _ik_check_pose = _missing_ik
        _ik_check_source_to_dest = _missing_ik
        _ik_jaw_symmetric = _missing_ik
        _ik_plan_return = _missing_ik
        _ik_prefilter_stages = _missing_ik
        _ik_solve_reachable = _missing_ik
        API = api_for(False)   # s3.API lists only what this workspace can call


def _load_workspace_snapshot_dict() -> dict | None:
    """读取沙盒工作区预载快照 `_spatial_snapshot.json`（若存在）。"""
    snap_path = _snapshot_path()
    if not snap_path.is_file():
        return None
    try:
        snap = json.loads(snap_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return snap if isinstance(snap, dict) else None


def _parse_object_map(objects, *, fallback_snap: dict | None = None) -> tuple[dict[str, dict], dict[str, float], float | None]:
    """从 `object_table` 字典、物体列表或工作区快照解析标准物体几何字典表、凹腔深度表与支撑面高度（单位 `m`）。"""
    src = objects
    cav_map: dict[str, float] = {}
    table_z: float | None = None
    if isinstance(fallback_snap, dict):
        if src is None:
            src = fallback_snap.get("object_table")
        if fallback_snap.get("table_z") is not None:
            table_z = float(fallback_snap["table_z"])
        for k, v in (fallback_snap.get("cavity_depths_m") or {}).items():
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                cav_map[str(k)] = float(v)
    obj_map: dict[str, dict] = {}
    if isinstance(src, dict):
        if src.get("table_z") is not None and table_z is None:
            table_z = float(src["table_z"])
        cols = src.get("columns")
        rows = src.get("rows")
        hints = src.get("grasp_hints") if isinstance(src.get("grasp_hints"), dict) else {}
        if isinstance(cols, list) and isinstance(rows, list):
            for row in rows:
                if not isinstance(row, (list, tuple)) or len(row) < 2:
                    continue
                d = dict(zip(cols, row))
                oid = str(d.get("id") or "")
                if not oid:
                    continue
                h = hints.get(oid) if isinstance(hints.get(oid), dict) else {}
                if isinstance(h.get("cavity_depth_m"), (int, float)):
                    cav_map.setdefault(oid, float(h["cavity_depth_m"]))
                cx, cy, cz = (float(v) for v in (d.get("centre_m") or (0.0, 0.0, 0.0)))
                if isinstance(d.get("size_m"), (list, tuple)) and len(d["size_m"]) == 3:
                    lx, wy, hz = (float(v) for v in d["size_m"])
                else:
                    lx = float(d["length"]) if isinstance(d.get("length"), (int, float)) else 0.04
                    wy = float(d["width"]) if isinstance(d.get("width"), (int, float)) else 0.04
                    tz_raw = d.get("top_z") if d.get("top_z") is not None else d.get("top_z_m")
                    bz_raw = d.get("bottom_z") if d.get("bottom_z") is not None else d.get("bottom_z_m")
                    hz = (
                        max(0.002, float(tz_raw) - float(bz_raw))
                        if isinstance(tz_raw, (int, float)) and isinstance(bz_raw, (int, float))
                        else 0.04
                    )
                tz_val = d.get("top_z") if d.get("top_z") is not None else d.get("top_z_m")
                bz_val = d.get("bottom_z") if d.get("bottom_z") is not None else d.get("bottom_z_m")
                top_z = float(tz_val) if isinstance(tz_val, (int, float)) else cz + 0.5 * hz
                bot_z = float(bz_val) if isinstance(bz_val, (int, float)) else top_z - hz
                obj_map[oid] = dict(
                    id=oid,
                    centre_m=[cx, cy, cz],
                    length=lx,
                    width=wy,
                    height=hz,
                    top_z=top_z,
                    bottom_z=bot_z,
                    yaw_deg=float(d["yaw_deg"]) if isinstance(d.get("yaw_deg"), (int, float)) else 0.0,
                )
        elif "objects" in src and isinstance(src["objects"], list):
            src = src["objects"]
    extra_items = list(src) if isinstance(src, list) else []
    if isinstance(fallback_snap, dict) and isinstance(fallback_snap.get("persisted_objects"), list):
        extra_items.extend(fallback_snap["persisted_objects"])
    for item in extra_items:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        oid = str(item["id"])
        if oid in obj_map:
            continue
        c = item.get("centre_m") or item.get("center_m") or [0.0, 0.0, 0.0]
        sz = item.get("size_m") or [
            item.get("length", item.get("length_m", 0.04)),
            item.get("width", item.get("width_m", 0.04)),
            item.get("height", 0.04),
        ]
        cx, cy, cz = float(c[0]), float(c[1]), float(c[2])
        lx, wy, hz = float(sz[0]), float(sz[1]), float(sz[2])
        top_z = float(item["top_z"]) if item.get("top_z") is not None else (
            float(item["top_z_m"]) if item.get("top_z_m") is not None else cz + 0.5 * hz
        )
        bot_z = float(item["bottom_z"]) if item.get("bottom_z") is not None else (
            float(item["bottom_z_m"]) if item.get("bottom_z_m") is not None else top_z - hz
        )
        if isinstance(item.get("cavity_depth_m"), (int, float)):
            cav_map.setdefault(oid, float(item["cavity_depth_m"]))
        obj_map[oid] = dict(
            id=oid,
            centre_m=[cx, cy, cz],
            length=lx,
            width=wy,
            height=hz,
            top_z=top_z,
            bottom_z=bot_z,
            yaw_deg=float(item["yaw_deg"]) if isinstance(item.get("yaw_deg"), (int, float)) else 0.0,
        )
    return obj_map, cav_map, table_z


def cavity_profile(
    region_or_obj_id,
    radius=0.06,
    cloud=None,
    objects=None,
    field=None,
    min_conf=0.05,
    robot=False,
    xyz=False,
):
    """基于纯 NumPy 点云与可选物体表测量容器、插孔或槽位的三维凹腔剖面（零重型空间场依赖）。

    参数:
        region_or_obj_id: 持久物体 ID（如 `'o2'`）或局部查询区域（`[x, y]`、`[x, y, z]`、
            `{'center': [x, y], 'radius': r}` 或 `(lo, hi)` 包围盒，坐标单位 `m`）。
        radius: 水平搜索半径，单位 `m` (`> 0`)，默认 `0.06` (`6 cm`)。
        cloud: 可选点云 `Cloud` 实例；省略则使用 `s3.load(...)` 或工作区预载点云。
        objects: 可选 `object_table` 字典或物体列表（当按物体 ID 查询时使用，省略则读取工作区快照）。
        field: 可选外部空间场实例（若具备 `inspect_cavity` 方法则直接委托）。
        min_conf: 最小点云深度置信度阈值（`[0, 1]`），默认 `0.05`。
        robot: 是否包含机械臂连杆邻域点云，默认 `False`。
        xyz: 是否在结果中附带匹配的三维点坐标数组，默认 `False`。

    返回:
        字典 `{rim_z_m, floor_z_m, cavity_depth_m, opening_center_xy_m, inner_clearance_radius_m, n_points}`，
        所有高程、坐标与净空半径单位均为 `m`（`env_origin` 坐标系）。

    示例:
        `s3.cavity_profile('o2', radius=0.06)`
        `s3.cavity_profile({'center': [0.22, -0.10], 'radius': 0.05})`
    """
    if isinstance(region_or_obj_id, Cloud):
        cloud, region_or_obj_id = region_or_obj_id, radius
        radius = 0.06
    if field is not None and hasattr(field, "inspect_cavity"):
        return field.inspect_cavity(region_or_obj_id, radius_m=float(radius))

    r = float(radius)
    if not math.isfinite(r) or r <= 0.0:
        raise ValueError("radius must be a positive finite number")

    snap = None if _is_workspace_cleared() else _load_workspace_snapshot_dict()
    obj_map, cav_map, snap_table_z = _parse_object_map(objects, fallback_snap=snap)
    c_obj = None
    if isinstance(cloud, Cloud):
        c_obj = cloud
    elif cloud is not None:
        c_obj = _cloud(cloud)
    elif _DEFAULT_SOURCE == "pinned" and _DEFAULT is not None:
        c_obj = _DEFAULT
    else:
        c_obj = _autoload_workspace_cloud()

    eff_table_z = float(c_obj.table) if c_obj is not None and c_obj.table is not None else snap_table_z
    base_mask = (
        c_obj.mask(float(min_conf), robot=bool(robot))
        if c_obj is not None and len(c_obj.xyz) > 0
        else None
    )
    valid_xyz = (
        c_obj.xyz[base_mask]
        if base_mask is not None
        else np.empty((0, 3), dtype=np.float64)
    )

    obj: dict | None = None
    ref_xy: np.ndarray | None = None
    search_r = r
    pts = np.empty((0, 3), dtype=np.float64)

    if isinstance(region_or_obj_id, str):
        obj = obj_map.get(region_or_obj_id)
        if obj is None and _DEFAULT_FIELD is not None and region_or_obj_id in getattr(_DEFAULT_FIELD, "objects", {}):
            return _DEFAULT_FIELD.inspect_cavity(region_or_obj_id, radius_m=r)
        if obj is None:
            raise ValueError(f"target_object_not_found:{region_or_obj_id}")
        ref_xy = np.asarray(obj["centre_m"][:2], dtype=np.float64)
        search_r = max(r, 0.5 * max(float(obj["length"]), float(obj["width"])))
        if len(valid_xyz) > 0:
            yaw_rad = math.radians(float(obj["yaw_deg"]))
            cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
            dx = valid_xyz[:, 0] - float(obj["centre_m"][0])
            dy = valid_xyz[:, 1] - float(obj["centre_m"][1])
            rx = cy * dx + sy * dy
            ry = -sy * dx + cy * dy
            z_lo_bound = (eff_table_z - 0.005) if eff_table_z is not None else (float(obj["bottom_z"]) - 0.005)
            in_foot = (
                (np.abs(rx) <= 0.5 * float(obj["length"]) + 0.008)
                & (np.abs(ry) <= 0.5 * float(obj["width"]) + 0.008)
                & (valid_xyz[:, 2] >= z_lo_bound)
                & (valid_xyz[:, 2] <= float(obj["top_z"]) + 0.015)
            )
            pts = valid_xyz[in_foot]
            m_track = re.fullmatch(r"o(\d+)", region_or_obj_id)
            if m_track and base_mask is not None and np.any(c_obj.obj_id == int(m_track.group(1))):
                mask_pts = c_obj.xyz[base_mask & (c_obj.obj_id == int(m_track.group(1)))]
                if len(mask_pts) > len(pts):
                    pts = mask_pts
    elif isinstance(region_or_obj_id, dict) and "center" in region_or_obj_id:
        c_raw = np.asarray(region_or_obj_id["center"], dtype=np.float64).ravel()
        ref_xy = c_raw[:2]
        search_r = float(region_or_obj_id.get("radius", r))
        if len(valid_xyz) > 0:
            d_xy = np.hypot(valid_xyz[:, 0] - ref_xy[0], valid_xyz[:, 1] - ref_xy[1])
            m = d_xy <= search_r
            if "z_min" in region_or_obj_id:
                m &= valid_xyz[:, 2] >= float(region_or_obj_id["z_min"])
            if "z_max" in region_or_obj_id:
                m &= valid_xyz[:, 2] <= float(region_or_obj_id["z_max"])
            pts = valid_xyz[m]
    elif (
        isinstance(region_or_obj_id, (list, tuple))
        and len(region_or_obj_id) == 2
        and isinstance(region_or_obj_id[0], (list, tuple, np.ndarray))
    ):
        lo = np.asarray(region_or_obj_id[0], dtype=np.float64).reshape(3)
        hi = np.asarray(region_or_obj_id[1], dtype=np.float64).reshape(3)
        ref_xy = 0.5 * (lo[:2] + hi[:2])
        search_r = 0.5 * float(max(hi[0] - lo[0], hi[1] - lo[1]))
        if len(valid_xyz) > 0:
            pts = valid_xyz[np.all((valid_xyz >= lo[None, :]) & (valid_xyz <= hi[None, :]), axis=1)]
    else:
        c_raw = np.asarray(region_or_obj_id, dtype=np.float64).ravel()
        if len(c_raw) < 2:
            raise ValueError("region must be an object id, [x, y], {'center': [x, y], 'radius': r}, or (lo, hi)")
        ref_xy = c_raw[:2]
        if len(valid_xyz) > 0:
            pts = valid_xyz[np.hypot(valid_xyz[:, 0] - ref_xy[0], valid_xyz[:, 1] - ref_xy[1]) <= search_r]

    xyz_extra = {"xyz": _r(pts)} if xyz else {}
    if len(pts) == 0:
        if obj is not None:
            cav = float(cav_map.get(obj["id"], 0.0))
            rim_z = float(obj["top_z"])
            floor_z = max(float(obj["bottom_z"]), rim_z - cav) if cav >= _MIN_CAVITY_DEPTH_M else rim_z
            return dict(
                rim_z_m=_r(rim_z, 4),
                floor_z_m=_r(floor_z, 4),
                cavity_depth_m=_r(max(0.0, rim_z - floor_z), 4),
                opening_center_xy_m=_r(ref_xy, 4),
                inner_clearance_radius_m=0.0,
                n_points=0,
                **xyz_extra,
            )
        return dict(
            rim_z_m=None,
            floor_z_m=None,
            cavity_depth_m=0.0,
            opening_center_xy_m=_r(ref_xy, 4) if ref_xy is not None else None,
            inner_clearance_radius_m=0.0,
            n_points=0,
            **xyz_extra,
        )

    rim_z = float(np.percentile(pts[:, 2], 95))
    z_p05 = float(np.percentile(pts[:, 2], 5))
    high_band = pts[pts[:, 2] >= rim_z - max(0.008, 0.25 * max(0.0, rim_z - z_p05))]
    if ref_xy is None:
        ref_xy = np.median(high_band[:, :2], axis=0) if len(high_band) > 0 else np.median(pts[:, :2], axis=0)

    if len(high_band) >= 4:
        lo_xy = np.min(high_band[:, :2], axis=0)
        hi_xy = np.max(high_band[:, :2], axis=0)
        span_r = 0.5 * float(np.linalg.norm(hi_xy - lo_xy))
    else:
        span_r = search_r

    inner_r_probe = max(0.010, min(search_r, 0.65 * span_r))
    d_ref = np.hypot(pts[:, 0] - ref_xy[0], pts[:, 1] - ref_xy[1])
    inner_pts = pts[d_ref <= inner_r_probe]
    floor_z = float(np.percentile(inner_pts[:, 2], 10)) if len(inner_pts) >= 3 else z_p05

    min_d_high = float(np.min(np.hypot(high_band[:, 0] - ref_xy[0], high_band[:, 1] - ref_xy[1]))) if len(high_band) > 0 else 0.0
    if rim_z - floor_z < _MIN_CAVITY_DEPTH_M:
        if obj is not None and cav_map.get(obj["id"], 0.0) >= _MIN_CAVITY_DEPTH_M:
            floor_z = max(float(obj["bottom_z"]), rim_z - float(cav_map[obj["id"]]))
        elif min_d_high >= 0.008:
            if obj is not None:
                floor_z = float(obj["bottom_z"])
            elif eff_table_z is not None:
                floor_z = float(eff_table_z)

    cavity_depth = max(0.0, rim_z - floor_z)
    if cavity_depth < _MIN_CAVITY_DEPTH_M:
        return dict(
            rim_z_m=_r(rim_z, 4),
            floor_z_m=_r(rim_z, 4),
            cavity_depth_m=0.0,
            opening_center_xy_m=_r(ref_xy, 4),
            inner_clearance_radius_m=0.0,
            n_points=int(len(pts)),
            **xyz_extra,
        )

    wall_z_cut = floor_z + max(_MIN_CAVITY_DEPTH_M, 0.35 * cavity_depth)
    wall_xy = pts[pts[:, 2] >= wall_z_cut, :2]
    if len(wall_xy) < 4:
        return dict(
            rim_z_m=_r(rim_z, 4),
            floor_z_m=_r(floor_z, 4),
            cavity_depth_m=_r(cavity_depth, 4),
            opening_center_xy_m=_r(ref_xy, 4),
            inner_clearance_radius_m=0.0,
            n_points=int(len(pts)),
            **xyz_extra,
        )

    w_lo = np.maximum(np.min(wall_xy, axis=0) + 0.002, ref_xy - search_r)
    w_hi = np.minimum(np.max(wall_xy, axis=0) - 0.002, ref_xy + search_r)
    if np.all(w_hi > w_lo):
        gx = np.linspace(w_lo[0], w_hi[0], 17)
        gy = np.linspace(w_lo[1], w_hi[1], 17)
        gxx, gyy = np.meshgrid(gx, gy)
        cands = np.vstack([ref_xy[None, :], np.column_stack([gxx.ravel(), gyy.ravel()])])
    else:
        cands = ref_xy[None, :]

    dists = np.linalg.norm(cands[:, None, :] - wall_xy[None, :, :], axis=2)
    min_wall_d = np.min(dists, axis=1)
    center_bias = 0.05 * np.linalg.norm(cands - ref_xy[None, :], axis=1)
    best_idx = int(np.argmax(min_wall_d - center_bias))
    best_xy = cands[best_idx]
    clearance_r = max(0.0, float(min_wall_d[best_idx]) - 0.5 * _SURFACE_VOXEL_RADIUS_M)
    return dict(
        rim_z_m=_r(rim_z, 4),
        floor_z_m=_r(floor_z, 4),
        cavity_depth_m=_r(cavity_depth, 4),
        opening_center_xy_m=_r(best_xy, 4),
        inner_clearance_radius_m=_r(clearance_r, 4),
        n_points=int(len(pts)),
        **xyz_extra,
    )


Cloud.cavity_profile = lambda self, *a, **kw: cavity_profile(*a, cloud=self, **kw)


def _coerce_grasp_pose(pose) -> tuple[np.ndarray, np.ndarray]:
    """解析起始刚体夹爪位姿为 `(pos_m (3,), R (3,3))`（位置单位 `m`，姿态角单位 `deg`）。"""
    if isinstance(pose, dict):
        pos_raw = pose.get("grasp_position_m") if "grasp_position_m" in pose else pose.get("position_m")
        ang_raw = pose.get("angles_deg")
        if pos_raw is None or ang_raw is None:
            raise ValueError("start_pose dict must contain grasp_position_m/position_m and angles_deg")
        pos = np.asarray(pos_raw, dtype=np.float64).reshape(3)
        ang = np.asarray(ang_raw, dtype=np.float64).reshape(3)
        return pos, quat_wxyz_to_matrix(angles_deg_to_quat_wxyz(ang[0], ang[1], ang[2]))
    if isinstance(pose, (list, tuple)) and len(pose) == 2:
        pos = np.asarray(pose[0], dtype=np.float64).reshape(3)
        ang = np.asarray(pose[1], dtype=np.float64).reshape(3)
        return pos, quat_wxyz_to_matrix(angles_deg_to_quat_wxyz(ang[0], ang[1], ang[2]))
    mat = np.asarray(pose, dtype=np.float64)
    if mat.shape == (4, 4):
        return mat[:3, 3].copy(), mat[:3, :3].copy()
    raise ValueError("start_pose must be (pos_m, angles_deg), pose dict, or 4x4 matrix")


def _matrix_to_angles_deg(R: np.ndarray) -> list[float]:
    """将 3x3 旋转矩阵还原为主分支 `angles_deg = [pitch_deg, roll_deg, yaw_deg]`（单位 `deg`）。"""
    p, r_ang, y = quat_wxyz_to_angles_deg(matrix_to_quat_wxyz(R))
    return [round(wrap180(p), 2), round(float(np.clip(r_ang, -90.0, 90.0)), 2), round(wrap180(y), 2)]


def se3_screw(
    start_pose,
    axis,
    pivot_m=None,
    angle_rad: float = 0.0,
    distance_m: float = 0.0,
    pitch_m_per_rad: float = 0.0,
    n_steps: int = 4,
    *,
    axis_dir=None,
    translation_m: float | None = None,
) -> list[dict]:
    """按 Chasles 旋量几何离散采样绕轴圆弧、铰链拨动、纯自旋或螺旋运动航点（纯函数，无外部类状态）。

    参数:
        start_pose: 起始夹爪位姿（`{'grasp_position_m': [x,y,z], 'angles_deg': [p,r,y]}`、`(pos_m, angles_deg)` 或 `4x4` 矩阵；位置单位 `m`，姿态角单位 `deg`）。
        axis: 世界系旋量轴方向向量 `[ax, ay, az]`（自动归一化；兼容关键字别名 `axis_dir`）。
        pivot_m: 旋量轴经过的世界系枢轴点 `[px, py, pz]`，单位 `m`；默认 `None` 表示轴穿过起始位置。
        angle_rad: 绕右手正向 `axis` 的总旋转角，单位 `rad`，默认 `0.0`。
        distance_m: 沿 `axis` 正向叠加的纯平移位移量，单位 `m`，默认 `0.0`（兼容关键字别名 `translation_m`）。
        pitch_m_per_rad: 螺旋螺距，单位 `m/rad`，默认 `0.0`。
        n_steps: 均匀采样航点个数（正整数 `>= 1`），默认 `4`。

    返回:
        采样航点字典列表 `[{'s': float, 'grasp_position_m': [x, y, z], 'angles_deg': [p, r, y]}, ...]`，
        位置单位 `m`，姿态角单位 `deg`，可直接填入 `robot_action` 的 `move` 阶段。

    示例:
        `wps = s3.se3_screw(([0.20, -0.10, 0.84], [0, 0, 0]), axis=[0, 1, 0], pivot_m=[0.28, -0.10, 0.80], angle_rad=0.7854, n_steps=3)`
    """
    eff_axis = axis_dir if axis is None else axis
    if eff_axis is None:
        raise ValueError("se3_screw requires axis=[ax, ay, az]")
    u = np.asarray(eff_axis, dtype=np.float64).reshape(3)
    u_norm = float(np.linalg.norm(u))
    if not math.isfinite(u_norm) or u_norm <= 1e-12:
        raise ValueError("se3_screw axis must be non-zero finite vector")
    u = u / u_norm
    steps = int(n_steps)
    if steps < 1:
        raise ValueError("n_steps must be >= 1")
    pos0, R0 = _coerce_grasp_pose(start_pose)
    piv = pos0.copy() if pivot_m is None else np.asarray(pivot_m, dtype=np.float64).reshape(3)
    tot_ang = float(angle_rad)
    tot_slide = (float(translation_m) if translation_m is not None else float(distance_m)) + float(pitch_m_per_rad) * tot_ang
    out: list[dict] = []
    for idx in range(1, steps + 1):
        s = idx / steps
        d_rot = _fo_so3_exp(u * (s * tot_ang))
        p_s = piv + d_rot @ (pos0 - piv) + (s * tot_slide) * u
        R_s = d_rot @ R0
        out.append(
            dict(
                s=round(s, 4),
                grasp_position_m=_r(p_s, 4),
                angles_deg=_matrix_to_angles_deg(R_s),
            )
        )
    return out


def align_feature_to_target(
    grasp_pos_m,
    grasp_angles_deg,
    feature_point_m,
    feature_axis,
    target_point_m,
    target_axis,
    *,
    feature_secondary_axis=None,
    target_secondary_axis=None,
    roll_about_target_axis_deg: float = 0.0,
    contact_region: str = "pad_center",
    contact_depth_m: float | None = None,
) -> dict:
    """三维刚体手内功能特征精确配准（求解将手内特征点与特征主轴对齐到任意三维倾斜目标点与目标轴所需的夹爪位姿）。

    参数:
        grasp_pos_m: 测量手内特征时的参考夹爪位置 `[x, y, z]`，单位 `m`（`env_origin` 世界系）。
        grasp_angles_deg: 测量手内特征时的参考夹爪姿态角 `[pitch_deg, roll_deg, yaw_deg]`，单位 `deg`。
        feature_point_m: 手内物体功能特征点的世界系三维坐标 `[x, y, z]`，单位 `m`。
        feature_axis: 手内功能特征主轴三维方向向量 `[ax, ay, az]`（无量纲，自动归一化）。
        target_point_m: 目标配准点的世界系三维坐标 `[x, y, z]`，单位 `m`。
        target_axis: 目标配准主轴三维方向向量 `[tx, ty, tz]`（无量纲，支持任意三维倾斜角）。
        feature_secondary_axis: 可选手内副参考轴向量 `[sx, sy, sz]`，用于唯一锁定绕主轴滚转自由度，默认 `None`。
        target_secondary_axis: 可选目标副参考轴向量 `[tsx, tsy, tsz]`，与 `feature_secondary_axis` 配对使用，默认 `None`。
        roll_about_target_axis_deg: 配准后绕单位 `target_axis` 附加的右手滚转微调角，单位 `deg`，默认 `0.0`。
        contact_region: 接触截面名称（`'pad_center'` | `'finger_middle'` | `'fingertip'`），默认 `'pad_center'`。
        contact_depth_m: 可选自定义接触截面深度偏移，单位 `m`（范围 `[-0.0075, 0.0625]`），默认 `None`。

    返回:
        字典 `{target_pos_m, target_grasp_position_m, target_angles_deg, flange_position_m, quaternion_xyzw, rotation_angle_deg, feature_pos_err_m, feature_axis_err_deg}`，
        其中 `target_pos_m` 单位 `m`，`target_angles_deg` 规约至 `[-180,180] x [-90,90] x [-180,180]` `deg`。

    示例:
        `reg = s3.align_feature_to_target([0.0, -0.2, 0.85], [0.0, 0.0, 0.0], [0.04, -0.2, 0.83], [1.0, 0.0, 0.0], [0.16, -0.1, 0.93], [0.6, 0.0, -0.8])`
    """
    return _fo_align_feature(
        grasp_pos_m=grasp_pos_m,
        grasp_angles_deg=grasp_angles_deg,
        feature_point_m=feature_point_m,
        feature_axis=feature_axis,
        target_point_m=target_point_m,
        target_axis=target_axis,
        feature_secondary_axis=feature_secondary_axis,
        target_secondary_axis=target_secondary_axis,
        roll_about_target_axis_deg=roll_about_target_axis_deg,
        contact_region=contact_region,
        contact_depth_m=contact_depth_m,
    )


align_in_hand_feature_to_target = align_feature_to_target
query_spatial_clearance_and_gradient = _fo_query_clearance_grad


def check_reachability(
    arm: str,
    position_m,
    angles_deg=None,
    contact_region: str = "pad_center",
    *,
    quaternion_xyzw=None,
    pose_kind: str = "grasp",
    contact_depth_m: float | None = None,
    q_seed=None,
    robot_model=None,
    robot_state: dict | None = None,
    table_z_m: float | None = None,
    pos_tol_m: float = 0.002,
    rot_tol_rad: float = 0.02,
) -> dict:
    """在沙盒内毫秒级 (`< 1 ms`) 预筛单臂三维目标位姿的 6R 逆运动学可达性、关节余量、单臂自碰与可选支撑面净空（支持对比平行夹爪 `yaw` 与 `yaw +- 180` `deg` 对称等价姿态）。

    参数:
        arm: 机械臂名称（`'left'` | `'right'`）。
        position_m: 目标三维位置 `[x, y, z]`，单位 `m`（`pose_kind='grasp'` 时为夹爪接触点，`'flange'` 时为法兰中心）。
        angles_deg: 夹爪姿态角 `[pitch_deg, roll_deg, yaw_deg]`，单位 `deg`（`pose_kind='grasp'` 时必填）。
        contact_region: 接触截面名称（`'pad_center'` | `'finger_middle'` | `'fingertip'`），默认 `'pad_center'`。
        quaternion_xyzw: 可选法兰单位四元数 `[qx, qy, qz, qw]`（`pose_kind='flange'` 时必填）。
        pose_kind: `'grasp'`（默认）或 `'flange'`。
        contact_depth_m: 可选自定义接触截面深度偏移，单位 `m`，默认 `None`。
        q_seed: 可选 6 轴关节角逆解初值 `[q1..q6]`，单位 `rad`；省略时从 `robot_state` 读取，再无则取关节中点并做均匀确定性流形扰动。
        robot_model: 可选运行时机器人运动学模型对象/字典（默认 `None` 使用 ARX X5 URDF 硬件规格）。
        robot_state: 可选实测双臂状态字典；省略时自动从沙盒 `_spatial_snapshot.json` 预载。
        table_z_m: 可选显式支撑台面高度，单位 `m`；默认 `None`（不做任何隐式桌面高度假设，仅在显式提供时检测台面穿透）。
        pos_tol_m: 可达判定位置容差，单位 `m`，默认 `0.002 m`。
        rot_tol_rad: 可达判定姿态容差，单位 `rad`，默认 `0.02 rad`。

    返回:
        字典 `{reachable, joint_q_rad, pos_err_m, rot_err_rad, min_joint_margin_rad, self_clearance_m, table_clearance_m, reason}`。

    示例:
        `chk = s3.check_reachability('right', [0.18, -0.15, 0.82], [0.0, 0.0, -30.0])`
    """
    return _ik_check_pose(
        arm=arm,
        position_m=position_m,
        angles_deg=angles_deg,
        quaternion_xyzw=quaternion_xyzw,
        pose_kind=pose_kind,
        contact_region=contact_region,
        contact_depth_m=contact_depth_m,
        q_seed=q_seed,
        robot_model=robot_model,
        robot_state=robot_state,
        table_z_m=table_z_m,
        pos_tol_m=pos_tol_m,
        rot_tol_rad=rot_tol_rad,
    )


check_pose_reachability = check_reachability


def prefilter_stages(
    stages: list[dict],
    initial_robot_state: dict | None = None,
    *,
    robot_model=None,
    table_z_m: float | None = None,
    pos_tol_m: float = 0.002,
    rot_tol_rad: float = 0.02,
    bimanual_min_clearance_m: float = 0.0,
    batch_joint_travel_preset: str | None = None,
) -> dict:
    """在沙盒内毫秒级 (`< 3 ms`) 预筛整批 `robot_action` `stages` 的双臂 6R 运动学可达性、关节行程准入与自碰撞/互碰撞风险。

    参数:
        stages: 标准 `robot_action` 阶段字典列表（支持 `move`、`gripper`、`observe` 及跨臂 `depends_on`）。
        initial_robot_state: 可选实测双臂起始状态字典（含 `q_rad` `rad` 与 `ee_pose` `m`）；省略时从沙盒快照自动预载。
        robot_model: 可选运行时机器人运动学模型对象/字典（默认 `None` 使用 ARX X5 URDF 硬件规格）。
        table_z_m: 可选显式支撑台面高度，单位 `m`；默认 `None`（仅显式提供时检测支撑面穿透）。
        pos_tol_m: 逆解位置容差，单位 `m`，默认 `0.002 m`。
        rot_tol_rad: 逆解姿态容差，单位 `rad`，默认 `0.02 rad`。
        bimanual_min_clearance_m: 双臂连杆胶囊包络最小允许净空阈值，单位 `m`，默认 `0.0 m`。
        batch_joint_travel_preset: 批次默认关节行程预设（`'standard'` | `'extended'` | `'unconstrained'`），默认 `None`。

    返回:
        字典 `{valid, blocking_stage_ids, reasons, min_bimanual_clearance_m, min_self_clearance_m, min_table_clearance_m, max_joint_step_travel_rad, stage_reports, terminal_joint_q_rad}`。

    示例:
        `pf = s3.prefilter_stages(stages)`
        `if pf['valid']: client.action(stages=stages, mode='execute_if_clean')`
    """
    return _ik_prefilter_stages(
        stages=stages,
        initial_robot_state=initial_robot_state,
        robot_model=robot_model,
        table_z_m=table_z_m,
        pos_tol_m=pos_tol_m,
        rot_tol_rad=rot_tol_rad,
        bimanual_min_clearance_m=bimanual_min_clearance_m,
        batch_joint_travel_preset=batch_joint_travel_preset,
    )


prefilter_stages_kinematics = prefilter_stages


def jaw_symmetric_angles_deg(angles_deg) -> list[float]:
    """求解任意三维夹爪姿态角 `angles_deg` 精确的 $SO(3)$ 平行夹爪对称姿态（绕接近轴 `+X_local` 滚转 `180 deg`，保持接近向量与夹持闭合线不变）。

    参数:
        angles_deg: 夹爪姿态角 `[pitch_deg, roll_deg, yaw_deg]`，单位 `deg`。

    返回:
        规约至 `[-180, 180] x [-90, 90] x [-180, 180]` 的等价姿态角列表 `[pitch_deg, roll_deg, yaw_deg]`，单位 `deg`。

    示例:
        `twin = s3.jaw_symmetric_angles_deg([90.0, 0.0, 0.0])  # -> [-90.0, 0.0, -180.0]`
    """
    return _ik_jaw_symmetric(angles_deg)


def solve_reachable_pose(
    arm: str,
    position_m,
    angles_deg,
    contact_region: str = "pad_center",
    *,
    include_jaw_symmetry: bool = True,
    approach_tilt_offsets_deg: tuple[float, ...] | list[float] = (0.0, -20.0, 20.0, -35.0, 35.0, -50.0, 50.0, -65.0, 65.0),
    yaw_offsets_deg: tuple[float, ...] | list[float] = (0.0,),
    candidate_angles_deg=None,
    contact_depth_m: float | None = None,
    q_seed=None,
    robot_model=None,
    robot_state: dict | None = None,
    table_z_m: float | None = None,
    pos_tol_m: float = 0.002,
    rot_tol_rad: float = 0.02,
    joint_travel_preset: str | None = "standard",
) -> dict:
    """在固定三维抓取点 `position_m` 处毫秒级搜索无自碰撞、关节余量与行程合规的最优可达 6R 姿态（涵盖 $SO(3)$ 平行夹爪对称解与绕闭合轴接近角倾角）。

    参数:
        arm: 机械臂名称（`'left'` | `'right'`）。
        position_m: 目标抓取接触点三维坐标 `[x, y, z]`，单位 `m`（`env_origin` 世界系）。
        angles_deg: 参考夹爪姿态角 `[pitch_deg, roll_deg, yaw_deg]`，单位 `deg`。
        contact_region: 接触截面名称（`'pad_center'` | `'finger_middle'` | `'fingertip'`），默认 `'pad_center'`。
        include_jaw_symmetry: 是否自动评估绕接近轴 `180 deg` 翻转的对称等价姿态，默认 `True`。
        approach_tilt_offsets_deg: 绕指间闭合轴的接近角倾角候选，单位 `deg`，默认 `(0.0, -20.0, 20.0, -35.0, 35.0, -50.0, 50.0, -65.0, 65.0)`。
        yaw_offsets_deg: 绕世界竖直轴 `Z` 的偏航微调候选，单位 `deg`，默认 `(0.0,)`。
        candidate_angles_deg: 可选显式附加候选姿态列表 `[[p, r, y], ...]`，单位 `deg`，默认 `None`。
        contact_depth_m: 可选自定义接触截面深度偏移，单位 `m`，默认 `None`。
        q_seed: 可选起始 6 轴关节角 `[q1..q6]`，单位 `rad`；省略时从 `robot_state` 或工作区快照读取。
        robot_model: 可选运行时机器人运动学模型对象/字典，默认 `None`。
        robot_state: 可选实测双臂状态字典；省略时自动从沙盒 `_spatial_snapshot.json` 预载。
        table_z_m: 可选显式支撑台面高度，单位 `m`，默认 `None`。
        pos_tol_m: 逆解位置容差，单位 `m`，默认 `0.002 m`。
        rot_tol_rad: 逆解姿态容差，单位 `rad`，默认 `0.02 rad`。
        joint_travel_preset: 关节行程预设（`'standard'` | `'extended'` | `'unconstrained'`），默认 `'standard'`。

    返回:
        字典 `{reachable, selected_angles_deg, is_jaw_symmetric_twin, approach_tilt_deg, yaw_offset_deg, joint_q_rad, pos_err_m, rot_err_rad, min_joint_margin_rad, self_clearance_m, table_clearance_m, max_joint_travel_rad, joint_travel_admissible, recommended_joint_travel_preset, jaw_symmetric_twin_angles_deg, candidates, reason}`。

    示例:
        `sol = s3.solve_reachable_pose('left', [-0.145, -0.056, 0.90], [0.0, 0.0, -36.0])`
    """
    return _ik_solve_reachable(
        arm=arm,
        position_m=position_m,
        angles_deg=angles_deg,
        contact_region=contact_region,
        include_jaw_symmetry=include_jaw_symmetry,
        approach_tilt_offsets_deg=approach_tilt_offsets_deg,
        yaw_offsets_deg=yaw_offsets_deg,
        candidate_angles_deg=candidate_angles_deg,
        contact_depth_m=contact_depth_m,
        q_seed=q_seed,
        robot_model=robot_model,
        robot_state=robot_state,
        table_z_m=table_z_m,
        pos_tol_m=pos_tol_m,
        rot_tol_rad=rot_tol_rad,
        joint_travel_preset=joint_travel_preset,
    )


def check_source_to_dest(
    source_pos_m,
    source_angles_deg,
    dest_pos_m,
    dest_angles_deg=None,
    contact_region: str = "pad_center",
    *,
    dest_contact_region: str | None = None,
    arms: tuple[str, ...] | list[str] = ("left", "right"),
    include_jaw_symmetry: bool = True,
    dest_include_jaw_symmetry: bool | None = None,
    approach_tilt_offsets_deg: tuple[float, ...] | list[float] = (0.0, -20.0, 20.0, -35.0, 35.0, -50.0, 50.0, -65.0, 65.0),
    yaw_offsets_deg: tuple[float, ...] | list[float] = (0.0,),
    contact_depth_m: float | None = None,
    dest_contact_depth_m: float | None = None,
    robot_model=None,
    robot_state: dict | None = None,
    table_z_m: float | None = None,
    pos_tol_m: float = 0.002,
    rot_tol_rad: float = 0.02,
    joint_travel_preset: str | None = "standard",
) -> dict:
    """在执行物理抓取前毫秒级 (`< 3 ms`) 评估双臂中是否存在单臂可同时直达源抓取点 `source_pos_m` 与目标放置/插入点 `dest_pos_m`（含 $SO(3)$ 平行夹爪对称解与源到目标真实关节行程 `rad`）。

    参数:
        source_pos_m: 源抓取接触点三维世界坐标 `[x, y, z]`，单位 `m`（`env_origin` 世界系）。
        source_angles_deg: 源抓取参考夹爪姿态角 `[pitch_deg, roll_deg, yaw_deg]`，单位 `deg`。
        dest_pos_m: 目标放置/插入接触点三维世界坐标 `[x, y, z]`，单位 `m`（`env_origin` 世界系）。
        dest_angles_deg: 可选目标参考姿态角 `[pitch_deg, roll_deg, yaw_deg]`，单位 `deg`；省略时沿用 `source_angles_deg`。
        contact_region: 源点接触截面（`'pad_center'` | `'finger_middle'` | `'fingertip'`），默认 `'pad_center'`。
        dest_contact_region: 可选目标点接触截面；省略时沿用 `contact_region`。
        arms: 待核验机械臂列表，默认 `('left', 'right')`。
        include_jaw_symmetry: 源点是否自动评估绕接近轴 `180 deg` 对称等价姿态，默认 `True`。
        dest_include_jaw_symmetry: 目标点是否评估平行夹爪对称等价姿态；省略时沿用 `include_jaw_symmetry`。
        approach_tilt_offsets_deg: 绕指间闭合轴的接近角倾角候选，单位 `deg`，默认 `(0.0, -20.0, 20.0, -35.0, 35.0, -50.0, 50.0, -65.0, 65.0)`。
        yaw_offsets_deg: 绕世界竖直轴 `Z` 的偏航微调候选，单位 `deg`，默认 `(0.0,)`。
        contact_depth_m: 可选源点自定义接触截面深度偏移，单位 `m`，默认 `None`。
        dest_contact_depth_m: 可选目标点自定义接触截面深度偏移，单位 `m`，默认 `None`。
        robot_model: 可选运行时机器人运动学模型对象/字典，默认 `None`。
        robot_state: 可选实测双臂状态字典；省略时自动从沙盒 `_spatial_snapshot.json` 预载。
        table_z_m: 可选显式支撑台面高度，单位 `m`，默认 `None`。
        pos_tol_m: 逆解位置容差，单位 `m`，默认 `0.002 m`。
        rot_tol_rad: 逆解姿态容差，单位 `rad`，默认 `0.02 rad`。
        joint_travel_preset: 关节行程预设（`'standard'` | `'extended'` | `'unconstrained'`），默认 `'standard'`。

    返回:
        字典 `{single_arm_candidates, recommended_arm, requires_handoff, source_reachable_arms, dest_reachable_arms, arms}`。

    示例:
        `route = s3.check_source_to_dest([-0.13, -0.11, 0.87], [0.0, 0.0, -90.0], [0.16, -0.14, 0.96], [0.0, 0.0, 90.0])`
    """
    return _ik_check_source_to_dest(
        source_pos_m=source_pos_m,
        source_angles_deg=source_angles_deg,
        dest_pos_m=dest_pos_m,
        dest_angles_deg=dest_angles_deg,
        contact_region=contact_region,
        dest_contact_region=dest_contact_region,
        arms=arms,
        include_jaw_symmetry=include_jaw_symmetry,
        dest_include_jaw_symmetry=dest_include_jaw_symmetry,
        approach_tilt_offsets_deg=approach_tilt_offsets_deg,
        yaw_offsets_deg=yaw_offsets_deg,
        contact_depth_m=contact_depth_m,
        dest_contact_depth_m=dest_contact_depth_m,
        robot_model=robot_model,
        robot_state=robot_state,
        table_z_m=table_z_m,
        pos_tol_m=pos_tol_m,
        rot_tol_rad=rot_tol_rad,
        joint_travel_preset=joint_travel_preset,
    )


def plan_joint_safe_return(
    target_robot_state: dict,
    current_robot_state: dict | None = None,
    *,
    arms: tuple[str, ...] | list[str] = ("left", "right"),
    stage_prefix: str = "return_home",
    intermediate_z_offset_m: float = 0.08,
    bimanual_min_clearance_m: float = 0.02,
    robot_model=None,
    table_z_m: float | None = None,
) -> dict:
    """生成经 6R 逆解、关节行程缠绕准入与双臂胶囊互碰核验的安全复位 `stages` 序列（按需自动设置 `'extended'`、插入抬升中点或通过 `depends_on` 串行化双臂）。

    参数:
        target_robot_state: 目标复位状态字典（支持初始观测 `obs_1`、`obs_1['robot_state']` 或 `{arm: {'position_m', 'quaternion_xyzw'}}`，位置单位 `m`）。
        current_robot_state: 可选当前实测状态字典；省略时自动从沙盒 `_spatial_snapshot.json` 预载。
        arms: 需要复位的机械臂列表，默认 `('left', 'right')`。
        stage_prefix: 生成阶段 `id` 命名前缀，默认 `'return_home'`。
        intermediate_z_offset_m: 单段直返受阻时自动插入过渡中点的抬升高度，单位 `m`，默认 `0.08 m`。
        bimanual_min_clearance_m: 双臂同步回位允许的最小连杆胶囊净空，单位 `m`，默认 `0.02 m`。
        robot_model: 可选运行时机器人运动学模型对象/字典，默认 `None`。
        table_z_m: 可选显式支撑台面高度，单位 `m`，默认 `None`。

    返回:
        字典 `{valid, stages, sequenced_bimanual, recommended_joint_travel_preset, prefilter_report}`。

    示例:
        `ret = s3.plan_joint_safe_return(obs_1['robot_state'])`
    """
    return _ik_plan_return(
        target_robot_state=target_robot_state,
        current_robot_state=current_robot_state,
        arms=arms,
        stage_prefix=stage_prefix,
        intermediate_z_offset_m=intermediate_z_offset_m,
        bimanual_min_clearance_m=bimanual_min_clearance_m,
        robot_model=robot_model,
        table_z_m=table_z_m,
    )
