"""会话级持久化场景障碍物登记表、释放位置自动包络记录与水平跨障安全抬升规划。

Embodiment (design 13 section 5.3, finding C4): obstacles reach a backend as ``review_proxies``. The per-episode
registry stores and replays them only for an embodiment that honours review proxies
(``"review_proxies" in EmbodimentProfile.supports``, the run's ``harness.embodiment``; default robot ARX-X5). On any
other robot every request is stateless: its own ``obstacles`` / ``obstacle_object_ids`` still reach the backend (which
refuses them honestly), and a later request without obstacle arguments carries none. The grasp centre of a commanded
TCP pose is the profile's ``gripper`` reference (``grasp_reference_offset_m`` along ``approach_axis_local``; ARX-X5
0.1501 m along link6 +x) and the table fallback height is ``support_surface_z_m``.

Zero upward imports: this module depends only on ``ni_rua.core`` and ``ni_rua.robot``. Callers in ``ni_rua.runtime``
pass ``table_resolver`` via ``ni_rua.runtime.actions.obstacle_registry`` (or attach ``_obs3d_memo`` / a table dict).
"""
from copy import deepcopy
import math

import numpy as np

from ...core.config import motion_thresholds
from ...core.contracts import IDENTITY
from ...core.embodiment import EmbodimentProfile, default_embodiment
from .proxy_contract import validate_boxes
from ..model.roboprobe_pose import quat_to_matrix

_BASE_THRESHOLDS = motion_thresholds()
MAX_PERSISTENT_OBSTACLES = 32


def _embodiment(harness):
    """The run's EmbodimentProfile (``harness.embodiment``), else the default robot ARX-X5 (zero perception imports)."""
    if isinstance(harness, EmbodimentProfile):
        return harness
    profile = getattr(harness, 'embodiment', None)
    if isinstance(profile, EmbodimentProfile):
        return profile
    # Ensure default embodiment is registered if called in a fresh worker process before embodiments loaded.
    try:
        return default_embodiment()
    except RuntimeError:
        from ..model import embodiments  # noqa: F401
        return default_embodiment()


def _resolve_attached_table(harness_or_table, observation_id, table_resolver=None):
    """Resolve an ``object_table`` dict without importing ``ni_rua.perception``."""
    if isinstance(harness_or_table, dict) and ('rows' in harness_or_table or 'columns' in harness_or_table):
        return harness_or_table
    if callable(table_resolver):
        return table_resolver(harness_or_table, observation_id)
    memo = getattr(harness_or_table, '_obs3d_memo', None)
    if isinstance(memo, tuple) and len(memo) == 2 and memo[0] == observation_id and isinstance(memo[1], dict):
        entry = memo[1].get('object_table')
        return entry[0] if entry else None
    history = getattr(harness_or_table, '_obs3d_history', None)
    if isinstance(history, dict) and observation_id in history and isinstance(history[observation_id], dict):
        entry = history[observation_id].get('object_table')
        return entry[0] if entry else None
    return None


def _registry(harness):
    """获取当前 episode 绑定的持久化障碍物字典；环境切换时自动重置。"""
    ident = getattr(harness, '_persistent_obstacles_identity', None)
    current_ident = {k: harness.identity.get(k) for k in IDENTITY}
    if ident != current_ident or not hasattr(harness, '_persistent_obstacles'):
        harness._persistent_obstacles_identity = current_ident
        harness._persistent_obstacles = {}
    return harness._persistent_obstacles


def _extract_stage_grasp_xy(stage, gripper=None):
    """提取阶段声明的目标水平坐标 (x, y)，用于排除目标点自身所在位置的自动放置包络。

    ``gripper``: the embodiment's GripperProfile (None = the default ARX-X5 robot) for the grasp centre of a
    commanded TCP pose (``_stage_grasp_center_m``)."""
    if not isinstance(stage, dict) or stage.get('kind') != 'move':
        return None
    for key in ('grasp_position_m', 'position_m'):
        pos = stage.get(key)
        if isinstance(pos, list) and len(pos) == 3:
            return float(pos[0]), float(pos[1])
    gp = stage.get('grasp_pose') or (stage.get('action') or {}).get('input_grasp_pose')
    if isinstance(gp, dict) and isinstance(gp.get('position_m'), list) and len(gp['position_m']) == 3:
        return float(gp['position_m'][0]), float(gp['position_m'][1])
    center = _stage_grasp_center_m(stage, gripper)
    if center is not None:
        return float(center[0]), float(center[1])
    arm = stage.get('arm')
    wps = stage.get('waypoints') or ((stage.get('action') or {}).get('path') or {}).get('waypoints') or []
    if wps and arm and isinstance(wps[-1], dict):
        pose = wps[-1].get(arm) if arm in wps[-1] else wps[-1]
        if isinstance(pose, dict) and isinstance(pose.get('position_m'), list) and len(pose['position_m']) == 3:
            return float(pose['position_m'][0]), float(pose['position_m'][1])
    return None


# 目标点自动障碍物豁免半径预设（默认 normal=5.5 cm，支持 GPT 显式选预设或自由给值）
DEFAULT_TARGET_EXEMPTION_RADIUS_M = _BASE_THRESHOLDS['target_exemption_radius_m']
TARGET_EXEMPTION_PRESETS = {
    'none': 0.0,
    'tight': 0.030,
    'normal': DEFAULT_TARGET_EXEMPTION_RADIUS_M,
    'wide': 0.080,
}

# 水平跨障安全抬升高度预设（默认 normal=0.13 m，与方向二 v2 一致；支持 GPT 显式选预设或自由给值）
DEFAULT_CLEARANCE_HEIGHT_M = _BASE_THRESHOLDS['clearance_height_m']
CLEARANCE_PRESETS = {
    'none': 0.0,
    'low': 0.09,
    'normal': DEFAULT_CLEARANCE_HEIGHT_M,
    'high': 0.18,
}

OBSTACLE_CLEARANCE_FIELDS = (
    'obstacle_object_ids',
    'obstacle_policy',
    'ignore_obstacle_ids',
    'target_exemption_preset',
    'target_exemption_radius_m',
    'clearance_preset',
    'clearance_height_m',
)
STAGE_CLEARANCE_FIELDS = (
    'clearance_preset',
    'clearance_height_m',
)


def resolve_target_exemption_radius(args):
    """解析目标交互点自动障碍物豁免半径：支持预设（none/tight/normal/wide）或自由数值 target_exemption_radius_m。"""
    if 'target_exemption_preset' in args and 'target_exemption_radius_m' in args:
        raise ValueError('conflicting_target_exemption_preset_and_target_exemption_radius_m')
    preset = args.get('target_exemption_preset')
    if preset is not None:
        if preset not in TARGET_EXEMPTION_PRESETS:
            raise ValueError('invalid_target_exemption_preset')
        radius = TARGET_EXEMPTION_PRESETS[preset]
    else:
        radius = args.get('target_exemption_radius_m', DEFAULT_TARGET_EXEMPTION_RADIUS_M)
    if type(radius) not in (int, float) or not 0.0 <= float(radius) <= 0.15:
        raise ValueError('invalid_target_exemption_radius_m')
    return float(radius)


def resolve_stage_clearance(fields, batch_default_m=DEFAULT_CLEARANCE_HEIGHT_M):
    """解析跨障抬升安全高度：支持预设（none/low/normal/high）或自由数值 clearance_height_m。"""
    if 'clearance_preset' in fields and 'clearance_height_m' in fields:
        raise ValueError('conflicting_clearance_preset_and_clearance_height_m')
    preset = fields.pop('clearance_preset', None)
    if preset is not None:
        if preset not in CLEARANCE_PRESETS:
            raise ValueError('invalid_clearance_preset')
        clearance_m = CLEARANCE_PRESETS[preset]
    else:
        clearance_m = fields.pop('clearance_height_m', batch_default_m)
    if type(clearance_m) not in (int, float) or not 0.0 <= float(clearance_m) <= 0.25:
        raise ValueError('invalid_clearance_height_m')
    return float(clearance_m)


#: Size rules of an object_table row turned into an obstacle box (``obstacles_from_object_table``), metres: a row
#: without length / width uses ROW_DEFAULT_FOOTPRINT_M, a shorter one is raised to ROW_MIN_FOOTPRINT_M; each half extent
#: of the yawed footprint along base x / y is at least ROW_MIN_HALF_EXTENT_M; the box is at least ROW_MIN_HEIGHT_M
#: tall (its bottom is lowered); a row without top_z gets centre z + ROW_DEFAULT_TOP_ABOVE_CENTRE_M and one without
#: bottom_z the table height, else top - ROW_MIN_HEIGHT_M; every box has margin_m ROW_OBSTACLE_MARGIN_M. The SO-101
#: field help publishes them and its preview reports each one applied (``path_check.object_row_boxes``).
ROW_DEFAULT_FOOTPRINT_M = 0.03
ROW_MIN_FOOTPRINT_M = 0.015
ROW_MIN_HALF_EXTENT_M = 0.008
ROW_MIN_HEIGHT_M = 0.010
ROW_DEFAULT_TOP_ABOVE_CENTRE_M = 0.02
ROW_OBSTACLE_MARGIN_M = 0.005


def _note_skip(report, ids, reason):
    """Add ``ids`` to ``report["skipped"]`` with ``reason``; nothing (no empty list) when there are none."""
    if report is not None and ids:
        report.setdefault('skipped', []).extend({'id': oid, 'reason': reason} for oid in ids)


def obstacles_from_object_table(
    harness,
    observation_id,
    object_ids=None,
    *,
    min_height_above_table_m=0.020,
    table_resolver=None,
    report=None,
):
    """将缓存的 3D object_table 行转换为世界坐标系 AABB 障碍物包络（支持指定 object_ids 或自动筛选高置信度物体）。

    ``report`` (dict or None): filled with what the conversion of the requested ``object_ids`` did besides copying
    a row: ``skipped`` ``[{"id", "reason"}]`` and ``size_floors`` ``[{"id", "field", "row_value", "used_value"}]`` (m).
    """
    wanted_ids = list(dict.fromkeys(object_ids)) if object_ids is not None else []
    if not observation_id:
        _note_skip(report, wanted_ids, 'no_object_table')
        return []
    table = _resolve_attached_table(harness, observation_id, table_resolver=table_resolver)
    if not isinstance(table, dict) or not table.get('rows'):
        _note_skip(report, wanted_ids, 'no_object_table')
        return []
    columns = list(table.get('columns') or [])
    # the table's support height, else the profile's support surface (X5 table top 0.765 m; None = unknown). A table
    # height of exactly 0 m is a height (SO-101: so101_base sits about at the table plane), not a missing value.
    table_z = table.get('table_z')
    if not isinstance(table_z, (int, float)) or isinstance(table_z, bool):
        table_z = _embodiment(harness).support_surface_z_m
    if table_z is None and object_ids is None:
        return []   # the automatic selection needs the support height to drop flat rows
    table_z = None if table_z is None else float(table_z)
    wanted = set(object_ids) if object_ids is not None else None
    boxes, seen, floors = [], set(), []
    for raw in table.get('rows') or []:
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            continue
        row = dict(zip(columns, raw))
        oid = row.get('id')
        if not isinstance(oid, str):
            continue
        if wanted is not None and oid not in wanted:
            continue
        quality = str(row.get('quality') or 'ok')
        if str(row.get('support') or '').startswith('held_') or quality == 'held_fk_depth':
            if wanted is not None and oid not in seen:
                _note_skip(report, [oid], 'held_row')
            seen.add(oid)
            continue
        if wanted is None and not quality.startswith('ok'):
            continue
        centre = row.get('centre_m')
        if not isinstance(centre, list) or len(centre) != 3:
            if wanted is not None and oid not in seen:
                _note_skip(report, [oid], 'no_centre')
            seen.add(oid)
            continue
        cx, cy, cz = float(centre[0]), float(centre[1]), float(centre[2])
        top_z = float(row['top_z']) if isinstance(row.get('top_z'), (int, float)) else cz + ROW_DEFAULT_TOP_ABOVE_CENTRE_M
        given_bottom = float(row['bottom_z']) if isinstance(row.get('bottom_z'), (int, float)) else None
        bottom_z = (given_bottom if given_bottom is not None
                    else table_z if table_z is not None else top_z - ROW_MIN_HEIGHT_M)
        if wanted is None and (top_z - table_z) < min_height_above_table_m:
            continue
        length = max(float(row.get('length') or ROW_DEFAULT_FOOTPRINT_M), ROW_MIN_FOOTPRINT_M)
        width = max(float(row.get('width') or ROW_DEFAULT_FOOTPRINT_M), ROW_MIN_FOOTPRINT_M)
        yaw_rad = math.radians(float(row.get('yaw_deg') or 0.0))
        half_x = 0.5 * abs(length * math.cos(yaw_rad)) + 0.5 * abs(width * math.sin(yaw_rad))
        half_y = 0.5 * abs(length * math.sin(yaw_rad)) + 0.5 * abs(width * math.cos(yaw_rad))
        dx = max(half_x, ROW_MIN_HALF_EXTENT_M)
        dy = max(half_y, ROW_MIN_HALF_EXTENT_M)
        z_lo = min(bottom_z, top_z - ROW_MIN_HEIGHT_M)
        z_hi = max(top_z, z_lo + ROW_MIN_HEIGHT_M)
        if wanted is not None and report is not None:
            given = lambda key: float(row[key]) if isinstance(row.get(key), (int, float)) and row[key] else None  # noqa: E731
            for field, before, after in (
                    ('length_m', given('length'), length), ('width_m', given('width'), width),
                    ('half_extent_x_m', half_x, dx), ('half_extent_y_m', half_y, dy),
                    ('top_z_m', float(row['top_z']) if isinstance(row.get('top_z'), (int, float)) else None, z_hi),
                    ('bottom_z_m', given_bottom, z_lo)):
                if before is None or round(before, 4) != round(after, 4):
                    floors.append({'id': oid, 'field': field,
                                   'row_value': None if before is None else round(before, 4),
                                   'used_value': round(after, 4)})
        seen.add(oid)
        box_id = oid if wanted is not None else f'vggt_auto_{oid}'
        boxes.append(
            dict(
                id=box_id,
                min_m=[round(cx - dx, 4), round(cy - dy, 4), round(z_lo, 4)],
                max_m=[round(cx + dx, 4), round(cy + dy, 4), round(z_hi, 4)],
                margin_m=ROW_OBSTACLE_MARGIN_M,
                evidence=f'vggt_object_table:{observation_id}:{oid}',
            )
        )
    _note_skip(report, [oid for oid in wanted_ids if oid not in seen], 'not_in_table')
    if report is not None and floors:
        report.setdefault('size_floors', []).extend(floors)
    return boxes


def merge_and_persist_obstacles(harness, args, compiled_stages=None, *, table_resolver=None, report=None):
    """按 obstacle_policy 管理会话级障碍物并合并历史包络；支持 ignore_obstacle_ids 与逐阶段目标豁免半径。

    Registry gate (C4): only an embodiment with ``review_proxies`` in ``supports`` stores (``persist``) or replays
    obstacles. Otherwise the request is handled as ``stateless`` whatever ``obstacle_policy`` says: this request's
    own obstacles still go to the backend (which refuses review proxies it cannot honour) and nothing is kept, so a
    later request without obstacle arguments is not refused for them. ``report`` (dict or None): receives the
    ``obstacle_object_ids`` conversion report of ``obstacles_from_object_table``; the returned args are the same."""
    policy = args.get('obstacle_policy', 'persist')
    if policy not in ('persist', 'stateless', 'clear'):
        raise ValueError('invalid_obstacle_policy')
    profile = _embodiment(harness)
    if 'review_proxies' not in profile.supports:
        policy = 'stateless'
    reg = _registry(harness)
    if policy == 'clear':
        reg.clear()
    exemption_r = resolve_target_exemption_radius(args)
    if compiled_stages and exemption_r != DEFAULT_TARGET_EXEMPTION_RADIUS_M:
        for stage in compiled_stages:
            if isinstance(stage, dict) and stage.get('kind') == 'move':
                path = ((stage.get('action') or {}).get('path'))
                if isinstance(path, dict):
                    path['target_exemption_radius_m'] = exemption_r
    supplied = list(args.get('obstacles') or [])
    obs_id = args.get('observation_id')
    if isinstance(obs_id, str) and obs_id and policy != 'clear':
        # 逐观测 VGGT 物体表 ID 与包络仅对其所属 observation_id 有效，推进到新观测后自动清理过期 VGGT 实心包络（保留显式世界系 obstacles）
        stale_vggt_keys = [
            k for k, box in reg.items()
            if isinstance(box, dict)
            and str(box.get('evidence', '')).startswith('vggt_object_table:')
            and not str(box.get('evidence', '')).startswith(f'vggt_object_table:{obs_id}:')
        ]
        for k in stale_vggt_keys:
            reg.pop(k, None)
    explicit_ids = args.get('obstacle_object_ids')
    if explicit_ids:
        supplied.extend(
            obstacles_from_object_table(
                harness, obs_id, object_ids=explicit_ids, table_resolver=table_resolver, report=report
            )
        )
    validated_supplied = []
    if supplied:
        validated = validate_boxes(supplied)
        for box in validated:
            clean_box = {k: deepcopy(v) for k, v in box.items() if k != 'source_kind'}
            validated_supplied.append(clean_box)
            if policy != 'stateless':
                reg[box['id']] = clean_box
        if policy != 'stateless':
            while len(reg) > MAX_PERSISTENT_OBSTACLES:
                reg.pop(next(iter(reg)))
    active_source = {box['id']: box for box in validated_supplied} if policy == 'stateless' else reg
    if not active_source:
        return args
    ignored = set(args.get('ignore_obstacle_ids') or [])
    raw_stages = args.get('stages') or []
    target_xys = [xy for s in raw_stages if (xy := _extract_stage_grasp_xy(s, profile.gripper)) is not None]
    if not target_xys and compiled_stages:
        target_xys = [xy for s in compiled_stages if (xy := _extract_stage_grasp_xy(s, profile.gripper)) is not None]
    merged_by_id = {}
    for k, v in active_source.items():
        if k in ignored:
            continue
        is_auto = str(k).startswith('placed_') or str(k).startswith('vggt_auto_')
        if is_auto and target_xys and exemption_r > 0.0:
            low, high = v.get('min_m', [0, 0, 0]), v.get('max_m', [0, 0, 0])
            if all(low[0] - exemption_r <= tx <= high[0] + exemption_r and low[1] - exemption_r <= ty <= high[1] + exemption_r for tx, ty in target_xys):
                continue
        merged_by_id[k] = deepcopy(v)
    if validated_supplied:
        for box in validated_supplied:
            if isinstance(box, dict) and isinstance(box.get('id'), str) and box['id'] not in ignored:
                merged_by_id[box['id']] = deepcopy(box)
    if not merged_by_id:
        if 'obstacles' in args:
            result = dict(args)
            result.pop('obstacles', None)
            return result
        return args
    merged = list(merged_by_id.values())[-MAX_PERSISTENT_OBSTACLES:]
    result = dict(args)
    result['obstacles'] = merged
    return result


def _stage_grasp_center_m(stage, gripper=None):
    """从移动阶段的 input_grasp_pose 或末端 waypoint 推算抓取中心 3D 坐标。

    The waypoint form needs a fixed TCP offset: ``gripper`` (GripperProfile; None = the default ARX-X5 robot) with
    ``grasp_reference == "tcp_offset"`` gives TCP + grasp_reference_offset_m * R @ approach_axis_local (X5 0.1501 m
    along link6 +x, design 13 V-C); any other reference (the SO-101 jaw midpoint depends on the jaw angle) gives
    None, and callers use the commanded TCP position."""
    own = (stage or {}).get('action') or {}
    gp = own.get('input_grasp_pose')
    if isinstance(gp, dict) and isinstance(gp.get('position_m'), list) and len(gp['position_m']) == 3:
        return np.asarray(gp['position_m'], dtype=float)
    if gripper is None:
        gripper = _embodiment(None).gripper
    if gripper.grasp_reference != 'tcp_offset':
        return None
    arm = (stage or {}).get('arm')
    wps = (own.get('path') or {}).get('waypoints') or []
    if wps and arm and isinstance(wps[-1].get(arm), dict):
        pose = wps[-1][arm]
        if 'position_m' in pose and 'quaternion_xyzw' in pose:
            origin = np.asarray(pose['position_m'], dtype=float)
            xyzw = np.asarray(pose['quaternion_xyzw'], dtype=float)
            norm = float(np.linalg.norm(xyzw))
            if norm >= 1e-6:
                xyzw = xyzw / norm
                rot = quat_to_matrix(np.r_[xyzw[3], xyzw[:3]])
                approach = rot @ np.asarray(gripper.approach_axis_local, dtype=float)
                return origin + gripper.grasp_reference_offset_m * approach
    return None


def record_placed_obstacles(harness, action, receipt):
    """不自动凭空生成 placed_* 虚拟障碍物包络；障碍物列表完全由模型通过 obstacles 参数显式声明与管理。"""
    return


def _segment_xy_distance_to_box(p0_xy, p1_xy, low_xy, high_xy):
    """计算 2D 线段 (p0_xy -> p1_xy) 到轴对齐矩形 [low_xy, high_xy] 的精确最小水平距离。"""
    p0 = np.asarray(p0_xy, dtype=float)
    p1 = np.asarray(p1_xy, dtype=float)
    lo = np.asarray(low_xy, dtype=float)
    hi = np.asarray(high_xy, dtype=float)
    ext = np.maximum(hi - lo, 0.0)
    ctr = 0.5 * (lo + hi)
    try:
        import coal
    except ImportError:
        bmin = ctr - 0.5 * ext
        bmax = ctr + 0.5 * ext
        d = p1 - p0
        t_min, t_max = 0.0, 1.0
        hit = True
        for i in range(2):
            if abs(float(d[i])) <= 1e-15:
                if float(p0[i]) < float(bmin[i]) or float(p0[i]) > float(bmax[i]):
                    hit = False
                    break
            else:
                inv_d = 1.0 / float(d[i])
                t1 = (float(bmin[i]) - float(p0[i])) * inv_d
                t2 = (float(bmax[i]) - float(p0[i])) * inv_d
                if t1 > t2:
                    t1, t2 = t2, t1
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                if t_min > t_max:
                    hit = False
                    break
        if hit:
            return 0.0
        best_sq = float("inf")
        for pt in (p0, p1):
            dx = max(float(bmin[0] - pt[0]), 0.0, float(pt[0] - bmax[0]))
            dy = max(float(bmin[1] - pt[1]), 0.0, float(pt[1] - bmax[1]))
            best_sq = min(best_sq, dx * dx + dy * dy)
        dd = float(d[0] * d[0] + d[1] * d[1])
        if dd > 1e-30:
            for cx in (float(bmin[0]), float(bmax[0])):
                for cy in (float(bmin[1]), float(bmax[1])):
                    t = min(1.0, max(0.0, ((cx - float(p0[0])) * float(d[0]) + (cy - float(p0[1])) * float(d[1])) / dd))
                    rx = float(p0[0]) + t * float(d[0]) - cx
                    ry = float(p0[1]) + t * float(d[1]) - cy
                    best_sq = min(best_sq, rx * rx + ry * ry)
        return math.sqrt(max(0.0, best_sq))

    delta = p1 - p0
    length = float(math.hypot(float(delta[0]), float(delta[1])))
    mid = 0.5 * (p0 + p1)
    if length > 1e-12:
        c, s = float(delta[0]) / length, float(delta[1]) / length
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    else:
        rot = np.eye(3, dtype=float)
    seg_geom = coal.Box(length, 0.0, 0.0)
    seg_tf = coal.Transform3s(rot, np.array([float(mid[0]), float(mid[1]), 0.0], dtype=float))
    box_geom = coal.Box(float(ext[0]), float(ext[1]), 1.0)
    box_tf = coal.Transform3s(np.eye(3, dtype=float), np.array([float(ctr[0]), float(ctr[1]), 0.0], dtype=float))
    dist = float(coal.distance(seg_geom, seg_tf, box_geom, box_tf, coal.DistanceRequest(False), coal.DistanceResult()))
    return max(0.0, dist)


def plan_move_with_obstacle_clearance(
    adapter,
    robot,
    arm,
    own,
    target,
    start_pos_m,
    expected_q,
    obstacles,
    contact_entity_id,
    local_diagnostics,
    *,
    before_state=None,
    cap_remaining=None,
    **options,
):
    """当水平跨障移动穿过非目标障碍物上方且安全高度不足时，按声明的 clearance_height_m 尝试双段抬升跨越弧线。"""
    base_plan = adapter._plan_segment(robot, target, local_diagnostics, 0, **options)
    if base_plan is None or not obstacles or start_pos_m is None:
        return base_plan, None
    path_cfg = (own or {}).get('path') or {}
    clearance_h = float(path_cfg.get('clearance_height_m', DEFAULT_CLEARANCE_HEIGHT_M))
    if clearance_h <= 0.0:
        return base_plan, None
    exemption_r = float(path_cfg.get('target_exemption_radius_m', DEFAULT_TARGET_EXEMPTION_RADIUS_M))
    p0 = np.asarray(start_pos_m, dtype=float)
    p1 = np.asarray(target['position_m'], dtype=float)
    horiz_dist = float(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    if horiz_dist < 0.06:
        return base_plan, None
    min_link6_z = min(float(p0[2]), float(p1[2]))
    required_z = None
    gp_pos = ((own or {}).get('input_grasp_pose') or {}).get('position_m')
    for box in obstacles:
        if not isinstance(box, dict) or box.get('id') == contact_entity_id:
            continue
        low = np.asarray(box['min_m'], dtype=float)
        high = np.asarray(box['max_m'], dtype=float)
        margin = float(box.get('margin_m', 0.0))
        # 仅当当前阶段自身的起点、终点或抓取中心落在该障碍物 XY 豁免邻域内时豁免该阶段，同 batch 其他过境阶段仍正常跨障
        bid = str(box.get('id', ''))
        pad = exemption_r if (bid.startswith('placed_') or bid.startswith('vggt_auto_') or str(box.get('evidence', '')).startswith('vggt_object_table:')) else 0.015
        if (low[0] - pad <= p0[0] <= high[0] + pad and low[1] - pad <= p0[1] <= high[1] + pad):
            continue
        if (low[0] - pad <= p1[0] <= high[0] + pad and low[1] - pad <= p1[1] <= high[1] + pad):
            continue
        if isinstance(gp_pos, list) and len(gp_pos) == 3 and (
            low[0] - pad <= float(gp_pos[0]) <= high[0] + pad and low[1] - pad <= float(gp_pos[1]) <= high[1] + pad
        ):
            continue
        xy_dist = _segment_xy_distance_to_box(p0[:2], p1[:2], low[:2] - margin, high[:2] + margin)
        if xy_dist <= 0.035:
            clearance_link6_z = float(high[2]) + margin + clearance_h
            if min_link6_z < clearance_link6_z:
                required_z = clearance_link6_z if required_z is None else max(required_z, clearance_link6_z)
    if required_z is None:
        return base_plan, None
    safe_z = min(max(float(p0[2]), float(p1[2]), required_z), max(float(p0[2]), float(p1[2])) + 0.18, 1.18)
    if safe_z <= min_link6_z + 0.015:
        return base_plan, None
    mid_target = deepcopy(target)
    mid_target['position_m'] = [
        round(0.5 * (float(p0[0]) + float(p1[0])), 4),
        round(0.5 * (float(p0[1]) + float(p1[1])), 4),
        round(safe_z, 4),
    ]
    mid_diag, end_diag = [], []
    plan1 = adapter._plan_segment(robot, mid_target, mid_diag, 0, **options)
    if plan1 is None or len(plan1[0]) == 0:
        return base_plan, None
    pos1 = adapter._resample_action_plan(plan1[0], arm, own, start_q=np.asarray(expected_q))
    if len(pos1) == 0:
        return base_plan, None
    mid_q = np.asarray(pos1[-1], dtype=float)
    options2 = dict(options, start_q=mid_q)
    plan2 = adapter._plan_segment(robot, target, end_diag, 1, **options2)
    if plan2 is None or len(plan2[0]) == 0:
        return base_plan, None
    pos2 = adapter._resample_action_plan(plan2[0], arm, own, start_q=mid_q)
    if len(pos2) == 0:
        return base_plan, None
    positions = np.vstack([pos1, pos2])
    if cap_remaining is not None and len(positions) >= cap_remaining:
        return base_plan, None
    if before_state is not None:
        from .motion_safety import adapter_assessment
        state = deepcopy(before_state)
        state['arms'][arm]['q_rad'] = list(expected_q)
        assessment = adapter_assessment(
            adapter,
            state,
            {arm: [row.tolist() for row in positions]},
            native_steps=len(positions),
            request_remaining=cap_remaining or len(positions) + 1,
        )
        if assessment.get('enabled') and assessment.get('status') == 'reject':
            return base_plan, None
    return base_plan, positions


def filter_review_obstacles(obstacles, records):
    """自动记录的 placed_* 与自动桥接的 vggt_auto_* 包络仅用于跨障弧线规划，不注入几何碰撞预览告警，仅保留显式传入或登记的精确障碍物。"""
    if not obstacles:
        return []
    return [
        box
        for box in obstacles
        if not (str(box.get('id', '')).startswith('placed_') or str(box.get('id', '')).startswith('vggt_auto_'))
    ]
