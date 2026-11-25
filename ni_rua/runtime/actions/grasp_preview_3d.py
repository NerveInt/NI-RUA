"""3D 感知驱动的 robot_action 抓取几何与净空客观参考模块。

当外部 3D 感知（object_table / object_table_observe）启用且当前观察存在缓存的 object_table 时，
在 preview / preflight / execute_if_clean 阶段自动计算中性客观的 3D 几何参考指标：
1. blocking_risks 恒为空列表（`[]`）：环境几何估计绝不充当盲裁判硬阻断模型动作；
   `execute_if_clean` 的硬阻断严格只守卫机器人本体运动学/限位/自碰不可行以及模型显式声明的 `obstacle_object_ids` / `obstacles`。
2. advisory_notes 与 stages 客观记录测量到的指尖净空、最小表面距离、主轴夹角、侧向偏心与门禁开度对比，
   供模型自主判断，绝不越俎代庖下达策略指令或拦截合法物理接触。

Embodiment (design 13 section 5.3): the audit reads the commanded grasp poses in the RoboProbe ``angles_deg``
convention for an embodiment with ``"grasp_angles" in profile.pose_inputs`` (ARX-X5), and the grasp-centre requests
(``grasp_centre_m`` + ``object_width_m``) of an embodiment with ``"jaw_grasp_centre"`` (SO-101:
``audit_batch_jaw_grasp``); every other robot gets (None, []). Arms, full opening (``gripper.full_open_aperture_m``),
fingertip reach (``gripper.fingertip_beyond_reference_m``) and preset minimums come from the run's profile.
"""
from __future__ import annotations

import math

from ...core.episode_state import slot_peek
from ...core.pose_math import (
    axis_difference_deg as axis_difference,
    closure_yaws_deg as closure_yaws,
    nearest_yaw_deg as nearest,
)
from ...robot.motion.grip_guard import resolve_grip_guard
from ...perception.external import embodiment_of
from ...perception.external.object_table import is_horizontally_elongated_graspable
from ...perception.external import obs3d
from ...perception.external.pointcloud_field import PointCloudSpatialField
from ...perception.external.virtual_field import VirtualSpatialField, flange_pose_to_grasp_pose
from ...perception.world_model import attached_table as wm_attached_table, get_world_observation

BLOCKING_RISK_CODES: frozenset[str] = frozenset()

ADVISORY_NOTE_CODES = frozenset({
    "warn_finger_collides_neighbor",
    "warn_transport_collision",
    "warn_closed_jaw_descent",
    "warn_wider_than_max_opening",
    "warn_along_long_axis",
    "warn_fingertip_below_support",
    "warn_above_object_top",
    "warn_guard_exceeds_object_width",
})


def _resolve_robot_state(harness, obs_id):
    """从 harness 当前观测或历史观测归档中提取实测 robot_state。"""
    obs = None
    cur_fn = getattr(harness, "current_observation", None)
    if callable(cur_fn):
        try:
            cand = cur_fn()
            if isinstance(cand, dict) and (not obs_id or cand.get("observation_id") == obs_id):
                obs = cand
        except Exception:
            obs = None
    if obs is None and obs_id and getattr(harness, "scene", None) is not None:
        try:
            obs = harness.scene.observation(getattr(harness, "identity", {}), obs_id)
        except Exception:
            obs = None
    if not isinstance(obs, dict):
        return None
    if isinstance(obs.get("robot_state"), dict):
        return obs["robot_state"]
    ref = obs.get("robot_state_ref")
    if ref:
        resolve_json = getattr(getattr(harness, "scene", None), "resolve_json", None)
        if callable(resolve_json):
            try:
                res = resolve_json(ref)
                if isinstance(res, dict):
                    return res
            except Exception:
                pass
        resolve_art = getattr(getattr(harness, "store", None), "resolve_artifact", None)
        if callable(resolve_art):
            try:
                import json
                _, raw = resolve_art(ref)
                res = json.loads(raw)
                if isinstance(res, dict):
                    return res
            except Exception:
                pass
    return None


def _stage_pose_for_arm(stage, arm, *, profile=None):
    own = (stage or {}).get("action") or {}
    gp = own.get("input_grasp_pose")
    if isinstance(gp, dict) and arm in gp:
        gp = gp[arm]
    if isinstance(gp, dict) and "position_m" in gp and "angles_deg" in gp:
        return gp
    waypoints = ((own.get("path") or {}).get("waypoints")) if isinstance(own.get("path"), dict) else None
    if isinstance(waypoints, list) and waypoints:
        wp = waypoints[-1]
        if isinstance(wp, dict) and arm in wp and isinstance(wp[arm], dict):
            wp = wp[arm]
        if isinstance(wp, dict) and "position_m" in wp and "quaternion_xyzw" in wp:
            grip = getattr(profile, "gripper", None)
            kw = {}
            if grip is not None:
                if getattr(grip, "grasp_reference_offset_m", None) is not None:
                    kw["grasp_point_offset_m"] = float(grip.grasp_reference_offset_m)
                if getattr(grip, "fingertip_beyond_reference_m", None) is not None:
                    kw["fingertip_below_pad_m"] = float(grip.fingertip_beyond_reference_m)
            conv = flange_pose_to_grasp_pose(wp["position_m"], wp["quaternion_xyzw"], profile=profile, **kw)
            return dict(position_m=conv["grasp_position_m"], angles_deg=conv["angles_deg"])
    return None


def _initial_closed_empty_arms(harness, obs_id: str, vf: VirtualSpatialField, profile=None) -> dict[str, float]:
    """返回批次起始时处于闭合空爪状态（开度低于空抓阈值且未夹持追踪物体）的机械臂及其开度（单位 m）。"""
    profile = profile or embodiment_of(harness)
    arms = profile.arms
    zero = profile.gripper.empty_close_aperture_m
    thresh_m = (0.0 if zero is None else float(zero)) + 0.003
    held_arms = {o.held_by_arm for o in vf.objects.values() if o.held_by_arm in arms}
    obs = getattr(harness, "latest", None)
    grip_facts = obs.get("grip_facts") if isinstance(obs, dict) and obs.get("observation_id") == obs_id and isinstance(obs.get("grip_facts"), dict) else {}
    closed_empty: dict[str, float] = {}
    for arm in arms:
        if arm in held_arms:
            continue
        ap_m = None
        gf = grip_facts.get(arm)
        if isinstance(gf, dict) and isinstance(gf.get("measured_aperture_m"), (int, float)):
            ap_m = float(gf["measured_aperture_m"])
        elif arm in vf.arm_apertures_m:
            # VirtualSpatialField 仅在 robot_state 含实测夹爪数据时更新 arm_apertures_m，未知时保持 MAX_OPENING_M
            ap_m = float(vf.arm_apertures_m[arm])
        if ap_m is not None and ap_m < thresh_m:
            closed_empty[arm] = ap_m
    return closed_empty


def _stage_jaw_goals(stage: dict) -> list[float]:
    """从已编译阶段（action.gripper_events）或原始阶段字段提取归一化夹爪目标序列 [0, 1]。"""
    own = stage.get("action") or {}
    events = own.get("gripper_events") or stage.get("gripper_events") or []
    goals = [
        float(ev["goal_normalized"])
        for ev in events
        if isinstance(ev, dict) and isinstance(ev.get("goal_normalized"), (int, float))
    ]
    if goals:
        return goals
    for key in ("gripper_goal_normalized", "goal_normalized"):
        if isinstance(stage.get(key), (int, float)):
            return [float(stage[key])]
    return []


def _extract_closing_grasp_stages(action, arms=None, *, closed_empty_arms: dict[str, float] | None = None, profile=None):
    """提取批次中发生抓握闭合动作的阶段（含由开到闭抓握及已闭爪直接指令闭爪阶段）。"""
    stages = (action or {}).get("stages") or []
    arm_ids = arms if arms is not None else (profile.arms if profile is not None else embodiment_of(None).arms)
    closed_empty = dict(closed_empty_arms or {})
    aperture = {
        arm: (0.0 if arm in closed_empty else 1.0)
        for arm in arm_ids
    }
    opened_in_batch: set[str] = set()
    closing = []
    for idx, stage in enumerate(stages):
        if not isinstance(stage, dict):
            continue
        arm = stage.get("arm")
        kind = stage.get("kind")
        goals = _stage_jaw_goals(stage)
        if kind == "gripper":
            if arm in aperture and goals:
                goal = goals[-1]
                aperture[arm] = goal
                if goal >= 0.5:
                    opened_in_batch.add(arm)
            continue
        if kind != "move" or arm not in aperture:
            continue
        gp = _stage_pose_for_arm(stage, arm, profile=profile)
        if gp is None and isinstance(stage.get("grasp_position_m"), (list, tuple)) and isinstance(stage.get("angles_deg"), (list, tuple)):
            gp = dict(position_m=stage["grasp_position_m"], angles_deg=stage["angles_deg"])
        if gp is None:
            if goals:
                goal = goals[-1]
                aperture[arm] = goal
                if goal >= 0.5:
                    opened_in_batch.add(arm)
            continue
        was_open = aperture[arm] >= 0.5
        already_closed_empty = (arm in closed_empty) and (arm not in opened_in_batch) and (not was_open)
        own_closing = bool(goals) and goals[0] < 0.5
        own_closing_strict = bool(goals) and goals[0] <= 0.2
        if goals:
            goal = goals[-1]
            aperture[arm] = goal
            if goal >= 0.5:
                opened_in_batch.add(arm)
        is_closing = was_open and own_closing
        is_closed_grasp_attempt = already_closed_empty and own_closing_strict
        guard_stage = stage
        next_same_arm_move_z = None
        for follow in stages[idx + 1 :]:
            if not isinstance(follow, dict) or follow.get("arm") != arm:
                continue
            if follow.get("kind") == "move":
                fgp = _stage_pose_for_arm(follow, arm, profile=profile)
                if fgp is None and isinstance(follow.get("grasp_position_m"), (list, tuple)) and len(follow["grasp_position_m"]) >= 3:
                    next_same_arm_move_z = float(follow["grasp_position_m"][2])
                elif fgp is not None:
                    next_same_arm_move_z = float(fgp["position_m"][2])
                break
            if follow.get("kind") == "gripper" and not is_closing and not is_closed_grasp_attempt:
                f_goals = _stage_jaw_goals(follow)
                if f_goals:
                    f_goal = f_goals[0]
                    if was_open and f_goal < 0.5:
                        is_closing = True
                        guard_stage = follow
                    elif already_closed_empty and f_goal <= 0.2:
                        is_closed_grasp_attempt = True
                        guard_stage = follow
        if is_closing or is_closed_grasp_attempt:
            cur_z = float(gp["position_m"][2])
            # 过滤高处预定位过渡路点（后续同臂立即继续显著下探 >15mm 时不误判为最终抓取面）
            if is_closing and not is_closed_grasp_attempt and next_same_arm_move_z is not None and (cur_z - next_same_arm_move_z) > 0.015:
                continue
            guard = resolve_grip_guard(guard_stage, action, profile=profile)
            own_action = stage.get("action") or {}
            own_path = own_action.get("path") if isinstance(own_action.get("path"), dict) else {}
            tol_preset = stage.get("tolerance_preset") or own_path.get("tolerance_preset") or action.get("tolerance_preset") or "normal"
            stall_preset = stage.get("stall_guard_preset") or own_path.get("stall_guard_preset") or action.get("stall_guard_preset") or "off"
            explicit_dz = stage.get("max_upward_dz_m")
            if explicit_dz is None:
                explicit_dz = own_path.get("max_upward_dz_m", action.get("max_upward_dz_m"))
            has_vertical_compliance = (
                tol_preset in ("compliant", "seating")
                or stall_preset == "contact_stop"
                or (isinstance(explicit_dz, (int, float)) and float(explicit_dz) > 0.0)
            )
            closing.append(
                dict(
                    stage_id=str(stage["id"]).removesuffix("/motion"),
                    arm=arm,
                    position_m=[float(v) for v in gp["position_m"]],
                    angles_deg=[float(v) for v in gp["angles_deg"]],
                    contact_region=str(gp.get("contact_region") or stage.get("contact_region") or "pad_center"),
                    contact_depth_m=gp.get("contact_depth_m") if gp.get("contact_depth_m") is not None else stage.get("contact_depth_m"),
                    grip_guard=guard,
                    has_vertical_compliance=has_vertical_compliance,
                    already_closed_before_grasp=is_closed_grasp_attempt,
                    initial_aperture_m=closed_empty.get(arm),
                )
            )
    return closing


def _match_object_row(table, gx, gy, max_opening_m):
    """在 object_table 中匹配水平距离最近且可单爪抓取（排除超宽容器/支架及手中已夹持物）的 3D 物体行。

    ``max_opening_m`` (m): the gripper's full opening; wider rows do not fit between the jaws."""
    columns = list(table.get("columns") or [])
    best_row, best_dist = None, math.inf
    for raw in table.get("rows") or []:
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            continue
        row = dict(zip(columns, raw))
        if str(row.get("support") or "").startswith("held_") or row.get("quality") == "held_fk_depth":
            continue
        if (
            not isinstance(row.get("length"), (int, float))
            or not isinstance(row.get("width"), (int, float))
            or not isinstance(row.get("yaw_deg"), (int, float))
        ):
            continue
        centre = row.get("centre_m")
        if not isinstance(centre, list) or len(centre) != 3:
            continue
        width = float(row.get("width") or 0.0)
        if width > max_opening_m:
            continue
        cx, cy = float(centre[0]), float(centre[1])
        length = float(row.get("length") or 0.0)
        dist = math.hypot(gx - cx, gy - cy)
        reach = max(0.5 * length + 0.035, 0.5 * width + 0.035, 0.065)
        if dist <= reach and dist < best_dist:
            best_row, best_dist = row, dist
    return best_row, best_dist


def _get_world_obs(harness, obs_id: str | None):
    """获取与 `obs_id` 对应的不可变 `WorldObservation3D` 快照。"""
    if not obs_id:
        return None
    return get_world_observation(harness, obs_id)


def _attached_table(harness, obs_id: str | None):
    """获取挂载于 `obs_id` 的 `object_table` 字典。"""
    if not obs_id:
        return None
    return wm_attached_table(harness, obs_id)


def _build_audit_spatial_field(
    harness,
    obs_id: str,
    table: dict,
    support_z,
    robot_state: dict | None,
    arms=("left", "right"),
    *,
    world_obs=None,
    profile=None,
) -> VirtualSpatialField:
    """为 3D 预检构建与 `obs_id` 对齐的独立空间模拟场副本，保留点云/凹腔几何且绝不污染共享实例。"""
    eff_profile = profile if profile is not None else embodiment_of(harness)
    shared_vf = slot_peek(harness, obs3d.VIRTUAL_FIELD_SLOT, None, legacy_attr="_virtual_spatial_field")
    if isinstance(shared_vf, VirtualSpatialField) and getattr(shared_vf, "observation_id", None) == obs_id:
        vf = shared_vf.clone()
        if isinstance(robot_state, dict):
            vf._update_robot_state(robot_state)
            for obj in vf.objects.values():
                if obj.held_by_arm in arms and obj.rel_offset_in_gripper is not None:
                    obj.update_from_gripper(vf.gripper_frame(obj.held_by_arm))
        return vf

    if world_obs is None:
        world_obs = _get_world_obs(harness, obs_id)

    raw_objects = None
    art_ref = None
    if world_obs is not None:
        raw_objects = world_obs.raw_objects()
        art_ref = world_obs.cloud_artifact_ref
    else:
        memo = obs3d._memo(harness, obs_id)
        ot_entry = memo.get("object_table")
        if isinstance(ot_entry, tuple) and len(ot_entry) >= 3 and isinstance(ot_entry[2], dict):
            raw_objects = (ot_entry[2].get("diagnostics") or {}).get("objects")
        s3_entry = memo.get("scene3d")
        if isinstance(s3_entry, tuple) and s3_entry and isinstance(s3_entry[0], dict):
            art_ref = s3_entry[0].get("artifact")

    cloud_npz = None
    resolve_art = getattr(getattr(harness, "store", None), "resolve_artifact", None)
    if isinstance(art_ref, str) and art_ref and callable(resolve_art):
        _, cloud_npz = resolve_art(art_ref)

    return PointCloudSpatialField(
        objects=table,
        table_z=support_z,
        robot_state=robot_state,
        cloud_npz=cloud_npz,
        raw_objects=raw_objects,
        profile=eff_profile,
    )


def _match_sim_grasp_target(
    table: dict,
    vf: VirtualSpatialField,
    gx: float,
    gy: float,
    gz: float,
    max_opening_m: float,
    *,
    world_obs=None,
) -> str | None:
    """匹配闭爪抓取接触点对应的 3D 目标物体 ID（优先可单爪跨抓小物体，其次回退至 25mm 内直接接触的宽器皿/织物边缘）。"""
    import numpy as np

    if world_obs is not None and world_obs.has_table:
        row, _ = world_obs.match_graspable_row(gx, gy, max_opening_m)
    else:
        row, _ = _match_object_row(table, gx, gy, max_opening_m)
    if row is not None:
        b_z = row.get("bottom_z")
        t_z = row.get("top_z")
        if (not isinstance(b_z, (int, float)) or gz >= float(b_z) - 0.025) and (
            not isinstance(t_z, (int, float)) or gz <= float(t_z) + 0.035
        ):
            return str(row["id"])
    pt = np.array([float(gx), float(gy), float(gz)], dtype=np.float64)
    best_oid: str | None = None
    best_dist = 0.025
    for obj in vf.objects.values():
        if obj.held_by_arm is not None:
            continue
        obb = obj.obb()
        dist = float(np.linalg.norm(pt - obb.clamp_point(pt)))
        if dist <= best_dist:
            best_dist = dist
            best_oid = str(obj.id)
    return best_oid


def _prepare_simulation_stages(
    raw_stages: list,
    table: dict,
    vf: VirtualSpatialField,
    closed_empty_arms: dict[str, float],
    *,
    arms=("left", "right"),
    max_opening_m: float = 0.088,
    world_obs=None,
    profile=None,
) -> tuple[list[dict], dict[str, str | None]]:
    """合并已编译的 `id/motion + id(gripper)` 原子阶段对，记录竖直闭爪下探目标接触豁免与持握物体映射。"""
    from copy import deepcopy

    consumed: set[int] = set()
    coalesced: list[dict] = []
    n_raw = len(raw_stages)
    for idx, st in enumerate(raw_stages):
        if idx in consumed or not isinstance(st, dict):
            continue
        sid = str(st.get("id", ""))
        arm = st.get("arm")
        if st.get("kind") == "move" and sid.endswith("/motion") and arm in arms:
            base_id = sid.removesuffix("/motion")
            paired_idx = None
            for j in range(idx + 1, n_raw):
                cand = raw_stages[j]
                if not isinstance(cand, dict) or cand.get("arm") != arm:
                    continue
                if cand.get("kind") == "gripper" and str(cand.get("id", "")) == base_id:
                    paired_idx = j
                break
            if paired_idx is not None:
                merged = deepcopy(st)
                merged["id"] = base_id
                own = merged.setdefault("action", {})
                g_evs = deepcopy(((raw_stages[paired_idx].get("action") or {}).get("gripper_events")) or [])
                if g_evs:
                    own["gripper_events"] = g_evs
                consumed.add(paired_idx)
                coalesced.append(merged)
                continue
        coalesced.append(deepcopy(st))

    arm_xyz: dict[str, tuple[float, float, float]] = {}
    for arm in arms:
        pose = vf.arm_poses.get(arm)
        if isinstance(pose, dict) and isinstance(pose.get("position_m"), (list, tuple)) and len(pose["position_m"]) >= 3:
            arm_xyz[arm] = (
                float(pose["position_m"][0]),
                float(pose["position_m"][1]),
                float(pose["position_m"][2]),
            )
    arm_open = {arm: (arm not in closed_empty_arms) for arm in arms}
    arm_held_oid: dict[str, str | None] = {
        arm: next((o.id for o in vf.objects.values() if o.held_by_arm == arm), None)
        for arm in arms
    }
    stage_held_oid: dict[str, str | None] = {}

    for st in coalesced:
        sid = str(st.get("id", ""))
        arm = st.get("arm")
        kind = st.get("kind")
        goals = _stage_jaw_goals(st)
        if arm in arms:
            stage_held_oid[sid] = arm_held_oid.get(arm)
        if kind == "move" and arm in arms:
            gp = _stage_pose_for_arm(st, arm, profile=profile)
            if gp is None and isinstance(st.get("grasp_position_m"), (list, tuple)) and isinstance(st.get("angles_deg"), (list, tuple)):
                gp = dict(position_m=st["grasp_position_m"], angles_deg=st["angles_deg"])
            if gp is not None:
                gx, gy, gz = float(gp["position_m"][0]), float(gp["position_m"][1]), float(gp["position_m"][2])
                if arm_open.get(arm, True) and goals and goals[0] < 0.5:
                    oid = _match_sim_grasp_target(table, vf, gx, gy, gz, max_opening_m, world_obs=world_obs)
                    if oid is not None:
                        allow_ids = list(st.get("allow_contact_ids") or [])
                        if oid not in allow_ids:
                            allow_ids.append(oid)
                        st["allow_contact_ids"] = allow_ids
                        arm_held_oid[arm] = oid
                        arm_open[arm] = False
                elif goals:
                    if goals[-1] >= 0.5:
                        arm_open[arm] = True
                        arm_held_oid[arm] = None
                    elif goals[-1] < 0.5:
                        arm_open[arm] = False
                arm_xyz[arm] = (gx, gy, gz)
        elif kind == "gripper" and arm in arms and goals:
            if goals[-1] >= 0.5:
                arm_open[arm] = True
                arm_held_oid[arm] = None
            elif goals[-1] < 0.5:
                if arm_open.get(arm, True) and arm in arm_xyz:
                    oid = _match_sim_grasp_target(
                        table, vf, arm_xyz[arm][0], arm_xyz[arm][1], arm_xyz[arm][2], max_opening_m, world_obs=world_obs
                    )
                    if oid is not None:
                        arm_held_oid[arm] = oid
                arm_open[arm] = False

    return coalesced, stage_held_oid


def _box_extent_m(direction, row):
    """Extent (m) of an object-table row's box along the unit ``direction`` (frame axes): length along the row's
    yaw, width across it and height top_z - bottom_z (0 when unknown), each projected on ``direction``."""
    yaw = math.radians(float(row["yaw_deg"]))
    top, bottom = row.get("top_z"), row.get("bottom_z")
    height = float(top) - float(bottom) if isinstance(top, (int, float)) and isinstance(bottom, (int, float)) else 0.0
    return (float(row["length"]) * abs(direction[0] * math.cos(yaw) + direction[1] * math.sin(yaw))
            + float(row["width"]) * abs(-direction[0] * math.sin(yaw) + direction[1] * math.cos(yaw))
            + max(height, 0.0) * abs(direction[2]))


def _closing_jaw_grasp_stages(action, stage_plans, initial_goals, *, profile=None):
    """Grasp-centre move stages after which the jaw closes: ``[{stage_id, arm, report, arrival_goal, close_goal,
    grip_guard}]``."""
    stages = [stage for stage in (action or {}).get("stages") or [] if isinstance(stage, dict)]
    goals, closing = dict(initial_goals), []
    for index, stage in enumerate(stages):
        arm, events = stage.get("arm"), (stage.get("action") or {}).get("gripper_events") or []
        if events:
            goals[arm] = float(events[-1].get("goal_normalized", goals.get(arm) or 0.0))
        report = ((stage_plans or {}).get(stage.get("id")) or {}).get("grasp_centre")
        if stage.get("kind") != "move" or not isinstance(report, dict) or "jaw" not in report:
            continue
        follow = next((item for item in stages[index + 1:] if item.get("arm") == arm), None)
        if follow is None or follow.get("kind") != "gripper":
            continue
        close_goal = float(follow["action"]["gripper_events"][0]["goal_normalized"])
        arrival = goals.get(arm)
        if arrival is not None and close_goal > arrival:
            continue
        closing.append(dict(stage_id=stage["id"], arm=arm, report=report, arrival_goal=arrival, close_goal=close_goal,
                            grip_guard=resolve_grip_guard(follow, action, profile=profile)))
    return closing


def _commanded_jaw_goals(harness, observation_id, arms):
    """``{arm: commanded jaw goal_normalized or None}`` from the robot_state of ``observation_id``."""
    import json
    goals = dict.fromkeys(arms)
    try:
        bundle = harness.store.read("observation", observation_id)
        _, data = harness.store.resolve_artifact(bundle["robot_state_ref"], allowed_scopes=("actor",))
        for arm, state in json.loads(data).get("arms", {}).items():
            goal = (state.get("gripper") or {}).get("goal_normalized")
            if arm in goals and isinstance(goal, (int, float)):
                goals[arm] = float(goal)
    except Exception:
        pass
    return goals


def audit_batch_jaw_grasp(harness, action, observation_id=None, stage_plans=None):
    """``grasp_3d_review`` of a fixed-jaw / moving-jaw gripper (pose input ``jaw_grasp_centre``, SO-101)."""
    profile = embodiment_of(harness)
    gripper, bounds = profile.gripper, profile.gripper.jaw_grasp
    obs_id = observation_id or (action or {}).get("based_on_observation_id")
    closing = _closing_jaw_grasp_stages(
        action,
        stage_plans,
        _commanded_jaw_goals(harness, obs_id, profile.arms) if obs_id else dict.fromkeys(profile.arms),
        profile=profile,
    )
    if not closing:
        return None, []
    world_obs = _get_world_obs(harness, obs_id) if obs_id else None
    table = world_obs.table_dict() if world_obs is not None else (_attached_table(harness, obs_id) if obs_id else None)
    table = table if isinstance(table, dict) and table.get("rows") else None
    leaves = {stage["id"]: stage.get("action") or {} for stage in action["stages"] if isinstance(stage, dict)}
    mm = lambda value: f"{value * 1000:.1f} mm"
    stage_audits, targets_3d, warnings = [], [], []
    for item in closing:
        report, jaw, sid = item["report"], item["report"]["jaw"], item["stage_id"]
        centre, width = report["grasp_centre_m"], report["object_width_m"]
        direction, approach = report["fixed_jaw_direction"], report["approach_direction"]
        stage_warnings = []
        audit = dict(stage_id=sid, arm=item["arm"], grasp_centre_m=centre, object_width_m=width,
                     arrival_jaw_goal_normalized=item["arrival_goal"],
                     straddle_goal_normalized=jaw["straddle_goal_normalized"],
                     closing_jaw_goal_normalized=item["close_goal"],
                     closed_on_object_goal_normalized=jaw["closed_on_object_goal_normalized"])
        opening = "ok"
        if item["arrival_goal"] is None or jaw["straddle_goal_normalized"] is None:
            opening = "unknown"
        elif item["arrival_goal"] < jaw["straddle_goal_normalized"]:
            opening = "warn_jaw_not_open_enough"
            stage_warnings.append(
                f"Stage '{sid}' arrives with jaw goal {item['arrival_goal']:g}, below the {jaw['straddle_goal_normalized']:g} "
                f"that opens the fingertips to object_width_m + fixed_jaw_clearance_m = {mm(jaw['straddle_gap_m'])}: the "
                "moving fingertip would come down on the object. Open the jaw before this move.")
        elif (jaw["closed_on_object_goal_normalized"] is not None
              and item["close_goal"] > jaw["closed_on_object_goal_normalized"]):
            opening = "warn_closing_goal_above_object_width"
            stage_warnings.append(
                f"After stage '{sid}' the jaw closes to goal {item['close_goal']:g}, above the "
                f"{jaw['closed_on_object_goal_normalized']:g} at which the fingertips meet a {mm(width)} object: it would "
                "not grip that width.")
        audit["jaw_opening_check"] = opening
        guard, guard_status = item.get("grip_guard") or {}, "ok"
        if guard.get("min_aperture_m") is not None and float(guard["min_aperture_m"]) > width:
            guard_status = "warn_guard_exceeds_object_width"
            stage_warnings.append(
                f"Stage '{sid}' grip guard min_aperture_m={mm(float(guard['min_aperture_m']))} exceeds object_width_m="
                f"{mm(width)}: a good grasp of that width would pause.")
        audit["grip_guard_check"] = guard_status
        planned, tolerance = report.get("planned"), (leaves.get(sid, {}).get("path") or {}).get("position_tolerance_m")
        centre_status = "ok"
        if isinstance(planned, dict):
            audit["planned_centre_error_m"] = planned["centre_error_m"]
            if isinstance(tolerance, (int, float)) and planned["centre_error_m"] > tolerance:
                centre_status = "warn_planned_centre_outside_position_tolerance"
                stage_warnings.append(
                    f"Stage '{sid}' planned joints put the grasp centre {mm(planned['centre_error_m'])} from grasp_centre_m "
                    f"(position_tolerance_m {mm(float(tolerance))}): the accepted orientation residual swings the centre "
                    "about the TCP. Tighten the orientation tolerance or choose a reachable orientation.")
        audit["centre_check"] = centre_status
        if world_obs is not None and world_obs.has_table:
            row, xy_dist = world_obs.match_graspable_row(centre[0], centre[1], bounds.max_gap_m)
        else:
            row, xy_dist = (_match_object_row(table, centre[0], centre[1], bounds.max_gap_m) if table is not None
                            else (None, None))
        audit["matched_object_id"] = None if row is None else row["id"]
        if row is not None:
            oid = row["id"]
            row_centre = [float(value) for value in row["centre_m"]]
            length, row_width, obj_yaw = float(row["length"]), float(row["width"]), float(row["yaw_deg"])
            top_z, bottom_z = row.get("top_z"), row.get("bottom_z")
            support = float(bottom_z) if isinstance(bottom_z, (int, float)) else (
                float(table["table_z"]) if isinstance(table.get("table_z"), (int, float)) else None)
            fingertip_z = report["tcp_target_position_m"][2] + gripper.fingertip_beyond_reference_m * approach[2]
            height_status = "ok"
            if isinstance(top_z, (int, float)) and centre[2] > float(top_z):
                height_status = "warn_centre_above_object_top"
                stage_warnings.append(
                    f"Stage '{sid}' grasp_centre_m z={centre[2]:.4f} m is {mm(centre[2] - float(top_z))} above {oid} "
                    f"top_z={float(top_z):.4f} m (object centre z={row_centre[2]:.4f} m).")
            elif isinstance(bottom_z, (int, float)) and centre[2] < float(bottom_z):
                height_status = "warn_centre_below_object_bottom"
                stage_warnings.append(
                    f"Stage '{sid}' grasp_centre_m z={centre[2]:.4f} m is {mm(float(bottom_z) - centre[2])} below {oid} "
                    f"bottom_z={float(bottom_z):.4f} m.")
            elif support is not None and fingertip_z < support:
                height_status = "warn_fingertip_below_support"
                stage_warnings.append(
                    f"Stage '{sid}' fingertips reach z={fingertip_z:.4f} m, {mm(support - fingertip_z)} below the "
                    f"support at z={support:.4f} m under {oid} (normal for an object thinner than the fingertip reach; "
                    "the arm then presses on the support).")
            extent = _box_extent_m(direction, row)
            offset = sum((centre[index] - row_centre[index]) * direction[index] for index in range(3))
            fixed_gap = offset + width / 2 + report["fixed_jaw_clearance_m"] - extent / 2
            fixed_status = "ok"
            if fixed_gap < 0:
                fixed_status = "warn_fixed_jaw_inside_object"
                stage_warnings.append(
                    f"Stage '{sid}' puts the fixed fingertip face {mm(-fixed_gap)} inside {oid} along the jaw line "
                    f"({oid} spans {mm(extent)} there, centre {mm(abs(offset))} from grasp_centre_m; object_width_m="
                    f"{mm(width)}, fixed_jaw_clearance_m={mm(report['fixed_jaw_clearance_m'])}): the fixed jaw would land "
                    "on the object or push it.")
            width_status = "ok"
            if extent > bounds.max_gap_m:
                width_status = "warn_wider_than_max_gap"
                stage_warnings.append(
                    f"Object {oid} spans {mm(extent)} along this jaw line, more than the widest jaw gap "
                    f"{mm(bounds.max_gap_m)}; choose another jaw line or a narrower part.")
            elongated = is_horizontally_elongated_graspable(length, row_width, top_z, bottom_z, row.get("tilt_deg"),
                                                            max_opening_m=bounds.max_gap_m)
            axis_status, to_long_axis = "ok", None
            horizontal = math.hypot(direction[0], direction[1])
            if horizontal > 0:
                jaw_yaw = math.degrees(math.atan2(direction[1], direction[0]))
                to_long_axis = round(axis_difference(jaw_yaw, obj_yaw), 1)
                if elongated and to_long_axis < 45.0:
                    axis_status = "warn_along_long_axis"
                    stage_warnings.append(
                        f"Stage '{sid}' jaw line is {to_long_axis:.0f} deg from the long axis of elongated {oid} "
                        f"(length {mm(length)}, width {mm(row_width)}, yaw {obj_yaw:.0f} deg): the jaws would close "
                        "along its length. The line across it is at yaw "
                        f"{nearest(closure_yaws(obj_yaw), jaw_yaw):.1f} deg (or that + 180 with the jaws swapped).")
            audit.update(xy_distance_m=round(xy_dist, 4),
                         object_top_z_m=round(float(top_z), 4) if isinstance(top_z, (int, float)) else None,
                         object_bottom_z_m=round(float(bottom_z), 4) if isinstance(bottom_z, (int, float)) else None,
                         fingertip_z_m=round(fingertip_z, 4), height_check=height_status,
                         object_extent_on_jaw_line_m=round(extent, 4), fixed_jaw_gap_to_object_m=round(fixed_gap, 4),
                         fixed_jaw_check=fixed_status, width_check=width_status,
                         jaw_line_to_long_axis_deg=to_long_axis, axis_alignment=axis_status)
            targets_3d.append(dict(id=oid, centre_m=row_centre, length_m=length, width_m=row_width, yaw_deg=obj_yaw,
                                   top_z_m=float(top_z) if isinstance(top_z, (int, float)) else row_centre[2]))
        audit["risk_flags"] = [value for key, value in audit.items()
                               if key.endswith("_check") or key == "axis_alignment"
                               if isinstance(value, str) and value.startswith("warn_")]
        audit["warnings"] = stage_warnings
        warnings.extend(stage_warnings)
        stage_audits.append(audit)
    review = dict(observation_id=obs_id, status="warnings_detected" if warnings else "aligned",
                  risk_flags=[flag for stage in stage_audits for flag in stage["risk_flags"]],
                  stages=stage_audits, warnings=warnings)
    return review, targets_3d


def audit_batch_grasp_3d(harness, action, observation_id=None, stage_plans=None):
    """对批次中的闭合抓取阶段与搬运轨迹计算中性客观的 3D 几何参考；无缓存 object_table 时返回 (None, [])。

    Runs for an embodiment with the grasp-angle convention (``"grasp_angles" in profile.pose_inputs``); an embodiment
    with the grasp-centre input (``"jaw_grasp_centre"``) gets ``audit_batch_jaw_grasp``, which reads the preview's
    ``stage_plans``. The profile is ``harness.embodiment`` (default robot ARX-X5 without one)."""
    profile = embodiment_of(harness)
    if "jaw_grasp_centre" in profile.pose_inputs:
        return audit_batch_jaw_grasp(harness, action, observation_id, stage_plans)
    if "grasp_angles" not in profile.pose_inputs:
        return None, []
    gripper = profile.gripper
    max_opening_m, fingertip_m = gripper.full_open_aperture_m, gripper.fingertip_beyond_reference_m
    obs_id = observation_id or (action or {}).get("based_on_observation_id")
    if not obs_id:
        return None, []
    world_obs = _get_world_obs(harness, obs_id)
    table = world_obs.table_dict() if world_obs is not None else _attached_table(harness, obs_id)
    if not isinstance(table, dict) or not table.get("rows"):
        return None, []
    support_z = table.get("table_z")
    robot_state = _resolve_robot_state(harness, obs_id)
    vf = _build_audit_spatial_field(
        harness, obs_id, table, support_z, robot_state, arms=profile.arms, world_obs=world_obs, profile=profile
    )
    closed_empty_arms = _initial_closed_empty_arms(harness, obs_id, vf, profile=profile)
    closing_stages = _extract_closing_grasp_stages(action, profile.arms, closed_empty_arms=closed_empty_arms, profile=profile)
    raw_stages = (action or {}).get("stages") or []
    sim_stages, stage_held_oid = _prepare_simulation_stages(
        raw_stages,
        table,
        vf,
        closed_empty_arms,
        arms=profile.arms,
        max_opening_m=max_opening_m,
        world_obs=world_obs,
        profile=profile,
    )
    sim_report = vf.simulate_stages(sim_stages)
    stage_audits, targets_3d, warnings = [], [], []
    blocking_risks: list[dict] = []
    advisory_notes: list[dict] = []

    for item in closing_stages:
        gx, gy, gz = item["position_m"]
        pitch, roll, yaw = item["angles_deg"]
        if world_obs is not None and world_obs.has_table:
            row, xy_dist = world_obs.match_graspable_row(gx, gy, max_opening_m)
        else:
            row, xy_dist = _match_object_row(table, gx, gy, max_opening_m)
        if row is None:
            continue
        oid = row["id"]
        centre = [float(v) for v in row["centre_m"]]
        length, width = float(row["length"]), float(row["width"])
        obj_yaw = float(row["yaw_deg"])
        top_z, bottom_z = row.get("top_z"), row.get("bottom_z")
        tilt = row.get("tilt_deg")
        ratio = length / width if width > 0 else 1.0
        elongated = is_horizontally_elongated_graspable(length, width, top_z, bottom_z, tilt, max_opening_m=max_opening_m)
        closing_angle = round(axis_difference(yaw, obj_yaw), 1)
        across_candidates = closure_yaws(obj_yaw)
        rec_yaw = nearest(across_candidates, yaw) if elongated else None

        stage_warnings: list[str] = []
        stage_blocking: list[dict] = []
        stage_advisories: list[dict] = []

        closed_jaw_status = "ok"
        if item.get("already_closed_before_grasp"):
            closed_jaw_status = "warn_closed_jaw_descent"
            ap_mm = (item.get("initial_aperture_m") or 0.0) * 1000.0
            msg = (
                f"Stage '{item['stage_id']}' approaches {oid} while {item['arm']} jaw is closed "
                f"(measured_aperture_m={ap_mm:.1f}mm)."
            )
            stage_warnings.append(msg)
            stage_advisories.append(
                dict(
                    code="warn_closed_jaw_descent",
                    stage_id=item["stage_id"],
                    arm=item["arm"],
                    object_id=oid,
                    measured_aperture_m=round(float(item.get("initial_aperture_m") or 0.0), 4),
                    message=msg,
                )
            )

        axis_status = "ok"
        if elongated and closing_angle < 45.0:
            axis_status = "warn_along_long_axis"
            msg = (
                f"Stage '{item['stage_id']}' jaw closing axis is {closing_angle:.0f} deg from elongated {oid} "
                f"(length/width={ratio:.2f}, yaw={obj_yaw:.0f} deg; across-axis yaw reference={rec_yaw:.1f} deg)."
            )
            stage_warnings.append(msg)
            stage_advisories.append(
                dict(
                    code="warn_along_long_axis",
                    stage_id=item["stage_id"],
                    arm=item["arm"],
                    object_id=oid,
                    closing_to_long_axis_deg=closing_angle,
                    recommended_across_yaw_deg=rec_yaw,
                    message=msg,
                )
            )

        support = float(bottom_z) if isinstance(bottom_z, (int, float)) else (
            float(support_z) if isinstance(support_z, (int, float)) else None
        )
        pad_z = centre[2]
        height_status = "ok"
        fingertip_clearance = round((gz - fingertip_m) - support, 4) if support is not None else None
        if isinstance(top_z, (int, float)) and gz > float(top_z) + 0.012:
            height_status = "warn_above_object_top"
            msg = (
                f"Stage '{item['stage_id']}' grasp z={gz:.3f}m is {(gz - float(top_z))*1000:.0f}mm above {oid} "
                f"visible top_z={float(top_z):.3f}m (object centre pad_z={pad_z:.3f}m)."
            )
            stage_warnings.append(msg)
            stage_advisories.append(
                dict(
                    code="warn_above_object_top",
                    stage_id=item["stage_id"],
                    arm=item["arm"],
                    object_id=oid,
                    planned_z_m=round(gz, 4),
                    recommended_pad_z_m=round(pad_z, 4),
                    message=msg,
                )
            )
        elif support is not None and (gz - fingertip_m) < support - 0.003:
            height_status = "warn_fingertip_below_support"
            compliance_note = (
                ""
                if item.get("has_vertical_compliance")
                else "; effective max_upward_dz_m=0.0m, so set stall_guard_preset='contact_stop' to prevent support-contact tracking_error before gripper closure"
            )
            msg = (
                f"Stage '{item['stage_id']}' fingertip z={gz - fingertip_m:.3f}m reaches below "
                f"support={support:.3f}m (object centre pad_z={pad_z:.3f}m{compliance_note})."
            )
            stage_warnings.append(msg)
            stage_advisories.append(
                dict(
                    code="warn_fingertip_below_support",
                    stage_id=item["stage_id"],
                    arm=item["arm"],
                    object_id=oid,
                    fingertip_support_clearance_m=fingertip_clearance,
                    recommended_pad_z_m=round(pad_z, 4),
                    message=msg,
                )
            )

        guard = item.get("grip_guard") or {}
        guard_status = "ok"
        if width > max_opening_m:
            guard_status = "warn_wider_than_max_opening"
            msg = (
                f"Object {oid} estimated width {width*1000:.0f}mm exceeds the {max_opening_m*1000:g}mm max jaw opening."
            )
            stage_warnings.append(msg)
            stage_advisories.append(
                dict(
                    code="warn_wider_than_max_opening",
                    stage_id=item["stage_id"],
                    arm=item["arm"],
                    object_id=oid,
                    object_width_m=round(width, 4),
                    message=msg,
                )
            )
        elif guard.get("min_aperture_m") is not None and float(guard["min_aperture_m"]) > width + 0.002:
            guard_status = "warn_guard_exceeds_object_width"
            msg = (
                f"Stage '{item['stage_id']}' min_aperture_m={float(guard['min_aperture_m'])*1000:.1f}mm exceeds "
                f"{oid} estimated width={width*1000:.1f}mm."
            )
            stage_warnings.append(msg)
            stage_advisories.append(
                dict(
                    code="warn_guard_exceeds_object_width",
                    stage_id=item["stage_id"],
                    arm=item["arm"],
                    object_id=oid,
                    min_aperture_m=round(float(guard["min_aperture_m"]), 4),
                    object_width_m=round(width, 4),
                    message=msg,
                )
            )

        # 调用虚拟空间模拟场执行双指三维实体净空与咬合几何评估（仅作为中性参考，不产生阻断）
        grasp_eval = vf.evaluate_grasp(
            arm=item["arm"],
            position_m=[gx, gy, gz],
            angles_deg=[pitch, roll, yaw],
            target_id=oid,
            open_aperture_m=vf.jaw_geometry.max_opening_m,
            contact_region=item.get("contact_region", "pad_center"),
            contact_depth_m=item.get("contact_depth_m"),
        )
        colliding_neighbors = grasp_eval.get("colliding_neighbor_ids") or []
        neighbor_clearance = grasp_eval.get("neighbor_clearance_m")
        finger_collision_status = "ok"
        if colliding_neighbors and isinstance(neighbor_clearance, (int, float)) and neighbor_clearance < -0.005:
            finger_collision_status = "warn_finger_collides_neighbor"
            msg = (
                f"Stage '{item['stage_id']}' open-jaw box overlaps neighboring 3D object(s) {', '.join(colliding_neighbors)} "
                f"(neighbor_clearance_m={neighbor_clearance*1000:.1f}mm)."
            )
            stage_warnings.append(msg)
            stage_advisories.append(
                dict(
                    code="warn_finger_collides_neighbor",
                    stage_id=item["stage_id"],
                    arm=item["arm"],
                    object_id=oid,
                    colliding_neighbor_ids=colliding_neighbors,
                    neighbor_clearance_m=neighbor_clearance,
                    message=msg,
                )
            )

        bite_status = "ok"

        stage_risk_flags = [
            s
            for s in (closed_jaw_status, axis_status, height_status, guard_status, finger_collision_status, bite_status)
            if s != "ok"
        ]
        warnings.extend(stage_warnings)
        advisory_notes.extend(stage_advisories)
        stage_audits.append(
            dict(
                stage_id=item["stage_id"],
                arm=item["arm"],
                matched_object_id=oid,
                xy_distance_m=round(xy_dist, 4),
                object_length_m=round(length, 3),
                object_width_m=round(width, 3),
                object_yaw_deg=round(obj_yaw, 1),
                planned_yaw_deg=round(yaw, 1),
                closing_to_long_axis_deg=closing_angle,
                recommended_across_yaw_deg=rec_yaw,
                axis_alignment=axis_status,
                planned_z_m=round(gz, 4),
                object_top_z_m=round(float(top_z), 4) if isinstance(top_z, (int, float)) else None,
                recommended_pad_z_m=round(pad_z, 4),
                fingertip_support_clearance_m=fingertip_clearance,
                height_check=height_status,
                grip_guard_check=guard_status,
                finger_collision_check=finger_collision_status,
                bite_check=bite_status,
                closed_jaw_check=closed_jaw_status,
                theoretically_graspable=grasp_eval["theoretically_graspable"],
                vertical_bite_m=grasp_eval["vertical_bite_m"],
                lateral_eccentricity_m=grasp_eval["lateral_eccentricity_m"],
                neighbor_clearance_m=neighbor_clearance,
                colliding_neighbor_ids=colliding_neighbors,
                risk_flags=stage_risk_flags,
                blocking_risks=stage_blocking,
                advisory_notes=stage_advisories,
                warnings=stage_warnings,
            )
        )
        targets_3d.append(
            dict(
                id=oid,
                centre_m=centre,
                length_m=length,
                width_m=width,
                yaw_deg=obj_yaw,
                top_z_m=float(top_z) if isinstance(top_z, (int, float)) else centre[2],
            )
        )
    all_risk_flags = [flag for stage in stage_audits for flag in stage.get("risk_flags", [])]
    already_flagged_closed_stages = {
        r["stage_id"] for r in advisory_notes if r.get("code") == "warn_closed_jaw_descent"
    }
    arm_opened = set()
    closed_empty_stage_ids: set[str] = set()
    for st in sim_stages:
        if not isinstance(st, dict):
            continue
        s_arm = st.get("arm")
        s_goals = _stage_jaw_goals(st)
        if s_arm in closed_empty_arms and s_arm not in arm_opened and st.get("kind") == "move":
            closed_empty_stage_ids.add(str(st.get("id", "")))
        if s_goals and s_goals[-1] >= 0.5 and s_arm in profile.arms:
            arm_opened.add(s_arm)

    clean_stage_reports: list[dict] = []
    for sr in sim_report.get("stage_reports") or []:
        sid = str(sr.get("stage_id", ""))
        held_ghost = stage_held_oid.get(sid)
        col_objs = [o for o in (sr.get("colliding_objects") or []) if o != held_ghost]
        obj_clr_map = sr.get("min_clearance_by_object_m") or {}
        if held_ghost and held_ghost in (sr.get("colliding_objects") or []):
            rem_clrs = [float(obj_clr_map[o]) for o in col_objs if isinstance(obj_clr_map.get(o), (int, float))]
            if "table" in col_objs and isinstance(sr.get("path_clearance_m"), (int, float)):
                rem_clrs.append(float(sr["path_clearance_m"]))
            path_clr = min(rem_clrs) if rem_clrs else None
        else:
            path_clr = sr.get("path_clearance_m")

        sr_clean = {
            k: v
            for k, v in sr.items()
            if k not in ("recommended_safe_z_m",)
        }
        sr_clean["colliding_objects"] = col_objs
        if not col_objs and sr_clean.get("status") == "path_collision":
            sr_clean["status"] = "ok"
        clean_stage_reports.append(sr_clean)

        if sr_clean.get("status") not in ("path_collision", "grasp_risk") or not col_objs or not isinstance(path_clr, (int, float)):
            continue
        non_table_objs = [o for o in col_objs if o != "table"]
        if sid in closed_empty_stage_ids and non_table_objs and sid not in already_flagged_closed_stages and path_clr < -0.002:
            if "warn_closed_jaw_descent" not in all_risk_flags:
                all_risk_flags.append("warn_closed_jaw_descent")
            msg = (
                f"Stage '{sid}' moves with closed {sr.get('arm')} jaws near/into 3D object(s) {', '.join(non_table_objs)} "
                f"(min_surface_clearance={path_clr*1000:.1f}mm)."
            )
            warnings.append(msg)
            advisory_notes.append(
                dict(
                    code="warn_closed_jaw_descent",
                    stage_id=sid,
                    arm=sr.get("arm"),
                    colliding_objects=non_table_objs,
                    path_clearance_m=round(float(path_clr), 4),
                    message=msg,
                )
            )
            continue
        if path_clr < -0.005:
            if "warn_transport_collision" not in all_risk_flags:
                all_risk_flags.append("warn_transport_collision")
            msg = (
                f"Stage '{sid}' trajectory overlaps estimated 3D surface(s) {', '.join(col_objs)} "
                f"(min_surface_clearance={path_clr*1000:.1f}mm)."
            )
            warnings.append(msg)
            advisory_notes.append(
                dict(
                    code="warn_transport_collision",
                    stage_id=sid,
                    arm=sr.get("arm"),
                    colliding_objects=col_objs,
                    path_clearance_m=round(float(path_clr), 4),
                    message=msg,
                )
            )
    if not stage_audits and not warnings and not blocking_risks and not advisory_notes:
        return None, []
    severity = "advisory" if advisory_notes else "clean"
    any_sim_risk = any(r.get("status") not in (None, "ok") for r in clean_stage_reports)
    review = dict(
        observation_id=obs_id,
        status="warnings_detected" if warnings else "aligned",
        severity=severity,
        risk_flags=all_risk_flags,
        blocking_risk_flags=list(dict.fromkeys(r["code"] for r in blocking_risks)),
        advisory_flags=list(dict.fromkeys(n["code"] for n in advisory_notes)),
        blocking_risks=blocking_risks,
        advisory_notes=advisory_notes,
        stages=stage_audits,
        warnings=warnings,
        virtual_field_simulation=dict(
            status="risk_detected" if any_sim_risk else "clean",
            theoretically_valid=not any_sim_risk,
            min_path_clearance_m=sim_report["min_path_clearance_m"],
            transport_clearance_m=sim_report["min_path_clearance_m"],
            min_clearance_by_object_m=sim_report.get("min_clearance_by_object_m", {}),
            stage_reports=clean_stage_reports,
        ),
    )
    return review, targets_3d
