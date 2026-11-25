"""实测三维轨迹紧凑纯数值摘要生成模块（供无图动作回执与诊断回执共享）。

仅消费已落盘的公开关节测量、URDF 正运动学与规划边界，不访问后端或特权仿真状态。
所有空间长度与坐标均采用声明坐标系（默认 env_origin），物理单位为米（m）。
"""
import numpy as np

from .rollout_tracker import build_process_trajectory
from ...robot.model.embodiments import ARX_X5_PROFILE
from ...robot.motion.trajectory_evidence import ARMS

GRASP_CENTER_OFFSET_M = ARX_X5_PROFILE.gripper.grasp_reference_offset_m

# 用于计算双臂运动净空的活动连杆集合（排除固定躯干底座 base_link/link1）
_MOVING_LINKS = ("link2", "link3", "link4", "link5", "link6", "link7", "link8")


def _round_vec(vec, digits=4):
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if not np.isfinite(arr).all():
        return None
    return [round(float(v), digits) for v in arr]


def _round_scalar(val, digits=4):
    if val is None or not np.isfinite(val):
        return None
    return round(float(val), digits)


def _tcp_position_m(arm_state, tcp="link6"):
    """提取单步单臂的 TCP 三维位置（单位 m），优先使用公开 FK，其次使用测量 ee_pose。"""
    if not isinstance(arm_state, dict):
        return None
    pose = arm_state.get(f"{tcp}_pose")
    if isinstance(pose, dict) and "position_m" in pose:
        arr = np.asarray(pose["position_m"], dtype=float)
        if arr.shape == (3,) and np.isfinite(arr).all():
            return arr
    links = arm_state.get("link_positions_m")
    if isinstance(links, dict) and tcp in links:
        arr = np.asarray(links[tcp], dtype=float)
        if arr.shape == (3,) and np.isfinite(arr).all():
            return arr
    ee_pose = arm_state.get("measured_ee_pose")
    if isinstance(ee_pose, dict) and "position_m" in ee_pose:
        arr = np.asarray(ee_pose["position_m"], dtype=float)
        if arr.shape == (3,) and np.isfinite(arr).all():
            return arr
    return None


def _link6_position_m(arm_state):
    """兼容别名：提取单步单臂的 link6 三维位置（单位 m）。"""
    return _tcp_position_m(arm_state, "link6")


def _grasp_center_m(arm_state, tcp="link6", offset_m=GRASP_CENTER_OFFSET_M):
    """提取单步单臂的指垫夹持中心三维坐标（单位 m）。"""
    if not isinstance(arm_state, dict) or offset_m is None:
        return None
    pose = arm_state.get(f"{tcp}_pose")
    if not isinstance(pose, dict) or "position_m" not in pose or "rotation_matrix" not in pose:
        return None
    origin = np.asarray(pose["position_m"], dtype=float)
    rot = np.asarray(pose["rotation_matrix"], dtype=float)
    if origin.shape != (3,) or rot.shape != (3, 3) or not (np.isfinite(origin).all() and np.isfinite(rot).all()):
        return None
    return origin + float(offset_m) * rot[:, 0]


def _measured_aperture_m(arm_state):
    """从实测夹爪关节行程提取物理双指开度（单位 m，负向 PD 过压钳位至物理零点 0.0m）。"""
    if not isinstance(arm_state, dict):
        return None
    jaw = arm_state.get("measured_jaw")
    if not isinstance(jaw, dict):
        return None
    pos = jaw.get("position_m")
    if not isinstance(pos, (list, tuple, np.ndarray)) or len(pos) == 0:
        return None
    arr = np.asarray(pos, dtype=float).reshape(-1)
    if not np.isfinite(arr).all():
        return None
    if arr.size == 1:
        return max(0.0, float(arr[0])) * 2.0
    return float(sum(max(0.0, float(v)) for v in arr[:2]))


def _arm_clearance_points(arm_state, tcp="link6", grasp_offset_m=GRASP_CENTER_OFFSET_M):
    """收集单臂在当前步参与双臂净空计算的三维关键点集合（单位 m）。"""
    pts = []
    links = (arm_state or {}).get("link_positions_m")
    if isinstance(links, dict):
        for name in _MOVING_LINKS:
            if name in links:
                arr = np.asarray(links[name], dtype=float)
                if arr.shape == (3,) and np.isfinite(arr).all():
                    pts.append(arr)
    gc = _grasp_center_m(arm_state, tcp=tcp, offset_m=grasp_offset_m)
    if gc is not None:
        pts.append(gc)
    if not pts:
        p_tcp = _tcp_position_m(arm_state, tcp)
        if p_tcp is not None:
            pts.append(p_tcp)
    return np.asarray(pts, dtype=float) if pts else None


def _divergence_stats(measured_by_step, planned_by_step, arm, steps, tcp="link6"):
    """计算指定步集合上规划与实测 TCP 的最大三维偏差范数（m）及峰值步信息。"""
    max_div = None
    peak_step = None
    peak_pos = None
    for step in steps:
        m_sample = measured_by_step.get(step)
        p_sample = planned_by_step.get(step)
        if m_sample is None or p_sample is None:
            continue
        p_meas = _tcp_position_m((m_sample.get("arms") or {}).get(arm), tcp)
        p_plan = _tcp_position_m((p_sample.get("arms") or {}).get(arm), tcp)
        if p_meas is None or p_plan is None:
            continue
        div = float(np.linalg.norm(p_meas - p_plan))
        if max_div is None or div > max_div:
            max_div = div
            peak_step = int(step)
            peak_pos = p_meas
    return _round_scalar(max_div), peak_step, _round_vec(peak_pos)


def build_compact_trajectory_summary(paths, action, receipt, *, profile=None):
    """构建实测三维运动摘要，提供各臂/各阶段起止坐标、峰值跟踪偏差、夹爪开度与双臂最小净空。"""
    actual = (paths or {}).get("measured") or {}
    samples = actual.get("samples") or []
    if not samples:
        return None
    arms_order = profile.arms if profile is not None else ARMS
    tcp = profile.tcp_link if profile is not None else "link6"
    default_frame = profile.frame_id if profile is not None else "env_origin"
    grasp_offset_m = (
        GRASP_CENTER_OFFSET_M
        if profile is None
        else (profile.gripper.grasp_reference_offset_m if profile.gripper.grasp_reference == "tcp_offset" else None)
    )
    use_process_tracker = (
        profile is None or (profile.clock == "simulated_steps" and "grasp_angles" in profile.pose_inputs)
    )
    planned_samples = ((paths or {}).get("planned") or {}).get("samples") or []
    measured_by_step = {s["native_step"]: s for s in samples if "native_step" in s}
    completed_limit = receipt.get("completed_native_steps", receipt.get("executed_steps", len(samples) - 1))
    planned_by_step = {
        s["native_step"]: s
        for s in planned_samples
        if "native_step" in s and s.get("execution_state") != "not_recorded_completed"
    }
    sync_steps = [s["native_step"] for s in samples if 0 < s.get("native_step", 0) <= completed_limit]
    if not sync_steps:
        sync_steps = [s["native_step"] for s in samples if "native_step" in s]

    arms_summary = {}
    for arm in arms_order:
        positions = [_tcp_position_m((s.get("arms") or {}).get(arm), tcp) for s in samples]
        valid_positions = [p for p in positions if p is not None]
        path_len = 0.0
        for prev, curr in zip(valid_positions, valid_positions[1:]):
            path_len += float(np.linalg.norm(curr - prev))
        start_state = (samples[0].get("arms") or {}).get(arm, {})
        end_state = (samples[-1].get("arms") or {}).get(arm, {})
        max_div, peak_step, peak_pos = _divergence_stats(measured_by_step, planned_by_step, arm, sync_steps, tcp=tcp)
        apertures = [_measured_aperture_m((s.get("arms") or {}).get(arm)) for s in samples]
        valid_ap = [a for a in apertures if a is not None]
        arm_entry = {
            f"start_{tcp}_m": _round_vec(_tcp_position_m(start_state, tcp)),
            f"end_{tcp}_m": _round_vec(_tcp_position_m(end_state, tcp)),
            "path_length_m": _round_scalar(path_len),
            "max_divergence_m": max_div,
            "peak_divergence_step": peak_step,
            "peak_divergence_position_m": peak_pos,
        }
        start_gc = _grasp_center_m(start_state, tcp=tcp, offset_m=grasp_offset_m)
        end_gc = _grasp_center_m(end_state, tcp=tcp, offset_m=grasp_offset_m)
        if start_gc is not None or end_gc is not None:
            arm_entry["start_grasp_center_m"] = _round_vec(start_gc)
            arm_entry["end_grasp_center_m"] = _round_vec(end_gc)
        if valid_ap:
            arm_entry["measured_aperture_m"] = dict(
                start_m=_round_scalar(apertures[0] if apertures[0] is not None else valid_ap[0]),
                end_m=_round_scalar(apertures[-1] if apertures[-1] is not None else valid_ap[-1]),
                min_m=_round_scalar(min(valid_ap)),
                max_m=_round_scalar(max(valid_ap)),
            )
        arms_summary[arm] = arm_entry

    # 计算双臂全程最小三维欧氏净空距离（m）
    min_clearance = None
    min_clearance_step = None
    if len(arms_order) >= 2 and "left" in arms_order and "right" in arms_order:
        for sample in samples:
            left_pts = _arm_clearance_points((sample.get("arms") or {}).get("left"), tcp=tcp, grasp_offset_m=grasp_offset_m)
            right_pts = _arm_clearance_points((sample.get("arms") or {}).get("right"), tcp=tcp, grasp_offset_m=grasp_offset_m)
            if left_pts is None or right_pts is None:
                continue
            diffs = left_pts[:, None, :] - right_pts[None, :, :]
            dist = float(np.sqrt(np.sum(diffs * diffs, axis=-1)).min())
            if min_clearance is None or dist < min_clearance:
                min_clearance = dist
                min_clearance_step = int(sample["native_step"])

    # 提取多阶段批次动作的逐阶段三维轨迹摘要
    stage_steps_by_id = {}
    for p_sample in planned_samples:
        step = p_sample.get("native_step")
        if step is None or step <= 0:
            continue
        for arm in arms_order:
            sid = ((p_sample.get("arms") or {}).get(arm) or {}).get("stage_id")
            if sid:
                stage_steps_by_id.setdefault(sid, []).append(int(step))

    stage_results = receipt.get("stage_results") if isinstance(receipt, dict) else None
    action_stages = action.get("stages") if isinstance(action, dict) and isinstance(action.get("stages"), list) else []
    ordered_stage_ids = []
    stage_meta = {}
    for item in action_stages:
        if isinstance(item, dict) and item.get("id"):
            sid = item["id"]
            ordered_stage_ids.append(sid)
            stage_meta[sid] = dict(arm=item.get("arm"), kind=item.get("kind"))
    if isinstance(stage_results, dict):
        for sid, info in stage_results.items():
            if sid not in stage_meta:
                ordered_stage_ids.append(sid)
                stage_meta[sid] = {}
            if isinstance(info, dict):
                stage_meta[sid].update(
                    arm=info.get("arm", stage_meta[sid].get("arm")),
                    kind=info.get("kind", stage_meta[sid].get("kind")),
                    status=info.get("status"),
                )

    last_measured_step = samples[-1]["native_step"]
    stages_out = []
    for sid in ordered_stage_ids:
        meta = stage_meta.get(sid, {})
        arm = meta.get("arm")
        steps = sorted(set(stage_steps_by_id.get(sid, [])))
        entry = dict(
            stage_id=sid,
            arm=arm,
            kind=meta.get("kind"),
            status=meta.get("status"),
        )
        if arm in arms_order and steps:
            start_boundary = max(0, steps[0] - 1)
            end_boundary = min(steps[-1], last_measured_step)
            s_state = (measured_by_step.get(start_boundary, samples[0]).get("arms") or {}).get(arm, {})
            e_state = (measured_by_step.get(end_boundary, samples[-1]).get("arms") or {}).get(arm, {})
            executed_stage_steps = [st for st in steps if st <= last_measured_step]
            s_div, s_peak_step, s_peak_pos = _divergence_stats(
                measured_by_step, planned_by_step, arm, executed_stage_steps, tcp=tcp
            )
            entry.update(
                start_step=start_boundary,
                end_step=end_boundary,
                start_position_m=_round_vec(_tcp_position_m(s_state, tcp)),
                end_position_m=_round_vec(_tcp_position_m(e_state, tcp)),
                max_divergence_m=s_div,
                peak_divergence_step=s_peak_step,
                peak_divergence_position_m=s_peak_pos,
                measured_aperture_end_m=_round_scalar(_measured_aperture_m(e_state)),
            )
            s_gc = _grasp_center_m(s_state, tcp=tcp, offset_m=grasp_offset_m)
            e_gc = _grasp_center_m(e_state, tcp=tcp, offset_m=grasp_offset_m)
            if s_gc is not None or e_gc is not None:
                entry["start_grasp_center_m"] = _round_vec(s_gc)
                entry["end_grasp_center_m"] = _round_vec(e_gc)
        elif isinstance(stage_results, dict) and isinstance(stage_results.get(sid), dict):
            grip_facts = stage_results[sid].get("grip_facts")
            if isinstance(grip_facts, dict) and ("measured_aperture_m" in grip_facts or "aperture_after_m" in grip_facts):
                # 优先读取实测孔径键 measured_aperture_m，兼容历史归档键 aperture_after_m
                ap_val = grip_facts.get("measured_aperture_m")
                if ap_val is None and "aperture_after_m" in grip_facts:
                    ap_val = grip_facts.get("aperture_after_m")
                entry["measured_aperture_end_m"] = _round_scalar(ap_val)
        stages_out.append(entry)

    result = dict(
        frame_id=actual.get("frame_id", default_frame),
        unit="m",
        executed_native_steps=int(actual.get("executed_native_steps", len(samples) - 1)),
        min_inter_arm_clearance_m=_round_scalar(min_clearance),
        min_clearance_step=min_clearance_step,
        arms=arms_summary,
    )
    if stages_out:
        result["stages"] = stages_out

    # 融合运动期 4D 过程轨迹（保留 Harness.execute 记录的过程观测帧 ID 与局部光流，并补充公开 FK 精度与偏差峰值）
    if not use_process_tracker:
        return result
    existing_pt = receipt.get("process_trajectory") if isinstance(receipt, dict) else None
    published_obs = []
    if isinstance(existing_pt, dict):
        for kf in existing_pt.get("event_keyframes") or []:
            if isinstance(kf, dict) and isinstance(kf.get("executed_step"), int) and isinstance(kf.get("observation_id"), str):
                published_obs.append((kf["executed_step"], kf["observation_id"]))
        for co in (existing_pt.get("co_motion") or {}).values():
            if isinstance(co, dict) and isinstance(co.get("slip_onset_step"), int) and isinstance(co.get("slip_observation_id"), str):
                published_obs.append((co["slip_onset_step"], co["slip_observation_id"]))
    if isinstance(receipt, dict):
        if isinstance(receipt.get("execution_start_observation_id"), str):
            published_obs.append((0, receipt["execution_start_observation_id"]))
        if isinstance(receipt.get("result_observation_id"), str):
            published_obs.append((last_measured_step, receipt["result_observation_id"]))
    peak_div_by_arm = {
        arm: (arms_summary[arm].get("max_divergence_m"), arms_summary[arm].get("peak_divergence_step"))
        for arm in arms_order
        if arm in arms_summary
    }
    process_traj = build_process_trajectory(
        samples,
        action=action,
        receipt=receipt,
        planned_samples=planned_samples,
        published_observations=published_obs or None,
        peak_divergence_by_arm=peak_div_by_arm,
    )
    if isinstance(process_traj, dict):
        if isinstance(existing_pt, dict) and isinstance(existing_pt.get("co_motion"), dict):
            for arm_key, prev_co in existing_pt["co_motion"].items():
                if (
                    isinstance(prev_co, dict)
                    and "fingertip_flow_px" in prev_co
                    and arm_key in process_traj.get("co_motion", {})
                ):
                    process_traj["co_motion"][arm_key]["fingertip_flow_px"] = prev_co["fingertip_flow_px"]
        result["process_trajectory"] = process_traj
    return result
