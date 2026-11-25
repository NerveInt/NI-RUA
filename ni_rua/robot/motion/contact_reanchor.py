"""接触即停（contact_stop）阶段驻停后的后续阶段起点重锚定与有界保形插值模块。

当显式配置 `stall_guard_preset == "contact_stop"` 的移动阶段在合法接触顺应窗口内提前驻停于实测关节角 `held_q` 时：
1. 将当前移动阶段剩余步及紧邻同臂纯夹爪（`gripper`）阶段的保持关节角同步更新为 `held_q`；
2. 在保持预分配共享时间步窗口 `[start, end)` 与原碰撞自由轨迹曲率不变的前提下，将起点残差 `held_q - nominal_start_q` 沿窗口线性衰减至 `0`（精确锚定终点 `target_q`），并严格遵守请求的阶段速度上限（绝不静默放大 `speed_scale < 2.0`）；
3. 仅当缓存指令行实际发生修改时才记录 `original_rows_q` 并在 `has_reanchored_rows` 中返回 `True`。
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .batch_scheduler import ArmLayout

EXACT_CACHED_SEMANTICS = "exact cached native rows; no replanning, resampling or jaw rebase"
REANCHORED_CACHED_SEMANTICS = (
    "cached native rows with contact_stop hold/start-residual reanchoring; no replanning or jaw rebase"
)


def has_reanchored_rows(records: dict[str, dict]) -> bool:
    """Return True only when cached rows were actually modified via contact_stop hold or start-residual reanchoring."""
    return any(
        isinstance(record, dict) and bool(record.get("original_rows_q"))
        for record in records.values()
    )


def arm_row_offset(arm: str, *, layout: ArmLayout | None = None) -> int:
    """返回原生指令行中指定手臂主动关节段的起始列下标（支持任意 `ArmLayout`，默认双臂 ARX-X5）。"""
    if layout is not None:
        return layout.offset(arm)
    if arm == "left":
        return 0
    if arm == "right":
        return 7
    raise ValueError(f"invalid_arm_name:{arm}")


def clamp_stage_rows_to_held_q(
    record: dict,
    rows: list[list[float]],
    start_cursor: int,
    held_q: list[float],
    offset: int,
    *,
    dof: int | None = None,
) -> bool:
    """将阶段自 `start_cursor` 起的共享指令行与阶段局部指令行关节目标统一锁定为 `held_q`；返回是否实际改写了任何行。"""
    stage_start = int(record["start"])
    stage_end = int(record["end"])
    q_values = [float(x) for x in held_q]
    n_joints = len(q_values) if dof is None else int(dof)
    if len(q_values) != n_joints or n_joints < 1:
        raise ValueError("invalid_contact_held_q")

    first_idx = max(stage_start, int(start_cursor))
    if first_idx >= stage_end:
        return False

    joint_slice = slice(offset, offset + n_joints)
    will_modify = False
    for r_idx in range(first_idx, stage_end):
        if not np.allclose(rows[r_idx][joint_slice], q_values, atol=1e-12, rtol=0.0):
            will_modify = True
            break
    if not will_modify:
        return False

    record.setdefault("original_rows_q", [list(r[joint_slice]) for r in record["rows"]])
    for r_idx in range(first_idx, stage_end):
        rows[r_idx][joint_slice] = list(q_values)
        local_idx = r_idx - stage_start
        if 0 <= local_idx < len(record["rows"]):
            record["rows"][local_idx][joint_slice] = list(q_values)
    return True


def sync_reanchor_stage_result(stage_result: dict, record: dict) -> None:
    """将阶段记录上的接触重锚定与开爪姿态解脱元数据同步投影至公开阶段回执字典。"""
    if record.get("reanchored_from_contact") is True and isinstance(record.get("contact_reanchor"), dict):
        stage_result["reanchored_from_contact"] = True
        stage_result["contact_reanchor"] = deepcopy(record["contact_reanchor"])
    elif isinstance(record.get("contact_reanchor_error"), dict):
        stage_result["reanchored_from_contact"] = False
        stage_result["contact_reanchor"] = deepcopy(record["contact_reanchor_error"]["details"])
    if record.get("jaw_symmetry_resolved") is True and isinstance(record.get("resolved_angles_deg"), (list, tuple)):
        stage_result["jaw_symmetry_resolved"] = True
        stage_result["resolved_angles_deg"] = [float(v) for v in record["resolved_angles_deg"]]
        if record.get("approach_tilt_resolved") is True:
            stage_result["approach_tilt_resolved"] = True


def _waterfill_bounded(slack: np.ndarray, amount: float) -> np.ndarray:
    """在非负步长余量向量 `slack` 上做精确注水分配，使分配量之和等于 `amount` 且最小化单步峰值扰动。"""
    n = int(slack.shape[0])
    if n == 0 or amount <= 1e-12:
        return np.zeros(n, dtype=float)
    total_slack = float(np.sum(slack))
    if total_slack <= amount + 1e-12:
        return slack.copy()
    sv = np.sort(slack)
    prev = 0.0
    acc = 0.0
    level = float(sv[-1])
    for idx in range(n):
        rem_cnt = n - idx
        span = (float(sv[idx]) - prev) * rem_cnt
        if acc + span >= amount - 1e-12:
            level = prev + max(0.0, amount - acc) / rem_cnt
            break
        acc += span
        prev = float(sv[idx])
    return np.minimum(slack, level)


def _project_scurve_preserving_deltas(
    raw_deltas: np.ndarray,
    step_limits: np.ndarray,
) -> np.ndarray:
    """逐关节裁剪局部巡航超限峰值并将差额优先回填至内点步长余量，完整保留 S-curve 末端平滑减速与避障弧度。"""
    window_steps, n_joints = raw_deltas.shape
    out = raw_deltas.copy()
    if window_steps >= 5:
        interior_mask = np.zeros(window_steps, dtype=bool)
        interior_mask[1 : window_steps - 2] = True
    elif window_steps >= 3:
        interior_mask = np.zeros(window_steps, dtype=bool)
        interior_mask[1 : window_steps - 1] = True
    else:
        interior_mask = np.ones(window_steps, dtype=bool)
    boundary_mask = ~interior_mask

    for j in range(n_joints):
        lim = float(step_limits[j])
        dj = raw_deltas[:, j]
        if float(np.max(np.abs(dj))) <= lim + 1e-9:
            continue
        clipped = np.clip(dj, -lim, lim)
        rem = float(np.sum(dj - clipped))
        if abs(rem) <= 1e-12:
            out[:, j] = clipped
            continue
        sign_r = 1.0 if rem > 0.0 else -1.0
        need = abs(rem)
        slack = np.maximum(0.0, lim - sign_r * clipped)
        if float(np.sum(slack)) < need - 1e-9:
            # 净关节跨度超出窗口总步长容量：保留原始超限增量，交由下游门控诚实拒绝
            continue
        corr = np.zeros(window_steps, dtype=float)
        int_slack = slack[interior_mask]
        take_int = min(need, float(np.sum(int_slack)))
        if take_int > 1e-12:
            corr[interior_mask] = _waterfill_bounded(int_slack, take_int)
        rem_after = need - float(np.sum(corr[interior_mask]))
        if rem_after > 1e-12 and np.any(boundary_mask):
            corr[boundary_mask] = _waterfill_bounded(slack[boundary_mask], rem_after)
        out[:, j] = clipped + sign_r * corr
    return out


def reanchor_move_stage_from_contact(
    adapter,
    arm: str,
    source_stage_id: str,
    source_record: dict,
    next_record: dict,
    rows: list[list[float]],
    held_q: list[float],
    held_pos,
    *,
    default_pos_tol: float = 0.010,
    layout: ArmLayout | None = None,
) -> dict | None:
    """对紧随 `contact_stop` 阶段之后的同臂 `move` 阶段执行安全校验与起点平滑重插值（支持任意 `N`-DoF 与 `ArmLayout`）。"""
    offset = arm_row_offset(arm, layout=layout)
    nominal_start_raw = next_record.get("original_expected_start_q", next_record["expected_start_q"])
    nominal_start_q = np.asarray(nominal_start_raw, dtype=float)
    if nominal_start_q.ndim != 1 or len(nominal_start_q) < 1 or not np.isfinite(nominal_start_q).all():
        raise ValueError("invalid_nominal_start_q")
    dof = layout.dof(arm) if layout is not None else int(nominal_start_q.shape[0])
    if nominal_start_q.shape != (dof,):
        raise ValueError("invalid_nominal_start_q")
    joint_slice = slice(offset, offset + dof)

    held_q_arr = np.asarray(held_q, dtype=float)
    held_pos_arr = np.asarray(held_pos, dtype=float)
    if held_q_arr.shape != (dof,) or not np.isfinite(held_q_arr).all():
        raise ValueError("invalid_contact_held_q")
    if held_pos_arr.shape != (3,) or not np.isfinite(held_pos_arr).all():
        raise ValueError("invalid_contact_held_position")

    max_start_delta = float(np.max(np.abs(held_q_arr - nominal_start_q)))
    # 若前置阶段既未中途早停且实测终点关节与原计划起点几乎重合，保留原规划轨迹不改写（绝不写入 original_rows_q）
    if not source_record.get("contact_stopped_early") and max_start_delta <= 1e-5:
        next_record.pop("contact_reanchor_error", None)
        return None

    orig_rows_q = next_record.get("original_rows_q") or [list(r[joint_slice]) for r in next_record["rows"]]
    orig_q = np.asarray(orig_rows_q, dtype=float)
    target_q_arr = np.asarray(orig_q[-1], dtype=float)
    window_steps = int(next_record["end"] - next_record["start"])
    if window_steps < 1 or orig_q.shape != (window_steps, dof):
        raise ValueError("invalid_move_stage_window_steps")

    base_details = dict(
        source_stage_id=source_stage_id,
        nominal_start_q_rad=[round(float(x), 6) for x in nominal_start_q],
        actual_start_q_rad=[round(float(x), 6) for x in held_q_arr],
        max_start_joint_delta_rad=round(max_start_delta, 6),
        actual_start_position_m=[round(float(x), 5) for x in held_pos_arr],
        allocated_steps=window_steps,
    )

    # 沿原碰撞自由轨迹做端点锚定残差线性衰减，完整保留 cuRobo 避障几何曲率与 S-curve 末端减速段
    start_residual = held_q_arr - nominal_start_q
    weights = 1.0 - (np.arange(1, window_steps + 1, dtype=float)[:, None] / window_steps)
    raw_reanchored_q = orig_q + weights * start_residual[None, :]
    raw_reanchored_q[-1] = target_q_arr

    action_dict = next_record.get("action") or {}
    speed_scale = (
        float(adapter._waypoint_speed_scale(action_dict, 0))
        if hasattr(adapter, "_waypoint_speed_scale")
        else float(action_dict.get("speed_scale", 1.0))
    )
    # 严格遵守调用方显式请求的 speed_scale（绝不通过 max(speed_scale, 2.0) 静默提速）：
    # - 当 speed_scale < 1.0（精细慢速阶段）时，严格使用 hw_step_limits；
    # - 当 speed_scale >= 1.0 时，允许不超过原已审阅 S-curve 轨迹自身的峰值巡航步长 orig_peak_step。
    hw_step_limits = np.asarray(adapter._speed_step_limits(arm, speed_scale), dtype=float).reshape(dof)
    if speed_scale < 1.0 - 1e-9:
        step_limits = hw_step_limits
    else:
        orig_peak_step = np.max(np.abs(np.diff(np.vstack([nominal_start_q, orig_q]), axis=0)), axis=0)
        step_limits = np.maximum(hw_step_limits, orig_peak_step)

    raw_deltas = np.diff(np.vstack([held_q_arr, raw_reanchored_q]), axis=0)
    raw_step_delta = np.max(np.abs(raw_deltas), axis=0)
    if not bool(np.any(raw_step_delta > step_limits + 1e-9)):
        reanchored_q = raw_reanchored_q
    else:
        projected_deltas = _project_scurve_preserving_deltas(raw_deltas, step_limits)
        reanchored_q = held_q_arr[None, :] + np.cumsum(projected_deltas, axis=0)
        if not bool(np.any(np.max(np.abs(projected_deltas), axis=0) > step_limits + 1e-9)):
            reanchored_q[-1] = target_q_arr

    final_deltas = np.diff(np.vstack([held_q_arr, reanchored_q]), axis=0)
    per_step_delta = np.max(np.abs(final_deltas), axis=0)
    per_joint_cost = np.sum(np.abs(final_deltas), axis=0) / np.maximum(step_limits, 1e-12)
    required_steps = max(1, int(math.ceil(float(np.max(per_joint_cost)) - 1e-9)))

    # 步长限幅门控：重锚定轨迹相邻步最大增量不得超出物理速度步长上限；拒绝时绝不改写缓存行或设置 original_rows_q
    if bool(np.any(per_step_delta > step_limits + 1e-9)):
        step_ratios = per_step_delta / np.maximum(step_limits, 1e-12)
        required_steps = max(
            required_steps,
            window_steps + 1,
            int(math.ceil(window_steps * float(np.max(step_ratios)) - 1e-9)),
        )
        err_details = dict(
            base_details,
            status="rejected",
            reason="contact_reanchor_step_budget_exceeded",
            required_min_steps=required_steps,
            max_step_delta_rad=[round(float(x), 6) for x in per_step_delta],
            step_limits_rad=[round(float(x), 6) for x in step_limits],
        )
        next_record.pop("reanchored_from_contact", None)
        next_record.pop("contact_reanchor", None)
        next_record["contact_reanchor_error"] = dict(
            reason="contact_reanchor_step_budget_exceeded",
            details=err_details,
        )
        return err_details

    next_record.pop("contact_reanchor_error", None)
    next_record.setdefault("original_expected_start_q", [float(x) for x in nominal_start_q])
    if not np.allclose(reanchored_q, orig_q, atol=1e-12, rtol=0.0):
        next_record.setdefault("original_rows_q", orig_rows_q)
    if "planned_positions_m" in next_record:
        next_record.setdefault("original_planned_positions_m", deepcopy(next_record["planned_positions_m"]))

    next_record["expected_start_q"] = [float(x) for x in held_q_arr]
    stage_start = int(next_record["start"])
    for idx in range(window_steps):
        q_list = [float(x) for x in reanchored_q[idx]]
        rows[stage_start + idx][joint_slice] = q_list
        next_record["rows"][idx][joint_slice] = list(q_list)

    orig_planned = next_record.get("original_planned_positions_m")
    if isinstance(orig_planned, list) and len(orig_planned) == window_steps:
        orig_pos_arr = np.asarray(orig_planned, dtype=float)
        src_target_pos = (source_record.get("target") or {}).get("position_m") if isinstance(source_record, dict) else None
        if isinstance(src_target_pos, (list, tuple)) and len(src_target_pos) == 3:
            nominal_start_pos = np.asarray(src_target_pos, dtype=float)
        elif window_steps > 1:
            nominal_start_pos = orig_pos_arr[0] - (orig_pos_arr[-1] - orig_pos_arr[0]) / (window_steps - 1)
        else:
            nominal_start_pos = orig_pos_arr[0]
        pos_residual = held_pos_arr - nominal_start_pos
        reanchored_pos = orig_pos_arr + weights * pos_residual[None, :]
        reanchored_pos[-1] = orig_pos_arr[-1]
        next_record["planned_positions_m"] = reanchored_pos.tolist()

    reanchor_info = dict(
        base_details,
        status="reanchored",
        required_min_steps=required_steps,
    )
    next_record["reanchored_from_contact"] = True
    next_record["contact_reanchor"] = reanchor_info
    return reanchor_info


def propagate_contact_hold_and_reanchor(
    adapter,
    records: dict[str, dict],
    rows: list[list[float]],
    results: dict[str, dict],
    *,
    arm: str,
    after_step: int,
    held_q: list[float],
    held_pos,
    source_stage_id: str | None = None,
    source_record: dict | None = None,
    reanchor_next_move: bool = False,
    default_pos_tol: float = 0.010,
    layout: ArmLayout | None = None,
) -> None:
    """将接触驻停关节角传播至后续连续同臂 `gripper` 阶段，并可选平滑重锚定紧邻的下一同臂 `move` 阶段。"""
    offset = arm_row_offset(arm, layout=layout)
    dof = layout.dof(arm) if layout is not None else len(held_q)
    for other_id, other in sorted(records.items(), key=lambda item: item[1]["start"]):
        if other["start"] < after_step or other["kind"] == "observe" or other.get("arm") != arm:
            continue
        if other["kind"] == "gripper":
            other["expected_start_q"] = [float(x) for x in held_q]
            if reanchor_next_move and source_stage_id is not None:
                other["contact_anchor_source_id"] = source_stage_id
            clamp_stage_rows_to_held_q(other, rows, other["start"], held_q, offset, dof=dof)
            continue
        if other["kind"] == "move":
            if reanchor_next_move and source_stage_id is not None and source_record is not None:
                reanchor_move_stage_from_contact(
                    adapter,
                    arm,
                    source_stage_id,
                    source_record,
                    other,
                    rows,
                    held_q,
                    held_pos,
                    default_pos_tol=default_pos_tol,
                    layout=layout,
                )
                if other_id in results:
                    sync_reanchor_stage_result(results[other_id], other)
            break
