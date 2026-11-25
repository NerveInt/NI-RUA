"""显式选择阶段续跑与局部重编译辅助模块。

当多阶段批次在某个阶段暂停或失败时，保留原计划的阶段定义与已完成阶段记录，
由调用方显式指定 source_plan_id、start_from_stage_id 及可选 stage_overrides，
基于最新观测的实测状态重新编译剩余阶段，不做静默跳过或后台暗自重算。
"""
from copy import deepcopy

from ...core.contracts import IDENTITY
from ...robot.motion.batch_continuation import (
    _apply_compiled_stage_override,
    _normalize_stage_id,
    apply_stage_override,
    build_next_action,
    extract_and_override_stages,
    normalize_stage_overrides,
    resolve_backend_continuation,
    summarize_stage_progress,
)
from ...robot.motion.grip_guard import CARRY_SLIP_GUARD_FIELDS
from ...robot.motion.motion_tolerance import SPLIT_ORIENTATION_FIELDS


def lookup_source_plan(harness, source_plan_id):
    """在当前运行缓存或持久化记录中查找源计划，校验环境身份。"""
    records = getattr(harness, 'unified_action_plans', {})
    record = records.get(source_plan_id)
    if record is None:
        record = next((item for item in records.values()
                       if item.get('native_plan_id') == source_plan_id), None)
    if record is None and hasattr(harness, 'store'):
        try:
            record = harness.store.read('unified_action_plan', source_plan_id)
        except KeyError:
            record = None
    if record is None:
        raise ValueError('robot_action_plan_unknown_or_expired')
    if any(record['identity'].get(key) != harness.identity[key] for key in IDENTITY):
        raise ValueError('stale_environment_identity')
    return record


def resolve_continuation_args(harness, args):
    """解析 robot_action 的 source_plan_id 续跑参数，生成完整的阶段列表与溯源元信息。"""
    if 'source_plan_id' not in args:
        if 'start_from_stage_id' in args or 'stage_overrides' in args:
            raise ValueError('continuation_fields_require_source_plan_id')
        return deepcopy(args), None
    if args.get('stages') is not None:
        raise ValueError('choose_stages_or_source_plan_id')
    source_plan_id = args.get('source_plan_id')
    if not isinstance(source_plan_id, str) or not source_plan_id.strip():
        raise ValueError('robot_action_plan_unknown_or_expired')
    start_from_stage_id = args.get('start_from_stage_id')
    if not isinstance(start_from_stage_id, str) or not start_from_stage_id.strip():
        raise ValueError('source_plan_continuation_requires_start_from_stage_id')
    source_record = lookup_source_plan(harness, source_plan_id)
    source_stages = source_record.get('source_stages')
    if not isinstance(source_stages, list) or not source_stages:
        source_stages = source_record.get('action', {}).get('stages', [])
    remaining, overridden_ids = extract_and_override_stages(
        source_stages, start_from_stage_id, args.get('stage_overrides'))
    resolved = deepcopy(args)
    resolved.pop('source_plan_id', None)
    resolved.pop('start_from_stage_id', None)
    resolved.pop('stage_overrides', None)
    resolved['stages'] = remaining
    source_action = source_record.get('action', {})
    source_args = {
        **{k: source_action[k] for k in CARRY_SLIP_GUARD_FIELDS if isinstance(source_action, dict) and k in source_action},
        **(source_record.get('source_args') or {}),
    }
    if 'checkpoint_after_gripper' not in resolved and 'checkpoint_after_gripper' in source_action:
        resolved['checkpoint_after_gripper'] = source_action['checkpoint_after_gripper']
    if 'allow_jaw_symmetry' not in resolved and 'allow_jaw_symmetry' in source_action:
        resolved['allow_jaw_symmetry'] = source_action['allow_jaw_symmetry']
    if 'allow_approach_tilt' not in resolved and 'allow_approach_tilt' in source_action:
        resolved['allow_approach_tilt'] = source_action['allow_approach_tilt']
    for key in (
        'checkpoint_after_gripper',
        'allow_jaw_symmetry',
        'allow_approach_tilt',
        'tolerance_preset', 'position_tolerance_m', 'orientation_tolerance_rad', 'max_upward_dz_m', 'max_downward_dz_m',
        *SPLIT_ORIENTATION_FIELDS,
        'stall_guard_preset', 'max_tracking_divergence_m',
        'arrival_correction_rounds', 'arrival_correction_max_offset_rad',
        'grip_guard_preset', 'min_aperture_m', 'max_aperture_m',
        'joint_travel_preset', 'max_joint_travel_rad', 'max_backtracking_rad',
        'obstacle_object_ids', 'obstacle_policy', 'ignore_obstacle_ids',
    ):
        if key not in resolved and key in source_args:
            resolved[key] = deepcopy(source_args[key])
    # 防滑脱三键逐键继承；仅当调用方显式传 carry_slip_guard=False 且未手传从属键时不继承从属键
    if 'carry_slip_guard' not in resolved and 'carry_slip_guard' in source_args:
        resolved['carry_slip_guard'] = deepcopy(source_args['carry_slip_guard'])
    if args.get('carry_slip_guard') is not False:
        for key in ('carry_slip_scope', 'slip_consecutive_steps'):
            if key not in resolved and key in source_args:
                resolved[key] = deepcopy(source_args[key])
    for group in (
        ('target_exemption_preset', 'target_exemption_radius_m'),
        ('clearance_preset', 'clearance_height_m'),
    ):
        if not any(k in resolved for k in group):
            for k in group:
                if k in source_args:
                    resolved[k] = deepcopy(source_args[k])
    continuation_meta = dict(
        source_plan_id=source_plan_id,
        start_from_stage_id=start_from_stage_id,
        overridden_stage_ids=overridden_ids,
        remaining_stage_ids=[stage['id'] for stage in remaining],
        source_completed_stage_ids=deepcopy(source_record.get('completed_stage_ids', [])),
    )
    return resolved, continuation_meta


__all__ = [
    "_apply_compiled_stage_override",
    "_normalize_stage_id",
    "apply_stage_override",
    "build_next_action",
    "extract_and_override_stages",
    "lookup_source_plan",
    "normalize_stage_overrides",
    "resolve_backend_continuation",
    "resolve_continuation_args",
    "summarize_stage_progress",
]
