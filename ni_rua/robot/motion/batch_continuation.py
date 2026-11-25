"""Pure stage-continuation, override normalization, and backend plan-slice helpers (`batch_continuation.py`).

Extracted from ``ni_rua.runtime.actions.plan_continuation`` so ``ni_rua/robot/motion/`` resolves
backend ``source_plan_id`` continuations with zero upward imports into ``ni_rua.runtime``.
"""
from __future__ import annotations

from copy import deepcopy

from ...core.embodiment import JAW_GRASP_CENTRE_FIELDS, SPLIT_ORIENTATION_FIELDS


def normalize_stage_overrides(stage_overrides):
    """将字典或数组形式的 stage_overrides 归一化为 {stage_id: override_dict}。"""
    if stage_overrides is None:
        return {}
    normalized = {}
    if isinstance(stage_overrides, dict):
        for stage_id, patch in stage_overrides.items():
            if not isinstance(stage_id, str) or not stage_id.strip() or not isinstance(patch, dict):
                raise ValueError('invalid_stage_overrides')
            normalized[stage_id] = deepcopy(patch)
        return normalized
    if isinstance(stage_overrides, list):
        for item in stage_overrides:
            if not isinstance(item, dict):
                raise ValueError('invalid_stage_overrides')
            stage_id = item.get('stage_id') or item.get('id')
            if not isinstance(stage_id, str) or not stage_id.strip():
                raise ValueError('stage_override_requires_stage_id')
            if stage_id in normalized:
                raise ValueError(f'duplicate_stage_override:{stage_id}')
            patch = {key: deepcopy(value) for key, value in item.items()
                     if key not in ('stage_id', 'id', 'overrides')}
            if 'overrides' in item:
                if not isinstance(item['overrides'], dict):
                    raise ValueError('invalid_stage_overrides')
                patch.update(deepcopy(item['overrides']))
            normalized[stage_id] = patch
        return normalized
    raise ValueError('invalid_stage_overrides')


def _apply_compiled_stage_override(stage, patch):
    """对底层已编译的 BatchAction stage 应用局部覆盖。"""
    own = stage.get('action')
    arm = stage.get('arm')
    if isinstance(own, dict):
        path = own.get('path')
        waypoints = path.get('waypoints') if isinstance(path, dict) else None
        target = waypoints[0].get(arm) if waypoints and isinstance(waypoints[0], dict) else None
        if 'target_pose' in patch:
            pose = patch.pop('target_pose')
            if not isinstance(pose, dict) or target is None:
                raise ValueError('invalid_stage_override_target_pose')
            target.update(deepcopy(pose))
        for field in ('position_m', 'quaternion_xyzw', 'grasp_position_m', 'angles_deg'):
            if field in patch and target is not None:
                target[field] = deepcopy(patch.pop(field))
        if isinstance(path, dict):
            if 'tolerance_preset' in patch:
                path['tolerance_preset'] = patch['tolerance_preset']
                if 'position_tolerance_m' not in patch:
                    path.pop('position_tolerance_m', None)
                    stage.pop('position_tolerance_m', None)
                if 'orientation_tolerance_rad' not in patch:
                    path.pop('orientation_tolerance_rad', None)
                    stage.pop('orientation_tolerance_rad', None)
                if 'max_upward_dz_m' not in patch:
                    path.pop('max_upward_dz_m', None)
                    stage.pop('max_upward_dz_m', None)
                if 'max_downward_dz_m' not in patch:
                    path.pop('max_downward_dz_m', None)
                    stage.pop('max_downward_dz_m', None)
                for field in SPLIT_ORIENTATION_FIELDS:
                    if field not in patch:
                        path.pop(field, None)
                        stage.pop(field, None)
            if 'position_tolerance_m' in patch:
                path['position_tolerance_m'] = float(patch.pop('position_tolerance_m'))
            if 'orientation_tolerance_rad' in patch:
                path['orientation_tolerance_rad'] = float(patch.pop('orientation_tolerance_rad'))
            if 'max_upward_dz_m' in patch:
                path['max_upward_dz_m'] = float(patch.pop('max_upward_dz_m'))
            if 'max_downward_dz_m' in patch:
                path['max_downward_dz_m'] = float(patch.pop('max_downward_dz_m'))
            for field in SPLIT_ORIENTATION_FIELDS:
                if field in patch:
                    path[field] = float(patch.pop(field))
            if 'stall_guard_preset' in patch:
                path['stall_guard_preset'] = patch['stall_guard_preset']
                if 'max_tracking_divergence_m' not in patch:
                    path.pop('max_tracking_divergence_m', None)
                    stage.pop('max_tracking_divergence_m', None)
            if 'max_tracking_divergence_m' in patch:
                path['max_tracking_divergence_m'] = float(patch.pop('max_tracking_divergence_m'))
            if 'clearance_preset' in patch or 'clearance_height_m' in patch:
                from .scene_obstacles import DEFAULT_CLEARANCE_HEIGHT_M, resolve_stage_clearance
                clearance_patch = {
                    k: patch.pop(k) for k in ('clearance_preset', 'clearance_height_m') if k in patch
                }
                path['clearance_height_m'] = resolve_stage_clearance(
                    clearance_patch,
                    path.get('clearance_height_m', DEFAULT_CLEARANCE_HEIGHT_M),
                )
        if 'speed_scale' in patch:
            own['speed_scale'] = float(patch.pop('speed_scale'))
            own.pop('speed_preset', None)
        if 'speed_preset' in patch:
            from .motion_speed import SPEED_PRESETS
            preset = patch.pop('speed_preset')
            if preset not in SPEED_PRESETS:
                raise ValueError('invalid_speed_preset')
            own['speed_scale'] = SPEED_PRESETS[preset]
        if 'gripper_speed_scale' in patch:
            own['gripper_speed_scale'] = float(patch.pop('gripper_speed_scale'))
        if 'gripper_speed_preset' in patch:
            from .motion_speed import GRIPPER_SPEED_PRESETS
            preset = patch.pop('gripper_speed_preset')
            if preset not in GRIPPER_SPEED_PRESETS:
                raise ValueError('invalid_gripper_speed_preset')
            own['gripper_speed_scale'] = GRIPPER_SPEED_PRESETS[preset]
        if 'goal_normalized' in patch and own.get('gripper_events'):
            own['gripper_events'][0]['goal_normalized'] = float(patch.pop('goal_normalized'))
        if 'gripper_goal_normalized' in patch and own.get('gripper_events'):
            own['gripper_events'][0]['goal_normalized'] = float(patch.pop('gripper_goal_normalized'))
    # 显式关闭防滑脱监控且未同传从属参数时，自动清除原阶段继承的从属键
    if patch.get('carry_slip_guard') is False:
        if 'carry_slip_scope' not in patch:
            stage.pop('carry_slip_scope', None)
        if 'slip_consecutive_steps' not in patch:
            stage.pop('slip_consecutive_steps', None)
    for key, value in patch.items():
        if key in ('id', 'stage_id'):
            continue
        stage[key] = deepcopy(value)


def apply_stage_override(stage, override):
    """对单个阶段定义应用字段覆盖，自动处理互斥字段切换。"""
    if not isinstance(override, dict):
        raise ValueError('invalid_stage_overrides')
    patch = deepcopy(override)
    patch.pop('id', None)
    patch.pop('stage_id', None)
    if not patch:
        return stage
    if 'action' in stage:
        _apply_compiled_stage_override(stage, patch)
        return stage
    if 'target_pose' in patch:
        target_pose = patch.pop('target_pose')
        if isinstance(target_pose, dict):
            patch.update(deepcopy(target_pose))
        elif isinstance(target_pose, (list, tuple)) and len(target_pose) == 3:
            pose_key = next((key for key in ('grasp_position_m', 'grasp_centre_m') if key in stage), 'position_m')
            patch[pose_key] = list(target_pose)
        else:
            raise ValueError('invalid_stage_override_target_pose')
    if ('position_m' in patch and 'quaternion_xyzw' in patch
            and 'grasp_position_m' not in patch and 'angles_deg' not in patch):
        for key in ('grasp_position_m', 'angles_deg', 'contact_region', 'contact_depth_m'):
            stage.pop(key, None)
    elif ('grasp_position_m' in patch and 'angles_deg' in patch
            and 'position_m' not in patch and 'quaternion_xyzw' not in patch):
        for key in ('position_m', 'quaternion_xyzw'):
            stage.pop(key, None)
    if 'position_m' in patch and 'grasp_centre_m' not in patch:
        for key in JAW_GRASP_CENTRE_FIELDS:
            stage.pop(key, None)
    elif 'grasp_centre_m' in patch and 'position_m' not in patch:
        stage.pop('position_m', None)
    if 'fixed_jaw_yaw_deg' in patch and 'quaternion_xyzw' not in patch:
        stage.pop('quaternion_xyzw', None)
    elif 'quaternion_xyzw' in patch and 'fixed_jaw_yaw_deg' not in patch:
        stage.pop('fixed_jaw_yaw_deg', None)
    if 'contact_region' in patch and 'contact_depth_m' not in patch:
        stage.pop('contact_depth_m', None)
    elif 'contact_depth_m' in patch and 'contact_region' not in patch:
        stage.pop('contact_region', None)
    if 'speed_preset' in patch and 'speed_scale' not in patch:
        stage.pop('speed_scale', None)
    elif 'speed_scale' in patch and 'speed_preset' not in patch:
        stage.pop('speed_preset', None)
    if 'gripper_speed_preset' in patch and 'gripper_speed_scale' not in patch:
        stage.pop('gripper_speed_scale', None)
    elif 'gripper_speed_scale' in patch and 'gripper_speed_preset' not in patch:
        stage.pop('gripper_speed_preset', None)
    if 'tolerance_preset' in patch:
        if 'position_tolerance_m' not in patch:
            stage.pop('position_tolerance_m', None)
        if 'orientation_tolerance_rad' not in patch:
            stage.pop('orientation_tolerance_rad', None)
        if 'max_upward_dz_m' not in patch:
            stage.pop('max_upward_dz_m', None)
        if 'max_downward_dz_m' not in patch:
            stage.pop('max_downward_dz_m', None)
        for field in SPLIT_ORIENTATION_FIELDS:
            if field not in patch:
                stage.pop(field, None)
    if 'stall_guard_preset' in patch and 'max_tracking_divergence_m' not in patch:
        stage.pop('max_tracking_divergence_m', None)
    if 'grip_guard_preset' in patch:
        if 'min_aperture_m' not in patch:
            stage.pop('min_aperture_m', None)
        if 'max_aperture_m' not in patch:
            stage.pop('max_aperture_m', None)
    if patch.get('carry_slip_guard') is False:
        if 'carry_slip_scope' not in patch:
            stage.pop('carry_slip_scope', None)
        if 'slip_consecutive_steps' not in patch:
            stage.pop('slip_consecutive_steps', None)
    if 'joint_travel_preset' in patch:
        if 'max_joint_travel_rad' not in patch:
            stage.pop('max_joint_travel_rad', None)
        if 'max_backtracking_rad' not in patch:
            stage.pop('max_backtracking_rad', None)
    if 'clearance_preset' in patch and 'clearance_height_m' not in patch:
        stage.pop('clearance_height_m', None)
    elif 'clearance_height_m' in patch and 'clearance_preset' not in patch:
        stage.pop('clearance_preset', None)
    stage.update(patch)
    return stage


def _normalize_stage_id(stage_id, source_ids):
    if not isinstance(stage_id, str):
        return None
    if stage_id in source_ids:
        return stage_id
    if stage_id.endswith('/motion'):
        base = stage_id[:-len('/motion')]
        if base in source_ids:
            return base
    return stage_id


def extract_and_override_stages(source_stages, start_from_stage_id, stage_overrides=None):
    """从 source_stages 提取自 start_from_stage_id 起的剩余阶段并应用 stage_overrides。"""
    if not isinstance(source_stages, list) or not source_stages:
        raise ValueError('source_plan_has_no_stages')
    if not isinstance(start_from_stage_id, str) or not start_from_stage_id.strip():
        raise ValueError('source_plan_continuation_requires_start_from_stage_id')
    ids = [stage.get('id') for stage in source_stages if isinstance(stage, dict)]
    start_from_stage_id = _normalize_stage_id(start_from_stage_id, ids)
    if start_from_stage_id not in ids:
        raise ValueError(f'unknown_start_from_stage_id:{start_from_stage_id}')
    start_index = ids.index(start_from_stage_id)
    excluded_ids = set(ids[:start_index])
    remaining = deepcopy(source_stages[start_index:])
    remaining_ids = {stage['id'] for stage in remaining}
    for stage in remaining:
        if 'depends_on' in stage and isinstance(stage['depends_on'], list):
            kept = [dep for dep in stage['depends_on'] if dep not in excluded_ids]
            if kept:
                stage['depends_on'] = kept
            else:
                stage.pop('depends_on', None)
    raw_overrides = normalize_stage_overrides(stage_overrides)
    overrides = {_normalize_stage_id(stage_id, remaining_ids): patch for stage_id, patch in raw_overrides.items()}
    unknown = [stage_id for stage_id in overrides if stage_id not in remaining_ids]
    if unknown:
        raise ValueError(f'unknown_override_stage_id:{unknown[0]}')
    for stage in remaining:
        if stage['id'] in overrides:
            apply_stage_override(stage, overrides[stage['id']])
    return remaining, list(overrides)


def resolve_backend_continuation(adapter, action):
    """解析直接传入 preview_batch 的底层 source_plan_id 续跑动作。"""
    if 'source_plan_id' not in action:
        if 'start_from_stage_id' in action or 'stage_overrides' in action:
            raise ValueError('continuation_fields_require_source_plan_id')
        return action, None
    if action.get('stages') is not None:
        raise ValueError('choose_stages_or_source_plan_id')
    source_plan_id = action.get('source_plan_id')
    cache = getattr(adapter, '_batch_plans', {}).get(source_plan_id)
    if cache is None:
        raise ValueError('batch_plan_unknown_or_expired')
    start_from_stage_id = action.get('start_from_stage_id')
    remaining, overridden_ids = extract_and_override_stages(
        cache.get('source_stages', []), start_from_stage_id, action.get('stage_overrides'))
    obs_id = action.get('based_on_observation_id')
    if obs_id is not None:
        for stage in remaining:
            if isinstance(stage.get('action'), dict):
                stage['action']['based_on_observation_id'] = obs_id
    resolved = deepcopy(cache.get('source_action', {}))
    resolved.update(deepcopy(action))
    if action.get('carry_slip_guard') is False:
        if 'carry_slip_scope' not in action:
            resolved.pop('carry_slip_scope', None)
        if 'slip_consecutive_steps' not in action:
            resolved.pop('slip_consecutive_steps', None)
    resolved.pop('source_plan_id', None)
    resolved.pop('start_from_stage_id', None)
    resolved.pop('stage_overrides', None)
    resolved['stages'] = remaining
    continuation_meta = dict(
        source_plan_id=source_plan_id,
        start_from_stage_id=start_from_stage_id,
        overridden_stage_ids=overridden_ids,
        remaining_stage_ids=[stage['id'] for stage in remaining],
        source_completed_stage_ids=deepcopy(cache.get('completed_stage_ids', [])),
    )
    return resolved, continuation_meta


def summarize_stage_progress(record, receipt):
    """将底层执行回执映射为源计划的阶段完成、失败与剩余阶段列表。"""
    source_stages = record.get('source_stages') or record.get('action', {}).get('stages', [])
    source_ids = [stage['id'] for stage in source_stages if isinstance(stage, dict) and 'id' in stage]
    raw_completed = set(receipt.get('completed_stage_ids') or [])
    if source_ids:
        completed_stage_ids = [sid for sid in source_ids if sid in raw_completed]
        completed_set = set(completed_stage_ids)
        remaining_stage_ids = [sid for sid in source_ids if sid not in completed_set]
    else:
        completed_stage_ids = list(receipt.get('completed_stage_ids') or [])
        remaining_stage_ids = list(receipt.get('pending_stage_ids') or [])
    failed_stage_id = _normalize_stage_id(receipt.get('failed_stage_id'), set(source_ids))
    if failed_stage_id in remaining_stage_ids:
        start_from_stage_id = failed_stage_id
    elif remaining_stage_ids:
        start_from_stage_id = remaining_stage_ids[0]
    else:
        start_from_stage_id = failed_stage_id
    return dict(
        completed_stage_ids=completed_stage_ids,
        failed_stage_id=failed_stage_id,
        remaining_stage_ids=remaining_stage_ids,
        start_from_stage_id=start_from_stage_id,
    )


def build_next_action(identifier, record, receipt, result):
    """构建动作回执中的 next_action 事实与显式续跑参数模板。"""
    if receipt.get('status') == 'completed':
        return dict(mode='inspect', plan_id=identifier)
    progress = summarize_stage_progress(record, receipt)
    obs_id = (result.get('current_observation_id')
              or receipt.get('result_observation_id')
              or record.get('summary', {}).get('observation_id'))
    start_from = progress['start_from_stage_id']
    preview_remaining = dict(
        mode='preview',
        observation_id=obs_id,
        source_plan_id=identifier,
        start_from_stage_id=start_from,
        stage_overrides={},
    ) if start_from else None
    direct_remaining = dict(
        mode='direct',
        observation_id=obs_id,
        source_plan_id=identifier,
        start_from_stage_id=start_from,
        stage_overrides={},
    ) if start_from else None
    if receipt.get('batch_status') == 'awaiting_observation':
        return dict(
            mode='execute',
            plan_id=identifier,
            expected_cursor=receipt.get('batch_cursor'),
            expected_revision=receipt.get('batch_revision', record.get('summary', {}).get('plan_revision')),
            completed_stage_ids=progress['completed_stage_ids'],
            failed_stage_id=progress['failed_stage_id'],
            remaining_stage_ids=progress['remaining_stage_ids'],
            source_plan_id=identifier,
            start_from_stage_id=start_from,
            continuation_example=preview_remaining,
            preview_remaining=preview_remaining,
            direct_remaining=direct_remaining,
        )
    return dict(
        mode='preview',
        observation_id=obs_id,
        source_plan_id=identifier,
        start_from_stage_id=start_from,
        stage_overrides={},
        completed_stage_ids=progress['completed_stage_ids'],
        failed_stage_id=progress['failed_stage_id'],
        remaining_stage_ids=progress['remaining_stage_ids'],
        continuation_example=preview_remaining,
        preview_remaining=preview_remaining,
        direct_remaining=direct_remaining,
    )


__all__ = [
    "_apply_compiled_stage_override",
    "_normalize_stage_id",
    "apply_stage_override",
    "build_next_action",
    "extract_and_override_stages",
    "normalize_stage_overrides",
    "resolve_backend_continuation",
    "summarize_stage_progress",
]
