"""基于已有批次计划与最新执行回执，支持由 GPT 显式控制局部阶段替换、跳过失败移动阶段及检查点漂移无缝重编译。"""
from copy import deepcopy

from ...robot.motion.grip_guard import CARRY_SLIP_GUARD_FIELDS
from ...robot.motion.motion_tolerance import SPLIT_ORIENTATION_FIELDS

CheckpointDriftReasons = (
    'batch_plan_stale_state_requires_new_preview',
    'batch_resume_first_joint_step_exceeds_limit',
    'batch_resume_first_jaw_step_exceeds_limit',
)


def build_repaired_args(harness, args, lookup_fn, *, repair_report=None):
    """根据原计划记录与最近回执，生成从当前实测状态重编译剩余阶段的请求参数。"""
    record = lookup_fn(harness, args['plan_id'])
    summary = record.get('summary') or {}
    completed_ids = set(summary.get('completed_stage_ids') or [])
    failed_stage_id = summary.get('failed_stage_id')
    skipped_ids = set()
    if args.get('skip_failed_stage') and isinstance(failed_stage_id, str) and failed_stage_id:
        skipped_ids.add(failed_stage_id)
        if not failed_stage_id.endswith('/motion'):
            skipped_ids.add(failed_stage_id.removesuffix('/motion'))
    done_ids = completed_ids | skipped_ids

    raw_stages = deepcopy(record.get('raw_stages') or [])
    uncompleted = []
    for stage in raw_stages:
        sid = stage.get('id')
        if sid in done_ids:
            continue
        # 若带到点后夹爪动作的 move 阶段其 /motion 子阶段已完成或被显式跳过而夹爪阶段未完成，保留其未完成的夹爪阶段
        if (stage.get('kind') == 'move' and 'gripper_goal_normalized' in stage
                and 'gripper_motion' not in stage and (sid + '/motion') in done_ids):
            stage = {
                'id': sid,
                'kind': 'gripper',
                'arm': stage['arm'],
                'goal_normalized': stage['gripper_goal_normalized'],
                **{k: stage[k] for k in (
                    'depends_on', 'gripper_speed_scale', 'gripper_speed_preset',
                    'checkpoint_after_gripper',
                    'grip_guard_preset', 'min_aperture_m', 'max_aperture_m',
                    *CARRY_SLIP_GUARD_FIELDS, 'grip_squeeze_rad',
                ) if k in stage},
            }
        uncompleted.append(stage)

    patch_stages = deepcopy(args.get('stages') or [])
    re_executed_stage_ids = []
    if patch_stages:
        uncompleted_index = {s.get('id'): idx for idx, s in enumerate(uncompleted)}
        prefix = []
        for patch_stage in patch_stages:
            pid = patch_stage.get('id')
            norm_pid = pid.removesuffix('/motion') if isinstance(pid, str) else None
            target_id = pid if pid in uncompleted_index else (
                norm_pid if norm_pid in uncompleted_index else None
            )
            if target_id is not None:
                uncompleted[uncompleted_index[target_id]] = dict(patch_stage, id=target_id)
            else:
                # 补丁阶段不在未完成列表中时统一前插；若属于已完成阶段则记录在摘要中诚实告知调用方将重跑
                if (pid in completed_ids or (norm_pid and norm_pid in completed_ids)) and pid not in re_executed_stage_ids:
                    re_executed_stage_ids.append(pid)
                prefix.append(patch_stage)
        remaining_stages = prefix + uncompleted
    else:
        remaining_stages = uncompleted

    if isinstance(repair_report, dict) and re_executed_stage_ids:
        repair_report['re_executed_stage_ids'] = re_executed_stage_ids

    for stage in remaining_stages:
        if 'depends_on' in stage:
            stage['depends_on'] = [d for d in stage.get('depends_on', []) if d not in done_ids]

    if not remaining_stages:
        raise ValueError('batch_repair_no_remaining_stages')

    repaired = {k: deepcopy(v) for k, v in args.items() if k not in ('plan_id', 'skip_failed_stage')}
    repaired['stages'] = remaining_stages
    source_action = record.get('action') if isinstance(record.get('action'), dict) else {}
    raw_batch_opts = {
        **{k: source_action[k] for k in ('checkpoint_after_gripper', 'allow_jaw_symmetry', 'allow_approach_tilt', *CARRY_SLIP_GUARD_FIELDS) if k in source_action},
        **(record.get('source_args') or {}),
        **(record.get('batch_options') or {}),
    }
    for key in ('allow_jaw_symmetry', 'allow_approach_tilt', 'obstacle_object_ids', 'obstacle_policy', 'ignore_obstacle_ids'):
        if key not in repaired and key in raw_batch_opts:
            repaired[key] = deepcopy(raw_batch_opts[key])
    # 防滑脱三键采用逐键覆盖继承；仅当显式传 carry_slip_guard=False 且未手传从属键时跳过继承从属键
    if 'carry_slip_guard' not in repaired and 'carry_slip_guard' in raw_batch_opts:
        repaired['carry_slip_guard'] = deepcopy(raw_batch_opts['carry_slip_guard'])
    if args.get('carry_slip_guard') is not False:
        for key in ('carry_slip_scope', 'slip_consecutive_steps'):
            if key not in repaired and key in raw_batch_opts:
                repaired[key] = deepcopy(raw_batch_opts[key])
    for group in (
        (
            'tolerance_preset',
            'position_tolerance_m',
            'orientation_tolerance_rad',
            'max_upward_dz_m',
            'max_downward_dz_m',
            *SPLIT_ORIENTATION_FIELDS,
        ),
        ('stall_guard_preset', 'max_tracking_divergence_m'),
        ('arrival_correction_rounds',),
        ('arrival_correction_max_offset_rad',),
        ('grip_guard_preset', 'min_aperture_m', 'max_aperture_m'),
        ('joint_travel_preset', 'max_joint_travel_rad', 'max_backtracking_rad'),
        ('target_exemption_preset', 'target_exemption_radius_m'),
        ('clearance_preset', 'clearance_height_m'),
    ):
        if not any(k in repaired for k in group):
            for k in group:
                if k in raw_batch_opts:
                    repaired[k] = deepcopy(raw_batch_opts[k])
    if 'checkpoint_after_gripper' not in repaired and 'checkpoint_after_gripper' in raw_batch_opts:
        repaired['checkpoint_after_gripper'] = deepcopy(raw_batch_opts['checkpoint_after_gripper'])
    return repaired


def build_repair_hint(identifier, receipt):
    """不向模型注入诱导性的 direct/skip_failed_stage 预设模板，统一由 next_action 提供客观阶段进度。"""
    return None
