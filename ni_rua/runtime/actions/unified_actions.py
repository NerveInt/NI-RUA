"""One actor action interface for complete cached motion batches and inspection."""
from copy import deepcopy
import math

from ...core.presentation.actor_payload import lean_action_result
from ...core.contracts import IDENTITY, canonical, validate_action
from ...core.embodiment import EmbodimentProfile, REACH_REQUEST_FIELDS, REACH_STAGE_FIELDS
from ...core.tool_errors import ToolInputError, _pointer
from ...robot.motion.grip_guard import (CARRY_SLIP_GUARD_FIELDS, CLOSING_JAW_GOAL_BELOW, GRIP_GUARD_FIELDS,
    GRIP_SQUEEZE_FIELD, jaw_goal, validate_batch_carry_slip_guard, validate_grip_guard_fields, validate_grip_squeeze)
from ...robot.motion.motion_safety import JOINT_TRAVEL_FIELDS, extract_joint_travel_options, joint_travel_presets
from ...robot.motion.motion_speed import resolve_action_speed
from ...robot.motion.motion_tolerance import (ARRIVAL_CORRECTION_FIELDS, STALL_GUARD_FIELDS, TOLERANCE_FIELDS,
    resolve_arrival_correction, resolve_stage_tolerance, resolve_stall_guard, validate_stall_guard_fields,
    validate_tolerance_fields)
from ...core.tool_output import ToolOutput, ToolResponse, output_contract_error
from ...robot.motion.action_compilation import compile_action, uses_arx_actions, validate_backend_action
from ...robot.motion.batch_contract import IMAGE_DELIVERY_MODES
from ...robot.motion.scene_obstacles import (
    CLEARANCE_PRESETS,
    DEFAULT_CLEARANCE_HEIGHT_M,
    OBSTACLE_CLEARANCE_FIELDS,
    STAGE_CLEARANCE_FIELDS,
    TARGET_EXEMPTION_PRESETS,
    resolve_stage_clearance,
)
from .inspection import embodiment_of
from .robot_action_schema import (
    SCHEMA_GRIP_GUARD_PRESETS,
    SCHEMA_STALL_GUARD_PRESETS,
    SCHEMA_TOLERANCE_PRESETS,
    _validate_stage,
    build as robot_action_contract,
)
from ...core.action_validation import capability_arms, embodiment_or_default
from ...core.episode_state import StateSlot, slot_get, slot_peek, slot_set

UNIFIED_ACTION_PLANS_SLOT: StateSlot[dict] = StateSlot("unified_action_plans", dict, scope="run")
UNIFIED_PLAN_SEQ_SLOT: StateSlot = StateSlot("unified_plan_sequence", lambda: None, scope="run")
UNIFIED_ACTION_SEQ_SLOT: StateSlot = StateSlot("unified_action_sequence", lambda: None, scope="run")
UNIFIED_ACTION_CALL_IDS_SLOT: StateSlot[dict] = StateSlot("unified_action_call_ids", dict, scope="run")
UNIFIED_TOOL_RESULTS_SLOT: StateSlot[dict] = StateSlot("unified_tool_results", dict, scope="run")


def _records(harness):
    records = slot_get(harness, UNIFIED_ACTION_PLANS_SLOT, legacy_attr="unified_action_plans")
    if not isinstance(records, dict):
        records = {}
        slot_set(harness, UNIFIED_ACTION_PLANS_SLOT, records, legacy_attr="unified_action_plans")
    return records


def _next(harness, kind):
    slot = UNIFIED_PLAN_SEQ_SLOT if kind == 'plan' else UNIFIED_ACTION_SEQ_SLOT
    legacy_attr = '_unified_' + kind + '_sequence'
    prefix = 'p' if kind == 'plan' else 'a'
    record_kind = 'unified_action_plan' if kind == 'plan' else 'action_request'
    with harness.lock:
        current = slot_peek(harness, slot, None, legacy_attr=legacy_attr)
        if current is None:
            current = harness.store.record_sequence(record_kind, prefix)
        value = int(current) + 1
        slot_set(harness, slot, value, legacy_attr=legacy_attr)
        return prefix + str(value)


def compile_request(harness, args, *, obstacle_report=None):
    """Compile one robot_action preview/direct/reach request into a validated ``BatchAction``.

    ``obstacle_report`` (dict or None): receives what converting ``obstacle_object_ids`` rows into boxes skipped or
    resized (``scene_obstacles.obstacles_from_object_table``); the action is the same either way."""
    from ...robot.motion.batch_contract import validate_batch_action
    from .plan_continuation import resolve_continuation_args
    args, _ = resolve_continuation_args(harness, args)
    stages = args.get('stages')
    if not isinstance(stages, list) or not stages:
        raise ValueError('robot_action_requires_stages')
    profile = embodiment_of(harness)  # the run's EmbodimentProfile (ARX_X5_PROFILE for a harness without one)
    contract = robot_action_contract(profile)
    # Request deadline when the caller omits timeout_s (s): the embodiment's (X5 120, SO-101 deployment default).
    timeout_s = args.get('timeout_s', profile.default_action_timeout_s)
    original_ids = [stage.get('id') if isinstance(stage, dict) else None for stage in stages]
    if any(not isinstance(identifier, str) or not identifier for identifier in original_ids) or len(set(original_ids)) != len(original_ids):
        raise ValueError('batch_requires_unique_stage_ids')
    validate_tolerance_fields(args, profile=profile)
    validate_stall_guard_fields(args, profile=profile)
    if any(k in args for k in STAGE_CLEARANCE_FIELDS):
        batch_clearance = resolve_stage_clearance(
            {k: args[k] for k in STAGE_CLEARANCE_FIELDS if k in args},
            DEFAULT_CLEARANCE_HEIGHT_M,
        )
    else:
        batch_clearance = None
    output = []
    for stage in stages:
        common = {key: deepcopy(stage[key]) for key in ('id', 'arm', 'kind', 'depends_on') if key in stage}
        fields = {key: value for key,value in stage.items() if key not in common}
        kind, arm = common.get('kind'), common.get('arm')
        if kind == 'observe':
            if set(fields) != {'reason'}:
                raise ValueError('observe_requires_only_reason')
            if not isinstance(fields['reason'], str) or not fields['reason'].strip():
                raise ValueError('batch_observe_requires_reason')
            _validate_stage(stage, profile)
            output.append(dict(common, reason=fields['reason']))
            continue
        if arm not in capability_arms(getattr(harness, 'capabilities', None), profile):
            raise ValueError('robot_action_stage_requires_arm')
        if kind not in ('move', 'gripper'):
            raise ValueError('unsupported_robot_action_stage_fields')
        allowed = set(contract.stage_schemas[kind]['properties']) - set(contract.stage_common) - {'kind'}
        if set(fields)-allowed:
            raise ValueError('unsupported_robot_action_stage_fields')
        if 'allow_jaw_symmetry' in args and not isinstance(args['allow_jaw_symmetry'], bool):
            raise ValueError('invalid_allow_jaw_symmetry')
        if 'allow_approach_tilt' in args and not isinstance(args['allow_approach_tilt'], bool):
            raise ValueError('invalid_allow_approach_tilt')
        jaw_after = kind == 'move' and 'gripper_goal_normalized' in fields and 'gripper_motion' not in fields
        checkpoint_after_gripper = fields.pop('checkpoint_after_gripper', None)
        if checkpoint_after_gripper is not None:
            if not isinstance(checkpoint_after_gripper, bool):
                raise ValueError('invalid_checkpoint_after_gripper')
            if kind == 'move' and not jaw_after and 'gripper_goal_normalized' not in fields:
                _validate_stage(stage, profile)
                raise ValueError('checkpoint_after_gripper_requires_gripper_event')
        allow_jaw_symmetry = fields.pop('allow_jaw_symmetry', None)
        if allow_jaw_symmetry is not None:
            if kind != 'move' or not isinstance(allow_jaw_symmetry, bool):
                _validate_stage(stage, profile)
                raise ValueError('invalid_allow_jaw_symmetry')
        elif kind == 'move' and 'allow_jaw_symmetry' in args:
            allow_jaw_symmetry = args['allow_jaw_symmetry']
        allow_approach_tilt = fields.pop('allow_approach_tilt', None)
        if allow_approach_tilt is not None:
            if kind != 'move' or not isinstance(allow_approach_tilt, bool):
                _validate_stage(stage, profile)
                raise ValueError('invalid_allow_approach_tilt')
        elif kind == 'move' and 'allow_approach_tilt' in args:
            allow_approach_tilt = args['allow_approach_tilt']
        sym_field = {'allow_jaw_symmetry': allow_jaw_symmetry} if allow_jaw_symmetry is not None else {}
        tilt_field = {'allow_approach_tilt': allow_approach_tilt} if allow_approach_tilt is not None else {}
        grip_guard_fields = {k: fields.pop(k) for k in GRIP_GUARD_FIELDS if k in fields}
        if grip_guard_fields:
            validate_grip_guard_fields(grip_guard_fields, profile=profile)
        for k in CARRY_SLIP_GUARD_FIELDS:
            if k in fields:
                grip_guard_fields[k] = fields.pop(k)
        if GRIP_SQUEEZE_FIELD in fields:
            # Holding squeeze of this jaw command (rad of the jaw joint); travels with the stage that commands the jaw.
            validate_grip_squeeze(fields, profile=profile)
            grip_guard_fields[GRIP_SQUEEZE_FIELD] = fields.pop(GRIP_SQUEEZE_FIELD)
        joint_travel_fields = {k: fields.pop(k) for k in JOINT_TRAVEL_FIELDS if k in fields}
        if joint_travel_fields:
            extract_joint_travel_options(joint_travel_fields)
        stage_tol_fields = {key: fields.pop(key) for key in TOLERANCE_FIELDS if key in fields}
        stage_guard_fields = {key: fields.pop(key) for key in STALL_GUARD_FIELDS if key in fields}
        stage_correction_fields = {key: fields.pop(key) for key in ARRIVAL_CORRECTION_FIELDS if key in fields}
        stage_clearance_fields = {key: fields.pop(key) for key in STAGE_CLEARANCE_FIELDS if key in fields}
        if kind == 'move':
            resolved_tol = resolve_stage_tolerance(stage_tol_fields, args, profile=profile)
            resolved_guard = resolve_stall_guard(stage_guard_fields, args, profile=profile)
            resolved_clearance = None
            if stage_clearance_fields or batch_clearance is not None:
                resolved_clearance = resolve_stage_clearance(
                    dict(stage_clearance_fields),
                    DEFAULT_CLEARANCE_HEIGHT_M if batch_clearance is None else batch_clearance,
                )
        jaw = {}
        if jaw_after:
            jaw = {key: fields.pop(key) for key in ('gripper_speed_scale', 'gripper_speed_preset') if key in fields}
            jaw['goal_normalized'] = fields.pop('gripper_goal_normalized')
        grasp = 'grasp_position_m' in fields or 'angles_deg' in fields
        if kind == 'move' and (grasp and ('position_m' in fields or 'quaternion_xyzw' in fields)):
            raise ValueError('batch_move_requires_one_pose_convention')
        if allow_jaw_symmetry is None and kind == 'move' and grasp and uses_arx_actions(harness.capabilities):
            allow_jaw_symmetry = True
        sym_field = {'allow_jaw_symmetry': allow_jaw_symmetry} if allow_jaw_symmetry is not None else {}
        name = ('grasp' if grasp else 'move') if kind == 'move' else 'gripper'
        try:
            own = compile_action(getattr(harness, "robot", None), name,
                dict(fields, arm=arm, observation_id=args['observation_id'], timeout_s=timeout_s), harness.identity, harness.capabilities)
        except KeyError:
            # Missing public pose/goal fields must be input errors, while valid
            # stages still retain unrelated caller/runtime KeyError behavior.
            _validate_stage(stage, profile)
            raise
        own = resolve_action_speed(own, harness.config.features.default_motion_speed_scale, profile=profile)
        if kind == 'move':
            own['path'].update(resolved_tol, stall_guard_preset=resolved_guard['stall_guard_preset'])
            if resolved_guard['max_tracking_divergence_m'] is not None:
                own['path']['max_tracking_divergence_m'] = resolved_guard['max_tracking_divergence_m']
            # Arrival-correction bounds (arrival_correction tag): only what the stage or the batch sets.
            own['path'].update(resolve_arrival_correction(stage_correction_fields, args, profile=profile))
            if resolved_clearance is not None:
                own['path']['clearance_height_m'] = resolved_clearance
        chk_field = {'checkpoint_after_gripper': checkpoint_after_gripper} if checkpoint_after_gripper is not None else {}
        if jaw_after:
            internal = common['id']+'/motion'
            if internal in original_ids:
                raise ValueError('batch_generated_stage_id_conflict')
            jaw_action = compile_action(getattr(harness, "robot", None), 'gripper', dict(jaw, arm=arm,
                observation_id=args['observation_id'], timeout_s=timeout_s), harness.identity, harness.capabilities)
            jaw_action = resolve_action_speed(jaw_action, harness.config.features.default_motion_speed_scale,
                                              profile=profile)
            _validate_stage(stage, profile)
            output.append(dict(common, id=internal, action=own, **sym_field, **tilt_field, **joint_travel_fields))
            output.append(dict(id=common['id'], arm=arm, kind='gripper', depends_on=[internal],
                action=jaw_action, **chk_field, **grip_guard_fields))
        else:
            _validate_stage(stage, profile)
            output.append(dict(common, action=own, **chk_field, **sym_field, **tilt_field, **grip_guard_fields, **joint_travel_fields))
    action = dict(schema_version='0.1', **harness.identity, world_kind='observed', type='BatchAction',
        action_id=args.get('action_id') or _next(harness, 'action'), based_on_observation_id=args['observation_id'],
        mode='absolute', frame_id=harness.capabilities['frame_id'], tcp_config_id=harness.capabilities['tcp_config_id'],
        limits_profile=harness.capabilities.get('limits_profile', 'simulation-slow-v1'),
        timeout_s=timeout_s, stages=output)
    if 'checkpoint_after_gripper' in args:
        if not isinstance(args['checkpoint_after_gripper'], bool):
            raise ValueError('invalid_checkpoint_after_gripper')
        action['checkpoint_after_gripper'] = args['checkpoint_after_gripper']
    if 'allow_jaw_symmetry' in args:
        if not isinstance(args['allow_jaw_symmetry'], bool):
            raise ValueError('invalid_allow_jaw_symmetry')
        action['allow_jaw_symmetry'] = args['allow_jaw_symmetry']
    if 'allow_approach_tilt' in args:
        if not isinstance(args['allow_approach_tilt'], bool):
            raise ValueError('invalid_allow_approach_tilt')
        action['allow_approach_tilt'] = args['allow_approach_tilt']
    if 'image_delivery' in args:
        if args['image_delivery'] not in IMAGE_DELIVERY_MODES:
            raise ValueError('invalid_image_delivery')
        action['image_delivery'] = args['image_delivery']
    for key in (*TOLERANCE_FIELDS, *STALL_GUARD_FIELDS, *GRIP_GUARD_FIELDS, *CARRY_SLIP_GUARD_FIELDS, *JOINT_TRAVEL_FIELDS):
        if key in args:
            action[key] = args[key]
    if any(k in action for k in GRIP_GUARD_FIELDS):
        validate_grip_guard_fields({k: action[k] for k in GRIP_GUARD_FIELDS if k in action}, profile=profile)
    validate_batch_carry_slip_guard(action, profile=profile)
    from .obstacle_registry import merge_and_persist_obstacles
    args = merge_and_persist_obstacles(
        harness,
        args,
        output,
        report=obstacle_report,
    )
    proxy_fields = ('obstacles','held_objects','contact_entity_id','initial_contact','placement_contact')
    proxies = {key:deepcopy(args[key]) for key in proxy_fields if key in args}
    if 'held_object' in args:
        if 'held_objects' in args:
            raise ValueError('choose_held_object_or_held_objects')
        box=deepcopy(args['held_object']); arm=box.pop('arm',None)
        owners={stage['arm'] for stage in output if stage['kind']!='observe'}
        if arm is None:
            if len(owners)!=1:
                raise ValueError('dual_batch_held_object_requires_arm')
            arm=next(iter(owners))
        proxies['held_objects']={arm:box}
    if proxies:
        from ...robot.motion.batch_geometry import validate_proxies
        action['review_proxies']=validate_proxies(proxies)
    validate_batch_action(action, harness.capabilities, profile=profile)
    if not uses_arx_actions(harness.capabilities):
        validate_backend_action(harness.robot, action, harness.capabilities)
    return action


def _summarize_effective_tolerances(harness, action, raw=None):
    """提取每个 move 阶段经编译解析后的透明有效容差与停滞门禁配置（与 batch_motion 执行期一致）。"""
    profile = embodiment_of(harness)
    valid_arms = capability_arms(getattr(harness, 'capabilities', None) if harness is not None else None, profile)
    stages = (action or {}).get('stages') or []
    raw_costs = (raw or {}).get('stage_costs') if isinstance(raw, dict) else None
    summary = {}
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        arm = stage.get('arm')
        kind = stage.get('kind')
        if kind != 'move' or arm not in valid_arms:
            continue
        user_id = str(stage['id']).removesuffix('/motion')
        raw_tol = (
            (raw_costs.get(stage['id']) or {}).get('effective_tolerances')
            if isinstance(raw_costs, dict)
            else None
        )
        if isinstance(raw_tol, dict):
            summary[user_id] = deepcopy(raw_tol)
            continue
        own = stage.get('action') or {}
        path = own.get('path') if isinstance(own.get('path'), dict) else {}
        stage_tol_spec = (
            {k: path[k] for k in TOLERANCE_FIELDS if k in path}
            | {k: stage[k] for k in TOLERANCE_FIELDS if k in stage}
        )
        stage_guard_spec = (
            {k: path[k] for k in STALL_GUARD_FIELDS if k in path}
            | {k: stage[k] for k in STALL_GUARD_FIELDS if k in stage}
        )
        resolved_tol = resolve_stage_tolerance(stage_tol_spec, action, profile=profile)
        resolved_guard = resolve_stall_guard(stage_guard_spec, action, profile=profile)
        pos_tol = float(resolved_tol['position_tolerance_m'])
        ori_tol = float(resolved_tol['orientation_tolerance_rad'])
        preset = str(resolved_tol['tolerance_preset'])
        stall_preset = str(resolved_guard['stall_guard_preset'])
        if resolved_tol.get('max_upward_dz_m') is not None:
            raw_up = float(resolved_tol['max_upward_dz_m'])
            eff_dz = raw_up if raw_up > 0.0 else 0.0
        elif stall_preset == 'contact_stop' and profile.robot_model_id in ('arx-x5', 'libero-franka-panda'):
            eff_dz = max(pos_tol, 0.040)
        else:
            eff_dz = 0.0
        entry = dict(
            tolerance_preset=preset,
            position_tolerance_m=round(pos_tol, 5),
            orientation_tolerance_rad=round(ori_tol, 5),
            max_upward_dz_m=round(eff_dz, 5),
            stall_guard_preset=stall_preset,
        )
        if resolved_tol.get('max_downward_dz_m') is not None:
            raw_down = float(resolved_tol['max_downward_dz_m'])
            entry['max_downward_dz_m'] = round(raw_down if raw_down > 0.0 else 0.0, 5)
        if resolved_guard['max_tracking_divergence_m'] is not None:
            entry['max_tracking_divergence_m'] = round(float(resolved_guard['max_tracking_divergence_m']), 5)
        summary[user_id] = entry
    return summary


def _sample_ee_pos_m(arm_sample, tcp='link6'):
    if not isinstance(arm_sample, dict):
        return None
    keys = (f'{tcp}_pose', 'link6_pose', 'measured_ee_pose', 'ee_pose') if tcp != 'link6' else ('link6_pose', 'measured_ee_pose', 'ee_pose')
    for key in keys:
        pose = arm_sample.get(key)
        if isinstance(pose, dict) and isinstance(pose.get('position_m'), list) and len(pose['position_m']) == 3:
            return [round(float(v), 4) for v in pose['position_m']]
    return None


def _summarize_candidate_trajectory(trajectory, action, raw, *, profile=None):
    """提取零图像 preflight / preview 所需的紧凑数值关节与末端轨迹摘要。"""
    arms_order = profile.arms if profile is not None else ('left', 'right')
    tcp = profile.tcp_link if profile is not None else 'link6'
    samples = trajectory.get('samples') if isinstance(trajectory, dict) and isinstance(trajectory.get('samples'), list) else []
    stages = (action or {}).get('stages') or []
    active_arms = [
        arm for arm in arms_order
        if any(isinstance(s, dict) and s.get('arm') == arm for s in stages)
    ]
    arm_costs = raw.get('arm_costs') if isinstance(raw, dict) and isinstance(raw.get('arm_costs'), dict) else {}
    arms_summary = {}
    for arm in active_arms:
        info = {}
        if samples:
            first_arm = (samples[0].get('arms') or {}).get(arm) or {}
            mid_arm = (samples[len(samples) // 2].get('arms') or {}).get(arm) or {}
            last_arm = (samples[-1].get('arms') or {}).get(arm) or {}
            if isinstance(first_arm.get('q_rad'), list):
                info['start_q_rad'] = [round(float(v), 4) for v in first_arm['q_rad']]
            if isinstance(last_arm.get('q_rad'), list):
                info['end_q_rad'] = [round(float(v), 4) for v in last_arm['q_rad']]
            start_pos = _sample_ee_pos_m(first_arm, tcp)
            mid_pos = _sample_ee_pos_m(mid_arm, tcp)
            end_pos = _sample_ee_pos_m(last_arm, tcp)
            if start_pos is not None:
                info[f'start_{tcp}_position_m'] = start_pos
            if mid_pos is not None and len(samples) > 2:
                info[f'mid_{tcp}_position_m'] = mid_pos
            if end_pos is not None:
                info[f'end_{tcp}_position_m'] = end_pos
            for prefix, src in (('start_jaw_goal_normalized', first_arm), ('end_jaw_goal_normalized', last_arm)):
                jaw_val = src.get('jaw_goal_normalized', src.get('jaw_command_normalized'))
                if isinstance(jaw_val, (int, float)):
                    info[prefix] = round(float(jaw_val), 4)
        cost = arm_costs.get(arm)
        if isinstance(cost, dict):
            if isinstance(cost.get('sum_joint_travel_rad'), (int, float)):
                info['sum_joint_travel_rad'] = round(float(cost['sum_joint_travel_rad']), 4)
            if isinstance(cost.get('joint_travel_rad'), list):
                info['joint_travel_rad'] = [round(float(v), 4) for v in cost['joint_travel_rad']]
        arms_summary[arm] = info

    stage_costs = raw.get('stage_costs') if isinstance(raw, dict) and isinstance(raw.get('stage_costs'), dict) else {}
    stage_entries = {}
    for st in stages:
        if not isinstance(st, dict) or not isinstance(st.get('id'), str):
            continue
        raw_id = st['id']
        user_id = raw_id.removesuffix('/motion')
        entry = stage_entries.setdefault(user_id, dict(arm=st.get('arm'), kind=st.get('kind'), native_steps=0))
        if raw_id.endswith('/motion'):
            entry['kind'] = 'move'
        sc = stage_costs.get(raw_id)
        if isinstance(sc, dict):
            if isinstance(sc.get('native_steps'), int):
                entry['native_steps'] += sc['native_steps']
            if isinstance(sc.get('planned_ik_residual_m'), (int, float)):
                entry['planned_ik_residual_m'] = round(float(sc['planned_ik_residual_m']), 5)
            if sc.get('jaw_symmetry_resolved') and isinstance(sc.get('resolved_angles_deg'), list):
                entry['jaw_symmetry_resolved'] = True
                entry['resolved_angles_deg'] = [round(float(v), 2) for v in sc['resolved_angles_deg']]
            if sc.get('approach_tilt_resolved'):
                entry['approach_tilt_resolved'] = True
                if isinstance(sc.get('resolved_angles_deg'), list):
                    entry['resolved_angles_deg'] = [round(float(v), 2) for v in sc['resolved_angles_deg']]
        own = st.get('action') or {}
        gp = own.get('input_grasp_pose')
        if isinstance(gp, dict) and st.get('arm') in gp:
            gp = gp[st['arm']]
        if isinstance(gp, dict) and isinstance(gp.get('position_m'), list) and isinstance(gp.get('angles_deg'), list):
            entry['target_grasp_position_m'] = [round(float(v), 4) for v in gp['position_m']]
            entry['target_angles_deg'] = [round(float(v), 2) for v in gp['angles_deg']]
        else:
            wps = ((own.get('path') or {}).get('waypoints')) if isinstance(own.get('path'), dict) else None
            if isinstance(wps, list) and wps and isinstance(wps[-1], dict):
                wp = wps[-1].get(st.get('arm'))
                if isinstance(wp, dict) and isinstance(wp.get('position_m'), list):
                    entry[f'target_{tcp}_position_m'] = [round(float(v), 4) for v in wp['position_m']]
        evs = own.get('gripper_events') or []
        if evs and isinstance(evs[-1].get('goal_normalized'), (int, float)):
            entry['jaw_goal_normalized'] = round(float(evs[-1]['goal_normalized']), 4)

    result = dict(
        total_samples=len(samples),
        planned_native_steps=raw.get('planned_native_steps', max(0, len(samples) - 1) if samples else 0),
        active_arms=active_arms,
        arms=arms_summary,
        stages=stage_entries,
    )
    if isinstance(raw.get('nominal_sim_time_s'), (int, float)):
        result['nominal_sim_time_s'] = round(float(raw['nominal_sim_time_s']), 4)
    return result


def _closed_jaw_before_grasp_arms(args, harness):
    """检测批次中是否存在未先开爪便直接指令闭爪抓取（goal_normalized <= 0.2）的已闭爪机械臂。"""
    if not (isinstance(args, dict) and isinstance(args.get('stages'), list) and harness is not None):
        return []
    profile = embodiment_of(harness)
    gripper = profile.gripper
    if gripper.empty_close_aperture_m is None:
        return []
    from ...core.config import motion_thresholds
    from ...robot.motion.grip_guard import _extract_jaw
    closed_thresh_m = gripper.empty_close_aperture_m + float(
        motion_thresholds(harness).get('grip_holding_threshold_mm', 3.0)) / 1000.0
    obs = getattr(harness, 'latest', None) or {}
    live_state = getattr(harness, 'robot_state', None)
    if not isinstance(obs, dict):
        return []
    valid_arms = capability_arms(getattr(harness, 'capabilities', None), profile)
    grip_facts = obs.get('grip_facts') if isinstance(obs.get('grip_facts'), dict) else {}
    r_state = obs.get('robot_state') if isinstance(obs.get('robot_state'), dict) else {}
    grip_pos = r_state.get('gripper_positions') if isinstance(r_state.get('gripper_positions'), dict) else {}
    opened_in_batch = set()
    flagged = []
    for st in args['stages']:
        if not isinstance(st, dict):
            continue
        arm = st.get('arm')
        if arm not in valid_arms:
            continue
        goal = st.get('goal_normalized') if st.get('kind') == 'gripper' else st.get('gripper_goal_normalized')
        if isinstance(goal, (int, float)):
            if float(goal) >= 0.5:
                opened_in_batch.add(arm)
            elif float(goal) <= 0.2 and arm not in opened_in_batch and arm not in flagged:
                arm_gf = grip_facts.get(arm)
                ap = arm_gf.get('measured_aperture_m') if isinstance(arm_gf, dict) else None
                if ap is None:
                    ap = grip_pos.get(arm)
                if ap is None and isinstance(live_state, dict):
                    jaw = _extract_jaw(live_state, arm)
                    ap = None if jaw is None else float(jaw) * gripper.full_open_aperture_m
                if isinstance(ap, (int, float)) and float(ap) < closed_thresh_m:
                    flagged.append(arm)
    return flagged


def _is_blocking_geometry_risk(item):
    """仅将未声明接触的显式场景障碍碰撞（scene_proxy_intersection）判定为硬阻断风险。"""
    return (
        isinstance(item, dict)
        and item.get('kind') == 'scene_proxy_intersection'
        and not item.get('contact_declaration')
    )


def _extract_measured_support_z_m(harness, observation_id=None) -> float | None:
    """仅从当前观测已缓存的公开 3D 感知结果 (`WorldObservation3D`) 中提取实测支撑台面高度 `table_z_m`（无缓存时返回 `None`，绝不注入硬编码默认高度）。"""
    from ...perception.world_model import get_world_observation
    wobs = get_world_observation(harness, observation_id)
    if wobs is not None:
        # 保持与 scene3d.table_z 优先、其次行实测 bottom_z 中位数的严格语义一致
        if wobs.cloud_table_z_m is not None:
            return float(wobs.cloud_table_z_m)
        table_bzs = sorted(
            float(r.bottom_z)
            for r in wobs.rows
            if r.support == 'table' and r.bottom_z is not None and math.isfinite(r.bottom_z)
        )
        if table_bzs:
            return table_bzs[len(table_bzs) // 2]
        if wobs.table_z_m is not None:
            return float(wobs.table_z_m)
    from ...perception.external.obs3d import OBS3D_MEMO_SLOT
    memo = slot_peek(harness, OBS3D_MEMO_SLOT, None, legacy_attr='_obs3d_memo')
    if not (isinstance(memo, tuple) and len(memo) == 2 and isinstance(memo[1], dict)):
        return None
    memo_obs_id, payload = memo
    if observation_id and memo_obs_id and str(memo_obs_id) != str(observation_id):
        return None
    s3_entry = payload.get('scene3d')
    if isinstance(s3_entry, tuple) and s3_entry and isinstance(s3_entry[0], dict):
        tz = s3_entry[0].get('table_z')
        if isinstance(tz, (int, float)) and math.isfinite(tz):
            return float(tz)
    ot_entry = payload.get('object_table')
    if isinstance(ot_entry, tuple) and ot_entry and isinstance(ot_entry[0], dict):
        ot = ot_entry[0]
        cols = ot.get('columns')
        rows = ot.get('rows')
        if isinstance(cols, list) and isinstance(rows, list) and 'bottom_z' in cols and 'support' in cols:
            bz_idx = cols.index('bottom_z')
            sup_idx = cols.index('support')
            table_bzs = [
                float(r[bz_idx])
                for r in rows
                if isinstance(r, list)
                and len(r) > max(bz_idx, sup_idx)
                and r[sup_idx] == 'table'
                and isinstance(r[bz_idx], (int, float))
                and math.isfinite(r[bz_idx])
            ]
            if table_bzs:
                table_bzs.sort()
                return table_bzs[len(table_bzs) // 2]
    return None


def _diagnose_motion_safety_failure(summary):
    """提取零步 `motion_safety_rejected` 的精确子因（步数预算不足 vs 关节行程超限），消除黑箱报错。"""
    checks = summary.get('motion_safety_checks')
    if not isinstance(checks, list):
        return {}, None, None
    sub_reasons = []
    rejected_stages = []
    ep_rem = None
    req_rem = None
    first_travel_sid = None
    for check in checks:
        if not isinstance(check, dict) or check.get('status') != 'reject':
            continue
        raw_sid = str(check.get('stage_id') or '')
        clean_sid = raw_sid.removesuffix('/motion') if raw_sid else None
        for arm_name, arm_data in (check.get('arms') or {}).items():
            if not isinstance(arm_data, dict) or arm_data.get('status') != 'reject':
                continue
            reasons = [str(r) for r in (arm_data.get('reasons') or [])]
            for r in reasons:
                if r not in sub_reasons:
                    sub_reasons.append(r)
            if type(arm_data.get('episode_remaining_native_steps')) is int:
                ep_rem = arm_data['episode_remaining_native_steps']
            if type(arm_data.get('request_remaining_native_steps')) is int:
                req_rem = arm_data['request_remaining_native_steps']
            entry = dict(stage_id=clean_sid, arm=arm_name, reasons=reasons)
            if 'insufficient_remaining_native_budget' in reasons:
                entry['cumulative_native_steps'] = arm_data.get('nominal_native_steps')
                entry['exceeded_budgets'] = deepcopy(arm_data.get('exceeded_budgets') or {})
            if 'abnormal_joint_travel' in reasons:
                if first_travel_sid is None and clean_sid:
                    first_travel_sid = clean_sid
                entry['offending_joints'] = [
                    dict(
                        joint_name=w.get('joint_name'),
                        reason=w.get('reason'),
                        total_travel_rad=round(float(w.get('total_travel_rad', 0.0)), 4),
                        signed_net_rad=round(float(w.get('signed_net_rad', 0.0)), 4),
                        shortest_periodic_delta_rad=round(float(w.get('shortest_periodic_delta_rad', 0.0)), 4),
                    )
                    for w in (arm_data.get('witnesses') or [])
                    if isinstance(w, dict)
                ]
            rejected_stages.append(entry)
    if not sub_reasons:
        return {}, None, None

    planned_steps = summary.get('planned_native_steps')
    budget_caps = [x for x in (ep_rem, req_rem) if type(x) is int]
    eff_cap = min(budget_caps) if budget_caps else None
    msg_parts = []
    remediations = []
    if 'insufficient_remaining_native_budget' in sub_reasons:
        fits_fast = bool(type(planned_steps) is int and eff_cap is not None and eff_cap > 0 and planned_steps <= 2 * eff_cap)
        msg_parts.append(
            f"Batch requires {planned_steps} native steps, exceeding remaining budget "
            f"(effective_limit={eff_cap}, episode_remaining={ep_rem}, per_request_limit={req_rem})."
        )
        remediations.append(
            f"Increase speed_scale (up to 2.0) or drop/shorten non-essential stages to fit within {eff_cap} native steps "
            f"(fits_at_speed_scale_2={fits_fast}; joint_travel_preset does NOT change step count)."
        )
    if 'abnormal_joint_travel' in sub_reasons:
        msg_parts.append("One or more stages exceeded joint-travel or periodic wrist-winding thresholds.")
        remediations.append(
            "Switch to jaw_symmetric_twin_angles_deg (180-deg symmetric jaw twin) to unwind wrist joint6, "
            "or set joint_travel_preset='extended' / 'unconstrained' when a wide rotation is intentional."
        )

    remediation_text = " ".join(remediations)
    risk_patch = dict(
        sub_reasons=sub_reasons,
        planned_native_steps=planned_steps,
        episode_remaining_native_steps=ep_rem,
        request_remaining_native_steps=req_rem,
        message=" ".join([*msg_parts, remediation_text]).strip(),
    )
    note = dict(
        code='motion_safety_diagnosis',
        sub_reasons=sub_reasons,
        planned_native_steps=planned_steps,
        episode_remaining_native_steps=ep_rem,
        request_remaining_native_steps=req_rem,
        rejected_stages=rejected_stages,
        remediation=remediation_text,
    )
    return risk_patch, note, first_travel_sid


def _collect_plan_risks_and_notes(summary, args=None, harness=None):
    """汇总运动安全、3D 抓取与碰撞审查、AABB 几何代理及闭爪下探检查，生成顶层透明的 blocking_risks 与 advisory_notes。"""
    blocking_risks = []
    advisory_notes = []
    if not isinstance(summary, dict):
        return blocking_risks, advisory_notes

    if summary.get('status') != 'planned':
        risk_entry = dict(
            code=str(summary.get('reason') or 'planning_failed'),
            failed_stage_id=summary.get('failed_stage_id'),
            failed_arm=summary.get('failed_arm'),
            message=f"Batch planning failed with reason '{summary.get('reason') or 'unknown'}'.",
        )
        if summary.get('reason') == 'motion_safety_rejected':
            risk_patch, safety_note, _ = _diagnose_motion_safety_failure(summary)
            risk_entry.update(risk_patch)
            blocking_risks.append(risk_entry)
            if safety_note is not None:
                advisory_notes.append(safety_note)
            return blocking_risks, advisory_notes

        if (
            summary.get('reason') == 'path_infeasible'
            and isinstance(summary.get('feasible_approach_tilt_angles_deg'), list)
            and summary['feasible_approach_tilt_angles_deg']
        ):
            tilts = deepcopy(summary['feasible_approach_tilt_angles_deg'])
            clean_sid = str(summary.get('failed_stage_id') or '').removesuffix('/motion') or None
            hint = str(
                summary.get('failure_hint')
                or (
                    f"allow_approach_tilt=false left -15/+15/-30/+30 deg closing-axis tilts unsearched; "
                    f"setting allow_approach_tilt=true on the failed stage (or passing explicit "
                    f"angles_deg={tilts[0]} deg, keeping grasp_position_m fixed) resolves path_infeasible."
                )
            )
            risk_entry['feasible_approach_tilt_angles_deg'] = tilts
            risk_entry['message'] = f"{risk_entry['message']} {hint}"
            advisory_notes.append(
                dict(
                    code='approach_tilt_available',
                    stage_id=clean_sid,
                    arm=summary.get('failed_arm'),
                    allow_approach_tilt=False,
                    feasible_approach_tilt_angles_deg=deepcopy(tilts),
                    remediation=hint,
                )
            )

        blocking_risks.append(risk_entry)
        return blocking_risks, advisory_notes

    safety = summary.get('motion_safety')
    if isinstance(safety, dict) and safety.get('status') not in (None, 'pass'):
        rejected_ids = [c.get('stage_id') for c in (safety.get('checks') or []) if isinstance(c, dict) and c.get('status') == 'reject']
        blocking_risks.append(
            dict(
                code='motion_safety_reject',
                stage_ids=rejected_ids,
                message='Joint-travel or motion-safety admission rejected one or more stages.',
            )
        )

    grasp_review = summary.get('grasp_3d_review')
    if isinstance(grasp_review, dict):
        if 'blocking_risks' in grasp_review:
            blocking_risks.extend(deepcopy(grasp_review.get('blocking_risks') or []))
            advisory_notes.extend(deepcopy(grasp_review.get('advisory_notes') or []))
        elif (
            grasp_review.get('status') not in (None, 'aligned')
            or grasp_review.get('warnings')
            or grasp_review.get('risk_flags')
        ):
            advisory_notes.append(
                dict(
                    code='grasp_3d_review_warnings',
                    risk_flags=list(grasp_review.get('risk_flags') or []),
                    warnings=list(grasp_review.get('warnings') or []),
                )
            )

    geom_review = summary.get('geometry_review')
    if isinstance(geom_review, dict) and geom_review.get('risks'):
        for item in geom_review['risks']:
            if _is_blocking_geometry_risk(item):
                blocking_risks.append(dict(code='geometry_review_risk', detail=deepcopy(item)))
            elif isinstance(item, dict) and item.get('contact_declaration'):
                advisory_notes.append(dict(code='declared_contact_proxy_advisory', detail=deepcopy(item)))
            else:
                advisory_notes.append(dict(code='inter_arm_proxy_advisory', detail=deepcopy(item)))

    already_closed_arms = {
        r.get('arm')
        for r in (*blocking_risks, *advisory_notes)
        if isinstance(r, dict) and r.get('code') in ('warn_closed_jaw_descent', 'closed_jaw_before_grasp')
    }
    for arm in _closed_jaw_before_grasp_arms(args, harness):
        if arm not in already_closed_arms:
            advisory_notes.append(
                dict(
                    code='closed_jaw_before_grasp',
                    arm=arm,
                    message=f"Arm '{arm}' commands a closing jaw goal while measured aperture is <3mm without a preceding open-jaw stage.",
                )
            )

    stage_costs = summary.get('stage_costs')
    eff_tols = summary.get('effective_tolerances')
    if isinstance(stage_costs, dict) and isinstance(eff_tols, dict):
        for raw_sid, sc in stage_costs.items():
            if not isinstance(sc, dict):
                continue
            ik_res = sc.get('planned_ik_residual_m')
            user_sid = str(raw_sid).removesuffix('/motion')
            tol_info = eff_tols.get(user_sid)
            pos_tol = tol_info.get('position_tolerance_m') if isinstance(tol_info, dict) else None
            if (
                isinstance(ik_res, (int, float))
                and isinstance(pos_tol, (int, float))
                and pos_tol > 0
                and float(ik_res) > 0.65 * float(pos_tol)
            ):
                advisory_notes.append(
                    dict(
                        code='warn_planned_ik_residual_near_tolerance',
                        stage_id=user_sid,
                        planned_ik_residual_m=round(float(ik_res), 5),
                        position_tolerance_m=round(float(pos_tol), 5),
                    )
                )
    return blocking_risks, advisory_notes


def _has_closing_jaw_stage(action):
    """检查编译后的 BatchAction 阶段列表中是否包含会武装防滑脱监控的闭爪阶段。"""
    for st in (action or {}).get('stages') or []:
        if not isinstance(st, dict) or st.get('kind') == 'observe' or st.get('carry_slip_guard') is False:
            continue
        goal = jaw_goal(st.get('action'))
        if goal is not None and goal < CLOSING_JAW_GOAL_BELOW:
            return True
    return False


def _preview(harness, args, correlation, *, render_overlay=True, repair_report=None):
    from .inspection import _trajectory_image
    from .plan_continuation import resolve_continuation_args
    resolved_args, continuation_of = resolve_continuation_args(harness, args)
    row_boxes = {}
    action = compile_request(harness, resolved_args, obstacle_report=row_boxes)
    settle = None
    if 'settle' in resolved_args:
        from .stability import validate_options
        settle = validate_options(resolved_args['settle'], profile=embodiment_of(harness))
    with harness.lock:
        if not harness.accepting or harness.active_action:
            raise ValueError('robot_action_preview_requires_idle_active_run')
        observation = harness.current_observation()
        validate_action(action, harness.identity, observation['observation_id'], harness.capabilities,
                        profile=embodiment_of(harness))
        validate_backend_action(harness.robot, action, harness.capabilities)
        budget_before = harness.action_budget()
        with harness.robot_lock:
            raw = harness.robot.preview_batch(action)
        effective_tolerances = _summarize_effective_tolerances(harness, action, raw=raw)
        from ...robot.motion.motion_failure import archive_collision_witnesses
        archive_collision_witnesses(harness.store, raw, harness.observation_profile)
        if raw.get('status') != 'planned':
            binding = dict(observation_id=observation['observation_id'], sim_step=observation.get('sim_step'),
                step_domain=observation.get('step_domain'), robot_state_ref=observation.get('robot_state_ref'))
            zero_step = (raw.get('physics_advanced') is False
                and type(raw.get('executed_native_steps')) is int and raw['executed_native_steps'] == 0)
            failure = {key: deepcopy(raw[key]) for key in (
                'status', 'reason', 'failed_stage_id', 'failed_arm', 'failure_feedback',
                'feasible_approach_tilt_angles_deg', 'failure_hint',
                'motion_safety_checks', 'planned_native_steps', 'stage_costs', 'physics_advanced',
                'executed_native_steps') if key in raw}
            if isinstance(failure.get('stage_costs'), dict):
                for _sc in failure['stage_costs'].values():
                    if isinstance(_sc, dict):
                        _sc.pop('effective_tolerances', None)
            if isinstance(raw.get('stage_costs'), dict):
                for _sc in raw['stage_costs'].values():
                    if isinstance(_sc, dict):
                        _sc.pop('effective_tolerances', None)
            if type(raw.get('planned_native_steps')) is not int:
                costs = [arm.get('nominal_native_steps')
                    for check in raw.get('motion_safety_checks', []) if isinstance(check, dict)
                    for arm in (check.get('arms') or {}).values() if isinstance(arm, dict)
                    and type(arm.get('nominal_native_steps')) is int and arm['nominal_native_steps'] >= 0]
                if costs:
                    inferred_steps = max(costs)
                    raw['planned_native_steps'] = inferred_steps
                    failure['planned_native_steps'] = inferred_steps
            failure.update(observation_id=observation['observation_id'], state_binding=binding,
                action_budget=budget_before, zero_step_check_passed=zero_step)
            raw['action_budget'] = deepcopy(budget_before)
            raw['effective_tolerances'] = effective_tolerances
            blocking_risks, advisory_notes = _collect_plan_risks_and_notes(raw, args=resolved_args, harness=harness)
            raw['blocking_risks'] = blocking_risks
            raw['advisory_notes'] = advisory_notes
            failure['blocking_risks'] = deepcopy(blocking_risks)
            failure['advisory_notes'] = deepcopy(advisory_notes)
            harness.store.event('action.batch_preview', failure, correlation, world_kind='internal',
                sim_step=observation.get('sim_step'), step_domain=observation.get('step_domain'))
            return raw, []
        short = _next(harness, 'plan')
        trajectory = raw.pop('trajectory_data')
        trajectory.update(identity=deepcopy(harness.identity), observation_id=resolved_args['observation_id'], label=short)
        from .grasp_preview_3d import audit_batch_grasp_3d
        grasp_3d_review, grasp_3d_targets = audit_batch_grasp_3d(harness, action, resolved_args['observation_id'],
                                                                 stage_plans=raw.get('stage_plans'))
        if grasp_3d_targets:
            trajectory['grasp_3d_targets'] = grasp_3d_targets
        reference = harness.store.put_artifact(canonical(trajectory).encode(), 'candidate_trajectory', 'application/json')
        profile = embodiment_of(harness)
        trajectory_summary = _summarize_candidate_trajectory(trajectory, action, raw, profile=profile)
        if render_overlay:
            obs_cameras = {c.get('camera_id') for c in (observation.get('cameras') or []) if isinstance(c, dict)}
            # The embodiment's overview camera (X5 cam_head, SO-101 fixed) when this observation has it, else the
            # observation's first camera (a backend without the profile hook reports the default robot's camera).
            overview = profile.overview_camera
            default_cam = overview if overview in obs_cameras else next(iter(sorted(obs_cameras - {None})), overview)
            metadata, images = _trajectory_image(harness, trajectory, resolved_args.get('camera_id') or default_cam)
        else:
            skip_reason = (
                'image_delivery_suppressed'
                if resolved_args.get('image_delivery') == 'none'
                else f"{resolved_args.get('mode', 'preflight')}_zero_image_mode"
            )
            metadata, images = dict(status='skipped', reason=skip_reason, candidate_label=short), []
        summary = {key: deepcopy(value) for key,value in raw.items() if key != 'plan_id'}
        if isinstance(summary.get('stage_costs'), dict):
            for _sc in summary['stage_costs'].values():
                if isinstance(_sc, dict):
                    _sc.pop('effective_tolerances', None)
        if grasp_3d_review is not None:
            summary['grasp_3d_review'] = grasp_3d_review
        # A collision-world robot (SO-101 with its IK service) states what obstacle_object_ids did beyond copying a
        # row: ids it skipped (with the reason) and the size rules that changed a box (scene_obstacles.ROW_*).
        if (row_boxes and isinstance(summary.get('path_check'), dict)
                and 'collision_world' in embodiment_of(harness).supports):
            summary['path_check']['object_row_boxes'] = deepcopy(row_boxes)
        if 'motion_safety_checks' in summary:
            checks = summary.pop('motion_safety_checks')
            safety_ref = harness.store.put_artifact(canonical(checks).encode(), 'batch_motion_safety', 'application/json')
            summary['motion_safety'] = dict(details_ref=safety_ref, checked_stages=len(checks),
                enabled=any(check['enabled'] for check in checks),
                status='reject' if any(check['status']=='reject' for check in checks) else 'pass',
                checks=[dict(stage_id=check['stage_id'], status=check['status'], start_state_source=check['start_state_source']) for check in checks])
        summary['effective_tolerances'] = effective_tolerances
        blocking_risks, advisory_notes = _collect_plan_risks_and_notes(summary, args=resolved_args, harness=harness)
        unarmed_carry_slip = (
            (continuation_of is not None or repair_report is not None)
            and action.get('carry_slip_guard') is True
            and not _has_closing_jaw_stage(action)
        )
        if unarmed_carry_slip:
            advisory_notes.append(
                dict(
                    code='carry_slip_guard_not_armed_without_closing_jaw',
                    message=(
                        'Batch carry_slip_guard=true was inherited across continuation/repair, and remaining stages '
                        'have no closing jaw stage (goal_normalized < 0.5), so per-step carry-slip monitoring is not pre-armed.'
                    ),
                )
            )
        summary.update(
            plan_id=short,
            observation_id=resolved_args['observation_id'],
            trajectory_ref=reference,
            trajectory_overlay=metadata,
            blocking_risks=blocking_risks,
            advisory_notes=advisory_notes,
            trajectory_summary=trajectory_summary,
            action_budget=harness.action_budget(),
            executed_batch_native_steps=0,
            plan_revision=0,
            next_action=dict(mode='execute', plan_id=short, expected_cursor=0, expected_revision=0),
        )
        if unarmed_carry_slip:
            summary['carry_slip_guard_armed'] = False
        if continuation_of is not None:
            cont_copy = deepcopy(continuation_of)
            if unarmed_carry_slip:
                cont_copy['carry_slip_guard_armed'] = False
            summary['continuation_of'] = cont_copy
        if isinstance(repair_report, dict) and repair_report.get('re_executed_stage_ids'):
            summary['re_executed_stage_ids'] = deepcopy(repair_report['re_executed_stage_ids'])
        if settle is not None:
            summary['settle'] = deepcopy(settle)
            summary['additional_settle_budget'] = dict(max_native_steps=settle['max_native_steps'],
                timing='Only after the complete batch finishes, never at an observation checkpoint.',
                semantics='Additional bounded measured-RGB settling requests; separate from cached motion rows and their cost.')
        from .action_state_ledger import attach_preview_ledger
        summary = attach_preview_ledger(harness, action, summary)
        record = dict(identity=deepcopy(harness.identity), action=action, native_plan_id=raw['plan_id'],
            source_args=deepcopy(resolved_args),
            source_stages=deepcopy(resolved_args['stages']), raw_stages=deepcopy(resolved_args['stages']),
            completed_stage_ids=[],
            remaining_stage_ids=[s['id'] for s in resolved_args['stages']], failed_stage_id=None,
            executable=True, continuation_of=continuation_of,
            summary=summary, trajectory_ref=reference, settle=settle)
        _records(harness)[short] = record
        from ...core.config import motion_thresholds
        cache_limit = int(motion_thresholds(harness)['batch_plan_cache_size'])
        while len(_records(harness)) > cache_limit:
            _records(harness).pop(next(iter(_records(harness))))
        harness.store.save('unified_action_plan', short, record)
        harness.store.event('action.batch_preview', summary, correlation, world_kind='internal')
        return summary, images


def _reach_stage_fields(profile):
    """Stage fields one reach query admits on ``profile``: ``REACH_STAGE_FIELDS`` plus the grasp-centre pose fields
    (``embodiments.JAW_GRASP_CENTRE_FIELDS``) when the profile has ``jaw_grasp_centre``, and ``grasp_angles`` fields
    when the profile has ``grasp_angles``."""
    from ...core.embodiment import JAW_GRASP_CENTRE_FIELDS
    extra = frozenset(JAW_GRASP_CENTRE_FIELDS) if 'jaw_grasp_centre' in profile.pose_inputs else frozenset()
    if 'grasp_angles' in profile.pose_inputs:
        extra = extra | frozenset({'grasp_position_m', 'angles_deg', 'contact_region', 'contact_depth_m'})
    return REACH_STAGE_FIELDS | extra


def _reach(harness, args):
    """robot_action ``mode=reach``: inverse kinematics of every move stage pose; read-only.

    ``args``: ``REACH_REQUEST_FIELDS`` with 1..N move stages of ``_reach_stage_fields(profile)``. The stages are compiled exactly
    as a preview compiles them (``compile_request``: the same pose, tolerance and backend validation), the batch is
    validated against the current observation under the preview's idle rule, and the backend answers
    (``robot.reach_batch``: each stage solved independently from the joints measured at the call, every stage
    answered). Nothing is planned, cached, executed or invalidated and no plan record is written (compiling consumes an
    action-id sequence number, as a preview's compilation does).

    Returns the backend's verdicts (``status`` "solved", ``stage_count``, ``reachable_stage_count``,
    ``stage_results``, ``physics_advanced``, ``executed_native_steps``) plus ``observation_id``, the current
    observation the query was grounded on. A failure of the whole query (``status`` "failed": e.g.
    ``ik_service_unreachable``, an unreadable joint state) raises its reason, so it reaches the model as a tool error
    with that code."""
    reach_batch = getattr(getattr(harness, 'robot', None), 'reach_batch', None)
    if not callable(reach_batch):
        raise ValueError('robot_action_reach_unavailable')
    if set(args) - REACH_REQUEST_FIELDS:
        raise ValueError('robot_action_reach_has_unsupported_fields')
    if 'observation_id' not in args:
        raise ValueError('robot_action_requires_observation_id')
    stages = args.get('stages')
    if not isinstance(stages, list) or not stages:
        raise ValueError('robot_action_requires_stages')
    profile = embodiment_of(harness)
    stage_fields = _reach_stage_fields(profile)
    if any(not isinstance(stage, dict) or stage.get('kind') != 'move' or set(stage) - stage_fields
           for stage in stages):
        raise ValueError('robot_action_reach_accepts_move_poses_only')
    action = compile_request(harness, args)
    with harness.lock:
        if not harness.accepting or harness.active_action:
            raise ValueError('robot_action_reach_requires_idle_active_run')
        observation = harness.current_observation()
        validate_action(action, harness.identity, observation['observation_id'], harness.capabilities, profile=profile)
        validate_backend_action(harness.robot, action, harness.capabilities)
        with harness.robot_lock:
            raw = reach_batch(action)
    if not isinstance(raw, dict) or raw.get('status') != 'solved':
        reason = raw.get('reason') if isinstance(raw, dict) else None
        raise ValueError(reason if isinstance(reason, str) and reason else 'robot_action_reach_failed')
    return dict(deepcopy(raw), observation_id=observation['observation_id'])


def _lookup(harness, identifier):
    record = _records(harness).get(identifier)
    if record is None:
        raise ValueError('robot_action_plan_unknown_or_expired')
    if any(record['identity'].get(key) != harness.identity[key] for key in IDENTITY):
        raise ValueError('stale_environment_identity')
    return record


def _execute(harness, args, correlation):
    identifier = args['plan_id']
    record = _lookup(harness, identifier)
    call_ids = slot_get(harness, UNIFIED_ACTION_CALL_IDS_SLOT, legacy_attr='_unified_action_call_ids')
    key = correlation.get('tool_call_id')
    if key and key not in call_ids:
        read_state = getattr(getattr(harness, 'store', None), 'read_state', None)
        persisted_id = read_state('unified_action_call', key) if callable(read_state) else None
        if isinstance(persisted_id, str) and persisted_id:
            call_ids[key] = persisted_id
        else:
            call_ids[key] = _next(harness, 'action')
            set_state = getattr(getattr(harness, 'store', None), 'set_state', None)
            if callable(set_state):
                try:
                    set_state('unified_action_call', key, call_ids[key])
                except Exception:
                    pass
    action_id = args.get('action_id') or (call_ids[key] if key else _next(harness, 'action'))
    replay = False
    try:
        previous = harness.store.read('action_request', action_id)
    except KeyError:
        previous = None
    if previous is not None:
        if previous.get('batch_plan_id') != record['native_plan_id']:
            raise ValueError('action_id_content_conflict')
        if 'expected_cursor' in args and previous.get('batch_expected_cursor') != args['expected_cursor']:
            raise ValueError('action_id_content_conflict')
        if 'expected_revision' in args and previous.get('batch_expected_revision') != args['expected_revision']:
            raise ValueError('action_id_content_conflict')
        action, replay = previous, True
    else:
        action = dict(record['action'], action_id=action_id, batch_plan_id=record['native_plan_id'],
            batch_expected_cursor=args.get('expected_cursor',record['summary'].get('executed_batch_native_steps',0)),
            batch_expected_revision=args.get('expected_revision',record['summary'].get('plan_revision',0)),
            based_on_observation_id=harness.current_observation()['observation_id'])
        if 'image_delivery' in args:
            if args['image_delivery'] not in IMAGE_DELIVERY_MODES:
                raise ValueError('invalid_image_delivery')
            action['image_delivery'] = args['image_delivery']
        programs = getattr(harness, 'programs', None)
        if programs is not None and isinstance(correlation, dict) and correlation.get('program_id'):
            entry = getattr(programs, 'entries', {}).get(correlation['program_id'])
            if (isinstance(entry, dict) and entry.get('allow_direct_actions')
                    and action.get('timeout_s') != entry.get('_explicit_timeout')):
                from ..programs.runner import ACTION_RESERVE_S, program_remaining
                available = program_remaining(entry) - ACTION_RESERVE_S - 0.5
                if available >= 0.1 and float(action.get('timeout_s', 0)) > available:
                    action['timeout_s'] = round(available, 2)
        if not uses_arx_actions(harness.capabilities):
            # Cached continuation binds every leaf to the same current observation.
            action['stages'] = [dict(stage, action=dict(stage['action'],
                based_on_observation_id=action['based_on_observation_id'])) if 'action' in stage else dict(stage)
                for stage in action['stages']]
    result = harness.execute(action, correlation)
    receipt = result['receipt']
    result = dict(result, plan_id=identifier, action_id=action_id)
    if not replay:
        record['executable'] = receipt.get('batch_status') == 'awaiting_observation'
        record['summary']['status'] = receipt.get('batch_status', receipt.get('status', 'unknown'))
        record['summary']['last_action_id'] = action_id
        if 'batch_cursor' in receipt:
            record['summary']['executed_batch_native_steps'] = receipt['batch_cursor']
        if 'batch_revision' in receipt:
            record['summary']['plan_revision'] = receipt['batch_revision']
    try:
        from ...perception.external.obs3d import attached_table
        from .action_state_ledger import attach_execution_ledger, record_execution_outcome
        from .batch_repair import build_repair_hint
        from .plan_continuation import build_next_action, summarize_stage_progress
        if not replay:
            record_execution_outcome(
                harness,
                identifier,
                action,
                receipt,
                object_table=attached_table(harness, action.get('based_on_observation_id')),
            )
            progress = summarize_stage_progress(record, receipt)
            record['completed_stage_ids'] = progress['completed_stage_ids']
            record['remaining_stage_ids'] = progress['remaining_stage_ids']
            record['failed_stage_id'] = progress['failed_stage_id']
            record['summary']['completed_stage_ids'] = progress['completed_stage_ids']
            record['summary']['remaining_stage_ids'] = progress['remaining_stage_ids']
            record['summary']['failed_stage_id'] = progress['failed_stage_id']
            if receipt.get('batch_status') == 'awaiting_observation':
                record['summary'].pop('repair_action', None)
                record['summary']['checkpoint_observation_id'] = receipt.get('result_observation_id')
            else:
                hint = build_repair_hint(identifier, receipt)
                if hint is not None:
                    record['summary']['repair_action'] = hint
                else:
                    record['summary'].pop('repair_action', None)
            record['summary']['next_action'] = build_next_action(identifier, record, receipt, result)
        result = attach_execution_ledger(
            harness,
            dict(result, plan_id=identifier, action_id=action_id, next_action=deepcopy(record['summary']['next_action'])),
            record['summary'],
        )
        if 'repair_action' in record['summary']:
            result['repair_action'] = deepcopy(record['summary']['repair_action'])
        if record.get('continuation_of') is not None:
            result['continuation_of'] = deepcopy(record['continuation_of'])
        hook = getattr(harness, 'unified_action_postprocess', None)
        if hook is not None and not replay:
            try:
                updated = hook(result, correlation, settle=record.get('settle'))
                if updated is not None:
                    result = updated
            except Exception as error:
                result = dict(result, postprocess=dict(status='unavailable', reason=type(error).__name__,
                    action_execution='preserved_in_receipt_do_not_replay'))
            record['summary']['next_action'] = build_next_action(identifier, record, receipt, result)
            result['next_action'] = deepcopy(record['summary']['next_action'])
        return _action_response(harness, result, action, record, correlation)
    except Exception as error:
        # Optional post-execution bookkeeping or rendering cannot discard acknowledged physical execution.
        if not replay:
            record['executable'] = receipt.get('batch_status') == 'awaiting_observation'
        fallback = dict(
            result,
            plan_id=identifier,
            action_id=action_id,
            post_execution_bookkeeping_error=dict(
                error_type=type(error).__name__,
                action_replay_permitted=False,
            ),
        )
        if (correlation or {}).get('program_id'):
            stage_res = receipt.get('stage_results') if isinstance(receipt, dict) else None
            fallback = dict(
                fallback,
                status=receipt.get('status') if isinstance(receipt, dict) else 'unknown',
                reason=receipt.get('reason') if isinstance(receipt, dict) else None,
                failed_stage_id=receipt.get('failed_stage_id') if isinstance(receipt, dict) else None,
                failed_arm=receipt.get('failed_arm') if isinstance(receipt, dict) else None,
                observation_id=(
                    fallback.get('current_observation_id')
                    or (receipt.get('result_observation_id') if isinstance(receipt, dict) else None)
                ),
                result_observation_id=receipt.get('result_observation_id') if isinstance(receipt, dict) else None,
                stages=deepcopy(stage_res if isinstance(stage_res, dict) else {}),
            )
        return output_contract_error(
            ToolOutput(fallback, kind="action", projection="action_result").response(),
            "robot_action",
            "action_feedback_unavailable",
            "/content",
        )


def _action_response(harness, result, action, record, correlation):
    identifier, action_id = result['plan_id'], result['action_id']
    receipt = result['receipt']
    if record['summary'].get('advisory_notes') and 'advisory_notes' not in result:
        result = dict(result, advisory_notes=deepcopy(record['summary']['advisory_notes']))
    details = harness.store.put_artifact(canonical(result).encode(), 'unified_action_result', 'application/json')
    program_data = result
    if correlation.get('program_id'):
        program_data = dict(
            result,
            status=receipt.get('status'),
            reason=receipt.get('reason'),
            failed_stage_id=receipt.get('failed_stage_id'),
            failed_arm=receipt.get('failed_arm'),
            observation_id=result.get('current_observation_id') or receipt.get('result_observation_id'),
            result_observation_id=receipt.get('result_observation_id'),
            stages=deepcopy(receipt.get('stage_results', {})),
        )
        presented = program_data
    else:
        projected_receipt = lean_action_result(result)['receipt']
        presented = dict(plan_id=identifier, action_id=action_id,
            receipt=projected_receipt,
            planned_native_steps=record['summary'].get('planned_native_steps'),
            nominal_sim_time_s=record['summary'].get('nominal_sim_time_s'),
            action_budget=harness.action_budget(), details_ref=details)
        presented['next_action'] = deepcopy(record['summary']['next_action'])
        if 'repair_action' in record['summary']:
            presented['repair_action'] = deepcopy(record['summary']['repair_action'])
        if record['summary'].get('advisory_notes'):
            presented['advisory_notes'] = deepcopy(record['summary']['advisory_notes'])
        presented['stages'] = projected_receipt.pop('stage_results', {})
        for key in ('settling', 'postprocess', 'current_observation_id', 'result_observation_is_current', 'continuation_of', 'repeat_failure_warning'):
            if key in result:
                presented[key] = result[key]
        if projected_receipt.get('batch_status', projected_receipt.get('status')) in ('stopped', 'failed') and 'episode_ledger' in result:
            presented['episode_ledger'] = result['episode_ledger']
    from .strategy_checkpoint import adaptive_action_checkpoint, checkpoint_triggered
    checkpoint = adaptive_action_checkpoint(harness.store, harness.identity)
    if checkpoint_triggered(checkpoint):
        presented = {**presented, "strategy_checkpoint": checkpoint}
    from ...core.presentation.feedback_delivery import action_trajectory_feedback
    trajectory, trajectory_images = action_trajectory_feedback(harness, action, receipt, correlation)
    refined_process_traj = (
        ((trajectory or {}).get('compact_3d_trajectory') or {}).get('process_trajectory')
        if isinstance(trajectory, dict)
        else None
    ) or (receipt.get('process_trajectory') if isinstance(receipt, dict) else None)
    if isinstance(refined_process_traj, dict):
        if isinstance(program_data.get('receipt'), dict):
            program_data['receipt']['process_trajectory'] = deepcopy(refined_process_traj)
        if not correlation.get('program_id') and isinstance(presented.get('receipt'), dict):
            presented['receipt']['process_trajectory'] = deepcopy(refined_process_traj)
    if trajectory is not None:
        presented = dict(presented, trajectory_feedback=trajectory)
        if trajectory.get('trajectory_feedback_ref'):
            harness.store.set_state('unified_action_feedback',action_id,
                dict(trajectory_ref=trajectory['trajectory_feedback_ref'],identity=deepcopy(harness.identity)))
    content = ToolOutput(presented, extra_content=tuple(trajectory_images),
        projection="action_result" if correlation.get("program_id") else "action_display").content_parts()
    observation = result.get('current_observation_id') or receipt.get('result_observation_id')
    if observation:
        bundle = harness.store.read('observation', observation)
        effective_receipt = (
            {**receipt, 'process_trajectory': deepcopy(refined_process_traj)}
            if isinstance(refined_process_traj, dict) and 'process_trajectory' not in receipt
            else receipt
        )
        content += harness.observation_content(bundle, correlation,
            feedback_action=action, feedback_receipt=effective_receipt, structured=True)
        pipeline = getattr(harness, 'observation_pipeline', None)
        if pipeline is not None:
            content += list(pipeline.enrich(
                bundle, correlation, reply='action',
                before_observation_id=action.get('based_on_observation_id'),
            ).parts)
        else:
            from ...perception.external.obs3d import action_attachments
            content += action_attachments(harness, bundle, correlation, before_observation_id=action.get('based_on_observation_id'))
    return ToolResponse(content, data=program_data, kind="action", data_projection="action_result")


_PLANNING_ALLOWED_FIELDS = frozenset({
    'mode', 'image_delivery', 'observation_id', 'action_id', 'plan_id', 'skip_failed_stage',
    'stages', 'source_plan_id', 'start_from_stage_id', 'stage_overrides',
    'checkpoint_after_gripper', 'allow_jaw_symmetry', 'allow_approach_tilt', 'timeout_s', 'camera_id', 'detail', 'settle',
    'obstacles', 'held_object', 'held_objects', 'contact_entity_id', 'initial_contact', 'placement_contact',
    'plan_update', 'auto_execute_if_clean',
    *TOLERANCE_FIELDS, *STALL_GUARD_FIELDS, *GRIP_GUARD_FIELDS, *CARRY_SLIP_GUARD_FIELDS, *JOINT_TRAVEL_FIELDS,
    *OBSTACLE_CLEARANCE_FIELDS, *ARRIVAL_CORRECTION_FIELDS,
})
_EXECUTE_ALLOWED_FIELDS = frozenset({
    'mode', 'plan_id', 'action_id', 'image_delivery', 'expected_cursor', 'expected_revision',
    'observation_id', 'timeout_s', 'camera_id', 'detail', 'plan_update', 'auto_execute_if_clean',
})
_INSPECT_ALLOWED_FIELDS = frozenset({
    'mode', 'plan_id', 'action_id', 'trajectory_ref', 'path_kind', 'camera_id', 'detail',
    'native_steps', 'start_native_step', 'max_samples',
})
_BATCH_PRESET_FIELDS = (
    'tolerance_preset',
    'stall_guard_preset',
    'grip_guard_preset',
    'joint_travel_preset',
    'clearance_preset',
    'target_exemption_preset',
)
_MOVE_STAGE_PRESET_FIELDS = (
    'tolerance_preset',
    'stall_guard_preset',
    'grip_guard_preset',
    'joint_travel_preset',
    'clearance_preset',
)


def _resolve_profile(embodiment=None):
    """将传入的 EmbodimentProfile、Harness 或空值统一解析为 EmbodimentProfile。"""
    if isinstance(embodiment, EmbodimentProfile):
        return embodiment
    if embodiment is not None and (hasattr(embodiment, 'embodiment') or hasattr(embodiment, 'capabilities')):
        return embodiment_of(embodiment)
    return embodiment_or_default(embodiment)


def _check_preset_value(mapping, field, path_parts, *, profile, allow_legacy_aliases):
    """校验单个预设字段在当前本体上的可用性与合法取值，非法时抛出带精确 JSON Pointer 的 ToolInputError。"""
    if not isinstance(mapping, dict) or field not in mapping:
        return
    val = mapping[field]
    field_path = _pointer(path_parts)
    if field == 'tolerance_preset':
        published = [name for name in SCHEMA_TOLERANCE_PRESETS if name in profile.tolerance_presets]
        accepted = set(profile.tolerance_presets) if allow_legacy_aliases else set(published)
        code = 'invalid_tolerance_preset'
    elif field == 'stall_guard_preset':
        if 'stall_guard' not in profile.supports:
            raise ToolInputError(
                'unsupported_capability: stall_guard_preset',
                field_path=field_path,
                actual=val,
                suggestion='stall_guard_preset is not supported on this robot; omit stall_guard_preset.',
            )
        published = [name for name in SCHEMA_STALL_GUARD_PRESETS if name in profile.stall_guard_presets_m]
        accepted = (
            set(profile.stall_guard_presets_m)
            if (allow_legacy_aliases and profile.robot_model_id == 'arx-x5')
            else set(published)
        )
        if (
            isinstance(val, str)
            and val not in accepted
            and profile.robot_model_id != 'arx-x5'
            and val in profile.stall_guard_presets_m
        ):
            robot_label = 'SO-101' if profile.robot_model_id == 'so101' else profile.robot_model_id
            err = ToolInputError(
                f"unsupported_capability: stall_guard_preset {val} (presets on this {robot_label}: {', '.join(published)})",
                field_path=field_path,
                actual=val,
                suggestion=f"Choose one of the supported stall_guard_preset values on this robot: {', '.join(published)}.",
                minimal_example={'stall_guard_preset': published[0]} if published else None,
            )
            err.code = 'unsupported_capability'
            raise err
        code = 'invalid_stall_guard_preset'
    elif field == 'grip_guard_preset':
        if profile.gripper.empty_close_aperture_m is None:
            raise ToolInputError(
                'invalid_grip_guard_preset',
                field_path=field_path,
                actual=val,
                suggestion='Named grip_guard_preset values require a measured empty-close reading on this robot; use min_aperture_m / max_aperture_m.',
            )
        published = ['none', *(name for name in SCHEMA_GRIP_GUARD_PRESETS if name in profile.gripper.preset_margins_m)]
        accepted = (
            set(profile.gripper.preset_margins_m)
            if (allow_legacy_aliases and profile.robot_model_id == 'arx-x5')
            else set(published)
        )
        code = 'invalid_grip_guard_preset'
    elif field == 'joint_travel_preset':
        if 'joint_travel' not in profile.supports:
            raise ToolInputError(
                'unsupported_capability: joint_travel_preset',
                field_path=field_path,
                actual=val,
                suggestion='joint_travel_preset is not supported on this robot; omit joint_travel_preset.',
            )
        published = list(joint_travel_presets())
        accepted = set(published)
        code = 'invalid_joint_travel_preset'
    elif field == 'clearance_preset':
        if 'clearance' not in profile.supports:
            raise ToolInputError(
                'unsupported_capability: clearance_preset',
                field_path=field_path,
                actual=val,
                suggestion='clearance_preset is not supported on this robot; omit clearance_preset.',
            )
        published = list(CLEARANCE_PRESETS)
        accepted = set(published)
        code = 'invalid_clearance_preset'
    elif field == 'target_exemption_preset':
        if 'review_proxies' not in profile.supports:
            raise ToolInputError(
                'unsupported_capability: target_exemption_preset',
                field_path=field_path,
                actual=val,
                suggestion='target_exemption_preset is not supported on this robot; omit target_exemption_preset.',
            )
        published = list(TARGET_EXEMPTION_PRESETS)
        accepted = set(published)
        code = 'invalid_target_exemption_preset'
    else:
        return
    if not isinstance(val, str) or val not in accepted:
        raise ToolInputError(
            code,
            field_path=field_path,
            actual=val,
            suggestion=f"Choose one of the supported {field} values on this robot: {', '.join(published)}.",
            minimal_example={field: published[0]} if published else None,
        )


def validate_raw_action_presets(args, embodiment=None) -> None:
    """在 plan_update 与受审 Program 投影前，对原始 robot_action 请求校验模式字段边界与 6 类预设合法性。"""
    if not isinstance(args, dict):
        return
    profile = _resolve_profile(embodiment)
    mode = args.get('mode', 'preview')
    is_cached_clean_confirm = (
        mode == 'execute_if_clean'
        and 'plan_id' in args
        and not any(k in args for k in ('stages', 'source_plan_id', 'skip_failed_stage'))
    )
    if mode == 'inspect':
        if args.get('plan_update') is not None or bool(args.get('auto_execute_if_clean', False)):
            raise ValueError('robot_action_inspect_has_command_fields')
        if set(args) - (_INSPECT_ALLOWED_FIELDS | {'plan_update', 'auto_execute_if_clean'}):
            raise ValueError('robot_action_inspect_has_command_fields')
        return
    if mode == 'execute' or is_cached_clean_confirm:
        if 'plan_id' not in args:
            raise ValueError('robot_action_requires_plan_id')
        if set(args) - _EXECUTE_ALLOWED_FIELDS:
            raise ValueError('cached_batch_commands_cannot_be_changed')
        return
    if mode == 'reach':
        if 'reach' not in profile.supports:
            return
        if set(args) - REACH_REQUEST_FIELDS:
            raise ValueError('robot_action_reach_has_unsupported_fields')
        _check_preset_value(args, 'tolerance_preset', ('tolerance_preset',), profile=profile, allow_legacy_aliases=True)
        for idx, stage in enumerate(args.get('stages') or []):
            if isinstance(stage, dict) and stage.get('kind') == 'move':
                _check_preset_value(
                    stage, 'tolerance_preset', ('stages', idx, 'tolerance_preset'),
                    profile=profile, allow_legacy_aliases=False,
                )
        return
    if mode in ('preview', 'preflight', 'direct', 'execute_if_clean'):
        if set(args) - _PLANNING_ALLOWED_FIELDS:
            raise ValueError('robot_action_preview_has_execute_or_inspect_fields')
        for field in _BATCH_PRESET_FIELDS:
            _check_preset_value(args, field, (field,), profile=profile, allow_legacy_aliases=True)
        stages = args.get('stages')
        if isinstance(stages, list):
            for idx, stage in enumerate(stages):
                if not isinstance(stage, dict):
                    continue
                kind = stage.get('kind')
                stage_fields = (
                    _MOVE_STAGE_PRESET_FIELDS if kind == 'move'
                    else ('grip_guard_preset',) if kind == 'gripper'
                    else ()
                )
                for field in stage_fields:
                    _check_preset_value(
                        stage, field, ('stages', idx, field),
                        profile=profile, allow_legacy_aliases=False,
                    )
        overrides = args.get('stage_overrides')
        if isinstance(overrides, dict):
            for sid, patch in overrides.items():
                if isinstance(patch, dict):
                    for field in _MOVE_STAGE_PRESET_FIELDS:
                        _check_preset_value(
                            patch, field, ('stage_overrides', sid, field),
                            profile=profile, allow_legacy_aliases=True,
                        )
        elif isinstance(overrides, list):
            for idx, item in enumerate(overrides):
                if not isinstance(item, dict):
                    continue
                for field in _MOVE_STAGE_PRESET_FIELDS:
                    _check_preset_value(
                        item, field, ('stage_overrides', idx, field),
                        profile=profile, allow_legacy_aliases=True,
                    )
                nested = item.get('overrides')
                if isinstance(nested, dict):
                    for field in _MOVE_STAGE_PRESET_FIELDS:
                        _check_preset_value(
                            nested, field, ('stage_overrides', idx, 'overrides', field),
                            profile=profile, allow_legacy_aliases=True,
                        )


def _preview_downgrade_reasons(summary, args=None, harness=None):
    """检查 preview/preflight 产物是否存在硬阻断风险（严格仅限本体运动学/安全拒绝或显式障碍冲突），环境几何提示绝不阻断。"""
    if not isinstance(summary, dict) or summary.get('status') != 'planned':
        return ['preview_not_planned']
    reasons = []
    safety = summary.get('motion_safety')
    if isinstance(safety, dict) and safety.get('status') not in (None, 'pass'):
        reasons.append('motion_safety_reject')
    grasp_review = summary.get('grasp_3d_review')
    if isinstance(grasp_review, dict):
        if grasp_review.get('blocking_risks'):
            reasons.append('grasp_3d_blocking_risks')
        elif (
            harness is not None
            and 'jaw_grasp_centre' in embodiment_of(harness).pose_inputs
            and (
                grasp_review.get('status') not in (None, 'aligned')
                or grasp_review.get('warnings')
                or grasp_review.get('risk_flags')
            )
        ):
            reasons.append('grasp_3d_review_warnings')
    geom_review = summary.get('geometry_review')
    if isinstance(geom_review, dict) and any(_is_blocking_geometry_risk(r) for r in (geom_review.get('risks') or [])):
        reasons.append('geometry_review_risks')
    path_check = summary.get('path_check')
    if isinstance(path_check, dict) and path_check.get('status') != 'clear':
        reasons.append('path_check_collisions' if path_check.get('status') == 'collisions' else 'path_check_unavailable')
    if harness is not None and embodiment_of(harness).clock != 'simulated_steps':
        for arm in _closed_jaw_before_grasp_arms(args, harness):
            reasons.append(f'closed_jaw_before_grasp:{arm}')
            break
    return reasons


def preview_downgrade_reasons_for_args(runtime, args, summary=None) -> list[str]:
    """统一将新建、source_plan_id 续跑、修补计划与零游标初始缓存计划解析为展开后的阶段参数再复核清洁度。"""
    if not isinstance(args, dict):
        return _preview_downgrade_reasons(summary, args=None, harness=runtime) if summary is not None else []
    if (
        'plan_id' in args
        and not any(k in args for k in ('stages', 'source_plan_id', 'skip_failed_stage'))
    ):
        record = _lookup(runtime, args['plan_id'])
        plan_summary = summary if isinstance(summary, dict) else record.get('summary')
        if not isinstance(plan_summary, dict) or plan_summary.get('status') != 'planned':
            return []
        if type(plan_summary.get('executed_batch_native_steps')) is not int or plan_summary['executed_batch_native_steps'] != 0:
            return []
        if plan_summary.get('plan_revision', 0) != 0 or plan_summary.get('last_action_id'):
            return []
        expanded_args = record.get('source_args')
        if not isinstance(expanded_args, dict):
            expanded_args = {'stages': record.get('raw_stages') or record.get('source_stages') or []}
        return _preview_downgrade_reasons(plan_summary, args=expanded_args, harness=runtime)
    if 'source_plan_id' in args:
        from .plan_continuation import resolve_continuation_args
        expanded_args, _ = resolve_continuation_args(runtime, args)
        return _preview_downgrade_reasons(summary, args=expanded_args, harness=runtime)
    if 'plan_id' in args:
        from .batch_repair import build_repaired_args
        expanded_args = build_repaired_args(runtime, args, _lookup)
        return _preview_downgrade_reasons(summary, args=expanded_args, harness=runtime)
    return _preview_downgrade_reasons(summary, args=args, harness=runtime)


def _is_preview_clean(summary, args=None, harness=None):
    """判断 preview 产物是否满足直通执行条件（无规划失败、无本体运动安全拒绝、无显式障碍冲突）。"""
    return len(_preview_downgrade_reasons(summary, args=args, harness=harness)) == 0


def _apply_plan_update(harness, plan_update, args, correlation):
    """在 robot_action 内部顺带原子执行 task_plan 的 submit 或 update，消灭独立 task_plan 回合。"""
    if not isinstance(plan_update, dict) or not plan_update:
        return None
    tp = getattr(harness, 'task_planning', None)
    if tp is None:
        return None
    with harness.lock:
        previous = tp.current(harness.identity)
        op = 'update' if previous is not None else 'submit'
        tp_args = {'operation': op, **deepcopy(plan_update)}
        if 'ledger' in tp_args and isinstance(tp_args['ledger'], list):
            normalized_ledger = []
            seen_ids = set()
            for idx, item in enumerate(tp_args['ledger'][:3]):
                if not isinstance(item, dict):
                    continue
                expected = str(item.get('expected') or item.get('event') or item.get('step') or '').strip()
                if not expected:
                    continue
                lid = str(item.get('id') or f'L{idx + 1}').strip()
                while lid in seen_ids:
                    lid = f'{lid}_{idx + 1}'
                seen_ids.add(lid)
                raw_st = str(item.get('status') or 'pending').strip()
                if raw_st == 'completed':
                    st = 'supported'
                elif raw_st in ('pending', 'supported', 'contradicted', 'unknown'):
                    st = raw_st
                else:
                    st = 'pending'
                basis = str(item.get('basis') or 'unknown').strip() or 'unknown'
                next_check = str(item.get('next_check') or 'After current action observation').strip()
                normalized_ledger.append({
                    'id': lid[:80],
                    'expected': expected[:800],
                    'status': st,
                    'basis': basis[:800],
                    'next_check': next_check[:800],
                })
            tp_args['ledger'] = normalized_ledger
        elif op == 'submit' and 'ledger' not in tp_args:
            tp_args['ledger'] = []
        obs = getattr(harness, 'latest', None)
        if obs is None and hasattr(harness, 'current_observation'):
            try:
                obs = harness.current_observation()
            except Exception:
                obs = None
        if args.get('observation_id'):
            tp_args['observation_id'] = args['observation_id']
        elif isinstance(obs, dict) and obs.get('observation_id'):
            tp_args['observation_id'] = obs['observation_id']
        return tp.call(tp_args, harness.identity, obs, correlation)


def _render_downgraded_overlay(harness, summary, camera_id=None):
    """在降级返回计划摘要且显式请求 image_delivery='all' 时，按需渲染候选轨迹叠加图。"""
    import json
    from .inspection import _trajectory_image
    _, raw_traj = harness.store.resolve_artifact(summary['trajectory_ref'], allowed_scopes=('actor',))
    fallback_cam = embodiment_of(harness).overview_camera or 'cam_head'
    overlay_meta, images = _trajectory_image(
        harness, json.loads(raw_traj), camera_id or fallback_cam
    )
    summary['trajectory_overlay'] = overlay_meta
    return images


def _dispatch(harness, args, correlation=None):
    from jsonschema import validate
    profile = embodiment_of(harness)
    # The run's contract: the same build(profile) the registry publishes (fields this embodiment cannot honour are
    # absent, so they fail here instead of being accepted and rejected or ignored later).
    validate(args, robot_action_contract(profile).tool['inputSchema'])
    correlation = correlation or {}
    args = dict(args)
    if args.get('mode') == 'reach':
        # Read-only IK query: refused with plan_update / auto_execute_if_clean before either could act.
        if callable(getattr(getattr(harness, 'robot', None), 'reach_batch', None)):
            validate_raw_action_presets(args, profile)
        return ToolOutput(_reach(harness, args), kind="data").response()
    validate_raw_action_presets(args, profile)
    if 'image_delivery' in args and args.get('mode', 'preview') != 'inspect' and args['image_delivery'] not in IMAGE_DELIVERY_MODES:
        raise ValueError('invalid_image_delivery')
    plan_update = args.pop('plan_update', None)
    auto_execute_if_clean = bool(args.pop('auto_execute_if_clean', False))
    mode = args.get('mode', 'preview')
    is_cached_clean_confirm = (
        mode == 'execute_if_clean'
        and 'plan_id' in args
        and not any(k in args for k in ('stages', 'source_plan_id', 'skip_failed_stage'))
    )
    plan_update_receipt = _apply_plan_update(harness, plan_update, args, correlation) if plan_update is not None else None
    if is_cached_clean_confirm:
        requested_camera_id = args.get('camera_id')
        for benign_key in ('observation_id', 'timeout_s', 'camera_id', 'detail'):
            args.pop(benign_key, None)
        if set(args) - {'mode', 'plan_id', 'action_id', 'image_delivery', 'expected_cursor', 'expected_revision'}:
            raise ValueError('cached_batch_commands_cannot_be_changed')
        exec_args = dict(args, mode='execute')
        has_prior_action = False
        if exec_args.get('action_id'):
            try:
                harness.store.read('action_request', exec_args['action_id'])
                has_prior_action = True
            except KeyError:
                has_prior_action = False
        if not has_prior_action:
            record = _lookup(harness, exec_args['plan_id'])
            cached_summary = record.get('summary') or {}
            if (
                exec_args.get('expected_cursor', 0) == 0
                and exec_args.get('expected_revision', 0) == 0
            ):
                downgrade_reasons = preview_downgrade_reasons_for_args(harness, exec_args, summary=cached_summary)
                if downgrade_reasons:
                    out_summary = deepcopy(cached_summary)
                    out_summary['downgraded_from_execute_if_clean'] = True
                    out_summary['downgrade_reasons'] = downgrade_reasons
                    out_summary.setdefault('physics_advanced', False)
                    out_summary.setdefault('executed_native_steps', 0)
                    if plan_update_receipt is not None:
                        out_summary['plan_update_receipt'] = plan_update_receipt
                    images = (
                        _render_downgraded_overlay(harness, out_summary, camera_id=requested_camera_id)
                        if exec_args.get('image_delivery') == 'all' and out_summary.get('trajectory_ref')
                        else []
                    )
                    return ToolOutput(out_summary, extra_content=tuple(images), kind="plan", projection="action_plan").response()
        return _execute(harness, exec_args, correlation)
    if mode in ('preview', 'preflight', 'direct', 'execute_if_clean'):
        if set(args) - (_PLANNING_ALLOWED_FIELDS - {'plan_update', 'auto_execute_if_clean'}):
            raise ValueError('robot_action_preview_has_execute_or_inspect_fields')
        if 'observation_id' not in args:
            raise ValueError('robot_action_requires_observation_id')
        repair_report = None
        if 'plan_id' in args:
            from .batch_repair import build_repaired_args
            repair_report = {}
            args = build_repaired_args(harness, args, _lookup, repair_report=repair_report)
        render_overlay = (
            mode == 'preview'
            and not auto_execute_if_clean
            and args.get('image_delivery') != 'none'
        )
        summary, images = _preview(harness, args, correlation, render_overlay=render_overlay, repair_report=repair_report)
        if plan_update_receipt is not None and isinstance(summary, dict):
            summary['plan_update_receipt'] = plan_update_receipt
        exec_passthrough = {k: args[k] for k in ('action_id', 'image_delivery') if k in args}
        if mode == 'direct':
            if summary.get('status') == 'planned':
                return _execute(harness, dict(plan_id=summary['plan_id'], **exec_passthrough), correlation)
            summary = dict(summary, receipt=dict(status=summary.get('status', 'rejected'),
                reason=summary.get('reason'), failed_stage_id=summary.get('failed_stage_id'),
                failed_arm=summary.get('failed_arm'), physics_advanced=False, executed_native_steps=0,
                **({k: deepcopy(summary[k]) for k in ('feasible_approach_tilt_angles_deg', 'failure_hint') if k in summary})))
        elif mode == 'execute_if_clean' or auto_execute_if_clean:
            downgrade_reasons = preview_downgrade_reasons_for_args(harness, args, summary=summary)
            if not downgrade_reasons:
                return _execute(harness, dict(plan_id=summary['plan_id'], **exec_passthrough), correlation)
            if isinstance(summary, dict) and summary.get('status') == 'planned':
                summary['downgraded_from_execute_if_clean'] = True
                summary['downgrade_reasons'] = downgrade_reasons
                summary.setdefault('physics_advanced', False)
                summary.setdefault('executed_native_steps', 0)
                if args.get('image_delivery') == 'all' and summary.get('trajectory_ref'):
                    images = _render_downgraded_overlay(harness, summary, camera_id=args.get('camera_id'))
        return ToolOutput(summary, extra_content=tuple(images), kind="plan", projection="action_plan").response()
    if mode == 'execute':
        if 'plan_id' not in args:
            raise ValueError('robot_action_requires_plan_id')
        # 容忍调用方在确认执行缓存计划时顺带传入不修改动作阶段的上下文元字段（如 observation_id、timeout_s）
        for benign_key in ('observation_id', 'timeout_s', 'camera_id', 'detail'):
            args.pop(benign_key, None)
        if set(args)-{'mode', 'plan_id', 'action_id', 'image_delivery', 'expected_cursor', 'expected_revision'}:
            raise ValueError('cached_batch_commands_cannot_be_changed')
        return _execute(harness, args, correlation)
    if set(args)-_INSPECT_ALLOWED_FIELDS:
        raise ValueError('robot_action_inspect_has_command_fields')
    if sum(key in args for key in ('plan_id','action_id','trajectory_ref')) != 1:
        raise ValueError('inspect_requires_one_plan_action_or_trajectory_reference')
    from .inspection import inspect_trajectory
    inspection = {key: value for key,value in args.items() if key not in ('mode', 'plan_id','action_id')}
    prefix=[]
    if 'plan_id' in args:
        record = _lookup(harness, args['plan_id'])
        inspection['trajectory_ref']=record['trajectory_ref']
        prefix=ToolOutput(deepcopy(record['summary']), projection='owner_record').content_parts()
    elif 'action_id' in args:
        action=harness.store.read('action_request',args['action_id'])
        if any(action.get(key)!=harness.identity[key] for key in IDENTITY):
            raise ValueError('stale_environment_identity')
        stored=harness.store.read_state('unified_action_feedback',args['action_id'])
        if stored is None:
            from .trajectory_feedback import build_action_trajectory_feedback
            feedback,_=build_action_trajectory_feedback(harness,action,harness.store.read('action_receipt',args['action_id']),correlation,diagnostic=True)
            if not feedback.get('trajectory_feedback_ref'):
                return ToolOutput(dict(action_id=args['action_id'],available=False,trajectory_feedback=feedback)).response()
            stored=dict(trajectory_ref=feedback['trajectory_feedback_ref'],identity=deepcopy(harness.identity))
            harness.store.set_state('unified_action_feedback',args['action_id'],stored)
        inspection['trajectory_ref']=stored['trajectory_ref']
        inspection.setdefault('path_kind','measured')
    content = inspect_trajectory(harness, inspection, correlation, cached_action=True, structured=True)
    inspected_data = deepcopy(content[0].value)
    if 'plan_id' in args:
        plan_data = dict(deepcopy(record['summary']), inspection=inspected_data)
        return ToolResponse(prefix+content, data=plan_data, kind="plan", data_projection="action_plan")
    return ToolResponse(prefix+content, data=dict(inspected_data, available=True), kind="data", data_projection="owner_record")


def dispatch(harness, args, correlation=None):
    """Replay the same tool call without planning, execution or postprocessing."""
    from ...core.contracts import digest
    correlation = correlation or {}
    key = (correlation.get('program_id'), correlation.get('tool_call_id'), args.get('mode', 'preview'))
    cache = None
    if key[1]:
        cache = slot_get(harness, UNIFIED_TOOL_RESULTS_SLOT, legacy_attr='_unified_tool_results')
        prior = cache.get(key)
        if prior is not None:
            if prior[0] != digest(args):
                raise ValueError('robot_action_tool_call_content_conflict')
            return deepcopy(prior[1])
    result = _dispatch(harness, args, correlation)
    if key[1] and isinstance(cache, dict):
        cache[key] = (digest(args), deepcopy(result))
        while len(cache) > 128:
            cache.pop(next(iter(cache)))
    return result
