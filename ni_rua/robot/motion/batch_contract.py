"""Strict native contract for a completely compiled, reviewable action batch.

Robot facts come from the run's ``EmbodimentProfile`` (``profile``; None = the default robot, ARX-X5 at
code-default thresholds): frame, TCP and limits profiles of the ARX action language, the request deadline maximum
``max_timeout_s`` (s), the stage limits ``max_stages`` / ``stages_per_arm``, tolerance, stall-guard and grip-guard
presets and bounds, and the default / maximum leaf tolerances of backend-compiled (non-ARX) actions.
"""
import math

from ...core.action_validation import capability_arms, embodiment_or_default
from ...core.contracts import register_batch_validator

IMAGE_DELIVERY_MODES = ('none', 'on_anomaly', 'focus_roi', 'head_only', 'all')


def __getattr__(name):
    """``ARMS``: the ARX-X5 worker-internal arm layout (``batch_motion``), the default robot's arms; generic
    validation uses ``capability_arms(capabilities, profile)``."""
    if name == 'ARMS':
        return embodiment_or_default().arms
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def validate_batch_action(action, capabilities=None, *, profile=None):
    """Validate one compiled ``BatchAction``; returns ``{stage id: set of prerequisite stage ids}``.

    ``capabilities``: the backend's capability document (None: the ARX action language). Actions of a backend
    whose capabilities are not the ARX-X5 calibration (``action_compilation.uses_arx_actions``) are checked with
    the shared leaf contract (``core.contracts.validate_action``) and the frame, TCP and limits profile that
    backend reports. ``profile``: the run's ``EmbodimentProfile`` (``harness.embodiment``; None = the default
    robot, ARX-X5).
    """
    from ..model.arx_contract import validate_action
    from .action_compilation import uses_arx_actions
    from .motion_tolerance import (TOLERANCE_FIELDS, STALL_GUARD_FIELDS,
        validate_tolerance_fields, validate_stall_guard_fields)
    from .grip_guard import (
        CARRY_SLIP_GUARD_FIELDS,
        GRIP_GUARD_FIELDS,
        GRIP_SQUEEZE_FIELD,
        validate_batch_carry_slip_guard,
        validate_grip_guard_fields,
        validate_grip_squeeze,
    )
    from .motion_safety import JOINT_TRAVEL_FIELDS, extract_joint_travel_options
    profile = embodiment_or_default(profile)
    custom = capabilities is not None and not uses_arx_actions(capabilities)
    if action.get('type') != 'BatchAction':
        raise ValueError('batch_action_type_required')
    motion_fields = set(TOLERANCE_FIELDS) | set(STALL_GUARD_FIELDS)
    allowed = {'schema_version', 'run_id', 'episode_id', 'env_instance_id', 'env_generation',
        'world_kind', 'type', 'action_id', 'based_on_observation_id', 'mode', 'frame_id',
        'tcp_config_id', 'limits_profile', 'timeout_s', 'stages', 'batch_plan_id',
        'batch_expected_cursor', 'batch_expected_revision', 'review_proxies', 'checkpoint_after_gripper',
        'allow_jaw_symmetry', 'allow_approach_tilt', 'image_delivery',
        *GRIP_GUARD_FIELDS, *CARRY_SLIP_GUARD_FIELDS, *JOINT_TRAVEL_FIELDS} | motion_fields
    if set(action) - allowed:
        raise ValueError('unsupported_batch_action_fields')
    if 'image_delivery' in action and action['image_delivery'] not in IMAGE_DELIVERY_MODES:
        raise ValueError('invalid_image_delivery')
    if 'checkpoint_after_gripper' in action and not isinstance(action['checkpoint_after_gripper'], bool):
        raise ValueError('invalid_checkpoint_after_gripper')
    if 'allow_jaw_symmetry' in action and not isinstance(action['allow_jaw_symmetry'], bool):
        raise ValueError('invalid_allow_jaw_symmetry')
    if 'allow_approach_tilt' in action and not isinstance(action['allow_approach_tilt'], bool):
        raise ValueError('invalid_allow_approach_tilt')
    validate_tolerance_fields(action, profile=profile)
    validate_stall_guard_fields(action, profile=profile)
    if any(k in action for k in GRIP_GUARD_FIELDS):
        validate_grip_guard_fields({k: action[k] for k in GRIP_GUARD_FIELDS if k in action}, profile=profile)
    if any(k in action for k in JOINT_TRAVEL_FIELDS):
        extract_joint_travel_options(action)
    frame = capabilities.get('frame_id') if custom else profile.frame_id
    tcp = capabilities.get('tcp_config_id') if custom else profile.tcp_config_id
    if not frame or not tcp or action.get('mode') != 'absolute' or action.get('frame_id') != frame or action.get('tcp_config_id') != tcp:
        raise ValueError('unsupported_batch_frame_or_tcp')
    limits = {capabilities.get('limits_profile')} if custom else set(profile.limits_profiles)
    if not action.get('limits_profile') or action.get('limits_profile') not in limits:
        raise ValueError('unsupported_batch_limits')
    timeout = action.get('timeout_s')
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not .1 <= timeout <= profile.max_timeout_s:
        raise ValueError('invalid_batch_timeout')
    if 'batch_plan_id' in action and (not isinstance(action['batch_plan_id'], str) or not action['batch_plan_id']):
        raise ValueError('invalid_batch_plan_id')
    if 'batch_expected_cursor' in action and (type(action['batch_expected_cursor']) is not int or action['batch_expected_cursor'] < 0):
        raise ValueError('invalid_batch_expected_cursor')
    if 'batch_expected_revision' in action and (type(action['batch_expected_revision']) is not int or action['batch_expected_revision'] < 0):
        raise ValueError('invalid_batch_expected_revision')
    if 'review_proxies' in action:
        from .proxy_contract import validate_proxies
        validate_proxies(action['review_proxies'])
    stages = action.get('stages')
    # Error codes keep their historical spelling; the limits are the profile's (both robots: 32 / 16 per arm).
    if not isinstance(stages, list) or not 1 <= len(stages) <= profile.max_stages:
        raise ValueError('batch_requires_one_to_thirty_two_stages')
    arms = capability_arms(capabilities, profile)
    ids, dependencies, last, counts = set(), {}, {}, dict.fromkeys(arms, 0)
    stage_allowed = {'id', 'arm', 'kind', 'depends_on', 'action', 'reason', 'checkpoint_after_gripper',
        'allow_jaw_symmetry', 'allow_approach_tilt',
        *GRIP_GUARD_FIELDS, *CARRY_SLIP_GUARD_FIELDS, *JOINT_TRAVEL_FIELDS} | motion_fields
    if profile.gripper.hold_squeeze is not None:
        # grip_squeeze_rad: a jaw-stage field only on a gripper with a bounded holding squeeze (SO-101).
        stage_allowed = stage_allowed | {GRIP_SQUEEZE_FIELD}
    for stage in stages:
        if not isinstance(stage, dict) or set(stage) - stage_allowed:
            raise ValueError('invalid_batch_stage_fields')
        if 'allow_jaw_symmetry' in stage and (stage.get('kind') != 'move' or not isinstance(stage['allow_jaw_symmetry'], bool)):
            raise ValueError('invalid_allow_jaw_symmetry')
        if 'allow_approach_tilt' in stage and (stage.get('kind') != 'move' or not isinstance(stage['allow_approach_tilt'], bool)):
            raise ValueError('invalid_allow_approach_tilt')
        identifier = stage.get('id')
        if not isinstance(identifier, str) or not identifier or len(identifier) > 80 or identifier in ids:
            raise ValueError('batch_requires_unique_stage_ids')
        ids.add(identifier)
        kind, arm = stage.get('kind'), stage.get('arm')
        if kind not in ('move', 'gripper', 'observe') or arm not in (*arms, None):
            raise ValueError('invalid_batch_stage_kind_or_arm')
        deps = stage.get('depends_on', [])
        if not isinstance(deps, list) or any(not isinstance(d, str) for d in deps) or len(set(deps)) != len(deps):
            raise ValueError('invalid_batch_dependencies')
        deps = set(deps)
        owners = arms if arm is None else (arm,)
        for side in owners:
            if side in last:
                deps.add(last[side])
            last[side] = identifier
            counts[side] += 1
        dependencies[identifier] = deps
        if kind == 'observe':
            if ('action' in stage or 'checkpoint_after_gripper' in stage or (set(stage) & motion_fields)
                    or any(k in stage for k in (*GRIP_GUARD_FIELDS, *CARRY_SLIP_GUARD_FIELDS, *JOINT_TRAVEL_FIELDS, GRIP_SQUEEZE_FIELD))
                    or not isinstance(stage.get('reason'), str) or not stage['reason'].strip()):
                raise ValueError('batch_observe_requires_reason')
        else:
            if arm is None or 'reason' in stage or not isinstance(stage.get('action'), dict):
                raise ValueError('batch_command_requires_arm_action')
            if kind == 'gripper' and (set(stage) & motion_fields):
                raise ValueError('invalid_batch_stage_fields')
            validate_tolerance_fields(stage, profile=profile)
            validate_stall_guard_fields(stage, profile=profile)
            own = stage['action']
            if 'checkpoint_after_gripper' in stage:
                if not isinstance(stage['checkpoint_after_gripper'], bool):
                    raise ValueError('invalid_checkpoint_after_gripper')
                if kind == 'move' and not own.get('gripper_events'):
                    raise ValueError('checkpoint_after_gripper_requires_gripper_event')
            if any(k in stage for k in GRIP_GUARD_FIELDS):
                validate_grip_guard_fields({k: stage[k] for k in GRIP_GUARD_FIELDS if k in stage}, profile=profile)
            if GRIP_SQUEEZE_FIELD in stage:
                validate_grip_squeeze(stage, profile=profile)
                if kind == 'move' and not own.get('gripper_events'):
                    raise ValueError('grip_squeeze_rad_requires_gripper_event')
            if any(k in stage for k in JOINT_TRAVEL_FIELDS):
                if kind != 'move':
                    raise ValueError('joint_travel_options_require_move_stage')
                extract_joint_travel_options(stage)
            if own.get('type', 'EEAction') != 'EEAction':
                raise ValueError('batch_stage_requires_ee_action')
            if isinstance(own.get('path'), dict):
                validate_tolerance_fields(own['path'], profile=profile)
                validate_stall_guard_fields(own['path'], profile=profile)
            if custom:
                from ...core.contracts import IDENTITY, validate_action as validate_common
                validate_common(own, {key: action[key] for key in IDENTITY},
                    action['based_on_observation_id'], capabilities, profile=profile)
                points = own['path']['waypoints']
                selected = next(iter(points[0])) if len(points) == 1 and len(points[0]) == 1 else None
                # A leaf without explicit tolerances uses the embodiment default (m / rad).
                tolerance = own['path'].get('position_tolerance_m', profile.default_tolerance.position_m)
                if (type(tolerance) not in (int, float) or not math.isfinite(tolerance)
                        or not 0 < tolerance <= profile.max_position_tolerance_m):
                    raise ValueError('invalid_position_tolerance_m')
                ori_tolerance = own['path'].get('orientation_tolerance_rad', profile.default_tolerance.orientation_rad)
                if (type(ori_tolerance) not in (int, float) or not math.isfinite(ori_tolerance)
                        or not 0 < ori_tolerance <= profile.max_orientation_tolerance_rad):
                    raise ValueError('invalid_orientation_tolerance_rad')
                if own.get('limits_profile') != action['limits_profile']:
                    raise ValueError('unsupported_batch_limits')
                if own['path']['kind'] == 'hold' and points:
                    raise ValueError('batch_hold_requires_empty_waypoints')
            else:
                selected, _ = validate_action(own)
            if kind == 'move':
                if selected != arm or len(own['path']['waypoints']) != 1:
                    raise ValueError('batch_move_requires_one_owned_target')
                if any(event['phase'] != 'during_motion' for event in own.get('gripper_events', [])):
                    raise ValueError('batch_after_motion_jaw_requires_separate_stage')
            elif own['path']['kind'] != 'hold' or len(own.get('gripper_events', [])) != 1:
                raise ValueError('batch_gripper_requires_one_jaw_event')
            if any(event['arm'] != arm for event in own.get('gripper_events', [])):
                raise ValueError('batch_stage_cannot_control_peer_jaw')
    validate_batch_carry_slip_guard(action, profile=profile)
    if any(count > profile.stages_per_arm for count in counts.values()):
        raise ValueError('batch_exceeds_sixteen_stages_per_arm')
    if any(deps - ids or identifier in deps for identifier, deps in dependencies.items()):
        raise ValueError('batch_dependency_unknown_or_self_reference')
    remaining, done = dict(dependencies), set()
    while remaining:
        ready = [identifier for identifier, deps in remaining.items() if deps <= done]
        if not ready:
            raise ValueError('batch_dependency_cycle')
        for identifier in ready:
            done.add(identifier)
            del remaining[identifier]
    if capabilities is not None:
        if capabilities.get('motion_execution_mode', 'resampled') != 'resampled':
            raise ValueError('batch_requires_resampled_execution')
        if len({s.get('arm') for s in stages if s['kind'] != 'observe'}) > 1 and not capabilities.get('dual_arm', False):
            raise ValueError('unsupported_capability:dual_arm')
    return dependencies


def _batch_validator_hook(action, capabilities=None, *, profile=None):
    return validate_batch_action(action, capabilities, profile=profile)


register_batch_validator(_batch_validator_hook)
