"""Compile complete two-arm stage graphs, then execute only the reviewed rows.

Future starting states are commanded endpoints, never future measurements.
Checkpoints stop the clock and retain the exact unexecuted suffix. A changed
state invalidates that suffix; execution never invokes a planner or resampler.
"""
from copy import deepcopy
import math
import time
import uuid

import numpy as np

from .batch_contract import ARMS, validate_batch_action
from .batch_scheduler import ArmLayout, BatchScheduler, pause_receipt, skip_dependents, summarize_batch
from ...core.config import motion_thresholds
from .contact_diagnostics import build_contact_block_summary
from .contact_reanchor import (
    EXACT_CACHED_SEMANTICS,
    REANCHORED_CACHED_SEMANTICS,
    clamp_stage_rows_to_held_q,
    has_reanchored_rows,
    propagate_contact_hold_and_reanchor,
    sync_reanchor_stage_result,
)
from ...core.contracts import digest
from .grip_guard import (
    CARRY_SLIP_GUARD_FIELDS,
    DOWNWARD_MOVE_DZ_M,
    GRIP_GUARD_FIELDS,
    _aperture_from_jaw,
    _extract_jaw,
    compute_grip_facts,
    format_effective_grip_guard,
    jaw_checkpoints,
    jaw_goal,
    measured_aperture_m,
    stage_grip_guard,
)
from .motion_safety import JOINT_TRAVEL_FIELDS, extract_joint_travel_options
from .motion_tolerance import (
    STALL_GUARD_FIELDS,
    TOLERANCE_FIELDS,
    compute_endpoint_residual,
    resolve_stage_tolerance,
    resolve_stall_guard,
)


def _embodiment(adapter):
    """The embodiment profile of this worker's run thresholds and execution mode."""
    explicit = getattr(adapter, 'embodiment', None)
    if explicit is not None:
        return explicit
    from ..model.embodiments import arx_x5_profile
    return arx_x5_profile(thresholds=adapter,
                          motion_execution_mode=getattr(adapter, 'motion_execution_mode', 'resampled'))


def _layout(adapter):
    explicit = getattr(adapter, 'arm_layout', None)
    if isinstance(explicit, ArmLayout):
        return explicit
    return ArmLayout.from_profile(_embodiment(adapter))


def motion_digest(action):
    payload = {key: action.get(key) for key in ('stages', 'frame_id', 'tcp_config_id', 'limits_profile', 'review_proxies', 'checkpoint_after_gripper', 'allow_jaw_symmetry', 'allow_approach_tilt')}
    for key in (*TOLERANCE_FIELDS, *STALL_GUARD_FIELDS, *GRIP_GUARD_FIELDS, *CARRY_SLIP_GUARD_FIELDS, *JOINT_TRAVEL_FIELDS):
        if key in action:
            payload[key] = action[key]
    return digest(payload)


def _arm_grip_state(adapter, state, arm):
    """提取用于计算 grip_facts 的单臂状态，优先使用真实夹爪物理测量开度。"""
    q = list(state['arms'][arm].get('q_rad', [0.0] * 6))
    q_six = q[:6] if len(q) >= 6 else [*q, *([0.0] * (6 - len(q)))]
    jaw = _extract_jaw(state, adapter=adapter, arm=arm)
    if jaw is not None:
        return [*q_six, float(jaw)]
    arm_state = deepcopy(state['arms'][arm])
    if 'gripper' not in arm_state and hasattr(adapter, 'last_dispatched_goals'):
        goal = adapter.last_dispatched_goals.get(arm)
        if goal is not None:
            return [*q_six, float(goal)]
    return arm_state


def _jaw_rows(adapter, stage, held, initial, *, layout=None):
    action, arm = stage['action'], stage['arm']
    slot = (layout or _layout(adapter)).slot(arm)
    jaw_idx = slot.jaw
    goal = action['gripper_events'][0]['goal_normalized']
    scale = action.get('gripper_speed_scale', 1.)
    dwell = max(1, math.ceil(abs(goal-held[jaw_idx]) / adapter.eef_gripper_dwell_step))
    values = []
    if scale < 1:
        count = math.ceil(abs(goal-initial) / (adapter.eef_gripper_dwell_step*scale))
        values += [initial+(goal-initial)*i/count for i in range(1, count+1)]
    values += [goal]*dwell
    rows = []
    for value in values:
        row = list(held)
        row[jaw_idx] = value
        rows.append(row)
    return rows


def _evidence(adapter, before, action, rows, contexts):
    from .trajectory_evidence import build_trajectory_evidence, compact_trajectory_evidence
    layout = _layout(adapter)
    compiled = {arm: (np.array([row[layout.joint_slice(arm)] for row in rows]), rows, [], None)
                for arm in layout.arms}
    # A checkpoint-only batch has a valid initial frame and no command samples.
    if not rows:
        rows_for_evidence = [adapter._held_row(before)]
        compiled = {arm: (np.array([before['arms'][arm]['q_rad']]), rows_for_evidence, [], None) for arm in layout.arms}
    _, full = build_trajectory_evidence(adapter, before, compiled, 'independent', action)
    if not rows:
        full['samples'] = full['samples'][:1]
    for sample, context in zip(full['samples'][1:], contexts):
        for arm in layout.arms:
            if arm in sample.get('arms', {}) and arm in context:
                sample['arms'][arm].update(context[arm])
    full.update(execution_binding='exact_cached_rows; no replanning or resampling',
        jaw_scope='exact cached jaw commands; measured starting aperture checked before dispatch',
        plan_scope='complete batch including checkpoint suffix; future states are planned, not measured')
    compact = compact_trajectory_evidence(full)
    compact['limits'] = [text.replace('execution replans.', 'execution uses these exact cached commands or stops.')
                         for text in compact['limits']]
    compact['details'] = 'robot_action mode=inspect reads the original complete cached trajectory and geometry.'
    return compact, full


def _populate_planned_positions(adapter, before, records, full):
    """为每个 move 阶段生成每步规划末端位置，用于执行期 stall_guard 实时偏差对比。"""
    samples = full.get('samples') or []
    if full.get('geometry', {}).get('status') == 'available' and len(samples) > 1:
        for record in records.values():
            if record['kind'] != 'move':
                continue
            arm = record['arm']
            record['planned_positions_m'] = [
                list(samples[step + 1]['arms'][arm]['link6_pose']['position_m'])
                for step in range(record['start'], record['end'])
            ]
        return

    model = getattr(adapter, 'robot_model', None)
    fk_fn = None
    bases = {}
    ee_links = {}
    get_robot = getattr(getattr(adapter, 'robot', None), 'get_robot_by_arm_name', None)
    if isinstance(model, dict) and model.get('joints'):
        try:
            from ..model.geometry import array, fk, transform
            from ..model.roboprobe_pose import quat_to_matrix
            fk_fn = fk
            for arm in _layout(adapter).arms:
                if arm not in before.get('arms', {}):
                    continue
                robot = get_robot(arm + '_arm') if callable(get_robot) else None
                ee_link = getattr(robot, 'ee_link_name', 'link6')
                ee_links[arm] = ee_link
                state = before['arms'][arm]
                joint_names_src = state.get('joint_names') or getattr(robot, 'arm_joints_name', None)
                if not joint_names_src:
                    raise ValueError('joint_names_unavailable')
                names = list(joint_names_src)
                values = list(state['q_rad'])
                base_ee = fk({**model, 'T_world_base': np.eye(4)},
                    {'joint_names': names, 'q_rad': values, 'source_kind': 'measurement'})[ee_link]
                pose = state['ee_pose']
                q = array(pose['quaternion_xyzw'], (4,))
                world_ee = np.eye(4)
                world_ee[:3, :3] = quat_to_matrix(q[[3, 0, 1, 2]])
                world_ee[:3, 3] = array(pose['position_m'], (3,))
                bases[arm] = transform(world_ee @ np.linalg.inv(base_ee))
        except (ValueError, KeyError, np.linalg.LinAlgError):
            fk_fn = None

    layout = _layout(adapter)
    start_poses = {arm: list(before['arms'][arm]['ee_pose']['position_m']) for arm in layout.arms}
    for _, record in sorted(records.items(), key=lambda item: item[1]['start']):
        if record['kind'] != 'move':
            continue
        arm = record['arm']
        j_slice = layout.joint_slice(arm)
        target_pos = np.asarray(record['target']['position_m'], dtype=float)
        start_pos = np.asarray(start_poses[arm], dtype=float)
        positions = None
        if fk_fn is not None and arm in bases:
            try:
                robot = get_robot(arm + '_arm') if callable(get_robot) else None
                joint_names_src = before['arms'][arm].get('joint_names') or getattr(robot, 'arm_joints_name', None)
                if not joint_names_src:
                    raise ValueError('joint_names_unavailable')
                names = list(joint_names_src)
                cand = [
                    fk_fn({**model, 'T_world_base': bases[arm]},
                          {'joint_names': names, 'q_rad': row[j_slice], 'source_kind': 'estimate'})[ee_links[arm]][:3, 3].tolist()
                    for row in record['rows']
                ]
                if cand and float(np.linalg.norm(np.asarray(cand[-1]) - target_pos)) <= 0.02:
                    positions = cand
                    record['planned_positions_source'] = 'fk'
            except (ValueError, KeyError, np.linalg.LinAlgError):
                positions = None
        if positions is None:
            count = len(record['rows'])
            positions = [
                (start_pos + (target_pos - start_pos) * ((i + 1) / count)).tolist()
                for i in range(count)
            ]
            record['planned_positions_source'] = 'linear_fallback'
        record['planned_positions_m'] = positions
        start_poses[arm] = list(target_pos)


def _check_pose_arrival(measured_pose, target_pose, pos_tol, ori_tol, *,
                        allow_vertical_compliance=False, max_upward_dz_m=None, max_downward_dz_m=None):
    """检查末端到位误差：支持标准 3D 球面容差，或在水平 (dx, dy) 严格对准下的有向 Z 轴接触/负载顺应（零顺应半空间严守 3D 球面）。"""
    from .feedback_motion import pose_error
    position, orientation = pose_error(measured_pose, target_pose)
    residual = compute_endpoint_residual(target_pose, measured_pose)
    if not math.isfinite(position + orientation):
        return False, position, orientation, residual, False
    if position <= pos_tol and orientation <= ori_tol:
        return True, position, orientation, residual, False
    if allow_vertical_compliance:
        xy_error = math.hypot(residual['dx_m'], residual['dy_m'])
        dz = residual['dz_m']
        z_down_tol = float(max_downward_dz_m) if max_downward_dz_m is not None else 0.0
        z_up_tol = float(max_upward_dz_m) if max_upward_dz_m is not None else max(pos_tol, 0.040)
        if xy_error <= pos_tol and -z_down_tol <= dz <= z_up_tol and orientation <= ori_tol:
            return True, position, orientation, residual, True
    return False, position, orientation, residual, False


def _open_jaw_symmetric_grasp_twin(own, arm, target):
    """构造开爪状态下绕局部接近轴 `+X_local` 旋转 180° 的精确 SO(3) 平行双指对称等价动作副本（零三维位置与轴向偏差）。"""
    from ..model import roboprobe_pose

    gp_meta = own.get('input_grasp_pose')
    gp_arm = gp_meta.get(arm) if isinstance(gp_meta, dict) and arm in gp_meta else gp_meta
    waypoints = (own.get('path') or {}).get('waypoints') or []
    quat = target.get('quaternion_xyzw') if isinstance(target, dict) else None
    if (
        len(waypoints) != 1
        or not isinstance(gp_arm, dict)
        or not isinstance(gp_arm.get('angles_deg'), (list, tuple))
        or len(gp_arm['angles_deg']) != 3
        or not isinstance(quat, (list, tuple))
        or len(quat) != 4
    ):
        return None
    qx, qy, qz, qw = (float(v) for v in quat)
    twin_quat_xyzw = [qw, qz, -qy, -qx]
    p_deg, r_deg, y_deg = roboprobe_pose.quat_to_angles(np.asarray([-qx, qw, qz, -qy], dtype=float))
    twin_angles = [round(float(p_deg), 4), round(float(r_deg), 4), round(float(y_deg), 4)]
    twin_target = {**deepcopy(target), 'quaternion_xyzw': twin_quat_xyzw}
    twin_own = deepcopy(own)
    twin_own['path']['waypoints'][0][arm] = deepcopy(twin_target)
    twin_gp = twin_own['input_grasp_pose'][arm] if arm in twin_own['input_grasp_pose'] else twin_own['input_grasp_pose']
    twin_gp['requested_angles_deg'] = [float(v) for v in gp_arm['angles_deg']]
    twin_gp['angles_deg'] = twin_angles
    twin_gp['jaw_symmetry_resolved'] = True
    return twin_own, twin_target, twin_angles


def _open_jaw_tilted_grasp_candidates(
    own,
    arm,
    target,
    tilt_offsets_deg=(-15.0, 15.0, -30.0, 30.0),
    *,
    include_symmetric_twin=True,
):
    """构造开爪状态下固定三维接触点 `grasp_position_m` 不变、绕夹爪闭合轴微倾的候选动作序列（默认每档倾角含主分支与 180° 平行双指孪生）。"""
    from ..model import roboprobe_pose
    from ...core.pose_math import quat_multiply, quat_wxyz_to_matrix

    gp_meta = own.get('input_grasp_pose')
    gp_arm = gp_meta.get(arm) if isinstance(gp_meta, dict) and arm in gp_meta else gp_meta
    waypoints = (own.get('path') or {}).get('waypoints') or []
    quat = target.get('quaternion_xyzw') if isinstance(target, dict) else None
    flange_pos = target.get('position_m') if isinstance(target, dict) else None
    offset_raw = gp_arm.get('offset_m') if isinstance(gp_arm, dict) else None
    if (
        len(waypoints) != 1
        or not isinstance(gp_arm, dict)
        or not isinstance(gp_arm.get('position_m'), (list, tuple))
        or len(gp_arm['position_m']) != 3
        or not isinstance(gp_arm.get('angles_deg'), (list, tuple))
        or len(gp_arm['angles_deg']) != 3
        or not isinstance(quat, (list, tuple))
        or len(quat) != 4
        or not isinstance(flange_pos, (list, tuple))
        or len(flange_pos) != 3
        or isinstance(offset_raw, bool)
        or not isinstance(offset_raw, (int, float))
    ):
        return []
    offset_m = float(offset_raw)
    qx, qy, qz, qw = (float(v) for v in quat)
    q0_wxyz = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    r0 = quat_wxyz_to_matrix(q0_wxyz)
    p_contact = np.asarray(gp_arm['position_m'], dtype=np.float64)
    p_flange0 = np.asarray(flange_pos, dtype=np.float64)
    if float(np.linalg.norm((p_flange0 + offset_m * r0[:, 0]) - p_contact)) > 0.005:
        return []

    q_flip_180 = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    twin_choices = (False, True) if include_symmetric_twin else (False,)
    candidates = []
    for tilt_deg in tilt_offsets_deg:
        half = 0.5 * math.radians(float(tilt_deg))
        q_tilt = np.asarray([math.cos(half), 0.0, math.sin(half), 0.0], dtype=np.float64)
        q_tilted = quat_multiply(q0_wxyz, q_tilt)
        for is_twin in twin_choices:
            q_cand = quat_multiply(q_tilted, q_flip_180) if is_twin else q_tilted
            r_cand = quat_wxyz_to_matrix(q_cand)
            cand_flange = p_contact - offset_m * r_cand[:, 0]
            if float(cand_flange[2]) < float(p_contact[2]) - 0.01:
                continue
            p_deg, r_deg, y_deg = roboprobe_pose.quat_to_angles(q_cand)
            cand_angles = [round(float(p_deg), 4), round(float(r_deg), 4), round(float(y_deg), 4)]
            if not all(
                lo <= val <= hi
                for val, (lo, hi) in zip(cand_angles, roboprobe_pose.ANGLE_BOUNDS.values())
            ):
                continue
            cand_quat_xyzw = [float(q_cand[1]), float(q_cand[2]), float(q_cand[3]), float(q_cand[0])]
            cand_target = {
                **deepcopy(target),
                'position_m': [float(cand_flange[0]), float(cand_flange[1]), float(cand_flange[2])],
                'quaternion_xyzw': cand_quat_xyzw,
            }
            cand_own = deepcopy(own)
            cand_own['path']['waypoints'][0][arm] = deepcopy(cand_target)
            cand_gp = cand_own['input_grasp_pose'][arm] if arm in cand_own['input_grasp_pose'] else cand_own['input_grasp_pose']
            cand_gp['requested_angles_deg'] = [float(v) for v in gp_arm['angles_deg']]
            cand_gp['angles_deg'] = cand_angles
            cand_gp['jaw_symmetry_resolved'] = True
            candidates.append((cand_own, cand_target, cand_angles))
    return candidates


def _probe_feasible_approach_tilt_angles(
    *,
    adapter,
    robot,
    arm,
    slot_dof,
    orig_own,
    orig_target,
    start_pos_m,
    expected_q,
    start_jaw,
    obstacles,
    contact_entity_id,
    before_state,
    rows_so_far,
    cap,
    deadline,
    joint_travel_options,
    options,
    include_symmetric_twin=True,
):
    """Read-only diagnostic probe for feasible +/-15/+/-30 deg closing-axis tilts on `path_infeasible`.

    Called strictly when an open-jaw `grasp_pose` move fails planning with `allow_approach_tilt=False`.
    Never mutates stage records, diagnostics, or adapter state. Returns deduplicated feasible
    `[pitch, roll, yaw]` orientations in `deg` (top-down world XYZ convention) keeping `grasp_position_m`
    fixed, or `[]` when no tilted candidate passes planning and motion-safety admission within `deadline`.
    """
    from .motion_safety import adapter_assessment as _stage_safety_assessment
    from .scene_obstacles import plan_move_with_obstacle_clearance

    feasible: list[list[float]] = []
    seen: set[tuple[float, float, float]] = set()
    try:
        candidates = _open_jaw_tilted_grasp_candidates(
            orig_own,
            arm,
            orig_target,
            include_symmetric_twin=include_symmetric_twin,
        )
        for tilt_own, tilt_target, tilt_angles in candidates:
            if time.monotonic() >= deadline:
                break
            probe_diag = []
            tilt_plan, tilt_arc = plan_move_with_obstacle_clearance(
                adapter,
                robot,
                arm,
                tilt_own,
                tilt_target,
                start_pos_m,
                expected_q,
                obstacles,
                contact_entity_id,
                probe_diag,
                before_state=before_state,
                cap_remaining=cap - rows_so_far,
                **options,
            )
            if tilt_plan is None:
                continue
            tilt_positions = (
                tilt_arc
                if tilt_arc is not None
                else adapter._resample_action_plan(
                    tilt_plan[0], arm, tilt_own, start_q=np.asarray(expected_q)
                )
            )
            if tilt_positions is None or len(tilt_positions) == 0:
                continue
            event = next(iter(tilt_own.get('gripper_events', [])), None)
            if event:
                if start_jaw is None:
                    continue
                from .gripper_motion import synchronize

                synchronized = synchronize(
                    tilt_positions,
                    expected_q,
                    start_jaw,
                    event,
                    adapter.eef_gripper_dwell_step * tilt_own.get('gripper_speed_scale', 1.0),
                    cap,
                )
                if synchronized is None:
                    continue
                tilt_positions = synchronized[0]
            if rows_so_far + len(tilt_positions) > cap:
                continue
            st_tilt = deepcopy(before_state)
            st_tilt['arms'][arm]['q_rad'] = expected_q
            tilt_safety = _stage_safety_assessment(
                adapter,
                st_tilt,
                {arm: [q[:slot_dof].tolist() for q in tilt_positions]},
                native_steps=rows_so_far + len(tilt_positions),
                request_remaining=cap,
                **joint_travel_options,
            )
            if tilt_safety.get('enabled') and tilt_safety.get('status') == 'reject':
                continue
            rounded = [round(float(v), 2) for v in tilt_angles]
            key = (rounded[0], rounded[1], rounded[2])
            if key not in seen:
                seen.add(key)
                feasible.append(rounded)
    except ValueError:
        return feasible
    return feasible


def preview_batch(adapter, action):
    try:
        from .batch_continuation import resolve_backend_continuation
        action, continuation_of = resolve_backend_continuation(adapter, action)
    except ValueError as error:
        return dict(status='failed', reason=str(error), physics_advanced=False, executed_native_steps=0)
    profile = _embodiment(adapter)
    layout = _layout(adapter)
    arms_tuple = layout.arms
    dependencies = validate_batch_action(action, adapter.capabilities(), profile=profile)
    if getattr(adapter, 'motion_execution_mode', 'resampled') != 'resampled':
        raise ValueError('batch_requires_resampled_execution')
    if adapter.env.end_flag[0]:
        raise ValueError('batch_episode_ended')
    before = deepcopy(adapter.state())
    binding = adapter._preview_state_binding(before)
    held = adapter._held_row(before)
    initial_jaws = {}
    for arm in arms_tuple:
        try:
            initial_jaws[arm] = adapter._measured_gripper_goal(before, arm)
        except (ValueError, KeyError, AttributeError):
            # Missing jaw measurements matter only if a slow/synchronized jaw is
            # requested; a motion-only batch preserves existing jaw commands.
            initial_jaws[arm] = None
    cap = int(math.ceil(10*adapter.control_hz))
    deadline = time.monotonic()+action['timeout_s']
    stages = action['stages']
    pending = {s['id']: s for s in stages}
    completed, active, records = set(), {}, {}
    targets = {arm: deepcopy(before['arms'][arm]['ee_pose']) for arm in arms_tuple}
    base_thresholds = motion_thresholds(adapter)
    tolerances = dict.fromkeys(arms_tuple, float(base_thresholds.get('default_position_tolerance_m', .010)))
    orientation_tolerances = dict.fromkeys(arms_tuple, float(base_thresholds.get('orientation_tolerance_rad', .15)))
    rows, contexts, checkpoints, boundaries, diagnostics = [], [], [], [], []
    jaw_commanded = set()
    symmetry_resolved_arms = set()
    def verify():
        if adapter._preview_state_binding() != binding:
            raise ValueError('batch_state_changed_during_preview')
        if time.monotonic() > deadline:
            raise ValueError('batch_planning_timeout')
    try:
        while pending or active:
            verify()
            ready = [s for s in pending.values() if dependencies[s['id']] <= completed
                     and (s.get('arm') is None or s['arm'] not in active)]
            observe = [s for s in ready if s['kind'] == 'observe']
            if observe:
                checkpoints.append(dict(cursor=len(rows), stage_ids=[s['id'] for s in observe],
                    reasons={s['id']: s['reason'] for s in observe}))
                for stage in observe:
                    identifier = stage['id']
                    completed.add(identifier)
                    del pending[identifier]
                    records[identifier] = dict(kind='observe', arm=stage.get('arm'), start=len(rows), end=len(rows))
                continue
            for stage in ready:
                arm, identifier, own = stage['arm'], stage['id'], stage['action']
                slot = layout.slot(arm)
                jaw_idx = slot.jaw
                expected_q = layout.extract_arm_q(held, arm)
                held_jaw = float(held[jaw_idx]) if jaw_idx is not None else 1.0
                if arm in jaw_commanded or (stage['kind'] == 'move' and own.get('gripper_speed_scale', 1.) >= 1.):
                    start_jaw = held_jaw if jaw_idx is not None else None
                else:
                    start_jaw = initial_jaws[arm]
                    if start_jaw is not None and len(rows) and held_jaw > start_jaw:
                        start_jaw = min(held_jaw, start_jaw+len(rows)*adapter.eef_gripper_dwell_step)
                jaw_events = own.get('gripper_events') or []
                prev_jaw_val = held_jaw
                jaw_goal_val = float(jaw_events[0].get('goal_normalized', 1.0)) if jaw_events else None
                is_closing_jaw = jaw_goal_val is not None and jaw_goal_val < 0.85
                arm_is_open = prev_jaw_val >= 0.90 and (start_jaw is None or float(start_jaw) >= 0.90)
                grip_guard = stage_grip_guard(stage, action, jaw_goal(own), profile=profile)
                joint_travel_options = extract_joint_travel_options(stage, action) if stage['kind'] == 'move' else {}
                record = dict(kind=stage['kind'], arm=arm, start=len(rows), expected_start_q=expected_q,
                    expected_start_jaw=start_jaw, action=own, target=deepcopy(targets[arm]),
                    position_tolerance_m=tolerances[arm], orientation_tolerance_rad=orientation_tolerances[arm],
                    source_kind='planned_command_state',
                    checkpoint_after_gripper=stage.get('checkpoint_after_gripper'),
                    grip_guard=grip_guard, joint_travel_options=joint_travel_options)
                if stage['kind'] == 'move':
                    path = own.get('path', {})
                    path_tol_fields = {k: path[k] for k in TOLERANCE_FIELDS if k in path}
                    stage_tol_spec = path_tol_fields | {k: stage[k] for k in TOLERANCE_FIELDS if k in stage}
                    resolved_tol = resolve_stage_tolerance(stage_tol_spec, action, profile=profile)
                    stage_guard_spec = {k: path[k] for k in STALL_GUARD_FIELDS if k in path} | {k: stage[k] for k in STALL_GUARD_FIELDS if k in stage}
                    resolved_guard = resolve_stall_guard(stage_guard_spec, action, profile=profile)
                    robot = adapter.robot.get_robot_by_arm_name(arm+'_arm')
                    target = own['path']['waypoints'][0][arm]
                    orig_own, orig_target = own, target
                    options = dict(start_q=np.asarray(expected_q))
                    if getattr(adapter, 'joint_replan_enabled', False):
                        options['joint_replan_context'] = dict(deadline=deadline, speed_scale=own.get('speed_scale', 1.),
                            position_tolerance_m=resolved_tol['position_tolerance_m'])
                    local_diagnostics = []
                    proxies = action.get('review_proxies') or {}
                    from .motion_safety import adapter_assessment as _stage_safety_assessment
                    from .scene_obstacles import plan_move_with_obstacle_clearance
                    resolved_twin_angles = None
                    used_exact_twin = False
                    used_approach_tilt = False
                    plan, resampled_arc = plan_move_with_obstacle_clearance(
                        adapter,
                        robot,
                        arm,
                        own,
                        target,
                        (targets.get(arm) or {}).get('position_m'),
                        expected_q,
                        proxies.get('obstacles') or [],
                        proxies.get('contact_entity_id'),
                        local_diagnostics,
                        before_state=before,
                        cap_remaining=cap - len(rows),
                        **options,
                    )
                    positions = (
                        (resampled_arc if resampled_arc is not None else adapter._resample_action_plan(plan[0], arm, own, start_q=np.asarray(expected_q)))
                        if plan is not None else None
                    )
                    can_try_jaw_twin = (
                        bool(stage.get('allow_jaw_symmetry'))
                        and arm_is_open
                        and own.get('input_grasp_pose') is not None
                    )
                    can_try_approach_tilt = (
                        bool(stage.get('allow_approach_tilt', action.get('allow_approach_tilt', False)))
                        and arm_is_open
                        and own.get('input_grasp_pose') is not None
                    )
                    if can_try_jaw_twin or can_try_approach_tilt:
                        primary_rejected = False
                        branch_flip_needed = False
                        if positions is not None and len(positions):
                            st_check = deepcopy(before)
                            st_check['arms'][arm]['q_rad'] = expected_q
                            prim_safety = _stage_safety_assessment(
                                adapter, st_check, {arm: [q[:slot.dof].tolist() for q in positions]},
                                native_steps=len(rows) + len(positions), request_remaining=cap, **joint_travel_options,
                            )
                            prim_arm_reasons = ((prim_safety.get('arms') or {}).get(arm) or {}).get('reasons') or ()
                            primary_rejected = bool(
                                prim_safety.get('enabled')
                                and prim_safety.get('status') == 'reject'
                                and 'abnormal_joint_travel' in prim_arm_reasons
                            )
                            branch_flip_needed = (
                                can_try_jaw_twin
                                and arm in symmetry_resolved_arms
                                and abs(float(positions[-1][5]) - float(expected_q[5])) > 2.4
                            )
                        if plan is None or primary_rejected or branch_flip_needed:
                            if can_try_jaw_twin:
                                twin_bundle = _open_jaw_symmetric_grasp_twin(orig_own, arm, orig_target)
                                if twin_bundle is not None:
                                    twin_own, twin_target, twin_angles = twin_bundle
                                    twin_diag = []
                                    twin_plan, twin_arc = plan_move_with_obstacle_clearance(
                                        adapter,
                                        robot,
                                        arm,
                                        twin_own,
                                        twin_target,
                                        (targets.get(arm) or {}).get('position_m'),
                                        expected_q,
                                        proxies.get('obstacles') or [],
                                        proxies.get('contact_entity_id'),
                                        twin_diag,
                                        before_state=before,
                                        cap_remaining=cap - len(rows),
                                        **options,
                                    )
                                    if twin_plan is not None:
                                        twin_positions = (
                                            twin_arc if twin_arc is not None
                                            else adapter._resample_action_plan(twin_plan[0], arm, twin_own, start_q=np.asarray(expected_q))
                                        )
                                        st_twin = deepcopy(before)
                                        st_twin['arms'][arm]['q_rad'] = expected_q
                                        twin_safety = _stage_safety_assessment(
                                            adapter, st_twin, {arm: [q[:slot.dof].tolist() for q in twin_positions]},
                                            native_steps=len(rows) + len(twin_positions), request_remaining=cap, **joint_travel_options,
                                        )
                                        twin_ok = not (twin_safety.get('enabled') and twin_safety.get('status') == 'reject')
                                        if twin_ok and (
                                            plan is None
                                            or primary_rejected
                                            or abs(float(twin_positions[-1][5]) - float(expected_q[5])) < 1.0
                                        ):
                                            own, target, plan, resampled_arc, positions = (
                                                twin_own, twin_target, twin_plan, twin_arc, twin_positions
                                            )
                                            local_diagnostics = twin_diag
                                            record['action'] = twin_own
                                            resolved_twin_angles = [round(float(v), 2) for v in twin_angles]
                                            used_exact_twin = True
                            if can_try_approach_tilt and resolved_twin_angles is None and (plan is None or primary_rejected):
                                for tilt_own, tilt_target, tilt_angles in _open_jaw_tilted_grasp_candidates(
                                    orig_own,
                                    arm,
                                    orig_target,
                                    include_symmetric_twin=bool(stage.get('allow_jaw_symmetry', True)),
                                ):
                                    tilt_diag = []
                                    tilt_plan, tilt_arc = plan_move_with_obstacle_clearance(
                                        adapter,
                                        robot,
                                        arm,
                                        tilt_own,
                                        tilt_target,
                                        (targets.get(arm) or {}).get('position_m'),
                                        expected_q,
                                        proxies.get('obstacles') or [],
                                        proxies.get('contact_entity_id'),
                                        tilt_diag,
                                        before_state=before,
                                        cap_remaining=cap - len(rows),
                                        **options,
                                    )
                                    if tilt_plan is None:
                                        continue
                                    tilt_positions = (
                                        tilt_arc if tilt_arc is not None
                                        else adapter._resample_action_plan(tilt_plan[0], arm, tilt_own, start_q=np.asarray(expected_q))
                                    )
                                    st_tilt = deepcopy(before)
                                    st_tilt['arms'][arm]['q_rad'] = expected_q
                                    tilt_safety = _stage_safety_assessment(
                                        adapter, st_tilt, {arm: [q[:slot.dof].tolist() for q in tilt_positions]},
                                        native_steps=len(rows) + len(tilt_positions), request_remaining=cap, **joint_travel_options,
                                    )
                                    if not (tilt_safety.get('enabled') and tilt_safety.get('status') == 'reject'):
                                        own, target, plan, resampled_arc, positions = (
                                            tilt_own, tilt_target, tilt_plan, tilt_arc, tilt_positions
                                        )
                                        local_diagnostics = tilt_diag
                                        record['action'] = tilt_own
                                        resolved_twin_angles = [round(float(v), 2) for v in tilt_angles]
                                        used_approach_tilt = True
                                        break
                    if used_exact_twin and not is_closing_jaw:
                        symmetry_resolved_arms.add(arm)
                    elif is_closing_jaw or not arm_is_open or can_try_jaw_twin:
                        symmetry_resolved_arms.discard(arm)
                    diagnostics.extend(dict(item, arm=arm, stage_id=identifier) for item in local_diagnostics)
                    verify()
                    if plan is None:
                        feedback = adapter._failure_feedback(own, arm, 0, local_diagnostics)
                        feasible_tilts = []
                        if (
                            'grasp_angles' in profile.pose_inputs
                            and arm_is_open
                            and orig_own.get('input_grasp_pose') is not None
                            and not can_try_approach_tilt
                        ):
                            feasible_tilts = _probe_feasible_approach_tilt_angles(
                                adapter=adapter,
                                robot=robot,
                                arm=arm,
                                slot_dof=slot.dof,
                                orig_own=orig_own,
                                orig_target=orig_target,
                                start_pos_m=(targets.get(arm) or {}).get('position_m'),
                                expected_q=expected_q,
                                start_jaw=start_jaw,
                                obstacles=proxies.get('obstacles') or [],
                                contact_entity_id=proxies.get('contact_entity_id'),
                                before_state=before,
                                rows_so_far=len(rows),
                                cap=cap,
                                deadline=deadline,
                                joint_travel_options=joint_travel_options,
                                options=options,
                                include_symmetric_twin=bool(stage.get('allow_jaw_symmetry', True)),
                            )
                        failure_result = dict(
                            status='failed',
                            reason='path_infeasible',
                            failed_stage_id=identifier,
                            failed_arm=arm,
                            physics_advanced=False,
                            executed_native_steps=0,
                            failure_feedback=feedback,
                        )
                        if feasible_tilts:
                            tilt_hint = (
                                f"allow_approach_tilt=false left -15/+15/-30/+30 deg closing-axis tilts unsearched; "
                                f"setting allow_approach_tilt=true on the failed stage (or passing explicit "
                                f"angles_deg={feasible_tilts[0]} deg, keeping grasp_position_m fixed) resolves path_infeasible."
                            )
                            failure_result['feasible_approach_tilt_angles_deg'] = feasible_tilts
                            failure_result['failure_hint'] = tilt_hint
                            if isinstance(feedback, dict):
                                feedback['allow_approach_tilt'] = False
                                feedback['feasible_approach_tilt_angles_deg'] = deepcopy(feasible_tilts)
                                adjustments = feedback.get('suggested_adjustments')
                                if isinstance(adjustments, list):
                                    feedback['suggested_adjustments'] = [tilt_hint, *adjustments]
                                else:
                                    feedback['suggested_adjustments'] = [tilt_hint]
                        return failure_result
                    event = next(iter(own.get('gripper_events', [])), None)
                    jaw_values, schedule = None, None
                    if event:
                        if start_jaw is None:
                            raise ValueError('batch_jaw_requires_measured_initial_opening')
                        from .gripper_motion import synchronize
                        synchronized = synchronize(positions, expected_q, start_jaw, event,
                            adapter.eef_gripper_dwell_step*own.get('gripper_speed_scale', 1.), cap)
                        if synchronized is None:
                            raise ValueError('motion_duration_limit')
                        positions, jaw_values, schedule = synchronized
                        jaw_commanded.add(arm)
                    local = []
                    for i, q in enumerate(positions):
                        row = list(held)
                        layout.write_arm_q(row, arm, q.tolist())
                        if jaw_values is not None and jaw_idx is not None:
                            row[jaw_idx] = float(jaw_values[i])
                        local.append(row)
                    record.update(
                        target=deepcopy(target),
                        tolerance_preset=resolved_tol['tolerance_preset'],
                        position_tolerance_m=resolved_tol['position_tolerance_m'],
                        orientation_tolerance_rad=resolved_tol['orientation_tolerance_rad'],
                        max_upward_dz_m=resolved_tol.get('max_upward_dz_m'),
                        max_downward_dz_m=resolved_tol.get('max_downward_dz_m'),
                        stall_guard_preset=resolved_guard['stall_guard_preset'],
                        max_tracking_divergence_m=resolved_guard['max_tracking_divergence_m'],
                        gripper_motion=schedule,
                        **({'jaw_symmetry_resolved': True, 'resolved_angles_deg': resolved_twin_angles,
                            **({'approach_tilt_resolved': True} if used_approach_tilt else {})}
                           if resolved_twin_angles is not None else {}),
                    )
                else:
                    if start_jaw is None and own.get('gripper_speed_scale', 1.) < 1:
                        raise ValueError('batch_jaw_requires_measured_initial_opening')
                    local = _jaw_rows(adapter, stage, held, start_jaw, layout=layout)
                    jaw_commanded.add(arm)
                    if is_closing_jaw:
                        symmetry_resolved_arms.discard(arm)
                if not local:
                    raise ValueError('batch_empty_stage_commands')
                record.update(rows=local, end=len(rows)+len(local))
                records[identifier] = record
                active[arm] = dict(id=identifier, rows=local, cursor=0)
                del pending[identifier]
            if not active:
                raise ValueError('batch_dependency_deadlock')
            if len(rows) >= cap:
                raise ValueError('motion_duration_limit')
            context = {}
            for arm in arms_tuple:
                arm_slot = layout.slot(arm)
                current = active.get(arm)
                if current:
                    src_row = current['rows'][current['cursor']]
                    held[arm_slot.joints] = src_row[arm_slot.joints]
                    if arm_slot.jaw is not None:
                        held[arm_slot.jaw] = src_row[arm_slot.jaw]
                    context[arm] = dict(phase=records[current['id']]['kind'], stage_id=current['id'])
                else:
                    context[arm] = dict(phase='hold', stage_id=None)
            rows.append(list(held))
            contexts.append(context)
            finished = []
            for arm, current in list(active.items()):
                current['cursor'] += 1
                if current['cursor'] == len(current['rows']):
                    identifier = current['id']
                    completed.add(identifier)
                    finished.append(identifier)
                    if records[identifier]['kind'] == 'move':
                        targets[arm] = records[identifier]['target']
                        tolerances[arm] = records[identifier]['position_tolerance_m']
                        orientation_tolerances[arm] = records[identifier]['orientation_tolerance_rad']
                    del active[arm]
            if finished:
                boundaries.append(dict(cursor=len(rows), finished_stage_ids=finished))
        def previous_jaw_goal(record):
            previous = rows[record['start']-1] if record['start'] else adapter._held_row(before)
            j_col = layout.jaw_index(record['arm'])
            return previous[j_col] if j_col is not None else 1.0
        jaw_checkpoints(records, stages, dependencies, checkpoints,
            bool(action.get('checkpoint_after_gripper', False)), previous_jaw_goal)
        verify()
        from .motion_safety import adapter_assessment
        safety = []
        for identifier, record in records.items():
            if record['kind'] == 'observe':
                continue
            arm = record['arm']
            arm_j_slice = layout.joint_slice(arm)
            state = deepcopy(before)
            state['arms'][arm]['q_rad'] = record['expected_start_q']
            assessment = adapter_assessment(adapter, state, {arm: [r[arm_j_slice] for r in record['rows']]},
                native_steps=record['end'], request_remaining=cap, **record.get('joint_travel_options', {}))
            assessment['start_state_source'] = 'initial_measurement' if record['start'] == 0 else 'preceding_planned_command'
            if record['start']:
                assessment['information_sources'][0] = 'preceding planned named joint endpoint, not future measurement'
            safety.append(dict(stage_id=identifier, **assessment))
        rejected_checks = [s for s in safety if s['enabled'] and s['status'] == 'reject']
        if rejected_checks:
            first_rej = rejected_checks[0]
            first_arm = next((a for a, v in (first_rej.get('arms') or {}).items() if v.get('status') == 'reject'), None)
            return dict(status='failed', reason='motion_safety_rejected',
                failed_stage_id=first_rej.get('stage_id'), failed_arm=first_arm,
                planned_native_steps=len(rows),
                stage_costs={key: dict(arm=value['arm'], kind=value['kind'], start_native_step=value['start'],
                    end_native_step=value['end'], native_steps=value['end']-value['start']) for key, value in records.items()},
                motion_safety_checks=safety,
                physics_advanced=False, executed_native_steps=0)
        compact, full = _evidence(adapter, before, action, rows, contexts)
        _populate_planned_positions(adapter, before, records, full)
        # 透明记录每个有规划位置序列的 move 阶段离散 IK 末端残差（单位 m），不干预到达判定
        for record in records.values():
            if record['kind'] == 'move' and record.get('planned_positions_m'):
                record['planned_ik_residual_m'] = round(
                    float(
                        np.linalg.norm(
                            np.asarray(record['planned_positions_m'][-1], dtype=float)
                            - np.asarray(record['target']['position_m'], dtype=float)
                        )
                    ),
                    5,
                )
        from .batch_geometry import review_batch
        from .scene_obstacles import filter_review_obstacles
        review_proxies = deepcopy(action.get('review_proxies', {}))
        if review_proxies.get('obstacles'):
            kept = filter_review_obstacles(review_proxies['obstacles'], records)
            if kept:
                review_proxies['obstacles'] = kept
            else:
                review_proxies.pop('obstacles', None)
        geometry_review = review_batch(adapter, before, full, review_proxies)
        full['batch_geometry_review'] = geometry_review
        verify()
    except ValueError as error:
        return dict(status='failed', reason=str(error), physics_advanced=False, executed_native_steps=0)
    identifier = 'batch-'+uuid.uuid4().hex
    cache = getattr(adapter, '_batch_plans', None)
    if cache is None:
        cache = adapter._batch_plans = {}
    cache_limit = int(base_thresholds.get('batch_plan_cache_size', 16))
    while len(cache) >= cache_limit:
        cache.pop(next(iter(cache)))
    cache[identifier] = dict(action_digest=motion_digest(action), state_binding=binding, cursor=0,
        checkpoint_index=0, revision=0, status='planned', executable=True, rows=rows, contexts=contexts, stages=records,
        source_action=deepcopy({k: v for k, v in action.items() if k != 'stages'}),
        source_stages=deepcopy(action['stages']), completed_stage_ids=[],
        pending_stage_ids=[s['id'] for s in action['stages']], failed_stage_id=None,
        continuation_of=continuation_of,
        checkpoints=checkpoints, boundaries=boundaries, completed=set(), targets=deepcopy({a: before['arms'][a]['ee_pose'] for a in arms_tuple}),
        tolerances=dict.fromkeys(arms_tuple, float(base_thresholds.get('default_position_tolerance_m', .010))),
        orientation_tolerances=dict.fromkeys(arms_tuple, float(base_thresholds.get('orientation_tolerance_rad', .15))),
        downward_dz_tolerances=dict.fromkeys(arms_tuple, 0.0),
        post_gripper_rebound_dz_m=dict.fromkeys(arms_tuple, 0.0),
        initial_state=before)
    arm_costs = {}
    for arm in arms_tuple:
        arm_j_slice = layout.joint_slice(arm)
        trajectory = np.asarray([before['arms'][arm]['q_rad'], *[row[arm_j_slice] for row in rows]])
        travel = np.abs(np.diff(trajectory, axis=0)).sum(axis=0)
        arm_costs[arm] = dict(motion_native_steps=sum(c[arm]['phase'] == 'move' for c in contexts),
            gripper_native_steps=sum(c[arm]['phase'] == 'gripper' for c in contexts),
            hold_native_steps=sum(c[arm]['phase'] == 'hold' for c in contexts),
            joint_travel_rad=travel.tolist(), sum_joint_travel_rad=float(travel.sum()))
    def _stage_effective_tolerances(value):
        if value.get('kind') != 'move':
            return {}
        pos_tol = float(value['position_tolerance_m'])
        ori_tol = float(value['orientation_tolerance_rad'])
        stall_preset = str(value.get('stall_guard_preset') or 'off')
        raw_up = value.get('max_upward_dz_m')
        if raw_up is not None:
            eff_up = float(raw_up) if float(raw_up) > 0.0 else 0.0
        elif stall_preset == 'contact_stop':
            eff_up = max(pos_tol, 0.040)
        else:
            eff_up = 0.0
        out = dict(
            tolerance_preset=str(value.get('tolerance_preset') or 'normal'),
            position_tolerance_m=round(pos_tol, 5),
            orientation_tolerance_rad=round(ori_tol, 5),
            max_upward_dz_m=round(eff_up, 5),
            stall_guard_preset=stall_preset,
        )
        if value.get('max_downward_dz_m') is not None:
            raw_down = float(value['max_downward_dz_m'])
            out['max_downward_dz_m'] = round(raw_down if raw_down > 0.0 else 0.0, 5)
        if value.get('max_tracking_divergence_m') is not None:
            out['max_tracking_divergence_m'] = round(float(value['max_tracking_divergence_m']), 5)
        return {'effective_tolerances': out}

    planned = dict(status='planned', plan_id=identifier, physics_advanced=False, executed_native_steps=0,
        planned_native_steps=len(rows), nominal_sim_time_s=len(rows)/adapter.control_hz,
        arm_costs=arm_costs,
        stage_costs={
            key: dict(
                arm=value['arm'],
                kind=value['kind'],
                start_native_step=value['start'],
                end_native_step=value['end'],
                native_steps=value['end'] - value['start'],
                **_stage_effective_tolerances(value),
                **({'effective_grip_guard': format_effective_grip_guard(value['grip_guard'])}
                   if isinstance(value.get('grip_guard'), dict) and 'carry_slip_guard' in value['grip_guard'] else {}),
                **({'planned_ik_residual_m': value['planned_ik_residual_m']} if 'planned_ik_residual_m' in value else {}),
                **({'jaw_symmetry_resolved': True, 'resolved_angles_deg': value['resolved_angles_deg'],
                    **({'approach_tilt_resolved': True} if value.get('approach_tilt_resolved') else {})}
                   if value.get('jaw_symmetry_resolved') else {}),
            )
            for key, value in records.items()
        },
        checkpoints=checkpoints, trajectory_evidence=compact, trajectory_data=full, motion_safety_checks=safety,
        geometry_review=geometry_review,
        execution_semantics='exact cached shared rows; stop on changed state or failed measured checkpoint; no replan')
    if continuation_of is not None:
        planned['continuation'] = continuation_of
        planned['continuation_of'] = continuation_of
    return planned


def _validate_paused_state(adapter, cache, action, binding_unchanged):
    """在恢复缓存批次前校验暂停期间是否插入了其他动作，并与阶段衔接容差统一对齐。"""
    from .feedback_motion import pose_error
    if adapter.env.end_flag[0] or not cache.get('paused_stop_confirmed', True):
        return False
    expected_obs = cache.get('expected_observation_id')
    if expected_obs is not None and action.get('based_on_observation_id') != expected_obs:
        return False
    binding = cache.get('state_binding')
    expected_steps = cache.get('paused_native_action_count',
        binding.get('native_action_count') if isinstance(binding, dict) else None)
    native_action_steps = int(adapter.env.take_action_cnt[0])
    if expected_steps is not None and native_action_steps != expected_steps:
        cache['resume_validation'] = dict(
            status='invalid',
            reason='native_action_steps_changed_during_pause',
            state_binding='unchanged' if binding_unchanged else 'revalidated',
            native_action_steps=native_action_steps,
            expected_native_action_steps=expected_steps)
        return False
    thresholds = motion_thresholds(adapter)
    joint_tol = float(action.get('resume_drift_tolerance_rad', getattr(adapter, 'resume_drift_tolerance_rad', thresholds['resume_max_joint_drift_rad'])))
    pos_tol = float(action.get('resume_ee_drift_tolerance_m', thresholds['resume_max_ee_drift_m']))
    angle_tol = float(action.get('resume_orientation_drift_tolerance_rad', max(thresholds['resume_max_orientation_drift_rad'], joint_tol)))
    jaw_tol = float(action.get('resume_jaw_drift_tolerance_normalized', thresholds['resume_max_jaw_drift_normalized']))
    current, paused = adapter.state(), cache['paused_state']
    valid_resume = adapter.last_dispatched_goals == cache['paused_jaw_goals']
    for arm in _layout(adapter).arms:
        q_error = float(np.max(np.abs(np.asarray(current['arms'][arm]['q_rad'])-paused['arms'][arm]['q_rad'])))
        position, angle = pose_error(current['arms'][arm]['ee_pose'], paused['arms'][arm]['ee_pose'])
        valid_resume = (valid_resume and math.isfinite(q_error+position+angle)
                        and q_error <= joint_tol and position <= pos_tol and angle <= angle_tol)
        old_jaw = cache['paused_measured_jaws'][arm]
        if old_jaw is not None:
            try:
                valid_resume = valid_resume and abs(adapter._measured_gripper_goal(current, arm)-old_jaw) <= jaw_tol
            except (ValueError, KeyError, AttributeError):
                valid_resume = False
    cache['resume_validation'] = dict(
        status='valid' if valid_resume else 'invalid',
        state_binding='unchanged' if binding_unchanged else 'revalidated',
        native_action_steps=native_action_steps,
        expected_native_action_steps=expected_steps,
        max_joint_drift_rad=joint_tol, max_ee_drift_m=pos_tol, max_orientation_drift_rad=angle_tol,
        max_jaw_opening_drift_normalized=jaw_tol,
        scope='public robot state and unchanged jaw goals; cached commands unchanged; no scene/contact proof')
    return valid_resume


def inspect_batch(adapter, plan_id):
    """Return a read-only summary of cached batch plan ``plan_id``."""
    cache = getattr(adapter, '_batch_plans', {}).get(plan_id)
    if cache is None:
        raise ValueError('batch_plan_unknown_or_expired')
    return {
        'plan_id': plan_id,
        'status': cache['status'],
        'executable': bool(cache.get('executable', False)),
        'cursor': cache['cursor'],
        'revision': cache['revision'],
        'planned_native_steps': len(cache['rows']),
        'completed_stage_ids': list(cache.get('completed_stage_ids', [])),
        'pending_stage_ids': list(cache.get('pending_stage_ids', [])),
        'failed_stage_id': cache.get('failed_stage_id'),
    }


def consume_batch(adapter, action):
    if 'source_plan_id' in action:
        from .batch_continuation import resolve_backend_continuation
        resolved, _ = resolve_backend_continuation(adapter, action)
        action.clear()
        action.update(resolved)
    validate_batch_action(action, adapter.capabilities(), profile=_embodiment(adapter))
    identifier = action.get('batch_plan_id')
    cache = getattr(adapter, '_batch_plans', {}).get(identifier)
    BatchScheduler.admit_cached_plan(cache, action, digest_value=motion_digest(action))
    cache['resuming'] = cache['status'] == 'awaiting_observation'
    binding_unchanged = cache['state_binding'] == adapter._preview_state_binding()
    if cache['resuming']:
        if not _validate_paused_state(adapter, cache, action, binding_unchanged):
            cache['status'] = 'invalidated'
            cache['executable'] = False
            raise ValueError('batch_plan_stale_state_requires_new_preview')
    elif not binding_unchanged:
        cache['status'] = 'invalidated'
        cache['executable'] = False
        raise ValueError('batch_plan_stale_state_requires_new_preview')
    cache['status'] = 'executing'
    cache['revision'] += 1
    return cache


def _propagate_droop_expected_start_q(records, *, arm, after_step, measured_q):
    """在 -Z 静态负载下垂或开爪回弹合法通过后，同步紧邻同臂 gripper 及下一 move 阶段的起点实测基准关节角（不改动规划指令行 rows）。"""
    q_list = [float(x) for x in measured_q]
    for _, other in sorted(records.items(), key=lambda item: item[1]['start']):
        if other['start'] < after_step or other['kind'] == 'observe' or other.get('arm') != arm:
            continue
        other.setdefault('original_expected_start_q', list(other['expected_start_q']))
        other['expected_start_q'] = list(q_list)
        if other['kind'] == 'move':
            break


def execute_batch(adapter, action, cache, receipt, segments, progress, interrupted, dispatch):
    from .feedback_motion import pose_error
    from .motion_safety import admit_motion
    thresholds = motion_thresholds(adapter)
    profile = _embodiment(adapter)
    layout = _layout(adapter)
    scheduler = BatchScheduler(adapter, layout=layout)
    arms_tuple = layout.arms
    rows, records = cache['rows'], cache['stages']
    start = cache['cursor']
    trace = dict(action_space='native_joint', execution_mode='cached_batch', actions=deepcopy(rows[start:]),
        row_contexts=deepcopy(cache['contexts'][start:]), dt_s=1/adapter.control_hz,
        batch_start_native_step=start, complete_batch_native_steps=len(rows))
    segments.append(trace)
    stage_grip_facts = cache.setdefault('stage_grip_facts', {})
    stage_before_states = cache.setdefault('stage_before_states', {})
    stage_start_z = cache.setdefault('stage_start_z', {})
    pending_final_stage_ids = cache.setdefault('pending_final_stage_ids', [])
    active_arm_grip_guard = cache.setdefault('active_arm_grip_guard', {})
    active_carry_slip_monitors = cache.setdefault('active_carry_slip_monitors', {})
    active_stage_carry_monitors = {}
    results = {identifier: dict(kind=record['kind'], arm=record['arm'],
        status='completed' if identifier in cache['completed'] else 'pending',
        planned_steps=record['end']-record['start'],
        completed_steps=max(0, min(cache['cursor'], record['end'])-record['start']),
        **({'grip_facts': deepcopy(stage_grip_facts[identifier])} if identifier in stage_grip_facts else {}),
        **({'effective_grip_guard': format_effective_grip_guard(record['grip_guard'])}
           if isinstance(record.get('grip_guard'), dict) and 'carry_slip_guard' in record['grip_guard'] else {}))
        for identifier,record in records.items()}
    for identifier, record in records.items():
        sync_reanchor_stage_result(results[identifier], record)
    def _sync_execution_semantics():
        trace['actions'] = deepcopy(rows[start:])
        receipt['preview_execution_semantics'] = (
            REANCHORED_CACHED_SEMANTICS if has_reanchored_rows(records) else EXACT_CACHED_SEMANTICS
        )

    def _save_paused_cache_state(stop_conf):
        cache['paused_stop_confirmed'] = bool(
            isinstance(stop_conf, dict)
            and stop_conf.get('confirmed') is not False
            and stop_conf.get('status') != 'unknown'
        )
        cache['state_binding'] = adapter._preview_state_binding()
        cache['paused_native_action_count'] = int(adapter.env.take_action_cnt[0])
        cache['paused_state'] = deepcopy(adapter.state())
        cache['paused_jaw_goals'] = dict(adapter.last_dispatched_goals)
        cache['paused_measured_jaws'] = {}
        for arm in arms_tuple:
            try:
                cache['paused_measured_jaws'][arm] = adapter._measured_gripper_goal(cache['paused_state'], arm)
            except (ValueError, KeyError, AttributeError):
                cache['paused_measured_jaws'][arm] = None

    receipt.update(batch_plan_id=action['batch_plan_id'], batch_status='executing', stage_results=results)
    _sync_execution_semantics()
    if cache.get('resume_validation'):
        receipt['resume_validation'] = deepcopy(cache['resume_validation'])
    checked_starts = set()
    gripper_start_pos = {}
    stall_state = {}
    move_arms = {record['arm'] for record in records.values() if record['kind'] == 'move'}
    active_arms = {record['arm'] for record in records.values() if record['kind'] != 'observe' and record.get('arm') in arms_tuple}
    failed_recorded = False
    def fail(reason, identifier=None, **extra):
        nonlocal failed_recorded
        already_failed = failed_recorded
        failed_recorded = True
        if not already_failed:
            receipt.update(status='failed', reason=reason, **extra)
            if identifier:
                receipt.update(failed_stage_id=identifier, failed_arm=records[identifier]['arm'])
        if identifier:
            results[identifier]['status'] = 'failed'
            skip_dependents(identifier, action['stages'], stage_results=results)
        if not already_failed or 'stop_confirmation' not in receipt:
            receipt['stop_confirmation'] = scheduler.hold()
    def endpoint(identifier):
        already_failed = failed_recorded
        record = records[identifier]; arm = record['arm']
        measured_pose = adapter.state()['arms'][arm]['ee_pose']
        ref_target = record.get('original_target', record['target'])
        pos_tol = float(record.get('position_tolerance_m', thresholds.get('default_position_tolerance_m', .010)))
        ori_tol = float(record.get('orientation_tolerance_rad', thresholds.get('orientation_tolerance_rad', .15)))
        is_contact_stop = record.get('stall_guard_preset') == 'contact_stop'
        explicit_dz = record.get('max_upward_dz_m')
        explicit_down_dz = record.get('max_downward_dz_m')
        if explicit_dz is not None:
            max_upward_dz = float(explicit_dz)
            has_upward = max_upward_dz > 0.0
        elif is_contact_stop:
            has_upward = True
            max_upward_dz = max(pos_tol, 0.040)
        else:
            has_upward = False
            max_upward_dz = 0.0
        has_downward = explicit_down_dz is not None and float(explicit_down_dz) > 0.0
        max_downward_dz = float(explicit_down_dz) if has_downward else 0.0
        allow_vertical = has_upward or has_downward
        ok, position, orientation, residual, vertical_compliant = _check_pose_arrival(
            measured_pose, ref_target, pos_tol, ori_tol,
            allow_vertical_compliance=allow_vertical,
            max_upward_dz_m=max_upward_dz if has_upward else 0.0,
            max_downward_dz_m=max_downward_dz if has_downward else 0.0,
        )
        eff_tol_dict = dict(
            position_tolerance_m=round(pos_tol, 5),
            orientation_tolerance_rad=round(ori_tol, 5),
            max_upward_dz_m=round(max_upward_dz if has_upward else 0.0, 5),
        )
        if explicit_down_dz is not None:
            eff_tol_dict['max_downward_dz_m'] = round(max_downward_dz, 5)
        results[identifier]['endpoint_error'] = dict(
            position_error_m=position,
            orientation_error_rad=orientation,
            effective_tolerances=eff_tol_dict,
            **({'planned_ik_residual_m': record['planned_ik_residual_m']} if 'planned_ik_residual_m' in record else {}),
        )
        if vertical_compliant:
            results[identifier]['endpoint_error']['xy_error_m'] = round(math.hypot(residual['dx_m'], residual['dy_m']), 5)
            results[identifier]['endpoint_error']['vertical_compliance_accepted'] = True
        results[identifier]['endpoint_residual_m'] = residual
        if not already_failed:
            receipt['endpoint_residual_m'] = residual
        if record['action'].get('input_grasp_pose'):
            from ..model.grasp_contact import contact_target_feedback
            try:
                results[identifier]['grasp_contact_feedback'] = contact_target_feedback(record['action'], adapter.state())
            except (ValueError, KeyError, TypeError, ArithmeticError) as error:
                results[identifier]['grasp_contact_feedback'] = dict(available=False, reason='diagnostic_unavailable', error_type=type(error).__name__)
        if not ok:
            if not already_failed and 'first_divergence' in results[identifier]:
                receipt['first_divergence'] = deepcopy(results[identifier]['first_divergence'])
            summary = build_contact_block_summary(
                identifier, arm, ref_target, measured_pose,
                results[identifier].get('grasp_contact_feedback'),
                completed_steps=results[identifier]['completed_steps'],
                planned_steps=results[identifier]['planned_steps'],
            )
            results[identifier]['contact_block_summary'] = summary
            if not already_failed:
                receipt['contact_block_summary'] = summary
            expected_jaw = record.get('expected_start_jaw')
            if expected_jaw is not None and float(expected_jaw) < 0.85:
                cur_jaw = _extract_jaw(adapter.state(), adapter=adapter, arm=arm)
                cur_ap = _aperture_from_jaw(cur_jaw, adapter=adapter, arm=arm)
                if cur_ap is not None and cur_ap > 0.004:
                    results[identifier]['in_hand_tracking_error'] = True
                    receipt['in_hand_tracking_error'] = True
            fail('tracking_error', identifier)
            return False
        if vertical_compliant:
            record['vertical_compliance_accepted'] = True
            record['accepted_pose'] = deepcopy(measured_pose)
            results[identifier]['completion_quality'] = 'vertical_compliance_within_xy_tolerance'
        elif record.get('kind') == 'move' and 0.005 < position <= pos_tol:
            results[identifier]['endpoint_error']['within_1cm_soft_tolerance'] = True
            results[identifier]['completion_quality'] = 'within_1cm_tolerance'
        return True

    def _complete_finished_stages(finished, post_step_state=None, *, pause_on_guard=True):
        if not finished:
            return True, []
        # Check all move arms ending on the row before admitting any successor.
        finished_moves = [identifier for identifier in finished if records[identifier]['kind'] == 'move']
        valid = [endpoint(identifier) for identifier in finished_moves]
        if not all(valid):
            return False, []
        after_state = adapter.state()
        guard_triggered = []
        for identifier in finished:
            record = records[identifier]
            arm = record['arm']
            cache['completed'].add(identifier)
            results[identifier]['status'] = 'completed'
            if record['kind'] == 'move':
                pos_tol = float(record.get('position_tolerance_m', thresholds.get('default_position_tolerance_m', .010)))
                ori_tol = float(record.get('orientation_tolerance_rad', thresholds.get('orientation_tolerance_rad', .15)))
                res_dz = (results[identifier].get('endpoint_residual_m') or {}).get('dz_m', 0.0)
                # 仅对 +Z 向支撑接触早停钳位后续保持行；-Z 向静态负载重力下垂必须维持原规划关节指令以保留 PD 抗重力恢复力矩
                upward_contact_accepted = bool(
                    record.get('contact_stopped_early')
                    or (record.get('vertical_compliance_accepted') and res_dz >= 0.0)
                )
                if upward_contact_accepted:
                    cache['targets'][arm] = deepcopy(record.get('accepted_pose', record['target']))
                    cache.setdefault('downward_dz_tolerances', {})[arm] = 0.0
                else:
                    cache['targets'][arm] = record['target']
                    eff_down = (
                        float(record['max_downward_dz_m'])
                        if record.get('max_downward_dz_m') is not None and float(record['max_downward_dz_m']) > 0.0
                        else 0.0
                    )
                    cache.setdefault('downward_dz_tolerances', {})[arm] = eff_down
                cache.setdefault('post_gripper_rebound_dz_m', {})[arm] = 0.0
                is_contact_stop = record.get('stall_guard_preset') == 'contact_stop'
                if res_dz > 0.001 or upward_contact_accepted:
                    held_q = [float(x) for x in after_state['arms'][arm]['q_rad']]
                    held_pos = after_state['arms'][arm]['ee_pose']['position_m']
                    propagate_contact_hold_and_reanchor(
                        adapter, records, rows, results,
                        arm=arm, after_step=record['end'], held_q=held_q, held_pos=held_pos,
                        source_stage_id=identifier, source_record=record, reanchor_next_move=is_contact_stop,
                        default_pos_tol=float(thresholds.get('default_position_tolerance_m', .010)),
                        layout=_layout(adapter),
                    )
                    _sync_execution_semantics()
                elif record.get('vertical_compliance_accepted') and res_dz < 0.0:
                    _propagate_droop_expected_start_q(
                        records,
                        arm=arm,
                        after_step=record['end'],
                        measured_q=after_state['arms'][arm]['q_rad'],
                    )
                cache['tolerances'][arm] = pos_tol
                cache.setdefault('orientation_tolerances', {})[arm] = ori_tol
            elif record['kind'] == 'gripper':
                if post_step_state is None:
                    post_step_state = after_state
                end_pose = post_step_state['arms'][arm]['ee_pose']
                end_pos = np.asarray(end_pose['position_m'], dtype=float)
                start_pos = gripper_start_pos.get(identifier, np.asarray(record['target']['position_m'], dtype=float))
                results[identifier]['compliance_shift_m'] = round(float(np.linalg.norm(end_pos - start_pos)), 5)
                prev_down = float(cache.get('downward_dz_tolerances', {}).get(arm, 0.0))
                cache['targets'][arm] = deepcopy(end_pose)
                cache.setdefault('post_gripper_rebound_dz_m', {})[arm] = prev_down
                anchor_src_id = record.get('contact_anchor_source_id')
                if anchor_src_id and anchor_src_id in records:
                    held_q = [float(x) for x in after_state['arms'][arm]['q_rad']]
                    propagate_contact_hold_and_reanchor(
                        adapter, records, rows, results,
                        arm=arm, after_step=record['end'], held_q=held_q, held_pos=end_pos,
                        source_stage_id=anchor_src_id, source_record=records[anchor_src_id], reanchor_next_move=True,
                        default_pos_tol=float(thresholds.get('default_position_tolerance_m', .010)),
                        layout=_layout(adapter),
                    )
                    _sync_execution_semantics()
                elif prev_down > 0.0:
                    arm_pos_tol = float(cache.get('tolerances', {}).get(arm, thresholds.get('default_position_tolerance_m', .010)))
                    arm_ori_tol = float(cache.get('orientation_tolerances', {}).get(arm, thresholds.get('orientation_tolerance_rad', .15)))
                    ok_rebound, _, _, _, _ = _check_pose_arrival(
                        end_pose,
                        record['target'],
                        arm_pos_tol,
                        arm_ori_tol,
                        allow_vertical_compliance=True,
                        max_upward_dz_m=max(arm_pos_tol, prev_down),
                        max_downward_dz_m=max(arm_pos_tol, prev_down),
                    )
                    if ok_rebound:
                        _propagate_droop_expected_start_q(
                            records,
                            arm=arm,
                            after_step=record['end'],
                            measured_q=after_state['arms'][arm]['q_rad'],
                        )
            before_grip = stage_before_states.get(identifier)
            if before_grip is None:
                before_grip = [*record['expected_start_q'], record['expected_start_jaw'] if record['expected_start_jaw'] is not None else 0.0]
            after_grip = _arm_grip_state(adapter, after_state, arm)
            grip_facts = compute_grip_facts(before_grip, after_grip, adapter=adapter, arm=arm)
            results[identifier]['grip_facts'] = grip_facts
            stage_grip_facts[identifier] = deepcopy(grip_facts)
            has_remaining_stages = any(other_id not in cache['completed'] and other_id not in finished for other_id in records)
            has_arm_remaining = any(
                other_id not in cache['completed'] and other_id not in finished and records[other_id].get('arm') == arm
                for other_id in records
            )
            start_z = (stall_state.get(identifier) or {}).get('start_z', stage_start_z.get(identifier))
            target_z = float(record['target']['position_m'][2]) if record.get('target') else None
            is_downward_move = (start_z is not None and target_z is not None and (target_z - start_z) < DOWNWARD_MOVE_DZ_M)
            stage_goal = jaw_goal(record.get('action'))
            active_stage_carry_monitors.pop(identifier, None)
            BatchScheduler.finish_stage_carry_slip(active_carry_slip_monitors, arm, record['kind'], stage_goal)
            guard_eval, triggered = BatchScheduler.evaluate_post_stage_grip_guard(
                active_arm_grip_guard,
                active_carry_slip_monitors,
                arm,
                identifier,
                record['kind'],
                stage_goal,
                record.get('grip_guard'),
                grip_facts,
                has_remaining_stages=has_remaining_stages,
                has_arm_remaining=has_arm_remaining,
                is_downward_move=is_downward_move,
                profile=profile,
            )
            if guard_eval is not None:
                results[identifier]['grip_guard'] = guard_eval
            if triggered:
                guard_triggered.append((identifier, guard_eval))
        if guard_triggered and pause_on_guard:
            for identifier, guard_eval in guard_triggered:
                results[identifier]['status'] = 'paused_at_checkpoint'
                results[identifier]['checkpoint_reason'] = guard_eval['reason']
                results[identifier]['checkpoint_reason_code'] = guard_eval['reason']
            if cache['checkpoint_index'] < len(cache['checkpoints']) and cache['checkpoints'][cache['checkpoint_index']]['cursor'] == cache['cursor']:
                marker = cache['checkpoints'][cache['checkpoint_index']]
                cache['checkpoint_index'] += 1
                for obs_id in marker['stage_ids']:
                    cache['completed'].add(obs_id)
                    if obs_id not in dict(guard_triggered):
                        results[obs_id]['status'] = 'completed'
            cache['status'] = 'awaiting_observation'
            stop_conf = scheduler.hold()
            first_reason = guard_triggered[0][1]['reason']
            receipt.update(
                pause_receipt(
                    [i for i, _ in guard_triggered],
                    {i: g['message'] for i, g in guard_triggered},
                    stop_conf,
                    checkpoint_reason=first_reason,
                    checkpoint_reason_code=first_reason,
                )
            )
            _save_paused_cache_state(stop_conf)
            return False, guard_triggered
        return True, guard_triggered

    try:
        remaining = scheduler.step_budget_remaining()
        if remaining is not None:
            receipt['batch_budget_admission'] = dict(remaining_native_steps=remaining,
                required_suffix_native_steps=len(rows)-cache['cursor'])
            if len(rows)-cache['cursor'] > remaining:
                fail('batch_remaining_native_budget_insufficient')
                return
        if cache.get('resuming') and pending_final_stage_ids:
            deferred = [sid for sid in list(pending_final_stage_ids) if sid not in cache['completed']]
            pending_final_stage_ids.clear()
            ok_deferred, _ = _complete_finished_stages(deferred)
            if not ok_deferred:
                return
        if cache.get('resuming') and start < len(rows):
            state = adapter.state()
            paused = cache.get('paused_state', state)
            paused_jaws = cache.get('paused_measured_jaws', {})
            for arm in arms_tuple:
                slot_arm = layout.slot(arm)
                j_slice = slot_arm.joints
                jaw_col = slot_arm.jaw
                identifier = cache['contexts'][start][arm]['stage_id']
                if identifier is None:
                    continue
                record = records[identifier]; own = record['action']
                reanchor_err = record.get('contact_reanchor_error')
                if reanchor_err:
                    sync_reanchor_stage_result(results[identifier], record)
                    fail(reanchor_err['reason'], identifier, contact_reanchor=deepcopy(reanchor_err['details']))
                    return
                prev_row_q = (
                    np.asarray(record['expected_start_q'], dtype=float)
                    if record['start'] == start and (record.get('reanchored_from_contact') or record.get('contact_hold_clamped'))
                    else np.asarray(rows[start-1][j_slice])
                )
                reference_q = (np.asarray(state['arms'][arm]['q_rad']) if start == 0 else
                    prev_row_q+np.asarray(state['arms'][arm]['q_rad'])-np.asarray(paused['arms'][arm]['q_rad']))
                jump = np.abs(np.asarray(rows[start][j_slice])-reference_q)
                limits = adapter._speed_step_limits(arm, own.get('speed_scale', 1.))
                if start > 0:
                    planned_step = np.abs(np.asarray(rows[start][j_slice]) - np.asarray(rows[start-1][j_slice]))
                    limits = np.maximum(limits, planned_step)
                if (jump > limits+1e-9).any():
                    fail('batch_resume_first_joint_step_exceeds_limit', identifier,
                        first_joint_step_rad=jump.tolist(), max_joint_step_rad=limits.tolist())
                    return
                if jaw_col is not None and own.get('gripper_events') and (own.get('gripper_speed_scale', 1.) < 1 or own['gripper_events'][0]['phase'] == 'during_motion'):
                    measured = adapter._measured_gripper_goal(state, arm)
                    planned_jaw = record['expected_start_jaw'] if record['start'] == start else rows[start-1][jaw_col]
                    paused_jaw = paused_jaws.get(arm)
                    reference_jaw = (measured if planned_jaw is None or paused_jaw is None else
                        planned_jaw+(measured-paused_jaw))
                    bound = adapter.eef_gripper_dwell_step*own.get('gripper_speed_scale', 1.)
                    if abs(rows[start][jaw_col]-reference_jaw) > bound+1e-12:
                        fail('batch_resume_first_jaw_step_exceeds_limit', identifier)
                        return
                if record['start'] < start:
                    mon = BatchScheduler.prepare_stage_carry_slip(
                        active_carry_slip_monitors, arm, identifier, record['kind'],
                        jaw_goal(own), results[identifier], record.get('grip_guard'),
                    )
                    if mon is not None:
                        active_stage_carry_monitors[identifier] = mon
        while True:
            if interrupted('execution'):
                return
            cursor = cache['cursor']
            marker_index = cache['checkpoint_index']
            if marker_index < len(cache['checkpoints']) and cache['checkpoints'][marker_index]['cursor'] == cursor:
                marker = cache['checkpoints'][marker_index]
                cache['checkpoint_index'] += 1
                for identifier in marker['stage_ids']:
                    cache['completed'].add(identifier)
                    results[identifier]['status'] = 'completed'
                cache['status'] = 'awaiting_observation'
                stop_conf = scheduler.hold()
                reason_code = (
                    'observation_checkpoint'
                    if any(records.get(sid, {}).get('kind') == 'observe' for sid in marker['stage_ids'])
                    else 'jaw_checkpoint'
                )
                receipt.update(
                    pause_receipt(
                        marker['stage_ids'],
                        marker['reasons'],
                        stop_conf,
                        checkpoint_reason_code=reason_code,
                    )
                )
                _save_paused_cache_state(stop_conf)
                return
            if cursor == len(rows):
                current_state = adapter.state()
                for arm in arms_tuple:
                    if arm not in active_arms:
                        initial_pose = cache['initial_state']['arms'][arm]['ee_pose']
                        drift_pos, _ = pose_error(current_state['arms'][arm]['ee_pose'], initial_pose)
                        if math.isfinite(drift_pos):
                            receipt['passive_arm_drift_m'] = round(float(drift_pos), 5)
                            receipt.setdefault('passive_arm_drifts_m', {})[arm] = round(float(drift_pos), 5)
                        continue
                    if arm not in move_arms:
                        continue
                    measured_pose = current_state['arms'][arm]['ee_pose']
                    pos_tol = cache['tolerances'][arm]
                    ori_tol = cache.get('orientation_tolerances', {}).get(arm, thresholds.get('orientation_tolerance_rad', .15))
                    down_tol = float(cache.get('downward_dz_tolerances', {}).get(arm, 0.0))
                    rebound_tol = float(cache.get('post_gripper_rebound_dz_m', {}).get(arm, 0.0))
                    use_vertical = (down_tol > 0.0) or (rebound_tol > 0.0)
                    ok, _, _, residual, _ = _check_pose_arrival(
                        measured_pose, cache['targets'][arm], pos_tol, ori_tol,
                        allow_vertical_compliance=use_vertical,
                        max_upward_dz_m=rebound_tol if rebound_tol > 0.0 else 0.0,
                        max_downward_dz_m=down_tol if down_tol > 0.0 else 0.0,
                    )
                    if not ok:
                        receipt['endpoint_residual_m'] = residual
                        fail('tracking_error', failed_arm=arm, failed_phase='final_held')
                        return
                cache['status'] = 'completed'
                receipt.update(status='completed', reason=None)
                return
            state = adapter.state()
            starts = [identifier for identifier,record in records.items() if record['kind'] != 'observe'
                      and record['start'] == cursor and identifier not in checked_starts]
            base_drift_limit = float(thresholds.get('segment_start_joint_drift_rad', .10))
            for identifier in starts:
                record = records[identifier]; arm = record['arm']; own = record['action']
                reanchor_err = record.get('contact_reanchor_error')
                if reanchor_err:
                    sync_reanchor_stage_result(results[identifier], record)
                    fail(reanchor_err['reason'], identifier, contact_reanchor=deepcopy(reanchor_err['details']))
                    return
                sync_reanchor_stage_result(results[identifier], record)
                arm_slot = layout.slot(arm)
                start_drift_limit = max(base_drift_limit, float(cache.get('orientation_tolerances', {}).get(arm, .10)))
                drift = float(np.max(np.abs(np.asarray(state['arms'][arm]['q_rad'])-record['expected_start_q'])))
                if drift > start_drift_limit:
                    fail('batch_segment_start_changed', identifier, max_joint_start_error_rad=drift)
                    return
                if record['kind'] == 'gripper':
                    gripper_start_pos[identifier] = np.asarray(state['arms'][arm]['ee_pose']['position_m'], dtype=float).copy()
                elif record['kind'] == 'move':
                    start_pos_arr = np.asarray(state['arms'][arm]['ee_pose']['position_m'], dtype=float).copy()
                    stage_start_z[identifier] = float(start_pos_arr[2])
                    stall_state[identifier] = {
                        'consecutive': 0,
                        'last_div': None,
                        'last_pos': start_pos_arr,
                        'start_z': float(start_pos_arr[2]),
                    }
                events = own.get('gripper_events', [])
                if arm_slot.jaw is not None and events and (own.get('gripper_speed_scale', 1.) < 1 or events[0]['phase'] == 'during_motion'):
                    measured = adapter._measured_gripper_goal(state, arm)
                    reference_jaw = record['expected_start_jaw'] if record['expected_start_jaw'] is not None else measured
                    first = record['rows'][0][arm_slot.jaw]
                    if abs(first-reference_jaw) > adapter.eef_gripper_dwell_step*own.get('gripper_speed_scale', 1.)+1e-12:
                        fail('gripper_start_state_changed', identifier)
                        return
                if not admit_motion(adapter, state, {arm: [r[arm_slot.joints] for r in record['rows']]}, receipt,
                        native_steps=len(rows)-cursor, request_remaining=int(math.ceil(10*adapter.control_hz))-cursor,
                        stage_id=identifier, **record.get('joint_travel_options', {})):
                    return
                stage_before_states[identifier] = _arm_grip_state(adapter, state, arm)
                mon = BatchScheduler.prepare_stage_carry_slip(
                    active_carry_slip_monitors, arm, identifier, record['kind'],
                    jaw_goal(own), results[identifier], record.get('grip_guard'),
                )
                if mon is not None:
                    active_stage_carry_monitors[identifier] = mon
                else:
                    active_stage_carry_monitors.pop(identifier, None)
                checked_starts.add(identifier)
                results[identifier]['status'] = 'running'
            previous = receipt['executed_steps']
            attempted = [context['stage_id'] for context in cache['contexts'][cursor].values() if context['stage_id']]
            for identifier in attempted:
                results[identifier]['status'] = 'running'
                results[identifier]['dispatch_attempted_steps'] = results[identifier].get('dispatch_attempted_steps', 0)+1
            try:
                succeeded = dispatch(rows[cursor], command_phase='batch')
            except Exception:
                if receipt['executed_steps'] > previous:
                    cache['cursor'] += 1
                    for arm in arms_tuple:
                        identifier = cache['contexts'][cursor][arm]['stage_id']
                        if identifier:
                            results[identifier]['completed_steps'] += 1
                for identifier in attempted:
                    results[identifier]['execution_uncertain'] = True
                raise
            if receipt['executed_steps'] > previous:
                cache['cursor'] += 1
                for arm in arms_tuple:
                    identifier = cache['contexts'][cursor][arm]['stage_id']
                    if identifier:
                        results[identifier]['completed_steps'] += 1
            if not succeeded:
                return
            post_step_state = None
            slip_triggers = []
            for arm in arms_tuple:
                identifier = cache['contexts'][cursor][arm]['stage_id']
                if not identifier or identifier not in active_stage_carry_monitors:
                    continue
                record = records[identifier]
                if record['kind'] != 'move':
                    continue
                if post_step_state is None:
                    post_step_state = adapter.state()
                ap = measured_aperture_m(post_step_state, adapter=adapter, arm=arm)
                local_step = cursor - record['start']
                slip = BatchScheduler.check_carry_slip_step(
                    active_stage_carry_monitors[identifier], identifier, local_step, ap
                )
                if slip is not None:
                    slip_triggers.append((identifier, arm, slip))
            if slip_triggers:
                slip_ids = {sid for sid, _, _ in slip_triggers}
                peer_finished = [
                    oid for oid, orec in records.items()
                    if orec['kind'] != 'observe'
                    and orec['end'] == cache['cursor']
                    and oid not in slip_ids
                    and oid not in cache['completed']
                ]
                ok_peers, peer_guards = _complete_finished_stages(
                    peer_finished, post_step_state=post_step_state, pause_on_guard=False
                )
                if not ok_peers:
                    return
                for sid, _, _ in slip_triggers:
                    active_stage_carry_monitors.pop(sid, None)
                    if records[sid]['end'] == cache['cursor'] and sid not in pending_final_stage_ids:
                        pending_final_stage_ids.append(sid)
                stop_conf = scheduler.hold()
                first_sid, _, first_slip = slip_triggers[0]
                BatchScheduler.record_carry_slip_pause(
                    cache,
                    receipt,
                    results,
                    first_sid,
                    first_slip,
                    stop_conf,
                    stages=action['stages'],
                    carry_monitors=active_carry_slip_monitors,
                )
                for extra_sid, _, extra_slip in slip_triggers[1:]:
                    BatchScheduler.record_carry_slip_pause(
                        cache,
                        {},
                        results,
                        extra_sid,
                        extra_slip,
                        stop_conf,
                        stages=action['stages'],
                        carry_monitors=active_carry_slip_monitors,
                    )
                    if extra_sid not in receipt['checkpoint_stage_ids']:
                        receipt['checkpoint_stage_ids'].append(extra_sid)
                    receipt['checkpoint_reasons'][extra_sid] = str(extra_slip['message'])
                for pid, p_eval in peer_guards:
                    results[pid]['status'] = 'paused_at_checkpoint'
                    results[pid]['checkpoint_reason'] = p_eval['reason']
                    results[pid]['checkpoint_reason_code'] = p_eval['reason']
                    if pid not in receipt['checkpoint_stage_ids']:
                        receipt['checkpoint_stage_ids'].append(pid)
                    receipt['checkpoint_reasons'][pid] = p_eval['message']
                _save_paused_cache_state(stop_conf)
                return
            for arm in arms_tuple:
                identifier = cache['contexts'][cursor][arm]['stage_id']
                if not identifier or records[identifier]['kind'] != 'move':
                    continue
                record = records[identifier]
                planned_positions = record.get('planned_positions_m')
                local_step = cursor - record['start']
                if not planned_positions or not (0 <= local_step < len(planned_positions)):
                    continue
                if post_step_state is None:
                    post_step_state = adapter.state()
                measured_pose = post_step_state['arms'][arm]['ee_pose']
                measured_pos = np.asarray(measured_pose['position_m'], dtype=float)
                planned_pos = np.asarray(planned_positions[local_step], dtype=float)
                step_div = float(np.linalg.norm(measured_pos - planned_pos))
                max_div = record.get('max_tracking_divergence_m')
                div_threshold = max_div if max_div is not None else record.get('position_tolerance_m', .010)
                if math.isfinite(step_div) and step_div > div_threshold and 'first_divergence' not in results[identifier]:
                    results[identifier]['first_divergence'] = {
                        'native_step': cursor + 1,
                        'stage_step': local_step + 1,
                        'divergence_m': round(step_div, 5),
                        'planned_position_m': [round(float(x), 5) for x in planned_pos],
                        'measured_position_m': [round(float(x), 5) for x in measured_pos],
                    }
                if max_div is not None and math.isfinite(step_div) and not record.get('contact_stopped_early'):
                    st = stall_state.setdefault(identifier, {'consecutive': 0, 'last_div': None, 'last_pos': None})
                    step_move = None if st['last_pos'] is None else float(np.linalg.norm(measured_pos - st['last_pos']))
                    st['last_pos'] = measured_pos.copy()
                    vel_thresh = float(thresholds.get('stall_guard_step_velocity_threshold_m', 0.0015))
                    consec_req = int(thresholds.get('stall_guard_consecutive_steps', 2))
                    prev_planned = np.asarray(planned_positions[local_step - 1], dtype=float) if local_step > 0 else None
                    planned_step_move = None if prev_planned is None else float(np.linalg.norm(planned_pos - prev_planned))
                    stalled = (step_move is None or step_move <= vel_thresh) and (planned_step_move is None or planned_step_move > vel_thresh)
                    if step_div > max_div and stalled and cursor + 1 < record['end'] - 1:
                        st['consecutive'] += 1
                        st['last_div'] = step_div
                    else:
                        st['consecutive'] = 0
                        st['last_div'] = step_div
                    if st['consecutive'] >= consec_req:
                        ref_target = record.get('original_target', record['target'])
                        pos_tol = float(record.get('position_tolerance_m', thresholds.get('default_position_tolerance_m', .010)))
                        ori_tol = float(record.get('orientation_tolerance_rad', thresholds.get('orientation_tolerance_rad', .15)))
                        ok_contact = False
                        if record.get('stall_guard_preset') == 'contact_stop':
                            explicit_dz = record.get('max_upward_dz_m')
                            explicit_down = record.get('max_downward_dz_m')
                            contact_max_dz = float(explicit_dz) if explicit_dz is not None else max(pos_tol, 0.040)
                            contact_down_dz = (
                                float(explicit_down)
                                if explicit_down is not None and float(explicit_down) > 0.0
                                else 0.0
                            )
                            ok_contact, _, _, _, _ = _check_pose_arrival(
                                measured_pose, ref_target, pos_tol, ori_tol,
                                allow_vertical_compliance=(contact_max_dz > 0.0) or (contact_down_dz > 0.0),
                                max_upward_dz_m=contact_max_dz if contact_max_dz > 0.0 else 0.0,
                                max_downward_dz_m=contact_down_dz,
                            )
                        if ok_contact:
                            record.setdefault('original_target', deepcopy(record['target']))
                            record['contact_stopped_early'] = True
                            held_q = [float(x) for x in post_step_state['arms'][arm]['q_rad']]
                            slot_arm = _layout(adapter).slot(arm)
                            clamp_stage_rows_to_held_q(record, rows, cache['cursor'], held_q, slot_arm.offset, dof=slot_arm.dof)
                            propagate_contact_hold_and_reanchor(
                                adapter, records, rows, results,
                                arm=arm, after_step=record['end'], held_q=held_q, held_pos=measured_pos,
                                source_stage_id=identifier, source_record=record, reanchor_next_move=True,
                                default_pos_tol=float(thresholds.get('default_position_tolerance_m', .010)),
                                layout=_layout(adapter),
                            )
                            _sync_execution_semantics()
                            results[identifier]['contact_stopped_early'] = True
                            results[identifier]['actual_contact_position_m'] = [round(float(x), 5) for x in measured_pos]
                            st['consecutive'] = 0
                        else:
                            position, orientation = pose_error(measured_pose, ref_target)
                            residual = compute_endpoint_residual(ref_target, measured_pose)
                            results[identifier]['endpoint_error'] = dict(position_error_m=position, orientation_error_rad=orientation)
                            results[identifier]['endpoint_residual_m'] = residual
                            results[identifier]['failed_phase'] = 'stall_guard_stopped'
                            receipt['endpoint_residual_m'] = residual
                            if 'first_divergence' in results[identifier]:
                                receipt['first_divergence'] = deepcopy(results[identifier]['first_divergence'])
                            summary = build_contact_block_summary(
                                identifier, arm, ref_target, measured_pose,
                                results[identifier].get('grasp_contact_feedback'),
                                completed_steps=results[identifier]['completed_steps'],
                                planned_steps=results[identifier]['planned_steps'],
                            )
                            results[identifier]['contact_block_summary'] = summary
                            receipt['contact_block_summary'] = summary
                            fail('tracking_error', identifier, failed_phase='stall_guard_stopped',
                                 tracking_divergence_m=round(step_div, 5), max_tracking_divergence_m=max_div)
                            return
            finished = [
                identifier for identifier, record in records.items()
                if record['kind'] != 'observe'
                and record['end'] == cache['cursor']
                and identifier not in cache['completed']
            ]
            ok_fin, _ = _complete_finished_stages(finished, post_step_state=post_step_state)
            if not ok_fin:
                return
    finally:
        summarize_batch(
            action['batch_plan_id'],
            cache,
            [s['id'] for s in action['stages']],
            results,
            receipt,
            complete_batch_native_steps=len(rows),
        )
