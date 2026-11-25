"""robot_action input schema and the field help delivered with it; dispatch and planning live in unified_actions.

Generated from the run's ``EmbodimentProfile`` (design 13 section 5.2): ``build(profile)`` returns the published tool,
the per-kind stage schemas the compiler admits fields against, and the field help, all from one declaration per
field. Every number in a help text is read from the profile (arm ids, TCP label, frame, gripper zero point and full
opening, grip-guard preset bounds, speed ranges and presets, tolerance and stall-guard presets, timeouts, stage
limits, overview camera, clock); a field whose ``supports`` / ``pose_inputs`` tag the profile lacks is absent from
the schema, the description and the field guide (decision Q3), and one generated sentence names those features.
The ``reach`` tag adds ``mode=reach`` (per-stage IK queries, ``unified_actions._reach``) and ``approach_roll`` the split
orientation bounds ``approach_tolerance_rad`` / ``roll_tolerance_rad``; the pose input ``jaw_grasp_centre`` adds the
grasp-centre move fields (``embodiments.JAW_GRASP_CENTRE_FIELDS``, bounds from ``gripper.jaw_grasp``). All three are
opt-in additions, so the sentence never names them and a profile without them (ARX-X5) renders the historical catalogue.

``ROBOT_ACTION_TOOL``, ``_STAGE_SCHEMAS``, ``_STAGE_COMMON``, ``_STAGE``, ``_STAGE_FIELD_HELP`` and
``_ACTION_FIELD_HELP`` remain importable: they are the rendering of the default robot
(``embodiments.ARX_X5_PROFILE``), built on first access. ``build`` is cached per profile; callers must treat the
result as read-only (``public_definitions`` publishes deep copies).

Three more opt-in tags (the SO-101; never named in the absent-features sentence): ``arrival_correction`` adds the
move-stage and batch fields ``arrival_correction_rounds`` / ``arrival_correction_max_offset_rad`` with the profile's
``arrival_correction_limits`` as their maxima and defaults; ``collision_world`` adds ``obstacles`` and ``held_object``
(boxes that enter the planner's collision world; with the ``obstacle_object_ids`` tag also ``obstacle_object_ids``,
whose help states the row size rules of ``scene_obstacles``); ``path_check`` names the preview's ``path_check`` review
among the reviews ``execute_if_clean`` requires clean (``embodiments.preview_reviews``).

A gripper with a bounded holding squeeze (``profile.gripper.hold_squeeze``; the SO-101 with deployment
``gripper.hold_squeeze_rad``) adds the jaw-stage field ``grip_squeeze_rad`` (rad of the jaw joint, 0..its ``max_rad``,
default its ``default_rad``); without it the field is absent everywhere, like the opt-in tags above.
"""

from copy import deepcopy
import functools
import math
from types import MappingProxyType, SimpleNamespace

from ...core.contracts import INSPECT_STEP_LIMIT

from ...robot.model.embodiments import (
    JAW_GRASP_CENTRE_FIELDS,
    ORIENTATION_FREE_TOLERANCE_RAD,
    PREVIEW_REVIEW_CLEAN,
    PUBLISHED_STALL_GUARD_PRESETS,
    ROBODOJO_REQUEST_DURATION_LIMIT_S,
    preview_reviews,
)
from ...robot.motion.batch_contract import IMAGE_DELIVERY_MODES
from ...robot.motion.motion_tolerance import MAX_DOWNWARD_DZ_M, MAX_UPWARD_DZ_M


_NUMBER = {'type': 'number'}
_POSE = {'type': 'array', 'items': _NUMBER, 'minItems': 3, 'maxItems': 3}
_BOX_POSE = {'type': 'array', 'items': _NUMBER, 'maxItems': 3}
_BOX = dict(type='object', properties={
    'min_m': _BOX_POSE, 'max_m': _BOX_POSE, 'margin_m': {'type':'number','minimum':0,'maximum':.2},
    'evidence': {'type':'string','minLength':1,'maxLength':1000}})
# The presets the schema offers, in published order; a profile offers those it defines (the validators also accept
# legacy aliases, which stay unpublished).
SCHEMA_GRIP_GUARD_PRESETS = ('non_empty', 'thin_object', 'rigid_object', 'wide_object')
SCHEMA_TOLERANCE_PRESETS = ('tight', 'normal', 'relaxed', 'compliant', 'seating', 'coarse', 'unconstrained_orientation')
SCHEMA_STALL_GUARD_PRESETS = PUBLISHED_STALL_GUARD_PRESETS
# Usage note of each tolerance preset (equal tolerances with equal notes are published as one group, e.g. relaxed/compliant).
_TOLERANCE_USE = {
    'tight': 'strictly for final millimeter-clearance pin/slot insertion',
    'relaxed': 'for compliant placement', 'compliant': 'for compliant placement', 'seating': 'for compliant placement',
    'coarse': 'for large-clearance transit/retract',
    'unconstrained_orientation': 'when wrist orientation around a symmetric target is free'}
# Features a profile may lack, in the words of the Q3 sentence (no field names: the fields are absent).
_ABSENT_FEATURES = (
    ('pose:grasp_angles', 'grasp-angle pose input (a grasp centre plus top-down angles)'),
    ('contact_sections', 'contact-section selection'),
    ('gripper_motion', 'jaw motion during a move'),
    ('grip_presets', 'named grip-guard presets (they need the measured empty-close reading of this gripper)'),
    ('joint_travel', 'joint-travel admission limits'),
    ('max_upward_dz', 'a vertical arrival window'),
    ('stall_guard', 'a tracking stall guard'),
    ('clearance', 'obstacle-crossing clearance arcs'),
    ('review_proxies', 'obstacle and held-object geometry review'),
    ('obstacle_object_ids', 'converting object-table rows into obstacle boxes'),
    ('settle', 'the bounded settle check'),
    ('execute_if_clean', 'execute-if-clean mode'),
)


# ---- number spelling ------------------------------------------------------------------------------------------
def _num(value):
    """A profile number as the schema spells it: an integral float as int (2, 300), anything else unchanged."""
    return int(value) if isinstance(value, float) and value.is_integer() else value


def _metres(value):
    """Metres with trailing zeros stripped (design 13 section 6): 0.0015 -> '0.0015', 0.088 -> '0.088', 0 -> '0'."""
    return f'{value:.6f}'.rstrip('0').rstrip('.') or '0'


def _metres3(value):
    """Metres with at least three decimals: 0 -> '0.000', 0.0045 -> '0.0045'."""
    whole, _, decimals = f'{value:.6f}'.rstrip('0').partition('.')
    return whole + '.' + decimals.ljust(3, '0')


def _mm(value):
    """Millimetres as %g of a metre value: 0.088 -> '88', 0.0015 -> '1.5'."""
    return f'{value * 1000:g}'


def _g(value):
    return f'{value:g}'


def _rad(value):
    return f'{value:.5f}' if value >= ORIENTATION_FREE_TOLERANCE_RAD else f'{value:.2f}'


def _series(items, conjunction):
    """'a, b, <conjunction> c' (serial comma); a single item alone."""
    items = list(items)
    return items[0] if len(items) == 1 else ', '.join(items[:-1]) + f', {conjunction} ' + items[-1]


def _or_list(items):
    items = list(items)
    return items[0] if len(items) == 1 else ', '.join(items[:-1]) + ' or ' + items[-1]


def _exclusive_fields(first, second):
    return {'not': {'required': [first, second]}}


def _axis(vector):
    """A unit TCP-frame axis as text: (0, 0, 1) -> '+z', (-1, 0, 0) -> '-x'; any other vector as its components."""
    for index, value in enumerate(vector):
        if abs(value) == 1 and all(other == 0 for position, other in enumerate(vector) if position != index):
            return ('+' if value > 0 else '-') + 'xyz'[index]
    return '[' + ','.join(_g(value) for value in vector) + ']'


# ---- profile facts used by the templates ------------------------------------------------------------------------
def _grip_presets(profile):
    """``[(preset, (min_m or None, max_m or None))]`` of the schema presets this gripper resolves, with the bounds the
    validator enforces (``grip_guard.preset_aperture_bounds``: zero point + margin, m). Empty when the empty-close
    reading is unmeasured (presets cannot resolve; the field is then absent)."""
    from ...robot.motion.grip_guard import preset_aperture_bounds
    gripper = profile.gripper
    if gripper.empty_close_aperture_m is None:
        return []
    return [(name, preset_aperture_bounds(name, profile=profile))
            for name in SCHEMA_GRIP_GUARD_PRESETS if name in gripper.preset_margins_m]


def _bounds_text(bounds, form):
    low, high = bounds
    if form == 'mm':
        parts = ([f'>={_mm(low)}mm'] if low is not None else []) + ([f'<={_mm(high)}mm'] if high is not None else [])
    else:
        parts = ([f'min {_metres(low)}m'] if low is not None else []) + ([f'max {_metres(high)}m'] if high is not None else [])
    return '[' + ', '.join(parts) + ']'


def _tolerance_groups(profile):
    """Schema tolerance presets of ``profile`` grouped: consecutive presets with equal numbers and equal usage notes."""
    groups = []
    for name in SCHEMA_TOLERANCE_PRESETS:
        if name not in profile.tolerance_presets:
            continue
        tolerance = profile.tolerance_presets[name]
        if groups and groups[-1][1] == tolerance and _TOLERANCE_USE.get(groups[-1][0][-1]) == _TOLERANCE_USE.get(name):
            groups[-1][0].append(name)
        else:
            groups.append(([name], tolerance))
    return groups


def _default_preset(profile):
    """The preset name that equals the embodiment default tolerance, or None (the default is its own value)."""
    name = profile.default_tolerance_preset
    return name if profile.tolerance_presets.get(name) == profile.default_tolerance else None


def _review_checks(profile):
    """The preview reviews this embodiment produces (execute_if_clean requires them clean;
    ``embodiments.preview_reviews``)."""
    return [check for check, _ in preview_reviews(profile)]


def _step_budget(profile):
    """Native steps one simulated-clock request may consume: the worker's cap (RoboDojo: 10 s x control rate), or
    None on a real-time clock (no native step budget)."""
    if profile.clock != 'simulated_steps':
        return None
    return profile.per_request_step_limit or math.ceil(ROBODOJO_REQUEST_DURATION_LIMIT_S * profile.control_hz)


def _absent(profile, presets):
    """Phrases of ``_ABSENT_FEATURES`` this profile lacks (the Q3 sentence; empty for ARX-X5)."""
    tags = set(profile.supports) | {'pose:' + name for name in profile.pose_inputs}
    # Contact sections select along the grasp-angle pose; without that pose form its phrase already covers them.
    if 'grasp_angles' not in profile.pose_inputs or profile.gripper.contact_sections is not None:
        tags.add('contact_sections')
    if presets:
        tags.add('grip_presets')
    absent = [(tag, text) for tag, text in _ABSENT_FEATURES if tag not in tags]
    # A profile with its own grasp-centre input (jaw_grasp_centre) lacks only the angle convention: "a grasp centre
    # ... not available" would be false there.
    own_centre = 'jaw_grasp_centre' in profile.pose_inputs
    return ['grasp-angle pose input (top-down angle triples)' if tag == 'pose:grasp_angles' and own_centre else text
            for tag, text in absent]


def _row_box_help():
    """Field help of ``obstacle_object_ids`` on a collision-world robot (SO-101 with its IK service): how a row becomes
    a box, with the size rules of ``scene_obstacles`` (ROW_*, metres) and where the preview reports them."""
    from ...robot.motion import scene_obstacles as rows
    return ('Preview/direct only: optional object_table row ids of the current observation (e.g. ["o1","o2"]); each '
            'row becomes an obstacles box with the row id as the box id: its centre, length and width turned by its '
            f'yaw, from its bottom to its top, margin_m {_g(rows.ROW_OBSTACLE_MARGIN_M)}. Size rules (m): a length or '
            f'width under {_g(rows.ROW_MIN_FOOTPRINT_M)} is raised to {_g(rows.ROW_MIN_FOOTPRINT_M)} (a missing one is '
            f'{_g(rows.ROW_DEFAULT_FOOTPRINT_M)}); each half extent of the footprint along x and y is at least '
            f'{_g(rows.ROW_MIN_HALF_EXTENT_M)}; the box is at least {_g(rows.ROW_MIN_HEIGHT_M)} tall (its bottom is '
            f'lowered); a missing top is centre z + {_g(rows.ROW_DEFAULT_TOP_ABOVE_CENTRE_M)}, a missing bottom the '
            f'table height (else top - {_g(rows.ROW_MIN_HEIGHT_M)}). Rows the table marks as held, ids the table lacks '
            'and rows without a centre add no box. The preview reports both in path_check.object_row_boxes: skipped '
            '[{id, reason: not_in_table | held_row | no_centre | no_object_table}] and size_floors [{id, field, '
            'row_value, used_value}] (m; row_value null when the row had none); absent when nothing was skipped or '
            'resized. Same use and limits as obstacles.')


# ---- builder ----------------------------------------------------------------------------------------------------
def build(profile=None):
    """The robot_action contract of ``profile`` (``EmbodimentProfile``; None = the default robot ARX-X5).

    Returns a namespace with ``tool`` (published definition without the generated field guide), ``stage_schemas``
    (``{kind: JSON schema}`` the compiler admits stage fields against), ``stage_common``, ``stage`` (flattened stage
    schema), ``stage_field_help`` and ``action_field_help``. Read-only: one cached build per profile."""
    if profile is None:
        from ...robot.model.embodiments import ARX_X5_PROFILE as profile
    return _build(profile)


@functools.lru_cache(maxsize=None)
def _build(profile):
    from ...robot.motion.motion_tolerance import MAX_UPWARD_DZ_M, SPLIT_ORIENTATION_FIELDS, arrival_correction_bounds
    arms = list(profile.arms)
    multi = len(arms) > 1
    tcp, frame, gripper = profile.tcp_label, profile.frame_id, profile.gripper
    supports, simulated = profile.supports, profile.clock == 'simulated_steps'
    grasp_angles = 'grasp_angles' in profile.pose_inputs
    sections = grasp_angles and gripper.contact_sections is not None
    # jaw_grasp_centre: grasp_centre_m + object_width_m of a fixed-jaw / moving-jaw gripper (bounds: gripper.jaw_grasp).
    jaw_grasp = gripper.jaw_grasp if 'jaw_grasp_centre' in profile.pose_inputs else None
    jaw_motion = 'gripper_motion' in supports
    vertical = 'max_upward_dz' in supports
    downward = vertical and profile.robot_model_id in ('arx-x5', 'libero-franka-panda')
    # reach: mode=reach (per-stage IK queries, no plan, no motion); approach_roll: the split orientation bounds.
    reach, split = 'reach' in supports, 'approach_roll' in supports
    # Who reads the batch-wide tolerance defaults and timeout_s (the X5 text is "Preview/direct only").
    batch_scope = 'Preview/direct' + ('/reach' if reach else '') + ' only'
    split_schema = {field: {'type': 'number', 'exclusiveMinimum': 0, 'maximum': math.pi}
                    for field in SPLIT_ORIENTATION_FIELDS} if split else {}
    # arrival_correction: bounds of the goal shift after a move settles short of its target (SO-101 sag correction).
    max_rounds, max_offset = arrival_correction_bounds(profile)
    known = profile.arrival_correction_limits is not None
    rounds_default, offset_default = (f'{max_rounds}, ', f'{_g(max_offset)}, ') if known else ('', '')
    correction_schema = {
        'arrival_correction_rounds': {'type': 'integer', 'minimum': 0, 'maximum': max_rounds},
        'arrival_correction_max_offset_rad': {'type': 'number', 'exclusiveMinimum': 0, 'maximum': _num(max_offset)}} \
        if 'arrival_correction' in supports else {}
    # collision_world: obstacle / held-object boxes that enter the planner's collision world (no proxy review).
    world = 'collision_world' in supports and 'review_proxies' not in supports
    # Move stages one reach query admits: the batch limits (validate_batch_action: total and per arm).
    reach_limit = min(profile.max_stages, profile.stages_per_arm * len(arms))
    presets = _grip_presets(profile)
    zero, full = gripper.empty_close_aperture_m, gripper.full_open_aperture_m
    speed, jaw_speed = profile.speed_scale, profile.gripper_speed_scale
    tolerance_names = [name for name in SCHEMA_TOLERANCE_PRESETS if name in profile.tolerance_presets]
    stall_names = [name for name in SCHEMA_STALL_GUARD_PRESETS if name in profile.stall_guard_presets_m]
    divergence_low, divergence_high = profile.tracking_divergence_range_m
    default_name = _default_preset(profile)
    default_tolerance = profile.default_tolerance

    stage_common = {
        'id': {'type': 'string', 'minLength': 1, 'maxLength': 80},
        'arm': {'type': 'string', 'enum': list(arms), 'description': (
            'Required for move/gripper. For observe, omit to wait for both arms\' prior stages; name an arm to pause both when that arm reaches this checkpoint.'
            if multi else f'Required for move/gripper: "{arms[0]}", the only arm. Optional on observe; the checkpoint pauses the arm either way.')},
        'depends_on': {'type': 'array', 'items': {'type': 'string'}, 'uniqueItems': True, 'description': (
            'Stage IDs that must finish before this stage starts. Same-arm order is automatic; add cross-arm dependencies only when physically needed. Independent left/right stages need none.'
            if multi else 'Stage IDs that must finish before this stage starts. Stages already run in list order on the one arm.')}}
    jaw_speed_schema = {
        'gripper_speed_scale': {'type': 'number', 'minimum': _num(jaw_speed.min), 'maximum': _num(jaw_speed.max)},
        'gripper_speed_preset': {'type': 'string', 'enum': list(profile.gripper_speed_presets)}}
    grip_guard_schema = {
        **({'grip_guard_preset': {'type': 'string', 'enum': ['none', *(name for name, _ in presets)]}} if presets else {}),
        'min_aperture_m': {'type': 'number', 'minimum': 0, 'maximum': _num(full)},
        'max_aperture_m': {'type': 'number', 'minimum': 0, 'maximum': _num(full)}}
    carry_slip_schema = {
        'carry_slip_guard': {'type': 'boolean'},
        'carry_slip_scope': {'type': 'string', 'enum': ['next_move', 'until_release']},
        'slip_consecutive_steps': {'type': 'integer', 'minimum': 1, 'maximum': 50}}
    # Bounded holding squeeze of a closing jaw (SO-101 with deployment gripper.hold_squeeze_rad): rad of the jaw joint.
    squeeze = gripper.hold_squeeze
    squeeze_schema = {'grip_squeeze_rad': {'type': 'number', 'minimum': 0, 'maximum': _num(squeeze.max_rad)}} \
        if squeeze is not None else {}
    joint_travel_schema = {
        'joint_travel_preset': {'type': 'string', 'enum': ['standard', 'extended', 'unconstrained']},
        'max_joint_travel_rad': {'type': 'number', 'exclusiveMinimum': 0},
        'max_backtracking_rad': {'type': 'number', 'exclusiveMinimum': 0}} if 'joint_travel' in supports else {}

    def stage_schema(kind, properties, required=(), **constraints):
        return dict(type='object', additionalProperties=False,
            required=['id', 'kind', *required],
            properties={field: deepcopy(value) for field, value in
                dict(stage_common, kind={'const': kind}, **properties).items()}, **constraints)

    move = {}
    if grasp_angles:
        from ...robot.model.roboprobe_pose import ANGLE_BOUNDS
        move.update({
            'grasp_position_m': _POSE,
            'angles_deg': dict(type='array', minItems=3, maxItems=3,
                prefixItems=[dict(type='number', minimum=lower, maximum=upper)
                    for lower, upper in ANGLE_BOUNDS.values()], items=_NUMBER)})
    move.update({'position_m': _POSE,
        'quaternion_xyzw': {'type': 'array', 'items': _NUMBER, 'minItems': 4, 'maxItems': 4}})
    if jaw_grasp is not None:
        move.update({
            'grasp_centre_m': _POSE,
            'object_width_m': {'type': 'number', 'minimum': 0, 'maximum': _num(jaw_grasp.max_gap_m)},
            'fixed_jaw_clearance_m': {'type': 'number', 'minimum': 0, 'maximum': _num(jaw_grasp.max_gap_m)},
            'grasp_depth_m': {'type': 'number', 'minimum': _num(jaw_grasp.min_grasp_depth_m),
                              'maximum': _num(jaw_grasp.max_grasp_depth_m)},
            'fixed_jaw_yaw_deg': {'type': 'number', 'minimum': -180, 'maximum': 180}})
    if sections:
        from ...robot.model.grasp_contact import MIN_CONTACT_DEPTH_M, MAX_CONTACT_DEPTH_M
        move.update({
            'contact_region': {'type': 'string', 'enum': list(gripper.contact_sections)},
            'contact_depth_m': {'type': 'number', 'minimum': MIN_CONTACT_DEPTH_M, 'maximum': MAX_CONTACT_DEPTH_M}})
    move.update({
        'speed_scale': {'type': 'number', 'minimum': _num(speed.min), 'maximum': _num(speed.max)},
        'speed_preset': {'type': 'string', 'enum': list(profile.speed_presets)},
        **jaw_speed_schema,
        'gripper_goal_normalized': {'type': 'number', 'minimum': 0, 'maximum': 1}})
    if jaw_motion:
        move['gripper_motion'] = dict(type='object', additionalProperties=False, required=['start_fraction', 'end_fraction'],
            properties={'start_fraction': {'type': 'number', 'minimum': 0, 'exclusiveMaximum': 1},
                'end_fraction': {'type': 'number', 'exclusiveMinimum': 0, 'maximum': 1}})
    move.update({
        'checkpoint_after_gripper': {'type': 'boolean'},
        **({'allow_jaw_symmetry': {'type': 'boolean'},
            'allow_approach_tilt': {'type': 'boolean'}} if grasp_angles else {}),
        **grip_guard_schema,
        **deepcopy(squeeze_schema),
        **joint_travel_schema,
        'tolerance_preset': {'type': 'string', 'enum': list(tolerance_names)},
        'position_tolerance_m': {'type': 'number', 'exclusiveMinimum': 0, 'maximum': _num(profile.max_position_tolerance_m)},
        'orientation_tolerance_rad': {'type': 'number', 'exclusiveMinimum': 0, 'maximum': _num(profile.max_orientation_tolerance_rad)},
        **deepcopy(split_schema)})
    if vertical:
        move['max_upward_dz_m'] = {'type': 'number', 'minimum': 0, 'maximum': MAX_UPWARD_DZ_M}
    if downward:
        move['max_downward_dz_m'] = {'type': 'number', 'minimum': 0, 'maximum': MAX_DOWNWARD_DZ_M}
    if 'stall_guard' in supports:
        move.update({
            'stall_guard_preset': {'type': 'string', 'enum': list(stall_names)},
            'max_tracking_divergence_m': {'type': 'number', 'minimum': _num(divergence_low), 'maximum': _num(divergence_high)}})
    move.update(deepcopy(correction_schema))
    if 'clearance' in supports:
        move.update({
            'clearance_preset': {'type': 'string', 'enum': ['none', 'low', 'normal', 'high']},
            'clearance_height_m': {'type': 'number', 'minimum': 0, 'maximum': 0.25}})
    move.update(deepcopy(carry_slip_schema))
    if grasp_angles:
        grasp_fields = ('grasp_position_m', 'angles_deg', *(('contact_region', 'contact_depth_m') if sections else ()))
        pose_rules = dict(required=['arm'], oneOf=[
            {'required': ['grasp_position_m', 'angles_deg'],
                'not': {'anyOf': [{'required': ['position_m']}, {'required': ['quaternion_xyzw']}]}},
            {'required': ['position_m', 'quaternion_xyzw'],
                'not': {'anyOf': [{'required': [field]} for field in grasp_fields]}}])
    elif jaw_grasp is not None:
        # One complete pose: the TCP pair, or the grasp centre with its width and exactly one orientation input
        # (three flat alternatives, so a schema error can name the complete field groups).
        pose_rules = dict(required=['arm'], oneOf=[
            {'required': ['position_m', 'quaternion_xyzw'],
                'not': {'anyOf': [{'required': [field]} for field in JAW_GRASP_CENTRE_FIELDS]}},
            {'required': ['grasp_centre_m', 'object_width_m', 'quaternion_xyzw'],
                'not': {'anyOf': [{'required': ['position_m']}, {'required': ['fixed_jaw_yaw_deg']}]}},
            {'required': ['grasp_centre_m', 'object_width_m', 'fixed_jaw_yaw_deg'],
                'not': {'anyOf': [{'required': ['position_m']}, {'required': ['quaternion_xyzw']}]}}])
    else:
        pose_rules = dict(required=['arm', 'position_m', 'quaternion_xyzw'])
    exclusive = ([_exclusive_fields('contact_region', 'contact_depth_m')] if sections else []) + [
        _exclusive_fields('speed_scale', 'speed_preset'),
        _exclusive_fields('gripper_speed_scale', 'gripper_speed_preset')] + (
        [_exclusive_fields('clearance_preset', 'clearance_height_m')] if 'clearance' in supports else [])
    # Both the public schema and compiler field admission use these declarations.
    # Runtime state, dependencies, capabilities and relational numeric checks stay
    # with the existing action/batch owners.
    stage_schemas = {
        'move': stage_schema('move', move, **pose_rules,
            dependentRequired={field: ['gripper_goal_normalized'] for field in
                (*(('gripper_motion',) if jaw_motion else ()), 'gripper_speed_scale', 'gripper_speed_preset',
                 'checkpoint_after_gripper', *squeeze_schema)},
            allOf=exclusive),
        'gripper': stage_schema('gripper', {
            'goal_normalized': {'type': 'number', 'minimum': 0, 'maximum': 1},
            'checkpoint_after_gripper': {'type': 'boolean'}, **grip_guard_schema, **deepcopy(squeeze_schema),
            **jaw_speed_schema, **deepcopy(carry_slip_schema)},
            required=['arm', 'goal_normalized'],
            allOf=[_exclusive_fields('gripper_speed_scale', 'gripper_speed_preset')]),
        'observe': stage_schema('observe', {
            'reason': {'type': 'string', 'minLength': 1, 'pattern': r'\S'}}, required=['reason'])}

    # ---- stage field help ----
    closed = f'closed {_metres(zero)}m' if zero is not None else 'empty-close reading not measured'
    thin_reading = f'{_metres3(zero)}m' if zero is not None else 'about the empty-close reading'
    preset_list = ', '.join(f'{name} {_bounds_text(bounds, "m")}' for name, bounds in presets)
    tolerance_entries = []
    for names, tolerance in _tolerance_groups(profile):
        notes = ['default'] if default_name in names else []
        upward_part = (
            f', max_upward_dz_m={tolerance.max_upward_dz_m:.3f} m'
            if vertical and tolerance.max_upward_dz_m is not None
            else ''
        )
        entry = '/'.join(names) + f'=({tolerance.position_m:.3f} m, {_rad(tolerance.orientation_rad)} rad{upward_part}'
        if 'normal' in names and vertical and simulated:
            # The X5 (RoboDojo) batch executor's implicit vertical windows for an omitted max_upward_dz_m; a real-time
            # profile (SO-101) defaults to no window and has no vertical compliance (its vertical_window_semantics).
            notes.append(f'requires <={_mm(tolerance.position_m)} mm horizontal alignment with vertical contact/load compliance')
        entry += (', ' + '; '.join(notes) if notes else '') + ')'
        use = _TOLERANCE_USE.get(names[0])
        if use:
            entry += ' ' + use
        if names[0] == 'tight' and simulated:
            # A measurement of the simulated PD arm (RoboDojo X5); a real-time embodiment publishes no droop figure.
            entry += ' (avoid tight on tabletop grasps or pushing where PD droop/friction is ~4-6mm)'
        tolerance_entries.append(entry)
    overrides = (
        'position_tolerance_m, orientation_tolerance_rad, max_upward_dz_m or max_downward_dz_m'
        if downward else
        'position_tolerance_m, orientation_tolerance_rad or max_upward_dz_m'
        if vertical else
        'position_tolerance_m or orientation_tolerance_rad'
    )
    default_text = ('' if default_name is not None else f' Without a preset or explicit tolerance a move uses '
                    f'({default_tolerance.position_m:.3f} m, {_rad(default_tolerance.orientation_rad)} rad).')
    position_default = (f'{default_name}={default_tolerance.position_m:.3f} m' if default_name is not None
                        else f'{default_tolerance.position_m:.3f} m')
    orientation_default = (f'{default_name}={default_tolerance.orientation_rad:.2f} rad' if default_name is not None
                           else f'{_rad(default_tolerance.orientation_rad)} rad')
    max_orientation = ('pi' if profile.max_orientation_tolerance_rad >= ORIENTATION_FREE_TOLERANCE_RAD
                       else _g(profile.max_orientation_tolerance_rad))
    stall_entries = {
        'off': 'off (default for batch, free-space moves, and pressing spring-loaded levers through resistance)',
        'contact_stop': f'contact_stop ({profile.stall_guard_presets_m.get("contact_stop") or 0:.3f} m stall trigger during '
                        f'motion; {profile.stall_guard_semantics})'}
    # Receipt fields of a tracking failure. first_divergence, contact_block_summary and in_hand_tracking_error come
    # from the batch executor that plans against collision geometry (the one that honours review proxies, X5
    # cuRobo); other embodiments report the measured TCP residual in the stage result.
    failure_fields = ('endpoint_residual_m={dx_m,dy_m,dz_m,position_error_m,orientation_error_rad}, first_divergence, '
                      'contact_block_summary, in_hand_tracking_error (true when an object remains held between the fingers at a blocked pose) and next_action'
                      if 'review_proxies' in supports else
                      'the measured TCP residual (position_error_m, orientation_error_rad) in the stage result and next_action'
                      if 'arrival_correction' not in supports else  # the SO-101 worker mirrors the residual to receipt level
                      'the measured TCP residual in the stage result (dx_m, dy_m, dz_m, position_error_m, orientation_error_rad), '
                      'the same residual at receipt level as endpoint_residual_m, contact_block_summary, in_hand_tracking_error '
                      '(true when the jaw was commanded closed and still measures wider than an empty close, i.e. something '
                      'is between the fingers at the failed pose) and next_action')
    stage_field_help = {
        'id': 'Unique ID within this batch; referenced by depends_on and returned stage results.',
        'grasp_position_m': f'Move only: target center between fingers at the selected contact section [x,y,z] in {frame} metres, paired with angles_deg. This is not either individual finger surface. Tool offset is converted once; do not add a manual offset.',
        'angles_deg': 'Move only: [pitch,roll,yaw] extrinsic world +x,+y,+z rotations from the top-down reference, in degrees; [0,0,0] points tool +x down world -z, jaw opening along world -x. Bounds [-180,180], [-90,90], [-180,180]; not absolute Euler angles. On open-jaw grasp moves (allow_jaw_symmetry=true by default), if the primary orientation hits IK/joint-travel limits, planning tries strictly the exact 180 deg parallel-jaw SO(3) twin (reporting resolved_angles_deg); approach tilts require explicit allow_approach_tilt=true.',
        'position_m': f'Move only: target {tcp} [x,y,z] in {frame} metres, paired with quaternion_xyzw'
                      + ('; mutually exclusive with grasp pose/contact fields.' if grasp_angles else
                         '; mutually exclusive with grasp_centre_m and its fields.' if jaw_grasp is not None else '.'),
        'quaternion_xyzw': f'Move only: {tcp} orientation [x,y,z,w], a unit quaternion in {frame}; paired with position_m'
                           + (' or with grasp_centre_m.' if jaw_grasp is not None else '.'),
        'contact_region': 'Grasp move only: pad_center (default, fingertips 7.5 mm beyond target along tool +x), finger_middle (27.5 mm toward palm, fingertips 35 mm beyond target), or fingertip (at target). Selects a longitudinal section, not a contacting finger or proof of object contact; keep fingertips above rigid supports on top-down grasps. Choose this or contact_depth_m.',
        'contact_depth_m': 'Grasp move only: custom depth from pad_center in metres [-0.0075,0.0625], positive toward palm; choose this or contact_region.',
        'speed_scale': f'Move only: arm trajectory speed multiplier [{_g(speed.min)},{_g(speed.max)}]; choose this or speed_preset. Omission uses run default.',
        'speed_preset': 'Move only: named arm speed; ' + ', '.join(f'{name}={_g(value)}' for name, value in profile.speed_presets.items()) + '. Choose this or speed_scale.',
        'gripper_speed_scale': f'Jaw speed multiplier [{_g(jaw_speed.min)},{_g(jaw_speed.max)}], default {_g(jaw_speed.default)}; choose this or gripper_speed_preset. On move requires gripper_goal_normalized.',
        'gripper_speed_preset': 'Jaw speed: ' + _or_list(f'{name}={_g(value)}' for name, value in profile.gripper_speed_presets.items()) + '; choose this or gripper_speed_scale. On move requires gripper_goal_normalized.',
        'gripper_goal_normalized': 'Move only: jaw target [0,1], 0 closed and 1 open; omission preserves the last commanded goal, not the measured opening or a guaranteed grip force. Runs after measured arrival'
                                   + (' unless gripper_motion is supplied' if jaw_motion else '') + '. A tracking failure stops the batch before this after-arrival command.',
        'gripper_motion': 'Move only: synchronize jaw motion over start_fraction..end_fraction of this move, with start < end; requires gripper_goal_normalized.',
        'checkpoint_after_gripper': f'Move (with gripper_goal_normalized) or gripper stage only: optional on-demand boolean (default inherits batch checkpoint_after_gripper, which defaults to false). Set true to unconditionally pause {"both arms" if multi else "the arm"} at the stage boundary before dependent move stages so you can inspect RGB and grip_facts and resume the cached plan_id.',
        'allow_jaw_symmetry': 'Move only: optional boolean (defaults to true on open-jaw grasp moves when omitted; inherits batch allow_jaw_symmetry when set). When true on an open-jaw move (jaw goal >= 0.90 normalized [0, 1]), if the requested angles_deg (deg) fails cuRobo IK or exceeds joint-travel/wrist-winding thresholds (rad), planning tries strictly the exact 180 deg SO(3) parallel-jaw symmetric twin around the approach axis (0 m contact shift, 0 deg approach tilt), reporting jaw_symmetry_resolved and resolved_angles_deg (deg) in stage_costs and stage_results. Set false for strict finger polarity; never flips or tilts closed/holding moves.',
        'allow_approach_tilt': f'Move only: optional opt-in boolean (default false; inherits batch allow_approach_tilt when set). When explicitly true on an open-jaw grasp move (jaw goal >= 0.90 normalized [0, 1]) and both primary angles_deg (deg) and its exact 180 deg parallel-jaw twin fail cuRobo IK or joint-travel admission (rad), planning additionally searches contact-point-invariant approach tilts (-15, +15, -30, +30 deg around the jaw closing axis while keeping grasp_position_m [x,y,z] in {frame} metres fixed and flange Z >= contact Z - 0.01 m), reporting jaw_symmetry_resolved, approach_tilt_resolved, and resolved_angles_deg (deg). Keep default false on vertical slot/peg/cavity grasps where approach tilt would violate clearance.',
        'grip_guard_preset': f'Move or gripper stage: optional aperture guard preset (none, {preset_list}). Choose from object thickness/rigidity: use rigid_object/wide_object (or min_aperture_m) on rigid blocks/bottles/tubes so an empty closure pauses automatically while a valid grasp continues; use thin_object or none on deformable cloth/thin sheets where a pinched jaw closes nearly to {_metres(zero) if zero is not None else "the empty-close reading"} m.',
        'min_aperture_m': f'Move or gripper stage: optional minimum expected fingertip aperture in metres [0, {_metres(full)}] (full open {_mm(full)}mm, {closed}; default none/disabled). Pauses with aperture_below_min_m if measured_aperture_m < min_aperture_m'
                          + ('; overrides preset lower bound' if presets else '') + f'. Note that thin parts (<=3mm such as coins, cards, thin rims or cloth) can read {thin_reading} when firmly held, so omit min_aperture_m for thin parts.',
        'max_aperture_m': f'Move or gripper stage: optional maximum expected fingertip aperture in metres [0, {_metres(full)}] (full open {_mm(full)}mm, {closed}). Pauses with aperture_above_max_m if measured_aperture_m > max_aperture_m'
                          + ('; overrides preset upper bound.' if presets else '.'),
        'carry_slip_guard': (
            f'Move (with gripper_goal_normalized) or gripper stage: optional boolean (default false; inherits batch carry_slip_guard when set). '
            f'When true on a closing jaw stage (goal < 0.5) that resolves a non-null min_aperture_m (in metres) and passes its initial close check, '
            f'monitors measured fingertip aperture (metres) on every {"native step" if simulated else "control step"} of subsequent same-arm move stages '
            f'and pauses immediately with checkpoint_reason_code="carry_slip_aperture_below_min_m" and slip_detected_at if measured_aperture_m < min_aperture_m '
            f'for slip_consecutive_steps consecutive {"native steps" if simulated else "control steps"}. '
            f'Example: {{"id":"close","kind":"gripper","arm":"{arms[0]}","goal_normalized":0,'
            + ('"grip_guard_preset":"rigid_object",' if presets else '"min_aperture_m":0.01,')
            + '"carry_slip_guard":true,"carry_slip_scope":"until_release"}.'
        ),
        'carry_slip_scope': (
            'Move (with gripper_goal_normalized) or gripper stage: optional coverage scope for carry_slip_guard=true, enum '
            '"next_move" (default, monitors only the immediately following same-arm move stage) or '
            '"until_release" (monitors all subsequent same-arm move stages across lift/transport/align until an opening jaw command with goal >= 0.5). '
            'Requires carry_slip_guard=true.'
        ),
        'slip_consecutive_steps': (
            f'Move (with gripper_goal_normalized) or gripper stage: optional integer in [1, 50] (unit: {"native steps" if simulated else "control steps"}, default 1) '
            'giving the consecutive below-min_aperture_m samples required before carry_slip_guard pauses execution. '
            'Set 2..3 to debounce transient jaw sensor jitter. Requires carry_slip_guard=true.'
        ),
        'joint_travel_preset': 'Move only: optional joint-travel admission preset: standard (default, total<=pi or excess<pi/2), extended (total<=1.5*pi or excess<pi for wide-arc sweeps, large orientation flips or deformable folding), or unconstrained (disables travel/backtracking rejection while keeping step-budget checks).',
        'max_joint_travel_rad': 'Move only: optional custom per-joint total travel threshold in radians (>0) before backtracking/winding checks apply; overrides joint_travel_preset.',
        'max_backtracking_rad': 'Move only: optional custom per-joint backtracking/periodic-winding excess threshold in radians (>0); overrides joint_travel_preset.',
        'goal_normalized': 'Gripper stage only, required: jaw target [0,1], 0 closed and 1 open. Command completion and measured aperture do not establish object retention. Use this exact field, not gripper, goal, or gripper_goal_normalized.',
        'tolerance_preset': 'Move only: named arrival tolerance preset; ' + ', '.join(tolerance_entries)
                            + f'. Explicit {overrides} overrides the corresponding component.' + default_text,
        'position_tolerance_m': f'Move only: measured {profile.tcp_link} arrival position tolerance in metres, positive and at most {profile.max_position_tolerance_m:.2f}. Overrides the position component of tolerance_preset (default {position_default}). Tracking failure stops later stages, including gripper commands, and returns {failure_fields}.',
        'orientation_tolerance_rad': f'Move only: measured {profile.tcp_link} arrival orientation tolerance in radians, in (0, {max_orientation}]. Overrides the orientation component of tolerance_preset (default {orientation_default}).',
        'max_upward_dz_m': profile.vertical_window_semantics,
        'max_downward_dz_m': 'Move only: optional downward -Z static payload-droop compliance window in metres [0, 0.15] when horizontal (dx, dy) and orientation are within tolerance and measured Z sags below target under held object weight (-max_downward_dz_m <= dz_m <= 0; default 0 m). Set e.g. max_downward_dz_m=0.020-0.060 m on loaded lift/transport/hover moves when held payload weight causes static joint-PD gravity sag while horizontal (dx, dy) stays aligned.',
        'stall_guard_preset': 'Move only: real-time tracking divergence early-stop preset; '
                              + _series([stall_entries[name] if name in stall_entries
                                         else f'{name} ({profile.stall_guard_presets_m[name]:.3f} m)'
                                         for name in stall_names], 'or') + '.',
        'max_tracking_divergence_m': f'Move only: custom real-time tracking divergence threshold in metres [{_g(divergence_low)}, {_g(divergence_high)}]; overrides stall_guard_preset.',
        'clearance_preset': 'Move only: optional obstacle-crossing vertical clearance preset (none=0 m, low=0.09 m, normal=0.13 m default, high=0.18 m) when obstacles are supplied; mutually exclusive with clearance_height_m.',
        'clearance_height_m': 'Move only: optional custom obstacle-crossing vertical clearance height in metres [0, 0.25] above obstacle top when obstacles are supplied; mutually exclusive with clearance_preset.',
        'reason': 'Observe stage only, required nonblank text: what evidence to inspect at this checkpoint.'}
    if jaw_grasp is not None:
        approach_axis = _axis(gripper.approach_axis_local)
        fixed_axis = _axis(tuple(-value for value in gripper.jaw_opening_axis_local))
        stage_field_help.update({
            'grasp_centre_m': (
                f'Move only: [x,y,z] in {frame} metres where the centre of the object sits between the jaws; use it '
                f'instead of position_m and the {tcp} target is computed for you (do not add a jaw offset yourself). '
                f'Requires object_width_m and one orientation (quaternion_xyzw or fixed_jaw_yaw_deg). This gripper has '
                f'one fixed and one moving jaw: the fixed fingertip face is placed object_width_m/2 + '
                f'fixed_jaw_clearance_m from the centre on the {tcp} {fixed_axis} side, the same at every jaw opening, and '
                f'only the moving jaw closes onto the opposite face. Conversion: {tcp} position = grasp_centre_m + '
                'R(orientation) * [object_width_m/2 + fixed_jaw_clearance_m'
                + (f' - {_metres(jaw_grasp.fixed_face_x_m)}' if jaw_grasp.fixed_face_x_m else '')
                + f', 0, grasp_depth_m] in {tcp} axes. Preview stage_plans[<stage>].grasp_centre and reach '
                'stage_results[<stage>].grasp_centre report the converted target, the offset and the jaw goals that '
                'straddle and close on the width. Also usable to carry a held object: give the place its centre should '
                'reach, with the same width and depth it was grasped at. Example: {"id":"descend","kind":"move",'
                f'"arm":"{arms[0]}","grasp_centre_m":[0.24,-0.06,0.015],"object_width_m":0.03,"fixed_jaw_clearance_m":0.004,'
                '"fixed_jaw_yaw_deg":0}.'),
            'object_width_m': (
                f'Move with grasp_centre_m, required: extent of the object along the jaw line ({tcp} x, the direction '
                f'fixed_jaw_yaw_deg or the quaternion gives it) in metres, [0, {_metres(jaw_grasp.max_gap_m)}] together '
                'with fixed_jaw_clearance_m (the widest gap between the fingertips along that line); no default. It '
                'sets how far the fixed jaw stands from the centre, so take it along the jaw line you chose, not the '
                'object\'s longest side.'),
            'fixed_jaw_clearance_m': (
                'Move with grasp_centre_m: gap in metres kept between the fixed fingertip face and the object at '
                f'arrival, [0, {_metres(jaw_grasp.max_gap_m)}], default 0 (the face touches the object, so a centre or '
                'width error of a few millimetres lands the fixed jaw on top of it). With a clearance the closing '
                'moving jaw slides the object that far toward the fixed jaw; the jaw must arrive open by more than '
                'object_width_m + fixed_jaw_clearance_m.'),
            'grasp_depth_m': (
                f'Move with grasp_centre_m: how far the grasp centre lies behind the {tcp} origin plane, against the '
                f'approach axis ({tcp} {approach_axis}), in metres [{_metres(jaw_grasp.min_grasp_depth_m)}, '
                f'{_metres(jaw_grasp.max_grasp_depth_m)}] (fixed fingertip end .. pocket floor), default 0 (centre on '
                f'the {tcp} origin plane, fingertips {_mm(gripper.fingertip_beyond_reference_m)} mm beyond it). Positive '
                'takes the object deeper between the fingers, so the fingertips reach that much further past its '
                f'centre. The moving fingertip swings back as the jaw opens: an object wider than '
                f'{_metres(jaw_grasp.tcp_plane_gap_m)} m is met by it behind the origin plane, so give such an object a '
                'positive depth.'),
            'fixed_jaw_yaw_deg': (
                f'Move with grasp_centre_m, instead of quaternion_xyzw: straight-down approach ({tcp} {approach_axis} '
                f'along {frame} -z) with the fixed jaw on the side of this yaw, in degrees [-180, 180] about {frame} +z '
                f'from +x (0: fixed jaw toward +x and moving jaw toward -x of the centre; 90: fixed jaw toward +y). '
                'Equals quaternion_xyzw [cos(yaw/2), sin(yaw/2), 0, 0]. yaw and yaw+180 close along the same line with '
                'the jaws swapped, and the arm may reach only one of them; reach answers both in one call. For a '
                'tilted approach give quaternion_xyzw.')})
    if correction_schema:
        after = ('Move only: after the move the worker waits for the joints to settle, then shifts the joint goal past '
                 'the planned target by the settled shortfall (gravity sag of the servos) and settles again, before the '
                 'arrival verdict. ')
        stage_field_help.update({
            'arrival_correction_rounds': (
                after + f'arrival_correction_rounds is the largest number of such shifts (integer count, 0..{max_rounds}; '
                f'default when omitted {rounds_default}this robot\'s limit, effective_limits.arrival_correction_rounds; a larger '
                'value is refused, never clamped). 0 = no correction: the goal stays the planned target and the '
                'verdict judges where the arm settles; use it when the arm is to rest on a support above the target '
                '(placing a held object, pressing), where each shift pushes harder; in free space the arm then keeps the '
                'servo sag, so the tolerance must allow for it. Set on a move stage or batch-wide (the stage value wins). '
                'The stage result reports effective_tolerances.arrival_correction_rounds and arrival_correction.rounds '
                '(the shifts applied). Example: {"kind":"move",...,"arrival_correction_rounds":0,"max_upward_dz_m":0.01}.'),
            'arrival_correction_max_offset_rad': (
                after + 'arrival_correction_max_offset_rad is the largest total shift of any joint goal from the planned '
                f'target in radians, in (0, {_g(max_offset)}] (default when omitted {offset_default}this robot\'s limit, '
                'effective_limits.arrival_correction_max_offset_rad; a larger value is refused, never clamped). A '
                'smaller value bounds how hard the arm pushes on something that blocks it; typical free-space sag needs '
                '0.02..0.06 rad. Reported as effective_tolerances.arrival_correction_max_offset_rad; '
                'arrival_correction.offset_limit_reached tells when the bound was hit.')})
    if squeeze is not None:
        stage_field_help['grip_squeeze_rad'] = (
            'Move (with gripper_goal_normalized) or gripper stage: optional holding squeeze in radians of the jaw joint, '
            f'[0, {_g(squeeze.max_rad)}], default {_g(squeeze.default_rad)} (this robot). When a closing jaw settles short '
            'of its goal (stopped by an object or the other fingertip), the held jaw goal becomes the measured jaw angle '
            'minus this squeeze toward closed, so the holding torque grows with the squeeze instead of staying at the '
            'servo limit (which overheats the gripper in a long hold). Larger = firmer hold for heavy or slippery objects '
            '(values inside the ~0.03..0.04 rad unloaded static-friction deadband at P=16 give weak hold; ~0.08 rad holds '
            'a 29 mm rigid cube firmly), smaller = gentler for fragile ones; 0 holds the contact angle. No effect on an '
            'opening command or a jaw that reaches its goal. The stage result reports gripper_execution.hold_squeeze. '
            'Example: {"kind":"gripper","goal_normalized":0,"grip_squeeze_rad":' + _g(squeeze.default_rad) + '}.')
    if split:
        approach = f'{tcp} local {_axis(gripper.approach_axis_local)}'
        per_component = ('Default unset: orientation is then judged as one angle against orientation_tolerance_rad. '
                         'When approach_tolerance_rad or roll_tolerance_rad is set, orientation is judged per component '
                         'instead, an unset component uses the stage\'s resolved orientation_tolerance_rad, and the '
                         'IK and arrival residuals in receipts also report approach_error_rad and roll_error_rad (rad).')
        stage_field_help.update({
            'approach_tolerance_rad': (
                f'Move only: optional maximum angle in radians between the achieved and the requested approach axis '
                f'({approach}, outward along the jaws), in (0, pi]; a value >= 3.14159 leaves the approach direction '
                f'free. {per_component} Example: top-down grasp of a cube far from the base with quaternion_xyzw '
                'pointing the approach axis straight down at the jaw yaw you want, approach_tolerance_rad=0.35 '
                '(accepts up to ~20 deg tilt) and roll_tolerance_rad=0.1.'),
            'roll_tolerance_rad': (
                f'Move only: optional maximum rotation in radians about the approach axis ({approach}) left after '
                'aligning the achieved approach axis with the requested one, i.e. the jaw-direction error seen along the '
                'approach, in (0, pi]; a value >= 3.14159 leaves the jaw direction free (e.g. a rotationally symmetric '
                f'object). {per_component} See approach_tolerance_rad for an example.')})
    for kind, schema in stage_schemas.items():
        for field, text in stage_field_help.items():
            if field in schema['properties']:
                schema['properties'][field]['description'] = text
        schema['properties']['kind']['description'] = 'Stage type: ' + kind + '.'
    if jaw_motion:
        motion = stage_schemas['move']['properties']['gripper_motion']['properties']
        motion['start_fraction']['description'] = 'Start fraction of move in [0,1), strictly before end_fraction.'
        motion['end_fraction']['description'] = 'End fraction of move in (0,1], strictly after start_fraction.'
    # Discoverable fields are flat; the same conditional branches still enforce kind-specific admission.
    enum_omit = frozenset({
        'tolerance_preset', 'grip_guard_preset', 'stall_guard_preset', 'joint_travel_preset',
        'clearance_preset', 'speed_preset', 'gripper_speed_preset', 'contact_region', 'carry_slip_scope',
    })
    min_items_omit = frozenset({'angles_deg', 'grasp_position_m', 'position_m', 'quaternion_xyzw', 'grasp_centre_m'})
    stage_flat_props = {
        field: {k: deepcopy(v) for k, v in value.items()
                if not (k == 'enum' and field in enum_omit)
                and not (k == 'minItems' and field in min_items_omit)
                and k not in {'minLength', 'required', 'additionalProperties'}}
        for schema in stage_schemas.values()
        for field, value in schema['properties'].items()
    }
    carry_flat_props = {k: stage_flat_props.pop(k) for k in carry_slip_schema if k in stage_flat_props}
    stage = dict(type='object', required=['kind'],
        properties={**stage_flat_props,
            'kind': {'type': 'string', 'enum': list(stage_schemas),
                'description': ('move requires arm and one complete pose pair' if grasp_angles
                                else 'move requires arm and one complete pose (position_m + quaternion_xyzw, or '
                                     'grasp_centre_m + object_width_m with quaternion_xyzw or fixed_jaw_yaw_deg)'
                                if jaw_grasp is not None
                                else 'move requires arm, position_m and quaternion_xyzw')
                               + '; gripper requires arm and goal_normalized; observe requires reason. joint/home/reset are not supported kinds.'},
            **carry_flat_props},
        allOf=[{'if': {'properties': {'kind': {'const': kind}}, 'required': ['kind']},
            'then': schema} for kind, schema in stage_schemas.items()])

    # ---- tool description ----
    checks = _review_checks(profile)
    preview_effect = 'without physics' if simulated else 'without moving the robot'
    closed_jaw = ('; closed_jaw_before_grasp:<arm> is set when a closing jaw stage (goal <= 0.2) comes before any '
                  'opening stage (goal >= 0.5) while the jaw already reads closed' if zero is not None else '')
    if checks and simulated:
        clean_run = 'plans the complete batch, checks ' + ', '.join(checks[:-1]) + (' and ' if len(checks) > 1 else '') + checks[-1] + ', '
        clean_when = 'and immediately executes in the SAME call when blocking_risks is empty'
        clean_else = ' (returning the preview plan_id without moving if blocking_risks is non-empty; advisory_notes do not block). '
    elif checks:
        # A real-time arm with a review (SO-101 grasp_3d_review on grasp-centre stages): say which stages it covers and
        # what else gates execution (unified_actions._preview_downgrade_reasons).
        named = [f'{check} ({scope})' if scope else check for check, scope in preview_reviews(profile)]
        clean_run = ('plans the complete batch, runs ' + ', '.join(named[:-1]) + (' and ' if len(named) > 1 else '')
                     + named[-1] + ', ')
        clean_when = ('and immediately executes in the SAME call when all checks pass with zero warnings/risks, the '
                      'preview is planned (every move stage met its tolerances) and there are no downgrade_reasons'
                      + closed_jaw)
        clean_else = ' (otherwise it returns the preview plan_id with downgrade_reasons, without moving). '
    elif simulated:
        clean_run = 'plans the complete batch '
        clean_when = 'and immediately executes in the SAME call when blocking_risks is empty'
        clean_else = ' (returning the preview plan_id without moving if blocking_risks is non-empty; advisory_notes do not block). '
    else:
        # No review runs on this embodiment (unified_actions._preview_downgrade_reasons): planning status and, with a
        # measured empty-close gap, the closed-jaw check are what gate execution.
        clean_run = 'plans the complete batch '
        clean_when = ('and immediately executes in the SAME call when the preview is planned (every move stage met its '
                      'tolerances) with no downgrade_reasons; this robot runs no motion-safety, grasp or geometry review'
                      + closed_jaw)
        clean_else = ' (otherwise it returns the preview plan_id with downgrade_reasons, without moving). '
    # A simulated clock replays the exact cached native-step rows; a real-time arm (SO-101) replays cached IK joint
    # targets, re-interpolated from the joints measured at execution.
    preview_returns = ('returns plan_id, cost, exact trajectory and calibrated robot-geometry evidence. Review it, then '
                       'call this SAME tool with mode=execute and plan_id to execute those exact cached rows. No silent '
                       'replanning or changed commands. ' if simulated else
                       'returns plan_id, cost, the per-stage IK joint targets with their residuals and robot-geometry '
                       'evidence. Review it, then call this SAME tool with mode=execute and plan_id to run those cached '
                       'joint targets: IK is not re-solved, the joint path is re-interpolated from the joints measured at '
                       'execution (refused if the arm drifted from the preview start) and bounded arrival corrections are '
                       'reported. No silent replanning. ')
    preset_list_mm = ', '.join(f'{name} {_bounds_text(bounds, "mm")}' for name, bounds in presets)
    guard_clause = f'or use grip_guard_preset ({preset_list_mm}, none) and ' if presets else 'or use '
    absent = _absent(profile, presets)
    tolerance_list = ', '.join(names[0] for names, _ in _tolerance_groups(profile))
    if 'stall_guard' not in supports:
        stall_clause = ' per stage. '
    elif vertical:
        stall_clause = (' per stage, and set stall_guard_preset="contact_stop" specifically on contact descent/press stages '
                        '(keep batch default "off" for free-space transits). ')
    else:
        stall_clause = (f' per stage; stall_guard_preset="contact_stop" {profile.stall_guard_semantics} '
                        '(keep batch default "off" for free-space transits). ')
    budget = _step_budget(profile)
    description = (
        'Unified bounded robot action. Default mode=preview plans the COMPLETE '
        + ('single/dual-arm ' if multi else '') + f'stage batch {preview_effect}, '
        + preview_returns
        + ('mode=execute_if_clean (or auto_execute_if_clean=true) ' + clean_run + clean_when + clean_else
           if 'execute_if_clean' in supports else '')
        + f'mode=preflight plans {preview_effect}, caches plan_id for mode=execute, and returns numerical costs/tolerances/advisory_notes with 0 images. '
        + 'mode=direct explicitly previews and immediately executes without returning for model review. '
        + (f'mode=reach takes the current observation_id and 1..{reach_limit} move stages (id, kind, arm, '
           'position_m, quaternion_xyzw' + (', or the grasp_centre_m pose fields,' if jaw_grasp is not None else '')
           + ' and optional tolerance fields only) and solves IK for each candidate pose '
           'independently from the joints measured at the call: no plan, no cache, no motion, and every stage is answered '
           '(no stop at the first unreachable pose); use it to compare candidate grasp/place poses before preview. It '
           'returns stage_count, reachable_stage_count and stage_results[id] with ok (joints within the limits meet the '
           'stage\'s effective tolerances), ik (position_error_m in m, orientation_error_rad in rad; when not ok the '
           'closest residuals and, if a position-first solve reached the position, position_priority with the quaternion '
           'reachable there), tolerances, reached (position_m, quaternion_xyzw, approach_axis) and, when ok, q_rad and '
           'joint_travel_rad (rad); a failure of the whole query (e.g. ik_service_unreachable) is a tool error naming '
           'its reason. ' if reach else '')
        + 'Optional plan_update submits or updates task_plan inline in the same call without a separate task_plan round trip. '
        + ('stages are ordered per arm; IDs are unique and depends_on adds cross-arm completion dependencies. ' if multi
           else 'stages run in list order on the one arm; IDs are unique. ')
        + (f'move uses grasp_position_m in {frame} metres and angles_deg=[pitch,roll,yaw] world rotations from the top-down '
           'reference, not conventional absolute Euler angles. '
           + ('Choose contact_region/contact_depth_m at each stage. ' if sections else '')
           + f'Alternatively move uses position_m + quaternion_xyzw for a measured {tcp} target. Supply exactly one complete pose pair. '
           if grasp_angles else f'move uses position_m + quaternion_xyzw for a measured {tcp} target in {frame} metres. '
           + (f'To grasp or carry an object, move may instead give grasp_centre_m (where the object centre sits between '
              f'the jaws, {frame} metres) + object_width_m (its extent along the jaw line, metres) with quaternion_xyzw '
              f'or fixed_jaw_yaw_deg (degrees, straight-down approach); the {tcp} target is computed from the gripper '
              'geometry and reported in stage_plans[<stage>].grasp_centre. ' if jaw_grasp is not None else ''))
        + 'Routes use successive move stages, not via fields. There are no joint/home/reset kinds; a return uses saved measured poses. '
        'Every move has its own arm speed and optional gripper_goal_normalized/speed; '
        + ('gripper_motion fractions synchronize that jaw within THIS move. Without fractions its jaw runs after measured arrival. '
           if jaw_motion else 'its jaw runs after measured arrival. ')
        + 'A gripper stage requires goal_normalized (0 closed, 1 open), not gripper or goal. '
        + ('observe requires reason and stops BOTH arms; omit arm for a barrier after both prior arm stages, or name an arm '
           'to checkpoint when only that arm arrives. ' if multi else 'observe requires reason and pauses the arm at that checkpoint. ')
        + 'Later stages remain cached. After inspecting RGB, execute the same '
        'plan_id to continue only if its current-state validity check passes; otherwise preview a new plan. '
        'Optional checkpoint_after_gripper=true (on a grasp/gripper stage or the batch) pauses after a jaw stage '
        'before the next move stage (pre-lift) to inspect finger contact and grip_facts in RGB before continuing '
        'the cached plan_id; ' + guard_clause
        + f'min_aperture_m/max_aperture_m [0..{_metres(full)}m] to conditionally pause only when measured_aperture_m falls outside '
        f'the expected bounds (valid grasps continue into lift without pausing). Use tolerance_preset ({tolerance_list}) '
        'or position_tolerance_m/orientation_tolerance_rad'
        + (' (approach_tolerance_rad/roll_tolerance_rad bound the approach-axis tilt and the roll about it separately)'
           if split else '') + stall_clause
        + ('Use joint_travel_preset (standard, extended, unconstrained) or '
           'max_joint_travel_rad/max_backtracking_rad on move stages or the batch when legitimate large wrist rotations need wider travel thresholds. '
           if 'joint_travel' in supports else '')
        + ('On open-jaw grasp moves (jaw >= 0.90), allow_jaw_symmetry=true is enabled by default (set false for strict finger polarity) so exact 180-deg parallel-jaw symmetry resolves wrist-wrap or IK failures automatically (reporting resolved_angles_deg); set allow_approach_tilt=true explicitly only when -15/+15/-30/+30 deg approach tilts are permitted; closed/holding moves never flip or tilt. '
           if grasp_angles else '')
        + ('Independent arm sequences overlap. Express paired barriers using depends_on; arrival/closed-jaw commands do not '
           'prove contact, support or retention. ' if multi else 'Arrival/closed-jaw commands do not prove contact, support or retention. ')
        + (f'Shared rows cost one native step; the complete batch retains one {budget}-step '
           'budget including resumed portions. ' if budget is not None else '')
        + 'mode=inspect accepts plan_id for cached evidence, action_id for executed '
        'measured/planned paths, or trajectory_ref for an existing artifact; path_kind selects measured/planned. '
        + ('Supply observed/estimated obstacles world AABBs '
           'and held_objects={' + '/'.join(arms) + ': current world AABB} for '
           'full-timeline proxy clearance estimates; contact declarations annotate overlaps without suppressing risks. '
           if 'review_proxies' in supports else '')
        + (('Supply obstacles (boxes you estimate)' + (', obstacle_object_ids (object-table rows)' if 'obstacle_object_ids' in supports else '')
            + ' and held_object (the box of what the jaw holds) to add them to the collision world: planned joint targets '
            'prefer solutions free of them, and the path check reports the first collision of each move; both are advisory '
            'facts about boxes you supplied, not proof of clearance. ') if world else '')
        + 'Auxiliary review is reserved; the main model owns execution approval. '
        + ('Geometry risks are advisory, not a '
           'joint-collision, environment-collision or shared-object constraint proof. ' if 'review_proxies' in supports else '')
        + f'Usage: preview/preflight{"/execute_if_clean" if "execute_if_clean" in supports else ""}/direct require current observation_id and either 1..{profile.max_stages} stages, source_plan_id + start_from_stage_id (with optional stage_overrides to recompile remaining stages from current measured state), or plan_id + skip_failed_stage/partial stages (from repair_action); execute accepts plan_id, optional action_id, optional image_delivery and expected_cursor/expected_revision; '
        'inspect requires exactly one plan_id/action_id/trajectory_ref plus inspection options. '
        'Example checkpoint preview: {"observation_id":"obs1","stages":[{"id":"check","kind":"observe","reason":"inspect current support"}]}. '
        'After reviewing a returned plan: {"mode":"execute","plan_id":"p1","expected_cursor":0,"expected_revision":0}. '
        'Explicit continuation after a paused/failed stage: {"mode":"preview","observation_id":"obs2","source_plan_id":"p1","start_from_stage_id":"carry","stage_overrides":{"carry":{"position_tolerance_m":0.015}}} or {"mode":"preview","observation_id":"obs2","plan_id":"p1","skip_failed_stage":true}. '
        + (f'Not available on this robot, so absent from this schema: {_series(absent, "and") if len(absent) > 2 else " and ".join(absent)}. '
           if absent else '')
        + 'Gripper stage syntax: {"id":"open","kind":"gripper","arm":"' + arms[0] + '","goal_normalized":1}; choose the goal from the intended action.')

    # ---- batch-level fields ----
    modes = [
        'preview',
        'preflight',
        'execute',
        'inspect',
        'direct',
        *(['execute_if_clean'] if 'execute_if_clean' in supports else []),
        *(['reach'] if reach else []),
    ]
    proxies = 'review_proxies' in supports
    # The per-episode obstacle registry (obstacle_policy, ignore_obstacle_ids) exists only with review proxies.
    stored_obstacles = proxies or ('obstacle_object_ids' in supports and not world)
    properties = {
        'mode': {'type': 'string', 'enum': modes},
        'image_delivery': {'type': 'string', 'minLength': 1},
        **({'auto_execute_if_clean': {'type': 'boolean'}} if 'execute_if_clean' in supports else {}),
        'plan_update': {'type': 'object'},
        'observation_id': {'type': 'string'}, 'action_id': {'type': 'string'}, 'plan_id': {'type': 'string'},
        'source_plan_id': {'type': 'string', 'minLength': 1},
        'start_from_stage_id': {'type': 'string', 'minLength': 1},
        'stage_overrides': {'type': 'object'},
        'skip_failed_stage': {'type': 'boolean'},
        'expected_cursor': {'type':'integer','minimum':0},
        'expected_revision': {'type':'integer','minimum':0},
        'trajectory_ref': {'type':'string'}, 'path_kind': {'type':'string'},
        'stages': {'type': 'array', 'items': stage, 'minItems': 1, 'maxItems': profile.max_stages,
            'description': (f'1..{profile.max_stages} stages total, at most {profile.stages_per_arm} per arm, on one shared execution timeline. Include both arms here for simultaneous motion. List order applies within each arm, not across arms; a global observe creates a barrier between preceding and following stages. A move with an after-arrival jaw goal expands into a motion and a jaw stage and consumes two stage slots.'
                            if multi else f'1..{profile.max_stages} stages total, at most {profile.stages_per_arm} per arm, run in list order. A move with an after-arrival jaw goal expands into a motion and a jaw stage and consumes two stage slots.')
                           + (' mode=reach accepts move stages only, each with only id, kind, arm, position_m, '
                              'quaternion_xyzw' + (' (or the grasp_centre_m pose fields)' if jaw_grasp is not None else '')
                              + ' and optional tolerance fields.' if reach else '')},
        'checkpoint_after_gripper': {'type': 'boolean'},
        **({'allow_jaw_symmetry': {'type': 'boolean'},
            'allow_approach_tilt': {'type': 'boolean'}} if grasp_angles else {}),
        **({'grip_guard_preset': {'type': 'string', 'minLength': 1}} if presets else {}),
        'min_aperture_m': {'type': 'number', 'minimum': 0, 'maximum': _num(full)},
        'max_aperture_m': {'type': 'number', 'minimum': 0, 'maximum': _num(full)},
        **({'joint_travel_preset': {'type': 'string', 'minLength': 1},
            'max_joint_travel_rad': {'type': 'number', 'exclusiveMinimum': 0},
            'max_backtracking_rad': {'type': 'number', 'exclusiveMinimum': 0}} if 'joint_travel' in supports else {}),
        'tolerance_preset': {'type': 'string', 'minLength': 1},
        'position_tolerance_m': {'type': 'number', 'exclusiveMinimum': 0, 'maximum': _num(profile.max_position_tolerance_m)},
        'orientation_tolerance_rad': {'type': 'number', 'exclusiveMinimum': 0, 'maximum': _num(profile.max_orientation_tolerance_rad)},
        **deepcopy(split_schema),
        **({'max_upward_dz_m': {'type': 'number', 'minimum': 0, 'maximum': MAX_UPWARD_DZ_M}} if vertical else {}),
        **({'max_downward_dz_m': {'type': 'number', 'minimum': 0, 'maximum': MAX_DOWNWARD_DZ_M}} if downward else {}),
        **({'stall_guard_preset': {'type': 'string', 'minLength': 1},
            'max_tracking_divergence_m': {'type': 'number', 'minimum': _num(divergence_low), 'maximum': _num(divergence_high)}}
           if 'stall_guard' in supports else {}),
        **deepcopy(correction_schema),
        'timeout_s': {'type': 'number', 'minimum': .1, 'maximum': _num(profile.max_timeout_s)},
        **({'settle': {'type': 'object'}} if 'settle' in supports else {}),
        **({'obstacles': {'type':'array','maxItems':40,'items': dict(_BOX,
            properties=dict(_BOX['properties'],id={'type':'string','minLength':1,'maxLength':160}))}} if proxies or world else {}),
        **({'obstacle_object_ids': {'type': 'array', 'maxItems': 40, 'items': {'type': 'string', 'minLength': 1, 'maxLength': 80}}}
           if 'obstacle_object_ids' in supports else {}),
        **({'obstacle_policy': {'type': 'string', 'minLength': 1},
            'ignore_obstacle_ids': {'type': 'array', 'maxItems': 40, 'items': {'type': 'string', 'minLength': 1, 'maxLength': 160}}}
           if stored_obstacles else {}),
        **({'target_exemption_preset': {'type': 'string', 'minLength': 1},
            'target_exemption_radius_m': {'type': 'number', 'minimum': 0, 'maximum': 0.15}} if proxies else {}),
        **({'clearance_preset': {'type': 'string', 'minLength': 1},
            'clearance_height_m': {'type': 'number', 'minimum': 0, 'maximum': 0.25}} if 'clearance' in supports else {}),
        **({'held_objects': dict(type='object',properties={arm: deepcopy(_BOX) for arm in arms}),
            'held_object': dict(_BOX,properties=dict(_BOX['properties'],arm={'type':'string','minLength':1})),
            'contact_entity_id': {'type':'string'},
            'initial_contact': {'type': 'object'},
            'placement_contact': {'type': 'object'}} if proxies else {}),
        **({'held_object': deepcopy(_BOX)} if world else {}),
        'camera_id': {'type': 'string'}, 'detail': {'type': 'string', 'minLength': 1},
        'native_steps': {'type': 'array', 'items': {'type': 'integer'}, 'maxItems': INSPECT_STEP_LIMIT},
        'start_native_step': {'type': 'integer', 'minimum': 0},
        'max_samples': {'type': 'integer', 'minimum': 1, 'maximum': INSPECT_STEP_LIMIT},
        'carry_slip_guard': {'type': 'boolean'},
        'carry_slip_scope': {'type': 'string', 'minLength': 1},
        'slip_consecutive_steps': {'type': 'integer', 'minimum': 1, 'maximum': 50}}

    overview = profile.overview_camera
    overview_label = overview if overview is not None else 'the overview camera'
    wrist_label = ('both close-range wrist RGB' if multi else
                   (f'{next(iter(profile.wrist_cameras.values()))} wrist RGB' if profile.wrist_cameras else 'wrist RGB'))
    step = 'native-step' if simulated else 'control-step'  # the unit of cursors and trajectory sample ids
    if simulated:
        mode_clean = 'execute_if_clean plans and executes in one call when blocking_risks is empty'
        auto_when = 'status is planned and blocking_risks is empty (advisory_notes do not block)'
    elif checks:
        mode_clean = ('execute_if_clean plans and immediately executes when ' + ', '.join(checks[:-1])
                      + (' and ' if len(checks) > 1 else '') + checks[-1]
                      + (' have' if len(checks) > 1 else ' has') + ' zero warnings/risks')
        listed = ['status is planned', *(PREVIEW_REVIEW_CLEAN[check] for check in checks)]
        auto_when = ', '.join(listed[:-1]) + ', and ' + listed[-1]
    else:
        mode_clean = ('execute_if_clean plans and immediately executes when the preview is planned with no '
                      'downgrade_reasons (no review runs on this robot)')
        auto_when = 'status is planned and downgrade_reasons is empty'
    action_field_help = {
        'mode': f'preview (default) plans {preview_effect} and renders the trajectory overlay; '
                + (mode_clean + ' (otherwise returns preview plan_id for inspection); ' if 'execute_if_clean' in supports else '')
                + f'preflight plans {preview_effect} with 0 images and caches plan_id; '
                + 'execute runs cached plan_id; direct plans and immediately executes; inspect reads saved trajectories'
                + ('; reach solves IK of each move stage pose independently from the joints measured at the call, '
                   'with no plan, no cache and no motion' if reach else '') + '.',
        'image_delivery': (
            'Optional visual image policy ("none", "on_anomaly", "focus_roi", "head_only", "all"): '
            f'omit on physical actions to use run default ("all": {overview_label} + {wrist_label} + trajectory overlay in one reply). '
            f'"head_only" drops wrist views; "focus_roi" attaches cropped ROI (px); '
            '"on_anomaly" attaches images only on stop/pause, slip_detected=true, or unavailable 3D; "none" suppresses images.'
        ),
        'auto_execute_if_clean': f'Preview/preflight/execute_if_clean only: optional boolean (default false). When true, automatically executes the planned batch in the same call if {auto_when}; otherwise returns the preview plan_id for inspection.',
        'plan_update': 'Optional inline task_plan payload {plan?, attention?, ledger?}: atomically submits (if no task_plan exists yet) or updates task_plan in the same call without a separate task_plan round trip.',
        'observation_id': (
            ('Required for preview/preflight/execute_if_clean/direct: actual current observation ID grounding the complete batch.'
             if 'execute_if_clean' in supports else
             'Required for preview/preflight/direct: actual current observation ID grounding the complete batch.')
            if simulated else
            ('Required for preview/preflight' + ('/execute_if_clean' if 'execute_if_clean' in supports else '')
             + '/direct' + ('/reach' if reach else '')
             + ': actual current observation ID grounding the complete batch.')
        ),
        'action_id': 'Optional caller-chosen unique execution ID for preview/preflight/direct/execute; inspect selects this existing executed action. Never reuse for different commands.',
        'plan_id': 'Required for execute; inspect may select it. Use the ID returned by preview/preflight in this active run; only 16 recent plans remain cached.',
        'source_plan_id': 'Preview/direct only: existing plan_id whose stage definitions are reused for explicit continuation from start_from_stage_id. Mutually exclusive with stages.',
        'start_from_stage_id': 'Preview/direct with source_plan_id: stage ID in source_plan_id to start recompiling from (inclusive of this stage and all subsequent stages) using the current observation_id measured state.',
        'stage_overrides': 'Preview/direct with source_plan_id: optional object {stage_id: {field: value}} overriding target_pose/pose fields, position_tolerance_m, speed_preset, etc. on selected remaining stages without rewriting all stages.',
        'skip_failed_stage': 'Preview/direct with plan_id only: optional boolean (default false). Set true to skip the failed stage and recompile remaining uncompleted stages from current measured state.',
        'expected_cursor': f'Execute only: expected executed {step} count; use next_action returned by preview/checkpoint. Omission uses current cached cursor.',
        'expected_revision': 'Execute only: expected plan revision; use returned next_action. Omission uses current cached revision.',
        'trajectory_ref': 'Inspect only: existing actor-accessible trajectory artifact reference; mutually exclusive with plan_id/action_id.',
        'path_kind': 'Inspect action feedback only: measured (default) or planned, if listed in available_paths. Omit for preview trajectory artifacts.',
        'checkpoint_after_gripper': 'Preview/direct only: optional boolean (default false). Set true to unconditionally pause at stage boundaries after jaw stages before dependent move stages unless overridden on a stage.',
        'allow_jaw_symmetry': 'Preview/preflight/execute_if_clean/direct: optional batch-default boolean (open-jaw grasp moves default to true when omitted; set false for strict finger polarity). When true on open-jaw move stages (jaw goal >= 0.90), tries strictly the exact 180 deg SO(3) parallel-jaw symmetric twin if primary angles_deg (deg) fails cuRobo IK or joint-travel (rad) admission; overridden by stage allow_jaw_symmetry.',
        'allow_approach_tilt': f'Preview/preflight/execute_if_clean/direct: optional batch-default opt-in boolean (default false). When explicitly true on open-jaw grasp moves (jaw goal >= 0.90), allows fallback search over contact-invariant approach tilts (-15/+15/-30/+30 deg around the jaw closing axis at fixed grasp_position_m in {frame} metres) if both primary angles_deg and the exact 180 deg twin are infeasible; overridden by stage allow_approach_tilt.',
        'grip_guard_preset': f'Preview/direct only: optional batch-default aperture guard preset (none, {preset_list}) applied to closing jaw stages unless overridden.',
        'min_aperture_m': f'Preview/direct only: optional batch-default minimum expected fingertip aperture in metres [0, {_metres(full)}]; overridden by stage-level min_aperture_m.',
        'max_aperture_m': f'Preview/direct only: optional batch-default maximum expected fingertip aperture in metres [0, {_metres(full)}]; overridden by stage-level max_aperture_m.',
        'carry_slip_guard': (
            f'Preview/direct only: optional batch-default boolean (default false) enabling per-step fingertip aperture slip monitoring '
            f'(against min_aperture_m in metres) after passing closing jaw stages (goal < 0.5); overridden by stage-level carry_slip_guard. '
            'Example: {'
            + ('"grip_guard_preset":"rigid_object"' if presets else '"min_aperture_m":0.01')
            + ',"carry_slip_guard":true,"carry_slip_scope":"until_release"}.'
        ),
        'carry_slip_scope': (
            'Preview/direct only: optional batch-default scope for carry_slip_guard=true, enum "next_move" (default) or '
            '"until_release" (monitors subsequent same-arm moves until jaw goal >= 0.5); requires carry_slip_guard=true and overridden by stage-level carry_slip_scope.'
        ),
        'slip_consecutive_steps': (
            f'Preview/direct only: optional batch-default integer in [1, 50] (unit: {"native steps" if simulated else "control steps"}, default 1) '
            'of consecutive below-min_aperture_m samples required before pausing; requires carry_slip_guard=true and overridden by stage-level slip_consecutive_steps.'
        ),
        'joint_travel_preset': 'Preview/direct only: optional batch-default joint-travel admission preset: standard (default), extended (1.5*pi total / pi excess), or unconstrained.',
        'max_joint_travel_rad': 'Preview/direct only: optional batch-default per-joint total travel threshold in radians (>0); overridden by stage-level settings.',
        'max_backtracking_rad': 'Preview/direct only: optional batch-default per-joint backtracking/winding excess threshold in radians (>0); overridden by stage-level settings.',
        'tolerance_preset': f'{batch_scope}: batch-wide default move arrival tolerance preset ({", ".join(tolerance_names)}); overridden by stage-level tolerance settings.',
        'position_tolerance_m': f'{batch_scope}: batch-wide default move arrival position tolerance in metres (0, {profile.max_position_tolerance_m:.2f}]; overrides batch tolerance_preset.',
        'orientation_tolerance_rad': f'{batch_scope}: batch-wide default move arrival orientation tolerance in radians (0, {max_orientation}]; overrides batch tolerance_preset.',
        'approach_tolerance_rad': f'{batch_scope}: optional batch-wide default approach-axis tolerance in radians, in (0, pi] (>= 3.14159 = free), for every move stage that sets none; default unset (orientation judged as one angle). Same meaning as the stage field.',
        'roll_tolerance_rad': f'{batch_scope}: optional batch-wide default roll tolerance about the approach axis in radians, in (0, pi] (>= 3.14159 = free), for every move stage that sets none; default unset (orientation judged as one angle). Same meaning as the stage field.',
        'max_upward_dz_m': f'{batch_scope}: optional batch-wide default vertical Z compliance window in metres [0, 0.15]; overridden by stage-level max_upward_dz_m.',
        'max_downward_dz_m': 'Preview/direct only: optional batch-wide default downward -Z payload-droop compliance window in metres [0, 0.15] when horizontal (dx, dy) and orientation are within tolerance; overridden by stage-level max_downward_dz_m.',
        'stall_guard_preset': f'Preview/direct only: batch-wide default move stall guard preset ({", ".join(stall_names)}); overridden by stage-level stall guard settings.',
        'max_tracking_divergence_m': f'Preview/direct only: optional batch-wide default real-time tracking divergence threshold in metres [{_g(divergence_low)}, {_g(divergence_high)}]; overrides batch stall_guard_preset.',
        'arrival_correction_rounds': f'Preview/direct only: optional batch-wide default of the move-stage field arrival_correction_rounds (integer count, 0..{max_rounds}; 0 = no goal shift after settling) for every move stage that sets none; omitted = {rounds_default}this robot\'s limit.',
        'arrival_correction_max_offset_rad': f'Preview/direct only: optional batch-wide default of the move-stage field arrival_correction_max_offset_rad in radians, in (0, {_g(max_offset)}], for every move stage that sets none; omitted = {offset_default}this robot\'s limit.',
        'timeout_s': f'{batch_scope}: action timeout in seconds, default {_g(profile.default_action_timeout_s)}, range 0.1..{_g(profile.max_timeout_s)}. In programs, omission is capped to remaining work time minus the 5-second action reserve and 1-second dispatch margin before preview. Explicit values and cached plans are unchanged; final admission still checks remaining time. Read-only preview remains available without execution reserve.'
                     + (' On reach it bounds the whole IK query (s); there omission uses the default without the program cap.'
                        if reach else ''),
        'settle': 'Preview/direct only: optional bounded RGB stability check after the complete batch, never at a checkpoint; adds separate '
                  + ('native wait steps' if simulated else f'robot_wait steps ({_g(1 / profile.control_hz)} s each, real time)')
                  + '. Requires robot_wait and a visible ROI.',
        'obstacles': (
            'Preview/direct only: up to 40 observed/estimated world AABBs, each with ID and evidence, for advisory whole-timeline clearance checks.'
            if not world else
            f'Preview/direct only: up to 40 axis-aligned boxes in {frame} (metres) you estimate from permitted evidence, each '
            'with id, min_m, max_m, evidence and optional margin_m; default none. They apply to this request only (nothing '
            'is remembered). Each box, grown by its margin_m, enters the collision world of this preview: a planned joint '
            'target prefers a solution clear of it (stage_plans[<move>].ik.obstacle_free=false when none is), and the '
            'path check names it in stage_plans[<move>].path.collision when the arm or the held object overlaps it on '
            'the way. Advisory: nothing is refused or re-routed. Leave out the object a stage is about to grasp or push '
            '(the fingers must reach it), or expect it named at that stage.'
        ),
        'obstacle_object_ids': (
            'Preview/preflight/execute_if_clean/direct: optional list of non-target intermediate 3D object IDs from the current observation_id (e.g. ["o1", "o2"]) converted into observation-scoped solid world AABB obstacles (m) to route over/around. Never include the target object or hollow racks, hooks, slots, or open containers being entered.'
            if not world else _row_box_help()
        ),
        'obstacle_policy': 'Preview/direct only: obstacle persistence policy across actions in the episode: persist (default, retains registered obstacles), stateless (applies only to this call without reading/writing persistent obstacles), or clear (clears previously persisted obstacles first).',
        'ignore_obstacle_ids': 'Preview/direct only: optional list of obstacle IDs to exclude from obstacle checks in this batch.',
        'target_exemption_preset': 'Preview/direct only: optional XY target-neighborhood obstacle exemption preset (none=0 m, tight=0.030 m, normal=0.055 m default, wide=0.080 m); mutually exclusive with target_exemption_radius_m.',
        'target_exemption_radius_m': 'Preview/direct only: optional custom XY target-neighborhood obstacle exemption radius in metres [0, 0.15]; mutually exclusive with target_exemption_preset.',
        'clearance_preset': 'Preview/direct only: optional batch-default obstacle-crossing vertical clearance preset (none=0 m, low=0.09 m, normal=0.13 m default, high=0.18 m); overridden by stage-level settings.',
        'clearance_height_m': 'Preview/direct only: optional batch-default obstacle-crossing vertical clearance height in metres [0, 0.25] above obstacle top; overridden by stage-level settings.',
        'held_objects': f'Preview/direct only: {"/".join(arms)} current held-object world AABBs with evidence. Choose this or held_object.',
        'held_object': 'Preview/direct only: single held-object AABB; arm required for a dual-arm batch, otherwise inferred. Choose this or held_objects.'
                       if not world else
                       f'Preview/direct only: the box (min_m, max_m in {frame} metres as it is now, evidence, optional margin_m) '
                       'of the object the jaw currently holds; default none (a held object is then not modelled). The box is '
                       'fixed to the measured gripper pose and moves with it: it is checked against the table, the obstacles '
                       'and the arm (not the gripper) at every planned target and along every move, up to and including the '
                       'first stage of this batch that opens the jaw; later stages are checked without it, and the released '
                       'object is not added as an obstacle. The box is filled with up to 32 equal spheres inside it '
                       '(path_check.held_object reports their count and radius), so its corners are slightly under-covered. '
                       'Example: {"min_m":[0.20,-0.015,0.0],"max_m":[0.23,0.015,0.03],"evidence":"o2 row of obs12, 3 cm cube"}.',
        'contact_entity_id': 'Preview/direct only: ID of the intended contact entity among supplied obstacles; annotates overlap without hiding risks.',
        'initial_contact': 'Preview/direct only: obstacle entity IDs and evidence explaining pre-existing contact.',
        'placement_contact': 'Preview/direct only: intended support entity ID and evidence, with optional permitted penetration estimate.',
        'camera_id': (f'Overlay camera ID, default {overview}, for preview/direct or inspect.' if overview is not None
                      else 'Overlay camera ID, default the first camera of the observation, for preview/direct or inspect.')
                     + ' Must exist in the referenced observation.',
        'detail': 'Inspect: summary (default) or full including selected geometry. Accepted but has no effect on preview/direct.',
        'native_steps': f'Inspect only: 1..{INSPECT_STEP_LIMIT} distinct existing {step} IDs; mutually exclusive with pagination. Default (no selector) selects only the first, middle and last.',
        'start_native_step': f'Inspect pagination: existing {step} ID to start from, default 0. Follow returned next_native_step when present.',
        'max_samples': f'Inspect pagination: 1..{INSPECT_STEP_LIMIT} samples per page, default {INSPECT_STEP_LIMIT}. Mutually exclusive with native_steps.'}
    for field, text in action_field_help.items():
        if field in properties:
            properties[field]['description'] = text
    all_of = [
        {'if': {'required': ['image_delivery']}, 'then': {'properties': {
            'image_delivery': {'type': 'string', 'enum': list(IMAGE_DELIVERY_MODES),
                               'description': action_field_help['image_delivery']}}}},
        {'if': {'required': ['detail']}, 'then': {'properties': {
            'detail': {'type': 'string', 'enum': ['summary', 'full'],
                       'description': action_field_help['detail']}}}},
        {'if': {'required': ['native_steps']}, 'then': {'properties': {
            'native_steps': {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 1, 'maxItems': INSPECT_STEP_LIMIT,
                             'description': action_field_help['native_steps']}}}},
        {'if': {'required': ['carry_slip_scope']}, 'then': {'properties': {
            'carry_slip_scope': {'type': 'string', 'enum': ['next_move', 'until_release'],
                                 'description': action_field_help['carry_slip_scope']}}}},
    ]
    if proxies or world:
        for box in (properties['obstacles']['items'],
                    *(properties['held_objects']['properties'].values() if proxies else ()), properties['held_object']):
            box['properties'] = {field: deepcopy(value) for field, value in box['properties'].items()}
            for field, text in {'min_m': 'World AABB lower [x,y,z] corner in metres.',
                    'max_m': 'World AABB upper [x,y,z] corner in metres; each coordinate must exceed min_m. Both corners must lie in [-3,3] metres.',
                    'margin_m': 'Nonnegative clearance inflation in metres, default 0.01, at most 0.2.',
                    'evidence': 'Required nonblank description of current permitted evidence and estimation uncertainty.',
                    'id': ('Unique obstacle entity ID referenced by contact declarations.' if proxies else
                           'Unique box id within this request (1..160 characters); path_check.obstacle_ids and '
                           'stage_plans[<move>].path.collision.world[].obstacles name the box by it.'),
                    'arm': 'Holding arm; required when a single held_object is used with both arms.'}.items():
                if field in box['properties']:
                    box['properties'][field]['description'] = text
    if proxies:
        for arm in arms:
            properties['held_objects']['properties'][arm]['description'] = 'Current held-object world AABB for this arm, derived from permitted evidence.'
        initial_contact_props = {
            'entity_ids': {'type': 'array', 'items': {'type': 'string'},
                           'description': 'IDs of existing obstacles in pre-existing contact.'},
            'evidence': {'type': 'string', 'minLength': 1,
                         'description': 'Required nonblank observed basis for the contact declaration.'},
        }
        placement_contact_props = {
            'entity_id': {'type': 'string', 'description': 'ID of the existing obstacle supporting placement.'},
            'evidence': {'type': 'string', 'minLength': 1,
                         'description': 'Required nonblank observed basis for the contact declaration.'},
            'max_penetration_m': {'type': 'number', 'minimum': 0, 'maximum': .003,
                                  'description': 'Maximum declared placement penetration in metres, default 0.002, range 0..0.003.'},
        }
        all_of.extend([
            {'if': {'required': ['held_object']}, 'then': {'properties': {
                'held_object': {'type': 'object', 'description': action_field_help['held_object'], 'properties': {
                    'arm': {'type': 'string', 'enum': list(arms),
                            'description': properties['held_object']['properties']['arm']['description']}}}}}},
            {'if': {'required': ['initial_contact']}, 'then': {'properties': {
                'initial_contact': dict(type='object', description=action_field_help['initial_contact'],
                                        properties=initial_contact_props)}}},
            {'if': {'required': ['placement_contact']}, 'then': {'properties': {
                'placement_contact': dict(type='object', description=action_field_help['placement_contact'],
                                          properties=placement_contact_props)}}},
        ])
    if 'settle' in properties:
        settle_props = {
            'camera_id': {'type': 'string', 'description': f'RGB camera ID, default {overview}.' if overview is not None else 'RGB camera ID, required.'},
            'roi': {'type': 'array', 'items': {'type':'integer'}, 'maxItems': 4, 'description': 'Target pixel [x,y,width,height]; nonnegative origin, width/height at least 4, inside the image.'},
            'max_native_steps': {'type': 'integer', 'minimum': 1, 'maximum': 50, 'description': 'Maximum additional wait steps, default 25, range 1..50.'},
            'sample_steps': {'type': 'integer', 'minimum': 1, 'maximum': 10, 'description': 'Wait steps per RGB sample, default 5, range 1..10.'},
            'stable_samples': {'type': 'integer', 'minimum': 2, 'maximum': 5, 'description': 'Required consecutive quiet comparisons, default 3, range 2..5.'},
            'max_motion_px': {'type': 'number', 'minimum': 0, 'maximum': 5, 'description': 'Maximum tracked-feature motion in pixels for a quiet sample, default 0.5, range 0..5; not proof of physical stability.'},
        }
        all_of.append({'if': {'required': ['settle']}, 'then': {'properties': {
            'settle': dict(type='object', description=action_field_help['settle'], properties=settle_props)}}})
    tool = dict(type='function', name='robot_action', description=description,
                inputSchema=dict(type='object', additionalProperties=False, properties=properties, allOf=all_of))
    preflight_example = (
        {'mode': 'preflight', 'observation_id': 'obs1', 'stages': [{'id': 'reach', 'kind': 'move', 'arm': arms[-1], 'grasp_position_m': [0.1, -0.2, 0.8], 'angles_deg': [0, 0, -60]}]}
        if grasp_angles else
        {'mode': 'preflight', 'observation_id': 'obs1', 'stages': [{'id': 'reach', 'kind': 'move', 'arm': arms[0], 'position_m': [0.1, -0.2, 0.2], 'quaternion_xyzw': [0, 0, 0, 1]}]}
    )
    tool['inputSchema']['examples'] = [
        {'observation_id': 'obs1', 'stages': [{'id': 'check', 'kind': 'observe', 'reason': 'inspect current support'}]},
        preflight_example,
        {'mode': 'execute', 'plan_id': 'p1', 'expected_cursor': 0, 'expected_revision': 0},
        {'mode': 'inspect', 'action_id': 'a1', 'path_kind': 'measured', 'detail': 'full'}]
    if reach:
        # Syntax only: two candidate poses of one target compared in one IK query (replace with poses from evidence).
        candidate = {'kind': 'move', 'arm': arms[0], 'position_m': [0.25, 0.0, 0.05], 'quaternion_xyzw': [1, 0, 0, 0]}
        tool['inputSchema']['examples'].append({'mode': 'reach', 'observation_id': 'obs1', 'stages': [
            dict(candidate, id='strict'),
            dict(candidate, id='tilt_ok', **({'approach_tolerance_rad': 0.35, 'roll_tolerance_rad': 0.1} if split
                                             else {'orientation_tolerance_rad': 0.35}))]})
    return SimpleNamespace(tool=tool, stage_schemas=stage_schemas, stage_common=stage_common, stage=stage,
                           stage_field_help=MappingProxyType(stage_field_help),
                           action_field_help=MappingProxyType(action_field_help))


def _validate_stage(stage, profile=None):
    from ..tools.registry import validator_for
    issue = next(validator_for(build(profile).stage_schemas[stage['kind']]).iter_errors(stage), None)
    if issue is not None:
        raise ValueError('invalid_robot_action_stage: ' + issue.message)


_DEFAULT_ROBOT_VIEWS = {
    'ROBOT_ACTION_TOOL': 'tool', '_STAGE_SCHEMAS': 'stage_schemas', '_STAGE_COMMON': 'stage_common',
    '_STAGE': 'stage', '_STAGE_FIELD_HELP': 'stage_field_help', '_ACTION_FIELD_HELP': 'action_field_help'}


def __getattr__(name):
    """Old module names: the default robot's rendering (``build(None)``), built on first access."""
    if name in _DEFAULT_ROBOT_VIEWS:
        return getattr(build(None), _DEFAULT_ROBOT_VIEWS[name])
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
