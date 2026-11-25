"""运动容差预设、受阻早停配置与末端残差诊断计算。

Presets and bounds are the run embodiment's (``EmbodimentProfile``, ``robot/model/embodiments.py``; built from
the run thresholds ``features.*`` on ARX-X5 and from the deployment file on SO-101): ``tolerance_presets``,
``default_tolerance_preset`` / ``default_tolerance``, ``max_position_tolerance_m`` / ``max_orientation_tolerance_rad``,
``stall_guard_presets_m`` and ``tracking_divergence_range_m``. Every function takes ``profile`` (None = the default
robot, ARX-X5 at code-default thresholds). Nothing is copied at import: ``TOLERANCE_PRESETS``,
``STALL_GUARD_PRESETS``, ``DEFAULT_TOLERANCE_PRESET``, ``MAX_POSITION_TOLERANCE_M``,
``MAX_ORIENTATION_TOLERANCE_RAD``, ``MIN_TRACKING_DIVERGENCE_M`` and ``MAX_TRACKING_DIVERGENCE_M`` are views of the
default robot's profile, kept for old imports.
"""

import math

import numpy as np

from ...core.action_validation import embodiment_or_default
from ..model.embodiments import (MAX_ARRIVAL_CORRECTION_OFFSET_RAD, MAX_ARRIVAL_CORRECTION_ROUNDS,
                                 ORIENTATION_FREE_TOLERANCE_RAD)
from .feedback_motion import pose_error

# ORIENTATION_FREE_TOLERANCE_RAD (rad, pi rounded to 5 decimals, so 3.14159, 3.14159265 and math.pi all qualify;
# defined once in embodiments): orientation tolerances at or above it leave the TCP orientation unconstrained: IK
# backends solve position only and the arrival check skips orientation. The ARX-X5 maximum
# (max_orientation_tolerance_rad = 3.14159265) and the unconstrained_orientation preset qualify.

def orientation_is_free(orientation_tolerance_rad):
    """True when an orientation tolerance (rad) leaves the TCP orientation unconstrained."""
    return float(orientation_tolerance_rad) >= ORIENTATION_FREE_TOLERANCE_RAD


#: Stall-guard preset applied when neither the stage nor the batch names one (no stall guard).
DEFAULT_STALL_GUARD_PRESET = 'off'
#: Largest accepted max_upward_dz_m (m): the vertical arrival window of the shared action language.
MAX_UPWARD_DZ_M = 0.15
MAX_DOWNWARD_DZ_M = 0.15

#: Split orientation bounds (rad): the TCP approach axis (local +Z) and the roll about it, each in (0, pi]; offered only
#: by embodiments with the ``approach_roll`` tag (the SO-101 with its cuRobo IK service).
SPLIT_ORIENTATION_FIELDS = ('approach_tolerance_rad', 'roll_tolerance_rad')
TOLERANCE_FIELDS = (
    'tolerance_preset',
    'position_tolerance_m',
    'orientation_tolerance_rad',
    'max_upward_dz_m',
    'max_downward_dz_m',
    *SPLIT_ORIENTATION_FIELDS,
)
STALL_GUARD_FIELDS = ('stall_guard_preset', 'max_tracking_divergence_m')
#: Arrival-correction bounds of a move (embodiments with the ``arrival_correction`` tag, the SO-101): how many times
#: (count) and how far (rad per joint) the joint goal may be shifted past the planned target to cancel the settled
#: shortfall. The schema and ``resolve_arrival_correction`` accept [0, max_rounds] and (0, max_offset_rad] of the
#: profile's ``arrival_correction_limits`` (SO-101: the deployment's own bounds, which are also the default);
#: MAX_ARRIVAL_CORRECTION_ROUNDS / MAX_ARRIVAL_CORRECTION_OFFSET_RAD (``embodiments``) bound every profile's limits.
ARRIVAL_CORRECTION_FIELDS = ('arrival_correction_rounds', 'arrival_correction_max_offset_rad')


def arrival_correction_bounds(profile=None):
    """``(max_rounds, max_offset_rad)`` a request may set on ``profile`` (None = the default robot): its
    ``arrival_correction_limits``, else the shared maxima (a profile restored from an archive written before the
    limits existed). Example: the SO-101 example deployment -> ``(3, 0.1)``."""
    limits = embodiment_or_default(profile).arrival_correction_limits
    if limits is None:
        return MAX_ARRIVAL_CORRECTION_ROUNDS, MAX_ARRIVAL_CORRECTION_OFFSET_RAD
    return limits.max_rounds, limits.max_offset_rad


def tolerance_preset_table(profile=None):
    """``{preset: {"position_tolerance_m" (m), "orientation_tolerance_rad" (rad)[, "max_upward_dz_m" (m)]}}`` of
    ``profile`` (None = the default robot), in profile order. Example: ``tolerance_preset_table()["normal"]`` ->
    ``{"position_tolerance_m": 0.01, "orientation_tolerance_rad": 0.15}``."""
    resolved = embodiment_or_default(profile)
    return {
        name: {
            'position_tolerance_m': value.position_m,
            'orientation_tolerance_rad': value.orientation_rad,
            **({'max_upward_dz_m': value.max_upward_dz_m} if value.max_upward_dz_m is not None else {}),
        }
        for name, value in resolved.tolerance_presets.items()
    }


_DEFAULT_ROBOT_VIEWS = {
    'TOLERANCE_PRESETS': tolerance_preset_table,
    'STALL_GUARD_PRESETS': lambda: dict(embodiment_or_default().stall_guard_presets_m),
    'DEFAULT_TOLERANCE_PRESET': lambda: embodiment_or_default().default_tolerance_preset,
    'MAX_POSITION_TOLERANCE_M': lambda: embodiment_or_default().max_position_tolerance_m,
    'MAX_ORIENTATION_TOLERANCE_RAD': lambda: embodiment_or_default().max_orientation_tolerance_rad,
    'MIN_TRACKING_DIVERGENCE_M': lambda: embodiment_or_default().tracking_divergence_range_m[0],
    'MAX_TRACKING_DIVERGENCE_M': lambda: embodiment_or_default().tracking_divergence_range_m[1],
}


def __getattr__(name):
    """Old module names, read from the default robot's profile (never an import-time copy)."""
    if name in _DEFAULT_ROBOT_VIEWS:
        return _DEFAULT_ROBOT_VIEWS[name]()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def validate_tolerance_fields(mapping, *, profile=None):
    """校验 mapping 中的容差参数合法性。

    ``tolerance_preset`` must be a preset of ``profile`` (None = the default robot); ``position_tolerance_m`` (m) in
    (0, profile.max_position_tolerance_m]; ``orientation_tolerance_rad`` (rad) in
    (0, profile.max_orientation_tolerance_rad]; ``max_upward_dz_m`` (m) in [0, MAX_UPWARD_DZ_M];
    ``approach_tolerance_rad`` / ``roll_tolerance_rad`` (rad) in (0, pi], only on a profile with ``approach_roll``.
    """
    profile = embodiment_or_default(profile)
    split = [field for field in SPLIT_ORIENTATION_FIELDS if mapping.get(field) is not None]
    if split and 'approach_roll' not in profile.supports:
        raise ValueError('unsupported_capability: ' + ' / '.join(split))
    for field in split:
        value = mapping[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not (0 < float(value) <= math.pi + 1e-9):
            raise ValueError('invalid_' + field)
    preset = mapping.get('tolerance_preset')
    if preset is not None and (not isinstance(preset, str) or preset not in profile.tolerance_presets):
        raise ValueError('invalid_tolerance_preset')
    if 'position_tolerance_m' in mapping and mapping['position_tolerance_m'] is not None:
        pos_tol = mapping['position_tolerance_m']
        if isinstance(pos_tol, bool) or not isinstance(pos_tol, (int, float)) or not math.isfinite(pos_tol) or not (0 < float(pos_tol) <= profile.max_position_tolerance_m):
            raise ValueError('invalid_position_tolerance_m')
    if 'orientation_tolerance_rad' in mapping and mapping['orientation_tolerance_rad'] is not None:
        ori_tol = mapping['orientation_tolerance_rad']
        if isinstance(ori_tol, bool) or not isinstance(ori_tol, (int, float)) or not math.isfinite(ori_tol) or not (0 < float(ori_tol) <= profile.max_orientation_tolerance_rad):
            raise ValueError('invalid_orientation_tolerance_rad')
    if 'max_upward_dz_m' in mapping and mapping['max_upward_dz_m'] is not None:
        dz = mapping['max_upward_dz_m']
        if isinstance(dz, bool) or not isinstance(dz, (int, float)) or not math.isfinite(dz) or not (0.0 <= float(dz) <= MAX_UPWARD_DZ_M):
            raise ValueError('invalid_max_upward_dz_m')
    if 'max_downward_dz_m' in mapping and mapping['max_downward_dz_m'] is not None:
        down_dz = mapping['max_downward_dz_m']
        if isinstance(down_dz, bool) or not isinstance(down_dz, (int, float)) or not math.isfinite(down_dz) or not (0.0 <= float(down_dz) <= MAX_DOWNWARD_DZ_M):
            raise ValueError('invalid_max_downward_dz_m')


def validate_stall_guard_fields(mapping, *, profile=None):
    """校验 mapping 中的受阻早停参数合法性。

    ``stall_guard_preset`` must be a preset of ``profile.stall_guard_presets_m`` (None profile = the default robot);
    ``max_tracking_divergence_m`` (m) inside ``profile.tracking_divergence_range_m`` (inclusive).
    """
    profile = embodiment_or_default(profile)
    preset = mapping.get('stall_guard_preset')
    if preset is not None and (not isinstance(preset, str) or preset not in profile.stall_guard_presets_m):
        raise ValueError('invalid_stall_guard_preset')
    if 'max_tracking_divergence_m' in mapping and mapping['max_tracking_divergence_m'] is not None:
        div = mapping['max_tracking_divergence_m']
        low, high = profile.tracking_divergence_range_m
        if isinstance(div, bool) or not isinstance(div, (int, float)) or not math.isfinite(div) or not (low <= float(div) <= high):
            raise ValueError('invalid_max_tracking_divergence_m')


def resolve_stage_tolerance(stage, batch_defaults=None, *, profile=None):
    """解析单个 move 阶段的生效位置、姿态及上下双向垂直顺应容差。

    优先级：stage 显式数值 > stage tolerance_preset > batch 显式数值 > batch tolerance_preset > the embodiment default.
    Presets are ``profile.tolerance_presets`` (None profile = the default robot). The default, applied when neither
    the stage nor the batch names a preset, is ``profile.default_tolerance`` (ARX-X5: the ``normal`` preset,
    0.01 m / 0.15 rad at default thresholds; SO-101: the deployment ``motion.default_*_tolerance_*``). The returned
    ``tolerance_preset`` names the preset whose numbers were the base, or None when the embodiment default is not
    equal to its ``default_tolerance_preset``. Returns ``tolerance_preset``, ``position_tolerance_m`` (m),
    ``orientation_tolerance_rad`` (rad) and, when set, ``max_upward_dz_m`` / ``max_downward_dz_m`` (m) and the split
    orientation bounds ``approach_tolerance_rad`` / ``roll_tolerance_rad`` (rad; stage value, else the batch value;
    presets set none).
    """
    profile = embodiment_or_default(profile)
    defaults = dict(batch_defaults or {})
    current = dict(stage or {})
    validate_tolerance_fields(defaults, profile=profile)
    validate_tolerance_fields(current, profile=profile)

    presets = profile.tolerance_presets

    def numbers(tolerance):
        entry = {'position_tolerance_m': tolerance.position_m, 'orientation_tolerance_rad': tolerance.orientation_rad}
        if tolerance.max_upward_dz_m is not None:
            entry['max_upward_dz_m'] = tolerance.max_upward_dz_m
        return entry

    batch_preset = defaults.get('tolerance_preset')
    if batch_preset is None:
        default_name = profile.default_tolerance_preset
        resolved_preset = default_name if presets.get(default_name) == profile.default_tolerance else None
        resolved = numbers(profile.default_tolerance)
    else:
        resolved = numbers(presets[batch_preset])
        resolved_preset = batch_preset

    if defaults.get('position_tolerance_m') is not None:
        resolved['position_tolerance_m'] = float(defaults['position_tolerance_m'])
    if defaults.get('orientation_tolerance_rad') is not None:
        resolved['orientation_tolerance_rad'] = float(defaults['orientation_tolerance_rad'])
    if defaults.get('max_upward_dz_m') is not None:
        resolved['max_upward_dz_m'] = float(defaults['max_upward_dz_m'])
    if defaults.get('max_downward_dz_m') is not None:
        resolved['max_downward_dz_m'] = float(defaults['max_downward_dz_m'])

    stage_preset = current.get('tolerance_preset')
    if stage_preset is not None:
        resolved = numbers(presets[stage_preset])
        resolved_preset = stage_preset

    if current.get('position_tolerance_m') is not None:
        resolved['position_tolerance_m'] = float(current['position_tolerance_m'])
    if current.get('orientation_tolerance_rad') is not None:
        resolved['orientation_tolerance_rad'] = float(current['orientation_tolerance_rad'])
    if current.get('max_upward_dz_m') is not None:
        resolved['max_upward_dz_m'] = float(current['max_upward_dz_m'])
    if current.get('max_downward_dz_m') is not None:
        resolved['max_downward_dz_m'] = float(current['max_downward_dz_m'])

    out = {
        'tolerance_preset': resolved_preset,
        'position_tolerance_m': round(float(resolved['position_tolerance_m']), 5),
        'orientation_tolerance_rad': round(float(resolved['orientation_tolerance_rad']), 5),
    }
    if resolved.get('max_upward_dz_m') is not None:
        out['max_upward_dz_m'] = round(float(resolved['max_upward_dz_m']), 5)
    if resolved.get('max_downward_dz_m') is not None:
        out['max_downward_dz_m'] = round(float(resolved['max_downward_dz_m']), 5)
    for field in SPLIT_ORIENTATION_FIELDS:
        value = current.get(field)
        if value is None:
            value = defaults.get(field)
        if value is not None:
            out[field] = round(float(value), 5)
    return out


def resolve_arrival_correction(stage, batch_defaults=None, *, profile=None):
    """The arrival-correction bounds one move stage sets: ``{field: value}`` of ``ARRIVAL_CORRECTION_FIELDS``.

    Stage value, else the batch value; a field neither sets is absent (the backend then uses its deployment bound).
    ``arrival_correction_rounds``: integer in [0, max_rounds] (0 = no correction); ``arrival_correction_max_offset_rad``:
    number in (0, max_offset_rad] rad, the bounds of ``arrival_correction_bounds(profile)``; a value outside raises
    ``invalid_arrival_correction_<field>: ... must be in ...`` naming the bound (refused, never clamped). Any of them on
    a profile without the ``arrival_correction`` tag raises ``unsupported_capability``.
    Example: ``resolve_arrival_correction({"arrival_correction_rounds": 0}, {})`` -> ``{"arrival_correction_rounds": 0}``.
    """
    profile = embodiment_or_default(profile)
    max_rounds, max_offset = arrival_correction_bounds(profile)
    out = {}
    for mapping in (batch_defaults or {}, stage or {}):
        for field in ARRIVAL_CORRECTION_FIELDS:
            value = mapping.get(field)
            if value is None:
                continue
            if 'arrival_correction' not in profile.supports:
                raise ValueError('unsupported_capability: ' + field)
            if field == 'arrival_correction_rounds':
                if type(value) is not int or not 0 <= value <= max_rounds:
                    raise ValueError(f'invalid_arrival_correction_rounds: must be an integer in [0, {max_rounds}]')
                out[field] = value
            else:
                if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                        or not 0 < float(value) <= max_offset):
                    raise ValueError(f'invalid_arrival_correction_max_offset_rad: must be in (0, {max_offset:g}] rad')
                out[field] = float(value)
    return out


def resolve_stall_guard(stage, batch_defaults=None, *, profile=None):
    """解析单个 move 阶段的受阻早停配置。

    优先级：stage max_tracking_divergence_m > stage stall_guard_preset > batch max_tracking_divergence_m > batch stall_guard_preset > 默认 off (None)。
    Presets are ``profile.stall_guard_presets_m`` (m or None = off; None profile = the default robot).
    """
    profile = embodiment_or_default(profile)
    defaults = dict(batch_defaults or {})
    current = dict(stage or {})
    validate_stall_guard_fields(defaults, profile=profile)
    validate_stall_guard_fields(current, profile=profile)

    presets = profile.stall_guard_presets_m
    batch_preset = defaults.get('stall_guard_preset') or DEFAULT_STALL_GUARD_PRESET
    resolved_div = presets.get(batch_preset)
    resolved_preset = batch_preset

    if defaults.get('max_tracking_divergence_m') is not None:
        resolved_div = float(defaults['max_tracking_divergence_m'])

    stage_preset = current.get('stall_guard_preset')
    if stage_preset is not None:
        resolved_div = presets[stage_preset]
        resolved_preset = stage_preset

    if current.get('max_tracking_divergence_m') is not None:
        resolved_div = float(current['max_tracking_divergence_m'])

    return {
        'stall_guard_preset': resolved_preset,
        'max_tracking_divergence_m': None if resolved_div is None else round(float(resolved_div), 5),
    }


def compute_endpoint_residual(target_pose, measured_pose):
    """计算带符号三轴位置残差（实测减目标）及标量位置/姿态误差。"""
    target_pos = np.asarray(target_pose['position_m'], dtype=float)
    measured_pos = np.asarray(measured_pose['position_m'], dtype=float)
    delta = measured_pos - target_pos
    pos_err, ori_err = pose_error(measured_pose, target_pose)
    return {
        'dx_m': round(float(delta[0]), 5),
        'dy_m': round(float(delta[1]), 5),
        'dz_m': round(float(delta[2]), 5),
        'position_error_m': round(pos_err, 5),
        'orientation_error_rad': round(ori_err, 5),
    }
