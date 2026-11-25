"""Resolved motion, contact, gripper, travel, and recovery thresholds."""
from __future__ import annotations

import json
from pathlib import Path

from ..embodiment import FrozenDict

_ARX_X5_DEFAULTS_PATH = Path(__file__).resolve().parents[2] / 'robot' / 'embodiments' / 'arx_x5' / 'defaults.json'

MOTION_THRESHOLD_BOUNDS: dict[str, tuple[type, float | int, float | int]] = {
    'tight_position_tolerance_m': (float, 0.001, 0.05),
    'default_position_tolerance_m': (float, 0.001, 0.05),
    'relaxed_position_tolerance_m': (float, 0.001, 0.05),
    'compliant_position_tolerance_m': (float, 0.001, 0.05),
    'coarse_position_tolerance_m': (float, 0.001, 0.05),
    'max_position_tolerance_m': (float, 0.005, 0.2),
    'tight_orientation_tolerance_rad': (float, 0.005, 3.14159265),
    'orientation_tolerance_rad': (float, 0.005, 3.14159265),
    'relaxed_orientation_tolerance_rad': (float, 0.005, 3.14159265),
    'coarse_orientation_tolerance_rad': (float, 0.005, 3.14159265),
    'max_orientation_tolerance_rad': (float, 0.05, 3.14159265),
    'contact_divergence_threshold_m': (float, 0.0005, 0.05),
    'stall_guard_strict_divergence_m': (float, 0.002, 0.15),
    'stall_guard_contact_stop_divergence_m': (float, 0.002, 0.15),
    'stall_guard_compliant_divergence_m': (float, 0.002, 0.15),
    'stall_guard_min_divergence_m': (float, 0.001, 0.05),
    'stall_guard_max_divergence_m': (float, 0.01, 0.30),
    'stall_guard_step_velocity_threshold_m': (float, 0.0001, 0.02),
    'stall_guard_consecutive_steps': (int, 1, 20),
    'max_gripper_aperture_m': (float, 0.01, 0.20),
    'grip_empty_threshold_mm': (float, 0.0, 40.0),
    'grip_thin_threshold_mm': (float, 0.0, 40.0),
    'grip_holding_threshold_mm': (float, 0.1, 40.0),
    'grip_solid_threshold_mm': (float, 0.1, 40.0),
    'grip_non_empty_min_aperture_m': (float, 0.0, 0.20),
    'grip_thin_min_aperture_m': (float, 0.0, 0.20),
    'grip_rigid_min_aperture_m': (float, 0.0, 0.20),
    'grip_wide_min_aperture_m': (float, 0.0, 0.20),
    'joint_travel_standard_max_rad': (float, 0.5, 20.0),
    'joint_travel_standard_backtracking_rad': (float, 0.2, 20.0),
    'joint_travel_extended_max_rad': (float, 0.5, 20.0),
    'joint_travel_extended_backtracking_rad': (float, 0.2, 20.0),
    'segment_start_joint_drift_rad': (float, 0.001, 3.14159265),
    'resume_max_joint_drift_rad': (float, 0.0001, 1.0),
    'resume_max_ee_drift_m': (float, 0.0001, 0.1),
    'resume_max_orientation_drift_rad': (float, 0.0001, 1.0),
    'resume_max_jaw_drift_normalized': (float, 0.0001, 1.0),
    'batch_plan_cache_size': (int, 1, 256),
    'hotspot_xy_radius_m': (float, 0.005, 0.20),
    'hotspot_z_radius_m': (float, 0.005, 0.20),
    'ledger_max_recent_outcomes': (int, 1, 20),
    'ledger_max_hotspots': (int, 1, 32),
    'target_exemption_radius_m': (float, 0.0, 0.15),
    'clearance_height_m': (float, 0.0, 0.25),
}

MOTION_THRESHOLD_FIELDS = tuple(MOTION_THRESHOLD_BOUNDS)


def _load_packaged_thresholds() -> FrozenDict:
    raw = json.loads(_ARX_X5_DEFAULTS_PATH.read_text(encoding='utf-8'))['thresholds']
    return FrozenDict((name, raw[name]) for name in MOTION_THRESHOLD_FIELDS)


_DEFAULT_MOTION_THRESHOLDS = _load_packaged_thresholds()


def _restore_threshold_map(mapping: dict, overrides: dict) -> ThresholdMap:
    return ThresholdMap(mapping, overrides=overrides)


class ThresholdMap(FrozenDict):
    """Immutable dict + attribute mapping of resolved embodiment motion thresholds."""

    __slots__ = ('_overrides',)

    def __init__(self, mapping=(), *, overrides=()):
        super().__init__(mapping)
        object.__setattr__(self, '_overrides', FrozenDict(overrides))

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f'{type(self).__name__!r} object has no attribute {name!r}') from None

    def __setattr__(self, name: str, value) -> None:
        raise TypeError('embodiment_thresholds_are_immutable')

    def __reduce__(self):
        return _restore_threshold_map, (dict(self), dict(self._overrides))


def motion_thresholds(source=None):
    """提取统一配置中的运动、接触、夹爪、行程与恢复经验阈值，支持 Config/EmbodimentConfig/FeaturesConfig/Harness/Adapter 或字典输入。"""
    result = dict(_DEFAULT_MOTION_THRESHOLDS)
    if source is None:
        return result
    if isinstance(source, dict):
        nested_feat = source.get('features')
        if isinstance(nested_feat, dict):
            for name in MOTION_THRESHOLD_FIELDS:
                if name in nested_feat and nested_feat[name] is not None:
                    result[name] = nested_feat[name]
        nested_emb = source.get('embodiment')
        if isinstance(nested_emb, dict) and isinstance(nested_emb.get('thresholds'), dict):
            for name in MOTION_THRESHOLD_FIELDS:
                if name in nested_emb['thresholds'] and nested_emb['thresholds'][name] is not None:
                    result[name] = nested_emb['thresholds'][name]
        for name in MOTION_THRESHOLD_FIELDS:
            if name in source and source[name] is not None:
                result[name] = source[name]
        return result

    cfg = getattr(source, 'config', source)
    emb = getattr(cfg, 'embodiment', None)
    if emb is not None and isinstance(getattr(emb, 'thresholds', None), dict):
        for name in MOTION_THRESHOLD_FIELDS:
            val = emb.thresholds.get(name)
            if val is not None:
                result[name] = val
    elif isinstance(getattr(source, 'thresholds', None), dict):
        for name in MOTION_THRESHOLD_FIELDS:
            val = source.thresholds.get(name)
            if val is not None:
                result[name] = val

    feat = getattr(cfg, 'features', None)
    for candidate in (feat, source):
        if candidate is None or isinstance(candidate, dict):
            continue
        bound = getattr(candidate, '__dict__', {}).get('_embodiment_thresholds')
        if isinstance(bound, dict):
            for name in MOTION_THRESHOLD_FIELDS:
                if name in bound and bound[name] is not None:
                    result[name] = bound[name]

    explicit_dict = getattr(source, 'motion_thresholds', None)
    if isinstance(explicit_dict, dict):
        for name in MOTION_THRESHOLD_FIELDS:
            if name in explicit_dict and explicit_dict[name] is not None:
                result[name] = explicit_dict[name]

    for target in (feat, source):
        if target is None or isinstance(target, dict) or hasattr(target, 'to_dict'):
            continue
        for name in MOTION_THRESHOLD_FIELDS:
            val = getattr(target, name, None)
            if val is not None and type(val) in (int, float):
                result[name] = val
    return result
