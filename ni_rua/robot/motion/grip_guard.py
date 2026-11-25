"""夹爪本体事实计算与基于指尖开度区间 (米) 的条件门禁。

The guarded gripper is described by the run's ``EmbodimentProfile`` (``profile.gripper``, design 13 section 5.1):
its preset margins ``preset_margins_m``, its zero point ``empty_close_aperture_m`` and its single full opening
``full_open_aperture_m`` (decision V-D). Every rule below takes ``profile`` (None = the default robot, ARX-X5 at
code-default thresholds: zero 0.0 m, full open 0.088 m) and optional explicit ``empty_close_aperture_m`` /
``max_aperture_m`` (m) that override the profile's values (backends that pass their measured gripper directly).
No preset table is copied at import: the names ``GRIP_GUARD_PRESETS``, ``DEFAULT_MAX_APERTURE_M`` and
``DEFAULT_EMPTY_CLOSE_APERTURE_M`` are views of the default robot's profile, kept for old imports.
"""

from copy import deepcopy
import math
import numpy as np

from ...core.action_validation import embodiment_or_default
from ...core.config import motion_thresholds

GRIP_GUARD_FIELDS = ("grip_guard_preset", "min_aperture_m", "max_aperture_m")
CARRY_SLIP_GUARD_FIELDS = ("carry_slip_guard", "carry_slip_scope", "slip_consecutive_steps")
CARRY_SLIP_SCOPES = ("next_move", "until_release")
CARRY_SLIP_CONSECUTIVE_STEPS_BOUNDS = (1, 50)
CARRY_SLIP_REASON = "carry_slip_aperture_below_min_m"
#: Jaw-stage field of a gripper with a bounded holding squeeze (``profile.gripper.hold_squeeze``, SO-101): rad of the
#: jaw joint the held goal lies beyond the measured contact angle, in the closing direction.
GRIP_SQUEEZE_FIELD = "grip_squeeze_rad"


def validate_grip_squeeze(mapping, *, profile=None):
    """The ``grip_squeeze_rad`` of one stage (rad of the jaw joint, float), or None when the stage sets none.

    ``profile``: the run's ``EmbodimentProfile`` (None = the default robot, ARX-X5, which has no holding squeeze).
    Refused, never clamped: ``unsupported_capability`` on a gripper without ``gripper.hold_squeeze``, and
    ``invalid_grip_squeeze_rad`` for a value that is not a finite number in ``[0, hold_squeeze.max_rad]``.
    Example: ``validate_grip_squeeze({"grip_squeeze_rad": 0.02}, profile=so101_profile)`` -> ``0.02``."""
    if not isinstance(mapping, dict) or GRIP_SQUEEZE_FIELD not in mapping:
        return None
    squeeze = embodiment_or_default(profile).gripper.hold_squeeze
    if squeeze is None:
        raise ValueError("unsupported_capability: grip_squeeze_rad (this gripper applies no holding squeeze; SO-101: "
                         "deployment gripper.hold_squeeze_rad is not set)")
    value = mapping[GRIP_SQUEEZE_FIELD]
    if type(value) not in (int, float) or not math.isfinite(value) or not 0.0 <= value <= squeeze.max_rad:
        raise ValueError(f"invalid_grip_squeeze_rad: must be a number in [0, {squeeze.max_rad:g}] rad of the jaw joint "
                         "(refused, never clamped)")
    return float(value)


def guard_gripper(profile=None, *, empty_close_aperture_m=None, max_aperture_m=None):
    """``(preset_margins_m, empty_close_aperture_m, max_aperture_m)`` of the guarded gripper.

    ``preset_margins_m``: ``{preset: (min_margin_m or None, max_margin_m or None)}`` above the zero point
    (``profile.gripper.preset_margins_m``; its keys are the accepted ``grip_guard_preset`` names).
    ``empty_close_aperture_m`` (m, or None = not measured, presets then cannot resolve): the explicit argument,
    else ``profile.gripper.empty_close_aperture_m``. ``max_aperture_m`` (m): the explicit argument, else
    ``profile.gripper.full_open_aperture_m``, the upper bound of explicit ``min_aperture_m`` / ``max_aperture_m``.
    ``profile``: the run's ``EmbodimentProfile``; None = the default robot.
    Example: ``guard_gripper()[1:]`` -> ``(0.0, 0.088)``.
    """
    gripper = embodiment_or_default(profile).gripper
    zero = gripper.empty_close_aperture_m if empty_close_aperture_m is None else empty_close_aperture_m
    maximum = gripper.full_open_aperture_m if max_aperture_m is None else max_aperture_m
    return gripper.preset_margins_m, zero, maximum


def grip_guard_presets(profile=None):
    """``{preset: {"min_aperture_m", "max_aperture_m"}}`` margins (m above the empty-close zero point, None =
    unbounded) of ``profile`` (None = the default robot), in profile order."""
    margins, _, _ = guard_gripper(profile)
    return {name: {"min_aperture_m": low, "max_aperture_m": high} for name, (low, high) in margins.items()}


def __getattr__(name):
    """Old module names, read from the default robot's profile (never an import-time copy):
    ``GRIP_GUARD_PRESETS`` (``grip_guard_presets()``), ``DEFAULT_MAX_APERTURE_M`` (full opening, m) and
    ``DEFAULT_EMPTY_CLOSE_APERTURE_M`` (zero point, m)."""
    if name == "GRIP_GUARD_PRESETS":
        return grip_guard_presets()
    if name == "DEFAULT_MAX_APERTURE_M":
        return guard_gripper()[2]
    if name == "DEFAULT_EMPTY_CLOSE_APERTURE_M":
        return guard_gripper()[1]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _max_aperture_m(adapter=None, arm=None):
    """获取当前夹爪归一化 1.0 对应的指尖最大间距（米）。

    ``adapter.grasp_geometry`` when it declares one, else the run threshold ``max_gripper_aperture_m`` of the
    adapter (``core.config.motion_thresholds(adapter)``; the ARX-X5 worker receives the run's values), which is the
    ARX-X5 profile's ``gripper.full_open_aperture_m`` (default 0.088 m)."""
    geom = getattr(adapter, "grasp_geometry", None) if adapter is not None else None
    if isinstance(geom, dict) and type(geom.get("max_aperture_m")) in (int, float):
        return float(geom["max_aperture_m"])
    if hasattr(geom, "max_aperture_m") and type(geom.max_aperture_m) in (int, float):
        return float(geom.max_aperture_m)
    return float(motion_thresholds(adapter)["max_gripper_aperture_m"])


def _aperture_from_jaw(jaw, adapter=None, arm=None):
    """由归一化开度计算指尖间距（米）。优先使用 adapter.grasp_geometry 几何映射。"""
    if jaw is None:
        return None
    geom = getattr(adapter, "grasp_geometry", None) if adapter is not None else None
    if hasattr(geom, "aperture_m") and callable(geom.aperture_m):
        return float(geom.aperture_m(jaw))
    if callable(geom):
        value = geom(jaw) if arm is None else geom(jaw, arm=arm)
        if type(value) in (int, float):
            return float(value)
    return float(jaw) * _max_aperture_m(adapter, arm)


def _measured_physical_jaw(adapter, state, arm):
    """提取真实物理夹爪开度归一化值 [0, 1]（0.0 对应双指物理闭合接触 0mm，忽略 PD 控制器负向过压区间）。"""
    from collections.abc import Mapping
    arm_state = state["arms"].get(arm) if isinstance(state, Mapping) and isinstance(state.get("arms"), Mapping) else state
    if not isinstance(arm_state, Mapping):
        return None
    gripper = arm_state.get("gripper")
    if not isinstance(gripper, Mapping) or gripper.get("position_source_kind") != "measurement":
        return None
    pos = gripper.get("position_m")
    if not isinstance(pos, (list, tuple, np.ndarray)) or len(pos) == 0:
        return None
    get_robot = getattr(getattr(adapter, "robot", None), "get_robot_by_arm_name", None)
    if callable(get_robot):
        try:
            robot = get_robot(f"{arm}_arm")
            move = getattr(robot, "gripper_move", None)
            joint_names_raw = getattr(robot, "gripper_joints_name", None)
            scale = getattr(robot, "gripper_scale", None)
            if (
                isinstance(move, Mapping)
                and "base" in move
                and "sign" in move
                and joint_names_raw is not None
                and isinstance(scale, (list, tuple, np.ndarray))
                and len(scale) == 2
            ):
                names = list(joint_names_raw)
                base = names.index(move["base"])
                if 0 <= base < len(pos):
                    value = float(pos[base])
                    lower, upper = scale
                    if move["sign"] == 1:
                        phys_lower = max(0.0, float(lower))
                        span = float(upper) - phys_lower
                        normalized = (value - phys_lower) / span if span > 0 else 0.0
                    else:
                        phys_upper = min(0.0, float(upper))
                        span = phys_upper - float(lower)
                        normalized = (phys_upper - value) / span if span > 0 else 0.0
                    if math.isfinite(normalized):
                        return float(np.clip(normalized, 0.0, 1.0))
        except ValueError:
            pass
    val = float(pos[0])
    if math.isfinite(val):
        half_max = _max_aperture_m(adapter, arm) / 2.0
        return float(np.clip(val / half_max if half_max > 0 else 0.0, 0.0, 1.0))
    return None


def _extract_jaw(state, adapter=None, arm=None):
    """从 7 维向量、标量或机器人状态字典中提取真实物理归一化夹爪开度 [0, 1]。"""
    if state is None:
        return None
    if type(state) in (int, float) and math.isfinite(state):
        return float(np.clip(state, 0.0, 1.0))
    if isinstance(state, (list, tuple, np.ndarray)):
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 1 and arr.size >= 7 and math.isfinite(arr[6]):
            return float(np.clip(arr[6], 0.0, 1.0))
        return None
    if isinstance(state, dict):
        if "arms" in state and isinstance(state["arms"], dict):
            if arm in state["arms"]:
                phys = _measured_physical_jaw(adapter, state, arm)
                if phys is not None:
                    return phys
                return _extract_jaw(state["arms"][arm], adapter=adapter, arm=arm)
            return None
        if arm is not None:
            phys = _measured_physical_jaw(adapter, {"arms": {arm: state}}, arm)
            if phys is not None:
                return phys
        if type(state.get("measured_jaw")) in (int, float) and math.isfinite(state["measured_jaw"]):
            return float(np.clip(state["measured_jaw"], 0.0, 1.0))
        gripper = state.get("gripper")
        if isinstance(gripper, dict):
            pos = gripper.get("position_m")
            if isinstance(pos, (list, tuple, np.ndarray)) and len(pos) > 0:
                val = float(pos[0])
                if math.isfinite(val):
                    half_max = _max_aperture_m(adapter, arm) / 2.0
                    return float(np.clip(val / half_max if half_max > 0 else 0.0, 0.0, 1.0))
            for key in ("goal_normalized", "command_normalized"):
                val = gripper.get(key)
                if type(val) in (int, float) and math.isfinite(val):
                    return float(np.clip(val, 0.0, 1.0))
    return None


def compute_grip_facts(arm_state_before, arm_state_after, adapter=None, *, arm=None):
    """计算夹爪动作前后的客观本体测量事实，不作空抓或接触猜测。"""
    jaw_before = _extract_jaw(arm_state_before, adapter=adapter, arm=arm)
    jaw_after = _extract_jaw(arm_state_after, adapter=adapter, arm=arm)
    ap_before = _aperture_from_jaw(jaw_before, adapter=adapter, arm=arm)
    ap_after = _aperture_from_jaw(jaw_after, adapter=adapter, arm=arm)
    return grip_facts_from_measurements(jaw_before, jaw_after, ap_before, ap_after)


def grip_facts_from_measurements(jaw_before, jaw_after, aperture_before_m, aperture_after_m):
    """``grip_facts`` from measured jaw openings (normalized, 0 closed .. 1 open) and fingertip gaps (m).

    Any value may be None (not measured). Used directly by backends that measure the gap
    themselves (SO-101: CAD opening of the measured jaw angle), so their receipts share the
    ARX-X5 ``grip_facts`` shape: ``measured_jaw_before/after``, ``jaw_delta``,
    ``measured_aperture_m`` (gap after the stage) and ``aperture_delta_m``.
    """
    ap_before, ap_after = aperture_before_m, aperture_after_m
    jaw_delta = (jaw_after - jaw_before) if (jaw_before is not None and jaw_after is not None) else None
    ap_delta = (ap_after - ap_before) if (ap_before is not None and ap_after is not None) else None
    return {
        "measured_jaw_before": round(float(jaw_before), 6) if jaw_before is not None else None,
        "measured_jaw_after": round(float(jaw_after), 6) if jaw_after is not None else None,
        "jaw_delta": round(float(jaw_delta), 6) if jaw_delta is not None else None,
        "measured_aperture_m": round(float(ap_after), 6) if ap_after is not None else None,
        "aperture_delta_m": round(float(ap_delta), 6) if ap_delta is not None else None,
    }


def preset_aperture_bounds(preset, *, profile=None, empty_close_aperture_m=None):
    """Effective ``(min_aperture_m, max_aperture_m)`` fingertip gaps (m) of one preset.

    ``preset``: a key of the profile's ``gripper.preset_margins_m``, or None (no bound). Preset numbers are margins
    above the gripper's empty-close gap (``empty_close_aperture_m`` argument, else the profile's; ARX-X5 0.0), so
    the same preset keeps its meaning on a gripper whose empty close reads a nonzero gap. An unmeasured gap (None)
    resolves no bounded preset (``ValueError``). ``profile``: the run's ``EmbodimentProfile``; None = the default
    robot. Example: ``preset_aperture_bounds("non_empty", empty_close_aperture_m=0.005)`` -> ``(0.0065, None)``
    with the default 1.5 mm non_empty margin.
    """
    if preset is None:
        return (None, None)
    margins, zero, _ = guard_gripper(profile, empty_close_aperture_m=empty_close_aperture_m)
    bounds = margins[preset]
    if zero is None and any(margin is not None for margin in bounds):
        raise ValueError("unsupported_capability: grip_guard_preset requires the measured empty-close fingertip gap "
                         "of this gripper (gripper.empty_close_aperture_m, m); use explicit min_aperture_m / "
                         "max_aperture_m (m) instead")
    return tuple(None if margin is None else float(zero) + margin for margin in bounds)


def _has_carry_slip_fields(mapping):
    return isinstance(mapping, dict) and any(k in mapping for k in CARRY_SLIP_GUARD_FIELDS)


def validate_carry_slip_syntax(mapping):
    """Validate syntax, types, and bounds of ``CARRY_SLIP_GUARD_FIELDS`` on ``mapping`` without resolving aperture bounds.

    Parameters:
        mapping (dict): Stage or batch dictionary that may contain:
            - ``carry_slip_guard`` (``bool``, default ``False``): opt-in per-step fingertip aperture monitor
              armed by a passing closing jaw stage (``goal_normalized < 0.5``).
            - ``carry_slip_scope`` (``"next_move" | "until_release"``, default ``"next_move"``): whether per-step
              slip checks cover only the immediately following same-arm ``move`` stage or all subsequent same-arm
              ``move`` stages until a jaw opening event (``goal_normalized >= 0.5``).
            - ``slip_consecutive_steps`` (``int`` in ``[1, 50]``, unit: control steps/ticks, default ``1``):
              consecutive native steps/ticks with ``measured_aperture_m < min_aperture_m`` required to pause.

    Returns:
        tuple[bool, str, int]: ``(carry_slip_guard, carry_slip_scope, slip_consecutive_steps)``.

    Raises:
        ValueError: With an ``invalid_carry_slip_*`` code if types, bounds, or combinations are invalid.

    Example:
        ``validate_carry_slip_syntax({"carry_slip_guard": True, "slip_consecutive_steps": 2})``
        -> ``(True, "next_move", 2)``.
    """
    if not isinstance(mapping, dict):
        raise ValueError("invalid_grip_guard_mapping")
    if "carry_slip_guard" in mapping:
        enabled = mapping["carry_slip_guard"]
        if type(enabled) is not bool:
            raise ValueError("invalid_carry_slip_guard: carry_slip_guard must be a boolean")
    else:
        enabled = False
    has_scope = "carry_slip_scope" in mapping
    has_steps = "slip_consecutive_steps" in mapping
    scope = mapping.get("carry_slip_scope", "next_move")
    if has_scope:
        if not isinstance(scope, str) or scope not in CARRY_SLIP_SCOPES:
            raise ValueError(
                f"invalid_carry_slip_scope: carry_slip_scope must be one of {CARRY_SLIP_SCOPES}"
            )
    steps = mapping.get("slip_consecutive_steps", 1)
    if has_steps:
        lo, hi = CARRY_SLIP_CONSECUTIVE_STEPS_BOUNDS
        if type(steps) is not int or not lo <= steps <= hi:
            raise ValueError(
                f"invalid_carry_slip_consecutive_steps: slip_consecutive_steps must be an integer in [{lo}, {hi}]"
            )
    if not enabled and (has_scope or has_steps):
        raise ValueError(
            "invalid_carry_slip_guard_disabled: carry_slip_scope and slip_consecutive_steps require carry_slip_guard=True"
        )
    return enabled, (scope if enabled else "next_move"), (steps if enabled else 1)


def validate_grip_guard_fields(mapping, *, profile=None, empty_close_aperture_m=None, max_aperture_m=None):
    """校验阶段或批次上的 grip_guard 参数合法性。

    ``profile``: the run's ``EmbodimentProfile`` (None = the default robot, ARX-X5): accepted presets and their
    margins, the zero point and the full opening. ``empty_close_aperture_m`` (m) overrides the profile's
    empty-close gap, the zero point of presets (ARX-X5 0.0). ``max_aperture_m`` (m) overrides the profile's full
    opening, which bounds explicit ``min_aperture_m`` / ``max_aperture_m`` values (absolute fingertip gaps in
    [0, max]; ARX-X5 0.088). Returns ``{"grip_guard_preset", "min_aperture_m", "max_aperture_m"}`` (m or None)
    when no ``CARRY_SLIP_GUARD_FIELDS`` are present, or additionally ``carry_slip_guard`` (bool),
    ``carry_slip_scope`` (str), and ``slip_consecutive_steps`` (int) when any ``CARRY_SLIP_GUARD_FIELDS`` are given.
    """
    if not isinstance(mapping, dict):
        raise ValueError("invalid_grip_guard_mapping")
    margins, _, maximum = guard_gripper(profile, empty_close_aperture_m=empty_close_aperture_m,
                                        max_aperture_m=max_aperture_m)
    preset = mapping.get("grip_guard_preset")
    if preset is not None:
        if not isinstance(preset, str) or preset not in margins:
            raise ValueError("invalid_grip_guard_preset")
    for field in ("min_aperture_m", "max_aperture_m"):
        if field in mapping and mapping[field] is not None:
            value = mapping[field]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0.0 <= float(value) <= maximum:
                raise ValueError(f"invalid_{field}")
    base_min, base_max = preset_aperture_bounds(preset, profile=profile, empty_close_aperture_m=empty_close_aperture_m)
    min_ap = float(mapping["min_aperture_m"]) if mapping.get("min_aperture_m") is not None else base_min
    max_ap = float(mapping["max_aperture_m"]) if mapping.get("max_aperture_m") is not None else base_max
    if min_ap is not None and max_ap is not None and min_ap > max_ap:
        raise ValueError("invalid_grip_guard_aperture_bounds")
    out = {"grip_guard_preset": preset, "min_aperture_m": min_ap, "max_aperture_m": max_ap}
    if not _has_carry_slip_fields(mapping):
        return out
    cs_enabled, cs_scope, cs_steps = validate_carry_slip_syntax(mapping)
    if cs_enabled and min_ap is None:
        raise ValueError(
            "invalid_carry_slip_requires_min_aperture: carry_slip_guard=True requires a non-null min_aperture_m "
            "(from grip_guard_preset or explicit min_aperture_m > 0 m)"
        )
    out.update(
        carry_slip_guard=cs_enabled,
        carry_slip_scope=cs_scope,
        slip_consecutive_steps=cs_steps,
    )
    return out


def resolve_grip_guard(stage, batch_defaults=None, **gripper):
    """合并批次默认与阶段显式 grip_guard 配置，返回实际生效的上下界。

    ``gripper``: optional ``profile`` (the run's ``EmbodimentProfile``) and ``empty_close_aperture_m`` /
    ``max_aperture_m`` (m) of the guarded gripper, forwarded to ``validate_grip_guard_fields`` (defaults: the
    default robot, ARX-X5).
    """
    stage = stage if isinstance(stage, dict) else {}
    batch_defaults = batch_defaults if isinstance(batch_defaults, dict) else {}
    has_stage_cs = _has_carry_slip_fields(stage)
    has_batch_cs = _has_carry_slip_fields(batch_defaults)
    has_stage = any(stage.get(k) is not None for k in GRIP_GUARD_FIELDS)
    has_batch = any(batch_defaults.get(k) is not None for k in GRIP_GUARD_FIELDS)
    if not has_stage_cs and not has_batch_cs:
        if not has_stage and not has_batch:
            return None
        source = stage if has_stage else batch_defaults
        validated = validate_grip_guard_fields(source, **gripper)
        if validated["min_aperture_m"] is None and validated["max_aperture_m"] is None and validated["grip_guard_preset"] in (None, "none"):
            return None
        return validated
    if has_batch_cs:
        validate_carry_slip_syntax(batch_defaults)
    if has_stage_cs and "carry_slip_guard" in stage:
        validate_carry_slip_syntax(stage)
    aperture_source = stage if has_stage else batch_defaults
    merged = {k: aperture_source[k] for k in GRIP_GUARD_FIELDS if k in aperture_source}
    cs_merged = {k: batch_defaults[k] for k in CARRY_SLIP_GUARD_FIELDS if k in batch_defaults}
    if stage.get("carry_slip_guard") is False:
        cs_merged = {"carry_slip_guard": False}
    else:
        for k in CARRY_SLIP_GUARD_FIELDS:
            if k in stage:
                cs_merged[k] = stage[k]
    merged.update(cs_merged)
    validated = validate_grip_guard_fields(merged, **gripper)
    if (
        validated["min_aperture_m"] is None
        and validated["max_aperture_m"] is None
        and validated["grip_guard_preset"] in (None, "none")
        and not validated.get("carry_slip_guard", False)
    ):
        return None
    return validated


def evaluate_grip_guard(grip_facts, guard_config, **gripper):
    """根据计算出的 grip_facts 与门禁上下界判断是否触发暂停门禁。

    ``gripper``: the same optional ``profile`` / ``empty_close_aperture_m`` / ``max_aperture_m`` (m) used to
    resolve ``guard_config`` (defaults: the default robot, ARX-X5).
    """
    if not guard_config:
        return {"triggered": False, "reason": None, "message": None}
    if any(k in guard_config for k in GRIP_GUARD_FIELDS) or _has_carry_slip_fields(guard_config):
        resolved = validate_grip_guard_fields(guard_config, **gripper)
    else:
        resolved = guard_config
    min_ap = resolved.get("min_aperture_m")
    max_ap = resolved.get("max_aperture_m")
    measured = grip_facts.get("measured_aperture_m") if isinstance(grip_facts, dict) else None
    if measured is None or (min_ap is None and max_ap is None):
        return {
            "triggered": False,
            "reason": None,
            "message": None,
            "min_aperture_m": min_ap,
            "max_aperture_m": max_ap,
            "measured_aperture_m": measured,
        }
    if min_ap is not None and measured < min_ap - 1e-9:
        reason = "aperture_below_min_m"
        message = (
            f"Grip guard triggered ({reason}): measured_aperture_m={measured:.4f}m "
            f"< min_aperture_m={min_ap:.4f}m."
        )
        return {
            "triggered": True,
            "reason": reason,
            "message": message,
            "min_aperture_m": min_ap,
            "max_aperture_m": max_ap,
            "measured_aperture_m": measured,
        }
    if max_ap is not None and measured > max_ap + 1e-9:
        reason = "aperture_above_max_m"
        message = (
            f"Grip guard triggered ({reason}): measured_aperture_m={measured:.4f}m "
            f"> max_aperture_m={max_ap:.4f}m."
        )
        return {
            "triggered": True,
            "reason": reason,
            "message": message,
            "min_aperture_m": min_ap,
            "max_aperture_m": max_ap,
            "measured_aperture_m": measured,
        }
    return {
        "triggered": False,
        "reason": None,
        "message": None,
        "min_aperture_m": min_ap,
        "max_aperture_m": max_ap,
        "measured_aperture_m": measured,
    }


# --------------------------------------------------------------------------------------------
# Batch rules shared by every batch executor (ARX-X5 batch_motion, SO-101 worker). Each number
# below is part of the public grip-guard contract and is described to the model by the backend.
# --------------------------------------------------------------------------------------------
BATCH_DEFAULT_JAW_GOAL_BELOW = 0.85  # goal_normalized: batch-level guard fields apply to jaw stages below it
CLOSING_JAW_GOAL_BELOW = 0.5  # goal_normalized: a jaw command below it arms the post-grasp re-check; at/above disarms
JAW_NOT_CLOSED_DELTA = -1e-3  # normalized: a measured jaw change at/above it means the stage did not close the jaw
SLIP_MIN_JAW_DROP = 0.08  # normalized: post-grasp re-check needs the jaw to close at least this much during the move
SLIP_MAX_GAP_ABOVE_EMPTY_CLOSE_M = 0.002  # m: ... and to end within this gap of the empty-close zero point
DOWNWARD_MOVE_DZ_M = -0.005  # m: moves whose target is this much below their measured start skip the re-check
JAW_CHECKPOINT_REASON = (
    'Jaw stage completed before dependent move (pre-lift); inspect finger contact alignment in RGB before continuing, '
    'or verify object co-motion after a lift stage.')


def jaw_goal(action):
    """First jaw event goal_normalized of one compiled stage action (missing goal = 1.0), or None."""
    events = (action or {}).get('gripper_events') or []
    return float(events[0].get('goal_normalized', 1.0)) if events else None


def stage_grip_guard(stage, batch_action, goal, **gripper):
    """Preview-time guard of one stage: stage fields win; batch fields apply to jaw goals below 0.85."""
    closing = goal is not None and goal < BATCH_DEFAULT_JAW_GOAL_BELOW
    return resolve_grip_guard(stage, batch_action if closing else None, **gripper)


def after_stage_grip_guard(armed, arm, kind, goal, guard, grip_facts, *, has_remaining_stages, has_arm_remaining,
                           is_downward_move, post_grasp_recheck=True, **gripper):
    """Shared post-stage grip-guard rule. Returns ``(evaluation or None, triggered)``.

    ``armed``: per-arm dict of guards armed by a passing closing stage (mutated; persists
    across checkpoint resumes). ``kind``: ``move`` or ``gripper``; ``goal``: the stage jaw
    goal_normalized or None; ``guard``: the stage's resolved guard or None; ``grip_facts``:
    measured facts across the stage. A jaw stage with a guard is evaluated unless the jaw did
    not close (``jaw_delta >= -0.001``) or nothing follows it. A passing closing stage arms a
    single re-check on the next same-arm move without a jaw command (not on moves going down
    more than 5 mm): it pauses only when the jaw closed by at least 0.08 normalized during
    that move and ends within 2 mm of the empty-close gap, i.e. the object slipped out. The
    re-check is consumed by that first move whatever its outcome. ``post_grasp_recheck=False``
    (a gripper whose empty-close gap is not measured, so "within 2 mm of it" is undefined) never
    arms it; ARX-X5 keeps the default True with its 0.0 m zero point. ``gripper``: optional ``profile`` /
    ``empty_close_aperture_m`` / ``max_aperture_m`` (m), as for ``validate_grip_guard_fields``; a zero point that is
    not measured (None) never fires the re-check.
    """
    closing = goal is not None and goal < CLOSING_JAW_GOAL_BELOW
    if goal is not None and not closing:
        armed.pop(arm, None)
    if guard:
        jaw_delta = grip_facts.get('jaw_delta')
        evaluation = evaluate_grip_guard(grip_facts, guard, **gripper)
        if not has_remaining_stages or (jaw_delta is not None and jaw_delta >= JAW_NOT_CLOSED_DELTA):
            evaluation = dict(evaluation, triggered=False, reason=None, message=None)
        if evaluation['triggered']:
            armed.pop(arm, None)
        elif closing and post_grasp_recheck:
            armed[arm] = deepcopy(guard)
        return evaluation, evaluation['triggered']
    if kind == 'move' and goal is None:
        pending = armed.pop(arm, None)
        jaw_delta = grip_facts.get('jaw_delta') or 0.0
        after = grip_facts.get('measured_aperture_m')
        _, zero, _ = guard_gripper(gripper.get('profile'), empty_close_aperture_m=gripper.get('empty_close_aperture_m'))
        if (pending and zero is not None and has_arm_remaining and not is_downward_move
                and jaw_delta <= -SLIP_MIN_JAW_DROP and after is not None
                and after <= float(zero) + SLIP_MAX_GAP_ABOVE_EMPTY_CLOSE_M):
            evaluation = evaluate_grip_guard(grip_facts, pending, **gripper)
            if evaluation['triggered'] and evaluation['reason'] == 'aperture_below_min_m':
                return evaluation, True
    return None, False


def jaw_checkpoints(records, stages, dependencies, checkpoints, batch_checkpoint, previous_goal):
    """Add ``checkpoint_after_gripper`` pause markers; returns ``checkpoints`` sorted by cursor.

    ``records``: stage id -> ``{kind, arm, start, end, action, checkpoint_after_gripper}`` with
    native-step cursors; ``stages``: the batch stages in request order; ``dependencies``: stage
    id -> set of prerequisite ids (declared plus same-arm order); ``checkpoints``: existing
    markers ``{cursor, stage_ids, reasons}`` (observe stages); ``batch_checkpoint``: the batch
    flag; ``previous_goal(record)``: commanded jaw goal_normalized just before that stage.
    The last jaw stage of a same-arm window gets a marker at the start of its first dependent
    move when it or an earlier jaw stage of the window set the flag (any jaw change), or the
    batch flag is set and the stage closes the jaw. A stage flag False opts out; an explicit
    observe stage already inside the window replaces the marker.
    """
    for identifier, record in records.items():
        events = record.get('action', {}).get('gripper_events', [])
        if not events:
            continue
        # 同一手臂在进入下一个 move 或显式 observe 前若还有后续纯夹爪阶段，由最后一个纯夹爪阶段代表该窗口
        later_same_arm_jaw = any(
            other_id != identifier and other['kind'] == 'gripper' and other['arm'] == record['arm']
            and other['start'] >= record['end']
            and other.get('action', {}).get('gripper_events')
            and not any(
                mid['start'] >= record['end'] and mid['end'] <= other['start']
                and (mid['kind'] == 'observe' or (mid['kind'] == 'move' and mid['arm'] == record['arm']))
                for mid in records.values()
            )
            for other_id, other in records.items()
        )
        if later_same_arm_jaw:
            continue
        window_records = [
            other for other in records.values()
            if other['arm'] == record['arm'] and other['end'] <= record['end']
            and other.get('action', {}).get('gripper_events')
            and not any(
                mid['start'] >= other['end'] and mid['end'] <= record['start']
                and (mid['kind'] == 'observe' or (mid['kind'] == 'move' and mid['arm'] == record['arm']))
                for mid in records.values()
            )
        ]
        if record.get('checkpoint_after_gripper') is False:
            continue
        stage_explicit = any(item.get('checkpoint_after_gripper') is True for item in window_records)
        if not (stage_explicit or batch_checkpoint):
            continue
        dependent_move_starts = [
            records[stage['id']]['start'] for stage in stages
            if stage['kind'] == 'move' and identifier in dependencies[stage['id']]
        ]
        if not dependent_move_starts:
            continue
        previous = previous_goal(record)
        goal = events[0]['goal_normalized']
        if goal == previous or (not stage_explicit and goal >= previous - 1e-4):
            continue
        first_move_start = min(dependent_move_starts)
        # 若窗口 [record['end'], first_move_start] 内已有显式 observe 阶段，不再重复插入自动检查点
        if any(
            record['end'] <= item['cursor'] <= first_move_start
            and any(s['id'] in item['stage_ids'] and s['kind'] == 'observe' for s in stages)
            for item in checkpoints
        ):
            continue
        marker = next((item for item in checkpoints if item['cursor'] == first_move_start), None)
        if marker is None:
            marker = dict(cursor=first_move_start, stage_ids=[], reasons={})
            checkpoints.append(marker)
        if identifier not in marker['stage_ids']:
            marker['stage_ids'].append(identifier)
        marker['reasons'][identifier] = JAW_CHECKPOINT_REASON
    checkpoints.sort(key=lambda item: item['cursor'])
    return checkpoints


def measured_aperture_m(state, adapter=None, *, arm=None):
    """Return live measured fingertip aperture in ``m`` (or ``None`` if unmeasured) for ``arm``."""
    jaw = _extract_jaw(state, adapter=adapter, arm=arm)
    if jaw is None:
        return None
    return _aperture_from_jaw(jaw, adapter=adapter, arm=arm)


def format_effective_grip_guard(guard):
    """Return canonical model-visible ``effective_grip_guard`` dict for a configured/active carry-slip guard.

    Fields:
        - ``grip_guard_preset`` (``str | None``): preset name or ``None``.
        - ``min_aperture_m`` (``float | None``, unit: ``m``): lower fingertip gap threshold.
        - ``max_aperture_m`` (``float | None``, unit: ``m``): upper fingertip gap threshold.
        - ``carry_slip_guard`` (``bool``): whether per-step slip monitoring is enabled.
        - ``carry_slip_scope`` (``"next_move" | "until_release"``): stage coverage scope.
        - ``slip_consecutive_steps`` (``int``, unit: steps/ticks): required consecutive below-min samples.
    """
    return {
        "grip_guard_preset": guard.get("grip_guard_preset"),
        "min_aperture_m": guard.get("min_aperture_m"),
        "max_aperture_m": guard.get("max_aperture_m"),
        "carry_slip_guard": bool(guard.get("carry_slip_guard", False)),
        "carry_slip_scope": str(guard.get("carry_slip_scope") or "next_move"),
        "slip_consecutive_steps": int(guard.get("slip_consecutive_steps") or 1),
    }


def validate_batch_carry_slip_guard(action, **gripper):
    """Validate ``CARRY_SLIP_GUARD_FIELDS`` across batch defaults and all stages of a ``BatchAction``.

    Raises ``ValueError`` with ``batch_observe_requires_reason`` on ``observe`` stages carrying
    ``CARRY_SLIP_GUARD_FIELDS``, and ``invalid_carry_slip_*`` on non-jaw stages, invalid types/bounds,
    disabled ``carry_slip_guard`` with scope/steps, or ``carry_slip_guard=True`` with ``min_aperture_m=None``.
    """
    if not isinstance(action, dict):
        return
    has_batch_cs = _has_carry_slip_fields(action)
    batch_cs_enabled = False
    batch_min_ap = None
    if has_batch_cs:
        batch_cs_enabled, _, _ = validate_carry_slip_syntax(action)
        batch_grip_only = {k: action[k] for k in GRIP_GUARD_FIELDS if k in action}
        batch_resolved = validate_grip_guard_fields(batch_grip_only, **gripper)
        batch_min_ap = batch_resolved["min_aperture_m"]
    stages = action.get("stages") or []
    armed_closing_count = 0
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        kind = stage.get("kind")
        has_stage_cs = _has_carry_slip_fields(stage)
        if kind == "observe":
            if has_stage_cs:
                raise ValueError("batch_observe_requires_reason")
            continue
        own = stage.get("action") if isinstance(stage.get("action"), dict) else {}
        events = own.get("gripper_events") or []
        if kind == "move" and not events:
            if has_stage_cs:
                validate_carry_slip_syntax(stage)
                if stage.get("carry_slip_guard") is True:
                    raise ValueError(
                        "invalid_carry_slip_requires_jaw_stage: carry_slip_guard applies to closing jaw stages"
                    )
            continue
        goal = jaw_goal(own)
        if has_stage_cs or has_batch_cs:
            resolved = stage_grip_guard(stage, action, goal, **gripper)
            if (
                isinstance(resolved, dict)
                and resolved.get("carry_slip_guard") is True
                and goal is not None
                and goal < CLOSING_JAW_GOAL_BELOW
                and resolved.get("min_aperture_m") is not None
            ):
                armed_closing_count += 1
    if batch_cs_enabled and batch_min_ap is None and armed_closing_count == 0:
        raise ValueError(
            "invalid_carry_slip_requires_min_aperture: carry_slip_guard=True requires a non-null min_aperture_m"
        )


class CarrySlipMonitor:
    """Stateful per-arm fingertip aperture monitor for opt-in per-step ``carry_slip_guard``.

    Armed after a closing jaw stage (``goal_normalized < 0.5``) passes its post-stage ``grip_guard``
    with ``carry_slip_guard=True`` and ``min_aperture_m > 0`` (unit: ``m``). Checks live measured
    fingertip aperture on every native physics step (RoboDojo) or control tick (SO-101) of covered
    same-arm ``move`` stages (``scope="next_move"`` or ``"until_release"``), pausing immediately
    when ``measured_aperture_m < min_aperture_m - 1e-9`` for ``slip_consecutive_steps`` consecutive
    steps/ticks.

    Example:
        >>> mon = CarrySlipMonitor("left", 0.015, scope="next_move", slip_consecutive_steps=2)
        >>> mon.observe_step("lift", 0, 0.002) is None
        True
        >>> hit = mon.observe_step("lift", 1, 0.002)
        >>> hit["reason"]
        'carry_slip_aperture_below_min_m'
    """

    def __init__(
        self,
        arm: str,
        min_aperture_m: float,
        *,
        scope: str = "next_move",
        slip_consecutive_steps: int = 1,
        source_stage_id: str | None = None,
        effective_grip_guard: dict | None = None,
    ) -> None:
        if scope not in CARRY_SLIP_SCOPES:
            raise ValueError(f"invalid_carry_slip_scope:{scope}")
        lo, hi = CARRY_SLIP_CONSECUTIVE_STEPS_BOUNDS
        if type(slip_consecutive_steps) is not int or not lo <= slip_consecutive_steps <= hi:
            raise ValueError(f"invalid_carry_slip_consecutive_steps:{slip_consecutive_steps}")
        if type(min_aperture_m) not in (int, float) or not math.isfinite(min_aperture_m) or float(min_aperture_m) <= 0.0:
            raise ValueError(f"invalid_carry_slip_requires_min_aperture:{min_aperture_m}")
        self.arm = str(arm)
        self.min_aperture_m = float(min_aperture_m)
        self.scope = str(scope)
        self.slip_consecutive_steps = int(slip_consecutive_steps)
        self.source_stage_id = source_stage_id
        self.streak = 0
        base_guard = effective_grip_guard or {
            "grip_guard_preset": None,
            "min_aperture_m": self.min_aperture_m,
            "max_aperture_m": None,
            "carry_slip_guard": True,
            "carry_slip_scope": self.scope,
            "slip_consecutive_steps": self.slip_consecutive_steps,
        }
        self.effective_grip_guard = format_effective_grip_guard(base_guard)

    def reset_streak(self) -> None:
        """Reset consecutive below-minimum step counter (e.g., at the start of a new move stage)."""
        self.streak = 0

    def observe_step(self, stage_id: str, step_index: int, aperture_m: float | None) -> dict | None:
        """Observe one step's measured fingertip aperture (``m``); return trigger dict when streak reaches threshold."""
        if aperture_m is None or not math.isfinite(float(aperture_m)):
            return None
        ap = float(aperture_m)
        if ap < self.min_aperture_m - 1e-9:
            self.streak += 1
        else:
            self.streak = 0
        if self.streak < self.slip_consecutive_steps:
            return None
        slip_at = {
            "stage_id": str(stage_id),
            "arm": self.arm,
            "step_index": int(step_index),
            "measured_aperture_m": round(ap, 6),
            "min_aperture_m": round(self.min_aperture_m, 6),
            "consecutive_steps": int(self.streak),
        }
        message = (
            f"Carry slip guard triggered ({CARRY_SLIP_REASON}) on arm '{self.arm}' at stage '{stage_id}' "
            f"step {int(step_index)}: measured_aperture_m={ap:.4f}m < min_aperture_m={self.min_aperture_m:.4f}m "
            f"for {self.streak} consecutive step(s)."
        )
        return {
            "triggered": True,
            "reason": CARRY_SLIP_REASON,
            "message": message,
            "effective_grip_guard": deepcopy(self.effective_grip_guard),
            "slip_detected_at": slip_at,
        }
