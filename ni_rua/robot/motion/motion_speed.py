"""Canonical speed selection and bounded joint-path resampling."""

import math

import numpy as np

from ...core.action_validation import (arm_action, dual_path_shape, embodiment_or_default, validate_action_speed,
                                      validate_arm_options)
# The shared named multipliers, defined once in embodiments; a run accepts only those inside its embodiment's
# speed_scale / gripper_speed_scale range (``profile.speed_presets`` / ``profile.gripper_speed_presets``).
from ..model.embodiments import GRIPPER_SPEED_PRESETS, SPEED_PRESETS


PATH_TOLERANCE_RAD = .001
ARM_PRESETS = SPEED_PRESETS
ARM_SPEED_INPUT_FIELDS = ("speed_preset", "speed_scale", "gripper_speed_preset", "gripper_speed_scale",
                          "waypoint_speed_presets", "waypoint_speed_scales")
SPEED_INPUT_FIELDS = (*ARM_SPEED_INPUT_FIELDS, "arm_options")
ARM_SPEED_EFFECTIVE_FIELDS = ("speed_scale", "gripper_speed_scale", "waypoint_speed_scales")
SPEED_EFFECTIVE_FIELDS = (*ARM_SPEED_EFFECTIVE_FIELDS, "arm_options")


def resolve_action_speed(action, default_arm_scale=1.0, *, profile=None):
    """Return a shallow copy with fixed presets/defaults resolved to floats.

    Call before storing/hashing an action and before preview compilation. The
    worker wire contract accepts only the resulting numeric fields. Nested
    values are preserved and never mutated. No selection provenance is added.
    Defaults apply only to a backend_free path (arm) or actual gripper events
    (jaw); waits and policy prefixes are not retimed. A preset has a fixed
    absolute value, independent of the configured omitted-arm default.

    ``profile``: the run's ``EmbodimentProfile`` (None = the default robot, ARX-X5). Accepted presets are
    ``profile.speed_presets`` / ``profile.gripper_speed_presets`` (the shared presets inside the embodiment's
    ranges); ``default_arm_scale`` (the run's ``features.default_motion_speed_scale``, dimensionless) must lie in
    ``profile.speed_scale`` (ARX-X5 [0.25, 2.0]).
    """

    profile = embodiment_or_default(profile)
    if not isinstance(action, dict):
        raise ValueError("invalid_action_speed_action: expected object")
    scale = profile.speed_scale
    if (type(default_arm_scale) not in (int, float) or not math.isfinite(default_arm_scale)
            or not scale.min <= default_arm_scale <= scale.max):
        raise ValueError(f"invalid_default_motion_speed_scale: expected finite number in [{scale.min}, {scale.max}]")
    arm_presets, jaw_presets = profile.speed_presets, profile.gripper_speed_presets
    result = dict(action)
    raw_arm_options = validate_arm_options(action, ARM_SPEED_INPUT_FIELDS) if "arm_options" in action else None
    result.pop("arm_options", None)
    selections = (("speed_preset", "speed_scale", arm_presets),
                  ("gripper_speed_preset", "gripper_speed_scale", jaw_presets))
    fields = set(SPEED_INPUT_FIELDS)
    if result.get("type") == "JointPrefixAction" or "hold_native_steps" in result:
        if fields.intersection(action):
            raise ValueError("action_speed_unsupported_for_wait_or_joint_prefix")
        return result
    for preset, numeric, choices in selections:
        if preset not in result:
            continue
        if numeric in result:
            raise ValueError(f"conflicting_{preset}_and_{numeric}")
        choice = result.pop(preset)
        if not isinstance(choice, str) or choice not in choices:
            raise ValueError(f"invalid_{preset}: expected one of {', '.join(choices)}")
        result[numeric] = choices[choice]
    path = result.get("path")
    if isinstance(path, dict) and path.get("kind") == "backend_free":
        result.setdefault("speed_scale", default_arm_scale)
    waypoint_fields = {"waypoint_speed_presets", "waypoint_speed_scales"}.intersection(result)
    if waypoint_fields:
        if len(waypoint_fields) != 1:
            raise ValueError("conflicting_waypoint_speed_presets_and_waypoint_speed_scales")
        if not isinstance(path, dict) or path.get("kind") != "backend_free":
            raise ValueError("waypoint_speed_scales_requires_motion_path")
        name = next(iter(waypoint_fields))
        values = result.pop(name)
        points = path.get("waypoints", [])
        if not isinstance(points, list) or not isinstance(values, list) or not values or len(values) != len(points):
            raise ValueError(f"invalid_{name}: expected one entry per waypoint including final target")
        resolved = []
        for value in values:
            if value is None:
                value = result["speed_scale"]
            elif name == "waypoint_speed_presets":
                if not isinstance(value, str) or value not in arm_presets:
                    raise ValueError("invalid_waypoint_speed_presets: expected " + ", ".join(arm_presets) + " or null")
                value = arm_presets[value]
            resolved.append(value)
        result["waypoint_speed_scales"] = resolved
    if result.get("gripper_events"):
        result.setdefault("gripper_speed_scale", 1.)
    validate_action_speed(result, profile=profile)
    for name in ("speed_scale", "gripper_speed_scale"):
        if name in result:
            result[name] = float(result[name])
    if "waypoint_speed_scales" in result:
        result["waypoint_speed_scales"] = [float(value) for value in result["waypoint_speed_scales"]]
    if dual_path_shape(result):
        # Preserve null entries until each arm's request speed is known. A
        # shared [null, "slow"] means that arm's transit speed, then slow.
        source = {**action, "arm_options": raw_arm_options or {}}
        result["arm_options"] = {}
        for arm in ("left", "right"):
            selected = resolve_action_speed(arm_action(source, arm), default_arm_scale, profile=profile)
            result["arm_options"][arm] = {key: selected[key] for key in ARM_SPEED_EFFECTIVE_FIELDS if key in selected}
        validate_action_speed(result, profile=profile)
    return result


def bounded_resample(
    positions,
    step_limits,
    *,
    tolerance=PATH_TOLERANCE_RAD,
    speed_scale=1.0,
    native_step_limits=None,
    max_acceleration_rad_s2=None,
    max_jerk_rad_s3=None,
    rate_hz=None,
):
    """Keep significant polyline turns, then bound every commanded joint delta.

    Simplification compares each original knot with interpolation at the same
    cumulative joint-space arc length. Thus reversals and nonuniform timing are
    not mistaken for a straight path. Every retained knot is emitted exactly;
    removed knots differ by at most ``tolerance`` per joint. When
    ``max_acceleration_rad_s2``, ``max_jerk_rad_s3``, and ``rate_hz`` are set,
    each retained knot segment is timed along a 1-DoF ``ruckig`` 3rd-order
    jerk-limited S-curve ``s in [0, 1]`` while still respecting ``step_limits``.
    This is a geometric/kinematic trajectory parameterization, not a new
    collision check or a physical velocity guarantee.
    """
    positions = np.asarray(positions, dtype=float)
    limits = np.asarray(step_limits, dtype=float)
    if (type(speed_scale) not in (int, float) or not math.isfinite(speed_scale) or speed_scale <= 0):
        raise ValueError("invalid_speed_resampling_inputs")
    limits = limits * float(speed_scale)
    if native_step_limits is not None:
        native = np.asarray(native_step_limits, dtype=float)
        if native.shape != limits.shape or not np.isfinite(native).all() or (native <= 0).any():
            raise ValueError("invalid_speed_resampling_inputs")
        limits = np.minimum(native, limits)
    if (positions.ndim != 2 or not len(positions) or limits.shape != positions.shape[1:]
            or not np.isfinite(positions).all() or not np.isfinite(limits).all()
            or (limits <= 0).any()):
        raise ValueError("invalid_speed_resampling_inputs")
    use_ruckig = max_acceleration_rad_s2 is not None or max_jerk_rad_s3 is not None
    if use_ruckig:
        if max_acceleration_rad_s2 is None or max_jerk_rad_s3 is None or rate_hz is None:
            raise ValueError("invalid_speed_resampling_inputs")
        if type(rate_hz) not in (int, float) or not math.isfinite(rate_hz) or rate_hz <= 0:
            raise ValueError("invalid_speed_resampling_inputs")
        rate = float(rate_hz)
        try:
            accel = np.broadcast_to(np.asarray(max_acceleration_rad_s2, dtype=float), limits.shape)
            jerk = np.broadcast_to(np.asarray(max_jerk_rad_s3, dtype=float), limits.shape)
        except ValueError as error:
            raise ValueError("invalid_speed_resampling_inputs") from error
        if (not np.isfinite(accel).all() or (accel <= 0).any()
                or not np.isfinite(jerk).all() or (jerk <= 0).any()):
            raise ValueError("invalid_speed_resampling_inputs")
    if len(positions) == 1:
        return positions.copy()
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
    kept = {0, len(positions) - 1}
    pending = [(0, len(positions) - 1)]
    while pending:
        first, last = pending.pop()
        if last <= first + 1 or distance[last] == distance[first]:
            continue
        fraction = (distance[first + 1:last] - distance[first]) / (distance[last] - distance[first])
        chord = positions[first] + fraction[:, None] * (positions[last] - positions[first])
        deviations = np.max(np.abs(positions[first + 1:last] - chord), axis=1)
        index = int(np.argmax(deviations))
        if deviations[index] > tolerance:
            middle = first + 1 + index
            kept.add(middle)
            pending.extend(((first, middle), (middle, last)))
    knots = positions[sorted(kept)]
    rows = []
    if use_ruckig:
        import ruckig

        otg = ruckig.Ruckig(1)
        for start, end in zip(knots, knots[1:]):
            delta = end - start
            abs_delta = np.abs(delta)
            if float(abs_delta.max()) <= 1e-12:
                rows.append(end[None, :].copy())
                continue
            active = abs_delta > 1e-12
            inp = ruckig.InputParameter(1)
            inp.current_position = [0.0]
            inp.current_velocity = [0.0]
            inp.current_acceleration = [0.0]
            inp.target_position = [1.0]
            inp.target_velocity = [0.0]
            inp.target_acceleration = [0.0]
            inp.max_velocity = [float(np.min((limits[active] * rate) / abs_delta[active]))]
            inp.max_acceleration = [float(np.min(accel[active] / abs_delta[active]))]
            inp.max_jerk = [float(np.min(jerk[active] / abs_delta[active]))]
            traj = ruckig.Trajectory(1)
            res = otg.calculate(inp, traj)
            if res not in (ruckig.Result.Working, ruckig.Result.Finished):
                raise ValueError(f"ruckig_trajectory_failed:{res}")
            duration = float(traj.duration)
            count = max(1, int(math.ceil(duration * rate)))
            s_vals = np.empty(count, dtype=float)
            for k in range(1, count):
                pos, _, _ = traj.at_time(min(k / rate, duration))
                s_vals[k - 1] = float(np.clip(pos[0], 0.0, 1.0))
            s_vals[-1] = 1.0
            segment = start + s_vals[:, None] * delta
            segment[-1] = end
            rows.append(segment)
        return np.vstack(rows)
    for start, end in zip(knots, knots[1:]):
        count = max(1, int(np.ceil(np.max(np.abs(end - start) / limits))))
        segment = start + np.arange(1, count + 1)[:, None] / count * (end - start)
        segment[-1] = end
        rows.append(segment)
    return np.vstack(rows)
