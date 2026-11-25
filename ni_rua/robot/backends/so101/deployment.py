"""SO-101 deployment file: machine bindings and every motion/telemetry threshold.

The file (``configs/so101.local.json``, git-ignored; template
``configs/so101.example.json``) is the only source of tuning numbers for the
``so101_real`` backend. Numeric fields in ``FIELDS`` carry physical units and
bounds and are validated at load time. Existing fields are required; explicitly
declared defaults for added trajectory controls keep older deployments loadable
with sliding disabled. The optional sections ``ik_service`` and ``perception``
(``validate_perception``) may be absent. Supplied values are never overwritten.
Keys beginning with ``_`` are comments and ignored.
No credentials belong in this file.

Pure Python (no numpy, LeRobot or serial imports) so the parent process can
validate the binding before launching the worker interpreter. ``embodiment_profile``
turns a validated deployment into the run's ``EmbodimentProfile`` (it imports the
CAD geometry, and with it numpy, only when called).
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import uuid

from ....core.config import motion_thresholds
from ...model.embodiments import (MAX_ARRIVAL_CORRECTION_OFFSET_RAD, MAX_ARRIVAL_CORRECTION_ROUNDS,
                                 grip_guard_preset_margins, tolerance_presets)
from ni_rua.robot.calibration.workflow.defaults import calibration_default

SCHEMA = "ni_rua.so101.deployment.v1"
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
CAMERA_IDS = ("wrist", "fixed")
_UNSET = object()
# robot_action limits profile of this backend: every bound comes from this deployment file.
LIMITS_PROFILE = "so101-deployment-v1"
# sim_step counter domain of SO-101 observations: dispatched joint-goal commands, not time.
STEP_DOMAIN = "so101_control_step"
# robot_action presets the SO-101 offers (shared action-language names); their numbers are deployment fields.
GRIP_GUARD_PRESET_NAMES = ("non_empty", "thin_object", "rigid_object", "wide_object")
TOLERANCE_PRESET_NAMES = ("tight", "normal", "relaxed", "compliant", "coarse", "unconstrained_orientation")
# Largest explicit TCP position tolerance (m) and the speed multiplier ranges (dimensionless) of this backend:
# speed_scale 1 is motion.max_joint_speed_rad_s times the operator speed_scale_cap, never faster; 0.25 is the
# shared action-language minimum.
MAX_POSITION_TOLERANCE_M = 0.1
SPEED_SCALE_RANGE = (0.25, 1.0)
GRIPPER_SPEED_SCALE_RANGE = (0.25, 1.0)
# Shared defaults of the preset tables (code-default thresholds; features.* are not read for SO-101, decision Q4).
_SHARED_THRESHOLDS = motion_thresholds()
_DEFAULT_PRESET_MARGINS_M = {name: list(grip_guard_preset_margins(_SHARED_THRESHOLDS)[name])
                             for name in GRIP_GUARD_PRESET_NAMES}
_DEFAULT_TOLERANCE_PRESETS = {name: [tolerance.position_m, tolerance.orientation_rad]
                              for name, tolerance in tolerance_presets(_SHARED_THRESHOLDS).items()
                              if name in TOLERANCE_PRESET_NAMES}


def _num(unit, low, high, doc, *, integer=False, low_open=False, high_open=False, default=_UNSET, nullable=False):
    spec = dict(kind="int" if integer else "num", unit=unit, low=low, high=high, low_open=low_open, doc=doc)
    if default is not _UNSET:
        spec["default"] = default
    if nullable:
        spec["nullable"] = True
    if high_open:
        spec["high_open"] = True
    return spec


def _vec(size, unit, doc):
    return dict(kind="vec", size=size, unit=unit, doc=doc)


def _list(max_size, unit, low, high, doc):
    """1..max_size finite numbers, each in [low, high]."""
    return dict(kind="list", max_size=max_size, unit=unit, low=low, high=high, doc=doc)


def _xy(max_size, unit, low, high, doc):
    """1..max_size points ``[x, y]``, each coordinate finite and in [low, high]."""
    return dict(kind="xy", max_size=max_size, unit=unit, low=low, high=high, doc=doc)


def _table(keys, columns, doc, *, default):
    """An object with exactly the names ``keys``, each a list with one value per column.

    ``columns``: ``(label, spec)`` pairs, ``spec`` a ``_num`` spec (unit, range, nullable) applied to that position
    of every row. ``default``: the whole table used when the deployment omits the field."""
    return dict(kind="table", keys=tuple(keys), columns=tuple(columns), doc=doc, default=default,
                unit=", ".join(f"{label} {spec['unit']}" for label, spec in columns))


# section -> field -> spec. Units are part of the contract and are echoed into
# runtime capabilities so the model sees the same numbers the worker enforces.
FIELDS = {
    "gripper": {
        "lerobot_closed": _num("LeRobot RANGE_0_100 percent", 0, 100, "LeRobot gripper reading/command when the jaw is closed (goal_normalized 0)."),
        "lerobot_open": _num("LeRobot RANGE_0_100 percent", 0, 100, "LeRobot gripper reading/command when the jaw is open (goal_normalized 1)."),
        "jaw_closed_rad": _num("rad", -0.174533, 1.74533, "URDF jaw joint angle matching lerobot_closed."),
        "jaw_open_rad": _num("rad", -0.174533, 1.74533, "URDF jaw joint angle matching lerobot_open."),
        "max_speed_rad_s": _num("rad/s", 0, 10, "Jaw command ramp speed at gripper_speed_scale 1.", low_open=True),
        "settle_timeout_s": _num("s", 0, 30, "Time after the last jaw command to wait for measured jaw velocity to settle; a blocked jaw (object) is reported, not failed."),
        "empty_close_aperture_m": _num("m", 0, 0.05, "Fingertip gap reported after an EMPTY full close (goal_normalized 0, nothing between the fingers): the CAD opening (so101_gripper_geometry.opening_width_m) of the measured jaw angle, measured on this unit. Set it to the largest value seen over several empty closes (this unit, 2026-11-10..13: 0.0042-0.0051 m; re-measure after recalibration). It is the zero point of robot_action grip_guard presets (preset minimum = this + preset margin) and of the post-grasp slip re-check. Default null: presets are rejected, the re-check is off, explicit min_aperture_m / max_aperture_m still apply; measure and set it before real runs.", nullable=True, default=None),
        "full_open_aperture_m": _num("m", 0, 0.2, "Fingertip gap reported at goal_normalized 1 on this unit (CAD opening of the measured jaw angle); upper bound of explicit robot_action min_aperture_m / max_aperture_m. Default null: the CAD opening at jaw_open_rad is used; measure and set it (development unit 0.1361 m).", nullable=True, default=None),
        "preset_margins_m": _table(GRIP_GUARD_PRESET_NAMES, (
            ("min_margin", _num("m", 0, 0.05, "")),
            ("max_margin", _num("m", 0, 0.2, "", low_open=True, nullable=True))),
            "Margins (m) of the robot_action grip_guard_preset names above empty_close_aperture_m: preset bound = empty_close_aperture_m + margin, as [min_margin_m, max_margin_m or null]; a measured empty-close gap is required to resolve any preset. min_margin_m in [0, 0.05] m (a guarded jaw stage whose measured gap ends below empty_close_aperture_m + min_margin_m pauses the batch), max_margin_m null (no upper bound) or in (min_margin_m, 0.2] m. Default: the shared margins non_empty [0.0015, null], thin_object [0.0005, null], rigid_object [0.002, null], wide_object [0.003, null]; e.g. rigid_object [0.004, null] requires 4 mm above the empty-close gap. With a measured empty_close_aperture_m and full_open_aperture_m every resolved minimum must stay below the full opening. Read by the embodiment profile (EmbodimentProfile.gripper.preset_margins_m).",
            default=_DEFAULT_PRESET_MARGINS_M),
        "hold_squeeze_rad": _num("rad", 0, 1.919863, "Holding squeeze of a closing jaw, in rad of the jaw joint (URDF angle, the unit of jaw_closed_rad / jaw_open_rad). The STS3215 jaw servo is position-P controlled: while its goal lies beyond where the jaw stopped it pushes with a torque that grows with (goal - measured) up to its torque limit, for as long as the goal is held. With this key set, once a closing jaw command (gripper stage, or move with gripper_goal_normalized) has settled short of its goal by more than the squeeze (stopped by an object, or by the other fingertip when empty), the worker sets the jaw goal to the measured jaw angle minus the squeeze toward closed; the steady holding torque is then bounded instead of staying at the torque limit. Fingertip travel equivalent (CAD, so101_gripper_geometry.opening_width_m): 0.010 rad is about 0.8 mm of fingertip gap at 10-50 mm openings (0.65 mm near 5 mm); one encoder tick (0.00153 rad) about 0.12 mm. At servo P_coefficient 16 the unloaded jaw has a ~0.03-0.04 rad static-friction deadband (values inside it generate weak holding torque and do not close further on object loss; 0.08 rad held a 29 mm rigid cube with ~20 % load, 0-1 degC/min rise and no transport slip on the development unit). 0 holds the measured contact angle. An opening command, a jaw that reaches its goal, or one still moving at settle_timeout_s keeps the commanded goal. Default of the robot_action jaw-stage field grip_squeeze_rad, which is offered only when this key is set. Default null: no squeeze is applied and the field is not offered (a closed jaw keeps pushing toward its commanded goal; the behaviour before this key existed). Choose it from scripts/so101_grip_bench.py: the smallest value whose transport shows no slip.", nullable=True, default=None),
        "max_hold_squeeze_rad": _num("rad", 0, 1.919863, "Largest robot_action grip_squeeze_rad a stage may request, in rad of the jaw joint (the schema maximum; a larger request is refused, never clamped). Bounds how hard the model may ask the jaw to hold; a squeeze beyond the servo's torque-limit error (limit / P gain) holds at the torque limit like no squeeze at all. Must be at least hold_squeeze_rad and at most the jaw range |jaw_open_rad - jaw_closed_rad|. Default null: the jaw range. Only read when hold_squeeze_rad is set.", low_open=True, nullable=True, default=None),
    },
    "motion": {
        "max_joint_speed_rad_s": _num("rad/s", 0, 6, "Per-joint speed of the interpolated arm command at speed_scale 1 and speed_scale_cap 1.", low_open=True),
        "max_command_delta_rad": _num("rad", 0, 0.5, "Largest change of any arm joint goal between two consecutive commands.", low_open=True),
        "command_rate_hz": _num("Hz", 1, 200, "Goal streaming and measurement rate during motion, stop and wait."),
        "tracking_error_limit_rad": _num("rad", 0, 3.2, "Maximum |commanded - measured| of any arm joint during motion before stopping with tracking_error.", low_open=True),
        "stall_window_s": _num("s", 0, 10, "Window for stall detection: commanded motion without measured motion.", low_open=True),
        "stall_min_progress_rad": _num("rad", 0, 1, "Minimum measured joint motion in stall_window_s; less counts as a stall when the command moved at least stall_min_commanded_rad.", low_open=True),
        "stall_min_commanded_rad": _num("rad", 0, 1, "Commanded joint motion within stall_window_s needed before a stall can be declared; slow final approaches below servo deadband+sag never trip it.", low_open=True),
        "settle_velocity_rad_s": _num("rad/s", 0, 2, "Measured joint speed below which the arm counts as settled (arrival and stop confirmation).", low_open=True),
        "settle_hold_s": _num("s", 0, 5, "Continuous time below settle_velocity_rad_s required to confirm settling."),
        "arrival_timeout_s": _num("s", 0, 30, "Time after the final arm command to wait for settling before verifying arrival."),
        "arrival_correction_rounds": _num("count", 0, MAX_ARRIVAL_CORRECTION_ROUNDS, "Settled-error correction rounds after a move: the joint goal is shifted by the measured shortfall (q_target - q_measured) to cancel gravity sag of low-gain servos. 0 disables. Default and ceiling of robot_action arrival_correction_rounds (the schema maximum); with 0 (or a 0 rad max offset) those fields are not offered.", integer=True),
        "arrival_correction_min_error_rad": _num("rad", 0, 0.5, "Per-joint settled error below which no correction is applied.", low_open=True),
        "arrival_correction_max_offset_rad": _num("rad", 0, MAX_ARRIVAL_CORRECTION_OFFSET_RAD, "Largest total |goal - target| offset any joint may accumulate from correction (and that hold may re-apply). Default and ceiling of robot_action arrival_correction_max_offset_rad (the schema maximum)."),
        "stop_confirm_timeout_s": _num("s", 0, 30, "Time to confirm a stop by measured settling; otherwise stop state is unknown.", low_open=True),
        "joint_limit_margin_rad": _num("rad", 0, 0.5, "Margin inside the URDF joint limits for IK solutions and home."),
        "default_position_tolerance_m": _num("m", 0, MAX_POSITION_TOLERANCE_M, "TCP position arrival tolerance when a request omits position_tolerance_m.", low_open=True),
        "default_orientation_tolerance_rad": _num("rad", 0, math.pi, "TCP orientation arrival tolerance when a request omits orientation_tolerance_rad.", low_open=True),
        "max_waypoints": _num("count", 1, 8, "Waypoints per single move request.", integer=True),
        "default_timeout_s": _num("s", 0.1, 300, "Request wall-clock deadline when a compiled request omits timeout_s."),
        "max_timeout_s": _num("s", 0.1, 300, "Largest accepted request deadline."),
        "default_wait_timeout_s": _num("s", 0.1, 300, "Request wall-clock deadline of a bounded wait (robot_wait) that omits timeout_s; default 30 s, range [0.1, 300] s and at most max_timeout_s. A wait of n steps lasts about n / command_rate_hz s, so the default leaves ample margin (50 steps at 20 Hz = 2.5 s). Read by the embodiment profile (EmbodimentProfile.default_wait_timeout_s).", default=30.0),
        "tolerance_presets": _table(TOLERANCE_PRESET_NAMES, (
            ("position", _num("m", 0, MAX_POSITION_TOLERANCE_M, "", low_open=True)),
            ("orientation", _num("rad", 0, math.pi, "", low_open=True))),
            "Arrival tolerances of the robot_action tolerance_preset names as [position_tolerance_m, orientation_tolerance_rad]: position in (0, 0.1] m, orientation in (0, pi] rad; an orientation >= 3.14159 rad solves position only. Default: the shared presets tight [0.003, 0.05], normal [0.01, 0.15], relaxed [0.015, 0.2], compliant [0.015, 0.2], coarse [0.025, 0.35], unconstrained_orientation [0.015, 3.14159]; e.g. normal [0.012, 0.15] accepts 12 mm. A move stage without preset or explicit tolerance uses default_position_tolerance_m / default_orientation_tolerance_rad instead. Read by the embodiment profile (EmbodimentProfile.tolerance_presets).",
            default=_DEFAULT_TOLERANCE_PRESETS),
        "max_acceleration_rad_s2": _num(
            "rad/s^2",
            0,
            100,
            "Maximum per-joint angular acceleration for 3rd-order jerk-limited S-curve trajectory compilation "
            "(ruckig), in rad/s^2, range (0, 100] rad/s^2. Must be set together with max_jerk_rad_s3 or both left "
            "null. Default null retains constant-velocity linear joint interpolation; setting both compiles "
            "jerk-limited ruckig trajectories into trajectory_data rows so preview and execute step counts match "
            "identically. Typical 5..30 rad/s^2; e.g. max_acceleration_rad_s2=12.0, max_jerk_rad_s3=80.0.",
            low_open=True,
            nullable=True,
            default=None,
        ),
        "max_jerk_rad_s3": _num(
            "rad/s^3",
            0,
            1000,
            "Maximum per-joint angular jerk for 3rd-order jerk-limited S-curve trajectory compilation (ruckig), "
            "in rad/s^3, range (0, 1000] rad/s^3. Must be set together with max_acceleration_rad_s2 or both left "
            "null. Default null retains constant-velocity linear joint interpolation; setting both compiles "
            "jerk-limited ruckig trajectories into trajectory_data rows so preview and execute step counts match "
            "identically. Typical 30..300 rad/s^3; e.g. max_acceleration_rad_s2=12.0, max_jerk_rad_s3=80.0.",
            low_open=True,
            nullable=True,
            default=None,
        ),
    },
    "servo": {
        "p_coefficient": _num("Feetech register units", 0, 254, "STS3215 position P gain written at connect (LeRobot default 16).", integer=True),
        "i_coefficient": _num("Feetech register units", 0, 254, "STS3215 position I gain written at connect (LeRobot default 0 = pure P+D loop: a gravity-loaded joint then rests short of its goal, the sag that arrival correction and motion_compensation cancel from outside). A value above 0 lets the servo integrate the position error and remove that steady-state sag itself, in transit and at rest. Contact risk: against an obstacle the error does not shrink, so the integral keeps raising the motor torque up to the servo's own limits (integral windup) for as long as the goal stays beyond the obstacle: a blocked move presses harder the longer it lasts, the stall window and stall guard still stop it only by measured position, and a contact_stop hold (goal = measured pose) releases it. Start small (e.g. 4) and measure both effects with scripts/so101_motion_bench.py (transit sag and the contact test's Present_Load) before real runs. Applies to every arm joint and the jaw.", integer=True),
        "d_coefficient": _num("Feetech register units", 0, 254, "STS3215 position D gain (LeRobot default 32).", integer=True),
    },
    "ik": {
        "max_iterations": _num("count", 1, 10000, "Damped least squares iterations per IK attempt.", integer=True),
        "restarts": _num("count", 0, 64, "Extra deterministic IK seeds tried when the measured seed does not converge.", integer=True),
        "orientation_weight_m_per_rad": _num("m/rad", 0, 10, "IK cost weight of rotation-vector error relative to position error. A request with orientation_tolerance_rad >= 3.14159 (pi; ORIENTATION_FREE_TOLERANCE_RAD) solves position only (weight 0). When the weighted solve misses a requested tolerance, a position-first solve (orientation minimised without moving the TCP) is tried and used if it meets both."),
    },
    "plan": {
        "max_start_drift_rad": _num("rad", 0, 1, "Largest measured arm joint change between preview (or checkpoint pause) and execution of a cached plan.", low_open=True),
        "max_age_s": _num("s", 0, 3600, "Cached plan lifetime; the real world keeps moving.", low_open=True),
        "cache_size": _num("count", 1, 256, "Cached plans kept by the worker.", integer=True),
    },
    "workspace": {
        "min_m": _vec(3, "m", "Lower corner of the accepted TCP target box in so101_base."),
        "max_m": _vec(3, "m", "Upper corner of the accepted TCP target box in so101_base."),
    },
    "home": {
        "q_rad": _vec(5, "rad", "Recorded home arm joint configuration (ARM_JOINTS order) for reset/go_home."),
        "gripper_normalized": _num("normalized", 0, 1, "Jaw goal at home (0 closed, 1 open)."),
        "arrival_tolerance_rad": _num("rad", 0, 0.5, "Largest measured arm joint error at home for reset/go_home to count as completed.", low_open=True),
    },
    "telemetry": {
        "joint_hz": _num("Hz", 0, 100, "Operator telemetry joint/state rate.", low_open=True),
        "frame_hz": _num("Hz", 0, 60, "Operator telemetry preview frame rate.", low_open=True),
        "motor_status_hz": _num("Hz", 0, 10, "Rate of temperature/load/voltage reads (0 disables).",),
        "preview_max_side_px": _num("px", 16, 640, "Longest side of operator preview JPEGs.", integer=True),
    },
    "preflight": {
        "min_voltage_v": _num("V", 0, 20, "Lowest servo supply voltage accepted at power-on (STS3215 nominal 12 V supply)."),
        "max_voltage_v": _num("V", 0, 20, "Highest servo supply voltage accepted at power-on."),
        "max_temperature_c": _num("degC", 0, 100, "Highest servo temperature accepted at power-on."),
        "urdf_limit_tolerance_rad": _num("rad", 0, 0.5, "Allowed excess of a mapped joint reading beyond its URDF limit before the joint mapping is rejected."),
        "calibration_range_tolerance_rad": _num("rad", 0, 0.5, "Allowed excess of a raw servo position beyond the LeRobot calibration range (range_min / range_max, recorded by hand at calibration) at power-on, in rad of the servo shaft (4096 ticks = 2*pi rad). A relaxed arm resting on a mechanical end stop can read a few ticks past the hand-recorded range; a larger excess points to a wrong calibration file or a slipped servo horn and stops startup. The excess is reported per joint in the preflight record. Default 0.05 rad (about 33 ticks), the same allowance as urdf_limit_tolerance_rad; 0 restores the strict check.", default=0.05),
    },
    "operator": {
        "min_speed_scale_cap": _num("ratio", 0, 1, "Lowest operator speed_scale_cap; the cap can only lower deployment speed.", low_open=True),
        "default_speed_scale_cap": _num("ratio", 0, 1, "Initial operator speed_scale_cap at worker startup, scaling motion.max_joint_speed_rad_s and gripper.max_speed_rad_s; default 0.9, range [min_speed_scale_cap, 1]. Runtime set_runtime_limits overrides it.", low_open=True, default=calibration_default("default_speed_scale_cap")),
    },
    "camera": {
        "frame_timeout_s": _num("s", 0, 10, "Wait for a frame captured after the observation request.", low_open=True),
        "video_crf": _num("H.264 CRF", 0, 51, "Motion-only observer video quality, default 23; typical 18..28, lower is higher quality/larger files. E.g. 20 retains more detail. No resolution or frame-rate reduction.", integer=True, default=calibration_default("video_crf")),
        "video_queue_mb": _num("MiB per camera", 1, 1024, "Maximum pending uncompressed motion-video frames per camera, default 64 MiB, typical 32..128. E.g. 128 tolerates longer encoder stalls. Overflow is reported as dropped frames and never blocks camera or motor control.", integer=True, default=calibration_default("video_queue_mb")),
    },
    "calibration": {
        "min_board_corners": _num("count", 4, 200, "ChArUco inner corners EVERY camera must detect before the one-click operator calibration moves the arm; board_check reports against the same value.", integer=True),
        "sample_min_wrist_corners": _num("count", 4, 200, "Wrist-camera corners needed to keep a pose sample; poses with fewer are skipped and recorded.", integer=True),
        "speed_scale": _num("ratio", 0, 1, "Fraction of the effective joint speed (deployment max x operator speed_scale_cap) used for calibration moves.", low_open=True),
        "arrival_tolerance_rad": _num("rad", 0, 0.5, "Largest settled arm joint error at a calibration pose (move_joints arrival).", low_open=True),
        "settle_timeout_s": _num("s", 0, 30, "Extra measured-settling wait after arrival, before the frames of a sample are captured.", low_open=True),
        "capture_max_drift_rad": _num("rad", 0, math.pi, "S2/S4: maximum measured endpoint joint drift across asynchronous frame capture; sample rejected above it. Default null explicitly derives twice arrival_tolerance_rad (legacy behavior); an explicit positive value is used unchanged, e.g. 0.02 rad. Endpoint drift is not proof of continuous stillness. Effective bound is archived per session/sample.", low_open=True, nullable=True, default=calibration_default("capture_max_drift_rad")),
        "z_floor_m": _num("m", -2, 2, "S2/S4 optional path screen: skip a joint-interpolated pose path if a moving link origin falls below this base-frame z. Default null disables the existing screen; e.g. -0.02 m screens link origins below -20 mm. This is sparse FK of origins, not mesh collision or measured table clearance; it does not alter targets.", nullable=True, default=calibration_default("z_floor_m")),
        "pan_offsets_rad": _list(8, "rad", -1.0, 1.0, "shoulder_pan offsets from home.q_rad; the grid is every pan x flex x roll combination."),
        "flex_offsets_rad": _list(8, "rad", -1.0, 1.0, "wrist_flex offsets from home.q_rad (grid axis)."),
        "roll_offsets_rad": _list(8, "rad", -1.5, 1.5, "wrist_roll offsets from home.q_rad (grid axis); varied rotation axes are what hand-eye needs."),
        "lift_offsets_rad": _list(8, "rad", -0.5, 0.5, "Extra poses that change arm height/distance: shoulder_lift + d and elbow_flex - lift_elbow_coupling * d."),
        "lift_elbow_coupling": _num("ratio", 0, 2, "elbow_flex counter-rotation per rad of shoulder_lift offset in the lift poses (keeps the wrist camera aimed at the table)."),
        "capture_timeout_s": _num("s", 1, 1800, "Worker wall-clock budget for visiting poses; no new pose is started after it (the arm then parks at home)."),
        "solve_timeout_s": _num("s", 1, 1800, "Parent wait budget for the offline solve after capture (added to capture_timeout_s and ipc.response_margin_s)."),
        "min_samples": _num("count", 3, 200, "Kept samples required before solving; fewer ends with insufficient_samples and keeps the active calibration.", integer=True),
        "max_pnp_reprojection_px": _num("px", 0, 20, "Wrist samples whose board PnP RMS exceeds this are excluded from hand-eye.", low_open=True),
        "max_intrinsics_rms_px": _num("px", 0, 20, "Gate: intrinsics RMS reprojection of every camera.", low_open=True),
        "max_hand_eye_translation_m": _num("m", 0, 0.1, "Gate: hand-eye translation RMS spread of the static board in so101_base.", low_open=True),
        "max_hand_eye_rotation_rad": _num("rad", 0, 1, "Gate: hand-eye rotation RMS spread.", low_open=True),
        "hand_eye_score_lever_m": _num("m/rad", 0, 10, "S2 hand-eye method ranking: score_m = translation_rms_m + this coefficient times rotation_rms_rad. Default 0.1 m/rad; e.g. 0.2 weights angular spread twice as strongly, while 0 ranks by translation only. A chosen numerical tradeoff, not a measured camera distance or accuracy threshold. Effective value is archived; acceptance gates stay separate.", default=calibration_default("hand_eye_score_lever_m")),
        "max_fixed_reprojection_px": _num("px", 0, 100, "Gate: fixed-camera end-to-end chain reprojection RMS (board via wrist chain projected into the fixed camera).", low_open=True),
        "handheld_target_views": _num("count", 8, 200, "Hand-held fixed-camera intrinsics (console 手持标定固定相机): capture completes after this many kept views.", integer=True),
        "handheld_min_views": _num("count", 8, 200, "Hand-held: kept views required before handheld_solve runs (OpenCV intrinsics need >= 8 distinct views).", integer=True),
        "handheld_still_px": _num("px", 0, 20, "Hand-held: mean ChArUco corner motion between consecutive fixed-camera frames below which the board counts as held still (a view is kept only when still).", low_open=True),
        "handheld_min_novelty": _num("ratio", 0, 1, "Hand-held: minimum view-signature distance using the same full reference board corners projected from detected ID/pixel correspondences (centroid/spread as image-size fractions plus in-plane orientation), not the changing detected subset. Smaller keeps more similar views.", low_open=True),
        "handheld_timeout_s": _num("s", 10, 1800, "Hand-held: capture stops (status timeout) after this wall-clock time; kept views stay usable for handheld_solve."),
        "handheld_near_fraction": _num("image fraction", 0, 2, "S3 coverage only: full projected inner-corner bounding extent at or above this fraction of image width/height counts near. Default 0.55; e.g. 0.6 requires 60 percent extent. Does not gate solve.", low_open=True, default=calibration_default("handheld_near_fraction")),
        "handheld_far_fraction": _num("image fraction", 0, 2, "S3 coverage only: projected inner-corner extent below this image fraction counts far. Default 0.3; e.g. 0.25 means below 25 percent. Must be below handheld_near_fraction; does not gate solve.", low_open=True, default=calibration_default("handheld_far_fraction")),
        "handheld_tilt_small": _num("edge-length ratio", 1, 100, "S3 coverage only: maximum opposite projected board-edge ratio at or above this counts small tilt. Default 1.06; 1 is frontal perspective. Must not exceed handheld_tilt_large. It is not a measured tilt angle or a diversity certificate.", default=calibration_default("handheld_tilt_small")),
        "handheld_tilt_large": _num("edge-length ratio", 1, 100, "S3 coverage only: opposite projected board-edge ratio at or above this counts large tilt. Default 1.15; e.g. 1.2 denotes 20 percent edge ratio imbalance. It is not a measured tilt angle or a solve gate.", default=calibration_default("handheld_tilt_large")),
        "handheld_still_frames": _num("frames", 2, 100, "S3: consecutive camera detections required for stillness, sharing min_board_corners with motion below handheld_still_px. Default 2 (one quiet transition); e.g. 3 requires two consecutive quiet transitions. Real elapsed time depends on camera delivery.", integer=True, default=calibration_default("handheld_still_frames")),
        "handheld_min_sharpness": _num("squared 8-bit gray-level Laplacian units", 0, 1e9, "S3: minimum variance of the full grayscale frame's discrete OpenCV Laplacian response (ksize=1). Default 0 disables the gate. An explicit positive threshold, e.g. 50, rejects lower or unavailable measurements; optics/exposure dependent, not metric precision. The latest frame verdict reports the measured value.", default=calibration_default("handheld_min_sharpness")),
        "max_handheld_intrinsics_rms_px": _num("px", 0, 20, "Gate: RMS reprojection of the hand-held fixed-camera intrinsics solve; above it the active calibration file is kept.", low_open=True),
        # ---- calibration workflow (calibration_workflow.py, page /calibration) ----
        "board_thickness_m": _num("m", 0, 0.05, "Workflow default board thickness: printed face above the table surface. Shown on step S2 and editable per run."),
        "extrinsic_pose_count": _num("count", 1, 50, "S4 fixed-camera extrinsic: poses visited with the board static (home first, then the S2 grid order).", integer=True),
        "touch_points_xy_m": _xy(8, "m", -0.6, 0.6, "S1 touch-off: table contact targets [x, y] in so101_base; at each one the closed fingertips descend until contact."),
        "verify_points_xy_m": _xy(8, "m", -0.6, 0.6, "S6 verification touch targets [x, y] in so101_base; pick points not in touch_points_xy_m."),
        "touch_hover_z_m": _num("m", 0, 0.3, "S1/S6: TCP height of the hover pose above each touch point before descending."),
        "touch_min_z_m": _num("m", -0.2, 0.1, "S1/S6: lowest commanded TCP z of a descent; no contact above it ends that point with no_contact_before_min_z."),
        "touch_step_m": _num("m", 0, 0.01, "S1/S6: vertical spacing of requested IK endpoints from measured hover; template 0.002 m. Joint interpolation subdivides each endpoint at the deployment speed/command-delta bounds, so step/period is an upper nominal descent rate, not guaranteed actual TCP velocity. A candidate holds the current interpolated goal.", low_open=True),
        "touch_step_period_s": _num("s", 0.01, 2, "S1/S6: measurement wait after each descent, confirmation or sliding command; template 0.12 s."),
        "touch_contact_lag_rad": _num("rad", 0, 0.5, "S1/S6: obstruction candidate when the largest |sent goal - fixed hover sag offset - measured| arm joint residual exceeds this; does not alone prove table contact. Template 0.03 rad.", low_open=True),
        "touch_contact_load_delta": _num("uncalibrated STS3215 Present_Load raw units", 0, 1000, "S1/S6: obstruction candidate when |Present_Load| of shoulder_lift/elbow_flex/wrist_flex rises this much above its free-air baseline; 0 disables this trigger. Template 120 raw units. This is not a measured force."),
        "touch_warmup_steps": _num("count", 0, 50, "S1/S6: first descent steps that only learn the free-air load; contact cannot be declared during them.", integer=True),
        "touch_confirm_reads": _num("count", 1, 10, "S1/S6: consecutive reads above an obstruction trigger while retaining the same actual command; template 2. Confirmation establishes persistent obstruction evidence, not independently observed table contact.", integer=True),
        "touch_jaw_normalized": _num("normalized", 0, 1, "S1/S6: jaw goal during touches (0 closed: fingertips together)."),
        "touch_speed_scale": _num("ratio", 0, 1, "S1/S6: fraction of the effective joint speed for hover, lift and park moves.", low_open=True),
        "touch_arrival_tolerance_rad": _num("rad", 0, 0.5, "S1/S6: settled arm joint error accepted at hover/lift/park poses.", low_open=True),
        "touch_ik_max_position_error_m": _num("m", 0, 0.05, "S1/S6: maximum IK position residual for every hover, descent and sliding target; template 0.005 m. A worse hover is skipped; a worse active trajectory target stops further motion.", low_open=True),
        "touch_orientation_weight": _num("1/rad", 0, 10, "S1/S6: multiplier of orientation-log residual (rad) in IK, with the position weight fixed at 1/m. Default 0 preserves position-only touch IK; positive values explicitly trade position against downward-TCP attitude, e.g. 0.3/rad. Applies to hover/descent/sliding; not an orientation tolerance or contact assertion.", default=calibration_default("touch_orientation_weight")),
        "touch_settle_s": _num("s", 0, 5, "S1/S6: time holding the measured pose with fixed sag compensation before lifting; this does not guarantee zero force."),
        "touch_timeout_s": _num("s", 10, 1800, "S1/S6: worker budget for one touch sequence; no new point is started after it."),
        "touch_slide_distance_m": _num("m", 0, 0.05, "S1/S6 optional trajectory collection: outward base-XY distance followed by a return over the same path at the candidate commanded z, without additional downward commands. Template 0 disables sliding; e.g. 0.01 requests 10 mm out and 10 mm back. Range 0..0.05 m; requires readable load monitoring and operator supervision; does not assert motion lies on the true table plane.", default=calibration_default("touch_slide_distance_m")),
        "touch_slide_direction_rad": _num("rad", -math.pi, math.pi, "S1/S6 optional sliding: outward direction in so101_base XY, measured counterclockwise from +X; template 0 (+X), range -pi..pi. Ignored when distance is 0; e.g. pi/2 travels toward +Y.", default=calibration_default("touch_slide_direction_rad")),
        "touch_slide_step_m": _num("m", 0, 0.005, "S1/S6 optional sliding: maximum base-XY increment per command at touch_step_period_s; template 0.001 m, range (0,0.005] m. Final increment is shortened to the requested endpoint; e.g. 0.001 at 0.12 s requests about 8.3 mm/s before joint limits." , low_open=True, default=calibration_default("touch_slide_step_m")),
        "touch_max_tracking_error_rad": _num("rad", 0, 0.5, "S1/S6 descent, confirmation and sliding: stop if any |sent goal - fixed hover sag offset - measured| residual exceeds this bound. Template 0.15 rad, range (0,0.5] rad; e.g. 0.1 bounds residual at 0.1 rad. Limits servo position error, not contact force.", low_open=True, default=calibration_default("touch_max_tracking_error_rad")),
        "touch_max_load_rise": _num("uncalibrated STS3215 Present_Load raw units", 0, 1000, "S1/S6 descent, confirmation and sliding: stop if any monitored absolute load rises above this bound relative to the fixed measured-hover baseline (never adapted during warmup). Template 300 raw units, range (0,1000]; e.g. 200 applies a lower raw-load bound. Sliding requires readable load monitoring. This is not a calibrated torque or contact-force limit.", low_open=True, default=calibration_default("touch_max_load_rise")),
        "touch_slide_max_duration_s": _num("s", 0, 60, "S1/S6 optional sliding: maximum elapsed time for the entire outward-and-return trajectory; template 10 s, range (0,60] s. Global touch_timeout_s also applies; e.g. 5 stops a trajectory unfinished after 5 s.", low_open=True, default=calibration_default("touch_slide_max_duration_s")),
        "touch_video_fps": _num("Hz", 0, 60, "S1/S6 operator evidence: requested sampling frequency per wrist/fixed camera for bounded local video-frame archives; default 10 Hz, range 0..60 Hz. 0 disables recording (e.g. CPU-only tests); otherwise both cameras must exist. Uses available frames without changing camera rate or motor control. Does not start SAM, GPU inference or network requests.", default=calibration_default("touch_video_fps")),
        "touch_video_max_frames": _num("frames per camera", 1, 36000, "S1/S6 local video evidence: maximum saved frames per camera per touch-sequence attempt; default 600, range 1..36000. E.g. 300 bounds each camera to 300 frames. Truncation is recorded as incomplete visual evidence and never changes the physical motion receipt; ignored when touch_video_fps is 0.", integer=True, default=calibration_default("touch_video_max_frames")),
        "touch_video_max_duration_s": _num("s", 0, 1800, "S1/S6 local video evidence: maximum recording duration per touch-sequence attempt; default 60 s, range (0,1800] s. E.g. 30 records at most 30 s even if motion continues; the frame-count limit also applies. Not a motion timeout; ignored when touch_video_fps is 0.", low_open=True, default=calibration_default("touch_video_max_duration_s")),
        "joint_offset_prior_rad": _num("rad", 0, 1, "S5 joint solve: 1-sigma Gaussian prior on each estimated joint zero-offset change.", low_open=True),
        "joint_plane_sigma_m": _num("m", 0, 0.05, "S5: 1-sigma of the prior 'board top = table + board thickness'.", low_open=True),
        "joint_touch_sigma_m": _num("m", 0, 0.05, "S5: 1-sigma of each touch constraint 'contact point z = table'.", low_open=True),
        "joint_pixel_sigma_px": _num("px", 0, 10, "S5: 1-sigma ChArUco corner noise weighting the reprojection residuals.", low_open=True),
        "joint_solver_max_nfev": _num("function evaluations", 1, 100000, "S5 joint solve: maximum residual evaluations passed to scipy least_squares. Default 400; e.g. 800 raises the numerical budget only, not a physical acceptance limit.", integer=True, default=calibration_default("joint_solver_max_nfev")),
        "joint_pixel_robust_scale": _num("normalized residual units", 0, 100, "S5 joint solve: soft-L1 transition of visual residuals after pixel_sigma and existing corner-count normalization. Default 3; e.g. 2 downweights outliers earlier. Not a pixel-error threshold and does not change physical gates.", low_open=True, default=calibration_default("joint_pixel_robust_scale")),
        "joint_solver_ftol": _num("dimensionless", 1e-15, 1, "S5 scipy least_squares relative cost termination tolerance. Default 1e-8; e.g. 1e-9 requests tighter numerical convergence, not better physical precision.", low_open=True, default=calibration_default("joint_solver_ftol")),
        "joint_solver_xtol": _num("dimensionless", 1e-15, 1, "S5 scipy least_squares relative parameter-step termination tolerance. Default 1e-8; e.g. 1e-9 requests tighter numerical convergence. Does not replace pose/contact gates.", low_open=True, default=calibration_default("joint_solver_xtol")),
        "joint_solver_gtol": _num("dimensionless", 1e-15, 1, "S5 scipy least_squares scaled-gradient termination tolerance. Default 1e-8; e.g. 1e-9 requests tighter numerical convergence. Does not establish parameter identifiability.", low_open=True, default=calibration_default("joint_solver_gtol")),
        "joint_solver_x_scale": _num("dimensionless", 0, 1000, "S5 scipy least_squares uniform variable scaling. Default 1 retains unscaled parameters; e.g. 0.5 rescales numerical trust-region steps. No change to parameter units or priors.", low_open=True, default=calibration_default("joint_solver_x_scale")),
        "table_prior_sigma_m": _num("m", 0, 0.1, "S5: 1-sigma of the table-height prior centred on the S1 touch mean (nominal kinematics).", low_open=True),
        "max_joint_reprojection_px": _num("px", 0, 20, "Gate S5: wrist reprojection RMS of the joint solve.", low_open=True),
        "max_board_height_error_m": _num("m", 0, 0.05, "Gate S5: |board top - (table + board thickness)| after the joint solve.", low_open=True),
        "max_touch_residual_m": _num("m", 0, 0.05, "Gate S5: largest |contact point z - table z| over the touches after the joint solve.", low_open=True),
        "max_joint_offset_rad": _num("rad", 0, 1, "Gate S5: largest |joint_mapping offset_rad| (current + estimated change) of any arm joint.", low_open=True),
        "max_verify_touch_error_m": _num("m", 0, 0.05, "Gate S6: largest |verification contact z - calibrated table z| after apply.", low_open=True),
        "xy_max_error_m": _num("m", 0, 0.1, "S6 operator XY-check form default: maximum lateral norm of triangulated fixed-tip minus CAD/FK. Default 0.005 m; e.g. 0.004 requests 4 mm. Explicit per-check input remains authoritative. Separate from table-touch height acceptance; shared-camera consistency is not absolute metrology.", low_open=True, high_open=True, default=calibration_default("xy_max_error_m")),
        "xy_max_reprojection_error_px": _num("px", 0, 100, "S6 operator XY-check form default: maximum stereo reprojection residual for an accepted triangulation. Default 3 px; e.g. 2 rejects poorer pixel fits. Explicit check input is recorded; this is not metric point accuracy.", low_open=True, high_open=True, default=calibration_default("xy_max_reprojection_error_px")),
        "xy_min_parallax_deg": _num("deg", 0, 90, "S6 operator XY-check form default: minimum angle between triangulation rays. Default 1 degree; e.g. 2 excludes weaker stereo geometry. Explicit check input remains authoritative. Rejected geometry stays unknown, not a zero error.", low_open=True, high_open=True, default=calibration_default("xy_min_parallax_deg")),
        "selfcheck_max_telemetry_age_s": _num("s", 0, 60, "Gate S0: newest joint read must be younger than this.", low_open=True),
        "staleness_max_board_shift_px": _num("px", 0, 200, "Status hint: RMS px between the board corners the fixed camera detects and where the active calibration predicts them; above it the calibration is shown as possibly stale.", low_open=True),
        "staleness_check_period_s": _num("s", 0, 600, "Status hint: shortest interval between two staleness checks (each runs ChArUco detection on the latest frames).", low_open=True),
    },
    "ipc": {
        "response_margin_s": _num("s", 1, 600, "Parent wait beyond an action deadline before reporting status unknown."),
        "operator_timeout_s": _num("s", 1, 120, "Parent deadline for an operator command reply."),
    },
}

CAMERA_FIELDS = {
    "index_or_path": "OpenCV device index (int) or device path (str). Used only when unique_id and device_name are "
                     "both null (index binding). An index is a position in the present device list, so it shifts when "
                     "a camera is unplugged and then opens another device; the worker records the device it opened "
                     "(name, unique id) in <run_dir>/camera_binding.json, robot-worker.log and S0. Still required; "
                     "ignored when an identity is set.",
    "unique_id": "Optional stable device identity (string); null = not used. macOS: AVCaptureDevice uniqueID, shown by "
                 "scripts/so101_cameras.py and as spcamera_unique-id by system_profiler SPCameraDataType, e.g. "
                 "\"0x1340001b170209\" for a USB camera (USB location + vendor + product id: it changes when the camera "
                 "moves to another USB port or hub). Linux: a /dev/v4l/by-id/... or /dev/v4l/by-path/... link, full "
                 "path. Must match exactly one present device, else the camera refuses to start "
                 "(so101_camera_not_found / so101_camera_ambiguous); never falls back to index_or_path. Two cameras "
                 "may not share one unique_id.",
    "device_name": "Optional case-insensitive substring of the device name or model (macOS localizedName/modelID, "
                   "Linux sysfs name); null = not used. With unique_id set, both must hold. Must match exactly one "
                   "present device, else the camera refuses to start (so101_camera_not_found / "
                   "so101_camera_ambiguous); never falls back to index_or_path. Identical camera models report the "
                   "same name: bind those by unique_id.",
    "width": "Capture width in px (after rotate_deg; must match the calibration image_size when calibrated).",
    "height": "Capture height in px (after rotate_deg).",
    "fps": "Requested capture rate in Hz.",
    "fourcc": "Optional four-character codec, e.g. MJPG; null keeps the driver default.",
    "rotate_deg": "Optional clockwise frame rotation in degrees (0, 90, 180 or 270; default 0).",
}


def _check_number(value, spec, key):
    if value is None and spec.get("nullable"):
        return None
    if spec["kind"] == "int":
        if type(value) is not int:
            raise ValueError(f"so101_deployment_invalid:{key}: integer required ({spec['unit']})")
    elif type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"so101_deployment_invalid:{key}: finite number required ({spec['unit']})")
    low, high = spec["low"], spec["high"]
    if value > high or value < low or (spec["low_open"] and value <= low) or (spec.get("high_open") and value >= high):
        bound = "(" if spec["low_open"] else "["
        upper = ")" if spec.get("high_open") else "]"
        raise ValueError(f"so101_deployment_out_of_range:{key}: {value} not in {bound}{low}, {high}{upper} {spec['unit']}")
    return float(value) if spec["kind"] == "num" else value


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items() if not key.startswith("_")}
    return value


IK_SERVICE_SOLVERS = ("curobo", "pink")
IK_SERVICE_FIELDS = {
    "solver": "IK and path-collision solver: \"curobo\" (remote NVIDIA cuRobo v2 multi-seed GPU IK HTTP service, "
              "services/curobo_ik; requires loopback url) or \"pink\" (local in-process Pinocchio + Pink + DAQP IK "
              "with Coal CAD link-hull and open-jaw Box collision checking; url default null, zero SSH tunnel). "
              "Required when ik_service is set; e.g. \"pink\" or \"curobo\".",
    "url": "Service base URL; required for solver \"curobo\" (loopback only: http://127.0.0.1:<port> or "
           "http://localhost:<port>, normally the SSH tunnel of remote.services curobo_ik, e.g. "
           "\"http://127.0.0.1:8341\"). Optional or null for solver \"pink\" (default null: runs locally in-process "
           "with zero network sockets; if provided, must still be a valid loopback HTTP URL).",
    "timeout_s": "Wall-clock limit of one HTTP IK request in s, range (0, 60] s; a slower answer fails planning with "
                 "ik_service_unreachable (default 10.0 s, typical 2..15 s, e.g. 10.0; unused for in-process solver "
                 "\"pink\").",
    "num_solutions": "Candidate joint solutions evaluated per pose, integer count in [1, 32]; the planner keeps the "
                     "valid one with the least joint travel from seed (default 8, typical 4..16, e.g. 8; for \"pink\", "
                     "evaluates the seed plus num_solutions - 1 deterministic restarts).",
    "table_top_z_m": "Top plane height z of the support table checked for collision, in m in so101_base, range "
                     "[-0.3, 0.3] m, or null for no table collision check (default null, typical -0.03..0.02 m, "
                     "e.g. -0.02).",
    "path_check_step_rad": "Largest joint angle change between consecutive samples of the preview path collision "
                           "check, in rad, range [0.002, 0.2] rad (default 0.01 rad: about 4 mm of fingertip travel "
                           "at 0.4 m reach; typical 0.005..0.02 rad, e.g. 0.01). A smaller step finds thinner overlaps "
                           "and checks more samples.",
}


def validate_ik_service(value):
    """The normalized ``ik_service`` section, or None when absent / null (planning then uses the packaged local IK).

    Fields and defaults: ``IK_SERVICE_FIELDS``. Examples:
    ``{"solver": "pink", "table_top_z_m": -0.02, "path_check_step_rad": 0.01}`` (local in-process, ``url`` null) or
    ``{"solver": "curobo", "url": "http://127.0.0.1:8341", "timeout_s": 10, "num_solutions": 8, "table_top_z_m": -0.02,
    "path_check_step_rad": 0.01}``."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("so101_deployment_invalid:ik_service: object or null required")
    value = _clean(value)
    unknown = set(value) - set(IK_SERVICE_FIELDS)
    if unknown:
        raise ValueError("so101_deployment_invalid:ik_service: unknown " + ",".join(sorted(unknown)))
    solver = value.get("solver")
    if solver not in IK_SERVICE_SOLVERS:
        raise ValueError(f"so101_deployment_invalid:ik_service.solver: one of {list(IK_SERVICE_SOLVERS)}")
    url = value.get("url")
    if solver == "pink" and url is None:
        normalized_url = None
    else:
        from urllib.parse import urlsplit
        parts = urlsplit(url) if isinstance(url, str) else None
        if parts is None or parts.scheme != "http" or parts.hostname not in ("127.0.0.1", "localhost") or not parts.port:
            raise ValueError("so101_deployment_invalid:ik_service.url: http://127.0.0.1:<port> or http://localhost:<port>")
        normalized_url = url.rstrip("/")
    table = value.get("table_top_z_m")
    return dict(solver=solver, url=normalized_url,
                timeout_s=_check_number(value.get("timeout_s", 10.0), _num("s", 0, 60, "", low_open=True), "ik_service.timeout_s"),
                num_solutions=_check_number(value.get("num_solutions", 8), _num("count", 1, 32, "", integer=True),
                                            "ik_service.num_solutions"),
                table_top_z_m=None if table is None else _check_number(table, _num("m", -0.3, 0.3, ""),
                                                                       "ik_service.table_top_z_m"),
                path_check_step_rad=_check_number(value.get("path_check_step_rad", 0.01), _num("rad", 0.002, 0.2, ""),
                                                  "ik_service.path_check_step_rad"))


#: Deployment section ``motion_compensation`` (optional; absent or null = every mechanism off, the behaviour before
#: the section existed). Each key switches one mechanism independently, so they can be compared on hardware.
MOTION_COMPENSATION_FIELDS = {
    "carry_arrival_offset": "bool, default false. true: a move streams its interpolated joint path plus the sag offset "
                            "(rad per arm joint) the last settled free-space arrival ended with, and its own arrival "
                            "correction starts from that offset instead of 0, so the goal no longer drops back to the "
                            "sagged pose at the start of every move. The offset is bounded by the move's "
                            "arrival_correction_max_offset_rad and is 0 when the move disables correction "
                            "(arrival_correction_rounds 0), on a contact / compliant move (stall_guard_preset "
                            "contact_stop, tolerance_preset compliant, max_upward_dz_m > 0), after a stop signal or "
                            "contact hold, and after any goal written by something other than an arrival or a pause. "
                            "Tracking, stall and divergence guards judge against the path without the offset. Use it "
                            "when stage results show transit.max_below_path_m of several mm at the start of moves.",
    "no_response_stop": "bool, default false. true: an arrival-correction round after which no corrected joint moved "
                        "by more than one encoder tick (driver.ENCODER_RESOLUTION_RAD, 2*pi/4096 rad) ends the "
                        "correction with termination_reason no_response; the goal returns to the last offset the arm "
                        "answered, so the step that stayed inside gear static friction is neither kept as stored "
                        "servo effort nor reused by holds and later moves. false: every round adds its measured error.",
    "stop_correction_within_tolerance": "bool, default false. true: a robot_action move's arrival correction sends a "
                                        "round only while the settled TCP is outside the move's own arrival rule "
                                        "(position_tolerance_m, orientation_tolerance_rad or the approach / roll "
                                        "bounds, max_upward_dz_m); once a settled pose is inside it the correction "
                                        "ends with termination_reason within_tolerance and that same pose is judged, "
                                        "so correction can never turn an arrival inside the requested tolerance into "
                                        "a failure, and an arm resting on a support inside a max_upward_dz_m window "
                                        "is not pushed harder. Trade-off: the arm stops anywhere inside the "
                                        "tolerance instead of being driven on to the IK joint target, so the final "
                                        "error can be as large as the tolerance the move asked for; a move that "
                                        "needs more accuracy asks for a tighter tolerance. false: every round drives "
                                        "the joints toward the IK target, which improves position but can worsen an "
                                        "orientation the IK itself only met by a small margin. Operator and "
                                        "calibration joint moves are not affected.",
    "arrival_breakaway_rad": "object {arm joint: rad in (0, 0.1]} or null (default null = off). The smallest "
                             "arrival-correction goal step that makes that joint move from rest (its static-friction "
                             "band in goal units, from scripts/so101_gravity_identify.py --breakaway). A correction "
                             "step smaller than it is enlarged to it, keeping its sign; the round reports the joints "
                             "in breakaway_joints. Joints left out are corrected by the measured error as before. The "
                             "enlarged step is the band itself, never more, so the goal moves past the measured "
                             "error by at most band - error and the joint can overshoot by up to that (rad; times "
                             "the distance from the joint to the TCP in m at the tool). Requires "
                             "stop_correction_within_tolerance true, so a round that lands inside the move's "
                             "tolerance is not followed by another enlarged step, and a set-down target at least "
                             "band x reach above the support. Use only with values identified on this unit.",
    "gravity_feedforward": "object or null (default null = off). Adds to every streamed arm goal the offset "
                           "-gravity_torque(goal) / joint stiffness (rad), the position error a pure-P servo needs to "
                           "hold the pose against gravity (so101_gravity: URDF link masses, gravity along -Z of "
                           "so101_base, no held object). Keys: joint_stiffness_nm_per_rad {arm joint: N*m/rad in "
                           "(0, 1000]}, the identified servo position stiffness of each joint (restoring torque per "
                           "rad of error at the deployed servo.p_coefficient; joints left out get no feedforward); "
                           "max_offset_rad, rad in (0, 0.3], bound of each joint's feedforward offset; provenance, a "
                           "non-empty string naming the identification the values came from (e.g. the proposal file "
                           "and date). Arrival correction and carry_arrival_offset then act on the residual only; "
                           "their bounds do not include the feedforward. It also acts on contact moves: when a "
                           "guard stops a press, the servo no longer leaves the arm's weight to the support and "
                           "pushes harder than without it by about the gravity torque. The LeRobot driver clamps "
                           "every goal to max_relative_target_deg from the present position, so the deployment must "
                           "satisfy motion.arrival_correction_max_offset_rad + max_offset_rad + "
                           "motion.max_command_delta_rad <= radians(max_relative_target_deg) (the largest offset a "
                           "goal can carry plus one command step); otherwise it is rejected, because a clamped goal "
                           "would silently drop the compensation the receipts report. Re-identify after changing "
                           "servo gains.",
}
_STIFFNESS_SPEC = _num("N*m/rad", 0, 1000, "", low_open=True)
_FEEDFORWARD_OFFSET_SPEC = _num("rad", 0, 0.3, "", low_open=True)
_BREAKAWAY_SPEC = _num("rad", 0, 0.1, "", low_open=True)


def _joint_table(value, spec, key):
    """``{arm joint: number}`` for a non-empty subset of ``ARM_JOINTS``, each checked against ``spec``."""
    if not isinstance(value, dict) or not value or set(value) - set(ARM_JOINTS):
        raise ValueError(f"so101_deployment_invalid:{key}: object with arm joint names {list(ARM_JOINTS)} required")
    return {name: _check_number(value[name], spec, f"{key}.{name}") for name in ARM_JOINTS if name in value}


def validate_motion_compensation(value):
    """The normalized ``motion_compensation`` section; every mechanism off when the section is absent / null.

    Fields, units and defaults: ``MOTION_COMPENSATION_FIELDS``. Always returns all five keys
    (``carry_arrival_offset`` / ``no_response_stop`` / ``stop_correction_within_tolerance`` bool, ``arrival_breakaway_rad`` / ``gravity_feedforward`` object
    or None). Example: ``{"carry_arrival_offset": true, "no_response_stop": true, "gravity_feedforward":
    {"joint_stiffness_nm_per_rad": {"shoulder_lift": 9.5, "elbow_flex": 8.2}, "max_offset_rad": 0.1,
    "provenance": "gravity-identify 2026-11-17"}}``."""
    key = "motion_compensation"
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError(f"so101_deployment_invalid:{key}: object or null required")
    value = _clean(value)
    unknown = set(value) - set(MOTION_COMPENSATION_FIELDS)
    if unknown:
        raise ValueError(f"so101_deployment_invalid:{key}: unknown " + ",".join(sorted(unknown)))
    out = {}
    for switch in ("carry_arrival_offset", "no_response_stop", "stop_correction_within_tolerance"):
        if type(value.get(switch, False)) is not bool:
            raise ValueError(f"so101_deployment_invalid:{key}.{switch}: bool required")
        out[switch] = value.get(switch, False)
    breakaway = value.get("arrival_breakaway_rad")
    out["arrival_breakaway_rad"] = (None if breakaway is None else
                                    _joint_table(breakaway, _BREAKAWAY_SPEC, f"{key}.arrival_breakaway_rad"))
    if out["arrival_breakaway_rad"] and not out["stop_correction_within_tolerance"]:
        # An enlarged step can overshoot; without this rule the next round would answer with another full band.
        raise ValueError(f"so101_deployment_invalid:{key}.arrival_breakaway_rad requires "
                         "stop_correction_within_tolerance true")
    feedforward = value.get("gravity_feedforward")
    if feedforward is not None:
        name = f"{key}.gravity_feedforward"
        fields = {"joint_stiffness_nm_per_rad", "max_offset_rad", "provenance"}
        if not isinstance(feedforward, dict) or set(_clean(feedforward)) != fields:
            raise ValueError(f"so101_deployment_invalid:{name}: object with exactly {sorted(fields)} or null required")
        if not isinstance(feedforward["provenance"], str) or not feedforward["provenance"].strip():
            raise ValueError(f"so101_deployment_invalid:{name}.provenance: non-empty string required")
        feedforward = dict(
            joint_stiffness_nm_per_rad=_joint_table(feedforward["joint_stiffness_nm_per_rad"], _STIFFNESS_SPEC,
                                                    f"{name}.joint_stiffness_nm_per_rad"),
            max_offset_rad=_check_number(feedforward["max_offset_rad"], _FEEDFORWARD_OFFSET_SPEC,
                                         f"{name}.max_offset_rad"),
            provenance=feedforward["provenance"].strip())
    out["gravity_feedforward"] = feedforward
    return out


#: Optional section ``perception``: switches of the SO-101 perception reports (docs/development/so101-integration.md
#: section 13.4). An absent section, an absent key or null takes the default; a deployment without the section keeps
#: the behaviour before the section existed except for the reporting fields marked "default on".
PERCEPTION_FIELDS = {
    "robot_occlusion_report": "bool, default true: every object table of a camera reports value.robot_occlusion, the "
                              "share of the table within the arm's reach that the arm's FK silhouette covers in that "
                              "camera and the earlier rows now under it (perception.external.robot_view; profile tag "
                              "robot_occlusion). Reporting only: no row changes. false removes the field.",
    "contact_top_estimate": "bool, default true: object_table cross_view rows (target_camera_ids) also give "
                            "contact_top_z / contact_top_z_range (m) / contact_fit_iou, the two-view fit of a prism "
                            "standing on the table (perception.external.contact_height; profile tag contact_top; "
                            "needs both cameras). Reporting only: rows are never replaced. false removes the fields.",
    "attach_cross_view": "bool, default false: the object table attached to every observation reply (obs3d) also "
                         "carries cross_view against the wrist camera (one more /object_table call per observation, "
                         "about 0.5 s on a local GPU, 2-3 s through an SSH tunnel; profile tag cross_view_attach; needs "
                         "both cameras). false keeps cross_view on demand (geometry_query target_camera_ids).",
    "clear_view_q_rad": "null (default) or 5 joint angles in rad (shoulder_pan..wrist_roll, URDF convention, inside the "
                        "URDF limits shrunk by motion.joint_limit_margin_rad, TCP inside workspace): a pose whose FK "
                        "silhouette leaves the table visible to the fixed camera, found offline by "
                        "scripts/so101_clear_view_pose.py for this calibration. When set, the motion instructions name "
                        "its TCP pose (FK) as a robot_action target the model may choose; nothing moves on its own. "
                        "null names no pose.",
}
PERCEPTION_DEFAULTS = {"robot_occlusion_report": True, "contact_top_estimate": True, "attach_cross_view": False,
                       "clear_view_q_rad": None}


def validate_perception(value, motion=None):
    """The normalized ``perception`` section (``PERCEPTION_FIELDS``; defaults ``PERCEPTION_DEFAULTS`` for an absent
    section, key or null). ``motion``: the validated ``motion`` section, whose ``joint_limit_margin_rad`` (rad) shrinks
    the packaged URDF limits ``clear_view_q_rad`` must lie in (None: the URDF limits). Example:
    ``validate_perception({"clear_view_q_rad": [0.3, -1.27, 0.41, 0.53, 0.0]})``."""
    if value is None:
        return dict(PERCEPTION_DEFAULTS)
    if not isinstance(value, dict):
        raise ValueError("so101_deployment_invalid:perception: object or null required")
    value = _clean(value)
    unknown = set(value) - set(PERCEPTION_FIELDS)
    if unknown:
        raise ValueError("so101_deployment_invalid:perception: unknown " + ",".join(sorted(unknown)))
    out = dict(PERCEPTION_DEFAULTS)
    for key in ("robot_occlusion_report", "contact_top_estimate", "attach_cross_view"):
        if value.get(key) is not None:
            if type(value[key]) is not bool:
                raise ValueError(f"so101_deployment_invalid:perception.{key}: bool or null required")
            out[key] = value[key]
    q = value.get("clear_view_q_rad")
    if q is not None:
        from ...model.embodiments import physical_facts
        limits = {joint["name"]: joint["declared_limits_rad"] for joint in physical_facts("so101")["kinematics"]["joints"]
                  if joint.get("declared_limits_rad")}
        margin = float((motion or {}).get("joint_limit_margin_rad", 0.0))
        if (not isinstance(q, list) or len(q) != len(ARM_JOINTS)
                or any(type(item) not in (int, float) or not math.isfinite(item) for item in q)):
            raise ValueError("so101_deployment_invalid:perception.clear_view_q_rad: 5 finite joint angles (rad) or null")
        for name, item in zip(ARM_JOINTS, q):
            low, high = limits[name][0] + margin, limits[name][1] - margin
            if not low <= item <= high:
                raise ValueError(f"so101_deployment_out_of_range:perception.clear_view_q_rad.{name}: {item} not in "
                                 f"[{low:.4f}, {high:.4f}] rad (URDF limits minus motion.joint_limit_margin_rad)")
        out["clear_view_q_rad"] = [float(item) for item in q]
    return out


def validate_deployment(document, *, base=None):
    """Return a normalized deployment dict or raise ``ValueError`` naming the bad field."""
    if not isinstance(document, dict):
        raise ValueError("so101_deployment_object_required")
    data = _clean(document)
    if data.get("schema") != SCHEMA:
        raise ValueError(f"so101_deployment_schema_must_be:{SCHEMA}")
    known = {"schema", "python", "port", "robot_id", "calibration_dir", "camera_calibration_file", "calibration_bundle_dir",
             "board_file", "board", "disable_torque_on_disconnect", "max_relative_target_deg", "num_read_retries",
             "joint_mapping", "cameras", "ik_service", "motion_compensation", "perception", *FIELDS}
    unknown = set(data) - known
    if unknown:
        raise ValueError("so101_deployment_unknown_keys:" + ",".join(sorted(unknown)))
    base = Path(base) if base is not None else Path.cwd()

    def path(value, key, optional=False):
        if value is None and optional:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"so101_deployment_invalid:{key}: path string required")
        resolved = Path(value).expanduser()
        return str(resolved if resolved.is_absolute() else (base / resolved).resolve())

    out = {"schema": SCHEMA, "python": path(data.get("python"), "python"),
           "calibration_dir": path(data.get("calibration_dir"), "calibration_dir", optional=True),
           "camera_calibration_file": path(data.get("camera_calibration_file"), "camera_calibration_file", optional=True),
           "calibration_bundle_dir": path(data.get("calibration_bundle_dir"), "calibration_bundle_dir", optional=True),
           "board_file": path(data.get("board_file"), "board_file", optional=True)}
    board_raw = data.get("board")
    if board_raw is None and out["board_file"] is not None:
        if not Path(out["board_file"]).is_file():
            raise ValueError(f"so101_deployment_invalid:board_file: not found: {out['board_file']}")
        try:
            board_raw = _clean(json.loads(Path(out["board_file"]).read_text(encoding="utf-8")))
        except (OSError, ValueError) as error:
            raise ValueError(f"so101_deployment_invalid:board_file: {error}") from error
    if board_raw is not None:
        from ni_rua.robot.calibration.workflow import charuco as cal
        if not isinstance(board_raw, dict):
            raise ValueError("so101_deployment_invalid:board: object required")
        try:
            spec = cal.validate_board_spec({k: v for k, v in board_raw.items() if k != "thickness_m"})
        except ValueError as error:
            raise ValueError(f"so101_deployment_invalid:board: {error}") from error
        if "thickness_m" in board_raw and board_raw["thickness_m"] is not None:
            spec["thickness_m"] = _check_number(board_raw["thickness_m"], _num("m", 0, 0.05, ""), "board.thickness_m")
        out["board"] = spec
    else:
        out["board"] = None
    for key in ("port", "robot_id"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"so101_deployment_invalid:{key}: non-empty string required")
        out[key] = data[key]
    if type(data.get("disable_torque_on_disconnect")) is not bool:
        raise ValueError("so101_deployment_invalid:disable_torque_on_disconnect: bool required")
    out["disable_torque_on_disconnect"] = data["disable_torque_on_disconnect"]
    out["max_relative_target_deg"] = _check_number(data.get("max_relative_target_deg"),
        _num("deg", 0, 90, "", low_open=True), "max_relative_target_deg")
    out["num_read_retries"] = _check_number(data.get("num_read_retries"), _num("count", 0, 20, "", integer=True), "num_read_retries")
    mapping = data.get("joint_mapping")
    if not isinstance(mapping, dict) or set(mapping) != set(ARM_JOINTS):
        raise ValueError("so101_deployment_invalid:joint_mapping: one entry per arm joint required")
    out["joint_mapping"] = {}
    for name in ARM_JOINTS:
        entry = mapping[name]
        if not isinstance(entry, dict) or set(entry) != {"sign", "offset_rad"} or entry["sign"] not in (1, -1):
            raise ValueError(f"so101_deployment_invalid:joint_mapping.{name}: sign (+1/-1) and offset_rad required")
        out["joint_mapping"][name] = dict(sign=float(entry["sign"]), offset_rad=_check_number(
            entry["offset_rad"], _num("rad", -math.pi, math.pi, ""), f"joint_mapping.{name}.offset_rad"))
    cameras = data.get("cameras")
    if not isinstance(cameras, dict) or not cameras or set(cameras) - set(CAMERA_IDS):
        raise ValueError("so101_deployment_invalid:cameras: non-empty subset of wrist/fixed required")
    out["cameras"] = {}
    for name, camera in cameras.items():
        if not isinstance(camera, dict) or set(camera) - set(CAMERA_FIELDS) or {"index_or_path", "width", "height", "fps"} - set(camera):
            raise ValueError(f"so101_deployment_invalid:cameras.{name}: fields {sorted(CAMERA_FIELDS)}")
        index = camera["index_or_path"]
        if not (type(index) is int and index >= 0 or isinstance(index, str) and index.strip()):
            raise ValueError(f"so101_deployment_invalid:cameras.{name}.index_or_path")
        fourcc = camera.get("fourcc")
        if fourcc is not None and (not isinstance(fourcc, str) or len(fourcc) != 4):
            raise ValueError(f"so101_deployment_invalid:cameras.{name}.fourcc")
        device_name = camera.get("device_name")
        if device_name is not None and (not isinstance(device_name, str) or not device_name.strip()):
            raise ValueError(f"so101_deployment_invalid:cameras.{name}.device_name")
        unique_id = camera.get("unique_id")
        if unique_id is not None and (not isinstance(unique_id, str) or not unique_id.strip()):
            raise ValueError(f"so101_deployment_invalid:cameras.{name}.unique_id: non-empty string or null")
        rotate_deg = camera.get("rotate_deg", 0)
        if rotate_deg is None:
            rotate_deg = 0
        if type(rotate_deg) is not int or rotate_deg not in (0, 90, 180, 270):
            raise ValueError(f"so101_deployment_invalid:cameras.{name}.rotate_deg: must be 0, 90, 180 or 270")
        out["cameras"][name] = dict(index_or_path=index,
            device_name=None if device_name is None else device_name.strip(),
            unique_id=None if unique_id is None else unique_id.strip(),
            fourcc=fourcc, rotate_deg=rotate_deg,
            width=_check_number(camera["width"], _num("px", 16, 8192, "", integer=True), f"cameras.{name}.width"),
            height=_check_number(camera["height"], _num("px", 16, 8192, "", integer=True), f"cameras.{name}.height"),
            fps=_check_number(camera["fps"], _num("Hz", 1, 240, "", integer=True), f"cameras.{name}.fps"))
    unique_ids = [camera["unique_id"] for camera in out["cameras"].values() if camera["unique_id"] is not None]
    if len(unique_ids) != len(set(unique_ids)):
        raise ValueError("so101_deployment_invalid:cameras.unique_id: two cameras name the same device")
    out["ik_service"] = validate_ik_service(data.get("ik_service"))
    out["motion_compensation"] = validate_motion_compensation(data.get("motion_compensation"))
    for section, specs in FIELDS.items():
        values = data.get(section)
        if isinstance(values, dict):
            values = {key: spec["default"] for key, spec in specs.items() if "default" in spec} | values
        if not isinstance(values, dict) or set(values) != set(specs):
            missing = sorted(set(specs) - set(values or {}))
            extra = sorted(set(values or {}) - set(specs)) if isinstance(values, dict) else []
            raise ValueError(f"so101_deployment_invalid:{section}: missing={missing} unknown={extra}")
        out[section] = {}
        for key, spec in specs.items():
            name = f"{section}.{key}"
            if spec["kind"] == "list":
                items = values[key]
                if (not isinstance(items, list) or not 1 <= len(items) <= spec["max_size"]
                        or any(type(item) not in (int, float) or not math.isfinite(item)
                               or not spec["low"] <= item <= spec["high"] for item in items)):
                    raise ValueError(f"so101_deployment_invalid:{name}: 1..{spec['max_size']} finite numbers in "
                                     f"[{spec['low']}, {spec['high']}] ({spec['unit']})")
                out[section][key] = [float(item) for item in items]
            elif spec["kind"] == "xy":
                points = values[key]
                if (not isinstance(points, list) or not 1 <= len(points) <= spec["max_size"]
                        or any(not isinstance(point, list) or len(point) != 2
                               or any(type(item) not in (int, float) or not math.isfinite(item)
                                      or not spec["low"] <= item <= spec["high"] for item in point) for point in points)):
                    raise ValueError(f"so101_deployment_invalid:{name}: 1..{spec['max_size']} points [x, y] with each "
                                     f"coordinate in [{spec['low']}, {spec['high']}] ({spec['unit']})")
                out[section][key] = [[float(item) for item in point] for point in points]
            elif spec["kind"] == "table":
                table = values[key]
                if not isinstance(table, dict) or set(table) != set(spec["keys"]):
                    raise ValueError(f"so101_deployment_invalid:{name}: object with exactly the keys "
                                     f"{list(spec['keys'])} required")
                labels = [label for label, _ in spec["columns"]]
                out[section][key] = {}
                for entry in spec["keys"]:
                    row = table[entry]
                    if not isinstance(row, list) or len(row) != len(spec["columns"]):
                        raise ValueError(f"so101_deployment_invalid:{name}.{entry}: list {labels} ({spec['unit']}) required")
                    out[section][key][entry] = [_check_number(item, column, f"{name}.{entry}.{label}")
                                                for item, (label, column) in zip(row, spec["columns"])]
            elif spec["kind"] == "vec":
                vector = values[key]
                if (not isinstance(vector, list) or len(vector) != spec["size"]
                        or any(type(item) not in (int, float) or not math.isfinite(item) for item in vector)):
                    raise ValueError(f"so101_deployment_invalid:{name}: {spec['size']} finite numbers ({spec['unit']})")
                out[section][key] = [float(item) for item in vector]
            else:
                out[section][key] = _check_number(values[key], spec, name)
    gripper, motion = out["gripper"], out["motion"]
    if gripper["lerobot_closed"] == gripper["lerobot_open"] or gripper["jaw_closed_rad"] == gripper["jaw_open_rad"]:
        raise ValueError("so101_deployment_invalid:gripper: closed and open must differ")
    if (gripper["empty_close_aperture_m"] is not None and gripper["full_open_aperture_m"] is not None
            and gripper["empty_close_aperture_m"] >= gripper["full_open_aperture_m"]):
        raise ValueError("so101_deployment_invalid:gripper.empty_close_aperture_m must be below full_open_aperture_m")
    jaw_range = abs(gripper["jaw_open_rad"] - gripper["jaw_closed_rad"])
    squeeze, squeeze_max = gripper["hold_squeeze_rad"], gripper["max_hold_squeeze_rad"]
    if squeeze_max is not None and (squeeze is None or squeeze_max > jaw_range or squeeze > squeeze_max):
        raise ValueError("so101_deployment_invalid:gripper.max_hold_squeeze_rad requires hold_squeeze_rad and "
                         f"hold_squeeze_rad <= max_hold_squeeze_rad <= the jaw range {jaw_range:g} rad")
    if squeeze is not None and squeeze > jaw_range:
        raise ValueError(f"so101_deployment_invalid:gripper.hold_squeeze_rad exceeds the jaw range {jaw_range:g} rad")
    if any(lo >= hi for lo, hi in zip(out["workspace"]["min_m"], out["workspace"]["max_m"])):
        raise ValueError("so101_deployment_invalid:workspace: min_m must be below max_m on every axis")
    if out["calibration"]["handheld_min_views"] > out["calibration"]["handheld_target_views"]:
        raise ValueError("so101_deployment_invalid:calibration.handheld_min_views exceeds handheld_target_views")
    if out["calibration"]["handheld_far_fraction"] >= out["calibration"]["handheld_near_fraction"]:
        raise ValueError("so101_deployment_invalid:calibration.handheld_far_fraction must be below handheld_near_fraction")
    if out["calibration"]["handheld_tilt_small"] > out["calibration"]["handheld_tilt_large"]:
        raise ValueError("so101_deployment_invalid:calibration.handheld_tilt_small exceeds handheld_tilt_large")
    if out["calibration"]["touch_min_z_m"] >= out["calibration"]["touch_hover_z_m"]:
        raise ValueError("so101_deployment_invalid:calibration.touch_min_z_m must be below touch_hover_z_m")
    if out["operator"]["default_speed_scale_cap"] < out["operator"]["min_speed_scale_cap"]:
        raise ValueError("so101_deployment_invalid:operator.default_speed_scale_cap must be at least min_speed_scale_cap")
    if motion["default_timeout_s"] > motion["max_timeout_s"]:
        raise ValueError("so101_deployment_invalid:motion.default_timeout_s exceeds max_timeout_s")
    if motion["default_wait_timeout_s"] > motion["max_timeout_s"]:
        raise ValueError("so101_deployment_invalid:motion.default_wait_timeout_s exceeds max_timeout_s")
    for preset, (low, high) in gripper["preset_margins_m"].items():
        if high is not None and high <= low:
            raise ValueError(f"so101_deployment_invalid:gripper.preset_margins_m.{preset}: max_margin must exceed min_margin")
        if (gripper["empty_close_aperture_m"] is not None and gripper["full_open_aperture_m"] is not None
                and gripper["empty_close_aperture_m"] + low >= gripper["full_open_aperture_m"]):
            raise ValueError(f"so101_deployment_invalid:gripper.preset_margins_m.{preset}: empty_close_aperture_m + "
                             "min_margin must stay below full_open_aperture_m")
    feedforward = out["motion_compensation"]["gravity_feedforward"]
    if feedforward is not None:
        # LeRobot clamps |goal - present| per command (max_relative_target_deg); a clamped goal would silently lose
        # the offsets the receipts report. Largest offset a goal can carry + one command step must fit inside it.
        needed = motion["arrival_correction_max_offset_rad"] + feedforward["max_offset_rad"] + motion["max_command_delta_rad"]
        allowed = math.radians(out["max_relative_target_deg"])
        if needed > allowed:
            raise ValueError("so101_deployment_invalid:motion_compensation.gravity_feedforward: "
                             "motion.arrival_correction_max_offset_rad + max_offset_rad + motion.max_command_delta_rad = "
                             f"{needed:.4f} rad exceeds max_relative_target_deg ({allowed:.4f} rad): the driver would "
                             "clamp compensated goals")
    if (motion["max_acceleration_rad_s2"] is None) != (motion["max_jerk_rad_s3"] is None):
        raise ValueError("so101_deployment_invalid:motion: max_acceleration_rad_s2 and max_jerk_rad_s3 must both be set or both null")
    if motion["max_command_delta_rad"] * motion["command_rate_hz"] < motion["max_joint_speed_rad_s"]:
        raise ValueError("so101_deployment_invalid:motion: max_command_delta_rad * command_rate_hz must reach max_joint_speed_rad_s")
    out["perception"] = validate_perception(data.get("perception"), motion)
    q = out["perception"]["clear_view_q_rad"]
    if q is not None:   # the named pose must be a target robot_action accepts (imports the numpy kinematics)
        from ...model import so101_kinematics as kinematics
        tcp = kinematics.fk(kinematics.load_model(), q)[:3, 3]
        low, high = out["workspace"]["min_m"], out["workspace"]["max_m"]
        if not all(lo <= float(v) <= hi for lo, v, hi in zip(low, tcp, high)):
            raise ValueError("so101_deployment_invalid:perception.clear_view_q_rad: its TCP "
                             f"{[round(float(v), 4) for v in tcp]} m is outside workspace min_m/max_m")
    return out


def load_deployment(path):
    path = Path(path).expanduser().resolve()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(
            f"so101_deployment_unreadable:{path.name}: {error} "
            f"(run `ni-rua-so101 init-config --output {path}` to generate a deployment file)"
        ) from error
    except (OSError, ValueError) as error:
        raise ValueError(f"so101_deployment_unreadable:{path.name}: {error}") from error
    return validate_deployment(document, base=path.parent)


def calibration_bundle_dir(deployment):
    """Directory of the calibration bundle (versions, workflow runs, ``active`` pointer).

    ``calibration_bundle_dir`` when set; otherwise ``calibration/`` next to ``camera_calibration_file``.
    ``None`` when neither is configured (the workflow then refuses to start).
    """
    if deployment.get("calibration_bundle_dir"):
        return Path(deployment["calibration_bundle_dir"])
    if deployment.get("camera_calibration_file"):
        return Path(deployment["camera_calibration_file"]).parent / "calibration"
    return None


def write_joint_offsets(path, offsets_rad, *, stamp=None):
    """Set ``joint_mapping.<joint>.offset_rad`` (rad) in the deployment file ``path``; nothing else changes.

    The raw JSON (including ``_`` comment keys) is edited, the whole document is validated exactly like
    :func:`load_deployment`, the previous file is copied to ``<name>.bak-<UTC>`` and the new file replaces
    the old one with one ``os.replace``. ``offsets_rad``: ``{joint: rad}`` for any subset of ``ARM_JOINTS``.
    Returns ``{"backup": str, "previous_offsets_rad": {...}, "offsets_rad": {...}, "deployment": validated}``.
    Raises ``ValueError`` (unknown joint, non-finite value, document invalid) without touching the file.
    """
    path = Path(path).expanduser().resolve()
    unknown = set(offsets_rad) - set(ARM_JOINTS)
    if unknown:
        raise ValueError("so101_joint_offsets_unknown_joints:" + ",".join(sorted(unknown)))
    text = path.read_text(encoding="utf-8")
    document = json.loads(text)
    mapping = document.get("joint_mapping")
    if not isinstance(mapping, dict):
        raise ValueError("so101_deployment_invalid:joint_mapping: object required")
    previous = {name: (mapping.get(name) or {}).get("offset_rad") for name in ARM_JOINTS}
    for name, value in offsets_rad.items():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"so101_joint_offsets_invalid:{name}: finite rad required")
        if not isinstance(mapping.get(name), dict):
            raise ValueError(f"so101_deployment_invalid:joint_mapping.{name}")
        # JSON preserves a binary float's round trip. Rounding here makes the
        # installed mapping disagree with the accepted candidate/live driver.
        mapping[name]["offset_rad"] = float(value)
    validated = validate_deployment(document, base=path.parent)
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.name}.bak-{stamp}")
    backup.write_text(text, encoding="utf-8")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return {"backup": str(backup), "previous_offsets_rad": previous,
            "offsets_rad": {name: validated["joint_mapping"][name]["offset_rad"] for name in ARM_JOINTS},
            "deployment": validated}


def write_gripper_limits(path, *, jaw_closed_rad=None, jaw_open_rad=None, stamp=None):
    """Set ``gripper.jaw_closed_rad`` and/or ``gripper.jaw_open_rad`` (rad) in the deployment file ``path``.

    The raw JSON (including ``_`` comment keys) is edited, the whole document is validated via
    :func:`validate_deployment`, the previous file is backed up to ``<name>.bak-<UTC>`` and replaced via
    ``os.replace``. Returns ``{"backup": str, "previous": {...}, "gripper": {...}, "deployment": validated}``.
    """
    if jaw_closed_rad is None and jaw_open_rad is None:
        raise ValueError("so101_gripper_limits_empty: specify jaw_closed_rad and/or jaw_open_rad")
    path = Path(path).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    document = json.loads(text)
    gripper = document.get("gripper")
    if not isinstance(gripper, dict):
        raise ValueError("so101_deployment_invalid:gripper: object required")
    previous = {"jaw_closed_rad": gripper.get("jaw_closed_rad"), "jaw_open_rad": gripper.get("jaw_open_rad")}
    for key, value in (("jaw_closed_rad", jaw_closed_rad), ("jaw_open_rad", jaw_open_rad)):
        if value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"so101_gripper_limits_invalid:{key}: finite rad required")
        gripper[key] = round(float(value), 6)
    validated = validate_deployment(document, base=path.parent)
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.name}.bak-{stamp}")
    backup.write_text(text, encoding="utf-8")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return {"backup": str(backup), "previous": previous,
            "gripper": {"jaw_closed_rad": validated["gripper"]["jaw_closed_rad"],
                        "jaw_open_rad": validated["gripper"]["jaw_open_rad"]},
            "deployment": validated}


def check_workspace(deployment, position_m):
    low, high = deployment["workspace"]["min_m"], deployment["workspace"]["max_m"]
    if any(not lo <= value <= hi for value, lo, hi in zip(position_m, low, high)):
        raise ValueError(f"out_of_workspace: target {list(position_m)} m outside so101_base box "
                         f"min_m={low} max_m={high}")


def active_motion_compensation(deployment):
    """The model-visible summary of the ``motion_compensation`` mechanisms this deployment switches on, or None when
    all are off (also for a deployment dict built before the section existed): ``carry_arrival_offset`` /
    ``no_response_stop`` / ``stop_correction_within_tolerance`` / ``arrival_breakaway`` (true only) and ``gravity_feedforward`` = ``{"max_offset_rad"}`` (rad,
    the bound of the feedforward offset a move's ``path_compensation.gravity_feedforward_rad`` can show).

    Identified servo values (joint stiffness, per-joint breakaway bands) and the provenance note are tuning data the
    model cannot act on: they stay in the deployment file (``deployment["motion_compensation"]``, which benches and
    audits read) and never enter ``public_limits``, capabilities, receipts or the motion text."""
    section = deployment.get("motion_compensation") or {}
    active = {key: True for key in ("carry_arrival_offset", "no_response_stop", "stop_correction_within_tolerance")
              if section.get(key)}
    if section.get("arrival_breakaway_rad"):
        active["arrival_breakaway"] = True
    if section.get("gravity_feedforward"):
        active["gravity_feedforward"] = {"max_offset_rad": section["gravity_feedforward"]["max_offset_rad"]}
    return active or None


def public_limits(deployment, speed_scale_cap=1.0):
    """Model-visible motion limits; the same values the worker enforces. ``motion_compensation`` (the mechanisms
    switched on, ``active_motion_compensation``) is present only when the deployment enables one;
    ``max_acceleration_rad_s2`` and ``max_jerk_rad_s3`` are present only when jerk-limited timing is configured."""
    motion = deployment["motion"]
    compensation = active_motion_compensation(deployment)
    units = dict(speed="rad/s", delta="rad", rate="Hz", tracking="rad", time="s", workspace="m in so101_base")
    extra = {}
    if motion.get("max_acceleration_rad_s2") is not None:
        extra["max_acceleration_rad_s2"] = motion["max_acceleration_rad_s2"]
        extra["max_jerk_rad_s3"] = motion["max_jerk_rad_s3"]
        units["acceleration"] = "rad/s^2"
        units["jerk"] = "rad/s^3"
    if compensation:
        extra["motion_compensation"] = compensation
    return dict(max_joint_speed_rad_s=motion["max_joint_speed_rad_s"], speed_scale_cap=speed_scale_cap,
                effective_max_joint_speed_rad_s=motion["max_joint_speed_rad_s"] * speed_scale_cap,
                max_command_delta_rad=motion["max_command_delta_rad"], command_rate_hz=motion["command_rate_hz"],
                gripper_max_speed_rad_s=deployment["gripper"]["max_speed_rad_s"],
                tracking_error_limit_rad=motion["tracking_error_limit_rad"],
                stall_window_s=motion["stall_window_s"], stall_min_progress_rad=motion["stall_min_progress_rad"],
                stall_min_commanded_rad=motion["stall_min_commanded_rad"],
                settle_velocity_rad_s=motion["settle_velocity_rad_s"], settle_hold_s=motion["settle_hold_s"],
                arrival_correction_rounds=motion["arrival_correction_rounds"],
                arrival_correction_max_offset_rad=motion["arrival_correction_max_offset_rad"],
                workspace_min_m=list(deployment["workspace"]["min_m"]), workspace_max_m=list(deployment["workspace"]["max_m"]),
                units=units,
                **extra)


def documentation():
    """Field -> unit/range/meaning, for docs and the example file."""
    return deepcopy(FIELDS)


# Text of profile.stall_guard_semantics (rendered after the preset numbers in the SO-101 robot_action field help and
# the motion section); the worker rule is SO101Adapter._move (design 13 R7 phase 2).
STALL_GUARD_SEMANTICS = ("stops the move when the TCP diverges from the commanded joint path by more than the threshold; "
                         "the joint tracking limit and the stall window stop every move as well. With contact_stop any "
                         "of these stops holds the measured pose and, once settling confirms the hold, judges the "
                         "stage by its arrival rule (position_tolerance_m, orientation_tolerance_rad, max_upward_dz_m): "
                         "inside it the stage completes with contact_stopped_early and later stages, including gripper "
                         "stages, run from the held pose; outside it, and with every other preset, the stage fails with "
                         "tracking_error and contact_block_summary and later stages do not run. There is no force "
                         "sensor: a stop shows obstructed or lagging motion, not proven contact")
# Text of profile.vertical_window_semantics (the SO-101 robot_action help of max_upward_dz_m); the worker rule is
# so101.worker.arrival_verdict (design 13 R6): no default window, no preset or stall guard implies one.
VERTICAL_WINDOW_SEMANTICS = ("Move only: optional vertical arrival window in metres [0, 0.15]: with the horizontal error "
                             "sqrt(dx_m^2 + dy_m^2) <= position_tolerance_m and the orientation within tolerance, a "
                             "measured Z from position_tolerance_m below the target up to max_upward_dz_m above it "
                             "counts as arrived (stage result vertical_compliance_accepted=true and xy_error_m). Default "
                             "0 m (no window) when omitted, also with stall_guard_preset contact_stop. An acceptance rule "
                             "only: the servos still drive toward the target (no vertical compliance or force control). "
                             "The window used is reported as effective_tolerances.max_upward_dz_m in stage_plans and "
                             "stage_results.")
APERTURE_READING = ("fingertip gap: CAD opening (so101_gripper_geometry.opening_width_m) of the measured jaw angle after "
                    "the jaw settles; an empty close reads empty_close_aperture_m, not 0")
HOME_CHECK_VERSION = "measured-so101-gripper-frame-home-v1"


def hold_squeeze(deployment):
    """``(default_rad, max_rad)`` of the jaw holding squeeze (rad of the jaw joint), or None when the deployment sets
    no ``gripper.hold_squeeze_rad`` (no squeeze is applied). ``max_rad``: ``gripper.max_hold_squeeze_rad``, else the
    jaw range ``|jaw_open_rad - jaw_closed_rad|``. ``.get``: a deployment dict built before the keys existed is off.
    Example: ``hold_squeeze(load_deployment("configs/so101.local.json"))`` -> ``(0.02, 1.919863)`` or None."""
    gripper = deployment["gripper"]
    default = gripper.get("hold_squeeze_rad")
    if default is None:
        return None
    limit = gripper.get("max_hold_squeeze_rad")
    return float(default), float(abs(gripper["jaw_open_rad"] - gripper["jaw_closed_rad"]) if limit is None else limit)


def embodiment_profile(deployment):
    """The ``EmbodimentProfile`` of one validated deployment (``validate_deployment`` / ``load_deployment`` output).

    Pure: packaged SO-101 facts (``robot_information_data/so101/physical.json``, CAD from
    ``so101_gripper_geometry``) and this deployment only. No worker, no NI-RUA configuration and no ``features.*``
    thresholds (decision Q4), so the parent backend and the worker build the same profile from the same file.
    Imports numpy lazily (CAD geometry); importing this module stays numpy-free.

    Deployment -> profile: ``gripper.empty_close_aperture_m`` / ``full_open_aperture_m`` (null: CAD opening at
    ``jaw_open_rad``, 0.1 mm) / ``preset_margins_m`` / ``hold_squeeze_rad`` + ``max_hold_squeeze_rad``
    (``gripper.hold_squeeze``, None without the first); ``motion.command_rate_hz`` (control_hz),
    ``default_position_tolerance_m`` + ``default_orientation_tolerance_rad`` (default_tolerance),
    ``tolerance_presets``, ``default_timeout_s`` / ``default_wait_timeout_s`` / ``max_timeout_s``,
    ``arrival_correction_rounds`` + ``arrival_correction_max_offset_rad`` (arrival_correction_limits and the
    ``arrival_correction`` tag, both absent when either is 0); ``cameras`` ids.
    Example: ``embodiment_profile(load_deployment("configs/so101.local.json")).gripper.full_open_aperture_m``.
    """
    from ...model import embodiments as e
    from ...model import so101_gripper_geometry as cad
    facts = e.physical_facts("so101")
    identity, kinematics = facts["identity"], facts["kinematics"]
    if tuple(kinematics["arm_joint_order"]) != ARM_JOINTS:
        raise ValueError("so101_packaged_joint_order_differs_from_deployment_joints")
    arms = tuple(identity["public_arm_slots"])
    gripper, motion = deployment["gripper"], deployment["motion"]
    full_open, source = gripper["full_open_aperture_m"], "deployment gripper.full_open_aperture_m"
    if full_open is None:
        full_open, source = round(cad.opening_width_m(gripper["jaw_open_rad"]), 4), "CAD opening at gripper.jaw_open_rad"
    geometry = cad.derive_gripper_geometry()
    speed, gripper_speed = e.ScaleRange(*SPEED_SCALE_RANGE, 1.0), e.ScaleRange(*GRIPPER_SPEED_SCALE_RANGE, 1.0)
    cameras = tuple(sorted(deployment["cameras"]))
    two_views = "fixed" in cameras and "wrist" in cameras
    perception = {**PERCEPTION_DEFAULTS, **(deployment.get("perception") or {})}
    shared = motion_thresholds()
    # Sag-correction bounds: the deployment's own (default and ceiling of the per-move fields); a deployment that
    # corrects nothing (0 rounds or a 0 rad offset) offers no arrival_correction fields at all.
    rounds, offset = int(motion["arrival_correction_rounds"]), float(motion["arrival_correction_max_offset_rad"])
    correction = e.ArrivalCorrectionLimits(rounds, offset) if rounds > 0 and offset > 0 else None
    return e.EmbodimentProfile(
        robot=e.SO101, tcp_link=kinematics["tcp_link"], tcp_label=kinematics["tcp_link"],
        limits_profiles=(LIMITS_PROFILE,), arms=arms, max_simultaneous_arms=len(arms),
        arm_joint_names=e.FrozenDict((arm, tuple(kinematics["arm_joint_order"])) for arm in arms),
        jaw_joint_names=e.FrozenDict((arm, tuple(kinematics["gripper_joint_order"])) for arm in arms),
        stages_per_arm=e.MAX_STAGES_PER_ARM, max_stages=e.MAX_BATCH_STAGES,
        gripper=e.GripperProfile(
            empty_close_aperture_m=gripper["empty_close_aperture_m"], full_open_aperture_m=full_open,
            full_open_source=source,
            preset_margins_m=e.FrozenDict(none=(None, None), **{
                name: tuple(gripper["preset_margins_m"][name]) for name in GRIP_GUARD_PRESET_NAMES}),
            aperture_reading=APERTURE_READING,
            fingertip_beyond_reference_m=geometry["protrusion_beyond_tcp_m"]["max"],
            grasp_reference="jaw_fingertip_midpoint", grasp_reference_offset_m=None,
            approach_axis_local=tuple(float(v) for v in geometry["approach_axis_tcp"]),
            jaw_opening_axis_local=(-1.0, 0.0, 0.0), mask_segment=None, contact_sections=None,
            # Grasp-centre pose input (grasp_centre_m + object_width_m; contract.grasp_centre_pose): its CAD bounds.
            jaw_grasp=e.JawGraspGeometry(**cad.jaw_grasp_facts(gripper["jaw_open_rad"])),
            # Bounded holding squeeze of closing jaw stages (SO101Adapter._hold_squeeze) and the grip_squeeze_rad field.
            hold_squeeze=None if hold_squeeze(deployment) is None else e.HoldSqueeze(*hold_squeeze(deployment))),
        pose_inputs=frozenset({"tcp_quaternion", "jaw_grasp_centre"}),
        # gripper_motion / max_upward_dz: the during-motion jaw ramp and the vertical arrival window (design 13 R5, R6;
        # SO101Adapter._segment / arrival_verdict). reach: robot_action mode=reach (SO101Adapter.reach_batch).
        # approach_roll: split approach / roll orientation tolerances, which only the cuRobo IK service can solve
        # (ik_plan.curobo_solve), so they are offered only when the deployment configures ik_service.
        # held_perception: FK robot pixel mask from the packaged SO-101 meshes and the held-object row
        # (perception.external.held_perception.SO101Body). cross_view: object_table target_camera_ids also report the
        # two-view check of the rows against the target camera's masks (object_table.cross_view_rows); it needs two
        # calibrated cameras, so it is offered only with both the overview and the wrist camera.
        # arrival_correction: per-stage bounds of the sag correction (SO101Adapter._arrive), up to the deployment's
        # motion.arrival_correction_rounds / arrival_correction_max_offset_rad (arrival_correction_limits), offered
        # when the deployment corrects at all. path_check / collision_world
        # / obstacle_object_ids: the preview's cuRobo path collision check and the obstacle and held-object boxes of
        # its collision world (ik_plan.collision_world), which exist only with ik_service.
        # robot_occlusion / contact_top / cross_view_attach: the deployment's perception switches (validate_perception;
        # a deployment dict without the section takes PERCEPTION_DEFAULTS); the two cross-view ones need both cameras.
        supports=frozenset({"stall_guard", "settle", "execute_if_clean", "gripper_motion", "max_upward_dz", "reach",
                            "held_perception", *(("arrival_correction",) if correction is not None else ()),
                            *(("cross_view",) if "fixed" in cameras and "wrist" in cameras else ()),
                            *(("robot_occlusion",) if perception["robot_occlusion_report"] else ()),
                            *(("contact_top",) if perception["contact_top_estimate"] and two_views else ()),
                            *(("cross_view_attach",) if perception["attach_cross_view"] and two_views else ()),
                            *(("approach_roll", "path_check", "collision_world", "obstacle_object_ids")
                              if deployment.get("ik_service") else ())}),
        stall_guard_semantics=STALL_GUARD_SEMANTICS, vertical_window_semantics=VERTICAL_WINDOW_SEMANTICS,
        speed_scale=speed, gripper_speed_scale=gripper_speed,
        speed_presets=e.presets_within(e.SPEED_PRESETS, speed),
        gripper_speed_presets=e.presets_within(e.GRIPPER_SPEED_PRESETS, gripper_speed),
        tolerance_presets=e.FrozenDict((name, e.Tolerance(*motion["tolerance_presets"][name]))
                                       for name in TOLERANCE_PRESET_NAMES),
        default_tolerance_preset="normal",
        default_tolerance=e.Tolerance(motion["default_position_tolerance_m"], motion["default_orientation_tolerance_rad"]),
        max_position_tolerance_m=MAX_POSITION_TOLERANCE_M, max_orientation_tolerance_rad=math.pi,
        stall_guard_presets_m=e.stall_guard_presets(shared),
        tracking_divergence_range_m=(shared["stall_guard_min_divergence_m"], shared["stall_guard_max_divergence_m"]),
        clock="real_time", control_hz=motion["command_rate_hz"], step_domain=STEP_DOMAIN, step_label="command period",
        per_request_step_limit=None, default_action_timeout_s=motion["default_timeout_s"],
        default_wait_timeout_s=motion["default_wait_timeout_s"], max_timeout_s=motion["max_timeout_s"],
        cameras=cameras, overview_camera="fixed" if "fixed" in cameras else None,
        wrist_cameras=e.FrozenDict({arms[0]: "wrist"} if "wrist" in cameras else {}),
        camera_examples=tuple(camera for camera in ("fixed", "wrist") if camera in cameras),
        support_surface_z_m=None, handoff=False,
        home_check=e.HomeCheckProfile(HOME_CHECK_VERSION, e.SO101.frame_id, e.SO101.tcp_config_id, arms),
        policy_step=False, arrival_correction_limits=correction)
