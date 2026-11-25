"""SO-101 motion bench: one fixed, repeatable protocol on the real arm under ONE deployment file.

Run it once per condition (baseline, carried offset, gravity feedforward, servo integral gain, no-response stop and
combinations). A condition is nothing but a deployment file: copy ``configs/so101.local.json``, change the setting
under test and pass the copy with a label. The protocol, the code path and the metrics stay the same.

Execution path: the agent's own. The bench builds a ``Harness`` over the real ``so101_real`` backend (worker
subprocess in the deployment's interpreter) and sends every motion as a ``robot_action`` preview + execute through
``unified_actions.dispatch`` -> ``Harness.execute``. No model, no perception service (``features.external_perception``
off), no motion outside that entry point. Run it in the ``ni-rua`` environment with the gateway stopped (one process
owns the serial bus) and the table clear except for the cube of the carry test.

Protocol (per repetition; all positions m in so101_base)
  park      Move to the TCP pose of the deployment ``home.q_rad`` (also before and after every test).
  transit   Empty gripper. For every ``--transit-heights-m`` z and ``--transit-reaches-m`` r (horizontal distance
            from the shoulder_pan axis): one batch [to (r, -a, z), sweep to (r, +a, z), sweep back] with a =
            ``--transit-azimuth-deg``; and per height one radial batch [to (r_min, 0, z), out to (r_max, 0, z), back].
            Plus one batch ``gravity_sign`` [to ``--sign-cross-high-xz-m``, down-and-out to ``--sign-cross-low-xz-m``,
            back]: a retracted high pose and an extended low pose (both at y = 0, defaults 0.30 m and 0.12 m high, far
            above the table) between which the shoulder's gravity torque changes sign. A carried offset is constant
            along a move, so an offset learned on one side is wrong-signed on the other: this pair is where
            ``carry_arrival_offset`` alone can be WORSE than the baseline. Each transit stage records the planned
            gravity torques at its start and target (``gravity_torque``, N*m) and ``sign_change_joints``.
  carry     (with ``--cube-xy-m``) One batch: open, above the cube, down, close, lift, over ``--carry-over-xy-m``
            at ``--carry-height-m``, back above the cube, down to ``--place-clearance-m`` above its rest height,
            open, up.
            Grasp poses are given as grasp_centre_m + object_width_m + fixed_jaw_yaw_deg, as the agent gives them.
  contact   (with ``--contact-xy-m`` and ``--contact-tcp-z-m``) Close the jaw, hover, then descend with
            stall_guard_preset contact_stop (stop at ``--contact-divergence-m`` of TCP divergence, arrival correction
            off) toward ``--contact-overtravel-m`` below the given contact height, hold for ``--contact-dwell-s``
            (robot_wait) and lift. Present_Load is sampled from the worker's operator telemetry
            (``telemetry/state.json``) during the whole test. The commanded target must lie inside the deployment
            ``workspace`` box, whose floor is normally just above the table: put a rigid block at least
            overtravel + 10 mm high (e.g. 40 mm) under the contact point and give the TCP z at which the closed
            fingertips touch ITS top. The bench refuses a target below the workspace floor before anything moves.

Metrics (``bench-<condition>.json``)
  Per move stage, from its stage result: ``transit`` (max_below_path_m, max_deviation_m and their fractions: FK of
  the measured joint samples against the commanded path), ``residual_before_m`` / ``residual_after_m`` (TCP position
  error at the first settle and at the verdict; dz of both), ``correction_rounds``, ``unanswered_rounds``,
  ``termination_reason``, ``duration_s`` (wall clock), ``tracking_stop`` (guard trip kind) and ``carry``.
  Contact test: ``peak_load_pct`` / ``hold_load_pct`` / ``baseline_load_pct`` per loaded joint (percent of the
  servo's maximum torque, uncalibrated Present_Load; not a force), ``below_contact_m`` (contact height minus the
  lowest measured TCP z; > 0 = the TCP went below the given contact height) and ``held_above_contact_m``.
  ``summary``: mean, sample standard deviation, min, max and n of every numeric metric over the repetitions, and
  counts of every categorical one.

IK: the worker plans with the deployment's solver. A deployment with ``ik_service`` needs its cuRobo tunnel up
(the gateway normally opens it); without it every preview fails with ``ik_service_unreachable`` and the bench stops
at the first park having moved nothing. For a bench without the GPU host, remove ``ik_service`` from EVERY
condition file (packaged local IK), so all conditions plan the same way.

Load sampling follows ``telemetry.joint_hz`` / ``telemetry.motor_status_hz`` of the deployment (template 10 / 1 Hz).
Set ``telemetry.motor_status_hz`` to 10 in EVERY condition file before comparing peak loads; the bench records both.

Commands (repository root; ``D=~/.cache/ni-rua/so101/bench``; table at z=-0.020 m on the development unit, so a
29 mm cube has its centre at z=-0.0055 and the closed fingertips touch a 40 mm block near TCP z=0.027: measure yours)

  # 0. identification for the feedforward / breakaway conditions (so101 interpreter, writes a proposal only)
  PYTHONPATH=. <so101 python> scripts/so101_gravity_identify.py --deployment configs/so101.local.json \\
      --breakaway --out $D/identify.json

  # 1. one deployment copy per condition; edit ONLY the listed keys
  #    baseline        configs/so101.local.json unchanged (no motion_compensation section)
  #    m1-carry        "motion_compensation": {"carry_arrival_offset": true}
  #    m3-feedforward  "motion_compensation": {"gravity_feedforward": <proposal.motion_compensation.gravity_feedforward>}
  #    m4-integral     "servo": {"i_coefficient": 4}   (keep p and d; try 2, 4, 8 as separate conditions)
  #    m5-no-response  "motion_compensation": {"no_response_stop": true}
  #    m5-breakaway    "motion_compensation": {"no_response_stop": true, "stop_correction_within_tolerance": true,
  #                                            "arrival_breakaway_rad": <proposal ...>}
  #                    (validation requires stop_correction_within_tolerance with a breakaway band; the enlarged
  #                    step can overshoot by up to band - error, i.e. band x reach at the tool: 0.02 rad x 0.3 m =
  #                    6 mm, so run the carry with --place-clearance-m at least that large for this condition)
  #    m6-within-tol   "motion_compensation": {"stop_correction_within_tolerance": true}
  #    m1+m3+m5        all three keys in one section

  # 2. the same command for every condition, changing only --deployment and --condition
  mamba run -n ni-rua python scripts/so101_motion_bench.py run --deployment $D/so101.baseline.json \\
      --condition baseline --out $D/baseline --repetitions 5 \\
      --cube-xy-m 0.22 0.06 --cube-centre-z-m -0.0055 --carry-over-xy-m 0.24 -0.08 --carry-height-m 0.08 \\
      --contact-xy-m 0.20 0.00 --contact-tcp-z-m 0.027      # closed fingertips on a 40 mm block

  # 3. compare: mean +- stdev per stage and metric, one column per condition
  mamba run -n ni-rua python scripts/so101_motion_bench.py compare $D/*/bench-*.json

First run of any new condition: keep the default ``--speed-scale 0.5`` and one repetition, hand on the power
switch; raise the speed only for conditions that ran clean.

What to look at per condition
  m1  ``transit/gravity_sign/*`` first: if ``max_below_path_m`` or ``max_deviation_m`` there is above the baseline,
      the constant carried offset is hurting across the torque sign change; judge M1 together with M3.
  m3  the contact rows (``contact peak_load_pct.*``, ``below_contact_m``): the feedforward also acts while pressing,
      so a stopped press pushes harder than the baseline by about the arm's own weight. A clearly higher peak load
      is the price of the transit gain.
  m4  the same contact rows (integral windup) and ``duration_s``.

Reading the comparison: a mechanism helps when the mean of ``max_below_path_m`` (transit and the loaded carry
stages) drops by clearly more than the standard deviations of both conditions, without more guard trips, longer
``duration_s`` or a higher contact ``peak_load_pct``. Five repetitions give a rough standard deviation only; repeat
a close call with more.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCHEMA = "ni_rua.so101.motion_bench.v1"
ARM = "right"
#: Arm joints whose Present_Load the contact test reports (the gravity-loaded ones that press on a support).
LOAD_JOINTS = ("shoulder_lift", "elbow_flex", "wrist_flex")
#: Numeric per-stage metrics that ``summarise`` aggregates over repetitions.
NUMERIC = ("max_below_path_m", "max_below_at_fraction", "max_deviation_m", "max_deviation_at_fraction",
           "residual_before_m", "residual_before_dz_m", "residual_after_m", "residual_after_dz_m",
           "correction_rounds", "unanswered_rounds", "duration_s")
CATEGORICAL = ("status", "termination_reason", "tracking_stop", "carry")


# ------------------------------------------------------------------ protocol ---

def pan_axis_xy(model):
    """Position of the shoulder_pan axis in so101_base (m): reaches are measured from it."""
    joint = next(item for item in model["joints"] if item["name"] == "shoulder_pan")
    return float(joint["T_parent_joint"][0][3]), float(joint["T_parent_joint"][1][3])


def polar_point(model, reach_m, azimuth_deg, z_m):
    """``[x, y, z]`` (m, so101_base) at horizontal distance ``reach_m`` from the shoulder_pan axis, ``azimuth_deg``
    from base +x about +z, height ``z_m``."""
    x0, y0 = pan_axis_xy(model)
    angle = math.radians(azimuth_deg)
    return [round(x0 + reach_m * math.cos(angle), 5), round(y0 + reach_m * math.sin(angle), 5), float(z_m)]


def move_stage(identifier, position_m, yaw_deg, arguments, **fields):
    """A public robot_action move stage to a TCP position with the straight-down orientation at ``yaw_deg``
    (``contract.straight_down_quaternion``); the orientation only binds below ``--orientation-tolerance-rad`` pi."""
    from ni_rua.robot.backends.so101.contract import straight_down_quaternion
    return dict(id=identifier, kind="move", arm=ARM, position_m=[float(v) for v in position_m],
                quaternion_xyzw=straight_down_quaternion(float(yaw_deg)),
                position_tolerance_m=arguments.position_tolerance_m,
                orientation_tolerance_rad=arguments.orientation_tolerance_rad, speed_scale=arguments.speed_scale, **fields)


def park_action(arguments, model, deployment):
    """The park move: the TCP pose of the deployment home joints (FK), reached like any other move."""
    from ni_rua.robot.backends.so101.worker import matrix_to_quaternion
    from ni_rua.robot.model import so101_kinematics
    pose = so101_kinematics.fk(model, deployment["home"]["q_rad"])
    return [dict(id="park", kind="move", arm=ARM, position_m=[float(v) for v in pose[:3, 3]],
                 quaternion_xyzw=[float(v) for v in matrix_to_quaternion(pose[:3, :3])],
                 position_tolerance_m=arguments.position_tolerance_m,
                 orientation_tolerance_rad=arguments.orientation_tolerance_rad, speed_scale=arguments.speed_scale)]


def transit_actions(arguments, model):
    """``[(action name, stages)]`` of the empty-gripper transits (module docstring, ``transit``)."""
    actions, azimuth = [], arguments.transit_azimuth_deg
    for z in arguments.transit_heights_m:
        for reach in arguments.transit_reaches_m:
            name = f"sweep_z{round(z * 1000)}_r{round(reach * 1000)}"
            actions.append((name, [
                move_stage("approach", polar_point(model, reach, -azimuth, z), -azimuth, arguments),
                move_stage("sweep_out", polar_point(model, reach, azimuth, z), azimuth, arguments),
                move_stage("sweep_back", polar_point(model, reach, -azimuth, z), -azimuth, arguments)]))
        near, far = min(arguments.transit_reaches_m), max(arguments.transit_reaches_m)
        if far > near:
            actions.append((f"radial_z{round(z * 1000)}", [
                move_stage("approach", polar_point(model, near, 0.0, z), 0.0, arguments),
                move_stage("reach_out", polar_point(model, far, 0.0, z), 0.0, arguments),
                move_stage("reach_back", polar_point(model, near, 0.0, z), 0.0, arguments)]))
        if arguments.sign_cross_high_xz_m and arguments.sign_cross_low_xz_m and z == arguments.transit_heights_m[0]:
            (high_x, high_z), (low_x, low_z) = arguments.sign_cross_high_xz_m, arguments.sign_cross_low_xz_m
            actions.append(("gravity_sign", [
                move_stage("approach", [high_x, 0.0, high_z], 0.0, arguments),
                move_stage("cross_out", [low_x, 0.0, low_z], 0.0, arguments),
                move_stage("cross_back", [high_x, 0.0, high_z], 0.0, arguments)]))
    return actions


def torque_facts(model, inertials, start_q, end_q):
    """Planned gravity torques of one move (N*m, ``so101_gravity.gravity_torques`` at the planned joints) for
    ``LOAD_JOINTS`` at its start and target, and the joints whose torque changes sign between them."""
    from ni_rua.robot.embodiments.so101.gravity import gravity_torques
    from ni_rua.robot.backends.so101.driver import ARM_JOINTS
    start, end = gravity_torques(model, inertials, start_q), gravity_torques(model, inertials, end_q)
    index = {name: ARM_JOINTS.index(name) for name in LOAD_JOINTS}
    return {"start_nm": {name: round(float(start[i]), 4) for name, i in index.items()},
            "end_nm": {name: round(float(end[i]), 4) for name, i in index.items()},
            "sign_change_joints": [name for name, i in index.items() if start[i] * end[i] < 0]}


def carry_stages(arguments):
    """Stages of the cube carry (module docstring, ``carry``): grasp-centre poses at the cube's yaw."""
    x, y = arguments.cube_xy_m
    over_x, over_y = arguments.carry_over_xy_m
    low, high = arguments.cube_centre_z_m, arguments.carry_height_m

    def grasp(identifier, centre, **fields):
        return dict(id=identifier, kind="move", arm=ARM, grasp_centre_m=[float(v) for v in centre],
                    object_width_m=arguments.cube_width_m, fixed_jaw_yaw_deg=arguments.cube_yaw_deg,
                    position_tolerance_m=arguments.position_tolerance_m,
                    orientation_tolerance_rad=arguments.grasp_orientation_tolerance_rad,
                    speed_scale=arguments.speed_scale, **fields)
    close = dict(id="close", kind="gripper", arm=ARM, goal_normalized=0.0)
    if arguments.grip_guard_preset:
        close["grip_guard_preset"] = arguments.grip_guard_preset
    return [dict(id="open", kind="gripper", arm=ARM, goal_normalized=1.0),
            grasp("above_cube", [x, y, high]), grasp("down_to_cube", [x, y, low]), close,
            grasp("lift", [x, y, high]), grasp("carry_over", [over_x, over_y, high]),
            grasp("carry_back", [x, y, high]),
            # Set-down: the target stays --place-clearance-m above the rest height, so neither the streamed goals nor
            # the arrival correction (which only moves the goal toward this target) drive the cube into the table.
            grasp("set_down", [x, y, low + arguments.place_clearance_m]),
            dict(id="release", kind="gripper", arm=ARM, goal_normalized=1.0),
            grasp("retreat", [x, y, high])]


def contact_stages(arguments):
    """``(descent stages, lift stages)`` of the contact test (module docstring, ``contact``)."""
    x, y = arguments.contact_xy_m
    contact, hover = arguments.contact_tcp_z_m, arguments.contact_tcp_z_m + arguments.contact_hover_m
    yaw = math.degrees(math.atan2(y, x))
    descent = [dict(id="close", kind="gripper", arm=ARM, goal_normalized=0.0),
               move_stage("hover", [x, y, hover], yaw, arguments),
               # The path is commanded below the contact so the stop signal ends it; the window accepts the hold.
               # No arrival correction: a descent the guard did not stop must not push on with a shifted goal.
               move_stage("press", [x, y, contact - arguments.contact_overtravel_m], yaw, arguments,
                          stall_guard_preset="contact_stop", max_tracking_divergence_m=arguments.contact_divergence_m,
                          arrival_correction_rounds=0,
                          max_upward_dz_m=arguments.contact_overtravel_m + arguments.position_tolerance_m)]
    return descent, [move_stage("lift", [x, y, hover], yaw, arguments)]


# ------------------------------------------------------------------- metrics ---

def stage_metrics(result):
    """Bench metrics of one robot_action stage result (worker ``_move`` fields); absent facts stay None.

    ``residual_before_m`` / ``residual_before_dz_m``: TCP position error and its z part at the first settle, before
    any correction (``arrival_correction.initial_tcp_residual``); ``residual_after_m`` / ``residual_after_dz_m``: at
    the verdict. ``correction_rounds``: rounds sent; ``unanswered_rounds``: rounds with ``responded`` False.
    ``tracking_stop``: the guard that stopped the move (``joint_tracking_limit``, ``tcp_divergence``, ``stall``) or
    None. ``carry``: where the carried offset came from (``path_compensation.carry``) or None."""
    correction = result.get("arrival_correction") or {}
    initial = correction.get("initial_tcp_residual") or {}
    transit = result.get("transit") or {}
    rounds = correction.get("rounds") or []
    return {"status": result.get("status"), "kind": result.get("kind"),
            "max_below_path_m": transit.get("max_below_path_m"),
            "max_below_at_fraction": transit.get("max_below_at_fraction"),
            "max_deviation_m": transit.get("max_deviation_m"),
            "max_deviation_at_fraction": transit.get("max_deviation_at_fraction"),
            "residual_before_m": initial.get("position_error_m"), "residual_before_dz_m": initial.get("dz_m"),
            "residual_after_m": result.get("position_error_m"), "residual_after_dz_m": result.get("dz_m"),
            "correction_rounds": len(rounds) if "arrival_correction" in result else None,
            "unanswered_rounds": (sum(1 for entry in rounds if entry.get("responded") is False)
                                  if "arrival_correction" in result else None),
            "termination_reason": correction.get("termination_reason"),
            "final_goal_offset_rad": correction.get("final_goal_offset_rad"),
            "duration_s": result.get("duration_s"),
            "tracking_stop": (result.get("tracking_stop") or {}).get("kind"),
            "contact_stopped_early": result.get("contact_stopped_early"),
            "carry": (result.get("path_compensation") or {}).get("carry"),
            "carried_offset_rad": (result.get("path_compensation") or {}).get("carried_offset_rad"),
            "gravity_feedforward_rad": (result.get("path_compensation") or {}).get("gravity_feedforward_rad")}


def contact_metrics(samples, press_window, hold_window, contact_tcp_z_m):
    """Load and depth facts of one contact test from telemetry samples.

    ``samples``: ``[{"t": monotonic s, "load": {joint: percent of max torque or None}, "tcp_z_m": float or None}]``
    (worker ``telemetry/state.json`` reads). ``press_window`` / ``hold_window``: ``(start, end)`` monotonic s of the
    descent action and of the dwell. Samples before the press are the free-air baseline. Returns per ``LOAD_JOINTS``
    joint ``baseline_load_pct`` (mean), ``peak_load_pct`` (largest magnitude during press and hold, signed) and
    ``hold_load_pct`` (mean during the dwell); ``lowest_tcp_z_m``; ``below_contact_m`` = ``contact_tcp_z_m`` -
    lowest measured TCP z during press and hold (> 0: below the given contact height); ``held_above_contact_m`` =
    mean TCP z during the dwell - ``contact_tcp_z_m``; and the sample counts. A value without samples is None."""
    def inside(window):
        return [sample for sample in samples if window[0] <= sample["t"] <= window[1]]
    before = [sample for sample in samples if sample["t"] < press_window[0]]
    press, hold = inside(press_window), inside(hold_window)
    loaded = press + hold

    def loads(group, joint):
        return [sample["load"][joint] for sample in group if (sample.get("load") or {}).get(joint) is not None]

    def mean(values):
        return round(statistics.fmean(values), 3) if values else None
    out = {"samples": {"baseline": len(before), "press": len(press), "hold": len(hold)},
           "baseline_load_pct": {}, "peak_load_pct": {}, "hold_load_pct": {}}
    for joint in LOAD_JOINTS:
        out["baseline_load_pct"][joint] = mean(loads(before, joint))
        peak = loads(loaded, joint)
        out["peak_load_pct"][joint] = round(max(peak, key=abs), 3) if peak else None
        out["hold_load_pct"][joint] = mean(loads(hold, joint))
    heights = [sample["tcp_z_m"] for sample in loaded if sample.get("tcp_z_m") is not None]
    held = [sample["tcp_z_m"] for sample in hold if sample.get("tcp_z_m") is not None]
    out["lowest_tcp_z_m"] = round(min(heights), 5) if heights else None
    out["below_contact_m"] = round(contact_tcp_z_m - min(heights), 5) if heights else None
    out["held_above_contact_m"] = round(statistics.fmean(held) - contact_tcp_z_m, 5) if held else None
    return out


def _flatten(prefix, value, out):
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else key, item, out)
    elif type(value) in (int, float):
        out[prefix] = float(value)


def summarise(repetitions):
    """Aggregate the repetitions: ``{"<test>/<action>/<stage>": {metric: {mean, stdev, min, max, n}, ...,
    "counts": {categorical metric: {value: count}}}}`` plus ``"contact": {metric: {...}}``. ``stdev`` is the sample
    standard deviation (None with one value). A metric that is None in a repetition does not count toward ``n``."""
    def stats(values):
        return {"mean": round(statistics.fmean(values), 6),
                "stdev": round(statistics.stdev(values), 6) if len(values) > 1 else None,
                "min": round(min(values), 6), "max": round(max(values), 6), "n": len(values)}
    numeric, counts, contact = {}, {}, {}
    for repetition in repetitions:
        for test, actions in (repetition.get("tests") or {}).items():
            for action in actions:
                for stage_id, metrics in (action.get("stages") or {}).items():
                    key = f"{test}/{action['name']}/{stage_id}"
                    for name in NUMERIC:
                        if metrics.get(name) is not None:
                            numeric.setdefault(key, {}).setdefault(name, []).append(float(metrics[name]))
                    for name in CATEGORICAL:
                        bucket = counts.setdefault(key, {}).setdefault(name, {})
                        bucket[str(metrics.get(name))] = bucket.get(str(metrics.get(name)), 0) + 1
                if action.get("contact"):
                    flat = {}
                    _flatten("", {key: value for key, value in action["contact"].items() if key != "samples"}, flat)
                    for name, value in flat.items():
                        contact.setdefault(name, []).append(value)
    summary = {key: {**{name: stats(values) for name, values in metrics.items()}, "counts": counts.get(key, {})}
               for key, metrics in numeric.items()}
    for key in counts:
        summary.setdefault(key, {"counts": counts[key]})
    if contact:
        summary["contact"] = {name: stats(values) for name, values in contact.items()}
    return summary


# ------------------------------------------------------------------- session ---

class TelemetrySampler:
    """Polls the worker's operator telemetry (``read()`` -> the ``state.json`` document or None) on a thread and keeps
    one ``{"t", "load", "tcp_z_m"}`` entry per new telemetry sequence number. ``period_s``: polling period (s).
    ``entry(state)`` builds the fields kept besides ``t`` (monotonic s); a bench that samples other telemetry
    (``so101_grip_bench``) overrides it."""

    def __init__(self, read, period_s=0.05):
        self.read, self.period_s, self.samples = read, period_s, []
        self._stop, self._seen, self._thread = threading.Event(), None, None

    def poll(self):
        try:
            state = self.read()
        except (OSError, ValueError):
            return
        if not isinstance(state, dict) or state.get("seq") == self._seen:
            return
        self._seen = state.get("seq")
        self.samples.append({"t": time.monotonic(), **self.entry(state)})

    def entry(self, state):
        position = state.get("tcp_position_m")
        return {"telemetry_monotonic_s": state.get("monotonic_s"),
                "load": {joint: ((state.get("motor") or {}).get(joint) or {}).get("load") for joint in LOAD_JOINTS},
                "tcp_z_m": None if not position else float(position[2])}

    def __enter__(self):
        def loop():
            while not self._stop.is_set():
                self.poll()
                self._stop.wait(self.period_s)
        self._thread = threading.Thread(target=loop, daemon=True, name="bench-telemetry")
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=2)
        self.poll()


class Session:
    """What the bench needs from a run: the ``Harness``, a telemetry reader and a cleanup (test seam)."""

    def __init__(self, harness, read_telemetry, close):
        self.harness, self.read_telemetry, self.close = harness, read_telemetry, close


def open_session(arguments, out):
    """A ``Harness`` over the real SO-101 backend for this deployment (worker subprocess), without a model and
    without external perception; the run store and the worker directory live under ``out``."""
    from ni_rua.core.config import Config
    from ni_rua.core.contracts import new_id
    from ni_rua.core.storage.store import Store
    from ni_rua.robot.backends.factory import create_backend
    from ni_rua.runtime.harness import Harness
    config = Config.from_dict({
        "backend": {"environment": "so101_real", "robot_model_id": "so101",
                    "deployment": str(Path(arguments.deployment).expanduser().resolve())},
        "features": {"observation_profile": arguments.observation_profile, "external_perception": False}})
    identity = dict(run_id=out.name, episode_id=new_id("ep"), env_instance_id=new_id("so101-bench"), env_generation=1)
    store = Store(out / "store", identity)
    robot_dir = out / "private" / "robot"
    backend = create_backend(config=config, run_dir=robot_dir, observation_profile=arguments.observation_profile)
    harness = Harness(store, backend, observation_profile=arguments.observation_profile, config=config)
    harness.start()
    harness.task_planning.required = False  # no model plans here; the protocol is fixed

    def read_telemetry():
        return json.loads((robot_dir / "telemetry" / "state.json").read_text())

    def close():
        errors = []
        for step in (harness.request_stop, backend.close, store.close):
            try:
                step()
            except Exception as error:  # cleanup must reach every step
                errors.append(f"{type(error).__name__}: {error}")
        return errors
    return Session(harness, read_telemetry, close)


def run_action(session, name, stages, *, metrics=None, **request):
    """Preview then execute one robot_action batch through the shared tool path, resuming at checkpoints.

    Returns ``{"name", "preview_status", "status", "reason", "wall_clock_s", "worker_duration_s", "stages":
    {stage id: stage_metrics}, "window": (monotonic start, end)}``; a failed preview has ``failure`` and no stages.
    ``metrics``: the function turning one stage result into its ``stages`` entry (default ``stage_metrics``)."""
    metrics = metrics or stage_metrics
    from ni_rua.runtime.actions.unified_actions import dispatch
    harness = session.harness
    started = time.monotonic()
    record = {"name": name, "stages": {}}
    preview = dispatch(harness, dict(observation_id=harness.latest["observation_id"], stages=stages, **request),
                       {"tool_call_id": f"bench-preview-{started}"}).data
    record["preview_status"] = preview.get("status")
    if preview.get("status") != "planned":
        record.update(status="not_executed", reason=preview.get("reason"),
                      failure={key: preview.get(key) for key in ("failed_stage_id", "failure_feedback", "issues")
                               if preview.get(key) is not None})
    else:
        record["path_check"] = (preview.get("path_check") or {}).get("status")
        # Planned joints of every move stage (rad), in stage order: what the gravity-torque facts are computed from.
        record["planned_q_rad"] = {stage["id"]: [float(v) for v in preview["stage_plans"][stage["id"]]["q_rad"]]
                                   for stage in stages if stage["kind"] == "move"
                                   and "q_rad" in (preview.get("stage_plans") or {}).get(stage["id"], {})}
        duration, attempt = 0.0, 0
        while True:
            attempt += 1
            receipt = dispatch(harness, dict(mode="execute", plan_id=preview["plan_id"]),
                               {"tool_call_id": f"bench-execute-{started}-{attempt}"}).data.get("receipt") or {}
            duration += float(receipt.get("wall_clock_duration_s") or 0.0)
            for stage_id, result in (receipt.get("stage_results") or {}).items():
                # A resumed receipt lists stages completed before the pause with their kind, arm and status only:
                # the entry recorded when the stage ran is kept.
                bare = stage_id in record["stages"] and set(result) <= {"kind", "arm", "status"}
                if result.get("status") not in (None, "pending") and not bare:
                    record["stages"][stage_id] = metrics(result)
            if not receipt.get("paused_at_checkpoint") or attempt > len(stages):
                break
        record.update(status=receipt.get("status"), reason=receipt.get("reason"), worker_duration_s=round(duration, 3),
                      **({"failed_stage_id": receipt["failed_stage_id"]} if receipt.get("failed_stage_id") else {}))
    ended = time.monotonic()
    record.update(wall_clock_s=round(ended - started, 3), window=(started, ended))
    return record


def run_wait(session, dwell_s, rate_hz):
    """Hold the last commanded goals for ``dwell_s`` through ``robot_wait``; returns the monotonic window."""
    harness = session.harness
    started = time.monotonic()
    action = harness._action_from_args("robot_wait", dict(native_steps=max(1, int(round(dwell_s * rate_hz))),
                                                          observation_id=harness.latest["observation_id"],
                                                          action_id=f"bench-wait-{started}"))
    receipt = harness.execute(action)["receipt"]
    return receipt.get("status"), (started, time.monotonic())


def source_binding(out):
    """The source revision this bench ran from; an uncommitted tree is saved as ``source.patch`` next to the report."""
    def git(*command):
        return subprocess.run(["git", *command], cwd=ROOT, capture_output=True, text=True, check=False).stdout
    binding = {"revision": git("rev-parse", "HEAD").strip() or None, "dirty": bool(git("status", "--porcelain").strip())}
    if binding["dirty"]:
        (out / "source.patch").write_text(git("diff", "HEAD"))
        binding["patch"] = "source.patch"
    return binding


# ----------------------------------------------------------------------- run ---

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run the protocol under one deployment file.")
    run.add_argument("--deployment", required=True, help="Deployment file of this condition (read only).")
    run.add_argument("--condition", required=True, help="Label of this condition (file name and report field).")
    run.add_argument("--out", required=True, help="New or empty output directory.")
    run.add_argument("--repetitions", type=int, default=5, help="Repetitions of the whole protocol; default 5.")
    run.add_argument("--tests", nargs="+", default=["transit", "carry", "contact"],
                     choices=("transit", "carry", "contact"),
                     help="Tests to run; carry and contact are skipped without their positions.")
    run.add_argument("--observation-profile", default="rgb_proprio_v1",
                     help="Observation profile of the run; default rgb_proprio_v1 (no camera calibration needed).")
    run.add_argument("--speed-scale", type=float, default=0.5,
                     help="robot_action speed_scale of every move, [0.25, 1]; default 0.5 for first runs of a "
                          "condition. Compare conditions at the same value; 1 is the agent's default.")
    run.add_argument("--sign-cross-high-xz-m", type=float, nargs=2, default=[0.15, 0.30],
                     help="x z (m, so101_base, y = 0) of the retracted high pose of the gravity_sign transit; default "
                          "0.15 0.30 (shoulder gravity torque about -0.2 N*m with the local IK).")
    run.add_argument("--sign-cross-low-xz-m", type=float, nargs=2, default=[0.28, 0.12],
                     help="x z (m) of its extended low pose; default 0.28 0.12 (shoulder torque about +0.26 N*m; "
                          "14 cm above a table at z=-0.02).")
    run.add_argument("--position-tolerance-m", type=float, default=0.01,
                     help="Arrival position tolerance of every move (m); default 0.01 (the normal preset).")
    run.add_argument("--orientation-tolerance-rad", type=float, default=3.14159,
                     help="Orientation tolerance of park, transit and contact moves (rad); default 3.14159 = position "
                          "only, reachable at every reach. A value below pi requires the straight-down pose.")
    run.add_argument("--transit-heights-m", type=float, nargs="+", default=[0.06, 0.12],
                     help="TCP heights of the transits (m, so101_base z); default 0.06 0.12.")
    run.add_argument("--transit-reaches-m", type=float, nargs="+", default=[0.18, 0.24, 0.30],
                     help="Horizontal distances from the shoulder_pan axis (m); default 0.18 0.24 0.30.")
    run.add_argument("--transit-azimuth-deg", type=float, default=30.0,
                     help="Half-angle of the lateral sweeps about the pan axis (deg); default 30.")
    run.add_argument("--cube-xy-m", type=float, nargs=2, help="Cube centre x y (m, so101_base); enables the carry test.")
    run.add_argument("--cube-centre-z-m", type=float, help="Cube centre z when it rests on the table (m, so101_base).")
    run.add_argument("--cube-width-m", type=float, default=0.029, help="Cube width along the jaw line (m); default 0.029.")
    run.add_argument("--cube-yaw-deg", type=float, default=90.0,
                     help="fixed_jaw_yaw_deg of the grasp (deg about base +z from +x); default 90 (jaw line across "
                          "the arm's radial direction near y=0). A yaw inside the wrist_roll band or a carry height "
                          "the straight-down pose cannot reach fails the preview: the test is then not executed.")
    run.add_argument("--carry-over-xy-m", type=float, nargs=2, help="Point the cube is carried over, x y (m).")
    run.add_argument("--carry-height-m", type=float, help="Grasp-centre height during the carry (m, so101_base z).")
    run.add_argument("--place-clearance-m", type=float, default=0.002,
                     help="Height above its rest height at which the cube is released (m); default 0.002.")
    run.add_argument("--grasp-orientation-tolerance-rad", type=float, default=0.15,
                     help="Orientation tolerance of the carry moves (rad); default 0.15 (the normal preset).")
    run.add_argument("--grip-guard-preset", default=None,
                     help="grip_guard_preset of the close stage (e.g. rigid_object); default none.")
    run.add_argument("--contact-xy-m", type=float, nargs=2, help="Table point of the contact test, x y (m).")
    run.add_argument("--contact-tcp-z-m", type=float,
                     help="TCP z at which the CLOSED fingertips touch the table (m, so101_base; from touch-off).")
    run.add_argument("--contact-hover-m", type=float, default=0.04,
                     help="Height of the hover pose above the contact height (m); default 0.04.")
    run.add_argument("--contact-overtravel-m", type=float, default=0.03,
                     help="How far below the contact height the descent is commanded (m); default 0.03. It must "
                          "exceed --contact-divergence-m, or the guard never stops the descent.")
    run.add_argument("--contact-divergence-m", type=float, default=0.02,
                     help="max_tracking_divergence_m of the press (m, [0.005, 0.15]): TCP distance from the commanded "
                          "path at which the descent stops; default 0.02 (the contact_stop preset's value).")
    run.add_argument("--contact-dwell-s", type=float, default=2.0,
                     help="Hold time on the table after the contact stop (s); default 2.")
    compare = commands.add_parser("compare", help="Print mean +- stdev per stage and metric for several reports.")
    compare.add_argument("reports", nargs="+", help="bench-<condition>.json files.")
    compare.add_argument("--metrics", nargs="+", default=["max_below_path_m", "max_deviation_m", "residual_before_m",
                                                          "residual_after_m", "correction_rounds", "duration_s"],
                         help="Per-stage metrics to print.")
    return parser


def run(arguments, session_factory=open_session):
    """Run the protocol; ``session_factory(arguments, out)`` -> ``Session`` (test seam). Returns the report."""
    from ni_rua.robot.backends.so101.deployment import active_motion_compensation, load_deployment
    from ni_rua.robot.model import so101_kinematics
    if arguments.repetitions < 1 or not 0.25 <= arguments.speed_scale <= 1:
        raise SystemExit("--repetitions >= 1 and --speed-scale in [0.25, 1] required")
    out = Path(arguments.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"{out} is not empty: every run uses a fresh directory")
    deployment = load_deployment(arguments.deployment)
    model = so101_kinematics.load_model()
    from ni_rua.robot.embodiments.so101.gravity import load_inertials
    inertials = load_inertials()
    tests = list(arguments.tests)
    skipped = {}
    if "carry" in tests and None in (arguments.cube_xy_m, arguments.cube_centre_z_m, arguments.carry_over_xy_m,
                                     arguments.carry_height_m):
        tests.remove("carry")
        skipped["carry"] = "needs --cube-xy-m, --cube-centre-z-m, --carry-over-xy-m and --carry-height-m"
    if "contact" in tests and None in (arguments.contact_xy_m, arguments.contact_tcp_z_m):
        tests.remove("contact")
        skipped["contact"] = "needs --contact-xy-m and --contact-tcp-z-m"
    if "contact" in tests:
        floor = deployment["workspace"]["min_m"][2]
        if arguments.contact_overtravel_m <= arguments.contact_divergence_m:
            raise SystemExit("--contact-overtravel-m must exceed --contact-divergence-m")
        if arguments.contact_tcp_z_m - arguments.contact_overtravel_m < floor:
            raise SystemExit(f"contact target z {arguments.contact_tcp_z_m - arguments.contact_overtravel_m:.4f} m is below "
                             f"the deployment workspace floor {floor} m: press on a rigid block (module docstring)")
    report = {"schema": SCHEMA, "condition": arguments.condition,
              "started_at": datetime.now(timezone.utc).isoformat(),
              "deployment": str(Path(arguments.deployment).expanduser().resolve()),
              "settings": {"servo": dict(deployment["servo"]),
                           # Full deployment values (stiffness, bands, provenance): an operator report.
                           "motion_compensation": (deployment["motion_compensation"]
                                                   if active_motion_compensation(deployment) else None),
                           "arrival_correction": {key: deployment["motion"][key] for key in (
                               "arrival_correction_rounds", "arrival_correction_min_error_rad",
                               "arrival_correction_max_offset_rad")},
                           "tracking": {key: deployment["motion"][key] for key in (
                               "max_joint_speed_rad_s", "command_rate_hz", "tracking_error_limit_rad", "stall_window_s",
                               "stall_min_progress_rad", "stall_min_commanded_rad")},
                           "operator_speed_scale_cap": deployment["operator"]["default_speed_scale_cap"],
                           "telemetry_hz": {"joint": deployment["telemetry"]["joint_hz"],
                                            "motor_status": deployment["telemetry"]["motor_status_hz"]}},
              "source": source_binding(out), "protocol": {key: value for key, value in vars(arguments).items()},
              "tests": tests, "skipped_tests": skipped, "repetitions": [], "aborted": None}
    path = out / f"bench-{arguments.condition}.json"

    def save():
        report["summary"] = summarise(report["repetitions"])
        path.write_text(json.dumps(report, indent=2, default=str))
    save()
    session = session_factory(arguments, out)
    rate = deployment["motion"]["command_rate_hz"]
    park = park_action(arguments, model, deployment)

    class Stop(RuntimeError):
        pass

    def must(record, what, *, unplanned_ok=False):
        """Stop the bench unless the action completed; ``unplanned_ok``: a preview that did not plan moved nothing,
        so the test is only recorded as not executed."""
        if record["status"] != "completed" and not (unplanned_ok and record["status"] == "not_executed"):
            raise Stop(f"{what}: {record['status']} {record.get('reason')}")
        return record

    try:
        for index in range(arguments.repetitions):
            entry = {"index": index, "tests": {}}
            report["repetitions"].append(entry)
            entry["park"] = must(run_action(session, "park", park), "park")
            if "transit" in tests:
                actions = entry["tests"]["transit"] = []
                for name, stages in transit_actions(arguments, model):
                    record = run_action(session, name, stages)
                    planned = list((record.get("planned_q_rad") or {}).items())
                    record["gravity_torque"] = {stage_id: torque_facts(model, inertials, planned[i - 1][1], q)
                                                for i, (stage_id, q) in enumerate(planned) if i > 0}
                    actions.append(record)
                    print(f"[{arguments.condition} #{index}] transit {name}: {record['status']} {record.get('reason') or ''}",
                          flush=True)
                    if record["status"] not in ("completed", "not_executed"):
                        # A stopped transit leaves the arm held somewhere on its path: park before the next one.
                        must(run_action(session, "park", park), f"park after {name}")
                    save()
                must(run_action(session, "park", park), "park after transit")
            if "carry" in tests:
                record = run_action(session, "carry", carry_stages(arguments))
                entry["tests"]["carry"] = [record]
                print(f"[{arguments.condition} #{index}] carry: {record['status']} {record.get('reason') or ''}", flush=True)
                save()
                # The cube's state is unknown after a failed carry: an operator has to look before anything moves.
                must(record, "carry", unplanned_ok=True)
                must(run_action(session, "park", park), "park after carry")
            if "contact" in tests:
                descent, lift = contact_stages(arguments)
                with TelemetrySampler(session.read_telemetry) as sampler:
                    time.sleep(1.0 / max(deployment["telemetry"]["motor_status_hz"], 1.0))  # a free-air baseline sample
                    record = run_action(session, "contact", descent)
                    hold_window = None
                    if record["status"] == "completed":
                        _, hold_window = run_wait(session, arguments.contact_dwell_s, rate)
                    lifted = run_action(session, "contact_lift", lift) if record["status"] != "not_executed" else None
                press = record["window"]
                record["contact"] = contact_metrics(sampler.samples, press, hold_window or (press[1], press[1]),
                                                    arguments.contact_tcp_z_m)
                entry["tests"]["contact"] = [record] + ([lifted] if lifted else [])
                print(f"[{arguments.condition} #{index}] contact: {record['status']} {json.dumps(record['contact'])}",
                      flush=True)
                save()
                must(record, "contact", unplanned_ok=True)
                if lifted:
                    must(lifted, "contact lift")
                must(run_action(session, "park", park), "park after contact")
            save()
    except Stop as error:
        report["aborted"] = str(error)
        print("ABORT:", error, flush=True)
    except BaseException as error:
        report["aborted"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["cleanup_errors"] = session.close()
        save()
    print("REPORT", path, "aborted=", report["aborted"], flush=True)
    return report


def compare(arguments):
    """Print, per stage and metric, ``mean +- stdev (n)`` of every report side by side; returns the table rows."""
    reports = [json.loads(Path(item).expanduser().read_text()) for item in arguments.reports]
    labels = [report["condition"] for report in reports]
    keys = sorted({key for report in reports for key in report.get("summary", {})})
    rows = []

    def report_n(text):
        """Total count of one categorical cell (JSON text of {value: count})."""
        return sum((json.loads(text) or {}).values())

    def cell(stats):
        if not stats:
            return "-"
        spread = "" if stats.get("stdev") is None else f" +- {stats['stdev']:.4g}"
        return f"{stats['mean']:.4g}{spread} (n={stats['n']})"
    for key in keys:
        names = (sorted({name for report in reports for name in report["summary"].get(key, {})}) if key == "contact"
                 else arguments.metrics)
        for name in names:
            values = [report["summary"].get(key, {}).get(name) for report in reports]
            if any(values):
                rows.append((key, name, [cell(value) for value in values]))
        if key != "contact":
            for name in ("tracking_stop", "termination_reason", "status"):
                # Shown when the conditions differ, and always for a guard trip.
                counts = [json.dumps((report["summary"].get(key, {}).get("counts") or {}).get(name), sort_keys=True)
                          for report in reports]
                tripped = name == "tracking_stop" and any(text not in ("null", '{"None": %d}' % report_n(text))
                                                          for text in counts)
                if len(set(counts)) > 1 or tripped:
                    rows.append((key, name, ["-" if text == "null" else text for text in counts]))
    width = max([len(f"{key} {name}") for key, name, _ in rows] + [10])
    print(" " * width, *(f"{label:>28s}" for label in labels))
    for key, name, cells in rows:
        print(f"{key + ' ' + name:<{width}s}", *(f"{text:>28s}" for text in cells))
    return rows


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    return compare(arguments) if arguments.command == "compare" else run(arguments)


if __name__ == "__main__":
    main()
