"""SO-101 grip bench: holding load, current and temperature of the gripper servo, and whether a held cube stays in
the jaw during a transport, under ONE deployment file.

Why: the jaw servo (STS3215) is position-controlled. A closing stage commands the closed limit, the jaw stops on the
object (or on the other fingertip when empty) and the servo then pushes at its torque limit for as long as the goal is
held. On 2026-11-16 that heated the gripper servo to 46 degC within about ten carry cycles (the arm joints read
31-33 degC) and the servo stopped answering until the arm was power-cycled. Deployment ``gripper.hold_squeeze_rad``
bounds the held goal to the measured contact angle minus a squeeze (rad of the jaw joint; docs/development/
so101-integration.md section 4.13). This bench measures what a squeeze value costs (load, current, temperature rise)
and what it risks (the cube slipping), so the value is chosen from data.

A condition is nothing but a deployment file: copy ``configs/so101.local.json``, change ``gripper.hold_squeeze_rad``
and pass the copy with a label. The protocol, the code path and the metrics stay the same.

Execution path: the agent's own, shared with ``scripts/so101_motion_bench.py`` (its ``open_session`` / ``run_action``):
a ``Harness`` over the real ``so101_real`` backend, every motion a ``robot_action`` preview + execute through
``unified_actions.dispatch`` -> ``Harness.execute``. No model, no perception service. Run it in the ``ni-rua``
environment with the gateway stopped (one process owns the serial bus).

Protocol (per repetition; positions m in so101_base)
  gate      Before every hold: wait, jaw open and unloaded, until the gripper servo reads at most
            ``--start-temperature-c`` (worker telemetry). Not reached within ``--cool-timeout-s``: the bench stops.
            Every hold therefore starts from the same temperature and none starts on a hot servo.
  empty     At the park pose: open, close on nothing, hold ``--hold-s``, open.
  cube      (with ``--grasp-centre-m``) open, above the cube, down, close; hold ``--hold-s`` on the table; then the
            transport: lift to ``--carry-height-m``, lateral move over ``--carry-over-xy-m`` and back at
            ``--transport-speed-scale`` (default 1: the agent's normal speed), set down ``--place-clearance-m`` above
            the rest height; release; retreat. Grasp poses are grasp_centre_m + object_width_m + fixed_jaw_yaw_deg.
  During every hold the gripper temperature is watched: at ``--abort-temperature-c`` the jaw is opened at once and
  the bench stops (the cube is still on the table at that point).

Metrics (``grip-<condition>.json``; all from worker telemetry ``telemetry/state.json`` and stage results)
  hold / transport windows: ``load_pct`` (gripper Present_Load, percent of the servo's maximum torque, uncalibrated,
  not a force; mean, peak magnitude, last), ``current_ma`` (Present_Current, mA; absent when the driver does not read
  it), ``temperature_c`` (start, end, max, rise), ``aperture_m`` (CAD fingertip gap of the measured jaw angle; start,
  end, min, change) and ``squeeze_remaining_rad`` (measured jaw angle minus the held jaw goal, rad: what the servo is
  still pushing against).
  close stage: ``gripper_execution`` (measured opening, ``hold_squeeze`` with the contact angle, the hold goal and the
  loads at contact and at the hold goal), ``grip_facts`` and ``grip_guard``.
  slip: ``aperture_at_close_m``, ``aperture_before_release_m`` (the release stage's reading before it opens),
  ``jaw_closed_further_m`` (their difference: how far the jaw closed between the grasp and the release),
  ``jaw_closed_beyond_threshold`` (that difference above ``--slip-aperture-m``), ``transport_aperture_change_m`` (sum
  of the transport stages' ``aperture_delta_m``), ``squeeze_remaining_rad`` per transport stage (``grip_hold``) and
  ``recheck`` (a transport stage the post-grasp re-check or a guard paused, with its reason). The jaw reading cannot
  tell a lost cube from a cube squeezed further: WATCH the transport and note what you saw next to the report.
  ``summary``: mean, sample standard deviation, min, max and n of every numeric metric over the repetitions.

Sampling follows ``telemetry.motor_status_hz`` of the deployment (template 1 Hz: one load / temperature sample per
second). Use the same value in every condition file; the bench records it.

Commands (repository root; ``D=~/.cache/ni-rua/so101/grip``; a 29 mm cube whose centre rests at z=-0.0055 on the
development unit: measure yours)

  # 1. one deployment copy per condition; edit ONLY gripper.hold_squeeze_rad
  #    baseline    "hold_squeeze_rad": null     (the jaw pushes toward the closed limit: the behaviour being fixed)
  #    sq-005      "hold_squeeze_rad": 0.005    (about 0.4 mm of fingertip gap at a 29 mm opening)
  #    sq-010      "hold_squeeze_rad": 0.010    (about 0.8 mm)
  #    sq-020      "hold_squeeze_rad": 0.020    (about 1.6 mm)
  #    sq-040      "hold_squeeze_rad": 0.040    (about 3.3 mm)

  # 2. gentlest condition first, the baseline LAST (it is the one that overheats); same command, changing only
  #    --deployment and --condition
  mamba run -n ni-rua python scripts/so101_grip_bench.py run --deployment $D/so101.sq-010.json \\
      --condition sq-010 --out $D/sq-010 --repetitions 3 --hold-s 60 \\
      --grasp-centre-m 0.22 0.06 -0.0055 --object-width-m 0.029 --fixed-jaw-yaw-deg 90 \\
      --carry-over-xy-m 0.24 -0.08 --carry-height-m 0.08

  # 3. compare: mean +- stdev per metric, one column per condition
  mamba run -n ni-rua python scripts/so101_grip_bench.py compare $D/*/grip-*.json

Cooling and the temperature gate
  The gate is part of every run: nothing is held until the gripper servo has cooled to ``--start-temperature-c``
  (default 36 degC, a few degrees above the 31-33 degC the idle joints read on the development unit: set it to your
  idle reading + 3). Between conditions simply start the next command; it waits at its first gate with the jaw open,
  printing the temperature. If it times out (``--cool-timeout-s``, default 900 s), leave the arm parked with the jaw
  open and run again later; do not raise the limit to get going. ``--abort-temperature-c`` (default 42 degC) is below
  the 46 degC read before the servo dropped off the bus; keep it there. A servo that reports no temperature stops
  the bench (the gate never guesses). After a ``servo_missing`` fault power-cycle the arm and let it cool before
  anything else.

Reading the comparison: pick the smallest squeeze whose cube transports show no jaw closing (``jaw_closed_further_m``
near 0, ``squeeze_remaining_rad`` near the squeeze at every transport stage) and no slip seen by eye, and whose hold
``load_pct`` / ``temperature_c.rise`` are clearly below the baseline's. Three repetitions give a rough spread only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import so101_motion_bench as motion_bench  # noqa: E402  (shared session, action and sampler helpers)

SCHEMA = "ni_rua.so101.grip_bench.v1"
ARM = motion_bench.ARM
#: Stage ids of ``motion_bench.carry_stages`` that form each part of the cube test.
GRASP_STAGES = ("open", "above_cube", "down_to_cube", "close")
TRANSPORT_STAGES = ("lift", "carry_over", "carry_back", "set_down")
RELEASE_STAGES = ("release", "retreat")
#: Keys of a jaw stage's ``gripper_execution`` the report keeps.
JAW_EXECUTION_FIELDS = ("goal_normalized", "measured_jaw_rad", "measured_opening_m", "measured_normalized", "jaw_settled",
                        "hold_squeeze")


# ------------------------------------------------------------------- metrics ---

def grip_stage_facts(result):
    """What the bench keeps of one robot_action stage result: ``status``, ``kind``, ``checkpoint_reason``, the jaw
    facts (``gripper_execution`` fields of ``JAW_EXECUTION_FIELDS``, ``grip_facts``, ``grip_guard``, ``grip_hold``)
    and, for a move, the measured TCP ``position_error_m`` (m). Absent facts are left out."""
    facts = {"status": result.get("status"), "kind": result.get("kind")}
    for key in ("checkpoint_reason", "grip_facts", "grip_guard", "grip_hold", "position_error_m"):
        if result.get(key) is not None:
            facts[key] = result[key]
    jaw = result.get("gripper_execution")
    if isinstance(jaw, dict):
        facts["gripper_execution"] = {key: jaw[key] for key in JAW_EXECUTION_FIELDS if key in jaw}
    return facts


class GripSampler(motion_bench.TelemetrySampler):
    """``TelemetrySampler`` keeping the gripper servo's telemetry: ``load_pct`` (Present_Load, percent of maximum
    torque), ``current_ma`` (Present_Current, mA), ``temperature_c`` (degC), ``jaw_rad`` / ``jaw_goal_rad`` (measured
    and held jaw angle, rad) and ``aperture_m`` (CAD fingertip gap of the measured jaw angle, m). Unread values are
    None."""

    def entry(self, state):
        from ni_rua.robot.model.so101_gripper_geometry import opening_width_m
        motor = (state.get("motor") or {}).get("gripper") or {}
        jaw = (state.get("q_rad") or {}).get("gripper")
        goal = (state.get("q_goal_rad") or {}).get("gripper")
        return {"load_pct": motor.get("load"), "current_ma": motor.get("current_ma"),
                "temperature_c": motor.get("temperature_c"), "jaw_rad": jaw, "jaw_goal_rad": goal,
                "aperture_m": None if jaw is None else round(float(opening_width_m(jaw)), 5)}


def window_metrics(samples, window):
    """Gripper facts of the samples inside ``window`` (``(start, end)`` monotonic s; ``GripSampler`` entries).

    ``load_pct`` / ``current_ma``: ``mean``, ``peak`` (largest magnitude, signed) and ``last``; ``temperature_c``:
    ``start``, ``end``, ``max`` and ``rise`` (end - start); ``aperture_m``: ``start``, ``end``, ``min`` and
    ``change`` (end - start; negative = the jaw closed); ``squeeze_remaining_rad``: ``mean`` and ``last`` of the
    measured jaw angle minus the held jaw goal (rad). A quantity without samples is None; ``samples`` is the count."""
    inside = [sample for sample in samples if window[0] <= sample["t"] <= window[1]]

    def values(key):
        return [sample[key] for sample in inside if sample.get(key) is not None]

    def load(key):
        series = values(key)
        return None if not series else {"mean": round(statistics.fmean(series), 3),
                                        "peak": round(max(series, key=abs), 3), "last": round(series[-1], 3)}
    temperature, aperture = values("temperature_c"), values("aperture_m")
    remaining = [sample["jaw_rad"] - sample["jaw_goal_rad"] for sample in inside
                 if sample.get("jaw_rad") is not None and sample.get("jaw_goal_rad") is not None]
    return {"samples": len(inside), "duration_s": round(window[1] - window[0], 3),
            "load_pct": load("load_pct"), "current_ma": load("current_ma"),
            "temperature_c": None if not temperature else {
                "start": temperature[0], "end": temperature[-1], "max": max(temperature),
                "rise": round(temperature[-1] - temperature[0], 3)},
            "aperture_m": None if not aperture else {
                "start": aperture[0], "end": aperture[-1], "min": min(aperture),
                "change": round(aperture[-1] - aperture[0], 5)},
            "squeeze_remaining_rad": None if not remaining else {
                "mean": round(statistics.fmean(remaining), 5), "last": round(remaining[-1], 5)}}


def slip_facts(grasp, transport, release, slip_aperture_m):
    """Did the jaw close between the grasp and the release? From the three action records of the cube test.

    ``aperture_at_close_m``: the close stage's settled reading (after the hold squeeze). ``aperture_before_release_m``:
    the release stage's reading before it opened (``measured_aperture_m - aperture_delta_m``).
    ``jaw_closed_further_m``: their difference (> 0: the jaw closed during hold and transport).
    ``jaw_closed_beyond_threshold``: that difference above ``slip_aperture_m`` (m); None when either reading is
    missing. ``transport_aperture_change_m``: sum of the transport stages' ``aperture_delta_m``.
    ``squeeze_remaining_rad``: ``{transport stage: grip_hold.squeeze_remaining_rad}``. ``recheck``: ``[{stage,
    reason}]`` of transport stages a grip guard or the post-grasp re-check paused. The jaw alone cannot tell a lost
    cube from one squeezed further, so nothing here is called "slipped"."""
    def stage(record, identifier):
        return ((record or {}).get("stages") or {}).get(identifier) or {}
    at_close = (stage(grasp, "close").get("grip_facts") or {}).get("measured_aperture_m")
    released = stage(release, "release").get("grip_facts") or {}
    before_release = (None if released.get("measured_aperture_m") is None or released.get("aperture_delta_m") is None
                      else round(released["measured_aperture_m"] - released["aperture_delta_m"], 6))
    closed = None if at_close is None or before_release is None else round(at_close - before_release, 6)
    deltas = [(stage(transport, identifier).get("grip_facts") or {}).get("aperture_delta_m") for identifier in TRANSPORT_STAGES]
    return {"aperture_at_close_m": at_close, "aperture_before_release_m": before_release, "jaw_closed_further_m": closed,
            "slip_aperture_m": slip_aperture_m,
            "jaw_closed_beyond_threshold": None if closed is None else bool(closed > slip_aperture_m),
            "transport_aperture_change_m": (round(sum(deltas), 6) if deltas and None not in deltas else None),
            "squeeze_remaining_rad": {identifier: stage(transport, identifier)["grip_hold"]["squeeze_remaining_rad"]
                                      for identifier in TRANSPORT_STAGES if "grip_hold" in stage(transport, identifier)},
            "recheck": [{"stage": identifier, "reason": stage(transport, identifier)["checkpoint_reason"]}
                        for identifier in TRANSPORT_STAGES if stage(transport, identifier).get("checkpoint_reason")]}


def summarise(repetitions):
    """``{"<test>.<metric path>": {mean, stdev, min, max, n}}`` over the repetitions, for every numeric value of each
    test's ``close`` facts, ``hold`` and ``transport`` windows and ``slip`` facts (``stdev``: sample standard
    deviation, None with one value). Booleans and missing values do not count."""
    series = {}
    for repetition in repetitions:
        for test in ("empty", "cube"):
            entry = repetition.get(test) or {}
            flat = {}
            motion_bench._flatten(test, {key: entry.get(key) for key in ("close", "hold", "transport", "slip")}, flat)
            for name, value in flat.items():
                series.setdefault(name, []).append(value)
    return {name: {"mean": round(statistics.fmean(values), 6),
                   "stdev": round(statistics.stdev(values), 6) if len(values) > 1 else None,
                   "min": round(min(values), 6), "max": round(max(values), 6), "n": len(values)}
            for name, values in sorted(series.items())}


# ------------------------------------------------------------------ protocol ---

def gripper_temperature_c(state):
    """Gripper servo temperature (degC) of one telemetry document, or None when it carries none."""
    value = (((state or {}).get("motor") or {}).get("gripper") or {}).get("temperature_c")
    return float(value) if type(value) in (int, float) else None


def cool_gate(read_telemetry, limit_c, timeout_s, *, period_s=5.0, sleep=time.sleep, now=time.monotonic, report=print):
    """Wait until the gripper servo reads at most ``limit_c`` (degC). Returns ``{"limit_c", "start_c", "end_c",
    "waited_s", "ok", "reason"}``; ``ok`` False after ``timeout_s`` (``reason`` ``cool_timeout``) or when the
    telemetry carries no gripper temperature (``temperature_unknown``: the gate never guesses). ``period_s``: time
    between readings (s). Nothing is commanded here: the caller leaves the jaw open and unloaded."""
    started, first, value = now(), None, None
    while True:
        try:
            value = gripper_temperature_c(read_telemetry())
        except (OSError, ValueError):
            value = None
        first = value if first is None else first
        waited = now() - started
        if value is not None and value <= limit_c:
            return dict(limit_c=limit_c, start_c=first, end_c=value, waited_s=round(waited, 1), ok=True, reason=None)
        if waited >= timeout_s:
            return dict(limit_c=limit_c, start_c=first, end_c=value, waited_s=round(waited, 1), ok=False,
                        reason="temperature_unknown" if value is None else "cool_timeout")
        report(f"cooling: gripper {value} degC, waiting for <= {limit_c} degC ({waited:.0f}/{timeout_s:.0f} s)")
        sleep(period_s)


def hold(sampler, hold_s, abort_c, *, confirm_samples=1, period_s=1.0, sleep=time.sleep, now=time.monotonic):
    """Hold for ``hold_s`` s (nothing is commanded: the worker keeps the jaw goal), watching the sampler's latest
    gripper temperature every ``period_s`` s. Returns ``(window, aborted_at_c)``: the monotonic ``(start, end)`` and
    the temperature (degC) once ``confirm_samples`` consecutive fresh checks inside the hold window read at or above
    ``abort_c`` and ended the hold early, else None."""
    started, consecutive, cursor = now(), 0, len(sampler.samples)
    while now() - started < hold_s:
        sleep(min(period_s, max(hold_s - (now() - started), 0.0)))
        fresh = [sample["temperature_c"] for sample in sampler.samples[cursor:]
                 if sample.get("temperature_c") is not None]
        cursor = len(sampler.samples)
        if not fresh:
            continue
        latest = fresh[-1]
        if latest >= abort_c:
            consecutive += 1
            if consecutive >= confirm_samples:
                return (started, now()), latest
        else:
            consecutive = 0
    return (started, now()), None


def cube_stages(arguments):
    """``(grasp, transport, release)`` stage lists of the cube test: ``motion_bench.carry_stages`` (the motion
    bench's carry, same grasp-centre poses) cut at the close and at the release. The transport runs at
    ``--transport-speed-scale``; ``--grip-squeeze-rad`` sets the close stage's ``grip_squeeze_rad``."""
    x, y, z = arguments.grasp_centre_m
    carry = argparse.Namespace(
        cube_xy_m=[x, y], cube_centre_z_m=z, carry_over_xy_m=arguments.carry_over_xy_m,
        carry_height_m=arguments.carry_height_m, cube_width_m=arguments.object_width_m,
        cube_yaw_deg=arguments.fixed_jaw_yaw_deg, position_tolerance_m=arguments.position_tolerance_m,
        grasp_orientation_tolerance_rad=arguments.grasp_orientation_tolerance_rad, speed_scale=arguments.speed_scale,
        grip_guard_preset=arguments.grip_guard_preset, place_clearance_m=arguments.place_clearance_m)
    stages = {stage["id"]: stage for stage in motion_bench.carry_stages(carry)}
    if arguments.grip_squeeze_rad is not None:
        stages["close"]["grip_squeeze_rad"] = arguments.grip_squeeze_rad
    for identifier in TRANSPORT_STAGES:
        stages[identifier]["speed_scale"] = arguments.transport_speed_scale
    return tuple([stages[identifier] for identifier in group] for group in (GRASP_STAGES, TRANSPORT_STAGES, RELEASE_STAGES))


def jaw_stage(identifier, goal, arguments=None):
    """A gripper stage to ``goal`` (normalized); a closing one carries ``--grip-squeeze-rad`` when given."""
    stage = dict(id=identifier, kind="gripper", arm=ARM, goal_normalized=goal)
    if arguments is not None and arguments.grip_squeeze_rad is not None:
        stage["grip_squeeze_rad"] = arguments.grip_squeeze_rad
    return stage


# ----------------------------------------------------------------------- run ---

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run the protocol under one deployment file.")
    run.add_argument("--deployment", required=True, help="Deployment file of this condition (read only).")
    run.add_argument("--condition", required=True, help="Label of this condition (file name and report field).")
    run.add_argument("--out", required=True, help="New or empty output directory.")
    run.add_argument("--repetitions", type=int, default=3, help="Repetitions of the whole protocol; default 3.")
    run.add_argument("--tests", nargs="+", default=["empty", "cube"], choices=("empty", "cube"),
                     help="Tests to run; cube is skipped without --grasp-centre-m, --carry-over-xy-m and --carry-height-m.")
    run.add_argument("--observation-profile", default="rgb_proprio_v1",
                     help="Observation profile of the run; default rgb_proprio_v1 (no camera calibration needed).")
    run.add_argument("--hold-s", type=float, default=60.0, help="Hold time of each close (s); default 60.")
    run.add_argument("--grasp-centre-m", type=float, nargs=3,
                     help="Cube centre x y z as it rests on the table (m, so101_base): the grasp_centre_m of the grasp.")
    run.add_argument("--object-width-m", type=float, default=0.029,
                     help="Cube width along the jaw line (m): object_width_m of the grasp; default 0.029.")
    run.add_argument("--fixed-jaw-yaw-deg", type=float, default=90.0,
                     help="fixed_jaw_yaw_deg of the grasp (deg about base +z from +x); default 90.")
    run.add_argument("--carry-over-xy-m", type=float, nargs=2,
                     help="Point the cube is carried over and back from, x y (m, so101_base): the lateral move.")
    run.add_argument("--carry-height-m", type=float, help="Grasp-centre height during the transport (m, so101_base z).")
    run.add_argument("--place-clearance-m", type=float, default=0.002,
                     help="Height above its rest height at which the cube is released (m); default 0.002.")
    run.add_argument("--speed-scale", type=float, default=0.5,
                     help="robot_action speed_scale of the park, approach and retreat moves, [0.25, 1]; default 0.5.")
    run.add_argument("--transport-speed-scale", type=float, default=1.0,
                     help="speed_scale of the lift, lateral and set-down moves, [0.25, 1]; default 1, the agent's "
                          "normal speed (the accelerations a held cube sees in real runs).")
    run.add_argument("--position-tolerance-m", type=float, default=0.01,
                     help="Arrival position tolerance of every move (m); default 0.01 (the normal preset).")
    run.add_argument("--orientation-tolerance-rad", type=float, default=3.14159,
                     help="Orientation tolerance of the park move (rad); default 3.14159 = position only.")
    run.add_argument("--grasp-orientation-tolerance-rad", type=float, default=0.15,
                     help="Orientation tolerance of the cube moves (rad); default 0.15 (the normal preset).")
    run.add_argument("--grip-guard-preset", default=None,
                     help="grip_guard_preset of the cube close (e.g. rigid_object); default none. The close is the "
                          "last stage of its batch, so the guard is evaluated and recorded (grip_guard) but never "
                          "pauses the bench.")
    run.add_argument("--grip-squeeze-rad", type=float, default=None,
                     help="grip_squeeze_rad of the close stages (rad of the jaw joint); default: not sent, the "
                          "deployment's gripper.hold_squeeze_rad applies. Refused by a deployment without that key.")
    run.add_argument("--slip-aperture-m", type=float, default=0.001,
                     help="Fingertip-gap drop between the close and the release reported as "
                          "jaw_closed_beyond_threshold (m); default 0.001 (about 8 encoder ticks; the jaw of a lost "
                          "cube closes by the squeeze's fingertip travel, 0.8 mm per 0.01 rad, so a squeeze below "
                          "about 0.0125 rad cannot cross it: watch those runs).")
    run.add_argument("--start-temperature-c", type=float, default=36.0,
                     help="Gripper servo temperature every hold starts at or below (degC); default 36. Set it to the "
                          "idle reading of your arm + 3.")
    run.add_argument("--abort-temperature-c", type=float, default=42.0,
                     help="Gripper servo temperature that ends a hold at once and stops the bench (degC); default 42, "
                          "below the 46 degC read before the servo stopped answering on 2026-11-16.")
    run.add_argument("--abort-confirm-samples", type=int, default=2,
                     help="Consecutive hold-check readings >= --abort-temperature-c required to abort and open the jaw "
                          "(count >= 1); default 2, so a single noisy bus sample does not abort a 60 s hold.")
    run.add_argument("--cool-timeout-s", type=float, default=900.0,
                     help="Longest wait at one temperature gate (s); default 900. The bench stops when it expires.")
    compare = commands.add_parser("compare", help="Print mean +- stdev per metric for several reports.")
    compare.add_argument("reports", nargs="+", help="grip-<condition>.json files.")
    compare.add_argument("--match", nargs="+", default=["load_pct", "current_ma", "temperature_c.rise",
                                                        "squeeze_remaining_rad", "jaw_closed_further_m",
                                                        "contact_opening_m", "transport_aperture_change_m"],
                         help="Substrings of the metric names to print; default the load, current, temperature rise, "
                              "remaining squeeze and slip rows.")
    return parser


def run(arguments, session_factory=motion_bench.open_session):
    """Run the protocol; ``session_factory(arguments, out)`` -> ``motion_bench.Session`` (test seam). Returns the report."""
    from ni_rua.robot.backends.so101.deployment import hold_squeeze, load_deployment
    from ni_rua.robot.model import so101_kinematics
    if arguments.repetitions < 1 or not all(0.25 <= value <= 1 for value in (arguments.speed_scale, arguments.transport_speed_scale)):
        raise SystemExit("--repetitions >= 1 and --speed-scale / --transport-speed-scale in [0.25, 1] required")
    if arguments.hold_s <= 0 or arguments.abort_temperature_c <= arguments.start_temperature_c or arguments.abort_confirm_samples < 1:
        raise SystemExit("--hold-s > 0, --abort-confirm-samples >= 1 and --abort-temperature-c above --start-temperature-c required")
    out = Path(arguments.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"{out} is not empty: every run uses a fresh directory")
    deployment = load_deployment(arguments.deployment)
    if arguments.grip_squeeze_rad is not None and hold_squeeze(deployment) is None:
        raise SystemExit("--grip-squeeze-rad needs a deployment with gripper.hold_squeeze_rad (the stage field is not "
                         "offered without it)")
    model = so101_kinematics.load_model()
    tests, skipped = list(arguments.tests), {}
    if "cube" in tests and None in (arguments.grasp_centre_m, arguments.carry_over_xy_m, arguments.carry_height_m):
        tests.remove("cube")
        skipped["cube"] = "needs --grasp-centre-m, --carry-over-xy-m and --carry-height-m"
    gripper = deployment["gripper"]
    report = {"schema": SCHEMA, "condition": arguments.condition,
              "started_at": datetime.now(timezone.utc).isoformat(),
              "deployment": str(Path(arguments.deployment).expanduser().resolve()),
              "settings": {"gripper": {key: gripper.get(key) for key in (
                               "hold_squeeze_rad", "max_hold_squeeze_rad", "empty_close_aperture_m", "jaw_closed_rad",
                               "jaw_open_rad", "max_speed_rad_s", "settle_timeout_s")},
                           "servo": dict(deployment["servo"]),
                           "operator_speed_scale_cap": deployment["operator"]["default_speed_scale_cap"],
                           "telemetry_hz": {"joint": deployment["telemetry"]["joint_hz"],
                                            "motor_status": deployment["telemetry"]["motor_status_hz"]}},
              "source": motion_bench.source_binding(out), "protocol": dict(vars(arguments)),
              "tests": tests, "skipped_tests": skipped, "repetitions": [], "aborted": None}
    path = out / f"grip-{arguments.condition}.json"

    def save():
        report["summary"] = summarise(report["repetitions"])
        path.write_text(json.dumps(report, indent=2, default=str))
    save()
    session = session_factory(arguments, out)
    park = motion_bench.park_action(arguments, model, deployment)
    label = f"[{arguments.condition}"

    class Stop(RuntimeError):
        pass

    def action(name, stages):
        return motion_bench.run_action(session, name, stages, metrics=grip_stage_facts)

    def must(record, what):
        if record["status"] != "completed":
            raise Stop(f"{what}: {record['status']} {record.get('reason')}")
        return record

    def gate(entry, name):
        result = cool_gate(session.read_telemetry, arguments.start_temperature_c, arguments.cool_timeout_s)
        entry.setdefault("gates", {})[name] = result
        save()
        if not result["ok"]:
            raise Stop(f"temperature gate before {name}: {result['reason']} (gripper {result['end_c']} degC, "
                       f"limit {result['limit_c']} degC)")

    def held(entry, sampler, release_stages, what):
        """The hold of one test: sample, and on the abort temperature open the jaw before stopping."""
        window, hot = hold(sampler, arguments.hold_s, arguments.abort_temperature_c,
                           confirm_samples=arguments.abort_confirm_samples)
        entry["hold"] = window_metrics(sampler.samples, window)
        if hot is not None:
            entry["hold"]["aborted_at_temperature_c"] = hot
            entry["release"] = action(f"{what}_release", release_stages)
            save()
            raise Stop(f"{what} hold: gripper reached {hot} degC (abort at {arguments.abort_temperature_c} degC); jaw opened")

    try:
        for index in range(arguments.repetitions):
            entry = {"index": index}
            report["repetitions"].append(entry)
            entry["park"] = must(action("park", park), "park")
            with GripSampler(session.read_telemetry) as sampler:
                if "empty" in tests:
                    empty = entry["empty"] = {}
                    must(action("empty_open", [jaw_stage("open", 1.0)]), "empty open")
                    gate(entry, "empty")
                    record = empty["grasp"] = must(action("empty_close", [jaw_stage("close", 0.0, arguments)]), "empty close")
                    empty["close"] = record["stages"].get("close")
                    held(empty, sampler, [jaw_stage("release", 1.0)], "empty")
                    empty["release"] = must(action("empty_release", [jaw_stage("release", 1.0)]), "empty release")
                    print(f"{label} #{index}] empty hold: {json.dumps(empty['hold'])}", flush=True)
                    save()
                if "cube" in tests:
                    cube = entry["cube"] = {}
                    grasp, transport, release = cube_stages(arguments)
                    must(action("cube_open", [jaw_stage("open", 1.0)]), "cube open")
                    gate(entry, "cube")
                    record = cube["grasp"] = must(action("cube_grasp", grasp), "cube grasp")
                    cube["close"] = record["stages"].get("close")
                    held(cube, sampler, release, "cube")
                    moved = cube["transport_action"] = action("cube_transport", transport)
                    cube["transport"] = window_metrics(sampler.samples, moved["window"])
                    save()
                    # The cube's state is unknown after a failed transport: an operator has to look before anything moves.
                    must(moved, "cube transport")
                    cube["release"] = must(action("cube_release", release), "cube release")
                    cube["slip"] = slip_facts(cube["grasp"], moved, cube["release"], arguments.slip_aperture_m)
                    print(f"{label} #{index}] cube hold: {json.dumps(cube['hold'])}\n"
                          f"{label} #{index}] cube slip: {json.dumps(cube['slip'])}", flush=True)
                    save()
            entry["park_after"] = must(action("park", park), "park after tests")
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
    """Print ``mean +- stdev (n)`` of every summary metric whose name contains one of ``--match``, one column per
    report; returns the table rows ``[(metric, [cell per report])]``."""
    reports = [json.loads(Path(item).expanduser().read_text()) for item in arguments.reports]
    labels = [f"{report['condition']} ({(report['settings']['gripper'] or {}).get('hold_squeeze_rad')} rad)" for report in reports]
    names = sorted({name for report in reports for name in report.get("summary", {})
                    if any(part in name for part in arguments.match)})

    def cell(stats):
        if not stats:
            return "-"
        spread = "" if stats.get("stdev") is None else f" +- {stats['stdev']:.4g}"
        return f"{stats['mean']:.4g}{spread} (n={stats['n']})"
    rows = [(name, [cell(report.get("summary", {}).get(name)) for report in reports]) for name in names]
    for report in reports:
        if report.get("aborted"):
            print(f"NOTE {report['condition']}: aborted: {report['aborted']}")
    width = max([len(name) for name, _ in rows] + [10])
    print(" " * width, *(f"{label:>28s}" for label in labels))
    for name, cells in rows:
        print(f"{name:<{width}s}", *(f"{text:>28s}" for text in cells))
    return rows


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    return compare(arguments) if arguments.command == "compare" else run(arguments)


if __name__ == "__main__":
    main()
