"""SO-101 servo identification for motion compensation: joint stiffness under gravity and static-friction band.

Developer bench (about 4 minutes, supervised). It writes a PROPOSAL file; it never edits a deployment file. Copy the
``proposal.motion_compensation`` block into a copy of ``configs/so101.local.json`` yourself after reading the fit.

What it measures
  1. Stiffness (``motion_compensation.gravity_feedforward.joint_stiffness_nm_per_rad``, N*m/rad). The arm holds a
     set of free-space joint poses with arrival correction and every motion compensation OFF. At each pose the
     settled ``goal - measured`` (rad, averaged over ``--dwell-s``) is paired with the gravity torque at the measured
     joints (``so101_gravity.gravity_torques``: URDF link masses, N*m). A pure-P servo rests at
     ``error = -torque / stiffness``; each joint is fitted by least squares (``motion_compensation.fit_stiffness``).
     Every pose is visited twice, in forward and in reverse order, so each is approached from two directions and
     the spread between the two (``approach_spread_rad``) shows the friction the fit averages over.
  2. With ``--breakaway``: the static-friction band (``motion_compensation.arrival_breakaway_rad``, rad). At
     ``--breakaway-pose`` each joint's goal is stepped one encoder tick (2*pi/4096 rad) every
     ``--breakaway-period-s`` until the measured joint moves by more than one tick, in both directions. The proposal
     takes the larger of the two (a step of that size moves the joint either way); both are reported.

Why the worker adapter and not robot_action: the measurement needs uncorrected joint-space holds and one-tick goal
steps, which the actor interface deliberately does not offer. It runs the real worker adapter in-process (so101
interpreter), like ``scripts/so101_commission.py``, so the power-on self-check, speed caps and tracking / stall
guards are the ones of Harness actions. The gateway must NOT be running (one process owns the serial bus).

Poses: ``--poses-file`` (JSON list of five arm joint angles, rad) or the built-in grid (shoulder_lift x elbow_flex
x wrist_flex at shoulder_pan 0, wrist_roll 0). Height screen (forward kinematics, so101_base z; NOT a collision
check: keep the table clear): the lowest of every moving link origin and the two CAD fingertip points, at both ends
of the jaw range, must stay at or above ``--min-z-m``

  - at every pose (whose TCP must also lie inside the deployment ``workspace`` box),
  - along the joint-space line the arm really travels between consecutive poses (sampled every
    ``--screen-step-rad``), from the start pose to the first pose, and from every pose back to ``home.q_rad`` (the
    abort path),
  - at ``--breakaway-pose`` and on the lines to and from it.

The start pose, ``home.q_rad`` and ``--breakaway-pose`` are given, not chosen: when one of them is itself below
``--min-z-m`` the floor of its lines is its own height (the arm may leave or reach it but not dip below it). A pose
whose inbound line or return-home line fails is left out; a failing breakaway line refuses the run before anything
moves. ``--dry-run`` prints every pose and line with its lowest point and exits without opening
the bus; it assumes the arm starts at ``home.q_rad`` (the real run screens the line from the measured pose).

Acceptance: a joint's fit enters the proposal only when ``r_squared >= --min-r-squared`` and the mean difference
between its two approach directions is at most ``--max-approach-spread-ratio`` of the error range the fit explains;
a rejected joint is listed with ``accepted: false`` and the reasons, and gets no feedforward.

Safety: the script refuses to run with ``disable_torque_on_disconnect`` true (the arm would drop when the script
ends) and, after an abort other than Ctrl-C, returns to ``home.q_rad`` once the stop is confirmed. The breakaway
steps go through the worker's goal writer with the admission gate checked and the stop generation bound, like the
calibration touch-off, so a hold stops them; they cannot be a robot_action because the actor interface has no
one-tick joint step.

  PYTHONPATH=. <so101 python> scripts/so101_gravity_identify.py --deployment configs/so101.local.json --dry-run
  PYTHONPATH=. <so101 python> scripts/so101_gravity_identify.py --deployment configs/so101.local.json \\
      --breakaway --out ~/.cache/ni-rua/so101/gravity-identify-<UTC>.json

Requires ``servo.i_coefficient`` 0: with an integral gain the settled error is not ``-torque / stiffness``.
Re-run after changing ``servo.p_coefficient`` or the supply voltage.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ni_rua.robot.backends.so101.deployment import (ARM_JOINTS, load_deployment,  # noqa: E402
                                                     validate_motion_compensation)
from ni_rua.robot.backends.so101.driver import ENCODER_RESOLUTION_RAD  # noqa: E402
from ni_rua.robot.backends.so101.motion_compensation import fit_stiffness  # noqa: E402
from ni_rua.robot.embodiments.so101 import gravity as so101_gravity  # noqa: E402
from ni_rua.robot.model import so101_kinematics  # noqa: E402

SCHEMA = "ni_rua.so101.gravity_identification.v1"
#: Built-in pose grid (rad): shoulder_lift x elbow_flex x wrist_flex. It spreads the gravity torque of the three
#: loaded joints (development unit: shoulder_lift -0.13..0.56, elbow_flex 0.27..0.52, wrist_flex 0.04..0.14 N*m).
GRID_RAD = ((-1.3, -0.9, -0.5, -0.1), (0.2, 0.7, 1.2), (0.4, 1.0))
#: Link origins that never move with the arm joints below shoulder_pan (not part of the height check).
STATIC_LINKS = ("base_link", "shoulder_link")


class Abort(RuntimeError):
    pass


def candidate_poses(poses_file):
    if poses_file:
        poses = json.loads(Path(poses_file).expanduser().read_text())
        if (not isinstance(poses, list) or not poses
                or any(not isinstance(q, list) or len(q) != 5 or not np.isfinite(q).all() for q in poses)):
            raise SystemExit("--poses-file: a JSON list of [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, "
                             "wrist_roll] in rad is required")
        return [[float(v) for v in q] for q in poses]
    return [[0.0, lift, elbow, wrist, 0.0] for lift, elbow, wrist in itertools.product(*GRID_RAD)]


def lowest_z(model, q, jaw_range_rad):
    """Lowest so101_base z (m) the arm reaches at joints ``q`` (rad): every moving link origin and the two CAD
    fingertip points (``so101_gripper_geometry.fingertip_reference``, TCP frame) at both jaw angles of
    ``jaw_range_rad`` (closed, open). Forward kinematics of origins and tips only, not of the meshes."""
    from ni_rua.robot.model.so101_gripper_geometry import fingertip_reference
    lowest = np.inf
    for jaw in jaw_range_rad:
        poses = so101_kinematics.fk_all(model, q, gripper_rad=jaw)
        lowest = min(lowest, *(float(pose[2, 3]) for link, pose in poses.items() if link not in STATIC_LINKS))
        tips = fingertip_reference(float(jaw))
        tcp = poses[so101_kinematics.TCP_LINK]
        for point in (tips["fixed_tip_m"], tips["moving_tip_m"]):
            lowest = min(lowest, float((tcp @ np.append(np.asarray(point, dtype=float), 1.0))[2]))
    return lowest


def line_facts(model, start_q, end_q, jaw_range_rad, step_rad, min_z_m, label):
    """Height screen of the joint-space line ``start_q`` -> ``end_q`` (the path ``move_joints`` streams), sampled so
    that no joint changes by more than ``step_rad`` between samples. Returns ``label``, ``floor_m``, ``lowest_z_m``,
    ``lowest_at_fraction`` (0 = start, 1 = end), ``samples`` and ``ok`` (lowest_z_m >= floor_m). ``min_z_m`` is the
    floor of the line; pass a lower value for a line that starts or ends at a pose the arm already rests at or the
    deployment's home (``screen_lines``): such an endpoint is given, the line only must not dip below it."""
    start, end = np.asarray(start_q, dtype=float), np.asarray(end_q, dtype=float)
    count = max(2, int(np.ceil(float(np.abs(end - start).max()) / step_rad)) + 1)
    heights = [lowest_z(model, start + (end - start) * fraction, jaw_range_rad) for fraction in np.linspace(0.0, 1.0, count)]
    worst = int(np.argmin(heights))
    return {"line": label, "floor_m": round(float(min_z_m), 4), "lowest_z_m": round(float(heights[worst]), 4),
            "lowest_at_fraction": round(worst / (count - 1), 3), "samples": count,
            "ok": bool(heights[worst] >= min_z_m - 1e-9)}


def screen_lines(model, deployment, facts, start_q, arguments):
    """Choose the poses whose travel lines pass the height screen, and screen the remaining lines.

    ``facts``: ``pose_facts`` of the candidates, in visiting order; a pose usable on its own is kept when the line
    from the previously kept pose (or ``start_q``) and its line back to ``home.q_rad`` both pass; otherwise its
    ``reason`` becomes ``line_below_min_z``. Returns ``(usable pose facts, lines, problems)``: ``lines`` every
    screened line (``line_facts``), ``problems`` the labels of failing lines that no pose choice can avoid (the
    breakaway pose and its lines)."""
    g = deployment["gripper"]
    jaws = (g["jaw_closed_rad"], g["jaw_open_rad"])
    home = list(deployment["home"]["q_rad"])

    def line(a, b, label, given=None):
        # ``given``: an endpoint that is not ours to choose (where the arm is now, the deployment home). When it is
        # itself under --min-z-m, the line's floor is that endpoint's own height: leaving or reaching it is allowed,
        # dipping below it is not.
        floor = arguments.min_z_m if given is None else min(arguments.min_z_m, lowest_z(model, given, jaws))
        return line_facts(model, a, b, jaws, arguments.screen_step_rad, floor, label)
    usable, lines, problems, previous, previous_label = [], [], [], list(start_q), "start"
    for index, item in enumerate(facts):
        if item["reason"] is not None:
            continue
        inbound = line(previous, item["q_rad"], f"{previous_label} -> pose {index}",
                       given=previous if previous_label == "start" else None)
        home_line = line(item["q_rad"], home, f"pose {index} -> home", given=home)
        lines.extend((inbound, home_line))
        if inbound["ok"] and home_line["ok"]:
            usable.append(item)
            previous, previous_label = item["q_rad"], f"pose {index}"
        else:
            item["reason"] = "line_below_min_z"
    if arguments.breakaway and usable:
        pose = list(arguments.breakaway_pose)
        # The breakaway pose is an operator choice (default: the deployment home): given, like home itself.
        checks = [line(usable[0]["q_rad"], pose, "first pose -> breakaway pose", given=pose),
                  line(pose, home, "breakaway pose -> home", given=home if lowest_z(model, home, jaws) < lowest_z(model, pose, jaws) else pose)]
        lines.extend(checks)
        problems.extend(check["line"] for check in checks if not check["ok"])
    return usable, lines, problems


def pose_facts(model, inertials, deployment, q, min_z_m):
    """FK facts of one candidate pose and why it is unusable (``reason`` None = usable)."""
    limits = so101_kinematics.joint_limits_rad(model)
    margin = deployment["motion"]["joint_limit_margin_rad"]
    poses = so101_kinematics.fk_all(model, q)
    tcp = poses[so101_kinematics.TCP_LINK][:3, 3]
    lowest = lowest_z(model, q, (deployment["gripper"]["jaw_closed_rad"], deployment["gripper"]["jaw_open_rad"]))
    low, high = deployment["workspace"]["min_m"], deployment["workspace"]["max_m"]
    reason = None
    if any(not limits[name][0] + margin <= value <= limits[name][1] - margin for name, value in zip(ARM_JOINTS, q)):
        reason = "outside_joint_limits_with_margin"
    elif any(not lo <= value <= hi for value, lo, hi in zip(tcp, low, high)):
        reason = "tcp_outside_deployment_workspace"
    elif lowest < min_z_m:
        reason = "lowest_point_below_min_z"
    return {"q_rad": [float(v) for v in q], "tcp_position_m": [round(float(v), 4) for v in tcp],
            "lowest_point_z_m": round(lowest, 4),
            "gravity_torque_nm": [round(float(v), 4) for v in so101_gravity.gravity_torques(model, inertials, q)],
            "reason": reason}


def mean_joints(adapter, dwell_s):
    """Mean measured arm joints (rad) and jaw (rad) over ``dwell_s`` at the command rate."""
    period = 1 / adapter.deployment["motion"]["command_rate_hz"]
    rows, jaws, end = [], [], adapter.clock.now() + dwell_s
    while True:
        sample = adapter.read()
        rows.append(sample["q_rad"])
        jaws.append(sample["gripper_rad"])
        if adapter.clock.now() >= end:
            return np.mean(rows, axis=0), float(np.mean(jaws))
        adapter.clock.sleep(period)


def hold(adapter, q, arguments, label):
    """Uncorrected joint move to ``q`` and its settled goal error. Raises ``Abort`` unless it completes."""
    result = adapter.move_joints(q, None, arguments.arrival_tolerance_rad, speed_scale=arguments.speed_scale,
                                 correct_sag=False, action_id=label)
    if result["status"] != "completed":
        raise Abort(f"{label}: {result['status']} {result.get('reason')} {result.get('diagnostic')}")
    measured, jaw = mean_joints(adapter, arguments.dwell_s)
    goal = np.asarray(adapter.arm_goal)
    return goal, measured, jaw


def breakaway(adapter, joint, arguments, progress):
    """Goal change (rad) that first moves ``joint`` from rest at the breakaway pose, per direction.

    Returns ``{"plus_rad", "minus_rad"}``; a direction is None when the joint had not moved by more than one encoder
    tick at ``--breakaway-max-rad``."""
    index, out = ARM_JOINTS.index(joint), {}
    for sign, key in ((1, "plus_rad"), (-1, "minus_rad")):
        base, rest, _ = hold(adapter, arguments.breakaway_pose, arguments, f"breakaway_{joint}_{key}")
        out[key] = None
        ticks = 0
        # Same admission and stop rules as an operator move: refused while relaxed, ended by any hold after this point.
        gate, generation = adapter._gate(), adapter.stop_generation
        if gate:
            raise Abort(f"breakaway {joint}: {gate}")
        while (ticks + 1) * ENCODER_RESOLUTION_RAD <= arguments.breakaway_max_rad:
            ticks += 1
            goal = base.copy()
            goal[index] += sign * ticks * ENCODER_RESOLUTION_RAD
            if not adapter._send(goal.tolist(), adapter.jaw_goal, generation):
                raise Abort(f"breakaway {joint}: stopped ({adapter.stop_reason or 'operator_hold'})")
            adapter.clock.sleep(arguments.breakaway_period_s)
            moved = abs(adapter.read()["q_rad"][index] - rest[index])
            if int(round(moved / ENCODER_RESOLUTION_RAD)) > 1:
                out[key] = round(ticks * ENCODER_RESOLUTION_RAD, 5)
                break
        progress(f"breakaway {joint} {key}: {out[key]}")
    return out


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deployment", required=True, help="Deployment file to identify under (read only).")
    parser.add_argument("--out", help="Proposal JSON to write (must not exist; required unless --dry-run).")
    parser.add_argument("--poses-file", help="JSON list of arm joint poses (rad); default: the built-in grid.")
    parser.add_argument("--min-z-m", type=float, default=0.06,
                        help="Lowest so101_base z (m) any moving link origin or fingertip may have at a pose or on a "
                             "line between poses; default 0.06 (development unit: table at z=-0.02 m, so 8 cm above it).")
    parser.add_argument("--screen-step-rad", type=float, default=0.02,
                        help="Largest joint change between two samples of a screened line (rad); default 0.02 "
                             "(about 6 mm of tool travel at 0.3 m reach).")
    parser.add_argument("--min-r-squared", type=float, default=0.9,
                        help="A joint's fit is accepted only when the gravity model explains at least this share of "
                             "the variance of its settled error; default 0.9. Below it the stiffness is not proposed.")
    parser.add_argument("--max-approach-spread-ratio", type=float, default=0.5,
                        help="A joint's fit is accepted only when the mean difference between the two approach "
                             "directions (friction hysteresis, rad) is at most this share of the error range the fit "
                             "explains (compliance x torque range, rad); default 0.5: beyond it friction, not "
                             "gravity, decides where the joint rests.")
    parser.add_argument("--dwell-s", type=float, default=1.0,
                        help="Averaging time (s) of the settled joints at each pose; default 1.")
    parser.add_argument("--speed-scale", type=float, default=0.5,
                        help="Fraction of the effective joint speed for the moves, (0, 1]; default 0.5.")
    parser.add_argument("--arrival-tolerance-rad", type=float, default=0.2,
                        help="Largest uncorrected settled joint error (rad) accepted at a pose; default 0.2 "
                             "(the sag being measured is expected inside it).")
    parser.add_argument("--min-torque-range-nm", type=float, default=0.05,
                        help="A joint whose gravity torque varies less than this over the poses (N*m) is not fitted; "
                             "default 0.05 (leaves out shoulder_pan and wrist_roll on a level base).")
    parser.add_argument("--max-offset-rad", type=float, default=0.1,
                        help="gravity_feedforward.max_offset_rad written into the proposal (rad, (0, 0.3]); default "
                             "0.1. The report lists the largest feedforward offset over the poses to compare.")
    parser.add_argument("--breakaway", action="store_true", help="Also identify the static-friction band per joint.")
    parser.add_argument("--breakaway-joints", nargs="+", default=["shoulder_lift", "elbow_flex", "wrist_flex"],
                        choices=ARM_JOINTS, help="Joints of the breakaway test; default the three loaded joints.")
    parser.add_argument("--breakaway-pose", type=float, nargs=5, default=None,
                        help="Pose of the breakaway test (rad); default the deployment home.q_rad.")
    parser.add_argument("--breakaway-max-rad", type=float, default=0.06,
                        help="Largest goal change tried per direction (rad); default 0.06. A joint that has not "
                             "moved by then is reported null.")
    parser.add_argument("--breakaway-period-s", type=float, default=0.3,
                        help="Wait after each one-tick goal step before reading the joint (s); default 0.3.")
    parser.add_argument("--dry-run", action="store_true", help="Print the usable poses and exit; no bus access.")
    return parser


def main(argv=None, adapter_factory=None):
    """``adapter_factory(deployment, run_dir)``: test seam returning an ``SO101Adapter``; default the real one."""
    arguments = build_parser().parse_args(argv)
    if not (0 < arguments.speed_scale <= 1 and arguments.dwell_s > 0 and 0 < arguments.max_offset_rad <= 0.3
            and arguments.breakaway_period_s > 0 and 0 < arguments.breakaway_max_rad <= 0.1
            and arguments.screen_step_rad > 0 and 0 <= arguments.min_r_squared <= 1
            and arguments.max_approach_spread_ratio >= 0):
        raise SystemExit("arguments outside their documented ranges")
    deployment = load_deployment(arguments.deployment)
    if deployment["disable_torque_on_disconnect"] and not arguments.dry_run:
        raise SystemExit("disable_torque_on_disconnect is true: the arm would drop when this script ends. Set it "
                         "false in the deployment copy used for identification")
    if deployment["servo"]["i_coefficient"] != 0:
        raise SystemExit("servo.i_coefficient must be 0 for identification: with an integral gain the settled error "
                         "is not -torque / stiffness")
    # Measure the raw servo: no carried offset, no feedforward, no correction (hold() moves with correct_sag=False).
    deployment["motion_compensation"] = validate_motion_compensation(None)
    model, inertials = so101_kinematics.load_model(), so101_gravity.load_inertials()
    facts = [pose_facts(model, inertials, deployment, q, arguments.min_z_m) for q in candidate_poses(arguments.poses_file)]
    if arguments.breakaway_pose is None:
        arguments.breakaway_pose = list(deployment["home"]["q_rad"])
    home = list(deployment["home"]["q_rad"])

    def screen(start_q, start_name):
        """Screen poses and lines from ``start_q``, print them, and return ``(usable, lines, problems)``."""
        for item in facts:
            if item["reason"] == "line_below_min_z":
                item["reason"] = None  # re-screened from this start
        usable, lines, problems = screen_lines(model, deployment, facts, start_q, arguments)
        print(f"height screen from {start_name}, floor {arguments.min_z_m} m", flush=True)
        for item in facts:
            print(("use  " if item["reason"] is None else "skip ") + json.dumps(item), flush=True)
        for item in lines:
            print(("ok   " if item["ok"] else "LOW  ") + json.dumps(item), flush=True)
        return usable, lines, problems
    if arguments.dry_run:
        usable, lines, problems = screen(home, "home.q_rad (assumed start)")
        print(f"{len(usable)} usable poses of {len(facts)}; problems: {problems or 'none'}; dry run, nothing moved", flush=True)
        return None
    if not arguments.out:
        raise SystemExit("--out is required")
    out = Path(arguments.out).expanduser().resolve()
    if out.exists() or out == Path(arguments.deployment).expanduser().resolve():
        raise SystemExit(f"{out} exists; the proposal is always a new file")
    if adapter_factory is None:
        from ni_rua.robot.backends.so101.worker import SO101Adapter

        def adapter_factory(deploy, run_dir):
            return SO101Adapter(deploy, run_dir, "rgb_proprio_v1", start_telemetry=False)
    adapter = adapter_factory(deployment, out.with_suffix(".worker"))
    # The real start pose: screen the lines again from the measured joints before anything moves.
    usable, lines, problems = screen(adapter.read()["q_rad"], "the measured pose")
    if len(usable) < 3 or problems:
        adapter.close()
        raise SystemExit(f"height screen refused the run: {len(usable)} usable poses, problems: {problems or 'none'}")
    report = {"schema": SCHEMA, "started_at": datetime.now(timezone.utc).isoformat(),
              "deployment": str(Path(arguments.deployment).expanduser().resolve()),
              "servo": dict(deployment["servo"]), "arguments": vars(arguments),
              "identified_with": "motion_compensation off; uncorrected joint holds (move_joints correct_sag=False)",
              "poses": facts, "screened_lines": lines, "holds": [], "ok": False}

    def progress(text):
        print(text, flush=True)

    try:
        order = list(enumerate(usable))
        for approach, sequence in (("forward", order), ("reverse", order[::-1])):
            for index, item in sequence:
                goal, measured, jaw = hold(adapter, item["q_rad"], arguments, f"pose_{index}_{approach}")
                torque = so101_gravity.gravity_torques(model, inertials, measured, gripper_rad=jaw)
                report["holds"].append({"pose_index": index, "approach": approach,
                                        "goal_q_rad": [round(float(v), 6) for v in goal],
                                        "measured_q_rad": [round(float(v), 6) for v in measured],
                                        "error_rad": [round(float(v), 6) for v in goal - measured],
                                        "gravity_torque_nm": [round(float(v), 5) for v in torque]})
                progress(f"pose {index} {approach}: error_rad={report['holds'][-1]['error_rad']}")
        errors = np.array([entry["error_rad"] for entry in report["holds"]])
        torques = np.array([entry["gravity_torque_nm"] for entry in report["holds"]])
        fit = fit_stiffness(errors, torques, arguments.min_torque_range_nm)
        half = len(usable)
        spread = np.abs(errors[:half] - errors[half:][::-1])  # the same pose approached from the other side
        for joint_index, name in enumerate(ARM_JOINTS):
            fit[name]["approach_spread_rad"] = {"mean": round(float(spread[:, joint_index].mean()), 5),
                                                "max": round(float(spread[:, joint_index].max()), 5)}
        for name in ARM_JOINTS:
            entry = fit[name]
            if entry["stiffness_nm_per_rad"] is None:
                continue
            # Acceptance (module docstring): the gravity model must explain the error, and friction must not dominate.
            explained = abs(entry["compliance_rad_per_nm"]) * entry["torque_range_nm"]
            entry["approach_spread_ratio"] = round(entry["approach_spread_rad"]["mean"] / explained, 3)
            rejected = []
            if entry["r_squared"] is None or entry["r_squared"] < arguments.min_r_squared:
                rejected.append(f"r_squared below {arguments.min_r_squared}")
            if entry["approach_spread_ratio"] > arguments.max_approach_spread_ratio:
                rejected.append(f"approach_spread_ratio above {arguments.max_approach_spread_ratio}")
            entry["accepted"] = not rejected
            if rejected:
                entry["rejected"] = rejected
        report["fit"] = fit
        stiffness = {name: fit[name]["stiffness_nm_per_rad"] for name in ARM_JOINTS if fit[name].get("accepted")}
        proposal = {}
        if stiffness:
            offsets = np.abs(np.array([so101_gravity.feedforward_offset_rad(
                row, [stiffness.get(name) for name in ARM_JOINTS], np.inf) for row in torques]))
            report["largest_feedforward_offset_rad"] = {name: round(float(offsets[:, index].max()), 5)
                                                        for index, name in enumerate(ARM_JOINTS) if name in stiffness}
            servo = deployment["servo"]
            proposal["gravity_feedforward"] = {
                "joint_stiffness_nm_per_rad": stiffness, "max_offset_rad": arguments.max_offset_rad,
                "provenance": f"so101_gravity_identify {out.name} {report['started_at'][:10]}; servo p/i/d "
                              f"{servo['p_coefficient']}/{servo['i_coefficient']}/{servo['d_coefficient']}; "
                              f"{len(report['holds'])} holds"}
        if arguments.breakaway:
            report["breakaway"] = {joint: breakaway(adapter, joint, arguments, progress)
                                   for joint in arguments.breakaway_joints}
            bands = {joint: max(values["plus_rad"], values["minus_rad"]) for joint, values in report["breakaway"].items()
                     if values["plus_rad"] is not None and values["minus_rad"] is not None}
            if bands:
                proposal["arrival_breakaway_rad"] = {name: bands[name] for name in ARM_JOINTS if name in bands}
                # A breakaway band is only valid together with this switch (deployment validation).
                proposal["stop_correction_within_tolerance"] = True
        report["proposal"] = {"motion_compensation": proposal}
        report["ok"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        progress("ABORT: " + report["error"])
        report["abort_stop"] = adapter._stop()
        # Do not leave the arm extended: go home once the stop is confirmed (every pose's line home was screened).
        # Ctrl-C means the operator wants it to stay where it stopped.
        report["return_home"] = (isinstance(error, Exception) and bool(report["abort_stop"].get("confirmed"))
                                 and not adapter._gate())
    finally:
        try:
            if report["ok"] or report.get("return_home"):
                park = adapter.move_joints(list(deployment["home"]["q_rad"]), None, arguments.arrival_tolerance_rad,
                                           speed_scale=arguments.speed_scale, action_id="park_home")
                report["park_home"] = {key: park.get(key) for key in ("status", "reason", "max_joint_error_rad")}
        except BaseException as error:
            report["park_home"] = {"status": "error", "reason": f"{type(error).__name__}: {error}"}
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str))
        adapter.close()
    print("PROPOSAL", out, "ok=", report["ok"], json.dumps(report.get("proposal")), flush=True)
    return report


if __name__ == "__main__":
    main()
