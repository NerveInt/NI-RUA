"""SO-101 first-power-on commissioning: characterise the real arm in ~1 minute.

Runs the real worker adapter in-process (so101 interpreter: LeRobot + OpenCV),
so it goes through the same power-on self-check, connect order, speed caps,
tracking/stall guards and sag correction as Harness actions. Any step that does
not complete stops the sequence with a confirmed hold.

Sequence
  1. Rest -> ready: joint move from the measured rest pose to ``--ready-q-rad``.
  2. Joint steps: each arm joint +step, -step, back (``--step-rad``), first
     without and then with sag correction -> direction, settle time, sag.
  3. Speed: shoulder_pan +-``--sweep-rad`` at the deployment speed cap -> max lag.
  4. Gripper: open 1.0, close 0.0, back to ``--jaw-park`` -> measured range.
  5. Cartesian: +-``--cartesian-m`` along x, y, z through ``execute`` (IK +
     arrival correction) -> per-axis residuals.
  6. Park at the ready pose (not the folded rest: its elbow sits on the hard stop).
     Images of both cameras at rest, ready and after cartesian.

Writes ``<out>/commissioning.json`` and ``<out>/images/*.jpg``. Launch from a
terminal that has macOS camera permission:

  PYTHONPATH=. ~/miniforge3/envs/so101/bin/python scripts/so101_commission.py \
      --deployment configs/so101.local.json --out ~/.cache/ni-rua/so101/commission
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ni_rua.robot.backends.so101 import contract  # noqa: E402
from ni_rua.robot.backends.so101.deployment import ARM_JOINTS, load_deployment  # noqa: E402
from ni_rua.robot.backends.so101.worker import SO101Adapter, pose_matrix, pose_residual  # noqa: E402

IDENTITY = dict(run_id="commissioning", episode_id="ep", env_instance_id="env", env_generation=1)


class Abort(RuntimeError):
    pass


def summarise_move(result, q_start, q_target):
    trace = result.get("trace") or []
    delta = np.subtract(q_target, q_start)
    joint = int(np.argmax(np.abs(delta)))
    out = {"status": result["status"], "reason": result.get("reason"),
           "duration_s": result.get("wall_clock_duration_s"), "commands": result.get("segment_commands"),
           "uncorrected_error_rad": result.get("uncorrected_joint_error_rad"),
           "final_error_rad": result.get("final_joint_error_rad"),
           "correction_rounds": len((result.get("arrival_correction") or {}).get("rounds", [])),
           "diagnostic": result.get("diagnostic")}
    if result["status"] != "completed":
        out["trace"] = [{"t_s": sample["t_s"], "q_rad": [round(v, 4) for v in sample["q_rad"]]} for sample in trace]
        out["dispatched"] = [[round(v, 4) for v in d["q_goal_rad"]] for d in result.get("dispatched", [])]
    if trace and abs(delta[joint]) > 1e-6:
        q0 = trace[0]["q_rad"][joint]
        moved = [abs(sample["q_rad"][joint] - q0) for sample in trace]
        ramp_s = abs(delta[joint]) / result["_speed"]
        reach = next((sample["t_s"] for sample, m in zip(trace, moved) if m >= 0.9 * abs(delta[joint])), None)
        out.update(joint=ARM_JOINTS[joint], commanded_rad=round(float(delta[joint]), 4),
                   measured_rad=round(float(trace[-1]["q_rad"][joint] - q0), 4),
                   direction_ok=bool(np.sign(trace[-1]["q_rad"][joint] - q0) == np.sign(delta[joint])),
                   ramp_s=round(ramp_s, 3), t90_s=reach, lag_after_ramp_s=None if reach is None else round(reach - ramp_s, 3),
                   max_tracking_lag_rad=round(max(abs(np.subtract(d["q_goal_rad"], s["q_rad"])[joint])
                                                  for d, s in zip(result.get("dispatched", []), trace[1:])), 4)
                   if result.get("dispatched") else None)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--out", required=True, help="Output directory (created; must be new or empty).")
    parser.add_argument("--ready-q-rad", type=float, nargs=5, default=[0.0, -0.95, 0.9, 1.0, 0.0],
                        help="Ready pose (rad, shoulder_pan..wrist_roll): TCP ~8 cm above the table, pointing down.")
    parser.add_argument("--step-rad", type=float, default=0.1, help="Per-joint step amplitude (rad).")
    parser.add_argument("--sweep-rad", type=float, default=0.3, help="shoulder_pan speed sweep amplitude (rad).")
    parser.add_argument("--cartesian-m", type=float, default=0.02, help="Cartesian test displacement (m).")
    parser.add_argument("--cartesian-tolerance-m", type=float, default=0.005, help="Cartesian arrival tolerance (m).")
    parser.add_argument("--jaw-park", type=float, default=0.5, help="Jaw goal after the gripper test (normalized).")
    args = parser.parse_args()
    if not all(0 < value <= limit for value, limit in ((args.step_rad, 0.2), (args.sweep_rad, 0.5), (args.cartesian_m, 0.04))):
        raise SystemExit("amplitudes outside the commissioning envelope (step<=0.2 rad, sweep<=0.5 rad, cartesian<=0.04 m)")
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    (out / "images").mkdir()
    deployment = load_deployment(args.deployment)
    report = {"schema": "ni_rua.so101.commissioning.v1", "started_at": datetime.now(timezone.utc).isoformat(),
              "deployment": str(Path(args.deployment).resolve()), "arguments": vars(args), "steps": {}, "ok": False}
    adapter = SO101Adapter(deployment, out / "worker", "rgb_proprio_v1", start_telemetry=False)
    report["preflight"] = getattr(adapter.driver, "preflight_report", None)
    speed = deployment["motion"]["max_joint_speed_rad_s"]
    limits = adapter.kin.joint_limits_rad(adapter.model)
    margin = deployment["motion"]["joint_limit_margin_rad"]

    def snapshot(label):
        observation = adapter.observe()
        for camera in observation["cameras"]:
            source = Path(camera.get("rgb_path") or camera.get("png_path"))
            target = out / "images" / f"{label}_{camera['camera_id']}{source.suffix}"
            target.write_bytes(source.read_bytes())
        return observation["robot_state"]["arms"]["right"]

    def joints():
        return list(adapter.read()["q_rad"])

    def move(label, q_target, *, correct=True, tolerance=None, jaw=None, speed_scale=1.0):
        for name, value in zip(ARM_JOINTS, q_target):
            lower, upper = limits[name]
            if not lower + margin <= value <= upper - margin and label != "return_rest":
                raise Abort(f"{label}: {name}={value:.3f} rad outside URDF limits minus {margin} rad margin")
        q_start = joints()
        result = adapter.move_joints(q_target, jaw, tolerance, speed_scale=speed_scale, correct_sag=correct, action_id=label)
        result["_speed"] = speed * speed_scale
        summary = summarise_move(result, q_start, q_target)
        report["steps"][label] = summary
        print(f"{label:28s} {summary['status']:10s} dur={summary['duration_s']}s final_err="
              f"{None if summary['final_error_rad'] is None else round(max(map(abs, summary['final_error_rad'])), 4)}"
              f" uncorrected={None if summary['uncorrected_error_rad'] is None else round(max(map(abs, summary['uncorrected_error_rad'])), 4)}",
              flush=True)
        if result["status"] != "completed":
            raise Abort(f"{label}: {result['status']} {result.get('reason')}")
        return summary

    rest = joints()
    report["rest_q_rad"] = rest
    try:
        snapshot("rest")
        ready = list(args.ready_q_rad)
        move("rest_to_ready", ready, tolerance=0.05)
        ready_state = snapshot("ready")
        report["ready_tcp"] = ready_state["ee_pose"]

        # 2. per-joint steps: uncorrected out-and-back, then corrected, gives sag in both directions
        report["joint_characterisation"] = {}
        for index, name in enumerate(ARM_JOINTS):
            rows = {}
            for sign in (1, -1):
                target = list(ready)
                target[index] += sign * args.step_rad
                rows["plus" if sign > 0 else "minus"] = move(f"{name}_{'plus' if sign > 0 else 'minus'}_raw", target,
                                                             correct=False, tolerance=0.2)
                corrected = move(f"{name}_{'plus' if sign > 0 else 'minus'}_corrected", target, tolerance=0.02)
                rows["corrected_final_error_rad"] = corrected["final_error_rad"][index]
            move(f"{name}_center", ready, tolerance=0.02)
            report["joint_characterisation"][name] = {
                "direction_ok": rows["plus"]["direction_ok"] and rows["minus"]["direction_ok"],
                "sag_plus_rad": rows["plus"]["uncorrected_error_rad"][index],
                "sag_minus_rad": rows["minus"]["uncorrected_error_rad"][index],
                "t90_s": [rows["plus"]["t90_s"], rows["minus"]["t90_s"]],
                "lag_after_ramp_s": [rows["plus"]["lag_after_ramp_s"], rows["minus"]["lag_after_ramp_s"]]}

        # 3. speed sweep on the base joint at the deployment cap
        for sign, label in ((1, "pan_sweep_plus"), (-1, "pan_sweep_minus")):
            target = list(ready)
            target[0] += sign * args.sweep_rad
            move(label, target, tolerance=0.03)
        move("pan_sweep_center", ready, tolerance=0.02)

        # 4. gripper range
        report["gripper"] = {}
        for goal in (1.0, 0.0, args.jaw_park):
            result = adapter.move_joints(ready, goal, 0.05, action_id=f"jaw_{goal}")
            jaw = result.get("gripper_execution") or {}
            report["gripper"][str(goal)] = {key: jaw.get(key) for key in
                                            ("measured_normalized", "measured_jaw_rad", "jaw_settled", "measured_vs_goal_normalized")}
            print(f"jaw {goal:<4} -> measured {jaw.get('measured_normalized')} settled={jaw.get('jaw_settled')}", flush=True)
            if result["status"] != "completed":
                raise Abort(f"jaw {goal}: {result['status']} {result.get('reason')}")

        # 5. cartesian closed loop through the actor execute path
        caps = adapter.capabilities()
        report["cartesian"] = {}
        base = adapter.robot_state(adapter.read())["arms"]["right"]["ee_pose"]
        for axis, name in enumerate("xyz"):
            for sign in (1, -1):
                position = list(base["position_m"])
                position[axis] += sign * args.cartesian_m
                action = contract.compile_action("move", dict(arm="right", position_m=position,
                    quaternion_xyzw=base["quaternion_xyzw"], observation_id="commissioning"), IDENTITY, caps)
                # position-only: lateral moves of a 5-DOF arm need a base yaw change; the orientation residual is still reported
                action["path"].update(position_tolerance_m=args.cartesian_tolerance_m, orientation_tolerance_rad=math.pi)
                receipt = adapter.execute(action, lambda update: None, threading.Event())
                label = f"{name}{'+' if sign > 0 else '-'}"
                waypoint = (receipt.get("waypoint_results") or [{}])[0]
                report["cartesian"][label] = {"status": receipt["status"], "reason": receipt.get("reason"),
                    "position_error_m": receipt.get("position_error_m"), "orientation_error_rad": receipt.get("orientation_error_rad"),
                    "ik_position_error_m": (waypoint.get("ik") or {}).get("position_error_m"),
                    "correction_rounds": len((waypoint.get("arrival_correction") or {}).get("rounds", [])),
                    "duration_s": receipt.get("wall_clock_duration_s")}
                print(f"cartesian {label}: {receipt['status']} err={receipt.get('position_error_m')} "
                      f"rounds={report['cartesian'][label]['correction_rounds']}", flush=True)
                if receipt["status"] != "completed":
                    raise Abort(f"cartesian {label}: {receipt['status']} {receipt.get('reason')}")
                back = contract.compile_action("move", dict(arm="right", position_m=base["position_m"],
                    quaternion_xyzw=base["quaternion_xyzw"], observation_id="commissioning"), IDENTITY, caps)
                back["path"].update(position_tolerance_m=args.cartesian_tolerance_m, orientation_tolerance_rad=math.pi)
                receipt = adapter.execute(back, lambda update: None, threading.Event())
                if receipt["status"] != "completed":
                    raise Abort(f"cartesian {label} return: {receipt['status']} {receipt.get('reason')}")
        snapshot("after_cartesian")
        report["ok"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        print("ABORT:", report["error"], flush=True)
        report["abort_stop"] = adapter._stop()
    finally:
        try:
            if adapter.read() and report.get("ok") is not None:
                # Park at the ready pose: the folded rest pose sits on the elbow hard stop, and a
                # joint move into it stalls (tracking_error) by design.
                result = adapter.move_joints(list(args.ready_q_rad), args.jaw_park, 0.05, action_id="park_ready")
                report["park_ready"] = {k: result.get(k) for k in ("status", "reason", "max_joint_error_rad")}
                print("park_ready", report["park_ready"], flush=True)
        except BaseException as error:
            report["park_ready"] = {"status": "error", "reason": f"{type(error).__name__}: {error}"}
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (out / "commissioning.json").write_text(json.dumps(report, indent=2, default=str))
        adapter.close()
    print("REPORT", out / "commissioning.json", "ok=", report["ok"], flush=True)


if __name__ == "__main__":
    main()
