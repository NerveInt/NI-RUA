"""Measure how fast the real SO-101 can move, separating our limits from the servo's.

Moves shoulder_pan (least gravity load) from the ready pose by +-AMPLITUDE at increasing
commanded joint speeds, then one instant step (single goal write). Reports duration, peak
measured velocity and tracking lag. Deployment limits are raised only in-process for the probe.
Run from a terminal with camera permission (worker opens cameras):
  PYTHONPATH=. <so101 python> scripts/so101_speed_probe.py --deployment configs/so101.local.json
"""
import argparse, copy, json, math, sys, threading, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ni_rua.robot.backends.so101.deployment import load_deployment  # noqa: E402
from ni_rua.robot.backends.so101.worker import SO101Adapter  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--deployment", required=True)
parser.add_argument("--amplitude-rad", type=float, default=0.4)
parser.add_argument("--speeds", type=float, nargs="+", default=[0.3, 0.6, 1.0, 1.5, 2.5])
parser.add_argument("--out", default="/tmp/so101_speed_probe.json")
args = parser.parse_args()
base = load_deployment(args.deployment)
adapter = SO101Adapter(copy.deepcopy(base), "/tmp/so101_speed_probe_worker", "rgb_proprio_v1", start_telemetry=False)
ready = list(base["home"]["q_rad"])
results = []

def peak_velocity(trace, joint=0):
    v = [abs(b["q_rad"][joint] - a["q_rad"][joint]) / max(b["t_s"] - a["t_s"], 1e-6) for a, b in zip(trace, trace[1:])]
    return round(max(v), 3) if v else None

interrupted = False
try:
    r = adapter.move_joints(ready, None, 0.05, action_id="probe_ready")
    print("ready", r["status"], flush=True)
    for speed in args.speeds:
        m = adapter.deployment["motion"]
        m["max_joint_speed_rad_s"] = speed
        m["max_command_delta_rad"] = max(base["motion"]["max_command_delta_rad"], speed / m["command_rate_hz"] * 1.05)
        m["tracking_error_limit_rad"] = max(base["motion"]["tracking_error_limit_rad"], 0.4)
        for sign in (1, -1, 0):
            target = list(ready); target[0] += sign * args.amplitude_rad
            r = adapter.move_joints(target, None, 0.05, action_id=f"v{speed}_{sign}", correct_sag=False)
            row = dict(speed_cmd=speed, target_offset=sign * args.amplitude_rad, status=r["status"], reason=r.get("reason"),
                       duration_s=r.get("wall_clock_duration_s"), commands=r.get("segment_commands"),
                       peak_measured_rad_s=peak_velocity(r.get("trace") or []),
                       max_lag_rad=round(max((abs(d["q_goal_rad"][0] - s["q_rad"][0]) for d, s in zip(r.get("dispatched", []), (r.get("trace") or [])[1:])), default=0), 3),
                       final_error_rad=r.get("max_joint_error_rad"))
            results.append(row); print(json.dumps(row), flush=True)
            if r["status"] != "completed":
                raise RuntimeError(f"stopped at {row}")
    # Instant step: one goal write, servo runs at its own internal speed/acceleration.
    q0 = adapter.read()["q_rad"]
    target = list(q0); target[0] += args.amplitude_rad
    t0 = time.monotonic(); adapter._send(target, adapter.jaw_goal); trace = []
    while time.monotonic() - t0 < 2.0:
        s = adapter.read(); trace.append({"t_s": s["t"] - t0, "q_rad": s["q_rad"]}); time.sleep(0.01)
    reach = next((x["t_s"] for x in trace if abs(x["q_rad"][0] - q0[0]) >= 0.9 * args.amplitude_rad), None)
    step = dict(mode="instant_step", amplitude=args.amplitude_rad, t90_s=reach and round(reach, 3), peak_measured_rad_s=peak_velocity(trace))
    results.append(step); print(json.dumps(step), flush=True)
except BaseException as error:
    interrupted = isinstance(error, KeyboardInterrupt)
    try:
        stop = adapter._stop()
    except BaseException as stop_error:
        stop = {"confirmed": False, "error": f"{type(stop_error).__name__}: {stop_error}"}
    results.append(dict(aborted=f"{type(error).__name__}: {error}", abort_stop=stop))
    raise
finally:
    if not interrupted:
        try:
            r = adapter.move_joints(ready, None, 0.05, action_id="probe_park")
            print("park", r["status"], flush=True)
        except BaseException as park_error:
            results.append(dict(park_error=f"{type(park_error).__name__}: {park_error}"))
            print("park error", type(park_error).__name__, park_error, flush=True)
    try:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=1))
    finally:
        adapter.close()
