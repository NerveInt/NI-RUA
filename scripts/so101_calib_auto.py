"""Automatic SO-101 camera calibration capture (board static on the table, robot moves itself).

Thin CLI over ``ni_rua/robot/calibration/workflow/capture.py`` (the same code the operator
console's one-click 重新标定相机 runs inside the worker). Uses the real worker adapter (power-on
self-check, speed caps, sag correction, stall guards) to visit ``home.q_rad`` plus a grid of joint
offsets, waits for settling, and saves both camera frames + measured joints in the session format
of ``scripts/so101_calibrate.py solve``. Poses where the wrist camera sees fewer than
``--min-corners`` board corners are skipped (recorded). Defaults come from the deployment
``calibration`` section; flags override them for this capture only.

  PYTHONPATH=. <so101 python> scripts/so101_calib_auto.py --deployment configs/so101.local.json \\
      --out ~/.cache/ni-rua/so101/calib-<UTC>
Then: <python> scripts/so101_calibrate.py solve --session <out> --fixed-intrinsics-from-wrist
(or ``--solve`` here: gated solve into <out>/camera_calibration.json; the active file is never touched).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ni_rua.robot.calibration.workflow import capture  # noqa: E402
from ni_rua.robot.backends.so101.deployment import load_deployment  # noqa: E402
from ni_rua.robot.backends.so101.worker import SO101Adapter  # noqa: E402

pose_grid = capture.pose_grid  # backward-compatible name


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--out", required=True, help="fresh session directory (must not exist)")
    parser.add_argument("--pan", type=float, nargs="+", help="shoulder_pan offsets from home (rad)")
    parser.add_argument("--flex", type=float, nargs="+", help="wrist_flex offsets from home (rad)")
    parser.add_argument("--roll", type=float, nargs="+", help="wrist_roll offsets from home (rad)")
    parser.add_argument("--lift", type=float, nargs="+", help="shoulder_lift offsets from home (rad)")
    parser.add_argument("--min-corners", type=int, help="wrist corners required to keep a sample")
    parser.add_argument("--speed-scale", type=float, help="fraction of the effective joint speed")
    parser.add_argument("--solve", action="store_true", help="also run the gated solve into <out>/camera_calibration.json")
    args = parser.parse_args()
    out = Path(args.out).expanduser()
    if out.exists():
        raise SystemExit(f"{out} exists; every capture uses a fresh directory")
    deployment = load_deployment(args.deployment)
    settings = dict(deployment["calibration"])
    for key, value in (("pan_offsets_rad", args.pan), ("flex_offsets_rad", args.flex), ("roll_offsets_rad", args.roll),
                       ("lift_offsets_rad", args.lift), ("sample_min_wrist_corners", args.min_corners),
                       ("speed_scale", args.speed_scale)):
        if value is not None:
            settings[key] = value
    deployment["calibration"] = settings
    adapter = SO101Adapter(deployment, out / "worker", "rgb_proprio_v1", start_telemetry=False)
    try:
        result = capture.capture_session(adapter, out, settings=settings,
                                         progress=lambda update: print(json.dumps(update), flush=True))
    finally:
        adapter.close()
    print(f"capture {result['status']} {result['reason'] or ''} kept={result['kept']}/{result['pose_count']}", flush=True)
    if args.solve and result["status"] == "completed":
        report = capture.solve_for_adapter(adapter, out, out / "camera_calibration.json", settings)
        print(json.dumps(capture.report_residuals(report), indent=2))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
