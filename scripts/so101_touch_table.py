"""Developer tool: SO-101 table touch-off from the command line (thin CLI over touch_off.py).

Operators use the guided calibration page ``/calibration`` (step S1 关节零位与桌面高度 and the S6
verification touches), which runs the same code inside the gateway's worker. This CLI runs the real worker
adapter in-process (preflight, speed caps, stop), so the gateway must NOT be running. Launch it from a
terminal with camera permission:

  PYTHONPATH=. <so101 python> scripts/so101_touch_table.py --deployment configs/so101.local.json \\
      --points 0.20,0.00 0.24,0.06 --out ~/.cache/ni-rua/so101/touch-<UTC>.json

Every touch parameter defaults to the deployment ``calibration.touch_*`` fields (units in
``deployment.FIELDS``); flags override them for this run only. ``--camera-calibration`` additionally
measures, at each contact, the contact point in the ChArUco board frame through the wrist camera chain
(joint-angle independent); touch ON the board for that.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ni_rua.robot.backends.so101.deployment import load_deployment, validate_deployment  # noqa: E402
from ni_rua.robot.calibration.workflow.touch_off import touch_points  # noqa: E402
from ni_rua.robot.backends.so101.worker import SO101Adapter  # noqa: E402

OVERRIDES = {"hover_z_m": "touch_hover_z_m", "min_z_m": "touch_min_z_m", "step_m": "touch_step_m",
             "step_period_s": "touch_step_period_s", "contact_lag_rad": "touch_contact_lag_rad",
             "contact_load_delta": "touch_contact_load_delta", "jaw": "touch_jaw_normalized"}


def camera_measurement(adapter, calibration_path):
    """``on_contact`` hook: contact point in the board frame via the wrist camera chain (developer check)."""
    import cv2
    from ni_rua.robot.calibration.workflow import charuco as cal
    document = cal.read_calibration_file(calibration_path)
    wrist, board = document["cameras"]["wrist"], cal.make_board(document["board"])

    def measure(entry, q):
        # This check observes the post-hold pose, not the first obstruction candidate.
        before = adapter.read()
        frame = adapter.capture_frames(before["t"])["wrist"]
        after = adapter.read()
        alignment = {"measurement_phase": "post_hold", "q_before_rad": before["q_rad"],
                     "q_after_rad": after["q_rad"], "joint_interval_monotonic_s": [before["t"], after["t"]],
                     "max_endpoint_drift_rad": float(np.max(np.abs(np.subtract(after["q_rad"], before["q_rad"])))),
                     "exposure_alignment": "asynchronous; exposure latency unknown", "contact_evidence": "unknown"}
        if alignment["max_endpoint_drift_rad"] > adapter.deployment["calibration"]["touch_arrival_tolerance_rad"]:
            return dict(alignment, status="unknown", reason="pose_changed_during_frame_capture")
        detection = cal.detect_charuco(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), board)
        if detection is None:
            return dict(alignment, board_detected=False)
        T_cb, rms = cal.board_pose(detection["corners"], detection["ids"], wrist["K"], wrist["distortion"], board)
        poses = adapter.kin.fk_all(adapter.model, before["q_rad"], gripper_rad=before["gripper_rad"])
        link = np.linalg.inv(poses["gripper_link"]) @ poses[entry["contact_link"]] @ np.append(entry["contact_point_in_link_m"], 1.0)
        point = np.linalg.inv(T_cb) @ np.linalg.inv(wrist["T_link_camera"]) @ link
        T_base_board = poses["gripper_link"] @ wrist["T_link_camera"] @ T_cb
        return {**alignment, "board_detected": True, "corners": int(len(detection["ids"])), "pnp_rms_px": float(rms),
                "contact_point_board_m": [float(v) for v in point[:3]], "board_top_z_via_chain_m": float(T_base_board[2, 3])}
    return measure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--points", nargs="+", help="x,y targets in so101_base (m); default calibration.touch_points_xy_m")
    for flag, key in OVERRIDES.items():
        parser.add_argument("--" + flag.replace("_", "-"), type=float, help=f"override calibration.{key}")
    parser.add_argument("--camera-calibration", default=None, help="camera_calibration.json for the board-frame check")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args(argv)
    deployment = load_deployment(arguments.deployment)
    for flag, key in OVERRIDES.items():
        if getattr(arguments, flag) is not None:
            deployment["calibration"][key] = getattr(arguments, flag)
    if arguments.points:
        deployment["calibration"]["touch_points_xy_m"] = [[float(v) for v in raw.split(",")] for raw in arguments.points]
    # Per-invocation flags obey the same units, bounds and cross-field rules as the file.
    # Validate before constructing the adapter: invalid input must never open the motor bus.
    deployment = validate_deployment(deployment)
    settings = dict(deployment["calibration"])
    points = settings["touch_points_xy_m"]
    adapter = SO101Adapter(deployment, Path(arguments.out).with_suffix(".worker"), "rgb_proprio_v1", start_telemetry=False)
    hook = camera_measurement(adapter, arguments.camera_calibration) if arguments.camera_calibration else None
    report = {"status": "failed"}
    try:
        report = touch_points(adapter, points, settings, label="touch", on_contact=hook,
                              progress=lambda update: print(json.dumps(update), flush=True))
        report["args"] = vars(arguments)
    finally:
        Path(arguments.out).write_text(json.dumps(report, indent=2, default=str))
        adapter.close()
        print("REPORT", arguments.out, report.get("status"), report.get("table_z_nominal_m"), flush=True)


if __name__ == "__main__":
    main()
