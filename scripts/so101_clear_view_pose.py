"""Find an SO-101 arm pose that leaves the table visible to the fixed camera (offline: calibration + FK, no hardware).

Searches joint configurations (``robot_view.clear_view_search``) inside the planner's joint limits (URDF limits minus
the deployment's ``motion.joint_limit_margin_rad``), with the TCP inside the deployment ``workspace`` box and no
moving link mesh below the table plane, and ranks them by how much of the fixed camera's table ROI (the table within
the arm's horizontal reach, robot base excluded) the arm's FK silhouette covers. It prints the Pareto front of
(coverage, largest joint travel from ``home.q_rad``): the first line is the least-covering pose with the least travel
among those; later lines trade coverage for shorter moves. For the chosen pose it also solves the local IK of its TCP
pose from the home pose and reports the coverage of that solution, because robot_action plans the TCP pose and may
reach it with other joints.

Nothing is written to the deployment: copy the printed ``perception`` snippet into it (``clear_view_q_rad``, rad,
URDF convention) and restart the gateway; the motion instructions then name the pose's TCP target. Re-run after a
fixed-camera recalibration. Example (NI-RUA env, reads only)::

    PYTHONPATH=. mamba run -n ni-rua python scripts/so101_clear_view_pose.py \\
        --deployment ~/.cache/ni-rua/so101/so101.curobo.local.json --report /tmp/clear-view.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ni_rua.perception.external import robot_view  # noqa: E402
from ni_rua.robot.backends.so101.deployment import ARM_JOINTS, load_deployment  # noqa: E402
from ni_rua.robot.backends.so101.driver import gripper_rad_from_normalized  # noqa: E402
from ni_rua.robot.calibration.workflow import charuco as cal  # noqa: E402
from ni_rua.robot.model import so101_kinematics as kin  # noqa: E402


def camera_of(calibration, name):
    """dict(K, T, width, height) of a calibration camera (``T`` = T_base_camera; fixed cameras only)."""
    entry = calibration["cameras"][name]
    if "T_base_camera" not in entry:
        raise SystemExit(f"camera {name} has no static T_base_camera (only a fixed camera can be cleared)")
    width, height = entry["image_size"]
    return dict(K=np.asarray(entry["K"], float), T=np.asarray(entry["T_base_camera"], float), width=int(width),
                height=int(height))


def ik_check(model, chosen, limits, home):
    """The local IK solution of the chosen pose's TCP target from the home pose (rad) and its residuals."""
    target = kin.fk(model, chosen["q_rad"])
    found = kin.ik(model, target, home, limits=limits, position_tolerance_m=1e-3, orientation_tolerance_rad=1e-2)
    return dict(q_rad=[round(v, 4) for v in found["q_rad"]], converged=found["converged"],
                position_error_m=round(found["position_error_m"], 5),
                orientation_error_rad=round(found["orientation_error_rad"], 5),
                max_joint_difference_rad=round(float(np.max(np.abs(np.subtract(found["q_rad"], chosen["q_rad"])))), 4))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deployment", required=True, help="SO-101 deployment file (read only).")
    parser.add_argument("--calibration", help="Camera calibration file (default: the deployment's camera_calibration_file).")
    parser.add_argument("--camera", default="fixed", help="Static camera whose view is cleared (default fixed).")
    parser.add_argument("--table-z-m", type=float, help="Table plane z (m, so101_base); default: the calibration's "
                                                       "joint_calibration.table_z_m.")
    parser.add_argument("--grid", type=int, nargs=4, default=[13, 9, 9, 7], metavar=("PAN", "LIFT", "ELBOW", "FLEX"),
                        help="Samples across the limits of shoulder_pan, shoulder_lift, elbow_flex, wrist_flex "
                             "(default 13 9 9 7; wrist_roll stays at --wrist-roll-rad).")
    parser.add_argument("--wrist-roll-rad", type=float, default=0.0, help="wrist_roll of every candidate (rad, default 0).")
    parser.add_argument("--gripper-normalized", type=float, help="Jaw goal 0..1 the silhouette is rendered with "
                                                                "(default: home.gripper_normalized).")
    parser.add_argument("--report", help="Also write the front and the IK check as JSON to this new file.")
    args = parser.parse_args(argv)
    deployment = load_deployment(args.deployment)
    path = args.calibration or deployment.get("camera_calibration_file")
    if not path:
        raise SystemExit("no calibration file: pass --calibration")
    calibration = cal.read_calibration_file(Path(path).expanduser())
    table_z = args.table_z_m
    if table_z is None:
        table_z = (calibration.get("joint_calibration") or {}).get("table_z_m")
        if table_z is None:
            raise SystemExit("the calibration has no joint_calibration.table_z_m: pass --table-z-m")
    model = kin.load_model()
    margin = deployment["motion"]["joint_limit_margin_rad"]
    limits = {name: (low + margin, high - margin) for name, (low, high) in kin.joint_limits_rad(model).items()
              if name in ARM_JOINTS}
    home = list(deployment["home"]["q_rad"])
    goal = deployment["home"]["gripper_normalized"] if args.gripper_normalized is None else args.gripper_normalized
    gripper_rad = gripper_rad_from_normalized(deployment, goal)
    camera = camera_of(calibration, args.camera)
    front = robot_view.clear_view_search(camera, float(table_z), limits_rad=limits, home_q_rad=home,
                                         workspace_min_m=deployment["workspace"]["min_m"],
                                         workspace_max_m=deployment["workspace"]["max_m"], gripper_rad=gripper_rad,
                                         grid=tuple(args.grid), wrist_roll_rad=args.wrist_roll_rad)
    if not front:
        raise SystemExit("no candidate inside the workspace box above the table; widen --grid")
    home_coverage = robot_view.pose_coverage(camera, float(table_z), home, gripper_rad)
    print(f"table z {table_z:.4f} m; reach {robot_view.reach_facts()['reach_m']} m; home coverage "
          f"{home_coverage if home_coverage is None else round(home_coverage, 4)}")
    print("coverage  travel_rad  q_rad                                     tcp_position_m")
    for item in front:
        print(f"{item['coverage']:8.4f}  {item['travel_rad']:10.3f}  {str(item['q_rad']):40s}  {item['tcp_position_m']}")
    chosen = front[0]
    check = ik_check(model, chosen, limits, home)
    check["coverage"] = robot_view.pose_coverage(camera, float(table_z), check["q_rad"], gripper_rad)
    print("IK of the chosen TCP pose from home:", json.dumps(check))
    print("deployment snippet:", json.dumps({"perception": {"clear_view_q_rad": chosen["q_rad"]}}))
    if args.report:
        out = Path(args.report).expanduser()
        if out.exists():
            raise SystemExit(f"{out} exists; choose a new report path")
        out.write_text(json.dumps(dict(table_z_m=table_z, camera=args.camera, home_q_rad=home,
                                       home_coverage=home_coverage, gripper_rad=gripper_rad, front=front,
                                       ik_check=check, reach=robot_view.reach_facts()), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
