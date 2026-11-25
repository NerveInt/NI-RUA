"""SO-101 automatic camera calibration: board -> capture -> solve -> verify.

Runs in the ``so101`` mamba env (LeRobot 0.6.1 + OpenCV with aruco); ``solve`` and ``verify`` are
CPU-only and offline and also run in the ``ni-rua`` env. Design: docs/development/so101-integration.md,
section 5. Output file schema: ``ni_rua.so101.camera_calibration.v1`` (section 3.2).

Steps:
  board             render the ChArUco board (PNG + A4 PDF) to print.
  capture --manual  safest first run: torque OFF, you pose the arm by hand (or with a leader arm via
                    ``--leader-port``), press Enter to record a sample (measured joints + both cameras).
  capture           automatic: the arm visits generated poses that look at the board, slowly (joint-delta
                    cap per command, speed cap, settle wait). Needs an approximate board pose and wrist-camera
                    mount guess (``--seed-calibration`` from a previous manual solve, or explicit flags).
  capture-handheld  fixed-camera (or wrist) intrinsics: hold the board in front of a camera at varied
                    angles and press Enter per frame. Robot not used.
  solve             detect, calibrate intrinsics, eye-in-hand, fixed extrinsic; gate residuals and write
                    camera_calibration.json (+ solve_report.json with per-sample residuals).
  verify            reproject board corners through the solved chain (cross-camera) and report px errors.

Every capture writes into a fresh directory ``<output-root>/so101-calib-<UTC>-<id>`` and never reuses one.

Deployment config (``--config``, JSON; keys may also sit under a top-level ``"so101"`` object):
  port (str, serial port of the follower), robot_id (str, LeRobot calibration id), calibration_dir (optional),
  cameras: {"wrist"|"fixed": {"index_or_path", "unique_id"?, "device_name"?, "width", "height", "fps", "fourcc"?}}
           (bound like the robot worker: an identity must name exactly one present device),
  joint_mapping: {joint: {"sign": +-1, "offset_rad": rad}}  (LeRobot deg -> URDF rad; default identity,
                 must be verified on hardware), motion: {"max_joint_speed_rad_s", "max_command_delta_rad"}.
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
from typing import Any
import uuid

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from ni_rua.robot.calibration.workflow import charuco as cal  # noqa: E402
from ni_rua.robot.calibration.workflow.hand_eye_solve import (  # noqa: E402,F401  (re-exported for callers/tests)
    REPORT_SCHEMA, SESSION_SCHEMA, _detect, board_spec_with_overrides, default_fk_link, handheld_images,
    load_kinematics, load_session, solve_session, source_revision, utc_now, verify_session, write_json)

GRIPPER = "gripper"

DEFAULT_SETTINGS = {k: v for k, v in json.loads(Path(__file__).with_name(
    "so101_calibrate_defaults.json").read_text()).items() if not k.startswith("_")}
# Compatibility aliases; the JSON is the numeric-default owner.
DEFAULT_MAX_JOINT_SPEED_RAD_S = DEFAULT_SETTINGS["capture"]["max_joint_speed_rad_s"]
DEFAULT_MAX_COMMAND_DELTA_RAD = DEFAULT_SETTINGS["capture"]["max_command_delta_rad"]
DEFAULT_COMMAND_RATE_HZ = DEFAULT_SETTINGS["capture"]["rate_hz"]
DEFAULT_LEROBOT_MAX_RELATIVE_TARGET_DEG = DEFAULT_SETTINGS["capture"]["max_relative_target_deg"]


# ----------------------------------------------------------------------------------------------
# Config / session IO (solve/verify and session helpers: ni_rua/robot/model/so101_calibration_solve.py)
# ----------------------------------------------------------------------------------------------

def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("so101", data), dict):
        raise ValueError("deployment config must be an object")
    return dict(data.get("so101", data))


def resolve_settings(args) -> dict:
    """Resolve legacy numeric settings before any hardware access; mutate args to effective values.

    args: parsed CLI namespace. Units/defaults are in each CLI option and the companion JSON.
    Precedence: packaged defaults < equivalent deployment fields < partial --settings < explicit flags.
    Returns {values, sources}; sources identify each winning layer, never imply physical acceptance.
    Example: resolve_settings(build_parser().parse_args(['solve', '--session', 'capture', '--config', 'robot.json'])).
    """
    if args.step not in DEFAULT_SETTINGS:
        return {"values": {}, "sources": {}}
    config = load_config(getattr(args, "config", None))
    calibration, motion = config.get("calibration", {}), config.get("motion", {})
    if not isinstance(calibration, dict) or not isinstance(motion, dict):
        raise ValueError("deployment calibration/motion must be objects")
    mappings = {
        "capture": {"max_joint_speed_rad_s": (motion, "max_joint_speed_rad_s", "motion"),
            "max_command_delta_rad": (motion, "max_command_delta_rad", "motion"),
            "rate_hz": (motion, "command_rate_hz", "motion"),
            "max_relative_target_deg": (config, "max_relative_target_deg", "deployment"),
            "settle_timeout_s": (calibration, "settle_timeout_s", "calibration"),
            "z_floor_m": (calibration, "z_floor_m", "calibration")},
        "capture-handheld": {key: (calibration, field, "calibration") for key, field in {
            "views": "handheld_target_views", "timeout_s": "handheld_timeout_s", "min_corners": "min_board_corners",
            "still_px": "handheld_still_px", "min_novelty": "handheld_min_novelty"}.items()},
        "solve": {key: (calibration, key, "calibration") for key in DEFAULT_SETTINGS["solve"]},
        "verify": {}}
    values = dict(DEFAULT_SETTINGS[args.step])
    sources = {key: "packaged_defaults" for key in values}
    for key, (section, field, owner) in mappings[args.step].items():
        if field in section:
            values[key], sources[key] = section[field], f"config.{owner}.{field}"
    document = {} if not getattr(args, "settings", None) else json.loads(Path(args.settings).expanduser().read_text())
    if not isinstance(document, dict):
        raise ValueError("legacy settings must be an object")
    for section, entries in document.items():
        if section.startswith("_"):
            continue
        if section not in DEFAULT_SETTINGS or not isinstance(entries, dict):
            raise ValueError(f"unknown or invalid legacy settings section: {section}")
        for key, value in entries.items():
            if key.startswith("_"):
                continue
            if key not in DEFAULT_SETTINGS[section]:
                raise ValueError(f"unknown legacy setting: {section}.{key}")
            _validate_setting(section, key, value)
            if section == args.step:
                values[key], sources[key] = value, "settings"
    for key in values:
        if hasattr(args, key):
            values[key], sources[key] = getattr(args, key), "cli"
        _validate_setting(args.step, key, values[key])
    if args.step == "capture":
        if values["tilt_min_deg"] > values["tilt_max_deg"] or values["distance_min_m"] > values["distance_max_m"]:
            raise ValueError("minimum pose tilt/distance must not exceed maximum")
    for key, value in values.items():
        setattr(args, key, value)
    args.effective_settings = {"values": values, "sources": sources}
    return args.effective_settings


def _validate_setting(section, key, value):
    """Validate one documented legacy scalar in its original unit; nullable floors/gates retain None."""
    if value is None and key in ("z_floor_m", "max_fixed_reprojection_px", "max_reprojection_px"):
        return
    default = DEFAULT_SETTINGS[section][key]
    if type(value) not in (int, float) or not math.isfinite(value) or (type(default) is int and type(value) is not int):
        raise ValueError(f"invalid numeric legacy setting: {section}.{key}")
    if key == "hand_eye_score_lever_m" and not 0 <= value <= 10:
        raise ValueError("hand_eye_score_lever_m must be in [0,10] m/rad")
    if key not in ("board_yaw_deg", "z_floor_m") and (value < 0 or (value == 0 and key not in ("seed", "joint_margin_deg", "tilt_min_deg", "tilt_max_deg", "hand_eye_score_lever_m"))):
        raise ValueError(f"invalid bound for legacy setting: {section}.{key}")


def new_session_dir(output_root: str) -> Path:
    root = Path(output_root).expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / f"so101-calib-{stamp}-{uuid.uuid4().hex[:6]}"
    path.mkdir(parents=True, exist_ok=False)
    (path / "images").mkdir()
    return path


# ----------------------------------------------------------------------------------------------
# Hardware (LeRobot) helpers - only used by capture steps
# ----------------------------------------------------------------------------------------------

def make_camera(name: str, cfg: dict):
    """LeRobot camera for deployment camera ``name``, bound like the robot worker binds it.

    ``unique_id`` / ``device_name`` must name exactly one present device (else ``CameraBindingError``; no fallback to
    ``index_or_path``); without them ``index_or_path`` is opened as before. The bound device is printed.
    """
    from lerobot.cameras.configs import ColorMode, Cv2Backends
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
    from ni_rua.robot.backends.so101.camera_devices import binding_summary, list_video_devices, resolve_camera
    binding = resolve_camera(name, cfg, list_video_devices())
    print(f"camera {name}: {binding_summary(binding)}")
    index = binding["opened"]
    camera_cfg = OpenCVCameraConfig(index_or_path=index if isinstance(index, int) else Path(index),
                                    fps=cfg.get("fps"), width=cfg.get("width"), height=cfg.get("height"),
                                    color_mode=ColorMode.BGR, fourcc=cfg.get("fourcc"),
                                    backend=Cv2Backends[binding["api"].upper()])
    return OpenCVCamera(camera_cfg)


def fresh_frame(camera) -> np.ndarray:
    """A frame captured after this call (the first new frame is discarded to drop sensor-buffered exposure)."""
    camera.read()
    return np.asarray(camera.read())


def make_robot(config: dict, args):
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    port = args.port or config.get("port")
    if not port:
        raise SystemExit("serial port missing: set 'port' in --config or pass --port")
    kwargs = dict(port=port, id=args.robot_id or config.get("robot_id"), use_degrees=True,
                  max_relative_target=float(args.max_relative_target_deg),
                  disable_torque_on_disconnect=False)
    if config.get("calibration_dir"):
        kwargs["calibration_dir"] = Path(config["calibration_dir"]).expanduser()
    robot = SO101Follower(SO101FollowerConfig(**kwargs))
    robot.connect(calibrate=False)
    if not robot.is_calibrated:
        robot.disconnect()
        raise SystemExit("LeRobot calibration missing/mismatched for this arm; run lerobot-calibrate first")
    return robot


def read_joints_deg(robot) -> dict[str, float]:
    values = robot.bus.sync_read("Present_Position", num_retry=robot.config.num_read_retries)
    return {k: float(v) for k, v in values.items()}


def hold_position(robot) -> None:
    """Best effort: command the measured position as goal so the arm holds still with torque on."""
    try:
        present = read_joints_deg(robot)
        robot.bus.sync_write("Goal_Position", {j: present[j] for j in cal.ARM_JOINTS})
    except Exception as error:  # pragma: no cover - hardware path
        print(f"WARNING: hold command failed ({error}); check the arm", file=sys.stderr)


def wait_settled(robot, *, tolerance_deg: float, window_s: float, timeout_s: float) -> tuple[bool, dict]:
    """Poll joints until every arm joint moved < ``tolerance_deg`` over ``window_s``; False on ``timeout_s``."""
    start = time.monotonic()
    history: list[tuple[float, dict]] = []
    while True:
        now = time.monotonic()
        joints = read_joints_deg(robot)
        history.append((now, joints))
        history = [h for h in history if now - h[0] <= window_s]
        if now - history[0][0] >= window_s * 0.95:
            span = max(max(h[1][j] for h in history) - min(h[1][j] for h in history) for j in cal.ARM_JOINTS)
            if span < tolerance_deg:
                return True, joints
        if now - start > timeout_s:
            return False, joints
        time.sleep(0.03)


def move_arm(robot, target_deg: dict[str, float], *, speed_deg_s: float, step_deg: float, rate_hz: float) -> None:
    """Linear joint-space interpolation from the measured pose to ``target_deg`` (arm joints only; the gripper
    is never commanded). Each command changes any joint by <= min(step_deg, speed_deg_s / rate_hz)."""
    start = read_joints_deg(robot)
    delta = max(abs(target_deg[j] - start[j]) for j in cal.ARM_JOINTS)
    per_step = min(step_deg, speed_deg_s / rate_hz)
    steps = max(1, math.ceil(delta / per_step))
    for k in range(1, steps + 1):
        goal = {f"{j}.pos": start[j] + (target_deg[j] - start[j]) * k / steps for j in cal.ARM_JOINTS}
        robot.send_action(goal)
        time.sleep(1.0 / rate_hz)


# ----------------------------------------------------------------------------------------------
# capture
# ----------------------------------------------------------------------------------------------

def _session_record(args, config, mode: str, board: dict, mapping: dict, extra: dict | None = None) -> dict:
    try:
        import lerobot
        lerobot_version = getattr(lerobot, "__version__", None)
    except ImportError:
        lerobot_version = None
    return {"schema": SESSION_SCHEMA, "created_at": utc_now(), "mode": mode, "board": board,
            "joint_mapping": mapping, "cameras": config.get("cameras", {}), "lerobot_version": lerobot_version,
            "opencv_version": cv2.__version__, "source": source_revision(),
            "legacy_effective_settings": getattr(args, "effective_settings", {}), **(extra or {})}


def _save_sample(session_dir: Path, index: int, joints_before: dict, joints_after: dict, frames: dict,
                 settled: bool, board, planned_q_rad=None) -> dict:
    images = {}
    for name, frame in frames.items():
        rel = f"images/{index:04d}_{name}.png"
        if not cv2.imwrite(str(session_dir / rel), frame):
            raise OSError(f"failed to write {rel}")
        images[name] = rel
    detected = {name: (lambda d: 0 if d is None else len(d["ids"]))(cal.detect_charuco(frame, board))
                for name, frame in frames.items()}
    record = {"index": index, "captured_at": utc_now(), "joints_deg": joints_before,
              "joints_deg_after": joints_after, "settled": bool(settled), "images": images,
              "planned_q_rad": None if planned_q_rad is None else [float(v) for v in planned_q_rad],
              "detected_corners_at_capture": detected}
    with open(session_dir / "samples.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return record


def _camera_guess(args, kin, model) -> np.ndarray:
    if args.seed_calibration:
        doc = cal.read_calibration_file(args.seed_calibration)
        return np.asarray(doc["cameras"]["wrist"]["T_link_camera"])
    if args.camera_guess_xyz_rpy:
        x, y, z, r, p, yw = args.camera_guess_xyz_rpy
        R = cal.rotvec_to_matrix([0, 0, yw]) @ cal.rotvec_to_matrix([0, p, 0]) @ cal.rotvec_to_matrix([r, 0, 0])
        return cal.make_transform(R, [x, y, z])
    if args.camera_guess_link:
        P = kin.fk_all(model, np.zeros(5))
        print(f"camera mount guess: URDF link {args.camera_guess_link!r}, ASSUMING its +z is the optical axis")
        return cal.invert_transform(P[kin.WRIST_CAMERA_PARENT_LINK]) @ P[args.camera_guess_link]
    raise SystemExit("automatic capture needs a wrist camera mount guess: --seed-calibration FILE (from a manual "
                     "solve), --camera-guess-xyz-rpy X Y Z ROLL PITCH YAW (m, rad, in gripper_link) or "
                     "--camera-guess-link LINK")


def _board_guess(args, board) -> np.ndarray:
    if args.board_center_m is not None:
        return cal.board_frame_from_center(args.board_center_m, board, yaw_rad=math.radians(args.board_yaw_deg))
    if args.seed_calibration:
        report = Path(args.seed_calibration).with_name("solve_report.json")
        if report.exists():
            return np.asarray(json.loads(report.read_text(encoding="utf-8"))["T_base_board"])
    raise SystemExit("automatic capture needs the board location: --board-center-m X Y Z [--board-yaw-deg] or a "
                     "--seed-calibration whose solve_report.json records T_base_board")


def plan_poses(args, config, board, mapping, q_start_rad) -> list[dict]:
    kin, model = load_kinematics()
    limits = kin.joint_limits_rad(model)
    margin = math.radians(args.joint_margin_deg)
    shrunk = {j: (limits[j][0] + margin, limits[j][1] - margin) for j in cal.ARM_JOINTS}
    T_link_tcp = cal.invert_transform(kin.fk(model, np.zeros(5), kin.WRIST_CAMERA_PARENT_LINK)) @ \
        kin.fk(model, np.zeros(5), kin.TCP_LINK)

    def ik(T_link, seed):
        # fewer restarts than the default: the generator already tries many rolls per viewpoint
        return kin.ik(model, T_link @ T_link_tcp, seed, limits=shrunk, max_iterations=100, restarts=2)

    def fk(q):
        return kin.fk(model, q, kin.WRIST_CAMERA_PARENT_LINK)

    # base_link and shoulder_link origins do not move vertically (base on the table); every other link
    # origin, including the TCP on the fixed jaw's inner fingertip face, must stay above the z floor.
    static = {kin.BASE_LINK, "shoulder_link"}

    def fk_points(q):
        return [T for name, T in kin.fk_all(model, q).items() if name not in static]

    return cal.generate_calibration_poses(
        _board_guess(args, board), board, ik, fk, q_start_rad, [limits[j] for j in cal.ARM_JOINTS],
        T_link_camera_guess=_camera_guess(args, kin, model), n_poses=args.n_poses,
        tilt_range_rad=(math.radians(args.tilt_min_deg), math.radians(args.tilt_max_deg)),
        distance_range_m=(args.distance_min_m, args.distance_max_m), joint_margin_rad=margin,
        # The legacy generator expects a scalar; -inf expresses the explicitly disabled
        # deployment screen without changing any pose target or inventing a floor.
        z_floor_m=-math.inf if args.z_floor_m is None else args.z_floor_m, fk_points=fk_points, seed=args.seed)


def path_floor_violation(kin, model, q_from, q_to, z_floor_m: float, *, step_rad: float = math.radians(2.0)):
    """First interpolated joint configuration (linear joint path, as executed by ``move_arm``) at which a moving
    link origin goes below ``z_floor_m`` (m); ``None`` if the whole segment is clear."""
    q_from, q_to = np.asarray(q_from, dtype=float), np.asarray(q_to, dtype=float)
    steps = max(1, math.ceil(float(np.max(np.abs(q_to - q_from))) / step_rad))
    for k in range(steps + 1):
        q = q_from + (q_to - q_from) * k / steps
        for name, T in kin.fk_all(model, q).items():
            if name not in (kin.BASE_LINK, "shoulder_link") and T[2, 3] < z_floor_m:
                return {"q_rad": q.tolist(), "link": name, "z_m": float(T[2, 3])}
    return None


def check_start_within_limits(kin, model, q_rad, margin_rad: float = math.radians(2.0)) -> None:
    """Abort before any motion if measured joints (after joint_mapping) are outside the URDF limits: this
    almost always means the LeRobot-degree -> URDF-radian mapping (sign/offset) is wrong for this arm."""
    limits = kin.joint_limits_rad(model)
    bad = {j: round(math.degrees(q), 1) for j, q in zip(cal.ARM_JOINTS, q_rad)
           if not limits[j][0] - margin_rad <= q <= limits[j][1] + margin_rad}
    if bad:
        raise RuntimeError(f"measured joints outside URDF limits (deg, after joint_mapping): {bad}; verify "
                           "config joint_mapping (sign/offset) against the URDF before automatic motion")


def cmd_capture(args) -> int:
    config = load_config(args.config)
    board_spec = cal.validate_board_spec(cal.DEFAULT_BOARD_SPEC)
    board = cal.make_board(board_spec)
    mapping = cal.normalize_joint_mapping(config.get("joint_mapping"))
    speed_deg_s = math.degrees(args.max_joint_speed_rad_s)
    step_deg = math.degrees(args.max_command_delta_rad)
    cam_cfgs = {k: v for k, v in config.get("cameras", {}).items() if k in ("wrist", "fixed")}
    if "wrist" not in cam_cfgs:
        raise SystemExit("config.cameras.wrist is required for capture")

    if args.dry_run:
        if args.manual:
            raise SystemExit("--dry-run applies to automatic capture only")
        q0 = np.asarray(args.dry_run_q_rad if args.dry_run_q_rad else np.zeros(5), dtype=float)
        poses = plan_poses(args, config, board, mapping, q0)
        _print_plan(poses)
        return 0

    session_dir = new_session_dir(args.output_root)
    print(f"session directory: {session_dir}")
    cameras = {name: make_camera(name, cfg) for name, cfg in cam_cfgs.items()}
    robot = None
    try:
        for camera in cameras.values():
            camera.connect()
        robot = make_robot(config, args)
        start_deg = read_joints_deg(robot)
        write_json(session_dir / "session.json", _session_record(
            args, config, "manual" if args.manual else "automatic", board_spec, mapping,
            {"start_joints_deg": start_deg, "motion": {"max_joint_speed_deg_s": speed_deg_s,
                                                       "max_command_delta_deg": step_deg,
                                                       "rate_hz": args.rate_hz}}))
        if args.manual:
            _capture_manual(args, config, robot, cameras, session_dir, board)
        else:
            _capture_auto(args, config, robot, cameras, session_dir, board, mapping, start_deg,
                          speed_deg_s, step_deg)
    except BaseException as error:
        print(f"ABORT: {type(error).__name__}: {error}", file=sys.stderr)
        if robot is not None and not args.manual:
            hold_position(robot)
            print("torque kept ON, holding the measured position. Support the arm before releasing torque.",
                  file=sys.stderr)
        raise
    finally:
        for camera in cameras.values():
            try:
                camera.disconnect()
            except Exception:
                pass
        if robot is not None:
            try:
                # Manual mode already runs torque-off; automatic mode keeps holding unless asked.
                robot.bus.disconnect(disable_torque=bool(args.manual or args.release_torque_at_end))
            except Exception as error:
                print(f"WARNING: disconnect failed: {error}", file=sys.stderr)
    print(f"samples saved under {session_dir}; next: so101_calibrate.py solve --session {session_dir}")
    return 0


def _print_plan(poses: list[dict]) -> None:
    print(f"{len(poses)} planned poses (joint deg | view angle deg | distance m | ik pos err mm):")
    for i, p in enumerate(poses):
        q = ", ".join(f"{math.degrees(v):7.1f}" for v in p["q_rad"])
        print(f"  {i:2d}: [{q}] | {math.degrees(p['view_angle_rad']):5.1f} | {p['distance_m']:.3f} | "
              f"{p['ik_position_error_m'] * 1000:5.1f}")


def _capture_auto(args, config, robot, cameras, session_dir, board, mapping, start_deg, speed_deg_s, step_deg):
    q_start = cal.lerobot_deg_to_urdf_rad(start_deg, mapping)
    kin, model = load_kinematics()
    check_start_within_limits(kin, model, q_start)
    if not config.get("joint_mapping"):
        print("WARNING: config has no joint_mapping; using identity (LeRobot deg == URDF deg). Verify on hardware "
              "(capture --manual + solve first) before trusting automatic poses.")
    poses = plan_poses(args, config, board, mapping, q_start)
    if len(poses) < 3:
        raise RuntimeError(f"only {len(poses)} reachable calibration poses; check board/camera guesses")
    q_prev = q_start
    for i, pose in enumerate(poses + [{"q_rad": q_start}]):
        hit = None if args.z_floor_m is None else path_floor_violation(kin, model, q_prev, pose["q_rad"], args.z_floor_m)
        if hit is not None:
            raise RuntimeError(f"joint path to pose {i} dips below z floor: {hit}; no motion was sent")
        q_prev = pose["q_rad"]
    _print_plan(poses)
    if not args.yes and input("Clear the workspace. Type 'yes' to move the arm: ").strip().lower() != "yes":
        print("not confirmed; no motion")
        return
    for i, pose in enumerate(poses):
        target = cal.urdf_rad_to_lerobot_deg(pose["q_rad"], mapping)
        print(f"pose {i + 1}/{len(poses)}")
        move_arm(robot, target, speed_deg_s=speed_deg_s, step_deg=step_deg, rate_hz=args.rate_hz)
        settled, before = wait_settled(robot, tolerance_deg=args.settle_tolerance_deg,
                                       window_s=args.settle_window_s, timeout_s=args.settle_timeout_s)
        frames = {name: fresh_frame(cam) for name, cam in cameras.items()}
        after = read_joints_deg(robot)
        record = _save_sample(session_dir, i, before, after, frames, settled, board, pose["q_rad"])
        print(f"  settled={settled} corners={record['detected_corners_at_capture']}")
    print("returning to the start pose")
    move_arm(robot, {j: start_deg[j] for j in cal.ARM_JOINTS}, speed_deg_s=speed_deg_s, step_deg=step_deg,
             rate_hz=args.rate_hz)
    wait_settled(robot, tolerance_deg=args.settle_tolerance_deg, window_s=args.settle_window_s,
                 timeout_s=args.settle_timeout_s)


def _capture_manual(args, config, robot, cameras, session_dir, board):
    leader, stop, paused, thread = None, threading.Event(), threading.Event(), None
    if args.leader_port:
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
        leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id=args.leader_id, use_degrees=True))
        leader.connect(calibrate=False)

        def mirror():
            while not stop.is_set():
                if not paused.is_set():
                    action = leader.get_action()
                    robot.send_action({k: v for k, v in action.items()
                                       if k.removesuffix(".pos") in cal.ARM_JOINTS})
                time.sleep(1.0 / 30.0)

        print("leader arm mirrors the follower arm joints (gripper untouched)")
        thread = threading.Thread(target=mirror, daemon=True)
        thread.start()
    else:
        input("Manual mode turns torque OFF: hold/support the arm now, then press Enter.")
        robot.bus.disable_torque()
    index = 0
    try:
        while True:
            line = input(f"[{index} samples] pose the arm so the wrist camera sees the board; Enter = capture, "
                         "q = finish: ").strip().lower()
            if line == "q":
                break
            paused.set()
            settled, before = wait_settled(robot, tolerance_deg=args.settle_tolerance_deg,
                                           window_s=args.settle_window_s, timeout_s=args.settle_timeout_s)
            frames = {name: fresh_frame(cam) for name, cam in cameras.items()}
            after = read_joints_deg(robot)
            paused.clear()
            record = _save_sample(session_dir, index, before, after, frames, settled, board)
            print(f"  settled={settled} corners={record['detected_corners_at_capture']}")
            index += 1
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=2)
        if leader is not None:
            leader.disconnect()


def cmd_capture_handheld(args) -> int:
    config = load_config(args.config)
    cfg = config.get("cameras", {}).get(args.camera)
    if cfg is None:
        raise SystemExit(f"config.cameras.{args.camera} missing")
    board = cal.make_board()
    session_dir = Path(args.session).expanduser().resolve() if args.session else new_session_dir(args.output_root)
    if not (session_dir / "session.json").exists():
        write_json(session_dir / "session.json",
                   _session_record(args, config, "handheld", cal.validate_board_spec(cal.DEFAULT_BOARD_SPEC),
                                   cal.normalize_joint_mapping(config.get("joint_mapping"))))
    out = session_dir / "handheld" / args.camera
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "legacy_effective_settings.json", getattr(args, "effective_settings", {}))
    index = len(list(out.glob("*.png")))
    camera = make_camera(args.camera, cfg)
    camera.connect()
    if args.auto:
        try:
            return _auto_handheld(camera, board, out, index, args)
        finally:
            camera.disconnect()
    try:
        print("Hold the board still at varied distances, tilts (15-45 deg) and image positions/corners.")
        while input(f"[{index}] Enter = capture, q = finish: ").strip().lower() != "q":
            frame = fresh_frame(camera)
            detection = cal.detect_charuco(frame, board)
            cv2.imwrite(str(out / f"{index:04d}.png"), frame)
            print(f"  corners={0 if detection is None else len(detection['ids'])}")
            index += 1
    finally:
        camera.disconnect()
    print(f"handheld images in {out}")
    return 0


def _view_signature(detection, image_size) -> np.ndarray:
    """Board view descriptor for novelty (moved to calibration.workflow.capture.view_signature)."""
    from ni_rua.robot.calibration.workflow.capture import view_signature
    return view_signature(detection, image_size)


def _auto_handheld(camera, board, out: Path, index: int, args) -> int:
    """Keep a frame only when the board is held still (corners move < still_px between frames) and the view is new
    (signature distance >= min_novelty to every kept view). Stops at --views or --timeout-s. No key presses.
    Selection logic: :class:`ni_rua.robot.calibration.workflow.capture.HandheldViewSelector`."""
    import time
    from ni_rua.robot.calibration.workflow.capture import HandheldViewSelector
    selector = HandheldViewSelector(target_views=args.views, min_corners=args.min_corners, still_px=args.still_px,
                                    min_novelty=args.min_novelty)
    start = time.monotonic()
    print(f"AUTO: hold the board in front of the {args.camera} camera; move it to new positions/tilts and pause ~1 s at each.",
          flush=True)
    while not selector.complete and time.monotonic() - start < args.timeout_s:
        frame = fresh_frame(camera)
        detection = cal.detect_charuco(frame, board, min_corners=args.min_corners)
        if not selector.offer(detection, (frame.shape[1], frame.shape[0]))["kept"]:
            continue
        cv2.imwrite(str(out / f"{index:04d}.png"), frame)
        print(f"  kept view {selector.kept_count}/{args.views}: corners={len(detection['ids'])}", flush=True)
        index += 1
    print(f"AUTO done: {selector.kept_count} views in {time.monotonic() - start:.0f} s -> {out}", flush=True)
    return 0 if selector.kept_count >= 8 else 3


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------

def cmd_board(args) -> int:
    sys.path.insert(0, str(REPO / "scripts"))
    import so101_charuco_board
    paths = so101_charuco_board.write_board_files(args.output_dir)
    so101_charuco_board.print_instructions(paths)
    return 0


def _thresholds(args) -> dict:
    return {"max_intrinsics_rms_px": args.max_intrinsics_rms_px,
            "max_hand_eye_translation_m": args.max_hand_eye_translation_m,
            "max_hand_eye_rotation_rad": args.max_hand_eye_rotation_rad,
            "max_fixed_reprojection_px": args.max_fixed_reprojection_px}


def _mapping_arg(value):
    if value is None:
        return None
    path = Path(value)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else json.loads(value)
    return data.get("joint_mapping", data)


def cmd_solve(args) -> int:
    write_json(Path(args.session) / "legacy_solve_settings.json", getattr(args, "effective_settings", {}))
    try:
        report = solve_session(args.session, output=args.output, square_m=args.square_m, marker_m=args.marker_m,
                               joint_mapping=_mapping_arg(args.joint_mapping), thresholds=_thresholds(args),
                               force=args.force_save, overwrite=args.overwrite, intrinsics_from=args.intrinsics_from,
                               exclude=set(args.exclude or []), include_unsettled=args.include_unsettled,
                               max_pnp_reprojection_px=args.max_pnp_reprojection_px,
                               hand_eye_score_lever_m=args.hand_eye_score_lever_m,
                               fixed_intrinsics_from_wrist=args.fixed_intrinsics_from_wrist)
    except cal.CalibrationGateError as error:
        print(f"NOT SAVED: {error}\nsee {Path(args.session) / 'solve_report.json'}; re-capture or pass --force-save",
              file=sys.stderr)
        return 2
    he = report["hand_eye"]
    print(f"hand-eye: method={he['method']} samples={he['samples']} translation_rms={he['translation_rms_m'] * 1000:.2f} mm"
          f" (max {he['translation_max_m'] * 1000:.2f}) rotation_rms={he['rotation_rms_rad']:.4f} rad"
          f" (max {he['rotation_max_rad']:.4f})")
    for name, cam in report["cameras"].items():
        if "error" in cam:
            print(f"{name}: {cam['error']}")
        else:
            print(f"{name}: intrinsics rms={cam['intrinsics_rms_px']:.3f} px ({cam['intrinsics_source']})")
    if "fixed" in report:
        print(f"fixed: chain reprojection={report['fixed']['chain_reprojection_px']:.3f} px")
    if report["skipped"]:
        print(f"skipped samples: {report['skipped']}")
    if report.get("forced"):
        print(f"WARNING: saved with --force-save despite: {report['gate_failures']}")
    print(f"calibration: {report['calibration_file']}")
    return 0


def cmd_verify(args) -> int:
    result = verify_session(args.session, args.calibration, joint_mapping=_mapping_arg(args.joint_mapping))
    result["legacy_effective_settings"] = getattr(args, "effective_settings", {})
    print(json.dumps(result["summary"], indent=2))
    if args.output:
        write_json(Path(args.output), result)
    worst = max((v["rms_px"] for v in result["summary"].values() if v), default=float("nan"))
    if args.max_reprojection_px is not None and not worst <= args.max_reprojection_px:
        print(f"FAIL: worst rms {worst:.3f} px > {args.max_reprojection_px}", file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)

    p = sub.add_parser("board", help="render the ChArUco board PNG/PDF")
    p.add_argument("--output-dir", default=".")
    p.set_defaults(func=cmd_board)

    p = sub.add_parser("capture", help="record calibration samples with the robot")
    p.add_argument("--config", required=True, help="deployment JSON (port, robot_id, cameras, joint_mapping, motion)")
    p.add_argument("--output-root", required=True, help="a fresh session directory is created inside it")
    p.add_argument("--port", help="override follower serial port")
    p.add_argument("--robot-id", help="override LeRobot calibration id")
    p.add_argument("--manual", action="store_true", help="torque off; pose the arm by hand, Enter to capture")
    p.add_argument("--leader-port", help="manual mode: mirror an SO-101 leader arm instead of torque-off")
    p.add_argument("--leader-id", help="LeRobot calibration id of the leader arm")
    p.add_argument("--seed-calibration", help="previous camera_calibration.json: wrist mount guess and board pose")
    p.add_argument("--camera-guess-xyz-rpy", type=float, nargs=6,
                   help="wrist camera pose guess in gripper_link: x y z (m) roll pitch yaw (rad, OpenCV axes)")
    p.add_argument("--camera-guess-link", help="URDF link used as camera guess (its +z assumed optical axis)")
    p.add_argument("--board-center-m", type=float, nargs=3, help="approximate board centre in base (m)")
    p.add_argument("--board-yaw-deg", type=float, help="board x axis yaw about base z (deg)")
    p.add_argument("--n-poses", type=int, help="number of automatic poses (15-25 recommended)")
    p.add_argument("--tilt-min-deg", type=float, help="min view tilt from board normal (deg)")
    p.add_argument("--tilt-max-deg", type=float, help="max view tilt from board normal (deg)")
    p.add_argument("--distance-min-m", type=float, help="min camera-board distance (m)")
    p.add_argument("--distance-max-m", type=float, help="max camera-board distance (m)")
    p.add_argument("--joint-margin-deg", type=float, help="URDF joint limits shrunk by this (deg)")
    p.add_argument("--z-floor-m", type=float,
                   help="min base-frame z (m) of every moving link origin (incl. TCP) and the camera; base_link origin "
                   "is at z=0 - set to table height relative to it + clearance")
    p.add_argument("--seed", type=int, help="pose generator RNG seed")
    p.add_argument("--max-joint-speed-rad-s", type=float, help=f"joint speed cap (rad/s, default config or "
                   f"{DEFAULT_MAX_JOINT_SPEED_RAD_S:.3f})")
    p.add_argument("--max-command-delta-rad", type=float, help=f"max joint goal change per command (rad, default "
                   f"config or {DEFAULT_MAX_COMMAND_DELTA_RAD:.4f})")
    p.add_argument("--max-relative-target-deg", type=float, help="LeRobot hardware-side goal-minus-present clamp (deg)")
    p.add_argument("--rate-hz", type=float, help="command rate (Hz)")
    p.add_argument("--settle-tolerance-deg", type=float, help="settled: joint span below this (deg)")
    p.add_argument("--settle-window-s", type=float, help="settle observation window (s)")
    p.add_argument("--settle-timeout-s", type=float, help="give up settling after this (s)")
    p.add_argument("--release-torque-at-end", action="store_true",
                   help="automatic mode: disable torque on exit (default keeps holding)")
    p.add_argument("--yes", action="store_true", help="skip the motion confirmation prompt")
    p.add_argument("--dry-run", action="store_true", help="plan automatic poses only; no hardware")
    p.add_argument("--dry-run-q-rad", type=float, nargs=5, help="start joints for --dry-run (rad)")
    p.set_defaults(func=cmd_capture)

    p = sub.add_parser("capture-handheld", help="hand-held board frames for one camera's intrinsics")
    p.add_argument("--config", required=True)
    p.add_argument("--camera", choices=("fixed", "wrist"), default="fixed")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--session", help="existing session directory (images go to <session>/handheld/<camera>)")
    g.add_argument("--output-root", help="create a fresh session directory here")
    p.add_argument("--auto", action="store_true", help="no key presses: keep still, novel views automatically")
    p.add_argument("--views", type=int, help="auto: stop after this many kept views")
    p.add_argument("--timeout-s", type=float, help="auto: give up after this many seconds")
    p.add_argument("--min-corners", type=int, help="auto: ChArUco corners required per view")
    p.add_argument("--still-px", type=float, help="auto: mean corner motion between frames to count as still (px)")
    p.add_argument("--min-novelty", type=float,
                   help="auto: minimum view-signature distance to every kept view (normalised centre/size/orientation)")
    p.set_defaults(func=cmd_capture_handheld)

    p = sub.add_parser("solve", help="offline solve of a capture session")
    p.add_argument("--config", help="optional deployment JSON; calibration thresholds override packaged defaults")
    p.add_argument("--session", required=True)
    p.add_argument("--output", help="calibration file path (default <session>/camera_calibration.json)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing calibration file")
    p.add_argument("--square-m", type=float, help="measured printed square edge (m); marker scaled likewise")
    p.add_argument("--marker-m", type=float, help="measured printed marker edge (m)")
    p.add_argument("--joint-mapping", help="JSON file or string overriding the session joint_mapping")
    p.add_argument("--intrinsics-from", help="reuse K/distortion per camera from an existing calibration file")
    p.add_argument("--exclude", type=int, nargs="*", help="sample indices to leave out")
    p.add_argument("--include-unsettled", action="store_true", help="also use samples captured while moving")
    p.add_argument("--fixed-intrinsics-from-wrist", action="store_true",
                   help="fixed camera has no varied views (static board): reuse wrist intrinsics (same camera model); "
                        "recorded as unverified and checked by the fixed chain reprojection gate")
    p.add_argument("--max-pnp-reprojection-px", type=float,
                   help="skip wrist samples whose board PnP RMS exceeds this (px)")
    p.add_argument("--hand-eye-score-lever-m", type=float,
                   help="hand-eye method ranking coefficient (m/rad, [0,10], default 0.1); 0 uses translation only; not an acceptance gate")
    p.add_argument("--max-intrinsics-rms-px", type=float, help="gate: intrinsics RMS (px)")
    p.add_argument("--max-hand-eye-translation-m", type=float, help="gate: hand-eye RMS spread (m)")
    p.add_argument("--max-hand-eye-rotation-rad", type=float, help="gate: hand-eye RMS spread (rad)")
    p.add_argument("--max-fixed-reprojection-px", type=float,
                   help="gate: fixed camera chain reprojection (px); default not gated")
    p.add_argument("--force-save", action="store_true", help="write even if gates fail (residuals recorded)")
    p.set_defaults(func=cmd_solve)

    p = sub.add_parser("verify", help="cross-camera reprojection check")
    p.add_argument("--session", required=True, help="session to check (prefer a fresh held-out capture)")
    p.add_argument("--calibration", required=True)
    p.add_argument("--joint-mapping")
    p.add_argument("--output", help="write the full per-sample result JSON here")
    p.add_argument("--max-reprojection-px", type=float, help="exit 2 if any summary RMS exceeds this (px)")
    p.set_defaults(func=cmd_verify)
    for step, defaults in DEFAULT_SETTINGS.items():
        command = sub.choices[step]
        command.add_argument("--settings", help="partial legacy numeric-settings JSON; overrides deployment values, explicit flags win")
        command.add_argument("--print-settings", action="store_true", help="resolve and print effective settings/sources, then exit without hardware or solving")
        for action in command._actions:
            if action.dest in defaults:
                action.default = argparse.SUPPRESS
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    effective = resolve_settings(args)
    if getattr(args, "print_settings", False):
        print(json.dumps(effective, indent=2))
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
