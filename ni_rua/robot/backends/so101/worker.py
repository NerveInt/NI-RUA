"""SO-101 real-hardware worker. Runs in the ``so101`` interpreter and owns the
serial bus and cameras; the parent talks to it only through JSONL.

Clock: real time. ``sim_step`` is the count of dispatched joint-goal commands
(``step_domain = so101_control_step``), never simulated time. Every tuning
number comes from the validated deployment file.

Receipt honesty rules implemented here:
- IK residuals are checked against the requested stage tolerance before any
  command is sent (``planning_failed`` with ``physics_advanced=False``).
- Arrival is verified from measured joints (FK) against the request's
  ``position_tolerance_m`` / ``orientation_tolerance_rad``; true residuals and
  ``effective_tolerances`` are always reported.
- Stop sends the measured position as the goal and is confirmed only when the
  measured joints settle; otherwise ``stop_confirmation.confirmed=False``.
- A driver/serial exception after the first command makes the status
  ``unknown``. Nothing is ever reported ``completed`` without measured arrival.
- A move stopped by a stop signal fails with ``tracking_error`` and a
  ``contact_block_summary``; only a ``contact_stop`` stage holds the measured
  pose and is then judged by the same arrival rule (``SO101Adapter._move``).
- Every executed action returns ``before_observation``: one observation
  (joints, then one frame per camera) taken before its first command.
- A move runs on the straight joint-space line to its cached IK target
  (``ik_plan.joint_interpolation``); preview evaluates the same rows by FK
  (``stage_plans[<move>].path``) and, with the cuRobo service, checks them for
  collision (``path_check``; advisory, never blocks or re-routes). Execution
  reports ``path_start_deviation_rad``, how far its measured start was from
  the start the preview assumed.
- A batch that fails at a move mirrors that move's measured TCP residual to
  receipt level (``endpoint_residual_m``; ``in_hand_tracking_error`` when the
  closed jaw still measures wider than an empty close).
- Every move reports ``transit``: how far the measured TCP left the commanded
  path while it was streamed (``motion_compensation.transit_deviation``). The
  preview's path facts and ``path_check`` describe the commanded path only.
- Goal offsets added on top of the commanded path (deployment
  ``motion_compensation``: carried sag offset, gravity feedforward) are
  reported per move as ``path_compensation``; tracking, stall and divergence
  guards and every residual are judged against the path without them.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import queue
import signal
import stat
import sys
import threading
import time
import traceback
import uuid

import numpy as np

from ni_rua.robot.ipc import WIRE_PREFIX, serve_worker
from ni_rua.robot.backends.so101 import contract
from ni_rua.robot.backends.so101.deployment import STEP_DOMAIN, hold_squeeze, load_deployment, public_limits
from ni_rua.robot.backends.so101.driver import (ARM_JOINTS, ENCODER_RESOLUTION_RAD, Clock,
                                                 gripper_normalized_from_rad, gripper_rad_from_normalized)
from ni_rua.robot.backends.so101.motion_compensation import contact_stage, transit_deviation
from ni_rua.robot.model.embodiments import (PREVIEW_REVIEW_CLEAN, PREVIEW_REVIEW_DOWNGRADES, SO101,
                                           preview_reviews)
from ni_rua.robot.motion.batch_scheduler import (ArmLayout, BatchScheduler, jaw_pauses, order_stages, pause_receipt,
                                                 summarize_batch)
from ni_rua.robot.motion.grip_guard import (CARRY_SLIP_GUARD_FIELDS, CARRY_SLIP_REASON, CLOSING_JAW_GOAL_BELOW,
                                           DOWNWARD_MOVE_DZ_M, GRIP_SQUEEZE_FIELD, JAW_NOT_CLOSED_DELTA,
                                           grip_facts_from_measurements, stage_grip_guard,
                                           validate_batch_carry_slip_guard)
from ni_rua.robot.motion.gripper_motion import synchronize
from ni_rua.robot.motion.motion_tolerance import orientation_is_free

# STEP_DOMAIN (imported from deployment, the profile's step_domain): sim_step counts dispatched joint-goal commands.
JOINT_NAMES = (*ARM_JOINTS, "gripper")
# board_check is read-only (latest preview frames, no bus access) so it may run while a
# long main-thread operation (calibrate, an action) is in progress. Keep in sync with backend.py.
# handheld_capture start/status/stop only starts/stops a background sampler of the existing fixed
# camera source (no bus access), so it is answered off the main thread too; it refuses to start
# while an action is active.
# calib_workflow_status / calib_step_status read state and latest frames only; calib_step_cancel stops the
# running workflow step like hold. Motion steps (calib_step_start), calib_apply and calib_rollback are serialized.
PREEMPTIVE_OPERATOR_COMMANDS = ("hold", "relax", "set_runtime_limits", "torque_on", "board_check", "handheld_capture",
                                "calib_workflow_status", "calib_step_status", "calib_step_cancel")
CALIBRATION_WORKFLOW_COMMANDS = ("calib_workflow_status", "calib_workflow_begin", "calib_step_start", "calib_step_status",
                                 "calib_step_cancel", "calib_apply", "calib_rollback")
TELEMETRY_SCHEMA = "ni_rua.so101.telemetry.v1"

protocol_output = sys.stdout
protocol_lock = threading.Lock()


def emit(message):
    with protocol_lock:
        print(WIRE_PREFIX + json.dumps(message, allow_nan=False), file=protocol_output, flush=True)


# ---------------------------------------------------------------- geometry ---
# Pose geometry, the arrival rule and both IK solvers live in ik_plan (shared with the reach query); the names stay
# importable from this module.
from ni_rua.robot.backends.so101.ik_plan import (  # noqa: E402
    _finite, arrival_verdict, collision_world, curobo_solve, interpolation_steps, joint_interpolation, local_solve,
    matrix_to_quaternion, path_check_segments, path_facts, pose_matrix, pose_residual, quaternion_to_matrix,
    split_tolerances)
from ni_rua.robot.backends.so101.ik_service import CuroboClient, IKServiceError  # noqa: E402

#: Keys of a measured TCP residual (``pose_residual``) mirrored to receipt level as ``endpoint_residual_m`` (m / rad).
RESIDUAL_FIELDS = ("dx_m", "dy_m", "dz_m", "position_error_m", "orientation_error_rad", "approach_error_rad",
                   "roll_error_rad")
#: ``_effective_tolerances`` keys a reach query leaves out of ``stage_results[<id>].tolerances``: it never moves, so
#: the measured-arrival basis, the vertical window, the stall guard and the arrival-correction bounds mean nothing there.
REACH_EXCLUDED_TOLERANCES = ("checked_on", "max_upward_dz_m", "max_tracking_divergence_m", "stall_guard_preset",
                             "arrival_correction_rounds", "arrival_correction_max_offset_rad")
#: Samples of one move stage stored for the preview's trajectory overlay and inspection (evenly spaced commands of the
#: move, its last one included). Bounds the candidate-trajectory artifact; the path facts use every command.
PATH_OVERLAY_SAMPLES = 8


class DriverFault(RuntimeError):
    """A bus/driver exception; physical state after it is unknown."""


# ----------------------------------------------------------------- adapter ---

class SO101Adapter:
    def __init__(self, deployment, run_dir, observation_profile, *, driver=None, cameras=None, clock=None,
                 kinematics=None, image_settings=None, start_telemetry=True, deployment_path=None):
        from ni_rua.core.observation_policy import ACTOR_PROFILES, allows_depth, has_calibration
        if observation_profile not in ACTOR_PROFILES or allows_depth(observation_profile):
            raise ValueError(f"so101_unsupported_observation_profile:{observation_profile}")
        self.deployment = deployment
        # The only arm id of this embodiment (EmbodimentProfile.arms, from the packaged SO-101 facts).
        self.arm = contract.embodiment(deployment).arms[0]
        # Deployment file the calibration workflow writes joint_mapping offsets into (None: offsets read-only).
        self.deployment_path = None if deployment_path is None else str(deployment_path)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.profile = observation_profile
        self.calibrated = has_calibration(observation_profile)
        self.clock = clock or Clock()
        if kinematics is None:
            from ni_rua.robot.model import so101_kinematics as kinematics
        self.kin = kinematics
        self.model = kinematics.load_model()
        margin = deployment["motion"]["joint_limit_margin_rad"]
        urdf = kinematics.joint_limits_rad(self.model)
        self.limits = {name: (urdf[name][0] + margin, urdf[name][1] - margin) for name in ARM_JOINTS}
        # IK solver: the cuRobo service when the deployment configures ik_service (ik_plan.curobo_solve), otherwise
        # the packaged local solver (ik_plan.local_solve). Never both, never a silent fallback.
        self.ik_client = CuroboClient(deployment["ik_service"], self.limits) if deployment.get("ik_service") else None
        self._fixed_jaw_protrusion_m = None  # CAD protrusion of the fixed jaw beyond the TCP (m), read on first use
        home = deployment["home"]["q_rad"]
        if any(not lo <= q <= hi for q, (lo, hi) in zip(home, self.limits.values())):
            raise ValueError("so101_home_outside_joint_limits_with_margin")
        if image_settings is None:
            from ni_rua.core.storage.images import image_settings as load_settings
            image_settings = load_settings(json.loads(os.environ["NI_RUA_IMAGE_SETTINGS"])
                                           if os.environ.get("NI_RUA_IMAGE_SETTINGS") else None)
        self.image_settings = image_settings
        # Single source of the active camera calibration: the deployment camera_calibration_file
        # path. It is (re)read at start, on operator reload_calibration and after calibrate.
        self.calibration_path = deployment.get("camera_calibration_file")
        self.camera_calibration = None   # published in observations (calibrated profiles only)
        self.calibration_identity = None  # operator/audit summary of the loaded file
        self.calibration_error = None
        if self.calibrated:
            if not self.calibration_path or not Path(self.calibration_path).is_file():
                raise ValueError(
                    f"so101_camera_calibration_missing: camera_calibration_file ({self.calibration_path or 'unset'}) "
                    "not found for calibrated observation profile; for cold-start calibration, launch with "
                    "`--observation-profile rgb_proprio_v1` and open `/calibration` in the workbench UI"
                )
            self.camera_calibration, self.calibration_identity = self._load_calibration(self.calibration_path)
        elif self.calibration_path and Path(self.calibration_path).is_file():
            # Uncalibrated profile: the file is never published, but its identity is shown to the operator.
            try:
                _, self.calibration_identity = self._load_calibration(self.calibration_path)
            except (OSError, ValueError) as error:
                self.calibration_error = f"{type(error).__name__}: {error}"
        self.bus_lock = threading.RLock()
        self.state_lock = threading.Lock()
        self.operator_lock = threading.Lock()
        self.stop_generation = 0
        self.stop_reason = None
        self.relaxed = False
        self.observe_required = False
        self.speed_scale_cap = float(deployment["operator"]["default_speed_scale_cap"])
        self.control_step = 0
        self.motion_phase = "idle"
        self.active_action_id = None
        self.last_tracking_error = None
        self.last_sample = None
        self.faults = []
        # Learned settled shortfall (goal - measured) per arm joint, rad; re-applied by hold so a
        # low-gain servo does not sag further each time the goal is pinned to a measured pose.
        self.hold_offset = [0.0] * 5
        # True when hold_offset was learned at a goal that contained the gravity feedforward (a free-space arrival):
        # a hold then adds the feedforward of the pose it holds. False after a contact hold, torque_on or relax,
        # whose goal is the measured pose itself.
        self.hold_feedforward = False
        # Deployment motion_compensation (deployment.MOTION_COMPENSATION_FIELDS); {} = every mechanism off.
        self.compensation = deployment.get("motion_compensation") or {}
        # carry_arrival_offset: the arrival-correction offset (goal - feedforward - target, rad per arm joint) the
        # last settled free-space arrival ended with, and the arm goal that arrival (or a confirmed pause, _stop)
        # left. The offset is valid for the next move only while arm_goal still equals path_offset_goal: any other
        # goal write (contact hold, stop signal, torque_on, touch-off, calibration) ends it (_carry).
        self.path_offset = np.zeros(5)
        self.path_offset_goal = None
        feedforward = self.compensation.get("gravity_feedforward")
        self._gravity = None  # (inertials, stiffness per ARM_JOINTS or None, max_offset_rad) with gravity_feedforward
        if feedforward:
            from ni_rua.robot.embodiments.so101.gravity import load_inertials
            self._gravity = (load_inertials(), [feedforward["joint_stiffness_nm_per_rad"].get(name) for name in ARM_JOINTS],
                             feedforward["max_offset_rad"])
        breakaway = self.compensation.get("arrival_breakaway_rad")
        # Per-joint smallest responsive correction step (rad); 0 = none identified for that joint.
        self._breakaway = None if not breakaway else np.array([breakaway.get(name, 0.0) for name in ARM_JOINTS])
        self.torque_known = None
        self.handheld = None  # operator hand-held fixed-camera capture (calibration_capture.HandheldCapture)
        self._workflow = None  # calibration_runner.CalibrationWorkflow, created on first use
        self._plans = {}
        if driver is None:
            from ni_rua.robot.backends.so101.driver import LeRobotSO101Driver
            driver = LeRobotSO101Driver(deployment)
        self.driver = driver
        if cameras is None:
            from ni_rua.robot.backends.so101.driver import OpenCVCameraSource
            cameras = {name: OpenCVCameraSource(name, config, self.clock)
                       for name, config in deployment["cameras"].items()}
        self.cameras = cameras
        # Bind every camera to its physical device before the bus is touched: a camera whose unique_id/device_name
        # names no present device, or several, refuses startup here (no fallback to index_or_path). Operator/audit
        # record only (camera_binding.json, robot-worker.log, S0); never published to the actor.
        from .camera_devices import CameraBindingError, bind_cameras, log_lines
        self.camera_binding = bind_cameras(cameras)
        if self.camera_binding is not None:
            (self.run_dir / "camera_binding.json").write_text(
                json.dumps(dict(self.camera_binding, checked_at=self.clock.utc()), indent=2, ensure_ascii=False),
                encoding="utf-8")
            for line in log_lines(self.camera_binding):
                print(line, file=sys.stderr, flush=True)
            if self.camera_binding["refused"]:
                raise CameraBindingError("; ".join(self.camera_binding["refused"]))
        from .motion_video import MotionVideoRecorder
        self.motion_video = MotionVideoRecorder(self)
        self._video_action_id = None
        started_cameras = []
        driver_connected = False
        try:
            with self.bus_lock:
                try:
                    self.driver.connect()
                    driver_connected = True
                finally:
                    # Operator/audit record only (run_dir); never published to the actor.
                    report = getattr(self.driver, "preflight_report", None) or getattr(
                        getattr(sys.exc_info()[1], "report", None), "copy", lambda: None)()
                    if report:
                        (self.run_dir / "preflight.json").write_text(json.dumps(report, indent=2))
                first = self._read()
                # Pin the goal to the present pose so torque never chases a stale register.
                self.arm_goal = list(first["q_rad"])
                self.jaw_goal = float(np.clip(first["gripper_normalized"], 0, 1))
                self._pin_jaw_state()
                self.driver.send_goal(self.arm_goal, self.jaw_goal)
                self.control_step += 1
            for camera in cameras.values():
                camera.start()
                started_cameras.append(camera)
            self.telemetry_dir = self.run_dir / "telemetry"
            self.telemetry_dir.mkdir(parents=True, exist_ok=True)
            self.telemetry_seq = 0
            self._telemetry_stop = threading.Event()
            self._last_motor_read = None
            self._last_motor_status = {}
            self._last_frame_write = None
            self.telemetry_thread = None
            if start_telemetry:
                self.telemetry_thread = threading.Thread(target=self._telemetry_loop, daemon=True, name="so101-telemetry")
                self.telemetry_thread.start()
        except BaseException:
            if hasattr(self, "_telemetry_stop") and self._telemetry_stop is not None:
                self._telemetry_stop.set()
            for camera in reversed(started_cameras):
                try:
                    camera.close()
                except Exception:
                    pass
            if driver_connected:
                with self.bus_lock:
                    try:
                        self.driver.disconnect()
                    except Exception:
                        pass
            raise

    # ------------------------------------------------------------ basics ---
    @property
    def embodiment(self):
        """The ``EmbodimentProfile`` of this deployment (``contract.embodiment``): the same profile the parent backend
        builds from the same file. Every published robot fact and bound below is read from it."""
        return contract.embodiment(self.deployment)

    @property
    def arm_layout(self):
        """``ArmLayout`` of this SO-101 deployment (1-arm 5-DoF + 1 jaw column)."""
        override = getattr(self, "_arm_layout_override", None)
        if isinstance(override, ArmLayout):
            return override
        return ArmLayout.from_profile(self.embodiment)

    @arm_layout.setter
    def arm_layout(self, value):
        self._arm_layout_override = value

    def state(self):
        """Current measured ``robot_state`` (``MotionControllerPort``)."""
        return self.robot_state(self.read())

    def read_state(self):
        """Read current measured ``robot_state`` (``MotionControllerPort``)."""
        return self.state()

    def hold(self, run=None, *, sag_offset=True):
        """Hold measured arm position and confirm by settling (``MotionControllerPort``)."""
        return self._confirm_stop(run, sag_offset=sag_offset)

    def dispatch_row(self, row, *, command_phase="batch", run=None, **_kwargs):
        """Dispatch one flat command row ``[q0..q4, jaw_normalized]`` (``MotionControllerPort``)."""
        slot = self.arm_layout.slot(self.arm)
        q_vals = [float(v) for v in row[slot.joints]]
        jaw_val = self.jaw_goal if slot.jaw is None else float(row[slot.jaw])
        gen = run["generation"] if isinstance(run, dict) and "generation" in run else self.stop_generation
        return bool(self._send(q_vals, jaw_val, gen, run=run, kind=command_phase))

    def step_budget_remaining(self):
        """Real-time hardware has no simulated step budget limit (``MotionControllerPort``)."""
        return None

    def settle_and_correct(self, run, q_target, pose, tolerances, **kwargs):
        """Settle and run bounded hardware ``ArrivalCorrection`` (``MotionControllerPort``)."""
        return self._arrive(run, q_target, pose, tolerances, **kwargs)

    def _load_calibration(self, path):
        """Read, validate and convert one calibration file. Raises ValueError/OSError; no state change.

        Checks: schema (``read_calibration_file``), one entry per configured camera, and each
        calibrated ``image_size`` equal to the configured capture ``[width, height]`` (px).
        Returns ``(published, identity)``: the observation ``camera_calibration`` document and the
        operator/audit identity (``created_at``, sample count and residuals; no hashes).
        """
        from ni_rua.robot.calibration.workflow.charuco import read_calibration_file
        document = read_calibration_file(path)

        def plain(value):
            return value.tolist() if isinstance(value, np.ndarray) else value
        cameras = {}
        for name, camera in document["cameras"].items():
            if name not in self.deployment["cameras"]:
                continue
            item = {key: plain(camera[key]) for key in ("K", "distortion", "distortion_model", "image_size", "optical_axes")}
            if name == "wrist":
                item.update(arm=self.arm, link_name=self.kin.WRIST_CAMERA_PARENT_LINK, T_link_camera=plain(camera["T_link_camera"]))
            else:
                item["T_base_camera"] = plain(camera["T_base_camera"])
            cameras[name] = item
        missing = set(self.deployment["cameras"]) - set(cameras)
        if missing:
            raise ValueError("so101_camera_calibration_missing_cameras:" + ",".join(sorted(missing)))
        for name, item in cameras.items():
            configured = self.deployment["cameras"][name]
            if item["image_size"] != [configured["width"], configured["height"]]:
                raise ValueError(f"so101_camera_calibration_mismatch:{name}: calibrated image_size {item['image_size']} "
                                 f"!= configured [{configured['width']}, {configured['height']}] px")
        wrist, fixed = document["cameras"].get("wrist", {}), document["cameras"].get("fixed", {})

        def number(value):
            return None if value is None else round(float(value), 6)
        # Measurement precision of this calibration (control-relevant uncertainty, not provenance).
        residuals = {key: value for key, value in {
            "wrist_intrinsics_rms_px": number(wrist.get("intrinsics_rms_px")),
            "wrist_hand_eye_translation_rms_m": number((wrist.get("hand_eye_residual") or {}).get("translation_m")),
            "wrist_hand_eye_rotation_rms_rad": number((wrist.get("hand_eye_residual") or {}).get("rotation_rad")),
            "fixed_intrinsics_rms_px": number(fixed.get("intrinsics_rms_px")),
            "fixed_chain_reprojection_rms_px": number((fixed.get("extrinsic_residual") or {}).get("reprojection_px")),
        }.items() if value is not None and (key.startswith("wrist") and "wrist" in cameras or key.startswith("fixed") and "fixed" in cameras)}
        published = {"profile_id": "so101-charuco-v1", "source_kind": "so101_charuco_hand_eye_calibration",
                     "reference_frame_id": SO101.frame_id, "calibrated_at": document.get("created_at"),
                     "calibration_residuals": residuals, "cameras": cameras}
        if "joint_calibration" in document:
            published["joint_calibration"] = deepcopy(document["joint_calibration"])
        identity = {"created_at": document.get("created_at"), "samples": document.get("samples"),
                    "residuals": residuals, "file": str(path)}
        return published, identity

    def validate_calibration_file(self, path):
        """Raise if ``path`` would not load (same checks as reload). Returns its identity."""
        return self._load_calibration(path)[1]

    def reload_calibration(self):
        """Re-read ``calibration_path``. On failure keep the previous calibration and report why."""
        previous = self.calibration_identity
        if not self.calibration_path:
            return {"command": "reload_calibration", "status": "failed", "reason": "camera_calibration_file_not_configured",
                    "calibration": previous, "kept_previous": True}
        try:
            published, identity = self._load_calibration(self.calibration_path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            return {"command": "reload_calibration", "status": "failed", "reason": f"{type(error).__name__}: {error}",
                    "calibration": previous, "kept_previous": True}
        # One reference swap: an observation in progress uses either the old or the new document whole.
        if self.calibrated:
            self.camera_calibration = published
        self.calibration_identity, self.calibration_error = identity, None
        return {"command": "reload_calibration", "status": "reloaded", "calibration": identity, "previous": previous,
                "published_in_observations": self.calibrated,
                "applies_to": "observations captured after this command"}

    def capture_frames(self, after_t):
        """RGB frames with legacy host stamps after ``after_t`` (adapter-monotonic s).

        ``last_capture_timing`` retains each returned frame's stamp, host wait/read
        interval and unknown exposure time for operator calibration archives.
        The return value remains the camera-to-RGB mapping for existing callers.
        """
        timeout = self.deployment["camera"]["frame_timeout_s"]
        frames = {}
        self.last_capture_timing = {}
        for name, camera in sorted(self.cameras.items()):
            started = self.clock.now()
            image, stamp = camera.read_after(after_t, timeout)
            finished = self.clock.now()
            interval = [float(stamp), None]
            timed_reader = getattr(camera, "latest_timed", None)
            if callable(timed_reader):
                try:
                    timed = timed_reader()
                    if timed is not None and timed.get("capture_monotonic_s") == stamp:
                        interval = list(timed["host_read_interval_s"])
                except Exception:
                    pass  # Timing annotation failure cannot erase an acquired frame.
            frames[name] = np.asarray(image, dtype=np.uint8)
            self.last_capture_timing[name] = dict(capture_monotonic_s=float(stamp), host_read_interval_s=interval,
                capture_wait_interval_s=[started, finished], exposure_monotonic_s=None)
        return frames

    def board_check(self, *, fresh=False, board_spec=None):
        """Count ChArUco corners per camera. Read-only: no bus access, no motion.

        ``fresh=False`` uses each camera's latest preview frame (cheap, for live console feedback);
        ``fresh=True`` waits for frames received after this call (pre-motion check of calibrate).
        ``ok`` is True when every configured camera sees at least ``calibration.min_board_corners``.
        ``board_spec``: board to detect (default the packaged board).
        """
        from ni_rua.robot.calibration.workflow.capture import count_corners
        from ni_rua.robot.calibration.workflow import charuco as cal
        default_spec = self.deployment.get("board") or (self.workflow.default_board() if getattr(self, "workflow", None) else None)
        spec_in = board_spec or default_spec or cal.DEFAULT_BOARD_SPEC
        board = cal.make_board({k: v for k, v in spec_in.items() if k != "thickness_m"})
        minimum = self.deployment["calibration"]["min_board_corners"]
        corners, errors = {}, {}
        for name, camera in sorted(self.cameras.items()):
            try:
                frame = (camera.read_after(self.clock.now(), self.deployment["camera"]["frame_timeout_s"]) if fresh
                         else camera.latest())
                if frame is None:
                    corners[name], errors[name] = 0, "no_frame_yet"
                    continue
                corners[name] = count_corners(frame[0], board)
            except Exception as error:
                corners[name], errors[name] = 0, f"{type(error).__name__}: {error}"
        below = sorted(name for name, count in corners.items() if count < minimum)
        result = {"command": "board_check", "status": "checked", "ok": not below, "corners": corners,
                  "min_board_corners": minimum, "frames": "fresh" if fresh else "latest_preview",
                  "calibration": self.calibration_identity}
        if errors:
            result["errors"] = errors
        if below:
            result["reason"] = "board_not_visible:" + ",".join(below)
        return result

    def _read(self):
        """One measured sample. Caller holds the bus lock. Raises DriverFault."""
        try:
            raw = self.driver.read()
        except Exception as error:
            raise DriverFault(f"driver_read_failed: {type(error).__name__}: {error}") from error
        q, jaw = [float(v) for v in raw["q_rad"]], float(raw["gripper_rad"])
        if len(q) != 5 or not _finite(q + [jaw]):
            raise DriverFault("driver_read_nonfinite_or_wrong_dimension")
        sample = {"t": self.clock.now(), "q_rad": q, "gripper_rad": jaw,
                  "gripper_normalized": gripper_normalized_from_rad(self.deployment, jaw)}
        self.last_sample = sample
        return sample

    def read(self):
        with self.bus_lock:
            return self._read()

    def _pin_jaw_state(self):
        """Jaw bookkeeping after the jaw goal was pinned to the measured jaw (start, torque_on): the commanded goal
        ``jaw_command`` (normalized) is that goal, the last jaw command did not close (``jaw_closing`` False) and no
        holding squeeze is in effect (``jaw_hold`` None)."""
        self.jaw_command, self.jaw_closing, self.jaw_hold = self.jaw_goal, False, None

    def _send(self, q_rad, jaw_normalized, generation=None, run=None, kind=None, commanded=None, hold_squeeze=None,
              jaw_command=False):
        """Write one goal unless a newer operator stop arrived. Returns False if suppressed.

        Jaw bookkeeping: ``jaw_goal`` is the goal the jaw servo holds (normalized). A write that changes it, or starts
        an explicit jaw command (``jaw_command=True``, even when its target equals ``jaw_goal``), is a jaw command:
        ``self.jaw_command`` becomes that value, ``jaw_closing`` says whether it moved the goal toward closed and any
        holding squeeze ends (``jaw_hold`` None). ``hold_squeeze`` (``_hold_squeeze`` only): this write sets the
        squeezed hold goal; it is not a jaw command (``jaw_command`` / ``jaw_closing`` keep the command it follows)
        and the given dict becomes ``jaw_hold``. Writes that repeat ``jaw_goal`` (moves, holds, waits, or trailing
        steps of a during-motion ramp) change nothing.

        ``commanded``: the planned joint row this goal stands for when ``q_rad`` carries motion-compensation
        offsets on top of it (rad; recorded as ``q_commanded_rad``, the row ``planned_trace`` shows); None = the
        goal is the planned row.

        ``run``: the motion run whose ``dispatched`` list records this command (step, joint goals
        in rad, jaw goal normalized, optional ``kind``); None for operator-only writes.
        """
        with self.bus_lock:
            if generation is not None and generation != self.stop_generation:
                return False
            try:
                if self._video_action_id is not None:
                    self.motion_video.begin(self._video_action_id)
                self.driver.send_goal(list(q_rad), float(jaw_normalized))
            except Exception as error:
                raise DriverFault(f"driver_write_failed: {type(error).__name__}: {error}") from error
            jaw = float(jaw_normalized)
            if hold_squeeze is not None:
                self.jaw_hold = hold_squeeze
            elif jaw != self.jaw_goal:
                self.jaw_command, self.jaw_closing, self.jaw_hold = jaw, jaw < self.jaw_goal, None
            elif jaw_command:
                self.jaw_command, self.jaw_closing, self.jaw_hold = jaw, False, None
            self.arm_goal, self.jaw_goal = [float(v) for v in q_rad], jaw
            self.control_step += 1
            if run is not None:
                run["dispatched"].append({"step": self.control_step, "q_goal_rad": list(self.arm_goal),
                                          "gripper_goal_normalized": self.jaw_goal, **({"kind": kind} if kind else {}),
                                          **({} if commanded is None else
                                             {"q_commanded_rad": [float(v) for v in commanded]})})
            return True

    def _gripper_geometry(self):
        from ni_rua.robot.model.so101_gripper_geometry import gripper_geometry_payload
        g = self.deployment["gripper"]
        return gripper_geometry_payload(g["jaw_closed_rad"], g["jaw_open_rad"])

    def aperture_m(self, gripper_rad):
        """Measured CAD fingertip gap (m) at ``gripper_rad`` (rad)."""
        from ni_rua.robot.model.so101_gripper_geometry import opening_width_m
        return float(opening_width_m(gripper_rad))

    @staticmethod
    def _pose_dict(T):
        return {"position_m": [float(v) for v in T[:3, 3]], "quaternion_xyzw": [float(v) for v in matrix_to_quaternion(T[:3, :3])]}

    def tcp(self, q_rad):
        return np.asarray(self.kin.fk(self.model, list(q_rad)))

    def _cap(self):
        with self.state_lock:
            return self.speed_scale_cap

    def effective_limits(self, speed_scale=1.0, gripper_speed_scale=1.0):
        limits = public_limits(self.deployment, self._cap())
        limits.update(request_speed_scale=speed_scale, request_gripper_speed_scale=gripper_speed_scale,
                      commanded_max_joint_speed_rad_s=limits["effective_max_joint_speed_rad_s"] * speed_scale,
                      commanded_gripper_speed_rad_s=self.deployment["gripper"]["max_speed_rad_s"] * gripper_speed_scale * self._cap())
        return limits

    # ------------------------------------------------------ capabilities ---
    def capabilities(self):
        """Published capabilities: every robot fact and bound is the run profile's (``self.embodiment``)."""
        d, motion, profile, arm = self.deployment, self.deployment["motion"], self.embodiment, self.arm
        rate = profile.control_hz
        scale = lambda value: {"min": value.min, "max": value.max, "default": value.default}  # noqa: E731
        return {
            "backend": "so101_lerobot", "data_origin": "so101_real_hardware", "world_kind": "observed",
            "robot_model_id": profile.robot_model_id, "frame_id": profile.frame_id, "tcp_config_id": profile.tcp_config_id,
            "arms": list(profile.arms), "dual_arm": "dual_arm" in profile.supports,
            "max_simultaneous_arms": profile.max_simultaneous_arms, "standalone_gripper": True,
            "observation_profile": self.profile, "observation_profiles": [self.profile],
            "cameras": sorted(self.cameras),
            "modes": ["absolute"], "path_kinds": ["backend_free", "hold"], "max_waypoints": motion["max_waypoints"],
            "max_timeout_s": profile.max_timeout_s, "default_timeout_s": profile.default_action_timeout_s,
            "limits_profile": profile.limits_profiles[0], "motion_execution_mode": "resampled",
            "gripper_after_motion": True, "orientation_modes": ["backend_free"],
            "gripper_phases": ["after_motion", *(["during_motion"] if "gripper_motion" in profile.supports else [])],
            "arm_joint_names": {arm: list(profile.arm_joint_names[arm])},
            "arm_joint_limits_rad": {arm: {name: list(bounds) for name, bounds in self.limits.items()}},
            "joint_limit_source": "URDF nominal limits shrunk by deployment joint_limit_margin_rad; not verified hardware stops",
            "motion_tolerance_defaults": {"position_tolerance_m": profile.default_tolerance.position_m,
                                          "orientation_tolerance_rad": profile.default_tolerance.orientation_rad},
            "runtime_limits": public_limits(d, self._cap()),
            "native_control_hz": rate,
            "bounded_wait": {"max_native_steps": contract.MAX_WAIT_STEPS, "native_control_hz": rate,
                             "default_timeout_s": profile.default_wait_timeout_s,
                             "semantics": f"real-time hold: each step is one {1 / rate:.3f} s command period holding the "
                                          "last commanded joint goals; the world keeps moving; reports measured wall-clock duration"},
            "action_speed": {"available": True, "speed_scale": scale(profile.speed_scale),
                             "gripper_speed_scale": scale(profile.gripper_speed_scale),
                             "waypoint_speed_scales": {"available": False},
                             "semantics": "scales runtime_limits.effective_max_joint_speed_rad_s (joint-space interpolation); "
                                          "1 is the deployment maximum times the operator speed_scale_cap; never faster"},
            "tool_geometry": {"tcp_link": profile.tcp_link, "reference_link": profile.tcp_link,
                              "conversion": self._pose_input_text(),
                              "gripper_geometry": self._gripper_geometry()},
            "kinematic_reachability": self._reachability_text(),
            "gripper_semantics": {"goal_normalized": "0=closed, 1=open; linear onto the deployment jaw range; latched",
                                  "measured": "q_rad[5] is the measured jaw joint angle (rad); not a grasp-success signal",
                                  "contact_feedback": "unavailable",
                                  **({"during_motion": "linear ramp from the latched jaw goal over start_fraction..end_fraction "
                                                       "of the move's commands; the move is subdivided until no jaw step "
                                                       "exceeds gripper.max_speed_rad_s x gripper_speed_scale x operator cap; "
                                                       "the jaw is measured after arrival and settling; command completion "
                                                       "does not prove a grasp"}
                                     if "gripper_motion" in profile.supports else {}),
                                  **contract.hold_squeeze_capability(d)},
            "clock_policy": "real time: the arm and world keep evolving while the model reasons; nothing is paused",
            "step_semantics": "sim_step counts dispatched joint-goal commands (step_domain so101_control_step); it is not time",
            "observation_timing": "joints are read first, then each camera returns its first frame received after that read; "
                                  "per-frame offsets are reported; exposure latency is unknown",
            "cancellation": "checked before every command; stop holds the measured position and is confirmed by measured settling",
            "collision_scope": self._collision_scope_text(),
            "contact_sensing": "unavailable: no force or contact sensor",
            "motion_preview": {"available": False},
            "unified_action": {"available": True, "requires_camera_calibration": False,
                               "modes": ["preview", "execute", "inspect", "direct",
                                         *(["execute_if_clean"] if "execute_if_clean" in profile.supports else []),
                                         *(["reach"] if "reach" in profile.supports else [])],
                               **({"reach": "solves IK for each move stage independently from the measured joints at the "
                                            "call and answers every stage; no plan, no motion"}
                                  if "reach" in profile.supports else {}),
                               **({"execute_if_clean": self._execute_if_clean_text()}
                                  if "execute_if_clean" in profile.supports else {}),
                               "max_stages": profile.max_stages, "max_stages_per_arm": profile.stages_per_arm,
                               "ik_solver": getattr(self.ik_client, "solver", "curobo") if self.ik_client is not None else "local",
                               "planning": "IK per move stage, seeded from the previous stage's planned joints; residuals checked "
                                           "against each stage tolerance at preview; stage_plans[<move>].joint_travel_rad is "
                                           "|target - start| per arm joint (straight joint interpolation, no travel limit)",
                               "checkpoint_resume": "observe stages, checkpoint_after_gripper and grip-guard violations pause with a "
                                                    "confirmed hold; resume re-reads measured joints and requires drift <= plan "
                                                    "max_start_drift_rad; cached IK targets are never re-solved",
                               "execution": "joint-space interpolation from the measured start to each cached IK target"},
            "grip_guard": contract.grip_guard_capability(d),
            "reset": True, "reset_strategy": "arm_home_only_world_unchanged",
            "native_evaluator": None, "observer_video": True,
            "action_budget": {"native_step_limit": None, "source": "no native task budget on real hardware"},
        }

    def _pose_input_text(self):
        """``tool_geometry.conversion``: the move pose inputs of the profile (``pose_inputs``) and how each reaches the
        TCP target; the grasp-centre input only with ``jaw_grasp_centre``."""
        profile = self.embodiment
        text = (f"robot_action move stages take position_m + quaternion_xyzw of {profile.tcp_label} in "
                f"{profile.frame_id}: the TCP on the fixed jaw's inner fingertip face (gripper_geometry x = 0), with the "
                "whole jaw gap on TCP -X; no offset is applied to them")
        if "jaw_grasp_centre" not in profile.pose_inputs:
            return text + "; this is the only pose input"
        return (text + ". A move may instead give grasp_centre_m + object_width_m (where the object's centre sits "
                "between the jaws, m) with quaternion_xyzw or fixed_jaw_yaw_deg: the Harness converts it with the "
                "gripper CAD to the TCP target (robot_action field help of grasp_centre_m) and reports the conversion "
                "in stage_plans[<stage>].grasp_centre")

    def _execute_if_clean_text(self):
        """``unified_action.execute_if_clean``: what gates the mode on this profile, generated from the same review list
        as the robot_action text (``embodiments.preview_reviews``) so both name each review and its scope."""
        profile = self.embodiment
        reviews = preview_reviews(profile)
        closed_jaw = (["closed_jaw_before_grasp:<arm>"] if profile.gripper.empty_close_aperture_m is not None else [])
        if not reviews:
            return ("executes when the preview is planned and no downgrade reason applies; this backend produces no "
                    "collision, motion-safety or grasp review, so IK planning success is the main condition "
                    "(downgrade_reasons names any other" + (": " + closed_jaw[0] if closed_jaw else "") + ")")
        named = [f"{PREVIEW_REVIEW_CLEAN[name]} ({name}: {scope})" if scope else PREVIEW_REVIEW_CLEAN[name]
                 for name, scope in reviews]
        codes = [code for name, _ in reviews for code in PREVIEW_REVIEW_DOWNGRADES[name]] + closed_jaw
        return ("executes when the preview is planned, " + ", ".join(named) + " and no other downgrade reason applies; "
                "otherwise it returns the preview plan_id and downgrade_reasons names what kept it ("
                + ", ".join(codes) + ")")

    # -------------------------------------------------------- observation ---
    def robot_state(self, sample, step=None):
        """Measured robot_state of one joint sample; ``step``: its control step (default: the current one)."""
        T = self.tcp(sample["q_rad"])
        return {"arms": {self.arm: {
                    "joint_names": list(JOINT_NAMES), "q_rad": [*sample["q_rad"], sample["gripper_rad"]],
                    "source_kind": "measurement",
                    "ee_pose": {"position_m": T[:3, 3].tolist(), "quaternion_xyzw": matrix_to_quaternion(T[:3, :3]),
                                "frame_id": SO101.frame_id, "tcp_config_id": SO101.tcp_config_id, "source_kind": "measurement"},
                    # The commanded jaw goal (jaw_command), also under a holding squeeze: readers of the state take
                    # it as what the jaw was told to do (closed below 0.5). The servo's hold goal is in the receipts.
                    "gripper": {"goal_normalized": self.jaw_command, "goal_source_kind": "command",
                                "command_normalized": self.jaw_command, "command_source_kind": "command",
                                "contact_feedback": None, "contact_feedback_reason": "no contact sensor"}}},
                "frame_id": SO101.frame_id, "tcp_config_id": SO101.tcp_config_id,
                "sim_step": self.control_step if step is None else step, "step_domain": STEP_DOMAIN}

    def observe(self):
        from PIL import Image
        from ni_rua.core.observation_policy import actor_robot_state
        from ni_rua.core.storage.images import encode_image
        with self.bus_lock:
            sample = self._read()
            step = self.control_step
        observation_id = "raw-" + uuid.uuid4().hex
        directory = self.run_dir / "observations" / observation_id
        directory.mkdir(parents=True)
        cameras, poses = [], {}
        calibration_document = self.camera_calibration  # one consistent document even if reloaded meanwhile
        for name in sorted(self.cameras):
            image, stamp = self.cameras[name].read_after(sample["t"], self.deployment["camera"]["frame_timeout_s"])
            image = np.asarray(image, dtype=np.uint8)
            path = directory / f"{name}.jpg"
            data, encoding = encode_image(Image.fromarray(image), self.image_settings, storage=True)
            path.write_bytes(data)
            offset = stamp - sample["t"]
            frame_id = f"{observation_id}-{name}"
            cameras.append({"camera_id": name, "frame_id": frame_id, "rgb_path": str(path), "image_encoding": encoding,
                            "width": int(image.shape[1]), "height": int(image.shape[0]), "pixel_transform": "identity",
                            "capture_alignment": f"asynchronous real-time capture: frame received {offset:+.3f} s after the "
                                                 "joint read; exposure latency unknown; the scene may move between them",
                            "delivered_to_actor": False})
            if calibration_document is not None:
                static = calibration_document["cameras"][name]
                if name == "wrist":
                    T_link = np.asarray(self.kin.fk(self.model, sample["q_rad"], link=self.kin.WRIST_CAMERA_PARENT_LINK))
                    poses[name] = {"T_world_camera": (T_link @ np.asarray(static["T_link_camera"])).tolist(),
                                   "T_world_link": T_link.tolist(), "T_world_base": np.eye(4).tolist(),
                                   "arm": self.arm, "link_name": static["link_name"], "frame_id": frame_id,
                                   "source_kind": "robot_kinematics_at_joint_read",
                                   "joint_to_frame_offset_s": round(offset, 4)}
                else:
                    poses[name] = {"T_world_camera": static["T_base_camera"], "frame_id": frame_id,
                                   "source_kind": "static_extrinsic_calibration"}
        state = self.robot_state(sample)
        observation = {"raw_observation_id": observation_id, "data_origin": "so101_real_hardware", "world_kind": "observed",
                       "observation_profile": self.profile, "robot_model_id": SO101.robot_model_id, "cameras": cameras,
                       "robot_state": state if self.calibrated else actor_robot_state(state, self.profile),
                       "sim_step": step, "step_domain": STEP_DOMAIN, "sim_time_s": None,
                       "control_dt_s": 1 / self.deployment["motion"]["command_rate_hz"],
                       "captured_at": self.clock.utc(sample["t"])}
        if calibration_document is not None:
            calibration = {key: value for key, value in calibration_document.items()}
            calibration["cameras"] = {
                name: (
                    {key: value for key, value in item.items() if key != "T_base_camera"}
                    | ({"T_world_camera": item["T_base_camera"]} if "T_base_camera" in item and "T_link_camera" not in item else {})
                    | ({"arm": item.get("arm", self.arm), "link_name": item.get("link_name", self.kin.WRIST_CAMERA_PARENT_LINK)} if "T_link_camera" in item else {})
                )
                for name, item in calibration_document["cameras"].items()
            }
            observation.update(camera_calibration=calibration, robot_model=self.model,
                               camera_pose_sample={"reference_frame_id": SO101.frame_id, "sim_step": step,
                                                   "step_domain": STEP_DOMAIN, "cameras": poses})
        with self.state_lock:
            self.observe_required = False
        return observation

    # ------------------------------------------------------ motion helpers ---
    def _reachability_text(self):
        """Model-facing summary of how this arm's IK plans a 6D TCP target (robot_capabilities.kinematic_reachability)."""
        plane = ("5-DOF arm: a general 6D pose is usually unreachable; the approach axis (TCP +Z) tilts only in the "
                 "arm's vertical plane through shoulder_pan and the roll about it is set by wrist_roll (a straight-down "
                 "approach reaches every yaw outside the wrist_roll band named in the motion instructions). ")
        if self.ik_client is None:
            return (plane + "IK minimises position plus weighted orientation error; set orientation_tolerance_rad=3.1416 "
                    "to solve position only. IK residuals above the stage tolerance fail planning before any motion.")
        return (plane + "IK is NVIDIA cuRobo multi-seed GPU IK; every "
                "solution is re-checked with this arm's FK and the valid one with the least joint travel is used. "
                "approach_tolerance_rad / roll_tolerance_rad bound the approach tilt and the roll separately; "
                "orientation_tolerance_rad=3.1416 solves position only; robot_action mode=reach checks many candidate "
                "poses in one call. IK residuals above the stage tolerance fail planning before any motion.")

    def _collision_scope_text(self):
        """Model-facing collision coverage of planning (robot_capabilities.collision_scope)."""
        if self.ik_client is None:
            return "none: joint limits and the deployment workspace box only; no object, table or self-collision checking"
        table = self.deployment["ik_service"].get("table_top_z_m")
        box = f" and of a table box with its top at z={table:g} m" if table is not None else ""
        return (f"IK targets: each planned joint target is checked for self-collision{box}. "
                f"Paths: preview samples arm moves and standalone jaw sweeps (at most "
                f"{self.deployment['ik_service']['path_check_step_rad']:g} rad of joint change apart) against "
                f"self-collision{box}, the request's obstacles boxes and its held_object box, and reports the first "
                "collision per stage (path_check, stage_plans[<stage>].path.collision). It checks the commanded joint "
                "path, not the measured one: the loaded arm runs below it by the amount each executed move reports as "
                "transit.max_below_path_m. Advisory: nothing is blocked or "
                "re-routed; objects not passed as obstacles / held_object are not checked; execution itself checks nothing")

    def _ik(self, pose, seed, tolerances, split=None, world=None):
        """Plan one TCP pose from ``seed`` (rad) with the deployment's IK solver; returns ``(q_rad, record)``.

        ``world`` (cuRobo only; ``ik_plan.collision_world`` plus ``jaw_rad``, or None): obstacle boxes and held-object
        spheres; solutions free of them are preferred and the record says ``obstacle_free`` (``curobo_solve``).

        ``tolerances``: ``(position_tolerance_m, orientation_tolerance_rad)``; ``split``: the path's approach / roll
        bounds (``ik_plan.split_tolerances``) or None. With ``ik_service`` the cuRobo service proposes solutions and
        ``ik_plan.curobo_solve`` checks them with this arm's FK; an unreachable service fails the pose with
        ``reason`` (``ik_service_unreachable`` ...), never falling back to the local solver. Without it the packaged
        solver plans (``ik_plan.local_solve``: weighted pose IK, then the position-first solve)."""
        if self.ik_client is None:
            return local_solve(self.kin, self.model, pose, seed, tolerances, self.deployment["ik"], self.limits)
        try:
            return curobo_solve(self.ik_client, self.kin, self.model, self.limits, [(pose, tolerances, split)], seed,
                                world)[0]
        except IKServiceError as error:
            return list(seed), dict(ok=False, solver="curobo", reason=error.reason)

    def _split(self, path):
        """The path's split approach / roll orientation bounds (rad), or None (one orientation angle)."""
        return split_tolerances(path) if "approach_roll" in self.embodiment.supports else None

    @staticmethod
    def _tolerances(path):
        return float(path["position_tolerance_m"]), float(path["orientation_tolerance_rad"])

    @staticmethod
    def _window(path):
        """Vertical arrival window of one move (m): the request's ``max_upward_dz_m``, 0 (none) when it sets none."""
        return float(path.get("max_upward_dz_m") or 0.0)

    def _feedforward(self, q_rad):
        """Gravity feedforward goal offset at ``q_rad`` (rad per arm joint; ``so101_gravity.feedforward_offset_rad``:
        ``-gravity_torque / joint stiffness``, bounded by ``gravity_feedforward.max_offset_rad``), or None when the
        deployment sets no ``motion_compensation.gravity_feedforward``."""
        if self._gravity is None:
            return None
        from ni_rua.robot.embodiments.so101.gravity import feedforward_offset_rad, gravity_torques
        inertials, stiffness, limit = self._gravity
        return feedforward_offset_rad(gravity_torques(self.model, inertials, q_rad), stiffness, limit)

    def _learn_hold_offset(self, measured_q, feedforward=None):
        """Store the hold offset of a settled pose: ``arm_goal - feedforward - measured`` (rad per arm joint), bounded
        by ``motion.arrival_correction_max_offset_rad``. ``feedforward``: the gravity feedforward contained in
        ``arm_goal`` (``_feedforward`` of the joint target) or None when the goal has none (contact hold, no
        ``gravity_feedforward``). Only the residual is stored, so the bound does not grow with the feedforward and a
        later hold at another pose adds the feedforward of THAT pose (``_stop``)."""
        limit = self.deployment["motion"]["arrival_correction_max_offset_rad"]
        residual = np.asarray(self.arm_goal) - np.asarray(measured_q) - (0.0 if feedforward is None else feedforward)
        self.hold_offset = residual.clip(-limit, limit).tolist()
        self.hold_feedforward = feedforward is not None

    def _forget_offsets(self):
        """Drop every learned offset (hold offset, carried offset): torque went off or the goal was pinned to the
        present pose, so nothing learned under the previous goal describes the arm any more."""
        self.hold_offset, self.hold_feedforward, self.path_offset_goal = [0.0] * 5, False, None

    def _carry(self, path, correction):
        """The sag offset one move carries along its path (``motion_compensation.carry_arrival_offset``).

        ``correction``: the move's ``(rounds, max_offset_rad)`` (``contract.arrival_correction``). Returns
        ``(offset, source)``: ``(None, None)`` when the deployment leaves the mechanism off; else a 5-vector (rad)
        and why: ``last_arrival`` (the offset the last settled free-space arrival ended with, clipped to this move's
        ``max_offset_rad``), ``correction_disabled`` (this move sets 0 rounds or a 0 rad bound: zeros),
        ``contact_stage`` (``motion_compensation.contact_stage``: zeros) or ``unavailable`` (no such arrival, or the
        arm goal changed since it: first move, after a contact hold, stop signal, torque_on or operator motion: zeros)."""
        if not self.compensation.get("carry_arrival_offset"):
            return None, None
        rounds, max_offset = correction
        if rounds == 0 or max_offset <= 0:
            return np.zeros(5), "correction_disabled"
        if contact_stage(path):
            return np.zeros(5), "contact_stage"
        if self.path_offset_goal is None or self.path_offset_goal != self.arm_goal:
            return np.zeros(5), "unavailable"
        return np.clip(self.path_offset, -max_offset, max_offset), "last_arrival"

    def _effective_tolerances(self, path):
        """The arrival tolerances and stall guard this move uses, exactly as the compiled request gives them.

        ``max_upward_dz_m`` (m; 0 = no vertical window, the default) when the profile supports the window (rule:
        ``arrival_verdict``). ``max_tracking_divergence_m`` (m) and ``stall_guard_preset`` appear when a stall guard is
        on: a measured TCP further than that from the commanded joint path stops the move (``_move``: a failed stage,
        or with ``contact_stop`` a hold judged by the arrival rule). ``tolerance_preset`` names the preset whose numbers
        were the base, when one was. ``arrival_correction_rounds`` (count) / ``arrival_correction_max_offset_rad`` (rad
        per joint): the bounds of the sag correction after settling (``_arrive``), the path's own or the deployment's."""
        position, orientation = self._tolerances(path)
        result = dict(position_tolerance_m=position, orientation_tolerance_rad=orientation,
                      orientation_checked=not orientation_is_free(orientation), source="request",
                      checked_on="FK of measured joints after settling")
        if "max_upward_dz" in self.embodiment.supports:
            result["max_upward_dz_m"] = self._window(path)
        if path.get("max_tracking_divergence_m") is not None:
            result["max_tracking_divergence_m"] = float(path["max_tracking_divergence_m"])
            if path.get("stall_guard_preset") not in (None, "off"):
                result["stall_guard_preset"] = path["stall_guard_preset"]
        if path.get("tolerance_preset"):
            result["tolerance_preset"] = path["tolerance_preset"]
        split = self._split(path)
        if split is not None:
            # Orientation is judged per component: the approach axis (TCP +Z) and the roll about it.
            result.update(split, orientation_checked=not all(orientation_is_free(value) for value in split.values()),
                          orientation_check="approach_and_roll")
        if "arrival_correction" in self.embodiment.supports:
            # The sag-correction bounds this move uses: its own fields, else the deployment's (contract.arrival_correction).
            rounds, offset = contract.arrival_correction(path, self.deployment)
            result.update(arrival_correction_rounds=rounds, arrival_correction_max_offset_rad=offset)
        return result

    def _interrupt(self, run, phase):
        """Return a stop reason (str) or None. Never sends commands."""
        if self.stop_generation != run["generation"]:
            return self.stop_reason or "operator_hold"
        if run["cancel"].is_set():
            return "cancelled"
        if self.clock.now() > run["deadline"]:
            return phase + "_timeout"
        return None

    def _tick(self, run, period):
        run["next_tick"] = max(run["next_tick"] + period, self.clock.now()) if run.get("next_tick") else self.clock.now() + period
        self.clock.sleep(run["next_tick"] - self.clock.now())
        with self.bus_lock:
            sample = self._read()
            step = self.control_step
        run["trace"].append({"t_s": round(sample["t"] - run["t0"], 4), "q_rad": sample["q_rad"], "gripper_rad": sample["gripper_rad"]})
        self._boundary_sample(run, sample, step)
        return sample

    def _boundary_sample(self, run, sample, step):
        """Keep one measured robot_state per control step for an action run's ``actual_trace``.

        The last joint read taken while ``step`` was the latest dispatched command replaces any
        earlier read at that step: one control period after an interpolation command, the settled
        read after the last command of a stage. ``t_s``: host monotonic s since the run started.
        """
        boundary = run.get("boundary")
        if boundary is None:
            return
        start = not boundary or boundary[0]["sim_step"] == step
        entry = {"sim_step": step, "t_s": round(sample["t"] - run["t0"], 4), "state": self._trace_state(sample, step, full=start)}
        if boundary and boundary[-1]["sim_step"] == step:
            boundary[-1] = entry
        else:
            boundary.append(entry)

    def _trace_state(self, sample, step, *, full):
        """robot_state for trace evidence. Later samples round joints (rad), TCP position (m) and quaternion to 1e-6.

        The start sample (also ``before_state``) is complete and unrounded: it is the same joint read as the
        execution-start observation, and post-action geometry/overlays of an arm-mounted camera match the two exactly
        (``trajectory_overlay``, atol 1e-8), so rounding it would make the wrist overlay unavailable. Later samples drop what cannot change
        within one action (joint_names, top-level frame_id/tcp_config_id, the command echo and the
        constant "no contact sensor" note); trajectory feedback takes joint names from before_state.
        Size: about 0.4 KB per control step (see docs/development/so101-integration.md §4.2).
        """
        state = self.robot_state(sample, step)
        arm = state["arms"][self.arm]
        if full:
            return state
        arm["q_rad"] = [round(float(v), 6) for v in arm["q_rad"]]
        for key in ("position_m", "quaternion_xyzw"):
            arm["ee_pose"][key] = [round(float(v), 6) for v in arm["ee_pose"][key]]
        if not full:
            arm.pop("joint_names")
            for key in ("command_normalized", "command_source_kind", "contact_feedback", "contact_feedback_reason"):
                arm["gripper"].pop(key, None)
            state.pop("frame_id")
            state.pop("tcp_config_id")
        return state

    def _start_sample(self, run):
        """Measured start of an action run (before its first command): before_state and the first boundary sample.

        Returns ``(sample, step)``; ``step`` is the action's start_step, read together with the joints.
        """
        with self.bus_lock:
            sample = self._read()
            step = self.control_step
        self._boundary_sample(run, sample, step)
        return sample, step

    def _start_frame(self, receipt):
        """Execution-start observation (design 13 E3): ``observe()`` (joints, then one frame per camera received after
        that read) before the action's first command, so its ``sim_step`` is the action's start step. Stored as
        ``receipt["before_observation"]``, which the Harness publishes as a historical observation
        (``execution_start_observation_id``) for post-action geometry and overlays. A failed capture never blocks the
        motion: the receipt then carries ``execution_start_observation_error`` instead."""
        try:
            receipt["before_observation"] = self.observe()
        except Exception as error:
            receipt["execution_start_observation_error"] = f"{type(error).__name__}: {error}"

    def _settle(self, run, timeout_s, *, joints="arm", interruptible=True):
        """Wait for measured velocity below threshold for settle_hold_s. Returns (settled, sample, reason)."""
        motion, period = self.deployment["motion"], 1 / self.deployment["motion"]["command_rate_hz"]
        start = self.clock.now()
        previous = self.read()
        quiet_since = None
        while True:
            if interruptible:
                reason = self._interrupt(run, "settle")
                if reason:
                    return False, previous, reason
            sample = self._tick(run, period)
            slip_hook = run.get("_carry_slip_hook")
            if slip_hook is not None and joints == "arm" and interruptible:
                slip_reason = slip_hook(sample)
                if slip_reason is not None:
                    return False, sample, slip_reason
            if interruptible:
                reason = self._interrupt(run, "settle")
                if reason:
                    return False, sample, reason
            if sample["t"] - start > timeout_s:
                return False, sample, "settle_timeout"
            dt = max(sample["t"] - previous["t"], 1e-9)
            values = (np.abs(np.subtract(sample["q_rad"], previous["q_rad"])) if joints == "arm"
                      else np.abs([sample["gripper_rad"] - previous["gripper_rad"]]))
            velocity = float(values.max()) / dt
            previous = sample
            if velocity < motion["settle_velocity_rad_s"]:
                quiet_since = sample["t"] if quiet_since is None else quiet_since
                if sample["t"] - quiet_since >= motion["settle_hold_s"]:
                    return True, sample, None
            else:
                quiet_since = None
            if sample["t"] - start > timeout_s:
                return False, sample, "settle_timeout"

    def _stop(self, run=None, *, sag_offset=True):
        """Hold the measured position and confirm by settling. Never raises.

        ``sag_offset`` True (default, every stop and pause): the goal is the measured joints plus the learned
        sag offset ``hold_offset`` (rad). False (a contact_stop hold, ``_move``): the goal is the measured joints
        exactly. The stop confirms the measured settling window under that single goal. Writing another goal after
        settling would invalidate the confirmation. The offset used is reported as ``hold_offset_rad``.

        A sag-offset hold taken while the goal is still the one the last free-space arrival left (a pause between
        stages) keeps that arrival's carried offset valid for the next move (``_carry``): the verified goal becomes
        its reference. Every other stop ends it.

        ``hold_offset`` is the residual learned at the last settled pose (``_learn_hold_offset``); when that pose's
        goal contained the gravity feedforward, the hold adds ``_feedforward`` of the joints it holds, so the
        compensation follows the pose instead of being frozen at the one where it was learned."""
        motion = self.deployment["motion"]
        local = run or self._new_run(threading.Event(), motion["stop_confirm_timeout_s"] * 2, "operator")
        previous_phase = self.motion_phase
        self.motion_phase = "stopping"
        carried = sag_offset and self.path_offset_goal is not None and self.path_offset_goal == self.arm_goal
        self.path_offset_goal = None
        try:
            with self.bus_lock:
                sample = self._read()
                offset = np.asarray(self.hold_offset) if sag_offset else np.zeros(len(self.hold_offset))
                if sag_offset and self.hold_feedforward:
                    offset = offset + self._feedforward(sample["q_rad"])
                self._send((np.asarray(sample["q_rad"]) + offset).tolist(), self.jaw_goal, run=local, kind="hold")
            settled, _, reason = self._settle(local, motion["stop_confirm_timeout_s"], interruptible=False)
            if settled and carried:
                self.path_offset_goal = list(self.arm_goal)
            self.motion_phase = "holding" if settled else "fault"
            return {"confirmed": bool(settled), "method": "measured_position_goal_then_measured_settling",
                    "hold_offset_rad": [round(float(v), 5) for v in offset],
                    "settle_velocity_rad_s": motion["settle_velocity_rad_s"], "settle_hold_s": motion["settle_hold_s"],
                    **({} if settled else {"state": "unknown", "reason": reason})}
        except Exception as error:
            self.motion_phase = "fault"
            self.faults = (self.faults + [f"stop_failed: {type(error).__name__}"])[-10:]
            return {"confirmed": False, "state": "unknown", "reason": f"{type(error).__name__}: {error}"}
        finally:
            if self.motion_phase == "stopping":
                self.motion_phase = previous_phase

    def _confirm_stop(self, run=None, *, sag_offset=True):
        """Serialize measured hold with operator controls; a relaxed arm cannot confirm a hold.

        ``run``: the interrupted motion run, so its hold commands and settling reads stay in its traces.
        ``sag_offset``: as for ``_stop``.
        """
        with self.operator_lock:
            if self.relaxed:
                return {"confirmed": False, "state": "torque_off", "reason": "operator_relax"}
            return self._stop(run, sag_offset=sag_offset)

    def _new_run(self, cancel, timeout_s, action_id, *, boundary=False):
        """``boundary=True`` (Harness actions) also keeps per-step measured robot_state samples for ``actual_trace``."""
        now = self.clock.now()
        return {"cancel": cancel, "deadline": now + timeout_s, "t0": now, "generation": self.stop_generation,
                "trace": [], "dispatched": [], "action_id": action_id, "commands": 0,
                "boundary": [] if boundary else None}

    def _jaw_step_normalized(self, gripper_speed_scale):
        """Largest jaw goal change per command (normalized 0..1): gripper.max_speed_rad_s (rad/s) x gripper_speed_scale x
        the operator speed cap, over the jaw range (rad) and motion.command_rate_hz (Hz)."""
        g, rate = self.deployment["gripper"], self.deployment["motion"]["command_rate_hz"]
        return g["max_speed_rad_s"] * gripper_speed_scale * self._cap() / abs(g["jaw_open_rad"] - g["jaw_closed_rad"]) / rate

    def _jaw_schedule(self, goals, start_q, initial_jaw, event, gripper_speed_scale, max_commands):
        """During-motion jaw schedule (design 13 R5) of one interpolated move, shared by preview and execution.

        ``goals``: the move's joint goals (rad, one row per command); ``initial_jaw``: the latched jaw goal (normalized)
        the ramp starts from; ``event``: the during_motion gripper event (``goal_normalized``, ``start_fraction``,
        ``end_fraction``: fractions of the move's commands). The shared ``gripper_motion.synchronize`` subdivides every
        interval by the smallest integer factor that keeps each jaw step within ``_jaw_step_normalized`` and keeps
        every original joint goal. Returns ``(goals, jaw_goals, metadata)`` or None when the schedule would exceed
        ``max_commands`` (= request timeout_s x command_rate_hz). ``metadata`` drops the per-command jaw list (it is the
        jaw column of planned_trace)."""
        scheduled = synchronize(np.asarray(goals, dtype=float), np.asarray(start_q, dtype=float), float(initial_jaw),
                                event, self._jaw_step_normalized(gripper_speed_scale), int(max_commands))
        if scheduled is None:
            return None
        positions, jaws, metadata = scheduled
        metadata = {key: value for key, value in metadata.items() if key != "goal_normalized_by_step"}
        metadata.update(timing_basis="this_move_commands", initial_goal_normalized=round(float(initial_jaw), 6),
                        max_goal_step=round(float(metadata["max_goal_step"]), 6))
        return positions, jaws, metadata

    def _segment(self, run, q_target, speed_scale, divergence_m=None, jaw=None, carry=None, *,
                 carry_slip_monitor=None, stage_id=None):
        """Stream an interpolated joint path to q_target. Returns ``(ok, reason, diagnostic, info)``.

        ``carry``: None, or the sag offset (rad per arm joint, ``_carry``) added to every streamed goal. With the
        deployment's gravity feedforward each goal also gets ``_feedforward`` of its own row. The joint tracking
        limit, the stall window and the TCP divergence compare the measured joints with the interpolated row itself,
        without either offset: the offsets exist so the arm follows that row. ``info["transit"]``: ``(commanded rows,
        measured joints)`` of the commands sent (rad, offset-free rows), for ``transit_deviation``.

        The offset is slewed, not switched: it starts at what the present goal already holds over the measured start
        (``arm_goal - start``: the previous arrival's offsets, or 0 after a contact hold) and moves toward
        ``carry + feedforward(row)`` by at most what ``motion.max_command_delta_rad`` leaves after the row's own
        step, so no sent goal changes by more than that published bound. ``info["offset_ramp_commands"]``: commands
        sent before the offset reached its value (0 = from the first command). Without carry and feedforward nothing
        is added and the goals are the rows themselves.

        ``jaw``: None (the jaw goal stays latched), or ``{"event", "gripper_speed_scale", "max_commands"}`` for a
        during-motion jaw ramp (``_jaw_schedule``; refused with ``gripper_motion_schedule_exceeds_timeout`` before any
        command when it does not fit). ``diagnostic`` (tracking stops only) names the stop signal ``kind``:
        ``joint_tracking_limit``, ``tcp_divergence`` (stall guard) or ``stall`` (stall window). ``info``:
        ``planned_steps`` / ``completed_steps`` (commands of this move) and, with ``jaw``, the ``gripper_motion``
        schedule metadata, and ``start_q`` (the measured joints the interpolation started from, rad)."""
        motion = self.deployment["motion"]
        rate = motion["command_rate_hz"]
        period = 1 / rate
        speed = motion["max_joint_speed_rad_s"] * self._cap() * speed_scale
        accel = motion.get("max_acceleration_rad_s2")
        jerk = motion.get("max_jerk_rad_s3")
        start = np.asarray(self.read()["q_rad"])
        steps = interpolation_steps(
            start, q_target, speed, rate, max_acceleration_rad_s2=accel, max_jerk_rad_s3=jerk
        )
        goals = joint_interpolation(
            start,
            q_target,
            steps,
            speed_rad_s=speed,
            rate_hz=rate,
            max_acceleration_rad_s2=accel,
            max_jerk_rad_s3=jerk,
        )  # the rows preview evaluated, from the measured start
        jaws = None
        info = {"planned_steps": steps, "completed_steps": 0, "start_q": start.tolist(), "transit": ([], [])}
        if jaw is not None:
            schedule = self._jaw_schedule(goals, start, self.jaw_goal, jaw["event"], jaw["gripper_speed_scale"],
                                          jaw["max_commands"])
            if schedule is None:
                return False, "gripper_motion_schedule_exceeds_timeout", None, info
            goals, jaws, info["gripper_motion"] = schedule
            info["planned_steps"] = len(goals)
        history = []
        compensated = carry is not None or self._gravity is not None
        applied, previous_row = np.asarray(self.arm_goal) - start, start
        info["offset_ramp_commands"] = 0
        for index, row in enumerate(goals):
            reason = self._interrupt(run, "execution")
            if reason:
                return False, reason, None, info
            goal = [float(v) for v in row]  # the commanded row: what the guards and every residual refer to
            sent = goal
            if compensated:
                feedforward = self._feedforward(goal)
                wanted = (0.0 if carry is None else carry) + (np.zeros(5) if feedforward is None else feedforward)
                room = np.maximum(motion["max_command_delta_rad"] - np.abs(np.asarray(goal) - previous_row), 0.0)
                gap = wanted - applied
                applied = np.where(np.abs(gap) <= room, wanted, applied + np.sign(gap) * room)
                if not np.array_equal(applied, wanted):
                    info["offset_ramp_commands"] = index + 1
                sent, previous_row = (np.asarray(goal) + applied).tolist(), np.asarray(goal)
            if not self._send(sent, self.jaw_goal if jaws is None else float(jaws[index]), run["generation"], run=run,
                              commanded=goal if compensated else None, jaw_command=(jaws is not None and index == 0)):
                return False, self.stop_reason or "operator_hold", None, info
            run["commands"] += 1
            info["completed_steps"] += 1
            sample = self._tick(run, period)
            info["transit"][0].append(goal)
            info["transit"][1].append(sample["q_rad"])
            if carry_slip_monitor is not None:
                ap = self.aperture_m(sample["gripper_rad"])
                slip = carry_slip_monitor.observe_step(stage_id or "", index, ap)
                if slip is not None:
                    info["carry_slip"] = slip
                    return False, CARRY_SLIP_REASON, None, info
            error = np.abs(np.asarray(goal) - np.asarray(sample["q_rad"]))
            self.last_tracking_error = float(error.max())
            if error.max() > motion["tracking_error_limit_rad"]:
                joint = int(error.argmax())
                return False, "tracking_error", dict(kind="joint_tracking_limit", joint=ARM_JOINTS[joint],
                    tracking_error_rad=round(float(error[joint]), 5), limit_rad=motion["tracking_error_limit_rad"]), info
            if divergence_m is not None:
                divergence = float(np.linalg.norm(self.tcp(goal)[:3, 3] - self.tcp(sample["q_rad"])[:3, 3]))
                if divergence > divergence_m:
                    return False, "tracking_error", dict(kind="tcp_divergence", divergence_m=round(divergence, 5),
                                                         max_tracking_divergence_m=divergence_m), info
            history.append((sample["t"], np.asarray(goal), np.asarray(sample["q_rad"])))
            while history and sample["t"] - history[0][0] > motion["stall_window_s"]:
                oldest = history.pop(0)
                commanded = np.abs(history[-1][1] - oldest[1]) if history else np.zeros(5)
                measured = np.abs(history[-1][2] - oldest[2]) if history else np.zeros(5)
                stalled = (commanded >= motion["stall_min_commanded_rad"]) & (measured < motion["stall_min_progress_rad"])
                if stalled.any():
                    joint = int(np.argmax(stalled))
                    return False, "tracking_error", dict(kind="stall", joint=ARM_JOINTS[joint],
                        commanded_progress_rad=round(float(commanded[joint]), 5),
                        measured_progress_rad=round(float(measured[joint]), 5), window_s=motion["stall_window_s"]), info
        return True, None, None, info

    def _jaw_measurement(self, run, goal_normalized):
        """Wait (bounded by gripper.settle_timeout_s) for the measured jaw to settle, then measure it.

        Returns ``(ok, reason, facts, sample)``; ``facts``: ``measured_jaw_rad`` (rad), ``measured_opening_m`` (CAD
        fingertip gap, m), ``measured_normalized``, ``jaw_settled`` (False: still moving at the timeout, not a final
        reading) and ``measured_vs_goal_normalized`` (against the commanded ``goal_normalized``); ``sample``: the
        measured sample they describe. An interruption (stop, cancel, deadline) returns ``ok=False``, no facts and
        no sample."""
        settled, sample, reason = self._settle(run, self.deployment["gripper"]["settle_timeout_s"], joints="gripper")
        if reason and reason not in ("settle_timeout",):
            return False, reason, {}, None
        from ni_rua.robot.model.so101_gripper_geometry import opening_width_m
        return True, None, dict(measured_jaw_rad=round(sample["gripper_rad"], 5),
                                measured_opening_m=round(opening_width_m(sample["gripper_rad"]), 4),
                                measured_normalized=round(sample["gripper_normalized"], 4), jaw_settled=settled,
                                measured_vs_goal_normalized=round(sample["gripper_normalized"] - goal_normalized, 4)), sample

    def _squeeze(self, stage=None):
        """Holding squeeze (rad of the jaw joint) of one jaw command, or None when the deployment applies none
        (``gripper.hold_squeeze_rad`` null): the stage's ``grip_squeeze_rad`` (validated, ``contract.validate_action``)
        when it sets one, else the deployment default. ``stage``: the batch stage dict, or None (single requests and
        operator moves use the default)."""
        limits = hold_squeeze(self.deployment)
        if limits is None:
            return None
        value = (stage or {}).get(GRIP_SQUEEZE_FIELD)
        return limits[0] if value is None else float(value)

    def _jaw_load(self):
        """Present_Load of the jaw servo in percent of its maximum torque (signed, uncalibrated; not a force), or
        None when the driver cannot read it (``driver.arm_load``) or the read fails. Never raises."""
        reader = getattr(self.driver, "arm_load", None)
        if reader is None:
            return None
        try:
            with self.bus_lock:
                value = float(reader(("gripper",))["gripper"])
        except Exception:
            return None
        return round(value / 10.0, 1) if math.isfinite(value) else None

    def _hold_squeeze(self, run, goal_normalized, facts, sample, squeeze_rad):
        """Bounded holding squeeze after a jaw command settled. Returns ``(ok, reason, facts, report)``.

        ``facts`` / ``sample``: the settled measurement of ``_jaw_measurement``; ``squeeze_rad``: ``_squeeze`` (rad
        of the jaw joint, None = off). The servo is position-P controlled, so a jaw stopped short of its goal pushes
        with a torque that grows with (goal - measured) up to its torque limit for as long as the goal is held. When
        the last jaw command closed the jaw (``jaw_closing``), the jaw settled, and it rests further than
        ``squeeze_rad`` from the commanded goal (contact: an object, or the other fingertip), the jaw goal is set once
        to the measured jaw angle moved ``squeeze_rad`` toward closed (command ``kind`` ``hold_squeeze``, counted in
        the run's commands; the arm goal is kept) and the jaw is measured again after settling: the returned
        ``facts`` are that reading, the one every later consumer (grip_facts, grip guard) sees. ``jaw_goal`` stays
        at this hold goal through later moves, waits, pauses and holds (they resend it) until the next jaw command.

        ``report`` (``gripper_execution.hold_squeeze``): None when no squeeze is configured or the command did not
        close the jaw (opening: nothing to report). Otherwise ``squeeze_rad`` and ``applied``; not applied carries
        ``reason`` ``jaw_within_squeeze_of_goal`` (no contact: the commanded goal already holds no harder than the
        squeeze) or ``jaw_not_settled`` (still moving at gripper.settle_timeout_s: no final contact reading; the
        commanded goal is kept). Applied carries ``contact_jaw_rad`` (rad) and ``contact_opening_m`` (CAD fingertip
        gap, m) measured at contact before the goal change, ``hold_goal_normalized`` / ``hold_goal_rad`` (the
        commanded hold goal) and, when the driver reads it, ``contact_load_pct`` / ``hold_load_pct`` (jaw
        Present_Load before the change and after settling at the hold goal, percent of maximum torque).
        ``ok`` False: a stop, cancel or deadline interrupted the hold write or its settling (``reason``)."""
        if squeeze_rad is None or not self.jaw_closing:
            return True, None, facts, None
        report = {"squeeze_rad": round(float(squeeze_rad), 5), "applied": False}
        if not facts["jaw_settled"]:
            return True, None, facts, dict(report, reason="jaw_not_settled")
        g = self.deployment["gripper"]
        hold = sample["gripper_normalized"] - squeeze_rad / abs(g["jaw_open_rad"] - g["jaw_closed_rad"])
        if hold <= goal_normalized:
            return True, None, facts, dict(report, reason="jaw_within_squeeze_of_goal")
        contact_load = self._jaw_load()
        report.update(applied=True, contact_jaw_rad=facts["measured_jaw_rad"], contact_opening_m=facts["measured_opening_m"],
                      hold_goal_normalized=round(hold, 4),
                      hold_goal_rad=round(gripper_rad_from_normalized(self.deployment, hold), 5),
                      **({} if contact_load is None else {"contact_load_pct": contact_load}))
        state = {"hold_goal_normalized": hold, "squeeze_rad": float(squeeze_rad)}
        if not self._send(self.arm_goal, hold, run["generation"], run=run, kind="hold_squeeze", hold_squeeze=state):
            return False, self.stop_reason or "operator_hold", facts, dict(report, applied=False, reason="interrupted")
        run["commands"] += 1
        ok, reason, held, _ = self._jaw_measurement(run, goal_normalized)
        if not ok:
            return False, reason, facts, report
        hold_load = self._jaw_load()
        if hold_load is not None:
            report["hold_load_pct"] = hold_load
        return True, None, held, report

    def _grip_hold(self, sample):
        """``grip_hold`` of a move made while a holding squeeze is in effect (``jaw_hold``): ``hold_goal_normalized``
        (the held jaw goal) and ``squeeze_remaining_rad`` (rad of the jaw joint: measured jaw angle minus the hold
        goal, toward open). While a rigid object resists it stays near the squeeze applied; if the object slips out or
        compresses, an unloaded STS3215 jaw closes toward ``hold_goal_normalized`` up to its static-friction deadband
        (about 0.03-0.04 rad at P=16 on real hardware, or 0 in an ideal frictionless model)."""
        g = self.deployment["gripper"]
        remaining = (sample["gripper_normalized"] - self.jaw_hold["hold_goal_normalized"]) * abs(g["jaw_open_rad"] - g["jaw_closed_rad"])
        return {"hold_goal_normalized": round(self.jaw_hold["hold_goal_normalized"], 4),
                "squeeze_remaining_rad": round(remaining, 5)}

    def _ramp_jaw(self, run, event, gripper_speed_scale, squeeze_rad=None):
        """Jaw result of a move that carried a during-motion ramp (``event``), after its arrival. Returns
        ``(ok, reason, gripper_execution)``. ``squeeze_rad``: ``_squeeze`` of the stage; the holding squeeze
        (``_hold_squeeze``) is applied after the arrival and the settled jaw reading, as for a gripper stage, so
        between the end of the ramp and the arrival the jaw still pushes toward the commanded goal.

        Normally the ramp ended inside the move and the jaw is only measured after settling (``_jaw_measurement``).
        When a contact_stop hold ended the move before the ramp reached its goal, the rest of the ramp runs at the held
        pose like a gripper stage (``_gripper``: same jaw speed limit, arm goal kept), as on ARX-X5 where the held rows
        keep their jaw schedule; ``finished_at_held_pose`` is then true."""
        goal = event["goal_normalized"]
        if abs(self.jaw_goal - goal) > 1e-9:
            ok, reason, jaw = self._gripper(run, goal, gripper_speed_scale, squeeze_rad)
            return ok, reason, dict(jaw, phase="during_motion", finished_at_held_pose=True)
        ok, reason, facts, sample = self._jaw_measurement(run, goal)
        report = None
        if ok:
            ok, reason, facts, report = self._hold_squeeze(run, goal, facts, sample, squeeze_rad)
        return ok, reason, dict(goal_normalized=goal, phase="during_motion",
                                semantics="jaw command completion; not grasp or retention proof", **facts,
                                **({} if report is None else {"hold_squeeze": report}))

    def _gripper(self, run, goal_normalized, gripper_speed_scale, squeeze_rad=None):
        """Ramp the jaw goal from the held goal ``jaw_goal``, wait (bounded) for the measured jaw to settle, then
        apply the holding squeeze (``_hold_squeeze``; ``squeeze_rad``: ``_squeeze`` of the stage, None = off). The
        measured facts in the result are the last settled reading (after the squeeze when one was applied)."""
        rate = self.deployment["motion"]["command_rate_hz"]
        step = self._jaw_step_normalized(gripper_speed_scale)
        start = self.jaw_goal
        count = max(1, int(math.ceil(abs(goal_normalized - start) / step)))
        result = {"goal_normalized": goal_normalized, "phase": "after_motion", "planned_steps": count, "completed_steps": 0,
                  "command_complete": False, "semantics": "jaw command completion; not grasp or retention proof"}
        for index in range(1, count + 1):
            reason = self._interrupt(run, "gripper")
            if reason:
                return False, reason, result
            value = start + (goal_normalized - start) * index / count
            if not self._send(self.arm_goal, value, run["generation"], run=run, jaw_command=(index == 1)):
                return False, self.stop_reason or "operator_hold", result
            run["commands"] += 1
            result["completed_steps"] += 1
            self._tick(run, 1 / rate)
        result["command_complete"] = True
        ok, reason, facts, sample = self._jaw_measurement(run, goal_normalized)
        if not ok:
            return False, reason, result
        ok, reason, facts, report = self._hold_squeeze(run, goal_normalized, facts, sample, squeeze_rad)
        result.update(facts)
        if report is not None:
            result["hold_squeeze"] = report
        return ok, reason, result

    def _contact_block_summary(self, sample, pose, position_tolerance_m, info, stage_id=None):
        """Where a failed or contact-stopped move ended relative to its target (design 13 R7; ARX-X5 key names).

        From one measured joint sample (FK of the TCP, m in the profile frame) and the target ``pose``:
        ``vertical_shortfall_m`` = measured z - target z (> 0: above the target), ``horizontal_offset_m``,
        ``approach_shortfall_m`` = (target - measured) along the target's TCP approach axis (profile
        ``gripper.approach_axis_local``, TCP +Z; > 0: stopped short of the target), ``blocked_above_target`` =
        ``vertical_shortfall_m > position_tolerance_m`` (the stage's own tolerance; no other threshold),
        ``completed_steps`` / ``planned_steps`` (commands of the move)."""
        measured, target = self.tcp(sample["q_rad"]), pose_matrix(pose)
        here, there = measured[:3, 3], target[:3, 3]
        axis = target[:3, :3] @ np.asarray(self.embodiment.gripper.approach_axis_local, dtype=float)
        vertical = float(here[2] - there[2])
        return {"arm": self.arm, **({"stage_id": stage_id} if stage_id is not None else {}),
                "measured_position_m": [round(float(v), 4) for v in here],
                "target_position_m": [round(float(v), 4) for v in there],
                "vertical_shortfall_m": round(vertical, 4),
                "horizontal_offset_m": round(float(math.hypot(*(here[:2] - there[:2]))), 4),
                "approach_shortfall_m": round(float(np.dot(there - here, axis)), 4),
                "blocked_above_target": bool(vertical > position_tolerance_m),
                "completed_steps": int(info["completed_steps"]), "planned_steps": int(info["planned_steps"])}

    def _move(self, run, q_target, pose, path, *, speed_scale=1.0, jaw=None, stage_id=None, planned_start_q=None,
              carry_slip_monitor=None):
        """One move to a cached IK joint target with its stop and arrival rules. Returns ``(ok, reason, fields)``.

        The segment streams the interpolated path (``_segment``; with ``jaw`` the during-motion jaw ramp). After a
        full segment ``_arrive`` settles, corrects sag and judges the arrival rule (``arrival_verdict`` with the
        path's ``max_upward_dz_m`` window). A stop signal (``tracking_error``: joint tracking limit, stall window or
        the stall guard's TCP divergence) fails the move, except on a ``stall_guard_preset == "contact_stop"`` path:
        the arm then holds the measured pose (``_confirm_stop`` without the sag offset, confirmed by settling; no
        arrival correction) and the held pose is judged by the same arrival rule; within it the move completes (``contact_stopped_early``),
        otherwise it fails. ``fields`` (stage / waypoint result): the residual and verdict (``dx_m`` ..
        ``within_tolerance``, ``vertical_compliance_accepted`` + ``xy_error_m`` when the window passed it),
        ``arrival_correction``, ``tracking_stop`` (stop signal with ``completed_steps`` / ``planned_steps``),
        ``hold_confirmation`` (the contact hold), ``gripper_motion`` (jaw schedule with ``completed_steps``) and, on a
        failed ``tracking_error`` or a contact hold, ``contact_block_summary``. With ``planned_start_q`` (the joints the
        preview assumed this move starts from, rad): ``path_start_deviation_rad``, the largest joint difference between
        that and the measured start the interpolation really used (the previewed path facts hold up to that shift).
        A failed ``tracking_error`` also returns ``endpoint_residual_m`` (the measured TCP residual: the arrival
        residual, or the residual of a fresh joint read after a stop), which callers move to receipt level.

        The sag correction is bounded by the path's ``arrival_correction_rounds`` / ``arrival_correction_max_offset_rad``
        (``contract.arrival_correction``; default the deployment's).

        Always: ``transit`` (``transit_deviation``: the measured TCP against the commanded path while streaming; absent
        when no command was sent) and ``duration_s`` (wall-clock s of this move, streaming to verdict). With
        deployment ``motion_compensation``: ``path_compensation`` = ``carried_offset_rad`` + ``carry`` (``_carry``:
        the sag offset added to every streamed goal and where it came from) and / or ``gravity_feedforward_rad``
        (the feedforward offset at the joint target, rad per arm joint); the offsets slew in under
        ``max_command_delta_rad`` (``_segment``; every sent offset is in ``planned_trace.goal_offset_rad``). A stop
        signal, a failed arrival and an arrival that used its whole offset bound
        end the carried offset (``_arrive``)."""
        began = self.clock.now()
        correction = contract.arrival_correction(path, self.deployment)
        tolerances = (*self._tolerances(path), self._window(path), self._split(path), correction)
        carry, carry_source = self._carry(path, correction)
        ok, reason, diagnostic, info = self._segment(run, q_target, speed_scale, path.get("max_tracking_divergence_m"), jaw,
                                                     carry, carry_slip_monitor=carry_slip_monitor, stage_id=stage_id)
        fields = {}
        compensation = {}
        if carry_source is not None:
            compensation.update(carried_offset_rad=[round(float(v), 5) for v in carry], carry=carry_source)
        feedforward = self._feedforward(q_target)
        if feedforward is not None:
            compensation["gravity_feedforward_rad"] = [round(float(v), 5) for v in feedforward]
        if compensation:
            fields["path_compensation"] = compensation
        if not ok:
            self.path_offset_goal = None  # a stop (signal, hold, cancel) is not a free-space arrival
        if planned_start_q is not None and "start_q" in info:
            fields["path_start_deviation_rad"] = round(float(np.abs(np.subtract(info["start_q"], planned_start_q)).max()), 5)
        if "gripper_motion" in info:
            fields["gripper_motion"] = dict(info["gripper_motion"], completed_steps=info["completed_steps"])
        if ok:
            # The arrival keeps its offset for the next move only from a free-space move (never from contact).
            arrive_opts = dict(
                start_offset=carry,
                keep_offset=carry_source in ("last_arrival", "unavailable"),
                stop_within_tolerance=bool(self.compensation.get("stop_correction_within_tolerance")),
                carry_slip_monitor=carry_slip_monitor,
                stage_id=stage_id,
                step_offset=info["completed_steps"],
            )
            if "require_settled" in path:
                arrive_opts["require_settled"] = bool(path["require_settled"])
            ok, reason, residual = self.settle_and_correct(run, q_target, pose, tolerances, **arrive_opts)
            if not ok:
                self.path_offset_goal = None
            fields.update(residual or {})
        elif reason == CARRY_SLIP_REASON and "carry_slip" in info:
            fields["carry_slip"] = info["carry_slip"]
        elif reason == "tracking_error":
            fields["tracking_stop"] = dict(diagnostic, completed_steps=info["completed_steps"],
                                           planned_steps=info["planned_steps"])
            if path.get("stall_guard_preset") == "contact_stop":
                # Hold at the contact: the goal is the measured pose, without the free-space sag offset, which on a
                # support would lift the arm off it (real arm 2026-11-14: about 7 mm with a 0.05 rad shoulder offset).
                hold = self._confirm_stop(run, sag_offset=False)
                if hold.get("confirmed"):
                    fields["hold_confirmation"] = hold
                    self.motion_phase = "moving"
                    with self.bus_lock:
                        sample = self._read()
                        step = self.control_step
                    # The hold offset of the held pose, by the rule of _arrive (held goal - measured after settling,
                    # bounded): about 0 on a support, the servo sag in free space. A later pause (observe stage,
                    # checkpoint, grip guard: _stop) then re-commands the held goal and the arm does not move; the stale
                    # free-space offset would lift it off a support (real arm: about 7 mm), and none would let a
                    # free-space hold sag further.
                    self._learn_hold_offset(sample["q_rad"])
                    # The held pose is this run's measured sample of the hold command's step (actual_trace).
                    self._boundary_sample(run, sample, step)
                    residual = pose_residual(self.tcp(sample["q_rad"]), pose_matrix(pose), split=tolerances[3] is not None)
                    ok, vertical = arrival_verdict(residual, *tolerances[:3], split=tolerances[3])
                    residual.update(settled=True, within_tolerance=ok, max_joint_error_rad=round(float(
                        np.abs(np.subtract(sample["q_rad"], q_target)).max()), 5))
                    if vertical:
                        residual.update(vertical_compliance_accepted=True,
                                        xy_error_m=round(math.hypot(residual["dx_m"], residual["dy_m"]), 5))
                    fields.update(residual, contact_stopped_early=True,
                                  contact_block_summary=self._contact_block_summary(sample, pose, tolerances[0], info,
                                                                                    stage_id))
                    reason = None if ok else "tracking_error"
        if not ok and reason in ("tracking_error", "not_settled"):
            sample = None
            if "contact_block_summary" not in fields:
                sample = self.read()
                fields["contact_block_summary"] = self._contact_block_summary(sample, pose, tolerances[0], info, stage_id)
            # The measured residual of the failure: the arrival / held-pose residual when one was judged, else that of
            # the joint read the contact summary used (a stop without a hold has no arrival verdict).
            measured = fields if "position_error_m" in fields else pose_residual(
                self.tcp((sample or self.read())["q_rad"]), pose_matrix(pose), split=tolerances[3] is not None)
            fields["endpoint_residual_m"] = {key: measured[key] for key in RESIDUAL_FIELDS if key in measured}
            fields["endpoint_error"] = {
                "position_error_m": measured["position_error_m"],
                "orientation_error_rad": measured["orientation_error_rad"],
                "settled": bool(fields.get("settled", False)),
                "within_tolerance": bool(fields.get("within_tolerance", False)),
                "effective_tolerances": self._effective_tolerances(path),
            }
        # Telemetry of the streamed part, computed after the verdict so it never delays a command.
        transit = transit_deviation(self.tcp, info["start_q"], *info["transit"], info["planned_steps"])
        if transit is not None:
            fields["transit"] = transit
        fields["duration_s"] = round(self.clock.now() - began, 3)
        return ok, reason, fields

    def _arrive(self, run, q_target, pose, tolerances, *, start_offset=None, keep_offset=False,
                stop_within_tolerance=False, require_settled: bool = True, carry_slip_monitor=None,
                stage_id=None, step_offset: int = 0):
        """Settle, then verify measured TCP against the requested tolerances: ``(position_tolerance_m,
        orientation_tolerance_rad[, max_upward_dz_m[, split[, correction]]])`` (m, rad, m; ``arrival_verdict``; no
        third value = no window; ``split``: the path's approach / roll bounds, ``ik_plan.split_tolerances``, or None).

        ``correction`` (fifth value): ``(rounds, max_offset_rad)`` of the sag correction for this move
        (``contract.arrival_correction``: the move's own bounds, never above the deployment's); absent = the
        deployment's ``motion.arrival_correction_rounds`` / ``arrival_correction_max_offset_rad``. ``rounds`` 0 sends
        no correction: the goal stays ``q_target`` and the verdict judges where the arm settled
        (``termination_reason`` ``"disabled"``). The result's ``arrival_correction.max_rounds`` / ``max_offset_rad``
        are the bounds used.

        Deployment ``motion_compensation`` (``deployment.MOTION_COMPENSATION_FIELDS``; all off by default):

        - ``start_offset`` (rad per arm joint, ``_carry``; None = zeros): the offset the move streamed with. The
          correction accumulates from it (clipped to ``max_offset_rad``) instead of from 0 and reports it as
          ``arrival_correction.start_goal_offset_rad``. ``keep_offset``: a settled arrival stores its final offset
          for the next move's ``_carry``; False (contact / compliant move, correction disabled, operator moves)
          clears the stored one. It is also cleared when the verdict fails or the correction reached its offset
          bound: an arm that stopped short of its target against something has an offset that is contact, not sag,
          and the recovery move must not carry it toward the obstacle.
        - gravity feedforward: every goal sent here is ``q_target + _feedforward(q_target) + offset``; the offsets
          and bounds of the correction do not include the feedforward.
        - ``arrival_breakaway_rad``: a correction step smaller than the joint's band is enlarged to it (sign kept);
          the round lists those joints in ``breakaway_joints``.
        - ``no_response_stop``: judged per joint. A joint whose goal changed in a round but which moved by at most one
          encoder tick (``ENCODER_RESOLUTION_RAD``) gets its step taken back (``unanswered_joints``; the goal is
          re-sent as ``arrival_correction_revert`` and the arm settles again); joints that answered keep theirs.
          ``responded`` is False when none answered, which ends the correction with ``termination_reason``
          ``"no_response"``. A discarded step therefore never enters ``hold_offset`` or the carried offset.

        - ``stop_within_tolerance`` (``_move`` passes ``motion_compensation.stop_correction_within_tolerance``): a
          round is sent only while the settled TCP is outside the arrival rule of ``tolerances`` (``arrival_verdict``);
          a settled pose inside it ends the correction with ``termination_reason`` ``"within_tolerance"`` and is the
          pose the verdict judges, so the correction cannot turn an in-tolerance arrival into a failure. The arm then
          stops inside the tolerance instead of at the IK joint target.

        ``arrival_correction.final_goal_offset_rad`` (reported with any of these on): the offset in force at the end."""
        motion = self.deployment["motion"]
        max_rounds, max_offset = (tolerances[4] if len(tolerances) > 4 else
                                  contract.arrival_correction_limits(self.deployment))
        slip_hit = None
        tick_index = int(step_offset)
        last_checked_sample_id = None

        def _check_slip_sample(sample_item):
            nonlocal slip_hit, tick_index, last_checked_sample_id
            if carry_slip_monitor is None or sample_item is None:
                return None
            last_checked_sample_id = id(sample_item)
            ap = self.aperture_m(sample_item["gripper_rad"])
            hit = carry_slip_monitor.observe_step(stage_id or "", tick_index, ap)
            tick_index += 1
            if hit is not None:
                slip_hit = hit
                return CARRY_SLIP_REASON
            return None

        def _settle_arm():
            if carry_slip_monitor is not None:
                run["_carry_slip_hook"] = _check_slip_sample
            try:
                settled_res, sample_res, reason_res = self._settle(run, motion["arrival_timeout_s"])
            finally:
                run.pop("_carry_slip_hook", None)
            if (
                carry_slip_monitor is not None
                and slip_hit is None
                and sample_res is not None
                and id(sample_res) != last_checked_sample_id
            ):
                extra = _check_slip_sample(sample_res)
                if extra is not None:
                    return False, sample_res, extra
            return settled_res, sample_res, reason_res

        settled, sample, reason = _settle_arm()
        if reason == CARRY_SLIP_REASON and slip_hit is not None:
            return False, CARRY_SLIP_REASON, {"carry_slip": slip_hit}
        interrupted = self._interrupt(run, "settle")
        if interrupted:
            return False, interrupted, None
        if reason and reason != "settle_timeout":
            return False, reason, None
        # Gravity-sag correction: shift the goal by the measured settled shortfall, bounded and
        # fully reported. It never changes the target or the tolerance used for the verdict.
        target, corrections = np.asarray(q_target, dtype=float), []
        initial_residual = pose_residual(self.tcp(sample["q_rad"]), pose_matrix(pose))
        termination = "round_limit" if max_rounds else "disabled"
        feedforward = self._feedforward(target)
        base = target if feedforward is None else target + feedforward  # the goal before any correction offset
        offset = np.zeros(5) if start_offset is None else np.clip(np.asarray(start_offset, dtype=float), -max_offset, max_offset)
        started_from = offset.copy()
        stop_unanswered = bool(self.compensation.get("no_response_stop"))
        rule = (tolerances[0], tolerances[1], tolerances[2] if len(tolerances) > 2 else 0.0)
        rule_split = tolerances[3] if len(tolerances) > 3 else None

        def planned_row(value):
            """planned_trace row of a correction command: the target shifted by the correction made HERE (as before
            motion compensation); None when the goal carries no compensation offset on top of it."""
            return None if start_offset is None and feedforward is None else (target + value - started_from).tolist()

        def inside(joints):
            """The arrival verdict of one settled joint sample (the rule the final verdict applies)."""
            return arrival_verdict(pose_residual(self.tcp(joints), pose_matrix(pose), split=rule_split is not None),
                                   *rule, split=rule_split)[0]
        for round_index in range(max_rounds):
            interrupted = self._interrupt(run, "arrival_correction")
            if interrupted:
                return False, interrupted, None
            if not settled:
                termination = "not_settled"
                break
            error = target - np.asarray(sample["q_rad"])
            if float(np.abs(error).max()) < motion["arrival_correction_min_error_rad"]:
                termination = "joint_error_below_correction_threshold"
                break
            if stop_within_tolerance and inside(sample["q_rad"]):
                termination = "within_tolerance"
                break
            step = np.where(np.abs(error) >= motion["arrival_correction_min_error_rad"], error, 0.)
            enlarged = np.zeros(5, dtype=bool)
            if self._breakaway is not None:
                enlarged = (step != 0) & (np.abs(step) < self._breakaway)
                step = np.where(enlarged, np.sign(step) * self._breakaway, step)
            previous, before = offset, np.asarray(sample["q_rad"])
            offset = np.clip(offset + step, -max_offset, max_offset)
            if not self._send((base + offset).tolist(), self.jaw_goal, run["generation"], run=run, kind="arrival_correction",
                              commanded=planned_row(offset)):
                return False, self.stop_reason or "operator_hold", None
            run["commands"] += 1
            settled, sample, reason = _settle_arm()
            if reason == CARRY_SLIP_REASON and slip_hit is not None:
                return False, CARRY_SLIP_REASON, {"carry_slip": slip_hit}
            if reason and reason != "settle_timeout":
                return False, reason, None
            corrections.append({"round": round_index + 1, "goal_offset_rad": [round(float(v), 5) for v in offset],
                                "tcp_residual": pose_residual(self.tcp(sample["q_rad"]), pose_matrix(pose)),
                                "settled_joint_error_rad": [round(float(v), 5) for v in target - np.asarray(sample["q_rad"])]})
            if enlarged.any():
                corrections[-1]["breakaway_joints"] = [ARM_JOINTS[index] for index in np.flatnonzero(enlarged)]
            changed = offset != previous
            if stop_unanswered and settled and changed.any():
                # Whole encoder ticks each joint moved; one tick is reading noise. Judged per joint: a joint that did
                # not answer its own step loses it even when another joint moved.
                ticks = np.rint(np.abs(np.asarray(sample["q_rad"]) - before) / ENCODER_RESOLUTION_RAD)
                unanswered = changed & (ticks <= 1)
                corrections[-1]["responded"] = bool((changed & ~unanswered).any())
                if unanswered.any():
                    # Take the unanswered steps back: they stayed inside static friction and would only be stored effort.
                    corrections[-1]["unanswered_joints"] = [ARM_JOINTS[index] for index in np.flatnonzero(unanswered)]
                    offset = np.where(unanswered, previous, offset)
                    if not self._send((base + offset).tolist(), self.jaw_goal, run["generation"], run=run,
                                      kind="arrival_correction_revert", commanded=planned_row(offset)):
                        return False, self.stop_reason or "operator_hold", None
                    run["commands"] += 1
                    settled, sample, reason = _settle_arm()
                    if reason == CARRY_SLIP_REASON and slip_hit is not None:
                        return False, CARRY_SLIP_REASON, {"carry_slip": slip_hit}
                    if reason and reason != "settle_timeout":
                        return False, reason, None
                    if not corrections[-1]["responded"]:
                        termination = "no_response"
                        break
        if not settled:
            termination = "not_settled"
        if settled:
            self._learn_hold_offset(sample["q_rad"], feedforward)
        position, orientation = tolerances[:2]
        window = tolerances[2] if len(tolerances) > 2 else 0.0
        split = tolerances[3] if len(tolerances) > 3 else None
        residual = pose_residual(self.tcp(sample["q_rad"]), pose_matrix(pose), split=split is not None)
        within_tol, vertical = arrival_verdict(residual, position, orientation, window, split=split)
        if vertical:
            residual.update(vertical_compliance_accepted=True,
                            xy_error_m=round(math.hypot(residual["dx_m"], residual["dy_m"]), 5))
        if not within_tol:
            ok, fail_reason = False, "tracking_error"
        elif require_settled and not settled:
            ok, fail_reason = False, "not_settled"
        else:
            ok, fail_reason = True, None
        limit_reached = bool(corrections) and bool(np.any(np.abs(offset) >= max_offset))
        # The offset the next move may carry (_carry): only an arrival that settled inside its rule in free space,
        # with offset to spare. A failed verdict or a correction at its bound means the arm may be held back by
        # something, and that offset is not gravity sag.
        self.path_offset = offset.copy()
        self.path_offset_goal = list(self.arm_goal) if settled and keep_offset and ok and not limit_reached else None
        residual.update(settled=settled, within_tolerance=within_tol,
                        max_joint_error_rad=round(float(np.abs(np.subtract(sample["q_rad"], q_target)).max()), 5),
                        arrival_correction={"rounds": corrections, "max_rounds": max_rounds,
                                            "initial_tcp_residual": initial_residual, "termination_reason": termination,
                                            "offset_limit_reached": limit_reached,
                                            "max_offset_rad": max_offset})
        if start_offset is not None:
            residual["arrival_correction"]["start_goal_offset_rad"] = [round(float(v), 5) for v in started_from]
        if (start_offset is not None or stop_unanswered or stop_within_tolerance or self._breakaway is not None
                or feedforward is not None):
            residual["arrival_correction"]["final_goal_offset_rad"] = [round(float(v), 5) for v in offset]
        return ok, fail_reason, residual

    def _fingertips(self, jaw_rad):
        """The two CAD fingertip points in the TCP frame (m) at a jaw angle (rad), for ``path_facts``: the fixed jaw's
        inner tip point at the jaw's furthest protrusion beyond the TCP along the approach axis, and the moving jaw's
        inner tip point (``so101_gripper_geometry.fingertip_reference``; CAD, not measured on this unit)."""
        from ni_rua.robot.model.so101_gripper_geometry import derive_gripper_geometry, fingertip_reference
        if self._fixed_jaw_protrusion_m is None:  # CAD constant; deriving it takes about 10 ms
            self._fixed_jaw_protrusion_m = float(derive_gripper_geometry()["protrusion_beyond_tcp_m"]["fixed_jaw"])
        tips = fingertip_reference(float(jaw_rad))
        fixed = list(tips["fixed_tip_m"])
        fixed[2] = max(fixed[2], self._fixed_jaw_protrusion_m)
        return [fixed, list(tips["moving_tip_m"])]

    def _in_hand(self, sample):
        """True when the jaw was last commanded to close (goal_normalized below the shared closing bound 0.5) and its
        measured CAD fingertip opening is above an empty close by more than the ``non_empty`` preset margin, i.e.
        something is between the fingers; False when it is not; None when the empty-close opening is not measured."""
        from ni_rua.robot.model.so101_gripper_geometry import opening_width_m
        gripper = self.embodiment.gripper
        if gripper.empty_close_aperture_m is None:
            return None
        margin = (gripper.preset_margins_m.get("non_empty") or (None,))[0]
        if margin is None:
            return None
        return bool(self.jaw_command < CLOSING_JAW_GOAL_BELOW
                    and opening_width_m(sample["gripper_rad"]) > gripper.empty_close_aperture_m + margin)

    def _mirror_failure(self, receipt, result):
        """Move a failed move's measured residual from its result to receipt level (``endpoint_residual_m``, the
        ARX-X5 receipt field: dx_m / dy_m / dz_m / position_error_m in m, orientation_error_rad in rad, plus the split
        components when judged) and add ``in_hand_tracking_error`` = True when the closed jaw still holds something
        (``_in_hand``). Condensed history keeps receipt-level fields only (``lean_projection``)."""
        residual = result.pop("endpoint_residual_m", None)
        if residual is None:
            return
        receipt["endpoint_residual_m"] = residual
        try:
            if self._in_hand(self.read()):
                receipt["in_hand_tracking_error"] = True
        except DriverFault:
            pass  # unknown: the field stays absent

    def _max_commands(self, action):
        """Largest command count of one during-motion jaw schedule: the request's ``timeout_s`` (s) x
        ``motion.command_rate_hz`` (Hz); a schedule needing more could not finish before the deadline."""
        return int(float(action["timeout_s"]) * self.deployment["motion"]["command_rate_hz"])

    # ------------------------------------------------------------ execute ---
    def _gate(self):
        with self.state_lock:
            if self.relaxed:
                return "operator_relaxed_requires_torque_on"
            if self.observe_required:
                return "fresh_observation_required_after_torque_on"
        return None

    def _rejected(self, reason, **extra):
        return {"status": "rejected", "reason": reason, "physics_advanced": False, "executed_steps": 0,
                "executed_native_steps": 0, "completed_native_steps": 0, "executed_control_steps": 0,
                "terminal_observation": None, "planned_trace": [], "dispatched_trace": [], "actual_trace": [],
                "execution_interval": {"start_sim_step": self.control_step, "end_sim_step": self.control_step,
                                       "step_domain": STEP_DOMAIN}, **extra}

    def execute(self, action, progress, cancellation):
        # Begin only on the first actual dispatch, after admission/planning. Always
        # close on unexpected exceptions too; no encoder work runs on this thread.
        self._video_action_id = action.get("action_id", "unidentified_action")
        try:
            return self._execute(action, progress, cancellation)
        finally:
            self.motion_video.end()
            self._video_action_id = None

    def _execute(self, action, progress, cancellation):
        gate = self._gate()
        if gate:
            return self._rejected(gate)
        try:
            contract.validate_action(action, self.deployment)
            if action.get("type") == "BatchAction":
                validate_batch_carry_slip_guard(action, **contract.guard_parameters(self.deployment))
        except ValueError as error:
            return self._rejected(str(error))
        if action.get("type") == "BatchAction":
            return self._execute_batch(action, progress, cancellation)
        timeout = float(action["timeout_s"])
        run = self._new_run(cancellation, timeout, action.get("action_id"), boundary=True)
        path = action["path"]
        receipt = {"status": "failed", "reason": None, "waypoint_results": [], "stop_confirmation": None,
                   "effective_limits": self.effective_limits(action.get("speed_scale", 1.), action.get("gripper_speed_scale", 1.))}
        planned = []
        if path["kind"] == "backend_free":
            tolerances = self._tolerances(path)
            receipt["effective_tolerances"] = self._effective_tolerances(path)
            try:
                seed = self.read()["q_rad"]
            except DriverFault as error:
                return self._rejected(f"backend_error: {error}")
            for index, point in enumerate(path["waypoints"]):
                q, ik = self._ik(point[self.arm], seed, tolerances, self._split(path))
                if not ik["ok"]:
                    return self._rejected("planning_failed", status="failed", failed_waypoint_index=index,
                                          ik_result=ik, effective_tolerances=receipt["effective_tolerances"])
                planned.append({"waypoint_index": index, "q_rad": q, "ik": ik})
                seed = q
        try:
            _, start_step = self._start_sample(run)
        except DriverFault as error:
            return self._rejected(f"backend_error: {error}")
        self._start_frame(receipt)
        self.active_action_id = action.get("action_id")
        self.motion_phase = "moving"
        progress({"phase": "started", "sim_step": self.control_step})
        try:
            if "hold_native_steps" in action:
                period = 1 / self.deployment["motion"]["command_rate_hz"]
                for _ in range(action["hold_native_steps"]):
                    reason = self._interrupt(run, "wait")
                    if reason:
                        receipt["reason"] = reason
                        break
                    if not self._send(self.arm_goal, self.jaw_goal, run["generation"], run=run, kind="wait_hold"):
                        receipt["reason"] = self.stop_reason or "operator_hold"
                        break
                    run["commands"] += 1
                    self._tick(run, period)
                else:
                    receipt.update(status="completed", reason=None)
                receipt["wait"] = {"requested_steps": action["hold_native_steps"], "executed_steps": run["commands"],
                                   "measured_wall_clock_s": round(self.clock.now() - run["t0"], 4),
                                   "semantics": "real-time hold of last commanded joint goals"}
            else:
                ok = True
                ramp = next((event for event in action.get("gripper_events", []) if event["phase"] == "during_motion"), None)
                for item in planned:
                    progress({"phase": "executing", "waypoint_index": item["waypoint_index"], "sim_step": self.control_step,
                              "executed_steps": run["commands"]})
                    # A during-motion jaw ramp runs over the final waypoint's segment (ARX-X5 timing basis).
                    jaw = (None if ramp is None or item is not planned[-1] else
                           dict(event=ramp, gripper_speed_scale=action.get("gripper_speed_scale", 1.),
                                max_commands=self._max_commands(action)))
                    ok, reason, fields = self._move(run, item["q_rad"], path["waypoints"][item["waypoint_index"]][self.arm],
                                                    path, speed_scale=action.get("speed_scale", 1.), jaw=jaw)
                    # Waypoint results keep the stop signal's keys at their top level (single-request receipt shape).
                    result = {"waypoint_index": item["waypoint_index"], "ik": item["ik"], **fields.pop("tracking_stop", {}),
                              **fields}
                    receipt["waypoint_results"].append(result)
                    if not ok:
                        receipt.update(reason=reason, failed_waypoint_index=item["waypoint_index"])
                        if "position_error_m" in fields:
                            receipt.update(position_error_m=fields["position_error_m"],
                                           orientation_error_rad=fields["orientation_error_rad"])
                        if "contact_block_summary" in fields:
                            receipt["contact_block_summary"] = fields["contact_block_summary"]
                        if "hold_confirmation" in fields:
                            receipt["stop_confirmation"] = fields["hold_confirmation"]
                        self._mirror_failure(receipt, result)
                        break
                    receipt.update(position_error_m=result["position_error_m"], orientation_error_rad=result["orientation_error_rad"])
                if ok:
                    for event in action.get("gripper_events", []):
                        if event["phase"] == "during_motion":
                            ok, reason, jaw = self._ramp_jaw(run, event, action.get("gripper_speed_scale", 1.),
                                                             self._squeeze())
                        else:
                            ok, reason, jaw = self._gripper(run, event["goal_normalized"],
                                                            action.get("gripper_speed_scale", 1.), self._squeeze())
                        receipt["gripper_execution"] = {self.arm: jaw}
                        if not ok:
                            receipt["reason"] = reason
                    if ok:
                        receipt.update(status="completed", reason=None)
            if receipt["status"] != "completed":
                self._finish_interrupted(receipt, run)
        except DriverFault as error:
            receipt.update(status="unknown" if run["commands"] or self.control_step != start_step else "failed",
                           reason=f"backend_error: {error}")
            receipt["stop_confirmation"] = self._confirm_stop()
            if receipt["status"] == "failed":
                receipt["physics_advanced"] = False
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            try:
                stop = self._stop()
            except Exception as stop_err:
                stop = {"confirmed": False, "state": "unknown", "reason": f"{type(stop_err).__name__}: {stop_err}"}
            moved = bool(run["commands"] or self.control_step != start_step)
            uncertain = moved or not bool(stop.get("confirmed"))
            receipt.update(
                status="unknown" if uncertain else "failed",
                reason=f"internal_runtime_error: {type(error).__name__}: {error}",
                stop_confirmation=stop,
            )
            if uncertain:
                receipt["execution_uncertain"] = True
            else:
                receipt["physics_advanced"] = False
        finally:
            self.active_action_id = None
            if self.motion_phase == "moving":
                self.motion_phase = "idle"
        return self._close_receipt(receipt, run, start_step, planned)

    def _finish_interrupted(self, receipt, run):
        reason = receipt.get("reason")
        if reason in ("cancelled", "operator_hold") or (reason or "").startswith("operator_"):
            receipt["status"] = "stopped"
        if reason == "operator_relax":
            receipt["stop_confirmation"] = {"confirmed": False, "state": "torque_off", "reason": "operator_relax"}
        elif receipt.get("stop_confirmation") is None:
            # Harness action runs keep the hold commands and settling reads in their own traces;
            # operator/calibration moves keep their dispatched list to the commands they planned.
            receipt["stop_confirmation"] = self._confirm_stop(run if run.get("boundary") is not None else None)

    def _close_receipt(self, receipt, run, start_step, planned):
        self.motion_video.end()
        self.active_action_id = None
        if self.motion_phase in ("moving", "stopping"):
            self.motion_phase = "idle"
        receipt.setdefault("physics_advanced", self.control_step != start_step)
        receipt.update(executed_steps=run["commands"], executed_native_steps=self.control_step - start_step,
                       completed_native_steps=self.control_step - start_step, executed_control_steps=run["commands"],
                       wall_clock_duration_s=round(self.clock.now() - run["t0"], 4),
                       execution_interval={"start_sim_step": start_step, "end_sim_step": self.control_step,
                                           "step_domain": STEP_DOMAIN})
        evidence = self._trajectory_evidence(run, start_step, planned)
        if receipt.get("status") == "unknown":
            # A failed write may still have reached the servo: no row records it, so rows and samples cannot align.
            evidence["trace_sampling"].update(actual_boundary_samples=False,
                                              incomplete_reason="a command write failed; it may have partially executed")
        receipt.update(evidence)
        if receipt.get("stop_confirmation") is None:
            receipt.pop("stop_confirmation", None)
        elif receipt["stop_confirmation"].get("confirmed") is False:
            receipt["stop_state"] = "unknown"
        if receipt["status"] == "unknown":
            receipt["execution_uncertain"] = True
        try:
            receipt["terminal_observation"] = self.observe()
        except Exception as error:
            receipt.update(terminal_observation=None, terminal_observation_error=f"{type(error).__name__}: {error}")
        return receipt

    def _trajectory_evidence(self, run, start_step, planned):
        """Receipt evidence read by Harness trajectory feedback (popped into actor artifacts).

        ``before_state``: measured robot_state before the first command. ``actual_trace``: one
        ``{sim_step, t_s, state}`` per control step (``_boundary_sample``), plus a final read when
        the last command had none. ``planned_trace``: the commanded rows, one per dispatched
        command (5 arm goals + commanded jaw angle, rad, in ``joint_names`` order), with the cached
        IK targets. ``trace_sampling.actual_boundary_samples`` is true only when every control step
        from the start to the end of the action has its measured sample.
        """
        boundary, sampling = run.get("boundary"), {"actual_every_native_actions": 1}
        if boundary is not None and (not boundary or boundary[-1]["sim_step"] != self.control_step):
            try:
                with self.bus_lock:
                    sample = self._read()
                    step = self.control_step
                self._boundary_sample(run, sample, step)
            except Exception as error:
                sampling["terminal_state_sample_error"] = str(error)
        boundary = boundary if boundary is not None else []
        steps = [entry["sim_step"] for entry in boundary]
        # Complete only if this run measured every step AND dispatched every command in the interval:
        # a foreign command (an operator hold written by another thread) shifts plan/measurement rows.
        sampling.update(actual_boundary_samples=(steps == list(range(start_step, self.control_step + 1))
                                                 and [entry["step"] for entry in run["dispatched"]]
                                                 == list(range(start_step + 1, self.control_step + 1))),
                        semantics="one measured robot_state per control step (dispatched joint-goal command): the last "
                                  "joint read before the next command, or the final read; t_s is host monotonic s since "
                                  "the action started")
        # The planned row of each command: the goal itself, or the offset-free row it stood for (q_commanded_rad).
        rows = [[*entry.get("q_commanded_rad", entry["q_goal_rad"]),
                 gripper_rad_from_normalized(self.deployment, entry["gripper_goal_normalized"])]
                for entry in run["dispatched"]]
        planned_trace = {"arm": self.arm, "joint_names": list(JOINT_NAMES), "q_rad": rows, "completed_native_steps": len(rows),
                         "row_kinds": [entry.get("kind", "motion") for entry in run["dispatched"]],
                         "ik_targets": [{"waypoint_index": item.get("waypoint_index"), "q_rad": item["q_rad"]} for item in planned],
                         "semantics": "commanded joint goals (rad), one row per dispatched control step including "
                                      "arrival corrections and holds; the gripper column is the commanded jaw angle"}
        if any("q_commanded_rad" in entry for entry in run["dispatched"]):
            # Motion compensation: what was sent on top of each planned row (carried sag offset + gravity
            # feedforward), so the plan is drawn where it was planned and the sent goals stay recoverable.
            planned_trace["goal_offset_rad"] = [
                [round(sent - row, 6) for sent, row in zip(entry["q_goal_rad"], entry.get("q_commanded_rad", entry["q_goal_rad"]))]
                for entry in run["dispatched"]]
            planned_trace["semantics"] += ("; goal_offset_rad: the motion-compensation offset (rad per arm joint) "
                                           "sent on top of each row, so the goal the servo received is row + offset")
        # The commanded rows are stored once (planned_trace.q_rad); dispatched_trace keeps the step of each row.
        dispatched_trace = {"scope": "commands this action dispatched; row k of planned_trace.q_rad was sent as step steps[k]",
                            "steps": [entry["step"] for entry in run["dispatched"]]}
        return {"before_state": boundary[0]["state"] if boundary and boundary[0]["sim_step"] == start_step else None,
                "planned_trace": planned_trace, "dispatched_trace": dispatched_trace, "actual_trace": boundary,
                "trace_sampling": sampling}

    # -------------------------------------------------------------- batch ---
    @staticmethod
    def _batch_digest(action):
        from ni_rua.core.contracts import digest
        stages = [{key: value for key, value in stage.items() if key != "action"} | (
            {"action": {key: value for key, value in stage["action"].items()
                        if key not in ("action_id", "based_on_observation_id")}} if "action" in stage else {})
                  for stage in action["stages"]]
        batch = {key: action[key] for key in ("checkpoint_after_gripper", "review_proxies",
                                              *contract.GRIP_GUARD_FIELDS, *CARRY_SLIP_GUARD_FIELDS)
                 if key in action}
        return digest({"stages": stages, "frame_id": action["frame_id"], "tcp_config_id": action["tcp_config_id"],
                       "limits_profile": action["limits_profile"], **batch})

    @staticmethod
    def _ordered(stages):
        """Single-arm order: list order respecting declared depends_on."""
        return order_stages(stages)

    def reach_batch(self, action):
        """robot_action ``mode=reach``: plan every move stage independently from the measured joints; no plan, no motion.

        ``action``: a compiled BatchAction of move stages only (the Harness compiles it like a preview). Each pose is
        solved by the deployment's IK solver (``_ik`` / ``ik_plan``) from the joints read now, not chained through the
        previous stage, and every stage is answered (no stop at the first unreachable one). Returns ``status`` solved,
        ``stage_count``, ``reachable_stage_count`` and ``stage_results[<stage id>]``: ``ok``, ``ik`` (the planner
        record with residuals in m / rad), ``tolerances`` (effective tolerances), ``reached`` (``position_m`` m,
        ``quaternion_xyzw`` and ``approach_axis``, the TCP +Z unit vector in so101_base, of the planned joints; of
        the closest candidate when not ok), and for reachable stages ``q_rad`` and ``joint_travel_rad`` (|target -
        measured| per arm joint, rad). Nothing is cached or commanded."""
        zero = {"physics_advanced": False, "executed_native_steps": 0}
        try:
            contract.validate_action(action, self.deployment)
            start = self.read()
        except (ValueError, DriverFault) as error:
            return {"status": "failed", "reason": str(error), **zero}
        stages = action["stages"]
        if any(stage.get("kind") != "move" for stage in stages):
            return {"status": "failed", "reason": "reach_accepts_move_stages_only", **zero}
        seed = [float(v) for v in start["q_rad"]]
        requests = [(stage["action"]["path"]["waypoints"][0][self.arm], self._tolerances(stage["action"]["path"]),
                     self._split(stage["action"]["path"])) for stage in stages]
        if self.ik_client is None:
            solved = [local_solve(self.kin, self.model, pose, seed, tolerances, self.deployment["ik"], self.limits)
                      for pose, tolerances, _ in requests]
        else:
            try:
                solved = curobo_solve(self.ik_client, self.kin, self.model, self.limits, requests, seed)
            except IKServiceError as error:
                return {"status": "failed", "reason": error.reason, **zero}
        results = {}
        for stage, (q, record) in zip(stages, solved):
            T = self.tcp(q)
            # The IK acceptance bounds only: a reach query never moves, so the arrival, window, stall-guard and
            # arrival-correction facts of _effective_tolerances do not apply.
            tolerances = {key: value for key, value in self._effective_tolerances(stage["action"]["path"]).items()
                          if key not in REACH_EXCLUDED_TOLERANCES}
            tolerances["checked_on"] = "FK of the planned joints (IK acceptance; nothing moves)"
            entry = {"ok": bool(record["ok"]), "ik": record, "tolerances": tolerances,
                     "reached": {"position_m": [round(float(v), 4) for v in T[:3, 3]],
                                 "quaternion_xyzw": [round(float(v), 5) for v in matrix_to_quaternion(T[:3, :3])],
                                 "approach_axis": [round(float(v), 4) for v in T[:3, 2]]}}
            if record["ok"]:
                entry.update(q_rad=[round(float(v), 4) for v in q],
                             joint_travel_rad=[round(abs(float(a) - b), 4) for a, b in zip(q, seed)])
            results[stage["id"]] = entry
        return {"status": "solved", "stage_count": len(stages),
                "reachable_stage_count": sum(1 for entry in results.values() if entry["ok"]),
                "stage_results": results, **zero}

    def preview_batch(self, action):
        """Plan a batch without moving: IK per move stage, the path each move will take, and its collision check.

        Per move stage ``stage_plans[<id>]`` carries ``q_rad``, ``ik``, ``tolerances``, ``joint_travel_rad`` and
        ``path``: FK facts of the joint interpolation the worker will stream (``ik_plan.path_facts`` over
        ``joint_interpolation``, the rows of ``_segment``, including the subdivision of a during-motion jaw ramp):
        ``samples``, ``min_tcp_z_m``, ``min_fingertip_z_m`` (CAD fingertips, ``_fingertips``), ``lowest_at_fraction``,
        ``max_line_deviation_m``, ``dip_below_endpoints_m`` (m in so101_base) and, when the path check found one,
        ``collision`` (``_path_check``). Execution starts each move from the measured joints, not the planned ones;
        its stage result reports the difference (``path_start_deviation_rad``).

        With ``ik_service`` the request's ``review_proxies`` (obstacle boxes, held-object box) form the collision
        world (``ik_plan.collision_world``): IK prefers solutions clear of it (``ik.obstacle_free``) and the top-level
        ``path_check`` summarises the sampled path check. Both are advisory: a colliding plan is still ``planned``.
        The held box counts as attached up to and including the first stage that opens the jaw (a jaw goal above the
        goal commanded before it); later stages are planned and checked without it.

        ``trajectory_data.samples``: up to ``PATH_OVERLAY_SAMPLES`` commands of every move (the last included), so the
        overlay polyline follows the interpolated path instead of joining the stage targets."""
        zero = {"physics_advanced": False, "executed_native_steps": 0}
        try:
            contract.validate_action(action, self.deployment)
            validate_batch_carry_slip_guard(action, **contract.guard_parameters(self.deployment))
            stages = self._ordered(action["stages"])
            start = self.read()
        except (ValueError, DriverFault) as error:
            return {"status": "failed", "reason": str(error), **zero}
        rate, motion = self.deployment["motion"]["command_rate_hz"], self.deployment["motion"]
        # jaw: the goal the servo holds (a squeezed hold goal after a squeezed close), where the next jaw ramp starts;
        # commanded: the last commanded goal, which checkpoint_after_gripper compares a stage's goal with.
        q, jaw, commanded, cursor = list(start["q_rad"]), self.jaw_goal, self.jaw_command, 0
        records, costs, checkpoints, samples = {}, {}, [], []
        guard_parameters = contract.guard_parameters(self.deployment)
        empty_close = contract.grip_guard_gripper(self.deployment)["empty_close_aperture_m"]
        previous_goals = {}
        # Collision world of this request (cuRobo service only). jaw_rad: the measured jaw until the plan commands it.
        world = (collision_world(action.get("review_proxies"), self.arm, self.tcp(start["q_rad"]))
                 if self.ik_client is not None else None)
        jaw_rad, attached, released_at = float(start["gripper_rad"]), bool(world and world["attached_spheres"]), None
        segments = []

        def stage_world():
            """The collision world as this stage sees it: without the held spheres once the jaw has opened."""
            if world is None:
                return None
            return dict(obstacles=world["obstacles"], attached_spheres=world["attached_spheres"] if attached else [],
                        jaw_rad=jaw_rad)

        def command_jaw(identifier, goal):
            """Jaw state after a jaw command: commanded angle; an opening command releases the held object."""
            nonlocal jaw, commanded, jaw_rad, attached, released_at
            if attached and float(goal) > float(jaw):
                attached, released_at = False, identifier
            jaw, jaw_rad = goal, gripper_rad_from_normalized(self.deployment, goal)
            commanded = goal

        def jaw_squeeze(identifier, stage):
            """The holding squeeze (rad) this jaw stage will use, shown in its stage plan (absent when off)."""
            squeeze_rad = self._squeeze(stage)
            if squeeze_rad is not None:
                records[identifier]["hold_squeeze_rad"] = squeeze_rad

        def jaw_guard(stage, goal):
            """Resolved grip guard of one jaw command (gripper stage or move with a during-motion jaw), or None."""
            guard = stage_grip_guard(stage, action, float(goal), **guard_parameters)
            if guard is None:
                return None
            guard = dict(guard, empty_close_aperture_m=empty_close, source=(
                "stage" if any(stage.get(field) is not None for field in (*contract.GRIP_GUARD_FIELDS, *CARRY_SLIP_GUARD_FIELDS))
                else "batch_default"),
                detects_empty_close=contract.detects_empty_close(guard["min_aperture_m"], self.deployment))
            if float(goal) < CLOSING_JAW_GOAL_BELOW:
                guard["post_grasp_recheck"] = (
                    {"available": True} if contract.post_grasp_recheck_available(self.deployment)
                    else {"available": False, "reason": contract.post_grasp_recheck_unavailable_reason(self.deployment)})
            return guard

        for stage in stages:
            identifier, kind = stage["id"], stage["kind"]
            begin = cursor
            previous_goals[identifier] = commanded
            if kind == "observe":
                checkpoints.append({"stage_id": identifier, "cursor": cursor, "reason": stage["reason"]})
                records[identifier] = {"kind": kind, "arm": stage.get("arm")}
            elif kind == "move":
                leaf = stage["action"]
                pose = leaf["path"]["waypoints"][0][self.arm]
                target, ik = self._ik(pose, q, self._tolerances(leaf["path"]), self._split(leaf["path"]), stage_world())
                if not ik["ok"]:
                    return {"status": "failed", "reason": "planning_failed", "failed_stage_id": identifier, "failed_arm": self.arm,
                            "failure_feedback": {"ik": ik, "effective_tolerances": self._effective_tolerances(leaf["path"])},
                            **zero}
                speed = motion["max_joint_speed_rad_s"] * self._cap() * leaf.get("speed_scale", 1.)
                accel = motion.get("max_acceleration_rad_s2")
                jerk = motion.get("max_jerk_rad_s3")
                steps = interpolation_steps(
                    q, target, speed, rate, max_acceleration_rad_s2=accel, max_jerk_rad_s3=jerk
                )
                rows = joint_interpolation(
                    q,
                    target,
                    steps,
                    speed_rad_s=speed,
                    rate_hz=rate,
                    max_acceleration_rad_s2=accel,
                    max_jerk_rad_s3=jerk,
                )
                # R8: the joint travel of this straight joint interpolation (rad per arm joint); reported, never limited.
                records[identifier] = {"kind": kind, "q_rad": target, "pose": pose, "ik": ik, "start_q": list(q),
                                       "tolerances": self._effective_tolerances(leaf["path"]),
                                       "joint_travel_rad": [round(abs(float(end) - float(begin_q)), 4)
                                                            for end, begin_q in zip(target, q)]}
                start_jaw_rad, was_attached = jaw_rad, attached
                row_jaws = [jaw_rad] * steps
                event = next(iter(leaf.get("gripper_events") or []), None)  # validated: one during_motion event or none
                if event is not None:
                    # R5: the during-motion jaw ramp; its slowdown is part of this stage's command cost.
                    schedule = self._jaw_schedule(rows, np.asarray(q, dtype=float), jaw, event,
                                                  leaf.get("gripper_speed_scale", 1.), self._max_commands(action))
                    if schedule is None:
                        return {"status": "failed", "reason": "gripper_motion_schedule_exceeds_timeout",
                                "failed_stage_id": identifier, "failed_arm": self.arm,
                                "failure_feedback": {"max_commands": self._max_commands(action),
                                                     "max_jaw_step_normalized": round(self._jaw_step_normalized(
                                                         leaf.get("gripper_speed_scale", 1.)), 6)}, **zero}
                    rows = list(schedule[0])
                    steps = len(rows)
                    row_jaws = [gripper_rad_from_normalized(self.deployment, float(value)) for value in schedule[1]]
                    goal = event["goal_normalized"]
                    records[identifier].update(gripper_motion=schedule[2], goal_normalized=goal)
                    guard = jaw_guard(stage, goal)
                    if guard is not None:
                        records[identifier]["grip_guard"] = guard
                    jaw_squeeze(identifier, stage)
                    command_jaw(identifier, goal)
                # The path this move takes: FK of every commanded row (the fingertip points follow the jaw).
                tips = ([self._fingertips(value) for value in (start_jaw_rad, *row_jaws)] if event is not None
                        else self._fingertips(jaw_rad))
                records[identifier]["path"] = path_facts(self.kin, self.model, q, rows, tips)
                # Checked segments: cut at the jaw ramp's fractions, so the checked jaw follows the streamed schedule.
                ramp = (event["start_fraction"], event["end_fraction"]) if event is not None else None
                segments.extend((identifier, low, high, segment) for low, high, segment in path_check_segments(
                    q, target, start_jaw_rad, jaw_rad, was_attached, ramp))
                picked = sorted({int(round(value)) for value in np.linspace(0, steps - 1, min(steps, PATH_OVERLAY_SAMPLES))}
                                | {steps - 1})
                for index in picked:
                    row = [float(v) for v in rows[index]]
                    links = self.kin.fk_all(self.model, row, gripper_rad=row_jaws[index])
                    samples.append({
                        "stage_id": identifier,
                        "q_rad": row,
                        "tcp_position_m": np.asarray(links[self.kin.TCP_LINK])[:3, 3].tolist(),
                        "native_step": int(cursor + index + 1),
                        "execution_state": "not_executed",
                        "arms": {
                            self.arm: {
                                "q_rad": [*row, row_jaws[index]],
                                "phase": "move",
                                "link_positions_m": {k: np.asarray(v)[:3, 3].tolist() for k, v in links.items()},
                            }
                        },
                    })
                cursor += steps
                q = target
            else:
                goal = leaf_goal = stage["action"]["gripper_events"][0]["goal_normalized"]
                step = self._jaw_step_normalized(stage["action"].get("gripper_speed_scale", 1.))
                cursor += max(1, int(math.ceil(abs(goal - jaw) / step)))
                records[identifier] = {"kind": kind, "goal_normalized": leaf_goal}
                guard = jaw_guard(stage, goal)
                if guard is not None:
                    records[identifier]["grip_guard"] = guard
                jaw_squeeze(identifier, stage)
                start_jaw_rad, was_attached = jaw_rad, attached
                command_jaw(identifier, goal)
                segments.extend((identifier, low, high, segment) for low, high, segment in path_check_segments(
                    q, q, start_jaw_rad, jaw_rad, was_attached))
            costs[identifier] = {"arm": self.arm, "kind": kind, "start_native_step": begin, "end_native_step": cursor,
                                 "native_steps": cursor - begin}
        path_check = self._path_check(segments, world, records, released_at)
        jaw_pauses = self._jaw_pauses(action, stages, records, costs, checkpoints, previous_goals)
        plan_id = "so101-batch-" + uuid.uuid4().hex
        while len(self._plans) >= self.deployment["plan"]["cache_size"]:
            self._plans.pop(next(iter(self._plans)))
        self._plans[plan_id] = {"digest": self._batch_digest(action), "created_at": self.clock.now(), "status": "planned",
                                "order": [stage["id"] for stage in stages], "records": records, "index": 0, "cursor": 0,
                                "revision": 0, "start_q": list(start["q_rad"]), "completed": [],
                                "jaw_pauses": jaw_pauses, "armed_grip_guard": {}}
        return {"status": "planned", "plan_id": plan_id, **zero, "planned_native_steps": cursor,
                "nominal_duration_s": round(cursor / rate, 3), "stage_costs": costs,
                "stage_plans": {key: {k: v for k, v in value.items() if k in ("q_rad", "ik", "tolerances", "joint_travel_rad",
                                                                           "goal_normalized", "grip_guard", "gripper_motion",
                                                                           "path", "hold_squeeze_rad")}
                                for key, value in records.items() if value["kind"] != "observe"},
                **({"path_check": path_check} if path_check is not None else {}),
                "checkpoints": checkpoints, "effective_limits": self.effective_limits(),
                "trajectory_data": {"source_kind": "candidate_command_trajectory_not_measurement",
                                    "samples": samples, "frame_id": SO101.frame_id,
                                    "robot_model": self.model,
                                    "T_world_bases": {self.arm: np.eye(4).tolist()},
                                    "camera_start_q_rad": {self.arm: [*start["q_rad"], start["gripper_rad"]]},
                                    "execution_binding": "cached IK joint targets; interpolation regenerated from the measured start"},
                "execution_semantics": "cached IK joint targets per stage; joint interpolation regenerated from the measured "
                                       "start (bounded by plan.max_start_drift_rad); no IK re-solve; stop on failed arrival",
                "plan_lifetime_s": self.deployment["plan"]["max_age_s"]}

    def _path_check(self, segments, world, records, released_at):
        """The preview's path collision review (cuRobo service), or None without ``ik_service`` or without a move or
        gripper stage.

        ``segments``: ``[(stage id, from_fraction, to_fraction, segment)]`` of the planned moves and standalone gripper
        stages (``ik_plan.path_check_segments``: one per move, or up to three cut at a during-motion jaw ramp's fractions
        so the checked jaw matches the streamed one; one per gripper stage, the arm still and the jaw sweeping from its
        angle before the stage to the commanded goal, a held object still attached during the sweep that opens the jaw;
        ``segment`` in ``CuroboClient.check_path`` form). Every segment's
        joint-space line is sampled at most ``ik_service.path_check_step_rad`` apart and checked against
        self-collision, the table box, the world's obstacle boxes and, while attached, the held-object spheres.
        Returns ``status`` (``clear``: no sample collides; ``collisions``; ``unavailable``: the check did not run, with
        ``reason``), ``checked_path`` (always ``commanded``: the planned joint line, not where the loaded arm will
        really be; execution reports the measured difference as ``stage_results[<move>].transit``),
        ``colliding_stage_ids``, ``max_joint_step_rad`` (rad between samples), ``obstacle_ids`` and, with a
        held box, ``held_object`` (``spheres``, ``sphere_radius_m`` (m), ``attached_through_stage_id``: the stage that
        opens the jaw, or null when none does). A colliding stage also gets ``records[<id>]["path"]["collision"]``:
        ``fraction`` of the stage (0 = start, 1 = target), ``tcp_position_m`` (m) and ``jaw_rad`` (jaw angle, rad; when
        the service reports it) of the first colliding sample,
        ``colliding_samples`` / ``samples`` (summed over the stage's segments), ``world`` ([{link, penetration_m,
        obstacles}]; link ``attached_object`` = the held box) and ``self`` ([{links, penetration_m}]). Advisory facts of a
        sphere model: never blocks."""
        if self.ik_client is None or not segments:
            return None
        review = {"status": "clear", "colliding_stage_ids": [], "checked_path": "commanded",
                  "max_joint_step_rad": self.ik_client.path_check_step_rad,
                  "obstacle_ids": [box["id"] for box in world["obstacles"]] if world else []}
        if world and world["held_object"] is not None:
            review["held_object"] = dict(world["held_object"], attached_through_stage_id=released_at)
        try:
            answer = self.ik_client.check_path([segment for *_, segment in segments], world)
        except IKServiceError as error:
            review.update(status="unavailable", reason=error.reason)
            return review
        moves = {}  # stage id -> its segments' (from, to, result), in move order
        for (identifier, low, high, _), result in zip(segments, answer["segments"]):
            moves.setdefault(identifier, []).append((low, high, result))
        for identifier, pieces in moves.items():
            hit = next(((low, high, result) for low, high, result in pieces if result["first_collision"] is not None),
                        None)
            if hit is None:
                continue
            low, high, result = hit
            first = result["first_collision"]
            review["colliding_stage_ids"].append(identifier)
            fraction = first.get("fraction")
            collision = {"fraction": (round(low + (high - low) * float(fraction), 4)
                                      if isinstance(fraction, (int, float)) and len(pieces) > 1 else fraction),
                         "colliding_samples": sum(item["colliding_samples"] for *_, item in pieces),
                         "samples": sum(item["samples"] for *_, item in pieces),
                         "world": first.get("world") or [], "self": first.get("self") or []}
            q = first.get("q_rad")
            if isinstance(q, list) and len(q) == 5 and _finite(q):
                collision["tcp_position_m"] = [round(float(v), 4) for v in self.tcp(q)[:3, 3]]
            if isinstance(first.get("jaw_rad"), (int, float)):
                collision["jaw_rad"] = first["jaw_rad"]
            records[identifier].setdefault("path", {})["collision"] = collision
        if review["colliding_stage_ids"]:
            review["status"] = "collisions"
        return review

    @staticmethod
    def _jaw_pauses(action, stages, records, costs, checkpoints, previous_goals):
        """checkpoint_after_gripper on SO-101: delegates to shared ``batch_scheduler.jaw_pauses``."""
        return jaw_pauses(action, stages, records, costs, checkpoints, previous_goals)

    def _pause_batch(self, run, plan, receipt, stage_ids, reasons, checkpoint_reason=None, checkpoint_reason_code=None):
        """Pause a batch at a stage boundary with a measured hold; resume with the same plan_id."""
        stop = self.hold(run)
        if stop.get("confirmed") is True:
            plan["status"] = "awaiting_observation"
            plan["start_q"] = self.read()["q_rad"]
            plan["created_at"] = self.clock.now()
        elif str(stop.get("reason") or "").startswith("DriverFault:"):
            err = DriverFault(str(stop["reason"]).removeprefix("DriverFault: ").strip())
            err.stop_confirmation = stop
            raise err
        else:
            plan["status"] = "failed"
        receipt.update(
            pause_receipt(
                stage_ids,
                reasons,
                stop,
                checkpoint_reason=checkpoint_reason,
                checkpoint_reason_code=checkpoint_reason_code,
            )
        )

    def _grip_facts(self, begin, end):
        """Shared ``grip_facts`` from two measured samples: normalized jaw and CAD fingertip gap (m)."""
        from ni_rua.robot.model.so101_gripper_geometry import opening_width_m
        return grip_facts_from_measurements(begin["gripper_normalized"], end["gripper_normalized"],
                                            opening_width_m(begin["gripper_rad"]), opening_width_m(end["gripper_rad"]))

    def _execute_batch(self, action, progress, cancellation):
        plan_id = action.get("batch_plan_id")
        plan = self._plans.get(plan_id)
        scheduler = BatchScheduler(self, plans=self._plans)
        try:
            scheduler.admit_cached_plan(
                plan,
                action,
                digest_value=self._batch_digest(action),
                max_age_s=self.deployment["plan"]["max_age_s"],
                now=self.clock.now(),
            )
        except ValueError as error:
            return self._rejected(str(error))
        run = self._new_run(cancellation, float(action["timeout_s"]), action.get("action_id"), boundary=True)
        try:
            # Read before touching the plan: a read fault leaves it executable (nothing moved).
            measured, start_step = self._start_sample(run)
        except DriverFault as error:
            return self._rejected(f"backend_error: {error}")
        drift = float(np.abs(np.subtract(measured["q_rad"], plan["start_q"])).max())
        if drift > self.deployment["plan"]["max_start_drift_rad"]:
            reference = "checkpoint" if plan["status"] == "awaiting_observation" else "preview"
            plan["status"] = "invalidated"
            reference_xyz = self.tcp(plan["start_q"])[:3, 3]
            measured_xyz = self.tcp(measured["q_rad"])[:3, 3]
            return self._rejected("batch_plan_stale_state_requires_new_preview", start_drift_rad=round(drift, 5),
                                  max_start_drift_rad=self.deployment["plan"]["max_start_drift_rad"],
                                  tcp_drift={"frame_id": contract.FRAME_ID, "reference": reference,
                                             "reference_position_m": reference_xyz.tolist(),
                                             "measured_position_m": measured_xyz.tolist(),
                                             "delta_position_m": (measured_xyz - reference_xyz).tolist()})
        plan["status"], plan["revision"] = "executing", plan["revision"] + 1
        guard_parameters = contract.guard_parameters(self.deployment)
        receipt = {"status": "failed", "reason": None, "stage_results": {}, "effective_limits": self.effective_limits()}
        self._start_frame(receipt)
        self.active_action_id, self.motion_phase = action.get("action_id"), "moving"
        progress({"phase": "started", "sim_step": self.control_step})

        def _run_stage(identifier, stage, record, result, carry_monitor):
            begin = self.read()
            if record["kind"] == "move":
                leaf = stage["action"]
                ramp = next(iter(leaf.get("gripper_events") or []), None)
                jaw = (None if ramp is None else dict(event=ramp, gripper_speed_scale=leaf.get("gripper_speed_scale", 1.),
                                                      max_commands=self._max_commands(action)))
                ok, reason, fields = self._move(run, record["q_rad"], record["pose"], leaf["path"],
                                                speed_scale=leaf.get("speed_scale", 1.), jaw=jaw, stage_id=identifier,
                                                planned_start_q=record.get("start_q"),
                                                carry_slip_monitor=carry_monitor)
                result.update(ik=record["ik"], effective_tolerances=record["tolerances"], **fields)
                if ok and ramp is not None:
                    ok, reason, result["gripper_execution"] = self._ramp_jaw(
                        run, ramp, leaf.get("gripper_speed_scale", 1.), self._squeeze(stage))
            else:
                ok, reason, jaw = self._gripper(run, record["goal_normalized"],
                                                stage["action"].get("gripper_speed_scale", 1.), self._squeeze(stage))
                result["gripper_execution"] = jaw
            end = self.read() if ok else None
            return ok, reason, begin, end

        def _after_facts(_identifier, record, result, end_sample):
            if record.get("goal_normalized") is None and self.jaw_hold is not None:
                result["grip_hold"] = self._grip_hold(end_sample)

        def _adapt_grip_eval(_identifier, record, result, facts, remaining, evaluation, triggered):
            preset = (record.get("grip_guard") or {}).get("grip_guard_preset")
            evaluation = dict(evaluation, **({"grip_guard_preset": preset} if preset else {}))
            if record.get("goal_normalized") is not None:
                settled = result["gripper_execution"].get("jaw_settled")
                evaluation["jaw_settled"] = settled
                closed = facts["jaw_delta"] is not None and facts["jaw_delta"] < JAW_NOT_CLOSED_DELTA
                if settled is False and closed and remaining:
                    plan["armed_grip_guard"].pop(self.arm, None)
                    triggered = True
                    evaluation.update(triggered=True, reason="jaw_not_settled", message=(
                        "Grip guard paused (jaw_not_settled): the jaw was still moving when gripper.settle_timeout_s="
                        f"{self.deployment['gripper']['settle_timeout_s']} s expired; measured_aperture_m="
                        f"{evaluation.get('measured_aperture_m')} m is not a settled reading."))
            return evaluation, triggered

        def _on_stage_failed(identifier, reason, result):
            receipt.update(reason=reason, failed_stage_id=identifier, failed_arm=self.arm)
            if "contact_block_summary" in result:
                receipt["contact_block_summary"] = result["contact_block_summary"]
            if "hold_confirmation" in result:
                receipt["stop_confirmation"] = result["hold_confirmation"]
            self._mirror_failure(receipt, result)

        def _on_carry_slip_hold():
            stop = self.hold(run)

            def _sync():
                if stop.get("confirmed") is True:
                    plan["start_q"] = self.read()["q_rad"]
                    plan["created_at"] = self.clock.now()
                elif str(stop.get("reason") or "").startswith("DriverFault:"):
                    err = DriverFault(str(stop["reason"]).removeprefix("DriverFault: ").strip())
                    err.stop_confirmation = stop
                    raise err
                else:
                    plan["status"] = "failed"

            return stop, _sync

        try:
            scheduler.run_stage_plan(
                plan_id=plan_id,
                plan=plan,
                action=action,
                receipt=receipt,
                default_arm=self.arm,
                failed_reasons=("tracking_error", "not_settled", "gripper_motion_schedule_exceeds_timeout"),
                on_stage_progress=lambda sid: progress({"phase": "executing", "sim_step": self.control_step,
                                                        "executed_steps": run["commands"],
                                                        "arm_context": {self.arm: {"stage_id": sid}}}),
                on_pause=lambda sids, rmap, checkpoint_reason=None, checkpoint_reason_code=None: self._pause_batch(
                    run, plan, receipt, sids, rmap, checkpoint_reason=checkpoint_reason,
                    checkpoint_reason_code=checkpoint_reason_code),
                on_carry_slip_hold=_on_carry_slip_hold,
                run_stage=_run_stage,
                compute_facts=self._grip_facts,
                is_downward_move=lambda rec, beg: (
                    rec["kind"] == "move"
                    and rec["pose"]["position_m"][2] - float(self.tcp(beg["q_rad"])[2, 3]) < DOWNWARD_MOVE_DZ_M
                ),
                post_grasp_recheck=contract.post_grasp_recheck_available(self.deployment),
                guard_parameters=guard_parameters,
                on_after_stage_facts=_after_facts,
                adapt_grip_evaluation=_adapt_grip_eval,
                on_stage_failed=_on_stage_failed,
                on_batch_failed=lambda: self._finish_interrupted(receipt, run),
            )
        except DriverFault as error:
            order = plan["order"]
            current_sid = plan.pop("active_stage_id", None)
            if current_sid is None and order:
                current_sid = order[min(max(int(plan["index"]), 0), len(order) - 1)]
            if current_sid in plan["completed"]:
                plan["completed"] = [sid for sid in plan["completed"] if sid != current_sid]
            plan["status"] = "failed"
            for pause_key in (
                "checkpoint_stage_ids",
                "checkpoint_reasons",
                "checkpoint_reason",
                "checkpoint_reason_code",
                "checkpoint_stage_id",
                "effective_grip_guard",
                "slip_detected_at",
            ):
                receipt.pop(pause_key, None)
            receipt.update(
                status="unknown",
                reason=f"backend_error: {error}",
                paused_at_checkpoint=False,
                failed_stage_id=current_sid,
                failed_arm=self.arm,
            )
            if current_sid is not None and current_sid in receipt["stage_results"]:
                stage_entry = receipt["stage_results"][current_sid]
                stage_entry["status"] = "unknown"
                stage_entry.pop("checkpoint_reason", None)
                stage_entry.pop("checkpoint_reason_code", None)
            receipt["stop_confirmation"] = getattr(error, "stop_confirmation", None) or self._confirm_stop()
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            try:
                stop = self._stop()
            except Exception as stop_err:
                stop = {"confirmed": False, "state": "unknown", "reason": f"{type(stop_err).__name__}: {stop_err}"}
            moved = bool(run["commands"] or self.control_step != start_step)
            uncertain = moved or not bool(stop.get("confirmed"))
            order = plan["order"]
            current_sid = plan.pop("active_stage_id", None)
            if current_sid is None and order:
                idx = int(plan["index"])
                if 0 <= idx < len(order):
                    current_sid = order[idx]
            plan["status"] = "failed"
            for pause_key in (
                "checkpoint_stage_ids",
                "checkpoint_reasons",
                "checkpoint_reason",
                "checkpoint_reason_code",
                "checkpoint_stage_id",
                "effective_grip_guard",
                "slip_detected_at",
            ):
                receipt.pop(pause_key, None)
            receipt.update(
                status="unknown" if uncertain else "failed",
                reason=f"internal_runtime_error: {type(error).__name__}: {error}",
                paused_at_checkpoint=False,
                failed_stage_id=current_sid,
                failed_arm=self.arm,
                stop_confirmation=stop,
            )
            if uncertain:
                receipt["execution_uncertain"] = True
            else:
                receipt["physics_advanced"] = False
            if (
                current_sid is not None
                and current_sid in receipt["stage_results"]
                and receipt["stage_results"][current_sid].get("status") != "completed"
            ):
                stage_entry = receipt["stage_results"][current_sid]
                stage_entry["status"] = "unknown" if uncertain else "failed"
                if uncertain:
                    stage_entry["execution_uncertain"] = True
                stage_entry.pop("checkpoint_reason", None)
                stage_entry.pop("checkpoint_reason_code", None)
        finally:
            self.active_action_id = None
            if self.motion_phase == "moving":
                self.motion_phase = "idle"
        plan["cursor"] += run["commands"]
        summarize_batch(plan_id, plan, plan["order"], receipt["stage_results"], receipt)
        return self._close_receipt(receipt, run, start_step, [
            {"waypoint_index": sid, "q_rad": plan["records"][sid]["q_rad"]} for sid in plan["order"] if "q_rad" in plan["records"][sid]])

    # -------------------------------------------------------- reset/home ---
    def _home(self, cancellation=None):
        d = self.deployment
        result = self.move_joints(d["home"]["q_rad"], d["home"]["gripper_normalized"], d["home"]["arrival_tolerance_rad"],
                                  cancellation=cancellation, action_id="home")
        result["home_q_rad"] = list(d["home"]["q_rad"])
        if result.get("reason") == "joint_arrival_out_of_tolerance":
            result["reason"] = "home_arrival_out_of_tolerance"
        return result

    def move_joints(self, q_target, jaw_normalized=None, arrival_tolerance_rad=None, *, speed_scale=1.0,
                    cancellation=None, action_id="joint_move", correct_sag=True,
                    expected_generation=None, deadline_s=None):
        """Operator/commissioning joint-space move (never an actor action).

        ``q_target``: 5 arm joint goals (rad, URDF convention). ``jaw_normalized``: optional
        jaw goal after arrival (0 closed, 1 open). ``arrival_tolerance_rad``: largest settled
        joint error counted as arrival (default motion.default_orientation_tolerance_rad).
        Uses the same speed cap, tracking/stall guards and bounded sag correction as robot_action;
        returns a receipt-like dict with the per-step trace for analysis.
        ``expected_generation``: optional stop-generation integer captured at the start of a multi-move
        sequence; None binds this move to the current generation. An older token cannot resume motion
        after an intervening hold. ``deadline_s``: optional absolute adapter-monotonic deadline in s;
        None uses motion.max_timeout_s, otherwise the earlier bound applies to every motion/settle phase.
        Example: capture passes one generation/deadline to every pose and park move.
        """
        d = self.deployment
        if expected_generation is not None and type(expected_generation) is not int:
            raise ValueError("move_joints_expected_generation_must_be_integer")
        if deadline_s is not None and (type(deadline_s) not in (int, float) or not math.isfinite(deadline_s)):
            raise ValueError("move_joints_deadline_s_must_be_finite")
        gate = self._gate()
        if gate:
            return {"status": "rejected", "reason": gate, "physics_advanced": False}
        tolerance = d["motion"]["default_orientation_tolerance_rad"] if arrival_tolerance_rad is None else arrival_tolerance_rad
        run = self._new_run(cancellation or threading.Event(), d["motion"]["max_timeout_s"], action_id)
        if expected_generation is not None:
            run["generation"] = expected_generation
        if deadline_s is not None:
            run["deadline"] = min(run["deadline"], float(deadline_s))
        start_step = self.control_step
        self.motion_phase = "moving"
        result = {"status": "failed", "reason": None, "q_target_rad": [float(v) for v in q_target]}
        try:
            ok, reason, diagnostic, _ = self._segment(run, q_target, speed_scale)
            result["segment_commands"] = run["commands"]
            if ok:
                settled, sample, reason = self._settle(run, d["motion"]["arrival_timeout_s"])
                if not settled:
                    ok, reason = False, reason or "settle_unconfirmed"
                result["uncorrected_joint_error_rad"] = [round(float(v), 5) for v in np.subtract(q_target, sample["q_rad"])]
                if correct_sag and settled:
                    ok_c, reason_c, residual = self._arrive(run, q_target, self._pose_dict(self.tcp(q_target)), (math.inf, math.pi))
                    result["arrival_correction"] = (residual or {}).get("arrival_correction")
                    settled = bool((residual or {}).get("settled", False))
                    if not ok_c or not settled:
                        ok, reason = False, reason_c or "settle_unconfirmed"
                    sample = self.read()
                error = float(np.abs(np.subtract(sample["q_rad"], q_target)).max())
                result.update(max_joint_error_rad=round(error, 5), settled=settled, arrival_tolerance_rad=tolerance,
                              final_joint_error_rad=[round(float(v), 5) for v in np.subtract(q_target, sample["q_rad"])])
                if ok:
                    ok = error <= tolerance
                    reason = None if ok else "joint_arrival_out_of_tolerance"
            interrupted = self._interrupt(run, "joint_move")
            if interrupted:
                ok, reason = False, interrupted
            if ok and jaw_normalized is not None:
                ok, reason, jaw = self._gripper(run, jaw_normalized, 1.0, self._squeeze())
                result["gripper_execution"] = jaw
            interrupted = self._interrupt(run, "joint_move")
            if interrupted:
                ok, reason = False, interrupted
            if ok:
                result.update(status="completed", reason=None)
            else:
                result.update(reason=reason, diagnostic=diagnostic)
                self._finish_interrupted(result, run)
        except DriverFault as error:
            result.update(status="unknown", reason=f"backend_error: {error}", stop_confirmation=self._confirm_stop())
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            try:
                stop = self._stop()
            except Exception as stop_err:
                stop = {"confirmed": False, "state": "unknown", "reason": f"{type(stop_err).__name__}: {stop_err}"}
            moved = bool(run["commands"] or self.control_step != start_step)
            uncertain = moved or not bool(stop.get("confirmed"))
            result.update(
                status="unknown" if uncertain else "failed",
                reason=f"internal_runtime_error: {type(error).__name__}: {error}",
                stop_confirmation=stop,
            )
            if uncertain:
                result["execution_uncertain"] = True
        finally:
            if self.motion_phase == "moving":
                self.motion_phase = "idle"
        result.update(executed_native_steps=self.control_step - start_step, physics_advanced=self.control_step != start_step,
                      wall_clock_duration_s=round(self.clock.now() - run["t0"], 4),
                      trace=run["trace"], dispatched=run["dispatched"])
        if (result.get("stop_confirmation") or {}).get("confirmed") is False:
            result["status"] = "unknown"
        return result

    def reset(self, seed=0, layout_id=0):
        before = self.robot_state(self.read())
        home = self._home()
        if home["status"] != "completed":
            raise RuntimeError(f"so101_home_motion_{home['status']}: {home.get('reason')}")
        observation = self.observe()
        return {"reset_id": str(uuid.uuid4()), "episode_index": 0, "seed": seed, "layout_id": layout_id,
                "before_state": before, "robot_state": observation["robot_state"], "observation": observation,
                "clock": "real_time", "reset_strategy": "arm_home_only_world_unchanged",
                "world_reset": False, "seed_layout_semantics": "recorded only; real hardware cannot restore a layout",
                "home": {key: home[key] for key in ("max_joint_error_rad", "arrival_tolerance_rad", "settled") if key in home}}

    # ----------------------------------------------------------- operator ---
    def operator(self, command, progress=None, **arguments):
        """hold / relax / torque_on / set_runtime_limits / go_home / board_check / reload_calibration /
        calibrate / handheld_capture(action=start|status|stop) / handheld_solve (§6.2, §6.4). ``progress(dict)`` receives calibrate phase updates."""
        if command == "hold":
            with self.operator_lock:
                with self.state_lock:
                    self.stop_generation += 1
                    self.stop_reason = "operator_hold"
                if self.relaxed:
                    return {"command": command, "status": "rejected", "reason": "torque_off_cannot_hold"}
                stop = self._stop()
                return {"command": command, "status": "held" if stop["confirmed"] else "unknown",
                        "stop_confirmation": stop, "sim_step": self.control_step}
        if command == "relax":
            with self.operator_lock:
                with self.state_lock:
                    self.stop_generation += 1
                    self.stop_reason = "operator_relax"
                    self.relaxed = True
                self._forget_offsets()
                try:
                    with self.bus_lock:
                        self.driver.set_torque(False)
                        readback = self.driver.torque_enabled()
                except Exception as error:
                    self.motion_phase = "fault"
                    return {"command": command, "status": "unknown", "torque_enabled": None,
                            "stop_confirmation": {"confirmed": False, "state": "unknown"},
                            "reason": f"{type(error).__name__}: {error}", "admission_gate": "latched"}
                self.torque_known = readback
                self.motion_phase = "idle"
                confirmed = readback is False
                return {"command": command, "status": "relaxed" if confirmed else "unknown", "torque_enabled": readback,
                        "stop_confirmation": {"confirmed": False, "state": "torque_off", "reason": "operator_relax"},
                        "admission_gate": "latched_until_torque_on_and_fresh_observation",
                        "warning": "torque off: the arm falls under gravity"}
        if command == "torque_on":
            with self.operator_lock:
                if self.active_action_id is not None:
                    return {"command": command, "status": "rejected", "reason": "action_active"}
                try:
                    with self.bus_lock:
                        sample = self._read()
                        jaw = float(np.clip(sample["gripper_normalized"], 0, 1))
                        self.driver.send_goal(sample["q_rad"], jaw)
                        self.arm_goal, self.jaw_goal = list(sample["q_rad"]), jaw
                        self._pin_jaw_state()  # the jaw goal is the measured jaw: any holding squeeze ended
                        self._forget_offsets()
                        self.control_step += 1
                        self.driver.set_torque(True)
                        readback = self.driver.torque_enabled()
                except Exception as error:
                    return {"command": command, "status": "unknown", "torque_enabled": None,
                            "reason": f"{type(error).__name__}: {error}"}
                self.torque_known = readback
                ok = readback is not False
                with self.state_lock:
                    if ok:
                        self.relaxed = False
                        self.observe_required = True
                self.motion_phase = "holding"
                return {"command": command, "status": "torque_on" if readback is True else "unknown" if readback is None else "failed",
                        "torque_enabled": readback, "goal": "present_position", "admission_gate": "fresh_observation_required"}
        if command == "set_runtime_limits":
            cap = arguments.get("speed_scale_cap")
            low = self.deployment["operator"]["min_speed_scale_cap"]
            if set(arguments) != {"speed_scale_cap"} or type(cap) not in (int, float) or not low <= cap <= 1:
                raise ValueError(f"invalid_speed_scale_cap: expected {low}..1.0 (fraction of deployment max joint speed)")
            with self.state_lock:
                self.speed_scale_cap = float(cap)
            return {"command": command, "status": "applied", "speed_scale_cap": float(cap),
                    "applies_to": "actions and previews started after this command",
                    "runtime_limits": public_limits(self.deployment, float(cap))}
        if command == "go_home":
            return dict(self._home(), command=command, world="objects_not_reset")
        if command in CALIBRATION_WORKFLOW_COMMANDS:
            return self._calibration_workflow(command, progress, arguments)
        if command == "handheld_capture":
            from ni_rua.robot.calibration.workflow.capture import handheld_capture_command
            if set(arguments) != {"action"}:
                raise ValueError("handheld_capture_takes_action: start, status or stop")
            return handheld_capture_command(self, arguments["action"])
        if command == "handheld_solve":
            from ni_rua.robot.calibration.workflow.capture import run_handheld_solve
            if arguments:
                raise ValueError("handheld_solve_takes_no_arguments")
            if self.active_action_id is not None:
                return {"command": command, "status": "rejected", "reason": "action_active", "physics_advanced": False}
            self.active_action_id = "operator_handheld_solve"
            try:
                return run_handheld_solve(self)
            finally:
                self.active_action_id = None
        if command in ("board_check", "reload_calibration", "calibrate") and arguments:
            raise ValueError(f"{command}_takes_no_arguments")
        if command == "board_check":
            return self.board_check()
        if command == "reload_calibration":
            if self.active_action_id is not None:
                return {"command": command, "status": "rejected", "reason": "action_active"}
            return self.reload_calibration()
        if command == "calibrate":
            from ni_rua.robot.calibration.workflow.capture import run_calibration
            if self.handheld is not None and self.handheld.running:
                self.handheld.stop("stopped_for_calibrate")
            if self.active_action_id is not None:
                return {"command": command, "status": "rejected", "reason": "action_active", "physics_advanced": False}
            self.active_action_id = "operator_calibrate"
            try:
                return run_calibration(self, progress=progress)
            finally:
                self.active_action_id = None
        raise ValueError(f"unsupported_operator_command:{command}")

    @property
    def workflow(self):
        if self._workflow is None:
            from ni_rua.robot.calibration.workflow.runner import CalibrationWorkflow
            self._workflow = CalibrationWorkflow(self)
        return self._workflow

    def _calibration_workflow(self, command, progress, arguments):
        """Guided calibration workflow ops (calibration/workflow/runner.py; docs so101-integration.md "Calibration workflow").

        ``calib_workflow_status`` (``summary`` bool: bundle + staleness only), ``calib_workflow_begin`` (``mode``,
        ``changes``, optional ``board``), ``calib_step_start`` (``step``, ``params``), ``calib_step_status``
        (``step``), ``calib_step_cancel``, ``calib_apply``, ``calib_rollback`` (optional ``version``).
        """
        from ni_rua.robot.calibration.workflow.runner import StepRefused
        allowed = {"calib_workflow_status": {"summary"}, "calib_workflow_begin": {"mode", "changes", "board"},
                   "calib_step_start": {"step", "params"}, "calib_step_status": {"step"}, "calib_step_cancel": set(),
                   "calib_apply": set(), "calib_rollback": {"version"}}[command]
        if set(arguments) - allowed:
            raise ValueError(f"{command}_unexpected_arguments:" + ",".join(sorted(set(arguments) - allowed)))
        workflow = self.workflow
        base = {"command": command, "physics_advanced": False}
        if command == "calib_workflow_status":
            if arguments.get("summary"):
                return dict(base, status="ok", bundle=workflow.bundle_status(), staleness=workflow.staleness_hint(),
                            running_step=workflow.running_step)
            return dict(base, status="ok", **workflow.status())
        if command == "calib_step_status":
            step = arguments.get("step")
            run = workflow.run or {}
            if step not in (run.get("steps") or {}):
                return dict(base, status="rejected", reason=f"calibration_step_unknown_or_no_run:{step}")
            return dict(base, status="ok", step=step, state=deepcopy(run["steps"][step]),
                        readiness=workflow.readiness(step), running_step=workflow.running_step,
                        handheld=workflow.capture.snapshot() if step == "S3" and workflow.capture is not None else None)
        if command == "calib_step_cancel":
            return workflow.cancel()
        if self.active_action_id is not None and not self.active_action_id.startswith("operator_calibration_"):
            return dict(base, status="rejected", reason="action_active")
        if command == "calib_workflow_begin":
            try:
                return dict(base, status="started", **workflow.begin(arguments.get("mode"), arguments.get("changes") or (),
                                                                       arguments.get("board")))
            except (StepRefused, ValueError) as error:
                return dict(base, status="rejected", reason=str(error))
        if command == "calib_step_start":
            if self.handheld is not None and self.handheld.running and self.handheld is not workflow.capture:
                self.handheld.stop("stopped_for_calibration_workflow")
            return workflow.start_step(arguments.get("step"), arguments.get("params"), progress)
        if command == "calib_apply":
            return workflow.apply()
        return workflow.rollback(arguments.get("version"))

    # ---------------------------------------------------------- telemetry ---
    def _write_atomic(self, path, data):
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_bytes(data)
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def telemetry_tick(self, *, frames=True):
        """One telemetry write. Operator-only; never raises into execution."""
        now = self.clock.now()
        faults = list(self.faults)
        sample, connected = self.last_sample, True
        period = 1 / self.deployment["telemetry"]["joint_hz"]
        if sample is None or now - sample["t"] > period:
            try:
                sample = self.read()
            except Exception as error:
                connected, sample = False, None
                faults.append(f"telemetry_read_failed: {type(error).__name__}")
        motor_hz = self.deployment["telemetry"]["motor_status_hz"]
        if connected and motor_hz > 0 and (self._last_motor_read is None or now - self._last_motor_read >= 1 / motor_hz):
            self._last_motor_read = now
            try:
                with self.bus_lock:
                    self._last_motor_status = self.driver.motor_status()
                    self.torque_known = self.driver.torque_enabled()
            except Exception as error:
                faults.append(f"motor_status_read_failed: {type(error).__name__}")
        tcp_position = tcp_quaternion = None
        if sample is not None:
            T = self.tcp(sample["q_rad"])
            tcp_position, tcp_quaternion = T[:3, 3].tolist(), matrix_to_quaternion(T[:3, :3])
        goals = None if self.arm_goal is None else dict(zip(ARM_JOINTS, self.arm_goal))
        if goals is not None:
            goals["gripper"] = gripper_rad_from_normalized(self.deployment, self.jaw_goal)
        cameras = {}
        for name, camera in self.cameras.items():
            try:
                cameras[name] = camera.status()
            except Exception as error:
                cameras[name] = {"ok": False, "fps": None, "last_frame_at": None, "error": type(error).__name__}
        self.telemetry_seq += 1
        state = {"schema": TELEMETRY_SCHEMA, "seq": self.telemetry_seq, "captured_at": self.clock.utc(now), "monotonic_s": now,
                 "connected": connected, "torque_enabled": self.torque_known,
                 "q_rad": None if sample is None else {**dict(zip(ARM_JOINTS, sample["q_rad"])), "gripper": sample["gripper_rad"]},
                 "q_goal_rad": goals,
                 "gripper_normalized": None if sample is None else sample["gripper_normalized"],
                 "tcp_position_m": tcp_position, "tcp_quaternion_xyzw": tcp_quaternion,
                 "motor": {name: {key: value.get(key) for key in ("temperature_c", "load", "voltage_v", "current_ma")}
                           for name, value in self._last_motor_status.items()},
                 "motion": {"active_action_id": self.active_action_id, "phase": self.motion_phase,
                            "tracking_error_rad": self.last_tracking_error},
                 "cameras": cameras, "faults": faults[-10:]}
        self._write_atomic(self.telemetry_dir / "state.json", json.dumps(state, allow_nan=False).encode())
        frame_period = 1 / self.deployment["telemetry"]["frame_hz"]
        if frames and (self._last_frame_write is None or now - self._last_frame_write >= frame_period):
            self._last_frame_write = now
            self._write_frames()
        return state

    def _write_frames(self):
        from PIL import Image
        import io
        side = self.deployment["telemetry"]["preview_max_side_px"]
        for name, camera in self.cameras.items():
            try:
                latest = camera.latest()
                if latest is None:
                    continue
                image = Image.fromarray(np.asarray(latest[0], dtype=np.uint8))
                image.thumbnail((side, side))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=80)
                self._write_atomic(self.telemetry_dir / f"{name}.jpg", buffer.getvalue())
            except Exception:
                continue

    def _telemetry_loop(self):
        period = 1 / self.deployment["telemetry"]["joint_hz"]
        while not self._telemetry_stop.wait(period):
            try:
                self.telemetry_tick()
            except Exception as error:
                self.faults = (self.faults + [f"telemetry_failed: {type(error).__name__}"])[-10:]

    # -------------------------------------------------------------- close ---
    def close(self, progress=None):
        if self.handheld is not None and self.handheld.running:
            self.handheld.stop("worker_closing")
        self._telemetry_stop.set()
        if self.telemetry_thread is not None:
            self.telemetry_thread.join(timeout=2)
        errors = []
        for camera in self.cameras.values():
            try:
                camera.close()
            except Exception as error:
                errors.append(f"camera:{type(error).__name__}")
        with self.bus_lock:
            self.driver.disconnect()
        try:
            self.motion_video.save()
        except Exception as error:
            errors.append(f"motion_video:{type(error).__name__}:{error}")
        if progress:
            progress({"phase": "driver_disconnected", "camera_errors": errors})


# ------------------------------------------------------------- JSONL loop ---

def serve(adapter_factory, arguments, emit_message):
    """Run the shared ``ipc.serve_worker`` loop with SO-101 operator preemption and EOF hold."""
    serve_worker(
        adapter_factory,
        arguments,
        emit_message,
        preemptive_operator_commands=PREEMPTIVE_OPERATOR_COMMANDS,
        on_eof=lambda adapter: adapter.operator("hold"),
    )


def supervised_by_pipe(fd=0):
    """True when ``fd`` (stdin) is a pipe or socket, i.e. the JSONL owner process feeds this worker."""
    try:
        mode = os.fstat(fd).st_mode
    except OSError:
        return False
    return stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode)


def _factory(arguments):
    return SO101Adapter(load_deployment(arguments.deployment), arguments.run_dir, arguments.observation_profile,
                        deployment_path=arguments.deployment)


def main():
    global protocol_output
    # The worker shares the terminal's process group (plain Popen). Closing the terminal (SIGHUP)
    # or Ctrl-C (SIGINT to the whole group) must not kill it before the gateway sends ``close``:
    # the gateway owns shutdown, and stdin EOF (owner gone) still holds the arm and exits in
    # ``serve``. SIGINT is ignored only when supervised (stdin is a pipe from the parent); a
    # worker started by hand from a terminal keeps the default Ctrl-C behaviour.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    if supervised_by_pipe():
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    protocol_output = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--observation-profile", required=True)
    parser.add_argument("--robot-model-id", default=SO101.robot_model_id)
    arguments = parser.parse_args()
    if arguments.robot_model_id != SO101.robot_model_id:
        raise SystemExit(f"unsupported_so101_robot:{arguments.robot_model_id}")
    import ni_rua
    # Operator-only provenance in robot-worker.log: which interpreter and checkout this worker runs.
    print(f"so101_worker_start pid={os.getpid()} python={sys.executable} ni_rua={ni_rua.__file__}", file=sys.stderr, flush=True)
    serve(_factory, arguments, emit)


if __name__ == "__main__":
    main()
