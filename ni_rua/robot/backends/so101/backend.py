"""Parent-process ``so101_real`` backend: one SO-101 arm behind a JSONL worker.

The worker runs in the deployment's own interpreter (mamba env ``so101`` with
LeRobot) and exclusively owns the serial bus and cameras. This process owns
only the subprocess, validation and request serialization; it never imports
LeRobot, opens a port or reads a camera. Every physical action still enters
through ``Harness.execute``; operator commands go through ``operator``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from ....core.observation_policy import ACTOR_PROFILES, allows_depth, has_calibration
from ...ipc import WorkerProcess
from ...model.embodiments import SO101
from . import contract
from .deployment import embodiment_profile, load_deployment

ENVIRONMENT = "so101_real"
CALIBRATION_WORKFLOW_COMMANDS = ("calib_workflow_status", "calib_workflow_begin", "calib_step_start", "calib_step_status",
                                 "calib_step_cancel", "calib_apply", "calib_rollback")
OPERATOR_COMMANDS = ("hold", "relax", "torque_on", "set_runtime_limits", "go_home",
                     "board_check", "reload_calibration", "calibrate", "handheld_capture", "handheld_solve",
                     *CALIBRATION_WORKFLOW_COMMANDS)
# Preemptive commands are answered off the worker execution thread (no operation_lock). Keep in sync with worker.py.
PREEMPTIVE_OPERATOR_COMMANDS = ("hold", "relax", "torque_on", "set_runtime_limits", "board_check", "handheld_capture",
                                "calib_workflow_status", "calib_step_status", "calib_step_cancel")


class RobotBackend(WorkerProcess):
    robot_description = SO101

    def __init__(self, *, config=None, run_dir="runs/robot", observation_profile=None, deployment=None, **_ignored):
        """``_ignored`` absorbs simulator-only Workbench arguments (task, gpu_id,
        seed, layout_id, tracking_guard, joint_replan); none applies to hardware."""
        from ....core.config import Config
        config = config if config is not None else Config.load()
        if config.backend.environment != ENVIRONMENT:
            raise ValueError(f"unsupported_so101_environment: {config.backend.environment}")
        if config.backend.robot_model_id != SO101.robot_model_id:
            raise ValueError(f"unsupported_so101_real_robot: {config.backend.robot_model_id!r}")
        if config.features.motion_execution_mode != "resampled":
            raise ValueError("so101_real_requires_motion_execution_mode_resampled")
        path = deployment if deployment is not None else config.backend.deployment
        if not path:
            raise ValueError(
                "so101_deployment_required: set backend.deployment to configs/so101.local.json or "
                "~/.config/ni-rua/so101.local.json (run `ni-rua-so101 init-config` to generate one)"
            )
        self.deployment_path = str(Path(path).expanduser().resolve())
        self.deployment = load_deployment(self.deployment_path)
        profile = config.features.observation_profile if observation_profile is None else observation_profile
        if profile not in ACTOR_PROFILES or allows_depth(profile):
            raise ValueError(f"so101_unsupported_observation_profile: {profile}")
        if has_calibration(profile):
            calibration = self.deployment.get("camera_calibration_file")
            if not calibration or not Path(calibration).is_file():
                raise ValueError(
                    f"so101_camera_calibration_missing: camera_calibration_file ({calibration or 'unset'}) not found "
                    "for calibrated profile; for cold-start calibration, launch with "
                    "`--observation-profile rgb_proprio_v1` and open `/calibration` in the workbench UI"
                )
        self.observation_profile = profile
        self.python = self.deployment["python"]
        from ....core.storage.images import image_settings
        self.image_settings = image_settings(config.images.to_dict())
        super().__init__(run_dir)

    # ---------------------------------------------------- pure hooks ---
    compile_action = staticmethod(contract.compile_action)

    def validate_action(self, action):
        return contract.validate_action(action, self.deployment)

    def actor_motion_instructions(self):
        """SO-101 replacement for the X5 motion/coordinate prompt sections."""
        return contract.actor_motion_instructions(self.deployment)

    def joint_limits_rad(self):
        """URDF arm joint limits ``{joint: [lower_rad, upper_rad]}`` (rad, nominal, not shrunk by
        ``motion.joint_limit_margin_rad``) for the operator console gauges; pure, no worker."""
        from ...model import so101_kinematics
        return {name: list(bounds) for name, bounds in so101_kinematics.joint_limits_rad(so101_kinematics.load_model()).items()}

    def embodiment_profile(self):
        """The run's ``EmbodimentProfile`` from the loaded deployment file (pure, parent-side, no worker; the worker
        builds the same profile from the same file with ``deployment.embodiment_profile``). Read once by
        ``Harness.__init__``."""
        return embodiment_profile(self.deployment)

    # ----------------------------------------------------- lifecycle ---
    def start(self):
        if self.process is not None and self.capabilities is not None and self.process.poll() is None:
            return self.capabilities
        self.run_dir.mkdir(parents=True, exist_ok=True)
        project = str(Path(__file__).resolve().parents[4])
        environment = os.environ.copy()
        environment.update(NI_RUA_IMAGE_SETTINGS=json.dumps(self.image_settings), PYTHONDONTWRITEBYTECODE="1",
                           PYTHONPATH=os.pathsep.join(item for item in (project, environment.get("PYTHONPATH", "")) if item))
        command = [self.python, "-u", "-m", "ni_rua.robot.backends.so101.worker",
                   "--deployment", self.deployment_path, "--run-dir", str(self.run_dir),
                   "--observation-profile", self.observation_profile, "--robot-model-id", SO101.robot_model_id]
        return self.start_process(command, cwd=project, environment=environment)

    def observe(self):
        with self.operation_lock:
            return self._request("observe")

    def record(self, enabled=True):
        """Enable/pause motion-only recording of every camera (bool, default True)."""
        with self.operation_lock:
            return self._request("record", {"enabled": bool(enabled)})

    def save_video(self):
        """Seal per-camera H.264 videos, compose side-by-side timeline MP4, and return observer report."""
        with self.operation_lock:
            report = self._request("save_video", timeout=90)
        from .motion_video_composite import compose
        report = dict(report)
        try:
            report["composite"] = compose(report)
        except Exception as error:
            report["composite"] = dict(audience="observer_only", kind="side_by_side_composite", status="failed",
                                       error=f"{type(error).__name__}: {error}")
        return report

    def execute(self, action, progress_callback=None, cancel_flag=None):
        self.validate_action(action)
        with self.operation_lock:
            receipt = self._request("execute", action, timeout=float(action["timeout_s"]) + self.deployment["ipc"]["response_margin_s"],
                                    progress=progress_callback, cancel=cancel_flag)
        # Stages compiled from a grasp centre name it and the TCP target it became (contract.grasp_centre_report).
        return contract.annotate_grasp_centres(receipt, action, self.deployment, key="stage_results", compact=True,
                                               ran_only=True)

    def preview_batch(self, action):
        self.validate_action(action)
        with self.operation_lock:
            plan = self._request("preview_batch", action,
                                 timeout=float(action["timeout_s"]) + self.deployment["ipc"]["response_margin_s"])
        # stage_plans[<stage>].grasp_centre: the conversion of a grasp-centre request, computed here (parent side,
        # pure CAD) from the same compiled request the worker planned.
        return contract.annotate_grasp_centres(plan, action, self.deployment, key="stage_plans")

    def reach_batch(self, action):
        """robot_action ``mode=reach`` (``SO101Adapter.reach_batch``): IK of every move stage from the measured joints,
        no plan and no motion. ``action``: a compiled BatchAction of move stages, validated like a preview."""
        self.validate_action(action)
        with self.operation_lock:
            verdicts = self._request("reach_batch", action,
                                     timeout=float(action["timeout_s"]) + self.deployment["ipc"]["response_margin_s"])
        return contract.annotate_grasp_centres(verdicts, action, self.deployment, key="stage_results")

    def validate_reset(self, seed=0, layout_id=0):
        """Real hardware has no layouts; seed/layout are recorded, never restored."""
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (seed, layout_id)):
            raise ValueError("seed and layout_id must be nonnegative integers")

    def reset(self, seed=0, layout_id=0):
        self.validate_reset(seed, layout_id)
        with self.operation_lock:
            return self._request("reset", {"seed": seed, "layout_id": layout_id},
                                 timeout=self.deployment["motion"]["max_timeout_s"] + self.deployment["ipc"]["response_margin_s"])

    def calibrate_timeout_s(self):
        """Parent wait for ``calibrate``: capture budget + one in-flight pose move + park + solve + margin (s)."""
        calibration, motion = self.deployment["calibration"], self.deployment["motion"]
        return (calibration["capture_timeout_s"] + 2 * motion["max_timeout_s"] + calibration["solve_timeout_s"]
                + self.deployment["ipc"]["response_margin_s"])

    def calibration_step_timeout_s(self):
        """Parent wait for ``calib_step_start``: the longest step (grid capture + solve, or a touch sequence)
        plus one in-flight move and park and the IPC margin (s)."""
        calibration, motion = self.deployment["calibration"], self.deployment["motion"]
        longest = max(calibration["capture_timeout_s"] + calibration["solve_timeout_s"], calibration["touch_timeout_s"])
        return longest + 2 * motion["max_timeout_s"] + self.deployment["ipc"]["response_margin_s"]

    def reload_deployment(self):
        """Reload deployment configuration (e.g. after ``calib_apply`` / ``calib_rollback`` updates joint_mapping)."""
        self.deployment = load_deployment(self.deployment_path)
        return self.deployment

    def operator(self, command, progress_callback=None, **arguments):
        """Operator-only commands (docs/development/so101-integration.md §6.2, §6.4).

        hold/relax/torque_on/set_runtime_limits/board_check never take ``operation_lock``:
        the worker handles them off its execution thread, so they preempt a running action.
        ``handheld_capture`` (action start/status/stop; samples the worker's own fixed camera) is preemptive too.
        ``go_home``, ``calibrate`` (both move the arm), ``reload_calibration`` and ``handheld_solve`` are serialized.
        ``progress_callback(dict)`` receives worker progress (calibrate phases).
        """
        if command not in OPERATOR_COMMANDS:
            raise ValueError(f"unsupported_operator_command:{command}")
        payload = dict(arguments, command=command)
        if command not in PREEMPTIVE_OPERATOR_COMMANDS:
            timeout = {"go_home": self.deployment["motion"]["max_timeout_s"] + self.deployment["ipc"]["response_margin_s"],
                       "calibrate": self.calibrate_timeout_s(),
                       "reload_calibration": self.deployment["ipc"]["operator_timeout_s"],
                       "handheld_solve": self.deployment["calibration"]["solve_timeout_s"]
                       + self.deployment["ipc"]["response_margin_s"],
                       "calib_step_start": self.calibration_step_timeout_s(),
                       "calib_workflow_begin": self.deployment["ipc"]["operator_timeout_s"],
                       "calib_apply": self.deployment["ipc"]["operator_timeout_s"],
                       "calib_rollback": self.deployment["ipc"]["operator_timeout_s"]}[command]
            with self.operation_lock:
                result = self._request("operator", payload, timeout=timeout, progress=progress_callback)
        else:
            result = self._request("operator", payload, timeout=self.deployment["ipc"]["operator_timeout_s"])
        if command == "set_runtime_limits" and isinstance(result, dict) and result.get("status") == "applied" \
                and isinstance(self.capabilities, dict):
            self.capabilities["runtime_limits"] = result["runtime_limits"]
        if command in ("calib_apply", "calib_rollback") and isinstance(result, dict) \
                and result.get("status") in ("applied", "rolled_back"):
            self.reload_deployment()
        return result


SO101Backend = RobotBackend
