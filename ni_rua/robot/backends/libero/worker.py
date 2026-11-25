"""Clean-room simulation-only LIBERO (robosuite MuJoCo) Franka Emika Panda worker.

Runs inside the LIBERO simulator Python interpreter (Python 3.10+) and supports:
1. ``--stdio``: single-episode child worker supervised by ``WorkerProcess`` over stdin/stdout JSONL.
2. ``--socket <path>``: warm persistent Unix domain socket daemon serving ``open_episode`` / ``close_episode``
   in ~25 ms across pooled RL rollouts without recreating MuJoCo or EGL contexts when the BDDL scene matches.

All motion execution dispatches 20 Hz closed-loop operational-space deltas directly to robosuite's native
``OSC_POSE`` controller with continuous closed-loop ``PandaGripper`` aperture control (no binary snap).
Strictly simulation-only; NOT calibrated for physical Franka Emika hardware.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace
from typing import Any
import uuid

import numpy as np

from ni_rua.core.contracts import canonical, digest, validate_depth_metadata
from ni_rua.core.observation_policy import ACTOR_PROFILES, actor_robot_state, allows_depth, has_calibration
from ni_rua.core.storage.images import camera_path, encode_image, image_settings as load_image_settings
from ni_rua.robot.backends.libero import contract
from ni_rua.robot.embodiments.libero_franka_panda import grasp_contact, pose_convention
from ni_rua.robot.embodiments.libero_franka_panda.profile import (
    ARM_ID,
    ARM_JOINTS,
    GRIPPER_JOINTS,
    LIBERO_FRANKA_PANDA,
    PANDA_EMPTY_CLOSE_APERTURE_M,
    PANDA_FULL_OPEN_APERTURE_M,
    TCP_LINK,
    WRIST_CAMERA_PARENT_LINK,
    fk,
    fk_all,
    ik,
    ik_position_priority,
    joint_limits_rad,
    libero_franka_panda_profile,
    load_model,
)
from ni_rua.robot.ipc import WIRE_PREFIX, serve_worker
from ni_rua.robot.ipc.protocol import _error_frame
from ni_rua.robot.motion.batch_scheduler import (
    ArmLayout,
    BatchScheduler,
    jaw_pauses,
    order_stages,
    pause_receipt,
    summarize_batch,
)
from ni_rua.robot.motion.grip_guard import (
    CARRY_SLIP_GUARD_FIELDS,
    CLOSING_JAW_GOAL_BELOW,
    DOWNWARD_MOVE_DZ_M,
    JAW_NOT_CLOSED_DELTA,
    grip_facts_from_measurements,
    stage_grip_guard,
    validate_batch_carry_slip_guard,
)
from ni_rua.robot.motion.motion_tolerance import orientation_is_free

STEP_DOMAIN = "libero_control_step"
CAMERA_MAP: tuple[tuple[str, str], ...] = (
    ("agentview", "agentview"),
    ("wrist", "robot0_eye_in_hand"),
)
LIBERO_SUITES: tuple[str, ...] = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
)
PATH_OVERLAY_SAMPLES = 8
DEPTH_CLIP_M: tuple[float, float] = (0.02, 5.0)
REACH_EXCLUDED_TOLERANCES = frozenset({
    "checked_on",
    "max_upward_dz_m",
    "max_downward_dz_m",
    "max_tracking_divergence_m",
    "stall_guard_preset",
})

_STDIO_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _emit_stdio(message: dict[str, Any]) -> None:
    with _STDIO_LOCK:
        print(WIRE_PREFIX + json.dumps(message, allow_nan=False), file=sys.stdout, flush=True)


def _ensure_libero_config(work_dir: Path) -> Path:
    """Ensure ``LIBERO_CONFIG_PATH`` points to a valid non-interactive ``config.yaml`` before importing LIBERO."""
    spec = importlib.util.find_spec("libero.libero")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("libero_package_unavailable: libero.libero is not installed in backend.python")
    libero_root = Path(next(iter(spec.submodule_search_locations))).resolve()

    existing_dir = os.environ.get("LIBERO_CONFIG_PATH")
    if existing_dir:
        cfg_path = Path(existing_dir) / "config.yaml"
        if cfg_path.is_file():
            try:
                import yaml

                loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                if (
                    isinstance(loaded, dict)
                    and Path(str(loaded.get("bddl_files", ""))).is_dir()
                    and Path(str(loaded.get("init_states", ""))).is_dir()
                ):
                    return libero_root
            except Exception:
                pass

    import yaml

    cfg_dir = (work_dir / ".libero").resolve()
    datasets_dir = cfg_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_root": str(libero_root),
        "bddl_files": str(libero_root / "bddl_files"),
        "init_states": str(libero_root / "init_files"),
        "datasets": str(datasets_dir),
        "assets": str(libero_root / "assets"),
    }
    (cfg_dir / "config.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(cfg_dir)
    return libero_root


def _resolve_libero_task(
    *,
    libero_root: Path,
    task: str | None,
    suite_name: str | None,
    task_id: int | None,
    bddl_file: str | None,
    init_state_file: str | None = None,
) -> dict[str, Any]:
    """Resolve ``(suite_name, task_id, task_name, language_instruction, bddl_file, init_states)``."""
    import torch
    from libero.libero import benchmark

    explicit_init = None
    if init_state_file:
        init_p = Path(init_state_file).expanduser().resolve()
        if not init_p.is_file():
            raise ValueError(f"libero_init_state_file_not_found: {init_p}")
        explicit_init = torch.load(init_p, map_location="cpu", weights_only=False)

    if bddl_file:
        bddl_path = Path(bddl_file).expanduser().resolve()
        if not bddl_path.is_file():
            raise ValueError(f"libero_bddl_not_found: {bddl_path}")
        import libero.libero.envs.bddl_utils as bddl_utils

        info = bddl_utils.get_problem_info(str(bddl_path))
        folder = bddl_path.parent.name
        stem = bddl_path.stem
        return {
            "suite_name": suite_name or folder,
            "task_id": int(task_id or 0),
            "task_name": stem,
            "language_instruction": str(info.get("language_instruction") or stem.replace("_", " ")),
            "bddl_file": str(bddl_path),
            "init_states": explicit_init,
        }

    raw_task = (task or "").strip()
    if raw_task.endswith(".bddl") and Path(raw_task).expanduser().is_file():
        return _resolve_libero_task(
            libero_root=libero_root,
            task=None,
            suite_name=suite_name,
            task_id=task_id,
            bddl_file=raw_task,
        )

    parsed_suite = suite_name
    parsed_id = task_id
    parsed_query: str | None = None
    if parsed_suite is None and raw_task:
        for sep in (":", "/"):
            if sep in raw_task:
                left, right = raw_task.split(sep, 1)
                if left in LIBERO_SUITES:
                    parsed_suite = left
                    if right.isdigit():
                        parsed_id = int(right)
                    else:
                        parsed_query = right.strip()
                    break
        if parsed_suite is None:
            if raw_task in LIBERO_SUITES:
                parsed_suite = raw_task
                parsed_id = 0 if parsed_id is None else parsed_id
            elif raw_task.isdigit():
                parsed_suite = "libero_spatial"
                parsed_id = int(raw_task)
            else:
                parsed_query = raw_task

    bench_dict = benchmark.get_benchmark_dict()
    if parsed_suite is not None:
        if parsed_suite not in bench_dict:
            raise ValueError(f"unsupported_libero_suite: {parsed_suite}")
        suite = bench_dict[parsed_suite]()
        if parsed_id is None and parsed_query:
            q_norm = parsed_query.lower().removesuffix(".bddl")
            for idx in range(suite.n_tasks):
                t_obj = suite.get_task(idx)
                if q_norm in {
                    t_obj.name.lower(),
                    Path(t_obj.bddl_file).stem.lower(),
                    t_obj.language.lower(),
                }:
                    parsed_id = idx
                    break
            if parsed_id is None:
                raise ValueError(f"unknown_libero_task: {parsed_suite}:{parsed_query}")
        idx = 0 if parsed_id is None else int(parsed_id)
        if not 0 <= idx < suite.n_tasks:
            raise ValueError(f"libero_task_id_out_of_range: {parsed_suite}:{idx} not in 0..{suite.n_tasks - 1}")
        t_obj = suite.get_task(idx)
    else:
        q_norm = (parsed_query or "0").lower().removesuffix(".bddl")
        matched = None
        for s_name in LIBERO_SUITES:
            if s_name not in bench_dict:
                continue
            suite = bench_dict[s_name]()
            for idx in range(suite.n_tasks):
                t_obj = suite.get_task(idx)
                if q_norm in {
                    t_obj.name.lower(),
                    Path(t_obj.bddl_file).stem.lower(),
                    t_obj.language.lower(),
                }:
                    matched = (s_name, idx, t_obj)
                    break
            if matched is not None:
                break
        if matched is None:
            raise ValueError(f"unknown_libero_task: {raw_task!r}")
        parsed_suite, idx, t_obj = matched

    bddl_path = (libero_root / "bddl_files" / t_obj.problem_folder / t_obj.bddl_file).resolve()
    if explicit_init is not None:
        init_states = explicit_init
    else:
        init_path = (libero_root / "init_files" / t_obj.problem_folder / t_obj.init_states_file).resolve()
        init_states = torch.load(init_path, map_location="cpu", weights_only=False) if init_path.is_file() else None
    return {
        "suite_name": str(parsed_suite),
        "task_id": int(idx),
        "task_name": str(t_obj.name),
        "language_instruction": str(t_obj.language),
        "bddl_file": str(bddl_path),
        "init_states": init_states,
    }


def _matrix_to_quaternion_xyzw(rot: np.ndarray) -> list[float]:
    """Convert 3x3 proper rotation matrix into unit ``[qx, qy, qz, qw]``."""
    m = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (m[2, 1] - m[1, 2]) * s
        qy = (m[0, 2] - m[2, 0]) * s
        qz = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(max(1e-16, 1.0 + m[0, 0] - m[1, 1] - m[2, 2]))
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(max(1e-16, 1.0 + m[1, 1] - m[0, 0] - m[2, 2]))
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(max(1e-16, 1.0 + m[2, 2] - m[0, 0] - m[1, 1]))
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    nrm = float(np.linalg.norm(q))
    if nrm > 0.0:
        q /= nrm
    if q[3] < 0.0:
        q = -q
    return [float(v) for v in q]


def _quaternion_xyzw_to_matrix(quat_xyzw: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    nrm = float(np.linalg.norm(q))
    if nrm <= 1e-12:
        raise ValueError("zero_norm_quaternion")
    x, y, z, w = q / nrm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose_to_matrix(pose: dict[str, Any]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quaternion_xyzw_to_matrix(pose["quaternion_xyzw"])
    T[:3, 3] = np.asarray(pose["position_m"], dtype=np.float64).reshape(3)
    return T


def _so3_log(rot_err: np.ndarray) -> np.ndarray:
    """Axis-angle vector ``omega * theta`` (rad) for proper rotation ``rot_err``."""
    cos_angle = float(np.clip(0.5 * (float(np.trace(rot_err)) - 1.0), -1.0, 1.0))
    angle = math.acos(cos_angle)
    if angle < 1e-7:
        return 0.5 * np.array(
            [
                rot_err[2, 1] - rot_err[1, 2],
                rot_err[0, 2] - rot_err[2, 0],
                rot_err[1, 0] - rot_err[0, 1],
            ],
            dtype=np.float64,
        )
    sin_angle = math.sin(angle)
    if abs(sin_angle) < 1e-6:
        eigvals, eigvecs = np.linalg.eig(rot_err)
        idx = int(np.argmin(np.abs(eigvals - 1.0)))
        axis = np.real(eigvecs[:, idx])
        nrm = float(np.linalg.norm(axis))
        if nrm > 1e-9:
            axis /= nrm
        return axis * angle
    skew = np.array(
        [
            rot_err[2, 1] - rot_err[1, 2],
            rot_err[0, 2] - rot_err[2, 0],
            rot_err[1, 0] - rot_err[0, 1],
        ],
        dtype=np.float64,
    )
    return (0.5 * angle / sin_angle) * skew


def _save_depth_capture(path: Path, depth_m: np.ndarray, width: int, height: int) -> dict[str, Any]:
    depth = np.asarray(depth_m, dtype=np.float32).reshape(height, width)
    near, far = DEPTH_CLIP_M
    fields = {
        "depth_path": str(path),
        "depth_kind": "z",
        "depth_unit": "m",
        "depth_clip_m": [float(near), float(far)],
        "depth_shape": [int(height), int(width)],
    }
    validate_depth_metadata(dict(fields, width=int(width), height=int(height)), "depth_path")
    valid = np.isfinite(depth) & (depth >= near) & (depth <= far)
    np.savez(path, depth=depth, valid=valid)
    return fields


class LiberoAdapter:
    """MuJoCo LIBERO environment owner implementing NI-RUA's worker adapter contract."""

    def __init__(self, arguments: Any) -> None:
        self.env = None
        self._loaded_bddl_file: str | None = None
        self._loaded_resolution: int | None = None
        self.model = load_model()
        self.limits = joint_limits_rad(self.model)
        self.arm = ARM_ID
        self.open_episode(arguments)

    # ----------------------------------------------------- episode setup ---
    def open_episode(self, arguments: Any) -> dict[str, Any]:
        run_dir = Path(getattr(arguments, "run_dir", "runs/robot")).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        libero_root = _ensure_libero_config(run_dir)

        obs_profile = getattr(arguments, "observation_profile", "rgb_proprio_calibrated_v1")
        if obs_profile not in ACTOR_PROFILES:
            raise ValueError(f"libero_unsupported_observation_profile:{obs_profile}")
        resolution = int(getattr(arguments, "camera_resolution", 512) or 512)
        if not 64 <= resolution <= 2048:
            raise ValueError("invalid_camera_resolution")

        raw_img_settings = getattr(arguments, "image_settings", None)
        if raw_img_settings is None and os.environ.get("NI_RUA_IMAGE_SETTINGS"):
            raw_img_settings = json.loads(os.environ["NI_RUA_IMAGE_SETTINGS"])
        self.image_settings = load_image_settings(raw_img_settings)

        from ni_rua.robot.embodiments.libero_franka_panda.profile import panda_defaults

        raw_thresholds = getattr(arguments, "motion_thresholds", None)
        if raw_thresholds is None and os.environ.get("NI_RUA_MOTION_THRESHOLDS"):
            raw_thresholds = json.loads(os.environ["NI_RUA_MOTION_THRESHOLDS"])
        merged_thresholds = dict(panda_defaults()["thresholds"])
        if isinstance(raw_thresholds, dict):
            merged_thresholds.update({k: v for k, v in raw_thresholds.items() if v is not None})
        self.motion_thresholds = merged_thresholds

        spec = _resolve_libero_task(
            libero_root=libero_root,
            task=getattr(arguments, "task", None),
            suite_name=getattr(arguments, "suite_name", None),
            task_id=getattr(arguments, "task_id", None),
            bddl_file=getattr(arguments, "bddl_file", None),
            init_state_file=getattr(arguments, "init_state_file", None),
        )

        self._libero_root = libero_root
        self.run_dir = run_dir
        self.observation_profile = obs_profile
        self.calibrated = has_calibration(obs_profile)
        self.want_depth = allows_depth(obs_profile)
        self.camera_resolution = resolution
        self.max_episode_steps = int(getattr(arguments, "max_episode_steps", 1000) or 1000)
        self.stop_on_success = bool(getattr(arguments, "stop_on_success", False))
        self.seed = int(getattr(arguments, "seed", 0) or 0)
        self.layout_id = int(getattr(arguments, "layout_id", 0) or 0)
        raw_init_id = getattr(arguments, "init_state_id", None)
        self.explicit_init_state_id = None if raw_init_id is None else int(raw_init_id)
        self.init_state_file = getattr(arguments, "init_state_file", None)
        raw_lang_override = getattr(arguments, "language_override", None)
        self.language_override = str(raw_lang_override) if raw_lang_override is not None else None
        raw_pert = getattr(arguments, "perturbation", None)
        if raw_pert is None and getattr(arguments, "perturbation_json", None):
            raw_pert = json.loads(arguments.perturbation_json)
        self.perturbation = deepcopy(raw_pert) if isinstance(raw_pert, dict) else None

        self.suite_name = spec["suite_name"]
        self.task_id = spec["task_id"]
        self.task_name = spec["task_name"]
        self.native_language_instruction = spec["language_instruction"]
        self.language_instruction = self.language_override or self.native_language_instruction
        self.bddl_file = spec["bddl_file"]
        self._init_states = spec["init_states"]
        self.task = f"{self.suite_name}:{self.task_id}"

        need_new_env = (
            self.env is None
            or self._loaded_bddl_file != self.bddl_file
            or self._loaded_resolution != self.camera_resolution
        )
        if need_new_env:
            if self.env is not None:
                try:
                    self.env.close()
                except Exception:
                    pass
                self.env = None
            from libero.libero.envs import OffScreenRenderEnv

            self.env = OffScreenRenderEnv(
                bddl_file_name=self.bddl_file,
                use_camera_obs=False,
                ignore_done=True,
                hard_reset=False,
                camera_heights=self.camera_resolution,
                camera_widths=self.camera_resolution,
            )
            self._loaded_bddl_file = self.bddl_file
            self._loaded_resolution = self.camera_resolution
            self._snapshot_pristine_scene()

        self.episode_index = 0
        self._plans: dict[str, dict[str, Any]] = {}
        self._batch_plans = self._plans
        self.recording = False
        self.recorded_frames: list[dict[str, Any]] = []
        self.video_export: dict[str, Any] | None = None
        self._reset_sim_state(seed=self.seed, layout_id=self.layout_id)
        return self.capabilities()

    def _snapshot_pristine_scene(self) -> None:
        m = self.env.sim.model
        self._pristine_cam_pos = np.array(m.cam_pos, copy=True)
        self._pristine_cam_quat = np.array(m.cam_quat, copy=True)
        self._pristine_light_pos = np.array(m.light_pos, copy=True) if m.nlight > 0 else None
        self._pristine_light_diffuse = np.array(m.light_diffuse, copy=True) if m.nlight > 0 else None
        self._pristine_headlight_ambient = np.array(m.vis.headlight.ambient, copy=True)
        self._pristine_headlight_diffuse = np.array(m.vis.headlight.diffuse, copy=True)

    def _restore_pristine_scene(self) -> None:
        m = self.env.sim.model
        if hasattr(self, "_pristine_cam_pos"):
            m.cam_pos[:] = self._pristine_cam_pos
            m.cam_quat[:] = self._pristine_cam_quat
            if self._pristine_light_pos is not None and m.nlight > 0:
                m.light_pos[:] = self._pristine_light_pos
            if self._pristine_light_diffuse is not None and m.nlight > 0:
                m.light_diffuse[:] = self._pristine_light_diffuse
            m.vis.headlight.ambient[:] = self._pristine_headlight_ambient
            m.vis.headlight.diffuse[:] = self._pristine_headlight_diffuse

    def _apply_perturbation(self, pert: dict[str, Any] | None) -> None:
        if not isinstance(pert, dict) or not pert:
            return
        from scipy.spatial.transform import Rotation

        m = self.sim.model
        q_delta = pert.get("robot_init_qpos_delta")
        if q_delta is not None:
            delta_arr = np.asarray(q_delta, dtype=np.float64).reshape(7)
            cur_q = np.array([float(self.sim.data.qpos[idx]) for idx in self._arm_qpos_idx], dtype=np.float64)
            bounds = [self.limits[name] for name in ARM_JOINTS]
            new_q = np.array(
                [float(np.clip(val, lo, hi)) for val, (lo, hi) in zip(cur_q + delta_arr, bounds)],
                dtype=np.float64,
            )
            if callable(getattr(self.robot, "set_robot_joint_positions", None)):
                self.robot.set_robot_joint_positions(new_q)
            else:
                for idx, val in zip(self._arm_qpos_idx, new_q):
                    self.sim.data.qpos[idx] = val
            self.sim.forward()

        orbit = pert.get("camera_orbit")
        if isinstance(orbit, dict):
            cam_name = str(orbit.get("camera") or "agentview")
            mj_cam = dict(CAMERA_MAP).get(cam_name, cam_name)
            cid = int(m.camera_name2id(mj_cam))
            pivot = np.asarray(orbit.get("pivot", [0.0, 0.0, 0.8]), dtype=np.float64).reshape(3)
            yaw_deg = float(orbit.get("yaw_deg", 0.0))
            pitch_deg = float(orbit.get("pitch_deg", 0.0))
            dist_scale = float(orbit.get("distance_scale", 1.0))

            orig_pos = np.asarray(self.sim.data.cam_xpos[cid], dtype=np.float64).copy()
            orig_cmat = np.asarray(self.sim.data.cam_xmat[cid], dtype=np.float64).reshape(3, 3).copy()
            rel = (orig_pos - pivot) * dist_scale

            R_yaw = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
            horiz_right = np.array([orig_cmat[0, 0], orig_cmat[1, 0], 0.0], dtype=np.float64)
            r_norm = float(np.linalg.norm(horiz_right))
            horiz_right = horiz_right / r_norm if r_norm > 1e-6 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
            R_pitch = Rotation.from_rotvec(math.radians(pitch_deg) * horiz_right).as_matrix()
            R_orb = R_yaw @ R_pitch

            new_pos = pivot + R_orb @ rel
            new_cmat = R_orb @ orig_cmat
            xyzw = _matrix_to_quaternion_xyzw(new_cmat)
            m.cam_pos[cid] = new_pos
            m.cam_quat[cid] = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
            self.sim.forward()

        lighting = pert.get("lighting")
        if isinstance(lighting, dict):
            amb_s = float(lighting.get("ambient_scale", 1.0))
            dif_s = float(lighting.get("diffuse_scale", 1.0))
            m.vis.headlight.ambient[:] = np.clip(m.vis.headlight.ambient * amb_s, 0.0, 2.0)
            m.vis.headlight.diffuse[:] = np.clip(m.vis.headlight.diffuse * dif_s, 0.0, 2.0)
            if m.nlight > 0:
                m.light_diffuse[:] = np.clip(m.light_diffuse * dif_s, 0.0, 2.0)
                if "pos_delta" in lighting and lighting["pos_delta"] is not None:
                    p_delta = np.asarray(lighting["pos_delta"], dtype=np.float64).reshape(1, 3)
                    m.light_pos[:] = m.light_pos + p_delta
            self.sim.forward()

    def _reset_sim_state(self, *, seed: int, layout_id: int) -> int | None:
        self._restore_pristine_scene()
        self.env.seed(int(seed))
        self.env.reset()
        chosen_idx: int | None = None
        if self._init_states is not None and len(self._init_states) > 0:
            raw_idx = self.explicit_init_state_id if self.explicit_init_state_id is not None else int(layout_id)
            chosen_idx = int(raw_idx) % len(self._init_states)
            self.env.set_init_state(self._init_states[chosen_idx])
        self.sim = self.env.sim
        self.robot = self.env.env.robots[0]
        self.sim.forward()
        # Official LIBERO benchmark protocol (LIBERO README / lifelong/metric.py):
        # run 10 zero-action physics steps after set_init_state so objects settle
        # onto support surfaces under gravity before observation 0 (without advancing
        # actor step counters).
        settle_steps = max(0, int(getattr(self, "reset_settle_steps", 10)))
        if settle_steps > 0:
            zero_act = np.zeros(7, dtype=np.float64)
            for _ in range(settle_steps):
                self.env.step(zero_act)
            self.sim.forward()
        self._bind_mujoco_handles()
        self._apply_perturbation(self.perturbation)
        self._bind_mujoco_handles()

        sample = self.read()
        self.jaw_goal = float(sample["jaw_normalized"])
        self.jaw_servo_goal = float(sample["jaw_normalized"])
        self._apply_gripper_goal(self.jaw_servo_goal)
        self.sim.forward()

        self.control_hz = float(getattr(self.env.env, "control_freq", 20.0))
        self.physics_dt_s = float(self.sim.model.opt.timestep)
        self.substeps_per_control = max(1, int(round(1.0 / (self.control_hz * self.physics_dt_s))))
        self.control_step = 0
        self.native_actions_dispatched = 0
        self.evaluator_settle_steps = 0
        self._locked_terminal: dict[str, Any] | None = None
        self._plans.clear()
        self.video_export = None
        self.table_z_m = self._measure_table_z_m()
        self.camera_calibration = self._build_camera_calibration() if self.calibrated else None
        self.active_init_state_id = chosen_idx
        return chosen_idx

    def _bind_mujoco_handles(self) -> None:
        m = self.sim.model
        self._link0_bid = m.body_name2id("robot0_link0")
        self._grip_sid = m.site_name2id("gripper0_grip_site")
        self._arm_qpos_idx = [int(m.jnt_qposadr[m.joint_name2id(f"robot0_joint{i}")]) for i in range(1, 8)]
        self._arm_qvel_idx = [int(m.jnt_dofadr[m.joint_name2id(f"robot0_joint{i}")]) for i in range(1, 8)]
        self._finger_qpos_idx = [
            int(m.jnt_qposadr[m.joint_name2id("gripper0_finger_joint1")]),
            int(m.jnt_qposadr[m.joint_name2id("gripper0_finger_joint2")]),
        ]
        self._cam_ids = {pub: int(m.camera_name2id(mj)) for pub, mj in CAMERA_MAP}
        T_wb = np.eye(4, dtype=np.float64)
        T_wb[:3, :3] = self.sim.data.body_xmat[self._link0_bid].reshape(3, 3)
        T_wb[:3, 3] = self.sim.data.body_xpos[self._link0_bid]
        self._T_world_base = T_wb
        self._T_base_world = np.linalg.inv(T_wb)

    def _measure_table_z_m(self) -> float:
        arena = getattr(getattr(self.env.env, "model", None), "mujoco_arena", None)
        offset = getattr(arena, "table_offset", None)
        if offset is None:
            offset = getattr(self.env.env, "table_offset", None)
        if isinstance(offset, (list, tuple, np.ndarray)) and len(offset) >= 3:
            surface_world_z = float(offset[2])
        else:
            surface_world_z = 0.0
        return round(float(surface_world_z - float(self._T_world_base[2, 3])), 6)

    def _camera_intrinsics(self, mj_cam_id: int) -> np.ndarray:
        res = float(self.camera_resolution)
        fovy_deg = float(self.sim.model.cam_fovy[mj_cam_id])
        fy = 0.5 * res / math.tan(math.radians(fovy_deg) * 0.5)
        return np.array(
            [
                [fy, 0.0, 0.5 * res],
                [0.0, fy, 0.5 * res],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _camera_pose_in_base(self, pub_name: str) -> np.ndarray:
        cid = self._cam_ids[pub_name]
        T_wc = np.eye(4, dtype=np.float64)
        cmat = self.sim.data.cam_xmat[cid].reshape(3, 3)
        # Convert MuJoCo OpenGL camera axes (+X right, +Y up, -Z forward) to OpenCV (+X right, +Y down, +Z forward)
        T_wc[:3, :3] = cmat @ np.diag([1.0, -1.0, -1.0])
        T_wc[:3, 3] = self.sim.data.cam_xpos[cid]
        return self._T_base_world @ T_wc

    def _build_camera_calibration(self) -> dict[str, Any]:
        sample = self.read()
        T_base_hand = fk(self.model, sample["q_rad"], link=WRIST_CAMERA_PARENT_LINK)
        cameras: dict[str, dict[str, Any]] = {}
        for pub_name, _mj_name in CAMERA_MAP:
            cid = self._cam_ids[pub_name]
            K = self._camera_intrinsics(cid)
            T_base_cam = self._camera_pose_in_base(pub_name)
            entry: dict[str, Any] = {
                "K": K.tolist(),
                "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
                "distortion_model": "pinhole_no_distortion",
                "image_size": [self.camera_resolution, self.camera_resolution],
                "optical_axes": "+x right,+y down,+z forward",
                "calibration_source": "LIBERO MuJoCo simulator camera parameters (simulation-only)",
            }
            if pub_name == "wrist":
                T_link_cam = np.linalg.inv(T_base_hand) @ T_base_cam
                entry.update(
                    arm=self.arm,
                    link_name=WRIST_CAMERA_PARENT_LINK,
                    T_world_base=np.eye(4, dtype=np.float64).tolist(),
                    T_link_camera=T_link_cam.tolist(),
                )
            else:
                entry["T_world_camera"] = T_base_cam.tolist()
            cameras[pub_name] = entry
        return {
            "profile_id": "libero-franka-panda-pinhole-v1",
            "source_kind": "simulator_configuration",
            "simulation_only": True,
            "reference_frame_id": LIBERO_FRANKA_PANDA.frame_id,
            "joint_calibration": {
                "table_z_m": self.table_z_m,
                "source": "LIBERO MuJoCo arena surface offset relative to franka_base",
            },
            "cameras": cameras,
        }

    # ------------------------------------------------------------ basics ---
    @property
    def embodiment(self):
        return libero_franka_panda_profile(thresholds=self.motion_thresholds)

    @property
    def arm_layout(self) -> ArmLayout:
        override = getattr(self, "_arm_layout_override", None)
        if isinstance(override, ArmLayout):
            return override
        return ArmLayout.from_profile(self.embodiment)

    @arm_layout.setter
    def arm_layout(self, value: ArmLayout) -> None:
        self._arm_layout_override = value

    def capabilities(self) -> dict[str, Any]:
        prof = self.embodiment
        scale_dict = lambda s: {"min": s.min, "max": s.max, "default": s.default}  # noqa: E731
        arena = getattr(getattr(self.env.env, "model", None), "mujoco_arena", None)
        return {
            "backend": "libero_mujoco",
            "data_origin": "libero_mujoco_simulation",
            "simulation_only": True,
            "simulator": "LIBERO (robosuite MuJoCo)",
            "world_kind": "observed",
            "robot_model_id": prof.robot_model_id,
            "frame_id": prof.frame_id,
            "tcp_config_id": prof.tcp_config_id,
            "arms": list(prof.arms),
            "dual_arm": False,
            "max_simultaneous_arms": 1,
            "standalone_gripper": True,
            "observation_profile": self.observation_profile,
            "observation_profiles": [self.observation_profile],
            "cameras": [pub for pub, _ in CAMERA_MAP],
            "camera_resolution": [self.camera_resolution, self.camera_resolution],
            "modes": ["absolute"],
            "path_kinds": ["backend_free", "hold"],
            "max_waypoints": contract.MAX_WAYPOINTS,
            "max_timeout_s": prof.max_timeout_s,
            "default_timeout_s": prof.default_action_timeout_s,
            "limits_profile": prof.limits_profiles[0],
            "motion_execution_mode": "resampled",
            "gripper_after_motion": True,
            "gripper_phases": ["after_motion", "during_motion"],
            "orientation_modes": ["backend_free"],
            "arm_joint_names": {self.arm: list(ARM_JOINTS)},
            "arm_joint_limits_rad": {self.arm: {k: list(v) for k, v in self.limits.items()}},
            "motion_tolerance_defaults": {
                "position_tolerance_m": prof.default_tolerance.position_m,
                "orientation_tolerance_rad": prof.default_tolerance.orientation_rad,
            },
            "native_control_hz": self.control_hz,
            "physics_dt_s": self.physics_dt_s,
            "native_physics_steps_per_action": self.substeps_per_control,
            "bounded_wait": {
                "max_native_steps": contract.MAX_WAIT_STEPS,
                "native_control_hz": self.control_hz,
                "default_timeout_s": prof.default_wait_timeout_s,
                "semantics": "simulated hold of current Panda EE pose and continuous jaw target; each step consumes 1 native control step (50 ms sim time)",
            },
            "action_speed": {
                "available": True,
                "speed_scale": scale_dict(prof.speed_scale),
                "gripper_speed_scale": scale_dict(prof.gripper_speed_scale),
                "waypoint_speed_scales": {"available": False},
                "semantics": "scales closed-loop OSC_POSE step delta cap (default 1.0 -> 0.0325 m / 0.325 rad per 20 Hz step; max 2.0 -> 0.050 m / 0.500 rad)",
            },
            "tool_geometry": {
                "simulation_only": True,
                "source": "simulated LIBERO robosuite Franka Panda URDF & MuJoCo gripper0_grip_site",
                "tcp_link": TCP_LINK,
                "grip_site_to_pad_center_local_m": [pose_convention.GRASP_POINT_OFFSET_M, 0.0, 0.0],
                "grasp_center_reference": "pad_center_coincident_with_gripper0_grip_site",
                "contact_selection": grasp_contact.contact_capabilities(),
                "approach_axis_local": [0, 0, 1],
                "jaw_opening_axis_local": [1, 0, 0],
            },
            "gripper_semantics": {
                "goal_normalized": "0=closed (0.000 m commanded, ~0.0010 m empty pad contact), 1=open (0.080 m); continuous closed-loop PD aperture servo without binary snap",
                "contact_feedback": "unavailable",
                "during_motion": "linear ramp of continuous jaw target over start_fraction..end_fraction of the move stage",
            },
            "grip_guard": contract.grip_guard_capability(prof),
            "scene_calibration": {
                "table_z_m": self.table_z_m,
                "arena_type": type(arena).__name__ if arena is not None else None,
            },
            "task_metadata": {
                "suite_name": self.suite_name,
                "task_id": self.task_id,
                "task_name": self.task_name,
                "language_instruction": self.language_instruction,
                "init_state_id": self.active_init_state_id,
            },
            "clock_policy": "physics advances only during worker action execution; paused completely during agent reasoning",
            "cancellation": "checked before every 20 Hz control step; stop holds current measured EE pose",
            "collision_scope": "7-DoF URDF kinematic reachability check in preview/reach; no scene-object collision proof",
            "motion_preview": {"available": False},
            "unified_action": {
                "available": True,
                "requires_camera_calibration": False,
                "modes": ["preview", "execute", "inspect", "direct", "execute_if_clean", "reach"],
                "max_stages": prof.max_stages,
                "max_stages_per_arm": prof.stages_per_arm,
                "planning": "7-DoF URDF IK reachability check per move stage without advancing MuJoCo physics",
                "execution": "20 Hz closed-loop operational-space control via robosuite native OSC_POSE",
            },
            "reset": True,
            "native_evaluator": "privileged IPC only; never actor observation",
            "observer_video": True,
            "action_budget": {
                "native_step_limit": self.max_episode_steps,
                "source": "libero_episode_horizon",
            },
        }

    # -------------------------------------------------- state & observe ---
    def read(self) -> dict[str, Any]:
        q_arm = [float(self.sim.data.qpos[idx]) for idx in self._arm_qpos_idx]
        qdot_arm = [float(self.sim.data.qvel[idx]) for idx in self._arm_qvel_idx]
        f1 = float(self.sim.data.qpos[self._finger_qpos_idx[0]])
        f2 = float(self.sim.data.qpos[self._finger_qpos_idx[1]])
        aperture_m = max(0.0, f1 - f2)
        jaw_norm = float(np.clip(aperture_m / PANDA_FULL_OPEN_APERTURE_M, 0.0, 1.0))

        T_w_tcp = np.eye(4, dtype=np.float64)
        T_w_tcp[:3, :3] = self.sim.data.site_xmat[self._grip_sid].reshape(3, 3)
        T_w_tcp[:3, 3] = self.sim.data.site_xpos[self._grip_sid]
        T_b_tcp = self._T_base_world @ T_w_tcp
        return {
            "q_rad": q_arm,
            "qdot_rad_s": qdot_arm,
            "finger_q_m": [f1, f2],
            "aperture_m": aperture_m,
            "jaw_normalized": jaw_norm,
            "T_base_tcp": T_b_tcp,
            "position_m": [float(v) for v in T_b_tcp[:3, 3]],
            "quaternion_xyzw": _matrix_to_quaternion_xyzw(T_b_tcp[:3, :3]),
        }

    def robot_state(self, sample: dict[str, Any] | None = None, *, step: int | None = None) -> dict[str, Any]:
        s = self.read() if sample is None else sample
        return {
            "arms": {
                self.arm: {
                    "joint_names": list(ARM_JOINTS),
                    "q_rad": list(s["q_rad"]),
                    "qdot_rad_s": list(s["qdot_rad_s"]),
                    "source_kind": "measurement",
                    "ee_pose": {
                        "position_m": list(s["position_m"]),
                        "quaternion_xyzw": list(s["quaternion_xyzw"]),
                        "frame_id": LIBERO_FRANKA_PANDA.frame_id,
                        "tcp_config_id": LIBERO_FRANKA_PANDA.tcp_config_id,
                        "source_kind": "measurement",
                    },
                    "gripper": {
                        "goal_normalized": round(float(self.jaw_goal), 6),
                        "goal_source_kind": "command",
                        "command_normalized": round(float(self.jaw_goal), 6),
                        "command_source_kind": "command",
                        "joint_names": list(GRIPPER_JOINTS),
                        "position_m": list(s["finger_q_m"]),
                        "position_source_kind": "measurement",
                        "contact_feedback": None,
                        "contact_feedback_reason": "unavailable",
                    },
                }
            },
            "frame_id": LIBERO_FRANKA_PANDA.frame_id,
            "tcp_config_id": LIBERO_FRANKA_PANDA.tcp_config_id,
            "sim_step": self.control_step if step is None else int(step),
            "step_domain": STEP_DOMAIN,
        }

    def state(self) -> dict[str, Any]:
        return self.robot_state()

    def read_state(self) -> dict[str, Any]:
        return self.robot_state()

    def observe(self, *, video_only: bool = False) -> dict[str, Any]:
        from PIL import Image

        self.sim.forward()
        sample = self.read()
        step = self.control_step
        obs_id = "raw-" + uuid.uuid4().hex
        sub_dir = "video_frames" if video_only else "observations"
        directory = self.run_dir / sub_dir / obs_id
        directory.mkdir(parents=True, exist_ok=True)

        extent = float(self.sim.model.stat.extent)
        near = float(self.sim.model.vis.map.znear) * extent
        far = float(self.sim.model.vis.map.zfar) * extent

        cameras: list[dict[str, Any]] = []
        poses: dict[str, dict[str, Any]] = {}
        T_base_hand = fk(self.model, sample["q_rad"], link=WRIST_CAMERA_PARENT_LINK) if self.calibrated else None

        for pub_name, mj_name in CAMERA_MAP:
            rendered = self.sim.render(
                width=self.camera_resolution,
                height=self.camera_resolution,
                camera_name=mj_name,
                depth=self.want_depth and not video_only,
            )
            if self.want_depth and not video_only:
                raw_rgb, raw_dep = rendered
            else:
                raw_rgb, raw_dep = rendered, None
            # MuJoCo OpenGL offscreen buffers are bottom-up; flip vertically to top-down OpenCV raster.
            rgb = np.ascontiguousarray(raw_rgb[::-1], dtype=np.uint8)
            ext = "png" if self.image_settings.get("storage_format") == "png" else "jpg"
            img_path = directory / f"{pub_name}.{ext}"
            data, encoding = encode_image(Image.fromarray(rgb), self.image_settings, storage=True)
            img_path.write_bytes(data)
            frame_id = f"{obs_id}-{pub_name}"
            cam_entry: dict[str, Any] = {
                "camera_id": pub_name,
                "frame_id": frame_id,
                "rgb_path": str(img_path),
                "image_encoding": encoding,
                "width": self.camera_resolution,
                "height": self.camera_resolution,
                "pixel_transform": "identity",
                "sim_step": step,
                "step_domain": STEP_DOMAIN,
                "delivered_to_actor": False,
            }
            if raw_dep is not None:
                dep_flip = np.asarray(raw_dep[::-1], dtype=np.float32)
                if dep_flip.ndim == 3:
                    dep_flip = dep_flip[..., 0]
                z_m = near / (1.0 - np.clip(dep_flip, 0.0, 0.999999) * (1.0 - near / far))
                dep_fields = _save_depth_capture(
                    directory / f"{pub_name}_depth.npz",
                    z_m,
                    self.camera_resolution,
                    self.camera_resolution,
                )
                cam_entry.update(dep_fields)
            cameras.append(cam_entry)

            if self.calibrated and not video_only:
                T_base_cam = self._camera_pose_in_base(pub_name)
                if pub_name == "wrist" and T_base_hand is not None:
                    poses[pub_name] = {
                        "T_world_camera": T_base_cam.tolist(),
                        "T_world_link": T_base_hand.tolist(),
                        "T_world_base": np.eye(4, dtype=np.float64).tolist(),
                        "arm": self.arm,
                        "link_name": WRIST_CAMERA_PARENT_LINK,
                        "frame_id": frame_id,
                        "source_kind": "robot_kinematics",
                    }
                else:
                    poses[pub_name] = {
                        "T_world_camera": T_base_cam.tolist(),
                        "frame_id": frame_id,
                        "source_kind": "fixed_calibration",
                    }

        sim_time_s = round(step / self.control_hz, 4)
        frame_record = {
            "raw_observation_id": obs_id,
            "sim_step": step,
            "step_domain": STEP_DOMAIN,
            "sim_time_s": sim_time_s,
            "cameras": cameras,
        }
        if self.recording and (
            not self.recorded_frames or self.recorded_frames[-1]["sim_step"] != step
        ):
            self.recorded_frames.append(frame_record)
        if video_only:
            return frame_record

        raw_state = self.robot_state(sample, step=step)
        observation: dict[str, Any] = {
            "raw_observation_id": obs_id,
            "data_origin": "libero_mujoco_simulation",
            "world_kind": "observed",
            "observation_profile": self.observation_profile,
            "robot_model_id": LIBERO_FRANKA_PANDA.robot_model_id,
            "instruction": self.language_instruction,
            "cameras": cameras,
            "robot_state": raw_state if self.calibrated else actor_robot_state(raw_state, self.observation_profile, arms=(self.arm,)),
            "sim_step": step,
            "step_domain": STEP_DOMAIN,
            "sim_time_s": sim_time_s,
            "control_dt_s": round(1.0 / self.control_hz, 6),
            "captured_at": _utc_now(),
        }
        if self.calibrated and self.camera_calibration is not None:
            observation.update(
                camera_calibration=deepcopy(self.camera_calibration),
                robot_model=self.model,
                camera_pose_sample={
                    "reference_frame_id": LIBERO_FRANKA_PANDA.frame_id,
                    "sim_step": step,
                    "step_domain": STEP_DOMAIN,
                    "cameras": poses,
                },
            )
        return observation

    # ---------------------------------------------- IK & preview / reach ---
    def _effective_tolerances(self, path: dict[str, Any]) -> dict[str, Any]:
        prof = self.embodiment
        preset = path.get("tolerance_preset")
        base_tol = prof.tolerance_presets.get(preset, prof.default_tolerance) if preset else prof.default_tolerance
        pos_tol = float(path.get("position_tolerance_m", base_tol.position_m))
        ori_tol = float(path.get("orientation_tolerance_rad", base_tol.orientation_rad))
        stall_preset = path.get("stall_guard_preset") or "off"
        upward = path.get("max_upward_dz_m")
        if upward is None:
            if stall_preset == "contact_stop":
                upward = max(pos_tol, 0.040)
            else:
                upward = float(base_tol.max_upward_dz_m or 0.0)
        downward = float(path.get("max_downward_dz_m") or 0.0)
        div = path.get("max_tracking_divergence_m")
        if div is None:
            div = prof.stall_guard_presets_m.get(stall_preset)
        return {
            "tolerance_preset": preset or prof.default_tolerance_preset,
            "position_tolerance_m": round(pos_tol, 6),
            "orientation_tolerance_rad": round(ori_tol, 6),
            "max_upward_dz_m": round(float(upward), 6),
            "max_downward_dz_m": round(float(downward), 6),
            "stall_guard_preset": stall_preset,
            "max_tracking_divergence_m": None if div is None else round(float(div), 6),
            "checked_on": "measured MuJoCo gripper0_grip_site pose in franka_base",
        }

    def _solve_stage_ik(
        self,
        pose: dict[str, Any],
        seed_q: list[float],
        tolerances: dict[str, Any],
    ) -> tuple[list[float], dict[str, Any]]:
        T_target = _pose_to_matrix(pose)
        pos_tol = float(tolerances["position_tolerance_m"])
        ori_tol = float(tolerances["orientation_tolerance_rad"])
        ik_pos_tol = min(pos_tol, 1e-3)
        ik_ori_tol = min(ori_tol, 5e-3)

        if orientation_is_free(ori_tol):
            res = ik_position_priority(self.model, T_target, seed_q, position_tolerance_m=ik_pos_tol)
            pos_err = float(res["position_error_m"])
            ori_err = float(res["orientation_error_rad"])
            ok = pos_err <= pos_tol
            record = {
                "ok": ok,
                "solver": "urdf_dls_7dof",
                "solve": "position_only",
                "position_error_m": round(pos_err, 6),
                "orientation_error_rad": round(ori_err, 6),
                "iterations": int(res["iterations"]),
            }
            return [float(v) for v in res["q_rad"]], record

        res = ik(
            self.model,
            T_target,
            seed_q,
            position_tolerance_m=ik_pos_tol,
            orientation_tolerance_rad=ik_ori_tol,
        )
        pos_err = float(res["position_error_m"])
        ori_err = float(res["orientation_error_rad"])
        solve_kind = "pose"
        q_sol = [float(v) for v in res["q_rad"]]
        ok = pos_err <= pos_tol and ori_err <= ori_tol
        pos_prio_summary: dict[str, Any] | None = None

        if not ok:
            pp = ik_position_priority(self.model, T_target, seed_q, position_tolerance_m=ik_pos_tol)
            pp_q = [float(v) for v in pp["q_rad"]]
            T_pp = fk(self.model, pp_q, link=TCP_LINK)
            pp_pos_err = float(np.linalg.norm(T_target[:3, 3] - T_pp[:3, 3]))
            pp_ori_err = float(np.linalg.norm(_so3_log(T_target[:3, :3] @ T_pp[:3, :3].T)))
            pos_prio_summary = {
                "quaternion_xyzw": [round(v, 5) for v in _matrix_to_quaternion_xyzw(T_pp[:3, :3])],
                "position_error_m": round(pp_pos_err, 6),
                "orientation_error_rad": round(pp_ori_err, 6),
            }
            if pp_pos_err <= pos_tol and pp_ori_err <= ori_tol:
                ok = True
                solve_kind = "position_priority"
                q_sol = pp_q
                pos_err, ori_err = pp_pos_err, pp_ori_err

        record = {
            "ok": ok,
            "solver": "urdf_dls_7dof",
            "solve": solve_kind,
            "position_error_m": round(pos_err, 6),
            "orientation_error_rad": round(ori_err, 6),
            "iterations": int(res["iterations"]),
        }
        if not ok and pos_prio_summary is not None:
            record["position_priority"] = pos_prio_summary
        return q_sol, record

    @staticmethod
    def _estimate_move_steps(p_start: np.ndarray, R_start: np.ndarray, T_target: np.ndarray, speed_scale: float) -> int:
        dist = float(np.linalg.norm(T_target[:3, 3] - p_start))
        ang = float(np.linalg.norm(_so3_log(T_target[:3, :3] @ R_start.T)))
        eff = max(0.1, float(speed_scale))
        # Match actual robosuite 20 Hz PD OSC_POSE cruise speed (~0.015 m/step) + exponential convergence tail.
        step_pos = min(0.030, 0.015 * eff)
        step_ori = min(0.35, 0.18 * eff)
        n_pos = int(math.ceil(dist / max(1e-4, step_pos)))
        n_ori = int(math.ceil(ang / max(1e-4, step_ori)))
        return max(12, min(160, max(n_pos, n_ori) + 14))

    @staticmethod
    def _sample_entry(
        *,
        stage_id: str,
        native_step: int,
        q_arm: list[float],
        jaw_goal: float,
        phase: str,
        model: dict[str, Any],
    ) -> dict[str, Any]:
        half = 0.5 * PANDA_FULL_OPEN_APERTURE_M * float(np.clip(jaw_goal, 0.0, 1.0))
        links = fk_all(model, q_arm, finger_q_m=(half, -half))
        T_tcp = np.asarray(links[TCP_LINK], dtype=np.float64)
        return {
            "stage_id": stage_id,
            "q_rad": [float(v) for v in q_arm],
            "tcp_position_m": T_tcp[:3, 3].tolist(),
            "native_step": int(native_step),
            "execution_state": "not_executed",
            "arms": {
                ARM_ID: {
                    "q_rad": [float(v) for v in q_arm],
                    "jaw_goal_normalized": round(float(jaw_goal), 4),
                    "phase": phase,
                    "link_positions_m": {k: np.asarray(v)[:3, 3].tolist() for k, v in links.items()},
                    f"{TCP_LINK}_pose": {
                        "position_m": T_tcp[:3, 3].tolist(),
                        "rotation_matrix": T_tcp[:3, :3].tolist(),
                    },
                }
            },
        }

    @staticmethod
    def _batch_digest(action: dict[str, Any]) -> str:
        stages = [
            {key: value for key, value in stage.items() if key != "action"}
            | (
                {
                    "action": {
                        key: value
                        for key, value in stage["action"].items()
                        if key not in ("action_id", "based_on_observation_id")
                    }
                }
                if "action" in stage
                else {}
            )
            for stage in action["stages"]
        ]
        batch = {
            key: action[key]
            for key in ("checkpoint_after_gripper", *contract.GRIP_GUARD_FIELDS, *CARRY_SLIP_GUARD_FIELDS)
            if key in action
        }
        return digest(
            {
                "stages": stages,
                "frame_id": action["frame_id"],
                "tcp_config_id": action["tcp_config_id"],
                "limits_profile": action["limits_profile"],
                **batch,
            }
        )

    def reach_batch(self, action: dict[str, Any]) -> dict[str, Any]:
        zero = {"physics_advanced": False, "executed_native_steps": 0}
        try:
            contract.validate_action(action, self.embodiment)
            start = self.read()
        except ValueError as error:
            return {"status": "failed", "reason": str(error), **zero}
        stages = action["stages"]
        if any(stage.get("kind") != "move" for stage in stages):
            return {"status": "failed", "reason": "reach_accepts_move_stages_only", **zero}

        seed = list(start["q_rad"])
        results: dict[str, Any] = {}
        for stage in stages:
            leaf = stage["action"]
            pose = leaf["path"]["waypoints"][0][self.arm]
            tols = self._effective_tolerances(leaf["path"])
            q_sol, record = self._solve_stage_ik(pose, seed, tols)
            T_reached = fk(self.model, q_sol, link=TCP_LINK)
            reach_tols = {k: v for k, v in tols.items() if k not in REACH_EXCLUDED_TOLERANCES}
            reach_tols["checked_on"] = "FK of 7-DoF URDF IK solution (nothing moves)"
            entry: dict[str, Any] = {
                "ok": bool(record["ok"]),
                "ik": record,
                "tolerances": reach_tols,
                "reached": {
                    "position_m": [round(float(v), 4) for v in T_reached[:3, 3]],
                    "quaternion_xyzw": [round(float(v), 5) for v in _matrix_to_quaternion_xyzw(T_reached[:3, :3])],
                    "approach_axis": [round(float(v), 4) for v in T_reached[:3, 2]],
                },
            }
            if record["ok"]:
                entry["q_rad"] = [round(float(v), 4) for v in q_sol]
                entry["joint_travel_rad"] = [round(abs(float(a) - float(b)), 4) for a, b in zip(q_sol, seed)]
            results[stage["id"]] = entry
        return {
            "status": "solved",
            "stage_count": len(stages),
            "reachable_stage_count": sum(1 for item in results.values() if item["ok"]),
            "stage_results": results,
            **zero,
        }

    def preview_batch(self, action: dict[str, Any]) -> dict[str, Any]:
        zero = {"physics_advanced": False, "executed_native_steps": 0}
        prof = self.embodiment
        try:
            contract.validate_action(action, prof)
            validate_batch_carry_slip_guard(action, **contract.guard_parameters(prof))
            stages = order_stages(action["stages"])
            start = self.read()
        except ValueError as error:
            return {"status": "failed", "reason": str(error), **zero}

        q_cur = list(start["q_rad"])
        T_cur = np.asarray(start["T_base_tcp"], dtype=np.float64)
        jaw_cur = float(self.jaw_goal)
        cursor = 0
        records: dict[str, dict[str, Any]] = {}
        costs: dict[str, dict[str, Any]] = {}
        checkpoints: list[dict[str, Any]] = []
        previous_goals: dict[str, float] = {}
        guard_params = contract.guard_parameters(prof)

        samples: list[dict[str, Any]] = [
            self._sample_entry(
                stage_id=stages[0]["id"],
                native_step=0,
                q_arm=q_cur,
                jaw_goal=jaw_cur,
                phase="initial_measured_state",
                model=self.model,
            )
        ]

        def _jaw_guard(stage_obj: dict[str, Any], goal_val: float) -> dict[str, Any] | None:
            guard = stage_grip_guard(stage_obj, action, float(goal_val), **guard_params)
            if guard is None:
                return None
            return dict(
                guard,
                empty_close_aperture_m=PANDA_EMPTY_CLOSE_APERTURE_M,
                source=(
                    "stage"
                    if any(
                        stage_obj.get(f) is not None
                        for f in (*contract.GRIP_GUARD_FIELDS, *CARRY_SLIP_GUARD_FIELDS)
                    )
                    else "batch_default"
                ),
            )

        for stage in stages:
            sid, kind = stage["id"], stage["kind"]
            begin_step = cursor
            previous_goals[sid] = jaw_cur
            if kind == "observe":
                checkpoints.append({"stage_id": sid, "cursor": cursor, "reason": stage["reason"]})
                records[sid] = {"kind": kind, "arm": stage.get("arm")}
            elif kind == "move":
                leaf = stage["action"]
                pose = leaf["path"]["waypoints"][0][self.arm]
                tols = self._effective_tolerances(leaf["path"])
                q_target, ik_rec = self._solve_stage_ik(pose, q_cur, tols)
                if not ik_rec["ok"]:
                    return {
                        "status": "failed",
                        "reason": "planning_failed",
                        "failed_stage_id": sid,
                        "failed_arm": self.arm,
                        "failure_feedback": {"ik": ik_rec, "effective_tolerances": tols},
                        **zero,
                    }
                T_target = _pose_to_matrix(pose)
                steps = self._estimate_move_steps(
                    T_cur[:3, 3],
                    T_cur[:3, :3],
                    T_target,
                    float(leaf.get("speed_scale", 1.0)),
                )
                event = next(iter(leaf.get("gripper_events") or []), None)
                start_jaw = jaw_cur
                if event is not None:
                    jaw_cur = float(event["goal_normalized"])
                n_pts = min(steps, PATH_OVERLAY_SAMPLES)
                fracs = np.linspace(1.0 / max(1, n_pts), 1.0, n_pts)
                q_arr_0 = np.asarray(q_cur, dtype=np.float64)
                q_arr_1 = np.asarray(q_target, dtype=np.float64)
                for frac in fracs:
                    step_idx = max(1, int(round(frac * steps)))
                    q_interp = (q_arr_0 + float(frac) * (q_arr_1 - q_arr_0)).tolist()
                    if event is not None:
                        s_f = float(event["start_fraction"])
                        e_f = float(event["end_fraction"])
                        if e_f <= s_f:
                            j_interp = jaw_cur if frac >= s_f else start_jaw
                        else:
                            alpha = float(np.clip((frac - s_f) / (e_f - s_f), 0.0, 1.0))
                            j_interp = start_jaw + alpha * (jaw_cur - start_jaw)
                    else:
                        j_interp = jaw_cur
                    samples.append(
                        self._sample_entry(
                            stage_id=sid,
                            native_step=begin_step + step_idx,
                            q_arm=q_interp,
                            jaw_goal=j_interp,
                            phase="move",
                            model=self.model,
                        )
                    )
                rec: dict[str, Any] = {
                    "kind": kind,
                    "arm": self.arm,
                    "q_rad": q_target,
                    "pose": pose,
                    "ik": ik_rec,
                    "start_q": list(q_cur),
                    "tolerances": tols,
                    "joint_travel_rad": [round(abs(float(b) - float(a)), 4) for a, b in zip(q_cur, q_target)],
                }
                if event is not None:
                    rec["goal_normalized"] = jaw_cur
                    rec["gripper_motion"] = {
                        "start_fraction": float(event["start_fraction"]),
                        "end_fraction": float(event["end_fraction"]),
                        "goal_normalized": jaw_cur,
                    }
                    guard = _jaw_guard(stage, jaw_cur)
                    if guard is not None:
                        rec["grip_guard"] = guard
                records[sid] = rec
                cursor += steps
                q_cur = q_target
                T_cur = fk(self.model, q_cur, link=TCP_LINK)
            else:
                goal = float(stage["action"]["gripper_events"][0]["goal_normalized"])
                g_scale = float(stage["action"].get("gripper_speed_scale", 1.0))
                jaw_steps = max(4, int(math.ceil(abs(goal - jaw_cur) / max(0.05, 0.25 * g_scale))) + 2)
                cursor += jaw_steps
                jaw_cur = goal
                rec = {"kind": kind, "arm": self.arm, "goal_normalized": goal}
                guard = _jaw_guard(stage, goal)
                if guard is not None:
                    rec["grip_guard"] = guard
                records[sid] = rec
                samples.append(
                    self._sample_entry(
                        stage_id=sid,
                        native_step=cursor,
                        q_arm=q_cur,
                        jaw_goal=jaw_cur,
                        phase="gripper",
                        model=self.model,
                    )
                )

            costs[sid] = {
                "arm": self.arm if kind != "observe" else stage.get("arm"),
                "kind": kind,
                "start_native_step": begin_step,
                "end_native_step": cursor,
                "native_steps": cursor - begin_step,
                **(
                    {"planned_ik_residual_m": records[sid]["ik"]["position_error_m"]}
                    if kind == "move"
                    else {}
                ),
            }

        pauses = jaw_pauses(action, stages, records, costs, checkpoints, previous_goals)
        plan_id = "libero-batch-" + uuid.uuid4().hex
        while len(self._plans) >= 64:
            self._plans.pop(next(iter(self._plans)))
        self._plans[plan_id] = {
            "digest": self._batch_digest(action),
            "created_at": time.monotonic(),
            "status": "planned",
            "order": [stage["id"] for stage in stages],
            "records": records,
            "index": 0,
            "cursor": 0,
            "revision": 0,
            "start_q": list(start["q_rad"]),
            "completed": [],
            "jaw_pauses": pauses,
            "armed_grip_guard": {},
            "nominal_steps": cursor,
        }
        return {
            "status": "planned",
            "plan_id": plan_id,
            **zero,
            "planned_native_steps": cursor,
            "nominal_sim_time_s": round(cursor / self.control_hz, 4),
            "stage_costs": costs,
            "stage_plans": {
                key: {
                    k: v
                    for k, v in val.items()
                    if k
                    in (
                        "q_rad",
                        "ik",
                        "tolerances",
                        "joint_travel_rad",
                        "goal_normalized",
                        "grip_guard",
                        "gripper_motion",
                    )
                }
                for key, val in records.items()
                if val["kind"] != "observe"
            },
            "checkpoints": checkpoints,
            "trajectory_data": {
                "source_kind": "candidate_command_trajectory_not_measurement",
                "samples": samples,
                "frame_id": LIBERO_FRANKA_PANDA.frame_id,
                "tcp_config_id": LIBERO_FRANKA_PANDA.tcp_config_id,
                "joint_names": {self.arm: list(ARM_JOINTS)},
                "robot_model": self.model,
                "T_world_bases": {self.arm: np.eye(4, dtype=np.float64).tolist()},
                "camera_start_q_rad": {self.arm: list(start["q_rad"])},
                "execution_binding": "20 Hz closed-loop OSC_POSE toward verified stage targets",
            },
            "execution_semantics": "20 Hz closed-loop robosuite OSC_POSE per stage; stops on failed arrival or guard pause",
        }

    # ------------------------------------------ closed-loop 20 Hz motion ---
    def _apply_gripper_goal(self, goal_normalized: float) -> None:
        g = float(np.clip(goal_normalized, 0.0, 1.0))
        self.robot.gripper.current_action = np.array([2.0 * g - 1.0, 1.0 - 2.0 * g], dtype=np.float64)

    def _step_osc(
        self,
        target_T_base: np.ndarray,
        *,
        jaw_servo_goal: float,
        speed_scale: float = 1.0,
        run: dict[str, Any] | None = None,
        kind: str = "motion",
        count_actor_step: bool = True,
    ) -> dict[str, Any]:
        cur = self.read()
        T_cur = np.asarray(cur["T_base_tcp"], dtype=np.float64)
        R_wb = self._T_world_base[:3, :3]

        pos_err_base = target_T_base[:3, 3] - T_cur[:3, 3]
        pos_err_world = R_wb @ pos_err_base

        R_cur_world = R_wb @ T_cur[:3, :3]
        R_tgt_world = R_wb @ target_T_base[:3, :3]
        rot_err_world = _so3_log(R_tgt_world @ R_cur_world.T)

        cap = min(1.0, max(0.08, 0.65 * float(speed_scale)))
        raw_pos = pos_err_world / 0.05
        raw_ori = rot_err_world / 0.50
        pos_mag = float(np.max(np.abs(raw_pos))) if raw_pos.size else 0.0
        if pos_mag > cap:
            raw_pos = raw_pos * (cap / pos_mag)
        ori_mag = float(np.max(np.abs(raw_ori))) if raw_ori.size else 0.0
        if ori_mag > cap:
            raw_ori = raw_ori * (cap / ori_mag)

        self.jaw_servo_goal = float(np.clip(jaw_servo_goal, 0.0, 1.0))
        self._apply_gripper_goal(self.jaw_servo_goal)
        action_vec = np.array([*raw_pos, *raw_ori, 0.0], dtype=np.float64)
        self.env.step(action_vec)
        self.sim.forward()
        self.control_step += 1

        if count_actor_step:
            self.native_actions_dispatched += 1
            if self.stop_on_success and self._locked_terminal is None and bool(self.env.check_success()):
                self._locked_terminal = {
                    "ended": True,
                    "success": True,
                    "reason": "native_success",
                    "native_action_count": self.native_actions_dispatched,
                    "native_step_limit": self.max_episode_steps,
                }
            elif self._locked_terminal is None and self.native_actions_dispatched >= self.max_episode_steps:
                succ = bool(self.env.check_success())
                self._locked_terminal = {
                    "ended": True,
                    "success": succ,
                    "reason": "native_step_limit",
                    "native_action_count": self.native_actions_dispatched,
                    "native_step_limit": self.max_episode_steps,
                }

        after = self.read()
        if run is not None:
            run["commands"] += 1
            run["dispatched"].append(
                {
                    "step": self.control_step,
                    "kind": kind,
                    "q_rad": list(after["q_rad"]),
                    "gripper_goal_normalized": round(float(self.jaw_servo_goal), 6),
                    "osc_action": [round(float(x), 5) for x in action_vec],
                }
            )
            run["boundary"].append(
                {
                    "sim_step": self.control_step,
                    "t_s": round(self.control_step / self.control_hz, 4),
                    "state": self.robot_state(after, step=self.control_step),
                }
            )
        if self.recording and self.control_step % 4 == 0:
            self.observe(video_only=True)
        return after

    def hold(self, run: dict[str, Any] | None = None, *, settle_steps: int = 2) -> dict[str, Any]:
        snap = self.read()
        T_hold = np.asarray(snap["T_base_tcp"], dtype=np.float64)
        steps_taken = 0
        for _ in range(max(0, int(settle_steps))):
            if self.step_budget_remaining() == 0:
                break
            after = self._step_osc(
                T_hold,
                jaw_servo_goal=self.jaw_servo_goal,
                speed_scale=0.5,
                run=run,
                kind="hold",
            )
            steps_taken += 1
            if max(abs(v) for v in after["qdot_rad_s"]) < 0.03:
                break
        return {"confirmed": True, "settling_steps": steps_taken}

    def step_budget_remaining(self) -> int:
        return max(0, self.max_episode_steps - self.native_actions_dispatched)

    def episode_terminal(self) -> dict[str, Any]:
        if self._locked_terminal is not None:
            return deepcopy(self._locked_terminal)
        return {
            "ended": False,
            "success": None,
            "reason": None,
            "native_action_count": self.native_actions_dispatched,
            "native_step_limit": self.max_episode_steps,
        }

    @staticmethod
    def _pose_errors(cur: dict[str, Any], target_T: np.ndarray) -> dict[str, float]:
        T_cur = np.asarray(cur["T_base_tcp"], dtype=np.float64)
        diff = T_cur[:3, 3] - target_T[:3, 3]
        dx, dy, dz = float(diff[0]), float(diff[1]), float(diff[2])
        xy_err = math.hypot(dx, dy)
        pos_err = float(np.linalg.norm(diff))
        ori_err = float(np.linalg.norm(_so3_log(target_T[:3, :3] @ T_cur[:3, :3].T)))
        approach_axis = target_T[:3, 2]
        shortfall = float(np.dot(target_T[:3, 3] - T_cur[:3, 3], approach_axis))
        return {
            "dx_m": round(dx, 6),
            "dy_m": round(dy, 6),
            "dz_m": round(dz, 6),
            "xy_error_m": round(xy_err, 6),
            "position_error_m": round(pos_err, 6),
            "orientation_error_rad": round(ori_err, 6),
            "approach_shortfall_m": round(shortfall, 6),
        }

    @staticmethod
    def _arrival_ok(errs: dict[str, float], tols: dict[str, Any]) -> tuple[bool, bool]:
        pos_tol = float(tols["position_tolerance_m"])
        ori_tol = float(tols["orientation_tolerance_rad"])
        up_dz = float(tols.get("max_upward_dz_m") or 0.0)
        down_dz = float(tols.get("max_downward_dz_m") or 0.0)
        ori_ok = orientation_is_free(ori_tol) or errs["orientation_error_rad"] <= ori_tol
        if not ori_ok:
            return False, False
        if errs["position_error_m"] <= pos_tol:
            return True, False
        if (up_dz > 0.0 or down_dz > 0.0) and errs["xy_error_m"] <= pos_tol:
            dz = errs["dz_m"]
            if -(pos_tol + down_dz) <= dz <= max(pos_tol, up_dz):
                return True, True
        return False, False

    @staticmethod
    def _contact_summary(
        cur: dict[str, Any],
        target_T: np.ndarray,
        errs: dict[str, float],
        tols: dict[str, Any],
        completed_steps: int,
        planned_steps: int,
    ) -> dict[str, Any]:
        pos_tol = float(tols["position_tolerance_m"])
        return {
            "measured_position_m": [round(float(v), 5) for v in cur["position_m"]],
            "target_position_m": [round(float(v), 5) for v in target_T[:3, 3]],
            "vertical_shortfall_m": errs["dz_m"],
            "horizontal_offset_m": errs["xy_error_m"],
            "approach_shortfall_m": errs["approach_shortfall_m"],
            "blocked_above_target": errs["dz_m"] > pos_tol,
            "completed_steps": int(completed_steps),
            "planned_steps": int(planned_steps),
        }

    def _gripper_stage(
        self,
        run: dict[str, Any],
        goal_normalized: float,
        gripper_speed_scale: float = 1.0,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        goal = float(np.clip(goal_normalized, 0.0, 1.0))
        self.jaw_goal = goal
        start_sample = self.read()
        T_hold = np.asarray(start_sample["T_base_tcp"], dtype=np.float64)
        max_step = max(0.05, 0.25 * float(gripper_speed_scale))
        ramp_steps = max(4, int(math.ceil(abs(goal - self.jaw_servo_goal) / max_step)))
        max_total = ramp_steps + 18

        prev_ap = float(start_sample["aperture_m"])
        settled_streak = 0
        jaw_settled = False
        steps_used = 0
        interrupt_reason: str | None = None

        for step_idx in range(max_total):
            interrupt_reason = self._check_interrupt(run)
            if interrupt_reason:
                break
            delta = goal - self.jaw_servo_goal
            if abs(delta) <= max_step:
                next_g = goal
            else:
                next_g = self.jaw_servo_goal + math.copysign(max_step, delta)
            after = self._step_osc(T_hold, jaw_servo_goal=next_g, speed_scale=0.5, run=run, kind="gripper")
            steps_used += 1
            cur_ap = float(after["aperture_m"])
            if abs(next_g - goal) < 1e-6 and step_idx >= 3:
                if abs(cur_ap - prev_ap) < 4e-4:
                    settled_streak += 1
                    if settled_streak >= 2:
                        jaw_settled = True
                        break
                else:
                    settled_streak = 0
            prev_ap = cur_ap

        end_sample = self.read()
        info = {
            "goal_normalized": round(goal, 6),
            "target_aperture_m": round(goal * PANDA_FULL_OPEN_APERTURE_M, 6),
            "measured_opening_m": round(float(end_sample["aperture_m"]), 6),
            "planned_steps": ramp_steps,
            "completed_steps": steps_used,
            "started": steps_used > 0,
            "command_complete": abs(self.jaw_servo_goal - goal) < 1e-6,
            "jaw_settled": jaw_settled,
        }
        if interrupt_reason:
            return False, interrupt_reason, info
        return True, None, info

    def _move_stage(
        self,
        run: dict[str, Any],
        pose: dict[str, Any],
        path: dict[str, Any],
        *,
        speed_scale: float = 1.0,
        during_jaw: dict[str, Any] | None = None,
        gripper_speed_scale: float = 1.0,
        stage_id: str = "move",
        carry_slip_monitor: Any = None,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        tols = self._effective_tolerances(path)
        T_target = _pose_to_matrix(pose)
        start = self.read()
        T_start = np.asarray(start["T_base_tcp"], dtype=np.float64)
        total_dist = float(np.linalg.norm(T_target[:3, 3] - T_start[:3, 3]))
        planned_steps = self._estimate_move_steps(T_start[:3, 3], T_start[:3, :3], T_target, speed_scale)
        max_steps = max(40, min(240, int(round(planned_steps * 2.4))))

        stall_preset = tols["stall_guard_preset"]
        div_thresh = tols["max_tracking_divergence_m"]
        stall_vel_thresh = float(self.motion_thresholds.get("stall_guard_step_velocity_threshold_m", 0.001))
        stall_needed = max(4, int(self.motion_thresholds.get("stall_guard_consecutive_steps", 6)))

        start_jaw = float(self.jaw_servo_goal)
        end_jaw = float(during_jaw["goal_normalized"]) if during_jaw is not None else start_jaw
        if during_jaw is not None:
            self.jaw_goal = end_jaw

        prev_pos = np.asarray(start["position_m"], dtype=np.float64)
        stall_streak = 0
        steps_used = 0
        contact_stopped = False
        interrupt_reason: str | None = None
        slip_trigger: dict[str, Any] | None = None
        cur = start

        for step_idx in range(max_steps):
            interrupt_reason = self._check_interrupt(run)
            if interrupt_reason:
                break

            if during_jaw is not None:
                step_frac = float(step_idx + 1) / float(max(1, planned_steps))
                rem_dist = float(np.linalg.norm(T_target[:3, 3] - np.asarray(cur["position_m"], dtype=np.float64)))
                dist_frac = 1.0 - (rem_dist / total_dist) if total_dist > 0.01 else step_frac
                prog = float(np.clip(max(step_frac, dist_frac), 0.0, 1.0))
                s_f = float(during_jaw["start_fraction"])
                e_f = float(during_jaw["end_fraction"])
                if e_f <= s_f:
                    target_g = end_jaw if prog >= s_f else start_jaw
                else:
                    alpha = float(np.clip((prog - s_f) / (e_f - s_f), 0.0, 1.0))
                    target_g = start_jaw + alpha * (end_jaw - start_jaw)
                max_j_step = max(0.05, 0.25 * float(gripper_speed_scale))
                j_delta = target_g - self.jaw_servo_goal
                jaw_cmd = target_g if abs(j_delta) <= max_j_step else self.jaw_servo_goal + math.copysign(max_j_step, j_delta)
            else:
                jaw_cmd = self.jaw_servo_goal

            cur = self._step_osc(
                T_target,
                jaw_servo_goal=jaw_cmd,
                speed_scale=speed_scale,
                run=run,
                kind="motion",
            )
            steps_used += 1

            if carry_slip_monitor is not None:
                slip_trigger = BatchScheduler.check_carry_slip_step(
                    carry_slip_monitor,
                    stage_id,
                    steps_used - 1,
                    float(cur["aperture_m"]),
                )
                if slip_trigger is not None:
                    break

            cur_pos = np.asarray(cur["position_m"], dtype=np.float64)
            step_move = float(np.linalg.norm(cur_pos - prev_pos))
            prev_pos = cur_pos
            errs = self._pose_errors(cur, T_target)
            arrived, vertical_window_used = self._arrival_ok(errs, tols)
            if arrived and not vertical_window_used and (
                step_move < 0.0025 or errs["position_error_m"] <= 0.5 * float(tols["position_tolerance_m"])
            ):
                break

            if div_thresh is not None and stall_preset != "off" and step_idx >= 4:
                if errs["position_error_m"] > max(float(tols["position_tolerance_m"]), float(div_thresh)) and step_move < stall_vel_thresh:
                    stall_streak += 1
                    if stall_streak >= stall_needed:
                        hold_conf = self.hold(run, settle_steps=2)
                        cur = self.read()
                        errs = self._pose_errors(cur, T_target)
                        summary = self._contact_summary(cur, T_target, errs, tols, steps_used, planned_steps)
                        if stall_preset == "contact_stop":
                            ok_contact, v_used = self._arrival_ok(errs, tols)
                            if ok_contact:
                                contact_stopped = True
                                vertical_window_used = v_used
                                fields: dict[str, Any] = {
                                    **errs,
                                    "completed_steps": steps_used,
                                    "contact_stopped": True,
                                    "contact_block_summary": summary,
                                    "hold_confirmation": hold_conf,
                                }
                                if vertical_window_used:
                                    fields["vertical_compliance_accepted"] = True
                                if during_jaw is not None:
                                    ok_j, r_j, j_info = self._gripper_stage(run, end_jaw, gripper_speed_scale)
                                    fields["gripper_execution"] = dict(j_info, finished_at_held_pose=True)
                                    if not ok_j:
                                        return False, r_j, fields
                                return True, None, fields
                        return False, "tracking_error", {
                            **errs,
                            "completed_steps": steps_used,
                            "contact_block_summary": summary,
                            "hold_confirmation": hold_conf,
                        }
                else:
                    stall_streak = 0

        if slip_trigger is not None:
            errs = self._pose_errors(cur, T_target)
            return False, contract.CARRY_SLIP_GUARD_FIELDS[0], {
                **errs,
                "completed_steps": steps_used,
                "_carry_slip_trigger": slip_trigger,
            }

        if interrupt_reason:
            errs = self._pose_errors(cur, T_target)
            return False, interrupt_reason, {**errs, "completed_steps": steps_used}

        if during_jaw is not None and abs(self.jaw_servo_goal - end_jaw) > 1e-6:
            ok_j, r_j, j_info = self._gripper_stage(run, end_jaw, gripper_speed_scale)
            cur = self.read()
            if not ok_j:
                errs = self._pose_errors(cur, T_target)
                return False, r_j, {**errs, "completed_steps": steps_used, "gripper_execution": j_info}
        elif during_jaw is not None:
            # Settle jaw briefly at arrival
            ok_j, r_j, j_info = self._gripper_stage(run, end_jaw, gripper_speed_scale)
            cur = self.read()
            if not ok_j:
                errs = self._pose_errors(cur, T_target)
                return False, r_j, {**errs, "completed_steps": steps_used, "gripper_execution": j_info}
        else:
            j_info = None

        errs = self._pose_errors(cur, T_target)
        arrived, vertical_window_used = self._arrival_ok(errs, tols)
        fields = {
            **errs,
            "completed_steps": steps_used,
        }
        if j_info is not None:
            fields["gripper_execution"] = j_info
        if arrived:
            if vertical_window_used:
                fields["vertical_compliance_accepted"] = True
            return True, None, fields

        hold_conf = self.hold(run, settle_steps=2)
        cur = self.read()
        errs = self._pose_errors(cur, T_target)
        arrived_after_hold, vertical_window_used = self._arrival_ok(errs, tols)
        fields.update(errs)
        if arrived_after_hold:
            if vertical_window_used:
                fields["vertical_compliance_accepted"] = True
            return True, None, fields

        fields["contact_block_summary"] = self._contact_summary(cur, T_target, errs, tols, steps_used, planned_steps)
        fields["hold_confirmation"] = hold_conf
        return False, "tracking_error", fields

    # ----------------------------------------------------- execute entry ---
    def _new_run(self, cancellation: threading.Event | None, timeout_s: float) -> dict[str, Any]:
        snap = self.read()
        return {
            "t0": time.monotonic(),
            "deadline": time.monotonic() + float(timeout_s),
            "cancellation": cancellation,
            "commands": 0,
            "dispatched": [],
            "boundary": [
                {
                    "sim_step": self.control_step,
                    "t_s": round(self.control_step / self.control_hz, 4),
                    "state": self.robot_state(snap, step=self.control_step),
                }
            ],
        }

    def _check_interrupt(self, run: dict[str, Any]) -> str | None:
        cancel = run.get("cancellation")
        if cancel is not None and cancel.is_set():
            return "cancelled"
        if self._locked_terminal is not None and self._locked_terminal.get("ended"):
            return str(self._locked_terminal.get("reason") or "native_step_limit")
        if self.step_budget_remaining() <= 0:
            return "native_step_limit"
        if time.monotonic() >= float(run["deadline"]):
            return "action_timeout"
        return None

    def _rejected(self, reason: str, *, status: str = "rejected", **extra: Any) -> dict[str, Any]:
        return {
            "status": status,
            "reason": reason,
            "physics_advanced": False,
            "executed_steps": 0,
            "executed_native_steps": 0,
            "completed_native_steps": 0,
            "executed_control_steps": 0,
            "execution_interval": {
                "start_sim_step": self.control_step,
                "end_sim_step": self.control_step,
                "step_domain": STEP_DOMAIN,
            },
            "episode_terminal": self.episode_terminal(),
            **extra,
        }

    def _grip_facts(self, begin_sample: dict[str, Any], end_sample: dict[str, Any] | None) -> dict[str, Any]:
        end = self.read() if end_sample is None else end_sample
        return grip_facts_from_measurements(
            begin_sample["jaw_normalized"],
            end["jaw_normalized"],
            begin_sample["aperture_m"],
            end["aperture_m"],
        )

    @staticmethod
    def _mirror_failure(receipt: dict[str, Any], stage_result: dict[str, Any]) -> None:
        keys = ("dx_m", "dy_m", "dz_m", "position_error_m", "orientation_error_rad", "approach_shortfall_m")
        if any(k in stage_result for k in keys):
            receipt["endpoint_residual_m"] = {k: stage_result[k] for k in keys if k in stage_result}

    def _close_receipt(
        self,
        receipt: dict[str, Any],
        run: dict[str, Any],
        start_step: int,
        start_native: int,
    ) -> dict[str, Any]:
        delta_steps = self.control_step - start_step
        delta_native = self.native_actions_dispatched - start_native
        receipt.setdefault("physics_advanced", delta_steps > 0)
        receipt.update(
            executed_steps=delta_native,
            executed_native_steps=delta_native,
            completed_native_steps=delta_native,
            executed_control_steps=delta_steps * self.substeps_per_control,
            wall_clock_duration_s=round(time.monotonic() - float(run["t0"]), 4),
            execution_interval={
                "start_sim_step": start_step,
                "end_sim_step": self.control_step,
                "step_domain": STEP_DOMAIN,
            },
            before_state=run["boundary"][0]["state"] if run["boundary"] else None,
            planned_trace={
                "arm": self.arm,
                "joint_names": list(ARM_JOINTS),
                "q_rad": [entry["q_rad"] for entry in run["dispatched"]],
                "completed_native_steps": len(run["dispatched"]),
                "row_kinds": [entry["kind"] for entry in run["dispatched"]],
                "semantics": "20 Hz closed-loop OSC_POSE joint trajectory samples in franka_base",
            },
            dispatched_trace={
                "scope": "20 Hz OSC_POSE commands",
                "samples": [
                    {"sim_step": entry["step"], "osc_action": entry["osc_action"], "jaw_goal": entry["gripper_goal_normalized"]}
                    for entry in run["dispatched"]
                ],
            },
            actual_trace=run["boundary"],
            episode_terminal=self.episode_terminal(),
        )
        if receipt.get("stop_confirmation") is None:
            receipt.pop("stop_confirmation", None)
        receipt["terminal_observation"] = self.observe()
        return receipt

    def execute(
        self,
        action: dict[str, Any],
        progress: Any = None,
        cancellation: threading.Event | None = None,
    ) -> dict[str, Any]:
        emit_progress = progress if callable(progress) else (lambda _u: None)
        prof = self.embodiment
        try:
            contract.validate_action(action, prof)
            if action.get("type") == "BatchAction":
                validate_batch_carry_slip_guard(action, **contract.guard_parameters(prof))
        except ValueError as error:
            return self._rejected(str(error))

        if self.step_budget_remaining() <= 0:
            return self._rejected("native_step_limit", status="stopped")

        if action.get("type") == "BatchAction":
            return self._execute_batch(action, emit_progress, cancellation)

        start_step = self.control_step
        start_native = self.native_actions_dispatched
        run = self._new_run(cancellation, float(action.get("timeout_s", 60.0)))
        receipt: dict[str, Any] = {
            "status": "failed",
            "reason": None,
            "waypoint_results": [],
            "before_observation": self.observe(),
        }
        emit_progress({"phase": "started", "sim_step": self.control_step})

        path = action["path"]
        if "hold_native_steps" in action:
            snap = self.read()
            T_hold = np.asarray(snap["T_base_tcp"], dtype=np.float64)
            req_steps = int(action["hold_native_steps"])
            for idx in range(req_steps):
                intr = self._check_interrupt(run)
                if intr:
                    receipt["reason"] = intr
                    receipt["status"] = "stopped"
                    receipt["stop_confirmation"] = self.hold(run, settle_steps=1)
                    break
                self._step_osc(T_hold, jaw_servo_goal=self.jaw_servo_goal, speed_scale=0.5, run=run, kind="wait_hold")
                done_steps = idx + 1
                if done_steps % 10 == 0 and done_steps < req_steps:
                    emit_progress(
                        {
                            "phase": "executing",
                            "sim_step": self.control_step,
                            "executed_steps": done_steps,
                            "observation": self.observe(),
                        }
                    )
            else:
                receipt.update(status="completed", reason=None)
            receipt["wait"] = {
                "requested_steps": req_steps,
                "executed_steps": run["commands"],
                "semantics": "20 Hz simulated hold of Panda EE pose and continuous jaw target",
            }
            return self._close_receipt(receipt, run, start_step, start_native)

        ok = True
        if path["kind"] == "backend_free":
            tols = self._effective_tolerances(path)
            receipt["effective_tolerances"] = tols
            seed = list(self.read()["q_rad"])
            waypoints = path["waypoints"]
            planned_ik = []
            for wp_idx, wp in enumerate(waypoints):
                q_sol, ik_rec = self._solve_stage_ik(wp[self.arm], seed, tols)
                if not ik_rec["ok"]:
                    return self._rejected(
                        "planning_failed",
                        status="failed",
                        failed_waypoint_index=wp_idx,
                        ik_result=ik_rec,
                        effective_tolerances=tols,
                    )
                planned_ik.append((wp_idx, wp[self.arm], ik_rec))
                seed = q_sol

            ramp_ev = next((ev for ev in action.get("gripper_events", []) if ev["phase"] == "during_motion"), None)
            for wp_idx, wp_pose, ik_rec in planned_ik:
                emit_progress(
                    {
                        "phase": "executing",
                        "waypoint_index": wp_idx,
                        "sim_step": self.control_step,
                        "executed_steps": run["commands"],
                    }
                )
                use_ramp = ramp_ev if (ramp_ev is not None and wp_idx == len(planned_ik) - 1) else None
                ok, reason, fields = self._move_stage(
                    run,
                    wp_pose,
                    path,
                    speed_scale=float(action.get("speed_scale", 1.0)),
                    during_jaw=use_ramp,
                    gripper_speed_scale=float(action.get("gripper_speed_scale", 1.0)),
                )
                wp_res = {"waypoint_index": wp_idx, "ik": ik_rec, **fields}
                receipt["waypoint_results"].append(wp_res)
                if not ok:
                    receipt.update(reason=reason, failed_waypoint_index=wp_idx)
                    if reason in ("cancelled", "native_step_limit", "native_success", "native_failure", "action_timeout"):
                        receipt["status"] = "stopped"
                    if "contact_block_summary" in fields:
                        receipt["contact_block_summary"] = fields["contact_block_summary"]
                    receipt["stop_confirmation"] = fields.get("hold_confirmation") or self.hold(run, settle_steps=1)
                    self._mirror_failure(receipt, wp_res)
                    break
                receipt.update(
                    position_error_m=fields["position_error_m"],
                    orientation_error_rad=fields["orientation_error_rad"],
                )

        if ok:
            for ev in action.get("gripper_events", []):
                if ev["phase"] == "after_motion":
                    begin_g = self.read()
                    ok, reason, g_info = self._gripper_stage(
                        run,
                        float(ev["goal_normalized"]),
                        float(action.get("gripper_speed_scale", 1.0)),
                    )
                    receipt["gripper_execution"] = {self.arm: g_info}
                    receipt["grip_facts"] = {self.arm: self._grip_facts(begin_g, self.read())}
                    if not ok:
                        receipt["reason"] = reason
                        receipt["status"] = (
                            "stopped"
                            if reason in ("cancelled", "native_step_limit", "native_success", "native_failure", "action_timeout")
                            else "failed"
                        )
                        receipt["stop_confirmation"] = self.hold(run, settle_steps=1)
                        break
            if ok:
                receipt.update(status="completed", reason=None)

        return self._close_receipt(receipt, run, start_step, start_native)

    def _execute_batch(
        self,
        action: dict[str, Any],
        emit_progress: Any,
        cancellation: threading.Event | None,
    ) -> dict[str, Any]:
        prof = self.embodiment
        plan_id = action.get("batch_plan_id")
        if not plan_id:
            preview = self.preview_batch(action)
            if preview.get("status") != "planned":
                return self._rejected(
                    str(preview.get("reason") or "planning_failed"),
                    status="failed",
                    failed_stage_id=preview.get("failed_stage_id"),
                    failed_arm=preview.get("failed_arm"),
                    failure_feedback=preview.get("failure_feedback"),
                )
            plan_id = preview["plan_id"]

        plan = self._plans.get(plan_id)
        scheduler = BatchScheduler(self, plans=self._plans)
        try:
            scheduler.admit_cached_plan(plan, action, digest_value=self._batch_digest(action))
        except ValueError as error:
            return self._rejected(str(error))

        measured = self.read()
        drift = float(np.max(np.abs(np.subtract(measured["q_rad"], plan["start_q"]))))
        if drift > 0.15:
            plan["status"] = "invalidated"
            return self._rejected(
                "batch_plan_stale_state_requires_new_preview",
                start_drift_rad=round(drift, 5),
                max_start_drift_rad=0.15,
            )

        start_step = self.control_step
        start_native = self.native_actions_dispatched
        run = self._new_run(cancellation, float(action.get("timeout_s", 120.0)))
        plan["status"] = "executing"
        plan["revision"] = int(plan.get("revision", 0)) + 1

        receipt: dict[str, Any] = {
            "status": "failed",
            "reason": None,
            "stage_results": {},
            "before_observation": self.observe(),
        }
        emit_progress({"phase": "started", "sim_step": self.control_step})
        guard_params = contract.guard_parameters(prof)

        def _pause(stage_ids, reasons, checkpoint_reason=None, checkpoint_reason_code=None):
            stop = self.hold(run, settle_steps=1)
            plan["status"] = "awaiting_observation"
            plan["start_q"] = list(self.read()["q_rad"])
            plan["cursor"] = self.native_actions_dispatched - start_native + int(plan.get("cursor", 0))
            receipt.update(
                pause_receipt(
                    stage_ids,
                    reasons,
                    stop,
                    checkpoint_reason=checkpoint_reason,
                    checkpoint_reason_code=checkpoint_reason_code,
                )
            )

        def _on_carry_slip_hold():
            stop = self.hold(run, settle_steps=1)

            def _sync():
                plan["start_q"] = list(self.read()["q_rad"])
                plan["cursor"] = self.native_actions_dispatched - start_native + int(plan.get("cursor", 0))

            return stop, _sync

        def _run_stage(sid, stage_obj, record, result, carry_monitor):
            begin = self.read()
            if record["kind"] == "move":
                leaf = stage_obj["action"]
                ramp = next(iter(leaf.get("gripper_events") or []), None)
                ok, reason, fields = self._move_stage(
                    run,
                    record["pose"],
                    leaf["path"],
                    speed_scale=float(leaf.get("speed_scale", 1.0)),
                    during_jaw=ramp,
                    gripper_speed_scale=float(leaf.get("gripper_speed_scale", 1.0)),
                    stage_id=sid,
                    carry_slip_monitor=carry_monitor,
                )
                slip_trig = fields.pop("_carry_slip_trigger", None)
                result.update(ik=record["ik"], effective_tolerances=record["tolerances"], **fields)
                if slip_trig is not None:
                    result["carry_slip"] = slip_trig
                    return False, "carry_slip_aperture_below_min_m", begin, self.read()
            else:
                leaf = stage_obj["action"]
                ok, reason, g_info = self._gripper_stage(
                    run,
                    float(record["goal_normalized"]),
                    float(leaf.get("gripper_speed_scale", 1.0)),
                )
                result["gripper_execution"] = g_info
            end = self.read()
            return ok, reason, begin, end

        def _on_stage_failed(sid, reason, result):
            receipt.update(reason=reason, failed_stage_id=sid, failed_arm=self.arm)
            if reason in ("cancelled", "native_step_limit", "native_success", "native_failure", "action_timeout"):
                receipt["status"] = "stopped"
            if "contact_block_summary" in result:
                receipt["contact_block_summary"] = result["contact_block_summary"]
            receipt["stop_confirmation"] = result.get("hold_confirmation") or self.hold(run, settle_steps=1)
            self._mirror_failure(receipt, result)

        scheduler.run_stage_plan(
            plan_id=plan_id,
            plan=plan,
            action=action,
            receipt=receipt,
            default_arm=self.arm,
            failed_reasons=("tracking_error",),
            on_stage_progress=lambda sid: emit_progress(
                {
                    "phase": "executing",
                    "sim_step": self.control_step,
                    "executed_steps": run["commands"],
                    "arm_context": {self.arm: {"stage_id": sid}},
                }
            ),
            on_pause=_pause,
            on_carry_slip_hold=_on_carry_slip_hold,
            run_stage=_run_stage,
            compute_facts=self._grip_facts,
            is_downward_move=lambda rec, beg: (
                rec["kind"] == "move"
                and float(rec["pose"]["position_m"][2]) - float(beg["position_m"][2]) < DOWNWARD_MOVE_DZ_M
            ),
            post_grasp_recheck=True,
            guard_parameters=guard_params,
            on_stage_failed=_on_stage_failed,
            on_batch_failed=lambda: None,
        )

        if plan["index"] >= len(plan["order"]) and plan.get("status") == "executing":
            plan["status"] = "completed"
            plan["cursor"] = self.native_actions_dispatched - start_native + int(plan.get("cursor", 0))
            receipt.update(status="completed", reason=None)

        summarize_batch(
            plan_id,
            plan,
            plan["order"],
            receipt["stage_results"],
            receipt,
            complete_batch_native_steps=int(plan.get("nominal_steps", 0)),
        )
        # Collect batch-level grip_facts across executed stages
        last_gf = None
        for sid in plan["order"]:
            st_gf = receipt["stage_results"].get(sid, {}).get("grip_facts")
            if isinstance(st_gf, dict):
                last_gf = st_gf
        if last_gf is not None:
            receipt["grip_facts"] = {self.arm: deepcopy(last_gf)}

        return self._close_receipt(receipt, run, start_step, start_native)

    # ------------------------------------ reset, evaluate, video, close ---
    def reset(
        self,
        seed: int = 0,
        layout_id: int = 0,
        *,
        bddl_file: str | None = None,
        init_state_file: str | None = None,
        language_override: str | None = None,
        perturbation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (seed, layout_id)):
            raise ValueError("seed and layout_id must be nonnegative integers")
        before = self.robot_state()
        self.episode_index += 1
        self.seed, self.layout_id = int(seed), int(layout_id)
        if bddl_file is not None or init_state_file is not None:
            import torch

            if bddl_file is not None:
                spec = _resolve_libero_task(
                    libero_root=self._libero_root,
                    task=self.task,
                    suite_name=self.suite_name,
                    task_id=self.task_id,
                    bddl_file=bddl_file,
                    init_state_file=init_state_file,
                )
                self.suite_name = spec["suite_name"]
                self.task_id = spec["task_id"]
                self.task_name = spec["task_name"]
                self.native_language_instruction = spec["language_instruction"]
                self._init_states = spec["init_states"]
                if spec["bddl_file"] != self.bddl_file:
                    self.bddl_file = spec["bddl_file"]
                    if self.env is not None:
                        try:
                            self.env.close()
                        except Exception:
                            pass
                    from libero.libero.envs import OffScreenRenderEnv

                    self.env = OffScreenRenderEnv(
                        bddl_file_name=self.bddl_file,
                        use_camera_obs=False,
                        ignore_done=True,
                        hard_reset=False,
                        camera_heights=self.camera_resolution,
                        camera_widths=self.camera_resolution,
                    )
                    self._loaded_bddl_file = self.bddl_file
                    self._snapshot_pristine_scene()
            elif init_state_file is not None:
                init_p = Path(init_state_file).expanduser().resolve()
                if not init_p.is_file():
                    raise ValueError(f"libero_init_state_file_not_found: {init_p}")
                self._init_states = torch.load(init_p, map_location="cpu", weights_only=False)
        if language_override is not None:
            self.language_override = str(language_override)
        self.language_instruction = self.language_override or self.native_language_instruction
        if perturbation is not None:
            self.perturbation = deepcopy(perturbation)
        chosen_idx = self._reset_sim_state(seed=self.seed, layout_id=self.layout_id)
        obs = self.observe()
        return {
            "reset_id": str(uuid.uuid4()),
            "episode_index": self.episode_index,
            "seed": self.seed,
            "layout_id": self.layout_id,
            "init_state_id": chosen_idx,
            "before_state": before,
            "robot_state": obs["robot_state"],
            "observation": obs,
            "capabilities": self.capabilities(),
            "clock": "paused",
            "reset_strategy": "libero_set_init_state",
        }

    def evaluate(self, final_check: bool = True, settle_steps: int = 10) -> dict[str, Any]:
        terminal = self.episode_terminal()
        if final_check and not terminal["ended"] and int(settle_steps) > 0:
            snap = self.read()
            T_hold = np.asarray(snap["T_base_tcp"], dtype=np.float64)
            for _ in range(int(settle_steps)):
                self._step_osc(
                    T_hold,
                    jaw_servo_goal=self.jaw_servo_goal,
                    speed_scale=0.25,
                    run=None,
                    kind="eval_settle",
                    count_actor_step=False,
                )
                self.evaluator_settle_steps += 1

        raw_success = bool(self.env.check_success())
        task_success = bool(terminal["success"]) if (terminal["ended"] and terminal["success"] is not None) else raw_success
        reward = 1.0 if task_success else 0.0
        eval_id = str(uuid.uuid4())
        result = {
            "schema_version": "0.1",
            "evaluation_id": eval_id,
            "simulation_only": True,
            "task": self.task,
            "suite_name": self.suite_name,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "language_instruction": self.language_instruction,
            "seed": self.seed,
            "layout_id": self.layout_id,
            "init_state_id": self.active_init_state_id,
            "episode_index": self.episode_index,
            "task_success": task_success,
            "native_reward": reward,
            "native_process_score": None,
            "normalized_score": reward,
            "valid_episode": True,
            "authority": "installed LIBERO BDDL _check_success",
            "audience": "observer_only",
            "final_check": bool(final_check),
            "settle_steps": int(settle_steps) if (final_check and not terminal["ended"]) else 0,
            "evaluator_actions": self.native_actions_dispatched,
            "success_authority": "locked_native_terminal" if terminal["ended"] else "native_final_check",
            "episode_terminal": terminal,
            "sim_step": self.control_step,
            "step_domain": STEP_DOMAIN,
        }
        eval_dir = self.run_dir / "evaluations"
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_path = eval_dir / f"{eval_id}.json"
        eval_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        return {**result, "evaluation_path": str(eval_path)}

    def observer_snapshot(self) -> dict[str, Any]:
        self.sim.forward()
        m = self.sim.model
        objects: list[dict[str, Any]] = []
        skip_prefixes = ("world", "robot0_", "gripper0_", "table", "floor", "mounted_")
        for bid in range(1, m.nbody):
            bname = m.body_id2name(bid) or ""
            if not bname or bname.startswith(skip_prefixes):
                continue
            T_wb = np.eye(4, dtype=np.float64)
            T_wb[:3, :3] = self.sim.data.body_xmat[bid].reshape(3, 3)
            T_wb[:3, 3] = self.sim.data.body_xpos[bid]
            T_bb = self._T_base_world @ T_wb
            objects.append(
                {
                    "instance_name": bname,
                    "kind": "Rigid",
                    "position_m": [round(float(v), 6) for v in T_bb[:3, 3]],
                    "quaternion_xyzw": [round(float(v), 6) for v in _matrix_to_quaternion_xyzw(T_bb[:3, :3])],
                }
            )
        return {
            "audience": "observer_only",
            "simulation_only": True,
            "source_kind": "simulator_hidden_scene_state",
            "frame_id": LIBERO_FRANKA_PANDA.frame_id,
            "sim_step": self.control_step,
            "step_domain": STEP_DOMAIN,
            "objects": objects,
        }

    def record(self, enabled: bool = True) -> dict[str, Any]:
        self.recording = bool(enabled)
        if self.recording:
            self.recorded_frames = []
            self.video_export = None
            self.observe(video_only=True)
        return {
            "enabled": self.recording,
            "sampling": "actual RGB every 4 control steps (5 Hz sim time), plus observation boundaries",
        }

    def save_video(self) -> dict[str, Any]:
        from PIL import Image

        if self.video_export is not None:
            return self.video_export
        if self.recording:
            self.observe(video_only=True)
        self.recording = False
        if not self.recorded_frames:
            raise ValueError("no recorded frames; call record before execution")

        vid_dir = self.run_dir / "videos" / str(uuid.uuid4())
        vid_dir.mkdir(parents=True, exist_ok=True)
        cam_specs = list(self.recorded_frames[0]["cameras"])
        videos: list[dict[str, Any]] = []

        def _encode_mp4(out_path: Path, frames_rgb: list[np.ndarray]) -> None:
            h, w = frames_rgb[0].shape[:2]
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{w}x{h}",
                "-framerate",
                "5",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out_path),
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            assert proc.stdin is not None
            for arr in frames_rgb:
                proc.stdin.write(np.ascontiguousarray(arr, dtype=np.uint8).tobytes())
            proc.stdin.close()
            err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            if proc.wait() != 0:
                raise RuntimeError(f"ffmpeg_failed: {err.strip()}")

        per_cam_arrays: dict[str, list[np.ndarray]] = {}
        for spec in cam_specs:
            cid = spec["camera_id"]
            imgs: list[np.ndarray] = []
            for fr in self.recorded_frames:
                src = next(item for item in fr["cameras"] if item["camera_id"] == cid)
                with Image.open(camera_path(src)) as im:
                    imgs.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
            mp4_path = vid_dir / f"{cid}.mp4"
            _encode_mp4(mp4_path, imgs)
            per_cam_arrays[cid] = imgs
            videos.append(
                {
                    "camera_id": cid,
                    "video_path": str(mp4_path),
                    "frame_count": len(imgs),
                    "fps": 5,
                    "width": int(imgs[0].shape[1]),
                    "height": int(imgs[0].shape[0]),
                }
            )

        composite_entry = None
        if len(per_cam_arrays) >= 2:
            ordered_cids = [pub for pub, _ in CAMERA_MAP if pub in per_cam_arrays]
            comp_frames = [
                np.concatenate([per_cam_arrays[cid][idx] for cid in ordered_cids], axis=1)
                for idx in range(len(self.recorded_frames))
            ]
            comp_path = vid_dir / "composite.mp4"
            _encode_mp4(comp_path, comp_frames)
            composite_entry = {
                "audience": "observer_only",
                "kind": "side_by_side_composite",
                "status": "completed",
                "video_path": str(comp_path),
                "frame_count": len(comp_frames),
                "fps": 5,
            }

        self.video_export = {
            "audience": "observer_only",
            "videos": videos,
            **({"composite": composite_entry} if composite_entry is not None else {}),
        }
        return self.video_export

    def close_episode(self) -> dict[str, Any]:
        self._plans.clear()
        self.recording = False
        self.recorded_frames = []
        self.video_export = None
        return {"closed": True, "pooled_worker_retained": True}

    def close(self, progress: Any = None) -> None:
        emit_progress = progress if callable(progress) else (lambda _u: None)
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None
        emit_progress({"phase": "environment_closed"})
        emit_progress({"phase": "application_close_started", "fast_shutdown": True})
        emit_progress({"phase": "application_closed"})


def _serve_socket_daemon(socket_path: Path, default_args: argparse.Namespace) -> None:
    """Serve pooled RL episodes over a persistent Unix domain socket."""
    socket_path = socket_path.expanduser().resolve()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    os.chmod(socket_path, 0o600)
    server.listen(16)

    adapter: LiberoAdapter | None = None
    episode_open = False
    episodes_served = 0
    raw_max_ep = default_args.max_episodes
    if raw_max_ep is None:
        try:
            raw_max_ep = int(os.environ.get("EAI_SIM_MAX_EPISODES", "0") or 0)
        except ValueError:
            raw_max_ep = 0
    max_episodes = max(0, int(raw_max_ep))
    shutting_down = threading.Event()
    work_queue: queue.Queue[tuple[dict[str, Any], Any, dict[str, threading.Event], threading.Lock]] = queue.Queue()

    def _handle_client(conn: socket.socket) -> None:
        write_lock = threading.Lock()
        cancellations: dict[str, threading.Event] = {}
        cancel_lock = threading.Lock()

        def emit_conn(msg: dict[str, Any]) -> None:
            raw = (WIRE_PREFIX + json.dumps(msg, allow_nan=False) + "\n").encode("utf-8")
            with write_lock:
                try:
                    conn.sendall(raw)
                except OSError:
                    pass

        try:
            rfile = conn.makefile("r", encoding="utf-8", buffering=1)
            for line in rfile:
                line = line.strip()
                if not line:
                    continue
                req: Any = None
                try:
                    req = json.loads(line)
                    op = req["op"]
                    req_id = req["id"]
                    if op == "cancel":
                        with cancel_lock:
                            ev = cancellations.get(req.get("target"))
                            if ev is not None:
                                ev.set()
                        emit_conn({"id": req_id, "result": {"stop_requested": ev is not None, "confirmed": False}})
                    elif op == "hello":
                        emit_conn(
                            {
                                "id": req_id,
                                "result": {
                                    "ok": True,
                                    "status": "ok",
                                    "pid": os.getpid(),
                                    "backend": "libero_mujoco",
                                    "simulation_only": True,
                                    "episodes_served": episodes_served,
                                    "max_episodes": max_episodes,
                                    "needs_recycle": bool(max_episodes > 0 and episodes_served >= max_episodes),
                                    "active_episode": episode_open,
                                    "episode_open": episode_open,
                                },
                            }
                        )
                    else:
                        if op == "execute":
                            with cancel_lock:
                                cancellations[req_id] = threading.Event()
                        work_queue.put((req, emit_conn, cancellations, cancel_lock))
                except Exception as err:
                    if isinstance(req, dict) and "id" in req:
                        emit_conn(_error_frame(req["id"], err, operation=req.get("op")))
        except OSError:
            pass
        finally:
            with cancel_lock:
                for ev in cancellations.values():
                    ev.set()

    def _accept_loop() -> None:
        while not shutting_down.is_set():
            try:
                conn, _ = server.accept()
            except OSError:
                break
            threading.Thread(target=_handle_client, args=(conn,), daemon=True).start()

    threading.Thread(target=_accept_loop, daemon=True, name="libero-socket-accept").start()

    try:
        while not shutting_down.is_set():
            req, emit_conn, cancellations, cancel_lock = work_queue.get()
            req_id = req["id"]
            op = req["op"]
            payload = req.get("payload")
            try:
                if op == "shutdown":
                    if adapter is not None:
                        adapter.close()
                        adapter = None
                    episode_open = False
                    emit_conn({"id": req_id, "result": {"ok": True, "closed": True, "shutting_down": True}})
                    shutting_down.set()
                    break
                if op == "open_episode":
                    merged = vars(default_args).copy()
                    merged.update(payload or {})
                    ns = SimpleNamespace(**merged)
                    if adapter is None:
                        adapter = LiberoAdapter(ns)
                    else:
                        adapter.open_episode(ns)
                    episodes_served += 1
                    episode_open = True
                    emit_conn({"id": req_id, "result": adapter.capabilities()})
                    continue
                if op in ("close_episode", "close"):
                    res = dict(adapter.close_episode()) if adapter is not None else {"closed": True}
                    episode_open = False
                    res["episodes_served"] = episodes_served
                    res["max_episodes"] = max_episodes
                    res["needs_recycle"] = bool(max_episodes > 0 and episodes_served >= max_episodes)
                    emit_conn({"id": req_id, "result": res})
                    continue

                if adapter is None or not episode_open:
                    raise RuntimeError("robot_worker_not_running: call open_episode first")

                if op == "ready":
                    result = adapter.capabilities()
                elif op == "observe":
                    result = adapter.observe()
                elif op == "preview_batch":
                    result = adapter.preview_batch(payload or {})
                elif op == "reach_batch":
                    result = adapter.reach_batch(payload or {})
                elif op == "reset":
                    result = adapter.reset(**(payload or {}))
                elif op == "evaluate":
                    result = adapter.evaluate(**(payload or {}))
                elif op == "observer_snapshot":
                    result = adapter.observer_snapshot()
                elif op == "record":
                    result = adapter.record(**(payload or {}))
                elif op == "save_video":
                    result = adapter.save_video()
                elif op == "execute":
                    with cancel_lock:
                        cancel_ev = cancellations.setdefault(req_id, threading.Event())
                    try:
                        result = adapter.execute(
                            payload or {},
                            lambda upd: emit_conn({"id": req_id, "progress": upd}),
                            cancel_ev,
                        )
                    finally:
                        with cancel_lock:
                            cancellations.pop(req_id, None)
                else:
                    raise ValueError("unsupported operation")
                emit_conn({"id": req_id, "result": result})
            except Exception as err:
                traceback.print_exc(file=sys.stderr)
                emit_conn(_error_frame(req_id, err, operation=op, bootstrap=(op == "open_episode")))
    finally:
        try:
            server.close()
        except OSError:
            pass
        if socket_path.exists():
            try:
                socket_path.unlink()
            except OSError:
                pass
        os._exit(0)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulation-only LIBERO MuJoCo robot worker for NI-RUA")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--stdio", action="store_true", help="Serve a single owner over stdin/stdout JSONL")
    mode.add_argument("--socket", type=str, default=None, help="Serve pooled episodes over a Unix domain socket")
    parser.add_argument("--run-dir", type=str, default="runs/robot")
    parser.add_argument("--task", type=str, default="libero_spatial:0")
    parser.add_argument("--suite-name", type=str, default=None)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=None)
    parser.add_argument("--bddl-file", type=str, default=None)
    parser.add_argument("--init-state-file", type=str, default=None)
    parser.add_argument("--language-override", type=str, default=None)
    parser.add_argument("--perturbation-json", type=str, default=None)
    parser.add_argument("--observation-profile", type=str, default="rgb_proprio_calibrated_v1")
    parser.add_argument("--camera-resolution", type=int, default=512)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--stop-on-success", action="store_true")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum pooled episodes before reporting needs_recycle=True (default: EAI_SIM_MAX_EPISODES or 0)",
    )
    parser.add_argument(
        "--render-backend",
        type=str,
        choices=("egl", "osmesa"),
        default=None,
        help="Optional rendering backend override for MUJOCO_GL and PYOPENGL_PLATFORM",
    )
    parser.add_argument(
        "--egl-device-id",
        type=int,
        default=None,
        help="Optional MUJOCO_EGL_DEVICE_ID override",
    )
    parser.add_argument(
        "--lp-num-threads",
        type=int,
        default=None,
        help="Optional LP_NUM_THREADS override for software Mesa llvmpipe",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE = 0
        except Exception:
            pass
    args = _parse_args(argv)
    if args.render_backend:
        os.environ["MUJOCO_GL"] = args.render_backend
        os.environ["PYOPENGL_PLATFORM"] = args.render_backend
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if args.egl_device_id is not None:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.egl_device_id)
    if args.lp_num_threads is not None:
        os.environ["LP_NUM_THREADS"] = str(args.lp_num_threads)
    if args.socket:
        _serve_socket_daemon(Path(args.socket), args)
        return
    serve_worker(
        lambda ns: LiberoAdapter(ns),
        args,
        _emit_stdio,
        on_eof=lambda adapter: adapter.close(),
    )


if __name__ == "__main__":
    main()
