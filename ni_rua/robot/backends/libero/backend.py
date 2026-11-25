"""Parent-process ``libero`` backend: simulation-only LIBERO (robosuite MuJoCo) Franka Emika Panda.

Supports two execution topologies over the exact same JSONL protocol:
1. Standalone subprocess mode (default when ``socket_path`` is unset): spawns and supervises one
   ``ni_rua.robot.backends.libero.worker --stdio`` child process in ``config.backend.python``.
2. Pooled Unix socket mode (``socket_path`` set): connects to a warm persistent
   ``ni_rua.robot.backends.libero.worker --socket <path>`` daemon via ``LiberoWorkerClient``,
   opening and closing episodes in ~25 ms without restarting MuJoCo or EGL.
"""
from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import socket
import threading
import time
from typing import Any
import uuid

from ....core.observation_policy import ACTOR_PROFILES
from ...embodiments.libero_franka_panda.profile import (
    LIBERO_FRANKA_PANDA,
    joint_limits_rad as panda_joint_limits_rad,
    libero_franka_panda_profile,
)
from ...ipc import WIRE_PREFIX, WorkerProcess
from ...ipc.process import _remote_error_from_wire
from ...motion.batch_contract import validate_batch_action
from ...motion.batch_scheduler import ArmLayout
from . import contract

ENVIRONMENT = "libero"
DEFAULT_CAMERA_RESOLUTION = 512
DEFAULT_MAX_EPISODE_STEPS = 1000


class LiberoWorkerClient:
    """Thread-safe JSONL RPC client over a Unix domain socket to a pooled ``libero.worker`` daemon."""

    def __init__(self, socket_path: str | Path, *, connect_timeout: float = 30.0) -> None:
        self.socket_path = str(Path(socket_path).expanduser())
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(connect_timeout)
        self._sock.connect(self.socket_path)
        self._sock.settimeout(None)
        self._rfile = self._sock.makefile("r", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending: dict[str, tuple[threading.Event, dict[str, Any], Callable[[Any], None] | None]] = {}
        self._failure: str | None = None
        self._closed = False
        self._reader = threading.Thread(target=self._read_loop, daemon=True, name="libero-socket-reader")
        self._reader.start()

    def _read_loop(self) -> None:
        try:
            for line in self._rfile:
                if not line.startswith(WIRE_PREFIX):
                    continue
                try:
                    message = json.loads(line[len(WIRE_PREFIX):])
                except ValueError:
                    continue
                req_id = message.get("id")
                is_progress = "progress" in message
                with self._lock:
                    entry = self._pending.get(req_id)
                    if entry is not None and not is_progress:
                        entry[1].update(message)
                        entry[0].set()
                if entry is not None and is_progress and entry[2] is not None:
                    try:
                        entry[2](message["progress"])
                    except Exception:
                        pass
        except (OSError, ValueError):
            pass
        finally:
            if not self._closed:
                self._fail("robot_worker_disconnected: socket_eof")

    def _fail(self, reason: str) -> None:
        with self._lock:
            self._failure = self._failure or reason
            for event, result, _ in self._pending.values():
                if not event.is_set():
                    result.setdefault("error", self._failure)
                    event.set()

    def _send(self, message: dict[str, Any], deadline: float) -> None:
        raw = (json.dumps(message, allow_nan=False) + "\n").encode("utf-8")
        acquired = self._write_lock.acquire(timeout=max(0.0, deadline - time.monotonic()))
        if not acquired:
            self._fail("robot status unknown: socket request write deadline")
            raise TimeoutError("robot status unknown: socket request write deadline; do not replay action")
        try:
            self._sock.sendall(raw)
        except OSError as error:
            self._fail(f"robot_worker_disconnected: {error}")
            raise
        finally:
            self._write_lock.release()

    def request(
        self,
        operation: str,
        payload: Any = None,
        *,
        timeout: float = 180.0,
        progress: Callable[[dict[str, Any]], None] | None = None,
        cancel: Any = None,
    ) -> Any:
        deadline = time.monotonic() + float(timeout)
        request_id = str(uuid.uuid4())
        done = threading.Event()
        result: dict[str, Any] = {}
        with self._lock:
            if self._failure:
                raise _remote_error_from_wire(self._failure, operation=operation)
            if self._closed:
                raise _remote_error_from_wire("robot_worker_not_running", operation=operation)
            self._pending[request_id] = (done, result, progress)
        try:
            self._send({"id": request_id, "op": operation, "payload": payload}, deadline)
            cancellation_sent = False
            while not done.wait(min(0.05, max(0.0, deadline - time.monotonic()))):
                if cancel is not None and cancel.is_set() and not cancellation_sent:
                    try:
                        self._send({"id": str(uuid.uuid4()), "op": "cancel", "target": request_id}, deadline)
                    except OSError:
                        pass
                    cancellation_sent = True
                if time.monotonic() >= deadline:
                    if not cancellation_sent:
                        try:
                            self._send({"id": str(uuid.uuid4()), "op": "cancel", "target": request_id}, deadline + 1.0)
                        except OSError:
                            pass
                    raise TimeoutError("robot status unknown: worker response deadline; do not replay action")
            if "error" in result:
                detail = result.get("error_detail") if isinstance(result.get("error_detail"), dict) else None
                raise _remote_error_from_wire(str(result["error"]), detail=detail, operation=operation)
            return result.get("result")
        finally:
            with self._lock:
                self._pending.pop(request_id, None)

    def close(self) -> None:
        self._closed = True
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except (OSError, ValueError):
            pass
        try:
            self._rfile.close()
        except (OSError, ValueError):
            pass
        try:
            self._sock.close()
        except (OSError, ValueError):
            pass
        self._reader.join(timeout=2.0)

    @classmethod
    def hello(cls, socket_path: str | Path, *, timeout: float = 15.0) -> dict[str, Any]:
        client = cls(socket_path, connect_timeout=timeout)
        try:
            return client.request("hello", timeout=timeout)
        finally:
            client.close()

    @classmethod
    def shutdown(cls, socket_path: str | Path, *, timeout: float = 15.0) -> dict[str, Any]:
        client = cls(socket_path, connect_timeout=timeout)
        try:
            return client.request("shutdown", timeout=timeout)
        finally:
            client.close()


def _load_optional_deployment(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise ValueError(f"libero_deployment_not_found: {resolved}")
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("libero_deployment_must_be_json_object")
    return data


class LiberoBackend(WorkerProcess):
    """Clean-room simulation-only ``libero`` backend implementing ``RobotBackend``, ``EvaluatorBackend``, and ``VideoBackend``."""

    robot_description = LIBERO_FRANKA_PANDA
    compile_action = staticmethod(contract.compile_action)

    def __init__(
        self,
        *,
        config: Any = None,
        python: str | Path | None = None,
        gpu_id: int | str | None = None,
        task: str | None = None,
        seed: int | None = None,
        layout_id: int | None = None,
        run_dir: str | Path = "runs/robot",
        observation_profile: str | None = None,
        deployment: str | Path | dict[str, Any] | None = None,
        socket_path: str | Path | None = None,
        suite_name: str | None = None,
        task_id: int | None = None,
        init_state_id: int | None = None,
        bddl_file: str | Path | None = None,
        init_state_file: str | Path | None = None,
        language_override: str | None = None,
        perturbation: dict[str, Any] | None = None,
        camera_resolution: int | None = None,
        max_episode_steps: int | None = None,
        stop_on_success: bool | None = None,
        **_ignored: Any,
    ) -> None:
        from ....core.config import Config
        from ....core.storage.images import image_settings

        config = config if config is not None else Config.load()
        if config.backend.environment != ENVIRONMENT:
            raise ValueError(f"unsupported_libero_environment: {config.backend.environment}")
        if config.backend.robot_model_id != LIBERO_FRANKA_PANDA.robot_model_id:
            raise ValueError(f"unsupported_libero_robot: {config.backend.robot_model_id!r}")

        if isinstance(deployment, dict):
            dep = dict(deployment)
        else:
            dep = _load_optional_deployment(deployment if deployment is not None else config.backend.deployment)
        self.deployment = dep

        raw_python = python or dep.get("python") or config.backend.python
        self.python = str(Path(raw_python).expanduser()) if raw_python else ""
        raw_socket = socket_path if socket_path is not None else dep.get("socket_path")
        self.socket_path = str(Path(raw_socket).expanduser()) if raw_socket else None
        if not self.socket_path and not self.python:
            raise ValueError("libero_python_or_socket_required: set backend.python or socket_path")

        self.gpu_id = config.run.gpu if gpu_id is None else gpu_id
        self.task = (dep.get("task") or config.run.task or "libero_spatial:0") if task is None else str(task)
        self.suite_name = dep.get("suite_name") if suite_name is None else suite_name
        self.task_id = dep.get("task_id") if task_id is None else int(task_id)
        self.seed = int(config.run.seed if seed is None else seed)
        self.layout_id = int(config.run.layout_id if layout_id is None else layout_id)
        self.init_state_id = dep.get("init_state_id") if init_state_id is None else int(init_state_id)
        raw_bddl = dep.get("bddl_file") if bddl_file is None else bddl_file
        self.bddl_file = str(Path(raw_bddl).expanduser()) if raw_bddl else None
        raw_init_file = dep.get("init_state_file") if init_state_file is None else init_state_file
        self.init_state_file = str(Path(raw_init_file).expanduser()) if raw_init_file else None
        raw_lang = dep.get("language_override") if language_override is None else language_override
        self.language_override = str(raw_lang) if raw_lang is not None else None
        raw_pert = dep.get("perturbation") if perturbation is None else perturbation
        if raw_pert is not None and not isinstance(raw_pert, dict):
            raise ValueError("invalid_perturbation: must be a dict or None")
        self.perturbation = dict(raw_pert) if isinstance(raw_pert, dict) else None

        res = dep.get("camera_resolution", DEFAULT_CAMERA_RESOLUTION) if camera_resolution is None else camera_resolution
        if type(res) is not int or not 64 <= res <= 2048:
            raise ValueError("invalid_camera_resolution: must be an integer in [64, 2048]")
        self.camera_resolution = res

        steps = dep.get("max_episode_steps", DEFAULT_MAX_EPISODE_STEPS) if max_episode_steps is None else max_episode_steps
        if type(steps) is not int or steps < 1:
            raise ValueError("invalid_max_episode_steps: must be a positive integer")
        self.max_episode_steps = steps
        self.stop_on_success = bool(dep.get("stop_on_success", False) if stop_on_success is None else stop_on_success)

        profile = config.features.observation_profile if observation_profile is None else observation_profile
        if profile not in ACTOR_PROFILES:
            raise ValueError(f"libero_unsupported_observation_profile: {profile}")
        self.observation_profile = profile

        self.image_settings = image_settings(config.images.to_dict())
        self.motion_thresholds = dict(config.embodiment.thresholds._overrides)
        self._socket_client: LiberoWorkerClient | None = None
        super().__init__(run_dir)

    # ---------------------------------------------------- pure hooks ---
    def embodiment_profile(self):
        return libero_franka_panda_profile(thresholds=self.motion_thresholds)

    @property
    def arm_layout(self) -> ArmLayout:
        return ArmLayout.from_profile(self.embodiment_profile())

    def validate_action(self, action: dict[str, Any]):
        return contract.validate_action(action, self.embodiment_profile())

    def actor_motion_instructions(self) -> str:
        table_z = None
        if isinstance(self.capabilities, dict):
            table_z = (self.capabilities.get("scene_calibration") or {}).get("table_z_m")
        return contract.actor_motion_instructions(self.embodiment_profile(), table_z_m=table_z)

    def joint_limits_rad(self) -> dict[str, list[float]]:
        return {name: list(bounds) for name, bounds in panda_joint_limits_rad().items()}

    @property
    def language_instruction(self) -> str | None:
        if isinstance(self.capabilities, dict):
            return (self.capabilities.get("task_metadata") or {}).get("language_instruction")
        return None

    # ----------------------------------------------------- transport ---
    def _episode_payload(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "task": self.task,
            "suite_name": self.suite_name,
            "task_id": self.task_id,
            "seed": self.seed,
            "layout_id": self.layout_id,
            "init_state_id": self.init_state_id,
            "bddl_file": self.bddl_file,
            "init_state_file": self.init_state_file,
            "language_override": self.language_override,
            "perturbation": self.perturbation,
            "observation_profile": self.observation_profile,
            "camera_resolution": self.camera_resolution,
            "max_episode_steps": self.max_episode_steps,
            "stop_on_success": self.stop_on_success,
            "image_settings": self.image_settings,
            "motion_thresholds": self.motion_thresholds,
        }

    def _rpc(
        self,
        operation: str,
        payload: Any = None,
        *,
        timeout: float = 180.0,
        progress: Callable[[dict[str, Any]], None] | None = None,
        cancel: Any = None,
    ) -> Any:
        if self.socket_path is not None:
            if self._socket_client is None:
                raise _remote_error_from_wire("robot_worker_not_running", operation=operation)
            return self._socket_client.request(
                operation,
                payload,
                timeout=timeout,
                progress=progress,
                cancel=cancel,
            )
        return self._request(operation, payload, timeout=timeout, progress=progress, cancel=cancel)

    # ----------------------------------------------------- lifecycle ---
    def start(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.socket_path is not None:
            if self._socket_client is not None and self.capabilities is not None:
                return self.capabilities
            client = LiberoWorkerClient(self.socket_path)
            try:
                caps = client.request("open_episode", self._episode_payload(), timeout=300.0)
            except BaseException:
                client.close()
                raise
            self._socket_client = client
            self.capabilities = caps
            return caps

        if self.process is not None and self.capabilities is not None and self.process.poll() is None:
            return self.capabilities
        from ....core.config import DEFAULT_SUBPROCESS_THREAD_ENV

        project = str(Path(__file__).resolve().parents[4])
        environment = os.environ.copy()
        for key, value in DEFAULT_SUBPROCESS_THREAD_ENV.items():
            environment.setdefault(key, value)
        environment.setdefault("MUJOCO_GL", "egl")
        environment.setdefault("PYOPENGL_PLATFORM", "egl")
        if self.gpu_id is not None:
            environment.setdefault("MUJOCO_EGL_DEVICE_ID", str(self.gpu_id))
            environment["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        environment.update(
            NI_RUA_IMAGE_SETTINGS=json.dumps(self.image_settings),
            NI_RUA_MOTION_THRESHOLDS=json.dumps(self.motion_thresholds),
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=os.pathsep.join(
                item for item in (project, environment.get("PYTHONPATH", "")) if item
            ),
        )
        command = [
            self.python,
            "-u",
            "-m",
            "ni_rua.robot.backends.libero.worker",
            "--stdio",
            "--run-dir",
            str(self.run_dir),
            "--task",
            str(self.task),
            "--seed",
            str(self.seed),
            "--layout-id",
            str(self.layout_id),
            "--observation-profile",
            self.observation_profile,
            "--camera-resolution",
            str(self.camera_resolution),
            "--max-episode-steps",
            str(self.max_episode_steps),
        ]
        if self.suite_name is not None:
            command += ["--suite-name", str(self.suite_name)]
        if self.task_id is not None:
            command += ["--task-id", str(self.task_id)]
        if self.init_state_id is not None:
            command += ["--init-state-id", str(self.init_state_id)]
        if self.bddl_file is not None:
            command += ["--bddl-file", str(self.bddl_file)]
        if self.init_state_file is not None:
            command += ["--init-state-file", str(self.init_state_file)]
        if self.language_override is not None:
            command += ["--language-override", str(self.language_override)]
        if self.perturbation is not None:
            command += ["--perturbation-json", json.dumps(self.perturbation)]
        if self.stop_on_success:
            command.append("--stop-on-success")
        return self.start_process(command, cwd=project, environment=environment)

    def observe(self) -> dict[str, Any]:
        with self.operation_lock:
            return self._rpc("observe")

    def preview_batch(self, action: dict[str, Any]) -> dict[str, Any]:
        validate_batch_action(action, self.capabilities, profile=self.embodiment_profile())
        self.validate_action(action)
        with self.operation_lock:
            return self._rpc("preview_batch", action, timeout=float(action.get("timeout_s", 120.0)) + 60.0)

    def reach_batch(self, action: dict[str, Any]) -> dict[str, Any]:
        validate_batch_action(action, self.capabilities, profile=self.embodiment_profile())
        self.validate_action(action)
        with self.operation_lock:
            return self._rpc("reach_batch", action, timeout=float(action.get("timeout_s", 60.0)) + 30.0)

    def execute(
        self,
        action: dict[str, Any],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_flag: Any = None,
    ) -> dict[str, Any]:
        if action.get("type") == "BatchAction":
            validate_batch_action(action, self.capabilities, profile=self.embodiment_profile())
        self.validate_action(action)
        with self.operation_lock:
            return self._rpc(
                "execute",
                action,
                timeout=float(action.get("timeout_s", 60.0)) + 60.0,
                progress=progress_callback,
                cancel=cancel_flag,
            )

    def validate_reset(self, seed: int = 0, layout_id: int = 0) -> None:
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (seed, layout_id)):
            raise ValueError("seed and layout_id must be nonnegative integers")

    def reset(
        self,
        seed: int = 0,
        layout_id: int = 0,
        *,
        bddl_file: str | Path | None = None,
        init_state_file: str | Path | None = None,
        language_override: str | None = None,
        perturbation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.validate_reset(seed, layout_id)
        with self.operation_lock:
            self.seed, self.layout_id = int(seed), int(layout_id)
            if bddl_file is not None:
                self.bddl_file = str(Path(bddl_file).expanduser())
            if init_state_file is not None:
                self.init_state_file = str(Path(init_state_file).expanduser())
            if language_override is not None:
                self.language_override = str(language_override)
            if perturbation is not None:
                self.perturbation = dict(perturbation)
            payload: dict[str, Any] = {"seed": int(seed), "layout_id": int(layout_id)}
            if bddl_file is not None:
                payload["bddl_file"] = self.bddl_file
            if init_state_file is not None:
                payload["init_state_file"] = self.init_state_file
            if language_override is not None:
                payload["language_override"] = self.language_override
            if perturbation is not None:
                payload["perturbation"] = self.perturbation
            res = self._rpc("reset", payload, timeout=120.0)
            if isinstance(res, dict) and isinstance(res.get("capabilities"), dict):
                self.capabilities = res["capabilities"]
            return res

    def evaluate(self, final_check: bool = True, *, settle_steps: int = 10) -> dict[str, Any]:
        """Observer-only native LIBERO BDDL evaluation; never exposed to actor context."""
        with self.operation_lock:
            return self._rpc(
                "evaluate",
                {"final_check": bool(final_check), "settle_steps": int(settle_steps)},
                timeout=90.0,
            )

    def observer_snapshot(self) -> dict[str, Any]:
        """Observer-only MuJoCo ground-truth object poses; never exposed to actor context."""
        with self.operation_lock:
            return self._rpc("observer_snapshot", timeout=60.0)

    def record(self, enabled: bool = True) -> dict[str, Any]:
        with self.operation_lock:
            return self._rpc("record", {"enabled": bool(enabled)}, timeout=30.0)

    def save_video(self) -> dict[str, Any]:
        with self.operation_lock:
            return self._rpc("save_video", timeout=120.0)

    def close(self) -> dict[str, Any]:
        if self.socket_path is not None:
            client = self._socket_client
            self._socket_client = None
            self.capabilities = None
            if client is None:
                return {
                    "native_close_acknowledged": False,
                    "forced_termination": None,
                    "process_exit_confirmed": True,
                    "reader_exit_confirmed": True,
                    "pooled_worker_retained": True,
                    "not_started": True,
                    "errors": [],
                }
            errors: list[str] = []
            ack = False
            res: dict[str, Any] = {}
            try:
                raw_res = client.request("close_episode", timeout=30.0)
                if isinstance(raw_res, dict):
                    res = raw_res
                ack = res.get("closed") is True
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
            finally:
                client.close()
            return {
                "closed": ack,
                "pooled_socket": True,
                "episodes_served": int(res.get("episodes_served", 0)),
                "max_episodes": int(res.get("max_episodes", 0)),
                "needs_recycle": bool(res.get("needs_recycle", False)),
                "native_close_acknowledged": ack,
                "forced_termination": None,
                "process_exit_confirmed": True,
                "reader_exit_confirmed": True,
                "pooled_worker_retained": True,
                "errors": errors,
            }
        return super().close()


RobotBackend = LiberoBackend
