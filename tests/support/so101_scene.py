"""Synthetic SO-101 scene backend for live gateway tests: no serial port, camera, simulator or GPU.

``SceneBackend`` is the real parent-side ``so101_real`` backend (``ni_rua.robot.backends.so101.backend.RobotBackend``:
deployment loading, calibration check, action validation and compilation hooks, motion instructions, operation lock)
with exactly one seam replaced: the JSONL worker process. ``_request(op, payload)`` dispatches to an in-process
``SO101Adapter`` the way the worker's ``serve`` loop does (``ready``, ``observe``, ``execute``, ``preview_batch``,
``reset``, ``record``, ``save_video``, ``operator``, ``close``), and every payload, progress update and result crosses
a JSON round trip, as on the worker pipe; an adapter exception becomes the worker's ``"<Type>: <message>"`` error. The
adapter is the real one (kinematics, IK, interpolation, guards, receipts, telemetry, motion video) over
``FakeArmDriver`` (first-order servo lag in real time, starting at the deployment home pose) and ``SceneCamera``.

``SceneCamera`` renders what a calibrated pinhole camera sees of a static scene: a checkered table plane at
``table_z_m`` (m, base frame) with one solid-coloured cube, and the arm as a dark polyline through its URDF link
frames. It runs a background grabber at ``fps`` (frames/s) like ``OpenCVCameraSource``: ``read_after(t)`` returns the
first frame rendered after ``t``, and frames feed the motion-video sink. The fixed camera pose and the wrist camera
mount are read from the run's camera calibration file (``write_scene``), so published calibration and pixels agree;
the wrist view follows the measured joints. Nothing here reads model output or decides anything for the actor:
actions reach this backend only through ``Harness.execute``.

Gateway entry (``PYTHONPATH=tests``, cwd = checkout): ``python -m support.so101_scene <gateway CLI arguments>`` runs
the real gateway CLI (``ni_rua.apps.gateway.cli.main``) with the Workbench's ``create_backend`` replaced by
``create_scene_backend``. The scene JSON comes from the file named by ``NI_RUA_SO101_SCENE``.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path

import numpy as np

from ni_rua.robot.backends.so101.backend import RobotBackend
from ni_rua.robot.backends.so101.driver import CameraSource, Clock, FakeArmDriver
from ni_rua.robot.calibration.workflow import charuco as calibration
from ni_rua.robot.model import so101_kinematics as kinematics

SCENE_ENV = "NI_RUA_SO101_SCENE"
REPOSITORY = Path(__file__).resolve().parents[2]
EXAMPLE_DEPLOYMENT = REPOSITORY / "configs" / "so101.example.json"
# Cube colours a test may choose: name -> RGB (0..255). Distinct hues; the table and wall are neutral greys.
COLOURS = {"red": (205, 35, 35), "green": (35, 165, 60), "blue": (35, 70, 210), "yellow": (235, 205, 30),
           "purple": (135, 45, 175), "orange": (240, 125, 25)}
# Words a model's answer may use for each colour (live checks).
COLOUR_WORDS = {"red": {"red"}, "green": {"green"}, "blue": {"blue"}, "yellow": {"yellow"},
                "purple": {"purple", "violet", "magenta"}, "orange": {"orange"}}
# Pinhole intrinsics of both synthetic cameras (px; principal point at the image centre of 640x480 frames).
FOCAL_PX = 500.0
# Fixed camera: position and look-at target in so101_base (m).
FIXED_POSITION_M, FIXED_TARGET_M = (0.42, -0.22, 0.30), (0.17, 0.04, 0.0)
# Wrist camera at the home pose: offset from the gripper_link origin and look-at target in so101_base (m).
WRIST_OFFSET_M, WRIST_TARGET_M = (0.0, 0.0, 0.03), (0.24, 0.06, 0.0)
TABLE_SQUARE_M = 0.05          # checker square edge on the table (m)
TABLE_GREYS = ((168, 166, 160), (146, 144, 139))
WALL_RGB = (206, 209, 214)
ARM_RGB = (58, 58, 64)
FAR_M = 2.0                    # table hits farther than this from the camera are drawn as wall (m)


def look_at(position, target):
    """4x4 camera-to-base transform of an OpenCV camera (+x right, +y down, +z forward) at ``position`` looking at
    ``target`` (both so101_base, m), with the image "up" toward +z."""
    position, target = np.asarray(position, float), np.asarray(target, float)
    forward = (target - position) / np.linalg.norm(target - position)
    right = np.cross(forward, (0.0, 0.0, 1.0))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack([right, down, forward])
    transform[:3, 3] = position
    return transform


def scene_document(colour, *, cube_centre_m=(0.23, 0.08), cube_edge_m=0.04, table_z_m=0.0, fps=10.0,
                   execute_hold_s=0.0):
    """Scene JSON: ``colour`` (a ``COLOURS`` name), cube centre ``[x, y]`` on the table (m, so101_base), cube edge
    (m), table height ``table_z_m`` (m) and camera render rate ``fps`` (frames/s).

    ``execute_hold_s`` (s, finite, >= 0, default 0 = off) is a test-only slow mode: the in-process worker answers
    every ``execute`` this long after the adapter finished it (``SceneBackend``), so one ``robot_action`` call
    outlasts an agent host's MCP call timeouts (host end-to-end tests: more than 300 s for Claude Code's idle timer).
    The hold comes after the motion, so the motion, its plan lifetime (``plan.max_age_s``), its ``timeout_s``
    deadline and its receipt are those of a normal call; an operator stop (the Harness cancel event) ends the hold
    at once."""
    if colour not in COLOURS:
        raise ValueError(f"unknown_cube_colour:{colour}")
    if isinstance(execute_hold_s, bool) or not isinstance(execute_hold_s, (int, float)) \
            or not math.isfinite(execute_hold_s) or execute_hold_s < 0:
        raise ValueError("invalid_execute_hold_s")
    return dict(colour=colour, rgb=list(COLOURS[colour]), cube_centre_m=list(cube_centre_m), cube_edge_m=cube_edge_m,
                table_z_m=table_z_m, fps=fps, execute_hold_s=float(execute_hold_s))


def write_scene(directory, scene):
    """Write ``deployment.json`` (``configs/so101.example.json`` with ``camera_calibration_file`` pointing to the
    synthetic calibration), ``camera_calibration.json`` (both cameras, 640x480, zero distortion) and ``scene.json``
    into ``directory``. Returns ``(deployment path, scene path)``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    deployment = json.loads(EXAMPLE_DEPLOYMENT.read_text())
    deployment["camera_calibration_file"] = "camera_calibration.json"
    width, height = (deployment["cameras"]["fixed"][key] for key in ("width", "height"))
    intrinsics = [[FOCAL_PX, 0.0, width / 2], [0.0, FOCAL_PX, height / 2], [0.0, 0.0, 1.0]]
    model = kinematics.load_model()
    home = deployment["home"]["q_rad"]
    link = np.asarray(kinematics.fk(model, home, link=kinematics.WRIST_CAMERA_PARENT_LINK))
    wrist = look_at(link[:3, 3] + np.asarray(WRIST_OFFSET_M), WRIST_TARGET_M)
    calibration.write_calibration_file(directory / "camera_calibration.json", board=calibration.DEFAULT_BOARD_SPEC,
        samples=20, joint_calibration={"table_z_m": scene["table_z_m"]}, cameras={
            "wrist": dict(K=intrinsics, distortion=[0.0] * 5, image_size=[width, height],
                          T_link_camera=(np.linalg.inv(link) @ wrist).tolist(), intrinsics_rms_px=0.3,
                          hand_eye_residual=dict(translation_m=0.001, rotation_rad=0.002)),
            "fixed": dict(K=intrinsics, distortion=[0.0] * 5, image_size=[width, height],
                          T_base_camera=look_at(FIXED_POSITION_M, FIXED_TARGET_M).tolist(), intrinsics_rms_px=0.3,
                          extrinsic_residual=dict(reprojection_px=0.5))})
    (directory / "deployment.json").write_text(json.dumps(deployment, indent=2))
    (directory / "scene.json").write_text(json.dumps(scene, indent=2))
    return directory / "deployment.json", directory / "scene.json"


class SceneRenderer:
    """Pure renderer of the scene for one camera: ``render(T_base_camera, q_rad) -> uint8 HxWx3`` RGB."""

    def __init__(self, scene, intrinsics, width, height):
        self.scene, self.width, self.height = scene, width, height
        self.K = np.asarray(intrinsics, float)
        self.model = kinematics.load_model()
        chain = {joint["name"]: joint["child"] for joint in self.model["joints"]}
        self.arm_links = [chain[name] for name in kinematics.ARM_JOINTS] + [kinematics.TCP_LINK]
        u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
        self.rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(self.K).T
        self._background = None

    def _table(self, T):
        """Background: checkered table where the pixel ray hits the plane in front of the camera, wall elsewhere."""
        key = np.round(T, 6).tobytes()
        if self._background is not None and self._background[0] == key:
            return self._background[1].copy()
        directions = self.rays @ T[:3, :3].T
        origin, z = T[:3, 3], self.scene["table_z_m"]
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = (z - origin[2]) / directions[..., 2]
        hit = np.isfinite(distance) & (distance > 0) & (distance * np.linalg.norm(directions, axis=-1) < FAR_M)
        points = origin + directions * np.where(hit, distance, 0)[..., None]
        parity = (np.floor(points[..., 0] / TABLE_SQUARE_M) + np.floor(points[..., 1] / TABLE_SQUARE_M)) % 2
        image = np.empty((self.height, self.width, 3), np.uint8)
        image[:] = WALL_RGB
        image[hit & (parity == 0)] = TABLE_GREYS[0]
        image[hit & (parity == 1)] = TABLE_GREYS[1]
        self._background = (key, image)
        return image.copy()

    def _project(self, T, points):
        """(pixels Nx2, depth N) of base-frame ``points`` (Nx3, m)."""
        local = (np.asarray(points, float) - T[:3, 3]) @ T[:3, :3]
        depth = local[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            pixels = (local @ self.K.T)[:, :2] / depth[:, None]
        return pixels, depth

    def render(self, T, q_rad):
        from PIL import Image, ImageDraw
        T = np.asarray(T, float)
        image = Image.fromarray(self._table(T))
        draw = ImageDraw.Draw(image)
        shapes = []                                       # (depth, kind, pixels, colour, width)
        cx, cy = self.scene["cube_centre_m"]
        half, z0 = self.scene["cube_edge_m"] / 2, self.scene["table_z_m"]
        corners = np.array([[cx + sx * half, cy + sy * half, z0 + h * 2 * half]
                            for h in (0, 1) for sy in (-1, 1) for sx in (-1, 1)])
        faces = (((4, 5, 7, 6), (0, 0, 1), 1.0), ((0, 1, 5, 4), (0, -1, 0), 0.78), ((2, 3, 7, 6), (0, 1, 0), 0.7),
                 ((0, 2, 6, 4), (-1, 0, 0), 0.62), ((1, 3, 7, 5), (1, 0, 0), 0.85))
        pixels, depth = self._project(T, corners)
        if np.all(depth > 0.05):
            rgb = np.asarray(self.scene["rgb"], float)
            for indices, normal, shade in faces:
                centre = corners[list(indices)].mean(0)
                if np.dot(normal, T[:3, 3] - centre) > 0:      # face turned toward the camera
                    colour = tuple(int(c) for c in np.clip(rgb * shade, 0, 255))
                    shapes.append((float(depth[list(indices)].mean()), "polygon",
                                   [tuple(pixels[i]) for i in indices], colour, 0))
        joints = [np.zeros(3)] + [np.asarray(kinematics.fk(self.model, q_rad, link=name))[:3, 3]
                                  for name in self.arm_links]
        pixels, depth = self._project(T, joints)
        for a in range(len(joints) - 1):
            if depth[a] > 0.05 and depth[a + 1] > 0.05:
                width = max(2, int(FOCAL_PX * 0.03 / max(0.1, (depth[a] + depth[a + 1]) / 2)))
                shapes.append((float((depth[a] + depth[a + 1]) / 2), "line",
                               [tuple(pixels[a]), tuple(pixels[a + 1])], ARM_RGB, width))
        for _, kind, points, colour, width in sorted(shapes, key=lambda item: -item[0]):
            if kind == "polygon":
                draw.polygon(points, fill=colour)
            else:
                draw.line(points, fill=colour, width=width)
        return np.asarray(image, np.uint8)


class SceneCamera(CameraSource):
    """Background-grabber camera rendering ``SceneRenderer`` frames at ``scene["fps"]`` (frames/s).

    ``pose(q_rad) -> T_base_camera`` gives the camera pose; ``joints() -> q_rad`` the measured arm joints (wrist
    camera) at render time. ``read_after(t, timeout_s)`` waits for the first frame rendered after monotonic ``t``."""

    def __init__(self, camera_id, config, clock, scene, intrinsics, pose, joints):
        self.camera_id, self.config, self.clock = camera_id, config, clock
        self.renderer = SceneRenderer(scene, intrinsics, config["width"], config["height"])
        self.pose, self.joints = pose, joints
        self.period = 1.0 / float(scene["fps"])
        self.recording_fps, self.recording_fps_source = float(scene["fps"]), "synthetic_render"
        self.condition = threading.Condition()
        self.frame = self.stamp = self.read_finished = self.error = None
        self.running, self.thread, self.stamps = False, None, []

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._grab, daemon=True, name=f"scene-camera-{self.camera_id}")
        self.thread.start()

    def _grab(self):
        while self.running:
            started = self.clock.now()
            try:
                q = self.joints()
                frame, error = self.renderer.render(self.pose(q), q), None
            except Exception as failure:  # noqa: BLE001 - reported like a camera read failure
                frame, error = None, f"render_failed:{type(failure).__name__}"
            finished = self.clock.now()
            with self.condition:
                self.error = error
                if frame is not None:
                    self.frame, self.stamp, self.read_finished = frame, started, finished
                    self.stamps = (self.stamps + [started])[-30:]
                    self._deliver_frame(frame, started, finished)
                self.condition.notify_all()
            time.sleep(max(0.0, self.period - (self.clock.now() - started)))

    def latest(self):
        with self.condition:
            return None if self.frame is None else (self.frame, self.stamp)

    def latest_timed(self):
        with self.condition:
            if self.frame is None:
                return None
            return dict(image_rgb=self.frame, capture_monotonic_s=self.stamp,
                        host_read_interval_s=[self.stamp, self.read_finished], exposure_monotonic_s=None)

    def read_after(self, t, timeout_s):
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while self.stamp is None or self.stamp <= t:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"so101_camera_frame_timeout:{self.camera_id}:{self.error or 'no_new_frame'}")
                self.condition.wait(remaining)
            return self.frame, self.stamp

    def status(self):
        with self.condition:
            fps = ((len(self.stamps) - 1) / (self.stamps[-1] - self.stamps[0])
                   if len(self.stamps) > 1 and self.stamps[-1] > self.stamps[0] else None)
            return {"ok": self.frame is not None and self.error is None, "fps": fps, "error": self.error,
                    "last_frame_at": None if self.stamp is None else self.clock.utc(self.stamp)}

    def close(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2)


def _wire(value):
    """The value as it arrives on the other side of the worker pipe (JSON, no NaN)."""
    return None if value is None else json.loads(json.dumps(value, allow_nan=False))


class SceneBackend(RobotBackend):
    """``so101_real`` backend whose worker is an in-process ``SO101Adapter`` over the synthetic scene (module doc).

    ``scene["execute_hold_s"]`` (s, default 0) delays the answer to every ``execute`` after the adapter finished it
    (test-only slow mode, ``scene_document``)."""

    def __init__(self, *, scene, **kwargs):
        super().__init__(**kwargs)
        self.scene = scene
        self.adapter = None
        self.close_report = None

    def start(self):
        if self.adapter is not None:
            return self.capabilities
        from ni_rua.robot.backends.so101.worker import SO101Adapter
        self.run_dir.mkdir(parents=True, exist_ok=True)
        clock = Clock()
        home = self.deployment["home"]
        driver = FakeArmDriver(self.deployment, clock, q_rad=home["q_rad"], gripper_normalized=home["gripper_normalized"])
        document = calibration.read_calibration_file(self.deployment["camera_calibration_file"])
        model = kinematics.load_model()

        def pose(name):
            camera = document["cameras"][name]
            if name == "wrist":
                mount = np.asarray(camera["T_link_camera"])
                return lambda q: np.asarray(kinematics.fk(model, q, link=kinematics.WRIST_CAMERA_PARENT_LINK)) @ mount
            return lambda q: np.asarray(camera["T_base_camera"])
        cameras = {name: SceneCamera(name, config, clock, self.scene, document["cameras"][name]["K"], pose(name),
                                     joints=lambda: list(driver.q))
                   for name, config in self.deployment["cameras"].items()}
        self.adapter = SO101Adapter(self.deployment, self.run_dir, self.observation_profile, driver=driver,
                                    cameras=cameras, clock=clock, image_settings=self.image_settings,
                                    deployment_path=self.deployment_path)
        self.capabilities = self._request("ready")
        return self.capabilities

    def _request(self, operation, payload=None, timeout=180, progress=None, cancel=None):
        """The worker ``serve`` dispatch, in process (``timeout`` is not enforced in process)."""
        with self.lock:
            if self.failure:
                raise RuntimeError(self.failure)
            if self.adapter is None:
                raise RuntimeError("robot_worker_not_running")
        adapter, payload = self.adapter, _wire(payload) or {}

        def report(update):
            self.last_activity_at = time.monotonic()
            if progress is not None:
                progress(_wire(update))
        try:
            if operation == "ready":
                result = adapter.capabilities()
            elif operation == "observe":
                result = adapter.observe()
            elif operation == "preview_batch":
                result = adapter.preview_batch(payload)
            elif operation == "reach_batch":
                result = adapter.reach_batch(payload)
            elif operation == "record":
                result = adapter.motion_video.record(**payload)
            elif operation == "save_video":
                result = adapter.motion_video.save()
            elif operation == "reset":
                result = adapter.reset(**payload)
            elif operation == "operator":
                command = payload.pop("command")
                payload.pop("progress", None)
                result = adapter.operator(command, progress=report, **payload)
            elif operation == "execute":
                cancel = cancel if cancel is not None else threading.Event()
                result = adapter.execute(payload, report, cancel)
                cancel.wait(float((self.scene or {}).get("execute_hold_s", 0.0)))     # test-only slow mode (scene_document)
            elif operation == "close":
                adapter.close(report)
                result = {"closed": True}
            else:
                raise ValueError("unsupported operation")
        except Exception as error:  # noqa: BLE001 - the worker answers {"error": "<Type>: <message>"}
            raise RuntimeError(f"{type(error).__name__}: {error}") from error
        self.last_activity_at = time.monotonic()
        return _wire(result)

    def close(self):
        if self.adapter is None:
            return {"native_close_acknowledged": False, "forced_termination": None, "process_exit_confirmed": True,
                    "reader_exit_confirmed": True, "not_started": True, "errors": []}
        if self.close_report is not None:
            return self.close_report
        report = {"pid": None, "native_close_acknowledged": False, "forced_termination": None,
                  "process_exit_confirmed": True, "reader_exit_confirmed": True, "returncode": 0, "errors": []}
        try:
            report["native_close_acknowledged"] = self._request("close").get("closed") is True
        except RuntimeError as error:
            report["errors"].append(str(error))
        self.close_report = report
        return report


def create_scene_backend(*, config, **kwargs):
    """``create_backend`` replacement: a ``SceneBackend`` with the scene of ``$NI_RUA_SO101_SCENE``."""
    scene = json.loads(Path(os.environ[SCENE_ENV]).read_text())
    return SceneBackend(scene=scene, config=config, **kwargs)


def main():
    from ni_rua.apps.gateway import workbench
    workbench.create_backend = create_scene_backend
    from ni_rua.apps.gateway.cli import main as gateway_main
    gateway_main()


if __name__ == "__main__":
    main()
