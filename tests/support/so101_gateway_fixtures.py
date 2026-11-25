"""Shared SO-101 calibration and video-recording fixtures reused across robot and apps zones."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

from ni_rua.core.config import Config
from ni_rua.robot.backends.so101.deployment import load_deployment
from ni_rua.robot.backends.so101.driver import ARM_JOINTS, FakeArmDriver, FakeCameraSource, FakeClock
from ni_rua.robot.backends.so101.motion_video import MotionVideoRecorder
from ni_rua.robot.backends.so101.worker import SO101Adapter
from ni_rua.robot.embodiments.so101.contact import lowest_gripper_point
from ni_rua.robot.model import so101_kinematics as Kin
from support.paths import CONFIGS_ROOT

try:
    import cv2
    HAVE_ARUCO = hasattr(cv2, "aruco") and hasattr(cv2.aruco, "CharucoDetector")
except ImportError:  # pragma: no cover
    HAVE_ARUCO = False

if HAVE_ARUCO:
    from ni_rua.robot.calibration.workflow import charuco as cal

EXAMPLE = CONFIGS_ROOT / "so101.example.json"

RED, GREEN, BLUE, YELLOW = [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]
CAMERAS = {"fixed": dict(width=160, height=120, fps=30), "wrist": dict(width=128, height=96, fps=30)}

SQ, MARGIN = 120, 60
MODEL = Kin.load_model()
SIZE = (640, 480)
K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
TRUE_ZERO = np.array([0.0, 0.02, -0.03, 0.015, 0.0])  # physical = motor + TRUE_ZERO (rad)
TABLE_Z = -0.02
THICKNESS = 0.003
HOME = np.array([0.0, -0.95, 0.9, 1.0, 0.0])
OPERATOR_OK = ["table_clear", "workspace_clear", "board_matches", "touches_physically_confirmed"]


def render_board(board, T_camera_board, K_mat, image_size, *, blur_sigma=0.7):
    src = board.generateImage((7 * SQ + 2 * MARGIN, 5 * SQ + 2 * MARGIN), marginSize=MARGIN, borderBits=1)
    s = board.getSquareLength() / SQ
    S = np.array([[s, 0, -(MARGIN - 0.5) * s], [0, s, -(MARGIN - 0.5) * s], [0, 0, 1]])
    H = K_mat @ np.column_stack([T_camera_board[:3, 0], T_camera_board[:3, 1], T_camera_board[:3, 3]]) @ S
    img = cv2.warpPerspective(src, H, tuple(image_size), flags=cv2.INTER_LINEAR, borderValue=255)
    return cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), blur_sigma), cv2.COLOR_GRAY2BGR)


def sealed_recording(root, script):
    """Record ``script`` = [(action_id, [(camera, offset_s, rgb), ...]), ...] and seal it."""
    clock = FakeClock()
    adapter = SimpleNamespace(
        run_dir=Path(root),
        clock=clock,
        control_step=0,
        deployment={"cameras": CAMERAS, "camera": dict(video_crf=23, video_queue_mb=4)},
        cameras={
            name: FakeCameraSource(name, clock=clock, width=c["width"], height=c["height"])
            for name, c in CAMERAS.items()
        },
    )
    recorder = MotionVideoRecorder(adapter)
    recorder.record()
    for action_id, frames in script:
        recorder.begin(action_id)
        begin = clock.now()
        for camera, offset_s, color in sorted(frames, key=lambda item: item[1]):
            clock.sleep(max(0.0, begin + offset_s - clock.now()))
            source = adapter.cameras[camera]
            source._deliver_frame(np.full((source.height, source.width, 3), color, np.uint8), clock.now(), clock.now())
        recorder.end()
        clock.sleep(600)
    return recorder.save()


def decode(path, width, height):
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)


def fk_link(q):
    return Kin.fk(MODEL, q, Kin.WRIST_CAMERA_PARENT_LINK)


class Scene:
    """True geometry: hand-eye X, static board B on the table, fixed camera pose T_base_fixed."""

    def __init__(self):
        self.board = cal.make_board()
        self.X = (
            cal.invert_transform(fk_link(HOME))
            @ Kin.fk(MODEL, HOME)
            @ cal.make_transform(cal.rotvec_to_matrix([0.0, 0.0, 0.3]), [0.0, 0.02, -0.05])
        )
        T = fk_link(HOME) @ self.X
        depth = (T[2, 3] - (TABLE_Z + THICKNESS)) / -T[2, 2]
        centre = T[:3, 3] + T[:3, 2] * depth
        self.B = cal.board_frame_from_center([centre[0], centre[1], TABLE_Z + THICKNESS], self.board)
        eye, target = np.array([0.50, 0.0, 0.25]), np.array([centre[0], centre[1], TABLE_Z])
        z = (target - eye) / np.linalg.norm(target - eye)
        x = np.cross(z, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        self.T_base_fixed = np.eye(4)
        self.T_base_fixed[:3, :3] = np.column_stack([x, np.cross(z, x), z])
        self.T_base_fixed[:3, 3] = eye


class SceneArm(FakeArmDriver):
    """Fake arm in motor angles: reported q = motor + deployment offset_rad, physical = motor + TRUE_ZERO."""

    def _physical(self, motor):
        return np.asarray(motor) + TRUE_ZERO

    def _offsets(self):
        return np.array([self.deployment["joint_mapping"][name]["offset_rad"] for name in ARM_JOINTS])

    def _advance(self):
        before = list(self.q)
        super()._advance()
        if lowest_gripper_point(MODEL, self._physical(self.q), self.jaw)["z_m"] < TABLE_Z:
            low, high = 0.0, 1.0
            for _ in range(20):
                mid = (low + high) / 2
                trial = np.asarray(before) + mid * (np.asarray(self.q) - np.asarray(before))
                if lowest_gripper_point(MODEL, self._physical(trial), self.jaw)["z_m"] < TABLE_Z:
                    high = mid
                else:
                    low = mid
            self.q = list(np.asarray(before) + low * (np.asarray(self.q) - np.asarray(before)))

    def read(self):
        sample = super().read()
        return dict(sample, q_rad=list(np.asarray(sample["q_rad"]) + self._offsets()))

    def send_goal(self, q_rad, gripper_normalized):
        super().send_goal(list(np.asarray(q_rad) - self._offsets()), gripper_normalized)

    def arm_load(self, names):
        return {name: 100.0 for name in names}


class SceneCamera(FakeCameraSource):
    """Renders the board as seen from the true camera pose at the current physical arm state."""

    def __init__(self, camera_id, clock, scene, arm):
        super().__init__(camera_id, width=SIZE[0], height=SIZE[1], clock=clock)
        self.scene, self.arm, self.queue = scene, arm, []

    def _render(self):
        if self.queue:
            return self.queue.pop(0)
        if self.camera_id == "wrist":
            T_base_camera = fk_link(self.arm._physical(self.arm.q)) @ self.scene.X
        else:
            T_base_camera = self.scene.T_base_fixed
        bgr = render_board(self.scene.board, cal.invert_transform(T_base_camera) @ self.scene.B, K, SIZE)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def latest(self):
        self.last = (self._render(), self.clock.now())
        return self.last

    def read_after(self, t, timeout_s):
        self.last = (self._render(), max(t, self.clock.now()) + self.delay_s)
        return self.last


def write_initial_calibration(path, scene):
    wrong_X = scene.X @ cal.make_transform(cal.rotvec_to_matrix([0.01, -0.01, 0.0]), [0.003, -0.002, 0.002])
    wrong_fixed = scene.T_base_fixed @ cal.make_transform(np.eye(3), [0.03, 0.0, 0.0])
    cal.write_calibration_file(
        path,
        board=cal.DEFAULT_BOARD_SPEC,
        samples=10,
        created_at="2026-11-01T00:00:00Z",
        cameras={
            "wrist": dict(
                K=K,
                distortion=[0] * 5,
                image_size=list(SIZE),
                T_link_camera=wrong_X,
                intrinsics_rms_px=0.2,
                hand_eye_residual=dict(translation_m=0.003, rotation_rad=0.01),
            ),
            "fixed": dict(
                K=K,
                distortion=[0] * 5,
                image_size=list(SIZE),
                T_base_camera=wrong_fixed,
                intrinsics_rms_px=0.2,
                extrinsic_residual=dict(reprojection_px=4.0),
            ),
        },
        joint_calibration={"table_z_m": -0.0185, "joint_offsets_rad": {name: 0.0 for name in ARM_JOINTS}},
    )


def make_workflow_adapter(directory):
    """Adapter over a temporary deployment file (written by the joint_mapping writer) and the scene."""
    directory = Path(directory)
    camera_file = directory / "camera_calibration.json"
    scene = Scene()
    write_initial_calibration(camera_file, scene)
    document = json.loads(EXAMPLE.read_text())
    document["camera_calibration_file"] = str(camera_file)
    for name in ("wrist", "fixed"):
        document["cameras"][name].update(width=SIZE[0], height=SIZE[1])
    document["home"]["q_rad"] = HOME.tolist()
    document["motion"].update(max_joint_speed_rad_s=1.0, max_command_delta_rad=0.06)
    document["calibration"].update(
        pan_offsets_rad=[-0.12, 0.12],
        flex_offsets_rad=[-0.1, 0.1],
        roll_offsets_rad=[-0.3, 0.3],
        lift_offsets_rad=[-0.06, 0.06],
        speed_scale=1.0,
        min_samples=8,
        settle_timeout_s=0.5,
        touch_points_xy_m=[[0.20, 0.0], [0.24, 0.06], [0.24, -0.06]],
        verify_points_xy_m=[[0.22, 0.03], [0.26, -0.04]],
        touch_speed_scale=1.0,
        touch_contact_load_delta=0.0,
        board_thickness_m=THICKNESS,
        extrinsic_pose_count=5,
    )
    document["calibration"]["touch_video_fps"] = 0.0
    deployment_path = directory / "so101.local.json"
    deployment_path.write_text(json.dumps(document, indent=2))
    deployment = load_deployment(deployment_path)
    clock = FakeClock()
    arm = SceneArm(deployment, clock, q_rad=(HOME - TRUE_ZERO).tolist())
    cameras = {name: SceneCamera(name, clock, scene, arm) for name in ("wrist", "fixed")}
    adapter = SO101Adapter(
        deployment,
        directory / "robot",
        "rgb_proprio_calibrated_v1",
        driver=arm,
        cameras=cameras,
        clock=clock,
        image_settings=Config().images.to_dict(),
        start_telemetry=False,
        deployment_path=deployment_path,
    )
    (adapter.run_dir / "preflight.json").write_text(json.dumps({"ok": True, "checked_at": "2026-11-09T00:00:00Z"}))
    return adapter, arm, scene, deployment_path, camera_file


def start(adapter, step, **params):
    return adapter.operator("calib_step_start", step=step, params=dict({"confirmed": OPERATOR_OK}, **params))
