"""SO-101 calibration workflow ops in the worker (CPU only): fake arm with hidden joint-zero errors, a table,
and cameras that render the ChArUco board from the true (offset) arm pose.

Covers: begin/status/readiness, S0 self-check, S1 touch-off, S2 grid capture + solve, S3 hand-held capture +
solve, S4 fixed extrinsic, S5 joint solve recovering the injected offsets and table height with cross-checks,
apply (bundle version, atomic install, joint_mapping writer with backup, live offset switch), S6
verification touches, rollback, staleness hint, refusal/cancel paths and atomic install failure. Real optics,
servo compliance and contact forces are not covered.
"""

NI_RUA_TEST_TIER = "fast"

# === tests/test_so101_calibration_runner.py ===

import copy
import functools
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from ni_rua.core.config import Config
from ni_rua.robot.backends.so101.deployment import load_deployment
from ni_rua.robot.backends.so101.driver import FakeArmDriver, FakeCameraSource, FakeClock, ARM_JOINTS
from ni_rua.robot.backends.so101.worker import SO101Adapter
from ni_rua.robot.model import so101_kinematics as Kin
from ni_rua.robot.embodiments.so101.contact import lowest_gripper_point
from robot.so101.test_so101_backend import EXAMPLE

try:
    import cv2
    HAVE_ARUCO = hasattr(cv2, "aruco") and hasattr(cv2.aruco, "CharucoDetector")
except ImportError:  # pragma: no cover
    HAVE_ARUCO = False

if HAVE_ARUCO:
    from ni_rua.robot.calibration.workflow import charuco as cal
    from robot.calibration.test_so101_calibration import render_board

MODEL = Kin.load_model()
SIZE = (640, 480)
K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
TRUE_ZERO = np.array([0.0, 0.02, -0.03, 0.015, 0.0])  # physical = motor + TRUE_ZERO (rad)
TABLE_Z = -0.02
THICKNESS = 0.003
HOME = np.array([0.0, -0.95, 0.9, 1.0, 0.0])


def fk_link(q):
    return Kin.fk(MODEL, q, Kin.WRIST_CAMERA_PARENT_LINK)


class Scene:
    """True geometry: hand-eye X, static board B on the table, fixed camera pose T_base_fixed."""

    def __init__(self):
        self.board = cal.make_board()
        self.X = cal.invert_transform(fk_link(HOME)) @ Kin.fk(MODEL, HOME) @ cal.make_transform(
            cal.rotvec_to_matrix([0.0, 0.0, 0.3]), [0.0, 0.02, -0.05])
        T = fk_link(HOME) @ self.X
        depth = (T[2, 3] - (TABLE_Z + THICKNESS)) / -T[2, 2]
        centre = T[:3, 3] + T[:3, 2] * depth
        self.B = cal.board_frame_from_center([centre[0], centre[1], TABLE_Z + THICKNESS], self.board)
        eye, target = np.array([0.50, 0.0, 0.25]), np.array([centre[0], centre[1], TABLE_Z])
        z = (target - eye) / np.linalg.norm(target - eye)
        x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
        self.T_base_fixed = np.eye(4)
        self.T_base_fixed[:3, :3] = np.column_stack([x, np.cross(z, x), z])
        self.T_base_fixed[:3, 3] = eye


class SceneArm(FakeArmDriver):
    """Fake arm in motor angles: reported q = motor + deployment offset_rad, physical = motor + TRUE_ZERO.

    The lowest gripper mesh point cannot go below TABLE_Z (bisected onto it), like a real table contact.
    """

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
    """Active file before the workflow: right intrinsics, slightly wrong hand-eye, fixed extrinsic 3 cm off."""
    wrong_X = scene.X @ cal.make_transform(cal.rotvec_to_matrix([0.01, -0.01, 0.0]), [0.003, -0.002, 0.002])
    wrong_fixed = scene.T_base_fixed @ cal.make_transform(np.eye(3), [0.03, 0.0, 0.0])
    cal.write_calibration_file(path, board=cal.DEFAULT_BOARD_SPEC, samples=10, created_at="2026-11-01T00:00:00Z",
                               cameras={"wrist": dict(K=K, distortion=[0] * 5, image_size=list(SIZE), T_link_camera=wrong_X,
                                                      intrinsics_rms_px=0.2, hand_eye_residual=dict(translation_m=0.003, rotation_rad=0.01)),
                                        "fixed": dict(K=K, distortion=[0] * 5, image_size=list(SIZE), T_base_camera=wrong_fixed,
                                                      intrinsics_rms_px=0.2, extrinsic_residual=dict(reprojection_px=4.0))},
                               joint_calibration={"table_z_m": -0.0185, "joint_offsets_rad": {name: 0.0 for name in ARM_JOINTS}})


@functools.lru_cache(maxsize=None)
def _cached_scene_and_example():
    return Scene(), EXAMPLE.read_text()


def make_workflow_adapter(directory):
    """Adapter over a temporary deployment file (written by the joint_mapping writer) and the scene."""
    directory = Path(directory)
    camera_file = directory / "camera_calibration.json"
    base_scene, example_text = _cached_scene_and_example()
    scene = copy.copy(base_scene)
    scene.T_base_fixed = base_scene.T_base_fixed.copy()
    write_initial_calibration(camera_file, scene)
    document = json.loads(example_text)
    document["camera_calibration_file"] = str(camera_file)
    for name in ("wrist", "fixed"):
        document["cameras"][name].update(width=SIZE[0], height=SIZE[1])
    document["home"]["q_rad"] = HOME.tolist()
    document["motion"].update(max_joint_speed_rad_s=1.0, max_command_delta_rad=0.06)
    document["calibration"].update(pan_offsets_rad=[-0.12, 0.12], flex_offsets_rad=[-0.1, 0.1], roll_offsets_rad=[-0.3, 0.3],
                                   lift_offsets_rad=[-0.06, 0.06], speed_scale=1.0, min_samples=8, settle_timeout_s=0.5,
                                   touch_points_xy_m=[[0.20, 0.0], [0.24, 0.06], [0.24, -0.06]],
                                   verify_points_xy_m=[[0.22, 0.03], [0.26, -0.04]], touch_speed_scale=1.0,
                                   touch_contact_load_delta=0.0, board_thickness_m=THICKNESS, extrinsic_pose_count=5)
    document["calibration"]["touch_video_fps"] = 0.0  # fake-arm solver tests do not archive video
    deployment_path = directory / "so101.local.json"
    deployment_path.write_text(json.dumps(document, indent=2))
    deployment = load_deployment(deployment_path)
    clock = FakeClock()
    arm = SceneArm(deployment, clock, q_rad=(HOME - TRUE_ZERO).tolist())
    cameras = {name: SceneCamera(name, clock, scene, arm) for name in ("wrist", "fixed")}
    adapter = SO101Adapter(deployment, directory / "robot", "rgb_proprio_calibrated_v1", driver=arm, cameras=cameras,
                           clock=clock, image_settings=Config().images.to_dict(), start_telemetry=False,
                           deployment_path=deployment_path)
    (adapter.run_dir / "preflight.json").write_text(json.dumps({"ok": True, "checked_at": "2026-11-09T00:00:00Z"}))
    return adapter, arm, scene, deployment_path, camera_file


OPERATOR_OK = ["table_clear", "workspace_clear", "board_matches", "touches_physically_confirmed"]


def start(adapter, step, **params):
    return adapter.operator("calib_step_start", step=step, params=dict({"confirmed": OPERATOR_OK}, **params))


@unittest.skipUnless(HAVE_ARUCO, "OpenCV with aruco CharucoDetector required")
class WorkflowEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, True)
        self.adapter, self.arm, self.scene, self.deployment_path, self.camera_file = make_workflow_adapter(self.directory)

    def test_s5_forwards_and_archives_explicit_solver_configuration(self):
        requested = {"joint_solver_max_nfev": 401, "joint_pixel_robust_scale": 3.01,
                     "joint_solver_ftol": 1.1e-8, "joint_solver_xtol": 1.2e-8,
                     "joint_solver_gtol": 1.3e-8, "joint_solver_x_scale": 1.01}
        self.adapter.deployment["calibration"].update(requested)
        flow = self.adapter.workflow
        flow.begin("first_run", board=dict(flow.default_board(), thickness_m=THICKNESS))
        for step in ("S0", "S1", "S2", "S4", "S5"):
            result = start(self.adapter, step)
            self.assertEqual(result["status"], "passed", result)
        effective = result["result"]["solver"]["effective_settings"]
        for key, target in (("joint_solver_max_nfev", "max_nfev"), ("joint_pixel_robust_scale", "pixel_robust_scale"),
                            ("joint_solver_ftol", "ftol"), ("joint_solver_xtol", "xtol"),
                            ("joint_solver_gtol", "gtol"), ("joint_solver_x_scale", "x_scale")):
            self.assertEqual(effective[target], requested[key])
        self.assertGreater(result["result"]["solver"]["nfev"], 0)
        report = json.loads((flow._run_dir() / "reports" / "S5.json").read_text())
        self.assertEqual(report["effective_calibration_settings"], flow.run["calibration_settings"])
        self.assertEqual(report["result"]["solver"]["effective_settings"], effective)
        self.assertEqual(flow.run["candidate"]["source_binding"]["calibration_settings"], flow.settings)

    def test_first_run_recovers_offsets_applies_verifies_and_rolls_back(self):
        adapter = self.adapter
        original_camera = self.camera_file.read_bytes()
        original_deployment = self.deployment_path.read_text()
        status = adapter.operator("calib_workflow_status")
        self.assertIsNone(status["run"])
        self.assertTrue(status["bundle"]["present"])
        self.assertEqual(status["staleness"]["status"], "shifted", status["staleness"])  # fixed extrinsic 3 cm off
        begun = adapter.operator("calib_workflow_begin", mode="first_run",
                                 board=dict(status["default_board"], thickness_m=THICKNESS))
        self.assertEqual(begun["status"], "started")
        self.assertEqual(begun["run"]["plan"], ["S0", "S1", "S2", "S3", "S4", "S5", "S6"])

        refused = adapter.operator("calib_step_start", step="S1", params={"confirmed": []})
        self.assertEqual(refused["status"], "rejected")
        self.assertIn("readiness_failed:torque_on", refused["reason"])  # torque not yet confirmed by S0

        s0 = start(adapter, "S0")
        self.assertEqual(s0["status"], "passed", s0)
        sends = self.arm.sends
        refused = adapter.operator("calib_step_start", step="S1", params={"confirmed": ["table_clear"]})
        self.assertEqual(refused["reason"], "readiness_unconfirmed:workspace_clear")
        self.assertEqual(self.arm.sends, sends)  # a refused start never moves the arm

        s1 = start(adapter, "S1")
        self.assertEqual(s1["status"], "passed", s1)
        self.assertEqual(s1["result"]["measures"]["contacts"], 3)
        s2 = start(adapter, "S2")
        self.assertEqual(s2["status"], "passed", s2["result"].get("verdict"))
        s4 = start(adapter, "S4")
        self.assertEqual(s4["status"], "passed", s4)
        self.assertEqual(s4["result"]["fixed_intrinsics_source"], "active")  # S3 not run: active file intrinsics
        s5 = start(adapter, "S5")
        self.assertEqual(s5["status"], "passed", s5["result"]["verdict"])
        joint = s5["result"]["joint_calibration"]
        recovered = np.array([joint["joint_offsets_rad"][name] for name in ARM_JOINTS])
        estimated = [ARM_JOINTS.index(name) for name in ("shoulder_lift", "elbow_flex", "wrist_flex")]
        np.testing.assert_allclose(recovered[estimated], TRUE_ZERO[estimated], atol=0.006)
        self.assertAlmostEqual(joint["table_z_m"], TABLE_Z, delta=0.002)
        diff = {row["key"]: row for row in s5["result"]["diff"]}
        self.assertEqual(diff["table_z_m"]["old"], -0.0185)
        self.assertEqual(diff["joint_offset.elbow_flex"]["old"], 0.0)

        # S6 before apply: refused, nothing moves.
        sends = self.arm.sends
        self.assertIn("readiness_failed:applied", start(adapter, "S6")["reason"])
        self.assertEqual(self.arm.sends, sends)

        held = adapter.arm_goal and self.arm.goal[:]
        applied = adapter.operator("calib_apply")
        self.assertEqual(applied["status"], "applied", applied)
        self.assertEqual(applied["table_z_m"], joint["table_z_m"])
        self.assertEqual((applied["table_z_unit"], applied["table_z_frame"]), ("m", "so101_base"))
        np.testing.assert_allclose(self.arm.goal, held, atol=1e-9)  # offset switch never moves the arm
        self.assertTrue(applied["imported_version"].endswith("-imported"))
        backup = Path(applied["deployment_backup"])
        self.assertEqual(backup.read_text(), original_deployment)
        written = load_deployment(self.deployment_path)["joint_mapping"]
        self.assertAlmostEqual(written["elbow_flex"]["offset_rad"], joint["joint_offsets_rad"]["elbow_flex"], places=5)
        self.assertAlmostEqual(adapter.deployment["joint_mapping"]["elbow_flex"]["offset_rad"],
                               joint["joint_offsets_rad"]["elbow_flex"])
        active = json.loads(self.camera_file.read_text())
        self.assertAlmostEqual(active["joint_calibration"]["table_z_m"], joint["table_z_m"])
        self.assertEqual(adapter.observe()["camera_calibration"]["calibrated_at"], active["created_at"])

        s6 = start(adapter, "S6")
        self.assertEqual(s6["status"], "passed", s6["result"])
        self.assertLess(s6["result"]["measures"]["max_verify_error_m"], 0.002)
        status = adapter.operator("calib_workflow_status")
        bundle = status["bundle"]
        self.assertEqual(bundle["active_version"], applied["version"])
        self.assertTrue(bundle["matches_active_version"])
        self.assertTrue(bundle["has_joint_calibration"])
        self.assertTrue(bundle["versions"][0]["verification"]["passed"])
        self.assertEqual(bundle["verification"], bundle["versions"][0]["verification"])
        with patch.object(adapter.workflow.bundle, "versions", return_value=[]):
            self.assertEqual(adapter.workflow.bundle_status()["verification"], bundle["verification"])
        with patch.object(adapter.workflow.bundle, "read_version_json", return_value=None):
            self.assertIsNone(adapter.workflow.bundle_status()["verification"])
        self.assertEqual(set(bundle["versions"][0]["steps"]), {"S0", "S1", "S2", "S4", "S5"})  # steps done at apply
        adapter.workflow._stale_checked_at = None
        adapter.operator("go_home")
        adapter.workflow._stale_checked_at = None
        self.assertEqual(adapter.operator("calib_workflow_status")["staleness"]["status"], "ok")
        version_dir = adapter.workflow.bundle.versions_dir / applied["version"]
        for name in ("camera_calibration.json", "joint_calibration.json", "touches.json", "reports/S5.json", "manifest.json"):
            self.assertTrue((version_dir / name).is_file(), name)

        rolled = adapter.operator("calib_rollback")
        self.assertEqual((rolled["status"], rolled["version"]), ("rolled_back", applied["imported_version"]), rolled)
        self.assertEqual(rolled["table_z_m"], -0.0185)
        self.assertEqual((rolled["table_z_unit"], rolled["table_z_frame"]), ("m", "so101_base"))
        self.assertEqual(self.camera_file.read_bytes(), original_camera)  # joint_calibration record restored with it
        self.assertEqual(load_deployment(self.deployment_path)["joint_mapping"]["elbow_flex"]["offset_rad"], 0.0)
        self.assertEqual(adapter.deployment["joint_mapping"]["elbow_flex"]["offset_rad"], 0.0)
        self.assertEqual(adapter.calibration_identity["created_at"], "2026-11-01T00:00:00Z")
        self.assertIsNone(adapter.operator("calib_workflow_status")["bundle"]["verification"])
        self.assertIn("readiness_failed:applied", start(adapter, "S6")["reason"])  # rolled back: re-apply first

    def test_partial_recalibrate_fixed_moved_preserves_wrist_and_joint_offsets(self):
        adapter = self.adapter
        adapter.operator("calib_workflow_begin", mode="first_run",
                         board=dict(adapter.workflow.default_board(), thickness_m=THICKNESS))
        for step in ("S0", "S1", "S2", "S4", "S5"):
            res = start(adapter, step)
            self.assertEqual(res["status"], "passed", (step, res))
        applied1 = adapter.operator("calib_apply")
        self.assertEqual(applied1["status"], "applied", applied1)
        self.assertEqual(start(adapter, "S6")["status"], "passed")
        doc1 = cal.read_calibration_file(self.camera_file)
        wrist_X1 = np.asarray(doc1["cameras"]["wrist"]["T_link_camera"], float).copy()
        offsets1 = dict(doc1["joint_calibration"]["joint_offsets_rad"])
        table1 = float(doc1["joint_calibration"]["table_z_m"])

        # Move fixed camera by 4 cm and run partial recalibration ("fixed_moved": S4 -> S5 -> apply -> S6).
        self.scene.T_base_fixed = self.scene.T_base_fixed @ cal.make_transform(np.eye(3), [0.04, -0.015, 0.01])
        begun = adapter.operator("calib_workflow_begin", mode="recalibrate", changes=["fixed_moved"],
                                 board=dict(adapter.workflow.default_board(), thickness_m=THICKNESS))
        self.assertEqual(begun["run"]["plan"], ["S4", "S5", "S6"])
        s4 = start(adapter, "S4")
        self.assertEqual(s4["status"], "passed", s4)
        s5 = start(adapter, "S5")
        self.assertEqual(s5["status"], "passed", s5["result"]["verdict"])
        applied2 = adapter.operator("calib_apply")
        self.assertEqual(applied2["status"], "applied", applied2)
        s6 = start(adapter, "S6")
        self.assertEqual(s6["status"], "passed", s6)

        doc2 = cal.read_calibration_file(self.camera_file)
        wrist_X2 = np.asarray(doc2["cameras"]["wrist"]["T_link_camera"], float)
        offsets2 = dict(doc2["joint_calibration"]["joint_offsets_rad"])
        table2 = float(doc2["joint_calibration"]["table_z_m"])
        fixed_T2 = np.asarray(doc2["cameras"]["fixed"]["T_base_camera"], float)

        np.testing.assert_allclose(wrist_X2, wrist_X1, atol=1e-9)
        for name in ARM_JOINTS:
            self.assertAlmostEqual(offsets2[name], offsets1[name], places=9)
        self.assertAlmostEqual(table2, table1, places=9)
        self.assertLess(float(np.linalg.norm(fixed_T2[:3, 3] - self.scene.T_base_fixed[:3, 3])), 0.005)

    def test_backend_operator_hot_reloads_deployment_on_apply_and_rollback(self):
        from ni_rua.robot.backends.so101.backend import RobotBackend
        from ni_rua.robot.backends.so101.deployment import write_joint_offsets
        from ni_rua.robot.model.embodiments import SO101
        cfg = Config.from_dict({"backend": {"environment": "so101_real", "robot_model_id": SO101.robot_model_id,
                                            "deployment": str(self.deployment_path)},
                                "features": {"motion_execution_mode": "resampled",
                                             "observation_profile": "rgb_proprio_calibrated_v1"}})
        backend = RobotBackend(config=cfg, run_dir=self.directory / "parent_run")
        self.assertEqual(backend.deployment["joint_mapping"]["elbow_flex"]["offset_rad"], 0.0)
        write_joint_offsets(self.deployment_path, {"elbow_flex": -0.029})
        with patch.object(backend, "_request", return_value={"status": "applied"}):
            backend.operator("calib_apply")
        self.assertAlmostEqual(backend.deployment["joint_mapping"]["elbow_flex"]["offset_rad"], -0.029, places=6)
        write_joint_offsets(self.deployment_path, {"elbow_flex": 0.0})
        with patch.object(backend, "_request", return_value={"status": "rolled_back"}):
            backend.operator("calib_rollback")
        self.assertAlmostEqual(backend.deployment["joint_mapping"]["elbow_flex"]["offset_rad"], 0.0, places=6)

    def test_rerun_marks_dependents_stale_and_candidate_unappliable(self):
        adapter = self.adapter
        adapter.operator("calib_workflow_begin", mode="recalibrate", changes=["fixed_moved"])
        self.assertEqual(adapter.workflow.run["plan"], ["S4", "S5", "S6"])
        start(adapter, "S0")
        self.assertEqual(start(adapter, "S1")["status"], "passed")
        self.assertEqual(start(adapter, "S4")["status"], "passed")
        self.assertEqual(start(adapter, "S5")["status"], "passed")
        self.assertEqual(start(adapter, "S4")["status"], "passed")  # re-run: S5 (and its candidate) go stale
        steps = adapter.operator("calib_workflow_status")["run"]["steps"]
        self.assertEqual((steps["S5"]["status"], steps["S5"]["reason"]), ("stale", "S4_rerun"))
        self.assertEqual(steps["S1"]["status"], "passed")
        refused = adapter.operator("calib_apply")
        self.assertEqual(refused["status"], "rejected")
        self.assertIn("candidate_stale", refused["reason"])
        self.assertEqual(adapter.workflow.bundle.active_version(), None)

    def test_install_failure_keeps_camera_file_offsets_and_pointer(self):
        adapter = self.adapter
        adapter.operator("calib_workflow_begin", mode="recalibrate", changes=["fixed_moved"])
        start(adapter, "S0"); start(adapter, "S1"); start(adapter, "S4")
        self.assertEqual(start(adapter, "S5")["status"], "passed")
        camera, deployment_text = self.camera_file.read_bytes(), self.deployment_path.read_text()
        real_reload = adapter.reload_calibration
        calls = []

        def failing_reload():
            calls.append(1)
            if len(calls) == 1:
                return {"status": "failed", "reason": "synthetic"}
            return real_reload()
        with patch.object(adapter, "reload_calibration", side_effect=failing_reload):
            result = adapter.operator("calib_apply")
        self.assertEqual(result["status"], "install_failed", result)
        self.assertEqual(self.camera_file.read_bytes(), camera)
        self.assertEqual(json.loads(self.deployment_path.read_text())["joint_mapping"],
                         json.loads(deployment_text)["joint_mapping"])
        self.assertEqual(adapter.deployment["joint_mapping"]["elbow_flex"]["offset_rad"], 0.0)
        self.assertTrue(result["active_version"].endswith("-imported"))  # pointer unchanged (import only)
        self.assertEqual(adapter.calibration_identity["created_at"], "2026-11-01T00:00:00Z")

    def test_hold_during_touch_stops_step_and_status_is_off_main_thread(self):
        adapter = self.adapter
        adapter.operator("calib_workflow_begin", mode="first_run")
        start(adapter, "S0")
        stops = []

        def progress(update):
            if update.get("phase") == "descending" and not stops:
                stops.append(adapter.operator("hold"))
                # status/readiness never touch the bus and never block on the running step
                with patch.object(self.arm, "read", side_effect=AssertionError("status must not read the bus")):
                    snapshot = adapter.operator("calib_workflow_status")
                self.assertEqual(snapshot["running_step"], "S1")
        result = adapter.operator("calib_step_start", step="S1", params={"confirmed": OPERATOR_OK}, progress=progress)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result.get("stopped"))
        self.assertEqual(stops[0]["status"], "held")
        self.assertEqual(adapter.workflow.run["steps"]["S1"]["status"], "failed")
        self.assertIsNone(adapter.workflow.running_step)
        self.assertIsNone(adapter.active_action_id)

    def test_refusals(self):
        adapter = self.adapter
        self.assertEqual(start(adapter, "S0")["reason"], "calibration_run_not_started")
        adapter.operator("calib_workflow_begin", mode="first_run")
        self.assertEqual(start(adapter, "S9")["reason"], "calibration_step_unknown:S9")
        adapter.active_action_id = "act-1"
        self.assertEqual(start(adapter, "S0")["reason"], "action_active")
        self.assertEqual(adapter.operator("calib_apply")["reason"], "action_active")
        adapter.active_action_id = None
        self.assertIn("no_candidate", adapter.operator("calib_apply")["reason"])
        self.assertIn("rollback_target_unknown", adapter.operator("calib_rollback")["reason"])
        bad = start(adapter, "S2", board=dict(adapter.workflow.default_board(), marker_m=0.05))
        self.assertIn("board_spec_invalid", bad["reason"])
        self.assertEqual(adapter.operator("calib_workflow_begin", mode="recalibrate", changes=[])["status"], "rejected")
        with self.assertRaisesRegex(ValueError, "unexpected_arguments"):
            adapter.operator("calib_apply", force=True)
        self.assertEqual(adapter.operator("calib_step_cancel")["status"], "idle")


@unittest.skipUnless(HAVE_ARUCO, "OpenCV with aruco CharucoDetector required")
class HandheldStepTests(unittest.TestCase):
    def test_s3_capture_and_solve_feed_s4(self):
        from robot.so101.test_so101_handheld import K_FIXED, handheld_frames
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, True)
        adapter, _, _, _, _ = make_workflow_adapter(directory)
        adapter.deployment["calibration"].update(handheld_target_views=10, handheld_min_novelty=0.02)
        adapter.operator("calib_workflow_begin", mode="recalibrate", changes=["fixed_replaced"])
        self.assertEqual(adapter.workflow.run["plan"], ["S3", "S4", "S5", "S6"])
        frames = handheld_frames(12)
        adapter.cameras["fixed"].queue = list(frames)
        result = start(adapter, "S3")
        self.assertEqual(result["status"], "running", result)
        self.assertIn("S3_capture_running", start(adapter, "S0")["reason"])  # one step at a time
        deadline = time.monotonic() + 60
        while adapter.workflow.capture.running and adapter.workflow.capture.frames < len(frames):
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.01)
        solved = start(adapter, "S3", action="solve")
        self.assertEqual(solved["status"], "passed", solved)
        fixed = adapter.workflow.run["steps"]["S3"]["result"]["fixed"]
        self.assertAlmostEqual(np.asarray(fixed["K"])[0, 0], K_FIXED[0, 0], delta=K_FIXED[0, 0] * 0.03)
        self.assertEqual(adapter.workflow._fixed_intrinsics()[1], "S3")


@unittest.skipUnless(HAVE_ARUCO, "OpenCV with aruco CharucoDetector required")
class WorkflowBindingTests(unittest.TestCase):
    """Fast file/state regressions; no physical device or expensive joint solve."""

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, True)
        self.adapter, self.arm, _, self.deployment_path, self.camera_file = make_workflow_adapter(self.directory)
        self.flow = self.adapter.workflow
        self.flow.begin("first_run", board=dict(self.flow.default_board(), thickness_m=0.003))
        self.adapter.torque_known = True
        mapping = {n: dict(self.adapter.deployment["joint_mapping"][n]) for n in ARM_JOINTS}
        self.version = self.flow.bundle.create_version(camera_file=self.camera_file, manifest={
            "joint_offsets_rad": self.flow._offsets(), "joint_mapping": mapping, "table_z_m": -0.0185})
        self.flow.bundle.set_active(self.version)
        self.flow.run.update(applied_version=self.version, candidate={"table_z_m": 0.9})
        self.report = {"status": "completed", "physics_advanced": True, "contacts": 1, "planned_points": 1,
                       "touches": [{"status": "contact", "contact_z_m": -0.0185}]}

    def test_resumed_settings_change_and_legacy_unknown_refuse_reuse_without_motion(self):
        from ni_rua.robot.calibration.workflow.runner import CalibrationWorkflow
        from copy import deepcopy
        baseline = deepcopy(self.flow.run["calibration_settings"])
        self.flow._step("S4")["status"] = "passed"
        self.flow._save()
        resumed = CalibrationWorkflow(self.adapter)
        self.assertEqual(resumed._run_settings_binding()["status"], "matched")
        self.adapter.deployment["calibration"]["joint_solver_max_nfev"] += 1
        self.assertEqual(resumed.run["calibration_settings"], baseline)
        self.assertEqual(resumed._run_settings_binding()["changed_keys"], ["joint_solver_max_nfev"])
        sends = self.arm.sends
        with patch.object(resumed, "_s5", side_effect=AssertionError("must not reuse old stages")):
            rejected = resumed.start_step("S5", params={"confirmed": OPERATOR_OK})
        self.assertEqual(rejected["status"], "rejected")
        self.assertIn("calibration_settings_changed", rejected["reason"])
        self.assertEqual(self.arm.sends, sends)
        self.assertEqual(resumed._step("S4")["status"], "passed")  # preserve historical evidence
        self.adapter.deployment["calibration"].update(baseline)
        resumed.run.pop("calibration_settings")
        resumed._save()
        legacy = CalibrationWorkflow(self.adapter)
        self.assertEqual(legacy.status()["settings_binding"]["status"], "unknown")
        rejected = legacy.start_step("S5", params={"confirmed": OPERATOR_OK})
        self.assertIn("calibration_settings_unrecorded", rejected["reason"])
        self.assertEqual(self.arm.sends, sends)

    def test_unknown_run_settings_preserve_bound_historical_verification_badge(self):
        from ni_rua.apps.gateway.extensions.calibration import calibration_badge
        from ni_rua.robot.calibration.workflow.bundle import write_json
        manifest = self.flow.bundle.manifest(self.version)
        manifest["verification"] = {"passed": True, "max_error_m": .002}
        write_json(self.flow.bundle.version_file(self.version, "manifest.json"), manifest)
        before = calibration_badge({"bundle": self.flow.bundle_status(), "staleness": {"status": "ok"}})
        self.assertEqual(before["status"], "valid")
        self.flow.run.pop("calibration_settings")
        self.assertEqual(self.flow._run_settings_binding()["status"], "unknown")
        after = calibration_badge({"bundle": self.flow.bundle_status(), "staleness": {"status": "ok"}})
        self.assertEqual(after, before)

    def test_board_default_sources_and_explicit_run_override_are_recorded(self):
        self.adapter.deployment["board"] = None
        adjacent = self.camera_file.parent / "board.json"
        adjacent.write_text(json.dumps({"thickness_m": .0032}))
        board, source = self.flow._default_board_with_source()
        self.assertEqual(board["thickness_m"], .0032)
        self.assertEqual(source["fields"]["square_m"], "active_camera_calibration.board")
        self.assertEqual(source["fields"]["thickness_m"], "adjacent_board.json")
        self.flow.begin("first_run")
        self.assertEqual(self.flow.run["board_source"], source)
        self.flow.begin("first_run", board=dict(board, thickness_m=.003))
        self.assertEqual(self.flow.run["board"]["thickness_m"], .003)
        self.assertEqual(self.flow.run["board_source"]["thickness"], "operator_begin_request")

    def test_s3_pending_writer_blocks_solve_restart_stop_and_cancel_without_motion(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        pending = {"status": "unknown", "reason": "writer_shutdown_pending", "stop_confirmed": False,
                   "kept": 20, "image_size": [640, 480]}
        capture = SimpleNamespace(running=True, thread=SimpleNamespace(is_alive=lambda: True),
                                  stop=Mock(return_value=pending), kept_images=Mock())
        self.flow.capture = self.adapter.handheld = capture
        sends = self.arm.sends
        for action in ("solve", "capture", "stop", "cancel"):
            with self.subTest(action=action):
                self.flow._step("S3")["status"] = "running"
                with patch("ni_rua.robot.calibration.workflow.capture.handheld_intrinsics",
                           side_effect=AssertionError("must not solve while writer may mutate images")), \
                     patch("ni_rua.robot.calibration.workflow.capture.HandheldCapture",
                           side_effect=AssertionError("must not replace live capture")):
                    result = self.flow.cancel() if action == "cancel" else self.flow._s3(action, None)
                self.assertEqual(result["status"], "failed")
                self.assertFalse(result["physics_advanced"])
                self.assertFalse(result["capture_stop_confirmed"])
                self.assertEqual(result["capture"]["status"], "unknown")
                self.assertEqual(self.flow._step("S3")["status"], "failed")
                self.assertIs(self.flow.capture, capture)
        capture.kept_images.assert_not_called()
        self.assertEqual(self.arm.sends, sends)

    def test_s3_requires_receipt_and_thread_exit_then_allows_retry(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        stopped = {"status": "stopped", "reason": "stopped_for_solve", "stop_confirmed": True,
                   "kept": 20, "image_size": [640, 480]}
        alive = [True]
        capture = SimpleNamespace(running=False, thread=SimpleNamespace(is_alive=lambda: alive[0]),
                                  stop=Mock(return_value=stopped))
        self.flow.capture = capture
        self.flow._step("S3")["status"] = "running"
        with patch("ni_rua.robot.calibration.workflow.capture.handheld_intrinsics",
                   return_value=(K, np.zeros(5), 0.1, 20)) as solve:
            self.assertEqual(self.flow._s3("solve", None)["status"], "failed")
            solve.assert_not_called()  # a false stop receipt cannot override an alive thread
            alive[0] = False
            stopped["stop_confirmed"] = False
            self.assertEqual(self.flow._s3("solve", None)["status"], "failed")
            solve.assert_not_called()  # absence of an alive thread does not invent confirmation
            stopped["stop_confirmed"] = True
            result = self.flow._s3("solve", None)
            self.assertEqual(result["status"], "passed", result)
            solve.assert_called_once_with(capture, self.flow.settings["handheld_min_views"])

    def test_s6_uses_installed_version_table_not_mutable_candidate(self):
        with patch.object(self.flow, "_touch", return_value=self.report):
            result = self.flow._s6(lambda _: None)
        self.assertEqual(result["table_z_m"], -0.0185)
        self.assertEqual(result["errors_m"], [0.0])
        verification = self.flow.bundle.manifest(self.version)["verification"]
        self.assertTrue(verification["passed"])
        self.assertEqual(verification["scope"], "holdout_model_consistency")
        self.assertIsNone(verification["independent_physical_accuracy"])

    def test_install_persists_offsets_smaller_than_old_equality_epsilon_exactly(self):
        previous = self.flow._offsets()
        target = dict(previous, wrist_flex=5e-10)
        document = json.loads(self.camera_file.read_text())
        document["joint_calibration"]["joint_offsets_rad"] = target
        candidate = self.directory / "tiny-offset-candidate.json"
        candidate.write_text(json.dumps(document))
        mapping = {n: dict(self.adapter.deployment["joint_mapping"][n], offset_rad=target[n]) for n in ARM_JOINTS}
        version = self.flow.bundle.create_version(camera_file=candidate, manifest={
            "joint_offsets_rad": target, "joint_mapping": mapping, "table_z_m": -0.0185})
        self.flow._install(version, target, previous)
        self.assertEqual(self.flow._offsets(), target)
        self.assertEqual(load_deployment(self.deployment_path)["joint_mapping"]["wrist_flex"]["offset_rad"], 5e-10)
        self.flow.run["applied_version"] = version
        self.assertEqual(self.flow._verification_binding()["joint_mapping"], mapping)

    def test_legacy_rounding_diagnostic_is_exact_and_never_claims_unknown_mapping(self):
        recorded = dict(self.flow._offsets(), wrist_flex=4.53883009187e-7)
        result = self.flow._binding_diagnostics({"joint_offsets_rad": recorded})
        self.assertFalse(result["offsets_exact"])
        self.assertTrue(result["matches_legacy_six_decimal_serialization"])
        self.assertFalse(result["joint_mapping_recorded"])
        self.assertEqual(result["max_abs_offset_difference_rad"], recorded["wrist_flex"])
        self.assertEqual(result["recovery"], "new_acquisition_solve_apply")
        # A smaller difference is not accepted as legacy serialization merely for being tiny.
        self.adapter.deployment["joint_mapping"]["wrist_flex"]["offset_rad"] = 3e-7
        self.assertFalse(self.flow._binding_diagnostics({"joint_offsets_rad": recorded})[
            "matches_legacy_six_decimal_serialization"])

    def test_apply_rejects_changed_or_unrecorded_candidate_source_without_writes(self):
        from copy import deepcopy
        self.flow._step("S5")["status"] = "passed"
        self.flow.run["applied_version"] = None
        candidate = {"file": str(self.camera_file), "table_z_m": -0.0185,
                     "joint_offsets_rad": self.flow._offsets(), "stale": False, "diff": []}
        baseline = self.flow._candidate_source_binding()
        original_camera = self.camera_file.read_bytes()
        original_board = deepcopy(self.flow.run["board"])
        original_mapping = deepcopy(self.adapter.deployment["joint_mapping"])
        original_deployment = self.deployment_path.read_text()
        def change_persisted_settings():
            document = json.loads(original_deployment)
            document["calibration"]["joint_solver_max_nfev"] = baseline["calibration_settings"]["joint_solver_max_nfev"] + 1
            self.deployment_path.write_text(json.dumps(document))
        mutations = {
            "unrecorded": lambda: candidate.pop("source_binding"),
            "mapping_sign": lambda: self.adapter.deployment["joint_mapping"]["wrist_flex"].update(sign=-1),
            "installed_camera": lambda: self.camera_file.write_bytes(original_camera + b"\n"),
            "board": lambda: self.flow.run["board"].update(thickness_m=0.004),
            "persisted_settings": change_persisted_settings,
        }
        for name, mutate in mutations.items():
            with self.subTest(name):
                candidate["source_binding"] = deepcopy(baseline)
                self.flow.run["candidate"] = candidate
                mutate()
                with patch.object(self.flow.bundle, "create_version", side_effect=AssertionError("must not create version")), \
                     patch("ni_rua.robot.calibration.workflow.runner.write_joint_offsets",
                           side_effect=AssertionError("must not write mapping")):
                    result = self.flow.apply()
                self.assertEqual(result["status"], "rejected")
                self.assertIn("candidate_source_changed_or_unrecorded", result["reason"])
                self.assertEqual(self.flow.bundle.active_version(), self.version)
                self.camera_file.write_bytes(original_camera)
                self.flow.run["board"] = deepcopy(original_board)
                self.adapter.deployment["joint_mapping"] = deepcopy(original_mapping)
                self.deployment_path.write_text(original_deployment)

    def test_apply_rejects_modified_reviewed_artifact_before_install(self):
        directory = self.flow._run_dir() / "candidate"
        directory.mkdir()
        camera = directory / "camera_calibration.json"
        camera.write_bytes(self.camera_file.read_bytes())
        (directory / "touches.json").write_text('{"touches":[]}')
        candidate = {"file": str(camera), "table_z_m": -0.0185,
                     "joint_offsets_rad": self.flow._offsets(), "stale": False, "diff": [],
                     "source_binding": self.flow._candidate_source_binding()}
        candidate["artifact_binding"] = self.flow._candidate_artifact_binding(candidate)
        self.flow.run.update(candidate=candidate, applied_version=None)
        self.flow._step("S5")["status"] = "passed"
        document = json.loads(camera.read_text())
        document["joint_calibration"]["table_z_m"] += 0.001
        camera.write_text(json.dumps(document))
        with patch.object(self.flow, "_install", side_effect=AssertionError("must not install changed artifact")), \
             patch.object(self.flow.bundle, "create_version", side_effect=AssertionError("must not archive changed candidate")):
            result = self.flow.apply()
        self.assertEqual(result["status"], "rejected")
        self.assertIn("candidate_artifact_changed", result["reason"])
        self.assertEqual(self.flow.bundle.active_version(), self.version)

    def test_failed_or_stale_camera_stages_cannot_fall_back_to_active(self):
        from ni_rua.robot.calibration.workflow.runner import StepRefused
        for step, getter, readiness in (("S2", self.flow._wrist, "wrist_calibration_available"),
                                        ("S3", self.flow._fixed_intrinsics, "fixed_calibration_available")):
            for status in ("failed", "stale", "running"):
                with self.subTest(step=step, status=status):
                    self.flow._step(step)["status"] = status
                    with self.assertRaisesRegex(StepRefused, f"calibration_source_unusable:{step}:{status}"):
                        getter()
                    self.assertFalse(self.flow._auto_check(readiness, False)[0])
            self.flow._step(step)["status"] = "not_started"

    def test_recalibration_required_inputs_block_but_unplanned_inputs_can_reuse_active(self):
        from ni_rua.robot.calibration.workflow.runner import StepRefused
        self.flow.run.update(mode="recalibrate", plan=["S3", "S4", "S5", "S6"])
        self.assertEqual(self.flow._wrist()[1], "active")  # unchanged, outside the recalibration plan
        with self.assertRaisesRegex(StepRefused, "calibration_source_required_by_plan:S3"):
            self.flow._fixed_intrinsics()
        self.flow.run["plan"] = ["S1", "S2", "S4", "S5", "S6"]
        with self.assertRaisesRegex(StepRefused, "calibration_source_required_by_plan:S2"):
            self.flow._wrist()
        with self.assertRaisesRegex(StepRefused, "calibration_source_required_by_plan:S1"):
            self.flow._touch_report()
        self.assertEqual(self.flow._fixed_intrinsics()[1], "active")

    def test_first_run_manual_active_reuse_is_explicit_and_not_full_plan_completion(self):
        self.assertEqual(self.flow._fixed_intrinsics()[1], "active")
        self.assertEqual(self.flow._source_evidence("S3", "active"),
                         {"source": "active", "step": "S3", "step_status": "not_started", "planned": True})
        status = self.flow.status()
        self.assertFalse(status["plan_completion"]["all_steps_passed"])
        self.assertIn("S3", status["plan_completion"]["not_passed_steps"])
        self.assertIn("source=active; S3=not_started", status["readiness"]["S4"]["fixed_calibration_available"]["detail"])

    def test_s6_rejects_stale_pointer_loaded_file_and_mapping_before_motion(self):
        from copy import deepcopy
        original_file = self.camera_file.read_bytes()
        original_camera = deepcopy(self.adapter.camera_calibration)
        original_mapping = deepcopy(self.adapter.deployment["joint_mapping"])
        mutations = {
            "pointer": lambda: self.flow.run.update(applied_version="old-version"),
            "installed_file": lambda: self.camera_file.write_bytes(original_file + b"\n"),
            "loaded_camera": lambda: self.adapter.camera_calibration["cameras"]["fixed"]["K"][0].__setitem__(0, 600),
            "offsets": lambda: self.adapter.deployment["joint_mapping"]["elbow_flex"].update(offset_rad=0.1),
            "sign": lambda: self.adapter.deployment["joint_mapping"]["elbow_flex"].update(sign=-1),
            "deployment_file": lambda: self.deployment_path.write_text(self.deployment_path.read_text().replace('"offset_rad": 0.0', '"offset_rad": 0.1', 1)),
        }
        original_deployment = self.deployment_path.read_text()
        for name, change in mutations.items():
            with self.subTest(name):
                change()
                sends = self.arm.sends
                result = start(self.adapter, "S6")
                self.assertEqual(result["status"], "rejected", result)
                self.assertIn("readiness_failed:applied", result["reason"])
                self.assertEqual(self.arm.sends, sends)
                self.assertNotIn("verification", self.flow.bundle.manifest(self.version))
                self.flow.run["applied_version"] = self.version
                self.camera_file.write_bytes(original_file)
                self.adapter.camera_calibration = deepcopy(original_camera)
                self.adapter.deployment["joint_mapping"] = deepcopy(original_mapping)
                self.deployment_path.write_text(original_deployment)

    def test_s6_change_during_capture_preserves_motion_without_writing_verdict(self):
        def changed(*args):
            self.flow.run["applied_version"] = "another-version"
            return self.report
        with patch.object(self.flow, "_touch", side_effect=changed):
            result = self.flow._s6(lambda _: None)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["physics_advanced"])
        self.assertEqual(result["motion_status"], "completed")
        self.assertFalse(result["verification_recorded"])
        self.assertEqual(result["captured_report"], self.report)
        self.assertNotIn("verification", self.flow.bundle.manifest(self.version))

    def test_legacy_version_without_full_mapping_is_readable_but_cannot_verify(self):
        from ni_rua.robot.calibration.workflow.bundle import write_json
        path = self.flow.bundle.version_file(self.version, "manifest.json")
        manifest = self.flow.bundle.manifest(self.version)
        manifest.pop("joint_mapping")
        write_json(path, manifest)
        self.assertEqual(self.flow.bundle.manifest(self.version)["table_z_m"], -0.0185)
        self.assertFalse(self.flow._auto_check("applied", False)[0])
        self.assertIn("joint_mapping_not_recorded", self.flow._auto_check("applied", False)[1])

    def test_s6_rejects_driver_mapping_drift(self):
        from copy import deepcopy
        self.arm.mapping = deepcopy(self.adapter.deployment["joint_mapping"])
        self.arm.mapping["wrist_flex"]["offset_rad"] += 0.01
        self.assertEqual(self.flow._auto_check("applied", False),
                         (False, "verification_calibration_mismatch:driver_mapping"))

    def test_status_keeps_historical_verification_without_lending_it_to_changed_binding(self):
        from copy import deepcopy
        from ni_rua.apps.gateway.extensions.calibration import calibration_badge
        from ni_rua.robot.calibration.workflow.bundle import write_json
        path = self.flow.bundle.version_file(self.version, "manifest.json")
        manifest = self.flow.bundle.manifest(self.version)
        verification = {"passed": True, "scope": "holdout_model_consistency", "max_error_m": 0.0}
        write_json(path, dict(manifest, verification=verification))
        # Historical active calibration can remain valid when a new workflow has not applied anything.
        self.flow.run["applied_version"] = None
        bundle = self.flow.bundle_status()
        self.assertTrue(bundle["matches_active_version"])
        self.assertEqual(bundle["verification"], verification)
        original_camera = deepcopy(self.adapter.camera_calibration)
        mutations = (
            lambda: self.adapter.deployment["joint_mapping"]["wrist_flex"].update(offset_rad=0.01),
            lambda: self.adapter.camera_calibration["cameras"]["fixed"]["K"][0].__setitem__(0, 600),
        )
        for mutate in mutations:
            mutate()
            with patch.object(self.arm, "read", side_effect=AssertionError("status must not read motors")):
                bundle = self.flow.bundle_status()
            self.assertFalse(bundle["matches_active_version"])
            self.assertIsNotNone(bundle["binding_error"])
            self.assertEqual(bundle["verification"], verification)
            self.assertEqual(calibration_badge({"bundle": bundle})["status"], "maybe_stale")
            self.adapter.deployment["joint_mapping"]["wrist_flex"]["offset_rad"] = 0.0
            self.adapter.camera_calibration = deepcopy(original_camera)

    def test_explicit_three_mm_board_survives_s5_and_restored_run(self):
        self.flow.settings["board_thickness_m"] = 0.008
        self.flow.begin("first_run", board=dict(self.flow.default_board(), thickness_m=0.003))
        self.flow._save()
        self.flow._load_latest_run()
        self.flow._step("S4").update(status="passed", result={"session_dir": "unused"})
        with patch.object(self.flow, "_touch_inputs", return_value=([], {}, "S1")), \
             patch.object(self.flow, "_session_samples", return_value=([], [])):
            result = self.flow._s5(lambda _: None)
        self.assertEqual(result["reason"], "no_wrist_board_views_for_joint_solve")
        self.assertEqual(self.flow.run["board"]["thickness_m"], 0.003)

    def test_legacy_touches_require_post_capture_attestation_and_bind_report(self):
        from ni_rua.core.contracts import digest
        from ni_rua.robot.calibration.workflow.bundle import write_json
        from ni_rua.robot.calibration.workflow.runner import StepRefused
        report = {"joint_offsets_rad": self.flow._offsets(), "touches": [{"status": "contact",
            "q_measured_rad": HOME.tolist(), "contact_link": "gripper", "contact_point_in_link_m": [0, 0, 0]}]}
        self.flow._step("S1")["status"] = "passed"
        path = self.flow._run_dir() / "S1_touches.json"
        write_json(path, report)
        with self.assertRaisesRegex(StepRefused, "touches_unreviewed"):
            self.flow._touch_inputs()
        report.pop("joint_offsets_rad")
        write_json(path, report)
        self.flow.run["contact_review"] = {"source": "S1", "report_digest": digest(report)}
        with self.assertRaisesRegex(StepRefused, "touch_basis_unknown"):
            self.flow._touch_inputs()
        report["joint_offsets_rad"] = self.flow._offsets()
        write_json(path, report)
        self.flow.run["contact_review"] = {"source": "S1", "report_digest": digest(report)}
        with self.assertRaisesRegex(StepRefused, "touch_basis_unknown"):
            self.flow._touch_inputs()  # offsets alone do not establish historical signs
        report["joint_mapping"] = {n: dict(self.adapter.deployment["joint_mapping"][n]) for n in ARM_JOINTS}
        write_json(path, report)
        self.flow.run["contact_review"] = {"source": "S1", "report_digest": digest(report)}
        self.assertEqual(len(self.flow._touch_inputs()[0]), 1)
        report["joint_offsets_rad"]["wrist_flex"] = 0.1
        write_json(path, report)
        with self.assertRaisesRegex(StepRefused, "touches_unreviewed"):
            self.flow._touch_inputs()

    def test_touch_rebase_reconstructs_same_encoder_values_across_sign_and_offset_change(self):
        from ni_rua.core.contracts import digest
        from ni_rua.robot.calibration.workflow.bundle import write_json
        from ni_rua.robot.calibration.workflow.runner import StepRefused
        old = {n: {"sign": -1.0 if i % 2 else 1.0, "offset_rad": 0.01 * i}
               for i, n in enumerate(ARM_JOINTS)}
        raw = [0.4, -0.2, 0.7, 0.5, -0.6]
        report = {"joint_mapping": old,
                  "joint_offsets_rad": {n: old[n]["offset_rad"] for n in ARM_JOINTS},
                  "touches": [{"status": "contact", "q_measured_rad": [old[n]["sign"] * raw[i] + old[n]["offset_rad"]
                               for i, n in enumerate(ARM_JOINTS)], "contact_link": "gripper",
                               "contact_point_in_link_m": [0, 0, 0]}]}
        self.flow._step("S1")["status"] = "passed"
        path = self.flow._run_dir() / "S1_touches.json"
        write_json(path, report)
        self.flow.run["contact_review"] = {"source": "S1", "report_digest": digest(report)}
        now = self.adapter.deployment["joint_mapping"]
        now["elbow_flex"].update(sign=-1, offset_rad=0.08)
        expected = [now[n]["sign"] * raw[i] + now[n]["offset_rad"] for i, n in enumerate(ARM_JOINTS)]
        np.testing.assert_allclose(self.flow._touch_inputs()[0][0][0], expected, atol=1e-15)
        report["joint_offsets_rad"]["elbow_flex"] += 0.001
        write_json(path, report)
        self.flow.run["contact_review"] = {"source": "S1", "report_digest": digest(report)}
        with self.assertRaisesRegex(StepRefused, "touch_basis_mismatch"):
            self.flow._touch_inputs()

    def test_s6_rechecks_binding_under_verdict_lock(self):
        real_binding = self.flow._verification_binding
        calls = []

        def binding():
            calls.append(1)
            if len(calls) == 3:
                self.flow.run["applied_version"] = "changed-after-postcheck"
            return real_binding()

        with patch.object(self.flow, "_touch", return_value=self.report), \
             patch.object(self.flow, "_verification_binding", side_effect=binding):
            result = self.flow._s6(lambda _: None)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["physics_advanced"])
        self.assertFalse(result["verification_recorded"])
        self.assertNotIn("verification", self.flow.bundle.manifest(self.version))

    def test_final_step_archive_failure_preserves_physics_stop_and_rejects_stale_touches(self):
        from ni_rua.robot.calibration.workflow.runner import StepRefused, write_json
        stale = {"touches": [{"status": "contact"}], "joint_mapping": {}}
        write_json(self.flow._run_dir() / "S1_touches.json", stale)
        write_json(self.flow.bundle.version_file(self.version, "touches.json"), stale)
        outcome = {"status": "unknown", "motion_status": "unknown", "physics_advanced": True,
                   "stop_confirmation": {"confirmed": False}, "measures": {"contacts": 1, "planned_points": 1}}
        for target in ("run.json", "S1.json"):
            with self.subTest(target=target):
                saves = []

                def write(path, value):
                    if path.name == "run.json":
                        saves.append(1)
                    if path.name == target and (target != "run.json" or len(saves) > 1):
                        raise OSError("synthetic final evidence write failed")
                    return write_json(path, value)

                with patch("ni_rua.robot.calibration.workflow.runner.write_json", side_effect=write):
                    result = self.flow._execute("S1", lambda _: outcome, None)
                self.assertEqual(result["status"], "unknown")
                self.assertTrue(result["physics_advanced"])
                self.assertEqual(result["stop_confirmation"], {"confirmed": False})
                self.assertEqual(result["result"]["motion_status"], "unknown")
                self.assertFalse(result["result"]["evidence_persisted"])
                self.assertEqual(self.flow._step("S1")["status"], "failed")
                with self.assertRaisesRegex(StepRefused, "current S1 did not pass"):
                    self.flow._touch_report()
                self.flow._load_latest_run()
                self.assertEqual(self.flow._step("S1")["status"], "failed")
                with self.assertRaisesRegex(StepRefused, "current S1 did not pass"):
                    self.flow._touch_report()

    def test_s1_touch_archive_failure_fails_gate_without_losing_confirmed_stop(self):
        report = dict(self.report, status="stopped", evidence_persisted=False, archive_error="disk full",
                      stop_confirmation={"confirmed": True}, table_z_nominal_m=None)
        with patch.object(self.flow, "_touch", return_value=report):
            result = self.flow._execute("S1", self.flow._s1, None)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["stopped"])
        self.assertTrue(result["physics_advanced"])
        self.assertEqual(result["result"]["motion_status"], "stopped")
        self.assertEqual(result["result"]["stop_confirmation"], {"confirmed": True})
        self.assertFalse(self.flow._auto_check("touches_available", False)[0])

    def test_report_write_failure_cannot_leave_completed_step_persisted_as_passed(self):
        from ni_rua.robot.calibration.workflow.runner import write_json
        outcome = {"status": "completed", "motion_status": "completed", "physics_advanced": True,
                   "stop_confirmation": {"confirmed": True}, "measures": {"contacts": 1, "planned_points": 1}}

        def write(path, value):
            if path.name == "S1.json":
                raise OSError("synthetic report write failed")
            return write_json(path, value)

        with patch("ni_rua.robot.calibration.workflow.runner.write_json", side_effect=write):
            result = self.flow._execute("S1", lambda _: outcome, None)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["physics_advanced"])
        self.assertEqual(result["result"]["motion_status"], "completed")
        self.flow._load_latest_run()
        self.assertEqual(self.flow._step("S1")["status"], "failed")
        self.assertFalse(self.flow._step("S1")["result"]["evidence_persisted"])

    def test_s6_verification_write_failure_preserves_unknown_physical_receipt(self):
        from ni_rua.robot.calibration.workflow.runner import write_json
        report = dict(self.report, status="unknown", stop_confirmation={"confirmed": False})

        def write(path, value):
            if path.name == "manifest.json":
                raise OSError("synthetic manifest write failure")
            return write_json(path, value)

        with patch.object(self.flow, "_touch", return_value=report), \
             patch("ni_rua.robot.calibration.workflow.runner.write_json", side_effect=write):
            result = self.flow._execute("S6", self.flow._s6, None)
        self.assertEqual(result["status"], "unknown")
        self.assertTrue(result["physics_advanced"])
        self.assertEqual(result["stop_confirmation"], {"confirmed": False})
        self.assertFalse(result["result"]["verification_recorded"])
        self.assertFalse(result["result"]["evidence_persisted"])
        self.assertNotIn("verification", self.flow.bundle.manifest(self.version))

    def test_s6_archive_failure_does_not_borrow_historical_pass_for_current_badge(self):
        from ni_rua.apps.gateway.extensions.calibration import calibration_badge
        from ni_rua.robot.calibration.workflow.runner import write_json
        manifest_path = self.flow.bundle.version_file(self.version, "manifest.json")
        historical = {"passed": True, "at": "2026-11-01T00:00:00Z", "max_error_m": 0.001}
        write_json(manifest_path, dict(self.flow.bundle.manifest(self.version), verification=historical))
        self.flow._step("S6").update(status="failed", finished_at="2026-11-02T00:00:00Z",
                                    result={"evidence_persisted": False, "archive_error": "disk full"})
        bundle = self.flow.bundle_status()
        self.assertTrue(bundle["matches_active_version"])
        self.assertFalse(bundle["verification"]["passed"])
        self.assertFalse(bundle["verification"]["evidence_persisted"])
        self.assertEqual(bundle["historical_verification"], historical)
        self.assertEqual(self.flow.bundle.manifest(self.version)["verification"], historical)
        self.assertNotEqual(calibration_badge({"bundle": bundle})["status"], "valid")

    def test_s5_requires_explicit_post_capture_review_and_records_binding(self):
        from ni_rua.robot.calibration.workflow.bundle import write_json
        report = {"joint_offsets_rad": self.flow._offsets(), "touches": [{"status": "contact"}]}
        self.flow._step("S1")["status"] = "passed"
        self.flow._step("S4")["status"] = "passed"
        write_json(self.flow._run_dir() / "S1_touches.json", report)
        refused = start(self.adapter, "S5", confirmed=[])
        self.assertEqual(refused["status"], "rejected")
        self.assertIn("touches_physically_confirmed", refused["reason"])
        self.assertNotIn("contact_review", self.flow.run)
        with patch.object(self.flow, "_s5", return_value={"status": "failed", "measures": {}}):
            result = start(self.adapter, "S5")
        self.assertEqual(result["status"], "failed")
        review = self.flow.run["contact_review"]
        self.assertEqual(review["kind"], "operator_attestation")
        self.assertEqual(review["acquisition_evidence"], "legacy_unverified")
        self.assertEqual(review["basis"], report["joint_offsets_rad"])

# === tests/test_so101_xy_verification.py ===

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ni_rua.robot.model.geometry import project
from ni_rua.robot.model.so101_gripper_geometry import fingertip_reference
from ni_rua.robot.calibration.workflow.xy_verification import evaluate_xy, validate_xy_request


class XYGeometryTests(unittest.TestCase):
    def fixture(self, delta=(0, 0, 0)):
        tip = np.asarray(fingertip_reference(0)["fixed_tip_m"])
        T = np.eye(4); T[:3, 3] = [0, 0, 1] - tip
        cameras = {}
        for name, x in (("fixed", 0), ("wrist", .2)):
            C = np.eye(4); C[0, 3] = x
            cameras[name] = dict(T=C, K=np.array([[500., 0, 320], [0, 500, 240], [0, 0, 1]]),
                                 D=np.zeros(5), width=640, height=480)
        snapshot = dict(snapshot_id="test", cameras=cameras, T_base_tcp=T)
        point = np.array([0, 0, 1]) + delta
        p = dict(action="xy_evaluate", snapshot_id="test", max_xy_error_m=.005,
                 max_reprojection_error_px=3, min_parallax_deg=1)
        for name, cam in cameras.items():
            p[name + "_pixel"] = project(cam, [point])[0][0].tolist()
        return snapshot, p

    def test_external_xy_offset_is_detected_without_changing_calibration(self):
        for delta, expected in (((0, 0, 0), True), ((.012, -.006, 0), False), ((0, 0, .02), True)):
            s, p = self.fixture(delta)
            before = deepcopy(s)
            r = evaluate_xy(s, p)
            self.assertEqual(r["passed"], expected)
            np.testing.assert_allclose(r["observed_minus_predicted_m"], delta, atol=1e-10)
            np.testing.assert_equal(s["T_base_tcp"], before["T_base_tcp"])
            self.assertIsNone(r["annotation_uncertainty_m"])

    def test_bad_correspondence_and_stale_snapshot_never_pass(self):
        s, p = self.fixture()
        p["wrist_pixel"][1] += 30
        self.assertIsNone(evaluate_xy(s, p)["passed"])
        p["snapshot_id"] = "stale"
        with self.assertRaisesRegex(ValueError, "snapshot_mismatch"):
            evaluate_xy(s, p)
        for change in ({"max_xy_error_m": float("nan")}, {"max_xy_error_m": True}, {"fixed_pixel": [1]}, {"unexpected": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_xy_request({**p, **change})

    def test_defaults_are_worker_supplied_and_explicit_thresholds_win(self):
        snapshot, params = self.fixture()
        thresholds = {key: params.pop(key) for key in ("max_xy_error_m", "max_reprojection_error_px", "min_parallax_deg")}
        self.assertEqual(validate_xy_request(params), params)  # gateway does not invent defaults
        with self.assertRaisesRegex(ValueError, "thresholds_unresolved"):
            evaluate_xy(snapshot, params)
        thresholds.update(max_xy_error_m=.004, max_reprojection_error_px=2.5, min_parallax_deg=1.2)
        resolved = validate_xy_request(dict(params, max_xy_error_m=.003), defaults=thresholds)
        self.assertEqual(resolved["max_xy_error_m"], .003)
        self.assertEqual(resolved["max_reprojection_error_px"], 2.5)
        self.assertEqual(resolved["min_parallax_deg"], 1.2)


@unittest.skipUnless(HAVE_ARUCO, "OpenCV aruco required")
class XYCaptureTests(unittest.TestCase):
    def test_capture_is_read_only_token_bound_and_rejects_changed_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, arm, _, deployment, camera = make_workflow_adapter(tmp)
            a.operator("calib_workflow_begin", mode="first_run")
            goals = len(arm.goals)
            before = (deployment.read_bytes(), camera.read_bytes())
            r = a.operator("calib_step_start", step="S6", params={"action": "xy_capture"})
            self.assertEqual(r["status"], "captured", r)
            self.assertFalse(r["physics_advanced"])
            self.assertEqual(len(arm.goals), goals)
            identifier = r["snapshot"]["snapshot_id"]
            self.assertTrue((a.telemetry_dir / f"xy-{identifier}-fixed.jpg").is_file())
            p = dict(action="xy_evaluate", snapshot_id="wrong", fixed_pixel=[320, 240], wrist_pixel=[320, 240],
                     max_xy_error_m=.005, max_reprojection_error_px=3, min_parallax_deg=1)
            r = a.operator("calib_step_start", step="S6", params=p)
            self.assertEqual(r["status"], "rejected")
            p["snapshot_id"] = identifier
            a.workflow.settings.update(xy_max_error_m=.004, xy_max_reprojection_error_px=2.5, xy_min_parallax_deg=1.2)
            without_thresholds = {k: v for k, v in p.items() if k not in (
                "max_xy_error_m", "max_reprojection_error_px", "min_parallax_deg")}
            r = a.operator("calib_step_start", step="S6", params=without_thresholds)
            self.assertEqual(r["status"], "evaluated", r)
            self.assertEqual(r["result"]["parameters"]["max_xy_error_m"], .004)
            self.assertEqual(r["result"]["parameters"]["max_reprojection_error_px"], 2.5)
            self.assertEqual(r["result"]["parameters"]["min_parallax_deg"], 1.2)
            self.assertIn(r["result"]["status"], ("passed", "failed", "unknown"))
            state = a.operator("calib_workflow_status")["run"]
            self.assertEqual(len(state["xy_verifications"]), 1)
            self.assertEqual(state["steps"]["S6"]["status"], "not_started")
            a.camera_calibration["cameras"]["fixed"]["K"][0][0] += 1
            r = a.operator("calib_step_start", step="S6", params=p)
            self.assertEqual(r["status"], "rejected", r)
            self.assertIn("calibration_changed", r["reason"])
            self.assertEqual(before, (deployment.read_bytes(), camera.read_bytes()))
            self.assertEqual(len(arm.goals), goals)

    def test_workflow_execute_motion_crash_preserves_unknown_and_xy_catches_driver_timeout_faults(self):
        from unittest.mock import MagicMock, patch
        from ni_rua.robot.backends.so101.worker import DriverFault
        from ni_rua.robot.calibration.workflow import bundle as wf_bundle
        from ni_rua.robot.calibration.workflow import hand_eye_solve as wf_he

        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        a, _arm, _cams, _deployment, _camera = make_workflow_adapter(str(tmp))
        a.operator("calib_workflow_begin", mode="first_run")

        # DEF-10 (#1): unexpected crash during motion step after control_step advanced calls _stop() and reports status='unknown', physics_advanced=True.
        stop_spy = MagicMock(wraps=a._stop)

        def move_then_crash(_progress):
            a.control_step += 2
            raise RuntimeError("post-touch crash")

        with patch.object(a, "_stop", stop_spy):
            rec = a.workflow._execute("S1", move_then_crash, None)
        self.assertEqual(stop_spy.call_count, 1)
        self.assertEqual(rec["status"], "unknown")
        self.assertTrue(rec["physics_advanced"])
        self.assertIsNone(a.active_action_id)
        self.assertIsNone(a.workflow.running_step)

        # DEF-10 (#2): xy_capture sets active_action_id while running and catches DriverFault / TimeoutError cleanly as status='rejected'.
        seen_active = []

        def check_active_then_fault(*_args, **_kwargs):
            seen_active.append(a.active_action_id)
            raise DriverFault("serial read fault during xy_capture")

        with patch.object(a, "_settle", side_effect=check_active_then_fault):
            rej = a.operator("calib_step_start", step="S6", params={"action": "xy_capture"})
        self.assertEqual(seen_active, ["operator_calibration_S6_xy"])
        self.assertIsNone(a.active_action_id)
        self.assertEqual(rej["status"], "rejected")
        self.assertIn("serial read fault", rej["reason"])

        with patch.object(a.cameras["fixed"], "read_after", side_effect=TimeoutError("so101_camera_frame_timeout:fixed")):
            rej_t = a.operator("calib_step_start", step="S6", params={"action": "xy_capture"})
        self.assertEqual(rej_t["status"], "rejected")
        self.assertIsNone(a.active_action_id)

        # DEF-10 (#3): _install attaches offset_rollback_failed note instead of masking install_camera_file error.
        ver = a.workflow.bundle.import_existing({n: 0.0 for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")})
        with patch("ni_rua.robot.calibration.workflow.runner.install_camera_file", side_effect=OSError("camera install failed")), \
             patch("ni_rua.robot.calibration.workflow.runner.write_joint_offsets", side_effect=[{"ok": True}, OSError("rollback disk full")]):
            with self.assertRaisesRegex(OSError, "camera install failed") as cm_inst:
                a.workflow._install(
                    ver,
                    {n: 0.01 for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")},
                    {n: 0.0 for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")},
                )
        self.assertTrue(any("offset_rollback_failed" in n for n in getattr(cm_inst.exception, "__notes__", ())))

        # DEF-16 (#4, #5): atomic write helpers unlink temporary files on replace failure.
        bundle_json = tmp / "bundle_test.json"
        with patch("ni_rua.robot.calibration.workflow.bundle.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                wf_bundle.write_json(bundle_json, {"a": 1})
        self.assertEqual(list(tmp.glob(".bundle_test.json.*.tmp")), [])

        he_json = tmp / "he_test.json"
        with patch.object(Path, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                wf_he.write_json(he_json, {"b": 2})
        self.assertFalse((tmp / "he_test.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
