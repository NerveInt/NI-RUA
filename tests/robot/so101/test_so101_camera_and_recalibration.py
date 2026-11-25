"""One-click SO-101 camera recalibration and live calibration reload (CPU only, fake driver/cameras).

Proves the software contract: reload validation (good file swaps, bad file keeps the old one),
the no-motion board check, capture through ``move_joints`` (speed cap, park, stop on hold), and
the install orchestration (gate failure keeps the active file; pass replaces it atomically and
reloads). Solve quality and real board detection on hardware are not covered here.
"""

NI_RUA_TEST_TIER = "fast"

# === tests/test_so101_recalibration.py ===

import json
from pathlib import Path
from support.paths import CONFIGS_ROOT, REPO_ROOT
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from ni_rua.robot.calibration.workflow import capture
from ni_rua.robot.backends.so101.driver import FakeCameraSource
from robot.so101.test_so101_backend import EXAMPLE, deployment, make_adapter

try:
    import cv2
    HAVE_ARUCO = hasattr(cv2, "aruco") and hasattr(cv2.aruco, "CharucoDetector")
except ImportError:  # pragma: no cover
    HAVE_ARUCO = False

if HAVE_ARUCO:
    from ni_rua.robot.calibration.workflow import charuco as cal


def write_calibration(path, size, *, created_at="2026-11-01T00:00:00Z", translation_m=0.003, overwrite=True):
    K = [[500., 0, size[0] / 2], [0, 500., size[1] / 2], [0, 0, 1]]
    fixed = np.eye(4)
    fixed[:3, 3] = [0.5, 0, 0.4]
    cal.write_calibration_file(path, board=cal.DEFAULT_BOARD_SPEC, samples=19, created_at=created_at, overwrite=overwrite,
                               cameras={"wrist": dict(K=K, distortion=[0] * 5, image_size=list(size), T_link_camera=np.eye(4).tolist(),
                                                      intrinsics_rms_px=.24, hand_eye_residual=dict(translation_m=translation_m, rotation_rad=.014)),
                                        "fixed": dict(K=K, distortion=[0] * 5, image_size=list(size), T_base_camera=fixed.tolist(),
                                                      intrinsics_rms_px=.24, extrinsic_residual=dict(reprojection_px=4.0))})


def board_image(width=640, height=480):
    """Frontal RGB view of the default ChArUco board (all 24 inner corners visible)."""
    gray = cal.make_board(cal.DEFAULT_BOARD_SPEC).generateImage((width, height), marginSize=40, borderBits=1)
    return np.repeat(gray[..., None], 3, axis=2)


class BoardCamera(FakeCameraSource):
    """Fake camera whose frames show ``image`` (or blank frames when ``image`` is None)."""

    def __init__(self, camera_id, clock, image, size=(640, 480)):
        super().__init__(camera_id, width=image.shape[1] if image is not None else size[0],
                         height=image.shape[0] if image is not None else size[1], clock=clock)
        self.image = image

    def _frame(self, stamp):
        self.count += 1
        self.last = (self.image.copy() if self.image is not None else np.zeros((self.height, self.width, 3), np.uint8), stamp)
        return self.last


def adapter_with(directory, *, visible=("wrist", "fixed"), profile="rgb_proprio_v1", calibration_file=None, size=(640, 480), **options):
    deploy = deployment()
    for name in ("wrist", "fixed"):
        deploy["cameras"][name].update(width=size[0], height=size[1])
    deploy["camera_calibration_file"] = None if calibration_file is None else str(calibration_file)
    adapter, driver, clock = make_adapter(directory, profile=profile, deploy=deploy, **options)
    image = board_image(*size) if size[0] >= 320 else None  # tiny reload fixtures: blank frames
    adapter.cameras = {name: BoardCamera(name, clock, image if name in visible else None, size) for name in ("wrist", "fixed")}
    return adapter, driver, clock


@unittest.skipUnless(HAVE_ARUCO, "OpenCV with aruco CharucoDetector required")
class ReloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.directory, True)
        self.path = self.directory / "camera_calibration.json"

    def test_good_file_swaps_published_calibration(self):
        size = (64, 48)
        write_calibration(self.path, size, created_at="2026-11-01T00:00:00Z")
        adapter, _, _ = adapter_with(self.directory, profile="rgb_proprio_calibrated_v1", calibration_file=self.path, size=size)
        self.assertEqual(adapter.observe()["camera_calibration"]["calibrated_at"], "2026-11-01T00:00:00Z")
        write_calibration(self.path, size, created_at="2026-11-08T09:00:00Z", translation_m=0.0021)
        result = adapter.operator("reload_calibration")
        self.assertEqual(result["status"], "reloaded")
        self.assertEqual(result["previous"]["created_at"], "2026-11-01T00:00:00Z")
        self.assertEqual(result["calibration"]["residuals"]["wrist_hand_eye_translation_rms_m"], 0.0021)
        published = adapter.observe()["camera_calibration"]
        self.assertEqual(published["calibrated_at"], "2026-11-08T09:00:00Z")
        # Precision is model-relevant; no hash, path or provenance enters the published document.
        self.assertEqual(published["calibration_residuals"]["fixed_chain_reprojection_rms_px"], 4.0)
        self.assertNotIn(str(self.path), json.dumps(published))
        self.assertFalse({"file", "sha256", "digest"} & set(published))

    def test_bad_file_keeps_previous_calibration(self):
        size = (64, 48)
        write_calibration(self.path, size)
        adapter, _, _ = adapter_with(self.directory, profile="rgb_proprio_calibrated_v1", calibration_file=self.path, size=size)
        before = adapter.camera_calibration
        for label, damage, reason in (
                ("image_size", lambda: write_calibration(self.path, (640, 480), created_at="2026-11-09T00:00:00Z"), "image_size"),
                ("json", lambda: self.path.write_text("{not json"), "JSONDecodeError"),
                ("schema", lambda: self.path.write_text(json.dumps({"schema": "other"})), "ValueError"),
                ("missing", lambda: self.path.unlink(), "FileNotFoundError")):
            with self.subTest(label):
                damage()
                result = adapter.operator("reload_calibration")
                self.assertEqual((result["status"], result["kept_previous"]), ("failed", True), result)
                self.assertIn(reason, result["reason"])
                self.assertEqual(result["calibration"]["created_at"], "2026-11-01T00:00:00Z")
                self.assertIs(adapter.camera_calibration, before)
                self.assertEqual(adapter.observe()["camera_calibration"]["calibrated_at"], "2026-11-01T00:00:00Z")

    def test_uncalibrated_profile_reports_identity_but_never_publishes(self):
        write_calibration(self.path, (640, 480))
        adapter, _, _ = adapter_with(self.directory, calibration_file=self.path)
        self.assertEqual(adapter.calibration_identity["created_at"], "2026-11-01T00:00:00Z")
        result = adapter.operator("reload_calibration")
        self.assertEqual((result["status"], result["published_in_observations"]), ("reloaded", False))
        self.assertNotIn("camera_calibration", adapter.observe())

    def test_reload_refused_while_action_active(self):
        write_calibration(self.path, (640, 480))
        adapter, _, _ = adapter_with(self.directory, calibration_file=self.path)
        adapter.active_action_id = "act-1"
        self.assertEqual(adapter.operator("reload_calibration")["reason"], "action_active")
        self.assertEqual(adapter.operator("calibrate")["reason"], "action_active")


@unittest.skipUnless(HAVE_ARUCO, "OpenCV with aruco CharucoDetector required")
class BoardCheckAndCalibrateTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.directory, True)
        self.path = self.directory / "deploy" / "camera_calibration.json"
        self.path.parent.mkdir()
        write_calibration(self.path, (640, 480), created_at="2026-11-01T00:00:00Z")
        self.original = self.path.read_bytes()

    def test_board_check_counts_corners_on_latest_frames_without_bus(self):
        adapter, driver, _ = adapter_with(self.directory, visible=("wrist",), calibration_file=self.path)
        self.assertEqual(adapter.operator("board_check")["corners"], {"fixed": 0, "wrist": 0})  # no frame yet
        for camera in adapter.cameras.values():
            camera.read_after(0, 1)
        sends = driver.sends
        with patch.object(driver, "read", side_effect=AssertionError("board_check must not touch the bus")):
            result = adapter.operator("board_check")
        self.assertEqual(result["corners"], {"fixed": 0, "wrist": 24})
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "board_not_visible:fixed")
        self.assertEqual(driver.sends, sends)

    def test_board_not_visible_refuses_motion(self):
        adapter, driver, _ = adapter_with(self.directory, visible=("wrist",), calibration_file=self.path)
        sends, step = driver.sends, adapter.control_step
        result = adapter.operator("calibrate")
        self.assertEqual(result["status"], "board_not_visible")
        self.assertEqual(result["corners"], {"fixed": 0, "wrist": 24})
        self.assertEqual(result["min_board_corners"], adapter.deployment["calibration"]["min_board_corners"])
        self.assertFalse(result["physics_advanced"])
        self.assertEqual((driver.sends, adapter.control_step), (sends, step))
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_calibrate_refused_while_relaxed(self):
        adapter, driver, _ = adapter_with(self.directory, calibration_file=self.path)
        adapter.operator("relax")
        sends = driver.sends
        self.assertEqual(adapter.operator("calibrate")["reason"], "operator_relaxed_requires_torque_on")
        self.assertEqual(driver.sends, sends)

    def fake_capture(self, kept=10, status="completed"):
        def run(adapter, session_dir, *, settings, progress):
            Path(session_dir).mkdir(parents=True)
            progress({"phase": "sample", "pose_index": 1, "pose_count": 1, "kept_count": kept})
            return {"status": status, "reason": None if status == "completed" else "operator_hold", "kept": kept,
                    "pose_count": 13, "poses": [], "park": None, "physics_advanced": True}
        return run

    def test_gate_failure_keeps_active_file(self):
        adapter, _, _ = adapter_with(self.directory, calibration_file=self.path)

        def solve(adapter, session_dir, output, settings):
            (Path(session_dir) / "solve_report.json").write_text(json.dumps({
                "hand_eye": {"translation_rms_m": 0.02, "rotation_rms_rad": 0.05, "samples": 10},
                "gate_failures": ["wrist.hand_eye_residual.translation_m=0.02 > max_hand_eye_translation_m=0.005"],
                "thresholds": {"max_hand_eye_translation_m": 0.005}}))
            raise cal.CalibrationGateError(["wrist.hand_eye_residual.translation_m=0.02 > max_hand_eye_translation_m=0.005"])
        progress = []
        result = capture.run_calibration(adapter, progress=progress.append, capture=self.fake_capture(), solve=solve)
        self.assertEqual(result["status"], "gate_failed")
        self.assertEqual(result["residuals"]["hand_eye_translation_rms_m"], 0.02)
        self.assertTrue(result["gate_failures"])
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(adapter.calibration_identity["created_at"], "2026-11-01T00:00:00Z")
        self.assertEqual([update["phase"] for update in progress], ["board_check", "sample", "solving"])

    def test_pass_replaces_atomically_and_reloads(self):
        adapter, _, _ = adapter_with(self.directory, calibration_file=self.path)
        replaced = []

        def solve(adapter, session_dir, output, settings):
            self.assertEqual(Path(output).parent, self.path.parent)
            self.assertNotEqual(Path(output), self.path)
            write_calibration(output, (640, 480), created_at="2026-11-08T10:00:00Z", translation_m=0.0025, overwrite=False)
            return {"hand_eye": {"translation_rms_m": 0.0025, "rotation_rms_rad": 0.01, "samples": 12},
                    "cameras": {"wrist": {"intrinsics_rms_px": 0.3}}, "fixed": {"chain_reprojection_px": 3.1},
                    "gate_failures": [], "thresholds": {}}
        real_replace = capture.os.replace
        with patch.object(capture.os, "replace", side_effect=lambda a, b: (replaced.append((Path(a), Path(b))), real_replace(a, b))):
            result = capture.run_calibration(adapter, capture=self.fake_capture(), solve=solve)
        self.assertEqual(result["status"], "calibrated", result)
        self.assertEqual(replaced[-1][1], self.path)  # one rename onto the active path
        self.assertEqual(json.loads(self.path.read_text())["created_at"], "2026-11-08T10:00:00Z")
        self.assertEqual(result["calibration"]["created_at"], "2026-11-08T10:00:00Z")
        self.assertEqual(adapter.calibration_identity["created_at"], "2026-11-08T10:00:00Z")
        self.assertEqual(Path(result["previous_file_backup"]).read_bytes(), self.original)
        self.assertTrue(Path(result["candidate_file"]).is_file())
        self.assertEqual(result["residuals"]["fixed_chain_reprojection_px"], 3.1)
        self.assertFalse(list(self.path.parent.glob(".*.tmp")))

    def test_install_keeps_the_touch_off_record_of_the_active_file(self):
        document = json.loads(self.path.read_text())
        document["joint_calibration"] = {"table_z_m": -0.02, "joint_offsets_rad": {"elbow_flex": -0.063}}
        self.path.write_text(json.dumps(document))
        adapter, _, _ = adapter_with(self.directory, calibration_file=self.path)

        def solve(adapter, session_dir, output, settings):
            write_calibration(output, (640, 480), created_at="2026-11-08T10:00:00Z", overwrite=False)
            return {"gate_failures": [], "thresholds": {}}
        result = capture.run_calibration(adapter, capture=self.fake_capture(), solve=solve)
        self.assertEqual(result["status"], "calibrated", result)
        self.assertEqual(json.loads(self.path.read_text())["joint_calibration"]["table_z_m"], -0.02)

    def test_candidate_that_would_not_load_is_never_installed(self):
        adapter, _, _ = adapter_with(self.directory, calibration_file=self.path)

        def solve(adapter, session_dir, output, settings):
            write_calibration(output, (320, 240), overwrite=False)  # wrong capture size
            return {"gate_failures": []}
        result = capture.run_calibration(adapter, capture=self.fake_capture(), solve=solve)
        self.assertEqual(result["status"], "install_failed")
        self.assertIn("image_size", result["reason"])
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_insufficient_samples_and_stopped_capture_never_solve(self):
        adapter, _, _ = adapter_with(self.directory, calibration_file=self.path)
        never = lambda *a: self.fail("solve must not run")  # noqa: E731
        self.assertEqual(capture.run_calibration(adapter, capture=self.fake_capture(kept=2), solve=never)["status"],
                         "insufficient_samples")
        self.assertEqual(capture.run_calibration(adapter, capture=self.fake_capture(status="stopped"), solve=never)["status"],
                         "stopped")
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_run_calibration_preserves_handheld_fixed_intrinsics_and_syncs_bundle(self):
        from ni_rua.robot.calibration.workflow import hand_eye_solve as solve_mod
        doc = json.loads(self.path.read_text())
        doc["cameras"]["fixed"]["intrinsics_source"] = "handheld"
        doc["cameras"]["fixed"]["K"] = [[712.0, 0.0, 325.0], [0.0, 714.0, 242.0], [0.0, 0.0, 1.0]]
        doc["cameras"]["fixed"]["intrinsics_rms_px"] = 0.19
        self.path.write_text(json.dumps(doc))
        adapter, _, _ = adapter_with(self.directory, calibration_file=self.path)
        captured_kwargs = {}

        def fake_solve_session(session_dir, **kwargs):
            captured_kwargs.update(kwargs)
            write_calibration(kwargs["output"], (640, 480), created_at="2026-11-08T12:00:00Z", overwrite=False)
            out_doc = json.loads(Path(kwargs["output"]).read_text())
            out_doc["cameras"]["fixed"]["K"] = np.asarray(kwargs["fixed_intrinsics"]).tolist()
            out_doc["cameras"]["fixed"]["intrinsics_source"] = "handheld"
            Path(kwargs["output"]).write_text(json.dumps(out_doc))
            return {"hand_eye": {"translation_rms_m": 0.002, "rotation_rms_rad": 0.01, "samples": 10},
                    "cameras": {"wrist": {"intrinsics_rms_px": 0.25},
                                "fixed": {"intrinsics_rms_px": kwargs.get("fixed_intrinsics_rms_px"),
                                          "intrinsics_source": "handheld"}},
                    "fixed": {"chain_reprojection_px": 2.5}, "gate_failures": [], "thresholds": {}}

        with patch.object(solve_mod, "solve_session", side_effect=fake_solve_session):
            result = capture.run_calibration(adapter, capture=self.fake_capture())
        self.assertEqual(result["status"], "calibrated", result)
        self.assertIsNotNone(captured_kwargs.get("fixed_intrinsics"))
        np.testing.assert_allclose(captured_kwargs["fixed_intrinsics"][0, 0], 712.0)
        installed = json.loads(self.path.read_text())
        self.assertEqual(installed["cameras"]["fixed"]["intrinsics_source"], "handheld")
        self.assertEqual(installed["cameras"]["fixed"]["K"][0][0], 712.0)
        bundle_status = adapter.workflow.bundle_status()
        self.assertTrue(bundle_status["matches_active_version"])
        manifest = adapter.workflow.bundle.manifest(bundle_status["active_version"])
        self.assertEqual(manifest["joint_mapping"],
                         {name: dict(adapter.deployment["joint_mapping"][name]) for name in capture.ARM_JOINTS})
        # Importing the older camera file must not manufacture historical sign evidence.
        imported = adapter.workflow.bundle.manifest(manifest["previous"])
        self.assertNotIn("joint_mapping", imported)
        self.assertFalse(bundle_status["has_joint_calibration"])
        adapter.workflow.begin("first_run")
        adapter.workflow.run["applied_version"] = bundle_status["active_version"]
        self.assertEqual(adapter.workflow._auto_check("applied", False),
                         (False, "verification_calibration_mismatch:table_height_not_calibrated"))
        adapter.deployment["joint_mapping"]["wrist_flex"]["sign"] *= -1
        self.assertFalse(adapter.workflow.bundle_status()["matches_active_version"])


@unittest.skipUnless(HAVE_ARUCO, "OpenCV with aruco CharucoDetector required")
class CaptureMotionTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.directory, True)

    def small_grid(self, adapter):
        settings = dict(adapter.deployment["calibration"], pan_offsets_rad=[-0.1, 0.1], flex_offsets_rad=[0.0],
                        roll_offsets_rad=[0.2], lift_offsets_rad=[0.05], speed_scale=0.5)
        adapter.deployment["calibration"] = settings
        return settings

    def test_capture_visits_grid_under_speed_cap_and_parks(self):
        adapter, driver, _ = adapter_with(self.directory)
        settings = self.small_grid(adapter)
        adapter.operator("set_runtime_limits", speed_scale_cap=0.5)
        start = len(driver.goals)
        progress = []
        result = capture.capture_session(adapter, self.directory / "session", settings=settings, progress=progress.append)
        self.assertEqual((result["status"], result["kept"], result["pose_count"]), ("completed", 4, 4))
        self.assertEqual(result["park"]["status"], "completed")
        np.testing.assert_allclose(driver.q, adapter.deployment["home"]["q_rad"], atol=settings["arrival_tolerance_rad"])
        goals = np.asarray([goal for goal, _ in driver.goals[start:]])
        motion = adapter.deployment["motion"]
        # Interpolation steps never exceed the capped speed; sag-correction jumps are bounded separately.
        steps = np.abs(np.diff(goals, axis=0)).max(axis=1)
        limit = motion["max_joint_speed_rad_s"] * 0.5 * 0.5 / motion["command_rate_hz"] + 1e-9
        self.assertTrue((steps <= max(limit, motion["arrival_correction_max_offset_rad"])).all())
        self.assertGreater((steps <= limit).mean(), 0.9)
        from ni_rua.robot.calibration.workflow.hand_eye_solve import load_session
        session, samples = load_session(self.directory / "session")
        self.assertEqual(len(samples), 4)
        self.assertEqual(set(samples[0]["images"]), {"wrist", "fixed"})
        self.assertTrue((self.directory / "session" / samples[0]["images"]["wrist"]).is_file())
        self.assertEqual(samples[0]["detected_corners_at_capture"], {"fixed": 24, "wrist": 24})
        self.assertIn("pose_index", progress[0])

    def test_hold_mid_capture_stops_without_park(self):
        adapter, _, _ = adapter_with(self.directory)
        settings = self.small_grid(adapter)
        held = {}

        def progress(update):
            if update.get("phase") == "sample" and not held:
                held.update(adapter.operator("hold"))
        result = capture.capture_session(adapter, self.directory / "session", settings=settings, progress=progress)
        self.assertEqual(held["status"], "held")
        self.assertEqual((result["status"], result["reason"]), ("stopped", "operator_hold"))
        self.assertIsNone(result["park"])
        self.assertEqual(result["kept"], 1)

    def test_real_solve_on_degenerate_views_keeps_old_calibration(self):
        """Identical frames at every pose cannot pass the hand-eye gates: the active file stays."""
        path = self.directory / "camera_calibration.json"
        write_calibration(path, (640, 480))
        original = path.read_bytes()
        adapter, _, _ = adapter_with(self.directory, calibration_file=path)
        self.small_grid(adapter)
        adapter.deployment["calibration"]["min_samples"] = 3
        result = adapter.operator("calibrate")
        self.assertIn(result["status"], ("gate_failed", "solve_failed"), result)
        self.assertTrue(result["physics_advanced"])
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(adapter.calibration_identity["created_at"], "2026-11-01T00:00:00Z")

    def test_unsettled_or_drifting_frames_and_floor_violations_rejected(self):
        adapter, driver, _ = adapter_with(self.directory)
        settings = dict(self.small_grid(adapter), z_floor_m=0.50)
        # High z_floor_m skips every pose that goes below 0.50 m
        floor_res = capture.capture_session(adapter, self.directory / "floor_session", settings=settings)
        self.assertEqual(floor_res["kept"], 0)
        self.assertTrue(all(p["reason"] == "below_z_floor" for p in floor_res["poses"]))

        # Unsettled frame is rejected with reason="arm_not_settled"
        settings = self.small_grid(adapter)
        original_settle = adapter._settle
        def unsettled_capture(run, timeout, **kwargs):
            if run["action_id"] == "calibration_settle":
                return False, adapter.read(), "settle_timeout"
            return original_settle(run, timeout, **kwargs)
        with patch.object(adapter, "_settle", side_effect=unsettled_capture):
            unsettled_res = capture.capture_session(adapter, self.directory / "unsettled_session",
                                                    settings=settings, poses=[adapter.deployment["home"]["q_rad"]])
        self.assertEqual(unsettled_res["kept"], 0)
        self.assertEqual(unsettled_res["poses"][0]["reason"], "arm_not_settled")

        # Joint drift between before and after capture is rejected with reason="joint_drift_during_capture"
        orig_capture = adapter.capture_frames

        def drift_during_capture(t):
            frames = orig_capture(t)
            driver.q[0] += 0.25
            return frames

        with patch.object(adapter, "capture_frames", side_effect=drift_during_capture):
            drift_res = capture.capture_session(adapter, self.directory / "drift_session",
                                                settings=settings, poses=[adapter.deployment["home"]["q_rad"]])
        self.assertEqual(drift_res["kept"], 0)
        self.assertEqual(drift_res["poses"][0]["reason"], "joint_drift_during_capture")

        # fixed_extrinsic_from_session skips unsettled samples
        sess = self.directory / "ext_sess"
        (sess / "images").mkdir(parents=True)
        (sess / "session.json").write_text(json.dumps({"schema": capture.SESSION_SCHEMA, "board": cal.DEFAULT_BOARD_SPEC}))
        img = board_image(640, 480)
        cv2.imwrite(str(sess / "images" / "0000_fixed.png"), img)
        cv2.imwrite(str(sess / "images" / "0000_wrist.png"), img)
        (sess / "samples.jsonl").write_text(json.dumps({
            "index": 0, "joints_deg": {n: 0.0 for n in cal.ARM_JOINTS}, "settled": False,
            "images": {"fixed": "images/0000_fixed.png", "wrist": "images/0000_wrist.png"}}) + "\n")
        K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
        with self.assertRaisesRegex(LookupError, "session_has_no_fixed_board_views"):
            capture.fixed_extrinsic_from_session(sess, {"K": K, "distortion": np.zeros(5), "T_link_camera": np.eye(4)},
                                                 K, np.zeros(5), cal.make_board(), lambda q: np.eye(4))


class BackendRoutingTests(unittest.TestCase):
    def test_calibrate_is_serialized_with_long_timeout_and_board_check_preempts(self):
        from ni_rua.core.config import Config
        from ni_rua.robot.backends.factory import create_backend
        config = Config.from_dict({"backend": {"environment": "so101_real", "robot_model_id": "so101", "deployment": str(EXAMPLE)},
                                   "features": {"observation_profile": "rgb_proprio_v1"}})
        with tempfile.TemporaryDirectory() as directory:
            backend = create_backend(config=config, run_dir=directory)
            calls = []

            def request(operation, payload=None, timeout=180, progress=None, cancel=None):
                calls.append((payload["command"], timeout, backend.operation_lock.locked(), progress))
                return {"status": "ok"}
            backend._request = request
            sink = []
            backend.operator("calibrate", progress_callback=sink.append)
            backend.operator("board_check")
            backend.operator("reload_calibration")
            self.assertEqual(calls[0][0], "calibrate")
            self.assertTrue(calls[0][2])
            self.assertIs(calls[0][3].__self__, sink)
            self.assertGreaterEqual(calls[0][1], backend.deployment["calibration"]["capture_timeout_s"]
                                    + backend.deployment["calibration"]["solve_timeout_s"])
            self.assertEqual(calls[1][:3], ("board_check", backend.deployment["ipc"]["operator_timeout_s"], False))
            self.assertTrue(calls[2][2])
            calls.clear()
            for command in ("calib_workflow_status", "calib_step_status", "calib_step_cancel"):
                backend.operator(command)
            backend.operator("calib_step_start", step="S1", params={})
            backend.operator("calib_apply")
            self.assertEqual([call[2] for call in calls], [False, False, False, True, True])  # status/cancel preempt
            self.assertEqual(calls[3][1], backend.calibration_step_timeout_s())
            from ni_rua.robot.backends.so101 import backend as parent, worker
            self.assertEqual(set(parent.PREEMPTIVE_OPERATOR_COMMANDS), set(worker.PREEMPTIVE_OPERATOR_COMMANDS))


class DeploymentCalibrationSectionTests(unittest.TestCase):
    def test_calibration_section_is_required_and_bounded(self):
        from ni_rua.robot.backends.so101.deployment import validate_deployment
        document = json.loads(EXAMPLE.read_text())
        for change, message in ((lambda d: d.pop("calibration"), "calibration: missing="),
                                (lambda d: d["calibration"].update(pan_offsets_rad=[]), "pan_offsets_rad"),
                                (lambda d: d["calibration"].update(roll_offsets_rad=[3.0]), "roll_offsets_rad"),
                                (lambda d: d["calibration"].update(min_board_corners=2), "out_of_range")):
            candidate = json.loads(json.dumps(document))
            change(candidate)
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                validate_deployment(candidate, base=EXAMPLE.parent)

    def test_pose_grid_matches_legacy_script(self):
        poses = capture.pose_grid([0, -.95, .9, 1, 0], [-.2, .2], [0], [-.35, .35], [.12])
        self.assertEqual(len(poses), 5)
        np.testing.assert_allclose(poses[-1], [0, -.83, .9 - .072, 1, 0])

    def test_camera_rotate_deg_device_name_and_frame_timestamp(self):
        from ni_rua.robot.backends.so101.deployment import validate_deployment
        from ni_rua.robot.backends.so101.camera_devices import CameraBindingError, list_video_devices, resolve_camera
        from ni_rua.robot.backends.so101.driver import FakeClock, OpenCVCameraSource, rotate_frame
        document = json.loads(EXAMPLE.read_text())
        document["cameras"]["wrist"]["rotate_deg"] = 180
        document["cameras"]["wrist"]["device_name"] = "Wrist UVC"
        validated = validate_deployment(document, base=EXAMPLE.parent)
        self.assertEqual(validated["cameras"]["wrist"]["rotate_deg"], 180)
        self.assertEqual(validated["cameras"]["wrist"]["device_name"], "Wrist UVC")
        self.assertEqual(validated["cameras"]["fixed"]["rotate_deg"], 0)
        self.assertIsNone(validated["cameras"]["fixed"]["device_name"])
        for bad_rot in (45, -90, "90"):
            bad = json.loads(EXAMPLE.read_text())
            bad["cameras"]["wrist"]["rotate_deg"] = bad_rot
            with self.assertRaisesRegex(ValueError, "rotate_deg"):
                validate_deployment(bad, base=EXAMPLE.parent)
        frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        self.assertEqual(rotate_frame(frame, 0).shape, (2, 3, 3))
        self.assertEqual(rotate_frame(frame, 90).shape, (3, 2, 3))
        self.assertEqual(rotate_frame(frame, 180).shape, (2, 3, 3))
        self.assertEqual(rotate_frame(frame, 270).shape, (3, 2, 3))
        np.testing.assert_array_equal(rotate_frame(rotate_frame(frame, 90), 270), frame)
        with tempfile.TemporaryDirectory() as v4l:
            v0 = Path(v4l) / "video2"
            v0.mkdir()
            (v0 / "name").write_text("USB Wrist Camera\n")
            inventory = list_video_devices(platform="linux", v4l_root=v4l, v4l_links=Path(v4l) / "none",
                                           opencv_version="4.13.0")
            self.assertEqual(resolve_camera("wrist", {"index_or_path": 0, "device_name": "Wrist"}, inventory)["opened"],
                             "/dev/video2")
            # A name that matches nothing refuses; it no longer falls back silently to index_or_path.
            with self.assertRaisesRegex(CameraBindingError, "^so101_camera_not_found:wrist:device_name=Missing"):
                resolve_camera("wrist", {"index_or_path": 0, "device_name": "Missing"}, inventory)
        # Verify OpenCVCameraSource stamps frame with read_started (before read() completes) and rotates before size check
        clock = FakeClock(start=100.0)
        src = OpenCVCameraSource("wrist", {"index_or_path": 0, "width": 2, "height": 3, "fps": 30, "rotate_deg": 90}, clock)

        class FakeCap:
            def read(self_cap):
                clock.sleep(0.05)  # exposure/transfer takes 50 ms; stamp must be 100.0 (start), not 100.05
                src.running = False
                return True, np.zeros((2, 3, 3), dtype=np.uint8)
        src.capture, src.running = FakeCap(), True
        src._grab()
        self.assertIsNone(src.error)
        self.assertEqual(src.frame.shape, (3, 2, 3))
        self.assertAlmostEqual(src.stamp, 100.0)

# === tests/test_so101_camera_devices.py ===

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2

from ni_rua.core.config import Config
from ni_rua.robot.backends.so101 import camera_devices
from ni_rua.robot.backends.so101.camera_devices import (
    CameraBindingError, bind_cameras, binding_summary, confirm_unchanged, list_video_devices, log_lines, resolve_camera)
from ni_rua.robot.backends.so101.deployment import validate_deployment
from ni_rua.robot.backends.so101.driver import FakeArmDriver, FakeCameraSource, FakeClock, OpenCVCameraSource
from ni_rua.robot.backends.so101.worker import SO101Adapter


# AVFoundation devices as macOS reports them (unsorted; the order system_profiler also prints).
FACETIME = {"unique_id": "1FD4B3A2-236E-492B-8CE5-255DD288CE50", "name": "FaceTime高清相机", "model": "FaceTime高清相机",
            "transport": "bltn"}
FIXED_USB = {"unique_id": "0x1340001b170209", "name": "2USB camer", "model": "UVC Camera VendorID_6935 ProductID_521",
             "transport": "usb"}
WRIST_USB = {"unique_id": "0x1310001b170209", "name": "2USB camer", "model": "UVC Camera VendorID_6935 ProductID_521",
             "transport": "usb"}
IPHONE = {"unique_id": "6A0F5C1E-0B7D-4E2A-9C3F-1D2E3F405162", "name": "iPhone Camera", "model": "iPhone15,2",
          "transport": None}


def mac(*devices, opencv_version="4.13.0"):
    return list_video_devices(platform="darwin", avfoundation=lambda: [dict(device) for device in devices],
                              opencv_version=opencv_version)


def camera(index_or_path=0, unique_id=None, device_name=None):
    return {"index_or_path": index_or_path, "unique_id": unique_id, "device_name": device_name,
            "width": 64, "height": 48, "fps": 30, "fourcc": None, "rotate_deg": 0}


class MacOSOrderTests(unittest.TestCase):
    def test_opencv_index_is_the_position_in_the_unique_id_sorted_list(self):
        # Earlier session with everything attached: OpenCV 0 wrist USB, 1 fixed USB, 2 FaceTime, 3 iPhone.
        inventory = mac(FACETIME, IPHONE, FIXED_USB, WRIST_USB)
        self.assertEqual(inventory["index_rule"], "avfoundation_sorted_unique_id")
        self.assertEqual([(d["index"], d["unique_ids"][0]) for d in inventory["devices"]],
                         [(0, WRIST_USB["unique_id"]), (1, FIXED_USB["unique_id"]), (2, FACETIME["unique_id"]),
                          (3, IPHONE["unique_id"])])
        wrist = resolve_camera("wrist", camera(5, unique_id=WRIST_USB["unique_id"]), inventory)
        fixed = resolve_camera("fixed", camera(5, unique_id=FIXED_USB["unique_id"]), inventory)
        self.assertEqual((wrist["mode"], wrist["opened"], wrist["api"]), ("unique_id", 0, "avfoundation"))
        self.assertEqual((fixed["opened"], fixed["device"]["transport"]), (1, "usb"))

    def test_system_profiler_order_is_not_the_opencv_order(self):
        # Today: system_profiler/AVFoundation list FaceTime first, but OpenCV opens the USB camera at index 0.
        inventory = mac(FACETIME, FIXED_USB)
        self.assertEqual(resolve_camera("fixed", camera(1, unique_id=FIXED_USB["unique_id"]), inventory)["opened"], 0)

    def test_wrist_missing_refuses_and_never_falls_back_to_the_index(self):
        """The exact 2026-11-16 case: wrist unplugged, index 1 would now open FaceTime as ``fixed``."""
        inventory = mac(FACETIME, FIXED_USB)
        with self.assertRaises(CameraBindingError) as raised:
            resolve_camera("wrist", camera(0, unique_id=WRIST_USB["unique_id"]), inventory)
        message = str(raised.exception)
        self.assertTrue(message.startswith(f"so101_camera_not_found:wrist:unique_id={WRIST_USB['unique_id']}"), message)
        self.assertIn("no fallback to index_or_path", message)
        self.assertIn(FIXED_USB["unique_id"], message)  # the operator sees what is present
        self.assertIn(FACETIME["unique_id"], message)
        # The fixed camera still binds to its own device, whatever its stale index says.
        fixed = resolve_camera("fixed", camera(1, unique_id=FIXED_USB["unique_id"]), inventory)
        self.assertEqual((fixed["opened"], fixed["device"]["name"]), (0, "2USB camer"))
        # Index-only (the old deployment) still opens index 1 but reports which device that is: the built-in camera.
        legacy = resolve_camera("fixed", camera(1), inventory)
        self.assertEqual((legacy["mode"], legacy["opened"], legacy["api"]), ("index", 1, "avfoundation"))
        self.assertEqual((legacy["device"]["unique_ids"], legacy["device"]["transport"]), ([FACETIME["unique_id"]], "bltn"))

    def test_same_name_is_ambiguous_and_name_plus_unique_id_must_both_hold(self):
        inventory = mac(FACETIME, FIXED_USB, WRIST_USB)
        with self.assertRaisesRegex(CameraBindingError, r"^so101_camera_ambiguous:wrist:device_name=2USB: 2 devices"):
            resolve_camera("wrist", camera(0, device_name="2USB"), inventory)
        with self.assertRaisesRegex(CameraBindingError, r"^so101_camera_not_found:fixed:unique_id=0x1340001b170209,"
                                                        r"device_name=FaceTime"):
            resolve_camera("fixed", camera(0, unique_id=FIXED_USB["unique_id"], device_name="FaceTime"), inventory)
        both = resolve_camera("fixed", camera(0, unique_id=FIXED_USB["unique_id"], device_name="usb camer"), inventory)
        self.assertEqual((both["mode"], both["opened"]), ("unique_id", 1))
        single = resolve_camera("fixed", camera(0, device_name="facetime"), inventory)  # unique name, any case
        self.assertEqual((single["mode"], single["opened"]), ("device_name", 2))

    def test_index_only_binding_keeps_working(self):
        inventory = mac(FACETIME, FIXED_USB, WRIST_USB)
        binding = resolve_camera("wrist", camera(0), inventory)
        self.assertEqual((binding["mode"], binding["opened"], binding["device"]["unique_ids"]),
                         ("index", 0, [WRIST_USB["unique_id"]]))
        beyond = resolve_camera("wrist", camera(7), inventory)
        self.assertEqual((beyond["opened"], beyond["device"], beyond["device_unknown_reason"]),
                         (7, None, "no_present_device_at_7"))
        failed = list_video_devices(platform="darwin", avfoundation=lambda: 1 / 0, opencv_version="4.13.0")
        self.assertIn("ZeroDivisionError", failed["error"])
        unknown = resolve_camera("wrist", camera(0), failed)  # an unreadable list never blocks index binding
        self.assertEqual((unknown["opened"], unknown["device"]), (0, None))
        with self.assertRaisesRegex(CameraBindingError, "^so101_camera_device_list_unavailable:wrist:unique_id="):
            resolve_camera("wrist", camera(0, unique_id=WRIST_USB["unique_id"]), failed)

    def test_opencv_without_the_sorted_order_refuses_identity_binding(self):
        for version, reason in (("4.6.0", "opencv_4.6.0_indexes_the_unsorted_avfoundation_list"),
                                (None, "opencv_not_importable")):
            with self.subTest(version=version), patch.object(camera_devices, "_opencv_version", return_value=None):
                inventory = mac(FACETIME, FIXED_USB, opencv_version=version)
                self.assertEqual((inventory["index_rule"], inventory["index_rule_error"]), (None, reason))
                self.assertEqual([d["index"] for d in inventory["devices"]], [None, None])
                with self.assertRaisesRegex(CameraBindingError, f"^so101_camera_index_rule_unverified:fixed:.*{reason}"):
                    resolve_camera("fixed", camera(1, unique_id=FIXED_USB["unique_id"]), inventory)
                self.assertEqual(resolve_camera("fixed", camera(1), inventory)["device_unknown_reason"], reason)
        odd = mac(FACETIME, dict(FIXED_USB, unique_id="0x13é"))
        self.assertEqual(odd["index_rule_error"], "unique_id_missing_or_not_ascii")

    def test_other_platforms_refuse_identity_and_keep_index_binding(self):
        inventory = list_video_devices(platform="win32")
        with self.assertRaisesRegex(CameraBindingError, "^so101_camera_device_list_unavailable:wrist:.*win32"):
            resolve_camera("wrist", camera(0, device_name="USB"), inventory)
        self.assertEqual(resolve_camera("wrist", camera(0), inventory)["api"], "any")

    def test_device_list_change_while_opening_is_detected(self):
        before = mac(FACETIME, FIXED_USB)
        confirm_unchanged("fixed", before, mac(FIXED_USB, FACETIME))  # same devices, other enumeration order
        with self.assertRaisesRegex(CameraBindingError, "^so101_camera_devices_changed:fixed"):
            confirm_unchanged("fixed", before, mac(FACETIME, FIXED_USB, WRIST_USB))  # wrist plugged back in


class LinuxTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        sysfs, links = self.root / "sys", self.root / "dev" / "v4l" / "by-id"
        for number, name in ((0, "2USB camer"), (1, "2USB camer"), (2, "USB Wrist Camera"), (3, "USB Wrist Camera")):
            (sysfs / f"video{number}").mkdir(parents=True)
            (sysfs / f"video{number}" / "name").write_text(name + "\n")
        links.mkdir(parents=True)
        for link, node in (("usb-2USB_camer_A-video-index0", 0), ("usb-2USB_camer_A-video-index1", 1),
                           ("usb-Wrist-video-index0", 2), ("usb-Wrist-video-index1", 3)):
            os.symlink(f"../../video{node}", links / link)
        by_path = self.root / "dev" / "v4l" / "by-path"
        by_path.mkdir()
        os.symlink("../../video2", by_path / "platform-xhci-hcd.0-usb-0:1:1.0-video-index0")
        self.inventory = list_video_devices(platform="linux", v4l_root=sysfs, v4l_links=self.root / "dev" / "v4l",
                                            opencv_version="4.13.0")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def test_by_id_and_by_path_links_bind_the_v4l2_node(self):
        self.assertEqual(self.inventory["devices"][2]["unique_ids"],
                         ["/dev/v4l/by-id/usb-Wrist-video-index0",
                          "/dev/v4l/by-path/platform-xhci-hcd.0-usb-0:1:1.0-video-index0"])
        for identity in ("/dev/v4l/by-id/usb-Wrist-video-index0",
                         "/dev/v4l/by-path/platform-xhci-hcd.0-usb-0:1:1.0-video-index0"):
            binding = resolve_camera("wrist", camera(0, unique_id=identity), self.inventory)
            self.assertEqual((binding["mode"], binding["opened"], binding["api"]), ("unique_id", "/dev/video2", "v4l2"))
        fixed = resolve_camera("fixed", camera(1, unique_id="/dev/v4l/by-id/usb-2USB_camer_A-video-index0"), self.inventory)
        self.assertEqual(fixed["opened"], "/dev/video0")
        with self.assertRaisesRegex(CameraBindingError, "^so101_camera_not_found:wrist:unique_id=/dev/v4l/by-id/usb-Gone"):
            resolve_camera("wrist", camera(0, unique_id="/dev/v4l/by-id/usb-Gone-video-index0"), self.inventory)

    def test_uvc_metadata_node_makes_a_name_ambiguous(self):
        with self.assertRaisesRegex(CameraBindingError, r"^so101_camera_ambiguous:wrist:device_name=Wrist: 2 devices.*"
                                                        r"/dev/video2.*/dev/video3"):
            resolve_camera("wrist", camera(0, device_name="Wrist"), self.inventory)
        self.assertEqual(resolve_camera("wrist", camera(2), self.inventory)["device"]["name"], "USB Wrist Camera")


class BindCamerasTests(unittest.TestCase):
    class Configured:
        def __init__(self, name, config):
            self.name, self.config = name, config

        def bind(self, inventory):
            return resolve_camera(self.name, self.config, inventory)

    def test_record_collects_refusals_and_duplicate_devices(self):
        inventory = mac(FACETIME, FIXED_USB)
        record = bind_cameras({"wrist": self.Configured("wrist", camera(0, unique_id=WRIST_USB["unique_id"])),
                               "fixed": self.Configured("fixed", camera(1, unique_id=FIXED_USB["unique_id"]))},
                              list_devices=lambda: inventory)
        self.assertEqual(len(record["refused"]), 1)
        self.assertTrue(record["refused"][0].startswith("so101_camera_not_found:wrist:"))
        self.assertEqual(binding_summary(record["cameras"]["fixed"])["opened"], 0)
        self.assertEqual(len(record["present"]), 2)
        duplicate = bind_cameras({"wrist": self.Configured("wrist", camera(0, device_name="2USB")),
                                  "fixed": self.Configured("fixed", camera(1, unique_id=FIXED_USB["unique_id"]))},
                                 list_devices=lambda: inventory)
        self.assertEqual(duplicate["refused"], ["so101_camera_duplicate_device:fixed,wrist: all open avfoundation 0"])
        self.assertIsNone(bind_cameras({"wrist": FakeCameraSource("wrist")}))  # fakes without bind: no record

    def test_log_lines_name_the_device_of_every_index_binding(self):
        inventory = mac(FACETIME, FIXED_USB)
        record = bind_cameras({"wrist": self.Configured("wrist", camera(0)), "fixed": self.Configured("fixed", camera(1))},
                              list_devices=lambda: inventory)
        self.assertEqual(record["refused"], [])
        lines = log_lines(record)
        fixed = next(line for line in lines if "camera=fixed" in line)
        self.assertIn("mode=index opened=1 name='FaceTime高清相机'", fixed)
        self.assertIn("transport=bltn", fixed)
        self.assertIn("set_cameras.fixed.unique_id", fixed)


class FakeCapture:
    """Stands in for cv2.VideoCapture; records what was opened."""
    opened = []

    def __init__(self, target, api=None):
        FakeCapture.opened.append((target, api))
        self.released = False

    def set(self, *_):
        return True

    def get(self, _):
        return 30.0

    def isOpened(self):
        return True

    def read(self):
        return False, None

    def release(self):
        self.released = True


class OpenCVCameraSourceTests(unittest.TestCase):
    def setUp(self):
        FakeCapture.opened = []

    def test_start_opens_the_resolved_index_with_avfoundation(self):
        inventory = mac(FACETIME, FIXED_USB)
        source = OpenCVCameraSource("fixed", camera(1, unique_id=FIXED_USB["unique_id"]), FakeClock())
        source.bind(inventory)
        with patch.object(cv2, "VideoCapture", FakeCapture), \
                patch.object(camera_devices, "list_video_devices", return_value=deepcopy(inventory)):
            source.start()
        try:
            self.assertEqual(FakeCapture.opened, [(0, cv2.CAP_AVFOUNDATION)])
        finally:
            source.close()

    def test_start_refuses_when_the_device_list_changed_after_resolution(self):
        source = OpenCVCameraSource("fixed", camera(1, unique_id=FIXED_USB["unique_id"]), FakeClock())
        source.bind(mac(FACETIME, FIXED_USB))
        with patch.object(cv2, "VideoCapture", FakeCapture), \
                patch.object(camera_devices, "list_video_devices", return_value=mac(FACETIME, FIXED_USB, WRIST_USB)), \
                self.assertRaisesRegex(CameraBindingError, "^so101_camera_devices_changed:fixed"):
            source.start()
        self.assertIsNone(source.capture)
        self.assertIsNone(source.thread)


class BindingCamera(FakeCameraSource):
    """Fake frames, real binding: lets the worker run its startup binding without OpenCV."""

    def __init__(self, camera_id, config, clock):
        super().__init__(camera_id, width=config["width"], height=config["height"], clock=clock)
        self.config, self.started = config, False

    def bind(self, inventory):
        return resolve_camera(self.camera_id, self.config, inventory)

    def start(self):
        self.started = True


class WorkerStartupTests(unittest.TestCase):
    def deployment(self, **cameras):
        document = json.loads(EXAMPLE.read_text())
        for name, values in cameras.items():
            document["cameras"][name].update(values)
        return validate_deployment(document, base=EXAMPLE.parent)

    def adapter(self, deployment, directory, inventory):
        clock = FakeClock()
        driver = FakeArmDriver(deployment, clock)
        cameras = {name: BindingCamera(name, config, clock) for name, config in deployment["cameras"].items()}
        with patch.object(camera_devices, "list_video_devices", return_value=inventory):
            adapter = SO101Adapter(deployment, directory, "rgb_proprio_v1", driver=driver, cameras=cameras, clock=clock,
                                   image_settings=Config().images.to_dict(), start_telemetry=False)
        return adapter, driver, cameras

    def test_missing_wrist_refuses_before_the_bus_or_any_camera_is_touched(self):
        deployment = self.deployment(wrist={"unique_id": WRIST_USB["unique_id"]}, fixed={"unique_id": FIXED_USB["unique_id"]})
        with tempfile.TemporaryDirectory() as directory:
            clock = FakeClock()
            driver = FakeArmDriver(deployment, clock)
            cameras = {name: BindingCamera(name, config, clock) for name, config in deployment["cameras"].items()}
            with patch.object(camera_devices, "list_video_devices", return_value=mac(FACETIME, FIXED_USB)), \
                    self.assertRaisesRegex(CameraBindingError, "^so101_camera_not_found:wrist:"):
                SO101Adapter(deployment, directory, "rgb_proprio_v1", driver=driver, cameras=cameras, clock=clock,
                             image_settings=Config().images.to_dict(), start_telemetry=False)
            self.assertFalse(driver.connected)
            self.assertEqual(driver.sends, 0)
            self.assertFalse(any(camera.started for camera in cameras.values()))
            record = json.loads((Path(directory) / "camera_binding.json").read_text())
            self.assertIn("refused", record["cameras"]["wrist"])
            self.assertEqual(record["cameras"]["fixed"]["opened"], 0)
            self.assertEqual([d["unique_ids"][0] for d in record["present"]], [FIXED_USB["unique_id"], FACETIME["unique_id"]])

    def test_index_binding_records_the_opened_device_for_the_operator_only(self):
        deployment = self.deployment()  # example: wrist 0, fixed 1, no identity
        with tempfile.TemporaryDirectory() as directory, patch.object(sys, "stderr") as stderr:
            adapter, driver, cameras = self.adapter(deployment, directory, mac(FACETIME, FIXED_USB))
            self.assertTrue(driver.connected and all(camera.started for camera in cameras.values()))
            record = json.loads((Path(directory) / "camera_binding.json").read_text())
            self.assertEqual(record["cameras"]["fixed"]["device"]["transport"], "bltn")
            self.assertTrue(record["checked_at"].endswith("Z"))
            logged = "".join(call.args[0] for call in stderr.write.call_args_list)
            self.assertIn("so101_camera_binding camera=fixed mode=index opened=1 name='FaceTime高清相机'", logged)
            s0 = adapter.workflow._s0(lambda update: None)
            self.assertEqual(s0["detail"]["cameras"]["fixed"]["device"]["unique_ids"], [FACETIME["unique_id"]])
            # The actor's view never carries device identities.
            published = json.dumps([adapter.capabilities(), adapter.observe()], default=str)
            for value in (FACETIME["unique_id"], FIXED_USB["unique_id"], "FaceTime", "camera_binding"):
                self.assertNotIn(value, published)


class DeploymentFieldTests(unittest.TestCase):
    def test_unique_id_is_optional_stripped_and_not_shared(self):
        document = json.loads(EXAMPLE.read_text())
        self.assertIsNone(validate_deployment(document, base=EXAMPLE.parent)["cameras"]["fixed"]["unique_id"])
        document["cameras"]["fixed"]["unique_id"] = "  0x1340001b170209 "
        self.assertEqual(validate_deployment(document, base=EXAMPLE.parent)["cameras"]["fixed"]["unique_id"],
                         "0x1340001b170209")
        for bad in ("", "  ", 7):
            candidate = deepcopy(document)
            candidate["cameras"]["wrist"]["unique_id"] = bad
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "cameras.wrist.unique_id"):
                validate_deployment(candidate, base=EXAMPLE.parent)
        document["cameras"]["wrist"]["unique_id"] = "0x1340001b170209"
        with self.assertRaisesRegex(ValueError, "cameras.unique_id: two cameras name the same device"):
            validate_deployment(document, base=EXAMPLE.parent)

# === tests/test_so101_capture_safety.py ===

import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from ni_rua.robot.calibration.workflow import capture, touch_off


class CaptureSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.adapter, self.driver, self.clock = adapter_with(self.root)
        self.settings = dict(self.adapter.deployment["calibration"])
        self.pose = list(self.adapter.deployment["home"]["q_rad"])
        self.pose[0] += .1

    def run_capture(self, **kwargs):
        return capture.capture_session(self.adapter, self.root / "session", settings=self.settings,
                                       poses=[self.pose], **kwargs)

    def test_hold_before_first_move_prevents_all_pose_and_park_requests(self):
        def progress(row):
            if row["phase"] == "moving":
                self.adapter.operator("hold")
        with patch.object(self.adapter, "move_joints", wraps=self.adapter.move_joints) as move:
            result = self.run_capture(progress=progress)
        self.assertEqual(result["status"], "stopped")
        self.assertTrue(result["stop_confirmation"]["confirmed"])
        self.assertIsNone(result["park"])
        move.assert_not_called()

    def test_hold_in_last_sample_or_before_park_never_starts_park(self):
        for phase in ("sample", "parking"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                adapter, _, _ = adapter_with(Path(directory))
                def progress(row):
                    if row["phase"] == phase:
                        adapter.operator("hold")
                with patch.object(adapter, "move_joints", wraps=adapter.move_joints) as move:
                    result = capture.capture_session(adapter, Path(directory) / "capture", poses=[self.pose], progress=progress)
                self.assertEqual(result["status"], "stopped")
                self.assertIsNone(result["park"])
                self.assertEqual([call.kwargs["action_id"] for call in move.call_args_list], ["calibration_pose_0"])

    def test_stop_between_boundary_check_and_move_cannot_rebind_generation(self):
        original = self.adapter.move_joints
        def race(*args, **kwargs):
            self.adapter.operator("hold")
            return original(*args, **kwargs)
        with patch.object(self.adapter, "move_joints", side_effect=race):
            result = self.run_capture()
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["poses"][0]["status"], "stopped")
        self.assertIsNone(result["park"])
        self.assertEqual(result["kept"], 0)

    def test_deadline_during_final_capture_stops_before_sample_commit_and_park(self):
        original = self.adapter.capture_frames
        def slow(t):
            frames = original(t)
            self.clock.sleep(self.settings["capture_timeout_s"] + 1)
            return frames
        with patch.object(self.adapter, "capture_frames", side_effect=slow):
            result = self.run_capture()
        self.assertEqual((result["status"], result["reason"]), ("failed", "capture_timeout"))
        self.assertEqual(result["kept"], 0)
        self.assertIsNone(result["park"])
        self.assertTrue(result["physics_advanced"])

    def test_camera_exception_keeps_physical_receipt_and_stops(self):
        with patch.object(self.adapter, "capture_frames", side_effect=TimeoutError("camera unavailable")):
            result = self.run_capture()
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["physics_advanced"])
        self.assertTrue(result["stop_confirmation"]["confirmed"])
        self.assertIsNone(result["park"])

    def test_progress_exception_keeps_partial_capture_and_confirms_stop(self):
        def progress(row):
            if row["phase"] == "sample":
                raise RuntimeError("progress sink closed")
        result = self.run_capture(progress=progress)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["physics_advanced"])
        self.assertEqual(result["kept"], 1)
        self.assertTrue(result["stop_confirmation"]["confirmed"])
        self.assertIsNone(result["park"])

    def test_unconfirmed_hold_remains_unknown(self):
        def progress(row):
            if row["phase"] == "sample":
                self.adapter.operator("hold")
        with patch.object(self.adapter, "_stop", return_value={"confirmed": False, "state": "unknown"}):
            result = self.run_capture(progress=progress)
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["stop_confirmation"]["confirmed"])
        self.assertIsNone(result["park"])

    def test_relax_never_sends_a_new_hold_or_park(self):
        def progress(row):
            if row["phase"] == "sample":
                self.adapter.operator("relax")
        with patch.object(self.adapter, "_stop", wraps=self.adapter._stop) as stop:
            result = self.run_capture(progress=progress)
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["stop_confirmation"]["state"], "torque_off")
        self.assertIsNone(result["park"])
        stop.assert_not_called()

    def test_capture_timing_is_host_bracketed_not_claimed_synchronous(self):
        result = self.run_capture()
        self.assertEqual(result["status"], "completed")
        row = json.loads((self.root / "session" / "samples.jsonl").read_text().splitlines()[0])
        self.assertLessEqual(row["joint_before_monotonic_s"], row["capture_host_interval_s"][0])
        self.assertLessEqual(row["capture_host_interval_s"][1], row["joint_after_monotonic_s"])
        self.assertEqual(set(row["camera_timing"]), {"fixed", "wrist"})
        self.assertTrue(all(t["exposure_monotonic_s"] is None for t in row["camera_timing"].values()))

    def test_capture_archive_error_does_not_erase_motion(self):
        original = Path.write_text
        def write(path, *args, **kwargs):
            if path.name == "capture.json":
                raise OSError("disk full")
            return original(path, *args, **kwargs)
        with patch.object(Path, "write_text", write):
            result = self.run_capture()
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["physics_advanced"])
        self.assertFalse(result["evidence_persisted"])
        self.assertEqual(result["park"]["status"], "completed")

    def test_move_settle_failure_is_not_overwritten_by_small_position_error(self):
        original = self.adapter._settle
        calls = 0
        def settle(run, timeout, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return False, self.adapter.read(), "settle_timeout"
            return original(run, timeout, **kwargs)
        with patch.object(self.adapter, "_settle", side_effect=settle), \
                patch.object(self.adapter, "_gripper", wraps=self.adapter._gripper) as jaw:
            result = self.adapter.move_joints(self.adapter.read()["q_rad"], jaw_normalized=.5)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "settle_timeout")
        jaw.assert_not_called()

    def test_touch_hold_before_home_cannot_start_new_sequence(self):
        def progress(row):
            if row["phase"] == "moving":
                self.adapter.operator("hold")
        result = touch_off.touch_points(self.adapter, [[.22, 0]], self.settings, progress=progress)
        self.assertEqual(result["status"], "stopped")
        self.assertNotIn("descent_traces", result)
        self.assertIsNone(result["park"])
        self.assertEqual(result["motion_segments"][0]["result"]["dispatched"], [])

    def test_final_sag_settling_fact_replaces_initial_settled_fact(self):
        with patch.object(self.adapter, "_arrive", return_value=(True, None, {"settled": False})):
            result = self.adapter.move_joints(self.adapter.read()["q_rad"])
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["settled"])
        self.assertEqual(result["reason"], "settle_unconfirmed")

    def test_stop_after_final_jaw_read_is_not_completed(self):
        def jaw(*args):
            self.adapter.operator("hold")
            return True, None, {"jaw_settled": True}
        with patch.object(self.adapter, "_gripper", side_effect=jaw):
            result = self.adapter.move_joints(self.adapter.read()["q_rad"], jaw_normalized=.5)
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["reason"], "operator_hold")

    def test_settle_final_tick_cannot_send_sag_correction_past_deadline(self):
        self.adapter.deployment["motion"]["arrival_correction_rounds"] = 1
        target = list(self.adapter.read()["q_rad"])
        target[0] += .03
        run = self.adapter._new_run(threading.Event(), .275, "deadline_repro")
        with patch.object(self.adapter, "_send", wraps=self.adapter._send) as send:
            ok, reason, _ = self.adapter._arrive(run, target, self.adapter._pose_dict(self.adapter.tcp(target)),
                                                (float("inf"), 3.141592653589793))
        self.assertFalse(ok)
        self.assertEqual(reason, "settle_timeout")
        send.assert_not_called()

    def test_driver_fault_after_relax_cannot_send_hold_goals(self):
        from ni_rua.robot.backends.so101.worker import DriverFault
        def segment(*args, **kwargs):
            self.adapter.operator("relax")
            raise DriverFault("uncertain operation")
        with patch.object(self.adapter, "_segment", side_effect=segment), \
                patch.object(self.adapter, "_send", wraps=self.adapter._send) as send:
            result = self.adapter.move_joints(self.pose)
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["stop_confirmation"]["confirmed"])
        self.assertEqual(result["stop_confirmation"]["state"], "torque_off")
        send.assert_not_called()

    def test_handheld_stop_waits_for_writer_and_cannot_solve_pending_png(self):
        import cv2
        handheld = capture.HandheldCapture(self.adapter, self.root / "handheld")
        entered, release = threading.Event(), threading.Event()
        original = cv2.imwrite
        def blocked(*args, **kwargs):
            entered.set()
            release.wait(10)
            return original(*args, **kwargs)
        try:
            with patch.object(cv2, "imwrite", side_effect=blocked):
                handheld.start()
                self.assertTrue(entered.wait(2))
                stopped = handheld.stop()
                self.assertEqual(stopped["status"], "unknown")
                self.assertFalse(stopped["stop_confirmed"])
                self.assertFalse(stopped["sealed"])
                self.assertTrue(handheld.running)
                with self.assertRaisesRegex(RuntimeError, "writer_not_stopped"):
                    handheld.kept_images()
                release.set()
                handheld.thread.join(timeout=2)
        finally:
            release.set()
            handheld.thread.join(timeout=2)
        stopped = handheld.stop()
        self.assertTrue(stopped["stop_confirmed"])
        self.assertTrue(stopped["sealed"])
        self.assertFalse(handheld.running)
        self.assertEqual(len(handheld.kept_images()), 1)

    def test_handheld_stop_during_camera_wait_does_not_commit_late_frame(self):
        handheld = capture.HandheldCapture(self.adapter, self.root / "handheld")
        camera = self.adapter.cameras["fixed"]
        entered, release = threading.Event(), threading.Event()
        original = camera.read_after
        def blocked(*args):
            entered.set()
            release.wait(5)
            return original(*args)
        try:
            with patch.object(camera, "read_after", side_effect=blocked):
                handheld.start()
                self.assertTrue(entered.wait(2))
                with patch.object(handheld.thread, "join", return_value=None):
                    self.assertFalse(handheld.stop()["stop_confirmed"])
                release.set()
                handheld.thread.join(timeout=2)
        finally:
            release.set()
            handheld.thread.join(timeout=2)
        self.assertTrue(handheld.stop()["stop_confirmed"])
        self.assertEqual(handheld.kept_images(), [])

    def test_transient_within_tolerance_without_settling_fails_with_not_settled(self):
        target = list(self.adapter.read()["q_rad"])
        pose = self.adapter._pose_dict(self.adapter.tcp(target))
        run = self.adapter._new_run(threading.Event(), 10.0, "transient_pass_through")
        sample = dict(self.adapter.read(), q_rad=list(target))
        with patch.object(self.adapter, "_settle", return_value=(False, sample, "settle_timeout")):
            ok, reason, residual = self.adapter._arrive(run, target, pose, (0.005, 0.1))
            self.assertFalse(ok)
            self.assertEqual(reason, "not_settled")
            self.assertFalse(residual["settled"])
            self.assertTrue(residual["within_tolerance"])

            ok_opt_out, reason_opt_out, residual_opt_out = self.adapter._arrive(
                run, target, pose, (0.005, 0.1), require_settled=False
            )
            self.assertTrue(ok_opt_out)
            self.assertIsNone(reason_opt_out)
            self.assertFalse(residual_opt_out["settled"])
            self.assertTrue(residual_opt_out["within_tolerance"])


# === tests/test_so101_joint_calibration.py ===

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from ni_rua.robot.calibration.workflow import state_machine as wf
from ni_rua.robot.backends.so101.deployment import load_deployment
from ni_rua.robot.calibration.workflow import charuco as cal
from ni_rua.robot.model import so101_kinematics as Kin
from ni_rua.robot.embodiments.so101.contact import lowest_gripper_point
from ni_rua.robot.calibration.workflow.joint_calibration import (
    JOINTS,
    Sample,
    board_corner_height_error_m,
    solve_joint_calibration,
)
from robot.so101.test_so101_backend import EXAMPLE


class JointCalibrationTests(unittest.TestCase):
    def test_explicit_solver_budget_tolerances_and_robust_scale_are_effective(self):
        from ni_rua.robot.calibration.workflow import joint_calibration as joint
        samples, board, K, dist, fk, X, B, _model, _offsets, table_z, thickness = self.scene()
        requested = dict(max_nfev=1, pixel_robust_scale=2.5, ftol=2e-8, xtol=3e-8, gtol=4e-8, x_scale=1.5)
        with patch.object(joint, "least_squares", wraps=joint.least_squares) as solve, \
             patch.object(joint, "_robust_pixel_residual", wraps=joint._robust_pixel_residual) as robust:
            result = joint.solve_joint_calibration(samples, board, K, dist, fk,
                T_gripper_camera0=X, T_base_board0=B, table_z_m=table_z,
                board_thickness_m=thickness, **requested)
        self.assertEqual(result.nfev, 1)
        self.assertFalse(result.success)  # a budget exhaustion is not a passing solve
        for name, value in requested.items():
            self.assertEqual(result.effective_solver_settings[name], value)
            if name != "pixel_robust_scale":
                self.assertEqual(solve.call_args.kwargs[name], value)
        self.assertTrue(robust.call_args_list)
        self.assertTrue(all(call.args[1] == 2.5 for call in robust.call_args_list))
        self.assertEqual(result.to_dict()["nfev"], 1)
        for invalid in ({"max_nfev": True}, {"max_nfev": 0}, {"pixel_robust_scale": 0},
                        {"ftol": float("nan")}, {"xtol": 1e-16}, {"gtol": 0}, {"x_scale": -1}):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "invalid_joint_solver"):
                joint.solve_joint_calibration(samples, board, K, dist, fk,
                    T_gripper_camera0=X, T_base_board0=B, table_z_m=table_z,
                    board_thickness_m=thickness, **invalid)

    def test_recovers_injected_joint_offsets_and_board_height(self):
        rng = np.random.default_rng(3)
        board = cal.make_board()
        model = Kin.load_model()
        fk = lambda q: Kin.fk(model, q, Kin.WRIST_CAMERA_PARENT_LINK)
        K = np.array([[510., 0, 320], [0, 510., 240], [0, 0, 1]]); dist = np.zeros(5)
        true_offsets = np.radians([0, 3.0, -4.0, 2.0, 0])
        table_z, thickness = -0.002, 0.005
        ready = np.array([0.0, -0.95, 0.9, 1.0, 0.0])
        # camera looks along the TCP approach axis (down at the ready pose), 5 cm behind the TCP
        q0 = ready + true_offsets
        X = cal.invert_transform(fk(q0)) @ Kin.fk(model, q0) @ cal.make_transform(
            cal.rotvec_to_matrix([0.0, 0.0, 0.3]), [0.0, 0.02, -0.05])
        T_cam = fk(q0) @ X
        depth = (T_cam[2, 3] - (table_z + thickness)) / -T_cam[2, 2]
        centre = T_cam[:3, 3] + T_cam[:3, 2] * depth
        B = cal.board_frame_from_center([centre[0], centre[1], table_z + thickness], board)
        samples = []
        for _ in range(18):
            q = ready + rng.normal(0, [0.12, 0.06, 0.06, 0.12, 0.3])
            T_cb = cal.invert_transform(fk(q + true_offsets) @ X) @ B
            ids = np.arange(len(cal.board_corner_points(board)))
            obj = cal.board_corner_points(board, ids)
            pts, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), cv2.Rodrigues(T_cb[:3, :3])[0], T_cb[:3, 3], K, dist)
            pts = pts.reshape(-1, 2)
            keep = (pts[:, 0] > 5) & (pts[:, 0] < 635) & (pts[:, 1] > 5) & (pts[:, 1] < 475) & (T_cb[2, 3] > 0.05)
            if keep.sum() >= 8:
                samples.append(Sample(q, pts[keep] + rng.normal(0, 0.2, (keep.sum(), 2)), ids[keep]))
        self.assertGreaterEqual(len(samples), 10)
        guess_X = X @ cal.make_transform(cal.rotvec_to_matrix([0.02, -0.02, 0.03]), [0.004, -0.003, 0.005])
        result = solve_joint_calibration(samples, board, K, dist, fk, T_gripper_camera0=guess_X,
                                         T_base_board0=B @ cal.make_transform(np.eye(3), [0.01, -0.01, 0.01]),
                                         table_z_m=table_z, board_thickness_m=thickness)
        recovered = np.array([result.joint_offsets_rad[name] for name in JOINTS])
        self.assertLess(np.abs(recovered - true_offsets).max(), np.radians(0.5), np.degrees(recovered))
        self.assertLess(result.reprojection_rms_px, 0.5)
        self.assertAlmostEqual(result.board_top_z_m, table_z + thickness, delta=0.002)
        self.assertEqual(result.unobservable_joints, ("shoulder_pan", "wrist_roll"))

    def test_estimates_table_height_from_touches_under_a_prior(self):
        """Touches + board plane: the table height is recovered from a 6 mm-off nominal prior."""
        samples, board, K, dist, fk, X, B, model, true_offsets, table_z, thickness = self.scene()
        touches = []
        for xy_q in ([0.0, -0.2, 1.0, 0.3, 0.0], [0.15, 0.3, 0.6, -0.05, 0.0], [-0.15, 0.0, 0.85, 0.15, 0.0]):
            q_true = np.asarray(xy_q)
            low = lowest_gripper_point(model, q_true)
            # move the true arm so the lowest point sits exactly on the table: shift shoulder_lift until z = table
            for _ in range(60):
                low = lowest_gripper_point(model, q_true)
                q_true = q_true + np.array([0, 0.5 * (low["z_m"] - table_z), 0, 0, 0])
            touches.append((q_true - true_offsets, low["link"], low["point_in_link_m"]))
        fk_all = lambda q: Kin.fk_all(model, q)
        result = solve_joint_calibration(samples, board, K, dist, fk, T_gripper_camera0=X, T_base_board0=B,
                                         table_z_m=table_z + 0.006, board_thickness_m=thickness, touches=touches,
                                         fk_all=fk_all, table_prior_sigma_m=0.01)
        self.assertAlmostEqual(result.table_z_m, table_z, delta=0.0015)
        self.assertEqual(len(result.touch_residuals_m), 3)
        self.assertLess(max(abs(v) for v in result.touch_residuals_m), 0.002)
        self.assertAlmostEqual(result.board_top_z_m, result.table_z_m + thickness, delta=0.002)

    def test_downward_facing_board_converges_across_pi_rotvec_boundary(self):
        """Local Lie-algebra tangent perturbations avoid the |r| = pi Rodrigues branch-cut singularity."""
        samples, board, K, dist, fk, X, B, _model, true_offsets, table_z, thickness = self.scene()
        self.assertAlmostEqual(float(np.linalg.norm(cal.matrix_to_rotvec(B[:3, :3]))), np.pi, places=6)
        rng = np.random.default_rng(17)
        for trial in range(6):
            with self.subTest(trial=trial):
                rot_noise_B = rng.normal(0.0, 0.06, 3)
                pos_noise_B = rng.normal(0.0, 0.012, 3)
                rot_noise_X = rng.normal(0.0, 0.03, 3)
                pos_noise_X = rng.normal(0.0, 0.005, 3)
                guess_B = B @ cal.make_transform(cal.rotvec_to_matrix(rot_noise_B), pos_noise_B)
                guess_X = X @ cal.make_transform(cal.rotvec_to_matrix(rot_noise_X), pos_noise_X)
                result = solve_joint_calibration(samples, board, K, dist, fk, T_gripper_camera0=guess_X,
                                                 T_base_board0=guess_B, table_z_m=table_z,
                                                 board_thickness_m=thickness)
                self.assertTrue(result.success)
                self.assertLess(result.reprojection_rms_px, 0.5)
                self.assertLess(result.board_tilt_rad, 0.01)
                recovered = np.array([result.joint_offsets_rad[name] for name in JOINTS])
                self.assertLess(np.abs(recovered - true_offsets).max(), np.radians(0.6))

    def test_repeated_recalibration_with_initial_offsets_does_not_random_walk(self):
        """Penalizing total offset (initial_offsets_rad + delta) prevents drift across repeated solves."""
        samples0, board, K, dist, fk, X, B, _model, true_offsets, table_z, thickness = self.scene()
        current_offsets = {name: 0.0 for name in JOINTS}
        for _ in range(5):
            cur_vec = np.array([current_offsets[name] for name in JOINTS])
            shifted_samples = [Sample(s.q_rad + cur_vec, s.corners_px, s.ids, board=s.board) for s in samples0]
            result = solve_joint_calibration(shifted_samples, board, K, dist, fk, T_gripper_camera0=X,
                                             T_base_board0=B, table_z_m=table_z, board_thickness_m=thickness,
                                             initial_offsets_rad=current_offsets)
            for name in JOINTS:
                current_offsets[name] += result.joint_offsets_rad[name]
        final_vec = np.array([current_offsets[name] for name in JOINTS])
        self.assertLess(np.abs(final_vec - true_offsets).max(), np.radians(0.5))

    def test_tilted_board_with_valid_center_height_fails_corner_height_check(self):
        """A tilted board whose center sits at table_z + thickness still fails 4-corner board_height_error_m."""
        board = cal.make_board()
        table_z, thickness = -0.02, 0.003
        B_flat = cal.board_frame_from_center([0.25, 0.0, table_z + thickness], board)
        outline = cal.board_outline_points(board)
        center_in_board = 0.5 * (outline[0] + outline[2])
        # Tilt by 10 degrees about an axis passing through the board center so center z stays at table_z + thickness.
        tilt_R = cal.rotvec_to_matrix([np.radians(10.0), 0.0, 0.0])
        T_pivot = cal.make_transform(np.eye(3), center_in_board) @ cal.make_transform(tilt_R, [0, 0, 0]) @ cal.make_transform(
            np.eye(3), -center_in_board)
        B_tilted = B_flat @ T_pivot
        center_z = float(np.mean((B_tilted[:3, :3] @ outline.T).T[:, 2] + B_tilted[2, 3]))
        self.assertAlmostEqual(center_z, table_z + thickness, places=6)
        corner_err = board_corner_height_error_m(B_tilted, board, table_z, thickness)
        settings = load_deployment(EXAMPLE)["calibration"]
        self.assertGreater(corner_err, settings["max_board_height_error_m"])
        verdict = wf.evaluate("S5", {"joint_reprojection_px": 0.4, "board_height_error_m": corner_err,
                                     "max_touch_residual_m": 0.001, "fixed_chain_reprojection_px": 0.8,
                                     "max_joint_offset_rad": 0.02}, settings)
        self.assertFalse(verdict["passed"])
        failed = {row["id"] for row in verdict["criteria"] if not row["passed"]}
        self.assertIn("board_height_error_m", failed)

    def test_dual_camera_bundle_adjustment_optimizes_fixed_extrinsic(self):
        """Simultaneous wrist + fixed camera BA recovers T_base_fixed alongside joint offsets and board pose."""
        samples, board, K, dist, fk, X, B, _model, true_offsets, table_z, thickness = self.scene()
        rng = np.random.default_rng(11)
        eye, target = np.array([0.48, 0.02, 0.26]), np.array([0.24, 0.0, table_z])
        z = (target - eye) / np.linalg.norm(target - eye)
        x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
        T_base_fixed_true = cal.make_transform(np.column_stack([x, np.cross(z, x), z]), eye)
        T_fb = cal.invert_transform(T_base_fixed_true) @ B
        ids = np.arange(len(cal.board_corner_points(board)))
        obj = cal.board_corner_points(board, ids)
        pts, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), cv2.Rodrigues(T_fb[:3, :3])[0], T_fb[:3, 3], K, dist)
        pts = pts.reshape(-1, 2)
        fixed_samples = [Sample(np.zeros(5), pts + rng.normal(0, 0.15, pts.shape), ids, board=0) for _ in range(5)]
        guess_fixed = T_base_fixed_true @ cal.make_transform(cal.rotvec_to_matrix([0.04, -0.03, 0.02]), [0.025, -0.015, 0.02])
        result = solve_joint_calibration(samples, board, K, dist, fk, T_gripper_camera0=X, T_base_board0=B,
                                         table_z_m=table_z, board_thickness_m=thickness,
                                         fixed_samples=fixed_samples, fixed_K=K, fixed_dist=dist,
                                         T_base_fixed0=guess_fixed)
        self.assertIsNotNone(result.T_base_fixed)
        self.assertIsNotNone(result.fixed_reprojection_rms_px)
        self.assertLess(result.fixed_reprojection_rms_px, 0.5)
        self.assertLess(float(np.linalg.norm(result.T_base_fixed[:3, 3] - T_base_fixed_true[:3, 3])), 0.003)
        recovered = np.array([result.joint_offsets_rad[name] for name in JOINTS])
        self.assertLess(np.abs(recovered - true_offsets).max(), np.radians(0.5))

    def scene(self):
        rng = np.random.default_rng(3)
        board = cal.make_board()
        model = Kin.load_model()
        fk = lambda q: Kin.fk(model, q, Kin.WRIST_CAMERA_PARENT_LINK)
        K = np.array([[510., 0, 320], [0, 510., 240], [0, 0, 1]]); dist = np.zeros(5)
        true_offsets = np.radians([0, 3.0, -4.0, 2.0, 0])
        table_z, thickness = -0.002, 0.005
        ready = np.array([0.0, -0.95, 0.9, 1.0, 0.0])
        q0 = ready + true_offsets
        X = cal.invert_transform(fk(q0)) @ Kin.fk(model, q0) @ cal.make_transform(
            cal.rotvec_to_matrix([0.0, 0.0, 0.3]), [0.0, 0.02, -0.05])
        T_cam = fk(q0) @ X
        depth = (T_cam[2, 3] - (table_z + thickness)) / -T_cam[2, 2]
        centre = T_cam[:3, 3] + T_cam[:3, 2] * depth
        B = cal.board_frame_from_center([centre[0], centre[1], table_z + thickness], board)
        samples = []
        for _ in range(18):
            q = ready + rng.normal(0, [0.12, 0.06, 0.06, 0.12, 0.3])
            T_cb = cal.invert_transform(fk(q + true_offsets) @ X) @ B
            ids = np.arange(len(cal.board_corner_points(board)))
            pts, _ = cv2.projectPoints(cal.board_corner_points(board, ids).reshape(-1, 1, 3),
                                       cv2.Rodrigues(T_cb[:3, :3])[0], T_cb[:3, 3], K, dist)
            pts = pts.reshape(-1, 2)
            keep = (pts[:, 0] > 5) & (pts[:, 0] < 635) & (pts[:, 1] > 5) & (pts[:, 1] < 475) & (T_cb[2, 3] > 0.05)
            if keep.sum() >= 8:
                samples.append(Sample(q, pts[keep] + rng.normal(0, 0.2, (keep.sum(), 2)), ids[keep]))
        return samples, board, K, dist, fk, X, B, model, true_offsets, table_z, thickness


if __name__ == "__main__":
    unittest.main()
