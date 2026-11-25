"""Hand-held fixed-camera intrinsics inside the SO-101 worker (CPU only, fake driver/cameras).

Covers: capture samples the worker's existing fixed camera source (synthetic ChArUco renders), keeps
still and novel views, drops frames whose size differs from the configured capture size, refuses to
start during an action; handheld_solve gates on calibration.max_handheld_intrinsics_rms_px, installs a
file with the fixed camera's own K and intrinsics_source "handheld", re-solves T_base_camera only when
the last robot session record exists, and keeps the old file on failure. Real optics are not covered.
"""

NI_RUA_TEST_TIER = "fast"

# === tests/test_so101_handheld.py ===

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

import numpy as np

from ni_rua.robot.calibration.workflow import capture
from ni_rua.robot.backends.so101.driver import FakeCameraSource
from robot.so101.test_so101_backend import deployment, make_adapter

try:
    import cv2
    HAVE_ARUCO = hasattr(cv2, "aruco") and hasattr(cv2.aruco, "CharucoDetector")
except ImportError:  # pragma: no cover
    HAVE_ARUCO = False

if HAVE_ARUCO:
    from ni_rua.robot.calibration.workflow import charuco as cal
    from robot.calibration.test_so101_calibration import board_in_view, random_rotation, render_board
    from robot.so101.test_so101_camera_and_recalibration import write_calibration

SIZE = (640, 480)
K_FIXED = np.array([[700.0, 0, 330], [0, 700.0, 250], [0, 0, 1]])


def handheld_frames(count, seed=3):
    """RGB renders of the board at ``count`` varied poses in front of K_FIXED, each repeated twice (held still)."""
    rng, board, frames = np.random.default_rng(seed), cal.make_board(), []
    while len(frames) < 2 * count:
        T = cal.make_transform(random_rotation(rng, 0.6), [rng.uniform(-0.2, 0.0), rng.uniform(-0.14, -0.02),
                                                           rng.uniform(0.3, 0.5)])
        if board_in_view(board, T, K_FIXED, SIZE, margin_px=20):
            rgb = cv2.cvtColor(render_board(board, T, K_FIXED, SIZE), cv2.COLOR_BGR2RGB)
            frames += [rgb, rgb]
    return frames


class SequenceCamera(FakeCameraSource):
    """Fake camera that plays ``frames`` once (then times out like a camera with no new frame)."""

    def __init__(self, camera_id, clock, frames):
        super().__init__(camera_id, width=SIZE[0], height=SIZE[1], clock=clock)
        self.frames, self.lock = list(frames), threading.Lock()

    def read_after(self, t, timeout_s):
        with self.lock:
            if not self.frames:
                time.sleep(0.005)
                raise TimeoutError("so101_camera_frame_timeout:fixed:no_new_frame")
            image = self.frames.pop(0)
        self.last = (image, max(t, self.clock.now()) + 0.01)
        return self.last


def adapter_with(directory, frames, calibration_file=None, **calibration):
    deploy = deployment(**{f"calibration.{key}": value for key, value in calibration.items()})
    for name in ("wrist", "fixed"):
        deploy["cameras"][name].update(width=SIZE[0], height=SIZE[1])
    deploy["camera_calibration_file"] = None if calibration_file is None else str(calibration_file)
    adapter, _, clock = make_adapter(directory, deploy=deploy)
    adapter.cameras = {"wrist": FakeCameraSource("wrist", clock=clock, width=SIZE[0], height=SIZE[1]),
                       "fixed": SequenceCamera("fixed", clock, frames)}
    return adapter


def run_capture(adapter, frame_count):
    """Start a capture and stop it once all ``frame_count`` frames were judged (dropped for their size or offered
    to the view selector), or when the capture ends by itself. ``HandheldCapture.frames`` counts a frame when it is
    read, before it is judged, and a stop discards a frame still being judged, so stopping on that counter alone
    made ``kept`` depend on thread timing."""
    result = adapter.operator("handheld_capture", action="start")
    deadline = time.monotonic() + 60
    capture = adapter.handheld
    while capture.running and capture.dropped + capture.selector.frames < frame_count:
        if time.monotonic() > deadline:
            raise AssertionError("handheld capture did not consume the frames")
        time.sleep(0.01)
    return result, adapter.operator("handheld_capture", action="stop")


@unittest.skipUnless(HAVE_ARUCO, "OpenCV with aruco CharucoDetector required")
class HandheldCaptureTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.directory, True)
        self.path = self.directory / "camera_calibration.json"

    def test_keeps_still_novel_views_and_reports_coverage(self):
        frames = handheld_frames(12)
        adapter = adapter_with(self.directory, frames, handheld_target_views=10, handheld_min_novelty=0.02)
        started, stopped = run_capture(adapter, len(frames))
        self.assertEqual(started["status"], "running")
        self.assertEqual(stopped["status"], "completed", stopped)
        self.assertEqual(stopped["kept"], 10)
        self.assertTrue(stopped["ready_to_solve"])
        self.assertEqual(len(adapter.handheld.kept_images()), 10)
        self.assertEqual(sum(map(sum, stopped["coverage"]["grid"])), 10)
        self.assertEqual(stopped["guidance"], ["采集已足够，可以求解"])
        status = json.loads((adapter.telemetry_dir / capture.HANDHELD_STATUS_FILE).read_text())
        self.assertEqual(status["kept"], 10)
        self.assertTrue((adapter.telemetry_dir / capture.HANDHELD_PREVIEW_FILE).is_file())

    def test_wrong_size_frames_are_dropped(self):
        small = [np.zeros((360, 640, 3), np.uint8)] * 4
        adapter = adapter_with(self.directory, small + handheld_frames(1))
        _, stopped = run_capture(adapter, 6)
        self.assertEqual(stopped["dropped_wrong_size"], 4)
        self.assertEqual(stopped["kept"], 1)
        self.assertEqual(len(adapter.handheld.kept_images()), 1)

    def test_refused_while_action_active(self):
        adapter = adapter_with(self.directory, [])
        adapter.active_action_id = "a1"
        result = adapter.operator("handheld_capture", action="start")
        self.assertEqual((result["status"], result["reason"]), ("rejected", "action_active"))
        self.assertIsNone(adapter.handheld)
        self.assertEqual(adapter.operator("handheld_solve")["reason"], "action_active")
        with self.assertRaises(ValueError):
            adapter.operator("handheld_capture", action="explode")

    def test_solve_installs_fixed_intrinsics_and_keeps_extrinsic_without_robot_session(self):
        write_calibration(self.path, SIZE)
        original = self.path.read_bytes()
        frames = handheld_frames(12)
        adapter = adapter_with(self.directory, frames, calibration_file=self.path, handheld_target_views=12,
                               handheld_min_novelty=0.02)
        run_capture(adapter, len(frames))
        result = adapter.operator("handheld_solve")
        self.assertEqual(result["status"], "calibrated", result)
        self.assertTrue(result["passed"])
        self.assertLess(result["rms_px"], result["gate_px"])
        self.assertFalse(result["extrinsic_updated"])
        self.assertIn("last_robot_session_unavailable", result["extrinsic_reason"])
        document = json.loads(self.path.read_text())
        fixed = document["cameras"]["fixed"]
        self.assertEqual(fixed["intrinsics_source"], "handheld")
        self.assertLess(abs(fixed["K"][0][0] - 700.0) / 700.0, 0.02)
        old = json.loads(original)
        self.assertEqual(fixed["T_base_camera"], old["cameras"]["fixed"]["T_base_camera"])
        self.assertEqual(document["cameras"]["wrist"]["K"], old["cameras"]["wrist"]["K"])
        self.assertEqual(Path(result["previous_file_backup"]).read_bytes(), original)
        self.assertEqual(adapter.calibration_identity["created_at"], document["created_at"])

    def test_gate_failure_and_too_few_views_keep_old_file(self):
        write_calibration(self.path, SIZE)
        original = self.path.read_bytes()
        frames = handheld_frames(12)
        adapter = adapter_with(self.directory, frames, calibration_file=self.path, handheld_target_views=12,
                               handheld_min_novelty=0.02, max_handheld_intrinsics_rms_px=0.001)
        self.assertEqual(adapter.operator("handheld_solve")["reason"], "no_handheld_capture")
        run_capture(adapter, len(frames))
        result = adapter.operator("handheld_solve")
        self.assertEqual(result["status"], "gate_failed", result)
        self.assertFalse(result["passed"])
        self.assertIn("max_handheld_intrinsics_rms_px", result["reason"])
        self.assertEqual(self.path.read_bytes(), original)
        few = adapter_with(self.directory / "few", handheld_frames(3), calibration_file=self.path)
        run_capture(few, 6)
        self.assertEqual(few.operator("handheld_solve")["status"], "insufficient_views")
        self.assertEqual(self.path.read_bytes(), original)

    def test_extrinsic_resolved_from_last_robot_session(self):
        board = cal.make_board()
        T_bb = cal.board_frame_from_center([0.22, 0.0, 0.0], board, yaw_rad=0.1)
        center = T_bb[:3, :3] @ np.array([0.105, 0.075, 0]) + T_bb[:3, 3]
        pos = np.array([0.22, -0.45, 0.42])
        T_base_fixed = cal.make_transform(cal._look_rotation(center - pos, 0.0), pos)
        session = self.directory / "robot-session"
        (session / "images").mkdir(parents=True)
        (session / "session.json").write_text(json.dumps({"schema": capture.SESSION_SCHEMA, "board": cal.DEFAULT_BOARD_SPEC}))
        cv2.imwrite(str(session / "images" / "0000_fixed.png"),
                    render_board(board, cal.invert_transform(T_base_fixed) @ T_bb, K_FIXED, SIZE))
        cv2.imwrite(str(session / "images" / "0000_wrist.png"), np.full((SIZE[1], SIZE[0], 3), 255, np.uint8))
        (session / "samples.jsonl").write_text(json.dumps({"index": 0, "joints_deg": {}, "settled": True, "images": {
            "fixed": "images/0000_fixed.png", "wrist": "images/0000_wrist.png"}}) + "\n")
        write_calibration(self.path, SIZE)
        capture.robot_session_sidecar(self.path).write_text(json.dumps({"session_dir": str(session),
                                                                        "T_base_board": T_bb.tolist()}))
        frames = handheld_frames(12)
        adapter = adapter_with(self.directory, frames, calibration_file=self.path, handheld_target_views=12,
                               handheld_min_novelty=0.02)
        run_capture(adapter, len(frames))
        result = adapter.operator("handheld_solve")
        self.assertEqual(result["status"], "calibrated", result)
        self.assertTrue(result["extrinsic_updated"], result.get("extrinsic_reason"))
        T = np.asarray(json.loads(self.path.read_text())["cameras"]["fixed"]["T_base_camera"])
        self.assertLess(np.linalg.norm(T[:3, 3] - T_base_fixed[:3, 3]), 0.01)

    def test_selector_resolution_scaling_and_multi_frame_still_window(self):
        board = cal.make_board()
        base_corners = np.stack(np.meshgrid(np.arange(6), np.arange(4)), -1).reshape(-1, 2).astype(float) * 20.0 + 100.0
        ids = np.arange(24, dtype=np.int32)
        det0 = {"corners": base_corners, "ids": ids}
        det_shift_3px = {"corners": base_corners + [3.0, 0.0], "ids": ids}

        # At 640x480, 3.0 px exceeds still_px=1.5 -> not still
        sel_vga = capture.HandheldViewSelector(board=board, still_px=1.5, still_frames=2)
        sel_vga.offer(det0, (640, 480))
        self.assertFalse(sel_vga.offer(det_shift_3px, (640, 480))["still"])

        # At 1920x1080, scale=3.0 -> effective_still_px=4.5 px -> 3.0 px is still!
        sel_hd = capture.HandheldViewSelector(board=board, still_px=1.5, still_frames=2)
        sel_hd.offer(det0, (1920, 1080))
        v_hd = sel_hd.offer(det_shift_3px, (1920, 1080))
        self.assertTrue(v_hd["still"] and v_hd["kept"])

        # With still_frames=3, need 3 consecutive frames (2 still transitions) before kept
        sel_3 = capture.HandheldViewSelector(board=board, still_px=1.5, still_frames=3, min_sharpness=50.0)
        self.assertFalse(sel_3.offer(det0, (640, 480), sharpness=100.0)["still"])
        self.assertFalse(sel_3.offer(det0, (640, 480), sharpness=100.0)["still"])
        self.assertFalse(sel_3.offer(det0, (640, 480), sharpness=10.0)["still"])  # blurry frame resets streak
        self.assertFalse(sel_3.offer(det0, (640, 480), sharpness=100.0)["still"])
        v3 = sel_3.offer(det0, (640, 480), sharpness=100.0)
        self.assertTrue(v3["still"] and v3["kept"])

# === tests/test_so101_handheld_geometry.py ===

import unittest

import numpy as np

from ni_rua.robot.calibration.workflow import capture
from ni_rua.robot.calibration.workflow import charuco as cal


class CanonicalBoardViewTests(unittest.TestCase):
    def setUp(self):
        self.board = cal.make_board(dict(squares_x=12, squares_y=9, square_m=.015,
                                        marker_m=.01125, dictionary="DICT_5X5_100"))
        self.points = cal.board_corner_points(self.board)[:, :2]
        self.H = np.array([[2000., 180., 80.], [40., 2000., 60.], [.3, .6, 1.]])
        self.size = (640, 480)

    def detection(self, ids=None, H=None):
        ids = np.arange(len(self.points)) if ids is None else np.asarray(ids)
        points = np.column_stack([self.points[ids], np.ones(len(ids))]) @ (self.H if H is None else H).T
        return {"ids": ids, "corners": points[:, :2] / points[:, 2:]}

    def test_changing_subset_reproduces_old_false_novelty_but_new_keeps_one(self):
        full, subset = self.detection(), self.detection(np.arange(44))
        old = capture.HandheldViewSelector()  # legacy detected-cloud descriptor
        new = capture.HandheldViewSelector(board=self.board)
        for selector in (old, new):
            selector.offer(full, self.size)
            first = selector.offer(full, self.size)
            second = selector.offer(subset, self.size)
            self.assertEqual(second["motion_px"], 0.)
            if selector is old:
                self.assertTrue(second["kept"])
                self.assertNotEqual(first["cell"], second["cell"])
            else:
                self.assertFalse(second["kept"])
                self.assertEqual(first["cell"], second["cell"])
                self.assertEqual(first["distance"], second["distance"])
                self.assertEqual(selector.kept_count, 1)

    def test_signature_is_invariant_to_corner_subset_and_order(self):
        reference = capture.view_signature(self.detection(), self.size, board=self.board)
        for ids in (np.arange(44), np.arange(44, 88), np.arange(88)[::2], np.arange(88)[::-1]):
            actual = capture.view_signature(self.detection(ids), self.size, board=self.board)
            np.testing.assert_allclose(actual, reference, atol=2e-7)

    def test_true_translation_scale_rotation_and_perspective_remain_novel(self):
        transforms = [np.array([[1, 0, 100], [0, 1, 0], [0, 0, 1]]),
                      np.diag([1.5, 1.5, 1]),
                      np.array([[.866, -.5, 140], [.5, .866, -70], [0, 0, 1]]),
                      np.array([[1, 0, 0], [0, 1, 0], [.002, 0, 1]])]
        for transform in transforms:
            selector = capture.HandheldViewSelector(board=self.board)
            for H in (self.H, transform @ self.H):
                det = self.detection(H=H)
                selector.offer(det, self.size)
                self.assertTrue(selector.offer(det, self.size)["kept"])
            self.assertEqual(selector.kept_count, 2)

    def test_rank_deficient_detection_is_rejected_and_resets_stillness(self):
        selector = capture.HandheldViewSelector(board=self.board)
        bad = self.detection(np.arange(11))  # one physical board row
        for _ in range(3):
            verdict = selector.offer(bad, self.size)
            self.assertFalse(verdict["kept"])
            self.assertEqual(verdict["descriptor_error"], "degenerate_board_projection")
        self.assertFalse(selector.offer(self.detection(), self.size)["still"])

    def test_horizon_crossing_board_is_rejected(self):
        H = self.H.copy()
        H[2] = [1, 0, -.09]
        ids = np.flatnonzero(self.points[:, 0] > .11)
        selector = capture.HandheldViewSelector(board=self.board)
        verdict = selector.offer(self.detection(ids, H), self.size)
        self.assertFalse(verdict["kept"])
        self.assertEqual(verdict["descriptor_error"], "board_projection_crosses_horizon")

    def test_invalid_correspondences_are_rejected(self):
        for ids in ([-1] * 10, [100] * 10, list(range(9)) + [0]):
            selector = capture.HandheldViewSelector(board=self.board)
            verdict = selector.offer({"ids": ids, "corners": np.zeros((10, 2))}, self.size)
            self.assertEqual(verdict["descriptor_error"], "invalid_board_correspondences")
            self.assertFalse(verdict["kept"])


if __name__ == "__main__":
    unittest.main()
