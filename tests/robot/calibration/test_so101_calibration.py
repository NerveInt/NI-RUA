"""SO-101 camera calibration math, schema and offline solve on synthetic data (CPU only, no hardware)."""

NI_RUA_TEST_TIER = "fast"

# === tests/test_so101_calibration.py ===

import importlib.util
import json
import math
from pathlib import Path
from support.paths import REPO_ROOT
import tempfile
import unittest

import numpy as np

try:
    import cv2
    HAVE_ARUCO = hasattr(cv2, "aruco") and hasattr(cv2.aruco, "CharucoDetector")
except ImportError:  # pragma: no cover
    HAVE_ARUCO = False

if HAVE_ARUCO:
    from ni_rua.robot.calibration.workflow import charuco as cal

REPO = REPO_ROOT


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def random_rotation(rng, max_angle):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return cal.rotvec_to_matrix(axis * rng.uniform(0, max_angle))


def perturb(T, rng, translation_sigma_m, rotation_sigma_rad):
    noise = cal.make_transform(cal.rotvec_to_matrix(rng.normal(scale=rotation_sigma_rad, size=3)),
                               rng.normal(scale=translation_sigma_m, size=3))
    return T @ noise


def pose_error(T_a, T_b):
    return (float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3])), cal.rotation_angle(T_a[:3, :3].T @ T_b[:3, :3]))


# Board rendering: board.generateImage with ``M`` px quiet zone and ``SQ`` px per square; pixel centres sit at
# (i + 0.5), so board metric coordinate = (pixel - (M - 0.5)) * square_m / SQ.
SQ, MARGIN = 120, 60


def render_board(board, T_camera_board, K, image_size, *, blur_sigma=0.7):
    src = board.generateImage((7 * SQ + 2 * MARGIN, 5 * SQ + 2 * MARGIN), marginSize=MARGIN, borderBits=1)
    s = board.getSquareLength() / SQ
    S = np.array([[s, 0, -(MARGIN - 0.5) * s], [0, s, -(MARGIN - 0.5) * s], [0, 0, 1]])
    H = K @ np.column_stack([T_camera_board[:3, 0], T_camera_board[:3, 1], T_camera_board[:3, 3]]) @ S
    img = cv2.warpPerspective(src, H, tuple(image_size), flags=cv2.INTER_LINEAR, borderValue=255)
    return cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), blur_sigma), cv2.COLOR_GRAY2BGR)


def board_in_view(board, T_camera_board, K, image_size, margin_px=15):
    outline = cal.board_outline_points(board)
    pts = (T_camera_board @ np.c_[outline, np.ones(4)].T)[:3].T
    if np.any(pts[:, 2] <= 0.05):
        return False
    uv = (K @ (pts / pts[:, 2:3]).T)[:2].T
    w, h = image_size
    return bool(np.all(uv >= margin_px) and np.all(uv[:, 0] < w - margin_px) and np.all(uv[:, 1] < h - margin_px))


@unittest.skipUnless(HAVE_ARUCO, "OpenCV with aruco CharucoDetector required")
class BoardAndDetectionTests(unittest.TestCase):
    def test_board_spec_validation(self):
        board = cal.make_board(cal.DEFAULT_BOARD_SPEC)
        self.assertEqual(tuple(board.getChessboardSize()), (7, 5))
        self.assertAlmostEqual(board.getSquareLength(), 0.030)
        self.assertEqual(cal.board_corner_points(board).shape, (24, 3))
        with self.assertRaises(ValueError):
            cal.validate_board_spec({"marker_m": 0.031})
        with self.assertRaises(ValueError):
            cal.validate_board_spec({"dictionary": "DICT_NOPE"})

    def test_detect_and_board_pose_recover_rendered_pose(self):
        board = cal.make_board()
        K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
        T = cal.make_transform(cal.rotvec_to_matrix([0.3, -0.2, 0.1]), [-0.1, -0.07, 0.33])
        detection = cal.detect_charuco(render_board(board, T, K, (640, 480)), board)
        self.assertIsNotNone(detection)
        self.assertEqual(len(detection["ids"]), 24)
        T_est, rms_px = cal.board_pose(detection["corners"], detection["ids"], K, np.zeros(5), board)
        dt, dr = pose_error(T, T_est)
        self.assertLess(rms_px, 0.6)
        self.assertLess(dt, 0.002)
        self.assertLess(dr, 0.01)

    def test_blank_image_returns_none(self):
        self.assertIsNone(cal.detect_charuco(np.full((480, 640, 3), 255, np.uint8), cal.make_board()))


@unittest.skipUnless(HAVE_ARUCO, "OpenCV with aruco CharucoDetector required")
class IntrinsicsTests(unittest.TestCase):
    def test_recovers_intrinsics_and_distortion_from_noisy_corners(self):
        rng = np.random.default_rng(1)
        board = cal.make_board()
        K = np.array([[610.0, 0, 322], [0, 605.0, 238], [0, 0, 1]])
        dist = np.array([-0.12, 0.05, 0.001, -0.0005, 0.0])
        detections = []
        while len(detections) < 15:
            T = cal.make_transform(random_rotation(rng, 0.6), [rng.uniform(-0.13, -0.08), rng.uniform(-0.1, -0.05),
                                                               rng.uniform(0.25, 0.4)])
            if not board_in_view(board, T, K, (640, 480)):
                continue
            uv = cal.project_board_corners(T, np.arange(24), K, dist, board) + rng.normal(scale=0.15, size=(24, 2))
            detections.append({"corners": uv, "ids": np.arange(24, dtype=np.int32)})
        K_est, dist_est, rms = cal.calibrate_intrinsics(detections, (640, 480), board)
        self.assertLess(rms, 0.3)
        self.assertLess(abs(K_est[0, 0] - K[0, 0]) / K[0, 0], 0.01)
        self.assertLess(abs(K_est[1, 1] - K[1, 1]) / K[1, 1], 0.01)
        self.assertLess(abs(K_est[0, 2] - K[0, 2]), 3.0)
        self.assertLess(abs(K_est[1, 2] - K[1, 2]), 3.0)
        self.assertLess(abs(dist_est[0] - dist[0]), 0.03)
        self.assertEqual(dist_est.shape, (5,))

    def test_too_few_views_refused(self):
        with self.assertRaises(ValueError):
            cal.calibrate_intrinsics([{"corners": np.zeros((24, 2)), "ids": np.arange(24)}] * 3, (640, 480),
                                     cal.make_board())


def synthetic_hand_eye(rng, n, *, pixel_like_noise=True):
    X = cal.make_transform(cal.rotvec_to_matrix([0.05, -0.03, 1.5]), [0.025, -0.04, 0.03])
    T_base_board = cal.make_transform(cal.rotvec_to_matrix([math.pi, 0, 0.2]), [0.2, -0.1, 0.0])
    A, B, B_true = [], [], []
    for _ in range(n):
        R = cal.rotvec_to_matrix([math.pi, 0, 0]) @ random_rotation(rng, 0.6)
        T_bg = cal.make_transform(R, [rng.uniform(0.15, 0.3), rng.uniform(-0.15, 0.05), rng.uniform(0.25, 0.35)])
        T_cb = cal.invert_transform(T_bg @ X) @ T_base_board
        A.append(T_bg)
        B_true.append(T_cb)
        B.append(perturb(T_cb, rng, 0.0005, 0.002) if pixel_like_noise else T_cb)
    return X, T_base_board, A, B


@unittest.skipUnless(HAVE_ARUCO, "OpenCV with aruco CharucoDetector required")
class HandEyeTests(unittest.TestCase):
    def test_opencv_hand_eye_and_charuco_bindings_present(self):
        self.assertTrue(hasattr(cv2, "calibrateHandEye") and hasattr(cv2.aruco, "CharucoDetector"))
        for name in ("TSAI", "PARK", "HORAUD", "DANIILIDIS", "ANDREFF"):
            self.assertTrue(hasattr(cv2, "CALIB_HAND_EYE_" + name), name)

    def test_missing_calibrate_hand_eye_raises_runtime_error(self):
        from types import SimpleNamespace
        from unittest import mock

        _, _, A, B = synthetic_hand_eye(np.random.default_rng(1), 6, pixel_like_noise=False)
        with mock.patch.object(cal, "cv2", SimpleNamespace()):
            with self.assertRaisesRegex(RuntimeError, "opencv_calibrate_hand_eye_unavailable"):
                cal.solve_eye_in_hand(A, B)

    def test_exact_chain_recovered_by_every_method(self):
        X, T_bb, A, B = synthetic_hand_eye(np.random.default_rng(2), 12, pixel_like_noise=False)
        result = cal.solve_eye_in_hand(A, B)
        for name, entry in result["methods"].items():
            if "error" not in entry:
                self.assertLess(entry["translation_rms_m"], 1e-6, name)
        dt, dr = pose_error(X, result["T_gripper_camera"])
        self.assertLess(dt, 1e-6)
        self.assertLess(dr, 1e-6)

    def test_noisy_chain_recovered_within_2mm_and_10mrad(self):
        X, T_bb, A, B = synthetic_hand_eye(np.random.default_rng(3), 20)
        A = [perturb(T, np.random.default_rng(i), 0.0003, 0.001) for i, T in enumerate(A)]  # FK/encoder noise
        result = cal.solve_eye_in_hand(A, B)
        dt, dr = pose_error(X, result["T_gripper_camera"])
        self.assertLess(dt, 0.002)
        self.assertLess(dr, 0.01)
        bt, br = pose_error(T_bb, result["T_base_board"])
        self.assertLess(bt, 0.002)
        self.assertLess(br, 0.01)
        self.assertGreater(result["translation_rms_m"], 0.0)  # honest residual, not zero with noise
        self.assertLess(result["translation_rms_m"], 0.005)
        self.assertEqual(len(result["translation_errors_m"]), 20)
        self.assertIn(result["method"], result["methods"])
        expected = {"TSAI", "PARK", "HORAUD", "DANIILIDIS", "ANDREFF"}
        self.assertEqual(set(result["methods"]), expected)

    def test_picks_lowest_consistency_and_records_failures(self):
        X, _, A, B = synthetic_hand_eye(np.random.default_rng(4), 10)
        result = cal.solve_eye_in_hand(A, B, methods=["PARK", "HORAUD", "DANIILIDIS", "BOGUS"])
        self.assertIn("error", result["methods"]["BOGUS"])
        scores = {k: v["score_m"] for k, v in result["methods"].items() if "score_m" in v}
        self.assertEqual(result["method"], min(scores, key=scores.get))

    def test_degenerate_rotation_rejected(self):
        X, T_bb, _, _ = synthetic_hand_eye(np.random.default_rng(5), 1)
        A = [cal.make_transform(cal.rotvec_to_matrix([0, 0, a]), [0.2, 0, 0.3]) for a in (0.0, 0.3, 0.6, 0.9)]
        B = [cal.invert_transform(T @ X) @ T_bb for T in A]  # rotations about one axis only
        with self.assertRaises(ValueError):
            cal.solve_eye_in_hand(A, B)

    def test_fixed_camera_recovered(self):
        rng = np.random.default_rng(6)
        T_bb = cal.make_transform(cal.rotvec_to_matrix([math.pi, 0, 0.2]), [0.2, -0.1, 0.0])
        T_base_cam = cal.make_transform(cal.rotvec_to_matrix([-2.2, 0.1, 0.3]), [0.25, -0.5, 0.4])
        obs = [perturb(cal.invert_transform(T_base_cam) @ T_bb, rng, 0.0005, 0.002) for _ in range(10)]
        result = cal.solve_fixed_camera(perturb(T_bb, rng, 0.0005, 0.001), obs)
        dt, dr = pose_error(T_base_cam, result["T_base_camera"])
        self.assertLess(dt, 0.003)
        self.assertLess(dr, 0.01)
        self.assertEqual(result["samples"], 10)
        self.assertGreater(result["translation_rms_m"], 0.0)


@unittest.skipUnless(HAVE_ARUCO, "OpenCV with aruco CharucoDetector required")
class JointMappingTests(unittest.TestCase):
    def test_identity_and_sign_offset_round_trip(self):
        deg = {"shoulder_pan.pos": 10.0, "shoulder_lift.pos": -20.0, "elbow_flex.pos": 30.0,
               "wrist_flex.pos": 5.0, "wrist_roll.pos": -90.0, "gripper.pos": 50.0}
        np.testing.assert_allclose(cal.lerobot_deg_to_urdf_rad(deg), np.radians([10, -20, 30, 5, -90]))
        mapping = {"elbow_flex": {"sign": -1, "offset_rad": 0.1}}
        q = cal.lerobot_deg_to_urdf_rad(deg, mapping)
        self.assertAlmostEqual(q[2], -math.radians(30) + 0.1)
        back = cal.urdf_rad_to_lerobot_deg(q, mapping)
        self.assertAlmostEqual(back["elbow_flex"], 30.0)
        with self.assertRaises(ValueError):
            cal.normalize_joint_mapping({"elbow_flex": {"sign": 2}})
        with self.assertRaises(ValueError):
            cal.normalize_joint_mapping({"gripper": {"sign": 1}})


@unittest.skipUnless(HAVE_ARUCO, "OpenCV with aruco CharucoDetector required")
class PoseGenerationTests(unittest.TestCase):
    """Uses an ideal 6-DOF fake arm (q = [x, y, z, rx, ry, rz]) so the test does not depend on SO-101 IK."""

    def setUp(self):
        self.board = cal.make_board()
        self.T_bb = cal.board_frame_from_center([0.25, 0.0, 0.0], self.board)

        def fk(q):
            q = np.asarray(q, dtype=float)
            return cal.make_transform(cal.rotvec_to_matrix(q[3:]), q[:3])

        def ik(T, seed):
            return {"q_rad": np.concatenate([T[:3, 3], cal.matrix_to_rotvec(T[:3, :3])]),
                    "position_error_m": 0.0, "orientation_error_rad": 0.0, "converged": True}

        self.fk, self.ik = fk, ik
        self.limits = [(-1.0, 1.0)] * 3 + [(-4.0, 4.0)] * 3

    def generate(self, **kwargs):
        args = dict(n_poses=12, seed=7, roll_candidates=3)
        args.update(kwargs)
        return cal.generate_calibration_poses(self.T_bb, self.board, self.ik, self.fk, np.zeros(6), self.limits,
                                              **args)

    def test_poses_look_at_board_from_diverse_tilts(self):
        poses = self.generate()
        self.assertEqual(len(poses), 12)
        center = self.T_bb[:3, :3] @ np.array([0.105, 0.075, 0.0]) + self.T_bb[:3, 3]
        tilts = []
        for p in poses:
            T = p["T_base_camera"]
            hit = T[:3, 3] + T[:3, 2] * (-T[2, 3] / T[2, 2])
            hit_board = cal.invert_transform(self.T_bb) @ np.append(hit, 1.0)
            self.assertTrue(0 < hit_board[0] < 0.21 and 0 < hit_board[1] < 0.15)
            self.assertGreaterEqual(T[2, 3], 0.05)
            tilts.append(math.degrees(math.acos(float((T[:3, 3] - center)[2] / np.linalg.norm(T[:3, 3] - center)))))
            np.testing.assert_allclose(p["T_base_link"], self.fk(p["q_rad"]))
        self.assertGreaterEqual(min(tilts), 14.0)
        self.assertLessEqual(max(tilts), 36.0)
        self.assertGreater(max(tilts) - min(tilts), 10.0)

    def test_deterministic_for_seed(self):
        a, b = self.generate(), self.generate()
        for pa, pb in zip(a, b):
            np.testing.assert_allclose(pa["q_rad"], pb["q_rad"])

    def test_joint_margin_and_z_floor_enforced(self):
        rejections = {}
        self.assertEqual(self.generate(z_floor_m=0.5, rejections=rejections), [])
        self.assertGreater(rejections.get("z_floor", 0), 0)
        self.limits = [(-1.0, 1.0)] * 3 + [(-0.2, 0.2)] * 3  # camera must point down: rotation ~ pi
        rejections = {}
        self.assertEqual(self.generate(rejections=rejections), [])
        self.assertGreater(rejections.get("ik_joint_limits", 0), 0)

    def test_fk_points_floor_and_image_projection_checks(self):
        poses = self.generate(fk_points=lambda q: [np.array([0.0, 0.0, 0.01])], rejections={})
        self.assertEqual(poses, [])
        K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
        poses = self.generate(K_guess=K, image_size=(640, 480))
        for p in poses:
            self.assertTrue(board_in_view(self.board, cal.invert_transform(p["T_base_camera"]) @ self.T_bb, K,
                                          (640, 480), margin_px=0))


def valid_cameras():
    K = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
    T = cal.make_transform(cal.rotvec_to_matrix([0.1, 0.2, 0.3]), [0.01, 0.02, 0.03])
    return {
        "wrist": {"K": K, "distortion": [0.1, -0.05, 0.0, 0.0, 0.01], "image_size": [640, 480],
                  "T_link_camera": T, "intrinsics_rms_px": 0.4,
                  "hand_eye_residual": {"translation_m": 0.002, "rotation_rad": 0.004}},
        "fixed": {"K": K, "distortion": [0.0] * 5, "image_size": [640, 480], "T_base_camera": T,
                  "intrinsics_rms_px": 0.3, "extrinsic_residual": {"reprojection_px": 0.8}},
    }


THRESHOLDS = {"max_intrinsics_rms_px": 1.0, "max_hand_eye_translation_m": 0.005, "max_hand_eye_rotation_rad": 0.02}


@unittest.skipUnless(HAVE_ARUCO, "OpenCV with aruco CharucoDetector required")
class CalibrationFileTests(unittest.TestCase):
    def test_round_trip_exact_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera_calibration.json"
            cal.write_calibration_file(path, board=cal.DEFAULT_BOARD_SPEC, cameras=valid_cameras(), samples=18,
                                       thresholds=THRESHOLDS, created_at="2026-11-04T00:00:00Z")
            raw = json.loads(path.read_text())
            self.assertEqual(raw["schema"], "ni_rua.so101.camera_calibration.v1")
            self.assertEqual(raw["reference_frame_id"], "so101_base")
            self.assertEqual(set(raw), {"schema", "reference_frame_id", "created_at", "board", "cameras", "samples"})
            self.assertEqual(raw["board"], cal.DEFAULT_BOARD_SPEC)
            self.assertEqual(set(raw["cameras"]["wrist"]), {
                "K", "distortion", "distortion_model", "image_size", "optical_axes", "arm", "link_name",
                "T_link_camera", "intrinsics_rms_px", "hand_eye_residual"})
            self.assertEqual(raw["cameras"]["wrist"]["link_name"], "gripper_link")
            self.assertEqual(raw["cameras"]["wrist"]["arm"], "right")
            self.assertEqual(raw["cameras"]["wrist"]["distortion_model"], "opencv_plumb_bob")
            self.assertEqual(raw["cameras"]["wrist"]["optical_axes"], "opencv: +x right, +y down, +z forward")
            self.assertIn("T_base_camera", raw["cameras"]["fixed"])
            self.assertEqual(raw["cameras"]["fixed"]["extrinsic_residual"], {"reprojection_px": 0.8})
            self.assertEqual(raw["samples"], 18)
            doc = cal.read_calibration_file(path)
            np.testing.assert_allclose(doc["cameras"]["wrist"]["T_link_camera"],
                                       valid_cameras()["wrist"]["T_link_camera"])
            self.assertEqual(doc["cameras"]["wrist"]["K"].shape, (3, 3))
            with self.assertRaises(FileExistsError):
                cal.write_calibration_file(path, board=cal.DEFAULT_BOARD_SPEC, cameras=valid_cameras(), samples=1)

    def test_invalid_matrices_rejected(self):
        bad = [("T_link_camera", np.eye(3)), ("T_link_camera", np.full((4, 4), np.nan)),
               ("T_link_camera", np.diag([2.0, 1.0, 1.0, 1.0])), ("K", np.eye(4)),
               ("K", [[600, 0, 320], [0, float("inf"), 240], [0, 0, 1]]), ("distortion", [0.1, 0.2])]
        for key, value in bad:
            cameras = valid_cameras()
            cameras["wrist"][key] = value
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    cal.write_calibration_file(Path(directory) / "c.json", board=cal.DEFAULT_BOARD_SPEC,
                                               cameras=cameras, samples=1)
                self.assertFalse((Path(directory) / "c.json").exists())

    def test_reader_rejects_wrong_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c.json"
            path.write_text(json.dumps({"schema": "other", "reference_frame_id": "so101_base"}))
            with self.assertRaises(ValueError):
                cal.read_calibration_file(path)

    def test_residual_gate_refuses_then_force_records_honest_residuals(self):
        cameras = valid_cameras()
        cameras["wrist"]["hand_eye_residual"] = {"translation_m": 0.012, "rotation_rad": 0.004}
        cameras["fixed"]["intrinsics_rms_px"] = 1.7
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c.json"
            with self.assertRaises(cal.CalibrationGateError) as caught:
                cal.write_calibration_file(path, board=cal.DEFAULT_BOARD_SPEC, cameras=cameras, samples=5,
                                           thresholds=THRESHOLDS)
            self.assertFalse(path.exists())
            self.assertEqual(len(caught.exception.failures), 2)
            cal.write_calibration_file(path, board=cal.DEFAULT_BOARD_SPEC, cameras=cameras, samples=5,
                                       thresholds=THRESHOLDS, force=True)
            raw = json.loads(path.read_text())
            self.assertEqual(raw["cameras"]["wrist"]["hand_eye_residual"]["translation_m"], 0.012)
            self.assertEqual(raw["cameras"]["fixed"]["intrinsics_rms_px"], 1.7)


@unittest.skipUnless(HAVE_ARUCO, "OpenCV with aruco CharucoDetector required")
class OfflineSolveEndToEndTests(unittest.TestCase):
    """Renders a synthetic capture session and runs scripts/so101_calibrate.py solve/verify on it."""

    @classmethod
    def setUpClass(cls):
        cls.script = load_script("so101_calibrate")
        cls.board = cal.make_board()
        cls.K_w = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
        cls.K_f = np.array([[700.0, 0, 330], [0, 700.0, 250], [0, 0, 1]])
        cls.X = cal.make_transform(cal.rotvec_to_matrix([0.04, -0.03, 0.1]), [0.02, -0.03, 0.04])
        cls.T_bb = cal.board_frame_from_center([0.22, 0.0, 0.0], cls.board, yaw_rad=0.1)
        center = cls.T_bb[:3, :3] @ np.array([0.105, 0.075, 0]) + cls.T_bb[:3, 3]
        pos = np.array([0.22, -0.45, 0.42])
        cls.T_base_fixed = cal.make_transform(cal._look_rotation(center - pos, 0.0), pos)

    @staticmethod
    def fk_link(q):
        q = np.asarray(q, dtype=float)
        R = (cal.rotvec_to_matrix([math.pi, 0, 0]) @ cal.rotvec_to_matrix([0, 0, q[0]])
             @ cal.rotvec_to_matrix([0, q[1], 0]) @ cal.rotvec_to_matrix([q[2], 0, 0]))
        return cal.make_transform(R, [0.22 + 0.1 * q[3], 0.1 * q[4], 0.36])

    def make_session(self, directory, n=14, seed=11):
        rng = np.random.default_rng(seed)
        d = Path(directory)
        (d / "images").mkdir(parents=True)
        (d / "handheld" / "fixed").mkdir(parents=True)
        (d / "session.json").write_text(json.dumps({
            "schema": self.script.SESSION_SCHEMA, "board": cal.DEFAULT_BOARD_SPEC,
            "joint_mapping": cal.default_joint_mapping()}))
        T_fb = cal.invert_transform(self.T_base_fixed) @ self.T_bb
        fixed_img = render_board(self.board, T_fb, self.K_f, (640, 480))
        lines, index = [], 0
        while index < n:
            q = np.array([rng.uniform(-0.8, 0.8), rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.35),
                          rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)])
            T_cb = cal.invert_transform(self.fk_link(q) @ self.X) @ self.T_bb
            if not board_in_view(self.board, T_cb, self.K_w, (640, 480), margin_px=20):
                continue
            cv2.imwrite(str(d / "images" / f"{index:04d}_wrist.png"), render_board(self.board, T_cb, self.K_w,
                                                                                    (640, 480)))
            cv2.imwrite(str(d / "images" / f"{index:04d}_fixed.png"), fixed_img)
            joints = {name: math.degrees(v) for name, v in zip(cal.ARM_JOINTS, q)}
            lines.append(json.dumps({"index": index, "joints_deg": joints, "settled": True,
                                     "images": {"wrist": f"images/{index:04d}_wrist.png",
                                                "fixed": f"images/{index:04d}_fixed.png"}}))
            index += 1
        (d / "samples.jsonl").write_text("\n".join(lines) + "\n")
        count = 0
        while count < 16:  # hand-held board in front of the fixed camera for its intrinsics
            T = cal.make_transform(random_rotation(rng, 0.6), [rng.uniform(-0.15, -0.06), rng.uniform(-0.1, -0.05),
                                                               rng.uniform(0.3, 0.45)])
            if board_in_view(self.board, T, self.K_f, (640, 480), margin_px=20):
                cv2.imwrite(str(d / "handheld" / "fixed" / f"{count:04d}.png"),
                            render_board(self.board, T, self.K_f, (640, 480)))
                count += 1
        return d

    def test_solve_and_verify_recover_synthetic_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self.make_session(Path(directory) / "s")
            thresholds = dict(THRESHOLDS, max_fixed_reprojection_px=2.0)
            report = self.script.solve_session(session, thresholds=thresholds, fk_link=self.fk_link,
                                               hand_eye_score_lever_m=.27)
            self.assertEqual(report["effective_settings"]["hand_eye_score_lever_m"], .27)
            self.assertEqual(report["hand_eye"]["score_lever_m"], .27)
            self.assertEqual(report["gate_failures"], [])
            doc = cal.read_calibration_file(session / "camera_calibration.json")
            wrist, fixed = doc["cameras"]["wrist"], doc["cameras"]["fixed"]
            dt, dr = pose_error(self.X, wrist["T_link_camera"])
            self.assertLess(dt, 0.002)
            self.assertLess(dr, 0.01)
            dt, dr = pose_error(self.T_base_fixed, fixed["T_base_camera"])
            self.assertLess(dt, 0.004)
            self.assertLess(dr, 0.01)
            self.assertLess(abs(wrist["K"][0, 0] - 600.0), 6.0)
            self.assertLess(abs(fixed["K"][0, 0] - 700.0), 7.0)
            self.assertLess(wrist["intrinsics_rms_px"], 1.0)
            self.assertEqual(doc["samples"], 14)
            saved = json.loads((session / "solve_report.json").read_text())
            self.assertEqual(len(saved["per_sample"]), 14)
            self.assertIn("hand_eye_translation_error_m", saved["per_sample"][0])
            result = self.script.verify_session(session, session / "camera_calibration.json", fk_link=self.fk_link)
            for key in ("wrist_from_wrist_loo", "wrist_from_fixed", "fixed_from_wrist_loo"):
                self.assertLess(result["summary"][key]["rms_px"], 3.0, key)

    def test_solve_refuses_to_save_when_gate_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self.make_session(Path(directory) / "s", n=10, seed=12)
            tight = {"max_hand_eye_translation_m": 1e-7}
            with self.assertRaises(cal.CalibrationGateError):
                self.script.solve_session(session, thresholds=tight, fk_link=self.fk_link)
            self.assertFalse((session / "camera_calibration.json").exists())
            saved = json.loads((session / "solve_report.json").read_text())
            self.assertTrue(saved["gate_failures"])
            self.assertIsNone(saved["calibration_file"])
            report = self.script.solve_session(session, thresholds=tight, force=True, fk_link=self.fk_link)
            self.assertTrue(report["forced"])
            raw = json.loads((session / "camera_calibration.json").read_text())
            self.assertGreater(raw["cameras"]["wrist"]["hand_eye_residual"]["translation_m"], 1e-7)

    def test_unsettled_and_excluded_samples_are_skipped_with_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self.make_session(Path(directory) / "s", n=10, seed=13)
            lines = (session / "samples.jsonl").read_text().splitlines()
            first = json.loads(lines[0])
            first["settled"] = False
            lines[0] = json.dumps(first)
            (session / "samples.jsonl").write_text("\n".join(lines) + "\n")
            report = self.script.solve_session(session, exclude={1}, fk_link=self.fk_link, thresholds={})
            self.assertEqual(report["skipped"], {"0": "not_settled", "1": "excluded"})
            self.assertEqual(report["hand_eye"]["samples"], 8)


@unittest.skipUnless(HAVE_ARUCO, "OpenCV with aruco CharucoDetector required")
class CaptureSafetyTests(unittest.TestCase):
    """Pre-motion checks of the capture step against the packaged SO-101 kinematics (no hardware)."""

    def setUp(self):
        try:
            from ni_rua.robot.model import so101_kinematics as kin
            self.kin, self.model = kin, kin.load_model()
        except Exception as error:  # pragma: no cover - kinematics package not present
            self.skipTest(f"so101_kinematics unavailable: {error}")
        self.script = load_script("so101_calibrate")

    def test_joint_path_below_floor_is_detected(self):
        for fn in (cal.path_floor_violation, self.script.path_floor_violation):
            self.assertIsNone(fn(self.kin, self.model, np.zeros(5), np.radians([0, -80, 55, 75, -80]), 0.05))
            hit = fn(self.kin, self.model, np.zeros(5), np.radians([0, 60, 60, 60, 0]), 0.05)
            self.assertIsNotNone(hit)
            self.assertLess(hit["z_m"], 0.05)

    def test_start_outside_urdf_limits_refuses_motion(self):
        self.script.check_start_within_limits(self.kin, self.model, np.zeros(5))
        with self.assertRaises(RuntimeError):
            self.script.check_start_within_limits(self.kin, self.model, np.radians([0, 0, 170, 0, 0]))


@unittest.skipUnless(HAVE_ARUCO, "OpenCV with aruco CharucoDetector required")
class BoardRenderTests(unittest.TestCase):
    def test_a4_page_scale_and_detectable(self):
        module = load_script("so101_charuco_board")
        page, info = module.render_board_page(px_per_mm=8)
        self.assertEqual(page.shape, (210 * 8, 297 * 8))
        self.assertEqual(info["board_mm"], [210.0, 150.0])
        detection = cal.detect_charuco(page, cal.make_board())
        self.assertIsNotNone(detection)
        self.assertEqual(len(detection["ids"]), 24)
        T, rms = cal.board_pose(detection["corners"], detection["ids"],
                                np.array([[8000.0, 0, page.shape[1] / 2], [0, 8000.0, page.shape[0] / 2], [0, 0, 1]]),
                                np.zeros(5), cal.make_board())
        self.assertLess(rms, 0.5)  # 8 px/mm page imaged as a planar target: exact 30 mm square scale

    def test_board_spec_override_scales_marker(self):
        module = load_script("so101_calibrate")
        spec = module.board_spec_with_overrides(cal.DEFAULT_BOARD_SPEC, 0.0297, None)
        self.assertAlmostEqual(spec["square_m"], 0.0297)
        self.assertAlmostEqual(spec["marker_m"], 0.022 * 0.0297 / 0.030)

    def test_cli_default_renders_png_and_pdf(self):
        # 默认 CLI 参数应在 24 px/mm 下生成可检出 24 个内角点的 A4 横向 PNG 与 PDF
        module = load_script("so101_charuco_board")
        out_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.assertEqual(module.main(["--output-dir", str(out_dir)]), 0)
        png = out_dir / "so101_charuco_board.png"
        pdf = out_dir / "so101_charuco_board.pdf"
        self.assertTrue(png.is_file())
        self.assertTrue(pdf.is_file())
        page = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(page.shape, (210 * 24, 297 * 24))
        detection = cal.detect_charuco(page, cal.make_board())
        self.assertIsNotNone(detection)
        self.assertEqual(len(detection["ids"]), 24)

    def test_cli_board_config_12x9_renders_and_detects(self):
        # 使用 configs/so101_board.json（12x9, 15mm, 11.25mm）应在默认 24 px/mm 下正常渲染并检出 88 个内角点
        module = load_script("so101_charuco_board")
        out_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        config_path = REPO / "configs" / "so101_board.json"
        self.assertEqual(module.main(["--board-config", str(config_path), "--output-dir", str(out_dir)]), 0)
        png = out_dir / "so101_charuco_board.png"
        pdf = out_dir / "so101_charuco_board.pdf"
        self.assertTrue(png.is_file())
        self.assertTrue(pdf.is_file())
        page = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(page.shape, (210 * 24, 297 * 24))
        expected_spec = {
            "type": "charuco",
            "squares_x": 12,
            "squares_y": 9,
            "square_m": 0.015,
            "marker_m": 0.01125,
            "dictionary": "DICT_5X5_100",
        }
        detection = cal.detect_charuco(page, cal.make_board(expected_spec))
        self.assertIsNotNone(detection)
        self.assertEqual(len(detection["ids"]), 88)

    def test_cli_custom_overrides_and_validation(self):
        # 自定义 CLI 覆盖参数（含在 --board-config 基础上覆盖）及非法参数校验
        module = load_script("so101_charuco_board")
        out_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        rc = module.main([
            "--output-dir", str(out_dir),
            "--px-per-mm", "8",
            "--squares-x", "8",
            "--squares-y", "6",
            "--square-mm", "25",
            "--marker-mm", "18",
            "--dictionary", "DICT_4X4_50",
        ])
        self.assertEqual(rc, 0)
        page = cv2.imread(str(out_dir / "so101_charuco_board.png"), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(page.shape, (210 * 8, 297 * 8))
        custom_spec = {
            "type": "charuco",
            "squares_x": 8,
            "squares_y": 6,
            "square_m": 0.025,
            "marker_m": 0.018,
            "dictionary": "DICT_4X4_50",
        }
        detection = cal.detect_charuco(page, cal.make_board(custom_spec))
        self.assertIsNotNone(detection)
        self.assertEqual(len(detection["ids"]), 35)
        # 在 --board-config 基础上覆盖方格和标记尺寸
        override_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        config_path = REPO / "configs" / "so101_board.json"
        self.assertEqual(module.main([
            "--board-config", str(config_path),
            "--square-mm", "16",
            "--marker-mm", "12",
            "--px-per-mm", "8",
            "--output-dir", str(override_dir),
        ]), 0)
        override_page = cv2.imread(str(override_dir / "so101_charuco_board.png"), cv2.IMREAD_GRAYSCALE)
        override_spec = {
            "type": "charuco",
            "squares_x": 12,
            "squares_y": 9,
            "square_m": 0.016,
            "marker_m": 0.012,
            "dictionary": "DICT_5X5_100",
        }
        override_det = cal.detect_charuco(override_page, cal.make_board(override_spec))
        self.assertIsNotNone(override_det)
        self.assertEqual(len(override_det["ids"]), 88)
        # 非法规格：marker_mm >= square_mm、方格像素非整数、超出 A4 留白范围均应报错
        with self.assertRaises(ValueError):
            module.main(["--output-dir", str(out_dir), "--square-mm", "20", "--marker-mm", "20"])
        with self.assertRaises(ValueError):
            module.main(["--output-dir", str(out_dir), "--square-mm", "15.1", "--marker-mm", "11", "--px-per-mm", "8"])
        with self.assertRaises(ValueError):
            module.main(["--output-dir", str(out_dir), "--squares-x", "15", "--squares-y", "10",
                         "--square-mm", "30", "--marker-mm", "22", "--px-per-mm", "8"])

class AutoHandheldCaptureTests(unittest.TestCase):
    def test_keeps_only_still_and_novel_views(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest import mock
        script = load_script("so101_calibrate")
        grid = np.stack(np.meshgrid(np.arange(6), np.arange(4)), -1).reshape(-1, 2).astype(float)
        ids = np.arange(len(grid))

        def view(cx, cy, scale):
            return {"corners": grid * scale + [cx, cy], "ids": ids}
        # still pair at A, moving frames, still pair at A again (not novel), still pair at B (novel)
        sequence = [view(100, 100, 20), view(100, 100.5, 20), view(150, 140, 20), view(300, 90, 25),
                    view(100, 100, 20), view(100, 100.2, 20), view(400, 300, 30), view(400.3, 300, 30)]
        frames = iter(sequence)
        detections = {}

        def fake_frame(_camera):
            detection = next(frames)
            frame = np.zeros((480, 640, 3), np.uint8)
            detections[id(frame)] = detection
            return frame
        args = SimpleNamespace(views=5, timeout_s=5, min_corners=10, still_px=1.5, min_novelty=0.08, camera="fixed")
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(script, "fresh_frame", side_effect=lambda c: fake_frame(c)), \
                mock.patch.object(script.cal, "detect_charuco", side_effect=lambda f, b, min_corners=6: detections[id(f)]):
            try:
                script._auto_handheld(None, None, Path(directory), 0, args)
            except StopIteration:
                pass
            kept = sorted(Path(directory).glob("*.png"))
        self.assertEqual(len(kept), 2)


@unittest.skipUnless(HAVE_ARUCO, "OpenCV with aruco CharucoDetector required")
class HandheldViewSelectorTests(unittest.TestCase):
    def setUp(self):
        from ni_rua.robot.calibration.workflow.capture import HandheldViewSelector
        self.board = cal.make_board()
        self.K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
        self.Selector = HandheldViewSelector

    def detection(self, T):
        return {"corners": cal.project_board_corners(T, np.arange(24), self.K, np.zeros(5), self.board),
                "ids": np.arange(24, dtype=np.int32)}

    def pose(self, x, y, z, tilt=0.0):
        return cal.make_transform(cal.rotvec_to_matrix([tilt, 0, 0]), [x, y, z])

    def test_still_novel_and_coverage(self):
        selector = self.Selector(board=self.board, target_views=10)
        centre = self.detection(self.pose(-0.09, -0.06, 0.45))
        self.assertFalse(selector.offer(None, (640, 480))["kept"])
        self.assertEqual(selector.guidance()[0], "未检测到标定板，请把标定板完整放到固定相机前")
        first = selector.offer(centre, (640, 480))
        self.assertFalse(first["still"])  # no previous frame to compare
        moved = dict(centre, corners=centre["corners"] + [5.0, 0.0])
        self.assertFalse(selector.offer(moved, (640, 480))["still"])
        self.assertIn("板子拿稳", selector.guidance())
        kept = selector.offer(dict(moved, corners=moved["corners"] + [0.3, 0.0]), (640, 480))
        self.assertTrue(kept["still"] and kept["novel"] and kept["kept"])
        self.assertEqual(kept["cell"], [1, 1])
        again = selector.offer(dict(moved, corners=moved["corners"] + [0.4, 0.0]), (640, 480))
        self.assertTrue(again["still"])
        self.assertFalse(again["novel"])
        self.assertIn("这个位置已采过，换个位置", selector.guidance())
        coverage = selector.coverage()
        self.assertEqual(coverage["grid"][1][1], 1)
        self.assertEqual(coverage["kept"], 1)
        self.assertEqual(sum(coverage["distance"].values()), 1)
        hints = " ".join(selector.guidance())
        self.assertIn("还缺：画面左上角", hints)
        self.assertIn("倾斜一点", hints)

    def test_tilt_and_distance_buckets(self):
        selector = self.Selector(board=self.board, min_novelty=0.02)
        frontal = self.detection(self.pose(-0.09, -0.06, 0.45))
        tilted = self.detection(self.pose(-0.09, -0.06, 0.45, tilt=0.6))
        near = self.detection(self.pose(-0.09, -0.06, 0.22))
        for det in (frontal, tilted, near):
            selector.previous = None
            selector.offer(det, (640, 480))
            verdict = selector.offer(det, (640, 480))
            self.assertTrue(verdict["kept"], verdict)
        tilt = selector.coverage()["tilt"]
        self.assertEqual(tilt["frontal"], 2)  # the near view is frontal too
        self.assertEqual(tilt["small"] + tilt["large"], 1)
        self.assertEqual(selector.coverage()["distance"]["near"], 1)
        self.assertNotIn("再靠近一些", selector.guidance())

    def test_complete_guidance(self):
        selector = self.Selector(target_views=1)
        det = self.detection(self.pose(-0.09, -0.06, 0.45))
        selector.offer(det, (640, 480))
        selector.offer(det, (640, 480))
        self.assertTrue(selector.complete)
        self.assertEqual(selector.guidance(), ["采集已足够，可以求解"])


class JointMappingCommentKeyTests(unittest.TestCase):
    def test_comment_keys_are_ignored_but_unknown_joints_rejected(self):
        mapping = cal.normalize_joint_mapping({"_comment": "x", "elbow_flex": {"sign": 1, "offset_rad": 0.1}})
        self.assertEqual(mapping["elbow_flex"]["offset_rad"], 0.1)
        with self.assertRaises(ValueError):
            cal.normalize_joint_mapping({"elbow": {"sign": 1}})

# === tests/test_so101_calibration_configuration.py ===

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ni_rua.robot.calibration.workflow import capture, touch_off
from ni_rua.robot.calibration.workflow.defaults import calibration_default
from ni_rua.robot.backends.so101.deployment import FIELDS, validate_deployment
from ni_rua.robot.calibration.workflow import charuco as cal
from robot.so101.test_so101_camera_and_recalibration import adapter_with




class CalibrationConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.document = json.loads((REPO_ROOT / "configs/so101.example.json").read_text())

    def test_defaults_template_and_legacy_resolution_have_one_value(self):
        optional = {key: spec["default"] for key, spec in FIELDS["calibration"].items() if "default" in spec}
        for key, value in optional.items():
            self.assertEqual(value, calibration_default(key), key)
            self.assertEqual(self.document["calibration"][key], value, key)
            del self.document["calibration"][key]
        before = deepcopy(self.document)
        result = validate_deployment(self.document)["calibration"]
        self.assertEqual(self.document, before)
        self.assertEqual({key: result[key] for key in optional}, optional)
        # Presence in the helper defaults file must not relax an existing required field.
        del self.document["calibration"]["handheld_target_views"]
        with self.assertRaisesRegex(ValueError, "handheld_target_views"):
            validate_deployment(self.document)

    def test_motion_video_defaults_share_data_source_without_changing_values(self):
        for key in ("video_crf", "video_queue_mb"):
            self.assertEqual(FIELDS["camera"][key]["default"], calibration_default(key))
            self.assertEqual(self.document["camera"][key], calibration_default(key))
        self.assertEqual(FIELDS["operator"]["default_speed_scale_cap"]["default"], calibration_default("default_speed_scale_cap"))
        self.assertEqual(self.document["operator"]["default_speed_scale_cap"], 0.9)
        legacy = deepcopy(self.document)
        del legacy["operator"]["default_speed_scale_cap"]
        self.assertEqual(validate_deployment(legacy)["operator"]["default_speed_scale_cap"], 0.9)
        invalid = deepcopy(self.document)
        invalid["operator"]["min_speed_scale_cap"] = 0.5
        invalid["operator"]["default_speed_scale_cap"] = 0.4
        with self.assertRaisesRegex(ValueError, "default_speed_scale_cap"):
            validate_deployment(invalid)

    def test_packaged_board_and_explicit_hand_eye_score_keep_independent_residuals(self):
        self.assertEqual(cal.DEFAULT_BOARD_SPEC, calibration_default("default_board"))
        board = calibration_default("default_board")
        board["square_m"] *= 2
        self.assertNotEqual(board, calibration_default("default_board"))
        X, _, A, B = synthetic_hand_eye(np.random.default_rng(14), 10)
        baseline = cal.hand_eye_consistency(A, X, B, score_lever_m=0.)
        weighted = cal.hand_eye_consistency(A, X, B, score_lever_m=.27)
        self.assertEqual(baseline["translation_rms_m"], weighted["translation_rms_m"])
        self.assertEqual(baseline["rotation_rms_rad"], weighted["rotation_rms_rad"])
        self.assertAlmostEqual(weighted["score_m"] - baseline["score_m"], .27 * baseline["rotation_rms_rad"])
        result = cal.solve_eye_in_hand(A, B, score_lever_m=.27)
        self.assertEqual(result["score_lever_m"], .27)
        for row in result["methods"].values():
            if "score_m" in row:
                self.assertAlmostEqual(row["score_m"], row["translation_rms_m"] + .27 * row["rotation_rms_rad"])

    def test_one_click_and_workflow_pass_explicit_hand_eye_ranking_control(self):
        from ni_rua.robot.calibration.workflow import hand_eye_solve as solve
        adapter, _, _ = adapter_with(self.directory)
        settings = adapter.deployment["calibration"]
        settings["hand_eye_score_lever_m"] = .27
        with patch.object(solve, "solve_session", side_effect=RuntimeError("end fixture")) as call:
            with self.assertRaisesRegex(RuntimeError, "end fixture"):
                capture.solve_for_adapter(adapter, self.directory, self.directory / "candidate", settings)
            self.assertEqual(call.call_args.kwargs["hand_eye_score_lever_m"], .27)
        adapter.deployment["calibration_bundle_dir"] = str(self.directory / "bundle")
        flow = adapter.workflow
        flow.begin("first_run")
        with patch.object(capture, "capture_session", return_value={"status": "completed", "kept": 20,
                          "pose_count": 20, "park": None, "physics_advanced": False}), \
                patch.object(solve, "solve_session", side_effect=RuntimeError("end fixture")) as call:
            flow._s2(lambda event: None)
            self.assertEqual(call.call_args.kwargs["hand_eye_score_lever_m"], .27)

    def test_explicit_missing_board_file_fails_but_unused_inline_path_does_not(self):
        document = deepcopy(self.document)
        document.pop("board", None)
        document["board_file"] = str(self.directory / "does-not-exist.json")
        with self.assertRaisesRegex(ValueError, "board_file: not found"):
            validate_deployment(document)
        document["board"] = dict(cal.DEFAULT_BOARD_SPEC, thickness_m=.003)
        self.assertEqual(validate_deployment(document)["board"]["thickness_m"], .003)

    def test_xy_configuration_bounds_match_existing_request_contract(self):
        for key, upper in (("xy_max_error_m", .1), ("xy_max_reprojection_error_px", 100),
                           ("xy_min_parallax_deg", 90)):
            for value in (0, upper):
                document = deepcopy(self.document)
                document["calibration"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_deployment(document)

    def test_explicit_controls_and_nullable_defaults_are_preserved_and_validated(self):
        explicit = dict(touch_orientation_weight=.12, capture_max_drift_rad=.01, z_floor_m=-.04,
                        handheld_near_fraction=.7, handheld_far_fraction=.2, handheld_tilt_small=1.1,
                        handheld_tilt_large=1.3, handheld_still_frames=3, handheld_min_sharpness=50.)
        self.document["calibration"].update(explicit)
        result = validate_deployment(self.document)["calibration"]
        self.assertEqual({key: result[key] for key in explicit}, explicit)
        for change in (dict(capture_max_drift_rad=0), dict(touch_orientation_weight=-1),
                       dict(handheld_far_fraction=.8), dict(handheld_tilt_small=1.4),
                       dict(handheld_still_frames=1), dict(handheld_min_sharpness=-1),
                       dict(z_floor_m=float("nan")), dict(joint_solver_ftol=0)):
            bad = deepcopy(self.document)
            bad["calibration"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_deployment(bad)

    def test_capture_drift_override_is_exact_and_archived(self):
        for limit, kept in ((None, 1), (.01, 0), (.06, 1)):
            with self.subTest(limit=limit), tempfile.TemporaryDirectory() as directory:
                adapter, driver, _ = adapter_with(Path(directory))
                settings = dict(adapter.deployment["calibration"], capture_max_drift_rad=limit,
                                arrival_tolerance_rad=.05)
                original = adapter.capture_frames
                def drift(t):
                    frames = original(t)
                    driver.q[0] += .04
                    return frames
                out = Path(directory) / "session"
                with patch.object(adapter, "capture_frames", side_effect=drift), \
                        patch.object(capture, "count_corners", return_value=100):
                    report = capture.capture_session(adapter, out, settings=settings,
                                                    poses=[adapter.deployment["home"]["q_rad"]])
                effective = .1 if limit is None else limit
                self.assertEqual(report["kept"], kept, report)
                self.assertEqual(report["poses"][0]["effective_capture_max_drift_rad"], effective)
                session = json.loads((out / "session.json").read_text())
                self.assertEqual(session["settings"]["capture_max_drift_rad"], limit)
                self.assertEqual(session["effective_capture_max_drift_rad"], effective)
                if kept:
                    row = json.loads((out / "samples.jsonl").read_text().splitlines()[0])
                    self.assertEqual(row["effective_capture_max_drift_rad"], effective)

    def test_touch_orientation_weight_reaches_ik_and_report(self):
        adapter, _, _ = adapter_with(self.directory)
        settings = dict(adapter.deployment["calibration"], touch_orientation_weight=.12)
        with patch.object(adapter.kin, "ik", return_value={"position_error_m": 1., "q_rad": [0.] * 5}) as ik:
            report = touch_off.touch_points(adapter, [[.2, 0]], settings)
        self.assertEqual(ik.call_args.kwargs["orientation_weight"], .12)
        self.assertEqual(report["settings"]["touch_orientation_weight"], .12)

    def test_handheld_controls_reach_selector_and_effective_archive(self):
        adapter, _, _ = adapter_with(self.directory)
        explicit = dict(handheld_near_fraction=.7, handheld_far_fraction=.2, handheld_tilt_small=1.1,
                        handheld_tilt_large=1.3, handheld_still_frames=3, handheld_min_sharpness=50.)
        adapter.deployment["calibration"].update(explicit)
        out = self.directory / "handheld"
        sampler = capture.HandheldCapture(adapter, out)
        for key, value in explicit.items():
            self.assertEqual(getattr(sampler.selector, key.removeprefix("handheld_")), value)
            self.assertEqual(sampler.snapshot()["effective_settings"][key], value)
            self.assertEqual(json.loads((out / "session.json").read_text())["settings"][key], value)

    def test_positive_sharpness_gate_uses_measured_pixels_and_unknown_does_not_pass(self):
        board = cal.make_board()
        detection = {"ids": np.arange(len(cal.board_corner_points(board))),
                     "corners": cal.board_corner_points(board)[:, :2] * 1000 + [100, 100]}
        selector = capture.HandheldViewSelector(board=board, min_sharpness=1)
        selector.offer(detection, (640, 480))
        verdict = selector.offer(detection, (640, 480))
        self.assertFalse(verdict["kept"])
        self.assertFalse(verdict["sharpness_ok"])
        verdict = selector.offer(detection, (640, 480), sharpness=float("nan"))
        self.assertFalse(verdict["kept"])
        self.assertIsNone(verdict["sharpness"])
        for threshold, kept in ((1., 0), (0., 1)):
            with self.subTest(threshold=threshold), tempfile.TemporaryDirectory() as directory:
                adapter, _, _ = adapter_with(Path(directory))
                adapter.deployment["calibration"]["handheld_min_sharpness"] = threshold
                sampler = capture.HandheldCapture(adapter, Path(directory) / "handheld")
                size = sampler.size
                black = np.zeros((size[1], size[0], 3), np.uint8)
                with patch.object(adapter.cameras["fixed"], "read_after", side_effect=[
                        (black, 1.), (black, 2.), RuntimeError("end of fixture")]), \
                        patch.object(cal, "detect_charuco", return_value=detection):
                    sampler._run()
                self.assertEqual(sampler.selector.kept_count, kept)
                if threshold:
                    self.assertEqual(sampler.selector.last["sharpness"], 0.)

    def test_cli_flags_are_validated_before_adapter_creation(self):
        spec = importlib.util.spec_from_file_location("touch_cli", REPO_ROOT / "scripts/so101_touch_table.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        path = self.directory / "deployment.json"
        path.write_text(json.dumps(self.document))
        base = ["--deployment", str(path), "--out", str(self.directory / "result.json")]
        for flags in (["--step-m", "0"], ["--points", "0.2,0.0,0.1"], ["--min-z-m", "0.08"]):
            with self.subTest(flags=flags), patch.object(module, "SO101Adapter") as adapter, self.assertRaises(ValueError):
                module.main(base + flags)
            adapter.assert_not_called()
        with patch.object(module, "SO101Adapter", side_effect=RuntimeError("validated before hardware")) as adapter:
            with self.assertRaisesRegex(RuntimeError, "validated before hardware"):
                module.main(base + ["--step-m", "0.001", "--points", "0.2,0.01"])
        effective = adapter.call_args.args[0]["calibration"]
        self.assertEqual(effective["touch_step_m"], .001)
        self.assertEqual(effective["touch_points_xy_m"], [[.2, .01]])
        self.assertEqual(json.loads(path.read_text()), self.document)

# === tests/test_so101_calibration_video.py ===

import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from ni_rua.robot.calibration.workflow.video import CalibrationVideoRecorder
from ni_rua.robot.backends.so101.driver import CameraSource, FakeClock, OpenCVCameraSource


class SnapshotCamera(CameraSource):
    def __init__(self):
        self.frame = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
        self.stamp = 1.0
        self.fail = False

    def latest(self):
        if self.fail:
            raise RuntimeError("camera unavailable")
        return None if self.frame is None else (self.frame, self.stamp)


class CalibrationVideoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.camera = SnapshotCamera()
        self.adapter = SimpleNamespace(cameras={"fixed": self.camera}, clock=FakeClock(),
            deployment={"joint_mapping": {"wrist_flex": {"sign": 1, "offset_rad": .0123456789}},
                        "cameras": {"fixed": {"width": 16, "height": 12}}},
            camera_calibration={"cameras": {"fixed": {"K": np.eye(3)}}},
            last_sample={"t": .5, "q_rad": [0., .1, .2, .3, .4], "gripper_rad": .02},
            read=Mock(side_effect=AssertionError("must not read motors")),
            capture_frames=Mock(side_effect=AssertionError("must not synchronously capture")))

    def recorder(self, **settings):
        return CalibrationVideoRecorder(self.adapter, self.root / "frames", **settings)

    def test_full_resolution_pixels_timing_and_frozen_basis_are_preserved(self):
        recorder = self.recorder()
        frozen = self.camera.frame.copy()
        recorder.capture_once()
        self.adapter.camera_calibration["cameras"]["fixed"]["K"][0, 0] = 999
        self.adapter.deployment["joint_mapping"]["wrist_flex"]["offset_rad"] = .5
        self.adapter.last_sample["q_rad"][0] = 999
        self.camera.frame[:] = 0
        result = recorder.stop()
        self.assertEqual(result["status"], "completed")
        manifest = json.loads(Path(result["manifest_path"]).read_text())
        frame = manifest["cameras"]["fixed"]["frames"][0]
        self.assertEqual(frame["host_read_interval_s"], [1., None])
        self.assertIsNone(frame["exposure_monotonic_s"])
        self.assertEqual(frame["joint_sample"]["t"], .5)
        self.assertEqual(frame["joint_sample"]["q_rad"][0], 0.)
        self.assertEqual(manifest["camera_calibration"]["cameras"]["fixed"]["K"][0][0], 1.)
        self.assertEqual(manifest["joint_mapping"]["wrist_flex"]["offset_rad"], .0123456789)
        image_path = self.root / "frames" / frame["path"]
        self.assertEqual(frame["source_sha256"], hashlib.sha256(image_path.read_bytes()).hexdigest())
        np.testing.assert_array_equal(np.asarray(Image.open(image_path)), frozen)
        self.adapter.read.assert_not_called()
        self.adapter.capture_frames.assert_not_called()

    def test_duplicates_regressions_invalid_and_errors_are_not_fabricated_frames(self):
        recorder = self.recorder()
        recorder.capture_once()
        recorder.capture_once()
        self.camera.stamp = .9
        recorder.capture_once()
        self.camera.stamp = float("nan")
        recorder.capture_once()
        self.camera.stamp, self.camera.frame = 2., np.zeros((10, 10), np.uint8)
        recorder.capture_once()
        self.camera.fail = True
        recorder.capture_once()
        self.camera.fail, self.camera.frame = False, None
        recorder.capture_once()
        result = recorder.stop()["cameras"]["fixed"]
        self.assertEqual({key: result[key] for key in ("frames", "duplicates", "nonmonotonic", "invalid", "errors", "missing")},
                         dict(frames=1, duplicates=1, nonmonotonic=1, invalid=2, errors=1, missing=1))

    def test_visual_availability_is_separate_from_completed_lifecycle(self):
        self.camera.frame = None
        recorder = self.recorder()
        recorder.capture_once()
        empty = recorder.stop()
        self.assertEqual(empty["status"], "completed")
        self.assertEqual(empty["visual_evidence_status"], "unavailable")
        self.assertEqual(empty["coverage"]["cameras_without_frames"], ["fixed"])
        self.camera.frame = np.zeros((12, 16, 3), np.uint8)
        recorder = CalibrationVideoRecorder(self.adapter, self.root / "valid")
        recorder.capture_once()
        self.camera.stamp = 2.
        recorder.capture_once()
        complete = recorder.stop()
        self.assertEqual(complete["visual_evidence_status"], "available")
        self.assertEqual(complete["coverage"]["shared_host_time_interval_s"], [1., 2.])
        self.assertEqual(complete["coverage"]["scope"], "archived_rgb_availability_not_contact_or_motion_validation")

    def test_configured_missing_camera_cannot_appear_as_complete_visual_evidence(self):
        self.adapter.deployment["cameras"]["wrist"] = dict(width=16, height=12)
        recorder = self.recorder()
        recorder.capture_once()
        self.camera.stamp = 2.
        recorder.capture_once()
        result = recorder.stop()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["visual_evidence_status"], "incomplete")
        self.assertEqual(result["coverage"]["frames_per_camera"], dict(fixed=2, wrist=0))
        self.assertEqual(result["coverage"]["cameras_missing_source"], ["wrist"])
        self.assertIsNone(result["coverage"]["shared_host_time_interval_s"])

    def test_sampler_timestamp_precedes_png_encoding_and_write_latency(self):
        recorder = self.recorder()
        original = Image.Image.save
        before = self.adapter.clock.now()
        def slow_encoding(image, *args, **kwargs):
            self.adapter.clock.sleep(.75)
            return original(image, *args, **kwargs)
        with patch.object(Image.Image, "save", slow_encoding):
            recorder.capture_once()
        manifest = json.loads(Path(recorder.stop()["manifest_path"]).read_text())
        frame = manifest["cameras"]["fixed"]["frames"][0]
        self.assertEqual(frame["sampler_monotonic_s"], before)
        self.assertEqual(self.adapter.clock.now(), before + .75)

    def test_per_camera_frame_cap_and_duration_are_explicit_truncation(self):
        recorder = self.recorder(max_frames=2)
        recorder.capture_once()
        self.camera.stamp = 2.
        recorder.capture_once()
        self.camera.stamp = 3.
        recorder.capture_once()
        result = recorder.stop()
        self.assertEqual((result["status"], result["termination_reason"], result["cameras"]["fixed"]["frames"]),
                         ("truncated", "max_frames", 2))
        recorder = CalibrationVideoRecorder(self.adapter, self.root / "duration", max_duration_s=.01)
        recorder.capture_once()
        recorder._wall_start -= 1.
        recorder.capture_once()
        result = recorder.stop()
        self.assertEqual((result["status"], result["termination_reason"]), ("truncated", "max_duration_s"))

    def test_background_writer_isolated_from_caller_and_shutdown_pending_is_honest(self):
        recorder = self.recorder()
        entered, release = threading.Event(), threading.Event()
        original = Image.Image.save
        def blocked(image, *args, **kwargs):
            entered.set()
            release.wait(5)
            return original(image, *args, **kwargs)
        with patch.object(Image.Image, "save", blocked):
            recorder.start()
            self.assertTrue(entered.wait(1))
            result = recorder.stop()
            self.assertEqual(result["status"], "unknown")
            self.assertEqual(result["termination_reason"], "writer_shutdown_pending")
            release.set()
            recorder.thread.join(timeout=2)
            self.assertFalse(recorder.thread.is_alive())
        self.assertEqual(recorder.stop()["status"], "completed")

    def test_native_camera_timed_snapshot_retains_host_read_interval(self):
        clock = FakeClock()
        source = OpenCVCameraSource("fixed", dict(width=16, height=12, rotate_deg=0), clock)
        source.running = True
        def read():
            clock.sleep(.025)
            source.running = False
            return True, np.zeros((12, 16, 3), np.uint8)
        source.capture = SimpleNamespace(read=read)
        before = clock.now()
        source._grab()
        legacy = source.latest()
        timed = source.latest_timed()
        self.assertEqual(timed["capture_monotonic_s"], legacy[1])
        self.assertEqual(timed["host_read_interval_s"], [before, before + .025])
        self.assertIsNone(timed["exposure_monotonic_s"])

    def test_bad_parameters_and_existing_directory_rejected_before_capture(self):
        for settings in (dict(fps=0), dict(fps=float("nan")), dict(max_frames=True),
                         dict(max_frames=0), dict(max_duration_s=-1)):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.recorder(**settings)
        (self.root / "frames").mkdir()
        with self.assertRaises(FileExistsError):
            self.recorder().start()

# === tests/test_so101_calibration_workflow.py ===

import json
from pathlib import Path
import tempfile
import unittest

from ni_rua.robot.calibration.workflow import state_machine as wf
from ni_rua.robot.backends.so101.deployment import (FIELDS, calibration_bundle_dir, load_deployment,
                                                     validate_deployment, write_joint_offsets)

EXAMPLE = REPO_ROOT / "configs" / "so101.example.json"


class WorkflowDefinitionTests(unittest.TestCase):
    def test_mode_to_step_mapping(self):
        self.assertEqual(wf.plan_for("first_run"), ["S0", "S1", "S2", "S3", "S4", "S5", "S6"])
        cases = {"fixed_moved": ["S4", "S5", "S6"], "fixed_replaced": ["S3", "S4", "S5", "S6"],
                 "wrist_bumped": ["S2", "S4", "S5", "S6"], "table_or_base_moved": ["S1", "S2", "S4", "S5", "S6"],
                 "board_changed": ["S2", "S4", "S5", "S6"], "everything": list(wf.STEP_IDS)}
        for change, steps in cases.items():
            with self.subTest(change):
                self.assertEqual(wf.plan_for("recalibrate", [change]), steps)
        self.assertEqual(wf.plan_for("recalibrate", ["fixed_moved", "fixed_replaced", "wrist_bumped"]),
                         ["S2", "S3", "S4", "S5", "S6"])
        for mode, changes, message in (("recalibrate", [], "requires_changes"), ("recalibrate", ["x"], "change_unknown"),
                                       ("other", [], "mode_unknown")):
            with self.assertRaisesRegex(ValueError, message):
                wf.plan_for(mode, changes)

    def test_dependency_staleness(self):
        self.assertEqual(wf.dependents("S1"), ["S5", "S6"])
        self.assertEqual(wf.dependents("S2"), ["S4", "S5", "S6"])
        self.assertEqual(wf.dependents("S3"), ["S4", "S5", "S6"])
        self.assertEqual(wf.dependents("S0"), [])
        self.assertEqual(wf.dependents("S6"), [])
        statuses = {"S1": "passed", "S2": "passed", "S3": "not_started", "S4": "failed", "S5": "passed", "S6": "not_started"}
        out = wf.mark_stale(statuses, "S2")
        self.assertEqual(out, {"S1": "passed", "S2": "passed", "S3": "not_started", "S4": "stale", "S5": "stale",
                               "S6": "not_started"})
        self.assertEqual(statuses["S4"], "failed")  # input not mutated

    def test_every_threshold_is_a_deployment_field_and_evaluate_is_strict(self):
        settings = load_deployment(EXAMPLE)["calibration"]
        for step in wf.STEPS:
            self.assertTrue(step["title"] and step["purpose"] and step["operator"] and step["machine"])
            for criterion in step["criteria"]:
                if criterion["threshold"] is not None:
                    self.assertIn(criterion["threshold"], FIELDS["calibration"], criterion)
            for check in step["readiness"]:
                self.assertIn(check, wf.READINESS)
        ok = wf.evaluate("S5", {"joint_reprojection_px": 0.9, "board_height_error_m": 0.001, "max_touch_residual_m": 0.004,
                                "fixed_chain_reprojection_px": 3.0, "max_joint_offset_rad": 0.06}, settings)
        self.assertTrue(ok["passed"], ok)
        bad = wf.evaluate("S5", {"joint_reprojection_px": 0.9, "board_height_error_m": 0.01, "max_touch_residual_m": 0.004,
                                 "fixed_chain_reprojection_px": 3.0}, settings)
        self.assertFalse(bad["passed"])
        failed = {row["id"] for row in bad["criteria"] if not row["passed"]}
        self.assertEqual(failed, {"board_height_error_m", "max_joint_offset_rad"})  # missing measure fails
        self.assertTrue(wf.evaluate("S1", {"contacts": 4, "planned_points": 4}, settings)["passed"])
        self.assertFalse(wf.evaluate("S1", {"contacts": 3, "planned_points": 4}, settings)["passed"])
        described = wf.describe(settings)
        s2 = next(step for step in described["steps"] if step["id"] == "S2")
        self.assertEqual(s2["criteria"][0]["threshold_value"], settings["min_samples"])
        steps = {step["id"]: step for step in described["steps"]}
        # Run-time comparison target: no value before the run, but a label so the rule is never blank.
        s6_contacts = next(c for c in steps["S6"]["criteria"] if c["id"] == "contacts")
        self.assertIsNone(s6_contacts["threshold_value"])
        self.assertEqual(s6_contacts["threshold_label"], "计划点数")
        # Constant comparison target: value filled in before the run.
        s4_samples = next(c for c in steps["S4"]["criteria"] if c["id"] == "fixed_samples")
        self.assertEqual(s4_samples["threshold_value"], 1)
        self.assertEqual(len(described["changes"]), 6)
        json.dumps(described, ensure_ascii=False)


class DeploymentWorkflowFieldsTests(unittest.TestCase):
    def test_new_fields_validated(self):
        document = json.loads(EXAMPLE.read_text())
        for change, message in ((lambda d: d["calibration"].update(touch_points_xy_m=[[0.2]]), "touch_points_xy_m"),
                                (lambda d: d["calibration"].update(touch_points_xy_m=[]), "touch_points_xy_m"),
                                (lambda d: d["calibration"].update(verify_points_xy_m=[[0.2, 9]]), "verify_points_xy_m"),
                                (lambda d: d["calibration"].update(touch_min_z_m=0.07), "touch_min_z_m"),
                                (lambda d: d["calibration"].pop("max_joint_offset_rad"), "missing=\\['max_joint_offset_rad'\\]")):
            candidate = json.loads(json.dumps(document))
            change(candidate)
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                validate_deployment(candidate, base=EXAMPLE.parent)

    def test_bundle_dir_defaults_next_to_camera_file(self):
        deployment = load_deployment(EXAMPLE)
        self.assertEqual(calibration_bundle_dir(deployment), Path(deployment["camera_calibration_file"]).parent / "calibration")
        deployment["calibration_bundle_dir"] = "/tmp/x"
        self.assertEqual(calibration_bundle_dir(deployment), Path("/tmp/x"))


class JointOffsetWriterTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.directory, True)
        self.path = self.directory / "so101.local.json"
        self.path.write_text(EXAMPLE.read_text())
        self.original = self.path.read_text()

    def test_writes_offsets_with_backup_keeping_comments(self):
        result = write_joint_offsets(self.path, {"elbow_flex": -0.0634, "wrist_flex": -0.0436}, stamp="20261109T000000Z")
        self.assertEqual(Path(result["backup"]).read_text(), self.original)
        document = json.loads(self.path.read_text())
        self.assertEqual(document["joint_mapping"]["elbow_flex"]["offset_rad"], -0.0634)
        self.assertIn("_comment", document["joint_mapping"])
        self.assertEqual(document["joint_mapping"]["shoulder_lift"], json.loads(self.original)["joint_mapping"]["shoulder_lift"])
        self.assertEqual(result["previous_offsets_rad"]["elbow_flex"], 0.0)
        self.assertEqual(load_deployment(self.path)["joint_mapping"]["wrist_flex"]["offset_rad"], -0.0436)
        self.assertFalse(list(self.directory.glob(".*.tmp")))

    def test_invalid_offsets_never_touch_the_file(self):
        for offsets, message in (({"gripper": 0.1}, "unknown_joints"), ({"elbow_flex": float("nan")}, "invalid"),
                                 ({"elbow_flex": 4.0}, "out_of_range")):
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                write_joint_offsets(self.path, offsets)
            self.assertEqual(self.path.read_text(), self.original)
        self.assertFalse(list(self.directory.glob("*.bak-*")))


if __name__ == "__main__":
    unittest.main()
