"""External perception with a distorted, non-X5 camera (SO-101 'fixed'): lens rectification and configured camera role."""
NI_RUA_TEST_TIER = "component"

import base64
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from ni_rua.core.config import Config
from ni_rua.core.contracts import canonical
from ni_rua.core.storage.store import Store
from ni_rua.perception.external import obs3d
from ni_rua.perception.external.object_table import default_camera, effective_support_z_m, object_table_query
from ni_rua.perception.external.rectify import Rectified, clear_rectified_cache, map_npz_uv, rectified_cached
from ni_rua.perception.scene.store import SceneStore
from support.geometry_fixture import IDENTITY
from support.perception_fixtures import FakeService, fake_object

K = [[510., 0, 346.], [0, 509., 196.], [0, 0, 1.]]
D = [-0.015, 0.224, -0.010, 0.015, -0.386]   # the SO-101 calibration of 2026-11-08 (plumb_bob)
T_FIXED = [[1, 0, 0, 0.3], [0, -1, 0, 0], [0, 0, -1, 0.4], [0, 0, 0, 1]]


def fixed_camera_observation(store, distortion=D):
    image = cv2.GaussianBlur(np.random.default_rng(5).integers(0, 256, (480, 640, 3), dtype=np.uint8), (5, 5), 0)
    data = cv2.imencode(".jpg", image)[1].tobytes()
    calibration = {"reference_frame_id": "so101_base", "cameras": {"fixed": {
        "K": K, "distortion": {"model": "opencv", "coefficients": list(distortion)}, "image_size": [640, 480],
        "T_world_camera": T_FIXED}}}
    calibration_ref = store.put_artifact(canonical(calibration).encode(), "calibration", "application/json")
    model_ref = store.put_artifact(canonical({"base_link": "base", "joints": []}).encode(), "robot_model", "application/json")
    state_ref = store.put_artifact(canonical({"source_kind": "measurement", "sim_step": 0, "step_domain": "control_step"}).encode(),
                                   "robot_state", "application/json")
    cameras = [{"camera_id": "fixed", "frame_id": "frame-fixed", "width": 640, "height": 480, "sim_step": 0,
                "step_domain": "control_step", "rgb_ref": store.put_artifact(data, "rgb", "image/jpeg"),
                "calibration_ref": calibration_ref, "pixel_transform": "identity"}]
    return {"schema_version": "0.1", **IDENTITY, "world_kind": "observed", "data_origin": "synthetic_fixture",
            "observation_id": "obs-fixed", "sim_step": 0, "step_domain": "control_step",
            "clock_domain": "synthetic/generation-1", "calibration_id": "synthetic-calibration-v1",
            "robot_model_id": "synthetic-camera-rig", "robot_model_ref": model_ref,
            "observation_profile": "rgb_proprio_calibrated_synthetic_v1", "robot_state_ref": state_ref,
            "cameras": cameras}, data


class RectifiedMapping(unittest.TestCase):
    def test_zero_distortion_passes_through(self):
        rect = Rectified(dict(K=K, D=[]), b"not-an-image")
        self.assertFalse(rect.active)
        self.assertEqual(rect.image_bytes, b"not-an-image")
        np.testing.assert_array_equal(rect.to_original([[10, 20]]), [[10, 20]])
        objects = [dict(label_px=[1, 2], contour_px=[[1, 2]], bbox_px=[0, 0, 3, 3])]
        self.assertEqual(rect.map_objects(objects), [dict(label_px=[1, 2], contour_px=[[1, 2]], bbox_px=[0, 0, 3, 3])])

    def test_distorted_round_trip_jpeg_quality95_and_lossless_png(self):
        import hashlib
        from unittest import mock

        noisy = cv2.GaussianBlur(np.random.default_rng(11).integers(0, 256, (480, 640, 3), dtype=np.uint8), (5, 5), 0)
        jpeg_in = cv2.imencode(".jpg", noisy)[1].tobytes()
        png_in = cv2.imencode(".png", noisy)[1].tobytes()

        rect_jpg = Rectified(dict(K=K, D=D), jpeg_in)
        self.assertTrue(rect_jpg.active)
        self.assertTrue(rect_jpg.image_bytes.startswith(b"\xff\xd8\xff"))
        self.assertLess(len(rect_jpg.image_bytes), int(len(jpeg_in) * 1.6))  # avoids 5x PNG bloat on JPEG cameras
        self.assertEqual(Rectified(dict(K=K, D=D), jpeg_in).image_bytes, rect_jpg.image_bytes)
        self.assertEqual(rect_jpg.sha256, hashlib.sha256(rect_jpg.image_bytes).hexdigest())
        decoded = cv2.imdecode(np.frombuffer(rect_jpg.image_bytes, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, (480, 640, 3))

        rect_png = Rectified(dict(K=K, D=D), png_in)
        self.assertTrue(rect_png.active)
        self.assertTrue(rect_png.image_bytes.startswith(b"\x89PNG"))
        self.assertGreater(len(rect_png.image_bytes), len(rect_jpg.image_bytes) * 2)

        pixels = np.array([[50., 40.], [346., 196.], [600., 450.], [320., 240.]])
        back = rect_jpg.to_original(rect_jpg.to_rectified(pixels))
        np.testing.assert_allclose(back, pixels, atol=0.05)
        self.assertGreater(np.abs(rect_jpg.to_rectified(pixels[:1]) - pixels[:1]).max(), 1.0)   # corners do move

        clear_rectified_cache()
        self.addCleanup(clear_rectified_cache)
        with mock.patch("cv2.undistort", wraps=cv2.undistort) as m_undistort:
            cached_1 = rectified_cached(dict(K=K, D=D), jpeg_in)
            cached_2 = rectified_cached(dict(K=K, D=D), bytearray(jpeg_in))
            cached_3 = rectified_cached(dict(K=K, D=D), memoryview(jpeg_in))
        self.assertIs(cached_1, cached_2)
        self.assertIs(cached_1, cached_3)
        self.assertIsInstance(cached_1.image_bytes, bytes)
        self.assertEqual(m_undistort.call_count, 1)
        self.assertEqual(cached_1.image_bytes, rect_jpg.image_bytes)
        mut_zero = bytearray(b"raw-zero-distortion")
        rect_zero = rectified_cached(dict(K=K, D=[]), mut_zero)
        mut_zero[0] = ord("X")
        self.assertIsInstance(rect_zero.image_bytes, bytes)
        self.assertEqual(rect_zero.image_bytes, b"raw-zero-distortion")

        with self.assertRaisesRegex(ValueError, "invalid_rgb"):
            Rectified(dict(K=K, D=D), b"\xff\xd8\xffcorrupt-jpeg")
        with mock.patch("cv2.imencode", return_value=(False, np.zeros(0, dtype=np.uint8))):
            with self.assertRaisesRegex(ValueError, "rectified_image_encoding_failed"):
                Rectified(dict(K=K, D=D), png_in)

    def test_npz_uv_mapped_to_original_image(self):
        rect = Rectified(dict(K=K, D=D), cv2.imencode(".png", np.zeros((480, 640, 3), np.uint8))[1].tobytes())
        buffer = io.BytesIO()
        uv = np.array([[30, 20], [346, 196]], np.uint16)
        np.savez_compressed(buffer, xyz=np.zeros((2, 3), np.float32), uv=uv, support_z=np.float32(0.0))
        with np.load(io.BytesIO(map_npz_uv(buffer.getvalue(), rect))) as npz:
            mapped = npz["uv"]
            self.assertEqual(mapped.dtype, np.uint16)
            np.testing.assert_allclose(mapped, np.rint(rect.to_original(uv.astype(float))))
            self.assertEqual(float(npz["support_z"]), 0.0)


class FixedCameraObjectTable(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = self.enterContext(tempfile.TemporaryDirectory())
        self.artifacts = Store(self.tmp_dir, IDENTITY); self.addCleanup(self.artifacts.close)
        self.scene = SceneStore(Path(self.tmp_dir) / "scene", self.artifacts.resolve_artifact, self.artifacts.put_artifact)
        observation, self.jpeg = fixed_camera_observation(self.artifacts)
        self.scene.import_observation(observation)

    def runtime(self, endpoint, **features):
        config = Config().with_overrides({"features": {"external_perception": False, "object_table": True,
                                                       "vggt_endpoint": endpoint, "vggt_timeout_s": 5.0, **features}})
        return SimpleNamespace(config=config, scene=self.scene, image_settings=None, identity=dict(IDENTITY),
                               store=self.artifacts, features=config.features)

    def test_x5_defaults_unchanged(self):
        features = Config().features
        # vggt_support_z_m defaults to the embodiment's support height (WP12, F7), which is 0.765 m for ARX-X5.
        self.assertEqual((features.vggt_camera_id, features.vggt_support_z_m), ("cam_head", "profile"))
        self.assertEqual(effective_support_z_m(features), 0.765)
        self.assertEqual(default_camera(SimpleNamespace(config=Config())), "cam_head")
        self.assertEqual(obs3d.camera_of(SimpleNamespace(config=Config())), "cam_head")

    def test_before_fix_default_camera_was_missing(self):
        fake = FakeService(); self.addCleanup(fake.close)
        result, _ = object_table_query(self.runtime(fake.url), {"kind": "object_table", "observation_id": "obs-fixed"})
        self.assertEqual(result["status"], "unknown")      # cam_head does not exist on this rig
        self.assertIn("unknown_camera", result["reason"])
        self.assertEqual(fake.requests, [])

    def test_configured_camera_distorted_image_and_support_plane(self):
        fake = FakeService(objects=[fake_object((0.3, 0.0, 0.02), 0.04, 0.0, label=(100, 80))]); self.addCleanup(fake.close)
        runtime = self.runtime(fake.url, vggt_camera_id="fixed", vggt_support_z_m=0.0)
        self.assertEqual(obs3d.camera_of(runtime), "fixed")
        result, overlays = object_table_query(runtime, {"kind": "object_table", "observation_id": "obs-fixed",
                                                        "pixels": [[60, 50]]})
        self.assertEqual(result["status"], "conditional", result)
        self.assertEqual(result["value"]["camera_id"], "fixed")
        self.assertEqual([name for name, _, _ in overlays], ["fixed"])
        _, sent = fake.requests[0]
        self.assertEqual(sent["support_z_m"], 0.0)
        self.assertNotIn("exclude_boxes_px", sent)          # X5 head-mount box never applies to another camera
        sent_image = base64.b64decode(sent["image_b64"])
        self.assertTrue(sent_image.startswith(b"\xff\xd8\xff"))   # undistorted JPEG q=95, not the raw distorted JPEG
        self.assertNotEqual(sent_image, self.jpeg)
        np.testing.assert_allclose(sent["K"], K)
        rect = Rectified(dict(K=K, D=D), self.jpeg)
        np.testing.assert_allclose(sent["pixels"], rect.to_rectified([[60, 50]]), atol=1e-6)
        # service pixels come back in the original image; the requested pixel too
        expected = np.rint(rect.to_original([[100, 80]])[0]).astype(int).tolist()
        self.assertEqual(result["diagnostics"]["objects"][0]["label_px"], expected)
        np.testing.assert_allclose(result["value"]["requested"][0]["pixel"], rect.to_original(sent["pixels"])[0], atol=0.06)

    def test_zero_distortion_sends_stored_bytes(self):
        tmp_dir = self.enterContext(tempfile.TemporaryDirectory())
        artifacts = Store(tmp_dir, IDENTITY); self.addCleanup(artifacts.close)
        scene = SceneStore(Path(tmp_dir) / "scene", artifacts.resolve_artifact, artifacts.put_artifact)
        observation, jpeg = fixed_camera_observation(artifacts, distortion=[])
        scene.import_observation(observation)
        fake = FakeService(); self.addCleanup(fake.close)
        runtime = self.runtime(fake.url, vggt_camera_id="fixed", vggt_support_z_m=0.0)
        runtime.scene, runtime.store = scene, artifacts
        result, _ = object_table_query(runtime, {"kind": "object_table", "observation_id": "obs-fixed", "pixels": [[60, 50]]})
        self.assertEqual(result["status"], "conditional", result)
        self.assertEqual(base64.b64decode(fake.requests[0][1]["image_b64"]), jpeg)
        self.assertEqual(fake.requests[0][1]["pixels"], [[60.0, 50.0]])

    def test_so101_default_camera_and_calibrated_table_z_linkage(self):
        tmp_dir = self.enterContext(tempfile.TemporaryDirectory())
        artifacts = Store(tmp_dir, IDENTITY); self.addCleanup(artifacts.close)
        scene = SceneStore(Path(tmp_dir) / "scene", artifacts.resolve_artifact, artifacts.put_artifact)
        observation, _ = fixed_camera_observation(artifacts)
        observation["robot_model_id"] = "so101"
        calibration = {"reference_frame_id": "so101_base", "joint_calibration": {"table_z_m": -0.018}, "cameras": {"fixed": {
            "K": K, "distortion": {"model": "opencv_plumb_bob", "coefficients": list(D)}, "image_size": [640, 480],
            "T_world_camera": T_FIXED}}}
        cal_ref = artifacts.put_artifact(canonical(calibration).encode(), "calibration", "application/json")
        observation["cameras"][0]["calibration_ref"] = cal_ref
        scene.import_observation(observation)
        fake = FakeService(objects=[fake_object((0.25, 0.0, -0.005), 0.03, 0.0, label=(100, 80))]); self.addCleanup(fake.close)
        runtime = self.runtime(fake.url, vggt_support_z_m=None)
        runtime.scene, runtime.store = scene, artifacts
        self.assertEqual(default_camera(runtime, observation), "fixed")
        self.assertEqual(obs3d.camera_of(runtime, observation), "fixed")
        from ni_rua.robot.model.geometry import GeometryTools
        cam = GeometryTools(scene).camera(IDENTITY, "obs-fixed", "fixed")
        self.assertEqual(effective_support_z_m(runtime.features, cam), -0.018)
        result, _ = object_table_query(runtime, {"kind": "object_table", "observation_id": "obs-fixed"})
        self.assertEqual(result["status"], "conditional", result)
        self.assertEqual(result["value"]["camera_id"], "fixed")
        self.assertEqual(result["value"]["support_z_m"], -0.018)
        self.assertEqual(fake.requests[0][1]["support_z_m"], -0.018)

    def test_observation_support_is_sent_by_all_three_primary_request_paths(self):
        from ni_rua.perception.external import scene3d, vggt_height
        from support.perception_fixtures import FakePoints
        from perception.external.test_vggt_height import FakeService as FakeHeight
        table_service, points_service, height_service = FakeService(), FakePoints(), FakeHeight(center=-.005, top=.01)
        for service in (table_service, points_service, height_service):
            self.addCleanup(service.close)
        for index, table_z in enumerate((-.018, -.016, None)):
            observation, _ = fixed_camera_observation(self.artifacts)
            observation['observation_id'] = f'obs-support-{index}'
            calibration = {'reference_frame_id': 'so101_base', 'cameras': {'fixed': {
                'K': K, 'distortion': {'model': 'opencv', 'coefficients': D}, 'image_size': [640, 480],
                'T_world_camera': T_FIXED}}}
            if table_z is not None:
                calibration['joint_calibration'] = {'table_z_m': table_z}
            observation['cameras'][0]['calibration_ref'] = self.artifacts.put_artifact(
                canonical(calibration).encode(), 'calibration', 'application/json')
            self.scene.import_observation(observation)
            for kind, service in (('object_table', table_service), ('scene3d', points_service), ('pixel_plane', height_service)):
                with self.subTest(table_z=table_z, kind=kind):
                    runtime = self.runtime(service.url, vggt_camera_id='fixed', vggt_support_z_m=None,
                                           scene3d=True, vggt_height=True)
                    request = {'kind': kind, 'observation_id': observation['observation_id'], 'camera_id': 'fixed'}
                    before = len(service.requests)
                    if kind == 'object_table':
                        result, _ = object_table_query(runtime, request)
                    elif kind == 'scene3d':
                        result, _ = scene3d.scene3d_query(runtime, request)
                    else:
                        result = vggt_height.pixel_plane_with_estimate(runtime, dict(request, pixel=[320, 240], plane_source='vggt'))
                    if table_z is None:
                        self.assertEqual(result['status'], 'unknown', result)
                        self.assertIn('observation_support_z', result['reason'])
                        self.assertEqual(len(service.requests), before)
                    else:
                        self.assertEqual(result['status'], 'conditional', result)
                        body = service.requests[-1] if kind == 'pixel_plane' else service.requests[-1][1]
                        self.assertEqual(body['support_z_m'], table_z)

class SupportPlaneConfigTests(unittest.TestCase):
    def test_support_plane_may_lie_below_the_robot_base_origin(self):
        from ni_rua.core.config import Config
        self.assertEqual(Config.from_dict({"features": {"vggt_support_z_m": -0.02}}).features.vggt_support_z_m, -0.02)
        self.assertEqual(Config().features.vggt_support_z_m, "profile")
        self.assertEqual(effective_support_z_m(Config().features), 0.765)   # the ARX-X5 table top

    def test_explicit_numeric_height_is_authoritative_including_x5_default(self):
        camera = {'joint_calibration': {'table_z_m': -.018}}
        for configured in (.765, -.02, 0., 3., -3.):
            features = Config.from_dict({'features': {'vggt_support_z_m': configured}}).features
            self.assertEqual(effective_support_z_m(features, camera), configured)

    def test_observation_mode_uses_each_frozen_camera_not_global_active_state(self):
        features = Config.from_dict({'features': {'vggt_support_z_m': None}}).features
        old = {'joint_calibration': {'table_z_m': -.018}}
        new = {'joint_calibration': {'table_z_m': -.016}}
        for camera, expected in ((old, -.018), (new, -.016), (old, -.018),
                                 ({'calibration': {'joint_calibration': {'table_z_m': 0.}}}, 0.)):
            self.assertEqual(effective_support_z_m(features, camera), expected)
        self.assertIsNone(features.vggt_support_z_m)

    def test_observation_mode_rejects_missing_or_invalid_evidence(self):
        features = Config.from_dict({'features': {'vggt_support_z_m': None}}).features
        for camera in (None, {}, {'joint_calibration': {}}, *[
                {'joint_calibration': {'table_z_m': value}} for value in (None, True, '0', float('nan'), float('inf'), 3.1)]):
            with self.subTest(camera=camera), self.assertRaisesRegex(ValueError, 'observation_support_z'):
                effective_support_z_m(features, camera)


if __name__ == "__main__":
    unittest.main()
