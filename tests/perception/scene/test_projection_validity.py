"""Defect D1: ``geometry.project`` must not fold points beyond the lens distortion model's range back into the image.

Real data: ``tests/fixtures/so101_projection_fold.json`` (SO-101 calibration and pose numbers of the WP11 real-arm
run; numbers only).
"""
NI_RUA_TEST_TIER = "component"

from copy import deepcopy
import json
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from ni_rua.core.contracts import canonical
from ni_rua.robot.model.geometry import GeometryTools, distortion_monotonic_limit, project
from support.paths import CONFIGS_ROOT, FIXTURES_ROOT

FIXTURE = json.loads((FIXTURES_ROOT / "so101_projection_fold.json").read_text())
TIPS = [FIXTURE["fingertips_m"]["fixed"], FIXTURE["fingertips_m"]["moving"]]


def real_camera(name):
    c = FIXTURE["cameras"][name]
    return dict(K=np.array(c["K"]), D=np.array(c["distortion"]), T=np.array(c["T_world_camera"]),
                width=c["width"], height=c["height"])


def opencv_pixels(camera, points):
    inverse = np.linalg.inv(camera["T"])
    return cv2.projectPoints(np.asarray(points, float), cv2.Rodrigues(inverse[:3, :3])[0], inverse[:3, 3],
                             camera["K"], camera["D"])[0].reshape(-1, 2)


def legacy_project(camera, points):
    """``project`` as it was from 0.2.0 through 8916d78 (the D1 behaviour), for the byte-identity checks."""
    points = np.asarray(points, float).reshape(-1, 3)
    inverse = np.linalg.inv(camera["T"])
    local = points @ inverse[:3, :3].T + inverse[:3, 3]
    pixels = opencv_pixels(camera, points)
    visible = ((local[:, 2] > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < camera["width"]) & (pixels[:, 1] >= 0)
               & (pixels[:, 1] < camera["height"]))
    return pixels, local[:, 2], visible


def camera_frame(camera, points):
    inverse = np.linalg.inv(camera["T"])
    return np.asarray(points, float) @ inverse[:3, :3].T + inverse[:3, 3]


def first_fold_along_x(distortion, r_max=6.0, samples=600001):
    """Numerical reference: first normalised radius where OpenCV's distorted x stops increasing along +x."""
    r = np.linspace(1e-6, r_max, samples)
    rays = np.column_stack((r, np.zeros_like(r), np.ones_like(r)))
    x = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), np.eye(3), np.asarray(distortion, float))[0][:, 0, 0]
    falling = np.flatnonzero(np.diff(x) <= 0)
    return float(r[falling[0]]) if len(falling) else math.inf


def so101_profile():
    from ni_rua.robot.backends.so101.deployment import embodiment_profile, validate_deployment
    document = json.loads((CONFIGS_ROOT / "so101.example.json").read_text())
    return embodiment_profile(validate_deployment(document, base=CONFIGS_ROOT))


def untouched(before, after, pixel, radius=6):
    u, v = (int(round(value)) for value in pixel)
    window = np.s_[max(0, v - radius):v + radius + 1, max(0, u - radius):u + radius + 1]
    return np.array_equal(before[window], after[window])


class RealSo101FoldTest(unittest.TestCase):
    """The WP11 case: moving fingertip 61.1 deg off the wrist camera axis, radial factor -0.237."""

    def test_wrist_fingertip_beyond_the_fold_is_not_visible_and_has_no_pixel(self):
        camera = real_camera("wrist")
        folded = opencv_pixels(camera, TIPS)[1]
        # The defect itself: OpenCV folds the point to a pixel inside the 640x480 image.
        np.testing.assert_allclose(folded, FIXTURE["observed_before_fix"]["wrist_moving_tip_folded_px"], atol=.05)
        local = camera_frame(camera, TIPS)
        radius = np.hypot(local[:, 0], local[:, 1]) / local[:, 2]
        limit = distortion_monotonic_limit(camera["D"])
        self.assertAlmostEqual(math.degrees(math.atan(limit)), 52.0, delta=.05)
        self.assertLess(radius[0], limit)
        self.assertGreater(radius[1], limit)
        pixels, depth, visible = project(camera, TIPS)
        self.assertEqual(visible.tolist(), [True, False])
        self.assertTrue(np.isnan(pixels[1]).all())
        np.testing.assert_allclose(depth, local[:, 2], rtol=0, atol=1e-15)
        self.assertGreater(depth[1], 0)  # in front of the camera, but outside its view
        # The point inside the monotonic range keeps OpenCV's exact pixel.
        self.assertEqual(pixels[0].tobytes(), opencv_pixels(camera, TIPS)[0].tobytes())
        np.testing.assert_allclose(pixels[0], FIXTURE["observed_before_fix"]["wrist_fixed_tip_px"], atol=.05)

    def test_fixed_camera_tips_inside_the_range_keep_exact_pixels(self):
        camera = real_camera("fixed")
        pixels, _, visible = project(camera, TIPS)
        self.assertEqual(visible.tolist(), [True, True])
        self.assertEqual(pixels.tobytes(), opencv_pixels(camera, TIPS).tobytes())
        expected = FIXTURE["observed_before_fix"]
        np.testing.assert_allclose(pixels, [expected["fixed_fixed_tip_px"], expected["fixed_moving_tip_px"]], atol=.05)
        self.assertAlmostEqual(math.degrees(math.atan(distortion_monotonic_limit(camera["D"]))), 50.2, delta=.1)

    def test_off_axis_sweep_never_folds_back(self):
        """Along any azimuth, every returned pixel moves outward with the off-axis angle; past the limit: no pixel."""
        for name in ("wrist", "fixed"):
            camera = real_camera(name)
            limit = distortion_monotonic_limit(camera["D"])
            principal = camera["K"][:2, 2]
            angles = np.radians(np.linspace(.5, 89, 400))
            for azimuth in np.radians(np.arange(0, 360, 30)):
                with self.subTest(camera=name, azimuth_deg=math.degrees(azimuth)):
                    local = np.column_stack((np.tan(angles) * np.cos(azimuth), np.tan(angles) * np.sin(azimuth),
                                             np.ones_like(angles)))
                    world = local @ camera["T"][:3, :3].T + camera["T"][:3, 3]
                    pixels, depth, visible = project(camera, world)
                    located = np.isfinite(pixels).all(axis=1)
                    np.testing.assert_array_equal(located, np.tan(angles) <= limit)
                    distance = np.linalg.norm(pixels[located] - principal, axis=1)
                    self.assertTrue(np.all(np.diff(distance) > 0))
                    self.assertFalse(visible[~located].any())
                    self.assertTrue(np.all(depth > 0))
                    self.assertEqual(pixels[located].tobytes(), opencv_pixels(camera, world)[located].tobytes())


class MonotonicLimitTest(unittest.TestCase):
    def test_zero_distortion_has_no_limit(self):
        for size in (0, 4, 5, 8, 12, 14):
            self.assertEqual(distortion_monotonic_limit(np.zeros(size)), math.inf)
        self.assertEqual(distortion_monotonic_limit([]), math.inf)
        # A sensor tilt alone (14-coefficient model) does not fold.
        self.assertEqual(distortion_monotonic_limit([0.] * 12 + [.01, -.02]), math.inf)

    def test_single_barrel_term_matches_the_analytic_fold(self):
        # r_d = r (1 + k1 r^2): dr_d/dr = 1 + 3 k1 r^2 = 0 at r = 1/sqrt(-3 k1).
        for k1 in (-.05, -.1, -.3, -.5):
            self.assertAlmostEqual(distortion_monotonic_limit([k1, 0, 0, 0, 0]), 1 / math.sqrt(-3 * k1), places=10)

    def test_pincushion_and_rational_without_fold_are_unlimited(self):
        self.assertEqual(distortion_monotonic_limit([.1, .01, 0, 0, 0]), math.inf)
        self.assertEqual(distortion_monotonic_limit([.1, .01, 0, 0, 0, .5, 0, 0]), math.inf)

    def test_rational_denominator_zero_ends_the_range(self):
        # r_d = r / (1 - 0.5 r^2) rises monotonically until its pole at r = sqrt(2), then jumps to -inf.
        self.assertAlmostEqual(distortion_monotonic_limit([0, 0, 0, 0, 0, -.5, 0, 0]), math.sqrt(2), places=10)

    def test_radial_limits_agree_with_opencv_sampling(self):
        cases = [[-.3, 0, 0, 0, 0], [-.0030043622417742265, .02039943178023648, 0, 0, -.04061331606269772],
                 [-.007205513430464183, .029881620400173454, 0, 0, -.060596879334350935],
                 [.05, -.02, 0, 0, 0], [-.2, .05, 0, 0, -.01, .1, .02, 0],
                 [.2, 0, 0, 0, 0, 3., 0, 0]]  # the last folds at r = 0.687 and rises again past r = 1.88
        for distortion in cases:
            with self.subTest(distortion=distortion):
                expected = first_fold_along_x(distortion)
                limit = distortion_monotonic_limit(distortion)
                self.assertTrue(math.isfinite(limit))
                self.assertAlmostEqual(limit, expected, delta=2e-5)

    def test_tangential_bound_is_exact_at_the_worst_azimuth(self):
        """With p1, p2 the limit is where the along-ray radial slope reaches zero at the azimuth where the tangential
        terms pull inward most (p1 sin t + p2 cos t = -hypot(p1, p2), t = atan2(-p1, -p2))."""
        for name in ("wrist", "fixed"):
            distortion = np.array(FIXTURE["cameras"][name]["distortion"])
            k1, k2, p1, p2, k3 = distortion
            limit = distortion_monotonic_limit(distortion)
            self.assertLess(limit, distortion_monotonic_limit([k1, k2, 0, 0, k3]))
            azimuth = math.atan2(-p1, -p2)
            direction = np.array([math.cos(azimuth), math.sin(azimuth)])

            def slope(r, step=1e-6):
                rays = np.array([[*(r - step) * direction, 1.], [*(r + step) * direction, 1.]])
                distorted = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), np.eye(3), distortion)[0].reshape(2, 2)
                return float(np.diff(distorted @ direction)[0] / (2 * step))
            with self.subTest(camera=name):
                self.assertAlmostEqual(slope(limit), 0., delta=1e-6)
                self.assertGreater(slope(.98 * limit), 0)
                self.assertLess(slope(1.02 * limit), 0)
                # Any other azimuth keeps a positive slope at the limit: the bound is conservative.
                for other in np.radians(np.arange(0, 360, 15)):
                    d = np.array([math.cos(other), math.sin(other)])
                    rays = np.array([[*(limit - 1e-6) * d, 1.], [*(limit + 1e-6) * d, 1.]])
                    points = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), np.eye(3), distortion)[0].reshape(2, 2)
                    self.assertGreater(float(np.diff(points @ d)[0]), -1e-12)

    def test_thin_prism_terms_make_the_limit_smaller(self):
        radial = distortion_monotonic_limit([-.1, 0, 0, 0, 0, 0, 0, 0])
        prism = distortion_monotonic_limit([-.1, 0, 0, 0, 0, 0, 0, 0, .01, 0, -.01, .001])
        self.assertLess(prism, radial)

    def test_invalid_coefficient_vectors_are_refused(self):
        for distortion in ([0.] * 15, [math.nan, 0, 0, 0, 0], [math.inf, 0, 0, 0]):
            with self.subTest(distortion=distortion), self.assertRaises(ValueError):
                distortion_monotonic_limit(distortion)


class ZeroDistortionIdentityTest(unittest.TestCase):
    """X5 / RoboDojo simulated cameras use D = 0: project() must be byte-identical to the legacy function."""

    def test_zero_distortion_is_byte_identical_to_the_legacy_projection(self):
        rng = np.random.default_rng(7)
        rotation = cv2.Rodrigues(np.array([.3, -.2, .5]))[0]
        T = np.eye(4); T[:3, :3] = rotation; T[:3, 3] = [.1, -.2, .3]
        points = np.vstack((rng.normal(0, 1, (500, 3)),                                    # in front and behind
                            T[:3, 3] + rng.normal(0, 1, (100, 3)) * [1, 1, 1e-4] @ rotation.T))  # near z = 0
        for D in (np.zeros(5), np.zeros(0), np.zeros(8), np.zeros(14), np.zeros(4)):
            camera = dict(K=np.array([[500., 0, 320], [0, 500, 240], [0, 0, 1]]), D=D, T=T, width=640, height=480)
            with self.subTest(size=len(D)):
                new, old = project(camera, points), legacy_project(camera, points)
                for a, b in zip(new, old):
                    self.assertEqual(a.tobytes(), b.tobytes())

    def test_distorted_points_inside_the_range_are_byte_identical(self):
        camera = real_camera("wrist")
        rng = np.random.default_rng(11)
        limit = distortion_monotonic_limit(camera["D"])
        local = np.column_stack((rng.uniform(-1, 1, (800, 2)) * limit / math.sqrt(2), np.ones(800)))
        local *= rng.uniform(.05, 2., (800, 1))
        local[::7, 2] *= -1  # some behind the camera: OpenCV's pixel and depth <= 0 are kept as before
        world = local @ camera["T"][:3, :3].T + camera["T"][:3, 3]
        for a, b in zip(project(camera, world), legacy_project(camera, world)):
            self.assertEqual(a.tobytes(), b.tobytes())


class CallerTest(unittest.TestCase):
    """Every caller reports an unlocated point as outside the image and draws nothing at the folded pixel."""

    def setUp(self):
        self.camera = real_camera("wrist")
        self.folded = FIXTURE["observed_before_fix"]["wrist_moving_tip_folded_px"]
        arm = FIXTURE["measured_right_arm"]
        self.arm = deepcopy(arm)
        self.image = np.full((480, 640, 3), 40, np.uint8)

    def bundle(self):
        sync = dict(sim_step=7, step_domain="control_step")
        state = dict(sync, arms=dict(right=dict(source_kind="measurement", joint_names=self.arm["joint_names"],
                                                q_rad=self.arm["q_rad"], ee_pose=self.arm["ee_pose"])))
        return dict(self.camera, observation=dict(sync, world_kind="observed", observation_id="obs-d1"),
                    frame=dict(sync, camera_id="wrist", frame_id="wrist-7", pixel_transform="identity",
                               rgb_ref="artifact:rgb"),
                    state=state, reference_frame_id="so101_base", pose_source="robot_kinematics")

    def test_contact_overlay_reports_the_moving_tip_outside_the_image(self):
        from ni_rua.perception.overlays.contact_overlay import render_contact_overlay
        output, metadata = render_contact_overlay(self.bundle(), self.image, arms=["right"], profile=so101_profile())
        points = {point["reference"]: point for point in metadata["points"]}
        moving, fixed = points["moving_tip"], points["fixed_tip"]
        np.testing.assert_allclose(moving["position_m"], TIPS[1], atol=1e-9)
        self.assertIsNone(moving["pixel_xy"])
        self.assertFalse(moving["in_image"])
        self.assertTrue(moving["in_front_of_camera"])
        self.assertTrue(fixed["in_image"])
        np.testing.assert_allclose(fixed["pixel_xy"], FIXTURE["observed_before_fix"]["wrist_fixed_tip_px"], atol=.05)
        self.assertTrue(untouched(self.image, output, self.folded))
        json.dumps(metadata, allow_nan=False, default=lambda value: np.asarray(value).tolist())

    def test_pose_axis_frame_skips_only_the_unlocated_fingertip(self):
        from ni_rua.perception.overlays.pose_axis_overlay import draw_pose_axis_frame
        from ni_rua.robot.model.roboprobe_pose import quat_to_matrix
        x, y, z, w = self.arm["ee_pose"]["quaternion_xyzw"]
        arm_data = dict(q_rad=self.arm["q_rad"], gripper_frame_link_pose=dict(
            position_m=self.arm["ee_pose"]["position_m"], rotation_matrix=quat_to_matrix([w, x, y, z]).tolist()))
        output = self.image.copy()
        result = draw_pose_axis_frame(output, self.camera, "right", arm_data, (0, 0, 255), 640, 480,
                                      profile=so101_profile(), joint_names=self.arm["joint_names"])
        self.assertIsNotNone(result)  # the world references do not depend on the projection
        np.testing.assert_allclose(result["moving_fingertip_m"], np.round(TIPS[1], 4), atol=1e-9)
        self.assertTrue(untouched(self.image, output, self.folded))
        fixed_px = project(self.camera, TIPS)[0][0]
        self.assertFalse(untouched(self.image, output, fixed_px, radius=2))  # the visible fixed tip is still drawn

    def test_overlays_treat_a_nan_pixel_like_a_point_behind_the_camera(self):
        """Trajectory, pose-axis and batch-proxy drawings: a NaN pixel draws exactly what a z <= 0 point draws."""
        from unittest import mock
        from ni_rua.perception.overlays import pose_axis_overlay, trajectory_overlay
        from ni_rua.robot.model import geometry
        from ni_rua.robot.motion.batch_geometry import annotate_overlay
        camera = dict(K=np.array([[250., 0, 160], [0, 250., 120], [0, 0, 1]]), D=np.zeros(5), T=np.eye(4),
                      width=320, height=240, observation=dict(world_kind="observed", observation_id="obs-1",
                      sim_step=1, step_domain="control_step"), frame=dict(camera_id="head", frame_id="head",
                      sim_step=1, step_domain="control_step", rgb_ref="rgb-1", pixel_transform="identity"),
                      reference_frame_id="env_origin", pose_source="fixed_calibration")
        arm_data = dict(phase="move", link_positions_m=dict(link5=[0., -.1, 1.5], link6=[0., 0., 1.5]),
                        link6_pose=dict(position_m=[0., 0., 1.5], rotation_matrix=np.eye(3).tolist()))
        samples = [dict(native_step=i, arms=dict(left=deepcopy(arm_data))) for i in range(3)]
        for i, sample in enumerate(samples):
            sample["arms"]["left"]["link_positions_m"]["link6"] = [.02 * i, 0., 1.5]
        trajectory = dict(source_kind="candidate_command_trajectory_not_measurement", frame_id="env_origin",
                          label="plan-1", robot_model=dict(joints=[dict(parent="link5", child="link6")]),
                          samples=samples)
        image = np.zeros((240, 320, 3), np.uint8)
        real = geometry.project

        def hide(index, how):
            def wrapped(camera_, points):
                pixels, depth, visible = real(camera_, points)
                pixels, depth, visible = pixels.copy(), depth.copy(), visible.copy()
                if index < len(pixels):
                    if how == "nan":
                        pixels[index] = np.nan
                    else:
                        depth[index] = -1.
                    visible[index] = False
                return pixels, depth, visible
            return wrapped

        def render(how, index):
            with mock.patch.object(trajectory_overlay, "project", hide(index, how)), \
                    mock.patch.object(pose_axis_overlay, "project", hide(index, how)):
                return trajectory_overlay.render_trajectory_overlay(camera, image, trajectory, native_steps=[0, 2])
        for index in range(6):
            with self.subTest(drawing="trajectory+pose_axis", index=index):
                (nan_image, nan_meta), (behind_image, behind_meta) = render("nan", index), render("behind", index)
                np.testing.assert_array_equal(nan_image, behind_image)
                self.assertEqual(nan_meta, behind_meta)
        with mock.patch.object(pose_axis_overlay, "project", hide(5, "nan")):
            self.assertIsNotNone(pose_axis_overlay.draw_pose_axis_frame(image.copy(), camera, "left", arm_data,
                                                                         (255, 160, 48), 320, 240))
        box = dict(id="box", min_m=[-.05, -.05, 1.4], max_m=[.05, .05, 1.5])
        batch = dict(samples=[dict(native_step=0, arms={})], batch_geometry_review=dict(obstacles=[box]))
        metadata = dict(panel_width=320, panel_height=240, panels=[dict(native_step=0)])
        for index in range(8):
            with self.subTest(drawing="batch_proxy", index=index):
                drawn = {}
                for how in ("nan", "behind"):
                    with mock.patch.object(geometry, "project", hide(index, how)):
                        drawn[how] = annotate_overlay(camera, image.copy(), deepcopy(metadata), batch)[0]
                np.testing.assert_array_equal(drawn["nan"], drawn["behind"])


class GeometryToolResultTest(unittest.TestCase):
    """The model-visible geometry_query results: null pixel, in_image false, strict JSON, valid scene contract."""

    def setUp(self):
        from ni_rua.core.storage.store import Store
        from ni_rua.perception.scene.store import SceneStore
        from support.geometry_fixture import IDENTITY, recorded_plane
        self.identity = IDENTITY
        self.tmp_dir = self.enterContext(tempfile.TemporaryDirectory())
        self.artifacts = Store(self.tmp_dir, IDENTITY); self.addCleanup(self.artifacts.close)
        self.scene = SceneStore(Path(self.tmp_dir) / "scene", self.artifacts.resolve_artifact,
                                self.artifacts.put_artifact)
        self.obs = recorded_plane(self.artifacts)
        for observation in self.obs:
            self.scene.import_observation(observation)
        self.geometry = GeometryTools(self.scene)
        wrist = real_camera("wrist")
        # The fixture camera sits at the world origin of obs-0 (x = 0) and obs-2 (x = 0.05): these are the two tips
        # in the real wrist-camera frame, so the real K/D reproduce the WP11 projection.
        self.local_tips = camera_frame(wrist, TIPS).tolist()

    def distorted(self, index, identifier):
        observation = deepcopy(self.obs[index])
        observation["observation_id"] = identifier
        calibration = json.loads(self.artifacts.resolve_artifact(observation["cameras"][0]["calibration_ref"])[1])
        calibration["cameras"]["head"].update(K=FIXTURE["cameras"]["wrist"]["K"],
            distortion={"model": "opencv", "coefficients": FIXTURE["cameras"]["wrist"]["distortion"]})
        observation["cameras"][0]["calibration_ref"] = self.artifacts.put_artifact(
            canonical(calibration).encode(), "calibration", "application/json")
        self.scene.import_observation(observation)
        return identifier

    def check(self, result):
        from ni_rua.perception.scene.scene_contracts import validate_response
        json.dumps(result, allow_nan=False)
        canonical(result)
        validate_response("geometry/query", result, method="POST")

    def test_project_points_reports_the_folded_point_as_outside_the_image(self):
        identifier = self.distorted(0, "obs-wrist-d")
        result = self.geometry.execute(dict(self.identity, kind="project_points", observation_id=identifier,
                                            camera_id="head", points_m=self.local_tips, connect_points=True))
        self.assertEqual(result["status"], "supported", result)
        value = result["value"]
        self.assertEqual(value["in_image"], [True, False])
        self.assertIsNone(value["pixels"][1])
        np.testing.assert_allclose(value["pixels"][0], FIXTURE["observed_before_fix"]["wrist_fixed_tip_px"], atol=.05)
        self.assertTrue(all(depth > 0 for depth in value["depth_m"]))
        self.check(result)

    def test_heldout_alignment_leaves_an_unlocated_point_out_of_the_statistics(self):
        built = self.geometry.execute(dict(self.identity, kind="build_scene", camera_id="head",
            observation_ids=["obs-0", "obs-1"], static_scene_assumption=True, base_scene_version=0,
            roi=[100, 80, 400, 320]))
        self.scene.apply_patch(built["value"]["proposal"])
        identifier = self.distorted(2, "obs-2-wrist-d")
        offset = np.array([.05, 0, 0])  # obs-2's camera sits at x = 0.05 m
        inside, beyond = (np.array(tip) + offset for tip in self.local_tips)
        request = dict(self.identity, kind="alignment", camera_id="head", observation_ids=[identifier],
                       scene_version=1, points_m=[inside.tolist(), beyond.tolist()],
                       pixels=[[422., 252.], [232., 407.]])
        result = self.geometry.execute(request)
        self.assertEqual(result["status"], "supported", result)
        view = result["value"]["views"][0]
        self.assertIsNone(view["errors_px"][1])
        self.assertIsNone(view["projected_pixels"][1])
        self.assertEqual((view["count"], view["attempted_count"], view["coverage"]), (1, 2, .5))
        self.assertAlmostEqual(view["max_px"], view["errors_px"][0])
        self.assertEqual(result["value"]["count"], 1)
        self.check(result)
        only_beyond = self.geometry.execute(dict(request, points_m=[beyond.tolist()], pixels=[[232., 407.]]))
        self.assertEqual((only_beyond["status"], only_beyond["reason"]),
                         ("unknown", "insufficient_heldout_correspondences"))


if __name__ == "__main__":
    unittest.main()
