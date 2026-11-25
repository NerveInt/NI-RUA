"""SO-101 held / self perception (held_perception.SO101Body) and the object-table cross-view check.

Synthetic geometry (the packaged SO-101 URDF + meshes, FK, a pinhole camera, a fake VGGT service) plus one numeric
real-arm regression (tests/fixtures/so101_wrist_holding.json: a measured joint state, the wrist-camera calibration
and the outline of the held cube; no image). The ARX-X5 path is covered by test_held_perception / test_object_table.
python -m unittest test_so101_held_perception"""

NI_RUA_TEST_TIER = "component"

import base64
import dataclasses
import functools
import json
from support.paths import CONFIGS_ROOT, FIXTURES_ROOT, REPO_ROOT
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from ni_rua.core.config import Config
from ni_rua.core.contracts import canonical
from ni_rua.core.storage.store import Store
from ni_rua.perception.external import held_perception as hp
from ni_rua.perception.external import object_table, obs3d
from ni_rua.perception.scene.store import SceneStore
from ni_rua.robot.model import embodiments, so101_gripper_geometry as cad, so101_kinematics as kin
from ni_rua.robot.model.geometry import grasp_reference
from ni_rua.runtime.tools.definitions import public_definitions
from support.geometry_fixture import IDENTITY
from support.perception_fixtures import HeldService
from support.perception_fixtures import FakeService, fake_object

REPOSITORY = REPO_ROOT
FIXTURE = json.loads((FIXTURES_ROOT / "so101_wrist_holding.json").read_text())
JOINTS = [*kin.ARM_JOINTS, kin.GRIPPER_JOINT]
K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
EMPTY_CLOSE_M = 0.0051


def so101_profile(empty_close=EMPTY_CLOSE_M, full_open=None):
    """The example deployment's profile with ``empty_close`` / ``full_open`` (m; None = not measured / the CAD
    opening at jaw_open_rad) as the measured gripper openings."""
    from ni_rua.robot.backends.so101.deployment import embodiment_profile, validate_deployment
    document = json.loads((REPOSITORY / "configs" / "so101.example.json").read_text())
    document["gripper"]["empty_close_aperture_m"] = empty_close
    document["gripper"]["full_open_aperture_m"] = full_open
    return embodiment_profile(validate_deployment(document, base=REPOSITORY / "configs"))


PROFILE = so101_profile()
BODY = hp.body_of(PROFILE)
MODEL = kin.load_model()


def arm_state(q_arm, jaw_rad, goal):
    tcp = kin.fk_all(MODEL, q_arm, gripper_rad=jaw_rad)[kin.TCP_LINK]
    quaternion = cv2.Rodrigues(tcp[:3, :3])[0].reshape(3)
    angle = float(np.linalg.norm(quaternion))
    axis = quaternion / angle if angle > 1e-12 else np.array([1.0, 0.0, 0.0])
    xyzw = [*(axis * np.sin(angle / 2)).tolist(), float(np.cos(angle / 2))]
    return dict(joint_names=list(JOINTS), q_rad=[*map(float, q_arm), float(jaw_rad)], source_kind="measurement",
                gripper=dict(goal_normalized=goal, command_normalized=goal),
                ee_pose=dict(frame_id="so101_base", position_m=tcp[:3, 3].tolist(), quaternion_xyzw=xyzw,
                             source_kind="measurement", tcp_config_id="so101-gripper-frame-v1"))


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """T_world_camera of an OpenCV camera (x right, y down, z forward) at ``eye`` looking at ``target``."""
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    T[:3, 3] = eye
    return T


def project(T, point, intrinsics=K):
    pc = np.linalg.inv(T) @ np.append(np.asarray(point, float), 1.0)
    return intrinsics[0, 0] * pc[0] / pc[2] + intrinsics[0, 2], intrinsics[1, 1] * pc[1] / pc[2] + intrinsics[1, 2]


def poses_of(q_arm, jaw_rad):
    return {"right": kin.fk_all(MODEL, q_arm, gripper_rad=jaw_rad)}


def polygon_mask(points, shape=(480, 640)):
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.asarray(points, np.int32).reshape(-1, 1, 2)], 1)
    return mask.astype(bool)


class Meshes(unittest.TestCase):
    def test_packaged_archive_gives_every_meshed_link_in_metres(self):
        meshes = BODY.meshes(None)
        self.assertEqual(set(meshes), {"base_link", "shoulder_link", "upper_arm_link", "lower_arm_link", "wrist_link",
                                       "gripper_link", "moving_jaw_so101_v1_link", "wrist_camera_mount_link",
                                       "wrist_camera_link"})
        for link, (V, F) in meshes.items():
            extent = V.max(axis=0) - V.min(axis=0)
            self.assertTrue(0.01 < float(extent.max()) < 0.25, (link, extent))    # the mm-scaled mount included
            self.assertTrue(100 < len(F) < 8000, (link, len(F)))                 # decimated to 2 mm cells
            self.assertLess(int(F.max()), len(V))
        self.assertIs(BODY.meshes(None), meshes)                                 # cached per archive
        # the fixed jaw is part of gripper_link: its inner fingertip (TCP frame) lies on that mesh
        tcp = kin.fk_all(MODEL, np.zeros(5))[kin.TCP_LINK]
        gripper = kin.fk_all(MODEL, np.zeros(5))["gripper_link"]
        tip = np.linalg.inv(gripper) @ tcp @ np.append(cad.fingertip_reference(0.5)["fixed_tip_m"], 1.0)
        self.assertLess(float(np.linalg.norm(meshes["gripper_link"][0] - tip[:3], axis=1).min()), 0.004)

    def test_bodies_by_robot_model(self):
        self.assertIsInstance(hp.body_of(embodiments.ARX_X5_PROFILE), hp.X5Body)
        self.assertIsInstance(BODY, hp.SO101Body)
        self.assertIsNone(hp.body_of(SimpleNamespace(robot_model_id="other")))
        runtime = SimpleNamespace(embodiment=PROFILE)
        self.assertEqual(hp.body_and_profile(runtime), (BODY, PROFILE))
        body, profile = hp.body_and_profile(SimpleNamespace())
        self.assertIsInstance(body, hp.X5Body)
        self.assertIs(profile, embodiments.ARX_X5_PROFILE)


class RobotMask(unittest.TestCase):
    Q = np.array([0.0, -0.3, 0.6, 1.2, 0.0])     # jaws pointing down and forward over the table
    JAW = 0.8

    def setUp(self):
        self.poses = poses_of(self.Q, self.JAW)
        tcp = self.poses["right"][kin.TCP_LINK]
        self.tips = cad.fingertip_reference(self.JAW)
        self.world = lambda local: tcp[:3, 3] + tcp[:3, :3] @ np.asarray(local, float)
        approach = tcp[:3, :3] @ np.asarray(PROFILE.gripper.approach_axis_local, float)
        self.grasp = self.world(self.tips["midpoint_m"])
        # a camera 0.3 m beside the open jaws (along the axis perpendicular to the approach axis and the jaw line)
        third = tcp[:3, :3] @ BODY.third_local(PROFILE)
        self.T = look_at(self.grasp + 0.3 * third, self.grasp, up=-approach)
        self.camera = dict(K=K, T=self.T, width=640, height=480)

    def mask(self, skip=None):
        posed = hp.posed_links(BODY.meshes(None), self.poses, skip_camera_of=skip, camera_links=BODY.camera_links)
        return BODY.mask(self.camera, posed)

    def test_jaws_are_robot_and_the_space_between_them_is_not(self):
        mask = self.mask()
        for name in ("fixed_tip_m", "moving_tip_m"):
            u, v = project(self.T, self.world(self.tips[name]))
            self.assertTrue(mask[int(v), int(u)], name)
        u, v = project(self.T, self.grasp)
        gap_px = K[0, 0] * self.tips["opening_m"] / 0.3
        self.assertGreater(gap_px, 100)                    # the jaws are > 100 px apart in this view
        half = int(gap_px / 4)
        window = mask[int(v) - half:int(v) + half, int(u) - half:int(u) + half]
        self.assertFalse(window.any())                     # an object held there keeps its pixels
        # the 6 cm capsule around the joint chain (the path without held perception) covers that whole window
        self.assertLess(float(np.linalg.norm(self.grasp - self.poses["right"][kin.TCP_LINK][:3, 3])), 0.06)

    def test_dilation_is_the_body_value_and_the_own_wrist_camera_is_not_rendered(self):
        posed = hp.posed_links(BODY.meshes(None), self.poses, camera_links=BODY.camera_links)
        bare = hp.render_mask(K, self.T, 640, 480, posed, dilate_px=0)
        mask = self.mask()
        self.assertTrue(mask[bare].all())
        grown = cv2.dilate(bare.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))).astype(bool)
        self.assertTrue((mask == grown).all())             # exactly BODY.dilate_px = 5 px
        self.assertEqual(BODY.dilate_px, 5)
        links = len(posed)
        skipped = hp.posed_links(BODY.meshes(None), self.poses, skip_camera_of="right", camera_links=BODY.camera_links)
        self.assertEqual(links - len(skipped), len(BODY.camera_links))
        # the X5 default camera links do not remove SO-101 links
        self.assertEqual(len(hp.posed_links(BODY.meshes(None), self.poses, skip_camera_of="right")), links)

    def test_real_wrist_pose_keeps_the_held_cube_and_masks_the_jaws(self):
        fixture = FIXTURE
        state = dict(joint_names=fixture["joint_names"], q_rad=fixture["q_rad"])
        poses = {"right": kin.fk_all(MODEL, fixture["q_rad"][:5], gripper_rad=fixture["q_rad"][5])}
        T = poses["right"][fixture["link_name"]] @ np.asarray(fixture["T_link_camera"])
        np.testing.assert_allclose(T, fixture["T_world_camera"], atol=1e-6)      # FK of the measured joints
        camera = dict(K=np.asarray(fixture["K"]), T=T, width=640, height=480)
        posed = hp.posed_links(BODY.meshes(None), poses, skip_camera_of="right", camera_links=BODY.camera_links)
        mask = BODY.mask(camera, posed)
        cube = np.zeros((480, 640), bool)
        for outline in fixture["cube_outlines_px"]:
            cube |= polygon_mask(outline)
        self.assertGreater(int(cube.sum()), 20000)
        self.assertLess(float(mask[cube].mean()), 0.25)    # service rule: robot when >= 0.5 -> the cube is kept
        for u, v in fixture["jaw_px"]:
            self.assertTrue(mask[v, u], (u, v))
        self.assertTrue(0.08 * 640 * 480 < int(mask.sum()) < 0.20 * 640 * 480, int(mask.sum()))
        # the capsule path (6 cm around the joint-chain segments, as the service draws it) covers the whole cube
        capsules = np.zeros((480, 640), np.uint8)
        Ti = np.linalg.inv(T)
        for a, b in np.asarray(fixture["robot_segments_m"]):
            for q in np.linspace(0.0, 1.0, 12):
                pc = Ti[:3, :3] @ (a + q * (b - a)) + Ti[:3, 3]
                if pc[2] > 0.05:
                    u, v = project(T, a + q * (b - a), camera["K"])
                    cv2.circle(capsules, (int(u), int(v)), int(np.ceil(camera["K"][0, 0] * 0.06 / pc[2])), 1, -1)
        self.assertGreater(float(capsules.astype(bool)[cube].mean()), 0.99)
        # the jaws hold the cube (goal 0, stalled 24 mm above the empty close); the grasp centre is on the cube
        profile = so101_profile(fixture["empty_close_aperture_m"])
        arm = dict(state, gripper=dict(goal_normalized=fixture["goal_normalized"]))
        self.assertTrue(BODY.holding(profile, "right", arm))
        excess, measured = BODY.opening_excess_m(profile, "right", arm)
        self.assertAlmostEqual(excess, 0.0235, delta=0.002)
        tcp = poses["right"][kin.TCP_LINK]
        grasp = tcp[:3, 3] + tcp[:3, :3] @ BODY.grasp_local(profile, "right", arm)
        u, v = project(T, grasp, camera["K"])
        hull = cv2.convexHull(np.asarray([p for outline in fixture["cube_outlines_px"] for p in outline], np.float32))
        self.assertGreater(cv2.pointPolygonTest(hull, (float(u), float(v)), True), 20.0)   # well inside the cube


class Holding(unittest.TestCase):
    def state(self, jaw_rad, goal):
        return dict(joint_names=list(JOINTS), q_rad=[0.0, 0.0, 0.0, 0.0, 0.0, jaw_rad], gripper=dict(goal_normalized=goal))

    def test_stalled_closed_jaw_holds_and_free_or_empty_jaw_does_not(self):
        closed = cad.jaw_rad_for_opening(EMPTY_CLOSE_M)
        opened = cad.jaw_rad_for_opening(PROFILE.gripper.full_open_aperture_m)
        at = lambda goal: closed + goal * (opened - closed)
        cases = [(closed, 0.0, False),                    # empty close
                 (0.156, 0.0, True),                      # goal 0 stalled on a 29 mm object
                 (at(0.25), 0.25, False),                 # partial close reached in free air
                 (at(0.25) - 0.03, 0.25, False),          # 2-3 mm short of the commanded opening (servo error)
                 (at(0.25) + 0.15, 0.25, True),           # partial close stalled ~12 mm above the command
                 (0.156, 0.6, False),                     # not commanded closed
                 (opened, 0.0, False)]                    # fully open
        for jaw, goal, expected in cases:
            self.assertEqual(BODY.holding(PROFILE, "right", self.state(jaw, goal)), expected, (jaw, goal))
        self.assertEqual(hp.holding_arms({"arms": {"right": self.state(0.156, 0.0)}}, PROFILE), ["right"])
        self.assertEqual(hp.holding_arms({"arms": {"right": self.state(closed, 0.0)}}, PROFILE), [])
        self.assertEqual(hp.holding_arms({"arms": {"right": self.state(0.156, 0.0)}},
                                         SimpleNamespace(robot_model_id="other", arms=("right",))), [])

    def test_margin_is_the_profile_non_empty_preset(self):
        margin = PROFILE.gripper.preset_margins_m["non_empty"][0]
        closed = cad.jaw_rad_for_opening(EMPTY_CLOSE_M)
        low, high = closed, closed + 0.3
        for _ in range(50):                               # jaw angle whose opening is empty close + margin
            mid = (low + high) / 2
            low, high = (mid, high) if cad.opening_width_m(mid) < EMPTY_CLOSE_M + margin else (low, mid)
        self.assertFalse(BODY.holding(PROFILE, "right", self.state(low - 0.005, 0.0)))
        self.assertTrue(BODY.holding(PROFILE, "right", self.state(high + 0.005, 0.0)))

    def test_unknown_inputs_never_hold(self):
        self.assertFalse(BODY.holding(so101_profile(None), "right", self.state(0.156, 0.0)))   # no measured empty close
        self.assertFalse(BODY.holding(PROFILE, "right", dict(self.state(0.156, 0.0), joint_names=list(kin.ARM_JOINTS))))
        self.assertFalse(BODY.holding(PROFILE, "right", dict(joint_names=list(JOINTS), q_rad=[0.0] * 6)))
        self.assertFalse(BODY.holding(PROFILE, "right", {}))

    def test_undecided_names_why_the_rule_cannot_decide(self):
        stalled = self.state(0.156, 0.0)                  # would hold with a usable profile
        cases = [(so101_profile(None), stalled, "empty_close_aperture_m_not_measured"),
                 (PROFILE, dict(stalled, joint_names=list(kin.ARM_JOINTS)), "no_measured_jaw_angle"),
                 (PROFILE, dict(joint_names=list(JOINTS), q_rad=[0.0] * 6), "no_goal_normalized"),
                 (PROFILE, {}, "no_goal_normalized"),
                 # measured openings outside the CAD opening range of the declared jaw limits (3.1..137.2 mm)
                 (so101_profile(0.0025), stalled, "empty_close_aperture_m_outside_cad_opening_range"),
                 (so101_profile(EMPTY_CLOSE_M, 0.15), stalled, "full_open_aperture_m_outside_cad_opening_range")]
        for profile, state, reason in cases:
            self.assertEqual(BODY.undecided(profile, "right", state), reason)
            self.assertFalse(BODY.holding(profile, "right", state))
            self.assertIsNone(BODY.opening_excess_m(profile, "right", state))
        # decided: a usable reading either way, or a jaw that is not commanded closed (whatever the profile lacks)
        self.assertIsNone(BODY.undecided(PROFILE, "right", stalled))
        self.assertIsNone(BODY.undecided(PROFILE, "right", self.state(cad.jaw_rad_for_opening(EMPTY_CLOSE_M), 0.0)))
        self.assertIsNone(BODY.undecided(so101_profile(None), "right", self.state(0.156, 0.6)))
        self.assertIsNone(hp.X5Body().undecided(embodiments.ARX_X5_PROFILE, "left", {}))

    def test_archived_real_readings(self):
        # tests/fixtures/so101_jaw_readings.json: 103 distinct (goal, measured jaw angle) readings of 468 real
        # observations. With the jaw commanded closed, the readings where the jaw reached its command end at least
        # 1.5 mm below the non_empty margin (1.5 mm above the commanded opening) and the stalled ones far above it.
        data = json.loads((FIXTURES_ROOT / "so101_jaw_readings.json").read_text())
        profile = so101_profile(data["empty_close_aperture_m"], data["full_open_aperture_m"])
        margin = profile.gripper.preset_margins_m["non_empty"][0]
        self.assertEqual((margin, sum(count for _, _, count in data["readings"])), (0.0015, 468))
        excess = {}
        for goal, jaw, _ in data["readings"]:
            state = self.state(jaw, goal)
            found, _ = BODY.opening_excess_m(profile, "right", state)
            band = "closed" if goal == 0.0 else "partial" if goal < 0.5 else "open"
            holding = BODY.holding(profile, "right", state)
            excess.setdefault((band, holding), []).append(found)
            self.assertIsNone(BODY.undecided(profile, "right", state))
        self.assertEqual(sorted(excess), [("closed", False), ("closed", True), ("open", False), ("partial", False)])
        self.assertLess(max(excess[("closed", False)]), 0.0001)       # empty goal-0 closes: -1.5..0.0 mm
        self.assertGreater(min(excess[("closed", True)]), 0.009)      # stalled on an object: +9.2..+24.9 mm
        self.assertLess(max(excess[("partial", False)]), -0.0001)     # 0 < goal < 0.5: -4.3..-0.2 mm
        self.assertGreater(min(excess[("partial", False)]), -0.0045)

    def test_grasp_frame_is_the_profile_reference(self):
        state = self.state(0.4, 0.0)
        local = BODY.grasp_local(PROFILE, "right", state)
        np.testing.assert_allclose(local, cad.fingertip_reference(0.4)["midpoint_m"])
        np.testing.assert_allclose(local, grasp_reference(PROFILE.gripper, np.zeros(3), np.eye(3), 0.4)[0])
        facts = BODY.closure_facts(PROFILE, "right", state)
        np.testing.assert_allclose(facts["grasp_local"], local)
        self.assertEqual(facts["approach_local"], [0.0, 0.0, 1.0])
        self.assertEqual(facts["closing_local"], [-1.0, 0.0, 0.0])
        np.testing.assert_allclose(np.abs(facts["third_local"]), [0.0, 1.0, 0.0])
        self.assertEqual(hp.X5Body().closure_facts(embodiments.ARX_X5_PROFILE, "left", {}), {})


@functools.lru_cache(maxsize=None)
def _default_so101_observation_png():
    rng = np.random.default_rng(4)
    return cv2.imencode(".png", cv2.GaussianBlur(rng.integers(0, 256, (480, 640, 3), dtype=np.uint8), (5, 5), 0))[1].tobytes()


def so101_observation(store, q_arm, jaw_rad, goal, *, step=5, observation_id="obs-so101"):
    """A two-camera SO-101 observation: fixed overview camera above the table, wrist camera on gripper_link."""
    image = _default_so101_observation_png()
    fixed = look_at([0.25, -0.25, 0.45], [0.22, 0.0, 0.0])
    calibration = {"reference_frame_id": "so101_base", "joint_calibration": {"table_z_m": 0.0}, "cameras": {
        "fixed": {"K": K.tolist(), "distortion": [0.0] * 5, "distortion_model": "opencv_plumb_bob",
                  "image_size": [640, 480], "T_world_camera": fixed.tolist()},
        "wrist": {"K": K.tolist(), "distortion": [0.0] * 5, "distortion_model": "opencv_plumb_bob",
                  "image_size": [640, 480], "arm": "right", "link_name": FIXTURE["link_name"],
                  "T_link_camera": FIXTURE["T_link_camera"]}}}
    calibration_ref = store.put_artifact(canonical(calibration).encode(), "calibration", "application/json")
    model_ref = store.put_artifact(canonical(MODEL).encode(), "robot_model", "application/json")
    state = {"arms": {"right": arm_state(q_arm, jaw_rad, goal)}, "frame_id": "so101_base", "sim_step": step,
             "step_domain": "so101_control_step", "tcp_config_id": "so101-gripper-frame-v1"}
    state_ref = store.put_artifact(canonical(state).encode(), "robot_state", "application/json")
    links = kin.fk_all(MODEL, q_arm, gripper_rad=jaw_rad)
    link = links[FIXTURE["link_name"]]
    sample = {"reference_frame_id": "so101_base", "sim_step": step, "step_domain": "so101_control_step", "cameras": {
        "fixed": {"T_world_camera": fixed.tolist(), "source_kind": "static_extrinsic_calibration"},
        "wrist": {"T_world_base": np.eye(4).tolist(), "T_world_link": link.tolist(), "arm": "right",
                  "T_world_camera": (link @ np.asarray(FIXTURE["T_link_camera"])).tolist(),
                  "link_name": FIXTURE["link_name"], "source_kind": "robot_kinematics_at_joint_read"}}}
    sample_ref = store.put_artifact(canonical(sample).encode(), "camera_pose_sample", "application/json")
    cameras = [{"camera_id": name, "frame_id": "frame-%s-%s" % (observation_id, name), "width": 640, "height": 480,
                "sim_step": step, "step_domain": "so101_control_step",
                "rgb_ref": store.put_artifact(image, "rgb", "image/png"), "calibration_ref": calibration_ref,
                "pose_sample_ref": sample_ref, "pixel_transform": "identity"} for name in ("fixed", "wrist")]
    return {"schema_version": "0.1", **IDENTITY, "world_kind": "observed", "data_origin": "synthetic_fixture",
            "observation_id": observation_id, "sim_step": step, "step_domain": "so101_control_step",
            "clock_domain": "synthetic/generation-1", "calibration_id": "synthetic-calibration-v1",
            "robot_model_id": "so101", "robot_model_ref": model_ref,
            "observation_profile": "rgb_proprio_calibrated_synthetic_v1", "robot_state_ref": state_ref,
            "cameras": cameras}, fixed


class So101Scene:
    """A store with one synthetic SO-101 observation and a runtime on a fake service (mixed into the test cases)."""

    Q = np.array([0.0, -0.3, 0.6, 1.2, 0.0])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.artifacts = Store(self.tmp.name, IDENTITY); self.addCleanup(self.artifacts.close)
        self.scene = SceneStore(Path(self.tmp.name) / "scene", self.artifacts.resolve_artifact, self.artifacts.put_artifact)

    def observe(self, jaw_rad, goal, record=False, **kwargs):
        """Imports one observation; ``record`` also saves its observation record, which held_perception.track reads."""
        observation, self.fixed = so101_observation(self.artifacts, self.Q, jaw_rad, goal, **kwargs)
        self.scene.import_observation(observation)
        if record:
            self.artifacts.save("observation", observation["observation_id"], observation)
        return observation

    def runtime(self, endpoint, profile=PROFILE, held=True):
        config = Config().with_overrides({"features": {
            "external_perception": False, "object_table": True, "held_perception": held, "vggt_endpoint": endpoint,
            "vggt_support_z_m": None, "vggt_timeout_s": 5.0}})
        return SimpleNamespace(config=config, scene=self.scene, image_settings=None, identity=dict(IDENTITY),
                               store=self.artifacts, features=config.features, embodiment=profile)

    def query(self, runtime, **request):
        return object_table.object_table_query(runtime, {"kind": "object_table", "observation_id": "obs-so101", **request})


class Integration(So101Scene, unittest.TestCase):
    def test_profile_offers_held_perception_and_cross_view(self):
        self.assertIn("held_perception", PROFILE.supports)
        self.assertIn("cross_view", PROFILE.supports)
        self.assertNotIn("cross_view", embodiments.ARX_X5_PROFILE.supports)
        runtime = self.runtime("http://127.0.0.1:9")
        self.assertTrue(hp.enabled(runtime) and object_table.held_perception_enabled(runtime))
        self.assertFalse(hp.enabled(self.runtime("http://127.0.0.1:9", held=False)))
        self.assertEqual(hp.instruction(runtime.config.features, PROFILE), hp.INSTRUCTION)

    def test_free_arm_sends_the_mesh_mask_without_capsule_segments(self):
        self.observe(0.8, 0.5)
        fake = HeldService(objects=[fake_object((0.2, 0.0, 0.015), 0.03, 0.0)])
        self.addCleanup(fake.close)
        result, _ = self.query(self.runtime(fake.url))
        self.assertEqual(result["status"], "conditional", result.get("reason"))
        self.assertEqual([path for path, _ in fake.requests], ["/object_table"])
        body = fake.requests[0][1]
        self.assertEqual(body["robot_segments_m"], [])      # the service's pixel rule decides, not the capsules
        self.assertNotIn("held", body)
        mask = np.unpackbits(np.frombuffer(base64.b64decode(body["robot_mask_b64"]), np.uint8))[:480 * 640]
        mask = mask.reshape(480, 640).astype(bool)
        self.assertEqual(body["robot_mask_shape"], [480, 640])
        poses = poses_of(self.Q, 0.8)["right"]
        for link in ("lower_arm_link", "gripper_link", "moving_jaw_so101_v1_link"):
            V = BODY.meshes(None)[link][0]
            centre = poses[link][:3, 3] + poses[link][:3, :3] @ V.mean(axis=0)
            u, v = project(self.fixed, centre)
            self.assertTrue(mask[int(v), int(u)], link)
        tcp = poses[kin.TCP_LINK]
        grasp = tcp[:3, 3] + tcp[:3, :3] @ np.asarray(cad.fingertip_reference(0.8)["midpoint_m"])
        below = np.array([grasp[0], grasp[1], 0.0])          # the table point under the open jaws
        self.assertLess(float(np.linalg.norm(below - tcp[:3, 3])), 0.06)
        diagnostics = result["diagnostics"]["held_perception"]
        self.assertEqual((diagnostics["mesh_links"], diagnostics["arms"], diagnostics["holding"]), (9, ["right"], []))
        self.assertEqual(diagnostics["robot_mask_px"], int(mask.sum()))
        self.assertNotIn("held", result["value"])

    def test_without_the_tag_the_capsule_path_is_unchanged(self):
        self.observe(0.8, 0.5)
        fake = HeldService(objects=[fake_object((0.2, 0.0, 0.015), 0.03, 0.0)])
        self.addCleanup(fake.close)
        plain = dataclasses.replace(PROFILE, supports=PROFILE.supports - {"held_perception"})
        result, _ = self.query(self.runtime(fake.url, profile=plain))
        body = fake.requests[0][1]
        self.assertNotIn("robot_mask_b64", body)
        self.assertTrue(body["robot_segments_m"])
        self.assertNotIn("held_perception", result["diagnostics"])

    def test_holding_arm_gets_a_held_row_at_the_fingertip_midpoint(self):
        self.observe(0.156, 0.0)
        poses = poses_of(self.Q, 0.156)["right"]
        tcp = poses[kin.TCP_LINK]
        grasp = tcp[:3, 3] + tcp[:3, :3] @ np.asarray(cad.fingertip_reference(0.156)["midpoint_m"])
        held = dict(status="ok", arm="right", mask_source="sam31_text", mask_px=900, centre_m=list(grasp + [0.004, 0, 0]),
                    dist_px=0.0, grasp_px=[320.0, 240.0], grasp_depth_m=0.4, object_index=None,
                    contour_px=[[300, 230], [340, 230], [340, 250], [300, 250]], label_px=[320, 240],
                    gk=dict(top_z_m=grasp[2] + 0.015, low_z_m=grasp[2] - 0.015, length_m=0.03, width_m=0.03,
                            yaw_deg=10.0, elongation=1.0, axis_tilt_deg=5.0))
        fake = HeldService(held_head=held, held_wrist=dict(held, mask_px=2500),
                           objects=[fake_object((0.3, 0.1, 0.015), 0.03, 0.0)])
        self.addCleanup(fake.close)
        runtime = self.runtime(fake.url)
        result, _ = self.query(runtime)
        self.assertEqual([path for path, _ in fake.requests], ["/object_table", "/held"])
        item = fake.requests[0][1]["held"][0]
        self.assertEqual(item["arm"], "right")
        np.testing.assert_allclose(item["grasp_centre_m"], grasp, atol=1e-9)
        self.assertEqual(fake.requests[1][1]["held"][0]["grasp_centre_m"], item["grasp_centre_m"])
        value = result["value"]
        row = dict(zip(value["columns"], value["rows"][-1]))
        self.assertEqual((row["support"], row["quality"]), ("held_right", "held_fk_depth"))
        np.testing.assert_allclose(row["centre_m"], np.round(grasp, 3), atol=1e-9)   # the FK grasp centre
        meta = value["held"][row["id"]]
        self.assertEqual((meta["arm"], meta["camera_id"], meta["centre"], meta["group"]), ("right", "wrist", "grasp", "compact"))
        self.assertIsNone(meta["uncertainty_p50_mm"])      # the X5 truth-check statistics are not quoted
        self.assertIn("centre error not measured on this robot", meta["note"])
        self.assertNotIn("p50", meta["note"])
        self.assertEqual(meta["retention_status"], "unverified")
        self.assertNotIn(row["id"], value["grasp_hints"])
        self.assertEqual(result["diagnostics"]["held_perception"]["holding"], ["right"])

    def test_undecided_jaw_rule_is_a_diagnostic(self):
        self.observe(0.156, 0.0)                           # stalled closed jaw, but no measured empty close
        fake = HeldService(objects=[fake_object((0.3, 0.1, 0.015), 0.03, 0.0)])
        self.addCleanup(fake.close)
        result, _ = self.query(self.runtime(fake.url, profile=so101_profile(None)))
        self.assertEqual(result["status"], "conditional", result.get("reason"))
        diagnostics = result["diagnostics"]["held_perception"]
        self.assertEqual(diagnostics["holding_undecided"], {"right": "empty_close_aperture_m_not_measured"})
        self.assertEqual(diagnostics["holding"], [])
        self.assertNotIn("held", result["value"])
        self.assertEqual([path for path, _ in fake.requests], ["/object_table"])
        # a decided jaw (here: open) adds no entry
        self.observe(0.8, 0.5, observation_id="obs-open")
        result, _ = self.query(self.runtime(fake.url, profile=so101_profile(None)), observation_id="obs-open")
        self.assertNotIn("holding_undecided", result["diagnostics"]["held_perception"])

    def test_carried_pregrasp_row_uses_the_so101_grasp_frame(self):
        # closure state kept by track(): the SO-101 TCP axes and the fingertip midpoint at the measured jaw angle
        state = arm_state(self.Q, 0.156, 0.0)
        tcp = poses_of(self.Q, 0.156)["right"][kin.TCP_LINK]
        facts = BODY.closure_facts(PROFILE, "right", state)
        grasp = tcp[:3, 3] + tcp[:3, :3] @ np.asarray(facts["grasp_local"])
        row = dict(index=0, centre_m=(grasp + [0.004, 0.003, 0.0]).tolist(), top_z_m=grasp[2] + 0.015,
                   bottom_z_m=grasp[2] - 0.015, length_m=0.03, width_m=0.03, yaw_deg=20.0, axis_tilt_deg=None,
                   elongation=1.0, contour_px=[], flags=[], quality="ok", support="table", label=None)
        table = dict(observation_id="obs-pre", sim_step=3, camera_id="fixed", K=K.tolist(), T=np.eye(4).tolist(),
                     rows=[row], free_arms=["right"], source=dict(seq=1, request="plain"), size=[640, 480],
                     link6={"right": tcp.tolist()})
        closure = dict(observation_id="obs-c", sim_step=5, T6=tcp.tolist(), **facts)
        carried, reason = hp.capture(dict(tables=[table]), "right", closure)
        self.assertIsNone(reason)
        self.assertEqual(carried["grasp_local"], facts["grasp_local"])
        # across the jaws (TCP x) the carried centre is the jaw centre; the other components are the row's
        offset = np.asarray(carried["offset_local"]) - np.asarray(facts["grasp_local"])
        self.assertAlmostEqual(float(offset[0]), 0.0, places=9)
        obj, meta = hp.held_object({}, "right", tcp, camera_ids=dict(head="fixed", wrist="wrist"), carried=carried,
                                   grasp_local=np.asarray(facts["grasp_local"]), third_local=BODY.third_local(PROFILE),
                                   uncertainty=False)
        np.testing.assert_allclose(obj["centre_m"], grasp, atol=1e-9)      # compact group: the grasp centre
        self.assertAlmostEqual(obj["top_z_m"] - obj["bottom_z_m"], 0.03, places=6)
        self.assertEqual((meta["source"], meta["geometry_evidence"]), ("held_pregrasp_row", "prediction_only"))
        self.assertIsNone(meta["uncertainty_p50_mm"])
        self.assertIn("centre error not measured on this robot", meta["note"])


class Tracking(unittest.TestCase):
    Q = np.array([0.0, -0.3, 0.6, 1.2, 0.0])

    def test_track_keeps_the_so101_closure_frame_and_carries_the_pregrasp_row(self):
        from support.perception_fixtures import FakeRecords
        records = FakeRecords()
        runtime = SimpleNamespace(store=records, embodiment=PROFILE)
        lifted = np.array([0.0, -0.6, 0.6, 1.2, 0.0])
        records.add("o0", 0, dict(right=arm_state(lifted, 0.8, 0.5)))             # open above the object
        records.add("o1", 50, dict(right=arm_state(self.Q, 0.156, 0.0)))          # closure: stalled on the object
        records.add("o2", 100, dict(right=arm_state(lifted, 0.156, 0.0)))         # lifted, still holding
        memory = hp.track(runtime)
        streak = memory["arms"]["right"]
        self.assertTrue(streak["holding"])
        closure = streak["closure"]
        self.assertEqual((closure["observation_id"], closure["sim_step"]), ("o1", 50))
        tcp = kin.fk_all(MODEL, self.Q, gripper_rad=0.156)[kin.TCP_LINK]
        np.testing.assert_allclose(closure["T6"], tcp, atol=1e-6)                 # the measured TCP (ee_pose)
        np.testing.assert_allclose(closure["grasp_local"], cad.fingertip_reference(0.156)["midpoint_m"])
        self.assertEqual(closure["closing_local"], [-1.0, 0.0, 0.0])
        grasp = tcp[:3, 3] + tcp[:3, :3] @ np.asarray(closure["grasp_local"])
        row = fake_object((grasp[0] + 0.003, grasp[1], grasp[2]), grasp[2] + 0.015, grasp[2] - 0.015, elongation=1.0)
        hp.record_table(runtime, "o0", 0, "fixed", dict(K=K, T=np.eye(4)), [row],
                        {"right": {kin.TCP_LINK: kin.fk_all(MODEL, lifted)[kin.TCP_LINK]}}, ["right"],
                        tcp_link=PROFILE.tcp_link)
        self.assertIn("right", memory["tables"][0]["link6"])                      # kept under the profile's tcp_link
        carried, why = hp.carried_for(runtime, "right", 100)
        self.assertEqual((carried["observation_id"], why), ("o0", None))
        self.assertEqual(carried["grasp_local"], closure["grasp_local"])
        records.add("o3", 150, dict(right=arm_state(lifted, 0.8, 0.5)))           # release
        hp.track(runtime)
        self.assertFalse(memory["arms"]["right"]["holding"])
        self.assertNotIn("right", memory["carried"])


def box_outline(T, centre, size=0.03, yaw=0.0):
    """Convex outline (px) seen by camera ``T`` of a box centred at ``centre`` (m): edge ``size`` (m; a number = a
    cube, or [x, y, z] sides), turned by ``yaw`` (rad) about +z."""
    half = np.broadcast_to(np.asarray(size, float), (3,)) / 2
    R = cv2.Rodrigues(np.array([0.0, 0.0, float(yaw)]))[0]
    corners = [np.asarray(centre, float) + R @ (half * [a, b, c]) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
    pixels = np.array([project(T, corner) for corner in corners], np.float32)
    return cv2.convexHull(pixels.reshape(-1, 1, 2)).reshape(-1, 2).tolist()


class CrossView(unittest.TestCase):
    FIXED = look_at([0.25, -0.25, 0.45], [0.22, 0.0, 0.0])
    WRIST = look_at([0.10, 0.05, 0.12], [0.24, 0.0, 0.0])
    TRUTH = [np.array([0.24, 0.02, 0.015]), np.array([0.20, -0.05, 0.015]), np.array([0.60, 0.60, 0.015])]

    def camera(self, T, D=()):
        return dict(K=K, T=T, D=np.asarray(D, float))

    def rows(self, T, centres, bias=(0.0, 0.0, 0.0)):
        return [dict(centre_m=(centre + bias).tolist(), contour_px=box_outline(T, centre), label_px=[0, 0],
                     support="table", quality="ok") for centre in centres]

    def test_consistent_views_meet_and_report_the_monocular_bias(self):
        bias = np.array([0.025, 0.004, -0.006])              # the overview table is 2.5 cm off, the wrist view is right
        fixed_rows = self.rows(self.FIXED, self.TRUTH, bias)
        wrist_rows = self.rows(self.WRIST, self.TRUTH[:2])[::-1]            # other order, third object not in view
        rows, unmatched, skipped = object_table.cross_view_rows(self.camera(self.FIXED), fixed_rows,
                                                                self.camera(self.WRIST), wrist_rows)
        self.assertEqual([row["id"] for row in rows], ["o1", "o2"])
        self.assertEqual((unmatched, skipped), (["o3"], {}))
        for row, truth in zip(rows, self.TRUTH):
            np.testing.assert_allclose(row["target_view_centre_m"], truth, atol=1e-9)
            np.testing.assert_allclose(row["target_view_minus_row_m"], -bias, atol=1e-9)
            # the two outline centroids see different faces: the midpoint lies inside the 30 mm cube
            self.assertLess(float(np.linalg.norm(np.asarray(row["stereo_centre_m"]) - truth)), 0.010)
            np.testing.assert_allclose(np.asarray(row["stereo_minus_row_m"]),
                                       np.asarray(row["stereo_centre_m"]) - (truth + bias), atol=1.1e-3)
            self.assertLess(row["ray_gap_m"], 0.006)
            self.assertGreater(row["parallax_deg"], 20.0)
            self.assertNotIn("ambiguous", row)
        self.assertEqual(fixed_rows[0]["centre_m"], (self.TRUTH[0] + bias).tolist())   # nothing was corrected

    def test_miscalibrated_view_is_paired_and_its_rays_do_not_meet(self):
        # the overview camera's real pose is 5 deg off its calibration: masks come from the real pose, rays from the
        # calibrated one. Pairing by monocular centres still finds the object and the ray gap exposes the conflict.
        real = self.FIXED.copy()
        real[:3, :3] = real[:3, :3] @ cv2.Rodrigues(np.array([np.radians(5.0), 0.0, 0.0]))[0]
        fixed_rows = self.rows(real, self.TRUTH[:2], (0.03, 0.0, 0.0))
        wrist_rows = self.rows(self.WRIST, self.TRUTH[:2])
        rows, unmatched, _ = object_table.cross_view_rows(self.camera(self.FIXED), fixed_rows, self.camera(self.WRIST),
                                                          wrist_rows)
        self.assertEqual(([row["id"] for row in rows], unmatched), (["o1", "o2"], []))
        for row in rows:
            # 22 and 36 mm: more than twice the most exact calibrations leave for a 30 mm cube (0.25 x 0.03 m)
            self.assertGreater(row["ray_gap_m"], 2 * object_table.CROSS_VIEW_EXACT_GAP_MAX * 0.03)
            self.assertAlmostEqual(row["target_view_minus_row_m"][0], -0.03, places=6)

    def test_exact_calibrations_leave_a_size_dependent_ray_gap(self):
        # The two mask-centre rays pass through the centroids of masks showing different faces, so with exact
        # calibrations their gap scales with the object: the CROSS_VIEW_EXACT_* facts quoted in the assumption text.
        rng = np.random.default_rng(0)
        gap_ratios, midpoint_ratios, gaps = [], [], {}
        for dims in [(0.03, 0.03, 0.03), (0.12, 0.03, 0.03), (0.12, 0.03, 0.015), (0.06, 0.04, 0.04),
                     (0.03, 0.03, 0.10), (0.08, 0.08, 0.01)]:
            dims = np.asarray(dims)
            for _ in range(150):
                centre = np.array([rng.uniform(0.15, 0.32), rng.uniform(-0.1, 0.1), dims[2] / 2])
                yaw = rng.uniform(-np.pi, np.pi)
                azimuth, elevation = rng.uniform(-np.pi, np.pi), np.radians(rng.uniform(20.0, 70.0))
                eye = centre + rng.uniform(0.10, 0.25) * np.array([np.cos(elevation) * np.cos(azimuth),
                                                                    np.cos(elevation) * np.sin(azimuth),
                                                                    np.sin(elevation)])
                wrist = look_at(eye, centre)
                src = [dict(centre_m=centre.tolist(), contour_px=box_outline(self.FIXED, centre, dims, yaw))]
                dst = [dict(centre_m=centre.tolist(), contour_px=box_outline(wrist, centre, dims, yaw))]
                rows, _, _ = object_table.cross_view_rows(self.camera(self.FIXED), src, self.camera(wrist), dst)
                if rows[0]["ray_gap_m"] is None:
                    continue
                gap_ratios.append(rows[0]["ray_gap_m"] / float(dims.max()))
                midpoint_ratios.append(float(np.linalg.norm(np.asarray(rows[0]["stereo_centre_m"]) - centre))
                                       / float(dims.max()))
                gaps.setdefault(tuple(dims), []).append(rows[0]["ray_gap_m"])
        self.assertGreater(len(gap_ratios), 800)
        self.assertLess(max(gap_ratios), object_table.CROSS_VIEW_EXACT_GAP_MAX + 0.02)
        self.assertLess(float(np.percentile(gap_ratios, 90)), object_table.CROSS_VIEW_EXACT_GAP_P90 + 0.01)
        self.assertLess(float(np.percentile(midpoint_ratios, 90)), object_table.CROSS_VIEW_EXACT_MIDPOINT_P90 + 0.01)
        self.assertLess(max(gaps[(0.03, 0.03, 0.03)]), 0.003)                    # a 30 mm cube: a few mm at most
        self.assertGreater(max(gaps[(0.12, 0.03, 0.03)]), 0.012)                 # a 120 mm bar: above 1 cm, no fault
        self.assertNotIn("0.01 m or more", object_table.CROSS_VIEW_ASSUMPTION)

    def test_pairing_bounds_ambiguity_parallax_and_held_rows(self):
        fixed_rows = self.rows(self.FIXED, self.TRUTH[:1])
        far = self.rows(self.WRIST, [self.TRUTH[0] + [0.07, 0.0, 0.0]])            # > 6 cm: another object
        self.assertEqual(object_table.cross_view_rows(self.camera(self.FIXED), fixed_rows, self.camera(self.WRIST), far),
                         ([], ["o1"], {}))
        two = self.rows(self.WRIST, [self.TRUTH[0], self.TRUTH[0] + [0.0, 0.04, 0.0]])
        rows, _, _ = object_table.cross_view_rows(self.camera(self.FIXED), fixed_rows, self.camera(self.WRIST), two)
        self.assertTrue(rows[0]["ambiguous"])
        np.testing.assert_allclose(rows[0]["target_view_centre_m"], self.TRUTH[0], atol=1e-9)
        # mutual nearest: two source rows cannot take the same target row
        both = self.rows(self.FIXED, [self.TRUTH[0], self.TRUTH[0] + [0.0, 0.03, 0.0]])
        rows, unmatched, _ = object_table.cross_view_rows(self.camera(self.FIXED), both, self.camera(self.WRIST),
                                                          self.rows(self.WRIST, self.TRUTH[:1]))
        self.assertEqual(([row["id"] for row in rows], unmatched), (["o1"], ["o2"]))
        # the same camera twice: no parallax, so no two-view point, but the pair and the offsets are reported
        rows, _, _ = object_table.cross_view_rows(self.camera(self.FIXED), fixed_rows, self.camera(self.FIXED),
                                                  self.rows(self.FIXED, self.TRUTH[:1], (0.01, 0.0, 0.0)))
        self.assertIsNone(rows[0]["stereo_centre_m"])
        self.assertIsNone(rows[0]["stereo_minus_row_m"])
        self.assertIsNone(rows[0]["ray_gap_m"])
        self.assertEqual(rows[0]["target_view_minus_row_m"], [0.01, 0.0, 0.0])
        self.assertEqual(object_table.cross_view_rows(self.camera(self.FIXED), fixed_rows, self.camera(self.WRIST), []),
                         ([], ["o1"], {}))

    def test_rows_that_are_not_compared_are_listed_with_the_reason(self):
        # every source id is in exactly one of rows, unmatched and skipped
        fixed_rows = self.rows(self.FIXED, self.TRUTH[:3])
        source = [dict(fixed_rows[0], support="held_right", quality="held_fk_depth"),    # o1 held row
                  fixed_rows[1],                                                          # o2 paired
                  dict(fixed_rows[2]),                                                    # o3 outside the wrist view
                  dict(fixed_rows[1], centre_m=None),                                     # o4 no centre
                  dict(fixed_rows[1], contour_px=[], label_px=None)]                      # o5 no mask pixel
        wrist_rows = self.rows(self.WRIST, self.TRUTH[:2])
        wrist_rows.append(dict(wrist_rows[1], centre_m=(self.TRUTH[1] + [0.01, 0.0, 0.0]).tolist(),
                               support="held_right", quality="held_fk_depth"))   # 1 cm from o2, never a partner
        rows, unmatched, skipped = object_table.cross_view_rows(self.camera(self.FIXED), source,
                                                                self.camera(self.WRIST), wrist_rows)
        self.assertEqual([row["id"] for row in rows], ["o2"])
        self.assertEqual(unmatched, ["o3"])
        self.assertEqual(skipped, {"o1": "held_row", "o4": "no_centre", "o5": "no_mask_pixel"})
        self.assertNotIn("ambiguous", rows[0])                    # the wrist held row is not a second candidate

    def test_distorted_pixels_are_undistorted_before_triangulation(self):
        D = [-0.25, 0.08, 0.001, -0.001, 0.0]
        def distorted(T, centre):
            outline = np.asarray(box_outline(T, centre), np.float64)
            rays = np.column_stack([(outline[:, 0] - K[0, 2]) / K[0, 0], (outline[:, 1] - K[1, 2]) / K[1, 1],
                                    np.ones(len(outline))])
            return cv2.projectPoints(rays, np.zeros(3), np.zeros(3), K, np.asarray(D))[0].reshape(-1, 2).tolist()
        truth = self.TRUTH[0]
        src = [dict(centre_m=truth.tolist(), contour_px=distorted(self.FIXED, truth), support="table")]
        dst = [dict(centre_m=truth.tolist(), contour_px=distorted(self.WRIST, truth), support="table")]
        rows, _, _ = object_table.cross_view_rows(self.camera(self.FIXED, D), src, self.camera(self.WRIST, D), dst)
        self.assertLess(float(np.linalg.norm(np.asarray(rows[0]["stereo_centre_m"]) - truth)), 0.010)
        self.assertLess(rows[0]["ray_gap_m"], 0.006)
        ignored, _, _ = object_table.cross_view_rows(self.camera(self.FIXED), src, self.camera(self.WRIST), dst)
        self.assertGreater(float(np.linalg.norm(np.asarray(ignored[0]["stereo_centre_m"]) - truth)),
                           float(np.linalg.norm(np.asarray(rows[0]["stereo_centre_m"]) - truth)))

    def test_mask_centre_is_the_outline_centroid_else_the_label_pixel(self):
        self.assertEqual(object_table.mask_centre_px(dict(contour_px=[[0, 0], [10, 0], [10, 20], [0, 20]], label_px=[1, 1])),
                         [5.0, 10.0])
        self.assertEqual(object_table.mask_centre_px(dict(contour_px=[[3, 4]], label_px=[7, 9])), [7.0, 9.0])
        self.assertIsNone(object_table.mask_centre_px(dict(contour_px=[])))


class CrossViewQuery(So101Scene, unittest.TestCase):
    """geometry_query kind object_table with target_camera_ids on the SO-101 profile."""

    def rows_for(self, T, centres, bias=(0.0, 0.0, 0.0)):
        out = []
        for centre in centres:
            outline = box_outline(T, centre)
            item = fake_object(tuple(np.asarray(centre) + bias), 0.03, 0.0, elongation=1.0)
            item.update(contour_px=[[int(round(u)), int(round(v))] for u, v in outline],
                        label_px=[int(round(np.mean([p[0] for p in outline]))), int(round(np.mean([p[1] for p in outline])))])
            out.append(item)
        return out

    def test_target_camera_table_is_computed_and_compared(self):
        self.observe(0.8, 0.5)
        runtime = self.runtime("http://127.0.0.1:9", held=False)
        wrist_T = GeometryTools_camera(runtime, "wrist")["T"]
        truth = [np.array([0.26, 0.03, 0.015])]
        answers = {"fixed": self.rows_for(self.fixed, truth, (0.02, 0.0, 0.0)), "wrist": self.rows_for(wrist_T, truth)}
        fake = CameraService(answers, {"fixed": self.fixed, "wrist": wrist_T})
        self.addCleanup(fake.close)
        runtime = self.runtime(fake.url, held=False)
        result, overlays = self.query(runtime, target_camera_ids=["wrist"])
        self.assertEqual(result["status"], "conditional", result.get("reason"))
        self.assertEqual(fake.cameras, ["fixed", "wrist"])                       # one more call for the target view
        value = result["value"]
        self.assertEqual(value["rows"][0][1], [0.28, 0.03, 0.015])               # the row is not replaced
        cross = value["cross_view"]["wrist"]
        self.assertEqual((cross["method"], cross["target_table"], cross["unmatched"], cross["skipped"]),
                         ("mask_centre_rays", "computed", [], {}))
        row = cross["rows"][0]
        self.assertEqual(row["id"], "o1")
        self.assertEqual(row["target_view_centre_m"], [0.26, 0.03, 0.015])
        self.assertEqual(row["target_view_minus_row_m"], [-0.02, 0.0, 0.0])
        self.assertLess(float(np.linalg.norm(np.asarray(row["stereo_centre_m"]) - truth[0])), 0.010)
        self.assertLess(row["ray_gap_m"], 0.006)
        self.assertIn("cross_view:", result["assumptions"][-1])
        self.assertEqual(result["diagnostics"]["cross_view"]["wrist"]["matched"], 1)
        self.assertEqual([camera for camera, _, _ in overlays], ["fixed", "wrist"])
        json.dumps(value)                                                        # plain JSON

    def test_attached_table_of_the_target_camera_is_reused(self):
        self.observe(0.8, 0.5)
        runtime = self.runtime("http://127.0.0.1:9", held=False)
        wrist_T = GeometryTools_camera(runtime, "wrist")["T"]
        truth = [np.array([0.26, 0.03, 0.015])]
        answers = {"fixed": self.rows_for(self.fixed, truth, (0.02, 0.0, 0.0)), "wrist": self.rows_for(wrist_T, truth)}
        fake = CameraService(answers, {"fixed": self.fixed, "wrist": wrist_T})
        self.addCleanup(fake.close)
        runtime = self.runtime(fake.url, held=False)
        attached, _ = self.query(runtime)                                        # as obs3d attaches the overview table
        obs3d._remember(runtime, "obs-so101", "object_table", ({}, None, attached))
        result, _ = self.query(runtime, camera_id="wrist", target_camera_ids=["fixed"])
        self.assertEqual(fake.cameras, ["fixed", "wrist"])                       # no second overview call
        cross = result["value"]["cross_view"]["fixed"]
        self.assertEqual(cross["target_table"], "attached")
        self.assertEqual(cross["rows"][0]["target_view_minus_row_m"], [0.02, 0.0, 0.0])

    def test_target_camera_table_is_not_a_pregrasp_table(self):
        # Regression: the cross-view check's own target-camera table is never returned to the model. It must not be
        # kept as a candidate pre-grasp table of held perception or take a per-run table number; otherwise the next
        # jaw closure carries the size and centre of that hidden wrist table and its note names a table the model
        # never received.
        self.observe(0.8, 0.5, record=True, step=5, observation_id="obs-pre")         # open above the cube
        runtime = self.runtime("http://127.0.0.1:9")
        wrist_T = GeometryTools_camera(runtime, "wrist", "obs-pre")["T"]
        truth = [np.array([0.225, 0.0, 0.015])]
        answers = {"fixed": self.rows_for(self.fixed, truth, (0.015, 0.0, 0.0)), "wrist": self.rows_for(wrist_T, truth)}
        answers["wrist"][0].update(length_m=0.05, width_m=0.05)                     # the hidden table differs
        grasp_jaw = 0.156
        tcp = poses_of(self.Q, grasp_jaw)["right"][kin.TCP_LINK]
        grasp = tcp[:3, 3] + tcp[:3, :3] @ np.asarray(cad.fingertip_reference(grasp_jaw)["midpoint_m"])
        held = dict(status="ok", mask_source="sam31_text", mask_px=2500, centre_m=grasp.tolist(), dist_px=0.0,
                    grasp_px=[320.0, 240.0], grasp_depth_m=0.12, object_index=None,
                    contour_px=[[300, 230], [340, 230], [340, 250], [300, 250]], label_px=[320, 240])
        # only the hidden wrist table's held mask has a VGGT extent (which would set the carried closure length)
        hidden_gk = dict(held, gk=dict(top_z_m=grasp[2] + 0.02, low_z_m=grasp[2] - 0.02, length_m=0.07, width_m=0.04,
                                       yaw_deg=0.0, elongation=1.75, axis_tilt_deg=5.0))
        held_for = lambda path, camera: hidden_gk if (path, camera) == ("/object_table", "wrist") else held
        fake = CameraService(answers, {"fixed": self.fixed, "wrist": wrist_T}, held=held_for)
        self.addCleanup(fake.close)
        runtime = self.runtime(fake.url)
        result, _ = self.query(runtime, observation_id="obs-pre", target_camera_ids=["wrist"])
        self.assertEqual(result["status"], "conditional", result.get("reason"))
        self.assertEqual(fake.cameras, ["fixed", "wrist"])                       # the target table was computed
        self.assertEqual(len(result["value"]["cross_view"]["wrist"]["rows"]), 1)
        memory = hp.tracker(runtime)
        self.assertEqual([(t["camera_id"], t["source"]) for t in memory["tables"]],
                         [("fixed", dict(seq=1, request="plain"))])                 # only the table the model got
        self.assertEqual(memory["table_seq"], 1)
        # the jaws close on the cube; the next table carries the fixed row the model saw (object table #1)
        self.observe(grasp_jaw, 0.0, record=True, step=10, observation_id="obs-closed")
        result, _ = self.query(runtime, observation_id="obs-closed", target_camera_ids=["wrist"])
        self.assertEqual(result["status"], "conditional", result.get("reason"))
        value = result["value"]
        row = dict(zip(value["columns"], value["rows"][-1]))
        self.assertEqual(row["support"], "held_right")
        cross = value["cross_view"]["wrist"]
        self.assertEqual(([r["id"] for r in cross["rows"]], cross["unmatched"], cross["skipped"]),
                         (["o1"], [], {row["id"]: "held_row"}))                    # the held row is listed, not paired
        meta = value["held"][row["id"]]
        self.assertEqual(meta["pregrasp_table_source"],
                         dict(seq=1, request="plain", observation_id="obs-pre", camera_id="fixed"))
        self.assertEqual((meta["pregrasp_row_id"], meta["pregrasp_observation_id"]), ("o1", "obs-pre"))
        self.assertIn("object table #1", meta["note"])
        # the fixed row's 0.10 x 0.03 m footprint (re-measured as a box turned with the gripper), not the wrist 0.05
        np.testing.assert_allclose([row["length"], row["width"]], [0.1, 0.03], atol=0.002)
        np.testing.assert_allclose(meta["pregrasp_row_centre_m"], truth[0] + [0.015, 0.0, 0.0], atol=1e-9)
        self.assertEqual(memory["table_seq"], 2)                                  # the hidden wrist tables: no number
        self.assertEqual(fake.cameras, ["fixed", "wrist", "fixed", "wrist"])
        self.assertEqual(fake.paths, ["/object_table", "/object_table", "/object_table", "/held", "/object_table"])
        self.assertIsNone(memory["carried"]["right"]["closure_length_m"])         # not set by the hidden table

    def test_unavailable_target_table_keeps_the_table(self):
        self.observe(0.8, 0.5)
        runtime = self.runtime("http://127.0.0.1:9", held=False)
        wrist_T = GeometryTools_camera(runtime, "wrist")["T"]
        fake = CameraService({"fixed": self.rows_for(self.fixed, [np.array([0.26, 0.03, 0.015])])},
                             {"fixed": self.fixed, "wrist": wrist_T})
        self.addCleanup(fake.close)
        result, _ = self.query(self.runtime(fake.url, held=False), target_camera_ids=["wrist"])
        self.assertEqual(result["status"], "conditional")
        self.assertEqual(len(result["value"]["rows"]), 1)
        self.assertIn("unavailable", result["value"]["cross_view"]["wrist"])

    def test_profile_without_the_tag_is_unchanged(self):
        self.observe(0.8, 0.5)
        fake = FakeService()
        self.addCleanup(fake.close)
        plain = dataclasses.replace(PROFILE, supports=PROFILE.supports - {"cross_view"})
        result, _ = self.query(self.runtime(fake.url, profile=plain, held=False), target_camera_ids=["wrist"])
        self.assertEqual(len(fake.requests), 1)
        self.assertNotIn("cross_view", result["value"])
        self.assertEqual(len(result["assumptions"]), 1)
        self.assertNotIn("cross_view", result["diagnostics"])

    def test_model_facing_text_names_cross_view_only_with_the_tag(self):
        def target_field(profile):
            tools = public_definitions(profile=profile, object_table=True)
            geometry = next(tool for tool in tools if tool["name"] == "geometry_query")
            branch = next(b for b in geometry["inputSchema"]["properties"]["request"]["oneOf"]
                          if b.get("properties", {}).get("kind", {}).get("const") == "object_table")
            return branch["properties"]["target_camera_ids"]["description"]
        self.assertIn("cross_view", target_field(PROFILE))
        self.assertIn("ray_gap_m", target_field(PROFILE))
        self.assertEqual(target_field(None), object_table.FIELD_DESCRIPTIONS["target_camera_ids"])
        plain = dataclasses.replace(PROFILE, supports=PROFILE.supports - {"cross_view"})
        self.assertEqual(target_field(plain), object_table.FIELD_DESCRIPTIONS["target_camera_ids"])
        features = Config().features
        self.assertIn("`cross_view`", obs3d.developer_instructions(features, PROFILE))
        self.assertNotIn("cross_view", obs3d.developer_instructions(features, plain))
        self.assertNotIn("cross_view", obs3d.developer_instructions(features, None))


def GeometryTools_camera(runtime, camera_id, observation_id="obs-so101"):
    from ni_rua.robot.model.geometry import GeometryTools
    tools = GeometryTools(runtime.scene, settings=runtime.image_settings, profile=runtime.embodiment)
    full = {"observation_id": observation_id, **runtime.identity, "schema_version": "0.1", "camera_id": camera_id}
    return tools.camera(full, observation_id, camera_id)


class CameraService:
    """Fake /object_table service answering per camera (recognised by the request's T_world_camera). ``held``: the
    held-object result returned for every held item of a request (``/object_table`` with ``held`` and ``/held``), or a
    function (path, camera) -> that result."""

    def __init__(self, objects_by_camera, poses, held=None):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self.cameras, self.paths, owner = [], [], self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                T = np.asarray(body["T_world_camera"])
                camera = next(name for name, pose in poses.items() if np.allclose(pose, T, atol=1e-6))
                owner.paths.append(self.path)
                result = held(self.path, camera) if callable(held) else held
                items = [dict(result or dict(status="no_mask"), arm=item["arm"]) for item in body.get("held") or []]
                if self.path.startswith("/held"):
                    answer, code = dict(status="ok", held=items), 200
                elif camera in objects_by_camera:
                    owner.cameras.append(camera)
                    answer, code = dict(status="ok", version="fake", objects=objects_by_camera[camera], requested=[],
                                        omitted=0, labels=[], support_z_m=body["support_z_m"], dropped={}, n_prompts=1,
                                        timing={}, support_fit={}, scale=1.0), 200
                    if body.get("robot_mask_b64"):
                        answer["robot_mask"] = "pixels"
                    if items:
                        answer["held"] = items
                else:
                    owner.cameras.append(camera)
                    answer, code = dict(status="error", error="no_table_for_" + camera), 422
                data = json.dumps(answer).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


if __name__ == "__main__":
    unittest.main()
