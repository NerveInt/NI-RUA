"""SO-101 perception view: arm occlusion report (P1), clear-view pose (P2), capture resolution (P3), two-view
table-contact top height (P4), model-visible text budgets and the perception bench's metric code.

Synthetic geometry (packaged SO-101 URDF + meshes, pinhole cameras, a fake VGGT service) plus a real-run regression
(tests/fixtures/so101_occlusion_real.json: measured joints, the fixed-camera calibration and the model-visible tables
of six observations of a real stacking run; no images).
python -m unittest test_so101_perception_view"""

NI_RUA_TEST_TIER = "component"

import base64
import importlib.util
import functools
import json
from support.paths import CONFIGS_ROOT, FIXTURES_ROOT, REPO_ROOT
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from ni_rua.core.contracts import canonical
from ni_rua.perception.external import contact_height, held_perception as hp, object_table, obs3d, robot_view
from ni_rua.robot.backends.so101 import contract
from ni_rua.robot.backends.so101 import deployment as so101_deployment
from ni_rua.robot.calibration.workflow import charuco as cal
from ni_rua.robot.model import embodiments, so101_kinematics as kin
from support.geometry_fixture import IDENTITY
from robot.so101.test_so101_held_perception import (FIXTURE, MODEL, PROFILE, CameraService, So101Scene, arm_state, box_outline,
                                        look_at, so101_profile)
from support.perception_fixtures import fake_object

REPOSITORY = REPO_ROOT
EXAMPLE = CONFIGS_ROOT / "so101.example.json"
REAL = json.loads((FIXTURES_ROOT / "so101_occlusion_real.json").read_text())
K640 = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
FIXED = look_at([0.25, -0.25, 0.45], [0.22, 0.0, 0.0])
OVER_CUBE_Q = [0.0, -0.3, 0.6, 1.2, 0.0]      # the arm low over the table in front of the base (So101Scene.Q)


def example_deployment(**perception):
    document = json.loads(EXAMPLE.read_text())
    if perception:
        document["perception"] = perception
    return so101_deployment.validate_deployment(document, base=EXAMPLE.parent)


def camera(T=FIXED, K=K640, width=640, height=480):
    return dict(K=K, T=T, width=width, height=height)


def masks(q, cam, jaw=0.3):
    body = hp.SO101Body()
    poses = {"right": kin.fk_all(MODEL, q, gripper_rad=jaw)}
    return robot_view.silhouettes(body, body.meshes(None), poses, cam, "fixed", {"right": "wrist"})


def box(centre_xy, edge=0.03, top=0.03, bottom=0.0):
    return dict(centre_m=[*centre_xy, (top + bottom) / 2], length_m=edge, width_m=edge, yaw_deg=0.0,
                bottom_z_m=bottom, top_z_m=top)


def under_tcp(q, T=FIXED, height=0.015):
    """xy (m) of the point at ``height`` (m) that camera ``T`` sees behind the gripper link of the arm at ``q``: the
    camera ray through the gripper link meets that height there, so a box centred there is hidden by the gripper."""
    centre = kin.fk_all(MODEL, q, gripper_rad=0.3)["gripper_link"][:3, 3]
    ray = centre - T[:3, 3]
    return (T[:3, 3] + (height - T[2, 3]) / ray[2] * ray)[:2]


class OcclusionGeometry(unittest.TestCase):
    """P1: the table ROI, the coverage fraction and the covered share of an earlier row's box."""

    @classmethod
    def setUpClass(cls):
        cls.cam = camera()
        cls.robot_mask, cls.base_mask = masks(OVER_CUBE_Q, cls.cam)
        cls.roi = robot_view.table_roi(K640, FIXED, 640, 480, 0.0)

    def test_reach_facts_come_from_the_urdf(self):
        facts = robot_view.reach_facts()
        self.assertEqual(facts["base_link"], "base_link")
        self.assertAlmostEqual(facts["pan_axis_xy_m"][0], 0.0388, places=3)
        self.assertTrue(0.40 < facts["reach_m"] < 0.48, facts)

    def test_table_roi_is_the_table_within_reach_in_front_of_the_camera(self):
        roi = robot_view.table_roi(K640, FIXED, 640, 480, 0.0)
        facts = robot_view.reach_facts()
        self.assertTrue(0 < roi.mean() < 1)
        for u, v in ((320, 240), (100, 400)):
            ray = FIXED[:3, :3] @ np.linalg.inv(K640) @ np.array([u + 0.5, v + 0.5, 1.0])
            point = FIXED[:3, 3] + (0.0 - FIXED[2, 3]) / ray[2] * ray
            inside = np.hypot(*(point[:2] - facts["pan_axis_xy_m"])) <= facts["reach_m"]
            self.assertEqual(bool(roi[v, u]), bool(inside))
        looking_up = look_at([0.2, 0.0, 0.1], [0.2, 0.0, 1.0], up=(0.0, 1.0, 0.0))
        self.assertFalse(robot_view.table_roi(K640, looking_up, 64, 48, 0.0).any())   # no table in front

    def test_arm_over_the_table_covers_a_box_under_the_gripper_and_not_a_far_one(self):
        cam = self.cam
        robot, base = self.robot_mask, self.base_mask
        covered = robot_view.coverage(robot, base, self.roi)
        self.assertTrue(0.02 < covered < 0.6, covered)
        self.assertFalse((base & ~robot).any())                 # the base silhouette is part of the robot's
        near = robot_view.box_covered(K640, FIXED, robot, box(under_tcp(OVER_CUBE_Q)))
        far = robot_view.box_covered(K640, FIXED, robot, box([0.25, 0.20]))
        self.assertGreater(near, 0.5)
        self.assertEqual(far, 0.0)
        self.assertIsNone(robot_view.box_covered(K640, FIXED, robot, box([0.25, -0.30], top=0.6, bottom=0.5)))

    def test_previous_rows_under_the_arm_are_hidden_with_id_observation_and_share(self):
        cam = self.cam
        robot, base = self.robot_mask, self.base_mask
        xy = under_tcp(OVER_CUBE_Q)
        objects = [dict(id="o1", **box(xy), support="table", quality="ok"),
                   dict(id="o2", **box([0.25, 0.20]), support="table", quality="ok"),
                   dict(id="o3", **box(xy), support="held_right", quality="held_fk_depth")]
        carried = dict(id="o4", seen_in="obs-older", box=box(xy + [0.025, 0.0]), covered=1.0)
        duplicate = dict(id="o5", seen_in="obs-older", box=box(xy + [0.004, 0.0]), covered=1.0)   # inside o1's footprint
        result = dict(value=dict(camera_id="fixed"),
                      diagnostics=dict(objects=objects, robot_occlusion=dict(hidden=[carried, duplicate])))
        runtime = type("R", (), {})()
        obs3d._remember(runtime, "obs-before", "object_table", ({}, None, result))
        value, diagnostics = robot_view.occlusion(runtime, cam, "fixed", robot, base, 0.0, "obs-now")
        ids = [(item["id"], item["seen_in"]) for item in value["hidden"]]
        self.assertIn(("o1", "obs-before"), ids)
        self.assertIn(("o4", "obs-older"), ids)                      # an earlier hidden entry is carried on
        self.assertNotIn(("o5", "obs-older"), ids)                   # ... unless a newer row lists that object
        self.assertNotIn("o2", [item["id"] for item in value["hidden"]])   # not under the arm
        self.assertNotIn("o3", [item["id"] for item in value["hidden"]])   # held rows move with the arm
        first = value["hidden"][0]
        self.assertEqual(set(first), {"id", "seen_in", "centre_m", "top_z", "covered"})
        self.assertGreater(first["covered"], 0.5)
        self.assertEqual(diagnostics["reference_observation_id"], "obs-before")
        self.assertEqual(value["table_covered"], round(robot_view.coverage(
            robot, base, robot_view.table_roi(K640, FIXED, 640, 480, 0.0)), 2))
        # the same observation's own table is never its reference; another camera's table neither
        runtime2 = type("R", (), {})()
        obs3d._remember(runtime2, "obs-now", "object_table", ({}, None, result))
        obs3d._remember(runtime2, "obs-wrist", "object_table", ({}, None, dict(result, value=dict(camera_id="wrist"))))
        value2, _ = robot_view.occlusion(runtime2, cam, "fixed", robot, base, 0.0, "obs-now")
        self.assertNotIn("hidden", value2)

    def test_hidden_list_stays_bounded_for_an_object_seen_again_and_again(self):
        """Regression (review): an object still listed as a row while partly under the arm used to come back once
        more per observation (each earlier hidden entry carried on with its own seen_in)."""
        cam = self.cam
        robot, base = self.robot_mask, self.base_mask
        item = dict(id="o1", **box(under_tcp(OVER_CUBE_Q)), support="table", quality="ok")
        runtime = type("R", (), {})()
        for step in range(6):
            observation = "obs-%d" % step
            value, diagnostics = robot_view.occlusion(runtime, cam, "fixed", robot, base, 0.0, observation)
            self.assertLessEqual(len(value.get("hidden", [])), 1, (step, value))
            result = dict(value=dict(camera_id="fixed"), diagnostics=dict(objects=[item], robot_occlusion=diagnostics))
            obs3d._remember(runtime, observation, "object_table", ({}, None, result))

    def test_reference_is_the_table_before_the_observation(self):
        """Regression (review): a query of an older observation used the newest kept table, a later one."""
        runtime = type("R", (), {})()
        for step in range(6):
            result = dict(value=dict(camera_id="fixed"), diagnostics=dict(objects=[]))
            obs3d._remember(runtime, "obs-%d" % step, "object_table", ({}, None, result))
        self.assertEqual(robot_view.previous_entries(runtime, "obs-2", "fixed")[0], "obs-1")
        self.assertEqual(robot_view.previous_entries(runtime, "obs-new", "fixed")[0], "obs-5")
        self.assertEqual(robot_view.previous_entries(runtime, "obs-0", "fixed"), (None, []))

    def test_real_run_empty_tables_are_the_arm_covering_the_cubes(self):
        """Regression on a real run: the tables that came back empty are frames where the arm's FK silhouette covers
        a third of the table and the cubes the previous table listed."""
        fixed = REAL["fixed_camera"]
        cam = dict(K=np.asarray(fixed["K"]), T=np.asarray(fixed["T_world_camera"]), width=640, height=480)
        body = hp.SO101Body()
        meshes = body.meshes(None)
        roi = robot_view.table_roi(cam["K"], cam["T"], 640, 480, fixed["table_z_m"])
        frames = {frame["observation"]: frame for frame in REAL["frames"]}
        report = {}
        for name, frame in frames.items():
            poses = {"right": kin.fk_all(MODEL, frame["q_rad"][:5], gripper_rad=frame["q_rad"][5])}
            robot, base = robot_view.silhouettes(body, meshes, poses, cam, "fixed", {"right": "wrist"})
            report[name] = (robot, robot_view.coverage(robot, base, roi))
        clear, occluded = report["obs-b1c2e8c2"], report["obs-b737b457"]
        self.assertEqual(len(frames["obs-b737b457"]["rows"]), 0)
        self.assertEqual(len(frames["obs-b1c2e8c2"]["rows"]), 4)
        self.assertLess(clear[1], 0.2)
        self.assertGreater(occluded[1], 0.3)
        hidden = {o["id"]: robot_view.box_covered(cam["K"], cam["T"], occluded[0], robot_view.row_box(o))
                  for o in frames["obs-b1c2e8c2"]["objects"]}
        self.assertGreater(hidden["o1"], 0.9)                 # blue cube
        self.assertGreater(hidden["o3"], 0.9)                 # orange cube
        self.assertGreater(hidden["o2"], 0.3)                 # white cube, half under the arm
        for name in ("obs-8233fc4e", "obs-9c9dfe10"):         # two more empty / thinned tables
            self.assertGreater(report[name][1], 0.3, name)


class OcclusionQuery(So101Scene, unittest.TestCase):
    """P1 through object_table_query and the obs3d attachment."""

    Q = np.array(OVER_CUBE_Q)

    def service(self, objects):
        fake = CameraService({"fixed": objects, "wrist": objects}, {"fixed": FIXED, "wrist": self.wrist()})
        self.addCleanup(fake.close)
        return fake

    def wrist(self):
        link = kin.fk_all(MODEL, self.Q, gripper_rad=0.8)[FIXTURE["link_name"]]
        return link @ np.asarray(FIXTURE["T_link_camera"])

    def test_report_names_the_covered_share_and_the_earlier_rows_under_the_arm(self):
        self.observe(0.8, 0.5, observation_id="obs-before")
        xy = under_tcp(self.Q)
        row = fake_object((xy[0], xy[1], 0.015), 0.03, 0.0, elongation=1.0)
        row.update(length_m=0.03, width_m=0.03)
        fake = self.service([row])
        runtime = self.runtime(fake.url)
        first, _ = self.query(runtime, observation_id="obs-before")
        first_value = first["value"]
        self.assertIn("robot_occlusion", first_value)
        self.assertNotIn("hidden", first_value["robot_occlusion"])          # no earlier table yet
        obs3d._remember(runtime, "obs-before", "object_table", (first_value, None, first))
        after = self.observe(0.8, 0.5, observation_id="obs-so101")
        empty = self.service([])
        runtime.config = runtime.config.with_overrides({"features": {
            "vggt_endpoint": empty.url, "object_table_observe": True, "scene3d_observe": False}})
        result, _ = self.query(runtime)
        self.assertEqual(result["status"], "conditional", result.get("reason"))
        self.assertEqual(result["value"]["rows"], [])                        # rows are not invented
        occlusion = result["value"]["robot_occlusion"]
        self.assertGreater(occlusion["table_covered"], 0.0)
        self.assertEqual([(item["id"], item["seen_in"]) for item in occlusion["hidden"]], [("o1", "obs-before")])
        self.assertGreater(occlusion["hidden"][0]["covered"], 0.5)
        self.assertEqual(result["diagnostics"]["robot_occlusion"]["reference_observation_id"], "obs-before")
        parts = obs3d.action_attachments(runtime, after, {}, before_observation_id="obs-before")
        delta = parts[0].value["object_state_delta"]
        self.assertEqual(delta, dict(unmatched_under_arm=[dict(before_id="o1", covered=occlusion["hidden"][0]["covered"])],
                                     vacated=[], moved=[], appeared=[], unchanged_count=0))
        self.assertEqual(self.artifacts.read_state("perception", "obs-so101")["delta"], delta)
        self.assertEqual(obs3d.consume_last_object_table_delta(runtime, "obs-so101"), ("obs-before", "obs-so101", delta))

    def test_delta_uses_the_unmatched_source_row_and_keeps_partial_coverage(self):
        centres = [[float(i), 0., 0.] for i in range(5)]
        before = dict(columns=["id", "centre_m"], rows=[[f"o{i}", c] for i, c in enumerate(centres)])
        after = dict(columns=before["columns"], rows=[["o1", centres[0]]], robot_occlusion=dict(hidden=[
            dict(seen_in="before", id="o0", centre_m=centres[0], covered=0.75),  # matched, despite overlap
            dict(seen_in="older", id="o1", centre_m=centres[1], covered=1.),      # carried from another frame
            dict(seen_in="before", id="o2", centre_m=centres[1], covered=1.),    # other table's local ID
            dict(seen_in="before", id="o3", centre_m=centres[3], covered=0.12),
            dict(seen_in="before", id="o4", centre_m=centres[4], covered=0),
        ]))
        delta = object_table.compute_object_table_delta(before, after, before_observation_id="before")
        self.assertEqual(delta["unmatched_under_arm"], [dict(before_id="o3", covered=0.12)])
        self.assertEqual([v["before_id"] for v in delta["vacated"]], ["o1", "o2", "o4"])
        self.assertEqual((delta["unchanged_count"], delta["moved"], delta["appeared"]), (1, [], []))
        for kwargs, table in [({}, after), (dict(before_observation_id="before"), dict(after, robot_occlusion={}))]:
            legacy = object_table.compute_object_table_delta(before, table, **kwargs)
            self.assertNotIn("unmatched_under_arm", legacy)
            self.assertEqual(len(legacy["vacated"]), 4)

    def test_without_held_perception_the_report_renders_its_own_silhouette(self):
        self.observe(0.8, 0.5)
        fake = self.service([])
        with_held, _ = self.query(self.runtime(fake.url, held=True))
        without, _ = self.query(self.runtime(fake.url, held=False))
        self.assertEqual(with_held["value"]["robot_occlusion"], without["value"]["robot_occlusion"])

    def test_a_failed_render_reports_a_reason_code_and_keeps_the_table(self):
        self.observe(0.8, 0.5)
        fake = self.service([fake_object((0.2, 0.0, 0.015), 0.03, 0.0)])
        original = robot_view.silhouettes

        def broken(*args, **kwargs):
            raise FileNotFoundError("/private/path/meshes.zip")
        robot_view.silhouettes = broken
        self.addCleanup(setattr, robot_view, "silhouettes", original)
        result, _ = self.query(self.runtime(fake.url))
        self.assertEqual(result["status"], "conditional")
        self.assertEqual(len(result["value"]["rows"]), 1)
        self.assertEqual(result["value"]["robot_occlusion"], {"unavailable": "render_failed"})   # no path or text
        self.assertIn("FileNotFoundError", result["diagnostics"]["robot_occlusion"]["error"])

    def test_switch_off_and_other_robots_have_no_field(self):
        self.observe(0.8, 0.5)
        fake = self.service([])
        off = so101_profile()
        off = embodiments.EmbodimentProfile(**{**{f: getattr(off, f) for f in off.__dataclass_fields__},
                                              "supports": off.supports - {"robot_occlusion"}})
        result, _ = self.query(self.runtime(fake.url, profile=off))
        self.assertNotIn("robot_occlusion", result["value"])
        self.assertNotIn("robot_occlusion", embodiments.ARX_X5_PROFILE.supports)
        profile = so101_deployment.embodiment_profile(example_deployment(robot_occlusion_report=False))
        self.assertNotIn("robot_occlusion", profile.supports)

    def test_obs3d_attachment_carries_the_report(self):
        self.observe(0.8, 0.5)
        fake = self.service([])
        runtime = self.runtime(fake.url)
        runtime.config = runtime.config.with_overrides({"features": {"object_table_observe": True,
                                                                     "scene3d_observe": False}})
        shown, overlay, _ = obs3d._object_table(runtime, "obs-so101", 5.0, 10, camera_id="fixed")
        self.assertIn("robot_occlusion", shown)
        self.assertEqual(shown["rows"], [])


class ClearView(unittest.TestCase):
    """P2: the offline clear-view search and its single model-facing sentence."""

    @classmethod
    def setUpClass(cls):
        deployment = example_deployment()
        margin = deployment["motion"]["joint_limit_margin_rad"]
        cls.limits = {name: (low + margin, high - margin) for name, (low, high) in kin.joint_limits_rad(MODEL).items()
                      if name in kin.ARM_JOINTS}
        cls.home = list(deployment["home"]["q_rad"])
        cls.workspace = deployment["workspace"]
        cls.front = robot_view.clear_view_search(camera(), 0.0, limits_rad=cls.limits, home_q_rad=cls.home,
                                                 workspace_min_m=cls.workspace["min_m"],
                                                 workspace_max_m=cls.workspace["max_m"], gripper_rad=0.0,
                                                 grid=(7, 5, 5, 4))

    def test_chosen_pose_is_reachable_inside_the_limits_and_clears_the_table(self):
        self.assertTrue(self.front)
        chosen = self.front[0]
        for name, value in zip(kin.ARM_JOINTS, chosen["q_rad"]):
            low, high = self.limits[name]
            self.assertTrue(low - 1e-6 <= value <= high + 1e-6, (name, value))
        self.assertTrue(all(lo <= v <= hi for lo, v, hi in zip(self.workspace["min_m"], chosen["tcp_position_m"],
                                                               self.workspace["max_m"])))
        self.assertGreaterEqual(chosen["min_link_z_m"], 0.0)
        home = robot_view.pose_coverage(camera(), 0.0, self.home, 0.0)
        self.assertLess(chosen["coverage"], home)
        self.assertLess(chosen["coverage"], home / 4)
        self.assertAlmostEqual(robot_view.pose_coverage(camera(), 0.0, chosen["q_rad"], 0.0), chosen["coverage"], 3)
        target = kin.fk(MODEL, chosen["q_rad"])
        found = kin.ik(MODEL, target, self.home, limits=self.limits)
        self.assertTrue(found["converged"])                 # the TCP pose is reachable within the planner's limits
        coverages = [item["coverage"] for item in self.front]
        travels = [item["travel_rad"] for item in self.front]
        self.assertEqual(coverages, sorted(coverages))
        self.assertTrue(all(a > b for a, b in zip(travels, travels[1:])))   # a Pareto front

    def test_deployment_accepts_the_pose_and_the_text_names_its_tcp_target(self):
        q = self.front[0]["q_rad"]
        deployment = example_deployment(clear_view_q_rad=q)
        self.assertEqual(deployment["perception"]["clear_view_q_rad"], q)
        sentence = contract.clear_view_sentence(deployment)
        T = kin.fk(MODEL, q)
        self.assertIn(str([round(float(v), 4) for v in T[:3, 3]]), sentence)
        self.assertIn(sentence, contract.actor_motion_instructions(deployment))
        self.assertLess(len(sentence), 480)
        self.assertEqual(contract.clear_view_sentence(example_deployment()), "")
        self.assertNotIn("Clear fixed-camera view", contract.actor_motion_instructions(example_deployment()))
        with self.assertRaisesRegex(ValueError, "clear_view_q_rad.shoulder_pan"):
            example_deployment(clear_view_q_rad=[3.0, 0.0, 0.0, 0.0, 0.0])
        low = [0.0, 0.6, 0.6, 1.2, 0.0]                        # joints within the limits, TCP below the workspace floor
        self.assertLess(kin.fk(MODEL, low)[2, 3], example_deployment()["workspace"]["min_m"][2])
        with self.assertRaisesRegex(ValueError, "clear_view_q_rad: its TCP"):
            example_deployment(clear_view_q_rad=low)
        with self.assertRaisesRegex(ValueError, "perception: unknown"):
            example_deployment(clear_pose=True)
        self.assertEqual(example_deployment()["perception"], so101_deployment.PERCEPTION_DEFAULTS)


@functools.lru_cache(maxsize=None)
def _observation_pngs(width, height):
    rng = np.random.default_rng(5)
    image = cv2.imencode(".png", rng.integers(0, 256, (height, width, 3), dtype=np.uint8))[1].tobytes()
    wrist_image = cv2.imencode(".png", rng.integers(0, 256, (480, 640, 3), dtype=np.uint8))[1].tobytes()
    return image, wrist_image


def observation_at(store, size, K, *, q=OVER_CUBE_Q, jaw=0.8, goal=0.5, observation_id="obs-hd", D=(0.0,) * 5):
    """A two-camera SO-101 observation whose fixed camera is ``size`` = (width, height) px with intrinsics ``K``."""
    width, height = size
    image, wrist_image = _observation_pngs(width, height)
    calibration = {"reference_frame_id": "so101_base", "joint_calibration": {"table_z_m": 0.0}, "cameras": {
        "fixed": {"K": np.asarray(K).tolist(), "distortion": list(D), "distortion_model": "opencv_plumb_bob",
                  "image_size": [width, height], "T_world_camera": FIXED.tolist()},
        "wrist": {"K": K640.tolist(), "distortion": [0.0] * 5, "distortion_model": "opencv_plumb_bob",
                  "image_size": [640, 480], "arm": "right", "link_name": FIXTURE["link_name"],
                  "T_link_camera": FIXTURE["T_link_camera"]}}}
    calibration_ref = store.put_artifact(canonical(calibration).encode(), "calibration", "application/json")
    model_ref = store.put_artifact(canonical(MODEL).encode(), "robot_model", "application/json")
    state = {"arms": {"right": arm_state(np.asarray(q), jaw, goal)}, "frame_id": "so101_base", "sim_step": 5,
             "step_domain": "so101_control_step", "tcp_config_id": "so101-gripper-frame-v1"}
    state_ref = store.put_artifact(canonical(state).encode(), "robot_state", "application/json")
    link = kin.fk_all(MODEL, q, gripper_rad=jaw)[FIXTURE["link_name"]]
    sample = {"reference_frame_id": "so101_base", "sim_step": 5, "step_domain": "so101_control_step", "cameras": {
        "fixed": {"T_world_camera": FIXED.tolist(), "source_kind": "static_extrinsic_calibration"},
        "wrist": {"T_world_base": np.eye(4).tolist(), "T_world_link": link.tolist(), "arm": "right",
                  "T_world_camera": (link @ np.asarray(FIXTURE["T_link_camera"])).tolist(),
                  "link_name": FIXTURE["link_name"], "source_kind": "robot_kinematics_at_joint_read"}}}
    sample_ref = store.put_artifact(canonical(sample).encode(), "camera_pose_sample", "application/json")
    cameras = [{"camera_id": name, "frame_id": "frame-%s-%s" % (observation_id, name), "width": w, "height": h,
                "sim_step": 5, "step_domain": "so101_control_step",
                "rgb_ref": store.put_artifact(data, "rgb", "image/png"), "calibration_ref": calibration_ref,
                "pose_sample_ref": sample_ref, "pixel_transform": "identity"}
               for name, w, h, data in (("fixed", width, height, image), ("wrist", 640, 480, wrist_image))]
    return {"schema_version": "0.1", **IDENTITY, "world_kind": "observed", "data_origin": "synthetic_fixture",
            "observation_id": observation_id, "sim_step": 5, "step_domain": "so101_control_step",
            "clock_domain": "synthetic/generation-1", "calibration_id": "synthetic-calibration-%dx%d" % size,
            "robot_model_id": "so101", "robot_model_ref": model_ref,
            "observation_profile": "rgb_proprio_calibrated_synthetic_v1", "robot_state_ref": state_ref,
            "cameras": cameras}


class Resolution(So101Scene, unittest.TestCase):
    """P3: intrinsics for another capture mode, and the perception path at 1280x720 / 1920x1080."""

    def test_pixel_mapping_fit_and_remapped_intrinsics_project_like_the_new_mode(self):
        rng = np.random.default_rng(2)
        source = rng.uniform([0, 0], [640, 480], (40, 2))
        scale, offset = np.array([2.25, 2.25]), np.array([-240.0 + 0.6, 0.6])       # a 1440x1080 crop of 1920x1080
        fit = cal.fit_pixel_mapping(source, source * scale + offset + rng.normal(0, 0.05, (40, 2)))
        np.testing.assert_allclose(fit["scale"], scale, atol=1e-3)
        np.testing.assert_allclose(fit["offset_px"], offset, atol=0.1)
        self.assertLess(fit["rms_px"], 0.12)
        self.assertEqual(fit["points"], 40)
        with self.assertRaises(ValueError):
            cal.fit_pixel_mapping(source[:2], source[:2])
        D = [-0.012, 0.042, -0.0007, -0.0004, -0.065]
        entry = dict(K=K640.tolist(), distortion=D, image_size=[640, 480], intrinsics_rms_px=0.4,
                     T_base_camera=FIXED.tolist(), extrinsic_residual={"reprojection_px": 1.0})
        exact = dict(scale=scale.tolist(), offset_px=offset.tolist(), rms_px=0.0, points=40)
        remapped = cal.remap_camera_intrinsics(entry, exact, [1920, 1080], from_size=[640, 480])
        self.assertEqual(remapped["image_size"], [1920, 1080])
        self.assertEqual(remapped["distortion"], D)
        self.assertEqual(remapped["T_base_camera"], FIXED.tolist())
        self.assertAlmostEqual(remapped["intrinsics_rms_px"], 0.4 * 2.25)
        self.assertIn("remapped from 640x480", remapped["intrinsics_source"])
        points = np.array([[0.22, 0.0, 0.0], [0.30, 0.08, 0.03], [0.15, -0.05, 0.01]])
        R, t = cv2.Rodrigues(np.linalg.inv(FIXED)[:3, :3])[0], np.linalg.inv(FIXED)[:3, 3]
        old = cv2.projectPoints(points, R, t, K640, np.asarray(D))[0].reshape(-1, 2)
        new = cv2.projectPoints(points, R, t, np.asarray(remapped["K"]), np.asarray(D))[0].reshape(-1, 2)
        np.testing.assert_allclose(new, old * scale + offset, atol=1e-6)

    def test_remap_script_measures_the_map_from_two_board_images(self):
        """scripts/so101_remap_calibration.py end to end: a 640x480 mode that is the centre 1440x1080 of the
        1920x1080 sensor mode, scaled by 1 / 2.25 (a common UVC layout), recovered from one board seen in both."""
        spec = cal.validate_board_spec({})
        board = cal.make_board(spec)
        high = cv2.warpAffine(board.generateImage((700, 500), marginSize=20, borderBits=1),
                              np.array([[1.2, 0.1, 500.0], [-0.08, 1.2, 200.0]]), (1920, 1080),
                              dst=np.full((1080, 1920), 255, np.uint8), borderMode=cv2.BORDER_TRANSPARENT)
        low = cv2.resize(high[:, 240:1680], (640, 480), interpolation=cv2.INTER_AREA)
        work = Path(tempfile.mkdtemp(prefix="so101-remap-test-"))
        self.addCleanup(shutil.rmtree, work, True)
        cv2.imwrite(str(work / "low.png"), low)
        cv2.imwrite(str(work / "high.png"), high)
        K = [[491.0, 0.0, 333.1], [0.0, 490.6, 242.0], [0.0, 0.0, 1.0]]
        document = cal.build_calibration_document(board=spec, samples=0, cameras={"fixed": dict(
            K=K, distortion=[-0.012, 0.042, -0.0007, -0.0004, -0.065], image_size=[640, 480],
            T_base_camera=FIXED.tolist(), extrinsic_residual={"reprojection_px": 1.0}, intrinsics_rms_px=0.3)})
        (work / "calibration.json").write_text(json.dumps(document))
        spec_ = importlib.util.spec_from_file_location("so101_remap_calibration",
                                                       REPOSITORY / "scripts" / "so101_remap_calibration.py")
        script = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(script)
        arguments = ["--calibration", str(work / "calibration.json"), "--camera", "fixed", "--from-image",
                     str(work / "low.png"), "--to-image", str(work / "high.png"), "--out", str(work / "new.json")]
        script.main(arguments)
        fixed = json.loads((work / "new.json").read_text())["cameras"]["fixed"]
        self.assertEqual(fixed["image_size"], [1920, 1080])
        np.testing.assert_allclose([fixed["K"][0][0], fixed["K"][1][1]], [491.0 * 2.25, 490.6 * 2.25], rtol=2e-3)
        # pixel centres: low u <-> high 2.25 u + 240 + (2.25 - 1) / 2 under INTER_AREA
        np.testing.assert_allclose([fixed["K"][0][2], fixed["K"][1][2]],
                                   [2.25 * 333.1 + 240.625, 2.25 * 242.0 + 0.625], atol=1.0)
        self.assertEqual(fixed["T_base_camera"], FIXED.tolist())
        self.assertIn("remapped from 640x480", fixed["intrinsics_source"])
        with self.assertRaises(SystemExit):                     # an existing output file is never replaced
            script.main(arguments)

    def test_silhouette_dilation_and_overlay_scale_follow_the_image_width(self):
        body = hp.SO101Body()
        self.assertEqual([body.dilate_for(w) for w in (640, 1280, 1920)], [5, 10, 15])
        self.assertEqual(object_table.overlay_scale(np.zeros((480, 640, 3))), 1.0)
        self.assertEqual(object_table.overlay_scale(np.zeros((1080, 1920, 3))), 3.0)
        image = np.zeros((480, 640, 3), np.uint8)
        legacy = image.copy()                                   # the drawing before the scale existed, at 640 px
        (w, h), _ = cv2.getTextSize("o1", cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        x, y = int(round(300 - w / 2)), int(round(200 + h / 2))
        cv2.rectangle(legacy, (x - 2, y - h - 2), (x + w + 1, y + 3), (0, 0, 0), -1)
        cv2.putText(legacy, "o1", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        object_table._draw_label(image, "o1", 300, 200)
        self.assertTrue((image == legacy).all())                # byte-identical at <= 640 px

    def check_size(self, size):
        width, height = size
        scale = width / 640.0
        K = np.array([[500.0 * scale, 0.0, width / 2], [0.0, 500.0 * scale, height / 2 + 0.25], [0.0, 0.0, 1.0]])
        self.scene.import_observation(observation_at(self.artifacts, size, K))
        centre = np.array([0.26, 0.03, 0.015])
        T = FIXED
        outline = [[(K[0, 0] * p[0] / p[2] + K[0, 2]), (K[1, 1] * p[1] / p[2] + K[1, 2])]
                   for p in [(np.linalg.inv(T) @ np.append(c, 1.0))[:3] for c in
                             (centre + np.array([a, b, z]) for a in (-0.015, 0.015) for b in (-0.015, 0.015)
                              for z in (-0.015, 0.015))]]
        hull = cv2.convexHull(np.asarray(outline, np.float32)).reshape(-1, 2)
        row = fake_object(tuple(centre), 0.03, 0.0, elongation=1.0)
        row.update(contour_px=np.rint(hull).astype(int).tolist(), label_px=np.rint(hull.mean(0)).astype(int).tolist())
        fake = CameraService({"fixed": [row]}, {"fixed": FIXED})
        self.addCleanup(fake.close)
        sent = []
        original = object_table.query_service

        def spy(endpoint, body, *args, **kwargs):
            sent.append(body)
            return original(endpoint, body, *args, **kwargs)
        object_table.query_service = spy
        self.addCleanup(setattr, object_table, "query_service", original)
        runtime = self.runtime(fake.url)
        result, overlays = object_table.object_table_query(runtime, {"kind": "object_table", "observation_id": "obs-hd",
                                                                     "camera_id": "fixed"})
        self.assertEqual(result["status"], "conditional", result.get("reason"))
        body = sent[0]
        np.testing.assert_allclose(body["K"], K)                                  # the calibration's own K
        self.assertEqual(body["robot_mask_shape"], [height, width])
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(body["image_b64"]), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape[:2], (height, width))                      # the service gets the full frame
        self.assertEqual(result["value"]["rows"][0][1], [0.26, 0.03, 0.015])
        overlay = cv2.imdecode(np.frombuffer(overlays[0][1], np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(overlay.shape[:2], (height, width))
        self.assertIn("robot_occlusion", result["value"])
        from ni_rua.core.storage.images import image_settings, image_url
        _, encoding = image_url(overlays[0][1], image_settings(None))
        self.assertEqual((encoding["width"], encoding["height"]), (width, height))  # delivered at full size

    def test_1280x720(self):
        self.check_size((1280, 720))

    def test_1920x1080(self):
        self.check_size((1920, 1080))


def prism_views(centre, size, fixed=FIXED, wrist=None, shift=(0.0, 0.0)):
    """Two pinhole views (fixed + a wrist-like camera) of a box standing on the table z = 0."""
    wrist = look_at([centre[0] - 0.12, centre[1] + 0.05, 0.16], [centre[0], centre[1], 0.0]) if wrist is None else wrist
    sides = np.broadcast_to(np.asarray(size, float), (3,))
    middle = np.array([centre[0], centre[1], sides[2] / 2])
    out = []
    for T, delta in ((fixed, (0.0, 0.0)), (wrist, shift)):
        outline = np.asarray(box_outline(T, middle, sides), float) + np.asarray(delta)
        out.append((K640, T, [outline], (640, 480)))
    return out


class ContactTop(unittest.TestCase):
    """P4: the two-view table-contact top height."""

    def test_box_top_is_recovered_without_monocular_depth(self):
        rng = np.random.default_rng(3)
        errors = []
        for _ in range(6):
            centre = [0.20 + 0.1 * rng.random(), -0.05 + 0.1 * rng.random()]
            size = [0.02 + 0.04 * rng.random(), 0.02 + 0.03 * rng.random(), 0.015 + 0.05 * rng.random()]
            found = contact_height.contact_top(prism_views(centre, size, shift=rng.normal(0, 1.0, 2)), 0.0, 0.3)
            self.assertIn("contact_top_z", found, found)
            errors.append(found["contact_top_z"] - size[2])
            low, high = found["contact_top_z_range"]
            self.assertLessEqual(low, found["contact_top_z"])
            self.assertGreaterEqual(high, found["contact_top_z"])
            self.assertGreater(found["contact_fit_iou"], 0.85)
        self.assertLess(float(np.percentile(np.abs(errors), 50)), 0.002, errors)
        self.assertLess(float(np.max(np.abs(errors))), 0.006, errors)

    def test_range_is_null_when_a_changed_outline_cannot_be_fitted(self):
        views = prism_views([0.25, 0.0], 0.03)
        tiny = np.array([[300.0, 200.0], [302.0, 200.0], [301.0, 202.0]])     # erodes to nothing
        found = contact_height.contact_top([views[0], (K640, views[1][1], [tiny], (640, 480))], 0.0, 0.3)
        self.assertTrue("contact_reason" in found or found["contact_top_z_range"] is None, found)

    def test_reasons_instead_of_a_value(self):
        views = prism_views([0.25, 0.0], 0.03)
        clipped = [views[0], (K640, views[1][1], [np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]])], (640, 480))]
        self.assertEqual(contact_height.contact_top(clipped, 0.0, 0.3), dict(contact_reason="mask_at_image_border:target"))
        empty = [views[0], (K640, views[1][1], [], (640, 480))]
        self.assertEqual(contact_height.contact_top(empty, 0.0, 0.3), dict(contact_reason="outline_missing:target"))
        far = prism_views([0.10, 0.20], 0.03)
        apart = [views[0], far[1]]
        self.assertIn(contact_height.contact_top(apart, 0.0, 0.3)["contact_reason"],
                      ("views_do_not_overlap_on_table", "no_consistent_footprint"))


class ContactQuery(So101Scene, unittest.TestCase):
    """P4 through geometry_query object_table with target_camera_ids: fields present, rows never overwritten."""

    Q = np.array([0.0, -0.6, 0.5, 1.3, 0.0])     # TCP 9 cm above the table; the wrist camera looks at CENTRE
    CENTRE = np.array([0.253, -0.024, 0.015])     # a 30 mm cube on the table, inside both views

    def rows_for(self, T, centre, top_reported, size=0.03):
        item = fake_object(tuple(centre), top_reported, 0.0, elongation=1.0)
        item.update(length_m=size, width_m=size, contour_px=[[int(round(u)), int(round(v))] for u, v in
                                                             box_outline(T, centre, size)])
        item["label_px"] = np.rint(np.mean(item["contour_px"], axis=0)).astype(int).tolist()
        return item

    def test_contact_top_beside_an_unchanged_row(self):
        from robot.so101.test_so101_held_perception import GeometryTools_camera
        self.observe(0.8, 0.5)
        runtime = self.runtime("http://127.0.0.1:9", held=False)
        wrist_T = GeometryTools_camera(runtime, "wrist")["T"]
        centre = self.CENTRE                                    # a 30 mm cube on the table (top 0.03 m)
        fixed_row = self.rows_for(self.fixed, centre, 0.012)    # monocular top 18 mm too low, as on the real arm
        wrist_row = self.rows_for(wrist_T, centre, 0.026)
        fake = CameraService({"fixed": [fixed_row], "wrist": [wrist_row]}, {"fixed": self.fixed, "wrist": wrist_T})
        self.addCleanup(fake.close)
        result, _ = self.query(self.runtime(fake.url, held=False), target_camera_ids=["wrist"])
        self.assertEqual(result["status"], "conditional", result.get("reason"))
        self.assertEqual(result["value"]["rows"][0][2], 0.012)               # the measured row is never replaced
        row = result["value"]["cross_view"]["wrist"]["rows"][0]
        self.assertEqual(row["target_view_top_z"], 0.026)
        self.assertIn("contact_top_z", row, row)
        self.assertLess(abs(row["contact_top_z"] - 0.03), 0.004, row)
        self.assertEqual(len(row["contact_top_z_range"]), 2)
        self.assertIn("contact_top_z:", result["assumptions"][-2])
        self.assertIn("cross_view:", result["assumptions"][-1])

    def test_stacked_row_is_fitted_with_its_support_and_unknown_support_has_a_reason(self):
        from robot.so101.test_so101_held_perception import GeometryTools_camera
        self.observe(0.8, 0.5)
        runtime = self.runtime("http://127.0.0.1:9", held=False)
        wrist_T = GeometryTools_camera(runtime, "wrist")["T"]
        lower, upper = self.CENTRE, self.CENTRE + [0.0, 0.0, 0.03]
        fixed_rows = [self.rows_for(self.fixed, lower, 0.03), self.rows_for(self.fixed, upper, 0.05)]
        fixed_rows[1]["support"] = 0
        fixed_rows[1]["centre_m"] = upper.tolist()
        wrist_rows = [self.rows_for(wrist_T, lower, 0.03), self.rows_for(wrist_T, upper, 0.06)]
        fake = CameraService({"fixed": fixed_rows, "wrist": wrist_rows}, {"fixed": self.fixed, "wrist": wrist_T})
        self.addCleanup(fake.close)
        result, _ = self.query(self.runtime(fake.url, held=False), target_camera_ids=["wrist"])
        rows = {row["id"]: row for row in result["value"]["cross_view"]["wrist"]["rows"]}
        self.assertIn("o2", rows)
        self.assertLess(abs(rows["o2"]["contact_top_z"] - 0.06), 0.006, rows["o2"])   # the stack's top
        self.assertEqual(object_table._support_chain([dict(support="unknown")], 0), None)
        self.assertEqual(object_table._support_chain([dict(support="table"), dict(support=0)], 1), [1, 0])
        self.assertIsNone(object_table._support_chain([dict(support=1), dict(support=0)], 0))   # a loop

    def test_unpaired_support_is_named_and_a_failed_fit_keeps_the_cross_view_row(self):
        from robot.so101.test_so101_held_perception import GeometryTools_camera
        self.observe(0.8, 0.5)
        runtime = self.runtime("http://127.0.0.1:9", held=False)
        wrist_T = GeometryTools_camera(runtime, "wrist")["T"]
        lower, upper = self.CENTRE, self.CENTRE + [0.0, 0.0, 0.03]
        fixed_rows = [self.rows_for(self.fixed, lower, 0.03), self.rows_for(self.fixed, upper, 0.05)]
        fixed_rows[1].update(support=0, centre_m=upper.tolist())
        wrist_rows = [self.rows_for(wrist_T, upper, 0.06)]                     # the lower cube not segmented there
        fake = CameraService({"fixed": fixed_rows, "wrist": wrist_rows}, {"fixed": self.fixed, "wrist": wrist_T})
        self.addCleanup(fake.close)
        result, _ = self.query(self.runtime(fake.url, held=False), target_camera_ids=["wrist"])
        rows = {row["id"]: row for row in result["value"]["cross_view"]["wrist"]["rows"]}
        self.assertEqual(rows["o2"].get("contact_unpaired_support"), ["o1"], rows["o2"])
        original = contact_height.contact_top

        def broken(*args, **kwargs):
            raise cv2.error("synthetic")
        contact_height.contact_top = broken
        self.addCleanup(setattr, contact_height, "contact_top", original)
        result, _ = self.query(self.runtime(fake.url, held=False), target_camera_ids=["wrist"])
        row = result["value"]["cross_view"]["wrist"]["rows"][0]
        self.assertEqual(row["contact_reason"], "fit_failed")
        self.assertIn("stereo_centre_m", row)                                  # the cross-view row survives

    def test_tag_off_leaves_cross_view_without_contact_fields(self):
        from robot.so101.test_so101_held_perception import GeometryTools_camera
        self.observe(0.8, 0.5)
        runtime = self.runtime("http://127.0.0.1:9", held=False)
        wrist_T = GeometryTools_camera(runtime, "wrist")["T"]
        centre = self.CENTRE
        fake = CameraService({"fixed": [self.rows_for(self.fixed, centre, 0.02)],
                              "wrist": [self.rows_for(wrist_T, centre, 0.03)]}, {"fixed": self.fixed, "wrist": wrist_T})
        self.addCleanup(fake.close)
        profile = so101_deployment.embodiment_profile(example_deployment(contact_top_estimate=False))
        self.assertNotIn("contact_top", profile.supports)
        profile = embodiments.EmbodimentProfile(**{**{f: getattr(PROFILE, f) for f in PROFILE.__dataclass_fields__},
                                                  "supports": PROFILE.supports - {"contact_top"}})
        result, _ = self.query(self.runtime(fake.url, profile=profile, held=False), target_camera_ids=["wrist"])
        row = result["value"]["cross_view"]["wrist"]["rows"][0]
        self.assertNotIn("contact_top_z", row)
        self.assertIn("target_view_top_z", row)

    def test_attach_switch_adds_the_wrist_target_to_the_attached_table(self):
        profile = so101_deployment.embodiment_profile(example_deployment(attach_cross_view=True))
        self.assertIn("cross_view_attach", profile.supports)
        self.assertEqual(obs3d.attached_targets(profile, "fixed", ["fixed", "wrist"]), ["wrist"])
        self.assertEqual(obs3d.attached_targets(profile, "fixed", ["fixed"]), [])
        self.assertEqual(obs3d.attached_targets(PROFILE, "fixed", ["fixed", "wrist"]), [])   # default off
        self.assertEqual(obs3d.attached_targets(embodiments.ARX_X5_PROFILE, "cam_head", ["cam_head", "cam_left_wrist"]), [])


class TextBudget(unittest.TestCase):
    """Model-visible additions stay short and appear only on the SO-101 tags."""

    def test_developer_instruction_and_catalogue_additions(self):
        from ni_rua.core.config import Config
        features = Config().with_overrides({"features": {"object_table": True, "object_table_observe": True}}).features
        so101 = obs3d.developer_instructions(features, PROFILE)
        base = embodiments.EmbodimentProfile(**{**{f: getattr(PROFILE, f) for f in PROFILE.__dataclass_fields__},
                                               "supports": PROFILE.supports - {"robot_occlusion", "contact_top"}})
        before = obs3d.developer_instructions(features, base)
        self.assertIn("robot_occlusion", so101)
        self.assertIn("contact_top_z", so101)
        self.assertLess(len(so101) - len(before), 1150)
        self.assertNotIn("robot_occlusion", obs3d.developer_instructions(features, embodiments.ARX_X5_PROFILE))
        self.assertLess(len(object_table.CONTACT_TOP_FIELD), 260)

    def test_occlusion_value_size(self):
        hidden = [dict(id="o%d" % i, seen_in="obs-" + "0123456789abcdef" * 2, centre_m=[0.279, -0.064, -0.005],
                       top_z=0.013, covered=0.92) for i in range(1, 4)]
        value = dict(table_covered=0.33, hidden=hidden)
        self.assertLess(len(canonical(value)), 450)              # three hidden rows
        self.assertLess(len(canonical(dict(table_covered=0.13))), 30)


def load_bench():
    spec = importlib.util.spec_from_file_location("so101_perception_bench",
                                                  REPOSITORY / "scripts" / "so101_perception_bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BenchMetrics(unittest.TestCase):
    """The bench's metric code on the recorded real tables (reference cubes from the first, unoccluded table)."""

    @classmethod
    def setUpClass(cls):
        cls.bench = load_bench()
        fixed = REAL["fixed_camera"]
        cls.truth = dict(table_z_m=fixed["table_z_m"], cubes=REAL["reference_cubes"])
        cam = dict(K=fixed["K"], T=fixed["T_world_camera"], width=640, height=480)
        body = hp.SO101Body()
        meshes = body.meshes(None)
        roi = robot_view.table_roi(np.asarray(cam["K"]), np.asarray(cam["T"]), 640, 480, fixed["table_z_m"])
        records, previous = [], None
        for frame in REAL["frames"]:
            poses = {"right": kin.fk_all(MODEL, frame["q_rad"][:5], gripper_rad=frame["q_rad"][5])}
            robot, base = robot_view.silhouettes(body, meshes, poses, dict(cam, K=np.asarray(cam["K"]),
                                                                          T=np.asarray(cam["T"])), "fixed",
                                                 {"right": "wrist"})
            hidden = []
            for item in previous or []:
                fraction = robot_view.box_covered(np.asarray(cam["K"]), np.asarray(cam["T"]), robot,
                                                  robot_view.row_box(item))
                if fraction is not None and round(fraction, 2) > 0:
                    hidden.append(dict(id=item["id"], seen_in="previous", centre_m=item["centre_m"],
                                       top_z=item["top_z_m"], covered=round(fraction, 2)))
            table = dict(columns=frame["columns"], rows=frame["rows"],
                         robot_occlusion=dict(table_covered=round(robot_view.coverage(robot, base, roi), 2),
                                              **({"hidden": hidden} if hidden else {})))
            pose = "parked" if not frame["rows"] else "clear"
            records.append(dict(frame=len(records), pose=pose, height_method="vggt", fixed_size=[640, 480],
                                shown=dict(object_table=table), q_rad=frame["q_rad"], table_z_m=fixed["table_z_m"],
                                fixed_camera=cam, latency=dict(attach_s=0.5, object_table_s=0.4, cross_view_s=None),
                                sizes=dict(fixed_image_bytes=40000, text_chars=len(canonical(table)))))
            if frame["rows"]:
                previous = frame["objects"]
        cls.records = records

    def test_detection_and_errors_per_condition(self):
        covered = self.bench.occlusion_truth(self.records, self.truth)
        metrics = self.bench.score_records(self.records, self.truth, match_radius_m=0.03, truth_covered=covered)
        self.assertEqual(set(metrics), {"640x480|clear|vggt", "640x480|parked|vggt"})
        clear, parked = metrics["640x480|clear|vggt"], metrics["640x480|parked|vggt"]
        self.assertEqual(parked["frames"], 2)
        self.assertEqual(clear["cubes"]["blue"]["detection_rate"], round(2 / 4, 3))   # blue was moved after frame 3
        for colour in ("blue", "white", "orange"):
            self.assertEqual(parked["cubes"][colour]["detection_rate"], 0.0)
        white = clear["cubes"]["white"]
        self.assertEqual(white["detection_rate"], 1.0)
        self.assertEqual(white["xy_error_m"]["n"], 4)
        self.assertLess(white["xy_error_m"]["p90"], 0.03)
        self.assertLess(white["vggt_top_error_m"]["mean"], 0.0)       # the monocular top of the white cube reads low
        self.assertEqual(white["contact_top_error_m"]["n"], 0)        # no cross view in these tables
        occlusion = parked["cubes"]["orange"]["occlusion"]
        self.assertGreaterEqual(occlusion["truth_hidden_frames"], 1)
        self.assertEqual(occlusion["detected_when_hidden"], 0.0)
        self.assertGreater(parked["table_covered"]["p50"], 0.3)
        self.assertLess(clear["table_covered"]["p50"], 0.2)

    def test_capture_targets_are_checked_before_the_arm_moves(self):
        deployment = example_deployment()
        self.bench.check_targets([("home", list(deployment["home"]["q_rad"])), ("current", None)], deployment, -0.02)
        with self.assertRaisesRegex(SystemExit, "below the table"):
            self.bench.check_targets([("low", list(deployment["home"]["q_rad"]))], deployment, 0.05)
        with self.assertRaisesRegex(SystemExit, "planning limits"):
            self.bench.check_targets([("far", [3.0, 0.0, 0.0, 0.0, 0.0])], deployment, -0.02)

    def test_matching_and_token_estimates(self):
        rows = [dict(centre_m=[0.0, 0.0, 0.0]), dict(centre_m=[0.02, 0.0, 0.0]), dict(centre_m=[0.5, 0.0, 0.0], support="held_right")]
        cubes = [dict(centre_xy_m=[0.015, 0.0]), dict(centre_xy_m=[0.0, 0.0]), dict(centre_xy_m=[0.5, 0.0])]
        self.assertEqual(self.bench.match_rows(rows, cubes, 0.03), {0: 1, 1: 0})   # greedy nearest, held row unused
        self.assertEqual(self.bench.token_estimates(640, 480), dict(tile_512=425, pixels_750=410))
        self.assertEqual(self.bench.token_estimates(1920, 1080), dict(tile_512=1105, pixels_750=1844))
        self.assertEqual(self.bench.token_estimates(1280, 720), dict(tile_512=1105, pixels_750=1229))
        self.assertEqual(self.bench.parse_size("1920x1080"), (1920, 1080))

    def test_contact_range_and_cross_view_fields_are_scored_separately(self):
        record = dict(self.records[0])
        table = json.loads(json.dumps(record["shown"]["object_table"]))
        rows = self.bench.table_rows(table)
        white = next(row for row in rows if abs(row["centre_m"][0] - 0.282) < 0.01)
        table["cross_view"] = {"wrist": {"rows": [dict(id=white["id"], target_view_top_z=0.010, contact_top_z=0.014,
                                                        contact_top_z_range=[0.012, 0.016])]}}
        record["shown"] = dict(object_table=table)
        scores = self.bench.frame_scores(record, self.truth, match_radius_m=0.03)
        score = scores[1]
        top = self.truth["cubes"][1]["top_z_m"]
        self.assertAlmostEqual(score["cross_view_top_error_m"], round(0.010 - top, 4))
        self.assertAlmostEqual(score["contact_top_error_m"], round(0.014 - top, 4))
        self.assertTrue(score["contact_range_hit"])
        self.assertAlmostEqual(score["vggt_top_error_m"], round(white["top_z"] - top, 4))


class BenchPipeline(unittest.TestCase):
    """capture -> perceive -> score without hardware: a worker-format capture directory, the harness attachment path
    against a fake VGGT service, and the metrics file."""

    def raw_observation(self, directory, q, index):
        image = cv2.imencode(".jpg", np.full((480, 640, 3), 128, np.uint8))[1].tobytes()
        (directory / "fixed.jpg").write_bytes(image)
        (directory / "wrist.jpg").write_bytes(image)
        state = {"arms": {"right": arm_state(np.asarray(q), 0.3, 0.5)}, "frame_id": "so101_base",
                 "tcp_config_id": "so101-gripper-frame-v1", "sim_step": index, "step_domain": "so101_control_step"}
        link = kin.fk_all(MODEL, q, gripper_rad=0.3)[FIXTURE["link_name"]]
        calibration = {"profile_id": "so101-charuco-v1", "reference_frame_id": "so101_base",
                       "joint_calibration": {"table_z_m": 0.0}, "cameras": {
                           "fixed": {"K": K640.tolist(), "distortion": [0.0] * 5, "distortion_model": "opencv_plumb_bob",
                                     "image_size": [640, 480], "T_world_camera": FIXED.tolist()},
                           "wrist": {"K": K640.tolist(), "distortion": [0.0] * 5, "distortion_model": "opencv_plumb_bob",
                                     "image_size": [640, 480], "arm": "right", "link_name": FIXTURE["link_name"],
                                     "T_link_camera": FIXTURE["T_link_camera"]}}}
        poses = {"fixed": {"T_world_camera": FIXED.tolist(), "source_kind": "static_extrinsic_calibration"},
                 "wrist": {"T_world_camera": (link @ np.asarray(FIXTURE["T_link_camera"])).tolist(),
                           "T_world_link": link.tolist(), "T_world_base": np.eye(4).tolist(), "arm": "right",
                           "link_name": FIXTURE["link_name"], "source_kind": "robot_kinematics_at_joint_read"}}
        return {"raw_observation_id": "raw-%d" % index, "data_origin": "so101_real_hardware", "world_kind": "observed",
                "observation_profile": "rgb_proprio_calibrated_v1", "robot_model_id": "so101",
                "cameras": [{"camera_id": name, "rgb_path": name + ".jpg", "width": 640, "height": 480}
                            for name in ("fixed", "wrist")],
                "robot_state": state, "sim_step": index, "step_domain": "so101_control_step", "sim_time_s": None,
                "captured_at": "2026-11-16T00:00:00Z", "camera_calibration": calibration, "robot_model": MODEL,
                "camera_pose_sample": {"reference_frame_id": "so101_base", "sim_step": index,
                                       "step_domain": "so101_control_step", "cameras": poses}}

    def test_perceive_and_score_a_capture(self):
        bench = load_bench()
        work = Path(tempfile.mkdtemp(prefix="so101-bench-test-"))
        self.addCleanup(shutil.rmtree, work, True)
        clear_q, parked_q = [0.3116, -1.2715, 0.41, 0.536, 0.0], OVER_CUBE_Q
        cube_xy = under_tcp(parked_q)
        frames = []
        for index, (pose, q) in enumerate((("clear", clear_q), ("parked", parked_q))):
            directory = work / "frames" / ("%03d-%s" % (index, pose))
            directory.mkdir(parents=True)
            raw = self.raw_observation(directory, q, index)
            (directory / "observation.json").write_text(json.dumps(dict(pose=pose, observation=raw)))
            frames.append(dict(index=index, pose=pose, directory=directory.name))
        (work / "capture.json").write_text(json.dumps(dict(deployment=str(EXAMPLE), frames=frames)))
        row = fake_object((cube_xy[0], cube_xy[1], 0.015), 0.012, 0.0, elongation=1.0)
        row.update(length_m=0.03, width_m=0.03)
        wrist = kin.fk_all(MODEL, clear_q, gripper_rad=0.3)[FIXTURE["link_name"]] @ np.asarray(FIXTURE["T_link_camera"])
        fake = CameraService({"fixed": [row]}, {"fixed": FIXED, "wrist": wrist})
        self.addCleanup(fake.close)
        bench.main(["perceive", "--capture", str(work), "--endpoint", fake.url, "--height-method", "vggt",
                    "--timeout-s", "5"])
        records = bench.load_records(work)
        with self.assertRaises(SystemExit):                     # one file per method, never appended to
            bench.main(["perceive", "--capture", str(work), "--endpoint", fake.url, "--height-method", "vggt"])
        self.assertEqual([record["pose"] for record in records], ["clear", "parked"])
        table = records[1]["shown"]["object_table"]
        self.assertIn("robot_occlusion", table)
        self.assertEqual(records[1]["fixed_size"], [640, 480])
        self.assertGreater(records[1]["sizes"]["fixed_image_bytes"], 0)
        self.assertGreater(records[1]["sizes"]["overlay_image_bytes"], 0)
        self.assertEqual(records[1]["sizes"]["fixed_tokens_tile_512"], 425)
        truth = dict(table_z_m=0.0, cubes=[dict(colour="red", centre_xy_m=[float(v) for v in cube_xy], top_z_m=0.03)])
        truth_path = work / "truth.json"
        truth_path.write_text(json.dumps(truth))
        bench.main(["score", "--capture", str(work), "--truth", str(truth_path)])
        metrics = json.loads((work / "metrics.json").read_text())["conditions"]
        self.assertEqual(set(metrics), {"640x480|clear|vggt", "640x480|parked|vggt"})
        red = metrics["640x480|clear|vggt"]["cubes"]["red"]
        self.assertEqual(red["detection_rate"], 1.0)
        self.assertAlmostEqual(red["vggt_top_error_m"]["mean"], -0.018, places=3)
        parked = metrics["640x480|parked|vggt"]["cubes"]["red"]["occlusion"]
        self.assertEqual(parked["truth_hidden_frames"], 1)                     # the gripper hides the cube
        self.assertEqual(parked["reported_when_hidden"], 1.0)                  # ... and the report says so


if __name__ == "__main__":
    unittest.main()
