"""
- test_depth_integration.py: End-to-end CPU RGB-D publication, actor gates and numerical deliveries.
- test_geometry.py: Meaningful component/failure checks: python -m unittest discover -s tests -p test_geometry.py.
- test_grasp_contact.py: Selected finger contacts retain a calibrated, orientation-aware tool frame.

These CPU checks establish coordinate and diagnostic behavior, not retention,
contact safety, or the benefit of any contact region on a physical object.
"""
from support.program_fixture import program_services
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from ni_rua.core.config import Config
from ni_rua.core.contracts import canonical
from ni_rua.perception.depth.depth_payload import compact_action_feedback
from ni_rua.apps.gateway.workbench import Workbench
from ni_rua.runtime.harness import Harness
from ni_rua.core.observation_policy import ACTOR_PROFILE, DEPTH_ACTOR_PROFILE
from ni_rua.runtime.programs.runner import ProgramRunner
from ni_rua.robot.backends.robodojo.robodojo import FRAME_ID, TCP_CONFIG_ID
from ni_rua.core.storage.store import Store
from ni_rua.core.contracts import canonical,digest
from ni_rua.robot.model.geometry import GeometryTools,fit_geometry,triangulate,fk,project
from ni_rua.perception.scene.store import SceneStore
from support.geometry_fixture import IDENTITY,recorded_plane
from ni_rua.robot.model import roboprobe_pose
from ni_rua.robot.motion.arx_actions import compile_action
from ni_rua.robot.model.grasp_contact import contact_target_feedback, resolve_contact_spec, values_to_contact_pose
from ni_rua.robot.motion.motion_failure import failure_feedback
from support.fixture_robot import FixtureRobot

NI_RUA_TEST_TIER = "component"


# --- Folded from tests/test_depth_integration.py ---

class DepthRobot:
    def __init__(self, root):
        self.root = Path(root)
        self.executions = 0
        self.position = [0., 0., .7501]
        self.goal = 1.
        self.observation_profile = DEPTH_ACTOR_PROFILE
        texture = np.random.default_rng(8).integers(0, 255, (96, 96, 3), dtype=np.uint8)
        self.image = self.root / "rgb.png"
        cv2.imwrite(str(self.image), cv2.GaussianBlur(texture, (3, 3), 0))
        self.depth = self.root / "depth.npz"
        np.savez_compressed(self.depth, depth=np.full((96, 96), .5, np.float32), valid=np.ones((96, 96), bool))

    def start(self):
        return dict(data_origin="synthetic_fixture", frame_id=FRAME_ID, tcp_config_id=TCP_CONFIG_ID,
                    path_kinds=["backend_free"], max_waypoints=8, dual_arm=False, gripper_after_motion=True)

    def observe(self):
        frame_id = "f" + str(self.executions)
        pose = np.diag([1., -1., -1., 1.]); pose[2, 3] = 1.
        robot = dict(source_kind="measurement", frame_id=FRAME_ID, sim_step=self.executions,
                     step_domain="adapter_control_step", arms=dict(left=dict(
                         q_rad=[0.] * 6, joint_names=["joint" + str(i) for i in range(1, 7)],
                         source_kind="measurement", ee_pose=dict(position_m=self.position,
                             quaternion_xyzw=[-.5, .5, .5, .5], frame_id=FRAME_ID,
                             tcp_config_id=TCP_CONFIG_ID, source_kind="measurement"),
                         gripper=dict(command_normalized=self.goal, command_source_kind="command",
                             goal_normalized=self.goal, goal_source_kind="command",
                             joint_names=["joint7", "joint8"], position_m=[.02, .02], position_source_kind="measurement"))))
        camera = dict(camera_id="head", frame_id=frame_id, width=96, height=96,
                      png_path=str(self.image), depth_path=str(self.depth), depth_kind="z", depth_unit="m",
                      depth_clip_m=[.01, 10.], depth_shape=[96, 96])
        calibration = dict(reference_frame_id=FRAME_ID, cameras=dict(head=dict(
            K=[[200., 0., 48.], [0., 200., 48.], [0., 0., 1.]], distortion=[0.] * 5,
            image_size=[96, 96], T_world_camera=pose.tolist())))
        return dict(observation_profile=DEPTH_ACTOR_PROFILE, data_origin="synthetic_fixture", world_kind="observed",
                    cameras=[camera], camera_calibration=calibration, robot_model=dict(base_link="base", joints=[]),
                    camera_pose_sample=dict(sim_step=self.executions, step_domain="adapter_control_step",
                        cameras=dict(head=dict(T_world_camera=pose.tolist(), frame_id=frame_id, source_kind="measurement"))),
                    robot_state=robot, sim_step=self.executions, step_domain="adapter_control_step")

    def execute(self, action, progress_callback=None, cancel_flag=None):
        self.executions += 1
        for waypoint in action.get("path", {}).get("waypoints", []):
            self.position = waypoint["left"]["position_m"]
        for event in action.get("gripper_events", []):
            self.goal = event["goal_normalized"]
        return dict(status="completed", terminal_observation=self.observe(), executed_native_steps=1)


class DepthIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "store", dict(run_id="depth-test", episode_id="ep", env_instance_id="env", env_generation=1))
        self.addCleanup(self.store.close)
        self.robot = DepthRobot(self.root)

    def harness(self, **features):
        config = Config.from_dict(dict(features=dict(observation_profile=DEPTH_ACTOR_PROFILE, motion_review=False, **features)))
        harness = Harness(self.store, self.robot, config=config)
        harness.start()
        harness.capabilities.update(bounded_wait=dict(max_native_steps=50), path_kinds=["backend_free", "hold"])
        return harness

    def wait(self, harness, **args):
        return harness.dispatch("robot_wait", dict(observation_id=harness.latest["observation_id"], native_steps=5, **args),
                                dict(tool_call_id=args.get("action_id", "wait")))

    def query(self, harness, **request):
        result = harness.dispatch("depth_query", dict(request=request), {})
        self.assertEqual(len(result["contentItems"]), 1)
        return json.loads(result["contentItems"][0]["text"])

    def test_geometry_depth_compatibility_keeps_owner_validation_for_both_roles(self):
        harness = self.harness()
        request = dict(kind="depth_points", observation_id=harness.latest["observation_id"],
                       camera_id="head", pixels=[[48, 48]])
        target = harness.invoke("depth_query", dict(request=dict(kind="register_target",
            observation_id=request["observation_id"], camera_id="head", roi=[36, 36, 24, 24])), {}, role="program")
        target_id = target.data["target_id"]
        for role in ("actor", "program"):
            result = harness.invoke("geometry_query", dict(request=request), {}, role=role)
            self.assertEqual(result.data["status"], "supported")
            for invalid, reason in ((dict(request, radius_px=5), "invalid_radius_px"),
                    (dict(kind="region_geometry", target_id=target_id, observation_id=request["observation_id"]),
                     "depth_target_fields_cannot_be_overridden"),
                    (dict(kind="refresh_target", target_id=target_id, observation_id=request["observation_id"],
                          camera_id="head", roi=[0, 0, 4, 4]), "invalid_depth_target_fields")):
                with self.subTest(role=role, invalid=invalid), self.assertRaisesRegex(ValueError, reason):
                    harness.invoke("geometry_query", dict(request=invalid), {}, role=role)
        self.assertEqual(self.robot.executions, 0)

    def test_default_profile_hides_tools_and_rejects_all_depth_routes(self):
        from support.fixture_robot import FixtureRobot
        config = Config.from_dict({})
        harness = Harness(self.store, FixtureRobot(self.root), config=config, motion_review=False)
        for name, args in (("depth_query", dict(request=dict(kind="help"))),
                           ("geometry_query", dict(request=dict(kind="depth_points", observation_id="obs",
                                                                camera_id="head", pixels=[[1, 2]])))):
            with self.subTest(tool=name), self.assertRaisesRegex(ValueError, "depth_not_enabled"):
                harness.dispatch(name, args, {})
        workbench = Workbench.__new__(Workbench)
        workbench.config = config
        workbench.pi05_enabled = False
        workbench.visual_review_model = None
        workbench.observation_profile = ACTOR_PROFILE
        self.assertNotIn("depth_query", {item["name"] for item in workbench.actor_tools()})
        with self.assertRaises(ValueError):
            harness.publish_observation(self.robot.observe())

    def test_binary_publication_numeric_query_and_candidate_identity(self):
        harness = self.harness()
        obs = harness.latest
        camera = obs["cameras"][0]
        info, raw = self.store.resolve_artifact(camera["depth_ref"])
        self.assertEqual(info["kind"], "depth")
        with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
            self.assertEqual(arrays["depth"].dtype, np.float32)
        points = self.query(harness, kind="depth_points", observation_id=obs["observation_id"], camera_id="head", pixels=[[48, 48]])
        self.assertEqual(points["status"], "supported")
        np.testing.assert_allclose(points["value"]["points"][0]["point_m"], [0, 0, .5], atol=1e-6)
        self.assertLess(len(canonical(points)), 2000)
        again = self.query(harness, kind="depth_points", observation_id=obs["observation_id"], camera_id="head", pixels=[[48, 48]])
        self.assertEqual(again["measurement_id"], points["measurement_id"])
        target = self.query(harness, kind="register_target", observation_id=obs["observation_id"], camera_id="head", roi=[36, 36, 24, 24], arm="left")
        measured = self.query(harness, kind="region_geometry", target_id=target["target_id"])
        self.assertIn("visible_dimensions_m", measured["value"])
        candidates = self.query(harness, kind="grasp_candidates", target_id=target["target_id"])
        self.assertTrue(candidates["value"]["candidates"], candidates)
        self.wait(harness)
        self.assertEqual(self.robot.executions, 1)
        with self.assertRaisesRegex(ValueError, "current_observation"):
            self.query(harness, kind="region_geometry", target_id=target["target_id"])

    def test_rgbd_history_and_program_artifacts_cannot_cross_rgb_boundary(self):
        harness = self.harness()
        obs = harness.latest
        harness.observation_profile = ACTOR_PROFILE
        with self.assertRaisesRegex(ValueError, "matching_observation"):
            harness.dispatch("history_view_observation", dict(observation_id=obs["observation_id"]), {})
        runner = ProgramRunner(program_services(harness))
        self.addCleanup(runner.close)
        entry = dict(cancel=threading.Event(), deadline=time.monotonic() + 30, call_lock=threading.RLock())
        with self.assertRaisesRegex(ValueError, "depth_artifact"):
            runner.dispatch(entry, dict(artifact_ref=obs["cameras"][0]["depth_ref"]))
        ref = self.store.put_artifact(b'{"depth_derived":true,"value":{"lift_m":0.02}}', "computed", "application/json")
        with self.assertRaisesRegex(ValueError, "privileged_data"):
            runner.dispatch(entry, dict(artifact_ref=ref))

    def test_persisted_measurement_does_not_bypass_source_integrity(self):
        harness = self.harness()
        request = dict(kind="depth_points", observation_id=harness.latest["observation_id"],
                       camera_id="head", pixels=[[48, 48]])
        self.query(harness, **request)
        reference = harness.latest["cameras"][0]["depth_ref"]
        _, path = self.store.artifact(reference)
        path.write_bytes(b"corrupted source")
        with self.assertRaisesRegex(ValueError, "artifact_digest_mismatch"):
            self.query(harness, **request)

    def test_depth_skills_are_separately_disabled_and_rgb_comparison_independent(self):
        harness = self.harness(vision_compare=False)
        self.assertIn("queries", self.query(harness, kind="help"))
        workbench = Workbench.__new__(Workbench)
        workbench.config = harness.config.with_overrides(dict(features=dict(depth_skills=True)))
        workbench.pi05_enabled = False
        workbench.visual_review_model = None
        workbench.vision_compare = False
        workbench.observation_profile = DEPTH_ACTOR_PROFILE
        tools = {item["name"]: item for item in workbench.actor_tools()}
        self.assertTrue(workbench.config.features.depth_skills)
        self.assertIn("depth_query", tools)
        self.assertIn("robot_action", tools)
        self.assertIn("operation_memory", tools)
        self.assertNotIn("skill_run", tools)
        self.assertNotIn("manipulation_run", tools)
        self.assertNotIn("vision_compare", tools)

    def test_numeric_delivery_keeps_exception_and_periodic_images(self):
        harness = self.harness(depth_auto_feedback=True, depth_event_payload=True)
        measurement = dict(status="supported", value=dict(evidence="rigid_surface_motion",
                           world_displacement_m=[0, 0, 0], world_rotation_deg=0))
        action = dict(action_id="a", path={}, gripper_events=[])
        result = dict(request=action, receipt=dict(status="completed", result_observation_id=harness.latest["observation_id"]))
        self.assertIsNotNone(compact_action_feedback(harness, result, [measurement], {}))
        action["action_id"] = "b"
        self.assertIsNotNone(compact_action_feedback(harness, result, [measurement], {}))
        action["action_id"] = "c"
        self.assertIsNone(compact_action_feedback(harness, result, [measurement], {}))
        self.assertIsNone(compact_action_feedback(harness, result, [{**measurement, "status": "uncertain"}], {}))
        self.assertIsNone(compact_action_feedback(harness, {**result, "receipt": {**result["receipt"], "status": "unknown"}}, [measurement], {}))
        self.assertIsNone(compact_action_feedback(harness, result, [measurement], dict(program_id="p")))
        self.assertIsNone(compact_action_feedback(harness, {**result, "request": {**action, "gripper_events": [dict(goal_normalized=1)]}}, [measurement], {}))

    def test_feedback_preserves_seed_and_old_action_does_not_mutate_new_target(self):
        harness = self.harness(depth_auto_feedback=True)
        first = harness.latest["observation_id"]
        self.wait(harness, action_id="empty-target-action")
        saved = self.store.read("depth_action_feedback", "empty-target-action")
        self.assertEqual(saved["measurements"], [])
        target = self.query(harness, kind="register_target", observation_id=harness.latest["observation_id"],
                            camera_id="head", roi=[30, 30, 35, 35], seed_pixel=[40, 40], arm="left")
        harness.dispatch("robot_wait", dict(observation_id=first, native_steps=5, action_id="empty-target-action"),
                         dict(tool_call_id="move1-repeat"))
        tracked = harness.depth._target(target["target_id"])
        self.assertEqual(tracked["status"], "tracked")
        fake = dict(status="supported", value=dict(after_roi=[31, 31, 35, 35], after_seed_pixel=[41, 41]))
        with patch.object(harness.depth, "_measure", return_value=fake) as measure:
            self.wait(harness, action_id="seed-action")
        self.assertEqual(measure.call_args.args[0]["seed_pixel"], [40, 40])
        self.assertEqual(harness.depth._target(target["target_id"])["seed_pixel"], [41, 41])

    def test_optional_perception_failure_preserves_completed_action_and_rgb(self):
        harness = self.harness(depth_auto_feedback=True, depth_event_payload=True)
        with patch.object(harness.depth, "feedback", side_effect=RuntimeError("synthetic perception failure")):
            result = self.wait(harness, action_id="perception-error")
        answer = json.loads(result["contentItems"][0]["text"])
        self.assertEqual(answer["receipt"]["status"], "completed")
        self.assertEqual(answer["depth_feedback"][0]["status"], "uncertain")
        self.assertTrue(any(item["type"] == "inputImage" for item in result["contentItems"]))
        self.assertEqual(self.robot.executions, 1)



# --- Folded from tests/test_geometry.py ---

class GeometryCheck(unittest.TestCase):
    def test_gripper_relative_requires_matching_measured_x5_tcp(self):
        # SO-101's TCP is already near the fixed finger. Applying the X5
        # flange-to-pad transform moves the reported reference by 150 mm.
        for tcp, source, frame in (
            ("so101-gripper-frame-v1", "measurement", "env_origin"),
            (None, "measurement", "env_origin"),
            ("unknown-tcp", "measurement", "env_origin"),
            ("arx-x5-link6-v1", "command", "env_origin"),
            ("arx-x5-link6-v1", "measurement", "other-base"),
        ):
            with self.subTest(tcp=tcp, source=source, frame=frame):
                observation = deepcopy(self.obs[1])
                observation["observation_id"] = "unsupported-gripper"
                state = json.loads(self.artifacts.resolve_artifact(observation["robot_state_ref"])[1])
                state["arms"] = {"right": {"source_kind": "measurement", "ee_pose": {
                    "position_m": [.21255, .00054, .09431],
                    "quaternion_xyzw": [0., 1., 0., 0.],
                    "tcp_config_id": tcp, "frame_id": frame, "source_kind": source,
                }}}
                observation["robot_state_ref"] = self.artifacts.put_artifact(
                    canonical(state).encode(), "robot_state", "application/json")
                # Use a fresh identity because imported observations are immutable.
                observation["observation_id"] += f"-{tcp}-{source}-{frame}"
                self.scene.import_observation(observation)
                result = self.geometry.execute(self.plane_request(observation_id=observation["observation_id"]))
                self.assertEqual(result["status"], "conditional", result)
                np.testing.assert_allclose(result["value"]["point_m"], [.2, .1, 2], atol=1e-10)
                self.assertNotIn("gripper_relative", result["value"])

    def plane_request(self, **changes):
        return dict(IDENTITY, kind="pixel_plane", observation_id="obs-1", camera_id="head",
                    pixel=[345, 265], plane_origin_m=[0, 0, 2], plane_normal=[0, 0, 1]) | changes

    def test_so101_reference_uses_measured_swinging_jaw_and_tcp_rotation(self):
        from ni_rua.robot.model.so101_gripper_geometry import fingertip_reference
        first = dict(reference_frame_id="so101_base", state={"arms": {"right": {
            "source_kind": "measurement", "joint_names": ["gripper"], "q_rad": [0.],
            "ee_pose": dict(position_m=[.25, .02, .05], quaternion_xyzw=[0., 1., 0., 0.],
                            tcp_config_id="so101-gripper-frame-v1", frame_id="so101_base", source_kind="measurement")}}})
        midpoints = []
        for jaw in (0., .5):
            first["state"]["arms"]["right"]["q_rad"] = [jaw]
            value = {}
            self.geometry._attach_gripper_relative(first, value, [[.26, .03, .01]])
            ref = value["gripper_relative"]["right"]
            expected = np.array([.25, .02, .05]) + np.diag([-1, 1, -1]) @ np.array(fingertip_reference(jaw)["midpoint_m"])
            np.testing.assert_allclose(ref["grasp_position_m"], expected, atol=1e-5)
            self.assertEqual(ref["reference"], "cad_inner_fingertip_midpoint_at_measured_jaw")
            self.assertLess(np.linalg.norm(expected - [.25, .02, .05]), .1)
            midpoints.append(expected)
        self.assertGreater(np.linalg.norm(midpoints[1] - midpoints[0]), .01)
        # A command alone cannot stand in for the measured jaw angle.
        first["state"]["arms"]["right"]["joint_names"] = []
        value = {}; self.geometry._attach_gripper_relative(first, value, [])
        self.assertNotIn("gripper_relative", value)

    def test_pixel_plane_uses_saved_camera_and_explicit_plane_only(self):
        from ni_rua.perception.scene.scene_contracts import validate_response
        with patch.object(self.scene, "query", side_effect=AssertionError("no reconstructed scene")):
            result = self.geometry.execute(self.plane_request())
        self.assertEqual(result["status"], "conditional", result)
        np.testing.assert_allclose(result["value"]["point_m"], [.2, .1, 2], atol=1e-10)
        self.assertEqual(result["value"]["scope"], "caller_assumed_infinite_plane")
        self.assertIn("not measured depth", " ".join(result["assumptions"]))
        self.assertGreater(result["value"]["ray_plane_condition_factor"], 1)
        validate_response("geometry/query", result, method="POST")
        # Scaling or reversing the normal does not change the geometric plane.
        for normal in ([0, 0, -4], [0, 0, 1e300], [0, 0, 1e-300]):
            other = self.geometry.execute(self.plane_request(plane_normal=normal))
            np.testing.assert_allclose(other["value"]["point_m"], [.2, .1, 2], atol=1e-10)
        changed = self.geometry.execute(self.plane_request(plane_origin_m=[0, 0, 3]))
        np.testing.assert_allclose(changed["value"]["point_m"], [.25, .15, 3], atol=1e-10)

    def test_pixel_plane_optional_alt_plane_reports_assumption_delta(self):
        result = self.geometry.execute(self.plane_request(plane_origin_alt_m=[0, 0, 3]))
        self.assertEqual(result["status"], "conditional", result)
        np.testing.assert_allclose(result["value"]["point_m"], [.2, .1, 2], atol=1e-10)
        np.testing.assert_allclose(result["value"]["alternate_point_m"], [.25, .15, 3], atol=1e-10)
        np.testing.assert_allclose(result["value"]["d_point_d_normal_offset_m"], [.05, .05, 1.], atol=1e-10)
        self.assertEqual(result["value"]["cross_camera_projections"], {})
        self.assertEqual(result["value"]["plane_origin_alt_m"], [0., 0., 3.])
        self.assertAlmostEqual(result["value"]["alternate_ray_distance_m"], result["value"]["ray_distance_m"] * 1.5, places=10)
        self.assertAlmostEqual(result["value"]["assumption_delta_m"], np.linalg.norm([.05, .05, 1]), places=10)
        self.assertIn("not a depth error bound", " ".join(result["assumptions"]))
        from ni_rua.perception.scene.scene_contracts import validate_response
        validate_response("geometry/query", result, method="POST")
        behind = self.geometry.execute(self.plane_request(plane_origin_alt_m=[0, 0, -1]))
        self.assertEqual(behind["status"], "unknown", behind)
        self.assertIsNone(behind["value"])
        self.assertTrue(behind["reason"].startswith("ray_plane_not_in_front_of_camera"), behind)

    def test_pixel_plane_cross_camera_projections_and_target_overlay(self):
        observation = deepcopy(self.obs[1])
        observation["observation_id"] = "multi-cam"
        calibration = json.loads(self.artifacts.resolve_artifact(observation["cameras"][0]["calibration_ref"])[1])
        calibration["cameras"]["wrist"] = deepcopy(calibration["cameras"]["head"])
        cal_ref = self.artifacts.put_artifact(canonical(calibration).encode(), "calibration", "application/json")
        observation["cameras"][0]["calibration_ref"] = cal_ref
        wrist_cam = deepcopy(observation["cameras"][0])
        wrist_cam.update(camera_id="wrist", frame_id="frame-wrist", calibration_ref=cal_ref)
        observation["cameras"].append(wrist_cam)
        state = json.loads(self.artifacts.resolve_artifact(observation["robot_state_ref"])[1])
        t_wrist = np.eye(4); t_wrist[0, 3] = 0.20
        state["camera_poses"]["wrist"] = dict(T_world_camera=t_wrist.tolist(), frame_id="frame-wrist",
                                              sim_step=state["sim_step"], step_domain=state["step_domain"],
                                              source_kind="robot_kinematics")
        state["arms"] = {
            "left": {
                "source_kind": "measurement",
                "ee_pose": {
                    "position_m": [0.05, 0.10, 1.85],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "frame_id": "env_origin",
                    "tcp_config_id": "arx-x5-link6-v1",
                    "source_kind": "measurement",
                },
            }
        }
        observation["robot_state_ref"] = self.artifacts.put_artifact(canonical(state).encode(), "robot_state", "application/json")
        self.scene.import_observation(observation)
        result = self.geometry.execute(self.plane_request(
            observation_id="multi-cam", plane_origin_alt_m=[0, 0, 3], target_camera_id="wrist"))
        self.assertEqual(result["status"], "conditional", result)
        self.assertIn("gripper_relative", result["value"])
        np.testing.assert_allclose(result["value"]["gripper_relative"]["left"]["grasp_position_m"], [0.20007, 0.10, 1.85], atol=1e-4)
        np.testing.assert_allclose(result["value"]["gripper_relative"]["left"]["point_minus_grasp_world_m"], [-0.00007, 0.0, 0.15], atol=1e-4)
        np.testing.assert_allclose(result["value"]["gripper_relative"]["left"]["jaw_midplane_normal"], [0.0, 1.0, 0.0], atol=1e-4)
        cross = result["value"]["cross_camera_projections"]
        self.assertIn("wrist", cross)
        np.testing.assert_allclose(cross["wrist"]["pixel"], [320., 265.], atol=1e-8)
        np.testing.assert_allclose(cross["wrist"]["alternate_pixel"], [328.33333333, 265.], atol=1e-6)
        self.assertEqual(len(cross["wrist"]["d_pixel_d_normal_offset_px"]), 2)
        self.assertGreater(abs(cross["wrist"]["d_pixel_d_normal_offset_px"][0]), 1.0)
        self.assertEqual(result["value"]["target_camera_id"], "wrist")
        self.assertEqual(len(result["value"]["pixels"]), 2)
        self.assertTrue(result["value"]["overlay_ref"].startswith("artifact:"))
        from ni_rua.perception.scene.scene_contracts import validate_response
        validate_response("geometry/query", result, method="POST")
        single = self.geometry.execute(self.plane_request(observation_id="multi-cam", target_camera_id="wrist"))
        self.assertEqual(len(single["value"]["pixels"]), 1)
        validate_response("geometry/query", single, method="POST")

    def test_pixel_plane_respects_rotated_camera_distortion_and_sloped_plane(self):
        observation = deepcopy(self.obs[0])
        observation["observation_id"] = "rotated"
        calibration = json.loads(self.artifacts.resolve_artifact(observation["cameras"][0]["calibration_ref"])[1])
        calibration["cameras"]["head"]["distortion"]["coefficients"] = [.05, -.02, .001, -.001, 0]
        observation["cameras"][0]["calibration_ref"] = self.artifacts.put_artifact(
            canonical(calibration).encode(), "calibration", "application/json")
        state = json.loads(self.artifacts.resolve_artifact(observation["robot_state_ref"])[1])
        state["camera_poses"]["head"]["T_world_camera"] = [
            [1, 0, 0, 1], [0, 0, -1, 2], [0, 1, 0, 3], [0, 0, 0, 1]]
        observation["robot_state_ref"] = self.artifacts.put_artifact(
            canonical(state).encode(), "robot_state", "application/json")
        self.scene.import_observation(observation)
        camera = self.geometry.camera(IDENTITY, "rotated", "head")
        expected = [1.2, 0, 3.1]
        pixel = project(camera, [expected])[0][0].tolist()
        result = self.geometry.execute(self.plane_request(observation_id="rotated", pixel=pixel,
            plane_origin_m=expected, plane_normal=[1, 2, 3]))
        self.assertEqual(result["status"], "conditional", result)
        np.testing.assert_allclose(result["value"]["point_m"], expected, atol=1e-7)

    def test_pixel_plane_degeneracies_do_not_publish_a_point(self):
        cases = [
            (dict(plane_normal=[0, 0, 0]), "invalid_input", "invalid_plane_normal"),
            (dict(pixel=[320, 240], plane_normal=[1, 0, 0]), "unknown", "ray_plane_near_parallel"),
            (dict(pixel=[320, 240], plane_normal=[1, 0, .005]), "unknown", "ray_plane_near_parallel"),
            (dict(plane_origin_m=[0, 0, -1]), "unknown", "ray_plane_not_in_front_of_camera"),
            (dict(plane_origin_m=[0, 0, 0]), "unknown", "ray_plane_not_in_front_of_camera"),
            (dict(pixel=[640, 240]), "invalid_input", "pixel_out_of_bounds"),
            (dict(frame_id="other"), "invalid_input", "unsupported_frame"),
            (dict(observation_id="missing"), "unknown", None),
        ]
        for changes, status, reason in cases:
            with self.subTest(changes=changes):
                result = self.geometry.execute(self.plane_request(**changes))
                self.assertEqual(result["status"], status, result)
                self.assertIsNone(result["value"])
                if reason:
                    self.assertTrue(result["reason"].startswith(reason), result)

    def test_pixel_plane_actor_and_isolated_program_reuse_projection_without_motion(self):
        from ni_rua.runtime.harness import Harness
        from ni_rua.runtime.programs.runner import ProgramRunner
        from support.fixture_robot import FixtureRobot
        robot = FixtureRobot(Path(self.tmp.name))
        harness = Harness(self.artifacts, robot, observation_profile="rgb_proprio_calibrated_v1")
        harness.scene = self.scene
        harness.thread_id = "thread"
        request = {key: value for key, value in self.plane_request().items() if key not in IDENTITY}
        shown = harness.handle_tool(dict(threadId="thread", turnId="turn", callId="plane",
            tool="geometry_query", arguments=dict(request=request)))
        result = json.loads(shown["contentItems"][0]["text"])
        self.assertEqual(result["status"], "conditional")
        self.assertIn("not measured depth", " ".join(result["assumptions"]))
        self.assertNotIn("implementation_digest", result)
        overlay = harness.handle_tool(dict(threadId="thread", turnId="turn", callId="projection",
            tool="geometry_query", arguments=dict(request=dict(kind="project_points", observation_id="obs-1",
                camera_id="head", points_m=[result["value"]["point_m"]]))))
        self.assertTrue(any(item["type"] == "inputImage" for item in overlay["contentItems"]))
        runner = ProgramRunner(harness.program_services())
        self.addCleanup(runner.close)
        source = ("from ni_rua_client import NiRuaClient\n"
                  "r = NiRuaClient().call('geometry_query', " + repr(dict(request=request)) + ")\n"
                  "assert r['kind'] == 'data' and r['data']['status'] == 'conditional'\n"
                  "assert abs(r['data']['value']['point_m'][0] - .2) < 1e-9\n")
        job = runner.run(dict(source=source), dict(thread_id="thread", turn_id="turn", tool_call_id="program-plane"))
        self.assertEqual(job["status"], "succeeded", job)
        self.assertEqual(job["call_summary"]["failed_calls"], 0)
        self.assertEqual(robot.executions, 0)
        self.assertEqual(self.artifacts.records("action_request"), [])

    def test_pixel_plane_invalid_schema_never_reaches_geometry_at_either_port(self):
        from ni_rua.runtime.harness import Harness
        from ni_rua.runtime.programs.runner import ProgramRunner
        from support.fixture_robot import FixtureRobot
        robot = FixtureRobot(Path(self.tmp.name))
        harness = Harness(self.artifacts, robot, observation_profile="rgb_proprio_calibrated_v1")
        harness.thread_id = "thread"
        harness.scene = self.scene
        request = {key: value for key, value in self.plane_request().items() if key not in IDENTITY}
        invalid = [request | change for change in (
            dict(plane_origin_m=[0, True, 2]), dict(plane_normal=[0, 1]), dict(pixel=[False, 240]),
            dict(plane_normal=[0, 0, float("inf")]), dict(plane_origin_alt_m=[0, float("inf"), 3]),
            dict(plane_origin_alt_m=[0, 3]), dict(typo=True), dict(observation_ids=["obs-1"]))]
        invalid.append({key: value for key, value in request.items() if key != "plane_origin_m"})
        runner = ProgramRunner(harness.program_services())
        self.addCleanup(runner.close)
        with patch.object(GeometryTools, "execute", side_effect=AssertionError("invalid request reached geometry")) as execute:
            for i, bad in enumerate(invalid):
                with self.subTest(request=bad), self.assertRaises(ValueError):
                    harness.handle_tool(dict(threadId="thread", turnId="turn", callId=f"invalid-{i}",
                        tool="geometry_query", arguments=dict(request=bad)))
            bad = request | dict(typo=True)
            source = ("from ni_rua_client import NiRuaClient\n"
                      "r = NiRuaClient().call('geometry_query', " + repr(dict(request=bad)) + ")\n"
                      "assert r['success'] is False\n")
            job = runner.run(dict(source=source), dict(thread_id="thread", turn_id="turn", tool_call_id="program-invalid-plane"))
            self.assertEqual(job["status"], "succeeded", job)
            self.assertEqual(job["call_summary"]["failed_calls"], 1)
            execute.assert_not_called()
        self.assertEqual(robot.executions, 0)

    def test_public_geometry_metadata_preserves_measurements_and_complete_raw_result(self):
        from ni_rua.core.presentation.actor_output import ActorOutput
        from ni_rua.core.presentation.actor_presentation import ActorPresentation
        from ni_rua.runtime.tools.handlers import geometry_query
        runtime = SimpleNamespace(scene=self.scene, identity=IDENTITY, image_settings={})
        response = geometry_query(runtime, dict(request=dict(kind="pixel_ray", camera_id="head",
            observation_id="obs-0", pixel=[320, 240])), {})
        original = deepcopy(response.data)
        self.assertEqual(original["status"], "supported")
        self.assertIn("library_versions", original)
        actor = ActorOutput(self.artifacts, ActorPresentation(self.artifacts))
        shown = json.loads(actor.present(response)["contentItems"][0]["text"])
        self.assertTrue({"library_versions", "algorithm", "implementation_digest", "duration_s",
                         "schema_version", "run_id", "episode_id", "env_instance_id", "env_generation"}.isdisjoint(shown))
        for key in ("value", "status", "camera_pose_source", "camera_pose_diagnostics", "observation_states"):
            self.assertEqual(shown[key], original[key])
        self.assertEqual(response.data, original)

    def test_geometry_error_echo_excludes_injected_identity_before_bounding(self):
        result = self.geometry.execute({**self.request, "kind": "pixel_ray", "pixel": [-1, 2]})
        actual = result["diagnostics"]["input_guidance"]["actual"]
        injected = {"schema_version", "run_id", "episode_id", "env_instance_id", "env_generation"}
        self.assertTrue(injected.isdisjoint(actual))
        self.assertTrue(injected.isdisjoint(actual.get("keys", [])))
        self.assertEqual(result["reason"], "pixel_out_of_bounds")
        self.assertEqual(result["run_id"], IDENTITY["run_id"])

    def test_geometry_display_retains_executable_scene_patch_identity_and_revision(self):
        from ni_rua.core.presentation.actor_output import project_metadata
        result = self.geometry.execute({**self.request, "kind": "build_scene", "base_scene_version": 0,
                                        "roi": [100, 80, 400, 320]})
        shown = project_metadata(result, "geometry_result")
        self.assertNotIn("library_versions", shown)
        self.assertEqual(shown["value"]["proposal"], result["value"]["proposal"])
        scene = self.scene.apply_patch(shown["value"]["proposal"])
        self.assertEqual(scene["scene_version"], 1)

    def test_public_tracking_keeps_numeric_roi_accepted_by_direct_geometry_owner(self):
        from ni_rua.runtime.tools.builtins import create_registry
        request = dict(kind="sparse_track", observation_ids=["obs-0", "obs-1"], camera_id="head",
                       static_scene_assumption=True, roi=[100.5, 80.5, 400., 320.])
        registry = create_registry()
        for role in ("actor", "program"):
            registry.validate("geometry_query", dict(request=request), role=role)
        result = self.geometry.execute({**IDENTITY, **request})
        self.assertEqual(result["status"], "supported")
        self.assertGreater(result["value"]["accepted"], 0)

    def test_invalid_geometry_result_keeps_reason_and_supplies_help(self):
        result = self.geometry.execute({**self.request, 'kind': 'pixel_ray', 'pixel': [-1, 2]})
        self.assertEqual(result['status'], 'invalid_input')
        self.assertEqual(result['reason'], 'pixel_out_of_bounds')
        guidance = result['diagnostics']['input_guidance']
        self.assertEqual(guidance['minimal_example'], {'request': {'kind': 'pixel_ray'}})
        self.assertIn('required fields', guidance['suggestion'])
        self.assertIsNone(result['value'])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.artifacts = Store(self.tmp.name,IDENTITY); self.addCleanup(self.artifacts.close)
        self.scene = SceneStore(Path(self.tmp.name)/"scene",self.artifacts.resolve_artifact,self.artifacts.put_artifact)
        self.obs = recorded_plane(self.artifacts)
        for obs in self.obs: self.scene.import_observation(obs)
        self.geometry = GeometryTools(self.scene)
        self.request = {**IDENTITY,"camera_id":"head","observation_ids":["obs-0","obs-1"],"static_scene_assumption":True}

    def test_rgb_build_heldout_alignment_and_snapshot(self):
        result = self.geometry.execute({**self.request,"kind":"build_scene","base_scene_version":0,"roi":[100,80,400,320]})
        self.assertEqual(result["status"],"conditional",result)
        value = result["value"]
        self.assertGreater(value["geometry_result"]["accepted"],100)
        scene = self.scene.apply_patch(value["proposal"])
        entity = next(iter(scene["entities"].values()))
        self.assertLess(abs(entity["geometry"]["origin_m"][2]-2),0.005)
        self.assertLess(entity["uncertainty"]["fit_rms_m"],0.005)
        projected=self.geometry.execute({**self.request,"kind":"project_geometry","scene_version":1})
        self.assertEqual(projected["status"],"supported",projected)
        self.assertEqual(len(projected["value"]["pixels"]),4)
        hit=self.geometry.execute({**self.request,"kind":"ray_surface","scene_version":1,"entity_id":next(iter(scene["entities"])),"pixel":[320,240]})
        self.assertEqual(hit["status"],"conditional",hit)
        self.assertAlmostEqual(hit["value"]["point_m"][2],2,places=2)
        points = [[0,0,2],[.2,.1,2],[-.2,-.1,2]]
        result = self.geometry.execute({**self.request,"kind":"alignment","observation_ids":["obs-2"],"scene_version":1,"points_m":points,"pixels":[[307.5,240],[357.5,265],[257.5,215]]})
        self.assertEqual(result["value"]["rms_px"],0)
        rejected = self.geometry.execute({**self.request,"kind":"alignment","observation_ids":["obs-0"],"scene_version":1,"points_m":points,"pixels":[[0,0]]*3})
        self.assertIn("heldout",rejected["reason"])
        before = self.scene.snapshot(IDENTITY,1,"obs-1")
        self.assertEqual(before["quality"]["artifact_resolution"],"verified_digest_closed")
        self.assertGreaterEqual(len(before["resolved_artifacts"]),6)
        self.assertEqual(self.scene.read_snapshot(IDENTITY,before["snapshot_id"]),before)
        patch = deepcopy(value["proposal"]); patch.update(base_scene_version=1,observation_ids=["obs-2"])
        patch["operations"] = [{"op":"mark_lost","entity_id":next(iter(scene["entities"])),"source_kind":"observation_inference","reason":"occluded"}]
        self.scene.apply_patch(patch)
        self.assertEqual(digest(before),digest(self.scene.snapshot(IDENTITY,1,"obs-1")))
        patch["base_scene_version"] = 2; patch["observation_ids"] = ["obs-0"]
        with self.assertRaisesRegex(ValueError,"late_observation"):
            self.scene.apply_patch(patch)

    def test_rays_projection_and_degenerate_matrix(self):
        help_result = self.geometry.execute({**IDENTITY, "kind": "help"})
        self.assertIn("pixels_a", help_result["requests"]["triangulate"])
        self.assertIn("pixel", help_result["requests"]["pixel_ray"])
        ray_result = self.geometry.execute({**self.request,"kind":"pixel_ray","pixel":[320,240]})
        self.assertEqual(ray_result["value"]["direction"],[0.,0.,1.])
        camera = self.geometry.camera(IDENTITY,"obs-0","head")
        with self.assertRaisesRegex(ValueError,"pure_rotation"):
            triangulate(camera,camera,[[320,240]],[[310,240]])
        second = self.geometry.camera(IDENTITY,"obs-1","head")
        low = triangulate(camera,second,[[320,240]],[[319.9,240]])
        self.assertEqual(low["point_status"],["insufficient_parallax"])
        wrong = triangulate(camera,second,[[320,240]],[[295,270]])
        self.assertEqual(wrong["point_status"],["inconsistent_correspondence"])
        behind = triangulate(camera,second,[[320,240]],[[345,240]])
        self.assertEqual(behind["point_status"],["negative_depth"])
        unsync = deepcopy(self.obs[0]); unsync["observation_id"]="unsync"; unsync["cameras"][0]["sim_step"]=999
        self.scene.import_observation(unsync)
        result = self.geometry.execute({**self.request,"kind":"pixel_ray","pixel":[1,1],"observation_ids":["unsync"]})
        self.assertEqual(result["reason"],"unsynchronized_observation")
        missing = deepcopy(self.obs[0]); missing["observation_id"]="uncalibrated"; missing["cameras"][0]["calibration_ref"]=None; missing["observation_profile"]="rgb_proprio_v1"; missing["calibration_id"]=None
        self.scene.import_observation(missing)
        result = self.geometry.execute({**self.request,"kind":"pixel_ray","pixel":[1,1],"observation_ids":["uncalibrated"]})
        self.assertEqual(result["reason"],"missing_calibration")
        overlay = self.geometry.execute({**self.request,"kind":"project_points","points_m":[[0,0,2]]})
        self.assertEqual(overlay["value"]["pixels"],[[320.,240.]])
        self.artifacts.resolve_artifact(overlay["value"]["overlay_ref"])

    def test_projection_keeps_robot_measurements_at_the_selected_image_time(self):
        observation = deepcopy(self.obs[0])
        observation['observation_id'] = 'measured-frame'
        observation['observation_profile'] = 'rgb_proprio_calibrated_v1'
        state = dict(sim_step=0, step_domain='control_step', arms=dict(left=dict(
            source_kind='measurement', joint_names=['joint1'], q_rad=[.125], qdot_rad_s=[.25],
            ee_pose=dict(position_m=[.1, .2, .3], quaternion_xyzw=[0., 0., 0., 1.],
                         source_kind='measurement', frame_id='env_origin', tcp_config_id='link6'),
            gripper=dict(position_m=[.02, .02], position_source_kind='measurement',
                         joint_names=['jaw1', 'jaw2'], goal_normalized=1., goal_source_kind='command'))))
        observation['robot_state_ref'] = self.artifacts.put_artifact(canonical(state).encode(), 'robot_state', 'application/json')
        calibration = dict(cameras=dict(head=dict(K=[[500., 0, 320.], [0, 500., 240.], [0, 0, 1.]],
            distortion=[], image_size=[640, 480], T_world_camera=np.eye(4).tolist())))
        observation['cameras'][0]['calibration_ref'] = self.artifacts.put_artifact(canonical(calibration).encode(), 'calibration', 'application/json')
        self.scene.import_observation(observation)
        result = self.geometry.execute(dict(IDENTITY, kind='project_points', observation_id='measured-frame',
                                            camera_id='head', points_m=[[0, 0, 2]]))
        self.assertEqual(result['status'], 'supported', result)
        evidence = result['observation_states'][0]
        self.assertEqual(evidence['observation_id'], 'measured-frame')
        self.assertEqual(evidence['robot_state'], state)
        self.assertEqual(evidence['sim_step'], 0)

    def test_dynamic_updates_dirty_occlusion_and_model_rebuild(self):
        initial=self.geometry.execute({**self.request,"kind":"build_scene","entity_id":"surface","base_scene_version":0})
        self.scene.apply_patch(initial["value"]["proposal"])
        old=self.scene.version(IDENTITY,1)["entities"]["surface"]["geometry"]["origin_m"]
        patch={"schema_version":"0.1",**IDENTITY,"world_kind":"internal","base_scene_version":1,"observation_ids":["obs-2"],"builder_digest":digest("move"),"operations":[{"op":"update_pose","entity_id":"surface","translation_m":[0.1,0,0],"source_kind":"explicit_prior","assumptions":["Synthetic explicit rigid translation fixture."]}]}
        moved=self.scene.apply_patch(patch)
        self.assertAlmostEqual(moved["entities"]["surface"]["geometry"]["origin_m"][0],old[0]+0.1)
        self.assertAlmostEqual(self.scene.version(IDENTITY,1)["entities"]["surface"]["geometry"]["origin_m"][0],old[0])
        self.assertTrue(any("previous_entity_geometry" in item.get("region",{}) for item in moved["unknown_regions"] if isinstance(item.get("region"),dict)))
        for index,operation in enumerate(("mark_dirty","mark_occluded","mark_lost"),start=2):
            patch["base_scene_version"]=index
            patch["operations"]=[{"op":operation,"entity_id":"surface","source_kind":"observation_inference","reason":"synthetic visibility/state fixture"}]
            self.scene.apply_patch(patch)
            result=self.scene.query(IDENTITY,index+1,{"kind":"entity_pose","entity_id":"surface"})
            self.assertEqual(result["status"],"unknown")
        obs=deepcopy(self.obs[2]); obs.update(observation_id="new-calibration",calibration_id="calibration-v2")
        self.scene.import_observation(obs)
        patch.update(base_scene_version=5,observation_ids=["new-calibration"])
        with self.assertRaisesRegex(ValueError,"rebuild required"):
            self.scene.apply_patch(patch)
        patch["rebuild"]=True; patch["operations"]=[{"op":"mark_unknown","source_kind":"observation_inference","region":"all","reason":"calibration changed"}]
        rebuilt=self.scene.apply_patch(patch)
        self.assertEqual(rebuilt["entities"],{})
        self.assertEqual(rebuilt["calibration_id"],"calibration-v2")

    def test_fk_is_authoritative_when_usd_camera_pose_is_stale(self):
        model={"base_link":"base","joints":[{"name":"slide","parent":"base","child":"camera_link","type":"prismatic","axis":[1,0,0],"T_parent_joint":np.eye(4).tolist()}]}
        calibration={"cameras":{"head":{"K":[[500,0,320],[0,500,240],[0,0,1]],"distortion":[],"image_size":[640,480],"link_name":"camera_link","T_link_camera":np.eye(4).tolist(),"T_world_base":np.eye(4).tolist()}}}
        moved=np.eye(4); moved[0,3]=0.1
        sample={"T_world_camera":np.eye(4).tolist(),"T_world_link":moved.tolist(),"T_world_base":np.eye(4).tolist(),"source_kind":"simulator_camera_transform","sim_step":0,"step_domain":"control_step","frame_id":"frame-0"}
        state={"source_kind":"measurement","joint_names":["slide"],"joint_positions_rad":[0.1],"sim_step":0,"step_domain":"control_step","camera_poses":{"head":sample}}
        observation=deepcopy(self.obs[0]); observation.update(observation_id="stale-usd",calibration_id="known-slide")
        for field,value,kind in (("robot_model_ref",model,"robot_model"),("robot_state_ref",state,"robot_state")):
            observation[field]=self.artifacts.put_artifact(canonical(value).encode(),kind,"application/json")
        observation["cameras"][0]["calibration_ref"]=self.artifacts.put_artifact(canonical(calibration).encode(),"calibration","application/json")
        self.scene.import_observation(observation)
        result=self.geometry.execute({**self.request,"kind":"pixel_ray","observation_ids":["stale-usd"],"pixel":[320,240]})
        self.assertEqual(result["value"]["origin_m"],[0.1,0,0])
        self.assertEqual(result["camera_pose_source"],"robot_kinematics")
        self.assertAlmostEqual(result["camera_pose_diagnostics"]["reported_camera_translation_residual_m"],0.1)
        self.assertEqual(result["camera_pose_diagnostics"]["fk_vs_measured_link_translation_m"],0)

    def test_seeded_rgb_association_rejects_implausible_motion(self):
        first=self.geometry.execute({**self.request,"kind":"build_scene","entity_id":"seeded-surface","base_scene_version":0})
        self.scene.apply_patch(first["value"]["proposal"])
        updated=self.geometry.execute({**self.request,"kind":"build_scene","entity_id":"seeded-surface","base_scene_version":1,"observation_ids":["obs-0","obs-2"]})
        operation=updated["value"]["proposal"]["operations"][0]
        self.assertEqual(operation["entity_id"],"seeded-surface")
        self.assertEqual(operation["op"],"upsert_geometry")
        self.assertEqual(operation["association"]["kind"],"appearance_and_geometry_gate")
        impossible=self.geometry.execute({**self.request,"kind":"build_scene","entity_id":"seeded-surface","base_scene_version":1,"observation_ids":["obs-0","obs-2"],"max_displacement_m":0.000001})
        self.assertEqual(impossible["value"]["proposal"]["operations"][0]["op"],"mark_lost")
        marked=self.scene.apply_patch(impossible["value"]["proposal"])
        self.assertEqual(marked["entities"]["seeded-surface"]["state"],"lost")

    def test_privileged_json_cannot_hide_in_actor_artifact(self):
        observation=deepcopy(self.obs[0]); observation["observation_id"]="forbidden-payload"
        observation["robot_state_ref"]=self.artifacts.put_artifact(canonical({"object_poses":[{"hidden":True}]}).encode(),"robot_state","application/json")
        with self.assertRaisesRegex(ValueError,"forbidden_observation_field"):
            self.scene.import_observation(observation)
        observation=deepcopy(self.obs[0]); observation["observation_id"]="internal-rgb"
        observation["cameras"][0]["rgb_ref"]=self.artifacts.put_artifact(b"internal","geometry_overlay","image/png","internal")
        with self.assertRaisesRegex(ValueError,"permitted_actor"):
            self.scene.import_observation(observation)

    def test_digest_fk_and_unconstrained_box(self):
        metadata,path = self.artifacts.artifact(self.obs[0]["cameras"][0]["rgb_ref"])
        path.write_bytes(b"corrupt")
        with self.assertRaises(ValueError): self.scene.closure(self.obs[0])
        model = {"base_link":"base","joints":[{"name":"j","parent":"base","child":"tip","type":"revolute","axis":[0,0,1],"T_parent_joint":np.eye(4).tolist()}]}
        computed = fk(model,{"joint_names":["j"],"joint_positions_rad":[np.pi/2],"source_kind":"measurement"})
        np.testing.assert_allclose(computed["tip"][:3,:3]@[1,0,0],[0,1,0],atol=1e-12)
        with self.assertRaisesRegex(ValueError,"thickness_unobserved"):
            fit_geometry([[i%3,i//3,1] for i in range(6)],"box")



# --- Folded from tests/test_grasp_contact.py ---

def values(position=(.2, -.1, .9), angles=(0., 0., 0.)):
    return dict(zip((*roboprobe_pose.POSITION_AXES, *roboprobe_pose.ANGLE_AXES),
                    (*position, *angles)))


def pose_args(position=(.2, -.1, .9), angles=(0., 0., 0.), **selection):
    return dict(grasp_position_m=list(position), angles_deg=list(angles), **selection)


def xyzw_pose(pose):
    return measured_pose(dict(position_m=pose[:3].tolist(), quaternion_xyzw=[*pose[4:].tolist(), float(pose[3])]))


def measured_pose(pose):
    return dict(deepcopy(pose), source_kind="measurement", tcp_config_id="arx-x5-link6-v1", frame_id="env_origin")


class ContactGeometryTests(unittest.TestCase):
    def test_default_is_exact_legacy_conversion_even_when_rotated(self):
        for angles in ((0, 0, 0), (0, 90, 0), (48, -34, 129), (-180, 0, 180)):
            with self.subTest(angles=angles):
                target = values(angles=angles)
                original = deepcopy(target)
                spec = resolve_contact_spec({"unrelated_action_option": 7})
                self.assertEqual(spec, dict(contact_region="pad_center", contact_depth_m=0., offset_m=.1501))
                np.testing.assert_array_equal(values_to_contact_pose(target, spec),
                                              roboprobe_pose.values_to_pose(target))
                self.assertEqual(target, original)

    def test_contact_reference_reaches_requested_point_in_arbitrary_orientation(self):
        target = np.array([.2, -.1, .9])
        for angles in ((0, 0, 0), (0, 90, 0), (48, -34, 129)):
            for selection in ({"contact_region": "finger_middle"}, {"contact_region": "fingertip"},
                              {"contact_depth_m": .04}):
                with self.subTest(angles=angles, selection=selection):
                    spec = resolve_contact_spec(selection)
                    converted = values_to_contact_pose(values(target, angles), spec)
                    axis = roboprobe_pose.tool_axis(converted[3:])
                    physical_contact = converted[:3] + spec["offset_m"] * axis
                    np.testing.assert_allclose(physical_contact, target, atol=1e-12)
                    legacy_center = np.array(list(roboprobe_pose.pose_to_values(converted).values())[:3])
                    np.testing.assert_allclose(legacy_center, target + spec["contact_depth_m"] * axis,
                                               atol=1e-12)

    def test_middle_insertion_moves_flange_down_for_top_down_and_forward_for_side_grasp(self):
        middle = resolve_contact_spec({"contact_region": "finger_middle"})
        self.assertEqual(middle["contact_depth_m"], .0275)
        self.assertAlmostEqual(middle["offset_m"], .1226)
        np.testing.assert_allclose(values_to_contact_pose(values(), middle)[:3], [.2, -.1, 1.0226], atol=1e-12)
        np.testing.assert_allclose(values_to_contact_pose(values(angles=(0, 90, 0)), middle)[:3],
                                   [.3226, -.1, .9], atol=1e-12)
        tip = resolve_contact_spec({"contact_region": "fingertip"})
        self.assertEqual(tip["contact_depth_m"], -.0075)
        np.testing.assert_allclose(values_to_contact_pose(values(), tip)[:3], [.2, -.1, 1.0576], atol=1e-12)

    def test_numeric_selection_and_inherited_selection_are_independent_copies(self):
        parent = resolve_contact_spec({"contact_region": "finger_middle"})
        original = deepcopy(parent)
        inherited = resolve_contact_spec({"grasp_position_m": [0, 0, 1]}, inherited=parent)
        self.assertEqual(inherited, parent)
        inherited["contact_depth_m"] = 0
        self.assertEqual(parent, original)
        custom = resolve_contact_spec({"contact_depth_m": .01}, inherited=parent)
        self.assertEqual(custom["contact_region"], "custom")
        self.assertAlmostEqual(custom["offset_m"], .1401)
        self.assertEqual(resolve_contact_spec({"contact_region": "pad_center"}, inherited=parent),
                         resolve_contact_spec({}))

    def test_invalid_selection_cannot_be_silently_ignored(self):
        invalid = [{"contact_region": value} for value in (None, "middle", "", True, [], {})]
        invalid += [{"contact_depth_m": value} for value in
                    (None, True, False, "0.02", float("nan"), float("inf"), -.0075001, .0625001)]
        invalid += [{"contact_region": "pad_center", "contact_depth_m": 0.}]
        for selection in invalid:
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                resolve_contact_spec(selection)
        for boundary in (-.0075, .0625):
            self.assertEqual(resolve_contact_spec({"contact_depth_m": boundary})["contact_depth_m"], boundary)


class ContactHarnessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = Store(Path(temporary.name) / "store", dict(run_id="contact", episode_id="ep",
            env_instance_id="env", env_generation=1))
        self.addCleanup(self.store.close)
        self.robot = FixtureRobot(temporary.name)
        caps = dict(self.robot.start(), max_waypoints=8, dual_arm=True)
        self.robot.start = lambda: caps
        self.harness = Harness(self.store, self.robot, motion_review=False)
        self.harness.start()

    def single(self, **updates):
        return dict(observation_id=self.harness.latest["observation_id"], arm="left", action_id="contact",
                    **pose_args(), **updates)

    def build(self, args):
        return compile_action("grasp", args, self.harness.identity, self.harness.capabilities)

    def test_omitted_selection_preserves_legacy_action_metadata_and_coordinates(self):
        action = self.build(self.single(via_grasp_poses=[pose_args(position=(.1, -.1, 1.))]))
        meta = action["input_grasp_pose"]
        self.assertEqual(set(meta), {"position_m", "angles_deg", "convention", "offset_m", "via_grasp_poses"})
        self.assertEqual(meta["offset_m"], .1501)
        np.testing.assert_allclose(action["path"]["waypoints"][0]["left"]["position_m"],
                                   [.1, -.1, 1.1501], atol=1e-12)
        np.testing.assert_allclose(action["path"]["waypoints"][1]["left"]["position_m"],
                                   [.2, -.1, 1.0501], atol=1e-12)

    def test_request_contact_inherits_to_vias_and_explicit_via_override_wins(self):
        args = self.single(contact_region="finger_middle", via_grasp_poses=[
            pose_args(position=(.1, -.1, 1.)),
            pose_args(position=(.15, -.1, .95), angles=(0, 90, 0), contact_region="pad_center")])
        original = deepcopy(args)
        action = self.build(args)
        np.testing.assert_allclose([p["left"]["position_m"] for p in action["path"]["waypoints"]],
                                   [[.1, -.1, 1.1226], [.3001, -.1, .95], [.2, -.1, 1.0226]], atol=1e-12)
        meta = action["input_grasp_pose"]
        self.assertEqual([s["contact_region"] for s in meta["waypoint_contact_frames"]],
                         ["finger_middle", "pad_center", "finger_middle"])
        self.assertEqual(args, original)

    def test_via_only_contact_override_does_not_change_default_final_reference(self):
        action = self.build(self.single(via_grasp_poses=[pose_args(contact_depth_m=.04)]))
        meta = action["input_grasp_pose"]
        self.assertEqual(meta["contact_region"], "pad_center")
        self.assertEqual(meta["contact_depth_m"], 0.)
        np.testing.assert_allclose([p["left"]["position_m"] for p in action["path"]["waypoints"]],
                                   [[.2, -.1, 1.0101], [.2, -.1, 1.0501]], atol=1e-12)

    def test_invalid_final_or_via_selection_never_compiles(self):
        for selector in ({"contact_depth_m": True}, {"contact_region": "middle"},
                         {"contact_depth_m": .063}, {"contact_region": "pad_center", "contact_depth_m": 0.}):
            for args in (self.single(**selector), self.single(via_grasp_poses=[pose_args(**selector)])):
                with self.subTest(args=args, selector=selector), self.assertRaises(ValueError):
                    self.build(args)
        self.assertEqual(self.robot.executions, 0)
        self.assertEqual(self.harness.action_count, 0)

    def test_failed_intermediate_reports_its_contact_not_final_contact(self):
        args = self.single(contact_region="finger_middle", via_grasp_poses=[
            pose_args(position=(.1, -.1, 1.), contact_depth_m=.01)])
        action = self.build(args)
        feedback = failure_feedback(action, "left", 0, [], "configured_robot_model")
        self.assertEqual(feedback["requested_grasp_pose"], args["via_grasp_poses"][0])
        self.assertAlmostEqual(feedback["grasp_to_ee_offset_m"], .1401)
        self.assertEqual(feedback["requested_ee_pose"], action["path"]["waypoints"][0]["left"])
        final = failure_feedback(action, "left", 1, [], "configured_robot_model")
        self.assertAlmostEqual(final["grasp_to_ee_offset_m"], .1226)
        self.assertEqual(final["requested_grasp_pose"]["contact_region"], "finger_middle")

    def test_feedback_uses_actual_orientation_and_selected_reference_from_measured_ee(self):
        action = self.build(self.single(contact_region="finger_middle"))
        requested = np.array([.2, -.1, .9])
        actual = requested + np.array([.01, 0, .03])
        measured = values_to_contact_pose(values(actual, (15, 0, 0)),
                                           resolve_contact_spec({"contact_region": "finger_middle"}))
        state = dict(arms=dict(left=dict(ee_pose=xyzw_pose(measured),
            grasp_pose=dict(position_m=[888, 888, 888], angles_deg=[0, 0, 0]))))
        original = deepcopy(state)
        feedback = contact_target_feedback(action, state)
        self.assertEqual(feedback["target_scope"], "requested_final_pose_not_completion_claim")
        arm = feedback["arms"]["left"]
        self.assertTrue(arm["available"])
        self.assertEqual(arm["contact_region"], "finger_middle")
        np.testing.assert_allclose(arm["requested_contact_position_m"], requested, atol=1e-12)
        np.testing.assert_allclose(arm["actual_contact_position_m"], actual, atol=1e-12)
        np.testing.assert_allclose(arm["position_error_vector_m"], [.01, 0, .03], atol=1e-12)
        self.assertAlmostEqual(arm["position_error_m"], np.linalg.norm([.01, 0, .03]))
        self.assertAlmostEqual(arm["orientation_error_rad"], np.deg2rad(15))
        self.assertAlmostEqual(arm["approach_shortfall_m"], .03)
        self.assertEqual(state, original)

    def test_feedback_shortfall_sign_distinguishes_stop_short_from_overshoot(self):
        action = self.build(self.single(contact_depth_m=.04))
        spec = resolve_contact_spec({"contact_depth_m": .04})
        for actual_z, expected_shortfall in ((.905, .005), (.895, -.005)):
            with self.subTest(actual_z=actual_z):
                pose = values_to_contact_pose(values(position=(.2, -.1, actual_z)), spec)
                state = dict(arms=dict(left=dict(ee_pose=xyzw_pose(pose))))
                arm = contact_target_feedback(action, state)["arms"]["left"]
                self.assertAlmostEqual(arm["approach_shortfall_m"], expected_shortfall)

    def test_partial_failure_feedback_still_labels_comparison_to_unreached_final_target(self):
        args = self.single(contact_region="finger_middle", via_grasp_poses=[
            pose_args(position=(.1, -.1, 1.), contact_region="pad_center")])
        action = self.build(args)
        state = dict(arms=dict(left=dict(ee_pose=measured_pose(action["path"]["waypoints"][0]["left"]))))
        feedback = contact_target_feedback(action, state)
        self.assertEqual(feedback["target_scope"], "requested_final_pose_not_completion_claim")
        arm = feedback["arms"]["left"]
        np.testing.assert_allclose(arm["requested_contact_position_m"], args["grasp_position_m"], atol=1e-12)
        # The observed flange is at the via endpoint; evaluating the FINAL chosen
        # reference there differs from both the via's pad center and final goal.
        np.testing.assert_allclose(arm["actual_contact_position_m"], [.1, -.1, 1.0275], atol=1e-12)
        self.assertGreater(arm["position_error_m"], .1)

    def test_predicted_wrong_tcp_or_wrong_frame_pose_cannot_be_reported_as_measured_contact(self):
        action = self.build(self.single(contact_region="finger_middle"))
        pose = measured_pose(action["path"]["waypoints"][-1]["left"])
        for change in (dict(source_kind="planner_prediction"), dict(tcp_config_id="another_tool"),
                       dict(frame_id="base_link")):
            with self.subTest(change=change):
                state = dict(arms=dict(left=dict(ee_pose=dict(pose, **change))))
                result = contact_target_feedback(action, state)["arms"]["left"]
                self.assertFalse(result["available"])
                self.assertNotIn("actual_contact_position_m", result)

    def test_failed_action_archives_selected_contact_feedback_without_changing_outcome(self):
        self.robot.outcome = "failed"
        original_observe = self.robot.observe

        def observe():
            observation = original_observe()
            observation["robot_state"]["arms"]["left"]["ee_pose"] = measured_pose(
                observation["robot_state"]["arms"]["left"]["ee_pose"])
            return observation

        self.robot.observe = observe
        receipt = self.harness.execute(self.build(self.single(contact_region="finger_middle")),
                                       {"tool_call_id": "failed-contact"})["receipt"]
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["reason"], "synthetic_fixture")
        feedback = receipt["grasp_contact_feedback"]
        self.assertTrue(feedback["arms"]["left"]["available"])
        self.assertEqual(feedback["target_scope"], "requested_final_pose_not_completion_claim")
        self.assertEqual(self.store.read("action_receipt", "contact")["grasp_contact_feedback"], feedback)
        self.assertEqual(self.robot.executions, 1)

    def test_diagnostic_exception_does_not_turn_executed_action_into_unknown_outcome(self):
        with patch("ni_rua.runtime.harness.contact_target_feedback", side_effect=ValueError("diagnostic fixture")):
            receipt = self.harness.execute(self.build(self.single(contact_region="finger_middle")),
                                           {"tool_call_id": "diagnostic-failure"})["receipt"]
        self.assertEqual(receipt["status"], "completed")
        self.assertFalse(receipt["grasp_contact_feedback"]["available"])
        self.assertEqual(receipt["grasp_contact_feedback"]["reason"], "diagnostic_unavailable")
        self.assertTrue(self.harness.accepting)
        self.assertEqual(self.robot.executions, 1)



if __name__ == "__main__":
    unittest.main()
