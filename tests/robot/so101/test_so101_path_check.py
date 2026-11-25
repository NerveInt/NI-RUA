"""SO-101 path between targets, collision world, arrival-correction bounds and receipt residuals (CPU only).

- ``ik_plan``: the shared joint interpolation, FK path facts, the held-box sphere fill, the collision world and the
  obstacle-aware candidate choice of ``curobo_solve``.
- ``CuroboClient.check_path`` and the world fields of ``solve`` against a fake opener (no network).
- ``services/curobo_ik/curobo_ik_service.py``: the pure request helpers of ``/path_check`` (no CUDA, no cuRobo).
- The worker's preview with a recording fake client: path facts per move, overlay samples of the interpolated path,
  the ``path_check`` review (clear / collisions / unavailable), the collision world of ``review_proxies`` and when the
  held box stops being attached; execution reports ``path_start_deviation_rad``.
- Per-stage ``arrival_correction_rounds`` / ``arrival_correction_max_offset_rad``, receipt-level
  ``endpoint_residual_m`` / ``in_hand_tracking_error``, the ``execute_if_clean`` downgrade, schema gating by profile
  tag and the condensed history projection.

These tests prove the software contract. The cuRobo collision verdict itself is checked live against the GPU service
(``docs/development/so101-integration.md`` section 12).
Run: ``PYTHONPATH=.:tests mamba run -n ni-rua python -m unittest test_so101_path_check -v``.
"""

NI_RUA_TEST_TIER = "component"

from copy import deepcopy
import dataclasses
import importlib.util
import io
import json
import math
from pathlib import Path
from support.paths import REPO_ROOT
import tempfile
import threading
import unittest
import urllib.error
from unittest.mock import PropertyMock, patch

import numpy as np

from ni_rua.core.config import Config
from ni_rua.core.presentation.actor_payload import compact_action_result, lean_action_result
from ni_rua.robot.backends.so101 import contract, ik_plan
from ni_rua.robot.backends.so101.deployment import embodiment_profile, validate_deployment
from ni_rua.robot.backends.so101.driver import FakeArmDriver, FakeCameraSource, FakeClock
from ni_rua.robot.backends.so101.ik_service import ARM_JOINTS, CuroboClient, IKServiceError
from ni_rua.robot.backends.so101.worker import PATH_OVERLAY_SAMPLES, SO101Adapter
from ni_rua.robot.model import embodiments
from ni_rua.robot.model import so101_kinematics as kin
from ni_rua.robot.motion import motion_tolerance as mt
from ni_rua.runtime.actions.robot_action_schema import build
from robot.so101.test_so101_backend import EXAMPLE, IDENTITY, InProcessBackend
from robot.so101.test_so101_motion_features import TableArmDriver, batch

REPOSITORY = REPO_ROOT
MODEL = kin.load_model()
SERVICE = dict(solver="curobo", url="http://127.0.0.1:9")
EMPTY_CLOSE_M = 0.0035
BOX = dict(id="o1", min_m=[0.30, -0.02, 0.0], max_m=[0.34, 0.02, 0.05], margin_m=0.01, evidence="test box")
HELD = dict(min_m=[0.19, -0.02, 0.10], max_m=[0.23, 0.02, 0.14], evidence="held cube")


def deployment(ik_service=True, **updates):
    document = json.loads(EXAMPLE.read_text())
    document["gripper"]["empty_close_aperture_m"] = EMPTY_CLOSE_M
    for dotted, value in updates.items():
        section, key = dotted.split(".")
        document[section][key] = value
    if ik_service:
        document["ik_service"] = dict(SERVICE)
    deploy = validate_deployment(document, base=EXAMPLE.parent)
    for name in ("fixed", "wrist"):
        deploy["cameras"][name].update(width=64, height=48)
    return deploy


def service_module():
    """``services/curobo_ik/curobo_ik_service.py`` imported by path (standard library only at import time)."""
    spec = importlib.util.spec_from_file_location("curobo_ik_service_under_test",
                                                  REPOSITORY / "services" / "curobo_ik" / "curobo_ik_service.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeService:
    """Stands in for ``CuroboClient`` inside the adapter: ``solve`` answers with the packaged local IK solution of each
    pose (``obstacle_free`` from ``labels`` when a world is sent) and ``check_path`` with ``collisions``
    (``{segment index: first_collision}``); every call is recorded."""

    path_check_step_rad = 0.01

    def __init__(self, adapter):
        self.adapter, self.solves, self.checks = adapter, [], []
        self.collisions, self.error, self.labels = {}, None, None

    def solve(self, poses, seed, weights, position_tolerance, orientation_tolerance, world=None):
        self.solves.append(dict(poses=deepcopy(poses), world=deepcopy(world)))
        out = []
        for pose in poses:
            q, _ = ik_plan.local_solve(kin, MODEL, pose, seed, (0.005, 0.1), self.adapter.deployment["ik"], self.adapter.limits)
            candidate = dict(q_rad=[float(v) for v in q], success=True, feasible=True)
            if world is not None and self.labels is not None:
                candidate["obstacle_free"] = self.labels
            out.append([candidate])
        return out

    def check_path(self, segments, world=None):
        self.checks.append(dict(segments=deepcopy(segments), world=deepcopy(world)))
        if self.error is not None:
            raise self.error
        return dict(check_ms=1.0, samples=10 * len(segments), segments=[
            dict(samples=10, colliding_samples=3 if index in self.collisions else 0,
                 first_collision=self.collisions.get(index)) for index in range(len(segments))])


class Case(unittest.TestCase):
    def adapter(self, *, ik_service=True, table_z_m=None, deploy=None, **driver_options):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        deploy = deploy or deployment(ik_service)
        clock = FakeClock()
        driver = TableArmDriver(deploy, clock, table_z_m, **driver_options)
        cameras = {name: FakeCameraSource(name, clock=clock, width=64, height=48) for name in ("fixed", "wrist")}
        adapter = SO101Adapter(deploy, directory.name, "rgb_proprio_v1", driver=driver, cameras=cameras, clock=clock,
                               image_settings=Config().images.to_dict(), start_telemetry=False)
        self.addCleanup(adapter.close)
        service = None
        if adapter.ik_client is not None:
            service = adapter.ik_client = FakeService(adapter)
        return adapter, driver, service

    @staticmethod
    def pose(adapter):
        return adapter.robot_state(adapter.read())["arms"]["right"]["ee_pose"]

    def move(self, adapter, identifier, dx=0.0, dz=0.0, *, base=None, **path):
        pose = base or self.pose(adapter)
        leaf = contract.compile_action("move", dict(
            arm="right", observation_id="o", quaternion_xyzw=pose["quaternion_xyzw"],
            position_m=[pose["position_m"][0] + dx, pose["position_m"][1], pose["position_m"][2] + dz]),
            IDENTITY, adapter.capabilities())
        leaf["path"].update(position_tolerance_m=0.005, orientation_tolerance_rad=0.1, **path)
        return dict(id=identifier, kind="move", arm="right", action=leaf)

    @staticmethod
    def jaw(adapter, identifier, goal):
        leaf = contract.compile_action("gripper", dict(arm="right", goal_normalized=goal, observation_id="o"), IDENTITY,
                                       adapter.capabilities())
        return dict(id=identifier, kind="gripper", arm="right", action=leaf)

    def run_batch(self, adapter, action):
        preview = adapter.preview_batch(action)
        self.assertEqual(preview["status"], "planned", preview)
        receipt = adapter.execute(dict(action, batch_plan_id=preview["plan_id"]), lambda update: None, threading.Event())
        return preview, receipt

    def harness(self, **options):
        from ni_rua.core.storage.store import Store
        from ni_rua.runtime.harness import Harness
        adapter, driver, service = self.adapter(**options)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = Store(Path(directory.name) / "store", IDENTITY)
        self.addCleanup(store.close)
        config = Config.from_dict({"backend": {"environment": "so101_real", "robot_model_id": "so101",
                                               "deployment": str(EXAMPLE)},
                                   "features": {"observation_profile": "rgb_proprio_v1", "record_video": False}})
        harness = Harness(store, InProcessBackend(adapter), observation_profile="rgb_proprio_v1", config=config)
        harness.start()
        harness.task_planning.required = False
        return harness, adapter, driver, service

    def dispatch(self, harness, **request):
        from ni_rua.runtime.actions.unified_actions import dispatch
        self.calls = getattr(self, "calls", 0) + 1
        with patch("ni_rua.runtime.actions.inspection._trajectory_image", return_value=({"status": "unavailable"}, [])):
            return dispatch(harness, dict(request), {"tool_call_id": f"call-{self.calls}"})

    def public_move(self, adapter, identifier, dx=0.0, dz=0.0, **fields):
        pose = self.pose(adapter)
        return dict(id=identifier, kind="move", arm="right", quaternion_xyzw=pose["quaternion_xyzw"],
                    position_m=[pose["position_m"][0] + dx, pose["position_m"][1], pose["position_m"][2] + dz],
                    position_tolerance_m=0.005, orientation_tolerance_rad=0.1, **fields)


# ---- ik_plan ------------------------------------------------------------------------------------------------------
class InterpolationTests(unittest.TestCase):
    def test_rows_end_at_the_target_and_step_count_follows_the_fastest_joint(self):
        start, target = [0.0, -1.0, 1.0, 0.6, 0.0], [0.4, -0.9, 1.0, 0.6, 0.1]
        steps = ik_plan.interpolation_steps(start, target, speed_rad_s=0.5, rate_hz=20)
        self.assertEqual(steps, math.ceil(0.4 / 0.5 * 20))
        rows = ik_plan.joint_interpolation(start, target, steps)
        self.assertEqual(len(rows), steps)
        np.testing.assert_allclose(rows[-1], target)
        np.testing.assert_allclose(rows[0], np.asarray(start) + (np.asarray(target) - start) / steps)
        self.assertEqual(ik_plan.interpolation_steps(start, start, 0.5, 20), 1)

    def test_path_facts_of_a_joint_line_report_the_dip_and_the_deviation(self):
        # shoulder_lift and elbow_flex both move: the TCP bulges off the straight Cartesian segment.
        start, target = [0.0, -0.6, 0.8, 1.3, 0.0], [0.0, 0.5, -0.5, 1.5, 0.0]
        rows = ik_plan.joint_interpolation(start, target, 80)
        facts = ik_plan.path_facts(kin, MODEL, start, rows, [[0.0, 0.0, 0.0063], [-0.0158, 0.0, 0.0063]])
        points = np.asarray([np.asarray(kin.fk(MODEL, list(q)))[:3, 3] for q in (start, *rows)])
        a, b = points[0], points[-1]
        along = np.clip((points - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
        deviation = np.linalg.norm(points - (a + along[:, None] * (b - a)), axis=1).max()
        self.assertEqual(facts["samples"], 81)
        self.assertAlmostEqual(facts["min_tcp_z_m"], float(points[:, 2].min()), places=4)
        self.assertAlmostEqual(facts["max_line_deviation_m"], float(deviation), places=4)
        self.assertGreater(facts["max_line_deviation_m"], 0.005)
        self.assertAlmostEqual(facts["dip_below_endpoints_m"], max(0.0, min(a[2], b[2]) - points[:, 2].min()), places=4)
        self.assertLess(facts["min_fingertip_z_m"], facts["min_tcp_z_m"] + 0.0064)
        self.assertTrue(0.0 <= facts["lowest_at_fraction"] <= 1.0)

    def test_fingertip_points_can_follow_the_jaw_per_sample(self):
        start, target = [0.0, -1.0, 1.0, 0.6, 0.0], [0.0, -0.9, 1.0, 0.6, 0.0]
        rows = ik_plan.joint_interpolation(start, target, 4)
        fixed = ik_plan.path_facts(kin, MODEL, start, rows, [[0.0, 0.0, 0.01]])
        moving = ik_plan.path_facts(kin, MODEL, start, rows, [[[0.0, 0.0, 0.01]]] * 4 + [[[0.0, 0.0, 0.01], [0.3, 0.0, 0.3]]])
        self.assertNotEqual(fixed["min_fingertip_z_m"], moving["min_fingertip_z_m"])
        self.assertNotIn("min_fingertip_z_m", ik_plan.path_facts(kin, MODEL, start, rows))

    def test_a_move_that_does_not_translate_the_tcp_has_no_deviation(self):
        q = [0.0, -1.0, 1.0, 0.6, 0.0]
        facts = ik_plan.path_facts(kin, MODEL, q, [q])
        self.assertEqual((facts["max_line_deviation_m"], facts["dip_below_endpoints_m"]), (0.0, 0.0))


class CollisionWorldTests(unittest.TestCase):
    def test_box_spheres_stay_inside_the_box(self):
        for low, high, margin in (([0, 0, 0], [0.03, 0.03, 0.03], 0.0), ([0, 0, 0], [0.2, 0.02, 0.01], 0.0),
                                  ([-0.01, 0.0, 0.1], [0.05, 0.04, 0.13], 0.01)):
            centres, radius = ik_plan.box_spheres(low, high, margin)
            self.assertTrue(1 <= len(centres) <= ik_plan.HELD_OBJECT_SPHERES)
            self.assertGreater(radius, 0)
            for centre in centres:
                for value, lo, hi in zip(centre, low, high):
                    self.assertGreaterEqual(value - radius, lo - margin - 1e-9)
                    self.assertLessEqual(value + radius, hi + margin + 1e-9)
        centres, radius = ik_plan.box_spheres([0, 0, 0], [0.03, 0.03, 0.03])
        self.assertEqual((len(centres), round(radius, 6)), (27, 0.005))

    def test_world_grows_obstacles_by_their_margin_and_fixes_the_held_box_to_the_tcp(self):
        T = np.asarray(kin.fk(MODEL, [0.3, -1.0, 1.0, 0.6, 0.2]))
        proxies = dict(obstacles=[BOX], held_objects=dict(right=dict(HELD, margin_m=0.0)))
        world = ik_plan.collision_world(proxies, "right", T)
        self.assertEqual(world["obstacles"], [dict(id="o1", min_m=[0.29, -0.03, -0.01], max_m=[0.35, 0.03, 0.06])])
        self.assertEqual(world["held_object"], dict(spheres=27, sphere_radius_m=round(0.04 / 6, 4)))
        back = [T[:3, :3] @ np.asarray(sphere[:3]) + T[:3, 3] for sphere in world["attached_spheres"]]
        for point in back:  # back in the base frame every sphere lies inside the held box
            for value, lo, hi in zip(point, HELD["min_m"], HELD["max_m"]):
                self.assertTrue(lo - 1e-9 <= value <= hi + 1e-9)

    def test_no_proxies_is_no_world_and_other_arms_are_ignored(self):
        self.assertIsNone(ik_plan.collision_world(None, "right", np.eye(4)))
        self.assertIsNone(ik_plan.collision_world(dict(obstacles=[], held_objects={}), "right", np.eye(4)))
        world = ik_plan.collision_world(dict(obstacles=[BOX], held_objects={}), "right", np.eye(4))
        self.assertEqual((world["attached_spheres"], world["held_object"]), ([], None))


class ObstacleAwareSolveTests(unittest.TestCase):
    """``curobo_solve(world=...)``: valid candidates not labelled colliding come first; never a failure."""

    SEED = [0.0, -1.0, 1.0, 0.6, 0.0]
    LIMITS = {name: (-3.0, 3.0) for name in ARM_JOINTS}

    class Client:
        def __init__(self, answers):
            self.answers, self.worlds = answers, []

        def solve(self, poses, seed, weights, position_tolerance, orientation_tolerance, **extra):
            self.worlds.append(extra.get("world"))
            return [deepcopy(self.answers) for _ in poses]

    def request(self, q):
        T = np.asarray(kin.fk(MODEL, q))
        return (dict(position_m=T[:3, 3].tolist(), quaternion_xyzw=ik_plan.matrix_to_quaternion(T[:3, :3])),
                (0.05, math.pi), None)

    def solve(self, answers, world):
        client = self.Client(answers)
        (q, record), = ik_plan.curobo_solve(client, kin, MODEL, self.LIMITS, [self.request(self.SEED)], self.SEED, world)
        return q, record, client

    def test_obstacle_free_candidate_wins_over_the_closer_colliding_one(self):
        far = [0.0, -1.0, 1.0, 0.6, 2.0]  # wrist_roll only: the TCP stays within the 5 cm tolerance, with more travel
        near = dict(q_rad=self.SEED, success=True, feasible=True, obstacle_free=False)
        clear = dict(q_rad=far, success=True, feasible=True, obstacle_free=True)
        world = dict(obstacles=[dict(BOX)], attached_spheres=[])
        q, record, client = self.solve([near, clear], world)
        self.assertEqual(q, far)
        self.assertTrue(record["ok"] and record["obstacle_free"])
        self.assertEqual(client.worlds, [world])

    def test_all_colliding_still_plans_and_says_so(self):
        q, record, _ = self.solve([dict(q_rad=self.SEED, success=True, feasible=True, obstacle_free=False)],
                                  dict(obstacles=[dict(BOX)], attached_spheres=[]))
        self.assertEqual(q, self.SEED)
        self.assertTrue(record["ok"])
        self.assertIs(record["obstacle_free"], False)

    def test_without_a_world_nothing_is_sent_or_labelled(self):
        q, record, client = self.solve([dict(q_rad=self.SEED, success=True, feasible=True, obstacle_free=False)], None)
        self.assertNotIn("obstacle_free", record)
        self.assertEqual(client.worlds, [None])

    def test_a_service_without_labels_leaves_the_record_unlabelled(self):
        _, record, _ = self.solve([dict(q_rad=self.SEED, success=True, feasible=True)],
                                  dict(obstacles=[dict(BOX)], attached_spheres=[]))
        self.assertTrue(record["ok"])
        self.assertNotIn("obstacle_free", record)


# ---- client -------------------------------------------------------------------------------------------------------
class Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class Server:
    def __init__(self, health=None, answer=None):
        self.requests, self.answer = [], answer
        self.health = health if health is not None else dict(status="ok", max_batch=4, path_check=dict(max_samples=100))

    def __call__(self, request, timeout=None):
        body = None if request.data is None else json.loads(request.data)
        self.requests.append(dict(url=request.full_url, body=body))
        if request.full_url.endswith("/health"):
            return Response(self.health)
        if request.full_url.endswith("/path_check"):
            if self.answer is not None:
                return Response(self.answer(body))
            return Response(dict(status="ok", check_ms=2.5, samples=0, segments=[
                dict(samples=3, colliding_samples=0, first_collision=None) for _ in body["segments"]]))
        return Response(dict(status="ok", solutions=[[dict(q_rad=[0.0] * 5, success=True)] for _ in body["poses"]]))


class ClientTests(unittest.TestCase):
    LIMITS = {name: (-1.0, 1.0) for name in ARM_JOINTS}

    def client(self, server, **settings):
        return CuroboClient(dict(SERVICE, timeout_s=5, num_solutions=2, table_top_z_m=-0.02, path_check_step_rad=0.01,
                                 **settings), self.LIMITS, opener=server)

    def segment(self, change=0.05, **fields):
        return dict(start_q_rad=[0.0] * 5, end_q_rad=[change, 0.0, 0.0, 0.0, 0.0], **fields)

    def test_path_check_body_and_answer(self):
        server = Server()
        world = dict(obstacles=[dict(id="o1", min_m=[0, 0, 0], max_m=[1, 1, 1])], attached_spheres=[[0, 0, 0.02, 0.01]],
                     held_object=dict(spheres=1, sphere_radius_m=0.01))
        answer = self.client(server).check_path([self.segment(start_jaw_rad=0.2, end_jaw_rad=0.9, attached=False)], world)
        body = server.requests[-1]["body"]
        self.assertTrue(server.requests[-1]["url"].endswith("/path_check"))
        self.assertEqual(body["segments"], [dict(start_q_rad=[0.0] * 5, end_q_rad=[0.05, 0.0, 0.0, 0.0, 0.0],
                                                 start_jaw_rad=0.2, end_jaw_rad=0.9, attached=False)])
        self.assertEqual((body["max_joint_step_rad"], body["table_top_z_m"]), (0.01, -0.02))
        self.assertEqual(body["joint_limits_rad"], {name: [-1.0, 1.0] for name in ARM_JOINTS})
        self.assertEqual(body["obstacles"], [dict(id="o1", min_m=[0.0, 0.0, 0.0], max_m=[1.0, 1.0, 1.0])])
        self.assertEqual(body["attached_spheres"], [[0.0, 0.0, 0.02, 0.01]])
        self.assertNotIn("held_object", body)
        self.assertEqual(answer, dict(check_ms=2.5, samples=3, segments=[
            dict(samples=3, colliding_samples=0, first_collision=None)]))

    def test_defaults_and_no_world(self):
        server = Server()
        self.client(server).check_path([self.segment()])
        body = server.requests[-1]["body"]
        self.assertEqual(body["segments"][0], dict(start_q_rad=[0.0] * 5, end_q_rad=[0.05, 0.0, 0.0, 0.0, 0.0],
                                                   start_jaw_rad=0.0, end_jaw_rad=0.0, attached=True))
        self.assertNotIn("obstacles", body)

    def test_segments_are_split_by_the_service_sample_limit_and_merged_in_order(self):
        server = Server(answer=lambda body: dict(status="ok", check_ms=1.0, segments=[
            dict(samples=int(round(segment["end_q_rad"][0] * 100)) + 1, colliding_samples=0, first_collision=None)
            for segment in body["segments"]]))
        segments = [self.segment(0.4), self.segment(0.5), self.segment(0.3), self.segment(0.2)]  # 42 + 52 + 32 + 22
        answer = self.client(server).check_path(segments)
        posts = [request["body"] for request in server.requests if request["url"].endswith("/path_check")]
        self.assertEqual([len(body["segments"]) for body in posts], [2, 2])
        self.assertEqual([item["samples"] for item in answer["segments"]], [41, 51, 31, 21])
        self.assertEqual((answer["samples"], answer["check_ms"]), (144, 2.0))

    def test_service_without_path_checker_is_named(self):
        with self.assertRaises(IKServiceError) as caught:
            self.client(Server(health=dict(status="ok", max_batch=4))).check_path([self.segment()])
        self.assertEqual(caught.exception.reason, "ik_service_path_check_unavailable")

    def test_bad_answers_and_http_errors_keep_stable_reasons(self):
        for answer in (lambda body: dict(status="ok", segments=[]),
                       lambda body: dict(status="ok", segments=[dict(samples="3", colliding_samples=0, first_collision=None)]),
                       lambda body: dict(status="ok", segments=[dict(samples=3, colliding_samples=0, first_collision=1)])):
            with self.assertRaises(IKServiceError) as caught:
                self.client(Server(answer=answer)).check_path([self.segment()])
            self.assertEqual(caught.exception.reason, "ik_service_bad_response")

        def refuse(request, timeout=None):
            if request.full_url.endswith("/health"):
                return Response(dict(status="ok", max_batch=4, path_check=dict(max_samples=100)))
            raise urllib.error.HTTPError(request.full_url, 400, "bad", {}, io.BytesIO(b'{"reason": "too_many_samples:9>8"}'))
        with self.assertRaises(IKServiceError) as caught:
            CuroboClient(dict(SERVICE, timeout_s=5, num_solutions=2), self.LIMITS, opener=refuse).check_path([self.segment()])
        self.assertEqual(caught.exception.reason, "ik_service_http_400:too_many_samples:9>8")

    def test_solve_sends_the_world_only_when_given(self):
        server = Server()
        client = self.client(server)
        pose = dict(position_m=[0.2, 0.0, 0.1], quaternion_xyzw=[1.0, 0.0, 0.0, 0.0])
        client.solve([pose], [0.0] * 5, [1, 1, 1, 0, 0, 0], 0.005, 0.1)
        self.assertFalse({"obstacles", "attached_spheres", "jaw_rad"} & set(server.requests[-1]["body"]))
        client.solve([pose], [0.0] * 5, [1, 1, 1, 0, 0, 0], 0.005, 0.1,
                     world=dict(obstacles=[dict(id="o1", min_m=[0, 0, 0], max_m=[1, 1, 1])], attached_spheres=[], jaw_rad=0.4))
        body = server.requests[-1]["body"]
        self.assertEqual((body["obstacles"][0]["id"], body["attached_spheres"], body["jaw_rad"]), ("o1", [], 0.4))


# ---- service pure helpers -----------------------------------------------------------------------------------------
class ServiceHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = service_module()

    def test_segment_samples_include_both_ends_and_respect_the_step(self):
        s = self.service
        rows = s.segment_samples(dict(start_q_rad=[0.0] * 5, end_q_rad=[0.1, 0.0, 0.0, 0.0, -0.05], end_jaw_rad=0.2), 0.02)
        self.assertEqual(len(rows), 1 + 10)  # the jaw moves most: 0.2 rad / 0.02
        self.assertEqual(rows[0], [0.0] * 6)
        np.testing.assert_allclose(rows[-1], [0.1, 0.0, 0.0, 0.0, -0.05, 0.2])
        self.assertLessEqual(max(abs(b - a) for first, second in zip(rows, rows[1:]) for a, b in zip(first, second)), 0.02 + 1e-9)
        self.assertEqual(len(s.segment_samples(dict(start_q_rad=[0.3] * 5, end_q_rad=[0.3] * 5), 0.02)), 1)
        for bad in (dict(start_q_rad=[0.0] * 4, end_q_rad=[0.0] * 5), dict(start_q_rad=[0.0] * 5),
                    dict(start_q_rad=[0.0] * 5, end_q_rad=[0.0] * 5, start_jaw_rad="x"),
                    dict(start_q_rad=[0.0] * 5, end_q_rad=[float("nan")] * 5)):
            with self.assertRaises(s.RequestError):
                s.segment_samples(bad, 0.02)

    def test_obstacles_and_spheres_are_validated(self):
        s = self.service
        self.assertEqual(s.parse_obstacles(None), [])
        self.assertEqual(s.parse_obstacles([dict(id="a", min_m=[0, 0, 0], max_m=[1, 2, 3])]), [("a", [0.0, 0.0, 0.0], [1.0, 2.0, 3.0])])
        for bad in ([dict(id="a", min_m=[0, 0, 0], max_m=[1, 1, 0])], [dict(id="table", min_m=[0, 0, 0], max_m=[1, 1, 1])],
                    [dict(id="a", min_m=[0, 0, 0], max_m=[1, 1, 1])] * 2, [dict(min_m=[0, 0, 0], max_m=[1, 1, 1])], "x",
                    [dict(id="a", min_m=[0, 0], max_m=[1, 1, 1])],
                    [dict(id=str(i), min_m=[0, 0, 0], max_m=[1, 1, 1]) for i in range(s.MAX_OBSTACLES + 1)]):
            with self.assertRaises(s.RequestError):
                s.parse_obstacles(bad)
        self.assertEqual(s.parse_spheres([[0, 0, 0.02, 0.01]]), [[0.0, 0.0, 0.02, 0.01]])
        for bad in ([[0, 0, 0, 0]], [[0, 0, 0]], [[0, 0, 0, 0.01]] * (s.ATTACHED_SPHERE_SLOTS + 1), 3):
            with self.assertRaises(s.RequestError):
                s.parse_spheres(bad)

    def test_box_overlaps_names_the_boxes_a_sphere_touches(self):
        boxes = [("a", [0.0, 0.0, 0.0], [0.1, 0.1, 0.1]), ("table", [-1.0, -1.0, -0.5], [1.0, 1.0, 0.0])]
        overlaps = self.service.box_overlaps
        self.assertEqual(overlaps([0.05, 0.05, 0.05, 0.001], boxes), ["a"])       # centre inside
        self.assertEqual(overlaps([0.05, 0.05, 0.11, 0.02], boxes), ["a"])        # touches the top face
        self.assertEqual(overlaps([0.05, 0.05, 0.13, 0.02], boxes), [])           # 3 cm above, radius 2 cm
        self.assertEqual(overlaps([0.5, 0.5, 0.005, 0.01], boxes), ["table"])
        self.assertEqual(overlaps([0.11, 0.11, 0.005, 0.02], boxes), ["a", "table"])

    def test_sphere_pairs_skip_same_link_ignored_pairs_and_unused_slots(self):
        spheres = [[0, 0, 0, 0.02], [0.03, 0, 0, 0.02], [0.03, 0, 0, 0.02], [0.03, 0, 0, -100.0], [0.031, 0, 0, 0.02]]
        links = ["a", "a", "b", "c", "d"]
        pairs = self.service.sphere_pair_overlaps(spheres, links, {frozenset(("a", "d"))})
        self.assertEqual([(a, b) for a, b, _ in pairs], [("a", "b"), ("b", "d")])  # deepest first, a-a / a-d / c skipped
        self.assertAlmostEqual(pairs[0][2], 0.04)  # the deeper of the two a-b pairs (coincident spheres)
        self.assertAlmostEqual(pairs[1][2], 0.039)

    def test_checker_model_frees_the_jaw_and_adds_the_attached_link(self):
        s = self.service
        source = dict(tool_frames=["gripper_frame_link"], lock_joints={"gripper": 0.0}, collision_link_names=["base_link"],
                      self_collision_ignore={"base_link": ["shoulder_link"]}, self_collision_buffer={"base_link": 0.0})
        out = s.checker_kinematics(source)
        self.assertEqual(source["lock_joints"], {"gripper": 0.0})  # the IK model is not modified
        self.assertIsNone(out["lock_joints"])
        self.assertEqual(out["collision_link_names"], ["base_link", s.ATTACHED_LINK])
        self.assertEqual(out["extra_collision_spheres"], {s.ATTACHED_LINK: s.ATTACHED_SPHERE_SLOTS})
        self.assertEqual(out["extra_links"][s.ATTACHED_LINK]["parent_link_name"], "gripper_frame_link")
        self.assertEqual(set(out["self_collision_ignore"][s.ATTACHED_LINK]), set(s.GRIPPER_LINKS))
        self.assertIn(frozenset(("base_link", "shoulder_link")), s.ignored_pairs(out))
        self.assertIn(frozenset((s.ATTACHED_LINK, "gripper_link")), s.ignored_pairs(out))


# ---- worker preview -----------------------------------------------------------------------------------------------
class PreviewPathTests(Case):
    def test_path_facts_and_overlay_samples_follow_the_interpolation_the_worker_executes(self):
        adapter, _, _ = self.adapter(ik_service=False)
        start = adapter.read()["q_rad"]
        action = batch(self.move(adapter, "out", dx=0.06), self.move(adapter, "up", dx=0.06, dz=0.03))
        preview, receipt = self.run_batch(adapter, action)
        self.assertNotIn("path_check", preview)  # no cuRobo service: FK facts only
        plan = preview["stage_plans"]["out"]
        self.assertEqual(set(plan["path"]), {"samples", "min_tcp_z_m", "min_fingertip_z_m", "lowest_at_fraction",
                                             "max_line_deviation_m", "dip_below_endpoints_m"})
        steps = preview["stage_costs"]["out"]["native_steps"]
        rows = ik_plan.joint_interpolation(start, plan["q_rad"], steps)
        self.assertEqual(plan["path"], ik_plan.path_facts(kin, MODEL, start, rows, adapter._fingertips(adapter.read()["gripper_rad"])))
        self.assertEqual(plan["path"]["samples"], steps + 1)
        # Overlay samples: several commands of each move, bounded, in step order, each move's last one at its target.
        samples = preview["trajectory_data"]["samples"]
        by_stage = {name: [s for s in samples if s["stage_id"] == name] for name in ("out", "up")}
        for name, items in by_stage.items():
            self.assertTrue(1 < len(items) <= PATH_OVERLAY_SAMPLES, len(items))
            np.testing.assert_allclose(items[-1]["q_rad"], preview["stage_plans"][name]["q_rad"])
            self.assertEqual(items[-1]["native_step"], preview["stage_costs"][name]["end_native_step"])
        steps_listed = [s["native_step"] for s in samples]
        self.assertEqual(steps_listed, sorted(set(steps_listed)))
        middle = by_stage["out"][len(by_stage["out"]) // 2]
        np.testing.assert_allclose(middle["q_rad"], rows[middle["native_step"] - 1], atol=1e-12)
        np.testing.assert_allclose(middle["arms"]["right"]["link_positions_m"]["gripper_frame_link"],
                                   np.asarray(kin.fk(MODEL, middle["q_rad"]))[:3, 3], atol=1e-12)
        # Execution streamed the same line: its motion rows of the first move lie on the previewed rows.
        self.assertEqual(receipt["status"], "completed", receipt)
        executed = [row[:5] for row, kind in zip(receipt["planned_trace"]["q_rad"], receipt["planned_trace"]["row_kinds"])
                    if kind == "motion"][:steps]
        np.testing.assert_allclose(executed, rows, atol=2e-3)
        deviation = receipt["stage_results"]["out"]["path_start_deviation_rad"]
        self.assertLess(deviation, 2e-3)
        self.assertAlmostEqual(float(np.abs(np.subtract(executed, rows)).max()), deviation, delta=1e-4)
        self.assertIn("path_start_deviation_rad", receipt["stage_results"]["up"])

    def test_a_jaw_ramp_keeps_the_path_on_the_same_joint_line(self):
        adapter, _, _ = self.adapter(ik_service=False, gripper_normalized=1.0)
        pose = self.pose(adapter)
        leaf = contract.compile_action("move", dict(
            arm="right", observation_id="o", quaternion_xyzw=pose["quaternion_xyzw"],
            position_m=[pose["position_m"][0] + 0.03, *pose["position_m"][1:]], gripper_goal_normalized=0.2,
            gripper_motion=dict(start_fraction=0.0, end_fraction=1.0)), IDENTITY, adapter.capabilities())
        leaf["path"].update(position_tolerance_m=0.005, orientation_tolerance_rad=0.1)
        preview = adapter.preview_batch(batch(dict(id="m", kind="move", arm="right", action=leaf)))
        self.assertEqual(preview["status"], "planned", preview)
        plan = preview["stage_plans"]["m"]
        self.assertEqual(plan["path"]["samples"], preview["stage_costs"]["m"]["native_steps"] + 1)
        jaws = [s["arms"]["right"]["q_rad"][5] for s in preview["trajectory_data"]["samples"]]
        self.assertGreater(jaws[0], jaws[-1])  # the overlay samples carry the ramping jaw

    def test_path_check_clear_and_what_is_sent(self):
        adapter, _, service = self.adapter()
        start = adapter.read()
        action = batch(self.move(adapter, "a", dx=0.03), self.jaw(adapter, "open", 1.0), self.move(adapter, "b", dx=0.05))
        preview = adapter.preview_batch(action)
        self.assertEqual(preview["status"], "planned", preview)
        self.assertEqual(preview["path_check"], dict(status="clear", colliding_stage_ids=[], max_joint_step_rad=0.01,
                                                     obstacle_ids=[], checked_path="commanded"))
        self.assertNotIn("collision", preview["stage_plans"]["a"]["path"])
        (check,) = service.checks
        self.assertIsNone(check["world"])
        first, sweep, second = check["segments"]
        opened = adapter.deployment["gripper"]["jaw_open_rad"]
        np.testing.assert_allclose(first["start_q_rad"], start["q_rad"])
        np.testing.assert_allclose(first["end_q_rad"], preview["stage_plans"]["a"]["q_rad"])
        np.testing.assert_allclose(second["start_q_rad"], first["end_q_rad"])
        # Jaw angle: the measured one until the plan commands the jaw, then the commanded goal.
        self.assertAlmostEqual(first["start_jaw_rad"], start["gripper_rad"])
        self.assertAlmostEqual(first["end_jaw_rad"], start["gripper_rad"])
        self.assertAlmostEqual(second["start_jaw_rad"], opened)
        self.assertAlmostEqual(second["end_jaw_rad"], opened)
        # The standalone gripper stage is its own segment: the arm stays where the first move ended and the jaw sweeps.
        np.testing.assert_allclose(sweep["start_q_rad"], first["end_q_rad"])
        np.testing.assert_allclose(sweep["end_q_rad"], first["end_q_rad"])
        self.assertAlmostEqual(sweep["start_jaw_rad"], start["gripper_rad"])
        self.assertAlmostEqual(sweep["end_jaw_rad"], opened)
        self.assertFalse(sweep["attached"])
        self.assertTrue(all(solve["world"] is None for solve in service.solves))

    def test_collisions_are_advisory_facts_and_the_plan_still_executes(self):
        adapter, _, service = self.adapter()
        action = batch(self.move(adapter, "a", dx=0.03), self.move(adapter, "b", dx=0.05))
        q = [0.0, -0.9, 0.9, 0.7, 0.0]
        service.collisions = {1: dict(sample_index=4, fraction=0.44, q_rad=q, jaw_rad=0.0,
                                      world=[dict(link="gripper_link", penetration_m=0.004, obstacles=["table"])],
                                      self=[dict(links=["gripper_link", "shoulder_link"], penetration_m=0.001)])}
        preview, receipt = self.run_batch(adapter, action)
        self.assertEqual((preview["path_check"]["status"], preview["path_check"]["colliding_stage_ids"]), ("collisions", ["b"]))
        collision = preview["stage_plans"]["b"]["path"]["collision"]
        self.assertEqual(collision, dict(
            fraction=0.44, colliding_samples=3, samples=10,
            world=[dict(link="gripper_link", penetration_m=0.004, obstacles=["table"])],
            self=[dict(links=["gripper_link", "shoulder_link"], penetration_m=0.001)],
            tcp_position_m=[round(float(v), 4) for v in np.asarray(kin.fk(MODEL, q))[:3, 3]], jaw_rad=0.0))
        self.assertNotIn("collision", preview["stage_plans"]["a"]["path"])
        self.assertEqual(receipt["status"], "completed", receipt)  # nothing was blocked or altered

    def test_unavailable_check_is_reported_with_its_reason(self):
        adapter, _, service = self.adapter()
        service.error = IKServiceError("ik_service_path_check_unavailable")
        preview = adapter.preview_batch(batch(self.move(adapter, "a", dx=0.03)))
        self.assertEqual(preview["status"], "planned")
        self.assertEqual(preview["path_check"]["status"], "unavailable")
        self.assertEqual(preview["path_check"]["reason"], "ik_service_path_check_unavailable")

    def test_a_gripper_only_batch_checks_its_jaw_sweep(self):
        adapter, _, service = self.adapter()
        start = adapter.read()
        action = batch(self.jaw(adapter, "open", 1.0))
        preview = adapter.preview_batch(action)
        self.assertEqual((preview["path_check"]["status"], preview["path_check"]["colliding_stage_ids"]), ("clear", []))
        (check,) = service.checks
        (sweep,) = check["segments"]
        np.testing.assert_allclose(sweep["start_q_rad"], start["q_rad"])
        np.testing.assert_allclose(sweep["end_q_rad"], start["q_rad"])  # the arm does not move
        self.assertAlmostEqual(sweep["start_jaw_rad"], start["gripper_rad"])
        self.assertAlmostEqual(sweep["end_jaw_rad"], adapter.deployment["gripper"]["jaw_open_rad"])
        self.assertNotIn("path", preview["stage_plans"]["open"])
        # A collision of the sweep is a fact of the gripper stage, with the jaw angle where it first collides.
        service.collisions = {0: dict(sample_index=2, fraction=0.5, q_rad=start["q_rad"], jaw_rad=0.6,
                                      world=[dict(link="moving_jaw_so101_v1_link", penetration_m=0.002,
                                                  obstacles=["table"])], self=[])}
        preview, receipt = self.run_batch(adapter, action)
        self.assertEqual((preview["path_check"]["status"], preview["path_check"]["colliding_stage_ids"]),
                         ("collisions", ["open"]))
        self.assertEqual(preview["stage_plans"]["open"]["path"]["collision"], dict(
            fraction=0.5, colliding_samples=3, samples=10, self=[], jaw_rad=0.6,
            world=[dict(link="moving_jaw_so101_v1_link", penetration_m=0.002, obstacles=["table"])],
            tcp_position_m=[round(float(v), 4) for v in np.asarray(kin.fk(MODEL, start["q_rad"]))[:3, 3]]))
        self.assertEqual(receipt["status"], "completed", receipt)  # advisory: the jaw still opens

    def test_a_batch_without_moves_or_gripper_stages_has_no_path_check(self):
        adapter, _, service = self.adapter()
        preview = adapter.preview_batch(batch(dict(id="look", kind="observe", reason="check")))
        self.assertNotIn("path_check", preview)
        self.assertEqual(service.checks, [])


class CollisionWorldPreviewTests(Case):
    def proxies(self, adapter):
        tcp = np.asarray(self.pose(adapter)["position_m"])
        held = dict(min_m=(tcp - 0.02).tolist(), max_m=(tcp + 0.02).tolist(), margin_m=0.0, evidence="held cube")
        return dict(obstacles=[dict(BOX)], held_objects=dict(right=held))

    def test_world_reaches_ik_and_path_check_and_the_held_box_detaches_when_the_jaw_opens(self):
        adapter, _, service = self.adapter(gripper_normalized=0.0)
        action = batch(self.move(adapter, "carry", dx=0.03), self.move(adapter, "lower", dx=0.03, dz=-0.02),
                       self.jaw(adapter, "release", 1.0), self.move(adapter, "retreat", dx=0.03),
                       review_proxies=self.proxies(adapter))
        service.labels = True
        preview = adapter.preview_batch(action)
        self.assertEqual(preview["status"], "planned", preview)
        review = preview["path_check"]
        self.assertEqual(review["obstacle_ids"], ["o1"])
        self.assertEqual(review["held_object"], dict(spheres=27, sphere_radius_m=round(0.04 / 6, 4),
                                                     attached_through_stage_id="release"))
        (check,) = service.checks
        # carry, lower, the release sweep (the box still rides the gripper while the jaw opens), retreat
        self.assertEqual([segment["attached"] for segment in check["segments"]], [True, True, True, False])
        self.assertEqual(check["world"]["obstacles"], [dict(id="o1", min_m=[0.29, -0.03, -0.01], max_m=[0.35, 0.03, 0.06])])
        self.assertEqual(len(check["world"]["attached_spheres"]), 27)
        # IK of each move saw the world as of that stage: the held spheres only while attached.
        self.assertEqual([len(solve["world"]["attached_spheres"]) for solve in service.solves], [27, 27, 0])
        self.assertEqual(service.solves[0]["world"]["obstacles"], check["world"]["obstacles"])
        self.assertAlmostEqual(service.solves[2]["world"]["jaw_rad"], adapter.deployment["gripper"]["jaw_open_rad"])
        self.assertIs(preview["stage_plans"]["carry"]["ik"]["obstacle_free"], True)

    def test_colliding_target_is_labelled_and_never_fails_planning(self):
        adapter, _, service = self.adapter()
        service.labels = False
        preview = adapter.preview_batch(batch(self.move(adapter, "m", dx=0.03), review_proxies=self.proxies(adapter)))
        self.assertEqual(preview["status"], "planned", preview)
        self.assertIs(preview["stage_plans"]["m"]["ik"]["obstacle_free"], False)

    def test_held_box_without_an_opening_stage_stays_attached(self):
        adapter, _, service = self.adapter(gripper_normalized=0.0)
        preview = adapter.preview_batch(batch(self.move(adapter, "a", dx=0.03), self.jaw(adapter, "squeeze", 0.0),
                                              self.move(adapter, "b", dx=0.03), review_proxies=self.proxies(adapter)))
        self.assertIsNone(preview["path_check"]["held_object"]["attached_through_stage_id"])
        self.assertEqual([segment["attached"] for segment in service.checks[0]["segments"]], [True, True, True])

    def test_proxies_are_part_of_the_plan_digest(self):
        adapter, _, _ = self.adapter()
        action = batch(self.move(adapter, "m", dx=0.03), review_proxies=self.proxies(adapter))
        preview = adapter.preview_batch(action)
        changed = deepcopy(action)
        changed["review_proxies"]["obstacles"][0]["max_m"][2] = 0.2
        receipt = adapter.execute(dict(changed, batch_plan_id=preview["plan_id"]), lambda update: None, threading.Event())
        self.assertEqual((receipt["status"], receipt["reason"]), ("rejected", "batch_plan_commands_changed"))

    def test_contract_accepts_proxies_only_with_the_tag_and_refuses_contact_declarations(self):
        with_service, plain = deployment(True), deployment(False)
        adapter, _, _ = self.adapter()
        action = batch(self.move(adapter, "m", dx=0.03), review_proxies=self.proxies(adapter))
        contract.validate_action(action, with_service)
        with self.assertRaisesRegex(ValueError, "unsupported_capability: review_proxies"):
            contract.validate_action(action, plain)
        declared = deepcopy(action)
        declared["review_proxies"]["contact_entity_id"] = "o1"
        with self.assertRaisesRegex(ValueError, "unsupported_capability: contact_entity_id"):
            contract.validate_action(declared, with_service)
        other_arm = deepcopy(action)
        other_arm["review_proxies"]["held_objects"] = dict(left=other_arm["review_proxies"]["held_objects"]["right"])
        with self.assertRaisesRegex(ValueError, "unsupported_arm:left"):
            contract.validate_action(other_arm, with_service)
        broken = deepcopy(action)
        broken["review_proxies"]["obstacles"][0]["max_m"] = broken["review_proxies"]["obstacles"][0]["min_m"]
        with self.assertRaisesRegex(ValueError, "invalid_motion_proxy_bounds"):
            contract.validate_action(broken, with_service)


# ---- robot_action through the Harness ---------------------------------------------------------------------------------
class RobotActionTests(Case):
    def test_obstacles_and_held_object_compile_into_the_collision_world(self):
        harness, adapter, _, service = self.harness()
        tcp = self.pose(adapter)["position_m"]
        held = dict(min_m=[tcp[0] - 0.02, tcp[1] - 0.02, tcp[2] - 0.04], max_m=[tcp[0] + 0.02, tcp[1] + 0.02, tcp[2]],
                    evidence="cube seen in the jaw")
        preview = self.dispatch(harness, observation_id=harness.latest["observation_id"],
                                stages=[self.public_move(adapter, "m", dx=0.03)],
                                obstacles=[dict(BOX)], held_object=held).data
        self.assertEqual(preview["status"], "planned", preview)
        self.assertEqual(preview["path_check"]["obstacle_ids"], ["o1"])
        self.assertEqual(preview["path_check"]["held_object"]["attached_through_stage_id"], None)
        world = service.checks[0]["world"]
        self.assertEqual(world["obstacles"][0]["id"], "o1")
        self.assertGreater(len(world["attached_spheres"]), 0)
        # Stateless: a later request without obstacle fields carries none.
        self.dispatch(harness, observation_id=harness.latest["observation_id"], stages=[self.public_move(adapter, "n", dx=0.03)])
        self.assertIsNone(service.checks[1]["world"])

    def test_object_table_rows_become_obstacle_boxes(self):
        harness, adapter, _, service = self.harness()
        table = dict(columns=["id", "centre_m", "top_z", "bottom_z", "length", "width", "yaw_deg", "quality", "support"],
                     table_z=0.0, rows=[["o1", [0.30, 0.05, 0.02], 0.04, 0.0, 0.04, 0.04, 0.0, "ok", "table"],
                                        ["o2", [0.10, 0.20, 0.02], 0.04, 0.0, 0.04, 0.04, 0.0, "ok", "table"],
                                        ["o3", [0.20, 0.00, 0.10], 0.12, 0.08, 0.04, 0.04, 0.0, "held_fk_depth", "held_right"]])
        with patch("ni_rua.perception.external.pose_skill.attached_table", return_value=table):
            preview = self.dispatch(harness, observation_id=harness.latest["observation_id"],
                                    stages=[self.public_move(adapter, "m", dx=0.03)],
                                    obstacle_object_ids=["o1", "o3", "missing"]).data
        self.assertEqual(preview["path_check"]["obstacle_ids"], ["o1"])  # held rows and unknown ids add nothing
        (box,) = service.checks[0]["world"]["obstacles"]
        np.testing.assert_allclose(box["min_m"], [0.275, 0.025, -0.005], atol=1e-9)  # footprint and height + 5 mm margin
        np.testing.assert_allclose(box["max_m"], [0.325, 0.075, 0.045], atol=1e-9)
        # Nothing is dropped silently: the preview names each id that added no box, with the reason.
        self.assertEqual(preview["path_check"]["object_row_boxes"], dict(skipped=[
            dict(id="o3", reason="held_row"), dict(id="missing", reason="not_in_table")]))

    def test_object_row_size_rules_are_reported_and_published(self):
        """Finding 4: the size floors of a row box are stated in the field help and every one applied is reported."""
        harness, adapter, _, service = self.harness()
        columns = ["id", "centre_m", "top_z", "bottom_z", "length", "width", "yaw_deg", "quality", "support"]
        table = dict(columns=columns, table_z=0.0, rows=[
            ["o1", [0.30, 0.05, 0.02], 0.04, 0.0, 0.04, 0.04, 0.0, "ok", "table"],
            # 1 cm long, no width, no bottom: length raised to 0.015, width 0.03, half extent x 0.0075 -> 0.008,
            # bottom = the table height, here 0.0 m (a real height, not a missing value).
            ["thin", [0.10, 0.20, 0.02], 0.04, None, 0.01, None, 0.0, "ok", "table"],
            ["nocentre", None, 0.04, 0.0, 0.04, 0.04, 0.0, "ok", "table"]])
        observation = harness.latest["observation_id"]
        with patch("ni_rua.perception.external.pose_skill.attached_table", return_value=table):
            preview = self.dispatch(harness, observation_id=observation, stages=[self.public_move(adapter, "m", dx=0.03)],
                                    obstacle_object_ids=["o1", "thin", "nocentre"]).data
            plain = self.dispatch(harness, observation_id=observation, stages=[self.public_move(adapter, "m", dx=0.03)],
                                  obstacle_object_ids=["o1"]).data
        self.assertEqual(preview["path_check"]["obstacle_ids"], ["o1", "thin"])
        self.assertEqual(preview["path_check"]["object_row_boxes"], dict(
            skipped=[dict(id="nocentre", reason="no_centre")],
            size_floors=[dict(id="thin", field="length_m", row_value=0.01, used_value=0.015),
                         dict(id="thin", field="width_m", row_value=None, used_value=0.03),
                         dict(id="thin", field="half_extent_x_m", row_value=0.0075, used_value=0.008),
                         dict(id="thin", field="bottom_z_m", row_value=None, used_value=0.0)]))
        thin = service.checks[0]["world"]["obstacles"][1]
        np.testing.assert_allclose(thin["min_m"], [0.10 - 0.008 - 0.005, 0.20 - 0.015 - 0.005, -0.005], atol=1e-9)
        # A request whose rows convert unchanged carries no report (nothing to add to the context).
        self.assertNotIn("object_row_boxes", plain["path_check"])
        # The observation has no object table: every requested id is named.
        with patch("ni_rua.perception.external.pose_skill.attached_table", return_value=None):
            empty = self.dispatch(harness, observation_id=observation, stages=[self.public_move(adapter, "m", dx=0.03)],
                                  obstacle_object_ids=["o1"]).data
        self.assertEqual(empty["path_check"]["object_row_boxes"], dict(skipped=[dict(id="o1", reason="no_object_table")]))
        self.assertEqual(empty["path_check"]["obstacle_ids"], [])
        # The field help publishes the rules with the numbers the converter uses.
        from ni_rua.robot.motion import scene_obstacles as rows
        text = build(harness.embodiment).tool["inputSchema"]["properties"]["obstacle_object_ids"]["description"]
        for needle in (f"under {rows.ROW_MIN_FOOTPRINT_M:g} is raised to {rows.ROW_MIN_FOOTPRINT_M:g}",
                       f"a missing one is {rows.ROW_DEFAULT_FOOTPRINT_M:g}", f"at least {rows.ROW_MIN_HALF_EXTENT_M:g}",
                       f"at least {rows.ROW_MIN_HEIGHT_M:g} tall", f"centre z + {rows.ROW_DEFAULT_TOP_ABOVE_CENTRE_M:g}",
                       f"margin_m {rows.ROW_OBSTACLE_MARGIN_M:g}", "path_check.object_row_boxes",
                       "not_in_table | held_row | no_centre | no_object_table"):
            self.assertIn(needle, text)
        # The X5 (proxy review) text keeps its wording; the boxes are the same with or without a report.
        x5 = build(None).tool["inputSchema"]["properties"]
        self.assertNotIn("object_row_boxes", x5["obstacle_object_ids"]["description"])
        self.assertEqual(x5["obstacles"]["items"]["properties"]["id"]["description"],
                         "Unique obstacle entity ID referenced by contact declarations.")
        so101_id = build(harness.embodiment).tool["inputSchema"]["properties"]["obstacles"]["items"]["properties"]["id"]
        self.assertNotIn("contact declarations", so101_id["description"])
        with patch("ni_rua.perception.external.pose_skill.attached_table", return_value=table):
            report = {}
            with_report = rows.obstacles_from_object_table(harness, observation, ["thin", "o1"], report=report)
            self.assertEqual(with_report, rows.obstacles_from_object_table(harness, observation, ["thin", "o1"]))
            self.assertEqual(rows.obstacles_from_object_table(harness, observation),
                             rows.obstacles_from_object_table(harness, observation, report={}))

    def test_obstacle_fields_are_absent_without_the_service(self):
        harness, adapter, _, _ = self.harness(ik_service=False)
        from jsonschema import ValidationError
        with self.assertRaises(ValidationError):
            self.dispatch(harness, observation_id=harness.latest["observation_id"],
                          stages=[self.public_move(adapter, "m", dx=0.03)], obstacles=[dict(BOX)])

    def test_execute_if_clean_runs_a_clear_plan_and_keeps_a_colliding_one_for_inspection(self):
        harness, adapter, _, service = self.harness()
        clean = self.dispatch(harness, mode="execute_if_clean", observation_id=harness.latest["observation_id"],
                              stages=[self.public_move(adapter, "m", dx=0.03)])
        self.assertEqual(clean.data["receipt"]["status"], "completed", clean.data)
        hit = dict(sample_index=1, fraction=0.1, q_rad=[0.0, -0.9, 0.9, 0.7, 0.0], jaw_rad=0.0,
                   world=[dict(link="gripper_link", penetration_m=0.002, obstacles=["o1"])], self=[])
        for collisions, error, reason in (({0: hit}, None, "path_check_collisions"),
                                          ({}, IKServiceError("ik_service_unreachable"), "path_check_unavailable")):
            service.collisions, service.error = collisions, error
            kept = self.dispatch(harness, mode="execute_if_clean", observation_id=harness.latest["observation_id"],
                                 stages=[self.public_move(adapter, "m", dx=0.03)]).data
            self.assertEqual(kept["status"], "planned")
            self.assertTrue(kept["downgraded_from_execute_if_clean"])
            self.assertEqual(kept["downgrade_reasons"], [reason])
            self.assertNotIn("receipt", kept)
        # The model can still execute the kept plan: the review is advisory.
        done = self.dispatch(harness, mode="execute", plan_id=kept["plan_id"])
        self.assertEqual(done.data["receipt"]["status"], "completed")

    def test_arrival_correction_fields_resolve_stage_over_batch_into_the_worker(self):
        harness, adapter, _, _ = self.harness(ik_service=False)
        preview = self.dispatch(harness, observation_id=harness.latest["observation_id"], arrival_correction_rounds=1,
                                arrival_correction_max_offset_rad=0.03,
                                stages=[self.public_move(adapter, "batch_default", dx=0.02),
                                        self.public_move(adapter, "own", dx=0.04, arrival_correction_rounds=0)]).data
        tolerances = {name: preview["stage_plans"][name]["tolerances"] for name in ("batch_default", "own")}
        self.assertEqual((tolerances["batch_default"]["arrival_correction_rounds"],
                          tolerances["batch_default"]["arrival_correction_max_offset_rad"]), (1, 0.03))
        self.assertEqual((tolerances["own"]["arrival_correction_rounds"],
                          tolerances["own"]["arrival_correction_max_offset_rad"]), (0, 0.03))
        from jsonschema import ValidationError
        for bad in (dict(arrival_correction_rounds=-1), dict(arrival_correction_rounds=1.5),
                    dict(arrival_correction_max_offset_rad=0), dict(arrival_correction_max_offset_rad=0.6)):
            with self.assertRaises((ValidationError, ValueError)):
                self.dispatch(harness, observation_id=harness.latest["observation_id"],
                              stages=[self.public_move(adapter, "m", dx=0.02, **bad)])
        # Above this deployment's own bound (3 rounds, 0.1 rad): the published schema carries that bound, so the
        # request fails the schema with it (finding: the schema used to publish 10 / 0.5 and only the backend refused).
        for bad, text in ((dict(arrival_correction_rounds=4), "4 is greater than the maximum of 3"),
                          (dict(arrival_correction_max_offset_rad=0.2), "0.2 is greater than the maximum of 0.1")):
            with self.assertRaisesRegex(ValidationError, text):
                self.dispatch(harness, observation_id=harness.latest["observation_id"],
                              stages=[self.public_move(adapter, "m", dx=0.02, **bad)])
            with self.assertRaisesRegex(ValidationError, text):  # the batch-wide default has the same bound
                self.dispatch(harness, observation_id=harness.latest["observation_id"],
                              stages=[self.public_move(adapter, "m", dx=0.02)], **bad)


# ---- arrival correction in the worker -----------------------------------------------------------------------------
class ArrivalCorrectionTests(Case):
    SAG = (0.0, 0.03, 0.0, 0.0, 0.0)  # shoulder_lift settles 0.03 rad short of its goal

    def test_default_is_the_deployment_bound_and_corrects_the_sag(self):
        adapter, _, _ = self.adapter(ik_service=False, sag_rad=self.SAG)
        preview, receipt = self.run_batch(adapter, batch(self.move(adapter, "m", dx=0.03)))
        motion = adapter.deployment["motion"]
        expected = (motion["arrival_correction_rounds"], motion["arrival_correction_max_offset_rad"])
        tolerances = preview["stage_plans"]["m"]["tolerances"]
        self.assertEqual((tolerances["arrival_correction_rounds"], tolerances["arrival_correction_max_offset_rad"]), expected)
        result = receipt["stage_results"]["m"]
        self.assertEqual(result["effective_tolerances"], tolerances)
        correction = result["arrival_correction"]
        self.assertEqual((correction["max_rounds"], correction["max_offset_rad"]), expected)
        self.assertGreaterEqual(len(correction["rounds"]), 1)
        self.assertLess(result["position_error_m"], correction["initial_tcp_residual"]["position_error_m"])

    def test_zero_rounds_sends_no_correction_and_judges_where_the_arm_settled(self):
        adapter, _, _ = self.adapter(ik_service=False, sag_rad=self.SAG)
        stage = self.move(adapter, "m", dx=0.03, arrival_correction_rounds=0)
        stage["action"]["path"]["position_tolerance_m"] = 0.03  # wide enough for the uncorrected sag
        preview, receipt = self.run_batch(adapter, batch(stage))
        self.assertEqual(preview["stage_plans"]["m"]["tolerances"]["arrival_correction_rounds"], 0)
        result = receipt["stage_results"]["m"]
        correction = result["arrival_correction"]
        self.assertEqual((correction["rounds"], correction["max_rounds"], correction["termination_reason"]), ([], 0, "disabled"))
        self.assertFalse(correction["offset_limit_reached"])
        self.assertNotIn("arrival_correction", receipt["planned_trace"]["row_kinds"])
        self.assertAlmostEqual(result["position_error_m"], correction["initial_tcp_residual"]["position_error_m"], delta=5e-4)
        self.assertAlmostEqual(result["max_joint_error_rad"], 0.03, delta=2e-3)

    def test_offset_bound_limits_the_shift(self):
        adapter, driver, _ = self.adapter(ik_service=False, sag_rad=self.SAG)
        stage = self.move(adapter, "m", dx=0.03, arrival_correction_max_offset_rad=0.01)
        stage["action"]["path"]["position_tolerance_m"] = 0.03
        _, receipt = self.run_batch(adapter, batch(stage))
        result = receipt["stage_results"]["m"]
        correction = result["arrival_correction"]
        self.assertEqual(correction["max_offset_rad"], 0.01)
        self.assertTrue(correction["offset_limit_reached"])
        self.assertTrue(all(max(abs(v) for v in item["goal_offset_rad"]) <= 0.01 + 1e-9 for item in correction["rounds"]))
        target = np.asarray(receipt["planned_trace"]["ik_targets"][0]["q_rad"])
        self.assertLessEqual(float(np.abs(np.asarray(driver.goal) - target).max()), 0.01 + 1e-6)

    def test_values_above_the_deployment_bound_and_untagged_profiles_are_refused(self):
        deploy = deployment(False)
        profile = contract.embodiment(deploy)
        base = dict(kind="backend_free")
        contract._validate_path_fields(dict(base, arrival_correction_rounds=3, arrival_correction_max_offset_rad=0.1), profile, deploy)
        for bad in (dict(arrival_correction_rounds=4), dict(arrival_correction_rounds=True), dict(arrival_correction_rounds=-1),
                    dict(arrival_correction_max_offset_rad=0.11), dict(arrival_correction_max_offset_rad=0)):
            with self.assertRaises(ValueError, msg=bad):
                contract._validate_path_fields(dict(base, **bad), profile, deploy)
        self.assertEqual(contract.arrival_correction({}, deploy), (3, 0.1))
        self.assertEqual(contract.arrival_correction(dict(arrival_correction_rounds=0), deploy), (0, 0.1))
        with self.assertRaisesRegex(ValueError, "unsupported_capability: arrival_correction_rounds"):
            mt.resolve_arrival_correction(dict(arrival_correction_rounds=0), {}, profile=embodiments.ARX_X5_PROFILE)
        self.assertEqual(mt.resolve_arrival_correction({}, {}, profile=embodiments.ARX_X5_PROFILE), {})
        self.assertEqual(mt.resolve_arrival_correction(dict(arrival_correction_rounds=0),
                                                       dict(arrival_correction_rounds=2, arrival_correction_max_offset_rad=0.05),
                                                       profile=profile),
                         dict(arrival_correction_rounds=0, arrival_correction_max_offset_rad=0.05))


# ---- receipts -----------------------------------------------------------------------------------------------------
class ReceiptResidualTests(Case):
    def blocked_descent(self, **options):
        """A 6 cm descent onto a table 3 cm above its target, without a stall guard: the move fails."""
        adapter, driver, _ = self.adapter(ik_service=False, **options)
        driver.table_z_m = self.pose(adapter)["position_m"][2] - 0.03
        return adapter, self.move(adapter, "down", dz=-0.06)

    def test_failed_move_mirrors_its_measured_residual_to_receipt_level(self):
        adapter, stage = self.blocked_descent(gripper_normalized=1.0)
        _, receipt = self.run_batch(adapter, batch(stage, self.move(adapter, "after", dz=0.0)))
        self.assertEqual((receipt["status"], receipt["reason"], receipt["failed_stage_id"]), ("failed", "tracking_error", "down"))
        residual = receipt["endpoint_residual_m"]
        self.assertEqual(set(residual), {"dx_m", "dy_m", "dz_m", "position_error_m", "orientation_error_rad"})
        self.assertAlmostEqual(residual["dz_m"], 0.03, delta=0.004)
        self.assertAlmostEqual(residual["dz_m"], receipt["contact_block_summary"]["vertical_shortfall_m"], delta=1e-3)
        self.assertNotIn("endpoint_residual_m", receipt["stage_results"]["down"])  # stored once, at receipt level
        self.assertNotIn("in_hand_tracking_error", receipt)  # jaw open: nothing held

    def test_completed_batch_has_no_receipt_level_residual(self):
        adapter, _, _ = self.adapter(ik_service=False)
        _, receipt = self.run_batch(adapter, batch(self.move(adapter, "m", dx=0.02)))
        self.assertEqual(receipt["status"], "completed")
        self.assertNotIn("endpoint_residual_m", receipt)
        self.assertNotIn("endpoint_residual_m", receipt["stage_results"]["m"])

    def test_in_hand_tracking_error_needs_a_closed_jaw_that_measures_wider_than_an_empty_close(self):
        # The fake jaw stops at 0.15 normalized (an object of about 3 cm) although commanded to close.
        adapter, stage = self.blocked_descent(gripper_normalized=1.0, jaw_min_normalized=0.15)
        _, receipt = self.run_batch(adapter, batch(self.jaw(adapter, "close", 0.0), stage))
        self.assertEqual(receipt["failed_stage_id"], "down")
        self.assertIs(receipt["in_hand_tracking_error"], True)
        # An empty close: the same failure reports no held object.
        adapter, stage = self.blocked_descent(gripper_normalized=1.0)
        _, receipt = self.run_batch(adapter, batch(self.jaw(adapter, "close", 0.0), stage))
        self.assertEqual(receipt["failed_stage_id"], "down")
        self.assertNotIn("in_hand_tracking_error", receipt)

    def test_single_move_request_mirrors_the_residual_too(self):
        adapter, stage = self.blocked_descent(gripper_normalized=1.0)
        action = dict(stage["action"], action_id="single")
        receipt = adapter.execute(action, lambda update: None, threading.Event())
        self.assertEqual((receipt["status"], receipt["reason"]), ("failed", "tracking_error"))
        self.assertAlmostEqual(receipt["endpoint_residual_m"]["dz_m"], 0.03, delta=0.004)
        self.assertNotIn("endpoint_residual_m", receipt["waypoint_results"][0])

    def test_condensed_history_keeps_the_residual_and_the_path_review(self):
        adapter, stage = self.blocked_descent(gripper_normalized=1.0, jaw_min_normalized=0.15)
        _, receipt = self.run_batch(adapter, batch(self.jaw(adapter, "close", 0.0), stage))
        for project in (lean_action_result, compact_action_result):
            kept = project(dict(receipt=receipt, request=dict(action_id="a1", based_on_observation_id="obs_1")))["receipt"]
            self.assertEqual(kept["endpoint_residual_m"], json.loads(json.dumps(
                {k: round(v, 6) if isinstance(v, float) else v for k, v in receipt["endpoint_residual_m"].items()}))
                if project is compact_action_result else receipt["endpoint_residual_m"])
            self.assertIs(kept["in_hand_tracking_error"], True)
            self.assertEqual(kept["contact_block_summary"].keys(), receipt["contact_block_summary"].keys())
        preview = dict(status="planned", plan_id="p2", planned_native_steps=40, stage_plans={"m": {"path": {
            "min_tcp_z_m": 0.03, "collision": {"fraction": 0.4}}}},
            path_check=dict(status="collisions", colliding_stage_ids=["m"], max_joint_step_rad=0.01, obstacle_ids=["o1"]))
        summary = lean_action_result(dict(receipt=preview))["receipt"]
        self.assertEqual(summary["path_check"]["status"], "collisions")
        self.assertEqual(summary["path_check"]["colliding_stage_ids"], ["m"])
        unavailable = lean_action_result(dict(receipt=dict(preview, path_check=dict(
            status="unavailable", reason="ik_service_unreachable", colliding_stage_ids=[]))))["receipt"]
        self.assertEqual(unavailable["path_check"]["status"], "unavailable")
        self.assertEqual(unavailable["path_check"]["reason"], "ik_service_unreachable")
        self.assertNotIn("path_check", lean_action_result(dict(receipt=dict(status="planned", plan_id="p3")))["receipt"])


# ---- schema and text ------------------------------------------------------------------------------------------------
class SchemaGatingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x5 = build(None)
        cls.plain = build(embodiment_profile(deployment(False)))
        cls.service = build(embodiment_profile(deployment(True)))

    def test_x5_carries_none_of_the_new_fields(self):
        properties = self.x5.tool["inputSchema"]["properties"]
        move = self.x5.stage_schemas["move"]["properties"]
        for field in mt.ARRIVAL_CORRECTION_FIELDS:
            self.assertNotIn(field, properties)
            self.assertNotIn(field, move)
        self.assertNotIn("path_check", self.x5.tool["description"])
        self.assertIn("held_objects", properties)  # the X5 proxy review is untouched

    def test_arrival_fields_come_with_the_tag_and_document_unit_default_and_receipt(self):
        for contract_ in (self.plain, self.service):
            move = contract_.stage_schemas["move"]["properties"]
            top = contract_.tool["inputSchema"]["properties"]
            self.assertEqual(move["arrival_correction_rounds"]["type"], "integer")
            # The maxima are this deployment's own bounds (example file: 3 rounds, 0.1 rad), the backend's refusal limit.
            self.assertEqual((move["arrival_correction_rounds"]["minimum"], move["arrival_correction_rounds"]["maximum"]),
                             (0, 3))
            self.assertEqual(move["arrival_correction_max_offset_rad"]["maximum"], 0.1)
            for field, maximum in zip(mt.ARRIVAL_CORRECTION_FIELDS, (3, 0.1)):
                self.assertEqual(top[field]["maximum"], maximum)
            for field in mt.ARRIVAL_CORRECTION_FIELDS:
                self.assertIn(field, top)
                self.assertNotIn(field, contract_.stage_schemas["gripper"]["properties"])
                for text in (move[field]["description"], top[field]["description"]):
                    self.assertTrue(text)
            rounds, offset = (move[field]["description"] for field in mt.ARRIVAL_CORRECTION_FIELDS)
            for needle in ("integer count, 0..3", "0 = no correction", "default when omitted 3",
                           "effective_tolerances.arrival_correction_rounds", "refused, never clamped", "Example:"):
                self.assertIn(needle, rounds)
            for needle in ("radians, in (0, 0.1]", "default when omitted 0.1",
                           "effective_tolerances.arrival_correction_max_offset_rad"):
                self.assertIn(needle, offset)

    def test_collision_world_fields_need_the_service(self):
        plain, service = (c.tool["inputSchema"]["properties"] for c in (self.plain, self.service))
        for field in ("obstacles", "held_object", "obstacle_object_ids"):
            self.assertNotIn(field, plain)
            self.assertIn(field, service)
        # No proxy review, registry or contact declarations on the SO-101.
        for field in ("held_objects", "obstacle_policy", "ignore_obstacle_ids", "contact_entity_id", "initial_contact",
                      "placement_contact", "target_exemption_radius_m"):
            self.assertNotIn(field, service)
        self.assertNotIn("arm", service["held_object"]["properties"])
        for needle in ("so101_base", "metres", "default none", "margin_m", "Advisory"):
            self.assertIn(needle, service["obstacles"]["description"])
        for needle in ("first stage of this batch that opens the jaw", "32 equal spheres", "Example:"):
            self.assertIn(needle, service["held_object"]["description"])
        for box in (service["obstacles"]["items"], service["held_object"]):
            for field in ("min_m", "max_m", "margin_m", "evidence"):
                self.assertTrue(box["properties"][field]["description"])
        self.assertIn("Supply obstacles", self.service.tool["description"])
        self.assertNotIn("Supply obstacles", self.plain.tool["description"])

    def test_execute_if_clean_names_the_path_check_only_with_the_service(self):
        self.assertIn("path_check (the joint path of every move)", self.service.tool["description"])
        self.assertIn("path_check.status is clear", self.service.action_field_help["auto_execute_if_clean"])
        self.assertNotIn("path_check", self.plain.tool["description"])

    def test_failure_text_names_the_receipt_level_fields(self):
        text = self.plain.stage_field_help["position_tolerance_m"]
        for needle in ("endpoint_residual_m", "contact_block_summary", "in_hand_tracking_error"):
            self.assertIn(needle, text)


class MotionTextTests(unittest.TestCase):
    def test_text_states_what_is_checked_on_each_deployment(self):
        plain = contract.actor_motion_instructions(deployment(False))
        service = contract.actor_motion_instructions(deployment(True))
        for text in (plain, service):
            # The arrival-correction bounds are explained once, in the field help (SchemaGatingTests); the motion
            # text points there and keeps the use of 0 rounds.
            for needle in ("straight line in JOINT space", "min_fingertip_z_m", "max_line_deviation_m",
                           "path_start_deviation_rad", "arrival_correction_rounds=0",
                           "their robot_action field help gives the limits and defaults"):
                self.assertIn(needle, text)
            self.assertNotIn("default and maximum", text)
        self.assertIn("Nothing checks the path for collision on this deployment.", plain)
        for needle in ("path_check", "obstacles", "held_object"):
            self.assertNotIn(needle, plain.replace("held object", ""))
        for needle in ("Path collision check (advisory)", "at most 0.01 rad", "path_check.status is clear, collisions or unavailable",
                       "obstacle_object_ids", "held_object", "ik.obstacle_free=false", "Nothing is blocked or re-routed"):
            self.assertIn(needle, service)

    def test_capabilities_collision_scope_matches(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        for ik_service, needle in ((False, "no object, table or self-collision checking"), (True, "Advisory")):
            deploy = deployment(ik_service)
            clock = FakeClock()
            adapter = SO101Adapter(deploy, directory.name, "rgb_proprio_v1", driver=FakeArmDriver(deploy, clock),
                                   cameras={name: FakeCameraSource(name, clock=clock, width=64, height=48)
                                            for name in ("fixed", "wrist")}, clock=clock,
                                   image_settings=Config().images.to_dict(), start_telemetry=False)
            self.addCleanup(adapter.close)
            self.assertIn(needle, adapter.capabilities()["collision_scope"])


# ---- review findings: capability text, reach tolerances, profile limits, jaw ramp segments, service self-check ------
class CapabilityTextTests(Case):
    """Finding 1: the worker's capability texts follow the profile tags and name the reviews as the schema does."""

    def test_pose_input_and_execute_if_clean_follow_the_profile(self):
        from ni_rua.robot.model.embodiments import preview_reviews
        for ik_service in (False, True):
            adapter, _, _ = self.adapter(ik_service=ik_service)
            capabilities = adapter.capabilities()
            conversion = capabilities["tool_geometry"]["conversion"]
            self.assertIn("grasp_centre_m + object_width_m", conversion)
            self.assertIn("stage_plans[<stage>].grasp_centre", conversion)
            self.assertNotIn("only pose input", conversion)
            gate = capabilities["unified_action"]["execute_if_clean"]
            description = build(adapter.embodiment).tool["description"]
            for name, scope in preview_reviews(adapter.embodiment):
                with self.subTest(ik_service=ik_service, review=name):
                    self.assertIn(f"({name}: {scope})", gate)
                    self.assertIn(f"{name} ({scope})", description)   # the robot_action text names the same scope
            self.assertIn("grasp_3d_review has no warnings/risk_flags", gate)
            self.assertIn("grasp_3d_review_warnings", gate)
            self.assertIn("closed_jaw_before_grasp:<arm>", gate)    # measured empty close in this deployment
            self.assertNotIn("grasp review", gate)                   # the old "produces no ... grasp review" is gone
            self.assertEqual("path_check.status is clear" in gate, ik_service)
            self.assertEqual("path_check_collisions, path_check_unavailable" in gate, ik_service)

    def test_a_profile_without_the_grasp_centre_input_keeps_the_tcp_only_text(self):
        adapter, _, _ = self.adapter(ik_service=False)
        profile = adapter.embodiment
        tcp_only = dataclasses.replace(profile, pose_inputs=frozenset({"tcp_quaternion"}),
                                       gripper=dataclasses.replace(profile.gripper, jaw_grasp=None))
        with patch.object(type(adapter), "embodiment", new_callable=PropertyMock, return_value=tcp_only):
            capabilities = adapter.capabilities()
        self.assertTrue(capabilities["tool_geometry"]["conversion"].endswith("this is the only pose input"))
        self.assertIn("produces no collision, motion-safety or grasp review",
                      capabilities["unified_action"]["execute_if_clean"])


class ReachToleranceTests(Case):
    def test_reach_reports_no_arrival_correction_bounds(self):
        """Finding 2: a reach query never moves, so the arrival-correction bounds are not among its tolerances."""
        adapter, _, _ = self.adapter(ik_service=False)
        action = batch(self.move(adapter, "a", dx=0.02), self.move(adapter, "b", dx=0.03, arrival_correction_rounds=0))
        answer = adapter.reach_batch(deepcopy(action))
        self.assertEqual(answer["status"], "solved", answer)
        for identifier in ("a", "b"):
            tolerances = answer["stage_results"][identifier]["tolerances"]
            for key in ("arrival_correction_rounds", "arrival_correction_max_offset_rad", "max_upward_dz_m",
                        "max_tracking_divergence_m", "stall_guard_preset"):
                self.assertNotIn(key, tolerances)
            self.assertEqual(tolerances["checked_on"], "FK of the planned joints (IK acceptance; nothing moves)")
        # A preview of the same moves still reports them (it executes).
        preview = adapter.preview_batch(deepcopy(action))
        self.assertEqual(preview["stage_plans"]["b"]["tolerances"]["arrival_correction_rounds"], 0)


class ArrivalCorrectionProfileTests(unittest.TestCase):
    """Finding 3: the deployment's bounds are the profile's, and the schema publishes exactly them."""

    def test_profile_limits_are_the_deployment_bounds_and_drive_the_schema(self):
        for rounds, offset in ((3, 0.1), (5, 0.2), (1, 0.03)):
            with self.subTest(rounds=rounds, offset=offset):
                deploy = deployment(False, **{"motion.arrival_correction_rounds": rounds,
                                              "motion.arrival_correction_max_offset_rad": offset})
                profile = embodiment_profile(deploy)
                self.assertEqual(profile.arrival_correction_limits, embodiments.ArrivalCorrectionLimits(rounds, offset))
                self.assertEqual(contract.arrival_correction_limits(deploy), (rounds, offset))
                contract_ = build(profile)
                for schema in (contract_.stage_schemas["move"]["properties"], contract_.tool["inputSchema"]["properties"]):
                    self.assertEqual(schema["arrival_correction_rounds"]["maximum"], rounds)
                    self.assertEqual(schema["arrival_correction_max_offset_rad"]["maximum"], offset)
                text = contract_.stage_field_help["arrival_correction_rounds"]
                self.assertIn(f"0..{rounds}; default when omitted {rounds}, this robot's limit", text)
                self.assertIn(f"(0, {offset:g}] (default when omitted {offset:g}",
                              contract_.stage_field_help["arrival_correction_max_offset_rad"])
                # The Harness resolver refuses above the same bound, naming it.
                with self.assertRaisesRegex(ValueError, rf"invalid_arrival_correction_rounds: must be an integer in \[0, {rounds}\]"):
                    mt.resolve_arrival_correction(dict(arrival_correction_rounds=rounds + 1), {}, profile=profile)
                with self.assertRaisesRegex(ValueError, "invalid_arrival_correction_max_offset_rad"):
                    mt.resolve_arrival_correction(dict(arrival_correction_max_offset_rad=offset + 0.01), {}, profile=profile)
                self.assertEqual(mt.resolve_arrival_correction(
                    dict(arrival_correction_rounds=rounds, arrival_correction_max_offset_rad=offset), {}, profile=profile),
                    dict(arrival_correction_rounds=rounds, arrival_correction_max_offset_rad=offset))

    def test_a_deployment_that_corrects_nothing_offers_no_fields(self):
        for updates in ({"motion.arrival_correction_rounds": 0}, {"motion.arrival_correction_max_offset_rad": 0.0}):
            with self.subTest(**updates):
                deploy = deployment(False, **updates)
                profile = embodiment_profile(deploy)
                self.assertNotIn("arrival_correction", profile.supports)
                self.assertIsNone(profile.arrival_correction_limits)
                contract_ = build(profile)
                for field in mt.ARRIVAL_CORRECTION_FIELDS:
                    self.assertNotIn(field, contract_.stage_schemas["move"]["properties"])
                    self.assertNotIn(field, contract_.tool["inputSchema"]["properties"])
                with self.assertRaisesRegex(ValueError, "unsupported_capability: arrival_correction_rounds .*applies no "
                                                        "arrival correction"):
                    contract._validate_path_fields(dict(kind="backend_free", arrival_correction_rounds=0), profile, deploy)

    def test_limits_are_validated_and_archived(self):
        profile = embodiment_profile(deployment(False))
        with self.assertRaisesRegex(ValueError, "invalid_embodiment_profile:arrival_correction_limits"):
            dataclasses.replace(profile, supports=profile.supports - {"arrival_correction"})
        for bad in ((0, 0.1), (11, 0.1), (3, 0.0), (3, 0.6), (3.0, 0.1)):
            with self.assertRaisesRegex(ValueError, "invalid_embodiment_profile:arrival_correction_limits"):
                dataclasses.replace(profile, arrival_correction_limits=embodiments.ArrivalCorrectionLimits(*bad))
        record = json.loads(json.dumps(embodiments.profile_record(profile)))
        self.assertEqual(record["profile"]["arrival_correction_limits"], dict(max_rounds=3, max_offset_rad=0.1))
        self.assertEqual(embodiments.profile_from_record(record), profile)
        self.assertNotIn("arrival_correction_limits",
                         embodiments.profile_record(embodiments.ARX_X5_PROFILE)["profile"])
        # An archive written before the limits existed: the tag without limits restores, and its schema falls back to
        # the shared maxima without claiming they are the default.
        old = deepcopy(record)
        del old["profile"]["arrival_correction_limits"]
        restored = embodiments.profile_from_record(old)
        self.assertIsNone(restored.arrival_correction_limits)
        self.assertIn("arrival_correction", restored.supports)
        self.assertEqual(mt.arrival_correction_bounds(restored),
                         (embodiments.MAX_ARRIVAL_CORRECTION_ROUNDS, embodiments.MAX_ARRIVAL_CORRECTION_OFFSET_RAD))
        text = build(restored).stage_field_help["arrival_correction_rounds"]
        self.assertIn("0..10; default when omitted this robot's limit", text)


class JawRampPathCheckTests(Case):
    """Finding 5: the checked jaw follows the streamed ramp (start_fraction..end_fraction), not the whole move."""

    def test_segments_cut_at_the_ramp_fractions(self):
        start, target = [0.0, -1.0, 1.0, 0.5, 0.0], [0.4, -0.6, 0.8, 0.9, 0.2]
        pieces = ik_plan.path_check_segments(start, target, 1.5, 0.1, True, (0.3, 0.7))
        self.assertEqual([(a, b) for a, b, _ in pieces], [(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)])
        first, ramp, last = (segment for *_, segment in pieces)
        self.assertEqual((first["start_jaw_rad"], first["end_jaw_rad"]), (1.5, 1.5))     # held before the ramp
        self.assertEqual((ramp["start_jaw_rad"], ramp["end_jaw_rad"]), (1.5, 0.1))
        self.assertEqual((last["start_jaw_rad"], last["end_jaw_rad"]), (0.1, 0.1))       # held after it
        np.testing.assert_allclose(first["end_q_rad"], np.add(start, 0.3 * np.subtract(target, start)))
        self.assertEqual(first["end_q_rad"], ramp["start_q_rad"])
        self.assertEqual(ramp["end_q_rad"], last["start_q_rad"])
        self.assertEqual(last["end_q_rad"], target)
        self.assertTrue(all(segment["attached"] for *_, segment in pieces))
        # A ramp over the whole move, or no ramp: one segment.
        self.assertEqual([(a, b) for a, b, _ in ik_plan.path_check_segments(start, target, 1.5, 0.1, False, (0.0, 1.0))],
                         [(0.0, 1.0)])
        (only,) = ik_plan.path_check_segments(start, target, 0.7, 0.7, False)
        self.assertEqual(only[2], dict(start_q_rad=start, end_q_rad=target, start_jaw_rad=0.7, end_jaw_rad=0.7,
                                       attached=False))
        # A ramp that ends at the target: two segments.
        self.assertEqual([(a, b) for a, b, _ in ik_plan.path_check_segments(start, target, 1.5, 0.1, False, (0.5, 1.0))],
                         [(0.0, 0.5), (0.5, 1.0)])

    def test_preview_checks_the_streamed_jaw_and_maps_the_collision_back_to_the_move(self):
        from ni_rua.robot.backends.so101.driver import gripper_rad_from_normalized
        adapter, _, service = self.adapter(gripper_normalized=1.0)
        start = adapter.read()
        pose = self.pose(adapter)
        leaf = contract.compile_action("move", dict(
            arm="right", observation_id="o", quaternion_xyzw=pose["quaternion_xyzw"],
            position_m=[pose["position_m"][0] + 0.04, *pose["position_m"][1:]], gripper_goal_normalized=0.2,
            gripper_motion=dict(start_fraction=0.3, end_fraction=0.7)), IDENTITY, adapter.capabilities())
        leaf["path"].update(position_tolerance_m=0.005, orientation_tolerance_rad=0.1)
        hit = dict(sample_index=5, fraction=0.5, q_rad=[0.0, -0.9, 0.9, 0.7, 0.0], jaw_rad=0.9,
                   world=[dict(link="moving_jaw_so101_v1_link", penetration_m=0.003, obstacles=["table"])], self=[])
        service.collisions = {1: hit}
        preview = adapter.preview_batch(batch(dict(id="m", kind="move", arm="right", action=leaf)))
        self.assertEqual(preview["status"], "planned", preview)
        (check,) = service.checks
        segments = check["segments"]
        self.assertEqual(len(segments), 3)
        open_rad, goal_rad = start["gripper_rad"], gripper_rad_from_normalized(adapter.deployment, 0.2)
        self.assertAlmostEqual(segments[0]["start_jaw_rad"], open_rad)
        self.assertAlmostEqual(segments[0]["end_jaw_rad"], open_rad)
        self.assertAlmostEqual(segments[2]["start_jaw_rad"], goal_rad)
        self.assertAlmostEqual(segments[2]["end_jaw_rad"], goal_rad)
        # Every streamed command lies on the checked polyline: arm and jaw of the schedule at its command progress.
        q0, target = np.asarray(start["q_rad"]), np.asarray(preview["stage_plans"]["m"]["q_rad"])
        schedule = adapter._jaw_schedule(ik_plan.joint_interpolation(q0, target, ik_plan.interpolation_steps(
            q0, target, adapter.deployment["motion"]["max_joint_speed_rad_s"] * adapter._cap(),
            adapter.deployment["motion"]["command_rate_hz"])), q0, adapter.jaw_goal, leaf["gripper_events"][0], 1.0, 10_000)
        rows, jaws = schedule[0], schedule[1]
        for index in range(len(rows)):
            progress = (index + 1) / len(rows)
            low, high, segment = next(item for item in zip((0.0, 0.3, 0.7), (0.3, 0.7, 1.0), segments)
                                      if item[0] <= progress <= item[1])
            share = (progress - low) / (high - low)
            np.testing.assert_allclose(rows[index], np.add(segment["start_q_rad"], share * np.subtract(
                segment["end_q_rad"], segment["start_q_rad"])), atol=1e-9)
            self.assertAlmostEqual(gripper_rad_from_normalized(adapter.deployment, float(jaws[index])),
                                   segment["start_jaw_rad"] + share * (segment["end_jaw_rad"] - segment["start_jaw_rad"]),
                                   places=6)
        # The collision in the middle segment is reported at its fraction of the whole move, samples summed.
        collision = preview["stage_plans"]["m"]["path"]["collision"]
        self.assertEqual((collision["fraction"], collision["samples"], collision["colliding_samples"]), (0.5, 30, 3))
        self.assertEqual(preview["path_check"]["colliding_stage_ids"], ["m"])


class ServiceSelfCheckTests(unittest.TestCase):
    """Finding 6: the startup self-check also checks the ready pose with the jaw at the deployment's open angle."""

    class Solver:
        info = dict(joint_limits_rad={name: [-1.0, 1.0] for name in ARM_JOINTS})

        def __init__(self, success=True, colliding_jaws=()):
            self.success, self.colliding_jaws, self.requests = success, set(colliding_jaws), []

        def solve(self, request):
            self.requests.append(("ik", request))
            return dict(solutions=[[dict(q_rad=[0.0, -0.9, 0.9, 1.0, 0.0], success=self.success)]])

        def path_check(self, request):
            self.requests.append(("path_check", request))
            return dict(segments=[dict(samples=1, colliding_samples=int(segment["start_jaw_rad"] in self.colliding_jaws),
                                       first_collision=(dict(jaw_rad=segment["start_jaw_rad"])
                                                        if segment["start_jaw_rad"] in self.colliding_jaws else None))
                                  for segment in request["segments"]])

    def test_open_jaw_collision_fails_the_self_check(self):
        service = service_module()
        clean = self.Solver()
        self.assertIsNone(service.self_check_failure(clean, 1.6))
        (_, ik), (_, probe) = clean.requests
        self.assertEqual(ik["poses"], [service.READY_POSE])
        self.assertEqual([(s["start_jaw_rad"], s["end_jaw_rad"]) for s in probe["segments"]], [(0.0, 0.0), (1.6, 1.6)])
        self.assertTrue(all(s["start_q_rad"] == s["end_q_rad"] == [0.0, -0.9, 0.9, 1.0, 0.0] for s in probe["segments"]))
        failure = service.self_check_failure(self.Solver(colliding_jaws={1.6}), 1.6)
        self.assertIn("with the jaw at 1.6 rad", failure)
        self.assertIn("with the jaw at 0 rad", service.self_check_failure(self.Solver(colliding_jaws={0.0}), 1.6))
        unsolved = self.Solver(success=False)
        self.assertIn("not solvable", service.self_check_failure(unsolved, 1.6))
        self.assertEqual([kind for kind, _ in unsolved.requests], ["ik"])
        self.assertEqual(service.DEFAULT_OPEN_JAW_RAD, json.loads(EXAMPLE.read_text())["gripper"]["jaw_open_rad"])


if __name__ == "__main__":
    unittest.main()
