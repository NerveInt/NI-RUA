"""features.object_table_observe / scene3d_observe: 3D scene information attached to robot_observe replies and to the
final observation of robot_action / robot_wait replies (text only there unless obs3d_action_overlay)
(fake VGGT service: up, down, error, timeout)."""
NI_RUA_TEST_TIER = "component"

import base64
import hashlib
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from ni_rua.core.config import Config
from ni_rua.core.contracts import canonical
from ni_rua.perception.scene.store import SceneStore
from ni_rua.core.storage.store import Store
from ni_rua.core.tool_output import StructuredText
from ni_rua.runtime.tools import handlers
from ni_rua.runtime.tools.builtins import create_registry
from ni_rua.runtime.tools.definitions import public_definitions
from ni_rua.perception.external import obs3d, scene3d
from ni_rua.perception.external import scene3d_helpers as s3
from support.geometry_fixture import IDENTITY
from support.perception_fixtures import fake_object, two_camera_observation
from perception.external.test_scene3d import TABLE, synthetic_cloud


class FakeService:
    """/object_table and /points of the VGGT service; status != 'ok' answers an error, delay_s waits on stop event."""
    def __init__(self, status="ok", delay_s=0.0):
        self.requests, self.windows, owner = [], [], self
        self._stop_event = threading.Event()
        objects = [fake_object((0.0, 0.0, 0.05), 0.10, 0.0),
                   fake_object((0.1, 0.05, 0.12), 0.14, 0.10, support=0, flags=("contact_mismatch",), label=(370, 215))]
        cloud = synthetic_cloud()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append((self.path, body))
                started = time.monotonic()
                if delay_s:
                    owner._stop_event.wait(delay_s)
                owner.windows.append((started, time.monotonic()))
                if status != "ok":
                    answer, code = dict(status="error", error=status), 422
                elif self.path == "/object_table":
                    answer, code = dict(status="ok", version="fake-ot", objects=objects[:body["max_objects"]],
                                        requested=[], omitted=max(0, len(objects) - body["max_objects"]),
                                        support_z_m=body["support_z_m"], timing=dict(total_s=0.01)), 200
                else:
                    answer, code = dict(status="ok", version="fake-points", n_points=3000, n_above_support=600, n_robot=25,
                                        voxel_m=body["voxel_m"], support_z_m=body["support_z_m"],
                                        bbox_m=[[-0.3, -0.3, 0.765], [0.3, 0.3, 1.055]],
                                        bbox_above_support_m=[[-0.12, -0.02, 0.77], [0.13, 0.12, 0.865]],
                                        npz_b64=base64.b64encode(cloud).decode(), timing=dict(total_s=0.01), cache_hit=True), 200
                raw = json.dumps(answer).encode()
                try:
                    self.send_response(code)
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                except OSError:
                    pass  # the client gave up (timeout test)

        self.cloud = cloud
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def close(self):
        self._stop_event.set()
        self.server.shutdown()
        self.server.server_close()


def image_shape(part):
    data = base64.b64decode(part["imageUrl"].split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR).shape


class Obs3dCheck(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = self.enterContext(tempfile.TemporaryDirectory())
        self.artifacts = Store(self.tmp_dir, IDENTITY); self.addCleanup(self.artifacts.close)
        self.scene = SceneStore(Path(self.tmp_dir) / "scene", self.artifacts.resolve_artifact, self.artifacts.put_artifact)
        self.bundle, self.images = two_camera_observation(self.artifacts)
        self.scene.import_observation(self.bundle)

    def runtime(self, ot=False, s3o=False, endpoint="http://127.0.0.1:9", timeout=5.0, rows=10, bundle=None,
                action_overlay=False, truth_log=False):
        config = Config().with_overrides({"features": {
            "tool_catalog_profile": "full_v1",   # the off-switch tests compare with create_registry(): the prose catalogue
            "external_perception": False,
            "object_table_observe": ot, "scene3d_observe": s3o, "vggt_endpoint": endpoint, "vggt_support_z_m": TABLE,
            "vggt_timeout_s": 60.0, "obs3d_timeout_s": timeout, "obs3d_max_objects": rows,
            "obs3d_action_overlay": action_overlay, "trajectory_preview": False},
            "analysis": {"scene_truth_log": truth_log}})
        bundle = self.bundle if bundle is None else bundle
        base = [StructuredText({"bundle": {"observation_id": bundle["observation_id"]}, "robot_state": {}}, "data"),
                *[dict(type="inputImage", imageUrl="data:image/png;base64," + base64.b64encode(self.images[c["camera_id"]]).decode())
                  for c in bundle["cameras"]]]
        return SimpleNamespace(config=config, scene=self.scene, image_settings=None, identity=dict(IDENTITY),
                               latest=None, active_action=None,  # main's robot_observe single-camera view path
                               store=self.artifacts, features=config.features, observe=lambda correlation: bundle,
                               observation_content=lambda *a, **k: list(base))

    def catalog_runtime(self, **kw):
        runtime = self.runtime(**kw)
        return SimpleNamespace(disabled_tools=(), observation_profile="rgb_proprio_calibrated_v1", motion_review=True,
                               enable_wait=True, vision_compare=True, visual_reviewer=True, action_policy=None,
                               capabilities=None, programs=True, config=runtime.config, features=runtime.config.features)

    def observe(self, runtime, correlation=None):
        return handlers.robot_observe(runtime, {}, correlation or {})

    def attached(self, response):
        extra = response.content[1 + len(self.bundle["cameras"]):]
        self.assertIsInstance(extra[0], StructuredText)
        return extra[0].value, extra[1:]

    # ------------------------------------------------------------------ switches off
    def test_off_reply_and_catalog_unchanged(self):
        runtime = self.runtime()
        response = self.observe(runtime)
        self.assertEqual(len(response.content), 1 + len(self.bundle["cameras"]))
        self.assertEqual(obs3d.flags(runtime), (False, False))
        self.assertEqual(canonical(public_definitions(observe3d=(False, False))), canonical(public_definitions()))
        digest = lambda rt: hashlib.sha256(canonical(create_registry(runtime=rt).catalog(rt, role="actor")).encode()).hexdigest()
        self.assertEqual(public_definitions(observe3d=(False, False, True)), public_definitions())
        self.assertEqual(digest(self.catalog_runtime()), hashlib.sha256(canonical(
            create_registry().catalog(self.catalog_runtime(), role="actor")).encode()).hexdigest())
        self.assertEqual(scene3d.workspace_modules(runtime), ())

    def test_catalog_changes_only_robot_observe_description(self):
        off = {d["name"]: d for d in public_definitions()}
        for flags in ((True, False), (False, True), (True, True), (True, False, True), (True, True, True)):
            on = {d["name"]: d for d in public_definitions(observe3d=flags)}
            for name in off:
                if name != "robot_observe":
                    self.assertEqual(canonical(on[name]), canonical(off[name]), name)
            added = obs3d.catalogue_sentence(*flags)
            self.assertEqual(on["robot_observe"]["description"].count(added), 1)
            self.assertEqual(on["robot_observe"]["description"].replace(added, "", 1), off["robot_observe"]["description"])
            self.assertLessEqual(len(added), 300)
            self.assertLessEqual(len(canonical(on["robot_observe"])) - len(canonical(off["robot_observe"])), 300)
            self.assertEqual(canonical(on["robot_observe"]["inputSchema"]), canonical(off["robot_observe"]["inputSchema"]))
        rt = self.catalog_runtime(ot=True, s3o=True)
        text = canonical(create_registry(runtime=rt).catalog(rt, role="actor"))
        self.assertIn(obs3d.catalogue_sentence(True, True), text)
        self.assertIn("robot_action / robot_wait", obs3d.catalogue_sentence(True, True))
        self.assertIn("robot_observe adds an id overlay", obs3d.catalogue_sentence(True, False))
        self.assertNotIn("overlay", obs3d.catalogue_sentence(False, True))
        self.assertNotIn("shape_tag", obs3d.catalogue_sentence(True, True, True))
        rt = self.catalog_runtime(ot=True, action_overlay=True)
        self.assertIn(obs3d.catalogue_sentence(True, False, True),
                      canonical(create_registry(runtime=rt).catalog(rt, role="actor")))

    # ------------------------------------------------------------------ service up
    def test_object_table_attached_with_one_head_overlay(self):
        fake = FakeService(); self.addCleanup(fake.close)
        runtime = self.runtime(ot=True, endpoint=fake.url)
        value, images = self.attached(self.observe(runtime))
        self.assertEqual(value["observation_id"], self.bundle["observation_id"])
        table = value["object_table"]
        self.assertEqual(table["columns"][0], "id")
        self.assertNotIn("label", table["columns"])
        self.assertEqual([row[0] for row in table["rows"]], ["o1", "o2"])
        self.assertEqual(table["rows"][1][-3:], ["o1", "low:contact_mismatch", [370, 215]])
        self.assertEqual(table["table_z"], TABLE)
        self.assertNotIn("scene3d", value)
        self.assertEqual(len(images), 1)
        frame = cv2.imdecode(np.frombuffer(self.images["cam_head"], np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image_shape(images[0])[:2], frame.shape[:2])
        self.assertEqual([path for path, _ in fake.requests], ["/object_table"])
        sent = fake.requests[0][1]
        self.assertEqual(base64.b64decode(sent["image_b64"]), self.images["cam_head"])
        self.assertEqual(sent["max_objects"], 10)
        self.assertNotIn("labels", sent); self.assertNotIn("pixels", sent)
        self.assertLess(len(canonical(value)), 1500)

    def test_row_limit(self):
        fake = FakeService(); self.addCleanup(fake.close)
        value, _ = self.attached(self.observe(self.runtime(ot=True, endpoint=fake.url, rows=1)))
        self.assertEqual((len(value["object_table"]["rows"]), value["object_table"]["omitted"]), (1, 1))

    def test_scene3d_attached_without_image(self):
        fake = FakeService(); self.addCleanup(fake.close)
        runtime = self.runtime(s3o=True, endpoint=fake.url)
        self.assertEqual(scene3d.workspace_modules(runtime), ("scene3d_helpers.py",))
        value, images = self.attached(self.observe(runtime))
        self.assertEqual(images, [])
        cloud = value["scene3d"]
        self.assertEqual(set(cloud), {"artifact", "points", "table_z", "use"})
        self.assertEqual((cloud["points"], cloud["table_z"]), (3000, TABLE))
        info, data = self.artifacts.resolve_artifact(cloud["artifact"])
        self.assertEqual((info["kind"], data), ("scene3d_points", fake.cloud))
        self.assertIn("s3.API", cloud["use"])
        self.assertNotIn("object_table", value)
        self.assertEqual([path for path, _ in fake.requests], ["/points"])
        self.assertLess(len(canonical(value)), 400)

    def test_operator_records_each_component_without_actor_diagnostics(self):
        fake = FakeService(); self.addCleanup(fake.close)
        from ni_rua.core.presentation.actor_output import project_metadata
        for ot, cloud in [(True, False), (False, True), (True, True)]:
            with self.subTest(object_table=ot, scene3d=cloud):
                runtime = self.runtime(ot=ot, s3o=cloud, endpoint=fake.url)
                parts = obs3d.attachments(runtime, self.bundle, {}, reply="action")
                self.assertEqual(len(parts), 1, "operator overlay must not add actor action images")
                record = self.artifacts.read_state("perception", self.bundle["observation_id"])
                self.assertEqual(isinstance(record["table"], dict), ot)
                self.assertEqual(isinstance(record["scene3d"], dict), cloud)
                self.assertEqual(bool(record["overlay_ref"]), ot)
                self.assertEqual(record["diagnostics"]["object_table"]["status"], "conditional" if ot else "disabled")
                self.assertEqual(record["diagnostics"]["scene3d"]["status"], "conditional" if cloud else "disabled")
                self.assertNotIn("perception", self.artifacts.observation_view(self.bundle))
                if ot:
                    raw = obs3d._memo(runtime, self.bundle["observation_id"])["object_table"][2]
                    self.assertIn("bbox_px", raw["diagnostics"]["objects"][0])
                    projected = project_metadata(raw, "geometry_result")
                    self.assertNotIn("diagnostics", projected)
                    self.assertEqual(projected["value"], raw["value"])

    def test_operator_overlay_storage_failure_preserves_estimates(self):
        from unittest.mock import patch
        fake = FakeService(); self.addCleanup(fake.close)
        runtime = self.runtime(ot=True, endpoint=fake.url)
        with patch.object(self.artifacts, "put_artifact", side_effect=OSError("fixture disk unavailable")):
            parts = obs3d.attachments(runtime, self.bundle, {}, reply="action")
        self.assertIsInstance(parts[0].value["object_table"], dict)
        record = self.artifacts.read_state("perception", self.bundle["observation_id"])
        self.assertIsNone(record["overlay_ref"])
        self.assertIn("OSError", record["diagnostics"]["object_table"]["overlay_error"])

    def test_operator_failure_is_recorded_without_inventing_objects(self):
        runtime = self.runtime(ot=True, s3o=False)
        obs3d.attachments(runtime, self.bundle, {}, reply="action")
        record = self.artifacts.read_state("perception", self.bundle["observation_id"])
        self.assertIsNone(record["table"])
        self.assertIsNone(record["overlay_ref"])
        self.assertEqual(record["objects"], [])
        self.assertNotEqual(record["diagnostics"]["object_table"]["status"], "conditional")
        self.assertTrue(record["diagnostics"]["object_table"]["reason"])

    def test_both(self):
        fake = FakeService(); self.addCleanup(fake.close)
        value, images = self.attached(self.observe(self.runtime(ot=True, s3o=True, endpoint=fake.url)))
        self.assertEqual(len(images), 1)
        self.assertIsInstance(value["object_table"], dict)
        self.assertIsInstance(value["scene3d"], dict)
        self.assertCountEqual([path for path, _ in fake.requests], ["/object_table", "/points"])
        bodies = [json.loads(body) for body, in self.artifacts.db.execute("SELECT body FROM events")]
        attached = [b for b in bodies if "geometry.obs3d_attached" in json.dumps(b)]
        self.assertEqual(len(attached), 1)
        self.assertIn("latency_s", json.dumps(attached[0]))

    # ------------------------------------------------------------------ service down / error / timeout
    def test_service_down(self):
        response = self.observe(self.runtime(ot=True, s3o=True))
        value, images = self.attached(response)
        self.assertEqual(value["object_table"], "unavailable: service_down")
        self.assertEqual(value["scene3d"], "unavailable: service_down")
        self.assertEqual(images, [])
        self.assertEqual(response.data, self.bundle)

    def test_service_error(self):
        fake = FakeService(status="no_horizontal_support_plane"); self.addCleanup(fake.close)
        value, images = self.attached(self.observe(self.runtime(ot=True, s3o=True, endpoint=fake.url)))
        self.assertEqual(value["object_table"], "unavailable: no_horizontal_support_plane")
        self.assertEqual(value["scene3d"], "unavailable: no_horizontal_support_plane")
        self.assertEqual(images, [])

    def test_service_timeout(self):
        fake = FakeService(delay_s=2.5); self.addCleanup(fake.close)
        started = time.monotonic()
        value, images = self.attached(self.observe(self.runtime(ot=True, endpoint=fake.url, timeout=1.0)))
        self.assertLess(time.monotonic() - started, 2.4)
        self.assertEqual(value["object_table"], "unavailable: timeout")
        self.assertEqual(images, [])

    # ------------------------------------------------------------------ scope
    def test_program_callers_and_no_head_camera_get_nothing(self):
        fake = FakeService(); self.addCleanup(fake.close)
        # program_run 子调用现开放无叠加图的 3D 结构化事实文本（1 个文本块，0 张叠加图）
        response = self.observe(self.runtime(ot=True, s3o=True, endpoint=fake.url), {"program_id": "job_1"})
        value, images = self.attached(response)
        self.assertEqual([row[0] for row in value["object_table"]["rows"]], ["o1", "o2"])
        self.assertEqual(value["scene3d"]["points"], 3000)
        self.assertEqual(images, [])
        fake.requests.clear()
        side_only = dict(self.bundle, cameras=[c for c in self.bundle["cameras"] if c["camera_id"] != "cam_head"])
        response = self.observe(self.runtime(ot=True, s3o=True, endpoint=fake.url, bundle=side_only))
        self.assertEqual(len(response.content), 2)
        self.assertEqual(fake.requests, [])

    def test_helper_api_matches_tool_api(self):
        self.assertEqual(list(s3.API), list(scene3d.API))

    def test_request_projection_treats_parts_like_other_output(self):
        from ni_rua.core.presentation.actor_output import ActorOutput
        from ni_rua.core.presentation.actor_presentation import ActorPresentation
        from ni_rua.cortex.session.wire.openai_responses import present_responses_request
        fake = FakeService(); self.addCleanup(fake.close)
        response = self.observe(self.runtime(ot=True, s3o=True, endpoint=fake.url))
        presentation = ActorPresentation(self.artifacts)
        shown = ActorOutput(self.artifacts, presentation, payload_profile="lean_v2", compact=True).present(response, tool="robot_observe")
        items = shown["contentItems"]
        text = json.loads(items[1 + len(self.bundle["cameras"])]["text"])
        self.assertTrue(text["scene3d"]["artifact"].startswith("artifact:ref_"))
        output = [dict(type="input_text", text=i["text"]) if i["type"] == "inputText" else
                  dict(type="input_image", image_url=i["imageUrl"]) for i in items]
        scrubbed = present_responses_request(
            {"input": [dict(type="function_call_output", call_id="c", output=output)]},
            presentation,
            deferred_tools=frozenset(),
            namespace="ni_rua",
        )
        parts = scrubbed["input"][0]["output"]
        self.assertEqual(sum(p["type"] == "input_image" for p in parts), len(self.bundle["cameras"]) + 1)
        obs3d_text = json.loads(parts[1 + len(self.bundle["cameras"])]["text"])
        self.assertTrue(obs3d_text["scene3d"]["artifact"].startswith("artifact:ref_"))


    # ------------------------------------------------------------------ robot_action / robot_wait replies
    def action_runtime(self, **kw):
        runtime = self.runtime(**kw)
        self.artifacts.save("observation", self.bundle["observation_id"], self.bundle)
        runtime.action_budget = lambda: {"native_steps_remaining": 200}
        runtime.depth_auto_feedback, runtime.actor_payload_profile = False, "full_v1"
        runtime._scheduling_advisory = lambda *a, **k: None
        return runtime

    def action_reply(self, runtime, correlation=None):
        from ni_rua.runtime.actions import unified_actions
        receipt = {"status": "completed", "executed_native_steps": 40, "result_observation_id": self.bundle["observation_id"]}
        record = {"summary": {"planned_native_steps": 40, "next_action": {"mode": "inspect", "plan_id": "plan_1"}}}
        return unified_actions._action_response(runtime, {"plan_id": "plan_1", "action_id": "act_1", "receipt": receipt},
                                                {"type": "ArmQueueAction", "action_id": "act_1"}, record, correlation or {})

    def wait_reply(self, runtime, correlation=None, obs3d_on=True):
        from ni_rua.runtime.tools import actions
        result = {"request": {"type": "WaitAction", "action_id": "act_w"},
                  "receipt": {"status": "completed", "executed_native_steps": 10,
                              "result_observation_id": self.bundle["observation_id"]}}
        return actions._action_response(runtime, result, correlation or {}, obs3d=obs3d_on)

    @staticmethod
    def obs3d_parts(response):
        """(obs3d text values, images after the first obs3d part, all images) of one reply."""
        content = response.content
        index = [k for k, part in enumerate(content) if isinstance(part, StructuredText) and isinstance(part.value, dict)
                 and set(part.value) & {"object_table", "scene3d"}]
        images = [part for part in content if isinstance(part, dict) and part.get("type") == "inputImage"]
        after = ([part for part in content[index[0]:] if isinstance(part, dict) and part.get("type") == "inputImage"]
                 if index else [])
        return [content[k].value for k in index], after, images

    def test_action_and_wait_replies_carry_text_without_image(self):
        for reply in (self.action_reply, self.wait_reply):
            fake = FakeService(); self.addCleanup(fake.close)
            runtime = self.action_runtime(ot=True, s3o=True, endpoint=fake.url)
            response = reply(runtime)
            values, after, images = self.obs3d_parts(response)
            self.assertEqual(len(values), 1, reply.__name__)
            self.assertEqual([row[0] for row in values[0]["object_table"]["rows"]], ["o1", "o2"])
            self.assertEqual(values[0]["scene3d"]["points"], 3000)
            self.assertEqual(after, [])
            self.assertEqual(len(images), len(self.bundle["cameras"]))  # the observation's own frames only
            self.assertEqual(response.content[-1].value, values[0])
            self.assertEqual(response.data["receipt"]["result_observation_id"], self.bundle["observation_id"])
            self.assertCountEqual([path for path, _ in fake.requests], ["/object_table", "/points"])
            bodies = [json.loads(body) for body, in self.artifacts.db.execute("SELECT body FROM events")]
            self.assertIn('"reply": "action"', json.dumps([b for b in bodies if "obs3d_attached" in json.dumps(b)][-1]))

    # ------------------------------------------------------------------ analysis.scene_truth_log: offline-scoring event
    def analysis_events(self):
        return [e for e in self.artifacts.events_by_type([obs3d.ANALYSIS_EVENT], limit=100)]

    def test_analysis_event_only_with_truth_log_and_reply_unchanged(self):
        fake = FakeService(); self.addCleanup(fake.close)
        off = self.action_runtime(ot=True, endpoint=fake.url)
        off_values = [self.obs3d_parts(self.action_reply(off))[0], self.obs3d_parts(self.observe(off))[0]]
        self.assertEqual(self.analysis_events(), [])
        on = self.action_runtime(ot=True, endpoint=fake.url, truth_log=True)
        action_reply, observe_reply = self.action_reply(on), self.observe(on)
        on_values = [self.obs3d_parts(action_reply)[0], self.obs3d_parts(observe_reply)[0]]
        self.assertEqual(canonical(on_values), canonical(off_values))           # model-facing parts identical
        self.assertNotIn(obs3d.ANALYSIS_EVENT, canonical([p.value if isinstance(p, StructuredText) else p
                                                          for r in (action_reply, observe_reply) for p in r.content]))
        events = self.analysis_events()
        self.assertEqual([(e["payload"]["reply"], e["payload"]["reused"]) for e in events], [("action", False), ("observe", True)])
        head = next(c["frame_id"] for c in self.bundle["cameras"] if c["camera_id"] == "cam_head")
        for event, value in zip(events, on_values):
            payload = event["payload"]
            self.assertEqual(event["world_kind"], "internal")
            self.assertEqual((payload["observation_id"], payload["frame_ids"], payload["available"]),
                             (self.bundle["observation_id"], [head], True))
            self.assertEqual(canonical([payload["columns"], payload["rows"], payload["table_z"]]),
                             canonical([value[0]["object_table"][k] for k in ("columns", "rows", "table_z")]))
        self.assertEqual(len([p for p, _ in fake.requests if p == "/object_table"]), 2)   # one per runtime (memo reuse)

    def test_analysis_event_records_unavailable_table(self):
        self.observe(self.runtime(ot=True, truth_log=True))
        (event,) = self.analysis_events()
        self.assertEqual((event["payload"]["available"], event["payload"]["unavailable"]), (False, "unavailable: service_down"))
        self.assertNotIn("rows", event["payload"])

    def test_observe_reply_has_text_and_overlay_and_reuses_action_result(self):
        fake = FakeService(); self.addCleanup(fake.close)
        runtime = self.action_runtime(ot=True, s3o=True, endpoint=fake.url)
        action_values, action_after, _ = self.obs3d_parts(self.action_reply(runtime))
        observe_values, observe_after, _ = self.obs3d_parts(self.observe(runtime))
        self.assertEqual((len(action_after), len(observe_after)), (0, 1))
        notes = [values[0]["object_table"]["note"] for values in (action_values, observe_values)]
        # Only the reply that carries the id overlay image says the ids are drawn on it.
        self.assertEqual(notes, [obs3d.object_table_note(None, "cam_head", overlay=overlay) for overlay in (False, True)])
        self.assertNotEqual(*notes)
        without_note = lambda values: [dict(value, object_table={k: v for k, v in value["object_table"].items() if k != "note"})
                                       for value in values]
        self.assertEqual(without_note(observe_values), without_note(action_values))  # same rows, same artifact handle
        self.assertCountEqual([path for path, _ in fake.requests], ["/object_table", "/points"])  # one call per kind

    def test_object_table_note_follows_the_delivered_overlay(self):
        """The note's overlay clause is present exactly when this reply carries the overlay image: not for a default
        action reply, a program caller or a failed image encoding, and a reply with the image does not change the
        note of a later reply without it (the table is memoised per observation)."""
        from unittest.mock import patch
        fake = FakeService(); self.addCleanup(fake.close)

        def delivered(parts):
            images = [part for part in parts[1:] if isinstance(part, dict) and part.get("type") == "inputImage"]
            note = parts[0].value["object_table"]["note"]
            self.assertEqual(note, obs3d.object_table_note(None, "cam_head", overlay=bool(images)))
            return len(images)

        runtime = self.runtime(ot=True, endpoint=fake.url)
        self.assertEqual(delivered(obs3d.attachments(runtime, self.bundle, {}, reply="observe")), 1)
        self.assertEqual(delivered(obs3d.attachments(runtime, self.bundle, {}, reply="action")), 0)
        self.assertEqual(delivered(obs3d.attachments(runtime, self.bundle, {"program_id": "job_1"}, reply="observe")), 0)
        with patch("ni_rua.core.storage.images.image_url", side_effect=ValueError("fixture encoder failure")):
            self.assertEqual(delivered(obs3d.attachments(runtime, self.bundle, {}, reply="observe")), 0)
        self.assertEqual(delivered(obs3d.attachments(runtime, self.bundle, {}, reply="observe")), 1)
        with_knob = self.runtime(ot=True, endpoint=fake.url, action_overlay=True)
        self.assertEqual(delivered(obs3d.attachments(with_knob, self.bundle, {}, reply="action")), 1)

    def test_action_overlay_knob(self):
        fake = FakeService(); self.addCleanup(fake.close)
        runtime = self.action_runtime(ot=True, endpoint=fake.url, action_overlay=True)
        _, after, _ = self.obs3d_parts(self.action_reply(runtime))
        self.assertEqual(len(after), 1)
        frame = cv2.imdecode(np.frombuffer(self.images["cam_head"], np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image_shape(after[0])[:2], frame.shape[:2])

    def test_action_reply_off_service_down_and_program(self):
        base = len(self.action_reply(self.action_runtime()).content)
        self.assertEqual(self.obs3d_parts(self.action_reply(self.action_runtime()))[0], [])
        response = self.action_reply(self.action_runtime(ot=True, s3o=True))
        values, after, _ = self.obs3d_parts(response)
        self.assertEqual((values[0]["object_table"], values[0]["scene3d"]), ("unavailable: service_down",) * 2)
        self.assertEqual((len(response.content), after), (base + 1, []))
        self.assertEqual(response.data["receipt"]["status"], "completed")
        fake = FakeService(); self.addCleanup(fake.close)
        runtime = self.action_runtime(ot=True, s3o=True, endpoint=fake.url)
        # program_run 动作与等待回执同样附带纯文本 3D 事实且不生成叠加图，同帧共享缓存不重复请求
        act_vals, act_after, _ = self.obs3d_parts(self.action_reply(runtime, {"program_id": "job_1"}))
        wait_vals, wait_after, _ = self.obs3d_parts(self.wait_reply(runtime, {"program_id": "job_1"}))
        self.assertEqual((len(act_vals), act_after), (1, []))
        self.assertEqual((len(wait_vals), wait_after), (1, []))
        self.assertEqual(self.obs3d_parts(self.wait_reply(runtime, obs3d_on=False))[0], [])  # other execute_action tools
        self.assertCountEqual([path for path, _ in fake.requests], ["/object_table", "/points"])

    def test_roi_view_gets_nothing(self):
        fake = FakeService(); self.addCleanup(fake.close)
        response = handlers.robot_observe(self.runtime(ot=True, s3o=True, endpoint=fake.url),
                                          {"camera_id": "cam_head", "roi": [0, 0, 100, 100], "scale": 2}, {})
        self.assertEqual(len(response.content), 1 + len(self.bundle["cameras"]))
        self.assertEqual(fake.requests, [])

    def test_request_projection_of_action_reply(self):
        from ni_rua.core.presentation.actor_output import ActorOutput
        from ni_rua.core.presentation.actor_presentation import ActorPresentation
        from ni_rua.cortex.session.wire.openai_responses import present_responses_request
        fake = FakeService(); self.addCleanup(fake.close)
        response = self.action_reply(self.action_runtime(ot=True, s3o=True, endpoint=fake.url))
        presentation = ActorPresentation(self.artifacts)
        shown = ActorOutput(self.artifacts, presentation, payload_profile="lean_v2", compact=True).present(response, tool="robot_action")
        items = shown["contentItems"]
        self.assertEqual(sum(i["type"] == "inputImage" for i in items), len(self.bundle["cameras"]))
        full = json.loads(items[-1]["text"])
        self.assertEqual(full["object_table"]["rows"][0][0], "o1")
        output = [dict(type="input_text", text=i["text"]) if i["type"] == "inputText" else
                  dict(type="input_image", image_url=i["imageUrl"]) for i in items]
        scrubbed = present_responses_request(
            {"input": [dict(type="function_call_output", call_id="c", output=output)]},
            presentation,
            deferred_tools=frozenset(),
            namespace="ni_rua",
        )
        parts = scrubbed["input"][0]["output"]
        self.assertEqual(sum(p["type"] == "input_image" for p in parts), len(self.bundle["cameras"]))
        scrubbed_obs3d = json.loads(parts[-1]["text"])
        self.assertEqual([row[0] for row in scrubbed_obs3d["object_table"]["rows"]], ["o1", "o2"])
        self.assertTrue(scrubbed_obs3d["scene3d"]["artifact"].startswith("artifact:ref_"))

    def test_program_final_observation_attaches_obs3d_and_action_evidence_parity(self):
        from ni_rua.runtime.programs.program_review import action_evidence
        from ni_rua.runtime.tools.program_tools import program_content
        from ni_rua.core.tool_output import ToolResponse
        fake = FakeService(); self.addCleanup(fake.close)
        runtime = self.action_runtime(ot=True, s3o=True, endpoint=fake.url)
        runtime._model_context_epoch = lambda: "epoch-1"
        runtime._program_output_delivery = SimpleNamespace(
            project=lambda result, **kw: result,
            invalidate=lambda: None,
        )
        # 1) Cache before-observation object_table so object_state_delta can be computed
        before_bundle = dict(self.bundle, observation_id="obs-before")
        self.artifacts.save("observation", "obs-before", before_bundle)
        self.observe(runtime)
        obs3d._remember(
            runtime,
            "obs-before",
            "object_table",
            (
                {
                    "columns": ["id", "centre_m", "top_z", "bottom_z", "length", "width", "yaw_deg", "tilt_deg", "support", "quality"],
                    "rows": [["o1", [0.10, -0.20, 0.78], 0.80, 0.76, 0.08, 0.04, 15, 4, "table", "ok"]],
                    "table_z": TABLE,
                },
                b"",
                {},
            ),
        )
        # 2) Verify program_id _action_response data convenience fields + action_evidence parity
        prog_resp = self.action_reply(runtime, {"program_id": "job_1"})
        self.assertEqual(prog_resp.data["status"], "completed")
        self.assertEqual(prog_resp.data["observation_id"], self.bundle["observation_id"])
        self.assertEqual(prog_resp.data["result_observation_id"], self.bundle["observation_id"])
        self.assertEqual(prog_resp.data["stages"], {})
        stopped_data = {
            "plan_id": "plan_1",
            "action_id": "act_1",
            "request": {"action_id": "act_1", "based_on_observation_id": "obs-before"},
            "receipt": {
                "action_id": "act_1",
                "status": "stopped",
                "reason": "tracking_error",
                "failed_stage_id": "grasp",
                "result_observation_id": self.bundle["observation_id"],
                "executed_native_steps": 12,
            },
            "repair_action": {"mode": "direct", "plan_id": "plan_1", "observation_id": self.bundle["observation_id"]},
            "episode_ledger": {"contact_object_3d": {"id": "o1"}},
        }
        ev = action_evidence(
            self.artifacts,
            "robot_action",
            {"mode": "direct", "observation_id": "obs-before"},
            ToolResponse([], data=stopped_data, kind="action"),
            "call_1",
            physical=True,
        )
        self.assertEqual(ev["based_on_observation_id"], "obs-before")
        self.assertEqual(ev["repair_action"]["plan_id"], "plan_1")
        self.assertEqual(ev["episode_ledger"]["contact_object_3d"]["id"], "o1")
        # 3) Verify program_content attaches obs3d (object_table, scene3d, object_state_delta) on final_observation_id
        job = {
            "job_id": "program-1",
            "status": "succeeded",
            "final_observation_id": self.bundle["observation_id"],
            "action_summary": {"action_calls": 1, "recent_actions": [ev]},
        }
        parts = program_content(runtime, job, dict(job), "summary", {})
        obs3d_items = [
            p.value for p in parts
            if isinstance(p, StructuredText) and isinstance(p.value, dict) and "object_table" in p.value
        ]
        self.assertEqual(len(obs3d_items), 1)
        self.assertEqual([r[0] for r in obs3d_items[0]["object_table"]["rows"]], ["o1", "o2"])
        self.assertEqual(obs3d_items[0]["scene3d"]["points"], 3000)
        self.assertIn("object_state_delta", obs3d_items[0])

    def test_concurrent_object_table_and_scene3d_requests_overlap_in_flight(self):
        fake = FakeService(delay_s=1.0)
        self.addCleanup(fake.close)
        runtime = self.runtime(ot=True, s3o=True, endpoint=fake.url)
        parts = obs3d.attachments(runtime, self.bundle, {})
        self.assertCountEqual([path for path, _ in fake.requests], ["/object_table", "/points"])
        (first_start, first_end), (second_start, second_end) = fake.windows
        self.assertLess(max(first_start, second_start), min(first_end, second_end))
        self.assertEqual(parts[0].value["scene3d"]["points"], 3000)
        self.assertEqual(len(parts[0].value["object_table"]["rows"]), 2)

    def test_u1_se3_and_ik_contract_discoverability_and_zero_task_names(self):
        """验证 U1 模型侧契约完整暴露通用 IK/特征配准/SE(3) 连续路径接口、严格对齐 API 元组且不含任何具体任务名词。"""
        import re

        # 1) scene3d.API 与 scene3d_helpers.API 必须逐字符 100% 一致
        self.assertEqual(tuple(scene3d.API), s3.API)

        cfg = Config.from_dict(
            {
                "features": {
                    "external_perception": True,
                    "object_table": True,
                    "object_table_observe": True,
                    "scene3d": True,
                    "scene3d_observe": True,
                }
            }
        )
        api_text = "\n".join(scene3d.API)
        inst_text = obs3d.actor_instructions(cfg.features)
        self.assertEqual(inst_text, obs3d.developer_instructions(cfg.features))

        # 2) 验证全部目标通用函数、关键参数、默认值、物理单位 (m, deg, rad) 与返回字段均已完整曝光，且移除重型死代码条目
        required_api_tokens = (
            "check_reachability(arm,position_m,angles_deg,contact_region='pad_center')",
            "reachable,joint_q_rad,pos_err_m,rot_err_rad,min_joint_margin_rad",
            "yaw+-180 deg",
            "prefilter_stages(stages)",
            "valid,blocking_stage_ids,reasons,min_bimanual_clearance_m",
            "align_feature_to_target(grasp_pos_m,grasp_angles_deg,feature_point_m,feature_axis,target_point_m,target_axis)",
            "target_grasp_position_m,target_pos_m,target_angles_deg,feature_pos_err_m,feature_axis_err_deg",
            "se3_screw(start_pose,axis,pivot_m=None,angle_rad=0.0,distance_m=0.0,pitch_m_per_rad=0.0,n_steps=4)",
        )
        for token in required_api_tokens:
            self.assertIn(token, api_text)

        for retired_token in ("se3_spline", "se3_flow", "bimanual_sync", "compile_se3_trajectory", "compiled_field"):
            self.assertNotIn(retired_token, api_text)
            self.assertNotIn(retired_token, inst_text)

        required_inst_tokens = (
            "s3.check_reachability(arm, position_m, angles_deg, contact_region='pad_center')",
            "s3.check_source_to_dest(source_pos_m, source_angles_deg, dest_pos_m, dest_angles_deg=None)",
            "s3.in_hand_axis(grasp_center_m, radius=0.065, cloud=None, min_conf=0.1)",
            "s3.surface_extrema(region, radius=0.04, min_conf=0.05)",
            "yaw +- 180",
            "s3.prefilter_stages(stages)",
            "min_bimanual_clearance_m",
            "s3.align_feature_to_target(grasp_pos_m, grasp_angles_deg, feature_point_m, feature_axis, target_point_m, target_axis)",
            "s3.se3_screw(start_pose, axis, pivot_m=None, angle_rad=0.0, distance_m=0.0, pitch_m_per_rad=0.0, n_steps=4)",
        )
        for token in required_inst_tokens:
            self.assertIn(token, inst_text)

        # 3) 严禁泄露任何具体评测任务物品名词
        forbidden_task_words = (
            "toaster",
            "mug",
            "egg",
            "bread",
            "kettle",
            "cup",
            "bottle",
            "clothes",
            "shirt",
            "charger",
            "screw",
            "nut",
            "doll",
            "drawer",
            "cabinet",
            "door",
            "lid",
        )
        pattern = re.compile(r"\b(" + "|".join(forbidden_task_words) + r")\b", re.IGNORECASE)
        self.assertIsNone(pattern.search(api_text), f"scene3d.API 含禁止任务词: {pattern.search(api_text)}")
        self.assertIsNone(pattern.search(inst_text), f"obs3d.actor_instructions 含禁止任务词: {pattern.search(inst_text)}")

    def test_postprocess_resilience_and_honest_action_attachments_fallback(self):
        from copy import deepcopy
        from unittest import mock

        fake = FakeService()
        self.addCleanup(fake.close)
        runtime = self.runtime(ot=True, s3o=True, endpoint=fake.url)
        # 注入首帧缓存后构造损坏 robot_state_ref + 抛错 delta_sink 的第二帧，验证已计算的 object_table / scene3d 绝不被丢弃
        obs3d.attachments(runtime, self.bundle, {})
        bundle_bad_rs = {
            **deepcopy(self.bundle),
            "observation_id": "obs-ot2",
            "robot_state_ref": "artifact:missing_robot_state",
        }
        self.scene.import_observation({**deepcopy(self.bundle), "observation_id": "obs-ot2"})
        runtime.on_object_table_delta = lambda *_a: (_ for _ in ()).throw(RuntimeError("delta_sink_boom"))
        parts = obs3d.attachments(runtime, bundle_bad_rs, {}, before_observation_id=self.bundle["observation_id"])
        self.assertIsInstance(parts[0], StructuredText)
        self.assertIsInstance(parts[0].value["object_table"], dict)
        self.assertIsInstance(parts[0].value["scene3d"], dict)

        # action_attachments 在内部未预期异常时必须返回诚实的 unavailable 结构化说明并写诊断事件，绝不静默返回 []
        with mock.patch.object(obs3d, "attachments", side_effect=RuntimeError("unexpected_attach_crash")):
            fallback_parts = obs3d.action_attachments(runtime, self.bundle, {"tool_call_id": "act-err-1"})
        self.assertEqual(len(fallback_parts), 1)
        self.assertIsInstance(fallback_parts[0], StructuredText)
        self.assertTrue(str(fallback_parts[0].value["3d_perception"]).startswith("unavailable:"))
        self.assertTrue(str(fallback_parts[0].value["object_table"]).startswith("unavailable:"))
        self.assertTrue(str(fallback_parts[0].value["scene3d"]).startswith("unavailable:"))
        err_events = [e for e in self.artifacts.events() if e["type"] == "obs3d.action_attachments_error"]
        self.assertEqual(len(err_events), 1)

    def test_obs3d_single_rectification_sha256_points_and_accurate_scene3d_latency(self):
        from unittest import mock
        from ni_rua.perception.external.rectify import clear_rectified_cache

        clear_rectified_cache()
        scene3d.close_pool()
        self.addCleanup(clear_rectified_cache)
        self.addCleanup(scene3d.close_pool)

        # 1) Distorted JPEG camera in obs3d: cv2.undistort runs ONCE across concurrent /object_table + /points,
        #    /object_table uploads JPEG q=95, and /points omits image_b64 in favour of image_sha256
        dist_img = cv2.GaussianBlur(np.random.default_rng(5).integers(0, 256, (480, 640, 3), dtype=np.uint8), (5, 5), 0)
        raw_jpeg = cv2.imencode(".jpg", dist_img)[1].tobytes()
        k_dist = [[510.0, 0.0, 346.0], [0.0, 509.0, 196.0], [0.0, 0.0, 1.0]]
        d_dist = [-0.015, 0.224, -0.010, 0.015, -0.386]
        t_dist = [[1.0, 0.0, 0.0, 0.3], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.4], [0.0, 0.0, 0.0, 1.0]]
        cal_ref = self.artifacts.put_artifact(
            canonical(
                {
                    "reference_frame_id": "so101_base",
                    "cameras": {
                        "fixed": {
                            "K": k_dist,
                            "distortion": {"model": "opencv", "coefficients": d_dist},
                            "image_size": [640, 480],
                            "T_world_camera": t_dist,
                        }
                    },
                }
            ).encode(),
            "calibration",
            "application/json",
        )
        model_ref = self.artifacts.put_artifact(canonical({"base_link": "base", "joints": []}).encode(), "robot_model", "application/json")
        state_ref = self.artifacts.put_artifact(
            canonical({"source_kind": "measurement", "sim_step": 0, "step_domain": "control_step"}).encode(),
            "robot_state",
            "application/json",
        )
        dist_obs = {
            "schema_version": "0.1",
            **IDENTITY,
            "world_kind": "observed",
            "data_origin": "synthetic_fixture",
            "observation_id": "obs-distorted-jpg",
            "sim_step": 0,
            "step_domain": "control_step",
            "clock_domain": "synthetic/generation-1",
            "calibration_id": "synthetic-calibration-v1",
            "robot_model_id": "synthetic-camera-rig",
            "robot_model_ref": model_ref,
            "observation_profile": "rgb_proprio_calibrated_synthetic_v1",
            "robot_state_ref": state_ref,
            "cameras": [
                {
                    "camera_id": "fixed",
                    "frame_id": "frame-fixed",
                    "width": 640,
                    "height": 480,
                    "sim_step": 0,
                    "step_domain": "control_step",
                    "rgb_ref": self.artifacts.put_artifact(raw_jpeg, "rgb", "image/jpeg"),
                    "calibration_ref": cal_ref,
                    "pixel_transform": "identity",
                }
            ],
        }
        self.scene.import_observation(dist_obs)
        fake = FakeService()
        self.addCleanup(fake.close)
        cfg_dist = Config().with_overrides(
            {
                "features": {
                    "external_perception": False,
                    "object_table_observe": True,
                    "scene3d_observe": True,
                    "vggt_camera_id": "fixed",
                    "vggt_endpoint": fake.url,
                    "vggt_support_z_m": 0.0,
                    "obs3d_timeout_s": 5.0,
                }
            }
        )
        rt_dist = SimpleNamespace(
            config=cfg_dist,
            scene=self.scene,
            image_settings=None,
            identity=dict(IDENTITY),
            latest=None,
            active_action=None,
            store=self.artifacts,
            features=cfg_dist.features,
        )
        with mock.patch("cv2.undistort", wraps=cv2.undistort) as m_undistort:
            parts_dist = obs3d.attachments(rt_dist, dist_obs, {}, reply="action")
        self.assertEqual(m_undistort.call_count, 1)
        self.assertIsInstance(parts_dist[0].value["object_table"], dict)
        self.assertIsInstance(parts_dist[0].value["scene3d"], dict)
        by_path = dict(fake.requests)
        ot_bytes = base64.b64decode(by_path["/object_table"]["image_b64"])
        self.assertTrue(ot_bytes.startswith(b"\xff\xd8\xff"))
        self.assertNotEqual(ot_bytes, raw_jpeg)
        self.assertNotIn("image_b64", by_path["/points"])
        self.assertEqual(by_path["/points"]["image_sha256"], hashlib.sha256(ot_bytes).hexdigest())

        # 2) Deterministic event-gated stepped clock proves s3_latency_s = fetch_s + post_join_store_s (zero join double-count)
        fake_fast = FakeService()
        self.addCleanup(fake_fast.close)
        rt_lat = self.runtime(ot=True, s3o=True, endpoint=fake_fast.url, timeout=5.0)
        prebuilt_s3 = obs3d._scene3d_fetch(rt_lat, self.bundle["observation_id"], 5.0, camera_id="cam_head")
        prebuilt_ot = obs3d._object_table(rt_lat, self.bundle["observation_id"], 5.0, 10, camera_id="cam_head")

        clock_lock = threading.Lock()
        clock_state = {"now": 100.0}
        ot_done_evt = threading.Event()
        self.addCleanup(ot_done_evt.set)

        def _fake_monotonic():
            with clock_lock:
                return clock_state["now"]

        def _advance(dt: float):
            with clock_lock:
                clock_state["now"] += dt

        def _fast_ot(*_args, **kwargs):
            _advance(0.02)
            on_before_send = kwargs.get("on_before_send")
            if callable(on_before_send):
                on_before_send()
            return prebuilt_ot

        orig_analysis_log = obs3d._analysis_log

        def _signalling_analysis_log(*args, **kwargs):
            res = orig_analysis_log(*args, **kwargs)
            ot_done_evt.set()
            return res

        def _stepped_s3_fetch(*_args, **_kwargs):
            ot_done_evt.wait(timeout=2.0)
            _advance(0.40)  # simulated /points fetch duration (old bug counted 0.40 + 0.43 = 0.83s!)
            return prebuilt_s3

        orig_s3_store = obs3d._scene3d_store

        def _stepped_s3_store(*args, **kwargs):
            _advance(0.03)
            return orig_s3_store(*args, **kwargs)

        with mock.patch.object(obs3d, "_monotonic", side_effect=_fake_monotonic), \
                mock.patch.object(obs3d, "_object_table", side_effect=_fast_ot), \
                mock.patch.object(obs3d, "_analysis_log", side_effect=_signalling_analysis_log), \
                mock.patch.object(obs3d, "_scene3d_fetch", side_effect=_stepped_s3_fetch), \
                mock.patch.object(obs3d, "_scene3d_store", side_effect=_stepped_s3_store):
            obs3d.attachments(rt_lat, self.bundle, {}, reply="action")

        rec_lat = self.artifacts.read_state("perception", self.bundle["observation_id"])
        self.assertEqual(rec_lat["diagnostics"]["object_table"]["latency_s"], 0.02)
        self.assertEqual(rec_lat["diagnostics"]["scene3d"]["latency_s"], 0.43)
        self.assertEqual(rec_lat["diagnostics"]["latency_s"], 0.45)

        # 3) Prefetch timeout reports unavailable and finite non-negative latency without mutating shared FeaturesConfig
        hang_evt = threading.Event()
        self.addCleanup(hang_evt.set)
        obs_to = dict(self.bundle, observation_id="obs-s3-timeout")
        self.scene.import_observation(obs_to)
        base_rt_to = self.runtime(ot=True, s3o=True, endpoint=fake_fast.url, timeout=1.0, bundle=obs_to)
        short_features = SimpleNamespace(
            **{k: getattr(base_rt_to.features, k) for k in dir(base_rt_to.features) if not k.startswith("_")}
        )
        short_features.obs3d_timeout_s = 0.03
        rt_to = SimpleNamespace(
            **{
                **vars(base_rt_to),
                "features": short_features,
                "config": SimpleNamespace(**{**vars(base_rt_to.config), "features": short_features}),
            }
        )

        def _hanging_s3_fetch(*_a, **_kw):
            hang_evt.wait(1.0)
            return prebuilt_s3

        with mock.patch.object(obs3d, "_scene3d_fetch", side_effect=_hanging_s3_fetch):
            parts_to = obs3d.attachments(rt_to, obs_to, {}, reply="action")
        hang_evt.set()
        self.assertEqual(parts_to[0].value["scene3d"], "unavailable: timeout")
        rec_to = self.artifacts.read_state("perception", "obs-s3-timeout")
        self.assertGreaterEqual(rec_to["diagnostics"]["scene3d"]["latency_s"], 0.0)
        self.assertLess(rec_to["diagnostics"]["scene3d"]["latency_s"], 2.0)

    def test_object_table_note_reflects_persistent_ids_and_preserves_columns(self):
        from unittest import mock

        # 1) 默认 persistent_ids=False 与历史常量 OBJECT_TABLE_NOTE 严格一致；persistent_ids=True 动态切换 ID 作用域说明
        self.assertEqual(obs3d.object_table_note(None, "cam_head", overlay=True), obs3d.OBJECT_TABLE_NOTE)
        self.assertIn("ids are per observation", obs3d.object_table_note(None, "cam_head", overlay=False, persistent_ids=False))
        note_p_no_ov = obs3d.object_table_note(None, "cam_head", overlay=False, persistent_ids=True)
        note_p_ov = obs3d.object_table_note(None, "cam_head", overlay=True, persistent_ids=True)
        self.assertIn("ids are persistent episode tracks (see track_meta)", note_p_no_ov)
        self.assertNotIn("drawn on the next image", note_p_no_ov)
        self.assertIn("ids are persistent episode tracks (see track_meta) and drawn on the next image", note_p_ov)

        # 2) 当 object_table 返回 track_meta.persistent_ids=True 时，observe(含叠加图) 与 action(无叠加图) 均动态写入对应 note 且保留 columns
        runtime = self.runtime(ot=True, s3o=False)
        fake_res = {
            "status": "conditional",
            "value": {
                "columns": ["id", "centre_m", "top_z", "bottom_z", "length", "width", "yaw_deg", "tilt_deg", "support", "quality"],
                "rows": [["o1", [0.1, 0.0, 0.8], 0.84, 0.765, 0.06, 0.04, 15.0, 2.0, "table", "ok"]],
                "support_z_m": TABLE,
                "track_meta": {"persistent_ids": True, "window_len": 2},
            },
            "diagnostics": {},
        }
        fake_overlays = [("cam_head", self.images["cam_head"], "artifact:orig_rgb")]
        with mock.patch("ni_rua.perception.external.object_table.object_table_query", return_value=(fake_res, fake_overlays)):
            obs_parts = obs3d.attachments(runtime, self.bundle, {}, reply="observe")
            act_parts = obs3d.attachments(runtime, self.bundle, {}, reply="action")
        self.assertEqual(obs_parts[0].value["object_table"]["note"], note_p_ov)
        self.assertEqual(act_parts[0].value["object_table"]["note"], note_p_no_ov)
        self.assertEqual(obs_parts[0].value["object_table"]["columns"][0], "id")
        self.assertEqual(act_parts[0].value["object_table"]["columns"][0], "id")

    def test_scene3d_use_deduplicated_within_same_context_epoch(self):
        fake = FakeService(); self.addCleanup(fake.close)
        runtime = self.action_runtime(ot=False, s3o=True, endpoint=fake.url)
        epoch = ["epoch-1"]
        runtime._model_context_epoch = lambda: epoch[0]

        # 1) 未缓存的 prefetch_observe_attachments 不得提前消费主模型上下文纪元
        prefetched = obs3d.prefetch_observe_attachments(runtime, self.bundle, {}, cache_pending=False)
        self.assertIn("use", prefetched["scene3d"])

        # 2) 同一上下文纪元内首条主模型回复携带 scene3d.use，第二条省略 use（但归档与 memo 仍保留完整 scene3d）
        first_obs = obs3d.attachments(runtime, self.bundle, {}, reply="observe")
        self.assertEqual(first_obs[0].value["scene3d"]["use"], obs3d.SCENE3D_USE)

        second_act = obs3d.attachments(runtime, self.bundle, {}, reply="action")
        self.assertNotIn("use", second_act[0].value["scene3d"])
        self.assertEqual(set(second_act[0].value["scene3d"]), {"artifact", "points", "table_z"})
        record = self.artifacts.read_state("perception", self.bundle["observation_id"])
        self.assertEqual(record["scene3d"]["use"], obs3d.SCENE3D_USE)

        # 3) program_run 子调用不受主模型去重影响，始终携带 use
        prog_parts = obs3d.attachments(runtime, self.bundle, {"program_id": "job_1"}, reply="observe")
        self.assertEqual(prog_parts[0].value["scene3d"]["use"], obs3d.SCENE3D_USE)

        # 4) 上下文重置（纪元变更）后自动重新投递一次 use，再次调用又恢复省略
        epoch[0] = "epoch-2"
        after_reset = obs3d.attachments(runtime, self.bundle, {}, reply="action")
        self.assertEqual(after_reset[0].value["scene3d"]["use"], obs3d.SCENE3D_USE)
        after_reset_2 = obs3d.attachments(runtime, self.bundle, {}, reply="observe")
        self.assertNotIn("use", after_reset_2[0].value["scene3d"])

        # 5) prefetch_action_attachments + action_attachments 消费路径正确在投递后记账，避免重复或漏发
        epoch[0] = "epoch-3"
        self.assertTrue(obs3d.prefetch_action_attachments(runtime, self.bundle, {"tool_call_id": "tc-3"}))
        # 尚未消费 pending 前不提前消费 epoch-3
        self.assertNotEqual(obs3d.slot_peek(runtime, obs3d.SCENE3D_USE_EPOCH_SLOT, None), "epoch-3")
        delivered_act = obs3d.action_attachments(runtime, self.bundle, {"tool_call_id": "tc-3"})
        self.assertEqual(delivered_act[0].value["scene3d"]["use"], obs3d.SCENE3D_USE)
        next_in_ep3 = obs3d.attachments(runtime, self.bundle, {}, reply="observe")
        self.assertNotIn("use", next_in_ep3[0].value["scene3d"])

        # 6) full 上下文投递模式（易变纪元每次返回不同标记）下每次均重新投递 use
        counter = [0]
        def _volatile_epoch():
            counter[0] += 1
            return ("volatile", counter[0])
        runtime._model_context_epoch = _volatile_epoch
        for _ in range(2):
            v_parts = obs3d.attachments(runtime, self.bundle, {}, reply="action")
            self.assertEqual(v_parts[0].value["scene3d"]["use"], obs3d.SCENE3D_USE)

        # 7) isolated 作用域与过时纪元（epoch != live_epoch）均返回完整 use 且不写入/破坏主链 SCENE3D_USE_EPOCH_SLOT
        from ni_rua.runtime.decision_context import DecisionContextBridge
        epoch[0] = "epoch-main"
        runtime._model_context_epoch = lambda: epoch[0]
        runtime.embodiment = obs3d.embodiment_of(runtime)
        runtime.thread_id = "thr-obs3d"
        runtime.decision_context = DecisionContextBridge(runtime)

        # 先由 isolated 调用：必须含 use，且不得播种主链 epoch-main
        iso_parts = obs3d.attachments(runtime, self.bundle, {"context_scope": "isolated"}, reply="observe")
        self.assertEqual(iso_parts[0].value["scene3d"]["use"], obs3d.SCENE3D_USE)
        self.assertNotEqual(obs3d.slot_peek(runtime, obs3d.SCENE3D_USE_EPOCH_SLOT, None), "epoch-main")

        # 过时纪元（绑定 epoch-old != 活跃 epoch-main）：必须含 use，且不得播种主链
        stale_parts = obs3d.attachments(
            runtime, self.bundle, {"context_scope": "main", "context_epoch": "epoch-old"}, reply="action",
        )
        self.assertEqual(stale_parts[0].value["scene3d"]["use"], obs3d.SCENE3D_USE)
        self.assertNotEqual(obs3d.slot_peek(runtime, obs3d.SCENE3D_USE_EPOCH_SLOT, None), "epoch-old")

        # 主链在活跃 epoch-main 首次投递后写入基线，第二次省略，期间穿插 isolated 仍返回完整且不破坏主链省略
        main_1 = obs3d.attachments(
            runtime, self.bundle, {"context_scope": "main", "context_epoch": "epoch-main"}, reply="observe",
        )
        self.assertEqual(main_1[0].value["scene3d"]["use"], obs3d.SCENE3D_USE)
        iso_mid = obs3d.attachments(runtime, self.bundle, {"context_scope": "isolated"}, reply="action")
        self.assertEqual(iso_mid[0].value["scene3d"]["use"], obs3d.SCENE3D_USE)
        main_2 = obs3d.attachments(
            runtime, self.bundle, {"context_scope": "main", "context_epoch": "epoch-main"}, reply="action",
        )
        self.assertNotIn("use", main_2[0].value["scene3d"])


if __name__ == "__main__":
    unittest.main()
