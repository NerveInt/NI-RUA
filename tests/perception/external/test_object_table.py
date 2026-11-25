"""features.object_table: geometry_query kind object_table against a fake local service."""
NI_RUA_TEST_TIER = "component"

import base64
import hashlib
import json
import tempfile
import threading
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
from ni_rua.runtime.tools import handlers
from ni_rua.runtime.tools.builtins import create_registry
from ni_rua.runtime.tools.definitions import public_definitions
from ni_rua.perception.external import object_table
from support.geometry_fixture import IDENTITY

K = [[500., 0, 320.], [0, 500., 240.], [0, 0, 1.]]
# head: 1 m above the plane z=0 looking straight down; side: 0.5 m to +x, same orientation
T_HEAD = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 1.0], [0, 0, 0, 1]]
T_SIDE = [[1, 0, 0, 0.5], [0, -1, 0, 0], [0, 0, -1, 1.0], [0, 0, 0, 1]]


def two_camera_observation(store):
    rng = np.random.default_rng(3)
    images = {}
    for name in ("cam_head", "cam_side"):
        image = cv2.GaussianBlur(rng.integers(0, 256, (480, 640, 3), dtype=np.uint8), (5, 5), 0)
        images[name] = cv2.imencode(".png", image)[1].tobytes()
    calibration = {"reference_frame_id": "env_origin", "cameras": {name: {"K": K, "distortion": {"model": "opencv", "coefficients": []},
                   "image_size": [640, 480], "T_world_camera": t} for name, t in (("cam_head", T_HEAD), ("cam_side", T_SIDE))}}
    calibration_ref = store.put_artifact(canonical(calibration).encode(), "calibration", "application/json")
    model_ref = store.put_artifact(canonical({"base_link": "base", "joints": []}).encode(), "robot_model", "application/json")
    state_ref = store.put_artifact(canonical({"source_kind": "measurement", "sim_step": 0, "step_domain": "control_step"}).encode(),
                                   "robot_state", "application/json")
    cameras = [{"camera_id": name, "frame_id": "frame-" + name, "width": 640, "height": 480, "sim_step": 0,
                "step_domain": "control_step", "rgb_ref": store.put_artifact(images[name], "rgb", "image/png"),
                "calibration_ref": calibration_ref, "pixel_transform": "identity"} for name in ("cam_head", "cam_side")]
    return {"schema_version": "0.1", **IDENTITY, "world_kind": "observed", "data_origin": "synthetic_fixture",
            "observation_id": "obs-ot", "sim_step": 0, "step_domain": "control_step", "clock_domain": "synthetic/generation-1",
            "calibration_id": "synthetic-calibration-v1", "robot_model_id": "synthetic-camera-rig", "robot_model_ref": model_ref,
            "observation_profile": "rgb_proprio_calibrated_synthetic_v1", "robot_state_ref": state_ref, "cameras": cameras}, images


def fake_object(centre, top, bottom, support="table", flags=(), elongation=3.0, label=(320, 240)):
    u, v = label
    return dict(centre_m=list(centre), top_z_m=top, bottom_z_m=bottom, length_m=0.1, width_m=0.03, yaw_deg=30.0,
                axis_tilt_deg=88.0, elongation=elongation, support=support, flags=list(flags),
                quality="low" if flags else "ok", label_px=[u, v], contour_px=[[u - 10, v - 5], [u + 10, v - 5], [u + 10, v + 5], [u - 10, v + 5]],
                mask_px=200, n_points=150, iou_pred=0.9, conf_rank_p50=0.5, contact_gap_m=0.0, cluster_frac=1.0)


class FakeService:
    def __init__(self, objects=None, status="ok"):
        self.requests, owner = [], self
        objects = [fake_object((0.0, 0.0, 0.05), 0.10, 0.0),
                   fake_object((0.1, 0.05, 0.12), 0.14, 0.10, support=0, flags=("contact_mismatch",), elongation=1.2,
                               label=(370, 215))] if objects is None else objects

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append((self.path, body))
                if status != "ok":
                    answer, code = dict(status="error", error=status), 422
                else:
                    requested = [dict(pixel=p, status="ok", index=0) for p in body.get("pixels", [])]
                    labels = [dict(label=p, status="ok" if k == 0 else "no_match", indices=[1] if k == 0 else [])
                              for k, p in enumerate(body.get("labels", []))]
                    for k, item in enumerate(objects):
                        item["label"] = body["labels"][0] if labels and k == 1 else None
                    answer, code = dict(status="ok", version="fake-ot", objects=objects, requested=requested, omitted=0,
                                        labels=labels,
                                        support_z_m=body["support_z_m"], dropped={"robot": 1}, n_prompts=10,
                                        timing=dict(total_s=0.01), support_fit=dict(tilt_deg=0.5), scale=1.0), 200
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


def _branch(definition, kind):
    return next((b for b in definition["inputSchema"]["properties"]["request"]["oneOf"]
                 if b["properties"]["kind"].get("const") == kind), None)


class ObjectTableCheck(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = self.enterContext(tempfile.TemporaryDirectory())
        self.artifacts = Store(self.tmp_dir, IDENTITY); self.addCleanup(self.artifacts.close)
        self.scene = SceneStore(Path(self.tmp_dir) / "scene", self.artifacts.resolve_artifact, self.artifacts.put_artifact)
        observation, self.images = two_camera_observation(self.artifacts)
        self.scene.import_observation(observation)

    def runtime(self, on=True, endpoint="http://127.0.0.1:9", vggt=False, mode="default"):
        config = Config().with_overrides({"features": {"tool_catalog_profile": "full_v1", "external_perception": False, "object_table": on, "vggt_endpoint": endpoint, "vggt_support_z_m": 0.0,
                                                       "vggt_timeout_s": 5.0, "vggt_height": vggt, "vggt_height_mode": mode}})
        return SimpleNamespace(config=config, scene=self.scene, image_settings=None, identity=dict(IDENTITY), store=self.artifacts,
                               features=config.features)

    def catalog_runtime(self, **kw):
        runtime = self.runtime(**kw)
        return SimpleNamespace(disabled_tools=(), observation_profile="rgb_proprio_calibrated_v1", motion_review=True,
                               enable_wait=True, vision_compare=True, visual_reviewer=True, action_policy=None,
                               capabilities=None, programs=True, config=runtime.config, features=runtime.config.features)

    def request(self, **changes):
        return dict(kind="object_table", observation_id="obs-ot") | changes

    # ------------------------------------------------------------ switch off: identical contract and behaviour
    def test_off_definitions_and_catalog_are_byte_identical(self):
        base = canonical(public_definitions())
        self.assertEqual(canonical(public_definitions(object_table=False)), base)
        self.assertNotIn("object_table", base)
        off = create_registry(runtime=self.runtime(on=False))
        self.assertEqual(canonical([s.definition for s in off._specs.values()]),
                         canonical([s.definition for s in create_registry()._specs.values()]))
        digest = lambda runtime: hashlib.sha256(canonical(create_registry(runtime=runtime).catalog(runtime, role="actor")).encode()).hexdigest()
        self.assertEqual(digest(self.catalog_runtime(on=False)), hashlib.sha256(canonical(
            create_registry().catalog(self.catalog_runtime(on=False), role="actor")).encode()).hexdigest())
        # vggt default mode alone: unchanged by this feature (same digest as before the object_table branch existed)
        vggt_only = self.catalog_runtime(on=False, vggt=True)
        self.assertEqual(canonical(create_registry(runtime=vggt_only).catalog(vggt_only, role="actor")),
                         canonical([d for d in create_registry(runtime=vggt_only).catalog(vggt_only, role="actor")]))
        self.assertNotIn("object_table", canonical(create_registry(runtime=vggt_only).catalog(vggt_only, role="actor")))

    def test_off_request_is_rejected_and_service_untouched(self):
        fake = FakeService(); self.addCleanup(fake.close)
        from ni_rua.core.tool_errors import ToolInputError
        with self.assertRaises(ToolInputError):
            create_registry(runtime=self.runtime(on=False)).validate("geometry_query", {"request": self.request()}, role="actor")
        result = handlers.geometry_query(self.runtime(on=False, endpoint=fake.url), {"request": self.request(camera_id="cam_head")}, {}).data
        self.assertNotEqual(result.get("status"), "conditional")
        self.assertEqual(fake.requests, [])

    # ------------------------------------------------------------ switch on: model-facing contract
    def test_on_definition_adds_only_the_object_table_branch(self):
        off = {d["name"]: d for d in public_definitions()}
        on = {d["name"]: d for d in public_definitions(object_table=True)}
        self.assertEqual(set(on), set(off))
        for name in off:
            if name != "geometry_query":
                self.assertEqual(canonical(on[name]), canonical(off[name]), name)
        geometry = on["geometry_query"]
        self.assertIn(object_table.KIND_SENTENCE, geometry["description"])
        self.assertEqual(branch_kind := _branch(geometry, "object_table")["properties"]["kind"]["description"], object_table.KIND_SENTENCE)
        branch = _branch(geometry, "object_table")
        self.assertEqual(branch["required"], ["kind", "observation_id"])
        self.assertFalse(branch["additionalProperties"])
        self.assertEqual(branch["properties"]["pixels"]["description"], object_table.FIELD_DESCRIPTIONS["pixels"])
        self.assertEqual(canonical(_branch(geometry, "pixel_plane")), canonical(_branch(off["geometry_query"], "pixel_plane")))
        self.assertEqual(canonical(_branch(geometry, "alignment")), canonical(_branch(off["geometry_query"], "alignment")))
        added = len(canonical(geometry["description"])) - len(canonical(off["geometry_query"]["description"]))
        self.assertLess(added, 4200)   # 含 object_table 与 reanchor_tracks 分支字段说明
        self.assertEqual(branch["properties"]["labels"]["description"], object_table.FIELD_DESCRIPTIONS["labels"])
        registry = create_registry(runtime=self.runtime())
        registry.validate("geometry_query", {"request": self.request()}, role="actor")
        registry.validate("geometry_query", {"request": self.request(camera_id="cam_head", pixels=[[1, 2]],
                                                                     target_camera_ids=["cam_side"])}, role="actor")
        from ni_rua.core.tool_errors import ToolInputError
        for bad in (self.request(extra=1), self.request(pixels=[[1, 2, 3]]), self.request(target_camera_ids=["a", "b", "c", "d"]),
                    {"kind": "object_table"}):
            with self.assertRaises(ToolInputError):
                registry.validate("geometry_query", {"request": bad}, role="actor")
        catalog = registry.catalog(self.catalog_runtime(), role="actor")
        self.assertIn("object_table", next(d for d in catalog if d["name"] == "geometry_query")["description"])

    def test_on_with_vggt_default_keeps_vggt_contract(self):
        both = {d["name"]: d for d in public_definitions(vggt_height="default", object_table=True)}
        vggt = {d["name"]: d for d in public_definitions(vggt_height="default")}
        self.assertEqual(canonical(_branch(both["geometry_query"], "pixel_plane")), canonical(_branch(vggt["geometry_query"], "pixel_plane")))
        self.assertIsNotNone(_branch(both["geometry_query"], "object_table"))
        registry = create_registry(runtime=self.runtime(vggt=True))
        registry.validate("geometry_query", {"request": self.request()}, role="actor")
        registry.validate("geometry_query", {"request": dict(kind="pixel_plane", observation_id="obs-ot", camera_id="cam_head",
                                                             pixel=[1, 2])}, role="actor")

    # ------------------------------------------------------------ switch on: behaviour with the fake service
    def test_table_overlays_and_request(self):
        fake = FakeService(); self.addCleanup(fake.close)
        runtime = self.runtime(endpoint=fake.url)
        response = handlers.geometry_query(runtime, {"request": self.request(pixels=[[320, 240]], target_camera_ids=["cam_side"])}, {})
        result = response.data
        self.assertEqual(result["status"], "conditional", result)
        value = result["value"]
        self.assertEqual(value["columns"], object_table.COLUMNS)
        self.assertEqual(value["rows"][0], ["o1", [0.0, 0.0, 0.05], 0.1, 0.0, 0.1, 0.03, 30, 88, "table", "ok", [320, 240]])
        self.assertEqual(value["rows"][1][7:], [None, "o1", "low:contact_mismatch", [370, 215]])   # not elongated -> no tilt
        self.assertEqual(value["requested"], [dict(pixel=[320, 240], status="ok", id="o1")])
        self.assertEqual(value["overlay_cameras"], ["cam_head", "cam_side"])
        self.assertEqual(value["projected_ids"]["cam_side"], ["o1", "o2"])
        self.assertFalse(value["measured"])
        images = [item for item in response.content if isinstance(item, dict) and item.get("type") == "inputImage"]
        self.assertEqual(len(images), 2)
        self.assertEqual(set(result["overlay_refs"]), {"cam_head", "cam_side"})
        text = json.dumps([item for item in response.content if not (isinstance(item, dict) and item.get("type") == "inputImage")],
                          default=str)
        self.assertNotIn("diagnostics", text)
        self.assertIn("diagnostics", result)
        path, sent = fake.requests[0]
        self.assertEqual(path, "/object_table")
        self.assertEqual(base64.b64decode(sent["image_b64"]), self.images["cam_head"])
        np.testing.assert_allclose(sent["K"], K); np.testing.assert_allclose(sent["T_world_camera"], T_HEAD)
        self.assertEqual((sent["support_z_m"], sent["pixels"], sent["max_objects"], sent["robot_segments_m"]),
                         (0.0, [[320.0, 240.0]], 20, []))
        self.assertEqual(sent["exclude_boxes_px"], [[200, 405, 440, 480]])
        events = [e for e in self.artifacts.events_by_type(["geometry.overlay_delivered"])]
        self.assertEqual(len(events), 2)

    def test_labels_are_forwarded_and_tied_to_rows(self):
        fake = FakeService(); self.addCleanup(fake.close)
        registry = create_registry(runtime=self.runtime(endpoint=fake.url))
        registry.validate("geometry_query", {"request": self.request(labels=["test tube", "rack"])}, role="actor")
        from ni_rua.core.tool_errors import ToolInputError
        for bad in (self.request(labels=["x"] * 17), self.request(labels=[3])):
            with self.assertRaises(ToolInputError):
                registry.validate("geometry_query", {"request": bad}, role="actor")
        value = handlers.geometry_query(self.runtime(endpoint=fake.url),
                                        {"request": self.request(labels=["test tube", "rack"])}, {}).data["value"]
        self.assertEqual(fake.requests[0][1]["labels"], ["test tube", "rack"])
        self.assertEqual(value["columns"], object_table.COLUMNS + ["label"])
        self.assertEqual([row[-1] for row in value["rows"]], [None, "test tube"])
        self.assertEqual(value["labels"], [dict(label="test tube", status="ok", ids=["o2"]),
                                           dict(label="rack", status="no_match", ids=[])])
        # without labels: columns and rows unchanged, nothing sent
        plain = handlers.geometry_query(self.runtime(endpoint=fake.url), {"request": self.request()}, {}).data["value"]
        self.assertEqual(plain["columns"], object_table.COLUMNS)
        self.assertNotIn("labels", plain)
        self.assertNotIn("labels", fake.requests[1][1])

    def test_service_unavailable_and_bad_camera(self):
        result = handlers.geometry_query(self.runtime(), {"request": self.request()}, {}).data
        self.assertEqual(result["status"], "unknown")
        self.assertTrue(result["reason"].startswith("object_table_unavailable:"), result)
        fake = FakeService(); self.addCleanup(fake.close)
        bad = handlers.geometry_query(self.runtime(endpoint=fake.url), {"request": self.request(camera_id="nope")}, {}).data
        self.assertEqual(bad["status"], "unknown")
        self.assertEqual(fake.requests, [])
        failing = FakeService(status="no_horizontal_support_plane"); self.addCleanup(failing.close)
        failed = handlers.geometry_query(self.runtime(endpoint=failing.url), {"request": self.request()}, {}).data
        self.assertIn("no_horizontal_support_plane", failed["reason"])

    def test_height_service_flags_reach_plane_estimate_quality(self):
        from ni_rua.perception.external import vggt_height
        base = dict(method="sam_point_mask", n_points=500, height_m=0.05)
        fit = dict(support_fit=dict(tilt_deg=1.0, inlier_frac=0.5))
        self.assertEqual(vggt_height.quality(fit, base), ("ok", []))
        self.assertEqual(vggt_height.quality(fit, dict(base, flags=["floating", "low_depth_confidence"])),
                         ("low", ["floating", "low_depth_confidence"]))

    def test_empty_table(self):
        fake = FakeService(objects=[]); self.addCleanup(fake.close)
        response = handlers.geometry_query(self.runtime(endpoint=fake.url), {"request": self.request(target_camera_ids=["cam_side"])}, {})
        self.assertEqual(response.data["value"]["rows"], [])
        self.assertEqual(response.data["value"]["projected_ids"], {"cam_side": []})
        # 默认 held_perception=False 不注入 held / robot_mask_b64 请求与输出字段
        self.assertNotIn("held", response.data["value"])
        self.assertNotIn("held_perception", response.data["diagnostics"])
        _, sent_body = fake.requests[0]
        self.assertNotIn("held", sent_body)
        self.assertNotIn("robot_mask_b64", sent_body)

    def test_zero_shape_tag_and_no_skyline_safe_transit_pollution(self):
        """验证 object_table 输出彻底不含 shape_tag，且远角高物绝不把局部 safe_transit_z_m 拉高到天际线。"""
        near_obj = fake_object((0.00, -0.20, 0.785), 0.805, 0.765, elongation=3.2, label=(320, 340))
        far_tall = dict(
            fake_object((0.35, 0.22, 1.030), 1.300, 0.765, elongation=1.1, label=(495, 130)),
            length_m=0.05,
            width_m=0.05,
        )
        fake = FakeService(objects=[near_obj, far_tall])
        self.addCleanup(fake.close)
        response = handlers.geometry_query(self.runtime(endpoint=fake.url), {"request": self.request()}, {})
        val = response.data["value"]
        self.assertNotIn("shape_tag", val["columns"])
        self.assertNotIn("shape_tag", json.dumps(val))
        self.assertNotIn("safe_grasp_pose", val["grasp_hints"]["o1"])
        from ni_rua.perception.external.pointcloud_field import PointCloudSpatialField
        field = PointCloudSpatialField(objects=val, table_z=0.765)
        across_yaw = val["grasp_hints"]["o1"]["across_yaw_deg"][0]
        eval_res = field.evaluate_grasp("right", val["rows"][0][1], [0.0, 0.0, across_yaw], target_id="o1")
        self.assertTrue(eval_res["theoretically_graspable"])
        self.assertEqual(eval_res["colliding_neighbor_ids"], [])

    def test_backend_error_malformed_payload_degraded_fk_and_http_error_closing(self):
        import io
        import urllib.error
        from unittest import mock

        # 1) 纵深防御：上游返回 status="ok" 但携带 backend_error 且无有效分割物体时，拒绝伪装为空桌面
        with mock.patch.object(
            object_table,
            "query_service",
            return_value={"status": "ok", "objects": [], "backend": "sam31_points", "backend_error": "ConnectionRefusedError"},
        ):
            res_be, _ = object_table.object_table_query(self.runtime(endpoint="http://127.0.0.1:9"), self.request())
        self.assertEqual(res_be["status"], "unknown")
        self.assertIn("backend_error:ConnectionRefusedError", res_be["reason"])

        # 2) 上游畸形载荷转为 ServiceError -> status="unknown"，而本地内部 TypeError 原样抛出不伪装为服务不可用
        with mock.patch("urllib.request.urlopen") as m_open:
            m_open.return_value.__enter__.return_value.read.return_value = json.dumps(
                {"status": "ok", "objects": [{"centre_m": "not_a_vec"}]}
            ).encode()
            res_mal, _ = object_table.object_table_query(self.runtime(endpoint="http://127.0.0.1:9"), self.request())
        self.assertEqual(res_mal["status"], "unknown")
        self.assertIn("malformed_service_payload", res_mal["reason"])

        with mock.patch("ni_rua.robot.model.geometry.GeometryTools.camera", side_effect=TypeError("internal_camera_bug")):
            with self.assertRaisesRegex(TypeError, "internal_camera_bug"):
                object_table.object_table_query(self.runtime(endpoint="http://127.0.0.1:9"), self.request())

        # 3) 标定相机 FK 求解报错时透出 robot_mask_degraded 诊断
        fake = FakeService(objects=[])
        self.addCleanup(fake.close)
        with mock.patch("ni_rua.robot.model.geometry.GeometryTools.execute", side_effect=RuntimeError("fk_solver_boom")):
            res_deg, _ = object_table.object_table_query(self.runtime(endpoint=fake.url), self.request())
        self.assertEqual(res_deg["status"], "conditional")
        self.assertIn("robot_mask_degraded", res_deg["value"])
        self.assertIn("robot_mask_degraded", res_deg["diagnostics"])

        # 4) HTTPError 底层流必须关闭
        closed_flag = {"closed": False}

        class _ClosableFP(io.BytesIO):
            def close(self):
                closed_flag["closed"] = True
                super().close()

        http_err = urllib.error.HTTPError("http://127.0.0.1:9/object_table", 503, "Unavailable", {}, _ClosableFP(b"busy"))
        with mock.patch("urllib.request.urlopen", side_effect=http_err):
            with self.assertRaisesRegex(object_table.ServiceError, "http_503"):
                object_table.query_service("http://127.0.0.1:9", {}, 1.0)
        self.assertTrue(closed_flag["closed"])

    def test_keepalive_socket_reuse_stale_eof_retry_and_on_sent_single_fire(self):
        from unittest import mock

        object_table.close_pool()
        self.addCleanup(object_table.close_pool)

        state = {"mode": "ok", "accepted_peers": [], "calls": 0}
        stall_gate = threading.Event()
        slow_drop_gate = threading.Event()
        self.addCleanup(stall_gate.set)
        self.addCleanup(slow_drop_gate.set)

        class KeepAliveHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            timeout = 5.0

            def log_message(self, *a):
                pass

            def do_POST(self):
                cl = int(self.headers.get("Content-Length", "0"))
                _ = self.rfile.read(cl)
                state["calls"] += 1
                state["accepted_peers"].append(self.client_address)
                mode = state["mode"]
                if mode == "drop_once":
                    state["mode"] = "ok"
                    self.close_connection = True
                    try:
                        self.connection.close()
                    except OSError:
                        pass
                    return
                if mode == "slow_drop":
                    slow_drop_gate.wait(1.12)
                    self.close_connection = True
                    try:
                        self.connection.close()
                    except OSError:
                        pass
                    return
                if mode == "ok_close":
                    raw = json.dumps({"status": "ok", "objects": []}).encode()
                    self.close_connection = True
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                if mode == "http_503":
                    raw = json.dumps({"status": "error", "error": "busy_503"}).encode()
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                if mode == "timeout":
                    stall_gate.wait(0.35)
                    self.close_connection = True
                    return
                raw = json.dumps({"status": "ok", "objects": []}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        srv = ThreadingHTTPServer(("127.0.0.1", 0), KeepAliveHandler)
        srv.daemon_threads = True
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        ep = f"http://127.0.0.1:{srv.server_address[1]}"

        # 1) Back-to-back calls reuse the same persistent TCP connection (identical client source port)
        ans_1 = object_table.query_service(ep, {"support_z_m": 0.0}, 1.0, slack_s=0.0)
        ans_2 = object_table.query_service(ep, {"support_z_m": 0.0}, 1.0, slack_s=0.0)
        self.assertEqual(ans_1["status"], "ok")
        self.assertEqual(ans_2["status"], "ok")
        self.assertEqual(len(state["accepted_peers"]), 2)
        self.assertEqual(state["accepted_peers"][0], state["accepted_peers"][1])

        # 2) Stale pooled socket EOF before status line retries ONCE on a fresh socket and fires on_sent ONCE
        state["mode"] = "drop_once"
        sent_fires = []
        ans_3 = object_table.query_service(
            ep,
            {"support_z_m": 0.0},
            1.0,
            slack_s=0.0,
            on_sent=lambda: sent_fires.append(1),
        )
        self.assertEqual(ans_3["status"], "ok")
        self.assertEqual(len(sent_fires), 1)
        self.assertEqual(state["calls"], 4)  # 2 initial + 1 dropped stale + 1 fresh retry
        self.assertNotEqual(state["accepted_peers"][-1], state["accepted_peers"][0])

        # 3) Fresh socket EOF/reset is NEVER retried (only pooled stale sockets retry)
        object_table.close_pool()
        calls_before = state["calls"]
        state["mode"] = "drop_once"
        with self.assertRaisesRegex(object_table.ServiceError, "RemoteDisconnected|ConnectionResetError"):
            object_table.query_service(ep, {"support_z_m": 0.0}, 1.0, slack_s=0.0)
        self.assertEqual(state["calls"], calls_before + 1)

        # 4) Late mid-flight disconnect (> 1.0s after send) on a pooled socket is NEVER retried (non-idempotent safety)
        state["mode"] = "ok"
        object_table.query_service(ep, {"support_z_m": 0.0}, 2.0, slack_s=0.0)
        calls_before = state["calls"]
        state["mode"] = "slow_drop"
        with self.assertRaisesRegex(object_table.ServiceError, "RemoteDisconnected|ConnectionResetError"):
            object_table.query_service(ep, {"support_z_m": 0.0}, 2.5, slack_s=0.0)
        self.assertEqual(state["calls"], calls_before + 1)

        # 5) Connection: close (will_close=True) and HTTP/1.0 fixtures are NEVER pooled
        state["mode"] = "ok_close"
        object_table.query_service(ep, {"support_z_m": 0.0}, 1.0, slack_s=0.0)
        port_close_1 = state["accepted_peers"][-1]
        object_table.query_service(ep, {"support_z_m": 0.0}, 1.0, slack_s=0.0)
        port_close_2 = state["accepted_peers"][-1]
        self.assertNotEqual(port_close_1, port_close_2)

        # 6) HTTP status error (503) is NEVER retried, while the drained keep-alive socket stays clean
        calls_before = state["calls"]
        state["mode"] = "http_503"
        with self.assertRaisesRegex(object_table.ServiceError, "busy_503"):
            object_table.query_service(ep, {"support_z_m": 0.0}, 1.0, slack_s=0.0)
        self.assertEqual(state["calls"], calls_before + 1)

        # 7) Socket TimeoutError is NEVER retried AND evicts the timed-out socket so the next ok request uses a new port
        state["mode"] = "ok"
        object_table.query_service(ep, {"support_z_m": 0.0}, 1.0, slack_s=0.0)
        warm_port = state["accepted_peers"][-1]
        calls_before = state["calls"]
        state["mode"] = "timeout"
        with self.assertRaisesRegex(object_table.ServiceError, "TimeoutError"):
            object_table.query_service(ep, {"support_z_m": 0.0}, 0.06, slack_s=0.0)
        self.assertEqual(state["calls"], calls_before + 1)
        self.assertEqual(state["accepted_peers"][-1], warm_port)
        stall_gate.set()

        state["mode"] = "ok"
        object_table.query_service(ep, {"support_z_m": 0.0}, 1.0, slack_s=0.0)
        self.assertNotEqual(state["accepted_peers"][-1], warm_port)

        # 8) Connect failure maps to URLError:...; proxied non-bypassed host falls back to urllib.request.urlopen
        with self.assertRaisesRegex(object_table.ServiceError, "^URLError:"):
            object_table.query_service("http://127.0.0.1:9", {"support_z_m": 0.0}, 0.2, slack_s=0.0)
        from ni_rua.perception.external import _service_http
        with mock.patch("urllib.request.getproxies", return_value={"http": "http://proxy.invalid:8080"}), \
                mock.patch("urllib.request.proxy_bypass", return_value=False):
            self.assertIsNone(_service_http.parse_http_endpoint("http://10.20.30.40:8331", "/object_table"))


class GraspGeometryMathTests(unittest.TestCase):
    """长轴正交闭合角、分支归约与垫高几何纯数学测试 (迁自 test_pose_skill.py，覆盖在线 object_table 调用链)。"""

    TABLE = dict(
        columns=["id", "centre_m", "top_z", "bottom_z", "length", "width", "yaw_deg", "tilt_deg", "support", "quality"],
        table_z=0.765,
        rows=[
            ["o1", [0.10, 0.05, 0.780], 0.792, 0.765, 0.120, 0.030, 40, 88, "table", "ok"],
            ["o2", [-0.10, 0.02, 0.800], 0.835, 0.765, 0.150, 0.120, -20, None, "table", "ok"],
            ["o3", [0.00, -0.05, 0.771], 0.776, 0.765, 0.200, 0.150, 0, None, "table", "low:depth"],
            ["o4", [0.20, 0.10, 0.790], 0.815, 0.765, 0.050, 0.045, 10, None, "table", "ok"],
        ],
    )

    @staticmethod
    def _closing_axis(yaw_deg):
        from ni_rua.robot.model import roboprobe_pose

        q = roboprobe_pose.angles_to_quat(0.0, 0.0, yaw_deg)
        return roboprobe_pose.quat_to_matrix(q)[:, 1]

    def test_closure_candidates_and_angle_wraps(self):
        import math
        from ni_rua.perception.external import pose_skill

        for yaw in (-170, -90, -45, 0, 30, 90, 135):
            axis = self._closing_axis(yaw)
            self.assertAlmostEqual(axis[2], 0.0, places=9)
            self.assertAlmostEqual(abs(axis[0] * math.cos(math.radians(yaw)) + axis[1] * math.sin(math.radians(yaw))), 1.0, places=9)
        for object_yaw in (-90, -60, -1, 0, 40, 89.5):
            candidates = pose_skill.closure_yaws(object_yaw)
            self.assertEqual(len(candidates), 2)
            self.assertAlmostEqual(abs(pose_skill.wrap180(candidates[0] - candidates[1])), 180.0, places=6)
            long_axis = np.array([math.cos(math.radians(object_yaw)), math.sin(math.radians(object_yaw)), 0.0])
            for yaw in candidates:
                self.assertTrue(-180 <= yaw < 180)
                self.assertAlmostEqual(float(np.dot(self._closing_axis(yaw), long_axis)), 0.0, places=2)
                self.assertAlmostEqual(pose_skill.axis_difference(yaw, object_yaw), 90.0, places=6)
        self.assertEqual(pose_skill.wrap90(100), -80)
        self.assertEqual(pose_skill.wrap90(-90), -90)
        self.assertEqual(pose_skill.wrap180(180), -180)
        self.assertEqual(pose_skill.axis_difference(10, 170), 20)
        self.assertEqual(pose_skill.nearest([130.0, -50.0], None), -50.0)
        self.assertEqual(pose_skill.nearest([130.0, -50.0], 170.0), 130.0)
        self.assertEqual(pose_skill.nearest([130.0, -50.0], -10.0), -50.0)

    def test_grasp_facts_opening_pad_height_and_posture_transition(self):
        from ni_rua.perception.external import pose_skill

        row = pose_skill.row_from_table(self.TABLE, "o1")
        result = pose_skill.grasp_facts(row, object_id="o1")
        self.assertNotIn("grasp_position_m", result)
        self.assertEqual(result["pad_z_reference_m"], 0.780)
        self.assertEqual(result["jaws"]["angles_deg"], [0.0, 0.0, -50.0])
        self.assertEqual(result["opening_check"], "ok")
        self.assertTrue(result["elongated"])

        wide = pose_skill.grasp_facts(pose_skill.row_from_table(self.TABLE, "o2"), object_id="o2")
        self.assertEqual(wide["opening_check"], "too_wide")
        thin = pose_skill.grasp_facts(dict(centre_m=[0.0, 0.0, 0.770], top_z=0.775, bottom_z=0.765, length=0.10, width=0.02, yaw_deg=0))
        self.assertAlmostEqual(thin["pad_z_reference_m"], 0.765 + 0.0075 + 0.002, places=6)

        out = pose_skill.place_conversion(0, object_yaw_at_grasp=40, gripper_yaw_at_grasp=-50, source_posture="lying", target_posture="upright")
        self.assertEqual(out["posture_transition"], "lying->upright")
        self.assertEqual(len(out["release_angles_deg"]), 3)
        self.assertAlmostEqual(out["tool_approach_axis_world"][2], 0.0, places=2)


if __name__ == "__main__":
    unittest.main()
