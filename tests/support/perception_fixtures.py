"""Shared perception fake services and synthetic clouds used across perception and robot zones."""
from __future__ import annotations

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from ni_rua.core.contracts import canonical
from support.geometry_fixture import IDENTITY

K = [[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]]
T_HEAD = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 1.0], [0, 0, 0, 1]]
T_SIDE = [[1, 0, 0, 0.5], [0, -1, 0, 0], [0, 0, -1, 1.0], [0, 0, 0, 1]]

KC = np.array([[300.0, 0, 320.0], [0, 300.0, 240.0], [0, 0, 1.0]])
T_CAM = np.eye(4)
TABLE = 0.765


def two_camera_observation(store):
    rng = np.random.default_rng(3)
    images = {}
    for name in ("cam_head", "cam_side"):
        image = cv2.GaussianBlur(rng.integers(0, 256, (480, 640, 3), dtype=np.uint8), (5, 5), 0)
        images[name] = cv2.imencode(".png", image)[1].tobytes()
    calibration = {
        "reference_frame_id": "env_origin",
        "cameras": {
            name: {
                "K": K,
                "distortion": {"model": "opencv", "coefficients": []},
                "image_size": [640, 480],
                "T_world_camera": t,
            }
            for name, t in (("cam_head", T_HEAD), ("cam_side", T_SIDE))
        },
    }
    calibration_ref = store.put_artifact(canonical(calibration).encode(), "calibration", "application/json")
    model_ref = store.put_artifact(canonical({"base_link": "base", "joints": []}).encode(), "robot_model", "application/json")
    state_ref = store.put_artifact(
        canonical({"source_kind": "measurement", "sim_step": 0, "step_domain": "control_step"}).encode(),
        "robot_state",
        "application/json",
    )
    cameras = [
        {
            "camera_id": name,
            "frame_id": "frame-" + name,
            "width": 640,
            "height": 480,
            "sim_step": 0,
            "step_domain": "control_step",
            "rgb_ref": store.put_artifact(images[name], "rgb", "image/png"),
            "calibration_ref": calibration_ref,
            "pixel_transform": "identity",
        }
        for name in ("cam_head", "cam_side")
    ]
    return (
        {
            "schema_version": "0.1",
            **IDENTITY,
            "world_kind": "observed",
            "data_origin": "synthetic_fixture",
            "observation_id": "obs-ot",
            "sim_step": 0,
            "step_domain": "control_step",
            "clock_domain": "synthetic/generation-1",
            "calibration_id": "synthetic-calibration-v1",
            "robot_model_id": "synthetic-camera-rig",
            "robot_model_ref": model_ref,
            "observation_profile": "rgb_proprio_calibrated_synthetic_v1",
            "robot_state_ref": state_ref,
            "cameras": cameras,
        },
        images,
    )


def fake_object(centre, top, bottom, support="table", flags=(), elongation=3.0, label=(320, 240)):
    u, v = label
    return dict(
        centre_m=list(centre),
        top_z_m=top,
        bottom_z_m=bottom,
        length_m=0.1,
        width_m=0.03,
        yaw_deg=30.0,
        axis_tilt_deg=88.0,
        elongation=elongation,
        support=support,
        flags=list(flags),
        quality="low" if flags else "ok",
        label_px=[u, v],
        contour_px=[[u - 10, v - 5], [u + 10, v - 5], [u + 10, v + 5], [u - 10, v + 5]],
        mask_px=200,
        n_points=150,
        iou_pred=0.9,
        conf_rank_p50=0.5,
        contact_gap_m=0.0,
        cluster_frac=1.0,
    )


class FakeService:
    def __init__(self, objects=None, status="ok"):
        self.requests, owner = [], self
        objects = (
            [
                fake_object((0.0, 0.0, 0.05), 0.10, 0.0),
                fake_object(
                    (0.1, 0.05, 0.12),
                    0.14,
                    0.10,
                    support=0,
                    flags=("contact_mismatch",),
                    elongation=1.2,
                    label=(370, 215),
                ),
            ]
            if objects is None
            else objects
        )

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
                    labels = [
                        dict(label=p, status="ok" if k == 0 else "no_match", indices=[1] if k == 0 else [])
                        for k, p in enumerate(body.get("labels", []))
                    ]
                    for k, item in enumerate(objects):
                        item["label"] = body["labels"][0] if labels and k == 1 else None
                    answer, code = (
                        dict(
                            status="ok",
                            version="fake-ot",
                            objects=objects,
                            requested=requested,
                            omitted=0,
                            labels=labels,
                            support_z_m=body["support_z_m"],
                            dropped={"robot": 1},
                            n_prompts=10,
                            timing=dict(total_s=0.01),
                            support_fit=dict(tilt_deg=0.5),
                            scale=1.0,
                        ),
                        200,
                    )
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
    return next(
        (
            b
            for b in definition["inputSchema"]["properties"]["request"]["oneOf"]
            if b["properties"]["kind"].get("const") == kind
        ),
        None,
    )


class HeldService:
    """Fake hp service: /object_table answers two objects (+ held results when asked), /held a wrist result."""

    def __init__(self, held_head=None, held_wrist=None, objects=None, version="fake-hp"):
        self.requests, owner = [], self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append((self.path, body))
                if self.path.startswith("/held"):
                    if version == "object-table-v5":
                        self.send_response(404)
                        self.end_headers()
                        return
                    answer = dict(status="ok", held=[held_wrist] if held_wrist else [dict(status="no_mask")])
                else:
                    answer = dict(
                        status="ok",
                        version=version,
                        objects=objects
                        or [fake_object((0.0, 0.0, 0.05), 0.10, 0.0), fake_object((0.1, 0.05, 0.9), 0.95, 0.85)],
                        requested=[],
                        omitted=0,
                        labels=[],
                        support_z_m=body["support_z_m"],
                        dropped={},
                        n_prompts=1,
                        timing={},
                        support_fit={},
                        scale=1.0,
                    )
                    if version != "object-table-v5" and body.get("robot_mask_b64"):
                        answer["robot_mask"] = "pixels"
                    if version != "object-table-v5" and body.get("held"):
                        answer["held"] = [held_head]
                        answer["held_s"] = 0.01
                data = json.dumps(answer).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class FakeRecords:
    """record_page('observation') + resolve_artifact over a list of (observation_id, sim_step, {arm: state})."""

    def __init__(self):
        self.items, self.states = [], {}

    def add(self, oid, step, arms):
        ref = "artifact:state-%s" % oid
        self.states[ref] = json.dumps(dict(arms=arms)).encode()
        self.items.append(
            dict(id=oid, value=dict(observation_id=oid, sim_step=step, robot_state_ref=ref), position=len(self.items) + 1)
        )

    def record_page(self, kind, after=0, limit=500):
        page = [e for e in self.items if e["position"] > after][:limit]
        return dict(
            entries=page,
            cursor=page[-1]["position"] if page else after,
            has_more=len([e for e in self.items if e["position"] > after]) > limit,
            reset=False,
        )

    def resolve_artifact(self, ref, allowed_scopes=("actor",)):
        return {}, self.states[ref]


def synthetic_cloud(seed=0):
    """Table grid (1 cm) at z=TABLE over [-0.3, 0.3]^2 with two objects and a few robot points."""
    rng = np.random.default_rng(seed)
    g = np.arange(-0.30, 0.3001, 0.01)
    X, Y = np.meshgrid(g, g)
    table = np.stack([X.ravel(), Y.ravel(), np.full(X.size, TABLE)], 1)
    box_top = [[x, y, TABLE + 0.04] for x in np.arange(0.07, 0.1301, 0.005) for y in np.arange(-0.02, 0.0201, 0.005)]
    box_side = [[0.07, y, z] for y in np.arange(-0.02, 0.0201, 0.005) for z in np.arange(TABLE + 0.005, TABLE + 0.0301, 0.005)]
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    cyl = [[-0.10 + 0.02 * np.cos(a), 0.10 + 0.02 * np.sin(a), z] for a in ang for z in np.arange(TABLE + 0.01, TABLE + 0.1001, 0.01)]
    cyl += [[-0.10 + r * np.cos(a), 0.10 + r * np.sin(a), TABLE + 0.10] for r in (0.0, 0.01) for a in ang[::3]]
    robot = [[0.2, 0.2, TABLE + z] for z in np.arange(0.05, 0.3, 0.01)]
    objects = np.asarray(box_top + box_side + cyl, float)
    keep = np.ones(len(table), bool)
    keep &= ~((np.abs(table[:, 0] - 0.10) <= 0.03) & (np.abs(table[:, 1]) <= 0.02))
    keep &= np.hypot(table[:, 0] + 0.10, table[:, 1] - 0.10) > 0.02
    xyz = np.concatenate([table[keep], objects, np.asarray(robot)])
    robot_flag = np.zeros(len(xyz), bool)
    robot_flag[-len(robot) :] = True
    conf = rng.integers(60, 256, len(xyz)).astype(np.uint8)
    conf[5] = 0
    uv = np.stack([320 + 500 * xyz[:, 0], 240 - 500 * xyz[:, 1]], 1).round().astype(np.uint16)
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        xyz=xyz.astype(np.float32),
        conf=conf,
        uv=uv,
        count=np.ones(len(xyz), np.uint16),
        robot=robot_flag,
        support_z=np.float64(TABLE),
        voxel_m=np.float64(0.01),
        image_size=np.asarray([640, 480], np.int32),
    )
    return buf.getvalue()


class FakePoints:
    def __init__(self, status="ok"):
        self.requests, owner = [], self
        data = synthetic_cloud()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append((self.path, body))
                if status != "ok":
                    answer, code = dict(status="error", error=status), 422
                else:
                    answer, code = (
                        dict(
                            status="ok",
                            version="fake-points",
                            n_points=3000,
                            n_above_support=600,
                            n_robot=25,
                            voxel_m=body["voxel_m"],
                            support_z_m=body["support_z_m"],
                            bbox_m=[[-0.3, -0.3, 0.765], [0.3, 0.3, 1.055]],
                            bbox_above_support_m=[[-0.12, -0.02, 0.77], [0.13, 0.12, 0.865]],
                            npz_b64=base64.b64encode(data).decode(),
                            npz_bytes=len(data),
                            timing=dict(total_s=0.01),
                            support_fit=dict(tilt_deg=0.5),
                            scale=1.0,
                        ),
                        200,
                    )
                raw = json.dumps(answer).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.data = data
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
