"""Real loopback HTTP scene service, bounded child jobs, cancellation, expiry, restart, upload/replay, and visual feedback."""
NI_RUA_TEST_TIER = "component"

import asyncio
import base64
from copy import deepcopy
import hashlib
import json
import tempfile
import unittest

from aiohttp import web, ClientSession

from ni_rua.core.contracts import canonical, digest, new_id
from ni_rua.perception.scene.scene_service import SceneService
from support.geometry_fixture import IDENTITY, recorded_plane


class _SceneTestClient:
    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")

    async def request(self, method: str, path: str, body=None, *, idempotency_key: str | None = None, params=None):
        headers = {"Idempotency-Key": idempotency_key or new_id("request")} if method == "POST" else {}
        async with ClientSession() as session:
            async with session.request(method, self.url + path, json=body, params=params, headers=headers) as resp:
                text = await resp.text()
                try:
                    result = json.loads(text)
                except ValueError:
                    result = {"error": text}
                if resp.status >= 400:
                    raise ValueError(f"scene_http_{resp.status}:{result}")
                return result

    async def capabilities(self): return await self.request("GET", "/v1/capabilities")
    async def observation(self, bundle): return await self.request("POST", "/v1/observations", bundle, idempotency_key="obs-" + digest(bundle))
    async def head(self, identity): return await self.request("GET", "/v1/scene/head", params=identity)
    async def geometry(self, req): return await self.request("POST", "/v1/geometry/query", req)
    async def update(self, req, key=None): return await self.request("POST", "/v1/scene/update-jobs", req, idempotency_key=key)
    async def snapshot(self, req, key=None): return await self.request("POST", "/v1/snapshots", req, idempotency_key=key)
    async def cancel(self, identifier, identity, reason="client_request"): return await self.request("POST", f"/v1/jobs/{identifier}/cancel", {**identity, "reason": reason})
    async def events(self, after=0): return await self.request("GET", "/v1/events", params={"after": after})

    async def upload(self, content: bytes, kind: str, mime_type: str, source_artifact_id=None):
        sha = hashlib.sha256(content).hexdigest()
        return await self.request("POST", "/v1/blobs", {"sha256": sha, "kind": kind, "mime_type": mime_type, "data_base64": base64.b64encode(content).decode(), "source_artifact_id": source_artifact_id}, idempotency_key=digest([sha, kind, mime_type, source_artifact_id]))

    async def wait(self, identifier: str, *, timeout_s: float = 15, poll_s: float = 0.02):
        deadline = asyncio.get_running_loop().time() + timeout_s
        while True:
            job = await self.request("GET", "/v1/jobs/" + identifier)
            if job["status"] in ("succeeded", "failed", "cancelled", "expired"):
                return job
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"job still running: {identifier}")
            await asyncio.sleep(poll_s)


class ServiceCheck(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.service = SceneService(self.tmp)
        self.obs = recorded_plane(self.service.store)
        self.runner = web.AppRunner(self.service.app()); await self.runner.setup()
        self.site = web.TCPSite(self.runner,"127.0.0.1",0); await self.site.start()
        port = self.site._server.sockets[0].getsockname()[1]
        self.client = _SceneTestClient(f"http://127.0.0.1:{port}")
        for observation in self.obs: await self.client.observation(observation)
        self.request = {"schema_version":"0.1",**IDENTITY,"observation_ids":["obs-0","obs-1"],"camera_id":"head","static_scene_assumption":True,"base_scene_version":0,"roi":[100,80,400,320]}

    async def asyncTearDown(self):
        await self.runner.cleanup()

    async def test_http_job_idempotency_snapshot_events(self):
        specification=await self.client.request("GET","/v1/openapi.json")
        self.assertEqual(specification["openapi"],"3.1.0")
        self.assertIn("post",specification["paths"]["/v1/scene/update-jobs"])
        self.assertTrue(specification["paths"]["/v1/scene/query"]["post"]["parameters"][0]["required"])
        created = await self.client.update(self.request,"once")
        repeated = await self.client.update(self.request,"once")
        self.assertEqual(created["job_id"],repeated["job_id"])
        with self.assertRaisesRegex(ValueError,"409"):
            await self.client.update({**self.request,"roi":[120,80,400,320]},"once")
        complete = await self.client.wait(created["job_id"])
        self.assertEqual(complete["status"],"succeeded",complete)
        self.assertEqual(complete["result"]["scene"]["scene_version"],1)
        snapshot = await self.client.snapshot({**IDENTITY,"scene_version":1,"anchor_observation_id":"obs-1"})
        self.assertEqual(snapshot["quality"]["artifact_resolution"],"verified_digest_closed")
        all_events = (await self.client.events())["events"]
        events = [item for item in all_events if item["type"].startswith("job.")]
        self.assertEqual([event["type"] for event in events],["job.queued","job.started","job.completed"])
        self.assertTrue(all("server_seq" not in event for event in events))
        self.assertTrue(all(event["run_id"]==IDENTITY["run_id"] for event in events))
        self.assertEqual(len({event["event_id"] for event in events}),3)
        self.assertIn("scene.updated",[item["type"] for item in all_events])
        self.assertIn("snapshot.created",[item["type"] for item in all_events])
        self.assertFalse((await self.client.capabilities())["worker_ready"])

    async def test_cancel_budget_and_restart(self):
        created = await self.client.update(self.request,"cancel")
        cancelled = await self.client.cancel(created["job_id"],IDENTITY)
        self.assertTrue(cancelled["cancel_requested"])
        terminal = await self.client.wait(created["job_id"])
        self.assertEqual(terminal["status"],"cancelled")
        self.assertTrue(terminal["stop_confirmed"])
        created = await self.client.update({**self.request,"budget":{"wall_time_s":0.001,"queue_time_s":0}},"budget")
        terminal = await self.client.wait(created["job_id"])
        self.assertIn(terminal["status"],("failed","expired"))
        self.assertIn(terminal["error"],("budget_exceeded","deadline_exceeded"))
        created = self.service.submit(self.request,"build_scene","recovery-test")
        self.service.tasks[created["job_id"]].cancel()
        await asyncio.gather(self.service.tasks[created["job_id"]],return_exceptions=True)
        self.service.transition(created,"running",attempt_token="old-worker")
        queued=self.service.submit(self.request,"build_scene","resume-queued")
        self.service.tasks[queued["job_id"]].cancel()
        await asyncio.gather(self.service.tasks[queued["job_id"]],return_exceptions=True)
        crash_jobs=[self.service.job("recovery-test"),self.service.job("resume-queued")]
        await self.runner.cleanup()
        self.service = SceneService(self.tmp)
        # Restore the durable rows a hard crash would have left, without invoking cleanup.
        with self.service.db:
            for crashed in crash_jobs:
                self.service.db.execute("INSERT OR REPLACE INTO scene_jobs VALUES (?,?)",(crashed["job_id"],canonical(crashed)))
        self.runner = web.AppRunner(self.service.app()); await self.runner.setup()
        job = self.service.job("recovery-test")
        self.assertEqual(job["status"],"failed")
        self.assertEqual(job["error"],"worker_lost_on_restart")
        await self.service.tasks["resume-queued"]
        self.assertEqual(self.service.job("resume-queued")["status"],"succeeded")

    async def test_worker_crash_and_queue_deadline_are_terminal(self):
        created=self.service.submit(self.request,"build_scene","kill-owned-worker")
        await asyncio.sleep(0)
        process=self.service.processes[created["job_id"]]
        self.assertTrue(process.is_alive())
        process.kill()
        failed=await self.client.wait(created["job_id"])
        self.assertEqual(failed["status"],"failed")
        self.assertEqual(failed["error"],"worker_crashed")
        await self.service.semaphore.acquire(); await self.service.semaphore.acquire()
        queued=self.service.submit({**self.request,"budget":{"queue_time_s":0.01,"wall_time_s":30}},"build_scene","queue-expiry")
        await asyncio.sleep(0.03)
        self.service.semaphore.release(); self.service.semaphore.release()
        expired=await self.client.wait(queued["job_id"])
        self.assertEqual(expired["status"],"expired")
        self.assertEqual(expired["error"],"deadline_exceeded")
        running=self.service.submit({**self.request,"budget":{"queue_time_s":30,"wall_time_s":0.001}},"build_scene","running-budget")
        failed=await self.client.wait(running["job_id"])
        self.assertEqual(failed["status"],"failed")
        self.assertEqual(failed["error"],"budget_exceeded")

    async def test_graceful_shutdown_cancels_running_and_queued(self):
        await self.service.semaphore.acquire()
        running=self.service.submit(self.request,"build_scene","shutdown-running")
        await asyncio.sleep(0)
        process=self.service.processes[running["job_id"]]
        self.assertTrue(process.is_alive())
        queued=self.service.submit(self.request,"build_scene","shutdown-queued")
        await asyncio.sleep(0)
        self.assertEqual(self.service.job(queued["job_id"])["status"],"queued")
        await self.service.stop_jobs()
        self.assertFalse(process.is_alive())
        self.assertFalse(self.service.processes)
        for identifier in (running["job_id"],queued["job_id"]):
            job=self.service.job(identifier)
            self.assertEqual(job["status"],"cancelled")
            self.assertTrue(job["cancel_requested"])
            self.assertTrue(job["stop_confirmed"])
            events=[json.loads(row[0]) for row in self.service.db.execute("SELECT body FROM events")]
            types=[event["type"] for event in events if event.get("correlation",{}).get("job_id")==identifier]
            self.assertEqual(types[-2:],["job.cancel_requested","job.cancelled"])
        with self.assertRaises(web.HTTPServiceUnavailable):
            self.service.submit(self.request,"build_scene")
        await self.runner.cleanup()
        await self.service.close()
        self.service=SceneService(self.tmp)
        self.runner=web.AppRunner(self.service.app()); await self.runner.setup()
        self.assertEqual(self.service.job(running["job_id"])["status"],"cancelled")
        self.assertEqual(self.service.job(queued["job_id"])["status"],"cancelled")
        self.assertFalse(self.service.tasks)

    async def test_upload_replay_and_unknown_generation(self):
        artifact = await self.client.upload(b"checked", "text", "text/plain")
        self.assertEqual(self.service.store.resolve_artifact(artifact["ref"])[1], b"checked")
        original = deepcopy(self.obs[0]); original["env_generation"] = 2
        await self.client.observation(original)
        self.assertEqual((await self.client.head({**IDENTITY, "env_generation": 2}))["scene_version"], 0)
        first = await self.client.observation(self.obs[0])
        second = await self.client.observation(self.obs[0])
        self.assertEqual(first, second)

    async def test_invalid_http_inputs_are_structured_and_never_admitted(self):
        before = self.service.store.event_cursor()
        invalid = [
            ("scene/update-jobs", {**self.request, "camera_id": False}),
            ("scene/update-jobs", {**self.request, "unknown_field": True}),
            ("scene/query", {**IDENTITY, "kind": "distance", "entity_id": "box", "point_m": [0, 0, 0]}),
            ("observations", {**self.obs[0], "observation_profile": "typo"}),
            ("rollout-jobs", {"world_kind": "observed", "snapshot_id": "none"}),
        ]
        async with ClientSession() as session:
            for route, body in invalid:
                with self.subTest(route=route, body=body):
                    async with session.post(self.client.url + "/v1/" + route, json=body,
                                            headers={"Idempotency-Key": "bad-" + route}) as response:
                        self.assertEqual(response.status, 422)
                        error = await response.json()
                        self.assertEqual(error["code"], "invalid_scene_request")
                        self.assertIn("field_path", error)
                        self.assertIn("error", error)  # Existing clients retain their string error.
            async with session.get(self.client.url + "/v1/events?after=-1") as response:
                self.assertEqual(response.status, 422)
                self.assertEqual((await response.json())["code"], "invalid_event_cursor")
            async with session.post(self.client.url + "/v1/rollout-jobs", json={"snapshot_id": "none"},
                                    headers={"Idempotency-Key": "unsupported"}) as response:
                self.assertEqual(response.status, 422)
                self.assertFalse((await response.json())["executed"])
        self.assertEqual(self.service.db.execute("SELECT COUNT(*) FROM scene_jobs").fetchone()[0], 0)
        self.assertEqual(self.service.store.event_cursor(), before)
        self.assertFalse(self.service.processes)

    async def test_http_geometry_values_and_response_boundary(self):
        base = dict(IDENTITY, observation_id="obs-0", camera_id="head")
        for kind, extra in (("camera_pose", {}), ("fk", {}), ("pixel_ray", {"pixel": [320, 240]})):
            response = await self.client.geometry(dict(base, kind=kind, **extra))
            if kind == "fk":
                self.assertEqual(response["status"], "unknown")
                self.assertEqual(response["reason"], "missing_robot_link_calibration")
                continue
            self.assertEqual(response["status"], "supported")
            self.assertIsInstance(response["value"], dict)
        projected = await self.client.geometry(dict(base, kind="project_points", points_m=[[0, 0, 2]]))
        finished = await self.client.wait(projected["job_id"])
        self.assertEqual(finished["status"], "succeeded", finished)
        self.assertEqual(finished["result"]["value"]["pixels"], [[320., 240.]])
        # A broken producer must not publish a valid-looking capabilities reply.
        original = self.service.scene.capabilities
        self.service.scene.capabilities = lambda: {**original(), "observed_execution": "enabled"}
        async with ClientSession() as session:
            async with session.get(self.client.url + "/v1/capabilities") as response:
                self.assertEqual(response.status, 500)
                self.assertEqual((await response.json())["code"], "invalid_scene_response")

    async def test_malformed_json_not_found_and_internal_errors_map_to_accurate_http_codes(self):
        async with ClientSession() as session:
            async with session.post(
                self.client.url + "/v1/observations",
                data=b"{broken_json",
                headers={"Content-Type": "application/json", "Idempotency-Key": "bad-json"},
            ) as resp_400:
                self.assertEqual(resp_400.status, 400)
                self.assertEqual((await resp_400.json())["code"], "invalid_json")

            async with session.get(self.client.url + "/v1/jobs/job-does-not-exist") as resp_job_404:
                self.assertEqual(resp_job_404.status, 404)
                self.assertEqual((await resp_job_404.json())["code"], "unknown_job")

            missing_snap = "snapshot-" + ("0" * 64)
            async with session.get(self.client.url + f"/v1/snapshots/{missing_snap}", params=IDENTITY) as resp_snap_404:
                self.assertEqual(resp_snap_404.status, 404)
                self.assertEqual((await resp_snap_404.json())["code"], "unknown_snapshot")

            async with session.get(self.client.url + "/v1/scene/versions/999", params=IDENTITY) as resp_ver_404:
                self.assertEqual(resp_ver_404.status, 404)
                self.assertEqual((await resp_ver_404.json())["code"], "unknown_scene_version")

            orig_head = self.service.scene.head
            self.service.scene.head = lambda _id: (_ for _ in ()).throw(TypeError("internal_bug_in_head"))
            try:
                async with session.get(self.client.url + "/v1/scene/head", params=IDENTITY) as resp_500:
                    self.assertEqual(resp_500.status, 500)
                    self.assertEqual((await resp_500.json())["code"], "internal_scene_error")
            finally:
                self.service.scene.head = orig_head


import tempfile
import unittest
from copy import deepcopy

import cv2
import numpy as np

from ni_rua.core.storage.store import Store
from ni_rua.perception.visual_feedback import compare_observations, track_region


class VisualFeedbackTest(unittest.TestCase):
    def test_translation_occlusion_and_source_boundaries(self):
        rng = np.random.default_rng(12)
        before = np.full((120, 160, 3), 80, np.uint8)
        before[30:80, 40:100] = rng.integers(0, 255, (50, 60, 3), dtype=np.uint8)
        after = cv2.warpAffine(before, np.float32([[1, 0, 7], [0, 1, -4]]), (160, 120), borderValue=(80, 80, 80))
        roi = [42, 32, 56, 46]
        result = track_region(before, after, roi)
        self.assertEqual(result["status"], "tracked_features")
        np.testing.assert_allclose(result["displacement_px"], [7, -4], atol=.25)
        self.assertNotEqual(track_region(before, np.full_like(before, 80), roi)["status"], "tracked_features")
        self.assertEqual(track_region(before, after, roi, max_motion_px=2)["status"], "ambiguous")
        for bad in ([True, 0, 10, 10], [0, 0, 0, 10], [100, 100, 99, 99]):
            with self.assertRaises(ValueError):
                track_region(before, after, bad)
        identity = dict(run_id="test", episode_id="ep", env_instance_id="env", env_generation=1)
        directory = self.enterContext(tempfile.TemporaryDirectory())
        store = Store(directory, identity)
        self.addCleanup(store.close)
        for i, image in enumerate((before, after)):
            ref = store.put_artifact(cv2.imencode(".png", image)[1].tobytes(), "rgb", "image/png")
            bundle = dict(**identity, observation_id=f"obs{i}", observation_profile="rgb_proprio_v1",
                          sim_step=i, step_domain="test_control_step", cameras=[dict(camera_id="head",
                          frame_id=f"frame{i}", rgb_ref=ref, width=160, height=120)])
            store.save("observation", f"obs{i}", bundle)
        args = dict(before_observation_id="obs0", after_observation_id="obs1", camera_id="head", roi=roi)
        result = compare_observations(store, identity, **args)
        self.assertEqual(result["target"]["status"], "tracked_features")
        self.assertEqual(len(result["sources"]), 2)
        with self.assertRaisesRegex(ValueError, "identity"):
            compare_observations(store, {**identity, "env_generation": 2}, **args)
        with self.assertRaisesRegex(ValueError, "order"):
            compare_observations(store, identity, **{**args, "before_observation_id": "obs1", "after_observation_id": "obs0"})
        for index in (0, 1):
            calibrated = {**store.read("observation", f"obs{index}"),
                          "observation_id": f"calibrated{index}",
                          "observation_profile": "rgb_proprio_calibrated_v1",
                          "camera_calibration": {"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}}
            store.save("observation", calibrated["observation_id"], calibrated)
        calibrated_result = compare_observations(store, identity, **{**args,
            "before_observation_id": "calibrated0", "after_observation_id": "calibrated1"})
        self.assertEqual(calibrated_result["target"]["status"], "tracked_features")
        for suffix, changes in (("calibrated", {"observation_profile": "rgb_proprio_calibrated_v1"}),
                                ("private", {"cameras": [dict(camera_id="head", width=160, height=120,
                                 rgb_ref=store.put_artifact(cv2.imencode('.png', before)[1].tobytes(), "rgb", "image/png", "observer"))]})):
            value = {**deepcopy(bundle), **changes, "observation_id": suffix}
            store.save("observation", suffix, value)
            with self.assertRaises(ValueError):
                compare_observations(store, identity, **{**args, "after_observation_id": suffix})


if __name__ == "__main__":
    unittest.main()
