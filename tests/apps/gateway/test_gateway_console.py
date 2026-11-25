"""Consolidated tests:

- test_console_model_free_text: Operator console set_model: free-text model ids (conservative MODEL_ID check instead of a fixed allowlist), session
- test_gateway_console: Operator console: lease, safety commands, resume semantics and telemetry staleness.
"""

from functools import lru_cache
import json
import unittest
from pathlib import Path
from ni_rua.apps.gateway.extensions.operator_console import MODEL_ID, NO_IMAGE_INPUT_LABEL
from ni_rua.core.config import Config
from ni_rua.cortex.agents.base import AgentCapabilityError, AgentSession
from support.paths import REPO_ROOT, WEB_ROOT, isolate_home
import asyncio
from datetime import datetime, timedelta, timezone
import shutil
import tempfile
import threading
from types import SimpleNamespace
from unittest import mock
from aiohttp.test_utils import TestClient, TestServer
from ni_rua.apps.gateway.extensions.operator_console import TELEMETRY_SCHEMA, telemetry_view, redact
from ni_rua.apps.gateway.workbench import Workbench


def setUpModule():
    isolate_home()   # a fresh Workbench opens the grasp memory store at its '~' default


NI_RUA_TEST_TIER = "component"

# --- test_gateway_console.py ---
SOURCE_PAGES = WEB_ROOT


class FakeRobot:
    def __init__(self, run_dir, supported=True):
        self.run_dir = run_dir
        self.calls = []
        self.results = {}
        self.block = None
        self.entered = None
        if not supported:
            self.operator = None

    def operator(self, command, progress_callback=None, **arguments):
        if progress_callback is not None:
            progress_callback({"phase": "moving", "pose_index": 1, "pose_count": 13, "kept_count": 0})
        if self.block is not None and command in ("go_home", "calibrate"):
            if self.entered is not None:
                self.entered.set()
            self.block.wait(5)
        self.calls.append((command, arguments))
        result = self.results.get(command)
        if isinstance(result, Exception):
            raise result
        return dict(result or dict(status="completed", stop_confirmed=True), **arguments)

    def close(self):
        return None


def telemetry_state(captured_at, **extra):
    return dict(schema=TELEMETRY_SCHEMA, seq=7, captured_at=captured_at, monotonic_s=12.5, connected=True,
                torque_enabled=True, q_rad={"shoulder_pan": 0.1}, q_goal_rad=None, gripper_normalized=0.5,
                tcp_position_m=[0.2, 0.0, 0.1], tcp_quaternion_xyzw=[0, 0, 0, 1], motor={},
                motion=dict(active_action_id=None, phase="idle", tracking_error_rad=None),
                cameras={"wrist": dict(ok=True, fps=5.0, last_frame_at=captured_at, error=None)}, faults=[], **extra)


@lru_cache(maxsize=1)
def _cached_base_config():
    return Config.load()


class ConsoleCheck(unittest.IsolatedAsyncioTestCase):
    @property
    def _base_config(self):
        return _cached_base_config()

    async def asyncSetUp(self):
        # Pages come from the source web/ directory, not a web/dist build: these tests check routing, redirects and
        # the console/calibration/records navigation links, which the build (it only versions script and style
        # URLs) leaves unchanged. A fresh checkout has no build until `npm --prefix web run build` runs, so
        # depending on web/dist would fail (404) or never run there.
        for module in ("routes", "extensions.operator_console", "extensions.calibration"):
            pages = mock.patch(f"ni_rua.apps.gateway.{module}.static_directory", return_value=SOURCE_PAGES)
            pages.start()
            self.addCleanup(pages.stop)
        self.temporary_dir = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(self.temporary_dir)
        self.robot_dir = root / "robot-run"
        self.robot = FakeRobot(self.robot_dir)
        self.workbench = Workbench(root / "synthetic-console", robot=self.robot, config=self._base_config)
        self.workbench.starting, self.workbench.status = False, "running"
        self.turns = []
        session = SimpleNamespace(thread_id="fixture-thread", turn_id=None, state="completed", effort="high")

        def run_turn(prompt):
            self.turns.append(prompt)
            session.turn_id, session.state = "fixture-turn", "running"
            return session.turn_id

        session.run_turn = run_turn
        self.steers = []

        def steer(text):
            if session.state != "running":
                raise ValueError("no_active_turn_to_steer")
            self.steers.append(text)
            return session.turn_id

        session.steer = steer
        session.interrupt = lambda: setattr(session, "state", "interrupted")
        session.close = lambda: setattr(session, "state", "closed")
        self.workbench.session = session
        self.observations = 0

        def observe(correlation=None):
            self.observations += 1
            observation = dict(observation_id=f"obs-fresh-{self.observations}")
            self.workbench.harness.latest = observation
            return observation

        self.workbench.harness.observe = observe
        self.workbench.harness.capabilities = {"robot_model_id": "so101"}
        self.client = TestClient(TestServer(self.workbench.app()))
        await self.client.start_server()
        self.sequence = 0

    async def test_so101_home_and_explicit_review_keep_calibration_reachable(self):
        response = await self.client.get("/", allow_redirects=False)
        self.assertEqual(response.status, 302)
        self.assertEqual(response.headers["Location"], "/console")
        for route, title in (("/console", "真机工作台"), ("/calibration", "标定向导"), ("/index.html", "轨迹与记录")):
            response = await self.client.get(route)
            self.assertEqual(response.status, 200, route)
            html = await response.text()
            self.assertIn(title, html)
            self.assertIn('href="/calibration"', html)
            self.assertIn('href="/console"', html)
        self.assertEqual(self.robot.calls, [])
        self.assertIsNone(self.workbench.controller, "page GETs must never take the writer lease")
        self.workbench.replay = True
        response = await self.client.get("/", allow_redirects=False)
        self.assertEqual(response.status, 200, "saved SO-101 must remain a review workspace")
        self.workbench.replay = False
        self.workbench.harness.capabilities = {"robot_model_id": "arx-x5"}
        response = await self.client.get("/", allow_redirects=False)
        self.assertEqual(response.status, 200, "simulation workbench remains the default")

    async def asyncTearDown(self):
        await self.client.close()
        await self.workbench.close()

    async def control(self, command, request_id=None, **fields):
        self.sequence += 1
        body = dict(command=command, request_id=request_id or f"req-{self.sequence}", **fields)
        response = await self.client.post("/v1/control", json=body, headers={"Origin": f"http://{self.client.host}:{self.client.port}"})
        return response.status, await response.json()

    async def claim(self):
        status, body = await self.control("claim_input", client_id="console-a")
        self.assertEqual(status, 200)
        return dict(client_id="console-a", writer_token=body["writer_token"])

    async def test_save_video_seals_composes_rearms_and_serves_one_mp4(self):
        if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
            self.skipTest("ffmpeg/ffprobe required")
        from support.so101_gateway_fixtures import BLUE, RED, sealed_recording
        report = sealed_recording(self.robot_dir, [("act-1", [("fixed", 0.0, RED), ("wrist", 0.0, BLUE), ("fixed", 1 / 30, RED)])])
        calls = []
        self.robot.save_video = lambda: calls.append("save") or report
        self.robot.record = lambda enabled=True: calls.append(("record", enabled)) or dict(enabled=enabled, error=None)
        self.workbench.harness.capabilities = {"robot_model_id": "so101", "observer_video": True}
        status, body = await self.control("save_video")
        self.assertEqual(status, 400, body)
        self.assertIn("input_owner_claim_required", body["error"])
        lease = await self.claim()
        self.workbench.session.state = "running"
        status, body = await self.control("save_video", **lease)
        self.assertEqual(status, 400, body)
        self.assertIn("observer_operation_requires_idle_run", body["error"])
        self.assertEqual(calls, [], "a running turn must not seal the recording")
        self.workbench.session.state = "completed"
        status, body = await self.control("save_video", **lease)
        self.assertEqual(status, 200, body)
        self.assertEqual(body["status"], "complete")
        self.assertEqual(calls, ["save", ("record", True)], "seal first, then re-arm for later tasks")
        self.assertTrue(body["recording_rearmed"])
        composite = body["composite"]
        self.assertEqual((composite["camera_order"], composite["frames"], composite["segments"]), (["fixed", "wrist"], 2, 1))
        self.assertEqual(composite["header_hud_height"], 40)
        self.assertEqual(composite["timeline_hud_height"], 132)
        self.assertIsInstance(composite["header_summary"], dict)
        self.assertTrue(composite["header_summary"]["label"])
        response = await self.client.get("/v1/artifacts/" + composite["video_ref"].removeprefix("artifact:"))
        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers["Content-Type"], "video/mp4")
        self.assertEqual(await response.read(), Path(composite["video_path"]).read_bytes())
        video = json.loads((self.workbench.root / "video.json").read_text())
        self.assertEqual(video["composite"]["video_ref"], composite["video_ref"])
        self.assertTrue(all("video_ref" in item for item in video["videos"]))
        state = await (await self.client.get("/v1/console/state")).json()
        self.assertEqual(state["video_export"]["request_id"], body["request_id"])
        types = [event["type"] for event in self.workbench.store.events(0)]
        for kind in ("observer.video_exported", "observer.video_composed", "operator.save_video"):
            self.assertIn(kind, types)
        self.assertEqual(self.workbench.status, "running")
        # A second click with no new action seals an empty recording; video.json keeps the saved composite.
        empty = sealed_recording(self.robot_dir / "second", [])
        self.robot.save_video = lambda: calls.append("save") or empty
        status, body = await self.control("save_video", **lease)
        self.assertEqual((status, body["status"]), (200, "empty"))
        self.assertEqual(json.loads((self.workbench.root / "video.json").read_text())["composite"]["video_ref"], composite["video_ref"])

    async def test_save_video_failure_is_reported_without_unknown_state(self):
        def fail():
            raise RuntimeError("so101_worker_timeout:save_video")
        self.robot.save_video = fail
        self.robot.record = lambda enabled=True: dict(enabled=enabled)
        lease = await self.claim()
        status, body = await self.control("save_video", **lease)
        self.assertEqual(status, 400, body)
        self.assertIn("observer_video_unavailable", body["error"])
        self.workbench.harness.capabilities = {"robot_model_id": "so101", "observer_video": True}
        status, body = await self.control("save_video", **lease)
        self.assertEqual(status, 200, body)
        self.assertEqual(body["status"], "failed")
        self.assertIn("so101_worker_timeout", body["error"])
        self.assertEqual(self.workbench.status, "running", "a video failure is not a physical unknown")
        self.assertEqual(self.workbench.unknown_items(), [])
        self.assertFalse((self.workbench.root / "video.json").exists())

    async def test_steer_joins_running_turn_and_needs_lease(self):
        lease = await self.claim()
        status, body = await self.control("steer", prompt="先别抓，先看一下橙色方块", **lease)
        self.assertEqual(status, 400, body)
        self.assertIn("no_active_turn_to_steer", body["error"])
        status, body = await self.control("prompt", prompt="把蓝色方块放到橙色方块上", **lease)
        self.assertEqual(status, 200, body)
        status, body = await self.control("prompt", prompt="second turn", **lease)
        self.assertEqual(status, 400, body)  # a running turn still refuses a NEW turn
        status, body = await self.control("steer", prompt="先别抓，先看一下橙色方块", **lease)
        self.assertEqual((status, body["status"], body["turn_id"]), (200, "steered", "fixture-turn"))
        self.assertEqual(self.steers, ["先别抓，先看一下橙色方块"])
        self.assertEqual(self.turns, ["把蓝色方块放到橙色方块上"])
        status, body = await self.control("steer", prompt="x")  # browser origin without lease
        self.assertEqual(status, 409, body)
        self.assertEqual(len(self.steers), 1)

    async def test_service_tier_switch_probes_provider_and_reports_unsupported(self):
        probes = []

        class Proxy:
            service_tier, service_tier_probes = None, {}

            def probe_service_tier(self, model, tier):
                probes.append(tier)
                return dict(tier=tier, supported=tier == "priority", served_tier=tier if tier == "priority" else "default", error=None)

        self.workbench.session.model_proxy, self.workbench.session.model = Proxy(), "gpt-6.1-sol"
        lease = await self.claim()
        status, body = await self.control("set_service_tier", tier="ultrafast", **lease)
        self.assertEqual((status, body["status"]), (200, "unsupported"), body)
        self.assertIn("不支持", body["message"])
        self.assertIsNone(self.workbench.session.model_proxy.service_tier)  # unchanged
        status, body = await self.control("set_service_tier", tier="priority", **lease)
        self.assertEqual(body["status"], "applied")
        self.assertEqual(self.workbench.session.model_proxy.service_tier, "priority")
        status, body = await self.control("set_service_tier", tier="default", **lease)
        self.assertEqual(body["status"], "applied")
        self.assertIsNone(self.workbench.session.model_proxy.service_tier)
        self.assertEqual(probes, ["ultrafast", "priority"])  # default never probed
        status, body = await self.control("set_service_tier", tier="turbo", **lease)
        self.assertEqual(status, 400)

    async def test_service_tier_probe_error_preserves_selection_and_allows_retry(self):
        class Proxy:
            service_tier, service_tier_probes = "ultrafast", {}
            attempts = 0

            def probe_service_tier(self, model, tier):
                self.attempts += 1
                return dict(tier=tier, supported=None if self.attempts == 1 else True,
                            served_tier=None if self.attempts == 1 else "default",
                            error="HTTP 403: service_tier acceptance unknown" if self.attempts == 1 else None)

        proxy = self.workbench.session.model_proxy = Proxy()
        self.workbench.session.model = "gpt-6.1-sol"
        lease = await self.claim()
        status, body = await self.control("set_service_tier", tier="priority", **lease)
        self.assertEqual((status, body["status"]), (200, "unknown"), body)
        self.assertIn("无法确认", body["message"])
        self.assertNotIn("不支持", body["message"])
        self.assertEqual(proxy.service_tier, "ultrafast")
        status, body = await self.control("set_service_tier", tier="priority", **lease)
        self.assertEqual((status, body["status"]), (200, "applied"), body)
        self.assertEqual(proxy.service_tier, "priority")
        self.assertEqual(proxy.attempts, 2)

    async def test_service_tier_probe_on_unavailable_model_names_the_model_not_the_tier(self):
        # Regression: the console offered a model ID the provider does not serve; every tier probe
        # then got HTTP 503 model_not_found and the UI only said "service_tier acceptance unknown".
        class Proxy:
            service_tier, service_tier_probes = None, {}

            def probe_service_tier(self, model, tier):
                return dict(tier=tier, supported=None, served_tier=None,
                            error="HTTP 503 model_not_found: service_tier acceptance unknown",
                            provider_error=dict(http_status=503, code="model_not_found", type="new_api_error",
                                                classification="configuration_error"))

        proxy = self.workbench.session.model_proxy = Proxy()
        self.workbench.session.model = "gpt-6.1-luna"
        lease = await self.claim()
        status, body = await self.control("set_service_tier", tier="priority", **lease)
        self.assertEqual((status, body["status"], body["model"]), (200, "unknown", "gpt-6.1-luna"), body)
        self.assertIn("gpt-6.1-luna", body["message"])
        self.assertIn("model_not_found", body["message"])
        self.assertIn("与档位无关", body["message"])
        self.assertNotIn("不支持", body["message"])
        self.assertIsNone(proxy.service_tier)

    async def test_hold_needs_no_lease_and_closes_admission(self):
        status, body = await self.control("hold")
        self.assertEqual(status, 200, body)
        self.assertEqual((body["status"], body["admission"]), ("completed", "closed"))
        self.assertFalse(self.workbench.harness.accepting)
        self.assertEqual(self.robot.calls, [("hold", {})])

    async def test_mutating_commands_require_lease_from_browser(self):
        for command, fields in (("torque_on", {}), ("go_home", {}), ("resume", {}),
                                ("set_runtime_limits", {"speed_scale_cap": .5}), ("relax", {"confirm": "relax"})):
            status, body = await self.control(command, **fields)
            self.assertEqual((status, body.get("error")), (400, "input_owner_claim_required"), command)
        self.assertEqual(self.robot.calls, [])
        lease = await self.claim()
        status, body = await self.control("torque_on", client_id="intruder", writer_token="wrong")
        self.assertEqual(body.get("error"), "input_owner_conflict")
        status, body = await self.control("set_runtime_limits", speed_scale_cap=.5, **lease)
        self.assertEqual((status, body["speed_scale_cap"]), (200, .5))

    async def test_relax_requires_explicit_confirm_and_latches_torque_off(self):
        lease = await self.claim()
        status, body = await self.control("relax", **lease)
        self.assertEqual((status, body["error"]), (400, "relax_requires_confirm:relax"))
        self.assertEqual(self.robot.calls, [])
        status, body = await self.control("relax", confirm="relax", **lease)
        self.assertEqual((status, body["torque"], body["admission"]), (200, "off", "closed"))
        status, body = await self.control("resume", **lease)
        self.assertIn("torque_off_requires_torque_on", body["error"])
        status, body = await self.control("go_home", **lease)
        self.assertEqual(body["error"], "go_home_requires_torque_on")
        status, body = await self.control("torque_on", **lease)
        self.assertEqual(body["torque"], "on")
        status, body = await self.control("resume", **lease)
        self.assertEqual((status, body["status"]), (200, "resumed"))

    async def test_unsupported_backend_returns_clear_error(self):
        self.robot.operator = None
        status, body = await self.control("hold")
        self.assertEqual((status, body["error"]), (400, "operator_command_unsupported:hold"))
        # Admission still closed: a failed hold must not leave motion open.
        self.assertFalse(self.workbench.harness.accepting)
        status, body = await self.control("estop")
        self.assertEqual(status, 200)
        self.assertEqual(body["hold"]["status"], "unsupported")
        self.assertEqual(body["stop"]["status"], "stopped")

    async def test_idempotent_request_id_does_not_resend(self):
        first = await self.control("hold", request_id="same-hold")
        second = await self.control("hold", request_id="same-hold")
        self.assertEqual(first, second)
        self.assertEqual(len(self.robot.calls), 1)

    async def test_stop_then_resume_reopens_admission_with_fresh_observation(self):
        lease = await self.claim()
        status, body = await self.control("stop")
        self.assertEqual(body["status"], "stopped")
        self.assertFalse(self.workbench.harness.accepting)
        status, body = await self.control("prompt", prompt="continue", **lease)
        self.assertEqual(body["error"], "run_not_ready_for_new_task")
        status, body = await self.control("resume", **lease)
        self.assertEqual((status, body["status"], body["observation_id"]), (200, "resumed", "obs-fresh-1"))
        self.assertTrue(self.workbench.harness.accepting)
        status, body = await self.control("prompt", prompt="continue", **lease)
        self.assertEqual((status, body["status"]), (200, "running"))
        self.assertEqual(self.turns, ["continue"])
        # A running turn blocks resume.
        status, body = await self.control("resume", **lease)
        self.assertIn("codex_turn_running", body["error"])

    async def test_stop_during_resume_observation_wins(self):
        lease = await self.claim()
        await self.control("hold")
        harness = self.workbench.harness

        def racing_observe(correlation=None):
            harness.request_stop()
            return dict(observation_id="obs-race")

        harness.observe = racing_observe
        status, body = await self.control("resume", **lease)
        self.assertEqual(body["error"], "resume_conflict_stop_requested_during_observation")
        self.assertFalse(harness.accepting)

    async def test_unknown_states_require_explicit_acknowledgement(self):
        lease = await self.claim()
        self.robot.results["hold"] = dict(status="stopped", stop_confirmed=False, reason="velocity_not_settled")
        status, body = await self.control("hold", request_id="hold-1")
        self.assertEqual(body["status"], "unknown")
        harness = self.workbench.harness
        harness.store.save("action_request", "act-1", dict(action_id="act-1", **self.workbench.identity))
        harness.states["act-1"] = dict(status="unknown", reason="worker_acknowledgement_missing")
        state = await (await self.client.get("/v1/console/state")).json()
        self.assertEqual(sorted(state["resume"]["requires_acknowledgement"]), ["act-1", "operator:hold-1"])
        status, body = await self.control("resume", **lease)
        self.assertTrue(body["error"].startswith("resume_requires_unknown_acknowledgement:"))
        status, body = await self.control("resume", acknowledge_unknown=["act-1"], **lease)
        self.assertTrue(body["error"].startswith("resume_requires_unknown_acknowledgement:"))
        self.assertFalse(harness.accepting)
        status, body = await self.control("resume", acknowledge_unknown=["operator:hold-1", "act-1"], **lease)
        self.assertEqual((status, body["status"]), (200, "resumed"))
        self.assertTrue(harness.accepting)
        # The receipt itself stays unknown for archive and finalization.
        self.assertEqual(harness.states["act-1"]["status"], "unknown")
        self.assertTrue(self.workbench.current_unknown_actions())
        state = await (await self.client.get("/v1/console/state")).json()
        self.assertEqual(state["resume"]["requires_acknowledgement"], [])

    async def test_operator_transport_failure_is_unknown(self):
        lease = await self.claim()
        self.robot.results["torque_on"] = RuntimeError("serial_timeout")
        status, body = await self.control("torque_on", **lease)
        self.assertEqual((status, body["status"]), (409, "unknown"))
        state = await (await self.client.get("/v1/console/state")).json()
        self.assertEqual(state["operator_gate"]["torque"], "unknown")
        self.assertIn("operator:", state["resume"]["requires_acknowledgement"][0])

    async def test_go_home_requires_idle_and_closes_admission(self):
        lease = await self.claim()
        self.workbench.session.state = "running"
        status, body = await self.control("go_home", **lease)
        self.assertEqual(body["error"], "go_home_requires_idle_turn_and_motion")
        self.workbench.session.state = "completed"
        status, body = await self.control("go_home", **lease)
        self.assertEqual((status, body["world"], body["admission"]), (200, "objects_not_reset", "closed_until_resume"))
        self.assertFalse(self.workbench.harness.accepting)

    async def test_hold_preempts_running_go_home(self):
        lease = await self.claim()
        self.robot.entered = threading.Event()
        self.robot.block = threading.Event()
        home = asyncio.create_task(self.control("go_home", **lease))
        self.assertTrue(await asyncio.to_thread(self.robot.entered.wait, 5))
        status, body = await asyncio.wait_for(self.control("hold"), 2)
        self.assertEqual(status, 200)
        self.robot.block.set()
        await home
        self.assertEqual([call[0] for call in self.robot.calls], ["hold", "go_home"])

    async def test_lease_does_not_change_hands_while_go_home_runs(self):
        # Any command holding the control lock, not only calibration steps: lease commands are answered at once,
        # but only the lease's last holder may claim it until the command ends (input_owner_busy_step).
        lease = await self.claim()
        self.robot.entered = threading.Event()
        self.robot.block = threading.Event()
        home = asyncio.create_task(self.control("go_home", **lease))
        try:
            self.assertTrue(await asyncio.to_thread(self.robot.entered.wait, 5))
            self.assertTrue((await (await self.client.get("/v1/run")).json())["control_busy"])
            status, body = await asyncio.wait_for(self.control("claim_input", client_id="console-b"), 2)
            self.assertEqual((status, body["error"]), (409, "input_owner_conflict"))
            self.workbench.controller["expires_at"] = 0  # the holder's lease lapses mid-command
            status, body = await asyncio.wait_for(self.control("claim_input", client_id="console-b"), 2)
            self.assertEqual(status, 409)
            self.assertTrue(body["error"].startswith("input_owner_busy_step:"), body)
            status, body = await asyncio.wait_for(self.control("claim_input", client_id="console-a"), 2)
            self.assertEqual((status, body["status"]), (200, "owned"))
            status, body = await asyncio.wait_for(self.control("estop"), 5)  # safety stays lease-free
            self.assertNotIn("error", body)
        finally:
            self.robot.block.set()
            await home
        self.assertFalse((await (await self.client.get("/v1/run")).json())["control_busy"])
        self.workbench.controller["expires_at"] = 0
        status, body = await self.control("claim_input", client_id="console-b")
        self.assertEqual((status, body["status"]), (200, "owned"))

    async def test_calibrate_gating_and_admission_reopen(self):
        lease = await self.claim()
        self.workbench.harness.accepting = True
        self.workbench.session.state = "running"
        for command in ("calibrate", "reload_calibration"):
            status, body = await self.control(command, **lease)
            self.assertEqual((status, body["error"]), (400, command + "_requires_idle_turn_and_motion"))
        self.workbench.session.state = "completed"
        self.workbench.harness.active_action = "act-9"
        status, body = await self.control("calibrate", **lease)
        self.assertEqual(body["error"], "calibrate_requires_idle_turn_and_motion")
        self.workbench.harness.active_action = None
        self.assertEqual(self.robot.calls, [])
        self.assertTrue(self.workbench.harness.accepting)
        status, body = await self.control("calibrate")  # another browser client while the lease is held
        self.assertEqual(body["error"], "input_owner_conflict")

        self.robot.results["calibrate"] = dict(status="calibrated", physics_advanced=True,
                                               residuals=dict(hand_eye_translation_rms_m=0.003))
        self.robot.entered = threading.Event()
        self.robot.block = threading.Event()
        task = asyncio.create_task(self.control("calibrate", **lease))
        self.assertTrue(await asyncio.to_thread(self.robot.entered.wait, 5))
        # Admission is closed while the worker calibrates; progress is visible to the console.
        self.assertFalse(self.workbench.harness.accepting)
        state = await (await self.client.get("/v1/console/state")).json()
        self.assertTrue(state["calibration"]["running"])
        self.assertEqual(state["calibration"]["progress"][0]["pose_index"], 1)
        self.robot.block.set()
        status, body = await task
        self.assertEqual((status, body["status"], body["admission"], body["observation_id"]),
                         (200, "calibrated", "open", "obs-fresh-1"))
        self.assertEqual(body["world"], "objects_not_reset")
        self.assertTrue(self.workbench.harness.accepting)
        state = await (await self.client.get("/v1/console/state")).json()
        self.assertFalse(state["calibration"]["running"])
        self.assertEqual(state["calibration"]["result"]["status"], "calibrated")

        status, body = await self.control("reload_calibration", **lease)
        self.assertEqual((status, body["admission"]), (200, "open"))
        self.assertEqual([call[0] for call in self.robot.calls], ["calibrate", "reload_calibration"])

    async def test_calibrate_failure_or_prior_stop_keeps_admission_closed(self):
        lease = await self.claim()
        self.workbench.harness.accepting = False  # stopped before: calibration never reopens it
        self.robot.results["calibrate"] = dict(status="gate_failed", physics_advanced=True)
        status, body = await self.control("calibrate", **lease)
        self.assertEqual((status, body["status"], body["admission"]), (200, "gate_failed", "closed_until_resume"))
        self.assertFalse(self.workbench.harness.accepting)
        self.workbench.harness.accepting = True
        self.robot.results["calibrate"] = dict(status="unknown", reason="backend_error", physics_advanced=True,
                                               stop_confirmation=dict(confirmed=False))
        status, body = await self.control("calibrate", request_id="cal-unknown", **lease)
        self.assertEqual((body["status"], body["admission"]), ("unknown", "closed_until_resume"))
        self.assertFalse(self.workbench.harness.accepting)
        self.assertIn("operator:cal-unknown", (await (await self.client.get("/v1/console/state")).json())["resume"]["requires_acknowledgement"])
        status, body = await self.control("calibrate", **lease)
        self.assertEqual(body["error"], "calibrate_requires_unknown_acknowledged_via_resume")

    async def test_hold_preempts_running_calibration(self):
        lease = await self.claim()
        self.workbench.harness.accepting = True
        self.robot.entered = threading.Event()
        self.robot.block = threading.Event()
        task = asyncio.create_task(self.control("calibrate", **lease))
        self.assertTrue(await asyncio.to_thread(self.robot.entered.wait, 5))
        status, body = await asyncio.wait_for(self.control("hold"), 2)
        self.assertEqual(status, 200)
        self.robot.block.set()
        status, body = await task
        self.assertEqual(body["admission"], "closed_until_resume")  # a stop arrived meanwhile: never reopened
        self.assertFalse(self.workbench.harness.accepting)

    async def test_handheld_capture_and_solve_gating_state_and_preview(self):
        status, body = await self.control("handheld_capture", action="start")  # browser client without the lease
        self.assertEqual((status, body["error"]), (400, "input_owner_claim_required"))
        lease = await self.claim()
        self.workbench.session.state = "running"
        for command, fields in (("handheld_capture", dict(action="start")), ("handheld_solve", {})):
            status, body = await self.control(command, **fields, **lease)
            self.assertEqual((status, body["error"]), (400, command + "_requires_idle_turn_and_motion"))
        self.workbench.session.state = "completed"
        self.workbench.harness.active_action = "act-3"
        status, body = await self.control("handheld_capture", action="start", **lease)
        self.assertEqual(body["error"], "handheld_capture_requires_idle_turn_and_motion")
        self.workbench.harness.active_action = None
        status, body = await self.control("handheld_capture", action="status", **lease)
        self.assertEqual(body["error"], "invalid_handheld_capture_action: expected start or stop")
        self.assertEqual(self.robot.calls, [])

        self.robot.results["handheld_capture"] = dict(status="running", kept=0)
        status, body = await self.control("handheld_capture", action="start", **lease)
        self.assertEqual((status, body["status"]), (200, "running"))
        self.assertEqual(self.robot.calls[-1], ("handheld_capture", {"action": "start"}))
        directory = self.robot_dir / "telemetry"
        directory.mkdir(parents=True)
        (directory / "handheld.json").write_text(json.dumps(dict(status="running", kept=4, target_views=20,
                                                                 guidance=["板子拿稳"])))
        (directory / "fixed_corners.jpg").write_bytes(b"\xff\xd8corners")
        state = await (await self.client.get("/v1/console/state")).json()
        handheld = state["calibration"]["handheld"]
        self.assertEqual((handheld["capture"]["kept"], handheld["capture"]["stale"], handheld["solving"]), (4, False, False))
        frame = await self.client.get("/v1/console/cameras/fixed_corners.jpg")
        self.assertEqual((frame.status, await frame.read()), (200, b"\xff\xd8corners"))

        self.workbench.harness.accepting = True
        self.robot.results["handheld_solve"] = dict(status="calibrated", rms_px=0.31, gate_px=1.0, passed=True,
                                                    extrinsic_updated=False, physics_advanced=False)
        status, body = await self.control("handheld_solve", **lease)
        self.assertEqual((status, body["status"], body["admission"]), (200, "calibrated", "open"))
        state = await (await self.client.get("/v1/console/state")).json()
        self.assertEqual(state["calibration"]["handheld"]["solve_result"]["rms_px"], 0.31)
        self.assertFalse(state["calibration"]["handheld"]["solving"])

    async def test_fixed_corner_preview_missing_is_404(self):
        self.assertEqual((await self.client.get("/v1/console/cameras/fixed_corners.jpg")).status, 404)
        state = await (await self.client.get("/v1/console/state")).json()
        self.assertIsNone(state["calibration"]["handheld"]["capture"])

    async def test_board_check_endpoint_is_read_only_and_lease_free(self):
        self.robot.results["board_check"] = dict(status="checked", ok=False, corners=dict(wrist=24, fixed=3), min_board_corners=8)
        body = await (await self.client.get("/v1/console/board_check")).json()
        self.assertEqual((body["available"], body["ok"], body["corners"]["fixed"]), (True, False, 3))
        self.robot.operator = None
        body = await (await self.client.get("/v1/console/board_check")).json()
        self.assertEqual((body["available"], body["reason"]), (False, "operator_command_unsupported:board_check"))

    async def test_effort_and_speed_validation(self):
        lease = await self.claim()
        status, body = await self.control("set_effort", effort="turbo", **lease)
        self.assertEqual(body["error"], "invalid_effort")
        status, body = await self.control("set_effort", effort="low", **lease)
        self.assertEqual((status, self.workbench.session.effort), (200, "low"))
        for cap in (0.05, 1.5, True, "0.5"):
            status, body = await self.control("set_runtime_limits", speed_scale_cap=cap, **lease)
            self.assertEqual(status, 400, cap)

    async def test_model_switch_and_default_speed_scale_cap(self):
        class Proxy:
            def __init__(self):
                self.service_tier = "ultrafast"
                self.by_model = {
                    "gpt-6.1-terra": {"ultrafast": {"tier": "ultrafast", "supported": True}},
                    "gpt-6-astra": {"ultrafast": {"tier": "ultrafast", "supported": False}},
                }
                self.service_tier_probes = self.by_model["gpt-6.1-terra"]

            def select_model(self, model):
                self.service_tier_probes = self.by_model.setdefault(model, {})
                if self.service_tier and self.service_tier_probes.get(self.service_tier, {}).get("supported") is False:
                    self.service_tier = None
                return self.service_tier_probes

        self.workbench.session.model = "gpt-6.1-terra"
        self.workbench.session.price_card = None
        self.workbench.session.model_proxy = Proxy()
        state = await (await self.client.get("/v1/console/state")).json()
        self.assertEqual(state["settings"]["runtime"]["speed_scale_cap"]["value"], 0.9)
        self.assertEqual(state["settings"]["runtime"]["speed_scale_cap"]["default"], 0.9)
        self.assertEqual(state["settings"]["runtime"]["model"]["value"], "gpt-6.1-terra")
        # No hardcoded suggestion list (design 16 section 6): a Codex-home run offers the pinned native catalogue's
        # listed models, plus the current model.
        choices = state["settings"]["runtime"]["model"]["choices"]
        self.assertEqual(choices[-1], "gpt-6.1-terra")
        self.assertIn("gpt-6.1-sol", choices)

        status, body = await self.control("set_model", model="gpt-6.1-sol")
        self.assertEqual((status, body.get("error")), (400, "input_owner_claim_required"))
        lease = await self.claim()
        for target in ("gpt-6.1-sol", "gpt-6-astra", "gpt-6-luna", "gpt-6.1-terra"):
            status, body = await self.control("set_model", model=target, **lease)
            self.assertEqual((status, body["status"], body["model"], self.workbench.session.model),
                             (200, "applied", target, target))
        # Switching to gpt-6-astra cleared the unsupported ultrafast tier and isolated probes per model.
        self.assertIsNone(self.workbench.session.model_proxy.service_tier)
        # Free text: any well-formed id is accepted (tests/test_console_model_free_text.py); an ill-formed one is not.
        status, body = await self.control("set_model", model="unknown model", **lease)
        self.assertEqual((status, body.get("error")), (400, "invalid_model"))
        self.workbench.session.price_card = {"input_per_million": 1.0}
        status, body = await self.control("set_model", model="gpt-6.1-sol", **lease)
        self.assertEqual((status, body.get("error")), (409, "set_model_conflicts_with_price_card"))

    async def test_telemetry_endpoints_and_staleness(self):
        from ni_rua.robot.backends.so101.backend import RobotBackend
        # The so101_real backend's console hook (URDF joint limits; pure, no worker) on the fake robot.
        self.robot.joint_limits_rad = lambda: RobotBackend.joint_limits_rad(None)
        missing = await (await self.client.get("/v1/console/telemetry")).json()
        self.assertEqual((missing["available"], missing["stale"], missing["reason"]), (False, True, "telemetry_unavailable"))
        directory = self.robot_dir / "telemetry"
        directory.mkdir(parents=True)
        now = datetime.now(timezone.utc)
        (directory / "state.json").write_text(json.dumps(telemetry_state(now.isoformat())))
        (directory / "wrist.jpg").write_bytes(b"\xff\xd8fixture")
        live = await (await self.client.get("/v1/console/telemetry")).json()
        self.assertEqual((live["available"], live["stale"]), (True, False))
        self.assertEqual(live["joint_limits_rad"]["gripper"][0], -0.174533)
        frame = await self.client.get("/v1/console/cameras/wrist.jpg")
        self.assertEqual((frame.status, frame.headers["Content-Type"], await frame.read()), (200, "image/jpeg", b"\xff\xd8fixture"))
        self.assertEqual((await self.client.get("/v1/console/cameras/fixed.jpg")).status, 404)
        self.assertEqual((await self.client.get("/v1/console/cameras/..%2Fstate.json")).status, 404)
        old = (now - timedelta(seconds=30)).isoformat()
        (directory / "state.json").write_text(json.dumps(telemetry_state(old)))
        stale = await (await self.client.get("/v1/console/telemetry")).json()
        self.assertEqual((stale["available"], stale["stale"], stale["reason"]), (True, True, "telemetry_stale"))
        self.assertGreater(stale["age_s"], 29)
        # Telemetry never enters the run store as an artifact or event.
        self.assertFalse(any("telemetry" in event["type"] for event in self.workbench.store.events(0)))

    def test_telemetry_view_rejects_bad_schema_and_future_clock(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            path.write_text(json.dumps(dict(schema="other")))
            self.assertEqual(telemetry_view(temporary)["reason"], "telemetry_schema_mismatch")
            path.write_text("{")
            self.assertEqual(telemetry_view(temporary)["reason"], "telemetry_unreadable")
            future = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
            path.write_text(json.dumps(telemetry_state(future)))
            view = telemetry_view(temporary)
            self.assertEqual((view["stale"], view["reason"]), (True, "telemetry_clock_ahead"))
            path.write_text(json.dumps(telemetry_state(None)))
            self.assertEqual(telemetry_view(temporary)["reason"], "telemetry_captured_at_missing")

    async def test_console_state_redacts_secrets(self):
        self.assertEqual(redact({"api_key": "sk-x", "nested": [{"auth_token": "t"}], "name": "m", "token": None}),
                         {"api_key": "[redacted]", "nested": [{"auth_token": "[redacted]"}], "name": "m", "token": None})
        state = await (await self.client.get("/v1/console/state")).json()
        self.assertIn("effective_config", state)
        self.assertEqual(state["settings"]["runtime"]["effort"]["value"], "high")
        self.assertTrue(state["operator_supported"])
        self.assertNotIn("sk-", json.dumps(state))

    async def test_operator_motion_triggers_external_motion_and_nonblocking_relax(self):
        lease = await self.claim()
        harness = self.workbench.harness
        harness._persistent_obstacles["o1"] = {"id": "o1"}
        harness._obs3d_memo = ("obs-0", {})
        status, body = await self.control("go_home", **lease)
        self.assertEqual((status, body["admission"]), (200, "closed_until_resume"))
        self.assertEqual(harness._persistent_obstacles, {})
        self.assertIsNone(harness._obs3d_memo)
        self.assertEqual(harness.latest, {"observation_id": "obs-fresh-1"})
        self.assertEqual(self.observations, 1)

        # Preemptive relax MUST NOT block behind a held robot_lock
        held = threading.Event()
        release = threading.Event()

        def hold_lock():
            with harness.robot_lock:
                held.set()
                release.wait(2.0)

        t = threading.Thread(target=hold_lock, daemon=True)
        t.start()
        self.assertTrue(held.wait(1.0))
        try:
            harness._persistent_obstacles["o2"] = {"id": "o2"}
            status, body = await asyncio.wait_for(self.control("relax", confirm="relax", **lease), 2.0)
            self.assertEqual((status, body["torque"], body["admission"]), (200, "off", "closed"))
            self.assertEqual(harness._persistent_obstacles, {})
            self.assertIsNone(harness.latest)
        finally:
            release.set()
            t.join(2.0)

        status, body = await self.control("torque_on", **lease)
        self.assertEqual((status, body["torque"]), (200, "on"))
        self.assertEqual(self.observations, 1)  # torque_on passes refresh_observation=False
        status, body = await self.control("resume", **lease)
        self.assertEqual((status, body["status"], body["observation_id"]), (200, "resumed", "obs-fresh-2"))
        ext_events = [e["payload"] for e in self.workbench.store.events(0) if e["type"] == "harness.external_motion"]
        self.assertEqual(
            [(p["phase"], p["source"]) for p in ext_events],
            [
                ("begin", "operator_go_home"),
                ("end", "operator_go_home"),
                ("begin", "torque_off_manual_motion"),
                ("end", "torque_off_manual_motion"),
            ],
        )

    async def test_program_reconcile_on_resume_and_transport_recovery_predicate_unification(self):
        from ni_rua.runtime.programs.runner import ProgramRunner

        lease = await self.claim()
        harness = self.workbench.harness
        runner = harness.programs = ProgramRunner(harness.program_services())
        harness.accepting = False

        # Job 1: exited process & broker -> reconcilable on resume
        harness.store.set_state("program", "job-rec-1", {
            "job_id": "job-rec-1",
            "identity": dict(self.workbench.identity),
            "status": "unknown",
            "stop_confirmed": False,
            "reason": "program_broker_operation_stop_unconfirmed",
        })
        runner._unresolved_handles["job-rec-1"] = {
            "pgid": None,
            "process": SimpleNamespace(poll=lambda: 0),
            "broker": SimpleNamespace(is_alive=lambda: False),
            "no_process_spawned": False,
        }
        # Job 2: post-restart unknown without handle -> requires operator acknowledgement
        harness.store.set_state("program", "job-manual-2", {
            "job_id": "job-manual-2",
            "identity": dict(self.workbench.identity),
            "status": "unknown",
            "stop_confirmed": False,
            "reason": "gateway_restarted_without_program_acknowledgement",
        })

        state = await (await self.client.get("/v1/console/state")).json()
        self.assertEqual(
            sorted(state["resume"]["requires_acknowledgement"]),
            ["program:job-manual-2", "program:job-rec-1"],
        )
        # Resume reconciles job-rec-1 automatically; missing acknowledgement for job-manual-2 still refuses
        status, body = await self.control("resume", **lease)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "resume_requires_unknown_acknowledgement:program:job-manual-2")
        self.assertTrue(harness.store.read_state("program", "job-rec-1")["late_stop_confirmed"])
        self.assertEqual(harness.store.read_state("program", "job-rec-1")["status"], "unknown")

        status, body = await self.control("resume", acknowledge_unknown=["program:job-manual-2"], **lease)
        self.assertEqual((status, body["status"]), (200, "resumed"))
        self.assertTrue(harness.accepting)
        # Archival/observer predicate stays True while transport recovery gate honors operator acknowledgement!
        self.assertTrue(self.workbench.current_unknown_actions())
        self.assertIsNone(self.workbench.transport_recovery_block())

    async def test_relax_dispatches_hardware_torque_off_even_when_external_motion_begin_raises(self):
        lease = await self.claim()
        harness = self.workbench.harness
        with mock.patch.object(harness, "external_motion", side_effect=ValueError("scene_invalidation_conflict:simulated")):
            status, body = await self.control("relax", confirm="relax", **lease)
        self.assertEqual((status, body["torque"], body["admission"]), (200, "off", "closed"))
        self.assertFalse(harness.accepting)
        self.assertIsNone(harness.latest)

class ModelFreeTextTests(unittest.IsolatedAsyncioTestCase):
    @property
    def _base_config(self):
        return _cached_base_config()

    asyncSetUp = ConsoleCheck.asyncSetUp
    asyncTearDown = ConsoleCheck.asyncTearDown
    control = ConsoleCheck.control
    claim = ConsoleCheck.claim

    async def model_setting(self):
        return (await (await self.client.get("/v1/console/state")).json())["settings"]["runtime"]["model"]

    async def test_any_well_formed_id_is_applied_and_ill_formed_refused(self):
        session = self.workbench.session
        session.model, session.price_card, session.model_proxy = "kimi-k2.6", None, None
        lease = await self.claim()
        status, body = await self.control("set_model", model="deepseek-flash", **lease)
        self.assertEqual((status, body["status"], body["model"], body["applies"], session.model),
                         (200, "applied", "deepseek-flash", "next_agent_turn", "deepseek-flash"))
        event = self.workbench.store.events_by_type(["operator.model_changed"])[-1]["payload"]
        self.assertEqual((event["previous"], event["model"], event["applies"]),
                         ("kimi-k2.6", "deepseek-flash", "next_agent_turn"))
        for bad in ("deepseek flash", "", 7, None):
            status, body = await self.control("set_model", model=bad, **lease)
            self.assertEqual((status, body.get("error")), (400, "invalid_model"), bad)
        self.assertEqual(session.model, "deepseek-flash")
        session.price_card = {"input_per_million": 1.0}
        status, body = await self.control("set_model", model="glm-5.3", **lease)
        self.assertEqual((status, body.get("error")), (409, "set_model_conflicts_with_price_card"))

    async def test_backend_refusal_reaches_the_console_unchanged(self):
        session = self.workbench.session
        session.model, session.price_card, session.model_proxy = "kimi-k2.6", None, None

        def refuse(model):
            raise AgentCapabilityError(f"opencode_model_not_declared:{model}: start a new run with --model {model}")
        session.set_model = refuse
        lease = await self.claim()
        status, body = await self.control("set_model", model="glm-5.3", **lease)
        self.assertEqual(status, 400)
        self.assertTrue(body["error"].startswith("opencode_model_not_declared:glm-5.3"), body)
        self.assertEqual(session.model, "kimi-k2.6")

    async def test_a_blocking_set_model_runs_off_the_event_loop(self):
        """OpenCode's set_model declares a new model over HTTP (config rewrite, GET /api/model polls, maybe a
        reload): the console must call it on a worker thread, so the gateway keeps answering meanwhile."""
        session = self.workbench.session
        session.model, session.price_card, session.model_proxy = "kimi-k2.6", None, None
        entered, release, threads = threading.Event(), threading.Event(), []

        def blocking(model):
            threads.append(threading.current_thread())
            entered.set()
            release.wait(10)
            session.model = model
        session.set_model = blocking
        lease = await self.claim()
        pending = asyncio.ensure_future(self.control("set_model", model="glm-5.3", **lease))
        self.assertTrue(await asyncio.to_thread(entered.wait, 5))
        state = await asyncio.wait_for(self.client.get("/v1/console/state"), 5)   # served while set_model blocks
        self.assertEqual(state.status, 200)
        self.assertFalse(pending.done())
        release.set()
        status, body = await asyncio.wait_for(pending, 10)
        self.assertEqual((status, body["status"], session.model), (200, "applied", "glm-5.3"))
        self.assertIsNot(threads[0], threading.main_thread())

    async def test_session_suggestions_with_labels_and_no_image_marker(self):
        session = self.workbench.session
        session.model = "my-free-text-model"
        session.model_options = lambda: [dict(value="kimi-k2.6", name="Kimi K2.6", image_input=True),
                                         dict(value="glm-5.3", name="GLM-5.3", image_input=False),
                                         dict(value="bare-id", name=None, image_input=None),
                                         dict(value="kimi-k2.6", name="duplicate"), dict(value="bad id"), "junk"]
        setting = await self.model_setting()
        self.assertEqual(setting["options"], [
            dict(value="kimi-k2.6", label="Kimi K2.6 (kimi-k2.6)", image_input=True),
            dict(value="glm-5.3", label="GLM-5.3 (glm-5.3)" + NO_IMAGE_INPUT_LABEL, image_input=False),
            dict(value="bare-id", label="bare-id", image_input=None),
            dict(value="my-free-text-model", label="my-free-text-model")])
        self.assertEqual(setting["choices"], ["kimi-k2.6", "glm-5.3", "bare-id", "my-free-text-model"])
        self.assertEqual((setting["value"], setting["free_text"], setting["applies"]),
                         ("my-free-text-model", True, "next agent turn"))

    async def test_without_session_suggestions_no_hardcoded_list(self):
        # Design 16 section 6: no hardcoded model list. A Codex-home run lists the pinned native Codex catalogue
        # (codex_models.json, visibility "list"); another backend without a profile lists the current model only.
        session = self.workbench.session
        session.model = "my-codex-model"
        catalogue = REPO_ROOT / "ni_rua/cortex/session/codex_models.json"
        native = [entry["slug"] for entry in sorted((entry for entry in json.loads(catalogue.read_text())["models"]
                                                     if entry.get("visibility") == "list"),
                                                    key=lambda entry: (entry.get("priority", 1 << 30), entry["slug"]))]
        self.assertIn("gpt-6.1-sol", native)

        def broken():
            raise RuntimeError("catalogue reader crashed")
        for hook in (None, lambda: None, lambda: [], lambda: [dict(value="bad id")], broken):
            session.model_options = hook
            setting = await self.model_setting()
            self.assertEqual(setting["choices"], native + ["my-codex-model"])
        sol = next(item for item in setting["options"] if item["value"] == "gpt-6.1-sol")
        self.assertEqual((sol["label"], sol["image_input"]), ("GPT-6.1-Sol (gpt-6.1-sol)", True))
        self.assertTrue(sol["reasoning_efforts"])          # native capability fields fill the suggestion
        session.backend_kind = "claude_code"
        self.assertEqual((await self.model_setting())["choices"], ["my-codex-model"])

    async def test_current_model_image_flag_is_reported(self):
        session = self.workbench.session
        session.model = "deepseek-v4.1-flash-1017"
        session.model_options = lambda: [dict(value="deepseek-v4.1-flash-1017", name="DeepSeek V4.1", image_input=False),
                                         dict(value="kimi-k2.6", name="Kimi K2.6", image_input=True)]
        self.assertIs((await self.model_setting())["image_input"], False)
        session.model = "kimi-k2.6"
        self.assertIs((await self.model_setting())["image_input"], True)
        session.model = "free-text-model"
        self.assertIsNone((await self.model_setting())["image_input"])

    async def test_blocked_transport_view_and_operator_retry(self):
        class Proxy:
            service_tier = None

            def __init__(self, error):
                self.blocked_error, self.retries = error, 0

            def manual_retry(self):
                if not self.blocked_error["manual_retry_allowed"]:
                    raise ValueError("manual_retry_requires_explicit_provider_error")
                self.retries += 1
                self.blocked_error = None

        session = self.workbench.session
        session.model, session.price_card = "deepseek-v4.1-flash-1017", None
        error = dict(logical_call_id="call-1", error_type="InfrastructureFailure", manual_retry_allowed=True,
                     evidence=dict(http_status=400, classification="configuration_error", code=None,
                                   type="invalid_request_error", message="IMAGE_INPUT_UNSUPPORTED"))
        session.model_proxy = Proxy(error)
        state = await (await self.client.get("/v1/console/state")).json()
        self.assertEqual(state["model_transport"], dict(
            blocked=True, error_type="InfrastructureFailure", manual_retry_allowed=True, http_status=400,
            classification="configuration_error", code=None, type="invalid_request_error",
            message="IMAGE_INPUT_UNSUPPORTED"))
        status, body = await self.control("retry_model_requests")
        self.assertEqual((status, body.get("error")), (400, "input_owner_claim_required"))   # lease-gated
        lease = await self.claim()
        status, body = await self.control("set_model", model="kimi-k2.6", **lease)
        self.assertEqual(status, 200)
        self.assertEqual(session.model_proxy.retries, 0)                                     # never automatic
        status, body = await self.control("retry_model_requests", **lease)
        self.assertEqual((status, body["status"], body["previous"]["message"], session.model_proxy.retries),
                         (200, "retry_authorized", "IMAGE_INPUT_UNSUPPORTED", 1))
        event = self.workbench.store.events_by_type(["operator.model_retry_authorized"])[-1]["payload"]
        self.assertEqual(event["previous"]["http_status"], 400)
        self.assertEqual((await (await self.client.get("/v1/console/state")).json())["model_transport"],
                         dict(blocked=False))
        status, body = await self.control("retry_model_requests", **lease)
        self.assertEqual((status, body.get("error")), (400, "model_transport_not_blocked"))
        session.model_proxy = Proxy(dict(error, manual_retry_allowed=False))
        status, body = await self.control("retry_model_requests", **lease)
        self.assertEqual((status, body.get("error")), (400, "manual_retry_requires_explicit_provider_error"))
        session.model_proxy = None
        status, body = await self.control("retry_model_requests", **lease)
        self.assertEqual((status, body.get("error")), (400, "model_proxy_unavailable"))

    async def test_save_video_ffmpeg_compose_failure_does_not_poison_workbench_status(self):
        from unittest.mock import patch

        self.workbench.status = "running"
        self.workbench.config = self.workbench.config.with_overrides({"features": {"record_video": True}})
        self.workbench.harness.capabilities = dict(self.workbench.harness.capabilities or {}, observer_video=True)
        self.workbench.robot.save_video = lambda: {
            "segments": [{"segment_id": "seg-1"}],
            "cameras": [{"camera_id": "fixed", "status": "saved", "frames": 10}],
        }
        self.workbench.robot.record = lambda enabled=True: {"enabled": enabled}
        lease = await self.claim()
        with patch(
            "ni_rua.apps.gateway.extensions.operator_console._compose_motion_video",
            side_effect=RuntimeError("ffmpeg_failed: codec missing"),
        ):
            status, body = await self.control("save_video", **lease)
        self.assertEqual(status, 200, body)
        self.assertEqual(body["status"], "failed")
        self.assertIn("compose_failed: RuntimeError", body["error"])
        self.assertEqual(self.workbench.status, "running")
        self.assertEqual(self.workbench.video_export_state["status"], "failed")


class ConsoleUnitChecks(unittest.TestCase):
    def test_model_id_pattern(self):
        for good in ("kimi-k2.6", "deepseek-v4-flash-0731", "qwen3.8-max", "vendor/model-1", "claude-x[1m]",
                     "gpt-6.1-sol", "a" * 200):
            self.assertTrue(MODEL_ID.match(good), good)
        for bad in ("", " kimi", "kimi k2", "-x", "a\"b", "x\n", "a" * 201, "model;rm"):
            self.assertFalse(MODEL_ID.match(bad), bad)

    def test_base_session_offers_no_suggestions(self):
        self.assertIsNone(AgentSession.__new__(AgentSession).model_options())

    def test_so101_speed_probe_holds_on_keyboard_interrupt_and_always_closes_adapter(self):
        import io
        import runpy
        import sys
        from contextlib import redirect_stdout
        from unittest.mock import patch

        script = Path(__file__).resolve().parents[3] / "scripts" / "so101_speed_probe.py"
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        fake_dep = {
            "home": {"q_rad": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},
            "motion": {"max_command_delta_rad": 0.05, "command_rate_hz": 50.0, "tracking_error_limit_rad": 0.2},
        }

        # 1. KeyboardInterrupt MUST call _stop(), MUST NOT call probe_park, MUST write output & close()
        calls_kb = []

        class KbAdapter:
            def __init__(self, dep, *a, **kw):
                self.deployment = dep

            def move_joints(self, q, jaw, tol, action_id=None, **kw):
                calls_kb.append(("move_joints", action_id))
                raise KeyboardInterrupt("operator abort")

            def _stop(self):
                calls_kb.append(("_stop", None))
                return {"confirmed": True}

            def close(self):
                calls_kb.append(("close", None))

        out_kb = Path(tmp) / "kb_out.json"
        with patch("ni_rua.robot.backends.so101.deployment.load_deployment", return_value=fake_dep), \
             patch("ni_rua.robot.backends.so101.worker.SO101Adapter", KbAdapter), \
             patch.object(sys, "argv", [str(script), "--deployment", "fake.json", "--out", str(out_kb)]), \
             redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                runpy.run_path(str(script), run_name="__main__")
        self.assertEqual(calls_kb, [("move_joints", "probe_ready"), ("_stop", None), ("close", None)])
        self.assertTrue(out_kb.exists())

        # 2. Runtime fault where probe_park also raises MUST STILL write output & close()
        calls_err = []

        class FaultAdapter:
            def __init__(self, dep, *a, **kw):
                self.deployment = dep

            def move_joints(self, q, jaw, tol, action_id=None, **kw):
                calls_err.append(("move_joints", action_id))
                raise RuntimeError(f"serial_lost:{action_id}")

            def _stop(self):
                calls_err.append(("_stop", None))
                return {"confirmed": False}

            def close(self):
                calls_err.append(("close", None))

        out_err = Path(tmp) / "err_out.json"
        with patch("ni_rua.robot.backends.so101.deployment.load_deployment", return_value=fake_dep), \
             patch("ni_rua.robot.backends.so101.worker.SO101Adapter", FaultAdapter), \
             patch.object(sys, "argv", [str(script), "--deployment", "fake.json", "--out", str(out_err)]), \
             redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "serial_lost:probe_ready"):
                runpy.run_path(str(script), run_name="__main__")
        self.assertEqual(
            calls_err,
            [("move_joints", "probe_ready"), ("_stop", None), ("move_joints", "probe_park"), ("close", None)],
        )
        saved_err = json.loads(out_err.read_text())
        self.assertTrue(any("park_error" in item for item in saved_err))


if __name__ == "__main__":
    unittest.main()
