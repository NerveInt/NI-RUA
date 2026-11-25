"""Consolidated tests:

- test_gateway_remote: Gateway wiring of the remote services: startup/shutdown hook and GET /v1/remote/status (no ssh, robot, Codex or model).
- test_remote_manager: ni_rua.apps.remote.manager: start-or-reuse, GPU selection, timeout, shutdown ownership, status shape (no real ssh or network).
- test_remote_tunnel: ni_rua.apps.remote: ssh forward command, health record, busy-port refusal (no real ssh or network).
"""

import asyncio
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from ni_rua.apps.gateway.server import serve
from ni_rua.apps.gateway.workbench import Workbench
from ni_rua.apps.remote.manager import RemoteError
from ni_rua.core.config import Config
from ni_rua.core.storage.store import Store
import subprocess
from ni_rua.apps.remote import manager as remote
from ni_rua.apps.remote.tunnel import Supervisor
from ni_rua.core.config import RemoteConfig, RemoteServiceConfig
import contextlib
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from ni_rua.apps.remote import manager as remote_manager
from ni_rua.apps.remote import tunnel
from support.paths import isolate_home


def setUpModule():
    isolate_home()   # a fresh Workbench opens the grasp memory store at its '~' default


NI_RUA_TEST_TIER = "component"

# --- test_gateway_remote.py ---
REMOTE = {"remote": {"enabled": True, "host": "gpuhost", "services": [{"name": "vggt", "remote_port": 8331}]}}


class FakeManager:
    """Stands in for RemoteManager: records the lifecycle the gateway drives."""
    instances = []

    def __init__(self, config, **kwargs):
        self.config, self.record_path, self.calls = config, None, []
        self.fail = None
        FakeManager.instances.append(self)

    def start(self):
        self.calls.append("start")
        if self.fail:
            raise self.fail

    def abort(self):
        self.calls.append("abort")

    def stop(self):
        self.calls.append("stop")

    def status(self):
        return {"enabled": True, "state": "unavailable"}


def serve_args(root, config):
    return SimpleNamespace(
        run_dir=str(root), resume=False, replay=False, price_card=None, category="diagnostic", task="fixture", title="fixture",
        tag=[], gpu=0, max_actions=1, observation_profile="rgb_proprio_v1", seed=0, layout_id=0, port=0, prompt="fixture",
        task_notes=None, task_notes_source=None, wall_budget_s=1, exit_after_turn=True, benchmark=False, resolved_config=config)


def fake_workbench(root):
    identity = dict(run_id="fixture", episode_id="episode", env_instance_id="env", env_generation=1)
    store = Store(root, identity)
    terminal = threading.Event()
    terminal.set()
    session = SimpleNamespace(thread_id="t", state="completed", turn_done=terminal, turn_result={"status": "completed", "error": None},
                              model="fixture", provider="fixture", price_card=None, model_proxy=SimpleNamespace(limit_notified=False))
    return SimpleNamespace(
        store=store, identity=identity, session=session, status="running",
        harness=SimpleNamespace(action_count=0, active_action=None, episode_done=threading.Event(), episode_terminal=None,
                                native_action_count=0, started_at=time.monotonic()),
        app=web.Application, snapshot=lambda: {"status": "running"}, start=AsyncMock(), stop=AsyncMock(),
        close=AsyncMock(side_effect=lambda: store.close() or {"steps": {}, "errors": []}),
        current_unknown_actions=lambda: False, control_lock=asyncio.Lock(), has_active_programs=lambda: False)


class GatewayRemoteHook(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        FakeManager.instances = []

    async def run_serve(self, config, fail=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workbench = fake_workbench(root)
            self.calls_at_workbench_start = []
            workbench.start = AsyncMock(side_effect=lambda *a, **k: self.calls_at_workbench_start.append(list(FakeManager.instances[0].calls) if FakeManager.instances else None))

            def make(remote_config, **kwargs):
                manager = FakeManager(remote_config, **kwargs)
                manager.fail = fail
                return manager
            error = None
            with patch("ni_rua.apps.gateway.server.Workbench", return_value=workbench), \
                    patch("ni_rua.apps.gateway.server.RemoteManager", side_effect=make), redirect_stdout(io.StringIO()):
                try:
                    await serve(serve_args(root, config))
                except RemoteError as caught:
                    error = caught
            return workbench, error, root

    async def test_remote_disabled_is_unchanged(self):
        workbench, error, _ = await self.run_serve(Config())
        self.assertIsNone(error)
        self.assertEqual(FakeManager.instances, [])
        self.assertIsNone(workbench.remote)
        workbench.start.assert_awaited_once()

    async def test_enabled_connects_before_start_and_releases_after_close(self):
        workbench, error, root = await self.run_serve(Config.from_dict(REMOTE))
        self.assertIsNone(error)
        (manager,) = FakeManager.instances
        self.assertEqual(manager.calls, ["start", "stop"])
        self.assertIs(workbench.remote, manager)
        self.assertEqual(Path(manager.record_path).resolve(), (root / "private" / "remote_services.json").resolve())
        self.assertEqual(self.calls_at_workbench_start, [["start"]])   # connected before the first actor turn, released after
        workbench.start.assert_awaited_once()
        workbench.close.assert_awaited_once()

    async def test_unavailable_service_does_not_block_startup_unless_required(self):
        # required=false: manager.start() returns normally (services merely unavailable) and the run proceeds.
        workbench, error, _ = await self.run_serve(Config.from_dict(REMOTE))
        self.assertIsNone(error)
        workbench.start.assert_awaited_once()
        # required=true: the manager raises, the robot is never started, and the tunnel is still released.
        workbench, error, _ = await self.run_serve(
            Config.from_dict({"remote": {**REMOTE["remote"], "required": True}}), fail=RemoteError("required remote services unavailable"))
        self.assertIsInstance(error, RemoteError)
        workbench.start.assert_not_awaited()
        self.assertEqual(FakeManager.instances[-1].calls, ["start", "stop"])

    async def test_replay_never_starts_remote_services(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = serve_args(root, Config.from_dict(REMOTE))
            args.replay, args.resume = True, False
            with patch("ni_rua.apps.gateway.server._serve", new=AsyncMock()) as inner, \
                    patch("ni_rua.apps.gateway.server.RemoteManager", side_effect=FakeManager):
                await serve(args)
            inner.assert_awaited_once_with(args, None)
        self.assertEqual(FakeManager.instances, [])

    async def test_shutdown_during_remote_start_never_starts_workbench(self):
        async def shutdown(manager, done):
            done.set()
        with patch("ni_rua.apps.gateway.server.start_services", side_effect=shutdown):
            workbench, error, _ = await self.run_serve(Config.from_dict(REMOTE))
        self.assertIsNone(error)
        workbench.start.assert_not_awaited()
        workbench.close.assert_awaited_once()
        self.assertEqual(FakeManager.instances[-1].calls, ["stop"])


class GatewaySummarySettle(unittest.IsolatedAsyncioTestCase):
    async def test_the_relay_is_settled_once_off_the_event_loop_before_the_summary(self):
        # ModelRequestProxy.settle blocks up to its bound; robot release and a handoff must not wait on a blocked loop.
        # It comes before the unknown-action check, so no await separates that check from the summary's reads.
        calls, summaries = [], []

        def settle():
            try:
                asyncio.get_running_loop()
                calls.append("event loop")
            except RuntimeError:
                calls.append("worker thread")
            return False                     # e.g. the bound passed: the summary says so

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workbench = fake_workbench(root)
            workbench.session.model_proxy = SimpleNamespace(limit_notified=False, settle=settle)
            workbench.current_unknown_actions = lambda: calls.append("unknown-action check") and False
            with patch("ni_rua.apps.gateway.server.Workbench", return_value=workbench), \
                    patch("ni_rua.apps.gateway.server.finish_run",
                          side_effect=lambda root, **kwargs: summaries.append(kwargs["summary"])), \
                    redirect_stdout(io.StringIO()):
                await serve(serve_args(root, Config()))
        self.assertEqual(calls.count("worker thread"), 1)
        self.assertEqual(calls[-2:], ["worker thread", "unknown-action check"])   # the finalisation's own check
        self.assertIs(summaries[-1]["model_records_settled"], False)


class FakeRobot:
    def close(self):
        return None


class RemoteStatusRoute(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        temporary_dir = self.enterContext(tempfile.TemporaryDirectory())
        self.workbench = Workbench(Path(temporary_dir) / "synthetic-remote", robot=FakeRobot())
        self.client = None

    async def asyncTearDown(self):
        if self.client:
            await self.client.close()
        await self.workbench.close()

    async def get(self):
        self.client = TestClient(TestServer(self.workbench.app()))
        await self.client.start_server()
        response = await self.client.get("/v1/remote/status")
        return response.status, response.headers.get("Cache-Control"), await response.json()

    async def test_disabled_reports_enabled_false(self):
        status, cache, body = await self.get()
        self.assertEqual((status, cache, body), (200, "no-store", {"enabled": False}))

    async def test_enabled_returns_the_manager_status(self):
        manager = Mock()
        manager.status.return_value = {"enabled": True, "host": "gpuhost", "state": "ready", "tunnel": {"state": "up"},
                                       "services": {"vggt": {"available": True, "origin": "reused"}}}
        self.workbench.remote = manager
        status, _, body = await self.get()
        self.assertEqual(status, 200)
        self.assertEqual(body["services"]["vggt"], {"available": True, "origin": "reused"})
        self.assertEqual(json.dumps(body, sort_keys=True), json.dumps(manager.status.return_value, sort_keys=True))

    async def test_archived_returns_saved_remote_services(self):
        self.workbench.remote = None
        private = Path(self.workbench.store.root) / "private"
        private.mkdir(parents=True, exist_ok=True)
        (private / "remote_services.json").write_text(json.dumps({"host": "gpu-archive", "services": {"vggt": {"available": True}}}))
        status, _, body = await self.get()
        self.assertEqual(status, 200)
        self.assertEqual(body["host"], "gpu-archive")
        self.assertTrue(body["archive"])
        self.assertTrue(body["enabled"])

    async def test_legacy_perception_after_first_event_page(self):
        store = self.workbench.store
        for _ in range(510): store.event("fixture.noop", {}, {})
        store.event("geometry.obs3d_attached", dict(observation_id="late", camera_id="fixed",
            object_table=dict(status="unavailable", reason="timeout")), {})
        self.client = TestClient(TestServer(self.workbench.app()))
        await self.client.start_server()
        response = await self.client.get("/v1/observations/late/perception")
        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        value = await response.json()
        self.assertEqual(value["diagnostics"]["object_table"]["status"], "unavailable")
        self.assertNotIn("available", value)

    async def test_observation_perception_endpoint(self):
        self.client = TestClient(TestServer(self.workbench.app()))
        await self.client.start_server()
        # Not found initially
        res404 = await self.client.get("/v1/observations/obs-test-1/perception")
        self.assertEqual(res404.status, 404)
        # Store perception state
        perception_data = {
            "observation_id": "obs-test-1",
            "camera_id": "cam_head",
            "objects": [{"id": "o1", "label": "cup", "centre_m": [0.1, 0.2, 0.78]}],
            "table": {"rows": [["o1", [0.1, 0.2, 0.78], 0.82, 0.765, 0.08, 0.08, 0, None, "table", "ok"]]},
        }
        self.workbench.store.set_state("perception", "obs-test-1", perception_data)
        res200 = await self.client.get("/v1/observations/obs-test-1/perception")
        self.assertEqual(res200.status, 200)
        payload = await res200.json()
        self.assertEqual(payload["observation_id"], "obs-test-1")
        self.assertEqual(payload["objects"][0]["label"], "cup")


# --- test_remote_manager.py ---
NVIDIA_SMI = "0, 20496\n1, 20514\n4, 81352\n5, 45867\n6, 8718\n7, 51562\n"
TOO_FULL = "0, 9000\n1, 9500\n"


class FakeProcess:
    pid = 4242

    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


class Fixture:
    """Scripted ssh runner, health answers and a fake ssh tunnel process."""

    def __init__(self, health=(), gpus=NVIDIA_SMI, start_reply="NI_RUA_PID 777\nNI_RUA_LOG /home/u/.cache/ni-rua/remote/vggt.8331.start.log\n",
                 start_code=0, launch_state="NI_RUA_ALIVE 1\n"):
        self.health_answers, self.gpus, self.start_reply, self.start_code = list(health), gpus, start_reply, start_code
        self.launch_state, self.scripts, self.processes, self.opened = launch_state, [], [], False
        self.last_health = {"available": False, "reason": "URLError:Connection refused"}

    # runner
    def run(self, script, timeout_s):
        self.scripts.append(script)
        if script.startswith("nvidia-smi"):
            return remote.Result(0, self.gpus, "")
        if "setsid nohup" in script:
            return remote.Result(self.start_code, self.start_reply, "")
        if "NI_RUA_ALIVE" in script:
            return remote.Result(0, self.launch_state, "")
        if script.startswith("tail "):
            return remote.Result(0, "Traceback\nRuntimeError: CUDA out of memory\n", "")
        if script.startswith("git "):
            return remote.Result(0, "abc123\n", "")
        if "kill -TERM" in script or "NI_RUA_RC" in script:
            return remote.Result(0, "NI_RUA_STOPPED 1\n" if "kill -TERM" in script else "NI_RUA_RC 0\n", "")
        return remote.Result(1, "", "unexpected script")

    def started(self):
        return [item for item in self.scripts if "setsid nohup" in item]

    def stopped(self):
        return [item for item in self.scripts if "kill -TERM" in item or "NI_RUA_RC" in item]

    # tunnel
    def popen(self, argv, **kwargs):
        self.opened = True
        process = FakeProcess()
        self.processes.append((argv, process))
        return process

    def health(self, url):
        if self.health_answers:
            answer = self.health_answers.pop(0)
            if answer is not None:
                self.last_health = answer
        return dict(self.last_health)

    def supervisor(self, host, forwards, urls, **kwargs):
        return Supervisor(host, forwards, urls, health_fn=self.health, port_probe=lambda port: self.opened, popen=self.popen,
                          log=io.StringIO(), **kwargs)


OK = {"available": True, "status": "ok", "version": "vggt-height-v2"}
DOWN = {"available": False, "reason": "URLError:Connection refused"}


def service(**fields):
    values = dict(name="vggt", remote_port=8331, command="bash start.sh", workdir="~/code/NI-RUA/services/vggt_height",
                  env=("VGGT_HOME=~/models/vggt",), gpu="auto", startup_timeout_s=5.0, stop="bash stop.sh")
    values.update(fields)
    return RemoteServiceConfig(**values)


def make(fixture, *, services=None, **fields):
    values = dict(enabled=True, host="gpuhost", services=tuple(services or (service(),)), poll_interval_s=0.001,
                  health_interval_s=1.0, remote_repo="~/code/NI-RUA")
    values.update(fields)
    messages = []
    manager = remote.RemoteManager(RemoteConfig(**values), runner=fixture, supervisor_factory=fixture.supervisor, say=messages.append)
    manager.messages = messages
    return manager


class ParsingAndScripts(unittest.TestCase):
    def test_parse_and_pick_gpu(self):
        gpus = remote.parse_gpus(NVIDIA_SMI + "garbage line\n")
        self.assertEqual(gpus[:3], [(0, 20496), (1, 20514), (4, 81352)])
        self.assertEqual(remote.pick_gpu(gpus, "auto", 15360), ("4", None))
        self.assertEqual(remote.pick_gpu(gpus, "0,4", 15360), ("0,4", None))
        index, reason = remote.pick_gpu(remote.parse_gpus(TOO_FULL), "auto", 15360)
        self.assertIsNone(index)
        self.assertIn("no_gpu_with_15360_MiB_free", reason)
        self.assertIn("GPU 1 with 9500", reason)
        self.assertEqual(remote.pick_gpu(gpus, "6", 15360)[0], None)
        self.assertIn("gpu_6_has_8718_MiB_free_below_required_15360", remote.pick_gpu(gpus, "6", 15360)[1])
        self.assertIn("gpu_9_not_present", remote.pick_gpu(gpus, "9", 15360)[1])
        self.assertEqual(remote.pick_gpu([], "auto", 1)[1], "no_gpu_reported_by_nvidia_smi")
        # equal free memory: the lowest index wins, deterministically
        self.assertEqual(remote.pick_gpu([(2, 50000), (3, 50000)], "auto", 1)[0], "2")

    def test_scripts_are_valid_bash_and_carry_the_contract(self):
        item = service()
        start = remote.start_script(item, "~/.cache/ni-rua/remote", "4")
        self.assertIn("export CUDA_VISIBLE_DEVICES=4", start)
        self.assertIn('export VGGT_HOME="$HOME"/models/vggt', start)
        self.assertIn('cd "$HOME"/code/NI-RUA/services/vggt_height', start)
        self.assertIn("setsid nohup", start)
        self.assertIn("remote_port_busy_but_not_healthy", start)
        self.assertIn("vggt.8331.start.log", start)
        stop_by_command = remote.stop_script(item, 777)
        self.assertIn("bash stop.sh", stop_by_command)
        stop_by_group = remote.stop_script(service(stop=None), 777)
        self.assertIn('kill -TERM -- "-$pid"', stop_by_group)
        for script in (start, stop_by_command, stop_by_group, remote.launch_state_script(777, "/tmp/x log"),
                       remote.tail_script("/tmp/x"), remote.revision_script("~/code/NI-RUA")):
            done = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
            self.assertEqual(done.returncode, 0, done.stderr)

    def test_quoting_keeps_hostile_values_inert(self):
        item = service(command="echo 'a b'; true", env=("X=a b; touch /tmp/pwn",), workdir="/srv/my dir")
        script = remote.start_script(item, "/var/log/ni rua", None)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", script)
        done = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("export X='a b; touch /tmp/pwn'", script)


class StartOrReuse(unittest.TestCase):
    def test_healthy_service_is_reused_and_never_started_or_stopped(self):
        fixture = Fixture(health=[OK])
        manager = make(fixture, stop_on_exit=True)
        manager.start()
        self.addCleanup(manager.stop)
        status = manager.status()
        item = status["services"]["vggt"]
        self.assertEqual((item["available"], item["origin"], item["started_by_us"], item["reason"]), (True, "reused", False, None))
        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["tunnel"]["state"], "up")
        self.assertEqual(fixture.started(), [])
        self.assertFalse(any(script.startswith("nvidia-smi") for script in fixture.scripts))
        manager.stop()
        self.assertEqual(fixture.stopped(), [])      # stop_on_exit is true but we did not start it
        self.assertEqual(manager.status()["tunnel"]["state"], "closed")
        self.assertEqual(fixture.processes[0][1].returncode, -15)   # the tunnel we opened is closed

    def test_unhealthy_service_is_started_on_the_freest_gpu_then_polled_until_healthy(self):
        fixture = Fixture(health=[DOWN, DOWN, DOWN, OK])
        manager = make(fixture)
        manager.start()
        self.addCleanup(manager.stop)
        item = manager.status()["services"]["vggt"]
        self.assertEqual((item["available"], item["origin"], item["started_by_us"], item["gpu"], item["remote_pid"]),
                         (True, "started", True, "4", 777))
        (script,) = fixture.started()
        self.assertIn("export CUDA_VISIBLE_DEVICES=4", script)
        self.assertEqual(item["health"], {"status": "ok", "version": "vggt-height-v2"})   # provenance notes stay out of health

    def test_default_leaves_a_started_service_running_and_stop_on_exit_stops_only_it(self):
        for stop_on_exit in (False, True):
            with self.subTest(stop_on_exit=stop_on_exit):
                fixture = Fixture(health=[DOWN, OK])
                manager = make(fixture, stop_on_exit=stop_on_exit)
                manager.start()
                manager.stop()
                manager.stop()   # idempotent
                self.assertEqual(len(fixture.stopped()), 1 if stop_on_exit else 0)
                if stop_on_exit:
                    self.assertIn("bash stop.sh", fixture.stopped()[0])
                self.assertEqual(manager.status()["state"], "stopped")

    def test_startup_timeout_reports_reason_and_tail_and_required_decides_the_launch(self):
        for required in (False, True):
            with self.subTest(required=required):
                fixture = Fixture(health=[DOWN])
                manager = make(fixture, services=[service(startup_timeout_s=0.05)], required=required)
                self.addCleanup(manager.stop)
                if required:
                    with self.assertRaises(remote.RemoteError) as caught:
                        manager.start()
                    self.assertIn("startup_timeout_after_0.05s", str(caught.exception))
                else:
                    manager.start()
                item = manager.status()["services"]["vggt"]
                self.assertFalse(item["available"])
                self.assertIn("startup_timeout_after_0.05s", item["reason"])
                self.assertIn("CUDA out of memory", item["reason"])
                self.assertTrue(item["started_by_us"])   # launched: shutdown may still stop it
                self.assertEqual(manager.status()["state"], "unavailable")

    def test_start_command_failure_is_reported_without_waiting_for_the_timeout(self):
        fixture = Fixture(health=[DOWN], launch_state="NI_RUA_EXIT 3\nNI_RUA_ALIVE 0\n")
        manager = make(fixture, services=[service(startup_timeout_s=60.0)])
        manager.start()
        self.addCleanup(manager.stop)
        self.assertIn("start_command_exit_3", manager.status()["services"]["vggt"]["reason"])
        fixture = Fixture(health=[DOWN], launch_state="NI_RUA_ALIVE 0\n")
        manager = make(fixture, services=[service(startup_timeout_s=60.0)])
        manager.start()
        self.addCleanup(manager.stop)
        self.assertIn("start_process_vanished", manager.status()["services"]["vggt"]["reason"])

    def test_exit_zero_launcher_keeps_waiting_for_health(self):
        # start.sh style: the detached command exits 0 once the processes are up; health decides.
        fixture = Fixture(health=[DOWN, DOWN, DOWN, DOWN, OK], launch_state="NI_RUA_EXIT 0\nNI_RUA_ALIVE 0\n")
        manager = make(fixture)
        manager.start()
        self.addCleanup(manager.stop)
        self.assertTrue(manager.status()["services"]["vggt"]["available"])

    def test_insufficient_gpu_memory_refuses_to_start(self):
        fixture = Fixture(health=[DOWN], gpus=TOO_FULL)
        manager = make(fixture)
        manager.start()
        self.addCleanup(manager.stop)
        item = manager.status()["services"]["vggt"]
        self.assertFalse(item["available"])
        self.assertIn("no_gpu_with_15360_MiB_free", item["reason"])
        self.assertEqual(fixture.started(), [])
        self.assertFalse(item["started_by_us"])
        required = make(Fixture(health=[DOWN], gpus=TOO_FULL), required=True)
        self.addCleanup(required.stop)
        with self.assertRaises(remote.RemoteError):
            required.start()

    def test_configured_minimum_is_respected(self):
        fixture = Fixture(health=[DOWN, OK], gpus="3, 9000\n")
        manager = make(fixture, services=[service(gpu_min_free_mib=8000)])
        manager.start()
        self.addCleanup(manager.stop)
        self.assertIn("export CUDA_VISIBLE_DEVICES=3", fixture.started()[0])

    def test_explicit_gpu_is_checked_too(self):
        fixture = Fixture(health=[DOWN], gpus=NVIDIA_SMI)
        manager = make(fixture, services=[service(gpu="6")])
        manager.start()
        self.addCleanup(manager.stop)
        self.assertIn("gpu_6_has_8718_MiB_free", manager.status()["services"]["vggt"]["reason"])
        self.assertEqual(fixture.started(), [])

    def test_busy_remote_port_marker_and_missing_command(self):
        fixture = Fixture(health=[DOWN], start_reply="NI_RUA_ERR remote_port_busy_but_not_healthy\n", start_code=4)
        manager = make(fixture)
        manager.start()
        self.addCleanup(manager.stop)
        self.assertEqual(manager.status()["services"]["vggt"]["reason"], "remote_port_busy_but_not_healthy")
        self.assertFalse(manager.status()["services"]["vggt"]["started_by_us"])
        attach = make(Fixture(health=[DOWN]), services=[service(command=None, gpu=None, stop=None)])
        attach.start()
        self.addCleanup(attach.stop)
        self.assertEqual(attach.status()["services"]["vggt"]["reason"], "not_running_and_no_start_command_configured")

    def test_reuse_if_healthy_false_refuses_an_existing_instance(self):
        fixture = Fixture(health=[OK])
        manager = make(fixture, services=[service(reuse_if_healthy=False)])
        manager.start()
        self.addCleanup(manager.stop)
        self.assertEqual(fixture.started(), [])
        item = manager.status()["services"]["vggt"]
        self.assertEqual((item["available"], item["started_by_us"], item["reason"]),
                         (False, False, "already_running_and_reuse_if_healthy_is_false"))

    def test_local_port_differs_from_remote_port_in_the_forward(self):
        fixture = Fixture(health=[OK])
        manager = make(fixture, services=[service(local_port=18331)])
        manager.start()
        self.addCleanup(manager.stop)
        argv = fixture.processes[0][0]
        self.assertIn("127.0.0.1:18331:127.0.0.1:8331", argv)
        self.assertEqual(manager.status()["services"]["vggt"]["local_port"], 18331)

    def test_existing_tunnel_with_healthy_service_is_adopted_and_left_open(self):
        fixture = Fixture(health=[OK])
        fixture.opened = True    # something already forwards the local port
        manager = make(fixture)
        manager.start()
        self.addCleanup(manager.stop)
        self.assertEqual(fixture.processes, [])
        self.assertEqual(manager.status()["tunnel"]["state"], "external")
        manager.stop()
        self.assertEqual(manager.status()["tunnel"]["state"], "closed")

    def test_busy_local_port_without_a_healthy_service_is_refused_not_taken(self):
        fixture = Fixture(health=[DOWN])
        fixture.opened = True
        manager = make(fixture)
        manager.start()
        self.addCleanup(manager.stop)
        status = manager.status()
        self.assertIn("already in use", status["tunnel"]["error"])
        self.assertFalse(status["services"]["vggt"]["available"])
        self.assertEqual(fixture.processes, [])
        self.assertEqual(fixture.started(), [])


class MonitorAndRecord(unittest.TestCase):
    def test_external_tunnel_loss_is_detected_and_reopened(self):
        fixture = Fixture(health=[OK])
        fixture.opened = True
        manager = make(fixture)
        self.addCleanup(manager.stop)
        manager.start()
        self.assertEqual(fixture.processes, [])
        fixture.opened = False
        self.assertEqual(manager.status()["tunnel"]["state"], "down")
        manager._tick()
        self.assertEqual(len(fixture.processes), 1)
        self.assertEqual(manager.status()["tunnel"]["state"], "up")
        manager.stop()
        self.assertEqual(fixture.processes[0][1].returncode, -15)

    def test_missing_ssh_reports_unavailable_and_required_failure(self):
        for required in (False, True):
            fixture = Fixture(health=[DOWN])
            fixture.popen = lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError("missing ssh"))
            manager = make(fixture, required=required)
            self.addCleanup(manager.stop)
            if required:
                with self.assertRaises(remote.RemoteError):
                    manager.start()
            else:
                manager.start()
            self.assertIn("ssh launch failed", manager.status()["tunnel"]["error"])
            self.assertFalse(fixture.started())

    def test_abort_during_gpu_probe_does_not_launch_service(self):
        fixture = Fixture(health=[DOWN])
        manager = make(fixture)
        self.addCleanup(manager.stop)
        run = fixture.run
        def abort_probe(script, timeout_s):
            if script.startswith("nvidia-smi"):
                manager.abort()
            return run(script, timeout_s)
        fixture.run = abort_probe
        manager.start()
        self.assertEqual(fixture.started(), [])

    def test_dead_tunnel_is_reopened_and_status_follows_health(self):
        fixture = Fixture(health=[OK])
        manager = make(fixture)
        manager.start()
        self.addCleanup(manager.stop)
        fixture.processes[0][1].returncode = 255            # ssh died
        fixture.opened, fixture.health_answers, fixture.last_health = False, [], DOWN
        self.assertEqual(manager.status()["tunnel"]["state"], "down")
        manager._tick()
        status = manager.status()
        self.assertEqual(len(fixture.processes), 2)          # reopened
        self.assertEqual(status["tunnel"]["restarts"], 1)
        self.assertEqual(status["tunnel"]["state"], "up")
        self.assertFalse(status["services"]["vggt"]["available"])
        fixture.opened, fixture.last_health = True, OK
        manager._tick()
        self.assertTrue(manager.status()["services"]["vggt"]["available"])

    def test_services_pending_because_the_tunnel_failed_are_ensured_when_it_recovers(self):
        fixture = Fixture(health=[DOWN, DOWN])
        fixture.opened = True        # first open attempt: port busy and unhealthy -> refused
        manager = make(fixture)
        manager.start()
        self.addCleanup(manager.stop)
        self.assertFalse(manager.status()["services"]["vggt"]["available"])
        fixture.opened, fixture.last_health = False, OK       # the stale forward is gone
        manager._tick()
        item = manager.status()["services"]["vggt"]
        self.assertEqual((item["available"], item["origin"]), (True, "reused"))

    def test_provenance_record_and_status_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = Path(tmp) / "private" / "remote_services.json"
            fixture = Fixture(health=[DOWN, OK])
            manager = remote.RemoteManager(
                RemoteConfig(enabled=True, host="gpuhost", services=(service(),), poll_interval_s=0.001, health_interval_s=1.0,
                             remote_repo="~/code/NI-RUA"),
                record=str(record), runner=fixture, supervisor_factory=fixture.supervisor, say=lambda text: None)
            manager.start()
            manager.stop()
            data = json.loads(record.read_text())
            self.assertEqual(data["host"], "gpuhost")
            self.assertEqual(data["forwards"], ["127.0.0.1:8331 -> gpuhost:127.0.0.1:8331"])
            self.assertEqual(data["remote_checkout"], {"repo": "~/code/NI-RUA", "revision": "abc123", "services_dirty": False})
            service_record = data["services"]["vggt"]
            self.assertEqual((service_record["origin"], service_record["started_by_us"], service_record["gpu"]), ("started", True, "4"))
            self.assertEqual(service_record["stop"], "left_running")
            self.assertEqual((service_record["available"], service_record["version"]), (True, "vggt-height-v2"))   # last health while serving
            self.assertEqual(data["tunnel"], "closed")
            status = manager.status()
            text = json.dumps(status)
            self.assertNotIn("abc123", text)                 # revision/provenance stay out of the operator status
            self.assertEqual(set(status), {"enabled", "host", "state", "required", "stop_on_exit", "tunnel", "services", "checked_at"})
            self.assertEqual(set(status["tunnel"]), {"state", "restarts", "error", "forwards"})
            self.assertEqual(set(status["services"]["vggt"]), {
                "name", "remote_port", "local_port", "available", "reason", "origin", "started_by_us", "gpu", "remote_pid",
                "remote_log", "checked_at", "health"})


class GatewayStartup(unittest.IsolatedAsyncioTestCase):
    async def test_cancelling_start_services_joins_start_before_cleanup(self):
        entered, halted, finished = threading.Event(), threading.Event(), threading.Event()
        class SlowManager:
            def start(self):
                entered.set()
                halted.wait(5)
                finished.set()
            def abort(self):
                halted.set()
        task = asyncio.create_task(remote.start_services(SlowManager(), asyncio.Event()))
        await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(finished.is_set())

    async def test_start_services_does_not_block_on_unavailable_unless_required(self):
        fixture = Fixture(health=[DOWN], gpus=TOO_FULL)
        manager = make(fixture)
        self.addCleanup(manager.stop)
        await remote.start_services(manager, asyncio.Event())
        self.assertEqual(manager.status()["state"], "unavailable")
        required = make(Fixture(health=[DOWN], gpus=TOO_FULL), required=True)
        self.addCleanup(required.stop)
        with self.assertRaises(remote.RemoteError):
            await remote.start_services(required, asyncio.Event())

    async def test_shutdown_signal_aborts_a_slow_start(self):
        fixture = Fixture(health=[DOWN])
        manager = make(fixture, services=[service(startup_timeout_s=60.0), service(name="second", remote_port=8332)],
                       poll_interval_s=0.01, required=True)
        self.addCleanup(manager.stop)
        done = asyncio.Event()
        asyncio.get_running_loop().call_later(0.1, done.set)
        await asyncio.wait_for(remote.start_services(manager, done), timeout=10)   # no RemoteError: the launch is being cancelled anyway
        self.assertIn("aborted_while_starting", manager.status()["services"]["vggt"]["reason"])
        self.assertEqual(len(fixture.started()), 1)


# --- test_remote_tunnel.py ---
def health_server(answer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            data = json.dumps(answer).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    return server


class RemoteTunnel(unittest.TestCase):
    def test_ssh_command_binds_loopback_and_fails_fast(self):
        command = tunnel.ssh_command("gpuhost", [8331, 8333])
        self.assertEqual(command[:2], ["ssh", "-N"])
        self.assertIn("ExitOnForwardFailure=yes", command)
        self.assertIn("BatchMode=yes", command)
        self.assertIn("127.0.0.1:8331:127.0.0.1:8331", command)
        self.assertIn("127.0.0.1:8333:127.0.0.1:8333", command)
        self.assertEqual(command[-1], "gpuhost")
        self.assertFalse(any("0.0.0.0" in part for part in command))

    def test_health_keeps_versions_only(self):
        server = health_server({"status": "ok", "version": "vggt-height-v2", "object_table": "object-table-v5-hp1",
                                "points": "scene3d-points-v1", "requests": 12, "pid": 99, "allocated_mb": 4364.0})
        self.addCleanup(server.server_close); self.addCleanup(server.shutdown)
        item = tunnel.health("http://127.0.0.1:%d/health" % server.server_address[1])
        self.assertEqual(item, {"available": True, "status": "ok", "version": "vggt-height-v2",
                                "object_table": "object-table-v5-hp1", "points": "scene3d-points-v1"})

    def test_health_down_is_unavailable(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        item = tunnel.health("http://127.0.0.1:%d/health" % port, timeout_s=1.0)
        self.assertFalse(item["available"])
        self.assertIn("reason", item)

    def test_busy_local_port_is_refused_and_recorded(self):
        listener = socket.socket(); listener.bind(("127.0.0.1", 0)); listener.listen(1)
        self.addCleanup(listener.close)
        port = listener.getsockname()[1]
        with tempfile.TemporaryDirectory() as tmp:
            record = Path(tmp) / "remote.json"
            code = tunnel.main(["--host", "gpuhost", "--port", str(port), "--record", str(record), "--once"])
            self.assertEqual(code, 2)
            data = json.loads(record.read_text())
            self.assertEqual(data["host"], "gpuhost")
            self.assertEqual(data["tunnel"], "closed")
            self.assertIn("already in use", data["error"])


class TunnelCliFakeManager:
    """Stands in for RemoteManager in the --config CLI path."""
    available, fail, calls, hosts = True, None, [], []

    def __init__(self, config, **kwargs):
        TunnelCliFakeManager.hosts.append(config.host)

    def start(self):
        TunnelCliFakeManager.calls.append("start")
        if TunnelCliFakeManager.fail:
            raise TunnelCliFakeManager.fail

    def status(self):
        return {"services": {"vggt": {"available": TunnelCliFakeManager.available}}}

    def say(self, text):
        pass

    def abort(self):
        pass

    def wait(self):
        TunnelCliFakeManager.calls.append("wait")

    def stop(self):
        TunnelCliFakeManager.calls.append("stop")


class RemoteConfigCli(unittest.TestCase):
    def config_file(self, tmp, **remote):
        path = Path(tmp) / "ni-rua.json"
        path.write_text(json.dumps({"remote": {"enabled": True, "host": "gpuhost", "services": [{"name": "vggt", "remote_port": 8331}],
                                               **remote}}))
        return str(path)

    def run_cli(self, argv):
        TunnelCliFakeManager.calls, TunnelCliFakeManager.hosts = [], []
        with patch.object(remote_manager, "RemoteManager", TunnelCliFakeManager), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            return tunnel.main(argv)

    def test_once_exit_codes_follow_availability_and_stop_always_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.config_file(tmp)
            TunnelCliFakeManager.available, TunnelCliFakeManager.fail = True, None
            self.assertEqual(self.run_cli(["--config", path, "--once"]), 0)
            self.assertEqual(TunnelCliFakeManager.calls, ["start", "stop"])
            TunnelCliFakeManager.available = False
            self.assertEqual(self.run_cli(["--config", path, "--once"]), 1)
            TunnelCliFakeManager.fail = remote_manager.RemoteError("required remote services unavailable")
            self.assertEqual(self.run_cli(["--config", path, "--once"]), 2)
            self.assertEqual(TunnelCliFakeManager.calls, ["start", "stop"])
            TunnelCliFakeManager.fail = None

    def test_without_once_it_monitors_until_stopped_and_host_flag_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            TunnelCliFakeManager.available = True
            self.assertEqual(self.run_cli(["--config", self.config_file(tmp), "--host", "otherhost"]), 0)
            self.assertEqual(TunnelCliFakeManager.calls, ["start", "wait", "stop"])
            self.assertEqual(TunnelCliFakeManager.hosts, ["otherhost"])

    def test_misuse_is_rejected_before_anything_connects(self):
        with tempfile.TemporaryDirectory() as tmp:
            disabled = Path(tmp) / "off.json"
            disabled.write_text("{}")
            for argv in (["--config", str(disabled), "--once"], ["--config", self.config_file(tmp), "--port", "8331"], ["--once"]):
                with self.subTest(argv=argv[:2]), self.assertRaises(SystemExit) as caught:
                    self.run_cli(argv)
                self.assertEqual(caught.exception.code, 2)
            self.assertEqual(TunnelCliFakeManager.calls, [])



if __name__ == "__main__":
    unittest.main()
