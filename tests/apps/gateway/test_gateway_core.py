"""Consolidated tests:

- test_gateway: Synthetic startup/stop lifecycle fixtures; no RoboDojo, Codex, or GPU.
- test_gateway_expanded: Integrated HTTP ownership, scene outbox, restart journal and stale-action checks.
- test_gateway_hangup: Gateway shutdown on a terminal hangup: SIGHUP finalises the run like SIGINT/SIGTERM, also with a dead stdout pipe.
- test_gateway_modular: Stage 6 modular gateway, capability routers, web manifest build, and runner polish checks.
- test_gateway_native_failure: Exercise the real serve/final-summary path without a model or robot.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from aiohttp import ClientSession, WSServerHandshakeError, web
from aiohttp.test_utils import TestClient, TestServer
from support.tool_fixture import install_registry
from ni_rua.apps.gateway.workbench import Workbench
from ni_rua.core.config import Config
from ni_rua.cortex.session.terminal import Terminal
import base64
import sys
from openai_codex.errors import InvalidParamsError, TransportClosedError
from unittest.mock import Mock
from ni_rua.perception.scene.scene_service import SceneService
from ni_rua.core.contracts import digest
from support.geometry_fixture import IDENTITY, recorded_plane
from support.fixture_robot import FixtureRobot
import os
import signal
import subprocess
from ni_rua.apps.gateway import server
import ast
from dataclasses import replace
import shutil
from ni_rua.apps.evaluation import checkout_layout, provenance
from ni_rua.apps.gateway import cli as gateway_cli
from ni_rua.apps.gateway.extensions import CapabilityRouter
from ni_rua.apps.gateway.read_router import RunReadRouter
from support.paths import REPO_ROOT, isolate_home
from ni_rua.apps.gateway.supervisor import RunSupervisor
from ni_rua.apps.library import service as library_service
from ni_rua.core.config import Config, cli_overrides, namespace_values
from ni_rua.core.embodiment import EmbodimentProfile
from ni_rua.robot.contracts import CalibratableBackend, ClockDomain, EvaluatorBackend, OperatorBackend, VideoBackend
from ni_rua.robot.model.embodiments import ARX_X5_PROFILE
from contextlib import redirect_stdout
import io
import time
from unittest.mock import AsyncMock, Mock, patch
from aiohttp import web
from ni_rua.apps.gateway.server import serve
from ni_rua.core.storage.store import Store


def setUpModule():
    isolate_home()   # a fresh Workbench opens the grasp memory store at its '~' default


NI_RUA_TEST_TIER = "component"

# --- test_gateway.py ---
class GatewayCheck(unittest.IsolatedAsyncioTestCase):
    async def test_so101_shutdown_exports_observer_video_before_worker_close(self):
        for fail in (False, True):
            with self.subTest(encoder_failure=fail), tempfile.TemporaryDirectory() as temporary:
                calls = []
                video = Path(temporary) / "fixed.mp4"
                video.write_bytes(b"synthetic-video")
                def save():
                    calls.append("save")
                    if fail:
                        raise OSError("synthetic disk failure")
                    return {"audience": "observer_only", "videos": [{"camera_id": "fixed", "video_path": str(video)}]}
                robot = SimpleNamespace(capabilities={"observer_video": True}, save_video=save,
                                        close=lambda: calls.append("close"))
                config = Config.from_dict({"backend": {"environment": "so101_real", "robot_model_id": "so101",
                                                       "deployment": "synthetic"}})
                owner = Workbench(Path(temporary) / "run", robot=robot, config=config)
                owner.starting = False
                result = await owner.close()
                self.assertEqual(calls, ["save", "close"])
                self.assertEqual(result["steps"]["motion_video"], "failed" if fail else "completed")
                self.assertEqual(result["steps"]["robot"], "completed")
                if not fail:
                    report = json.loads((owner.root / "video.json").read_text())
                    self.assertIn("video_ref", report["videos"][0])

    async def test_shutdown_keeps_saved_video_json_when_rearmed_recording_is_empty(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls = []
            robot = SimpleNamespace(capabilities={"observer_video": True}, close=lambda: calls.append("close"),
                                    save_video=lambda: calls.append("save") or {"audience": "observer_only", "segments": [],
                                                                                "cameras": [], "videos": []})
            config = Config.from_dict({"backend": {"environment": "so101_real", "robot_model_id": "so101",
                                                   "deployment": "synthetic"}})
            owner = Workbench(Path(temporary) / "run", robot=robot, config=config)
            owner.starting = False
            saved = {"segments": [{"action_id": "act-1"}], "videos": [], "composite": {"status": "complete"}}
            (owner.root / "video.json").write_text(json.dumps(saved))
            result = await owner.close()
            self.assertEqual(calls, ["save", "close"])
            self.assertEqual(result["steps"]["motion_video"], "completed")
            self.assertEqual(json.loads((owner.root / "video.json").read_text()), saved)

    async def test_shutdown_closes_live_browser_streams_before_native_cleanup(self):
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        closed = []
        workbench = Workbench(Path(temporary) / "synthetic-shutdown",
                              robot=SimpleNamespace(close=lambda: closed.append("robot")))
        workbench.terminal = Terminal(workbench.root, workbench.root)
        runner = web.AppRunner(workbench.app())
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        try:
            async with ClientSession() as client:
                stream = await client.get(url + "/v1/events")
                self.addCleanup(stream.close)
                self.assertEqual(stream.status, 200)
                await stream.content.readline()
                socket = await client.ws_connect(url + "/v1/terminal")
                self.addAsyncCleanup(socket.close)
                self.assertEqual((await socket.receive_json(timeout=2))["type"], "ready")
                # Neither browser connection is closed before runner cleanup.
                await asyncio.wait_for(runner.cleanup(), 2)
                self.assertEqual(closed, [])
                self.assertFalse(workbench.terminal._closed)
                await asyncio.wait_for(stream.read(), 2)
                await socket.receive(timeout=2)
                self.assertTrue(socket.closed)
                await socket.close()
        finally:
            await runner.cleanup()
            await workbench.close()
        self.assertEqual(closed, ["robot"])

    async def test_interrupt_ack_waits_for_native_turn_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            workbench = Workbench(Path(temporary) / "synthetic-stop", robot=SimpleNamespace(close=lambda: None))
            workbench.starting = False
            session = SimpleNamespace(state="running", interrupt=lambda: None, close=lambda: None)
            workbench.session = session
            completion = asyncio.get_running_loop().call_later(.1, setattr, session, "state", "interrupted")
            try:
                result = await workbench.stop("budget")
                self.assertEqual(session.state, "interrupted")
                self.assertEqual(result["status"], "stopped")
            finally:
                completion.cancel()
                await workbench.close()

    async def test_http_turn_ownership_and_terminal_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            workbench = Workbench(Path(temporary) / "synthetic-http", robot=SimpleNamespace(close=lambda: None))
            workbench.starting, workbench.status = False, "running"
            turns = []
            session = SimpleNamespace(thread_id="fixture-thread", turn_id=None, state="completed")

            def run_turn(prompt):
                turns.append(prompt)
                session.turn_id, session.state = "fixture-turn", "running"
                return session.turn_id

            session.run_turn = run_turn
            session.interrupt = lambda: setattr(session, "state", "interrupted")
            session.close = lambda: setattr(session, "state", "closed")
            workbench.session = session
            # A real Terminal object with no start() call: no Codex/PTY process.
            workbench.terminal = Terminal(workbench.root, workbench.root)
            client = TestClient(TestServer(workbench.app()))
            connection = None
            try:
                await client.start_server()
                request = dict(command="prompt", request_id="fixture-prompt", prompt="synthetic fixture task")
                mismatch = await client.post("/v1/control", json=request, headers={"X-Ni-Rua-Run-ID": "another-run"})
                self.assertEqual(mismatch.status, 404)
                self.assertEqual(turns, [])
                first = await client.post("/v1/control", json=request)
                self.assertEqual(first.status, 200)
                original = await first.json()
                duplicate = await client.post("/v1/control", json=request)
                self.assertEqual(duplicate.status, 200)
                self.assertEqual(await duplicate.json(), original)
                self.assertEqual(turns, [request["prompt"]])
                changed = await client.post("/v1/control", json=dict(request, prompt="changed task"))
                self.assertEqual(changed.status, 409)
                self.assertEqual((await client.get("/v1/runs/another-run")).status, 404)
                cross_origin = await client.post("/v1/control", json=dict(request, request_id="foreign"),
                                                 headers={"Origin": "http://foreign.invalid"})
                self.assertEqual(cross_origin.status, 403)
                with self.assertRaises(WSServerHandshakeError) as rejected:
                    await client.ws_connect("/v1/terminal", origin="http://foreign.invalid")
                self.assertEqual(rejected.exception.status, 403)
                connection = await client.ws_connect("/v1/terminal")
                self.assertEqual((await connection.receive_json(timeout=2))["type"], "ready")
                for forbidden in (
                    dict(type="input", data="must not reach the native terminal"),
                    dict(type="key", key="Enter"),
                    dict(type="mouse", button=0, col=10, row=10),
                    "non-dict-payload",
                    ["array-payload"],
                ):
                    await connection.send_json(forbidden)
                    reply = await connection.receive_json(timeout=2)
                    self.assertEqual(reply, dict(type="error", message="terminal_input_is_read_only"))
                for bad_scroll in (
                    dict(type="scroll"),
                    dict(type="scroll", lines=0),
                    dict(type="scroll", lines=True),
                    dict(type="scroll", lines=99),
                    dict(type="scroll", action="submit"),
                    dict(type="scroll", lines=3, action="page_up"),
                ):
                    await connection.send_json(bad_scroll)
                    reply = await connection.receive_json(timeout=2)
                    self.assertEqual(reply, dict(type="error", message="invalid_terminal_scroll"))
                # Valid scroll on unstarted PTY succeeds cleanly without error or side-effects;
                # with a live PTY attached, synthesized escape bytes reach the PTY slave.
                await connection.send_json(dict(type="scroll", lines=-2))
                await connection.send_json(dict(type="scroll", action="bottom"))
                await connection.send_json(dict(type="input", data="fence-before-pty"))
                self.assertEqual(
                    await connection.receive_json(timeout=2),
                    dict(type="error", message="terminal_input_is_read_only"),
                )
                import tty
                pty_master, pty_slave = os.openpty()
                tty.setraw(pty_slave)
                try:
                    workbench.terminal.master = pty_master
                    await connection.send_json(dict(type="scroll", lines=-1))
                    await connection.send_json(dict(type="scroll", action="page_up"))
                    await connection.send_json(dict(type="input", data="still-blocked"))
                    self.assertEqual(
                        await connection.receive_json(timeout=2),
                        dict(type="error", message="terminal_input_is_read_only"),
                    )
                    os.set_blocking(pty_slave, False)
                    self.assertEqual(os.read(pty_slave, 1024), b"\x1b[<64;50;10M\x1b[5~")
                finally:
                    workbench.terminal.master = None
                    os.close(pty_master)
                    os.close(pty_slave)
                self.assertEqual(turns, [request["prompt"]])
                self.assertIsNone(workbench.terminal.process)
            finally:
                if connection is not None:
                    await connection.close()
                await client.close()
                await workbench.close()

    async def test_stop_at_startup_boundaries_and_cleanup(self):
        for phase in ("robot", "session", "turn", "terminal", "success", "unsandboxed"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary:
                calls = []
                entered, release = threading.Event(), threading.Event()
                closing, close_release = threading.Event(), threading.Event()

                def gate(stage):
                    calls.append(stage)
                    if stage == phase:
                        entered.set()
                        if not release.wait(5):
                            raise TimeoutError("synthetic startup gate not released")

                def close_robot():
                    calls.append("robot.close")
                    closing.set()
                    if not close_release.wait(5):
                        raise TimeoutError("synthetic cleanup gate not released")

                sandbox_enabled = phase != "unsandboxed"
                workbench = Workbench(Path(temporary) / "synthetic-run", robot=SimpleNamespace(close=close_robot),
                                      config=Config.from_dict({'features': {'sandbox_enabled': sandbox_enabled}}),
                                      visual_review_model=None)
                harness = SimpleNamespace(accepting=True, active_action=None, states={}, action_count=0,
                                          capabilities=dict(data_origin="synthetic_fixture"), thread_id=None,
                                          scene=SimpleNamespace(capabilities=lambda: dict(physics="unsupported")))
                harness.start = lambda: gate("robot")
                harness.request_stop = lambda: setattr(harness, "accepting", False)
                session = SimpleNamespace(thread_id=None, turn_id=None, state="not_started", actor_dir=workbench.root,
                                          model="fixture-model", provider="fixture-provider", effort="high", price_card=None,
                                          cli_executable=Path("/fixture/native/codex"),
                                          command_prefix=["fixture-sandbox"] if sandbox_enabled else [],
                                          socket_path=workbench.root / "fixture.sock", env={}, process=SimpleNamespace(pid=-1))

                def start_session(*arguments):
                    gate("session")
                    session.thread_id, session.state = "fixture-thread", "idle"
                    return session.thread_id

                def run_turn(prompt):
                    session.state = "running"
                    gate("turn")
                    session.turn_id = "fixture-turn"
                    return session.turn_id

                def close_session():
                    calls.append("session.close")
                    session.state = "closed"

                session.start, session.run_turn, session.close = start_session, run_turn, close_session
                session.interrupt = lambda: setattr(session, "state", "interrupted")
                install_registry(harness, workbench.config)
                workbench.harness, workbench.session = harness, session

                class FixtureTerminal:
                    def __init__(self, *arguments, **keywords):
                        self.session_id, self.process = "fixture-pty", SimpleNamespace(pid=-1)
                        self.command_prefix = keywords["command_prefix"]

                    async def start(self, *arguments):
                        await asyncio.to_thread(gate, "terminal")

                    async def close(self):
                        calls.append("terminal.close")

                    def status(self):
                        return dict(mode="read_only", session_id=self.session_id)

                startup = None
                try:
                    with patch("ni_rua.apps.gateway.workbench.Terminal", FixtureTerminal):
                        startup = asyncio.create_task(workbench.start("synthetic fixture prompt", wall_budget_s=1200))
                        if phase in ("success", "unsandboxed"):
                            await asyncio.wait_for(startup, 3)
                            self.assertEqual(workbench.status, "running")
                            self.assertFalse(workbench.starting)
                            self.assertTrue((workbench.root / "manifest.json").exists())
                            manifest = json.loads((workbench.root / "manifest.json").read_text())
                            self.assertEqual(manifest["actor_wait_budget_s"], 1200)
                            self.assertEqual(manifest["codex_cli_executable"], str(session.cli_executable))
                            expected_prefix = session.command_prefix + [str(session.cli_executable)]
                            self.assertEqual(workbench.terminal.command_prefix[:len(expected_prefix)], expected_prefix)
                            self.assertEqual(manifest['effective_config']['features']['sandbox_enabled'], sandbox_enabled)
                            with self.assertRaisesRegex(ValueError, "artifact_access_scope_denied"):
                                workbench.store.resolve_artifact(manifest["source_ref"])
                            _, content = workbench.store.resolve_artifact(manifest["source_ref"], allowed_scopes=("observer",))
                            sources = json.loads(content)
                            self.assertEqual({name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}, manifest["source_sha256"])
                            self.assertEqual(calls, ["robot", "session", "turn", "terminal"])
                            continue
                        self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                        if phase == "robot":
                            client = TestClient(TestServer(workbench.app()))
                            await client.start_server()
                            try:
                                response = await client.post("/v1/control", json=dict(command="prompt", request_id="early", prompt="too early"))
                                self.assertEqual(response.status, 400)
                                self.assertEqual((await response.json())["error"], "run_not_ready_for_new_task")
                            finally:
                                await client.close()
                        first = await workbench.stop("stop-1")
                        self.assertEqual(first["status"], "stop_requested")
                        self.assertEqual(await workbench.stop("stop-1"), first)
                        self.assertEqual((await workbench.stop("stop-2"))["status"], "stop_requested")
                        self.assertTrue(workbench.starting)
                        self.assertFalse(startup.done())
                        release.set()
                        self.assertTrue(await asyncio.to_thread(closing.wait, 3))
                        self.assertTrue(workbench.starting)
                        self.assertEqual((await workbench.stop("during-cleanup"))["status"], "stop_requested")
                        close_release.set()
                        await asyncio.wait_for(startup, 3)
                        self.assertFalse(workbench.starting)
                        self.assertEqual(workbench.status, "stopped")
                        self.assertFalse((workbench.root / "manifest.json").exists())
                        self.assertFalse(any(event["type"] == "run.started" for event in workbench.store.events()))
                        phases = [stage for stage in calls if stage in ("robot", "session", "turn", "terminal")]
                        self.assertEqual(phases, ["robot", "session", "turn", "terminal"][:["robot", "session", "turn", "terminal"].index(phase) + 1])
                        self.assertEqual(calls.count("robot.close"), 1)
                        self.assertEqual(calls.count("session.close"), 1)
                        self.assertEqual(calls.count("terminal.close"), int(phase == "terminal"))
                finally:
                    release.set()
                    close_release.set()
                    if startup is not None and not startup.done():
                        await asyncio.wait_for(startup, 3)
                    await workbench.close()


# --- test_gateway_expanded.py ---
class ExpandedGatewayCheck(unittest.IsolatedAsyncioTestCase):
    async def test_native_prompt_rejection_is_terminal_but_transport_loss_stays_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            robot = FixtureRobot(root); robot.close = lambda: None
            config = Config.from_dict({'features': {'grasp_memory_dir': str(root/'memory'),
                'observation_profile': 'rgb_proprio_v1', 'motion_review': False}})
            workbench = Workbench(root/'run', robot=robot, config=config)
            workbench.harness.start()
            workbench.starting = False
            workbench.status = 'running'
            workbench.session.thread_id = 'fixture-thread'
            workbench.session.state = 'idle'
            calls = Mock(side_effect=[InvalidParamsError(-32602, 'fixture native rejection'),
                SimpleNamespace(turn=SimpleNamespace(id='fixture-turn')),
                TransportClosedError('fixture lost acknowledgement')])
            workbench.session.client = SimpleNamespace(turn_start=calls)
            client = TestClient(TestServer(workbench.app()))
            try:
                await client.start_server()
                origin = str(client.make_url('')).rstrip('/')
                owner = await (await client.post('/v1/control', headers={'Origin':origin}, json=dict(
                    request_id='claim', command='claim_input', client_id='fixture'))).json()
                async def prompt(identifier):
                    response = await client.post('/v1/control', headers={'Origin':origin}, json=dict(
                        request_id=identifier, command='prompt', prompt='fixture prompt',
                        client_id='fixture', writer_token=owner['writer_token']))
                    return response.status, await response.json()
                status, rejected = await prompt('native-rejection')
                self.assertEqual(status, 400)
                self.assertEqual(rejected['status'], 'rejected')
                self.assertEqual(rejected['native_rpc_code'], -32602)
                self.assertEqual(workbench.session.state, 'idle')
                self.assertTrue(workbench.harness.accepting)
                self.assertEqual(await prompt('native-rejection'), (status, rejected))
                self.assertEqual(calls.call_count, 1, 'same control id must not dispatch again')
                status, accepted = await prompt('next-prompt')
                self.assertEqual(status, 200)
                self.assertEqual(accepted['turn_id'], 'fixture-turn')
                workbench.session.state = 'completed'
                status, uncertain = await prompt('transport-loss')
                self.assertEqual(status, 409)
                self.assertEqual(uncertain['status'], 'unknown')
                self.assertEqual(workbench.session.state, 'disconnected')
                self.assertFalse(workbench.harness.accepting)
                self.assertEqual(await prompt('transport-loss'), (status, uncertain))
                self.assertEqual(calls.call_count, 3)
                self.assertEqual(robot.executions, 0)
            finally:
                workbench.session.client = None
                await client.close()
                await workbench.close()

    async def test_closed_run_replay_preserves_records_and_rejects_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'run'
            robot = FixtureRobot(temp); robot.close = lambda: None
            recorded = Workbench(root, robot=robot, observation_profile="rgb_proprio_v1", motion_review=False)
            thread_id = 'thread-synthetic-replay'
            program_id = 'program-synthetic-replay'
            try:
                observation = recorded.harness.start()
                recorded.starting = False; recorded.session.state = 'completed'
                recorded.session.thread_id = thread_id
                recorded.store.set_state('runtime', 'codex', {'thread_id': thread_id})
                item = dict(id='item-synthetic-replay', type='agentMessage', status='completed',
                            thread_id=thread_id, text='synthetic recorded reply')
                recorded.store.set_state('codex_item', item['id'], item)
                stdout_ref = recorded.store.put_artifact(b'synthetic program output', 'program_stdout', 'text/plain')
                program = dict(job_id=program_id, status='succeeded', returncode=0, stdout_ref=stdout_ref)
                recorded.store.set_state('program', program_id, program)
                (root/'manifest.json').write_text(json.dumps(dict(recorded.identity, data_origin='synthetic_fixture')))
                # An actual owned PTY, with a synthetic process instead of Codex/model.
                recorded.terminal = Terminal(root, root, command_prefix=(sys.executable, '-c',
                    "import sys; print('SYNTHETIC REPLAY TERMINAL', flush=True); sys.stdin.read()"))
                await recorded.terminal.start(f'unix://{recorded.session.socket_path}', thread_id)
                await asyncio.wait_for(recorded.terminal.changed.wait(), 2)
                session_id = recorded.terminal.session_id
            finally:
                await recorded.close()
            terminal_bytes = (root/'terminal.bin').read_bytes()
            self.assertIn(b'SYNTHETIC REPLAY TERMINAL', terminal_bytes)

            replay_robot = FixtureRobot(Path(temp)/'replay-robot'); replay_robot.close = lambda: None
            replay_robot.start = lambda: self.fail('replay must not start a robot worker')
            replay = Workbench(root, robot=replay_robot, replay=True)
            replay.session.start = lambda *args: self.fail('replay must not start Codex')
            client = TestClient(TestServer(replay.app()))
            try:
                await replay.start('must not initiate a turn')
                await client.start_server()
                run = await (await client.get('/v1/run')).json()
                self.assertEqual(run['status'], 'replay')
                self.assertEqual(run['thread_id'], thread_id)
                self.assertFalse(run['accepting_actions'])
                self.assertTrue(replay.terminal._closed)
                self.assertEqual(run['terminal']['thread_id'], thread_id)
                self.assertEqual(run['terminal']['session_id'], session_id)
                self.assertIsNone(run['terminal']['pid'])
                self.assertEqual(await (await client.get('/v1/observations/latest')).json(), observation)
                self.assertEqual(await (await client.get('/v1/items/'+item['id'])).json(), item)
                response = await client.get('/v1/jobs/'+program_id)
                self.assertEqual(response.status, 200)
                self.assertEqual(await response.json(), program)
                output = await client.get('/v1/artifacts/'+stdout_ref.removeprefix('artifact:'))
                self.assertEqual(await output.read(), b'synthetic program output')
                ranged=await client.get('/v1/artifacts/'+stdout_ref.removeprefix('artifact:'),headers={'Range':'bytes=0-8'})
                self.assertEqual(ranged.status,206)
                self.assertEqual(await ranged.read(),b'synthetic')
                self.assertEqual(ranged.headers['Content-Range'],'bytes 0-8/24')
                head=await client.head('/v1/artifacts/'+stdout_ref.removeprefix('artifact:'))
                self.assertEqual(int(head.headers['Content-Length']),24)
                self.assertEqual(await head.read(),b'')
                async with client.ws_connect('/v1/terminal') as socket:
                    ready = await socket.receive_json(timeout=2)
                    self.assertEqual(ready['status']['thread_id'], thread_id)
                    self.assertEqual(ready['status']['session_id'], session_id)
                    output = await socket.receive_json(timeout=2)
                    self.assertEqual(base64.b64decode(output['data']), terminal_bytes)
                    self.assertEqual((await socket.receive_json(timeout=2))['type'], 'status')
                    await socket.send_json(dict(type='scroll', lines=-2))
                    self.assertEqual(
                        await socket.receive_json(timeout=2),
                        dict(type='error', message='replay_is_read_only'),
                    )
                for command in ['claim_input', 'prompt', 'reset', 'stop']:
                    response = await client.post('/v1/control', json=dict(request_id='replay-'+command,
                        command=command, client_id='observer', prompt='must not execute'))
                    self.assertEqual(response.status, 400)
                    self.assertEqual((await response.json())['error'], 'replay_is_read_only')
                for path in ['/v1/observer/evaluate', '/v1/observer/record', '/v1/observer/video',
                             '/v1/jobs/'+program_id+'/cancel', '/v1/scene/update']:
                    response = await client.post(path, json={})
                    self.assertEqual(response.status, 400, path)
                response = await client.get('/v1/observer/snapshot')
                self.assertEqual(response.status, 400)
                self.assertEqual(replay_robot.executions, 0)
                self.assertIsNone(replay.session.process)
            finally:
                await client.close(); await replay.close()

    async def test_writer_journal_scene_jobs_and_global_cursor(self):
        with tempfile.TemporaryDirectory() as temp:
            robot=FixtureRobot(temp);robot.close=lambda:None
            workbench=Workbench(Path(temp)/'run',robot=robot, observation_profile="rgb_proprio_v1", motion_review=False)
            workbench.identity.update(IDENTITY)
            workbench.harness.start()
            workbench.starting=False;workbench.status='running'
            workbench.session.thread_id='thread';workbench.session.state='completed'
            turns=[]
            def prompt(value): turns.append(value);return 'turn'
            workbench.session.run_turn=prompt
            workbench.scene_service=SceneService(workbench.root/'twin-service')
            await workbench.scene_service.start()
            workbench.harness.scene=workbench.scene_service.scene
            for obs in recorded_plane(workbench.store): workbench.scene_service.import_bundle(workbench.store,obs)
            workbench.scene_forwarder=asyncio.create_task(workbench.forward_scene_events())
            client=TestClient(TestServer(workbench.app()))
            try:
                await client.start_server();origin=str(client.make_url('')).rstrip('/')
                async def control(body):
                    response=await client.post('/v1/control',json=body,headers={'Origin':origin})
                    return response.status,await response.json()
                status,owner=await control(dict(request_id='claim',command='claim_input',client_id='one'))
                self.assertEqual(status,200)
                status,_=await control(dict(request_id='claim-two',command='claim_input',client_id='two'))
                self.assertEqual(status,409)
                status,_=await control(dict(request_id='steal-same-client',command='claim_input',client_id='one'))
                self.assertEqual(status,409)
                request=dict(request_id='prompt-one',command='prompt',prompt='one',client_id='one',writer_token=owner['writer_token'])
                self.assertEqual((await control(request))[0],200)
                self.assertEqual((await control(request))[0],200)
                self.assertEqual(turns,['one'])
                unknown=dict(request_id='lost-response',command='prompt',prompt='do not resend',client_id='one',writer_token=owner['writer_token'])
                workbench.store.save('control_request','lost-response',unknown)
                workbench.store.set_state('control_response','lost-response',dict(request_id='lost-response',status='unknown'))
                self.assertEqual((await control(unknown))[0],409)
                self.assertEqual(turns,['one'])
                body=dict(request_id='build',static_scene_assumption=True,roi=[100,80,400,320],base_scene_version=0,observation_ids=['obs-0','obs-1'],camera_id='head',entity_id='table',primitive='plane')
                response=await client.post('/v1/scene/update',json=body,headers={'Origin':origin})
                self.assertEqual(response.status,200,await response.text())
                job=await response.json(); identifier=job['job_id']
                deadline=asyncio.get_running_loop().time()+10
                while job['status'] not in ('succeeded','failed','cancelled','expired') and asyncio.get_running_loop().time()<deadline:
                    await asyncio.sleep(.05); job=await (await client.get('/v1/jobs/'+identifier)).json()
                self.assertEqual(job['status'],'succeeded',job)
                head=await (await client.get('/v1/scene/head')).json()
                self.assertEqual(head['scene_version'],1)
                version=await (await client.get('/v1/scene/versions/1')).json()
                self.assertEqual(version['scene_version'],1)
                self.assertIn('obs-1',version['observation_ids'])
                response=await client.post('/v1/scene/snapshot',json=dict(request_id='snapshot',scene_version=1,anchor_observation_id='obs-1'))
                snapshot=await response.json();self.assertIn('snapshot_id',snapshot)
                await asyncio.sleep(.1)
                events=workbench.store.events(limit=1000)
                self.assertEqual([e['server_seq'] for e in events],list(range(1,len(events)+1)))
                self.assertTrue(any(e['type']=='job.completed' and e['source']=='scene_api' for e in events))
                source=json.loads(workbench.scene_service.db.execute("SELECT body FROM events LIMIT 1").fetchone()[0])
                before=workbench.store.event_cursor();workbench.store.ingest_event(source)
                self.assertEqual(before,workbench.store.event_cursor())
                foreign=await client.post('/v1/scene/update',json=dict(body,request_id='foreign',env_generation=99))
                self.assertEqual(foreign.status,409)
                sse=await client.get('/v1/events?after=99999')
                self.assertEqual((await sse.content.readline()).strip(),b'event: gap')
                sse.close()
            finally:
                await client.close();await workbench.close()

    async def test_observer_sampling_and_prompt_are_serialized(self):
        import threading
        with tempfile.TemporaryDirectory() as temp:
            robot=FixtureRobot(temp);robot.close=lambda:None
            workbench=Workbench(Path(temp)/'run',robot=robot, observation_profile="rgb_proprio_v1", motion_review=False)
            workbench.harness.start();workbench.starting=False;workbench.status='running'
            workbench.session.state='completed';workbench.session.thread_id='thread'
            entered,release=threading.Event(),threading.Event();turns=[]
            def evaluate(final_check=True):
                entered.set();release.wait(3);return {'task_success':False}
            robot.evaluate=evaluate
            workbench.session.run_turn=lambda prompt:turns.append(prompt) or 'turn'
            client=TestClient(TestServer(workbench.app()))
            try:
                await client.start_server()
                score=asyncio.create_task(client.post('/v1/observer/evaluate',json={}))
                self.assertTrue(await asyncio.to_thread(entered.wait,2))
                prompt=asyncio.create_task(client.post('/v1/control',json=dict(request_id='prompt',command='prompt',prompt='after score')))
                await asyncio.sleep(.05);self.assertEqual(turns,[])
                release.set()
                response=await score;self.assertEqual(response.status,200)
                result=await response.json();self.assertEqual(result['env_generation'],1)
                response=await prompt;self.assertEqual(response.status,200)
                self.assertEqual(turns,['after score'])
                event=next(e for e in workbench.store.events() if e['type']=='observer.evaluated')
                self.assertEqual(event['env_generation'],result['env_generation'])
            finally:release.set();await client.close();await workbench.close()


# --- test_gateway_hangup.py ---
REPOSITORY = REPO_ROOT
TIMEOUT_S = 60

GATEWAY = r'''
import asyncio, sys, threading, time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from aiohttp import web
from ni_rua.apps.gateway.server import serve
from ni_rua.core.storage.store import Store
from support.fake_agent import FakeAgentSession

root = Path(sys.argv[1])
identity = dict(run_id="fixture", episode_id="episode", env_instance_id="env", env_generation=1)
store = Store(root, identity)
session = FakeAgentSession()
session.thread_id, session.state = "thread", "idle"

def close():
    store.close()
    return {"steps": {}, "errors": [], "robot": {"native_close_acknowledged": True}}

workbench = SimpleNamespace(
    store=store, identity=identity, session=session, status="running",
    harness=SimpleNamespace(action_count=0, active_action=None, episode_done=threading.Event(), episode_terminal=None,
                            native_action_count=0, started_at=time.monotonic()),
    app=web.Application, snapshot=lambda: {"status": "running"}, start=AsyncMock(), stop=AsyncMock(),
    close=AsyncMock(side_effect=close), current_unknown_actions=lambda: False, control_lock=asyncio.Lock(),
    has_active_programs=lambda: False, observer_evaluate=Mock(return_value={}), observer_video=Mock(return_value={}))
args = SimpleNamespace(run_dir=str(root), resume=False, replay=False, price_card=None, category="diagnostic",
                       task="fixture", title="fixture", tag=[], gpu=0, max_actions=1, observation_profile="rgb_proprio_v1",
                       seed=0, layout_id=0, port=0, prompt="fixture", task_notes=None, task_notes_source=None,
                       wall_budget_s=None, exit_after_turn=True, benchmark=False)
with patch("ni_rua.apps.gateway.server.Workbench", return_value=workbench):
    asyncio.run(serve(args))
'''


def launched_from_a_terminal(sighup=signal.SIG_DFL):
    """``preexec_fn`` giving the child the signal dispositions of a gateway started from an interactive shell (SIGINT
    default, SIGHUP ``sighup``). Ignored dispositions are inherited across fork and exec: a test run started under
    ``nohup`` (SIGHUP ignored) or as a background job of a non-interactive shell (SIGINT ignored) would otherwise hand
    them to the gateway, which then keeps the ignored SIGHUP as designed and never shuts down on it."""
    def preexec():
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGHUP, sighup)
    return preexec


class GatewayHangupTests(unittest.TestCase):
    def start(self, sighup=signal.SIG_DFL):
        temporary_dir = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(temporary_dir) / "run"
        environment = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPOSITORY), str(REPOSITORY / "tests")]))
        self.process = subprocess.Popen([sys.executable, "-u", "-c", GATEWAY, str(self.root)], cwd=REPOSITORY,
                                        env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, preexec_fn=launched_from_a_terminal(sighup))
        self.addCleanup(self.reap)
        # The banner is printed after the signal handlers are installed and the HTTP site is listening.
        self.assertTrue(self.process.stdout.readline().startswith("Workbench http://"), "gateway did not start")

    def reap(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        for stream in (self.process.stdout, self.process.stderr):
            if stream and not stream.closed:
                stream.close()

    def assert_finalised(self):
        try:
            code = self.process.wait(TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self.fail("gateway did not shut down after the signal")
        summary = json.loads((self.root / "summary.json").read_text())
        runtime = json.loads((self.root / "runtime.json").read_text())
        self.assertEqual((summary["status"], summary["cleanup_status"]), ("stopped", "completed"))
        self.assertEqual(runtime["status"], "stopped")
        self.assertEqual(code, 0)

    def test_sighup_runs_the_graceful_shutdown_path(self):
        self.start()
        self.process.send_signal(signal.SIGHUP)
        self.assert_finalised()

    def test_sighup_with_stdout_pipe_closed_first_still_finalises_the_archive(self):
        self.start()
        self.process.stdout.close()      # `tee` died with the terminal: every later print raises BrokenPipeError
        self.process.send_signal(signal.SIGHUP)
        self.assert_finalised()

    def test_sigterm_still_finalises(self):
        self.start()
        self.process.send_signal(signal.SIGTERM)
        self.assert_finalised()

    def test_sighup_ignored_at_launch_stays_ignored(self):
        # nohup / detached launchers set SIGHUP to SIG_IGN: the gateway must survive its terminal then.
        self.start(sighup=signal.SIG_IGN)
        self.process.send_signal(signal.SIGHUP)
        with self.assertRaises(subprocess.TimeoutExpired):
            self.process.wait(1.5)
        self.process.send_signal(signal.SIGTERM)
        self.assert_finalised()


class ConsoleLineTests(unittest.TestCase):
    def test_dead_stdout_never_raises_and_is_redirected_to_devnull(self):
        with patch("builtins.print", side_effect=BrokenPipeError()), patch.object(server.os, "dup2") as dup2:
            server._say("line")
        self.assertEqual([call.args[1] for call in dup2.call_args_list], [1, 2])

    def test_shutdown_signals_include_sighup_unless_ignored(self):
        with patch.object(server.signal, "getsignal", return_value=signal.SIG_DFL):
            self.assertIn(signal.SIGHUP, server.shutdown_signals())
        with patch.object(server.signal, "getsignal", return_value=signal.SIG_IGN):
            self.assertEqual(server.shutdown_signals(), [signal.SIGINT, signal.SIGTERM])


# --- test_gateway_modular.py ---
GATEWAY_DIR = REPO_ROOT / "ni_rua" / "apps" / "gateway"


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            mod = "." * node.level + (node.module or "")
            names.append(mod)
            names.extend(f"{mod}.{alias.name}" for alias in node.names)
    return names


class GatewayModularArchitectureTests(unittest.TestCase):
    def test_gateway_core_modules_have_zero_so101_imports_and_no_hardcoded_checks(self):
        for rel in ("workbench.py", "routes.py", "server.py", "supervisor.py", "extensions/calibration.py"):
            path = GATEWAY_DIR / rel
            imports = _imported_modules(path)
            so101_imports = [name for name in imports if "so101" in name.lower()]
            self.assertEqual(so101_imports, [], f"unexpected so101 import in {rel}: {so101_imports}")
        for rel in ("workbench.py", "routes.py", "server.py", "supervisor.py"):
            source = (GATEWAY_DIR / rel).read_text(encoding="utf-8")
            self.assertNotIn('== "so101"', source, rel)
            self.assertNotIn("== 'so101'", source, rel)
            self.assertNotIn('== "so101_real"', source, rel)
            self.assertNotIn("== 'so101_real'", source, rel)

    def test_workbench_uses_explicit_composition_not_mixin_inheritance(self):
        self.assertEqual(Workbench.__mro__, (Workbench, object))
        with tempfile.TemporaryDirectory() as tmp:
            wb = Workbench(Path(tmp) / "run", robot=SimpleNamespace(close=lambda: None))
            try:
                self.assertGreaterEqual(len(wb.extensions), 4)
                for ext in wb.extensions:
                    self.assertIsInstance(ext, CapabilityRouter)
            finally:
                import asyncio
                asyncio.run(wb.close())

    def test_declarative_cli_preserves_flags_and_config_ast_contract(self):
        parser = gateway_cli.build_parser()
        ns = parser.parse_args(
            [
                "--config",
                "configs/ni-rua.json",
                "--no-local-config",
                "--agent-backend",
                "external",
                "--vggt-support-z-m",
                "profile",
                "--no-motion-review",
            ]
        )
        self.assertEqual(ns.config, "configs/ni-rua.json")
        self.assertTrue(ns.no_local_config)
        self.assertEqual(ns.agent_backend, "external")
        self.assertEqual(ns.vggt_support_z_m, "profile")
        self.assertFalse(ns.motion_review)

    def test_so101_gateway_config_uses_portable_relative_paths(self):
        raw = json.loads((REPO_ROOT / "configs" / "so101_gateway.json").read_text(encoding="utf-8"))
        self.assertNotIn("/Users/", json.dumps(raw))
        self.assertEqual(raw["backend"]["deployment"], "so101.local.json")
        loaded = Config.load(REPO_ROOT / "configs" / "so101_gateway.json", ignore_local=True)
        self.assertEqual(
            Path(loaded.backend.deployment).resolve(),
            (REPO_ROOT / "configs" / "so101.local.json").resolve(),
        )

    def test_checkout_layout_preserves_provenance_reexports(self):
        self.assertIs(provenance.CHECKOUT_LAYOUTS, checkout_layout.CHECKOUT_LAYOUTS)
        self.assertIs(provenance.checkout_layout, checkout_layout.checkout_layout)
        self.assertIs(provenance.checkout_environment, checkout_layout.checkout_environment)
        self.assertTrue(checkout_layout.is_queue_process_command(["-m", "ni_rua.apps.evaluation.queue"]))
        self.assertTrue(checkout_layout.is_queue_process_command(["/envs/ni-rua/bin/ni-rua-queue", "--manifest", "m.json"]))
        self.assertTrue(checkout_layout.is_queue_process_command(["scripts/benchmark_queue.py", "--manifest", "m.json"]))
        self.assertFalse(checkout_layout.is_queue_process_command(["-m", "other.module"]))

    def test_run_tests_pid_alive_works_without_proc_filesystem(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("ni_rua_run_tests", REPO_ROOT / "scripts" / "run_tests.py")
        runner = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules["ni_rua_run_tests"] = runner
        try:
            spec.loader.exec_module(runner)
        finally:
            sys.modules.pop("ni_rua_run_tests", None)

        real_is_dir = Path.is_dir

        def fake_is_dir(self_path: Path) -> bool:
            if str(self_path) == "/proc":
                return False
            return real_is_dir(self_path)

        with patch.object(Path, "is_dir", fake_is_dir):
            self.assertTrue(runner._pid_alive(os.getpid()))
            self.assertFalse(runner._pid_alive(99999999))


class CapabilityRouterMountingTests(unittest.IsolatedAsyncioTestCase):
    async def test_capability_routers_mount_by_backend_protocol(self):
        class SimulatedEvaluatorRobot:
            def evaluate(self, final_check=True):
                return {"success": True}

            def observer_snapshot(self):
                return {"ok": True}

            def close(self):
                return None

        class RealTimeCalibratableRobot:
            capabilities = {"observer_video": True}

            def embodiment_profile(self) -> EmbodimentProfile:
                return replace(
                    ARX_X5_PROFILE,
                    robot=replace(ARX_X5_PROFILE.robot, robot_model_id="generic-real-arm"),
                    clock=ClockDomain.REAL_TIME.value,
                )

            def operator(self, command, progress_callback=None, **arguments):
                return {"status": "ok", "command": command}

            def joint_limits_rad(self):
                return {"shoulder_pan": [-1.5, 1.5]}

            def calibrate_timeout_s(self):
                return 10.0

            def calibration_step_timeout_s(self):
                return 5.0

            def reload_deployment(self):
                return {}

            def record(self, enabled=True):
                return {"enabled": enabled}

            def save_video(self):
                return {"audience": "observer_only", "videos": [], "segments": [], "cameras": []}

            def close(self):
                return None

        self.assertIsInstance(SimulatedEvaluatorRobot(), EvaluatorBackend)
        real_robot = RealTimeCalibratableRobot()
        self.assertIsInstance(real_robot, OperatorBackend)
        self.assertIsInstance(real_robot, CalibratableBackend)
        self.assertIsInstance(real_robot, VideoBackend)
        self.assertEqual(real_robot.embodiment_profile().clock, ClockDomain.REAL_TIME)

        with tempfile.TemporaryDirectory() as tmp:
            sim_wb = Workbench(Path(tmp) / "sim", robot=SimulatedEvaluatorRobot())
            try:
                sim_wb.app()
                self.assertNotIn("calibration", sim_wb.mounted_extensions)
                self.assertIn("operator_console", sim_wb.mounted_extensions)
                self.assertIn("agent_settings", sim_wb.mounted_extensions)
                self.assertIn("handoff", sim_wb.mounted_extensions)
            finally:
                await sim_wb.close()

            real_wb = Workbench(Path(tmp) / "real", robot=real_robot)
            real_wb.starting = False
            try:
                app = real_wb.app()
                self.assertIn("calibration", real_wb.mounted_extensions)
                async with TestClient(TestServer(app)) as client:
                    resp = await client.get("/", allow_redirects=False)
                    self.assertEqual(resp.status, 302)
                    self.assertEqual(resp.headers["Location"], "/console")
            finally:
                report = await real_wb.close()
                self.assertEqual(report["steps"]["motion_video"], "completed")

    async def test_supervisor_closes_workbench_when_extension_mount_raises(self):
        class BrokenExtension:
            name = "broken"

            def supports(self, workbench) -> bool:
                return True

            def mount(self, app) -> None:
                raise RuntimeError("extension_mount_boom")

        closed = []

        def make_workbench(root, *pos, **kw):
            wb = Workbench(root, robot=SimpleNamespace(close=lambda: None))
            wb.extensions = (*wb.extensions, BrokenExtension())
            real_close = wb.close

            async def tracked_close():
                report = await real_close()
                closed.append(report)
                return report

            wb.close = tracked_close
            return wb

        with tempfile.TemporaryDirectory() as tmp:
            exp_root = Path(tmp) / "exp"
            run_dir = exp_root / "run-1"
            parsed = vars(
                gateway_cli.build_parser().parse_args(
                    [
                        "--run-dir",
                        str(run_dir),
                        "--experiment-root",
                        str(exp_root),
                        "--port",
                        "0",
                    ]
                )
            )
            cfg = Config().with_overrides(cli_overrides(parsed))
            args = SimpleNamespace(**namespace_values(cfg), resolved_config=None)
            supervisor = RunSupervisor(workbench_factory=make_workbench, say_fn=lambda _msg: None)
            with self.assertRaisesRegex(RuntimeError, "extension_mount_boom"):
                await supervisor.serve_run(args, remote=None)
            self.assertEqual(len(closed), 1)
            self.assertEqual(closed[0]["steps"]["store"], "completed")
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["infrastructure_error"], "extension_mount_boom")


class WebManifestBuildTests(unittest.IsolatedAsyncioTestCase):
    async def test_web_build_emits_manifest_and_library_serves_manifest_assets(self):
        if shutil.which("node") is None or not (REPO_ROOT / "web" / "node_modules").exists():
            self.skipTest("node and web/node_modules required")
        with tempfile.TemporaryDirectory() as tmp:
            dist = Path(tmp) / "dist"
            subprocess.run(
                ["node", str(REPO_ROOT / "web" / "build.mjs"), "--outdir", str(dist)],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest_path = dist / "manifest.json"
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["version"], 1)
            for asset in ("app.js", "app.css", "styles.css", "library.js", "library.css", "console.js", "calibration.js"):
                self.assertIn(asset, manifest["assets"])
                self.assertTrue((dist / asset).is_file())
            self.assertEqual(set(manifest["pages"]), {"index.html", "library.html", "console.html", "calibration.html"})

            exp_root = Path(tmp) / "exp"
            exp_root.mkdir()
            with patch.object(library_service, "STATIC", dist):
                lib = library_service.Library(exp_root)
                self.assertIsInstance(lib.read_router, RunReadRouter)
                async with TestClient(TestServer(lib.app())) as client:
                    self.assertEqual((await client.get("/assets/console.js")).status, 200)
                    self.assertEqual((await client.get("/console.js")).status, 200)
                    self.assertEqual((await client.get("/assets/app.js")).status, 200)
                    self.assertEqual((await client.get("/manifest.json")).status, 404)
                    self.assertEqual((await client.get("/nonexistent.js")).status, 404)


# --- test_gateway_native_failure.py ---
class NativeFailureSummaryTest(unittest.IsolatedAsyncioTestCase):
    async def test_native_failure_is_distinct_from_stop_and_keeps_unknown_motion(self):
        original = '{"error":{"message":"Failed to read request body","type":"invalid_request_error"}}'
        for native_status, unknown, message, limit in [
            ("failed", False, original, False), ("failed", True, original, False),
            ("failed", False, "x" * 5000, False), ("completed", False, None, False),
            ("interrupted", False, None, False), ("failed", False, "model request limit", True),
        ]:
            with self.subTest(status=native_status, unknown=unknown, length=len(message or "")), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                identity = dict(run_id="fixture", episode_id="episode", env_instance_id="env", env_generation=1)
                store = Store(root, identity)
                terminal = threading.Event()
                terminal.set()
                session = SimpleNamespace(
                    thread_id="fixture-thread", state=native_status, turn_done=terminal,
                    turn_result={"status": native_status, "error": {"message": message} if message else None},
                    model="fixture", provider="fixture", price_card=None,
                    model_proxy=SimpleNamespace(limit_notified=limit),
                )
                native_terminal = ({"ended": True, "success": False, "reason": "native_step_limit"}
                                   if native_status == "completed" else None)
                episode_done = threading.Event()
                if native_terminal:
                    episode_done.set()
                    store.set_state("runtime", "native_episode", native_terminal)

                def evaluate():
                    evaluation = {**identity, "result": {"task_success": False, "normalized_score": 0, "valid_episode": True}}
                    ref = store.put_artifact(json.dumps(evaluation).encode(), "evaluation", "application/json", "observer")
                    store.event("observer.evaluated", {"result_ref": ref})
                    return evaluation

                cleanup = {"steps": {"robot": "completed"}, "errors": [],
                    "robot": {"native_close_acknowledged": False,
                              "forced_termination": None if native_status == "interrupted" else "SIGTERM",
                              "process_exit_confirmed": True, "errors": ["synthetic native close timeout"]}}
                if native_status == "completed":
                    # A close ACK cannot conceal a later abnormal worker exit.
                    cleanup["robot"].update(native_close_acknowledged=True, forced_termination=None,
                                            returncode=-11, errors=[])
                def close():
                    store.close()
                    return cleanup

                workbench = SimpleNamespace(
                    store=store, identity=identity, session=session, status="running",
                    harness=SimpleNamespace(action_count=0, active_action=None, episode_done=episode_done,
                        episode_terminal=native_terminal, native_action_count=0, started_at=time.monotonic()),
                    app=web.Application, snapshot=lambda: {"status": "running"},
                    start=AsyncMock(), stop=AsyncMock(), close=AsyncMock(side_effect=close),
                    current_unknown_actions=lambda: unknown, control_lock=asyncio.Lock(),
                    has_active_programs=lambda: False,
                    observer_evaluate=Mock(side_effect=evaluate), observer_video=Mock(return_value={}),
                )
                args = SimpleNamespace(
                    run_dir=str(root), resume=False, replay=False, price_card=None,
                    category="diagnostic", task="fixture", title="fixture", tag=[],
                    gpu=0, max_actions=1, observation_profile="rgb_proprio_v1", seed=0, layout_id=0,
                    port=0, prompt="fixture", task_notes=None, task_notes_source=None,
                    wall_budget_s=1, exit_after_turn=True, benchmark=True,
                )
                with patch("ni_rua.apps.gateway.server.Workbench", return_value=workbench), redirect_stdout(io.StringIO()):
                    await serve(args)
                result = json.loads((root / "summary.json").read_text())
                expected = "unknown" if unknown else "stopped" if native_status == "interrupted" or limit else native_status
                self.assertEqual(result["status"], expected)
                self.assertEqual(result["codex_turn_status"], native_status)
                self.assertEqual(result["cleanup"], cleanup)
                expected_cleanup = ("worker_exit_failed" if native_status == "completed" else
                                    "native_close_unconfirmed" if native_status == "interrupted" else "forced_termination")
                self.assertEqual(result["cleanup_status"], expected_cleanup)
                self.assertEqual(result["cleanup"]["robot"]["native_close_acknowledged"], native_status == "completed")
                if message and not limit:
                    self.assertEqual(result["infrastructure_error"], "codex_turn_failed: " + message[:4096])
                else:
                    self.assertIsNone(result["infrastructure_error"])
                self.assertEqual(workbench.observer_evaluate.call_count, 0 if unknown else 1)
                self.assertEqual(result.get("valid_episode"), None if unknown else True)
                workbench.start.assert_awaited_once()
                if limit:
                    workbench.stop.assert_awaited_once()
                    # A budget stop is scoreable only after confirmed cleanup;
                    # this fixture required SIGTERM without a native close ACK.
                    self.assertEqual(result["primary_outcome"], "unscored_infrastructure")
                    self.assertFalse(result["scored_episode"])
                    self.assertIsNone(result["task_success"])
                    self.assertIsNone(result["normalized_score"])
                    self.assertEqual(result["native_evaluation_snapshot"]["normalized_score"], 0)
                    self.assertTrue(result["model_budget_exhausted"])
                elif native_terminal:
                    workbench.stop.assert_awaited_once()
                    self.assertEqual(result["termination_authority"], "robodojo_environment")
                else:
                    workbench.stop.assert_not_awaited()

    async def test_gateway_http_404_500_control_tombstone_and_ws_stream_error_handling(self):
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(tmp)
        robot = FixtureRobot(root)
        robot.close = lambda: None
        config = Config.from_dict({
            "features": {
                "grasp_memory_dir": str(root / "memory"),
                "observation_profile": "rgb_proprio_v1",
                "motion_review": False,
            }
        })
        workbench = Workbench(root / "run", robot=robot, config=config)
        workbench.starting = False
        workbench.status = "running"
        workbench.session.thread_id = "fixture-thread"
        workbench.session.state = "idle"

        # Inject a terminal stub whose stream() raises mid-stream and resize() exists
        async def failing_stream(_after):
            yield b"hello", 5
            raise RuntimeError("pty_read_failed")

        workbench.terminal = SimpleNamespace(
            session_id="term-1",
            cursor=5,
            status=lambda: {"session_id": "term-1", "cursor": 5, "exit_code": None},
            stream=failing_stream,
            resize=Mock(),
            scroll=Mock(),
            close=lambda: None,
        )

        app = workbench.app()

        async def buggy_route(_request):
            raise TypeError("internal_route_type_bug")

        app.router.add_get("/v1/debug-type-bug", buggy_route)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            # 1) GET /v1/observations/latest before first observation returns 404 (not 400 TypeError)
            latest_resp = await client.get("/v1/observations/latest")
            self.assertEqual(latest_resp.status, 404)

            # 2) Genuine internal TypeError in route returns HTTP 500 (not masked as 400)
            bug_resp = await client.get("/v1/debug-type-bug")
            self.assertEqual(bug_resp.status, 500)

            # 3) Empty Last-Event-ID header on SSE is accepted as cursor 0
            sse_resp = await client.get("/v1/events", headers={"Last-Event-ID": ""})
            self.assertEqual(sse_resp.status, 200)
            sse_resp.close()

            # 4) Unhashable command=[] returns 400 control_requires_command
            origin = str(client.make_url("")).rstrip("/")
            bad_cmd = await client.post(
                "/v1/control",
                headers={"Origin": origin},
                json={"request_id": "bad-cmd-list", "command": []},
            )
            self.assertEqual(bad_cmd.status, 400)
            self.assertEqual((await bad_cmd.json())["error"], "control_requires_command")

            # 5) Unexpected TypeError inside execute_control records 500 status="unknown" (no orphaned tombstone)
            with patch.object(workbench, "execute_control", side_effect=TypeError("unexpected_control_bug")):
                crash_ctrl = await client.post(
                    "/v1/control",
                    headers={"Origin": origin},
                    json={"request_id": "ctrl-type-err", "command": "stop"},
                )
            self.assertEqual(crash_ctrl.status, 500)
            crash_body = await crash_ctrl.json()
            self.assertEqual(crash_body["status"], "unknown")
            self.assertIn("TypeError", crash_body["reason"])
            self.assertEqual(workbench.status, "unknown")
            # Idempotent replay of the same request_id returns the persisted 500 unknown response
            replay_ctrl = await client.post(
                "/v1/control",
                headers={"Origin": origin},
                json={"request_id": "ctrl-type-err", "command": "stop"},
            )
            self.assertEqual(replay_ctrl.status, 500)

            # 6) WebSocket /v1/terminal reports terminal=True error and closes cleanly when output() raises
            async with client.ws_connect("/v1/terminal?after=", headers={"Origin": origin}) as ws:
                ready_msg = await ws.receive_json(timeout=2)
                self.assertEqual(ready_msg["type"], "ready")
                out_msg = await ws.receive_json(timeout=2)
                self.assertEqual(out_msg["type"], "output")
                err_msg = await ws.receive_json(timeout=2)
                self.assertEqual(err_msg["type"], "error")
                self.assertTrue(err_msg.get("terminal"))
            self.assertEqual(len(workbench.store.events_by_type(("terminal.stream_failed",))), 1)
        finally:
            await client.close()
            await workbench.close()



if __name__ == "__main__":
    unittest.main()
